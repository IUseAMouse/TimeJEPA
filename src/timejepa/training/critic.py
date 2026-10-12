"""
Energy and refinement primitives: how plausible a candidate future is for
the pretrained model, and one gradient step of the candidate down that
energy. Used by the finetune (training through the refinement) and by the
GIFT harness (inference-time refinement).

Everything here lives in the model's normalized frame: robust compression
(if any) then RevIN with the context statistics. Callers hand over tensors
already in that frame.

Energy of a candidate y for context x:
    z_pred = predictor(online_encoder(x), w)         [B, N_full, D]
    z_y    = encoder(patching(y))                    [B, N_tgt,  D]   (standalone)
           = encoder(patching([x || y]))[:, -N_tgt:]                  (contextualized)
    E      = 1 - cos(z_y, z_pred[:, :N_tgt])   or   mse(z_y, z_pred[:, :N_tgt])

The refinement moves the fan's center (median trajectory) by default and
translates every quantile with it, which preserves the head's monotone
parameterization. The `fan` target descends every quantile trajectory
separately and sorts afterwards.

Pass `z_pred.detach()` to keep the critic frozen during the descent, or
`z_pred` itself to train the energy landscape through the descent.
"""

from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

EnergyMode = Literal["cos", "mse"]
RefineTarget = Literal["center", "fan"]


def n_target_patches(model, horizon: int) -> int:
    """Number of patches a horizon of `horizon` steps yields under the model's patching."""
    return (horizon - model.patching.patch_size) // model.patching.stride + 1


def normalize_target_like_context(model, target: torch.Tensor) -> torch.Tensor:
    """Put a raw target in the frame of the context statistics the model just
    fitted: robust compression (if the model has one), then RevIN with the
    context mean/std, never the target's own statistics (future leak)."""
    if getattr(model, "robust_scaler", None) is not None:
        target = model.robust_scaler.transform(target)
    if model.revin is not None:
        target = (target - model.revin.mean) / model.revin.std
    return target


def effective_w(model, w: Optional[torch.Tensor], batch_size: int,
                device: torch.device) -> Optional[torch.Tensor]:
    """The forward_finetune rule: an explicit w=1 when the FiLM exists (its
    bias is trained, skipping it is not identity), None otherwise."""
    if w is not None:
        return w.to(device).reshape(-1)
    if getattr(model.predictor, "w_film", None) is not None:
        return torch.ones(batch_size, device=device)
    return None


def predict_latent(model, ctx_norm: torch.Tensor, w: Optional[torch.Tensor] = None):
    """(z_pred [B, N_full, D], ctx_emb [B, N_ctx, D]) from a normalized context.
    Differentiable in the weights; the caller decides whether to detach."""
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    w_eff = effective_w(model, w, ctx_emb.shape[0], ctx_emb.device)
    out = model.predictor.forward_simple(
        context_embeddings=ctx_emb, num_targets=model.num_target_patches, w=w_eff)
    z_pred = out[0] if isinstance(out, tuple) else out
    return z_pred, ctx_emb


def _pad_to_patch(model, y: torch.Tensor) -> torch.Tensor:
    """Right edge-pad a candidate shorter than one patch, inside the graph so
    the gradient still reaches the real steps."""
    P = model.patching.patch_size
    if y.shape[1] >= P:
        return y
    pad = y[:, -1:, :].expand(-1, P - y.shape[1], -1)
    return torch.cat([y, pad], dim=1)


def encode_candidate(model, ctx_norm: torch.Tensor, y_norm: torch.Tensor,
                     contextualized: bool = False, encoder=None) -> torch.Tensor:
    """Latent of a candidate future [B, N_tgt, D], differentiable in y_norm.

    `encoder` defaults to the online encoder (the descent needs a gradient);
    pass `model.target_encoder` for the joint-loss target. Contextualized
    encoding concatenates [ctx || y] on one grid, which is physically wrong
    when the two live on different grids (xres): refused on a FiLM model.
    """
    if contextualized and getattr(model.predictor, "w_film", None) is not None:
        raise ValueError("contextualized candidate encoding is not defined on a "
                         "cross-resolution model (context and target grids differ)")
    encoder = encoder if encoder is not None else model.online_encoder
    y = _pad_to_patch(model, y_norm)
    n_tgt = n_target_patches(model, y.shape[1])
    if contextualized:
        full = torch.cat([ctx_norm, y], dim=1)
        return encoder(model.patching(full))[:, -n_tgt:, :]
    return encoder(model.patching(y))


def energy_per_item(z_y: torch.Tensor, z_pred: torch.Tensor,
                    mode: EnergyMode = "cos") -> torch.Tensor:
    """[B] energy of each candidate latent against the predicted latent,
    truncated to the candidate's patch count (eval horizons are shorter
    than the native one)."""
    zp = z_pred[:, :z_y.shape[1], :]
    if mode == "mse":
        return (z_y - zp).pow(2).mean(dim=(1, 2))
    if mode == "cos":
        return 1.0 - F.cosine_similarity(z_y.flatten(1), zp.flatten(1), dim=1)
    raise ValueError(f"unknown energy mode {mode!r} (cos, mse)")


def energy_of_fan(model, ctx_norm: torch.Tensor, fan_norm: torch.Tensor,
                  z_pred: torch.Tensor, mode: EnergyMode = "cos",
                  contextualized: bool = False, target: RefineTarget = "center",
                  median_idx: int = 4) -> torch.Tensor:
    """Energy of a quantile fan [B, L, Q]: [B] for the center (median
    trajectory), [B, Q] for every quantile trajectory."""
    if target == "center":
        y = fan_norm[..., median_idx:median_idx + 1]
        return energy_per_item(encode_candidate(model, ctx_norm, y, contextualized),
                               z_pred, mode)
    if target == "fan":
        B, L, Q = fan_norm.shape
        y = fan_norm.permute(0, 2, 1).reshape(B * Q, L, 1)
        ctx_rep = ctx_norm.repeat_interleave(Q, dim=0)
        zp_rep = z_pred.repeat_interleave(Q, dim=0)
        e = energy_per_item(encode_candidate(model, ctx_rep, y, contextualized),
                            zp_rep, mode)
        return e.view(B, Q)
    raise ValueError(f"unknown refine target {target!r} (center, fan)")


def unit_linf_scale(grad: torch.Tensor, tiny: float = 1e-12) -> torch.Tensor:
    """Per-item L-inf norm of a [B, ...] tensor, detached, [B, 1, ..., 1];
    an all-zero item gets 1 (its step stays zero)."""
    m = grad.detach().abs().amax(dim=tuple(range(1, grad.ndim)), keepdim=True)
    return torch.where(m > tiny, m, torch.ones_like(m))


def refine_step(model, ctx_norm: torch.Tensor, fan_norm: torch.Tensor,
                z_pred: torch.Tensor, *, alpha: float, mode: EnergyMode = "cos",
                contextualized: bool = False, target: RefineTarget = "center",
                median_idx: int = 4, create_graph: bool = False,
                noise_sigma: float = 0.0, item_weight: Optional[torch.Tensor] = None,
                max_abs_delta: Optional[float] = None, generator=None,
                step_norm: bool = True):
    """One descent step of the fan down the energy.

    Returns (fan_next, energy_before, delta). The step is taken on a zero
    leaf `delta` added to the descended variable, so that with
    `create_graph=True` the update stays differentiable in the input fan
    (chain to the head's output) and, if `z_pred` is in the graph, in the
    critic's weights (route B). `item_weight` [B] (0/1) excludes items from
    the objective (padded targets); `max_abs_delta` clips the step.

    `step_norm=True` (default) scales the gradient per item to unit L-inf
    norm, so that `alpha` is the largest displacement of any point in one
    step (normalized units), whatever the objective's scale. The denominator
    is detached: the direction stays differentiable, its scale is a constant.
    """
    if target == "center":
        shape = (fan_norm.shape[0], fan_norm.shape[1], 1)
    else:
        shape = tuple(fan_norm.shape)
    delta = torch.zeros(shape, dtype=fan_norm.dtype, device=fan_norm.device,
                        requires_grad=True)
    fan_var = fan_norm + delta                       # broadcast over Q in center mode
    energy = energy_of_fan(model, ctx_norm, fan_var, z_pred, mode=mode,
                           contextualized=contextualized, target=target,
                           median_idx=median_idx)
    objective = energy if item_weight is None else energy * item_weight.view(-1, *([1] * (energy.ndim - 1)))
    grad = torch.autograd.grad(objective.sum(), delta, create_graph=create_graph)[0]
    if step_norm:
        grad = grad / unit_linf_scale(grad)
    step = -alpha * grad
    if noise_sigma > 0:
        step = step + noise_sigma * torch.randn(shape, dtype=step.dtype,
                                                device=step.device, generator=generator)
    if max_abs_delta is not None:
        step = step.clamp(-max_abs_delta, max_abs_delta)
    fan_next = fan_norm + step
    if target == "fan":
        fan_next = torch.sort(fan_next, dim=-1).values
    return fan_next, energy.detach(), step


def refine_loop(model, ctx_norm: torch.Tensor, fan_norm: torch.Tensor,
                z_pred: torch.Tensor, n_steps: int, **step_kwargs) -> Dict[str, List[torch.Tensor]]:
    """N steps of refine_step. fans[0] is the input, fans[i] the fan after
    step i; energies[i] is the energy of fans[i] (the last one measured
    without building a graph); deltas[i] the i-th step."""
    fans, energies, deltas = [fan_norm], [], []
    fan = fan_norm
    for _ in range(int(n_steps)):
        fan, e_before, step = refine_step(model, ctx_norm, fan, z_pred, **step_kwargs)
        fans.append(fan)
        energies.append(e_before)
        deltas.append(step)
    with torch.no_grad():
        energies.append(energy_of_fan(
            model, ctx_norm, fan, z_pred, mode=step_kwargs.get("mode", "cos"),
            contextualized=step_kwargs.get("contextualized", False),
            target=step_kwargs.get("target", "center"),
            median_idx=step_kwargs.get("median_idx", 4)).detach())
    return {"fans": fans, "energies": energies, "deltas": deltas}


# Denoising score matching on the energy (Vincent, 2011). The JEPA loss never
# sees a wrong future nearby, so the energy has no local valley at the truth.
# Here the valley is dug directly: perturb the true target, y_tilde = y + eps,
# and ask the energy's gradient at y_tilde to point back to y. At inference
# the energy refinement descends the field from the head's fan.

PERTURB_KINDS = ("level", "noise", "slope", "forecast")


def perturb_target(y_norm: torch.Tensor, kinds, weights, generator=None,
                   median: Optional[torch.Tensor] = None):
    """Perturbed copies of the true target [B, h, 1], one kind per item drawn
    from `kinds` with probabilities `weights`. Scalars are drawn on CPU with
    `generator` (deterministic under a seeded one) and moved to the device.

      level    : y + c, |c| ~ U[0.05, 0.5], random sign
      noise    : y + white noise, sigma ~ logU[0.02, 0.3]
      slope    : y + ramp 0 -> +-a, a ~ U[0.1, 0.5]
      forecast : the head's median itself (detached), the forecaster's real
                 error distribution; needs `median`

    Returns (y_tilde, kind_idx [B] long). The caller computes the target
    direction as y - y_tilde.
    """
    kinds = list(kinds)
    w = torch.as_tensor([float(x) for x in weights], dtype=torch.float32)
    if len(kinds) != len(w) or len(kinds) == 0:
        raise ValueError("perturb_target: kinds and weights must be non-empty and aligned")
    for k in kinds:
        if k not in PERTURB_KINDS:
            raise ValueError(f"unknown perturbation kind {k!r} (choose from {PERTURB_KINDS})")
    if float(w.sum()) <= 0:
        raise ValueError("perturb_target: weights sum to zero")
    if "forecast" in kinds and w[kinds.index("forecast")] > 0 and median is None:
        raise ValueError("perturbation 'forecast' needs the head's median")
    B, h, _ = y_norm.shape
    device, dtype = y_norm.device, y_norm.dtype
    kind_idx = torch.multinomial(w / w.sum(), B, replacement=True, generator=generator)
    u = torch.rand(B, 4, generator=generator)                     # CPU uniforms
    sign = torch.where(u[:, 0] < 0.5, -1.0, 1.0)
    level = (0.05 + 0.45 * u[:, 1]) * sign                        # |c| in [0.05, 0.5]
    sigma = torch.exp(torch.log(torch.tensor(0.02)) + u[:, 2]
                      * (torch.log(torch.tensor(0.3)) - torch.log(torch.tensor(0.02))))
    slope_a = (0.1 + 0.4 * u[:, 3]) * sign
    white = torch.randn(B, h, 1, generator=generator)
    ramp = torch.linspace(0.0, 1.0, h).view(1, h, 1)
    level_p = level.view(B, 1, 1) * torch.ones(1, h, 1)
    noise_p = sigma.view(B, 1, 1) * white
    slope_p = slope_a.view(B, 1, 1) * ramp
    y_cpu = y_norm.detach().to("cpu", dtype=torch.float32)
    out = y_cpu.clone()
    for i, k in enumerate(kinds):
        sel = kind_idx == i
        if not bool(sel.any()):
            continue
        if k == "level":
            out[sel] = y_cpu[sel] + level_p[sel]
        elif k == "noise":
            out[sel] = y_cpu[sel] + noise_p[sel]
        elif k == "slope":
            out[sel] = y_cpu[sel] + slope_p[sel]
        elif k == "forecast":
            out[sel] = median.detach().to("cpu", dtype=torch.float32)[sel]
    return out.to(device=device, dtype=dtype), kind_idx.to(device)


def score_cos(model, ctx_norm: torch.Tensor, y_norm: torch.Tensor, y_tilde: torch.Tensor,
              z_pred: torch.Tensor, *, mode: EnergyMode = "cos", contextualized: bool = False,
              create_graph: bool = False) -> torch.Tensor:
    """[B] cosine between the energy's descent direction at y_tilde and the
    true direction y - y_tilde. The loss is 1 - cos (caller).

    The gradient is normalized per item with the detached L-inf scale of
    refine_step, so the direction trained is the one the refinement follows.
    `create_graph=True` at train time: the loss then reaches the encoder
    weights and, if `z_pred` is in the graph, the predictor.
    """
    with torch.enable_grad():
        y_var = y_tilde.detach().requires_grad_(True)
        z_y = encode_candidate(model, ctx_norm, y_var, contextualized)
        e = energy_per_item(z_y, z_pred, mode)
        g = torch.autograd.grad(e.sum(), y_var, create_graph=create_graph)[0]
    g = g / unit_linf_scale(g)
    d = (y_norm - y_tilde).detach()
    return F.cosine_similarity(-g.flatten(1), d.flatten(1), dim=1, eps=1e-6)


@torch.no_grad()
def valley_witness(model, ctx_norm: torch.Tensor, y_norm: torch.Tensor, z_pred: torch.Tensor,
                   delta: float = 0.05, mode: EnergyMode = "cos",
                   contextualized: bool = False) -> torch.Tensor:
    """Scalar in [0, 1]: share of items where E(y + delta) > E(y) and
    E(y - delta) > E(y), a local valley at the truth."""
    def e(y):
        return energy_per_item(encode_candidate(model, ctx_norm, y, contextualized), z_pred, mode)
    e0, ep, em = e(y_norm), e(y_norm + delta), e(y_norm - delta)
    return ((ep > e0) & (em > e0)).float().mean()
