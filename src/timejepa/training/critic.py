"""
Shared energy and refinement primitives - the S6 critic loop (2026-09-06).

One implementation of "how plausible is this candidate future for the
pretrain" and "one gradient step of the candidate down that energy", used by
the finetune (training through the refinement) and by the GIFT harness
(inference-time refinement). The probes (scripts/probe_energy.py,
scripts/control_ebm_probe.py) predate this module and keep their own copy of
the recipe; this file is the reference.

Everything here lives in the model's NORMALIZED frame: robust compression
(if any) then RevIN with the CONTEXT statistics, which forward_finetune
fits on the context before encoding. Callers hand over tensors already in
that frame; nothing here refits a scaler.

Energy of a candidate y for context x:
    z_pred = predictor(online_encoder(x), w)         [B, N_full, D]
    z_y    = encoder(patching(y))                    [B, N_tgt,  D]   (standalone)
           = encoder(patching([x || y]))[:, -N_tgt:]                  (contextualized)
    E      = 1 - cos(z_y, z_pred[:, :N_tgt])   or   mse(z_y, z_pred[:, :N_tgt])

The refinement moves the fan's CENTER (median trajectory) by default and
translates every quantile with it - the head's monotone parameterization
(median +/- cumulative softplus) is preserved by construction. The `fan`
target descends every quantile trajectory separately and sorts afterwards.

Route A / route B is the caller's choice: pass `z_pred.detach()` to keep the
critic frozen during the descent (first order in the weights), or `z_pred`
itself to train the energy landscape through the descent (second order).
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
    context mean/std - never the target's own statistics (future leak)."""
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
    """Right edge-pad a candidate shorter than one patch, INSIDE the graph so
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
    norm, so that `alpha` IS the largest displacement of any point in one
    step (normalized units) and N steps move nothing farther than N*alpha,
    whatever the objective's scale (a mean pinball's gradient is O(1/h); an
    energy gradient through the encoder has an arbitrary scale). The
    denominator is detached: the direction stays differentiable, its scale
    is a constant.
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
