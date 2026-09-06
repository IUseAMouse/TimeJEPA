"""
Inference-time refinement of a quantile fan (S6, 2026-09-06).

Adapter around timejepa.training.critic for the GIFT harness: takes the
DENORMALIZED fan the harness already has, brings it into the judge's frame
with the batch context's statistics, descends the fan's center down the
energy (or, in the diagnostic `ceiling` mode, down the TRUE pinball - never
official), stops per instance when the energy no longer drops, and maps the
fan back. Defaults are inert: `mode="off"`, `n_max=0` or `alpha=0` return
the input object untouched.

Energy mode judges the first `hj = min(h', judge.prediction_length)` steps
(the predictor's native span); steps beyond are left untouched and reported
through `judge_cover`. Ceiling mode covers the whole horizon.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..training import critic


@dataclass
class RefineSpec:
    mode: str = "off"          # off | energy | ceiling
    target: str = "center"     # center | fan
    n_max: int = 8
    alpha: float = 0.1         # step, normalized units: with step_norm the largest
                               # displacement of any point per step, so N*alpha
                               # bounds the total move (the "box" shared by the
                               # energy and the ceiling)
    step_norm: bool = True     # per-item L-inf normalization of the gradient
    eps: float = 1e-3          # relative stop: (E_prev - E_new) < eps * max(E_0, 1e-8)
    noise: float = 0.0         # Langevin std per step, 0 = off
    energy: str = "cos"        # cos | mse
    contextualized: bool = False
    seed: int = 0

    @property
    def active(self) -> bool:
        return self.mode != "off" and self.n_max > 0 and self.alpha > 0

    def validate(self):
        if self.mode not in ("off", "energy", "ceiling"):
            raise ValueError(f"unknown refine mode {self.mode!r} (off, energy, ceiling)")
        if self.target not in ("center", "fan"):
            raise ValueError(f"unknown refine target {self.target!r} (center, fan)")
        if self.energy not in ("cos", "mse"):
            raise ValueError(f"unknown refine energy {self.energy!r} (cos, mse)")
        if self.n_max < 0 or self.alpha < 0 or self.eps < 0 or self.noise < 0:
            raise ValueError("refine_steps, refine_alpha, refine_eps and refine_noise "
                             "must be >= 0")
        return self


def normalize_with_context(judge, x_ctx: torch.Tensor, fan: torch.Tensor):
    """Fit the judge's scalers on the raw context and put context and fan in
    its frame (context statistics, the forward_pretrain convention). Always
    refit: after a flipped TTA variant the RevIN state belongs to -x."""
    x = x_ctx
    if getattr(judge, "robust_scaler", None) is not None:
        judge.robust_scaler.fit(x)
        x = judge.robust_scaler.transform(x)
        fan = judge.robust_scaler.transform(fan)
    if judge.revin is not None:
        ctx_norm = judge.revin(x, mode="norm")
        fan = (fan - judge.revin.mean) / judge.revin.std
    else:
        ctx_norm = x
    return ctx_norm, fan


def denormalize_like(judge, fan_norm: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize_with_context (context stats still in the judge)."""
    fan = fan_norm
    if judge.revin is not None:
        fan = fan * judge.revin.std + judge.revin.mean
    if getattr(judge, "robust_scaler", None) is not None:
        fan = judge.robust_scaler.inverse(fan)
    return fan


def pinball_fn(fan_norm: torch.Tensor, target_norm: torch.Tensor,
               levels: Sequence[float]) -> torch.Tensor:
    """[B] mean pinball (GluonTS x2 convention) of a fan [B, h, Q] against a
    target [B, h, 1]; NaN target steps are masked out."""
    q = torch.as_tensor(list(levels), dtype=fan_norm.dtype, device=fan_norm.device)
    valid = torch.isfinite(target_norm)                       # [B, h, 1]
    y = torch.where(valid, target_norm, torch.zeros_like(target_norm))
    diff = y - fan_norm                                        # [B, h, Q]
    loss = 2.0 * torch.maximum(q * diff, (q - 1.0) * diff) * valid.to(fan_norm.dtype)
    denom = valid.to(fan_norm.dtype).sum(dim=(1, 2)).clamp(min=1.0) * fan_norm.shape[-1]
    return loss.sum(dim=(1, 2)) / denom


def _median_idx(levels: Sequence[float]) -> int:
    return min(range(len(levels)), key=lambda j: abs(float(levels[j]) - 0.5))


def refine_fan(judge, ctx_norm: torch.Tensor, fan_norm: torch.Tensor,
               levels: Sequence[float], spec: RefineSpec,
               target_norm: Optional[torch.Tensor] = None,
               w: Optional[torch.Tensor] = None,
               z_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """Refine a normalized fan [B, h', Q]. Returns (fan, stats) - the input
    object itself when the spec is inert."""
    spec.validate()
    if not spec.active:
        return fan_norm, {}
    if spec.mode == "ceiling" and target_norm is None:
        raise ValueError("refine mode 'ceiling' needs the true target")
    B, h, Q = fan_norm.shape
    mid = _median_idx(levels)
    device, dtype = fan_norm.device, fan_norm.dtype
    if spec.mode == "energy":
        hj = min(h, int(judge.prediction_length))
    else:
        hj = h
    gen = torch.Generator(device="cpu").manual_seed(spec.seed) if spec.noise > 0 else None

    with torch.enable_grad():
        if spec.mode == "energy" and z_pred is None:
            z_pred = critic.predict_latent(judge, ctx_norm, w)[0].detach()
        fan = fan_norm.detach().clone()
        shape = (B, hj, 1) if spec.target == "center" else (B, hj, Q)
        cum_delta = torch.zeros(shape, dtype=dtype, device=device)

        def objective(fan_cur: torch.Tensor) -> torch.Tensor:
            head = fan_cur[:, :hj]
            if spec.mode == "energy":
                e = critic.energy_of_fan(judge, ctx_norm, head, z_pred, mode=spec.energy,
                                         contextualized=spec.contextualized,
                                         target=spec.target, median_idx=mid)
                return e if e.ndim == 1 else e.mean(dim=1)
            return pinball_fn(head, target_norm[:, :hj], levels)

        E_prev = objective(fan).detach()
        E0 = E_prev.clone()
        active = torch.isfinite(E_prev)
        steps = torch.zeros(B, dtype=torch.long, device=device)
        trace = [E_prev.clone()]
        for _ in range(spec.n_max):
            if not bool(active.any()):
                break
            delta = torch.zeros(shape, dtype=dtype, device=device, requires_grad=True)
            fan_var = fan.clone()
            fan_var[:, :hj] = fan[:, :hj] + delta
            obj = objective(fan_var)
            grad = torch.autograd.grad((obj * active.to(dtype)).sum(), delta)[0]
            if spec.step_norm:
                grad = grad / critic.unit_linf_scale(grad)
            step = -spec.alpha * grad
            if spec.noise > 0:
                step = step + spec.noise * torch.randn(shape, dtype=dtype, generator=gen).to(device)
            step = step * active.to(dtype).view(B, 1, 1)
            cand = fan.clone()
            cand[:, :hj] = fan[:, :hj] + step
            if spec.target == "fan":
                cand = torch.sort(cand, dim=-1).values
            E_new = objective(cand).detach()
            # rejection guard: a step that raises the objective is undone and
            # the instance frozen; a step that no longer pays freezes it too
            worse = active & (E_new > E_prev)
            improved = active & ~worse
            keep = improved.view(B, 1, 1).to(dtype)
            fan = fan * (1 - keep) + cand * keep
            cum_delta = cum_delta + step * keep
            E_next = torch.where(improved, E_new, E_prev)
            steps = steps + improved.long()
            small = improved & ((E_prev - E_new) < spec.eps * torch.clamp(E0, min=1e-8))
            active = improved & ~small
            E_prev = E_next
            trace.append(E_prev.clone())
    stats = {
        "steps": steps.cpu(),
        "stopped_early": (steps < spec.n_max).cpu(),
        "dE": (E0 - E_prev).cpu(),
        "E0": E0.cpu(),
        "abs_delta": cum_delta.abs().mean(dim=(1, 2)).cpu(),
        "E_trace": torch.stack(trace, dim=0).cpu(),
        "judge_cover": float(hj) / float(h),
    }
    return fan.detach(), stats


def decimated_target(target: np.ndarray, k: int, h_prime: int) -> np.ndarray:
    """Ceiling target on the model's grid: block-decimate the native target
    by k (the RateIN decimation) and pad with NaN to h' - steps without a
    real observation are masked in the pinball."""
    from . import ratein as ratein_mod
    t = np.asarray(target, dtype=np.float32)
    t = ratein_mod.decimate(t, k) if k > 1 else t
    out = np.full(h_prime, np.nan, dtype=np.float32)
    n = min(h_prime, len(t))
    out[:n] = t[:n]
    return out


def refine_out(model, x_ctx: torch.Tensor, out: Dict, spec: Optional[RefineSpec],
               judge=None, w: Optional[torch.Tensor] = None,
               targets: Optional[np.ndarray] = None) -> Tuple[Dict, Optional[Dict]]:
    """Harness adapter: refine `out["quantiles_denorm"]` (3-dim or 4-dim) and
    re-read the median. Returns (out, None) untouched when inert."""
    if spec is None or not spec.active:
        return out, None
    quants = out.get("quantiles_denorm")
    if quants is None:
        return out, None
    J = judge if judge is not None else model
    four = quants.ndim == 4
    fan = quants[..., 0] if four else quants
    device = fan.device
    x = x_ctx.to(device)
    ctx_norm, fan_norm = normalize_with_context(J, x, fan)
    target_norm = None
    if spec.mode == "ceiling":
        if targets is None:
            raise ValueError("refine mode 'ceiling' needs the chunk targets")
        t = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device).unsqueeze(-1)
        with torch.no_grad():
            valid = torch.isfinite(t)
            t_norm = critic.normalize_target_like_context(J, torch.where(valid, t, torch.zeros_like(t)))
            target_norm = torch.where(valid, t_norm, torch.full_like(t_norm, float("nan")))
    levels = list(out.get("quantile_levels") or [0.1 * j for j in range(1, 10)])
    fan_ref, stats = refine_fan(J, ctx_norm, fan_norm, levels, spec,
                                target_norm=target_norm, w=w)
    with torch.no_grad():
        fan_den = denormalize_like(J, fan_ref)
    res = dict(out)
    res["quantiles_denorm"] = fan_den.unsqueeze(-1) if four else fan_den
    mid = _median_idx(levels)
    res["forecast_denorm"] = fan_den[..., mid:mid + 1]
    return res, stats


def summarize_refine(chunks: List[Dict], spec: RefineSpec, judge_kind: str) -> Dict:
    """Config-level witnesses from the per-batch stats."""
    keys = ("steps", "stopped_early", "dE", "E0", "abs_delta")
    cat = {k: torch.cat([torch.as_tensor(c[k]).reshape(-1).float() for c in chunks])
           for k in keys} if chunks else {k: torch.zeros(0) for k in keys}
    n = int(cat["steps"].numel())
    f = lambda t: float(t.mean()) if t.numel() else float("nan")
    return {
        **asdict(spec),
        "judge": judge_kind,
        "judge_cover": (float(np.mean([c["judge_cover"] for c in chunks])) if chunks else None),
        "mean_steps": f(cat["steps"]),
        "frac_stopped_early": f(cat["stopped_early"]),
        "mean_dE": f(cat["dE"]),
        "mean_E0": f(cat["E0"]),
        "mean_abs_delta": f(cat["abs_delta"]),
        "n_refined": n,
        "official": spec.mode != "ceiling",
    }
