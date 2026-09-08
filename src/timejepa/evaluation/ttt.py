"""
JEPA test-time adaptation on the lookbacks (2026-09-08).

A pinball-only forecaster has nothing to learn from a context without a
target. A JEPA has: its self-supervised loss exists on the context alone.
Before forecasting a dataset, a few gradient steps of that loss on the
series' own histories adapt the representation to THIS regime - drift
statistics, volatility - from data the model has, not from its forecast
(the S6 / S6-b lesson: the forecast's error is what the context does not
determine; the context itself is information). Causal: only lookbacks, no
target, no test window.

The loss is the finetune-side JEPA term: each lookback is split into a
context part and a "future" part (the last `horizon` steps of the lookback),
z_pred = predictor(online_encoder(context part)), z_y = online_encoder(
future part) under stop-gradient (the target encoder is not shipped in the
finetuned checkpoints), MSE between the two. The adapted weights live in a
deep copy; the caller evaluates with the copy and drops it.

Two parameter sets: 'norm' (LayerNorm / RevIN affine only - the classic
test-time-training choice, robust) and 'all' (encoder + predictor).
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from ..training import critic
from . import refine as refine_mod


def select_params(model, which: str) -> List[torch.nn.Parameter]:
    names = []
    for name, p in model.named_parameters():
        if name.startswith("target_encoder") or name.startswith("decoder"):
            continue
        if which == "all":
            names.append((name, p))
        elif which == "norm":
            if "norm" in name.lower() or "revin" in name.lower() or name.endswith(".bias"):
                names.append((name, p))
        else:
            raise ValueError(f"unknown ttt params {which!r} (norm, all)")
    return [p for _, p in names]


def jepa_context_loss(model, x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Self-supervised loss of one batch of lookbacks [B, L, 1]: the last
    `horizon` steps are the future, the rest the context."""
    ctx, fut = x[:, :-horizon], x[:, -horizon:]
    ctx_norm, fut_norm = refine_mod.normalize_with_context(model, ctx, fut)
    z_pred, _ = critic.predict_latent(model, ctx_norm, None)
    with torch.no_grad():
        z_y = critic.encode_candidate(model, ctx_norm, fut_norm, contextualized=False)
    return (z_pred[:, :z_y.shape[1]] - z_y).pow(2).mean()


def adapt(model, contexts: List[np.ndarray], device, *, steps: int = 16,
          lr: float = 1e-5, params: str = "norm", batch_size: int = 64,
          horizon: Optional[int] = None, seed: int = 0) -> tuple:
    """Return (adapted copy of the model, stats). `contexts` are prepared
    lookbacks (stride-aligned, same length within a batch is NOT required:
    batches are built per length). Lookbacks shorter than 2 * horizon are
    skipped."""
    m = copy.deepcopy(model).to(device)
    m.train(False)                       # no dropout; RevIN fits per batch anyway
    h = int(horizon or model.prediction_length)
    by_len = {}
    for c in contexts:
        c = np.asarray(c, dtype=np.float32)
        if len(c) >= 2 * h + model.patching.patch_size and np.isfinite(c).all():
            by_len.setdefault(len(c), []).append(c)
    batches = []
    for L, items in by_len.items():
        for i in range(0, len(items), batch_size):
            batches.append(torch.from_numpy(np.stack(items[i:i + batch_size])).unsqueeze(-1))
    stats = {"n_contexts": sum(len(v) for v in by_len.values()), "n_batches": len(batches),
             "steps": 0, "loss_first": float("nan"), "loss_last": float("nan")}
    if not batches or steps <= 0:
        return m, stats
    plist = select_params(m, params)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in plist:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(plist, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        x = batches[int(rng.integers(len(batches)))].to(device)
        with torch.enable_grad():
            loss = jepa_context_loss(m, x, h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plist, 1.0)
            opt.step()
        losses.append(float(loss.detach()))
    for p in m.parameters():
        p.requires_grad_(False)
    stats.update({"steps": steps, "loss_first": losses[0], "loss_last": losses[-1],
                  "loss_mean_first_quarter": float(np.mean(losses[:max(1, steps // 4)])),
                  "loss_mean_last_quarter": float(np.mean(losses[-max(1, steps // 4):]))})
    return m, stats
