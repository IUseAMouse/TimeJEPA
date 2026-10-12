"""The pretrained JEPA read as an energy over (context, future) pairs.

    E(x, y) = 1 - cos(pred(enc(x)), enc(y))

`enc(y)` is the candidate encoded on its own (`contextualized_targets:
false`) or as the tail of [x || y]. `standalone` must match the checkpoint's
convention; a `from_pretrained` model carries it as
`model.judge_standalone_targets`.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def energy(model, context: np.ndarray, candidates: np.ndarray,
           device: Optional[torch.device] = None,
           standalone: Optional[bool] = None,
           w: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """context [L] raw values, candidates [Nc, h] raw values ->
    (energies_mse, energies_cos), each [Nc]; lower is more plausible.

    The context's normalization statistics are applied to the candidates
    (the pretraining convention). `standalone` defaults to the model's
    `judge_standalone_targets` attribute when it has one, else False."""
    if device is None:
        device = next(model.parameters()).device
    if standalone is None:
        standalone = bool(getattr(model, "judge_standalone_targets", False))
    if w != 1.0 and not hasattr(model.predictor, 'w_film'):
        raise ValueError("w != 1 requires a cross_resolution model (w_film)")
    context = np.asarray(context, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    h = candidates.shape[1]
    n_tgt = (h - model.patching.patch_size) // model.patching.stride + 1

    x_ctx = torch.from_numpy(context).reshape(1, -1, 1).to(device)
    cands = torch.from_numpy(candidates).unsqueeze(-1).to(device)   # [Nc, h, 1]

    if model.robust_scaler is not None:
        model.robust_scaler.fit(x_ctx)
        x_ctx = model.robust_scaler.transform(x_ctx)
        cands = model.robust_scaler.transform(cands)

    ctx_norm = model.revin(x_ctx, mode='norm') if model.revin is not None else x_ctx
    if model.revin is not None:
        cands_norm = (cands - model.revin.mean) / model.revin.std
    else:
        cands_norm = cands

    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb,
        num_targets=model.num_target_patches,
        w=(torch.full((1,), float(w), device=device)
           if hasattr(model.predictor, 'w_film') else None),
    )[:, :n_tgt, :]                                                  # [1, n, D]

    if standalone:
        z_cand = model.online_encoder(model.patching(cands_norm))
    else:
        full = torch.cat([ctx_norm.expand(cands_norm.shape[0], -1, -1),
                          cands_norm], dim=1)                        # [Nc, L+h, 1]
        z_cand = model.online_encoder(model.patching(full))[:, -n_tgt:, :]

    diff = z_cand - z_pred                                           # [Nc, n, D]
    e_mse = diff.pow(2).mean(dim=(1, 2))
    cos = torch.nn.functional.cosine_similarity(
        z_cand.flatten(1), z_pred.expand_as(z_cand).flatten(1), dim=1)
    return e_mse.cpu().numpy(), (1.0 - cos).cpu().numpy()
