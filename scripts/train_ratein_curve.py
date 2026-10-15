#!/usr/bin/env python
"""
RateIN layer, curve mode: the selector trained on the loss curves of a
forecaster that autograd cannot traverse (the third-party wrappers run under
no_grad). The end-to-end mode is the Lightning stack (train_ssm.py with a
`ratein_layer_e2e`-style config); this script is its small sibling for
those models, and reuses the same config, model builder and datamodule.

    python scripts/train_ratein_curve.py --config-path <repo>/configs --config-name <cfg> \\
        [+checkpoint_path=<inner ckpt>] [ratein_curve.max_batches=200]

The config is a `ratein_layer_e2e`-style config: `model.builder` names the
wrapper (timejepa.models.ratein_forecaster), `model.ratein_layer` the layer,
`model.ratein_layer.inner_builder` the frozen inner (e.g.
"timejepa.evaluation.external:build" with a `model.external` block), the
`data` block the corpus. Two phases:
  1. the loss of every window at every rate of the grid (K_CANDIDATES plus
     1/2, 1/3, 1/4), forecaster under no_grad, cached as npz;
  2. the selector trained on the kernel-smoothed curve, validation on the
     last `ratein_curve.val_fraction` of the windows: loss at the prior, at
     the learned rate (linear interpolation) and at the oracle.
The checkpoint is written in the Lightning format (`state_dict` with the
`model.` prefix) so the GIFT harness loads it through load_checkpoint with
the same config (`+checkpoint_path=<out>`).
"""

import math
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.data.datamodule import build_finetune_datamodule  # noqa: E402
from timejepa.data.frequency import BASE_SEASON  # noqa: E402
from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift, ratein as R  # noqa: E402

LEVELS = list(gift.QUANTILE_LEVELS)
GRID = tuple(sorted(tuple(float(k) for k in R.K_CANDIDATES) + (1 / 2, 1 / 3, 1 / 4)))
DEFAULTS = {"max_batches": 0, "val_fraction": 0.05, "steps": 2000, "lr": 1e-3, "sigma": 0.25,
            "grid_cache": None, "eval_every": 200, "seed": 0, "device": "cpu",
            "out": "checkpoints/ratein_layer/curve.pt"}


def tied_w(season: torch.Tensor, delta_range) -> torch.Tensor:
    """The frequency-tied base scale of SSMFinetuneModule._tie_delta at
    evaluation: 24 / season clamped, 1 for an unknown season."""
    known = season > 0
    lo, hi = delta_range if delta_range is not None else (0.0, float("inf"))
    w = (BASE_SEASON / season.clamp_min(1e-6)).clamp(lo, hi)
    return torch.where(known, w, torch.ones_like(w))


def collect(wrapper, loader, horizon, max_batches, device):
    """Phase 1: contexts, targets and the loss at every grid rate."""
    xs, losses, t0 = [], [], time.time()
    grid = torch.tensor(GRID, dtype=torch.float64)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x, y = batch["context"].to(device), batch["target"].to(device)
            season = batch.get("season")
            w_base = tied_w(season.float().to(device), wrapper.delta_range) if season is not None else None
            scale = RL.context_scale(x, horizon)
            row = np.zeros((x.shape[0], len(GRID)), dtype=np.float32)
            lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=device)
            for g, kval in enumerate(GRID):
                k = torch.exp(wrapper.layer.bound_log_k(torch.full((x.shape[0],), math.log(kval), device=device), lengths))
                yc, _ = RL.resample_context(x, k, wrapper.layer.model_len)
                n_m = wrapper.layer.model_horizon(horizon, k)
                out = wrapper.inner.forecast(yc.unsqueeze(-1), n=n_m, w=wrapper._inner_w(w_base, k))
                q = out["quantiles_denorm"]
                q = q[..., 0] if q.dim() == 4 else q
                fan = RL.fan_to_native(q.float(), k, horizon)
                row[:, g] = RL.pinball_wql(fan, y, LEVELS, denom=scale).cpu().numpy()
            xs.append(x.cpu()); losses.append(row)
            if bi % 10 == 0:
                print(f"  curve: batch {bi}, {time.time() - t0:.0f}s", flush=True)
    return torch.cat(xs), torch.tensor(np.concatenate(losses)), grid


@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    rc = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg.get("ratein_curve") or {})
    torch.manual_seed(int(rc.seed))
    device = torch.device(str(rc.device))
    wrapper = create_model_from_config(cfg).to(device)
    if cfg.get("checkpoint_path"):
        wrapper = load_checkpoint(wrapper, str(cfg.checkpoint_path), device)
    assert sum(p.numel() for p in wrapper.parameters() if p.requires_grad) == \
        sum(p.numel() for p in wrapper.layer.parameters()), "only the layer may train"
    horizon = int(cfg.model.prediction_length)

    cache = Path(rc.grid_cache) if rc.grid_cache else None
    if cache is not None and cache.exists():
        z = torch.load(cache, weights_only=False)
        X, L, grid = z["contexts"], z["losses"], z["grid"]
        print(f"curve cache read from {cache}: {len(X)} windows", flush=True)
    else:
        dm = build_finetune_datamodule(cfg)
        dm.prepare_data(); dm.setup("fit")
        X, L, grid = collect(wrapper, dm.train_dataloader(), horizon, int(rc.max_batches), device)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"contexts": X, "losses": L, "grid": grid}, cache)
    log_grid = torch.log(grid).to(device)
    L = L.to(device)
    n_val = int(len(X) * float(rc.val_fraction))
    val, train = list(range(n_val)), list(range(n_val, len(X)))
    k1 = int((grid == 1.0).nonzero()[0])
    print(f"{len(X)} windows ({len(train)} train, {n_val} val): loss at k=1 {L[:, k1].mean():.4f}, "
          f"oracle {L.min(1).values.mean():.4f}", flush=True)

    layer = wrapper.layer
    opt = torch.optim.AdamW(layer.parameters(), lr=float(rc.lr), weight_decay=0.0)
    rng = np.random.default_rng(int(rc.seed))
    batch_size = int(cfg.data.batch_size)

    def evaluate():
        layer.eval()
        with torch.no_grad():
            x = X[val].to(device); lk = layer.log_k(x, None, wrapper.log_k_prior)
            lengths = torch.full((len(val),), x.shape[1], dtype=torch.long, device=device)
            prior = layer.bound_log_k(torch.full_like(lk, wrapper.log_k_prior), lengths)
            out = {"learned": RL.interp_loss_curve(log_grid, L[val], lk).mean().item(),
                   "prior": RL.interp_loss_curve(log_grid, L[val], prior).mean().item(),
                   "oracle": L[val].min(1).values.mean().item()}
        layer.train()
        return out

    layer.train()
    for step in range(1, int(rc.steps) + 1):
        b = [int(i) for i in rng.choice(train, size=min(batch_size, len(train)), replace=False)]
        x = X[b].to(device)
        lk = layer.log_k(x, None, wrapper.log_k_prior)
        loss = RL.kernel_loss_curve(log_grid, L[b], lk, float(rc.sigma)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == 1:
            kk = torch.exp(lk.detach())
            print(f"step {step:5d} loss {loss.item():.4f} | k median {kk.median():.2f} min {kk.min():.2f} "
                  f"max {kk.max():.2f}", flush=True)
        if n_val and (step % int(rc.eval_every) == 0 or step == int(rc.steps)):
            ev = evaluate()
            share = ((ev["prior"] - ev["learned"]) / (ev["prior"] - ev["oracle"])
                     if ev["prior"] > ev["oracle"] else float("nan"))
            print(f"  val: prior {ev['prior']:.4f} learned {ev['learned']:.4f} oracle {ev['oracle']:.4f} "
                  f"| share of the oracle gain captured {share:.2f}", flush=True)

    out = Path(rc.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"model." + k: v.cpu() for k, v in wrapper.state_dict().items()},
                "ratein_curve": OmegaConf.to_container(rc, resolve=True)}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
