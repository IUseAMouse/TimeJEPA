#!/usr/bin/env python
"""
Train the RateIN layer (`timejepa.ratein_layer`) with the forecaster frozen.

A Hydra app, driven by a config that carries the forecaster (the same
`model` block the GIFT harness evaluates, built by `create_model_from_config`
and loaded from `+checkpoint_path=<ckpt>`, or a `model.external` block for a
third-party forecaster) and a `ratein_layer` block for the layer and the run:

    # from the TimeSSM repository, like scripts/eval_ssm.sh:
    scripts/train_ratein_layer.sh <checkpoint> [overrides...]
    # which runs
    python ../TimeJEPA/scripts/train_ratein_layer.py --config-path <repo>/configs \\
        --config-name ratein_layer_e2e +checkpoint_path=<ckpt> [ratein_layer.steps=...]

Two training modes, one layer at inference:

  e2e       end to end: the gradient of the native pinball loss flows through
            the frozen forecaster into the selector. The reference mode (needs
            a forecaster autograd can traverse: TimeSSM, TimeJEPA).
  curve     for forecasters that cannot be traversed (the third-party
            wrappers run under no_grad): the forecaster is called once per
            window and per rate of the grid, the per-window loss curves are
            cached, and the selector learns on the kernel-smoothed curve.

Data: a directory of `.npy [N, L]` arrays (the corpus format) and the
frequency table of the corpus; each row gives one window: the last h points
are the target (h drawn per batch from `ratein_layer.horizons`), a
random-length suffix before them is the native context. Nothing from
GIFT-Eval is read here. The checkpoint written holds the layer's state_dict
and the settings the harness needs (`+ratein=layer +ratein_layer=<file>`).
"""

import math
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.data import frequency as freq_mod  # noqa: E402
from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift, ratein as R  # noqa: E402

LEVELS = list(gift.QUANTILE_LEVELS)
GRID = tuple(float(k) for k in R.K_CANDIDATES) + (1 / 2, 1 / 3, 1 / 4)

DEFAULTS = {
    "mode": "e2e", "corpus_dir": None, "frequency_table": None, "k_prior_table": None,
    "horizons": [256], "min_context": 256, "windows": 12000, "val_fraction": 0.03,
    "batch": 16, "steps": 3000, "lr": 1e-3, "shrink": 1e-3, "sigma": 0.25,
    "n_modes": 48, "hidden": 32, "model_len": RL.MODEL_LEN, "min_blocks": 32,
    "grid_cache": None, "eval_every": 500, "seed": 0, "device": None,
    "out": "checkpoints/ratein_layer/layer.pt", "wandb": False,
}


# -------------------------------------------------------------------- corpus
def load_table(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {stem: ({"freq": e} if isinstance(e, str) else e) for stem, e in raw.items()}


def season_of(stem: str, entry, corpus_dir: Path):
    """Season in steps per row: a scalar from the frequency, the sidecar when
    the table says so, None (unknown) otherwise."""
    if entry is None:
        return None
    if entry.get("season") == freq_mod.PER_ROW or entry.get("freq") == freq_mod.PER_ROW:
        side = corpus_dir / "_season" / f"{stem}.npy"
        return np.load(side, mmap_mode="r") if side.exists() else None
    if entry.get("season") is not None:
        return float(entry["season"])
    return freq_mod.season_length(entry.get("freq"), weekly=bool(entry.get("weekly", False)))


class Windows:
    """Random windows over the corpus: (context, target, season, log_k_prior).
    Files are drawn uniformly, then rows: every family weighs the same."""

    def __init__(self, corpus_dir, table_path, horizons, min_context, n_windows, seed,
                 k_prior_table=None):
        self.horizons = sorted(int(h) for h in horizons)
        self.h = max(self.horizons)                 # the longest horizon bounds the context draw
        self.min_ctx = int(min_context)
        corpus_dir = Path(corpus_dir)
        table = load_table(table_path) if table_path else {}
        files = sorted(p for p in corpus_dir.glob("*.npy") if not p.name.startswith("_"))
        if not files:
            raise FileNotFoundError(f"no .npy under {corpus_dir}")
        self.arrays, self.seasons, self.priors, self.names = [], [], [], []
        for p in files:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] < self.min_ctx + self.h:
                continue
            if table and p.stem not in table:
                raise KeyError(f"{p.stem} missing from the frequency table (an explicit null is allowed)")
            entry = table.get(p.stem)
            self.arrays.append(arr)
            self.names.append(p.stem)
            self.seasons.append(season_of(p.stem, entry, corpus_dir))
            prior = 1.0
            if k_prior_table and entry and entry.get("freq") in k_prior_table:
                prior = float(k_prior_table[entry["freq"]])
            self.priors.append(math.log(prior))
        if not self.arrays:
            raise ValueError("no file long enough for the horizon and the minimum context")
        rng = np.random.default_rng(seed)
        self.index = []
        for _ in range(int(n_windows)):
            f = int(rng.integers(len(self.arrays)))
            r = int(rng.integers(self.arrays[f].shape[0]))
            L = self.arrays[f].shape[1]
            ctx_len = int(rng.integers(self.min_ctx, L - self.h + 1))
            self.index.append((f, r, ctx_len))

    def __len__(self):
        return len(self.index)

    def get(self, i, h=None):
        h = self.h if h is None else int(h)
        f, r, ctx_len = self.index[i]
        row = np.array(self.arrays[f][r], dtype=np.float32)      # a copy: the mmap is read-only
        L = len(row)
        context = row[L - h - ctx_len:L - h]
        target = row[L - h:]
        s = self.seasons[f]
        season = float(s[r]) if isinstance(s, np.ndarray) else s
        return context, target, season, self.priors[f]

    def batch(self, ids, device, h=None):
        items = [self.get(i, h) for i in ids]
        Lmax = max(len(c) for c, _, _, _ in items)
        x = torch.full((len(items), Lmax), float("nan"))
        for b, (c, _, _, _) in enumerate(items):
            x[b, Lmax - len(c):] = torch.from_numpy(c)
        lengths = torch.tensor([len(c) for c, _, _, _ in items])
        y = torch.stack([torch.from_numpy(t) for _, t, _, _ in items])
        season = torch.tensor([float("nan") if s is None else s for _, _, s, _ in items])
        prior = torch.tensor([p for _, _, _, p in items])
        return x.to(device), lengths.to(device), y.to(device), season.to(device), prior.to(device)


# ---------------------------------------------------------------- forecaster
def build_forecaster(cfg: DictConfig, device):
    """The forecaster of the config, frozen, behind
    forecast_fn(y, n, k, season) -> fan [B, n, Q]; whether autograd can
    traverse it; the model; its count of trainable parameters (must be 0).
    Same construction path as the GIFT harness."""
    checkpoint_path = cfg.get("checkpoint_path")
    external = cfg.model.get("external") if cfg.get("model") else None
    if not checkpoint_path and external is None:
        raise ValueError("pass +checkpoint_path=<...> (or a config with model.external)")
    if checkpoint_path and external is not None:
        raise ValueError("model.external and +checkpoint_path are exclusive")
    model = create_model_from_config(cfg)
    if external is None:
        model = load_checkpoint(model, str(checkpoint_path), device)
    model = model.to(device).eval()
    traversable = getattr(model, "kind", "") != "external"
    for p in model.parameters():
        p.requires_grad_(False)
    has_knob = getattr(model, "rate_knob", None) == "delta"
    lo, hi = getattr(model, "delta_range", (0.0, float("inf"))) if has_knob else (None, None)

    def forecast_fn(y, n, k, season):
        w = None
        if has_knob:
            # The rate knob follows the resampling, as the harness does with
            # +freq_delta: w = 24 / (season / k); unknown season -> w = k.
            w_base = torch.where(torch.isfinite(season), freq_mod.BASE_SEASON / season,
                                 torch.ones_like(season))
            w = (k.detach().to(w_base.dtype) * w_base).clamp(lo, hi)
            # One SSM kernel per distinct w: quantized to 1/8 octave so a batch
            # of continuous rates costs a handful of kernels, not one per item.
            w = torch.exp(torch.round(torch.log(w) * 8 / math.log(2)) * math.log(2) / 8).clamp(lo, hi)
            w = w.to(device)
        out = model.forecast(y.to(device).unsqueeze(-1), n, w=w)
        q = out["quantiles_denorm"]
        if q.dim() == 4:
            q = q[..., 0]
        return q.to(y.device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return forecast_fn, traversable, model, n_trainable


# ------------------------------------------------------------------- training
def grid_losses(windows, forecast_fn, layer, device, batch_size, log_every=20):
    """Loss of every window at every rate of the grid (no gradient), at the
    longest horizon."""
    grid = torch.tensor(sorted(GRID), dtype=torch.float64)
    losses = np.zeros((len(windows), len(grid)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            ids = list(range(start, min(start + batch_size, len(windows))))
            x, lengths, y, season, _ = windows.batch(ids, device)
            for g, kval in enumerate(grid.tolist()):
                k = torch.full((len(ids),), kval, dtype=torch.float64, device=device)
                k = torch.exp(layer.bound_log_k(torch.log(k), lengths))
                yc, _ = RL.resample_context(x, k, layer.model_len, lengths)
                n = RL.RateINLayer.model_horizon(windows.h, k)
                fan = forecast_fn(yc, n, k, season)
                native = RL.fan_to_native(fan, k, windows.h)
                losses[ids, g] = RL.pinball_wql(native, y, LEVELS,
                                                denom=RL.context_scale(x, windows.h, lengths)).cpu().numpy()
            if (start // batch_size) % log_every == 0:
                print(f"  grid: {min(start + batch_size, len(windows))}/{len(windows)} windows, "
                      f"{time.time() - t0:.0f}s", flush=True)
    return grid, losses


@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    rl = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg.get("ratein_layer") or {})
    if rl.corpus_dir is None:
        raise ValueError("ratein_layer.corpus_dir is required")
    mode = str(rl.mode)
    if mode not in ("e2e", "curve"):
        raise ValueError(f"ratein_layer.mode must be e2e or curve, got {mode!r}")
    torch.manual_seed(int(rl.seed))
    device = torch.device(rl.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    k_prior_table = yaml.safe_load(Path(rl.k_prior_table).read_text()) if rl.k_prior_table else None
    horizons = [int(h) for h in rl.horizons]
    windows = Windows(rl.corpus_dir, rl.frequency_table, horizons, rl.min_context,
                      rl.windows, rl.seed, k_prior_table)
    n_val = int(len(windows) * float(rl.val_fraction))
    train_ids, val_ids = list(range(n_val, len(windows))), list(range(n_val))
    print(f"{len(windows.arrays)} files, {len(windows)} windows ({len(train_ids)} train, {n_val} val), "
          f"h={horizons}", flush=True)

    layer = RL.RateINLayer(RL.RateSelector(int(rl.n_modes), int(rl.hidden)), model_len=int(rl.model_len),
                           min_blocks=int(rl.min_blocks)).to(device)
    forecast_fn, traversable, model, n_model = build_forecaster(cfg, device)
    n_sel = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"forecaster {getattr(model, 'name', type(model).__name__)}: trainable parameters "
          f"selector {n_sel}, forecaster {n_model}", flush=True)
    assert n_model == 0, "the forecaster must be frozen"
    if mode == "e2e" and not traversable:
        raise ValueError("ratein_layer.mode=e2e needs a forecaster autograd can traverse; use mode=curve")

    run = None
    if bool(rl.wandb):
        import wandb
        wb = cfg.get("wandb") or {}
        run = wandb.init(project=wb.get("project", "TimeJEPA"), entity=wb.get("entity"),
                         name=wb.get("run_name") or f"ratein-layer-{mode}-{getattr(model, 'name', 'model')}",
                         tags=list(wb.get("tags", [])) + ["ratein-layer", mode],
                         config=OmegaConf.to_container(cfg, resolve=True))

    opt = torch.optim.AdamW(layer.parameters(), lr=float(rl.lr), weight_decay=0.0)
    rng = np.random.default_rng(int(rl.seed))
    batch_size = int(rl.batch)

    log_grid = L_all = None
    if mode == "curve":
        cache = Path(rl.grid_cache) if rl.grid_cache else None
        if cache is not None and cache.exists():
            z = np.load(cache)
            grid, losses = torch.tensor(z["grid"], dtype=torch.float64), z["losses"]
            assert losses.shape[0] == len(windows), "grid cache built for another window set"
            print(f"grid losses read from {cache}", flush=True)
        else:
            print("computing the per-rate losses once (the only calls to the forecaster)", flush=True)
            grid, losses = grid_losses(windows, forecast_fn, layer, device, batch_size)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, grid=grid.numpy(), losses=losses)
        log_grid = torch.log(grid).to(device)
        L_all = torch.tensor(losses, device=device)
        k1 = int((grid == 1.0).nonzero()[0])
        print(f"grid: mean loss at k=1 {L_all[:, k1].mean():.4f}, oracle {L_all.min(1).values.mean():.4f}", flush=True)

    def eval_split(ids):
        layer.eval()
        tot_learned, tot_prior, tot_orc, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for bi, start in enumerate(range(0, len(ids), batch_size)):
                b = ids[start:start + batch_size]
                h = windows.horizons[bi % len(windows.horizons)]
                x, lengths, y, season, prior = windows.batch(b, device, h)
                lk = layer.log_k(x, lengths, prior)
                if mode == "curve":
                    Lb = L_all[b]
                    learned = RL.interp_loss_curve(log_grid, Lb, lk)
                    at_prior = RL.interp_loss_curve(log_grid, Lb, layer.bound_log_k(prior, lengths))
                    tot_orc += Lb.min(1).values.sum().item()
                else:
                    def run_at(lkk):
                        k = torch.exp(lkk)
                        yc, _ = RL.resample_context(x, k, layer.model_len, lengths)
                        n_m = RL.RateINLayer.model_horizon(h, k)
                        return RL.pinball_wql(RL.fan_to_native(forecast_fn(yc, n_m, k, season), k, h), y, LEVELS,
                                              denom=RL.context_scale(x, h, lengths))
                    learned, at_prior = run_at(lk), run_at(layer.bound_log_k(prior, lengths))
                tot_learned += learned.sum().item(); tot_prior += at_prior.sum().item(); n += len(b)
        layer.train()
        out = {"learned": tot_learned / n, "prior": tot_prior / n}
        if mode == "curve":
            out["oracle"] = tot_orc / n
        return out

    layer.train()
    t0 = time.time()
    steps = int(rl.steps)
    for step in range(1, steps + 1):
        b = [int(i) for i in rng.choice(train_ids, size=min(batch_size, len(train_ids)), replace=False)]
        h = int(rng.choice(windows.horizons))
        x, lengths, y, season, prior = windows.batch(b, device, h)
        g = layer.selector(x, lengths)
        lk = layer.bound_log_k(prior + g, lengths)
        if mode == "curve":
            loss_items = RL.kernel_loss_curve(log_grid, L_all[b], lk, float(rl.sigma))
        else:
            k = torch.exp(lk)
            yc, _ = RL.resample_context(x, k, layer.model_len, lengths)
            n_m = RL.RateINLayer.model_horizon(h, k)
            fan = forecast_fn(yc, n_m, k, season)
            loss_items = RL.pinball_wql(RL.fan_to_native(fan, k, h), y, LEVELS,
                                        denom=RL.context_scale(x, h, lengths))
        loss = loss_items.mean() + float(rl.shrink) * g.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()
        kk = torch.exp(lk.detach())
        if run is not None:
            run.log({"train/loss": float(loss_items.mean()), "train/g_mean": float(g.mean()),
                     "train/g_std": float(g.std()), "train/k_median": float(kk.median()),
                     "train/k_min": float(kk.min()), "train/k_max": float(kk.max()),
                     "train/frac_k_gt1": float((kk > 1.0001).float().mean()), "train/h": h}, step=step)
        if step % 20 == 0 or step == 1:
            print(f"step {step:5d} loss {loss_items.mean():.4f} | g mean {g.mean():+.3f} std {g.std():.3f} "
                  f"| k: median {kk.median():.2f} min {kk.min():.2f} max {kk.max():.2f} "
                  f"frac>1 {(kk > 1.0001).float().mean():.2f} | h {h} | {time.time() - t0:.0f}s", flush=True)
        if step % int(rl.eval_every) == 0 or step == steps:
            ev = eval_split(val_ids) if val_ids else {}
            msg = " ".join(f"{kname} {v:.4f}" for kname, v in ev.items())
            if "oracle" in ev and ev["prior"] > ev["oracle"]:
                share = (ev["prior"] - ev["learned"]) / (ev["prior"] - ev["oracle"])
                msg += f" | share of the oracle gain captured {share:.2f}"
            print(f"  val: {msg}", flush=True)
            if run is not None:
                run.log({f"val/{kname}": v for kname, v in ev.items()}, step=step)

    out = Path(rl.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": layer.state_dict(),
                "meta": {"n_modes": int(rl.n_modes), "hidden": int(rl.hidden), "model_len": int(rl.model_len),
                         "min_blocks": int(rl.min_blocks), "mode": mode,
                         "forecaster": getattr(model, "name", type(model).__name__),
                         "checkpoint_path": str(cfg.get("checkpoint_path")),
                         "horizons": horizons, "windows": len(windows), "steps": steps,
                         "k_prior_table": k_prior_table,
                         "overrides": list(HydraConfig.get().overrides.task),
                         "config": OmegaConf.to_container(rl, resolve=True)}}, out)
    print(f"saved {out}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
