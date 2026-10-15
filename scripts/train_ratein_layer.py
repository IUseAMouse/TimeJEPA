#!/usr/bin/env python
"""
Train the RateIN layer (`timejepa.ratein_layer`) with the forecaster frozen.

Two training modes, one layer at inference:

  e2e       end to end: the gradient of the native pinball loss flows through
            the frozen forecaster into the selector. The reference mode (needs
            a differentiable model: TimeSSM loaded with `timessm.from_pretrained`).
  curve     for forecasters that cannot be traversed (the third-party
            wrappers run under no_grad): the forecaster is called once per
            window and per rate of the grid, the per-window loss curves are
            cached, and the selector learns on the kernel-smoothed curve.

Data: a directory of `.npy [N, L]` arrays (the corpus format) and the
frequency table of the corpus; each row gives one window: the last
`--horizon` points are the target, a random-length suffix before them is the
native context. Nothing from GIFT-Eval is read here.

    python scripts/train_ratein_layer.py --mode e2e --model timessm \\
        --pretrained ../TimeSSM/export/timessm-2.5m \\
        --corpus-dir data/processed/lotsa_small --frequency-table configs/corpus_v3_frequencies.yaml \\
        --windows 12000 --out checkpoints/ratein_layer/e2e_timessm.pt

The checkpoint holds the layer's state_dict and the settings the harness
needs (`+ratein=layer +ratein_layer=<file>`).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.data import frequency as freq_mod  # noqa: E402
from timejepa.evaluation import gift, ratein as R  # noqa: E402

LEVELS = list(gift.QUANTILE_LEVELS)
GRID = tuple(float(k) for k in R.K_CANDIDATES) + (1 / 2, 1 / 3, 1 / 4)


# -------------------------------------------------------------------- corpus
def load_table(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for stem, entry in raw.items():
        if entry is None:
            out[stem] = None
            continue
        if isinstance(entry, str):
            entry = {"freq": entry}
        out[stem] = entry
    return out


def season_of(stem: str, entry, corpus_dir: Path, n_rows: int):
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
    """Random windows over the corpus: (context, target, season, log_k_prior)."""

    def __init__(self, corpus_dir, table_path, horizon, min_context, n_windows, seed,
                 k_prior_table=None):
        self.h = int(horizon)
        self.min_ctx = int(min_context)
        corpus_dir = Path(corpus_dir)
        table = load_table(table_path) if table_path else {}
        files = sorted(p for p in corpus_dir.glob("*.npy") if not p.name.startswith("_"))
        if not files:
            raise FileNotFoundError(f"no .npy under {corpus_dir}")
        self.arrays, self.seasons, self.priors = [], [], []
        for p in files:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] < self.min_ctx + self.h:
                continue
            if table and p.stem not in table:
                raise KeyError(f"{p.stem} missing from the frequency table (an explicit null is allowed)")
            entry = table.get(p.stem)
            self.arrays.append(arr)
            self.seasons.append(season_of(p.stem, entry, corpus_dir, arr.shape[0]))
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

    def get(self, i):
        f, r, ctx_len = self.index[i]
        row = np.array(self.arrays[f][r], dtype=np.float32)      # a copy: the mmap is read-only
        L = len(row)
        context = row[L - self.h - ctx_len:L - self.h]
        target = row[L - self.h:]
        s = self.seasons[f]
        season = float(s[r]) if isinstance(s, np.ndarray) else s
        return context, target, season, self.priors[f]

    def batch(self, ids, device):
        items = [self.get(i) for i in ids]
        Lmax = max(len(c) for c, _, _, _ in items)
        x = torch.full((len(items), Lmax), float("nan"))
        for b, (c, _, _, _) in enumerate(items):
            x[b, Lmax - len(c):] = torch.from_numpy(c)
        lengths = torch.tensor([len(c) for c, _, _, _ in items])
        y = torch.stack([torch.from_numpy(t) for _, t, _, _ in items])
        season = torch.tensor([float("nan") if s is None else s for _, _, s, _ in items])
        prior = torch.tensor([p for _, _, _, p in items])
        return x.to(device), lengths.to(device), y.to(device), season.to(device), prior.to(device)


# ---------------------------------------------------------------- forecasters
def build_forecaster(args, device):
    """-> (forecast_fn(y, n, k, season) -> fan [B, n, Q], differentiable: bool)."""
    if args.model == "timessm":
        from timessm import from_pretrained
        model = from_pretrained(args.pretrained, device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        lo, hi = model.delta_range

        def forecast_fn(y, n, k, season):
            # The rate knob follows the resampling: w = 24 / (season / k), as the
            # harness does with +freq_delta; unknown season -> w = k (base 1).
            w_base = torch.where(torch.isfinite(season), freq_mod.BASE_SEASON / season,
                                 torch.ones_like(season))
            w = (k.detach().to(w_base.dtype) * w_base).clamp(lo, hi)
            # One SSM kernel per distinct w: quantize to 1/8 octave so a batch of
            # continuous rates costs a handful of kernels, not one per item.
            w = torch.exp(torch.round(torch.log(w) * 8 / math.log(2)) * math.log(2) / 8).clamp(lo, hi)
            out = model.forecast(y.to(device).unsqueeze(-1), n, w=w.to(device))
            q = out["quantiles_denorm"]
            if q.dim() == 4:
                q = q[..., 0]
            return q
        return forecast_fn, True

    if args.model == "external":
        from omegaconf import OmegaConf
        from timejepa.evaluation import external
        cfg = OmegaConf.load(args.external_config)
        model = external.build(cfg)

        def forecast_fn(y, n, k, season):
            with torch.no_grad():
                return model.forecast(y.detach().cpu(), n=n)["quantiles_denorm"].to(y.device)
        return forecast_fn, False
    raise ValueError(args.model)


# ------------------------------------------------------------------- training
def grid_losses(windows, forecast_fn, layer, device, batch_size, log_every=20):
    """Loss of every window at every rate of the grid (no gradient)."""
    grid = torch.tensor(sorted(GRID), dtype=torch.float64)
    losses = np.zeros((len(windows), len(grid)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            ids = list(range(start, min(start + batch_size, len(windows))))
            x, lengths, y, season, _ = windows.batch(ids, device)
            for g, kval in enumerate(grid.tolist()):
                k = torch.full((len(ids),), kval, dtype=torch.float64, device=device)
                # the bound on k from the context length, as the layer applies it
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("e2e", "curve"), required=True)
    ap.add_argument("--model", choices=("timessm", "external"), required=True)
    ap.add_argument("--pretrained", default=None, help="timessm: export directory or HF id")
    ap.add_argument("--external-config", default=None, help="external: a yaml with model.external")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--frequency-table", default=None)
    ap.add_argument("--k-prior-table", default=None, help="yaml: frequency string -> prior k (default 1)")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--min-context", type=int, default=256)
    ap.add_argument("--windows", type=int, default=5000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shrink", type=float, default=1e-3, help="penalty on g^2 (the residual in log k)")
    ap.add_argument("--sigma", type=float, default=0.25, help="curve mode: kernel bandwidth in log k")
    ap.add_argument("--n-modes", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--model-len", type=int, default=RL.MODEL_LEN)
    ap.add_argument("--min-blocks", type=int, default=32)
    ap.add_argument("--grid-cache", default=None, help="curve mode: npz to reuse / write the per-k losses")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    k_prior_table = yaml.safe_load(Path(args.k_prior_table).read_text()) if args.k_prior_table else None
    windows = Windows(args.corpus_dir, args.frequency_table, args.horizon, args.min_context,
                      args.windows, args.seed, k_prior_table)
    n_val = int(len(windows) * args.val_fraction)
    train_ids, val_ids = list(range(n_val, len(windows))), list(range(n_val))
    print(f"{len(windows.arrays)} files, {len(windows)} windows ({len(train_ids)} train, {n_val} val), "
          f"h={args.horizon}", flush=True)

    layer = RL.RateINLayer(RL.RateSelector(args.n_modes, args.hidden), model_len=args.model_len,
                           min_blocks=args.min_blocks).to(device)
    forecast_fn, differentiable = build_forecaster(args, device)
    if args.mode == "e2e" and not differentiable:
        raise ValueError("--mode e2e needs a differentiable forecaster (timessm); use --mode curve")
    opt = torch.optim.AdamW(layer.parameters(), lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)

    grid = losses = None
    if args.mode == "curve":
        cache = Path(args.grid_cache) if args.grid_cache else None
        if cache is not None and cache.exists():
            z = np.load(cache)
            grid, losses = torch.tensor(z["grid"], dtype=torch.float64), z["losses"]
            assert losses.shape[0] == len(windows), "grid cache built for another window set"
            print(f"grid losses read from {cache}", flush=True)
        else:
            print("computing the per-rate losses once (the only calls to the forecaster)", flush=True)
            grid, losses = grid_losses(windows, forecast_fn, layer, device, args.batch)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, grid=grid.numpy(), losses=losses)
        log_grid = torch.log(grid).to(device)
        L_all = torch.tensor(losses, device=device)
        k1 = int((grid == 1.0).nonzero()[0])
        orc = L_all.min(1).values
        print(f"grid: mean loss at k=1 {L_all[:, k1].mean():.4f}, oracle {orc.mean():.4f}", flush=True)

    def eval_split(ids):
        layer.eval()
        tot_learned, tot_prior, tot_orc, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(ids), args.batch):
                b = ids[start:start + args.batch]
                x, lengths, y, season, prior = windows.batch(b, device)
                lk = layer.log_k(x, lengths, prior)
                if args.mode == "curve":
                    Lb = L_all[b]
                    learned = RL.interp_loss_curve(log_grid, Lb, lk)
                    at_prior = RL.interp_loss_curve(log_grid, Lb, layer.bound_log_k(prior, lengths))
                    tot_orc += Lb.min(1).values.sum().item()
                else:
                    def run(lkk):
                        k = torch.exp(lkk)
                        yc, _ = RL.resample_context(x, k, layer.model_len, lengths)
                        n_m = RL.RateINLayer.model_horizon(windows.h, k)
                        return RL.pinball_wql(RL.fan_to_native(forecast_fn(yc, n_m, k, season), k, windows.h), y, LEVELS,
                                              denom=RL.context_scale(x, windows.h, lengths))
                    learned, at_prior = run(lk), run(layer.bound_log_k(prior, lengths))
                tot_learned += learned.sum().item(); tot_prior += at_prior.sum().item(); n += len(b)
        layer.train()
        out = {"learned": tot_learned / n, "prior": tot_prior / n}
        if args.mode == "curve":
            out["oracle"] = tot_orc / n
        return out

    layer.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = [int(i) for i in rng.choice(train_ids, size=min(args.batch, len(train_ids)), replace=False)]
        x, lengths, y, season, prior = windows.batch(b, device)
        g = layer.selector(x, lengths)
        lk = layer.bound_log_k(prior + g, lengths)
        if args.mode == "curve":
            loss_items = RL.kernel_loss_curve(log_grid, L_all[b], lk, args.sigma)
        else:
            k = torch.exp(lk)
            yc, _ = RL.resample_context(x, k, layer.model_len, lengths)
            n_m = RL.RateINLayer.model_horizon(windows.h, k)
            fan = forecast_fn(yc, n_m, k, season)
            loss_items = RL.pinball_wql(RL.fan_to_native(fan, k, windows.h), y, LEVELS,
                                        denom=RL.context_scale(x, windows.h, lengths))
        loss = loss_items.mean() + args.shrink * g.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            kk = torch.exp(lk.detach())
            print(f"step {step:5d} loss {loss_items.mean():.4f} | g mean {g.mean():+.3f} std {g.std():.3f} "
                  f"| k: median {kk.median():.2f} "
                  f"min {kk.min():.2f} max {kk.max():.2f} frac>1 {(kk > 1.0001).float().mean():.2f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_split(val_ids) if val_ids else {}
            msg = " ".join(f"{kname} {v:.4f}" for kname, v in ev.items())
            if "oracle" in ev and ev["prior"] > ev["oracle"]:
                share = (ev["prior"] - ev["learned"]) / (ev["prior"] - ev["oracle"])
                msg += f" | share of the oracle gain captured {share:.2f}"
            print(f"  val: {msg}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": layer.state_dict(),
                "meta": {"n_modes": args.n_modes, "hidden": args.hidden, "model_len": args.model_len,
                         "min_blocks": args.min_blocks, "mode": args.mode, "model": args.model,
                         "horizon": args.horizon, "windows": len(windows), "steps": args.steps,
                         "k_prior_table": k_prior_table}}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
