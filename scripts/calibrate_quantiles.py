"""
G4.2 - UNIFORM conformal quantile calibration (split conformal, CQR-style).

    python scripts/calibrate_quantiles.py \\
        --checkpoint checkpoints/champions/ration45_mase0.8702_crps0.5959.ckpt \\
        --config-name lotsa_tiny_mix_zeroshot --flip

Status (decision 2026-08-25): EXPERIMENTAL ABLATION for the paper - not
necessarily the official reported number.

Measured problem: the fan is too narrow - [q10, q90] coverage of 42-72%
depending on the dataset vs 80% nominal, biased in the SAME direction everywhere.

The fix: one widening factor PER quantile level, around the median -
q'_k = med + gamma_k * (q_k - med) - a single vector of 9 scalars for all 97
GIFT configs ("one checkpoint, zero per-config adaptation" doctrine). MASE
invariant by construction (the median does not move).

Calibration: on the VALIDATION windows of the FINETUNE corpus - never GIFT
(the blind-eval test passes: gamma is frozen before submitting, it is part of
the model, like a classifier's temperature). For each level k, gamma_k is the
empirical quantile of the ratio r = (y-med)/(q_k-med) that restores nominal
coverage (Romano et al., CQR - multiplicative variant, scale-invariant per
window). Aggregation: MEDIAN of the gammas per dataset (one dataset = one
vote, large corpora do not vote more).

Stated limits:
  * coverage is invariant under monotone transforms, but gamma is fitted in
    the datamodule window space and applied in the denormalized GIFT space -
    the gap (arcsinh) is second-order next to an 8-38 point miscalibration;
  * gamma is assumed horizon-independent (calibrated at h=256, applied at
    h in [6, 720]);
  * calibrate WITH --flip if the official procedure is x flip (averaging two
    fans TIGHTENS the spread - the correction must see the procedure it
    corrects).

Output: evaluation/calibration/gamma_<ckpt>[<tags>].json
        {levels, gamma, coverage_before/after per dataset, meta}
Consumed by: evaluate_gift.py  +quantile_gamma=<this json>.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.data.datamodule import MultiDatasetMonashDataModule   # noqa: E402
from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from evaluate_gift import tta_forecast                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_quantiles")

GAMMA_CLAMP = (0.25, 4.0)   # guard: never a degenerate factor


def gamma_for_level(r: np.ndarray, k: float) -> float:
    """gamma restoring P(coverage) = k for a level k, from the ratios
    r = (y-med)/(q_k-med).  High level (den>0): P(r<=g)=k;
    low level (den<0, the inequality flips): P(r>=g)=k."""
    if len(r) < 50:
        return 1.0
    q = k if k > 0.5 else 1.0 - k
    return float(np.clip(np.quantile(r, q), *GAMMA_CLAMP))


def calibrate_dataset(model, get_item, idx, h: int, batch_size: int,
                      flip: bool, device):
    """Collect ONE dataset's ratios and coverages and return
    (levels, {gamma, coverage_before, n_windows}) - accumulators local to the
    function: state can no longer leak from one dataset to the next
    (2026-08-26 bug: init coupled to level discovery, only the first dataset
    got initialized)."""
    levels, ratios, covered = None, None, None
    for i0 in range(0, len(idx), batch_size):
        chunk = idx[i0:i0 + batch_size]
        items = [get_item(j) for j in chunk]
        ctx = torch.stack([torch.as_tensor(it['context'], dtype=torch.float32).reshape(-1)
                           for it in items]).unsqueeze(-1).to(device)  # [B, L, 1]
        y = torch.stack([torch.as_tensor(it['target'], dtype=torch.float32).reshape(-1)
                         for it in items]).cpu().numpy()               # [B, h]
        with torch.no_grad():
            out = tta_forecast(model, ctx, h, flip=flip)
        med = out["forecast_denorm"].squeeze(-1).cpu().numpy()         # [B, h]
        q = out["quantiles_denorm"].cpu().numpy()
        if q.ndim == 4:
            q = q[..., 0]                                              # [B, h, Q]
        if levels is None:
            levels = [float(x) for x in out["quantile_levels"]]
            ratios = [[] for _ in levels]
            covered = [[] for _ in levels]
        den = q - med[..., None]                                       # [B, h, Q]
        num = (y - med).ravel()                        # shared by all levels
        for ki in range(len(levels)):
            d_k = den[..., ki].ravel()
            ok = np.abs(d_k) > 1e-8
            ratios[ki].append((num[ok] / d_k[ok]).astype(np.float32))
            covered[ki].append((y.ravel() <= q[..., ki].ravel()).astype(np.float32))
    stats = {
        "gamma": [gamma_for_level(np.concatenate(ratios[ki]), levels[ki])
                  if levels[ki] != 0.5 else 1.0
                  for ki in range(len(levels))],
        "coverage_before": [float(np.concatenate(covered[ki]).mean())
                            for ki in range(len(levels))],
        "n_windows": int(len(idx)),
    }
    return levels, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="lotsa_tiny_mix_zeroshot",
                    help="FINETUNE config (fixes corpus + geometry)")
    ap.add_argument("--flip", action="store_true",
                    help="calibrate under the official x flip procedure")
    ap.add_argument("--per-dataset", type=int, default=192,
                    help="val windows sampled per dataset")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="evaluation/calibration")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dm = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets_finetune'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        balanced_sampling=cfg.data.balanced_sampling,
        sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio,
        batch_size=args.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        seed=cfg.data.seed,
        num_workers=0,
        use_mmap=bool(cfg.data.get('use_mmap', False)),
    )
    # Outside Lightning, prepare_data (.npy discovery) must be called
    # explicitly - setup('fit') alone sees an empty dataset_files and
    # concludes "every dataset skipped" (measured 2026-08-26).
    dm.prepare_data()
    dm.setup('fit')

    # val_dataset is the ordered concatenation of the datasets: per-dataset
    # sizes give the bounds - regular sampling inside each one.
    names = dm.dataset_names_order
    sizes = dm.val_dataset_sizes
    bounds = np.cumsum([0] + list(sizes))
    h = int(cfg.model.prediction_length)
    levels = None
    per_ds = {}

    for d, name in enumerate(names):
        lo, hi = int(bounds[d]), int(bounds[d + 1])
        if hi - lo < 8:
            logger.info(f"  {name}: {hi - lo} val windows, ignored")
            continue
        idx = np.linspace(lo, hi - 1, min(args.per_dataset, hi - lo)).astype(int)
        levels, stats = calibrate_dataset(
            model, lambda j: dm.val_dataset[int(j)], idx, h,
            args.batch_size, args.flip, device)
        per_ds[name] = stats
        g, cb = stats["gamma"], stats["coverage_before"]
        logger.info(f"  {name:32s} n={len(idx):4d} "
                    f"cov q10/q90 before {cb[0]:.2f}/{cb[-1]:.2f} "
                    f"gamma q10/q90 {g[0]:.2f}/{g[-1]:.2f}")

    if not per_ds:
        raise RuntimeError("no calibratable dataset")

    # One dataset = one vote.
    gamma = [1.0 if levels[ki] == 0.5 else
             float(np.median([per_ds[n]["gamma"][ki] for n in per_ds]))
             for ki in range(len(levels))]

    logger.info("\nLevel    gamma (median across datasets)")
    for lv, g in zip(levels, gamma):
        logger.info(f"  {lv:.1f}    {g:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem + ("_flip" if args.flip else "")
    payload = {
        "levels": levels, "gamma": gamma,
        "checkpoint": args.checkpoint, "config_name": args.config_name,
        "flip": args.flip, "per_dataset": per_ds,
        "doctrine": ("uniform over 97 configs, calibrated on the finetune "
                     "corpus, never GIFT - paper ablation status 2026-08-25"),
    }
    path = out_dir / f"gamma_{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"\nJSON: {path}")
    logger.info("Eval: python scripts/evaluate_gift.py --config-name "
                f"lotsa_tiny_mix_eval '+checkpoint_path=\"{args.checkpoint}\"' "
                + ("+tta_flip=true " if args.flip else "")
                + f"+quantile_gamma='{path}'")


if __name__ == "__main__":
    main()
