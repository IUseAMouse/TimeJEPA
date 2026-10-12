"""
JEPA energy readout probe.

    python scripts/probe_energy.py \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/last.ckpt

A pretrained JEPA is a trained energy function: pretraining minimizes
distance(pred(ctx), enc(true_future)) while SIGReg keeps the encodings of
other futures dispersed. If this energy discriminates, the model can be read
without a decoder: propose K candidate futures (block-bootstrap of the series
history), encode them, rank them by latent distance to the predicted future.

The test: for GIFT instances, generate K bootstrap candidates, the seasonal
naive and the true future, and measure the true one's normalized rank in the
energy ordering (~0.5 = random, systematically low = the latent knows things
the decoder does not read). Second witness: Spearman(energy, MAE(candidate,
truth)) among bootstraps, so the energy must track real proximity.

Encoding protocol, replicated from pretrain: contextualized targets are
encoded as [ctx||target] normalized with the context stats then sliced
(jepa_tst.forward_pretrain), robust_scale included when the checkpoint
carries it. Energy is measured on the first n patches, the same truncation as
forecast().

Two approximations: the online encoder stands in for the target encoder (the
eval loader skips the EMA), and on a finetune checkpoint nothing anchors
pred(ctx) ~ enc(future) anymore, so a degraded rank there measures the
finetune drift, not a pretrain failure.

Outputs: console table + JSON under evaluation/probe_energy/. Read-only.
The reference implementation of the candidate energy is
src/timejepa/training/critic.py; this script keeps its own copy.
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

from hydra import compose, initialize_config_dir                   # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.judge import energy  # noqa: E402
from timejepa.evaluation import gift                               # noqa: E402
from timejepa.evaluation import ratein as ratein_mod               # noqa: E402
from evaluate_gift import prepare_context                          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("probe_energy")

# Three configs where the model is weak (can the energy recognize
# morphologies the decoder cannot produce?) and three where it is good (the
# energy should be clearly discriminant there).
DEFAULT_CONFIGS = [
    "solar/10T/short",
    "bizitobs_l2c/5T/short",
    "electricity/15T/short",
    "electricity/H/short",
    "jena_weather/H/short",
    "sz_taxi/15T/short",
]


def block_bootstrap(history: np.ndarray, h: int, block: int,
                    rng: np.random.Generator) -> np.ndarray:
    """One candidate future: contiguous history blocks, glued up to length h."""
    finite = history[np.isfinite(history)]
    if len(finite) < block + 1:
        finite = np.resize(finite, block + 1)
    out = np.empty(h, dtype=np.float32)
    pos = 0
    while pos < h:
        start = rng.integers(0, len(finite) - block)
        take = min(block, h - pos)
        out[pos:pos + take] = finite[start:start + take]
        pos += take
    return out


@torch.no_grad()
def probe_instance(model, ctx: np.ndarray, candidates: np.ndarray, device,
                   standalone: bool = False, w: float = 1.0):
    """
    ctx        [L]        prepared context (same path as the GIFT eval)
    candidates [Nc, h]    candidate futures, the true one at position 0
    standalone: encode the candidate alone (contextualized_targets=false
                lineages) instead of the [ctx||cand] slice. Use the probed
                checkpoint's pretrain convention.
    w:          rate ratio target grid / context grid handed to the
                predictor's FiLM: a context decimated by k with candidates on
                the native grid is the training pair (k1=k, k2=1), i.e.
                w = 1/k. Must be 1 on a model without the FiLM.
    Returns (energies_mse, energies_cos) [Nc]. The body lives in
    timejepa.judge.energy.
    """
    return energy(model, ctx, candidates, device, standalone=standalone, w=w)


def normalized_rank(energies: np.ndarray) -> float:
    """Rank of the true future (position 0) in the ordering, in [0, 1]; 0 = best."""
    return float((energies < energies[0]).sum()) / (len(energies) - 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="GIFT configs, comma-separated")
    ap.add_argument("--instances", type=int, default=100, help="max per config")
    ap.add_argument("--candidates", type=int, default=32, help="bootstraps per instance")
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--model-config", default="lotsa_tiny_eval")
    ap.add_argument("--standalone-targets", action="store_true",
                    help="convention of the contextualized_targets=false lineages (mix/xres)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-k", type=int, default=0,
                    help="xres COHERENCE probe: decimate the context by k, keep "
                         "the candidates native, and rank the true future under "
                         "w=1/k (rate-aware) AND w=1 (blind) on the same inputs. "
                         "A FiLM that learned w ranks the truth lower when aware. "
                         "Requires a cross_resolution checkpoint.")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    gift_root = Path(args.gift_root)
    results = {}

    for config in args.configs.split(","):
        config = config.strip()
        h = gift.prediction_length(config)
        m = gift.seasonality(config.split("/")[1])
        block = max(8, min(m, h))
        series = gift.load_series(gift_root, config)
        windows = gift.num_windows(config, min(len(s) for s in series))

        ranks_mse, ranks_cos, spears, ranks_blind = [], [], [], []
        for inst in gift.iter_test_instances(series, h, windows):
            if len(ranks_mse) >= args.instances:
                break
            if np.isnan(inst.target).any() or len(inst.context) < 2 * block:
                continue
            hist = inst.context
            if args.rate_k > 1:
                hist = ratein_mod.decimate(
                    inst.context[-(model.input_length * args.rate_k):], args.rate_k)
                if len(hist) < model.patching.patch_size:
                    continue
            ctx = prepare_context(hist, model.input_length,
                                  model.patching.stride, model.patching.patch_size)
            if ctx is None:
                continue

            cands = np.stack(
                [inst.target.astype(np.float32)]
                + [gift.seasonal_naive_forecast(inst.context, h, m).astype(np.float32)]
                + [block_bootstrap(inst.context, h, block, rng)
                   for _ in range(args.candidates)])
            if not np.isfinite(cands).all():
                continue

            if args.rate_k > 1:
                # rate-aware (w = 1/k) vs blind (w = 1) on the same inputs:
                # the aware rank goes in the cos column, the blind rank is
                # kept apart below.
                _, e_aware = probe_instance(model, ctx, cands, device,
                                            standalone=args.standalone_targets,
                                            w=1.0 / args.rate_k)
                _, e_blind = probe_instance(model, ctx, cands, device,
                                            standalone=args.standalone_targets,
                                            w=1.0)
                ranks_cos.append(normalized_rank(e_aware))
                ranks_blind.append(normalized_rank(e_blind))
                e_cos = e_aware
                e_mse = e_aware
            else:
                e_mse, e_cos = probe_instance(model, ctx, cands, device,
                                              standalone=args.standalone_targets)
                ranks_cos.append(normalized_rank(e_cos))
            ranks_mse.append(normalized_rank(e_mse))
            # Does the energy track real proximity? (bootstraps only)
            mae = np.abs(cands[2:] - cands[0]).mean(axis=1)
            spears.append(spearman(e_cos[2:], mae))

        r_mse, r_cos = np.array(ranks_mse), np.array(ranks_cos)
        results[config] = {
            "n_instances": len(r_mse),
            "mean_rank_mse": float(r_mse.mean()),
            "mean_rank_cos": float(r_cos.mean()),
            "median_rank_cos": float(np.median(r_cos)),
            "frac_truth_top20pct_cos": float((r_cos <= 0.2).mean()),
            "spearman_energy_vs_mae": float(np.mean(spears)),
            "rate_k": int(args.rate_k),
            "mean_rank_cos_blind": (float(np.mean(ranks_blind))
                                    if ranks_blind else None),
        }
        r = results[config]
        logger.info(
            f"{config:28s} n={r['n_instances']:3d} | true rank (cos) "
            f"mean {r['mean_rank_cos']:.3f} med {r['median_rank_cos']:.3f} "
            f"top20% {r['frac_truth_top20pct_cos']:.2f} | mse {r['mean_rank_mse']:.3f} "
            f"| rho(E,MAE) {r['spearman_energy_vs_mae']:.3f}   [chance: 0.50 / 0.20 / 0.00]")
        if r["mean_rank_cos_blind"] is not None:
            logger.info(f"{'':28s}   rate-k={r['rate_k']}: true rank AWARE (w=1/k) "
                        f"{r['mean_rank_cos']:.3f} vs BLIND (w=1) "
                        f"{r['mean_rank_cos_blind']:.3f} "
                        f"(delta {r['mean_rank_cos'] - r['mean_rank_cos_blind']:+.3f}; "
                        f"negative = the FiLM learned w)")

    agg = {k: float(np.mean([r[k] for r in results.values()]))
           for k in ("mean_rank_cos", "mean_rank_mse",
                     "frac_truth_top20pct_cos", "spearman_energy_vs_mae")}
    logger.info(f"\nAGGREGATE {len(results)} configs: true rank (cos) "
                f"{agg['mean_rank_cos']:.3f} | top20% {agg['frac_truth_top20pct_cos']:.2f} "
                f"| rho(E,MAE) {agg['spearman_energy_vs_mae']:.3f}")

    out_dir = Path("evaluation/probe_energy")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_standalone" if args.standalone_targets else ""
    out = out_dir / f"{Path(args.checkpoint).stem}{suffix}.json"
    out.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "candidates": args.candidates,
         "per_config": results, "aggregate": agg}, indent=2))
    logger.info(f"JSON: {out}")


if __name__ == "__main__":
    main()
