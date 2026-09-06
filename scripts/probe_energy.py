"""
JEPA "energy readout" probe - the veto before building anything.

    python scripts/probe_energy.py \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/last.ckpt

Hypothesis: a pretrained JEPA is a trained ENERGY function - pretraining
minimizes distance(pred(ctx), enc(true_future)) while SIGReg keeps the
encodings of other futures dispersed. If this energy discriminates, the model
can be read WITHOUT a decoder: propose K realistic candidate futures
(block-bootstrap of the series history - real tails, real zeros, real bursts),
encode them, rank them by latent distance to the predicted future. Confidence
interval = quantiles weighted by softmax(-E/T). No training.

The test (one afternoon, CPU): for GIFT instances, generate K bootstrap
candidates + the seasonal naive + THE TRUE FUTURE, and measure the true one's
RANK in the energy ordering.
  * normalized rank ~0.5 = the energy ranks at random -> case closed.
  * systematically low rank = the JEPA latent "knows" things the decoder does
    not read -> the edifice (re-scoring, intervals) is worth building. Also
    the paper's central argument, measured.
Second witness: Spearman(energy, MAE(candidate, truth)) among bootstraps -
the energy must correlate with real proximity, not just spot the true future
through an artifact.

Encoding protocol - replicated from pretrain, not reinvented: pretrain targets
(tiny-full lineage: contextualized_targets=true) are encoded as [ctx||target]
normalized WITH THE CONTEXT STATS then sliced (jepa_tst.forward_pretrain). The
probe does exactly the same for each candidate, robust_scale included when the
checkpoint carries it. Energy is measured on the first n patches (the GIFT
horizon is shorter than the predictor's 256 steps - same truncation as
forecast()).

Two approximations, stated rather than hidden:
  * the ONLINE encoder stands in for the target encoder (the eval loader
    skips the EMA) - at end of pretrain the two are close;
  * on a FINETUNE checkpoint nothing anchors pred(ctx) ~ enc(future) anymore
    (pinball alone trains): a degraded rank pretrain -> finetune measures the
    full-finetune DRIFT, not a pretrain failure. Comparing both checkpoints
    is the probe's most interesting sub-result.

Outputs: console table + JSON under evaluation/probe_energy/.
Modifies nothing: no training code touched, read-only.
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
from timejepa.evaluation import gift                               # noqa: E402
from timejepa.evaluation import ratein as ratein_mod               # noqa: E402
from evaluate_gift import prepare_context                          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("probe_energy")

# Three configs from the E18 tail (must the energy recognize morphologies the
# decoder cannot produce?) + three where the model is good (the energy should
# be clearly discriminant there, otherwise nothing to read).
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
    candidates [Nc, h]    candidate futures, the TRUE one at position 0
    standalone: encode the candidate ALONE (convention of the
                contextualized_targets=false lineages - mix/xres) instead of
                the [ctx||cand] slice. Use the probed checkpoint's pretrain
                convention, otherwise the energy is queried outside its
                training regime.
    w:          rate ratio target grid / context grid handed to the
                predictor's FiLM (xres coherence probe, 2026-09-06): a
                context decimated by k with candidates on the native grid is
                the training pair (k1=k, k2=1), i.e. w = 1/k. Ignored (must
                be 1) on a model without the FiLM.
    Returns (energies_mse, energies_cos) [Nc].
    """
    if w != 1.0 and not hasattr(model.predictor, 'w_film'):
        raise ValueError("w != 1 requires a cross_resolution model (w_film)")
    h = candidates.shape[1]
    n_tgt = (h - model.patching.patch_size) // model.patching.stride + 1

    x_ctx = torch.from_numpy(ctx).reshape(1, -1, 1).to(device)
    cands = torch.from_numpy(candidates).unsqueeze(-1).to(device)   # [Nc, h, 1]

    # G8.4 - same stats (the context's) for the context AND the candidates.
    if model.robust_scaler is not None:
        model.robust_scaler.fit(x_ctx)
        x_ctx = model.robust_scaler.transform(x_ctx)
        cands = model.robust_scaler.transform(cands)

    # RevIN: context stats applied to the candidates - the exact
    # forward_pretrain convention (target normalized with context stats).
    ctx_norm = model.revin(x_ctx, mode='norm') if model.revin is not None else x_ctx
    if model.revin is not None:
        cands_norm = (cands - model.revin.mean) / model.revin.std
    else:
        cands_norm = cands

    # z_pred: the forward_finetune path, truncated to the candidate horizon.
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb,
        num_targets=model.num_target_patches,
        w=(torch.full((1,), float(w), device=device)
           if hasattr(model.predictor, 'w_film') else None),
    )[:, :n_tgt, :]                                                  # [1, n, D]

    # z_cand: per the probed pretrain's target convention.
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
                # rate-aware (w = 1/k) vs blind (w = 1) on the SAME inputs:
                # the aware rank goes in the cos column, the blind rank in the
                # mse column's slot is NOT reused - kept apart below.
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
