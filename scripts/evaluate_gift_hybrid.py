"""
G12 on GIFT - raw TTM-R3 vs the "TTM proposes, TimeJEPA judges" hybrid.

    python scripts/evaluate_gift_hybrid.py \\
        --judge-checkpoint checkpoints/timejepa_lotsa_tiny_v3/pretrain_True/<ckpt> \\
        --judge-config lotsa_tiny_v3_eval

STATUS: PAPER experiment (chapter G12), NEVER an official number - a
two-model hybrid is beyond the G11 line (decision 2026-08-24). Designed
2026-08-28 (user question: port the evaluate_energy/TTM duo to GIFT). The
leaderboard harness (evaluate_gift.py) is NOT touched.

Three readouts, three statuses:
  * `ttm`        - raw TTM-R3 on the GIFT windows. Its ABSOLUTE MASE compares
                   to the vendored official TTM-R3-PT CSV (harness
                   cross-validation + the "pipeline" share of their score).
                   Warning: its CRPS is a collapsed point forecast: do NOT cite it.
  * `hybrid_ttm` - pool of bootstrap + SN + drift + jittered TTM paths,
                   judged by OUR pretrain (contextualized encoding, E18c),
                   read as weighted quantiles. THE measure: an external point
                   forecaster becomes probabilistic (fan + coverage) via a
                   latent judge.
  * champion reference: NOT recomputed here - cite the existing full evals
                   (better statistics than our capped windows).

Pairing: BOTH readers see exactly the same instances (regular subsampling,
--instances per config) - internal comparisons are paired; absolute values on
capped windows do not replace a full eval.

TTM rollout for h > 96: autoregressive segment-by-segment reinjection (jitter
on the FIRST segment only - diversity is born at the start, later segments
extend each path deterministically).

Judge choice (to declare in the paper):
  * primary   = the pretrain's FINAL named checkpoint (zero selection);
  * secondary = probe-selected judge (early-peak law) - the probe reads GIFT
    data, so that column is labeled "declared selection".

Outputs: evaluation/gift_hybrid/<judge>/{per_config/*.json, summary.json}
(same format as the official harness, coverage included).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift                                # noqa: E402
from evaluate_gift import prepare_context                           # noqa: E402
from evaluate_energy import (TTMProposer, candidate_energies,       # noqa: E402
                             fan_from_energies)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gift_hybrid")


def ttm_rollout(prop: TTMProposer, ctx: np.ndarray, h: int,
                n_jitter: int, rng) -> np.ndarray:
    """[1+n_jitter, h] - TTM paths, autoregressive in pred_len segments."""
    ctxs, segs, rem = None, [], h
    while rem > 0:
        step = min(prop.pred_len, rem)
        if ctxs is None:
            p = prop.paths(ctx, step, n_jitter, rng)          # [N, step]
            ctxs = [np.concatenate([ctx, p[i]]) for i in range(len(p))]
        else:
            p = np.stack([prop.paths(c, step, 0, rng)[0] for c in ctxs])
            ctxs = [np.concatenate([ctxs[i], p[i]]) for i in range(len(p))]
        segs.append(p)
        rem -= step
    return np.concatenate(segs, axis=1)


def evaluate_config(config, judge, prop, gift_root, device, rng,
                    max_inst, K, n_jitter, centered=False):
    h = gift.prediction_length(config)
    m = gift.seasonality(config.split("/")[1])
    series = gift.load_series(gift_root, config)
    windows = gift.num_windows(config, min(len(s) for s in series))

    total = sum(1 for _ in gift.iter_test_instances(series, h, windows))
    stride = max(1, total // max_inst)

    readers = ("ttm",) if judge is None else ("ttm", "hybrid_ttm")
    accs = {r: gift.MetricAccumulator() for r in readers}
    sn_acc = gift.MetricAccumulator()
    n_used, n_ttm_nonfinite = 0, 0
    for i, inst in enumerate(gift.iter_test_instances(series, h, windows)):
        if i % stride or n_used >= max_inst:
            continue
        if np.isnan(inst.target).any():
            continue
        if judge is not None:
            ctx = prepare_context(inst.context, judge.input_length,
                                  judge.patching.stride, judge.patching.patch_size)
            if ctx is None:
                continue
        scale = gift.seasonal_error(inst.context, m)

        tp = ttm_rollout(prop, inst.context.astype(np.float32), h, n_jitter, rng)
        # TTM emits NaNs on some extreme contexts (variance ~0, bitbrains):
        # non-finite clean path -> instance skipped for BOTH readers (pairing
        # comes first); non-finite jitters simply dropped.
        if not np.isfinite(tp[0]).all():
            n_ttm_nonfinite += 1
            continue
        tp = tp[np.isfinite(tp).all(axis=1)]
        accs["ttm"].add(inst.target, tp[0], None, scale)      # clean path, point

        if judge is not None:
            # h_judge: the energy reads on the first steps <= the judge's
            # native horizon (single-shot by construction); quantiles on full h.
            cands, e = candidate_energies(judge, ctx, inst.context, h, m, K,
                                          rng, device, extra_cands=tp,
                                          h_judge=judge.prediction_length,
                                          centered=centered)
            fan = fan_from_energies(cands, e)                 # [h, 9]
            accs["hybrid_ttm"].add(inst.target, fan[:, 4], fan, scale)

        sn_acc.add(inst.target, gift.seasonal_naive_forecast(inst.context, h, m),
                   None, scale)
        n_used += 1

    out = {"config": config, "h": h, "n_instances": n_used,
           "n_ttm_nonfinite_skipped": n_ttm_nonfinite,
           "seasonal_naive_local": sn_acc.result()}
    for r, acc in accs.items():
        out[r] = acc.result()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--judge-checkpoint", default=None,
                    help="required unless --ttm-only")
    ap.add_argument("--ttm-only", action="store_true",
                    help="cross-validation of the TTM claim (2026-08-31): raw "
                         "TTM on the FULL windows, no judge, aggregated vs "
                         "local AND official SN. Its MASE compares to the "
                         "0.7240 of the leaderboard CSV; the gap measures "
                         "what their per-regime recipe + pipeline buy over "
                         "ONE variant (1024-96) rolled by us. Its 'CRPS' is "
                         "a collapsed point (=ND): only compare while "
                         "saying so.")
    ap.add_argument("--judge-config", default="lotsa_tiny_v3_eval")
    ap.add_argument("--ttm-model", default="ibm-granite/granite-timeseries-ttm-r3")
    ap.add_argument("--ttm-revision", default="1024-96-r3")
    ap.add_argument("--ttm-jitter", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=24, help="pool bootstraps")
    ap.add_argument("--instances", type=int, default=150, help="per config, paired")
    ap.add_argument("--configs", default=None,
                    help="GIFT subset 'a/b/c,d/e/f' (default: all 97)")
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centered-bootstrap", action="store_true",
                    help="G12c: seasonal-innovation bootstrap glued onto the "
                         "TTM path (anti-dilution by construction)")
    ap.add_argument("--tag", default=None,
                    help="output-directory suffix - REQUIRED for a pool "
                         "variant (otherwise marker collision with the same "
                         "judge's base run)")
    args = ap.parse_args()
    if not args.ttm_only and args.judge_checkpoint is None:
        ap.error("--judge-checkpoint is required (unless --ttm-only)")
    if args.ttm_only and args.instances == 150:
        args.instances = 10 ** 9           # full windows by default
    if args.ttm_only:
        args.ttm_jitter = 0                # single clean path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge = None
    if not args.ttm_only:
        config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=args.judge_config)
        judge = create_model_from_config(cfg)
        judge = load_checkpoint(judge, args.judge_checkpoint, device)
        judge.to(device).eval()
    prop = TTMProposer(args.ttm_model, device, revision=args.ttm_revision)

    configs = ([c.strip() for c in args.configs.split(",")] if args.configs
               else list(gift.GIFT_CONFIGS))
    rng = np.random.default_rng(args.seed)
    stem = ("ttm_raw" if args.ttm_only
            else Path(args.judge_checkpoint).stem)
    out_dir = (Path("evaluation") / "gift_hybrid"
               / (stem + (f"_{args.tag}" if args.tag else "")))
    (out_dir / "per_config").mkdir(parents=True, exist_ok=True)

    results = {}
    for i, config in enumerate(configs, 1):
        marker = out_dir / "per_config" / (config.replace("/", "__") + ".json")
        if marker.exists():
            results[config] = json.loads(marker.read_text())
            logger.info(f"[{i}/{len(configs)}] {config}: already done, skipped")
            continue
        t0 = time.time()
        try:
            res = evaluate_config(config, judge, prop, Path(args.gift_root),
                                  device, rng, args.instances,
                                  args.candidates, args.ttm_jitter,
                                  centered=args.centered_bootstrap)
        except Exception as exc:   # one broken config must not kill 96 others
            logger.error(f"[{i}/{len(configs)}] {config} FAILED: "
                         f"{type(exc).__name__}: {exc}")
            continue
        marker.write_text(json.dumps(res, indent=1))
        results[config] = res
        t = res["ttm"]
        if "hybrid_ttm" in res:
            hy = res["hybrid_ttm"]
            cov = hy.get("coverage") or {}
            extra = (f"hybrid MASE {hy['MASE']:.3f} CRPS {hy['CRPS']:.3f} "
                     f"cov80 {(cov.get('0.9', 0) - cov.get('0.1', 0)):.2f} ")
        else:
            extra = ""
        logger.info(
            f"[{i}/{len(configs)}] {config}: ttm MASE {t['MASE']:.3f} | "
            f"{extra}({res['n_instances']} inst, {time.time() - t0:.0f}s)")

    # Aggregates: geomean of ratios vs LOCAL SN (same windows - paired).
    def geomean_ratio(metric, reader):
        ratios = [results[c][reader][metric] / results[c]["seasonal_naive_local"][metric]
                  for c in results
                  if results[c]["seasonal_naive_local"][metric] > 0]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")

    readers = ("ttm",) if args.ttm_only else ("ttm", "hybrid_ttm")
    summary = {"judge": args.judge_checkpoint, "ttm": args.ttm_model,
               "revision": args.ttm_revision, "instances_cap": args.instances,
               "aggregates_vs_local_sn": {
                   r: {m: geomean_ratio(m, r) for m in ("MASE", "CRPS")}
                   for r in readers}}
    covs = [results[c]["hybrid_ttm"].get("coverage") for c in results
            if results[c].get("hybrid_ttm", {}).get("coverage")]
    if covs:
        summary["hybrid_coverage80_mean"] = float(
            np.mean([c["0.9"] - c["0.1"] for c in covs]))
    if args.ttm_only:
        # The cross-validation number: ratio vs the vendored OFFICIAL SN -
        # the leaderboard normalization (TTM-R3-PT: MASE 0.7240).
        summary["ttm_vs_official_sn"] = gift.aggregate(
            {c: results[c]["ttm"] for c in results},
            gift.official_seasonal_naive())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    for r in readers:
        a = summary["aggregates_vs_local_sn"][r]
        note = "  (collapsed point CRPS - do not cite)" if r == "ttm" else ""
        logger.info(f"  {r:11s} MASE ratio {a['MASE']:.4f} | "
                    f"CRPS ratio {a['CRPS']:.4f}{note}  [vs local SN]")
    if args.ttm_only:
        o = summary["ttm_vs_official_sn"]
        logger.info(
            f"  ttm vs OFFICIAL SN: MASE ratio {o['geomean_MASE_ratio']:.4f} "
            f"({o['n_configs_MASE']} configs) | TTM-R3-PT leaderboard claim: "
            f"0.7240 | point CRPS {o['geomean_CRPS_ratio']:.4f} vs their "
            f"probabilistic 0.5195 (not comparable)")
    if covs:
        logger.info(f"  hybrid 80% coverage: "
                    f"{summary['hybrid_coverage80_mean']:.3f} (nominal 0.800)")
    logger.info(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
