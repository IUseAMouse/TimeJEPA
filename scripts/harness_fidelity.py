"""
Does our GIFT harness reproduce the official numbers? (2026-10-05)

For a public model evaluated through `scripts/evaluate_gift.py` WITHOUT any
inference layer, compare each config's raw MASE and CRPS with the model's
official per-config results (the leaderboard's all_results.csv, vendored under
docs/assets/gift_leaderboard/<date>/raw/), and our local seasonal naive's MASE
with the official one - a model-free check of the instances, the seasonal
error and the aggregation.

    python scripts/harness_fidelity.py evaluation/chronos_bolt_small/amazon__chronos-bolt-small/gift \
        --official docs/assets/gift_leaderboard/2026-09-06/raw/chronos_bolt_small.csv

A faithful harness gives 0.00% on every config. A deviation is listed with
its config so the cause can be read (missing values, context length,
multivariate handling, precision).
"""

import argparse
import csv
import json
import math
from pathlib import Path

MASE = "eval_metrics/MASE[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
SNAPSHOT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "gift_leaderboard" / "2026-09-06" / "raw"


def load_official(path) -> dict:
    with open(path) as f:
        return {r["dataset"]: (float(r[MASE]), float(r[CRPS])) for r in csv.DictReader(f)}


def load_run(run_dir) -> dict:
    out = {}
    for p in sorted(Path(run_dir, "per_config").glob("*.json")):
        d = json.loads(p.read_text())
        out[d["config"]] = (d["model"]["MASE"], d["model"]["CRPS"],
                            (d.get("seasonal_naive_local") or {}).get("MASE"))
    return out


def geomean(xs) -> float:
    xs = [x for x in xs if x and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run dir (.../gift/) of a bare evaluation")
    ap.add_argument("--official", required=True, help="the model's official per-config CSV")
    ap.add_argument("--seasonal-naive", default=str(SNAPSHOT / "seasonal_naive.csv"))
    ap.add_argument("--tolerance", type=float, default=0.05, help="deviation listed above this, in percent")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    ours, official, sn = load_run(args.run), load_official(args.official), load_official(args.seasonal_naive)
    common = sorted(set(ours) & set(official) & set(sn))
    print(f"{len(common)} configs in common ({len(ours)} ours, {len(official)} official)")
    rows = []
    for c in common:
        mase, crps, sn_local = ours[c]
        rows.append((c, 100 * (mase / official[c][0] - 1), 100 * (crps / official[c][1] - 1),
                     100 * (sn_local / sn[c][0] - 1) if sn_local else float("nan")))
    for label, i, ref in (("model MASE", 1, 0), ("model CRPS", 2, 1)):
        agg_ours = geomean(ours[c][i - 1] / sn[c][ref] for c in common)
        agg_off = geomean(official[c][ref] / sn[c][ref] for c in common)
        exact = sum(abs(r[i]) <= args.tolerance for r in rows)
        print(f"{label}: ours {agg_ours:.4f} | official {agg_off:.4f} ({100 * (agg_ours / agg_off - 1):+.2f}%) | "
              f"{exact}/{len(rows)} configs within {args.tolerance}%")
    sn_rows = [r for r in rows if math.isfinite(r[3])]
    print(f"seasonal naive MASE (model-free): {sum(abs(r[3]) <= args.tolerance for r in sn_rows)}/{len(sn_rows)} "
          f"configs within {args.tolerance}%, worst {max((abs(r[3]) for r in sn_rows), default=float('nan')):.3f}%")
    off = sorted((r for r in rows if max(abs(r[1]), abs(r[2])) > args.tolerance),
                 key=lambda r: -max(abs(r[1]), abs(r[2])))
    if off:
        print(f"\n{len(off)} configs beyond {args.tolerance}% (ours relative to official):")
        print(f"  {'config':38s} {'MASE':>9s} {'CRPS':>9s} {'SN MASE':>9s}")
        for c, dm, dc, ds in off[:args.top]:
            print(f"  {c:38s} {dm:+8.2f}% {dc:+8.2f}% {ds:+8.2f}%")


if __name__ == "__main__":
    main()
