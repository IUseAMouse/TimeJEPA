"""
RateIN selection gap - where the backtest selector loses against the oracle.

Reads two evaluate_gift output directories for the SAME checkpoint (same TTA
layers): one run with +ratein=backtest, one with +ratein=oracle. The oracle
cache holds the full per-k CRPS table of every config (test side), the
backtest cache holds the k it chose (k_hist) and, for runs after 2026-09-05,
the pooled ratio table it saw. The aggregate is a geometric mean, so each
config's contribution to the residual is log(CRPS_bt) - log(CRPS_oracle),
divided by the number of configs.

The residual splits into four exclusive cases per config:
  missed     : selector kept k=1, oracle wants k>1  (margin / coverage / blind)
  false_pos  : selector chose k>1, oracle wants k=1
  wrong_k    : both k>1, different k
  match      : same k (residual = per-instance guard noise, ~0)

Usage:
  python scripts/ratein_selection_gap.py \
      --bt evaluation/<model>/<ckpt>/gift_flip_ratein-bt \
      --oracle evaluation/<model>/<ckpt>/gift_flip_ratein-oracle [--top 15]

Diagnostic only: the oracle peeks at the test set. Nothing here is a result,
it is the map of where a better CAUSAL selector could gain.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_dir(d: Path) -> dict:
    out = {}
    for f in sorted((d / "per_config").glob("*.json")):
        j = json.loads(f.read_text())
        out[j["config"]] = j
    return out


def chosen_k(entry: dict) -> int:
    """Dominant k of the run (v3 is uniform per config, up to the guard)."""
    hist = entry.get("ratein", {}).get("k_hist", {"1": 1})
    return int(max(hist, key=lambda k: hist[k]))


def classify(k_bt: int, k_or: int) -> str:
    if k_bt == k_or:
        return "match"
    if k_bt == 1:
        return "missed"
    if k_or == 1:
        return "false_pos"
    return "wrong_k"


def analyse(bt: dict, oracle: dict) -> dict:
    common = sorted(set(bt) & set(oracle))
    rows = []
    for c in common:
        b, o = bt[c], oracle[c]
        crps_bt = b["model"]["CRPS"]
        table = o["oracle"]["per_k_crps"]
        k_or = int(o["oracle"]["best_k"])
        crps_or = table[str(k_or)]
        if not (crps_bt > 0 and crps_or > 0):
            continue
        k_bt = chosen_k(b)
        gap = math.log(crps_bt) - math.log(crps_or)
        # What the selector's own k scores on the test (from the oracle
        # table): separates "wrong choice" from per-instance guard noise.
        crps_bt_k_on_test = table.get(str(k_bt))
        diag = b.get("ratein", {}).get("backtest")
        rows.append({
            "config": c, "k_bt": k_bt, "k_oracle": k_or,
            "crps_bt": crps_bt, "crps_oracle": crps_or,
            "gap_log": gap, "case": classify(k_bt, k_or),
            "oracle_gain": o["oracle"].get("gain_vs_k1", 0.0),
            "bt_ratio_at_oracle_k": (diag or {}).get("ratios", {}).get(str(k_or)),
            "bt_ratio_best": (min((diag or {}).get("ratios", {}).values())
                              if diag and diag.get("ratios") else None),
            "crps_bt_k_on_test": crps_bt_k_on_test,
        })
    n = len(rows)
    by_case = defaultdict(lambda: {"n": 0, "gap_log": 0.0})
    for r in rows:
        by_case[r["case"]]["n"] += 1
        by_case[r["case"]]["gap_log"] += r["gap_log"]
    total_gap = sum(r["gap_log"] for r in rows)
    geo_bt = math.exp(sum(math.log(r["crps_bt"]) for r in rows) / n) if n else float("nan")
    geo_or = math.exp(sum(math.log(r["crps_oracle"]) for r in rows) / n) if n else float("nan")
    # Counterfactual aggregate if one case were fixed at oracle quality.
    counterfactual = {
        case: math.exp(math.log(geo_bt) - v["gap_log"] / n)
        for case, v in by_case.items()
    } if n else {}
    # Margin sub-split of the missed configs, when the ratio table exists:
    # the selector saw a gain but below the margin, vs saw no gain at all.
    missed_margin = [r for r in rows if r["case"] == "missed"
                     and r["bt_ratio_best"] is not None
                     and r["bt_ratio_best"] < 1.0]
    return {"n_configs": n, "geomean_bt": geo_bt, "geomean_oracle": geo_or,
            "residual_pts": 100 * (geo_bt - geo_or),
            "by_case": dict(by_case), "counterfactual_geomean": counterfactual,
            "missed_below_margin": [r["config"] for r in missed_margin],
            "rows": sorted(rows, key=lambda r: -r["gap_log"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bt", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full analysis here")
    args = ap.parse_args()

    rep = analyse(load_dir(args.bt), load_dir(args.oracle))
    n = rep["n_configs"]
    print(f"{n} configs in common | geomean CRPS backtest {rep['geomean_bt']:.4f} "
          f"| oracle {rep['geomean_oracle']:.4f} | residual {rep['residual_pts']:.2f} pts")
    print("\ncase        n   share of residual   aggregate if fixed")
    total = sum(v["gap_log"] for v in rep["by_case"].values()) or 1.0
    for case in ("missed", "wrong_k", "false_pos", "match"):
        v = rep["by_case"].get(case)
        if not v:
            continue
        print(f"{case:<10} {v['n']:>3}   {100 * v['gap_log'] / total:>6.1f}%             "
              f"{rep['counterfactual_geomean'][case]:.4f}")
    if rep["missed_below_margin"]:
        print(f"\nmissed with a backtest gain below the margin "
              f"({len(rep['missed_below_margin'])}): "
              + ", ".join(rep["missed_below_margin"]))
    print(f"\ntop {args.top} contributors (gap in log CRPS, share of residual):")
    for r in rep["rows"][:args.top]:
        extra = ""
        if r["bt_ratio_at_oracle_k"] is not None:
            extra = f"  bt ratio at oracle k {r['bt_ratio_at_oracle_k']:.3f}"
        print(f"  {r['config']:<34} {r['case']:<9} k_bt={r['k_bt']:<2} k*={r['k_oracle']:<2} "
              f"CRPS {r['crps_bt']:.4f} -> {r['crps_oracle']:.4f} "
              f"({100 * r['gap_log'] / total:.1f}%){extra}")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
        print(f"\nfull analysis: {args.json}")


if __name__ == "__main__":
    main()
