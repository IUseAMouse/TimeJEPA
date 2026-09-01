# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round (the
# E18/E19 gap decomposition); kept per the no-delete policy.
"""
Per-config decomposition of the GIFT-Eval gap - us vs the target competitors.

    python scripts/gift_gap.py \
        --per-config evaluation/timejepa_lotsa_tiny_zs/mix1ep3e4_25pct_mase0.8914_crps0.6134/gift/per_config

Crosses our per-config files (JSON from the evaluate_gift.py harness) with the
competitors' vendored OFFICIAL per-config files
(docs/assets/gift_leaderboard/<date>/raw/), in the leaderboard convention
(ratios vs the OFFICIAL Seasonal_Naive, geomean).

Reports:
  1. the TAIL: configs ranked by contribution to the CRPS geomean
     (log-contribution), with the "geomean without the tail";
  2. aggregates by FREQUENCY and by TERM;
  3. the per-config duel against each competitor: where we lose most, where
     we win, and the what-if "tail brought down to the competitor's level".

This is the E19 map: it writes the falsifiable per-config predictions of the
next runs (v3 corpus, h512, xres) and checks the E18 prediction ("the mix
must compress the 16-config tail").
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

COMPETITORS = ["YingLong_6m", "Moirai_base", "TTM-R3-PT", "Toto-2.0-4m", "FlowState-9.1M"]
MASE_COL = "eval_metrics/MASE[0.5]"
WQL_COL = "eval_metrics/mean_weighted_sum_quantile_loss"


def load_official_csv(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[r["dataset"]] = (float(r[MASE_COL]), float(r[WQL_COL]))
            except (KeyError, ValueError):
                continue
    return out


def load_ours(per_config_dir: Path) -> dict:
    out = {}
    for p in sorted(per_config_dir.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["config"]] = (d["model"]["MASE"], d["model"]["CRPS"])
    return out


def geomean(vals):
    vals = [v for v in vals if v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--per-config", required=True,
                    help="per_config/ directory of the checkpoint to decompose")
    ap.add_argument("--snapshot", default=None,
                    help="leaderboard snapshot directory (default: most recent)")
    ap.add_argument("--competitors", default=",".join(COMPETITORS))
    ap.add_argument("--tail", type=int, default=16, help="size of the analyzed tail")
    args = ap.parse_args()

    root = Path("docs/assets/gift_leaderboard")
    snap = Path(args.snapshot) if args.snapshot else sorted(
        d for d in root.iterdir() if d.is_dir())[-1]
    raw = snap / "raw"

    sn_path = next((p for p in raw.iterdir()
                    if "seasonal" in p.name.lower() and "naive" in p.name.lower()), None)
    if sn_path is None:
        sys.exit(f"seasonal_naive not found in {raw}")
    sn = load_official_csv(sn_path)

    ours = load_ours(Path(args.per_config))
    common = sorted(c for c in ours if c in sn)
    if len(common) < 97:
        print(f"warning: only {len(common)}/97 configs (incomplete eval? "
              f"partial per_config?) - the analysis covers the intersection.")

    # our ratios (leaderboard convention: vs OFFICIAL SN)
    r_mase = {c: ours[c][0] / sn[c][0] for c in common}
    r_crps = {c: ours[c][1] / sn[c][1] for c in common}
    g_mase, g_crps = geomean(r_mase.values()), geomean(r_crps.values())
    n = len(common)
    print(f"\n{'=' * 80}\nE19 - CHECKPOINT DECOMPOSITION  [{args.per_config}]")
    print(f"{n} configs | geomean vs official SN: MASE {g_mase:.4f} | CRPS {g_crps:.4f}")
    print(f"{'=' * 80}")

    # 1. THE TAIL (by CRPS log-contribution to the geomean)
    contrib = sorted(common, key=lambda c: r_crps[c], reverse=True)
    tail = contrib[:args.tail]
    body = [c for c in common if c not in tail]
    print(f"\nTAIL - top {args.tail} contributions to the CRPS geomean "
          f"(without them: CRPS {geomean([r_crps[c] for c in body]):.4f}, "
          f"MASE {geomean([r_mase[c] for c in body]):.4f}):")
    print(f"  {'config':<38s}{'CRPS ratio':>11s}{'MASE ratio':>11s}{'pts geomean':>12s}")
    for c in tail:
        pts = (1 - math.exp(-math.log(max(r_crps[c], 1e-9)) / n)) * g_crps
        print(f"  {c:<38s}{r_crps[c]:>11.3f}{r_mase[c]:>11.3f}{pts * 100:>11.2f}%")

    # 2. Aggregates by frequency and by term
    def bucket(keyfn, label):
        groups = {}
        for c in common:
            groups.setdefault(keyfn(c), []).append(c)
        print(f"\nBY {label} (geomean CRPS ratio | MASE | n):")
        for k in sorted(groups, key=lambda k: -geomean([r_crps[c] for c in groups[k]])):
            cs = groups[k]
            print(f"  {k:<8s} {geomean([r_crps[c] for c in cs]):>7.3f}"
                  f"  {geomean([r_mase[c] for c in cs]):>7.3f}   n={len(cs)}")

    bucket(lambda c: c.split("/")[1], "FREQUENCY")
    bucket(lambda c: c.split("/")[2], "TERM")

    # 3. Duel against each competitor
    for comp in args.competitors.split(","):
        path = raw / f"{comp}.csv"
        if not path.exists():
            print(f"\nwarning {comp}: per-config missing from the snapshot, skipped")
            continue
        theirs = load_official_csv(path)
        both = [c for c in common if c in theirs]
        t_crps = {c: theirs[c][1] / sn[c][1] for c in both}
        rel = {c: r_crps[c] / t_crps[c] for c in both}  # >1 = we lose
        wins = sum(1 for c in both if rel[c] < 1.0)
        print(f"\n{'-' * 80}\nVS {comp}  (their geomean CRPS {geomean(t_crps.values()):.4f}) "
              f"- we win {wins}/{len(both)} configs")
        worst = sorted(both, key=lambda c: rel[c], reverse=True)[:8]
        best = sorted(both, key=lambda c: rel[c])[:5]
        print("  where we LOSE the most (us/them):")
        for c in worst:
            print(f"    {c:<38s} x{rel[c]:.2f}  (us {r_crps[c]:.3f} vs them {t_crps[c]:.3f})")
        print("  where we WIN the most:")
        for c in best:
            print(f"    {c:<38s} x{rel[c]:.2f}  (us {r_crps[c]:.3f} vs them {t_crps[c]:.3f})")
        # what-if: the tail brought down to THEIR level
        whatif = [min(r_crps[c], t_crps[c]) if c in tail else r_crps[c] for c in both]
        print(f"  WHAT-IF tail({args.tail}) brought to their level: "
              f"CRPS {geomean(whatif):.4f} (actual {geomean([r_crps[c] for c in both]):.4f})")

    print()


if __name__ == "__main__":
    main()
