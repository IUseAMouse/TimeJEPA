"""
Paired reading of the TTM inference-layer runs (2026-09-06).

    python scripts/ttm_layers_paired.py \
        --runs raw=evaluation/gift_hybrid/ttm_raw_inst10 \
               flip=evaluation/gift_hybrid/ttm_raw_flip_inst10 \
               mix=evaluation/gift_hybrid/ttm_raw_flip_ratein-mix-pool_inst10

Each run aggregates on its own set of finite configs (TTM emits NaNs on
near-constant contexts, and decimation changes which contexts are
near-constant), so the printed aggregates are NOT paired. This script
intersects the configs that are finite in EVERY run, re-aggregates each run
on that common set against the OFFICIAL Seasonal Naive (leaderboard
convention), and lists the configs the last run gains and loses the most
against the first (MASE ratio). Diagnostic on capped instances - a
paired comparison, not a leaderboard number.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import gift  # noqa: E402


def load(d: Path, reader: str = "ttm") -> dict:
    out = {}
    for f in sorted((d / "per_config").glob("*.json")):
        j = json.loads(f.read_text())
        m = j.get(reader, {})
        if m and m.get("n_instances", 0) > 0 and math.isfinite(m.get("MASE", float("nan"))):
            out[j["config"]] = {"MASE": m["MASE"], "CRPS": m["CRPS"],
                                "n": m["n_instances"],
                                "mix": (j.get("ttm_layers") or {}).get("mix")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir ...")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        runs[name] = load(Path(d))
    names = list(runs)
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    # instance counts must match too, else the pairing is broken on that config
    same_n = [c for c in common if len({runs[n][c]["n"] for n in names}) == 1]
    sn = gift.official_seasonal_naive()
    print(f"configs finite in every run: {len(common)} | with identical instance "
          f"counts: {len(same_n)} (aggregates below use these)")
    for n in names:
        a = gift.aggregate({c: runs[n][c] for c in same_n}, sn)
        print(f"  {n:<6} MASE ratio {a['geomean_MASE_ratio']:.4f} "
              f"(point CRPS {a['geomean_CRPS_ratio']:.4f}, collapsed - do not cite)")
    first, last = names[0], names[-1]
    rows = []
    for c in same_n:
        r = (runs[last][c]["MASE"] / sn[c]["MASE"]) / (runs[first][c]["MASE"] / sn[c]["MASE"])
        rows.append((r, c))
    rows.sort()
    print(f"\n{last} vs {first} - biggest gains (MASE ratio last/first):")
    for r, c in rows[:args.top]:
        print(f"  {c:<36} x{r:.2f}  mix {runs[last][c]['mix']}")
    print(f"\n{last} vs {first} - biggest losses:")
    for r, c in rows[-args.top:][::-1]:
        print(f"  {c:<36} x{r:.2f}  mix {runs[last][c]['mix']}")
    wins = sum(1 for r, _ in rows if r < 1)
    print(f"\n{last} better than {first} on {wins}/{len(rows)} configs; "
          f"geomean of per-config ratios {math.exp(sum(math.log(r) for r, _ in rows) / len(rows)):.4f}")


if __name__ == "__main__":
    main()
