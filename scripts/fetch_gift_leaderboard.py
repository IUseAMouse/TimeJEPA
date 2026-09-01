# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round (the
# 2026-08-24 leaderboard snapshot vendored under docs/assets/gift_leaderboard/);
# kept per the no-delete policy.
"""
Local snapshot of the GIFT-Eval leaderboard (HF space Salesforce/GIFT-Eval).

    python scripts/fetch_gift_leaderboard.py            # -> docs/assets/gift_leaderboard/<date>/

Downloads each leaderboard model's official all_results.csv, then recomputes
BOTH aggregates (MASE and CRPS/WQL) with the leaderboard formula - geometric
mean of per-config ratios against the official Seasonal_Naive - i.e. exactly
our harness's convention (evaluation/gift.py).

Why vendor rather than consult the site: (1) the ranks cited in the registry
(E17, E19...) must stay verifiable even if the leaderboard moves; (2) the
competitors' raw per-config files enable tail comparisons (the E18
decomposition against Moirai_small, TTM, etc.) without re-scraping.

Outputs:
    docs/assets/gift_leaderboard/<date>/raw/<model>.csv     (official 97 configs)
    docs/assets/gift_leaderboard/<date>/leaderboard.csv     (rank, MASE ratio, CRPS ratio)
A model missing the Seasonal_Naive's full config set is aggregated on the
intersection and marked (n_configs) - stated, not hidden.
"""

import argparse
import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

SPACE = "https://huggingface.co/spaces/Salesforce/GIFT-Eval/resolve/main"
API = "https://huggingface.co/api/spaces/Salesforce/GIFT-Eval/tree/main/results"
MASE_COL = "eval_metrics/MASE[0.5]"
WQL_COL = "eval_metrics/mean_weighted_sum_quantile_loss"


def http(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "timejepa-snapshot"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def read_results(blob: bytes) -> dict:
    """dataset -> (mase, wql), NaNs filtered out."""
    out = {}
    for row in csv.DictReader(io.StringIO(blob.decode("utf-8"))):
        try:
            out[row["dataset"]] = (float(row[MASE_COL]), float(row[WQL_COL]))
        except (KeyError, ValueError):
            continue
    return out


def geomean(vals):
    vals = [v for v in vals if v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def fetch_meta(models, out_dir: Path):
    """
    Enrichment (2026-08-24): submission config.json files carry the official
    ORG, model link, model_type, testdata_leakage,
    replication_code_available - and the linked model's HF API gives the real
    size (safetensors.total). Vendored in models_meta.csv, next to the
    ranking, for gift_rank.py.
    """
    rows = []
    for i, m in enumerate(models):
        meta = {"model": m, "org": "", "params_m": "", "model_type": "",
                "model_link": "", "testdata_leakage": "",
                "replication_code_available": ""}
        try:
            cfg = json.loads(http(
                f"{SPACE}/results/{urllib.parse.quote(m)}/config.json"))
            for k in ("org", "model_type", "model_link",
                      "testdata_leakage", "replication_code_available"):
                meta[k] = str(cfg.get(k, "") or "")
        except Exception:
            pass
        link = meta["model_link"]
        if "huggingface.co/" in link:
            repo = link.split("huggingface.co/")[-1].strip("/").removeprefix("models/")
            repo = "/".join(repo.split("/")[:2])
            try:
                info = json.loads(http(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}"))
                total = (info.get("safetensors") or {}).get("total")
                if total:
                    meta["params_m"] = f"{total / 1e6:.1f}"
            except Exception:
                pass
        rows.append(meta)
        if (i + 1) % 25 == 0:
            print(f"  meta {i + 1}/{len(models)}")

    with open(out_dir / "models_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_org = sum(1 for r in rows if r["org"])
    n_par = sum(1 for r in rows if r["params_m"])
    print(f"models_meta.csv: {n_org}/{len(rows)} orgs, {n_par}/{len(rows)} sizes")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="default: docs/assets/gift_leaderboard/<date>")
    ap.add_argument("--enrich-only", default=None, metavar="SNAPSHOT_DIR",
                    help="only add models_meta.csv to an existing snapshot "
                         "(without re-downloading the result CSVs)")
    args = ap.parse_args()

    if args.enrich_only:
        snap = Path(args.enrich_only)
        with open(snap / "leaderboard.csv") as f:
            models = [r["model"] for r in csv.DictReader(f)]
        print(f"{len(models)} models from snapshot {snap.name}")
        fetch_meta(models, snap)
        return

    out_dir = Path(args.out or f"docs/assets/gift_leaderboard/{date.today().isoformat()}")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    models = [e["path"].split("/")[-1] for e in json.loads(http(API))
              if e["type"] == "directory"]
    print(f"{len(models)} models on the leaderboard")

    results = {}
    for i, m in enumerate(models):
        try:
            blob = http(f"{SPACE}/results/{urllib.parse.quote(m)}/all_results.csv")
        except Exception as exc:                       # noqa: BLE001 - list it, do not hide it
            print(f"  x {m}: {exc}")
            continue
        (raw_dir / f"{m}.csv").write_bytes(blob)
        results[m] = read_results(blob)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(models)} downloaded")

    sn_key = next((m for m in results if "naive" in m.lower()
                   and "seasonal" in m.lower()), None)
    if sn_key is None:
        sys.exit(f"Seasonal Naive not found among: {sorted(results)[:10]}...")
    sn = results[sn_key]
    print(f"baseline: {sn_key} ({len(sn)} configs)")

    rows = []
    for m, res in results.items():
        common = [d for d in sn if d in res]
        if len(common) < 50:
            print(f"  warning {m}: only {len(common)} configs, excluded from ranking")
            continue
        rows.append({
            "model": m,
            "mase_ratio": geomean([res[d][0] / sn[d][0] for d in common]),
            "crps_ratio": geomean([res[d][1] / sn[d][1] for d in common]),
            "n_configs": len(common),
        })

    rows.sort(key=lambda r: r["crps_ratio"])
    for rank, r in enumerate(rows, 1):
        r["rank_crps"] = rank
    for rank, r in enumerate(sorted(rows, key=lambda r: r["mase_ratio"]), 1):
        r["rank_mase"] = rank

    with open(out_dir / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank_crps", "rank_mase", "model",
                                          "crps_ratio", "mase_ratio", "n_configs"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    fetch_meta(sorted(results), out_dir)

    print(f"\n{len(rows)} models ranked -> {out_dir / 'leaderboard.csv'}")
    print("Top 5 CRPS:")
    for r in rows[:5]:
        print(f"  {r['rank_crps']:3d}. {r['model']:32s} CRPS {r['crps_ratio']:.4f}  MASE {r['mase_ratio']:.4f}")


if __name__ == "__main__":
    main()
