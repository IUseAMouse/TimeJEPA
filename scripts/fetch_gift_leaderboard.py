#!/usr/bin/env python
"""
Snapshot local du leaderboard GIFT-Eval (espace HF Salesforce/GIFT-Eval).

    python scripts/fetch_gift_leaderboard.py            # -> docs/assets/gift_leaderboard/<date>/

Télécharge le all_results.csv officiel de chaque modèle du leaderboard, puis
recalcule les DEUX agrégats (MASE et CRPS/WQL) avec la formule du leaderboard —
moyenne géométrique des ratios par config contre le Seasonal_Naive officiel —
c'est-à-dire exactement la convention de notre harnais (evaluation/gift.py).

Pourquoi vendorer plutôt que consulter le site : (1) les rangs cités dans le
registre (E17, E19...) doivent rester vérifiables même si le leaderboard bouge ;
(2) les per-config bruts des concurrents permettent les comparaisons de queue
(la décomposition E18 contre Moirai_small, TTM, etc.) sans re-scraper.

Sorties :
    docs/assets/gift_leaderboard/<date>/raw/<modèle>.csv    (97 configs officiels)
    docs/assets/gift_leaderboard/<date>/leaderboard.csv     (rang, MASE ratio, CRPS ratio)
Un modèle sans le set complet de configs du Seasonal_Naive est agrégé sur
l'intersection et marqué (n_configs) — dit, pas caché.
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
    """dataset -> (mase, wql), NaN filtrés."""
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
    Enrichissement (2026-08-24) : les config.json de soumission portent l'ORG
    officielle, le lien modèle, model_type, testdata_leakage,
    replication_code_available — et l'API HF du modèle lié donne la taille
    réelle (safetensors.total). Vendored dans models_meta.csv, à côté du
    classement, pour gift_rank.py.
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
    print(f"models_meta.csv : {n_org}/{len(rows)} orgs, {n_par}/{len(rows)} tailles")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="défaut : docs/assets/gift_leaderboard/<date>")
    ap.add_argument("--enrich-only", default=None, metavar="SNAPSHOT_DIR",
                    help="n'ajoute que models_meta.csv à un snapshot existant "
                         "(sans re-télécharger les CSV de résultats)")
    args = ap.parse_args()

    if args.enrich_only:
        snap = Path(args.enrich_only)
        with open(snap / "leaderboard.csv") as f:
            models = [r["model"] for r in csv.DictReader(f)]
        print(f"{len(models)} modèles du snapshot {snap.name}")
        fetch_meta(models, snap)
        return

    out_dir = Path(args.out or f"docs/assets/gift_leaderboard/{date.today().isoformat()}")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    models = [e["path"].split("/")[-1] for e in json.loads(http(API))
              if e["type"] == "directory"]
    print(f"{len(models)} modèles au leaderboard")

    results = {}
    for i, m in enumerate(models):
        try:
            blob = http(f"{SPACE}/results/{urllib.parse.quote(m)}/all_results.csv")
        except Exception as exc:                       # noqa: BLE001 — on liste, on ne cache pas
            print(f"  ✗ {m}: {exc}")
            continue
        (raw_dir / f"{m}.csv").write_bytes(blob)
        results[m] = read_results(blob)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(models)} téléchargés")

    sn_key = next((m for m in results if "naive" in m.lower()
                   and "seasonal" in m.lower()), None)
    if sn_key is None:
        sys.exit(f"Seasonal Naive introuvable parmi : {sorted(results)[:10]}...")
    sn = results[sn_key]
    print(f"baseline : {sn_key} ({len(sn)} configs)")

    rows = []
    for m, res in results.items():
        common = [d for d in sn if d in res]
        if len(common) < 50:
            print(f"  ⚠ {m}: {len(common)} configs seulement, ignoré du classement")
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

    print(f"\n{len(rows)} modèles classés -> {out_dir / 'leaderboard.csv'}")
    print("Top 5 CRPS :")
    for r in rows[:5]:
        print(f"  {r['rank_crps']:3d}. {r['model']:32s} CRPS {r['crps_ratio']:.4f}  MASE {r['mase_ratio']:.4f}")


if __name__ == "__main__":
    main()
