"""
GIFT-Eval rankings restricted to FOUNDATION MODELS, from a vendored snapshot.

    python scripts/gift_foundation_rank.py --snapshot 2026-08-22 \
        --add "TimeJEPA-mini-head8:0.5340:0.7842:4.0" \
        --add "TimeJEPA-tiny:0.5588:0.8152:1.14" > docs/GIFT_RANKINGS.md

The leaderboard top is saturated by agentic systems (routers and ensembles
over several foundation models) and by per-dataset fine-tunes. This script
keeps the rows whose leaderboard `model_type` is `zero-shot` or `pretrained`
and whose `testdata_leakage` is `No`, which is what a single foundation
model evaluated zero-shot competes against. It prints two tables in
Markdown: the full foundation ranking and the sub-10M subset (parameter
counts from the leaderboard metadata; rows without a count are listed with
'n/a' and excluded from the sub-10M table unless the name carries the size).
Our checkpoints (--add name:crps:mase:params_m) are inserted at their rank.
"""

import argparse
import csv
import re
from pathlib import Path

KEEP_TYPES = {"zero-shot", "pretrained"}
SUB10 = 10.0
# The metadata type is self-declared and imperfect: some wrappers around
# other people's foundation models file as `pretrained`. Names that announce
# a wrapper are excluded on top of the type filter (stated in the output).
WRAPPER_PATTERNS = re.compile(r"STRIDE|\(\+|Agent|Ensemble|Route|Orchestra|Copilot|"
                              r"ZooCast|Synapse", re.I)


def load(snapshot: str):
    d = Path("docs/assets/gift_leaderboard") / snapshot
    meta = {r["model"]: r for r in csv.DictReader(open(d / "models_meta.csv"))}
    rows = []
    for r in csv.DictReader(open(d / "leaderboard.csv")):
        m = meta.get(r["model"], {})
        rows.append({"model": r["model"], "crps": float(r["crps_ratio"]),
                     "mase": float(r["mase_ratio"]),
                     "type": m.get("model_type", "?"),
                     "leak": m.get("testdata_leakage", "?"),
                     "params": _params(r["model"], m.get("params_m", "")),
                     "org": m.get("org", ""), "ours": False})
    return rows


def _params(name: str, raw: str):
    if raw not in ("", None):
        try:
            return float(raw)
        except ValueError:
            pass
    mm = re.search(r"[-_](\d+(?:\.\d+)?)[mM]\b", name)     # e.g. Kairos_10m, YingLong_6m
    return float(mm.group(1)) if mm else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--add", action="append", default=[],
                    help="name:crps:mase:params_m (repeatable)")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    rows = load(args.snapshot)
    for spec in args.add:
        name, crps, mase, p = spec.split(":")
        rows.append({"model": name, "crps": float(crps), "mase": float(mase),
                     "type": "zero-shot", "leak": "No", "params": float(p),
                     "org": "ours", "ours": True})
    n_all = len([r for r in rows if not r["ours"]])
    found = [r for r in rows if r["type"] in KEEP_TYPES and r["leak"] == "No"
             and not WRAPPER_PATTERNS.search(r["model"])]
    wrapped = sorted(r["model"] for r in rows if r["type"] in KEEP_TYPES
                     and r["leak"] == "No" and WRAPPER_PATTERNS.search(r["model"]))
    found.sort(key=lambda r: r["crps"])
    sub = [r for r in found if r["params"] is not None and r["params"] < SUB10]

    def table(items, title, limit=None):
        print(f"### {title}\n")
        print("| # | modèle | params (M) | CRPS | MASE | type |")
        print("|---|---|---|---|---|---|")
        for i, r in enumerate(items[:limit] if limit else items, 1):
            p = "n/a" if r["params"] is None else f"{r['params']:g}"
            name = f"**{r['model']}**" if r["ours"] else r["model"]
            print(f"| {i} | {name} | {p} | {r['crps']:.4f} | {r['mase']:.4f} | {r['type']} |")
        print()

    print(f"# Classements GIFT-Eval — fondations seules (snapshot {args.snapshot})\n")
    print(f"Source : `docs/assets/gift_leaderboard/{args.snapshot}/` (leaderboard.csv + "
          f"models_meta.csv, {n_all} entrées, agrégats recalculés avec la formule "
          "officielle : moyenne géométrique des ratios par config contre la Seasonal "
          "Naive officielle).\n")
    print("Filtre « fondation » : `model_type` ∈ {zero-shot, pretrained} et "
          "`testdata_leakage` = No. Exclus : `agentic` (routeurs et ensembles de "
          "plusieurs modèles), `fine-tuned` (adaptation par jeu de données), "
          "`deep-learning` (entraînés par jeu), `statistical`, et toute entrée avec "
          "fuite déclarée. Les lignes en gras sont nos checkpoints, insérés à leur rang "
          "(nos agrégats : même formule, mêmes CSV officiels de la Seasonal Naive).\n")
    kept = len([r for r in found if not r["ours"]])
    print(f"Sur {n_all} entrées, {kept} sont des fondations au sens ci-dessus. "
          f"Exclus en plus par leur nom (enveloppes déclarées `pretrained`) : "
          f"{', '.join(wrapped) if wrapped else 'aucun'}.\n")
    table(found, f"Fondations, top {args.top} (sur {kept})", args.top)
    table(sub, "Fondations de moins de 10M de paramètres")
    print("Notes : TempoPFN n'a pas de compte de paramètres dans les métadonnées ; "
          "son article (arXiv 2510.25502) indique 34.69M, il n'est donc pas dans la "
          "table sub-10M. FlowState apparaît trois fois (FlowState-9.1M, FlowState-r1.1 "
          "et Granite-FlowState-r1.1 : une lignée, deux versions) — un seul modèle "
          "au sens du classement. TTM-R3-FT est exclu (type fine-tuned), TTM-R3-PT "
          "est la référence pretrained. Le type est auto-déclaré par les auteurs : "
          "le filtre est le meilleur possible depuis les métadonnées, pas une "
          "vérité absolue.")


if __name__ == "__main__":
    main()
