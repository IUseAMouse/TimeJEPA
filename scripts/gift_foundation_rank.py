"""
GIFT-Eval rankings restricted to foundation models, from a vendored snapshot.

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


# Parameter counts the leaderboard metadata gets wrong (checked against the
# model cards): FlowState-r1.1 inherits the 9.1M of the r1 card but is
# ~18.5M (output gating on the S5 encoder, 4096 pretraining context, larger
# MLP), so it is not a sub-10M model. The sub-10M FlowState is FlowState-9.1M.
PARAMS_OVERRIDE = {"FlowState-r1.1": 18.5, "Granite-FlowState-r1.1": 18.5}


def _params(name: str, raw: str):
    if name in PARAMS_OVERRIDE:
        return PARAMS_OVERRIDE[name]
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
        print("| # | model | params (M) | CRPS | MASE | type |")
        print("|---|---|---|---|---|---|")
        for i, r in enumerate(items[:limit] if limit else items, 1):
            p = "n/a" if r["params"] is None else f"{r['params']:g}"
            name = f"**{r['model']}**" if r["ours"] else r["model"]
            print(f"| {i} | {name} | {p} | {r['crps']:.4f} | {r['mase']:.4f} | {r['type']} |")
        print()

    print(f"# GIFT-Eval rankings, foundation models only (snapshot {args.snapshot})\n")
    print(f"Source: `docs/assets/gift_leaderboard/{args.snapshot}/` (leaderboard.csv + "
          f"models_meta.csv, {n_all} entries, aggregates recomputed with the official "
          "formula: geometric mean of the per-config ratios against the official "
          "Seasonal Naive).\n")
    print("Foundation filter: `model_type` in {zero-shot, pretrained} and "
          "`testdata_leakage` = No. Excluded: `agentic` (routers and ensembles of "
          "several models), `fine-tuned` (per-dataset adaptation), "
          "`deep-learning` (trained per dataset), `statistical`, and any entry with "
          "declared leakage. Rows in bold are our checkpoints, inserted at their rank "
          "(our aggregates: same formula, same official Seasonal Naive CSVs).\n")
    kept = len([r for r in found if not r["ours"]])
    print(f"Of {n_all} entries, {kept} are foundation models in the above sense. "
          f"Also excluded by name (wrappers declared `pretrained`): "
          f"{', '.join(wrapped) if wrapped else 'none'}.\n")
    table(found, f"Foundation models, top {args.top} (of {kept})", args.top)
    table(sub, "Foundation models under 10M parameters")
    print("Notes: TempoPFN has no parameter count in the metadata; its paper "
          "(arXiv 2510.25502) reports 34.69M, so it is not in the sub-10M table. "
          "FlowState appears three times (FlowState-9.1M, FlowState-r1.1 and "
          "Granite-FlowState-r1.1: one lineage, two versions), one model in the "
          "sense of the ranking. TTM-R3-FT is excluded (type fine-tuned), TTM-R3-PT "
          "is the pretrained reference. The type is self-declared by the authors: "
          "the filter is the best the metadata allows, not ground truth.")


if __name__ == "__main__":
    main()
