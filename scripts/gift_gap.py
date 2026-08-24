#!/usr/bin/env python
"""
Décomposition per-config de l'écart GIFT-Eval — nous vs les concurrents cibles.

    python scripts/gift_gap.py \
        --per-config evaluation/timejepa_lotsa_tiny_zs/mix1ep3e4_25pct_mase0.8914_crps0.6134/gift/per_config

Croise nos per-config (JSON du harness evaluate_gift.py) avec les per-config
OFFICIELS vendorés des concurrents (docs/assets/gift_leaderboard/<date>/raw/),
dans la convention du leaderboard (ratios vs Seasonal_Naive OFFICIEL, geomean).

Rapporte :
  1. la QUEUE : les configs classées par contribution à la geomean CRPS
     (log-contribution), avec le « geomean sans la queue » ;
  2. les agrégats par FRÉQUENCE et par TERME ;
  3. le duel per-config contre chaque concurrent : où l'on perd le plus,
     où l'on gagne, et le what-if « queue ramenée au niveau du concurrent ».

C'est la carte E19 : elle écrit les prédictions falsifiables per-config des
runs suivants (corpus v3, h512, xres) et vérifie la prédiction d'E18
(« le mix doit comprimer la queue de 16 configs »).
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
                    help="dossier per_config/ du checkpoint à décomposer")
    ap.add_argument("--snapshot", default=None,
                    help="dossier snapshot leaderboard (défaut : le plus récent)")
    ap.add_argument("--competitors", default=",".join(COMPETITORS))
    ap.add_argument("--tail", type=int, default=16, help="taille de la queue analysée")
    args = ap.parse_args()

    root = Path("docs/assets/gift_leaderboard")
    snap = Path(args.snapshot) if args.snapshot else sorted(
        d for d in root.iterdir() if d.is_dir())[-1]
    raw = snap / "raw"

    sn_path = next((p for p in raw.iterdir()
                    if "seasonal" in p.name.lower() and "naive" in p.name.lower()), None)
    if sn_path is None:
        sys.exit(f"✗ seasonal_naive introuvable dans {raw}")
    sn = load_official_csv(sn_path)

    ours = load_ours(Path(args.per_config))
    common = sorted(c for c in ours if c in sn)
    if len(common) < 97:
        print(f"⚠ {len(common)}/97 configs seulement (éval incomplète ? "
              f"per_config partiel ?) — l'analyse porte sur l'intersection.")

    # nos ratios (convention leaderboard : vs SN OFFICIEL)
    r_mase = {c: ours[c][0] / sn[c][0] for c in common}
    r_crps = {c: ours[c][1] / sn[c][1] for c in common}
    g_mase, g_crps = geomean(r_mase.values()), geomean(r_crps.values())
    n = len(common)
    print(f"\n{'=' * 80}\nE19 — DÉCOMPOSITION DU CHECKPOINT  [{args.per_config}]")
    print(f"{n} configs | geomean vs SN officiel : MASE {g_mase:.4f} | CRPS {g_crps:.4f}")
    print(f"{'=' * 80}")

    # 1. LA QUEUE (par log-contribution CRPS à la geomean)
    contrib = sorted(common, key=lambda c: r_crps[c], reverse=True)
    tail = contrib[:args.tail]
    body = [c for c in common if c not in tail]
    print(f"\nQUEUE — top {args.tail} contributions à la geomean CRPS "
          f"(sans elles : CRPS {geomean([r_crps[c] for c in body]):.4f}, "
          f"MASE {geomean([r_mase[c] for c in body]):.4f}) :")
    print(f"  {'config':<38s}{'CRPS ratio':>11s}{'MASE ratio':>11s}{'pts geomean':>12s}")
    for c in tail:
        pts = (1 - math.exp(-math.log(max(r_crps[c], 1e-9)) / n)) * g_crps
        print(f"  {c:<38s}{r_crps[c]:>11.3f}{r_mase[c]:>11.3f}{pts * 100:>11.2f}%")

    # 2. Agrégats par fréquence et par terme
    def bucket(keyfn, label):
        groups = {}
        for c in common:
            groups.setdefault(keyfn(c), []).append(c)
        print(f"\nPAR {label} (geomean CRPS ratio | MASE | n) :")
        for k in sorted(groups, key=lambda k: -geomean([r_crps[c] for c in groups[k]])):
            cs = groups[k]
            print(f"  {k:<8s} {geomean([r_crps[c] for c in cs]):>7.3f}"
                  f"  {geomean([r_mase[c] for c in cs]):>7.3f}   n={len(cs)}")

    bucket(lambda c: c.split("/")[1], "FRÉQUENCE")
    bucket(lambda c: c.split("/")[2], "TERME")

    # 3. Duel contre chaque concurrent
    for comp in args.competitors.split(","):
        path = raw / f"{comp}.csv"
        if not path.exists():
            print(f"\n⚠ {comp}: per-config absent du snapshot, sauté")
            continue
        theirs = load_official_csv(path)
        both = [c for c in common if c in theirs]
        t_crps = {c: theirs[c][1] / sn[c][1] for c in both}
        rel = {c: r_crps[c] / t_crps[c] for c in both}  # >1 = on perd
        wins = sum(1 for c in both if rel[c] < 1.0)
        print(f"\n{'-' * 80}\nVS {comp}  (leur geomean CRPS {geomean(t_crps.values()):.4f}) "
              f"— on gagne {wins}/{len(both)} configs")
        worst = sorted(both, key=lambda c: rel[c], reverse=True)[:8]
        best = sorted(both, key=lambda c: rel[c])[:5]
        print("  où l'on PERD le plus (nous/eux) :")
        for c in worst:
            print(f"    {c:<38s} ×{rel[c]:.2f}  (nous {r_crps[c]:.3f} vs eux {t_crps[c]:.3f})")
        print("  où l'on GAGNE le plus :")
        for c in best:
            print(f"    {c:<38s} ×{rel[c]:.2f}  (nous {r_crps[c]:.3f} vs eux {t_crps[c]:.3f})")
        # what-if : la queue ramenée à LEUR niveau
        whatif = [min(r_crps[c], t_crps[c]) if c in tail else r_crps[c] for c in both]
        print(f"  WHAT-IF queue({args.tail}) ramenée à leur niveau : "
              f"CRPS {geomean(whatif):.4f} (réel {geomean([r_crps[c] for c in both]):.4f})")

    print()


if __name__ == "__main__":
    main()
