#!/usr/bin/env python
"""
Position d'un checkpoint sur le leaderboard GIFT-Eval — visuel CLI.

    python scripts/gift_rank.py --crps 0.6134 --mase 0.8914
    python scripts/gift_rank.py --crps 0.6134 --mase 0.8914 --name "mix1ep3e4@25%" --window 10

Utilise le snapshot local (docs/assets/gift_leaderboard/<date>/leaderboard.csv,
produit par fetch_gift_leaderboard.py) — les rangs cités restent donc
vérifiables même si le leaderboard en ligne bouge. Échelle des barres : CRPS
dans la fenêtre affichée (plus court = meilleur). Les deltas « prochain
barreau » donnent l'objectif chiffré immédiat.
"""

import argparse
import csv
import sys
from pathlib import Path

BAR_W = 34
HL = "\033[1;92m"      # highlight (vert gras)
DIM = "\033[2m"
RST = "\033[0m"

# Provenance CURATÉE (sous-chaîne, insensible à la casse -> organisation).
# Volontairement conservatrice : seuls les acteurs identifiables avec
# certitude sont étiquetés ; les soumissions communautaires/anonymes restent
# vides plutôt que devinées. Les baselines classiques (tft, n-beats,
# patchtst, itransformer...) sont courues par l'équipe du leaderboard.
PROVENANCE = [
    ("chronos", "Amazon"),
    ("toto", "Datadog"),
    ("moirai", "Salesforce"),
    ("timesfm", "Google"),
    ("ttm-", "IBM"),
    ("granite", "IBM"),
    ("flowstate", "IBM"),
    ("tirex", "NXAI"),
    ("sundial", "Tsinghua"),
    ("timer-", "Tsinghua"),
    ("lag-llama", "ServiceNow/Mila"),
    ("naive", "baseline GIFT"),
    ("tft", "baseline GIFT"),
    ("n-beats", "baseline GIFT"),
    ("nhits", "baseline GIFT"),
    ("n-hits", "baseline GIFT"),
    ("patchtst", "baseline GIFT"),
    ("itransformer", "baseline GIFT"),
    ("dlinear", "baseline GIFT"),
    ("deepar", "baseline GIFT"),
    ("autoarima", "baseline GIFT"),
    ("autoets", "baseline GIFT"),
    ("autotheta", "baseline GIFT"),
    ("crostonsba", "baseline GIFT"),
    ("visionts", "acad. (BJTU)"),
]


def load_meta(snapshot_dir: Path) -> dict:
    """models_meta.csv (org officielle des config.json de soumission + taille
    via l'API HF), produit par fetch_gift_leaderboard.py --enrich-only."""
    path = snapshot_dir / "models_meta.csv"
    meta = {}
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                meta[r["model"]] = r
    return meta


def org_of(model: str, meta: dict) -> str:
    m = meta.get(model, {})
    if m.get("org"):
        return m["org"]
    low = model.lower()
    for pat, org in PROVENANCE:
        if pat in low:
            return org
    return ""


def params_of(model: str, meta: dict) -> str:
    p = meta.get(model, {}).get("params_m", "")
    if not p:
        return ""
    v = float(p)
    return f"{v / 1000:.1f}B" if v >= 1000 else f"{v:.0f}M" if v >= 10 else f"{v:.1f}M"


def load_snapshot(snapshot: str | None):
    root = Path("docs/assets/gift_leaderboard")
    if snapshot:
        path = Path(snapshot)
    else:
        dates = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
        if not dates:
            sys.exit(f"✗ aucun snapshot sous {root} — lancer fetch_gift_leaderboard.py")
        path = dates[-1]
    csv_path = path / "leaderboard.csv"
    if not csv_path.exists():
        sys.exit(f"✗ {csv_path} introuvable")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({"model": r["model"], "crps": float(r["crps_ratio"]),
                         "mase": float(r["mase_ratio"])})
    return path.name, rows


def insertion_rank(rows, key, value):
    """Rang 1-indexé qu'obtiendrait `value` inséré dans le classement `key`."""
    return sum(1 for r in rows if r[key] < value) + 1


def ladder(rows, key, value, name, window, use_color, meta, my_org, my_params):
    hl, dim, rst = (HL, DIM, RST) if use_color else ("", "", "")
    ranked = sorted(rows, key=lambda r: r[key])
    rank = insertion_rank(rows, key, value)
    entries = ranked[max(0, rank - 1 - window):rank - 1] \
        + [{"model": name, key: value, "_me": True}] \
        + ranked[rank - 1:rank - 1 + window]

    vals = [e[key] for e in entries]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0

    print(f"\n  {key.upper()} — rang {rank}/{len(rows) + 1} "
          f"(bat {len(rows) - rank + 1} des {len(rows)} modèles du snapshot)")
    r = max(1, rank - window)
    for e in entries:
        me = e.get("_me", False)
        org = my_org if me else org_of(e["model"], meta)
        params = my_params if me else params_of(e["model"], meta)
        bar = "█" * max(1, round(BAR_W * (1 - (e[key] - lo) / span * 0.85)))
        line = (f"  {'→' if me else ' '} {r:>3d}. {e['model'][:28]:<28s} "
                f"{org[:18]:<18s} {params:>6s}  {e[key]:.4f}  {bar}")
        print(f"{hl}{line}{rst}" if me else f"{dim}{line}{rst}" if not me else line)
        r += 1

    above = [e for e in ranked if e[key] < value]
    if above:
        print(f"\n  Prochains barreaux ({key.upper()}) :")
        for e in above[-1:-4:-1]:
            org = org_of(e["model"], meta)
            params = params_of(e["model"], meta)
            tag = " · ".join(x for x in (org, params) if x)
            print(f"    {value - e[key]:+.4f}  pour passer {e['model'][:36]}"
                  f"{f' [{tag}]' if tag else ''} ({e[key]:.4f})")
    return rank


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--crps", type=float, required=True, help="CRPS ratio (vs SN officiel)")
    ap.add_argument("--mase", type=float, required=True, help="MASE ratio (vs SN officiel)")
    ap.add_argument("--name", default="TimeJEPA (ce checkpoint)")
    ap.add_argument("--affiliation", default="Y.Vincent")
    ap.add_argument("--params", default="1.1M", help="taille affichée pour notre ligne")
    ap.add_argument("--window", type=int, default=6, help="voisins affichés de chaque côté")
    ap.add_argument("--snapshot", default=None, help="dossier snapshot (défaut : le plus récent)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    date, rows = load_snapshot(args.snapshot)
    root = Path(args.snapshot) if args.snapshot else \
        Path("docs/assets/gift_leaderboard") / date
    meta = load_meta(root)
    use_color = not args.no_color and sys.stdout.isatty()

    print(f"\n{'=' * 84}")
    print(f"  GIFT-Eval — {args.name}   [snapshot {date}, {len(rows)} modèles classés]")
    print(f"{'=' * 84}")
    rc = ladder(rows, "crps", args.crps, args.name, args.window, use_color,
                meta, args.affiliation, args.params)
    rm = ladder(rows, "mase", args.mase, args.name, args.window, use_color,
                meta, args.affiliation, args.params)
    print(f"\n  Résumé : CRPS {args.crps:.4f} → {rc}e | MASE {args.mase:.4f} → {rm}e "
          f"| {len(rows) + 1} modèles avec celui-ci\n")


if __name__ == "__main__":
    main()
