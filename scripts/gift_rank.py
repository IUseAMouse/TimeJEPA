# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round (CLI
# ladder over the 2026-08-24 leaderboard snapshot); kept per the no-delete policy.
"""
A checkpoint's position on the GIFT-Eval leaderboard - CLI visual.

    python scripts/gift_rank.py --crps 0.6134 --mase 0.8914
    python scripts/gift_rank.py --crps 0.6134 --mase 0.8914 --name "mix1ep3e4@25%" --window 10

Uses the local snapshot (docs/assets/gift_leaderboard/<date>/leaderboard.csv,
produced by fetch_gift_leaderboard.py) - cited ranks stay verifiable even if
the online leaderboard moves. Bar scale: CRPS within the shown window
(shorter = better). The "next rung" deltas give the immediate numeric target.
"""

import argparse
import csv
import sys
from pathlib import Path

BAR_W = 34
HL = "\033[1;92m"      # highlight (bold green)
DIM = "\033[2m"
RST = "\033[0m"

# CURATED provenance (substring, case-insensitive -> organization).
# Deliberately conservative: only actors identifiable with certainty are
# labeled; community/anonymous submissions stay empty rather than guessed.
# The classic baselines (tft, n-beats, patchtst, itransformer...) are run by
# the leaderboard team.
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
    ("naive", "GIFT baseline"),
    ("tft", "GIFT baseline"),
    ("n-beats", "GIFT baseline"),
    ("nhits", "GIFT baseline"),
    ("n-hits", "GIFT baseline"),
    ("patchtst", "GIFT baseline"),
    ("itransformer", "GIFT baseline"),
    ("dlinear", "GIFT baseline"),
    ("deepar", "GIFT baseline"),
    ("autoarima", "GIFT baseline"),
    ("autoets", "GIFT baseline"),
    ("autotheta", "GIFT baseline"),
    ("crostonsba", "GIFT baseline"),
    ("visionts", "acad. (BJTU)"),
]


def load_meta(snapshot_dir: Path) -> dict:
    """models_meta.csv (official org from the submission config.json + size
    via the HF API), produced by fetch_gift_leaderboard.py --enrich-only."""
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
            sys.exit(f"no snapshot under {root} - run fetch_gift_leaderboard.py")
        path = dates[-1]
    csv_path = path / "leaderboard.csv"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({"model": r["model"], "crps": float(r["crps_ratio"]),
                         "mase": float(r["mase_ratio"])})
    return path.name, rows


def insertion_rank(rows, key, value):
    """1-indexed rank that `value` would get inserted into the `key` ranking."""
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

    print(f"\n  {key.upper()} - rank {rank}/{len(rows) + 1} "
          f"(beats {len(rows) - rank + 1} of the snapshot's {len(rows)} models)")
    r = max(1, rank - window)
    for e in entries:
        me = e.get("_me", False)
        org = my_org if me else org_of(e["model"], meta)
        params = my_params if me else params_of(e["model"], meta)
        bar = "█" * max(1, round(BAR_W * (1 - (e[key] - lo) / span * 0.85)))
        line = (f"  {'>' if me else ' '} {r:>3d}. {e['model'][:28]:<28s} "
                f"{org[:18]:<18s} {params:>6s}  {e[key]:.4f}  {bar}")
        print(f"{hl}{line}{rst}" if me else f"{dim}{line}{rst}" if not me else line)
        r += 1

    above = [e for e in ranked if e[key] < value]
    if above:
        print(f"\n  Next rungs ({key.upper()}):")
        for e in above[-1:-4:-1]:
            org = org_of(e["model"], meta)
            params = params_of(e["model"], meta)
            tag = " - ".join(x for x in (org, params) if x)
            print(f"    {value - e[key]:+.4f}  to pass {e['model'][:36]}"
                  f"{f' [{tag}]' if tag else ''} ({e[key]:.4f})")
    return rank


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--crps", type=float, required=True, help="CRPS ratio (vs official SN)")
    ap.add_argument("--mase", type=float, required=True, help="MASE ratio (vs official SN)")
    ap.add_argument("--name", default="TimeJEPA (this checkpoint)")
    ap.add_argument("--affiliation", default="Y.Vincent")
    ap.add_argument("--params", default="1.1M", help="size displayed for our row")
    ap.add_argument("--window", type=int, default=6, help="neighbors shown on each side")
    ap.add_argument("--snapshot", default=None, help="snapshot directory (default: most recent)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    date, rows = load_snapshot(args.snapshot)
    root = Path(args.snapshot) if args.snapshot else \
        Path("docs/assets/gift_leaderboard") / date
    meta = load_meta(root)
    use_color = not args.no_color and sys.stdout.isatty()

    print(f"\n{'=' * 84}")
    print(f"  GIFT-Eval - {args.name}   [snapshot {date}, {len(rows)} models ranked]")
    print(f"{'=' * 84}")
    rc = ladder(rows, "crps", args.crps, args.name, args.window, use_color,
                meta, args.affiliation, args.params)
    rm = ladder(rows, "mase", args.mase, args.name, args.window, use_color,
                meta, args.affiliation, args.params)
    print(f"\n  Summary: CRPS {args.crps:.4f} -> rank {rc} | MASE {args.mase:.4f} -> rank {rm} "
          f"| {len(rows) + 1} models including this one\n")


if __name__ == "__main__":
    main()
