# ARCHIVED - not wired to live code, do not import (see scripts/archive/README.md).
"""
Build the P0.8 before/after report from re-evaluation JSONs.

Answers three questions, in order of importance:

  1. How much did the normalization fix change the numbers?      (legacy vs fixed)
  2. Does TimeJEPA actually beat a trivial baseline?             (MASE vs seasonal naive)
  3. Where does it break down?                                   (skill by horizon)

Usage:
    python scripts/report_reevaluation.py
    python scripts/report_reevaluation.py --input lightning/reevaluation --md report.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

UNIVARIATE_ONLY = {"etth1", "etth2"}
BASELINE_ORDER = ["seasonal_naive", "naive_last", "context_mean", "linear_trend"]


def load_rows(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "all_reevaluation.json":
            continue
        ckpt = path.stem.replace("__", "/")
        blob = json.loads(path.read_text())

        for ds, horizons in blob.items():
            for h, row in horizons.items():
                if "error" in row or "fixed" not in row:
                    continue
                lg, fx = row.get("legacy", {}), row["fixed"]
                bl = row.get("baselines", {})
                entry = {
                    "checkpoint": ckpt,
                    "dataset": ds,
                    "horizon": int(h),
                    "mse_legacy": lg.get("mse"),
                    "mse_fixed": fx.get("mse"),
                    "mae_legacy": lg.get("mae"),
                    "mae_fixed": fx.get("mae"),
                    "mase_legacy": lg.get("mase"),
                    "mase_fixed": fx.get("mase"),
                    "r2_fixed": fx.get("r2"),
                    "wql_fixed": fx.get("wql"),
                    "skill_vs_sn": fx.get("skill_vs_seasonal_naive"),
                }
                for b in BASELINE_ORDER:
                    entry[f"mase_{b}"] = bl.get(b, {}).get("mase")
                    entry[f"wql_{b}"] = bl.get(b, {}).get("wql")
                rows.append(entry)

    return pd.DataFrame(rows)


def fmt_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def build_report(df: pd.DataFrame) -> str:
    out: List[str] = []
    A = out.append

    A("# P0.8 - Re-evaluation report\n")
    A("All numbers come from the **same checkpoints** and the **same windows**.")
    A("Only the evaluation protocol changes.\n")
    A("- `legacy`: `skip_revin=True` - the protocol that produced `TimeJEPA_2ndbatch_results/`")
    A("- `fixed` : `skip_revin=False` - RevIN active, the regime the model was trained in\n")
    A("> Note: **ETTh1 / ETTh2**: `datasetsforecast.LongHorizon` only ships one series (`OT`)")
    A("> for these groups, where published tables average the 7 ETT channels.")
    A("> These columns are **not** comparable to the literature.\n")

    # ---- 1. Impact of the fix -------------------------------------------------
    A("\n## 1. Impact of the normalization fix\n")
    d = df.dropna(subset=["mse_legacy", "mse_fixed"]).copy()
    if not d.empty:
        d["mse_delta"] = (d["mse_fixed"] - d["mse_legacy"]) / d["mse_legacy"]
        d["mase_delta"] = (d["mase_fixed"] - d["mase_legacy"]) / d["mase_legacy"]

        piv = d.pivot_table(index="dataset", columns="horizon", values="mse_delta", aggfunc="mean")
        A("Relative MSE change (negative = the fix improves):\n")
        A("| dataset | " + " | ".join(f"h={c}" for c in piv.columns) + " | mean |")
        A("|---|" + "---|" * (len(piv.columns) + 1))
        for ds, r in piv.iterrows():
            flag = " (!)" if ds in UNIVARIATE_ONLY else ""
            A(f"| {ds}{flag} | " + " | ".join(fmt_pct(v) for v in r) + f" | {fmt_pct(r.mean())} |")

        A(f"\n**Overall effect: MSE {fmt_pct(d['mse_delta'].mean())}, "
          f"MASE {fmt_pct(d['mase_delta'].mean())} (mean over everything).**")
        n_better = int((d["mse_delta"] < 0).sum())
        A(f"\nThe fix improves {n_better}/{len(d)} of the (dataset, horizon) pairs.")

    # ---- 2. vs baselines ------------------------------------------------------
    A("\n\n## 2. TimeJEPA vs baselines (MASE, lower = better)\n")
    A("MASE = 1.0 means \"as good as seasonal naive\".\n")
    cols = ["mase_fixed"] + [f"mase_{b}" for b in BASELINE_ORDER]
    g = df.groupby("dataset")[cols].mean()
    g.columns = ["TimeJEPA", "SeasonalNaive", "NaiveLast", "ContextMean", "LinearTrend"]
    g["winner"] = g.idxmin(axis=1)

    A("| dataset | TimeJEPA | SeasonalNaive | NaiveLast | ContextMean | LinearTrend | best |")
    A("|---|---|---|---|---|---|---|")
    for ds, r in g.iterrows():
        flag = " (!)" if ds in UNIVARIATE_ONLY else ""
        cells = " | ".join(
            "n/a" if pd.isna(r[c]) else f"{r[c]:.3f}"
            for c in ["TimeJEPA", "SeasonalNaive", "NaiveLast", "ContextMean", "LinearTrend"]
        )
        mark = "**TimeJEPA**" if r["winner"] == "TimeJEPA" else r["winner"]
        A(f"| {ds}{flag} | {cells} | {mark} |")

    wins = int((g["winner"] == "TimeJEPA").sum())
    A(f"\n**TimeJEPA is the best on {wins}/{len(g)} datasets.**")

    beats_sn = int((g["TimeJEPA"] < g["SeasonalNaive"]).sum())
    A(f"\nIt beats seasonal naive on {beats_sn}/{len(g)} datasets.")

    # ---- 3. Horizon degradation ----------------------------------------------
    A("\n\n## 3. Degradation per horizon (skill vs seasonal naive)\n")
    A("Positive = TimeJEPA wins. Negative = it loses.\n")
    piv = df.pivot_table(index="dataset", columns="horizon", values="skill_vs_sn", aggfunc="mean")
    A("| dataset | " + " | ".join(f"h={c}" for c in piv.columns) + " |")
    A("|---|" + "---|" * len(piv.columns))
    for ds, r in piv.iterrows():
        flag = " (!)" if ds in UNIVARIATE_ONLY else ""
        A(f"| {ds}{flag} | " + " | ".join(fmt_pct(v) for v in r) + " |")

    # ---- 4. Per checkpoint ----------------------------------------------------
    A("\n\n## 4. Per checkpoint\n")
    c = df.groupby("checkpoint").agg(
        mase_legacy=("mase_legacy", "mean"),
        mase_fixed=("mase_fixed", "mean"),
        mase_sn=("mase_seasonal_naive", "mean"),
        skill=("skill_vs_sn", "mean"),
        r2=("r2_fixed", "mean"),
        wql=("wql_fixed", "mean"),
    ).sort_values("mase_fixed")
    A("| checkpoint | MASE legacy | MASE fixed | MASE seasonal-naive | skill | R2 | WQL |")
    A("|---|---|---|---|---|---|---|")
    for name, r in c.iterrows():
        A(f"| {name} | {r.mase_legacy:.3f} | **{r.mase_fixed:.3f}** | {r.mase_sn:.3f} | "
          f"{fmt_pct(r.skill)} | {r.r2:.3f} | {r.wql:.3f} |")

    # ---- 5. GIFT-Eval framing -------------------------------------------------
    A("\n\n## 5. GIFT-Eval framing\n")
    A("GIFT-Eval normalizes by seasonal naive (MASE = CRPS = 1.00 by construction).")
    A("A raw MASE is therefore **not** comparable to its leaderboard: here `exchange`")
    A("has a MASE of ~9 with m=1 at h=96, which crushes any raw mean.")
    A("The ratios below are the only comparable quantity.\n")

    df = df.copy()
    df["mase_ratio"] = df["mase_fixed"] / df["mase_seasonal_naive"]
    df["wql_ratio"] = df["wql_fixed"] / df["wql_seasonal_naive"]

    per_ckpt = df.groupby("checkpoint")[["mase_ratio", "wql_ratio"]].mean().sort_values("mase_ratio")
    A("| checkpoint | MASE / SN | CRPS(WQL) / SN |")
    A("|---|---|---|")
    for name, r in per_ckpt.iterrows():
        A(f"| {name.split('/')[-1]} | {r.mase_ratio:.2f} | {r.wql_ratio:.2f} |")

    best_ratio = per_ckpt["mase_ratio"].min()
    best_wql_ratio = per_ckpt["wql_ratio"].min()

    A("\n**Positioning (normalized seasonal naive = 1.00):**\n")
    A("| | MASE | CRPS/WQL |")
    A("|---|---|---|")
    A("| Seasonal Naive (reference) | 1.00 | 1.00 |")
    A(f"| **TimeJEPA (best ckpt)** | **{best_ratio:.2f}** | **{best_wql_ratio:.2f}** |")
    A("| Toto-2.0-4m (~4M params) | 0.76 | 0.52 |")
    A("| Toto-2.0-22m | 0.72 | 0.50 |")
    A("| Top-5 GIFT-Eval | 0.61-0.66 | 0.42-0.47 |")
    A("\n*Computed on the 8 Nixtla long-horizon datasets at h=96, not on the 97")
    A("GIFT-Eval configs. Indicative order of magnitude, not a ranking.*")
    A("\n*A point model's WQL equals its ND by construction: the score it would")
    A("get on GIFT-Eval without a probabilistic head (cf. P2.1). The gap between")
    A("the MASE column and the CRPS column measures exactly what the missing")
    A("probabilistic head costs.*")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="lightning/reevaluation")
    ap.add_argument("--md", default="lightning/reevaluation/REPORT.md")
    ap.add_argument("--csv", default="lightning/reevaluation/reevaluation_long.csv")
    args = ap.parse_args()

    input_dir = REPO_ROOT / args.input
    df = load_rows(input_dir)
    if df.empty:
        print(f"No results found in {input_dir}")
        return

    out_csv = REPO_ROOT / args.csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    report = build_report(df)
    out_md = REPO_ROOT / args.md
    out_md.write_text(report)

    print(report)
    print(f"\n\n-> {out_md}\n-> {out_csv}")


if __name__ == "__main__":
    main()
