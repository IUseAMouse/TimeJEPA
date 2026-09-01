# DEPRECATED (2026-09-01 audit) - one-shot script from a closed round; kept
# per the no-delete policy. Reads nixtla_results_long.csv from the archived
# P0 round.
"""Paired statistics of the JEPA vs recon horizon signal (G6 arms, E15/E16).

Two sources, same checkpoints (epoch04 of both arms):
  A) horizon_metrics.json - MAE per STEP (0..255) on 8 local Monash datasets:
     slope of the relative recon/JEPA gap as a function of depth.
  B) nixtla_results_long.csv - MASE per (dataset x horizon 96/192/336/720):
     the 28 E15 cells, gap per horizon + permutation trend test.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "evaluation"
JEPA = ROOT / "timejepa_lotsa_tiny_zs/epoch04_valloss1.3454"
RECON = ROOT / "timejepa_lotsa_tiny_zs_recon/epoch04_valloss1.3507"
rng = np.random.default_rng(0)

# ---------- A) per-step (local Monash, 8 datasets, 256 steps) ----------
j = json.load(open(JEPA / "horizon_metrics.json"))
r = json.load(open(RECON / "horizon_metrics.json"))
print("A) Relative recon vs JEPA gap by depth (MAE, local Monash)")
slopes = []
for ds in j:
    mj = np.array([j[ds][str(t)]["mae"] for t in range(256)])
    mr = np.array([r[ds][str(t)]["mae"] for t in range(256)])
    gap = (mr - mj) / mj                      # >0 = recon worse
    # gap slope in %/100 steps (linear regression on t)
    t = np.arange(256)
    slope = np.polyfit(t, gap, 1)[0] * 100 * 100
    slopes.append(slope)
    q = lambda a, b: gap[a:b].mean() * 100
    print(f"  {ds:34s} gap steps0-63 {q(0,64):+6.2f}%  steps192-255 {q(192,256):+6.2f}%  slope {slope:+6.2f}%/100steps")
slopes = np.array(slopes)
# bootstrap over the datasets (n=8) of the mean slope
bs = np.array([rng.choice(slopes, len(slopes)).mean() for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  => mean slope {slopes.mean():+.2f}%/100steps, bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}], "
      f"datasets slope>0: {(slopes>0).sum()}/8")

# ---------- B) 28 Nixtla cells (MASE) ----------
def cells(path):
    out = {}
    for row in csv.DictReader(open(path / "nixtla_results_long.csv")):
        out[(row["Dataset"], int(row["Horizon"]))] = float(row["MASE"])
    return out

cj, cr = cells(JEPA), cells(RECON)
keys = sorted(set(cj) & set(cr))
hs = sorted({h for _, h in keys})
print("\nB) 28 Nixtla cells (MASE), relative recon vs JEPA gap per horizon")
gap_by_h = {}
for h in hs:
    g = np.array([(cr[k] - cj[k]) / cj[k] for k in keys if k[1] == h])
    gap_by_h[h] = g
    bs = np.array([rng.choice(g, len(g)).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  h={h:3d} n={len(g)} mean gap {g.mean()*100:+6.2f}%  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  "
          f"recon-worse cells {(g>0).sum()}/{len(g)}")

# trend test: observed Spearman(gap, horizon) vs INTRA-dataset horizon
# permutations (paired structure respected)
datasets = sorted({d for d, _ in keys})
obs_gaps = np.array([(cr[(d, h)] - cj[(d, h)]) / cj[(d, h)] for d in datasets for h in hs])
obs_hs = np.array([h for _ in datasets for h in hs], dtype=float)

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))

obs_rho = spear(obs_gaps, obs_hs)
null = []
hs_arr = np.array(hs, dtype=float)
for _ in range(20000):
    perm_h = np.concatenate([rng.permutation(hs_arr) for _ in datasets])
    null.append(spear(obs_gaps, perm_h))
null = np.array(null)
p = float((null >= obs_rho).mean())
print(f"\n  gap~horizon trend: observed Spearman {obs_rho:+.3f}, one-sided permutation p = {p:.4f}")

# and with the dominant etth1 cell removed (fragility noted in E15)
mask = [d != "etth1" for d in datasets for _ in hs]
obs_rho2 = spear(obs_gaps[mask], obs_hs[mask])
null2 = []
nd = sum(1 for d in datasets if d != "etth1")
for _ in range(20000):
    perm_h = np.concatenate([rng.permutation(hs_arr) for _ in range(nd)])
    null2.append(spear(obs_gaps[mask], perm_h))
p2 = float((np.array(null2) >= obs_rho2).mean())
print(f"  Without etth1         : observed Spearman {obs_rho2:+.3f}, p = {p2:.4f}")
