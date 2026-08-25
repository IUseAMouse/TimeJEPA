#!/usr/bin/env python3
"""Statistique appariée du signal d'horizon JEPA vs recon (arms G6, E15/E16).

Deux sources, mêmes checkpoints (epoch04 des deux arms) :
  A) horizon_metrics.json — MAE par PAS (0..255) sur 8 datasets Monash locaux :
     pente du gap relatif recon/JEPA en fonction de la profondeur.
  B) nixtla_results_long.csv — MASE par (dataset x horizon 96/192/336/720) :
     les 28 cellules d'E15, gap par horizon + test de tendance par permutation.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "evaluation"
JEPA = ROOT / "timejepa_lotsa_tiny_zs/epoch04_valloss1.3454"
RECON = ROOT / "timejepa_lotsa_tiny_zs_recon/epoch04_valloss1.3507"
rng = np.random.default_rng(0)

# ---------- A) per-step (Monash local, 8 datasets, 256 pas) ----------
j = json.load(open(JEPA / "horizon_metrics.json"))
r = json.load(open(RECON / "horizon_metrics.json"))
print("A) Gap relatif recon vs JEPA par profondeur (MAE, Monash local)")
slopes = []
for ds in j:
    mj = np.array([j[ds][str(t)]["mae"] for t in range(256)])
    mr = np.array([r[ds][str(t)]["mae"] for t in range(256)])
    gap = (mr - mj) / mj                      # >0 = recon pire
    # pente du gap en %/100 pas (régression linéaire sur t)
    t = np.arange(256)
    slope = np.polyfit(t, gap, 1)[0] * 100 * 100
    slopes.append(slope)
    q = lambda a, b: gap[a:b].mean() * 100
    print(f"  {ds:34s} gap pas0-63 {q(0,64):+6.2f}%  pas192-255 {q(192,256):+6.2f}%  pente {slope:+6.2f}%/100pas")
slopes = np.array(slopes)
# bootstrap sur les datasets (n=8) de la pente moyenne
bs = np.array([rng.choice(slopes, len(slopes)).mean() for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  => pente moyenne {slopes.mean():+.2f}%/100pas, IC95% bootstrap [{lo:+.2f}, {hi:+.2f}], "
      f"datasets pente>0 : {(slopes>0).sum()}/8")

# ---------- B) 28 cellules Nixtla (MASE) ----------
def cells(path):
    out = {}
    for row in csv.DictReader(open(path / "nixtla_results_long.csv")):
        out[(row["Dataset"], int(row["Horizon"]))] = float(row["MASE"])
    return out

cj, cr = cells(JEPA), cells(RECON)
keys = sorted(set(cj) & set(cr))
hs = sorted({h for _, h in keys})
print("\nB) 28 cellules Nixtla (MASE), gap relatif recon vs JEPA par horizon")
gap_by_h = {}
for h in hs:
    g = np.array([(cr[k] - cj[k]) / cj[k] for k in keys if k[1] == h])
    gap_by_h[h] = g
    bs = np.array([rng.choice(g, len(g)).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  h={h:3d} n={len(g)} gap moyen {g.mean()*100:+6.2f}%  IC95% [{lo*100:+.2f}, {hi*100:+.2f}]  "
          f"cellules recon-pire {(g>0).sum()}/{len(g)}")

# test de tendance : Spearman(gap, horizon) observé vs permutations des horizons
# INTRA-dataset (structure appariée respectée)
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
print(f"\n  Tendance gap~horizon : Spearman obs {obs_rho:+.3f}, p permutation (unilat.) = {p:.4f}")

# et en retirant la cellule dominante etth1 (fragilité notée en E15)
mask = [d != "etth1" for d in datasets for _ in hs]
obs_rho2 = spear(obs_gaps[mask], obs_hs[mask])
null2 = []
nd = sum(1 for d in datasets if d != "etth1")
for _ in range(20000):
    perm_h = np.concatenate([rng.permutation(hs_arr) for _ in range(nd)])
    null2.append(spear(obs_gaps[mask], perm_h))
p2 = float((np.array(null2) >= obs_rho2).mean())
print(f"  Sans etth1            : Spearman obs {obs_rho2:+.3f}, p = {p2:.4f}")
