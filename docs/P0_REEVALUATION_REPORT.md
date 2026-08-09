# P0.8 — Rapport de ré-évaluation

Tous les chiffres proviennent des **mêmes checkpoints** et des **mêmes fenêtres**.
Seul le protocole d'évaluation change.

- `legacy` : `skip_revin=True` — le protocole qui a produit `TimeJEPA_2ndbatch_results/`
- `fixed`  : `skip_revin=False` — RevIN actif, le régime dans lequel le modèle a été entraîné

> ⚠️ **ETTh1 / ETTh2** : `datasetsforecast.LongHorizon` ne livre qu'une seule série (`OT`)
> pour ces groupes, là où les tableaux publiés moyennent les 7 canaux ETT.
> Ces colonnes ne sont **pas** comparables à la littérature.


## 1. Impact du fix de normalisation

Variation relative de la MSE (négatif = le fix améliore) :

| dataset | h=96 | moyenne |
|---|---|---|
| electricity | -23.9% | -23.9% |
| etth1 ⚠️ | -62.8% | -62.8% |
| etth2 ⚠️ | -36.0% | -36.0% |
| ettm1 | -12.4% | -12.4% |
| ettm2 | -66.3% | -66.3% |
| exchange | -81.5% | -81.5% |
| traffic | -5.3% | -5.3% |
| weather | -48.8% | -48.8% |

**Effet global : MSE -42.1%, MASE -31.1% (moyenne sur tout).**

Le fix améliore 40/40 des couples (dataset, horizon).


## 2. TimeJEPA vs baselines (MASE, plus bas = mieux)

MASE = 1.0 signifie « aussi bon que seasonal naive ».

| dataset | TimeJEPA | SeasonalNaive | NaiveLast | ContextMean | LinearTrend | meilleur |
|---|---|---|---|---|---|---|
| electricity | 1.043 | 1.329 | 2.663 | 2.154 | 2.147 | **TimeJEPA** |
| etth1 ⚠️ | 1.081 | 1.259 | 1.085 | 1.297 | 1.532 | **TimeJEPA** |
| etth2 ⚠️ | 1.515 | 1.329 | 1.671 | 1.810 | 1.854 | SeasonalNaive |
| ettm1 | 1.309 | 0.998 | 1.597 | 1.385 | 1.438 | SeasonalNaive |
| ettm2 | 1.048 | 1.006 | 1.146 | 1.172 | 1.212 | SeasonalNaive |
| exchange | 9.082 | 7.821 | 7.821 | 19.266 | 13.641 | SeasonalNaive |
| traffic | 0.968 | 1.387 | 2.661 | 2.111 | 2.150 | **TimeJEPA** |
| weather | 0.793 | 0.976 | 0.934 | 1.261 | 1.220 | **TimeJEPA** |

**TimeJEPA est le meilleur sur 4/8 datasets.**

Il bat seasonal naive sur 4/8 datasets.


## 3. Dégradation par horizon (skill vs seasonal naive)

Positif = TimeJEPA gagne. Négatif = il perd.

| dataset | h=96 |
|---|---|
| electricity | +21.6% |
| etth1 ⚠️ | +13.9% |
| etth2 ⚠️ | -14.1% |
| ettm1 | -31.2% |
| ettm2 | -4.1% |
| exchange | -16.0% |
| traffic | +30.2% |
| weather | +18.7% |


## 4. Par checkpoint

| checkpoint | MASE legacy | MASE fixed | MASE seasonal-naive | skill | R² | WQL |
|---|---|---|---|---|---|---|
| timejepa_tiny/best-unfreeze-1-stride-48-full-datasets | 4.069 | **1.979** | 1.985 | +4.7% | 0.624 | 0.310 |
| timejepa_tiny/best-unfreeze-1-stride-48-restrained-datasets | 4.040 | **1.998** | 1.985 | +4.5% | 0.629 | 0.311 |
| timejepa_mini/best-reduced-datasets-late-unfreeze-1-stride-48 | 4.039 | **2.073** | 2.032 | +2.1% | 0.585 | 0.323 |
| timejepa_mini/best-reduced-datasets-early-unfreeze-1-stride-48 | 3.965 | **2.194** | 2.032 | +0.8% | 0.590 | 0.328 |
| timejepa_tiny/best-full-datasets-unfreeze-1-stride-48-512-196 | 6.114 | **2.280** | 2.032 | -0.3% | 0.600 | 0.331 |


## 5. Cadrage GIFT-Eval

GIFT-Eval normalise par seasonal naive (MASE = CRPS = 1.00 par construction).
Une MASE brute n'est donc **pas** comparable à sa leaderboard : ici `exchange`
a une MASE de ~9 avec m=1 sur h=96, ce qui écrase toute moyenne brute.
Les ratios ci-dessous sont la seule grandeur comparable.

| checkpoint | MASE / SN | CRPS(WQL) / SN |
|---|---|---|
| best-unfreeze-1-stride-48-full-datasets | 0.95 | 0.95 |
| best-unfreeze-1-stride-48-restrained-datasets | 0.95 | 0.95 |
| best-reduced-datasets-late-unfreeze-1-stride-48 | 0.98 | 0.98 |
| best-reduced-datasets-early-unfreeze-1-stride-48 | 0.99 | 0.99 |
| best-full-datasets-unfreeze-1-stride-48-512-196 | 1.00 | 1.00 |

**Positionnement (normalisé seasonal naive = 1.00) :**

| | MASE | CRPS/WQL |
|---|---|---|
| Seasonal Naive (référence) | 1.00 | 1.00 |
| **TimeJEPA (meilleur ckpt)** | **0.95** | **0.95** |
| Toto-2.0-4m (~4M params) | 0.76 | 0.52 |
| Toto-2.0-22m | 0.72 | 0.50 |
| Top-5 GIFT-Eval | 0.61–0.66 | 0.42–0.47 |

*Calculé sur les 8 datasets long-horizon Nixtla à h=96, pas sur les 97 configs
de GIFT-Eval. Ordre de grandeur indicatif, pas un classement.*

*Le WQL d'un modèle ponctuel égale sa ND par construction : c'est le score qu'il
obtiendrait sur GIFT-Eval sans tête probabiliste (cf. P2.1). L'écart entre la
colonne MASE et la colonne CRPS mesure exactement ce que coûte l'absence de tête
probabiliste.*