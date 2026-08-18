# TimeJEPA — Roadmap vers un niveau SOTA

> Branche de travail : `sota-roadmap` (master reste intact).
> **Règle absolue : aucune suppression de fichier. Lecture / écriture / modification uniquement.**
> Ce fichier est le point de reprise si la session est coupée. Mettre à jour les cases à cocher au fur et à mesure.

**Dernière mise à jour :** 2026-08-18 — E16 : GIFT-Eval 0.979 MASE / 0.677 CRPS à ~1M params en zero-shot ; signal d'horizon G6 répliqué. G7 : courbe d'échelle sur corpus complet.

---

## 0. Contexte du diagnostic (à lire en premier si reprise à froid)

Audit du repo à `6e581b3` (master). Constat : l'ingénierie est solide, mais **les résultats
d'évaluation actuels sont invalidés par des bugs de plomberie**, pas par la qualité du modèle.
Les chiffres de `../TimeJEPA_2ndbatch_results/` ne mesurent pas ce qu'ils prétendent mesurer.

### Bugs identifiés (référence)

| # | Sévérité | Fichier:ligne (à HEAD master) | Problème |
|---|---|---|---|
| B1 | 🔴 | `models/__init__.py:11`, `models/encoders/__init__.py:6`, `encoders/target_encoder.py:18` | `patchtst_encoder` supprimé mais toujours importé → `import timejepa.models` crashe, 0 test collecté |
| B2 | 🔴 | `scripts/evaluate.py:590` | `skip_revin=True` sur Nixtla : le modèle n'a JAMAIS vu de données non instance-normalisées. Confusion z-score global ≠ RevIN par fenêtre. Décrochage de niveau visible sur les plots |
| B3 | 🟠 | `models/jepa_tst.py:73,195` + `training/finetune_module.py:207,237` | RevIN `affine=True` appliqué au contexte mais pas à la cible → biais systématique après `_denormalize` |
| B4 | 🟠 | — (vérifié empiriquement) | `datasetsforecast.LongHorizon` ne livre qu'UNE série (`OT`) pour ETTh1/ETTh2 → incomparable aux tableaux publiés (qui moyennent 7 canaux) |
| B5 | 🟠 | `scripts/train.py:96-115` | `augmentation_config` jamais passé au DataModule → toutes les augmentations des YAML sont mortes |
| B6 | 🟠 | `training/utils/metrics.py:34-63` | VICReg : variance sur `reshape(-1, D)` (mélange batch×position) ; termes `tgt` détachés donc sans gradient ; `context_embeddings` jamais régularisés |
| B7 | 🟠 | `configs/model/{base,large}.yaml` | `variance/covariance_loss_weight: 0.0` + clé `invariance_loss_weight` absente → VICReg dégénéré en MSE pure (et `train.py:134` crasherait) |
| B8 | 🟠 | `training/jepa_pretrain_module.py:175` | `validation_step` n'passe pas `vicreg_weights` → défauts (25/25/1) ≠ objectif d'entraînement. Early-stopping / `save_top_k` sur un autre objectif |
| B9 | 🟠 | `models/jepa_tst.py:169-226` | Target encoder voit la fenêtre cible ISOLÉE (11-15 patchs vs 47-63) → mismatch de distribution. I-JEPA encode la fenêtre complète puis slice |
| B10 | 🔴 | `models/jepa_tst.py:331,350` | `revin.freeze()` n'existe pas → `AttributeError` sur tout rollout. Et mélange espace brut / espace normalisé dans la boucle |
| B11 | 🟡 | `training/utils/masking.py` | 327 lignes, zéro import. Code mort (le pretrain est du forecasting-JEPA, pas du masked-JEPA) |
| B12 | 🟡 | `models/decoders/linear_decoder.py:311` | `output_norm = LayerNorm(num_features)` jamais utilisé (et sortirait des zéros avec `num_features=1`) |
| B13 | 🟡 | `models/jepa_tst.py:145` | `ForecastingHead` créé sans `stride` → défaut 8. Avec `base.yaml` (patch 4) → `counts=0` → NaN. Masqué car `train.py`/`evaluate.py` remplacent le décodeur |
| B14 | 🟡 | `models/components/patching.py:229` | `UnPatching` overlap-add en boucle Python ; bords `count=1` vs `count=2` → artefacts de couture (bruit HF période=stride visible sur les plots) |
| B15 | 🟡 | `data/datamodule.py:358-372` | Split par plages contiguës sur un ordre série-majeur → split PAR SÉRIE, pas temporel. Défendable, mais `val_loss` ne prédit pas la perf benchmark |

### Cibles chiffrées (GIFT-Eval, août 2026)

| | Avg Rank | MASE | CRPS |
|---|---|---|---|
| Seasonal Naive | 104.2 | 1.00 | 1.00 |
| **Toto-2.0-4m (~4M) ← cible `tiny`** | 68.2 | 0.76 | 0.52 |
| Toto-2.0-22m | 53.6 | 0.72 | 0.50 |
| Top-5 (STRIDE / EXAONE / LS-Agent) | 13–19 | 0.61–0.66 | 0.42–0.45 |

⚠️ La métrique principale de GIFT-Eval est **probabiliste (CRPS/WQL)**. Un forecaster ponctuel
est structurellement non-compétitif → la tête quantile (P2) est un prérequis, pas un confort.

### Baseline honnête actuelle (R² h96, run `tiny/final-mlp`, protocole INVALIDE mais indicatif)

`exchange 0.77 · electricity 0.74 · ettm2 0.71 · weather 0.62 · traffic 0.61 · etth2 0.43 · ettm1 0.17 · etth1 0.13`
→ s'effondre à ≤0 sur ettm1/etth1 à h=720. **Aucun baseline dans le pipeline → ces chiffres ne sont pas interprétables.**

---

## P0 — Débloquer et re-mesurer
*Aucun changement de modèle. Objectif : savoir enfin où on en est.*

- [x] **P0.1** Réparer les imports `patchtst_encoder` (B1) → repo importable
      - `models/__init__.py`, `encoders/__init__.py` exportent `BareTransformerEncoder`
      - `target_encoder.py` : `DualEncoderWrapper` construit un `BareTransformerEncoder` (classe morte, marquée deprecated, **non supprimée**)
- [x] **P0.2** Réparer la suite de tests → **26 passed, 7 skipped, 0 failed**
      - imports `src.timejepa.*` → `timejepa.*`
      - tests visant l'API disparue (masked-JEPA, `create_jepa_tst_*`, `MultiHeadAttention`,
        signatures de callbacks) marqués `pytest.mark.skip` avec motif explicite — **fichiers conservés**
      - nouveau `tests/test_p0_regressions.py` : 20+ tests qui verrouillent tous les fixes P0
- [x] **P0.3** `skip_revin=False` par défaut (B2) + cohérence affine RevIN (B3)
      - `RevIN.denormalize_target_space()` : inverse cohérent avec l'espace de la loss
      - `RevIN.to_input_frame()` : réaligne le forecast avant ré-injection en rollout
      - `RevIN.freeze()/unfreeze()` (**B10** — n'existait pas, tout rollout crashait)
      - `JEPATST.forecast()` réécrit : rollout entièrement dans le frame normalisé
      - `ForecastingHead` utilise `denormalize_target_space`
      - Mesure de la dérive affine réelle dans les ckpts : w ∈ [0.86, 1.10], b ≤ 0.089
        → ~6-10 % d'erreur d'échelle. Réel mais **second ordre** vs B2
- [x] **P0.4** Baselines : `src/timejepa/training/utils/baselines.py`
      (seasonal-naive, naive-last, context-mean, linear-trend + table de saisonnalité)
- [x] **P0.5** Métriques scale-free dans `metrics.py` : MASE, ND, pinball, WQL
      (identité vérifiée : WQL == ND pour un forecast ponctuel)
- [x] **P0.6** Caveat ETTh1/ETTh2 univarié (B4) émis à l'éval + dans les rapports
- [x] **P0.7** Ré-évaluation des checkpoints (`scripts/reevaluate_checkpoints.py`) —
      5 checkpoints × 8 datasets × h=96, modes legacy et fixed sur les **mêmes fenêtres**.
      Mode legacy reproduit les anciens chiffres à la 3e décimale ⇒ harness fidèle.
      Bug trouvé et corrigé en route : MASE explosait (~1e4) sur les fenêtres plates
      (ETTm2, electricity) → agrégation poolée. Vérif : seasonal-naive donne
      MASE 0.99–1.00 sur ETTm1, la valeur théorique.
      *Reste à faire (optionnel) : h=192/336/720 — le script reprend sur incrément.*
- [x] **P0.8** Rapport : `lightning/reevaluation/REPORT.md` (+ `reevaluation_long.csv`)

### 🔑 Résultats P0

**1. Le fix de normalisation améliore 40/40 des couples mesurés — MSE −42 % en moyenne.**

| dataset | Δ MSE | | dataset | Δ MSE |
|---|---|---|---|---|
| exchange | −81.5 % | | etth2 | −36.0 % |
| ettm2 | −66.3 % | | electricity | −23.9 % |
| etth1 | −62.8 % | | ettm1 | −12.4 % |
| weather | −48.8 % | | traffic | −5.3 % |

**2. TimeJEPA bat seasonal naive sur 4/8 datasets** (h=96, skill = 1 − MASE/MASE_SN) :

| gagne | | perd | |
|---|---|---|---|
| traffic | **+30.2 %** | ettm1 | −31.2 % |
| electricity | **+21.6 %** | exchange | −16.0 % |
| weather | **+18.7 %** | etth2 ⚠️ | −14.1 % |
| etth1 ⚠️ | **+13.9 %** | ettm2 | −4.1 % |

**3. Meilleur checkpoint : `timejepa_tiny/best-unfreeze-1-stride-48-full-datasets`**
(ctx 384, h 96, decoder mlp) — MASE/SN = **0.95**, R² 0.62, WQL/SN 0.95.

**4. Positionnement (normalisé seasonal naive = 1.00) :**

| | MASE | CRPS/WQL |
|---|---|---|
| Seasonal Naive | 1.00 | 1.00 |
| **TimeJEPA tiny (1.6M)** | **0.95** | **0.95** |
| Toto-2.0-4m | 0.76 | 0.52 |
| Top-5 GIFT-Eval | 0.61–0.66 | 0.42–0.47 |

**Lecture.** Le modèle est un vrai forecaster sur les séries à forte saisonnalité et bon
SNR (traffic, electricity, weather), et perd sur le bruité/non-stationnaire
(ettm1, exchange, etth2) — ce qui **valide l'intuition initiale**, mais l'ancien
protocole la faisait paraître bien pire qu'elle n'est.
L'écart MASE 0.95 vs CRPS 0.52 (Toto) est presque entièrement imputable à l'absence
de tête probabiliste : c'est **le** levier P2.1.

**Critère de sortie P0 :** un tableau où chaque chiffre est comparable à un baseline, et où l'on sait
si TimeJEPA bat seasonal-naive, et de combien.

> 🛑 **POINT D'ARRÊT OBLIGATOIRE.** Ne PAS enchaîner sur P1 automatiquement.
> À la fin de P0 : produire un point d'update (avant/après, vs baselines, ce que ça change au
> diagnostic) et attendre l'arbitrage utilisateur avant de démarrer P1/P2.

---

### 🔬 Diagnostic ETTm — pourquoi le modèle échoue (`scripts/diagnose_ettm.py`)

Sortie brute : `docs/P1_frequency_diagnostic.txt`. Checkpoint : `tiny/best-unfreeze-1-stride-48-full-datasets`.

**Test 1 — contexte fixe (384), période variable :**

| cas | cycle | positions/patch | skill |
|---|---|---|---|
| ECL natif | 24 | 3 | **+28.5 %** |
| ETTm1 ×4 down | 24 | 3 | −2.6 % |
| ETTm1 ×2 down | 48 | 6 | −8.3 % |
| ETTm1 natif | 96 | 12 | −27.2 % |
| **ECL interpolé ×4** | **96** | **12** | **−136.3 %** |

Contrôle décisif : *mêmes données* ECL, simplement interpolées ×4 → +28.5 % devient −136 %.
L'interpolation **lisse** pourtant le signal (MASE de seasonal-naive 1.32 → 1.00), donc ce n'est
pas la difficulté du signal : c'est la **période exprimée en positions de patch**.

Direction contre-intuitive : le modèle marche quand le cycle occupe **peu** de positions (3),
pas beaucoup. À 3 positions/cycle le motif se répète 16× dans les 47 patchs ; à 12 positions
il ne se répète que 4× et demande une portée que le prédicteur (2 couches) n'a pas.

Explique `weather` (cycle 144 = 18 positions) qui gagne quand même : son horizon de 96 pas
fait **0.67 cycle**, il n'extrapole jamais un cycle complet.

> **Règle unifiée.** Succès si (a) cycle court en positions de patch et très répété, ou
> (b) horizon < 1 cycle. Échec dès qu'il faut **extrapoler un cycle de longue période**.

**Test 2 — période fixe, contexte variable (electricity, cycle=24) :**

| ctx | cycles dans ctx | skill |
|---|---|---|
| 96 | 4 | −51.3 % |
| 192 | 8 | +11.3 % |
| **384** | **16** | **+28.5 %** |
| 768 | 32 | **−103.8 %** |

Pic exactement sur **la longueur d'entraînement**. ETTm1 empire aussi avec plus de contexte
(−27 → −47 → −55 % à ctx 384/768/1536), alors que « plus de cycles » devrait aider.
⇒ **Le modèle a mémorisé une géométrie d'entrée fixe (47 patchs) et s'effondre des deux côtés.**

*Caveat non résolu :* à ctx=768 RevIN normalise sur une fenêtre plus longue et absorbe plus de
non-stationnarité. Une part de la dégradation peut venir de là, pas seulement de la géométrie.

**Conséquences actées :**
1. Pas de temporal-resolution encoding en dur (contraire au but du projet, et le problème n'est
   pas là).
2. **P1.8 (contexte variable) devient prioritaire** — c'est le fix direct du test 2.
3. **P1.7 (horizon aléatoire)** — apprendre à extrapoler des cycles de longueurs variées.
4. **La DRS actuelle est inopérante pour ce problème** : `augmentations.py:173` sous-échantillonne
   *puis ré-interpole à la même longueur* → simule un capteur moins précis, laisse le ratio
   période/patch **inchangé**. À réécrire : prélever une fenêtre brute plus longue et la décimer,
   ce qui change réellement le nombre de cycles dans la fenêtre. → **P1.12**

---

## P1 — Rendre le pretrain sain

- [x] **P1.1** SIGReg (LeJEPA, arXiv 2511.08544) — Epps-Pulley sur M projections 1D,
      quadrature trapézoïdale. `loss_type: 'sigreg'`, hyperparamètre unique `lambda`.
      **Complémentaire de VICReg, pas redondant** (mesuré sur embeddings synthétiques) :
      | distribution | SIGReg | VICReg variance |
      |---|---|---|
      | N(0,1) isotrope | 0.0006 | ~0 |
      | collapsed | 1.282 | max |
      | **bimodale** | **0.518** | **~0 (aveugle)** |
      | **queues lourdes** | **0.168** | **~0 (aveugle)** |
      | anisotrope 1 dim | 0.069 (faible) | covariance la capte |
- [x] **P1.2** VICReg réparé (B6) : variance/covariance à position FIXÉE (démonstration :
      sur un tenseur collapsé par position, pénalité poolée **0.0000** vs per-position **0.9990**) ;
      termes `targets` détachés retirés du gradient ; régularisation appliquée aux
      `context_embeddings`
- [x] **P1.3** Corriger `base.yaml` / `large.yaml` (B7) : clé `invariance_loss_weight` ajoutée
      (son absence faisait crasher `train.py:134`) + variance/covariance 0.0 → 15.0/1.0.
      *Fait en avance parce que c'était un crash certain, pas une optimisation.*
      `large.yaml` reçoit aussi les blocs `nixtla:` / `horizons:` qui lui manquaient.
      ⚠️ Reste : `tiny`/`mini` listent `ecl` dans `nixtla:`, absent de `NIXTLA_REGISTRY`
      (le dataset s'appelle `electricity`) → warning + skip à l'éval. Bénin.
- [x] **P1.4** `val_loss` et `train_loss` partagent `_compute_loss` (B8)
- [x] **P1.5** Target encoder I-JEPA (B9) : encode `[ctx ‖ tgt]` puis slice. Alignement des
      patchs vérifié exactement (les 11 derniers patchs de la fenêtre 480 démarrent à 384..464,
      identiques aux patchs cible autonomes)
- [x] **P1.6** `augmentation_config` câblé dans `train.py` (B5) — inerte sur TOUS les runs précédents
- [x] **P1.7** Horizon aléatoire, échantillonné une fois par batch
- [x] **P1.8** Contexte variable, échantillonné une fois par batch (tenseurs rectangulaires,
      pas de masque nécessaire : l'encodeur est agnostique à la longueur grâce à RoPE)
- [x] **P1.9** `collapse/context_std` + `collapse/effective_rank` loggés en première classe
- [ ] **P1.10** Pondération des datasets par prédictibilité ⚠️ **NON FAIT** — bitcoin et
      wikipedia poussent l'encodeur vers la moyenne conditionnelle, donc vers le collapse.
      Premier suspect si `collapse/context_std` dérive pendant le run. (bitcoin/wikipedia ne doivent pas dominer
      le gradient d'un objectif prédictif)
- [ ] **P1.11** Pretrain comparatif VICReg vs SIGReg (ablation), avec/sans EMA pour SIGReg
- [x] **P1.12** Vraie multi-résolution dans `TimeSeriesDataset.get_item` (décimation d'une
      fenêtre brute plus longue). Vérifié : facteur 1 → 4 cycles dans le contexte, facteur 4 → 16.
      Gated sur le split train. L'ancienne DRS est documentée pour ce qu'elle fait réellement.

**Critère de sortie P1 :** pas de collapse mesurable sur les séries bruitées, `val_loss` corrélée à
la perf downstream, ablation VICReg/SIGReg tranchée.

---

## 🚀 RUN P1 — commandes à lancer sur la VM (RunPod)

> **C'est le point où j'ai besoin de toi.** Tout le code est en place et testé, mais je n'ai
> pas de GPU ici. Deux pretrains à lancer, identiques sauf la loss — c'est l'ablation P1.11.

### Setup

```bash
git clone -b sota-roadmap https://github.com/IUseAMouse/TimeJEPA.git && cd TimeJEPA
make install && source .venv/bin/activate

# ⚠️ VÉRIFIER AVANT DE LANCER — voir « Pièges d'environnement » plus bas
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

make download-all          # ~2-3 h la première fois
```

### ⚠️ Pièges d'environnement (rencontrés en vrai, pas hypothétiques)

**1. Driver CUDA trop ancien pour le build torch.**

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
```

`12080` = le driver supporte CUDA 12.8. `pyproject.toml` demande `torch>=2.8.0`
sans borne haute, donc `uv` installe la dernière version, compilée contre une
CUDA plus récente. Sur RunPod/Colab le driver est fixé par l'hôte : impossible
de le mettre à jour depuis le conteneur. Installer un build torch assorti :

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# attendu : 2.9.1+cu128 True
```

Adapter `cu128` à ce que le driver supporte (`nvidia-smi` → colonne « CUDA Version »).
`torch 2.9.1+cu128` est la version de référence : c'est celle sur laquelle toute
la suite de tests est validée.

**2. `TypeError: object.__init__() takes exactly one argument`** — corrigé
(B18), mais c'est le symptôme d'un torch plus récent que celui de dev. Si un
symptôme du même genre apparaît ailleurs, la cause est probablement la même.

**3. Datasets trop courts pour le contexte** — corrigé (B17) : ils sont
maintenant ignorés avec un warning au lieu de tuer le run. À `context_length=512`,
`wikipedia-web-traffic-weekly` (114 pas) et `rideshare` (541 pas) sont exclus.
C'est attendu, pas une erreur.

### Run A — VICReg (référence corrigée)

```bash
python scripts/train.py --config-name tiny \
  training.loss.type=vicreg \
  wandb.run_name=p1-vicreg \
  wandb.tags="[p1,vicreg,tiny,contextualized-targets,multires]"
```

### Run B — SIGReg (la comparaison)

```bash
python scripts/train.py --config-name tiny \
  training.loss.type=sigreg \
  wandb.run_name=p1-sigreg \
  wandb.tags="[p1,sigreg,tiny,contextualized-targets,multires]"
```

Les deux héritent automatiquement de : cibles contextualisées, contexte/horizon aléatoires,
multi-résolution réelle (p=0.35), augmentations enfin actives, régularisation de la sortie
d'encodeur. `configs/model/tiny.yaml` est déjà prêt, rien à modifier.

### Ce qu'il faut surveiller dans W&B

| métrique | attendu | signal d'alarme |
|---|---|---|
| `collapse/context_std` | reste ≈ 1.0 | **descend vers 0 → collapse** |
| `collapse/effective_rank` | stable ou monte | s'effondre vers 1-2 |
| `val_loss` vs `train_loss` | même ordre de grandeur | divergence = overfit |
| `geometry/context_len` | varie sur 128…512 | constante = randomisation non active |
| `train_loss/sigreg` (run B) | décroît | plafonne haut |

⚠️ `val_loss` **n'est pas comparable entre A et B** (objectifs différents). L'arbitrage se fait
sur l'éval downstream, pas sur la loss.

### Après les runs

Rapatrier les deux checkpoints, puis :

```bash
make finetune CHECKPOINT=checkpoints/timejepa_tiny/pretrain_True/<best>.ckpt CONFIG=tiny
make evaluate CHECKPOINT=checkpoints/timejepa_tiny/pretrain_False/<best>.ckpt CONFIG=tiny
```

**Cible à battre : MASE/SN = 0.95** (meilleur checkpoint actuel), et surtout un skill positif
sur ETTm1, qui est à −31 % aujourd'hui.

### Points d'attention

- `data.batch_size=512` et `accumulate_grad_batches=3` dans tiny.yaml → batch effectif 1536.
  À ajuster selon la VRAM de la VM.
- SIGReg ajoute ~15 % de temps par step (quadrature). Normal.
- `max_epochs: 40` avec `early_stopping.patience: 25` — prévois large ou coupe manuellement.

---

## 🧪 File d'expériences — à lancer quand le pretrain SIGReg est fini

> Tout est prêt et validé (59 tests). Chaque config est un override mince de
> `tiny.yaml` : la comparaison est à variable unique par construction.

### E1 — Patch size (teste l'hypothèse « la taille de patch bloque ETTm »)

```bash
python scripts/train.py --config-name tiny_patch32 training.loss.type=sigreg wandb.run_name=e1-patch32
python scripts/train.py --config-name tiny_patch64 training.loss.type=sigreg wandb.run_name=e1-patch64
```

| config | positions de patch par cycle ETTm1 (96) | tokens de contexte |
|---|---|---|
| `tiny` (référence) | 12 | 63 |
| `tiny_patch32` | 6 | 31 |
| `tiny_patch64` | **3** ← régime des gagnants | 15 |

**Lecture.** Si le skill ETTm1 devient positif en patch64 → l'hypothèse est
confirmée, le remède est le patching multi-échelle (voir P2.10).
⚠️ Trois points sont nécessaires (12/6/3), pas seulement patch64 : à 15 tokens de
contexte, une dégradation serait ambiguë entre « mauvais ratio » et « trop peu de
tokens ». La tendance sur trois points sépare les deux.

### E2 — Profondeur du prédicteur (contrôle de E1)

```bash
python scripts/train.py --config-name tiny_deep_predictor training.loss.type=sigreg wandb.run_name=e2-deep-predictor
```

Le diagnostic a établi **que** le ratio pilote le skill ; le *mécanisme* proposé
(prédicteur à 2 couches trop court pour un motif répété seulement 4×) n'a jamais
été testé. Matrice de lecture :

| E1 améliore | E2 améliore | conclusion |
|---|---|---|
| ✅ | ❌ | le ratio période/patch est le driver |
| ❌ | ✅ | la capacité du prédicteur est le driver |
| ✅ | ✅ | les deux contribuent |
| ❌ | ❌ | le mécanisme est ailleurs |

⚠️ Confondant : +0.4M params sur 1.6M. Un gain pourrait être la capacité brute
plutôt que la profondeur.

### E3 — Datasets de finetune (8 vs 22)

Pas de nouvelle config, un override suffit. **Depuis le MÊME checkpoint pretrain :**

```bash
# A — les 8 datasets actuels
make finetune CHECKPOINT=<ckpt> CONFIG=tiny ARGS="wandb.run_name=e3-ft8"

# B — tous les datasets utilisables
python scripts/train.py --config-name tiny training.mode=finetune \
  +training.pretrained_encoder_path=<ckpt> \
  'data.datasets_finetune=${data.datasets}' \
  wandb.run_name=e3-ft-all
```

**Pourquoi ça compte.** Le finetune actuel utilise 8 des 24 datasets, tous
« saisonniers propres » : electricity, traffic, weather, m4-hourly… Il exclut
`bitcoin`, `kdd-cup-2018`, `saugeenday`, `sunspot-daily`, `nn5-daily`,
`london-smart-meters`, `fred-md`, `solar-*`, `windpower-*`, `rain-temperature`.

Autrement dit : **le décodeur est entraîné exactement sur le régime où le modèle
gagne déjà, et n'a jamais vu de série bruitée ou non stationnaire** — puis il est
évalué sur ETTm1 et exchange, où il perd. Ça pourrait expliquer une part du
−31 %, indépendamment du patching.

---

## 🌍 ROUND GÉOMÉTRIE + DONNÉES — l'étape en cours (2026-08-10)

**Constat après la tête quantile :** CRPS/SN estimé ~0.80–0.85 (était 0.95), Toto-2.0-4m à 0.52.
Le résidu est désormais surtout un problème de **données**, pas d'architecture : corpus Monash
sub-quotidien uniquement (B17 l'a prouvé), ~24 datasets, contre ~10¹² points pour Toto.

- [x] **G1. Round géométrie** — `configs/model/tiny_geo.yaml` (commit 3ddd570)
      - contexte d'entraînement → 1024 (balayage : +5 à +8 pts ETT déjà mesurés HORS distribution)
      - horizon natif 128 → 256 (h720 : 6 rolls → 3)
      - retour patch 16/8 (la motivation de patch32 a été renversée par le balayage ;
        arm patch32 = un override)
      - randomisation du contexte AUSSI au finetune (`p_random_context_finetune`, clé
        séparée, défaut 0 → configs existantes inchangées)
      - coûts documentés dans la config : 5 datasets exclus à fenêtre 1280 (dont m4-hourly),
        batch 256 × acc 6, epoch ~2-3× plus long
      - **après ce pretrain : re-balayer le contexte à l'éval** (l'optimum 640 était celui
        d'un modèle entraîné jusqu'à 512)
- [x] **G2. Rollout échantillonné** (commit 26560ce) — propage l'éventail, pas la médiane.
      Couplage comonotone (copie k continue à son niveau k), déterministe, coût B×Q sur les
      rolls uniquement. Corrige le rétrécissement mesuré des intervalles (exchange h720 :
      largeur 0.267 pour une incertitude vraie en √h). Hypothèse assumée : dépendance de rang
      parfaite entre rolls — borne par le haut, là où la médiane biaisait par le bas.
- [x] **G3. Couverture empirique** dans l'éval — `couverture X%/80%` à côté du WQL.
      C'est le chiffre qui remplace « les intervalles ont l'air ok » ; viser ~80 %.
- [x] **G4. RÉSULTATS DU ROUND (2026-08-12)** — meilleur batch du projet, de loin.

      MASE moyenne (7 datasets communs), vs référence patch32_quantile ctx512 ftall :

      | dataset     | réf   | geo 16/8 | geo vicreg | geo p32 |
      |-------------|-------|----------|------------|---------|
      | traffic     | 1.134 | **0.768**| 0.780      | 0.777   |
      | weather     | 0.997 | 0.967    | 0.967      | **0.960**|
      | electricity | 1.250 | **1.029**| 1.090      | 1.057   |
      | ettm2       | 1.334 | 1.231    | **1.225**  | 1.232   |
      | etth1       | 1.520 | 1.265    | 1.270      | **1.247**|
      | ettm1       | 1.454 | 1.369    | **1.286**  | 1.315   |
      | etth2       | 1.864 | 1.722    | 1.667      | **1.528**|
      | **moyenne** | **1.365** | 1.193 | 1.183      | **1.159**|

      - **-13 à -15 % de MASE moyenne, uniforme** : les 7 datasets s'améliorent, aucun
        ne régresse. traffic à 0.768 (skill +46 % à h96), 3 datasets sous 1.0.
        ETTh1 h720 à +39 % de skill chez p32 — le profil par horizon s'inverse
        (le pari « moins de rolls » paie). WQL suit : traffic 0.386 -> 0.283.
      - ⚠️ **Deux changements dans le bundle** : géométrie ET fix B20 (l'encodeur se
        dégèle réellement pour la 1re fois). Non séparables avec ces runs.
      - **Les 3 arms tiennent en 3 %** (bruit : etth2 h336/h720 = 2-5 fenêtres).
        => **patch32 n'est pas moins bon et coûte 4x moins cher** : c'est l'arm à
        porter vers LOTSA. La question du patching est tranchée — le driver est le
        nombre de cycles dans le contexte, pas la taille de patch.
        => SIGReg vs VICReg indiscernables en downstream ; garder SIGReg
        (un seul hyperparamètre). P1.11 est close.
      - **Restes faibles** : ettm1 toujours -20 à -30 % vs SN (dernier vrai échec) ;
        sous-couverture chronique etth2 (42-65 % vs 80 %) sur les 3 arms — structurel,
        pas le rollout, c'est le cas d'usage de la calibration conforme ;
        exchange h720 et ili inévaluables à ctx 1024 (fenêtre requise > série).

- [x] **G4.1 BALAYAGE DE CONTEXTE (2026-08-12)** — le round a acheté de la ROBUSTESSE,
      pas un optimum. MASE moyenne (7 datasets communs, arm p32) :

      | ctx        | 512   | 640   | **768**   | 1024  | 1280  |
      |------------|-------|-------|-----------|-------|-------|
      | moyenne    | 1.203 | 1.162 | **1.144** | 1.159 | 1.168 |

      - L'optimum est monté de 640 -> 768 (et non 1024 comme anticipé), mais le
        résultat important est la PLATITUDE : 640-1280 tient en 2 %, 512-1280 en 5 %.
        Avant le round, la même expérience donnait -33 % sur ettm1 entre longueurs.
        Sur les gros datasets (traffic/electricity/weather, milliers de fenêtres)
        chacun a un optimum différent et tous les écarts sont dans le bruit.
        => La randomisation du contexte tient sa promesse : le modèle accepte
        n'importe quelle longueur d'historique sans s'effondrer. C'est la garantie
        qu'il faut pour l'API `model.forecast(y)` (P2.7).
      - Seule la pénalité à 512 subsiste (+5 %), cohérente avec cycles-dans-le-contexte.
      - ⚠️ Réserve : le nombre de fenêtres de test change avec le contexte, donc les
        jeux ne sont pas strictement identiques. etth1/etth2 aux longs horizons
        (2-6 fenêtres) sont du bruit pur — ne conclure que sur les gros datasets.

      **=> POINT D'OPÉRATION RETENU : ctx 768.** Meilleure moyenne, meilleur bilan
      head-to-head du balayage (6/8), seule longueur >= 640 où exchange est
      entièrement évaluable (768+720=1488 < 1517) et où il sort son meilleur score
      (16.45), et moins cher que 1024 (47 patchs vs 63). Deux premières à 768 :
      ettm2 devient une victoire (1.182 vs SN 1.250) et etth2 son meilleur niveau (1.419).

      **Échecs structurels restants, insensibles au contexte** : ettm1 (-21 %) et
      exchange (-8 %). Ce sont les deux cibles à traiter par autre chose que la géométrie.

- [ ] **G4.2 Calibration conforme** (optionnel, quelques lignes, aucun réentraînement) :
      un facteur d'échelle par dataset ajusté sur un split de validation, pour ramener
      la couverture à 80 %. Motivé par etth2 (42-65 %) et weather h96 (68-72 %).

- [ ] **G4.5 — BASELINE SANS PRETRAINING** ⚡ *le contrôle du pari central, à faire AVANT LOTSA*

      Tout ce qui a été montré jusqu'ici : la recette JEPA *fonctionne*. Jamais montré :
      qu'elle *bat l'alternative*. Le contrôle décisif — même architecture, même budget,
      entraînée **supervisée de bout en bout sans pretraining** — n'a jamais tourné.
      Si le pipeline pretrain→finetune ne bat pas ce baseline, le pari central de TimeJEPA
      est infirmé quel que soit le reste ; s'il le bat, c'est la preuve qui manque au papier.

      Déjà supporté par le code : `training.mode=finetune` SANS `pretrained_encoder_path`
      = init aléatoire. **`full_finetune` obligatoire** (gradual_unfreeze gèlerait des poids
      aléatoires — et ne dégèle jamais l'encodeur).

      ```bash
      # arm p32 (jumeau du finetune geo-p32 en cours, seule différence : pas de pretrain)
      python scripts/train.py --config-name tiny_geo \
        training.mode=finetune training.finetune_mode=full_finetune \
        model.decoder.type=quantile \
        model.patch_length=32 model.stride=16 \
        model.name=timejepa_tiny_geo_p32_scratch \
        wandb.run_name=geo-p32-scratch

      # arm 16/8 (jumeau du finetune geo principal)
      python scripts/train.py --config-name tiny_geo \
        training.mode=finetune training.finetune_mode=full_finetune \
        model.decoder.type=quantile \
        model.name=timejepa_tiny_geo_scratch \
        wandb.run_name=geo-scratch
      ```

      Règles de lecture :
      - mêmes overrides que le finetune jumeau (LR, epochs, early stopping) — la SEULE
        variable est l'absence de poids pré-entraînés ;
      - le from-scratch a droit au même early stopping (patience 25 → il a la place de
        converger) ; s'il est arrêté court, lui redonner des époques avant de conclure —
        être généreux avec le baseline rend la victoire (ou la défaite) incontestable ;
      - juger sur l'éval benchmark (skill/WQL/couverture), pas sur la val_loss ;
      - un jumeau par arm suffit — commencer par celui du meilleur arm geo.

- [x] **G4.6 — TRANSFERT VERS DES DONNÉES INÉDITES : RÉUSSI** (2026-08-12) — **−26 % de MASE
      moyenne, 8/8 datasets** en faveur du pré-entraîné (1.470 vs 1.992). traffic-hourly :
      R² 0.76 contre 0.095. Le pari central reçoit son premier signal positif ; E8 était un
      plafond dû à la coïncidence des corpus, pas une absence de valeur. Détail en E11 du
      registre expérimental.
      ✅ **Run de robustesse fait (E12)** : à budget généreux, le scratch obtient une MEILLEURE
      val_loss de finetune (0.1594 vs 0.1807) et reste MOINS BON sur tous les benchmarks.
      L'ampleur tombe à **−8,7 %**, mais la revendication se précise et devient défendable :
      **le pretraining améliore le TRANSFERT, pas l'ajustement au domaine d'aval**.
      Corollaire : la val_loss de finetune est un mauvais critère de sélection.
      => **LOTSA (G5) devient prioritaire** : c'est le régime où le pretraining paie.

- [x] **G4.6-old — TRANSFERT VERS DES DONNÉES INÉDITES** ⚡ *le vrai test du pari* — **configs
      prêtes** : `tiny_geo_lowdata` (pré-entraîné) et `tiny_geo_scratch_lowdata` (contrôle),
      jumelles par héritage.
      Finetune sur **m4-hourly + nn5-daily**, les deux datasets que le pretrain n'a JAMAIS pu
      voir (écartés par le filtre de longueur à fenêtre 1280), à ctx 512 / pred 256 (768 pas
      requis, ils tiennent). ~12 k fenêtres au lieu de 8 M : held-out ET petit régime d'un
      coup. nn5-daily apporte en plus une fréquence quotidienne quasi absente du corpus.
      ⚠️ Vérifier d'abord `grep -i SKIPPING <log_pretrain_tiny_geo>` : les deux doivent y être.

      **Pourquoi G4.5 seul ne suffit pas.** Le corpus de pretrain et celui de finetune
      sont les MÊMES 24 datasets. Or la proposition de valeur du SSL est d'apprendre de
      données sur lesquelles on ne peut pas superviser : quand pretrain == finetune, le
      pretraining n'est plus qu'une initialisation, le modèle supervisé voit exactement
      les mêmes octets, et l'égalité est le résultat PAR DÉFAUT — pas une réfutation.

      Le protocole SSL standard teste donc le **régime pauvre en données**, là où la
      représentation pré-entraînée porte de l'information que le supervisé ne peut pas
      reconstruire à partir de trois fois rien :

      ```bash
      # pretrained, finetune sur UN SEUL dataset
      # NB: tiny_geo (16/8), PAS tiny_geo_p32 — tiny_geo_scratch hérite de tiny_geo,
      # donc l'arm pré-entraîné doit être 16/8 lui aussi, sinon la comparaison
      # mélange « pretraining » et « taille de patch ».
      python scripts/train.py --config-name tiny_geo training.mode=finetune \
        training.finetune_mode=full_finetune \
        'data.datasets_finetune=[electricity-hourly]' \
        model.name=timejepa_geo_lowdata \
        '+training.pretrained_encoder_path="<ckpt_pretrain>"' \
        wandb.run_name=lowdata-pretrained

      # scratch, strictement identique moins les poids
      python scripts/train.py --config-name tiny_geo_scratch \
        'data.datasets_finetune=[electricity-hourly]' \
        model.name=timejepa_scratch_lowdata \
        wandb.run_name=lowdata-scratch
      ```
      (variante : garder les 24 datasets mais monter `data.stride` pour ne voir que
      ~10 % des fenêtres — même idée, distribution préservée.)

      **Lecture :**
      - pretrained gagne nettement en petit régime => le pari est VALIDÉ ; l'égalité en
        gros régime signifie seulement « le corpus de finetune suffisait ». LOTSA devient
        la suite évidente, et plus urgente.
      - égalité même en petit régime => vrai signal négatif sur l'objectif de pretraining.

      ⚠️ Ce que G4.5 ne teste PAS, et qu'il ne faut pas lui faire dire : l'arm scratch
      utilise la MÊME architecture (encodeur + prédicteur + décodeur). Une égalité ne dit
      donc rien sur « encodeur vs decoder-only » — les deux bras sont le même encodeur.

- [ ] **G5.0 ⚠️ PROTOCOLE ZERO-SHOT — décidé le 2026-08-13, conditionne tout G5**
      Le corpus Monash **contient** les benchmarks : `electricity-hourly` EST le Nixtla
      `electricity` (derniers 20 %), `traffic-hourly` EST `traffic`. Le split étant séquentiel
      par série, l'entraînement les voit sur toute leur durée. Les chiffres traffic et
      electricity ne sont donc pas publiables, et le gain de transfert d'E12 — porté par
      traffic — devient suspect. Détail au §5 du registre expérimental.
      **Correction : pretrain ET finetune sur LOTSA, évaluation zero-shot sur Monash/Nixtla.**
      Corpus d'entraînement et d'évaluation disjoints par construction.
      - [x] Liste d'exclusion étendue : Nixtla (7 motifs) + GIFT-Eval (23), séparées et
            testées sur 27 noms de sources.
      - [x] `configs/model/lotsa_tiny_zeroshot.yaml` — finetune sur LOTSA. **Protocole
            principal.** `lotsa_tiny_finetune` reste comme borne haute, avec son caveat.
      - [x] ✅ Liste GIFT-Eval **vérifiée** (2026-08-13) contre le dépôt officiel
            `huggingface.co/api/datasets/Salesforce/GiftEval/tree/main` : les 28 répertoires
            sont couverts, test de non-régression sur les noms verbatim.
            Note : GIFT-Eval autorise l'entraînement sur LEUR split de train ; exclure le
            dataset entier est plus conservateur que le minimum requis — choix assumé.
      - [x] `configs/model/lotsa_tiny_eval.yaml` — géométrie d'entraînement + données
            d'évaluation. Ni lotsa_tiny (data_dir = corpus d'entraînement) ni tiny_geo
            (couplage fragile) ne conviennent ; un test verrouille la non-dérive.
      - [ ] Attendre une BAISSE des chiffres absolus vs les runs Monash : c'est la
            disparition d'un artefact, pas une régression.

- [ ] **G5. Corpus LOTSA** — *le premier régime où pretrain >> finetune, donc le test décisif du pari.*
      [LOTSA](https://huggingface.co/datasets/Salesforce/lotsa_data) (corpus de pretrain de
      Moirai) : ~27 Md d'observations, public sur HuggingFace, **toutes fréquences** — règle
      aussi le biais sub-quotidien du corpus actuel. Travail : un convertisseur
      LOTSA→`.npy`/memmap compatible `TimeSeriesDataset`, une politique d'échantillonnage
      (on ne charge pas 27 Md de points en RAM — memmap + sous-échantillonnage par dataset),
      et re-régler `sampling_temperature`. C'est le levier n°1 restant vers Toto : après le
      round géométrie, l'écart résiduel est principalement données+compute.
      ⚠️ Vérifier le chevauchement LOTSA ↔ benchmarks d'éval (LOTSA contient des datasets
      GIFT-Eval/Monash — exclure du pretrain tout ce qui sert à l'éval).

---

## P2 — Viser le SOTA

- [x] **P2.1** Tête quantile non paramétrique, **option B** (décodeur alimenté par le contexte)
      — `configs/model/tiny_patch32_quantile.yaml`, **finetune seul, aucun pretrain**.
      Grille GIFT-Eval (9 niveaux), pinball, monotonie par construction (médiane +
      largeurs softplus cumulées vers l'extérieur).
      Bénéfice de bord : MASE est basée sur la MAE, minimisée par la **médiane** —
      le pinball donne la médiane conditionnelle exacte, là où Huber donne un objet
      intermédiaire entre moyenne et médiane.
- [x] **P2.1b** `scripts/probe_uncertainty.py` — mesure, sans entraîner, si l'incertitude
      est récupérable depuis ce que le décodeur reçoit déjà. Compare `ctx_std` (baseline
      gratuite), `z_pred` (= option A), `z_ctx` (= apport de l'option B), et les deux.
      **À lancer sur le vrai checkpoint avant d'interpréter quoi que ce soit.**

- [ ] **P2.2 — OPTION C : prédicteur probabiliste** *(ablation, à plus long terme)*

      Le prédicteur apprend `E[z_cible | z_contexte]` sous MSE : un objet **moyen** par
      construction. Mesuré : `pred_var` 0.6 contre `target_var` 0.95. L'incertitude vit
      dans le résidu, que l'inférence ne voit jamais.

      C consisterait à faire prédire une **distribution sur les latents** (μ, σ) entraînée
      par NLL gaussienne contre le latent cible réel. Le latent porterait alors lui-même
      l'incertitude, au lieu que le décodeur doive la reconstruire.

      **Coût :** réécrit l'objectif de pretrain → invalide tous les checkpoints existants.
      **Risque identifié :** interaction non caractérisée avec SIGReg. Une tête de variance
      pourrait satisfaire le régularisateur en gonflant σ plutôt qu'en enrichissant la
      représentation — il faudrait vérifier que `collapse/effective_rank` ne s'effondre pas
      pendant que la loss baisse.
      **Prérequis :** avoir P2.1 comme référence. Sans point de comparaison probabiliste,
      C n'est pas interprétable.
      **À décider seulement si :** la sonde montre que l'incertitude n'est PAS encodée
      (aucun jeu de features ne bat `ctx_std`), ou si P2.1 plafonne loin de 0.52 de CRPS.

- [ ] **P2.2b** Tête Student-t (μ, σ, ν) → queues lourdes, bitcoin/exchange.
      Alternative paramétrique à P2.1, moins prioritaire : impose une forme unimodale
      et symétrique, et n'optimise plus directement la métrique de classement.
- [ ] **P2.3** Décodeur flatten-head PatchTST-style (remplace l'overlap-add, B14)
- [x] **P2.4** Rollout correct (B10) : tout dans l'espace normalisé, `revin.freeze()` implémenté
- [ ] **P2.5** Données synthétiques : KernelSynth-like (compositions de noyaux GP) + TSMixup
- [x] **P2.6** Harness GIFT-Eval complet (23 datasets, 97 configs) — livré 2026-08-18.
      Protocole transcrit du data.py officiel dans `src/timejepa/evaluation/gift.py`
      (37 tests), CLI `scripts/evaluate_gift.py`, sortie au format CSV du leaderboard
      + agrégat geomean vs Seasonal Naive officiel ET local. Données : `make gift-download`.
      `python scripts/evaluate_gift.py --config-name lotsa_tiny_eval +checkpoint_path=<ckpt>`
- [ ] **P2.7** Packaging HF : `PyTorchModelHubMixin` + safetensors,
      `TimeJEPA.from_pretrained("timejepa-tiny")`, `model.forecast(y, horizon, quantiles)`
- [ ] **P2.8** Model card + `examples/quickstart.ipynb`
- [ ] **P2.10** Patching multi-échelle appris (si E1 confirme) : encoder en parallèle
      à plusieurs tailles (16/32/64) et fusionner par pondération apprise. Le modèle
      **choisit** son échelle selon le signal — l'inverse d'un temporal-resolution
      encoding codé en dur. Alternatives écartées : encodeur pyramidal (plus intrusif),
      attention dilatée (moins expressif).
- [ ] **P2.9** Nettoyage : `masking.py` (B11), `output_norm` (B12), `stride` du décodeur (B13)
      — *marquer comme deprecated, NE PAS supprimer*

**Critère de sortie P2 :** figurer sur GIFT-Eval avec des chiffres défendables ; poids publiés.

---

## G6 — Ablation d'objectif : extrapolation latente contre reconstruction

**L'expérience la plus importante qui reste**, parce qu'elle teste la thèse du papier plutôt
qu'un de ses corollaires. E12 a montré que pré-entraîner aide (−8,7 %) et E14 qu'un corpus
plus large aide (ETTm1, −16,8 %) — mais le bras de comparaison d'E12 est l'ABSENCE de
pré-entraînement, jamais un objectif concurrent. « L'extrapolation latente bat la
reconstruction » figure donc au §4 du registre, la liste de ce qui n'est PAS établi.

Une seule variable change : le prédicteur régresse vers les patchs futurs BRUTS au lieu de
`target_encoder(patchs futurs)`. Tâche passé→futur, corpus, géométrie, budget, optimiseur :
tous hérités de `lotsa_tiny`. C'est l'objectif de TimesFM, donc une baseline reconnue.

L'arm tourne sur le corpus PLAFONNÉ (`data/processed/lotsa`), celui d'E14 — la comparaison
n'a de sens que si les deux partagent exactement les mêmes données. Ne pas l'écraser en
convertissant le corpus complet (voir `lotsa_tiny_full`, qui écrit dans `lotsa_full`).

- [x] **G6.1** Pretrain de l'arm reconstruction
      `python scripts/train.py --config-name lotsa_tiny_recon wandb.run_name=recon-pretrain`
- [x] **G6.2** Finetune zero-shot — protocole identique, nom de sortie distinct
      `python scripts/train.py --config-name lotsa_tiny_recon_zeroshot \`
      `    +training.pretrained_encoder_path=checkpoints/timejepa_lotsa_tiny_recon/pretrain_True/<ckpt> \`
      `    wandb.run_name=recon-zeroshot`
- [x] **G6.3** Eval et comparaison à E14
      `python scripts/evaluate.py --config-name lotsa_tiny_eval \`
      `    +checkpoint_path=checkpoints/timejepa_lotsa_tiny_recon_zs/pretrain_False/<ckpt>`

⚠️ Ne comparer QUE les MASE/WQL après finetune. Les `val_loss` de pretrain des deux arms sont
des grandeurs sans rapport (MSE latente contre MSE en espace des valeurs).

**✅ G6 TERMINÉ — verdict NÉGATIF (E15, 2026-08-18).** MASE 1.150 (JEPA) contre 1.166
(reconstruction), soit 1,4 % — mais la reconstruction gagne **17 des 28** comparaisons
appariées et l'écart médian par cellule est en sa faveur. Égalité à queues lourdes, pas
victoire. **La thèse « l'extrapolation latente bat la reconstruction » sort du papier.**

Conséquences, à traiter avant d'écrire quoi que ce soit :

- [ ] **G6.4** Recadrer le positionnement autour d'E14 (zero-shot à ~1 M paramètres sur corpus
      disjoint, ETTm1 résolu par le corpus) et NON autour de l'objectif. Le §9 du registre est
      à réécrire.
- [ ] **G6.5** Si l'on veut malgré tout dire quelque chose sur l'objectif : **≥3 graines par
      bras**. Un écart de 1,4 % sur une graine est au niveau du bruit. Seul signal
      éventuellement réel : la reconstruction se dégrade au long horizon (11/14 des cellules à
      h≤192, 6/14 à h≥336) — mais l'ampleur repose sur deux cellules.
- [x] ~~**G6.6** Troisième bras reconstruction + SIGReg~~ — ABANDONNÉ. Le bras recon tourne sans
      aucune régularisation d'encodeur, `context_var` chute de 0,80 à 0,20, et l'écart en aval
      reste de 1,4 %. La variance de représentation ne porte donc pas le transfert : isoler le
      régularisateur n'apprendrait plus grand-chose.

---

## G7 — Courbe d'échelle sur le corpus complet (tiny ~1M, mini ~5M)

E16 situe le tiny plafonné à 0.979/0.677 sur GIFT-Eval — à 3,3 points de Moirai-small qui a
14× ses paramètres. Les échecs sont concentrés sur les fréquences sous-représentées du corpus
plafonné (10S, 10T) : exactement le profil qu'E14 avait sur ETTm1 avant LOTSA. G7 teste les
deux leviers restants, corpus complet puis capacité, en ne bougeant qu'UNE variable par run.

- [ ] **G7.1** Conversion complète (en cours) : `make lotsa-download CHUNKS=1000000 OUT=data/processed/lotsa_full`
      puis AUDIT d'équilibre (en-tête de lotsa_tiny_full.yaml) — aucune famille > ~15 %.
      Si une famille domine (cmip6/era5 probables sans plafonds) : reconvertir CETTE famille
      seule via `--subsets <tranches> --max-chunks-per-subset N` après avoir retiré ses .npy
      de lotsa_full — seul cas où un effacement est nécessaire, à faire à la main.
- [ ] **G7.1b** DÉNOMBRER LE CORPUS EXACT de chaque pretrain, et le consigner au registre.
      Le nombre d'observations réellement vues est l'axe X de toute courbe de scaling, et il
      n'est écrit nulle part aujourd'hui — « ~12 % de LOTSA » est une estimation, pas une
      mesure. À produire pour les DEUX corpus (plafonné et complet), et à répéter pour tout
      corpus futur :
      `python -c "import numpy as np, glob; a=[np.load(f, mmap_mode='r') for f in glob.glob('data/processed/lotsa*/*.npy')]"`
      — plus précisément, pour chaque répertoire : nombre de fichiers, de morceaux, longueur
      de morceau, total d'observations, et le total distinct RÉELLEMENT vu par époque
      (fenêtres = (L−1280)/stride par morceau × morceaux, à distinguer du nombre
      d'observations : E13b a montré que 83 M de fenêtres venaient de ~800 k morceaux seulement).
      Consigner les deux chiffres — observations et fenêtres distinctes — pour chaque run du
      papier. Sans cela, la courbe d'échelle n'a pas d'abscisse défendable.
- [ ] **G7.2** Tiny sur corpus complet :
      `make train CONFIG=lotsa_tiny_full ARGS="wandb.run_name=tiny-full"`
      ⚠️ Sélection : prendre le DERNIER checkpoint, pas le meilleur val_loss — avec un recuit
      correct sur max_epochs=2 le dernier est le légitime (E12/E13).
- [ ] **G7.3** Finetune zero-shot + evals :
      `python scripts/train.py --config-name lotsa_tiny_full_zeroshot \`
      `    +training.pretrained_encoder_path=checkpoints/timejepa_lotsa_tiny_full/pretrain_True/<ckpt>`
      puis `evaluate.py`/`evaluate_gift.py --config-name lotsa_tiny_eval` sur
      `checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/<ckpt>`
- [ ] **G7.3b** POINT DE CONTRÔLE avant de payer le run mini : comparer les deux evals de
      G7.3 à E14 (Nixtla) et E16 (GIFT, ratio global vs 0.979). Regarder spécifiquement
      `bizitobs/10S` et `solar/10T` — la prédiction écrite ici AVANT le run est que le corpus
      complet les corrige comme LOTSA a corrigé ETTm1. Si tiny-full ne bouge pas, comprendre
      pourquoi avant de lancer G7.4.
- [ ] **G7.4** Mini sur corpus complet :
      `make train CONFIG=lotsa_mini_full ARGS="wandb.run_name=mini-full"`
- [ ] **G7.5** Finetune zero-shot mini + evals (config `lotsa_mini_full_zeroshot`, eval avec
      `lotsa_mini_eval`)

### G8 — Fermer l'écart fréquentiel (issu d'E17, à ordonnancer après G7)

E17 montre que l'écart au leaderboard suit la couverture fréquentielle (×1,08 sur 5T, ×2,42 sur
10S) et non l'horizon. Par effet attendu décroissant :

- [ ] **G8.1** Aligner les motifs d'exclusion sur `GiftEvalPretrain` (152 sous-ensembles) plutôt
      que sur une liste maison plus stricte : `solar_power`, `taxi_30min`, `kdd2022`, `LOS_LOOP`,
      `covid19_energy` y sont AUTORISÉS et nous les excluons. Vérifier un par un qu'il n'y a pas
      de recoupement réel avec un dataset d'éval avant de les réintégrer.
- [ ] **G8.2** Conditionnement fréquentiel explicite (façon FlowState `scale_factor`) — réponse
      architecturale directe au mode d'échec mesuré.
- [ ] **P2.5 (remonté)** Données synthétiques : couvrir les fréquences absentes du corpus.
- [ ] **G8.3** Contexte 2048 (contre 1024) — à évaluer coût attention vs gain.
- [ ] **G8.4** Scaler robuste (arcsinh) en alternative à RevIN, dont le plancher epsilon est une
      pathologie mesurée (G6).

### G9 — Équivariance d'échelle : la justification architecturale de JEPA

**Le constat.** E17 corrigé isole une faiblesse sub-horaire réelle (10T/15T dégradent sur
solar, jena_weather, ett1, ett2 ; le 5T va bien). FlowState (IBM, 9,1M, #2 du leaderboard)
règle ce problème par construction : son SSM est à temps continu, `Ā = e^{AΔ}`, et il suffit
de poser `Δ̄ = s_Δ·Δ` avec `s_Δ = 24/saisonnalité` pour qu'un cycle journalier occupe le même
angle dans l'espace d'état quelle que soit la fréquence. Limites déclarées par le papier
(arXiv:2508.05287) : équivariance APPROXIMATIVE et non prouvée, et `s_Δ` est fourni de
l'extérieur au niveau du dataset — il faut connaître la saisonnalité.

**Ce que nous pouvons faire, et que la reconstruction ne peut pas.** Le pendant transformer
de Δ existe déjà : RoPE tourne de `m·θᵢ`, un SSM diagonal de `ω·Δ` — c'est la même opération,
et mettre les positions à l'échelle est la « position interpolation » connue des LLM. Mais
surtout, **comparer des LATENTS permet d'imposer l'équivariance par l'objectif** : contexte à
la résolution k₁, encodeur cible voyant le même futur PHYSIQUE à la résolution k₂, prédicteur
conditionné par w = k₂/k₁. Avec une loss de reconstruction c'est impossible — deux résolutions
n'ont pas les mêmes valeurs cibles. **C'est la première justification proprement
architecturale du choix JEPA**, plus forte que le signal d'horizon d'E16.

⚠️ Viser l'ÉQUIVARIANCE (le latent dépend de l'échelle, le prédicteur est informé de
l'échelle cible), pas l'INVARIANCE (latent identique quelle que soit l'échelle) : une série à
5 min et sa décimation horaire diffèrent réellement, et exiger l'égalité détruirait de
l'information. C'est aussi pourquoi le paramètre `w` est nécessaire et non un ajout cosmétique.

⚠️ Ce que G9 ne règlera PAS : les écarts de DOMAINE mesurés en E17 (bizitobs ×2,4 à 10S mais
aussi ×2,04 à l'heure ; us_births ×2,07 en D/M/W indifféremment). Ceux-là relèvent du corpus.

- [ ] **G9.0** Rallumer la multi-résolution au pretrain LOTSA — config seule, mécanisme DÉJÀ
      présent dans `datamodule.py`. `tiny.yaml` l'utilise (`[1,2,3,4]`, p=0.35) depuis toujours ;
      seul `lotsa_tiny.yaml` l'a coupée, et c'est le seul modèle avec le trou sub-horaire.
      Proposé : `multi_resolution_factors: [1,2,3,4,6]`, `p_multi_resolution: 0.3`.
- [ ] **G9.1** Mise à l'échelle RoPE À L'INFÉRENCE SEULE, `s_Δ` dérivé de la saisonnalité que
      `evaluation/gift.py` calcule déjà. Aucun réentraînement : testable sur le checkpoint
      actuel en une soirée. Répond à la question qui conditionne tout le reste — « le modèle
      ne sait pas représenter cette échelle » ou « il ne l'a jamais vue » ?
- [ ] **G9.2** JEPA inter-résolution : k₁/k₂ tirés indépendamment, `w = k₂/k₁` conditionnant
      le prédicteur, cible = `target_encoder(futur décimé à k₂)`. SIGReg empêche la solution
      dégénérée. Arm additif, aucune config existante modifiée.
- [ ] **G9.3** Ablation : G9.2 contre G9.0 seul — l'équivariance par l'objectif bat-elle
      l'augmentation par décimation ? C'est ça, le résultat publiable.

**Critère de sortie G9 :** un écart 10T/15T ramené au niveau du 5T, et une réponse chiffrée à
« l'objectif latent permet-il une équivariance que la reconstruction ne permet pas ».

---

### P3 — Plan de release : versions successives et critères de publication

L'API d'inférence dépend de la classe `TimeJEPA` et de `forecast()`. Tant que le NOM de la
classe et la SIGNATURE de ses méthodes ne bougent pas, changer de modèle ne casse rien — les
nouveaux paramètres (ex. `scale_factor` de G9) s'ajoutent avec une valeur par défaut qui
reproduit le comportement actuel. Les deux vraies sources de rupture sont ailleurs :
1. **les clés du state_dict** — conditionner le prédicteur (G9.2) ajoute des poids, donc un
   checkpoint ancien ne charge pas une architecture nouvelle. D'où un numéro de version dans
   le checkpoint, et `load_checkpoint` qui refuse explicitement plutôt que de charger
   partiellement en silence (le mode d'échec de B20).
2. **la géométrie** (contexte, patch) — déjà une source de chiffres faux en silence.

**Critères de release, à trancher AVANT chaque publication de poids :**

| version | modèle | critère de sortie |
|---|---|---|
| v0.1 | tiny-full (G7.3) | ratio CRPS GIFT < 0.65 ET aucune régression vs 0.677 |
| v0.2 | mini-full (G7.5) | < 0.60, et dominer TTM-R2 sur les deux métriques |
| v0.3 | + équivariance (G9) | 10T/15T au niveau du 5T |
| v1.0 | publication HF | ≥3 graines, calibration traitée (G4.2), model card honnête |

- [ ] **P3.1** Couche d'inférence : `TimeJEPA.from_pretrained()` + `forecast(y, horizon,
      quantiles)` sur numpy nu, sans Hydra ni Lightning, en réutilisant `evaluation/loading.py`.
      À écrire PENDANT les runs G7 (le code ne bouge pas, les GPU tournent).
- [ ] **P3.2** Version de checkpoint + refus explicite au chargement si incompatible.
- [ ] **P3.3** HF (`PyTorchModelHubMixin` + safetensors) UNIQUEMENT au premier checkpoint qui
      passe un critère de release ci-dessus. Publier des poids dont la perf n'est pas située
      est ce que le §« Décisions actées » interdit depuis le début.

---

**Critère de sortie G7 :** trois points de courbe (tiny-plafonné 0.979 / tiny-complet /
mini-complet) sur GIFT-Eval, protocole identique, CHACUN avec sa taille de corpus mesurée
(G7.1b) — sans abscisse, ce n'est pas une courbe d'échelle. Si le corpus complet corrige bizitobs/solar
comme LOTSA a corrigé ETTm1, la thèse « le levier est le corpus » a sa troisième réplication.


---

## Décisions actées

- **Univarié conservé** pour l'instant. Channel-independent bat channel-mixing sur ces benchmarks
  (cf. PatchTST) et la matrice de corrélation cross-datasets en pretrain serait ultra-sparse.
  Covariables plus tard, en features exogènes conditionnelles.
- **SIGReg en alternative de VICReg**, pas en remplacement → les deux doivent rester testables.
- **Probabiliste au finetune, pas au pretrain.** Le JEPA prédit une représentation, pas une valeur.
  L'ambiguïté du futur se traite en pretrain par prédicteur stochastique / multi-horizons, pas par pinball.
- **Packaging HF en dernier.** Publier des poids dont on ne connaît pas la perf réelle n'a pas de sens.

## Journal

- 2026-08-18 (suite 2) — E17 corrigé (fréquence/domaine étaient confondus). G8.1 : 20
  sous-ensembles LOTSA réadmis. G9 ajouté (équivariance d'échelle) et P3 (plan de release).
- 2026-08-18 (suite) — E16 : GIFT-Eval livré et mesuré (0.979/0.677 à ~1M, zero-shot). Le
  signal d'horizon de G6 se réplique (monotone sur 97 configs). G7 ajouté : courbe d'échelle
  tiny/mini sur corpus complet.
- 2026-08-18 — G6 tranché, négativement (E15). La contribution du papier bascule de l'objectif
  vers le passage à l'échelle du corpus (E14). Priorité suivante : le harness GIFT-Eval (P2.6),
  sans lequel « battre le SOTA » reste non mesurable.
- 2026-08-16 — G6 ajouté : ablation d'objectif (`lotsa_tiny_recon`), le contrôle qui manquait
  à la thèse. Corpus complet redirigé vers `data/processed/lotsa_full` pour ne pas écraser le
  corpus d'E14, qui sert de contrôle à G6.
- 2026-08-09 — Audit complet. Branche `sota-roadmap` créée. PLAN.md initialisé. Démarrage P0.
