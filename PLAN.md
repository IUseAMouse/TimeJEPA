# TimeJEPA — Roadmap vers un niveau SOTA

> Branche de travail : `sota-roadmap` (master reste intact).
> **Règle absolue : aucune suppression de fichier. Lecture / écriture / modification uniquement.**
> Ce fichier est le point de reprise si la session est coupée. Mettre à jour les cases à cocher au fur et à mesure.

**Dernière mise à jour :** 2026-08-09 — pretrain SIGReg en cours ; configs E1/E2/E3 prêtes.

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

## P2 — Viser le SOTA

- [ ] **P2.1** Tête quantile (pinball, 9 quantiles, incréments softplus anti-croisement)
- [ ] **P2.2** Tête Student-t (μ, σ, ν) en ablation → queues lourdes, bitcoin/exchange
- [ ] **P2.3** Décodeur flatten-head PatchTST-style (remplace l'overlap-add, B14)
- [ ] **P2.4** Rollout correct (B10) : tout dans l'espace normalisé, `revin.freeze()` implémenté
- [ ] **P2.5** Données synthétiques : KernelSynth-like (compositions de noyaux GP) + TSMixup
- [ ] **P2.6** Harness GIFT-Eval complet (23 datasets, 97 configs)
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

## Décisions actées

- **Univarié conservé** pour l'instant. Channel-independent bat channel-mixing sur ces benchmarks
  (cf. PatchTST) et la matrice de corrélation cross-datasets en pretrain serait ultra-sparse.
  Covariables plus tard, en features exogènes conditionnelles.
- **SIGReg en alternative de VICReg**, pas en remplacement → les deux doivent rester testables.
- **Probabiliste au finetune, pas au pretrain.** Le JEPA prédit une représentation, pas une valeur.
  L'ambiguïté du futur se traite en pretrain par prédicteur stochastique / multi-horizons, pas par pinball.
- **Packaging HF en dernier.** Publier des poids dont on ne connaît pas la perf réelle n'a pas de sens.

## Journal

- 2026-08-09 — Audit complet. Branche `sota-roadmap` créée. PLAN.md initialisé. Démarrage P0.
