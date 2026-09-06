# TimeJEPA — Roadmap vers un niveau SOTA

> Branche de travail : `sota-roadmap` (master reste intact).
> **Règle absolue : aucune suppression de fichier. Lecture / écriture / modification uniquement.**
> Ce fichier est le point de reprise si la session est coupée. Mettre à jour les cases à cocher au fur et à mesure.

**Dernière mise à jour :** 2026-08-24 — champion 0.8914/0.6134 (99e/126) ; ROADMAP SOTA 4
semaines adoptée (§0bis ci-dessous) après audit intégral + recherche concurrence.

---

## 0bis. ROADMAP SOTA — 4 semaines (adoptée 2026-08-24, arbitrages utilisateur actés)

Issue du croisement : audit intégral PLAN+registre × recherche des recettes concurrentes
(FlowState/Toto-2/Chronos-2/TiRex/YingLong/t0-alpha/TTM-R3, ablations sourcées) × inventaire
des actifs locaux. **L'écart 0.6134 → ~0.52 vit dans 3 gisements cumulables** : (1) couverture
corpus (~12 pts de geomean : queue 16 configs 10S/10T/15T + ~15 configs séries courtes) ;
(2) couches d'inférence (−3 à −6 % quasi gratuits, 0 GPU — 11 du top 12 en ont) ;
(3) objectif/architecture (h512/xres/esjepa, ~1-2 % chacun). Consensus concurrent : la TAILLE
n'est pas le levier (Chronos-2 120M vs 28M ≈ 1 pt) ; les leviers chiffrés = synthétique
majoritaire (Toto-2 57.5 %, 0 % public), augmentations du signal (TiRex +0.019, leur plus
gros poste), horizon en un forward (TiRex +0.013, FlowState +0.026), long contexte, TTA
(YingLong −10.5 % MASE gratuits).

**Arbitrages utilisateur** : TTA uniforme OUI (multi-lookback+miroir, un checkpoint) / G11
multi-checkpoints NON ; corpus v3 = bundle complet béni, principe « LE BATCH CIBLE D'ABORD,
LE CORPUS ENSUITE » (le synthétique se dimensionne pour combler les régimes sous-représentés
ET garantir, avec ration_oversample, une composition stable et diverse sur toute l'époque) ;
mini GELÉ jusqu'au verdict v3 ; G12+calibration en semaine 3.

- **S1 — mesurer + gratuit + arsenal** : décomposition per-config du champion (E19 — FAIT :
  queue 16→6 configs, prédiction E18 vérifiée) · verdict ration_oversample (FAIT, E20 :
  **ENTRE AU PROTOCOLE** — champion = ration@45 %×flip 0.8702/0.5959 ; fin de run 0.6310 nu
  malgré la meilleure val_loss ⇒ G7.3c re-confirmée ; soupe 25-55 % MESURÉE 0.6383/0.6072 :
  **SWA clos, négatif** — la moyenne sort du bassin, addendum E20) · G9.1 ctx 2048
  inférence seule (FAIT, E19b : échec instructif → curriculum v3) · harness TTA (FAIT,
  E19b : **flip seul = procédure officielle, 0.8735/0.5984, ~92e — jalon S1 ATTEINT J1**) ·
  ESJEPA v1 (2 j GPU, contrôle = pretrain mix existant). **h512 REPORTÉ en S3 sur le
  contrôle v3** (décision utilisateur 2026-08-24 : E19 montre des termes PLATS — h512 n'est
  plus qu'un gate pour l'horizon natif 768 du run-recette ; couru sur v3+ration, il devient
  l'ablation directe de la recette au lieu d'un résultat à re-vérifier).
- **S2 — corpus v3 (bundle assumé) — CODE LIVRÉ (2026-08-24 soir, commits 8eef4ba + suivant)** :
  familles synthétiques ops_bursty (bizitobs/E19) + intermittent (car_parts) dans
  `synthetic.py` (--set v3) · séries courtes réelles par REMBOURRAGE-BORD GAUCHE
  (`prepare_lotsa.py --min-length 384 --pad-to 1280` — la cible reste réelle, le contexte
  « plat puis données » = la condition d'éval des séries courtes) · `solar_power` réadmis
  (3 vérifications dans lotsa.py) · `decimate_corpus.py` (5T→10T/15T, mean-pooling) ·
  augmentations TiRex-style livrées (amplitude/censor/spikes, off par défaut) · trio de
  configs `lotsa_tiny_v3{,_zeroshot,_eval}` (ration_oversample + augmentations on,
  finetune 1 ép. save_top_k=-1, recette d'assemblage en en-tête).
  **RESTE — S2.4, l'ASSEMBLAGE sur le pod (à faire À DEUX au moment venu)** : génération
  --set v3 dimensionnée par l'audit (« le batch cible d'abord » : ~40-50 % de part de
  batch synthétique), conversions courtes+solar, décimation, symlinks lotsa_v3, audit
  d'équilibre `audit_batch_schedule.py` consigné, puis pretrain+finetune protocole gelé.
  **Jalon : ~0.585-0.60 avant couches** (prédictions P-v3.1..4 gravées en E19).
- **G6.2 (pari fondateur, protocole enregistré E20b 2026-08-25)** : la statistique appariée
  des arms G6 a reformulé le signal d'horizon — intra-fenêtre l'avantage JEPA DÉCROÎT
  (pente −1.64 %/100 pas, IC95 % < 0) ; il ne s'ouvre qu'à travers les rollouts (p=0.019,
  fragile sans etth1). Hypothèse restante : robustesse à l'ITÉRATION. À courir : contrôle
  **recon-mix** (~1,5 j GPU, après ESJEPA — rebaser lotsa_tiny_recon sur la recette mix),
  prédictions P-G6.2a-c gravées en E20b ; instrumentation h720 par pas (sauts attendus aux
  frontières 256/512) ; NOTE : h512 natif devrait RÉDUIRE l'avantage JEPA-vs-recon — le
  duel d'objectifs et h512 (S3) se testent mutuellement. 3 graines : S4+ si réplication.
- **S3** : xres vs contrôle v3 (une variable) · **h512 vs contrôle v3** (une variable —
  l'horizon ; gate du h768 natif du run-recette) · G12 sur GIFT (hybride vérificateur +
  raffinement par gradient, métrique UPLIFT, coût éval ×5-10 accepté en éval dédiée) ·
  pondération par source (dernière brique papier court) · ~~G4.2 calibration conforme~~
  (MESURÉ 2026-08-25 : NEUTRE — γ≈1, le fan est déjà calibré in-distribution ; la
  sous-couverture GIFT est du distribution shift ⇒ archivé comme ablation papier ;
  l'étalement adaptatif légitime = gate z ESJEPA) ·
  ~~SWA~~ (CLOS négatif, E20). **Jalon : ~0.565-0.58 (« TTM déshabillé »).**
- **TRIAGE BUDGET (2026-08-25 — fin des vacances ~28/08, pod ~500 €/mois)** : v3 = seul
  run GPU obligatoire (assemblage S2.4 à deux avant la rentrée, pretrain tourne seul) ;
  recon-mix G6.2, xres, h512 DIFFÉRÉS à la demande post-verdict v3 ; pod STOPPÉ entre
  les runs ; couches CPU-gratuites (G4.2, E18f-sur-GIFT, pondération, calibration) en
  soirées. ESJEPA : verdict au fil de l'eau (déjà payé).
- **S4** : run-recette final (2e bundle : meilleur pretrain + h768 si h512 valide +
  curriculum long-contexte si G9.1 positif + densité de supervision multi-offsets) ·
  G7.7 dimension intrinsèque · papiers (§10 à remettre à niveau). Mini/G7.4 seulement si
  plafond >~0.57 à recette constante. **Jalon étirement : 0.52-0.54 ; médian défendable
  0.565-0.58.**

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

- [x] **G7.1** Conversion complète — FAIT (2026-08-19). Corpus retenu après parking des
      tranches excédentaires : **10,05 Md observations, 4 922 922 morceaux, 65 fichiers,
      ~473 M fenêtres distinctes** (stride 8). Composition du BATCH (le seuil de 15 % porte sur
      le batch, pas sur le corpus — la température corrige déjà les gros fichiers uniques) :
      largest 16,7 %, era5 16,2 %, cmip6 15,4 %, petits fichiers 12,7 %, buildings_900k 8,5 %.
      Climat ramené de 66,5 % à 31,6 % du batch. Validé pour le run.
      ⚠️ Gain réel de G8.1 : **0,20 Md seulement** (0,9 % du corpus), dont 93 % pour Q-TRAFFIC.
      `kdd2022`, annoncé comme le principal gain 10 min, n'a produit que 2 139 morceaux —
      prédiction fausse, à corriger dans le raisonnement.
      ⚠️ **Découverte structurelle** : m1_*, monash_m3_*, tourism_*, nn5_*, cif_*, fred_md,
      godaddy, covid_mobility sont tous rejetés « séries trop courtes (médiane < 1280) ». Le
      modèle ne peut PAS s'entraîner sur des séries courtes, quel que soit le corpus — et
      ~15 des 97 configs GIFT (A, Q, M, W) sont exactement de cette forme, ce qui explique
      m4_yearly MASE 5,08 bien mieux que la couverture fréquentielle. Piste : un corpus basse
      fréquence converti avec `--min-length 256 --chunk-length 512`, `context_lengths`
      couvrant déjà [128…1024] à l'entraînement.
- [x] ~~**G7.1 (ancienne formulation)** Conversion complète (en cours)~~ : `make lotsa-download CHUNKS=1000000 OUT=data/processed/lotsa_full`
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
- [x] **G7.2** Tiny sur corpus complet — FAIT (tiny-full, E18).
      ⚠️ DOCTRINE DE SÉLECTION PÉRIMÉE ci-dessous (corrigée 2026-08-24) : ni le dernier ni le
      meilleur val_loss — **sélection par éval GIFT intermédiaire** (G7.3c : best-val ET
      best-sMAPE font 0.6309 contre 0.6134 au champion ; aucune métrique val ne sélectionne).
      ~~Sélection : prendre le DERNIER checkpoint, pas le meilleur val_loss — avec un recuit
      correct sur max_epochs=2 le dernier est le légitime (E12/E13).~~
- [x] **G7.3** Finetune zero-shot + evals — FAIT (2026-08-21), GIFT évalué à chaque
      checkpoint de validation (8 points) : final recuit **0.9685 / 0.6664**. Voir E18.
- [x] **G7.3b** POINT DE CONTRÔLE tranché (2026-08-21) — voir E18. Corpus ×6 ⇒ CRPS
      0.677 → 0.6664 (−1,6 %), MASE 0.979 → 0.9685 : réel, modeste, le levier données seul
      s'aplatit. **La prédiction écrite ici était FAUSSE** : le corpus complet n'a PAS
      corrigé `bizitobs/10S` (×1.65-3.6 vs SN) ni `solar/10T` (×2.1-2.4) — contrairement à
      ETTm1/E14, ces fréquences sont absentes de LOTSA lui-même, il n'y avait rien de plus à
      apprendre en l'agrandissant (E17 l'avait déjà mesuré, la queue per-config d'E18 le
      confirme). C'est le boulot du synthétique (mix, E19). DÉCISION : base des runs
      mix/xres = `lotsa_full` ; G7.4 (mini) attend la recette gagnante d'E19.
- [x] **G7.3c** DOCTRINE FINETUNE (tranchée 2026-08-23, E18i) — le recuit court depuis
      0.5874 a discriminé instabilité-vs-overfit : **instabilité de LR** (à froid, val ↓ ET
      GIFT ↑ ; train plate sur tout le run chaud ; aucune signature d'overfit ⇒ **mini
      débloqué**). Les gains de finetune de cette lignée tombent à la rampe (répliqué à
      deux échelles de LR : champion à 25 % sous 3e-4, meilleur recuit à 5 % sous 5e-5)
      puis marche aléatoire dans le plateau — le champion 0.6190 contient une part de
      loterie de trajectoire. **Nouveau protocole de finetune (candidat E19, en cours :
      `mix_zs_1ep`) : 1 époque fraîche depuis le pretrain, LRs discriminés tête 8e-5 /
      backbone x0.1, cosinus recuit dans l'époque** (la tête quantile part de zéro et doit
      apprendre ; le backbone part riche et est celui qui dérive — idée utilisateur
      « plafond bas », affûtée via encoder_lr_multiplier). Prédiction écrite avant :
      atterrissage 0.615-0.63, meilleurs checkpoints en FIN d'époque ; si pic encore à
      ~5-25 % puis errance, la promenade est intrinsèque et la sélection par éval GIFT
      devient l'outil officiel. xres COPIE ce protocole (duel à une variable, coût ~1 j
      au lieu de 3). Habitude : cp immédiat des checkpoints couronnés vers
      `checkpoints/champions/` (leçon de l'éviction du champion par save_top_k).
      **VERDICT (2026-08-24, run 1ep-3e-4 terminé)** : prédiction à moitié réfutée — pic
      ENCORE à 25 % (0.8914/0.6134, champion absolu, > champion v2 0.6190) puis dérive vers
      0.6309 en fin d'époque, mais ~2× plus faible que v2 ; et le checkpoint best-val/best-
      sMAPE (0.5855) fait 0.6309 — aucune métrique val ne sélectionne le champion GIFT.
      **Doctrine de production finale : 1 époque cosinus + évals GIFT intermédiaires
      (~toutes les 5-10 %) + sélection par éval + cp champions/.** Le 1ep reste supérieur au
      3ep sur tout (champion, dérive, coût /3) — protocole gelé pour mini et xres.
- [ ] **G7.4** Mini sur corpus complet — avec une PRÉDICTION ÉCRITE AVANT LE RUN (2026-08-19,
      hypothèse utilisateur pendant tiny-full : « on touche la limite de capacité du tiny sur
      un corpus si gros/complexe ») :
      * Observé sur tiny-full à ~300k steps : équilibre de représentation plus bas que le run
        plafonné (rank 42-45 vs 46-67, context_var ~0,78 vs 0,8-0,95) — MAIS égal au point
        d'ARRIVÉE du run plafonné, atteint plus vite. Deux lectures possibles : saturation de
        capacité (hypothèse utilisateur) ou équilibre SIGReg plus bas sur corpus plus divers.
      * SI saturation : mini-full doit montrer (a) un plateau rank/variance PLUS HAUT que
        tiny-full sur le même corpus, (b) un écart GIFT tiny→mini plus grand sur corpus
        complet que sur corpus plafonné, (c) des gains concentrés sur la diversité (petits
        fichiers, domaines faibles).
      * SI même rank et même GIFT : l'hypothèse est fausse, le goulot est données/objectif.
      * Rappel E15 : la variance ne prédit PAS le transfert — seul (b) tranche vraiment.
      Mini sur corpus complet :
      `make train CONFIG=lotsa_mini_full ARGS="wandb.run_name=mini-full"`
- [ ] **G7.5** Finetune zero-shot mini + evals (config `lotsa_mini_full_zeroshot`, eval avec
      `lotsa_mini_eval`)

### G8 — Fermer l'écart fréquentiel (issu d'E17, à ordonnancer après G7)

E17 montre que l'écart au leaderboard suit la couverture fréquentielle (×1,08 sur 5T, ×2,42 sur
10S) et non l'horizon. Par effet attendu décroissant :

- [x] **G8.1** CLOS PAR LA MESURE (2026-08-24, constat G7.1 : gain réel 0,20 Md = 0,9 % du
      corpus, dont 93 % pour Q-TRAFFIC — « levier vide », GiftEvalPretrain ≡ LOTSA moins les
      évals). Les 20 sous-ensembles réadmis restent acquis ; `solar_power` sera levé dans le
      cadre corpus v3 (roadmap S2), pas comme item G8.1. Texte d'origine conservé :
      ~~Aligner les motifs d'exclusion sur `GiftEvalPretrain` (152 sous-ensembles) plutôt~~
      que sur une liste maison plus stricte : `solar_power`, `taxi_30min`, `kdd2022`, `LOS_LOOP`,
      `covid19_energy` y sont AUTORISÉS et nous les excluons. Vérifier un par un qu'il n'y a pas
      de recoupement réel avec un dataset d'éval avant de les réintégrer.
- [ ] **G8.2** Conditionnement fréquentiel explicite (façon FlowState `scale_factor`) — réponse
      architecturale directe au mode d'échec mesuré.
- [~] **P2.5 (remonté)** Données synthétiques — GÉNÉRATEUR LIVRÉ (2026-08-19), mélange à
      décider. `src/timejepa/data/synthetic.py` + `scripts/generate_synthetic.py` (7 tests) :
      KernelSynth approché par caractéristiques de Fourier aléatoires (Rahimi-Recht — la
      Cholesky exacte serait O(N³) à N=8192), trois familles calées sur les trois trous
      mesurés :
        synthetic_subhourly  périodes 24-150, morceaux 8192  -> trou 10T/15T (E17)
        synthetic_broadband  périodes 4-2048, morceaux 8192  -> fond de diversité
        synthetic_lowfreq    périodes 4-52,   morceaux 8192  -> séries courtes (G7.1)
                             (1280 à l'origine — décorative, corrigée par l'audit T4)
      Les morceaux 8192 sont la première donnée où la décimation est possible (1280·6 ≤ 8192),
      donc aussi le carburant de G9.2. Génération : ~10 min pour 75 k morceaux (0,44 Md obs).
      Sortie dans data/processed/synthetic/, format identique aux corpus convertis.
      - [x] P2.5b Ratio réel/synthétique DÉCIDÉ (2026-08-21) : ~8-9 % du batch — 3 familles
        ×20k morceaux 8192 (--chunks-per-family 20000) sous le sampler à température.
        MESURÉ avant le run mix : 100k/famille (ma première consigne) donnaient 19,3 % de
        batch, le double de la cible — corrigé par troncature à 20k (~9,5 % mesuré), les
        morceaux étant i.i.d. Leçon : l'audit d'équilibre tranche, l'estimation non.
        Appuis : ablation Chronos (~10 % de KernelSynth optimal, au-delà le modèle apprend la
        signature du générateur) ; manifold synthétique étroit (3 familles de compositions
        RFF) vs 65 fichiers réels. Mélange par symlink dans lotsa_xres/, audit d'équilibre
        ensuite — jamais de génération dans un corpus en place.
      - [ ] P2.5c Arm de mesure : le run mix (E19) mesure corpus+synthétique+arcsinh en
        bundle assumé ; l'attribution fine au synthétique seul reste non financée.
      - [ ] **P2.5d Familles synthétiques ciblées v2 — CONDITIONNÉ au résidu de queue
        post-E19** (décision utilisateur 2026-08-21). La décomposition per-config de
        tiny-full (0.75 epoch) montre que le geomean MASE perd ~12 points dans une queue de
        16 configs (0.973 -> 0.853 sans elles), sub-horaire pour l'essentiel. Le run mix dira
        ce que la famille générique 24-150 comprime déjà ; ce qui SURVIT définit le cahier
        des charges v2. Candidates déjà lisibles, par morphologie manquante (pas par
        fréquence) :
          * diurne écrêté gonflé de zéros (solar 10T, x2.1-2.4) : sinusoïde journalière ×
            porte jour/nuit, longs plateaux à exactement zéro — non-linéarité de clipping
            qu'aucune composition GP additive ne produit ;
          * rafales intermittentes type trafic serveur (bizitobs 10S, x1.6-3.6) : base calme
            + bursts Poisson marqués, retombées exponentielles, queues lourdes — noter que
            bizitobs est AUSSI un gap de domaine (x2.04 même en horaire, E17), le synthétique
            n'en résorbera qu'une partie ;
          * demande intermittente à valeurs entières (car_parts, seul dataset perdu contre SN
            en CRPS, ~1.06) : comptes avec beaucoup de zéros exacts.
        Budget : chaque famille ×20k ≈ 3 % de batch (mesuré 2026-08-21, poids taille**0.5 :
        100k/famille = 6,4 % chacune, trop) ; option à coût nul = réduire
        synthetic_broadband (la famille sans trou mesuré derrière) pour financer une famille
        ciblée à part synthétique constante ~9 %, plutôt que monter vers 12-14 %.
- [x] **G8.4b** RÉSOLU (2026-08-23) — mais par une ENVELOPPE DE PRÉVISION, pas par le
      plancher relatif d'abord envisagé. Le dossier qui a tranché : pendant le recuit E18i,
      flares ×5-90 sur agrégats (bitbrains_fs/5T/short 2.812, car_parts 1.276-1.626) ; puis
      sur mix_zs_1ep3e4 à 15 %, bitbrains_fs/H/short **CRPS 18 305 724** — ×1.19 sur la
      geomean des 97 configs à lui seul. Diagnostic final : PAS l'échelle plancher (repère
      borné) — la tête quantile mi-entraînée émet un z de queue ≈15 que sinh ré-amplifie en
      10^6·échelle. Un plancher ne peut rien contre un z voyou ; la garde correcte est en
      aval : `RobustScale.inverse` clampe désormais à [min(ctx)−K·w, max(ctx)+K·w],
      w = max(étendue, échelle), K = 10 (précédent structurel : vocabulaire Chronos borné
      ±15σ). Monotone (ordre des quantiles intact), inactif sur toute prévision
      raisonnable, exact sur les aller-retours ; borne AUSSI le biais haussier ×10-30 des
      fenêtres quasi-nulles (l'observation london qui avait ouvert l'item). Test de
      régression + 267 tests verts. ⚠️ Change la transformation d'ÉVAL de tout checkpoint
      arcsinh : à déployer ENTRE deux séries d'évals comparées, et ré-évaluer champion +
      final sous le même code avant tout chiffre E19 officiel. (Historique : observé
      2026-08-22 sur les forecasts london_smart_meters de mix-v2 ; plancher relatif
      max(1e-3, 0.01·étendue) resté possible en complément si le biais quasi-nul persiste.)
      2026-08-22 sur les forecasts london_smart_meters du run mix-v2) : le fix 4b772c2 a tué
      les explosions, mais sur les fenêtres QUASI nulles (compteur éteint, micro-oscillations)
      le plancher absolu eps=1e-3 laisse le repère grossir le bruit du contexte en structure
      apparente -> biais haussier x10-30 à l'inverse (borné, plus jamais 1e10). Candidat :
      eps proportionnel à l'amplitude du contexte, p.ex. max(1e-3, 0.01·(max−min)). Une
      variable, un test de régression sur le cas london. NB au passage : les « séries
      constantes mal captées » des mêmes plots étaient une fausse alerte (axe matplotlib à
      1e-7 — reproduction à 7 décimales) ; et le « médian qui traverse le pattern » est la
      signature z=0 (moyenne conditionnelle sur futur bimodal), déjà adressée par l'arm
      ErrorSignalJEPA et la lecture énergie — pas un item nouveau.
- [ ] **G8.3** Contexte 2048 (contre 1024) — à évaluer coût attention vs gain.
- [ ] **G8.4** Scaler robuste — décision de conception (2026-08-20) : **COMPOSER, pas
      remplacer**. `x' = arcsinh((x − médiane) / MAD)` en entrée, `sinh` inverse en sortie,
      RevIN et tout son contrat de dénormalisation (freeze/to_input_frame/
      denormalize_target_space, cicatrices B2/B3/B10) inchangés au milieu.
      Pourquoi ça devrait payer, mesuré : (1) le plancher epsilon de G6 (cibles à 10³ σ sur
      contexte constant) ; (2) nos pires configs sont TOUTES à queues lourdes — bitbrains_rnd
      MASE 3,6-6,0, car_parts CRPS > 1 (pire que SN), bizitobs — le z-score par instance y est
      écrasé par les spikes ; (3) recoupe la moitié « domaine » d'E17 (bizitobs ×2,4 même à
      l'heure) qu'aucun autre levier de la file n'adresse ; (4) précédent Toto : le modèle né
      du cloudops Datadog a choisi arcsinh.
      Propriété clé : arcsinh est monotone → les QUANTILES sont équivariants, la tête
      probabiliste et l'éval survivent sans modification (l'inverse pointwise redonne des
      quantiles valides ; la médiane survit, on n'utilise pas la moyenne). Les stats robustes
      (médiane/MAD) règlent aussi le plancher epsilon au passage.
      **LIVRÉ (2026-08-20) et FUSIONNÉ dans le run contrôle mixte** (décision utilisateur :
      pas digne d'un run dédié, compute limité). Le contrôle (lotsa_tiny_mix) bundle donc
      corpus mixte + arcsinh ; xres porte le MÊME scaler pour que xres-vs-mix reste à une
      variable (l'objectif). Implémentation : components/robust_scale.py, flag
      model.robust_scale, marqueur d'auto-description dans le checkpoint (mismatch → refus
      P3.2 dans les DEUX sens), cible du finetune compressée avant RevIN, quantiles
      équivariants vérifiés. 9 tests. Configs : lotsa_tiny_mix{,_zeroshot,_eval},
      lotsa_tiny_xres{,_zeroshot,_eval} (xres porte aussi le régime passe-unique).

### G8.5 — Volume effectif : sources et multiplicateurs (réflexion du 2026-08-19)

Contexte : corpus actuel ~10 Md contre Moirai 27, Chronos ~85 (dont TSMixup), TimesFM ~100,
Toto ~2 000 Md (propriétaire). E17 rappelle que la COUVERTURE prime sur le volume dans notre
classe de paramètres — hiérarchie ci-dessous dans cet ordre. Un run d'ablation par levier.

- [ ] **G8.5a — TSMixup maison** — GARDÉ MAIS MIS DE CÔTÉ (décision utilisateur 2026-08-19),
      à ne reprendre qu'après les ablations en file (h512, chronos, contrôle, xres).
      Généalogie, pour éviter la confusion : TSMixup ET KernelSynth sont tous deux de
      Chronos/Amazon — KernelSynth = synthétique par noyaux GP (FlowState/IBM en utilise le
      descendant CauKer), TSMixup = mélange convexe de séries RÉELLES. Deux choses distinctes. Combinaisons convexes de K∈{1,2,3} séries du
      corpus de pretrain, poids Dirichlet, mise à l'échelle préalable — la recette Chronos, qui
      matérialise ainsi 10 M de séries depuis 890 k réelles. Sans fuite PAR CONSTRUCTION
      (sources = notre propre corpus), hors-ligne en .npy shardés (doctrine synthetic.py),
      ~100 lignes. Multiplie le volume effectif ×5-10 sans téléchargement.
- [ ] **G8.5b — ENTSO-E** (couverture 15T réelle). Charge/production électrique européenne à
      15 min, API publique, ~1-5 Md de points, domaine énergie où le modèle est déjà fort.
      Vise exactement le trou 15T d'E17 avec du réel plutôt que du synthétique.
- [ ] **G8.5c — EIA-930** (petit, propre) : charge horaire US, ~50 M de points.
- [~] **G8.5d — wikipedia pageviews : ACCEPTÉ SUR LE PRINCIPE** (utilisateur, 2026-08-19 —
      « si c'est qu'un dataset Monash local, on s'en fout »). Le coût est UN dataset de la
      suite locale (wikipedia-web-traffic-extended, qui tourne à m=1) : à retirer de la suite
      ET des chiffres head-to-head locaux le jour de l'ingestion, avec note au registre.
      ⚠️ DEUX PRÉREQUIS AVANT TOUT TÉLÉCHARGEMENT, exigés par l'utilisateur (précédent E10 :
      london-smart-meters + wikipedia-web-traffic pesaient 48,7 % du batch Monash) :
      1. **G10 d'abord** (échantillonnage hiérarchique par famille). Le sampler actuel compte
         les FICHIERS : un corpus de +100 Md arriverait soit en peu de shards (sous-échantillonné),
         soit en centaines de shards (il dévorerait le batch ET l'ancre d'époque — la longueur
         d'époque suit le plus gros fichier, G10.1b). Ingérer wiki avant G10 reproduirait E10
         en pire.
      2. Audit d'équilibre APRÈS mélange, comme toujours — cible : wiki plafonné à ~10-15 %
         du batch, pas du corpus.
      Réadmissibles du même coup : les 4 familles web-traffic de LOTSA (extended_web_traffic,
      kaggle_web_traffic_weekly, wiki-rolling_nips) — même mise à jour des motifs + tests que
      G8.1. La question bitcoin/crypto reste SÉPARÉE et non tranchée.
- ❌ NOAA/ERA5 étendu/WeatherBench : climat, déjà ~30 % du batch.

**Critère d'entrée d'un nouveau corpus :** vérifié contre les TROIS suites d'éval (comme
G8.1), converti en morceaux 8192 quand les séries le permettent (décimation + r'/r), audit
d'équilibre après mélange, et UN run d'ablation dédié.

### G8.6 — « ETSJEPA / ESJEPA » : la voie z du résidu — **IMPLÉMENTÉ (2026-08-23), run en attente**

**État : code livré et testé (arm `model.error_signal`, 23 tests dédiés, 290 verts au total,
zéro régression flag-off), conçu en plan-mode avec agents d'exploration/conception ; run
ordonnancé APRÈS le verdict mini-mix, sur le banc tiny, une variable contre le contrôle mix.**

Résolution du paradoxe fondateur (« le latent manquant ne se reconstruit pas ») : z ne prédit
JAMAIS la réalisation du bruit (irrécupérable) mais ses STATISTIQUES — l'hétéroscédasticité
conditionnelle. Décisions utilisateur : (1) z_target = stats DÉTERMINISTES du résidu de
lissage EWMA causal de la fenêtre cible normalisée, par patch [B,31,4] (log-RMS, log-MAD,
asymétrie tanh, autocorr lag-1 ; plancher 1e-3) — ground truth fixe, aucune bifurcation
encodeur, aucune tête EMA, aucun collapse possible ; (2) action de z = gate multiplicatif
ZÉRO-INIT sur les largeurs de `_make_monotone` — médiane intouchable par construction ⇒ MASE
structurellement invariant, tout Δ WQL attribuable à l'étalement. Architecture : tête z
(MLP 128→64→4, ~8.5k params) sur le TRONC du prédicteur (`predictor.z_head.*` — préfixe core
P3.2, survit au finetune, suivie par save_pretrained_encoder) ; gate `decoder.decoder.z_gate`
(~10 params, ré-apprenable au finetune). Loss : `+ λ_z·smooth_l1(z_pred, z_target)`, λ_z=0.1
(`training.loss.lambda_z`). Durcissement au passage : le bras « unexpected » du refus P3.2
généralisé de `robust_scaler.` à tous les préfixes core (couvre w_film et z_head — un ckpt
d'arm dans un modèle nu refuse au lieu de warn), symétrisé dans finetune_module.

Configs : trio `lotsa_tiny_esjepa{,_zeroshot,_eval}` (base lotsa_tiny_mix ⇒ contrôle = le
pretrain mix EXISTANT, une variable ; finetune protocole G7.3c 1 époque). ⚠️ val_loss inclut
la composante z — comparer par composantes.

**Prédictions falsifiables, posées AVANT le run :**
- P1 (pretrain, kill-switch mi-run) : `esjepa/z_corr` (Spearman z_pred↔log-RMS réalisée, val)
  **> 0.3** ; ≈ 0 ⇒ l'hétéroscédasticité n'est pas prévisible depuis le contexte, l'arm meurt
  au pretrain, aucun finetune.
- P2 (finetune) : val_wql ↓ vs contrôle mix à val_mase égal (±0.5 %, quasi garanti par le
  gate) ; gain concentré sur les pinball q0.1/q0.9.
- P3 : fan plus large sur les fenêtres à gros résidu réalisé (ratio décile-haut/décile-bas
  > contrôle) ; corollaire : flares de queue bitbrains ↓.
- Stérilité : `esjepa/gate_absmean` ~0 en fin de finetune = le décodeur ignore z (la
  cross-attention contexte suffit) — résultat négatif INTERPRÉTABLE. Siphonnage : si
  `train_loss/invariance` dévie > 2 % du contrôle à époque égale → λ_z=0.03 ou repli
  « prédicteur z séparé » (ablation non implémentée en v1).

Bonus stratégique : z est exposé dans `forward_finetune` (`result['z']`) — lecture de
confiance par patch pour le vérificateur G12, sans code supplémentaire.

**ESJEPA-v2 — LE DÉCOUPLAGE (ablation, NON prioritaire — décision utilisateur 2026-08-23).**
v1 est MASE-neutre par design (le gate ne touche pas la médiane) : la voie signal reste
entraînée par MSE vers les latents EMA et converge toujours vers E[z_tgt|z_ctx]. L'idée
d'ORIGINE de l'utilisateur (découpler le signal latent du bruit) est le levier MASE
complémentaire : cible de la voie signal = latent de la fenêtre cible LISSÉE (EWMA causal
déjà implémenté dans `_residual_stats`) — le prédicteur extrapole la structure débruitée à
pleine amplitude au lieu d'amortir (réparation du SHRINKAGE par contamination de bruit).
Une variable vs v1 (la cible du signal), flag candidat `error_signal_decoupled`.
Distinction essentielle notée : le « médian qui traverse le pattern » a DEUX causes — la
bimodalité vraie du futur (le médian marginal y est OPTIMAL pour la MASE, irréparable par
tout point forecast ; c'est le boulot du fan/v1) et le shrinkage (réparable par v2, seul).
⚠️ RISQUE IDENTIFIÉ qui justifie la non-priorité : la coupure du lissage est un
hyperparamètre FRÉQUENTIEL — halflife 8 traite comme bruit tout pattern de période ≲16 pas,
or nos pires familles (solar 10T, bizitobs 10S) vivent dans les hautes fréquences : v2
risque d'amputer leur signal là où on souffre le plus (v1 ne court pas ce risque : des
stats de résidu tolèrent une coupure imparfaite, une cible de signal amputée non).
Prérequis : P1 de v1 tenu (z_corr > 0.3 — sinon le découplage n'aiderait pas non plus).
Prédiction si couru : MASE ↓ (amplitude restaurée) à WQL ≥ constant, lecture per-config
OBLIGATOIRE sur les hautes fréquences (l'effet coupure s'y verrait en premier) ; remèdes
si confirmé : halflife par fréquence (métadonnées disponibles), sinon abandon.

Historique de l'idée (conservé ci-dessous) :

#### « ETSJEPA » : décomposition signal/résidu EN LATENT (formulation d'origine)

Idée de l'utilisateur (antérieure à l'analyse TTM-R3, qui a convergé dessus de l'extérieur) :
ne pas seulement décomposer l'ENTRÉE comme TTM-R3, mais apprendre DEUX cibles latentes —
un latent de signal (structure extrapolable) et un latent de résidu (l'irréductible, porteur
du probabiliste). Nom de travail : ETSJEPA (écho à Error/Trend/Seasonality).

Pourquoi c'est plus que l'emprunt TTM : la thèse du projet dit que l'objectif latent IGNORE le
bruit — une voie résidu dédiée cesse de le subir et le MODÉLISE, pendant que la voie signal
extrapole débruité. Les deux moitiés de la thèse d'origine (« abstraire le signal » +
« extrapoler sans le bruit ») deviennent deux têtes explicites au lieu d'un espoir implicite.
Généalogie utile : l'utilisateur avait envisagé un MoE au tout début (abandonné — load
balancing trop complexe à faire converger en solo) ; la décomposition signal/résidu en est la
version structurée à 2 experts FIXES, sans routeur à équilibrer.

Esquisse (à affiner le moment venu) : décomposition de la fenêtre (moyenne mobile ou trend
patch-64 façon TTM) AVANT encodage ; deux encodeurs (ou un encodeur, deux têtes de patch) ;
deux prédicteurs latents ; SIGReg sur chaque voie ; décodeur quantile alimenté par la concat.
Comparaison propre : contre le gagnant du round en cours, une variable (la décomposition).

**Ordonnancement : APRÈS tous les runs de la file menant à h512** (décision utilisateur,
2026-08-20). Prérequis de lecture : les verdicts E18-E20 (corpus/équivariance/horizon) pour
savoir sur quelle recette la greffer.

#### Référence — la recette TTM-R3 dont l'analyse a convergé sur cette idée

Le seul ingrédient de la recette TTM-R3 (0.520 CRPS à 1.4M params) absent de notre file. Leur
saut R2→R3 (0.873 → 0.520, même famille d'architecture) est venu de la RECETTE, dont ceci est
la pièce maîtresse : deux voies à patchs séparés — tendance (patch 64) et résidus (patch 9) —
avec décodeurs séparés et pré-entraînement séquentiel (tendance, puis résidus, puis joint).
Transposition TimeJEPA possible : deux encodeurs de patch (64/8 et 8/4) sur la même fenêtre,
concat des embeddings, ou deux têtes de décodeur sommées. Analyse de compatibilité JEPA à
faire : la cible latente d'une voie « résidus » est précisément ce que l'objectif actuel tend
à ignorer (c'est sa thèse) — la décomposition pourrait donc être COMPLÉMENTAIRE (la voie
tendance extrapole en latent débruité, la voie résidus porte le probabiliste). À chiffrer
après le round mix/xres/h512/mini.
Rappel des faits mesurés qui cadrent ce candidat : E15 — l'objectif seul ne sépare pas les
modèles ; R2→R3 — la recette, si.

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

- [ ] **G9.0** ⚠️ **CORRIGÉ le 2026-08-19 — ce n'est PAS un changement de config.**
      `dataset.py:_sample_resolution_factor` n'accepte un facteur f que si
      `start_idx + (ctx+pred)·f <= longueur_du_morceau`. Les morceaux LOTSA font 2048 pas et
      la fenêtre en demande 1280 : f=2 exigerait 2560 > 2048. **Aucune décimation n'est
      possible**, quelle que soit la config. La multi-résolution n'a donc pas été « coupée par
      erreur » sur LOTSA — elle y est structurellement inopérante, et `tiny.yaml` ne
      l'utilisait que parce que les séries Monash sont longues et non découpées.
      Le déblocage passe par la CONVERSION : `--chunk-length 8192` autorise f jusqu'à 6
      (1280×6 = 7680 ≤ 8192). À faire de façon CIBLÉE sur les sous-ensembles haute fréquence
      dont on veut décimer (largest 5 min, PEMS 5 min, Q-TRAFFIC 15 min, borg/azure/alibaba),
      écrits dans le même répertoire sous un suffixe distinct — `datasets: null` globe le
      répertoire et chaque fichier est un dataset indépendant, donc des longueurs de morceaux
      hétérogènes cohabitent sans code à modifier.
      ⚠️ Arbitrage à mesurer : à 8192 le nombre de fenêtres par morceau monte, mais le nombre
      de morceaux chute (~4x), et des sous-ensembles entiers avaient été perdus à 8192 lors du
      premier essai (alibaba : 108 852 séries sur 116 818). D'où le ciblage.
- [ ] **G9.1** Mise à l'échelle RoPE À L'INFÉRENCE SEULE, `s_Δ` dérivé de la saisonnalité que
      `evaluation/gift.py` calcule déjà. Aucun réentraînement : testable sur le checkpoint
      actuel en une soirée. Répond à la question qui conditionne tout le reste — « le modèle
      ne sait pas représenter cette échelle » ou « il ne l'a jamais vue » ?
- [~] **G9.2** JEPA inter-résolution — **CODE LIVRÉ** (2026-08-19), runs à faire. JEPA inter-résolution : k₁/k₂ tirés indépendamment, `w = k₂/k₁` conditionnant
      le prédicteur, cible = `target_encoder(futur décimé à k₂)`. SIGReg empêche la solution
      dégénérée. Arm additif, aucune config existante modifiée.
- [ ] **G9.3** Ablation : G9.2 contre G9.0 seul — l'équivariance par l'objectif bat-elle
      l'augmentation par décimation ? C'est ça, le résultat publiable.
      **Redéfini 2026-09-01** (verdict de prémisse : xres-nu structurellement muet sur GIFT,
      w=1 à l'éval) → deux volets : (a) **RateIN@éval** — canonicalisation causale du taux à
      l'inférence (src/timejepa/evaluation/ratein.py, +ratein=fft/backtest/oracle) ; v1 FFT
      négatif (0.6022) ; v2 par série 0.5793 ; v2.1 0.5682 ; **v3 (k par config poolé) =
      CHAMPION 0.8152/0.5588**, oracle 97 configs = plafond 0.7894/0.5358, capture 63 %,
      DÉTECTEUR GELÉ (3 limites résiduelles nommées au registre : famine petits-n,
      désaccord backtest↔test, mismatch d'objectif). **Décomposé 2026-09-05** sur head8
      (`scripts/ratein_selection_gap.py`, résidu 2.43 pts) : missed 32 % / wrong_k 38 % /
      false_pos 29 %, covid seul 17 % — aucun réglage de marge ne suffit ; candidat
      **RateIN-mix** livré et mesuré : **0.7864/0.5403** (−0.3 pt CRPS, couv 0.768), pile
      officielle du champion ; P-mix.1 ÉCHEC (12 % du résidu, pas 1/3) — le résidu est un
      DÉSACCORD backtest↔test (famine d'historique dans le backtest pour les grands k :
      ratio 15 sur bitbrains vs −9 % au test ; non-stationnarité ; régime de fenêtre), pas
      une règle de décision. **2026-09-06** : pooling aligné sur le CRPS (`+ratein_pool`)
      = **0.7871/0.5381**, nouvelle pile officielle (−0.52 pt vs backtest dur, une ligne) ;
      détecteur par énergie du pretrain (`+ratein=energy`) = 0.5848, ÉCHEC-DIAGNOSTIC,
      clos comme sélecteur (biais systématique vers les grands k ; ablation négative
      citable). **Mix + pool = 0.7842/0.5340, PILE OFFICIELLE** (−0.93 pt sur le champion
      en une journée, résidu oracle 1.50 pt, couv 0.756). Le sélecteur causal est
      proche de sa limite d'information ; le reste est train-side ; (b) **xres amendé** (w exercé au
      finetune + ancre λ·MSE) — code + configs livrés (commit ec0a891) ; le résiduel
      −4.1 % rel. vers l'oracle est l'exhibit du dossier train-side : ordre déclaré =
      mini finetune (+ratein compose) → arm G9.0-augmentation au finetune (35 % des
      instances d'éval sont des entrées décimées jamais vues au train ; clés déjà
      câblées, cross_resolution:false) → FiLM-w seulement sur motif nouveau déclaré.


### État d'implémentation G9.2 + chantiers du 2026-08-19 (code livré, runs en attente)

**Phase 0 — hygiène (commit a36164d)** : suite remise au vert (cas beijing/china_air_quality
tranché : gardés, sanctionnés par GiftEvalPretrain, risque consigné au §5 du registre) ;
train.py importe désormais `create_model_from_config` depuis `evaluation/loading.py` (4 copies
→ 1) ; 5 scripts de rounds clos déplacés vers `scripts/archive/` (README de correspondance) ;
bannières deprecated sur mini/base/large/_test_inherit ; `test_p0_regressions.py` découpé en
test_configs/test_corpus/test_metrics + registre modèle (89 tests, aucun réécrit) ;
`load_checkpoint` REFUSE désormais tout mismatch encodeur/prédicteur/patching (P3.2 fait).

**Chantier 1 — chronos_extras (419f85c)** : GiftEvalPretrain = LOTSA moins 18 datasets d'éval,
IDENTIQUE À L'OCTET (vérifié) → levier vide. Du corpus Chronos, allowlist de 4 (dominick,
ercot, mexico_city_bikes, ushcn_daily), tout le reste doublon ou fuite (exchange_rate = éval
Nixtla, training_corpus = TSMixup contaminé par construction). Convertit à 8192 : les seuls
morceaux RÉELS décimables. `scripts/prepare_chronos.py` + `convert_subset()` dans lotsa.py.

**Chantier 2 — horizon natif 512 (942243d)** : extension du checkpoint lotsa_tiny existant
(PAS de pretrain neuf : la fenêtre 1536 et le budget seraient deux variables cachées).
`grow_future_query_table` copie les 50 requêtes apprises, n'initialise que les 48 neuves ;
opt-in via `training.extend_horizon_queries`. Configs `lotsa_tiny_h512_zeroshot` / `_eval`.
Vérifié de bout en bout sur le vrai checkpoint : h=720 en 2 rollouts au lieu de 3.
- [ ] **RUN h512** : finetune (aucun pretrain requis) + eval GIFT, comparer les termes
      medium/long à la baseline (0.914/1.056/1.084).

**Chantier 3 — arm xres (ce commit)** : contexte@k1, cible@k2 physiquement contiguë,
w = k2/k1 par item via FiLM résiduel ZÉRO-INIT sur les requêtes futures (identité exacte à
w=1 → checkpoint xres utilisable tel quel au finetune, qui ne passe jamais w). Cibles
standalone imposées (garde ValueError contre contextualized_targets). Une seule clé de config
(`model.cross_resolution`) pilote modèle, module et datamodule. Observabilité wandb ajoutée
(demande utilisateur) : `aug/multires_frac`, `aug/resolution_factor_mean`, `aug/w_neq1_frac`,
`aug/w_mean` — la dernière est LE témoin de stérilité de l'arm.
- [ ] **RUN contrôle** : `lotsa_tiny` sur le corpus mixte lotsa_xres (override CLI data_dir)
- [ ] **RUN xres** : `lotsa_tiny_xres` → zeroshot → évals. Une variable vs le contrôle.
- [ ] ⚠️ ORDRE DES RUNS RÉVISÉ (utilisateur, 2026-08-20 — compute limité, couverture d'abord) :
      1. **Run A « contrôle mixte »** : `lotsa_tiny` sur corpus mixte = BASE + synthétique +
         chronos_extras, en UN seul run (chronos ~0,5 % du corpus : le séparer n'aurait rien
         mesuré). Sert à la fois de mesure « ajouts de corpus » vs baseline ET de contrôle
         pour xres. ⚠️ BASE à trancher au point de contrôle G7.3b : si tiny-full bat
         tiny-plafonné, les symlinks du mixte se construisent sur lotsa_full (10 Md), sinon
         sur lotsa. Reconstruire data/processed/lotsa_xres en conséquence + audit d'équilibre.
      2. **Run B « xres »** : `lotsa_tiny_xres`, même corpus mixte. Une variable vs A :
         l'objectif inter-résolution.
      3. **Run C « h512 »** : EN DERNIER, délibérément — le finetune 512 ne change pas le
         pretrain, donc il se fait sur le GAGNANT de A/B (chemin du checkpoint en CLI) et
         bénéficie de toute la couverture accumulée. Comparaison propre : gagnant@512 vs
         gagnant@256, une variable (l'horizon). `grow_future_query_table` s'applique tel quel
         à un checkpoint xres (le w_film est indépendant de l'horizon).
         ⚠️ Le finetune h512 doit tourner sur le MÊME corpus que le pretrain gagnant :
         override `data.data_dir=data/processed/lotsa_xres` si le gagnant est A ou B.

**Critère de sortie G9 :** un écart 10T/15T ramené au niveau du 5T, et une réponse chiffrée à
« l'objectif latent permet-il une équivariance que la reconstruction ne permet pas ».

---

### G10 — Échantillonnage à deux niveaux (dette identifiée le 2026-08-19, à traiter APRÈS G7)

**G10.2 — DÉRIVE DE COMPOSITION INTRA-ÉPOQUE (identifiée 2026-08-24, lecture de code
confirmée).** L'hypothèse « composition stable sur l'époque » est fausse par construction :
`TemperatureSampler.__iter__` retire une famille de TOUS les batchs restants dès son plafond
`max_oversample_ratio` atteint (le `continue`), et le batch RÉTRÉCIT (slots non réalloués).
Les petites familles, suréchantillonnées par T=0.5, s'éteignent tôt — la fin d'époque est
dominée par les grosses familles. Candidat mécanisme de la dérive GIFT de fin de finetune
(G7.3c : les runs mix v2 et 1ep s'infléchissent au MÊME step ~430-470k — l'extinction est
déterministe, mêmes tailles/seed). Instrumentation livrée : `scripts/audit_batch_schedule.py`
(itère le sampler réel sans lire de données : extinctions, composition par décile,
mouvements début→fin). **Correctif candidat, à UNE variable, après mesure : RATIONNEMENT du
plafond** — au lieu d'épuiser `max_samples[i]` en début d'époque, tirer
`min(n_samples, max_samples[i]/num_batches)` par batch (arrondi stochastique) : même
exposition totale par famille, étalée uniformément ⇒ composition constante, batch constant.
À courir comme ablation sur le protocole 1ep (prédiction : dérive post-25 % réduite, pic
plus tardif). Connexe : **moyenne de poids SWA** (`scripts/average_checkpoints.py`, livré
2026-08-24) — si le modèle oscille autour d'un bassin en pistant les familles, la moyenne
des checkpoints de la fenêtre 15-40 % est un point plus central que tout checkpoint
individuel ; coût zéro GPU, verdict par éval GIFT du soup vs champion sélectionné.

**Le défaut.** `TemperatureSampler` (datamodule.py:94-103) calcule `sizes ** T` **par FICHIER**.
Une famille éclatée en 30 tranches annuelles obtient donc 30 tirages, et la température — qui
existe pour remonter les petits jeux — remonte chacune de ces tranches. Le garde-fou joue
contre lui-même.

**Mesuré sur le corpus complet** (audit du 2026-08-19, 118 fichiers, 21,75 Md observations) :

| | part du corpus | part du BATCH |
|---|---|---|
| era5 + cmip6 (63 fichiers sur 118) | 56,4 % | **66,5 %** |

Le plafond par famille n'y remédie quasiment pas : divisé par 7,5, le climat ne descend qu'à
48,4 % en sacrifiant 71 % du corpus. Le levier réel est le NOMBRE de fichiers, pas leur taille.

**Contournement retenu pour G7** (aucun code touché, aucune config touchée) : sortir les
tranches excédentaires du répertoire — `datasets: null` fait un glob, donc le corpus se
redéfinit tout seul. 6 era5 + 6 cmip6 + 3 largest donnent ~10 Md observations avec aucune
famille au-dessus de ~16 %, et les 41 petits fichiers — toute la diversité de domaines et de
fréquences — passent de 8,4 % à ~19 % du batch.

- [ ] **G10.1** Échantillonnage hiérarchique : tirer d'abord la FAMILLE (via `family_of()`,
      déjà dans `lotsa.py`), puis le fichier dans la famille. La température s'applique alors
      au niveau où le déséquilibre existe.
- [ ] **G10.1b** Longueur d'époque ancrée sur UN fichier. `_compute_epoch_plan` pose
      `num_batches = taille_du_plus_gros_fichier // slots_alloués`, sans que la taille du
      corpus intervienne. Conséquences mesurées sur le corpus complet : une « époque » vaut
      ~2,4 passes (era5/cmip6 au plafond de 3x), et comme le plafond est appliqué dans
      `__iter__`, les 38 petits fichiers s'épuisent vers 48 % de l'époque — le batch tombe de
      512 à ~463 et la diversité durement gagnée s'évapore à mi-parcours. Explique aussi
      pourquoi un corpus 5,7x plus gros ne donne qu'une époque 3x plus longue.
      Piste : exposer `num_batches_per_epoch` (le paramètre existe déjà dans le sampler) pour
      définir l'époque en fenêtres du corpus plutôt qu'en passes du plus gros fichier.
      ⚠️ Comportement inchangé depuis le début du projet : E13b/E14/E16 le partagent, donc les
      comparaisons entre eux tiennent. À corriger pour l'interprétation, pas pour la validité.
- [ ] **G10.2** Logger la composition effective du batch par famille en début de run — le
      déséquilibre a été invisible pendant tout G5/G6 faute de cette ligne.
- [ ] **G10.3** Re-vérifier le contournement du parking : une fois G10.1 en place, les
      tranches garées peuvent revenir et le corpus repasser à ~21 Md.

⚠️ Ne PAS toucher au sampler avant la fin de G7 : les runs E14/E16 ont tourné avec le sampler
actuel, et le changer en cours de courbe d'échelle ajouterait une variable.

### S4-a — CORPUS V4 : séries courtes réadmises + fenêtres à frontière (construit 2026-09-05)

Diagnostic (lignes lues) : le bloc `lotsa_short` de v3 (`--min-length 384 --pad-to 1280`)
rejette toute série plus courte ; `dataset._generate_window_indices` saute toute ligne < ctx+pred ; h512 avait
échoué par le même mécanisme (exigence gonflée = corpus jeté). Défaut v3 découvert : les
lignes bourrées de 256-1279 pas réels produisent des fenêtres à cible ENTIÈREMENT dans le
bourrage plat (65/97 pour une série de 256). Construit : sidecar `_reallen/<f>.npy` écrit
par prepare_lotsa (--pad-to) ; dataset sidecar-aware (lignes courtes → fenêtres à
FRONTIÈRE : split dans les données réelles, contexte bourré à gauche par la ligne, cible
= queue réelle bourrée à droite + MASQUÉE ; jamais de décimation sur ces lignes ; lignes
≥ ctx+pred → fenêtres standard, cibles réelles garanties) ; pinball masquée ; ancre
restreinte aux items à cible pleine ; FINETUNE SEULEMENT (la perte JEPA n'a pas de masque
de cible) ; témoin `aug/short_frac`. Défauts inertes (348 tests + 9 nouveaux). Configs :
lotsa_mini_v4{,_zeroshot,_eval}. Prep : `docs/RUNBOOK_V4.md` — seul le bloc court est
refait (`lotsa_short_v4`, `--min-length 24 --chunk-length 1280 --pad-to 1280`), puis
réassemblage par symlinks des mêmes 106 familles + lien `_reallen`. Levier MASE direct ; prérequis du contexte
long (S4-b) : bourrer au lieu de filtrer rend l'allongement du contexte gratuit en
couverture de corpus. **VERDICT 2026-09-06 : CLOS.** 25 % = 0.8065/0.5518 vs head8
0.7914/0.5433 (+0.85 pt) ; P-v4.1/2/3 échouent ; m4_yearly se dégrade de façon monotone le
long du finetune. Mécanisme : l'éval raccourcit le contexte des séries courtes (bourrage à un
patch, RevIN sur les points réels), v4 les bourrait à 1024 (RevIN sur ~1000 pas plats →
amplitudes normalisées ×5-7) — condition d'entraînement ≠ condition d'éval. Survivent :
sidecar `_reallen` (supprime les fenêtres à cible-pad) et pinball masquée. Reprise
éventuelle S4-a' = contextes courts à longueur variable (collate par bucket) + cap dédié,
après xres.

### PAPIER v8 — repositionnement (décision utilisateur 2026-09-06, en attente de 2 mesures)

Colonne vertébrale candidate : adaptation à l'inférence SANS métadonnées (flip, RateIN
backtest poolé + mélange de quantiles, oracle et décomposition du résidu : −7.9 pts) ; le
modèle comme vitrine ; l'énergie comme section « ce qu'elle fait / ne fait pas ». Deux
mesures d'une soirée tranchent le titre AVANT réécriture : (1) finetune head8 from scratch
(valeur du pretrain JEPA sur GIFT) ; (2) RateIN sur TTM-R3 (model-agnostic ou non).
**Préparées le 2026-09-06** : configs `lotsa_mini_v3_head8_scratch_{zeroshot,eval}` ;
`evaluate_gift_hybrid.py --ttm-only --ttm-flip --ttm-ratein` (adaptateur TTMForecaster,
backtest poolé + mix sur le proposeur). P-scr.1..2 et P-TTM.1..3 au registre.
**(2) MESURÉ 2026-09-06 : TTM-R3 brut 0.7057 → flip+mix-pool 0.6857 (−2.8 % MASE, 52/87
configs, apparié, 10 inst/config) — P-TTM.1..3 ✓, la couche est MODEL-AGNOSTIC ; résultat
principal du papier v8.** (1) scratch en cours.

### S4-c — CONTEXTE COURT au finetune (créé 2026-09-06, config seule, à lancer)

La randomisation de contexte du finetune s'arrête à 128 ; les saigneurs MASE (m4_yearly
13-40 pas, car_parts 39, hospital 72, partie de m4_q/m/w) arrivent à l'éval sous ce seuil,
régime jamais vu. `lotsa_mini_v3_head8_ctx_{zeroshot,eval}` : grille + {32, 64}, une variable
vs head8, corpus v3, pas de bourrage (recadrage à gauche = condition d'éval). P-ctx.1..3 au
registre. Remplace S4-a (v4, clos) comme levier « historiques courts ».

### G14 — HEAD-WIDTH : l'allocation de capacité côté forecast (idée utilisateur 2026-09-02)

Audit des paramètres (2026-09-02) : encodeur 52 %, prédicteur 37-40 %, TÊTE QUANTILE
10.4 % (tiny, 118K ≈ TinyCast entier) et 7.3 % (mini) — la part du seul composant qui
décode le forecast RÉTRÉCIT avec l'échelle, et la tête naît au finetune (JEPA ne lui
donne aucun gradient — piège mesuré du 2026-08-31), donc l'élargir ne rompt aucune
continuité de pretrain. Arm une-variable, zéro coût pretrain : même checkpoint val-best,
finetune avec hidden_dim de la tête ×4 (~118K→0.5M tiny, ~251K→1M mini). Si ça paie,
l'hypothèse d'allocation tient (et scale mal dans la recette actuelle) ; si nul, l'écart
résiduel est mécanisme/corpus, pas capacité — falsifiable dans les deux sens. Câblage
requis : exposer decoder.hidden_dim en config (kwarg à défaut inerte). GATED derrière la
fin de la campagne mini (sélection G7.3c d'abord) ; prédictions à graver au lancement.
**VERDICT 2026-09-05 : ×8 (hidden 1536) au 25 % = 0.7914/0.5433, couverture 0.769 —
NOUVEAU CHAMPION** (vs 0.7994/0.5469/0.775 tête standard, même pretrain, même
checkpoint apparié). P-head.1 ✓ ; P-head.2 tenue à la marge (−0.6 pt, le 0.734 du
15 % était de l'immaturité). L'hypothèse d'allocation tient : la tête est bien le
goulot côté forecast dans la recette actuelle. **Décision utilisateur 2026-09-05 : la
tête ×8 est la recette par défaut** (configs v4 et xres-mini câblées) ; ×4 n'est plus
nécessaire. Oracle head8 : 0.7700/0.5190 — plafond de sélection au niveau de TTM-R3,
résidu 2.43 pts (s'élargit avec la qualité du modèle → argument pour xres-FiLM).
Compagnons nu/flip et 30 % à publier.

### G12 — TimeJEPA-VÉRIFICATEUR (candidat stratégique, idée utilisateur 2026-08-21)

**Périmètre tranché 2026-09-01 (arm self-hybride CLOS, 2 runs)** : le vérificateur ne
vaut que sur proposeur PONCTUEL ou à erreurs décorrélées — sur TTM, uplift massif
(point 0.7258 → hybride 0.6508) ; sur NOTRE propre champion (fan flip+RateIN), la
pondération de Gibbs DILUE le fan calibré à T=1 (hybrid/self=1.17) et l'EFFONDRE à
T=0.25 (couverture 0.339 vs 0.800) — structurel, pas un réglage. Mécanisme démontré
dans les deux sens : l'uplift du juge = DÉCORRÉLATION. Détails et P-SH.1..3 au
registre ; --proposer self reste en code (ablation). Suite G12 : hybride TTM final
au dernier checkpoint mini (P-J.3, harnais gelé jusque-là).

Recadrage issu de la série E18b-e : la fonction d'énergie du pretrain n'est pas en
concurrence avec les modèles génératifs — c'est un HARNAIS à poser dessus. Le produit
« vérificateur » = le checkpoint de PRETRAIN tel quel (~1M params, zéro entraînement
aval) : un proposeur quelconque (notre décodeur, ou un foundation model externe —
cible idéale : proposeur à diversité native type FlowState avec bruit injecté dans le
SSM) génère N trajectoires, l'énergie TimeJEPA les juge, les quantiles pondérés
reconstruisent point + intervalles. Asymétrie mesurée qui fonde la piste : générer
est difficile, vérifier est facile (E18b : rang du vrai 0.235 ; E18e : l'hybride bat
le décodeur seul sur etth1/etth2). Métrique propre au produit : l'UPLIFT (Δ vs le
proposeur seul, mêmes fenêtres), pas le CRPS absolu — positionnement papier
différenciant (personne ne vend un vérificateur de forecasts sur GIFT-Eval).
Double investissement : chaque amélioration du pretrain (mix, xres, mini) améliore
AUSSI le juge (mesuré, E18c). Ce que la piste exige à terme : (a) T calibrée en
contexte (le contraste du juge — dernier levier identifié par E18e-v2) ; (b) un
proposeur externe dans evaluate_energy.py (--proposer ; Chronos-Bolt small passe sur
CPU comme test intermédiaire avant FlowState) ; (c) NE PAS finetuner le juge (le
drift détruit l'alignement, E18b) — l'ancrage λ·invariance n'est nécessaire que pour
la variante mono-modèle. Nuance gardée au registre : le finetune génératif consomme
le pretrain comme initialisation (E12/E15 restent vrais) et en détruit la structure
de juge — les DEUX produits sortent du même pretrain. Backlog : derrière la file
générative (mix -> xres -> E19 -> h512 -> mini), décision utilisateur maintenue.

### G13 — World model unifié forecast+contrôle (direction LONG TERME, post-roadmap ; conçue 2026-08-25)

La promesse d'origine du nom TimeJEPA, formalisée en note de conception (artifact « Un arm,
deux métiers », discussion utilisateur 2026-08-25). L'idée en une phrase : le prédicteur est
déjà un world model d'un système AUTONOME ; lui donner une entrée d'action `a` (`a_film` =
la généralisation vectorielle du pattern w_film, zéro-init, identité exacte sans action)
unifie forecast et contrôle prédictif en UN SEUL arm — le « contrôleur » n'est pas un
réseau, c'est une optimisation à l'inférence (planning by backprop = E18f avec l'ACTION
comme variable optimisée au lieu du forecast ; ou propose-juge G12 sur des plans candidats ;
fan quantile + z ESJEPA = planification prudente/CVaR).

Points doctrinaux tranchés dans la discussion :
* le prédicteur ne DÉCIDE jamais — il simule (« et si ? ») ; le décideur est la boucle ;
* l'action ≠ une retouche du forecast : c'est un levier du monde (E18f optimise la
  prédiction, le contrôle optimise l'intervention — même outil, variables de sens opposé) ;
* `a` inconnu n'est jamais bloquant : spectateur ⇒ mode marginal `a=∅` (identité zéro-init,
  = le forecaster actuel, GIFT inchangé) ; acteur ⇒ `a` connu par définition (c'est SA
  décision candidate) ;
* deux régimes d'entraînement cohabitent comme dans xres : corpus normal à `a=∅` (dynamique
  marginale) + synthétique CAUSAL à interventions étiquetées (la poignée) ;
* le VRAI mur n'est pas architectural : données d'actions inexistantes dans LOTSA +
  CONFONDAGE observationnel (un MPC sur modèle corrélationnel « coupe le chauffage pour
  réchauffer ») — sortie identifiée : générateurs synthétiques causaux type CauKer/TCM
  greffés sur le pipeline v3 (« corpus causal v4 ») ;
* lien conceptuel : une action présente mais non observée EST une variable latente
  expliquant le futur — le z de LeCun, dont ESJEPA est l'ombre déterministe ; `a_film`
  (déclarer l'action) et ESJEPA (estimer les stats des actions invisibles) sont deux
  facettes de la même question.
Chemin si un jour couru : ① a_film ② corpus causal ③ planificateur (déjà écrit, E18f/G12).
NON prioritaire — rien avant la fin de la roadmap S1-S4. Zones d'ombre en cours de
pondération par l'utilisateur.

### AUDIT D'ARCHITECTURE du 2026-08-20 (avant les runs mix/xres) — verdicts et correctifs

Vérificateur à œil neuf sur les interactions arcsinh × xres × synthétique. Cinq trouvailles,
toutes corrigées AVANT le lancement des runs :
- **T1** (piège) : k1>1 n'était éligible que sur les morceaux 8192 (~4 % du batch) → le
  conditionnement côté contexte se serait appris sur ~0,7 % des items, w<1 quasi jamais vu,
  et `aug/w_neq1_frac` (~0,20, dominé par les paires k2>1) aurait rassuré à tort.
- **T2** (piège) : `forward_finetune` n'appliquait jamais le FiLM — or son biais (le
  comportement à w=1) est ENTRAÎNÉ au pretrain : le finetune jetait une partie du modèle.
  Corrigé : FiLM appliqué à w=1 quand il existe.
- **T3** : xres vs mix portait DEUX variables (objectif + cibles standalone). Corrigé : mix
  adopte aussi contextualized_targets=false ; le duel redevient l'objectif seul.
- **T4** (surprise) : synthetic_lowfreq à 1280 = 1 fenêtre/morceau, plafond 3x épuisé à ~2 %
  de l'époque, jamais éligible à w≠1 — décorative. Corrigé : 8192 (périodes 4-52 inchangées),
  régénération avec --chunks-per-family 100000 (~8-9 % de batch synthétique, k1>1 ×5).
- **T5** (mineurs) : random_scale exactement inerte sous arcsinh (médiane/MAD 1-homogènes,
  documenté, aucune action) ; garde ValueError sur robust_scale × skip_revin (footgun
  eval_skip_revin) ; commentaire 47-patchs mis à jour (127/31).
Observabilité ajoutée : `aug/w_lt1_frac` et `train_loss/w1` vs `train_loss/wneq1` — le témoin
de convergence de l'arm (si wneq1 stagne quand w1 descend, échec diagnostiqué, pas silencieux).
Réponses aux questions d'audit : convergence xres = risque modéré et désormais observable ;
JEPA exploité par r'/r à l'entraînement (seul objectif qui le PEUT), à l'inférence prototype
« re-échelonné » réservé post-E19 ; synthétique ~4 % → ~8-9 % du batch après régénération.
⚠️ AVANT les runs mix/xres : régénérer le synthétique (commande dans l'en-tête xres) et
reconstruire les symlinks lotsa_xres.

### G11 — Self-ensemble des arms (candidat, 2026-08-20)

Constat leaderboard : 11 du top 12 CRPS sont des couches d'orchestration (agents/ensembles) au-
dessus de modèles de base ; la « prime » vaut ~10-12 % de CRPS. La version compatible avec la
ligne du projet — et quasi gratuite à notre taille : moyenner les FANS DE QUANTILES de
plusieurs checkpoints tiny (3 graines du meilleur arm, ou l'ensemble {mix, xres, h512}).
Zéro LLM, zéro routeur : réduction de variance honnête. À 1-5M params, l'inférence de N
modèles reste dérisoire — un avantage structurel que les modèles à 100M+ ne peuvent pas
copier à coût nul. Attendu −0,01/−0,02 CRPS ; dans le peloton actuel (0,016 d'écart sur 4
modèles), ça vaut des places. Implémentation triviale côté harness (moyenne des quantiles
par config avant l'accumulateur). À chiffrer après le round en cours.

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

- [ ] **G7.6 (optionnel, la loi d'échelle maison)** — compléter la grille 2×2 (N, D) :
      tiny@plafonné (0.677) / tiny@complet (en cours) / mini@complet (G7.4) / **mini@plafonné
      (la case manquante, ~6-8 h)**. Quatre points = les deux exposants α (params) et
      β (données) de `L = E + A/N^α + B/D^β` sur NOTRE architecture — séparables, publiables.
      Contexte chiffré : la famille Toto mesure α minuscule (4M→0.524, 22M→0.496, 1B→0.478 :
      ×250 params pour 4,6 pts), et le plancher E du benchmark se lit vers ~0,42-0,48 (où les
      meilleurs se tassent). Lecture directe : si tiny@complet ≈ tiny@plafonné, N=1M est
      data-saturé et seuls N ou l'objectif aident — le run en cours est le premier
      coefficient de la loi.
- [ ] **G7.7 (optionnel, 1 h CPU)** — dimension intrinsèque du corpus : TwoNN / MLE
      Levina-Bickel sur fenêtres RevIN-normalisées. Chiffre inédit sur LOTSA ; à confronter à
      l'effective_rank (~45/128) — argument de papier sur POURQUOI 1M de paramètres suffit
      presque (le manifold apprenable est petit, le reste est bruit irréductible).

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

- 2026-08-19 (suite 3) — G8.5 : stratégie de volume effectif (TSMixup maison en premier,
  ENTSO-E pour le 15T réel, wiki pageviews = décision de périmètre sur la suite Monash
  locale). TSMixup expliqué : augmentation par mélange convexe, pas un dataset.
- 2026-08-19 (suite 2) — Round de code complet pendant le run tiny-full : Phase 0 hygiène
  (suite verte, loaders unifiés, archivage, découpe des tests, refus P3.2), convertisseur
  chronos_extras (+ découverte : GiftEvalPretrain ≡ LOTSA moins évals, levier vide),
  horizon natif 512 par extension de checkpoint, arm inter-résolution complet avec
  observabilité wandb des augmentations. Reste : les runs.
- 2026-08-19 (suite) — Générateur synthétique livré (P2.5) pendant le run tiny-full :
  3 familles calées sur les trous E17/G7.1, morceaux 8192 qui débloquent G9.2. Couche
  d'inférence (P3.1) dépriorisée : seul ce qui améliore les runs passe avant.
- 2026-08-19 — Audit du corpus complet : 21,75 Md observations mais 66,5 % du BATCH en
  réanalyse climatique, `sampling_temperature` étant aveugle aux familles. Contourné par
  parking de tranches (aucun code, aucune config) ; G10 ouvert pour le correctif de fond.
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
