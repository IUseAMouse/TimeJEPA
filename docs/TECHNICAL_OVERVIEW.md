# TimeJEPA — Note technique des changements P0 / P1

> Objectif de ce document : que tu puisses reprendre chaque décision technique et la contester.
> Rien ici n'est « fais-moi confiance » — chaque affirmation est soit une mesure reproductible,
> soit un raisonnement que tu peux attaquer.
>
> Branche `sota-roadmap`. Convention : je distingue explicitement **ce qui est mesuré** de
> **ce qui est raisonné** de **ce qui est repris d'un papier**.

---

## Table des matières

1. [Le fil conducteur](#1-le-fil-conducteur)
2. [P0 — les bugs de protocole](#2-p0--les-bugs-de-protocole)
3. [Les métriques : pourquoi MASE et pas MSE](#3-les-métriques--pourquoi-mase-et-pas-mse)
4. [Le diagnostic ETTm](#4-le-diagnostic-ettm)
5. [SIGReg — explication complète](#5-sigreg--explication-complète)
6. [VICReg — ce qui était cassé](#6-vicreg--ce-qui-était-cassé)
7. [Les cibles contextualisées (I-JEPA)](#7-les-cibles-contextualisées-i-jepa)
8. [Randomisation de géométrie](#8-randomisation-de-géométrie)
9. [Multi-résolution réelle vs DRS](#9-multi-résolution-réelle-vs-drs)
10. [Le bug du prédicteur](#10-le-bug-du-prédicteur)
11. [Récapitulatif des fichiers touchés](#11-récapitulatif-des-fichiers-touchés)

---

## 1. Le fil conducteur

Trois familles de problèmes, dans cet ordre d'importance :

**(a) On mesurait faux.** L'évaluation Nixtla donnait au modèle des entrées hors distribution
et comparait sa sortie dans un espace différent de celui des cibles. Il n'y avait aucun
baseline, donc aucun moyen de savoir si un chiffre était bon. → **P0**

**(b) L'anti-collapse ne régularisait pas ce qu'il fallait.** VICReg mesurait la variance sur
le mauvais axe et la moitié de ses termes n'avait pas de gradient. La sortie de l'encodeur —
la représentation qui nous intéresse — n'était jamais contrainte. → **P1.1, P1.2**

**(c) Le modèle a mémorisé une géométrie d'entrée.** Découvert par expérience contrôlée après
P0. Ce n'était dans le plan initial que comme robustesse cosmétique. → **P1.7, P1.8, P1.12**

Plus trois bugs silencieux trouvés en chemin (rollout, MASE, prédicteur) qui ne faisaient
jamais planter le code — c'est précisément pour ça qu'ils avaient survécu.

---

## 2. P0 — les bugs de protocole

### 2.1 `skip_revin=True` : la confusion centrale

C'est le bug le plus coûteux, et il tient à une distinction sur laquelle il est facile de
glisser.

**Deux normalisations différentes portent le même nom courant.**

*Normalisation globale (z-score du dataset).* Les données Nixtla long-horizon sont livrées
z-scorées avec les statistiques du **train** :

```
y_normalisé[t] = (y_brut[t] − μ_train) / σ_train
```

μ et σ sont des constantes pour toute la série. Sur ETTh1, la partie test a une moyenne de
**−1.34** et un écart-type de **0.34** dans cet espace — elle a dérivé loin du train.

*Normalisation par instance (RevIN).* Le modèle, lui, a été entraîné à recevoir des fenêtres
re-normalisées **individuellement** :

```
z[t] = (y[t] − moyenne(fenêtre)) / std(fenêtre)
```

Chaque fenêtre a par construction une moyenne de 0 et un écart-type de 1.

**Ce que faisait le code.** `scripts/evaluate.py:590` passait `skip_revin=True`, avec le
commentaire « nixtla datasets are already normalized ». Conséquence : l'encodeur recevait des
fenêtres de moyenne −1.34 et d'écart-type 0.34, alors qu'il n'avait **jamais** vu autre chose
que des fenêtres centrées-réduites. Et sa sortie, exprimée dans l'espace instance-normalisé,
était comparée à des cibles vivant dans l'espace z-scoré global.

**La signature visuelle.** Sur les plots `etth1_h96_forecasts.png`, la prédiction a la bonne
*texture* mais un décrochage de niveau constant à la frontière. Un modèle simplement mauvais
produit des erreurs aléatoires ; un décalage constant est une signature de plomberie.

**Le fix.** `skip_revin=False`. Le modèle instance-normalise le contexte, prédit dans ce
repère, et `forecast_denorm` le ramène dans l'espace z-scoré global où vivent les cibles.
C'est exactement ce que font PatchTST, iTransformer et TimesNet : RevIN par-dessus des données
déjà standardisées globalement. Les deux normalisations se composent, elles ne se remplacent
pas.

**Mesure.** 40/40 des couples (dataset, checkpoint) s'améliorent. MSE −42 % en moyenne,
jusqu'à −81 % sur exchange. Le mode legacy est conservé derrière un flag et reproduit les
anciens chiffres à la 3ᵉ décimale — c'est ce qui garantit que la comparaison est honnête.

### 2.2 L'affine de RevIN

RevIN a des paramètres appris `w` et `b` appliqués **au contexte** :

```python
z_input = (x − μ)/σ · w + b          # revin._normalize
```

Mais les deux losses normalisent la **cible** en z-score simple, sans affine :

```python
target_norm = (target − revin.mean) / revin.std   # jepa_tst.py, finetune_module.py
```

Le décodeur apprend donc à produire des valeurs dans l'espace `(y−μ)/σ`. Or `_denormalize`
inversait l'affine :

```python
x = (x − b) / w        # inverse d'une transformation jamais appliquée à la sortie
x = x · σ + μ
```

**Mesure de l'ampleur.** J'ai inspecté les poids réels dans tes checkpoints : `w ∈ [0.86, 1.10]`,
`b` jusqu'à `0.089`. Soit ~6-10 % d'erreur d'échelle plus un décalage constant sur chaque
forecast dénormalisé. **Réel mais second ordre** par rapport à 2.1 — je l'ai signalé comme tel
dès le départ et la mesure l'a confirmé.

**Fix.** `RevIN.denormalize_target_space()` : l'inverse cohérent avec l'espace de la loss.
`_denormalize` est laissé intact pour les appelants qui font légitimement un aller-retour
complet.

### 2.3 Le rollout était cassé (jamais exécuté)

`JEPATST.forecast()` appelait `self.revin.freeze()`. **Cette méthode n'existait nulle part.**
Tout forecast avec `n > prediction_length` levait un `AttributeError`.

Pourquoi ça n'était jamais tombé : l'éval Nixtla utilisait `skip_revin=True` (branche court-
circuitée) et l'éval locale appelait `forecast(ctx)` avec `n=None` → cas mono-passe.

Et même réparé, la boucle mélangeait les espaces :

```python
current_context = cat([current_context[:, pred_len:],   # espace BRUT
                       forecast_norm])                   # espace NORMALISÉ
```

**Fix.** Tout le rollout se fait dans un repère unique : les statistiques RevIN sont calculées
une fois sur le vrai contexte, `freeze()` les épingle, chaque itération travaille dans ce
repère, et une seule dénormalisation à la fin. `to_input_frame()` réaligne la prédiction
(espace cible) vers l'espace d'entrée de l'encodeur (avec affine) avant ré-injection.

**Pourquoi ça compte pour la suite :** c'est le chemin de code de ton futur
`model.forecast(y, horizon=336)` sur HuggingFace.

### 2.4 ETTh1/ETTh2 ne sont pas comparables à la littérature

Vérifié empiriquement :

```
ETTh1      n_ids=1  ids=['OT']                        T=14400
ETTm1      n_ids=7  ids=['HUFL','HULL','LUFL','LULL'] T=57600
Weather    n_ids=21                                   T=52695
```

`datasetsforecast.LongHorizon` ne livre **qu'une seule série (`OT`)** pour ETTh1/ETTh2, alors
que les tableaux publiés moyennent les 7 canaux ETT. Ce n'est pas la même tâche. Un warning
est maintenant émis à l'éval et le caveat apparaît dans tous les rapports.

---

## 3. Les métriques : pourquoi MASE et pas MSE

### 3.1 Le problème avec MSE

Une MSE n'a de sens que rapportée à la variance de la cible. Exemple concret tiré de tes
anciens résultats : `etth1 h96 MSE = 0.068` semblait battre le SOTA (≈0.37) d'un facteur 5.
En réalité la variance du test ETTh1 dans cet espace est de 0.078 — donc 0.068 c'est à peine
mieux que prédire la moyenne. Le R² correspondant était 0.125.

Et une MSE ne s'agrège pas entre datasets : moyenner une MSE de 0.07 (etth1) avec une de 1.1
(ettm1) ne veut rien dire.

### 3.2 MASE

```
MASE = MAE(prédiction) / MAE(seasonal naive en interne au contexte)
```

Le dénominateur est l'erreur qu'aurait faite « répète le cycle précédent » sur l'historique.
Donc :

- **MASE = 1.0** → aussi bon que seasonal naive
- **MASE < 1.0** → meilleur
- C'est sans échelle, donc agrégeable entre datasets et comparable à GIFT-Eval

**Contrôle de sanité.** Sur ETTm1, seasonal-naive obtient MASE = 0.99–1.00, exactement la
valeur théorique. Si ce chiffre s'écarte de 1, la saisonnalité configurée est fausse.

### 3.3 Le bug MASE que j'ai introduit et corrigé

Ma première implémentation faisait la moyenne des ratios **par fenêtre** :

```python
mase = mean_i( MAE_i / scale_i )
```

Sur ETTm2 et electricity, certaines fenêtres sont quasi constantes → `scale_i ≈ 0` → le ratio
explose à ~1/eps et écrase la moyenne. Symptôme : MASE ≈ 10⁴ pour **tous** les modèles *et*
pour seasonal-naive lui-même, ce qui est manifestement absurde.

**Fix** : ratio d'agrégats (« poolé ») :

```python
mase = sum_i MAE_i / sum_i scale_i
```

Une fenêtre plate contribue alors proportionnellement à son erreur réelle au lieu d'exploser.
La forme par fenêtre reste disponible (`aggregate='per_series'`), en écartant les fenêtres
dégénérées plutôt qu'en les clampant.

### 3.4 WQL / CRPS et pourquoi la tête quantile est un prérequis

GIFT-Eval classe principalement sur le **CRPS** (approximé par le Weighted Quantile Loss).
Pour un ensemble de quantiles symétriques :

```
WQL = mean_q [ Σ_i QL_q(y_i, ŷ_{q,i}) ] / Σ_i |y_i|
QL_q(y, ŷ) = 2·[ q·(y−ŷ)⁺ + (1−q)·(ŷ−y)⁺ ]
```

**Fait important, vérifié par test** : si on donne la *même* prédiction ponctuelle à tous les
quantiles, le WQL se réduit **exactement** à la ND (`Σ|y−ŷ| / Σ|y|`). Démonstration : pour
y > ŷ, la somme sur q vaut `2·mean(q)·(y−ŷ) = (y−ŷ)`, idem par symétrie dans l'autre sens.

Conséquence directe : **un modèle ponctuel ne peut pas faire mieux que sa ND en CRPS.**
Ce n'est pas une limite de capacité, c'est une limite de format. C'est ce qui explique le
tableau :

| | MASE | CRPS |
|---|---|---|
| TimeJEPA tiny | 0.95 | 0.95 |
| Toto-2.0-4m | 0.76 | 0.52 |

L'écart en MASE (0.95 → 0.76) est un écart de qualité de modèle, modeste. L'écart en CRPS
(0.95 → 0.52) est **presque entièrement** l'absence de tête probabiliste.

---

## 4. Le diagnostic ETTm

`scripts/diagnose_ettm.py`. Deux expériences contrôlées sur
`tiny/best-unfreeze-1-stride-48-full-datasets`.

### Expérience 1 — contexte fixe (384), on fait varier la période

| cas | cycle | positions de patch/cycle | skill vs SN |
|---|---|---|---|
| ECL natif | 24 | 3 | **+28.5 %** |
| ETTm1 ÷4 | 24 | 3 | −2.6 % |
| ETTm1 ÷2 | 48 | 6 | −8.3 % |
| ETTm1 natif | 96 | 12 | −27.2 % |
| **ECL ×4 (interpolé)** | **96** | **12** | **−136.3 %** |

Le contrôle ECL est le point clé : **mêmes données**, simplement interpolées ×4. L'interpolation
*lisse* le signal (la MASE de seasonal-naive baisse de 1.32 à 1.00, donc la tâche devient plus
facile), et pourtant le modèle s'effondre. Ce n'est donc pas la difficulté du signal.

**Direction contre-intuitive** : le modèle est meilleur quand le cycle occupe **peu** de
positions de patch. Raisonnement : à 3 positions/cycle, le motif se répète 16 fois dans les 47
patchs du contexte et la structure périodique est visible à courte portée d'attention ; à 12
positions/cycle il ne se répète que 4 fois et il faut une dépendance longue portée que le
prédicteur (2 couches) n'a pas.

Ça explique aussi pourquoi `weather` (cycle 144, soit 18 positions — pire ratio qu'ETTm1)
gagne quand même à +18.7 % : son horizon de 96 pas ne fait que **0.67 cycle**. Il n'extrapole
jamais un cycle complet, une continuation locale suffit.

> **Règle unifiée.** Le modèle réussit si (a) le cycle est court en positions de patch et très
> répété, ou (b) l'horizon est plus court qu'un cycle. Il échoue dès qu'il doit **extrapoler un
> cycle de longue période**.

### Expérience 2 — période fixe, on fait varier le contexte

Ici on isole la variable confondue : le nombre de cycles dans la fenêtre.

| electricity | cycles dans ctx | skill |
|---|---|---|
| ctx=96 | 4 | −51.3 % |
| ctx=192 | 8 | +11.3 % |
| **ctx=384** | **16** | **+28.5 %** |
| ctx=768 | 32 | **−103.8 %** |

Le pic est **exactement** sur la longueur d'entraînement (384). Et ETTm1 empire quand on
allonge le contexte (−27 → −47 → −55 % à 384/768/1536), alors que « plus de cycles » devrait
aider.

**Conclusion : le modèle a mémorisé une géométrie d'entrée fixe (47 patchs)** et se dégrade des
deux côtés. Ce n'est pas un problème de capacité, c'est un défaut de généralisation.

**Caveat que je n'ai pas levé** : à ctx=768, RevIN normalise sur une fenêtre plus longue et
absorbe donc plus de non-stationnarité. Une part de la dégradation peut venir de là plutôt que
de la géométrie. Pour trancher il faudrait refaire l'expérience à statistiques RevIN gelées sur
les 384 derniers points. Je ne l'ai pas fait.

### Ce que ça change

Ton intuition « le patching fixe gêne la saisonnalité » est **validée sur l'effet**, avec deux
corrections : la direction est inversée (patchs grossiers = mieux), et il y a un second problème
indépendant, plus grave, que personne n'avait vu.

Et ça confirme ton refus du temporal-resolution encoding en dur : le problème n'est pas un
manque d'information sur la fréquence, c'est un manque de **variabilité à l'entraînement**.
La réponse est de l'augmentation, pas de l'injection.

---

## 5. SIGReg — explication complète

C'est la partie que tu voulais détaillée. Je pars de zéro.

### 5.1 Le problème que ça résout

Un JEPA prédit une représentation à partir d'une autre. Rien dans cet objectif n'empêche la
solution triviale : **si l'encodeur produit une constante, la prédiction est parfaite et la
loss est nulle.** C'est le *representation collapse*.

Toutes les méthodes JEPA/SSL ont un mécanisme anti-collapse. Historiquement : stop-gradient,
teacher-student EMA, batch normalization, whitening, termes de variance… Ce sont des
heuristiques : elles marchent, mais on ne sait pas exactement vers quoi elles poussent.

### 5.2 L'idée de LeJEPA : viser une distribution précise

LeJEPA (Balestriero & LeCun, arXiv 2511.08544) prend le problème à l'envers. Plutôt que
« empêcher le collapse », ils demandent : **quelle distribution d'embeddings minimise le risque
de prédiction downstream ?**

Réponse démontrée dans le papier : la **gaussienne isotrope** `N(0, I)`.

L'intuition (la démonstration est dans le papier, je ne la reproduis pas) : à variance fixée,
la gaussienne isotrope maximise l'entropie et répartit l'information uniformément dans toutes
les directions. Aucune direction n'est privilégiée ni gaspillée, donc une tête linéaire
downstream peut extraire n'importe quelle feature linéaire avec la même facilité, quelle que
soit la tâche.

L'anti-collapse devient alors un cas particulier : une distribution collapsée est très loin
d'une gaussienne isotrope, donc automatiquement pénalisée.

### 5.3 Obstacle : on ne peut pas comparer des distributions en grande dimension

On voudrait mesurer une distance entre la distribution empirique de nos embeddings
`z ∈ R^D` (D = 128 pour tiny) et `N(0, I_D)`. Mais estimer une densité en dimension 128 avec
quelques milliers d'échantillons est sans espoir — c'est la malédiction de la dimension.

### 5.4 La sortie : le théorème de Cramér-Wold

> Deux distributions de probabilité sur `R^D` sont identiques **si et seulement si** toutes
> leurs projections 1D coïncident.

Formellement : `P = Q` ⟺ `a^T z ~ a^T w` pour toute direction `a` de la sphère unité.

Ça transforme un problème D-dimensionnel en une famille de problèmes **1D**, où l'estimation
de densité est facile.

Et il y a un cadeau : si `z ~ N(0, I_D)`, alors pour **n'importe quelle** direction unitaire
`a`, on a `a^T z ~ N(0, 1)`. La cible est donc la *même* loi simple pour toutes les projections.
On n'a rien à recalculer par direction.

En pratique on ne teste évidemment pas toutes les directions : on en tire `M` au hasard
(M = 8–16 suffit d'après le papier), **ré-échantillonnées à chaque minibatch**. Au fil de
l'entraînement, la sphère est couverte. C'est le « **Sketched** » de SIGReg.

### 5.5 Le test 1D : Epps-Pulley

Il faut maintenant, pour chaque projection, un test d'adéquation à `N(0,1)` qui soit
**différentiable** (c'est une loss). Ça élimine :

- Kolmogorov-Smirnov : un max, non lisse
- Anderson-Darling : nécessite un tri
- Les moments : trop faibles (ne contraignent que 2 ou 3 nombres)

**Epps-Pulley** passe par la **fonction caractéristique**. Pour une variable aléatoire `U` :

```
φ_U(t) = E[e^{itU}] = E[cos(tU)] + i·E[sin(tU)]
```

C'est la transformée de Fourier de la densité. Propriété essentielle : **elle détermine
entièrement la loi**. Deux variables ont la même fonction caractéristique si et seulement si
elles ont la même distribution.

Pour `N(0,1)` : `φ(t) = e^{−t²/2}` (réelle, pas de partie imaginaire).

À partir de nos échantillons `u_1 … u_N` (les projections), la fonction caractéristique
empirique est simplement :

```
φ_N(t) = (1/N) Σ_j e^{i·t·u_j}
```

La statistique d'Epps-Pulley est la **distance L² pondérée** entre les deux :

```
EP = ∫ |φ_N(t) − e^{−t²/2}|² · w(t) dt
```

En développant (le terme cible est réel) :

```
|φ_N(t) − e^{−t²/2}|² = (Re φ_N(t) − e^{−t²/2})² + (Im φ_N(t))²
```

Le poids `w(t)` (gaussien) fait converger l'intégrale et concentre le test sur les fréquences
qui portent l'information.

**Pourquoi c'est puissant** : on compare la distribution *entière*, pas 2 moments. Une loi
bimodale de bonne moyenne et bonne variance a une fonction caractéristique qui **oscille**
différemment de celle d'une gaussienne, et EP le voit.

### 5.6 Mon implémentation

```python
# src/timejepa/training/utils/metrics.py :: sigreg_loss
u = z @ directions                      # [N, M] projections 1D
tu = t.view(-1,1,1) * u.unsqueeze(0)    # [Q, N, M]
re = cos(tu).mean(dim=1)                # Re φ_N   [Q, M]
im = sin(tu).mean(dim=1)                # Im φ_N   [Q, M]
integrand = ((re − exp(−t²/2))² + im²) · exp(−t²/2)
EP = 2 · trapèze(integrand sur t ∈ [0, t_max])
loss = EP.mean()                        # moyenne sur les M directions
```

Trois choix d'implémentation à connaître :

- **Intégration sur `[0, t_max]` puis ×2.** L'intégrande est paire en `t` : `Re φ_N` est paire,
  `Im φ_N` est impaire donc `Im²` est paire. On économise la moitié du calcul.
- **`t_max = 5`.** Le poids `e^{−t²/2}` vaut `4·10⁻⁶` en `t=5` — au-delà la contribution est
  négligeable.
- **`max_tokens = 8192`.** Le coût est en `O(Q·N·M)`. Avec B=512 et 47 patchs, `N` vaudrait
  24k ; on sous-échantillonne pour borner la mémoire, l'estimateur reste à faible variance.

**La loss totale** :

```
L = MSE(prédiction, cible)  +  λ · SIGReg(context_embeddings)
```

**Un seul hyperparamètre : `λ`.** À comparer aux trois poids de VICReg, qui interagissent et
qui étaient mal réglés (voire à 0) dans tes configs.

### 5.7 Ce que j'ai mesuré moi-même

Je n'ai pas repris les claims du papier sur parole. Sur des embeddings synthétiques dont je
contrôle la distribution :

| distribution | SIGReg | variance VICReg |
|---|---|---|
| **N(0,1) isotrope (la cible)** | **0.0006** | ~0 |
| collapsed (tout identique) | **1.282** | max |
| variance faible (0.1) | 0.393 | pénalise |
| variance forte (5) | 0.851 | ne pénalise pas (hinge unilatéral) |
| **bimodale (±3)** | **0.518** | **~0 — aveugle** |
| **queues lourdes** | **0.168** | **~0 — aveugle** |
| anisotrope (1 dim ×10) | 0.069 — **faible** | la covariance la capte |

Deux lectures :

1. **SIGReg voit ce que VICReg ne voit pas.** Bimodal et queues lourdes ont une variance par
   coordonnée correcte et une covariance quasi diagonale : VICReg les déclare saines. SIGReg
   les pénalise. C'est l'argument central en sa faveur.

2. **SIGReg est faible sur l'anisotropie mono-directionnelle.** Avec M=16 projections aléatoires
   en dimension 64, une pathologie confinée à une seule direction est rarement échantillonnée.
   Le terme de covariance de VICReg la capte directement.

> **Conclusion : ils sont complémentaires, pas concurrents.** C'est pourquoi les deux restent
> sélectionnables (`training.loss.type: vicreg | sigreg`) et pourquoi l'ablation P1.11 a un
> sens. Si l'ablation est ambiguë, une piste naturelle est de les combiner (SIGReg + le terme
> de covariance de VICReg) — mais je préfère qu'on mesure avant de complexifier.

### 5.8 Un bonus pratique

Le papier rapporte que la valeur de la loss SIGReg corrèle (r ≈ 0.8) avec la performance
downstream. Si ça se vérifie chez nous, ça résout un problème réel : ton `val_loss` est mesurée
sur des **séries held-out** (le split est par série, pas temporel — cf. B15) et ne prédit pas
la performance benchmark. Avoir un signal de sélection de modèle sans labels serait utile.
**Je n'ai pas vérifié cette corrélation** — c'est à observer sur les runs.

---

## 6. VICReg — ce qui était cassé

### 6.1 La variance était mesurée sur le mauvais axe

Les embeddings ont la forme `[B, N, D]` = (batch, position de patch, features). Le code faisait :

```python
pred_flat = predictions.reshape(-1, D)   # [B·N, D]  ← mélange batch ET position
std = pred_flat.std(dim=0)
```

En aplatissant batch et position ensemble, la variance mesurée inclut la variation **entre
positions de patch**, qui est naturellement grande (le patch 1 et le patch 11 sont différents
par construction, même pour un encodeur totalement collapsé).

**Conséquence : la diversité positionnelle seule suffit à satisfaire le hinge**, pendant que
les représentations collapsent *à position fixée* — c'est-à-dire exactement le mode de
défaillance qu'on cherche à détecter.

**Démonstration numérique.** Sur un tenseur construit pour être collapsé à chaque position
(tous les éléments du batch identiques à une position donnée, positions différentes entre
elles) :

| version | pénalité de variance |
|---|---|
| poolée (avant) | **0.0000** — ne voit rien |
| par position (après) | **0.9990** — quasi maximale |

**Fix** : `std(dim=0)` sur `[B, N, D]` → `[N, D]`, soit l'écart-type **à travers le batch, pour
chaque position**, puis moyenne. C'est ce que la variance est censée mesurer.

### 6.2 La moitié des termes n'avait pas de gradient

```python
var_loss_tgt = relu(1 − tgt_std).mean()      # targets vient de .detach()
cov_tgt = ...                                 # idem
var_loss = (var_loss_pred + var_loss_tgt) / 2
```

Les cibles sortent de l'encodeur EMA sous `no_grad` et sont `.detach()`-ées. Ces termes
contribuaient donc **exactement zéro gradient**, tout en gonflant la loss rapportée (et donc
en faussant early-stopping et `save_top_k`). Ils sont maintenant calculés sous `no_grad` et
retournés comme diagnostics uniquement.

### 6.3 L'encodeur n'était jamais contraint

La régularisation ne s'appliquait qu'à la sortie du **prédicteur**. Or la représentation qui
compte — celle que consomment le décodeur de forecasting et n'importe quelle sonde linéaire —
est la sortie de l'**encodeur** (`context_embeddings`).

`jepa_loss` accepte maintenant `context_embeddings` et applique la régularisation dessus
(`training.loss.regularize_context: true`).

### 6.4 Les configs étaient neutralisées

`base.yaml` et `large.yaml` avaient :

```yaml
variance_loss_weight: 0.0
covariance_loss_weight: 0.0
# et pas de clé invariance_loss_weight du tout
```

Donc : (a) `train.py:134` levait un `AttributeError` — ces configs n'avaient jamais pu
entraîner ; (b) même corrigées, VICReg dégénérait en MSE pure, **sans aucun terme
anti-collapse**. Et d'après les `eval_config.yaml`, tes runs `tiny` évalués tournaient aussi
en 0.0/0.0. Corrigé à 25/15/1.

---

## 7. Les cibles contextualisées (I-JEPA)

### Avant

```python
target_patches = patching(target_norm)          # 96 pas → 11 patchs
target_embeddings = target_encoder(target_patches)
```

L'encodeur cible voyait la fenêtre future **isolée** : 11 patchs, contre 47 pour l'encodeur
online. Deux problèmes :

1. **Décalage de distribution entre deux réseaux censés être une paire EMA.** L'encodeur cible
   est une moyenne mobile des poids de l'encodeur online, mais on l'applique à des entrées de
   longueur radicalement différente. Un transformer sur 11 tokens ne calcule pas la même chose
   que sur 47.
2. **Des cibles pauvres.** Une fenêtre de 96 points vue isolément, ce sont essentiellement des
   statistiques locales. C'est une cible faible à demander au prédicteur.

I-JEPA (le papier original, sur images) fait l'inverse : le target encoder voit **l'image
entière** et on *découpe* ensuite les blocs cibles dans la représentation.

### Après

```python
full_norm = cat([context_norm, target_norm], dim=1)     # 480 pas
full_embeddings = target_encoder(patching(full_norm))    # 59 patchs
target_embeddings = full_embeddings[:, -11:, :]          # slice
```

### Vérification de l'alignement

Il fallait s'assurer que les 11 derniers patchs de la fenêtre complète couvrent **exactement**
les mêmes pas de temps que les 11 patchs de la cible seule — sinon les cibles sont décalées
par rapport à ce qu'on demande au prédicteur.

```
fenêtre complète (480 pas) → 59 patchs, débuts aux pas 0, 8, …, 464
  les 11 derniers      : 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464
cible seule (96 pas)   : 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464
                          ✓ identiques
```

Vérifié par test automatisé. Les cibles couvrent les mêmes instants, mais sont désormais
**contextualisées** — l'encodeur cible sait ce qui précède.

Réglable par `training.contextualized_targets` (défaut `true`), donc ablatable.

---

## 8. Randomisation de géométrie

Réponse directe à l'expérience 2 du §4 : le modèle avait mémorisé 47 patchs.

### Implémentation

À chaque batch (**pas** à chaque échantillon) :

```python
L = choix aléatoire dans [128, 192, 256, 320, 384, 448, 512]
context = context[:, −L:]          # rognage par la GAUCHE

H = choix aléatoire dans [32, 64, 96, 128]
target = target[:, :H]             # rognage par la DROITE
```

### Trois décisions à noter

**Par batch et non par échantillon.** Ça garde tous les tenseurs rectangulaires — pas de
padding, pas de masque d'attention. C'est possible uniquement parce que l'encodeur est
**agnostique à la longueur** : il utilise RoPE et n'a aucune table de positions apprise.
Si on ajoutait un jour des embeddings positionnels absolus, ça casserait.

**Le contexte est rogné par la gauche.** On garde l'historique le plus récent, qui est ce
qu'un contexte plus court contiendrait réellement à l'inférence.

**La validation utilise toujours la géométrie native.** Sinon `val_loss` deviendrait bruitée et
non comparable entre époques. C'est un point important pour la sélection de modèle.

Les longueurs effectives sont loggées (`geometry/context_len`, `geometry/horizon_len`) — c'est
le contrôle qui permet de vérifier que la randomisation est bien active.

---

## 9. Multi-résolution réelle vs DRS

### Ce que faisait la DRS existante

```python
ctx_down = context[::factor]                        # décime
ctx_up   = interpolate(ctx_down, size=ctx_len)      # ré-interpole à la MÊME longueur
```

Lis bien : la fenêtre couvre **le même intervalle de temps** à la fin. C'est un **lissage** —
ça simule un capteur moins précis. Un cycle saisonnier occupe toujours le même nombre de pas,
donc **le ratio période/patch est inchangé**.

Elle ne pouvait donc pas, même de loin, corriger ce que mesure le §4. Le nom (« Diverse
Resolution Sampling ») suggérait pourtant exactement le contraire. C'est documenté honnêtement
dans le code maintenant.

*(Elle reste utile pour la robustesse à la qualité du capteur — je ne l'ai pas retirée.)*

### La vraie multi-résolution

Elle doit se faire au niveau du **Dataset**, parce qu'il faut accéder à la série brute, pas à
une fenêtre déjà découpée :

```python
# TimeSeriesDataset.get_item(allow_multi_resolution=True)
span = (ctx_len + pred_len) * factor
window = series[start : start + span : factor]     # prélève PLUS LONG, puis décime
```

On lit un morceau brut `factor` fois plus long et on le décime. Ça change **réellement** la
fréquence d'échantillonnage : une période de `m` pas devient `m/factor`.

### Vérification

Sur un signal synthétique de période 96 :

| facteur | période observée | cycles dans le contexte |
|---|---|---|
| 1 | 96 pas | 4.0 |
| 2 | 48 pas | 8.0 |
| 4 | 24 pas | 16.0 |

C'est précisément l'axe que le diagnostic a identifié comme déterminant.

### Détails d'implémentation

- Seuls les facteurs qui **tiennent** dans la série restante sont éligibles ; sinon repli sur 1.
  Pas de padding.
- Gated sur le split train via `AugmentedSubset.apply_augmentation` — val et test voient
  toujours la fréquence native.
- `__getitem__` ne l'applique **jamais** (compatibilité ascendante) ; seul `get_item(...,
  allow_multi_resolution=True)` le fait.
- Config : `data.multi_resolution_factors: [1,2,3,4]`, `data.p_multi_resolution: 0.35`.

---

## 10. Le bug du prédicteur

Trouvé en câblant l'horizon aléatoire. C'est le plus vicieux des trois bugs silencieux.

### Le mécanisme

`TransformerPredictor` avait une table de requêtes futures de taille fixe **16** :

```python
self.future_position_embedding = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
```

Utilisée ainsi :

```python
future_queries = self.future_position_embedding[:, :num_targets, :]
x = cat([context_embeddings, future_queries], dim=1)
... transformer ...
target_predictions = x[:, -num_targets:, :]
```

Si `num_targets = 23` :

1. `[:, :23, :]` sur une dimension de taille 16 → le slicing Python **renvoie 16 éléments, sans
   erreur**
2. `x` a donc `47 + 16 = 63` tokens
3. `x[:, -23:, :]` prend les 23 derniers de 63 → **7 embeddings de contexte + 16 vraies
   requêtes**

Les formes restent correctes de bout en bout (`[B, 23, D]` face à des cibles `[B, 23, D]`),
donc **rien ne plantait jamais**. Mais 7 des 23 « prédictions » étaient en réalité des
embeddings de contexte, entraînés et scorés comme des prédictions.

### Portée

`num_target_patches = (pred_len − patch) / stride + 1`

| config | patchs cibles | état |
|---|---|---|
| tiny / mini (pred 128, p16/s8) | 15 | ✅ |
| **large** (pred 192, p16/s8) | **23** | ❌ 7 faux |
| **base** (pred 128, p4/s4) | **32** | ❌ 16 faux |
| **checkpoints `*-512-196`** (pred 192) | **23** | ❌ |

### Corrélation avec les résultats

Ce n'est probablement pas un hasard :

| checkpoint | patchs cibles | MASE/SN |
|---|---|---|
| `best-unfreeze-1-stride-48-full-datasets` | 15 ✅ | **0.95** |
| `best-unfreeze-1-stride-48-restrained-datasets` | 15 ✅ | **0.95** |
| `best-reduced-datasets-late-…` | 23 ❌ | 0.98 |
| `best-reduced-datasets-early-…` | 23 ❌ | 0.99 |
| `best-full-datasets-…-512-196` | 23 ❌ | 1.00 |

Les deux seuls checkpoints sains sont les deux meilleurs. **Corrélation, pas preuve** — mais le
classement est cohérent avec le bug.

### Fix

- `_future_queries()` lève une `ValueError` explicite au lieu de tronquer
- `JEPATST` dimensionne la table depuis `prediction_length` avec de la marge :
  `max(16, int(num_target_patches * 1.5) + 4)`

---

## 11. Récapitulatif des fichiers touchés

| fichier | changement |
|---|---|
| `models/__init__.py`, `models/encoders/__init__.py` | imports `patchtst_encoder` réparés (B1) |
| `models/encoders/target_encoder.py` | `DualEncoderWrapper` sur `BareTransformerEncoder`, marqué deprecated |
| `models/components/revin.py` | `freeze/unfreeze`, `denormalize_target_space`, `to_input_frame` |
| `models/decoders/linear_decoder.py` | dénormalisation cohérente avec l'espace de la loss |
| `models/jepa_tst.py` | rollout réécrit, cibles contextualisées, dimensionnement de la table du prédicteur |
| `models/predictors/transformer_predictor.py` | refus de tronquer les requêtes futures (B16) |
| `training/utils/metrics.py` | SIGReg, VICReg corrigé, MASE/ND/WQL, `jepa_loss` étendue |
| `training/utils/baselines.py` | **nouveau** — seasonal-naive, naive-last, context-mean, linear-trend |
| `training/jepa_pretrain_module.py` | `_compute_loss` partagée, randomisation de géométrie, métriques de collapse |
| `data/dataset.py` | `get_item()` avec multi-résolution réelle |
| `data/datamodule.py` | propagation des paramètres multi-résolution |
| `data/augmentations.py` | DRS documentée pour ce qu'elle fait réellement |
| `scripts/train.py` | augmentations câblées (B5), SIGReg, géométrie |
| `scripts/evaluate.py` | `skip_revin=False`, baselines, MASE/WQL, caveat ETTh |
| `scripts/reevaluate_checkpoints.py` | **nouveau** — replay legacy vs fixed |
| `scripts/report_reevaluation.py` | **nouveau** — rapport avant/après |
| `scripts/diagnose_ettm.py` | **nouveau** — les deux expériences contrôlées |
| `configs/model/*.yaml` | poids de loss corrigés, blocs SIGReg / géométrie / multi-résolution |
| `tests/test_p0_regressions.py` | **nouveau** — 42 tests verrouillant tous les invariants |

**Aucun fichier supprimé.** Les tests visant l'API disparue sont marqués `pytest.mark.skip` avec
un motif explicite, et le code mort (`masking.py`, `DualEncoderWrapper`) est conservé et
documenté.

---

## Ce qu'il reste ouvert

Points où je n'ai **pas** de réponse et où ton jugement compte :

1. **Le caveat RevIN du §4** — je n'ai pas séparé « géométrie mémorisée » de « RevIN sur fenêtre
   plus longue ». L'expérience à faire : refaire la variation de contexte avec des statistiques
   RevIN gelées sur les 384 derniers points.

2. **λ pour SIGReg** — je l'ai mis à 1.0, ce qui est le défaut naturel mais que je n'ai pas
   calibré. Si les runs montrent que le terme SIGReg domine ou disparaît face à l'invariance,
   c'est le premier bouton à tourner.

3. **`p_multi_resolution: 0.35`** — choisi pour être substantiel sans dénaturer la distribution
   d'entraînement. Non optimisé.

4. **La corrélation loss/perf downstream** promise par LeJEPA — à observer, pas vérifiée.

5. **Les horizons 192/336/720** de la ré-évaluation n'ont pas été calculés (CPU trop lent ici).
   Le script reprend sur incrément, c'est à relancer sur la VM.
