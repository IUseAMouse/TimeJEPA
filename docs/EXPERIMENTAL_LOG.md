# TimeJEPA — registre expérimental

**Objet.** Trace des expériences menées, de leurs chiffres mesurés, et de ce que chacune
établit. Écrit pour servir de matière première à un article : chaque affirmation y est
rattachée à une mesure, et ce qui n'a PAS été mesuré est signalé comme tel.

**Ce document n'est pas** le plan de travail (`PLAN.md`, orienté tâches) ni la description
de l'architecture (`docs/TECHNICAL_OVERVIEW.md`). Les justifications détaillées de chaque
changement de code vivent dans les messages de commit de la branche `sota-roadmap`.

**Statut au 2026-08-13.** Le pari central est tranché, modestement : le pretraining
n'améliore pas l'ajustement au domaine d'aval mais améliore le **transfert** hors de ce
domaine (E11 puis E12, −8,7 % de MASE). En construisant l'ingestion de LOTSA, une réserve de
validité lourde est apparue — **le corpus d'entraînement contient deux des benchmarks**
(§5) — et le protocole bascule en conséquence : entraînement intégral sur LOTSA, évaluation
zero-shot. Le pretrain LOTSA (E13) n'a pas encore tourné.

---

## 1. Protocole de mesure

Ces conventions valent pour tous les chiffres du document. Elles ont changé au cours du
projet ; les résultats antérieurs à P0 ne sont **pas** comparables (voir §5).

| élément | choix | pourquoi |
|---|---|---|
| **MASE** | agrégation **poolée** (somme des erreurs / somme des erreurs du naïf saisonnier), pas la moyenne des ratios par fenêtre | une fenêtre plate met le dénominateur à ~0 et fait exploser la moyenne des ratios (mesuré : ~1e4 sur ETTm1). Le naïf saisonnier lit alors 0.99-1.00 comme attendu |
| **Skill vs SN** | `1 - MASE_modèle / MASE_SN`, > 0 = victoire | lisible, et centré sur la référence que la littérature considère non triviale |
| **WQL** | recalculé sur l'éventail complet de quantiles | pour un forecast ponctuel, WQL dégénère **exactement** en ND — d'où le champ séparé `wql_point` |
| **Couverture** | fraction des cibles dans la bande q10-q90, cible 80 % | seul chiffre qui distingue « les intervalles ont l'air corrects » de « les intervalles sont calibrés » |
| **Normalisation** | RevIN instance-wise, dénormalisation sans inverse affine | le décodeur est entraîné contre une cible z-scorée simple ; appliquer l'inverse affine introduit une erreur d'échelle de 6-10 % plus un offset |
| **Horizons** | 96 / 192 / 336 / 720, tronqués ou obtenus par rollout depuis l'horizon natif | protocole Nixtla long-horizon |

**Deux caveats à répéter dans tout article :**

1. **ETTh1 / ETTh2 ne sont pas comparables à la littérature.** `datasetsforecast.LongHorizon`
   ne livre qu'une seule série (`OT`) pour ces groupes, là où les tables publiées moyennent
   sur les 7 canaux ETT. À traiter comme une tâche univariée OT-only. L'avertissement est
   émis automatiquement à l'évaluation.
2. **Le nombre de fenêtres de test dépend de la longueur de contexte.** Un balayage de
   contexte ne compare donc pas des jeux strictement identiques. Sur les petits datasets
   (ETTh aux longs horizons : 2 à 6 fenêtres) c'est du bruit ; ne conclure que sur les
   gros (traffic, electricity, weather, ETTm : centaines à dizaines de milliers).

---

## 2. Registre chronologique

### E0 — Référence honnête (P0, 2026-08-09)

**Question.** Que valent réellement les modèles existants, une fois le protocole d'évaluation
réparé ?

**Ce qui a été réparé.** Normalisation à l'évaluation (B2), inverse affine RevIN (B3),
rollout (B10), MASE poolée, ajout des baselines (naïf saisonnier, naïf, moyenne, tendance
linéaire), métriques scale-free.

**Résultat.** Ré-évaluation de tous les checkpoints existants : **40/40 paires améliorées**,
MSE −42 % en moyenne — c'est-à-dire que les chiffres antérieurs sous-estimaient
systématiquement le modèle, par erreur de protocole et non par erreur de modèle.
Rapport : `docs/P0_REEVALUATION_REPORT.md`, détail dans `docs/P0_reevaluation_long.csv`.

**Établit.** Aucun résultat antérieur à P0 n'est utilisable. Toutes les comparaisons du
présent document partent d'ici.

---

### E1 — Diagnostic ETTm : cycles dans le contexte (2026-08-10)

**Question.** Pourquoi le modèle échoue-t-il sur ETTm alors qu'il réussit sur des séries
saisonnières comparables ? Hypothèse initiale de l'auteur : la taille de patch est mal
adaptée à la période saisonnière.

**Méthode.** Balayage de la longueur de contexte à l'évaluation, sur un modèle entraîné à
contexte 512, et comptage des cycles saisonniers contenus dans la fenêtre.

**Résultat.** La compétence suit le **nombre de cycles dans le contexte**, pas la taille de
patch : 5,3 cycles → échec, 21,3 cycles → victoire. Le balayage donne une courbe en cloche
dont l'optimum (640) coïncide avec 1,26× la longueur d'entraînement maximale — signature
d'un gain réel combattu par une pénalité hors distribution.

**Établit.** L'hypothèse « taille de patch » est **renversée**. Le levier est la quantité de
saisonnalité observable, indépendante du découpage. Confirmé plus tard par E5 (patch 16/8 et
32/16 à égalité).

---

### E2 — Régularisation anti-collapse : VICReg réparé et SIGReg (P1)

**Question.** Le pretrain JEPA est-il sain, et quel régularisateur anti-collapse utiliser ?

**Défaut trouvé.** Le VICReg implémenté calculait variance et covariance après
`reshape(-1, D)`, ce qui est **aveugle au collapse positionnel** : sur un cas construit où
toutes les positions d'un patch donné sont identiques, la variance poolée lit **0.9990**
quand la variance par position lit **0.0000**. De plus, les poids variance/covariance
étaient à 0.0 dans les configs — donc tous les runs historiques « VICReg » étaient en
réalité une MSE pure.

**Ajouts.** SIGReg (LeJEPA, arXiv 2511.08544) : test d'Epps-Pulley sur M projections 1D
aléatoires (Cramér-Wold), un seul hyperparamètre λ. Sélectionnable en alternative à VICReg.

**Instrumentation.** `collapse/context_std` et `collapse/effective_rank`, avec calibration :
sur signal réel et modèle non entraîné, le rang effectif lit **3,7-4,8** (et non 128 — le
signal univarié est intrinsèquement bas-rang). L'alarme est l'approche de 1,0, pas une
valeur « basse ».

**Complémentarité mesurée.** SIGReg détecte les distributions bimodales et à queues lourdes
que le terme de covariance de VICReg laisse passer ; la covariance de VICReg détecte
l'anisotropie sur une direction unique que SIGReg échantillonne mal. Les deux ne sont pas
redondants sur le plan théorique — mais voir E5 pour leur effet downstream.

---

### E3 — Corpus de finetune : curaté vs complet (2026-08-10)

**Question.** Faut-il finetuner sur les 8 datasets « propres » ou sur les 24 ?

**Résultat.** Les 24 gagnent : **+12 points de skill en moyenne**, et surtout
**exchange passe de −65,6 % à −0,6 %** — le décodeur n'avait simplement jamais vu de marche
aléatoire.

**Établit.** La diversité du corpus de finetune domine sa propreté. Conclusion inscrite en
dur dans les configs (`datasets_finetune: ${data.datasets}`) après qu'un run l'ait
accidentellement contredite (voir §5, incident de configuration).

---

### E4 — Tête quantile non paramétrique (P2.1, 2026-08-11)

**Conception.** 9 niveaux (grille GIFT-Eval), perte pinball (convention GluonTS, facteur 2),
monotonie **par construction** (médiane + largeurs softplus cumulées vers l'extérieur, donc
aucun croisement possible). Variante retenue : **option B**, décodeur alimenté par
cross-attention sur les embeddings de contexte.

**Motivation théorique.** MASE repose sur la MAE, minimisée par la **médiane**
conditionnelle. Le pinball donne la médiane exacte ; Huber donne un objet intermédiaire
entre moyenne et médiane. Le gain est donc attendu même sur les métriques ponctuelles.

**Résultat.** **−16 % de WQL relatif** contre le décodeur ponctuel. Hétéroscédasticité
confirmée : les largeurs d'intervalle s'ordonnent selon la difficulté du dataset.
Exchange bascule en victoire, la médiane y convergeant vers la persistance.

**Défaut mesuré, à l'origine de E6.** Les intervalles **rétrécissent** avec l'horizon :
exchange h720 largeur 0,267 alors que l'incertitude vraie d'une marche aléatoire croît
en √h. Mécanisme identifié : le rollout réinjecte la trajectoire médiane, plus lisse que
toute trajectoire réelle.

---

### E5 — Round géométrie (2026-08-11/12) — **le résultat principal à ce jour**

**Quatre changements groupés**, comparés en bundle à la référence E0+E3+E4
(`patch32_quantile`, contexte 512, finetune sur 24) :

1. contexte d'entraînement étendu à 1024 (motivé par E1)
2. horizon natif 128 → 256 (moins de rolls : h720 passe de 6 à 3, h192 devient mono-passe)
3. retour au patch 16/8 (la motivation de patch32 ayant été renversée par E1)
4. randomisation du contexte **aussi au finetune** — l'encodeur voyait 128-512, le décodeur
   uniquement 512, donc tout balayage mélangeait deux effets hors distribution

**Trois arms, une variable chacun.** MASE moyenne sur les 7 datasets communs :

| dataset | référence | geo 16/8 (SIGReg) | geo VICReg | geo p32 (SIGReg) |
|---|---|---|---|---|
| traffic | 1.134 | **0.768** | 0.780 | 0.777 |
| weather | 0.997 | 0.967 | 0.967 | **0.960** |
| electricity | 1.250 | **1.029** | 1.090 | 1.057 |
| ettm2 | 1.334 | 1.231 | **1.225** | 1.232 |
| etth1 | 1.520 | 1.265 | 1.270 | **1.247** |
| ettm1 | 1.454 | 1.369 | **1.286** | 1.315 |
| etth2 | 1.864 | 1.722 | 1.667 | **1.528** |
| **moyenne** | **1.365** | 1.193 | 1.183 | **1.159** |

**Résultats.**
- **−13 à −15 % de MASE moyenne, uniforme** : les 7 datasets s'améliorent, aucun ne régresse.
- traffic à **0.768** (skill +46 % à h96) ; trois datasets sous 1,0.
- Le profil par horizon d'ETTh1 s'inverse (+39 % de skill à h720 sur l'arm p32) : le pari
  « moins de rolls » se lit directement.
- **Les trois arms tiennent en 3 %**, ce qui est du bruit compte tenu des 2 à 5 fenêtres
  d'ETTh2 aux longs horizons.

**Établit.**
- **La taille de patch n'est pas un levier de qualité** : p32 égale (nominalement dépasse)
  16/8 pour **4× moins d'attention** (63 tokens vs 127 à contexte 1024). Conséquence
  pratique : c'est l'arm à porter vers un corpus de grande échelle.
- **SIGReg et VICReg sont indiscernables en downstream.** SIGReg est retenu sur l'argument
  du nombre d'hyperparamètres, pas sur une supériorité mesurée.

**⚠️ Confusion assumée.** Ce bundle mélange les changements de géométrie **et** le correctif
B20 (l'encodeur se dégèle réellement au finetune pour la première fois du projet). Les deux
effets ne sont pas séparables avec ces runs. Les arms sont comparables entre eux, pas au
protocole historique.

---

### E6 — Rollout à éventail propagé (2026-08-11)

**Problème** (mesuré en E4) : les intervalles rétrécissent avec l'horizon.

**Mécanisme.** Le batch est étendu Q fois (une copie par niveau de quantile), la copie k
reçoit la trajectoire de niveau k du roll précédent et continue à son propre niveau
(**couplage comonotone**) ; l'éventail marginal est le tri par pas de temps des Q chemins.
Déterministe — les niveaux **sont** l'échantillon stratifié, aucun RNG. Coût B×Q sur les
rolls seulement.

**Hypothèse assumée, écrite dans le code.** Dépendance de rang parfaite entre rolls. La
dépendance réelle est plus faible, donc la méthode **borne par le haut** là où la
réinjection médiane bornait par le bas : un biais systématique de signe connu est remplacé
par un biais plus petit, de signe connu également.

**Validation.** Un modèle non entraîné ne peut pas démontrer l'accumulation (son éventail
ne dépend pas de son entrée). La plomberie est donc verrouillée par un décodeur factice qui
conditionne sur le niveau reçu (persistance + éventail fixe [−1, +1]) : les largeurs
doivent sortir **exactement 2 / 4 / 6 / 8** sur quatre rolls sous couplage, et **plates à 2**
sous réinjection médiane. Les deux sont assertés en test.

**Effet mesuré sur checkpoint réel** (même modèle, seul le rollout change) :

| dataset | largeurs 10-90 avant (h96→h720) | après |
|---|---|---|
| ettm1 | 1.65 → 1.54 ↘ | 1.82 → 2.39 → 2.83 → **3.14** ↗ |
| electricity | 0.96 → 0.72 ↘ | 0.96 → 1.09 → 1.36 → **2.03** ↗ |
| exchange | 0.57 → 0.27 ↘ | 0.59 → 0.98 → 1.20 → **1.56** ↗ |

WQL : gagne là où la sous-couverture dominait (exchange h720 −13 %, etth2 h720 −14 %), cède
un peu là où le couplage sur-élargit (ettm1 h192, traffic h720). Net positif ; le gain moyen
de l'éventail à h720 remonte de −4 % à **−16 %**.

---

### E7 — Balayage de contexte post-round (2026-08-12)

**Question.** L'optimum de contexte s'est-il déplacé maintenant que l'entraînement va
jusqu'à 1024 ? (Prédiction faite avant mesure : il devait monter vers 1024.)

**Résultat** (arm p32, MASE moyenne sur 7 datasets) :

| contexte | 512 | 640 | **768** | 1024 | 1280 |
|---|---|---|---|---|---|
| moyenne | 1.203 | 1.162 | **1.144** | 1.159 | 1.168 |

**La prédiction est partiellement fausse et le vrai résultat est ailleurs.** L'optimum monte
de 640 à 768, pas à 1024. Mais **la courbe s'aplatit** : 640-1280 tient en 2 %, 512-1280 en
5 %, là où la même expérience avant le round montrait des écarts de **−33 %** sur ETTm1.
Sur les gros datasets, chacun a un optimum différent et tous les écarts sont dans le bruit.

**Établit.** La randomisation du contexte achète de la **robustesse en longueur**, pas un
pic de performance. C'est la propriété qu'un modèle de fondation doit garantir — l'utilisateur
fournit l'historique qu'il a. Seule la pénalité à 512 subsiste (+5 %), cohérente avec E1.

**Point d'opération retenu : contexte 768.** Meilleure moyenne, meilleur bilan head-to-head
du balayage (6/8), seule longueur ≥ 640 où exchange est entièrement évaluable
(768 + 720 = 1488 < 1517) et où il obtient son meilleur score, et moins cher que 1024.
Deux premières à cette longueur : ETTm2 devient une victoire (1.182 vs SN 1.250) et ETTh2
atteint son meilleur niveau (1.419).

---

### E8 — Baseline sans pretraining (G4.5, 2026-08-12) — **résultat central, négatif**

**Question.** Le pretraining JEPA apporte-t-il quelque chose ? Contrôle : même architecture,
même budget, mêmes 24 datasets de finetune, mêmes hyperparamètres — seule différence,
l'initialisation est aléatoire au lieu d'être issue du pretrain.

**Résultat.** MASE moyenne à contexte 1024, contre l'arm pré-entraîné du round géométrie :

| dataset | geo 16/8 (pré-entraîné) | **scratch** |
|---|---|---|
| traffic | 0.768 | **0.760** |
| electricity | 1.029 | **1.014** |
| etth2 | 1.722 | **1.606** |
| ettm2 | 1.231 | **1.229** |
| ettm1 | **1.369** | 1.378 |
| etth1 | **1.265** | 1.322 |
| weather | **0.967** | 1.003 |
| **moyenne** | 1.193 | **1.187** |

**Égalité stricte** : 0,5 % d'écart en faveur du scratch, 4 datasets contre 3. Le scratch
produit le meilleur traffic du projet (0.760, skill +47 % à h96) depuis un checkpoint
d'**epoch 02**, et sa val_loss dépasse la meilleure val_loss antérieure.

**Trajectoire d'entraînement observée.** Départ plus rapide pour l'arm pré-entraîné, **écart
refermé à ~10k steps**, puis trajectoires superposées. C'est la signature d'un avantage
d'**optimisation** (bonne initialisation), pas de **représentation**.

**Ce que ce résultat établit — et ce qu'il n'établit pas.**

Il établit que, **dans ce régime**, le pretraining JEPA n'apporte rien au-delà de ce que
l'entraînement supervisé extrait des mêmes données.

Il **n'établit pas** que le pari du §7 est faux, et la raison est structurelle : le corpus de
pretrain et celui de finetune sont **les mêmes 24 datasets**. La proposition de valeur du SSL
est d'apprendre de données sur lesquelles on ne peut pas superviser ; quand les deux corpus
coïncident, le modèle supervisé voit exactement les mêmes octets et **l'égalité est le
résultat par défaut**. Cette expérience ne pouvait donc pas soutenir le pari ; elle pouvait
seulement le réfuter de façon spectaculaire (si le scratch avait nettement dominé), ce qui
n'est pas le cas.

Il n'établit rien non plus sur « encodeur vs décodeur-seul » : les deux bras utilisent la
**même** architecture encodeur + prédicteur + décodeur.

**Signal secondaire, à surveiller.** Le scratch est moins bien calibré sur etth2 : couverture
**39-41 %** contre 42-65 % pour les arms pré-entraînés (cible 80 %). Le pretraining laisse
peut-être une trace sur l'estimation d'incertitude même là où le point est identique.
Échantillon faible (2 à 9 fenêtres), à confirmer.

**Réserves de protocole.**
- Le scratch tourne en `full_finetune`, l'arm de comparaison en `gradual_unfreeze` post-B20.
  La comparaison à mode identique (`tiny_geo` en full finetune) est en cours ; l'auteur
  rapporte une trajectoire de val_loss superposée, ce qui laisse attendre la même conclusion.
- Checkpoints d'époques différentes (scratch epoch 02, geo epoch 05).
- Balayage de contexte non encore effectué sur le scratch.

**Conséquence sur la suite.** L'expérience discriminante devient le **régime pauvre en
données** (finetune sur un seul dataset, ou ~10 % des fenêtres) : c'est là que le SSL doit
payer s'il vaut quelque chose. Et **LOTSA devient plus décisif, pas moins** : ce serait le
premier régime où les données de pretrain dépassent massivement celles de finetune.

---

### E9 — Balayage de contexte sur le scratch (2026-08-12) — attribution de la robustesse

**Question.** La robustesse en longueur de contexte établie en E7 vient-elle du pretraining
à contexte variable, ou de la randomisation appliquée au finetune ?

**Résultat.** MASE moyenne sur les 7 datasets communs, arm scratch (aucun pretraining) :

| contexte | 512 | 640 | **768** | 1024 | 1280 |
|---|---|---|---|---|---|
| **scratch** | 1.250 | 1.197 | **1.182** | 1.187 | 1.221 |
| *rappel p32 pré-entraîné (E7)* | *1.203* | *1.162* | ***1.144*** | *1.159* | *1.168* |

**Même optimum (768), même forme, même platitude.** Le scratch atteint lui aussi son meilleur
bilan head-to-head à 768 (6/8), et ETTm2 y devient une victoire (1.216 contre 1.250 pour le
naïf saisonnier).

**Établit.** **La robustesse en longueur de contexte est attribuable à la randomisation
appliquée au FINETUNE, pas au pretraining.** Un modèle qui n'a jamais vu de pretraining
présente exactement la même insensibilité à la longueur d'historique. C'est une attribution
propre, et elle renforce le résultat E7 plutôt qu'elle ne l'affaiblit : la propriété est
obtenue par une modification du protocole de finetune, donc disponible pour n'importe quel
modèle, pré-entraîné ou non.

**Nuance mesurée.** Le scratch se dégrade davantage aux extrêmes : +5,8 % à 512 et +3,3 % à
1280 par rapport à son optimum, contre +5,2 % et +2,1 % pour l'arm pré-entraîné. Le
pretraining aide donc **marginalement** à l'extrapolation hors de la plage vue, sans plus.
Écart faible, à ne pas surinterpréter.

**Réserve.** Les deux colonnes ne sont pas un contraste à variable unique : l'arm de référence
disponible en balayage est p32 (patch 32/16) alors que le scratch est en 16/8. La comparaison
des NIVEAUX entre les deux lignes est donc indicative ; c'est la comparaison des FORMES qui
porte la conclusion. Le balayage de l'arm `tiny_geo` 16/8 pré-entraîné lèverait cette réserve.

---

### E10 — Composition réelle du corpus de pretrain (lecture de log, 2026-08-12)

**Fait, tiré du log du pretrain `tiny_geo`.** Sur les 23 datasets configurés, **6 sont
écartés au chargement** par le filtre de longueur (fenêtre requise 1280) : rain-temperature
(725), m4-hourly (1008), nn5-daily (791), wikipedia-weekly (114), fred-md (728),
rideshare (541). Ils figurent dans `data.datasets` mais **0 fenêtre** en est créée —
l'encodeur ne les a jamais vus, même partiellement.

**Conséquence exploitée.** Ces six datasets forment un corpus **held-out gratuit**, obtenu
comme effet de bord de la géométrie du pretrain et non par une exclusion manuelle. C'est le
support de G4.6 (m4-hourly + nn5-daily à contexte 512).

**Fait plus important, et sous-estimé jusqu'ici.** Le corpus effectif n'est pas
« 17 datasets diversifiés » : il est **dominé par deux sources**.

| dataset | fenêtres train | part du batch après échantillonnage T=0.5 |
|---|---|---|
| wind-farms-minutely | 20 609 609 | 24,8 % |
| london-smart-meters | 19 135 198 | 23,9 % |
| *les 15 autres réunis* | *~10 900 000* | *51,3 %* |

**40 M des 50 M de fenêtres viennent de deux datasets**, tous deux à très haute fréquence
(éolien à la minute, compteurs semi-horaires), et l'échantillonnage par température ne les
ramène qu'à **48,7 % du batch cumulé**. La « diversité » du pretrain est donc bien plus
faible que le nombre de datasets ne le suggère.

**Portée.** C'est une explication candidate — non testée — du transfert faible mesuré en E8 :
un encodeur dont la moitié du signal d'entraînement vient de deux régimes haute fréquence a
peu de raisons de transférer vers du quotidien ou du mensuel. À énoncer comme limite du
corpus dans tout article, et à traiter par P1.10 (pondération des datasets) ou par un corpus
réellement diversifié (LOTSA, G5).

**Note de protocole.** Le pretrain a tourné 5 époques, arrêt anticipé (patience 25 records),
meilleur checkpoint à l'époque 02 (`val_loss` 0.3626). Les époques suivantes n'ont pas
amélioré la validation.

---

### E11 — G4.6, transfert vers des données inédites (2026-08-12) — **premier signal positif**

**Question.** Le pretraining transfère-t-il vers des données que l'encodeur n'a jamais vues,
en régime de faibles données d'aval ? (E8 avait mesuré une égalité à corpus de finetune
complet, mais ne pouvait pas trancher — pretrain et finetune y partageaient le même corpus.)

**Protocole.** Finetune sur **m4-hourly + nn5-daily**, les deux datasets écartés du pretrain
par le filtre de longueur (E10), à ctx 512 / pred 256. ~12 600 fenêtres au lieu de 8 M.
Configs jumelles par héritage, seule variable : présence ou non des poids pré-entraînés.
Évaluation sur les benchmarks Nixtla — inédits pour les deux arms.

**Résultat.** MASE moyenne, 7 datasets communs :

| dataset | **pré-entraîné** | scratch | écart |
|---|---|---|---|
| electricity | **1.486** | 3.076 | −52 % |
| traffic | **1.386** | 2.254 | −38 % |
| etth2 | **1.889** | 2.409 | −22 % |
| etth1 | **1.484** | 1.840 | −19 % |
| exchange | **25.48** | 31.26 | −18 % |
| ettm1 | **1.512** | 1.698 | −11 % |
| weather | **1.212** | 1.280 | −5 % |
| ettm2 | **1.324** | 1.386 | −4 % |
| **moyenne (7)** | **1.470** | 1.992 | **−26 %** |

**8 datasets sur 8** en faveur du pré-entraîné. Sur les datasets locaux l'écart est plus
frappant encore — traffic-hourly : R² **0,76** (corr 0,87) contre **0,095** (corr 0,33) ;
electricity-hourly MASE 1,35 contre 3,71. Le scratch n'a pratiquement rien appris.
Validation : 0,18 et encore descendante (epoch 38) contre 0,61 arrêtée (epoch 11).

**Ce que cela établit.** En régime **données inédites + peu de données d'aval**, le
pretraining apporte un gain massif. C'est le premier signal positif sur le pari central, et
il recadre E8 : l'égalité y était un **plafond** dû à la coïncidence des corpus, non une
absence de valeur. Corollaire direct : **LOTSA devient prioritaire**, puisque c'est le régime
où le pretraining paie.

**⚠️ Objection sérieuse, non levée : le budget en pas d'optimiseur.**
Avec ~12 600 fenêtres, batch 256 et `accumulate_grad_batches: 6` (hérité de tiny_geo), une
époque ne vaut que **~8 pas d'optimiseur**. Donc le pré-entraîné s'est arrêté vers **~310
pas** et le scratch vers **~90**. Un modèle from-scratch n'apprend essentiellement rien en 90
pas, et E8 a montré que l'écart se referme vers 10 k pas quand les données abondent. Il reste
donc possible que cette expérience mesure encore la **vitesse de convergence** plutôt que la
qualité atteignable. L'early stopping a bien déclenché, mais à ce régime de pas il peut
s'agir d'un plateau temporaire.

**Run de robustesse requis avant toute revendication publiable** — être généreux avec le
baseline :

```bash
python scripts/train.py --config-name tiny_geo_scratch_lowdata \
  trainer.accumulate_grad_batches=1 early_stopping.patience=100 \
  wandb.run_name=lowdata-scratch-long
```

(×6 pas d'optimiseur, patience très large.) Si le scratch reste loin derrière, la conclusion
devient incontestable ; s'il rattrape, l'effet est d'optimisation et non de représentation —
information tout aussi utile.

**Autres réserves.** Graine unique (l'effet est grand, mais une seconde graine reste une
assurance bon marché) ; nn5-daily n'apporte que ~330 des ~12 600 fenêtres, donc le finetune
est de fait dominé par m4-hourly.

---

### E12 — Run de robustesse : la revendication se resserre, et se précise (2026-08-12)

**Question.** Le gain d'E11 (−26 %) était-il un gain de qualité, ou l'artefact d'un baseline
étranglé par son budget (~90 pas d'optimiseur) ?

**Protocole.** Arm scratch relancé à l'identique avec `accumulate_grad_batches=1` et
`early_stopping.patience=100` — soit ×6 pas d'optimiseur et une marge très large. Tout le
reste inchangé.

**Résultat, en deux temps.**

*Sur la validation du finetune, le scratch PASSE DEVANT :* `val_loss` **0.1594** contre
**0.1807** pour le pré-entraîné. L'objection était donc fondée — le run court d'E11
sous-entraînait le baseline.

*Sur le benchmark, l'ordre ne s'inverse pas* (MASE moyenne) :

| dataset | pré-entraîné | scratch-long | écart |
|---|---|---|---|
| traffic | **1.386** | 1.880 | −26 % |
| weather | **1.212** | 1.281 | −5,4 % |
| etth2 | **1.889** | 1.975 | −4,4 % |
| etth1 | **1.484** | 1.545 | −3,9 % |
| exchange | **25.48** | 25.97 | −1,9 % |
| electricity | **1.486** | 1.491 | −0,3 % |
| **moyenne (5 hors exchange)** | **1.491** | 1.634 | **−8,7 %** |

*(les moyennes ettm1 / ettm2 manquaient dans la sortie transmise ; les horizons visibles les
placent très proches entre les deux arms.)*

Sur les datasets locaux : electricity-hourly MASE **1.35** contre **2.57**. Le scratch long a
bel et bien appris (R² 0,80 contre 0,095 pour le run court) — il reste deux fois moins bon.

**Ce que cela établit, et c'est le résultat le plus précis du projet.**
**Le scratch ajuste mieux le domaine d'aval ; le pré-entraîné généralise mieux en dehors.**
Avec assez de pas d'optimiseur, un modèle from-scratch se spécialise sur m4-hourly +
nn5-daily — et cette spécialisation coûte en transfert. Le pré-entraîné conserve une
structure qu'il n'abandonne pas.

La revendication défendable n'est donc ni « le pretraining converge plus vite » (E8) ni « le
pretraining bat le scratch en petit régime » (lecture naïve d'E11), mais :

> **Le pretraining n'améliore pas l'ajustement au domaine d'aval — il améliore le transfert
> hors de ce domaine.**

Formulation plus étroite, plus défendable, et rétrospectivement cohérente avec E8 : là-bas le
domaine d'aval contenait les benchmarks, il n'y avait donc rien à transférer.

**Corollaire méthodologique, à énoncer dans tout article.** La `val_loss` du finetune est un
**mauvais** critère de sélection pour un modèle de fondation : elle a désigné ici le modèle
qui généralise le moins bien. Seule l'évaluation hors domaine ordonne correctement.

**Honnêteté sur la magnitude.** L'ampleur passe de −26 % (E11) à **−8,7 %**, et à ~−3 % si
l'on retire traffic, qui porte l'essentiel du gain. Le signe est systématique (aucun dataset
en faveur du scratch), l'ampleur est modeste. À présenter ainsi.

**Deux points ouverts.**
- **traffic porte le résultat** (−26 % contre −0,3 % à −5 % ailleurs). Pourquoi ce
  dataset-là ? Non expliqué.
- **La calibration s'effondre chez les deux arms** en régime petit-données : couverture
  10-36 % (scratch) et 19-59 % (pré-entraîné) contre une cible de 80 %. Le régime casse les
  intervalles indépendamment du pretraining — motive G4.2 (calibration conforme).

---

### E13 — Ingestion de LOTSA et protocole zero-shot (2026-08-13) — *mis en place, non mesuré*

**Motivation, prédiction falsifiable.** E12 établit que le pretraining paie en **transfert**.
LOTSA (corpus de Moirai, ~27 Md d'observations, toutes fréquences) est donc le prolongement
direct, avec une prédiction testable : si le gain est bien un gain de transfert, un corpus
~1000× plus grand et réellement diversifié doit le creuser. S'il ne bouge pas, le plafond
n'est pas dans les données. E10 donne la seconde raison — le corpus actuel n'est diversifié
qu'en apparence (48,7 % du batch pour deux sources haute fréquence).

**Conversion.** Sortie **dense float32 exclusivement**, par segmentation en morceaux de
longueur fixe (8192) : les tableaux `object` cassent le copy-on-write de `fork` (B19) et
seraient rédhibitoires à cette échelle. Corollaire, les fichiers sont memmappables
(`data.use_mmap`, absent de toutes les autres configs donc sans effet sur elles). Coût
assumé : une fenêtre ne peut pas chevaucher deux morceaux, soit ~15 % des positions perdues
aux frontières. Plafond par sous-ensemble contre le déséquilibre d'E10.

**Exclusions, vérifiées sur la sortie réelle** (`--list`, 2026-08-13) : **47 sous-ensembles
exclus, 123 retenus**. Les motifs couvrent les benchmarks Nixtla (7) et les **28 répertoires
GIFT-Eval**, ces derniers confrontés au dépôt officiel
(`huggingface.co/api/datasets/Salesforce/GiftEval/tree/main`) plutôt qu'écrits de mémoire.

**Deux décisions prises en relisant cette sortie**, ce qui est la raison d'être de l'étape :
- `beijing_air_quality` et `china_air_quality` **exclus** : `kdd_cup_2018` (GIFT-Eval) EST la
  qualité de l'air à Pékin 2017-2018. Quasi-doublon d'un dataset d'évaluation.
- La famille **PEMS** (`PEMS03/04/07/08`, `PEMS_BAY`, `LOS_LOOP`, `largest_*`) **conservée**,
  et à déclarer explicitement dans tout article : elle partage le réseau routier californien
  avec le benchmark `traffic` mais aucun capteur, aucune année ni fréquence (PEMS-SF = 862
  capteurs horaires 2008-2009 ; les autres = 5 minutes, 2016-2021). Utiliser des données du
  même domaine issues d'autres sources est la pratique standard, mais traffic étant le
  résultat vedette, mieux vaut l'écrire que le laisser trouver.

**Protocole retenu.** Pretrain ET finetune sur LOTSA, évaluation zero-shot. L'architecture
impose un finetune (le décodeur est aléatoire après le pretrain), donc le faire sur Monash
rouvrirait la contamination du §5 — avec cibles supervisées cette fois. Entraîner les deux
étapes sur LOTSA rend les corpus disjoints **par construction**.

⚠️ **Limite connue** : la section « Local datasets » de l'évaluation porte sur Monash, dont
sept entrées ont un équivalent RETENU dans LOTSA (london-smart-meters, bitcoin,
wind-farms-minutely, rideshare, fred-md, sunspot-daily, melbourne-pedestrian-count). Ces
lignes sont de l'**in-domaine**, pas du zero-shot. La section Nixtla, elle, est intégralement
propre — c'est celle qui porte les chiffres publiables.

**Validé de bout en bout** : `--list` tourne contre le Hub, la conversion produit des
tableaux denses memmappables. Corpus obtenu : **83,3 M de fenêtres d'entraînement sur 112
sous-ensembles** (contre 50,6 M pour Monash), et le plus gros contributeur est ramené à
**7,1 % du batch** par l'échantillonnage par température — à comparer aux 48,7 % d'E10.
Volume : **~3-4 Md d'observations**, soit ~8-10× Monash, et non 1000× : les plafonds
prélèvent ~12 % de LOTSA. Ils achètent de la DIVERSITÉ plutôt que du volume, ce que E10
désigne comme la contrainte réellement mordante.

---

### E13a — Premier pretrain LOTSA : trois enseignements, aucun résultat downstream

Run `tiny` sur 3× RTX 3090, 3 époques, ~750 k pas, ~6 h l'époque. **Interrompu** : le
diagnostic ci-dessous montre qu'il ne pouvait pas converger.

**1. La calibration du rang effectif ne transfère pas d'un corpus à l'autre.**
Sur Monash, `collapse/effective_rank` lisait 3,7-4,8 à l'init. Sur LOTSA il lit **43 à 87** —
un ordre de grandeur au-dessus. La dimension intrinsèque des représentations suit la richesse
du corpus, ce qui valide au passage que LOTSA apporte bien de la diversité et pas seulement
du volume. **Conséquence : toute alarme « collapse » doit être calibrée PAR CORPUS.** Un 43
lu ici aurait été catastrophique sur Monash ; il est ordinaire sur LOTSA.

Diagnostic du run : rang 87 → 43 (contraction réelle), mais `context_std` **plat à 0,88-0,96**
et `pred_var` **non monotone** (0,63 → 0,48 → 0,57 — un collapse ne remonte pas). Verdict :
contraction modérée depuis un pic élevé, **pas de collapse**.

**2. La `val_loss` composite est polluée par son terme de régularisation.**
Décomposée, elle dit l'inverse de son agrégat :

| terme | trajectoire |
|---|---|
| `val_loss/mse` | 0,38 (pic 120 k) → **0,26 à 580 k** → 0,29 |
| `val_loss/sigreg_context` | 0,04 (creux 100 k) → **0,13** puis plateau |

La partie qui compte pour l'aval **a continué de s'améliorer jusqu'à 580 k**, bien après le
« plafond » apparent à 270 k. Écart train/val du terme SIGReg : **0,005 contre 0,13**, un
facteur 25 — attendu (les projections sont retirées à chaque batch) mais bruyant. C'est E12
sous une forme plus précise : la `val_loss` composite est un mauvais critère de sélection,
ici parce que son régularisateur dérive et noie le signal prédictif.

**3. Le scheduler doit être calibré sur le budget RÉEL, pas sur un défaut.**
`max_epochs` héritait 40 de `tiny.yaml` ; le cosinus était donc étalé sur dix jours de calcul
pour un run de trois époques. Mesuré : **le LR n'a décru que de 0,000749 à 0,000740 en
400 000 pas**. Le modèle a tourné à LR maximal du début à la fin.

Un modèle à LR constant maximal n'oscille pas autour d'un optimum : il ne s'y pose jamais.
Cela explique l'ensemble des symptômes — `val_loss` rebondissant entre 0,345 et 0,445 sans
tendance, `train_loss_epoch` descendant proprement (0,47 → 0,395), et un « meilleur
checkpoint » tombant sur un creux de bruit, **systématiquement vers le milieu de l'époque 1**.

**Corollaire opérationnel sur la sélection de checkpoint.**
- Run **recuit** (`max_epochs` = budget réel) : prendre le **dernier**. Le cosinus descend
  vers ~0, la fin EST le point de convergence, et le meilleur `val_loss` y coïncide.
- Run **non recuit** : ni le dernier ni le meilleur `val_loss` ne sont légitimes. Évaluer
  plusieurs checkpoints en aval et laisser le benchmark trancher (E12).

`max_epochs: 5` est désormais dans `lotsa_tiny` (hérité par `mini` et `base`), et **8 dans
`tiny_geo`** (hérité par p32, vicreg, scratch), avec le raisonnement en commentaire.

⚠️ **Portée rétrospective sur les résultats geo.** Les pretrains du round géométrie se sont
arrêtés à 5 époques sur 40 par early stopping : eux non plus n'ont donc **jamais été
recuits**. Les conclusions E5, E7 et E9 restent valides — tous les arms partageaient
exactement le même handicap, donc les comparaisons appariées tiennent — mais **les niveaux
absolus sont ceux de modèles jamais recuits**, et un run recuit devrait les améliorer sans
changer les classements. À énoncer comme tel dans un article, et à re-mesurer si un chiffre
absolu doit être publié.

**Aucun chiffre de performance downstream à ce stade.**

---

### E13b — Second pretrain, recuit : le corpus est épuisé en une époque

Même run relancé avec `max_epochs: 5`, donc un cosinus qui décroît réellement (0,0007 → 0,0002).
Le recuit a fait son travail — et a révélé ce que le bruit du premier run masquait.

**Train qui descend, validation qui monte.**

| | run 1 (LR constant) | run 2 (recuit) |
|---|---|---|
| `val_loss` | oscille 0,345-0,445 | 0,345 à 250 k → **monte à 0,51** |
| `val_loss/mse` | 0,26 à 580 k | 0,25 à 250 k → **remonte à 0,37** |
| `train_loss_epoch` | 0,47 → 0,395 | 0,475 → 0,428 |

Ce n'est plus une marche aléatoire : la divergence est monotone une fois le bruit du LR
constant retiré.

**La représentation se dégrade, elle ne compresse pas.** Tout se contracte — `target_var`
0,90 → 0,78, `context_var` 0,95 → 0,80, `pred_var` 0,58 → 0,47, `context_std` 0,925 → 0,84,
rang effectif 66 → 46 — **et le cosinus baisse aussi** (0,87 → 0,81). C'est la différence
décisive avec le run 1, où le cosinus MONTAIT pendant la contraction : une contraction avec
alignement préservé est défendable, une contraction qui perd l'alignement ne l'est pas.

Signature complémentaire : `val_loss/sigreg_context` monte à 0,17 alors que la version train
reste à **0,005** — un facteur 34. L'encodeur satisfait le régularisateur **uniquement sur la
distribution d'entraînement**.

**Les deux runs concordent : le meilleur checkpoint est à ~200-250 k pas, soit ~1 époque.**
Deux schedulers différents, même verdict — ce n'est donc pas un creux de bruit.

**Interprétation.** Les 83 M de fenêtres sont trompeuses : elles proviennent de **~800 k
morceaux distincts** (stride 8 sur des morceaux de 2048 = 97 fenêtres largement redondantes
chacun). Le contenu réellement indépendant est bien plus petit que le compte de fenêtres.
Le modèle épuise ce corpus en une passe, puis se spécialise.

**Ce que cela établit.** La réponse n'est pas plus d'époques mais **plus de données
distinctes** — ce qui valide, par la mesure et non par principe, la décision de passer au
corpus LOTSA complet (~20-25 Md d'observations après exclusions, contre ~3-4 Md ici).

**Deux corrections appliquées.**
- `max_oversample_ratio` 6,0 → 3,0 sur les configs LOTSA. Le défaut venait de Monash et ses
  24 datasets ; sur 112 sous-ensembles il faisait repasser des datasets de 648 échantillons
  **six fois par époque**. Contributeur direct au surapprentissage, gratuit à supprimer.
- `configs/model/lotsa_tiny_full.yaml` avec **`max_epochs: 1`** : sur le corpus complet une
  époque vaut des dizaines d'heures, et laisser 5 reproduirait le défaut d'E13a (un LR qui ne
  décroît jamais). Une passe unique recuite est aussi le régime standard des modèles de
  fondation.

⚠️ **À auditer après conversion sans plafonds** : rien ne garantit alors qu'une famille ne
domine pas, ce qui est exactement le défaut mesuré en E10. Script d'audit dans l'en-tête de
`lotsa_tiny_full.yaml` ; viser aucune famille au-dessus de ~15 %.

---

### E14 — Premier modèle zero-shot LOTSA : ETTm1 cède enfin (2026-08-15)

**Protocole.** Pretrain ET finetune quantile sur LOTSA seul (corpus plafonné, ~3-4 Md
d'observations), évaluation à contexte 1024 sur des benchmarks **jamais vus à aucune étape**.
Premier résultat du projet dont la section Nixtla est du zero-shot authentique.

**MASE moyenne**, contre les meilleurs arms du round géométrie :

| dataset | geo 16/8 | p32 | **LOTSA 0-shot** | écart vs geo |
|---|---|---|---|---|
| **ettm1** | 1.369 | 1.315 | **1.139** | **−16,8 %** |
| etth2 | 1.722 | 1.528 | 1.645 | −4,5 % |
| etth1 | 1.265 | 1.247 | **1.215** | −3,9 % |
| ettm2 | 1.231 | 1.232 | **1.189** | −3,3 % |
| electricity | 1.029 | 1.056 | 1.029 | ±0 |
| traffic | **0.768** | 0.777 | 0.779 | +1,5 % |
| weather | 0.966 | **0.960** | 1.053 | +8,9 % |
| **moyenne** | 1.193 | 1.159 | **1.150** | **−3,6 %** |

**L'agrégat sous-estime le résultat, et la décomposition dit pourquoi.** Les arms geo étaient
finetunés sur les 24 datasets Monash — dont les équivalents d'electricity, traffic et
weather — et **contaminés** sur electricity et traffic (§5). Le modèle LOTSA n'a vu aucun des
sept.

| | geo (finetuné, parfois contaminé) | LOTSA (jamais vu) |
|---|---|---|
| electricity, traffic, weather — *geo à domicile* | **0.921** | 0.954 |
| etth1/2, ettm1/2 — *hors domaine des deux côtés* | 1.397 | **1.297 (−7,2 %)** |

**À terrain neutre LOTSA gagne de 7 % ; là où l'adversaire jouait à domicile avec une fuite
en prime, il fait jeu égal.**

**ETTm1 : la prédiction faite d'avance se vérifie.** Skill −37 % → **−8,4 %**, MASE −16,8 %.
C'était le dernier échec structurel du projet, présent depuis E0, insensible à la géométrie
(E5), au patch (E5), au contexte (E7), au régularisateur (E5). L'explication retenue était le
corpus sous-quotidien — Monash n'a essentiellement rien à 15 minutes — et le pari était que
LOTSA le débloquerait. C'est ce qui s'est produit. **C'est la confirmation la plus nette du
levier « données » de tout le projet.**

`weather` régresse de 8,9 % : le revers attendu du zero-shot, puisque geo le finetunait et
que LOTSA l'exclut.

**Monash local : 7 datasets sur 8 battus, en zero-shot intégral** (seul bitcoin résiste, ce
qui est attendu d'une marche aléatoire).
⚠️ **Mais ces marges sont flattées** : la table de saisonnalité ne couvre pas les datasets
Monash locaux, donc `m=1` et `seasonal_naive` **=** `naive_last` (4.640 identiques sur les
huit). Ces chiffres disent « meilleur que la persistance », pas « meilleur qu'une baseline
saisonnière ». Seule la section Nixtla porte les bonnes saisonnalités (m=96, 24, 144) et donc
des chiffres publiables. **Compléter la table de saisonnalité locale reste à faire.**

**Ce que cela établit.** Un modèle entraîné intégralement sur un corpus disjoint égale ou bat,
en zero-shot, des modèles finetunés sur le domaine cible — et résout au passage l'échec
structurel que quatre rounds d'ablations architecturales n'avaient pas entamé. C'est la
première fois que les chiffres du projet sont **comparables à la littérature** (Chronos,
Moirai, TimesFM, Toto), le protocole étant maintenant le leur.

**Réserves.** Corpus plafonné (~3-4 Md) et une seule graine ; le run sur corpus complet
(~20-25 Md) est la suite directe. Et le harness GIFT-Eval (P2.6) reste à écrire : sans lui,
aucune position dans un classement publié n'est mesurable.

---

### E15 — G6 : l'ablation d'objectif ne confirme PAS la thèse (2026-08-18)

**Protocole.** Deux pretrains LOTSA identiques en tout — corpus plafonné, géométrie, budget,
optimiseur, scheduler — sauf l'espace où le prédicteur est noté : latent (JEPA) contre patchs
futurs bruts (reconstruction). Puis le MÊME finetune zero-shot et la MÊME évaluation. Les deux
checkpoints comparés sont à l'epoch 4, `val_loss` 1.3454 (JEPA) contre 1.3507 (recon), soit
0,4 % d'écart : les bras sont appariés.

**MASE moyenne :** JEPA **1.150**, reconstruction **1.166** — JEPA meilleur de **1,4 %**.

| dataset | JEPA | recon | écart |
|---|---|---|---|
| electricity | 1.029 | **1.028** | −0,1 % |
| traffic | 0.779 | **0.776** | −0,4 % |
| ettm1 | **1.139** | 1.142 | +0,2 % |
| weather | **1.053** | 1.060 | +0,7 % |
| ettm2 | **1.189** | 1.206 | +1,4 % |
| etth2 | **1.645** | 1.684 | +2,4 % |
| etth1 | **1.215** | 1.265 | +4,1 % |

**Et c'est là qu'il faut résister à la moyenne.** Sur les 28 cellules (7 datasets × 4
horizons), **la reconstruction gagne 17 fois sur 28**, et l'écart MÉDIAN par cellule est de
**−0,5 % en sa faveur**. La moyenne (+1,1 %) est portée par une poignée de cellules :
etth1 h720 (+24,3 %), etth2 h336 (+11,7 %), weather h720 (+9,3 %).

C'est la signature d'une **égalité à queues lourdes**, pas d'une victoire. La reconstruction
est le plus souvent très légèrement meilleure, et occasionnellement bien pire.

**Le signal d'horizon, réel mais fragile.** La reconstruction gagne 11/14 des cellules aux
horizons courts (96, 192) et seulement 6/14 aux horizons longs (336, 720) — direction
conforme à la thèse (la reconstruction dépense sa capacité sur l'imprévisible, ce qui coûte
d'autant plus que l'horizon s'allonge). Mais l'ampleur ne tient pas : l'écart moyen à h720
passe de **5,1 % à 1,9 %** en retirant la seule cellule etth1. Avec une graine unique, ce
n'est pas une preuve.

**Monash local :** match nul, 4 datasets chacun. La reconstruction est nettement pire sur
bitcoin (40,9 contre 35,9), meilleure sur solar-10-minute et saugeenday.

**Ce que cela établit — et c'est un résultat négatif qu'il faut assumer.**
**L'ablation ne montre pas que l'extrapolation latente bat la reconstruction.** À corpus,
architecture et budget identiques, les deux objectifs sont à 1,4 % l'un de l'autre sur la
moyenne, et la reconstruction gagne la majorité des comparaisons appariées. **La thèse
centrale du projet n'est pas confirmée**, et une graine unique ne permet pas de trancher un
écart de cette taille.

Conséquence directe pour le papier : **l'objectif n'est pas la contribution.** Le résultat
solide reste E14 — un modèle de ~1 M paramètres, entraîné sur un corpus disjoint, qui égale
en zero-shot des modèles finetunés sur le domaine cible et résout ETTm1 par le corpus. C'est
autour de cela qu'il faut écrire, pas autour de « JEPA plutôt que reconstruction ».

**Sous-produit : la variance de représentation prédit mal le transfert.** Le bras recon n'a
aucune régularisation de l'encodeur et son `context_var` tombe de 0,80 à ~0,20 — pour 1,4 %
d'écart en aval. Cela renforce la réserve déjà au §4 sur le rang effectif : ces métriques
sont des alarmes d'effondrement, pas des prédicteurs de qualité. Et cela retire l'essentiel
de l'intérêt du troisième bras envisagé (reconstruction + SIGReg sur le contexte) : si la
variance ne porte pas le transfert, l'isoler n'apprend plus grand-chose.

**Ce qu'il faudrait pour conclure quoi que ce soit sur l'objectif.** Au moins 3 graines par
bras. Un écart de 1,4 % sur une graine est au niveau du bruit, et le seul signal qui
survivrait peut-être — la dégradation au long horizon — repose aujourd'hui sur deux cellules.

---

### E16 — GIFT-Eval : premier positionnement leaderboard, et le signal d'horizon se réplique (2026-08-18)

**Protocole.** Harness P2.6 (97 configs officielles, fenêtres, saisonnalités et métriques
transcrites du data.py officiel, 37 tests), agrégation leaderboard = moyenne géométrique des
ratios par config vs Seasonal Naive officiel. Les deux checkpoints d'E15, une passe GPU de
3 min 30 chacun. Réserves : notre harness (dérive de convention mesurée : 0 sur MASE, ~2,7 %
sur CRPS), une graine par arm.

**Positionnement — agrégats calculés par la MÊME formule depuis les CSV officiels du repo
gift-eval :**

| modèle | params | protocole | MASE ratio | CRPS ratio |
|---|---|---|---|---|
| TimesFM-2.0 | 500M | zero-shot* | 0.758 | 0.550 |
| Chronos-Bolt small | ~48M | zero-shot* | 0.822 | 0.577 |
| VisionTS | ~100M | zero-shot | 0.863 | 0.755 |
| Moirai small | 14M | zero-shot | 0.946 | 0.650 |
| **TimeJEPA lotsa_tiny_zs** | **~1M** | **zero-shot** | **0.979** | **0.677** |
| TiDE | — | supervisé par dataset | 1.091 | 0.772 |
| Naive | — | — | 1.270 | 1.591 |
| DeepAR | — | supervisé par dataset | 1.343 | 0.853 |

*(\* le leaderboard note lui-même des recoupements de pretrain pour certains gros modèles.)*

**Sous 1,0 sur les deux métriques, en zero-shot intégral, à ~1M de paramètres, sur un corpus
plafonné à ~12 % de LOTSA.** Le modèle bat des baselines SUPERVISÉES entraînées sur chaque
dataset cible (TiDE, DeepAR) et se tient à 3,3 points de Moirai-small — 14× ses paramètres,
même corpus. Le probabiliste porte le modèle : 84/97 configs sous 1,0 en CRPS (65/97 en MASE),
CRPS 0.677 devant VisionTS (0.755). Les échecs sont concentrés sur les fréquences
sous-représentées au pretrain (bizitobs 10S, solar 10T, M4 yearly à 19 pas de contexte) —
le même diagnostic qu'ETTm1 avant E14, et donc la prédiction pour le corpus complet.

**G6 sur GIFT-Eval — le duel d'objectifs à plus grande échelle :**

| | MASE ratio | CRPS ratio |
|---|---|---|
| JEPA (extrapolation latente) | **0.979** | 0.6766 |
| reconstruction | 1.003 | **0.6764** |

MASE : JEPA meilleur de 2,4 %, et la ligne qualitative compte — **JEPA bat le Seasonal Naive,
la reconstruction non** (1.003). CRPS : égalité parfaite. Par config, recon gagne 37/97 en
MASE, 40/97 en CRPS.

**Le signal d'horizon d'E15 se réplique — c'est le résultat de l'entrée.** Sur E15 il
reposait sur deux cellules ; ici il est monotone sur 97 configs, benchmark indépendant :

| terme | n | MASE JEPA | MASE recon | écart |
|---|---|---|---|---|
| short | 55 | 0.914 | 0.921 | +0,8 % |
| medium | 21 | 1.056 | 1.089 | +3,1 % |
| long | 21 | 1.084 | 1.155 | **+6,6 %** |

L'écart croît strictement avec l'horizon, dans la direction que prédit la thèse (la
reconstruction paie l'imprévisible, et l'imprévisible croît avec l'horizon). Deux benchmarks
indépendants, même direction, ampleur croissante. Toujours une graine par arm — la
formulation défendable passe de « non confirmé » (E15) à : **« l'extrapolation latente et la
reconstruction sont indiscernables à court horizon ; l'avantage de la latente croît avec
l'horizon (+0,8 % → +3,1 % → +6,6 % sur GIFT-Eval), répliqué sur deux benchmarks, une graine
par arm ».** C'est plus étroit que la thèse d'origine, et c'est mesuré.

**Ce que cela change pour le papier.** Le pitch cesse d'être « JEPA bat la reconstruction »
(faux en l'état) pour devenir : (1) un modèle ~1M en zero-shot au niveau de baselines
supervisées et à portée de Moirai-small/14M — E14+E16 ; (2) l'objectif latent gagne
spécifiquement AU LONG HORIZON — E15+E16 ; (3) le levier dominant est le corpus — E12+E14.
Prochain point de la courbe : tiny puis mini (~5M) sur le corpus complet (G7).

---

### E17 — Diagnostic compétitif : d'où vient l'écart au leaderboard (2026-08-18)

**Contexte.** E16 situait TimeJEPA-tiny à 0.677 CRPS. Le leaderboard HF complet (123 modèles,
contre la poignée du repo GitHub) le place **105e/123 en CRPS, 107e en MASE** — la classe des
petits modèles est bien plus dense qu'estimé : Toto-2.0-**4m** 0.524, Metamorph-**4.5M** 0.555,
FlowState-**9.1M** 0.502, Kairos-**10m** 0.554. t0-alpha (The Forecasting Company) est à 0.494
mais fait **102M** paramètres — ce n'est pas un concurrent de catégorie.

**Méthode.** Comparaison config par config contre Toto-2.0-4m (4,14M), le voisin le plus
gênant, en décomposant l'écart selon trois axes.

| axe | résultat |
|---|---|
| **terme** (short/medium/long) | ×1,28 / ×1,32 / ×1,29 — **plat** |
| **nombre de variates** | ×1,22 (univarié) / ×1,38 (2-7) / ×1,36 (>7) |
| **fréquence** | ×1,08 (5T) ×1,16 (H) … ×1,94 (10T) **×2,42 (10S)** |

**Ce que cela élimine.** L'écart n'est PAS un problème d'horizon ni de rollout : il est plat
sur les termes, et la corrélation entre profondeur de rollout et ratio CRPS vaut −0,03.
Hypothèse plausible, testée, morte.

**Ce que cela semblait établir** (corrigé plus bas) : un écart dominé par la couverture
fréquentielle, facteur 2,2 entre meilleure et pire fréquence. Le multivarié compte, mais
au second ordre (~13 % d'écart supplémentaire contre un facteur 2,2 pour la fréquence).

**Le résultat positif, et il est fort.** Sur les fréquences que le corpus couvre bien,
TimeJEPA à 1M **bat** Toto-2.0-4m : electricity/H aux trois termes (×0,91-0,96), m_dense/H aux
trois termes (×0,82-0,89), loop_seattle/5T/medium, solar/W. Le bucket horaire (n=31, le plus
gros) n'est qu'à ×1,16. **Là où les données sont là, 1M de paramètres égalent 4M.** Les pires
écarts sont tous sur des fréquences quasi absentes du corpus plafonné : bizitobs/10S (×4,09),
solar/10T (×2,77), us_births.

**Sur-exclusion mesurée.** `GiftEvalPretrain` (corpus de pretrain SANCTIONNÉ par le benchmark,
152 sous-ensembles) contient `solar_power`, `taxi_30min`, `kdd2022`, `LOS_LOOP`, `covid19_energy`
— que nos motifs excluent. Il ne contient PAS bizitobs, jena_weather, us_births, ett : les
autres modèles ne trichent donc pas, mais **nous jouons plus strictement que le benchmark ne
l'exige**, et nous perdons de la couverture pour rien.

**⚠️ CORRECTION (même jour) — la décomposition par fréquence était CONFONDUE.** Le bucket
« 10S » ne contient que bizitobs, le bucket « 10T » que solar et jena_weather : fréquence et
domaine y sont inséparables. Refait à DATASET CONSTANT (même dataset, plusieurs fréquences),
ce qui est le seul test propre de l'effet fréquence :

| dataset | écart CRPS par fréquence |
|---|---|
| solar | 10T ×2,31 · D ×1,12 · H ×1,10 · W ×0,80 |
| jena_weather | 10T ×1,62 · D ×1,26 · H ×1,17 |
| ett1 | 15T ×1,36 · H ×1,13 · D ×1,04 · W ×1,01 |
| ett2 | 15T ×1,32 · H ×1,12 · D ×1,06 |
| loop_seattle | **5T ×0,97** · H ×1,14 · D ×1,27 |
| bizitobs_l2c | **5T ×1,05** · H ×2,04 |

Lecture corrigée, en deux effets distincts :
1. **Faiblesse sub-horaire réelle mais ÉTROITE** — 10T et 15T dégradent systématiquement sur
   quatre datasets indépendants (solar, jena_weather, ett1, ett2). Mais **5T va très bien**
   (loop_seattle ×0,97 — on gagne, bitbrains ×1,14-1,21, bizitobs_l2c ×1,05). Ce n'est donc
   pas « haute fréquence = mauvais » : c'est une couverture de corpus précise. Le 5 minutes
   est bien servi par la famille PEMS ; le 10 et 15 minutes ne le sont pas, d'autant qu'on
   excluait solar_power et kdd2022 (éolien à 10 min).
2. **Écarts de DOMAINE, sans rapport avec la fréquence** — bizitobs ×2,47/×2,37 à 10S mais
   aussi ×2,04 à l'heure, une fréquence bien couverte : c'est du CloudOps, le domaine
   propriétaire de Datadog, dont Toto est issu. us_births ×2,07 sur D/M/W indifféremment.
   Aucune invariance d'échelle ne comblera ces deux-là.

L'affirmation « l'écart suit la couverture fréquentielle avec un facteur 2,2 » est donc à
remplacer par : **un effet fréquence réel et localisé sur 10T/15T, plus des écarts de domaine
sur bizitobs et us_births qui relèvent du corpus, pas de l'architecture.**

**Leviers identifiés, par ordre d'effet attendu.**
1. Corpus complet + arrêt de la sur-exclusion (aligner les motifs sur GiftEvalPretrain).
2. **Conditionnement fréquentiel** — FlowState (9,1M, 0.502) est « timescale-invariant » via un
   `scale_factor` explicite. C'est une réponse architecturale directe à notre mode d'échec.
3. **Données synthétiques** (CauKer chez FlowState, KernelSynth chez Chronos) — c'est P2.5 du
   plan, jamais fait : la façon de couvrir des fréquences qu'on n'a pas.
4. **Contexte plus long** — FlowState 2048-4096 contre nos 1024.
5. **Scaler robuste** — Toto utilise un arcsinh ; notre RevIN a une pathologie MESURÉE (plancher
   epsilon, cibles à 1000 sigma, cf. G6).
6. Multivarié — réel mais second ordre.

---

### E18 — tiny-full : le point corpus-complet de la courbe d'échelle (2026-08-21)

**Protocole.** Même architecture (~1M), même géométrie, même harness qu'E16. Une variable :
le corpus de pretrain passe du LOTSA plafonné (~1,7 Md, ~12 % de LOTSA) à `lotsa_full`
équilibré (10,05 Md, 65 fichiers). Pretrain 1 époque (ancrée plus-gros-fichier, ≈2,2 passes
corpus mesurées — cf. G10.1b), finetune 1 époque LOTSA complet, LR plafond 7,5e-4 hérité.
Éval GIFT à chaque checkpoint de validation (initiative utilisateur — la trace complète est
un sous-produit précieux, cf. lecture du LR plus bas).

**Résultat principal :**

| | corpus | MASE ratio | CRPS ratio |
|---|---|---|---|
| E16 (tiny) | plafonné ~1,7 Md | 0.979 | 0.677 |
| **E18 (tiny-full), checkpoint final recuit** | **10,05 Md** | **0.9685** | **0.6664** |

**Corpus ×6 ⇒ CRPS −1,6 %, MASE −1,1 pt.** Réel, répliqué sur toute la bande de fin de run —
et modeste : le levier données seul s'aplatit à cette capacité. C'est le coefficient de la
courbe d'échelle qu'on venait chercher (G7) : la suite passe par la recette (mix/xres, E19)
et la capacité (mini 5M, G7.4), pas par plus de LOTSA.

**La trace par checkpoint — et ce qu'elle a appris sur le LR.** Huit évals GIFT du même
finetune : 20 % → 0.988/0.678 ; ~50 % → 0.9634/**0.6658** ; 0.70 → 0.9788/0.6688 ; 0.75 →
0.9727/0.6692 ; 0.80 → 0.9642/0.6658 ; 0.85 → 0.9662/**0.6656** ; 0.90 → 0.9699/0.6662 ;
final → 0.9685/0.6664. Lecture : progrès net ~nul de 100k à 400k steps (bande d'oscillation
±0.01 CRPS, pic d'instabilité à ~240k), TOUTE la descente une fois le cosinus sous ~3e-4,
puis plateau convergé (quatre derniers points dans ±0.004/±0.0008 — le choix best-val-loss
vs final est du bruit). Conséquence actée : plafond LR **3e-4** pour les runs mix/xres,
pretrain et finetune (les 3 époques de finetune vivent alors entièrement dans le régime
empiriquement productif).

**Décomposition per-config (checkpoint 0.75, baseline SN officielle) — où vit l'écart.**
Médiane des ratios MASE **0.893**, 62/97 configs sous 1,0. Le geomean 0.97 est fait par une
queue de 16 configs à ratio >1.25 : sans elles, **0.853**. Cette queue est le trou
sub-horaire d'E17 (bizitobs 10S ×1.65-3.59, solar 10T ×2.1-2.4, electricity/ett 15T
×1.3-1.45, m4_hourly), PAS le bloc séries-courtes (geomean 0.952, covid compris — SN y est
pire que nous). Et MASE/CRPS échouent sur les MÊMES 16 configs : pas de dissociation
« bon spread / mauvais médian » — l'asymétrie agrégée 0.97 vs 0.67 est surtout un artefact
de la baseline (SN est un point forecast, son WQL est gonflé ; TTM-R3 montre le même profil
0.727/0.520). L'hypothèse « le médian lisse est la limite de l'objectif JEPA » est donc
NON soutenue par cette décomposition — le corpus l'explique mieux.

**Prédiction falsifiable pour E19 :** le run mix (synthétique sub-horaire ~8-9 % du batch)
doit comprimer précisément cette queue de 16 configs ; si l'hypothèse E17 est la bonne,
c'est le MASE agrégé qui bouge le plus. Ce qui SURVIT de la queue après mix rédige le cahier
des charges des familles synthétiques v2 (P2.5d).

**Positionnement.** 0.6664 laisse le modèle ~105/123 au leaderboard complet : Chronos_small
(0.663) à portée de bruit, Moirai_small (0.650) exige la queue. Gate P3 v0.1 (< 0.65) : pas
atteint — c'est le travail d'E19/E20.

**Décisions actées par ce résultat :** (1) G7.3b tranché — la base des runs mix/xres est
`lotsa_full` ; (2) plafond LR 3e-4 partout sur la file ; (3) budgets 2 époques pretrain
(plafond, coupe manuelle) / 3 époques finetune, cosinus étalé dessus.

---

### E18b — Sonde d'énergie : le latent JEPA sait juger des futurs, et le full finetune le désapprend (2026-08-21)

**Question.** Un JEPA pré-entraîné est implicitement une fonction d'énergie :
E(x,y) = ‖pred(enc(x)) − enc(y)‖², abaissée sur les vrais couples par l'objectif,
creusée par SIGReg. Cette énergie discrimine-t-elle réellement les futurs plausibles ?
(Veto avant toute lecture « proposer-juger-pondérer » et avant l'arm ErrorSignalJEPA.)

**Protocole** (`scripts/probe_energy.py`, CPU, lecture seule, pendant le pretrain mix) :
6 configs GIFT × 100 instances × 34 candidats (32 block-bootstrap de l'historique +
seasonal naive + LE VRAI futur), tous encodés selon la convention de cible du pretrain
([ctx‖candidat] aux stats du contexte, tranche des derniers patches). Témoins : rang
normalisé du vrai (hasard 0.50), fraction top-20 % (hasard 0.20), Spearman(E, MAE réel)
(hasard 0.00). Approximation dite : encodeur online au lieu de l'EMA.

| checkpoint | rang vrai (moy) | top-20 % | ρ(E, MAE) |
|---|---|---|---|
| pretrain lignée E16 (corpus plafonné) | 0.245 | 0.57 | 0.50–0.74 |
| **pretrain tiny-full (corpus complet)** | **0.235** | **0.60** | **0.62–0.77 (toutes configs)** |
| full finetune (MÊME pretrain tiny-full, pinball seule) | 0.409 | 0.33 | erratique (−0.32 à +0.53) |

**Résultat 1 — l'énergie a du signal, partout.** Sur le pretrain : rang moyen 0.245,
electricity/H à 0.093 (médiane 0.030 !), et surtout ρ(E, MAE) positif et fort sur les
6 configs — l'énergie suit la proximité réelle, pas un artefact. Y compris solar/10T
(rang 0.295, ρ 0.58), config de la QUEUE E18 : **le latent sait des choses sur solar
que le décodeur ne rend pas.** La lecture par énergie mérite d'être construite ;
c'est l'argument « le latent JEPA porte plus que son argmin », mesuré.

**Résultat 2 — le full finetune détruit l'alignement énergie.** Même sonde sur le
checkpoint finetuné : dégradation générale, sz_taxi SOUS le hasard (0.617) avec ρ
négatif — sur la config où le modèle éval le mieux (0.55-0.62 vs SN). La pinball
seule n'ancre plus pred(x) ≈ enc(y) ; le drift du full finetune (question ouverte de
la discussion décodeur) a maintenant un COÛT MESURÉ : il sacrifie la structure
énergétique du pretrain. Confondeur LEVÉ le soir même (sonde sur le pretrain tiny-full, même lignée que le
finetune dégradé) : le pretrain corpus-complet est le MEILLEUR des trois (0.235,
ρ 0.62-0.77 — le corpus ×6 améliore aussi l'énergie), et son propre finetune tombe à
0.409. L'attribution est propre : c'est bien le full finetune qui détruit l'alignement,
pas le corpus. sz_taxi : 0.379 au pretrain → 0.617 (sous le hasard) après finetune.

**Conséquences.** (1) Prototype re-notation/intervalles pondérés : légitimé, sur
checkpoints de PRETRAIN ; (2) l'ancrage du finetune (garder le terme d'invariance du
pretrain à petit poids λ dans la loss de finetune) monte d'un cran dans le backlog —
sans lui, tout ce que ESJEPA ou la re-notation construiront sur le latent sera érodé
au finetune ; (3) la sonde devient l'instrument standard de santé énergétique d'un
checkpoint (une commande, 6 configs).

**E18c — addendum : le checkpoint mix à ~70 % d'époque, NON RECUIT, est déjà le
meilleur juge des quatre** (sondé le soir même, deux conventions d'encodage des
candidats — le script a gagné un drapeau `--standalone-targets` car mix s'entraîne
en cibles standalone, audit C1) :

| checkpoint (convention de sonde) | rang vrai (moy) | notes |
|---|---|---|
| pretrain E16 (ctx) | 0.245 | |
| pretrain tiny-full (ctx) | 0.235 | |
| **mix ~70 % époque (ctx)** | **0.153** | electricity/H : rang MÉDIAN 0.000 |
| **mix ~70 % époque (standalone = sa convention)** | **0.212** | |
| finetune tiny-full (ctx = sa convention) | 0.409 | |

Lecture : (a) la recette mix (arcsinh + synthétique + cibles standalone) améliore
nettement le juge latent AVANT recuit et avant tout finetune — premier signal aval
de la recette, des jours avant l'éval GIFT ; electricity/15T (queue E18) passe de
0.24 à 0.15-0.18 sous les DEUX conventions. (b) Nuance d'honnêteté : le gain
spectaculaire sur solar (0.145) n'existe qu'en lecture contextualisée — dans sa
propre convention standalone, solar est quasi hasard (0.480, ρ négatif) ; un futur
solaire encodé SEUL (une nuit = un plateau) porte peu d'information, la
contextualisation est ce qui le rend jugeable. Conséquence pour le prototype : la
lecture par énergie devrait encoder les candidats CONTEXTUALISÉS à l'inférence,
même sur une lignée entraînée standalone — c'est un choix de lecture, pas de loss.
(c) Le classement inter-checkpoints du tableau principal est inchangé (tous sondés
dans leur propre convention ou à convention égale). Suivi à époque 1 PILE (sonde
appariée, mêmes candidats au bit près) : agrégats 0.153 -> 0.162 (ctx) et
0.212 -> 0.211 (standalone) — plateau FONCTIONNEL recouvrant le plateau de
val_loss ; seule vraie évolution, la pathologie solar-standalone se répare
(0.48 -> 0.39, ρ −0.25 -> 0.00) : réarrangement, pas progression de compétence.
Charge de la preuve posée pour l'époque 2 : le checkpoint recuit devra décrocher
du plateau (~0.12-0.13 ctx), sinon l'époque 2 n'aura été qu'une assurance —
vérifiable en 30 min de CPU avec la même sonde appariée.
VERDICT (2026-08-22, checkpoint à 80 % d'époque 2, LR ~7e-6 = recuit quasi fini) :
agrégat 0.161 — le plateau tient jusqu'au bout. Série complète appariée :
0.153 / 0.162 / 0.160 / 0.161 / 0.155 / 0.161 sur ~1,1 époque de train.
L'ÉPOQUE 2 N'A PAS PAYÉ en compétence fonctionnelle (la val_loss, elle,
descendait : dissociation confirmée entre les deux thermomètres). Décisions :
(a) arrêt à 80 % d'époque 2 acté, finetune lancé depuis CE last.ckpt (sain,
le plus recuit d'une famille d'équivalents — les evaluate_energy sur 3
checkpoints candidats ne départagent rien hors bruit, et le best-val-loss
n'a aucun avantage fonctionnel) ; (b) enseignement budget pour les runs
FUTURS (mini) : 1 époque recuite peut suffire (~19 h GPU économisées) ;
(c) ⚠️ MAIS xres garde le MÊME budget que mix (2 époques coupées ~80 % de
l'époque 2) — le duel E19 reste à une variable, l'objectif seul.

**E18d — prototype v0 du forecast par énergie, évalué (Nixtla local, même harnais
pour trois lecteurs).** `scripts/evaluate_energy.py` : 32 bootstraps + SN + drift,
encodage contextualisé, poids softmax sur énergies standardisées (v0 sans
température libre — AUCUN réglage sur le test), quantiles pondérés 9 niveaux.
Lecteurs : energy (checkpoint de PRETRAIN tiny-full, zéro entraînement aval),
decoder (checkpoint finetuné, la voie générative), snaive. h=96, fenêtres
non chevauchantes, MASE poolée et WQL du repo — comparaison par ratios intra-run.

| dataset | energy MASE (vs SN) | energy WQL (vs SN) | decoder WQL (vs SN) |
|---|---|---|---|
| ettm1 | 1.21x | **0.97x** | 0.92x |
| ettm2 | 1.00x | **0.79x** | 0.74x |
| etth1 | **0.96x** | **0.79x** | 0.70x |
| etth2 | 1.15x | **0.91x** | 0.92x |
| weather | 1.19x | **0.96x** | 0.76x |
| exchange | 1.26x | 1.18x | 0.77x |

Lecture honnête : (1) LA MÉCANIQUE MARCHE — des intervalles calibrés sortent d'un
pretrain nu, WQL sous le seasonal naive sur 5/6 datasets sans une seule époque
d'entraînement aval ; sur etth2 le fan énergie ÉGALE le fan du décodeur finetuné
(0.91x vs 0.92x). (2) Le décodeur garde l'avantage partout ailleurs — attendu et
écrit avant le run (une époque de finetune contre zéro). (3) Le point forecast
énergie est faible (médiane pondérée de recombinaisons ≈ information de niveau SN).
(4) exchange échoue exactement comme prédit : série à dérive, le bootstrap ne
propose que des recombinaisons du passé — la limite d'enveloppe, mesurée. Les trois
leviers v1, dans l'ordre : trajectoires du décodeur dans les candidats (règle
exchange), calibration de T en contexte (règle la largeur), K plus grand. Verdict :
la lecture énergie est un COMPLÉMENT crédible (intervalles quasi gratuits, hybride
possible), pas un remplaçant du décodeur — la file générative garde la priorité,
conformément à la décision utilisateur.

**E18e — l'hybride « le décodeur propose, le pretrain juge » (protocole utilisateur),
mesuré le soir même.** Les 9 trajectoires-quantiles du décodeur FINETUNÉ entrent dans
le pool de candidats, le checkpoint de PRETRAIN (alignement énergie intact, E18b)
pondère tout le monde. WQL vs snaive :

| dataset | energy | decoder | hybrid |
|---|---|---|---|
| ettm1 | 1.00x | 0.92x | 0.95x |
| ettm2 | 0.80x | 0.74x | 0.76x |
| etth1 | 0.76x | 0.70x | **0.69x** |
| etth2 | 0.92x | 0.92x | **0.88x** |
| weather | 0.96x | **0.76x** | 0.87x |
| exchange | 1.15x | **0.77x** | 0.96x |

Lecture : (1) l'hybride bat energy-seul PARTOUT — les chemins du décodeur réparent
notamment l'échec d'enveloppe d'exchange (1.15x -> 0.96x), mécanisme confirmé ;
(2) il bat le décodeur finetuné sur etth1 (aussi en MASE : 0.84x vs 0.86x) et
nettement sur etth2 (0.88x vs 0.92x) — premier cas où le tandem à deux checkpoints
améliore la voie générative seule ; (3) mais il la DILUE sur weather/exchange :
9 chemins de décodeur noyés parmi 43 candidats, pondération naïve sans température —
là où le décodeur est fort, le pool le tire vers le bas. L'arbitrage n'est pas
encore assez contrasté pour « choisir » le bon proposeur par série. Leviers v2,
par ordre : calibration de T en contexte (le contraste), pondération par SOURCE de
proposition, échantillonnage de chemins cohérents (copule sur le fan) au lieu des
trajectoires-quantiles marginales. Parallèle assumé avec l'ensembling agentique du
haut du leaderboard : plusieurs proposeurs, un arbitre — sauf que l'arbitre est ici
une distance dans l'espace latent appris, pas un orchestrateur externe. Backlog,
derrière la file générative.

**E18e-v2 — l'hypothèse d'échantillonnage (utilisateur), testée** : pool enrichi à
75 candidats — 48 bootstraps à DEUX échelles de blocs (cycle entier + tiers) et
16 chemins MC-dropout du décodeur (Dropout seuls basculés en train le temps des
forwards, dans le script uniquement — trajectoires épistémiques COHÉRENTES, ce que
les chemins-quantiles ne sont pas). Effet mesuré, v1 -> v2 de l'hybride :

| | ettm1 | ettm2 | etth1 | etth2 | weather | exchange |
|---|---|---|---|---|---|---|
| hybrid WQL (vs SN) | 0.95->0.95 | 0.76->0.75 | 0.69->**0.68** | 0.88->0.91 | 0.87->**0.82** | 0.96->**0.90** |
| hybrid MASE (vs SN) | 1.18->1.16 | 0.94->**0.92** | 0.84 | 1.11->1.10 | 1.07->**0.97** | 1.04->**0.98** |

Verdict : l'échantillonnage ÉTAIT une partie du problème — la dilution weather se
resserre (0.87->0.82), exchange passe sous SN en point (0.98) — et en MASE l'hybride
bat maintenant le décodeur finetuné sur 4/6 datasets (ettm1, ettm2, etth1, etth2) :
LE VÉRIFICATEUR AMÉLIORE LE POINT FORECAST DU GÉNÉRATIF sur la majorité du banc.
Mais le décodeur garde weather/exchange en WQL (0.76/0.77 vs 0.82/0.90) : le résidu
n'est plus l'échantillonnage, c'est le CONTRASTE du juge — softmax non calibré, les
bons chemins ne dominent pas assez le pool. La calibration de T en contexte est
confirmée comme dernier levier v3, et comme prérequis (a) de G12.

**E18f — raffinement des candidats par gradient de E (« planning by backprop »),
deux bras appariés** (`--refine-steps`, script seulement) : à réglage doux
(3 pas, lr 0.05), résultat NUL — tous les deltas ≤ 0.01x, les gradients d'entrée
ne déplacent pas les candidats. À réglage fort (10 pas, lr 0.5, exchange) :
gain petit mais réel et SANS dégradation — hybrid MASE 0.97x -> 0.95x (2.533,
passe DEVANT le décodeur seul 2.544), WQL 0.89x -> 0.87x. Lecture : le paysage
n'est pas plat mais ses pentes sont douces à l'échelle des candidats — le
raffinement a une fenêtre utile avant Goodhart (aucun signe adversarial à
lr 0.5), et vaut ~1-2 % là où il compte. Levier d'appoint, derrière la
calibration de T ; coût x(1+2·pas) en forwards. Clos pour l'instant.

**E18g — G12(b) exécuté : premier proposeur EXTERNE, TTM-R3 (le SOTA sub-10M, 0.520
GIFT) sous notre juge.** `--proposer-ttm` dans evaluate_energy.py (granite-tsfm,
révision `1024-96-r3` — ⚠️ `main` charge une tête RÉINITIALISÉE, piège documenté) :
chemin propre + 4 contextes jitterés, dans le pool bootstrap+SN+drift, pondéré par
le pretrain tiny-full. WQL vs SN, mêmes fenêtres : etth1 0.85 -> 0.72 (−13 pts, MASE
aussi devant) ; etth2 0.95 -> 0.88 ; ettm2 0.79 -> 0.74 ; ettm1 0.86 -> 0.84 ;
weather 0.70 -> 0.85 (dilution) ; exchange 0.88 -> 0.97 (dilution).
LA PHRASE MESURÉE : un pretrain JEPA 1M, zéro entraînement dédié, améliore le CRPS
de TTM-R3 sur 4/6 datasets (jusqu'à −13 pts). Honnêteté : cette révision TTM est
point-forecast (WQL=ND) — une part de l'uplift est « ajouter des intervalles à un
point forecaster », ce qui EST la proposition de valeur du vérificateur. Les 2
échecs sont la RÉPLICATION n°3 de la signature de dilution (weather/exchange, même
paire qu'avec notre décodeur E18e et l'échantillonnage enrichi E18e-v2) : trois
proposeurs, même mécanisme — le softmax non calibré ne se concentre pas quand un
proposeur domine. La calibration de T bloque désormais TROIS victoires mesurées.

**E18h — calibration de T en contexte, v1 mesurée : neutre, et deux diagnostics
précieux.** `--calibrate-T` dans evaluate_energy.py : T par SÉRIE et par composition
de pool, choisi sur une grille {0.125..4} en rejouant le pipeline complet (proposeur
TTM compris) sur n_cal=2 sous-fenêtres passées du contexte, pinball minimale gagne ;
rng dédié -> tirages principaux appariés au bit près avec E18g. Résultat (WQL vs SN,
hybrid_ttm) : les 4 victoires sur TTM-R3 CONSERVÉES (etth1 0.72, etth2 0.88, ettm1
0.85, ettm2 0.77), exchange à moitié réparé (0.97 -> 0.94, TTM 0.88), weather PAS
réparé (0.85 -> 0.87, TTM 0.70).
Diagnostic 1 — ESTIMATEUR TROP BRUITÉ : 2 pinballs par série pour départager 6
températures = sélection au bruit ; les histogrammes de T le montrent (weather :
T=2-4 prescrits sur la moitié des séries — l'OPPOSÉ du remède — pendant qu'exchange
choisit correctement 0.125). Le smoke avait montré le calibrateur CAPABLE de trouver
le bon T ; le run complet montre qu'il ne le trouve pas fiablement à n_cal=2.
Diagnostic 2 — LIMITE STRUCTURELLE de T scalaire : quand UN proposeur écrase le pool
(weather), T -> 0 fait dégénérer le fan vers ce seul chemin, c.-à-d. TTM seul en
point, intervalles effondrés — la température ne peut au mieux que s'EFFACER devant
le proposeur dominant, jamais faire mieux que lui. Le fix structurellement correct
est la PONDÉRATION PAR SOURCE (un prior par famille de proposeurs — bootstrap /
ancres / TTM — calibré en contexte, l'énergie arbitrant à l'intérieur de chaque
famille) : le proposeur fort garde la masse dorsale, le bootstrap ne fournit que
l'étalement. Promu levier n°1 de la voie G12 ; calibration v2 = pooling par dataset
des scores (déjà calculés, quasi gratuit) + n_cal plus grand + prior par source.
VERDICT DE LA CASCADE (paire appariée, périmètre etth1/etth2/weather/exchange,
20 fenêtres x 8 séries, raffinement 10 pas lr 0.5) : le raffinement par gradient
GÉNÉRALISE aux chemins TTM, sélectivement — nul là où les propositions sont déjà
au fond des vallées (etth1/etth2/weather : deltas ~0, AUCUNE dégradation Goodhart),
décisif là où elles en sont loin : **exchange bascule en victoire** (hybrid_ttm WQL
0.92 -> 0.86 contre TTM seul 0.88 ; MASE 0.96 -> 0.91) et le bras energy-seul y
signe le plus gros effet de toute la série E18f-h (MASE x1.21 -> x1.05, WQL x1.18
-> x1.05). Mécanisme : exchange est le dataset à DÉRIVE où le bootstrap ne sort pas
de l'enveloppe historique — la descente de gradient extrapole à sa place en tirant
les candidats vers la vallée. L'antidote mesuré de la limite d'enveloppe d'E18d.
BILAN G12 consolidé : l'hybride bat TTM-R3 sur **5/6 datasets** (4 par rerank
E18g + exchange par raffinement) ; seul weather résiste, et son remède est déjà
diagnostiqué (pondération par source, E18h) — prochaine et dernière brique de la
voie avant le papier court.

### E18i — Le finetune mix v2 diagnostiqué : instabilité de LR, pas overfit (recuit court, 2026-08-23)

**Le problème.** Le finetune mix v2 (3 époques, plafond 3e-4) a produit son champion à
25 % d'époque 1 (0.8955/0.6190 — poids PERDUS, évincés par save_top_k sur val_loss),
puis 300k steps de dégradation GIFT (0.906-0.916 / 0.62-0.67) pendant que la val_loss
restait PLATE (0.5879 -> 0.5873) et que la train_loss_step lissée restait PLATE aussi.
Overfit exigerait train ↓ pendant val ↑ : absent. Restait à discriminer « instabilité
au LR chaud » (récupérable à froid) de « dégradation structurelle » (non récupérable).

**Le test : recuit court depuis le survivant le plus proche du champion**
(`epoch00_valloss0.5874.ckpt`, baseline évaluée 0.9186/0.6357). LR 5e-5, cosinus
prévu sur 0.15 époque, warmup 0.005, arrêté à 20 % (stagnation avérée). Critère posé
avant : récupération à <= ~0.62 => instabilité ; pas de récupération => plus profond.

**Résultat (évals GIFT par checkpoint du recuit) :**

| budget recuit | val_loss | MASE | CRPS |
|---|---|---|---|
| 0 % (baseline) | 0.5874 | 0.9186 | 0.6357 |
| 5 % | 0.5865 | 0.9110 | **0.6272** |
| 10 % | 0.5859 | 0.9082 | 0.6406* (~0.629 corrigé) |
| 15 % | 0.5858 | **0.9066** | 0.6315 |
| 20 % | 0.5859-v1 | 0.9090 | 0.6306 |

*flare G8.4b : `bitbrains_fast_storage/5T/short` CRPS 2.812 (0.44-0.48 partout
ailleurs), médiane saine (MASE 0.763, la meilleure des évals) — UNE config quasi-nulle
a coûté ~1.3 pt d'agrégat. Idem car_parts 1.276 à 15 %. Le plancher relatif G8.4b
passe de backlog à « avant les chiffres finaux E19 ».

**Verdict : instabilité confirmée en signe, récupération partielle en amplitude.**
(1) À LR froid, val_loss ↓ ET GIFT ↑ simultanément — ce que 300k steps à LR chaud
n'ont jamais produit : le bassin n'était pas épuisé, le LR l'empêchait de s'y poser.
Aucune signature d'overfit nulle part -> **mini est débloqué**. (2) Le recuit récupère
~la moitié de l'écart au champion (0.0085/0.0167) immédiatement, puis erre dans la
bande 0.627-0.632 : le 0.6190 du champion contenait une part de loterie de marche
aléatoire — ne pas le poursuivre à coups de recuits. (3) Le pattern « gains à la
rampe » se RÉPLIQUE à LR 6x plus bas : meilleur point à 5 % du recuit comme le
champion était à 25 % du finetune — deuxième observation indépendante, deux échelles
de LR. MASE, elle, s'améliore de façon monotone sur 4 évals (0.9186 -> 0.9066).

**Doctrine adoptée (change les budgets de TOUTE la suite) :** les gains de finetune
de cette lignée tombent en quelques dizaines de milliers de steps puis le reste est
diffusion dans le plateau. Protocole E19 candidat, lancé dans la foulée : **1 époque
fraîche depuis le pretrain, plafond tête 8e-5, backbone x0.1 (8e-6, LRs discriminés —
la tête quantile part de zéro, le backbone part riche et est celui qui dérive),
cosinus recuit dans l'époque** (`timejepa_lotsa_tiny_mix_zs_1ep`). Prédiction
falsifiable posée avant : atterrissage 0.615-0.63 avec les meilleurs checkpoints en
FIN d'époque (LR froid) ; si le pic est encore à ~5-25 % puis errance malgré le
backbone à 8e-6, la promenade est intrinsèque à la loss et la sélection de checkpoint
par éval GIFT devient l'outil de production officiel. Habitude actée après
l'éviction du champion : `cp` immédiat de tout checkpoint couronné par une éval
vers `checkpoints/champions/`.

### E19 — La carte per-config du champion : la recette mix A comprimé la queue (2026-08-24)

**Protocole.** Décomposition des 97 configs du champion mix-1ep3e4@25 % (0.8914/0.6134,
harnais enveloppe), convention leaderboard (ratios vs SN officiel), diff contre les
per-config OFFICIELS vendorés de 5 concurrents (`scripts/gift_gap.py`, snapshot 2026-08-22).

**1. La prédiction d'E18 est VÉRIFIÉE — la queue est passée de 16 à 6 configs.**
E18 prédisait : « le mix (synthétique sub-horaire) doit comprimer précisément la queue de
16 configs ». Mesuré : **seules 6 configs restent au-dessus de 1.0** (bizitobs_application/
service 10S long+medium ×1.55-2.02, electricity/15T/long 1.206, us_births/M 1.156).
Les anciennes coupables sont RENTRÉES DANS LE RANG : solar/10T ×2.1-2.4 (E17) → **0.94-0.99** ;
ett/electricity 15T ×1.3-1.45 → 0.78-0.98 ; 10T geomean **0.549**, 15T 0.780, 5T 0.593.
Le synthétique v1 + arcsinh ont fait leur travail sur 10T/15T. Queue(16) résiduelle ≈ 7 pts
de geomean (0.6134 → 0.5424 sans elle), contre ~12 pts en E18.

**2. Le noyau dur restant a un nom : bizitobs — un problème de DOMAINE, plus de fréquence.**
10S est la seule fréquence rouge (geomean 1.387, MASE 2.385) ; et contre TTM/Toto/FlowState,
bizitobs_l2c perd aussi en 5T et à l'HEURE (×1.7-2.8) — c'est le domaine IT-ops/CloudOps qui
manque, pas la grille. Piste croisée avec G10.2 : alibaba_cluster_trace (CloudOps 5T)
s'ÉTEIGNAIT à mi-époque dans l'ancien sampler — le rationnement le maintient présent, à
surveiller sur le run ration (indice au 15 % : bizitobs_application MASE 6.5 vs 11.2). Autres
résidus : bloc séries courtes A/Q/M (0.81-0.99, us_births/M 1.156, m4_yearly 0.989 —
cible --min-length 256) ; electricity/15T/long 1.206 (candidat contexte long : 1024 pts =
10.7 j à 15T, l'historique utile dépasse).

**3. La structure de l'écart aux concurrents a DEUX étages — et le what-if les sépare.**
- vs YingLong_6m (0.6090, 7.3M) : on gagne 44/97 configs ; queue→leur niveau = **0.5940**.
  Le barreau YingLong se prend avec la queue + un cheveu de corps.
- vs TTM-R3-PT/Toto-4m/FlowState (0.50-0.52) : on ne gagne que 17-23/97 ; queue→leur niveau
  = seulement **0.5637-0.5723**. L'écart à la classe 0.50 est LARGE, pas concentré : ils
  gagnent un peu partout (le corps de 81 configs vaut ~0.54 chez nous contre ~0.47 chez eux).
  ⇒ la queue paie le barreau YingLong ; la classe Toto exige des gains de CORPS
  (TTA/calibration/augmentations/densité de supervision) EN PLUS du corpus.
- Poches de fierté : solar/W ×0.43-0.75 contre TOUS ; m_dense/H bat Toto-4m ; bitbrains_rnd
  bat TTM et FlowState ; car_parts (ex-pire config) bat TTM.

**4. Termes plats confirmés** (short 0.611 / medium 0.619 / long 0.616) — l'horizon n'est
toujours pas l'axe du problème ; attente h512 recalibrée à « modeste, configs long hors
queue ».

**Prédictions v3 recalibrées par cette carte (remplacent celles du §0bis du PLAN) :**
P-v3.1 : bizitobs 10S geomean 1.387 → < 1.0 (familles synthétiques IT-ops : rafales,
zéro-inflation — aucune donnée publique n'existe) ; P-v3.2 : bloc A/Q/M/W −10 % via
--min-length 256 ; P-v3.3 : electricity/15T/long < 1.0 (contexte long ou données 15T
réelles ENTSO-E) ; P-v3.4 : agrégat ≤ 0.59 avant couches d'inférence. Le rationnement est
attendu comme co-acteur sur bizitobs (mécanisme alibaba ci-dessus).

### E19b — TTA uniforme : le miroir casse le mur des 0.60 (2026-08-24)

Trois procédures d'inférence mesurées UNE PAR UNE sur le champion (0.8914/0.6134), un seul
checkpoint, procédure identique sur les 97 configs (doctrine TTA validée par l'utilisateur) :

| procédure | MASE | CRPS | verdict |
|---|---|---|---|
| nu (référence) | 0.8914 | 0.6134 | 99e CRPS |
| multi-lookback {512,1024} | 0.8964 | 0.6228 | mitigé : CRPS −1.5 % mais MASE DÉGRADÉE — pas concluant seul |
| **miroir sign-flip** | **0.8735** | **0.5984** | **−2.0 %/−2.4 % ; passe sous Moirai_large/YingLong/Moirai_base/Reverso/TimeTron/tft → ~92e CRPS (+7 places), ~100e MASE. Coût ×2 forwards.** |
| ctx 2048 (G9.1, inférence seule) | 1.1898 | 0.8759 | échec global INSTRUCTIF (voir ci-dessous) |

**Le miroir** (`forecast(−x)` nié et renversé sur l'axe des niveaux, précédents TimesFM
`force_flip_invariance`/YingLong) : moyenner sur la transformation impose une équivariance
de signe que le modèle n'a qu'à moitié apprise, et réduit la variance d'erreurs
quasi indépendantes. Gains diffus sur tout le corps (loop_seattle/H 0.065 vs 0.071,
ett1/H long 0.263 vs 0.278, solar/10T ↓, m4_quarterly ↓) — exactement le « gain de corps »
que E19 réclamait pour la classe Toto.

**G9.1 tranché : le contexte long ne se prend PAS à l'inférence seule** — les configs
horaires s'effondrent (electricity/H MASE 4.5-5.0, solar/H ×3.0-3.7, m_dense/H ×3.5 :
255 patches jamais vus, RoPE hors distribution sur structures périodiques) — MAIS
bizitobs_l2c/5T long/medium : **CRPS divisé par 2** (0.314/0.252 vs 0.631/0.364) et
solar/10T/short −27 % : le modèle veut plus de contexte exactement sur la queue E19.
⇒ motivation chiffrée du CURRICULUM LONG-CONTEXTE au pretrain v3 (le levier est réel,
il s'achète à l'entraînement) ; ctx2048 EXCLU des moyennes TTA (pollution horaire).

**E19c — le tour du groupe de symétries (2026-08-25, idée utilisateur) :**
- **Échelle f(kx)=k·f(x) : NO-OP PROUVÉ, pas mesuré** — RobustScale+RevIN quotientent tout
  le groupe affine (médiane/MAD 1-homogènes ⇒ entrée normalisée bit-identique) ; seul le
  SIGNE échappe à la normalisation = le flip. Épinglé par test sur le vrai modèle
  (`test_scale_tta_is_a_provable_noop`). Même théorème que l'inertie de random_scale (T5),
  vu côté inférence.
- **Translation (+tta_shifts='1,3,5,7', impairs pour maximiser le changement de phase du
  grid — choix utilisateur) : NÉGATIF mesuré** — 0.8976/0.6269 vs 0.8950/0.6239 nu : la
  péremption (perdre les 1-7 points les plus récents) coûte plus que la décorrélation de
  phase ne rapporte. ARCHIVÉ comme le multi-lookback. Bug attrapé au passage : shift ≥
  horizon (m4_yearly h=6) cassait l'alignement — garde s<h + test de régression.
⇒ le groupe est épuisé : **flip seul reste la procédure officielle**. Équité vis-à-vis du
leaderboard (question utilisateur) : le TTA uniforme mono-checkpoint est dans les normes
déclarées du haut du classement (TimesFM expose force_flip_invariance en flag, YingLong
publie DCoT+multi-lookback, Toto-FnF est un ensemble de 10 modèles classé 1er) — notre
règle : TOUJOURS publier les deux chiffres (nu et ×flip) dans la table de fairness.

**VERDICT COMBINAISON (2026-08-24)** : flip+lb512-1024 = 0.8813/0.6063 — MOINS bon que le
flip seul (le multi-lookback dilue, cohérent avec sa mesure isolée).
**PROCÉDURE OFFICIELLE DU PROJET : miroir sign-flip seul** (`+tta_flip=true`), coût ×2
forwards, une ligne dans la table de fairness. Référence de soumission :
**champion × flip = MASE 0.8735 / CRPS 0.5984, ~92e CRPS / ~100e MASE.** Le multi-lookback
est ARCHIVÉ (ne revient que si un futur checkpoint entraîné à contextes variables le
réhabilite) ; ctx2048 attend le curriculum v3.

### E20 — Verdict final du run ration : le rationnement entre au protocole, la doctrine G7.3c re-confirmée (2026-08-25)

**Protocole.** Fin du finetune `timejepa_lotsa_tiny_mix_zs_1ep_ration` (1 ép. cosinus,
`save_top_k=-1`). Courbe de loss : descente PROGRESSIVE sur toute l'époque, meilleure
val_loss = DERNIER checkpoint — la stationnarité de composition supprime le sprint précoce
et la stagnation tardive de l'ancien sampler (prédiction G10.2 réalisée sur la loss aussi).
Les trois derniers checkpoints évalués nu + flip sur les 97 configs :

| checkpoint (val_loss) | nu MASE/CRPS | ×flip MASE/CRPS |
|---|---|---|
| 0.5857 | 0.8957 / 0.6307 | 0.8693 / 0.6016 |
| 0.5855 | 0.8954 / 0.6306 | 0.8685 / 0.6010 |
| **0.5855-v1 (dernier, meilleure val)** | 0.8955 / 0.6310 | 0.8692 / 0.6016 |
| *rappel : ration@45 % (champion)* | *0.8950 / 0.6239* | ***0.8702 / 0.5959*** |

**1. Le champion du run reste le 45 %.** Trajectoire GIFT nu complète : 0.6345(25 %) →
0.6305 → 0.6329 → 0.6303 → **0.6239(45 %)** → … → 0.6306-0.6310 (fin). Après 45 %, la
queue du cosinus ÉRODE GIFT (+0.007 nu) pendant que la val_loss continue de DESCENDRE.

**2. G7.3c re-confirmée une troisième fois, dans sa forme la plus pure.** Cette fois la
val_loss est propre (composition stationnaire, descente monotone) et elle désigne QUAND MÊME
le mauvais checkpoint : meilleure-val = dernier = 0.6310 nu, contre 0.6239 au 45 %. La
sélection par éval GIFT intermédiaire n'est pas un palliatif d'une val bruitée — c'est que
la val de finetune et le zero-shot GIFT ne mesurent pas la même chose. Doctrine intacte :
éval tous les 5-10 %, `cp champions/` immédiat.

**3. Fin de run = bassin plat.** Les trois derniers checkpoints sont quasi identiques
(écart 0.0004 nu / 0.0006 flip sur des dizaines de milliers d'instances) — le cosinus
terminal a convergé. Conséquence : une soupe des SEULS checkpoints tardifs est sans objet
(moyenner des points confondus) ; la soupe intéressante est celle de la FENÊTRE DE GAINS
(25-55 %), enfin possible avec save_top_k=-1. Prédiction avant mesure : la soupe atterrit
ENTRE le 45 % et la fin (~0.625-0.628 nu), ne bat pas le 45 % seul — les checkpoints d'un
walk-in-plateau sont des solutions différentes, pas des bruits autour d'une solution.

**VERDICT RATION : ENTRE AU PROTOCOLE.** Critère (fixé avant le run) : best-of-run ≥
champion sur la procédure officielle. Mesuré : ration@45 % × flip = **0.8702/0.5959**, bat
champion×flip (0.8735/0.5984) sur LES DEUX métriques (nu : CRPS en retrait 0.6239 vs
0.6134, MASE équivalente — le rationnement gagne là où on soumet). S'ajoutent le mécanisme
compris (G10.2), la loss propre, et l'amélioration tardive prédite puis observée.
`ration_oversample: true` devient DÉFAUT du protocole finetune (déjà inscrit dans les
configs v3). Champion inchangé : `champions/ration45_mase0.8702_crps0.5959.ckpt`.

**Addendum — soupe mesurée (2026-08-25 midi), prédiction confirmée et amplifiée.**
Soupe uniforme des 7 checkpoints de la fenêtre 25-55 % (identifiés par mtime + md5 du
champion — la loss n'était PAS monotone : remontée locale à 35 %, le tri par val_loss
mentait) : **0.9044/0.6383 nu, 0.8777/0.6072 ×flip** — pire que le 45 % seul (0.6239/0.5959)
et pire que la FIN de run (0.6310/0.6016). La prédiction disait « atterrit entre le 45 % et
la fin » ; la réalité est encore en dessous : les checkpoints du plateau ne sont pas des
bruits autour d'un bassin commun, la moyenne de leurs poids SORT du bassin (dégâts localisés
et grands : m4_weekly 2.76→4.13, covid 33.1→35.9 — signature d'interpolation entre solutions
non alignées). **VERDICT SWA : CLOS, négatif sur ce régime** — la soupe intra-finetune 1 ép.
ne revient que si un futur run montre une fenêtre de gains LONGUE et lisse (ex. multi-époques
v3). La sélection par éval intermédiaire reste l'unique mécanisme de récolte.

### E20b — Statistique appariée du signal d'horizon : l'avantage JEPA vit aux frontières de rollout, pas dans la profondeur (2026-08-25)

**Motivation.** Le « signal d'horizon » (E15 : 2 cellules ; E16 : +0.8/+3.1/+6.6 % monotone
sur GIFT) restait sans quantification d'incertitude — et c'est le pari fondateur (§7).
Zéro GPU : ré-analyse appariée des artefacts EXISTANTS des deux arms G6 (epoch04, JEPA
1.3454 vs recon 1.3507), script `horizon_stats.py` (bootstrap n=20000, permutation
intra-dataset).

**A. Intra-fenêtre (MAE par pas 0→255, 8 datasets Monash locaux) : le signal s'INVERSE.**
Pente du gap relatif recon-vs-JEPA : **−1.64 %/100 pas, IC95 % [−2.82, −0.56]**, 6/8
datasets en pente négative (solar −4.4, saugeen −3.6, wikipedia −2.7). La reconstruction
RATTRAPE avec la profondeur à l'intérieur d'une fenêtre. La lecture « la recon paie
l'imprévisible, d'autant plus que l'horizon s'allonge » est RÉFUTÉE comme effet de
profondeur de prédiction. (Contamination electricity/traffic sans objet ici : comparaison
appariée, les deux arms ont vu le même corpus.)

**B. À travers les rollouts (28 cellules Nixtla, h96/192/336/720 = 1 à 3 rollouts de 256) :
tendance réelle mais fragile.** Gap moyen par horizon : −0.09 % (h96), **−2.08 %
[−4.07, −0.48] (h192 : recon significativement MEILLEURE)**, +1.40 % (h336), +5.11 %
[−0.20, +12.08] (h720). Spearman gap~horizon **+0.404, p=0.019** (permutation
intra-dataset) — mais **p=0.069 sans etth1**, la même cellule dominante qu'E15.

**Reformulation de la thèse (plus étroite, mécaniste, falsifiable).** L'avantage latent
n'est PAS « prédire loin coûte moins cher en latent » (A le réfute) ; le candidat restant
est : **le latent se dégrade moins sous ITÉRATION sur ses propres sorties** (le gap ne
s'ouvre qu'au-delà d'un rollout). Corollaire testable en S3 : h512/h768 natif (moins de
rollouts) devrait RÉDUIRE l'avantage JEPA-vs-recon — le duel d'objectifs et la piste
horizon natif se testent mutuellement.

**Protocole enregistré (G6.2, le test du pari fondateur) :**
1. FAIT — la présente statistique.
2. GPU (~1,5 j, après ESJEPA) : contrôle **recon-mix** — rebaser `lotsa_tiny_recon` sur la
   recette moderne (mix, arcsinh, ration), pretrain 2 ép. + finetune protocole gelé. Le
   duel E15/E16 date du corpus plafonné ; jamais rejoué depuis. Prédictions AVANT le run :
   P-G6.2a gap h720 > gap h96 (le gradient de rollout se réplique) ; P-G6.2b pente
   intra-fenêtre ≤ 0 (le mécanisme profondeur reste mort) ; P-G6.2c le champion GIFT
   JEPA-mix bat recon-mix hors bruit (sinon l'objectif n'est pas un levier, point).
3. Mécanisme : instrumenter l'éval h720 pour l'erreur PAR PAS à travers les 3 rollouts —
   l'hypothèse frontière prédit des SAUTS du gap aux pas 256 et 512, pas une croissance
   continue.
4. 3 graines par arm : prix d'une conclusion de papier, S4+ seulement si (2) réplique.

### E20c — G13-T1/T2 mesurés : le juge connaît la flèche du temps, pas la continuité ; T2 invalide PAR CONSTRUCTION, redessiné (2026-08-25)

**T1 (checkpoint pré-spécifié mix/last, contextualisé, 8 instances × 16 candidats/famille) :**

| famille de violation | AUC vs cohérents (mix/last) | AUC (esjepa15_bestjudge) |
|---|---|---|
| renversement temporel | **0.808** (P-T1 ✓) | 0.603 |
| saut d'état initial ±2σ | **0.556 ≈ hasard (P-T1 ✗)** | 0.511 |
| réponse d'action inversée | 0.935 (caveat morphologie hors-gamme) | 1.000 |

Deux leçons : (1) **la continuité contexte→futur est quasi absente de l'énergie** — le
juge accepte un futur téléporté ; à retenir pour G13 (une contrainte de continuité devra
être explicite dans le coût, l'énergie ne la porte pas) ; (2) **le rang-probe ne prédit
pas la discrimination dynamique** — le « meilleur juge » par bootstrap (esjepa15, 0.205)
est PIRE sur renversement/saut que mix/last (0.291 en rang-probe). Troisième axe de
sélection de checkpoint, distinct du forecast ET du rang-probe.

**T2 v1 : P-T2 réfutée — mais procès reconnu truqué, deux vices de conception.**
Mesuré : gap d'optimisme ×3 et violations 10 %→30 % (38 % esjepa) AVEC l'énergie.
Autopsie : (a) déséquilibre d'échelle ~100:1 (coût ~0.005, λ_E·E ≈ 0.375 — l'objectif
était à ~99 % de la plausibilité pure, et le plan est tiré vers les consignes
HISTORIQUES 1.0/2.0 qui bordent la bande cible : le prior de politique comportementale
de Crasson/ThermoForce, MESURÉ en direct) ; (b) plus fondamental, le planning traversait
le simulateur VRAI — le régularisateur MPUR n'a de travail que contre l'erreur d'un
modèle FAUX ; avec le modèle parfait il ne peut qu'ajouter du biais. Le négatif v1 est
donc ininterprétable comme réfutation du mécanisme.

**T2 v2 MESURÉ (plan_beta=0.8, vrai 0.5 — cadre MPUR équitable) : P-T2b RÉFUTÉE,
proprement cette fois.** Deux runs qui ferment la question :
- e_ref=0.80 (terme DORMANT — la trajectoire imaginée sous le mauvais beta est
  morphologiquement plausible, E < seuil) : gap 0.1774 vs 0.1755, viol 78.7 % vs
  80.0 % — indiscernable. L'erreur de modèle vit dans l'écart imagination/réalité,
  un endroit que l'énergie DE l'imagination ne peut pas voir, par construction.
- e_ref=0.70 (terme ACTIF) : gap 0.1774 → **0.1979**, viol 78.7 % → **82.4 %** —
  forcée de s'exprimer, l'énergie injecte son prior « ressemble à l'historique »
  (la politique comportementale) et DÉGRADE le plan.

**VERDICT E20c (trois runs concordants) : E est un CLASSEUR, pas un RÉGULARISATEUR.**
Ce que l'énergie sait faire : ordonner des futurs candidats (flèche du temps 0.808,
morphologie hors-gamme 0.935) — la voie propose-juge-pondère d'`evaluate_energy`
(E18b/f, G12) reste intacte et validée. Ce qu'elle ne sait PAS faire : imposer la
continuité d'état (0.556), détecter l'erreur de modèle dans une imagination plausible
(T2b-dormant), servir de coût de prudence sans importer le prior de politique
(T2b-actif). Conséquences d'architecture G13, gravées : (1) la prudence du planning
viendra du FAN + z (l'incertitude PRÉDITE — ESJEPA) et/ou d'un désaccord d'ensemble,
pas de E — convergence mesurée avec le choix de conception de Henaff/MPUR (variance
d'ensemble, pas énergie) ; (2) la continuité d'état sera une contrainte EXPLICITE du
coût ; (3) la sélection du checkpoint-juge est un axe propre (ni val, ni GIFT, ni
rang-probe : discrimination dynamique).


---

## 3. Ce qui est établi

Affirmations soutenues par une mesure, avec le pointeur vers l'expérience.

1. **La compétence sur les séries saisonnières est gouvernée par le nombre de cycles dans
   la fenêtre de contexte**, pas par la taille de patch (E1, confirmé E5).
2. **La taille de patch n'est pas un levier de qualité** dans la plage 16-32 : 32/16 égale
   16/8 pour 4× moins de calcul d'attention (E5).
3. **SIGReg et VICReg (réparé) sont indiscernables en performance downstream** (E5). Leur
   complémentarité est théorique et instrumentale, pas mesurée en aval.
4. **La diversité du corpus de finetune domine sa propreté** : +12 points de skill,
   +65 points sur exchange (E3).
5. **Une tête quantile non paramétrique améliore aussi les métriques ponctuelles**, la MAE
   étant minimisée par la médiane que le pinball estime exactement (E4).
6. **Réduire le nombre de rolls et propager l'éventail corrige le rétrécissement des
   intervalles** aux longs horizons (E4 → E6).
7. **L'entraînement à contexte variable produit une robustesse en longueur** : ±5 % sur
   512-1280 contre −33 % avant (E7), et cette robustesse est attribuable à la randomisation
   côté **finetune**, pas au pretraining — un modèle sans pretraining présente la même
   courbe, même optimum (768) et même platitude (E9).
8. **Le bundle géométrique vaut −13 à −15 % de MASE**, gain uniforme sur 7 datasets (E5).
9. **Le pretraining n'améliore pas l'ajustement au domaine d'aval — il améliore le transfert
   hors de ce domaine.** À budget d'optimisation généreux, le scratch obtient une MEILLEURE
   `val_loss` de finetune (0.1594 vs 0.1807) et une MOINS BONNE performance sur tous les
   benchmarks (−8,7 % de MASE moyenne en faveur du pré-entraîné). Cohérent avec E8, où le
   domaine d'aval contenait les benchmarks (E11 + E12).
10. **La `val_loss` est un mauvais critère de sélection** pour un modèle de fondation, pour
   deux raisons distinctes : au finetune elle désigne le modèle qui généralise le moins bien
   (E12) ; au pretrain sa composante de régularisation dérive et noie le signal prédictif —
   mesuré, `val_loss/mse` continuait de s'améliorer 300 k pas après le « plafond » de
   l'agrégat (E13a).
11. **Le rang effectif doit être calibré par corpus** : 3,7-4,8 à l'init sur Monash contre
   43-87 sur LOTSA. Une même valeur signifie le collapse dans un cas et l'ordinaire dans
   l'autre (E13a).
12. **Le scheduler de LR doit être calibré sur le budget réel.** Un cosinus étalé sur 40
   époques pour un run de 3 laisse le LR à son maximum, et produit un plateau bruyant dont
   le « meilleur » checkpoint est un creux de bruit (E13a).
13. **Le corpus de pretrain est le levier de l'échec ETTm1**, là où l'architecture ne l'était
   pas : skill −37 % → −8,4 % en passant de Monash à LOTSA, après quatre rounds d'ablations
   géométriques sans effet sur ce dataset (E14).
14. **L'objectif de pretraining n'est PAS le levier** : à corpus, architecture et budget
   identiques, extrapolation latente et reconstruction sont à 1,4 % l'une de l'autre, la
   seconde gagnant la majorité des comparaisons appariées (E15). Le levier mesuré est le
   corpus, pas la loss.
15. **Un modèle entraîné sur un corpus disjoint égale en zero-shot des modèles finetunés sur
   le domaine cible** : −7,2 % de MASE là où le terrain est neutre, parité là où l'adversaire
   disposait du domaine et d'une fuite (E14).
16. **Le nombre de fenêtres n'est pas une mesure de la taille du corpus.** 83 M de fenêtres
   issues de ~800 k morceaux distincts s'épuisent en une époque : deux pretrains à
   schedulers différents désignent le même optimum vers 250 k pas, puis divergent
   (train ↓ / val ↑). Compter les échantillons INDÉPENDANTS (E13b).

## 4. Ce qui n'est PAS établi

À ne pas revendiquer sans mesure supplémentaire.

- **L'explication du fait que traffic porte l'essentiel du gain de transfert** (−26 % contre
  −0,3 % à −5 % ailleurs, E12). Non élucidé — et désormais suspect : `traffic-hourly` est
  dans le corpus d'entraînement ET est le benchmark `traffic` (voir §5, contamination).
- **Tout chiffre traffic ou electricity**, jusqu'à un run sur corpus disjoint (§5).
- **La généralité du résultat E12** : une seule graine, un seul couple de datasets d'aval
  (dominé par m4-hourly), un seul modèle tiny. La direction est systématique sur 6 datasets
  d'évaluation, l'ampleur est modeste (−8,7 %).
- **Que l'extrapolation latente batte la reconstruction** comme objectif de pretraining.
  ❌ **MESURÉ ET NON CONFIRMÉ (E15).** L'ablation G6 place les deux objectifs à 1,4 % l'un de
  l'autre, avec la reconstruction gagnante sur 17 des 28 comparaisons appariées et un écart
  médian par cellule en SA faveur. Ce n'est pas « JEPA perd » — c'est « on ne peut pas
  distinguer », ce qui suffit à retirer cette affirmation du papier.
  Attention au contresens qui a longtemps masqué le trou : E12 oppose le pré-entraînement à
  SON ABSENCE, pas à un objectif concurrent ; E12 et E14 établissent « pré-entraîner aide » et
  « plus de données aide », jamais « cette loss-ci aide ».
  Reste ouvert et testable : la dégradation de la reconstruction au LONG horizon (elle gagne
  11/14 des cellules à h≤192 et 6/14 à h≥336), conforme à la thèse mais portée par deux
  cellules sur une graine unique. Il faudrait ≥3 graines par bras pour en dire quoi que ce
  soit.
- **Le partage du gain de E5 entre géométrie et correctif B20.** Confondus par construction.
- **Toute comparaison à la littérature ETTh1/ETTh2** (série unique, voir §1).
- **Le rang effectif comme prédicteur de qualité downstream.** Instrumenté et calibré,
  jamais corrélé à une performance.
- **La pondération des datasets par prédictibilité** (P1.10) : jamais implémentée. E10 en
  renforce la motivation : deux datasets haute fréquence pèsent 48,7 % du batch de pretrain.
- **Que la concentration du corpus (E10) explique le transfert faible (E8)** : hypothèse
  plausible, non testée. Le test serait un pretrain à corpus rééquilibré.

---

## 5. Défauts qui ont invalidé des résultats

> **Risque assumé (2026-08-19)** — `beijing_air_quality` et `china_air_quality` sont au corpus
> de pretrain alors que GIFT-Eval évalue `kdd_cup_2018` (qualité de l'air, Pékin 2017-2018) :
> un chevauchement PARTIEL de fenêtres temporelles est concevable. Assumé parce que
> `GiftEvalPretrain`, le corpus de pré-entraînement publié par les auteurs du benchmark,
> contient les deux sous-ensembles — aucune entrée du leaderboard n'est pénalisée pour les
> avoir vus. À retirer (overrides + test, ensemble) si un relecteur le conteste.

Section essentielle pour un article : elle explique pourquoi les résultats antérieurs à
certaines dates ne sont pas comparables. Chaque entrée a son commit sur `sota-roadmap`.

| réf | défaut | portée de l'invalidation |
|---|---|---|
| **B2** | Normalisation globale z-score utilisée à l'évaluation là où le modèle attend une normalisation par instance | **tous** les résultats pré-P0 (MSE surestimée de ~42 %) |
| **B3** | Dénormalisation appliquant l'inverse affine RevIN, alors que le décodeur vise une cible z-scorée simple | erreur d'échelle 6-10 % + offset sur chaque forecast |
| **B10** | Rollout mélangeant espaces normalisé et brut ; `revin.freeze()` inexistant | tout horizon > horizon natif |
| **B16** | Table du prédicteur dimensionnée à 16 : les requêtes manquantes étaient **silencieusement** remplacées par des embeddings de contexte, les formes restant valides | corrélé au classement des checkpoints (propre 0.95 vs corrompus 0.98-1.00) |
| **B6** | VICReg calculé après pooling → aveugle au collapse positionnel (0.0000 vs 0.9990) ; poids variance/covariance à 0.0 | tous les runs « VICReg » historiques = MSE pure |
| **B5** | `augmentation_config` jamais transmis au DataModule | augmentations inertes sur tous les runs antérieurs |
| **B20** | `gradual_unfreeze` : l'optimiseur filtrait sur `requires_grad` à sa création (epoch 0, tout gelé), donc le dégel ultérieur ne mettait à jour **aucun** poids enregistré. Mesuré : 0/23 paramètres du prédicteur, 0/18 de l'encodeur dans l'optimiseur | **tous** les résultats historiques en mode gradual sont des probes à encodeur gelé — y compris les meilleurs checkpoints du 2ᵉ batch. Lecture : ils **sous-estiment** les représentations, aucun gain n'ayant jamais pu venir d'une adaptation de l'encodeur |
| **B17** | `SeriesTooShortError` non gérée : une série trop courte tuait un run de 23 datasets | robustesse, pas justesse |
| **B22** | `np.array(liste, dtype=object)` renvoie un tableau object **2-D** quand toutes les séries retenues ont la même longueur, et `np.stack` préserve ce dtype — la valeur atteint `torch.from_numpy`, qui refuse les tableaux object | tout dataset dont les survivants du filtre de longueur sont **uniformes**. Jamais déclenché avant parce que le filtre laissait toujours des longueurs mêlées ; apparu sur m4-hourly (1008 pas partout), c'est-à-dire **le dataset held-out de G4.6** — celui qui portait l'expérience décisive |
| **B21** | `auto_insert_metric_name` présent dans la config mais jamais transmis à `ModelCheckpoint` → noms de fichiers doublés contenant `=`, incompatibles avec la grammaire d'override Hydra | ergonomie ; a causé au moins un échec de commande |
| — | **Incident de configuration** : un finetune du round géométrie a tourné avec un décodeur ponctuel (défaut hérité `mlp`) sans qu'aucun signal ne l'indique. Détecté par le nombre de clés chargées (110 au lieu de 118) et l'absence du suffixe de couverture | une évaluation dont les colonnes « WQL » étaient en fait des ND ponctuels |
| — | **Incident de configuration** : un finetune a tourné sur la liste curatée de 8 datasets au lieu des 24, contredisant E3 | conservé comme arm d'ablation ft8 |
| — | **Piège de checkpointing n°2 (2026-08-26)** : sur le pod (Lightning 2.6.5), `last.ckpt` n'était PAS l'état courant mais une copie de la dernière sauvegarde top-k — avec une val plate, il pointait sur le checkpoint de ~35 % (0.5841). Détecté par la série probe : 18 valeurs bit-identiques entre « last » et 0.5841, confirmé par md5 (3 hashes égaux). L'utilisateur avait MESURÉ ce comportement ; démenti à tort sur la foi du code d'une AUTRE version (2.5.6 locale) — la mesure bat la lecture de code. Coupure du pretrain ESJEPA à 1,5 ép. ⇒ les poids tardifs sont PERDUS | le finetune ESJEPA part du best-val 0.5841 (~35 % du schedule, LR ~70 % du pic — sélection défendable sur une val plate, consignée comme 4e écart du duel). Parade mécanique : `save_top_k: -1` au pretrain (commité la veille pour v3/esjepa) — chaque val sauvegarde, le piège disparaît quelle que soit la version |

**⚠️ CONTAMINATION DU CORPUS PAR LES BENCHMARKS — la réserve la plus lourde du projet
(trouvée le 2026-08-13 en construisant la liste d'exclusion de LOTSA).**

Le corpus d'entraînement contient deux des huit benchmarks d'évaluation, sur les mêmes
séries et la même fenêtre temporelle :

| corpus (pretrain ET finetune) | benchmark | recoupement |
|---|---|---|
| `electricity-hourly` (321 séries × 26 304) | `electricity` (321 × 5 260) | 5 260/26 304 = les derniers 20 % |
| `traffic-hourly` (862 × 17 544) | `traffic` (862 × 3 508) | 3 508/17 544 = les derniers 20 % |

Le découpage train/val/test est `range(0, train_len)` sur des indices **groupés par série** :
l'entraînement prend donc les ~96 premiers pour cent des SÉRIES, **sur toute leur durée**,
période d'évaluation comprise. Le modèle a vu ces valeurs cibles au pretrain (auto-supervisé)
ET au finetune (supervisé).

**Portée.** Les chiffres **traffic et electricity ne sont pas publiables** en l'état. C'est
d'autant plus gênant que ce sont deux des trois datasets où le modèle brille (traffic MASE
0,768, +46 % de skill ; electricity 1,029) et que **traffic porte à lui seul l'essentiel du
gain de transfert d'E12** (−26 % contre −0,3 % à −5 % ailleurs) : on ne peut pas exclure que
ce gain soit en partie de la mémorisation.

**Ce qui n'est PAS touché.** ettm1, ettm2, etth1, etth2, exchange et weather paraissent
propres (weather Monash a une forme différente du weather Nixtla — à confirmer). Et les
conclusions **géométriques** (E5, E7, E9) sont des comparaisons appariées où la contamination
agit des deux côtés : elle ne peut pas créer un écart entre arms.

**Exclusions appliquées à LOTSA.** 7 motifs pour les benchmarks Nixtla, et 23 pour GIFT-Eval
**vérifiés contre le dépôt officiel** le 2026-08-13
(`huggingface.co/api/datasets/Salesforce/GiftEval/tree/main`) : les 28 répertoires y sont
couverts, avec un test de non-régression sur les noms verbatim. GIFT-Eval autorise en
principe l'entraînement sur son propre split de train ; exclure le dataset entier est donc
plus conservateur que le minimum requis — choix assumé, faute de pouvoir aligner nos
découpages sur les leurs.

**Vérification de la conversion (2026-08-13).** `prepare_lotsa.py --list` exécuté pour de
vrai : **45 sous-ensembles exclus, ~80 retenus**. L'exclusion attrape bien ett1/ett2,
electricity_15min, traffic_hourly, traffic_weekly, Q-TRAFFIC, weather, oikolab_weather,
solar_power, m1/m3/m4, nn5, LOOP_SEATTLE, M_DENSE, SZ_TAXI, taxi_30min, car_parts,
hierarchical_sales, kdd_cup_2018, temperature_rain, restaurant, saugeenday, hospital,
covid_*, us_births, tourism_*, wiki-rolling_nips.

**Limite trouvée à cette occasion.** La section « Local datasets » de `evaluate.py` porte sur
le corpus Monash, dont **sept entrées ont un équivalent RETENU dans LOTSA** :
london-smart-meters, bitcoin, wind-farms-minutely, rideshare, fred-md, sunspot-daily,
melbourne-pedestrian-count. Ces lignes sont donc de l'in-domaine et **ne doivent pas être
présentées comme du zero-shot**. La section Nixtla, elle, est intégralement propre — c'est
celle qui porte les chiffres publiables. Documenté dans `lotsa_tiny_eval.yaml` plutôt que
corrigé par une liste curée : maintenir une correspondance Monash↔LOTSA exhaustive (alias
compris) est précisément le genre de liste dont une omission produirait silencieusement une
revendication fausse.

**Correction retenue.** Ne pas purger le corpus Monash au cas par cas — ce serait une liste à
maintenir — mais changer de protocole : pretrain ET finetune sur LOTSA (dont la liste
d'exclusion couvre Nixtla et GIFT-Eval), évaluation zero-shot sur Monash et Nixtla. Corpus
d'entraînement et corpus d'évaluation deviennent alors disjoints **par construction**,
vérifiable dans le log de conversion. Voir `configs/model/lotsa_tiny_zeroshot.yaml`.

**Note honnête sur la découverte.** Le défaut est présent depuis le début du projet et n'a
été vu qu'en écrivant la discipline d'exclusion pour un AUTRE corpus. C'est un argument pour
énoncer explicitement la provenance des données dans tout article : la question « le corpus
d'entraînement recoupe-t-il l'évaluation ? » ne s'était jamais posée à voix haute.

**Leçon transversale, qui mérite une phrase dans un article :** la majorité de ces défauts
étaient **silencieux** — formes de tenseurs valides, pertes qui descendent, aucun avertissement.
Trois d'entre eux (B20, l'incident du décodeur ponctuel, l'incident de corpus) ont été
détectés en lisant des lignes de log, non par les tests. Les tests garantissent le mécanisme ;
ils ne garantissent pas la cohérence du protocole expérimental.

---

## 6. Reproduction

Configurations déclaratives, une variable chacune (aucun override nécessaire) :

| config | ce qu'elle isole |
|---|---|
| `tiny_geo` | référence du round : patch 16/8, SIGReg, contexte 1024, horizon 256, décodeur quantile, finetune sur 24 |
| `tiny_geo_p32` | patch 32/16 |
| `tiny_geo_vicreg` | perte VICReg |
| `tiny_geo_scratch` | aucun pretraining (G4.5) |
| `tiny_geo_lowdata` / `tiny_geo_scratch_lowdata` | transfert vers données inédites (G4.6, E11/E12) — jumelles par héritage |
| `lotsa_tiny` | pretrain sur LOTSA (G5) |
| `lotsa_tiny_zeroshot` | finetune sur LOTSA — **protocole principal**, évaluation zero-shot |
| `lotsa_tiny_finetune` | finetune sur Monash — borne haute, **contaminée**, non publiable |
| `lotsa_tiny_eval` | géométrie d'entraînement + données d'évaluation |

```bash
# pretrain
python scripts/train.py    --config-name <config>
# finetune
python scripts/train.py    --config-name <config> training.mode=finetune \
    +training.pretrained_encoder_path=<ckpt>
# évaluation (la géométrie suit la config, ne jamais la passer en override)
python scripts/evaluate.py --config-name <config> +checkpoint_path=<ckpt>
```

Suite de régression : `pytest tests/` (97 passed, 7 skipped au 2026-08-12). Chaque défaut
du §5 y a son test.

---

## 7. La thèse : ce que TimeJEPA parie

Section de fond. C'est ici que se trouve la contribution conceptuelle revendicable, et ses
faiblesses connues.

### 7.1 Le pari, énoncé précisément

TimeJEPA diverge du canon JEPA sur un point **structurel**, pas cosmétique.

Dans I-JEPA, le prédicteur est un **échafaudage** : il prédit des patchs masqués *à
l'intérieur* de la fenêtre, force l'encodeur à apprendre, puis **est jeté**. Le produit est
l'encodeur ; le downstream se branche dessus.

Dans TimeJEPA, le prédicteur est **l'organe central de l'inférence**. Il ne comble pas des
trous : il extrapole le futur en latent, et le décodeur n'est qu'une tête de lecture
(35 k paramètres sur 1,6 M). Le pari : *l'extrapolation latente est la bonne tâche de
pretraining pour un modèle de prévision, et le prédicteur mérite d'être conservé.*

### 7.2 Trois arguments a priori en sa faveur

1. **Alignement des tâches — l'asymétrie interpolation / extrapolation.** La reconstruction
   masquée entraîne une interpolation bidirectionnelle : combler un trou en regardant des
   deux côtés. La prévision est une extrapolation unidirectionnelle. Ce ne sont pas la même
   fonction. C'est le clivage BERT / GPT : le masquage l'emporte pour la compréhension,
   l'objectif causal pour la **génération** — et un forecast *est* une génération. Pour un
   modèle de fondation dont la mission est la prévision, l'objectif d'entraînement devrait
   être le geste demandé à l'inférence.
2. **La thèse JEPA s'applique mieux aux séries temporelles qu'aux images.** L'argument de
   prédire en latent plutôt qu'en pixels est d'abstraire l'imprévisible. En vision, la part
   imprévisible d'un patch masqué est modeste. En séries temporelles, la part
   **irréductiblement** imprévisible du futur est énorme — c'est la caractéristique dominante
   des datasets difficiles. Une perte en valeurs dépense de la capacité sur du bruit ; la
   perte latente laisse l'encodeur EMA l'absorber et concentre le prédicteur sur la structure.
3. **Le transfert est propre.** Encodeur + prédicteur portent l'intelligence, le finetune est
   une lecture bon marché. Observé : convergence rapide des finetunes, et greffe de la tête
   quantile sans toucher au tronc (E4).

### 7.3 Trois coûts, dont deux sont structurels

1. **Le prédicteur sous MSE produit une moyenne conditionnelle.** Mesuré : `pred_var` 0,6
   contre `target_var` 0,95. Le futur étant stochastique, l'optimum sous MSE est
   `E[z_futur | z_passé]` — un mélange flouté de futurs possibles, que le décodeur ne peut
   pas dé-mélanger. I-JEPA a ce problème aussi, mais l'entropie conditionnelle d'un patch
   masqué est bien plus faible que celle d'un futur. **Plus le pari est juste, plus ce coût
   est élevé.** C'est pourquoi l'architecture appelle structurellement une sortie
   distributionnelle : l'option B (E4) est un palliatif, l'option C (prédicteur
   probabiliste) en est la conséquence logique.
2. **La pression de collapse est plus forte.** Dans un JEPA masqué, cible et contexte se
   recouvrent : la tâche est facile, tricher par contraction rapporte peu. Ici la cible est
   véritablement difficile, donc écraser l'espace latent est une stratégie bien plus
   rentable pour la loss. Conséquence : **le régularisateur anti-collapse est porteur, pas
   accessoire** — davantage que dans I-JEPA. Observation de terrain cohérente : une val_loss
   qui baisse pendant que le rang effectif baisse, c'est-à-dire un critère de sélection de
   checkpoint qui sélectionne **pour** le collapse (voir §8, question ouverte).
3. **Le périmètre est scellé : prévision, pas représentation généraliste.** Un encodeur
   pré-entraîné au masquage sait relier des segments arbitraires — utile pour l'imputation,
   la détection d'anomalies, la classification. Celui-ci ne connaît que « passé contigu →
   futur ». Échange assumé, à énoncer comme tel.

### 7.4 Une conséquence dérivée, non explorée

La convention « prédicteur léger » (2 couches contre 3+ pour l'encodeur) vient d'I-JEPA,
**où le prédicteur est jetable**. Ici il est le prévisionniste. Rien ne dit que le bon ratio
de capacité encodeur/prédicteur soit le même dans les deux régimes. L'ablation
« prédicteur profond » existe en config et n'a jamais été lancée pour cette raison-là.

---

## 8. Décisions de conception et leur raisonnement

Ce que la section méthodes d'un article devrait justifier, avec la raison qui a présidé.

| décision | alternative écartée | raison |
|---|---|---|
| **RevIN par instance**, dénormalisation sans inverse affine | z-score global | le décodeur vise une cible z-scorée simple ; l'inverse affine introduit échelle + offset (B3). La normalisation par instance est aussi ce qui rend le modèle indifférent à l'échelle absolue des séries |
| **Cibles contextualisées** (l'encodeur EMA voit `[contexte ‖ cible]`, on découpe les N derniers patchs) | encoder la cible seule | reprend I-JEPA : la cible doit être représentée *en contexte*, sinon le prédicteur doit deviner une représentation acontextuelle. Alignement vérifié exact par test |
| **Géométrie tirée une fois par batch** | tirage par échantillon | garde les tenseurs rectangulaires — donc aucun padding, aucun masque d'attention supplémentaire. Rendu possible par RoPE, qui n'impose aucune table de positions apprise et rend l'encodeur agnostique à la longueur |
| **Monotonie des quantiles par construction** (médiane + largeurs softplus cumulées) | pénalité anti-croisement | garantie exacte plutôt qu'une pression douce ; aucun hyperparamètre ; aucun coût |
| **Couplage comonotone au rollout** | échantillonnage indépendant | l'indépendance sous-estime la dispersion (les erreurs se compensent), le comonotone la surestime. On choisit le biais de signe connu qui va *dans le sens sûr* pour un intervalle de prévision, et on l'écrit |
| **MASE poolée** | moyenne des ratios par fenêtre | robustesse aux fenêtres plates (voir §1) |
| **Vraie multi-résolution** (décimation d'une plage brute plus longue) | rééchantillonnage puis réinterpolation | l'ancienne version ne faisait que lisser : la période saisonnière en positions de patch ne changeait pas. La décimation la fait réellement varier |
| **Univarié par choix** | multivarié | décision de périmètre du projet. À énoncer comme telle dans un article : le modèle prévoit chaque série indépendamment, ce qui le rend applicable à tout jeu sans hypothèse sur le nombre de canaux |

---

## 9. Positionnement

À vérifier et compléter avant soumission — les chiffres ci-dessous viennent de lectures et
doivent être re-sourcés sur les articles originaux.

**Le paysage ne soutient pas l'idée qu'un paradigme aurait balayé les autres en prévision.**
Toto et TimesFM sont de type décodeur ; Moirai est un encodeur masqué ; Chronos tokenise les
valeurs dans une architecture séquence-à-séquence. Tous sont compétitifs sur GIFT-Eval. Une
contribution qui teste explicitement *quel objectif de pretraining* sert la prévision a donc
une place.

**Repères cibles (à re-vérifier).** Sur GIFT-Eval : naïf saisonnier 1,00 / 1,00
(MASE / CRPS) ; Toto-2.0-4m ≈ 0,76 / 0,52. Position actuelle estimée de TimeJEPA après le
round géométrie et la tête quantile : MASE/SN nettement sous 1,0 sur 5 datasets Nixtla sur 8,
CRPS non mesuré sur GIFT-Eval (le harness reste à écrire, P2.6).

**Voisinage direct.**
- **I-JEPA** (Assran et al.) — l'architecture parente ; divergence énoncée en §7.1.
- **LeJEPA / SIGReg** (arXiv 2511.08544) — le régularisateur alternatif implémenté, et sa
  promesse (corrélation loss ↔ downstream) que le projet est en position de tester.
- **PatchTST** — le patching et la tête aplatie ; l'alternative de décodeur non testée (P2.3).
- **LOTSA** (corpus de Moirai, ~27 Md d'observations) — le corpus de passage à l'échelle
  planifié, et le premier régime où données de pretrain ≫ données de finetune.

---

## 10. Squelette d'article et carte affirmations → preuves

Ce que l'on peut écrire aujourd'hui, ce qui manque, et où sont les chiffres.

| section | affirmation | preuve | statut |
|---|---|---|---|
| Intro / thèse | l'extrapolation latente est l'objectif aligné pour la prévision | §7.2 (argument), E5 (le système fonctionne) | **argument solide, preuve comparative manquante** |
| Méthode | architecture, RevIN, cibles contextualisées, géométrie aléatoire | §8, `TECHNICAL_OVERVIEW.md` | prêt |
| Méthode | tête quantile monotone par construction, option B | E4 | prêt |
| Méthode | rollout à éventail comonotone | E6 | prêt, hypothèse énoncée |
| Résultats | le bundle géométrique vaut −13 à −15 % de MASE | E5, table complète | prêt |
| Résultats | la taille de patch n'est pas un levier ; 4× de calcul économisé | E5 | prêt |
| Résultats | robustesse en longueur de contexte | E7 | prêt |
| Résultats | la tête quantile améliore aussi le ponctuel | E4 | prêt |
| Ablation | SIGReg vs VICReg | E5 | prêt (résultat nul, à publier comme tel) |
| Ablation | corpus de finetune curaté vs complet | E3 | prêt |
| **Ablation** | **le pretraining JEPA apporte-t-il quelque chose ?** | E8 + E11 + **E12 (run de robustesse)** | **POSITIF et précisé : gain de TRANSFERT (−8,7 %), pas d'ajustement — le scratch a une meilleure val_loss et une moins bonne généralisation** |
| Ablation | reconstruction masquée vs extrapolation latente | — | **non fait, et c'est le test du titre** |
| Évaluation | GIFT-Eval complet | P2.6 | harness à écrire |
| Discussion | validité : défauts silencieux du protocole | §5 | prêt, et inhabituel — un atout |

**Le manque critique.** Deux expériences séparent l'état actuel d'un article défendable :
(a) le pretraining apporte-t-il quelque chose (en cours), (b) l'extrapolation latente
bat-elle la reconstruction masquée à budget égal. La seconde n'est pas planifiée et
constitue le test le plus direct de la thèse du §7.

**Figures à produire.**
1. Schéma de l'architecture, avec le prédicteur mis en évidence comme organe conservé (le
   contraste avec I-JEPA est la figure la plus parlante).
2. Skill vs horizon, par dataset, référence contre round géométrie — l'inversion du profil
   d'ETTh1 s'y lit d'un coup d'œil.
3. Largeur d'intervalle vs horizon, réinjection médiane contre éventail propagé (table E6).
4. Courbe du balayage de contexte, avant et après randomisation — la platitude est le
   message.
5. Diagramme de fiabilité / couverture par dataset (montre aussi les échecs : ETTh2 à 42-65 %).

---

## 11. Journal des mises à jour

- **2026-09-06 (VEILLE : boucles de raisonnement dans l'architecture, générateur + critique —
  ce que la littérature dit, et ce que nos propres mesures disent déjà)** — Question
  utilisateur : GPT-6 Astra (profondeur récurrente) + notre paire juge latent / forecaster
  ⇒ optimiser la boucle critique pour converger vers un bon forecast ? Lignée vérifiée :
  Universal Transformer → Geiping 2025 « recurrent depth » (Huginn 3.5B : bloc récurrent
  déroulé à profondeur arbitraire, nombre d'itérations tiré d'une log-normale Poisson à
  l'entraînement, rétropropagation TRONQUÉE aux 8 dernières itérations, entrée ré-injectée
  à chaque pas, état initial aléatoire, sandwich-norm, sortie adaptative sur KL) ;
  HRM/TRM 2025 (7M params, récursion sur (z, y), supervision profonde, HRM gradient
  1-pas par détachement, TRM rétroprop complète, ACT) ; **EBT 2025** (Gladstone et al.,
  arXiv 2507.02092 : la prédiction EST une descente de gradient de ŷ dans une énergie
  E(x, ŷ) apprise, N pas, pas α et N randomisés, entraînement À TRAVERS l'optimisation par
  gradients du second ordre (produits Hessien-vecteur), +33-35 % de vitesse d'échelle vs
  Transformer++ y compris en vidéo continue, mais instable sans réglage, mauvais sur les
  distributions multimodales, exploré jusqu'à 800M) ; « Looped World Models » 2026
  (modèle nourri de ses propres sorties, BPTT tronquée, supervision profonde, horizons
  longs plus stables). Deux familles distinctes : (a) récurrence latente sans critique
  (Astra, Huginn, TRM) ; (b) optimisation contre un vérificateur (EBT, planning JEPA) — la
  question utilisateur est (b). **Ce que nous avons déjà mesuré en (b), à l'inférence,
  critique NON entraîné pour être descendu (E18f, 2026-08-21)** : raffinement doux (3 pas,
  lr 0.05) nul ; fort (10 pas, lr 0.5) = +1-2 % là où les propositions sont loin de la
  vallée (exchange, dérive : WQL 0.92 → 0.86 en hybride TTM), nul ailleurs, micro-Goodhart
  sur les tendances en pool centré (0.78 → 0.82). Lecture EBT : un paysage d'énergie
  appris par JEPA n'est pas appris pour être DESCENDU ; le gain de la boucle vient de
  l'entraîner à travers l'optimisation. Trois niveaux, coût croissant : L0 (existe)
  raffinement inférence, +1-2 % sélectif ; L1 (après H2b, une soirée) : décodeur entraîné
  à travers K pas de raffinement contre l'énergie PRÉSERVÉE (torch.autograd.grad avec
  create_graph=True, énergie gelée ⇒ second ordre en y seulement), métrique = bat le
  raffinement non entraîné d'E18f ; L2 (projet) : EBT temporel complet, la tête remplacée
  par la minimisation d'énergie, nouveau pretrain, risques documentés. Prérequis absolu
  de toute boucle : un critique qui SURVIT au finetune — c'est H2b. Pratique PyTorch
  (l'erreur LSTM de l'utilisateur « backward through the graph a second time ») : un seul
  backward sur la somme des pertes par pas (supervision profonde), ou détachement entre
  segments (BPTT tronquée, HRM), ou grad avec create_graph pour le second ordre (EBT),
  checkpointing pour la mémoire, nombre de pas randomisé.

- **2026-09-06 (CAP WORLD MODEL et H2b gravés au PLAN — réponse aux questions de direction)**
  — Le pretrain JEPA n'apporte probablement pas de précision zero-shot parce que le finetune
  repasse sur le MÊME corpus de 10 Md d'observations avec un objectif plus direct (E15,
  plateau du nu, scratch @5 %, E18b) ; il se justifie par ce qu'il conserve pour un world
  model : conditionnement (FiLM xres = gabarit du conditionnement par l'action), énergie
  (plausibilité, planning by backprop déjà écrit), fan (contrainte de couverture). H2b =
  loss jointe au finetune, critère = énergie préservée (sonde E18b ≤ 0.30) à coût CRPS
  ≤ 0.3 pt. Question « battre GIFT par les représentations riches ? » : aucune mesure ne le
  soutient à ce jour (l'énergie dilue notre propre fan, le raffinement est un substitut du
  centrage) ; la voie mesurée reste l'adaptation à l'inférence. Le seul chemin par lequel le
  world model pourrait payer sur GIFT : un juge PRÉSERVÉ (H2b) lisant un fan par une lecture
  qui ne dilue pas (centrée, T calibrée) — à tester après H2b, jamais avant.

- **2026-09-06 (SONDE DE COHÉRENCE w LIVRÉE : `probe_energy.py --rate-k K`)** — Instrument (3)
  du go/no-go xres. `probe_instance` prend `w` (refus si le modèle n'a pas de FiLM) ; avec
  `--rate-k K`, le contexte est décimé par K (la paire d'entraînement k1=K, k2=1), les
  candidats restent au taux natif, et le vrai futur est classé deux fois sur les MÊMES
  entrées : w=1/K (rate-aware) et w=1 (blind). Sortie : `mean_rank_cos` (aware),
  `mean_rank_cos_blind`, delta imprimé — négatif = le FiLM a appris w. Tests : identité
  exacte à l'init (FiLM à zéro), effet réel après perturbation, w=1 intouché (log2 1 = 0),
  refus sans FiLM. Usage au moment venu, sur le checkpoint de PRETRAIN xres :
  `python scripts/probe_energy.py --checkpoint <pretrain xres> --model-config
  lotsa_mini_xres_v3_eval --standalone-targets --rate-k 2` puis `--rate-k 4` ; référence :
  le même checkpoint sans `--rate-k` (rang natif) et un pretrain standard avec `--rate-k`
  (w ignoré ⇒ delta nul par construction, témoin négatif). Prédiction à graver au moment
  de la lecture : delta ≤ −0.05 à K=2 et K=4 sur les configs à cycle, sinon le pretrain n'a
  pas appris w et le finetune n'a rien à préserver.

- **2026-09-06 (SCRATCH HEAD8 @5 % : 0.8012/0.5582 — déjà au niveau du mini pré-entraîné à
  5 %, couverture 0.775 ; premier signal contre H1)** — Premier checkpoint (val 0.6699),
  flip+backtest : MASE 0.8012 / CRPS **0.5582** / couv 0.775 (q10 0.106, q90 0.882 — meilleure
  que head8), 34/97 décimées. Références à 5 % : mini standard pré-entraîné **0.5585**,
  v4 head8 pré-entraîné 0.5484 (deux variables). Un finetune SANS pretrain rejoint donc en
  5 % d'époque ce que le pretrain + finetune donnait au même point, à ~1 pt du meilleur 5 %
  connu. Configs courtes : m4_yearly 5.19, covid 51 — pires (l'extrapolation est ce que le
  pretrain apporte le plus tôt ?) ; le reste dans le bruit du head8. Verdict P-scr.1 au
  meilleur checkpoint du run ; mais si la trajectoire suit le motif habituel (pic vers
  25 %), le pretrain JEPA vaudra < 1 pt sur GIFT et H1 tombe. **Conséquence anticipée pour
  xres** (crainte utilisateur, justifiée) : si le pretrain n'apporte presque rien, un
  pretrain xres de 3 jours n'apporterait que ce que le finetune n'apprend pas seul ; or la
  capacité xres (FiLM + paires w≠1) peut être apprise AU FINETUNE — le FiLM est né à zéro
  (identité), `lotsa_mini_xres_v3_zeroshot` porte déjà p_multi_resolution_finetune 0.3 ;
  charger le pretrain STANDARD val-best dans un modèle cross_resolution=true (clés w_film
  absentes → allow_partial, identité exacte) donne un bras « xres-ft » d'une soirée.
  Décision à prendre au verdict scratch : pretrain xres (3 j) ou xres-ft (1 soirée) d'abord.

- **2026-09-06 (XRES : le risque de désapprentissage au finetune est RÉEL et NON MESURÉ —
  instruments go/no-go inscrits avant le lancement)** — Question utilisateur : ESJEPA a
  perdu z au finetune, E18b a montré que le full finetune détruit l'alignement énergétique
  (rang du vrai futur 0.245 → 0.409, sz_taxi sous le hasard) ; pourquoi xres survivrait-il ?
  État des pièces : la mitigation G9.3 (« une capacité survit si et seulement si elle est
  traversée par le gradient ») est CÂBLÉE et testée mécaniquement — paires w≠1 au finetune
  (`p_multi_resolution_finetune` 0.3, FiLM traversé par 30 % des items), ancre λ·MSE(z_pred,
  z_tgt) avec cible = copie de l'encodeur chargé (9 tests) — mais **jamais exercée dans un
  run réel** : tous les finetunes tiny/mini ont tourné avec lambda_anchor 0 et sans w.
  Différence de mécanisme avec ESJEPA/E18b : z n'était lu par AUCUNE perte de finetune
  (gradient nul par construction), alors que w entre dans la pinball via le FiLM sur les
  items w≠1 — la survie est plausible, pas prouvée. **Trois instruments, tous avant de
  faire confiance au finetune** : (1) témoins live `aug/w_neq1_frac` > 0 et `train_loss/
  anchor` stable (plateau, pas de dérive) ; (2) **test de sensibilité à w post-finetune**,
  éval seule : sur les configs décimées, `+ratein_w` (fan demandé à w=1/k) contre le chemin
  standard (w=1 + ré-interpolation) — bit-identiques ou quasi ⇒ FiLM mort, bras stérile ;
  écart franc ⇒ conditionnement vivant ; (3) **sonde de cohérence** sur le checkpoint de
  PRETRAIN : énergie E(ctx_k, y_k | w=1/k) vs E(ctx_k, y_k | w=1) sur des paires décimées
  — dit si le pretrain a appris w AVANT de payer le finetune (probe_energy passe w=1
  aujourd'hui, à étendre : petit travail). Règle : (3) après le pretrain, (1) pendant,
  (2) au premier checkpoint de finetune ; si (2) est plat, on n'attend pas le 25 %.

- **2026-09-06 (CHAMPION TINY IDENTIFIÉ ET REMESURÉ : mix-pool 0.8081/0.5529 — P-tmp.1 ÉCHEC,
  le prix de la capacité à pile égale est 1.9 pt, pas 1.1 ; DÉCISION : on reste sur mini,
  S5 (xres tiny) ANNULÉ, le mini xres redevient le bras xres)** — Champion tiny =
  `checkpoints/timejepa_lotsa_tiny_v3_zs/pretrain_False/epoch00_valloss0.5949.ckpt`
  (finetune v3 @50 %, à noter pour toute reprise). Flip + mix-pool : **MASE 0.8081 / CRPS
  0.5529**, couv 0.746, 53/97 configs décimées. P-tmp.1 prédisait 0.545 ± 0.005 : **ÉCHEC**,
  le tiny profite MOINS du mix-pool que le mini (0.5588 → 0.5529 = −0.59 pt, contre 0.5433 →
  0.5340 = −0.93 pt sur head8). À pile identique : tiny 0.5529 vs mini head8 0.5340 =
  **1.9 pt** pour 2.8M paramètres de plus (3.9M actifs au finetune vs 1.1M). Troisième
  observation du même motif : la capacité achète la RÉPONSE aux couches d'inférence
  (tiny : −4.5 pts nu→pile ; mini head8 : −7.9). Décision utilisateur : les bras
  d'ablation restent sur mini malgré 30 h/époque contre 3 j 4 h pour le pretrain — le
  transfert tiny→mini est trop incertain pour économiser le pretrain. S5 annulé ; P-xt.1..3
  se reportent sur le mini xres (référence appariée head8, tête ×8, `+ratein_w`). Le nu du
  0.5949 reste non mesuré (optionnel, ferme la ligne). Scratch head8 : premier checkpoint
  val 0.6699, courbe de train sMAPE qui rejoint les autres runs ; éval en cours.

- **2026-09-06 (CORRECTION DE LA TABLE D'ÉCHELLE : deux checkpoints tiny confondus — vérifié
  sur demande utilisateur)** — La ligne « tiny mix » de la table du 2026-09-05 mélangeait
  deux checkpoints à 0.0001 près en flip : le champion MIX (`mix1ep3e4_25pct`, corpus mix,
  nu 0.8914/0.6134, flip **0.8735/0.5984**, jamais passé sous RateIN v3) et le finetune
  **V3 @50 %** (corpus v3, flip **0.8633/0.5983**, « meilleure MASE du projet toutes
  lignées » le 30/08), qui est le « champion v3 » sur lequel la campagne RateIN a tourné
  (0.5983 → 0.5793 → 0.5682 → 0.5588, oracle 0.5358). Le nu du v3 @50 % n'a pas été
  consigné (n.m.). Table corrigée en deux lignes. Conséquences : (1) la lecture « le nu
  n'a pas bougé de tiny à mini » repose sur le nu du champion MIX (0.6134) vs mini
  (0.6235 / 0.6131) — elle tient, mais à corpus différent ; (2) la référence appariée du
  bras S5 (xres tiny v3) et la mesure P-tmp.1 (mix-pool) doivent porter sur le **v3 @50 %**,
  même corpus que xres tiny v3 — la commande donnée plus tôt visait le checkpoint mix, à
  corriger ; le fichier exact est sur le pod (`checkpoints/timejepa_lotsa_tiny_v3_zs/
  pretrain_False/`, celui dont le dossier d'éval porte `gift_flip_ratein-bt`).

- **2026-09-06 (DÉCISION : ablation XRES À PETITE ÉCHELLE d'abord — bras tiny xres v3 déclenché,
  mix-pool sur le champion tiny, prédictions gravées)** — Question utilisateur : combien vaut
  xres, et faut-il ablater sur tiny avant de scaler ? Faits : (1) aucun chiffre GIFT n'existe
  pour xres — G9.2 tiny était muet par construction (w=1 à l'éval) ; l'estimation « 0.5-1 pt »
  reposait sur le résidu de sélection (1.5 pt) et la ré-interpolation évitée par `+ratein_w`.
  (2) Le mix-pool n'a jamais été mesuré sur le tiny (arrêté au backtest v3, 0.5588). (3) Un
  seul bras a été mesuré aux deux échelles, RateIN, et son gain a grandi (−3.95 → −4.61 pt).
  Les configs `lotsa_tiny_xres_v3{,_zeroshot,_eval}` (G9.3 amendé : w exercé au finetune,
  ancre λ=0.1, p_multi_resolution_finetune 0.3) étaient GATED PAR DÉCLENCHEUR — les deux
  déclencheurs sont maintenant remplis (oracle > +5 % sur 35/97 configs ; le sélecteur
  causal cale à 1.5 pt du plafond). **Bras déclenché.** Référence appariée = champion tiny
  (nu 0.8914/0.6134, flip 0.5983, backtest v3 0.5588) remesuré en flip + mix-pool.
  **Prédictions** : P-tmp.1 tiny flip+mix-pool = **0.545 ± 0.005** (le prix de la capacité à
  pile égale ≈ 1.1 pt vs mini head8 0.5340) ; P-xt.1 tiny xres v3, finetune, flip+backtest
  dur ≤ référence tiny backtest − 0.5 pt ; P-xt.2 `+ratein_w` (fan au taux natif, k ≤ 4)
  apporte ≥ 0.3 pt de plus ; P-xt.3 l'oracle du tiny xres est plus bas que celui du tiny
  standard (0.5358) d'au moins 0.5 pt — le conditionnement monte le PLAFOND, pas seulement
  la capture ; ÉCHEC-DIAGNOSTIC si P-xt.1 et P-xt.3 faux : xres n'apporte rien que RateIN
  n'apporte déjà, le mini xres est annulé (deux jours économisés). Témoins : pretrain
  `aug/w_neq1_frac` > 0 ; finetune `aug/w_neq1_frac` > 0 et `train_loss/anchor` stable.

- **2026-09-06 (ÉTAT DES PREUVES, réponse à deux questions utilisateur : « l'EBM a-t-il un
  vrai avantage ? » et « la déviation prédicteur + décodeur était-elle mauvaise ? »)** —
  **EBM** : ce qui tient, mesuré et répliqué — sonde E18b (rang du vrai futur 0.235 vs 0.5,
  juge qui mûrit avec le pretrain), uplift sur proposeur PONCTUEL (Nixtla 6/6 avec deux
  juges ; GIFT TTM 0.7258 → hybride 0.6508, intervalles calibrés, zéro entraînement), gate
  ESJEPA +18.7 pts sur son protocole. Ce qui ne tient pas — dilution de notre propre fan,
  sélecteur de taux par énergie (0.5848), hybride 0.65 loin de la pile 0.534 ; et **le full
  finetune désapprend l'énergie** (E18b) : forecaster et juge sont deux checkpoints, la
  promesse « un modèle, deux usages » n'est pas tenue. Verdict : résultat réel, étroit,
  secondaire — section « ce que l'énergie fait et ne fait pas », pas une vitrine.
  **Prédicteur + décodeur** : la recette classique (prédicteur jeté, décodeur sur
  l'encodeur) n'a JAMAIS été testée — aucune entrée, aucune config. On ne peut ni la
  blâmer ni la défendre ; le nu plafonne à 0.613 sans mécanisme établi. H1 (scratch) teste
  les poids, H2 (linear probe) teste les features avec prédicteur gelé ; aucun des deux ne
  teste le prédicteur lui-même → **H4 conditionnel** : tête quantile directement sur les
  embeddings de l'encodeur, prédicteur court-circuité (code, quelques dizaines de lignes :
  la tête attend aujourd'hui les latents prédits et cross-attend au contexte), à lancer si
  H1/H2 laissent la question ouverte. Lecture pour le papier : sur GIFT, les preuves
  positives sont à l'inférence ; l'architecture de finetune est une hypothèse non testée,
  pas un acquis.

- **2026-09-06 (AUDIT DE RECOUVREMENT GiftEvalPretrain × notre corpus, et PLAN DU PLATEAU :
  trois hypothèses H1-H3, tests d'une soirée chacun)** — Question utilisateur : que
  recouvre-t-on du corpus de pretrain sanctionné ? Croisement (liste HF du jour, 152
  sous-ensembles) avec lotsa_v3 (67 fichiers LOTSA réels) et nos motifs d'exclusion :
  **67/152 dans v3, 78/152 avec le bloc court v4** ; les 74 manquants se répartissent en (a)
  **53 shards annuels volontairement sous-échantillonnés** (cmip6 6/41 années, era5 6/30,
  largest 3/5 — la queue plafonnée, choix de composition, pas une perte) ; (b) **8 exclus par
  nos motifs plus stricts que le benchmark** : traffic_hourly, traffic_weekly, weather,
  oikolab_weather, cdc_fluview_ilinet, extended_web_traffic, kaggle_web_traffic_weekly,
  wiki-rolling_nips — bloqués UNIQUEMENT par les suites locales Nixtla/Monash (décision de
  périmètre, pas de sécurité, déjà notée dans lotsa.py) ; (c) **13 jamais convertis, hors
  motifs** : BEIJING_SUBWAY_30MIN, HZMETRO, SHMETRO, cdc_fluview_who_nrevss, cif_2016_6/12,
  covid_mobility, fred_md, godaddy, rideshare_with_missing, taxi_30min, uber_tlc_daily,
  vehicle_trips_with_missing — petits sous-ensembles courts, tombés à la géométrie
  (min-length 1280 du bloc dense). Inversement, **rien dans v3 n'est hors de
  GiftEvalPretrain** : on joue strictement à l'intérieur du corpus sanctionné. Conclusion :
  pas de levier données caché ; le seul « corpus v5 » possible est la réadmission des 8
  (b) si les suites locales cessent d'être officielles (elles ne le sont plus de fait :
  GIFT est la cible) plus les 13 (c) par le mécanisme pad+sidecar — gain attendu petit
  (sous-ensembles de petite taille), à ne considérer qu'après H1-H3.
  **Plan du plateau (décision utilisateur « on les mène rigoureusement »)** : le nu est à
  0.613 depuis tiny, tout le gain est à l'inférence. Trois hypothèses, un test chacune :
  **H1** le pretrain n'est pas le goulot → scratch head8 (EN COURS, P-scr.1..2 gravées) ;
  **H2** le finetune dérive loin de GIFT (pic à ~25 % d'époque puis dégradation pendant que
  la val loss descend, sur tous les runs) → finetune `linear_probe` (encodeur gelé, tête
  seule) depuis le pretrain val-best, une soirée ; prédictions à graver au lancement (si nu
  ≈ 0.61 : les features plafonnent ; si nettement pire : la recette de finetune vaut le
  gain, et LR/arrêt précoce deviennent le levier) ; **H3** le centre de la tête quantile
  est mou (écart MASE 3× l'écart CRPS vs TTM) → terme de perte ponctuelle sur la médiane au
  finetune (`loss.finetune_type`), une soirée ; prédiction à graver : MASE apparié baisse
  de ≥ 1 % sans dégrader le CRPS de plus de 0.2 pt. Séquence GPU : scratch → S4-c →
  pretrain xres lancé (2 j) → H2 et H3 sur la carte libre pendant le pretrain → finetune
  xres (+ grille S4-c si positive) + `+ratein_w`. CPU en parallèle : backtest à 4 fenêtres
  (éval seule) et diagnostic W/M. Attribution : un bras = une variable, toujours.

- **2026-09-06 (RATEIN SUR TTM-R3, LECTURE APPARIÉE : brut 0.7057 → flip 0.6989 → flip +
  mix-pool 0.6857 sur 87 configs identiques — P-TTM.2 ✓ (−2.8 %), LA COUCHE EST
  MODEL-AGNOSTIC ; V4@30 % clôt la trajectoire, verdict inchangé ; scratch lancé)** —
  `ttm_layers_paired.py` : 88 configs finies partout, 87 à comptes d'instances identiques
  (agrégats dessus, SN officielle). **brut 0.7057 · flip 0.6989 (−0.96 %) · flip+mix-pool
  0.6857 (−2.8 % vs brut, −1.9 % vs flip)** ; le mix bat le brut sur **52/87** configs,
  geomean des ratios par config 0.9716. P-TTM.1 ✓ (flip < 1 %), **P-TTM.2 ✓** (≥ 2 %
  relatif ; 2.8 mesuré à 10 instances/config — barre d'erreur large, lecture par config
  concordante), P-TTM.3 ✓ (poids plus concentrés sur k=1 que chez nous). Gains max :
  bizitobs_l2c/5T/medium ×0.48 (k12), /long ×0.53 (k32-48), temperature_rain ×0.72 (k8),
  loop_seattle/5T/long ×0.72 (k12), jena/10T/long ×0.78 (k3), bizitobs_service 10S
  long/medium ×0.84 (k3) — les MÊMES configs et les MÊMES k que notre oracle : le
  mécanisme (cycles hors bande + rollout) est celui du benchmark, pas celui de notre
  modèle. Pertes max : bizitobs_application/10S/short ×1.53 (k6-8), bitbrains_fast_
  storage/H/short ×1.29 (k16), bizitobs_application/10S/long ×1.19, m4_hourly ×1.18,
  bitbrains_fast_storage/5T/medium ×1.17 — faux positifs d'un backtest sur proposeur
  ponctuel (pinball = MAE) à 32 séries. **Conséquence papier** : RateIN (backtest causal
  poolé + mélange de quantiles, zéro métadonnée) transporte sur un modèle étranger à
  recette figée — c'est le résultat principal ; TimeJEPA en est la vitrine (−7.9 pts) et
  TTM-R3 la preuve de transport (−2.8 % MASE). À citer avec la réserve « 10 instances par
  config, 87 configs » ; version à 30 instances si le budget le permet. **V4@30 %** (val
  0.6559, loss repartie à la hausse) : 0.8086/0.5485, couverture **0.729** (q10 0.132,
  q90 0.861 — la pire de la lignée mini), m4_yearly 4.77. Trajectoire close : 5 % 0.5484 ·
  10 % 0.5539 · 15 % 0.5512 · 20 % 0.5487 · 25 % 0.5518 · 30 % 0.5485 — plateau 0.548-0.554,
  jamais sous son 5 %, et une couverture qui s'ÉRODE le long du finetune (0.758 → 0.729) :
  signature cohérente avec le mécanisme RevIN (des items à amplitude normalisée ×5-7
  apprennent des fans mal calibrés). Verdict S4-a inchangé, clos. **Scratch head8 lancé**
  (utilisateur) : run laissé entier, la courbe de train sMAPE annonce des premiers
  checkpoints faibles ; règle de lecture prédéclarée : le MEILLEUR checkpoint du scratch
  (flip+backtest) contre le meilleur head8 (0.5433) — contrôle lu avec générosité.

- **2026-09-06 (RATEIN SUR TTM-R3, première lecture à 10 instances/config : brut 0.7125 →
  flip 0.7057 → flip + mix-pool 0.6829 (MASE vs SN officielle) — P-TTM.1 ✓, P-TTM.2 ✓
  SOUS RÉSERVE d'appariement, P-TTM.3 ✓)** — Trois runs `--ttm-only --instances 10 --tag
  inst10`. Brut : MASE 0.7125 sur 89 configs (leaderboard TTM-R3-PT 0.7240 : notre harnais
  et 10 instances reproduisent la claim à 1.6 % près). Flip : **0.7057** (89), −0.95 %
  relatif — P-TTM.1 (< 1 %) ✓ de justesse : TTM a son propre RevIN, la symétrie de signe
  est presque déjà là. Flip + RateIN mix-pool (backtest poolé sur 32 séries) : **0.6829**
  mais sur **88 configs** — bitbrains_rnd/5T/short (brut 6.47, catastrophique) tombe à 0
  instance sur contextes décimés (NaN TTM), ce qui AVANTAGE le mix ; bitbrains_fast_storage/
  5T/long passe de 2 à 1 instance. Lecture brute : −4.2 % relatif vs brut, −3.2 % vs flip ;
  lecture honnête à faire sur les configs communes à instances identiques
  (`scripts/ttm_layers_paired.py`, livré) — estimation : ~−3 % une fois bitbrains_rnd
  remis, P-TTM.2 (≥ 2 %) tiendrait. Par config, le motif prédit est là : gains massifs
  exactement où notre oracle gagne — bizitobs_l2c/5T/long 1.31 → **0.70** (k32-48),
  /medium 0.94 → **0.45** (k12), jena/10T/long 0.82 → 0.65 (k3), loop_seattle/5T/long
  1.05 → 0.76 (k12), electricity/15T long/medium 1.15 → 1.04, 0.72 → 0.65, bizitobs_service
  10S long/medium 1.25 → 1.05, 0.98 → 0.82 ; et des faux positifs francs — bizitobs_
  application/10S/short 1.68 → 2.58 (k6-8), bitbrains_fast_storage/H/short 0.47 → 0.61 (k16),
  electricity/15T/short 0.71 → 0.81, bitbrains_fast_storage/5T/medium 0.37 → 0.43. Le
  backtest sur un proposeur PONCTUEL (pinball d'un point = MAE) avec 32 séries est plus
  bruité que le nôtre. P-TTM.3 ✓ : ~35 configs à k>1 dominant chez TTM contre 57/97 chez
  nous — un modèle à embeddings de fréquence a moins besoin de canonicalisation, mais en
  a encore besoin. **Lecture stratégique (provisoire)** : la couche transporte sur un
  modèle étranger, à recette figée, sans métadonnées, et le mécanisme est le même
  (cycles hors bande + rollout) — model-agnostic. Verdict définitif sur la lecture
  appariée ; puis 30 instances si le budget le permet (barre d'erreur à 10).

- **2026-09-06 (LES DEUX EXPÉRIENCES DU TITRE PRÉPARÉES : contrôle scratch et RateIN sur
  TTM-R3 — prédictions gravées)** — (1) **Scratch** : `lotsa_mini_v3_head8_scratch_{zeroshot,
  eval}`, recette head8 à l'identique SANS `pretrained_encoder_path` (corpus v3, tête ×8, 1
  époque, LR 3e-4, lambda_anchor déjà 0 dans la lignée). Mesure directe de la valeur du
  pretrain JEPA sur GIFT. **P-scr.1** : le scratch au 25 % (flip+backtest) est ≥ 2 pts de
  CRPS au-dessus de head8 (0.5433) — le pretrain vaut au moins ce que E11/E12 mesuraient
  hors domaine ; lecture prédéclarée : < 1 pt ⇒ le pretrain n'est pas un levier GIFT et le
  papier est « petit forecaster + adaptation sans métadonnées », JEPA n'étant qu'un moyen ;
  entre 1 et 2 ⇒ contribution secondaire ; ≥ 2 ⇒ JEPA reste au titre. P-scr.2 : le scratch
  est PIRE en couverture (E8 : le scratch était moins calibré). (2) **RateIN sur TTM-R3**
  (`evaluate_gift_hybrid.py --ttm-only --ttm-flip --ttm-ratein`) : adaptateur
  `TTMForecaster` (point répété sur 9 niveaux, rollout autorégressif au-delà de 96, flip =
  moyenne de f(x) et −f(−x)), `_backtest_series_k` poolé sur 32 séries par config (cap de
  coût), `_mix_weights`, `ttm_layered_point` (décimation, forecast ⌈h/k⌉, ré-interpolation,
  moyenne pondérée). Sorties sous `evaluation/gift_hybrid/ttm_raw[_flip][_ratein-mix-pool]/`.
  Métrique : MASE contre la SN officielle (le CRPS d'un point est effondré, jamais cité) ;
  référence TTM brut mesurée par nous 0.7475 (leaderboard 0.7240 : l'écart est le pipeline
  TTM officiel, déjà documenté). 4 tests (adaptateur, flip exact sur un proposeur impair,
  moyenne des composantes, backtest + mix sur l'adaptateur). **P-TTM.1** : flip seul
  change la MASE de TTM de moins de 1 % (TTM a son propre RevIN, la symétrie de signe est
  ~déjà là) ; **P-TTM.2** : flip + RateIN mix-pool améliore la MASE de TTM d'au moins 2 %
  relatif (0.7475 → ≤ 0.732), gains concentrés sur les configs que l'oracle désigne chez
  nous (bizitobs, solar/10T, electricity/15T, loop/5T) ; **P-TTM.3** : les poids du mix
  sur TTM sont plus concentrés sur k=1 que chez nous (un modèle à embeddings de fréquence a
  moins besoin de canonicalisation). Lecture : P-TTM.2 vrai ⇒ la couche est model-agnostic
  et devient LE résultat ; faux ⇒ la sensibilité au taux est propre à notre lignée
  (argument latent/JEPA, à écrire ainsi). Coût estimé : backtest 32 séries × 2 fenêtres ×
  11 k rollouts par config, quelques heures sur les 97 configs à 150 instances.

- **2026-09-06 (REPOSITIONNEMENT DU PAPIER : ce que les mesures autorisent, et les DEUX
  expériences bon marché qui tranchent la colonne vertébrale)** — Constat utilisateur :
  4e sub-10M ; le papier doit mettre en avant l'adaptation à l'inférence (backtest, opérations
  sur quantiles, zéro encodage fréquentiel) plutôt que JEPA « qui sert à rien sur GIFT ».
  État des preuves, relu au registre : (a) **JEPA vs reconstruction** (E15/G6, tiny, corpus
  d'époque, Nixtla) : MASE moyenne −1.4 % pour JEPA mais la reconstruction gagne 17/28
  cellules appariées — égalité à queues lourdes ; seul signal robuste : l'écart croît avec
  l'horizon (+0.8/+3.1/+6.6 % short/medium/long). JAMAIS mesuré sur GIFT ni à la recette
  actuelle. « Sert à rien » n'est donc pas établi ; « pas démontré supérieur » l'est.
  (b) **Pretrain vs scratch** (E8-E12) : égalité en domaine, −26 % MASE hors domaine
  (E11/E12) — GIFT EST le hors-domaine, donc le pretrain devrait compter, mais jamais
  mesuré sur GIFT. (c) **Énergie sur GIFT** : uplift réel sur un proposeur PONCTUEL (TTM
  0.7258 → hybride 0.6508), 6/6 Nixtla ; échec sur notre propre fan (dilution) et comme
  sélecteur de taux (0.5848). Un résultat « ce que l'énergie fait et ne fait pas », pas une
  vitrine. (d) **Adaptation à l'inférence** : nu 0.6131 → flip 0.5842 → RateIN mix-pool
  0.5340 (−7.9 pts), oracle 0.5190, variante FFT publiée (Reverso) 0.6022 sur le même
  modèle ; capacité ×3.5 n'améliore PAS le nu mais améliore la réponse aux couches. C'est
  la claim la plus solide. **Deux expériences qui décident du titre, une soirée chacune** :
  (1) finetune head8 SANS pretrain (scratch, même recette, corpus v3, GPU une soirée) →
  mesure directe de ce que le pretrain JEPA vaut sur GIFT ; si < 1 pt, le papier est « un
  petit forecaster + adaptation sans métadonnées » et JEPA n'est qu'un moyen ; si > 2 pts,
  JEPA reste dans le titre ; (2) RateIN (flip + mix-pool) appliqué à TTM-R3 via le harnais
  hybride (éval seule) → si TTM gagne, la couche est model-agnostic et devient LE résultat ;
  sinon, la sensibilité au taux est propre à notre lignée (argument JEPA/latent). Les deux
  avant toute réécriture ; xres après.

- **2026-09-06 (CLASSEMENT « FONDATIONS SEULES » : snapshot frais du leaderboard, doc
  `docs/GIFT_RANKINGS.md`, script `gift_foundation_rank.py`)** — Demande utilisateur : se
  comparer aux vrais modèles de fondation, pas aux orchestrateurs. Snapshot
  2026-09-06 vendu (127 entrées ; nouveaux depuis le 22/08 : TimesFM-3 330M à 0.4557,
  Granite-PatchTST-FM-r2 ; rien ne bouge autour de nous). Filtre : `model_type` ∈
  {zero-shot, pretrained}, sans fuite déclarée, moins les enveloppes reconnaissables au nom
  (STRIDE +Chronos-2/+Timer-S1) → **63 fondations sur 127** ; le reste = 28 agentiques,
  6 fine-tuned, 10 deep-learning par jeu, 6 statistiques, 17 à fuite. Le sommet du
  leaderboard (EXAONE-Agent 0.4185, STRIDE_w_Synapse, CastStar, LS-Agent) est entièrement
  agentique ; la meilleure fondation seule est TimesFM-3 (0.4557, 330M). **Nos rangs** :
  mini head8 (flip+mix+pool 0.5340) = **37e / 63 fondations**, **6e des sub-10M** (5e en
  comptant FlowState une fois : r1.1, Granite-r1.1 et 9.1M sont une lignée) — devant
  TinyCast, goia, Kairos, Metamorph ; derrière FlowState, TTM-R3-PT (0.5195) et Toto-2.0-4m
  (0.5242). tiny (0.5588) = 11e sub-10M. Le README garde sa table sub-10M (à aligner sur ce
  doc : retirer TempoPFN, 34.7M). Réserve : le type est auto-déclaré par les auteurs.

- **2026-09-06 (POINT D'ÉTAPE après v4 : la randomisation de contexte EXISTE déjà au finetune
  mais s'arrête à 128 ; bras S4-c « contexte court » créé, config seule ; carte gift_gap
  lue)** — Question utilisateur (« on ne le fait pas déjà ? ») : SI. Valeurs effectives du
  finetune head8 (composition Hydra) : `context_lengths` {128, 192, 256, 384, 512, 640, 768,
  1024}, `p_random_context_finetune` 0.5, tirage par batch, recadrage à gauche (le témoin
  `geometry/context_len` de la capture wandb d'hier oscillait bien entre 256 et 1000).
  Le régime JAMAIS vu est donc < 128 : à l'éval, m4_yearly arrive avec 13-40 pas,
  car_parts 39, hospital 72, une partie de m4_quarterly/monthly/weekly — exactement les
  saigneurs MASE. Correction de ma proposition d'hier soir : pas un nouveau mécanisme,
  une extension de grille. **S4-c** : `lotsa_mini_v3_head8_ctx_{zeroshot,eval}` — même
  pretrain, même tête ×8, corpus lotsa_v3, UNE variable : la grille gagne 32 et 64 (≈ 10 %
  des batchs en régime court). Aucun bourrage : le recadrage garde les pas récents, RevIN
  voit des points réels — la condition d'éval, cette fois vérifiée. Coût : un finetune
  d'une soirée. **P-ctx (gravées, à lancer sur décision utilisateur ; référence appariée
  head8 flip+backtest 15 % 0.7974/0.5466, 25 % 0.7914/0.5433)** : P-ctx.1 ≥ 4 des 6
  configs à contexte court (m4_yearly 3.80, m4_quarterly 1.31, m4_monthly 1.01, m4_weekly
  2.35, hospital 0.79, car_parts 0.87) baissent en MASE au 25 % ; P-ctx.2 configs longues
  stables à ±1 % de CRPS ; P-ctx.3 MASE < 0.7914 et CRPS ≤ 0.5433 au 25 %. ÉCHEC-DIAGNOSTIC
  si P-ctx.1 < 4/6 : le régime court n'est pas le mécanisme du saignement MASE — c'est
  l'extrapolation elle-même (tête/objectif), et on arrête de chercher côté données.
  **Carte gift_gap (mix-pool head8, ratios officiels)** : vs Toto (0.5242) on gagne 36/97,
  pertes max bizitobs_application/10S/short ×1.79, bizitobs_service/10S/short ×1.38,
  us_births/M ×1.32, electricity/W ×1.27, bitbrains_fast_storage/5T long/medium ×1.24 ;
  gains max bizitobs_l2c/5T/long ×0.36, /medium ×0.56 (RateIN). **What-if queue(16) au
  niveau de Toto : 0.5271 — encore au-dessus de Toto (0.5242)** : l'écart à la 3e place est
  LARGE, pas concentré dans une queue. Vs FlowState (0.5019) : 24/97, pertes systématiques
  sur W et M (electricity/W ×1.60, solar/W ×1.53, us_births M/W/D ×1.26-1.39, m4_hourly
  ×1.46) ; what-if 0.5208. Lecture : (1) les basses fréquences à horizon court (W : h=8,
  M : h=12-18) sont une faiblesse structurelle, distincte des contextes courts (electricity/W
  a ~150 pas de contexte, us_births/M ~240) — mécanisme candidat à instrumenter : à h ≤ un
  patch, la tête quantile n'a qu'une fraction de patch à prédire ; (2) les 10S short
  (bizitobs) restent notre pire duel malgré RateIN ; (3) la 3e place demande un gain
  DIFFUS de ~1 pt sur le corps, pas une queue. Ordre proposé : S4-c (une soirée) → xres
  (deux jours, hérite) ; diagnostic W/M en parallèle sur caches (CPU).

- **2026-09-06 (V4 : VERDICT NÉGATIF au 25 % — P-v4.1/2/3 ÉCHEC ; mécanisme identifié :
  la condition d'entraînement des fenêtres à frontière N'EST PAS la condition d'éval)** —
  Trajectoire v4 flip+backtest : 5 % 0.5484 · 10 % 0.5539 · 15 % **0.5512** · 20 % 0.5487 ·
  25 % **0.5518** (val 0.6542, plate entre 20 et 25). Apparié head8 : 15 % 0.5466 (+0.46),
  25 % 0.5433 (**+0.85 pt**) ; MASE 0.8065 vs 0.7914. **P-v4.3 ÉCHEC.** P-v4.1 (≥ 4/7
  configs courtes en baisse au 25 %) : covid 35.9 (41.2) ✓, car_parts 0.859 (0.869) ✓,
  hospital 0.785 (0.793) ✓ ; m4_quarterly 1.315 (1.314) et m4_monthly 1.014 (1.010) plats ;
  m4_weekly 2.49 (2.35) ✗ ; **m4_yearly 4.53 (3.80) ✗✗** — 3/7, **ÉCHEC**. P-v4.2 (longues
  stables ±1 %) : bizitobs_l2c/5T/long 0.267 (0.238, +12 %), bitbrains/5T/medium 0.730
  (0.757, −4 %), solar/10T/short 0.523 (=) — **ÉCHEC**. Le signal le plus parlant : m4_yearly
  se DÉGRADE de façon monotone le long du finetune v4 (3.63 → 4.14 → 4.68 → 5.51 → 4.53)
  quand head8 s'améliore (4.19 → 3.80) : ce n'est pas du bruit, le bras enseigne quelque
  chose de nuisible à l'extrapolation annuelle — malgré une dose de 0.3 % du batch.
  **Mécanisme (vérifié dans le harnais, `prepare_context`)** : à l'ÉVAL, une série courte
  devient un contexte COURT (troncature à gauche au multiple du stride, bourrage à UN patch
  seulement ; l'encodeur RoPE accepte les longueurs variables, les buckets d'éval le font
  déjà) — une série annuelle de 30 points est vue comme 24-30 pas, RevIN calculé sur les
  points réels. À l'ENTRAÎNEMENT v4, la même série est un contexte de 1024 pas dont ~1000
  de bourrage plat : RevIN (moyenne/écart-type sur tout le contexte) voit un écart-type
  rétréci d'un facteur ≈ √(n_réel/1024) ≈ 0.15-0.2, donc les points réels et la cible
  normalisés sont amplifiés ×5-7 ; la pinball sur ces items pèse d'autant, et surtout
  l'item enseigne « après un long plateau, la cible saute à 5σ » — le contraire de ce
  qu'une extrapolation annuelle calibrée doit faire. Ma spécification du 2026-09-05
  (« exactement la condition que l'éval impose ») était FAUSSE : l'éval n'impose pas de
  bourrage à 1024, elle raccourcit le contexte. Le bloc court v3 (nn5 735/1280) portait le
  même défaut, atténué. **Statut S4-a : bras CLOS tel quel** (règle prédéclarée : les
  configs courtes se dégradent ⇒ mécanisme). Pas de v4-dose : monter la dose d'un item mal
  spécifié aggraverait. **Ce qui survit** : le sidecar `_reallen` (qui supprime les fenêtres
  à cible-pad du v3, sain) et la pinball masquée (inerte hors bras). **Si on y revient
  (S4-a')** : entraîner la condition RÉELLE d'éval = contextes COURTS à longueur variable
  pour les lignes courtes (collate par bucket de longueur, comme l'éval ; RevIN sur les
  points réels par construction) + dose relevée par cap dédié — un chantier de loader,
  pas une ligne. À décider après xres. Le 30 % v4 sera lu mais ne change pas le verdict.

- **2026-09-06 (V4@10 % : 0.8113/0.5539 — recul vs le 5 % (0.5484) et sous le 10 % standard
  (0.5517) ; les configs courtes reculent aussi)** — Checkpoint val 0.6571, flip+backtest :
  MASE 0.8113 / CRPS 0.5539 / couv **0.776** (q10 0.104, q90 0.880 — la meilleure
  calibration de la lignée mini), 37/97 décimées. Trajectoire v4 : 5 % 0.5484 → 10 %
  0.5539 (+0.55 pt). Standard : 5 % 0.5585 → 10 % 0.5517. Configs courtes vs le 5 % v4 :
  m4_yearly **4.138** (3.634 ; head8 25 % 3.801), covid **46.9** (33.0 ; 41.2), m4_quarterly
  1.377 (1.322), m4_weekly 2.693 (2.658) — toutes en RECUL ; hospital 0.775 (0.784) et
  car_parts 0.852 (0.853) stables. La val loss, meilleure qu'au 5 % (0.6571 vs 0.6598),
  désigne le mauvais sens une fois de plus (G7.3c, 4e observation sur la lignée). Lecture
  prudente : un checkpoint isolé sur une trajectoire bruitée (le standard a eu son creux
  au 20 %) — mais l'inversion des configs courtes entre 5 % et 10 % est le premier signal
  contre P-v4.1 ; si le 15 % ne les ramène pas sous le 5 %, le mécanisme « dose
  homéopathique = bruit » prend le dessus sur « transfert ». Verdict au 15 % puis 25 %.

- **2026-09-06 (MIX + POOL : 0.7842/0.5340 — PILE OFFICIELLE ; −0.93 pt de CRPS sur le
  champion en une journée d'éval, résidu oracle ramené de 2.43 à 1.50 pt)** — Head8 25 %,
  flip + mix poolé : **MASE 0.7842 / CRPS 0.5340 / couv 0.756** (q10 0.119, q90 0.876),
  57/97 configs majoritairement décimées, 58.8 % d'instances k>1. Trajectoire de la
  journée sur le même checkpoint : backtest dur 0.5433 → mix 0.5403 → backtest poolé
  0.5381 → **mix poolé 0.5340** (attendu ~0.536 : mieux). Les deux leviers sont
  ADDITIFS (−0.30 + −0.52 ≈ −0.93 en composition : le pooling corrige l'information, le
  mix corrige la décision — orthogonaux comme prévu). Résidu vs oracle 0.5190 : **1.50 pt**
  (68 % → 90 % de capture du plafond de sélection sur le mini standard → head8 : capture =
  1 − 1.50/(0.5842−0.5190) ≈ 77 % du gain RateIN total possible depuis flip). Coût :
  couverture 0.756 vs 0.769 en dur (−1.3 pt, le mix étale moins que la sélection dure
  quand les composantes s'accordent ; à surveiller, pas rédhibitoire). Lectures par
  config : bitbrains_fast_storage/5T/long 0.768 (dur 0.798, oracle 0.728) — le mix
  k8-24 récupère la moitié du manqué ; bitbrains/5T/short 0.417 (0.444, oracle 0.410) ;
  bizitobs_service/10S/medium 0.024 = oracle (k16 à 0.61) ; ett1/D 0.317 (0.372, oracle
  0.278) ; jena/10T/medium 0.050 = oracle. Revers : bitbrains_fast_storage/H/short 0.721
  (0.672 dur) et m4_hourly 0.030 (0.024) — les faux positifs du pooling persistent.
  **Pile officielle du champion : flip + mix + pool** (`+tta_flip=true +ratein=mix
  +ratein_pool=true`, causal, ≤ ×4 passes). Position sub-10M : 4e, Toto (0.524) à 1.0 pt,
  TTM-R3 (0.520) à 1.4. Les comparaisons appariées restent en flip+backtest dur.

- **2026-09-06 (VERDICTS : POOLING = NOUVELLE MEILLEURE PILE 0.7871/0.5381 (P-pool.1 ✓) ;
  ÉNERGIE = ÉCHEC-DIAGNOSTIC 0.5848, idée close comme sélecteur)** — Head8 25 %, flip.
  **Backtest poolé : MASE 0.7871 / CRPS 0.5381 / couv 0.760**, 43/97 configs décimées,
  44.3 % d'instances k>1. Vs backtest dur 0.7914/0.5433 : **−0.52 pt CRPS**, −0.43 MASE ;
  vs mix 0.5403 : −0.22 pt. Une ligne de code (Σ au lieu du geomean) fait plus que le
  mélange : l'objectif de sélection était le défaut le plus cher. Résidu vs oracle :
  0.1357 → 0.1308 (0.48 pt brut, ≈ 1.9 pts de ratio, contre 2.43 au départ) ; missed 15
  (30 %) · wrong_k 17 (44 %) · false_pos 10 (27 %) · match 55. Seconde clause de P-pool.1
  (« part wrong_k baisse ») NON tenue : la part monte (38→44 %), l'absolu est plat
  (0.23→0.21 pt) — le pooling a surtout converti des missed en match (22→15 missed, 47→55
  match). Nouveau faux positif notable : m4_hourly k=3 (0.034 vs 0.024, 9.8 % du résidu).
  Six missed « sous la marge » identifiés (bitbrains/5T/short, bitbrains_rnd/5T/medium,
  bizitobs_l2c/5T/short, ett2/15T/medium, ett2/H/medium, kdd/H/medium) : c'est le terrain
  du mix, à tester en `mix + pool` (attendu ~0.536). **Pile officielle du champion :
  flip + backtest-pool** ; les comparaisons appariées restent en flip+backtest (caches).
  **Énergie : MASE 0.8459 / CRPS 0.5848 / couv 0.720**, 57/97 décimées — P-E.1 ÉCHEC
  (pire que le backtest de 4.2 pts, pire que le nu+flip 0.5842) ; P-E.2 ✓ trivialement ;
  P-E.3 ÉCHEC (solar/10T : k=48 choisi, oracle 3/6/2 ; solar/H et loop/H : k=6-12, oracle
  k=1 ; un seul succès, loop_seattle/5T/medium 0.076 ≈ oracle 0.071 là où le backtest
  manque). Résidu 1.66 pts brut : false_pos 56 % (23 configs), wrong_k 37 % (30) — le
  biais est SYSTÉMATIQUE vers les grands k. **Mécanisme (nommé, non testé)** : l'énergie
  compare la prédictibilité de 256 pas DÉCIMÉS, soit k×256 pas réels — une tâche qui change
  avec k (série lissée par la décimation, horizon réel plus long) ; le ratio d'énergie
  n'est pas calibré entre k et ne mesure pas la qualité du forecast à l'horizon natif.
  Règle prédéclarée appliquée : P-E.1 ET P-E.3 faux ⇒ **idée close comme sélecteur** ; le
  mode reste en code (`+ratein=energy`) comme ablation négative citable (« l'énergie du
  pretrain préfère les grands k »). Variante non essayée, une ligne, si l'utilisateur y
  tient : span du juge apparié en temps réel (256/k pas décimés) — même horizon réel pour
  tous les k. Prochaine mesure : `+ratein=mix +ratein_pool=true` sur head8.

- **2026-09-06 (DEUX INSTRUMENTS DE SÉLECTION DE TAUX LIVRÉS : pooling aligné sur le CRPS,
  et RateIN-ENERGY, le k que le pretrain trouve naturel — prédictions gravées)** — Décision
  utilisateur (parcimonie) : parmi les cinq pistes proposées, garder (2) l'alignement de
  l'objectif et (4) le détecteur par énergie ; historique apparié et règle 1-SE écartés
  (complexité). **(2) `+ratein_pool=true`** (backtest, mix, energy) : la table des ratios
  passe du geomean par série (poids égaux) au ratio des SOMMES sur les séries, exactement
  la pondération de l'arbitre (CRPS = Σ2QL/Σ|y| sur la config : les séries de grande
  amplitude dominent). Règle 2/3 inchangée. Tag `-pool`, champ `pooling` dans le cache.
  **(4) `+ratein=energy +energy_ckpt=<pretrain>`** : pour chaque k, on décime le passé
  (avant toute cible de test), les 256 derniers pas décimés jouent le futur, le reste le
  contexte ; énergie = recette du juge hybride (E18/G12, encodeur online des deux côtés) :
  1 − cos entre le latent prédit par le prédicteur et le latent encodé du vrai futur. k =
  argmin du ratio poolé d'énergie vs k=1 (pas de marge), même règle 2/3, garde par instance.
  Aucun rollout, un passage encodeur+prédicteur par (série, k). Voit la canonicalisation
  des cycles (ce que le pretrain a appris), pas le rollout collapse. Juge = checkpoint de
  pretrain (val-best 0.5495), chargé à part (`+energy_config` optionnel, allow_partial).
  Limite connue : même famine que le backtest pour les grands k (il faut (1024+256)·k pas
  réels) ; sur m_dense/D (smoke) seul k=2 est scorable. Vérification : 24 tests verts
  (pooling : le geomean favorise k=2 quand la petite série gagne 50 %, le pooled le rejette
  quand la grande perd 20 % ; énergie : k uniforme par config, table de ratios, grands k
  disqualifiés sans crash) ; smoke CPU m_dense : pooled ratios 1.33-1.82 (K=1, identique),
  énergie ratio k=2 1.061 (K=1). **Prédictions (champion head8 25 %, flip, référence
  backtest 0.5433 / oracle 0.5190)** : P-pool.1 flip+backtest-pool ≤ 0.5433 et la part
  « wrong_k » du résidu baisse (bizitobs, où quelques séries dominent) ; P-E.1
  flip+energy ≤ 0.5433 = l'énergie sélectionne au moins aussi bien que le backtest sans
  forecast (succès fort si ≤ 0.538) ; P-E.2 énergie et backtest désaccordent sur ≥ 30 %
  des configs (sinon même information, pas de complémentarité) ; P-E.3 sur les configs à
  cycle (electricity/15T, jena/10T, solar/10T, loop/5T) l'énergie choisit le k oracle plus
  souvent que le backtest. ÉCHEC-DIAGNOSTIC P-E.1 > 0.5433 ET P-E.3 faux ⇒ l'énergie du
  pretrain n'encode pas la préférence de taux, l'idée est close ; P-E.1 échec mais P-E.3
  vrai ⇒ complémentaires, mix des deux tables à considérer.

- **2026-09-06 (V4@5 % : 0.8087/0.5484 — déjà au niveau du 15 % standard ; 5/7 configs
  courtes en baisse, non apparié)** — Premier checkpoint v4 (val 0.6598), flip+backtest :
  MASE **0.8087** / CRPS **0.5484** / couv 0.758, 31/97 configs décimées. Références :
  mini standard 5 % **0.5585** (seul 5 % disponible ; head8 n'a pas de 5 % évalué), donc
  −1.0 pt à budget égal — mais deux variables (tête ×8 + corpus v4), la part de chacune
  n'est lisible qu'au 15 %/25 % apparié head8 (0.5466/0.5433). Configs courtes vs head8
  25 % (non apparié, à titre indicatif) : m4_yearly **3.634** (3.801, −4.4 %), m4_monthly
  0.997 (1.010), hospital 0.784 (0.793), car_parts 0.853 (0.869), covid **33.0** (41.2,
  −20 %) en baisse ; m4_quarterly 1.322 (1.314) plat ; m4_weekly 2.658 (2.350) en hausse.
  m4_hourly 1.209 (1.389). Signal dans le sens de P-v4.1 dès 5 % malgré la dose ; verdict
  au 15 % puis 25 %.

- **2026-09-05 (soir — V4 LANCÉ : témoin positif mais DOSE FAIBLE ; P-v4.1..3 GRAVÉES)** —
  Corpus v4 assemblé et vérifié (gate 2 : 118 entrées, diff vs v3 = exactement les 11
  familles courtes réadmises, les deux dec3 synthétiques retirés comme en v3). Finetune
  `lotsa_mini_v4_zeroshot` (tête ×8, pretrain val-best 0.5495) lancé. **Témoin
  `aug/short_frac`** (wandb, 12k premiers steps) : > 0 donc sidecar lu, bras non stérile —
  MAIS pics isolés à ~0.3 % du batch (0.0030 au step 5449), zéro la plupart des steps.
  Mécanisme lu : poids du sampler en √(nb fenêtres) par fichier — les 12 familles courtes
  ont quelques centaines de fenêtres chacune contre des millions pour les denses — et le
  rationnement G10.2 étale ce petit budget sur l'époque (d'où l'intermittence). Ordre de
  grandeur : quelques milliers d'items courts sur l'époque pour ~6 900 lignes, soit moins
  d'un passage par ligne. **Le bras teste donc le mécanisme à dose homéopathique.**
  **Prédictions gravées (référence appariée head8 flip+backtest : 15 % 0.7974/0.5466,
  25 % 0.7914/0.5433)** : P-v4.1 (mécanisme) les configs à historique court (m4_yearly
  3.80, m4_quarterly 1.31, m4_monthly 1.01, m4_weekly 2.35, hospital 0.79, car_parts 0.87,
  covid 41.2) baissent en MASE au checkpoint apparié — bande large vu la dose, succès si
  ≥ 4 des 7 baissent ; P-v4.2 (innocuité) configs à long historique stables à ±1 % de
  CRPS ; P-v4.3 (agrégat) MASE < 0.78 et CRPS ≤ 0.5433 au 25 %. **Lecture d'échec
  prédéclarée** : si rien ne bouge (P-v4.1 < 4/7 et MASE ≥ 0.7914), le diagnostic est la
  DOSE (sampler), pas le mécanisme — bras suivant « v4-dose » : relever la part des
  familles courtes (poids par famille ou cap d'oversample dédié) AVANT de conclure sur
  les fenêtres à frontière. Si les configs courtes se DÉGRADENT, c'est le mécanisme
  (fenêtres à frontière nuisibles) et le bras est clos. **Audit de composition (reçu le
  06/09, queue de table)** : les 12 familles courtes sont TOUTES « capped » à 0.00 % de
  part de batch (m1_*, monash_m3_*, tourism_*, nn5_*), au même rang que covid19_energy ou
  favorita_sales ; part synthétique 57.5 % (v3 : ~51-55 %, acceptée par l'utilisateur).
  La dose est donc confirmée par l'audit statique, pas seulement par le témoin live : le
  cap d'oversample (max_oversample_ratio, global) est le verrou — un fichier de quelques
  centaines de fenêtres ne peut pas dépasser cap × ses fenêtres par époque. Le bras
  « v4-dose », si P-v4 échoue sans dégradation, devra desserrer ce cap POUR CES FAMILLES
  (override par famille, code à écrire) plutôt que globalement (sinon la queue réelle
  plafonnée se ré-inonde, verdict G10.2). Doctrine d'éval pour v4 : comparaisons
  appariées en flip+backtest ; le mix se pose uniquement sur le checkpoint retenu.

- **2026-09-05 (PRÉP V4, premier passage : deux défauts de prepare_lotsa attrapés au gate 1)**
  — Log du bloc court (min-length 24) lu par l'utilisateur : « 122 LOST » sur m3_quarterly
  alors que 756 chunks sont écrits pour 756 séries. Diagnostic : (1) le compteur
  `lost_to_chunking` ignore `--pad-to` (la série courte est gardée entière et paddée) —
  message faux ; (2) plus grave et silencieux : avec `--pad-to`, la longueur de chunk était
  quand même ADAPTÉE À LA MÉDIANE (44 pour m3_quarterly sur des séries de 24-72, 78 pour
  m1_monthly), puis `segment_series` garde les premiers morceaux et jette le reste → toute
  série plus longue que la médiane était TRONQUÉE À SES PREMIERS PAS (les plus récents
  perdus). Ce défaut touchait déjà le bloc court v3. (3) Pertes réelles au seuil 24 :
  monash_m3_yearly rejeté en bloc (médiane < 24), 78 tourism_yearly. **Correctifs** : avec
  `--pad-to`, pas d'adaptation à la médiane (effective = chunk_length ; subset gardé si une
  série ≥ min_length) ; compteur LOST inactif sous pad_to ; seuil `--min-length 20`
  (= 16 ctx + 4 cible, une fenêtre exacte). Test ajouté (série de 72 gardée entière, queue
  incluse, LOST = 0 ; 24 tests corpus verts). Runbook v4 mis à jour ; le premier
  `lotsa_short_v4` est à renommer `_trunc` (jamais supprimé) et la prép relancée.
  **Gate 2 (assemblage), attrapé par l'utilisateur** : 120 entrées au lieu de 118 — le
  `ln -s ../decimated/*.npy` remet `synthetic_broadband_dec3` et `synthetic_lowfreq_dec3`,
  retirés EXPRÈS de v3 le 2026-08-27 (part synthétique 55.2 % > cible 50-55 %). Retirés de
  v4 aussi (liens seulement, `decimated/` intact) ; runbook mis à jour. Sans ça, v4
  aurait porté une seconde variable (deux shards synthétiques de plus).

- **2026-09-05 (RATEIN-MIX : 0.7864/0.5403 — meilleure pile du projet, mais P-mix.1 ÉCHEC :
  12 % du résidu récupéré, pas un tiers ; le résidu est un désaccord backtest↔test, pas
  une règle de décision)** — head8 25 %, flip+mix : **MASE 0.7864 / CRPS 0.5403 / couv
  0.768** (q10 0.114, q90 0.881), 45/97 configs majoritairement décimées, 46.4 %
  d'instances k>1 (contre 35/97 et 36.1 % en sélection dure). Vs flip+backtest
  0.7914/0.5433/0.769 : **−0.50 pt MASE, −0.30 pt CRPS**, couverture −0.1 pt (P-mix.2
  tenue au bruit). **P-mix.1 ÉCHEC** (prédit ≤ 0.535 : le mélange devait rendre ≥ 1/3 des
  2.43 pts ; il en rend 0.30, soit 12 %). Pas d'échec-diagnostic (< 0.5433), le mix
  reste un gain net. Décomposition mix vs oracle : geomean 0.1362 → 0.1308, résidu 4.1 %
  (2.1 pts de ratio) ; missed 19 (25 %) · wrong_k 15 (34 %) · false_pos 16 (39 %) · match 47.
  Le mélange a déplacé les cas (missed 22→19, false_pos 9→16) sans les résoudre — attendu
  pour une règle de décision quand l'INFORMATION est fausse. **La table des ratios (nouveau
  champ) le prouve** : sur les plus gros manqués, le backtest ne se trompe pas de peu, il
  voit L'INVERSE du test — bitbrains_fast_storage/5T/medium ratio backtest de k=8 **15.28**
  (test : −9 %), 5T/long 3.19 (test −9 %), loop_seattle/5T/medium 1.40 (test −17 %),
  ett1/D 1.32 (test −25 %), jena/H/medium 1.12 (test −17 %). Et sur les wrong_k, le
  backtest est monotone en k (bizitobs_application : ratio 0.235 à k=3, encore plus bas à
  k=12) quand le test a un optimum intérieur (k=3/4) : le paysage backtest n'a pas la même
  forme que le paysage test. Mécanismes désignés : (a) **famine d'historique** — le
  backtest retire windows·h + h_bt pas du passé AVANT de décimer par k ; pour les termes
  medium/long des 5T (h ≥ 480), l'historique décimé par k=8 tombe à quelques patches →
  contexte dégradé → k pénalisé pour une raison qui n'existe pas au test (ratio 15 = un
  artefact, pas une mesure) ; (b) **non-stationnarité** (covid : k=4 ratio < 0.8 au
  backtest, k=1 au test — phases exponentielles) ; (c) fenêtre de backtest en régime
  différent du test (bizitobs). Conséquence : (a) est CORRIGEABLE (backtest à historique
  apparié : comparer k sur le même nombre de PATCHES que le test aurait, ou disqualifier
  un k dont l'historique de backtest est < 50 % de l'historique test) — c'est la
  prochaine (et dernière) itération raisonnable du sélecteur, P-bt4 à graver ; (b) et (c)
  ne sont pas récupérables causalement → xres-FiLM. **Statut** : flip+mix = pile
  officielle du champion (causal, ≤ ×4 passes) ; les comparaisons APPARIÉES entre
  checkpoints/bras restent en flip+backtest (moins cher, caches existants), le mix se
  pose sur le champion retenu. Position sub-10M inchangée : 4e (Toto 0.524).

- **2026-09-05 (HEAD8 : TABLE DOCTRINE COMPLÈTE + PREMIÈRE DÉCOMPOSITION DU RÉSIDU — trois
  cas à parts égales, covid seul pèse 17 %)** — Compagnons du champion head8 25 % :
  **nu 0.8877/0.6131 (couv 0.740) → flip 0.8543/0.5842 (0.781) → flip+RateIN
  0.7914/0.5433 (0.769)** ; oracle 0.7700/0.5190. Lecture à travers l'échelle (même
  procédure, 97 configs) :

  | lignée | params | nu | flip | flip+RateIN | oracle | couches (pts) |
  |---|---|---|---|---|---|---|
  | tiny mix (mix1ep3e4@25 %) | 1.14M | 0.6134 | 0.5984 | — | — | −3.6 (flip seul) |
  | tiny v3 (finetune @50 %, `epoch00_valloss0.5949`) | 1.14M | n.m. | 0.5983 | 0.5588 (mix-pool 0.5529) | 0.5358 | ≥ −4.5 |
  | mini std | 3.42M | 0.6235 | 0.5930 | 0.5469 | 0.5255 | −7.7 |
  | mini head8 | ~4.0M | 0.6131 | 0.5842 | 0.5433 | 0.5190 | −7.0 |

  Trois faits : (1) **le nu n'a PAS progressé de tiny à mini** (0.6134 → 0.6235 → 0.6131) :
  tout le gain d'échelle vit dans la RÉPONSE aux couches d'inférence (tiny −5.5 pts, mini
  −7.7) — la capacité achète de la composabilité, pas de la précision brute ; (2) la tête
  ×8 gagne surtout en nu (−1.04 pt vs std) et en flip (−0.88), le gain se comprime à −0.36
  sur la pile : couches et tête sont partiellement SUBSTITUTS (même motif que
  raffinement × centrage, 2026-08-31) ; (3) flip reste le calibrateur (couv +4.1 pts
  0.740 → 0.781), RateIN en rend 1.2. Conséquence pour le papier : la métrique nue
  sous-estime notre lignée de 7 pts et le classement inter-modèles en nu n'a pas de sens
  pour nous. **Décomposition du résidu (ratein_selection_gap, head8)** : 97 configs,
  geomean brut 0.1370 → 0.1308, résidu 4.7 % relatif (= les 2.43 pts de ratio, SN
  s'annule par config). Répartition : **missed 22 configs 32 % · wrong_k 13 configs
  38 % · false_pos 9 configs 29 %** · match 53 (0). AUCUN cas ne domine → aucun réglage
  de marge ne suffit (durcir la marge soigne les false_pos et aggrave les missed, et
  inversement). Un seul contributeur pèse 17 % : **covid_deaths false_pos k_bt=4, k*=1
  (0.0678 vs 0.0317)** — désaccord backtest↔test typique d'un régime non stationnaire
  (phases exponentielles) ; puis bizitobs_service/10S/medium wrong_k k=3 vs 16 (11.8 %),
  bizitobs_application/10S/long wrong_k 12 vs 3 (6.7 %), ett1/D missed k*=3 (6.6 %).
  Sous-split marge indisponible (caches antérieurs au champ `ratein.backtest`).
  **Candidat unique qui adresse les trois cas à la fois : MÉLANGE DE RANGS (RateIN-mix)** —
  au lieu d'un argmin + marge dure, pondérer les fans de plusieurs k (k=1 inclus, ratio 1)
  par w_k ∝ exp(−log ratio_k / τ) et moyenner les QUANTILES (Vincentization, préserve la
  finesse) : pas de seuil (missed), pas d'argmin (wrong_k), k=1 garde du poids
  (false_pos → covid divisé par ~2). Coût : ×(nb de k retenus) à l'éval, zéro
  entraînement, causal donc légal. Prédiction à graver avant la mesure : P-mix.1 récupère
  ≥ 1/3 du résidu (≤ 0.535) ; P-mix.2 couverture ≥ 0.769 (le mélange élargit le fan) ;
  ÉCHEC si ≥ 0.5433. **Implémenté le jour même (décision utilisateur « on va essayer »)** :
  `+ratein=mix` dans evaluate_gift.py — poids par config w_k ∝ exp(−ln ratio_k / τ), τ =
  0.05 (= l'ancienne marge : un k qui bat k=1 de la marge pèse ~e fois k=1), k=1 inclus à
  ratio 1, composantes < 2 % supprimées, 4 au plus, k disqualifiés (couverture < 2/3)
  absents ; fans des composantes moyennés en QUANTILES (Vincentization) ; garde
  par instance identique (composante abandonnée si historique décimé < patch, poids
  renormalisés). Coût ≤ ×4 passes. Cache `gift_flip_ratein-mix`, champ `ratein.mix`.
  Vérification : smoke CPU sur m_dense/D/short (tiny, 3 séries) — ancien script vs
  nouveau : backtest et nu BIT-IDENTIQUES (JSON clé par clé), mix = backtest quand tous
  les ratios > 1 (poids k1:1.00, attendu) ; 5 tests unitaires des poids.

- **2026-09-05 (INSTRUMENT : décomposition du résidu de sélection RateIN — préalable à tout
  « meilleur sélecteur »)** — Question utilisateur : le plafond 0.5190 laisse 2.43 pts au
  sélecteur causal, comment le rattraper ? Réponse de méthode : avant de changer le
  sélecteur, savoir OÙ il perd. `scripts/ratein_selection_gap.py` lit les deux caches d'un
  même checkpoint (backtest + oracle) et ventile le résidu (log-CRPS par config, agrégé en
  geomean) en quatre cas exclusifs : **missed** (k=1 gardé, oracle veut k>1 : marge,
  disqualification 2/3 ou aveuglement), **wrong_k** (k>1 des deux côtés, différents),
  **false_pos** (k>1 choisi, oracle à k=1), **match**. Contrefactuel « agrégat si ce cas
  était à la qualité oracle » par cas, et sous-liste des missed dont le backtest voyait un
  gain sous la marge de 5 %. `_backtest_series_k` retourne désormais aussi sa table de
  ratios poolés (cachée sous `ratein.backtest`, sans effet sur la sélection) — pour les
  runs futurs ; sur les caches head8 existants, la ventilation par cas fonctionne
  (k_hist), le sous-split marge non. Test synthétique (17 verts avec test_ratein).
  Candidats de sélecteur classés a priori, à trancher PAR la décomposition : (1) si
  « missed sous marge » domine → marge adaptative à la variance du ratio (1-SE rule
  réelle) ; (2) si wrong_k domine → MÉLANGE de rangs plutôt que sélection (fan moyenné
  des 2 meilleurs k pondérés par ratio : supprime la malédiction du gagnant, hedge
  backtest↔test, légal) ; (3) si false_pos domine → marge plus dure. Hors périmètre du
  sélecteur : les configs à k*=1 (m4, hospital, covid) → v4.

- **2026-09-05 (ORACLE HEAD8@25 % : 0.7700/0.5190 — le plafond de sélection est AU NIVEAU
  DE TTM-R3 ; tête ×8 = recette par défaut, décision utilisateur)** — Oracle-k (diagnostic,
  jamais officiel) sur le champion head8 : MASE **0.7700** / CRPS **0.5190** / couverture
  0.768 ; 35/97 configs gagnent > 5 % vs k=1. Contre l'oracle du mini standard
  (0.7744/0.5255) : −0.65 pt de CRPS de plafond — la tête large répond mieux aux entrées
  canonicalisées, comme RateIN composait déjà mieux à mini qu'à tiny. **Résidu de
  sélection** (backtest v3 0.5433 − oracle 0.5190) = **2.43 pts**, contre 2.14 sur le
  standard : le plafond a monté plus vite que le sélecteur ne le suit — le résidu
  capacité-orthogonal s'élargit avec la qualité du modèle, argument supplémentaire pour
  le conditionnement explicite (xres-FiLM) plutôt que pour un meilleur sélecteur causal.
  Lecture stratégique : le modèle, parfaitement sélectionné en taux, est DÉJÀ 2e des
  sub-10M (TTM-R3 0.520) ; tout l'écart à la 2e place est dans la sélection de taux et le
  MASE des historiques courts, pas dans la capacité. Configs à la traîne du plafond :
  bizitobs_service/10S/medium +65.6 % (k=16), bizitobs_l2c/5T/long +59.9 % (k=48),
  solar/10T +41-43 % (k=3/6) ; m4_*/hospital/car_parts/covid à k=1 (0 % : hors de portée
  de RateIN, c'est le territoire de v4). **Décision utilisateur : la tête ×8 (hidden 1536,
  ~4.0M params au total) devient la recette par défaut** — configs lotsa_mini_v4_{zeroshot,
  eval} héritent désormais de lotsa_mini_v3_head8_*, et lotsa_mini_xres_v3_{zeroshot,eval}
  reçoivent quantile_hidden_dim 1536 (le pretrain xres n'a pas de tête : inchangé). Les
  références appariées de P-v4 passent au head8 (P-v4.3 : MASE < 0.78, CRPS ≤ 0.5433 ;
  échec si MASE ≥ 0.7914). Scaling à 9M reporté (préférence utilisateur, cohérent avec E18
  et l'enveloppe −0.75 pt/doublement). **Ordre re-décidé (utilisateur)** : v4 D'ABORD
  (finetune d'une soirée depuis le pretrain existant, levier corpus validé seul), PUIS
  xres-mini (pretrain de deux jours, hérite de v4 + tête ×8), puis run final. Le finetune
  head8 est coupé après le 25 % : pas de 30 %, compagnons nu/flip à publier sur le 25 %.
  Tête ×8 dans v4 vérifiée par composition Hydra (quantile_hidden_dim 1536,
  short_series_windows true, data_dir lotsa_v4).

- **2026-09-05 (HEAD8@25 % : 0.7914/0.5433, couverture 0.769 — NOUVEAU CHAMPION ; P-head.1 ✓,
  P-head.2 tenue à la marge)** — Checkpoint apparié 25 % (val 0.6522), flip+ratein :
  MASE **0.7914** / CRPS **0.5433** / couverture **0.769** (q10 0.111, q90 0.880),
  RateIN 35/97 configs, 36.1 % d'instances k>1. Contre le champion standard au même
  point : 0.7994/0.5469/0.775 — **−0.8 pt MASE, −0.36 pt CRPS**, couverture −0.6 pt.
  P-head.1 (bat 0.5469 au 25 %) ✓. P-head.2 (couverture ne se dégrade pas) : le 0.734
  du 15 % s'est résorbé à 0.769 — l'érosion était de l'immaturité, pas un surajustement
  du fan ; tenue dans le bruit (−0.6 pt). Lecture : l'hypothèse d'allocation (G14) tient —
  la tête quantile (7 % du modèle en recette standard) était le goulot côté forecast ;
  ×8 la porte à ~19.5 % pour +1M params, zéro coût pretrain. Gains visibles où la forme
  compte : m4_quarterly 1.31 (vs 1.35 std), bizitobs_l2c/5T/short CRPS 0.078, sz_taxi
  0.20-0.21. TinyCast (0.545) est DÉPASSÉ pour la première fois ; prochain jalon TempoPFN
  0.533. Conséquences : (1) la tête ×8 devient la recette par défaut des bras suivants
  (xres-mini, v4, run final) — décision à confirmer par l'utilisateur ; (2) compagnons
  nu et flip à publier sur ce checkpoint (règle : toujours nu ET flip ET stack) ;
  (3) 30 % à évaluer pour confirmer le pic (même règle de fin que le standard).

- **2026-09-05 (CORPUS V4 CONSTRUIT : séries courtes réadmises, fenêtres à frontière,
  pinball masquée — le mécanisme A/Q/M/W annoncé le 2026-08-27 ; run gated)** —
  Diagnostic (code lu) : le bloc `lotsa_short` de v3 (`prepare_lotsa --min-length 384
  --pad-to 1280`, runbook S2.4 étape 3) rejette toute série plus courte — les yearly
  m1/m3/tourism en bloc ; `_generate_window_indices` saute toute ligne < ctx+pred ; h512 avait échoué par
  ce mécanisme exact (exigence gonflée ⇒ corpus jeté). **Défaut v3 découvert** : une
  ligne bourrée de r pas réels (256 ≤ r < 1280) produit des fenêtres à cible ENTIÈREMENT
  dans le bourrage plat (r=256 : 65 fenêtres sur 97) — le modèle apprend « contexte plat
  → cible plate », non masqué. **Construit** : (1) sidecar `_reallen/<fichier>.npy`
  (longueur réelle par ligne) écrit par prepare_lotsa quand --pad-to est donné — sous-dossier
  hors du glob `*.npy` ; (2) dataset sidecar-aware : lignes ≥ ctx+pred → fenêtres
  glissantes standard (cibles réelles garanties : cible ≥ ctx > longueur du pad) ; lignes
  plus courtes → PLUS de fenêtres standard (défaut v3 supprimé) et, avec
  `data.short_series_windows: true`, fenêtres à FRONTIÈRE : split dans les données réelles
  (≥16 pas de contexte réel, ≥4 de cible réelle), contexte bourré à gauche par la ligne
  elle-même, cible = queue réelle bourrée à droite (dernière valeur) et MASQUÉE au-delà,
  jamais de décimation ; `target_mask` émis pour tous les items quand le flag est on (règle
  tout-ou-rien du collate) ; (3) pinball masquée (moyenne sur les positions réelles ; ancre
  restreinte aux items à cible pleine ; perte ponctuelle masquée aussi) ; (4) FINETUNE
  SEULEMENT (train.py) : la perte JEPA n'a pas de masque de cible ; (5) témoin
  `aug/short_frac` ; (6) configs lotsa_mini_v4{,_zeroshot,_eval}. Défauts INERTES : sans
  sidecar ou flag off, fenêtres et dict d'item bit-identiques (9 tests nouveaux, suite
  complète relancée). **Prép v4 (corrigée en séance, question utilisateur sur la structure
  du corpus)** : lotsa_v3 est un dossier de SYMLINKS vers cinq sources (xres, synthetic_v3,
  lotsa_short, lotsa_solar, decimated) ; v4 ne refait QUE le bloc court
  (`lotsa_short_v4`, mêmes 12 subsets, `--min-length 24 --chunk-length 1280 --pad-to 1280`,
  sidecar écrit d'office) et réassemble les mêmes 106 noms + le lien `_reallen` —
  `docs/RUNBOOK_V4.md`, gates et P-v4.1..3 inclus. `audit_batch_schedule.py` plombé pour
  le flag v4 (il reconstruit le datamodule à la main). **Quelles
  familles entrent** : l'anti-fuite garde m4/hospital/car_parts/covid DEHORS (jeux GIFT) ;
  les overrides réadmettent m1_*, monash_m3_*, tourism_*, nn5_* — donc le levier agit par
  TRANSFERT du régime « historique court + cible courte » appris sur m1/m3/tourism vers les
  saigneurs GIFT ; bande de prédiction à graver au lancement en conséquence (plus
  incertaine qu'un apprentissage direct). Chargement : `datasets: null` + glob ⇒ familles
  réadmises chargées automatiquement. **Composition de batch (question utilisateur)** :
  oui, un audit ponctuel avec `scripts/audit_batch_schedule.py --config-name
  lotsa_mini_v4_zeroshot --mode finetune` après la prép (familles nombreuses à peu de
  fenêtres : interaction cap/rationnement à vérifier) + le témoin live short_frac ; pas de
  nouveau code. Ordre inchangé : head8 (25 % en cours) → xres-mini → v4 finetune (levier
  MASE) → contexte long → run final.

- **2026-09-05 (HEAD8@15 % : 0.7974/0.5466 — meilleur 15 % des trois bras, déjà sous le
  champion ; mais couverture 0.734, le drapeau P-head.2 est levé)** — Apparié 15 % :
  standard 0.7988/0.5482 (couv 0.770), aug 0.8015/0.5470 (0.732), head8
  0.7974/0.5466 (0.734). P-head.1 en bonne voie (verdict au 25 %, même règle de fin
  que l'aug). DÉMENTI PARTIEL de la prédiction consignée hier (« attente MASE
  faible ») : m4_yearly 4.72→4.19, m4_quarterly 1.35→1.31, solar/W 0.18→0.155 — la
  tête large aide aussi la forme extrapolée, pas seulement l'éventail. Revers :
  ett1/H/medium 0.26→0.32, et surtout la MÊME érosion de couverture que l'aug
  (−3.6 pts) — un gain CRPS au 25 % ne comptera que si la couverture ne paie pas
  l'addition ; à défaut, arbitrage à trancher explicitement (le critère officiel de
  sélection reste le CRPS, mais la calibration est un claim du papier).

- **2026-09-04 (SÉQUENCEMENT RE-TRANCHÉ, décision utilisateur : l'ingénierie AVANT le
  scaling — le gros run passe en DERNIER)** — Rationnel consigné : le scaling
  multiplie la recette qu'on lui donne (leçon du recadrage recipe-ceiling : la
  capacité n'a payé qu'une fois la recette v3 refaite) ; scaler maintenant = 5-6
  jours de 3090 pour multiplier une recette incomplète, puis re-scaler. Nouvel
  ordre : (1) head8 (en cours) ; (2) **xres-FiLM à échelle MINI** — déclencheur
  pleinement satisfait : résiduel oracle 2.14 pts stable à travers les échelles ET
  échec de l'exposition passive (aug) ; trio de configs lotsa_mini_xres_v3
  {,_zeroshot,_eval} créé et composé (miroir du trio tiny, duel une-variable contre
  le pretrain mini v3 ; P-xm.1..3 à graver au lancement) ; (3) chantier S4 : corpus
  re-chunké 8192 + crop-pad des séries courtes (le levier MASE direct : m4_yearly &
  co) + contexte long — le crop-pad reste À IMPLÉMENTER (seul vrai code manquant du
  périmètre) ; (4) LE gros run final (~7-9M) qui hérite de tout ce que 1-3 auront
  validé — un seul passage au tarif 3090, sur la meilleure recette.

- **2026-09-04 (ARM AUG CLOS : P-aug.3 ÉCHEC — pic propre à 15 % (0.5470, égalité
  champion), 25 % retombe à 0.5503 ; la branche échec-diagnostic s'active)** — Série
  aug complète : 0.5470 (15 %) → 0.5570 (20 %) → 0.5503 (25 %), couverture 0.732/
  0.720/0.740 (toujours sous le 0.775 du standard). P-aug.1 restait faiblement ✓
  (bizitobs/solar), P-aug.2 ✓, mais P-aug.3 jamais atteinte (ni 0.545 ni même
  <0.5469). Verdict mécanistique, comme gravé : l'exposition passive aux entrées
  décimées ne suffit pas — mini les encaissait déjà ; le résiduel de sélection
  (2.14 pts, oracle 0.5255) reste la propriété du dossier xres-FiLM (conditionnement
  explicite). Coût de la falsification : un finetune. Arm HEAD8 lancé (décision
  utilisateur, dose ×8 directe, P-head.1/2 s'appliquent). Question utilisateur
  consignée : l'écart MASE vs le peloton (0.7994 vs 0.70-0.77) vit dans m4_yearly/
  quarterly/weekly (4.5-5.3 / 1.35 / 2.4-2.7 de MASE), covid (35-40), bitbrains_rnd/H
  (6.0), saugeen/D (~3) — séries COURTES à tendance dominante + familles à outliers :
  un problème d'extrapolation de tendance sur historique court, pas de largeur de
  décodage — attente MASE du head8 : FAIBLE ; les leviers MASE = crop-pad + synthèse
  à tendance (corpus) et contexte long.

- **2026-09-04 (Aug@20 % : 0.8197/0.5570, couv 0.720 — le creux du 20 % se réplique ;
  verdict maintenu au 25 %)** — Même motif que le standard (0.5558 à 20 % avant le pic
  25 %) : creux répliqué, pas une cassure. L'avance appariée de l'aug (+0.12 pt à
  15 %) projetée au pic donnerait ~0.5455 ≈ TinyCast (0.5454) — le point 25 % est le
  seul qui peut trancher P-aug.3 ; règle de fin inchangée (éval 25 %, coupe à ~30 %,
  head4 ensuite). Couverture aug toujours en retrait (0.720). Note technique : le
  checkpoint aug 20 % s'appelle epoch00_valloss0.6558-v1.ckpt (collision de nom de
  val DANS le run aug, suffixe Lightning) — le harnais l'a rangé dans son propre
  répertoire, rien à faire.

- **2026-09-04 (ORACLE RE-PRICÉ sur le champion mini : plafond 0.7744/0.5255, capture
  v3 = 68 % ; le résiduel de sélection est ORTHOGONAL à la capacité — les deux leviers
  du fork restent vivants)** — Oracle-k (diagnostic, jamais officiel) sur le champion
  25 % : **0.7744 MASE | 0.5255 CRPS**, couv 0.778, 32/97 configs >5 % (34/97 à tiny).
  Détecteur v3 : 0.5469 → écart 2.14 pts (3.9 % rel.), contre 2.30 pts à tiny
  (0.5588→0.5358) : le gisement de sélection N'A PAS été absorbé par ×3 de capacité —
  il vit dans les mêmes configs famine-petits-n (bizitobs +50-66 % avec 2-84 inst,
  saugeen/D +33 % avec 1 série, solar/10T +42 %). Capture du backtest : 68 % (63 % à
  tiny) — RateIN compose mieux avec mini, cohérent avec l'entrée du 2026-09-03.
  **Lecture du fork base-vs-xres** : les territoires sont DISJOINTS et les deux
  exhibits mesurés. Base 7-9M : dé-risqué par deux points de scaling propres
  (1.14M→3.4M a payé partout), lève le corps, reste <10M ; espérance −1 à −1.5 pt
  (→ ~0.535). xres-FiLM : le résiduel 2.1 pts est STABLE à travers les échelles +
  l'au-delà-de-l'oracle (reinterp), mais plus de pièces mobiles (FiLM exercé,
  ratein_w). Ordre recommandé : head4 → (combo aug+head4 si les deux paient) → BASE
  d'abord (certitude, classe préservée) → xres ensuite. Teaser consigné : le plafond
  oracle du modèle ACTUEL (0.5255) se placerait entre Toto-4m (0.524) et TempoPFN
  (0.533) — les poids d'aujourd'hui contiennent déjà un Toto-class si la sélection de
  k était parfaite.
- **2026-09-04 (P-AUG, lecture provisoire au 15 % apparié : mieux de 0.12 pt mais
  couverture −4 pts ; le central 0.541 est mort, verdict au 25 %)** — Aug@15 % (pile
  complète) : 0.8015/0.5470, couv 0.732, contre standard@15 % 0.7988/0.5482 couv
  0.770 — et déjà l'égalité avec le champion 25 % (0.5469). P-aug.1 faiblement ✓ :
  les gains vivent exactement où prévu (bizitobs_l2c 0.256→0.241 et 0.307→0.293,
  solar/10T long 0.367→0.348) mais petits ; P-aug.2 ✓ modulo bruit bitbrains
  (fast/H 0.713→0.818, famille la plus bruitée) ; P-aug.3 : central 0.541±0.004
  improbable — le shift de distribution décimée achète ~0.1 pt, pas 0.5-1 : mini
  encaissait déjà bien les entrées mean-poolées. COUVERTURE en recul (0.732) : l'aug
  resserre le fan — à surveiller au verdict. Courbes de loss quasi identiques
  (observation utilisateur). **Règle de fin déclarée** : poursuivre le finetune aug
  jusqu'à ~30 %, évaluer 20 % et 25 % (le point de pic du standard = le point de
  verdict apparié propre) ; si aug@25 % ≤ 0.545 → P-aug.3 succès tardif ; sinon
  ÉCHEC-DIAGNOSTIC acté (voir des entrées décimées ne suffit pas) → coupe, head4
  prend le GPU, et le dossier xres-FiLM récupère le résiduel comme gravé. En
  parallèle (éval seule, pas de conflit GPU) : ORACLE re-pricing sur le champion
  standard mini — l'arbitre du fork base-vs-xres, toujours pas re-mesuré à cette
  échelle.
- **2026-09-03 (COMPAGNONS DU CHAMPION MESURÉS — la table doctrine est complète ;
  arm aug lancé)** — Champion mini 25 % (val 0.6528), les trois lignes officielles :
  **nu 0.8949/0.6235 (couv 0.736) → +flip 0.8612/0.5930 (couv 0.775) → +RateIN
  0.7994/0.5469 (couv 0.775)**. Contribution totale des couches d'inférence :
  −9.55 pts MASE, −7.66 pts CRPS. Deux observations : (1) RateIN compose MIEUX à
  l'échelle mini (−4.61 pts) qu'à tiny (−3.95) — la capacité améliore la réponse aux
  entrées canonicalisées ; (2) contrôle nu du 40 % (val-best 0.6495) : 0.8876/0.6204,
  légèrement MEILLEUR que le nu du champion — mais pire une fois les couches posées
  (0.5511 vs 0.5469) : le val-best re-démenti une 3e fois, et la sélection doit se
  faire SUR LA PILE COMPLÈTE, pas sur le nu. Notable pour le papier : solar/10T en nu
  est catastrophique (CRPS 0.81-1.33) et RateIN le divise par ~2.4 — la dépendance de
  la config hors-bande à la canonicalisation, chiffrée proprement. Arm G9.0-aug
  LANCÉ (P-aug.1..3 gravées hier) ; head4 en file.
- **2026-09-03 (SÉLECTION G7.3c CLOSE : champion mini = 25 % (0.7994/0.5469) ; série
  complète corrigée ; G14 câblé et vérifié ; P-aug et P-head GRAVÉES pour les deux arms
  suivants)** — Série finetune mini corrigée (mapping utilisateur) : 5 % 0.5585 →
  10 % 0.5517 → 15 % 0.5482 → 20 % 0.5558 → **25 % 0.5469 (val 0.6528, couverture
  0.775)** → 30 % 0.5539 (val 0.6596) → 35 % 0.5627 (couverture effondrée 0.701,
  m4_yearly instable) → 40 % 0.5511 (val-best 0.6495 mais GIFT pire — le proxy val
  re-démenti) → 45 % 0.5514. Quatre déclins consécutifs post-25 % : règle d'arrêt
  remplie, le 50 % = complétude seulement. Reste : compagnons nu et flip-only du
  champion. **G14 câblé** : ForecastingHead/loading.py/train.py exposent
  `decoder.quantile_hidden_dim` — clé NEUVE à dessein : 8 configs héritées portent un
  `decoder.hidden_dim` DÉCORATIF jamais plombé (le réveiller aurait cassé le chargement
  des checkpoints existants — attrapé en vérification d'inertie : mini eval 0.251M
  au byte près, tiny 0.118M, head4 0.473M) ; configs lotsa_mini_v3_head4_{zeroshot,
  eval} créées ; 31 tests ciblés verts. **P-aug (gravées, lancement imminent, pretrain
  val-best 0.5495)** : P-aug.1 les configs majoritairement décimées à l'éval
  s'améliorent vs champion apparié ; P-aug.2 les configs k=1 stables ±1 % ; P-aug.3
  agrégat flip+ratein ≤ 0.545 succès (passe TinyCast), central 0.541±0.004 ;
  ÉCHEC-DIAGNOSTIC si ≥ 0.5469 → voir des entrées décimées ne suffit pas, le dossier
  xres-FiLM reprend le résiduel. **P-head (gravées)** : P-head.1 flip+ratein bat
  0.5469 au checkpoint apparié (~25 %) ; P-head.2 la couverture ne se dégrade pas
  (une tête large ne doit pas acheter du CRPS en surajustant le fan) ; échec → l'écart
  résiduel est mécanisme/corpus, pas capacité côté forecast — arm clos au prix d'un
  finetune.
- **2026-09-03 (CHAMPION CRPS au 25 % : 0.7994/0.5469 — TinyCast à 0.3 % ; creux au
  20 %, trajectoire bruitée autour d'une pente descendante)** — Série finetune mini
  complète : 0.5585 (5 %) → 0.5517 (10 %) → 0.5482 (15 %) → 0.5558 (20 %, creux — val
  remontée aussi 0.6570) → **0.5469 (25 %, val-best 0.6528)**. Lecture corrigée en
  séance : l'utilisateur lisait le 25 % comme un déclin — c'est le meilleur CRPS du
  projet (MASE 0.7994, +0.6 pt de bruit vs 15 %). Critère de sélection G7.3c rappelé :
  le CRPS classe le leaderboard, la MASE à ±0.001 est du bruit. Motif tiny (pic
  finetune ~50 %) : poursuivre les évals jusqu'à ~50-60 %, arrêt sur 2-3 déclins CRPS
  consécutifs. covid_deaths/D : le backtest OSCILLE entre checkpoints (k=1 au 20 %,
  k>1 aux autres) — bruit de sélection inter-checkpoints déjà consigné, surveillance
  maintenue.
- **2026-09-03 (CHAMPION ckpt-3 (15 %) : 0.7988/0.5482 — les barres 0.80 et 0.55
  tombées ; TinyCast à 0.5 %)** — Troisième checkpoint (epoch00_valloss0.6558), pile
  complète : **MASE 0.7988 | CRPS 0.5482**, couverture 0.770. Série par checkpoint :
  0.5585 → 0.5517 → 0.5482 (deltas −0.68/−0.35 pt : décélération ~géométrique →
  asymptote estimée du run 0.542-0.548, passage de TinyCast 0.5454 ≈ pile-ou-face en
  fin d'epoch). MASE sub-0.80 : première fois du projet. La sélection G7.3c continue
  sur les checkpoints restants ; le champion final devra ses compagnons nu et
  flip-only.
- **2026-09-02 (Conformité leaderboard vérifiée aux sources : flip+RateIN sont légaux)**
  — Règles gift-eval (repo officiel) : contraintes sur l'USAGE DES DONNÉES (zéro fuite
  test, pas de train sur les splits GIFT pour le label zero-shot, adaptation par dataset
  = fine-tuned), AUCUNE restriction sur les procédures d'inférence. Pratique :
  t0-alpha/TFC publie NU (notebook de réplication lu : 1 forward, ctx 8192, zéro TTA) ;
  TinyCast inclut sign-flip averaging ET alignement de période FFT (= notre pile,
  0.545 au leaderboard) ; TimesFM expose force_flip_invariance ; FlowState règle son
  pas interne depuis la saisonnalité FOURNIE par le dataset. Notre pile est conforme
  (uniforme, causale, zéro regard test — les modes oracle restent interdits d'officiel).
  Obligations à la soumission : déclarer la procédure dans la description du modèle ;
  code de réplication (evaluate_gift.py l'est).
- **2026-09-02 (CHAMPION ckpt-2 : 0.8005/0.5517, couverture 0.795 — le peloton
  goia/Kairos/Metamorph est doublé)** — Deuxième checkpoint de finetune mini
  (epoch00_valloss0.6581), pile complète : **MASE 0.8005 | CRPS 0.5517**, couverture
  **0.795** (nominal 0.800 — calibration quasi parfaite, record). Progression par
  checkpoint : 0.8106/0.5585 → 0.8005/0.5517 alors que la val de finetune n'a bougé
  que de 0.6609→0.6581 — la sélection G7.3c par éval GIFT reste le bon instrument, la
  val de finetune est un proxy faible. Position <10M : 6e — devant goia 0.553, Kairos
  0.554, Metamorph 0.555 ; restent TinyCast 0.545 (à 1.2 %), TempoPFN 0.533, Toto
  0.524, TTM 0.520, FlowState 0.487. covid_deaths/D toujours décimée (k>1 100 %,
  CRPS 0.057 vs 0.112 au ckpt-1 — s'améliore mais surveillance maintenue). Barres
  utilisateur pour la suite de l'epoch : 0.80 MASE et 0.55 CRPS. RAPPEL pour le
  champion final sélectionné : publier aussi nu et flip-only (doctrine), et relancer
  l'hybride P-J.3 n'est PAS requis (juge = pretrain, indépendant du finetune).
- **2026-09-02 (NOUVEAU CHAMPION : mini ckpt-1 + pile complète = 0.8106/0.5585 — bat le
  champion tiny sur les DEUX métriques, au premier checkpoint de finetune)** — Pile
  officielle (flip + ratein=backtest) sur epoch00_valloss0.6609 : **MASE 0.8106 | CRPS
  0.5585** vs tiny 0.8152/0.5588 ; couverture **0.774** (vs 0.748) ; 37/97 configs
  décimées, 38.1 % d'instances k>1. La composition tient à l'échelle : flip 0.5911 →
  pile 0.5585 (−3.26 pts). P-mini se renforce : la capacité paie ET compose avec les
  couches d'inférence. Sub-0.56 atteint, TinyCast (0.545) à 2.5 %, et la sélection
  G7.3c parmi les checkpoints restants de l'epoch n'a pas commencé. **Watch-item** :
  covid_deaths/D régresse sous ratein pour mini (33.65→49.82 MASE, CRPS 0.037→0.112,
  k>1 100 %) là où le backtest de tiny choisissait k=1 — la sélection par config est
  bruitée AU CHANGEMENT DE CHECKPOINT sur les configs limites ; coût ~+1.1 % d'agrégat
  (sans elle ~0.552). Détecteur GELÉ, pas de changement de règle — consigné comme bruit
  de sélection inter-checkpoints, à surveiller sur les checkpoints suivants. **Roadmap** :
  arm G14 head-width ajouté au PLAN (décision utilisateur — audit d'allocation : tête
  10.4 %→7.3 % en part avec l'échelle, 118K ≈ TinyCast entier ; gated post-campagne).
- **2026-09-02 (P-MINI, PREMIÈRE LECTURE : le 1er checkpoint de finetune mini BAT le
  champion tiny final — la capacité paie)** — Finetune depuis le val-best 0.5495,
  premier checkpoint (epoch00_valloss0.6609), flip pur : **0.8568 MASE / 0.5911 CRPS**
  vs tiny champion 0.8633/0.5983 (−0.65 pt MASE, −0.72 pt CRPS, conditions égales).
  P-mini.2 confirmée dans les per-config : le gain vit dans le CORPS — m4_yearly
  4.75→3.93 MASE (CRPS 0.158→0.135), famille m4 en baisse générale, m_dense/H/short
  0.23→0.16 ; queue mitigée (bizitobs_application dégradé 0.046→0.096, solar/10T
  long/medium améliorés) — le partage corps/queue attendu, la queue reste le travail
  de RateIN. **Couverture 0.780** (vs ~0.72 tiny) : meilleur intervalle 80 % jamais
  mesuré en flip pur — la capacité calibre mieux. Suite : flip+ratein sur ce checkpoint
  (composition ~−2.4 pts sur tiny → ~0.567 projeté), puis sélection G7.3c parmi les
  checkpoints de l'epoch de finetune. Le 0.55 est en ligne de mire.
- **2026-09-02 (MINI CONVERGÉ : coupe du pretrain actée, P-J.3 ✓ (0.6513), finetune
  P-mini lancé depuis le val-best 0.5495)** — Trois instruments concordants : (1) val
  loss en plateau depuis ~400k pas, plancher 0.5495 (~900k), REMONTÉE à 0.5591 ensuite —
  la condition de coupe gravée (« plateau confirmé en fin de décroissance ») est remplie ;
  (2) juge convergé, série appariée 4 points : 0.6748 (15 %) → 0.6514 (30 %) → 0.6508
  (0.5495) → 0.6513 (0.5591) — **P-J.3 ✓** (0.6513 ≤ 0.652, bande de succès ; central
  0.648±0.003 raté de +0.003) ; (3) représentations : pred_std 0.58 / target_std 0.90 /
  context_std 0.87, tous MEILLEURS que tiny à corpus égal (la capacité enrichit le
  latent), MAIS effective_rank 61→30 en baisse continue — même signature que l'érosion ρ
  des probes ; les pas restants n'achètent plus rien. **Décision** : pretrain coupé ;
  finetune mini depuis le checkpoint VAL-BEST 0.5495 (miroir de la recette tiny, dont le
  champion est né du @50 % — la sélection de checkpoint de pretrain est la pratique
  établie, pas une entorse) ; contrôle depuis 0.5591 seulement sur motif si déception.
  Le run finetune est LE verdict P-mini (flip vs tiny 0.5983), puis flip+ratein =
  candidat champion (le 0.55 se joue là). Juge du papier : primaire = dernier checkpoint
  nommé (0.5591, zéro sélection, 0.6513) ; le val-best 0.6508 étiqueté « sélection
  déclarée ».
- **2026-09-01 (ARM SELF-HYBRIDE CLOS : le contrôle T confirme — dilution à T=1,
  COLLAPSE à T=0.25)** — Run de contrôle utilisateur (15 configs mixtes, T=0.25) :
  hybrid_self/self = 0.8563/0.5462 = **1.57** (pire que 1.17 à T=1) et couverture 80 %
  effondrée à **0.339** (nominal 0.800). Mécanisme symétrique : à T=1 le pool DILUE le
  fan calibré ; à petit T la masse se concentre sur quelques trajectoires et le fan
  COLLAPSE (perte de l'étalement, qui est la valeur même d'un fan). Même solar/W, seul
  gain franc, régresse entre T=1 et T=0.25 (0.128→0.167). Les deux directions du seul
  bouton échouent pour des raisons opposées → STRUCTUREL, pas un réglage : la
  pondération de Gibbs sur candidats convient à un proposeur PONCTUEL (elle fabrique
  une distribution : TTM 0.7258→0.6508), pas à un proposeur qui émet déjà une
  distribution calibrée. **Phrase du papier** : l'uplift du juge vient de la
  décorrélation avec un proposeur externe ponctuel ; re-pondérer son propre fan
  calibré ne peut que le diluer (T=1) ou l'effondrer (T petit). Arm clos pour 2 runs
  GPU-légers ; le mode --proposer self reste en code (table d'ablation, no-delete).
  Roadmap éval inchangée : fin d'epoch mini → finetune → flip puis flip+ratein
  (P-mini) → hybride TTM final (P-J.3, harnais gelé) → arm G9.0-augmentation (P-aug).
- **2026-09-01 (VERDICT SELF-HYBRIDE : P-SH.1 ✓ (comparateur corrigé), P-SH.2 ✗✗ à
  +17 % = DILUTION P-SH.3, la branche prévue ; contrôle T exposé)** — Run utilisateur
  (proposeur champion tiny flip+ratein, juge mini 0.5495, centered, 97 configs).
  **Lecture du baseline** : self 0.8183/0.6440 vs LOCAL SN se compare à 0.8147/0.6295
  (ligne vs_local du champion), PAS à 0.815/0.559 (SN officielle) — les 3 axes du
  harnais hybride (SN locale, plafond 150 inst/config, skip des cibles à NaN partiel)
  sont les MÊMES qui expliquaient le « TTM sous-performant » (0.7475 vs 0.7240).
  P-SH.1 ✓ en bord de bande : MASE +0.4 %, CRPS +2.3 %, dérive incarnée par
  bitbrains_fast/5T/long (10.227 sur 63 inst sous-échantillonnées vs 0.858 sur 4 860
  en officiel — bruit de sous-échantillon queue-lourde). **P-SH.2 ✗✗** :
  hybrid_self/self = 0.7541/0.6440 = 1.17, très hors [0.98, 1.02] → P-SH.3 (dilution)
  s'applique. Mécanisme : TTM gagnait car le pool transforme un POINT en distribution ;
  le champion EST déjà une distribution calibrée — la re-pondération de Gibbs sur
  trajectoires + bootstraps la délaye. Confirmation décorrélation : les seuls gains
  nets sont où le proposeur est FAIBLE (solar/W −41 %, electricity/H long/medium,
  m_dense/H, ett1/H long) — le juge aide où il est en désaccord informatif avec le
  proposeur. **Contrôle avant clôture (protocole P-SH.3)** : flag --temperature exposé
  (T sur énergies standardisées, anti-dilution E18e) ; UN run de contrôle déclaré sur
  sous-ensemble (~15 configs, T ∈ {0.5, 0.25}) ; si la dilution persiste à petit T,
  l'arm self-hybride est CLOS avec la phrase du papier : le gain du juge vient de la
  décorrélation avec un proposeur externe ponctuel, pas du re-mélange de son propre
  fan calibré. **Régularisation de l'éval (question utilisateur)** : le harnais
  hybride reste GELÉ tel quel jusqu'au run final P-J.3 (comparabilité de la série de
  juges appariés 0.6748/0.6514/0.6508) ; après P-J.3, soit il reste étiqueté
  diagnostic-apparié-jamais-leaderboard (son rôle), soit UN changement déclaré
  l'aligne (masquage NaN officiel + plafond levé). Ne jamais citer un chiffre hybride
  contre un chiffre leaderboard sans les 3 axes en note.
- **2026-09-01 (SELF-HYBRIDE livré : le champion propose, le pretrain juge — P-SH.1..3
  gravées ; question utilisateur : le juge peut-il améliorer NOTRE 0.559 ?)** — Mode
  `--proposer self` câblé dans evaluate_gift_hybrid.py : le proposeur devient notre
  champion finetuné avec sa pile officielle (fan flip + RateIN par backtest causal via
  `--proposer-ratein`), le pool reçoit les 9 trajectoires du fan + 4 chemins MC-dropout
  (recette E18d mesurée, réutilisée telle quelle), le juge pretrain pondère. Nouveau
  reader `self` = le FAN complet du champion → hybrid-vs-self est une comparaison
  appariée au niveau fan (impossible avec TTM, point-only). Aucun modèle externe,
  granite-tsfm non requis en mode self. Smoke mécanique 2 configs : self 0.6094 local
  (classe champion, couches composées) ; l'hybride y dégrade car le smoke utilise un
  juge FINETUNÉ (E18b : alignement détruit, probe 0.409) — plomberie validée, science
  non testée. **P-SH (gravées AVANT le run, juge = mini pretrain val 0.5495)** :
  P-SH.1 (validation interne) : reader self ≈ classe champion sur le sous-échantillon
  (CRPS local cohérent avec 0.6295 du run officiel ±0.02). P-SH.2 (LA question) :
  attente honnête = gain FAIBLE OU NUL, hybrid_self/self ∈ [0.98, 1.02] — juge et
  proposeur partagent la lignée corpus/objectif, leurs erreurs sont corrélées, alors
  que le gain TTM (0.7258 point → 0.6508) vit de la DÉCORRÉLATION ; un résultat nul est
  publiable tel quel (le mécanisme du juge = décorrélation d'erreurs). Si gain > 2 % :
  le système auto-contenu s'améliore lui-même → rouvrir le dossier « officiel
  2-checkpoints » (doctrine mono-checkpoint à re-trancher explicitement). P-SH.3
  (anti-dilution, leçon G12c) : si hybrid_self/self > 1.02, c'est la dilution du fan
  par le pool bootstrap — regarder la température avant de conclure. Coût : mode self
  plus lent que TTM (flip + 4 forwards dropout + backtest par config), prévoir
  plusieurs heures sur les 97.
- **2026-09-01 (Post-reprise mini : val 0.5495 NOUVEAU PLANCHER ; juge 0.6508, 3e point de
  la série inversée ; P-J.3 gravée. Nettoyage/merge master clos)** — Checkpoint
  post-reprise epoch00_valloss0.5495 : le plateau val des checkpoints 20-30 %
  (0.5545/0.5544/0.5550) est CASSÉ — la décroissance cosine paie, la décision de
  continuer le pretrain (au lieu de couper à 30 %) est validée par la mesure.
  **Juge (hybride centered apparié, bras TTM bit-identique 0.7649/0.7258)** :
  hybride@0.5495 = **0.6508**, meilleur juge mesuré — série à trois points :
  0.6748 (15 %) → 0.6514 (30 %) → 0.6508 (post-reprise). Le mûrissement s'aplatit
  (−2.3 pts puis −0.06 pt) : convergence vers ~0.650. Probe PLATE en parallèle (0.241,
  solar ρ(E,MAE) −0.19) → le découplage probe/juge tient sur trois points appariés.
  Couverture hybride ~0.595 stable quel que soit le juge (0.598@15 %, 0.595 ici) :
  propriété du pool TTM, pas du juge. **P-J.3 (gravée AVANT le run final)** :
  hybride@fin-d'epoch ≤ 0.652 succès, central 0.648±0.003 ; ÉCHEC si > 0.655
  (le mûrissement s'arrête avant la fin → sélection du juge par hybride sur les
  checkpoints intermédiaires). **Clôture du chantier nettoyage (décisions utilisateur)** :
  sota-roadmap réécrite en 31 commits thématiques (arbre byte-identique, auteur unique
  IUseAMouse, zéro trace de conversation dans code et historique — traces « user
  decision/option A-B », shebangs et trailers purgés), 348 tests verts, mergée
  fast-forward dans master et poussée ; historique complet conservé sous le tag LOCAL
  archive/sota-roadmap-pre-squash (jamais poussé). Les nouveaux commits n'emportent
  plus de trailer (doctrine ownership). Le crédit de collaboration vit dans la section
  Acknowledgments du papier, seule place décidée pour lui.
- **2026-09-01 (Pré-nettoyage : RateIN×w livré (gated), configs G9.0-aug créées, correctif
  v2/v2.1 au registre, crédit Claude au papier)** — Derniers morceaux de code du périmètre
  avant la passe de nettoyage/squash (décision utilisateur). **RateIN×w** : flag
  `+ratein_w=true` — sur les buckets décimés (1<k≤4, gamme entraînée de la FiLM,
  log₂w ∈ [−2,2] ; au-delà repli standard), le fan est demandé DIRECTEMENT au taux natif
  via w=1/k, h_fc=h, zéro ré-interpolation (la seule perte que même l'oracle ne peut
  éviter) ; w plombe `tta_forecast` (miroir : même w, le taux est invariant par négation) ;
  refus bruyant sans FiLM ou sans mode ratein actif ; tag `-w` ; test relais w (25 verts
  au total). GATED : n'a de sens qu'après un finetune xres. **Configs G9.0-aug** :
  `lotsa_{tiny,mini}_v3_aug_zeroshot` — mêmes facteurs [1,2,4] p 0.3 que xres
  (la paire aug-seul vs xres-complet ne diffère que du bundle FiLM/w/ancre),
  cross_resolution false, ancre 0, composition vérifiée ; P-aug.1..3 esquissées en
  en-tête, chiffres à graver AU LANCEMENT. **Papier** : section Acknowledgments ajoutée
  (collaboration Claude/Anthropic décrite honnêtement, direction et validation par
  l'auteur), recompilé. Décisions utilisateur actées : tag d'archive avant squash,
  retrait des auteurs Claude des commits au moment du squash (ownership légal du repo),
  anciennes versions PDF/gif non nécessaires dans l'historique (sauvegarde distante faite).
- **2026-09-01 (P-J.1 RÉFUTÉE ET INVERSÉE : la probe ne prédit pas le juge ; probe
  RÉTROGRADÉE en diagnostic ; verdict pretrain = CONTINUER jusqu'au bout de l'epoch)** —
  Runs hybride appariés (bras TTM bit-identique 0.7649/0.7258, centered, seed 0) :
  hybride@30 % (probe 0.241, « pire ») = **0.6514, MEILLEUR juge jamais mesuré en
  hybride** ; hybride@5 % (0.211) = 0.6554 ; hybride@15 % (probe 0.205, « meilleure ») =
  **0.6748, la pire des trois** — ordre INVERSE de P-J.1. La clause gravée s'applique :
  la probe (classement inter-instances, 6 configs) ne mesure pas la qualité opérante
  (discrimination intra-pool sur propositions TTM) → rétrogradée en diagnostic, sélection
  du juge PAR HYBRIDE désormais. Preuves du découplage dans les per-config : le juge@15 %
  est activement NOCIF en pool (bitbrains_fs/5T short : TTM 0.556 → hybride 0.880 ;
  bizitobs_service/10S short : 0.818 → 1.136) là où le juge@30 % reste sobre ; et
  solar/10T long s'AMÉLIORE en pool au juge@30 % (TTM 1.048 → 1.007) alors que sa probe y
  passait ρ négatif. Le motif « early peak du juge » ne décrit que la métrique de probe,
  PAS le métier de juge — la qualité opérante est plate-à-croissante avec le pretrain.
  **VERDICT PRETRAIN (question utilisateur : arrêter ?) : CONTINUER jusqu'à la fin de
  l'epoch.** (1) La pièce maîtresse du dossier « couper » — dégradation du juge — vient
  d'être réfutée au point d'arrivée ; (2) le plateau val (0.5544→0.5550) n'est pas une
  preuve, la décroissance cosine (~90 % du LR de pic à 30 %) n'est pas jouée ; (3) le
  verdict P-mini gravé exige la recette complète, et chaque branche de la règle de
  décision déclarée menait de toute façon à la reprise — le finetune diagnostic
  intermédiaire perd sa raison d'être, ANNULÉ : finetune unique en fin d'epoch, éval
  flip puis flip+ratein. Idée utilisateur (tester le dernier checkpoint) = la bonne.
- **2026-09-01 (Mini probes 15-30 % : PIC DU JUGE À 15 % (0.205, meilleur mesuré) ;
  finetune diagnostic déclaré ; P-J.1/2 gravées — le proxy probe passe au banc d'essai)**
  — Probes standalone mini : 5 % 0.211, 10 % 0.212, **15 % 0.205** (agrégat tronqué par
  tmux, recalculé = moyenne des 6 configs — MEILLEUR JUGE JAMAIS MESURÉ, ckpt 0.5575 au
  coffre), 20 % 0.223, 25 % 0.238, 30 % 0.241. Réplication à ×3 de capacité du motif
  tiny : le juge vit tôt, la capacité ne déplace pas le pic. La dégradation post-pic est
  PORTÉE PAR solar/10T (0.453→0.570, ρ(E,MAE) passe négatif) ; electricity/H reste
  excellent (0.028-0.062) — dégradation localisée hors-bande, pas un effondrement.
  Val pretrain plateau (0.5545/0.5544/0.5550 à 20/25/30 %) MAIS cosine à ~90 % du LR de
  pic : la phase de décroissance n'est pas jouée — le plateau ne condamne pas le
  forecaster. **Décision d'allocation déclarée** : pause à 30 %, finetune DIAGNOSTIC
  depuis le meilleur val (25 %, 0.5544 — mini n'a vu que ~30 % du corpus : comparaison à
  tiny confondue par construction, jamais un verdict P-mini) ; flip ≤0.590 = signal de
  capacité net (victoire malgré 3× moins de tokens) → reprise du pretrain (resume câblé,
  fdbcd46 ; caveat : le stream repart du début, ~30 % revus — entorse mineure consignée) ;
  égalité = ambiguë (tokens vs capacité), reprise quand même ; coupe définitive seulement
  si égalité à 30 % PUIS plateau confirmé en fin de décroissance. **P-J.1/2 (gravées
  AVANT les runs hybride — test de validité du proxy probe, question utilisateur)** :
  le juge opère dans l'hybride TTM (discrimination intra-pool), la probe est un
  classement inter-instances sur 6 configs — l'ordre doit se transférer. P-J.1 (ordre) :
  hybride@15 % ≤ hybride@5 % (0.6554 mesuré) < hybride@30 %. P-J.2 (magnitude) :
  hybride@15 % ∈ [0.645, 0.655] ; hybride@30 % ∈ [0.66, 0.69]. Si hybride@30 % ≤
  hybride@15 % : la probe ne mesure PAS la qualité opérante du juge → rétrogradée en
  diagnostic, sélection du juge directement par hybride. Si l'écart hybride est faible
  dans les deux sens : cohérent avec une dégradation localisée solar (l'hybride moyenne
  sur toutes les configs).
- **2026-09-01 (RateIN v3 : CHAMPION 0.8152/0.5588, P-RIN.7 agrégat ✓ ; DÉTECTEUR GELÉ ;
  les 3 limites résiduelles nommées = le dossier xres)** — Run utilisateur v3 (k par
  config poolé, 97 configs) : **MASE 0.8152 | CRPS 0.5588** (coverage 0.748, 34/97
  configs décimées en tout-ou-rien). Trajectoire finale de la campagne RateIN :
  0.5983 → 0.5793 (v2) → 0.5682 (v2.1) → **0.5588 (v3)**, plafond oracle 0.5358 —
  **capture 63 %**, 2e prédiction centrale touchée d'affilée (0.5588 ∈ 0.555±0.004).
  Sous-clauses : ✓ jena/D réparée (k=1), ✓ famille des 22 régressions v2.1 essentiellement
  éteinte (bitbrains ×5 bit-identiques, covid/electricity/m4 recalés) ; ✗ « aucune config
  >+2 % » : ett1/H/long 0.271→0.377 (+39 %, 7 séries), ett2/H/medium +13 %,
  loop_seattle/5T/medium +7 %, sz_taxi/15T/medium +4 % ; ✗ bitbrains ≤0.75 : le pool
  REFUSE la décimation (bit-identique, ni gain ni perte) ; abandons de gains v2.1
  par-série : ett1/D 0.289→0.344(=flip), ett2/15T/long 0.095→0.104(=flip). **Les 3
  limites résiduelles, nommées** : (1) FAMINE petits-n — ett/bizitobs à 2-21 séries, le
  pooling n'a rien à pooler, le winner's curse revient ; (2) DÉSACCORD backtest↔test —
  loop_seattle/5T/medium : le pool choisit k4 et le test réalise EXACTEMENT la colonne
  oracle k4 (0.094), mais le test préfère k1/k12 : le biais de fenêtre (la fin du passé
  n'est pas le régime du test), irréductible pour TOUT sélecteur causal ; (3) MISMATCH
  d'objectif — geomean équipondérée des ratios par série ≠ CRPS config pondéré par |y|
  (bitbrains : k16 aide l'agrégat via les grosses séries, le pool équipondéré dit non).
  **DÉCISION : détecteur GELÉ à v3** — corriger (3) reviendrait à imiter l'objectif de
  l'oracle (Goodhart), (1) et (2) sont des limites d'information, pas des bugs.
  [CORRECTIF 2026-09-01 : contrairement à la phrase initialement écrite ici, les
  sélecteurs v2/v2.1 ne sont PLUS en code — v3 les a remplacés dans la même fonction ;
  leurs résultats restent préservés (JSONs _ratein-bt_v2/_v21) et leur code vit dans
  l'historique git (tag d'archive à poser avant tout squash). Modes en code :
  fft / backtest-v3 / oracle.] Le résiduel
  0.5588→0.5358 (−4.1 % rel.) est structurellement HORS DE PORTÉE d'une sélection externe
  causale : c'est l'exhibit chiffré du dossier train-side (G9.0-augmentation d'abord —
  35 % des instances d'éval sont désormais des entrées décimées JAMAIS VUES au train —,
  FiLM-w ensuite, qui rend l'adaptivité par instance sans sélection donc sans curse).
- **2026-09-01 (Diagnostic v2.1 : WINNER'S CURSE par série ; v3 = k par config poolé,
  P-RIN.7 gravée)** — Le croisement k_hist×per_k_crps (script utilisateur) nomme le
  mécanisme des 52 % non capturés : l'argmin PAR SÉRIE sur 11 candidats avec 1-2 fenêtres
  bruitées sélectionne les coups de chance — jena_weather/D +163 % (14/42 instances à k=16,
  le PIRE k de la table oracle 0.146), et 22 régressions à +4-12 % (bitbrains ×5, covid,
  electricity ×3, ett ×5, loop ×2…) sont la même erreur à faible dose. Second symptôme :
  les MÉLANGES de k par série sous-performent le k uniforme sur les paysages accidentés
  (bitbrains_fast_storage/5T medium : mélange 0.896 vs ≤0.837 pour TOUTE la colonne
  oracle). **v3 (2e itération post-oracle, mécanisme nommé)** : k PAR CONFIG — ratio
  pinball k/k=1 par série, geomean entre séries (normalisation ôte échelle/difficulté),
  argmin + marge 5 % ; variance ÷ n_séries, granularité = celle de l'oracle (qui borne la
  capture) ; k coté sur <2/3 des séries disqualifié (sous-ensemble biaisé) ; garde par
  instance inchangée. Précédent : s_Δ par dataset de FlowState, en causal. Smoke : 4/4
  témoins corrects. **P-RIN.7 (gravée AVANT le run)** : agrégat ≤0.560 succès, central
  0.555±0.004 (capture ≥65 %) ; plus AUCUNE config >+2 % vs flip apparié ; bitbrains
  fast_storage/5T medium ≤0.75 (flip 0.801, mélange v2.1 0.896) ; jena/D revient ≤0.055 ;
  ÉCHEC si >0.5682 → champion reste v2.1 et détecteur GELÉ définitivement, décision
  G9.3/mini sur les chiffres v2.1.
- **2026-09-01 (RateIN v2.1 : CHAMPION 0.8154/0.5682, P-RIN.6 RÉUSSIE ; capture 48 % de
  l'oracle)** — Run utilisateur flip+backtest v2.1 (97 configs) : **MASE 0.8154 | CRPS
  0.5682** (coverage 0.742, 43/97 majoritairement décimées, 43.9 % k>1). Trajectoire du
  jour : 0.5983 (flip) → 0.5793 (v2) → **0.5682 (v2.1)**, plafond oracle 0.5358 — capture
  48 % de l'écart, à 0 GPU d'entraînement. **P-RIN.6 : agrégat ✓** (barre ≤0.575 ET central
  0.570±0.004 battus) ; **faux positifs coarse-freq réparés ✓** (m4_daily/m4_quarterly/
  m4_weekly/electricity_W à 0 % k>1, covid 58→32 %) ; **clause bizitobs_application ✗**
  (CRPS toujours 0.052 ≈ flip, oracle ~0.023 : 100 % k>1 mais TOUJOURS le mauvais k —
  le plus gros poisson reste au fond). Victoires v2.1 : bizitobs_l2c/5T long 0.571→0.307,
  electricity/15T long 0.130→0.093, electricity/W 0.109→0.073, solar/10T long 0.477→0.399,
  loop_seattle/5T long 0.093→0.079, us_births/M 0.022→0.015. Régressions locales à
  élucider : bitbrains_fast_storage/5T medium 0.727→0.896 (séries spiky — le backtest
  sur-adapte sa fenêtre ?), bizitobs_l2c/H long 0.321→0.377, saugeen/W 0.419→0.461 (la
  marge a tué un vrai gain). Discipline : P-RIN.6 non échouée → itérations détecteur encore
  PERMISES mais chacune doit nommer son mécanisme (pas de descente de gradient sur le test) ;
  prochaine étape = diagnostic par k_hist (JSON bt) × per_k_crps (JSON oracle) sur les
  configs à pire capture, PUIS décision : une itération ciblée vs bascule G9.3/mini.
  Le flip pur n'a plus de run sur le pod, mais per_k_crps["1"] de l'oracle EST le flip
  apparié — les scripts d'analyse s'appuient dessus.
- **2026-09-01 (RateIN v2 OFFICIEL : NOUVEAU CHAMPION 0.8393/0.5793 ; oracle-97 corrigé
  0.7894/0.5358 ; v2.1 livrée, P-RIN.6 gravée)** — Run utilisateur flip+backtest sur les 97
  configs : **MASE 0.8393 | CRPS 0.5793** (vs 0.8633/0.5983 flip seul, −3.2 % relatifs CRPS,
  coverage 0.731, 55/97 configs majoritairement décimées, 55.2 % d'instances k>1).
  **P-RIN.5 RÉUSSIE en v2** (barre ≤0.590, meilleur que le central 0.587±0.004) — après
  l'échec du v1 FFT (0.6022) : c'est la SÉLECTION qui était mauvaise, pas le mécanisme.
  Oracle complété sur 97 (les 6 configs réparées disent toutes best_k=1) : plafond vrai
  **0.7894/0.5358** — le 0.5232/91-configs d'hier était bien gonflé par les exclusions.
  **Capture v2 ≈ 30 %** de l'écart flip→oracle. Deux ratés diagnostiqués dans les logs :
  (1) h_bt plafonné à 256 = le collapse de rollout INVISIBLE à la sélection (à h_bt≤256,
  k=1 ne rollout jamais dans le backtest → bizitobs_application/10S long : 100 % k>1 mais
  CRPS ≈ flip, mauvais k choisi) ; (2) une seule fenêtre bruitée = faux positifs sur les
  fréquences grossières (m4_daily 3.89 vs flip 3.48, m4_weekly 2.73 vs 2.47, covid 58 %
  k>1). **v2.1** (une itération déclarée, principielle — fidélité à la tâche + réduction de
  variance, pas de tuning) : h_bt = h réel (repli si historique court), jusqu'à 2 fenêtres
  moyennées, marge no-op 5 % (esprit 1-SE ; les vrais gains oracle sont à +20-50 %). Smoke :
  electricity/H revient bit-identique au flip (faux positif tué), bizitobs long décime,
  solar plein gain, us_births k=1. **P-RIN.6 (gravée AVANT le run v2.1)** : agrégat flip+bt
  ≤ 0.575 succès, central 0.570±0.004 ; m4_daily/m4_weekly/covid_deaths/electricity_D
  reviennent à ±1 % du flip seul ; bizitobs_application/10S capture ≥ 50 % de son gain
  oracle ; ÉCHEC si > 0.579 (v2.1 pas mieux que v2) → stop itérations détecteur, décision
  G9.3/xres sur les chiffres v2. **TTM brut (--ttm-only)** : 0.7475 vs SN officielle (96
  configs) vs claim leaderboard 0.7240 — l'écart est l'APPROXIMATION DU WRAPPER, pas un
  handicap TimeJEPA : le wrapper saute toute instance à cible partiellement NaN (le
  protocole officiel ne saute que les 100 % NaN) + les contextes où TTM émet des NaN →
  comptes effondrés (hierarchical_sales/D 0 inst → nan → exclu, temperature_rain 3460 vs
  96212, kdd/H long 2 inst) ; le chemin TimeJEPA suit le protocole transcrit avec les
  dénominateurs SN officiels (drift SN locale/officielle affiché à chaque run : 0.8393 vs
  0.8388). L'hybride papier reste valide (apparié par construction). CRPS point TTM jamais
  citable.
- **2026-09-01 (VERDICT RateIN v1 + ORACLE : le mécanisme est ÉNORME, le détecteur FFT
  est le maillon faible ; v2 backtest livré)** — Runs utilisateur sur le champion v3.
  **ORACLE-k : 34/91 configs gagnent > 5 %** (bizitobs jusqu'à +57 %, solar/10T +44-47 %,
  electricity/15T +14-24 %, loop_seattle/5T +21 %) — l'échec-diagnostic est ÉVITÉ de très
  loin, la géométrie d'échelle EST le mécanisme dominant de la queue. ⚠️ L'agrégat oracle
  (0.7836/0.5232) n'est PAS comparable au flip-only (0.8633/0.5983) : 6 configs ont
  crashé (IndexError, historiques courts décimés à vide — corrigé par garde k=1) et leur
  exclusion (m4_yearly 4.14 de MASE incluse) gonfle l'agrégat ; comparer sur
  l'intersection. L'oracle révèle aussi un SECOND mécanisme que la période ne voit pas :
  bizitobs/H gagne +40 % à k=16 sur un cycle de 24 — c'est le COLLAPSE DE ROLLOUT (h'=30
  en un forward), pas la canonicalisation. **Détecteur FFT v1 : P-RIN.1 ✓ (solar réalisé
  0.608→0.562), P-RIN.3 ✓ (bizitobs_l2c réalisé 0.605→0.367), P-RIN.4 ✗✗ (D/W/M
  sur-décimés : us_births/D 0.583→1.530, m4_daily 3.48→4.56 — le pic annuel des séries
  journalières déclenche k=8-16 là où l'oracle dit k=1), P-RIN.5 ✗ (agrégat 0.6022 >
  flip-only 0.5983 : les catastrophes coarse-freq mangent les gains).**
  **RateIN v2 livré (+ratein=backtest)** : le k par SÉRIE choisi par BACKTEST CAUSAL —
  rejouer les k candidats sur les h_bt derniers pas du passé (jamais le test, la fenêtre
  précède la première cible d'éval), garder le meilleur pinball ; batché, ~|K| mini-passes
  d'une fenêtre par série ; la logique calibration-T d'E18h appliquée à k. Capture LES
  DEUX mécanismes sans métadonnée. Smoke : us_births/D reste k=1 (0.634 vs 1.530 en v1),
  solar garde son plein gain (0.485). Modes exposés : fft (v1) / backtest (v2) / oracle ;
  garde anti-crash partout ; relancer la commande oracle complète les 6 configs
  manquantes (marqueurs). Déclencheur G9.3 : ARMÉ (jus oracle + v1 qui bute), mais le
  backtest passe d'abord — s'il capture l'essentiel de l'oracle, la falsification de
  xres continue à 0 GPU.
- **2026-09-01 (G9.3 IMPLÉMENTÉ : RateIN@éval + xres-amendé ; prédictions P-RIN gravées
  AVANT tout run complet)** — Verdict de prémisse (agent architecte, mandat « contredire
  xres si l'analyse y mène ») : xres-nu a 3 défauts structurels (w=1 à l'éval GIFT ⇒
  capacité jamais utilisée ; gradient de pente FiLM nul au finetune, log₂(1)=0 — la loi
  de câblage E18b s'applique ; gamme w ≤ [1/7,7] apprise ~sur synthétique), et FlowState/
  TinyCast gagnent par CANONICALISATION D'ENTRÉE. Livré : **(A) RateIN@éval**
  (src/timejepa/evaluation/ratein.py + +ratein=true/oracle dans evaluate_gift) —
  détection de période causale (rfft + seuil de Fisher sur MAXIMA LOCAUX, zéro param,
  précédent TinyCast), règle « la PLUS PETITE période significative » (itérée UNE fois
  après smoke 2 séries : le pic dominant hebdo de electricity décimait la structure
  intra-journalière — itération DÉCLARÉE, pas de tuning au-delà), décimation seule vers
  [16,48] pas/cycle, h'=⌈h/k⌉, réinterp du fan, k=1 bit-identique (épinglé). Smoke
  exploratoire (2 séries) : solar/10T/short CRPS 0.786→0.516. **(B) G9.3 xres-amendé** :
  w plombé dans forward_finetune/forecast (défaut = T2 exact), paires xres au finetune
  (clés *_finetune, défauts inertes), ancre λ·MSE avec target_encoder ← copy_from(online
  chargé) (piège n°1 : sans la copie, l'ancre pointe vers le deepcopy ALÉATOIRE de la
  construction), refus linear_probe+λ>0, témoins aug/w_* + train_loss/anchor au
  finetune ; trio de configs lotsa_tiny_xres_v3 GARDÉ PAR DÉCLENCHEUR (queue post-mini
  encore inter-fréquences ET oracle RateIN > +5 % quelque part mais bute). 24 tests
  neufs verts (test_ratein 15, test_g93_xres_finetune 9).
  **PRÉDICTIONS RateIN (avant le run complet, champion v3, procédure ×flip+ratein)** :
  P-RIN.1 solar/10T CRPS −10 à −25 % ; P-RIN.2 ett/electricity 15T −5 à −15 % ;
  P-RIN.3 bizitobs_l2c/5T long/medium −20 à −40 % ; P-RIN.4 configs H/D/W/M/Q/A : k=1
  choisi ≥ 95 % des instances ; P-RIN.5 agrégat ×flip ≤ 0.590 succès, central
  0.587 ± 0.004. **ÉCHEC-DIAGNOSTIC (le test de falsification de xres, 0 GPU)** : si
  l'ORACLE-k lui-même ne gagne > 5 % sur AUCUNE config, la géométrie d'échelle n'est pas
  le mécanisme de la queue ⇒ G9.3/xres NON financé, la piste se clôt proprement.
- **2026-08-31 (RECADRAGE MAJEUR : « plafond de capacité » → « plafond de RECETTE » —
  amendé AVANT les résultats forecast de mini)** — Deux reviews externes + vérification à
  la source (arXiv fetchés, les chiffres des reviews étaient directionnels mais imprécis).
  FAITS VÉRIFIÉS : **FlowState-3M CRPS 0.496** (10.6M : 0.490 ; variante 3M-2k : 0.502 ±
  0.001 sur 3 graines) ; **TinyCast 146 505 params, nWQL 0.545** (périodicité CALCULÉE :
  FFT + test de Fisher, zéro paramètre ; flip par équivariance de signe = NOTRE TTA
  exact ; contexte 2048/champ 2047) ; ablation FlowState (3 graines) : sans équivariance
  d'échantillonnage 0.502→**0.553** (+0.051), sans parallel forecasts →**0.548** (+0.046) ;
  contexte d'entraînement FlowState 4096 ; corpus des deux = GiftEvalPretrain + synthétique
  (FlowState : CauKer). CONSÉQUENCES : (1) l'hypothèse « capacité liante à 1.14M » est
  RÉFUTÉE en lecture transversale — un 0.15M fait 0.545 ; la triple convergence
  0.597 ± 0.002 mesure le plafond de NOTRE RECETTE, pas de la taille. (2) **AMENDEMENT
  P-mini (gravé avant toute éval forecast de mini, pretrain à ~10 %)** : les bandes
  P-mini.1..4 restent telles quelles, mais l'ÉCHEC-DIAGNOSTIC change de lecture — flip
  > 0.593 ne dira plus « la capacité n'était pas le mur » (on le sait déjà par TinyCast),
  il dira « la capacité n'améliore pas CETTE recette » ; et un succès ≤ 0.580 dira « la
  capacité lève une part du plafond de recette », pas « la capacité était le mur ».
  (3) Le mur a maintenant un NOM et un PRIX, publiés par l'ablation d'un concurrent :
  équivariance d'échelle temporelle ~0.051 + multi-forecast parallèle ~0.046 ≈ 0.097 ≈
  notre écart total (0.094). Le levier n°1 devient **xres/G9.2** — implémenté, jamais
  couru, et c'est précisément la version JEPA-native de l'équivariance (la seule
  justification architecturale du JEPA qui a survécu à E15/E20b) — plus **contexte long
  ENTRAÎNÉ** (E19b l'avait déjà montré : bizitobs CRPS ÷2 à ctx2048 même hors
  distribution) et la piste périodicité-calculée de TinyCast (zéro paramètre, cousine de
  notre SN-dans-le-pool). (4) Corrections des reviews elles-mêmes : la « sur-exclusion
  auto-infligée » est mesurée à 0.9 % d'observations (G8.1 clos, levier vide) — ce point
  des reviews est faux chez nous ; TinyCast utilisant NOTRE flip renforce la note de
  fairness. (5) Fixes papier actés : confound de capacité NOMMÉ dans §objectif (le null
  est mesuré sous le plafond de la recette), abstract aligné sur la nuance du gate (le z
  achète la couverture nominale à CRPS égal, pas le CRPS), position de classe assumée
  dans la légende de la table leaderboard, baseline probabiliste pour le reranker
  (conformal sur résidus TTM) enregistrée comme expérience à courir. Rattraper 0.496 en
  solo : improbable ; la bande 0.52-0.55 : plausible par les leviers nommés ci-dessus.
- **2026-08-31 (mini @5 % : LE JUGE SUIT LA CAPACITÉ, et vite)** — Deux mesures sur le
  tout premier checkpoint mini-v3 (epoch00_valloss0.6836, ~5 % d'époque). (1) **Probe
  standalone : rang 0.211** (ρ 0.472, electricity/H méd 0.000) — meilleur juge à 5 %
  jamais mesuré (tiny-v3 @5 % : 0.233 ; pic esjepa : 0.205) ; seuil P-mini.4 (≤ 0.21)
  atteint au premier point ; solar-standalone ≈ hasard réplique (propriété de
  l'encodage, pas de la lignée). Question pic-tôt : si 10-15 % érodent, 4e observation,
  la plus précoce. (2) **GIFT hybride centré : le juge mini @5 % BAT le juge tiny
  FINAL** — hybrid 0.7834/0.6554/couv 0.555 vs 0.7939/0.6732/0.622 (instances
  appariées) ; taxe MASE +2.9 → +1.85 pts, fan vs point TTM effondré 5.3 → 7.0 pts.
  Couverture en recul MÉCANIQUE : juge plus discriminant ⇒ poids concentrés ⇒ fan plus
  étroit — le gap de couverture est un problème d'ÉTALEMENT DU POOL (innovations
  sous-dimensionnées aux longs horizons), que le meilleur juge révèle ; correctif côté
  génération de candidats, pas côté juge. Lecture d'ensemble : le pouvoir de jugement
  scale avec la capacité ET émerge en heures — double soutien à pic-tôt et à
  l'hypothèse capacité, avant même le premier point forecast de mini.
- **2026-08-31 (VERDICT H512 : ARM CLOS, coupé à 40 % ; mini lancé)** — Trajectoire
  ×flip : 5 % 0.9105/0.6423 · 30 % 0.8870/0.6137 · 35 % 0.8922/0.6194 · 40 %
  0.8921/0.6121. Plateau 0.612-0.619, jamais sous 0.61 — avec l'amputation corpus
  (+11.1 % sur 21 configs, ~2 pts d'agrégat structurels), h512 ne peut pas menacer le
  0.598 de h256. Verdict en trois lignes : (1) levier horizon RÉEL mais PETIT
  (différentiel −1.7 % multi-rollout vs single-roll — le signe d'E20b, l'ampleur d'un
  non-levier) ; (2) coût de fenêtre dominant (bloc décimé 1024/682 + lotsa_short 1280
  hors du finetune 1536) ; (3) **DONNÉE INATTENDUE : couverture 0.800 EXACTE au 40 %**
  (q10 0.102 / q90 0.901) — meilleure calibration jamais mesurée sur la lignée, candidat
  mécanisme : la randomisation d'horizon [64..512] apprend à la tête à calibrer à
  travers les horizons. Ingrédient à recycler en S4 (horizon randomisé large SANS
  étendre la fenêtre). Coupe réversible (save_top_k -1). MINI-V3 lancé dans la foulée
  (bundle et prédictions P-mini.1..4 gravés à l'entrée précédente).
- **2026-08-31 (soir — G12c-sur-GIFT : LE POOL CENTRÉ TRANSPLANTE ; bundle mini-v3 prêt,
  prédictions gravées)** — Run GIFT hybride centré (97 configs, 150 inst, vs SN local) :
  hybrid_ttm MASE **0.7939** / CRPS **0.6732** / couverture **0.622** — contre ttmonly
  0.9098/0.7735/0.206 et TTM seul 0.7649 (point effondré 0.7258). Les deux mécanismes
  d'échec sont morts : taxe MASE +14.5 → **+2.9 pts**, et le fan bat le point effondré de
  TTM de **5.3 pts de CRPS sur les mêmes instances** (l'uplift point→probabiliste,
  apparié, citable). Catastrophes D/W disparues (ett1/D 1.68 vs 5.29 ; m4_yearly :
  l'hybride BAT TTM 3.72 vs 4.37). Restes honnêtes : sous-couverture 0.622 (bande
  prédite 0.70-0.80 ratée — les innovations saisonnières sous-estiment l'étalement aux
  longs horizons où le rollout TTM est déterministe après le 1er segment) et la taxe
  MASE résiduelle. G12 est CLOS pour le papier : uplift Nixtla 6/6 (2 lignées de juges),
  transplantation GIFT gardée et mesurée avec ses deux gaps quantifiés.
  **MINI-V3 : trio de configs créé** (lotsa_mini_v3 / _zeroshot / _eval), composition
  Hydra validée, **3 416 404 entraînables = ×3.01 vs tiny** (5.2M avec la copie EMA).
  Bundle = recette v3 à l'identique (corpus lotsa_v3, ration des deux côtés, TiRex,
  LR 3e-4, 2 ép. pretrain / 1 ép. finetune, save_top_k -1, cibles standalone) ; UNE
  variable : la taille (d192, enc 4 / pred 3, d_ff 768). Écart déclaré : batch effectif
  1152 vs 1536 (mémoire 24 Go). **PRÉDICTIONS GRAVÉES AVANT LANCEMENT** (aussi en tête
  de config) : P-mini.1 succès CRPS ×flip ≤ 0.580, central 0.570 ± 0.010, étirement
  < 0.555 ; P-mini.2 le gain vit dans le CORPS (81 configs), pas la queue ; P-mini.3
  plateau de rang effectif plus haut que tiny-v3 ; P-mini.4 pic-tôt du juge (4e
  observation, rang ≤ 0.21) ; **ÉCHEC-DIAGNOSTIC : ×flip > 0.593 ⇒ la capacité n'était
  PAS le mur, la lecture triple-convergence est réfutée, sans appel.**
- **2026-08-31 (raffinement × centrage : SUBSTITUTS, PAS COMPLÉMENTS)** — Run G12c +
  refine (10 pas, lr 0.5). Prédiction « deltas ≤ 0.01x » tenue sur 5/6 ; la violation est
  exchange : hybrid centré 0.78 → **0.82** (+4 pts) — le juge sculpte la texture au-delà
  du centre, et sur une série en tendance il la sculpte MAL (micro-Goodhart mesuré).
  Dans le MÊME run, le reader energy (pool non centré) réplique E18f en grand : exchange
  MASE ×1.55 → ×1.11, WQL ×1.36 → ×1.10 — la réparation d'enveloppe existe toujours,
  le centrage la rend redondante. Verdict : centrage et raffinement sont des SUBSTITUTS ;
  ligne officielle G12c = centré SANS refine. Décision connexe (question utilisateur) :
  xres r/r' NI maintenant NI dans mini — mini teste la capacité SEULE (une variable) ;
  le déclencheur de retour de xres est falsifiable : queue résiduelle per-config encore
  inter-fréquences après mini. Sinon il reste « designed, never funded » au papier.
- **2026-08-31 (G12c VERDICT : 6/6, LA DILUTION EST MORTE + verdict h512 intermédiaire)** —
  Run Nixtla (juge v3, --proposer-ttm --calibrate-T --centered-bootstrap), WQL vs SN
  ttm/hybrid : ettm1 0.88/0.77 · ettm2 0.81/**0.72** · etth1 0.85/**0.71** · etth2
  0.95/**0.78** · weather 0.63/0.64 (égalité, ≤1.02× ✓) · exchange 0.88/**0.78** (de
  1.08 à victoire de 10 pts ; MASE 1.22→0.89 ≈ TTM). **P-G12c.1 ✓ (les deux) ;
  P-G12c.2 ✓** (3 marges sur 4 AGRANDIES). La signature de dilution, 4 réplications,
  est éliminée par le centrage — c'était bien un problème de CENTRE. Témoin de
  cohérence : la calibration T choisit 4.0 sur weather/exchange (pool centré + rien à
  trancher ⇒ poids flats, plus d'amplification). Claim consolidée : un juge JEPA gelé
  1.1M transforme TTM-R3 en probabiliste qui bat son WQL sur 6/6, zéro entraînement.
  **h512 (script de groupes, h512@30 vs h256@50)** : multi-rollout sain **0.995**
  (n=38) · single-roll sain 1.012 (n=38) · configs amputées **1.111** (n=21).
  Amputation corpus CONFIRMÉE (+11.1 % sur 10T + A/Q/M/W — bloc décimé 1024/682 et
  lotsa_short 1280 < fenêtre 1536) ; levier horizon RÉEL mais PETIT (différentiel
  −1.7 %, le signe d'E20b, l'ampleur d'un non-levier). Recommandation : ne pas
  réparer l'amputation (~1 pt de gain potentiel pour 1 h de pipeline + un re-run) —
  le gate mini est rempli par trois faits (corpus résolu, horizon testé et petit,
  capacité liante). Attendre la fin du run pour le verdict formel.
- **2026-08-31 (G12c IMPLÉMENTÉ, prédictions gravées avant run)** — `--centered-bootstrap`
  (evaluate_energy + evaluate_gift_hybrid) : candidats = chemin du proposeur + blocs
  rééchantillonnés des innovations saisonnières — même centre partout, dilution du centre
  impossible par construction ; drift hors du pool ; garde std(E)<1e-4 → poids uniformes.
  Smoke-test notable : proposeur volontairement décalé de +1 ⇒ le juge met sn à z=−3.2
  (il lâche un proposeur qui a tort — le mécanisme de sécurité espéré, observé).
  **Prédictions (protocole Nixtla E18g/h, juge v3, --proposer-ttm --calibrate-T
  --centered-bootstrap) : P-G12c.1** — weather et exchange cessent d'être des défaites
  (hybrid_ttm WQL ≤ 1.02× TTM sur les deux) ; **P-G12c.2** — les 4 victoires gardent au
  moins la moitié de leur marge (la texture centrée coûte moins que le bootstrap brut ne
  rapportait) ; **échec-diagnostic** : si weather dilue ENCORE à pool centré, le résidu
  n'est pas un problème de centre mais de texture (le juge préfère de mauvaises
  textures) — et la piste s'arrête là proprement.
- **2026-08-31 (suite — contre-vérification Nixtla : LE JUGE V3 RÉPLIQUE E18g)** —
  evaluate_energy.py, protocole d'origine (h=96 single-shot, pool bootstrap K=32 + 4
  jitters TTM, calibration T en contexte), juge = pretrain v3 final (0.6420). WQL vs SN,
  ttm / hybrid_ttm : ettm1 0.88/**0.73** · ettm2 0.81/**0.77** · etth1 0.85/**0.73** ·
  etth2 0.95/**0.83** · weather **0.63**/0.68 · exchange **0.88**/1.08. **4/6, les mêmes
  quatre victoires qu'E18g, dilution sur les deux mêmes datasets — QUATRIÈME réplication
  de la signature, avec un juge d'une AUTRE lignée.** Marges gagnantes plus grandes
  qu'E18g (ettm1 0.73 vs 0.84) ; sur ettm1 l'hybride égale la MASE de TTM (0.88) en
  écrasant son WQL. Verdict du diptyque GIFT/Nixtla : l'écart de transplantation est
  100 % PROTOCOLE (rollout + pool dégénéré + pas de cascade), 0 % juge. Claim papier
  consolidée : deux juges de deux pretrains répliquent l'uplift sur le protocole
  single-shot ; l'échec GIFT est mécanisé ; pondération par source = correctif enregistré.
  Sortie : evaluation/energy_nixtla/epoch01_valloss0.6420_h96.json.
- **2026-08-31 (G12-sur-GIFT, variante ttmonly : NÉGATIF MÉCANISÉ)** — Pool ancres (SN +
  drift) + 16 chemins TTM jitterés, juge v3 final (epoch01_valloss0.6420), 97 configs
  plafonnées 150 inst : ttm seul MASE ratio **0.7649** (vs 0.7240 CSV officiel : harnais
  validé à +5.6 % sur fenêtres plafonnées) ; hybrid_ttm **0.9098 / CRPS 0.7735 /
  couverture 0.206**. DEUX mécanismes identifiés, pas une défaite du juge : (a) TTM
  CONTRACTE le bruit d'entrée — 16 contextes jitterés donnent 16 chemins quasi identiques,
  pool sans dispersion, fan quasi-point (couverture 0.206, CRPS ≈ le point effondré TTM
  0.7258) ; (b) énergies quasi identiques → standardisation par σ_E minuscule → le softmax
  amplifie du bruit et pose la masse sur les ANCRES — catastrophes localisées pile sur les
  configs D/W où le drift linéaire diverge (ett1/D 1.63→5.29, ett1/W 1.58→4.97, ett2/W
  0.86→2.85, kdd/D 1.19→4.99). Le run re-prouve le levier n°1 promu en E18h : pondération
  PAR SOURCE (prior par famille de candidats, l'énergie arbitrant à l'intérieur), jamais
  construit. Donnée annexe : TTM émet des NaN sur 100 % de hierarchical_sales/D et
  restaurant/D (0 instance) — notre modèle y tourne. Statut papier : le PENDING hybrid-ttm
  devient un négatif propre (« ne se transplante pas naïvement sur GIFT ») ; les claims
  EBM qui tiennent restent la sonde, le gate +18.7 et l'uplift Nixtla E18d-h sur SON
  protocole. Contre-vérification lancée : evaluate_energy.py (protocole E18g/h, h=96
  single-shot) avec le juge v3 — réplique = écart 100 % protocole. À réclamer aussi :
  summary.json du run pool-complet (jamais transmis).
- **2026-08-31 (démo vidéo, et un piège cousin du last.ckpt)** — Démo de com
  `scripts/forecast_video.py` : forecast d'une vidéo pixel par pixel (chaque pixel = une
  série univariée, éclatement par canal comme le harnais) et en mode POD/PCA (forecast des
  k coefficients modaux du contexte — « prédire dans une représentation », la thèse du
  papier en démo). Deux scènes générées : pendule RK4 et allée de von Kármán (LBM D2Q9,
  v2 calibrée : warmup 20k + perturbation initiale, période 61 frames, autocorr 0.95 — la
  v1 à warmup 4k n'avait JAMAIS déclenché le lâcher, champ quasi statique, détecté par
  persistance MAE 0.0069 et autocorr sans pic AVANT tout verdict modèle). Batterie sur le
  champion mix nu (mix1ep3e4_25pct), ratio MAE vs persistance / couverture 80 :
  pendule pixel **0.728 / 0.786** · pendule PCA-16 **0.340 / 0.885** · vortex pixel
  **0.194 / 0.790 (nominale !)** · vortex PCA-12 **0.135 / 0.581** (sous-couverte :
  l'hypothèse d'indépendance inter-modes du MC est optimiste, déclarée). Le pixel-par-pixel
  sur objet translaté = régime intermittent binaire (impulsions sous le patch 16 :
  médiane pinball-optimale ~0, mesuré) ; la PCA le répare (coefficients = échelle
  d'harmoniques lisses). Champion v3@50 sur pod (runs utilisateur, mode pixel) :
  ratio 0.581, couverture 0.814 ; flip DÉGRADE (0.597, cov 0.952) — prior de symétrie de
  signe FAUX pour une intensité bornée à 0, jolie illustration que le TTA est un prior.
  **PIÈGE découvert (candidat §5)** : un checkpoint de PRETRAIN porte une tête quantile
  présente mais JAMAIS entraînée (le JEPA ne la traverse pas), et le contrat P3.2 ne
  couvre que le cœur — le chargement passe et sort du bruit plausible. Mesuré : corr 0.00
  sur sinusoïde nue (pretrain 0.5520) vs **1.00** (champion finetuné) ; une soirée de
  chiffres locaux invalidés (dont un faux « PCA décevant »). Discriminant fiable :
  `hyper_parameters` Lightning (`finetune_mode` vs `contextualized_targets`) ; garde-fou
  de refus ajouté à la démo. Leçon répétée : la baseline aussi doit prouver qu'elle est
  ce qu'elle prétend être.
- **2026-08-30 (soir, finetune v3 @75 %)** — ×flip : **0.8636/0.5988**, couverture 0.751
  (q10 0.123 / q90 0.874). Deuxième meilleur checkpoint du run, à l'épaisseur du bruit du
  50 % (0.8633/0.5983) : ΔMASE 0.0003, ΔCRPS 0.0005. Lecture : (1) PAS de dérive tardive
  façon mix/ration — la séquence 50→70→75 % fait 0.5983→0.6005→0.5988, un plateau qui
  respire, cohérent avec val encore descendante (0.5936 à 75 %) ; le pic-25-45 % des
  lignées précédentes ne se reproduit pas sur v3 (rationnement au pretrain ET au finetune,
  composition stationnaire — mécanisme G10.2 cohérent). (2) QUATRIÈME point sur le plancher
  0.597 ± 0.002 ×flip : la triple convergence devient un plateau de run entier, la lecture
  « capacité liante » se renforce. (3) Couverture 0.751 < champion 0.769 < esjepa 0.790 :
  v3 est la lignée la plus sous-couvrante — argument supplémentaire pour l'ablation
  « z sur pretrain sain » (E21 question ouverte). (4) Indices per-config bruts (ratios à
  confirmer par gift_gap) : m4_yearly MASE 4.14 et m4_quarterly 1.32 re-confirment
  P-v3.2 affaiblie ; car_parts CRPS 1.015 brut, la famille intermittent n'a pas encore
  visiblement payé ; us_births/M 0.534 MASE semble réparé (était dans la queue E19).
  Verdict P-v3.1/3.3/3.4 : attendre l'autopsie gift_gap sur le champion du run, pas
  config par config à l'œil. Reste : évals 80-100 %, puis autopsie, puis h512.
- **2026-08-30 (finetune v3 @30-50 %, et LA TRIPLE CONVERGENCE)** — 30 % :
  0.8791/0.6137 ; **50 % : 0.8633/0.5983** cov 0.746 — MEILLEURE MASE DU PROJET toutes
  lignées (champion 0.8702), CRPS à 0.24 pt du champion. Val ENCORE descendante à
  60-70 % (0.5936→0.5926) : lignée à pic tardif façon ration, verdict ouvert jusqu'aux
  évals 70-100 %. FAIT CENTRAL à nommer : mix 0.5959 / esjepa 0.5981 / v3 0.5983 —
  trois corpus/objectifs convergent vers **0.597 ± 0.002 ×flip** ⇒ la meilleure preuve
  à date que la CAPACITÉ (1.1M) est devenue la contrainte liante, pas les données.
  Décision (validée conversation) : (1) finir le verdict h256 (évals tardives + autopsie
  gift_gap vs P-v3.1..4 ; P-v3.4 nu ≤ 0.59 semble hors de portée, m4_yearly confirme la
  faiblesse P-v3.2) ; (2) h512 depuis le même pretrain (levier orthogonal — si medium/
  long cède, une part du « plafond » était du rollout, test E20b) ; (3) si rien ne casse
  ~0.59 : le gate de MINI est rempli par la triple convergence — « corpus résolu,
  capacité contrainte », dit par trois runs.
- **2026-08-29 (nuit, finetune v3 @10-15 %)** — ×flip 10 % : **0.8731/0.6042** cov 0.776 ;
  15 % : 0.8843/0.6079 cov 0.733 (wobble habituel). À 10 %, la MASE du champion final est
  DÉJÀ atteinte (0.8731 vs 0.8702) et le CRPS vaut celui de ration@~40 %. Projection
  honnête au rythme des lignées (−1 à −1.5 pt vers le pic 25-45 %) : best-of-run
  ~0.590-0.596 flip = NOUVEAU CHAMPION probable mais SOUS les bandes gravées (≤0.566) —
  sauf dynamique inédite, verdict pressenti « v3 bat le champion, rend moins que
  prédit », autopsie per-config au pic. Indices précoces : PRO-bundle solar/10T
  (short 1.10 MASE vs 1.4-1.8) ; CONTRE-bundle car_parts CRPS 1.07 (intermittent
  n'aide pas encore sa cible) et m4_yearly 0.146 (faiblesse P-v3.2 matérialisée où
  consignée).
- **2026-08-29 — PRETRAIN v3 COUPÉ à 1.4 époque (46h), COUPE RÉVERSIBLE, finetunes
  h256+h512 lancés** depuis `epoch01_valloss0.6420.ckpt` (état le plus récent, nommé,
  complet — optimiseur/scheduler inclus : reprise possible si le verdict suggère un
  sous-entraînement, l'asymétrie du piège esjepa a disparu grâce à save_top_k=-1).
  Justification : tout plat depuis ~500k ; 1.4 ép. v3 ≈ 2.9 époques-équivalent mix en
  volume (corpus ~2×) ; LR à ~40 % du pic (le finetune 1 ép. cosinus recuit).
  ⚠️ Lecture inter-runs : la val v3 (0.64) n'est PAS comparable à mix (0.575) — split
  de val à 54 % synthétique dont ops_bursty au plancher de mse structurellement haut.
  Santé de représentation : v3 = LA MEILLEURE des trois lignées (rang effectif 36-42
  sans plongeon, context_std ~0.84, variance VICReg la plus haute). Duel h256 vs h512 :
  même pretrain, l'écart medium/long mesure le levier horizon (gate du h768 natif S4)
  et croise l'hypothèse frontières-de-rollout (E20b).
- **2026-08-28 — G12-sur-GIFT : harnais hybride TTM×juge LIVRÉ**
  (`scripts/evaluate_gift_hybrid.py`, smoke-testé : rollout TTM continu sur 8 segments
  h=720, pool bootstrap+SN+drift+chemins TTM jitterés → fan monotone). Statut :
  EXPÉRIENCE PAPIER, jamais un chiffre officiel (bi-modèle > ligne G11). Trois lectures :
  ttm brut (MASE absolue vs CSV TTM-R3-PT vendoré = validation croisée du harnais ; CRPS
  point effondré, non citable), hybrid_ttm (LA mesure : un point forecaster externe
  devient probabiliste par juge latent — fan + couverture), champion cité des évals
  complètes. Appariement : mêmes instances plafonnées (~150/config) pour les deux
  readers ; agrégats vs SN local. Juge : primaire = checkpoint final v3 (zéro
  sélection), secondaire = juge probe-sélectionné (déclaré — le probe lit du GIFT).
  Prédictions AVANT run : (i) MASE ttm-brut à ±3 % du CSV officiel par config ;
  (ii) couverture hybride 0.70-0.80 sans calibration ; (iii) comparaison informative =
  hybrid vs champion (l'uplift vs ttm-point est trivial). À courir sur CPU quand v3
  libère du temps de cerveau — rien d'urgent.
- **2026-08-28 (nuit, série probe-juge v3 démarrée)** : 5 % **0.233** (ρ(E,MAE) **0.507**,
  record standalone) → 10 % 0.266 → 15 % 0.238. Le pattern PIC-TÔT se reproduit une
  TROISIÈME fois (mix : mi-run ; esjepa : 15 % ; v3 : ≤5-10 %) ⇒ promu LOI DU PROJET :
  le juge et le forecaster ont des optima temporels disjoints, le juge naît tôt — la
  sélection du checkpoint-juge (G12) se fait par probe précoce, jamais par val/GIFT.
  Records par config : electricity/H 0.032 (top20 0.99). Meilleur juge v3 à date :
  5 % (0.7461) — candidat champions/pretrain/ si rien ne le bat d'ici 30 %.
- **2026-08-27 (soir, prédictions complétées AVANT le premier checkpoint v3)** — en sus
  de P-v3.1..4 (E19), bandes chiffrées sur le CHAMPION v3 (finetune gelé, fenêtre
  25-55 %) : CRPS nu succès ≤ 0.590 (=P-v3.4), central **0.580 ± 0.008**, étirement
  < 0.570 ; CRPS ×flip succès ≤ 0.566, central **0.556 ± 0.010**, étirement < 0.545
  (passe la marche FLAIR/PatchTST 0.587 → zone ~75-80e) ; MASE nu ~0.868 / flip ~0.843 ;
  couverture ~0.77 (mesure). ÉCHEC DIAGNOSTIQUANT : nu > 0.605 (le bundle n'a pas rendu
  plus que la queue — autopsie per-config via P-v3.1/P-v3.3). **Probe energy** (série
  pretrain standalone) : succès rang ≤ 0.21, étirement ≤ 0.195 (nouveau meilleur juge) ;
  pattern attendu : pic du juge TÔT puis érosion (3e observation ⇒ loi du projet).
- **2026-08-27 (soir) — CORPUS v3 ASSEMBLÉ, GATE 6 VALIDÉ, PRETRAIN LANCÉ.** Pièce
  d'identité (audits avant/après dans evaluation/audit_v3_*.txt) : **106 familles**,
  **54.0 % de batch synthétique** (cible 50-55 ✓) dont ops_bursty 19.7 % (10 shards),
  intermittent ~9.9 %, subhourly 7.4 % (+dec), broadband 7.7 %, lowfreq 5.7 % ;
  stationnarité PARFAITE sur les 10 déciles (rationnement) ; queue réelle plafonnée
  24.6 % (era5/cmip6 ~1.2-1.3 % chacun, alibaba 0.65 %, solar_power 0.05 % = son
  plafond 3x en absolu ~1M fenêtres/époque) ; top réels : buildings 4.39 %,
  largest_* ~3 %, borg/azure ~2.5 %. Ajustement chirurgical post-audit-1 (55.2 %) :
  retrait des symlinks lowfreq_dec3 + broadband_dec3 (subhourly_dec3 GARDÉE — le trou
  10T/15T). Un checkpoint par 5 %, tous conservés (save_top_k=-1), sans z (E21-b),
  augmentations TiRex ON. Verdict attendu ~2 j : P-v3.1 (bizitobs<1.0), P-v3.2
  (affaiblie), P-v3.3, P-v3.4 (agrégat ≤0.59 avant couches).
- **2026-08-27 (assemblage, deux recalibrages consignés)** : (1) **séries courtes** —
  min-length 384 rejette en bloc m1/m3-partiel/tourism (médiane < 384 : une série
  annuelle de ~30 pts ne peut pas fournir une CIBLE réelle de 256 pas ; le pad-to est
  correct, la géométrie est la contrainte). 111 morceaux réels seulement ⇒ **P-v3.2
  AFFAIBLIE** (repose sur lowfreq + 111 morceaux) ; le vrai mécanisme A/Q/M/W =
  augmentation crop-court+pad à l'entraînement, NON construit (itération suivante si le
  bloc reste rouge). (2) **décimation** — familles réelles xres en morceaux 2048 ⇒
  seuls 8192/4096 décimables : 6 fichiers synthetic_*_dec + 6 chronos_*_dec. Effet de
  bord : les dec synthétiques comptent dans la part ⇒ prédiction gate 6 relevée à
  ~53-56 % ; si > 55 %, ajustement par RETRAIT DE SYMLINKS (jamais de fichiers).
  P-v3.3 s'appuie désormais surtout sur subhourly(+dec) + chronos_dec + solar_power.
- **2026-08-27 (après-midi) — DIMENSIONNEMENT v3 DÉCIDÉ (étape 2 du runbook, cible
  utilisateur 50-55 %)** : l'audit avant (rationnement actif, stationnarité parfaite sur
  les 10 déciles) mesure synthétique v1 = **11.2 %** (3×3.74 %, libres). Le poids T=0.5
  étant en √ PAR FICHIER, la cible se prend par SHARDING (26 fichiers de 25k morceaux —
  précédent dans le corpus : era5/cmip6/largest) et non par volume (×76 de fenêtres
  sinon). Allocation E19 : ops_bursty ×10, intermittent ×5, subhourly +3, broadband +3,
  lowfreq +2 ; seeds 1-23 sans collision v2. Prédiction gate 6 : synth ~51 %, libres
  réelles ~25 %, queue plafonnée ~24 % (dilution 42→24 % ASSUMÉE : era5/cmip6 halvés,
  alibaba compensé par ops_bursty). Table par famille ajoutée à l'audit (b33eae7).
- **2026-08-27 (midi) — E21 CLOS : issue (b) DÉFINITIVE.** Le finetune ESJEPA est allé au
  bout ; série ×flip complète 50→100 % : 0.6048/0.6015/0.6047/0.6070/0.6086/0.6030/
  0.6034/0.6044/0.6038/0.6045 — le 45 % (0.5981, cov 0.790) reste le meilleur point de la
  fenêtre ET du run, jamais réapproché. Verdict par la règle pré-enregistrée : z HORS du
  bundle v3, ablation papier (gate conditionnel load-bearing, +18.7 pts gate-off,
  couverture nominale au pic), re-testable sur v3 en ablation avec départ propre — la
  question « z avec un pretrain sain » reste OUVERTE, pas réfutée. Champion esjepa archivé
  `champions/esjepa45_mase0.8739_crps0.5981.ckpt` (2e meilleur résultat du projet).
  GPU libéré → assemblage corpus v3 (runbook).
- **2026-08-27 (1h, ablation de clôture E21 : gate ÉTEINT sur le checkpoint 45 %)** —
  CRPS ×flip 0.5981 → **0.7853** (+18.7 pts !), couverture 0.790 → 0.968 (sur-couverture
  massive q10 0.014/q90 0.982), MASE ~identique (0.8734 vs 0.8739 ; écart 0.06 % attribué
  à l'enveloppe sur fans extrêmes, non creusé). **Le gate ne MODULE pas le fan : il le
  PILOTE.** Co-adaptation apprise au finetune : la base softplus = enveloppe quasi
  worst-case, le gate = resserrement CONDITIONNEL (g négatifs sur fenêtres calmes —
  l'inverse de l'élargisseur attendu). z est load-bearing et conditionnel — la ligne
  centrale de l'ablation papier. Nuance gravée : 18.7 pts valent À L'INTÉRIEUR de ce
  modèle ; le contrefactuel « entraîné sans z » = champion 0.5959 (la base se calibre
  seule quand le gate n'existe pas). Question utilisateur d'origine (« z ≈ 0 sur la
  couverture ? ») : RÉFUTÉE — à checkpoint égal, z fabrique la calibration entière.
- **2026-08-27 (1h, 45 % — LA RÈGLE TRANCHE : issue (b), « neutre-plus »)** — ×flip
  **0.8739/0.5981**, couverture **0.790 (+2.1 pts)**. CRPS ∈ [0.5959, 0.601] ✓ et
  couverture ≥ +2 pts ✓ ⇒ (b) : **z N'ENTRE PAS dans le bundle v3**, reste ablation
  papier (couverture comme résultat), re-candidat plus tard. À noter pour l'honnêteté du
  récit : parti du best-val 35 % handicapé, l'arm finit à 0.22 pt du champion, DEVANT
  l'ancien champion mix×flip (0.5984), avec une meilleure couverture — deuxième meilleur
  résultat du projet. Fenêtre 25-55 % encore ouverte ~2-3 h : le run continue pendant
  l'assemblage v3 (CPU) demain matin ; éval 55 % → verdict final (a)/(b) → kill →
  lancement v3.
- **2026-08-27 (0h, 40 %)** — nu **0.9075/0.6321** cov 0.755 ; ×flip **0.8791/0.6076**
  cov **0.794 (+2.5 pts vs champion — condition couverture de l'issue (b) REMPLIE)**.
  Quasi-parité avec ration à % égal (nu 0.6321 vs 0.6303 : 0.18 pt) — le handicap du
  départ 35 % est presque résorbé ; l'écart flip d'ESJEPA se resserre (−2.45 pt). Si le
  45 % reproduit le saut ration (−0.64 pt nu), atterrissage ~0.601 flip = PILE sur la
  frontière (b)/(c). Le 45 % tranche au dixième près.
- **2026-08-26 (nuit, 30 % + plan de coupe)** — ×flip 0.8921/0.6116 cov 0.761 : RÉGRESSION
  sur tout vs le 25 % (0.8820/0.6052/0.782), qui ressemble au pic local. Décision : couper
  APRÈS l'éval du 45 % (précédent ration : −1.06 pt de CRPS entre 25 et 45 % — la fenêtre
  a historiquement payé là ; il faut −0.93 pt pour l'issue (a), improbable pas exclu).
  Plan : assemblage v3 (CPU, runbook 0-6) en parallèle demain matin → éval 45 % → règle
  E21 tranche → kill finetune → lancement pretrain v3. Zéro idle GPU, fenêtre respectée.
- **2026-08-26 (nuit, 25 %)** — ×flip **0.8820/0.6052**, couverture **0.782** : meilleur
  saut MASE du run (−1.1 pt), la couverture REBONDIT (le 0.766 du 20 % était du bruit
  inter-checkpoints, pas une érosion linéaire), mais le CRPS cale à ~0.605. Cap sur (b)
  ou (c) — pour (b) il faut couverture ≥ +2 pts (actuel : +1.3). Prochains points : 35 et
  45 %.
- **2026-08-26 (nuit, 20 %)** — ×flip 0.8928/0.6056 (MASE et CRPS avancent encore, en
  décélération : deltas CRPS −1.36 → −0.46 → −0.19 → −0.19 pt) mais **l'avantage z FOND** :
  couverture 0.800 → 0.792 → 0.789 → **0.766** = niveau champion (0.769). Le finetune
  érode le gate (« z se fait démolir », lecture utilisateur exacte). Trajectoire pointe
  vers l'issue (c) de la règle E21 — v3 sans z, défaut déjà en place au runbook.
  Prochaines évals : 30 % et 45 % seulement.
- **2026-08-26 (nuit)** — **E21 : RÈGLE DE DÉCISION GRAVÉE AVANT LES RÉSULTATS** (validée
  par l'utilisateur, fenêtre 25-55 % du finetune ESJEPA, procédure officielle ×flip) :
  (a) **CRPS < 0.5959** ⇒ z GAGNE, entre dans le pretrain v3 ; (b) **CRPS ∈ [0.5959,
  0.601] ET couverture ≥ +2 pts vs champion** ⇒ « neutre-plus » : PAS dans le bundle v3
  (une variable de moins), reste ablation papier avec la couverture comme résultat,
  re-candidat sur v3 si besoin du chiffre calibration ; (c) **CRPS > 0.601** ⇒ v3 sans z,
  verdict attribué au bundle départ-35 % + λ_z, sans appel. Trajectoire au moment du gel :
  5 % 0.9208/0.6257 cov 0.800 → 10 % 0.8975/0.6121 cov 0.792 → 15 % 0.8983/0.6075 cov
  0.789 — la MASE CALE à ~0.898 (coût persistant du départ 35 %, z ne peut pas la
  toucher), le CRPS avance en décélérant, les acquis z tiennent. Gate en croissance
  (produit gate×z_head +13 % entre 5 et 10 %).
- **2026-08-26 (soir, suite)** — **Première lecture de l'instrument de couverture (×flip)** :
  champion intervalle 80 % = **0.769** (std 0.101, 62/97 dans [0.75, 0.85]) ; **ESJEPA
  @ 5 % = 0.800, le NOMINAL EXACT** (std 0.095, 55/97 dans la bande). Le critère de win
  « couverture qui généralise » est atteint EN MOYENNE dès le premier checkpoint — mais la
  version forte (par config) ne tient pas : le décalage fait sortir par le HAUT des
  configs déjà calibrées ⇒ l'élargissement est en partie UNIFORME (il vit dans le biais
  du gate, b_g = 0.039 — le gate a appris son propre γ global interne) et en partie
  conditionnel (solar −30 %, inexplicable par une constante). Décision utilisateur :
  PAS de run de contrôle sans-z (ablation excessive, budget) — l'attribution E21
  s'appuiera sur l'inspection du gate + les signatures étalement-vs-médiane ; ablation
  GRATUITE proposée en remplacement : annuler b_g dans une copie du checkpoint (éval CPU)
  pour séparer constant vs conditionnel. **MESURÉE dans la foulée (b_g → 0)** : la
  CONDITIONNALITÉ est réelle et dominante — solar garde intégralement ses −30 %
  (0.468/0.460/0.639 vs 0.469/0.459/0.641), CRPS quasi inchangé (0.6264 vs 0.6257) ;
  le biais est un correcteur de MOYENNE (sans lui, intervalle 0.813 = légère
  sur-couverture ; avec, 0.800 exact — les poids sur-élargissent un peu en moyenne, le
  biais retire la constante). Bonus : MASE 0.9208 au chiffre près entre les deux évals —
  l'invariance de médiane vérifiée en production par chirurgie de paramètre. Le duel
  sans-z devient inutile : l'attribution est faite, gratuitement.
- **2026-08-26 (soir)** — **Finetune ESJEPA @ 5 % : le GATE EST VIVANT** (témoin
  pré-enregistré vérifié par inspection du checkpoint — `z_gate.weight` absmean 0.072 /
  absmax 0.122, bias 0.039, depuis un zéro-init exact ⇒ modulation e^g de ±7-13 % réclamée
  par le gradient ; NB : gate_absmean n'était pas loggé au finetune, l'inspection de
  checkpoint fait témoin — à câbler dans finetune_module pour v3). Chiffres @5 % :
  0.9545/0.6526 nu, 0.9208/0.6257 ×flip — déficit concentré sur la MASE (m4_yearly 5.34,
  us_births/D 1.22, m_dense/D 1.21), que z ne PEUT PAS toucher ⇒ signature de l'état de
  départ 35 % peu recuit, pas de l'arm. Indice pro-z fort : solar/10T ×flip CRPS
  0.469/0.459/0.641 vs 0.651/0.730/0.743 champion (−30 % sur les configs
  hétéroscédastiques, étalement amélioré avec médiane dégradée = signature du gate).
  Question E21 reformulée : la médiane recolle-t-elle dans la fenêtre 25-45 % ? (⚠️ éval
  @5 % faite AVANT le pull : la couverture n'y est pas — pull fait pour la suite.)
- **2026-08-26 (midi)** — **Piège n°2 CONFIRMÉ RÉTROACTIVEMENT sur la lignée du champion**
  (md5 : `mix/last.ckpt` == `epoch01_valloss0.5550`, qui n'est même pas le best-val
  0.5520 mais la DERNIÈRE entrée du top-3). Le protocole de fait du projet a toujours
  été « finetune depuis la sauvegarde top-k la plus récente », jamais depuis l'état
  final — champion 0.5959 inclus. Cohérence interne préservée (toutes les lignées ont
  subi le même mécanisme) ; l'écart n°4 du duel esjepa-vs-mix devient un écart de DEGRÉ
  (état d'époque 1 vs 35 % d'époque 0). Les vrais poids finaux de mix sont perdus comme
  ceux d'esjepa. v3 (save_top_k=-1) permettra pour la première fois l'ablation
  « finetune depuis l'état final vs best-val ». Indice manqué la veille : les probes
  `epoch01_valloss0.5550.json` et `last.json` étaient déjà identiques.
- **2026-08-26 (matin)** — **Pretrain ESJEPA COUPÉ à 1,5 ép.** (décision utilisateur,
  plateau sur toutes les observables : val, target/pred_var, z_corr 0.71, rang recollé à
  mix — précédent E13b, LR déjà à ~15 % du pic, budget). Au passage, **piège de
  checkpointing n°2 découvert** (§5) : `last.ckpt` du pod = copie du dernier top-k, pas
  l'état courant ⇒ poids tardifs perdus, le finetune part du best-val
  `esjepa_pretrain_bestval0.5841_35pct.ckpt` (~35 % du schedule). Le duel esjepa-vs-mix
  porte désormais QUATRE écarts déclarés : voie z, ration au pretrain, durée 1,5 ép.,
  état de départ peu recuit — l'attribution reste par les témoins (z_corr/gate), et le
  finetune 1 ép. cosinus recuit lui-même. Écart val expliqué et consigné : ~1/3 = taxe
  λ_z (composante loggée brute, contribution 0.009), ~2/3 = pénalité de variance (la
  voie z compacte un peu la géométrie — le témoin invariance est passé de « meilleur »
  en début de run à « égal/limite 2 % pire » ; à re-vérifier chiffres en main si le
  finetune déçoit ⇒ ablation λ_z=0.03 au budget près). Série probe-juge close :
  0.228 → 0.205(15 %, champion juge) → 0.239 → 0.248 → 0.264 → 0.243(35 % = best-val).
- **2026-08-25 (nuit)** — **E20c TRANCHÉ (3 runs concordants) : E est un CLASSEUR, pas un
  RÉGULARISATEUR.** T2b équitable (planner à beta faussé) : terme dormant = indiscernable
  (l'énergie de l'imagination ne voit pas l'erreur de modèle, par construction) ; terme
  forcé actif (e_ref 0.70) = PIRE (viol 78.7→82.4 % — il importe le prior de politique
  comportementale). Conséquences G13 gravées en E20c : prudence par FAN+z, continuité
  explicite dans le coût, sélection de juge = axe propre. Convergence mesurée avec le
  choix MPUR (variance, pas énergie).
- **2026-08-25 (nuit, verdict)** — **G4.2 MESURÉ : NEUTRE, et la raison est le résultat.**
  MASE 0.8702 au bit près (invariance vérifiée) ; CRPS 0.5961 vs 0.5959 (+0,03 %, bruit) —
  prédiction (−1 à −2,5 %) RÉFUTÉE. Cause lisible dans γ ≈ [1.01…1.12…1.03] : le fan est
  DÉJÀ conformement calibré sur le corpus d'entraînement (LOS_LOOP couv 0.11/0.91) — la
  sous-couverture 42-72 % de GIFT n'est pas un biais de tête, c'est du DISTRIBUTION SHIFT
  (sur-confiance hors distribution), invisible par construction pour un facteur uniforme
  appris in-distribution. La version par domaine/fréquence frôle la ligne rouge
  multi-config : NON PRISE. Conséquences : (1) G4.2 ARCHIVÉ comme ablation papier propre
  (« tête calibrée in-distribution ; la sous-couverture zero-shot est du shift ») ;
  (2) le mécanisme légitime d'étalement adaptatif est ESJEPA (gate z conditionnel à la
  fenêtre, un checkpoint, zéro adaptation par config) — G4.2-nul RENFORCE sa motivation.
  Trois bugs de plomberie attrapés sur le pod en route (prepare_data hors Lightning,
  batchs [B,L,1], état d'accumulateurs) — la boucle de collecte est désormais une
  fonction pure testée en intégration (remarque utilisateur légitime : tester les maths
  sans tester la boucle, c'est tester la moitié qui ne casse jamais).
- **2026-08-25 (nuit, suite)** — **G4.2 lancé en statut ABLATION PAPIER** (décision
  utilisateur : légitime côté « éval à l'aveugle » — gamma calibré corpus de finetune,
  jamais GIFT, un vecteur pour 97 configs — mais pas nécessairement le chiffre officiel
  communiqué). Livré : `scripts/calibrate_quantiles.py` (split conformal CQR
  multiplicatif, médiane inter-datasets, calibrer sous ×flip), flag
  `+quantile_gamma=<json>` dans evaluate_gift (cache isolé `_gamma-<tag>`, MASE
  invariante par construction), `tests/test_quantile_gamma.py`. Prédiction gravée :
  MASE bit-identique ; CRPS −1 à −2,5 % (0.5959 → ~0.581-0.590 ×flip) ; si dégradé ⇒
  miscalibrage non transférable, gamma archivé. Contexte budget consigné : fin des
  vacances ~28/08, pod ~500 €/mois ⇒ triage — v3 = seul run obligatoire, recon-mix G6.2
  DIFFÉRÉ (sert le papier, pas le leaderboard), pod stoppé entre les runs, couches
  CPU-gratuites (G4.2, E18f-sur-GIFT, pondération) en soirées post-rentrée.
- **2026-08-25 (soir)** — **G13-T1/T2 : premier test EBM-contrôle, protocole livré**
  (`scripts/control_ebm_probe.py`, CPU, zéro GPU — plomberie vérifiée sur modèle jouet,
  AUC ≈ 0.5 sur poids aléatoires comme attendu). Principe : simulateur thermostat POSSÉDÉ
  (linéaire, action cachée, historique sous politique bang-bang = confounding LOTSA en
  miniature) ⇒ chaque verdict du juge et chaque plan sont vérifiables contre la vraie
  dynamique. T1 = le juge sépare-t-il les futurs dynamiquement cohérents des violations
  (renversement temporel, saut d'état, réponse d'action inversée) ; T2 = planning par
  backprop de la COMMANDE u à travers le simulateur différentiable, coût seul vs
  coût+énergie (régularisateur MPUR), verdict par 200 rollouts bruités (gap d'optimisme +
  taux de violation). Réutilise la machinerie E18f d'`evaluate_energy.py` (variable
  optimisée : u, pas y). Checkpoint PRÉ-SPÉCIFIÉ : `timejepa_tiny_lotsa_mix/last.ckpt`
  (lignée E18b, choisi avant toute mesure) ; `esjepa15_bestjudge` en colonne secondaire
  déclarée « borne haute ». Prédictions AVANT run : P-T1 AUC > 0.7 sur renversement et
  saut (caveat : beta_flip peut être séparé par simple morphologie hors-gamme, pas par
  connaissance de l'action) ; P-T2 le terme d'énergie RÉDUIT le gap d'optimisme
  (sinon : le juge n'est pas encore un régularisateur de planning utilisable, négatif
  consigné).
- **2026-08-25 (après-midi, suite)** — ajout de **E20b** : statistique appariée du signal
  d'horizon sur les arms G6 existants (zéro GPU, `scripts/horizon_stats.py`). Résultat qui
  REFORMULE le pari fondateur : intra-fenêtre l'avantage JEPA DÉCROÎT avec la profondeur
  (pente −1.64 %/100 pas, IC95 % excluant 0) ; le gap ne s'ouvre qu'à travers les ROLLOUTS
  (Spearman +0.404 p=0.019, fragile sans etth1 p=0.069). Nouvelle hypothèse : le latent se
  dégrade moins sous itération — testable par sauts aux frontières 256/512, et corollaire
  S3 : h512 natif devrait RÉDUIRE l'avantage. Protocole G6.2 enregistré (recon-mix ~1,5 j
  GPU après ESJEPA, prédictions P-G6.2a-c gravées ; 3 graines seulement si réplication).
- **2026-08-25 (après-midi)** — **Pretrain ESJEPA en cours, P1 VÉRIFIÉE à mi-run** :
  `esjepa/z_corr = 0.63` (prédiction gravée : > 0.3 ; kill-switch ≈ 0 écarté) et
  `z_pred_std_ratio = 0.7` (pas de collapse marginal — et ratio MEILLEUR que la voie
  signal, pred_var/target_var ≈ 0.3-0.4 : la volatilité est plus prévisible que sa
  réalisation). `train_loss/z` 0.65→0.2 ; courbes sans inflexions de mi-époque
  (signature du rationnement, désormais aussi au pretrain — écart n°3 déclaré dans la
  config, règle d'attribution témoins-z gravée). Le run continue vers le finetune.
  **Suivi (soir, superposition wandb vs contrôle mix)** : le témoin de siphonnage λ_z
  est INVERSÉ — `val_loss/invariance` ESJEPA ~4-5 % MEILLEURE que mix à étape égale
  (repli λ_z=0.03 écarté ; la voie z régularise plutôt qu'elle ne vole).
  `collapse/effective_rank` glisse en revanche SOUS le contrôle (25 vs 33 à ~270k ;
  mix rebondit vers 400k et oscille 30-39 ensuite) sans se payer dans les objectifs —
  point de contrôle posé : inflexion du rang attendue vers 400-450k, sinon divergence
  à caractériser (action seulement si mse/mae décroche aussi). Série probe-juge
  (rang agrégé, standalone) : 0.228@10 % → **0.205@15 % (pic, copié
  champions/pretrain/esjepa15_bestjudge)** → 0.239 → 0.248 → 0.264 → 0.243@35 % —
  le juge et le forecaster ont des optima temporels différents ; et le probe du
  CHAMPION finetune (0.291 vs ~0.21 pretrain) réplique E18b : le finetune dégrade le
  juge ⇒ G12 utilisera un checkpoint-juge dédié, sélectionné par probe.
- **2026-08-25 (midi)** — ajout de **E20**, verdict final du run ration : fin de run évaluée
  (3 derniers checkpoints ≈ 0.6306-0.6310 nu / 0.6010-0.6016 flip, bassin plat), le champion
  du run RESTE le 45 % (0.8702/0.5959 ×flip). **`ration_oversample` ENTRE AU PROTOCOLE
  finetune** (critère battu sur la procédure officielle, mécanisme G10.2 compris, loss
  enfin propre). G7.3c re-confirmée dans sa forme la plus pure : val_loss quasi monotone
  ET meilleure-val = dernier = mauvais checkpoint GIFT. Prédiction soupe gravée avant
  mesure (~0.625-0.628 nu, ne bat pas le 45 %) puis MESURÉE : 0.6383 nu / 0.6072 flip —
  confirmée et amplifiée, la moyenne SORT du bassin. **SWA clos, négatif** (addendum E20) ;
  S1 intégralement vidée hors ESJEPA → prochain GPU : pretrain `lotsa_tiny_esjepa` sur le
  corpus mix EXISTANT (une variable ; v3 vient après, en S2).
- **2026-08-25** — **G13 posé au PLAN** : world model unifié forecast+contrôle (la promesse
  d'origine du nom TimeJEPA), conçu en discussion + note visuelle (artifact « Un arm, deux
  métiers »). Contenu : prédicteur conditionné par l'action via `a_film` (w_film
  vectorisé, zéro-init ⇒ `a=∅` = identité = le forecaster actuel, GIFT inchangé) ;
  contrôle = planning by backprop (E18f, variable optimisée = l'action) ou propose-juge
  (G12) ; planification prudente via fan + z ESJEPA. Mur identifié : pas d'actions dans
  LOTSA + confondage observationnel ⇒ sortie = synthétique CAUSAL (CauKer/TCM sur le
  pipeline v3). Clarifications doctrinales consignées en G13 (le prédicteur simule, ne
  décide pas ; action ≠ retouche de forecast ; a inconnu jamais bloquant — marginal vs
  acteur). Long terme, post-roadmap ; l'utilisateur pondère.
- **2026-08-25 (nuit)** — **NOUVEAU CHAMPION : ration@45 % × flip = 0.8702/0.5959 (~91e
  CRPS, Migas doublé ; ~99e MASE)** — bat champion×flip (0.8735/0.5984) sur les deux
  métriques. Et la prédiction G10.2 « pic plus tardif » SE RÉALISE : trajectoire nu du run
  ration 0.6345(25 %) → 0.6305 → 0.6329 → 0.6303 → **0.6239(45 %)** — amélioration tardive
  continue là où 1ep3e4 s'érodait dès 25 %. Le cosinus a encore 55 % d'époque sur
  composition constante. Checkpoint copié champions/. Verdicts finaux (rationnement au
  protocole ? soup ?) à la fin du run.
- **2026-08-24 (soir)** — **ROADMAP SOTA 4 semaines adoptée** (PLAN §0bis) après triple
  rapport : audit intégral du registre (trous relevés : per-config du champion jamais fait =
  E19 manquant ; 3 leviers positifs jamais exploités — raffinement E18f/h, hybride E18e/g,
  λ-ancrage E18b ; G9.1/G11/G4.2 jamais courus ; §10 périmé) × recherche des recettes
  concurrentes (ablations sourcées : synthétique majoritaire Toto-2 57.5 %/0 % public ;
  augmentations = plus gros poste TiRex +0.019 ; horizon un-forward +0.013-0.026 ; TTA
  YingLong −10.5 % MASE ; la TAILLE n'est pas le levier) × inventaire local (champion +
  data/gift_eval + per-config concurrents = décomposition 100 % offline). Arbitrages
  utilisateur : TTA uniforme OUI / G11 NON ; corpus v3 bundle complet avec principe « le
  batch cible d'abord, le corpus ensuite » ; mini gelé jusqu'au verdict v3 ; G12+calibration
  S3. Jalons : 0.605 (+8 places) / ~0.57 (TTM déshabillé) / 0.52-0.54 (étirement classe
  Toto-4m). Corrections d'audit : G8.1 clos « levier vide », doctrine G7.2 périmée barrée.
  Première action lancée : éval 97-configs du champion en local CPU → E19.
- **2026-08-24 (après-midi)** — **G10.2 MESURÉ et corrigé (opt-in).** L'audit
  (`audit_batch_schedule.py`, corpus mix finetune, 71 familles) chiffre la dérive de
  composition intra-époque : 16 familles éteintes avant 1 % de l'époque, ~53 avant la fin ;
  part des familles plafonnées 46.6 % → 35.7 % du batch entre premier et dernier décile ;
  batch 493 → 409 (−17 %, slots non réalloués) ; wind_farms/alibaba/m5/subseasonal → 0 %.
  La fin d'époque sur-entraîne les grosses familles (buildings_900k, largest_*) — mécanisme
  cohérent avec la dérive GIFT post-25 % (G7.3c) et l'inflexion au même step des deux runs
  (extinctions déterministes). Nuances de lecture : (a) N'INVALIDE PAS les comparaisons
  passées — tous les runs (mini en cours compris) partagent le même schedule ; ça plafonne
  le rendement du dernier tiers d'époque, plafond désormais levable ; (b) les augmentations
  ne peuvent pas compenser (elles diversifient les échantillons PRÉSENTS, pas les familles
  ABSENTES — et au finetune mix il ne reste qu'un jitter 1 % p=0.1, random_scale étant
  inerte sous arcsinh, T5). Correctif livré : `data.ration_oversample: true` — même budget
  3× par famille, étalé uniformément (quota fractionnaire par batch) ; opt-in strict,
  5 tests (dont itération bit-identique flag off et budget préservé), 295 verts au total.
  Ablation à courir sur tiny 1ep (une variable vs 1ep3e4, même seed) avant d'en faire le
  défaut. **SWA : verdict NON rendu** — les 3 checkpoints survivants du top-k val sont tous
  de fin d'époque (0.5855/64/65 → soup 0.9084/0.6312 ≈ le point tardif 0.6309) : le soup de
  la fenêtre des gains exige de garder tous les checkpoints → prochains finetunes avec
  `checkpoint.save_top_k=-1`.
- **2026-08-24 (midi)** — **Fin du run mix 1ep-3e-4 : verdict G7.3c rendu.** Fin d'époque
  (best-val 0.5855 @ ~95 % ; `last.ckpt` = le MÊME état, 97 configs bit-identiques — `last`
  pointe sur la dernière sauvegarde top-k) : **0.9098/0.6309**. La prédiction « pic en fin
  d'époque » est RÉFUTÉE : pic encore à 25 % (0.8914/0.6134) puis dérive — mais ~2× plus
  faible que v2 (courbes val_wql : v2 s'envole après ~470k, 1ep dérive mollement ; champion
  1ep 0.6134 < champion v2 0.6190). Enfoncement du clou sélection : le checkpoint à
  meilleure val_loss ET meilleur sMAPE fait 0.6309 vs 0.6134 au champion — AUCUNE métrique
  val ne sélectionne le champion GIFT. **Doctrine finale actée : la marche dans le plateau
  est intrinsèque à la loss ; protocole de production = finetune 1 époque cosinus (dérive
  amortie) + évals GIFT intermédiaires (~toutes les 5-10 %) + sélection par éval + cp
  champions/ immédiat.** Le schedule 1ep reste supérieur à 3ep (champion meilleur, dérive
  moitié moindre, coût /3) — c'est le protocole que mini et xres copient. **Mini-mix : GO**
  (configs d627509, smoke puis pretrain).
- **2026-08-24** — **Champion absolu : mix 1ep-3e-4 @ 25 %** (`epoch00_valloss0.5879` du
  dossier `_1ep3e4`, copié `champions/mix1ep3e4_25pct_mase0.8914_crps0.6134.ckpt`) :
  **GIFT MASE 0.8914 / CRPS 0.6134** — bat le champion v2 (0.8955/0.6190) au même point de
  la même trajectoire, seule variable le schedule (cosinus 1 époque, LR ~6 % plus froid à ce
  step), aucun flare. Rangs snapshot : 99e CRPS (paquet dense — ≤0.6121 pour TimeTron-33M,
  ≤0.6089 pour doubler YingLong_6m+Moirai_base), ~104e MASE (iTransformer doublé).
  **Nixtla (zero-shot authentique), même checkpoint — première table complète consignée** :
  MASE moyenne par dataset (h ∈ {96,192,336,720}) : electricity 1.025, etth1 1.176,
  etth2 1.560, ettm1 1.047, ettm2 1.164, traffic 0.769, weather 1.029 — moyenne ~1.110,
  **TimeJEPA meilleur modèle sur 7/7** vs SeasonalNaive/NaiveLast/ContextMean. Skill vs SN :
  traffic +41 %, electricity +22 %, etth1 +21 % ; poches négatives restantes : etth2 h192
  (−11 %), ettm2 h720 (−6 %), ettm1 h336/720 (~−1-2 %) — les longs horizons par rollout
  restent la faiblesse Nixtla. Mémo de comparaison (chiffres de conversation, non consignés
  à l'époque) : v2@5 % donnait ettm1 ~0.996 et moyenne ~1.098 — le champion GIFT n'est pas
  uniformément meilleur sur Nixtla (pondérations et horizons différents), les deux suites ne
  bougent pas ensemble. Attente posée : dégradation post-25 % plus faible que v2 puis
  reprise en fin d'époque quand le cosinus gèle la marche (prédiction G7.3c).
- **2026-08-23 (nuit)** — **ESJEPA implémenté** (G8.6, arm `model.error_signal`) : voie z =
  statistiques déterministes du résidu EWMA causal par patch cible [B,31,4], tête z sur le
  tronc du prédicteur (~8.5k params, `predictor.z_head.*` core P3.2, survit au finetune),
  gate d'étalement zéro-init sur les largeurs de `_make_monotone` (médiane intouchable ⇒
  MASE invariant, Δ WQL attribuable). λ_z=0.1, témoins `esjepa/z_corr` (kill-switch P1 > 0.3),
  `z_pred_std_ratio`, `gate_absmean`. Bras « unexpected » du refus P3.2 généralisé aux
  préfixes core (durcissement, couvre aussi w_film). Trio `lotsa_tiny_esjepa*` (base mix ⇒
  contrôle = pretrain mix existant, une variable ; finetune 1 époque G7.3c). 23 tests dédiés
  (dont rétrocompatibilité d'inférence des checkpoints pré-arm, demande utilisateur),
  290 verts, zéro régression flag-off. Prédictions P1-P3 gravées au PLAN. Run APRÈS le
  verdict mini-mix.
- **2026-08-23 (soir)** — **G8.4b résolu : enveloppe de prévision relative au contexte**
  dans `RobustScale.inverse` (clamp [min−10·w, max+10·w], w = max(étendue, échelle)).
  Déclencheur : CRPS 18 305 724 sur bitbrains_fs/H/short au checkpoint 15 % de
  mix_zs_1ep3e4 (×1.19 sur la geomean à lui seul) — un z de queue ≈15 de la tête
  mi-entraînée, ré-amplifié par sinh ; le plancher d'échelle n'y pouvait rien, la garde
  est en aval. Monotone, inactif sur les prévisions raisonnables, précédent Chronos ±15σ.
  Test de régression, 267 verts. ⚠️ Toute éval arcsinh postérieure à ce commit n'est
  comparable qu'aux évals du même code — ré-évaluer champion + final avant les chiffres
  E19. Configs mini-mix créées le même jour (G7.4 : pretrain recette mix à 5M, finetune
  protocole 1 époque, eval robust_scale porté).
- **2026-08-23** — ajout de **E18i** : le recuit court depuis 0.5874 tranche le diagnostic
  du finetune mix v2 — **instabilité de LR, pas overfit** (à froid : val ↓ ET GIFT ↑,
  0.6357 → 0.6272 à 5 % puis bande 0.627-0.632 ; train plate sur tout le run chaud).
  Mini débloqué ; champion 0.6190 requalifié « bassin + loterie de marche » ; pattern
  gains-à-la-rampe répliqué à LR 6x plus bas. Doctrine : finetune court recuit. Run
  protocole E19 lancé : 1 époque fraîche, tête 8e-5 / backbone x0.1, cosinus dans
  l'époque (`mix_zs_1ep`), prédiction 0.615-0.63 avec pic en fin d'époque. Deux flares
  G8.4b ont pollué des agrégats du recuit (bitbrains_fs/5T/short 2.812, car_parts 1.276)
  → plancher relatif promu « avant chiffres finaux E19 ». Habitude actée :
  `cp checkpoints/champions/` immédiat pour tout checkpoint couronné.
- **2026-08-12** — création. Couvre E0 à E7, la thèse, les décisions de conception, le
  positionnement et la carte affirmations → preuves.
- **2026-08-12 (soir)** — ajout de **E8**, le baseline sans pretraining : égalité (1.187 vs
  1.193 de MASE moyenne). §4 et la carte affirmations → preuves mis à jour en conséquence.
- **2026-08-12 (soir, suite)** — ajout de **E9**, balayage de contexte sur le scratch : même
  optimum et même platitude que l'arm pré-entraîné, donc la robustesse en longueur vient du
  finetune et non du pretraining. Affirmation 7 du §3 précisée.
  Restent à mesurer : l'arm `tiny_geo` en full finetune (mode identique à E8), et **G4.6**.
- **2026-08-12 (nuit)** — ajout de **E10**, composition réelle du corpus de pretrain lue dans
  le log : 6 datasets écartés au chargement (le corpus held-out de G4.6), et surtout deux
  datasets haute fréquence qui pèsent 48,7 % du batch. §4 complété d'une hypothèse
  explicative non testée pour E8.
- **2026-08-12 (nuit, suite)** — ajout de **E11**, G4.6 : **le pretraining transfère**, −26 %
  de MASE moyenne et 8/8 datasets en régime données inédites + peu de données.
- **2026-08-22** — Correctif RobustScale (plancher d'échelle conditionnel, `4b772c2`) après
  CRPS 10^10..inf mesurés sur le finetune mix à 5-10 % : 29 % des contextes bitbrains_rnd ont
  MAD exactement 0, l'échelle plancher 1e-8 créait un repère décalé de +18 que sinh explosait
  à l'inverse — structurel, pas transitoire (la pathologie G6 un étage plus haut). Décision
  utilisateur : couper et relancer le finetune avec le fix ; pretrain non affecté (LayerNorm
  borne les cibles latentes).
  **Note intermédiaire E19 (astérisque protocole)** : checkpoint mix à 15 % d'époque 1,
  entraîné AVEC le bug, évalué APRÈS le fix (le scaler est sans poids, le pull a changé la
  transformation d'éval — chimère train-ancien/éval-nouveau) : **MASE 0.9235 / CRPS 0.6453**.
  Même sous astérisque : (a) les explosions disparaissent sur un checkpoint entraîné avec le
  bug — le diagnostic « amplification à l'inverse » est confirmé ; (b) la recette mix bat le
  final de tiny-full (0.9685/0.6664) dès 15 % d'une époque sur trois — **la gate P3 v0.1
  (CRPS < 0.65) tombe pour la première fois**, Moirai_small (0.650) dépassé ; (c) trajectoire
  5 %→10 %→15 % : 0.9964 → 0.9679 → 0.9235, descente encore raide. Le verdict E19 officiel
  se prendra sur le run RELANCÉ propre (fix de bout en bout), ces chiffres servant de borne
  inférieure attendue. Rangs EXACTS (snapshot local du leaderboard, 125 modèles classés,
  `docs/assets/gift_leaderboard/2026-08-22/`, formule officielle sur les CSV officiels,
  script `fetch_gift_leaderboard.py`) : E16 108e CRPS/110e MASE ; E18 108e/110e — le gain
  −1,6 % de CRPS ne franchissait AUCUN barreau, le classement bouge par paliers ; mix-15 %
  **103e CRPS / 108e MASE**. Prochain palier : un paquet dense à CRPS 0.609-0.627
  (Reverso-Small, litespecformer, Lingjiang, iTransformer, TimeTron-33M, Reverso,
  Moirai_base, YingLong_6m) — atteindre ~0.605 = +8 places d'un coup, l'objectif chiffré
  naturel du run propre.
- **2026-08-21 (soir)** — ajout de **E18b**, la sonde d'énergie : le latent du pretrain
  classe le vrai futur au rang 0.245 (hasard 0.50) parmi 34 candidats bootstrap, ρ(E,MAE)
  0.5-0.74 partout, solar compris — la lecture « proposer-juger-pondérer » est légitimée.
  Et le full finetune DÉTRUIT cet alignement (0.409, ρ erratique, sz_taxi sous le hasard) :
  premier coût mesuré du drift. `scripts/probe_energy.py`, CPU, pendant que mix pré-entraîne.
- **2026-08-21** — ajout de **E18** : le point corpus-complet. Corpus ×6 ⇒ CRPS 0.677 → 0.6664
  (−1,6 %), MASE 0.979 → 0.9685 — réel et modeste, le levier données seul s'aplatit. La trace
  par checkpoint (8 évals GIFT du même finetune) montre tout le progrès sous LR ~3e-4 → plafond
  3e-4 acté pour mix/xres. Décomposition per-config : l'écart vit dans 16 configs sub-horaires
  (geomean 0.853 sans elles), pas dans l'objectif — prédiction falsifiable posée pour E19.
  G7.3b tranché : base mix/xres = lotsa_full.
- **2026-08-19 (run tiny-full, à ~40 % de l'époque)** — Observation à retenir pour la lecture
  de TOUTES les val_loss de pretrain JEPA : hausse soutenue de la val_loss (0,503 → 0,552 sur
  4 points consécutifs, ~135k → 270k steps) avec cosine en baisse, ALORS QUE la mémorisation
  est impossible (~0,85 passe sur le corpus, la hausse commence à ~0,45 passe) et que la
  représentation est INTACTE : effective_rank stable 42-49 (plage calibrée 43-87), context_std
  stable 0,85-0,875. Diagnostic : **artefact de cible mouvante** — `val_repr/target_var` monte
  de 0,75 à 0,79 en parallèle, l'encodeur cible EMA produit des cibles plus riches donc plus
  dures à prédire, et la loss monte sans que le modèle régresse. La val_loss JEPA n'est pas
  comparable dans le temps parce que la distribution des CIBLES évolue. Corollaire pratique :
  ne jamais diagnostiquer un pretrain JEPA sur sa val_loss seule — rank + std + target_var
  d'abord. (Recoupe E13b, dont la « divergence » après une époque contenait probablement une
  part de ce même artefact en plus du re-passage.) Vérification restante : la hausse doit
  s'infléchir quand le recuit mord (dernier tiers) — sinon réexaminer. Décision : à la fin du
  run, évaluer LES DEUX checkpoints (last + meilleur val ~135k) — l'expérience de sélection
  que E13 demandait.
- **2026-08-19** — G9.2 IMPLÉMENTÉ (pas encore couru) : arm inter-résolution — contexte@k1,
  cible@k2 contiguë, w=k2/k1 par item, FiLM zéro-init sur les requêtes du prédicteur
  (identité exacte à w=1), cibles standalone obligatoires. Corpus requis : morceaux 8192
  (synthétique + chronos_extras) — les 2048 n'autorisent que k1=1. Témoin de non-stérilité :
  `aug/w_neq1_frac` sur wandb. Comparaison à une variable : contre le run CONTRÔLE
  lotsa_tiny sur le même corpus mixte, jamais contre lotsa_tiny nu.
- **2026-08-18 (suite 2)** — ajout de **E17** : diagnostic compétitif sur 123 modèles. L'écart
  au leaderboard suit la COUVERTURE FRÉQUENTIELLE (facteur 2,2), pas l'horizon (plat) ni le
  multivarié (second ordre). Sur les fréquences couvertes, 1M bat un 4M SOTA.
- **2026-08-18 (suite)** — ajout de **E16** : premier positionnement GIFT-Eval (MASE ratio
  0.979, CRPS 0.677 à ~1M params, zero-shot, 97/97 configs) et réplication du signal
  d'horizon de G6 (écart JEPA-recon monotone : +0,8/+3,1/+6,6 % en short/medium/long).
- **2026-08-18** — ajout de **E15** : G6 tranche, et NÉGATIVEMENT. L'extrapolation latente ne
  se distingue pas de la reconstruction (1,4 % de MASE, recon gagnant 17/28 cellules). La
  thèse centrale sort du papier ; la contribution devient E14. Affirmation 14 ajoutée au §3,
  §4 mis à jour du verdict.
- **2026-08-15 (suite)** — ajout de **E14** : premier modèle zero-shot LOTSA. MASE moyenne
  1.150 contre 1.193 (geo) et 1.159 (p32), et surtout **ETTm1 de −37 % à −8,4 % de skill** —
  l'échec structurel du projet cède au corpus, pas à l'architecture. Affirmations 13 et 14
  ajoutées au §3.
- **2026-08-15** — ajout de **E13b** : le recuit révèle un surapprentissage net à partir
  d'une époque, concordant entre les deux runs. Affirmation 13 ajoutée au §3.
  `max_oversample_ratio` ramené à 3,0 et `lotsa_tiny_full` (max_epochs 1) créé pour le
  passage au corpus complet.
- **2026-08-14** — ajout de **E13a** : premier pretrain LOTSA interrompu, avec trois
  enseignements méthodologiques (calibration du rang effectif par corpus, `val_loss`
  polluée par son régularisateur, scheduler à caler sur le budget réel). Affirmations 10 à
  12 du §3 mises à jour. `max_epochs: 5` inscrit dans les configs LOTSA.
- **2026-08-13 (suite)** — ajout de **E13** : ingestion LOTSA et protocole zero-shot mis en
  place et validés de bout en bout (47 exclus / 123 retenus, conversion vérifiée), sans
  résultat d'entraînement. B22 ajouté au tableau du §5, §6 complété des configs LOTSA.
- **2026-08-13** — §5 : découverte de la **contamination du corpus par les benchmarks**
  (electricity-hourly et traffic-hourly SONT les benchmarks Nixtla correspondants), trouvée
  en construisant la liste d'exclusion de LOTSA. §4 complété : les chiffres traffic et
  electricity sont en attente d'un run sur corpus disjoint. Correction adoptée : protocole
  zero-shot, entraînement intégral sur LOTSA.
- **2026-08-12 (nuit, fin)** — ajout de **E12**, run de robustesse : le baseline généreux
  obtient une MEILLEURE val_loss de finetune et reste MOINS BON sur tous les benchmarks.
  L'ampleur tombe de −26 % à −8,7 %, mais la revendication se précise et devient défendable :
  gain de TRANSFERT, pas d'ajustement. Affirmations 9 et 10 réécrites au §3.
