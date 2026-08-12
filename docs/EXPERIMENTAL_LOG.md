# TimeJEPA — registre expérimental

**Objet.** Trace des expériences menées, de leurs chiffres mesurés, et de ce que chacune
établit. Écrit pour servir de matière première à un article : chaque affirmation y est
rattachée à une mesure, et ce qui n'a PAS été mesuré est signalé comme tel.

**Ce document n'est pas** le plan de travail (`PLAN.md`, orienté tâches) ni la description
de l'architecture (`docs/TECHNICAL_OVERVIEW.md`). Les justifications détaillées de chaque
changement de code vivent dans les messages de commit de la branche `sota-roadmap`.

**Statut au 2026-08-12 (soir).** Le baseline sans pretraining (E8) est tombé : **égalité**
avec l'arm pré-entraîné. Le pari central n'est pas réfuté pour autant — le régime testé ne
peut pas le réfuter (voir E8) — mais il n'est pas soutenu non plus. L'expérience discriminante
(régime pauvre en données) est la prochaine.

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
10. **La `val_loss` de finetune est un mauvais critère de sélection** pour un modèle de
   fondation : elle désigne le modèle qui généralise le moins bien (E12).

## 4. Ce qui n'est PAS établi

À ne pas revendiquer sans mesure supplémentaire.

- **L'explication du fait que traffic porte l'essentiel du gain de transfert** (−26 % contre
  −0,3 % à −5 % ailleurs, E12). Non élucidé.
- **La généralité du résultat E12** : une seule graine, un seul couple de datasets d'aval
  (dominé par m4-hourly), un seul modèle tiny. La direction est systématique sur 6 datasets
  d'évaluation, l'ampleur est modeste (−8,7 %).
- **Que l'extrapolation latente batte la reconstruction masquée** comme objectif de
  pretraining. Aucune expérience ne les compare ; c'est le pari central du projet et il
  reste non testé.
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
| **B21** | `auto_insert_metric_name` présent dans la config mais jamais transmis à `ModelCheckpoint` → noms de fichiers doublés contenant `=`, incompatibles avec la grammaire d'override Hydra | ergonomie ; a causé au moins un échec de commande |
| — | **Incident de configuration** : un finetune du round géométrie a tourné avec un décodeur ponctuel (défaut hérité `mlp`) sans qu'aucun signal ne l'indique. Détecté par le nombre de clés chargées (110 au lieu de 118) et l'absence du suffixe de couverture | une évaluation dont les colonnes « WQL » étaient en fait des ND ponctuels |
| — | **Incident de configuration** : un finetune a tourné sur la liste curatée de 8 datasets au lieu des 24, contredisant E3 | conservé comme arm d'ablation ft8 |

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
- **2026-08-12 (nuit, fin)** — ajout de **E12**, run de robustesse : le baseline généreux
  obtient une MEILLEURE val_loss de finetune et reste MOINS BON sur tous les benchmarks.
  L'ampleur tombe de −26 % à −8,7 %, mais la revendication se précise et devient défendable :
  gain de TRANSFERT, pas d'ajustement. Affirmations 9 et 10 réécrites au §3.
