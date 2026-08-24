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
