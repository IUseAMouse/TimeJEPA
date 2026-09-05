# Runbook S4-a — Assemblage du corpus v4 (pod, ~1 h)

Corpus v4 = corpus v3 à l'identique, **sauf le bloc des séries courtes**, refait
avec le seuil abaissé et le sidecar de longueurs réelles. Une seule variable
par rapport au champion : le bloc court + les fenêtres à frontière au finetune.
Doctrine inchangée : NE RIEN SUPPRIMER — `lotsa_v3/` et `lotsa_short/` restent
en place, v4 vit dans ses propres répertoires. Chaque étape a son GATE.

Rappel de la structure v3 (runbook S2.4) : `lotsa_v3/` est un dossier de
SYMLINKS vers cinq sources — `lotsa_xres/` (LOTSA dense + mix), `synthetic_v3/`
(23 shards), `lotsa_short/` (12 subsets courts, `--min-length 384 --pad-to 1280`),
`lotsa_solar/`, `decimated/`. C'est ce `--min-length 384` qui excluait les séries
courtes, et le pad-to sans sidecar qui créait les fenêtres à cible-pad (défaut v3,
registre 2026-09-05).

---

## Étape 0 — État des lieux (5 min)

```bash
cd /workspace/TimeJEPA && git pull
ls data/processed/                       # attendu : lotsa_v3 lotsa_short lotsa_xres synthetic_v3 lotsa_solar decimated
ls -l data/processed/lotsa_v3 | head -3  # confirmer : liens symboliques
ls data/processed/lotsa_v3 | wc -l       # attendu : 106
python -m pytest tests/test_short_series.py tests/test_v3_data.py -q
```

**GATE 0** : tests verts, les cinq sources présentes, 106 entrées dans lotsa_v3.

## Étape 1 — Bloc court v4 (15 min, CPU)

Mêmes 12 subsets que le bloc v3, même chunk et même pad ; seul le seuil change
(384 → 24). Le sidecar `_reallen/` est écrit automatiquement dès que `--pad-to`
est donné.

```bash
python scripts/prepare_lotsa.py --out data/processed/lotsa_short_v4 \
  --subsets m1_monthly m1_quarterly m1_yearly monash_m3_monthly \
  monash_m3_other monash_m3_quarterly monash_m3_yearly tourism_monthly \
  tourism_quarterly tourism_yearly nn5_daily_with_missing nn5_weekly \
  --min-length 20 --chunk-length 1280 --pad-to 1280
ls data/processed/lotsa_short_v4 data/processed/lotsa_short_v4/_reallen
```

Pourquoi 20 : 16 pas de contexte réel + 4 de cible réelle, les minima des
fenêtres à frontière — une série de 20 pas donne exactement une fenêtre ; en
dessous, rien n'est apprenable. (Premier passage du 2026-09-05 à 24 : monash_m3_yearly
rejeté en bloc, 78 tourism_yearly perdues.) Pourquoi pad 1280 et pas 2048 : c'est
la géométrie du finetune mini (1024 + 256) ; le contexte long (S4-b) refera sa
propre prép. Avec `--pad-to`, la longueur de chunk N'EST PLUS adaptée à la
médiane (correctif du 2026-09-05) : chaque série ≥ 20 est gardée ENTIÈRE puis
paddée — le premier passage tronquait les séries plus longues que la médiane à
leurs premiers pas (m3_quarterly coupé à 44 sur 24-72). Si un premier
`lotsa_short_v4` existe : `mv data/processed/lotsa_short_v4
data/processed/lotsa_short_v4_trunc` (jamais de suppression), puis relancer.

**GATE 1 (anti-contamination, BLOQUANT)** : le log affiche les exclusions G8.1
(m4_*, hospital, car_parts, covid restent DEHORS — jeux GIFT) ; aucun subset
hors des 12 listés ; `_reallen/` contient exactement un `.npy` par fichier
produit. Noter le nombre de séries par subset dans le log : les yearly (m1, m3,
tourism) doivent maintenant ÊTRE PRÉSENTS — en v3 ils étaient rejetés en bloc
(« series too short ») — et « N chunks » doit égaler « N series » moins les
« too short » (aucune ligne « chunk length adapted », aucun « LOST »). Si un
yearly manque encore, STOP.

## Étape 2 — Assemblage par symlinks (5 min)

```bash
mkdir -p data/processed/lotsa_v4
cd data/processed/lotsa_v4
ln -s ../lotsa_xres/*.npy .
ln -s ../synthetic_v3/*.npy .
ln -s ../lotsa_short_v4/*.npy .          # remplace lotsa_short
ln -s ../lotsa_solar/*.npy .
ln -s ../decimated/*.npy .
ln -s ../lotsa_short_v4/_reallen _reallen   # le sidecar, résolu par le dataset via <dossier>/_reallen/<fichier>
cd /workspace/TimeJEPA
ls data/processed/lotsa_v4 | wc -l        # attendu : 106 fichiers + _reallen = 107
diff <(ls data/processed/lotsa_v3) <(ls data/processed/lotsa_v4 | grep -v _reallen)
```

**GATE 2** : `ln` ne râle sur aucun nom ; le `diff` est VIDE (mêmes 106 noms de
familles — v4 ne change que le CONTENU du bloc court, pas la liste). Le dataset
cherche le sidecar dans `<dossier du .npy>/_reallen/<nom>.npy` : les fichiers
de `lotsa_xres`, `synthetic_v3`, `decimated`, `lotsa_solar` n'en ont pas et sont
traités comme pleins — comportement v3 bit-identique pour eux.

## Étape 3 — Audit de composition (10 min, CPU) — la question « surveiller le batch »

```bash
python scripts/audit_batch_schedule.py --config-name lotsa_mini_v4_zeroshot --mode finetune \
  2>&1 | tee evaluation/audit_v4_finetune.txt
python scripts/audit_batch_schedule.py --config-name lotsa_mini_v3_zeroshot --mode finetune \
  2>&1 | tee evaluation/audit_v3_finetune.txt      # référence appariée
```

L'audit reproduit la construction de train.py (flag v4 compris) : il compte les
fenêtres à frontière des lignes courtes.

**GATE 3** : (a) les 12 familles du bloc court apparaissent avec une part de
batch > 0 et aucune ne dépasse quelques % (elles ont peu de fenêtres — c'est le
cap/rationnement qui décide) ; (b) la part des autres familles bouge de moins
d'un point vs la référence v3 ; (c) composition stable à travers les déciles.
Si une famille courte est à 0 : le sidecar n'est pas résolu (vérifier le lien
`_reallen`). Consigner les deux audits au registre.

## Étape 4 — Lancement (une variable vs le champion)

Finetune depuis le pretrain mini v3 val-best. La config hérite de la tête x8
(recette par défaut depuis le verdict G14 du 2026-09-05) : pretrain, architecture
et recette identiques au champion head8 25 % (0.7914/0.5433, oracle 0.5190) ; la
seule variable est le corpus v4 + les fenêtres à frontière.

```bash
python scripts/train.py --config-name lotsa_mini_v4_zeroshot \
  '+training.pretrained_encoder_path="checkpoints/timejepa_lotsa_mini_v3/pretrain_True/epoch00_valloss0.5495.ckpt"'
```

**Témoin au premier décile (BLOQUANT)** : `aug/short_frac` > 0 sur wandb. À 0,
le sidecar n'est pas lu et le bras est stérile — couper, revenir au GATE 2.

## Étape 5 — Évaluation (namespace propre)

```bash
python scripts/evaluate_gift.py --config-name lotsa_mini_v4_eval \
  +checkpoint_path=checkpoints/timejepa_lotsa_mini_v4_zs/pretrain_False/<ckpt> +tta_flip=true +ratein=backtest
```

Toujours les trois lectures (nu, flip, flip+ratein) sur le checkpoint retenu.
Points de comparaison appariés (head8) : 15 % 0.7974/0.5466, 25 % 0.7914/0.5433 ;
oracle 25 % 0.7700/0.5190 (diagnostic).

## Prédictions P-v4 (à graver au registre AU LANCEMENT, avant le premier eval)

- P-v4.1 (mécanisme) : les configs à historique court (m4_yearly, m4_quarterly,
  m4_monthly, m4_weekly, hospital, car_parts, covid) baissent en MASE au
  checkpoint apparié — le levier agit par TRANSFERT (m1/m3/tourism appris,
  m4/hospital jamais vus), donc bande large.
- P-v4.2 (innocuité) : les configs à long historique stables à ±1 % de CRPS.
- P-v4.3 (agrégat) : MASE < 0.78 au 25 % ; CRPS flip+ratein ≤ 0.5433.
  ÉCHEC-DIAGNOSTIC : MASE ≥ 0.7914 ⇒ le régime court ne se transfère pas de
  m1/m3 vers m4 — le levier MASE est ailleurs (contexte, décodeur), pas dans le
  corpus.
