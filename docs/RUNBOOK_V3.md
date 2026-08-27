# Runbook S2.4 — Assemblage du corpus v3 (pod, ~une demi-journée)

Exécution de la recette gravée en tête de `configs/model/lotsa_tiny_v3.yaml`.
Bundle béni 2026-08-24 ; prédictions P-v3.1..4 au registre (E19). Doctrine :
NE RIEN SUPPRIMER — tout nouveau corpus vit dans son propre répertoire, les
sources restent intactes. Chaque étape a son GATE : on ne passe pas à la
suivante tant qu'il n'est pas vert.

Deux points de DÉCISION sont marqués ⚖️ — tout le reste est mécanique.

---

## Étape 0 — État des lieux (5 min)

```bash
cd /workspace/TimeJEPA && git pull
ls data/processed/                      # attendu : lotsa_xres (le corpus mix)
ls data/processed/lotsa_xres | grep synthetic   # noter les noms v1 en place (anti-doublon, étape 2)
df -h /workspace                        # ~30-60 Go libres nécessaires (~8-15 Go générés)
python -m pytest tests/test_v3_data.py tests/test_corpus.py -q
```

**GATE 0** : tests verts, `lotsa_xres/` présent, espace disque suffisant.

## Étape 1 — Audit de la composition ACTUELLE (15 min, CPU)

Le principe utilisateur : « le batch cible d'abord, le corpus ensuite ».
On mesure ce que le sampler sert AVANT d'ajouter quoi que ce soit.

```bash
python scripts/audit_batch_schedule.py --config-name lotsa_tiny_v3 --mode pretrain --ration \
  2>&1 | tee evaluation/audit_v3_avant.txt
```

(La config v3 pointe déjà `data_dir: lotsa_v3` — si l'audit refuse car le
répertoire n'existe pas encore, auditer `lotsa_tiny_mix` à la place : c'est la
composition de départ, même information.)

**GATE 1** : le rapport donne la part de batch par famille. Noter les trois
chiffres qui pilotent l'étape 2 : part synthétique actuelle, part 10S/10T-like,
part intermittente/courte.

## Étape 2 ⚖️ — DIMENSIONNEMENT du synthétique (décision à deux)

Cible (révisée par l'utilisateur 2026-08-27) : **50-55 % de part de BATCH synthétique** après
rationnement, réparti pour combler les trous per-config :

| famille | cible indicative | justification (E19) |
|---|---|---|
| synthetic_ops_bursty | LE gros morceau du delta | bizitobs 10S geomean 1.387, aucune donnée publique |
| synthetic_intermittent | modéré | car_parts 0.98, M/short épars |
| 3 familles v1 (subhourly/broadband/…) | maintien | la queue 10T/15T déjà comprimée (E18 vérifié E19) |

DIMENSIONNEMENT DÉCIDÉ (audit du 2026-08-27, cible utilisateur 50-55 %) :
le sampler T=0.5 pèse en √ PAR FICHIER ⇒ la part se pilote par le NOMBRE de
shards (précédent corpus : era5_*/cmip6_*/largest_*). Base mesurée : 3 familles
v1 = 11.2 % (3.74 % chacune, libres). 26 shards ≈ 51 % prédit.

```bash
OUT=data/processed/synthetic_v3
for s in $(seq 1 10);  do python scripts/generate_synthetic.py --set v3 --out $OUT \
  --families synthetic_ops_bursty   --suffix _s$s --seed $s; done
for s in $(seq 11 15); do python scripts/generate_synthetic.py --set v3 --out $OUT \
  --families synthetic_intermittent --suffix _s$s --seed $s; done
for s in 16 17 18; do python scripts/generate_synthetic.py --set v3 --out $OUT \
  --families synthetic_subhourly    --suffix _s$s --seed $s; done
for s in 19 20 21; do python scripts/generate_synthetic.py --set v3 --out $OUT \
  --families synthetic_broadband    --suffix _s$s --seed $s; done
for s in 22 23;    do python scripts/generate_synthetic.py --set v3 --out $OUT \
  --families synthetic_lowfreq      --suffix _s$s --seed $s; done
```

Allocation (priorités E19) : ops_bursty ×10 (~20 % de batch — bizitobs, trou
sans substitut) · intermittent ×5 · subhourly +3 · broadband +3 · lowfreq +2.
Seeds 1-23, dérivation interne seed×1000+i ⇒ aucune collision avec v2 (seed 0).
~19 Go, ~1h15. COÛT ASSUMÉ de la dilution : la queue plafonnée réelle passe de
42 % à ~24 % (era5/cmip6 halvés — sur-représentés, sain ; alibaba 1.3→0.65 %,
compensé par ops_bursty sur le même domaine). Composition prédite à vérifier
au GATE 6 : synth ~51 % / libres réelles ~25 % / queue ~24 %.

**GATE 2** : `ls data/processed/synthetic_v3 | wc -l` = 23 shards (5 familles
suffixées _s1.._s23) ; spot-check visuel d'un morceau ops_bursty (rafales,
zéros exacts, saturation).

## Étape 3 — Séries courtes réelles (pad-to) + solar_power (20 min)

```bash
python scripts/prepare_lotsa.py --out data/processed/lotsa_short \
  --subsets m1_monthly m1_quarterly m1_yearly monash_m3_monthly \
  monash_m3_other monash_m3_quarterly monash_m3_yearly tourism_monthly \
  tourism_quarterly tourism_yearly nn5_daily_with_missing nn5_weekly \
  --min-length 384 --chunk-length 1280 --pad-to 1280

python scripts/prepare_lotsa.py --out data/processed/lotsa_solar \
  --subsets solar_power
```

**GATE 3 (anti-contamination, BLOQUANT)** : les logs de conversion doivent
montrer les exclusions G8.1 actives ; `solar_power` passe par ses 3
vérifications (commentaire lotsa.py:175). Aucun subset hors des listes
ci-dessus ne doit apparaître. En cas de doute : STOP, on vérifie ensemble.

## Étape 4 — Décimation 5T → 10T/15T (15 min)

```bash
python scripts/decimate_corpus.py --src data/processed/lotsa_xres \
  --dst data/processed/decimated --factors 2,3
```

(Sans `--families` : le script prend les denses éligibles ; `--min-len 1280`
par défaut protège les courtes.)

**GATE 4** : comptage des .npy produits ; un spot-check — une série décimée ×2
doit avoir une période apparente doublée en pas.

## Étape 5 — Assemblage par symlinks (5 min)

```bash
mkdir -p data/processed/lotsa_v3
cd data/processed/lotsa_v3
ln -s ../lotsa_xres/*.npy .
ln -s ../synthetic_v3/*.npy .
ln -s ../lotsa_short/*.npy .
ln -s ../lotsa_solar/*.npy .
ln -s ../decimated/*.npy .
cd /workspace/TimeJEPA
ls data/processed/lotsa_v3 | wc -l
```

**GATE 5** : aucun conflit de nom (`ln` râle sinon — si collision, STOP :
c'est un doublon de dataset, pas un détail) ; le compte total est cohérent
avec la somme des sources.

## Étape 6 ⚖️ — Audit FINAL et consignation (15 min + décision)

```bash
python scripts/audit_batch_schedule.py --config-name lotsa_tiny_v3 --mode pretrain --ration \
  2>&1 | tee evaluation/audit_v3_apres.txt
```

**GATE 6 (le gate du bundle)** :
- part de batch synthétique dans [50 %, 55 %] (cible utilisateur 2026-08-27) —
  sinon AJOUTER/RETIRER des shards (le bouton à cran) et re-auditer ;
- ops_bursty visible à hauteur de son rôle (plusieurs % du batch) ;
- aucune famille < 1 % qui s'éteint (le rationnement doit la maintenir) ;
- composition stable à travers les déciles de l'époque.
→ CONSIGNER les deux audits (avant/après) au registre — c'est la pièce
d'identité du corpus v3 dans le papier.

## Étape 7 — Décision z (selon E21, règle gravée au registre)

- Issue (a) — CRPS < 0.5959 dans la fenêtre : ajouter `error_signal: true`
  (+ `lambda_z: 0.1`) à `lotsa_tiny_v3.yaml` avant lancement.
- Issues (b)/(c) : NE PAS ajouter z. Une variable de moins dans le bundle.
Si le verdict E21 n'est pas tombé au moment du lancement : lancer SANS z
(défaut conservateur — z restera testable en ablation sur v3).

## Étape 8 — Lancement

```bash
python scripts/train.py --config-name lotsa_tiny_v3 wandb.run_name=v3-pretrain
```

Garde-fous déjà dans la config : ration ON, augmentations TiRex ON,
save_top_k=-1 (TOUS les checkpoints — plus jamais le piège last.ckpt),
early stopping OFF. Budget : 2 époques, ~2 jours. Le run n'a pas besoin de
toi une fois lancé — évals et finetune en soirées post-rentrée.

Pendant le run, la routine habituelle : witnesses wandb (`aug/*` actifs,
collapse/*, composition), probes energy optionnels sur checkpoints (la série
juge v3), snapshot inutile (tout est sauvegardé).

## Prédictions déjà gravées (E19 — rappel, rien à réécrire)

P-v3.1 bizitobs 10S < 1.0 · P-v3.2 bloc A/Q/M/W −10 % · P-v3.3
electricity/15T/long < 1.0 · P-v3.4 agrégat ≤ 0.59 avant couches (flip en sus).
