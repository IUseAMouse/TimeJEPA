# Runbook TimeJEPA

Commandes pour mettre en place l'environnement, reconstruire le corpus, entraîner,
évaluer et publier. Toutes les commandes se lancent depuis la racine du dépôt.

## Environnement

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_PROJECT_ENVIRONMENT=~/venvs/timejepa   # un venv hors de tout dossier synchronisé
uv sync --all-extras
uv run pytest -q tests                            # environ 3 minutes sur CPU
```

Sur une machine louée, épingler l'index des roues torch à la version CUDA du pilote avant
`uv sync` ; un pilote trop ancien fait échouer torch à l'initialisation.

## Données

```bash
make gift-download          # les 97 configurations GIFT-Eval dans data/gift_eval (quelques Go)
scripts/build_corpus_v3.sh  # le corpus d'entraînement v3 : 106 fichiers, 15.85 milliards d'observations
scripts/build_corpus_v3.sh --check   # l'audit seul
```

Le script enchaîne les sous-ensembles LOTSA et Chronos aux révisions Hugging Face
épinglées, les familles synthétiques, les séries courtes, la décimation, puis assemble
`data/processed/lotsa_v3` par liens symboliques et vérifie le résultat contre la
référence. Il exige `configs/corpus_v3_manifest.txt` (nom de fichier vers dossier
source), généré sur la machine de référence :

```bash
for f in data/processed/lotsa_v3/*.npy; do
  echo "$(basename "$f" .npy) $(basename "$(dirname "$(readlink -f "$f")")")"
done | sort > configs/corpus_v3_manifest.txt
```

Les fréquences d'échantillonnage par fichier sont dans
`configs/corpus_v3_frequencies.yaml` (`scripts/build_frequency_table.py` pour la
regénérer ; les fichiers journaliers demandent une relecture du champ `weekly`).

## Entraînement

```bash
# pré-entraînement JEPA (tiny : 1.1M ; mini : 3.4M)
python scripts/train.py --config-name lotsa_tiny_v3 wandb.run_name=tiny-v3
python scripts/train.py --config-name lotsa_mini_v3 wandb.run_name=mini-v3
# finetuning en forecaster, une époque cosinus, tous les checkpoints conservés
python scripts/train.py --config-name lotsa_tiny_v3_zeroshot \
    '+training.pretrained_encoder_path="checkpoints/timejepa_lotsa_tiny_v3/pretrain_True/<ckpt>"'
```

Le pré-entraînement sauvegarde tous les checkpoints (`save_top_k=-1`). Le juge est lu sur
un checkpoint de pré-entraînement ; le forecaster sur un checkpoint de finetuning. Les
deux ne se confondent pas : le finetuning efface le juge.

## Évaluation GIFT-Eval

```bash
CK=checkpoints/timejepa_lotsa_tiny_v3_zs/pretrain_False/<ckpt>
python scripts/evaluate_gift.py --config-name lotsa_tiny_v3_eval "+checkpoint_path=$CK"                       # nu
python scripts/evaluate_gift.py --config-name lotsa_tiny_v3_eval "+checkpoint_path=$CK" +tta_flip=true        # + moyenne par retournement de signe
python scripts/evaluate_gift.py --config-name lotsa_tiny_v3_eval "+checkpoint_path=$CK" \
    +tta_flip=true +ratein=mix +ratein_pool=true                                                             # + resampling par backtest
```

Résultats par configuration dans `evaluation/<model.name>/<ckpt>/gift<tag>/per_config/`,
mis en cache ; l'agrégat est la moyenne géométrique des ratios au naïf saisonnier
officiel, comme sur le leaderboard. `scripts/harness_fidelity.py` compare le harnais aux
lignes officielles d'un modèle tiers. Un modèle externe passe par
`configs/model/ext_*_eval.yaml` (`model.builder`).

## Le juge (TimeJEPA-EBM)

```bash
PT=checkpoints/timejepa_lotsa_tiny_v3/pretrain_True/epoch01_valloss0.6420.ckpt
# rang du vrai futur parmi 34 candidats, six configurations GIFT
python scripts/probe_energy.py --checkpoint $PT --model-config lotsa_tiny_v3_eval --standalone-targets
# proposer-juger-repondérer avec TTM-R3 comme proposeur, Nixtla
python scripts/evaluate_energy.py --checkpoint $PT --model-config lotsa_tiny_v3_eval \
    --proposer-ttm --calibrate-T --centered-bootstrap
```

`--standalone-targets` est la convention des lignées v3 (`contextualized_targets: false`) :
le candidat est encodé seul. L'énergie est `timejepa.judge.energy`.

## Publication des poids

```bash
python scripts/export_hf.py --checkpoint $PT --model-config lotsa_tiny_v3_eval --standalone-targets \
    --card docs/model_card_timejepa_ebm.md --out export/TimeJEPA-EBM
hf auth login
hf upload IUseAMouse/TimeJEPA-EBM export/TimeJEPA-EBM . --repo-type model
```

L'export écrit `config.json` et `model.safetensors` (lus par
`timejepa.hub.from_pretrained`), le checkpoint d'origine et la carte, et vérifie que les
poids exportés reproduisent le checkpoint au bit près, prévisions et énergies.

## Démonstration vidéo

```bash
python scripts/forecast_video.py --scene vortex --checkpoint <ckpt finetuné> --model-config lotsa_tiny_v3_eval
```

Chaque pixel est prévu comme une série indépendante ; sortie dans
`evaluation/video_forecast/`.

## Règles de travail

Une variable par expérience. Les prédictions sont écrites dans
un journal d'expériences tenu hors du dépôt, avant le lancement, avec leurs seuils de succès et d'échec,
et le verdict après. Rien n'est réglé sur le test GIFT-Eval ; l'oracle est un
diagnostic. Aucun fichier n'est supprimé, il est déplacé. Code et commentaires en
anglais, documents en français, carte de modèle en anglais.
