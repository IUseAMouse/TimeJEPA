#!/usr/bin/env python
"""
Convertit les sous-ensembles UTILES du corpus Chronos au format TimeJEPA.

    # inventaire : ce qui est admis, ce qui est exclu et POURQUOI
    python scripts/prepare_chronos.py --list

    # conversion (4 datasets, quelques minutes)
    python scripts/prepare_chronos.py --out data/processed/chronos_extras

Pourquoi si peu, et pourquoi quand même
---------------------------------------
Vérifié le 2026-08-19 : `Salesforce/GiftEvalPretrain` est identique à l'octet à
LOTSA moins les 18 datasets d'éval GIFT — le « levier corpus » qu'on croyait y
trouver est vide, on s'entraîne déjà dessus. Du corpus Chronos
(`autogluon/chronos_datasets`, 53 sous-ensembles), tout ce qui n'est pas un
doublon de LOTSA ou une fuite d'évaluation tient dans une ALLOWLIST de quatre
noms — d'où une liste blanche explicite plutôt que des motifs d'exclusion :
pour 4 datasets fixés, énumérer ce qu'on admet est plus sûr qu'énumérer ce
qu'on refuse.

`--chunk-length 8192` par défaut, et c'est le vrai intérêt de ce corpus : la
décimation multi-résolution exige `1280·f ≤ longueur_du_morceau`, donc les
morceaux LOTSA de 2048 n'autorisent AUCUN facteur f ≥ 2. Ces fichiers-ci (et le
corpus synthétique) sont les seuls morceaux 8192, c'est-à-dire le carburant de
l'arm inter-résolution (G9.2). `choose_chunk_length` raccourcit automatiquement
quand les séries sont trop courtes (dominick est hebdomadaire, ~350 pas :
il sera probablement rejeté par --min-length — le résumé le dira).

Le mélange avec un corpus existant se fait par SYMLINK des .npy dans le
répertoire du corpus visé, jamais en générant dedans (doctrine du repo, cf.
generate_synthetic.py). Fichiers préfixés `chronos_` : provenance lisible dans
les logs de couverture, zéro collision de nom.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.lotsa import convert_subset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_chronos")
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets",
               "fsspec", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

CHRONOS_REPO = "autogluon/chronos_datasets"

# La liste blanche : les seuls sous-ensembles à la fois NOUVEAUX par rapport à
# LOTSA et sans recoupement avec les trois suites d'évaluation du projet
# (GIFT-Eval 97 configs, Nixtla, Monash locale). Vérifié un à un le 2026-08-19.
CHRONOS_ALLOWLIST = ("dominick", "ercot", "mexico_city_bikes", "ushcn_daily")

# Le reste du corpus, avec la raison — imprimé par --list pour audit. Les
# motifs à '*' couvrent des groupes entiers.
CHRONOS_EXCLUDED = {
    "exchange_rate": "EST l'éval Nixtla `exchange`",
    "electricity_15min": "source du Nixtla `electricity` (UCI)",
    "wiki_daily_100k": "recoupe l'éval Monash locale wikipedia-web-traffic",
    "solar* (solar, solar_1h)": "recoupe l'éval Monash locale solar-10-minute",
    "m4_*": "datasets d'éval GIFT",
    "monash_*": "doublons LOTSA et/ou datasets d'éval (traffic, weather, hospital…)",
    "taxi_*": "déjà dans LOTSA (taxi_30min)",
    "uber_tlc_*": "déjà dans LOTSA",
    "m5": "déjà dans LOTSA",
    "nn5": "déjà dans LOTSA (réadmis là-bas)",
    "weatherbench_*": "climat — era5+cmip6 pèsent déjà ~30 % du batch (audit G7.1)",
    "wind_farms_*": "déjà dans LOTSA (wind_farms_with_missing)",
    "training_corpus": "TSMixup du corpus Chronos entier : mélange des sources "
                       "ci-dessus, donc fuite par construction",
}

# ushcn_daily n'a pas de colonne `target` : chaque variable météo est une
# colonne. Chacune devient une série indépendante — le traitement univarié
# standard du projet.
USHCN_VALUE_COLUMNS = ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN")


def series_iter_chronos(subset: str):
    """Flux de séries 1-D float32 d'un sous-ensemble Chronos, en streaming."""
    from datasets import load_dataset

    ds = load_dataset(CHRONOS_REPO, subset, split="train", streaming=True)
    for row in ds:
        if subset == "ushcn_daily":
            for col in USHCN_VALUE_COLUMNS:
                v = row.get(col)
                if v is None:
                    continue
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1:
                    yield arr
            continue
        target = row.get("target")
        if target is None:
            continue
        arr = np.asarray(target, dtype=np.float32)
        if arr.ndim == 1:
            yield arr
        elif arr.ndim == 2:
            for channel in arr:
                yield channel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/chronos_extras"))
    ap.add_argument("--chunk-length", type=int, default=8192,
                    help="Maximum ; choose_chunk_length raccourcit par sous-"
                         "ensemble. 8192 = décimation possible jusqu'à f=6.")
    ap.add_argument("--min-length", type=int, default=1280,
                    help="Fenêtre requise (ctx 1024 + horizon 256).")
    ap.add_argument("--max-chunks-per-subset", type=int, default=200_000)
    ap.add_argument("--max-nan-fraction", type=float, default=0.05)
    ap.add_argument("--list", action="store_true", help="Inventaire seul.")
    ap.add_argument("--resume", action="store_true",
                    help="Sauter les sous-ensembles déjà convertis.")
    args = ap.parse_args()

    print()
    print("=" * 72)
    print(f"ADMIS ({len(CHRONOS_ALLOWLIST)}) — liste blanche explicite")
    print("=" * 72)
    for n in CHRONOS_ALLOWLIST:
        print(f"  ✓ {n}")
    print()
    print("=" * 72)
    print(f"EXCLUS ({len(CHRONOS_EXCLUDED)} groupes) — À RELIRE")
    print("=" * 72)
    for n, why in CHRONOS_EXCLUDED.items():
        print(f"  ✗ {n:<28} {why}")
    print()

    if args.list:
        print("--list : aucune conversion effectuée.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, subset in enumerate(CHRONOS_ALLOWLIST, 1):
        out_path = args.out / f"chronos_{subset}.npy"
        if args.resume and out_path.exists():
            logger.info(f"[{i}/{len(CHRONOS_ALLOWLIST)}] {subset}: déjà présent, sauté")
            continue
        logger.info(f"[{i}/{len(CHRONOS_ALLOWLIST)}] {subset} → {out_path.name}")
        t0 = time.time()
        try:
            written, stats, effective = convert_subset(
                series_iter_chronos(subset), out_path,
                chunk_length=args.chunk_length,
                min_length=args.min_length,
                max_chunks=args.max_chunks_per_subset,
                max_nan_fraction=args.max_nan_fraction,
            )
            if effective is None:
                logger.warning(f"    séries trop courtes (médiane < {args.min_length}) "
                               f"— inutilisable pour cette géométrie")
                continue
            if effective != args.chunk_length:
                logger.info(f"    longueur de morceau adaptée : {effective}")
            logger.info(f"    {stats.summary(effective, args.min_length)} "
                        f"({time.time() - t0:.0f} s)")
            total += written
        except Exception as exc:  # un sous-ensemble cassé ne tue pas les autres
            logger.error(f"  ✗ {subset} échoué : {type(exc).__name__}: {exc}")

    print()
    print(f"Terminé : {total:,} morceaux ÉCRITS ce run")
    print(f"Sortie : {args.out}")
    print()
    print("Mélange à un corpus (exemple) :")
    print("  mkdir -p data/processed/lotsa_chronos")
    print("  ln -s $(pwd)/data/processed/lotsa/*.npy data/processed/lotsa_chronos/")
    print("  ln -s $(pwd)/data/processed/chronos_extras/*.npy data/processed/lotsa_chronos/")
    print("  # puis l'audit d'équilibre sur le répertoire mixte")


if __name__ == "__main__":
    main()
