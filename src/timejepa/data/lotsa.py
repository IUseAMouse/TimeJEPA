"""
LOTSA → format TimeJEPA.

LOTSA (corpus de pretrain de Moirai, ~27 Md d'observations, HuggingFace
`Salesforce/lotsa_data`) est deux ordres de grandeur au-dessus du corpus Monash
actuel, et surtout couvre TOUTES les fréquences là où le corpus actuel est
sous-quotidien. E10 a mesuré que deux datasets haute fréquence pèsent 48,7 % du
batch de pretrain : c'est ce déséquilibre que LOTSA doit corriger.

Trois contraintes de conception, chacune tirée d'un incident du projet.

1. **Aucun tableau `object`.** B19 : les tableaux numpy d'objets ont un refcount
   par élément, ce qui casse le copy-on-write de `fork` et fait exploser la RAM
   dès qu'on a plusieurs workers (51 GiB observés). À l'échelle de LOTSA ce
   serait fatal. La conversion produit donc UNIQUEMENT des tableaux DENSES
   float32, en segmentant les séries longues en morceaux de longueur fixe. Les
   pages d'un tableau dense sont partagées entre processus sans copie.

2. **Lecture memmap.** Les fichiers font des Go : `TimeSeriesDataset(use_mmap=True)`
   les lit sans les charger en RAM. Le paramètre est additif et vaut False par
   défaut, donc les configs existantes sont inchangées au bit près.

3. **Exclusion des datasets d'évaluation.** LOTSA contient ETT, electricity,
   traffic, weather... c'est-à-dire une partie de nos benchmarks. Pré-entraîner
   dessus invaliderait toute l'évaluation. `EVAL_OVERLAP_PATTERNS` filtre par
   sous-chaîne, et la conversion IMPRIME ce qu'elle exclut : à relire avant de
   lancer un pretrain, la liste des noms LOTSA n'étant pas figée.

Le coût de la segmentation : une fenêtre ne peut pas chevaucher deux morceaux.
Avec `chunk_length` 8192 et une fenêtre de 1280, on perd au plus ~15 % des
positions possibles aux frontières, et zéro si les séries sont plus courtes que
`chunk_length` (elles passent alors entières).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Sous-chaînes des noms de sous-ensembles LOTSA qui recoupent une évaluation.
#
# Deux listes SÉPARÉES, parce qu'elles ont des provenances différentes et ne se
# vérifient pas de la même façon. L'union est `EVAL_OVERLAP_PATTERNS`.
#
# ⚠️ POURQUOI CETTE LISTE EST CRITIQUE. Un sous-ensemble oublié ici, c'est un
# benchmark vu pendant le pretrain — et une évaluation « zero-shot » qui n'en est
# pas une. Le corpus Monash du projet a précisément ce défaut : electricity-hourly
# et traffic-hourly SONT les séries et la fenêtre temporelle des benchmarks Nixtla
# `electricity` et `traffic` (5260/26304 et 3508/17544 = les derniers 20 %), et le
# découpage train/val/test est séquentiel sur des indices groupés par série, donc
# l'entraînement couvre ces séries sur toute leur durée. Voir le §5 du registre
# expérimental. C'est ce que le protocole LOTSA corrige par construction.
#
# ✅ LISTE GIFT-EVAL VÉRIFIÉE le 2026-08-13 contre le dépôt officiel
# (https://huggingface.co/api/datasets/Salesforce/GiftEval/tree/main), dont les
# 28 répertoires sont : LOOP_SEATTLE, M_DENSE, SZ_TAXI, bitbrains_fast_storage,
# bitbrains_rnd, bizitobs_application, bizitobs_l2c, bizitobs_service,
# car_parts_with_missing, covid_deaths, electricity, ett1, ett2,
# hierarchical_sales, hospital, jena_weather, kdd_cup_2018_with_missing,
# m4_{daily,hourly,monthly,quarterly,weekly,yearly}, restaurant, saugeenday,
# solar, temperature_rain_with_missing, us_births.
# Les motifs ci-dessous les couvrent tous les 28 (test de non-régression).
#
# Note : la politique GIFT-Eval autorise l'entraînement sur LEUR split de train
# (« carefully constructed using earlier horizons that do not overlap with the
# test set »), et n'exige de déclarer une fuite que si le corpus contient un
# dataset du corpus de test. Exclure le dataset ENTIER est donc plus conservateur
# que le minimum requis — choix assumé, faute de pouvoir aligner nos découpages
# sur les leurs.

# Benchmarks Nixtla long-horizon (ceux que scripts/evaluate.py mesure aujourd'hui).
NIXTLA_OVERLAP_PATTERNS: Tuple[str, ...] = (
    "ett",            # ETTh1/2, ETTm1/2
    "electricity",    # ECL
    "traffic",        # traffic
    "weather",        # weather (et jena_weather côté GIFT-Eval)
    "exchange",       # exchange rate
    "illness", "ili",  # ILI
)

# Sources de GIFT-Eval, requises pour une revendication zero-shot sur ce benchmark.
GIFT_EVAL_OVERLAP_PATTERNS: Tuple[str, ...] = (
    "m4", "m3", "m1",   # compétitions M
    "m_dense",
    "bizitobs",
    "bitbrains",
    "car_parts",
    "covid",
    "hierarchical",     # hierarchical_sales
    "hospital",
    "jena",             # jena_weather
    "kdd",              # kdd_cup_2018
    "air_quality",      # kdd_cup_2018 EST la qualité de l'air à Pékin 2017-2018 ;
                        # beijing_air_quality et china_air_quality mesurent la
                        # même chose et seraient un quasi-doublon d'un dataset
                        # GIFT-Eval. Repéré en relisant la sortie de --list.
    "loop_seattle", "seattle",
    "restaurant",
    "saugeen",          # présent AUSSI dans le corpus Monash local
    "solar",            # idem
    "taxi",             # sz_taxi
    "temperature_rain",
    "birth",            # us_births
    "tourism",
    "nn5",
    "wiki",             # wikipedia web traffic
)

EVAL_OVERLAP_PATTERNS: Tuple[str, ...] = tuple(
    dict.fromkeys(NIXTLA_OVERLAP_PATTERNS + GIFT_EVAL_OVERLAP_PATTERNS)
)


def is_eval_overlap(name: str, patterns: Sequence[str] = EVAL_OVERLAP_PATTERNS) -> bool:
    """Vrai si le nom du sous-ensemble LOTSA recoupe un benchmark d'évaluation."""
    lowered = name.lower()
    return any(p in lowered for p in patterns)


def segment_series(
    series: np.ndarray,
    chunk_length: int,
    min_length: int,
) -> List[np.ndarray]:
    """
    Découpe une série en morceaux de longueur fixe, pour produire un tableau dense.

    Les morceaux sont NON CHEVAUCHANTS : le chevauchement est le travail du
    dataset (fenêtres glissantes), le dupliquer ici gonflerait le fichier sans
    ajouter d'information. Le reste final est conservé s'il atteint `min_length`,
    mais il est alors plus court — le tableau ne serait plus dense. On le REJETTE
    donc, sauf s'il constitue le seul morceau (série plus courte que
    `chunk_length` mais assez longue pour être utile) : dans ce cas la série est
    laissée à l'appelant, qui regroupera par longueur.

    Renvoie une liste de morceaux de longueur EXACTEMENT `chunk_length`, ou bien
    une liste à un élément de longueur `len(series)` si la série est plus courte.
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    n = series.shape[0]

    if n < min_length:
        return []
    if n < chunk_length:
        return [series]

    n_chunks = n // chunk_length
    return [series[i * chunk_length:(i + 1) * chunk_length] for i in range(n_chunks)]


def _finite(chunk: np.ndarray) -> bool:
    """LOTSA contient des NaN (séries à trous). Un NaN empoisonne RevIN et la loss."""
    return bool(np.isfinite(chunk).all())


def iter_dense_chunks(
    series_iter: Iterable[np.ndarray],
    chunk_length: int,
    min_length: int,
    max_chunks: Optional[int] = None,
    drop_non_finite: bool = True,
) -> Iterator[np.ndarray]:
    """
    Transforme un flux de séries de longueurs quelconques en un flux de morceaux
    de longueur EXACTEMENT `chunk_length`, prêts à être empilés en dense.

    Les séries plus courtes que `chunk_length` sont écartées ici : les mélanger
    obligerait à un tableau `object`, ce que la contrainte 1 interdit. Elles
    restent accessibles via un second passage à `chunk_length` plus court, ce que
    fait `prepare_subset` en choisissant automatiquement la longueur.
    """
    emitted = 0
    for series in series_iter:
        for chunk in segment_series(series, chunk_length, min_length):
            if chunk.shape[0] != chunk_length:
                continue
            if drop_non_finite and not _finite(chunk):
                continue
            yield chunk
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return


def write_dense_npy(
    chunks: Iterable[np.ndarray],
    out_path: Path,
    chunk_length: int,
    max_chunks: int,
) -> int:
    """
    Écrit les morceaux dans un `.npy` DENSE (N, chunk_length) float32, en
    streaming via `open_memmap` — la RAM ne voit jamais plus d'un morceau.

    Le fichier est pré-alloué à `max_chunks` puis tronqué au nombre réellement
    écrit (réécriture de l'en-tête via un second `open_memmap`), ce qui évite de
    devoir compter les morceaux à l'avance.

    Renvoie le nombre de morceaux écrits.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp.npy")
    buf = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(max_chunks, chunk_length)
    )

    # Une conversion dure des heures et n'a aucun retour intermédiaire sans ça :
    # impossible de distinguer « ça avance lentement » de « c'est bloqué ».
    report_every = max(1, max_chunks // 20)

    written = 0
    for chunk in chunks:
        if written >= max_chunks:
            break
        buf[written] = chunk
        written += 1
        if written % report_every == 0:
            logger.info(
                f"    {out_path.name}: {written:,}/{max_chunks:,} morceaux "
                f"({100 * written / max_chunks:.0f} %)"
            )
    buf.flush()
    del buf

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        logger.warning(f"Aucun morceau écrit pour {out_path.name} — fichier non créé")
        return 0

    # Recopie tronquée : le fichier final ne contient que ce qui a été écrit.
    src = np.load(tmp_path, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(written, chunk_length)
    )
    step = max(1, 1_000_000 // max(1, chunk_length))
    for start in range(0, written, step):
        # Borne explicite : `src` a max_chunks lignes, `dst` seulement `written`.
        # Sans le min, la dernière tranche (ou la première si step > written)
        # lit plus de lignes que la destination n'en contient.
        end = min(start + step, written)
        dst[start:end] = src[start:end]
    dst.flush()
    del dst, src
    tmp_path.unlink(missing_ok=True)

    logger.info(f"✓ {out_path.name}: {written:,} morceaux x {chunk_length} pas")
    return written
