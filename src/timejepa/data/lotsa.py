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

# Sous-ensembles RÉADMIS malgré un motif qui les attrape (G8.1).
# -------------------------------------------------------------
# Les motifs sont des sous-chaînes, donc volontairement grossiers : « taxi »
# attrape sz_taxi (éval GIFT) ET taxi_30min (New York, sans rapport), « m1 »
# attrape m1_monthly alors que l'éval ne porte que sur M4. E17 a mesuré ce que
# coûte cette grossièreté : notre écart au leaderboard suit la couverture
# fréquentielle, et nous jetions gratuitement des fréquences.
#
# Référence de sécurité : `Salesforce/GiftEvalPretrain`, le corpus de
# pré-entraînement SANCTIONNÉ par le benchmark (152 sous-ensembles). Tout ce qui
# s'y trouve est déclaré non-fuitant pour GIFT-Eval par ses auteurs.
#
# ⚠️ MAIS cela ne suffit pas : nous évaluons aussi sur Nixtla et sur 8 datasets
# Monash locaux, que GIFT ne connaît pas. Chaque entrée ci-dessous est donc
# vérifiée contre LES TROIS suites, avec la raison en clair. Ne rien ajouter ici
# sans faire les trois vérifications.
EVAL_SAFE_OVERRIDES: Tuple[str, ...] = (
    # --- compétitions M1/M3 : l'éval ne porte que sur M4 -------------------
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "monash_m3_monthly", "monash_m3_other",
    "monash_m3_quarterly", "monash_m3_yearly",
    # Séries basse fréquence, précisément là où nos pires MASE se trouvent
    # (m4_yearly 5.08, m4_quarterly 1.54) : le corpus n'a presque rien
    # d'annuel ou de trimestriel.

    # --- tourisme : aucune suite d'éval ne le contient ---------------------
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",

    # --- nn5 : dans AUCUNE des trois suites d'éval actuelles ---------------
    # (nn5_daily a servi de corpus de transfert tenu à l'écart en G4.6 ; ce
    #  round est clos, et il n'apparaît ni dans Nixtla, ni dans les 8 Monash,
    #  ni dans les 97 configs GIFT.)
    "nn5_daily_with_missing", "nn5_weekly",

    # --- taxis New York, 30 min : l'éval GIFT porte sur SZ_TAXI (Shenzhen) --
    "taxi_30min",

    # --- KDD Cup 2022 : éolien à 10 MINUTES --------------------------------
    # L'éval GIFT porte sur kdd_cup_2018 (qualité de l'air de Pékin) : autre
    # compétition, autre domaine, autre année. Et c'est notre plus gros gain
    # attendu : 10T est la fréquence où nous perdons x1,94 (E17).
    "kdd2022",

    # --- COVID : l'éval GIFT porte sur covid_deaths ------------------------
    "covid19_energy",     # consommation électrique pendant la pandémie
    "covid_mobility",     # indices de mobilité

    # --- trafic Baidu (Chine) : l'éval Nixtla porte sur PEMS (Californie) ---
    "Q-TRAFFIC",

    # --- demande électrique australienne (30 min) --------------------------
    # L'éval Nixtla « electricity » est UCI/Portugal, l'éval Monash locale est
    # electricity-hourly (le même UCI). Le marché australien est un jeu
    # distinct, et GiftEvalPretrain le sanctionne.
    "australian_electricity_demand",

    # --- solar_power : LA donnée 10T réelle du corpus v3 (levée 2026-08-24) --
    # Les trois vérifications du contrat, faites une à une :
    # 1. GIFT-Eval : `solar_power` figure dans GiftEvalPretrain, le corpus de
    #    pré-entraînement SANCTIONNÉ par les auteurs du benchmark — déclaré
    #    non-fuitant vis-à-vis de leurs 97 configs, solar/10T comprise (même
    #    caution que kdd2022/taxi_30min ci-dessus, référence G8.1).
    # 2. Nixtla : aucun benchmark solaire dans les 7 (electricity, etth1/2,
    #    ettm1/2, traffic, weather) — aucun recoupement possible.
    # 3. Suite Monash LOCALE : elle évalue solar-10-minute — c'est la SEULE
    #    raison de l'ancienne exclusion, et cette suite est DÉPRÉCIÉE au profit
    #    de GIFT-Eval (décision de périmètre prévue explicitement par la note
    #    ci-dessous : « ils redeviennent admissibles »). Conséquence à dire :
    #    la ligne solar de la suite Monash locale cesse d'être zero-shot — la
    #    suite ne porte de toute façon aucun chiffre publiable (m=1, §1).
    "solar_power",

    # --- qualité de l'air chinoise -----------------------------------------
    # ⚠️ L'entrée la moins tranchée du lot : l'éval GIFT kdd_cup_2018 est aussi
    # de la qualité de l'air pékinoise, et un recouvrement de FENÊTRES
    # TEMPORELLES est concevable. GiftEvalPretrain les inclut tous les deux, ce
    # qui vaut caution des auteurs du benchmark. À retirer d'ici au moindre
    # doute — le gain (horaire, fréquence déjà bien couverte) est faible et ne
    # justifie pas un risque de contamination.
    "beijing_air_quality", "china_air_quality",
)

# Ce qui reste EXCLU parmi les sous-ensembles pourtant sanctionnés par GIFT, et
# pourquoi — la liste est aussi importante que la précédente :
#   traffic_hourly, traffic_weekly  : traffic_hourly EST le Nixtla `traffic` et
#       le Monash `traffic-hourly`. Contamination mesurée et documentée au §5.
#   weather                          : EST le Nixtla `weather`.
#   oikolab_weather                  : réanalyse météo, adjacente à `weather` —
#       et redondante, le corpus contient déjà era5 (30 tranches).
#   cdc_fluview_ilinet               : EST la source du Nixtla `illness`/ILI.
#   solar_power                      : le Monash local évalue solar-10-minute.
#   extended_web_traffic_with_missing, kaggle_web_traffic_weekly,
#   wiki-rolling_nips                : le Monash local évalue
#       wikipedia-web-traffic-extended.
# Les quatre derniers groupes ne sont bloqués que par la suite Monash LOCALE,
# dont le §1 note qu'elle tourne à saisonnalité m=1 (donc contre une baseline
# faible). Si cette suite est abandonnée au profit de GIFT-Eval, ils
# redeviennent admissibles — décision de périmètre, pas de sécurité.


def is_eval_overlap(name: str, patterns: Sequence[str] = EVAL_OVERLAP_PATTERNS,
                    overrides: Sequence[str] = EVAL_SAFE_OVERRIDES) -> bool:
    """
    Vrai si le nom du sous-ensemble LOTSA recoupe un benchmark d'évaluation.

    Un nom listé dans `overrides` est réadmis même s'il correspond à un motif :
    les motifs sont des sous-chaînes grossières et attrapent des homonymes sans
    rapport (voir EVAL_SAFE_OVERRIDES). La comparaison est exacte et
    insensible à la casse — jamais une sous-chaîne, sinon on rouvrirait par
    l'override le trou que le motif ferme.
    """
    lowered = name.lower()
    if lowered in {o.lower() for o in overrides}:
        return False
    return any(p in lowered for p in patterns)


def family_of(name: str) -> str:
    """
    Regroupe les sous-ensembles LOTSA qui ne sont qu'une tranche temporelle d'un
    même corpus : `cmip6_1850`…`cmip6_2010`, `era5_1989`…`era5_2018`,
    `largest_2017`…`largest_2021`, `gfc12_load`/`gfc14_load`/`gfc17_load`.

    Sans ce regroupement, le plafond par sous-ensemble ne protège de rien : E10 a
    mesuré que deux datasets pesaient 48,7 % du batch de pretrain, et LOTSA
    referait pire à plus grande échelle — cmip6 (33 tranches) et era5 (30) sont
    63 sous-ensembles sur 123, soit la moitié du corpus en réanalyse climatique,
    un signal lisse et saisonnier très éloigné des séries des benchmarks.
    """
    import re
    # suffixe d'année : _1850, _2018, ou collé comme gfc12_load
    base = re.sub(r"_(18|19|20)\d{2}$", "", name)
    base = re.sub(r"^gfc\d{2}_", "gfc_", base)
    return base


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


def impute_gaps(chunk: np.ndarray, max_nan_fraction: float) -> Optional[np.ndarray]:
    """
    Comble les trous par interpolation linéaire, ou renonce si le morceau est
    trop troué.

    Rejeter un morceau entier pour un seul NaN est intenable sur un corpus réel :
    mesuré sur LOTSA, HZMETRO perdait 160/160 morceaux et SHMETRO 2304/2304 —
    100 % du sous-ensemble pour quelques valeurs manquantes. Les trous sont la
    norme, pas l'exception (plusieurs sous-ensembles portent « _with_missing »
    dans leur nom).

    L'interpolation linéaire sur des trous courts est la pratique standard et
    reste honnête ; sur des trous longs elle fabrique du signal. D'où le seuil :
    au-delà de `max_nan_fraction`, on préfère perdre le morceau que l'inventer.

    Renvoie le morceau comblé, ou None s'il est trop troué / entièrement NaN.
    """
    finite = np.isfinite(chunk)
    n_missing = chunk.shape[0] - int(finite.sum())
    if n_missing == 0:
        return chunk
    if n_missing / chunk.shape[0] > max_nan_fraction:
        return None
    if not finite.any():
        return None

    filled = chunk.copy()
    idx = np.arange(chunk.shape[0])
    # np.interp étend par la valeur de bord, ce qui traite aussi les trous en
    # début et fin de morceau (un ffill/bfill implicite).
    filled[~finite] = np.interp(idx[~finite], idx[finite], chunk[finite])
    return filled


def choose_chunk_length(
    sample_lengths: Sequence[int],
    requested: int,
    min_length: int,
) -> Optional[int]:
    """
    Choisit la longueur de morceau EFFECTIVE d'un sous-ensemble.

    `chunk_length` est un MAXIMUM, pas une valeur imposée : chaque sous-ensemble
    produit son propre fichier dense, donc rien n'oblige deux sous-ensembles à
    partager la même longueur. Mesuré sur LOTSA : BEIJING_SUBWAY_30MIN n'a que
    des séries de 1572 pas et perdait ses 552 séries face à un chunk fixé à 2048.

    Règle : la médiane des longueurs observées, plafonnée par `requested` et
    plancher à `min_length`. Renvoie None si la médiane est sous `min_length` —
    le sous-ensemble ne peut alors produire aucune fenêtre utilisable.
    """
    if not sample_lengths:
        return None
    median = int(np.median(np.asarray(sample_lengths)))
    if median < min_length:
        return None
    return min(requested, median)


class ChunkStats:
    """
    Compte ce qui entre et ce qui sort de la segmentation.

    Sans ça, un sous-ensemble qui ne produit rien est indistinguable d'un
    sous-ensemble dont les séries sont trop courtes pour être utiles : le script
    disait « aucun morceau écrit » et laissait deviner. Or les deux cas appellent
    des décisions opposées — baisser `chunk_length`, ou accepter la perte.
    """

    __slots__ = ("series", "too_short", "lost_to_chunking", "non_finite", "imputed",
                 "emitted", "min_len", "max_len", "_nan_frac_sum")

    def __init__(self):
        self.series = 0
        self.too_short = 0          # < min_length : inutilisable de toute façon
        self.lost_to_chunking = 0   # >= min_length mais < chunk_length : RÉCUPÉRABLE
        self.non_finite = 0     # trop troués : renoncés
        self.imputed = 0        # trous courts : comblés par interpolation
        self.emitted = 0
        self.min_len = None
        self.max_len = None
        self._nan_frac_sum = 0.0

    def summary(self, chunk_length: int, min_length: int) -> str:
        if self.series == 0:
            return "aucune série lue (sous-ensemble vide ou colonne absente)"
        lengths = f"longueurs {self.min_len}–{self.max_len}"
        parts = [f"{self.series:,} séries ({lengths})",
                 f"{self.emitted:,} morceaux"]
        if self.too_short:
            parts.append(f"{self.too_short:,} trop courtes (<{min_length})")
        if self.lost_to_chunking:
            parts.append(
                f"⚠️ {self.lost_to_chunking:,} PERDUES entre {min_length} et "
                f"{chunk_length} — baisser --chunk-length les récupérerait"
            )
        if self.imputed:
            parts.append(f"{self.imputed:,} morceaux comblés (trous courts)")
        if self.non_finite:
            mean_frac = 100 * self._nan_frac_sum / self.non_finite
            parts.append(
                f"{self.non_finite:,} rejetés (trous : {mean_frac:.0f} % en moyenne)"
            )
            if mean_frac > 15:
                # Mesuré sur LOTSA : HZMETRO/SHMETRO ont ~23 % de NaN en blocs
                # RÉGULIERS de 23 pas — la fermeture nocturne du métro. Interpoler
                # y fabriquerait de la fréquentation à 3 h du matin. Monter le
                # seuil serait donc une erreur, pas une solution.
                parts.append(
                    "→ trous probablement STRUCTURELS (fermeture nocturne, "
                    "capteur hors service) : ne pas monter --max-nan-fraction, "
                    "ce sous-ensemble est inutilisable sans gestion des manquants"
                )
        return " | ".join(parts)


def iter_dense_chunks(
    series_iter: Iterable[np.ndarray],
    chunk_length: int,
    min_length: int,
    max_chunks: Optional[int] = None,
    max_nan_fraction: float = 0.05,
    stats: Optional[ChunkStats] = None,
    pad_to: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Transforme un flux de séries de longueurs quelconques en un flux de morceaux
    de longueur EXACTEMENT `chunk_length`, prêts à être empilés en dense.

    Les séries plus courtes que `chunk_length` sont écartées : les mélanger
    obligerait à un tableau `object`, ce que la contrainte 1 du module interdit.
    C'est un vrai coût — une série de 5 000 pas est utilisable pour une fenêtre
    de 1 280 mais sera perdue si `chunk_length` vaut 8 192 — d'où `ChunkStats`,
    qui le rend visible au lieu de le laisser deviner.

    `pad_to` (corpus v3, séries courtes — G7.1/roadmap S2) : si fourni, tout
    morceau accepté plus court que `pad_to` est REMBOURRÉ À GAUCHE par
    répétition de sa première valeur, jusqu'à `pad_to` exactement — la longueur
    d'émission devient max(chunk_length, pad_to). Pourquoi à gauche : la donnée
    RÉELLE occupe la fin du morceau, là où le dataset lit la CIBLE — la cible
    est donc toujours réelle, et le contexte « plat puis données » est
    précisément ce que l'évaluation impose déjà aux séries courtes
    (prepare_context edge-padde à gauche, evaluate_gift.py). Le modèle
    s'entraîne ainsi dans la condition où il sera évalué (m4_yearly : 19 pas
    de contexte). ⚠️ La cible (256 pas) doit être réelle : appeler avec
    `min_length ≥ 384` (256 de cible + ≥ 128 de contexte réel) — c'est le
    réglage recommandé `--min-length 384 --pad-to 1280`.
    """
    emit_len = max(chunk_length, pad_to) if pad_to else chunk_length
    emitted = 0
    for series in series_iter:
        arr = np.asarray(series, dtype=np.float32).ravel()
        n = arr.shape[0]
        if stats is not None:
            stats.series += 1
            stats.min_len = n if stats.min_len is None else min(stats.min_len, n)
            stats.max_len = n if stats.max_len is None else max(stats.max_len, n)
            if n < min_length:
                stats.too_short += 1
            elif n < chunk_length:
                stats.lost_to_chunking += 1

        for chunk in segment_series(arr, chunk_length, min_length):
            if pad_to and chunk.shape[0] < emit_len:
                # rembourrage-bord GAUCHE : la donnée réelle reste à la fin
                # (côté cible) — voir docstring.
                chunk = np.concatenate([
                    np.full(emit_len - chunk.shape[0], chunk[0],
                            dtype=np.float32), chunk])
            if chunk.shape[0] != emit_len:
                continue
            if not _finite(chunk):
                filled = impute_gaps(chunk, max_nan_fraction)
                if filled is None:
                    if stats is not None:
                        stats.non_finite += 1
                        stats._nan_frac_sum += float(
                            1.0 - np.isfinite(chunk).mean()
                        )
                    continue
                chunk = filled
                if stats is not None:
                    stats.imputed += 1
            yield chunk
            emitted += 1
            if stats is not None:
                stats.emitted = emitted
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


def convert_subset(
    series_stream: Iterator[np.ndarray],
    out_path: Path,
    *,
    chunk_length: int,
    min_length: int,
    max_chunks: int,
    max_nan_fraction: float = 0.05,
    sample_size: int = 200,
    pad_to: Optional[int] = None,
) -> Tuple[int, ChunkStats, Optional[int]]:
    """
    Convertit UN sous-ensemble (flux de séries 1-D) en un `.npy` dense.

    Factorisation du bloc « échantillonner les premières séries →
    `choose_chunk_length` → flux chaîné → `iter_dense_chunks` →
    `write_dense_npy` » pour les convertisseurs AUTRES que LOTSA
    (prepare_chronos.py, futurs corpus). `prepare_lotsa.py` garde sa version
    inline : c'est l'artefact qui a produit les corpus des runs reproduits
    (E14/E16), on ne le refactore pas sous un résultat publié.

    Retourne (morceaux_écrits, stats, chunk_length_effectif) —
    `chunk_length_effectif` vaut None si le sous-ensemble est inutilisable
    (médiane des longueurs < min_length), auquel cas rien n'est écrit.
    """
    sample: list = []
    for series in series_stream:
        sample.append(series)
        if len(sample) >= sample_size:
            break

    stats = ChunkStats()
    effective = choose_chunk_length(
        [len(x) for x in sample], chunk_length, min_length
    )
    if effective is None:
        return 0, stats, None

    def _chained(buffered=sample, rest=series_stream):
        yield from buffered
        yield from rest

    emit_len = max(effective, pad_to) if pad_to else effective
    chunks = iter_dense_chunks(
        _chained(),
        chunk_length=effective,
        min_length=min_length,
        max_chunks=max_chunks,
        max_nan_fraction=max_nan_fraction,
        stats=stats,
        pad_to=pad_to,
    )
    written = write_dense_npy(
        chunks, out_path, chunk_length=emit_len, max_chunks=max_chunks
    )
    return written, stats, emit_len
