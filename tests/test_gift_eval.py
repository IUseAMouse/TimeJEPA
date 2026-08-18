"""
Tests du protocole GIFT-Eval (src/timejepa/evaluation/gift.py).

Le module réimplémente le harness officiel constante par constante ; chaque
test ci-dessous épingle une de ces transcriptions contre une valeur vérifiable
indépendamment (le data.py officiel, la table de saisonnalité gluonts, ou une
propriété analytique de la métrique). Si l'un casse, c'est que la transcription
a dérivé — pas un détail : ces règles décident si nos chiffres sont comparables
au leaderboard ou juste décoratifs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import gift                                     # noqa: E402


# ---------------------------------------------------------------------------
# Dérivations par config — valeurs recoupées avec le data.py officiel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config,expected", [
    ("m4_yearly/A/short", 6), ("m4_quarterly/Q/short", 8),
    ("m4_monthly/M/short", 18), ("m4_weekly/W/short", 13),
    ("m4_daily/D/short", 14), ("m4_hourly/H/short", 48),
    ("electricity/15T/short", 48), ("electricity/15T/medium", 480),
    ("electricity/15T/long", 720), ("bizitobs_application/10S/short", 60),
    ("bizitobs_application/10S/long", 900), ("us_births/D/short", 30),
    ("saugeen/M/short", 12), ("hierarchical_sales/W/short", 8),
])
def test_prediction_lengths(config, expected):
    assert gift.prediction_length(config) == expected


@pytest.mark.parametrize("freq,m", [
    ("15T", 96), ("5T", 288), ("10T", 144), ("10S", 360),
    ("H", 24), ("D", 1), ("W", 1), ("W-WED", 1), ("M", 12),
    ("Q", 4), ("A", 1),
])
def test_seasonalities_match_gluonts(freq, m):
    assert gift.seasonality(freq) == m


def test_windows_formula():
    # min(max(1, ceil(0.1 * L / h)), 20), M4 toujours 1
    assert gift.num_windows("m4_daily/D/short", 100_000) == 1
    assert gift.num_windows("electricity/H/short", 26304) == 20   # cap
    assert gift.num_windows("covid_deaths/D/short", 212) == 1     # ceil(21.2/30)
    assert gift.num_windows("us_births/D/short", 7305) == 20      # ceil(730/30)=25 -> 20
    assert gift.num_windows("hospital/M/short", 84) == 1


def test_storage_paths_cover_the_renames():
    assert gift.storage_path("loop_seattle/5T/long") == "LOOP_SEATTLE/5T"
    assert gift.storage_path("sz_taxi/15T/short") == "SZ_TAXI/15T"
    assert gift.storage_path("m_dense/H/medium") == "M_DENSE/H"
    assert gift.storage_path("temperature_rain/D/short") == "temperature_rain_with_missing"
    assert gift.storage_path("kdd_cup_2018/H/long") == "kdd_cup_2018_with_missing/H"
    assert gift.storage_path("car_parts/M/short") == "car_parts_with_missing"
    assert gift.storage_path("saugeen/W/short") == "saugeenday/W"
    assert gift.storage_path("m4_yearly/A/short") == "m4_yearly"
    assert gift.storage_path("bizitobs_l2c/5T/short") == "bizitobs_l2c/5T"
    assert gift.storage_path("bizitobs_application/10S/short") == "bizitobs_application"


def test_the_97_configs_match_the_official_baseline_exactly():
    """
    Le CSV Seasonal Naive vendu avec le package est la liste de lignes du
    leaderboard. Toute divergence avec GIFT_CONFIGS signifie qu'une des deux
    listes a dérivé de l'amont — le test échoue plutôt que de laisser un
    agrégat silencieusement calculé sur un sous-ensemble.
    """
    sn = gift.official_seasonal_naive()
    assert len(gift.GIFT_CONFIGS) == 97
    assert set(sn) == set(gift.GIFT_CONFIGS)
    for v in sn.values():
        assert v["MASE"] > 0 and v["CRPS"] > 0


# ---------------------------------------------------------------------------
# Découpage des fenêtres de test
# ---------------------------------------------------------------------------

def test_test_windows_are_the_last_wh_steps_nonoverlapping():
    y = np.arange(100, dtype=np.float32)
    inst = list(gift.iter_test_instances([y], h=10, windows=3))
    assert len(inst) == 3
    # fenêtres : [70:80], [80:90], [90:100], contexte = tout ce qui précède
    np.testing.assert_array_equal(inst[0].target, np.arange(70, 80))
    np.testing.assert_array_equal(inst[1].target, np.arange(80, 90))
    np.testing.assert_array_equal(inst[2].target, np.arange(90, 100))
    assert len(inst[0].context) == 70
    assert len(inst[2].context) == 90
    np.testing.assert_array_equal(inst[2].context, np.arange(90))


def test_short_series_and_all_nan_windows_are_skipped():
    short = np.arange(5, dtype=np.float32)          # contexte vide au 1er window
    hole = np.arange(40, dtype=np.float32)
    hole[30:40] = np.nan                            # dernière fenêtre 100% NaN
    inst = list(gift.iter_test_instances([short, hole], h=10, windows=2))
    # short: fenêtre k=0 -> start=-15 <= 0, k=1 -> start=-5 <= 0 : rien
    # hole:  k=0 -> cible [20:30] ok ; k=1 -> cible [30:40] tout-NaN : sautée
    assert len(inst) == 1
    assert inst[0].series_idx == 1
    np.testing.assert_array_equal(inst[0].target, np.arange(20, 30))


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def test_mase_hand_computed():
    acc = gift.MetricAccumulator()
    target = np.array([10., 12., 14.])
    median = np.array([11., 11., 11.])              # |err| = 1,1,3 -> moyenne 5/3
    acc.add(target, median, None, scale=2.0)
    r = acc.result()
    assert r["MASE"] == pytest.approx((5 / 3) / 2.0)
    assert r["MAE"] == pytest.approx(5 / 3)
    assert r["n_instances"] == 1


def test_mase_scale_uses_the_full_past():
    # saisonnier m=2 sur [0,1,4,5,8,9] : |y_t - y_{t-2}| = 4 partout
    past = np.array([0., 1., 4., 5., 8., 9.])
    assert gift.seasonal_error(past, m=2) == pytest.approx(4.0)
    # passé plus court que m : repli sur m=1, jamais un NaN silencieux
    assert gift.seasonal_error(np.array([3., 7.]), m=24) == pytest.approx(4.0)


def test_nan_targets_are_masked_not_imputed():
    acc = gift.MetricAccumulator()
    target = np.array([10., np.nan, 14.])
    median = np.array([11., 999., 11.])             # le 999 ne doit PAS compter
    acc.add(target, median, None, scale=1.0)
    r = acc.result()
    assert r["MASE"] == pytest.approx(2.0)          # (1 + 3) / 2
    assert r["n_obs"] == 2 if "n_obs" in r else True


def test_crps_of_a_point_forecast_equals_nd():
    """
    Propriété analytique : si les 9 quantiles coïncident avec la médiane, la
    moyenne sur q de 2·|e|·|1{y<=q̂} − q| vaut |e| (la grille 0.1..0.9 est
    symétrique), donc CRPS == ND. C'est le test le plus fort de la convention
    gluonts sans dépendre de gluonts.
    """
    rng = np.random.default_rng(0)
    target = rng.normal(10, 3, size=50).astype(np.float64)
    median = target + rng.normal(0, 1, size=50)
    quantiles = np.repeat(median[:, None], 9, axis=1)
    acc = gift.MetricAccumulator()
    acc.add(target, median, quantiles, scale=1.0)
    r = acc.result()
    assert r["CRPS"] == pytest.approx(r["ND"], rel=1e-9)


def test_seasonal_naive_repeats_the_last_cycle():
    ctx = np.arange(10, dtype=np.float32)
    out = gift.seasonal_naive_forecast(ctx, h=5, m=3)
    np.testing.assert_array_equal(out, [7, 8, 9, 7, 8])
    # NaN dans le cycle copié -> dernière valeur finie, pas de NaN en sortie
    ctx[-1] = np.nan
    out = gift.seasonal_naive_forecast(ctx, h=3, m=3)
    assert not np.isnan(out).any()


def test_aggregate_is_one_when_model_equals_baseline():
    metrics = {c: {"MASE": 1.3, "CRPS": 0.7} for c in list(gift.GIFT_CONFIGS)[:10]}
    agg = gift.aggregate(metrics, metrics)
    assert agg["geomean_MASE_ratio"] == pytest.approx(1.0)
    assert agg["geomean_CRPS_ratio"] == pytest.approx(1.0)
    assert agg["n_configs_MASE"] == 10


# ---------------------------------------------------------------------------
# Préparation de contexte (script)
# ---------------------------------------------------------------------------

def test_prepare_context_contract():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_gift import prepare_context

    # tronque À GAUCHE au multiple du stride : le bord récent reste intact
    ctx = prepare_context(np.arange(1000, dtype=np.float32), 1024, 8, 16)
    assert len(ctx) == 1000 - (1000 % 8)
    assert ctx[-1] == 999.0

    # plafonne au contexte du modèle
    ctx = prepare_context(np.arange(5000, dtype=np.float32), 1024, 8, 16)
    assert len(ctx) == 1024 and ctx[-1] == 4999.0

    # série plus courte qu'un patch : rembourrée à gauche par la valeur de bord
    ctx = prepare_context(np.array([5., 6., 7.], dtype=np.float32), 1024, 8, 16)
    assert len(ctx) == 16 and ctx[0] == 5.0 and ctx[-1] == 7.0

    # NaN interpolés linéairement — entrée modèle uniquement
    y = np.array([0., np.nan, 2., np.nan, np.nan, 5.] + [1.0] * 26,
                 dtype=np.float32)
    ctx = prepare_context(y, 1024, 8, 16)
    assert not np.isnan(ctx).any()
    assert ctx[1] == pytest.approx(1.0)

    # tout-NaN : refusé, pas inventé
    assert prepare_context(np.full(50, np.nan, dtype=np.float32), 1024, 8, 16) is None
