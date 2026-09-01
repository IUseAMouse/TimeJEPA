"""
RateIN (canonicalisation du taux à l'inférence) — invariants épinglés :

  1. k=1 est un NO-OP STRICT (decimate/reinterp identités) — le défaut du
     harnais reste bit-identique, patron « défaut inerte » du repo.
  2. Le détecteur trouve une période propre, refuse le bruit blanc et les
     historiques trop courts (< 2 périodes) — sur du bruit, k reste 1.
  3. La décimation est alignée à droite (l'origine du forecast ne bouge pas).
  4. La ré-interpolation préserve les formes, la monotonie du fan et la
     phase (une sinusoïde décimée/réinterpolée reste corrélée > 0.99).
"""

import numpy as np
import pytest

from timejepa.evaluation.ratein import (BAND_HI, BAND_LO, K_CANDIDATES,
                                        choose_k, decimate, detect_period,
                                        reinterp_fan)


# --------------------------------------------------------------- détection

def test_detects_clean_sine_period():
    t = np.arange(4096)
    x = np.sin(2 * np.pi * t / 96) + 0.05 * np.random.default_rng(0).normal(size=4096)
    p = detect_period(x)
    assert p is not None and abs(p - 96) <= 1


def test_white_noise_gives_none():
    rng = np.random.default_rng(1)
    hits = sum(detect_period(rng.normal(size=2048)) is not None
               for _ in range(20))
    # Fisher + Bonferroni : ~alpha de faux positifs, tolérance large.
    assert hits <= 3


def test_short_history_gives_none():
    x = np.sin(2 * np.pi * np.arange(50) / 96)
    assert detect_period(x) is None


def test_smallest_significant_period_wins():
    # Deux cycles réels (journalier 24 DOMINÉ par l'hebdo 168) : la décision
    # correcte est la PLUS PETITE période significative — 24, déjà en bande,
    # donc k=1 (le cas electricity/H mesuré au smoke du 2026-08-31).
    t = np.arange(4096)
    x = (3.0 * np.sin(2 * np.pi * t / 168) + 1.0 * np.sin(2 * np.pi * t / 24)
         + 0.05 * np.random.default_rng(7).normal(size=4096))
    p = detect_period(x)
    assert p is not None and abs(p - 24) <= 1
    assert choose_k(p) == 1


def test_nan_and_constant_are_safe():
    assert detect_period(np.full(2048, 3.14)) is None
    x = np.sin(2 * np.pi * np.arange(2048) / 128)
    x[::7] = np.nan
    assert detect_period(x) is not None            # NaN filtrés, pas un crash


# --------------------------------------------------------------- choix de k

def test_choose_k_in_band_or_none_is_identity():
    assert choose_k(None) == 1
    assert choose_k(24) == 1                       # déjà dans [16, 48]
    assert choose_k(BAND_HI) == 1


def test_choose_k_brings_period_into_band():
    for period in (96, 144, 288, 720, 1440):
        k = choose_k(period)
        assert k in K_CANDIDATES and k > 1
        assert BAND_LO <= period / k <= BAND_HI, (period, k)


def test_choose_k_never_overshoots_below_band():
    # période énorme sans k exact : le repli ne passe jamais sous BAND_LO
    k = choose_k(10_000)
    assert 10_000 / k >= BAND_LO


# --------------------------------------------------------------- décimation

def test_decimate_k1_is_identity():
    x = np.random.default_rng(2).normal(size=100)
    assert decimate(x, 1) is x


def test_decimate_right_aligned():
    x = np.arange(10, dtype=np.float64)            # len 10, k=3 -> 3 blocs
    d = decimate(x, 3)
    assert len(d) == 3
    # blocs [1,2,3],[4,5,6],[7,8,9] — le dernier point natif reste couvert
    assert d[-1] == pytest.approx((7 + 8 + 9) / 3)
    assert d[0] == pytest.approx((1 + 2 + 3) / 3)  # l'excédent (0) est à gauche


# ---------------------------------------------------------- ré-interpolation

def test_reinterp_k1_truncates_only():
    fan = np.random.default_rng(3).normal(size=(40, 9))
    out = reinterp_fan(fan, 32, 1)
    assert np.array_equal(out, fan[:32])


def test_reinterp_shapes_and_ceil():
    h, k = 900, 12
    h_dec = -(-h // k)
    assert h_dec == 75
    fan = np.random.default_rng(4).normal(size=(h_dec, 9))
    out = reinterp_fan(fan, h, k)
    assert out.shape == (h, 9)


def test_reinterp_preserves_monotonicity():
    rng = np.random.default_rng(5)
    fan = np.sort(rng.normal(size=(20, 9)), axis=1)    # niveaux triés
    out = reinterp_fan(fan, 60, 3)
    assert (np.diff(out, axis=1) >= -1e-12).all()


def test_reinterp_preserves_phase():
    # sinusoïde -> décimation par blocs -> réinterp : corrélation > 0.99 avec
    # la version lissée au même noyau (la phase demi-bloc est le point testé).
    h, k = 192, 4
    t = np.arange(h, dtype=np.float64)
    sig = np.sin(2 * np.pi * t / 96)
    dec = decimate(sig, k)                              # [48]
    out = reinterp_fan(dec[:, None], h, k)[:, 0]
    assert np.corrcoef(out, sig)[0, 1] > 0.99


# ----------------------------------------------------- no-op du harnais k=1

def test_harness_k1_bit_identical():
    """Le chemin evaluate_config avec ratein actif mais k=1 partout (bruit
    blanc -> détecteur muet) doit produire les MÊMES entrées modèle que le
    chemin sans flag : décimation et réinterp sont des identités strictes."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=2048)
    assert detect_period(x) is None or choose_k(detect_period(x)) == 1
    assert decimate(x, 1) is x
    fan = rng.normal(size=(48, 9))
    assert np.array_equal(reinterp_fan(fan, 48, 1), fan)
