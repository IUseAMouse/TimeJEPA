"""
RateIN (rate canonicalization at inference) - pinned invariants:

  1. k=1 is a STRICT NO-OP (decimate/reinterp identities) - the harness
     default stays bit-identical, the repo's "inert default" pattern.
  2. The detector finds a clean period, refuses white noise and too-short
     histories (< 2 periods) - on noise, k stays 1.
  3. Decimation is right-aligned (the forecast origin does not move).
  4. Re-interpolation preserves shapes, fan monotonicity and phase (a
     decimated/reinterpolated sinusoid stays correlated > 0.99).
"""

import numpy as np
import pytest

from timejepa.evaluation.ratein import (BAND_HI, BAND_LO, K_CANDIDATES,
                                        choose_k, decimate, detect_period,
                                        reinterp_fan)


# --------------------------------------------------------------- detection

def test_detects_clean_sine_period():
    t = np.arange(4096)
    x = np.sin(2 * np.pi * t / 96) + 0.05 * np.random.default_rng(0).normal(size=4096)
    p = detect_period(x)
    assert p is not None and abs(p - 96) <= 1


def test_white_noise_gives_none():
    rng = np.random.default_rng(1)
    hits = sum(detect_period(rng.normal(size=2048)) is not None
               for _ in range(20))
    # Fisher + Bonferroni: ~alpha false positives, wide tolerance.
    assert hits <= 3


def test_short_history_gives_none():
    x = np.sin(2 * np.pi * np.arange(50) / 96)
    assert detect_period(x) is None


def test_smallest_significant_period_wins():
    # Two real cycles (daily 24 DOMINATED by weekly 168): the correct decision
    # is the SMALLEST significant period - 24, already in band, so k=1 (the
    # electricity/H case measured in the 2026-08-31 smoke test).
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
    assert detect_period(x) is not None            # NaN filtered, not a crash


# --------------------------------------------------------------- choice of k

def test_choose_k_in_band_or_none_is_identity():
    assert choose_k(None) == 1
    assert choose_k(24) == 1                       # already in [16, 48]
    assert choose_k(BAND_HI) == 1


def test_choose_k_brings_period_into_band():
    for period in (96, 144, 288, 720, 1440):
        k = choose_k(period)
        assert k in K_CANDIDATES and k > 1
        assert BAND_LO <= period / k <= BAND_HI, (period, k)


def test_choose_k_never_overshoots_below_band():
    # huge period with no exact k: the fallback never goes below BAND_LO
    k = choose_k(10_000)
    assert 10_000 / k >= BAND_LO


# --------------------------------------------------------------- decimation

def test_decimate_k1_is_identity():
    x = np.random.default_rng(2).normal(size=100)
    assert decimate(x, 1) is x


def test_decimate_right_aligned():
    x = np.arange(10, dtype=np.float64)            # len 10, k=3 -> 3 blocks
    d = decimate(x, 3)
    assert len(d) == 3
    # blocks [1,2,3],[4,5,6],[7,8,9] - the last native point stays covered
    assert d[-1] == pytest.approx((7 + 8 + 9) / 3)
    assert d[0] == pytest.approx((1 + 2 + 3) / 3)  # the excess (0) is on the left


# ---------------------------------------------------------- re-interpolation

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
    fan = np.sort(rng.normal(size=(20, 9)), axis=1)    # sorted levels
    out = reinterp_fan(fan, 60, 3)
    assert (np.diff(out, axis=1) >= -1e-12).all()


def test_reinterp_preserves_phase():
    # sinusoid -> block decimation -> reinterp: correlation > 0.99 with the
    # version smoothed by the same kernel (the half-block phase is the point
    # under test).
    h, k = 192, 4
    t = np.arange(h, dtype=np.float64)
    sig = np.sin(2 * np.pi * t / 96)
    dec = decimate(sig, k)                              # [48]
    out = reinterp_fan(dec[:, None], h, k)[:, 0]
    assert np.corrcoef(out, sig)[0, 1] > 0.99


# ----------------------------------------------------- harness no-op at k=1

def test_harness_k1_bit_identical():
    """The evaluate_config path with ratein active but k=1 everywhere (white
    noise -> mute detector) must produce the SAME model inputs as the
    flag-less path: decimation and reinterp are strict identities."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=2048)
    assert detect_period(x) is None or choose_k(detect_period(x)) == 1
    assert decimate(x, 1) is x
    fan = rng.normal(size=(48, 9))
    assert np.array_equal(reinterp_fan(fan, 48, 1), fan)


# --------------------------------------------------- RateIN x w (synergy)

def test_tta_forecast_relays_w():
    """The harness relays w to the model: exact identity at init (zero-init
    FiLM), real effect once the FiLM is perturbed, flip compatible."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import torch
    from evaluate_gift import tta_forecast
    from timejepa.models import JEPATST

    model = JEPATST(input_length=512, prediction_length=128, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp",
                    cross_resolution=True).eval()
    x = torch.randn(2, 512, 1)
    w = torch.full((2,), 0.5)
    with torch.no_grad():
        base = tta_forecast(model, x, 64)
        same = tta_forecast(model, x, 64, w=w)
        assert torch.allclose(base["forecast_denorm"],
                              same["forecast_denorm"], atol=0, rtol=0)
        model.predictor.w_film.weight.add_(0.05)
        diff = tta_forecast(model, x, 64, w=w)
        flip_w = tta_forecast(model, x, 64, flip=True, w=w)
    assert not torch.allclose(base["forecast_denorm"], diff["forecast_denorm"])
    assert flip_w["forecast_denorm"].shape == base["forecast_denorm"].shape
