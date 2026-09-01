"""
RateIN - sampling-rate canonicalization at inference.

Decision 2026-08-31 (G9.3 premise verdict): the two sub-3M models that beat
us handle time scale AT THE INPUT - FlowState adjusts its internal step from
the provided seasonality, TinyCast detects the period by FFT (zero
parameters) and realigns. Our internal law E1 says the same: skill tracks the
NUMBER OF CYCLES seen in the context (operating band ~16-48 steps/cycle, i.e.
2-6 patch positions), and input interpolation is catastrophic (ECL x4:
skill -136%). Hence:

  * CAUSAL period detection (rfft + Fisher test, zero parameters - same
    fairness status as RobustScale's median/MAD: a context statistic, one
    uniform rule for the 97 configs);
  * DECIMATION ONLY toward the [16, 48] steps/cycle band (never k<1: we do
    not fabricate points);
  * forecast at h' = ceil(h/k) on the decimated grid (bonus: fewer rollouts
    on high-frequency long-term configs), then reinterpolation of the full
    fan to the native grid.

k=1 (default, and the detector's choice on any series without a significant
peak or already in the band) = STRICTLY identical eval path - pinned by test.
It is also the cheapest falsification test of the xres hypothesis: if even
oracle-k gains nothing, scale geometry is not the mechanism of the tail.
"""

from typing import Optional

import numpy as np

# Target operating band in steps/period (E1: 2-6 patch positions at stride
# 8). The chosen k is the SMALLEST factor that brings the period into it.
# The grid goes up to 48: a daily cycle seen in minutes (period 1440) needs
# k~32; at high k the available history bounds the effective context by
# itself (decimate takes what there is - the model is length-agnostic). The
# fallback NEVER goes below the band (we do not over-decimate a cycle).
BAND_LO, BAND_HI = 16, 48
K_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def detect_period(history: np.ndarray, max_window: int = 8192,
                  alpha: float = 0.05,
                  min_period: int = 16) -> Optional[int]:
    """Dominant period of the last `max_window` points, or None.

    Fisher significance test on the periodogram (g-statistic), Bonferroni
    threshold on the number of tested frequencies - the TinyCast protocol,
    conservative by construction: on white noise, the detection probability
    is ~alpha, and without detection k stays 1. Requirements: >= 2 full
    periods in the window, period >= min_period (below that, the [16,48]
    band is already reached, no point decimating).
    """
    x = np.asarray(history, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[-max_window:]
    n = len(x)
    if n < 4 * min_period:
        return None
    x = x - x.mean()
    if not np.any(x):
        return None

    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]                                    # drop the DC component
    freqs_idx = np.arange(1, len(spec) + 1)
    periods = n / freqs_idx
    # candidates: at least 2 full periods, and period >= min_period
    valid = (periods <= n / 2) & (periods >= min_period)
    if not valid.any():
        return None
    m = int(valid.sum())
    g = spec[valid] / spec[valid].sum()
    # Fisher threshold (first-term approximation): g* such that
    # m*(1-g*)^(m-1) = alpha. Applied to ALL peaks (conservative), and we
    # keep the SMALLEST significant period, not the strongest - measured at
    # the smoke (2026-08-31): on electricity/H, the dominant peak is weekly
    # (168 -> k=6) and decimating destroys the intra-day structure the model
    # was exploiting; if a significant cycle already lives in the band, the
    # right decision is k=1.
    g_star = 1.0 - (alpha / m) ** (1.0 / (m - 1))
    # Local maxima only: spectral leakage of a true peak splashes neighboring
    # bins above the threshold, and "the smallest significant period" would
    # become a lobe (measured: P=96 sinusoid detected as 93). A lobe is not a
    # local maximum of the periodogram.
    gp = np.concatenate(([0.0], g, [0.0]))
    local_max = (g >= gp[:-2]) & (g >= gp[2:])
    sig = (g >= g_star) & local_max
    if not sig.any():
        return None
    return int(round(periods[valid][sig].min()))


def choose_k(period: Optional[int]) -> int:
    """Smallest k that brings period/k into [BAND_LO, BAND_HI]; 1 otherwise."""
    if period is None or period <= BAND_HI:
        return 1
    for k in K_CANDIDATES:
        if BAND_LO <= period / k <= BAND_HI:
            return k
    # huge period with no exact k in the grid: take the largest k that does
    # not go BELOW the band (never over-decimate a cycle).
    fallback = [k for k in K_CANDIDATES if period / k >= BAND_LO]
    return fallback[-1] if fallback else 1


def decimate(x: np.ndarray, k: int) -> np.ndarray:
    """Mean-pool in blocks of k, RIGHT-ALIGNED (the last point of the last
    block is the last point of the series - the forecast origin does not
    move); the left excess is truncated."""
    if k == 1:
        return x
    n = (len(x) // k) * k
    return x[len(x) - n:].reshape(-1, k).mean(axis=1)


def reinterp_fan(fan_dec: np.ndarray, h: int, k: int) -> np.ndarray:
    """[h', Q] on the decimated grid -> [h, Q] on the native grid.

    Decimated block i covers native steps [i*k, (i+1)*k); its value is placed
    at the CENTER of the block (i*k + (k-1)/2) and each quantile level is
    linearly interpolated between centers (constant extrapolation at the
    edges). Level monotonicity survives: convex combination of sorted
    vectors.
    """
    if k == 1:
        return fan_dec[:h]
    h_dec = fan_dec.shape[0]
    centers = np.arange(h_dec) * k + (k - 1) / 2.0
    t = np.arange(h, dtype=np.float64)
    out = np.empty((h, fan_dec.shape[1]), dtype=fan_dec.dtype)
    for q in range(fan_dec.shape[1]):
        out[:, q] = np.interp(t, centers, fan_dec[:, q])
    return out
