"""
RateIN: sampling-rate canonicalization at inference.

Skill tracks the number of cycles seen in the context (operating band about
16-48 steps per cycle), so the context is brought into that band before
forecasting, as FlowState does from the declared seasonality and TinyCast
from an FFT period estimate:

  * causal period detection (rfft + Fisher test, no parameters, a context
    statistic applied uniformly to every config);
  * decimation by an integer k toward the [16, 48] steps/cycle band;
  * forecast at h' = ceil(h/k) on the decimated grid, then reinterpolation of
    the full fan to the native grid.

k=1 (the default, and the detector's choice without a significant peak or
when already in the band) is the unchanged evaluation path. Below, k < 1
upsamples the context for low-frequency short-horizon configs.
"""

from typing import Optional

import numpy as np

# Target operating band in steps/period. The chosen k is the smallest factor
# that brings the period into it. The grid goes up to 48: a daily cycle seen
# in minutes (period 1440) needs k near 32. The fallback never goes below the
# band.
BAND_LO, BAND_HI = 16, 48
K_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def detect_period(history: np.ndarray, max_window: int = 8192,
                  alpha: float = 0.05,
                  min_period: int = 16) -> Optional[int]:
    """Dominant period of the last `max_window` points, or None.

    Fisher significance test on the periodogram (g-statistic) with a
    Bonferroni threshold on the number of tested frequencies, as in
    TinyCast: on white noise the detection probability is about alpha.
    Requires at least 2 full periods in the window and period >= min_period
    (below that the band is already reached).
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
    # Candidates: at least 2 full periods, and period >= min_period.
    valid = (periods <= n / 2) & (periods >= min_period)
    if not valid.any():
        return None
    m = int(valid.sum())
    g = spec[valid] / spec[valid].sum()
    # Fisher threshold (first-term approximation): g* such that
    # m*(1-g*)^(m-1) = alpha. The smallest significant period is kept, not
    # the strongest: on hourly electricity the dominant peak is weekly, and
    # decimating to it destroys the intra-day structure the model uses.
    g_star = 1.0 - (alpha / m) ** (1.0 / (m - 1))
    # Local maxima only: spectral leakage of a true peak pushes neighboring
    # bins above the threshold, and the smallest significant period would
    # otherwise be a lobe.
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
    # Huge period with no exact k in the grid: the largest k that does not
    # go below the band.
    fallback = [k for k in K_CANDIDATES if period / k >= BAND_LO]
    return fallback[-1] if fallback else 1


def decimate(x: np.ndarray, k: int) -> np.ndarray:
    """Mean-pool in blocks of k, right-aligned (the forecast origin does not
    move); the left excess is truncated."""
    if k == 1:
        return x
    n = (len(x) // k) * k
    return x[len(x) - n:].reshape(-1, k).mean(axis=1)


def reinterp_fan(fan_dec: np.ndarray, h: int, k: int) -> np.ndarray:
    """[h', Q] on the decimated grid -> [h, Q] on the native grid.

    Decimated block i covers native steps [i*k, (i+1)*k); its value is placed
    at the block center (i*k + (k-1)/2) and each quantile level is linearly
    interpolated between centers, constant at the edges. Level monotonicity
    survives (convex combination of sorted vectors).
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


# --------------------------------------------------------------------- k < 1
# RateIN-up: on low-frequency short-horizon configs (yearly to monthly at
# h <= 14) no integer k can bring a short cycle into the band. k = 1/m
# upsamples the context by linear interpolation between block centres (the
# inverse convention of decimate / reinterp_fan), asks for h*m steps, and
# pools the fan back by block means. k = 1 is unchanged.

def norm_k(k):
    """2.0 -> 2 (int), 0.5 stays float: one key per candidate."""
    return int(k) if float(k).is_integer() else float(k)


def upsample(x: np.ndarray, m: int) -> np.ndarray:
    """[n] -> [n*m]: native point i sits at fine position i*m + (m-1)/2 (its
    block centre, as in reinterp_fan); fine points are linearly interpolated
    between centres. decimate(upsample(x, m), m) == x on linear signals."""
    if m == 1:
        return x
    n = len(x)
    xf = np.asarray(x, dtype=np.float64)
    centers = np.arange(n) * m + (m - 1) / 2.0
    t = np.arange(n * m, dtype=np.float64)
    if n == 1:
        return np.full(m, xf[0], dtype=x.dtype)
    out = np.interp(t, centers, xf)
    # Linear extrapolation on the two edge half-blocks: np.interp holds the
    # edge value constant, which would bias the first and last block means.
    left, right = t < centers[0], t > centers[-1]
    out[left] = xf[0] + (t[left] - centers[0]) * (xf[1] - xf[0]) / m
    out[right] = xf[-1] + (t[right] - centers[-1]) * (xf[-1] - xf[-2]) / m
    return out.astype(x.dtype)


def resample_context(x: np.ndarray, k, max_len: int) -> np.ndarray:
    """The context the model sees at candidate k: decimated (k > 1, at most
    max_len*k native points), upsampled (k < 1, at most max_len/m native
    points so the result fits max_len), or x itself (k = 1)."""
    k = norm_k(k)
    if k == 1:
        return x
    if k > 1:
        return decimate(x[-(max_len * k):], k)
    m = int(round(1.0 / k))
    return upsample(x[-max(1, max_len // m):], m)


def fc_horizon(h: int, k) -> int:
    """Steps to ask the model for on its grid: ceil(h/k) decimated, h*m upsampled."""
    k = norm_k(k)
    if k >= 1:
        return -(-h // k)
    return h * int(round(1.0 / k))


def pool_fan(fan_fine: np.ndarray, h: int, m: int) -> np.ndarray:
    """[h*m, Q] on the fine grid -> [h, Q]: native step t covers fine steps
    [t*m, (t+1)*m) and its fan is their mean (a mean of sorted vectors is
    sorted)."""
    if m == 1:
        return fan_fine[:h]
    return fan_fine[:h * m].reshape(h, m, fan_fine.shape[1]).mean(axis=1)


def to_native_fan(fan: np.ndarray, h: int, k) -> np.ndarray:
    """Back to the native grid whatever the candidate: reinterp (k > 1), pool
    (k < 1), truncation (k = 1)."""
    k = norm_k(k)
    if k == 1:
        return fan[:h]
    if k > 1:
        return reinterp_fan(fan, h, k)
    return pool_fan(fan, h, int(round(1.0 / k)))
