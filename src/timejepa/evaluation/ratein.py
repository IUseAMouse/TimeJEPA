"""
RateIN: sampling-rate canonicalization at inference.

The operator resamples the context by a factor k, forecasts on the resampled
grid, and maps the forecast fan back to the native grid:

  * k > 1 (decimation): the context is mean-pooled in blocks of k,
    right-aligned so the forecast origin does not move; the model forecasts
    ceil(h/k) steps; each quantile level is linearly interpolated back to the
    native grid between block centres.
  * k = 1/m < 1 (upsampling): the context is linearly interpolated onto a grid
    m times finer; the model forecasts h*m steps; the native fan is the mean
    of each block of m fine steps.
  * k = 1 is the unchanged evaluation path.

The choice of k is made by the caller (see scripts/evaluate_gift.py). The
period-based selector `detect_period` + `choose_k` below is an alternative
selector (`+ratein=fft`): it detects the dominant period of the context and
picks the smallest k that brings it into a target band of steps per cycle.
"""

from typing import Optional

import numpy as np

# Target band in steps per period for the period-based selector. The chosen
# k is the smallest factor that brings the period into it. The grid goes up
# to 48 so that a daily cycle sampled per minute (period 1440) can be reached.
# The fallback never goes below the band.
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
    # the strongest, so that decimation preserves the shortest cycle.
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


# Upsampling (k < 1).
# k = 1/m upsamples the context by linear interpolation between block centres
# (the inverse convention of decimate / reinterp_fan), asks for h*m steps, and
# pools the fan back by block means. k = 1 is unchanged.

def norm_k(k: int | float) -> int | float:
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


def resample_context(x: np.ndarray, k: int | float, max_len: int) -> np.ndarray:
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


def fc_horizon(h: int, k: int | float) -> int:
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


def to_native_fan(fan: np.ndarray, h: int, k: int | float) -> np.ndarray:
    """Back to the native grid whatever the candidate: reinterp (k > 1), pool
    (k < 1), truncation (k = 1)."""
    k = norm_k(k)
    if k == 1:
        return fan[:h]
    if k > 1:
        return reinterp_fan(fan, h, k)
    return pool_fan(fan, h, int(round(1.0 / k)))


# ------------------------------------------------------------- controls
# Decimation by block means changes three things at once: the number of
# samples per cycle, the smoothing of the values, and the native span the
# model's positions cover. Each control keeps some of them and drops the
# others, for an integer k > 1 (oracle sweeps only):
#   smooth    : trailing mean of width k at the native rate; native length,
#               native horizon, no return step (smoothing only);
#   subsample : the value at the centre of each block instead of its mean
#               (rate and span, no smoothing);
#   samespan  : block means over the native span the model sees at k = 1
#               only (rate and smoothing, no extra history).
CONTROLS = ("smooth", "subsample", "samespan")


def control_is_native(control: str) -> bool:
    """True when the control forecasts at the native rate (no return step)."""
    return control == "smooth"


def control_context(x: np.ndarray, k: int, max_len: int, control: str) -> np.ndarray:
    """The context the model sees at factor k under `control`."""
    if control not in CONTROLS:
        raise ValueError(f"unknown RateIN control {control!r} ({', '.join(CONTROLS)})")
    k = int(k)
    if k == 1:
        return x
    if control == "smooth":
        x = np.asarray(x[-max_len:], dtype=np.float64)
        c = np.concatenate(([0.0], np.cumsum(x)))
        idx = np.arange(1, len(x) + 1)
        lo = np.maximum(idx - k, 0)
        return ((c[idx] - c[lo]) / (idx - lo)).astype(np.float32)
    if control == "subsample":
        x = x[-(max_len * k):]
        n = (len(x) // k) * k
        return x[len(x) - n:].reshape(-1, k)[:, k // 2]
    return decimate(x[-max_len:], k)
