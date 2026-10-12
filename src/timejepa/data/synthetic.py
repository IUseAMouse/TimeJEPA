"""
Synthetic series generation for pretraining.

Fills three holes of the real corpus: sub-hourly frequencies between 5 min
and hourly, short low-frequency series (yearly, quarterly, monthly), and
chunks long enough (8192 steps) for decimation factors up to 6 and for
cross-resolution pairs.

Method: KernelSynth (Chronos) samples GPs from compositions of kernels;
exact sampling costs an O(N^3) Cholesky, prohibitive at N=8192. Random
Fourier features (Rahimi and Recht, 2007) approximate a stationary kernel
with K sinusoids drawn from its spectral density, at cost O(N*K); K=64 per
component is enough. Components (seasonalities with decaying harmonics,
smooth RBF drift, linear or piecewise trend, Gaussian or Student-t noise,
rare level shifts and impulses) are composed additively, sometimes with a
multiplicative envelope.

Output has the same format as prepare_lotsa.py: one dense
[n_chunks, length] float32 `.npy` per family, memmappable, so the files
can be dropped into the corpus directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _seasonal(n: int, rng: np.random.Generator,
              period_range=(4.0, 2048.0), periods: Optional[list] = None) -> np.ndarray:
    """One seasonality: log-uniform period, 1-4 decaying harmonics. `periods`,
    when given, receives the drawn period (season recording, no extra draw)."""
    period = np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1])))
    if periods is not None:
        periods.append(float(period))
    t = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    n_harm = rng.integers(1, 5)
    for h in range(1, n_harm + 1):
        amp = rng.uniform(0.3, 1.0) / h          # decaying spectrum
        phase = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * h * t / period + phase)
    return out / max(np.std(out), 1e-8)


def _smooth_gp(n: int, rng: np.random.Generator, k: int = 64,
               lengthscale_range=(16.0, 1024.0)) -> np.ndarray:
    """
    RBF GP approximated by random Fourier features: the spectral density of an
    RBF with lengthscale l is a Gaussian with std 1/(2*pi*l).
    """
    ls = np.exp(rng.uniform(np.log(lengthscale_range[0]),
                            np.log(lengthscale_range[1])))
    freqs = rng.normal(0.0, 1.0 / (2 * np.pi * ls), size=k)
    phases = rng.uniform(0, 2 * np.pi, size=k)
    t = np.arange(n, dtype=np.float64)
    out = np.cos(2 * np.pi * np.outer(t, freqs) + phases).sum(axis=1)
    out *= np.sqrt(2.0 / k)
    return out / max(np.std(out), 1e-8)


def _trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Linear trend, or piecewise with one breakpoint."""
    t = np.linspace(-1.0, 1.0, n)
    slope = rng.uniform(-1.0, 1.0)
    out = slope * t
    if rng.random() < 0.3:                        # slope break
        cp = rng.integers(n // 4, 3 * n // 4)
        out[cp:] += rng.uniform(-1.0, 1.0) * (t[cp:] - t[cp])
    return out


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.15:                       # heavy tails
        return rng.standard_t(df=rng.integers(3, 8), size=n)
    return rng.normal(0.0, 1.0, size=n)


def _events(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Level shifts and impulses, rare - real data is full of them."""
    n = len(x)
    if rng.random() < 0.15:                       # level shift
        cp = rng.integers(n // 8, 7 * n // 8)
        x[cp:] += rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
    if rng.random() < 0.10:                       # impulses
        idx = rng.integers(0, n, size=rng.integers(1, 4))
        x[idx] += rng.uniform(3.0, 8.0, size=len(idx)) * rng.choice([-1, 1], size=len(idx))
    return x


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSpec:
    """
    A synthetic family: a distribution over compositions.

    `period_range` is the frequency-coverage lever: short periods (24-150
    steps) mimic a daily cycle seen sub-hourly, long periods (300-2048)
    mimic low frequency.

    `kind` selects the generator: "kernel" (KernelSynth/RFF compositions),
    "ops" (bizitobs-style IT operations) or "intermittent" (sparse integer
    demand, car_parts-style).
    """
    name: str
    chunk_length: int = 8192
    period_range: tuple = (4.0, 2048.0)
    p_seasonal: float = 0.9
    p_smooth: float = 0.7
    p_trend: float = 0.6
    noise_scale: tuple = (0.05, 0.5)
    kind: str = "kernel"


def _sample_ops_series(spec: SyntheticSpec, rng: np.random.Generator,
                       info: Optional[dict] = None) -> np.ndarray:
    """
    IT-ops series: small positive floor, bursts with Poisson arrivals (sharp
    rise, exponential decay, heavy-tailed amplitudes), zero-inflated segments
    (service off), diurnal modulation of the arrival rate, optional capacity
    saturation, optional quantization (counters). Zeros stay exactly zero:
    the quantile head must learn the near-zero regime.
    """
    n = spec.chunk_length
    t = np.arange(n, dtype=np.float64)

    base_level = np.exp(rng.uniform(np.log(0.02), np.log(2.0)))
    base = base_level * np.clip(1.0 + 0.3 * _smooth_gp(n, rng), 0.0, None)

    # diurnal modulation of the burst arrival rate
    rate_mod = np.ones(n)
    if rng.random() < 0.7:
        period = np.exp(rng.uniform(np.log(spec.period_range[0]),
                                    np.log(spec.period_range[1])))
        if info is not None:
            info["season"] = float(period)
        phase = rng.uniform(0, 2 * np.pi)
        rate_mod = np.clip(1.0 + rng.uniform(0.4, 1.0)
                           * np.sin(2 * np.pi * t / period + phase), 0.05, None)

    x = base.copy()
    expected_bursts = rng.uniform(3, 60)
    arrivals = np.nonzero(rng.random(n) < (expected_bursts / n) * rate_mod)[0]
    for idx in arrivals:
        amp = base_level * np.exp(rng.uniform(np.log(3.0), np.log(300.0)))
        dur = int(rng.integers(2, 96))
        decay = np.exp(-np.arange(dur) / max(1.0, dur / 4.0))
        end = min(n, idx + dur)
        x[idx:end] += amp * decay[:end - idx]

    # zero-inflation: segments where the service is off
    if rng.random() < 0.5:
        for _ in range(rng.integers(1, 4)):
            s = int(rng.integers(0, n))
            seg = int(rng.integers(n // 64, n // 8))
            x[s:s + seg] = 0.0

    # capacity saturation (the hard ceiling of "load to capacity" metrics)
    if rng.random() < 0.3:
        cap = np.quantile(x[x > 0] if (x > 0).any() else x,
                          rng.uniform(0.90, 0.995))
        x = np.minimum(x, max(cap, base_level))

    # quantization (rounded latencies, integer counters)
    if rng.random() < 0.4:
        step = base_level * float(rng.choice([0.01, 0.1, 1.0]))
        x = np.round(x / step) * step

    x = x * np.exp(rng.uniform(0.0, 4.0))     # positive scale, zeros intact
    return x.astype(np.float32)


def _sample_intermittent_series(spec: SyntheticSpec, rng: np.random.Generator,
                                info: Optional[dict] = None) -> np.ndarray:
    """
    Integer intermittent demand (car_parts, hierarchical_sales): mostly
    zeros, negative-binomial event sizes, optional occurrence seasonality,
    slow demand drift.
    """
    n = spec.chunk_length
    p_event = np.exp(rng.uniform(np.log(0.02), np.log(0.35)))
    occ = np.full(n, p_event)
    if rng.random() < 0.5:                     # occurrence seasonality
        period = np.exp(rng.uniform(np.log(spec.period_range[0]),
                                    np.log(spec.period_range[1])))
        if info is not None:
            info["season"] = float(period)
        occ *= np.clip(1.0 + 0.8 * np.sin(
            2 * np.pi * np.arange(n) / period + rng.uniform(0, 2 * np.pi)),
            0.05, None)
    events = rng.random(n) < np.clip(occ, 0.0, 0.95)
    sizes = 1.0 + rng.negative_binomial(int(rng.integers(1, 4)),
                                        rng.uniform(0.3, 0.8), size=n)
    x = np.where(events, sizes.astype(np.float64), 0.0)
    if rng.random() < 0.4:                     # slow demand drift
        x = np.round(x * np.clip(1.0 + 0.5 * _smooth_gp(n, rng), 0.2, None))
    unit = float(rng.choice([1.0, 1.0, 1.0, 6.0, 12.0]))   # sales units
    return (x * unit).astype(np.float32)


def sample_series(spec: SyntheticSpec, rng: np.random.Generator,
                  info: Optional[dict] = None) -> np.ndarray:
    """One series. `info` (optional dict) receives `season`: the period, in
    steps, of the series' reference cycle (the seasonal component with the
    largest weight for the kernel kind, the modulation period for ops and
    intermittent), or 0.0 when the series has none. Recording draws no random
    number, so the seasons of an existing file can be read back by replaying
    its seed (scripts/build_season_sidecars.py)."""
    if info is not None:
        info["season"] = 0.0
    if spec.kind == "ops":
        return _sample_ops_series(spec, rng, info)
    if spec.kind == "intermittent":
        return _sample_intermittent_series(spec, rng, info)
    n = spec.chunk_length
    parts = []
    seasonal = []                                 # (weight, period) of each seasonal component
    if rng.random() < spec.p_seasonal:
        for _ in range(rng.integers(1, 4)):
            # draw order: the weight, then the component's own draws
            weight = rng.uniform(0.5, 2.0)
            periods = []
            parts.append(weight * _seasonal(n, rng, spec.period_range, periods))
            seasonal.append((weight, periods[0]))
    if rng.random() < spec.p_smooth:
        parts.append(rng.uniform(0.5, 2.0) * _smooth_gp(n, rng))
    if rng.random() < spec.p_trend:
        parts.append(rng.uniform(0.5, 2.0) * _trend(n, rng))
    if not parts:                                 # never an empty series
        periods = []
        parts.append(_seasonal(n, rng, spec.period_range, periods))
        seasonal.append((1.0, periods[0]))
    if info is not None and seasonal:
        info["season"] = max(seasonal)[1]

    x = np.sum(parts, axis=0)
    if rng.random() < 0.25:                       # multiplicative composition
        x *= (1.0 + 0.5 * _smooth_gp(n, rng))
    x += rng.uniform(*spec.noise_scale) * _noise(n, rng)
    x = _events(x, rng)

    # Random scale and level: RevIN normalizes per instance, but the real
    # corpus is not zero-mean unit-variance and the synthetic data must not
    # be recognizable by its normalization.
    x = x * np.exp(rng.uniform(-1.0, 4.0)) + rng.uniform(-10.0, 1000.0)
    return x.astype(np.float32)


def write_synthetic_family(out_path: Path, spec: SyntheticSpec, n_chunks: int,
                           seed: int, log_every: int = 5000) -> Path:
    """Writes one family as a dense [n_chunks, chunk_length] float32 `.npy`."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_chunks, spec.chunk_length), dtype=np.float32)
    for i in range(n_chunks):
        out[i] = sample_series(spec, rng)
        if log_every and (i + 1) % log_every == 0:
            logger.info(f"  {spec.name}: {i + 1:,}/{n_chunks:,}")
    assert np.isfinite(out).all(), f"{spec.name}: non-finite values generated"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    logger.info(f"{out_path.name}: {n_chunks:,} chunks x {spec.chunk_length}")
    return out_path


# Names start with `synthetic_` so they are recognizable in sampler logs and
# excludable from a real-corpus glob.
DEFAULT_FAMILIES = (
    # Short cycles (24-150 steps): the daily cycle seen sub-hourly.
    SyntheticSpec("synthetic_subhourly", chunk_length=8192,
                  period_range=(24.0, 150.0)),
    # General-purpose KernelSynth: wide periods, the diversity baseline.
    SyntheticSpec("synthetic_broadband", chunk_length=8192,
                  period_range=(4.0, 2048.0)),
    # Low frequency: periods 4-52 steps, the shape of yearly, quarterly and
    # monthly series. The chunk stays at 8192 so the family has as many
    # windows per chunk as the others and is eligible for decimation; the
    # period, not the chunk length, makes the low frequency.
    SyntheticSpec("synthetic_lowfreq", chunk_length=8192,
                  period_range=(4.0, 52.0), p_trend=0.8),
)

# The default families plus two domain regimes the real corpus lacks.
V3_FAMILIES = DEFAULT_FAMILIES + (
    # bizitobs-style IT operations; no public data exists at 10 s. Periods
    # 256-4096 steps: the diurnal cycle on 10S to 5T grids.
    SyntheticSpec("synthetic_ops_bursty", chunk_length=8192,
                  period_range=(256.0, 4096.0), kind="ops"),
    # Integer intermittent demand (car_parts-style). Short periods: yearly
    # seasonality seen monthly.
    SyntheticSpec("synthetic_intermittent", chunk_length=8192,
                  period_range=(6.0, 52.0), kind="intermittent"),
)
