"""
Synthetic series generation for pretraining (G8 / P2.5).

Why this module exists - three measured holes no real data fills
----------------------------------------------------------------
1. Missing frequencies (E17): the 10T/15T configs degrade on four independent
   datasets because the corpus has almost nothing between 5 min (PEMS) and
   hourly. We cannot download what does not exist; we can generate it.
2. Short series (G7.1): m1_*, monash_m3_*, tourism_* are all rejected
   "median < 1280" - a yearly series is ~30 points. Fifteen of the 97
   GIFT-Eval configs (A/Q/M/W) have this shape and the model has NEVER seen
   it.
3. Decimation geometry (G9): `_sample_resolution_factor` requires
   `1280*f <= chunk_length`; real chunks are 2048, so even f=2 is impossible.
   Synthetic data generates at any length - 8192 by default here, which
   allows f in {1..6} and also provides the (r, r') pairs for
   cross-resolution JEPA (G9.2).

Method - KernelSynth via random Fourier features
------------------------------------------------
Chronos (KernelSynth) samples GPs from compositions of {linear, RBF, periodic}
kernels; FlowState does the same (CauKer). Exact GP sampling costs an O(N^3)
Cholesky - prohibitive at N=8192. We use random Fourier features (Rahimi &
Recht 2007): for a stationary kernel, summing K sinusoids with frequencies
drawn from its spectral density converges to the GP as K grows. At K=64 per
component the sample is indistinguishable by eye and the cost is O(N*K).

Component bank, drawn then composed additively (sometimes with a
multiplicative envelope, like the x composition):
  * seasonalities: 1-3 log-uniform periods in [4, 2048] steps, each with
    decaying harmonics and random phases - covers the daily cycle seen at ALL
    frequencies (24 steps hourly, 96 at 15 min, 144 at 10 min...);
  * smooth drift: RBF via RFF, log-uniform lengthscale [16, 1024];
  * trend: linear or piecewise (breakpoint);
  * noise: Gaussian, sometimes Student-t (heavy tails - the G6 RevIN floor
    showed the real corpus contains them);
  * rarely: level shifts and impulses.

Output is the SAME format as prepare_lotsa.py: one dense
[n_chunks, length] float32 `.npy` per "family", memmappable, directly
globbable by `datasets: null`. No training code to modify - to mix with the
real corpus, just drop (or symlink) the files into the corpus directory.
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
              period_range=(4.0, 2048.0)) -> np.ndarray:
    """One seasonality: log-uniform period, 1-4 decaying harmonics."""
    period = np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1])))
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
    A synthetic "family" = a distribution over compositions.

    `period_range` is the frequency-coverage lever: a family with short
    periods (24-150 steps) mimics a daily cycle seen sub-hourly, a family
    with long periods (300-2048) mimics low frequency.

    `kind` (v3, 2026-08-24) dispatches the generator:
      * "kernel"       : KernelSynth/RFF compositions (the original generator);
      * "ops"          : bizitobs-style IT-ops - the #1 target of the E19 map
        (the only domain still red: 10S geomean 1.387, and bizitobs also
        loses at 5T/H against the top - a DOMAIN problem, not a grid one);
      * "intermittent" : sparse integer demand, car_parts-style (Croston).
    """
    name: str
    chunk_length: int = 8192
    period_range: tuple = (4.0, 2048.0)
    p_seasonal: float = 0.9
    p_smooth: float = 0.7
    p_trend: float = 0.6
    noise_scale: tuple = (0.05, 0.5)
    kind: str = "kernel"


def _sample_ops_series(spec: SyntheticSpec, rng: np.random.Generator) -> np.ndarray:
    """
    IT-ops series: small positive floor, BURSTS with Poisson arrivals (sharp
    rise, exponential decay, heavy-tailed amplitudes), segment zero-inflation
    (service off), diurnal modulation of the arrival rate, optional capacity
    saturation (bizitobs "l2c"), optional quantization (counters). Zeros stay
    EXACTLY zero (no offset): this is the near-zero regime G8.4b and the
    quantile head must learn to handle, not an artifact to avoid.
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


def _sample_intermittent_series(spec: SyntheticSpec, rng: np.random.Generator) -> np.ndarray:
    """
    INTEGER intermittent demand (car_parts, hierarchical_sales): mostly zeros,
    negative-binomial event sizes, optional occurrence seasonality, slow
    demand drift. Integer values - the shape the quantile head must calibrate
    on the sparse M/short configs.
    """
    n = spec.chunk_length
    p_event = np.exp(rng.uniform(np.log(0.02), np.log(0.35)))
    occ = np.full(n, p_event)
    if rng.random() < 0.5:                     # occurrence seasonality
        period = np.exp(rng.uniform(np.log(spec.period_range[0]),
                                    np.log(spec.period_range[1])))
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


def sample_series(spec: SyntheticSpec, rng: np.random.Generator) -> np.ndarray:
    if spec.kind == "ops":
        return _sample_ops_series(spec, rng)
    if spec.kind == "intermittent":
        return _sample_intermittent_series(spec, rng)
    n = spec.chunk_length
    parts = []
    if rng.random() < spec.p_seasonal:
        for _ in range(rng.integers(1, 4)):
            parts.append(rng.uniform(0.5, 2.0) * _seasonal(n, rng, spec.period_range))
    if rng.random() < spec.p_smooth:
        parts.append(rng.uniform(0.5, 2.0) * _smooth_gp(n, rng))
    if rng.random() < spec.p_trend:
        parts.append(rng.uniform(0.5, 2.0) * _trend(n, rng))
    if not parts:                                 # never an empty series
        parts.append(_seasonal(n, rng, spec.period_range))

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


# The three default families, one per measured hole. Names start with
# `synthetic_` so they are recognizable in sampler logs and excludable from a
# real-corpus glob if needed (negative pattern).
DEFAULT_FAMILIES = (
    # Short cycles (24-150 steps): the daily cycle seen sub-hourly - E17.
    SyntheticSpec("synthetic_subhourly", chunk_length=8192,
                  period_range=(24.0, 150.0)),
    # General-purpose KernelSynth: wide periods, the diversity baseline.
    SyntheticSpec("synthetic_broadband", chunk_length=8192,
                  period_range=(4.0, 2048.0)),
    # Low frequency: periods 4-52 - the SHAPE of yearly/quarterly/monthly
    # series (G7.1). Audit 2026-08-20 (T4): the 1280-chunk version was
    # decorative - 1 window per chunk, so 25k windows, the 3x oversampling
    # cap exhausted at ~2% of the epoch (at max LR, nothing retained), and
    # ineligible for any k1>1 pair. At 8192: 865 windows/chunk, sampler
    # weight comparable to the other families, and the family joins
    # cross-resolution learning. Cycles stay short (4-52 steps): the period
    # makes the "low frequency", not the chunk length.
    SyntheticSpec("synthetic_lowfreq", chunk_length=8192,
                  period_range=(4.0, 52.0), p_trend=0.8),
)

# v3 families (roadmap S2, E19 map of 2026-08-24): the three v1 families
# PLUS the two regimes that the champion's per-config map flagged as the
# corpus's remaining holes. Per-family sizing at generation time ("target
# batch first" principle - balance audit after mixing, as always).
V3_FAMILIES = DEFAULT_FAMILIES + (
    # bizitobs-style IT-ops - THE still-red domain of E19 (10S 1.387, and
    # x1.7-2.8 against the top up to 5T/H). No public data exists at 10S:
    # synthetic data has NO alternative here. Periods 256-4096 steps: the
    # diurnal cycle seen at 10S-5T grids at the 8192-chunk scale.
    SyntheticSpec("synthetic_ops_bursty", chunk_length=8192,
                  period_range=(256.0, 4096.0), kind="ops"),
    # Integer intermittent demand (car_parts 0.98 CRPS ratio, sparse
    # M/short). Short periods: yearly seasonality seen monthly.
    SyntheticSpec("synthetic_intermittent", chunk_length=8192,
                  period_range=(6.0, 52.0), kind="intermittent"),
)
