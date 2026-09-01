"""
Tests for the synthetic generator (G8 / P2.5, src/timejepa/data/synthetic.py).

Two categories: the output CONTRACT (format identical to converted corpora,
otherwise the datamodule breaks silently), and the statistical PROPERTIES the
generator claims to have - a "sub-hourly" family must really put its spectral
energy in periods 24-150, otherwise it does not fill the E17 hole and nobody
would notice before a full GIFT eval.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.synthetic import (                                     # noqa: E402
    DEFAULT_FAMILIES,
    SyntheticSpec,
    sample_series,
    write_synthetic_family,
)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_matches_converted_corpus_format(tmp_path):
    """Same contract as write_dense_npy: [n, L] float32 dense, finite, memmappable."""
    spec = SyntheticSpec("t", chunk_length=1280)
    p = write_synthetic_family(tmp_path / "t.npy", spec, n_chunks=8, seed=0,
                               log_every=0)
    a = np.load(p, mmap_mode="r")            # memmap = the LOTSA pretrain mode
    assert a.shape == (8, 1280)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()


def test_generation_is_reproducible():
    rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
    spec = SyntheticSpec("t", chunk_length=512)
    np.testing.assert_array_equal(sample_series(spec, rng1),
                                  sample_series(spec, rng2))


def test_series_are_diverse_not_templates():
    """Two draws must not be correlated - otherwise we generate the same series N times."""
    rng = np.random.default_rng(0)
    spec = SyntheticSpec("t", chunk_length=2048)
    xs = [sample_series(spec, rng) for _ in range(12)]
    corrs = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            a = (xs[i] - xs[i].mean()) / (xs[i].std() + 1e-8)
            b = (xs[j] - xs[j].mean()) / (xs[j].std() + 1e-8)
            corrs.append(abs(float(np.mean(a * b))))
    assert np.median(corrs) < 0.3, f"series nearly identical (median |corr| {np.median(corrs):.2f})"


def test_scales_vary_revin_style():
    """The synthetic data must not be recognizable by an implicit normalization."""
    rng = np.random.default_rng(1)
    spec = SyntheticSpec("t", chunk_length=512)
    stds = [float(sample_series(spec, rng).std()) for _ in range(30)]
    means = [abs(float(sample_series(spec, rng).mean())) for _ in range(30)]
    assert max(stds) / max(min(stds), 1e-8) > 5, "scales too homogeneous"
    assert max(means) > 50, "levels too centered - a real corpus is not"


# ---------------------------------------------------------------------------
# Claimed spectral properties
# ---------------------------------------------------------------------------

def _dominant_period(x: np.ndarray) -> float:
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return len(x) / k if k > 0 else np.inf


def test_subhourly_family_puts_energy_in_short_periods():
    """
    The family meant to fill the 10T/15T hole (E17) must have its dominant
    period in [24, 150] steps most of the time. Tolerance: trend and smooth
    drift may dominate on a few draws, by design.
    """
    spec = next(f for f in DEFAULT_FAMILIES if f.name == "synthetic_subhourly")
    rng = np.random.default_rng(3)
    periods = [_dominant_period(sample_series(spec, rng)) for _ in range(40)]
    in_band = sum(1 for p in periods if 20 <= p <= 300)
    assert in_band >= 24, f"only {in_band}/40 draws in the sub-hourly band"


def test_lowfreq_family_has_long_chunks_and_short_cycles():
    """
    Audit 2026-08-20 (T4): at 1280, the family gave 1 window/chunk and ran out
    at ~2% of the epoch. At 8192 it weighs like the others AND becomes
    eligible for k1>1 pairs. It is the short PERIODS (4-52) that carry the
    "low frequency", not the chunk length.
    """
    spec = next(f for f in DEFAULT_FAMILIES if f.name == "synthetic_lowfreq")
    assert spec.chunk_length == 8192
    assert spec.period_range == (4.0, 52.0)
    rng = np.random.default_rng(4)
    x = sample_series(spec, rng)
    assert len(x) == 8192


def test_long_chunks_unlock_decimation_factors():
    """
    The point of length 8192 (G9): decimation requires 1280*f <= L.
    At 2048 (real chunks), no f>=2 is possible; at 8192, f up to 6.
    Since the T4 audit, ALL THREE families are at 8192 - all eligible.
    """
    for spec in DEFAULT_FAMILIES:
        assert spec.chunk_length == 8192, f"{spec.name} should be at 8192 (T4)"
        eligible = [f for f in (1, 2, 3, 4, 6) if 1280 * f <= spec.chunk_length]
        assert eligible == [1, 2, 3, 4, 6]
    assert [f for f in (1, 2, 3, 4, 6) if 1280 * f <= 2048] == [1], \
        "if this breaks, the geometry constraint changed - update G9"
