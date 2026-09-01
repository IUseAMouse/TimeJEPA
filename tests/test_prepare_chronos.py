"""
Tests for the Chronos converter (scripts/prepare_chronos.py + convert_subset).

The risk zone is the same as for LOTSA: a scope mistake contaminates every
number in the project at once. Hence tests on the allowlist itself, not only
on the mechanics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.data.lotsa import convert_subset                            # noqa: E402
from prepare_chronos import (                                             # noqa: E402
    CHRONOS_ALLOWLIST, CHRONOS_EXCLUDED, USHCN_VALUE_COLUMNS,
)


def test_allowlist_excludes_every_leak():
    """No leaking name may EVER enter the allowlist."""
    leaky = {"exchange_rate", "electricity_15min", "wiki_daily_100k",
             "solar", "solar_1h", "m4_daily", "m4_hourly", "m4_monthly",
             "m4_quarterly", "m4_weekly", "m4_yearly", "monash_traffic",
             "monash_weather", "monash_hospital", "training_corpus",
             "taxi_30min", "taxi_1h", "m5", "nn5"}
    assert not set(CHRONOS_ALLOWLIST) & leaky
    # and the excluded list documents at least the critical groups
    keys = " ".join(CHRONOS_EXCLUDED)
    for must in ("exchange_rate", "electricity_15min", "training_corpus",
                 "m4_*", "monash_*"):
        assert must in keys, f"{must} must stay documented as excluded"


def test_convert_subset_writes_dense_f32(tmp_path):
    """Same contract as write_dense_npy: dense (N, L) float32, memmappable."""
    rng = np.random.default_rng(0)
    stream = (rng.normal(size=3000).astype(np.float32) for _ in range(10))
    written, stats, eff = convert_subset(
        stream, tmp_path / "x.npy",
        chunk_length=2048, min_length=1280, max_chunks=100,
    )
    a = np.load(tmp_path / "x.npy", mmap_mode="r")
    assert a.dtype == np.float32 and a.ndim == 2 and a.shape[0] == written
    assert eff == 2048 and written > 0
    assert np.isfinite(a[:5]).all()


def test_convert_subset_adapts_chunk_length(tmp_path):
    """Series shorter than the maximum: adapted length, not zero output."""
    rng = np.random.default_rng(1)
    stream = (rng.normal(size=1500).astype(np.float32) for _ in range(8))
    written, stats, eff = convert_subset(
        stream, tmp_path / "y.npy",
        chunk_length=8192, min_length=1280, max_chunks=100,
    )
    assert eff is not None and eff < 8192, "the BEIJING_SUBWAY case: adapt, do not drop"
    assert written == 8


def test_convert_subset_refuses_too_short(tmp_path):
    """Median < min_length: nothing written, None returned, NO file."""
    stream = (np.ones(300, dtype=np.float32) for _ in range(20))
    written, stats, eff = convert_subset(
        stream, tmp_path / "z.npy",
        chunk_length=8192, min_length=1280, max_chunks=100,
    )
    assert written == 0 and eff is None
    assert not (tmp_path / "z.npy").exists()


def test_ushcn_columns_are_the_documented_five():
    assert USHCN_VALUE_COLUMNS == ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN")
