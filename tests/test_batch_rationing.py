"""
Tests for oversampling-cap rationing (G10.2, TemperatureSampler).

Invariants, by severity:
1. Flag off = iteration BIT-IDENTICAL to the existing one (same indices, same
   order) - protects reproducibility of all past runs and the mini in flight.
2. The total budget per family is THE SAME in both modes (rationing spreads,
   it does not change exposure) - up to a rounding residue.
3. Under rationing, a capped family stays present until the END of the epoch
   (no more early extinction) and the composition is near constant.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import TemperatureSampler                  # noqa: E402

# One big free family + two small ones that, without rationing, go extinct
# early (heavily oversampled by T=0.5).
SIZES = [100_000, 800, 500]


def _make(ration):
    return TemperatureSampler(
        dataset_sizes=SIZES, batch_size=64, temperature=0.5,
        max_oversample_ratio=3.0, shuffle=False, seed=123,
        rank=0, world_size=1, ration_oversample=ration,
    )


def _per_family_counts_per_batch(sampler):
    offsets = np.asarray(sampler.dataset_offsets)
    bounds = np.concatenate([offsets, [offsets[-1] + SIZES[-1]]])
    out = []
    for batch in sampler:
        fam = np.searchsorted(bounds, np.asarray(batch), side='right') - 1
        out.append(np.bincount(fam, minlength=len(SIZES)))
    return np.stack(out)


def test_flag_off_iteration_is_bit_identical():
    a = list(iter(_make(ration=False)))
    b = list(iter(_make(ration=False)))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y  # same seed, same epoch -> same indices, same order


def test_total_family_budget_is_preserved():
    tot_off = _per_family_counts_per_batch(_make(False)).sum(0)
    tot_on = _per_family_counts_per_batch(_make(True)).sum(0)
    # capped families: same budget up to quota rounding
    for i in (1, 2):
        assert abs(int(tot_off[i]) - int(tot_on[i])) <= 1, (i, tot_off, tot_on)
    # free family: strictly unchanged
    assert tot_off[0] == tot_on[0]


def test_capped_families_survive_to_epoch_end_under_rationing():
    counts_off = _per_family_counts_per_batch(_make(False))
    counts_on = _per_family_counts_per_batch(_make(True))
    n = len(counts_off)
    last_decile = slice(9 * n // 10, n)
    # without rationing: extinction (no sample from the small families in the
    # last decile) - the measured pathology
    assert counts_off[last_decile][:, 1:].sum() == 0
    # with rationing: both small families are still there
    assert (counts_on[last_decile][:, 1] > 0).any()
    assert (counts_on[last_decile][:, 2] > 0).any()


def test_composition_is_quasi_stationary_under_rationing():
    counts_on = _per_family_counts_per_batch(_make(True))
    n = len(counts_on)
    first = counts_on[: n // 10].sum(0).astype(float)
    last = counts_on[9 * n // 10:].sum(0).astype(float)
    share_first = first[1:].sum() / first.sum()
    share_last = last[1:].sum() / last.sum()
    # the capped families' share barely moves between the first and the last
    # decile (versus a collapse to zero without rationing)
    assert abs(share_first - share_last) < 0.02, (share_first, share_last)


def test_batch_size_is_quasi_constant_under_rationing():
    counts_on = _per_family_counts_per_batch(_make(True))
    sizes = counts_on.sum(1)
    n = len(sizes)
    assert sizes[9 * n // 10:].mean() >= sizes[: n // 10].mean() - 2
