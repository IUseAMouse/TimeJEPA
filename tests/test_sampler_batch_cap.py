"""
Cap on the realized batch of the rationed TemperatureSampler (2026-09-18).

With more families than batch_size, every family is clamped at one sample per
batch: the nominal batch is the number of families whatever batch_size, and
the realized one spikes at deterministic batch indices (three TimeSSM 10M
processes died of OOM at their 111,111th batch). The cap defers, never drops.
"""

import numpy as np
import pytest

from timejepa.data.datamodule import TemperatureSampler


def _sizes():
    rng = np.random.default_rng(0)
    return sorted([int(10 ** rng.uniform(2.5, 7)) for _ in range(83)] + [20000] * 23)


def _sampler(**kw):
    args = dict(dataset_sizes=_sizes(), batch_size=64, temperature=0.5, max_oversample_ratio=3.0,
                seed=7, rank=0, world_size=3, ration_oversample=True)
    args.update(kw)
    return TemperatureSampler(**args)


def _head(s, n):
    return [b for _, b in zip(range(n), iter(s))]


def test_batch_size_is_not_the_realized_batch_when_families_outnumber_it():
    a, b = _sampler(batch_size=48), _sampler(batch_size=64)
    assert a.actual_batch_size == b.actual_batch_size == 106        # one per family
    assert len(a) == len(b)
    assert _head(a, 400) == _head(b, 400)                           # the SAME sampler


def test_without_the_cap_the_iteration_is_unchanged():
    assert _head(_sampler(), 2000) == _head(_sampler(max_batch_size=None), 2000)
    assert _head(_sampler(), 2000) == _head(_sampler(max_batch_size=10_000), 2000)   # never binding


def test_cap_bounds_every_batch_and_defers_instead_of_dropping():
    n = 20000
    free = [len(b) for b in _head(_sampler(), n)]
    cap = int(np.ceil(np.mean(free) * 1.3))
    assert max(free) > cap                                          # the spikes exist
    capped = _head(_sampler(max_batch_size=cap), n)
    sizes = [len(b) for b in capped]
    assert max(sizes) <= cap
    # deferral: the exposure lost at the horizon is a bounded backlog, not a rate
    assert sum(free) - sum(sizes) <= 106
    # per family too: nobody is starved (largest backlog first)
    s = _sampler()
    bounds = np.cumsum([0] + list(s.dataset_sizes))
    def per_family(batches):
        idx = np.concatenate([np.asarray(b) for b in batches])
        return np.histogram(idx, bins=bounds)[0]
    diff = per_family(_head(_sampler(), n)) - per_family(capped)
    assert diff.min() >= 0 and diff.max() <= 2


def test_cap_needs_the_rationed_mode():
    with pytest.raises(ValueError):
        _sampler(ration_oversample=False, max_batch_size=32)
