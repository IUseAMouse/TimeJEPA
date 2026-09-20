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


# ------------------------------------------------- fractional allocation (2026-09-20)
def _family_counts(s, batches):
    bounds = np.cumsum([0] + list(s.dataset_sizes))
    idx = np.concatenate([np.asarray(b) for b in batches])
    return np.histogram(idx, bins=bounds)[0]


def test_fractional_off_is_bit_identical():
    for ration in (False, True):
        a = _sampler(ration_oversample=ration)
        b = _sampler(ration_oversample=ration, fractional_batch=False)
        assert _head(a, 1500) == _head(b, 1500)
        assert len(a) == len(b)


def test_fractional_train_mix_follows_the_temperature_whatever_batch_size():
    """T=0.5, cap 3, rationed: the realized batch averages batch_size and the
    families the budget does not bind get their p_i x B share, at batch 48
    and at batch 128 alike (the integer allocation gives 1 per family at 48)."""
    n = 6000
    for bs in (48, 128):
        s = _sampler(batch_size=bs, fractional_batch=True)
        batches = _head(s, n)
        sizes = np.array([len(b) for b in batches])
        assert abs(sizes.mean() - bs) / bs < 0.15, (bs, sizes.mean())
        counts = _family_counts(s, batches)
        expected = np.array(s.expected_per_dataset) * n
        quota = np.array([int(z * 3.0) for z in s.dataset_sizes]) / len(s) * n
        target = np.minimum(expected, quota)
        big = target > 50
        rel = np.abs(counts[big] - target[big]) / target[big]
        assert rel.max() < 0.08, rel.max()
        # no family ever exceeds one batch's share + 1 in a single batch
        per_batch_max = max(np.histogram(np.asarray(b), bins=np.cumsum([0] + list(s.dataset_sizes)))[0].max()
                            for b in batches[:500])
        assert per_batch_max <= int(max(s.expected_per_dataset)) + 2


def test_fractional_val_mix_is_proportional_and_independent_of_batch_size():
    """T=1, no oversampling, not rationed (the val sampler): the composition
    of the first 300 batches is proportional to family size at batch 48 as at
    batch 128 - the integer allocation made it uniform at 48."""
    shares = {}
    for bs in (48, 128):
        s = _sampler(batch_size=bs, temperature=1.0, max_oversample_ratio=1.0,
                     ration_oversample=False, fractional_batch=True)
        counts = _family_counts(s, _head(s, 300))
        shares[bs] = counts / counts.sum()
    prop = np.array(_sizes()) / sum(_sizes())
    big = prop > 0.01
    assert np.abs(shares[48][big] - prop[big]).max() < 0.02
    assert np.abs(shares[128][big] - prop[big]).max() < 0.02
    # the legacy allocation at 48: one per family, uniform, far from proportional
    legacy = _sampler(batch_size=48, temperature=1.0, max_oversample_ratio=1.0,
                      ration_oversample=False)
    lc = _family_counts(legacy, _head(legacy, 300))
    assert np.abs(lc / lc.sum() - prop)[big].max() > 0.05


def test_fractional_with_cap_at_batch_size():
    n = 20000
    s = _sampler(batch_size=48, fractional_batch=True, max_batch_size=48)
    sizes = [len(b) for b in _head(s, n)]
    assert max(sizes) <= 48
    free = [len(b) for b in _head(_sampler(batch_size=48, fractional_batch=True), n)]
    assert sum(free) - sum(sizes) <= 106
