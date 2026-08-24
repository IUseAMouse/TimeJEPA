"""
Tests du rationnement du plafond d'oversampling (G10.2, TemperatureSampler).

Invariants, par gravité :
1. Flag off = itération BIT-IDENTIQUE à l'existante (mêmes indices, même ordre)
   — protège la reproductibilité de tous les runs passés et du mini en cours.
2. Le budget total par famille est LE MÊME dans les deux modes (le rationnement
   étale, il ne change pas l'exposition) — à un résidu d'arrondi près.
3. Sous rationnement, une famille plafonnée reste présente jusqu'en FIN
   d'époque (plus d'extinction précoce) et la composition est quasi constante.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import TemperatureSampler                  # noqa: E402

# Une grosse famille libre + deux petites qui, sans rationnement, s'éteignent
# tôt (fortement suréchantillonnées par T=0.5).
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
        assert x == y  # même seed, même epoch -> mêmes indices, même ordre


def test_total_family_budget_is_preserved():
    tot_off = _per_family_counts_per_batch(_make(False)).sum(0)
    tot_on = _per_family_counts_per_batch(_make(True)).sum(0)
    # familles plafonnées : même budget à l'arrondi du quota près
    for i in (1, 2):
        assert abs(int(tot_off[i]) - int(tot_on[i])) <= 1, (i, tot_off, tot_on)
    # famille libre : strictement inchangée
    assert tot_off[0] == tot_on[0]


def test_capped_families_survive_to_epoch_end_under_rationing():
    counts_off = _per_family_counts_per_batch(_make(False))
    counts_on = _per_family_counts_per_batch(_make(True))
    n = len(counts_off)
    last_decile = slice(9 * n // 10, n)
    # sans rationnement : extinction (aucun échantillon des petites familles
    # dans le dernier décile) — c'est la pathologie mesurée
    assert counts_off[last_decile][:, 1:].sum() == 0
    # avec rationnement : les deux petites familles sont encore là
    assert (counts_on[last_decile][:, 1] > 0).any()
    assert (counts_on[last_decile][:, 2] > 0).any()


def test_composition_is_quasi_stationary_under_rationing():
    counts_on = _per_family_counts_per_batch(_make(True))
    n = len(counts_on)
    first = counts_on[: n // 10].sum(0).astype(float)
    last = counts_on[9 * n // 10:].sum(0).astype(float)
    share_first = first[1:].sum() / first.sum()
    share_last = last[1:].sum() / last.sum()
    # la part des familles plafonnées ne bouge presque plus entre le premier
    # et le dernier décile (contre un effondrement à zéro sans rationnement)
    assert abs(share_first - share_last) < 0.02, (share_first, share_last)


def test_batch_size_is_quasi_constant_under_rationing():
    counts_on = _per_family_counts_per_batch(_make(True))
    sizes = counts_on.sum(1)
    n = len(sizes)
    assert sizes[9 * n // 10:].mean() >= sizes[: n // 10].mean() - 2
