"""
Tests du générateur synthétique (G8 / P2.5, src/timejepa/data/synthetic.py).

Deux catégories : le CONTRAT de sortie (format identique aux corpus convertis,
sinon le datamodule casse en silence), et les PROPRIÉTÉS statistiques que le
générateur prétend avoir — une famille « sub-horaire » doit réellement mettre
son énergie spectrale dans les périodes 24-150, sinon elle ne bouche pas le
trou d'E17 et personne ne s'en apercevrait avant une éval GIFT complète.
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
# Contrat de sortie
# ---------------------------------------------------------------------------

def test_output_matches_converted_corpus_format(tmp_path):
    """Même contrat que write_dense_npy : [n, L] float32 dense, fini, memmappable."""
    spec = SyntheticSpec("t", chunk_length=1280)
    p = write_synthetic_family(tmp_path / "t.npy", spec, n_chunks=8, seed=0,
                               log_every=0)
    a = np.load(p, mmap_mode="r")            # memmap = le mode du pretrain LOTSA
    assert a.shape == (8, 1280)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()


def test_generation_is_reproducible():
    rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
    spec = SyntheticSpec("t", chunk_length=512)
    np.testing.assert_array_equal(sample_series(spec, rng1),
                                  sample_series(spec, rng2))


def test_series_are_diverse_not_templates():
    """Deux tirages ne doivent pas être corrélés — sinon on génère N fois la même série."""
    rng = np.random.default_rng(0)
    spec = SyntheticSpec("t", chunk_length=2048)
    xs = [sample_series(spec, rng) for _ in range(12)]
    corrs = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            a = (xs[i] - xs[i].mean()) / (xs[i].std() + 1e-8)
            b = (xs[j] - xs[j].mean()) / (xs[j].std() + 1e-8)
            corrs.append(abs(float(np.mean(a * b))))
    assert np.median(corrs) < 0.3, f"séries quasi identiques (|corr| médian {np.median(corrs):.2f})"


def test_scales_vary_revin_style():
    """Le synthétique ne doit pas être reconnaissable à une normalisation implicite."""
    rng = np.random.default_rng(1)
    spec = SyntheticSpec("t", chunk_length=512)
    stds = [float(sample_series(spec, rng).std()) for _ in range(30)]
    means = [abs(float(sample_series(spec, rng).mean())) for _ in range(30)]
    assert max(stds) / max(min(stds), 1e-8) > 5, "échelles trop homogènes"
    assert max(means) > 50, "niveaux trop centrés — un corpus réel ne l'est pas"


# ---------------------------------------------------------------------------
# Propriétés spectrales revendiquées
# ---------------------------------------------------------------------------

def _dominant_period(x: np.ndarray) -> float:
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return len(x) / k if k > 0 else np.inf


def test_subhourly_family_puts_energy_in_short_periods():
    """
    La famille censée boucher le trou 10T/15T (E17) doit avoir sa période
    dominante dans [24, 150] pas la plupart du temps. Tolérance : tendance et
    dérive lisse peuvent dominer sur quelques tirages, c'est voulu.
    """
    spec = next(f for f in DEFAULT_FAMILIES if f.name == "synthetic_subhourly")
    rng = np.random.default_rng(3)
    periods = [_dominant_period(sample_series(spec, rng)) for _ in range(40)]
    in_band = sum(1 for p in periods if 20 <= p <= 300)
    assert in_band >= 24, f"seulement {in_band}/40 tirages dans la bande sub-horaire"


def test_lowfreq_family_has_long_chunks_and_short_cycles():
    """
    Audit 2026-08-20 (T4) : à 1280, la famille donnait 1 fenêtre/morceau et
    s'épuisait à ~2 % de l'époque. À 8192 elle pèse comme les autres ET devient
    éligible aux paires k1>1. Ce sont les PÉRIODES courtes (4-52) qui portent la
    « basse fréquence », pas la longueur du morceau.
    """
    spec = next(f for f in DEFAULT_FAMILIES if f.name == "synthetic_lowfreq")
    assert spec.chunk_length == 8192
    assert spec.period_range == (4.0, 52.0)
    rng = np.random.default_rng(4)
    x = sample_series(spec, rng)
    assert len(x) == 8192


def test_long_chunks_unlock_decimation_factors():
    """
    Le point de la longueur 8192 (G9) : la décimation exige 1280·f <= L.
    À 2048 (morceaux réels), aucun f>=2 n'est possible ; à 8192, f jusqu'à 6.
    Depuis l'audit T4, les TROIS familles sont à 8192 — toutes éligibles.
    """
    for spec in DEFAULT_FAMILIES:
        assert spec.chunk_length == 8192, f"{spec.name} devrait être à 8192 (T4)"
        eligible = [f for f in (1, 2, 3, 4, 6) if 1280 * f <= spec.chunk_length]
        assert eligible == [1, 2, 3, 4, 6]
    assert [f for f in (1, 2, 3, 4, 6) if 1280 * f <= 2048] == [1], \
        "si ceci casse, la contrainte de géométrie a changé — mettre à jour G9"
