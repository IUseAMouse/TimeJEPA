"""
Tests du convertisseur Chronos (scripts/prepare_chronos.py + convert_subset).

La zone à risque est la même que pour LOTSA : une erreur de périmètre
contamine tous les chiffres du projet d'un coup. D'où des tests sur la liste
blanche elle-même, pas seulement sur la mécanique.
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
    """Aucun nom fuitant ne doit JAMAIS entrer dans la liste blanche."""
    leaky = {"exchange_rate", "electricity_15min", "wiki_daily_100k",
             "solar", "solar_1h", "m4_daily", "m4_hourly", "m4_monthly",
             "m4_quarterly", "m4_weekly", "m4_yearly", "monash_traffic",
             "monash_weather", "monash_hospital", "training_corpus",
             "taxi_30min", "taxi_1h", "m5", "nn5"}
    assert not set(CHRONOS_ALLOWLIST) & leaky
    # et la liste d'exclus documente au moins les groupes critiques
    keys = " ".join(CHRONOS_EXCLUDED)
    for must in ("exchange_rate", "electricity_15min", "training_corpus",
                 "m4_*", "monash_*"):
        assert must in keys, f"{must} doit rester documenté comme exclu"


def test_convert_subset_writes_dense_f32(tmp_path):
    """Même contrat que write_dense_npy : dense (N, L) float32, memmappable."""
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
    """Séries plus courtes que le maximum : longueur adaptée, pas zéro sortie."""
    rng = np.random.default_rng(1)
    stream = (rng.normal(size=1500).astype(np.float32) for _ in range(8))
    written, stats, eff = convert_subset(
        stream, tmp_path / "y.npy",
        chunk_length=8192, min_length=1280, max_chunks=100,
    )
    assert eff is not None and eff < 8192, "le cas BEIJING_SUBWAY : adapter, pas jeter"
    assert written == 8


def test_convert_subset_refuses_too_short(tmp_path):
    """Médiane < min_length : rien d'écrit, None retourné, PAS de fichier."""
    stream = (np.ones(300, dtype=np.float32) for _ in range(20))
    written, stats, eff = convert_subset(
        stream, tmp_path / "z.npy",
        chunk_length=8192, min_length=1280, max_chunks=100,
    )
    assert written == 0 and eff is None
    assert not (tmp_path / "z.npy").exists()


def test_ushcn_columns_are_the_documented_five():
    assert USHCN_VALUE_COLUMNS == ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN")
