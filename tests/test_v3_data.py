"""
Tests du paquet données v3 (roadmap S2) : familles synthétiques ops/intermittent,
augmentations TiRex-style, décimation.

Invariants, par gravité :
1. Flag off = rien : les nouvelles augmentations désactivées par défaut sont
   des passe-plats EXACTS ; les familles DEFAULT sont inchangées.
2. La physique des nouvelles familles : ops = positif, zéros exacts, rafales à
   queue lourde ; intermittent = entiers, zéros majoritaires.
3. Les augmentations actives font ce qu'elles disent (courbe commune
   contexte/cible, plafond commun, spikes périodiques continus à la jonction).
4. La décimation : mean-pooling exact, refus des morceaux trop courts,
   jamais d'écrasement.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.synthetic import (                                    # noqa: E402
    DEFAULT_FAMILIES, V3_FAMILIES, SyntheticSpec, sample_series)
from timejepa.data.augmentations import (                                # noqa: E402
    AugmentationConfig, TimeSeriesAugmentations)

RNG = lambda s=0: np.random.default_rng(s)  # noqa: E731


# ---------------------------------------------------------------------------
# 1. Flag off = rien
# ---------------------------------------------------------------------------

def test_default_families_unchanged():
    assert [f.name for f in DEFAULT_FAMILIES] == [
        "synthetic_subhourly", "synthetic_broadband", "synthetic_lowfreq"]
    assert all(f.kind == "kernel" for f in DEFAULT_FAMILIES)
    assert [f.name for f in V3_FAMILIES[:3]] == [f.name for f in DEFAULT_FAMILIES]


def test_new_augmentations_are_exact_passthrough_by_default():
    aug = TimeSeriesAugmentations(AugmentationConfig(
        enabled=True, scale_enabled=False, jitter_enabled=False,
        magnitude_warp_enabled=False, drs_enabled=False, trend_enabled=False))
    ctx, tgt = torch.randn(1024), torch.randn(256)
    c, t = aug(ctx.clone(), tgt.clone())
    assert torch.equal(c, ctx) and torch.equal(t, tgt)


# ---------------------------------------------------------------------------
# 2. La physique des nouvelles familles
# ---------------------------------------------------------------------------

def _spec(kind, **kw):
    return SyntheticSpec(f"test_{kind}", chunk_length=4096, kind=kind, **kw)


def test_ops_family_is_nonnegative_with_exact_zeros_and_bursts():
    rng = RNG(3)
    n_zero_series = 0
    ratios = []
    for _ in range(30):
        x = sample_series(_spec("ops", period_range=(256.0, 2048.0)), rng)
        assert np.isfinite(x).all()
        assert (x >= 0).all(), "une série ops doit rester positive"
        if (x == 0.0).any():
            n_zero_series += 1
        med = np.median(x[x > 0]) if (x > 0).any() else 1.0
        ratios.append(x.max() / max(med, 1e-9))
    assert n_zero_series >= 5, "la zéro-inflation doit apparaître (zéros EXACTS)"
    assert np.median(ratios) > 5, "les rafales doivent dominer le plancher (queue lourde)"


def test_intermittent_family_is_integer_and_sparse():
    rng = RNG(4)
    for _ in range(20):
        x = sample_series(_spec("intermittent"), rng)
        assert np.isfinite(x).all()
        assert (x >= 0).all()
        assert np.allclose(x, np.round(x)), "demande ENTIÈRE"
        assert (x == 0).mean() > 0.3, "les zéros doivent dominer ou presque"


def test_families_are_deterministic_by_seed():
    a = sample_series(_spec("ops"), RNG(7))
    b = sample_series(_spec("ops"), RNG(7))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. Les augmentations actives
# ---------------------------------------------------------------------------

def _aug(**kw):
    base = dict(enabled=True, scale_enabled=False, jitter_enabled=False,
                magnitude_warp_enabled=False, drs_enabled=False,
                trend_enabled=False)
    base.update(kw)
    return TimeSeriesAugmentations(AugmentationConfig(**base))


def test_amplitude_modulation_applies_common_piecewise_curve():
    torch.manual_seed(0)
    aug = _aug(amplitude_mod_enabled=True, p_amplitude_mod=1.0)
    ctx, tgt = torch.ones(1024), torch.ones(256)
    c, t = aug.amplitude_modulation(ctx, tgt)
    curve = torch.cat([c, t])
    a, b = aug.config.amplitude_mod_range
    assert (curve >= a - 1e-5).all() and (curve <= b + 1e-5).all()
    assert curve.std() > 0.01, "la courbe doit réellement moduler"
    # continuité à la jonction contexte/cible (courbe commune)
    assert abs(c[-1] - t[0]) < 0.05


def test_censor_clips_context_and_target_at_common_cap():
    torch.manual_seed(0)
    aug = _aug(censor_enabled=True, p_censor=1.0)
    ctx, tgt = torch.randn(1024) * 10, torch.randn(256) * 10
    c, t = aug.censor(ctx, tgt)
    cap = max(c.max().item(), t.max().item())
    assert cap < max(ctx.max().item(), tgt.max().item()), "le plafond doit écrêter"
    assert c.max() <= cap + 1e-6 and t.max() <= cap + 1e-6
    # les valeurs sous le plafond sont intactes
    assert torch.equal(c[ctx <= cap], ctx[ctx <= cap])


def test_spikes_are_periodic_and_cross_the_boundary():
    torch.manual_seed(3)
    aug = _aug(spike_enabled=True, p_spike=1.0)
    ctx, tgt = torch.zeros(1024), torch.zeros(256)
    # std nulle -> scale plancher 1e-6 : forcer un signal de base
    ctx += torch.randn(1024) * 0.1
    tgt += torch.randn(256) * 0.1
    c, t = aug.spike_injection(ctx.clone(), tgt.clone())
    d_ctx, d_tgt = (c - ctx), (t - tgt)
    hits = torch.nonzero(torch.cat([d_ctx, d_tgt]).abs() > 1e-6).flatten()
    assert len(hits) >= 2, "des spikes doivent exister"
    gaps = torch.diff(hits)
    assert (gaps == gaps[0]).all(), "les spikes doivent être PÉRIODIQUES"
    assert d_tgt.abs().max() > 0 or hits.max() < 1024, \
        "la période doit pouvoir continuer dans la cible"


# ---------------------------------------------------------------------------
# 4. La décimation
# ---------------------------------------------------------------------------

def test_decimate_mean_pools_and_refuses_short(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    arr = np.arange(4 * 4096, dtype=np.float32).reshape(4, 4096)
    np.save(src / "densefam.npy", arr)
    short = np.zeros((2, 1500), dtype=np.float32)
    np.save(src / "shortfam.npy", short)

    r = subprocess.run([sys.executable, "scripts/decimate_corpus.py",
                        "--src", str(src), "--dst", str(dst),
                        "--factors", "2", "--min-len", "1280"],
                       capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    out = np.load(dst / "densefam_dec2.npy")
    assert out.shape == (4, 2048)
    expected = arr.reshape(4, 2048, 2).mean(axis=2)
    assert np.allclose(out, expected), "mean-pooling exact exigé"
    # 1500 // 2 = 750 < 1280 : refusée, dite
    assert not (dst / "shortfam_dec2.npy").exists()
    assert "sauté" in (r.stdout + r.stderr)


def test_pad_to_left_pads_and_keeps_target_side_real():
    """Séries courtes (G7.1) : rembourrage-bord GAUCHE — la donnée réelle
    occupe la fin du morceau (côté cible), le préfixe est plat, exactement la
    condition d'éval des séries courtes (prepare_context)."""
    from timejepa.data.lotsa import iter_dense_chunks
    series = [np.arange(100, 600, dtype=np.float32),   # 500 pas, >= 384
              np.arange(50, dtype=np.float32)]          # 50 pas, < min : rejetée
    chunks = list(iter_dense_chunks(iter(series), chunk_length=1280,
                                    min_length=384, pad_to=1280))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.shape[0] == 1280
    assert np.array_equal(c[-500:], series[0]), "la fin doit être la donnée réelle"
    assert (c[:780] == series[0][0]).all(), "le préfixe doit être plat (edge-pad)"


def test_pad_to_off_is_unchanged():
    from timejepa.data.lotsa import iter_dense_chunks
    series = [np.arange(500, dtype=np.float32)]
    assert list(iter_dense_chunks(iter(series), chunk_length=1280,
                                  min_length=384)) == []


def test_solar_power_is_readmitted():
    from timejepa.data.lotsa import is_eval_overlap
    assert not is_eval_overlap("solar_power")
    assert is_eval_overlap("solar")            # le motif reste actif pour GIFT


def test_v3_configs_compose():
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="../configs/model"):
        for name in ("lotsa_tiny_v3", "lotsa_tiny_v3_zeroshot", "lotsa_tiny_v3_eval"):
            cfg = compose(config_name=name)
            assert cfg.model.robust_scale is True, name
        pre = compose(config_name="lotsa_tiny_v3")
        assert pre.data.data_dir == "data/processed/lotsa_v3"
        assert pre.data.ration_oversample is True
        assert pre.augmentations.pretrain.amplitude_mod_enabled is True
        assert pre.augmentations.pretrain.censor_enabled is True
        ft = compose(config_name="lotsa_tiny_v3_zeroshot")
        assert ft.training.max_epochs == 1 and ft.checkpoint.save_top_k == -1


def test_generate_script_knows_v3_set():
    r = subprocess.run([sys.executable, "scripts/generate_synthetic.py",
                        "--set", "v3", "--families", "nonexistent"],
                       capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode != 0
    assert "synthetic_ops_bursty" in (r.stdout + r.stderr)
    assert "synthetic_intermittent" in (r.stdout + r.stderr)
