"""
Tests de l'arm JEPA inter-résolution (G9.2, config lotsa_tiny_xres).

Trois invariants à protéger, par ordre de gravité :
1. Les configs EXISTANTES sont inchangées au bit près (state_dict, dict d'item)
   — c'est ce qui garde les checkpoints reproduits rechargeables.
2. w=1 est l'identité exacte à l'init (FiLM zéro-init) — c'est ce qui rend le
   checkpoint xres utilisable au finetune, qui ne passe jamais w.
3. La physique : la cible est le futur CONTIGU du contexte, jamais un saut.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.data.dataset import TimeSeriesDataset                      # noqa: E402
from timejepa.training.jepa_pretrain_module import JEPAPretrainModule    # noqa: E402


def _model(cross_resolution=False):
    return JEPATST(input_length=384, prediction_length=96, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="mlp",
                   cross_resolution=cross_resolution)


def _dataset(tmp_path, n_series=3, length=2048, **kw):
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(n_series, length)).astype(np.float32)
    p = tmp_path / "toy.npy"
    np.save(p, arr)
    return TimeSeriesDataset(p, context_length=1024, prediction_length=256,
                             stride=64, **kw)


# ---------------------------------------------------------------------------
# 1. Les configs existantes ne bougent pas
# ---------------------------------------------------------------------------

def test_default_state_dict_has_no_w_film():
    """Protège le rechargement de TOUS les checkpoints reproduits."""
    keys = set(_model(cross_resolution=False).state_dict())
    assert not any("w_film" in k for k in keys)


def test_default_item_dict_unchanged(tmp_path):
    """Sans flag : mêmes clés qu'avant, jamais de 'w' — collate intact."""
    ds = _dataset(tmp_path)
    item = ds.get_item(0, allow_multi_resolution=True)
    assert set(item) == {"context", "target", "series_id", "start_idx",
                         "resolution_factor"}


# ---------------------------------------------------------------------------
# 2. w=1 est l'identité, et le refus est bruyant
# ---------------------------------------------------------------------------

def test_w_identity_at_init():
    """FiLM zéro-init : w quelconque = identité à l'initialisation (eval)."""
    m = _model(cross_resolution=True).eval()
    ctx, tgt = torch.randn(3, 384, 1), torch.randn(3, 96, 1)
    with torch.no_grad():
        a = m.forward_pretrain(ctx, tgt, contextualized_targets=False)
        b = m.forward_pretrain(ctx, tgt, contextualized_targets=False,
                               w=torch.tensor([1.0, 2.0, 4.0]))
    assert torch.equal(a["predictions"], b["predictions"])


def test_w_without_film_raises():
    """Un w≠1 perdu en silence = un arm qui entraîne sans conditionnement."""
    m = _model(cross_resolution=False)
    with pytest.raises(ValueError, match="use_w_film"):
        m.forward_pretrain(torch.randn(2, 384, 1), torch.randn(2, 96, 1),
                           contextualized_targets=False,
                           w=torch.tensor([2.0, 1.0]))


def test_module_rejects_xres_with_contextualized_targets():
    """[ctx@k1‖cible@k2] n'a pas de sens physique : refus à la construction."""
    with pytest.raises(ValueError, match="contextualized_targets"):
        JEPAPretrainModule(model=_model(cross_resolution=True),
                           cross_resolution=True,
                           contextualized_targets=True)


# ---------------------------------------------------------------------------
# 3. La physique des paires (k1, k2)
# ---------------------------------------------------------------------------

def test_pair_headroom_2048_only_k1_equals_1(tmp_path):
    """
    Sur morceaux 2048 avec ctx 1024 / pred 256 : ctx·k1 + pred·k2 ≤ 2048
    n'autorise que k1=1 (k2 jusqu'à 4). Les morceaux 8192 sont le seul moyen
    d'ouvrir k1>1 — c'est la raison d'être du corpus mixte de l'arm.
    """
    ds = _dataset(tmp_path, cross_resolution=True,
                  multi_resolution_factors=[1, 2, 4], p_multi_resolution=1.0)
    np.random.seed(0)
    pairs = {ds._sample_resolution_pair(2048, 0) for _ in range(200)}
    assert all(k1 == 1 for k1, _ in pairs), f"k1>1 impossible à 2048 : {pairs}"
    assert any(k2 > 1 for _, k2 in pairs), "aucune paire non triviale tirée"
    # à 8192, l'espace complet s'ouvre
    pairs_big = {ds._sample_resolution_pair(8192, 0) for _ in range(300)}
    assert any(k1 > 1 for k1, _ in pairs_big), "8192 doit ouvrir k1>1"


def test_target_is_physically_contiguous(tmp_path):
    """Le premier point brut de la cible est series[start + ctx·k1], toujours."""
    ds = _dataset(tmp_path, cross_resolution=True,
                  multi_resolution_factors=[1, 2, 4], p_multi_resolution=1.0)
    series = ds.normalized_data[0]
    np.random.seed(1)
    for _ in range(30):
        item = ds.get_item(0, allow_multi_resolution=True)
        ctx, tgt = np.asarray(item["context"]), np.asarray(item["target"])
        k1 = item["resolution_factor"]
        w = item["w"]
        k2 = int(round(w * k1))
        start = item["start_idx"]
        expected_first = series[start + 1024 * k1]
        assert tgt[0] == pytest.approx(float(expected_first)), \
            "la cible doit commencer exactement où le contexte s'arrête"
        assert len(ctx) == 1024 and len(tgt) == 256, "géométrie rendue constante"


def test_item_carries_w_and_batch_collates(tmp_path):
    """La clé 'w' est par item, float, et le collate par défaut l'empile."""
    from torch.utils.data import DataLoader

    ds = _dataset(tmp_path, cross_resolution=True,
                  multi_resolution_factors=[1, 2, 4], p_multi_resolution=1.0)

    class _Aug(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, i): return ds.get_item(i, allow_multi_resolution=True)

    batch = next(iter(DataLoader(_Aug(), batch_size=8)))
    assert "w" in batch and batch["w"].shape == (8,)
    assert batch["context"].shape == (8, 1024)


def test_coverage_log_reports_pair_eligibility(tmp_path, caplog):
    """Le log de couverture doit dire la vérité sur l'éligibilité des paires."""
    import logging
    with caplog.at_level(logging.INFO, logger="timejepa.data.dataset"):
        _dataset(tmp_path, cross_resolution=True,
                 multi_resolution_factors=[1, 2, 4], p_multi_resolution=0.5)
    text = caplog.text
    assert "Inter-résolution" in text
    assert "STÉRILE" not in text, "2048 permet k1=1<k2 : pas stérile"
