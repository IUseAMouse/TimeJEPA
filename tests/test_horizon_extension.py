"""
Tests de l'extension d'horizon (chantier 2, grow_future_query_table).

La table de requêtes du prédicteur est le SEUL paramètre du modèle dont la
forme dépende de prediction_length. L'extension doit : préserver bit-à-bit les
lignes apprises, n'initialiser que les neuves, refuser les fusions qui n'ont
pas de sens, et rester STRICTEMENT opt-in — sans le flag, un mismatch reste un
échec bruyant (le refus P3.2), jamais une table aléatoire silencieuse.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.models.jepa_tst import grow_future_query_table             # noqa: E402
from timejepa.training.finetune_module import FinetuneModule             # noqa: E402

KEY = "predictor.future_position_embedding"


def _model(pred_len, d_model=32):
    return JEPATST(input_length=384, prediction_length=pred_len,
                   patch_size=16, stride=8, d_model=d_model,
                   num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="quantile")


def test_grow_table_copies_prefix_bit_exact():
    small, big = _model(96), _model(512)
    sd = dict(small.state_dict())
    merged = grow_future_query_table(big, sd)
    n = sd[KEY].shape[1]
    assert merged[KEY].shape == dict(big.state_dict())[KEY].shape
    assert torch.equal(merged[KEY][:, :n, :], sd[KEY]), "lignes apprises altérées"
    assert torch.equal(merged[KEY][:, n:, :],
                       dict(big.state_dict())[KEY][:, n:, :]), \
        "les lignes neuves doivent venir de l'init du MODÈLE (reproductible)"


def test_grow_table_refuses_shrink_and_dmodel_mismatch():
    small, big = _model(96), _model(512)
    with pytest.raises(ValueError, match="plus longue"):
        grow_future_query_table(small, dict(big.state_dict()))
    other = _model(96, d_model=64)
    with pytest.raises(ValueError, match="d_model"):
        grow_future_query_table(other, dict(small.state_dict()))


def test_grow_table_noop_when_shapes_match():
    m = _model(96)
    sd = dict(m.state_dict())
    assert grow_future_query_table(m, sd) is sd or \
        torch.equal(grow_future_query_table(m, sd)[KEY], sd[KEY])


def _save_pretrain_ckpt(model, path):
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, path)


def test_finetune_512_without_flag_raises(tmp_path):
    """Chemin historique préservé : mismatch non intentionnel = échec bruyant."""
    ckpt = tmp_path / "p96.ckpt"
    _save_pretrain_ckpt(_model(96), ckpt)
    module = FinetuneModule(model=_model(512))
    with pytest.raises(RuntimeError):
        module.load_pretrained_encoder(str(ckpt))


def test_finetune_512_with_flag_loads_clean(tmp_path):
    """Avec le flag : zéro clé cœur droppée, forward sur l'horizon long passe."""
    ckpt = tmp_path / "p96.ckpt"
    small = _model(96)
    _save_pretrain_ckpt(small, ckpt)

    big = _model(512)
    module = FinetuneModule(model=big, extend_horizon_queries=True)
    module.load_pretrained_encoder(str(ckpt))     # ne lève pas

    # les lignes pré-entraînées ont bien voyagé
    n = dict(small.state_dict())[KEY].shape[1]
    assert torch.equal(dict(big.state_dict())[KEY][:, :n, :],
                       dict(small.state_dict())[KEY])

    # et le forward finetune produit l'horizon 512 (63 patchs cibles)
    with torch.no_grad():
        out = big.forward_finetune(torch.randn(2, 384, 1))
    assert out["forecast"].shape[1] == 512
