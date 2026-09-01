"""
Tests for the horizon extension (workstream 2, grow_future_query_table).

The predictor's query table is the ONLY model parameter whose shape depends
on prediction_length. The extension must: preserve the learned rows bit for
bit, initialize only the new ones, refuse merges that make no sense, and stay
STRICTLY opt-in - without the flag, a mismatch remains a loud failure (the
P3.2 refusal), never a silent random table.
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
    assert torch.equal(merged[KEY][:, :n, :], sd[KEY]), "learned rows altered"
    assert torch.equal(merged[KEY][:, n:, :],
                       dict(big.state_dict())[KEY][:, n:, :]), \
        "the new rows must come from the MODEL's init (reproducible)"


def test_grow_table_refuses_shrink_and_dmodel_mismatch():
    small, big = _model(96), _model(512)
    with pytest.raises(ValueError, match="longer than"):
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
    """Historical path preserved: an unintentional mismatch = loud failure."""
    ckpt = tmp_path / "p96.ckpt"
    _save_pretrain_ckpt(_model(96), ckpt)
    module = FinetuneModule(model=_model(512))
    with pytest.raises(RuntimeError):
        module.load_pretrained_encoder(str(ckpt))


def test_finetune_512_with_flag_loads_clean(tmp_path):
    """With the flag: zero core keys dropped, forward on the long horizon passes."""
    ckpt = tmp_path / "p96.ckpt"
    small = _model(96)
    _save_pretrain_ckpt(small, ckpt)

    big = _model(512)
    module = FinetuneModule(model=big, extend_horizon_queries=True)
    module.load_pretrained_encoder(str(ckpt))     # does not raise

    # the pretrained rows did travel
    n = dict(small.state_dict())[KEY].shape[1]
    assert torch.equal(dict(big.state_dict())[KEY][:, :n, :],
                       dict(small.state_dict())[KEY])

    # and the finetune forward produces the 512 horizon (63 target patches)
    with torch.no_grad():
        out = big.forward_finetune(torch.randn(2, 384, 1))
    assert out["forecast"].shape[1] == 512


def test_ctor_with_pretrained_path_loads_before_horizon_attr(tmp_path):
    """
    Regression (2026-08-22): __init__ called load_pretrained_encoder BEFORE
    setting self.extend_horizon_queries, which the loader reads -
    AttributeError on EVERY finetune launched with
    +training.pretrained_encoder_path, i.e. the standard protocol. Never
    caught because the tests built the module without a path. This test
    follows the exact path of the mix finetune crash.
    """
    import torch
    from timejepa.models import JEPATST
    from timejepa.training.finetune_module import FinetuneModule

    def _model():
        return JEPATST(input_length=384, prediction_length=96, patch_size=16,
                       stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                       predictor_num_layers=1, predictor_num_heads=4,
                       predictor_d_ff=64, decoder_type="quantile")

    ckpt = tmp_path / "pretrain.ckpt"
    torch.save({"state_dict": {f"model.{k}": v
                               for k, v in _model().state_dict().items()}}, ckpt)
    module = FinetuneModule(model=_model(), pretrained_encoder_path=str(ckpt))
    assert module.extend_horizon_queries is False
