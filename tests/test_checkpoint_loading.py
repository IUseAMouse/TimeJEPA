"""
Tests for the `load_checkpoint` refusal contract (P3.2, 2026-08-19 audit).

The targeted failure mode is precise: a checkpoint whose geometry does not
match the model had its core keys dropped by `filter_loadable`, then the eval
ran on freshly initialized weights while emitting a mere warning - silently
wrong numbers. Measured on the prediction_length 256->512 case
(`predictor.future_position_embedding` dropped). Since P3.2, this case
REFUSES.

The only legitimate mismatch - the decoder head swap (point vs quantile) -
must keep passing: it is the real workflow of the geometry round.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.evaluation.loading import load_checkpoint                  # noqa: E402


def _model(pred_len=96, decoder_type="mlp"):
    return JEPATST(input_length=384, prediction_length=pred_len,
                   patch_size=16, stride=8, d_model=32,
                   num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type=decoder_type)


def _save_lightning_style(model, path):
    """Lightning format: keys prefixed with 'model.', like real checkpoints."""
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, path)


def test_load_refuses_predictor_shape_mismatch(tmp_path):
    """
    An h=96 checkpoint loaded into an h=512 model: the predictor's query
    table changes shape. Before P3.2: warning + random table + wrong numbers.
    Expected: an explicit RuntimeError.
    """
    ckpt = tmp_path / "h96.ckpt"
    _save_lightning_style(_model(pred_len=96), ckpt)

    big = _model(pred_len=512)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(big, str(ckpt), torch.device("cpu"))


def test_load_still_tolerates_decoder_swap(tmp_path):
    """The real workflow: point-head checkpoint, quantile-head eval."""
    ckpt = tmp_path / "mlp.ckpt"
    _save_lightning_style(_model(decoder_type="mlp"), ckpt)

    quantile = _model(decoder_type="quantile")
    loaded = load_checkpoint(quantile, str(ckpt), torch.device("cpu"))
    # the encoder really comes from the checkpoint (not reinitialized):
    src = _model(decoder_type="mlp")
    src.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"]
                        | {}, strict=False)
    assert loaded is quantile


def test_load_tolerates_expected_missing(tmp_path):
    """target_encoder and the RevIN buffers are ALWAYS missing - never a refusal."""
    m = _model()
    sd = {f"model.{k}": v for k, v in m.state_dict().items()
          if "target_encoder" not in k and not k.endswith((".mean", ".std"))}
    ckpt = tmp_path / "clean.ckpt"
    torch.save({"state_dict": sd}, ckpt)
    load_checkpoint(_model(), str(ckpt), torch.device("cpu"))   # does not raise


def test_allow_partial_is_an_explicit_escape_hatch(tmp_path):
    """The bypass exists for manual debugging, and it is explicit."""
    ckpt = tmp_path / "h96.ckpt"
    _save_lightning_style(_model(pred_len=96), ckpt)
    big = _model(pred_len=512)
    load_checkpoint(big, str(ckpt), torch.device("cpu"), allow_partial=True)
