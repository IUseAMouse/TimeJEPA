"""
Tests du contrat de refus de `load_checkpoint` (P3.2, audit du 2026-08-19).

Le mode d'échec visé est précis : un checkpoint dont la géométrie ne correspond
pas au modèle voyait ses clés cœur droppées par `filter_loadable`, puis l'éval
tournait sur des poids fraîchement initialisés en émettant un simple warning —
des chiffres silencieusement faux. Mesuré sur le cas prediction_length 256→512
(`predictor.future_position_embedding` droppée). Depuis P3.2, ce cas REFUSE.

Le seul mismatch légitime — l'échange de tête de décodeur (point ↔ quantile) —
doit continuer de passer : c'est le workflow réel du round géométrie.
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
    """Format Lightning : clés préfixées 'model.', comme les vrais checkpoints."""
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, path)


def test_load_refuses_predictor_shape_mismatch(tmp_path):
    """
    Checkpoint h=96 chargé dans un modèle h=512 : la table de requêtes du
    prédicteur change de forme. Avant P3.2 : warning + table aléatoire +
    chiffres faux. Attendu : RuntimeError explicite.
    """
    ckpt = tmp_path / "h96.ckpt"
    _save_lightning_style(_model(pred_len=96), ckpt)

    big = _model(pred_len=512)
    with pytest.raises(RuntimeError, match="core components"):
        load_checkpoint(big, str(ckpt), torch.device("cpu"))


def test_load_still_tolerates_decoder_swap(tmp_path):
    """Le workflow réel : checkpoint à tête point, éval à tête quantile."""
    ckpt = tmp_path / "mlp.ckpt"
    _save_lightning_style(_model(decoder_type="mlp"), ckpt)

    quantile = _model(decoder_type="quantile")
    loaded = load_checkpoint(quantile, str(ckpt), torch.device("cpu"))
    # l'encodeur vient bien du checkpoint (pas réinitialisé) :
    src = _model(decoder_type="mlp")
    src.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"]
                        | {}, strict=False)
    assert loaded is quantile


def test_load_tolerates_expected_missing(tmp_path):
    """target_encoder et les buffers RevIN manquent TOUJOURS — jamais un refus."""
    m = _model()
    sd = {f"model.{k}": v for k, v in m.state_dict().items()
          if "target_encoder" not in k and not k.endswith((".mean", ".std"))}
    ckpt = tmp_path / "clean.ckpt"
    torch.save({"state_dict": sd}, ckpt)
    load_checkpoint(_model(), str(ckpt), torch.device("cpu"))   # ne lève pas


def test_allow_partial_is_an_explicit_escape_hatch(tmp_path):
    """Le contournement existe pour le debug manuel, et il est explicite."""
    ckpt = tmp_path / "h96.ckpt"
    _save_lightning_style(_model(pred_len=96), ckpt)
    big = _model(pred_len=512)
    load_checkpoint(big, str(ckpt), torch.device("cpu"), allow_partial=True)
