"""
G9.3 — la transition xres pretrain -> finetune (2026-08-31).

Invariants protégés :
1. DÉFAUTS INERTES : sans les nouvelles clés (lambda_anchor=0, pas de w),
   le finetune est bit-identique à l'existant — la doctrine du repo.
2. w traverse forecast() et forward_finetune() : identité à l'init
   (FiLM zéro-init), effet réel une fois le FiLM bruité.
3. L'ancre : λ>0 fait bouger l'encodeur en full_finetune, refuse
   linear_probe, et le target est bien la copie de l'online chargé
   (jamais le deepcopy aléatoire de la construction — le piège n°1).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                    # noqa: E402
from timejepa.training.finetune_module import FinetuneModule           # noqa: E402


def _model(cross_resolution=False):
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="mlp",
                   cross_resolution=cross_resolution)


def _module(**kw):
    kw.setdefault("finetune_mode", "full_finetune")
    model = kw.pop("model", None) or _model(kw.pop("cross_resolution", False))
    return FinetuneModule(model=model, **kw)


def _snapshot(module):
    return {k: v.detach().clone() for k, v in module.model.named_parameters()}


def _moved(before, module, prefix):
    return any(not torch.equal(before[k], v)
               for k, v in module.model.named_parameters()
               if k.startswith(prefix))


def _step(module):
    # Optimiseur nu (configure_optimizers exige un Trainer pour le scheduler) —
    # même approche que test_p0_regressions.
    opt = torch.optim.AdamW(module.model.parameters(), lr=1e-3)
    module.model.train()
    loss, _, _ = module._forward_and_loss(torch.randn(4, 512, 1),
                                          torch.randn(4, 128, 1))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss


# ---------------------------------------------------------------------------
# 1. Défauts inertes
# ---------------------------------------------------------------------------

def test_lambda_zero_is_bit_identical():
    torch.manual_seed(0); m0 = _module(lambda_anchor=0.0)
    torch.manual_seed(0); m1 = _module()
    m0.model.eval(); m1.model.eval()          # dropout hors jeu : comparaison pure
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l0, _, _ = m0._forward_and_loss(x, y)
        l1, _, _ = m1._forward_and_loss(x, y)
    assert torch.equal(l0, l1)
    assert m0._last_anchor is None


def test_forward_and_loss_default_w_is_none_path():
    m = _module(); m.model.eval()
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l_no_kw, _, _ = m._forward_and_loss(x, y)
        l_none, _, _ = m._forward_and_loss(x, y, w=None)
    assert torch.equal(l_no_kw, l_none)


# ---------------------------------------------------------------------------
# 2. w traverse le modèle
# ---------------------------------------------------------------------------

def test_forecast_w_identity_at_init_then_real_after_noise():
    model = _model(cross_resolution=True).eval()
    x = torch.randn(3, 512, 1)
    w2 = torch.full((3,), 2.0)
    with torch.no_grad():
        base = model.forecast(x)["forecast"]
        same = model.forecast(x, w=w2)["forecast"]
    # FiLM zéro-init : w quelconque == identité exacte
    assert torch.allclose(base, same, atol=0, rtol=0)
    # FiLM bruité : w=2 doit changer la sortie (le chemin existe vraiment)
    with torch.no_grad():
        model.predictor.w_film.weight.add_(0.05)
        diff = model.forecast(x, w=w2)["forecast"]
        still = model.forecast(x, w=torch.ones(3))["forecast"]
    assert not torch.allclose(base, diff)
    assert torch.allclose(base, still)          # w=1 reste le régime T2


def test_forecast_w_rejected_without_film():
    model = _model(cross_resolution=False).eval()
    with pytest.raises(ValueError):
        with torch.no_grad():
            model.forecast(torch.randn(2, 512, 1), w=torch.full((2,), 2.0))


def test_rolling_forecast_accepts_w():
    model = _model(cross_resolution=True).eval()
    with torch.no_grad():
        out = model.forecast(torch.randn(2, 512, 1), n=300,
                             w=torch.ones(2))    # 3 rolls, w répété par niveau
    assert out["forecast"].shape[1] == 300


# ---------------------------------------------------------------------------
# 3. L'ancre
# ---------------------------------------------------------------------------

def test_anchor_moves_encoder_in_full_finetune():
    m = _module(lambda_anchor=0.5)
    before = _snapshot(m)
    _step(m)
    assert m._last_anchor is not None and torch.isfinite(m._last_anchor)
    assert _moved(before, m, "online_encoder")
    assert _moved(before, m, "predictor")
    assert not _moved(before, m, "target_encoder")   # l'ancre est immobile


def test_anchor_refuses_linear_probe():
    with pytest.raises(ValueError):
        _module(finetune_mode="linear_probe", lambda_anchor=0.1)


def test_anchor_targets_loaded_online_not_random_copy(tmp_path):
    # Un « pretrain » sauvé, rechargé au finetune avec λ>0 : le target doit
    # être la copie de l'ONLINE CHARGÉ (piège n°1 — sinon deepcopy aléatoire).
    src = _model()
    ckpt = tmp_path / "pre.ckpt"
    torch.save({"state_dict": {f"model.{k}": v
                               for k, v in src.state_dict().items()}}, ckpt)
    m = _module(lambda_anchor=0.1, pretrained_encoder_path=str(ckpt))
    got = m.model.target_encoder.encoder.state_dict()
    want = m.model.online_encoder.state_dict()
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_anchor_zero_keeps_target_random_copy_semantics(tmp_path):
    # λ=0 : AUCUN comportement nouveau, pas même la copie du target.
    src = _model()
    ckpt = tmp_path / "pre.ckpt"
    torch.save({"state_dict": {f"model.{k}": v
                               for k, v in src.state_dict().items()}}, ckpt)
    m = _module(lambda_anchor=0.0, pretrained_encoder_path=str(ckpt))
    got = m.model.target_encoder.encoder.state_dict()
    want = m.model.online_encoder.state_dict()
    assert any(not torch.equal(got[k], want[k]) for k in want)
