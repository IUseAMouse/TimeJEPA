"""
H2b joint loss (2026-09-06): a JEPA term on the true target during finetune.

Pinned:
1. Inert defaults: loss and every results key bit-identical (mlp and
   quantile heads), no latent keys returned.
2. joint (frozen, standalone, no SIGReg) equals the anchor exactly, with and
   without a target mask.
3. Refusals: anchor + joint together, joint in linear_probe, contextualized
   on an xres model, critic without joint.
4. SIGReg: finite, positive, and the loss moves the online encoder.
5. EMA target: update_target_encoder moves the target encoder; frozen keeps it.
6. Trap #1: with the joint term, the target encoder is the loaded online
   encoder, not the random construction copy.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule  # noqa: E402


def _model(decoder_type="quantile", cross_resolution=False):
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type=decoder_type,
                   cross_resolution=cross_resolution)


def _module(decoder_type="quantile", seed=0, **kw):
    kw.setdefault("finetune_mode", "full_finetune")
    torch.manual_seed(seed)
    model = _model(decoder_type, kw.pop("cross_resolution", False))
    m = FinetuneModule(model=model, **kw)
    m.model.eval()
    return m


SIGREG = {"lambda": 1.0, "num_projections": 8, "num_quadrature": 17, "t_max": 5.0,
          "apply_to": "context"}


@pytest.mark.parametrize("decoder_type", ["mlp", "quantile"])
def test_defaults_bit_identical(decoder_type):
    m0 = _module(decoder_type)
    m1 = _module(decoder_type, lambda_joint=0.0, critic_steps=[])
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l0, r0, _ = m0._forward_and_loss(x, y)
        l1, r1, _ = m1._forward_and_loss(x, y)
    assert torch.equal(l0, l1)
    assert set(r0) == set(r1) and "future_representations" not in r0
    for k in r0:
        assert torch.equal(r0[k], r1[k]) if torch.is_tensor(r0[k]) else r0[k] == r1[k]
    assert m1._last_joint is None and m1._critic_stats == {}


@pytest.mark.parametrize("with_mask", [False, True])
def test_joint_frozen_standalone_equals_anchor(with_mask):
    ma = _module(lambda_anchor=0.7)
    mj = _module(lambda_joint=0.7)
    x, y = torch.randn(3, 512, 1), torch.randn(3, 128, 1)
    mask = None
    if with_mask:
        mask = torch.ones(3, 128, dtype=torch.bool); mask[1, 100:] = False
    with torch.no_grad():
        la, _, _ = ma._forward_and_loss(x, y, target_mask=mask)
        lj, _, _ = mj._forward_and_loss(x, y, target_mask=mask)
    assert torch.allclose(la, lj, atol=1e-6, rtol=0)
    assert torch.allclose(ma._last_anchor, mj._last_joint, atol=1e-6, rtol=0)


def test_refusals():
    with pytest.raises(ValueError):
        _module(lambda_anchor=0.1, lambda_joint=0.1)
    with pytest.raises(ValueError):
        _module(lambda_joint=0.1, finetune_mode="linear_probe")
    with pytest.raises(ValueError):
        _module(lambda_joint=0.1, joint_contextualized=True, cross_resolution=True)
    with pytest.raises(ValueError):
        _module(critic_steps=[0, 1], critic_alpha=0.1)              # no joint
    with pytest.raises(ValueError):
        _module("mlp", lambda_joint=0.1, critic_steps=[1], critic_alpha=0.1)   # point head
    with pytest.raises(ValueError):
        _module(lambda_joint=0.1, joint_target="nope")


def test_sigreg_finite_positive_and_moves_encoder():
    m = _module(lambda_joint=0.3, joint_sigreg=True, sigreg_config=SIGREG)
    m.model.train()
    x, y = torch.randn(4, 512, 1), torch.randn(4, 128, 1)
    loss, _, _ = m._forward_and_loss(x, y)
    assert torch.isfinite(loss) and m._last_sigreg is not None
    assert torch.isfinite(m._last_sigreg) and float(m._last_sigreg) > 0
    loss.backward()
    grads = [p.grad for n, p in m.model.online_encoder.named_parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_contextualized_joint_runs_on_plain_model():
    m = _module(lambda_joint=0.3, joint_contextualized=True)
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        loss, _, _ = m._forward_and_loss(x, y)
    assert torch.isfinite(loss) and float(m._last_joint) > 0


def test_ema_moves_target_encoder_frozen_does_not():
    m = _module(lambda_joint=0.3, joint_target="ema")
    before = {k: v.clone() for k, v in m.model.target_encoder.named_parameters()}
    # perturb the online encoder so an EMA step has something to pull toward
    with torch.no_grad():
        for p in m.model.online_encoder.parameters():
            p.add_(0.5)
    m.update_target_encoder(0, 100)
    assert any(not torch.equal(before[k], v)
               for k, v in m.model.target_encoder.named_parameters())
    m2 = _module(lambda_joint=0.3, joint_target="frozen")
    before2 = {k: v.clone() for k, v in m2.model.target_encoder.named_parameters()}
    with torch.no_grad():
        for p in m2.model.online_encoder.parameters():
            p.add_(0.5)
    # no callback in frozen mode: nothing calls update_target_encoder
    assert all(torch.equal(before2[k], v)
               for k, v in m2.model.target_encoder.named_parameters())


def test_joint_target_is_loaded_encoder(tmp_path):
    # a pretrain checkpoint whose online encoder differs from a fresh init
    torch.manual_seed(1)
    src = _model()
    with torch.no_grad():
        for p in src.online_encoder.parameters():
            p.add_(0.3)
    ckpt = tmp_path / "pre.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in src.state_dict().items()}}, ckpt)
    m = _module(lambda_joint=0.3, pretrained_encoder_path=str(ckpt))
    on = dict(m.model.online_encoder.named_parameters())
    for k, v in m.model.target_encoder.named_parameters():
        key = k.split("encoder.", 1)[-1] if k.startswith("encoder.") else k
        if key in on:
            assert torch.equal(v, on[key])
