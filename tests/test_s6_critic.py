"""
S6 critic loop at finetune (2026-09-06).

Pinned:
1. critic_steps=[0] equals the plain (joint) finetune; the eval path is
   bit-identical when the loop is off.
2. The per-step pinballs and witnesses are logged (critic/pinball_i ...).
3. Route A gives the predictor NO gradient from the refinement pinball;
   route B does (the only path is z_pred inside the energy).
4. Refusals: linear_probe, no joint, point head.
5. xres w path works with the loop (the FiLM receives a gradient).
6. Eval uses N = max deterministically; the loss is the pinball of the
   refined fan and results carry the refined fan.
7. critic_batch_fraction restricts the loop to a sub-batch.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST  # noqa: E402
from timejepa.training import critic  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule  # noqa: E402


def _model(cross_resolution=False, decoder_type="quantile"):
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type=decoder_type,
                   cross_resolution=cross_resolution)


def _module(seed=0, **kw):
    kw.setdefault("finetune_mode", "full_finetune")
    kw.setdefault("lambda_joint", 0.3)
    torch.manual_seed(seed)
    model = _model(kw.pop("cross_resolution", False), kw.pop("decoder_type", "quantile"))
    m = FinetuneModule(model=model, **kw)
    m.model.eval()
    return m


def test_zero_steps_equals_joint_only():
    m0 = _module()
    m1 = _module(critic_steps=[0], critic_alpha=0.1)
    m0.eval(); m1.eval()
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l0, r0, _ = m0._forward_and_loss(x, y)
        l1, r1, _ = m1._forward_and_loss(x, y)
    assert torch.equal(l0, l1) and torch.equal(r0["quantiles"], r1["quantiles"])
    assert m1._critic_stats == {}


def test_train_loop_adds_step_pinballs_and_witnesses():
    m = _module(critic_steps=[3], critic_alpha=0.1)
    m.model.train()
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    loss, results, _ = m._forward_and_loss(x, y)
    st = m._critic_stats
    assert st["n_steps"] == 3.0
    for k in ("energy_0", "energy_N", "energy_drop", "pinball_0", "pinball_1",
              "pinball_2", "pinball_3", "pinball_N", "delta_abs", "delta_clipped_frac"):
        assert k in st and st[k] == st[k]          # present, not NaN
    assert st["delta_abs"] > 0 and torch.isfinite(loss)
    loss.backward()                                 # one backward through 3 steps
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value)
    m.training_step({"context": x, "target": y}, batch_idx=0)
    assert "critic/pinball_3" in logged and "train_loss/joint" in logged


def test_route_a_no_predictor_grad_route_b_has():
    x = torch.randn(2, 512, 1)
    y = torch.randn(2, 128, 1)
    for route, expect in (("A", False), ("B", True)):
        m = _module(critic_steps=[2], critic_alpha=0.1, critic_route=route)
        m.model.train()
        res = m.model.forecast(x, return_representations=True)
        head = m.model.decoder.decoder
        # fan0 cut from the graph: the only way from a refinement pinball to
        # the predictor weights is z_pred inside the energy
        fan0 = res["quantiles"].detach().requires_grad_(True)
        z = res["future_representations"]
        z_for_E = z.detach() if route == "A" else z
        out = critic.refine_loop(m.model, res["context_norm"].detach(), fan0, z_for_E,
                                 2, alpha=0.1, create_graph=True, median_idx=head.median_idx)
        tgt = critic.normalize_target_like_context(m.model, y)
        pb = head.loss(out["fans"][-1], tgt)
        m.model.zero_grad(set_to_none=True)
        pb.backward()
        has = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in m.model.predictor.parameters())
        assert has == expect, route
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.model.online_encoder.parameters())


def test_refusals():
    with pytest.raises(ValueError):
        _module(critic_steps=[1], critic_alpha=0.1, finetune_mode="linear_probe")
    with pytest.raises(ValueError):
        _module(lambda_joint=0.0, critic_steps=[1], critic_alpha=0.1)
    with pytest.raises(ValueError):
        _module(decoder_type="mlp", critic_steps=[1], critic_alpha=0.1)
    with pytest.raises(ValueError):
        _module(critic_steps=[1], critic_alpha=0.0)
    with pytest.raises(ValueError):
        _module(critic_steps=[1], critic_alpha=0.1, critic_route="C")


def test_xres_w_path_with_loop():
    m = _module(critic_steps=[1], critic_alpha=0.1, cross_resolution=True)
    m.model.train()
    x, y = torch.randn(4, 512, 1), torch.randn(4, 128, 1)
    w = torch.tensor([1.0, 0.5, 2.0, 1.0])
    loss, _, _ = m._forward_and_loss(x, y, w=w)
    assert torch.isfinite(loss)
    loss.backward()
    g = m.model.predictor.w_film.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_eval_uses_n_max_deterministically():
    m = _module(critic_steps=[0, 1, 2], critic_alpha=0.1)
    m.eval()                 # the LightningModule flag, as the Trainer sets it in validation
    head = m.model.decoder.decoder
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l1, r1, tgt = m._forward_and_loss(x, y)
        l2, r2, _ = m._forward_and_loss(x, y)
    # deterministic up to the non-associative CPU reductions of the backward
    # (the descent is a gradient): same N, same fan, same loss to 1e-6
    assert torch.allclose(l1, l2, atol=1e-6) and m._critic_stats["n_steps"] == 2.0
    assert torch.allclose(l1, head.loss(r1["quantiles"], tgt))
    assert torch.allclose(r1["quantiles"], r2["quantiles"], atol=1e-6)
    assert r1["forecast"].shape == (2, 128, 1) and "quantiles_denorm" in r1


def test_batch_fraction_restricts_loop():
    m = _module(critic_steps=[1], critic_alpha=0.1, critic_batch_fraction=0.5)
    m.model.train()
    x, y = torch.randn(4, 512, 1), torch.randn(4, 128, 1)
    loss, _, _ = m._forward_and_loss(x, y)
    assert torch.isfinite(loss) and m._critic_stats["n_steps"] == 1.0


def test_clip_witness_fires_at_huge_alpha():
    m = _module(critic_steps=[1], critic_alpha=1e6, critic_max_abs_delta=0.01)
    m.model.train()
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    loss, _, _ = m._forward_and_loss(x, y)
    assert m._critic_stats["delta_clipped_frac"] > 0 and torch.isfinite(loss)
