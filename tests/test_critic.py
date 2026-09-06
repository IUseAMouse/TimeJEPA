"""
S6 critic primitives (2026-09-06): the shared energy / refinement helpers.

Pinned:
1. normalize_target_like_context equals the block inlined in the finetune.
2. Center refinement translates the fan: quantile differences unchanged
   (exactly), so the head's monotone structure survives.
3. Fan refinement returns a sorted fan.
4. A small step decreases the energy on a random batch.
5. refine_loop returns n+1 fans and energies.
6. Contextualized encoding is refused on a cross-resolution model.
7. With create_graph=True a refinement-step pinball reaches the input fan
   (the chain to the head); without it the step is a constant.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST  # noqa: E402
from timejepa.training import critic  # noqa: E402


def _model(cross_resolution=False, decoder_type="quantile"):
    torch.manual_seed(0)
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type=decoder_type,
                   cross_resolution=cross_resolution).eval()


def _forward(model, B=3):
    torch.manual_seed(1)
    x = torch.randn(B, 512, 1)
    res = model.forecast(x, return_representations=True)
    return x, res


def test_normalize_target_matches_finetune_block():
    model = _model()
    x, res = _forward(model)
    y = torch.randn(3, 128, 1)
    ref = y
    if model.robust_scaler is not None:
        ref = model.robust_scaler.transform(ref)
    ref = (ref - model.revin.mean) / model.revin.std
    assert torch.equal(critic.normalize_target_like_context(model, y), ref)


def test_context_norm_and_latents_are_returned():
    model = _model()
    x, res = _forward(model)
    assert res["context_norm"].shape == (3, 512, 1)
    assert res["future_representations"].shape[0] == 3
    z_pred, ctx_emb = critic.predict_latent(model, res["context_norm"])
    assert torch.allclose(z_pred, res["future_representations"], atol=1e-6)


def test_center_step_translates_fan_exactly():
    model = _model()
    x, res = _forward(model)
    fan = res["quantiles"].detach()
    z_pred = res["future_representations"].detach()
    mid = model.decoder.decoder.median_idx
    fan_next, e0, step = critic.refine_step(
        model, res["context_norm"].detach(), fan, z_pred, alpha=0.5,
        median_idx=mid)
    assert step.shape == (3, 128, 1) and e0.shape == (3,)
    # translation: quantile differences unchanged up to float rounding of
    # the added delta (the structure, not the bits, is what matters here)
    assert torch.allclose(fan_next - fan_next[..., mid:mid + 1],
                          fan - fan[..., mid:mid + 1], atol=1e-5)
    assert (step.abs().sum() > 0)
    assert (fan_next[..., 1:] >= fan_next[..., :-1]).all()


def test_fan_mode_is_sorted():
    model = _model()
    x, res = _forward(model)
    fan = res["quantiles"].detach()
    fan_next, e0, step = critic.refine_step(
        model, res["context_norm"].detach(), fan,
        res["future_representations"].detach(), alpha=0.5, target="fan",
        median_idx=model.decoder.decoder.median_idx)
    assert e0.shape == (3, 9) and step.shape == fan.shape
    assert (fan_next[..., 1:] >= fan_next[..., :-1]).all()


@pytest.mark.parametrize("mode", ["cos", "mse"])
def test_small_step_decreases_energy(mode):
    model = _model()
    x, res = _forward(model)
    fan, z = res["quantiles"].detach(), res["future_representations"].detach()
    ctx = res["context_norm"].detach()
    out = critic.refine_loop(model, ctx, fan, z, n_steps=3, alpha=0.05,
                             mode=mode, median_idx=model.decoder.decoder.median_idx)
    assert len(out["fans"]) == 4 and len(out["energies"]) == 4
    e = torch.stack(out["energies"])                  # [4, B]
    assert (e[-1] < e[0]).all()


def test_item_weight_zero_freezes_item():
    model = _model()
    x, res = _forward(model)
    fan, z = res["quantiles"].detach(), res["future_representations"].detach()
    w = torch.tensor([1.0, 0.0, 1.0])
    fan_next, _, step = critic.refine_step(
        model, res["context_norm"].detach(), fan, z, alpha=0.5,
        item_weight=w, median_idx=model.decoder.decoder.median_idx)
    assert torch.equal(step[1], torch.zeros_like(step[1]))
    assert step[0].abs().sum() > 0


def test_clip_bounds_the_step():
    model = _model()
    x, res = _forward(model)
    fan, z = res["quantiles"].detach(), res["future_representations"].detach()
    _, _, step = critic.refine_step(
        model, res["context_norm"].detach(), fan, z, alpha=50.0,
        max_abs_delta=0.1, median_idx=model.decoder.decoder.median_idx)
    assert step.abs().max() <= 0.1 + 1e-7


def test_contextualized_refused_on_xres():
    model = _model(cross_resolution=True)
    x, res = _forward(model)
    with pytest.raises(ValueError):
        critic.encode_candidate(model, res["context_norm"], res["quantiles"][..., 4:5],
                                contextualized=True)


def test_short_candidate_is_padded_to_one_patch():
    model = _model()
    x, res = _forward(model)
    y = res["quantiles"][:, :8, 4:5].detach()          # 8 steps < patch 16
    z = critic.encode_candidate(model, res["context_norm"], y)
    assert z.shape[1] == 1


def test_create_graph_chains_refinement_pinball_to_input_fan():
    model = _model()
    x, res = _forward(model)
    fan0 = res["quantiles"].detach().requires_grad_(True)
    z = res["future_representations"].detach()
    ctx = res["context_norm"].detach()
    mid = model.decoder.decoder.median_idx
    y = torch.randn(3, 128, 1)
    # with the graph: the step-2 pinball depends on fan0 through the descent
    out = critic.refine_loop(model, ctx, fan0, z, n_steps=2, alpha=0.1,
                             median_idx=mid, create_graph=True)
    loss = model.decoder.decoder.loss(out["fans"][-1], y)
    g = torch.autograd.grad(loss, fan0)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    # without the graph: fan2 = fan0 + constant, the gradient is the plain
    # pinball gradient (finite, but the descent adds nothing)
    fan0b = res["quantiles"].detach().requires_grad_(True)
    out_b = critic.refine_loop(model, ctx, fan0b, z, n_steps=2, alpha=0.1,
                               median_idx=mid, create_graph=False)
    loss_b = model.decoder.decoder.loss(out_b["fans"][-1], y)
    g_b = torch.autograd.grad(loss_b, fan0b)[0]
    fan_ref = out_b["fans"][-1].detach().requires_grad_(True)
    plain = torch.autograd.grad(model.decoder.decoder.loss(fan_ref, y), fan_ref)[0]
    assert torch.allclose(g_b, plain, atol=1e-6)


def test_step_norm_bounds_the_displacement():
    from timejepa.training import critic as C
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(2, 512, 1)
    ctx = C.normalize_target_like_context  # noqa: F841  (module import check)
    res = model.forecast(x, return_representations=True)
    fan0 = res["quantiles"].detach()
    z = res["future_representations"].detach()
    fan1, _, step = C.refine_step(model, res["context_norm"].detach(), fan0, z, alpha=0.2,
                                  median_idx=model.decoder.decoder.median_idx)
    assert step.abs().amax(dim=(1, 2)).allclose(torch.full((2,), 0.2), atol=1e-6)
    _, _, raw = C.refine_step(model, res["context_norm"].detach(), fan0, z, alpha=0.2,
                              median_idx=model.decoder.decoder.median_idx, step_norm=False)
    assert not torch.allclose(raw, step)
