"""
S6-b denoising score matching on the energy (2026-09-08).

Pinned:
1. perturb_target: kinds behave as documented, forecast needs the median,
   zero weight never drawn, deterministic under a seeded generator.
2. score_cos: finite in [-1, 1]; gradient of 1 - cos reaches the online
   encoder; route B reaches the predictor, route A does not.
3. LEARNABILITY: 150 AdamW steps on the score term alone dig a local valley
   at the truth on a tiny model (valley_witness 0 -> >= 0.7 at delta 0.3).
4. Module: lambda_score = 0 is bit-identical to the joint arm; padded items
   are excluded; the validation path (no_grad) logs val_loss/score and
   val_score/valley_frac deterministically.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST  # noqa: E402
from timejepa.training import critic as C  # noqa: E402
from timejepa.training.finetune_module import FinetuneModule  # noqa: E402


def _model(seed=0):
    torch.manual_seed(seed)
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="quantile")


def _module(seed=0, **kw):
    kw.setdefault("finetune_mode", "full_finetune")
    kw.setdefault("lambda_joint", 0.3)
    torch.manual_seed(seed)
    m = FinetuneModule(model=_model(seed), **kw)
    m.model.eval()
    return m


def _latents(model, x):
    res = model.forecast(x, return_representations=True)
    return res["context_norm"], res["future_representations"], res


# --------------------------------------------------------------------- 1
def test_perturb_kinds():
    torch.manual_seed(0)
    y = torch.randn(6, 128, 1)
    med = y + 0.7
    gen = torch.Generator().manual_seed(1)
    for kind in ("level", "noise", "slope", "forecast"):
        yt, idx = C.perturb_target(y, [kind], [1.0], gen, median=med)
        assert yt.shape == y.shape and (idx == 0).all()
        eps = yt - y
        if kind == "level":
            assert torch.allclose(eps, eps[:, :1], atol=1e-6)
            assert (eps.abs().flatten(1)[:, 0] >= 0.05 - 1e-6).all() and (eps.abs() <= 0.5 + 1e-6).all()
        elif kind == "slope":
            assert torch.allclose(eps[:, 0], torch.zeros_like(eps[:, 0]), atol=1e-6)
            d = eps[:, 1:] - eps[:, :-1]
            assert torch.allclose(d, d[:, :1], atol=1e-5)
        elif kind == "noise":
            assert eps.abs().mean() > 0 and abs(float(eps.mean())) < 0.05
        else:
            assert torch.equal(yt, med)
    with pytest.raises(ValueError):
        C.perturb_target(y, ["forecast"], [1.0], gen)
    with pytest.raises(ValueError):
        C.perturb_target(y, ["nope"], [1.0], gen)
    # zero weight never drawn, deterministic under the same seed
    a, ia = C.perturb_target(y, ["level", "noise"], [1.0, 0.0], torch.Generator().manual_seed(3))
    b, ib = C.perturb_target(y, ["level", "noise"], [1.0, 0.0], torch.Generator().manual_seed(3))
    assert (ia == 0).all() and torch.equal(a, b)


# --------------------------------------------------------------------- 2
def test_score_cos_finite_and_gradient_routes():
    model = _model()
    torch.manual_seed(1)
    x = torch.randn(4, 512, 1) * 2 + 1
    y = torch.randn(4, 128, 1)
    model.train()
    for route, expect in (("A", False), ("B", True)):
        ctx, z, _ = _latents(model, x)
        _, y_norm = __import__("timejepa.evaluation.refine", fromlist=["x"]).normalize_with_context(model, x, y)
        yt, _ = C.perturb_target(y_norm, ["level", "noise"], [0.5, 0.5], torch.Generator().manual_seed(0))
        z_e = z.detach() if route == "A" else z
        cos = C.score_cos(model, ctx.detach(), y_norm, yt, z_e, create_graph=True)
        assert cos.shape == (4,) and torch.isfinite(cos).all()
        assert (cos.abs() <= 1 + 1e-5).all()
        model.zero_grad(set_to_none=True)
        (1 - cos).mean().backward()
        enc_has = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.online_encoder.parameters())
        pred_has = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.predictor.parameters())
        assert enc_has and pred_has == expect, route


# --------------------------------------------------------------------- 3
def test_learnability_digs_a_valley():
    """The decisive test: the score term alone creates a local valley at the
    truth along level shifts on a toy model, within 150 steps."""
    torch.manual_seed(0)
    model = _model()
    B, L, h = 8, 512, 128
    t = torch.arange(L + h).float()
    freqs = torch.rand(B, 1) * 0.2 + 0.05
    phase = torch.rand(B, 1) * 6.28
    series = torch.sin(freqs * t.view(1, -1) + phase) + 0.1 * torch.randn(B, L + h)
    x, y = series[:, :L].unsqueeze(-1), series[:, L:].unsqueeze(-1)
    from timejepa.evaluation.refine import normalize_with_context
    opt = torch.optim.AdamW([p for n, p in model.named_parameters()
                             if not n.startswith("target_encoder") and "decoder" not in n], lr=2e-3)

    def witness():
        # delta 0.3: the middle of the trained level range [0.05, 0.5]; at
        # 0.1 the witness is one item out of eight and flickers
        model.eval()
        with torch.no_grad():
            ctx, z, _ = _latents(model, x)
            _, y_norm = normalize_with_context(model, x, y)
        return float(C.valley_witness(model, ctx, y_norm, z, delta=0.3))

    before = witness()
    gen = torch.Generator().manual_seed(0)
    last_cos = 0.0
    for step in range(150):
        model.train()
        ctx, z, _ = _latents(model, x)
        _, y_norm = normalize_with_context(model, x, y)
        yt, _ = C.perturb_target(y_norm, ["level", "noise"], [0.7, 0.3], gen)
        cos = C.score_cos(model, ctx, y_norm, yt, z, create_graph=True)
        loss = (1 - cos).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_cos = float(cos.detach().mean())
    after = witness()
    assert after >= 0.7, (before, after, last_cos)
    assert last_cos > 0.3, (before, after, last_cos)
    assert before < after


# --------------------------------------------------------------------- 4
def test_module_defaults_bit_identical_and_padded_excluded():
    m0 = _module()
    m1 = _module(lambda_score=0.0)
    m0.eval(); m1.eval()
    x, y = torch.randn(2, 512, 1), torch.randn(2, 128, 1)
    with torch.no_grad():
        l0, r0, _ = m0._forward_and_loss(x, y)
        l1, r1, _ = m1._forward_and_loss(x, y)
    assert torch.equal(l0, l1) and torch.equal(r0["quantiles"], r1["quantiles"])
    assert m1._score_stats == {}
    # padded items excluded from the term: a mask that empties the sub-batch
    m2 = _module(lambda_score=0.3, score_batch_fraction=1.0,
                 score_perturb={"level": 1.0})
    m2.model.train()
    x, y = torch.randn(4, 512, 1), torch.randn(4, 128, 1)
    mask = torch.ones(4, 128, dtype=torch.bool); mask[:, -10:] = False
    loss, _, _ = m2._forward_and_loss(x, y, target_mask=mask)
    assert torch.isfinite(loss) and m2._score_stats.get("n_items", 0) == 0
    mask2 = torch.ones(4, 128, dtype=torch.bool); mask2[2:, -10:] = False
    loss2, _, _ = m2._forward_and_loss(x, y, target_mask=mask2)
    assert m2._score_stats["n_items"] == 2 and torch.isfinite(loss2)
    loss2.backward()


def test_module_train_and_val_paths():
    m = _module(lambda_score=0.3, score_route="B", score_batch_fraction=0.5,
                score_perturb={"level": 0.35, "noise": 0.25, "slope": 0.15, "forecast": 0.25})
    m.model.train()
    x, y = torch.randn(4, 512, 1), torch.randn(4, 128, 1)
    loss, _, _ = m._forward_and_loss(x, y)
    st = m._score_stats
    assert st["n_items"] == 2 and "cos" in st and torch.isfinite(loss)
    loss.backward()
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value)
    m.training_step({"context": x, "target": y}, batch_idx=0)
    assert "train_loss/score" in logged and "score/cos" in logged
    # validation: no_grad, deterministic, valley witness present
    m.eval()
    with torch.no_grad():
        l1, _, _ = m._forward_and_loss(x, y)
        s1 = dict(m._score_stats)
        l2, _, _ = m._forward_and_loss(x, y)
        s2 = dict(m._score_stats)
    assert torch.allclose(l1, l2, atol=1e-6) and abs(s1["cos"] - s2["cos"]) < 1e-6
    assert 0.0 <= s1["valley_frac"] <= 1.0
    logged.clear()
    m.validation_step({"context": x, "target": y}, batch_idx=0)
    assert "val_loss/score" in logged and "val_score/valley_frac" in logged


def test_refusals():
    with pytest.raises(ValueError):
        _module(lambda_score=0.3, finetune_mode="linear_probe")
    with pytest.raises(ValueError):
        _module(lambda_score=0.3, score_route="C")
    with pytest.raises(ValueError):
        _module(lambda_score=0.3, score_perturb={"bogus": 1.0})
