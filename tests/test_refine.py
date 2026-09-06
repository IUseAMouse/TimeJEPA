"""
S6 inference-time refinement (2026-09-06): the harness adapter.

Pinned:
1. Flag absent / inert spec: the same object comes back.
2. alpha=0 or n_max=0: identity, no steps.
3. Center mode keeps the fan shape (quantile differences) and order.
4. The energy trace never increases; eps=1 stops everything after one step.
5. Ceiling mode lowers the pinball against the true target; NaN targets are
   masked and never poison the output.
6. Fan mode returns a sorted fan.
7. The denormalize round trip is close to the input (alpha=0) for 3-dim and
   4-dim fans; forecast_denorm is the median column.
8. parse_refine_flags: unknown values raise, tags are distinct per mode.
9. decimated_target: length h' and NaN padding.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation import refine as R  # noqa: E402
from timejepa.models import JEPATST  # noqa: E402
from evaluate_gift import parse_refine_flags, tta_forecast  # noqa: E402


def _model():
    torch.manual_seed(0)
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="quantile").eval()


def _batch(B=3):
    torch.manual_seed(1)
    return torch.randn(B, 512, 1) * 3 + 10


def _out(model, x, h=64):
    with torch.no_grad():
        return tta_forecast(model, x, h)


LEVELS = tuple(0.1 * j for j in range(1, 10))


def test_inert_returns_same_object():
    model, x = _model(), _batch()
    out = _out(model, x)
    res, st = R.refine_out(model, x, out, None)
    assert res is out and st is None
    res, st = R.refine_out(model, x, out, R.RefineSpec(mode="off"))
    assert res is out
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS, R.RefineSpec(mode="energy", alpha=0.0))
    assert fan is fan_norm and st == {}
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS, R.RefineSpec(mode="energy", n_max=0))
    assert fan is fan_norm


def test_center_mode_preserves_shape_and_order():
    model, x = _model(), _batch()
    out = _out(model, x)
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="energy", alpha=0.5, n_max=4, eps=0.0))
    mid = 4
    assert torch.allclose(fan - fan[..., mid:mid + 1], fan_norm - fan_norm[..., mid:mid + 1], atol=1e-5)
    assert (fan[..., 1:] >= fan[..., :-1]).all()
    assert st["abs_delta"].sum() > 0 and st["judge_cover"] == 1.0
    assert st["steps"].max() >= 1


def test_energy_trace_never_increases_and_eps_one_stops():
    model, x = _model(), _batch()
    out = _out(model, x)
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="energy", alpha=0.3, n_max=6, eps=0.0))
    tr = st["E_trace"]                                   # [n+1, B]
    assert (tr[1:] <= tr[:-1] + 1e-6).all()
    assert (st["dE"] >= -1e-6).all()
    fan2, st2 = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                             R.RefineSpec(mode="energy", alpha=0.3, n_max=6, eps=1.0))
    assert (st2["steps"] <= 1).all() and st2["stopped_early"].all()


def test_ceiling_lowers_pinball_and_masks_nan():
    model, x = _model(), _batch()
    out = _out(model, x)
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    target = fan_norm[..., 4:5] + 1.0
    target[1, 10:20] = float("nan")
    before = R.pinball_fn(fan_norm, target, LEVELS)
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="ceiling", alpha=0.3, n_max=8, eps=0.0),
                           target_norm=target)
    after = R.pinball_fn(fan, target, LEVELS)
    assert (after < before).all() and torch.isfinite(fan).all()
    with pytest.raises(ValueError):
        R.refine_fan(model, ctx_norm, fan_norm, LEVELS, R.RefineSpec(mode="ceiling"))


def test_fan_mode_sorted():
    model, x = _model(), _batch()
    out = _out(model, x)
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="energy", target="fan", alpha=0.3, n_max=3, eps=0.0))
    assert (fan[..., 1:] >= fan[..., :-1]).all()


@pytest.mark.parametrize("four_dim", [False, True])
def test_refine_out_roundtrip_and_median(four_dim):
    model, x = _model(), _batch()
    out = _out(model, x)
    if four_dim:
        out = dict(out); out["quantiles_denorm"] = out["quantiles_denorm"].unsqueeze(-1)
    # alpha tiny: numerically an identity round trip through the scalers
    res, st = R.refine_out(model, x, out, R.RefineSpec(mode="energy", alpha=1e-12, n_max=1))
    q_in, q_out = out["quantiles_denorm"], res["quantiles_denorm"]
    assert q_out.ndim == q_in.ndim
    assert torch.allclose(q_in, q_out, rtol=1e-4, atol=1e-4)
    med = q_out[..., 0] if four_dim else q_out
    assert torch.allclose(res["forecast_denorm"], med[..., 4:5], atol=1e-6)
    # a real step moves the median and the fan together
    res2, st2 = R.refine_out(model, x, out, R.RefineSpec(mode="energy", alpha=0.5, n_max=3, eps=0.0))
    assert st2["abs_delta"].sum() > 0
    assert not torch.allclose(res2["forecast_denorm"], res["forecast_denorm"])


def test_horizon_beyond_judge_span_is_untouched():
    model, x = _model(), _batch()
    out = _out(model, x, h=200)                         # 200 > prediction_length 128
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="energy", alpha=0.5, n_max=2, eps=0.0))
    assert st["judge_cover"] == 128 / 200
    assert torch.equal(fan[:, 128:], fan_norm[:, 128:])


def test_short_horizon_below_patch_runs():
    model, x = _model(), _batch()
    out = _out(model, x, h=8)
    ctx_norm, fan_norm = R.normalize_with_context(model, x, out["quantiles_denorm"])
    fan, st = R.refine_fan(model, ctx_norm, fan_norm, LEVELS,
                           R.RefineSpec(mode="energy", alpha=0.3, n_max=2, eps=0.0))
    assert fan.shape == fan_norm.shape and torch.isfinite(fan).all()


def test_parse_refine_flags_and_tags():
    assert parse_refine_flags({}) == (None, "self", "")
    spec, jk, tag = parse_refine_flags({"refine": "energy"})
    assert spec.mode == "energy" and tag == "_refine-E8" and jk == "self"
    spec, jk, tag = parse_refine_flags({"refine": "ceiling", "refine_alpha": 0.3})
    assert tag == "_refine-ceiling8-a0.3" and not spec.mode == "energy"
    _, _, tag = parse_refine_flags({"refine": "energy", "refine_target": "fan", "refine_steps": 4})
    assert tag == "_refine-E4-fan"
    with pytest.raises(ValueError):
        parse_refine_flags({"refine": "foo"})
    with pytest.raises(ValueError):
        parse_refine_flags({"refine": "energy", "refine_judge": "ckpt"})
    with pytest.raises(ValueError):
        parse_refine_flags({"refine": "ceiling", "refine_judge": "ckpt", "energy_ckpt": "x"})
    with pytest.raises(ValueError):
        parse_refine_flags({"refine": "energy", "refine_target": "nope"})


def test_summarize_and_decimated_target():
    model, x = _model(), _batch()
    out = _out(model, x)
    _, st = R.refine_out(model, x, out, R.RefineSpec(mode="energy", alpha=0.3, n_max=2, eps=0.0))
    summ = R.summarize_refine([st, st], R.RefineSpec(mode="energy", alpha=0.3, n_max=2), "self")
    for k in ("mean_steps", "frac_stopped_early", "mean_dE", "mean_E0", "mean_abs_delta",
              "n_refined", "official", "judge_cover"):
        assert k in summ
    assert summ["n_refined"] == 6 and summ["official"] is True
    t = np.arange(20, dtype=np.float32)
    d = R.decimated_target(t, k=2, h_prime=16)
    assert d.shape == (16,) and np.isfinite(d[:10]).all() and np.isnan(d[10:]).all()
    assert np.array_equal(R.decimated_target(t, 1, 20), t)
