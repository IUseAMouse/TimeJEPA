"""
Tests for TTA (evaluate_gift.tta_forecast): flip, translation shifts, and the
scale-invariance theorem.

Invariants, by severity:
1. No options = strictly model.forecast (no parasitic path).
2. THE THEOREM: f(kx) = k*f(x) EXACTLY under RobustScale+RevIN (median and
   MAD are 1-homogeneous => identical normalized input) - scale TTA is a
   proven no-op, only the SIGN (flip) carries information.
3. Shift alignment: on a ramp with a continuation oracle, the realigned
   shifted variants predict EXACTLY the base - the average is the identity.
   Any index offset would fail this test.
4. The coverage mask: tail positions (not covered by shifted variants) only
   average the variants that see them.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import tta_forecast                                   # noqa: E402
from timejepa.models import JEPATST                                      # noqa: E402


class _RampOracle:
    """Perfect continuation of a slope-1 ramp: f(ctx)[j] = ctx[-1]+j+1.
    With a true origin shift, variant s predicts t-s+j+1 - realignment must
    give back exactly the base."""
    patching = SimpleNamespace(stride=8, patch_size=16)

    def forecast(self, ctx, n):
        base = ctx[:, -1:]
        steps = torch.arange(1, n + 1, dtype=ctx.dtype).unsqueeze(0)
        med = (base + steps).unsqueeze(-1)                    # [B, n, 1]
        fan = torch.cat([med - 1.0, med, med + 1.0], dim=-1)  # [B, n, 3]
        return {"forecast_denorm": med, "quantiles_denorm": fan,
                "quantile_levels": (0.1, 0.5, 0.9)}


class _MarkerOracle:
    """Returns a constant identifying the variant by its context length - to
    verify the coverage mask position by position."""
    patching = SimpleNamespace(stride=8, patch_size=16)

    def __init__(self, full_len):
        self.full_len = full_len

    def forecast(self, ctx, n):
        val = 0.0 if ctx.shape[1] == self.full_len else 1.0
        med = torch.full((ctx.shape[0], n, 1), val)
        return {"forecast_denorm": med}


def test_no_options_is_passthrough():
    m = _RampOracle()
    batch = torch.arange(64, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=16)
    ref = m.forecast(batch, n=16)
    assert torch.equal(out["forecast_denorm"], ref["forecast_denorm"])


def test_scale_tta_is_a_provable_noop():
    """f(kx) = k*f(x) up to float precision: the normalized input is identical
    (median and MAD are 1-homogeneous), denormalization multiplies by k."""
    m = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                predictor_num_layers=1, predictor_num_heads=4,
                predictor_d_ff=64, decoder_type="quantile",
                robust_scale=True).eval()
    m.set_pretrain_mode(False)
    x = torch.randn(2, 384, 1) * 5 + 40
    with torch.no_grad():
        f_x = m.forecast(x, n=96)["quantiles_denorm"]
        f_3x = m.forecast(3.0 * x, n=96)["quantiles_denorm"]
    torch.testing.assert_close(f_3x, 3.0 * f_x, rtol=1e-4, atol=1e-3)


def test_shift_alignment_is_exact_on_a_ramp():
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)  # ramp
    base = m.forecast(batch, n=32)["forecast_denorm"]
    out = tta_forecast(m, batch, h=32, shifts=[2, 4, 6])
    # variant s: truncated by s on the right THEN realigned to the stride on
    # the left - on the ramp, the realigned oracle gives back exactly the base
    torch.testing.assert_close(out["forecast_denorm"], base)
    # the fan too (weighted average of identical vectors)
    fan = out["quantiles_denorm"]
    assert (fan[..., 1:] >= fan[..., :-1]).all()


def test_coverage_mask_counts_only_covering_variants():
    h, s = 16, 4
    m = _MarkerOracle(full_len=160)
    batch = torch.zeros(1, 160)
    out = tta_forecast(m, batch, h=h, shifts=[s])["forecast_denorm"].squeeze()
    # positions 0..h-s-1: mean(base=0, shift=1) = 0.5; tail: base only
    assert torch.allclose(out[:h - s], torch.full((h - s,), 0.5))
    assert torch.allclose(out[h - s:], torch.zeros(s))


def test_flip_combines_with_shifts():
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=32, flip=True, shifts=[2])
    assert torch.isfinite(out["forecast_denorm"]).all()
    fan = out["quantiles_denorm"]
    assert (fan[..., 1:] >= fan[..., :-1]).all()


def test_shift_larger_than_horizon_is_dropped():
    """m4_yearly has h=6: a shift >= h covers no position - it must be
    dropped, not break the alignment (bug measured 2026-08-25)."""
    m = _RampOracle()
    batch = torch.arange(160, dtype=torch.float32).unsqueeze(0)
    out = tta_forecast(m, batch, h=6, shifts=[7])
    base = m.forecast(batch, n=6)["forecast_denorm"]
    torch.testing.assert_close(out["forecast_denorm"], base)
