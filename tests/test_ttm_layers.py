"""TTM + inference layers (2026-09-06): the adapter exposes the harness API,
flip is exact on an odd proposer, and the layered point averages the rate
components on the native grid."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gift import _backtest_series_k, _mix_weights  # noqa: E402
from evaluate_gift_hybrid import TTMForecaster, ttm_layered_point  # noqa: E402


class _LastValueProposer:
    """Point proposer: repeats the last context value (odd in the sign)."""
    ctx_len, pred_len = 64, 16

    def paths(self, ctx, h, n_jitter, rng):
        return np.tile(np.float32(ctx[-1]), (1 + n_jitter, h))


class _DriftProposer:
    """Continues the last slope - the k=2 forecast differs from k=1."""
    ctx_len, pred_len = 64, 16

    def paths(self, ctx, h, n_jitter, rng):
        slope = ctx[-1] - ctx[-2]
        return np.tile(ctx[-1] + slope * np.arange(1, h + 1, dtype=np.float32),
                       (1 + n_jitter, 1))


def test_adapter_forecast_api_and_rollout():
    fc = TTMForecaster(_LastValueProposer(), flip=False, rng=None)
    x = torch.randn(3, 64, 1)
    out = fc.forecast(x, n=40)                    # 40 > pred_len: rollout
    assert out["quantiles_denorm"].shape == (3, 40, 9)
    assert out["forecast_denorm"].shape == (3, 40, 1)
    assert torch.allclose(out["quantiles_denorm"][:, :, 4], x[:, -1, 0:1].expand(3, 40))
    assert fc.input_length == 64 and fc.patching.patch_size == 16


def test_flip_is_identity_on_an_odd_proposer():
    ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    a = TTMForecaster(_LastValueProposer(), flip=False, rng=None).point(ctx, 8)
    b = TTMForecaster(_LastValueProposer(), flip=True, rng=None).point(ctx, 8)
    assert np.allclose(a, b)


def test_layered_point_averages_rate_components():
    fc = TTMForecaster(_DriftProposer(), flip=False, rng=None)
    ctx = np.arange(200, dtype=np.float32) ** 1.5          # convex: slope grows
    h = 12
    p1 = ttm_layered_point(fc, ctx, h, [(1, 1.0)])
    p2 = ttm_layered_point(fc, ctx, h, [(2, 1.0)])
    mix = ttm_layered_point(fc, ctx, h, [(1, 0.25), (2, 0.75)])
    assert p1.shape == (h,) and p2.shape == (h,)
    assert not np.allclose(p1, p2)                          # k=2 sees a coarser slope
    assert np.allclose(mix, 0.25 * p1 + 0.75 * p2)
    # a component starved of history is dropped, weights renormalized
    short = ctx[-20:]
    assert np.allclose(ttm_layered_point(fc, short, h, [(1, 0.5), (48, 0.5)]),
                       ttm_layered_point(fc, short, h, [(1, 1.0)]))


def test_backtest_and_mix_run_on_the_adapter():
    rng = np.random.default_rng(1)
    t = np.arange(3000, dtype=np.float32)
    series = [(np.sin(2 * np.pi * t / 200) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
              for _ in range(4)]
    fc = TTMForecaster(_DriftProposer(), flip=True, rng=rng)
    ks, diag = _backtest_series_k(fc, series, h=24, windows=1, max_len=64, stride=8,
                                  patch=16, device=torch.device("cpu"), batch_size=4,
                                  pooled=True)
    assert diag["pooling"] == "crps" and diag["n_base"] == 4
    w = _mix_weights(diag["ratios"])
    assert abs(sum(w.values()) - 1.0) < 1e-9 and all(k >= 1 for k in w)
