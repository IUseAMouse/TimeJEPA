"""
RateIN as a layer (`timejepa.ratein_layer`).

1. Parity with the numpy primitives of `evaluation.ratein` at integer rates:
   resample_context == decimate (k > 1) and == upsample (k = 1/m);
   fan_to_native == reinterp_fan (k > 1) and == pool_fan (k = 1/m); k = 1 is
   the identity on both sides.
2. Left padding: a padded batch gives the same blocks as the unpadded series.
3. Gradients reach k through the resampling and through the return to the
   native grid; the selector starts on the prior exactly (g = 0).
4. The layer around a stub forecaster: shapes, the model horizon with and
   without upsampling, the bound on k from the context length.
5. pinball_wql reproduces the harness's CRPS on one instance; the grid-mode
   interpolation hits the grid values and interpolates between them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import gift, ratein as R  # noqa: E402
from timejepa import ratein_layer as RL  # noqa: E402

torch.manual_seed(0)
LEVELS = list(gift.QUANTILE_LEVELS)


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (10 + np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(n)).astype(np.float64)


# ---------------------------------------------------------------- 1. parity
@pytest.mark.parametrize("k", [2, 3, 4, 6, 8])
def test_resample_matches_decimate(k):
    x = _series(5000)
    out_len = 128
    ref = R.resample_context(x, k, out_len)                   # decimate(x[-(out_len*k):], k)
    y, valid = RL.resample_context(torch.tensor(x)[None], torch.tensor([float(k)]), out_len)
    assert valid.all()
    np.testing.assert_allclose(y[0].numpy(), ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_resample_matches_upsample(m):
    # The numpy path truncates to out_len // m native points first and
    # extrapolates its left edge from them; the layer sees the whole series.
    # Parity is exact on a series of exactly that length; on a longer one the
    # layer's first fine point uses the real previous sample instead of an
    # extrapolation and only that point may differ.
    out_len = 240
    x = _series(out_len // m)
    ref = R.resample_context(x, 1.0 / m, out_len)             # upsample(x[-(out_len//m):], m)
    k = torch.tensor([1.0 / m], dtype=torch.float64)
    y, valid = RL.resample_context(torch.tensor(x)[None], k, out_len)
    assert valid.all()
    np.testing.assert_allclose(y[0].numpy(), ref, rtol=0, atol=1e-9)
    x_long = _series(700)
    ref = R.resample_context(x_long, 1.0 / m, out_len)
    y, _ = RL.resample_context(torch.tensor(x_long)[None], k, out_len)
    skip = m // 2                                               # the extrapolated half-block
    np.testing.assert_allclose(y[0].numpy()[skip:], ref[skip:], rtol=0, atol=1e-9)


def test_resample_identity_at_k1():
    x = _series(2000)
    y, valid = RL.resample_context(torch.tensor(x)[None], torch.tensor([1.0]), 1024)
    np.testing.assert_allclose(y[0].numpy(), x[-1024:], atol=0)
    assert valid.all()


@pytest.mark.parametrize("k", [2, 3, 5])
def test_fan_matches_reinterp(k):
    h = 48
    h_dec = R.fc_horizon(h, k)
    fan = np.sort(np.random.default_rng(1).standard_normal((h_dec, 9)), axis=1)
    ref = R.to_native_fan(fan, h, k)
    out = RL.fan_to_native(torch.tensor(fan)[None], torch.tensor([float(k)]), h)
    np.testing.assert_allclose(out[0].numpy(), ref, atol=1e-12)
    assert np.all(np.diff(out[0].numpy(), axis=1) >= -1e-12)   # monotone levels survive


@pytest.mark.parametrize("m", [2, 3, 4])
def test_fan_matches_pool(m):
    h = 12
    fan = np.sort(np.random.default_rng(2).standard_normal((h * m, 9)), axis=1)
    ref = R.to_native_fan(fan, h, 1.0 / m)
    out = RL.fan_to_native(torch.tensor(fan)[None], torch.tensor([1.0 / m], dtype=torch.float64), h)
    np.testing.assert_allclose(out[0].numpy(), ref, atol=1e-12)


def test_fan_identity_at_k1():
    fan = torch.randn(2, 20, 9).sort(-1).values
    out = RL.fan_to_native(fan, torch.ones(2), 16)
    assert torch.equal(out, fan[:, :16])


def test_fan_refuses_short_model_horizon():
    with pytest.raises(ValueError):
        RL.fan_to_native(torch.zeros(1, 10, 9), torch.tensor([0.5]), 12)


# ----------------------------------------------------------- 2. left padding
def test_left_padding_matches_unpadded():
    x = _series(900)
    k = torch.tensor([2.5, 0.5])
    padded = torch.full((2, 1200), float("nan"), dtype=torch.float64)
    padded[:, -900:] = torch.tensor(x)
    y_pad, v_pad = RL.resample_context(padded, k, 256, lengths=torch.tensor([900, 900]))
    y_raw, v_raw = RL.resample_context(torch.tensor(x)[None].repeat(2, 1), k, 256)
    assert torch.equal(v_pad, v_raw)
    torch.testing.assert_close(y_pad, y_raw)


def test_invalid_blocks_are_edge_filled():
    x = torch.tensor(_series(100))[None]
    y, valid = RL.resample_context(x, torch.tensor([4.0]), 64)     # 64 blocks of 4 need 256 points
    assert valid.sum() == 25
    first = valid[0].to(torch.long).argmax()
    assert torch.all(y[0, :first] == y[0, first])
    assert torch.isfinite(y).all()


# --------------------------------------------------------------- 3. gradients
def test_gradient_reaches_k():
    x = torch.tensor(_series(3000))[None]
    k = torch.tensor([3.3], requires_grad=True)
    y, _ = RL.resample_context(x, k, 128)
    y.pow(2).sum().backward()
    assert k.grad is not None and torch.isfinite(k.grad).all() and k.grad.abs() > 0
    fan = torch.randn(1, 40, 9).sort(-1).values
    k2 = torch.tensor([2.2], requires_grad=True)
    RL.fan_to_native(fan, k2, 48).sum().backward()
    assert k2.grad is not None and k2.grad.abs() > 0


def test_selector_starts_on_prior():
    sel = RL.RateSelector()
    x = torch.tensor(np.stack([_series(1500, s) for s in range(3)]), dtype=torch.float32)
    g = sel(x)
    assert g.shape == (3,) and torch.all(g == 0)
    layer = RL.RateINLayer(sel)
    lk = layer.log_k(x, log_k_prior=torch.log(torch.tensor([1.0, 2.0, 0.5])))
    torch.testing.assert_close(torch.exp(lk), torch.tensor([1.0, 2.0, 0.5]))
    # features are finite and length-aware
    f = sel.features(x, lengths=torch.tensor([1500, 800, 300]))
    assert torch.isfinite(f).all() and f.shape == (3, 49)
    assert f[0, -1] > f[2, -1]


# ------------------------------------------------------------------ 4. layer
def _stub_forecast(y, n, k):
    """Persistence of the last block, nine equal quantiles: raw units in, raw out."""
    last = y[:, -1:]
    return last[:, :, None].expand(y.shape[0], n, 9).clone()


def test_layer_end_to_end_and_horizon():
    layer = RL.RateINLayer(model_len=64, min_blocks=8)
    x = torch.tensor(np.stack([_series(600, 1), _series(600, 2)]), dtype=torch.float32)
    out = layer(x, h=24, forecast_fn=_stub_forecast, log_k_prior=torch.log(torch.tensor([2.0, 1.0])))
    assert out["fan"].shape == (2, 24, 9) and out["model_horizon"] == 24
    torch.testing.assert_close(out["k"], torch.tensor([2.0, 1.0]))
    out_up = layer(x, h=24, forecast_fn=_stub_forecast, log_k_prior=torch.log(torch.tensor([0.5, 1.0])))
    assert out_up["model_horizon"] == 48 and out_up["fan"].shape == (2, 24, 9)


def test_layer_bounds_k_by_context_length():
    layer = RL.RateINLayer(model_len=64, min_blocks=16)
    x = torch.tensor(_series(100))[None].float()
    lk = layer.log_k(x, log_k_prior=torch.log(torch.tensor([48.0])))
    assert torch.exp(lk).item() == pytest.approx(100 / 16)      # at most len / min_blocks blocks
    lk = layer.log_k(x, log_k_prior=torch.log(torch.tensor([0.01])))
    assert torch.exp(lk).item() == pytest.approx(RL.K_MIN)


# ------------------------------------------------------------------- 5. loss
def test_pinball_matches_harness_crps():
    rng = np.random.default_rng(3)
    target = rng.standard_normal(30) + 5
    fan = np.sort(rng.standard_normal((30, 9)) + 5, axis=1)
    acc = gift.MetricAccumulator()
    acc.add(target, fan[:, 4], fan, 1.0)
    ref = acc.result()["CRPS"]
    got = RL.pinball_wql(torch.tensor(fan)[None], torch.tensor(target)[None], LEVELS)
    assert got.item() == pytest.approx(ref, rel=1e-10)


def test_interp_loss_curve():
    grid = torch.log(torch.tensor([1.0, 2.0, 4.0]))
    losses = torch.tensor([[1.0, 0.5, 0.8]])
    assert RL.interp_loss_curve(grid, losses, torch.log(torch.tensor([2.0]))).item() == pytest.approx(0.5)
    mid = RL.interp_loss_curve(grid, losses, torch.log(torch.tensor([2.0 ** 1.5]))).item()
    assert mid == pytest.approx(0.65)
    assert RL.interp_loss_curve(grid, losses, torch.log(torch.tensor([16.0]))).item() == pytest.approx(0.8)
    lk = torch.log(torch.tensor([1.5]), ).requires_grad_(True)
    RL.interp_loss_curve(grid, losses, lk).backward()
    assert lk.grad.item() < 0                                     # descending toward k = 2


def test_kernel_loss_curve_is_smooth_at_grid_points():
    grid = torch.log(torch.tensor([0.5, 1.0, 2.0, 4.0]))
    losses = torch.tensor([[1.0, 0.9, 0.5, 0.8]])
    lk = torch.zeros(1, requires_grad=True)                        # k = 1, a grid point
    RL.kernel_loss_curve(grid, losses, lk, 0.25).backward()
    assert lk.grad.item() < 0                                      # descends toward k = 2
    far = RL.kernel_loss_curve(grid, losses, torch.log(torch.tensor([2.0])), 0.05).item()
    assert far == pytest.approx(0.5, abs=1e-6)                     # a narrow kernel is the grid value
