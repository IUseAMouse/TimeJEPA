"""
RateIN-up (2026-09-26, plan B3'): upsampling candidates k = 1/m and a
backtest allowed on short horizons.

Pinned on a stub that only exploits a cycle whose fundamental period is
>= 16 steps on ITS grid (persistence otherwise): a monthly cycle of 12 is
useless at k = 1 and at every decimation, exact once upsampled by 2.
1. Defaults (min_bt 16, no k_up): on h = 12 the backtest never runs
   (n_base = 0, k_hist {1: n}) - the existing behaviour, bit-identical.
2. +ratein_min_bt=4 +ratein_k_up=2: the selector picks k = 0.5, the model is
   asked for 24 steps on a context of at most input_length fine points, the
   fan comes back on 12 native steps, the CRPS drops.
3. Primitives: upsample/decimate inverse on linear signals, fc_horizon,
   to_native_fan shape and monotonicity, resample_context length.
4. Flags known; the tag names the variant.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation import ratein as R  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]
PERIOD = 12
H = 12


class _Patching:
    stride, patch_size = 1, 1


def _fundamental(row, lo=2, hi=60):
    x = row - row.mean()
    if not np.any(x):
        return None
    for lag in range(lo, min(hi, len(x) // 3)):
        a, b = x[:-lag], x[lag:]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if c > 0.9:
            return lag
    return None


class _BandStub:
    """Seasonal naive when the fundamental period on its own grid is >= 16,
    persistence otherwise. Records every call."""
    input_length = 256
    rate_knob = None

    def __init__(self):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        self.calls.append({"len": x.shape[1], "n": n})
        meds = []
        for row in x:
            p = _fundamental(row)
            if p is None or p < 16:
                meds.append(np.full(n, row[-1]))
            else:
                meds.append(np.tile(row[-p:], n // p + 1)[:n])
        med = np.stack(meds)
        from scipy.stats import norm
        fan = med[:, :, None] + 0.5 * norm.ppf(np.asarray(LEVELS))[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


def _synthetic(rng, n_series=8, length=300):
    out = []
    for _ in range(n_series):
        level, amp = rng.uniform(20, 50), rng.uniform(2, 5)
        t = np.arange(length)
        y = level + amp * np.sin(2 * np.pi * t / PERIOD) + rng.normal(scale=0.02 * amp, size=length)
        out.append(y.astype(np.float32))
    return out


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    series = _synthetic(np.random.default_rng(7))
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: H)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/M/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


def test_defaults_keep_the_selector_off_on_short_horizons(harness):
    m = _BandStub()
    res = _run(harness, m, ratein_mode="backtest", ratein_pool=True)
    assert res["ratein"]["backtest"]["n_base"] == 0
    assert res["ratein"]["k_hist"] == {"1": 16}
    assert res["ratein"]["backtest"]["min_bt"] == 16
    off = _run(harness, _BandStub(), ratein_mode="off")
    assert res["model"]["CRPS"] == pytest.approx(off["model"]["CRPS"])


def test_short_backtest_with_upsampling_picks_k_half(harness):
    m = _BandStub()
    res = _run(harness, m, ratein_mode="backtest", ratein_pool=True,
               ratein_k_up=[2], ratein_min_bt=4, ratein_bt_windows=4)
    bt = res["ratein"]["backtest"]
    assert bt["n_base"] > 0 and bt["min_bt"] == 4 and bt["windows"] == 4
    assert 0.5 in bt["candidates"] and "0.5" in bt["ratios"]
    assert bt["ratios"]["0.5"] < 0.5                  # exact cycle vs persistence
    assert bt["K"] == 0.5 and res["ratein"]["k_hist"] == {"0.5": 16}
    # the test forecasts: 24 fine steps on at most input_length fine points
    test_calls = [c for c in m.calls if c["n"] == 24]
    assert test_calls and all(c["len"] <= 256 for c in test_calls)
    base = _run(harness, _BandStub(), ratein_mode="off")
    assert res["model"]["CRPS"] < 0.3 * base["model"]["CRPS"]
    assert res["model"]["MASE"] < 0.3 * base["model"]["MASE"]


def test_mix_mode_accepts_fractional_components(harness):
    m = _BandStub()
    res = _run(harness, m, ratein_mode="mix", ratein_pool=True,
               ratein_k_up=[2, 3], ratein_min_bt=4)
    w = res["ratein"]["mix"]["weights"]
    assert max(w, key=lambda k: w[k]) in ("0.5", 0.5)
    base = _run(harness, _BandStub(), ratein_mode="off")
    assert res["model"]["CRPS"] < 0.5 * base["model"]["CRPS"]


def test_primitives():
    x = 3 + 2 * np.arange(10, dtype=np.float64)
    for m in (2, 3, 4):
        u = R.upsample(x, m)
        assert len(u) == 10 * m and np.allclose(R.decimate(u, m), x)
    s = np.sin(np.arange(60) / 5.0)
    assert np.abs(R.decimate(R.upsample(s, 3), 3) - s).max() < 0.02
    assert R.fc_horizon(12, 0.5) == 24 and R.fc_horizon(12, 4) == 3 and R.fc_horizon(12, 1) == 12
    fan = np.sort(np.random.default_rng(0).random((24, 9)), axis=1)
    pooled = R.to_native_fan(fan, 12, 0.5)
    assert pooled.shape == (12, 9) and (np.diff(pooled, axis=1) >= 0).all()
    assert np.array_equal(R.to_native_fan(fan, 12, 1), fan[:12])
    assert len(R.resample_context(np.arange(1000.0), 0.5, 256)) == 256
    assert len(R.resample_context(np.arange(1000.0), 4, 256)) == 250
    assert R.norm_k(2.0) == 2 and isinstance(R.norm_k(2.0), int) and R.norm_k(0.5) == 0.5


def test_flags_known():
    import evaluate_gift as EG
    for f in ("ratein_k_up", "ratein_min_bt", "ratein_bt_windows"):
        assert f in EG.KNOWN_FLAGS
