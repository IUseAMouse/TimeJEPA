"""
BiasIN (2026-09-08): causal level-bias correction from the backtest.

Pinned:
1. level_scale is a robust unit (MAD), safe on constants; level_bias has the
   sign of (truth - median) in scale units; shift_fan translates every
   quantile and the median together.
2. choose_shrink: a bias that persists from the older to the recent window
   is accepted at lambda 1 (pooled pinball ratio well below 1 - margin); a
   bias whose sign flips between windows is refused (lambda 0, no-op).
3. Harness end-to-end on a stub forecaster with a constant level bias:
   +bias=backtest lowers the CRPS, chooses lambda > 0, tags official; the
   oracle (constant per-instance shift from the target) is at least as good;
   +bias absent is bit-identical to the plain path; mix path also shifted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation import biasin as B  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]


def test_scale_bias_shift():
    assert B.level_scale(np.ones(300)) == 1.0
    rng = np.random.default_rng(0)
    s = B.level_scale(rng.normal(size=2000) * 3.0)
    assert 2.5 < s < 3.5
    known = np.full(20, 5.0)
    med = np.full(20, 4.0)
    assert abs(B.level_bias(known, med, 2.0) - 0.5) < 1e-12
    # heavy tail: one spike must not drag the bias (median, not mean)
    spiky = known.copy(); spiky[3] = 5000.0
    assert abs(B.level_bias(spiky, med, 2.0) - 0.5) < 1e-12
    assert B.oracle_shift(spiky, med) == 1.0
    assert np.isnan(B.level_bias(np.full(3, np.nan), med[:3], 1.0))
    fan = rng.normal(size=(20, 9))
    fan_s, med_s = B.shift_fan(fan, med, 0.7)
    assert np.allclose(fan_s - fan, 0.7) and np.allclose(med_s - med, 0.7)
    assert B.shift_fan(None, med, 0.7)[0] is None


def _fan_from_median(med, width=0.5):
    q = np.asarray(LEVELS)
    from scipy.stats import norm
    return med[:, None] + width * norm.ppf(q)[None, :]


def test_choose_shrink_accepts_persistent_bias_refuses_flipping():
    rng = np.random.default_rng(1)
    h, n = 48, 30
    persistent, flipping = [], []
    for i in range(n):
        truth = rng.normal(size=h)
        scale = 1.0
        b = 0.8                                    # truth = median + b*scale
        fan = _fan_from_median(truth - b * scale)  # recent window, biased median
        persistent.append({"fan": fan, "known": truth, "beta_old": b, "scale": scale})
        flipping.append({"fan": fan, "known": truth,
                         "beta_old": b * (1 if i % 2 else -1), "scale": scale})
    acc = B.choose_shrink(persistent, LEVELS)
    assert acc["lambda"] == 1.0 and acc["ratios"]["1"] < 0.7 and acc["n_val"] == n
    ref = B.choose_shrink(flipping, LEVELS)
    assert ref["lambda"] == 0.0
    assert B.choose_shrink([], LEVELS)["lambda"] == 0.0
    assert B.series_beta([0.2, np.nan, 0.4]) == pytest.approx(0.3)
    assert B.test_shift(0.3, 0.0, np.ones(10)) == 0.0
    assert B.test_shift(float("nan"), 1.0, np.ones(10)) == 0.0


# ------------------------------------------------------------ harness stub
class _Patching:
    stride, patch_size = 8, 16


class _StubModel:
    """Seasonal-naive forecaster (period 24, exact on the synthetic sine) with
    a CONSTANT level bias of `bias` scale units: median = y[t-24] - bias *
    scale(context). A persistence stub would not do: its residual depends
    on the phase at the window start and is not a persistent bias. The
    quantile fan is a fixed-width Gaussian around the median."""
    input_length = 256

    def __init__(self, bias=0.6, width=0.5):
        self.patching = _Patching()
        self.bias, self.width = bias, width
        self.predictor = object()

    def forecast(self, batch, n=None, **kw):
        import torch
        x = batch[..., 0].cpu().numpy()
        meds, fans = [], []
        for row in x:
            sc = B.level_scale(row)
            season = row[-24:]
            med = np.tile(season, n // 24 + 1)[:n] - self.bias * sc
            meds.append(med); fans.append(_fan_from_median(med, self.width * sc))
        med = torch.tensor(np.stack(meds), dtype=torch.float32).unsqueeze(-1)
        fan = torch.tensor(np.stack(fans), dtype=torch.float32)
        return {"forecast_denorm": med, "quantiles_denorm": fan,
                "quantile_levels": LEVELS}


def _synthetic(rng, n_series=12, length=900):
    out = []
    for _ in range(n_series):
        level = rng.uniform(5, 50)
        amp = rng.uniform(1, 5)
        t = np.arange(length)
        y = level + amp * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3 * amp, size=length)
        out.append(y.astype(np.float32))
    return out


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    rng = np.random.default_rng(3)
    series = _synthetic(rng)
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 48)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: 24)
    return EG


def _run(EG, model, **kw):
    import torch
    return EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


def test_harness_backtest_bias_lowers_crps_and_oracle_bounds_it(harness):
    EG = harness
    model = _StubModel(bias=0.6)
    plain = _run(EG, model)
    plain2 = _run(EG, model, bias_mode="off")
    assert plain["model"]["CRPS"] == plain2["model"]["CRPS"] and "bias" not in plain2
    bt = _run(EG, model, bias_mode="backtest")
    assert bt["bias"]["official"] is True and bt["bias"]["lambda"] > 0
    assert bt["bias"]["n_val"] > 0 and bt["bias"]["frac_shifted"] > 0.9
    assert bt["model"]["CRPS"] < 0.8 * plain["model"]["CRPS"]
    assert bt["model"]["MASE"] < plain["model"]["MASE"]
    orc = _run(EG, model, bias_mode="oracle")
    assert orc["bias"]["official"] is False
    assert orc["model"]["CRPS"] <= bt["model"]["CRPS"] + 1e-9
    # an unbiased forecaster: the backtest refuses the correction (no-op)
    fair = _StubModel(bias=0.0)
    p0 = _run(EG, fair)
    b0 = _run(EG, fair, bias_mode="backtest")
    assert b0["bias"]["lambda"] == 0.0
    assert b0["model"]["CRPS"] == pytest.approx(p0["model"]["CRPS"], rel=1e-9)


def test_harness_mix_path_is_shifted_too(harness):
    EG = harness
    model = _StubModel(bias=0.6)
    plain = _run(EG, model, ratein_mode="mix", ratein_pool=True)
    bt = _run(EG, model, ratein_mode="mix", ratein_pool=True, bias_mode="backtest")
    assert bt["bias"]["lambda"] > 0 and bt["model"]["CRPS"] < 0.8 * plain["model"]["CRPS"]


def test_flags_and_tags():
    from evaluate_gift import check_unknown_flags
    check_unknown_flags(["+bias=backtest", "+bias_lambdas=0.5,1"])
    with pytest.raises(ValueError):
        check_unknown_flags(["+bias_lambda=1"])
