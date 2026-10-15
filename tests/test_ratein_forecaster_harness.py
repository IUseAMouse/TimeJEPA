"""
RateINForecaster through the plain path of the GIFT harness.

The wrapper is evaluated as any model (+ratein off); the harness only reads
the optional `last_k` / `last_log_k` of the model for its report.
1. An untrained wrapper (g = 0, prior k = 1) around the stub gives the stub's
   own results bit for bit (CRPS, MASE, instance count), with and without the
   sign flip and the frequency-tied w; the report carries k_hist {1: n}.
2. A wrapper with prior k = 2 reproduces the forced-k sweep at k = 2, and the
   stub sees w = 2 on both sides when the frequency is declared.
3. A batch with mixed rates keeps every instance; the stub is asked
   ceil(h / min k) steps.
4. Guards and flags, called directly: the old +ratein_layer flags are unknown,
   a RateIN mode around the wrapper is refused, +freq_delta needs the knob.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.models.ratein_forecaster import RateINForecaster  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]
PERIOD = 24
H = 48


class _Patching:
    stride, patch_size = 1, 1


class _RateStub:
    """Persistence when w is None or 1, seasonal naive when w != 1 (so the
    w plumbing is observable). Records every call."""
    input_length = 256
    rate_knob = "delta"
    delta_range = (1.0 / 48.0, 4.0)
    expects_frequency = False
    prediction_length = H

    def __init__(self, knob=True):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        if not knob:
            self.rate_knob = None
        self.calls = []

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        s = None if w is None else float(torch.as_tensor(w).reshape(-1)[0])
        self.calls.append({"len": x.shape[1], "n": n, "w": s})
        meds = []
        for row in x:
            if s is None or s == 1.0:
                med = np.full(n, row[-1])
            else:
                med = np.tile(row[-PERIOD:], n // PERIOD + 1)[:n]
            meds.append(med)
        med = np.stack(meds)
        from scipy.stats import norm
        fan = med[:, :, None] + 0.5 * norm.ppf(np.asarray(LEVELS))[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


class _WrappedStub(RateINForecaster):
    """The stub is not an nn.Module: register it as a plain attribute."""

    def __init__(self, stub, layer, k_prior=1.0):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "inner", stub)
        self.layer = layer
        self.log_k_prior = math.log(k_prior)
        self.k_quant_octave = 8
        self.name = "wrapped_stub"
        self.prediction_length = H
        self.input_length = int(layer.model_len * RL.K_MAX)
        self.rate_knob = stub.rate_knob
        self.delta_range = stub.delta_range
        self.expects_frequency = False
        self._pretrain_mode = False
        self.last_log_k = self.last_k = self.last_model_horizon = self._last_n_unique_w = None


def _synthetic(rng, n_series=8, length=900):
    out = []
    for i in range(n_series):
        if i == n_series - 1:
            length = 150                                 # shorter than model_len: no edge fill may reach the inner
        level, amp = rng.uniform(5, 50), rng.uniform(1, 5)
        t = np.arange(length)
        period = PERIOD if i % 2 == 0 else PERIOD * 3
        y = level + amp * np.sin(2 * np.pi * t / period) + rng.normal(scale=0.1 * amp, size=length)
        out.append(y.astype(np.float32))
    return out


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    series = _synthetic(np.random.default_rng(3))
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: H)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


def _layer():
    return RL.RateINLayer(RL.RateSelector(8, 4), model_len=256, min_blocks=8).eval()


def _wrap(k_prior=1.0, layer=None):
    stub = _RateStub()
    return _WrappedStub(stub, layer or _layer(), k_prior), stub


# ------------------------------------------------------------- 1. identity
@pytest.mark.parametrize("flags", [{}, {"tta_flip": True}, {"freq_season": 24.0}])
def test_untrained_wrapper_equals_the_stub(harness, flags):
    plain = _run(harness, _RateStub(), **flags)
    # Same context cap as the stub: the harness builds the same batches in
    # the same order, so the accumulated metrics are bit-identical.
    wrapper, stub = _wrap()
    wrapped = _run(harness, wrapper, max_context=256, **flags)
    for key in ("CRPS", "MASE"):
        assert wrapped["model"][key] == plain["model"][key]
    assert wrapped["seasonal_naive_local"]["CRPS"] == plain["seasonal_naive_local"]["CRPS"]
    assert wrapped["n_instances"] == plain["n_instances"] == 16
    # With the wrapper's own cap (the whole native past) the batches are
    # grouped by native length: same forecasts, another summation order.
    full = _run(harness, _wrap()[0], **flags)
    for key in ("CRPS", "MASE"):
        assert full["model"][key] == pytest.approx(plain["model"][key], rel=1e-12)
    assert all(c["len"] in (256, 150 - H, 150 - 2 * H) and c["n"] == H for c in stub.calls)
    assert {150 - H, 150 - 2 * H} <= {c["len"] for c in stub.calls}      # the short series reached the stub as is
    if "freq_season" in flags:
        assert all(c["w"] == 1.0 for c in stub.calls)
    else:
        assert all(c["w"] is None for c in stub.calls)
    assert wrapped["ratein"]["k_hist"] == {"1.0": 16} and wrapped["ratein"]["layer"]["log_k_mean"] == 0.0
    assert "ratein" not in plain


# -------------------------------------------------------------- 2. k = 2
def test_wrapper_prior_two_matches_forced_k(harness):
    forced_stub = _RateStub()
    forced = _run(harness, forced_stub, forced_k=2, freq_season=24.0)
    wrapper, stub = _wrap(k_prior=2.0)
    wrapped = _run(harness, wrapper, freq_season=24.0)
    assert wrapped["ratein"]["k_hist"] == {"2.0": 16}
    assert wrapped["model"]["CRPS"] == pytest.approx(forced["model"]["CRPS"], rel=1e-6)
    assert wrapped["model"]["MASE"] == pytest.approx(forced["model"]["MASE"], rel=1e-6)
    assert all(c["len"] <= 256 and c["n"] == H // 2 and c["w"] == 2.0 for c in stub.calls)
    assert all(c["w"] == 2.0 for c in forced_stub.calls)


# ------------------------------------------------------------ 3. mixed k
def test_mixed_rates_keep_every_instance(harness):
    layer = _layer()
    with torch.no_grad():
        layer.selector.mlp[-1].weight.fill_(0.3)
    wrapper, stub = _wrap(layer=layer)
    res = _run(harness, wrapper)
    assert res["n_instances"] == 16
    assert len(res["ratein"]["layer"]["k_hist"]) > 1 and res["ratein"]["layer"]["log_k_std"] > 0
    for c in stub.calls:
        assert c["len"] <= 256 and c["n"] >= H // 48


# -------------------------------------------------------- 4. guards, flags
def test_guards_and_flags(harness):
    from evaluate_gift import check_model_flags, check_unknown_flags
    check_unknown_flags(["+freq_delta=true", "+tta_flip=true", "+checkpoint_path=x", "+ratein_k_up=2x3"])
    for flag in ("+ratein_layer=x", "+ratein_layer_prior=x"):
        with pytest.raises(ValueError, match="unknown flag"):
            check_unknown_flags([flag])
    wrapper, _ = _wrap()
    check_model_flags(wrapper, "off", False, None, None, freq_delta=True)
    for mode in ("backtest", "mix", "delta", "fft"):
        with pytest.raises(ValueError, match="owns its sampling rate"):
            check_model_flags(wrapper, mode, False, None, None)
    noknob = _WrappedStub(_RateStub(knob=False), _layer())
    with pytest.raises(ValueError, match="rate knob"):
        check_model_flags(noknob, "off", False, None, None, freq_delta=True)
    check_model_flags(noknob, "off", False, None, None)
