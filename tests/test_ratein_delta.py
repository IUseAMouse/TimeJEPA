"""
RateIN-Delta (2026-09-09): the backtest selector of RateIN on a model's rate
knob (w = 1/k) instead of the data's decimation.

Pinned on a stub whose forecast is rate-sensitive (persistence at w = 1,
exact seasonal naive at any w < 1):
1. +ratein=delta: the selector picks a k > 1, the test forecasts receive
   w = 1/k, the contexts keep their NATIVE length (no decimation), the fans
   their native horizon (no re-interpolation), and the CRPS drops; the
   diagnostic says knob = delta.
2. +ratein=backtest on the same stub: decimated contexts, no w - the
   existing path is untouched.
3. Guards: delta refused on a model without rate_knob, with +ratein_w, and
   +refine / +ttt refused on a model without online encoder; flags known.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

LEVELS = [0.1 * j for j in range(1, 10)]
PERIOD = 24


class _Patching:
    stride, patch_size = 8, 16


class _RateStub:
    """Persistence when w is None or 1 (bad on a seasonal series); seasonal
    naive (exact on the synthetic sine) when w < 1. Records every call."""
    input_length = 256
    rate_knob = "delta"

    def __init__(self, knob=True):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        if not knob:
            self.rate_knob = None
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        s = None if w is None else float(w.reshape(-1)[0])
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


def _synthetic(rng, n_series=10, length=900):
    out = []
    for _ in range(n_series):
        level, amp = rng.uniform(5, 50), rng.uniform(1, 5)
        t = np.arange(length)
        y = level + amp * np.sin(2 * np.pi * t / PERIOD) + rng.normal(scale=0.1 * amp, size=length)
        out.append(y.astype(np.float32))
    return out


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    series = _synthetic(np.random.default_rng(3))
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 48)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


def test_delta_selects_a_rate_and_passes_w_without_decimation(harness):
    EG = harness
    off = _run(EG, _RateStub())
    stub = _RateStub()
    res = _run(EG, stub, ratein_mode="delta", ratein_pool=True)
    bt = res["ratein"]["backtest"]
    assert bt["knob"] == "delta" and bt["K"] > 1
    assert res["model"]["CRPS"] < 0.5 * off["model"]["CRPS"]
    # the selected k is applied as w = 1/K at the NATIVE horizon: the test
    # forecasts are the LAST calls (the backtest runs first, same h here)
    n_test_calls = -(-res["n_instances"] // 8)
    test_calls = stub.calls[-n_test_calls:]
    assert all(c["w"] == pytest.approx(1.0 / bt["K"]) for c in test_calls)
    # contexts never decimated: every call sees the native (capped) length
    native = 256 - (256 % 8)
    assert all(c["len"] == native for c in stub.calls)
    # backtest calls: k = 1 without w, k > 1 with w = 1/k, all at h_bt = 48
    bt_calls = [c for c in stub.calls if c["n"] == 48]
    ws = {c["w"] for c in bt_calls}
    assert None in ws and any(w is not None and w < 1 for w in ws)
    assert res["ratein"]["k_hist"] == {str(bt["K"]): res["n_instances"]}


def test_backtest_path_untouched_decimates_and_never_passes_w(harness):
    EG = harness
    stub = _RateStub()
    res = _run(EG, stub, ratein_mode="backtest", ratein_pool=True)
    assert res["ratein"]["backtest"]["knob"] == "decimation"
    assert all(c["w"] is None for c in stub.calls)
    # decimated contexts (shorter than native) and shorter horizons were asked
    assert any(c["len"] < 248 for c in stub.calls)
    assert any(c["n"] < 48 for c in stub.calls)


def test_guards_and_flags():
    from evaluate_gift import check_model_flags, check_unknown_flags
    check_unknown_flags(["+ratein=delta", "+ratein_pool=true"])
    check_model_flags(_RateStub(), "delta", False, None, None)
    with pytest.raises(ValueError, match="rate knob"):
        check_model_flags(_RateStub(knob=False), "delta", False, None, None)
    with pytest.raises(ValueError, match="exclusive"):
        check_model_flags(_RateStub(), "delta", True, None, None)
    spec = type("S", (), {"active": True})()
    with pytest.raises(ValueError, match="JEPA model"):
        check_model_flags(_RateStub(), "off", False, spec, None)
    with pytest.raises(ValueError, match="JEPA model"):
        check_model_flags(_RateStub(), "off", False, None, {"params": "norm"})
    check_model_flags(_RateStub(), "backtest", False, None, None)


def test_builder_dispatch_and_core_prefixes(tmp_path):
    from omegaconf import OmegaConf
    from timejepa.evaluation import loading
    import types
    mod = types.ModuleType("_fake_builder")
    calls = []

    class _M(torch.nn.Module):
        core_prefixes = ("blocks.",)

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.Linear(2, 2)

    mod.build = lambda cfg: (calls.append(cfg.model.name), _M())[1]
    sys.modules["_fake_builder"] = mod
    cfg = OmegaConf.create({"model": {"name": "x", "builder": "_fake_builder:build"}})
    m = loading.create_model_from_config(cfg)
    assert isinstance(m, _M) and calls == ["x"]
    with pytest.raises(ValueError):
        loading.create_model_from_config(OmegaConf.create({"model": {"builder": "nocolon"}}))
    # a checkpoint whose core does not match the model is refused
    ck = tmp_path / "m.ckpt"
    torch.save({"state_dict": {"model.blocks.weight": torch.zeros(3, 3),
                               "model.blocks.bias": torch.zeros(3)}}, ck)
    with pytest.raises(RuntimeError, match="core components"):
        loading.load_checkpoint(_M(), str(ck), torch.device("cpu"))
    torch.save({"state_dict": {"model.blocks.weight": torch.ones(2, 2),
                               "model.blocks.bias": torch.ones(2)}}, ck)
    m2 = loading.load_checkpoint(_M(), str(ck), torch.device("cpu"))
    assert torch.equal(m2.blocks.weight, torch.ones(2, 2))
