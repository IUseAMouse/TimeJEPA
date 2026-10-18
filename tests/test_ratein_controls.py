"""
RateIN controls: variants of the decimation operator that separate its three
effects (samples per cycle, smoothing, native span) in oracle sweeps.

1. Primitives: smooth keeps the native length and is a trailing mean;
   subsample keeps one value per block; samespan only reads the native span
   the model sees at k = 1; k = 1 returns the context unchanged.
2. Harness: without a control the forced-k path is unchanged; smooth asks the
   model for the native horizon on a native-length context; subsample and
   samespan ask for ceil(h/k) steps; samespan sees fewer coarse points.
3. Flags known.
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
H = 24


class _Patching:
    stride, patch_size = 1, 1


class _PersistStub:
    """Persistence forecaster that records the context length and horizon."""
    input_length = 64
    rate_knob = None

    def __init__(self):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        self.calls.append({"len": x.shape[1], "n": n})
        med = np.repeat(x[:, -1:], n, axis=1)
        fan = med[:, :, None] + 0.1 * (np.asarray(LEVELS) - 0.5)[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    rng = np.random.default_rng(3)
    t = np.arange(600)
    series = [(10 + np.sin(2 * np.pi * t / 96) + 0.3 * rng.standard_normal(600)).astype(np.float32)
              for _ in range(4)]
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: H)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: 96)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/15T/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


def test_primitives():
    x = np.arange(40, dtype=np.float32)
    sm = R.control_context(x, 4, 32, "smooth")
    assert len(sm) == 32 and np.allclose(sm[3:], np.convolve(x[-32:], np.ones(4) / 4, "valid"))
    assert np.allclose(R.control_context(x, 4, 100, "subsample"), x[2::4])
    assert np.allclose(R.control_context(x, 4, 16, "samespan"), R.decimate(x[-16:], 4))
    for c in R.CONTROLS:
        assert np.array_equal(R.control_context(x, 1, 16, c), x)
    with pytest.raises(ValueError):
        R.control_context(x, 2, 16, "nope")
    assert R.control_is_native("smooth") and not R.control_is_native("samespan")


def test_harness_paths(harness):
    base = _run(harness, _PersistStub(), forced_k=4)
    same = _run(harness, _PersistStub(), forced_k=4, ratein_control="")
    assert base["model"]["CRPS"] == same["model"]["CRPS"]

    m = _PersistStub()
    _run(harness, m, forced_k=4, ratein_control="smooth")
    assert {c["n"] for c in m.calls} == {H} and max(c["len"] for c in m.calls) == 64

    for ctl in ("subsample", "samespan"):
        m = _PersistStub()
        _run(harness, m, forced_k=4, ratein_control=ctl)
        assert {c["n"] for c in m.calls} == {H // 4}
    m_full, m_span = _PersistStub(), _PersistStub()
    _run(harness, m_full, forced_k=4)
    _run(harness, m_span, forced_k=4, ratein_control="samespan")
    assert max(c["len"] for c in m_span.calls) == 16 < max(c["len"] for c in m_full.calls)


def test_flags_known():
    import evaluate_gift as EG
    assert {"ratein_control", "ratein_oracle_ks"} <= EG.KNOWN_FLAGS
