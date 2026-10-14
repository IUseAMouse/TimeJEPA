"""
+ratein=layer in the GIFT harness.

1. An untrained layer (g = 0, prior k = 1) through `+ratein=layer` gives the
   plain path's results bit for bit on a stub: same CRPS, same MASE, same
   instance count, k_hist {1: n}.
2. A layer whose prior is k = 2 reproduces the forced-k path (the oracle's
   k = 2 sweep): the layer's resampling is the harness's decimation.
3. Flags are known; the tag names the layer file; +ratein=layer without a
   checkpoint, or backtest options with it, are refused.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa import ratein_layer as RL  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]
H = 24
PERIOD = 24


class _Patching:
    stride, patch_size = 1, 1


class _NaiveStub:
    """Seasonal naive on its own grid with a fixed fan; records calls."""
    input_length = 256
    rate_knob = None

    def __init__(self):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        self.calls.append({"len": x.shape[1], "n": n, "w": None if w is None else float(torch.as_tensor(w).flatten()[0])})
        med = np.stack([np.tile(row[-PERIOD:], n // PERIOD + 1)[:n] for row in x])
        from scipy.stats import norm
        fan = med[:, :, None] + 0.5 * norm.ppf(np.asarray(LEVELS))[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


def _series(rng, n_series=6, length=700):
    out = []
    for _ in range(n_series):
        t = np.arange(length)
        out.append((10 + np.sin(2 * np.pi * t / PERIOD) + 0.2 * rng.standard_normal(length)).astype(np.float32))
    return out


def _run(monkeypatch, model, series, **kw):
    import evaluate_gift as EG
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: H)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    return EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=4, **kw)


def _layer(prior_len=256):
    return RL.RateINLayer(RL.RateSelector(8, 4), model_len=prior_len, min_blocks=8).eval()


def test_untrained_layer_equals_plain_path(monkeypatch):
    rng = np.random.default_rng(0)
    series = _series(rng)
    plain = _run(monkeypatch, _NaiveStub(), series)
    stub = _NaiveStub()
    layered = _run(monkeypatch, stub, series, ratein_mode="layer", ratein_layer=_layer(), ratein_layer_prior=0.0)
    assert layered["model"]["CRPS"] == plain["model"]["CRPS"]
    assert layered["model"]["MASE"] == plain["model"]["MASE"]
    assert layered["seasonal_naive_local"]["CRPS"] == plain["seasonal_naive_local"]["CRPS"]
    assert layered["n_instances"] == plain["n_instances"] == 12
    assert layered["ratein"]["k_hist"] == {"1.0": 12}
    assert all(c["len"] == 256 and c["n"] == H for c in stub.calls)
    assert layered["ratein"]["layer"]["log_k_mean"] == 0.0


def test_layer_prior_k2_matches_forced_k(monkeypatch):
    rng = np.random.default_rng(1)
    series = _series(rng)
    forced = _run(monkeypatch, _NaiveStub(), series, forced_k=2)
    layered = _run(monkeypatch, _NaiveStub(), series, ratein_mode="layer", ratein_layer=_layer(),
                   ratein_layer_prior=float(np.log(2.0)))
    assert layered["ratein"]["k_hist"] == {"2.0": 12}
    assert layered["model"]["CRPS"] == pytest.approx(forced["model"]["CRPS"], rel=1e-6)
    assert layered["model"]["MASE"] == pytest.approx(forced["model"]["MASE"], rel=1e-6)


def test_layer_checkpoint_round_trip(tmp_path):
    layer = _layer()
    with torch.no_grad():
        layer.selector.mlp[-1].weight.fill_(0.01)
    torch.save({"state_dict": layer.state_dict(),
                "meta": {"n_modes": 8, "hidden": 4, "model_len": 256, "min_blocks": 8}}, tmp_path / "l.pt")
    back = RL.load_layer(tmp_path / "l.pt")
    x = torch.randn(2, 400)
    torch.testing.assert_close(back.log_k(x), layer.log_k(x))


def test_flags_and_tag():
    import evaluate_gift as EG
    for f in ("ratein_layer", "ratein_layer_prior"):
        assert f in EG.KNOWN_FLAGS
    src = Path(EG.__file__).read_text()
    assert '"layer": "layer"' in src and "_ratein-layer-" in src
    assert "+ratein=layer needs +ratein_layer" in src
    assert "takes no backtest options" in src
