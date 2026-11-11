"""
evaluate_config on a ConfigData (series, h, m, windows given directly, as for a held-out dev
subset) equals the GIFT loading path on the same series, plain and with the RateIN mix.
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


class _LastTwo:
    """Repeats its last two steps: wrong on a sine of period 24, close once the context
    is decimated by 12, so the RateIN mix has something to choose."""
    input_length = 256

    def __init__(self):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()

    def forecast(self, batch, n=None, **kw):
        x = batch[..., 0].cpu().numpy()
        med = np.stack([np.tile(row[-2:], n // 2 + 1)[:n] for row in x])
        from scipy.stats import norm
        fan = med[:, :, None] + 0.1 * norm.ppf(np.asarray(LEVELS))[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


def _sines(n_series=10, length=900):
    rng = np.random.default_rng(3)
    t = np.arange(length)
    return [(rng.uniform(5, 50) + rng.uniform(1, 5) * np.sin(2 * np.pi * t / PERIOD)
             + rng.normal(scale=0.05, size=length)).astype(np.float32) for _ in range(n_series)]


@pytest.mark.parametrize("kw", [{}, dict(ratein_mode="mix", ratein_pool=True, ratein_bt_windows=4)])
def test_config_data_matches_the_gift_loading(monkeypatch, kw):
    import evaluate_gift as EG
    series = _sines()
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 48)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)

    def run(data):
        return EG.evaluate_config(_LastTwo(), "dev/H/short", Path("."), torch.device("cpu"),
                                  batch_size=8, data=data, **kw)
    loaded, given = run(None), run(EG.gift.ConfigData(series, h=48, m=PERIOD, windows=2))
    assert loaded["model"] == given["model"] and loaded["n_instances"] == given["n_instances"]
    assert loaded.get("ratein") == given.get("ratein")
    if kw:
        assert given["ratein"]["mix"]["weights"] == {"12": 1.0}
