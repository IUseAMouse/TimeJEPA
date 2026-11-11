"""
+multivariate: the items of a multivariate config are forecast whole (VariateBatches over
forecast_multivariate). A model that forecasts each variate on its own must score exactly
as through the univariate split, plain, flipped and with the RateIN stack, including the
variates read as context only (a target or a past with nothing observed); a model that
reads across variates can do better; every variate of an item gets the same rate.
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
H, V = 48, 3
STACK = dict(ratein_mode="mix", ratein_pool=True, ratein_k_up=[2, 3, 4], ratein_min_bt=4,
             ratein_bt_windows=4)


class _Patching:
    stride, patch_size = 8, 16


class _Independent:
    """Repeats the last two steps of each variate (a NaN-aware copy, so a blank variate
    forecasts NaN); forecast_multivariate runs forecast on each variate: no reading across
    variates. Records the rows of every joint call."""
    input_length = 256
    handles_nan = True

    def __init__(self):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = (batch[..., 0] if batch.dim() == 3 else batch).cpu().numpy()
        med = np.stack([np.tile(row[-2:], n // 2 + 1)[:n] for row in x])
        from scipy.stats import norm
        fan = med[:, :, None] + 0.1 * norm.ppf(np.asarray(LEVELS))[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}

    def forecast_multivariate(self, x, n=None, w=None, **kw):
        self.calls.append((tuple(x.shape), None if w is None else w.clone()))
        outs = [self.forecast(x[:, c], n=n, w=w) for c in range(x.shape[1])]
        return {"forecast_denorm": torch.stack([o["forecast_denorm"] for o in outs], 1),
                "quantiles_denorm": torch.stack([o["quantiles_denorm"] for o in outs], 1),
                "quantile_levels": LEVELS}


class _Lead(_Independent):
    """Variate 1 leads variate 0 by H steps: reading it gives variate 0's horizon."""

    def forecast_multivariate(self, x, n=None, w=None, **kw):
        out = super().forecast_multivariate(x, n=n, w=w)
        lead = x[:, 1, -n:].float()
        out["forecast_denorm"][:, 0] = lead[..., None]
        out["quantiles_denorm"][:, 0] = (lead[..., None]
                                         + out["quantiles_denorm"][:, 0]
                                         - out["quantiles_denorm"][:, 0, :, 4:5])
        return out


def _items(n_items=4, length=900):
    """Items of V aligned sines; item 1 has a variate with nothing observed in the last
    test window (context only there), item 2 a variate with no past at all."""
    rng = np.random.default_rng(3)
    t = np.arange(length)
    series = []
    for i in range(n_items):
        base = rng.uniform(5, 50)
        for c in range(V):
            series.append((base + c + rng.uniform(1, 5) * np.sin(2 * np.pi * t / PERIOD)
                           + rng.normal(scale=0.05, size=length)).astype(np.float32))
    series[1 * V + 2][-H:] = np.nan
    series[2 * V + 1][:-2 * H] = np.nan
    return series


def _run(model, series, **kw):
    import evaluate_gift as EG
    data = EG.gift.ConfigData(series, h=H, m=PERIOD, windows=2,
                              n_variates=V if kw.pop("multivariate", False) else 1)
    return EG.evaluate_config(model, "mv/H/short", Path("."), torch.device("cpu"),
                              batch_size=8, data=data, multivariate=data.n_variates > 1, **kw)


@pytest.mark.parametrize("kw", [{}, {"tta_flip": True}, {"freq_season": PERIOD, **STACK},
                                {"tta_flip": True, **STACK}],
                         ids=["plain", "flip", "stack", "flip_stack"])
def test_independent_variates_score_as_the_univariate_split(kw):
    series = _items()
    uni = _run(_Independent(), series, **kw)
    model = _Independent()
    multi = _run(model, series, multivariate=True, **kw)
    assert multi["model"] == uni["model"]
    assert multi["seasonal_naive_local"] == uni["seasonal_naive_local"]
    assert multi["n_instances"] == uni["n_instances"]
    assert multi.get("ratein") == uni.get("ratein")
    assert model.calls and all(shape[1] == V for shape, _ in model.calls)


def test_reading_across_variates_helps():
    series = _items()
    for j in range(4):           # variate 1 of each item leads variate 0 by H steps
        series[j * V] = np.concatenate([np.full(H, series[j * V + 1][0]),
                                        series[j * V + 1][:-H]]).astype(np.float32)
    uni = _run(_Lead(), series)
    multi = _run(_Lead(), series, multivariate=True)
    # Variate 0 (a third of the scored variates) becomes exact, the others are unchanged.
    assert multi["model"]["CRPS"] < 0.8 * uni["model"]["CRPS"]


def test_one_rate_per_item():
    model = _Independent()
    _run(model, _items(), multivariate=True, freq_season=PERIOD, **STACK)
    # VariateBatches hands one w per item, so a joint call has as many w as items.
    assert all(w is not None and w.shape == (shape[0],) for shape, w in model.calls)


def test_items_stay_whole_and_unknown_modes_are_refused():
    import evaluate_gift as EG
    series = _items()
    with pytest.raises(ValueError, match="different lengths"):
        list(EG.gift.iter_test_instances(series[:2] + [series[2][:-5]], H, 2, V))
    with pytest.raises(ValueError, match="items of"):
        list(EG.gift.iter_test_instances(series[:4], H, 2, V))
    with pytest.raises(ValueError, match="composes"):
        _run(_Independent(), series, multivariate=True, ratein_mode="fft")
    with pytest.raises(ValueError, match="composes"):
        _run(_Independent(), series, multivariate=True, bias_mode="backtest")


def test_unscored_variates_are_context_only():
    import evaluate_gift as EG
    insts = list(EG.gift.iter_test_instances(_items(), H, 2, V))
    assert len(insts) == 4 * 2 * V
    assert [(i.series_idx, i.scored) for i in insts if not i.scored] == [(1 * V + 2, False)]
    uni = list(EG.gift.iter_test_instances(_items(), H, 2))
    assert len(uni) == len(insts) - 1


def test_flags_and_load_config(monkeypatch):
    import evaluate_gift as EG
    assert "multivariate" in EG.KNOWN_FLAGS
    with pytest.raises(ValueError, match="forecast_multivariate"):
        EG.check_model_flags(object(), "off", False, None, None, multivariate=True)
    no_layers = type("M", (), {"forecast_multivariate": lambda self: None, "variate_layers": ()})()
    with pytest.raises(ValueError, match="variate_layers"):
        EG.check_model_flags(no_layers, "off", False, None, None, multivariate=True)
    EG.check_model_flags(_Independent(), "mix", False, None, None, multivariate=True)
    series = _items()
    monkeypatch.setattr(EG.gift, "load_series_grouped", lambda root, cfg: (series, V))
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: H)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    data = EG.gift.load_config(Path("."), "mv/H/short", max_series=7, multivariate=True)
    assert data.n_variates == V and len(data.series) == 6
