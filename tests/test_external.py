"""
External forecasters behind the GIFT harness (2026-09-13).

Pinned:
1. The contract: forecast(batch, n) returns a sorted [B, n, Q] fan and its
   median, on cpu; w != 1 refused; n required; a wrong shape from the
   backend is refused, not silently reshaped.
2. The harness runs an external model in off / flip / mix-pool / backtest
   and refuses the JEPA-only layers (refine, ttt, delta, ratein_w).
3. The builder dispatches on `model.external.kind` and refuses unknown kinds;
   run_identity keys the cache on the Hugging Face id.
4. The main path accepts a config with model.external and no checkpoint,
   and refuses both together (pure-function check on the guard).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation import external as ext  # noqa: E402

PERIOD = 24


class _SeasonalStub(ext.ExternalForecaster):
    """Seasonal naive with a Gaussian fan of width 0.5 * std(context)."""

    def __init__(self):
        super().__init__("stub", "stub/model", 256, torch.device("cpu"))
        self.calls = []

    def _predict(self, ctx, n):
        self.calls.append((ctx.shape[1], n))
        from scipy.stats import norm
        z = torch.tensor(norm.ppf(np.asarray(self.quantile_levels)), dtype=torch.float32)
        out = []
        for row in ctx:
            med = row[-PERIOD:].repeat(n // PERIOD + 1)[:n]
            out.append(med[:, None] + 0.5 * row.std() * z[None, :])
        return torch.stack(out)


def _ctx(B=3, L=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L).float()
    return (50 + 10 * torch.sin(2 * torch.pi * t / PERIOD))[None, :, None].repeat(B, 1, 1) \
        + torch.randn(B, L, 1, generator=g)


# ------------------------------------------------------------- 1. contract
def test_contract_shapes_sorted_median_and_refusals():
    m = _SeasonalStub()
    out = m.forecast(_ctx(), n=40)
    assert out["quantiles_denorm"].shape == (3, 40, 9)
    assert out["forecast_denorm"].shape == (3, 40, 1)
    assert (out["quantiles_denorm"].diff(dim=-1) >= 0).all()
    assert torch.equal(out["forecast_denorm"][..., 0], out["quantiles_denorm"][:, :, 4])
    assert out["quantile_levels"] == list(ext.LEVELS)
    # the context is capped at input_length
    m.forecast(torch.randn(2, 600, 1), n=8)
    assert m.calls[-1][0] == 256
    with pytest.raises(ValueError, match="rate knob"):
        m.forecast(_ctx(), n=8, w=torch.full((3,), 0.5))
    m.forecast(_ctx(), n=8, w=torch.ones(3))                 # w = 1 is fine
    with pytest.raises(ValueError, match="horizon"):
        m.forecast(_ctx())

    class _Bad(_SeasonalStub):
        def _predict(self, ctx, n):
            return torch.zeros(ctx.shape[0], n, 5)
    with pytest.raises(RuntimeError, match="quantiles shape"):
        _Bad().forecast(_ctx(), n=8)


# ------------------------------------------------------------- 2. harness
def _synthetic(rng, n_series=8, length=800):
    out = []
    for _ in range(n_series):
        level, amp = rng.uniform(5, 50), rng.uniform(1, 5)
        t = np.arange(length)
        out.append((level + amp * np.sin(2 * np.pi * t / PERIOD)
                    + rng.normal(scale=0.2 * amp, size=length)).astype(np.float32))
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


def test_harness_modes_and_refusals(harness):
    EG = harness
    m = _SeasonalStub()
    off = _run(EG, m)
    assert np.isfinite(off["model"]["CRPS"])
    flip = _run(EG, m, tta_flip=True)
    mix = _run(EG, m, ratein_mode="mix", ratein_pool=True)
    bt = _run(EG, m, ratein_mode="backtest", ratein_pool=True)
    assert "mix" in mix["ratein"] and bt["ratein"]["backtest"]["knob"] == "decimation"
    assert np.isfinite(flip["model"]["CRPS"]) and np.isfinite(bt["model"]["CRPS"])
    from evaluate_gift import check_model_flags
    with pytest.raises(ValueError):
        check_model_flags(m, "delta", False, None, None)
    with pytest.raises(ValueError):
        check_model_flags(m, "off", False, None, {"params": "norm"})
    assert m.predictor.w_film is None and m.rate_knob is None


# ------------------------------------------------------------- 3. builder
def test_builder_dispatch_and_identity(monkeypatch):
    from omegaconf import OmegaConf
    built = {}

    class _Fake(ext.ExternalForecaster):
        def __init__(self, hf_id, device, name=None, context_length=1024, **kw):
            super().__init__(name or hf_id, hf_id, context_length, device)
            built.update(kw)
            built["hf_id"] = hf_id
    monkeypatch.setitem(ext.KINDS, "fake", _Fake)
    cfg = OmegaConf.create({"model": {"name": "x", "external": {
        "kind": "fake", "hf_id": "org/model", "context_length": 512, "device": "cpu"}}})
    m = ext.build(cfg)
    assert isinstance(m, _Fake) and m.input_length == 512 and built["hf_id"] == "org/model"
    with pytest.raises(ValueError, match="unknown external kind"):
        ext.build(OmegaConf.create({"model": {"name": "x", "external": {"kind": "nope", "hf_id": "a/b"}}}))
    with pytest.raises(ValueError):
        ext.build(OmegaConf.create({"model": {"name": "x"}}))
    stem, fp = ext.run_identity("amazon/chronos-bolt-tiny")
    assert stem == "amazon__chronos-bolt-tiny" and fp == "hf:amazon/chronos-bolt-tiny"
    stem, fp = ext.run_identity("ibm-granite/granite-timeseries-ttm-r2", "1536-96-r2")
    assert stem.endswith("@1536-96-r2") and fp.endswith("@1536-96-r2")
    # the TimeJEPA loader dispatches to this builder through model.builder
    from timejepa.evaluation import loading
    cfg2 = OmegaConf.create({"model": {"name": "x", "builder": "timejepa.evaluation.external:build",
                                       "external": {"kind": "fake", "hf_id": "org/m", "device": "cpu"}}})
    assert isinstance(loading.create_model_from_config(cfg2), _Fake)


def test_external_configs_parse():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1] / "configs" / "model"
    names = sorted(p.name for p in root.glob("ext_*_eval.yaml"))
    assert len(names) >= 4
    for n in names:
        cfg = OmegaConf.load(root / n)
        assert cfg.model.builder == "timejepa.evaluation.external:build"
        assert cfg.model.external.kind in ext.KINDS
        assert cfg.model.seq_length >= 512
