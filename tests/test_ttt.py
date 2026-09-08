"""JEPA test-time adaptation on the lookbacks (2026-09-08).

Pinned: parameter selection; the self-supervised loss on synthetic lookbacks
decreases under adaptation and the ORIGINAL model is untouched; the harness
path runs with the causal gate and records its diagnostics; flags parse.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.evaluation import ttt as T  # noqa: E402
from timejepa.models import JEPATST  # noqa: E402


def _model():
    torch.manual_seed(0)
    return JEPATST(input_length=512, prediction_length=128, patch_size=16,
                   stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                   predictor_num_layers=1, predictor_num_heads=4,
                   predictor_d_ff=64, decoder_type="quantile").eval()


def _lookbacks(n=16, L=512, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = np.arange(L)
        out.append((np.sin(2 * np.pi * t / rng.uniform(20, 60)) * rng.uniform(1, 5)
                    + rng.normal(scale=0.2, size=L)).astype(np.float32))
    return out


def test_param_selection():
    m = _model()
    n_all = len(T.select_params(m, "all"))
    n_norm = len(T.select_params(m, "norm"))
    assert 0 < n_norm < n_all
    assert not any(n.startswith("decoder") for n, _ in m.named_parameters()
                   if id(_) in {id(p) for p in T.select_params(m, "all")})
    with pytest.raises(ValueError):
        T.select_params(m, "nope")


def test_adapt_lowers_the_loss_and_leaves_the_original_untouched():
    m = _model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ctxs = _lookbacks()
    m2, st = T.adapt(m, ctxs, torch.device("cpu"), steps=30, lr=1e-3, params="all", batch_size=8)
    assert st["steps"] == 30 and st["n_contexts"] == 16
    assert st["loss_mean_last_quarter"] < st["loss_mean_first_quarter"]
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])
    assert any(not torch.equal(a, b) for a, b in zip(m2.state_dict().values(), before.values()))
    assert not any(p.requires_grad for p in m2.parameters())
    m3, st3 = T.adapt(m, ctxs, torch.device("cpu"), steps=0)
    assert st3["steps"] == 0
    m4, st4 = T.adapt(m, [np.ones(100, dtype=np.float32)], torch.device("cpu"), steps=5)
    assert st4["n_contexts"] == 0 and st4["steps"] == 0        # too short: skipped


def test_harness_path_with_gate(monkeypatch):
    import evaluate_gift as EG
    rng = np.random.default_rng(1)
    series = []
    for _ in range(6):
        t = np.arange(1400)
        series.append((10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.5, size=1400)).astype(np.float32))
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 48)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: 24)
    model = _model()
    spec = {"params": "norm", "steps": 4, "lr": 1e-3, "gate": True, "margin": 0.01, "seed": 0}
    res = EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                             batch_size=8, ttt_spec=spec)
    assert "ttt" in res and "gate_ratio" in res["ttt"] and res["ttt"]["official"] is True
    assert isinstance(res["ttt"]["accepted"], bool)
    res2 = EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=8, ttt_spec=dict(spec, gate=False))
    assert res2["ttt"]["accepted"] is True and res2["ttt"]["steps"] == 4
    plain = EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"), batch_size=8)
    assert "ttt" not in plain
    from evaluate_gift import check_unknown_flags
    check_unknown_flags(["+ttt=norm", "+ttt_steps=8", "+ttt_lr=1e-4", "+ttt_gate=false"])
