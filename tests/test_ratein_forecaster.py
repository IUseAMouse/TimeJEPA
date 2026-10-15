"""
RateINForecaster: the RateIN layer in front of a frozen forecaster, behind
the forecaster interface (`timejepa.models.ratein_forecaster`).

Pinned on a stub inner (seasonal naive plus a fixed fan, records its calls):
1. Prior k = 1 on a context of exactly model_len: every output of the
   wrapper equals the inner's, bit for bit; a given w reaches the inner
   unchanged, a missing w stays missing.
2. Prior k = 2: `quantiles_denorm` equals numpy decimate + inner +
   reinterp_fan; the inner is asked ceil(h / 2) steps and w * 2.
3. Gradients reach the selector only; the inner is frozen and stays in eval
   through `train()`.
4. The knob rule: passthrough without a knob, 1/8-octave grid, clamp, number
   of distinct w bounded by distinct w x distinct k_q.
5. Hooks: representations and pretrain mode refused, freeze_* no-ops,
   parameter counts.
6. `FinetuneModule` trains the wrapper: one training_step, optimizer groups,
   the rate diagnostics logged.
7. `build_from_config` with an inner builder of this module, with and
   without an inner checkpoint; a geometry mismatch is refused.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa import ratein_layer as RL  # noqa: E402
from timejepa.evaluation import ratein as R  # noqa: E402
from timejepa.models.decoders.quantile_head import pinball_loss  # noqa: E402
from timejepa.models.ratein_forecaster import RateINForecaster, build_from_config  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]
PERIOD = 24
H = 48


class _Head:
    quantile_levels = tuple(LEVELS)

    def loss(self, quantiles, target, mask=None):
        return pinball_loss(quantiles, target, self.quantile_levels, mask=mask)

    def median(self, q):
        return q[..., 4:5]


class _Inner(nn.Module):
    """Seasonal naive on its own grid with a fixed fan around it; w scales
    nothing (recorded only). One parameter, so freezing is observable."""
    rate_knob = "delta"
    delta_range = (1.0 / 48.0, 4.0)
    expects_frequency = False

    def __init__(self, knob=True, prediction_length=H):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.prediction_length = prediction_length
        self.input_length = 1024
        self.patching = SimpleNamespace(stride=1, patch_size=1)
        self.predictor = SimpleNamespace(w_film=None)
        self.revin = None
        self.robust_scaler = None
        self.decoder = SimpleNamespace(decoder=_Head())
        self.calls = []
        if not knob:
            self.rate_knob = None

    def set_pretrain_mode(self, mode=True):
        pass

    def forecast(self, context, n=None, w=None, **kw):
        x = context[..., 0]
        n = int(n or self.prediction_length)
        self.calls.append({"len": x.shape[1], "n": n, "w": None if w is None else torch.as_tensor(w).clone()})
        reps = n // PERIOD + 1
        med = x[:, -PERIOD:].repeat(1, reps)[:, :n] + 0.0 * self.lin.weight.sum()
        offs = torch.tensor([0.5 * (q - 0.5) for q in LEVELS], dtype=x.dtype)
        fan = med[:, :, None] + offs[None, None, :]
        return {"quantiles": fan, "quantiles_denorm": fan, "quantile_levels": LEVELS,
                "forecast": med[:, :, None], "forecast_denorm": med[:, :, None]}


def _series(B, L, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    rows = [10 + (1 + i) * np.sin(2 * np.pi * t / PERIOD) + 0.2 * rng.standard_normal(L) for i in range(B)]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _wrap(inner=None, model_len=256, k_prior=1.0, **kw):
    inner = inner or _Inner()
    layer = RL.RateINLayer(RL.RateSelector(8, 4), model_len=model_len, min_blocks=8)
    return RateINForecaster(inner, layer, log_k_prior=math.log(k_prior), **kw)


# ------------------------------------------------------------- 1. identity
def test_prior_one_is_the_inner_bit_for_bit():
    w = _wrap()
    x = _series(3, 256).unsqueeze(-1)
    ref = w.inner.forecast(x, n=H)
    out = w.forecast(x, n=H)
    for key in ("quantiles", "quantiles_denorm", "forecast", "forecast_denorm"):
        assert torch.equal(out[key], ref[key]), key
    assert out["quantile_levels"] == LEVELS
    assert w.inner.calls[-1]["w"] is None                     # missing w stays missing
    w.forecast(x, n=H, w=torch.tensor([3.0, 0.5, 1.0]))
    assert torch.equal(w.inner.calls[-1]["w"], torch.tensor([3.0, 0.5, 1.0]))
    assert w.input_length == 256 * RL.K_MAX and w.prediction_length == H


# ------------------------------------------------------------ 2. decimation
def test_prior_two_matches_numpy_decimate_and_reinterp():
    w = _wrap(k_prior=2.0)
    x = _series(2, 3000)
    out = w.forecast(x.unsqueeze(-1), n=H, w=torch.ones(2))
    assert w.inner.calls[-1]["n"] == H // 2
    torch.testing.assert_close(w.inner.calls[-1]["w"], torch.full((2,), 2.0))
    for b in range(2):
        hist = R.resample_context(x[b].numpy().astype(np.float64), 2, 256)
        ref_in = _Inner().forecast(torch.tensor(hist, dtype=torch.float32)[None, :, None], n=H // 2)
        ref = R.to_native_fan(ref_in["quantiles_denorm"][0].detach().numpy(), H, 2)
        np.testing.assert_allclose(out["quantiles_denorm"][b].detach().numpy(), ref, rtol=1e-6, atol=1e-5)


# -------------------------------------------------------------- 3. gradients
def test_gradients_reach_the_selector_only_and_inner_stays_eval():
    w = _wrap()
    with torch.no_grad():
        w.layer.selector.mlp[-1].weight.fill_(0.05)
    w.train()
    assert w.training and not w.inner.training
    x = _series(4, 900).unsqueeze(-1)
    out = w.forecast(x, n=H)
    target = _series(4, H, seed=5)
    loss = pinball_loss(out["quantiles"], target, LEVELS)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.layer.selector.parameters())
    assert all(not p.requires_grad and p.grad is None for p in w.inner.parameters())
    assert len(set(w.last_k.tolist())) > 1


# ------------------------------------------------------------- 4. knob rule
def test_inner_w_rule():
    w = _wrap()
    k = torch.tensor([1.0, 1.05, 2.0, 2.9, 40.0, 0.3])
    wi = w._inner_w(torch.ones(6), k)
    step = math.log(2) / 8
    assert torch.allclose(torch.log(wi[:4]) / step, torch.round(torch.log(wi[:4]) / step), atol=1e-5)
    assert wi[0].item() == 1.0 and wi[2].item() == pytest.approx(2.0)
    assert wi[4].item() == 4.0 and wi[5].item() == pytest.approx(0.3, rel=0.1)      # clamped / near
    assert w._inner_w(None, torch.ones(3)) is None
    w2 = _wrap(inner=_Inner(knob=False))
    assert w2._inner_w(None, k) is None and torch.equal(w2._inner_w(torch.ones(6), k), torch.ones(6))
    base = torch.tensor([1.0, 1.0, 0.25, 0.25])
    kk = torch.tensor([1.0, 2.0, 1.0, 2.0])
    assert torch.unique(w._inner_w(base, kk)).numel() <= 4


# ------------------------------------------------------------------ 5. hooks
def test_hooks():
    w = _wrap()
    with pytest.raises(NotImplementedError):
        w.forecast(_series(1, 300).unsqueeze(-1), n=H, return_representations=True)
    with pytest.raises(ValueError):
        w.set_pretrain_mode(True)
    w.set_pretrain_mode(False)
    before = [p.requires_grad for p in w.parameters()]
    w.freeze_encoder(); w.unfreeze_encoder(); w.freeze_predictor(); w.unfreeze_patching()
    assert [p.requires_grad for p in w.parameters()] == before
    parts = w.get_num_params()
    assert parts["selector"] == sum(p.numel() for p in w.layer.parameters())
    assert parts["total"] == parts["selector"] + 2
    assert w.count_parameters() == parts["selector"]
    assert w.rate_diagnostics() == {}
    w.forecast(_series(2, 300).unsqueeze(-1), n=H)
    assert set(w.rate_diagnostics()) >= {"k_median", "frac_k_gt1", "g_mean", "model_horizon"}


# --------------------------------------------------------- 6. Lightning step
def test_finetune_module_trains_the_wrapper():
    from timejepa.training.finetune_module import FinetuneModule
    w = _wrap()
    with torch.no_grad():
        w.layer.selector.mlp[-1].weight.fill_(0.05)
    module = FinetuneModule(w, finetune_mode="full_finetune", lr_scheduler="constant", learning_rate=1e-3)
    logged = {}
    module.log = lambda name, value, *a, **k: logged.__setitem__(name, float(torch.as_tensor(value).detach()))
    batch = {"context": _series(4, 700), "target": _series(4, H, seed=9)}
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in w.layer.selector.parameters())
    opt = module.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    elif isinstance(opt, (tuple, list)):
        opt = opt[0][0] if isinstance(opt[0], (tuple, list)) else opt[0]
    names = [g["name"] for g in opt.param_groups]
    assert "encoder" in names and "decoder" not in names
    assert "rate/k_median" in logged and "rate/frac_k_gt1" in logged


# --------------------------------------------------------------- 7. builder
def build_stub(cfg):
    return _Inner(prediction_length=int(cfg.model.prediction_length))


def _cfg(tmp_path, inner_checkpoint=None, model_len=256):
    return OmegaConf.create({"model": {
        "name": "stub_ratein", "builder": "timejepa.models.ratein_forecaster:build_from_config",
        "seq_length": 3000, "prediction_length": H,
        "ratein_layer": {"inner_builder": "test_ratein_forecaster:build_stub",
                         "inner_checkpoint": inner_checkpoint, "model_len": model_len,
                         "n_modes": 8, "hidden": 4, "min_blocks": 8, "k_prior": 2.0}}})


def test_build_from_config_and_checkpoint_round_trip(tmp_path):
    from timejepa.evaluation.loading import create_model_from_config, load_checkpoint
    w = create_model_from_config(_cfg(tmp_path))
    assert isinstance(w, RateINForecaster) and w.name == "stub_ratein"
    assert w.inner.input_length == 1024 and w.layer.model_len == 256 and w.log_k_prior == pytest.approx(math.log(2))
    # an inner checkpoint in the Lightning format is loaded through load_checkpoint
    inner = _Inner()
    with torch.no_grad():
        inner.lin.weight.fill_(7.0)
    torch.save({"state_dict": {"model." + k: v for k, v in inner.state_dict().items()}}, tmp_path / "inner.ckpt")
    w2 = create_model_from_config(_cfg(tmp_path, inner_checkpoint=str(tmp_path / "inner.ckpt")))
    assert w2.inner.lin.weight.item() == 7.0 and not w2.inner.lin.weight.requires_grad
    # the wrapper's own Lightning checkpoint round-trips, and a mismatching layer is refused
    with torch.no_grad():
        w2.layer.selector.mlp[-1].weight.fill_(0.03)
    torch.save({"state_dict": {"model." + k: v for k, v in w2.state_dict().items()}}, tmp_path / "wrap.ckpt")
    w3 = load_checkpoint(create_model_from_config(_cfg(tmp_path)), str(tmp_path / "wrap.ckpt"), torch.device("cpu"))
    x = _series(2, 900).unsqueeze(-1)
    torch.testing.assert_close(w3.forecast(x, n=H)["quantiles"], w2.forecast(x, n=H)["quantiles"])
    bad = create_model_from_config(_cfg(tmp_path, model_len=256))
    bad.layer.selector = RL.RateSelector(16, 4)                       # other geometry
    with pytest.raises(RuntimeError):
        load_checkpoint(bad, str(tmp_path / "wrap.ckpt"), torch.device("cpu"))
