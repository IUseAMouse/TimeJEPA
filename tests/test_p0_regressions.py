"""
Regression tests for the P0 fixes.

Each test here pins down a bug that silently corrupted results before. They are
cheap and should stay green forever; if one fails, a real invariant broke.

Covered:
  B1  package imports (patchtst_encoder removal left dangling imports)
  B2  normalization contract of `forecast()` (global z-score != instance norm)
  B3  RevIN affine consistency between the target space and denormalization
  B10 rolling forecast: `revin.freeze()` existed nowhere; spaces were mixed
  P0.4/P0.5 baselines and scale-free metrics
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.models import JEPATST                                      # noqa: E402
from timejepa.models.components.revin import RevIN                       # noqa: E402
from timejepa.training.utils.baselines import (                          # noqa: E402
    seasonal_naive_forecast,
    last_value_forecast,
    mean_forecast,
    linear_trend_forecast,
    compute_all_baselines,
    get_seasonality,
)
from timejepa.training.utils.metrics import (                            # noqa: E402
    mase,
    nd,
    weighted_quantile_loss,
    compute_forecasting_metrics_extended,
)


# =============================================================================
# B1 — package is importable
# =============================================================================

def test_package_imports():
    """`import timejepa.models` used to raise ModuleNotFoundError."""
    import timejepa.models as m
    import timejepa.data          # noqa: F401
    import timejepa.training      # noqa: F401
    assert hasattr(m, "JEPATST")
    assert hasattr(m, "BareTransformerEncoder")


# =============================================================================
# B3 — RevIN spaces
# =============================================================================

@pytest.fixture
def revin():
    r = RevIN(num_features=1, affine=True)
    with torch.no_grad():
        r.affine_weight.fill_(0.94)   # values in this range are what the
        r.affine_bias.fill_(0.025)    # released checkpoints actually learned
    return r


def test_denormalize_target_space_is_exact_inverse(revin):
    """
    The decoder is trained against a plain z-scored target, so its inverse must
    NOT undo the RevIN affine. `_denormalize` does undo it — that mismatch is a
    ~6-10% scale error plus a constant offset on every forecast.
    """
    x = torch.randn(4, 100, 1) * 5 + 3
    _ = revin(x, mode="norm")

    z_target = (x - revin.mean) / revin.std
    assert torch.allclose(revin.denormalize_target_space(z_target), x, atol=1e-4)

    # And the buggy path is measurably different — this is the bug, pinned.
    wrong = revin(z_target, mode="denorm")
    assert not torch.allclose(wrong, x, atol=1e-2)


def test_to_input_frame_matches_normalize(revin):
    """to_input_frame(z_target) must reproduce exactly what _normalize emits."""
    x = torch.randn(4, 100, 1) * 5 + 3
    z_input = revin(x, mode="norm")
    z_target = (x - revin.mean) / revin.std
    assert torch.allclose(revin.to_input_frame(z_target), z_input, atol=1e-5)


def test_revin_freeze_pins_statistics(revin):
    """`freeze()` did not exist, so any rolling forecast raised AttributeError."""
    x = torch.randn(4, 100, 1) * 5 + 3
    _ = revin(x, mode="norm")
    mean0, std0 = revin.mean.clone(), revin.std.clone()

    revin.freeze()
    assert revin.is_frozen
    _ = revin(torch.randn(4, 100, 1) * 999, mode="norm")
    assert torch.equal(revin.mean, mean0)
    assert torch.equal(revin.std, std0)

    revin.unfreeze()
    assert not revin.is_frozen
    _ = revin(torch.randn(4, 100, 1) * 999, mode="norm")
    assert not torch.equal(revin.mean, mean0)


# =============================================================================
# B10 — rolling forecast
# =============================================================================

@pytest.fixture(scope="module")
def small_model():
    m = JEPATST(
        input_length=384, prediction_length=96,
        patch_size=16, stride=8,
        d_model=32, num_layers=1, num_heads=4, d_ff=64,
        predictor_num_layers=1, predictor_num_heads=4, predictor_d_ff=64,
        decoder_type="mlp",
    )
    m.eval()
    return m


@pytest.mark.parametrize("n", [48, 96, 192, 336, 720])
def test_rollout_shapes_and_finiteness(small_model, n):
    ctx = torch.randn(3, 384, 1) * 10 + 50
    with torch.no_grad():
        out = small_model.forecast(ctx, n=n)
    assert out["forecast"].shape == (3, n, 1)
    assert out["forecast_denorm"].shape == (3, n, 1)
    assert torch.isfinite(out["forecast_denorm"]).all()


def test_rollout_is_level_anchored(small_model):
    """
    The denormalized forecast must live on the same scale as the context. The
    old rollout concatenated a normalized forecast onto a raw-space context,
    which drove the output far away from the input's scale.
    """
    ctx = torch.randn(4, 384, 1) * 3 + 500.0
    with torch.no_grad():
        out = small_model.forecast(ctx, n=336)
    pred_mean = out["forecast_denorm"].mean().item()
    ctx_mean = ctx.mean().item()
    assert abs(pred_mean - ctx_mean) < 10 * ctx.std().item(), (
        f"forecast mean {pred_mean:.2f} detached from context mean {ctx_mean:.2f}"
    )


def test_rollout_leaves_revin_unfrozen(small_model):
    """A leaked frozen RevIN would silently corrupt every subsequent batch."""
    ctx = torch.randn(2, 384, 1)
    with torch.no_grad():
        small_model.forecast(ctx, n=336)
    assert not small_model.revin.is_frozen


def test_single_shot_and_rolling_agree_on_first_window(small_model):
    """forecast(n=96) must equal the first 96 steps of forecast(n=192)."""
    ctx = torch.randn(2, 384, 1) * 4 + 20
    with torch.no_grad():
        a = small_model.forecast(ctx, n=96)["forecast_denorm"]
        b = small_model.forecast(ctx, n=192)["forecast_denorm"][:, :96]
    assert torch.allclose(a, b, atol=1e-4)


# =============================================================================
# P0.4 — baselines
# =============================================================================

















# =============================================================================
# P0.5 — scale-free metrics
# =============================================================================











# =============================================================================
# B17 — a too-short dataset must not kill a multi-dataset run
# =============================================================================



















# =============================================================================
# G5 — LOTSA integration must be purely additive
# =============================================================================

























# =============================================================================
# B22 — uniform-length survivors of the length filter became object arrays
# =============================================================================









# =============================================================================
# B21 / config hygiene — the experiment grid must be declarative
# =============================================================================







# =============================================================================
# B20 — gradual_unfreeze never actually trained anything but the decoder
# =============================================================================

def _step(module, optimizer):
    module.model.train()
    loss, _, _ = module._forward_and_loss(torch.randn(4, 512, 1), torch.randn(4, 128, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _snapshot(model, prefix):
    return {n: p.clone() for n, p in model.named_parameters() if n.startswith(prefix)}


def _moved(model, before):
    return any(not torch.equal(before[n], p)
               for n, p in model.named_parameters() if n in before)


def test_optimizer_registers_frozen_params_so_unfreezing_works():
    """
    THE B20 bug. The optimizer is created once, at epoch 0, when
    gradual_unfreeze has everything frozen. Filtering on requires_grad at that
    moment meant the later unfreeze flipped the flag and gradients flowed, but
    optimizer.step() silently never touched those weights: gradual_unfreeze
    trained the decoder alone for the entire run — in every run that used it,
    including the historical best checkpoints.
    """
    torch.manual_seed(0)
    mod = _finetune_module(finetune_mode="gradual_unfreeze", unfreeze_after_epoch=0,
                           lr_scheduler="constant", learning_rate=1e-2)
    opt = mod.configure_optimizers()

    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    m = mod.model
    assert all(id(p) in in_opt for p in m.predictor.parameters())
    assert all(id(p) in in_opt for p in m.online_encoder.parameters())
    assert all(id(p) in in_opt for p in m.patching.parameters())
    assert not any(id(p) in in_opt for p in m.target_encoder.parameters())

    # Phase 1 — still frozen: a step must move the decoder and nothing else
    enc0 = _snapshot(m, "online_encoder")
    pred0 = _snapshot(m, "predictor")
    dec0 = _snapshot(m, "decoder")
    _step(mod, opt)
    assert _moved(m, dec0)
    assert not _moved(m, enc0)
    assert not _moved(m, pred0)

    # Phase 2 — the epoch hook fires (detached module: current_epoch == 0)
    mod.on_train_epoch_start()
    enc1 = _snapshot(m, "online_encoder")
    pred1 = _snapshot(m, "predictor")
    _step(mod, opt)
    assert _moved(m, enc1), "encoder still frozen after gradual unfreeze"
    assert _moved(m, pred1), "predictor unfrozen but not updated by the optimizer"


def test_linear_probe_still_trains_only_the_decoder():
    """Registering frozen params must not leak training into a probe."""
    torch.manual_seed(0)
    mod = _finetune_module(finetune_mode="linear_probe",
                           lr_scheduler="constant", learning_rate=1e-2)
    opt = mod.configure_optimizers()
    m = mod.model
    enc0 = _snapshot(m, "online_encoder")
    pred0 = _snapshot(m, "predictor")
    _step(mod, opt)
    assert not _moved(m, enc0)
    assert not _moved(m, pred0)


# =============================================================================
# Geometry round — finetune-side context randomization
# =============================================================================

def _finetune_module(**kw):
    from timejepa.training.finetune_module import FinetuneModule
    m = JEPATST(input_length=512, prediction_length=128, patch_size=16, stride=8,
                d_model=32, num_layers=1, num_heads=4, d_ff=64,
                predictor_num_layers=1, predictor_num_heads=4, predictor_d_ff=64,
                decoder_type="mlp")
    kw.setdefault("finetune_mode", "linear_probe")
    return FinetuneModule(model=m, **kw)


def test_finetune_crops_context_from_the_left():
    """Keep the most recent history — what a short context contains at inference."""
    mod = _finetune_module(context_lengths=[128, 256], p_random_context_finetune=1.0)
    torch.manual_seed(0)
    ctx = torch.arange(512.0).view(1, 512, 1).repeat(3, 1, 1)
    cropped = mod._maybe_crop_context(ctx)
    assert cropped.shape[1] in (128, 256)
    # Left crop: the LAST timestep must survive
    assert torch.equal(cropped[:, -1], ctx[:, -1])


def test_finetune_context_randomization_is_off_by_default():
    """
    Existing finetune configs must keep their exact previous behavior: the
    probability key defaults to 0.0, so nothing changes unless a config opts in.
    """
    mod = _finetune_module()
    ctx = torch.randn(3, 512, 1)
    assert mod._maybe_crop_context(ctx).shape[1] == 512

    # Even with lengths configured, p=0 must be a no-op
    mod2 = _finetune_module(context_lengths=[128], p_random_context_finetune=0.0)
    assert mod2._maybe_crop_context(ctx).shape[1] == 512


def test_finetune_crop_never_upsamples():
    """A context already shorter than every option must pass through unchanged."""
    mod = _finetune_module(context_lengths=[256, 512], p_random_context_finetune=1.0)
    ctx = torch.randn(2, 192, 1)
    assert mod._maybe_crop_context(ctx).shape[1] == 192


# =============================================================================
# P2.1 — quantile head, and backward compatibility with pre-quantile models
# =============================================================================

def _model(decoder_type="mlp"):
    from timejepa.models.decoders import ForecastingHead
    m = JEPATST(input_length=512, prediction_length=128, patch_size=32, stride=16,
                d_model=64, num_layers=2, num_heads=4, d_ff=128,
                predictor_num_layers=2, predictor_num_heads=4, predictor_d_ff=128,
                decoder_type="mlp")
    if decoder_type == "quantile":
        m.decoder = ForecastingHead(
            d_model=64, patch_size=32, stride=16, prediction_length=128,
            num_features=1, decoder_type="quantile", revin=m.revin,
        )
    return m


@pytest.mark.parametrize("decoder_type", ["mlp", "linear", "attentive"])
def test_point_decoders_are_untouched(decoder_type):
    """Adding the quantile branch must not perturb any existing decoder."""
    m = JEPATST(input_length=512, prediction_length=128, patch_size=32, stride=16,
                d_model=64, num_layers=2, num_heads=4, d_ff=128,
                predictor_num_layers=2, predictor_num_heads=4, predictor_d_ff=128,
                decoder_type=decoder_type)
    m.eval()
    ctx = torch.randn(3, 512, 1) * 4 + 30
    with torch.no_grad():
        out = m.forward_finetune(ctx)
        rolled = m.forecast(ctx, n=336)
    assert out["forecast"].shape == (3, 128, 1)
    assert "quantiles" not in out
    assert rolled["forecast_denorm"].shape == (3, 336, 1)
    assert torch.isfinite(rolled["forecast_denorm"]).all()


def test_quantile_head_is_monotone_by_construction():
    """
    Independently regressed quantiles can cross. The head predicts the median
    plus softplus widths accumulated outward, so sorting is a property of the
    parameterization — checked here under deliberately extreme raw outputs.
    """
    m = _model("quantile")
    head = m.decoder.decoder
    torch.manual_seed(0)
    for scale in (1.0, 50.0, 500.0):
        mono = head._make_monotone(torch.randn(4, 128, 9) * scale)
        assert (mono.diff(dim=-1) >= 0).all(), f"crossing at scale {scale}"


def test_quantile_head_exposes_median_as_point_forecast():
    m = _model("quantile")
    m.eval()
    with torch.no_grad():
        out = m.forward_finetune(torch.randn(3, 512, 1) * 4 + 30)
    assert out["quantiles"].shape == (3, 128, 9)
    assert out["forecast"].shape == (3, 128, 1)
    mid = m.decoder.decoder.median_idx
    assert torch.equal(out["forecast"].squeeze(-1), out["quantiles"][..., mid])


def test_pinball_is_minimised_by_the_true_quantiles():
    """A sanity check on the loss itself, not just its shape."""
    from timejepa.models.decoders import pinball_loss, DEFAULT_QUANTILES
    from scipy import stats
    torch.manual_seed(0)
    y = torch.randn(4000, 1, 1)
    truth = torch.tensor([stats.norm.ppf(q) for q in DEFAULT_QUANTILES]).float()
    truth = truth.view(1, 1, 9).repeat(4000, 1, 1)
    wrong = torch.zeros(4000, 1, 9)
    assert pinball_loss(truth, y, DEFAULT_QUANTILES) < pinball_loss(wrong, y, DEFAULT_QUANTILES)


def test_pre_quantile_checkpoint_loads_into_a_quantile_model():
    """
    THE compatibility case. A point decoder and the quantile head both own
    `decoder.decoder.unpatching.projection`, sized patch*1 versus patch*9.
    load_state_dict(strict=False) tolerates missing keys but NOT shape
    mismatches, so this combination raised outright — blocking the very workflow
    the head exists for: reuse a pretrained encoder, relearn the head.
    """
    from timejepa.models.jepa_tst import filter_loadable
    old_sd = _model("mlp").state_dict()
    qm = _model("quantile")

    with pytest.raises(RuntimeError, match="size mismatch"):
        qm.load_state_dict(old_sd, strict=False)

    filtered, dropped = filter_loadable(qm, old_sd)
    assert any("unpatching.projection" in k for k, _, _ in dropped)

    missing, _ = qm.load_state_dict(filtered, strict=False)
    assert [k for k in missing if not k.startswith("decoder.")] == []


def test_transferred_encoder_is_bit_identical():
    """Reusing a checkpoint must actually reuse it, not silently reinitialise."""
    from timejepa.models.jepa_tst import filter_loadable
    old = _model("mlp")
    qm = _model("quantile")
    filtered, _ = filter_loadable(qm, old.state_dict())
    qm.load_state_dict(filtered, strict=False)
    for (_, a), (_, b) in zip(old.online_encoder.state_dict().items(),
                              qm.online_encoder.state_dict().items()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("n", [48, 96, 128, 192, 336, 720])
def test_quantile_fan_survives_truncation_and_rollout(n):
    """
    Two ways the fan used to be lost. Single-shot truncated `forecast` to n but
    left `quantiles` at prediction_length, silently mismatching them; and the
    rollout collected only the median, so every horizon past 128 had no fan at
    all — leaving nothing to compute a real WQL from.
    """
    m = _model("quantile")
    m.eval()
    with torch.no_grad():
        out = m.forecast(torch.randn(4, 512, 1) * 3 + 10, n=n)
    assert out["forecast_denorm"].shape == (4, n, 1)
    assert out["quantiles_denorm"].shape == (4, n, 9)
    assert (out["quantiles_denorm"].diff(dim=-1) >= 0).all()
    mid = m.decoder.decoder.median_idx
    assert torch.allclose(
        out["forecast_denorm"].squeeze(-1), out["quantiles_denorm"][..., mid], atol=1e-5
    )


def test_sampled_rollout_accumulates_uncertainty():
    """
    The measured defect: median feedback makes every later roll see a context
    smoother than real data, so intervals SHRINK with horizon (exchange h720:
    width 0.267 where truth grows as sqrt(h)).

    The plumbing is validated with a stub decoder that conditions on the level
    of the path it is fed (persistence + a fixed [-1, +1] fan). Under the
    comonotonic coupling the spread must accumulate LINEARLY — widths 2, 4, 6, 8
    across four rolls — while median feedback stays flat at 2. An untrained real
    model cannot show this because its fan does not depend on its input.
    """
    m = _model("quantile")
    m.eval()
    H = 128

    def stub(ctx, skip_revin=True, **kw):
        base = ctx[:, -1:, :].expand(-1, H, -1)
        q = base + torch.linspace(-1.0, 1.0, 9).view(1, 1, 9)
        return {"quantiles": q, "quantiles_denorm": q,
                "quantile_levels": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                "forecast": q[..., 4:5], "forecast_denorm": q[..., 4:5]}

    m.forward_finetune = stub
    ctx = torch.zeros(2, 512, 1)

    def widths(out):
        q = out["quantiles"]
        return [round(float((q[:, i*H:(i+1)*H, -1] - q[:, i*H:(i+1)*H, 0]).mean()), 3)
                for i in range(4)]

    with torch.no_grad():
        sampled = m.forecast(ctx, n=512, skip_revin=True, sample_paths=True)
        median = m.forecast(ctx, n=512, skip_revin=True, sample_paths=False)

    assert widths(sampled) == [2.0, 4.0, 6.0, 8.0]
    assert widths(median) == [2.0, 2.0, 2.0, 2.0]

    # Deterministic: the quantile levels ARE the stratified sample, no RNG.
    with torch.no_grad():
        again = m.forecast(ctx, n=512, skip_revin=True, sample_paths=True)
    assert torch.equal(sampled["quantiles"], again["quantiles"])


def test_sampled_rollout_matches_single_shot_within_native_horizon():
    """sample_paths must be a strict no-op when no rolling happens."""
    m = _model("quantile")
    m.eval()
    torch.manual_seed(0)
    ctx = torch.randn(3, 512, 1) * 3 + 10
    with torch.no_grad():
        a = m.forecast(ctx, n=96, sample_paths=True)
        b = m.forecast(ctx, n=96, sample_paths=False)
    assert torch.equal(a["quantiles_denorm"], b["quantiles_denorm"])


def test_sampled_rollout_first_roll_is_exact_and_output_is_monotone():
    m = _model("quantile")
    m.eval()
    torch.manual_seed(0)
    ctx = torch.randn(3, 512, 1) * 3 + 10
    with torch.no_grad():
        smp = m.forecast(ctx, n=384, sample_paths=True)
        med = m.forecast(ctx, n=384, sample_paths=False)
    # Roll 1 is a single exact forward in both schemes
    assert torch.allclose(smp["quantiles_denorm"][:, :128],
                          med["quantiles_denorm"][:, :128], atol=1e-5)
    assert (smp["quantiles_denorm"].diff(dim=-1) >= -1e-6).all()
    mid = m.decoder.decoder.median_idx
    assert torch.allclose(smp["forecast_denorm"].squeeze(-1),
                          smp["quantiles_denorm"][..., mid], atol=1e-5)
    assert not m.revin.is_frozen


def test_true_wql_differs_from_the_point_wql():
    """
    WQL over a point forecast collapses to ND by construction. If evaluation
    scored the median rather than the fan, the quantile head's entire benefit
    would be invisible in the reported metric.
    """
    from timejepa.training.utils.metrics import weighted_quantile_loss, nd
    torch.manual_seed(0)
    target = torch.randn(32, 96)
    median = target + torch.randn(32, 96) * 0.4
    fan = torch.stack([median + s for s in torch.linspace(-1.2, 1.2, 9)])  # [Q,B,H]

    point_wql = weighted_quantile_loss(median, target).item()
    assert abs(point_wql - nd(median, target).item()) < 1e-5    # the collapse
    assert weighted_quantile_loss(fan, target).item() != pytest.approx(point_wql, abs=1e-4)


def test_quantile_head_requires_context_when_configured_for_it():
    """Option B must fail loudly rather than silently degrade to option A."""
    m = _model("quantile")
    head = m.decoder.decoder
    assert head.use_context
    with pytest.raises(ValueError, match="context_embeddings"):
        head(torch.randn(2, 7, 64), context_embeddings=None)


# =============================================================================
# B13 — JEPATST built its decoder on the wrong stride
# =============================================================================

@pytest.mark.parametrize("patch,stride", [(16, 8), (32, 16), (64, 32), (8, 8)])
def test_internal_decoder_emits_the_full_horizon(patch, stride):
    """
    JEPATST created ForecastingHead without forwarding `stride`, so UnPatching
    reassembled on a default grid of 8. With patch_size=32 the forecast came out
    80 timesteps long instead of 128 — truncated silently, no error.

    Masked in practice because train.py and evaluate.py replace model.decoder,
    but any direct use of JEPATST (the packaged forecast API) got the broken one.
    """
    m = JEPATST(input_length=512, prediction_length=128,
                patch_size=patch, stride=stride,
                d_model=32, num_layers=1, num_heads=4, d_ff=64,
                predictor_num_layers=1, predictor_num_heads=4, predictor_d_ff=64,
                decoder_type="mlp")
    m.eval()
    with torch.no_grad():
        out = m.forward_finetune(torch.randn(2, 512, 1))["forecast"]
    assert out.shape == (2, 128, 1), (
        f"patch={patch}/stride={stride} produced {out.shape[1]} timesteps, expected 128"
    )


@pytest.mark.parametrize("config_name", ["tiny", "tiny_patch32", "tiny_patch64",
                                         "tiny_deep_predictor", "tiny_geo",
                                         "tiny_geo_p32", "tiny_geo_vicreg",
                                         "tiny_geo_scratch", "tiny_geo_lowdata",
                                         "tiny_geo_scratch_lowdata",
                                         "lotsa_tiny", "lotsa_tiny_finetune",
                                         "lotsa_tiny_zeroshot", "lotsa_tiny_eval",
                                         "lotsa_mini", "lotsa_mini_zeroshot",
                                         "lotsa_mini_eval", "lotsa_base",
                                         "lotsa_base_zeroshot", "lotsa_base_eval",
                                         "lotsa_tiny_full"])
def test_experiment_configs_are_runnable(config_name):
    """
    Every shipped config must build a model whose geometry works at the NOMINAL
    size and at every randomized context/horizon it declares. Changing
    patch_length silently makes some of those combinations degenerate (zero
    target patches crashes; too few is meaningless), so this is checked rather
    than reasoned about.
    """
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../configs/model"):
        cfg = compose(config_name=config_name)

    m = JEPATST(
        input_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        patch_size=cfg.model.patch_length, stride=cfg.model.stride,
        d_model=32, num_layers=1, num_heads=4, d_ff=64,
        predictor_num_layers=cfg.model.predictor.n_layers,
        predictor_num_heads=4, predictor_d_ff=64,
        decoder_type=cfg.model.decoder.type,
    )
    m.eval()

    with torch.no_grad():
        for L in cfg.training.context_lengths:
            for H in cfg.training.horizon_lengths:
                out = m.forward_pretrain(torch.randn(2, L, 1), torch.randn(2, H, 1))
                assert out["predictions"].shape == out["targets"].shape
                assert out["predictions"].shape[1] > 0, f"L={L} H={H} gave 0 target patches"

        rolled = m.forecast(torch.randn(2, cfg.model.seq_length, 1), n=336)
        assert rolled["forecast_denorm"].shape == (2, 336, 1)
        assert torch.isfinite(rolled["forecast_denorm"]).all()


# =============================================================================
# B18 — torch version drift in the sampler
# =============================================================================



# =============================================================================
# P1.9 — collapse diagnostics must never kill a run
# =============================================================================

def _pretrain_module(loss_type="sigreg"):
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule
    m = JEPATST(input_length=384, prediction_length=96, patch_size=16, stride=8,
                d_model=32, num_layers=1, num_heads=4, d_ff=64,
                predictor_num_layers=1, predictor_num_heads=4, predictor_d_ff=64,
                decoder_type="mlp")
    return JEPAPretrainModule(model=m, loss_type=loss_type,
                              sigreg_config={"lambda": 1.0})


def test_effective_rank_detects_collapse():
    mod = _pretrain_module()
    torch.manual_seed(0)
    healthy = mod._effective_rank(torch.randn(64, 47, 32))
    collapsed = mod._effective_rank(torch.ones(64, 47, 32) * 3.0)
    rank_one = mod._effective_rank(torch.randn(64, 47, 1) * torch.randn(1, 1, 32))

    assert healthy > 20, f"healthy embeddings should be near full rank, got {healthy}"
    assert collapsed <= 1.01
    assert rank_one <= 1.01


def test_effective_rank_works_under_bf16_autocast():
    """
    Casting the input to float32 is NOT enough: the matmul sits inside the
    bf16-mixed autocast region, so torch casts it straight back to bfloat16 and
    eigvalsh dies with
        "linalg_eigh_cuda" not implemented for 'BFloat16'
    Observed on the first GPU run — the guard turned it into a warning, so the
    metric was silently never reported. autocast has to be disabled explicitly.
    """
    mod = _pretrain_module()
    torch.manual_seed(0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        from_bf16 = mod._effective_rank(torch.randn(64, 47, 32).to(torch.bfloat16))
        from_f32 = mod._effective_rank(torch.randn(64, 47, 32))
    assert from_bf16 is not None, "effective_rank still unavailable under autocast"
    assert from_f32 is not None
    assert from_bf16 > 20 and from_f32 > 20


def test_effective_rank_survives_degenerate_input():
    """
    Iterative eigensolvers genuinely fail on degenerate matrices — verified:
    torch raises "failed to converge (error code: 30)" on all-NaN input. A
    monitoring metric that crashes exactly when the monitored failure occurs
    would be worse than no metric.
    """
    mod = _pretrain_module()
    assert mod._effective_rank(torch.full((64, 47, 32), float("nan"))) is None


def test_context_std_catches_positional_collapse():
    """
    Effective rank pools positions, so a per-position collapse keeps it high.
    `collapse/context_std` is the metric that catches it — they are
    complementary, which is why both are logged.
    """
    mod = _pretrain_module()
    torch.manual_seed(0)
    per_pos = torch.randn(1, 47, 32).repeat(64, 1, 1)
    assert mod._effective_rank(per_pos) > 10          # blind, as expected
    assert per_pos.std(dim=0).mean().item() < 1e-6    # but this one sees it


# =============================================================================
# B16 — predictor future-query table
# =============================================================================

def test_predictor_refuses_to_truncate_future_queries():
    """
    The table used to be a hard 16. Slicing past it returned fewer rows, and the
    downstream `x[:, -num_targets:]` then silently substituted the last CONTEXT
    embeddings for the missing predictions — which were trained and scored as if
    they were real. Affected large.yaml (23 target patches) and base.yaml (32).
    """
    from timejepa.models.predictors.transformer_predictor import TransformerPredictor
    p = TransformerPredictor(d_model=16, num_layers=1, num_heads=2, d_ff=32,
                             max_target_patches=16)
    ctx = torch.randn(2, 47, 16)
    p.forward_simple(ctx, num_targets=16)          # fine
    with pytest.raises(ValueError, match="max_target_patches"):
        p.forward_simple(ctx, num_targets=23)


@pytest.mark.parametrize("pred_len,patch,stride", [(96, 16, 8), (128, 16, 8),
                                                   (192, 16, 8), (128, 4, 4)])
def test_jepatst_sizes_the_query_table_for_its_horizon(pred_len, patch, stride):
    m = JEPATST(input_length=384, prediction_length=pred_len,
                patch_size=patch, stride=stride,
                d_model=16, num_layers=1, num_heads=2, d_ff=32,
                predictor_num_layers=1, predictor_num_heads=2, predictor_d_ff=32,
                decoder_type="mlp")
    m.eval()
    with torch.no_grad():
        out = m.forward_pretrain(torch.randn(2, 384, 1), torch.randn(2, pred_len, 1))
    assert out["predictions"].shape == out["targets"].shape
    assert m.predictor.future_position_embedding.shape[1] >= m.num_target_patches


# =============================================================================
# P1.5 — contextualized targets
# =============================================================================

def test_contextualized_targets_align_with_standalone_patches():
    """
    Encoding [context ‖ target] and slicing the last N patches must cover the
    exact same timesteps as patching the target alone — otherwise the target
    representations are shifted relative to what the predictor is asked for.
    """
    from timejepa.models.components.patching import Patching
    p = Patching(patch_size=16, stride=8, d_model=8, num_features=1)
    n_full = p.get_num_patches(480)
    n_tgt = p.get_num_patches(96)
    starts_full = torch.arange(n_full) * 8
    starts_tgt = 384 + torch.arange(n_tgt) * 8
    assert torch.equal(starts_full[-n_tgt:], starts_tgt)


def test_contextualized_targets_change_the_representation(small_model):
    """The whole point: contextualized targets must differ from isolated ones."""
    torch.manual_seed(0)
    ctx, tgt = torch.randn(4, 384, 1), torch.randn(4, 96, 1)
    with torch.no_grad():
        a = small_model.forward_pretrain(ctx, tgt, contextualized_targets=True)["targets"]
        b = small_model.forward_pretrain(ctx, tgt, contextualized_targets=False)["targets"]
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-4)


# =============================================================================
# P1.1 / P1.2 — anti-collapse regularizers
# =============================================================================



















# ---------------------------------------------------------------------------
# G6 — ablation d'objectif : reconstruction contre extrapolation latente.
#
# L'arm reconstruction re-déroule les patchs futurs à la main dans le module de
# pretrain, parce que `Patching` projette vers d_model et ne rend jamais les
# valeurs brutes. Deux sources de vérité pour la géométrie des patchs, c'est
# exactement ce qui dérive en silence — d'où ces tests.
# ---------------------------------------------------------------------------

# Horizons ALIGNÉS sur le pas de patch uniquement. Sur un horizon non aligné
# (ex. 100 avec patch 16 / stride 8), `Patching` rembourre et rend 12 patchs
# alors que `model.num_target_patches`, calculé à la construction sans padding,
# en annonce 11 — et le prédicteur, dont la table de requêtes est dimensionnée
# sur ce compte, refuse la géométrie. Contrainte PRÉEXISTANTE du modèle, sans
# rapport avec cette ablation ; toutes les configs réelles (96, 128, 192, 256)
# sont alignées.
@pytest.mark.parametrize("pred_len,patch,stride", [
    (96, 16, 8), (256, 16, 8), (128, 32, 16), (192, 16, 8), (96, 16, 16),
])
def test_reconstruction_patches_match_patching_geometry(pred_len, patch, stride):
    """Le nombre de patchs bruts doit égaler celui que produit `Patching`."""
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule

    model = JEPATST(input_length=384, prediction_length=pred_len,
                    patch_size=patch, stride=stride, d_model=32,
                    num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp")
    module = JEPAPretrainModule(model=model, reconstruction_target=True,
                                loss_type='mse')

    target = torch.randn(4, pred_len, 1)
    # On passe par le chemin RÉEL plutôt que par une prédiction fabriquée :
    # c'est `outputs['predictions']` que `_scored_pair` reçoit en production, et
    # son nombre de patchs vient du patching à l'exécution.
    outputs = model.forward_pretrain(torch.randn(4, 384, 1), target)
    preds, patches = module._scored_pair(target, outputs)

    expected = model.patching.get_num_patches(pred_len)
    assert patches.shape[1] == expected, (
        f"patchs bruts {patches.shape[1]} != Patching {expected}"
    )
    assert patches.shape == preds.shape, "les deux côtés de la MSE doivent coïncider"
    assert patches.shape[-1] == patch, "un patch brut porte patch_size valeurs"


def test_reconstruction_targets_live_in_revin_space():
    """
    Les cibles brutes doivent être normalisées avec les stats du CONTEXTE, comme
    `forward_pretrain` le fait. Sinon la loss suit l'échelle de chaque série au
    lieu de sa forme.
    """
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule

    model = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp")
    module = JEPAPretrainModule(model=model, reconstruction_target=True,
                                loss_type='mse')

    context = torch.randn(4, 384, 1) * 50 + 1000     # échelle volontairement absurde
    target = torch.randn(4, 96, 1) * 50 + 1000
    model.forward_pretrain(context, target)

    predictions = torch.randn(4, model.num_target_patches, 32)
    _, patches = module._scored_pair(target, {'predictions': predictions})

    assert patches.abs().max() < 20, (
        f"cibles non normalisées (max {patches.abs().max():.1f}) — "
        "la MSE mesurerait l'échelle, pas la forme"
    )


def test_jepa_arm_is_untouched_by_the_ablation_flag():
    """Par défaut, la paire notée reste strictement celle de JEPA."""
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule

    model = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp")
    module = JEPAPretrainModule(model=model)

    assert module.reconstruction_target is False
    assert not hasattr(module, 'recon_head'), \
        "aucun paramètre supplémentaire ne doit exister hors ablation"

    outputs = {'predictions': torch.randn(2, 3, 32), 'targets': torch.randn(2, 3, 32)}
    preds, targets = module._scored_pair(torch.randn(2, 96, 1), outputs)
    assert preds is outputs['predictions'] and targets is outputs['targets']


def test_reconstruction_loss_bypasses_the_anti_collapse_terms():
    """En mode reconstruction, la loss est un Huber nu — pas de SIGReg."""
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule
    import torch.nn.functional as F

    model = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp")
    module = JEPAPretrainModule(model=model, reconstruction_target=True,
                                loss_type='mse',
                                sigreg_config={'weight': 25.0})

    preds, targets = torch.randn(4, 11, 16), torch.randn(4, 11, 16)
    loss, components = module._compute_loss(
        preds, targets, {'context_embeddings': torch.randn(4, 47, 32)}
    )

    assert torch.allclose(loss, F.smooth_l1_loss(preds, targets)), \
        "un terme de régularisation s'est glissé dans l'objectif de reconstruction"
    assert 'reconstruction_huber' in components


def test_reconstruction_loss_is_robust_to_the_revin_epsilon_floor():
    """
    RevIN normalise avec sqrt(var + 1e-5). Sur un contexte quasi constant le
    plancher vaut 0.00316, et la cible normalisée part à des milliers de sigma.
    Sous MSE, un tel batch écrase tous les autres dans le gradient et l'objectif
    devient celui des fenêtres dégénérées — ce qui confondrait G6.
    """
    from timejepa.training.jepa_pretrain_module import JEPAPretrainModule
    import torch.nn.functional as F

    model = JEPATST(input_length=384, prediction_length=96, patch_size=16,
                    stride=8, d_model=32, num_layers=1, num_heads=4, d_ff=64,
                    predictor_num_layers=1, predictor_num_heads=4,
                    predictor_d_ff=64, decoder_type="mlp")
    module = JEPAPretrainModule(model=model, reconstruction_target=True,
                                loss_type='smooth_l1')

    preds = torch.zeros(4, 11, 16)
    sane = torch.randn(4, 11, 16)
    outlier = sane.clone()
    outlier[0, 0, 0] = 6300.0                      # une fenêtre au plancher epsilon

    sane_loss, _ = module._compute_loss(preds, sane, {})
    outlier_loss, components = module._compute_loss(preds, outlier, {})
    huber_ratio = (outlier_loss / sane_loss).item()

    # La grandeur qui compte n'est pas un seuil absolu — elle dépend de la
    # taille du batch — mais le rapport à ce que MSE aurait fait : quadratique
    # contre linéaire. Sur ce mini-batch de 704 éléments, MSE amplifie ~56 000x,
    # Huber ~22x ; en batch réel (90 k éléments) la contribution de l'aberration
    # tombe à ~0.07, donc négligeable.
    mse_ratio = (F.mse_loss(preds, outlier) / F.mse_loss(preds, sane)).item()

    assert huber_ratio < mse_ratio / 100, (
        f"Huber amplifie x{huber_ratio:.0f} contre x{mse_ratio:.0f} pour MSE — "
        "la borne sur la contribution par élément ne joue pas"
    )
    assert components['target_absmax'] > 6000, \
        "l'amplitude des cibles doit rester observable malgré la loss robuste"



# ---------------------------------------------------------------------------
# G8.1 — réadmission ciblée de sous-ensembles LOTSA (E17).
#
# Les motifs d'exclusion sont des sous-chaînes grossières ; EVAL_SAFE_OVERRIDES
# réadmet des homonymes sans rapport avec l'éval. C'est la zone du projet où une
# erreur invalide TOUS les chiffres d'un coup — d'où un test qui épingle les
# deux sens : ce qui doit rester dehors, et ce qui doit être rentré.
# ---------------------------------------------------------------------------





