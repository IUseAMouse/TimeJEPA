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

@pytest.fixture
def seasonal_batch():
    torch.manual_seed(0)
    t = torch.arange(700.0).view(1, -1)
    y = torch.sin(2 * torch.pi * t / 24).repeat(32, 1) * 3 + torch.randn(32, 700) * 0.5
    return y[:, :500], y[:, 500:596]


def test_seasonal_naive_repeats_last_cycle():
    ctx = torch.arange(48.0).view(1, -1).repeat(2, 1)
    out = seasonal_naive_forecast(ctx, horizon=24, season_length=24)
    assert torch.equal(out[0], torch.arange(24.0, 48.0))


def test_seasonal_naive_degenerates_to_last_value_when_m_is_1():
    ctx = torch.randn(4, 100)
    assert torch.equal(
        seasonal_naive_forecast(ctx, 20, 1), last_value_forecast(ctx, 20)
    )


def test_baselines_preserve_shape():
    for ctx, expected in [
        (torch.randn(4, 500), (4, 96)),
        (torch.randn(4, 500, 2), (4, 96, 2)),
    ]:
        for name, pred in compute_all_baselines(ctx, 96, 24).items():
            assert pred.shape == expected, (name, pred.shape)


def test_linear_trend_extrapolates_a_line():
    t = torch.arange(100.0).view(1, -1)
    ctx = (2.0 * t + 5.0).repeat(3, 1)
    out = linear_trend_forecast(ctx, 10)
    expected = 2.0 * torch.arange(100.0, 110.0) + 5.0
    assert torch.allclose(out[0], expected, atol=1e-2)


def test_seasonal_naive_has_mase_one(seasonal_batch):
    """By construction MASE(seasonal naive) ~= 1 on a seasonal series."""
    ctx, tgt = seasonal_batch
    pred = seasonal_naive_forecast(ctx, 96, 24)
    assert 0.85 < mase(pred, tgt, ctx, 24).item() < 1.15


def test_context_mean_is_worse_than_seasonal_naive(seasonal_batch):
    ctx, tgt = seasonal_batch
    sn = mase(seasonal_naive_forecast(ctx, 96, 24), tgt, ctx, 24)
    cm = mase(mean_forecast(ctx, 96), tgt, ctx, 24)
    assert cm > sn


def test_get_seasonality():
    assert get_seasonality("ettm1") == 96
    assert get_seasonality("etth1") == 24
    assert get_seasonality("weather") == 144
    assert get_seasonality("exchange") == 1      # random walk, not daily
    assert get_seasonality("unknown", freq="H") == 24
    assert get_seasonality("unknown") == 1


# =============================================================================
# P0.5 — scale-free metrics
# =============================================================================

def test_wql_equals_nd_for_point_forecast():
    """
    With a point forecast the weighted quantile loss collapses exactly to ND.
    This is the honest CRPS a deterministic model earns on GIFT-Eval, and the
    reason a quantile head is a prerequisite there rather than a nice-to-have.
    """
    torch.manual_seed(0)
    p = torch.randn(8, 96)
    g = torch.randn(8, 96).abs() + 1
    assert abs(nd(p, g).item() - weighted_quantile_loss(p, g).item()) < 1e-5


def test_wql_improves_with_calibrated_quantiles():
    """A spread of quantiles around the truth must beat the degenerate point."""
    torch.manual_seed(0)
    g = torch.randn(16, 96).abs() + 5
    p = g + torch.randn_like(g) * 0.5
    point = weighted_quantile_loss(p, g).item()
    spread = torch.stack([p + s for s in torch.linspace(-1.2, 1.2, 9)])
    assert weighted_quantile_loss(spread, g).item() < point


def test_mase_survives_flat_windows():
    """
    Real data contains near-constant windows (ETTm2, electricity). With the
    classic mean-of-per-window-ratios, their seasonal difference is ~0 and the
    ratio blows up to ~1/eps, dominating the average — observed in practice as
    MASE ~1e4 for every model AND for seasonal naive itself.
    """
    torch.manual_seed(0)
    ctx = torch.randn(16, 200) * 2.0
    ctx[:4] = 5.0                       # 4 perfectly flat windows
    tgt = torch.randn(16, 48)
    pred = tgt + torch.randn(16, 48) * 0.3

    pooled = mase(pred, tgt, ctx, 24, aggregate="pooled").item()
    per_series = mase(pred, tgt, ctx, 24, aggregate="per_series").item()

    assert pooled < 100, f"pooled MASE exploded: {pooled}"
    assert per_series < 100, f"per-series MASE exploded: {per_series}"


def test_mase_all_flat_falls_back_to_mae():
    """If every window is constant, a scaled error is undefined — report MAE."""
    ctx = torch.full((8, 100), 3.0)
    tgt = torch.zeros(8, 24)
    pred = torch.ones(8, 24)
    assert abs(mase(pred, tgt, ctx, 24, aggregate="per_series").item() - 1.0) < 1e-5


def test_mase_is_scale_invariant():
    """MASE must not change when the whole series is rescaled."""
    torch.manual_seed(0)
    ctx = torch.randn(8, 200).cumsum(1)
    tgt = torch.randn(8, 48)
    pred = torch.randn(8, 48)
    a = mase(pred, tgt, ctx, 1).item()
    b = mase(pred * 1000, tgt * 1000, ctx * 1000, 1).item()
    assert abs(a - b) / a < 1e-4


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

def _emb(b=128, n=12, d=64, scale=1.0):
    return torch.randn(b, n, d) * scale


def test_sigreg_is_near_zero_on_isotropic_gaussian():
    """SIGReg's optimum is exactly the standard isotropic Gaussian."""
    from timejepa.training.utils.metrics import sigreg_loss
    torch.manual_seed(0)
    assert sigreg_loss(_emb(), max_tokens=4096)["loss"].item() < 0.01


def test_sigreg_penalises_collapse_hardest():
    from timejepa.training.utils.metrics import sigreg_loss
    torch.manual_seed(0)
    collapsed = torch.randn(1, 1, 64).repeat(128, 12, 1)
    healthy = _emb()
    assert (sigreg_loss(collapsed, max_tokens=4096)["loss"].item()
            > 10 * sigreg_loss(healthy, max_tokens=4096)["loss"].item())


def test_sigreg_catches_what_vicreg_cannot():
    """
    A bimodal embedding has healthy per-coordinate variance and a near-diagonal
    covariance, so VICReg is satisfied. SIGReg is not — that is the reason to
    have it as an alternative.
    """
    from timejepa.training.utils.metrics import sigreg_loss, vicreg_loss
    torch.manual_seed(0)
    bimodal = (torch.randint(0, 2, (128, 12, 64)).float() * 2 - 1) * 3
    bimodal = bimodal + torch.randn(128, 12, 64) * 0.2
    healthy = _emb()

    v_bi = vicreg_loss(bimodal, bimodal.detach())["variance"].item()
    v_ok = vicreg_loss(healthy, healthy.detach())["variance"].item()
    s_bi = sigreg_loss(bimodal, max_tokens=4096)["loss"].item()
    s_ok = sigreg_loss(healthy, max_tokens=4096)["loss"].item()

    assert v_bi <= v_ok + 1e-6, "VICReg was expected to be blind to bimodality"
    assert s_bi > 10 * s_ok, "SIGReg should flag bimodality"


def test_vicreg_per_position_catches_positional_collapse():
    """
    THE B6 bug. If every batch element is identical at a given patch position
    but positions differ from each other, pooling batch and position makes the
    variance hinge see healthy spread and report no penalty at all.
    """
    from timejepa.training.utils.metrics import vicreg_loss
    torch.manual_seed(0)
    per_pos = torch.randn(1, 12, 64) * 3
    collapsed = per_pos.repeat(128, 1, 1) + torch.randn(128, 12, 64) * 0.001

    pooled = vicreg_loss(collapsed, collapsed.detach(), per_position=False)["variance"].item()
    fixed = vicreg_loss(collapsed, collapsed.detach(), per_position=True)["variance"].item()

    assert pooled < 0.01, "pooled variance was expected to be blind here"
    assert fixed > 0.9, "per-position variance must flag the collapse"


def test_jepa_loss_regularizes_the_encoder_output():
    """
    Anti-collapse used to touch only the predictor output. Passing
    context_embeddings must change the loss, otherwise the encoder is
    unconstrained.
    """
    from timejepa.training.utils.metrics import jepa_loss
    torch.manual_seed(0)
    pred, tgt = _emb(), _emb().detach()
    collapsed_ctx = torch.randn(1, 1, 64).repeat(128, 47, 1)

    without = jepa_loss(pred, tgt, loss_type="vicreg",
                        vicreg_weights={"invariance": 25.0, "variance": 15.0, "covariance": 1.0})
    with_ctx = jepa_loss(pred, tgt, loss_type="vicreg",
                         vicreg_weights={"invariance": 25.0, "variance": 15.0, "covariance": 1.0},
                         context_embeddings=collapsed_ctx)
    assert with_ctx.item() > without.item() + 1.0


def test_sigreg_path_through_jepa_loss():
    from timejepa.training.utils.metrics import jepa_loss
    torch.manual_seed(0)
    pred, tgt = _emb(), _emb().detach()
    loss, comp = jepa_loss(
        pred, tgt, loss_type="sigreg",
        sigreg_config={"lambda": 1.0, "apply_to": "both", "max_tokens": 2048},
        context_embeddings=_emb(b=128, n=47),
        return_components=True,
    )
    assert torch.isfinite(loss)
    assert {"invariance", "sigreg", "sigreg_context", "sigreg_predictions"} <= set(comp)


def test_regularizers_are_differentiable():
    from timejepa.training.utils.metrics import sigreg_loss, vicreg_loss
    for fn in (
        lambda z: sigreg_loss(z, max_tokens=1024)["loss"],
        lambda z: vicreg_loss(z, z.detach())["loss"],
    ):
        z = _emb(b=32, n=4, d=16).requires_grad_(True)
        fn(z).backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()


def test_extended_metrics_include_benchmark_keys():
    ctx = torch.randn(4, 200)
    tgt = torch.randn(4, 48)
    pred = torch.randn(4, 48)

    without = compute_forecasting_metrics_extended(pred, tgt)
    assert "mase" not in without          # no context -> no MASE, not a fake one
    assert {"nd", "wql", "mse", "mae", "r2"} <= set(without)

    with_ctx = compute_forecasting_metrics_extended(pred, tgt, context=ctx, season_length=24)
    assert "mase" in with_ctx
