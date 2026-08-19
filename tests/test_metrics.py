"""
Tests des MÉTRIQUES ET RÉGULARISEURS (extraits de test_p0_regressions.py,
audit du 2026-08-19 — déplacés à l'identique, aucun test réécrit).

Couvre : P0.4 (baselines), P0.5 (métriques sans échelle : MASE poolé, WQL),
P1.1/P1.2 (anti-effondrement : VICReg par position, SIGReg).
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
