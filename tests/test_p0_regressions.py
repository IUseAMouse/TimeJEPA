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
# B17 — a too-short dataset must not kill a multi-dataset run
# =============================================================================

def _write(dirpath, name, arr, allow_pickle=False):
    import numpy as np
    path = Path(dirpath) / f"{name}.npy"
    np.save(path, arr, allow_pickle=allow_pickle)
    return path


@pytest.fixture
def short_and_long_datasets(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    t = np.arange(20000.0)
    _write(tmp_path, "electricity-hourly",
           np.stack([np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(20000)
                     for _ in range(4)]).astype("float32"))
    # Exactly the shape that crashed the first real run: weekly data, 114 points
    _write(tmp_path, "wikipedia-web-traffic-weekly",
           rng.standard_normal((500, 114)).astype("float32"))
    # Same problem stored as a variable-length object array
    _write(tmp_path, "fred-md",
           np.array([rng.standard_normal(rng.integers(300, 500)).astype("float32")
                     for _ in range(50)], dtype=object), allow_pickle=True)
    return tmp_path


def test_too_short_fixed_length_dataset_raises_typed_error(short_and_long_datasets):
    from timejepa.data.dataset import TimeSeriesDataset, SeriesTooShortError
    with pytest.raises(SeriesTooShortError):
        TimeSeriesDataset(
            short_and_long_datasets / "wikipedia-web-traffic-weekly.npy",
            context_length=512, prediction_length=128,
        )


def test_too_short_variable_length_dataset_raises_too(short_and_long_datasets):
    """
    The variable-length path used to silently drop every series and then build a
    zero-window dataset, while the fixed-length path raised. Both now raise.
    """
    from timejepa.data.dataset import TimeSeriesDataset, SeriesTooShortError
    with pytest.raises(SeriesTooShortError):
        TimeSeriesDataset(
            short_and_long_datasets / "fred-md.npy",
            context_length=512, prediction_length=128,
        )


def test_error_reports_the_real_longest_series(short_and_long_datasets):
    """
    Variable-length series are filtered before window generation, so measuring
    what remains reports 0. The message must quote the pre-filter maximum, or it
    actively misleads whoever is debugging a config.
    """
    from timejepa.data.dataset import TimeSeriesDataset, SeriesTooShortError
    with pytest.raises(SeriesTooShortError) as exc:
        TimeSeriesDataset(
            short_and_long_datasets / "fred-md.npy",
            context_length=512, prediction_length=128,
        )
    assert exc.value.series_length >= 300, "reported 0 instead of the real length"
    assert "Longest usable context" in str(exc.value)


def test_window_indices_are_a_compact_array(short_and_long_datasets):
    """
    B19. window_indices was a Python list of 2-tuples: ~120 bytes per window
    (8 list pointer + 56 tuple + 56 for two int objects, start_idx being well
    past the small-int cache). At the corpus's ~54M windows that is ~6.5 GB —
    and paid PER PROCESS, because a dataloader worker walking the list bumps
    each tuple's refcount, writing to its page and defeating fork's
    copy-on-write. Observed as a steady climb to ~50 GB on a 57 GB host.

    An int32 [N, 2] array is 8 bytes per window and has no per-element Python
    objects, so the pages are genuinely shared.
    """
    import numpy as np
    from timejepa.data.dataset import TimeSeriesDataset

    ds = TimeSeriesDataset(
        short_and_long_datasets / "electricity-hourly.npy",
        context_length=512, prediction_length=128, stride=8,
    )
    wi = ds.window_indices
    assert isinstance(wi, np.ndarray)
    assert wi.dtype == np.int32
    assert wi.ndim == 2 and wi.shape[1] == 2
    assert wi.nbytes == len(ds) * 8


@pytest.mark.parametrize("layout", ["dense", "object"])
def test_window_indices_still_address_the_right_data(tmp_path, layout):
    """The compact layout must select byte-identical windows."""
    import numpy as np
    from timejepa.data.dataset import TimeSeriesDataset

    rng = np.random.default_rng(0)
    if layout == "dense":
        arr = rng.standard_normal((4, 8000)).astype("float32")
        _write(tmp_path, "d", arr)
        path = tmp_path / "d.npy"
    else:
        arr = np.array([rng.standard_normal(rng.integers(2000, 8000)).astype("float32")
                        for _ in range(6)], dtype=object)
        _write(tmp_path, "d", arr, allow_pickle=True)
        path = tmp_path / "d.npy"

    ds = TimeSeriesDataset(path, context_length=512, prediction_length=128, stride=64)
    for i in (0, len(ds) // 3, len(ds) - 1):
        series_idx, start = int(ds.window_indices[i][0]), int(ds.window_indices[i][1])
        expected = ds.normalized_data[series_idx][start:start + 512]
        assert np.allclose(ds[i]["context"].numpy(), expected)


def test_multidataset_skips_short_datasets_and_keeps_training(short_and_long_datasets):
    """
    THE B17 bug: wikipedia-web-traffic-weekly (114 weekly points vs 640 needed)
    aborted an entire 23-dataset pretraining run.
    """
    from timejepa.data.datamodule import MultiDatasetMonashDataModule
    dm = MultiDatasetMonashDataModule(
        data_dir=short_and_long_datasets,
        context_length=512, prediction_length=128,
        datasets=["electricity-hourly", "wikipedia-web-traffic-weekly", "fred-md"],
        batch_size=16, stride=64,
        normalize_mode="global", normalizer_type="identity", clip_outliers=False,
        train_val_test_split=(0.96, 0.02, 0.02), num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")

    assert dm.dataset_names_order == ["electricity-hourly"]
    assert len(dm.train_dataset) > 0


def test_multidataset_raises_when_everything_is_skipped(tmp_path):
    """Skipping is graceful; skipping *everything* must still be a hard error."""
    import numpy as np
    from timejepa.data.datamodule import MultiDatasetMonashDataModule
    _write(tmp_path, "a", np.random.randn(100, 114).astype("float32"))
    _write(tmp_path, "b", np.random.randn(100, 200).astype("float32"))
    dm = MultiDatasetMonashDataModule(
        data_dir=tmp_path, context_length=512, prediction_length=128,
        datasets=["a", "b"], batch_size=16, stride=64,
        normalize_mode="global", normalizer_type="identity", clip_outliers=False,
        train_val_test_split=(0.96, 0.02, 0.02), num_workers=0,
    )
    dm.prepare_data()
    with pytest.raises(RuntimeError, match="Every dataset was skipped"):
        dm.setup("fit")


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
                                         "tiny_deep_predictor"])
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

def test_temperature_sampler_constructs():
    """
    `Sampler.__init__` took a deprecated `data_source` argument; newer torch
    removed it, at which point `super().__init__(None)` reaches
    `object.__init__` and raises "takes exactly one argument". Killed a real
    run, and could not be reproduced locally because the local torch still had
    the old signature — hence this direct construction test.
    """
    from timejepa.data.datamodule import TemperatureSampler
    s = TemperatureSampler(dataset_sizes=[1000, 50000, 300], batch_size=64,
                           temperature=0.5)
    batch = next(iter(s))
    assert len(batch) == 64


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
