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
# G5 — LOTSA integration must be purely additive
# =============================================================================

def test_lotsa_segmentation_produces_dense_chunks():
    """
    The corpus MUST convert to dense arrays only: object arrays break fork's
    copy-on-write (B19) and would be fatal at LOTSA scale.
    """
    import numpy as np
    from timejepa.data.lotsa import segment_series, iter_dense_chunks

    # long series -> exact chunks, remainder dropped
    chunks = segment_series(np.arange(20_000.0), chunk_length=8192, min_length=1280)
    assert len(chunks) == 2
    assert all(c.shape == (8192,) for c in chunks)
    assert all(c.dtype == np.float32 for c in chunks)

    # too short -> nothing
    assert segment_series(np.arange(500.0), 8192, 1280) == []

    # a stream of mixed lengths yields only exact-length chunks
    stream = [np.arange(20_000.0), np.arange(900.0), np.arange(9000.0)]
    out = list(iter_dense_chunks(stream, chunk_length=8192, min_length=1280))
    assert all(c.shape == (8192,) for c in out)
    assert np.stack(out).dtype != object


@pytest.mark.parametrize("size,reference", [("mini", "mini"), ("base", "base")])
def test_lotsa_scale_configs_match_their_reference_dimensions(size, reference):
    """
    The dimensions are written out rather than inherited from mini.yaml/base.yaml,
    because those carry their own data block which would clobber the LOTSA
    corpus. Written-out values drift; this pins them.
    """
    ref = _compose(reference)
    pre = _compose(f"lotsa_{size}")
    for block in ("encoder", "predictor"):
        for key in ("d_model", "n_layers", "d_ff"):
            assert pre.model[block][key] == ref.model[block][key], f"{block}.{key}"
    # ...while the corpus and geometry stay those of the LOTSA round
    base = _compose("lotsa_tiny")
    assert pre.data.data_dir == base.data.data_dir
    assert pre.data.use_mmap is True
    assert pre.model.seq_length == base.model.seq_length
    assert pre.model.decoder.type == "quantile"
    assert pre.training.loss.type == "sigreg"


@pytest.mark.parametrize("size", ["tiny", "mini", "base"])
def test_lotsa_eval_config_matches_the_trained_model(size):
    """
    A shape mismatch at eval time only WARNS before producing silently wrong
    numbers — the trap already hit on the p32 arm. Capacity must match too, not
    just geometry.
    """
    zs = _compose(f"lotsa_{size}_zeroshot")
    ev = _compose(f"lotsa_{size}_eval")

    for key in ("seq_length", "prediction_length", "patch_length", "stride"):
        assert ev.model[key] == zs.model[key], f"eval/{key} drifted"
    for block in ("encoder", "predictor"):
        for key in ("d_model", "n_layers", "d_ff"):
            assert ev.model[block][key] == zs.model[block][key], f"eval/{block}.{key}"
    assert ev.model.decoder.type == zs.model.decoder.type

    # The eval config reads the HELD-OUT corpus, never LOTSA
    assert "lotsa" not in str(ev.data.data_dir).lower()
    assert ev.data.get("use_mmap", False) is False

    # The zero-shot arm trains its decoder on LOTSA only
    assert zs.data.datasets_finetune is None
    assert "lotsa" in str(zs.data.data_dir)


def test_lotsa_configs_share_one_effective_batch_regime():
    """
    Effective batch is batch x accumulation x GPUs. tiny_geo used accumulation 6
    on one or two cards; inheriting it here gave 6144 across four, a learning-rate
    regime unrelated to any previous run, and forced an override on every command.
    All three scales are now calibrated for four GPUs.
    """
    effective = {
        size: _compose(f"lotsa_{size}").data.batch_size
        * _compose(f"lotsa_{size}").trainer.accumulate_grad_batches
        * 4
        for size in ("tiny", "mini", "base")
    }
    assert all(1000 <= v <= 2048 for v in effective.values()), effective


def test_memmapped_windows_yield_writable_tensors(tmp_path):
    """
    B23. With use_mmap, np.load(mmap_mode="r") is read-only, ascontiguousarray on
    a contiguous slice does not copy, and .float() on float32 is a no-op — so the
    tensor aliased the mapping. Augmentations run right after, and an in-place
    write to a read-only mapping is a segfault at best and silent corruption of
    the .npy on disk at worst.
    """
    import numpy as np
    from timejepa.data.dataset import TimeSeriesDataset

    path = tmp_path / "dense.npy"
    np.save(path, np.sin(np.arange(4 * 4096.0).reshape(4, 4096)).astype(np.float32))
    original = np.load(path).copy()

    ds = TimeSeriesDataset(str(path), context_length=512, prediction_length=256,
                           use_mmap=True)
    item = ds[0]
    ctx = item["context"]

    # An in-place write must be safe and must NOT reach the file
    ctx.add_(1.0)
    assert np.array_equal(np.load(path), original), "in-place write reached the .npy"

    # The non-mmap path is unchanged
    plain = TimeSeriesDataset(str(path), context_length=512, prediction_length=256)
    plain[0]["context"].add_(1.0)
    assert np.array_equal(np.load(path), original)


def test_family_grouping_prevents_one_domain_dominating():
    """
    The per-subset cap protects nothing when one domain is split across many
    subsets. LOTSA ships cmip6 as 33 annual slices and era5 as 30, which is 63
    of the 123 kept subsets: at an equal cap each, half the pretraining corpus
    would be smooth seasonal climate reanalysis. E10 already measured that
    failure mode at smaller scale (two datasets holding 48.7% of the batch).
    """
    from timejepa.data.lotsa import family_of

    assert family_of("cmip6_1850") == family_of("cmip6_2010") == "cmip6"
    assert family_of("era5_1989") == family_of("era5_2018") == "era5"
    assert family_of("largest_2017") == family_of("largest_2021") == "largest"
    assert family_of("gfc12_load") == family_of("gfc17_load") == "gfc_load"

    # Subsets whose trailing number is not a year stay distinct
    assert family_of("PEMS03") != family_of("PEMS04")
    # ...and a lone year-suffixed subset is its own family, not merged away
    assert family_of("azure_vm_traces_2017") == "azure_vm_traces"
    assert family_of("borg_cluster_data_2011") == "borg_cluster_data"
    assert family_of("m5") == "m5"


def test_chunk_stats_explain_an_empty_subset():
    """
    A subset that yields nothing must say WHY. Series shorter than chunk_length
    are dropped to keep the output dense, so a series of 5000 steps — perfectly
    usable for a 1280-step window — is lost at chunk_length 8192. Without the
    breakdown, that is indistinguishable from a subset whose series are simply
    too short, and the two call for opposite decisions.
    """
    import numpy as np
    from timejepa.data.lotsa import iter_dense_chunks, ChunkStats

    series = [np.arange(20000.0), np.arange(5000.0), np.arange(900.0), np.arange(3000.0)]

    wide = ChunkStats()
    list(iter_dense_chunks(series, chunk_length=8192, min_length=1280, stats=wide))
    assert wide.series == 4
    assert wide.too_short == 1              # the 900-step one, genuinely unusable
    assert wide.lost_to_chunking == 2       # 5000 and 3000: recoverable
    assert "PERDUES" in wide.summary(8192, 1280)

    narrow = ChunkStats()
    list(iter_dense_chunks(series, chunk_length=2048, min_length=1280, stats=narrow))
    assert narrow.lost_to_chunking == 0
    assert narrow.emitted > wide.emitted    # 12 against 2 on identical input
    assert "PERDUES" not in narrow.summary(2048, 1280)


def test_short_gaps_are_imputed_and_structural_ones_refused():
    """
    Rejecting a whole chunk over one NaN is untenable on a real corpus: measured
    on LOTSA, HZMETRO lost 160/160 chunks and SHMETRO 2304/2304. Short gaps get
    interpolated; large ones are refused rather than invented.

    The refusal matters as much as the imputation. Those metro subsets carry
    ~23% NaN in REGULAR 23-step blocks — the nightly service closure. Filling
    that would fabricate 3am ridership, so raising the threshold would be a
    mistake rather than a fix, and the summary says so.
    """
    import numpy as np
    from timejepa.data.lotsa import iter_dense_chunks, ChunkStats

    short_gap = np.arange(4096.0, dtype=np.float32)
    short_gap[10:13] = np.nan          # 0.07%, well under the threshold
    clean = np.arange(4096.0, dtype=np.float32)

    st = ChunkStats()
    out = list(iter_dense_chunks([short_gap, clean], chunk_length=4096,
                                 min_length=1280, stats=st))
    assert len(out) == 2               # both kept, one imputed
    assert st.imputed == 1
    assert all(np.isfinite(c).all() for c in out)
    assert float(out[0][11]) == 11.0   # linear interpolation, not a constant

    # A metro-shaped series: regular 23-step blocks, ~23% missing
    structural = np.arange(4096.0, dtype=np.float32)
    for start in range(0, 4096, 100):
        structural[start:start + 23] = np.nan

    st2 = ChunkStats()
    out2 = list(iter_dense_chunks([structural], chunk_length=4096,
                                  min_length=1280, stats=st2))
    assert out2 == []
    assert st2.non_finite == 1
    summary = st2.summary(4096, 1280)
    assert "STRUCTURELS" in summary
    assert "ne pas monter --max-nan-fraction" in summary


def test_lotsa_excludes_every_nixtla_and_gift_eval_source():
    """
    A subset missed here is a benchmark seen during pretraining, and a zero-shot
    claim that is not one. The Monash corpus has exactly that defect
    (electricity-hourly IS the Nixtla electricity benchmark), which the LOTSA
    protocol avoids by construction.
    """
    from timejepa.data.lotsa import (
        is_eval_overlap, NIXTLA_OVERLAP_PATTERNS, GIFT_EVAL_OVERLAP_PATTERNS,
        EVAL_OVERLAP_PATTERNS,
    )

    nixtla = ["ett_h1", "ETTm2", "electricity_15min", "traffic_hourly",
              "weather", "exchange_rate", "illness"]
    # The 28 directories of the official GIFT-Eval repository, verbatim, read
    # from https://huggingface.co/api/datasets/Salesforce/GiftEval/tree/main
    # on 2026-08-13. Missing one means a benchmark seen during pretraining and
    # a zero-shot claim that is not one.
    gift = ["LOOP_SEATTLE", "M_DENSE", "SZ_TAXI", "bitbrains_fast_storage",
            "bitbrains_rnd", "bizitobs_application", "bizitobs_l2c",
            "bizitobs_service", "car_parts_with_missing", "covid_deaths",
            "electricity", "ett1", "ett2", "hierarchical_sales", "hospital",
            "jena_weather", "kdd_cup_2018_with_missing", "m4_daily", "m4_hourly",
            "m4_monthly", "m4_quarterly", "m4_weekly", "m4_yearly", "restaurant",
            "saugeenday", "solar", "temperature_rain_with_missing", "us_births"]
    assert len(gift) == 28
    for name in nixtla + gift:
        assert is_eval_overlap(name), f"{name} must be excluded"

    # ...without excluding the corpus that makes LOTSA worth having
    for name in ["azure_vm_traces", "borg_cluster_data", "alibaba_cluster_trace",
                 "cmip6_1850", "era5_1989", "godaddy", "favorita_sales",
                 "residential_load_power", "buildings_900k", "PEMS_BAY",
                 "largest_2017", "uber_tlc_hourly", "m5"]:
        assert not is_eval_overlap(name), f"{name} must be kept"

    # beijing_air_quality / china_air_quality ARE kdd_cup_2018 (Beijing air
    # quality 2017-2018), which GIFT-Eval evaluates on — near-duplicates that
    # must not sit in the pretraining corpus.
    for name in ["beijing_air_quality", "china_air_quality"]:
        assert is_eval_overlap(name), f"{name} duplicates kdd_cup_2018"

    # The union is what the converter uses; the two sources stay separable
    # because they are verified differently (Nixtla is what we measure today,
    # GIFT-Eval is from the benchmark's published composition).
    assert set(EVAL_OVERLAP_PATTERNS) == set(NIXTLA_OVERLAP_PATTERNS) | set(
        GIFT_EVAL_OVERLAP_PATTERNS)


def test_write_dense_npy_is_memmappable_and_truncated(tmp_path):
    import numpy as np
    from timejepa.data.lotsa import write_dense_npy

    chunks = [np.full(512, float(i), dtype=np.float32) for i in range(5)]
    out = tmp_path / "subset.npy"
    n = write_dense_npy(iter(chunks), out, chunk_length=512, max_chunks=100)

    assert n == 5
    arr = np.load(out, mmap_mode="r")          # the mode training will use
    assert arr.shape == (5, 512)               # truncated, not 100
    assert arr.dtype == np.float32
    assert float(arr[3][0]) == 3.0
    assert not (tmp_path / "subset.tmp.npy").exists()


def test_use_mmap_defaults_off_and_rejects_object_arrays(tmp_path):
    """
    Additive by construction: every existing config must load exactly as before,
    and the mmap path must refuse the object arrays it cannot handle.
    """
    import numpy as np
    from timejepa.data.dataset import TimeSeriesDataset

    dense = tmp_path / "dense.npy"
    np.save(dense, np.sin(np.arange(4 * 4096.0).reshape(4, 4096)).astype(np.float32))

    plain = TimeSeriesDataset(str(dense), context_length=512, prediction_length=256)
    mapped = TimeSeriesDataset(str(dense), context_length=512, prediction_length=256,
                               use_mmap=True)
    assert len(plain) == len(mapped)
    assert torch.allclose(plain[0]["context"], mapped[0]["context"])

    ragged = tmp_path / "ragged.npy"
    arr = np.empty(2, dtype=object)
    arr[0] = np.arange(5000.0); arr[1] = np.arange(4000.0)
    np.save(ragged, arr, allow_pickle=True)
    with pytest.raises(ValueError, match="dense array"):
        TimeSeriesDataset(str(ragged), context_length=512, prediction_length=256,
                          use_mmap=True)


def test_lotsa_configs_do_not_disturb_existing_ones():
    """LOTSA configs are additions; tiny_geo must be untouched by their presence."""
    base = _compose("tiny_geo")
    assert base.data.get("use_mmap", False) is False
    assert "lotsa" not in str(base.data.data_dir).lower()

    pre = _compose("lotsa_tiny")
    assert pre.data.use_mmap is True
    assert "lotsa" in str(pre.data.data_dir)
    assert pre.data.datasets is None          # glob the directory
    assert pre.model.seq_length == base.model.seq_length      # same geometry
    assert pre.model.patch_length == base.model.patch_length

    ft = _compose("lotsa_tiny_finetune")
    assert ft.training.mode == "finetune"
    # The domain-adapted arm stays on the Monash corpus (contaminated, documented)
    assert ft.data.data_dir == base.data.data_dir
    assert ft.data.get("use_mmap", False) is False
    assert len(ft.data.datasets_finetune) == len(base.data.datasets_finetune)

    # The zero-shot arm — the primary protocol — must train its decoder on LOTSA
    # only, so that Monash and Nixtla stay unseen at every stage.
    zs = _compose("lotsa_tiny_zeroshot")
    assert zs.training.mode == "finetune"
    assert zs.training.finetune_mode == "full_finetune"
    assert zs.data.datasets_finetune is None      # glob the LOTSA directory
    assert "lotsa" in str(zs.data.data_dir)
    assert zs.data.use_mmap is True
    assert zs.model.name != ft.model.name         # separate checkpoint trees

    # The eval config must carry the TRAINING geometry but the EVALUATION data.
    # If the two ever diverge, this breaks here rather than silently producing
    # wrong numbers: at eval time a geometry mismatch only WARNS.
    ev = _compose("lotsa_tiny_eval")
    for key in ("seq_length", "prediction_length", "patch_length", "stride"):
        assert ev.model[key] == zs.model[key], f"eval/{key} drifted from the trained model"
    assert ev.model.decoder.type == zs.model.decoder.type
    # ...and it must evaluate on the held-out corpus, never on LOTSA
    assert ev.data.data_dir == base.data.data_dir
    assert "lotsa" not in str(ev.data.data_dir).lower()
    assert ev.data.get("use_mmap", False) is False


# =============================================================================
# B22 — uniform-length survivors of the length filter became object arrays
# =============================================================================

def test_pack_series_returns_numeric_when_lengths_are_uniform():
    """
    B22. np.array(list_of_arrays, dtype=object) does NOT give a ragged 1-D array
    when every series has the same length — it silently gives a 2-D OBJECT array,
    and np.stack on that preserves dtype=object. The value then reaches
    torch.from_numpy, which rejects object arrays outright.

    Never triggered before because the length filter always left mixed lengths;
    it fires as soon as a dataset whose survivors are uniform is used
    (m4-hourly: every series is 1008 steps once filtered).
    """
    import numpy as np
    from timejepa.data.dataset import _pack_series

    uniform = _pack_series([np.arange(64.0), np.arange(64.0), np.arange(64.0)])
    assert uniform.dtype != object, "uniform survivors must become a numeric array"
    assert uniform.shape == (3, 64)
    # The operation that used to raise
    torch.from_numpy(np.ascontiguousarray(uniform[0])).float()


def test_pack_series_keeps_ragged_input_ragged():
    import numpy as np
    from timejepa.data.dataset import _pack_series

    ragged = _pack_series([np.arange(64.0), np.arange(48.0)])
    assert ragged.dtype == object
    assert ragged.ndim == 1 and len(ragged) == 2
    assert ragged[0].dtype != object
    torch.from_numpy(np.ascontiguousarray(ragged[1])).float()


def test_pack_series_empty_stays_detectable():
    """The empty case must keep working: callers raise SeriesTooShortError."""
    import numpy as np
    from timejepa.data.dataset import _pack_series
    assert len(_pack_series([])) == 0


def test_dataset_yields_float_tensors_for_uniform_variable_length_input(tmp_path):
    """End-to-end: the exact m4-hourly shape must produce usable tensors."""
    import numpy as np
    from timejepa.data.dataset import TimeSeriesDataset

    raw = np.empty(3, dtype=object)
    raw[0] = np.sin(np.arange(1008.0) / 10)
    raw[1] = np.sin(np.arange(1008.0) / 12)
    raw[2] = np.sin(np.arange(700.0) / 10)      # dropped by the length filter
    path = tmp_path / "uniform_after_filter.npy"
    np.save(path, raw, allow_pickle=True)

    ds = TimeSeriesDataset(str(path), context_length=512, prediction_length=256)
    item = ds[0]
    assert item["context"].dtype == torch.float32
    assert item["context"].shape[-1] == 512
    assert item["target"].shape[-1] == 256
    assert torch.isfinite(item["context"]).all()


# =============================================================================
# B21 / config hygiene — the experiment grid must be declarative
# =============================================================================

def _compose(name):
    from hydra import initialize, compose
    root = Path(__file__).resolve().parents[1]
    with initialize(version_base=None, config_path="../configs/model"):
        return compose(config_name=name)


@pytest.mark.parametrize("config_name", ["tiny", "tiny_geo", "tiny_geo_p32",
                                         "tiny_geo_vicreg", "tiny_geo_scratch"])
def test_checkpoint_filename_has_no_equals_sign(config_name):
    """
    B21. auto_insert_metric_name lived in the config but was never forwarded to
    ModelCheckpoint, so Lightning kept its default (True) and prefixed each
    metric name on top of the template's own text:
    'epochepoch=00_val_lossval_loss=0.3445.ckpt'. Hydra's override grammar
    treats '=' as a separator, so every downstream finetune and eval command
    needed quoting gymnastics — and one of them failed outright with backslashes
    surviving literally into the path.
    """
    cfg = _compose(config_name)
    assert "=" not in cfg.checkpoint.filename
    assert cfg.checkpoint.auto_insert_metric_name is False


def test_geo_arms_differ_only_in_their_declared_variable():
    """
    Each arm of the geometry grid is a named config, not a pile of overrides:
    a forgotten `model.patch_length=32` at EVAL time only warns
    ('re-initialising patching.*') and yields silently wrong numbers.
    Everything except the arm's own variable must match the base.
    """
    base = _compose("tiny_geo")
    # Round-wide defaults: no override should ever be needed for these
    assert base.training.loss.type == "sigreg"
    assert base.model.decoder.type == "quantile"
    assert len(base.data.datasets_finetune) == len(base.data.datasets)

    p32 = _compose("tiny_geo_p32")
    assert (p32.model.patch_length, p32.model.stride) == (32, 16)
    assert p32.model.name != base.model.name
    for key in ("seq_length", "prediction_length"):
        assert p32.model[key] == base.model[key]
    assert p32.training.loss.type == base.training.loss.type
    assert p32.model.decoder.type == base.model.decoder.type

    vic = _compose("tiny_geo_vicreg")
    assert vic.training.loss.type == "vicreg"
    assert vic.model.name != base.model.name
    assert (vic.model.patch_length, vic.model.stride) == (base.model.patch_length,
                                                          base.model.stride)

    scratch = _compose("tiny_geo_scratch")
    assert scratch.training.mode == "finetune"
    # Mandatory: gradual_unfreeze would freeze randomly initialised weights
    assert scratch.training.finetune_mode == "full_finetune"
    assert "pretrained_encoder_path" not in scratch.training


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
