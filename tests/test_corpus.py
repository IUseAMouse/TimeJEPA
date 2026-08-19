"""
Tests du PIPELINE DE CORPUS (extraits de test_p0_regressions.py, audit du
2026-08-19 — déplacés à l'identique, aucun test réécrit).

Couvre : B17 (dataset trop court ne tue pas un run multi-datasets), G5
(intégration LOTSA purement additive : exclusions, mmap, _pack_series), B22
(object arrays des survivants uniformes), B18 (dérive de version torch dans le
sampler), G8.1 (réadmissions EVAL_SAFE_OVERRIDES — la zone où une erreur
invalide tous les chiffres du projet).
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

    # beijing_air_quality / china_air_quality — position INVERSÉE le 2026-08-19
    # (G8.1, décision utilisateur). L'ancien raisonnement (« = kdd_cup_2018 »)
    # était trop large : kdd_cup_2018 évalue Pékin 2017-2018 et un chevauchement
    # PARTIEL de fenêtres est concevable, mais `Salesforce/GiftEvalPretrain` —
    # le corpus de pré-entraînement PUBLIÉ PAR LES AUTEURS du benchmark —
    # contient les deux sous-ensembles : aucune entrée du leaderboard n'est
    # pénalisée pour les avoir vus. Les exclure nous handicapait unilatéralement
    # (E17 : l'écart au leaderboard suit la couverture du corpus).
    # Le risque résiduel est consigné au §5 du registre expérimental ; si un
    # relecteur le conteste, retirer les deux d'EVAL_SAFE_OVERRIDES ET remettre
    # l'assertion inverse ici — les deux doivent bouger ensemble.
    for name in ["beijing_air_quality", "china_air_quality"]:
        assert not is_eval_overlap(name), \
            f"{name} est réadmis par EVAL_SAFE_OVERRIDES (sanctionné par GiftEvalPretrain)"
    # ...mais le dataset d'ÉVAL correspondant reste rigoureusement exclu :
    assert is_eval_overlap("kdd_cup_2018_with_missing")

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


@pytest.mark.parametrize("name", [
    # datasets d'évaluation GIFT-Eval
    "m4_yearly", "m4_hourly", "m4_daily", "m4_monthly", "m4_quarterly", "m4_weekly",
    "SZ_TAXI", "LOOP_SEATTLE", "M_DENSE", "covid_deaths", "hospital", "restaurant",
    "saugeenday", "us_births", "car_parts_with_missing", "hierarchical_sales",
    "kdd_cup_2018_with_missing", "temperature_rain_with_missing",
    # datasets d'évaluation Nixtla
    "traffic_hourly", "traffic_weekly", "weather", "oikolab_weather",
    "cdc_fluview_ilinet",
    # datasets d'évaluation Monash locale
    "solar_power", "wiki-rolling_nips", "extended_web_traffic_with_missing",
    "kaggle_web_traffic_weekly",
])
def test_eval_datasets_stay_excluded_from_pretraining(name):
    """Un seul de ces noms au pretrain invaliderait tous les chiffres du projet."""
    from timejepa.data.lotsa import is_eval_overlap
    assert is_eval_overlap(name), f"{name} FUIT dans le corpus de pré-entraînement"


@pytest.mark.parametrize("name", [
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "monash_m3_monthly", "monash_m3_other", "monash_m3_quarterly", "monash_m3_yearly",
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",
    "nn5_daily_with_missing", "nn5_weekly",
    "taxi_30min", "kdd2022", "covid19_energy", "covid_mobility",
    "Q-TRAFFIC", "australian_electricity_demand",
    "beijing_air_quality", "china_air_quality",
])
def test_safe_overrides_are_readmitted(name):
    """
    Ces sous-ensembles sont dans GiftEvalPretrain (corpus sanctionné par le
    benchmark) et n'ont de contrepartie dans AUCUNE de nos trois suites d'éval.
    Les exclure coûtait de la couverture fréquentielle pour rien (E17).
    """
    from timejepa.data.lotsa import is_eval_overlap
    assert not is_eval_overlap(name), f"{name} devrait être réadmis"


def test_overrides_match_exactly_never_as_substring():
    """
    Un override doit être une égalité de nom, jamais une sous-chaîne : sinon
    'taxi_30min' réadmettrait 'sz_taxi_30min_variant' et rouvrirait par
    l'override le trou que le motif ferme.
    """
    from timejepa.data.lotsa import is_eval_overlap
    assert is_eval_overlap("taxi_30min_extended")
    assert is_eval_overlap("m1_monthly_v2")
    assert is_eval_overlap("pre_kdd2022")
