"""
Delta tied to the declared frequency (2026-10-05): the convention, the corpus
table, the item key, the harness flag - and everything unchanged when off.

1. Convention: season lengths give FlowState's reference factors
   (get_fixed_factor, granite-tsfm); spellings; GIFT rules.
2. Table: forms accepted, `null` = no frequency, bad entries refused.
3. Data: no table = item dict and batches as before; with a table every item
   carries 'season', the windows are the same, a missing file is an error.
4. Harness: without +freq_delta the calls and the results are unchanged; with
   it every forecast receives w = 24 / (season / k); guards; flag known.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from timejepa.data import frequency as F  # noqa: E402
from timejepa.data.dataset import TimeSeriesDataset  # noqa: E402

LEVELS = [0.1 * j for j in range(1, 10)]
PERIOD = 24


# ------------------------------------------------------------ 1. convention
@pytest.mark.parametrize("freq, weekly, factor", [
    ("4S", False, 24 / 900), ("10S", False, 24 / 360), ("T", False, 24 / 1440),
    ("5T", False, 24 / 288), ("15T", False, 0.25), ("30T", False, 0.5), ("H", False, 1.0),
    ("6H", False, 6.0), ("D", True, 24 / 7), ("D", False, 24 / 365), ("2D", True, 2 * 24 / 7),
    ("W", False, 24 / (365 / 7)), ("W-SUN", False, 24 / (365 / 7)), ("M", False, 2.0),
    ("Q", False, 6.0), ("Q-DEC", False, 6.0), ("A", False, 6.0), ("A-DEC", False, 6.0),
])
def test_factors_match_flowstate_reference(freq, weekly, factor):
    assert 24.0 / F.season_length(freq, weekly) == pytest.approx(factor, rel=1e-9)


def test_spellings_and_unknowns():
    assert F.season_length("15min") == F.season_length("15T") == 96.0
    assert F.season_length("h") == F.season_length("1H") == 24.0
    assert F.season_length("ME") == F.season_length("MS") == 12.0
    assert F.season_length("QE-DEC") == 4.0 and F.season_length("YE") == 4.0
    assert F.season_length("B", weekly=True) == 7.0
    for bad in (None, "", "fortnight", "12"):
        assert F.season_length(bad) is None


def test_delta_scale_is_clamped():
    assert F.delta_scale(96.0, 1 / 48, 4.0) == 0.25
    assert F.delta_scale(4.0, 1 / 48, 4.0) == 4.0            # quarterly asks for 6
    assert F.delta_scale(8640.0, 1 / 48, 4.0) == 1 / 48


def test_gift_rules():
    assert F.gift_season("solar/10T/long") == 144.0
    assert F.gift_season("loop_seattle/D/short") == 7.0       # Transport: weekly
    assert F.gift_season("m4_daily/D/short") == 365.0         # Econ/Fin: no weekly cycle
    assert F.gift_season("bizitobs_l2c/H/long") == 24.0 * 7   # no daily cycle
    assert F.gift_season("bizitobs_service/10S/short") == 360.0
    from timejepa.evaluation import gift
    assert all(F.gift_season(c) is not None for c in gift.GIFT_CONFIGS)
    assert {c.split("/")[0] for c in gift.GIFT_CONFIGS} == set(F.GIFT_DOMAINS)


# ------------------------------------------------------------ 2. table
def test_table_forms(tmp_path):
    p = tmp_path / "freq.yaml"
    p.write_text("a: {freq: 15T}\nb: {freq: D, weekly: true}\nc: H\nd: {season: 50}\ne: null\n")
    assert F.load_frequency_table(p) == {"a": 96.0, "b": 7.0, "c": 24.0, "d": 50.0, "e": 0.0}
    p.write_text("a: {freq: fortnight}\n")
    with pytest.raises(ValueError, match="no usable frequency"):
        F.load_frequency_table(p)
    p.write_text("a: {freq: D, weekly: REVIEW}\n")
    with pytest.raises(ValueError, match="weekly"):
        F.load_frequency_table(p)


def test_table_builder_rules(tmp_path, monkeypatch):
    import build_frequency_table as B
    assert B.coarser("5T", 3) == "15T" and B.coarser("H", 2) == "2H" and B.coarser("D", 2) == "2D"
    monkeypatch.setattr(B, "lotsa_frequency", lambda subset, revision: {"taxi_30min": "30T", "nn5_daily": "D"}[subset])
    cache = {}
    assert B.resolve("synthetic_lowfreq_s22", "r", cache) == (None, "synthetic")
    assert B.resolve("synthetic_subhourly_dec2", "r", cache)[0] is None
    assert B.resolve("chronos_ercot_dec3", "r", cache)[0] == "3H"
    assert B.resolve("taxi_30min", "r", cache) == ("30T", "lotsa")
    for name in ("synthetic_a", "taxi_30min", "nn5_daily", "chronos_dominick"):
        (tmp_path / f"{name}.npy").touch()
    out = tmp_path / "freq.yaml"
    monkeypatch.setattr(sys, "argv", ["x", "--corpus-dir", str(tmp_path), "--out", str(out)])
    B.main()
    text = out.read_text()
    assert "synthetic_a: null" in text and 'taxi_30min: {freq: "30T"}' in text and "weekly: REVIEW" in text
    with pytest.raises(ValueError, match="weekly"):                # unusable until reviewed
        F.load_frequency_table(out)
    out.write_text(text.replace("REVIEW", "true"))
    assert F.load_frequency_table(out) == {"synthetic_a": 0.0, "taxi_30min": 48.0, "nn5_daily": 7.0,
                                           "chronos_dominick": 365.0 / 7.0}


# ------------------------------------------------------------ 3. data
def _file(tmp_path, name="toy", n_series=3, length=2048, seed=0):
    arr = np.random.default_rng(seed).normal(size=(n_series, length)).astype(np.float32)
    np.save(tmp_path / f"{name}.npy", arr)
    return tmp_path / f"{name}.npy"


def test_item_carries_the_season_only_when_asked(tmp_path):
    path = _file(tmp_path)
    kw = dict(context_length=1024, prediction_length=256, stride=64)
    plain, tied = TimeSeriesDataset(path, **kw), TimeSeriesDataset(path, season_length=96.0, **kw)
    assert "season" not in plain[0]
    assert set(tied[0]) == set(plain[0]) | {"season"}
    assert tied[0]["season"] == np.float32(96.0)
    for i in (0, len(plain) // 2, len(plain) - 1):
        assert torch.equal(plain[i]["context"], tied[i]["context"])
        assert torch.equal(plain[i]["target"], tied[i]["target"])
    batch = torch.utils.data.default_collate([tied[i] for i in range(4)])
    assert batch["season"].shape == (4,) and batch["season"].dtype == torch.float32
    unknown = TimeSeriesDataset(path, season_length=0.0, **kw)
    assert unknown[0]["season"] == np.float32(0.0)


def test_a_strided_window_sees_a_shorter_season(tmp_path):
    path = _file(tmp_path, length=8192)
    ds = TimeSeriesDataset(path, context_length=1024, prediction_length=256, stride=64,
                           multi_resolution_factors=[1, 2], p_multi_resolution=1.0, season_length=96.0)
    seen = set()
    for _ in range(40):
        item = ds.get_item(0, allow_multi_resolution=True)
        assert item["season"] == np.float32(96.0 / item["resolution_factor"])
        seen.add(item["resolution_factor"])
    assert seen == {1, 2}


def _datamodule(tmp_path, table=None):
    from timejepa.data.datamodule import MultiDatasetMonashDataModule
    dm = MultiDatasetMonashDataModule(
        data_dir=tmp_path, context_length=512, prediction_length=128, batch_size=8, stride=64,
        normalize_mode="global", normalizer_type="identity", clip_outliers=False,
        train_val_test_split=(0.8, 0.1, 0.1), num_workers=0, seed=3, frequency_table=table)
    dm.prepare_data()
    dm.setup("fit")
    return dm


def test_datamodule_with_and_without_the_table_yields_the_same_windows(tmp_path):
    _file(tmp_path, "hourly", seed=1)
    _file(tmp_path, "synthetic", seed=2)
    table = tmp_path / "freq.yaml"
    table.write_text("hourly: {freq: H}\nsynthetic: null\n")
    plain, tied = _datamodule(tmp_path), _datamodule(tmp_path, str(table))
    assert plain.dataset_names_order == tied.dataset_names_order
    assert len(plain.train_dataset) == len(tied.train_dataset)
    expected = {"hourly": 24.0, "synthetic": 0.0}
    bounds = np.cumsum([0] + list(tied.train_dataset_sizes))
    for d, name in enumerate(tied.dataset_names_order):
        for i in (int(bounds[d]), int(bounds[d + 1]) - 1):
            a, b = plain.train_dataset[i], tied.train_dataset[i]
            assert "season" not in a and float(b["season"]) == expected[name]
            assert torch.equal(a["context"], b["context"]) and torch.equal(a["target"], b["target"])
    a, b = plain.val_dataset[0], tied.val_dataset[0]
    assert torch.equal(a["context"], b["context"]) and "season" in b


def test_a_file_missing_from_the_table_is_an_error(tmp_path):
    _file(tmp_path, "hourly")
    _file(tmp_path, "forgotten", seed=2)
    table = tmp_path / "freq.yaml"
    table.write_text("hourly: {freq: H}\n")
    with pytest.raises(ValueError, match="forgotten"):
        _datamodule(tmp_path, str(table))


# ------------------------------------------------------------ 4. harness
class _Patching:
    stride, patch_size = 8, 16


class _RateStub:
    """Persistence, whatever w: only the calls matter here."""
    input_length = 256
    rate_knob = "delta"

    def __init__(self, knob=True, delta_range=None):
        self.patching = _Patching()
        self.predictor = type("P", (), {"w_film": None})()
        if not knob:
            self.rate_knob = None
        if delta_range is not None:
            self.delta_range = delta_range
        self.calls = []

    def forecast(self, batch, n=None, w=None, **kw):
        x = batch[..., 0].cpu().numpy()
        self.calls.append({"len": x.shape[1], "n": n,
                           "w": None if w is None else float(w.reshape(-1)[0])})
        med = np.repeat(x[:, -1:], n, axis=1)
        fan = med[:, :, None] + 0.5 * np.linspace(-1.3, 1.3, 9)[None, None, :]
        return {"forecast_denorm": torch.tensor(med, dtype=torch.float32).unsqueeze(-1),
                "quantiles_denorm": torch.tensor(fan, dtype=torch.float32),
                "quantile_levels": LEVELS}


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    rng = np.random.default_rng(3)
    t = np.arange(900)
    series = [(rng.uniform(5, 50) + rng.uniform(1, 5) * np.sin(2 * np.pi * t / PERIOD)
               + rng.normal(scale=0.2, size=900)).astype(np.float32) for _ in range(6)]
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 48)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: PERIOD)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/15T/short", Path("."), torch.device("cpu"),
                              batch_size=8, **kw)


@pytest.mark.parametrize("mode", [{}, {"tta_flip": True}, {"ratein_mode": "backtest"},
                                  {"ratein_mode": "mix", "ratein_pool": True}])
def test_without_the_flag_calls_and_results_are_unchanged(harness, mode):
    a, b = _RateStub(), _RateStub()
    ra = _run(harness, a, **mode)
    rb = _run(harness, b, freq_season=None, **mode)
    assert all(c["w"] is None for c in a.calls)
    assert a.calls == b.calls
    assert ra["model"] == rb["model"]


def test_with_the_flag_every_forecast_gets_the_tied_scale(harness):
    m = _RateStub()
    _run(harness, m, freq_season=96.0)
    assert m.calls and all(c["w"] == pytest.approx(0.25) and c["len"] == 256 for c in m.calls)
    # with RateIN by decimation: a context decimated by k runs at w * k, clamped
    m = _RateStub()
    _run(harness, m, freq_season=96.0, ratein_mode="backtest")
    seen = {(c["len"], c["w"]) for c in m.calls}
    assert any(length < 256 for length, _ in seen)                # some candidates were decimated
    for c in m.calls:
        assert c["w"] is not None and 0.25 - 1e-6 <= c["w"] <= 4.0
    assert {round(c["w"], 4) for c in m.calls if c["len"] == 256 and c["n"] in (48,)} >= {0.25}
    # the model's own trained range is honoured
    m = _RateStub(delta_range=(0.5, 2.0))
    _run(harness, m, freq_season=96.0)
    assert all(c["w"] == 0.5 for c in m.calls)


def test_tied_scale_of_a_decimated_context():
    import evaluate_gift as EG
    stub = _RateStub()
    assert EG.freq_delta_w(stub, 96.0, 1) == 0.25
    assert EG.freq_delta_w(stub, 96.0, 4) == 1.0                  # 15T decimated by 4 is hourly
    assert EG.freq_delta_w(stub, 96.0, 48) == 4.0                 # clamped
    assert EG.freq_delta_w(stub, 24.0, 0.5) == 0.5                # upsampled by 2


def test_guards_and_flag_registration():
    from evaluate_gift import KNOWN_FLAGS, check_model_flags
    assert "freq_delta" in KNOWN_FLAGS
    check_model_flags(_RateStub(), "off", False, None, None, True)
    check_model_flags(_RateStub(), "mix", False, None, None, True)
    with pytest.raises(ValueError, match="rate knob"):
        check_model_flags(_RateStub(knob=False), "off", False, None, None, True)
    with pytest.raises(ValueError, match="exclusive"):
        check_model_flags(_RateStub(), "delta", False, None, None, True)
    with pytest.raises(ValueError, match="exclusive"):
        check_model_flags(_RateStub(), "off", True, None, None, True)
    check_model_flags(_RateStub(knob=False), "off", False, None, None)       # default: no new demand


# ------------------------------------------------------------ 5. per-row seasons of the synthetic files
def test_recording_the_season_does_not_change_the_series():
    from timejepa.data.synthetic import V3_FAMILIES, sample_series
    for spec in V3_FAMILIES:
        a = [sample_series(spec, np.random.default_rng(5)) for _ in range(3)]
        rng, infos = np.random.default_rng(5), []
        b = []
        for _ in range(3):
            info = {}
            b.append(sample_series(spec, rng, info))
            infos.append(info)
        rng_plain = np.random.default_rng(5)
        a = [sample_series(spec, rng_plain) for _ in range(3)]
        assert all(np.array_equal(x, y) for x, y in zip(a, b)), spec.name
        assert all("season" in i and i["season"] >= 0 for i in infos)
        if spec.kind == "kernel":
            assert any(i["season"] > 0 for i in infos)
            for i in infos:
                if i["season"] > 0:
                    assert spec.period_range[0] <= i["season"] <= spec.period_range[1]


def test_season_sidecars_are_built_by_replay_and_verified(tmp_path, monkeypatch):
    import build_season_sidecars as B
    from timejepa.data.synthetic import DEFAULT_FAMILIES, write_synthetic_family
    spec = next(f for f in DEFAULT_FAMILIES if f.name == "synthetic_lowfreq")
    small = type(spec)(spec.name, chunk_length=512, period_range=spec.period_range, p_trend=spec.p_trend)
    monkeypatch.setitem(B.SPECS, spec.name, small)
    write_synthetic_family(tmp_path / "synthetic_lowfreq_s22.npy", small, n_chunks=12, seed=22000)
    write_synthetic_family(tmp_path / "synthetic_lowfreq.npy", small, n_chunks=12, seed=2)      # v1 rule: index 2
    arr = np.load(tmp_path / "synthetic_lowfreq.npy")
    np.save(tmp_path / "synthetic_lowfreq_dec2.npy", arr.reshape(12, 256, 2).mean(axis=2).astype(np.float32))
    assert B.family_and_seed("synthetic_lowfreq_s22") == ("synthetic_lowfreq", 22000)
    assert B.family_and_seed("synthetic_lowfreq") == ("synthetic_lowfreq", 2)
    assert B.family_and_seed("synthetic_ops_bursty_s3") == ("synthetic_ops_bursty", 3000)
    assert B.family_and_seed("beijing_air_quality") is None
    monkeypatch.setattr(sys, "argv", ["x", "--corpus-dir", str(tmp_path), "--jobs", "1"])
    B.main()
    s22 = np.load(tmp_path / "_season" / "synthetic_lowfreq_s22.npy")
    s1 = np.load(tmp_path / "_season" / "synthetic_lowfreq.npy")
    d2 = np.load(tmp_path / "_season" / "synthetic_lowfreq_dec2.npy")
    assert s22.shape == (12,) and s22.dtype == np.float32 and (s22 >= 0).all() and (s22 > 0).any()
    assert np.allclose(d2, s1 / 2)
    # a file that does not replay (one value changed) is refused, nothing written for it
    bad = np.load(tmp_path / "synthetic_lowfreq_s22.npy"); bad[3, 10] += 1.0
    np.save(tmp_path / "synthetic_lowfreq_s22.npy", bad)
    (tmp_path / "_season" / "synthetic_lowfreq_s22.npy").unlink()
    with pytest.raises(RuntimeError, match="row 3 differs"):
        B.replay(tmp_path / "synthetic_lowfreq_s22.npy")
    assert not (tmp_path / "_season" / "synthetic_lowfreq_s22.npy").exists()


def test_dataset_reads_per_row_seasons_from_the_sidecar(tmp_path):
    path = _file(tmp_path, "syn", n_series=4, length=2048)
    (tmp_path / "_season").mkdir()
    np.save(tmp_path / "_season" / "syn.npy", np.array([24.0, 0.0, 96.0, 7.0], dtype=np.float32))
    kw = dict(context_length=1024, prediction_length=256, stride=64)
    ds = TimeSeriesDataset(path, season_length=F.PER_ROW, **kw)
    seen = {}
    for i in range(len(ds)):
        item = ds[i]
        seen.setdefault(int(item["series_id"]), set()).add(float(item["season"]))
    assert seen == {0: {24.0}, 1: {0.0}, 2: {96.0}, 3: {7.0}}
    plain = TimeSeriesDataset(path, **kw)
    assert torch.equal(plain[5]["context"], ds[5]["context"])
    with pytest.raises(FileNotFoundError, match="per_row"):
        TimeSeriesDataset(_file(tmp_path, "nosidecar", seed=9), season_length=F.PER_ROW, **kw)
    np.save(tmp_path / "_season" / "syn.npy", np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="rows"):
        TimeSeriesDataset(path, season_length=F.PER_ROW, **kw)


def test_table_per_row_entry(tmp_path):
    p = tmp_path / "freq.yaml"
    p.write_text("a: {freq: H}\nb: {season: per_row}\nc: null\n")
    assert F.load_frequency_table(p) == {"a": 24.0, "b": F.PER_ROW, "c": 0.0}
