"""
GIFT-Eval protocol, reimplemented faithfully and self-contained.

Every constant and formula below is transcribed from the official harness
(github.com/SalesforceAIResearch/gift-eval, src/gift_eval/data.py, Apache-2)
and from gluonts' seasonality table — fetched and pinned on 2026-08-18.
The official code is NOT imported: it drags gluonts + dotenv + toolz into the
project for what amounts to one split rule and two metric definitions. Instead
the rules are restated here verbatim, each next to its source, so a diff
against upstream stays a five-minute job.

The protocol, in full
---------------------
* 97 configs, named ``dataset/freq/term`` (``GIFT_CONFIGS``). The list is not
  derived — it is the exact row set of the official leaderboard CSVs.
* ``prediction_length = base_horizon[freq] * multiplier[term]`` with the M4
  datasets on their own horizon table, and multipliers 1 / 10 / 15 for
  short / medium / long.
* Test windows per dataset: ``min(max(1, ceil(0.1 * min_series_length / h)), 20)``
  and always exactly 1 for M4. Windows are the LAST ``windows * h`` steps of
  each series, non-overlapping, stride ``h``; everything before a window is
  its context.
* Multivariate datasets are exploded into univariate series (one per channel),
  exactly like the official ``MultivariateToUnivariate`` — the project is
  univariate by design, and so is the official evaluation of univariate models.
* Metrics: MASE (per-instance seasonal scaling on the FULL past, gluonts
  convention) and mean weighted sum quantile loss ("CRPS" on the leaderboard,
  9 quantiles 0.1..0.9 — the decoder's native grid). NaN targets are masked,
  never imputed: the *_with_missing datasets are scored only where truth exists.
* Leaderboard aggregation: each config's metric is divided by Seasonal Naive's
  and the 97 ratios are combined by geometric mean. Two normalizations are
  reported side by side: against the OFFICIAL Seasonal Naive numbers (vendored
  in assets/gift_seasonal_naive.csv, the leaderboard's own denominators) and
  against a Seasonal Naive computed HERE through the same code path (immune to
  any convention drift between this module and gluonts).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — transcribed from gift_eval/data.py
# ---------------------------------------------------------------------------

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}

# gluonts DEFAULT_SEASONALITIES (time_feature/seasonality.py). The rule for a
# multiple is base // n when it divides, else 1 — e.g. 15T -> 1440/15 = 96.
DEFAULT_SEASONALITIES = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1,
                         "M": 12, "B": 5, "Q": 4}

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# The 97 configs, exactly the row set of the official results CSVs. Kept as a
# literal rather than generated: the benchmark is versioned by this list, and a
# generator would hide any upstream drift instead of failing loudly on it.
GIFT_CONFIGS: Tuple[str, ...] = (
    "bitbrains_fast_storage/5T/long", "bitbrains_fast_storage/5T/medium",
    "bitbrains_fast_storage/5T/short", "bitbrains_fast_storage/H/short",
    "bitbrains_rnd/5T/long", "bitbrains_rnd/5T/medium",
    "bitbrains_rnd/5T/short", "bitbrains_rnd/H/short",
    "bizitobs_application/10S/long", "bizitobs_application/10S/medium",
    "bizitobs_application/10S/short",
    "bizitobs_l2c/5T/long", "bizitobs_l2c/5T/medium", "bizitobs_l2c/5T/short",
    "bizitobs_l2c/H/long", "bizitobs_l2c/H/medium", "bizitobs_l2c/H/short",
    "bizitobs_service/10S/long", "bizitobs_service/10S/medium",
    "bizitobs_service/10S/short",
    "car_parts/M/short", "covid_deaths/D/short",
    "electricity/15T/long", "electricity/15T/medium", "electricity/15T/short",
    "electricity/D/short",
    "electricity/H/long", "electricity/H/medium", "electricity/H/short",
    "electricity/W/short",
    "ett1/15T/long", "ett1/15T/medium", "ett1/15T/short", "ett1/D/short",
    "ett1/H/long", "ett1/H/medium", "ett1/H/short", "ett1/W/short",
    "ett2/15T/long", "ett2/15T/medium", "ett2/15T/short", "ett2/D/short",
    "ett2/H/long", "ett2/H/medium", "ett2/H/short", "ett2/W/short",
    "hierarchical_sales/D/short", "hierarchical_sales/W/short",
    "hospital/M/short",
    "jena_weather/10T/long", "jena_weather/10T/medium", "jena_weather/10T/short",
    "jena_weather/D/short",
    "jena_weather/H/long", "jena_weather/H/medium", "jena_weather/H/short",
    "kdd_cup_2018/D/short",
    "kdd_cup_2018/H/long", "kdd_cup_2018/H/medium", "kdd_cup_2018/H/short",
    "loop_seattle/5T/long", "loop_seattle/5T/medium", "loop_seattle/5T/short",
    "loop_seattle/D/short",
    "loop_seattle/H/long", "loop_seattle/H/medium", "loop_seattle/H/short",
    "m4_daily/D/short", "m4_hourly/H/short", "m4_monthly/M/short",
    "m4_quarterly/Q/short", "m4_weekly/W/short", "m4_yearly/A/short",
    "m_dense/D/short",
    "m_dense/H/long", "m_dense/H/medium", "m_dense/H/short",
    "restaurant/D/short",
    "saugeen/D/short", "saugeen/M/short", "saugeen/W/short",
    "solar/10T/long", "solar/10T/medium", "solar/10T/short",
    "solar/D/short",
    "solar/H/long", "solar/H/medium", "solar/H/short",
    "solar/W/short",
    "sz_taxi/15T/long", "sz_taxi/15T/medium", "sz_taxi/15T/short",
    "sz_taxi/H/short",
    "temperature_rain/D/short",
    "us_births/D/short", "us_births/M/short", "us_births/W/short",
)

# Leaderboard name -> directory name inside the Salesforce/GiftEval snapshot
# (the storage layout follows cli/conf/analysis/datasets/all_datasets.yaml,
# which spells some datasets differently from the results CSVs).
_STORAGE_DATASET = {
    "loop_seattle": "LOOP_SEATTLE",
    "sz_taxi": "SZ_TAXI",
    "m_dense": "M_DENSE",
    "temperature_rain": "temperature_rain_with_missing",
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
    "saugeen": "saugeenday",
}

# Datasets stored as a single directory, without a per-frequency subdirectory.
_NO_FREQ_SUBDIR = {
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily",
    "m4_hourly", "hospital", "covid_deaths", "car_parts_with_missing",
    "restaurant", "temperature_rain_with_missing",
    "bizitobs_application", "bizitobs_service",
}


# ---------------------------------------------------------------------------
# Per-config derivations
# ---------------------------------------------------------------------------

def _freq_base_and_step(freq: str) -> Tuple[str, int]:
    """'15T' -> ('T', 15); 'H' -> ('H', 1); 'W-WED' -> ('W', 1)."""
    head = freq.split("-")[0]
    digits = "".join(c for c in head if c.isdigit())
    base = head[len(digits):] or head
    return base.upper() if base not in ("min",) else "T", int(digits or 1)


def prediction_length(config: str) -> int:
    dataset, freq, term = config.split("/")
    base, _ = _freq_base_and_step(freq)
    table = M4_PRED_LENGTH_MAP if dataset.startswith("m4") else PRED_LENGTH_MAP
    return table[base] * TERM_MULTIPLIER[term]


def seasonality(freq: str) -> int:
    """gluonts get_seasonality: base seasonality // step when it divides."""
    base, step = _freq_base_and_step(freq)
    s, rem = divmod(DEFAULT_SEASONALITIES.get(base, 1), step)
    return s if rem == 0 else 1


def storage_path(config: str) -> str:
    """Relative directory of this config's data inside the GiftEval snapshot."""
    dataset, freq, _ = config.split("/")
    stored = _STORAGE_DATASET.get(dataset, dataset)
    if stored in _NO_FREQ_SUBDIR:
        return stored
    return f"{stored}/{freq}"


def num_windows(config: str, min_series_length: int) -> int:
    if config.split("/")[0].startswith("m4"):
        return 1
    h = prediction_length(config)
    w = math.ceil(TEST_SPLIT * min_series_length / h)
    return min(max(1, w), MAX_WINDOW)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(gift_root: Path, config: str) -> List[np.ndarray]:
    """
    All series of a config as float32 1-D arrays (NaN preserved).

    Multivariate targets are exploded channel-wise, mirroring the official
    MultivariateToUnivariate — channel k of series i becomes its own series,
    in (series, channel) order so item counts match the official expansion.
    """
    import datasets as hf_datasets  # local import: only this loader needs it

    path = gift_root / storage_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — download the benchmark first:\n"
            f"  make gift-download"
        )
    ds = hf_datasets.load_from_disk(str(path)).with_format("numpy")

    out: List[np.ndarray] = []
    for row in ds:
        target = np.asarray(row["target"], dtype=np.float32)
        if target.ndim == 1:
            out.append(target)
        else:
            out.extend(target[k] for k in range(target.shape[0]))
    return out


@dataclass
class TestInstance:
    """One rolling-evaluation window: everything-before as context, h as target."""
    series_idx: int
    context: np.ndarray   # full past, length varies
    target: np.ndarray    # length = prediction_length, may contain NaN


def iter_test_instances(series: Sequence[np.ndarray], h: int,
                        windows: int) -> Iterator[TestInstance]:
    """
    The official split: the last ``windows * h`` steps of each series are test,
    cut into non-overlapping windows of length h (distance = h), each forecast
    from the entire history before it.
    """
    for idx, y in enumerate(series):
        for k in range(windows):
            start = len(y) - (windows - k) * h
            if start <= 0:
                continue
            target = y[start:start + h]
            if np.all(np.isnan(target)):
                continue
            yield TestInstance(idx, y[:start], target)


# ---------------------------------------------------------------------------
# Metrics — gluonts conventions, NaN-masked
# ---------------------------------------------------------------------------

def seasonal_error(past: np.ndarray, m: int) -> float:
    """Mean |y_t - y_{t-m}| over the full past (gluonts MASE denominator)."""
    if len(past) <= m:
        m = 1
    if len(past) <= 1:
        return np.nan
    diff = np.abs(past[m:] - past[:-m])
    diff = diff[~np.isnan(diff)]
    return float(diff.mean()) if diff.size else np.nan


@dataclass
class MetricAccumulator:
    """
    Streams per-instance results into the two leaderboard metrics.

    MASE  — mean over instances of mean_t|err| / seasonal_error(past).
    CRPS  — mean over quantiles of  sum(2·QL_q) / sum(|y|), pooled over the
            whole config (gluonts mean_weighted_sum_quantile_loss).
    """
    mase_terms: List[float] = field(default_factory=list)
    ql_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(QUANTILE_LEVELS), dtype=np.float64))
    abs_sum: float = 0.0
    n_instances: int = 0
    n_skipped_scale: int = 0
    # Pooled sums for the point metrics of the official results format
    # (gluonts axis=None aggregation: pooled over every observation).
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    smape_sum: float = 0.0
    mape_sum: float = 0.0
    n_obs: int = 0
    n_obs_nonzero: int = 0
    # Couverture empirique par niveau : count(y <= q_k) poolé sur la config.
    # Instrument du critère ESJEPA/E21 (« la couverture généralise-t-elle vers
    # le nominal ? ») — une MESURE, jamais une adaptation. n_obs_q sépare le
    # dénominateur : les instances sans fan (point forecast) n'y votent pas.
    cov_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(QUANTILE_LEVELS), dtype=np.float64))
    n_obs_q: int = 0

    def add(self, target: np.ndarray, median: np.ndarray,
            quantiles: Optional[np.ndarray], scale: float) -> None:
        mask = ~np.isnan(target)
        if not mask.any():
            return
        self.n_instances += 1

        y, yhat = target[mask], median[mask]
        err = np.abs(y - yhat)
        if np.isfinite(scale) and scale > 0:
            self.mase_terms.append(float(err.mean()) / scale)
        else:
            self.n_skipped_scale += 1

        self.abs_err_sum += float(err.sum())
        self.sq_err_sum += float(((y - yhat) ** 2).sum())
        self.n_obs += int(mask.sum())
        denom = np.abs(y) + np.abs(yhat)
        pos = denom > 0
        self.smape_sum += float((2.0 * err[pos] / denom[pos]).sum())
        nz = np.abs(y) > 0
        self.mape_sum += float((err[nz] / np.abs(y[nz])).sum())
        self.n_obs_nonzero += int(nz.sum())

        self.abs_sum += float(np.abs(target[mask]).sum())
        if quantiles is not None:
            y = target[mask][:, None]                       # [t, 1]
            q_pred = quantiles[mask]                        # [t, Q]
            q = np.asarray(QUANTILE_LEVELS)[None, :]        # [1, Q]
            ql = 2.0 * np.abs((q_pred - y) * ((y <= q_pred) - q))
            self.ql_sums += ql.sum(axis=0)
            self.cov_counts += (y <= q_pred).sum(axis=0)
            self.n_obs_q += q_pred.shape[0]
        else:
            # Point forecast: every quantile collapses onto the median.
            y = target[mask]
            e = median[mask] - y
            for j, qv in enumerate(QUANTILE_LEVELS):
                self.ql_sums[j] += float(
                    2.0 * np.abs(e * ((y <= median[mask]) - qv)).sum())

    def result(self) -> Dict[str, float]:
        mase = float(np.mean(self.mase_terms)) if self.mase_terms else np.nan
        crps = (float((self.ql_sums / self.abs_sum).mean())
                if self.abs_sum > 0 else np.nan)
        n = max(self.n_obs, 1)
        mse = self.sq_err_sum / n
        mae = self.abs_err_sum / n
        return {"MASE": mase, "CRPS": crps,
                "MSE": mse, "MAE": mae, "RMSE": math.sqrt(mse),
                "NRMSE": (math.sqrt(mse) / (self.abs_sum / n)
                          if self.abs_sum > 0 else np.nan),
                "ND": (self.abs_err_sum / self.abs_sum
                       if self.abs_sum > 0 else np.nan),
                "MAPE": (self.mape_sum / self.n_obs_nonzero
                         if self.n_obs_nonzero else np.nan),
                "sMAPE": self.smape_sum / n,
                "n_instances": self.n_instances,
                "n_skipped_scale": self.n_skipped_scale,
                "coverage": ({f"{lv:.1f}": float(self.cov_counts[j] / self.n_obs_q)
                              for j, lv in enumerate(QUANTILE_LEVELS)}
                             if self.n_obs_q else None)}


def seasonal_naive_forecast(context: np.ndarray, h: int, m: int) -> np.ndarray:
    """Repeat the last observed seasonal cycle (NaN-tolerant, ffill fallback)."""
    if len(context) >= m:
        tile = context[-m:]
        out = np.tile(tile, h // m + 1)[:h].astype(np.float32)
    else:
        out = np.full(h, context[-1], dtype=np.float32)
    # A NaN in the copied cycle would poison the loss — fall back to the last
    # finite value, which is what gluonts' predictor effectively does.
    if np.isnan(out).any():
        finite = context[~np.isnan(context)]
        fill = finite[-1] if finite.size else 0.0
        out = np.where(np.isnan(out), fill, out)
    return out


# ---------------------------------------------------------------------------
# Leaderboard aggregation
# ---------------------------------------------------------------------------

_ASSETS = Path(__file__).parent / "assets"


def official_seasonal_naive() -> Dict[str, Dict[str, float]]:
    """Official per-config Seasonal Naive numbers (the leaderboard denominators)."""
    path = _ASSETS / "gift_seasonal_naive.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent — le CSV est versionné avec le package (exception à la "
            f"règle *.csv du .gitignore). Un checkout incomplet ? `git pull` d'abord ; "
            f"sinon: curl -sL https://raw.githubusercontent.com/SalesforceAIResearch/"
            f"gift-eval/main/results/seasonal_naive/all_results.csv -o {path}"
        )
    out: Dict[str, Dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = {
                "MASE": float(row["eval_metrics/MASE[0.5]"]),
                "CRPS": float(row["eval_metrics/mean_weighted_sum_quantile_loss"]),
            }
    return out


def geometric_mean(values: Sequence[float]) -> float:
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0])
    return float(np.exp(np.log(v).mean())) if v.size else np.nan


def aggregate(results: Dict[str, Dict[str, float]],
              baseline: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Geometric mean of per-config metric ratios vs a Seasonal Naive table."""
    common = [c for c in results if c in baseline]
    out = {}
    for metric in ("MASE", "CRPS"):
        ratios = [results[c][metric] / baseline[c][metric]
                  for c in common
                  if np.isfinite(results[c][metric]) and baseline[c][metric] > 0]
        out[f"geomean_{metric}_ratio"] = geometric_mean(ratios)
        out[f"n_configs_{metric}"] = len(ratios)
    return out
