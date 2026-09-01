"""
LOTSA -> TimeJEPA format.

LOTSA (Moirai's pretrain corpus, ~27B observations, HuggingFace
`Salesforce/lotsa_data`) is two orders of magnitude above the current Monash
corpus and covers ALL frequencies where the current corpus is sub-daily only.
E10 measured that two high-frequency datasets carry 48.7% of the pretrain
batch; LOTSA is meant to fix that imbalance.

Three design constraints, each traced to a project incident.

1. No `object` arrays. B19: numpy object arrays refcount every element, which
   breaks `fork` copy-on-write and blows up RAM with multiple workers (51 GiB
   observed). At LOTSA scale that would be fatal. The conversion emits ONLY
   DENSE float32 arrays, segmenting long series into fixed-length chunks whose
   pages are shared across processes without copies.

2. Memmap reads. Files are GBs: `TimeSeriesDataset(use_mmap=True)` reads them
   without loading them into RAM. The parameter is additive and defaults to
   False, so existing configs are bit-identical.

3. Exclusion of evaluation datasets. LOTSA contains ETT, electricity, traffic,
   weather... i.e. part of our benchmarks. Pretraining on them would invalidate
   the whole evaluation. `EVAL_OVERLAP_PATTERNS` filters by substring, and the
   conversion PRINTS what it excludes: review that output before launching a
   pretrain, since the LOTSA name list is not frozen.

Cost of segmentation: a window cannot straddle two chunks. With `chunk_length`
8192 and a 1280 window, at most ~15% of possible positions are lost at chunk
boundaries, and zero when series are shorter than `chunk_length` (they pass
whole).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Substrings of LOTSA subset names that overlap an evaluation benchmark.
#
# Two SEPARATE lists because they have different provenances and are verified
# differently. Their union is `EVAL_OVERLAP_PATTERNS`.
#
# WHY THIS LIST IS CRITICAL: a subset forgotten here is a benchmark seen during
# pretrain, and a "zero-shot" evaluation that is not one. The project's Monash
# corpus has exactly that flaw: electricity-hourly and traffic-hourly ARE the
# series and time window of the Nixtla `electricity` and `traffic` benchmarks
# (5260/26304 and 3508/17544 = the last 20%), and the train/val/test split is
# sequential over per-series indices, so training covers those series over
# their full span. See section 5 of the experiment register. The LOTSA protocol
# fixes this by construction.
#
# GIFT-EVAL LIST VERIFIED 2026-08-13 against the official repo
# (https://huggingface.co/api/datasets/Salesforce/GiftEval/tree/main), whose 28
# directories are: LOOP_SEATTLE, M_DENSE, SZ_TAXI, bitbrains_fast_storage,
# bitbrains_rnd, bizitobs_application, bizitobs_l2c, bizitobs_service,
# car_parts_with_missing, covid_deaths, electricity, ett1, ett2,
# hierarchical_sales, hospital, jena_weather, kdd_cup_2018_with_missing,
# m4_{daily,hourly,monthly,quarterly,weekly,yearly}, restaurant, saugeenday,
# solar, temperature_rain_with_missing, us_births.
# The patterns below cover all 28 (non-regression test).
#
# Note: GIFT-Eval policy allows training on THEIR train split ("carefully
# constructed using earlier horizons that do not overlap with the test set")
# and only requires declaring leakage when the corpus contains a test-corpus
# dataset. Excluding the WHOLE dataset is therefore more conservative than the
# required minimum - a deliberate choice, since we cannot align our splits with
# theirs.

# Nixtla long-horizon benchmarks (what scripts/evaluate.py measures today).
NIXTLA_OVERLAP_PATTERNS: Tuple[str, ...] = (
    "ett",            # ETTh1/2, ETTm1/2
    "electricity",    # ECL
    "traffic",        # traffic
    "weather",        # weather (and jena_weather on the GIFT-Eval side)
    "exchange",       # exchange rate
    "illness", "ili",  # ILI
)

# GIFT-Eval sources, required for any zero-shot claim on that benchmark.
GIFT_EVAL_OVERLAP_PATTERNS: Tuple[str, ...] = (
    "m4", "m3", "m1",   # M competitions
    "m_dense",
    "bizitobs",
    "bitbrains",
    "car_parts",
    "covid",
    "hierarchical",     # hierarchical_sales
    "hospital",
    "jena",             # jena_weather
    "kdd",              # kdd_cup_2018
    "air_quality",      # kdd_cup_2018 IS Beijing air quality 2017-2018;
                        # beijing_air_quality and china_air_quality measure the
                        # same thing and would near-duplicate a GIFT-Eval
                        # dataset. Spotted while reviewing --list output.
    "loop_seattle", "seattle",
    "restaurant",
    "saugeen",          # ALSO present in the local Monash corpus
    "solar",            # same
    "taxi",             # sz_taxi
    "temperature_rain",
    "birth",            # us_births
    "tourism",
    "nn5",
    "wiki",             # wikipedia web traffic
)

EVAL_OVERLAP_PATTERNS: Tuple[str, ...] = tuple(
    dict.fromkeys(NIXTLA_OVERLAP_PATTERNS + GIFT_EVAL_OVERLAP_PATTERNS)
)

# Subsets READMITTED despite matching a pattern (G8.1).
# -------------------------------------------------------------
# Patterns are substrings, hence deliberately coarse: "taxi" catches sz_taxi
# (GIFT eval) AND taxi_30min (New York, unrelated), "m1" catches m1_monthly
# while the eval only covers M4. E17 measured the cost of that coarseness: our
# gap to the leaderboard tracks frequency coverage, and we were discarding
# frequencies for free.
#
# Safety reference: `Salesforce/GiftEvalPretrain`, the pretrain corpus
# SANCTIONED by the benchmark (152 subsets). Everything in it is declared
# non-leaking for GIFT-Eval by its authors.
#
# That is NOT sufficient on its own: we also evaluate on Nixtla and on 8 local
# Monash datasets that GIFT does not know about. Every entry below is therefore
# checked against ALL THREE suites, with the reason spelled out. Do not add
# anything here without doing the three checks.
EVAL_SAFE_OVERRIDES: Tuple[str, ...] = (
    # --- M1/M3 competitions: the eval only covers M4 -----------------------
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "monash_m3_monthly", "monash_m3_other",
    "monash_m3_quarterly", "monash_m3_yearly",
    # Low-frequency series, exactly where our worst MASE lives (m4_yearly
    # 5.08, m4_quarterly 1.54): the corpus has almost nothing yearly or
    # quarterly.

    # --- tourism: no eval suite contains it --------------------------------
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",

    # --- nn5: in NONE of the three current eval suites ---------------------
    # (nn5_daily served as a held-out transfer corpus in G4.6; that round is
    #  closed, and it appears in neither Nixtla, nor the 8 Monash, nor the 97
    #  GIFT configs.)
    "nn5_daily_with_missing", "nn5_weekly",

    # --- New York taxis, 30 min: the GIFT eval is SZ_TAXI (Shenzhen) -------
    "taxi_30min",

    # --- KDD Cup 2022: wind power at 10 MINUTES ----------------------------
    # The GIFT eval is kdd_cup_2018 (Beijing air quality): different
    # competition, domain, and year. And our biggest expected gain: 10T is the
    # frequency where we lose x1.94 (E17).
    "kdd2022",

    # --- COVID: the GIFT eval is covid_deaths ------------------------------
    "covid19_energy",     # electricity consumption during the pandemic
    "covid_mobility",     # mobility indices

    # --- Baidu traffic (China): the Nixtla eval is PEMS (California) -------
    "Q-TRAFFIC",

    # --- Australian electricity demand (30 min) ----------------------------
    # The Nixtla "electricity" eval is UCI/Portugal, the local Monash eval is
    # electricity-hourly (same UCI). The Australian market is a distinct
    # dataset, and GiftEvalPretrain sanctions it.
    "australian_electricity_demand",

    # --- solar_power: THE real 10T data of the v3 corpus (lifted 2026-08-24)
    # The three contract checks, one by one:
    # 1. GIFT-Eval: `solar_power` is in GiftEvalPretrain, the pretrain corpus
    #    SANCTIONED by the benchmark authors - declared non-leaking for their
    #    97 configs, solar/10T included (same warrant as kdd2022/taxi_30min
    #    above, reference G8.1).
    # 2. Nixtla: no solar benchmark among the 7 (electricity, etth1/2,
    #    ettm1/2, traffic, weather) - no overlap possible.
    # 3. LOCAL Monash suite: it evaluates solar-10-minute - the ONLY reason
    #    for the old exclusion, and that suite is DEPRECATED in favor of
    #    GIFT-Eval (scope decision explicitly anticipated by the note below:
    #    "they become admissible again"). Consequence to state: the solar line
    #    of the local Monash suite stops being zero-shot - that suite carries
    #    no publishable number anyway (m=1, section 1).
    "solar_power",

    # --- Chinese air quality -----------------------------------------------
    # The least clear-cut entry: the GIFT eval kdd_cup_2018 is also Beijing
    # air quality, and an overlap of TIME WINDOWS is conceivable.
    # GiftEvalPretrain includes both, which counts as the benchmark authors'
    # warrant. Remove at the slightest doubt - the gain (hourly, an already
    # well-covered frequency) is small and does not justify contamination
    # risk.
    "beijing_air_quality", "china_air_quality",
)

# What stays EXCLUDED among subsets that GIFT nonetheless sanctions, and why -
# this list matters as much as the previous one:
#   traffic_hourly, traffic_weekly  : traffic_hourly IS the Nixtla `traffic`
#       and the Monash `traffic-hourly`. Contamination measured and documented
#       in section 5.
#   weather                          : IS the Nixtla `weather`.
#   oikolab_weather                  : weather reanalysis, adjacent to
#       `weather` - and redundant, the corpus already has era5 (30 slices).
#   cdc_fluview_ilinet               : IS the source of the Nixtla
#       `illness`/ILI.
#   solar_power                      : the local Monash suite evaluates
#       solar-10-minute.
#   extended_web_traffic_with_missing, kaggle_web_traffic_weekly,
#   wiki-rolling_nips                : the local Monash suite evaluates
#       wikipedia-web-traffic-extended.
# The last four groups are blocked only by the LOCAL Monash suite, which
# section 1 notes runs at seasonality m=1 (i.e. against a weak baseline). If
# that suite is dropped in favor of GIFT-Eval, they become admissible again -
# a scope decision, not a safety one.


def is_eval_overlap(name: str, patterns: Sequence[str] = EVAL_OVERLAP_PATTERNS,
                    overrides: Sequence[str] = EVAL_SAFE_OVERRIDES) -> bool:
    """
    True if the LOTSA subset name overlaps an evaluation benchmark.

    A name listed in `overrides` is readmitted even when it matches a pattern:
    patterns are coarse substrings and catch unrelated homonyms (see
    EVAL_SAFE_OVERRIDES). The comparison is exact and case-insensitive - never
    a substring, or the override would reopen the hole the pattern closes.
    """
    lowered = name.lower()
    if lowered in {o.lower() for o in overrides}:
        return False
    return any(p in lowered for p in patterns)


def family_of(name: str) -> str:
    """
    Groups LOTSA subsets that are just time slices of one corpus:
    `cmip6_1850`...`cmip6_2010`, `era5_1989`...`era5_2018`,
    `largest_2017`...`largest_2021`, `gfc12_load`/`gfc14_load`/`gfc17_load`.

    Without this grouping, the per-subset cap protects nothing: E10 measured
    two datasets carrying 48.7% of the pretrain batch, and LOTSA would do worse
    at scale - cmip6 (33 slices) and era5 (30) are 63 subsets out of 123, half
    the corpus in climate reanalysis, a smooth seasonal signal far from the
    benchmark series.
    """
    import re
    # year suffix: _1850, _2018, or embedded as in gfc12_load
    base = re.sub(r"_(18|19|20)\d{2}$", "", name)
    base = re.sub(r"^gfc\d{2}_", "gfc_", base)
    return base


def segment_series(
    series: np.ndarray,
    chunk_length: int,
    min_length: int,
) -> List[np.ndarray]:
    """
    Cuts a series into fixed-length chunks so a dense array can be built.

    Chunks are NON-OVERLAPPING: overlap is the dataset's job (sliding windows);
    duplicating it here would inflate the file without adding information. The
    final remainder would be shorter than `chunk_length` and break density, so
    it is DROPPED - unless it is the only chunk (series shorter than
    `chunk_length` but long enough to be useful), in which case the whole
    series is returned and the caller groups by length.

    Returns a list of chunks of EXACTLY `chunk_length`, or a one-element list
    of length `len(series)` when the series is shorter.
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    n = series.shape[0]

    if n < min_length:
        return []
    if n < chunk_length:
        return [series]

    n_chunks = n // chunk_length
    return [series[i * chunk_length:(i + 1) * chunk_length] for i in range(n_chunks)]


def _finite(chunk: np.ndarray) -> bool:
    """LOTSA contains NaNs (gappy series). One NaN poisons RevIN and the loss."""
    return bool(np.isfinite(chunk).all())


def impute_gaps(chunk: np.ndarray, max_nan_fraction: float) -> Optional[np.ndarray]:
    """
    Fills gaps by linear interpolation, or gives up if the chunk is too gappy.

    Rejecting a whole chunk for a single NaN is untenable on a real corpus:
    measured on LOTSA, HZMETRO lost 160/160 chunks and SHMETRO 2304/2304 -
    100% of the subset for a few missing values. Gaps are the norm, not the
    exception (several subsets carry "_with_missing" in their name).

    Linear interpolation over short gaps is standard practice and stays honest;
    over long gaps it fabricates signal. Hence the threshold: beyond
    `max_nan_fraction`, losing the chunk beats inventing it.

    Returns the filled chunk, or None if it is too gappy / all-NaN.
    """
    finite = np.isfinite(chunk)
    n_missing = chunk.shape[0] - int(finite.sum())
    if n_missing == 0:
        return chunk
    if n_missing / chunk.shape[0] > max_nan_fraction:
        return None
    if not finite.any():
        return None

    filled = chunk.copy()
    idx = np.arange(chunk.shape[0])
    # np.interp extends with the edge value, which also handles gaps at the
    # start and end of the chunk (an implicit ffill/bfill).
    filled[~finite] = np.interp(idx[~finite], idx[finite], chunk[finite])
    return filled


def choose_chunk_length(
    sample_lengths: Sequence[int],
    requested: int,
    min_length: int,
) -> Optional[int]:
    """
    Picks the EFFECTIVE chunk length of a subset.

    `chunk_length` is a MAXIMUM, not an imposed value: each subset gets its own
    dense file, so nothing forces two subsets to share a length. Measured on
    LOTSA: BEIJING_SUBWAY_30MIN only has 1572-step series and lost its 552
    series against a chunk fixed at 2048.

    Rule: the median of observed lengths, capped at `requested` and floored at
    `min_length`. Returns None if the median is below `min_length` - the
    subset then cannot produce any usable window.
    """
    if not sample_lengths:
        return None
    median = int(np.median(np.asarray(sample_lengths)))
    if median < min_length:
        return None
    return min(requested, median)


class ChunkStats:
    """
    Counts what goes into and comes out of segmentation.

    Without it, a subset that produces nothing is indistinguishable from one
    whose series are too short to be useful: the script said "no chunks
    written" and left you guessing. Yet the two cases call for opposite
    decisions - lower `chunk_length`, or accept the loss.
    """

    __slots__ = ("series", "too_short", "lost_to_chunking", "non_finite", "imputed",
                 "emitted", "min_len", "max_len", "_nan_frac_sum")

    def __init__(self):
        self.series = 0
        self.too_short = 0          # < min_length: unusable anyway
        self.lost_to_chunking = 0   # >= min_length but < chunk_length: RECOVERABLE
        self.non_finite = 0     # too gappy: given up
        self.imputed = 0        # short gaps: filled by interpolation
        self.emitted = 0
        self.min_len = None
        self.max_len = None
        self._nan_frac_sum = 0.0

    def summary(self, chunk_length: int, min_length: int) -> str:
        if self.series == 0:
            return "no series read (empty subset or missing column)"
        lengths = f"lengths {self.min_len}-{self.max_len}"
        parts = [f"{self.series:,} series ({lengths})",
                 f"{self.emitted:,} chunks"]
        if self.too_short:
            parts.append(f"{self.too_short:,} too short (<{min_length})")
        if self.lost_to_chunking:
            parts.append(
                f"WARNING: {self.lost_to_chunking:,} LOST between {min_length} and "
                f"{chunk_length} - lowering --chunk-length would recover them"
            )
        if self.imputed:
            parts.append(f"{self.imputed:,} chunks filled (short gaps)")
        if self.non_finite:
            mean_frac = 100 * self._nan_frac_sum / self.non_finite
            parts.append(
                f"{self.non_finite:,} rejected (gaps: {mean_frac:.0f}% on average)"
            )
            if mean_frac > 15:
                # Measured on LOTSA: HZMETRO/SHMETRO have ~23% NaN in REGULAR
                # blocks of 23 steps - the subway's nightly closure.
                # Interpolating would fabricate ridership at 3 am. Raising the
                # threshold would be a mistake, not a fix.
                parts.append(
                    "-> gaps are probably STRUCTURAL (nightly closure, "
                    "sensor outage): do not raise --max-nan-fraction, "
                    "this subset is unusable without missing-data handling"
                )
        return " | ".join(parts)


def iter_dense_chunks(
    series_iter: Iterable[np.ndarray],
    chunk_length: int,
    min_length: int,
    max_chunks: Optional[int] = None,
    max_nan_fraction: float = 0.05,
    stats: Optional[ChunkStats] = None,
    pad_to: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Turns a stream of arbitrary-length series into a stream of chunks of
    EXACTLY `chunk_length`, ready to be stacked densely.

    Series shorter than `chunk_length` are dropped: mixing lengths would force
    an `object` array, which constraint 1 of the module forbids. That is a real
    cost - a 5,000-step series is usable for a 1,280 window but is lost when
    `chunk_length` is 8,192 - hence `ChunkStats`, which makes it visible
    instead of leaving you guessing.

    `pad_to` (v3 corpus, short series - G7.1/roadmap S2): if given, any
    accepted chunk shorter than `pad_to` is LEFT-PADDED by repeating its first
    value, up to exactly `pad_to` - the emit length becomes
    max(chunk_length, pad_to). Why left: the REAL data occupies the end of the
    chunk, where the dataset reads the TARGET - so the target is always real,
    and the "flat then data" context is exactly what evaluation already imposes
    on short series (prepare_context edge-pads on the left, evaluate_gift.py).
    The model thus trains in the condition it will be evaluated in (m4_yearly:
    19 context steps). The target (256 steps) must be real: call with
    `min_length >= 384` (256 target + >= 128 real context) - the recommended
    setting is `--min-length 384 --pad-to 1280`.
    """
    emit_len = max(chunk_length, pad_to) if pad_to else chunk_length
    emitted = 0
    for series in series_iter:
        arr = np.asarray(series, dtype=np.float32).ravel()
        n = arr.shape[0]
        if stats is not None:
            stats.series += 1
            stats.min_len = n if stats.min_len is None else min(stats.min_len, n)
            stats.max_len = n if stats.max_len is None else max(stats.max_len, n)
            if n < min_length:
                stats.too_short += 1
            elif n < chunk_length:
                stats.lost_to_chunking += 1

        for chunk in segment_series(arr, chunk_length, min_length):
            if pad_to and chunk.shape[0] < emit_len:
                # LEFT edge-padding: real data stays at the end (target side)
                # - see docstring.
                chunk = np.concatenate([
                    np.full(emit_len - chunk.shape[0], chunk[0],
                            dtype=np.float32), chunk])
            if chunk.shape[0] != emit_len:
                continue
            if not _finite(chunk):
                filled = impute_gaps(chunk, max_nan_fraction)
                if filled is None:
                    if stats is not None:
                        stats.non_finite += 1
                        stats._nan_frac_sum += float(
                            1.0 - np.isfinite(chunk).mean()
                        )
                    continue
                chunk = filled
                if stats is not None:
                    stats.imputed += 1
            yield chunk
            emitted += 1
            if stats is not None:
                stats.emitted = emitted
            if max_chunks is not None and emitted >= max_chunks:
                return


def write_dense_npy(
    chunks: Iterable[np.ndarray],
    out_path: Path,
    chunk_length: int,
    max_chunks: int,
) -> int:
    """
    Writes the chunks to a DENSE (N, chunk_length) float32 `.npy`, streaming
    through `open_memmap` - RAM never sees more than one chunk.

    The file is pre-allocated at `max_chunks` then truncated to the number
    actually written (header rewrite via a second `open_memmap`), which avoids
    counting chunks in advance.

    Returns the number of chunks written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp.npy")
    buf = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(max_chunks, chunk_length)
    )

    # A conversion takes hours and gives no intermediate feedback without this:
    # impossible to tell "slow progress" from "stuck".
    report_every = max(1, max_chunks // 20)

    written = 0
    for chunk in chunks:
        if written >= max_chunks:
            break
        buf[written] = chunk
        written += 1
        if written % report_every == 0:
            logger.info(
                f"    {out_path.name}: {written:,}/{max_chunks:,} chunks "
                f"({100 * written / max_chunks:.0f}%)"
            )
    buf.flush()
    del buf

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        logger.warning(f"No chunks written for {out_path.name} - file not created")
        return 0

    # Truncated copy: the final file only contains what was written.
    src = np.load(tmp_path, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(written, chunk_length)
    )
    step = max(1, 1_000_000 // max(1, chunk_length))
    for start in range(0, written, step):
        # Explicit bound: `src` has max_chunks rows, `dst` only `written`.
        # Without the min, the last slice (or the first if step > written)
        # reads more rows than the destination holds.
        end = min(start + step, written)
        dst[start:end] = src[start:end]
    dst.flush()
    del dst, src
    tmp_path.unlink(missing_ok=True)

    logger.info(f"{out_path.name}: {written:,} chunks x {chunk_length} steps")
    return written


def convert_subset(
    series_stream: Iterator[np.ndarray],
    out_path: Path,
    *,
    chunk_length: int,
    min_length: int,
    max_chunks: int,
    max_nan_fraction: float = 0.05,
    sample_size: int = 200,
    pad_to: Optional[int] = None,
) -> Tuple[int, ChunkStats, Optional[int]]:
    """
    Converts ONE subset (stream of 1-D series) into a dense `.npy`.

    Factors out the "sample the first series -> `choose_chunk_length` ->
    chained stream -> `iter_dense_chunks` -> `write_dense_npy`" block for
    converters OTHER than LOTSA (prepare_chronos.py, future corpora).
    `prepare_lotsa.py` keeps its inline version: it is the artifact that
    produced the corpora of reproduced runs (E14/E16); we do not refactor it
    under a published result.

    Returns (chunks_written, stats, effective_chunk_length) -
    `effective_chunk_length` is None if the subset is unusable (median length
    < min_length), in which case nothing is written.
    """
    sample: list = []
    for series in series_stream:
        sample.append(series)
        if len(sample) >= sample_size:
            break

    stats = ChunkStats()
    effective = choose_chunk_length(
        [len(x) for x in sample], chunk_length, min_length
    )
    if effective is None:
        return 0, stats, None

    def _chained(buffered=sample, rest=series_stream):
        yield from buffered
        yield from rest

    emit_len = max(effective, pad_to) if pad_to else effective
    chunks = iter_dense_chunks(
        _chained(),
        chunk_length=effective,
        min_length=min_length,
        max_chunks=max_chunks,
        max_nan_fraction=max_nan_fraction,
        stats=stats,
        pad_to=pad_to,
    )
    written = write_dense_npy(
        chunks, out_path, chunk_length=emit_len, max_chunks=max_chunks
    )
    return written, stats, emit_len
