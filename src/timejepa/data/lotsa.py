"""
LOTSA (Moirai's pretrain corpus, HuggingFace `Salesforce/lotsa_data`) to the
TimeJEPA .npy format.

Three design constraints:

1. No `object` arrays: their per-element refcounts break fork copy-on-write
   and multiply RAM by the number of dataloader workers. The conversion emits
   only dense float32 arrays, segmenting long series into fixed-length chunks.

2. Memmap reads: files are GBs, and `TimeSeriesDataset(use_mmap=True)` reads
   them without loading them into RAM.

3. Exclusion of evaluation datasets (ETT, electricity, traffic, weather, the
   GIFT-Eval sources). `EVAL_OVERLAP_PATTERNS` filters by substring and the
   conversion prints what it excludes; review that output before a pretrain.

A window cannot straddle two chunks: with `chunk_length` 8192 and a 1280
window, at most about 15% of positions are lost at chunk boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Substrings of LOTSA subset names that overlap an evaluation benchmark. A
# subset forgotten here is a benchmark seen during pretrain, and a zero-shot
# evaluation that is not one. Two lists with different provenances; their
# union is `EVAL_OVERLAP_PATTERNS`.
#
# The GIFT-Eval patterns cover the 28 directories of the official dataset
# (https://huggingface.co/datasets/Salesforce/GiftEval); a test pins this.
# GIFT-Eval allows training on its own train split, so excluding the whole
# dataset is more conservative than required: our splits cannot be aligned
# with theirs.

# Nixtla long-horizon benchmarks (scripts/evaluate.py).
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
    "air_quality",      # kdd_cup_2018 is Beijing air quality 2017-2018
    "loop_seattle", "seattle",
    "restaurant",
    "saugeen",          # also in the local Monash corpus
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

# Subsets readmitted despite matching a pattern. Patterns are coarse
# substrings: "taxi" catches sz_taxi (GIFT-Eval) and taxi_30min (New York,
# unrelated), "m1" catches m1_monthly while the eval only covers M4.
#
# Safety reference: `Salesforce/GiftEvalPretrain`, the pretrain corpus
# sanctioned by the benchmark; everything in it is declared non-leaking for
# GIFT-Eval by its authors. Every entry below is also checked against the
# Nixtla suite and the local Monash suite. Do not add an entry without the
# three checks.
EVAL_SAFE_OVERRIDES: Tuple[str, ...] = (
    # M1/M3 competitions: the eval only covers M4. Low-frequency series,
    # which the corpus otherwise lacks.
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "monash_m3_monthly", "monash_m3_other",
    "monash_m3_quarterly", "monash_m3_yearly",

    # tourism: no eval suite contains it
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",

    # nn5: in none of the three eval suites
    "nn5_daily_with_missing", "nn5_weekly",

    # New York taxis, 30 min: the GIFT eval is SZ_TAXI (Shenzhen)
    "taxi_30min",

    # KDD Cup 2022, wind power at 10 minutes: the GIFT eval is kdd_cup_2018
    # (Beijing air quality), a different competition, domain and year.
    "kdd2022",

    # COVID: the GIFT eval is covid_deaths
    "covid19_energy",     # electricity consumption during the pandemic
    "covid_mobility",     # mobility indices

    # Baidu traffic (China): the Nixtla eval is PEMS (California)
    "Q-TRAFFIC",

    # Australian electricity demand (30 min): the Nixtla and Monash
    # "electricity" evals are UCI/Portugal. GiftEvalPretrain sanctions it.
    "australian_electricity_demand",

    # solar_power: in GiftEvalPretrain, no Nixtla solar benchmark. The local
    # Monash suite evaluates solar-10-minute, but that suite is deprecated in
    # favor of GIFT-Eval, so its solar line is no longer zero-shot.
    "solar_power",

    # Chinese air quality: the GIFT eval kdd_cup_2018 is also Beijing air
    # quality and a time-window overlap is conceivable. GiftEvalPretrain
    # includes both. Remove at the slightest doubt: the gain (hourly data) is
    # small.
    "beijing_air_quality", "china_air_quality",
)

# What stays excluded among subsets that GIFT nonetheless sanctions:
#   traffic_hourly, traffic_weekly   : traffic_hourly is the Nixtla `traffic`
#       and the Monash `traffic-hourly`.
#   weather                          : is the Nixtla `weather`.
#   oikolab_weather                  : weather reanalysis, adjacent to
#       `weather`; the corpus already has era5.
#   cdc_fluview_ilinet               : the source of the Nixtla `illness`.
#   extended_web_traffic_with_missing, kaggle_web_traffic_weekly,
#   wiki-rolling_nips                : the local Monash suite evaluates
#       wikipedia-web-traffic-extended.


def is_eval_overlap(name: str, patterns: Sequence[str] = EVAL_OVERLAP_PATTERNS,
                    overrides: Sequence[str] = EVAL_SAFE_OVERRIDES) -> bool:
    """
    True if the LOTSA subset name overlaps an evaluation benchmark.

    A name listed in `overrides` is readmitted even when it matches a pattern
    (see EVAL_SAFE_OVERRIDES). The override comparison is exact and
    case-insensitive, never a substring, or it would reopen the hole the
    pattern closes.
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

    Without this grouping the per-subset cap protects nothing: cmip6 (33
    slices) and era5 (30) are 63 subsets out of 123, half the corpus in
    climate reanalysis.
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
    Cut a series into non-overlapping fixed-length chunks (overlap is the
    dataset's job, through sliding windows).

    The final remainder is dropped. A series shorter than `chunk_length` but
    at least `min_length` is returned whole as a one-element list.
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
    """LOTSA contains NaNs; one NaN poisons RevIN and the loss."""
    return bool(np.isfinite(chunk).all())


def impute_gaps(chunk: np.ndarray, max_nan_fraction: float) -> Optional[np.ndarray]:
    """
    Fill gaps by linear interpolation, or give up if the chunk is too gappy.

    Rejecting a chunk for a single NaN loses whole subsets (several carry
    "_with_missing" in their name). Interpolation over long gaps fabricates
    signal, hence the threshold.

    Returns the filled chunk, or None if it is too gappy or all-NaN.
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
    # start and end of the chunk.
    filled[~finite] = np.interp(idx[~finite], idx[finite], chunk[finite])
    return filled


def choose_chunk_length(
    sample_lengths: Sequence[int],
    requested: int,
    min_length: int,
) -> Optional[int]:
    """
    Pick the effective chunk length of a subset: the median of observed
    lengths, capped at `requested`.

    `requested` is a maximum, since each subset gets its own dense file; a
    fixed chunk length would lose every subset whose series are shorter.
    Returns None if the median is below `min_length`.
    """
    if not sample_lengths:
        return None
    median = int(np.median(np.asarray(sample_lengths)))
    if median < min_length:
        return None
    return min(requested, median)


class ChunkStats:
    """
    Counts what goes into and comes out of segmentation, so that a subset
    lost to a too-large `chunk_length` can be told from one whose series
    are too short to use.
    """

    __slots__ = ("series", "too_short", "lost_to_chunking", "non_finite", "imputed",
                 "emitted", "min_len", "max_len", "_nan_frac_sum")

    def __init__(self):
        self.series = 0
        self.too_short = 0          # < min_length: unusable
        self.lost_to_chunking = 0   # >= min_length but < chunk_length: recoverable
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
                # Subway subsets have about 23% NaN in regular blocks (the
                # nightly closure); interpolating would fabricate ridership.
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
    real_lens: Optional[list] = None,
) -> Iterator[np.ndarray]:
    """
    Turn a stream of arbitrary-length series into a stream of chunks of
    exactly `chunk_length` (or `pad_to`, if larger), ready to be stacked.

    Series shorter than `chunk_length` are dropped unless `pad_to` is given:
    mixing lengths would force an `object` array. `ChunkStats` makes that
    loss visible.

    `pad_to`: an accepted chunk shorter than `pad_to` is left-padded by
    repeating its first value. Real data then occupies the end of the chunk,
    where the dataset reads the target, and the "flat then data" context is
    what evaluation imposes on short series (edge-padding on the left).

    `real_lens`: if a list is given, the number of real trailing steps of
    every emitted chunk is appended to it. The caller stores it as a sidecar
    so the dataset can tell padding from data.
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
            elif n < chunk_length and not pad_to:
                # with pad_to the short series is kept whole
                stats.lost_to_chunking += 1

        for chunk in segment_series(arr, chunk_length, min_length):
            n_real = int(chunk.shape[0])
            if pad_to and chunk.shape[0] < emit_len:
                # Left edge-padding: real data stays at the end (target side)
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
            if real_lens is not None:
                real_lens.append(n_real)
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
    Write the chunks to a dense (N, chunk_length) float32 `.npy`, streaming
    through `open_memmap`.

    The file is pre-allocated at `max_chunks` then copied to its final size,
    which avoids counting chunks in advance. Returns the number of chunks
    written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp.npy")
    buf = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(max_chunks, chunk_length)
    )

    # A conversion takes hours; report progress every 5%.
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
        # `src` has max_chunks rows, `dst` only `written`.
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
    Convert one subset (stream of 1-D series) into a dense `.npy`: sample the
    first series, `choose_chunk_length`, `iter_dense_chunks`,
    `write_dense_npy`. Used by converters other than prepare_lotsa.py, which
    keeps its own inline version.

    Returns (chunks_written, stats, effective_chunk_length);
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
