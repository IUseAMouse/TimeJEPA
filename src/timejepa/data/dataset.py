"""
PyTorch Dataset for time series with sliding windows for JEPA training.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalizer import Normalizer, get_normalizer
from .augmentations import TimeSeriesAugmentations, AugmentationConfig, FinetuneAugmentations

logger = logging.getLogger(__name__)


class SeriesTooShortError(ValueError):
    """
    Raised when no window of `context_length + prediction_length` fits.

    A dedicated type so that a multi-dataset pipeline can skip the offending
    dataset and carry on, while a single-dataset run still fails loudly. It
    subclasses ValueError so existing `except ValueError` handlers keep working.
    """

    def __init__(self, series_length: int, required: int,
                 context_length: int, prediction_length: int,
                 data_path: Optional[Path] = None):
        self.series_length = series_length
        self.required = required
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.data_path = data_path

        # The most useful thing to tell someone here is the largest context that
        # WOULD work, so they can fix the config in one step.
        max_context = max(0, series_length - prediction_length)
        where = f" in {data_path.name}" if data_path is not None else ""
        super().__init__(
            f"Series too short{where}: {series_length} < {required} "
            f"(context={context_length} + pred={prediction_length}). "
            f"Longest usable context at this prediction_length: {max_context}."
        )


def _pack_series(series_list):
    """
    Pack a list of series into either a dense numeric array or a genuinely
    ragged 1-D object array.

    B22. `np.array(list_of_arrays, dtype=object)` does NOT always produce the
    ragged 1-D array one expects: when every kept series has the SAME length it
    silently returns a 2-D OBJECT array of shape (N, L). `np.stack` on that
    preserves dtype=object, so the value survives all the way to
    `torch.from_numpy`, which rejects object arrays:

        TypeError: can't convert np.ndarray of type numpy.object_

    The case never arose before because the length filter always left series of
    mixed lengths. It appears as soon as a dataset whose survivors are uniform
    is used - m4-hourly, whose series are all 1008 steps once filtered.
    """
    if len(series_list) == 0:
        # Preserved on purpose: callers detect the empty case and raise
        # SeriesTooShortError with a helpful message.
        return np.array([], dtype=object)

    lengths = {np.asarray(s).shape[-1] for s in series_list}
    if len(lengths) == 1:
        stacked = np.stack([np.asarray(s) for s in series_list])
        return stacked.astype(np.float32) if stacked.dtype == object else stacked

    out = np.empty(len(series_list), dtype=object)
    out[:] = list(series_list)
    return out


def _to_tensor(window: np.ndarray) -> "torch.Tensor":
    """
    Converts a window to a tensor, guaranteeing it is WRITABLE.

    B23. With `use_mmap=True`, `np.load(mmap_mode="r")` returns a read-only
    array. `np.ascontiguousarray` on an already contiguous slice does NOT
    copy, and `.float()` on float32 is a no-op: the tensor therefore shares
    the mapping's memory (verified: `t.data_ptr()` equals the array address).
    PyTorch warns about it - "writing to this tensor will result in undefined
    behavior" - and the augmentations are applied right after.

    An in-place write on a read-only mapping is at best a segfault, at worst
    silent corruption of the on-disk `.npy`. The copy only happens when
    needed: on the non-memmap path, the array is already writable and nothing
    changes.
    """
    arr = np.ascontiguousarray(window)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.from_numpy(arr).float()


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series with sliding windows.
    
    For JEPA training, each sample returns:
    - context: Past window (to be encoded by main encoder)
    - target: Future window(s) (to be encoded by target encoder)
    - metadata: Series ID, timestamps, etc.
    
    Note: Augmentations are NOT applied here. Use AugmentedSubset wrapper
    for train/val/test splits with controlled augmentation.
    """
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
        normalizer: Optional[Normalizer] = None,
        normalize_mode: str = "per_series",
        return_tensor: bool = True,
        max_series: Optional[int] = None,
        min_series_length: Optional[int] = None,
        augmentations: Optional[Union[TimeSeriesAugmentations, AugmentationConfig, Dict[str, Any]]] = None,
        multi_resolution_factors: Optional[List[int]] = None,
        p_multi_resolution: float = 0.0,
        # G9.2 - cross-resolution JEPA: context decimated at k1, target at
        # k2, (k1, k2) drawn independently from multi_resolution_factors. At
        # False (all existing configs), behavior bit-identical.
        cross_resolution: bool = False,
        use_mmap: bool = False,
        # Corpus v4 (2026-09-05) - short-series windows. Needs the _reallen/
        # sidecar written by prepare_lotsa --pad-to. Off (default) = the item
        # dict and the window set are bit-identical for every existing config.
        short_series_windows: bool = False,
        short_min_context: int = 16,
        short_min_target: int = 4,
    ):
        """
        Args:
            data_path: Path to .npy file with shape (num_series, seq_length)
                       or (num_series, num_channels, seq_length)
            context_length: Length of context window (past)
            prediction_length: Length of prediction window (future)
            stride: Stride for sliding window (1 = maximum overlap)
            normalizer: Pre-fitted normalizer, or None to fit on data
            normalize_mode: 'per_series' or 'global' (if fitting normalizer)
            return_tensor: If True, return torch tensors, else numpy
            max_series: Limit number of series (for debugging)
            min_series_length: Filter out series shorter than this
            augmentations: Augmentation config (used by AugmentedSubset, not here)
            multi_resolution_factors: Decimation factors for TRUE multi-resolution
                sampling, e.g. [1, 2, 3, 4]. See `get_item`.
            p_multi_resolution: Probability of applying it (train split only).
        """
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.return_tensor = return_tensor
        self.normalize_mode = normalize_mode
        # Store the augmentations so AugmentedSubset can access them
        self.augmentations = self._setup_augmentations(augmentations)
        self.multi_resolution_factors = list(multi_resolution_factors or [1])
        self.p_multi_resolution = float(p_multi_resolution)
        self.cross_resolution = bool(cross_resolution)
        self.short_series_windows = bool(short_series_windows)
        self.short_min_context = int(short_min_context)
        self.short_min_target = int(short_min_target)
        self.real_lens = None            # per-row real trailing length (sidecar)
        self._short_rows = None          # bool per row: real_len < ctx + pred

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        # use_mmap: for the LOTSA-scale corpora, files run to several GB and must
        # NOT be read into RAM. A memmapped DENSE array also keeps fork's
        # copy-on-write intact across dataloader workers, which object arrays
        # famously do not (B19: per-element refcounts turned 6.5 GB into ~32 GB
        # across 5 processes). Default False, so every existing config loads
        # exactly as before.
        if use_mmap:
            try:
                data = np.load(self.data_path, mmap_mode="r")
            except ValueError as exc:
                # numpy refuses object arrays with "Array can't be memory-mapped:
                # Python objects in dtype." - accurate but it does not say what
                # to do about it.
                raise ValueError(
                    f"use_mmap=True requires a dense array, but {self.data_path} "
                    f"holds an object (ragged) array. Convert it with "
                    f"scripts/prepare_lotsa.py, which segments series into "
                    f"fixed-length chunks precisely so the result is dense and "
                    f"memmappable. Original error: {exc}"
                ) from exc
            logger.info(f"  memmap enabled: {data.shape} {data.dtype} (not loaded into RAM)")
        else:
            data = np.load(self.data_path, allow_pickle=True)

        # Handle object arrays (variable length series)
        # `_longest_series_seen` is captured BEFORE filtering: once the short
        # series are dropped the array can be empty, and an error message that
        # then reports "longest = 0" would be actively misleading.
        self._longest_series_seen = None
        if data.dtype == object:
            logger.warning("Data contains variable-length series")
            min_len = context_length + prediction_length
            self._longest_series_seen = max((len(s) for s in data), default=0)
            data = _pack_series([s for s in data if len(s) >= min_len])
            logger.info(
                f"Kept {len(data)} series with length >= {min_len} "
                f"(longest available: {self._longest_series_seen})"
            )

        # Corpus v4 sidecar: real trailing length per row, written next to the
        # file under _reallen/ (outside the '*.npy' glob). Dense rows only.
        real_lens = None
        side = self.data_path.parent / "_reallen" / self.data_path.name
        if data.dtype != object and side.exists():
            real_lens = np.load(side).astype(np.int64)
            if real_lens.shape[0] != len(data):
                logger.warning(
                    f"  _reallen sidecar has {real_lens.shape[0]} rows for "
                    f"{len(data)} series - ignored")
                real_lens = None

        # Limit number of series if requested
        if max_series is not None and len(data) > max_series:
            logger.info(f"Limiting to {max_series} series (out of {len(data)})")
            data = data[:max_series]
            if real_lens is not None:
                real_lens = real_lens[:max_series]

        # Filter by minimum length
        if min_series_length is not None:
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = _pack_series([s for s in data if len(s) >= min_series_length])
            elif isinstance(data, np.ndarray):
                mask = np.array([s.shape[-1] >= min_series_length for s in data])
                data = data[mask]
                if real_lens is not None:
                    real_lens = real_lens[mask]
            logger.info(f"After length filter: {len(data)} series")

        # Convert to array if homogeneous
        if isinstance(data, np.ndarray) and data.dtype == object and len(data) > 0:
            lengths = [np.asarray(s).shape[-1] for s in data]
            if len(set(lengths)) == 1:
                # _pack_series already handles this on the filtered paths; kept
                # for inputs that arrive uniform without passing a filter.
                # .astype is essential: np.stack over object elements stays
                # object-typed (B22).
                data = np.stack([np.asarray(s) for s in data])
                if data.dtype == object:
                    data = data.astype(np.float32)
                logger.info(f"Converted to dense array: {data.shape}")

        self.data = data
        self.real_lens = real_lens
        if self.short_series_windows and real_lens is None:
            # A file without a sidecar is a dense full-length block - the
            # normal case for most of a mixed corpus (v4: 106 of 117 files).
            # Only the ABSENCE of the whole _reallen/ directory means the
            # mechanism is missing (sidecar not linked): that one is loud.
            side_dir = self.data_path.parent / "_reallen"
            if side_dir.is_dir():
                logger.debug(f"  no _reallen sidecar for {self.data_path.name}: "
                             "dense block, standard windows")
            else:
                logger.warning(
                    "  short_series_windows=True but no _reallen/ directory in "
                    f"{self.data_path.parent}: short-series windows INACTIVE "
                    "for the whole corpus (sidecar missing or not linked)")
        self.is_multivariate = data.ndim == 3
        
        logger.info(f"Data shape: {data.shape if data.dtype != object else f'({len(data)}, variable)'}")
        logger.info(f"Is multivariate: {self.is_multivariate}")
        
        # Fit or use provided normalizer
        if normalizer is None:
            logger.info("No normalizer provided, using IdentityNormalizer")
            self.normalizer = get_normalizer("identity")
        else:
            logger.info(f"Using provided normalizer: {normalizer.__class__.__name__}")
            self.normalizer = normalizer
        
        if not self.normalizer.is_fitted:
            self.normalizer.fit(self.data)
        
        self.normalized_data = self.normalizer.transform(self.data)
        self.window_indices = self._generate_window_indices()

        # Log augmentation config (used by AugmentedSubset)
        if self.augmentations is not None:
            logger.info(f"Augmentations configured (applied via AugmentedSubset)")
            self._log_augmentation_config()

        self._log_multi_resolution_coverage()

        logger.info(f"Created dataset with {len(self)} windows")

    def _setup_augmentations(
        self, 
        augmentations: Optional[Union[TimeSeriesAugmentations, AugmentationConfig, Dict[str, Any]]]
    ) -> Optional[TimeSeriesAugmentations]:
        """Setup augmentations from various input formats."""
        if augmentations is None:
            return None
        
        if isinstance(augmentations, TimeSeriesAugmentations):
            return augmentations
        
        if isinstance(augmentations, AugmentationConfig):
            return TimeSeriesAugmentations(augmentations)
        
        if isinstance(augmentations, dict):
            return TimeSeriesAugmentations.from_dict(augmentations)
        
        raise ValueError(f"Unknown augmentation type: {type(augmentations)}")

    def _log_augmentation_config(self):
        """Log which augmentations are enabled."""
        if self.augmentations is None:
            return
        
        cfg = self.augmentations.config
        enabled = []
        if cfg.scale_enabled:
            enabled.append(f"scale({cfg.scale_range}, p={cfg.p_scale})")
        if cfg.jitter_enabled:
            enabled.append(f"jitter(std={cfg.jitter_std}, p={cfg.p_jitter})")
        if cfg.magnitude_warp_enabled:
            enabled.append(f"mag_warp(sigma={cfg.magnitude_warp_sigma}, p={cfg.p_magnitude_warp})")
        if cfg.drs_enabled:
            enabled.append(f"DRS({cfg.drs_factors}, p={cfg.p_drs})")
        if cfg.trend_enabled:
            enabled.append(f"trend(mag={cfg.trend_magnitude}, p={cfg.p_trend})")
        
        logger.info(f"  Active augmentations: {', '.join(enabled) if enabled else 'none'}")

    def _log_multi_resolution_coverage(self):
        """
        Report what fraction of windows can actually be decimated at each factor.

        A factor `f` needs `start + (ctx+pred)*f` timesteps of headroom, so short
        series silently fall back to f=1. Without this line there is no way to
        tell from a training log whether multi-resolution did anything at all -
        which is exactly the question that came up on the first real run.
        """
        if self.p_multi_resolution <= 0 or len(self.multi_resolution_factors) <= 1:
            return

        need = self.context_length + self.prediction_length
        factors = [f for f in self.multi_resolution_factors if f > 1]
        total = len(self.window_indices)
        if total == 0 or not factors:
            return

        if self.cross_resolution:
            # G9.2 - the requirement is no longer (ctx+pred)*f but
            # ctx*k1 + pred*k2. The cheapest pair for a factor f only
            # decimates the target (k1=1, k2=f): need = ctx + pred*f. Without
            # this branch, the log would say "inactive" on 2048 chunks even
            # though k1=1 pairs live there fine - or worse, the reverse: a
            # silently sterile arm.
            series_ids = self.window_indices[:, 0]
            starts = self.window_indices[:, 1].astype(np.int64)
            if self.data.dtype == object:
                lengths = np.array([x.shape[-1] for x in self.normalized_data],
                                   dtype=np.int64)[series_ids]
            else:
                lengths = np.full(total, self.normalized_data.shape[-1],
                                  dtype=np.int64)
            elig_tgt = {f: int((starts + self.context_length
                                + self.prediction_length * f <= lengths).sum())
                        for f in factors}
            elig_ctx = {f: int((starts + self.context_length * f
                                + self.prediction_length <= lengths).sum())
                        for f in factors}
            parts = ", ".join(
                f"k2={f}:{n / total:.0%}" for f, n in sorted(elig_tgt.items()))
            parts_c = ", ".join(
                f"k1={f}:{n / total:.0%}" for f, n in sorted(elig_ctx.items()))
            coverage = max(elig_tgt.values()) / total
            if coverage == 0:
                logger.warning(
                    "  Cross-resolution INACTIVE here: no (k1,k2) pair fits "
                    "the series; the arm would be STERILE on this dataset "
                    "(chunks must be >= ctx + pred*k2)."
                )
            else:
                logger.info(
                    f"  Cross-resolution p={self.p_multi_resolution} | "
                    f"decimable target: {parts} | decimable context: {parts_c} "
                    f"| effective rate ~ {self.p_multi_resolution * coverage:.0%}"
                )
            return

        # Vectorized: window_indices holds tens of millions of rows on the full
        # corpus, so a Python loop here would take minutes at startup.
        series_ids = self.window_indices[:, 0]
        starts = self.window_indices[:, 1].astype(np.int64)
        if self.data.dtype == object:
            lengths = np.array([s.shape[-1] for s in self.normalized_data],
                               dtype=np.int64)[series_ids]
        else:
            lengths = np.full(total, self.normalized_data.shape[-1], dtype=np.int64)

        eligible = {f: int((starts + need * f <= lengths).sum()) for f in factors}

        parts = ", ".join(f"f={f}:{n / total:.0%}" for f, n in sorted(eligible.items()))
        coverage = max(eligible.values()) / total

        if coverage == 0:
            logger.warning(
                f"  Multi-resolution INACTIVE here: series are too short "
                f"(need {need * min(eligible)} timesteps for the smallest factor). "
                f"Every window falls back to f=1."
            )
        else:
            logger.info(
                f"  Multi-resolution p={self.p_multi_resolution} | "
                f"window eligibility: {parts} | effective rate "
                f"~ {self.p_multi_resolution * coverage:.0%}"
            )

    def _generate_window_indices(self) -> np.ndarray:
        """
        Generate (series_idx, start_idx) pairs for all valid windows,
        as an int32 array of shape [N, 2].

        Memory
        ------
        This used to be a Python list of tuples, which is what made training run
        out of RAM. The full 24-dataset corpus yields ~54M windows, and a list of
        2-tuples costs roughly:

            8 B   list pointer
          + 56 B  tuple object
          + 56 B  two int objects (start_idx is far past the small-int cache)
          = ~120 B per window  ->  ~6.5 GB

        Worse, that cost is paid PER PROCESS. Every dataloader worker walks the
        list, and touching a tuple bumps its refcount, which writes to its page
        and defeats fork's copy-on-write - so the parent's 6.5 GB got duplicated
        into each of the 4 workers as the epoch progressed. That is the steady
        climb to ~50 GB observed on a 57 GB host.

        An int32 [N, 2] array is 8 B per window (~430 MB for 54M) and has no
        per-element Python objects, so workers genuinely share the pages.
        Roughly 15x smaller, and flat over time instead of growing.

        int32 is checked, not assumed: the largest series count in the corpus is
        ~145k (wikipedia) and the longest series ~7.4M (solar-4-seconds), both
        comfortably inside int32.

        Both storage layouts behave the same way on short series. Previously the
        variable-length path silently dropped series that were too short, while
        the fixed-length path raised - so a uniformly-short fixed-shape dataset
        (e.g. wikipedia-web-traffic-weekly, 114 weekly points against a required
        640) killed an entire multi-dataset run, whereas the same data stored as
        an object array would just have been skipped.
        """
        min_length = self.context_length + self.prediction_length

        if self.data.dtype == object:
            # Build per series with numpy, then concatenate once: no Python-level
            # per-window objects are ever materialised.
            blocks = []
            for series_idx, series in enumerate(self.normalized_data):
                seq_length = series.shape[-1]
                if seq_length < min_length:
                    continue
                starts = np.arange(0, seq_length - min_length + 1, self.stride,
                                   dtype=np.int32)
                if starts.size == 0:
                    continue
                block = np.empty((starts.size, 2), dtype=np.int32)
                block[:, 0] = series_idx
                block[:, 1] = starts
                blocks.append(block)

            if not blocks:
                # Use the pre-filter maximum: by this point every series short
                # enough to be dropped already has been, so measuring what
                # remains would report 0 and hide the real length.
                longest = self._longest_series_seen
                if longest is None:
                    longest = max((s.shape[-1] for s in self.normalized_data), default=0)
                raise SeriesTooShortError(
                    longest, min_length,
                    self.context_length, self.prediction_length, self.data_path
                )
            return np.concatenate(blocks, axis=0)
        else:
            seq_length = self.normalized_data.shape[-1]
            if seq_length < min_length:
                raise SeriesTooShortError(
                    seq_length, min_length,
                    self.context_length, self.prediction_length, self.data_path
                )
            n_series = len(self.normalized_data)
            starts = np.arange(0, seq_length - min_length + 1, self.stride,
                               dtype=np.int32)
            if self.real_lens is None:
                # Fixed-length: every series has identical starts, so build the
                # two columns with a single outer product instead of a nested
                # loop.
                indices = np.empty((n_series * starts.size, 2), dtype=np.int32)
                indices[:, 0] = np.repeat(np.arange(n_series, dtype=np.int32),
                                          starts.size)
                indices[:, 1] = np.tile(starts, n_series)
                return indices
            return self._sidecar_window_indices(seq_length, min_length, starts)

    def _sidecar_window_indices(self, seq_length: int, min_length: int,
                                starts: np.ndarray) -> np.ndarray:
        """Corpus v4: rows are left-padded, real data at the end.

        Rows with real_len >= ctx + pred keep the standard sliding windows:
        every target then starts at >= ctx > pad length, so targets are real
        and only the context carries the flat prefix (the eval condition).
        Shorter rows would otherwise yield windows whose target lies in the
        pad (v3 defect). With short_series_windows they get BOUNDARY windows
        instead: the context/target split falls inside the real data, the
        context is left-padded by the row itself, the target holds the real
        tail and is right-padded + masked by get_item.
        """
        rl = self.real_lens
        full = rl >= min_length
        self._short_rows = ~full
        blocks = []
        full_ids = np.flatnonzero(full).astype(np.int32)
        if full_ids.size and starts.size:
            block = np.empty((full_ids.size * starts.size, 2), dtype=np.int32)
            block[:, 0] = np.repeat(full_ids, starts.size)
            block[:, 1] = np.tile(starts, full_ids.size)
            blocks.append(block)
        if self.short_series_windows:
            for row in np.flatnonzero(~full):
                pad_len = seq_length - int(rl[row])
                b_lo = pad_len + self.short_min_context
                b_hi = seq_length - self.short_min_target
                if b_hi < b_lo:
                    continue
                bounds = np.arange(b_lo, b_hi + 1, self.stride, dtype=np.int64)
                st = bounds - self.context_length
                st = st[st >= 0]
                if st.size == 0:
                    continue
                block = np.empty((st.size, 2), dtype=np.int32)
                block[:, 0] = row
                block[:, 1] = st
                blocks.append(block)
        if not blocks:
            raise SeriesTooShortError(
                int(rl.max()) if rl.size else 0, min_length,
                self.context_length, self.prediction_length, self.data_path)
        return np.concatenate(blocks, axis=0)

    def _is_short_row(self, series_idx: int) -> bool:
        return (self._short_rows is not None
                and bool(self._short_rows[int(series_idx)]))
    
    def __len__(self) -> int:
        return len(self.window_indices)
    
    def _sample_resolution_factor(self, series_len: int, start_idx: int) -> int:
        """
        Pick a decimation factor that still fits in the remaining series.

        Only factors with `start_idx + (ctx + pred) * f <= series_len` are
        eligible, so the window is always fully materialised rather than padded.
        """
        if self.p_multi_resolution <= 0.0 or len(self.multi_resolution_factors) <= 1:
            return 1
        if np.random.rand() >= self.p_multi_resolution:
            return 1

        need = self.context_length + self.prediction_length
        eligible = [
            f for f in self.multi_resolution_factors
            if f >= 1 and start_idx + need * f <= series_len
        ]
        if not eligible:
            return 1
        return int(np.random.choice(eligible))

    def _sample_resolution_pair(self, series_len: int, start_idx: int):
        """
        G9.2 - draws independent (k1, k2) for the cross-resolution arm.

        The context is read decimated at k1 and the target at k2: the raw
        window needed is `ctx*k1 + pred*k2`, NOT `(ctx+pred)*f` - the
        eligibility constraint accounts for this pair by pair. Fallback
        (1, 1) when no non-trivial pair fits (the case of ALL 2048 chunks on
        the k1 side: 1024*2 + 256 = 2304 > 2048; only k1=1 < k2 pairs live
        there, and the full space requires the 8192 chunks - synthetic,
        chronos_extras).
        """
        if self.p_multi_resolution <= 0.0 or len(self.multi_resolution_factors) <= 1:
            return 1, 1
        if np.random.rand() >= self.p_multi_resolution:
            return 1, 1
        pairs = [
            (k1, k2)
            for k1 in self.multi_resolution_factors
            for k2 in self.multi_resolution_factors
            if (k1, k2) != (1, 1)
            and start_idx + self.context_length * k1 + self.prediction_length * k2
            <= series_len
        ]
        if not pairs:
            return 1, 1
        return pairs[int(np.random.randint(len(pairs)))]

    def get_item(self, idx: int, allow_multi_resolution: bool = False) -> Dict[str, Any]:
        """
        Get a single sample (WITHOUT the augmentation pipeline).

        True multi-resolution sampling
        ------------------------------
        With `allow_multi_resolution=True` the window is read as
        `series[start : start + (ctx+pred)*f : f]`, i.e. a LONGER raw stretch
        decimated by `f`. That genuinely changes the sampling frequency, so a
        seasonal cycle of period `m` becomes `m/f` steps and the number of
        cycles inside the context changes by `f`.

        This is distinct from `augmentations.diverse_resolution_sampling`, which
        downsamples and then interpolates back to the SAME length: that
        simulates a coarser sensor (a smoothing operation) and leaves the
        period-to-patch ratio untouched.

        The distinction matters: `scripts/diagnose_ettm.py` shows the model's
        skill is governed by the seasonal period measured in patch positions
        (ECL interpolated x4 goes from +28.5% to -136% skill), and by how far
        the context length is from the one seen in training. Only the decimation
        form varies those quantities during pretraining.
        """
        series_idx, start_idx = self.window_indices[idx]
        series = self.normalized_data[series_idx]
        series_len = series.shape[-1]
        short_row = self._is_short_row(series_idx)
        if short_row:
            # Boundary window: never decimate (the pair check would read the
            # padded length and decimate the flat prefix).
            allow_multi_resolution = False

        if allow_multi_resolution and self.cross_resolution:
            # G9.2 - context at k1, target at k2. The target stays the
            # PHYSICALLY contiguous future: its first raw point is exactly
            # series[start + ctx*k1], whatever k2 is.
            k1, k2 = self._sample_resolution_pair(series_len, start_idx)
            ctx_end = start_idx + self.context_length * k1
            tgt_end = ctx_end + self.prediction_length * k2
            if self.is_multivariate:
                context = series[:, start_idx:ctx_end:k1]
                target = series[:, ctx_end:tgt_end:k2]
            else:
                context = series[start_idx:ctx_end:k1]
                target = series[ctx_end:tgt_end:k2]
            factor = k1
            w = float(k2) / float(k1)
        else:
            factor = self._sample_resolution_factor(series_len, start_idx) \
                if allow_multi_resolution else 1
            w = 1.0

            span = (self.context_length + self.prediction_length) * factor
            window_end = start_idx + span

            if self.is_multivariate:
                window = series[:, start_idx:window_end:factor]
                context = window[:, :self.context_length]
                target = window[:, self.context_length:self.context_length + self.prediction_length]
            else:
                window = series[start_idx:window_end:factor]
                context = window[:self.context_length]
                target = window[self.context_length:self.context_length + self.prediction_length]

        target_mask = None
        if self.short_series_windows:
            target_mask = np.ones(self.prediction_length, dtype=bool)
            n_tgt = target.shape[-1]
            if n_tgt < self.prediction_length:
                # Boundary window past the row end: right-pad the target with
                # its last real value and mask those positions out of the loss.
                # Never left of the real end - no observation is fabricated
                # inside the scored region.
                pad = self.prediction_length - n_tgt
                edge = np.take(target, [-1], axis=-1)
                target = np.concatenate(
                    [target, np.repeat(edge, pad, axis=-1)], axis=-1)
                target_mask[n_tgt:] = False

        if self.return_tensor:
            context = _to_tensor(context)
            target = _to_tensor(target)

        item = {
            'context': context,
            'target': target,
            'series_id': series_idx,
            'start_idx': start_idx,
            'resolution_factor': factor,
        }
        if self.cross_resolution:
            # w = k2/k1 per item (G9.2). Emitted ONLY in cross-resolution
            # mode: the default collate requires identical keys on all items,
            # and since the flag is a dataset attribute it is all or nothing
            # - existing configs never see this key (item dict unchanged,
            # pinned by test).
            item['w'] = w
        if target_mask is not None:
            # Same all-or-nothing rule: emitted for every item when the flag
            # is on (all-True on full windows), absent otherwise.
            item['target_mask'] = (torch.from_numpy(target_mask)
                                   if self.return_tensor else target_mask)
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample (WITHOUT augmentation or multi-resolution).

        Use AugmentedSubset for controlled augmentation in train/val/test.
        """
        return self.get_item(idx, allow_multi_resolution=False)
    
    def get_normalizer(self) -> Normalizer:
        return self.normalizer
    
    def get_raw_series(self, series_id: int) -> np.ndarray:
        if self.data.dtype == object:
            return self.data[series_id]
        return self.data[series_id]


class AugmentedSubset(Dataset):
    """
    Thread-safe subset with explicit augmentation control.
    
    Each instance has its own apply_augmentation flag,
    making it safe for multi-worker DataLoaders.
    """
    
    def __init__(
        self, 
        dataset: TimeSeriesDataset, 
        indices: List[int], 
        apply_augmentation: bool = True
    ):
        self.dataset = dataset
        self.indices = indices
        self.apply_augmentation = apply_augmentation
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Direct data access with controlled augmentation.
        
        Returns:

            Dictionary with keys:

            - 'context': Context window (past), shape (context_length,) or (C, context_length)

            - 'target': Target window (future), shape (prediction_length,) or (C, prediction_length)

            - 'series_id': Index of the time series

            - 'start_idx': Start index of the context window
        """
        real_idx = self.indices[idx]

        # Multi-resolution sampling is a training-time augmentation, so it is
        # gated on the same flag as the rest: val/test always see the native
        # sampling rate.
        item = self.dataset.get_item(
            real_idx, allow_multi_resolution=self.apply_augmentation
        )

        # Augmentation controlled by THIS instance
        if self.apply_augmentation and self.dataset.augmentations is not None:
            item['context'], item['target'] = self.dataset.augmentations(
                item['context'], item['target']
            )

        return item
    
    def __len__(self) -> int:
        return len(self.indices)


class MultiHorizonDataset(TimeSeriesDataset):
    """Extended dataset that returns multiple future horizons."""
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_lengths: List[int],
        stride: int = 1,
        normalizer: Optional[Normalizer] = None,
        normalize_mode: str = "per_series",
        return_tensor: bool = True,
        max_series: Optional[int] = None
    ):
        self.prediction_lengths = sorted(prediction_lengths)
        max_pred_len = max(prediction_lengths)
        
        super().__init__(
            data_path=data_path,
            context_length=context_length,
            prediction_length=max_pred_len,
            stride=stride,
            normalizer=normalizer,
            normalize_mode=normalize_mode,
            return_tensor=return_tensor,
            max_series=max_series
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        series_idx, start_idx = self.window_indices[idx]
        
        if self.data.dtype == object:
            series = self.normalized_data[series_idx]
        else:
            series = self.normalized_data[series_idx]
        
        context_end = start_idx + self.context_length
        
        if self.is_multivariate:
            context = series[:, start_idx:context_end]
        else:
            context = series[start_idx:context_end]
        
        targets = {}
        for pred_len in self.prediction_lengths:
            target_end = context_end + pred_len
            if self.is_multivariate:
                target = series[:, context_end:target_end]
            else:
                target = series[context_end:target_end]
            if self.return_tensor:
                target = torch.from_numpy(target).float()
            targets[pred_len] = target
        
        if self.return_tensor:
            context = torch.from_numpy(context).float()
        
        return {
            'context': context,
            'targets': targets,
            'series_id': series_idx,
            'start_idx': start_idx
        }