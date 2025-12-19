# src/timejepa/data/nixtla.py
"""
Nixtla Long-Horizon benchmark datasets adapter.

Downloads datasets via datasetsforecast and converts to .npy format
compatible with the existing TimeSeriesDataset/MonashDataModule.

These datasets come pre-normalized (z-score with train mean/std),
so use IdentityNormalizer and let RevIN handle instance normalization.

Reference: https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NixtlaDatasetInfo:
    """Metadata for a Nixtla long-horizon dataset."""
    group: str          # Name used by LongHorizon.load()
    freq: str           # Frequency string
    n_series: int       # Number of time series
    horizons: Tuple[int, ...]  # Standard benchmark horizons
    test_size: int      # Standard test set size (timestamps)
    val_size: int       # Standard validation set size


# Registry with official benchmark metadata
# Reference: https://github.com/Nixtla/datasetsforecast
NIXTLA_REGISTRY: Dict[str, NixtlaDatasetInfo] = {
    'ettm1': NixtlaDatasetInfo(
        group='ETTm1', freq='15min', n_series=7,
        horizons=(96, 192, 336, 720), test_size=11520, val_size=11520
    ),
    'ettm2': NixtlaDatasetInfo(
        group='ETTm2', freq='15min', n_series=7,
        horizons=(96, 192, 336, 720), test_size=11520, val_size=11520
    ),
    'etth1': NixtlaDatasetInfo(
        group='ETTh1', freq='1h', n_series=7,
        horizons=(96, 192, 336, 720), test_size=2880, val_size=2880
    ),
    'etth2': NixtlaDatasetInfo(
        group='ETTh2', freq='1h', n_series=7,
        horizons=(96, 192, 336, 720), test_size=2880, val_size=2880
    ),
    'electricity': NixtlaDatasetInfo(
        group='ECL', freq='1h', n_series=321,
        horizons=(96, 192, 336, 720), test_size=5260, val_size=2632
    ),
    'exchange': NixtlaDatasetInfo(
        group='Exchange', freq='1d', n_series=8,
        horizons=(96, 192, 336, 720), test_size=1517, val_size=760
    ),
    'traffic': NixtlaDatasetInfo(
        group='TrafficL', freq='1h', n_series=862,
        horizons=(96, 192, 336, 720), test_size=3508, val_size=1756
    ),
    'ili': NixtlaDatasetInfo(
        group='ILI', freq='1w', n_series=7,
        horizons=(24, 36, 48, 60), test_size=193, val_size=97
    ),
    'weather': NixtlaDatasetInfo(
        group='Weather', freq='10min', n_series=21,
        horizons=(96, 192, 336, 720), test_size=10539, val_size=5270
    ),
}


def get_available_datasets() -> List[str]:
    """Return list of available Nixtla dataset names."""
    return list(NIXTLA_REGISTRY.keys())


def get_dataset_info(name: str) -> NixtlaDatasetInfo:
    """Get metadata for a dataset."""
    key = name.lower()
    if key not in NIXTLA_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {get_available_datasets()}"
        )
    return NIXTLA_REGISTRY[key]


def get_benchmark_horizons(name: str) -> Tuple[int, ...]:
    """Get standard benchmark horizons for a dataset."""
    return get_dataset_info(name).horizons


def download_and_convert(
    dataset_name: str,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    split: Literal['train', 'val', 'test', 'all'] = 'test',
    force_download: bool = False,
) -> Path:
    """
    Download a Nixtla dataset and convert to .npy format.
    
    The output format matches what TimeSeriesDataset expects:
    - Shape: (n_series, seq_length) for univariate treatment per series
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ettm2', 'weather')
        output_dir: Where to save the .npy file
        cache_dir: Where datasetsforecast caches downloads
        split: Which split to save ('train', 'val', 'test', or 'all')
        force_download: Re-download even if file exists
    
    Returns:
        Path to the saved .npy file
    """
    try:
        from datasetsforecast.long_horizon import LongHorizon
    except ImportError:
        raise ImportError(
            "Nixtla support requires datasetsforecast. "
            "Install with: pip install datasetsforecast"
        )
    
    info = get_dataset_info(dataset_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"nixtla_{dataset_name}_{split}.npy"
    
    # Use cache if available
    if output_path.exists() and not force_download:
        logger.info(f"Using cached: {output_path}")
        return output_path
    
    # Setup cache directory
    cache_dir = cache_dir or (output_dir / '.nixtla_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from Nixtla
    logger.info(f"Downloading {info.group} from Nixtla...")
    Y_df, _, _ = LongHorizon.load(directory=str(cache_dir), group=info.group)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    
    # Pivot to wide format: (T, n_series)
    df_wide = Y_df.pivot(index='ds', columns='unique_id', values='y')
    df_wide = df_wide.sort_index()
    
    # Handle NaNs (forward-fill then backward-fill)
    if df_wide.isna().any().any():
        nan_count = df_wide.isna().sum().sum()
        logger.warning(f"Found {nan_count} NaN values, applying forward-fill")
        df_wide = df_wide.ffill().bfill()
    
    data = df_wide.values.astype(np.float32)  # (T, n_series)
    T = len(data)
    
    logger.info(f"Raw data shape: {data.shape} (T={T}, n_series={info.n_series})")
    
    # Split according to standard benchmark sizes
    test_size = info.test_size
    val_size = info.val_size
    train_size = T - test_size - val_size
    
    if train_size < 0:
        logger.warning(
            f"Dataset smaller than expected. "
            f"T={T}, expected train+val+test >= {test_size + val_size}"
        )
        # Fallback: use percentages
        train_size = int(T * 0.7)
        val_size = int(T * 0.1)
        test_size = T - train_size - val_size
    
    logger.info(f"Splits: train={train_size}, val={val_size}, test={test_size}")
    
    if split == 'train':
        data = data[:train_size]
    elif split == 'val':
        data = data[train_size:train_size + val_size]
    elif split == 'test':
        data = data[train_size + val_size:]
    # 'all' keeps everything
    
    # Transpose to (n_series, T) to match TimeSeriesDataset convention
    data = data.T  # Now (n_series, seq_length)
    
    logger.info(f"Saving {split} split: {data.shape} to {output_path}")
    np.save(output_path, data)
    
    return output_path


def prepare_all_splits(
    dataset_name: str,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Dict[str, Path]:
    """
    Prepare train/val/test .npy files for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Where to save files
        cache_dir: Where to cache downloads
        force_download: Force re-download
    
    Returns:
        Dict mapping split name to file path
    """
    paths = {}
    for split in ['train', 'val', 'test']:
        paths[split] = download_and_convert(
            dataset_name=dataset_name,
            output_dir=output_dir,
            cache_dir=cache_dir,
            split=split,
            force_download=force_download,
        )
    return paths