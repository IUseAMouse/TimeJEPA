# src/timejepa/data/dataset.py
"""
PyTorch Dataset for time series with sliding windows for JEPA training.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalizer import Normalizer, get_normalizer

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series with sliding windows.
    
    For JEPA training, each sample returns:
    - context: Past window (to be encoded by main encoder)
    - target: Future window(s) (to be encoded by target encoder)
    - metadata: Series ID, timestamps, etc.
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
        min_series_length: Optional[int] = None
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
        """
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.return_tensor = return_tensor
        self.normalize_mode = normalize_mode
        
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        data = np.load(self.data_path, allow_pickle=True)

        # Handle object arrays (variable length series)
        if data.dtype == object:
            logger.warning("Data contains variable-length series")
            # Filter by min length
            min_len = context_length + prediction_length
            data = [s for s in data if len(s) >= min_len]
            
            # 🔥 FIX: Reconvertir en array numpy object
            data = np.array(data, dtype=object)
            
            logger.info(f"Kept {len(data)} series with length >= {min_len}")

        # Limit number of series if requested
        if max_series is not None and len(data) > max_series:
            logger.info(f"Limiting to {max_series} series (out of {len(data)})")
            data = data[:max_series]

        # Filter by minimum length
        if min_series_length is not None:
            if isinstance(data, np.ndarray) and data.dtype == object:
                # 🔥 FIX: Filtrer puis reconvertir
                data = [s for s in data if len(s) >= min_series_length]
                data = np.array(data, dtype=object)
            elif isinstance(data, np.ndarray):
                # Homogeneous array
                mask = np.array([s.shape[-1] >= min_series_length for s in data])
                data = data[mask]
            
            logger.info(f"After length filter: {len(data)} series")

        # Convert to array if homogeneous
        if isinstance(data, np.ndarray) and data.dtype == object:
            # Check if all same length
            lengths = [s.shape[-1] for s in data]
            if len(set(lengths)) == 1:
                data = np.stack(data)
                logger.info(f"Converted to dense array: {data.shape}")

        self.data = data
        self.is_multivariate = data.ndim == 3
        
        logger.info(f"Data shape: {data.shape if data.dtype != object else f'({len(data)}, variable)'}")
        logger.info(f"Is multivariate: {self.is_multivariate}")
        
        # Fit or use provided normalizer
        if normalizer is None:
            logger.info("No normalizer provided, using IdentityNormalizer (data will be raw)")
            self.normalizer = get_normalizer("identity")
        else:
            logger.info(f"Using provided normalizer: {normalizer.__class__.__name__}")
            self.normalizer = normalizer
        
        # Si Identity, pas besoin de fit
        if not self.normalizer.is_fitted:
            self.normalizer.fit(self.data)
        
        # Normaliser (ou pas si Identity)
        self.normalized_data = self.normalizer.transform(self.data)
        
        # Generate window indices
        self.window_indices = self._generate_window_indices()
        
        logger.info(f"Created dataset with {len(self)} windows")
    
    def _generate_window_indices(self) -> List[Tuple[int, int]]:
        """
        Generate (series_idx, start_idx) pairs for all valid windows.
        
        Returns:
            List of (series_id, window_start) tuples
        """
        indices = []
        min_length = self.context_length + self.prediction_length
        
        if self.data.dtype == object:
            # Variable length series
            for series_idx, series in enumerate(self.normalized_data):
                seq_length = series.shape[-1]
                if seq_length < min_length:
                    continue
                
                # Generate sliding windows
                for start_idx in range(0, seq_length - min_length + 1, self.stride):
                    indices.append((series_idx, start_idx))
        else:
            # Homogeneous length
            seq_length = self.normalized_data.shape[-1]
            
            if seq_length < min_length:
                raise ValueError(
                    f"Series too short: {seq_length} < {min_length} "
                    f"(context={self.context_length} + pred={self.prediction_length})"
                )
            
            for series_idx in range(len(self.normalized_data)):
                for start_idx in range(0, seq_length - min_length + 1, self.stride):
                    indices.append((series_idx, start_idx))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.window_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - 'context': Context window (past), shape (context_length,) or (C, context_length)
            - 'target': Target window (future), shape (prediction_length,) or (C, prediction_length)
            - 'series_id': Index of the time series
            - 'start_idx': Start index of the context window
        """
        series_idx, start_idx = self.window_indices[idx]
        
        if self.data.dtype == object:
            series = self.normalized_data[series_idx]
        else:
            series = self.normalized_data[series_idx]
        
        # Extract windows
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        
        if self.is_multivariate:
            # Shape: (num_channels, seq_length)
            context = series[:, start_idx:context_end]
            target = series[:, context_end:target_end]
        else:
            # Shape: (seq_length,)
            context = series[start_idx:context_end]
            target = series[context_end:target_end]
        
        # Convert to tensor if requested
        if self.return_tensor:
            context = torch.from_numpy(context).float()
            target = torch.from_numpy(target).float()
        
        return {
            'context': context,
            'target': target,
            'series_id': series_idx,
            'start_idx': start_idx
        }
    
    def get_normalizer(self) -> Normalizer:
        """Get the fitted normalizer for evaluation/inference."""
        return self.normalizer
    
    def get_raw_series(self, series_id: int) -> np.ndarray:
        """Get raw (unnormalized) series by ID."""
        if self.data.dtype == object:
            return self.data[series_id]
        return self.data[series_id]


class MultiHorizonDataset(TimeSeriesDataset):
    """
    Extended dataset that returns multiple future horizons.
    
    Useful for training models to predict at multiple time scales.
    """
    
    def __init__(
        self,
        data_path: Path,
        context_length: int,
        prediction_lengths: List[int],  # Multiple horizons
        stride: int = 1,
        normalizer: Optional[Normalizer] = None,
        normalize_mode: str = "per_series",
        return_tensor: bool = True,
        max_series: Optional[int] = None
    ):
        """
        Args:
            prediction_lengths: List of prediction horizons (e.g., [30, 60, 120])
        """
        self.prediction_lengths = sorted(prediction_lengths)
        max_pred_len = max(prediction_lengths)
        
        super().__init__(
            data_path=data_path,
            context_length=context_length,
            prediction_length=max_pred_len,  # Use max for window generation
            stride=stride,
            normalizer=normalizer,
            normalize_mode=normalize_mode,
            return_tensor=return_tensor,
            max_series=max_series
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with:
            - 'context': Context window
            - 'targets': Dict of {horizon: target_window}
            - 'series_id': Series index
            - 'start_idx': Start index
        """
        series_idx, start_idx = self.window_indices[idx]
        
        if self.data.dtype == object:
            series = self.normalized_data[series_idx]
        else:
            series = self.normalized_data[series_idx]
        
        # Extract context
        context_end = start_idx + self.context_length
        
        if self.is_multivariate:
            context = series[:, start_idx:context_end]
        else:
            context = series[start_idx:context_end]
        
        # Extract multiple targets
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