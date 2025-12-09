# src/timejepa/data/normalizer.py

"""
Normalizers for time series data with reversible transformations.

⚠️ IMPORTANT: If your model uses RevIN (Reversible Instance Normalization) layers,
you should use IdentityNormalizer to avoid double normalization.

RevIN is designed to receive raw (unnormalized) data and handles normalization
internally with instance-level statistics computed during the forward pass.

Use StandardScaler/MinMaxScaler for:
- Models without RevIN (e.g., MLPs, LSTMs, classical baselines)
- Ablation studies comparing with/without RevIN
- Visualization and exploratory data analysis
- Single-dataset experiments where distribution shift is not a concern

For multi-dataset training with domain generalization (JEPA-style models),
always use IdentityNormalizer and let RevIN handle normalization.
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal
import numpy as np
from loguru import logger


class Normalizer(ABC):
    """Abstract base class for all normalizers."""
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> "Normalizer":
        """Fit normalizer to data.
        
        Args:
            data: Array of shape (n_series, seq_len) or (n_series, seq_len, n_features)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted statistics.
        
        Args:
            data: Array of shape (n_series, seq_len) or (n_series, seq_len, n_features)
            
        Returns:
            Normalized data with same shape as input
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization.
        
        Args:
            data: Normalized array
            
        Returns:
            Original scale data
        """
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class IdentityNormalizer(Normalizer):
    """
    No-op normalizer that returns data unchanged.
    
    Use this when:
    - Your model has RevIN layers
    - You want to train on raw data
    - You're doing multi-dataset training with domain generalization
    
    This normalizer is always fitted (is_fitted=True) since it requires no statistics.
    """
    
    def __init__(self):
        super().__init__()
        self.is_fitted = True  # ← CORRECTION : Toujours fitted dès l'init
    
    def fit(self, data: np.ndarray) -> "IdentityNormalizer":
        """No-op fit (already fitted)."""
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return data unchanged."""
        return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Return data unchanged."""
        return data


class StandardScaler(Normalizer):
    """
    Standardization: (x - μ) / σ
    
    Normalizes each time series to zero mean and unit variance.
    Can operate per-series or globally across all series.
    
    ⚠️ WARNING: Do not use with RevIN-based models (use IdentityNormalizer instead).
    """
    
    def __init__(self, mode: Literal["per_series", "global"] = "per_series", epsilon: float = 1e-8):
        """
        Args:
            mode: "per_series" normalizes each series independently,
                  "global" uses global statistics across all series
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray) -> "StandardScaler":
        """Compute mean and std from data.
        
        Args:
            data: Shape (n_series, seq_len) or (n_series, seq_len, n_features)
        """
        if self.mode == "per_series":
            # Mean/std per series: shape (n_series, 1) or (n_series, 1, n_features)
            axis = 1
            keepdims = True
        else:  # global
            # Single mean/std for all data
            axis = None
            keepdims = False

        print(data.shape)
        
        self.mean = np.mean(data, axis=axis, keepdims=keepdims)
        self.std = np.std(data, axis=axis, keepdims=keepdims)
        
        # Avoid division by zero
        self.std = np.where(self.std < self.epsilon, 1.0, self.std)
        
        self.is_fitted = True
        logger.info(f"StandardScaler fitted in '{self.mode}' mode")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Restore original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        
        return data * self.std + self.mean


class MinMaxScaler(Normalizer):
    """
    Min-Max scaling: (x - min) / (max - min) → [0, 1]
    
    Scales each time series to [0, 1] range.
    Can operate per-series or globally.
    
    ⚠️ WARNING: Do not use with RevIN-based models (use IdentityNormalizer instead).
    """
    
    def __init__(
        self, 
        mode: Literal["per_series", "global"] = "per_series",
        feature_range: tuple[float, float] = (0.0, 1.0),
        epsilon: float = 1e-8
    ):
        """
        Args:
            mode: "per_series" or "global"
            feature_range: Desired output range (min, max)
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.mode = mode
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Compute min and max from data."""
        if self.mode == "per_series":
            axis = 1
            keepdims = True
        else:
            axis = None
            keepdims = False
        
        self.data_min = np.min(data, axis=axis, keepdims=keepdims)
        self.data_max = np.max(data, axis=axis, keepdims=keepdims)
        
        # Avoid division by zero (if min == max, keep range as 1)
        data_range = self.data_max - self.data_min
        data_range = np.where(data_range < self.epsilon, 1.0, data_range)
        self.data_range = data_range
        
        self.is_fitted = True
        logger.info(f"MinMaxScaler fitted in '{self.mode}' mode, range={self.feature_range}")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale to [feature_range[0], feature_range[1]]."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        
        # Scale to [0, 1]
        normalized = (data - self.data_min) / self.data_range
        
        # Scale to feature_range
        min_val, max_val = self.feature_range
        return normalized * (max_val - min_val) + min_val
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Restore original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        
        # Reverse feature_range scaling
        min_val, max_val = self.feature_range
        normalized = (data - min_val) / (max_val - min_val)
        
        # Reverse [0, 1] scaling
        return normalized * self.data_range + self.data_min


def get_normalizer(
    normalizer_type: str,
    mode: str = "per_series",
    **kwargs
) -> Normalizer:
    """
    Factory function to create normalizers.
    
    Args:
        normalizer_type: One of "identity", "standard", "minmax"
        mode: "per_series" or "global" (ignored for identity)
        **kwargs: Additional arguments passed to the normalizer
    
    Returns:
        Normalizer instance
    
    Example:
        >>> norm = get_normalizer("identity")
        >>> norm = get_normalizer("standard", mode="per_series")
        >>> norm = get_normalizer("minmax", mode="global", feature_range=(-1, 1))
    """
    normalizer_type = normalizer_type.lower()
    
    if normalizer_type == "identity":
        return IdentityNormalizer()
    
    elif normalizer_type == "standard":
        return StandardScaler(mode=mode, **kwargs)
    
    elif normalizer_type == "minmax":
        return MinMaxScaler(mode=mode, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown normalizer type: '{normalizer_type}'. "
            f"Choose from: 'identity', 'standard', 'minmax'"
        )