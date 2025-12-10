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

Supports both:
- Fixed-length data: np.ndarray of shape (n_series, seq_len) or (n_series, seq_len, n_features)
- Variable-length data: list[np.ndarray] where each array has different length
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal, Union
import numpy as np
from loguru import logger


# Type alias for data that can be either fixed or variable length
DataType = Union[np.ndarray, list[np.ndarray]]


def robust_clip(
    series: np.ndarray, 
    n_sigma: float = 5.0,
    use_mad: bool = True
) -> np.ndarray:
    """
    Clip outliers beyond n_sigma from center.
    
    Uses median and MAD (Median Absolute Deviation) by default,
    which are more robust to outliers than mean/std.
    
    Args:
        series: Input array of any shape
        n_sigma: Number of standard deviations for clipping threshold
        use_mad: If True, use median/MAD (robust). If False, use mean/std.
        
    Returns:
        Clipped array with same shape as input
    """
    if use_mad:
        center = np.median(series)
        mad = np.median(np.abs(series - center))
        # Scale MAD to approximate std (for normal distribution)
        std_estimate = mad * 1.4826
    else:
        center = np.mean(series)
        std_estimate = np.std(series)
    
    # Avoid zero std
    if std_estimate < 1e-8:
        return series
    
    lower = center - n_sigma * std_estimate
    upper = center + n_sigma * std_estimate
    
    return np.clip(series, lower, upper)


def robust_clip_dataset(
    data: DataType,
    n_sigma: float = 5.0,
    use_mad: bool = True
) -> DataType:
    """
    Apply robust clipping to a dataset.
    
    Args:
        data: Either np.ndarray (n_series, seq_len, ...) or list of variable-length arrays
        n_sigma: Number of standard deviations for clipping threshold
        use_mad: If True, use median/MAD (robust). If False, use mean/std.
        
    Returns:
        Clipped data in same format as input
    """
    if isinstance(data, list):
        return [robust_clip(series, n_sigma, use_mad) for series in data]
    else:
        # Apply per-series for arrays
        return np.array([robust_clip(series, n_sigma, use_mad) for series in data])


class Normalizer(ABC):
    """Abstract base class for all normalizers."""
    
    def __init__(self):
        self.is_fitted = False
        self._input_was_list = False
    
    @abstractmethod
    def fit(self, data: DataType) -> "Normalizer":
        """Fit normalizer to data.
        
        Args:
            data: Array of shape (n_series, seq_len) or (n_series, seq_len, n_features),
                  OR list of variable-length arrays
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, data: DataType) -> DataType:
        """Transform data using fitted statistics.
        
        Args:
            data: Array or list of arrays (same format as fit)
            
        Returns:
            Normalized data with same format as input
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: DataType) -> DataType:
        """Reverse the normalization.
        
        Args:
            data: Normalized array or list of arrays
            
        Returns:
            Original scale data
        """
        pass
    
    def fit_transform(self, data: DataType) -> DataType:
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
    
    def __init__(self, clip_outliers: bool = False, clip_sigma: float = 5.0):
        """
        Args:
            clip_outliers: If True, apply robust clipping to remove extreme outliers
            clip_sigma: Number of sigma for outlier clipping (only if clip_outliers=True)
        """
        super().__init__()
        self.is_fitted = True
        self.clip_outliers = clip_outliers
        self.clip_sigma = clip_sigma
    
    def fit(self, data: DataType) -> "IdentityNormalizer":
        """No-op fit (already fitted)."""
        return self
    
    def transform(self, data: DataType) -> DataType:
        """Return data unchanged (or clipped if clip_outliers=True)."""
        if self.clip_outliers:
            "clipping"
            return robust_clip_dataset(data, self.clip_sigma)
        return data
    
    def inverse_transform(self, data: DataType) -> DataType:
        """Return data unchanged."""
        return data


class StandardScaler(Normalizer):
    """
    Standardization: (x - μ) / σ
    
    Normalizes each time series to zero mean and unit variance.
    Can operate per-series or globally across all series.
    
    Supports both fixed-length arrays and variable-length lists.
    
    ⚠️ WARNING: Do not use with RevIN-based models (use IdentityNormalizer instead).
    """
    
    def __init__(
        self, 
        mode: Literal["per_series", "global"] = "per_series", 
        epsilon: float = 1e-8,
        clip_outliers: bool = False,
        clip_sigma: float = 5.0
    ):
        """
        Args:
            mode: "per_series" normalizes each series independently,
                  "global" uses global statistics across all series
            epsilon: Small constant for numerical stability
            clip_outliers: If True, apply robust clipping before normalization
            clip_sigma: Number of sigma for outlier clipping
        """
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        self.clip_outliers = clip_outliers
        self.clip_sigma = clip_sigma
        
        # Statistics storage - can be arrays (fixed length) or lists (variable length)
        self.mean: Optional[Union[np.ndarray, list[float]]] = None
        self.std: Optional[Union[np.ndarray, list[float]]] = None
        
        # Global stats (for mode="global")
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
    
    def _is_variable_length(self, data: DataType) -> bool:
        """Check if data is variable-length (list of arrays)."""
        return isinstance(data, list)
    
    def fit(self, data: DataType) -> "StandardScaler":
        """Compute mean and std from data.
        
        Args:
            data: Shape (n_series, seq_len, ...) OR list of variable-length arrays
        """
        self._input_was_list = self._is_variable_length(data)
        
        if self._input_was_list:
            self._fit_variable_length(data)
        else:
            self._fit_fixed_length(data)
        
        self.is_fitted = True
        logger.info(f"StandardScaler fitted in '{self.mode}' mode (variable_length={self._input_was_list})")
        return self
    
    def _fit_fixed_length(self, data: np.ndarray) -> None:
        """Fit on fixed-length array."""
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        if self.mode == "per_series":
            axis = 1
            keepdims = True
            self.mean = np.mean(data, axis=axis, keepdims=keepdims)
            self.std = np.std(data, axis=axis, keepdims=keepdims)
            self.std = np.where(self.std < self.epsilon, 1.0, self.std)
        else:
            self.global_mean = float(np.mean(data))
            self.global_std = float(np.std(data))
            if self.global_std < self.epsilon:
                self.global_std = 1.0
    
    def _fit_variable_length(self, data: list[np.ndarray]) -> None:
        """Fit on variable-length list of arrays."""
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        if self.mode == "per_series":
            self.mean = []
            self.std = []
            for series in data:
                self.mean.append(float(np.mean(series)))
                std = float(np.std(series))
                self.std.append(std if std > self.epsilon else 1.0)
        else:
            all_values = np.concatenate([s.flatten() for s in data])
            self.global_mean = float(np.mean(all_values))
            self.global_std = float(np.std(all_values))
            if self.global_std < self.epsilon:
                self.global_std = 1.0
    
    def transform(self, data: DataType) -> DataType:
        """Normalize data to zero mean and unit variance."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")
        
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        is_list = self._is_variable_length(data)
        
        if is_list:
            return self._transform_variable_length(data)
        else:
            return self._transform_fixed_length(data)
    
    def _transform_fixed_length(self, data: np.ndarray) -> np.ndarray:
        """Transform fixed-length array."""
        if self.mode == "per_series":
            return (data - self.mean) / self.std
        else:
            return (data - self.global_mean) / self.global_std
    
    def _transform_variable_length(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Transform variable-length list."""
        result = []
        for i, series in enumerate(data):
            if self.mode == "per_series":
                normalized = (series - self.mean[i]) / self.std[i]
            else:
                normalized = (series - self.global_mean) / self.global_std
            result.append(normalized)
        return result
    
    def inverse_transform(self, data: DataType) -> DataType:
        """Restore original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        
        is_list = self._is_variable_length(data)
        
        if is_list:
            return self._inverse_transform_variable_length(data)
        else:
            return self._inverse_transform_fixed_length(data)
    
    def _inverse_transform_fixed_length(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform fixed-length array."""
        if self.mode == "per_series":
            return data * self.std + self.mean
        else:
            return data * self.global_std + self.global_mean
    
    def _inverse_transform_variable_length(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Inverse transform variable-length list."""
        result = []
        for i, series in enumerate(data):
            if self.mode == "per_series":
                original = series * self.std[i] + self.mean[i]
            else:
                original = series * self.global_std + self.global_mean
            result.append(original)
        return result


class MinMaxScaler(Normalizer):
    """
    Min-Max scaling: (x - min) / (max - min) → [0, 1]
    
    Scales each time series to [0, 1] range.
    Can operate per-series or globally.
    
    Supports both fixed-length arrays and variable-length lists.
    
    ⚠️ WARNING: Do not use with RevIN-based models (use IdentityNormalizer instead).
    """
    
    def __init__(
        self, 
        mode: Literal["per_series", "global"] = "per_series",
        feature_range: tuple[float, float] = (0.0, 1.0),
        epsilon: float = 1e-8,
        clip_outliers: bool = False,
        clip_sigma: float = 5.0
    ):
        """
        Args:
            mode: "per_series" or "global"
            feature_range: Desired output range (min, max)
            epsilon: Small constant to avoid division by zero
            clip_outliers: If True, apply robust clipping before normalization
            clip_sigma: Number of sigma for outlier clipping
        """
        super().__init__()
        self.mode = mode
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.clip_outliers = clip_outliers
        self.clip_sigma = clip_sigma
        
        # Statistics storage
        self.data_min: Optional[Union[np.ndarray, list[float]]] = None
        self.data_max: Optional[Union[np.ndarray, list[float]]] = None
        self.data_range: Optional[Union[np.ndarray, list[float]]] = None
        
        # Global stats
        self.global_min: Optional[float] = None
        self.global_max: Optional[float] = None
        self.global_range: Optional[float] = None
    
    def _is_variable_length(self, data: DataType) -> bool:
        """Check if data is variable-length (list of arrays)."""
        return isinstance(data, list)
    
    def fit(self, data: DataType) -> "MinMaxScaler":
        """Compute min and max from data."""
        self._input_was_list = self._is_variable_length(data)
        
        if self._input_was_list:
            self._fit_variable_length(data)
        else:
            self._fit_fixed_length(data)
        
        self.is_fitted = True
        logger.info(f"MinMaxScaler fitted in '{self.mode}' mode, range={self.feature_range} (variable_length={self._input_was_list})")
        return self
    
    def _fit_fixed_length(self, data: np.ndarray) -> None:
        """Fit on fixed-length array."""
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        if self.mode == "per_series":
            axis = 1
            keepdims = True
            self.data_min = np.min(data, axis=axis, keepdims=keepdims)
            self.data_max = np.max(data, axis=axis, keepdims=keepdims)
            data_range = self.data_max - self.data_min
            self.data_range = np.where(data_range < self.epsilon, 1.0, data_range)
        else:
            self.global_min = float(np.min(data))
            self.global_max = float(np.max(data))
            self.global_range = self.global_max - self.global_min
            if self.global_range < self.epsilon:
                self.global_range = 1.0
    
    def _fit_variable_length(self, data: list[np.ndarray]) -> None:
        """Fit on variable-length list of arrays."""
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        if self.mode == "per_series":
            self.data_min = []
            self.data_max = []
            self.data_range = []
            for series in data:
                min_val = float(np.min(series))
                max_val = float(np.max(series))
                range_val = max_val - min_val
                self.data_min.append(min_val)
                self.data_max.append(max_val)
                self.data_range.append(range_val if range_val > self.epsilon else 1.0)
        else:
            all_values = np.concatenate([s.flatten() for s in data])
            self.global_min = float(np.min(all_values))
            self.global_max = float(np.max(all_values))
            self.global_range = self.global_max - self.global_min
            if self.global_range < self.epsilon:
                self.global_range = 1.0
    
    def transform(self, data: DataType) -> DataType:
        """Scale to [feature_range[0], feature_range[1]]."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        
        if self.clip_outliers:
            data = robust_clip_dataset(data, self.clip_sigma)
        
        is_list = self._is_variable_length(data)
        
        if is_list:
            return self._transform_variable_length(data)
        else:
            return self._transform_fixed_length(data)
    
    def _transform_fixed_length(self, data: np.ndarray) -> np.ndarray:
        """Transform fixed-length array."""
        if self.mode == "per_series":
            normalized = (data - self.data_min) / self.data_range
        else:
            normalized = (data - self.global_min) / self.global_range
        
        min_val, max_val = self.feature_range
        return normalized * (max_val - min_val) + min_val
    
    def _transform_variable_length(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Transform variable-length list."""
        min_val, max_val = self.feature_range
        result = []
        for i, series in enumerate(data):
            if self.mode == "per_series":
                normalized = (series - self.data_min[i]) / self.data_range[i]
            else:
                normalized = (series - self.global_min) / self.global_range
            scaled = normalized * (max_val - min_val) + min_val
            result.append(scaled)
        return result
    
    def inverse_transform(self, data: DataType) -> DataType:
        """Restore original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        
        is_list = self._is_variable_length(data)
        
        if is_list:
            return self._inverse_transform_variable_length(data)
        else:
            return self._inverse_transform_fixed_length(data)
    
    def _inverse_transform_fixed_length(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform fixed-length array."""
        min_val, max_val = self.feature_range
        normalized = (data - min_val) / (max_val - min_val)
        
        if self.mode == "per_series":
            return normalized * self.data_range + self.data_min
        else:
            return normalized * self.global_range + self.global_min
    
    def _inverse_transform_variable_length(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Inverse transform variable-length list."""
        min_val, max_val = self.feature_range
        result = []
        for i, series in enumerate(data):
            normalized = (series - min_val) / (max_val - min_val)
            if self.mode == "per_series":
                original = normalized * self.data_range[i] + self.data_min[i]
            else:
                original = normalized * self.global_range + self.global_min
            result.append(original)
        return result


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
        **kwargs: Additional arguments passed to the normalizer:
            - clip_outliers (bool): Apply robust clipping to outliers
            - clip_sigma (float): Number of sigma for clipping threshold
            - feature_range (tuple): For minmax, the output range
            - epsilon (float): Numerical stability constant
    
    Returns:
        Normalizer instance
    
    Example:
        >>> norm = get_normalizer("identity")
        >>> norm = get_normalizer("identity", clip_outliers=True, clip_sigma=5.0)
        >>> norm = get_normalizer("standard", mode="per_series")
        >>> norm = get_normalizer("minmax", mode="global", feature_range=(-1, 1))
        
        # With variable-length data:
        >>> data = [np.random.randn(100), np.random.randn(150), np.random.randn(80)]
        >>> norm = get_normalizer("standard", mode="per_series")
        >>> normalized = norm.fit_transform(data)  # Returns list of normalized arrays
    """
    normalizer_type = normalizer_type.lower()
    
    if normalizer_type == "identity":
        return IdentityNormalizer(**kwargs)
    
    elif normalizer_type == "standard":
        return StandardScaler(mode=mode, **kwargs)
    
    elif normalizer_type == "minmax":
        return MinMaxScaler(mode=mode, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown normalizer type: '{normalizer_type}'. "
            f"Choose from: 'identity', 'standard', 'minmax'"
        )