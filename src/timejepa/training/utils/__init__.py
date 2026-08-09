# src/timejepa/training/utils/__init__.py
"""Training utilities."""

from .masking import get_masking_strategy
from .metrics import (
    jepa_loss,
    compute_pretrain_metrics,
    compute_forecasting_metrics_extended,
    mase,
    nd,
    quantile_loss,
    weighted_quantile_loss,
)
from .baselines import (
    BASELINES,
    compute_all_baselines,
    get_seasonality,
    seasonal_naive_forecast,
    last_value_forecast,
    mean_forecast,
    linear_trend_forecast,
)

__all__ = [
    'get_masking_strategy',
    'jepa_loss',
    'compute_pretrain_metrics',
    'compute_forecasting_metrics_extended',
    # Benchmark metrics
    'mase',
    'nd',
    'quantile_loss',
    'weighted_quantile_loss',
    # Baselines
    'BASELINES',
    'compute_all_baselines',
    'get_seasonality',
    'seasonal_naive_forecast',
    'last_value_forecast',
    'mean_forecast',
    'linear_trend_forecast',
]
