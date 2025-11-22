# src/timejepa/training/__init__.py
"""
Training infrastructure for TimeJEPA.
"""

from .jepa_pretrain_module import JEPAPretrainModule
from .finetune_module import FinetuneModule, MultiHorizonFinetuneModule
from .callbacks.ema_callback import EMACallback, GradientClipCallback
from .utils.masking import (
    get_masking_strategy,
    RandomMasking,
    BlockMasking,
    TemporalMasking,
    RandomTemporalMasking,
)
from .utils.metrics import (
    jepa_loss,
    compute_pretrain_metrics,
    compute_forecasting_metrics,
    mse,
    mae,
    rmse,
    mape,
    smape,
)

__all__ = [
    # Pretrain
    'JEPAPretrainModule',
    
    # Finetune
    'FinetuneModule',
    'MultiHorizonFinetuneModule',
    
    # Callbacks
    'EMACallback',
    'GradientClipCallback',
    
    # Masking
    'get_masking_strategy',
    'RandomMasking',
    'BlockMasking',
    'TemporalMasking',
    'RandomTemporalMasking',
    
    # Metrics
    'jepa_loss',
    'compute_pretrain_metrics',
    'compute_forecasting_metrics',
    'mse',
    'mae',
    'rmse',
    'mape',
    'smape',
]