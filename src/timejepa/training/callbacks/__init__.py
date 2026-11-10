# src/timejepa/training/callbacks/__init__.py
"""
Callbacks for PyTorch Lightning training.
"""

from .ema_callback import EMACallback

__all__ = [
    'EMACallback',
]