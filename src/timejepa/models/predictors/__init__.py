# src/timejepa/models/predictors/__init__.py
"""
Predictors for JEPA pretraining.
"""

from .transformer_predictor import (
    TransformerPredictor,
    MLPPredictor
)

__all__ = [
    'TransformerPredictor',
    'MLPPredictor',
]