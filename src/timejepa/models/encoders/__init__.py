# src/timejepa/models/encoders/__init__.py
"""
Encoders for TimeJEPA.

Patching is handled upstream by `models.components.patching.Patching`; the
encoder is a bare transformer stack (`BareTransformerEncoder`).
"""

from .bare_encoder import BareTransformerEncoder
from .target_encoder import (
    TargetEncoder,
    EMAUpdater,
    DualEncoderWrapper
)

__all__ = [
    # Encoder
    'BareTransformerEncoder',

    # Target encoder & EMA
    'TargetEncoder',
    'EMAUpdater',
    'DualEncoderWrapper',
]
