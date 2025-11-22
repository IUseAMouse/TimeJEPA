# src/timejepa/models/encoders/__init__.py
"""
Encoders for TimeJEPA.
"""

from .patchtst_encoder import (
    PatchTSTEncoder,
    ChannelIndependentPatchTSTEncoder
)
from .target_encoder import (
    TargetEncoder,
    EMAUpdater,
    DualEncoderWrapper
)

__all__ = [
    # PatchTST encoders
    'PatchTSTEncoder',
    'ChannelIndependentPatchTSTEncoder',
    
    # Target encoder & EMA
    'TargetEncoder',
    'EMAUpdater',
    'DualEncoderWrapper',
]