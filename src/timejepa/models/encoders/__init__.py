# src/timejepa/models/encoders/__init__.py
"""
Encoders for TimeJEPA.

Note: `PatchTSTEncoder` / `ChannelIndependentPatchTSTEncoder` were removed in
commit 02c5ac1 ("Remove deprecated encoder implementation"). Patching is now
handled upstream by `models.components.patching.Patching`, and the encoder is a
bare transformer stack (`BareTransformerEncoder`).
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
