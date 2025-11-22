# src/timejepa/models/__init__.py
"""
TimeJEPA Models.
"""

from .components.revin import RevIN
from .components.rope import RotaryPositionEmbedding
from .components.patching import Patching, UnPatching
from .components.attention import RoPEAttention, TransformerBlock

from .encoders.patchtst_encoder import (
    PatchTSTEncoder,
    ChannelIndependentPatchTSTEncoder
)
from .encoders.target_encoder import TargetEncoder, EMAUpdater

from .predictors.transformer_predictor import (
    TransformerPredictor,
    MLPPredictor
)

from .decoders.linear_decoder import (
    LinearDecoder,
    MLPDecoder,
    AttentiveDecoder,
    ForecastingHead
)

from .jepa_tst import (
    JEPATST,
    create_jepa_tst_tiny,
    create_jepa_tst_small,
    create_jepa_tst_base,
    create_jepa_tst_large,
)

__all__ = [
    # Components
    'RevIN',
    'RotaryPositionEmbedding',
    'Patching',
    'UnPatching',
    'RoPEAttention',
    'TransformerBlock',
    
    # Encoders
    'PatchTSTEncoder',
    'ChannelIndependentPatchTSTEncoder',
    'TargetEncoder',
    'EMAUpdater',
    
    # Predictors
    'TransformerPredictor',
    'MLPPredictor',
    
    # Decoders
    'LinearDecoder',
    'MLPDecoder',
    'AttentiveDecoder',
    'ForecastingHead',
    
    # Main model
    'JEPATST',
    'create_jepa_tst_tiny',
    'create_jepa_tst_small',
    'create_jepa_tst_base',
    'create_jepa_tst_large',
]