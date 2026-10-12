"""
Evaluation package.

`loading` is the single checkpoint/model-construction path shared by every
evaluation entry point; `gift` is the GIFT-Eval protocol (97 configs, official
splits and metrics).
"""

from .loading import create_model_from_config, load_checkpoint

__all__ = ["create_model_from_config", "load_checkpoint"]
