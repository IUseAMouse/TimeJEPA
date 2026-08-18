"""
Evaluation package.

`loading` is the single checkpoint/model-construction path shared by every
evaluation entry point; `gift` is the GIFT-Eval protocol (97 configs, official
splits and metrics). The Monash/Nixtla evaluation still lives in
scripts/evaluate.py and imports its loaders from here.
"""

from .loading import create_model_from_config, load_checkpoint

__all__ = ["create_model_from_config", "load_checkpoint"]
