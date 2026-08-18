"""
Model construction and checkpoint loading for evaluation.

Moved VERBATIM from scripts/evaluate.py (which now imports from here) so that
every evaluation entry point — the Monash/Nixtla script, the GIFT-Eval harness,
future ones — loads checkpoints through the same code. Two loaders is how the
B20-era bugs happened: a fix lands in one path and the other silently keeps the
old behaviour.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig

from ..models import JEPATST
from ..models.jepa_tst import filter_loadable
from ..models.decoders import ForecastingHead

logger = logging.getLogger(__name__)


def create_model_from_config(cfg: DictConfig) -> JEPATST:
    """
    Create JEPA-TST model from Hydra config with native architecture.

    The model's prediction_length is fixed at creation time.
    Use model.forecast(context, n=horizon) for different horizons.
    """
    model = JEPATST(
        input_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        d_model=cfg.model.encoder.d_model,
        num_layers=cfg.model.encoder.n_layers,
        num_heads=cfg.model.encoder.n_heads,
        d_ff=cfg.model.encoder.d_ff,
        dropout=cfg.model.encoder.dropout,
        activation=cfg.model.encoder.activation,
        predictor_type=cfg.model.predictor.type,
        predictor_num_layers=cfg.model.predictor.n_layers,
        predictor_num_heads=cfg.model.predictor.n_heads,
        predictor_d_ff=cfg.model.predictor.d_ff,
        decoder_type=cfg.model.decoder.type,
        ema_tau_base=cfg.model.target_encoder.momentum_base,
        ema_tau_end=cfg.model.target_encoder.momentum_final,
        use_revin=cfg.model.encoder.use_revin,
    )

    # Add forecasting decoder
    model.decoder = ForecastingHead(
        d_model=cfg.model.decoder.d_model,
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        decoder_type=cfg.model.decoder.type,
        revin=model.revin
    )

    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> torch.nn.Module:
    """
    Load checkpoint with support for different formats.

    Handles:
    - Lightning checkpoints (state_dict with 'model.' prefix)
    - Direct state dicts
    - Pretrained encoder format
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load with weights_only=False for OmegaConf compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine checkpoint format and extract state_dict
    if 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        logger.info("  Detected Lightning checkpoint format")

        # Clean keys: remove 'model.' and '_orig_mod.' prefixes
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("model.", "").replace("_orig_mod.", "")

            # Skip target encoder (not needed for inference)
            if "target_encoder" in clean_key:
                continue

            # Skip RevIN runtime buffers
            if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                continue

            cleaned_state_dict[clean_key] = v

    elif 'online_encoder' in checkpoint:
        # Direct save format from save_pretrained_encoder
        logger.info("  Detected pretrained encoder format")
        cleaned_state_dict = {}
        for component in ['online_encoder', 'predictor', 'patching', 'revin', 'decoder']:
            if component in checkpoint:
                for k, v in checkpoint[component].items():
                    cleaned_state_dict[f"{component}.{k}"] = v

    elif isinstance(checkpoint, dict) and any(k.startswith(('online_encoder', 'decoder', 'patching')) for k in checkpoint.keys()):
        # Raw state dict
        logger.info("  Detected raw state dict format")
        cleaned_state_dict = checkpoint

    else:
        # Try as raw state dict
        logger.warning(f"  Unknown format, attempting raw load. Keys: {list(checkpoint.keys())[:5]}...")
        cleaned_state_dict = checkpoint

    # Shape-mismatched entries must be dropped, not merely tolerated:
    # load_state_dict(strict=False) still raises on them. Swapping a point
    # decoder for the quantile head is exactly such a case.
    cleaned_state_dict, dropped = filter_loadable(model, cleaned_state_dict)
    for key, ckpt_shape, model_shape in dropped:
        logger.info(f"  ↷ re-initialising {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    # Analyze missing keys
    expected_missing_patterns = {'target_encoder', 'revin.mean', 'revin.std'}
    critical_missing = [
        k for k in missing
        if not any(pattern in k for pattern in expected_missing_patterns)
    ]

    # Log results
    logger.info(f"  ✓ Loaded {len(cleaned_state_dict)} keys")

    if missing:
        non_critical = len(missing) - len(critical_missing)
        logger.info(f"  Expected missing (target_encoder, buffers): {non_critical} keys")

    if critical_missing:
        logger.warning(f"  ⚠️ Potentially missing keys: {critical_missing[:10]}")
        if len(critical_missing) > 10:
            logger.warning(f"     ... and {len(critical_missing) - 10} more")

    if unexpected:
        logger.warning(f"  ⚠️ Unexpected keys: {unexpected[:5]}")

    model = model.to(device)
    model.eval()

    return model
