"""
Model construction and checkpoint loading for evaluation.

Every evaluation entry point loads checkpoints through this module, so a fix
lands in one place.
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


def create_model_from_config(cfg: DictConfig):
    """
    Create a JEPA-TST model from the Hydra config.

    The model's prediction_length is fixed at creation time; use
    model.forecast(context, n=horizon) for other horizons.

    `model.builder`, a dotted "package.module:function", builds a non-JEPA
    model duck-typed on JEPATST from a sibling package. Absent, the JEPA-TST
    below is built.
    """
    builder = cfg.model.get('builder')
    if builder:
        module_name, _, func_name = str(builder).partition(':')
        if not func_name:
            raise ValueError(f"model.builder must be 'module:function', got {builder!r}")
        import importlib
        build = getattr(importlib.import_module(module_name), func_name)
        logger.info(f"model.builder={builder}: model built outside timejepa")
        return build(cfg)
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
        # Scale-conditioning FiLM; absent means False.
        cross_resolution=bool(cfg.model.get('cross_resolution', False)),
        # Robust arcsinh compression. The flag carries no weights; the
        # robust_scaler.is_robust marker lets load_checkpoint refuse a
        # flag/checkpoint mismatch.
        robust_scale=bool(cfg.model.get('robust_scale', False)),
        # z head on the predictor plus spread gate on the quantile head. The
        # predictor.z_head.* weights sit under a core prefix, so a mismatch
        # is refused in both directions.
        error_signal=bool(cfg.model.get('error_signal', False)),
    )

    # Add forecasting decoder
    model.decoder = ForecastingHead(
        d_model=cfg.model.decoder.d_model,
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        decoder_type=cfg.model.decoder.type,
        revin=model.revin,
        # Optional head width. A separate key on purpose: old configs carry
        # a decoder.hidden_dim that was never wired, and waking it would
        # break existing checkpoints. Absent means the d_model fallback.
        quantile_hidden_dim=cfg.model.decoder.get('quantile_hidden_dim'),
        # The decoder rebuilt here must carry the same flag as the model.
        error_signal=bool(cfg.model.get('error_signal', False)),
    )

    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    allow_partial: bool = False,
) -> torch.nn.Module:
    """
    Load a checkpoint: Lightning format (state_dict with 'model.' prefix),
    the pretrained-encoder format, or a raw state dict.

    Refusal contract: a dropped, missing or unexpected key under a core
    prefix (`online_encoder.`, `predictor.`, `patching.`, `robust_scaler.`)
    raises RuntimeError instead of warning. The only legitimate mismatch is
    the decoder swap (point head to quantile head); an encoder or predictor
    mismatch would score freshly initialised weights. `allow_partial=True`
    is a manual-debugging escape hatch; no config sets it.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # weights_only=False for OmegaConf objects stored in the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info("  Detected Lightning checkpoint format")

        # Strip the 'model.' and '_orig_mod.' prefixes; drop the target
        # encoder (not needed for inference) and the RevIN runtime buffers.
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("model.", "").replace("_orig_mod.", "")

            if "target_encoder" in clean_key:
                continue

            if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                continue

            cleaned_state_dict[clean_key] = v

    elif 'online_encoder' in checkpoint:
        # Format written by save_pretrained_encoder.
        logger.info("  Detected pretrained encoder format")
        cleaned_state_dict = {}
        for component in ['online_encoder', 'predictor', 'patching', 'revin', 'decoder']:
            if component in checkpoint:
                for k, v in checkpoint[component].items():
                    cleaned_state_dict[f"{component}.{k}"] = v

    elif isinstance(checkpoint, dict) and any(k.startswith(('online_encoder', 'decoder', 'patching')) for k in checkpoint.keys()):
        logger.info("  Detected raw state dict format")
        cleaned_state_dict = checkpoint

    else:
        logger.warning(f"  Unknown format, attempting raw load. Keys: {list(checkpoint.keys())[:5]}...")
        cleaned_state_dict = checkpoint

    # load_state_dict(strict=False) still raises on shape mismatches, so
    # drop them first (the point decoder to quantile head swap).
    cleaned_state_dict, dropped = filter_loadable(model, cleaned_state_dict)
    for key, ckpt_shape, model_shape in dropped:
        logger.info(f"  re-initialising {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    expected_missing_patterns = {'target_encoder', 'revin.mean', 'revin.std'}
    critical_missing = [
        k for k in missing
        if not any(pattern in k for pattern in expected_missing_patterns)
    ]

    logger.info(f"  Loaded {len(cleaned_state_dict)} keys")

    if missing:
        non_critical = len(missing) - len(critical_missing)
        logger.info(f"  Expected missing (target_encoder, buffers): {non_critical} keys")

    # Refuse rather than warn when the core of the model is not the
    # checkpoint's: `dropped` covers shape mismatches, `critical_missing`
    # absent keys, `unexpected` core weights the model lacks (a checkpoint
    # with robust_scaler, z_head or w_film loaded into a model built without
    # the flag). A model built by model.builder declares its own core.
    core = tuple(getattr(model, 'core_prefixes',
                         ('online_encoder.', 'predictor.', 'patching.', 'robust_scaler.')))
    core_bad = ([k for k, _, _ in dropped if k.startswith(core)]
                + [k for k in critical_missing if k.startswith(core)]
                + [k for k in unexpected if k.startswith(core)])
    if core_bad and not allow_partial:
        raise RuntimeError(
            f"Checkpoint/model mismatch on core components: {core_bad[:6]}"
            f"{' ...' if len(core_bad) > 6 else ''} - evaluating would score "
            f"freshly initialised weights and produce silently wrong numbers. "
            f"Either the geometry/config does not match the checkpoint, or you "
            f"want an explicit extension (see grow_future_query_table). "
            f"Pass allow_partial=True only for manual debugging."
        )

    if critical_missing:
        logger.warning(f"  Potentially missing keys: {critical_missing[:10]}")
        if len(critical_missing) > 10:
            logger.warning(f"     ... and {len(critical_missing) - 10} more")

    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected[:5]}")

    model = model.to(device)
    model.eval()

    return model
