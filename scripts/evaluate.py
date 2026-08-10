"""
Unified evaluation script for finetuned TimeJEPA model.

Supports:
- Local Monash datasets (.npy files in data_dir)
- Nixtla Long-Horizon benchmarks (auto-downloaded)

Usage:
    # Evaluate on local datasets
    python scripts/evaluate.py +checkpoint_path=path/to/checkpoint.ckpt
    python scripts/evaluate.py +checkpoint_path=path/to/checkpoint.ckpt data.datasets_eval=[traffic,electricity]
    
    # Evaluate on Nixtla benchmarks
    python scripts/evaluate.py +checkpoint_path=path/to/ckpt +nixtla=[ettm1,ettm2,weather]
    
    # With specific horizons
    python scripts/evaluate.py +checkpoint_path=path/to/ckpt +nixtla=[ettm2] +horizons=[96,192,336,720]
    
    # Both local and Nixtla
    python scripts/evaluate.py +checkpoint_path=path/to/ckpt \
        data.datasets_eval=[traffic] +nixtla=[ettm2,weather]
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import MonashDataModule
from timejepa.training.finetune_module import FinetuneModule
from timejepa.training.utils.metrics import (
    compute_forecasting_metrics_extended,
    compute_per_horizon_metrics,
    weighted_quantile_loss,
)
from timejepa.training.utils.baselines import (
    compute_all_baselines,
    get_seasonality,
)
from timejepa.models import JEPATST
from timejepa.models.jepa_tst import filter_loadable
from timejepa.models.decoders import ForecastingHead

try:
    from timejepa.data.nixtla import (
        download_and_convert,
        get_available_datasets,
        NIXTLA_REGISTRY,
    )
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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


def find_best_checkpoint(cfg: DictConfig) -> Optional[str]:
    """Find the best checkpoint from the checkpoint directory."""
    ckpt_dir = Path(cfg.data.checkpoint_dir) / cfg.model.name / "pretrain_False"
    
    if not ckpt_dir.exists():
        return None
    
    # Look for 'best' or 'last' checkpoint first
    for pattern in ['*best*.ckpt', '*last*.ckpt', '*.ckpt']:
        ckpts = list(ckpt_dir.glob(pattern))
        if ckpts:
            # Sort by modification time (most recent first)
            ckpts = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)
            return str(ckpts[0])
    
    return None


def plot_forecasts(
    contexts: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dataset_name: str,
    output_dir: Path,
    num_samples: int = 9,
    seed: int = 42,
    quantiles: Optional[torch.Tensor] = None,
    quantile_levels: Optional[List[float]] = None,
):
    """
    Plot sample forecasts in a grid.

    When a quantile fan is supplied, the prediction interval is shaded around
    the median — nested bands, darkest at the centre. A probabilistic forecast
    judged only on its median tells you nothing about whether its intervals are
    calibrated, which is the whole point of having them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    
    n = min(num_samples, len(predictions))
    
    # Sélectionner des exemples variés (bons et mauvais)
    errors = torch.mean((predictions - targets) ** 2, dim=-1)
    if errors.ndim > 1:
        errors = errors.mean(dim=-1)
    errors = errors.numpy()
    
    # Mix: quelques bons, quelques moyens, quelques mauvais
    sorted_idx = np.argsort(errors)
    n_per_group = max(1, n // 3)
    good_idx = sorted_idx[:n_per_group]
    bad_idx = sorted_idx[-n_per_group:]
    mid_start = len(sorted_idx) // 2 - n_per_group // 2
    mid_idx = sorted_idx[mid_start:mid_start + n_per_group]
    
    # Combiner et mélanger
    selected_indices = np.concatenate([good_idx, mid_idx, bad_idx])
    np.random.shuffle(selected_indices)
    indices = selected_indices[:n]
    
    # Grid layout
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        ctx = contexts[idx].cpu().numpy()
        pred = predictions[idx].cpu().numpy()
        targ = targets[idx].cpu().numpy()
        
        # Handle multivariate (plot first channel)
        if ctx.ndim > 1:
            ctx = ctx[..., 0] if ctx.shape[-1] < ctx.shape[0] else ctx[0]
            pred = pred[..., 0] if pred.shape[-1] < pred.shape[0] else pred[0]
            targ = targ[..., 0] if targ.shape[-1] < targ.shape[0] else targ[0]
        
        ctx_len = len(ctx)
        pred_len = len(pred)
        
        # Time axis
        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len, ctx_len + pred_len)
        
        # Plot
        ax.plot(t_ctx, ctx, 'b-', label='Context', alpha=0.7, linewidth=1.5)

        # Prediction intervals first, so the lines stay readable on top.
        # Pairs are taken from the outside in: (q10,q90), (q20,q80), ...
        if quantiles is not None:
            q = quantiles[idx].cpu().numpy()
            if q.ndim > 2:
                q = q[..., 0] if q.shape[-1] == 1 else q
            n_q = q.shape[-1]
            for lo in range(n_q // 2):
                hi = n_q - 1 - lo
                lvl_lo = quantile_levels[lo] if quantile_levels else lo
                lvl_hi = quantile_levels[hi] if quantile_levels else hi
                ax.fill_between(
                    t_pred, q[:, lo], q[:, hi],
                    color='orange', alpha=0.13,
                    label=f'{lvl_lo:.0%}–{lvl_hi:.0%}' if lo == 0 else None,
                )
        else:
            ax.fill_between(t_pred, pred, targ, alpha=0.2, color='red')

        ax.plot(t_pred, targ, 'g-', label='Ground Truth', linewidth=2)
        ax.plot(t_pred, pred, 'r--',
                label='Median' if quantiles is not None else 'Prediction',
                linewidth=2, alpha=0.8)
        
        # Boundary line
        ax.axvline(x=ctx_len, color='gray', linestyle=':', alpha=0.5)
        
        # Metrics for this sample
        sample_mae = np.mean(np.abs(pred - targ))
        sample_rmse = np.sqrt(np.mean((pred - targ) ** 2))
        
        ax.set_title(f'Sample {idx} | MAE: {sample_mae:.2f} | RMSE: {sample_rmse:.2f}', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
    
    # Hide unused subplots. Uses len(indices), not n: the good/mid/bad selection
    # yields 3 * max(1, n//3) indices, which is fewer than n for most values of
    # n — leaving a blank axes frame in the grid.
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{dataset_name} - Forecast Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_forecasts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 Saved forecast plots to {save_path}")


def plot_error_analysis(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dataset_name: str,
    output_dir: Path
):
    """Plot comprehensive error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    errors = (predictions - targets).flatten().cpu().numpy()
    preds_flat = predictions.flatten().cpu().numpy()
    targs_flat = targets.flatten().cpu().numpy()
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(errors):.3f}')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    
    # 2. Predicted vs Actual scatter
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Subsample for scatter
    if len(preds_flat) > 10000:
        idx = np.random.choice(len(preds_flat), 10000, replace=False)
        p_sample, t_sample = preds_flat[idx], targs_flat[idx]
    else:
        p_sample, t_sample = preds_flat, targs_flat
    
    ax2.scatter(t_sample, p_sample, alpha=0.2, s=2, c='steelblue')
    
    min_val = min(t_sample.min(), p_sample.min())
    max_val = max(t_sample.max(), p_sample.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Predicted vs Actual')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. Absolute error by actual value
    ax3 = fig.add_subplot(gs[0, 2])
    abs_errors = np.abs(errors)
    
    # Bin by actual values
    bins = np.percentile(targs_flat, np.linspace(0, 100, 11))
    bin_indices = np.digitize(targs_flat, bins)
    
    bin_means = []
    bin_centers = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(np.mean(abs_errors[mask]))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    ax3.bar(range(len(bin_means)), bin_means, color='coral', edgecolor='black')
    ax3.set_xticks(range(len(bin_means)))
    ax3.set_xticklabels([f'{c:.1f}' for c in bin_centers], rotation=45)
    ax3.set_xlabel('Actual Value Range')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Error by Value Range')
    
    # 4. Error over horizon (using metrics.py function)
    ax4 = fig.add_subplot(gs[1, 0])
    horizon_metrics = compute_per_horizon_metrics(predictions, targets)
    horizon_mae = [horizon_metrics[h]['mae'] for h in sorted(horizon_metrics.keys())]
    pred_len = len(horizon_mae)
    
    ax4.plot(range(pred_len), horizon_mae, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax4.fill_between(range(pred_len), horizon_mae, alpha=0.3)
    ax4.set_xlabel('Horizon Step')
    ax4.set_ylabel('MAE')
    ax4.set_title('Error vs Prediction Horizon')
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Q plot
    ax5 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (vs Normal)')
    
    # 6. Box plot of errors per horizon quartile
    ax6 = fig.add_subplot(gs[1, 2])
    quartile_size = pred_len // 4 if pred_len >= 4 else 1
    quartile_errors = []
    quartile_labels = []
    
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else pred_len
        q_errors = (predictions[:, start:end] - targets[:, start:end]).flatten().cpu().numpy()
        quartile_errors.append(q_errors)
        quartile_labels.append(f'H{start+1}-{end}')
    
    # matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9 and removed
    # the old spelling in 3.11, which aborted the whole evaluation of a dataset
    # after its plots had already been written. Setting the tick labels
    # afterwards works on every version and needs no feature detection.
    bp = ax6.boxplot(quartile_errors, patch_artist=True)
    ax6.set_xticks(range(1, len(quartile_labels) + 1))
    ax6.set_xticklabels(quartile_labels)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax6.set_xlabel('Horizon Quartile')
    ax6.set_ylabel('Error')
    ax6.set_title('Error Distribution by Horizon')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'{dataset_name} - Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_error_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  📊 Saved error analysis to {save_path}")


def plot_summary_comparison(
    all_results: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """Plot comparison of metrics across all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(all_results) < 1:
        return
    
    datasets = list(all_results.keys())
    metrics_to_plot = ['rmse', 'mae', 'smape', 'r2']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    
    for ax, metric in zip(axes, metrics_to_plot):
        values = [all_results[d].get(metric, 0) for d in datasets]
        bars = ax.bar(datasets, values, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Dataset')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Metrics Comparison Across Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'summary_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Saved summary comparison to {save_path}")


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run evaluation on a single dataset using native prediction length.
    
    Args:
        model: Model in eval mode
        dataloader: Test dataloader
        device: Device to run on
        max_samples: Optional limit on number of samples (for large datasets)
    
    Returns:
        Dictionary with 'contexts', 'predictions', 'targets' tensors
    """
    model.eval()

    all_contexts = []
    all_predictions = []
    all_targets = []
    all_quantiles = []
    quantile_levels = None
    total_samples = 0

    for batch in tqdm(dataloader, desc="  Inference", leave=False):
        context = batch['context'].to(device)
        target = batch['target'].to(device)

        # Add channel dim if needed (univariate)
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        # Forward pass through the model (uses native prediction_length)
        output = model.forecast(context)

        # Handle dict output from JEPATST
        if isinstance(output, dict):
            predictions = output.get('forecast_denorm', output.get('forecast'))
            if 'quantiles_denorm' in output:
                all_quantiles.append(output['quantiles_denorm'].cpu())
                quantile_levels = list(output['quantile_levels'])
        else:
            predictions = output

        # Remove channel dim if univariate
        if context.shape[-1] == 1:
            context = context.squeeze(-1)
            predictions = predictions.squeeze(-1)
            target = target.squeeze(-1)

        all_contexts.append(context.cpu())
        all_predictions.append(predictions.cpu())
        all_targets.append(target.cpu())

        total_samples += len(context)
        if max_samples and total_samples >= max_samples:
            break

    out = {
        'contexts': torch.cat(all_contexts, dim=0),
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }
    if all_quantiles:
        out['quantiles'] = torch.cat(all_quantiles, dim=0)
        out['quantile_levels'] = quantile_levels
    return out


@torch.no_grad()
def evaluate_dataset_horizon(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    horizon: int,
    device: torch.device,
    max_samples: Optional[int] = None,
    skip_revin: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run evaluation on a dataset with specific horizon using model.forecast(n=horizon).

    Uses rolling forecasts if horizon > model.prediction_length.

    Normalization
    -------------
    `skip_revin=False` is the correct default and the regime the model was
    trained in. The previous code used `skip_revin=True` with the rationale
    "nixtla long horizon datasets are already normalized" — but those datasets
    are z-scored GLOBALLY with train statistics, which is a completely different
    thing from RevIN's per-window instance normalization. Passing them with
    skip_revin=True fed the encoder inputs whose mean was far from 0 (ETTh1 test:
    mean -1.34, std 0.34) while it had only ever seen mean-0/std-1 windows, and
    compared its output against targets living in yet another space. That is what
    produced the constant level offset visible in the h96 forecast plots.

    With skip_revin=False the model instance-normalizes each context, predicts in
    that frame, and denormalizes back into the globally z-scored space where the
    targets live — exactly what PatchTST / iTransformer / TimesNet do.

    Args:
        model: Model in eval mode
        dataloader: Test dataloader (must have target of length >= horizon)
        horizon: Target forecast horizon
        device: Device to run on
        max_samples: Optional limit on number of samples
        skip_revin: Bypass RevIN. Kept only to reproduce the old (broken) numbers
            for the before/after comparison.

    Returns:
        Dictionary with 'contexts', 'predictions', 'targets' tensors
    """
    model.eval()

    all_contexts = []
    all_predictions = []
    all_targets = []
    all_quantiles = []
    quantile_levels = None
    total_samples = 0

    for batch in tqdm(dataloader, desc="  Inference", leave=False):
        context = batch['context'].to(device)
        target = batch['target'].to(device)

        # Add channel dim if needed (univariate)
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        output = model.forecast(context, n=horizon, skip_revin=skip_revin)
        # 'forecast_denorm' brings the prediction back into the space the targets
        # live in. With skip_revin=True the two are identical anyway.
        predictions = output['forecast_denorm']

        # Truncate target to horizon (dataloader may provide more)
        target = target[:, :horizon]

        # Keep the quantile fan when the head is probabilistic. Without it the
        # reported WQL is computed from the point forecast, where it collapses
        # to ND — the score a deterministic model earns — and none of the
        # quantile head's benefit would ever appear in the numbers.
        if 'quantiles_denorm' in output:
            q = output['quantiles_denorm']
            all_quantiles.append(q.cpu())
            quantile_levels = list(output['quantile_levels'])

        # Remove channel dim if univariate
        if context.shape[-1] == 1:
            context = context.squeeze(-1)
            predictions = predictions.squeeze(-1)
            target = target.squeeze(-1)

        all_contexts.append(context.cpu())
        all_predictions.append(predictions.cpu())
        all_targets.append(target.cpu())

        total_samples += len(context)
        if max_samples and total_samples >= max_samples:
            break

    out = {
        'contexts': torch.cat(all_contexts, dim=0),
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0)
    }
    if all_quantiles:
        out['quantiles'] = torch.cat(all_quantiles, dim=0)
        out['quantile_levels'] = quantile_levels
    return out


def evaluate_with_baselines(
    contexts: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    season_length: int,
    quantiles: Optional[torch.Tensor] = None,
    quantile_levels: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Score the model AND every reference baseline on identical windows.

    Without this, an absolute MSE is uninterpretable. `skill_vs_seasonal_naive`
    is the headline number: >0 means TimeJEPA beats seasonal naive on MASE,
    <=0 means it does not.

    Returns:
        {'timejepa': {...}, 'seasonal_naive': {...}, ..., '_skill': {...}}
    """
    results = {}

    results['timejepa'] = compute_forecasting_metrics_extended(
        predictions, targets, context=contexts, season_length=season_length
    )

    # With a probabilistic head, recompute WQL over the actual quantile fan.
    # compute_forecasting_metrics_extended derives it from the point forecast,
    # where WQL collapses to ND by construction — the score a deterministic
    # model earns, which would hide the entire benefit of the head.
    # The baselines stay point forecasts, and that is correct: seasonal naive IS
    # deterministic, so its CRPS is its ND. That is also how GIFT-Eval
    # normalizes, with seasonal naive at 1.00 on both MASE and CRPS.
    if quantiles is not None:
        levels = list(quantile_levels) if quantile_levels else None
        q = quantiles
        if q.ndim == 3 and q.shape[-1] == len(levels or []):
            q = q.permute(2, 0, 1)            # [B, H, Q] -> [Q, B, H]
        results['timejepa']['wql_point'] = results['timejepa']['wql']
        results['timejepa']['wql'] = weighted_quantile_loss(q, targets, levels).item()
        results['timejepa']['interval_80_width'] = float(
            (quantiles[..., -1] - quantiles[..., 0]).mean()
        )

    horizon = targets.shape[1]
    for name, base_pred in compute_all_baselines(contexts, horizon, season_length).items():
        results[name] = compute_forecasting_metrics_extended(
            base_pred, targets, context=contexts, season_length=season_length
        )

    # Relative skill scores (positive = model better)
    model_mase = results['timejepa'].get('mase')
    if model_mase is not None:
        results['_skill'] = {}
        for name in ('seasonal_naive', 'naive_last', 'context_mean', 'linear_trend'):
            ref = results[name].get('mase')
            if ref and ref > 0:
                results['_skill'][f'mase_ratio_vs_{name}'] = model_mase / ref
                results['_skill'][f'skill_vs_{name}'] = 1.0 - (model_mase / ref)

    return results


def evaluate_nixtla_dataset(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset_name: str,
    horizons: List[int],
    device: torch.device,
    output_dir: Path,
    max_samples: int = 5000,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate on a Nixtla dataset across multiple horizons.
    
    Uses model.forecast(n=horizon) which handles:
    - Truncation for horizon <= native prediction_length
    - Rolling forecasts for horizon > native prediction_length
    
    Args:
        cfg: Hydra config
        model: Already loaded model (with fixed architecture)
        dataset_name: Name of Nixtla dataset (e.g., 'ettm2')
        horizons: List of prediction horizons to evaluate
        device: Torch device
        output_dir: Where to save results
        max_samples: Max samples per horizon (to limit memory/time)
    
    Returns:
        Dictionary mapping horizon -> metrics dict
    """
    if not NIXTLA_AVAILABLE:
        raise ImportError(
            "Nixtla support requires datasetsforecast. "
            "Install with: pip install datasetsforecast"
        )

    native_horizon = model.prediction_length
    context_length = cfg.model.seq_length
    season_length = get_seasonality(dataset_name)
    skip_revin = bool(cfg.get('eval_skip_revin', False))

    logger.info(f"  Model: context={context_length}, native_horizon={native_horizon}")
    logger.info(f"  Seasonality m={season_length} | RevIN={'OFF (legacy)' if skip_revin else 'ON'}")

    if dataset_name.lower() in ('etth1', 'etth2'):
        logger.warning(
            f"  ⚠️  {dataset_name}: datasetsforecast.LongHorizon ships only ONE series "
            f"('OT') for this group, whereas the published benchmark tables average "
            f"over all 7 ETT channels. These numbers are NOT comparable to the "
            f"literature — treat them as a univariate OT-only task."
        )

    results = {}
    nixtla_cache = Path(cfg.data.data_dir) / 'nixtla'
    
    # Download data once (all horizons use same data)
    data_path = download_and_convert(
        dataset_name=dataset_name,
        output_dir=nixtla_cache,
        split='test',
    )
    
    # Find max horizon to create dataloader with sufficient target length
    max_horizon = max(horizons)
    
    for horizon in horizons:
        # Determine forecast mode
        if horizon <= native_horizon:
            mode_str = f"truncate {native_horizon}→{horizon}"
        else:
            n_rolls = (horizon + native_horizon - 1) // native_horizon
            mode_str = f"rolling: {n_rolls}×{native_horizon}"
        
        logger.info(f"\n  📏 Horizon {horizon} ({mode_str})")
        
        # DataModule for this horizon
        # We need target of length=horizon for evaluation
        dm = MonashDataModule(
            data_path=data_path,
            context_length=context_length,
            prediction_length=horizon,  # Target length for evaluation
            batch_size=cfg.data.batch_size,
            stride=horizon,  # Non-overlapping for fair evaluation
            normalize_mode='per_series',
            normalizer_type='identity',  # Nixtla data is pre-normalized
            clip_outliers=False,
            train_val_test_split=(0.0, 0.0, 1.0),
            num_workers=4,
        )
        dm.prepare_data()
        dm.setup('fit')
        
        logger.info(f"     Samples: {len(dm.test_dataset)}")
        
        # Evaluate with this horizon
        result = evaluate_dataset_horizon(
            model=model,
            dataloader=dm.test_dataloader(),
            horizon=horizon,
            device=device,
            max_samples=max_samples,
            skip_revin=skip_revin,
        )

        # Model + every baseline, scored on identical windows
        scored = evaluate_with_baselines(
            result['contexts'], result['predictions'], result['targets'], season_length,
            quantiles=result.get('quantiles'),
            quantile_levels=result.get('quantile_levels'),
        )
        metrics = dict(scored['timejepa'])
        metrics['_baselines'] = {k: v for k, v in scored.items() if k not in ('timejepa', '_skill')}
        metrics.update(scored.get('_skill', {}))
        results[horizon] = metrics

        sn = scored['seasonal_naive']
        nl = scored['naive_last']
        probabilistic = 'wql_point' in metrics
        logger.info(
            f"     MSE {metrics['mse']:.4f} | MAE {metrics['mae']:.4f} | "
            f"MASE {metrics.get('mase', float('nan')):.3f} | "
            f"WQL {metrics['wql']:.4f}"
            + (f" (point {metrics['wql_point']:.4f}, "
               f"largeur 10-90 {metrics['interval_80_width']:.3f})" if probabilistic else "")
        )
        logger.info(
            f"     baselines MASE -> seasonal_naive {sn.get('mase', float('nan')):.3f} | "
            f"naive_last {nl.get('mase', float('nan')):.3f} | "
            f"skill vs SN: {metrics.get('skill_vs_seasonal_naive', float('nan')):+.1%}"
        )

        # Plot for first horizon only
        if horizon == horizons[0]:
            plot_forecasts(
                result['contexts'],
                result['predictions'],
                result['targets'],
                f"{dataset_name}_h{horizon}",
                output_dir / "plots",
                num_samples=6,
                quantiles=result.get('quantiles'),
                quantile_levels=result.get('quantile_levels'),
            )
    
    return results


def create_nixtla_benchmark_table(
    all_results: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Create benchmark results table in standard format (like PatchTST/iTransformer papers).
    
    Args:
        all_results: Dict of dataset_name -> {horizon -> metrics}
        output_dir: Where to save CSV files
    
    Returns:
        Long-format DataFrame with all results
    """
    rows = []
    for dataset, horizon_results in all_results.items():
        for horizon, metrics in sorted(horizon_results.items()):
            baselines = metrics.get('_baselines', {})
            rows.append({
                'Dataset': dataset,
                'Horizon': horizon,
                'MSE': metrics['mse'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'SMAPE': metrics['smape'],
                'MASE': metrics.get('mase'),
                'WQL': metrics.get('wql'),
                'R2': metrics.get('r2'),
                'MASE_seasonal_naive': baselines.get('seasonal_naive', {}).get('mase'),
                'MASE_naive_last': baselines.get('naive_last', {}).get('mase'),
                'MASE_context_mean': baselines.get('context_mean', {}).get('mase'),
                'skill_vs_seasonal_naive': metrics.get('skill_vs_seasonal_naive'),
                'skill_vs_naive_last': metrics.get('skill_vs_naive_last'),
            })

    df = pd.DataFrame(rows)

    # Create pivot tables (standard benchmark format)
    mse_pivot = df.pivot(index='Dataset', columns='Horizon', values='MSE')
    mae_pivot = df.pivot(index='Dataset', columns='Horizon', values='MAE')
    mase_pivot = df.pivot(index='Dataset', columns='Horizon', values='MASE')
    skill_pivot = df.pivot(index='Dataset', columns='Horizon', values='skill_vs_seasonal_naive')

    # Add average column
    for piv in (mse_pivot, mae_pivot, mase_pivot, skill_pivot):
        piv['Avg'] = piv.mean(axis=1)

    # Save all formats
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'nixtla_results_long.csv', index=False)
    mse_pivot.to_csv(output_dir / 'nixtla_mse.csv')
    mae_pivot.to_csv(output_dir / 'nixtla_mae.csv')
    mase_pivot.to_csv(output_dir / 'nixtla_mase.csv')
    skill_pivot.to_csv(output_dir / 'nixtla_skill_vs_seasonal_naive.csv')

    # Print tables
    print("\n" + "=" * 70)
    print("📊 NIXTLA BENCHMARK RESULTS - MSE")
    print("=" * 70)
    print(mse_pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("📊 NIXTLA BENCHMARK RESULTS - MAE")
    print("=" * 70)
    print(mae_pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("📊 MASE  (scale-free — 1.0 == seasonal naive, lower is better)")
    print("=" * 70)
    print(mase_pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("🎯 SKILL vs SEASONAL NAIVE  (>0 = TimeJEPA wins, <0 = it loses)")
    print("=" * 70)
    print((skill_pivot * 100).round(1).to_string())

    # Head-to-head summary against every baseline
    print("\n" + "=" * 70)
    print("📋 HEAD-TO-HEAD  (mean MASE across all horizons)")
    print("=" * 70)
    summary = df.groupby('Dataset')[
        ['MASE', 'MASE_seasonal_naive', 'MASE_naive_last', 'MASE_context_mean']
    ].mean()
    summary.columns = ['TimeJEPA', 'SeasonalNaive', 'NaiveLast', 'ContextMean']
    summary['winner'] = summary.idxmin(axis=1)
    print(summary.round(4).to_string())
    summary.to_csv(output_dir / 'nixtla_head_to_head.csv')

    n_wins = int((summary['winner'] == 'TimeJEPA').sum())
    print(f"\n  TimeJEPA is the best model on {n_wins}/{len(summary)} datasets.")

    return df


@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    print("=" * 80)
    print("🔍 TIMEJEPA EVALUATION")
    print("=" * 80)
    
    # Get checkpoint path
    checkpoint_path = cfg.get('checkpoint_path')
    
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(cfg)
        if checkpoint_path:
            logger.info(f"Found checkpoint: {checkpoint_path}")
        else:
            raise ValueError(
                "No checkpoint found. Specify with: +checkpoint_path=path/to/checkpoint.ckpt"
            )
    
    # Validate checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Output directory
    ckpt_name = Path(checkpoint_path).stem
    output_dir = Path(cfg.data.output_dir) / "evaluation" / cfg.model.name / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create model ONCE with native architecture
    logger.info("Creating model with native architecture...")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, checkpoint_path, device)
    model.set_pretrain_mode(mode=False)
    
    native_horizon = cfg.model.prediction_length
    context_length = cfg.model.seq_length
    logger.info(f"  ✓ Model: context={context_length}, prediction_length={native_horizon}")
    
    # =========================================================================
    # NIXTLA LONG-HORIZON BENCHMARKS
    # =========================================================================
    
    nixtla_datasets = cfg.get('nixtla', None)
    nixtla_results = {}
    
    if nixtla_datasets:
        if not NIXTLA_AVAILABLE:
            logger.error(
                "❌ Nixtla evaluation requested but datasetsforecast not installed.\n"
                "   Install with: pip install datasetsforecast"
            )
        else:
            # Convert ListConfig to list
            if isinstance(nixtla_datasets, ListConfig):
                nixtla_datasets = list(nixtla_datasets)
            
            logger.info(f"\n📦 Nixtla Long-Horizon Benchmarks: {nixtla_datasets}")
            
            # Get horizons (from config or default)
            horizons = cfg.get('horizons', None)
            if horizons:
                horizons = list(horizons) if isinstance(horizons, ListConfig) else horizons
            else:
                horizons = [96, 192, 336, 720]  # Standard benchmark horizons
            
            logger.info(f"   Horizons: {horizons}")
            logger.info(f"   Native model horizon: {native_horizon}")
            
            for name in nixtla_datasets:
                print("\n" + "=" * 60)
                print(f"📈 {name.upper()}")
                print("=" * 60)
                
                # Validate dataset name
                if name.lower() not in NIXTLA_REGISTRY:
                    logger.warning(
                        f"Unknown dataset: {name}. "
                        f"Available: {get_available_datasets()}"
                    )
                    continue
                
                try:
                    # ILI uses different horizons
                    ds_horizons = horizons
                    if name.lower() == 'ili':
                        ds_horizons = [h for h in [24, 36, 48, 60] if h in horizons]
                        if not ds_horizons:
                            ds_horizons = [24, 36, 48, 60]
                    
                    results = evaluate_nixtla_dataset(
                        cfg=cfg,
                        model=model,  # Pass the single loaded model
                        dataset_name=name,
                        horizons=ds_horizons,
                        device=device,
                        output_dir=output_dir,
                    )
                    nixtla_results[name] = results
                    
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create benchmark summary tables
            if nixtla_results:
                create_nixtla_benchmark_table(nixtla_results, output_dir)
                
                # Save as JSON
                json_results = {
                    ds: {str(h): m for h, m in hr.items()}
                    for ds, hr in nixtla_results.items()
                }
                with open(output_dir / 'nixtla_results.json', 'w') as f:
                    json.dump(json_results, f, indent=2)
    
 
    datasets_eval = cfg.data.get('datasets_eval', [])
    
    if not datasets_eval:
        # If no datasets specified, evaluate on all local .npy files
        data_dir = Path(cfg.data.data_dir)
        datasets_eval = [f.stem for f in data_dir.glob("*.npy") if not f.stem.startswith('nixtla')]
        if datasets_eval:
            logger.info(f"Evaluating on all local datasets: {datasets_eval}")
    
    all_results = {}
    all_horizon_metrics = {}
    
    if datasets_eval:
        logger.info(f"\n📦 Local datasets: {datasets_eval}")
        
        # Evaluate each dataset
        for dataset_name in datasets_eval:
            print("\n" + "=" * 60)
            print(f"📈 Evaluating: {dataset_name}")
            print("=" * 60)
            
            data_path = Path(cfg.data.data_dir) / f"{dataset_name}.npy"
            
            if not data_path.exists():
                logger.warning(f"  ⚠️ Dataset not found: {data_path}, skipping...")
                continue
            
            try:
                # Create datamodule
                dm = MonashDataModule(
                    data_path=data_path,
                    context_length=cfg.model.seq_length,
                    prediction_length=cfg.model.prediction_length,
                    batch_size=cfg.data.batch_size,
                    stride=cfg.data.stride,
                    normalize_mode=cfg.data.normalize_mode,
                    normalizer_type=cfg.data.normalizer_type,
                    clip_outliers=cfg.data.clip_outliers,
                    clip_sigma=cfg.data.clip_sigma,
                    train_val_test_split=cfg.data.train_val_test_split,
                    num_workers=4,
                    seed=cfg.data.seed
                )
                dm.prepare_data()
                dm.setup()
                
                logger.info(f"  Test samples: {len(dm.test_dataset)}")
                
                # Evaluate on test set
                test_loader = dm.test_dataloader()
                results = evaluate_dataset(model, test_loader, device)

                # Model + baselines on identical windows
                season_length = get_seasonality(dataset_name)
                scored = evaluate_with_baselines(
                    results['contexts'],
                    results['predictions'],
                    results['targets'],
                    season_length,
                    quantiles=results.get('quantiles'),
                    quantile_levels=results.get('quantile_levels'),
                )
                metrics = dict(scored['timejepa'])
                metrics['_baselines'] = {
                    k: v for k, v in scored.items() if k not in ('timejepa', '_skill')
                }
                metrics.update(scored.get('_skill', {}))
                all_results[dataset_name] = metrics

                # Compute per-horizon metrics
                horizon_metrics = compute_per_horizon_metrics(
                    results['predictions'],
                    results['targets']
                )
                all_horizon_metrics[dataset_name] = horizon_metrics
                
                # Print metrics
                print(f"\n  📊 Results (seasonality m={season_length}):")
                print(f"     RMSE:        {metrics['rmse']:.4f}")
                print(f"     MAE:         {metrics['mae']:.4f}")
                print(f"     SMAPE:       {metrics['smape']:.2f}%")
                print(f"     MASE:        {metrics.get('mase', float('nan')):.4f}")
                print(f"     WQL/ND:      {metrics['wql']:.4f}")
                print(f"     R²:          {metrics['r2']:.4f}")
                print(f"     Correlation: {metrics['correlation']:.4f}")
                print(f"  🎯 vs baselines (MASE):")
                for bname, bm in metrics['_baselines'].items():
                    marker = ""
                    if metrics.get('mase') is not None and bm.get('mase') is not None:
                        marker = " ← beats us" if bm['mase'] < metrics['mase'] else ""
                    print(f"     {bname:<16} {bm.get('mase', float('nan')):.4f}{marker}")
                
                # Generate plots
                plots_dir = output_dir / "plots"
                
                plot_forecasts(
                    results['contexts'],
                    results['predictions'],
                    results['targets'],
                    dataset_name,
                    plots_dir,
                    num_samples=9,
                    quantiles=results.get('quantiles'),
                    quantile_levels=results.get('quantile_levels'),
                )
                
                plot_error_analysis(
                    results['predictions'],
                    results['targets'],
                    dataset_name,
                    plots_dir
                )
                
            except Exception as e:
                logger.error(f"  ❌ Error evaluating {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    
    print("\n" + "=" * 80)
    print("📋 EVALUATION SUMMARY")
    print("=" * 80)
    
    # Local dataset summary
    if all_results:
        # Split the nested baseline block out before building the frame
        flat = {
            ds: {k: v for k, v in m.items() if k != '_baselines'}
            for ds, m in all_results.items()
        }
        df = pd.DataFrame(flat).T
        df.index.name = 'Dataset'

        print("\n📊 Local Datasets:")
        print(df.to_string())

        # Head-to-head against baselines (MASE)
        h2h_rows = {}
        for ds, m in all_results.items():
            row = {'TimeJEPA': m.get('mase')}
            for bname, bm in m.get('_baselines', {}).items():
                row[bname] = bm.get('mase')
            h2h_rows[ds] = row
        h2h = pd.DataFrame(h2h_rows).T
        if not h2h.empty and h2h.notna().any().any():
            h2h['winner'] = h2h.idxmin(axis=1)
            print("\n🎯 Head-to-head (MASE, lower is better):")
            print(h2h.round(4).to_string())
            n_wins = int((h2h['winner'] == 'TimeJEPA').sum())
            print(f"\n  TimeJEPA is best on {n_wins}/{len(h2h)} local datasets.")
            h2h.to_csv(output_dir / 'local_head_to_head.csv')

        # Compute averages over numeric columns only
        print("\n" + "-" * 60)
        numeric = df.apply(pd.to_numeric, errors='coerce')
        avg_row = numeric.mean()
        print(f"{'AVERAGE':<20}", end="")
        for col in numeric.columns:
            if pd.notna(avg_row[col]):
                print(f" {col}: {avg_row[col]:.4f}", end=" |")
        print()

        # Save results
        results_json_path = output_dir / "local_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        results_csv_path = output_dir / "local_results.csv"
        df.to_csv(results_csv_path)
        
        # Save horizon metrics
        horizon_json_path = output_dir / "horizon_metrics.json"
        with open(horizon_json_path, 'w') as f:
            serializable = {
                ds: {str(h): m for h, m in hm.items()} 
                for ds, hm in all_horizon_metrics.items()
            }
            json.dump(serializable, f, indent=2)
        
        # Plot summary comparison
        plot_summary_comparison(all_results, output_dir / "plots")
    
    # Nixtla summary (already printed in create_nixtla_benchmark_table)
    if nixtla_results:
        print(f"\n📊 Nixtla benchmark results saved to:")
        print(f"   - {output_dir / 'nixtla_mse.csv'}")
        print(f"   - {output_dir / 'nixtla_mae.csv'}")
    
    if not all_results and not nixtla_results:
        logger.error("No datasets were successfully evaluated!")
        return
    
    # Save config used for evaluation
    config_path = output_dir / "eval_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  📁 Results saved to: {output_dir}")
    if all_results:
        print(f"     - local_results.json / local_results.csv")
        print(f"     - horizon_metrics.json")
    if nixtla_results:
        print(f"     - nixtla_results.json")
        print(f"     - nixtla_mse.csv / nixtla_mae.csv")
    print(f"     - plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()