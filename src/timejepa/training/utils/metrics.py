# src/timejepa/training/utils/metrics.py
"""
Metrics for JEPA pretraining and finetuning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'mse',
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    JEPA loss: measure similarity between predicted and target representations.
    
    Args:
        predictions: Predicted representations [B, N_target, D]
        targets: Target representations [B, N_target, D] (detached)
        loss_type: 'mse', 'smooth_l1', or 'cosine'
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Loss scalar (or per-sample if reduction='none')
    """
    if loss_type == 'mse':
        loss = F.mse_loss(predictions, targets, reduction=reduction)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(predictions, targets, reduction=reduction)
    elif loss_type == 'cosine':
        # Cosine embedding loss (1 - cosine_similarity)
        # Normalize vectors
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        targ_norm = F.normalize(targets, p=2, dim=-1)
        # Cosine similarity
        cosine_sim = (pred_norm * targ_norm).sum(dim=-1)  # [B, N_target]
        # Convert to loss (1 - similarity)
        loss = 1 - cosine_sim
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss


def representation_variance(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Measure variance of representations (to detect collapse).
    
    Args:
        embeddings: [B, N, D] or [B, D]
    
    Returns:
        Variance scalar
    """
    if embeddings.ndim == 3:
        # Flatten batch and sequence dims
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    # Variance along feature dimension
    variance = embeddings.var(dim=0).mean()
    return variance


def representation_std(embeddings: torch.Tensor) -> torch.Tensor:
    """Standard deviation of representations."""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    std = embeddings.std(dim=0).mean()
    return std


def cosine_similarity_matrix(
    embeddings1: torch.Tensor,
    embeddings2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity.
    
    Args:
        embeddings1: [B, D]
        embeddings2: [B, D] or None (uses embeddings1)
    
    Returns:
        Similarity matrix [B, B]
    """
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    # Normalize
    emb1_norm = F.normalize(embeddings1, p=2, dim=-1)
    emb2_norm = F.normalize(embeddings2, p=2, dim=-1)
    
    # Similarity matrix
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return similarity


def compute_pretrain_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    context_embeddings: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all pretrain metrics.
    
    Args:
        predictions: [B, N_target, D]
        targets: [B, N_target, D]
        context_embeddings: [B, N_context, D] (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Main loss
    metrics['loss/mse'] = jepa_loss(predictions, targets, loss_type='mse').item()
    
    # Representation statistics (to monitor collapse)
    metrics['repr/pred_std'] = representation_std(predictions).item()
    metrics['repr/target_std'] = representation_std(targets).item()
    metrics['repr/pred_var'] = representation_variance(predictions).item()
    metrics['repr/target_var'] = representation_variance(targets).item()
    
    if context_embeddings is not None:
        metrics['repr/context_std'] = representation_std(context_embeddings).item()
        metrics['repr/context_var'] = representation_variance(context_embeddings).item()
    
    # Mean absolute error (alternative metric)
    mae = F.l1_loss(predictions, targets).item()
    metrics['loss/mae'] = mae
    
    # Cosine similarity between predictions and targets
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    targ_flat = targets.reshape(-1, targets.shape[-1])
    pred_norm = F.normalize(pred_flat, p=2, dim=-1)
    targ_norm = F.normalize(targ_flat, p=2, dim=-1)
    cosine_sim = (pred_norm * targ_norm).sum(dim=-1).mean().item()
    metrics['similarity/cosine'] = cosine_sim
    
    return metrics


# Forecasting metrics (for Phase 6)

def mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return F.mse_loss(predictions, targets)


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return F.l1_loss(predictions, targets)


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(mse(predictions, targets))


def mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100


def smape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets)) / 2 + epsilon
    return torch.mean(numerator / denominator) * 100


def compute_forecasting_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all forecasting metrics.
    
    Args:
        predictions: [B, L, C]
        targets: [B, L, C]
    
    Returns:
        Dictionary of metrics
    """
    return {
        'mse': mse(predictions, targets).item(),
        'mae': mae(predictions, targets).item(),
        'rmse': rmse(predictions, targets).item(),
        'mape': mape(predictions, targets).item(),
        'smape': smape(predictions, targets).item(),
    }