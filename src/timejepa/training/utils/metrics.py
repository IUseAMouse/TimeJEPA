# src/timejepa/training/utils/metrics.py
"""
Metrics for JEPA pretraining and finetuning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def vicreg_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 30.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    VICReg loss: Variance-Invariance-Covariance regularization.
    
    Args:
        predictions: [B, N_target, D]
        targets: [B, N_target, D]
        invariance_weight: Weight for MSE term
        variance_weight: Weight for variance term
        covariance_weight: Weight for covariance term
        variance_target: Minimum std target (hinge threshold)
        eps: Small constant for numerical stability
    
    Returns:
        Dictionary with 'loss' tensor and component values
    """
    # Flatten to [B*N, D]
    pred_flat = predictions.reshape(-1, predictions.size(-1))
    tgt_flat = targets.reshape(-1, targets.size(-1))
    
    # === Invariance (MSE) ===
    inv_loss = F.mse_loss(predictions, targets)
    
    # === Variance ===
    pred_std = pred_flat.std(dim=0)  # [D]
    tgt_std = tgt_flat.std(dim=0)
    
    # Hinge: pénalise si std < target
    var_loss_pred = torch.relu(variance_target - pred_std).mean()
    var_loss_tgt = torch.relu(variance_target - tgt_std).mean()
    var_loss = (var_loss_pred + var_loss_tgt) / 2
    
    # === Covariance ===
    pred_centered = pred_flat - pred_flat.mean(dim=0)
    tgt_centered = tgt_flat - tgt_flat.mean(dim=0)
    
    batch_size = pred_flat.size(0)
    
    cov_pred = (pred_centered.T @ pred_centered) / (batch_size - 1)
    cov_tgt = (tgt_centered.T @ tgt_centered) / (batch_size - 1)
    
    def off_diagonal_loss(cov_matrix: torch.Tensor) -> torch.Tensor:
        d = cov_matrix.size(0)
        off_diag = cov_matrix.pow(2).sum() - cov_matrix.diagonal().pow(2).sum()
        return off_diag / (d * (d - 1))
    
    cov_loss = (off_diagonal_loss(cov_pred) + off_diagonal_loss(cov_tgt)) / 2
    
    # === Total ===
    total = (invariance_weight * inv_loss + 
             variance_weight * var_loss + 
             covariance_weight * cov_loss)
    
    return {
        'loss': total,
        'invariance': inv_loss,
        'variance': var_loss,
        'covariance': cov_loss,
        'pred_std_mean': pred_std.mean(),
        'target_std_mean': tgt_std.mean()
    }


def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'mse',
    reduction: str = 'mean',
    vicreg_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    JEPA loss: measure similarity between predicted and target representations.
    
    Args:
        predictions: Predicted representations [B, N_target, D]
        targets: Target representations [B, N_target, D] (detached)
        loss_type: 'mse', 'smooth_l1', or 'cosine'
        reduction: 'mean', 'sum', or 'none'
        use_vicreg: If True, use VICReg loss instead
        vicreg_weights: Dict with 'invariance', 'variance', 'covariance' weights
    
    Returns:
        Loss scalar (or per-sample if reduction='none')
    """
    if loss_type == "vicreg":
        weights = vicreg_weights or {}
        result = vicreg_loss(
            predictions, targets,
            invariance_weight=weights.get('invariance', 25.0),
            variance_weight=weights.get('variance', 35.0),
            covariance_weight=weights.get('covariance', 1.0)
        )
        return result['loss']
    
    if loss_type == 'mse':
        loss = F.mse_loss(predictions, targets, reduction=reduction)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(predictions, targets, reduction=reduction)
    elif loss_type == 'cosine':
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        targ_norm = F.normalize(targets, p=2, dim=-1)
        cosine_sim = (pred_norm * targ_norm).sum(dim=-1)
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
    Compute all pretrain metrics including VICReg components.
    """
    metrics = {}
    
    # Main loss
    metrics['loss/mse'] = jepa_loss(predictions, targets, loss_type='mse').item()
    
    # VICReg components (pour monitoring même si pas utilisé dans la loss)
    vicreg_result = vicreg_loss(predictions, targets)
    metrics['vicreg/invariance'] = vicreg_result['invariance'].item()
    metrics['vicreg/variance'] = vicreg_result['variance'].item()
    metrics['vicreg/covariance'] = vicreg_result['covariance'].item()
    metrics['vicreg/total'] = vicreg_result['loss'].item()
    
    # Representation statistics
    metrics['repr/pred_std'] = representation_std(predictions).item()
    metrics['repr/target_std'] = representation_std(targets).item()
    metrics['repr/pred_var'] = representation_variance(predictions).item()
    metrics['repr/target_var'] = representation_variance(targets).item()
    
    if context_embeddings is not None:
        metrics['repr/context_std'] = representation_std(context_embeddings).item()
        metrics['repr/context_var'] = representation_variance(context_embeddings).item()
    
    # MAE
    mae = F.l1_loss(predictions, targets).item()
    metrics['loss/mae'] = mae
    
    # Cosine similarity
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


def mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100


def smape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
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