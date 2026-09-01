"""
Metrics for JEPA pretraining and finetuning.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# ANTI-COLLAPSE REGULARIZERS
#
# Two options, both selectable from config so they can be ablated against each
# other:
#   - vicreg_loss : variance/covariance hinge (Bardes et al.)
#   - sigreg_loss : Sketched Isotropic Gaussian Regularization (LeJEPA,
#                   Balestriero & LeCun, arXiv 2511.08544)
# =============================================================================


def _subsample_tokens(
    x: torch.Tensor,
    max_tokens: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Flatten to [N, D] and optionally subsample N rows to bound memory."""
    x = x.reshape(-1, x.shape[-1])
    if max_tokens is not None and x.shape[0] > max_tokens:
        idx = torch.randperm(x.shape[0], device=x.device, generator=generator)[:max_tokens]
        x = x[idx]
    return x


def sigreg_loss(
    embeddings: torch.Tensor,
    num_projections: int = 16,
    num_quadrature: int = 33,
    t_max: float = 5.0,
    max_tokens: Optional[int] = 8192,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).

    LeJEPA (Balestriero & LeCun, arXiv 2511.08544) proves the isotropic Gaussian
    is the embedding distribution that minimises downstream prediction risk, and
    enforces it with univariate goodness-of-fit tests on random 1D projections
    (Cramer-Wold: two distributions coincide iff all their 1D projections do).

        SIGReg(z) = (1/M) sum_m  EP( {a_m^T z_i}_i )

    where EP is the Epps-Pulley statistic: the weighted L2 distance between the
    empirical characteristic function of the projection and that of N(0, 1),

        EP = 2 * int_0^{t_max} [ (Re phi_n(t) - e^{-t^2/2})^2 + (Im phi_n(t))^2 ]
                               * e^{-t^2/2} dt

    (the integrand is even in t, hence integrating over the half-line and
    doubling). Evaluated by trapezoidal quadrature - no density estimation, no
    custom kernels, just GEMM and complex exponentials.

    Why this over VICReg
    --------------------
    VICReg's variance hinge only asks each coordinate to exceed a threshold, and
    its covariance term only decorrelates second moments; a distribution can
    satisfy both and still be badly non-Gaussian (bimodal, heavy-tailed).
    SIGReg constrains the whole distribution with a single hyperparameter, and
    the paper reports the objective correlating (r ~ 0.8) with downstream
    accuracy - useful here because our val_loss is measured on held-out *series*
    and tracks benchmark performance poorly.

    Directions are resampled every call: over many steps this covers the sphere,
    which is the "sketched" part.

    Args:
        embeddings: [..., D]. Flattened to [N, D].
        num_projections: M random directions (8-16 is enough per the paper).
        num_quadrature: Q trapezoid nodes on [0, t_max].
        t_max: Upper integration limit. The Gaussian weight makes the integrand
            negligible past ~5.
        max_tokens: Subsample N to bound the O(Q*N*M) intermediate. With
            B=512 and 47 patches, N would be 24k; 8192 keeps this cheap and the
            estimator is still low-variance.
        generator: Optional RNG for reproducible projections/subsampling.

    Returns:
        Dict with 'loss' plus diagnostics.
    """
    z = _subsample_tokens(embeddings, max_tokens, generator)
    n, d = z.shape

    # Random unit directions [D, M]
    directions = torch.randn(d, num_projections, device=z.device, dtype=z.dtype,
                             generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)

    u = z @ directions                                    # [N, M]

    t = torch.linspace(0.0, t_max, num_quadrature, device=z.device, dtype=z.dtype)
    tu = t.view(-1, 1, 1) * u.unsqueeze(0)                # [Q, N, M]

    re = torch.cos(tu).mean(dim=1)                        # [Q, M]
    im = torch.sin(tu).mean(dim=1)                        # [Q, M]

    gauss_cf = torch.exp(-0.5 * t.pow(2)).view(-1, 1)     # [Q, 1]
    weight = torch.exp(-0.5 * t.pow(2)).view(-1, 1)       # [Q, 1]

    integrand = ((re - gauss_cf).pow(2) + im.pow(2)) * weight     # [Q, M]

    # Trapezoidal rule over t, doubled for the negative half-line
    dt = t_max / max(num_quadrature - 1, 1)
    ep = 2.0 * dt * (integrand.sum(dim=0) - 0.5 * (integrand[0] + integrand[-1]))  # [M]

    loss = ep.mean()

    with torch.no_grad():
        proj_mean = u.mean(dim=0).abs().mean()
        proj_std = u.std(dim=0).mean()

    return {
        'loss': loss,
        'ep_mean': loss.detach(),
        'ep_max': ep.detach().max(),
        'proj_abs_mean': proj_mean,
        'proj_std': proj_std,
    }


def vicreg_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    per_position: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    VICReg loss: Variance-Invariance-Covariance regularization.

    Two corrections vs the original implementation here
    ---------------------------------------------------
    1. The variance/covariance statistics used to be computed on
       `reshape(-1, D)`, i.e. pooling the batch AND the patch-position axes.
       Different patch positions naturally differ, so positional diversity alone
       could satisfy the variance hinge while representations collapsed *at a
       fixed position* - precisely the failure mode observed on noisy series.
       With `per_position=True` the statistics are computed across the batch for
       each patch position separately, then averaged, so the hinge measures what
       it is supposed to measure.

    2. The `targets` half of the variance and covariance terms was computed on a
       detached tensor (targets come from the EMA encoder under no_grad), so it
       contributed exactly zero gradient - half the regularizer was decorative
       while still inflating the reported loss. The target terms are now
       computed under no_grad and returned as diagnostics only; only the
       predictions side enters the loss.

    Args:
        predictions: [B, N, D]
        targets: [B, N, D] (detached)
        invariance_weight: Weight for MSE term
        variance_weight: Weight for variance term
        covariance_weight: Weight for covariance term
        variance_target: Minimum std target (hinge threshold)
        per_position: Compute variance/covariance per patch position (default,
            correct) instead of pooling batch and position.

    Returns:
        Dictionary with 'loss' tensor and component values
    """
    def _variance_hinge(x: torch.Tensor) -> torch.Tensor:
        if per_position:
            # std across batch, for each (position, feature) -> [N, D]
            std = x.std(dim=0)
        else:
            std = x.reshape(-1, x.shape[-1]).std(dim=0)
        return torch.relu(variance_target - std).mean()

    def _off_diagonal(cov: torch.Tensor) -> torch.Tensor:
        d = cov.shape[-1]
        off = cov.pow(2).sum(dim=(-2, -1)) - cov.diagonal(dim1=-2, dim2=-1).pow(2).sum(-1)
        return (off / (d * (d - 1))).mean()

    def _covariance(x: torch.Tensor) -> torch.Tensor:
        if per_position:
            # [B, N, D] -> per-position covariance [N, D, D]
            centered = x - x.mean(dim=0, keepdim=True)
            b = centered.shape[0]
            cov = torch.einsum('bnd,bne->nde', centered, centered) / max(b - 1, 1)
        else:
            flat = x.reshape(-1, x.shape[-1])
            centered = flat - flat.mean(dim=0)
            cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
        return _off_diagonal(cov)

    # === Invariance (MSE) ===
    inv_loss = F.mse_loss(predictions, targets)

    # === Variance / Covariance - predictions only (targets carry no gradient) ===
    var_loss = _variance_hinge(predictions)
    cov_loss = _covariance(predictions)

    total = (invariance_weight * inv_loss +
             variance_weight * var_loss +
             covariance_weight * cov_loss)

    with torch.no_grad():
        tgt_var_loss = _variance_hinge(targets)
        pred_std_mean = predictions.std(dim=0).mean() if per_position \
            else predictions.reshape(-1, predictions.shape[-1]).std(dim=0).mean()
        tgt_std_mean = targets.std(dim=0).mean() if per_position \
            else targets.reshape(-1, targets.shape[-1]).std(dim=0).mean()

    return {
        'loss': total,
        'invariance': inv_loss,
        'variance': var_loss,
        'covariance': cov_loss,
        'variance_target_side': tgt_var_loss,   # diagnostic only, no gradient
        'pred_std_mean': pred_std_mean,
        'target_std_mean': tgt_std_mean,
    }


def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'mse',
    reduction: str = 'mean',
    vicreg_weights: Optional[Dict[str, float]] = None,
    context_embeddings: Optional[torch.Tensor] = None,
    sigreg_config: Optional[Dict[str, float]] = None,
    return_components: bool = False,
):
    """
    JEPA loss: measure similarity between predicted and target representations.

    Regularizing the right tensor
    -----------------------------
    Anti-collapse used to be applied only to the *predictor output*. The
    representation we actually care about downstream is the online encoder's
    output, and it was never constrained directly. Pass `context_embeddings` to
    regularize it - this is the tensor a linear probe or the forecasting decoder
    consumes.

    Args:
        predictions: Predicted representations [B, N_target, D]
        targets: Target representations [B, N_target, D] (detached)
        loss_type: 'mse' | 'smooth_l1' | 'cosine' | 'vicreg' | 'sigreg'
        reduction: 'mean', 'sum', or 'none'
        vicreg_weights: Dict with 'invariance', 'variance', 'covariance' weights
        context_embeddings: Online encoder output [B, N_ctx, D]. When provided,
            the regularizer is applied to it in addition to the predictions.
        sigreg_config: Dict for loss_type='sigreg' with keys
            'lambda' (regularization weight, the single hyperparameter),
            'num_projections', 'num_quadrature', 't_max', 'max_tokens'.
        return_components: If True, return (loss, components dict).

    Returns:
        Loss scalar, or (loss, components) when return_components=True.
    """
    components: Dict[str, torch.Tensor] = {}

    if loss_type == "sigreg":
        cfg = sigreg_config or {}
        lam = float(cfg.get('lambda', 1.0))
        kwargs = dict(
            num_projections=int(cfg.get('num_projections', 16)),
            num_quadrature=int(cfg.get('num_quadrature', 33)),
            t_max=float(cfg.get('t_max', 5.0)),
            max_tokens=cfg.get('max_tokens', 8192),
        )

        # L_total = L_JEPA + lambda * SIGReg   (LeJEPA eq. 1)
        inv_loss = F.mse_loss(predictions, targets)
        components['invariance'] = inv_loss

        reg_on = cfg.get('apply_to', 'context')
        reg_terms = []
        if reg_on in ('context', 'both') and context_embeddings is not None:
            r = sigreg_loss(context_embeddings, **kwargs)
            components['sigreg_context'] = r['loss']
            components['sigreg_context_proj_std'] = r['proj_std']
            reg_terms.append(r['loss'])
        if reg_on in ('predictions', 'both'):
            r = sigreg_loss(predictions, **kwargs)
            components['sigreg_predictions'] = r['loss']
            reg_terms.append(r['loss'])

        if not reg_terms:
            # Nothing to regularize (context requested but not supplied)
            total = inv_loss
        else:
            reg = torch.stack(reg_terms).mean()
            components['sigreg'] = reg
            total = inv_loss + lam * reg

        components['loss'] = total
        return (total, components) if return_components else total

    if loss_type == "vicreg":
        weights = vicreg_weights or {}
        result = vicreg_loss(
            predictions, targets,
            invariance_weight=weights.get('invariance', 25.0),
            variance_weight=weights.get('variance', 25.0),
            covariance_weight=weights.get('covariance', 1.0)
        )
        components.update(result)
        total = result['loss']

        # Also constrain the encoder output, not just the predictor output.
        if context_embeddings is not None:
            ctx_var = torch.relu(1.0 - context_embeddings.std(dim=0)).mean()
            centered = context_embeddings - context_embeddings.mean(dim=0, keepdim=True)
            b = centered.shape[0]
            cov = torch.einsum('bnd,bne->nde', centered, centered) / max(b - 1, 1)
            d = cov.shape[-1]
            off = cov.pow(2).sum(dim=(-2, -1)) - cov.diagonal(dim1=-2, dim2=-1).pow(2).sum(-1)
            ctx_cov = (off / (d * (d - 1))).mean()

            components['context_variance'] = ctx_var
            components['context_covariance'] = ctx_cov
            total = total + (weights.get('variance', 25.0) * ctx_var
                             + weights.get('covariance', 1.0) * ctx_cov)

        components['loss'] = total
        return (total, components) if return_components else total

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

    components['loss'] = loss
    return (loss, components) if return_components else loss


# No external call sites; kept per the no-delete policy.
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


# No external call sites; kept per the no-delete policy.
def representation_std(embeddings: torch.Tensor) -> torch.Tensor:
    """Standard deviation of representations."""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    std = embeddings.std(dim=0).mean()
    return std


# No external call sites; kept per the no-delete policy.
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
    
    # VICReg components (for monitoring even when not used in the loss)
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

# =============================================================================
# ADDITIONAL EVALUATION METRICS
# =============================================================================

def huber(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber Loss (smooth L1)."""
    return F.huber_loss(predictions, targets, reduction='mean', delta=delta)


# No external call sites; kept per the no-delete policy.
def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """R^2 (coefficient of determination)."""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def correlation(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient."""
    preds_flat = predictions.flatten()
    targs_flat = targets.flatten()
    
    corr_matrix = torch.corrcoef(torch.stack([preds_flat, targs_flat]))
    corr_val = corr_matrix[0, 1]
    
    # Handle NaN (constant predictions/targets)
    if torch.isnan(corr_val):
        return torch.tensor(0.0, device=predictions.device)
    return corr_val


# =============================================================================
# SCALE-FREE / BENCHMARK METRICS
#
# MSE and MAE are only comparable within a dataset, and only when the target
# variance is known. They are useless for cross-dataset aggregation and cannot
# be compared to published numbers. The metrics below are the ones GIFT-Eval
# and the Monash archive actually rank on.
# =============================================================================

def mase(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    context: torch.Tensor,
    season_length: int = 1,
    eps: float = 1e-8,
    aggregate: str = "pooled",
) -> torch.Tensor:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler).

        MASE = mean(|y - y_hat|) / mean(|y_t - y_{t-m}|  over the history)

    The denominator is the in-sample seasonal-naive MAE computed on the
    *context*, so MASE ~= 1.0 means "as good as seasonal naive". This is the
    only forecasting metric here that is directly comparable across datasets
    and against the GIFT-Eval leaderboard.

    Aggregation
    -----------
    `aggregate='pooled'` (default) computes a ratio of sums:

        sum_i sum_t |err|  /  sum_i sum_t |seasonal diff|

    `aggregate='per_series'` computes the classic mean of per-window ratios.

    Pooled is the default because the per-window form is numerically unstable on
    real data: a window whose seasonal difference is ~0 (flat or constant
    segments, which ETTm2 and electricity both contain) produces a ratio of
    ~1/eps and single-handedly dominates the average. Observed in practice -
    per-window MASE on ETTm2 returned ~1e4 for every model AND for seasonal
    naive itself, which is obviously meaningless. Pooling makes those windows
    contribute proportionally to their actual error instead of exploding.

    `per_series` additionally drops degenerate windows (scale <= eps) rather
    than clamping them, so it stays interpretable if you do want per-window
    averaging.

    Args:
        predictions: [B, H] or [B, H, C]
        targets:     [B, H] or [B, H, C]
        context:     [B, L] or [B, L, C] - the history the forecast was made from
        season_length: Seasonal period m (see utils.baselines.get_seasonality)
        aggregate: 'pooled' | 'per_series'

    Returns:
        Scalar MASE.
    """
    m = max(1, int(season_length))

    if context.shape[1] <= m:
        # Not enough history for a seasonal difference: fall back to m=1.
        m = 1
    if context.shape[1] <= 1:
        return mae(predictions, targets)

    seasonal_diff = (context[:, m:, ...] - context[:, :-m, ...]).abs()
    abs_err = (predictions - targets).abs()

    if aggregate == "pooled":
        num = abs_err.mean()
        den = seasonal_diff.mean().clamp_min(eps)
        return num / den

    if aggregate != "per_series":
        raise ValueError(f"Unknown aggregate: {aggregate}")

    # Per-window ratios, dropping windows with a degenerate scale.
    scale = seasonal_diff.reshape(seasonal_diff.shape[0], -1).mean(dim=1)   # [B]
    err = abs_err.reshape(abs_err.shape[0], -1).mean(dim=1)                 # [B]

    valid = scale > eps
    if not bool(valid.any()):
        # Every window is flat: seasonal naive is a perfect forecast there, so
        # a scaled error is undefined. Report unscaled MAE rather than 1/eps.
        return err.mean()

    return (err[valid] / scale[valid]).mean()


def nd(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Deviation: sum|y - y_hat| / sum|y|.

    Scale-free, aggregates across series. For a *point* forecast this is
    numerically identical to the weighted quantile loss (see `weighted_quantile_loss`),
    which makes it the right stand-in for CRPS until a probabilistic head exists.
    """
    num = (predictions - targets).abs().sum()
    den = targets.abs().sum().clamp_min(eps)
    return num / den


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """
    Pinball loss at quantile level q (summed, not averaged).

        QL_q(y, y_hat) = 2 * [ q*(y - y_hat)^+  +  (1-q)*(y_hat - y)^+ ]

    The factor 2 is the GluonTS/GIFT-Eval convention, which makes the average
    over a symmetric quantile grid reduce exactly to the absolute error.
    """
    diff = targets - predictions
    return 2.0 * torch.maximum(q * diff, (q - 1.0) * diff).sum()


def weighted_quantile_loss(
    quantile_predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: Optional[list] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted Quantile Loss (WQL) - the CRPS proxy GIFT-Eval ranks on.

        WQL = mean_q [ sum_i QL_q(y_i, y_hat_{q,i}) ] / sum_i |y_i|

    Args:
        quantile_predictions: [Q, B, H, ...] - one forecast per quantile level,
            OR [B, H, ...] for a point forecast (broadcast to every level).
        targets: [B, H, ...]
        quantile_levels: list of Q levels, default = [0.1, 0.2, ..., 0.9]

    Note:
        With a point forecast the result collapses to `nd()` exactly. That is
        the honest score for a deterministic model - it is what TimeJEPA would
        get on GIFT-Eval today, and it is why a quantile head (P2.1) is a
        prerequisite for competing there rather than a nice-to-have.
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if quantile_predictions.ndim == targets.ndim:
        # Point forecast: same prediction for every quantile level.
        quantile_predictions = quantile_predictions.unsqueeze(0).expand(
            len(quantile_levels), *quantile_predictions.shape
        )

    denom = targets.abs().sum().clamp_min(eps)
    total = torch.zeros((), device=targets.device, dtype=targets.dtype)
    for i, q in enumerate(quantile_levels):
        total = total + quantile_loss(quantile_predictions[i], targets, q)

    return total / (len(quantile_levels) * denom)


def compute_forecasting_metrics_extended(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    season_length: int = 1,
) -> Dict[str, float]:
    """
    Compute comprehensive forecasting metrics for evaluation.

    Args:
        predictions: [B, L] or [B, L, C]
        targets: [B, L] or [B, L, C]
        context: [B, L_ctx] or [B, L_ctx, C]. Required for MASE; when omitted
            the MASE/'skill' keys are simply absent from the result.
        season_length: Seasonal period for MASE scaling.

    Returns:
        Dictionary of metrics
    """
    # Flatten for global metrics
    preds = predictions.float()
    targs = targets.float()

    metrics = {
        'mse': mse(preds, targs).item(),
        'mae': mae(preds, targs).item(),
        'rmse': rmse(preds, targs).item(),
        'mape': mape(preds, targs).item(),
        'smape': smape(preds, targs).item(),
        'huber': huber(preds, targs).item(),
        'r2': r2_score(preds, targs).item(),
        'correlation': correlation(preds, targs).item(),
        # Scale-free / benchmark-comparable
        'nd': nd(preds, targs).item(),
        'wql': weighted_quantile_loss(preds, targs).item(),
    }

    if context is not None:
        metrics['mase'] = mase(preds, targs, context.float(), season_length).item()

    return metrics


def compute_per_horizon_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics per prediction horizon step.
    
    Args:
        predictions: [B, pred_len] or [B, pred_len, C]
        targets: [B, pred_len] or [B, pred_len, C]
    
    Returns:
        Dictionary mapping horizon index to metrics dict
    """
    pred_len = predictions.shape[1]
    horizon_metrics = {}
    
    for h in range(pred_len):
        pred_h = predictions[:, h]
        targ_h = targets[:, h]
        horizon_metrics[h] = {
            'mae': mae(pred_h, targ_h).item(),
            'rmse': rmse(pred_h, targ_h).item(),
            'smape': smape(pred_h, targ_h).item(),
        }
    
    return horizon_metrics