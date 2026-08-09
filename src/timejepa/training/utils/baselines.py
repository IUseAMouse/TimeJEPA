"""
Reference forecasting baselines.

Without these, an absolute MSE/MAE number is uninterpretable: you cannot tell
whether R^2 = 0.74 on electricity is a good result or worse than repeating last
week's values. Every TimeJEPA evaluation should report these side by side.

All baselines share the same signature so they can be swapped in an eval loop:

    forecast = baseline(context, horizon, season_length)

    context: [B, L] or [B, L, C]  (history, same normalization space as targets)
    returns: [B, horizon] or [B, horizon, C]
"""

from typing import Dict, Optional

import torch


# =============================================================================
# SEASONALITY
# =============================================================================

# Seasonal period per pandas-style frequency string (GluonTS convention).
FREQ_TO_SEASONALITY: Dict[str, int] = {
    "S": 3600,      # secondly
    "T": 1440,      # minutely
    "min": 1440,
    "10min": 144,
    "15min": 96,
    "15T": 96,
    "30min": 48,
    "H": 24,        # hourly
    "1h": 24,
    "D": 7,         # daily
    "1d": 7,
    "W": 1,         # weekly
    "1w": 1,
    "M": 12,        # monthly
    "Q": 4,         # quarterly
    "Y": 1,
    "A": 1,
}

# Per-dataset overrides. Exchange rates are close to a random walk: the
# frequency-derived m=7 would make the seasonal-naive denominator meaningless,
# so MASE for those uses m=1 (i.e. a plain random-walk scaling), which is the
# convention in the long-horizon literature.
DATASET_SEASONALITY: Dict[str, int] = {
    "ettm1": 96,        # 15min -> daily cycle
    "ettm2": 96,
    "etth1": 24,        # hourly -> daily cycle
    "etth2": 24,
    "electricity": 24,
    "ecl": 24,
    "traffic": 24,
    "weather": 144,     # 10min -> daily cycle
    "exchange": 1,      # random walk
    "ili": 1,           # weekly, no reliable short cycle
}


def get_seasonality(dataset_name: Optional[str] = None, freq: Optional[str] = None) -> int:
    """
    Resolve the seasonal period for a dataset.

    Dataset-specific overrides win over frequency-derived defaults.
    Falls back to 1 (random-walk scaling) when nothing matches.
    """
    if dataset_name is not None:
        key = dataset_name.lower()
        if key in DATASET_SEASONALITY:
            return DATASET_SEASONALITY[key]

    if freq is not None:
        if freq in FREQ_TO_SEASONALITY:
            return FREQ_TO_SEASONALITY[freq]
        if freq.lower() in FREQ_TO_SEASONALITY:
            return FREQ_TO_SEASONALITY[freq.lower()]

    return 1


# =============================================================================
# BASELINES
# =============================================================================

def seasonal_naive_forecast(
    context: torch.Tensor,
    horizon: int,
    season_length: int = 1,
) -> torch.Tensor:
    """
    Seasonal naive: repeat the last full seasonal cycle observed in the context.

    y_hat[t] = y[L - m + ((t) mod m)]

    This is the reference baseline of both the Monash archive and GIFT-Eval
    (where it is normalized to MASE = CRPS = 1.0 by construction).

    Args:
        context: [B, L] or [B, L, C]
        horizon: Number of steps to forecast
        season_length: Seasonal period m. m=1 degenerates to last-value.

    Returns:
        [B, horizon] or [B, horizon, C]
    """
    seq_len = context.shape[1]
    m = max(1, int(season_length))

    # Not enough history for a full cycle -> fall back to last value.
    if seq_len < m:
        return last_value_forecast(context, horizon)

    last_cycle = context[:, seq_len - m:, ...]              # [B, m, ...]
    repeats = (horizon + m - 1) // m
    tiled = last_cycle.repeat(
        *([1, repeats] + [1] * (context.ndim - 2))
    )                                                        # [B, repeats*m, ...]
    return tiled[:, :horizon, ...]


def last_value_forecast(context: torch.Tensor, horizon: int, **kwargs) -> torch.Tensor:
    """
    Naive / random-walk: repeat the last observed value.

    This is the optimal point forecast under a random walk, and is the honest
    bar to clear on financial-style series (bitcoin, exchange).
    """
    last = context[:, -1:, ...]
    return last.repeat(*([1, horizon] + [1] * (context.ndim - 2)))


def mean_forecast(context: torch.Tensor, horizon: int, **kwargs) -> torch.Tensor:
    """
    Context mean: the degenerate forecast an MSE-trained model collapses to.

    Useful as a *collapse detector*: if TimeJEPA does not beat this, it has
    learned nothing conditional on the context shape.
    """
    mean = context.mean(dim=1, keepdim=True)
    return mean.repeat(*([1, horizon] + [1] * (context.ndim - 2)))


def linear_trend_forecast(context: torch.Tensor, horizon: int, **kwargs) -> torch.Tensor:
    """
    Least-squares linear extrapolation of the context.

    Catches the case where a model is only learning "keep going in the same
    direction" and nothing more.
    """
    squeeze_back = context.ndim == 2
    x = context.unsqueeze(-1) if squeeze_back else context   # [B, L, C]

    b, seq_len, _ = x.shape
    t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    t_mean = t.mean()
    t_centered = t - t_mean

    y_mean = x.mean(dim=1, keepdim=True)                     # [B, 1, C]
    num = (t_centered.view(1, seq_len, 1) * (x - y_mean)).sum(dim=1, keepdim=True)
    den = (t_centered ** 2).sum().clamp_min(1e-8)
    slope = num / den                                        # [B, 1, C]
    intercept = y_mean - slope * t_mean

    t_future = torch.arange(
        seq_len, seq_len + horizon, device=x.device, dtype=x.dtype
    ).view(1, horizon, 1)
    out = intercept + slope * t_future                       # [B, horizon, C]

    return out.squeeze(-1) if squeeze_back else out


BASELINES = {
    "seasonal_naive": seasonal_naive_forecast,
    "naive_last": last_value_forecast,
    "context_mean": mean_forecast,
    "linear_trend": linear_trend_forecast,
}


def compute_all_baselines(
    context: torch.Tensor,
    horizon: int,
    season_length: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Run every baseline on the same context.

    Returns:
        Dict mapping baseline name -> forecast tensor of shape [B, horizon, ...]
    """
    return {
        "seasonal_naive": seasonal_naive_forecast(context, horizon, season_length),
        "naive_last": last_value_forecast(context, horizon),
        "context_mean": mean_forecast(context, horizon),
        "linear_trend": linear_trend_forecast(context, horizon),
    }
