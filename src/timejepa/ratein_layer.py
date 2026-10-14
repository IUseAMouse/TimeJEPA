"""
RateIN as a layer: a learned, continuous resampling rate in front of a frozen
forecaster.

The published RateIN picks the rate k per configuration by a backtest over a
grid (20 to 40 forecasts per series). This module replaces the search by a
function of the context, trained once with the forecaster frozen:

    log k = log k_prior + g(x)          g: a small diagonal-SSM filterbank
    y     = resample(x, k)              fixed model length, differentiable in k
    fan   = forecaster(y, H_m)          frozen, any model
    out   = to_native(fan, k, h)        back to the native grid

The resampling generalizes the integer operations of `evaluation.ratein` to
real k and reduces to them exactly when k is an integer (k > 1: right-aligned
block means, `decimate`; k = 1/m: interpolation between block centres,
`upsample`) and on the way back (`reinterp_fan`, `pool_fan`). Everything is
batched: a batch of B series with B different rates is one tensor, there is
no bucketing by k. Gradients reach k through the fractional sampling
positions and the block width.

Nothing here is used unless a layer is constructed; the harness and the
numpy primitives are untouched.
"""

import math
import os
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

K_MIN, K_MAX = 0.25, 48.0          # the grid of the backtest: 1/4 .. 48
MODEL_LEN = 1024                   # blocks handed to the forecaster
EPS = 1e-6


# ------------------------------------------------------------------ primitives
def _interp_cumsum(S: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """S [B, L+1] cumulative sums (S[0] = 0), p [B, M] positions in [0, L]:
    linear interpolation of S, i.e. the integral of a piecewise-constant
    signal up to a real position. Differentiable in p."""
    L = S.shape[1] - 1
    p = p.clamp(0.0, float(L))
    i0 = p.floor().clamp(max=L - 1)
    frac = p - i0
    i0 = i0.long()
    return S.gather(1, i0) + frac * (S.gather(1, i0 + 1) - S.gather(1, i0))


def _interp_points(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """x [B, L] samples at integer positions 0..L-1, p [B, M] real positions:
    linear interpolation between samples, linear extrapolation beyond the
    ends (the convention of `evaluation.ratein.upsample`)."""
    L = x.shape[1]
    if L == 1:
        return x.expand(-1, p.shape[1])
    i0 = p.floor().clamp(0, L - 2)
    frac = p - i0
    i0 = i0.long()
    x0, x1 = x.gather(1, i0), x.gather(1, i0 + 1)
    return x0 + frac * (x1 - x0)


def resample_context(x: torch.Tensor, k: torch.Tensor, out_len: int = MODEL_LEN,
                     lengths: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native context x [B, L] (left-padded; `lengths` [B] gives the valid
    suffix, default L) and rate k [B] -> (y [B, out_len], valid [B, out_len]).

    Block j of the output covers native positions
    [L - (out_len - j) k, L - (out_len - 1 - j) k): right-aligned on the last
    observation for every k, so the forecast origin never moves.
    k >= 1: block mean (cumulative sums, anti-aliased by construction).
    k < 1 : value interpolated between native sample centres, with linear
            extrapolation at the ends.
    Blocks that start before the valid suffix are invalid and filled with the
    first valid block of their row (edge padding, as the harness does for
    short series). A row with no valid block is returned as NaN: the caller
    bounds k so that this cannot happen (`RateINLayer.bound_log_k`).
    """
    B, L = x.shape
    # Sampling positions are computed in float64 whatever the data dtype: a
    # float32 k = 1/3 shifts the far end of a 1024-block grid by 1e-4 steps.
    k = k.reshape(B, 1).to(torch.float64)
    if lengths is None:
        lengths = torch.full((B,), L, dtype=torch.long, device=x.device)
    start = (L - lengths).to(torch.float64).reshape(B, 1)
    j = torch.arange(out_len, device=x.device, dtype=torch.float64)[None, :]
    b = L - (out_len - 1 - j) * k
    a = b - k
    valid = a >= start - 1e-4          # samples sit on integers: 1e-4 is noise, not a sample

    xz = torch.nan_to_num(x, nan=0.0)
    x64 = xz.to(torch.float64)
    S = torch.cat([torch.zeros(B, 1, dtype=torch.float64, device=x.device), x64.cumsum(1)], dim=1)
    y_area = (_interp_cumsum(S, b) - _interp_cumsum(S, a)) / k
    # k < 1: centre of the fine block at b - k/2, native sample i centred at i + 1/2
    y_pts = _interp_points(x64, (b - 0.5 * k) - 0.5)
    y = torch.where(k >= 1.0, y_area, y_pts)
    # k == 1 exactly: the samples themselves, so the layer at its prior is the
    # plain path bit for bit (no cumulative-sum round trip).
    idx1 = (b - 1.0).round().clamp(0, L - 1).long()
    y = torch.where(k == 1.0, x64.gather(1, idx1), y)

    any_valid = valid.any(1)
    first = valid.to(torch.long).argmax(1, keepdim=True)
    fill = y.gather(1, first)
    y = torch.where(valid, y, fill)
    y = torch.where(any_valid[:, None], y, torch.full_like(y, float("nan")))
    return y.to(x.dtype), valid


def fan_to_native(fan: torch.Tensor, k: torch.Tensor, h: int) -> torch.Tensor:
    """fan [B, H_m, Q] on the model grid, k [B] -> [B, h, Q] on the native
    grid. k >= 1: model step i covers native steps [i k, (i+1) k), its value
    sits at the centre (i + 1/2) k - 1/2 and native steps interpolate between
    centres, constant beyond the ends (`reinterp_fan`). k < 1: native step t
    is the mean of the fine steps it covers (`pool_fan`). Requires
    H_m >= ceil(h / k) on every row (checked)."""
    B, Hm, Q = fan.shape
    k = k.reshape(B, 1).to(torch.float64)
    need = torch.ceil(h / k - 1e-6)
    if bool((need > Hm).any()):
        raise ValueError(f"fan has {Hm} model steps, {int(need.max())} needed for h={h} at the smallest k")
    t = torch.arange(h, device=fan.device, dtype=torch.float64)[None, :]
    # k >= 1: interpolation between centres, constant beyond the last of the
    # ceil(h / k) steps the row needs (a batch may carry more steps than a row
    # uses; they must not leak into its edge, as in `reinterp_fan`).
    n_row = need.clamp(min=1.0)                                            # [B, 1]
    pos = torch.minimum(((t + 0.5) / k - 0.5).clamp(min=0.0), n_row - 1)
    i0 = pos.floor()
    i1 = torch.minimum(i0 + 1, n_row - 1)
    frac = (pos - i0).unsqueeze(-1)                                        # 0 exactly at k = 1
    i0 = i0.long().unsqueeze(-1).expand(B, h, Q)
    i1 = i1.long().unsqueeze(-1).expand(B, h, Q)
    f0 = fan.gather(1, i0)
    out_interp = f0 + frac * (fan.gather(1, i1) - f0)
    # k < 1: block means of the fine fan over [t/k, (t+1)/k)
    S = torch.cat([torch.zeros(B, 1, Q, dtype=fan.dtype, device=fan.device), fan.cumsum(1)], dim=1)
    lo, hi = (t / k).clamp(0.0, Hm), ((t + 1) / k).clamp(0.0, Hm)
    S_flat = S.permute(0, 2, 1).reshape(B * Q, Hm + 1)
    lo_f = lo.unsqueeze(1).expand(B, Q, h).reshape(B * Q, h)
    hi_f = hi.unsqueeze(1).expand(B, Q, h).reshape(B * Q, h)
    out_pool = ((_interp_cumsum(S_flat, hi_f) - _interp_cumsum(S_flat, lo_f))
                / (hi_f - lo_f).clamp(min=1e-9)).reshape(B, Q, h).permute(0, 2, 1)
    return torch.where((k >= 1.0).unsqueeze(-1), out_interp, out_pool).to(fan.dtype)


# -------------------------------------------------------------------- selector
class RateSelector(nn.Module):
    """g(x): a bank of one-pole complex filters (a diagonal SSM with B = 1,
    read at the last step), the log energy of each mode, and a small MLP to
    one scalar. The modes start as a constant-Q filterbank over periods
    [min_period, max_period]; rates and frequencies are learned. The last
    layer starts at zero: the layer begins on the prior."""

    def __init__(self, n_modes: int = 48, hidden: int = 32,
                 min_period: float = 2.0, max_period: float = 2048.0, q: float = 4.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_modes)
        theta = 2.0 * math.pi / periods
        self.theta = nn.Parameter(theta)
        self.log_decay = nn.Parameter(torch.log(theta / q))
        # Log energies span a dozen units: normalize before the MLP so the first
        # optimizer steps move k by a few percent, not by a factor.
        self.mlp = nn.Sequential(nn.LayerNorm(n_modes + 1), nn.Linear(n_modes + 1, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = x.shape
        if lengths is None:
            lengths = torch.full((B,), L, dtype=torch.long, device=x.device)
        pos = torch.arange(L, device=x.device)[None, :]
        valid = (pos >= (L - lengths)[:, None]) & torch.isfinite(x)
        xz = torch.where(valid, x, torch.zeros_like(x))
        n = valid.sum(1, keepdim=True).clamp(min=1).to(x.dtype)
        mean = xz.sum(1, keepdim=True) / n
        xc = torch.where(valid, x - mean, torch.zeros_like(x))
        std = (xc.pow(2).sum(1, keepdim=True) / n).sqrt()
        xn = xc / (std + EPS)
        lag = torch.arange(L, device=x.device, dtype=x.dtype)
        decay = torch.exp(self.log_decay)[:, None]                    # [N, 1]
        env = torch.exp(-decay * lag[None, :])                        # [N, L]
        phase = self.theta[:, None] * lag[None, :]
        kc, ks = env * torch.cos(phase), env * torch.sin(phase)
        xr = torch.flip(xn, dims=[1])                                  # lag 0 = last point
        sr, si = xr @ kc.T, xr @ ks.T                                  # [B, N]
        energy = sr.pow(2) + si.pow(2)
        f = torch.log(energy + EPS)
        return torch.cat([f, torch.log(n / float(MODEL_LEN))], dim=1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mlp(self.features(x, lengths)).squeeze(-1)


# ----------------------------------------------------------------------- layer
ForecastFn = Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]


class RateINLayer(nn.Module):
    """Selector + resampling + return to the native grid around a frozen
    forecaster `forecast_fn(y [B, model_len], n, k [B]) -> fan [B, n, Q]`
    (raw units; k is passed so a model with a rate knob can set it)."""

    def __init__(self, selector: Optional[RateSelector] = None, model_len: int = MODEL_LEN,
                 k_min: float = K_MIN, k_max: float = K_MAX, min_blocks: int = 32):
        super().__init__()
        self.selector = selector if selector is not None else RateSelector()
        self.model_len = int(model_len)
        self.log_k_min, self.log_k_max = math.log(k_min), math.log(k_max)
        self.min_blocks = int(min_blocks)

    def bound_log_k(self, log_k: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Clamp to the grid's range and to the rates that leave at least
        `min_blocks` valid blocks in the context (k <= len / min_blocks)."""
        hi = torch.log(lengths.to(log_k.dtype) / self.min_blocks).clamp(max=self.log_k_max)
        hi = torch.maximum(hi, torch.full_like(hi, self.log_k_min))
        return torch.maximum(torch.minimum(log_k, hi), torch.full_like(log_k, self.log_k_min))

    def log_k(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None,
              log_k_prior: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
        B, L = x.shape
        if lengths is None:
            lengths = torch.full((B,), L, dtype=torch.long, device=x.device)
        prior = torch.as_tensor(log_k_prior, dtype=x.dtype, device=x.device).expand(B) \
            if not torch.is_tensor(log_k_prior) or log_k_prior.dim() == 0 else log_k_prior
        return self.bound_log_k(prior + self.selector(x, lengths), lengths)

    @staticmethod
    def model_horizon(h: int, k: torch.Tensor) -> int:
        """Steps to ask the forecaster for: ceil(h / k) for the smallest k of
        the batch (h when a row runs at the native rate)."""
        kmin = float(k.detach().min())
        return int(math.ceil(h / kmin - 1e-6))

    def forward(self, x: torch.Tensor, h: int, forecast_fn: ForecastFn,
                lengths: Optional[torch.Tensor] = None,
                log_k_prior: Union[float, torch.Tensor] = 0.0) -> dict:
        B, L = x.shape
        if lengths is None:
            lengths = torch.full((B,), L, dtype=torch.long, device=x.device)
        lk = self.log_k(x, lengths, log_k_prior)
        k = torch.exp(lk)
        y, valid = resample_context(x, k, self.model_len, lengths)
        n = self.model_horizon(h, k)
        fan = forecast_fn(y, n, k)
        out = fan_to_native(fan, k, h)
        return {"fan": out, "k": k, "log_k": lk, "context": y, "valid": valid, "model_horizon": n}


# ------------------------------------------------------------------------ loss
def pinball_wql(fan: torch.Tensor, target: torch.Tensor, levels) -> torch.Tensor:
    """Weighted quantile loss per item, the harness's CRPS proxy:
    mean over levels of sum_t 2|(q - y)(1[y <= q] - tau)| / sum_t |y|,
    NaN targets masked. fan [B, h, Q], target [B, h] -> [B]."""
    q = torch.as_tensor(levels, dtype=fan.dtype, device=fan.device)[None, None, :]
    mask = torch.isfinite(target)
    y = torch.where(mask, target, torch.zeros_like(target)).unsqueeze(-1)
    ql = 2.0 * torch.abs((fan - y) * ((y <= fan).to(fan.dtype) - q))
    ql = (ql * mask.unsqueeze(-1)).sum(1).mean(1)
    denom = (y.squeeze(-1).abs() * mask).sum(1).clamp(min=EPS)
    return ql / denom


def kernel_loss_curve(log_k_grid: torch.Tensor, losses: torch.Tensor,
                      log_k: torch.Tensor, sigma: float = 0.25) -> torch.Tensor:
    """Grid mode, training objective: the loss curve smoothed by a Gaussian
    kernel in log k (bandwidth sigma, below the grid spacing), evaluated at
    log_k [B]. Smooth everywhere, so the gradient exists at the grid points
    where the selector starts (a piecewise-linear curve has a kink there, and
    torch >= 2.14 returns no gradient through a clamp at its bound). The
    kernel also averages the noise of neighbouring rates, the target we want
    the selector to regress."""
    d = (log_k[:, None] - log_k_grid[None, :].to(log_k.dtype)) / sigma
    w = torch.softmax(-0.5 * d * d, dim=1)
    return (w * losses.to(w.dtype)).sum(1)


def interp_loss_curve(log_k_grid: torch.Tensor, losses: torch.Tensor,
                      log_k: torch.Tensor) -> torch.Tensor:
    """Grid mode, reporting: losses [B, G] measured at the grid rates
    log_k_grid [G] (sorted), evaluated at a continuous log_k [B] by linear
    interpolation, constant beyond the ends. The point estimate of the loss
    at the chosen rate; use `kernel_loss_curve` to train."""
    G = log_k_grid.numel()
    idx = torch.searchsorted(log_k_grid, log_k.detach().contiguous()).clamp(1, G - 1)
    x0, x1 = log_k_grid[idx - 1], log_k_grid[idx]
    w = ((log_k - x0) / (x1 - x0)).clamp(0.0, 1.0)
    l0 = losses.gather(1, (idx - 1)[:, None]).squeeze(1)
    l1 = losses.gather(1, idx[:, None]).squeeze(1)
    return l0 + w * (l1 - l0)


# ------------------------------------------------------------------ checkpoint
def load_layer(path: Union[str, "os.PathLike"], device="cpu") -> "RateINLayer":
    """A layer saved by scripts/train_ratein_layer.py (state_dict + meta)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    layer = RateINLayer(RateSelector(int(meta["n_modes"]), int(meta["hidden"])),
                        model_len=int(meta["model_len"]), min_blocks=int(meta["min_blocks"]))
    layer.load_state_dict(ckpt["state_dict"])
    layer.meta = meta
    return layer.to(torch.device(device)).eval()
