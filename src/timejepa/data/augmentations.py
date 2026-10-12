"""
Data augmentations for time series JEPA pretraining.
Inspired by TTM's Diverse Resolution Sampling and scaling strategies.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for time series augmentations."""
    # Global toggle
    enabled: bool = True
    
    # Random scaling (crucial for MAPE robustness)
    scale_enabled: bool = True
    scale_range: Tuple[float, float] = (0.5, 2.0)
    scale_log_uniform: bool = True  # Log-uniform is more balanced
    p_scale: float = 0.5
    
    # Gaussian jitter (context only for JEPA)
    jitter_enabled: bool = True
    jitter_std: float = 0.03
    jitter_relative: bool = True  # Relative to signal std
    p_jitter: float = 0.3
    
    # Magnitude warping 
    magnitude_warp_enabled: bool = True
    magnitude_warp_sigma: float = 0.2
    magnitude_warp_knots: int = 4
    p_magnitude_warp: float = 0.3
    
    # Temporal shift/jitter 
    temporal_shift_enabled: bool = False
    temporal_shift_max: int = 2  # Max shift in timesteps
    p_temporal_shift: float = 0.2
    
    # Diverse Resolution Sampling
    drs_enabled: bool = True
    drs_factors: Tuple[int, ...] = (2, 3, 4)
    drs_interpolation: str = "linear"  # linear, cubic
    p_drs: float = 0.15
    
    # Trend injection (helps with non-stationary data)
    trend_enabled: bool = False
    trend_magnitude: float = 0.1
    p_trend: float = 0.2

    # TiRex-style augmentations, all disabled by default so existing configs
    # are unchanged. Unlike random scaling, which robust (median/MAD)
    # normalization cancels, these three change the shape of the signal.
    # Amplitude modulation: multiplication by a piecewise-linear trend.
    amplitude_mod_enabled: bool = False
    amplitude_mod_knots: Tuple[int, int] = (2, 6)
    amplitude_mod_range: Tuple[float, float] = (0.5, 1.5)
    p_amplitude_mod: float = 0.5
    # Censoring: clipping at a random quantile of the window (saturated
    # sensors, capacities).
    censor_enabled: bool = False
    censor_quantile_range: Tuple[float, float] = (0.85, 0.99)
    p_censor: float = 0.5
    # Sparse periodic spikes, applied to context and target together so the
    # pattern stays predictable.
    spike_enabled: bool = False
    spike_amp_range: Tuple[float, float] = (3.0, 10.0)
    p_spike: float = 0.05


class TimeSeriesAugmentations:
    """
    Augmentations for time series JEPA pretraining.
    
    Key design decisions:
    - Scale augmentation is applied to BOTH context and target (same factor)
    - Jitter is applied to context ONLY (target should be clean for JEPA)
    - Magnitude warp uses the same curve for context and target
    - DRS downsamples then interpolates back to original length
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
    @staticmethod
    def from_dict(cfg_dict: Dict[str, Any]) -> "TimeSeriesAugmentations":
        """Create from config dictionary."""
        config = AugmentationConfig(**cfg_dict)
        return TimeSeriesAugmentations(config)
    
    def random_scale(
        self, 
        context: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale context and target by the same random factor.

        Log-uniform by default: scale_range=(0.5, 2.0) gives equal
        probability to 0.5-1.0 and 1.0-2.0.
        """
        if not self.config.scale_enabled:
            return context, target
            
        if torch.rand(1).item() >= self.config.p_scale:
            return context, target
        
        low, high = self.config.scale_range
        
        if self.config.scale_log_uniform:
            log_low, log_high = np.log(low), np.log(high)
            log_scale = torch.empty(1).uniform_(log_low, log_high)
            scale = torch.exp(log_scale).item()
        else:
            scale = torch.empty(1).uniform_(low, high).item()
        
        return context * scale, target * scale
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to signal.
        
        For JEPA: apply to context only, not target.
        """
        if not self.config.jitter_enabled:
            return x
            
        if torch.rand(1).item() >= self.config.p_jitter:
            return x
        
        if self.config.jitter_relative:
            noise_std = self.config.jitter_std * x.std().item()
        else:
            noise_std = self.config.jitter_std
        
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def magnitude_warp(
        self, 
        context: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Smooth magnitude warping using cubic spline.
        
        Creates a smooth curve that multiplies the signal,
        simulating slow gain changes. Same curve for context and target.
        """
        if not self.config.magnitude_warp_enabled:
            return context, target
            
        if torch.rand(1).item() >= self.config.p_magnitude_warp:
            return context, target
        
        total_len = context.shape[-1] + target.shape[-1]
        n_knots = self.config.magnitude_warp_knots
        
        knot_values = torch.randn(n_knots) * self.config.magnitude_warp_sigma + 1.0

        # Linear interpolation between knots
        knot_positions = torch.linspace(0, 1, n_knots)
        full_positions = torch.linspace(0, 1, total_len)

        warp_curve = torch.zeros(total_len)
        for i in range(n_knots - 1):
            mask = (full_positions >= knot_positions[i]) & (full_positions <= knot_positions[i + 1])
            if mask.any():
                t = (full_positions[mask] - knot_positions[i]) / (knot_positions[i + 1] - knot_positions[i])
                warp_curve[mask] = knot_values[i] * (1 - t) + knot_values[i + 1] * t
        
        # Split curve for context and target
        ctx_len = context.shape[-1]
        warp_context = warp_curve[:ctx_len].to(context.device)
        warp_target = warp_curve[ctx_len:].to(target.device)
        
        # Apply (handle both 1D and 2D cases)
        if context.dim() == 1:
            context = context * warp_context
            target = target * warp_target
        else:
            context = context * warp_context.unsqueeze(0)
            target = target * warp_target.unsqueeze(0)
        
        return context, target
    
    def amplitude_modulation(
        self,
        context: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TiRex-style amplitude modulation: multiply by a piecewise-linear
        curve with random breakpoints (the magnitude warp uses a regular
        grid). One continuous curve over context + target, applied to both.
        """
        if not self.config.amplitude_mod_enabled:
            return context, target
        if torch.rand(1).item() >= self.config.p_amplitude_mod:
            return context, target

        total_len = context.shape[-1] + target.shape[-1]
        lo, hi = self.config.amplitude_mod_knots
        n_knots = int(torch.randint(lo, hi + 1, (1,)).item())
        # random breakpoint positions (sorted), independent values
        pos = torch.sort(torch.rand(n_knots)).values
        pos = torch.cat([torch.zeros(1), pos, torch.ones(1)])
        a, b = self.config.amplitude_mod_range
        vals = torch.rand(n_knots + 2) * (b - a) + a
        grid = torch.linspace(0, 1, total_len)
        idx = torch.searchsorted(pos, grid.clamp(max=pos[-1] - 1e-9)).clamp(min=1)
        left, right = pos[idx - 1], pos[idx]
        t = (grid - left) / (right - left).clamp_min(1e-9)
        curve = vals[idx - 1] * (1 - t) + vals[idx] * t

        ctx_len = context.shape[-1]
        c_curve = curve[:ctx_len].to(context.device)
        t_curve = curve[ctx_len:].to(target.device)
        if context.dim() == 1:
            return context * c_curve, target * t_curve
        return context * c_curve.unsqueeze(0), target * t_curve.unsqueeze(0)

    def censor(
        self,
        context: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TiRex-style censoring: clip at quantile q ~ U(censor_quantile_range)
        of the full window, with the same threshold for context and target.
        """
        if not self.config.censor_enabled:
            return context, target
        if torch.rand(1).item() >= self.config.p_censor:
            return context, target
        a, b = self.config.censor_quantile_range
        q = torch.rand(1).item() * (b - a) + a
        full = torch.cat([context.reshape(-1), target.reshape(-1)])
        cap = torch.quantile(full.float(), q).to(context.dtype)
        return torch.clamp(context, max=cap), torch.clamp(target, max=cap)

    def spike_injection(
        self,
        context: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TiRex-style sparse periodic spikes. The period continues from the
        context into the target, so the pattern is predictable.
        """
        if not self.config.spike_enabled:
            return context, target
        if torch.rand(1).item() >= self.config.p_spike:
            return context, target

        total_len = context.shape[-1] + target.shape[-1]
        period = int(torch.randint(16, max(17, total_len // 8), (1,)).item())
        offset = int(torch.randint(0, period, (1,)).item())
        a, b = self.config.spike_amp_range
        amp = torch.rand(1).item() * (b - a) + a
        sign = 1.0 if torch.rand(1).item() < 0.8 else -1.0
        full = torch.cat([context.reshape(1, -1) if context.dim() == 1
                          else context,
                          target.reshape(1, -1) if target.dim() == 1
                          else target], dim=-1)
        scale = full.float().std().clamp_min(1e-6)
        spikes = torch.zeros(total_len, device=full.device)
        spikes[offset::period] = sign * amp * scale
        ctx_len = context.shape[-1]
        if context.dim() == 1:
            return context + spikes[:ctx_len], target + spikes[ctx_len:]
        return (context + spikes[:ctx_len].unsqueeze(0),
                target + spikes[ctx_len:].unsqueeze(0))

    def diverse_resolution_sampling(
        self,
        context: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TTM-style Diverse Resolution Sampling: downsample by a random factor,
        then interpolate back to the same length.

        This is a low-pass operation that simulates a coarser sensor. The
        window still covers the same time span, so the period-to-patch ratio
        is unchanged. To change the sampling frequency itself, see
        `TimeSeriesDataset.get_item(allow_multi_resolution=True)`, which reads
        a longer raw stretch and decimates it.

        Applied to context and target together.
        """
        if not self.config.drs_enabled:
            return context, target
            
        if torch.rand(1).item() >= self.config.p_drs:
            return context, target
        
        factor = int(np.random.choice(self.config.drs_factors))
        
        # Process context
        ctx_len = context.shape[-1]
        if context.dim() == 1:
            ctx_down = context[::factor]
            ctx_up = F.interpolate(
                ctx_down.view(1, 1, -1),
                size=ctx_len,
                mode=self.config.drs_interpolation,
                align_corners=True if self.config.drs_interpolation != 'nearest' else None
            ).view(-1)
        else:
            # (C, L) -> (1, C, L)
            ctx_down = context[..., ::factor]
            ctx_up = F.interpolate(
                ctx_down.unsqueeze(0),
                size=ctx_len,
                mode=self.config.drs_interpolation,
                align_corners=True if self.config.drs_interpolation != 'nearest' else None
            ).squeeze(0)
        
        # Process target
        tgt_len = target.shape[-1]
        if target.dim() == 1:
            tgt_down = target[::factor]
            tgt_up = F.interpolate(
                tgt_down.view(1, 1, -1),
                size=tgt_len,
                mode=self.config.drs_interpolation,
                align_corners=True if self.config.drs_interpolation != 'nearest' else None
            ).view(-1)
        else:
            tgt_down = target[..., ::factor]
            tgt_up = F.interpolate(
                tgt_down.unsqueeze(0),
                size=tgt_len,
                mode=self.config.drs_interpolation,
                align_corners=True if self.config.drs_interpolation != 'nearest' else None
            ).squeeze(0)
        
        return ctx_up, tgt_up
    
    def add_trend(
        self, 
        context: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add a synthetic linear trend, scaled by the context std."""
        if not self.config.trend_enabled:
            return context, target
            
        if torch.rand(1).item() >= self.config.p_trend:
            return context, target
        
        total_len = context.shape[-1] + target.shape[-1]
        
        slope = (torch.rand(1).item() * 2 - 1) * self.config.trend_magnitude
        t = torch.linspace(0, 1, total_len)
        trend = slope * t * context.std().item()
        
        ctx_len = context.shape[-1]
        trend_ctx = trend[:ctx_len].to(context.device)
        trend_tgt = trend[ctx_len:].to(target.device)
        
        if context.dim() == 1:
            context = context + trend_ctx
            target = target + trend_tgt
        else:
            context = context + trend_ctx.unsqueeze(0)
            target = target + trend_tgt.unsqueeze(0)
        
        return context, target
    
    
    def __call__(
        self, 
        context: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all enabled augmentations.
        
        Order matters:
        1. DRS (changes resolution - do first)
        2. Scale (global magnitude)
        3. Magnitude warp (smooth local magnitude)
        4. Trend (additive component)
        5. Jitter (noise - context only)
        """
        if not self.config.enabled:
            return context, target
        
        # Joint augmentations (same transform for context and target)
        context, target = self.diverse_resolution_sampling(context, target)
        context, target = self.random_scale(context, target)
        context, target = self.magnitude_warp(context, target)
        # TiRex-style, off by default: amplitude modulation, then spikes, then
        # censoring (the ceiling clips everything before it, spikes included).
        context, target = self.amplitude_modulation(context, target)
        context, target = self.spike_injection(context, target)
        context, target = self.censor(context, target)
        context, target = self.add_trend(context, target)

        # Context-only augmentations (target stays clean for JEPA)
        context = self.jitter(context)

        return context, target


class FinetuneAugmentations(TimeSeriesAugmentations):
    """
    Lighter augmentations for finetuning.
    
    Less aggressive than pretraining augmentations.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        if config is None:
            config = AugmentationConfig(
                enabled=True,
                scale_enabled=True,
                scale_range=(0.8, 1.25),  # Less aggressive
                p_scale=0.3,
                jitter_enabled=True,
                jitter_std=0.01,  # Less noise
                p_jitter=0.2,
                magnitude_warp_enabled=False,  # Disable for finetune
                drs_enabled=False,  # Disable for finetune
                trend_enabled=False,
            )
        super().__init__(config)