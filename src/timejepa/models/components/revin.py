"""
Reversible Instance Normalization (RevIN)

Paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting
        against Distribution Shift" (ICLR 2022)

Key idea: Normalize each instance, then denormalize at output
- Helps model generalize across different scales/distributions
- Critical for time series with varying magnitudes
"""

import torch
import torch.nn as nn
from typing import Optional


class RevIN(nn.Module):
    """
    Reversible Instance Normalization layer.
    
    Normalizes each time series instance independently using its own
    mean and standard deviation, then stores these statistics to 
    reverse the normalization at the output.
    
    Args:
        num_features: Number of features (channels) in the input
        eps: Small constant for numerical stability
        affine: If True, learns affine parameters (scale and shift)
        subtract_last: If True, subtract the last value instead of mean
                      (useful for some forecasting tasks)
    """
    
    def __init__(
        self,
        num_features: int = 1,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        # Learnable affine parameters (optional)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Store statistics for denormalization
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'norm'
    ) -> torch.Tensor:
        """
        Forward pass: normalize or denormalize
        
        Args:
            x: Input tensor of shape [B, L, C] or [B, L]
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode should be 'norm' or 'denorm', got {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input.
        
        Args:
            x: Input tensor [B, L] or [B, L, C]
            
        Returns:
            Normalized tensor with same shape
        """
        # Handle 2D input [B, L] -> [B, L, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Compute statistics per instance (across time dimension)
        # x shape: [B, L, C]
        if self.subtract_last:
            # Subtract last timestep value
            self.mean = x[:, -1:, :].detach()  # [B, 1, C]
        else:
            # Subtract mean
            self.mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
        
        x = x - self.mean
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, 1, C]
        
        x = x / self.std
        
        # Apply affine transformation if enabled
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the output using stored statistics.
        
        Args:
            x: Normalized tensor [B, L] or [B, L, C]
            
        Returns:
            Denormalized tensor with same shape
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Reverse affine transformation if enabled
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        # Denormalize
        x = x * self.std + self.mean
        
        return x


class RevINMultivariate(nn.Module):
    """
    RevIN for multivariate time series.
    
    Applies RevIN independently to each feature/channel.
    
    Args:
        num_features: Number of features (channels)
        eps: Small constant for numerical stability
        affine: If True, learns separate affine params for each feature
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('std', torch.ones(1, 1, num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, C]
            mode: 'norm' or 'denorm'
            
        Returns:
            Processed tensor [B, L, C]
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode should be 'norm' or 'denorm', got {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each feature independently."""
        # x: [B, L, C]
        self.mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, 1, C]
        
        x = (x - self.mean) / self.std
        
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using stored statistics."""
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        x = x * self.std + self.mean
        return x