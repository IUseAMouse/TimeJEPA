# src/timejepa/models/components/rope.py
"""
Rotary Position Embedding (RoPE)

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
       (arXiv:2104.09864)

Used in: LLaMA, MOIRAI, and many modern transformers

Key advantages:
- Relative position encoding (better extrapolation)
- No position limit (unlike learned embeddings)
- Computationally efficient
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotation to query and key vectors based on their position.
    Encodes relative position information implicitly through rotation.
    
    Args:
        dim: Dimension of embeddings (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression (default: 10000)
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        assert dim % 2 == 0, f"Dimension must be even, got {dim}"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        # θ_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        self._set_cos_sin_cache(max_seq_len, device=device)
    
    def _set_cos_sin_cache(
        self, 
        seq_len: int, 
        device: Optional[torch.device] = None
    ):
        """Precompute cos and sin values for all positions."""
        self.max_seq_len_cached = seq_len
        
        # Create position indices [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inv_freq
        # freqs shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate to get [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Precompute cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for the given sequence length.
        
        Args:
            x: Input tensor (used only for device/dtype info)
            seq_len: Sequence length (if None, uses x.shape[1])
            
        Returns:
            (cos, sin) tensors of shape [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    This is used to apply the rotation matrix efficiently.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor with same shape
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [B, num_heads, seq_len, head_dim]
        k: Key tensor [B, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        sin: Sine values [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        position_ids: Optional position indices [B, seq_len]
        
    Returns:
        (rotated_q, rotated_k) with same shapes as inputs
    """
    # If position_ids provided, gather specific positions
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [B, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Reshape for broadcasting
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    Simplified wrapper for RoPE that can be used as a module.
    
    Usage:
        rope = RotaryEmbedding(dim=64, max_seq_len=2048)
        q_rot, k_rot = rope(q, k, seq_len=512)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.rope = RotaryPositionEmbedding(dim, max_seq_len, base)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.
        
        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim]
            k: Key tensor [B, num_heads, seq_len, head_dim]
            seq_len: Sequence length
            position_ids: Optional position indices
            
        Returns:
            (q_rotated, k_rotated)
        """
        cos, sin = self.rope(q, seq_len)
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)