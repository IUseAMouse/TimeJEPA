# src/timejepa/models/components/attention.py
"""
Multi-head attention with Rotary Position Embedding (RoPE).

Combines standard scaled dot-product attention with RoPE for
better position encoding and extrapolation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class RoPEAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding.
    
    Key features:
    - Uses RoPE for position encoding (better than absolute)
    - Supports causal masking for autoregressive tasks
    - Efficient implementation with proper scaling
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for RoPE
        causal: Whether to use causal masking
        rope_base: Base for RoPE frequencies
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        causal: bool = False,
        rope_base: float = 10000.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        # Causal mask (if needed)
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                    1, 1, max_seq_len, max_seq_len
                ),
                persistent=False
            )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, d_model]
            attention_mask: Optional mask [B, L] or [B, L, L]
            position_ids: Optional position indices [B, L]
            
        Returns:
            Output tensor [B, L, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, L, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [B, L, d_model] -> [B, num_heads, L, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k, seq_len=seq_len, position_ids=position_ids)
        
        # Compute attention scores
        # [B, num_heads, L, head_dim] @ [B, num_heads, head_dim, L]
        # -> [B, num_heads, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if attention_mask is not None:
            # Expand mask dimensions if needed
            if attention_mask.dim() == 2:
                # [B, L] -> [B, 1, 1, L]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [B, L, L] -> [B, 1, L, L]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (set masked positions to large negative value)
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        
        # Apply causal mask if enabled
        if self.causal:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(
                causal_mask == 0,
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, L, L] @ [B, num_heads, L, head_dim]
        # -> [B, num_heads, L, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to [B, L, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with RoPE attention.
    
    Architecture:
        x -> LayerNorm -> RoPEAttention -> Add -> LayerNorm -> FFN -> Add -> output
    
    Uses Pre-LN architecture (LayerNorm before attention/FFN) which is
    more stable for deep networks.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (usually 4 * d_model)
        dropout: Dropout rate
        activation: Activation function ('relu' or 'gelu')
        causal: Whether to use causal attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        causal: bool = False
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Multi-head attention with RoPE
        self.attention = RoPEAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, L, d_model]
            attention_mask: Optional attention mask
            
        Returns:
            Output [B, L, d_model]
        """
        # Pre-LN: Attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = x + residual
        
        # Pre-LN: FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x