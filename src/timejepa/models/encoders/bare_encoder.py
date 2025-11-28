# src/timejepa/models/encoders/bare_encoder.py
"""
Bare Transformer Encoder (sans patching) with RoPE support.

Pour JEPA qui gère le patching en externe.
Juste les transformer blocks + layer norm.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from ..components.rope import RotaryPositionEmbedding

class BareTransformerEncoder(nn.Module):
    """
    Encoder sans patching, juste transformer blocks.

    Pour JEPA où le patching est géré en amont pour appliquer les masks.

    Args:
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_rope: Whether to use Rotary Position Embedding
        max_seq_len: Maximum sequence length for RoPE cache
        rope_base: Base for RoPE frequency computation
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 24,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_rope: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        
        # Validate dimensions
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads
        
        # RoPE - computed once, shared across all layers
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                base=rope_base
            )
        else:
            self.rope = None
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithRoPE(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False  # Non-causal for JEPA encoder
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass.
        
        Args:
            x: Patch embeddings [B, num_patches, d_model]
            attention_mask: Optional attention mask [B, 1, 1, num_patches] or [B, num_patches]
            return_hidden_states: If True, returns all layer outputs
            
        Returns:
            Encoded representations [B, num_patches, d_model]
            Or tuple of (representations, hidden_states) if return_hidden_states=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Get RoPE cos/sin if using
        rope_cos, rope_sin = None, None
        if self.rope is not None:
            rope_cos, rope_sin = self.rope(x, seq_len=seq_len)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin
            )
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        if return_hidden_states:
            hidden_states.append(x)
            return x, hidden_states
        
        return x


class TransformerBlockWithRoPE(nn.Module):
    """
    Transformer block with RoPE support.
    
    Architecture: Pre-norm (like LLaMA, GPT-NeoX)
        x -> LayerNorm -> Attention(RoPE) -> + x
        x -> LayerNorm -> FFN -> + x
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function ('gelu', 'relu', 'swiglu')
        causal: Whether to use causal masking
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        causal: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention with RoPE
        self.attention = MultiHeadAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, seq_len, d_model]
            attention_mask: Optional mask
            rope_cos: Cosine values for RoPE [seq_len, head_dim]
            rope_sin: Sine values for RoPE [seq_len, head_dim]
            
        Returns:
            Output tensor [B, seq_len, d_model]
        """
        # Pre-norm attention with residual
        normed = self.norm1(x)
        attn_out = self.attention(
            normed,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=attention_mask
        )
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        
        return x


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding.
    
    RoPE is applied to Q and K before computing attention scores.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        causal: Whether to apply causal masking
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections (can be fused for efficiency)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, seq_len, d_model]
            rope_cos: Cosine for RoPE [seq_len, head_dim]
            rope_sin: Sine for RoPE [seq_len, head_dim]
            attention_mask: Optional mask [B, 1, 1, seq_len] or [B, seq_len]
            
        Returns:
            Output tensor [B, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [B, seq_len, d_model] -> [B, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        if rope_cos is not None and rope_sin is not None:
            q, k = self._apply_rope(q, k, rope_cos, rope_sin)
        
        # Compute attention scores
        # [B, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:
                # [B, seq_len] -> [B, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        # [B, num_heads, seq_len, head_dim] -> [B, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to Q and K.
        
        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim]
            k: Key tensor [B, num_heads, seq_len, head_dim]
            cos: Cosine values [seq_len, head_dim]
            sin: Sine values [seq_len, head_dim]
            
        Returns:
            (rotated_q, rotated_k)
        """
        # Reshape cos/sin for broadcasting
        # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation.
    
    Supports GELU, ReLU, and SwiGLU activations.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.activation_name = activation.lower()
        
        if self.activation_name == 'swiglu':
            # SwiGLU: gate * swish(x)
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
            self.activation = nn.SiLU()
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            
            if self.activation_name == 'gelu':
                self.activation = nn.GELU()
            elif self.activation_name == 'relu':
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.activation_name == 'swiglu':
            # SwiGLU: output = w2(swish(w1(x)) * w3(x))
            return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.fc2(self.activation(self.fc1(x))))