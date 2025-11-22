"""
Bare Transformer Encoder (sans patching).

Pour JEPA qui gère le patching en externe.
Juste les transformer blocks + layer norm.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..components.attention import TransformerBlock


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
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 24,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False  # Non-causal for JEPA
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
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Patch embeddings [B, num_patches, d_model]
            attention_mask: Optional attention mask
            return_hidden_states: If True, returns all layer outputs
            
        Returns:
            Encoded representations [B, num_patches, d_model]
        """
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        if return_hidden_states:
            hidden_states.append(x)
            return x, hidden_states
        
        return x