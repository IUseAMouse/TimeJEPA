# src/timejepa/models/predictors/transformer_predictor.py
"""
Transformer Predictor for JEPA.

This is a lightweight transformer that predicts target representations
from context representations. It's intentionally smaller than the encoder
(JEPA principle: predictor should be simpler than encoder).

Architecture: 4 layers vs 24 layers in encoder
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..components.attention import TransformerBlock


class TransformerPredictor(nn.Module):
    """
    Lightweight transformer predictor for JEPA.
    
    Takes context representations and predicts target representations.
    Key JEPA design: predictor is lighter than encoder (4 vs 24 layers).
    
    Architecture:
        Context embeddings [B, N_context, d_model]
        → Positional tokens for targets [B, N_target, d_model]
        → Concat [B, N_context + N_target, d_model]
        → Transformer blocks (4 layers)
        → Extract target predictions [B, N_target, d_model]
    
    Args:
        d_model: Model dimension (512)
        num_layers: Number of transformer layers (4, lighter than encoder)
        num_heads: Number of attention heads (8)
        d_ff: Feed-forward dimension (2048)
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_target_patches: int = 16
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.future_position_embedding = nn.Parameter(
            torch.randn(1, max_target_patches, d_model) * 0.02
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False  # Non-causal (can attend to all positions)
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Optional: prediction head (linear projection)
        # This can help with representation quality
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict target representations from context.
        
        Args:
            context_embeddings: Context representations [B, N_context, d_model]
            target_positions: Target position indices [B, N_target]
                             These indicate which positions to predict
            attention_mask: Optional attention mask
            
        Returns:
            Predicted target representations [B, N_target, d_model]
        """
        batch_size = context_embeddings.shape[0]
        num_targets = target_positions.shape[1]
        
        future_queries = self.future_position_embedding[:, :num_targets, :]
        future_queries = future_queries.expand(batch_size, -1, -1)
        
        x = torch.cat([context_embeddings, future_queries], dim=1)
        # x: [B, N_context + N_target, d_model]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Extract only the target predictions (last N_target tokens)
        target_predictions = x[:, -num_targets:, :]
        
        # Apply prediction head
        target_predictions = self.prediction_head(target_predictions)
        
        return target_predictions
    
    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simplified forward pass when target positions are just 'next N'.
        
        Args:
            context_embeddings: Context [B, N_context, d_model]
            num_targets: Number of targets to predict
            attention_mask: Optional mask
            
        Returns:
            Predictions [B, num_targets, d_model]
        """
        batch_size = context_embeddings.shape[0]
        
        future_queries = self.future_position_embedding[:, :num_targets, :]
        future_queries = future_queries.expand(batch_size, -1, -1)
        
        # Concat
        x = torch.cat([context_embeddings, future_queries], dim=1)
        
        # Transform
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.final_norm(x)
        
        # Extract targets
        target_predictions = x[:, -num_targets:, :]
        target_predictions = self.prediction_head(target_predictions)
        
        return target_predictions


class MLPPredictor(nn.Module):
    """
    Simple MLP-based predictor (alternative to transformer).
    
    Uses a lightweight MLP to predict each target independently.
    Faster but less powerful than TransformerPredictor.
    
    Args:
        d_model: Model dimension
        num_layers: Number of MLP layers (default: 2)
        hidden_dim: Hidden dimension (default: 2 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        hidden_dim = hidden_dim or 2 * d_model
        
        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict targets from context.
        
        For MLP, we use mean pooling of context to predict each target.
        
        Args:
            context_embeddings: Context [B, N_context, d_model]
            target_positions: Ignored for MLP (can predict any number)
            
        Returns:
            Predictions [B, N_context, d_model] (same size as input)
        """
        # Mean pool context
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        # [B, 1, d_model]
        
        # Expand to match input size (predict all positions)
        num_positions = context_embeddings.shape[1]
        context_pooled = context_pooled.expand(-1, num_positions, -1)
        
        # MLP prediction
        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions
    
    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        **kwargs
    ) -> torch.Tensor:
        """Simple forward for N targets."""
        # Mean pool
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        context_pooled = context_pooled.expand(-1, num_targets, -1)
        
        # Predict
        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions