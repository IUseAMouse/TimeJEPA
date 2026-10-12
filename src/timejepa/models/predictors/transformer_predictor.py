"""
Predictors for JEPA: predict target representations from context
representations. The predictor is kept lighter than the encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..components.attention import TransformerBlock


class TransformerPredictor(nn.Module):
    """
    Transformer predictor for JEPA.

    Architecture:
        Context embeddings [B, N_context, d_model]
        -> learned future queries [B, N_target, d_model]
        -> Concat [B, N_context + N_target, d_model]
        -> Transformer blocks
        -> Extract target predictions [B, N_target, d_model]

    Args:
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
        max_target_patches: Size of the future-query table
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_target_patches: int = 16,
        # Scale conditioning w = k2/k1 (cross-resolution). Opt-in: without
        # the flag the w_film attribute does not exist.
        use_w_film: bool = False,
        # z path (residual statistics). Same opt-in contract: without the
        # flag the z_head attribute does not exist.
        error_signal: bool = False,
        z_dim: int = 4,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        if use_w_film:
            # Residual FiLM on the future queries:
            # q * (1 + gamma(log2 w)) + beta(log2 w). Zero-init weights and
            # bias give an exact identity for any w at init, so early
            # training behaves like the baseline.
            self.w_film = nn.Linear(1, 2 * d_model)
            nn.init.zeros_(self.w_film.weight)
            nn.init.zeros_(self.w_film.bias)

        if error_signal:
            # z head on the predictor trunk: reads the target tokens after
            # final_norm (before prediction_head) and predicts per-patch
            # residual statistics [B, N_target, z_dim]. Its gradients flow
            # back into the trunk and the encoder by design.
            self.z_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, z_dim),
            )

        self.future_position_embedding = nn.Parameter(
            torch.randn(1, max_target_patches, d_model) * 0.02
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)

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

        future_queries = self._future_queries(batch_size, num_targets)

        x = torch.cat([context_embeddings, future_queries], dim=1)
        # x: [B, N_context + N_target, d_model]

        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)

        # Only the target predictions (last N_target tokens)
        target_predictions = x[:, -num_targets:, :]
        target_predictions = self.prediction_head(target_predictions)
        
        return target_predictions
    
    def _future_queries(self, batch_size: int, num_targets: int) -> torch.Tensor:
        """
        Fetch `num_targets` learned future queries, refusing to truncate.

        Slicing past the table size silently returns fewer rows, and the
        target slice `x[:, -num_targets:]` would then read context
        embeddings in place of the missing queries.
        """
        available = self.future_position_embedding.shape[1]
        if num_targets > available:
            raise ValueError(
                f"TransformerPredictor was built with max_target_patches="
                f"{available} but {num_targets} target patches were requested. "
                f"Increase max_target_patches (JEPATST sizes it from "
                f"prediction_length). Truncating here would silently feed "
                f"context embeddings in place of predictions."
            )
        return self.future_position_embedding[:, :num_targets, :].expand(batch_size, -1, -1)

    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        attention_mask: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        return_z: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass when the target positions are the next N patches.

        `w` (optional, [B]): scale ratio k2/k1 per item, since resolution is
        drawn per item. `return_z`: if True, also returns z_pred
        [B, num_targets, z_dim] from the z head; raises if the predictor was
        built without error_signal.

        Args:
            context_embeddings: Context [B, N_context, d_model]
            num_targets: Number of targets to predict
            attention_mask: Optional mask

        Returns:
            Predictions [B, num_targets, d_model]
            (or the (predictions, z_pred) tuple if return_z=True)
        """
        if return_z and not hasattr(self, 'z_head'):
            raise ValueError(
                "return_z=True but the predictor was built without "
                "error_signal - the ESJEPA arm requires "
                "model.error_signal=true at construction."
            )
        batch_size = context_embeddings.shape[0]

        future_queries = self._future_queries(batch_size, num_targets)

        # Per-item scale conditioning (w = k2/k1, [B]).
        if w is not None:
            if not hasattr(self, 'w_film'):
                # Refuse rather than silently drop the conditioning.
                if bool((w != 1).any()):
                    raise ValueError(
                        "w != 1 received but the predictor was built without "
                        "use_w_film - the cross-resolution arm requires the "
                        "model to be built with cross_resolution=true."
                    )
            else:
                film = self.w_film(
                    torch.log2(w.to(future_queries.dtype)).unsqueeze(-1))
                gamma, beta = film.chunk(2, dim=-1)              # [B, d] each
                future_queries = (
                    future_queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1))

        x = torch.cat([context_embeddings, future_queries], dim=1)

        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)

        trunk_targets = x[:, -num_targets:, :]
        target_predictions = self.prediction_head(trunk_targets)

        if return_z:
            # z is read from the shared trunk, not from prediction_head.
            return target_predictions, self.z_head(trunk_targets)
        return target_predictions


class MLPPredictor(nn.Module):
    """
    MLP predictor over the mean-pooled context.

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
        
        layers = []
        for i in range(num_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict targets from the mean-pooled context.

        Args:
            context_embeddings: Context [B, N_context, d_model]
            target_positions: Ignored for MLP (can predict any number)
            
        Returns:
            Predictions [B, N_context, d_model] (same size as input)
        """
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        # [B, 1, d_model]

        num_positions = context_embeddings.shape[1]
        context_pooled = context_pooled.expand(-1, num_positions, -1)

        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions
    
    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        w=None,
        **kwargs
    ) -> torch.Tensor:
        """Forward for N targets."""
        # The MLP has neither order nor queries, so scale conditioning has
        # nowhere to apply; refuse rather than let **kwargs swallow `w`.
        if w is not None and bool((w != 1).any()):
            raise NotImplementedError(
                "MLPPredictor does not support w conditioning (G9.2) - "
                "use predictor_type='transformer'."
            )
        # Likewise **kwargs must not swallow return_z.
        if kwargs.get('return_z', False):
            raise NotImplementedError(
                "MLPPredictor does not support the z path (ESJEPA) - "
                "use predictor_type='transformer'."
            )
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        context_pooled = context_pooled.expand(-1, num_targets, -1)

        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions