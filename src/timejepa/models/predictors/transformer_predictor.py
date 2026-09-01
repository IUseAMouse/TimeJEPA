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
        -> Positional tokens for targets [B, N_target, d_model]
        -> Concat [B, N_context + N_target, d_model]
        -> Transformer blocks (4 layers)
        -> Extract target predictions [B, N_target, d_model]
    
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
        max_target_patches: int = 16,
        # G9.2 - scale conditioning w = k2/k1 (cross-resolution JEPA).
        # OPT-IN AT CONSTRUCTION: without this flag the w_film attribute does
        # not exist, so the state_dict of all existing configs is
        # bit-identical (their checkpoints reload unchanged).
        use_w_film: bool = False,
        # ESJEPA - z path (residual statistics, conditional
        # heteroscedasticity). Same opt-in contract as use_w_film: flag off
        # => the z_head attribute does not exist, state_dict bit-identical.
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
            # q * (1 + gamma(log2 w)) + beta(log2 w). Weights AND bias
            # initialized to ZERO -> gamma=beta=0 -> exact identity for any w
            # at init. Two intended consequences: (a) the arm's early training
            # behaves like the baseline, conditioning only appears if the
            # gradient asks for it; (b) an xres checkpoint reloaded WITHOUT
            # passing w (finetune, forecast) is exactly the model at w=1 - no
            # never-seen regime.
            self.w_film = nn.Linear(1, 2 * d_model)
            nn.init.zeros_(self.w_film.weight)
            nn.init.zeros_(self.w_film.bias)

        if error_signal:
            # ESJEPA - z head on the predictor TRUNK: reads the target tokens
            # post-final_norm (BEFORE prediction_head, which is the signal
            # path's head) and predicts per-patch residual statistics
            # [B, N_target, z_dim]. The z-loss gradients flow back into the
            # trunk and the encoder: that is the intended mechanism - the
            # representation learns to keep dispersion information - dosed by
            # lambda_z on the module side. No BatchNorm (the target encoder's
            # EMA update skips num_batches_tracked; unrelated here but the
            # constraint is family-wide: LayerNorm only).
            self.z_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, z_dim),
            )

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

        future_queries = self._future_queries(batch_size, num_targets)

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
    
    def _future_queries(self, batch_size: int, num_targets: int) -> torch.Tensor:
        """
        Fetch `num_targets` learned future queries, refusing to truncate.

        Slicing `future_position_embedding[:, :num_targets]` past the table size
        silently returns fewer rows. The output shape stayed correct downstream
        because the target slice `x[:, -num_targets:]` reads from the
        concatenated sequence - so the missing queries were quietly replaced by
        the LAST CONTEXT EMBEDDINGS, and those were then trained and scored as
        if they were predictions.

        With patch=16 / stride=8 that silently corrupted every configuration
        with prediction_length > 136 (e.g. large.yaml at 192 -> 23 target
        patches, 7 of them fake), and base.yaml at patch=4 / stride=4 -> 32
        target patches, half of them fake.
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
        Simplified forward pass when target positions are just 'next N'.

        `w` (optional, [B]): scale ratio k2/k1 per ITEM (G9.2) - per item and
        not per batch, because resolution is drawn per item while geometry
        randomization is per batch; the two coexist.

        `return_z` (ESJEPA): if True, ALSO returns z_pred [B, num_targets,
        z_dim] - the residual statistics predicted by the z head. Loud
        refusal if the predictor was built without error_signal (a silently
        absent z is an arm that thinks it modulates its quantiles and
        modulates nothing). Flag off: signature and return unchanged.

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

        # G9.2 - per-item scale conditioning (w = k2/k1, [B]).
        if w is not None:
            if not hasattr(self, 'w_film'):
                # Refuse rather than ignore: a silently lost w is a
                # cross-resolution arm training WITHOUT conditioning and
                # numbers believed to be conditioned.
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

        # Concat
        x = torch.cat([context_embeddings, future_queries], dim=1)
        
        # Transform
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.final_norm(x)

        # Extract targets
        trunk_targets = x[:, -num_targets:, :]
        target_predictions = self.prediction_head(trunk_targets)

        if return_z:
            # ESJEPA - z read from the shared trunk (post-final_norm), not
            # from prediction_head's output: the two paths fork here.
            return target_predictions, self.z_head(trunk_targets)
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
        w=None,
        **kwargs
    ) -> torch.Tensor:
        """Simple forward for N targets."""
        # The MLP mean-pools the context: it has neither order nor queries,
        # so nowhere for scale conditioning to make sense. Without this
        # guard, **kwargs swallowed `w` silently - the cross-resolution arm
        # would have "run" without conditioning.
        if w is not None and bool((w != 1).any()):
            raise NotImplementedError(
                "MLPPredictor does not support w conditioning (G9.2) - "
                "use predictor_type='transformer'."
            )
        # Same guard family for ESJEPA: **kwargs must not swallow return_z
        # silently (a z never produced = quantiles never modulated).
        if kwargs.get('return_z', False):
            raise NotImplementedError(
                "MLPPredictor does not support the z path (ESJEPA) - "
                "use predictor_type='transformer'."
            )
        # Mean pool
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        context_pooled = context_pooled.expand(-1, num_targets, -1)
        
        # Predict
        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions