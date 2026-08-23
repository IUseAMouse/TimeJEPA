"""
Linear Decoder for generative finetuning.

After JEPA pretraining, we add a simple decoder head to generate
actual time series values for forecasting tasks.

Architecture: Representation [B, N, d_model] → Values [B, L_pred, C]
"""

import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

from ..components.patching import UnPatching


class LinearDecoder(nn.Module):
    """
    Simple linear decoder for forecasting.
    
    Projects patch-level representations back to time series values.
    Used during finetuning for generative tasks.
    
    Architecture:
        Patch representations [B, num_patches, d_model]
        → Linear projection [B, num_patches, patch_size * C]
        → Reshape to [B, L_pred, C]
    
    Args:
        d_model: Model dimension (512)
        patch_size: Size of each patch (16)
        stride: Gap between patches (8)
        prediction_length: Length of prediction horizon
        num_features: Number of output features/channels (1 for univariate)
        use_unpatching: Whether to use UnPatching layer
    """
    
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        use_unpatching: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.use_unpatching = use_unpatching
        
        if use_unpatching:
            # Use proper unpatching with overlap handling
            self.unpatching = UnPatching(
                patch_size=patch_size,
                stride=stride,
                d_model=d_model,
                num_features=num_features
            )
        else:
            # Simple linear projection
            self.projection = nn.Linear(d_model, patch_size * num_features)
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode representations to time series values.
        
        Args:
            x: Patch representations [B, num_patches, d_model]
            target_length: Target sequence length (default: prediction_length)
            
        Returns:
            Time series predictions [B, L_pred, C]
        """
        if target_length is None:
            target_length = self.prediction_length
        
        if self.use_unpatching:
            # Use unpatching layer
            output = self.unpatching(x, target_len=target_length)
        else:
            # Simple projection and reshape
            batch_size, num_patches, _ = x.shape
            
            # Project to patch values
            x = self.projection(x)  # [B, num_patches, patch_size * C]
            
            # Reshape
            x = x.reshape(batch_size, num_patches * self.patch_size, self.num_features)
            
            # Trim to target length
            output = x[:, :target_length, :]
        
        return output


class MLPDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.prediction_length = prediction_length
        self.num_features = num_features
        
        hidden_dim = hidden_dim or d_model
        
        # MLP projette vers d_model (pas directement vers patch_size * C)
        layers = []
        for i in range(num_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # UnPatching gère la projection finale + overlap
        self.unpatching = UnPatching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=num_features
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        if target_length is None:
            target_length = self.prediction_length
        
        # MLP enrichit les représentations
        x = self.mlp(x)  # [B, num_patches, d_model]
        
        # UnPatching gère le reste proprement
        output = self.unpatching(x, target_len=target_length)
        
        return output


class AttentiveDecoder(nn.Module):
    """
    Attention-based decoder for more sophisticated decoding.
    
    Uses cross-attention between learnable query embeddings and
    patch representations to generate predictions.
    
    Inspired by Perceiver and similar architectures.
    
    Args:
        d_model: Model dimension
        prediction_length: Prediction horizon
        num_features: Number of output features
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        prediction_length: int = 96,
        num_features: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.num_features = num_features
        
        # Learnable query embeddings for each output timestep
        self.query_embeddings = nn.Parameter(
            torch.randn(1, prediction_length, d_model) * 0.02
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_features)
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode with cross-attention.
        
        Args:
            x: Patch representations [B, num_patches, d_model] (keys/values)
            target_length: Target length
            
        Returns:
            Predictions [B, L_pred, C]
        """
        if target_length is None:
            target_length = self.prediction_length
        
        batch_size = x.shape[0]
        
        # Get query embeddings
        queries = self.query_embeddings[:, :target_length, :].expand(batch_size, -1, -1)
        
        # Cross-attention: queries attend to patch representations
        attended, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x
        )
        
        # Norm and project
        attended = self.norm(attended)
        output = self.output_proj(attended)  # [B, L_pred, C]
        
        return output


class ForecastingHead(nn.Module):
    """
    Complete forecasting head with multiple decoder options.
    
    Wraps different decoder types and provides a unified interface.
    Can also handle denormalization via RevIN.
    
    Args:
        d_model: Model dimension
        patch_size: Patch size
        prediction_length: Prediction horizon
        num_features: Number of features
        decoder_type: Type of decoder ('linear', 'mlp', 'attentive')
        revin: Optional RevIN layer for denormalization
    """
    
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        stride: int = 8,
        prediction_length: int = 96,
        num_features: int = 1,
        decoder_type: str = 'linear',
        revin: Optional[nn.Module] = None,
        quantile_levels: Optional[Sequence[float]] = None,
        quantile_use_context: bool = True,
        # ESJEPA — transmis à QuantileHead (gate d'étalement). Flag off ⇒
        # state_dict et comportement bit-identiques.
        error_signal: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.prediction_length = prediction_length
        self.decoder_type = decoder_type
        self.revin = revin
        
        # Create decoder
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'mlp':
            self.decoder = MLPDecoder(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'attentive':
            self.decoder = AttentiveDecoder(
                d_model=d_model,
                prediction_length=prediction_length,
                num_features=num_features
            )
        elif decoder_type == 'quantile':
            # Probabilistic head. Emits a sorted quantile fan instead of a point,
            # and optionally cross-attends to the context embeddings.
            from .quantile_head import QuantileHead, DEFAULT_QUANTILES
            self.decoder = QuantileHead(
                d_model=d_model,
                patch_size=patch_size,
                stride=stride,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels or DEFAULT_QUANTILES,
                use_context=quantile_use_context,
                use_error_signal=error_signal,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        if error_signal and decoder_type != 'quantile':
            # Un z entraîné au pretrain puis silencieusement inconsommé par un
            # décodeur point est exactement la dégradation que le projet refuse.
            raise ValueError(
                f"error_signal=True exige decoder_type='quantile' (la voie z "
                f"module l'étalement du fan) — reçu '{decoder_type}'."
            )

        self.output_norm = nn.LayerNorm(num_features)

    @property
    def is_probabilistic(self) -> bool:
        return self.decoder_type == 'quantile'

    def forward(
        self,
        x: torch.Tensor,
        skip_revin: bool = False,
        context_embeddings: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate forecasts.

        Args:
            x: Representations [B, num_patches, d_model]
            skip_revin: Return the normalized forecast unchanged as the "denorm" one
            context_embeddings: Encoder output [B, N_ctx, d_model]. Only the
                quantile head consumes it; the point decoders ignore it, so the
                caller can always pass it.
            z: ESJEPA — stats du résidu prédites [B, N, z_dim]. Contrairement à
                context_embeddings, un z fourni à un décodeur point est REFUSÉ :
                il n'existe que si l'arm est actif, le perdre serait silencieux.

        Returns:
            For point decoders: (forecast [B, L, C], forecast_denorm [B, L, C])
            For the quantile head: (quantiles [B, L, Q], quantiles_denorm [B, L, Q])
            — the caller extracts the median.
        """
        if self.decoder_type == 'quantile':
            predictions = self.decoder(x, context_embeddings=context_embeddings, z=z)
        else:
            if z is not None:
                raise ValueError(
                    "z (ESJEPA) reçu par un décodeur point — la voie z module "
                    "un fan quantile, decoder_type='quantile' requis."
                )
            predictions = self.decoder(x)

        if skip_revin or self.revin is None:
            predictions_denorm = predictions
        else:
            # `denormalize_target_space`, NOT `mode='denorm'`: the decoder is
            # trained against a plain z-scored target (no RevIN affine), so the
            # affine inverse in `_denormalize` would introduce a scale/offset
            # error. See RevIN.denormalize_target_space for the full rationale.
            predictions_denorm = self.revin.denormalize_target_space(predictions)

        return predictions, predictions_denorm