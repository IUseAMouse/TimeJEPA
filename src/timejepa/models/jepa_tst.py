# src/timejepa/models/jepa_tst.py
"""
JEPA-TST: Joint-Embedding Predictive Architecture for Time Series Forecasting.

This model learns to predict future representations from past context:
- Pretrain: Context → Encoder → Predictor → Predicted future repr
            Target → Target Encoder (EMA) → Target repr
            Loss: MSE(Predicted, Target)
- Finetune: Context → Encoder → Predictor → Decoder → Forecast values
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal

from .components.revin import RevIN
from .components.patching import Patching
from .encoders.bare_encoder import BareTransformerEncoder
from .encoders.target_encoder import TargetEncoder
from .predictors.transformer_predictor import TransformerPredictor, MLPPredictor
from .decoders.linear_decoder import ForecastingHead


class JEPATST(nn.Module):
    """
    JEPA-TST model for time series forecasting.

    Architecture:
        Pretrain (True Forecasting JEPA):
            Context [B, L_ctx, C] → RevIN → Patching → Online Encoder → context_repr
            Target [B, L_tgt, C] → RevIN (same stats) → Patching → Target Encoder (EMA) → target_repr
            Predictor(context_repr) → pred_repr
            Loss: MSE(pred_repr, target_repr.detach())
        
        Finetune:
            Context [B, L_ctx, C] → RevIN → Patching → Encoder → Predictor → Decoder → Forecast
            Loss: MSE(Forecast, Ground Truth)
    """

    def __init__(
        self,
        # Data params
        input_length: int = 384,
        prediction_length: int = 96,
        num_features: int = 1,
        
        # Patching
        patch_size: int = 16,
        stride: int = 8,
        
        # Encoder
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        
        # Predictor
        predictor_type: Literal['transformer', 'mlp'] = 'transformer',
        predictor_num_layers: int = 2,
        predictor_num_heads: int = 4,
        predictor_d_ff: int = 512,
        
        # Decoder
        decoder_type: Literal['linear', 'mlp', 'attentive'] = 'attentive',
        
        # EMA
        ema_tau_base: float = 0.996,
        ema_tau_end: float = 1.0,
        
        # RevIN
        use_revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        
        # Calculate number of patches for context and target
        self.num_patches = (input_length - patch_size) // stride + 1
        self.num_target_patches = (prediction_length - patch_size) // stride + 1
        
        # ===== RevIN Normalization =====
        if use_revin:
            self.revin = RevIN(
                num_features=num_features,
                affine=affine,
                subtract_last=subtract_last
            )
        else:
            self.revin = None
        
        # ===== Patching Layer (shared for context and target) =====
        self.patching = Patching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=num_features
        )
        
        # ===== Online Encoder =====
        self.online_encoder = BareTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # ===== Target Encoder (EMA of online encoder) =====
        self.target_encoder = TargetEncoder(
            encoder=self.online_encoder,
            ema_decay=ema_tau_base,
            ema_decay_end=ema_tau_end
        )
        
        # ===== Predictor =====
        if predictor_type == 'transformer':
            self.predictor = TransformerPredictor(
                d_model=d_model,
                num_layers=predictor_num_layers,
                num_heads=predictor_num_heads,
                d_ff=predictor_d_ff,
                dropout=dropout,
                activation=activation
            )
        elif predictor_type == 'mlp':
            self.predictor = MLPPredictor(
                d_model=d_model,
                num_layers=predictor_num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")
        
        # ===== Decoder (for finetuning) =====
        self.decoder = ForecastingHead(
            d_model=d_model,
            patch_size=patch_size,
            prediction_length=prediction_length,
            num_features=num_features,
            decoder_type=decoder_type,
            revin=self.revin
        )
        
        # Model state
        self._pretrain_mode = True
        
        print(f"JEPATST initialized:")
        print(f"  Context: {input_length} tp → {self.num_patches} patches")
        print(f"  Target: {prediction_length} tp → {self.num_target_patches} patches")
        
    def set_pretrain_mode(self, mode: bool = True):
        """Set whether model is in pretrain or finetune mode."""
        self._pretrain_mode = mode
        
    def is_pretrain_mode(self) -> bool:
        """Check if model is in pretrain mode."""
        return self._pretrain_mode

    def forward_pretrain(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA pretraining with TRUE forecasting objective.
        
        The model learns to predict representations of FUTURE timesteps,
        not masked patches within the same context window.
        
        Args:
            context: Past time series [B, context_length, C] (e.g., [B, 384, 1])
            target: Future time series [B, prediction_length, C] (e.g., [B, 96, 1])
        
        Returns:
            Dictionary with:
                - 'predictions': Predicted future representations [B, num_target_patches, d_model]
                - 'targets': Target encoder representations [B, num_target_patches, d_model]
                - 'context_embeddings': Context embeddings [B, num_context_patches, d_model]
        """
        # 1. RevIN normalization
        # IMPORTANT: Normalize target with SAME statistics as context
        if self.revin is not None:
            context_norm = self.revin(context, mode='norm')
            # Apply same normalization to target (using stored mean/std from context)
            target_norm = (target - self.revin.mean) / self.revin.std
        else:
            context_norm = context
            target_norm = target
        
        # 2. Patch context and target
        context_patches = self.patching(context_norm)  # [B, num_patches, d_model]
        target_patches = self.patching(target_norm)    # [B, num_target_patches, d_model]
        
        num_target_patches = target_patches.shape[1]
        
        # 3. Encode context with online encoder (gets gradients)
        context_embeddings = self.online_encoder(context_patches)
        # [B, num_patches, d_model]
        
        # 4. Encode target with target encoder (NO gradients - EMA updated)
        with torch.no_grad():
            target_embeddings = self.target_encoder(target_patches)
            # [B, num_target_patches, d_model]
        
        # 5. Predict target representations from context embeddings
        predictions = self.predictor.forward_simple(
            context_embeddings=context_embeddings,
            num_targets=num_target_patches
        )
        # [B, num_target_patches, d_model]
        
        return {
            'predictions': predictions,
            'targets': target_embeddings.detach(),
            'context_embeddings': context_embeddings,
        }

    def forward_finetune(
        self,
        context: torch.Tensor,
        return_representations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised finetuning (forecasting).
        
        Uses the learned encoder and predictor to generate future representations,
        then decodes them to actual forecast values.
        
        Args:
            context: Past time series [B, context_length, C]
            return_representations: If True, also return intermediate representations
        
        Returns:
            Dictionary with 'forecast' and 'forecast_denorm'
        """
        # 1. RevIN normalization
        if self.revin is not None:
            context_norm = self.revin(context, mode='norm')
        else:
            context_norm = context
        
        # 2. Patch context
        context_patches = self.patching(context_norm)  # [B, num_patches, d_model]
        
        # 3. Encode context with online encoder
        context_embeddings = self.online_encoder(context_patches)
        # [B, num_patches, d_model]
        
        # 4. Predict future representations
        predictions = self.predictor.forward_simple(
            context_embeddings=context_embeddings,
            num_targets=self.num_target_patches
        )
        # [B, num_target_patches, d_model]
        
        # 5. Decode to forecast values
        forecast, forecast_denorm = self.decoder(predictions)
        # forecast: [B, prediction_length, C] (normalized)
        # forecast_denorm: [B, prediction_length, C] (original scale)
        
        result = {
            'forecast': forecast,
            'forecast_denorm': forecast_denorm
        }
        
        if return_representations:
            result['context_embeddings'] = context_embeddings
            result['future_representations'] = predictions
        
        return result

    def forecast(
        self, 
        context: torch.Tensor,
        n: Optional[int] = None,
        return_representations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forecast n steps ahead with automatic rolling if needed.
        
        Args:
            context: Past time series [B, context_length, C]
            n: Number of steps to forecast. Defaults to self.prediction_length.
               - If n <= prediction_length: single forward pass, truncate output
               - If n > prediction_length: rolling forecast with m passes where
                 m = ceil(n / prediction_length)
            return_representations: If True, return intermediate representations
                (only meaningful when n <= prediction_length)
            
        Returns:
            Dictionary with:
                - 'forecast': Normalized forecast [B, n, C]
                - 'forecast_denorm': Denormalized forecast [B, n, C]
                - 'context_embeddings', 'future_representations' (if return_representations 
                  and n <= prediction_length)
        """
        if n is None:
            n = self.prediction_length
        
        # Case 1: n <= native prediction length - single pass with truncation
        if n <= self.prediction_length:
            result = self.forward_finetune(context, return_representations=return_representations)
            result['forecast'] = result['forecast'][:, :n]
            result['forecast_denorm'] = result['forecast_denorm'][:, :n]
            return result
        
        # Case 2: n > prediction_length - rolling forecast
        # Calculate number of rolls: m such that m * prediction_length >= n
        num_rolls = (n + self.prediction_length - 1) // self.prediction_length
        
        all_forecasts_denorm = []
        current_context = context.clone()
        
        for roll_idx in range(num_rolls):
            # Forward pass with current context
            result = self.forward_finetune(current_context, return_representations=False)
            all_forecasts_denorm.append(result['forecast_denorm'])
            
            # Prepare context for next roll (if not the last one)
            if roll_idx < num_rolls - 1:
                # Shift context window: drop oldest prediction_length values, 
                # append the new predictions
                current_context = torch.cat([
                    current_context[:, self.prediction_length:, :],
                    result['forecast_denorm']
                ], dim=1)
        
        # Concatenate all forecasts and truncate to exactly n steps
        forecast_denorm = torch.cat(all_forecasts_denorm, dim=1)[:, :n]
        
        # For normalized forecast in rolling mode, we return the denorm as placeholder
        # (true normalized values would require re-normalizing with original context stats)
        return {
            'forecast': forecast_denorm,  # Note: not truly normalized in rolling mode
            'forecast_denorm': forecast_denorm
        }

    def forward(
        self,
        context: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass (dispatches to pretrain or finetune).
        
        Args:
            context: Past time series [B, context_length, C]
            target: Future time series [B, prediction_length, C] (required for pretrain)
        
        Returns:
            Output dictionary (depends on mode)
        """
        if self._pretrain_mode:
            if target is None:
                raise ValueError("target is required in pretrain mode")
            return self.forward_pretrain(context, target)
        else:
            return self.forward_finetune(context, **kwargs)

    # ===== Freezing/Unfreezing methods (unchanged) =====
    
    def freeze_encoder(self):
        """Freeze encoder parameters (for finetuning)."""
        for param in self.online_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.online_encoder.parameters():
            param.requires_grad = True

    def freeze_target_encoder(self):
        """Freeze target encoder (should always be frozen)."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def freeze_predictor(self):
        """Freeze predictor for finetuning."""
        for param in self.predictor.parameters():
            param.requires_grad = False

    def unfreeze_predictor(self):
        """Unfreeze the predictor."""
        for param in self.predictor.parameters():
            param.requires_grad = True

    def freeze_patching(self):
        """Freeze patching layers."""
        for param in self.patching.parameters():
            param.requires_grad = False
    
    def unfreeze_patching(self):
        """Unfreeze patching layer."""
        for param in self.patching.parameters():
            param.requires_grad = True

    def freeze_revin(self):
        """Freeze revin layers."""
        if self.revin is not None:
            for param in self.revin.parameters():
                param.requires_grad = False

    def unfreeze_revin(self):
        """Unfreeze revin layers."""
        if self.revin is not None:
            for param in self.revin.parameters():
                param.requires_grad = True

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'online_encoder': sum(p.numel() for p in self.online_encoder.parameters()),
            'target_encoder': sum(p.numel() for p in self.target_encoder.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'patching': sum(p.numel() for p in self.patching.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    @torch.no_grad()
    def update_target_encoder(self, step: int, max_steps: int):
        """Update target encoder with EMA."""
        self.target_encoder.update(self.online_encoder, step, max_steps)

    def save_pretrained_encoder(self, save_path: str):
        """Save encoder and predictor weights for finetuning."""
        state = {
            'online_encoder': self.online_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'patching': self.patching.state_dict(),
        }
        if self.revin is not None:
            state['revin'] = {
                k: v for k, v in self.revin.state_dict().items()
                if not k.endswith('.mean') and not k.endswith('.std')
            }
        torch.save(state, save_path)
        print(f"✅ Saved pretrained encoder + predictor to {save_path}")