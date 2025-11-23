# src/timejepa/models/jepa_tst.py
"""
JEPA-TST: Joint-Embedding Predictive Architecture with PatchTST.

This is the main model that combines all components:
- Online Encoder (PatchTST)
- Target Encoder (EMA of online encoder)
- Predictor (Lightweight transformer)
- Decoder (For finetuning)

The model can work in two modes:
1. Pretrain mode: JEPA self-supervised learning (context → predict target)
2. Finetune mode: Supervised forecasting (input → forecast future)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Literal

from .components.revin import RevIN
from .components.patching import Patching
from .encoders.bare_encoder import BareTransformerEncoder
from .encoders.target_encoder import TargetEncoder
from .predictors.transformer_predictor import TransformerPredictor, MLPPredictor
from .decoders.linear_decoder import ForecastingHead


class JEPATST(nn.Module):
    """
    Complete JEPA-TST model for time series.
    
    Architecture:
        Pretrain:
            Input [B, L, C]
            → RevIN normalization
            → Patching [B, num_patches, d_model]
            → Context/Target masking
            → Online Encoder (context) → context_repr [B, N_ctx, d_model]
            → Target Encoder (target) → target_repr [B, N_tgt, d_model]
            → Predictor (context_repr) → pred_repr [B, N_tgt, d_model]
            → Loss: MSE(pred_repr, target_repr.detach())
        
        Finetune:
            Input [B, L_in, C]
            → RevIN normalization
            → Patching + Online Encoder → repr [B, num_patches, d_model]
            → Decoder → forecast [B, L_pred, C]
            → RevIN denormalization
            → Loss: MSE(forecast, ground_truth)
    
    Args:
        # Data params
        input_length: Length of input sequence (context_length in pretrain)
        prediction_length: Length of prediction horizon (for finetune)
        num_features: Number of input features/channels
        
        # Patching params
        patch_size: Size of each patch (16)
        stride: Stride for patching (8, allows 50% overlap)
        
        # Encoder params (shared by online and target)
        d_model: Model dimension (512)
        num_layers: Number of transformer layers in encoder (24)
        num_heads: Number of attention heads (8)
        d_ff: Feed-forward dimension (2048)
        dropout: Dropout rate
        activation: Activation function
        
        # Predictor params
        predictor_type: Type of predictor ('transformer' or 'mlp')
        predictor_num_layers: Number of layers in predictor (4)
        predictor_num_heads: Number of heads in predictor (8)
        predictor_d_ff: Feed-forward dim in predictor (2048)
        
        # Decoder params (for finetuning)
        decoder_type: Type of decoder ('linear', 'mlp', 'attentive')
        
        # EMA params
        ema_tau_base: Base EMA momentum (0.996)
        ema_tau_end: Final EMA momentum (1.0)
        
        # RevIN params
        use_revin: Whether to use RevIN normalization
        affine: Whether RevIN uses affine transformation
        subtract_last: Alternative normalization (subtract last value)
    """
    
    def __init__(
        self,
        # Data params
        input_length: int = 512,
        prediction_length: int = 96,
        num_features: int = 1,
        
        # Patching
        patch_size: int = 16,
        stride: int = 8,
        
        # Encoder
        d_model: int = 512,
        num_layers: int = 24,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        
        # Predictor
        predictor_type: Literal['transformer', 'mlp'] = 'transformer',
        predictor_num_layers: int = 4,
        predictor_num_heads: int = 8,
        predictor_d_ff: int = 2048,
        
        # Decoder
        decoder_type: Literal['linear', 'mlp', 'attentive'] = 'linear',
        
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
        self.predictor_type = predictor_type
        self.decoder_type = decoder_type
        self.use_revin = use_revin
        self.num_patches = (input_length - patch_size) // stride + 1
        
        # ===== RevIN Normalization =====
        if use_revin:
            self.revin = RevIN(
                num_features=num_features,
                affine=affine,
                subtract_last=subtract_last
            )
        else:
            self.revin = None
        
        # ===== Patching Layer =====
        self.patching = Patching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=num_features  # 🔥 Ajouter ce paramètre
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
            ema_decay=ema_tau_base
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
        
    def set_pretrain_mode(self, mode: bool = True):
        """Set whether model is in pretrain or finetune mode."""
        self._pretrain_mode = mode
        
    def is_pretrain_mode(self) -> bool:
        """Check if model is in pretrain mode."""
        return self._pretrain_mode
    
    def forward_pretrain(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA pretraining.
        
        Args:
            x: Input time series [B, L, C]
            context_mask: Boolean mask for context patches [B, num_patches]
                         True = keep, False = mask out
            target_mask: Boolean mask for target patches [B, num_patches]
                        True = predict this patch
        
        Returns:
            Dictionary with:
                - 'predictions': Predicted representations [B, N_target, d_model]
                - 'targets': Target representations [B, N_target, d_model]
                - 'context_embeddings': Context embeddings [B, N_context, d_model]
        """
        batch_size = x.shape[0]
        
        # 1. RevIN normalization
        if self.revin is not None:
            x = self.revin(x, mode='norm')
        
        # 2. Patching
        patches = self.patching(x)  # [B, num_patches, d_model]
        
        # 3. Apply context mask to get context patches
        # We zero out the masked patches (could also remove them)
        context_patches = patches.clone()
        context_patches[~context_mask] = 0  # Mask out non-context patches
        
        # 4. Encode context with online encoder
        context_embeddings = self.online_encoder(context_patches)
        # [B, num_patches, d_model] but only context positions are meaningful
        
        # Extract only context positions
        # We need to gather the valid context embeddings
        num_context = context_mask.sum(dim=1).max().item()  # Max context length in batch
        context_indices = torch.where(context_mask)
        context_emb_list = []
        for b in range(batch_size):
            b_mask = context_mask[b]
            b_context = context_embeddings[b][b_mask]  # [N_context_b, d_model]
            # Pad to num_context if needed
            if b_context.shape[0] < num_context:
                padding = torch.zeros(
                    num_context - b_context.shape[0],
                    self.d_model,
                    device=b_context.device,
                    dtype=b_context.dtype
                )
                b_context = torch.cat([b_context, padding], dim=0)
            context_emb_list.append(b_context)
        context_emb_clean = torch.stack(context_emb_list, dim=0)
        # [B, num_context, d_model]
        
        # 5. Get target patches (full patches, no masking)
        target_patches = patches  # Use full patches for target encoder
        
        # 6. Encode target with target encoder (no gradients)
        with torch.no_grad():
            target_embeddings = self.target_encoder(target_patches)
            # [B, num_patches, d_model]
        
        # Extract only target positions
        num_targets = target_mask.sum(dim=1).max().item()
        target_emb_list = []
        for b in range(batch_size):
            b_mask = target_mask[b]
            b_target = target_embeddings[b][b_mask]  # [N_target_b, d_model]
            # Pad to num_targets
            if b_target.shape[0] < num_targets:
                padding = torch.zeros(
                    num_targets - b_target.shape[0],
                    self.d_model,
                    device=b_target.device,
                    dtype=b_target.dtype
                )
                b_target = torch.cat([b_target, padding], dim=0)
            target_emb_list.append(b_target)
        target_emb_clean = torch.stack(target_emb_list, dim=0)
        # [B, num_targets, d_model]
        
        # 7. Predict target representations from context
        predictions = self.predictor.forward_simple(
            context_embeddings=context_emb_clean,
            num_targets=num_targets
        )
        # [B, num_targets, d_model]
        
        return {
            'predictions': predictions,
            'targets': target_emb_clean.detach(),
            'context_embeddings': context_emb_clean,
        }
    
    def forward_finetune(
        self,
        x: torch.Tensor,
        return_representations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised finetuning (forecasting).
        """
        # 1. RevIN normalization
        if self.revin is not None:
            x_norm = self.revin(x, mode='norm')
        else:
            x_norm = x
        
        # 2. Patching
        patches = self.patching(x_norm)  # [B, num_patches, d_model]
        
        # 3. Encode with online encoder (maintenant correct !)
        representations = self.online_encoder(patches)
        # [B, num_patches, d_model]
        
        # 4. Decode to forecasts
        forecast_norm = self.decoder(
            representations,
            denormalize=False
        )
        # [B, L_pred, C]
        
        # 5. RevIN denormalization
        if self.revin is not None:
            forecast = self.revin(forecast_norm, mode='denorm')
        else:
            forecast = forecast_norm
        
        result = {'forecast': forecast}
        if return_representations:
            result['representations'] = representations
        
        return result
    
    def forward(
        self,
        x: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass (dispatches to pretrain or finetune).
        
        Args:
            x: Input time series
            context_mask: Context mask (for pretrain mode)
            target_mask: Target mask (for pretrain mode)
        
        Returns:
            Output dictionary (depends on mode)
        """
        if self._pretrain_mode:
            if context_mask is None or target_mask is None:
                raise ValueError(
                    "context_mask and target_mask required in pretrain mode"
                )
            return self.forward_pretrain(x, context_mask, target_mask)
        else:
            return self.forward_finetune(x, **kwargs)
    
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
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'online_encoder': sum(p.numel() for p in self.online_encoder.parameters()),
            'target_encoder': sum(p.numel() for p in self.target_encoder.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    @torch.no_grad()
    def update_target_encoder(self, step: int, max_steps: int):
        """Update target encoder with EMA."""
        self.target_encoder.update(self.online_encoder, step, max_steps)
    
    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder weights."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract encoder weights
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith('online_encoder.'):
                new_key = key.replace('online_encoder.', '')
                encoder_state[new_key] = value
        
        # Load into online encoder
        self.online_encoder.load_state_dict(encoder_state)
        print(f"✅ Loaded pretrained encoder from {checkpoint_path}")
    
    def save_pretrained_encoder(self, save_path: str):
        """Save encoder weights only."""
        encoder_state = {
            f'online_encoder.{k}': v
            for k, v in self.online_encoder.state_dict().items()
        }
        torch.save(encoder_state, save_path)
        print(f"✅ Saved pretrained encoder to {save_path}")


def create_jepa_tst_tiny() -> JEPATST:
    """Create a tiny JEPA-TST for debugging/testing."""
    return JEPATST(
        input_length=128,
        prediction_length=32,
        num_features=1,
        patch_size=8,
        stride=4,
        d_model=64,
        num_layers=4,
        num_heads=4,
        d_ff=256,
        predictor_num_layers=2,
        predictor_num_heads=4,
        predictor_d_ff=256,
    )


def create_jepa_tst_small() -> JEPATST:
    """Create a small JEPA-TST."""
    return JEPATST(
        input_length=256,
        prediction_length=96,
        num_features=1,
        patch_size=16,
        stride=8,
        d_model=256,
        num_layers=12,
        num_heads=8,
        d_ff=1024,
        predictor_num_layers=3,
        predictor_num_heads=8,
        predictor_d_ff=1024,
    )


def create_jepa_tst_base() -> JEPATST:
    """Create the base JEPA-TST (as per paper specs)."""
    return JEPATST(
        input_length=512,
        prediction_length=96,
        num_features=1,
        patch_size=16,
        stride=8,
        d_model=512,
        num_layers=24,
        num_heads=8,
        d_ff=2048,
        predictor_num_layers=4,
        predictor_num_heads=8,
        predictor_d_ff=2048,
    )


def create_jepa_tst_large() -> JEPATST:
    """Create a large JEPA-TST."""
    return JEPATST(
        input_length=512,
        prediction_length=192,
        num_features=1,
        patch_size=16,
        stride=8,
        d_model=768,
        num_layers=32,
        num_heads=12,
        d_ff=3072,
        predictor_num_layers=6,
        predictor_num_heads=12,
        predictor_d_ff=3072,
    )