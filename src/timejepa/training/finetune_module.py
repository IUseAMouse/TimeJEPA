# src/timejepa/training/finetune_module.py
"""
PyTorch Lightning Module for supervised finetuning.

Finetunes pretrained JEPA encoder for time series forecasting.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
import logging

from ..models.jepa_tst import JEPATST
from .utils.metrics import compute_forecasting_metrics, mse, mae
from .utils.masking import get_masking_strategy

logger = logging.getLogger(__name__)


class FinetuneModule(pl.LightningModule):
    """
    Lightning Module for supervised finetuning.
    
    Training modes:
        - 'linear_probe': Freeze encoder, train only decoder (fast, good baseline)
        - 'full_finetune': Train encoder + decoder (best performance)
        - 'gradual_unfreeze': Start frozen, gradually unfreeze layers
    
    Workflow:
        1. Load pretrained encoder weights
        2. Switch model to finetune mode
        3. Apply freezing strategy
        4. Train with supervised forecasting loss
    """
    
    def __init__(
        self,
        model: JEPATST,
        pretrained_encoder_path: Optional[Path] = None,
        
        # Finetuning strategy
        masking_strategy: Literal['random', 'block', 'temporal', 'random_temporal'] = 'temporal',
        finetune_mode: Literal['linear_probe', 'full_finetune', 'gradual_unfreeze'] = 'full_finetune',
        unfreeze_after_epoch: int = 5,  # For gradual_unfreeze
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
        masking_kwargs: Optional[Dict[str, Any]] = None,
        
        # Loss
        loss_type: Literal['mse', 'mae', 'huber'] = 'mse',
        huber_delta: float = 1.0,
        
        # Multi-horizon (optional)
        prediction_horizons: Optional[List[int]] = None,
        horizon_weights: Optional[List[float]] = None,
        
        # Optimizer
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,  # Lower LR for pretrained encoder
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        
        # LR Scheduler
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        lr_scheduler: Literal['cosine', 'linear', 'plateau', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Regularization
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        """
        Args:
            model: JEPATST model instance
            pretrained_encoder_path: Path to pretrained encoder weights (.pt file)
            finetune_mode: Strategy for finetuning
            unfreeze_after_epoch: When to unfreeze encoder (for gradual_unfreeze)
            loss_type: Loss function for forecasting
            huber_delta: Delta for Huber loss
            prediction_horizons: List of horizons for multi-horizon training
            horizon_weights: Weights for each horizon loss
            learning_rate: Peak learning rate for decoder
            encoder_lr_multiplier: LR multiplier for encoder (typically < 1.0)
            weight_decay: AdamW weight decay
            betas: Adam beta parameters
            warmup_epochs: Warmup epochs
            max_epochs: Total epochs
            lr_scheduler: LR schedule type
            min_lr: Minimum LR
            dropout: Dropout rate
            label_smoothing: Label smoothing (not typically used for regression)
            log_every_n_steps: Logging frequency
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        masking_kwargs = masking_kwargs or {}
        self.masking_fn = get_masking_strategy(
            strategy_name=masking_strategy,
            num_patches=model.num_patches,
            context_ratio=context_ratio,
            target_ratio=target_ratio,
            **masking_kwargs
        )


        # Switch to finetune mode
        self.model.set_pretrain_mode(False)
        logger.info("✓ Model switched to finetune mode")
        
        # Load pretrained encoder if provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)
        
        # Apply finetuning strategy
        self.finetune_mode = finetune_mode
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._apply_finetune_strategy(finetune_mode)
        
        # Loss configuration
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        # Multi-horizon setup
        self.prediction_horizons = prediction_horizons
        self.horizon_weights = horizon_weights
        if prediction_horizons and horizon_weights:
            assert len(prediction_horizons) == len(horizon_weights), \
                "prediction_horizons and horizon_weights must have same length"
        
        # Optimizer params
        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
        
        # Metrics tracking
        self.val_metrics_history = []
    
    def load_pretrained_encoder(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("model.", "").replace("_orig_mod.", "")
            
            # Exclure target_encoder (pas utilisé en finetune)
            if "target_encoder" in clean_key:
                continue
            
            # Exclure les buffers runtime de RevIN
            if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                continue
            
            cleaned_state_dict[clean_key] = v
        
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        
        # Définir ce qui est "attendu" comme missing
        def is_expected_missing(k):
            if 'decoder' in k:
                return True  # Nouveau pour finetune
            if 'target_encoder' in k:
                return True  # Pas chargé, pas utilisé
            if 'revin' in k and (k.endswith('.mean') or k.endswith('.std')):
                return True  # Buffers runtime
            return False
        
        critical_missing = [k for k in missing if not is_expected_missing(k)]
        
        logger.info(f"📦 Loaded {len(cleaned_state_dict)} keys from checkpoint")
        logger.info(f"   Expected missing (decoder, target_encoder, buffers): {len(missing) - len(critical_missing)}")
        
        if critical_missing:
            logger.error(f"❌ CRITICAL missing: {critical_missing}")
            raise RuntimeError(f"Failed to load: {critical_missing}")
        
        if unexpected:
            logger.warning(f"⚠️ Unexpected keys: {unexpected}")
        
        logger.info("✓ Pretrained weights loaded successfully")

        # Avant le load, inspecte ce que contient le checkpoint
        print("🔍 Keys in checkpoint:")
        for k in sorted(cleaned_state_dict.keys()):
            print(f"   {k}")

        print("\n🔍 Keys in model:")
        for k in sorted(self.model.state_dict().keys()):
            print(f"   {k}")
        
        logger.info("✓ All pretrained components loaded successfully")
    
    def _apply_finetune_strategy(self, mode: str):
        """Apply freezing strategy based on finetune mode."""
        if mode == 'linear_probe':
            # Freeze encoder completely
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            self.model.freeze_revin()
            self.model.freeze_target_encoder()
            logger.info("✓ Applied LINEAR PROBE: encoder frozen, decoder trainable")
        
        elif mode == 'full_finetune':
            # Unfreeze everything
            self.model.unfreeze_encoder()
            logger.info("✓ Applied FULL FINETUNE: encoder + decoder trainable")
        
        elif mode == 'gradual_unfreeze':
            # Start frozen, will unfreeze later
            self.model.freeze_encoder()
            logger.info(f"✓ Applied GRADUAL UNFREEZE: starting frozen, "
                       f"will unfreeze after epoch {self.unfreeze_after_epoch}")
        
        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")
    
    def on_train_epoch_start(self):
        """Handle gradual unfreezing."""
        if self.finetune_mode == 'gradual_unfreeze':
            if self.current_epoch == self.unfreeze_after_epoch:
                logger.info(f"Epoch {self.current_epoch}: Unfreezing encoder")
                self.model.unfreeze_encoder()
    
    def forward(self, x: torch.Tensor, prediction_length: Optional[int] = None):
        """Forward pass for forecasting."""
        return self.model.forecast(x, prediction_length=prediction_length)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute forecasting loss.
        
        Args:
            predictions: [B, L, C]
            targets: [B, L, C]
        
        Returns:
            Loss scalar
        """
        if self.loss_type == 'mse':
            return mse(predictions, targets)
        elif self.loss_type == 'mae':
            return mae(predictions, targets)
        elif self.loss_type == 'huber':
            return nn.functional.huber_loss(predictions, targets, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary with 'context' (input) and 'target' (ground truth)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        # Get context and target
        context = batch['context']  # [B, L_context, C]
        target = batch['target']    # [B, L_target, C]
        batch_size = context.shape[0]

        context_mask, _ = self.masking_fn(batch_size, device=context.device)
        
        # Add channel dim if univariate
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        results = self.model.forecast(context, context_mask)
        predictions_norm = results['forecast']
        predictions_denorm = results['forecast_denorm']
        target_norm = self.model.revin(target, mode='norm')
        # predictions: [B, L_target, C]
        
        # Compute loss
        loss = self.compute_loss(predictions_norm, target_norm)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Additional metrics every N steps
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                metrics = compute_forecasting_metrics(predictions_norm, target_norm)
                for key, value in metrics.items():
                    self.log(f'train/{key}', value, on_step=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        context = batch['context']
        target = batch['target']
        batch_size = context.shape[0]
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        context_mask, _ = self.masking_fn(batch_size, device=context.device)
        
        results = self.model.forecast(context, context_mask)
        predictions_norm = results['forecast']
        target_norm = self.model.revin(target, mode='norm')
        
        # Compute loss
        loss = self.compute_loss(predictions_norm, target_norm)

        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Compute all metrics
        metrics = compute_forecasting_metrics(predictions_norm, target_norm)
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        context = batch['context']
        target = batch['target']
        batch_size = context.shape[0]
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        context_mask, _ = self.masking_fn(batch_size, device=context.device)

        results = self.model.forecast(context, context_mask)
        predictions_norm = results['forecast']
        target_norm = self.model.revin(target, mode='norm')
        
        # Compute loss
        loss = self.compute_loss(predictions_norm, target_norm)
        
        # Log
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Compute all metrics
        metrics = compute_forecasting_metrics(predictions_norm, target_norm)
        for key, value in metrics.items():
            self.log(f'test/{key}', value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            'loss': loss,
            'predictions': predictions_norm,
            'targets': target_norm,
            'metrics': metrics
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer with different LR for encoder vs decoder.
        
        Pretrained encoder gets lower LR to preserve learned representations.
        """
        # Separate parameters: encoder vs rest
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen params
            
            if 'decoder' in name and 'target_encoder' not in name:
                # Online encoder
                encoder_params.append(param)
            else:
                # Decoder, predictor, etc.
                decoder_params.append(param)
        
        # Create parameter groups with different LRs
        param_groups = []
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.learning_rate * self.encoder_lr_multiplier,
                'name': 'encoder'
            })
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.learning_rate,
                'name': 'decoder'
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        logger.info(f"Optimizer groups: {[g['name'] for g in param_groups]}")
        logger.info(f"Encoder LR: {self.learning_rate * self.encoder_lr_multiplier:.2e}")
        logger.info(f"Decoder LR: {self.learning_rate:.2e}")
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,  # Default LR (overridden by groups)
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'constant':
            return optimizer
        
        # Estimate steps
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) \
            if hasattr(self.trainer, 'datamodule') else 1000
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        if self.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.min_lr
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        elif self.lr_scheduler == 'linear':
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.min_lr / self.learning_rate,
                total_iters=total_steps - warmup_steps
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        elif self.lr_scheduler == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.min_lr,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
    
    def on_train_epoch_end(self):
        """Log learning rates at epoch end."""
        optimizer = self.optimizers()
        
        # Log LR for each parameter group
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.log(f'lr/{group_name}', param_group['lr'], on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """Track validation metrics."""
        # Get current val metrics
        if 'val/mse' in self.trainer.callback_metrics:
            val_mse = self.trainer.callback_metrics['val/mse'].item()
            self.val_metrics_history.append(val_mse)
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        context = batch['context']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        
        # Use model's default prediction length or from batch
        prediction_length = batch.get('prediction_length', None)
        predictions = self.model.forecast(context, prediction_length=prediction_length)
        
        return predictions


class MultiHorizonFinetuneModule(FinetuneModule):
    """
    Extended finetuning module for multi-horizon forecasting.
    
    Trains model to predict at multiple time scales simultaneously.
    """
    
    def __init__(
        self,
        model: JEPATST,
        prediction_horizons: List[int],
        horizon_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Args:
            model: JEPATST model
            prediction_horizons: List of horizons to predict (e.g., [24, 48, 96])
            horizon_weights: Loss weight for each horizon (default: equal weights)
            **kwargs: Arguments for FinetuneModule
        """
        # Set horizons before calling parent init
        if horizon_weights is None:
            horizon_weights = [1.0] * len(prediction_horizons)
        
        super().__init__(
            model=model,
            prediction_horizons=prediction_horizons,
            horizon_weights=horizon_weights,
            **kwargs
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for multi-horizon."""
        context = batch['context']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        
        # Check if batch has multiple targets
        if 'targets' in batch:
            # Multi-horizon dataset
            targets_dict = batch['targets']
        else:
            # Single horizon - create dict
            target = batch['target']
            if target.ndim == 2:
                target = target.unsqueeze(-1)
            targets_dict = {target.shape[1]: target}
        
        # Compute loss for each horizon
        total_loss = 0.0
        horizon_losses = {}
        
        for i, horizon in enumerate(self.prediction_horizons):
            if horizon not in targets_dict:
                continue
            
            target = targets_dict[horizon]
            prediction = self.model.forecast(context, prediction_length=horizon)
            
            loss = self.compute_loss(prediction, target)
            weight = self.horizon_weights[i]
            
            total_loss += weight * loss
            horizon_losses[f'train/loss_h{horizon}'] = loss.item()
        
        # Log
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for key, value in horizon_losses.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step for multi-horizon."""
        context = batch['context']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        
        if 'targets' in batch:
            targets_dict = batch['targets']
        else:
            target = batch['target']
            if target.ndim == 2:
                target = target.unsqueeze(-1)
            targets_dict = {target.shape[1]: target}
        
        # Compute metrics for each horizon
        total_loss = 0.0
        
        for i, horizon in enumerate(self.prediction_horizons):
            if horizon not in targets_dict:
                continue
            
            target = targets_dict[horizon]
            prediction = self.model.forecast(context, prediction_length=horizon)
            
            loss = self.compute_loss(prediction, target)
            weight = self.horizon_weights[i]
            
            total_loss += weight * loss
            
            # Compute metrics for this horizon
            metrics = compute_forecasting_metrics(prediction, target)
            for key, value in metrics.items():
                self.log(f'val/h{horizon}_{key}', value, on_epoch=True, prog_bar=False)
        
        self.log('val/loss', total_loss, on_epoch=True, prog_bar=True)
        
        return total_loss