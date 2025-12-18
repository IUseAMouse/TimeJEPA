# src/timejepa/training/finetune_module.py
"""
PyTorch Lightning Module for supervised finetuning.

Uses the pretrained encoder and predictor to forecast actual values.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
import logging

from ..models.jepa_tst import JEPATST
from .utils.metrics import compute_forecasting_metrics, mse, mae

logger = logging.getLogger(__name__)


class FinetuneModule(pl.LightningModule):
    """
    Lightning Module for supervised finetuning.
    
    Training modes:
        - 'linear_probe': Freeze encoder+predictor, train only decoder
        - 'full_finetune': Train encoder + predictor + decoder
        - 'gradual_unfreeze': Start frozen, gradually unfreeze layers
    
    Workflow:
        1. Load pretrained encoder + predictor weights
        2. Switch model to finetune mode
        3. Apply freezing strategy
        4. Train with supervised forecasting loss (MSE on actual values)
    """
    
    def __init__(
        self,
        model: JEPATST,
        pretrained_encoder_path: Optional[str] = None,
        
        # Finetuning strategy
        finetune_mode: Literal['linear_probe', 'full_finetune', 'gradual_unfreeze'] = 'linear_probe',
        unfreeze_after_epoch: int = 5,
        
        # Loss
        loss_type: Literal['mse', 'mae', 'huber'] = 'mse',
        huber_delta: float = 1.0,
        
        # Optimizer
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        
        # LR Scheduler
        warmup_epochs: float = 0.1,
        max_epochs: int = 50,
        lr_scheduler: Literal['cosine', 'linear', 'plateau', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Regularization
        dropout: float = 0.1,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        self.model.set_pretrain_mode(False)  # Switch to finetune mode
        logger.info("✓ Model switched to finetune mode")
        
        # Load pretrained weights if provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)
        
        # Apply finetuning strategy
        self.finetune_mode = finetune_mode
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._apply_finetune_strategy(finetune_mode)
        
        # Loss configuration
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        # Optimizer params
        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler_type = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
    
    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder and predictor weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            # Clean keys
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k.replace("model.", "").replace("_orig_mod.", "")
                if "target_encoder" in clean_key:
                    continue  # Skip target encoder
                if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                    continue  # Skip runtime buffers
                cleaned_state_dict[clean_key] = v
        elif 'online_encoder' in checkpoint:
            # Direct save format from save_pretrained_encoder
            cleaned_state_dict = {}
            for component in ['online_encoder', 'predictor', 'patching', 'revin']:
                if component in checkpoint:
                    for k, v in checkpoint[component].items():
                        cleaned_state_dict[f"{component}.{k}"] = v
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        
        # Load weights
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        
        # Check for critical missing keys
        expected_missing = {'decoder', 'target_encoder', 'revin'}
        critical_missing = [k for k in missing if not any(exp in k for exp in expected_missing)]
        
        if critical_missing:
            logger.error(f"❌ Critical missing keys: {critical_missing}")
            raise RuntimeError(f"Failed to load pretrained weights: {critical_missing}")
        
        logger.info(f"✓ Loaded pretrained weights ({len(cleaned_state_dict)} keys)")
        logger.info(f"  Expected missing (decoder): {len(missing) - len(critical_missing)} keys")
    
    def _apply_finetune_strategy(self, mode: str):
        """Apply freezing strategy based on finetune mode."""
        if mode == 'linear_probe':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            self.model.freeze_target_encoder()
            logger.info("✓ LINEAR PROBE: encoder frozen, predictor + decoder trainable")
        
        elif mode == 'full_finetune':
            self.model.unfreeze_encoder()
            self.model.unfreeze_predictor()
            self.model.unfreeze_patching()
            logger.info("✓ FULL FINETUNE: all components trainable")
        
        elif mode == 'gradual_unfreeze':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            logger.info(f"✓ GRADUAL UNFREEZE: frozen, will unfreeze at epoch {self.unfreeze_after_epoch}")
        
        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")
    
    def on_train_epoch_start(self):
        """Handle gradual unfreezing."""
        if self.finetune_mode == 'gradual_unfreeze':
            if self.current_epoch == self.unfreeze_after_epoch:
                logger.info(f"Epoch {self.current_epoch}: Unfreezing encoder and predictor")
                self.model.unfreeze_predictor()
    
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for forecasting."""
        return self.model.forecast(context)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute forecasting loss."""
        if self.loss_type == 'mse':
            return mse(predictions, targets)
        elif self.loss_type == 'mae':
            return mae(predictions, targets)
        elif self.loss_type == 'huber':
            return nn.functional.huber_loss(predictions, targets, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        context = batch['context']
        target = batch['target']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        # Forward pass
        results = self.model.forecast(context)
        predictions_denorm = results['forecast_denorm']
        predictions_norm = results['forecast']

        # Normalize target with same stats
        if self.model.revin is not None:
            target_norm = (target - self.model.revin.mean) / self.model.revin.std
        else:
            target_norm = target

        # Compute loss
        loss = self.compute_loss(predictions_norm, target_norm)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                metrics = compute_forecasting_metrics(predictions_norm, target_norm)
                for key, value in metrics.items():
                    self.log(f'train_{key}', value, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        context = batch['context']
        target = batch['target']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        results = self.model.forecast(context)
        predictions_denorm = results['forecast_denorm']
        predictions_norm = results['forecast']
        
        if self.model.revin is not None:
            target_norm = (target - self.model.revin.mean) / self.model.revin.std
        else:
            target_norm = target
        
        loss = self.compute_loss(predictions_norm, target_norm)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        metrics = compute_forecasting_metrics(predictions_norm, target_norm)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        context = batch['context']
        target = batch['target']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        results = self.model.forecast(context)
        predictions_norm = results['forecast']
        predictions_denorm = results['forecast_denorm']
        
        if self.model.revin is not None:
            target_norm = (target - self.model.revin.mean) / self.model.revin.std
        else:
            target_norm = target
        
        loss = self.compute_loss(predictions_norm, target_norm)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        metrics = compute_forecasting_metrics(predictions_norm, target_norm)
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            'loss': loss,
            'predictions': predictions_denorm,
            'targets': target,
            'metrics': metrics
        }
    
    def configure_optimizers(self):
        """Configure optimizer with different LR for encoder vs decoder."""
        # Separate parameters
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'decoder' in name:
                decoder_params.append(param)
            else:
                encoder_params.append(param)
        
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
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler_type == 'constant':
            return optimizer
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        
        if self.lr_scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=self.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        
        elif self.lr_scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=self.min_lr)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
    
    def on_train_epoch_end(self):
        """Log learning rates."""
        optimizer = self.optimizers()
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.log(f'lr_{group_name}', param_group['lr'], on_epoch=True, prog_bar=True, sync_dist=True)