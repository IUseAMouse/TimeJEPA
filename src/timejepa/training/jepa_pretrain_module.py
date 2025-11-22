# src/timejepa/training/jepa_pretrain_module.py
"""
PyTorch Lightning Module for JEPA pretraining.

Handles training loop, optimization, and logging for self-supervised JEPA learning.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal
from pathlib import Path

from ..models.jepa_tst import JEPATST
from .utils.masking import get_masking_strategy, MaskingStrategy
from .utils.metrics import jepa_loss, compute_pretrain_metrics


class JEPAPretrainModule(pl.LightningModule):
    """
    Lightning Module for JEPA pretraining.
    
    Training workflow:
        1. Get batch of sequences from DataLoader
        2. Generate context/target masks using masking strategy
        3. Forward pass through JEPA model
        4. Compute loss between predicted and target representations
        5. Update online encoder + predictor (backprop)
        6. Update target encoder with EMA (no backprop)
    """
    
    def __init__(
        self,
        model: JEPATST,
        # Masking strategy
        masking_strategy: Literal['random', 'block', 'temporal', 'random_temporal'] = 'temporal',
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
        masking_kwargs: Optional[Dict[str, Any]] = None,
        
        # Loss
        loss_type: Literal['mse', 'smooth_l1', 'cosine'] = 'mse',
        
        # Optimizer
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        betas: tuple = (0.9, 0.95),
        
        # LR Scheduler
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        lr_scheduler: Literal['cosine', 'linear', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        """
        Args:
            model: JEPATST model instance
            masking_strategy: Type of masking ('temporal' recommended for time series)
            context_ratio: Fraction of patches to use as context
            target_ratio: Fraction of patches to use as target
            masking_kwargs: Additional arguments for masking strategy
            loss_type: Loss function type
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            betas: Adam beta parameters
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            lr_scheduler: Type of LR schedule
            min_lr: Minimum learning rate for scheduler
            log_every_n_steps: Logging frequency
        """
        super().__init__()
        
        # Save hyperparameters (except model)
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        self.model.set_pretrain_mode(True)  # Ensure pretrain mode
        self.model.freeze_target_encoder()  # Target encoder never gets gradients
        
        # Masking strategy
        masking_kwargs = masking_kwargs or {}
        self.masking_fn = get_masking_strategy(
            strategy_name=masking_strategy,
            num_patches=model.num_patches,
            context_ratio=context_ratio,
            target_ratio=target_ratio,
            **masking_kwargs
        )
        
        # Loss
        self.loss_type = loss_type
        
        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
    
    def forward(self, x: torch.Tensor, context_mask: torch.Tensor, target_mask: torch.Tensor):
        """Forward pass (for inference/validation)."""
        return self.model(x, context_mask=context_mask, target_mask=target_mask)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary with 'context' key (we use context as input sequence)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        # Get input sequence (use 'context' field from dataset)
        x = batch['context']  # [B, L, C] where L = context_length
        batch_size = x.shape[0]
        
        # For univariate data, add channel dimension if needed
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        
        # Generate context and target masks
        context_mask, target_mask = self.masking_fn(batch_size, device=x.device)
        # context_mask: [B, num_patches] boolean
        # target_mask: [B, num_patches] boolean
        
        # Forward pass
        outputs = self.model(x, context_mask=context_mask, target_mask=target_mask)
        # outputs: {'predictions': [B, N_tgt, D], 'targets': [B, N_tgt, D], 'context_embeddings': [B, N_ctx, D]}
        
        predictions = outputs['predictions']
        targets = outputs['targets']
        
        # Compute loss
        loss = jepa_loss(predictions, targets, loss_type=self.loss_type, reduction='mean')
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Additional metrics every N steps
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                metrics = compute_pretrain_metrics(
                    predictions,
                    targets,
                    context_embeddings=outputs.get('context_embeddings')
                )
                for key, value in metrics.items():
                    self.log(f'train/{key}', value, on_step=True, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch['context']
        batch_size = x.shape[0]
        
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        
        # Generate masks
        context_mask, target_mask = self.masking_fn(batch_size, device=x.device)
        
        # Forward pass
        outputs = self.model(x, context_mask=context_mask, target_mask=target_mask)
        
        predictions = outputs['predictions']
        targets = outputs['targets']
        
        # Compute loss
        loss = jepa_loss(predictions, targets, loss_type=self.loss_type, reduction='mean')
        
        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Compute metrics
        metrics = compute_pretrain_metrics(
            predictions,
            targets,
            context_embeddings=outputs.get('context_embeddings')
        )
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer (standard for transformers)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'constant':
            return optimizer
        
        # Calculate number of training steps
        # Note: trainer.estimated_stepping_batches is only available during training
        # For now, we estimate based on max_epochs
        # This will be adjusted by Lightning during training
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if hasattr(self.trainer, 'datamodule') else 1000
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        if self.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of lr
                end_factor=1.0,     # End at 100% of lr
                total_iters=warmup_steps
            )
            
            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.min_lr
            )
            
            # Combine warmup + cosine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
        elif self.lr_scheduler == 'linear':
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            # Warmup
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Linear decay
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
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step
                'frequency': 1,
            }
        }
    
    def update_target_encoder(self, step: int, max_steps: int):
        """
        Update target encoder with EMA.
        
        This is called by the EMACallback.
        """
        self.model.update_target_encoder(step, max_steps)
    
    def on_train_epoch_end(self):
        """Log learning rate at epoch end."""
        # Get current LR
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, on_epoch=True, prog_bar=True)
    
    def save_pretrained(self, save_path: Path):
        """Save pretrained encoder for finetuning."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained_encoder(str(save_path))
        print(f"✅ Saved pretrained encoder to {save_path}")