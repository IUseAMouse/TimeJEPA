"""
PyTorch Lightning Module for JEPA pretraining with TRUE forecasting objective.

The model learns to predict representations of FUTURE timesteps.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal
from pathlib import Path

from ..models.jepa_tst import JEPATST
from .utils.metrics import jepa_loss, compute_pretrain_metrics


class JEPAPretrainModule(pl.LightningModule):
    """
    Lightning Module for JEPA pretraining.
    
    Training workflow:
        1. Get batch with 'context' (past) and 'target' (future)
        2. Context → Online Encoder → context representations
        3. Target → Target Encoder (EMA) → target representations
        4. Predictor predicts target representations from context
        5. Loss: MSE between predicted and actual target representations
        6. Update online encoder + predictor via backprop
        7. Update target encoder via EMA (no backprop)
    """
    
    def __init__(
        self,
        model: JEPATST,
        vicreg_weights: Dict[str, float] = None,
        sigreg_config: Dict[str, float] = None,

        # Loss
        loss_type: Literal['mse', 'smooth_l1', 'cosine', 'vicreg', 'sigreg'] = 'vicreg',

        # Anti-collapse target: regularize the ENCODER output, not just the
        # predictor output (the encoder output is what downstream consumes).
        regularize_context: bool = True,

        # I-JEPA-style targets: encode [context ‖ target] and slice, instead of
        # encoding the future window in isolation.
        contextualized_targets: bool = True,

        # Input-geometry randomization. scripts/diagnose_ettm.py shows skill
        # peaks exactly at the training context length and collapses on both
        # sides (electricity: +28.5% at ctx=384, -103.8% at ctx=768), i.e. the
        # model memorizes a fixed patch count. Sampling the geometry per batch
        # is the direct countermeasure.
        context_lengths: Optional[list] = None,
        p_random_context: float = 0.0,
        horizon_lengths: Optional[list] = None,
        p_random_horizon: float = 0.0,

        # Optimizer
        learning_rate: float = 1e-3,
        weight_decay: float = 0.02,
        betas: tuple = (0.9, 0.95),
        
        # LR Scheduler
        warmup_epochs: float = 0.1,
        max_epochs: int = 20,
        lr_scheduler: Literal['cosine', 'linear', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Logging
        log_every_n_steps: int = 50,
    ):
        """
        Args:
            model: JEPATST model instance
            loss_type: Loss function type ('mse', 'smooth_l1', 'cosine')
            vicreg_weights: Weights for vicreg loss
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            betas: Adam beta parameters
            warmup_epochs: Fraction of epochs for warmup
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
        self.model.set_pretrain_mode(True)
        self.model.freeze_target_encoder()  # Target encoder never gets gradients
        
        # Loss
        self.loss_type = loss_type
        # Always store both: `validation_step` used to call jepa_loss WITHOUT
        # the weights, silently falling back to the (25, 25, 1) defaults. So
        # early-stopping and save_top_k were selecting on a different objective
        # than the one being trained.
        self.vicreg_weights = vicreg_weights
        self.sigreg_config = sigreg_config or {}
        self.regularize_context = regularize_context
        self.contextualized_targets = contextualized_targets

        self.context_lengths = list(context_lengths) if context_lengths else None
        self.p_random_context = float(p_random_context)
        self.horizon_lengths = list(horizon_lengths) if horizon_lengths else None
        self.p_random_horizon = float(p_random_horizon)

        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler_type = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
        
        print(f"JEPAPretrainModule initialized:")
        print(f"  Loss type: {loss_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Predicting {model.num_target_patches} future patches")
    
    def forward(self, context: torch.Tensor, target: torch.Tensor):
        """Forward pass."""
        return self.model(context, target)

    def _randomize_geometry(self, context: torch.Tensor, target: torch.Tensor):
        """
        Sample the input geometry ONCE PER BATCH.

        Per-batch (not per-sample) keeps every tensor rectangular, so no padding
        or attention masking is needed — the encoder is length-agnostic (RoPE,
        no learned positional table) and simply sees a different patch count.

        Context is cropped from the LEFT (keep the most recent history, which is
        what a shorter context would actually contain at inference); the target
        is cropped from the right.
        """
        if self.context_lengths and torch.rand(1).item() < self.p_random_context:
            eligible = [L for L in self.context_lengths if L <= context.shape[1]]
            if eligible:
                length = int(eligible[torch.randint(len(eligible), (1,)).item()])
                context = context[:, -length:]

        if self.horizon_lengths and torch.rand(1).item() < self.p_random_horizon:
            eligible = [H for H in self.horizon_lengths if H <= target.shape[1]]
            if eligible:
                horizon = int(eligible[torch.randint(len(eligible), (1,)).item()])
                target = target[:, :horizon]

        return context, target

    def _compute_loss(self, predictions, targets, outputs):
        """
        Single entry point used by BOTH training_step and validation_step, so
        the two can never diverge again (see B8: validation_step used to omit
        vicreg_weights and silently score a different objective).
        """
        return jepa_loss(
            predictions,
            targets,
            loss_type=self.loss_type,
            reduction='mean',
            vicreg_weights=self.vicreg_weights,
            sigreg_config=self.sigreg_config,
            context_embeddings=(
                outputs.get('context_embeddings') if self.regularize_context else None
            ),
            return_components=True,
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step with TRUE forecasting objective.
        
        Args:
            batch: Dictionary with 'context' (past) and 'target' (future)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        # Get context (past) and target (future)
        context = batch['context']  # [B, context_length] or [B, context_length, C]
        target = batch['target']    # [B, prediction_length] or [B, prediction_length, C]
        
        # Add channel dimension if needed (univariate case)
        if context.ndim == 2:
            context = context.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        if target.ndim == 2:
            target = target.unsqueeze(-1)    # [B, L] -> [B, L, 1]

        # Randomize input geometry (training only)
        context, target = self._randomize_geometry(context, target)
        self.log('geometry/context_len', float(context.shape[1]),
                 on_step=True, on_epoch=False, logger=True)
        self.log('geometry/horizon_len', float(target.shape[1]),
                 on_step=True, on_epoch=False, logger=True)

        # Forward pass - predict future representations
        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets
        )

        predictions = outputs['predictions']  # [B, num_target_patches, d_model]
        targets = outputs['targets']          # [B, num_target_patches, d_model]

        # Compute JEPA loss
        loss, components = self._compute_loss(predictions, targets, outputs)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for key, value in components.items():
            if key == 'loss':
                continue
            self.log(f'train_loss/{key}', value, on_step=True, on_epoch=False,
                     logger=True, sync_dist=True)

        # Additional metrics every N steps
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                metrics = compute_pretrain_metrics(
                    predictions,
                    targets,
                    context_embeddings=outputs.get('context_embeddings')
                )
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

        # NOTE: validation deliberately uses the NATIVE geometry, never the
        # randomized one, so val_loss stays comparable across epochs and runs.
        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets
        )

        predictions = outputs['predictions']
        targets = outputs['targets']

        # Same objective as training — see _compute_loss
        loss, components = self._compute_loss(predictions, targets, outputs)

        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for key, value in components.items():
            if key == 'loss':
                continue
            self.log(f'val_loss/{key}', value, on_step=False, on_epoch=True,
                     logger=True, sync_dist=True)

        # Compute metrics
        metrics = compute_pretrain_metrics(
            predictions,
            targets,
            context_embeddings=outputs.get('context_embeddings')
        )
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Collapse is the failure mode this whole phase exists to prevent, so
        # surface it as a first-class number instead of burying it in metrics.
        ctx = outputs.get('context_embeddings')
        if ctx is not None:
            collapse = ctx.std(dim=0).mean()
            self.log('collapse/context_std', collapse, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            # Effective rank of the representation: a collapsed encoder puts all
            # its energy in a handful of directions.
            with torch.no_grad():
                flat = ctx.reshape(-1, ctx.shape[-1]).float()
                flat = flat - flat.mean(dim=0, keepdim=True)
                sv = torch.linalg.svdvals(flat)
                p = sv / sv.sum().clamp_min(1e-12)
                entropy = -(p * (p + 1e-12).log()).sum()
                self.log('collapse/effective_rank', entropy.exp(), on_step=False,
                         on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler_type == 'constant':
            return optimizer
        
        # Calculate number of training steps
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        
        if self.lr_scheduler_type == 'cosine':
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
            
        elif self.lr_scheduler_type == 'linear':
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
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def update_target_encoder(self, step: int, max_steps: int):
        """Update target encoder with EMA (called by EMACallback)."""
        self.model.update_target_encoder(step, max_steps)
    
    def on_train_epoch_end(self):
        """Log learning rate at epoch end."""
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def save_pretrained(self, save_path: Path):
        """Save pretrained encoder for finetuning."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_encoder(str(save_path))
        print(f"✅ Saved pretrained encoder to {save_path}")