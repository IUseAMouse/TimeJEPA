# src/timejepa/training/callbacks/ema_callback.py
"""
PyTorch Lightning callback for EMA (Exponential Moving Average) updates.

Updates the target encoder during JEPA training.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class EMACallback(Callback):
    """
    Callback to update target encoder with EMA of online encoder.
    
    The target encoder is updated after each training step using
    exponential moving average with a cosine schedule.
    """
    
    def __init__(
        self,
        momentum_base: float = 0.996,
        momentum_final: float = 1.0,
        schedule: str = "cosine",
        update_after_step: bool = True,
        update_after_epoch: bool = False,
    ):
        """
        Args:
            momentum_base: Initial EMA momentum (tau)
            momentum_final: Final EMA momentum (tau)
            schedule: Momentum schedule type ('cosine', 'linear', 'constant')
            update_after_step: Update EMA after each training step
            update_after_epoch: Update EMA after each epoch (usually not needed)
        """
        super().__init__()
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.schedule = schedule
        self.update_after_step = update_after_step
        self.update_after_epoch = update_after_epoch
        
        logger.info(
            f"EMACallback initialized with momentum_base={momentum_base}, "
            f"momentum_final={momentum_final}, schedule={schedule}"
        )
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA after each training step."""
        if not self.update_after_step:
            return
        
        # Check if module has update_target_encoder method
        if not hasattr(pl_module, 'update_target_encoder'):
            logger.warning(
                "LightningModule does not have 'update_target_encoder' method. "
                "EMACallback will have no effect."
            )
            return
        
        # Get current global step and max steps
        current_step = trainer.global_step
        max_steps = trainer.estimated_stepping_batches
        
        # Update target encoder
        pl_module.update_target_encoder(current_step, max_steps)
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Optionally update EMA after each epoch."""
        if not self.update_after_epoch:
            return
        
        if not hasattr(pl_module, 'update_target_encoder'):
            return
        
        current_step = trainer.global_step
        max_steps = trainer.estimated_stepping_batches
        
        pl_module.update_target_encoder(current_step, max_steps)
        
        # Log current EMA momentum
        if hasattr(pl_module.model, 'target_encoder'):
            tau = pl_module.model.target_encoder.get_current_tau(current_step, max_steps)
            pl_module.log('ema/tau', tau, on_epoch=True, prog_bar=False)


class GradientClipCallback(Callback):
    """
    Callback to monitor gradient norms.
    
    Useful for debugging training instabilities.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
        opt_idx: int = 0,
    ) -> None:
        """Log gradient norms before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        # Compute gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        pl_module.log('train/grad_norm', total_norm, on_step=True, prog_bar=False)