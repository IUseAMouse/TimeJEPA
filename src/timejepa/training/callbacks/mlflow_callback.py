#########################################################
# Deprecated since change to WandB, keeping as legacy
#########################################################
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class WandbCallback(Callback):
    """Wandb integration callback for PyTorch Lightning."""
    
    def __init__(
        self,
        project: str = "JEPA-TST",
        entity: str = "iuseamouse",
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        log_model: bool = True,
        log_gradients: bool = False,
        log_freq: int = 100,
    ):
        super().__init__()
        
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb required. Install: pip install wandb")
        
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.tags = tags or []
        self.log_model = log_model
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self._initialized = False
    
    @rank_zero_only
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Initialize wandb run."""
        if self._initialized:
            return
        
        # Start wandb run
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            tags=self.tags,
            config=pl_module.hparams if hasattr(pl_module, "hparams") else {},
        )
        
        # Watch model (logs gradients and params)
        if self.log_gradients:
            wandb.watch(pl_module, log="all", log_freq=self.log_freq)
        
        logger.info(f"Started wandb run: {wandb.run.name} ({wandb.run.id})")
        self._initialized = True
    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics."""
        if not self._initialized:
            return
        
        metrics = trainer.callback_metrics
        step = trainer.global_step
        
        # wandb accepte les '/' dans les noms (contrairement à MLflow)
        log_dict = {k: float(v) for k, v in metrics.items() if "train" in k}
        if log_dict:
            wandb.log(log_dict, step=step)
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics."""
        if not self._initialized:
            return
        
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        log_dict = {k: float(v) for k, v in metrics.items() if "val" in k}
        if log_dict:
            wandb.log(log_dict, step=epoch)
    
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Log final artifacts."""
        if not self._initialized:
            return
        
        # Log best checkpoint
        if self.log_model and trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path and Path(best_path).exists():
                wandb.save(best_path)
                logger.info(f"Logged checkpoint to wandb: {best_path}")
        
        wandb.finish()
        logger.info("Wandb run finished")
        self._initialized = False
    
    @rank_zero_only
    def on_exception(self, trainer, pl_module, exception):
        """Handle exceptions."""
        logger.error(f"Exception: {exception}")
        if self._initialized:
            wandb.finish(exit_code=1)