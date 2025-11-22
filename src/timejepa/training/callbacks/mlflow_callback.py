"""
MLflow integration for PyTorch Lightning.

Logs metrics, parameters, artifacts, and models to MLflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Install with: pip install mlflow")

logger = logging.getLogger(__name__)


class MLflowCallback(Callback):
    """
    MLflow integration callback for PyTorch Lightning.
    
    Features:
        - Automatic experiment and run management
        - Hyperparameter logging
        - Metrics logging (train/val/test)
        - Artifact logging (configs, checkpoints, plots)
        - Model logging (for deployment)
        - System metrics (GPU, CPU, memory)
    
    Usage:
        >>> callback = MLflowCallback(
        ...     tracking_uri="mlruns",
        ...     experiment_name="TimeJEPA",
        ...     run_name="pretrain_base",
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "TimeJEPA",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True,
        log_artifacts: bool = True,
        log_system_metrics: bool = True,
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            tracking_uri: MLflow tracking URI (local dir or remote server)
            experiment_name: Name of MLflow experiment
            run_name: Name of this run (auto-generated if None)
            tags: Tags to add to the run
            log_model: Whether to log the final model
            log_artifacts: Whether to log artifacts (configs, plots)
            log_system_metrics: Whether to log system metrics
            artifact_location: Custom artifact location
            run_id: Resume existing run (if provided)
            save_dir: Directory to save artifacts locally before upload
        """
        super().__init__()
        
        if not _MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.log_model = log_model
        self.log_artifacts = log_artifacts
        self.log_system_metrics = log_system_metrics
        self.artifact_location = artifact_location
        self.run_id = run_id
        self.save_dir = Path(save_dir) if save_dir else Path("mlflow_artifacts")
        
        self._mlflow_client: Optional[MlflowClient] = None
        self._initialized = False
    
    @rank_zero_only
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Initialize MLflow experiment and run."""
        if self._initialized:
            return
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
                logger.info(f"Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"Error creating/getting experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_id=experiment_id)
        
        # Start or resume run
        if self.run_id:
            # Resume existing run
            mlflow.start_run(run_id=self.run_id)
            logger.info(f"Resumed MLflow run: {self.run_id}")
        else:
            # Start new run
            mlflow.start_run(run_name=self.run_name)
            logger.info(f"Started MLflow run: {mlflow.active_run().info.run_name} "
                       f"(ID: {mlflow.active_run().info.run_id})")
        
        # Set tags
        if self.tags:
            mlflow.set_tags(self.tags)
        
        # Log system info
        mlflow.set_tag("framework", "PyTorch Lightning")
        mlflow.set_tag("stage", stage)
        
        # Enable system metrics
        if self.log_system_metrics:
            try:
                mlflow.enable_system_metrics_logging()
                logger.info("Enabled MLflow system metrics logging")
            except Exception as e:
                logger.warning(f"Could not enable system metrics: {e}")
        
        # Initialize client
        self._mlflow_client = MlflowClient(tracking_uri=self.tracking_uri)
        
        self._initialized = True
        
        # Log hyperparameters
        self._log_hyperparameters(trainer, pl_module)
    
    @rank_zero_only
    def _log_hyperparameters(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log model and training hyperparameters."""
        params = {}
        
        # Get hyperparameters from LightningModule
        if hasattr(pl_module, "hparams"):
            for key, value in pl_module.hparams.items():
                # MLflow doesn't support complex types
                if isinstance(value, (int, float, str, bool)):
                    params[key] = value
                elif isinstance(value, (list, tuple)):
                    params[key] = str(value)
                elif isinstance(value, dict):
                    # Flatten dict
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            params[f"{key}.{subkey}"] = subvalue
        
        # Add trainer params
        params["max_epochs"] = trainer.max_epochs
        params["devices"] = trainer.num_devices
        params["accelerator"] = trainer.accelerator.__class__.__name__
        if hasattr(trainer, "precision"):
            params["precision"] = str(trainer.precision)
        
        # Log to MLflow
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} hyperparameters to MLflow")
    
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx
    ):
        """Log training metrics."""
        if not self._initialized:
            return
        
        # Log metrics from trainer
        metrics = trainer.callback_metrics
        step = trainer.global_step
        
        for key, value in metrics.items():
            if "train" in key:
                try:
                    mlflow.log_metric(key, float(value), step=step)
                except (TypeError, ValueError):
                    pass  # Skip non-numeric metrics
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log validation metrics."""
        if not self._initialized:
            return
        
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        for key, value in metrics.items():
            if "val" in key:
                try:
                    mlflow.log_metric(key, float(value), step=epoch)
                except (TypeError, ValueError):
                    pass
    
    @rank_zero_only
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log test metrics."""
        if not self._initialized:
            return
        
        metrics = trainer.callback_metrics
        
        for key, value in metrics.items():
            if "test" in key:
                try:
                    mlflow.log_metric(key, float(value))
                except (TypeError, ValueError):
                    pass
    
    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log final artifacts and model."""
        if not self._initialized:
            return
        
        logger.info("Training finished. Logging final artifacts...")
        
        # Log checkpoint if available
        if self.log_artifacts and trainer.checkpoint_callback:
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path and Path(best_model_path).exists():
                try:
                    mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
                    logger.info(f"Logged best checkpoint: {best_model_path}")
                except Exception as e:
                    logger.warning(f"Could not log checkpoint: {e}")
        
        # Log model for deployment
        if self.log_model:
            try:
                # Save model signature and input example
                import torch
                
                # Create dummy input
                dummy_input = torch.randn(1, 512, pl_module.model.num_channels)
                
                mlflow.pytorch.log_model(
                    pytorch_model=pl_module.model,
                    artifact_path="model",
                    signature=None,  # Could infer from dummy_input
                )
                logger.info("Logged PyTorch model to MLflow")
            except Exception as e:
                logger.warning(f"Could not log model: {e}")
    
    @rank_zero_only
    def finalize(self, trainer: pl.Trainer, pl_module: pl.LightningModule, status: str = "FINISHED"):
        """Finalize MLflow run."""
        if not self._initialized:
            return
        
        # Set final status tag
        mlflow.set_tag("status", status)
        
        # End run
        mlflow.end_run()
        logger.info(f"MLflow run ended with status: {status}")
        
        self._initialized = False
    
    @rank_zero_only
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: Exception):
        """Handle exceptions by finalizing run with FAILED status."""
        logger.error(f"Exception occurred: {exception}")
        self.finalize(trainer, pl_module, status="FAILED")
    
    @rank_zero_only
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact."""
        if not self._initialized:
            logger.warning("MLflow not initialized. Cannot log artifact.")
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Could not log artifact {local_path}: {e}")
    
    @rank_zero_only
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        if not self._initialized:
            logger.warning("MLflow not initialized. Cannot log dict.")
            return
        
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.info(f"Logged dict artifact: {artifact_file}")
        except Exception as e:
            logger.warning(f"Could not log dict: {e}")
    
    @rank_zero_only
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure as an artifact."""
        if not self._initialized:
            logger.warning("MLflow not initialized. Cannot log figure.")
            return
        
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.info(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.warning(f"Could not log figure: {e}")