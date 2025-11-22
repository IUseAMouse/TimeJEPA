"""
Training script compatible with existing MonashDataModule.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import MultiDatasetMonashDataModule
from timejepa.training.jepa_pretrain_module import JEPAPretrainModule
from timejepa.training.finetune_module import FinetuneModule
from timejepa.training.callbacks import EMACallback, MLflowCallback

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/training", config_name="pretrain")
def main(cfg: DictConfig):
    """Main training function."""
    
    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.data.seed, workers=True)
    
    # Create data module
    logger.info("Creating data module...")
    datamodule = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.data.context_length,
        prediction_length=cfg.data.prediction_length,
        datasets=cfg.data.get('datasets'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        train_val_test_split=cfg.data.train_val_test_split,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        max_series=cfg.data.get('max_series'),
        min_series_length=cfg.data.get('min_series_length'),
        dataset_overrides=OmegaConf.to_container(cfg.data.get('dataset_overrides', {}), resolve=True),
        seed=cfg.data.seed,
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Create Lightning module based on mode
    is_pretrain = cfg.training.mode == "pretrain"
    
    if is_pretrain:
        logger.info("Creating JEPA pretraining module...")
        pl_module = JEPAPretrainModule(
            # Utilise context_length au lieu de seq_length
            seq_length=cfg.data.context_length,
            patch_length=cfg.model.patch_length,
            stride=cfg.model.stride,
            num_channels=cfg.model.encoder.get('num_channels', 1),
            d_model=cfg.model.encoder.d_model,
            n_heads=cfg.model.encoder.n_heads,
            n_layers=cfg.model.encoder.n_layers,
            d_ff=cfg.model.encoder.d_ff,
            dropout=cfg.model.encoder.dropout,
            activation=cfg.model.encoder.activation,
            use_revin=cfg.model.encoder.use_revin,
            use_rope=cfg.model.encoder.use_rope,
            
            # Predictor
            predictor_type=cfg.model.predictor.type,
            predictor_d_model=cfg.model.predictor.d_model,
            predictor_n_heads=cfg.model.predictor.n_heads,
            predictor_n_layers=cfg.model.predictor.n_layers,
            predictor_d_ff=cfg.model.predictor.d_ff,
            predictor_dropout=cfg.model.predictor.dropout,
            
            # Training
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
            lr_scheduler_type=cfg.training.lr_scheduler.type,
            warmup_epochs=cfg.training.lr_scheduler.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            min_lr=cfg.training.lr_scheduler.min_lr,
            
            # Masking
            masking_strategy=cfg.training.masking.strategy,
            context_ratio=cfg.training.masking.context_ratio,
            n_context_blocks=cfg.training.masking.get('n_context_blocks', 4),
            n_target_blocks=cfg.training.masking.get('n_target_blocks', 2),
            
            # Loss
            loss_type=cfg.training.loss.type,
        )
    else:
        logger.info("Creating finetuning module...")
        pl_module = FinetuneModule(
            seq_length=cfg.data.context_length,
            patch_length=cfg.model.patch_length,
            stride=cfg.model.stride,
            num_channels=cfg.model.encoder.get('num_channels', 1),
            prediction_length=cfg.data.prediction_length,
            d_model=cfg.model.encoder.d_model,
            n_heads=cfg.model.encoder.n_heads,
            n_layers=cfg.model.encoder.n_layers,
            d_ff=cfg.model.encoder.d_ff,
            dropout=cfg.model.encoder.dropout,
            
            # Decoder
            decoder_type=cfg.model.decoder.type,
            decoder_hidden_dim=cfg.model.decoder.hidden_dim,
            decoder_n_layers=cfg.model.decoder.n_layers,
            
            # Training
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            encoder_lr_multiplier=cfg.training.optimizer.get('encoder_lr_multiplier', 0.1),
            
            # Pretrained
            pretrained_encoder_path=cfg.training.get('pretrained_encoder_path'),
            finetune_mode=cfg.training.get('finetune_mode', 'full_finetune'),
            
            # Loss
            loss_type=cfg.training.loss.type,
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / cfg.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        filename=cfg.checkpoint.filename,
        verbose=True,
    ))
    
    # Early stopping
    if cfg.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            min_delta=cfg.early_stopping.min_delta,
            verbose=True,
        ))
    
    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # EMA (pretrain only)
    if is_pretrain and cfg.training.ema.enabled:
        callbacks.append(EMACallback(
            momentum_base=cfg.training.ema.momentum_base,
            momentum_final=cfg.training.ema.momentum_final,
            max_epochs=cfg.training.max_epochs,
            schedule=cfg.training.ema.schedule,
        ))
    
    # MLflow
    callbacks.append(MLflowCallback(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.mlflow.run_name or cfg.name,
        tags=OmegaConf.to_container(cfg.mlflow.tags, resolve=True),
        log_model=cfg.mlflow.log_model,
        log_artifacts=cfg.mlflow.log_artifacts,
        log_system_metrics=cfg.mlflow.log_system_metrics,
    ))
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        default_root_dir=cfg.paths.output_dir,
        deterministic=cfg.trainer.get('deterministic', False),
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    trainer.fit(pl_module, datamodule=datamodule)
    
    # Test best model
    if trainer.checkpoint_callback.best_model_path:
        logger.info("=" * 80)
        logger.info("TESTING BEST MODEL")
        logger.info("=" * 80)
        trainer.test(pl_module, datamodule=datamodule, ckpt_path='best')
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()