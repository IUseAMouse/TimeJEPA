# scripts/train.py
"""
Training script compatible with existing MonashDataModule.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import MultiDatasetMonashDataModule
from timejepa.training.jepa_pretrain_module import JEPAPretrainModule
from timejepa.training.finetune_module import FinetuneModule
from timejepa.training.callbacks import EMACallback
from timejepa.models import JEPATST
from timejepa.models.decoders import ForecastingHead

logger = logging.getLogger(__name__)

def create_model_from_config(cfg) -> JEPATST:
    """
    Create JEPA-TST model from Hydra config.
    """
    # Map config to model arguments
    model = JEPATST(
        # Data params
        input_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        
        # Patching
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        
        # Encoder
        d_model=cfg.model.encoder.d_model,
        num_layers=cfg.model.encoder.n_layers,
        num_heads=cfg.model.encoder.n_heads,
        d_ff=cfg.model.encoder.d_ff,
        dropout=cfg.model.encoder.dropout,
        activation=cfg.model.encoder.activation,
        
        # Predictor
        predictor_type=cfg.model.predictor.type,
        predictor_num_layers=cfg.model.predictor.n_layers,
        predictor_num_heads=cfg.model.predictor.n_heads,
        predictor_d_ff=cfg.model.predictor.d_ff,
        
        # Decoder (for finetuning)
        decoder_type=cfg.model.decoder.type,
        
        # EMA
        ema_tau_base=cfg.model.target_encoder.momentum_base,
        ema_tau_end=cfg.model.target_encoder.momentum_final,
        
        # RevIN
        use_revin=cfg.model.encoder.use_revin,
    )
    
    # Log model info
    num_params = model.get_num_params()
    print(f"✅ Created {cfg.model.name} model:")
    print(f"   - Input length: {cfg.model.seq_length}")
    print(f"   - Num patches: {model.num_patches}")
    print(f"   - Patch size: {cfg.model.patch_length}, stride: {cfg.model.stride}")
    print(f"   - d_model: {cfg.model.encoder.d_model}")
    print(f"   - Encoder layers: {cfg.model.encoder.n_layers}")
    print(f"   - Total params: {num_params['total']:,}")
    print(f"   - Trainable params: {num_params['trainable']:,}")
    
    return model


@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
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
    is_pretrain = cfg.training.mode == "pretrain"
    datamodule = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets') if is_pretrain else cfg.data.get('datasets_finetune'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        seed=cfg.data.seed,
        num_workers=8
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Create Model (Architecture)
    model = create_model_from_config(cfg)
    
    # Create Lightning Module (Training Logic)
    
    
    if is_pretrain:
        logger.info("Creating JEPA model...")
        model.train()
        
        logger.info("Creating JEPA pretraining module...")
        pl_module = JEPAPretrainModule(
            model=model,
            
            # Loss
            loss_type=cfg.training.loss.type,
            
            # Optimizer
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
            
            # Scheduler
            warmup_epochs=cfg.training.lr_scheduler.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            lr_scheduler=cfg.training.lr_scheduler.type,
            min_lr=cfg.training.lr_scheduler.min_lr,
            
            # Logging
            log_every_n_steps=cfg.training.log_every_n_steps,
        )
    else:
        logger.info("Creating finetuning module...")
        
        warmup_epochs = cfg.training.lr_scheduler.warmup_epochs

        model.decoder = ForecastingHead(
            d_model=cfg.model.decoder.d_model,
            patch_size=cfg.model.patch_length,
            stride=cfg.model.stride,
            prediction_length=cfg.model.prediction_length,
            num_features=cfg.model.num_channels,
            decoder_type=cfg.model.decoder.type,
            revin=model.revin
        )
        
        pl_module = FinetuneModule(
            model=model,  # <-- On passe l'instance du modèle créée plus haut
            
            # Pretrained weights & Strategy
            pretrained_encoder_path=cfg.training.get('pretrained_encoder_path'),
            finetune_mode=cfg.training.get('finetune_mode', 'linear_probe'),
            
            # Loss
            loss_type=cfg.training.loss.finetune_type,
            
            # Optimizer
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            encoder_lr_multiplier=cfg.training.optimizer.get('encoder_lr_multiplier', 0.1),
            betas=tuple(cfg.training.optimizer.betas),
            
            # Scheduler
            warmup_epochs=warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            lr_scheduler=cfg.training.lr_scheduler.type,
            min_lr=cfg.training.lr_scheduler.min_lr,
            
            # Regularization
            dropout=cfg.training.dropout,
            
            # Logging
            log_every_n_steps=cfg.training.log_every_n_steps,
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_dir = Path(cfg.data.checkpoint_dir) / cfg.model.name / f"pretrain:{is_pretrain}"
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
            schedule=cfg.training.ema.schedule,
        ))
    
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or cfg.model.name,
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=cfg.wandb.log_model
    )

    wandb_logger.watch(model)
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        default_root_dir=cfg.data.output_dir,
        deterministic=cfg.trainer.get('deterministic', False),
    )

    print(f"🔍 DEBUG Model:")
    print(f"  model.seq_length: {model.input_length}")
    print(f"  model.num_patches: {model.num_patches}")
    print(f"  Expected from config: {cfg.model.seq_length}")
    print(f"  Datamodule context_length: {datamodule.context_length}")
    
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