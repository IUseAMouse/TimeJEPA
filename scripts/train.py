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

logger = logging.getLogger(__name__)

def create_model_from_config(cfg) -> JEPATST:
    """
    Create JEPA-TST model from Hydra config.
    
    Args:
        cfg: Hydra config object with model, training, and data sections
    
    Returns:
        JEPATST model instance
    """
    
    # Map config to model arguments
    model = JEPATST(
        # Data params
        input_length=cfg.model.seq_length,  # 384
        prediction_length=cfg.model.prediction_length,  # 96
        num_features=cfg.model.num_channels,  # 1
        
        # Patching
        patch_size=cfg.model.patch_length,  # 16
        stride=cfg.model.stride,  # 8
        
        # Encoder
        d_model=cfg.model.encoder.d_model,  # 128
        num_layers=cfg.model.encoder.n_layers,  # 3
        num_heads=cfg.model.encoder.n_heads,  # 4
        d_ff=cfg.model.encoder.d_ff,  # 512
        dropout=cfg.model.encoder.dropout,  # 0.1
        activation=cfg.model.encoder.activation,  # "gelu"
        
        # Predictor
        predictor_type=cfg.model.predictor.type,  # "transformer"
        predictor_num_layers=cfg.model.predictor.n_layers,  # 2
        predictor_num_heads=cfg.model.predictor.n_heads,  # 4
        predictor_d_ff=cfg.model.predictor.d_ff,  # 512
        
        # Decoder (for finetuning)
        decoder_type=cfg.model.decoder.type,  # "linear"
        
        # EMA
        ema_tau_base=cfg.model.target_encoder.momentum_base,  # 0.996
        ema_tau_end=cfg.model.target_encoder.momentum_final,  # 1.0
        
        # RevIN
        use_revin=cfg.model.encoder.use_revin,  # true
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
    datamodule = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        train_val_test_split=cfg.data.train_val_test_split,
        seed=cfg.data.seed,
        num_workers=8
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Create Lightning module based on mode
    is_pretrain = cfg.training.mode == "pretrain"
    
    if is_pretrain:
        logger.info("Creating JEPA model...")
        
        # 🔥 CRÉER LE MODÈLE D'ABORD
        model = create_model_from_config(cfg)
        model.train()
        
        logger.info("Creating JEPA pretraining module...")
        
        # 🔥 PASSER LE MODÈLE AU MODULE
        pl_module = JEPAPretrainModule(
            model=model,  # ← Le modèle créé ci-dessus
            
            # Masking
            masking_strategy=cfg.training.masking.strategy,
            context_ratio=cfg.training.masking.context_ratio,
            masking_kwargs={
                'n_context_blocks': cfg.training.masking.get('n_context_blocks', 4),
                'context_block_length': cfg.training.masking.get('context_block_length', 3),
                'n_target_blocks': cfg.training.masking.get('n_target_blocks', 2),
                'target_block_length': cfg.training.masking.get('target_block_length', 3),
            },
            
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
    checkpoint_dir = Path(cfg.data.checkpoint_dir) / cfg.model.name
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
        log_model=cfg.wandb.log_model,
        log_freq=cfg.wandb.log_freq
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

    # Dans train.py, après model = JEPATST(...)
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