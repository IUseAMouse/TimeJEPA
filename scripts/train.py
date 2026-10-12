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
from timejepa.models import JEPATST  # noqa: F401 - historical re-export
from timejepa.models.decoders import ForecastingHead  # finetune path (l.162)
# Model construction has one source of truth, shared with evaluate.py and
# evaluate_gift.py; any new constructor option must live there only.
from timejepa.evaluation.loading import create_model_from_config

logger = logging.getLogger(__name__)

def apply_schedule_fraction(trainer_kwargs: dict, cfg) -> dict:
    """training.schedule_fraction < 1: the run ends where the cosine ends.
    Bounds the Trainer to that fraction of the epoch (limit_train_batches)
    unless the config already limits it; the modules shorten the scheduler
    with the same key. Inert at 1.0."""
    frac = float(cfg.training.get('schedule_fraction', 1.0))
    if not (0.0 < frac <= 1.0):
        raise ValueError('training.schedule_fraction must be in (0, 1]')
    if frac < 1.0 and trainer_kwargs.get('limit_train_batches') is None:
        trainer_kwargs['limit_train_batches'] = frac
        logger.info(f"schedule_fraction={frac}: cosine annealed and run bounded at "
                    f"{frac:.0%} of the epoch")
    return trainer_kwargs


@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    """Main training function."""
    
    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    
    pl.seed_everything(cfg.data.seed, workers=True)
    
    logger.info("Creating data module...")
    is_pretrain = cfg.training.mode == "pretrain"

    # Augmentations: pick the pretrain or finetune block of the config.
    aug_root = cfg.get('augmentations') or {}
    aug_key = 'pretrain' if is_pretrain else 'finetune'
    aug_cfg = aug_root.get(aug_key) if aug_root else None
    if aug_cfg is not None:
        aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)
        if not aug_cfg.get('enabled', True):
            aug_cfg = None
    logger.info(
        f"Augmentations ({aug_key}): "
        + ("enabled" if aug_cfg else "disabled")
    )

    datamodule = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets') if is_pretrain else cfg.data.get('datasets_finetune'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        balanced_sampling=cfg.data.balanced_sampling,
        sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio,
        # Uniform spreading of the oversampling cap over the epoch (opt-in).
        ration_oversample=bool(cfg.data.get('ration_oversample', False)),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        augmentation_config=aug_cfg,
        # True multi-resolution: reads a longer raw stretch and decimates it, so
        # the seasonal period in patch positions actually varies during training.
        # Train split only. See scripts/diagnose_ettm.py for why this matters.
        multi_resolution_factors=(
            list(cfg.data.get('multi_resolution_factors') or [1]) if is_pretrain
            else list(cfg.data.get('multi_resolution_factors_finetune') or [1])
        ),
        p_multi_resolution=(
            float(cfg.data.get('p_multi_resolution', 0.0)) if is_pretrain
            # Finetune can exercise the cross-resolution pairs too: a pretrain
            # capability only survives finetune if finetune passes through it.
            # Dedicated *_finetune keys, inert by default.
            else float(cfg.data.get('p_multi_resolution_finetune', 0.0))
        ),
        # Cross-resolution: context at k1, target at k2, per-item 'w' key.
        # Same config key as the model and the module, in both modes
        # (outside train, the dataset emits w=1).
        cross_resolution=bool(cfg.model.get('cross_resolution', False)),
        # Short-series windows, finetune only (the JEPA latent loss has no
        # target mask, so pretrain must not see padded targets). Needs the
        # _reallen sidecar next to the corpus; absent means inactive.
        short_series_windows=(not is_pretrain)
        and bool(cfg.data.get('short_series_windows', False)),
        short_min_context=int(cfg.data.get('short_min_context', 16)),
        short_min_target=int(cfg.data.get('short_min_target', 4)),
        seed=cfg.data.seed,
        # Tunable: with 20+ datasets in memory (some as numpy object arrays,
        # which defeat fork's copy-on-write), 8 train plus 8 validation
        # workers were enough to get the run OOM-killed.
        num_workers=int(cfg.data.get('num_workers', 4)),
        persistent_workers=bool(cfg.data.get('persistent_workers', False)),
        # LOTSA-scale corpora: read the .npy files without loading them into RAM.
        use_mmap=bool(cfg.data.get('use_mmap', False)),
    )
    
    datamodule.prepare_data()
    
    model = create_model_from_config(cfg)
    
    if is_pretrain:
        logger.info("Creating JEPA model...")
        model.train()
        
        logger.info("Creating JEPA pretraining module...")
        sigreg_cfg = cfg.training.loss.get('sigreg')
        sigreg_cfg = OmegaConf.to_container(sigreg_cfg, resolve=True) if sigreg_cfg else {}

        pl_module = JEPAPretrainModule(
            model=model,

            # Loss
            loss_type=cfg.training.loss.type,
            vicreg_weights={
                'invariance': cfg.training.loss.invariance_loss_weight,
                'variance': cfg.training.loss.variance_loss_weight,
                'covariance': cfg.training.loss.covariance_loss_weight
            },
            sigreg_config=sigreg_cfg,
            regularize_context=cfg.training.loss.get('regularize_context', True),
            contextualized_targets=cfg.training.get('contextualized_targets', True),
            # Reconstruction ablation arm; see configs/model/lotsa_tiny_recon.yaml.
            reconstruction_target=cfg.training.get('reconstruction_target', False),
            # Cross-resolution arm: a single config key (model.cross_resolution)
            # drives the model, the module and the datamodule.
            cross_resolution=cfg.model.get('cross_resolution', False),
            # ErrorSignal arm: a single key (model.error_signal) drives the
            # model (heads) and the module (z loss). lambda_z lives under
            # training.loss.
            error_signal=cfg.model.get('error_signal', False),
            lambda_z=float(cfg.training.loss.get('lambda_z', 0.1)),

            # Input-geometry randomization (train split only)
            context_lengths=list(cfg.training.get('context_lengths') or []),
            p_random_context=float(cfg.training.get('p_random_context', 0.0)),
            horizon_lengths=list(cfg.training.get('horizon_lengths') or []),
            p_random_horizon=float(cfg.training.get('p_random_horizon', 0.0)),

            # Optimizer
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
            
            # Scheduler
            warmup_epochs=cfg.training.lr_scheduler.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            schedule_fraction=float(cfg.training.get('schedule_fraction', 1.0)),
            lr_scheduler=cfg.training.lr_scheduler.type,
            min_lr=cfg.training.lr_scheduler.min_lr,
            
            # Logging
            log_every_n_steps=cfg.training.log_every_n_steps,
        )
    else:
        logger.info("Creating finetuning module...")
        
        warmup_epochs = cfg.training.lr_scheduler.warmup_epochs

        # ErrorSignal finetune guard: a z trained at pretrain must not be
        # silently left unconsumed by a point decoder (the ForecastingHead
        # also raises; this makes the message actionable at config level).
        if cfg.model.get('error_signal', False) and cfg.model.decoder.type != 'quantile':
            raise ValueError(
                "model.error_signal=true requires model.decoder.type='quantile' "
                "at finetune - the z path modulates the quantile fan spread."
            )

        model.decoder = ForecastingHead(
            d_model=cfg.model.decoder.d_model,
            patch_size=cfg.model.patch_length,
            stride=cfg.model.stride,
            prediction_length=cfg.model.prediction_length,
            num_features=cfg.model.num_channels,
            decoder_type=cfg.model.decoder.type,
            revin=model.revin,
            # Same config key as the model and the eval (loading.py), guarded
            # by a test.
            error_signal=bool(cfg.model.get('error_signal', False)),
            # Optional head width, same key as loading.py; absent means the
            # d_model fallback.
            quantile_hidden_dim=cfg.model.decoder.get('quantile_hidden_dim'),
        )
        
        pl_module = FinetuneModule(
            model=model,
            # Query-table extension (native horizon), inert when absent.
            extend_horizon_queries=cfg.training.get('extend_horizon_queries', False),  
            
            # Pretrained weights & Strategy
            pretrained_encoder_path=cfg.training.get('pretrained_encoder_path'),
            finetune_mode=cfg.training.get('finetune_mode', 'gradual_unfreeze'),
            unfreeze_after_epoch=cfg.training.unfreeze_after_epoch,
            
            # Loss
            loss_type=cfg.training.loss.finetune_type,
            # Invariance anchor: keep lambda*MSE(z_pred, z_tgt) at finetune so
            # drift destroys neither the judge nor cross-resolution coherence.
            lambda_anchor=float(cfg.training.loss.get('lambda_anchor', 0.0)),
            # Joint JEPA term on the true target during finetune, and the
            # critic loop; inert defaults.
            lambda_joint=float(cfg.training.loss.get('lambda_joint', 0.0)),
            joint_target=str(cfg.training.loss.get('joint_target', 'frozen')),
            joint_contextualized=bool(cfg.training.loss.get('joint_contextualized', False)),
            joint_sigreg=bool(cfg.training.loss.get('joint_sigreg', False)),
            sigreg_config=(OmegaConf.to_container(cfg.training.loss.sigreg, resolve=True)
                           if cfg.training.loss.get('sigreg') else {}),
            critic_steps=list(cfg.training.loss.get('critic_steps') or []),
            critic_alpha=float(cfg.training.loss.get('critic_alpha', 0.0)),
            critic_route=str(cfg.training.loss.get('critic_route', 'A')),
            critic_target=str(cfg.training.loss.get('critic_target', 'center')),
            critic_energy=str(cfg.training.loss.get('critic_energy', 'cos')),
            critic_contextualized=bool(cfg.training.loss.get('critic_contextualized', False)),
            critic_step_weights=str(cfg.training.loss.get('critic_step_weights', 'uniform')),
            critic_noise=float(cfg.training.loss.get('critic_noise', 0.0)),
            critic_batch_fraction=float(cfg.training.loss.get('critic_batch_fraction', 1.0)),
            critic_max_abs_delta=float(cfg.training.loss.get('critic_max_abs_delta', 5.0)),
            critic_step_norm=bool(cfg.training.loss.get('critic_step_norm', True)),
            # Score matching on the energy; inert defaults.
            lambda_score=float(cfg.training.loss.get('lambda_score', 0.0)),
            score_route=str(cfg.training.loss.get('score_route', 'B')),
            score_perturb=(OmegaConf.to_container(cfg.training.loss.score_perturb, resolve=True)
                           if cfg.training.loss.get('score_perturb') else None),
            score_batch_fraction=float(cfg.training.loss.get('score_batch_fraction', 0.5)),
            score_contextualized=bool(cfg.training.loss.get('score_contextualized', False)),
            score_energy=str(cfg.training.loss.get('score_energy', 'cos')),
            score_valley_delta=float(cfg.training.loss.get('score_valley_delta', 0.1)),
            
            # Optimizer
            learning_rate=cfg.training.optimizer.learning_rate,
            weight_decay=cfg.training.optimizer.weight_decay,
            encoder_lr_multiplier=cfg.training.optimizer.get('encoder_lr_multiplier', 0.1),
            betas=tuple(cfg.training.optimizer.betas),
            
            # Scheduler
            warmup_epochs=warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            schedule_fraction=float(cfg.training.get('schedule_fraction', 1.0)),
            lr_scheduler=cfg.training.lr_scheduler.type,
            min_lr=cfg.training.lr_scheduler.min_lr,
            
            # Regularization
            dropout=cfg.training.dropout,

            # Context-geometry randomization (train only). Reads the same
            # context_lengths list as the pretrain, but its own probability key
            # so existing finetune configs keep their previous fixed-context
            # behavior at the default of 0.0.
            context_lengths=list(cfg.training.get('context_lengths') or []),
            p_random_context_finetune=float(
                cfg.training.get('p_random_context_finetune', 0.0)
            ),

            # Logging
            log_every_n_steps=cfg.training.log_every_n_steps,
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_dir = Path(cfg.data.checkpoint_dir) / cfg.model.name / f"pretrain_{is_pretrain}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        filename=cfg.checkpoint.filename,
        # Must be forwarded: with Lightning's default (True) each metric name
        # is prefixed again on top of the template's own text, giving names
        # with '=' characters that Hydra's override grammar chokes on.
        auto_insert_metric_name=bool(
            cfg.checkpoint.get('auto_insert_metric_name', False)
        ),
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
    
    # EMA: pretrain, or a finetune whose joint-loss target is an EMA
    # (joint_target: ema); the same callback drives FinetuneModule.update_target_encoder.
    joint_ema = (not is_pretrain
                 and str(cfg.training.loss.get('joint_target', 'frozen')) == 'ema'
                 and float(cfg.training.loss.get('lambda_joint', 0.0)) > 0)
    if (is_pretrain and cfg.training.ema.enabled) or joint_ema:
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
    
    # `limit_val_batches` matters: the validation split of the full corpus is
    # over a million windows (2109 batches), and with val_check_interval=0.1
    # that would be more compute than the training itself. A few hundred
    # batches give a val_loss stable well under the differences we care about.
    trainer_kwargs = dict(
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
        deterministic=cfg.trainer.deterministic,
        strategy=cfg.trainer.strategy,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
    )
    for key in ('limit_train_batches', 'limit_val_batches'):
        if cfg.trainer.get(key) is not None:
            trainer_kwargs[key] = cfg.trainer.get(key)
    apply_schedule_fraction(trainer_kwargs, cfg)

    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Resume an interrupted run (optimizer/scheduler/step restored by
    # Lightning): '+training.resume_ckpt="<ckpt>"'. Default None = unchanged.
    trainer.fit(pl_module, datamodule=datamodule,
                ckpt_path=cfg.training.get("resume_ckpt", None))
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()