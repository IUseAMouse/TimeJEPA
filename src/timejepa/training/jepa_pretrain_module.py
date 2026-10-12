"""
PyTorch Lightning Module for JEPA pretraining: the model learns to predict
representations of future timesteps from the context.
"""

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal
from pathlib import Path

from ..models.jepa_tst import JEPATST
from .utils.metrics import jepa_loss, compute_pretrain_metrics

logger = logging.getLogger(__name__)


class JEPAPretrainModule(pl.LightningModule):
    """
    Lightning Module for JEPA pretraining.
    
    Training workflow:
        1. Get batch with 'context' (past) and 'target' (future)
        2. Context -> Online Encoder -> context representations
        3. Target -> Target Encoder (EMA) -> target representations
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

        # Anti-collapse: regularize the encoder output (what downstream
        # consumes), not only the predictor output.
        regularize_context: bool = True,

        # I-JEPA-style targets: encode [context + target] and slice, instead of
        # encoding the future window in isolation.
        contextualized_targets: bool = True,

        # Ablation: regress onto the raw future patches instead of the target
        # encoder's latents (the TimesFM-style objective). Only the scoring
        # space changes; task, geometry, corpus and optimizer are untouched.
        reconstruction_target: bool = False,

        # Cross-resolution JEPA: reads `w = k2/k1` per item from the batch and
        # forwards it to the predictor (FiLM). Requires standalone targets:
        # concatenating [ctx + target] has no meaning when the two live on
        # different grids, so the guard below refuses the combination.
        cross_resolution: bool = False,

        # ErrorSignal: auxiliary loss smooth_l1(z_pred, z_target) on the
        # smoothing-residual statistics (the model exposes
        # z_predictions/z_targets when built with model.error_signal=true).
        # lambda_z weights the z path against invariance.
        error_signal: bool = False,
        lambda_z: float = 0.1,

        # Input-geometry randomization: skill peaks at the training context
        # length and collapses on both sides (the model memorizes a fixed
        # patch count), so the geometry is sampled per batch.
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
        # Fraction of the planned epochs the run will actually get: the cosine
        # anneals to min_lr at that point instead of at the end of max_epochs.
        # train.py bounds the run to the same fraction.
        schedule_fraction: float = 1.0,
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
        
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.model.set_pretrain_mode(True)
        self.model.freeze_target_encoder()  # updated by EMA, never by backprop

        self.loss_type = loss_type
        self.vicreg_weights = vicreg_weights
        self.sigreg_config = sigreg_config or {}
        self.regularize_context = regularize_context
        self.contextualized_targets = contextualized_targets

        self.cross_resolution = bool(cross_resolution)
        if self.cross_resolution and contextualized_targets:
            raise ValueError(
                "cross_resolution=True requires contextualized_targets=false: "
                "the contextualized target concatenates [context + target] in "
                "sample steps, physically wrong when context and target are at "
                "different resolutions. Set "
                "training.contextualized_targets: false in the arm config."
            )

        self.error_signal = bool(error_signal)
        self.lambda_z = float(lambda_z)
        if self.error_signal and reconstruction_target:
            raise ValueError(
                "error_signal=True is incompatible with "
                "reconstruction_target=True: the reconstruction arm replaces "
                "the whole latent objective (_compute_loss bypasses "
                "jepa_loss), a z loss on top would be incoherent."
            )
        if self.error_signal and not getattr(model, 'error_signal', False):
            raise ValueError(
                "error_signal=True on the module but the model was built "
                "without model.error_signal - the z_predictions/z_targets "
                "keys would not exist. Both flags come from the same config "
                "key (model.error_signal), check the plumbing."
            )

        # Reconstruction head: d_model -> patch_size * num_features, the
        # inverse of the patch projection, so its shape is derived from the
        # model. It is not part of JEPATST: the finetune loads the checkpoint
        # by component name and ignores it.
        self.reconstruction_target = bool(reconstruction_target)
        if self.reconstruction_target:
            proj = model.patching.projection
            self.recon_head = nn.Linear(proj.out_features, proj.in_features)
            print("  WARNING: ABLATION: reconstruction target (raw patches), NOT latent")

        self.context_lengths = list(context_lengths) if context_lengths else None
        self.p_random_context = float(p_random_context)
        self.horizon_lengths = list(horizon_lengths) if horizon_lengths else None
        self.p_random_horizon = float(p_random_horizon)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.schedule_fraction = float(schedule_fraction)
        if not (0.0 < self.schedule_fraction <= 1.0):
            raise ValueError('schedule_fraction must be in (0, 1]')
        self.lr_scheduler_type = lr_scheduler
        self.min_lr = min_lr

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
        Sample the input geometry once per batch.

        Per-batch sampling keeps every tensor rectangular, so no padding or
        masking is needed: the encoder (RoPE, no positional table) simply sees
        a different patch count. The context keeps its most recent steps, as
        a shorter context would at inference; the target keeps its first steps.
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

    def _scored_pair(self, target, outputs):
        """
        The (prediction, target) pair the loss is computed on.

        JEPA: the predictor's latents against the target encoder's latents.
        Reconstruction: the same latents pushed through `recon_head` into
        value space, against the raw future patches taken in the model's
        normalized space (RevIN with the context statistics, as in
        `forward_pretrain`). The patch spans are those of the standalone
        target window, so both arms predict the same number of patches.
        """
        if not self.reconstruction_target:
            return outputs['predictions'], outputs['targets']

        patching = self.model.patching
        revin = self.model.revin
        x = (target - revin.mean) / revin.std if revin is not None else target

        # Mirrors Patching.forward's padding rule (pinned by a test).
        if patching.padding:
            remainder = (x.shape[1] - patching.patch_size) % patching.stride
            if remainder != 0:
                pad = patching.stride - remainder
                x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)

        patches = x.unfold(dimension=1, size=patching.patch_size, step=patching.stride)
        patches = patches.transpose(2, 3).reshape(x.shape[0], patches.shape[1], -1)

        return self.recon_head(outputs['predictions']), patches

    def _compute_loss(self, predictions, targets, outputs):
        """
        Single entry point for training_step and validation_step, so the two
        score the same objective.
        """
        if self.reconstruction_target:
            # Not routed through jepa_loss: raw patches are fixed, so there is
            # nothing to collapse and no anti-collapse term is needed.
            #
            # Huber rather than MSE: RevIN normalizes the target with the
            # context statistics, whose scale floors at sqrt(1e-5) on a
            # near-constant context (a flat sensor, solar at night), so any
            # movement in the future window lands at thousands of sigma.
            # Under MSE one such batch would outweigh millions of normal ones.
            # Huber bounds each element's gradient and keeps every window.
            loss = torch.nn.functional.smooth_l1_loss(predictions, targets)

            # Keep the pathology visible: if this climbs into the thousands,
            # the corpus is feeding degenerate windows.
            with torch.no_grad():
                absmax = targets.abs().max()
            return loss, {'loss': loss, 'reconstruction_huber': loss,
                          'target_absmax': absmax}

        loss, components = jepa_loss(
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

        # ErrorSignal auxiliary z loss. Huber bounds the contribution of
        # extreme-residual patches. No anti-collapse term: the target is
        # fixed data; collapse to the marginal mean is monitored by
        # esjepa/z_pred_std_ratio instead.
        if self.error_signal and 'z_predictions' in outputs:
            z_loss = torch.nn.functional.smooth_l1_loss(
                outputs['z_predictions'], outputs['z_targets'])
            loss = loss + self.lambda_z * z_loss
            components = dict(components)
            components['z'] = z_loss
            components['loss'] = loss

        return loss, components
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with 'context' (past) and 'target' (future)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        context = batch['context']  # [B, context_length] or [B, context_length, C]
        target = batch['target']    # [B, prediction_length] or [B, prediction_length, C]

        # Univariate case: add the channel dimension
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

        # Input-augmentation witnesses, averaged over the epoch. `w` only
        # exists on the cross-resolution arm.
        rf = batch.get('resolution_factor')
        if rf is not None and torch.is_tensor(rf):
            rf = rf.float()
            self.log('aug/multires_frac', (rf > 1).float().mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('aug/resolution_factor_mean', rf.mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
        w = batch.get('w') if self.cross_resolution else None
        if w is not None:
            w = w.float()
            # w_neq1_frac alone is dominated by k1=1<k2 pairs, which fit in
            # short chunks; w<1 (k1>k2) needs long chunks, hence w_lt1_frac.
            self.log('aug/w_neq1_frac', (w != 1).float().mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('aug/w_lt1_frac', (w < 1).float().mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('aug/w_mean', w.mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)

        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets,
            w=w,
        )

        predictions, targets = self._scored_pair(target, outputs)
        loss, components = self._compute_loss(predictions, targets, outputs)

        # Cross-resolution convergence witness: per-item loss conditioned on
        # w. If `wneq1` stalls while `w1` drops, the cross-resolution items
        # are not converging.
        if w is not None and bool((w != 1).any()):
            with torch.no_grad():
                per_item = (predictions - targets).pow(2).mean(dim=(1, 2))
                mask = (w != 1)
                self.log('train_loss/wneq1', per_item[mask].mean(),
                         on_step=False, on_epoch=True, logger=True, sync_dist=True)
                if bool((~mask).any()):
                    self.log('train_loss/w1', per_item[~mask].mean(),
                             on_step=False, on_epoch=True, logger=True, sync_dist=True)

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

        # Validation uses the native geometry, never the randomized one, so
        # val_loss stays comparable across epochs and runs. The validation
        # split applies no augmentations, so w is always 1 when it exists.
        w = batch.get('w') if self.cross_resolution else None
        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets,
            w=(w.float() if w is not None else None),
        )

        predictions, targets = self._scored_pair(target, outputs)
        loss, components = self._compute_loss(predictions, targets, outputs)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for key, value in components.items():
            if key == 'loss':
                continue
            self.log(f'val_loss/{key}', value, on_step=False, on_epoch=True,
                     logger=True, sync_dist=True)

        metrics = compute_pretrain_metrics(
            predictions,
            targets,
            context_embeddings=outputs.get('context_embeddings')
        )
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Collapse witnesses
        ctx = outputs.get('context_embeddings')
        if ctx is not None:
            collapse = ctx.std(dim=0).mean()
            self.log('collapse/context_std', collapse, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            # Effective rank: a collapsed encoder concentrates its energy in a
            # few directions. Computed on the first validation batch only.
            if batch_idx == 0:
                eff_rank = self._effective_rank(ctx)
                if eff_rank is not None:
                    self.log('collapse/effective_rank', eff_rank, on_step=False,
                             on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # ErrorSignal witnesses: esjepa/z_corr is the Spearman correlation
        # between the predicted log-RMS (z_pred[..., 0]) and the realized
        # log-RMS of the residual; esjepa/z_pred_std_ratio is
        # std(z_pred)/std(z_target), which detects collapse to the marginal
        # mean.
        if self.error_signal and 'z_predictions' in outputs:
            with torch.no_grad():
                zp = outputs['z_predictions'][..., 0].reshape(-1).float()
                zt = outputs['z_targets'][..., 0].reshape(-1).float()
                self.log('esjepa/z_corr', self._spearman(zp, zt),
                         on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, sync_dist=True)
                ratio = zp.std() / zt.std().clamp_min(1e-9)
                self.log('esjepa/z_pred_std_ratio', ratio, on_step=False,
                         on_epoch=True, logger=True, sync_dist=True)

        return loss

    @staticmethod
    @torch.no_grad()
    def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Spearman correlation (Pearson of ranks) without scipy; ties get
        appearance-order ranks, good enough for monitoring."""
        ra = a.argsort().argsort().float()
        rb = b.argsort().argsort().float()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
        return (ra * rb).sum() / denom

    @torch.no_grad()
    def _effective_rank(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        """
        exp(entropy of the normalized covariance spectrum).

        Uses the eigenvalues of the DxD covariance rather than an SVD of the
        [B*N, D] matrix (same spectrum, much smaller problem). Wrapped in
        try/except: eigensolvers can fail on near-degenerate input, which is
        the collapsed case this metric exists to detect.
        """
        try:
            # autocast must be disabled explicitly: inside the bf16 autocast
            # region the matmul is cast back to bfloat16, which eigvalsh
            # does not support.
            device_type = embeddings.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
                flat = flat - flat.mean(dim=0, keepdim=True)
                cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
                eigvals = torch.linalg.eigvalsh(cov.float()).clamp_min(0)
            total = eigvals.sum()
            if not torch.isfinite(total) or total <= 1e-12:
                # Fully collapsed: every direction is degenerate, rank is 1.
                return torch.ones((), device=embeddings.device)
            p = eigvals / total
            entropy = -(p * (p + 1e-12).log()).sum()
            return entropy.exp()
        except Exception as e:  # noqa: BLE001, diagnostics must never kill a run
            logger.warning(f"effective_rank unavailable this step: {e}")
            return None
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler_type == 'constant':
            return optimizer
        
        # Scheduler steps are optimizer steps (one per accumulate_grad_batches
        # batches), while the dataloader counts batches; without the division
        # the warmup and the cosine would be stretched by that factor.
        accumulate = max(1, int(getattr(self.trainer, 'accumulate_grad_batches', 1) or 1))
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // accumulate
        total_steps = int(self.max_epochs * steps_per_epoch * self.schedule_fraction)
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
        print(f"Saved pretrained encoder to {save_path}")