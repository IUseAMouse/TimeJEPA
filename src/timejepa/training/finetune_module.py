"""
PyTorch Lightning Module for supervised finetuning.

Uses the pretrained encoder and predictor to forecast actual values.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
import logging

from ..models.jepa_tst import JEPATST, filter_loadable, grow_future_query_table
from .utils.metrics import (
    compute_forecasting_metrics,
    mse,
    mae,
    weighted_quantile_loss,
    jepa_loss,
)
from . import critic

logger = logging.getLogger(__name__)


class FinetuneModule(pl.LightningModule):
    """
    Lightning Module for supervised finetuning.
    
    Training modes:
        - 'linear_probe': Freeze encoder+predictor, train only decoder
        - 'full_finetune': Train encoder + predictor + decoder
        - 'gradual_unfreeze': Start frozen, gradually unfreeze layers
    
    Workflow:
        1. Load pretrained encoder + predictor weights
        2. Switch model to finetune mode
        3. Apply freezing strategy
        4. Train with supervised forecasting loss (MSE on actual values)
    """
    
    def __init__(
        self,
        model: JEPATST,
        pretrained_encoder_path: Optional[str] = None,
        
        # Finetuning strategy
        finetune_mode: Literal['linear_probe', 'full_finetune', 'gradual_unfreeze'] = 'linear_probe',
        unfreeze_after_epoch: int = 5,
        
        # Loss
        loss_type: Literal['mse', 'mae', 'huber'] = 'mse',
        huber_delta: float = 1.0,

        # Context-geometry randomization, train only (validation keeps the
        # native geometry). Without it the decoder only ever sees one context
        # length while the encoder was pretrained on many. The horizon stays
        # fixed in finetune. A separate probability key from the pretrain's
        # p_random_context.
        context_lengths: Optional[List[int]] = None,
        p_random_context_finetune: float = 0.0,

        # Invariance anchor: lambda*MSE(z_hat, z_tgt) kept at finetune, target
        # encoder frozen, so the finetune does not destroy the pretrained
        # latent prediction.
        lambda_anchor: float = 0.0,

        # Joint loss: a JEPA term on the true target during finetune.
        # Mutually exclusive with lambda_anchor. joint_target: 'frozen' (copy
        # of the loaded online encoder) or 'ema' (updated by EMACallback).
        # joint_contextualized: encode [ctx || y] (refused on a
        # cross-resolution model). joint_sigreg: add the pretrain's SIGReg
        # regularizer on the context embeddings.
        lambda_joint: float = 0.0,
        joint_target: Literal['frozen', 'ema'] = 'frozen',
        joint_contextualized: bool = False,
        joint_sigreg: bool = False,
        sigreg_config: Optional[Dict[str, Any]] = None,

        # Critic loop: after the head's fan, N gradient steps of the fan's
        # center down the energy E(x, y_hat) (timejepa.training.critic), a
        # pinball at every step, one backward. critic_steps: the N values
        # sampled per batch ([] = off; eval runs max(critic_steps)). Route
        # 'A' detaches z_pred inside the energy, 'B' keeps it in the graph.
        # Requires lambda_joint > 0.
        critic_steps: Optional[List[int]] = None,
        critic_alpha: float = 0.0,
        critic_route: Literal['A', 'B'] = 'A',
        critic_target: Literal['center', 'fan'] = 'center',
        critic_energy: Literal['cos', 'mse'] = 'cos',
        critic_contextualized: bool = False,
        critic_step_weights: Literal['uniform', 'last'] = 'uniform',
        critic_noise: float = 0.0,
        critic_batch_fraction: float = 1.0,
        critic_max_abs_delta: float = 5.0,
        critic_step_norm: bool = True,

        # Denoising score matching on the energy: the true target is
        # perturbed (level shift, noise, slope, or the head's own median) and
        # the energy's descent direction at the perturbed point must point
        # back to the truth (critic.score_cos). Route 'B' keeps z_pred in the
        # graph so the predictor also learns from it.
        lambda_score: float = 0.0,
        score_route: Literal['A', 'B'] = 'B',
        score_perturb: Optional[Dict[str, float]] = None,
        score_batch_fraction: float = 0.5,
        score_contextualized: bool = False,
        score_energy: Literal['cos', 'mse'] = 'cos',
        score_valley_delta: float = 0.1,

        # Merge the query table of a short-horizon checkpoint into a
        # long-horizon model instead of dropping it (see
        # grow_future_query_table in jepa_tst.py). Without the flag a
        # mismatch is a loud failure.
        extend_horizon_queries: bool = False,
        
        # Optimizer
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        
        # LR Scheduler
        warmup_epochs: float = 0.1,
        max_epochs: int = 50,
        # Fraction of the planned epochs the run will actually get: the cosine
        # anneals to min_lr at that point instead of at the end of max_epochs.
        # train.py bounds the run to the same fraction.
        schedule_fraction: float = 1.0,
        lr_scheduler: Literal['cosine', 'linear', 'plateau', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Regularization
        dropout: float = 0.1,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.model.set_pretrain_mode(False)
        logger.info("Model switched to finetune mode")

        # Set before loading: load_pretrained_encoder reads this attribute.
        self.extend_horizon_queries = bool(extend_horizon_queries)
        self.lambda_anchor = float(lambda_anchor)
        if self.lambda_anchor > 0 and finetune_mode == 'linear_probe':
            # Encoder and predictor are frozen in linear_probe: the term
            # would be a constant added to the loss.
            raise ValueError(
                "lambda_anchor > 0 with finetune_mode='linear_probe': the "
                "anchor would have no gradient (everything frozen) and would "
                "skew val_loss.")

        # Joint-loss, critic and score settings, validated before loading.
        self.lambda_joint = float(lambda_joint)
        self.joint_target = str(joint_target)
        self.joint_contextualized = bool(joint_contextualized)
        self.joint_sigreg = bool(joint_sigreg)
        self.sigreg_config = dict(sigreg_config or {})
        steps = list(range(int(critic_steps) + 1)) if isinstance(critic_steps, int) \
            else sorted(set(int(n) for n in (critic_steps or [])))
        self.critic_steps = [n for n in steps if n >= 0]
        self.critic_n_max = max(self.critic_steps) if self.critic_steps else 0
        self.critic_alpha = float(critic_alpha)
        self.critic_route = str(critic_route)
        self.critic_target = str(critic_target)
        self.critic_energy = str(critic_energy)
        self.critic_contextualized = bool(critic_contextualized)
        self.critic_step_weights = str(critic_step_weights)
        self.critic_noise = float(critic_noise)
        self.critic_batch_fraction = float(critic_batch_fraction)
        self.critic_max_abs_delta = float(critic_max_abs_delta)
        self.critic_step_norm = bool(critic_step_norm)
        self.lambda_score = float(lambda_score)
        self.score_route = str(score_route)
        default_perturb = {'level': 0.35, 'noise': 0.25, 'slope': 0.15, 'forecast': 0.25}
        self.score_perturb = dict(score_perturb) if score_perturb else default_perturb
        self.score_batch_fraction = float(score_batch_fraction)
        self.score_contextualized = bool(score_contextualized)
        self.score_energy = str(score_energy)
        self.score_valley_delta = float(score_valley_delta)
        self._needs_latents = (self.lambda_joint > 0 or self.critic_n_max > 0
                               or self.lambda_score > 0)
        if self.lambda_score > 0:
            if finetune_mode == 'linear_probe':
                raise ValueError("lambda_score > 0 with finetune_mode='linear_probe': "
                                 "the score term needs a trainable encoder")
            if self.score_route not in ('A', 'B'):
                raise ValueError(f"score_route must be 'A' or 'B', got {score_route!r}")
            if self.score_energy not in ('cos', 'mse'):
                raise ValueError(f"score_energy must be 'cos' or 'mse', got {score_energy!r}")
            for k in self.score_perturb:
                if k not in critic.PERTURB_KINDS:
                    raise ValueError(f"unknown score perturbation {k!r} "
                                     f"(choose from {critic.PERTURB_KINDS})")
            if sum(float(v) for v in self.score_perturb.values()) <= 0:
                raise ValueError("score_perturb weights sum to zero")
            if (float(self.score_perturb.get('forecast', 0.0)) > 0
                    and not getattr(self.model.decoder, 'is_probabilistic', False)):
                raise ValueError("score perturbation 'forecast' needs a quantile head "
                                 "(the head's median is the perturbed point)")
            if not (0.0 < self.score_batch_fraction <= 1.0):
                raise ValueError("score_batch_fraction must be in (0, 1]")
        if self.lambda_joint > 0 or self.critic_n_max > 0 or self.lambda_score > 0:
            # Printed at init so a launch can be audited from the log alone.
            logger.info(
                f"H2b/S6 settings: lambda_joint={self.lambda_joint} joint_target={self.joint_target} "
                f"joint_sigreg={self.joint_sigreg} | critic_steps={self.critic_steps} "
                f"critic_alpha={self.critic_alpha} critic_route={self.critic_route} "
                f"critic_target={self.critic_target} critic_energy={self.critic_energy} "
                f"critic_batch_fraction={self.critic_batch_fraction} "
                f"critic_step_norm={getattr(self, 'critic_step_norm', True)} | "
                f"lambda_score={self.lambda_score} score_route={self.score_route} "
                f"score_perturb={self.score_perturb} score_batch_fraction={self.score_batch_fraction} "
                f"score_energy={self.score_energy} score_contextualized={self.score_contextualized}")
        if self.lambda_joint > 0 and self.lambda_anchor > 0:
            raise ValueError("lambda_joint and lambda_anchor are mutually exclusive "
                             "(the same latent MSE would be counted twice; the "
                             "joint loss with frozen standalone targets IS the anchor)")
        if self.lambda_joint > 0 and finetune_mode == 'linear_probe':
            raise ValueError("lambda_joint > 0 with finetune_mode='linear_probe': "
                             "encoder and predictor are frozen, the term would "
                             "have no gradient and would skew val_loss.")
        if self.joint_target not in ('frozen', 'ema'):
            raise ValueError(f"joint_target must be 'frozen' or 'ema', got {joint_target!r}")
        has_film = getattr(self.model.predictor, 'w_film', None) is not None
        if (self.joint_contextualized or self.critic_contextualized
                or self.score_contextualized) and has_film:
            raise ValueError("contextualized candidate encoding is not defined on a "
                             "cross-resolution model (context and target grids differ)")
        if self.critic_n_max > 0:
            if self.lambda_joint <= 0:
                raise ValueError("critic_steps > 0 requires lambda_joint > 0: without "
                                 "the joint term the encoder has no reason to keep a "
                                 "usable energy landscape (S6 is gated behind H2b)")
            if finetune_mode == 'linear_probe':
                raise ValueError("critic_steps > 0 with finetune_mode='linear_probe': "
                                 "the descent needs a trainable encoder")
            if not getattr(self.model.decoder, 'is_probabilistic', False):
                raise ValueError("critic_steps > 0 requires a quantile head (the loop "
                                 "refines a fan)")
            if self.critic_route not in ('A', 'B'):
                raise ValueError(f"critic_route must be 'A' or 'B', got {critic_route!r}")
            if self.critic_target not in ('center', 'fan'):
                raise ValueError(f"critic_target must be 'center' or 'fan', got {critic_target!r}")
            if self.critic_energy not in ('cos', 'mse'):
                raise ValueError(f"critic_energy must be 'cos' or 'mse', got {critic_energy!r}")
            if self.critic_step_weights not in ('uniform', 'last'):
                raise ValueError(f"critic_step_weights must be 'uniform' or 'last', "
                                 f"got {critic_step_weights!r}")
            if not (0.0 < self.critic_batch_fraction <= 1.0):
                raise ValueError("critic_batch_fraction must be in (0, 1]")
            if self.critic_alpha <= 0:
                raise ValueError("critic_steps > 0 requires critic_alpha > 0")

        # Load pretrained weights if provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)
        if self.lambda_anchor > 0 or self.lambda_joint > 0:
            # load_pretrained_encoder skips the target_encoder keys, so the
            # target encoder still holds random construction weights. Copy
            # the loaded online encoder instead (exact as the EMA rate goes
            # to 1 at the end of pretrain).
            self.model.target_encoder.copy_from(self.model.online_encoder)
            logger.info("G9.3 anchor / H2b joint: target_encoder <- copy of the "
                        f"loaded online encoder (lambda_anchor={self.lambda_anchor}, "
                        f"lambda_joint={self.lambda_joint})")

        self.finetune_mode = finetune_mode
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._apply_finetune_strategy(finetune_mode)

        self.loss_type = loss_type
        self.huber_delta = huber_delta

        self.context_lengths = list(context_lengths) if context_lengths else None
        self.p_random_context_finetune = float(p_random_context_finetune)

        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
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

    def update_target_encoder(self, step: int, max_steps: int):
        """EMA update of the joint-loss target; called by EMACallback when
        joint_target == 'ema'."""
        self.model.update_target_encoder(step, max_steps)

    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder and predictor weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            # Lightning checkpoint
            state_dict = checkpoint['state_dict']
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k.replace("model.", "").replace("_orig_mod.", "")
                if "target_encoder" in clean_key:
                    continue
                if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                    continue  # runtime buffers
                cleaned_state_dict[clean_key] = v
        elif 'online_encoder' in checkpoint:
            # Format written by save_pretrained_encoder
            cleaned_state_dict = {}
            for component in ['online_encoder', 'predictor', 'patching', 'revin']:
                if component in checkpoint:
                    for k, v in checkpoint[component].items():
                        cleaned_state_dict[f"{component}.{k}"] = v
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        
        # Horizon extension: merge the short query table before
        # filter_loadable, which would otherwise drop it.
        if self.extend_horizon_queries:
            cleaned_state_dict = grow_future_query_table(self.model, cleaned_state_dict)

        # Drop entries whose shape does not match (a point decoder swapped for
        # the quantile head reuses the same key with a different width);
        # strict=False only tolerates missing keys.
        cleaned_state_dict, dropped = filter_loadable(self.model, cleaned_state_dict)
        for key, ckpt_shape, model_shape in dropped:
            logger.info(f"  re-initialising {key}: checkpoint {ckpt_shape} vs model {model_shape}")

        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)

        expected_missing = {'decoder', 'target_encoder', 'revin'}
        critical_missing = [k for k in missing if not any(exp in k for exp in expected_missing)]
        # A checkpoint carrying core weights the model lacks (robust scaler,
        # z_head, w_film) must refuse as much as the reverse: finetuning
        # while amputating the pretrained architecture would be silent.
        core = ('online_encoder.', 'predictor.', 'patching.', 'robust_scaler.')
        critical_missing += [k for k in unexpected if k.startswith(core)]
        
        if critical_missing:
            logger.error(f"Critical missing keys: {critical_missing}")
            raise RuntimeError(f"Failed to load pretrained weights: {critical_missing}")

        logger.info(f"Loaded pretrained weights ({len(cleaned_state_dict)} keys)")
        logger.info(f"  Expected missing (decoder): {len(missing) - len(critical_missing)} keys")
    
    def _apply_finetune_strategy(self, mode: str):
        """Apply freezing strategy based on finetune mode."""
        if mode == 'linear_probe':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            self.model.freeze_target_encoder()
            logger.info("LINEAR PROBE: encoder frozen, predictor + decoder trainable")
        
        elif mode == 'full_finetune':
            self.model.unfreeze_encoder()
            self.model.unfreeze_predictor()
            self.model.unfreeze_patching()
            logger.info("FULL FINETUNE: all components trainable")
        
        elif mode == 'gradual_unfreeze':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            logger.info(f"GRADUAL UNFREEZE: frozen, will unfreeze at epoch {self.unfreeze_after_epoch}")
        
        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")
    
    def on_train_epoch_start(self):
        """Handle gradual unfreezing."""
        if self.finetune_mode == 'gradual_unfreeze':
            if self.current_epoch == self.unfreeze_after_epoch:
                logger.info(
                    f"Epoch {self.current_epoch}: unfreezing encoder, predictor and patching"
                )
                self.model.unfreeze_encoder()
                self.model.unfreeze_predictor()
                self.model.unfreeze_patching()
    
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for forecasting."""
        return self.model.forecast(context)
    
    def _masked_point_loss(self, predictions: torch.Tensor,
                           targets: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """compute_loss restricted to the real target steps."""
        if self.loss_type == 'mse':
            el = (predictions - targets).pow(2)
        elif self.loss_type == 'mae':
            el = (predictions - targets).abs()
        elif self.loss_type == 'huber':
            el = nn.functional.huber_loss(predictions, targets,
                                          delta=self.huber_delta,
                                          reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        m = mask.to(el.dtype)
        while m.dim() < el.dim():
            m = m.unsqueeze(-1)
        m = m.expand_as(el)
        return (el * m).sum() / torch.clamp(m.sum(), min=1.0)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute forecasting loss."""
        if self.loss_type == 'mse':
            return mse(predictions, targets)
        elif self.loss_type == 'mae':
            return mae(predictions, targets)
        elif self.loss_type == 'huber':
            return nn.functional.huber_loss(predictions, targets, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _forward_and_loss(self, context: torch.Tensor, target: torch.Tensor,
                          w: Optional[torch.Tensor] = None,
                          target_mask: Optional[torch.Tensor] = None):
        """
        Shared by train/val/test. Returns (loss, results, normalized target).

        With a probabilistic head the loss is the pinball over the whole
        quantile fan, so every quantile receives a gradient; the reported
        point metrics use the median. `w` (the batch's cross-resolution
        pairs, None otherwise) is relayed to forecast. The invariance anchor
        is computed on the raw target, captured before the transforms below.
        """
        raw_target = target
        results = self.model.forecast(context, w=w,
                                      return_representations=self._needs_latents)

        # The pinball compares quantiles in the model's normalized frame, so
        # the target gets the same transforms: robust compression (if any)
        # with the context statistics just set by forecast(), then RevIN.
        if getattr(self.model, 'robust_scaler', None) is not None:
            target = self.model.robust_scaler.transform(target)

        # Context statistics, never the target's own (future leak).
        if self.model.revin is not None:
            target = (target - self.model.revin.mean) / self.model.revin.std

        # `target_mask` [B, pred] marks the real target steps of short-series
        # windows; padded steps carry no gradient.
        if 'quantiles' in results:
            head = self.model.decoder.decoder
            loss = head.loss(results['quantiles'], target, mask=target_mask)
        elif target_mask is None:
            loss = self.compute_loss(results['forecast'], target)
        else:
            loss = self._masked_point_loss(results['forecast'], target,
                                           target_mask)
        loss = self._add_rate_fd(loss, context, raw_target, w)

        self._last_anchor = None
        self._last_joint = None
        self._last_sigreg = None
        self._critic_stats = {}
        self._score_stats = {}
        if self.lambda_anchor > 0:
            # Invariance MSE alone, no SIGReg: the target (frozen encoder) is
            # fixed, nothing can collapse. w follows the forward_finetune
            # rule: explicit w=1 if the FiLM exists.
            w_anchor = w
            if w_anchor is None and hasattr(self.model.predictor, 'w_film'):
                w_anchor = torch.ones(context.shape[0], device=context.device)
            pre = self.model.forward_pretrain(
                context, raw_target,
                contextualized_targets=False, w=w_anchor)
            if target_mask is None:
                anchor = torch.nn.functional.mse_loss(
                    pre['predictions'], pre['targets'])
            else:
                # Latent targets of padded steps are meaningless: keep the
                # anchor on items whose target is entirely real.
                full = target_mask.reshape(target_mask.shape[0], -1).all(dim=1)
                per_item = (pre['predictions'] - pre['targets']).pow(2) \
                    .flatten(1).mean(dim=1)
                anchor = (per_item[full].mean() if bool(full.any())
                          else per_item.sum() * 0.0)
            loss = loss + self.lambda_anchor * anchor
            self._last_anchor = anchor.detach()

        if self.lambda_joint > 0:
            loss = loss + self.lambda_joint * self._joint_term(results, target, target_mask)
        if self.lambda_score > 0:
            # Before the critic loop: in eval the loop replaces the fan by the
            # refined one, and the 'forecast' perturbation needs the raw median.
            loss = loss + self.lambda_score * self._score_term(results, target, target_mask)
        if self.critic_n_max > 0 and 'quantiles' in results:
            loss = self._critic_loop(loss, results, target, target_mask)

        return loss, results, target

    @staticmethod
    def _full_items(target_mask: Optional[torch.Tensor], batch_size: int,
                    device: torch.device) -> torch.Tensor:
        """[B] bool: items whose target is entirely real (latent targets of
        padded steps are meaningless)."""
        if target_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return target_mask.reshape(target_mask.shape[0], -1).all(dim=1)

    @staticmethod
    def _masked_item_mean(per_item: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        return per_item[full].mean() if bool(full.any()) else per_item.sum() * 0.0

    def _joint_term(self, results, target_norm, target_mask):
        """Joint loss: MSE between the predictor's latent (the tensor the head
        consumed) and the target encoder's latent of the true target;
        optional SIGReg on the context embeddings."""
        z_pred = results['future_representations']
        ctx_norm = results['context_norm']
        with torch.no_grad():
            z_y = critic.encode_candidate(self.model, ctx_norm, target_norm,
                                          self.joint_contextualized,
                                          encoder=self.model.target_encoder)
        full = self._full_items(target_mask, z_pred.shape[0], z_pred.device)
        per_item = (z_pred[:, :z_y.shape[1], :] - z_y).pow(2).flatten(1).mean(dim=1)
        joint = self._masked_item_mean(per_item, full)
        self._last_joint = joint.detach()
        if self.joint_sigreg:
            _, comps = jepa_loss(
                z_pred[:, :z_y.shape[1], :], z_y, loss_type='sigreg',
                reduction='mean', sigreg_config=self.sigreg_config,
                context_embeddings=results['context_embeddings'],
                return_components=True)
            reg = comps['sigreg']
            self._last_sigreg = reg.detach()
            joint = joint + float(self.sigreg_config.get('lambda', 1.0)) * reg
        return joint

    def _score_term(self, results, target_norm, target_mask) -> torch.Tensor:
        """Score term: 1 - cos(descent direction of E at a perturbed truth,
        direction to the truth), on a contiguous sub-batch of full-target
        items. Train: random perturbations, create_graph. Eval: seeded
        perturbations without a graph, plus the valley witness (share of
        items where E rises on both sides of the truth along a level shift)."""
        z_pred = results['future_representations']
        ctx_norm = results['context_norm']
        B = z_pred.shape[0]
        full = self._full_items(target_mask, B, z_pred.device)
        m = max(1, int(round(B * self.score_batch_fraction)))
        if self.training:
            start = int(torch.randint(B - m + 1, (1,)).item())
            gen = None
        else:
            start = 0
            gen = torch.Generator(device='cpu').manual_seed(0)   # recreated per call, never stored
        idx = torch.arange(start, start + m, device=z_pred.device)
        idx = idx[full[idx]]
        self._score_stats = {'n_items': int(idx.numel())}
        if idx.numel() == 0:
            return z_pred.sum() * 0.0
        kinds = [k for k, v in self.score_perturb.items() if float(v) > 0]
        weights = [float(self.score_perturb[k]) for k in kinds]
        median = None
        if 'forecast' in kinds:
            head = self.model.decoder.decoder
            median = head.median(results['quantiles']).detach()[idx]
        y = target_norm[idx]
        y_tilde, kind_idx = critic.perturb_target(y, kinds, weights, gen, median=median)
        z_e = z_pred[idx].detach() if self.score_route == 'A' else z_pred[idx]
        cos = critic.score_cos(self.model, ctx_norm[idx], y, y_tilde, z_e,
                               mode=self.score_energy,
                               contextualized=self.score_contextualized,
                               create_graph=self.training)
        st = self._score_stats
        st['cos'] = float(cos.detach().mean())
        for i, k in enumerate(kinds):
            sel = kind_idx == i
            st[f'frac_{k}'] = float(sel.float().mean())
            if bool(sel.any()):
                st[f'cos_{k}'] = float(cos.detach()[sel].mean())
        if not self.training:
            st['valley_frac'] = float(critic.valley_witness(
                self.model, ctx_norm[idx], y, z_pred[idx].detach(),
                delta=self.score_valley_delta, mode=self.score_energy,
                contextualized=self.score_contextualized))
        return (1.0 - cos).mean()

    def _critic_loop(self, loss, results, target_norm, target_mask):
        """Critic loop: N descent steps of the fan's center down the energy,
        a pinball at each step. Train: the step pinballs are added to the
        loss (one backward). Eval: N = max, deterministic, and the loss and
        results become those of the refined fan (the deployed forecast)."""
        head = self.model.decoder.decoder
        if self.training:
            n = int(self.critic_steps[torch.randint(len(self.critic_steps), (1,)).item()])
        else:
            n = self.critic_n_max
        self._critic_stats = {'n_steps': float(n)}
        if n == 0:
            return loss
        fan0 = results['quantiles']
        z_pred = results['future_representations']
        z_for_E = z_pred.detach() if self.critic_route == 'A' else z_pred
        ctx_norm = results['context_norm']
        B = fan0.shape[0]
        full = self._full_items(target_mask, B, fan0.device)
        sub = slice(None)
        if self.training and self.critic_batch_fraction < 1.0:
            m = max(1, int(round(B * self.critic_batch_fraction)))
            start = int(torch.randint(B - m + 1, (1,)).item())
            sub = slice(start, start + m)
        mask_sub = None if target_mask is None else target_mask[sub]
        # Validation/test run under no_grad (Lightning); the descent still
        # needs a graph for the gradient w.r.t. the fan.
        with torch.enable_grad():
            out = critic.refine_loop(
                self.model, ctx_norm[sub], fan0[sub], z_for_E[sub], n,
                alpha=self.critic_alpha, mode=self.critic_energy,
                contextualized=self.critic_contextualized, target=self.critic_target,
                median_idx=head.median_idx, create_graph=self.training,
                noise_sigma=self.critic_noise if self.training else 0.0,
                item_weight=full[sub].to(fan0.dtype),
                max_abs_delta=self.critic_max_abs_delta,
                step_norm=self.critic_step_norm)
        pinballs = [head.loss(f, target_norm[sub], mask=mask_sub) for f in out['fans'][1:]]
        e = torch.stack([en.reshape(en.shape[0], -1).mean(dim=1) for en in out['energies']])
        e_full = e[:, full[sub]] if bool(full[sub].any()) else e
        delta = sum(out['deltas'])
        stats = self._critic_stats
        stats['energy_0'] = float(e_full[0].mean())
        stats['energy_N'] = float(e_full[-1].mean())
        stats['energy_drop'] = stats['energy_0'] - stats['energy_N']
        stats['pinball_0'] = float(head.loss(fan0[sub], target_norm[sub], mask=mask_sub).detach())
        for i, pb in enumerate(pinballs, start=1):
            stats[f'pinball_{i}'] = float(pb.detach())
        stats['pinball_N'] = stats[f'pinball_{n}']
        # The absolute pinballs are dominated by a few extreme items of the
        # sub-batch; the gain and its relative version are readable.
        stats['pinball_gain'] = stats['pinball_0'] - stats['pinball_N']
        stats['pinball_rel_gain'] = stats['pinball_gain'] / max(stats['pinball_0'], 1e-8)
        stats['delta_abs'] = float(delta.detach().abs().mean())
        stats['delta_clipped_frac'] = float(
            (delta.detach().abs() >= self.critic_max_abs_delta - 1e-6).float().mean())
        if self.training:
            if self.critic_step_weights == 'last':
                return loss + pinballs[-1]
            return loss + sum(pinballs) / len(pinballs)
        # eval: the deployed forecast is the refined fan
        refined = out['fans'][-1].detach()
        if sub != slice(None):
            refined_full = fan0.detach().clone(); refined_full[sub] = refined
            refined = refined_full
        results['quantiles'] = refined
        results['forecast'] = head.median(refined)
        if self.model.revin is not None or getattr(self.model, 'robust_scaler', None) is not None:
            denorm = self.model.decoder.revin.denormalize_target_space(refined) \
                if getattr(self.model.decoder, 'revin', None) is not None else refined
            if getattr(self.model, 'robust_scaler', None) is not None:
                denorm = self.model.robust_scaler.inverse(denorm)
            results['quantiles_denorm'] = denorm
            results['forecast_denorm'] = head.median(denorm)
        return head.loss(refined, target_norm, mask=target_mask)
    
    def _maybe_crop_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Sample a context length once per batch and keep the most recent
        steps, as a shorter context would at inference. Mirrors
        JEPAPretrainModule._randomize_geometry without the horizon part.
        """
        if not self.context_lengths or self.p_random_context_finetune <= 0.0:
            return context
        if torch.rand(1).item() >= self.p_random_context_finetune:
            return context
        eligible = [L for L in self.context_lengths if L <= context.shape[1]]
        if not eligible:
            return context
        length = int(eligible[torch.randint(len(eligible), (1,)).item()])
        return context[:, -length:]

    def _add_rate_fd(self, loss, context, raw_target, w):
        """A model whose rate trains by finite differences (RateINForecaster,
        rate_grad='fd') gets its straight-through term added after a
        training forward; other models and evaluation return `loss` as is.
        The per-item loss is this module's pinball in the normalized frame
        of the shifted forward (target mask ignored: the term is a training
        signal for the rate, not the reported loss)."""
        if not (self.training and getattr(self.model, 'rate_grad', None) == 'fd'):
            return loss
        head = self.model.decoder.decoder
        levels = torch.tensor(head.quantile_levels, dtype=raw_target.dtype, device=raw_target.device)

        def per_item(results, target):
            if getattr(self.model, 'robust_scaler', None) is not None:
                target = self.model.robust_scaler.transform(target)
            if self.model.revin is not None:
                target = (target - self.model.revin.mean) / self.model.revin.std
            q = results['quantiles']
            t = target[..., 0] if target.ndim == 3 else target
            d = t.unsqueeze(-1) - q
            return (2.0 * torch.maximum(levels * d, (levels - 1.0) * d)).mean(dim=(1, 2))

        return loss + self.model.fd_surrogate(context, raw_target, w, raw_target.shape[1], per_item)

    def _log_rate_diagnostics(self, on_step: bool, on_epoch: bool) -> None:
        """A model that owns its sampling rate (RateINForecaster) reports
        `rate_diagnostics()`; other models have no such hook and log nothing."""
        diag = getattr(self.model, 'rate_diagnostics', None)
        if not callable(diag):
            return
        for key, value in diag().items():
            self.log(f'rate/{key}', value, on_step=on_step, on_epoch=on_epoch, logger=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        context = batch['context']
        target = batch['target']

        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        # Cross-resolution pairs: per-item w when the dataset emits it.
        w = batch.get('w')
        if w is not None:
            w = w.float()
            self.log('aug/w_neq1_frac', (w != 1).float().mean(),
                     on_step=True, on_epoch=False, logger=True)
            self.log('aug/w_mean', w.mean(),
                     on_step=True, on_epoch=False, logger=True)

        # Train only; validation_step and test_step keep the native geometry.
        context = self._maybe_crop_context(context)
        self.log('geometry/context_len', float(context.shape[1]),
                 on_step=True, on_epoch=False, logger=True)

        target_mask = batch.get('target_mask')
        if target_mask is not None:
            # Share of the batch made of short-series windows
            short = (~target_mask.reshape(target_mask.shape[0], -1).all(dim=1))
            self.log('aug/short_frac', short.float().mean(), on_step=True,
                     on_epoch=False)
        loss, results, target = self._forward_and_loss(
            context, target, w=w, target_mask=target_mask)
        predictions = results['forecast']

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._log_rate_diagnostics(on_step=True, on_epoch=False)
        if self._last_anchor is not None:
            self.log('train_loss/anchor', self._last_anchor,
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self._last_joint is not None:
            self.log('train_loss/joint', self._last_joint,
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self._last_sigreg is not None:
            self.log('train_loss/sigreg', self._last_sigreg,
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
        for key, value in self._critic_stats.items():
            self.log(f'critic/{key}', value, on_step=True, on_epoch=True,
                     logger=True, sync_dist=True)
        if self._score_stats:
            self.log('train_loss/score', 1.0 - self._score_stats.get('cos', 1.0),
                     on_step=True, on_epoch=True, logger=True, sync_dist=True)
            for key, value in self._score_stats.items():
                self.log(f'score/{key}', value, on_step=True, on_epoch=True,
                         logger=True, sync_dist=True)
        
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                metrics = compute_forecasting_metrics(predictions, target)
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
        
        loss, results, target = self._forward_and_loss(
            context, target, w=batch.get('w'),
            target_mask=batch.get('target_mask'))
        predictions = results['forecast']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._log_rate_diagnostics(on_step=False, on_epoch=True)
        if self._last_anchor is not None:
            self.log('val_loss/anchor', self._last_anchor,
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if self._last_joint is not None:
            self.log('val_loss/joint', self._last_joint,
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
        for key, value in self._critic_stats.items():
            self.log(f'val_critic/{key}', value, on_step=False, on_epoch=True,
                     logger=True, sync_dist=True)
        if self._score_stats:
            self.log('val_loss/score', 1.0 - self._score_stats.get('cos', 1.0),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            for key, value in self._score_stats.items():
                self.log(f'val_score/{key}', value, on_step=False, on_epoch=True,
                         logger=True, sync_dist=True)

        # WQL is the metric GIFT-Eval ranks on, so track it directly.
        if 'quantiles' in results:
            wql = weighted_quantile_loss(
                results['quantiles'].permute(2, 0, 1),
                target.squeeze(-1) if target.shape[-1] == 1 else target,
                list(results['quantile_levels']),
            )
            self.log('val_wql', wql, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        metrics = compute_forecasting_metrics(predictions, target)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        context = batch['context']
        target = batch['target']
        
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        loss, results, target = self._forward_and_loss(
            context, target, w=batch.get('w'),
            target_mask=batch.get('target_mask'))
        predictions = results['forecast']

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        metrics = compute_forecasting_metrics(predictions, target)
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': target,
            'metrics': metrics
        }
    
    def configure_optimizers(self):
        """Configure optimizer with different LR for encoder vs decoder."""
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            # The EMA target encoder is never trained, in any mode.
            if name.startswith('target_encoder'):
                continue

            # Register every parameter, frozen ones included: a frozen
            # parameter has grad=None and AdamW skips it, so the existing
            # optimizer picks it up when gradual_unfreeze unfreezes it.
            # Filtering on requires_grad here would leave the unfrozen
            # weights untrained.
            if 'decoder' in name:
                decoder_params.append(param)
            else:
                encoder_params.append(param)
        
        param_groups = []
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.learning_rate * self.encoder_lr_multiplier,
                'name': 'encoder'
            })
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.learning_rate,
                'name': 'decoder'
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        logger.info(f"Optimizer groups: {[g['name'] for g in param_groups]}")
        
        optimizer = torch.optim.AdamW(
            param_groups,
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
            
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=self.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        
        elif self.lr_scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=self.min_lr)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
    
    def on_train_epoch_end(self):
        """Log learning rates."""
        optimizer = self.optimizers()
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.log(f'lr_{group_name}', param_group['lr'], on_epoch=True, prog_bar=True, sync_dist=True)

        # ErrorSignal witness: the spread gate is zero-initialized; if it
        # stays at zero the decoder ignores z.
        head = getattr(getattr(self.model, 'decoder', None), 'decoder', None)
        z_gate = getattr(head, 'z_gate', None)
        if z_gate is not None:
            with torch.no_grad():
                absmean = torch.cat(
                    [z_gate.weight.abs().flatten(), z_gate.bias.abs().flatten()]
                ).mean()
            self.log('esjepa/gate_absmean', absmean, on_epoch=True, sync_dist=True)