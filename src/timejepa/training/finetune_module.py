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
)

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

        # Context-geometry randomization, TRAIN ONLY (validation keeps the
        # native geometry so val_loss stays comparable across epochs).
        #
        # Without this the decoder only ever sees one context length, while the
        # encoder was pretrained on many — so evaluating at any other length
        # puts the decoder out of distribution even where the encoder is fine.
        # The context sweep could not separate those two effects; this removes
        # the decoder's share. Horizon stays FIXED in finetune: eval already
        # truncates 128->96, and rolling horizons always use the full native
        # prediction_length per roll, so there is no mismatch to fix there.
        #
        # A separate probability key (not the pretrain's p_random_context) so
        # existing finetune configs keep their exact previous behavior at the
        # default of 0.0.
        context_lengths: Optional[List[int]] = None,
        p_random_context_finetune: float = 0.0,

        # Chantier 2 (horizon natif) — fusionner la table de requêtes d'un
        # checkpoint à horizon COURT dans un modèle à horizon LONG au lieu de la
        # dropper. Opt-in : sans ce flag, un mismatch reste un échec bruyant
        # (critical_missing), le comportement historique. Voir
        # grow_future_query_table dans jepa_tst.py.
        extend_horizon_queries: bool = False,
        
        # Optimizer
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        
        # LR Scheduler
        warmup_epochs: float = 0.1,
        max_epochs: int = 50,
        lr_scheduler: Literal['cosine', 'linear', 'plateau', 'constant'] = 'cosine',
        min_lr: float = 1e-6,
        
        # Regularization
        dropout: float = 0.1,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        self.model.set_pretrain_mode(False)  # Switch to finetune mode
        logger.info("✓ Model switched to finetune mode")
        
        # AVANT le chargement : load_pretrained_encoder lit cet attribut
        # (chemin h512). Le poser après, c'était un AttributeError sur tout
        # finetune lancé avec pretrained_encoder_path — jamais vu avant le
        # premier finetune post-h512 (mix, 2026-08-22) car tiny-full tournait
        # sur le commit pré-h512.
        self.extend_horizon_queries = bool(extend_horizon_queries)

        # Load pretrained weights if provided
        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)
        
        # Apply finetuning strategy
        self.finetune_mode = finetune_mode
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._apply_finetune_strategy(finetune_mode)
        
        # Loss configuration
        self.loss_type = loss_type
        self.huber_delta = huber_delta

        # Context-geometry randomization (train only)
        self.context_lengths = list(context_lengths) if context_lengths else None
        self.p_random_context_finetune = float(p_random_context_finetune)

        # Optimizer params
        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler_type = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
    
    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder and predictor weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            # Clean keys
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k.replace("model.", "").replace("_orig_mod.", "")
                if "target_encoder" in clean_key:
                    continue  # Skip target encoder
                if "revin" in clean_key and (clean_key.endswith('.mean') or clean_key.endswith('.std')):
                    continue  # Skip runtime buffers
                cleaned_state_dict[clean_key] = v
        elif 'online_encoder' in checkpoint:
            # Direct save format from save_pretrained_encoder
            cleaned_state_dict = {}
            for component in ['online_encoder', 'predictor', 'patching', 'revin']:
                if component in checkpoint:
                    for k, v in checkpoint[component].items():
                        cleaned_state_dict[f"{component}.{k}"] = v
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
        
        # Chantier 2 — extension d'horizon opt-in : fusionner la table de
        # requêtes courte AVANT filter_loadable, sinon ce dernier la droppe et
        # le garde critical_missing ci-dessous refuse (comportement voulu hors
        # extension intentionnelle).
        if self.extend_horizon_queries:
            cleaned_state_dict = grow_future_query_table(self.model, cleaned_state_dict)

        # Drop entries whose shape does not match — swapping a point decoder for
        # the quantile head reuses the same key path with a different width, and
        # strict=False does NOT tolerate that (it only tolerates missing keys).
        cleaned_state_dict, dropped = filter_loadable(self.model, cleaned_state_dict)
        for key, ckpt_shape, model_shape in dropped:
            logger.info(f"  ↷ re-initialising {key}: checkpoint {ckpt_shape} vs model {model_shape}")

        # Load weights
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        
        # Check for critical missing keys
        expected_missing = {'decoder', 'target_encoder', 'revin'}
        critical_missing = [k for k in missing if not any(exp in k for exp in expected_missing)]
        # Symétrie des poids d'arm : un checkpoint qui porte des poids core que
        # le modèle n'a pas (arcsinh -> nu, ESJEPA predictor.z_head -> nu, xres
        # w_film -> nu) arrive en 'unexpected' et doit refuser autant que
        # l'inverse — finetuner en amputant l'architecture pré-entraînée serait
        # silencieux. (Durci 2026-08-23, était limité à robust_scaler. ; aligné
        # sur loading.py.)
        core = ('online_encoder.', 'predictor.', 'patching.', 'robust_scaler.')
        critical_missing += [k for k in unexpected if k.startswith(core)]
        
        if critical_missing:
            logger.error(f"❌ Critical missing keys: {critical_missing}")
            raise RuntimeError(f"Failed to load pretrained weights: {critical_missing}")
        
        logger.info(f"✓ Loaded pretrained weights ({len(cleaned_state_dict)} keys)")
        logger.info(f"  Expected missing (decoder): {len(missing) - len(critical_missing)} keys")
    
    def _apply_finetune_strategy(self, mode: str):
        """Apply freezing strategy based on finetune mode."""
        if mode == 'linear_probe':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            self.model.freeze_target_encoder()
            logger.info("✓ LINEAR PROBE: encoder frozen, predictor + decoder trainable")
        
        elif mode == 'full_finetune':
            self.model.unfreeze_encoder()
            self.model.unfreeze_predictor()
            self.model.unfreeze_patching()
            logger.info("✓ FULL FINETUNE: all components trainable")
        
        elif mode == 'gradual_unfreeze':
            self.model.freeze_encoder()
            self.model.freeze_predictor()
            self.model.freeze_patching()
            logger.info(f"✓ GRADUAL UNFREEZE: frozen, will unfreeze at epoch {self.unfreeze_after_epoch}")
        
        else:
            raise ValueError(f"Unknown finetune_mode: {mode}")
    
    def on_train_epoch_start(self):
        """Handle gradual unfreezing."""
        if self.finetune_mode == 'gradual_unfreeze':
            if self.current_epoch == self.unfreeze_after_epoch:
                # B20: this used to call unfreeze_predictor() alone while
                # logging "encoder and predictor" — the encoder and patching
                # stayed frozen forever. Now the action matches the log.
                logger.info(
                    f"Epoch {self.current_epoch}: unfreezing encoder, predictor and patching"
                )
                self.model.unfreeze_encoder()
                self.model.unfreeze_predictor()
                self.model.unfreeze_patching()
    
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for forecasting."""
        return self.model.forecast(context)
    
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

    def _forward_and_loss(self, context: torch.Tensor, target: torch.Tensor):
        """
        Shared by train/val/test.

        With a probabilistic head the loss is the pinball over the whole quantile
        fan, not a point loss on the median — otherwise the outer quantiles would
        receive no gradient at all. The reported point metrics still use the
        median, which is the MAE-optimal estimate and what MASE scores.
        """
        results = self.model.forecast(context)

        # G8.4 — si le modèle compresse (arcsinh robuste), la cible doit subir
        # la MÊME compression avec les stats du contexte (posées par forecast()
        # à l'instant) avant la normalisation RevIN : la pinball compare des
        # quantiles en espace compressé+RevIN, la cible doit y vivre aussi.
        if getattr(self.model, 'robust_scaler', None) is not None:
            target = self.model.robust_scaler.transform(target)

        # Target normalized with the CONTEXT's statistics — never its own, which
        # would leak the future into the normalization.
        if self.model.revin is not None:
            target = (target - self.model.revin.mean) / self.model.revin.std

        if 'quantiles' in results:
            head = self.model.decoder.decoder
            loss = head.loss(results['quantiles'], target)
        else:
            loss = self.compute_loss(results['forecast'], target)

        return loss, results, target
    
    def _maybe_crop_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Sample a context length ONCE PER BATCH and crop from the LEFT (keep the
        most recent history — what a shorter context would actually contain at
        inference). Mirrors JEPAPretrainModule._randomize_geometry, minus the
        horizon part, which stays fixed in finetune by design.
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        context = batch['context']
        target = batch['target']

        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        # Train only — validation_step and test_step keep the native geometry.
        context = self._maybe_crop_context(context)
        # Same observability as the pretrain: without this line there is no way
        # to confirm from W&B that the randomization is actually active.
        self.log('geometry/context_len', float(context.shape[1]),
                 on_step=True, on_epoch=False, logger=True)

        loss, results, target = self._forward_and_loss(context, target)
        predictions = results['forecast']

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
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
        
        loss, results, target = self._forward_and_loss(context, target)
        predictions = results['forecast']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # WQL is the metric GIFT-Eval ranks on, so track it directly rather than
        # inferring it from the point losses.
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
        
        loss, results, target = self._forward_and_loss(context, target)
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
        # Separate parameters
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            # The EMA target encoder is never trained, in any mode.
            if name.startswith('target_encoder'):
                continue

            # B20: register EVERY parameter, frozen ones included. A frozen
            # parameter has grad=None and AdamW skips it, so registration is a
            # no-op until the parameter is unfrozen — at which point the
            # EXISTING optimizer picks it up, and the LR scheduler stays
            # consistent because the groups never change.
            #
            # The previous code filtered on requires_grad here. The optimizer
            # is built once, at epoch 0, when gradual_unfreeze has everything
            # frozen — so the later unfreeze flipped requires_grad, gradients
            # flowed, and optimizer.step() silently never updated those
            # weights. gradual_unfreeze therefore trained the decoder (plus
            # the RevIN affine) alone for the entire run, in every run that
            # ever used it.
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
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.max_epochs * steps_per_epoch
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

        # ESJEPA — témoin de stérilité du gate d'étalement : parti de zéro
        # (zéro-init), s'il Y RESTE le décodeur ignore z — résultat négatif
        # interprétable (la cross-attention contexte suffit), pas un échec
        # silencieux. Équivalent finetune du aug/w_neq1_frac d'xres.
        head = getattr(getattr(self.model, 'decoder', None), 'decoder', None)
        z_gate = getattr(head, 'z_gate', None)
        if z_gate is not None:
            with torch.no_grad():
                absmean = torch.cat(
                    [z_gate.weight.abs().flatten(), z_gate.bias.abs().flatten()]
                ).mean()
            self.log('esjepa/gate_absmean', absmean, on_epoch=True, sync_dist=True)