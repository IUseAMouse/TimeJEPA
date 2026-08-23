"""
PyTorch Lightning Module for JEPA pretraining with TRUE forecasting objective.

The model learns to predict representations of FUTURE timesteps.
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
        2. Context → Online Encoder → context representations
        3. Target → Target Encoder (EMA) → target representations
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

        # Anti-collapse target: regularize the ENCODER output, not just the
        # predictor output (the encoder output is what downstream consumes).
        regularize_context: bool = True,

        # I-JEPA-style targets: encode [context ‖ target] and slice, instead of
        # encoding the future window in isolation.
        contextualized_targets: bool = True,

        # ABLATION ARM (G6) — regress onto the RAW future patches instead of the
        # target encoder's latents. This is the control the whole thesis needs:
        # every other pretraining result in the project compares JEPA against NO
        # pretraining (E12), never against a competing objective, so "latent
        # extrapolation beats reconstruction" sits in §4 of the log — the list of
        # things NOT established.
        #
        # It changes exactly ONE variable. The task stays past -> future, the
        # geometry, corpus, budget and optimiser are untouched; only the space
        # the predictor is scored in moves from latent to observation. That makes
        # the result interpretable, and it is also the TimesFM-style objective,
        # so reviewers recognise the baseline.
        #
        # Off by default => every existing config is bit-identical.
        reconstruction_target: bool = False,

        # G9.2 — arm JEPA inter-résolution : lit `w = k2/k1` par item dans le
        # batch et le transmet au prédicteur (FiLM). Exige des cibles
        # STANDALONE : `contextualized_targets` concatène [ctx‖cible] en pas
        # d'échantillon, ce qui n'a pas de sens physique quand les deux vivent
        # sur des grilles différentes — la garde ci-dessous refuse la
        # combinaison plutôt que de laisser tourner une éval physiquement
        # fausse. Off par défaut => configs existantes bit-identiques.
        cross_resolution: bool = False,

        # ESJEPA — arm ErrorSignal : loss auxiliaire smooth_l1(z_pred, z_target)
        # sur les statistiques du résidu de lissage (le modèle expose
        # z_predictions/z_targets quand il est construit avec
        # model.error_signal=true). lambda_z dose la voie z contre l'invariance
        # (4 dims contre 128 — témoin de siphonnage : train_loss/invariance vs
        # baseline). Off par défaut => configs existantes bit-identiques.
        error_signal: bool = False,
        lambda_z: float = 0.1,

        # Input-geometry randomization. scripts/diagnose_ettm.py shows skill
        # peaks exactly at the training context length and collapses on both
        # sides (electricity: +28.5% at ctx=384, -103.8% at ctx=768), i.e. the
        # model memorizes a fixed patch count. Sampling the geometry per batch
        # is the direct countermeasure.
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
        
        # Save hyperparameters (except model)
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        self.model.set_pretrain_mode(True)
        self.model.freeze_target_encoder()  # Target encoder never gets gradients
        
        # Loss
        self.loss_type = loss_type
        # Always store both: `validation_step` used to call jepa_loss WITHOUT
        # the weights, silently falling back to the (25, 25, 1) defaults. So
        # early-stopping and save_top_k were selecting on a different objective
        # than the one being trained.
        self.vicreg_weights = vicreg_weights
        self.sigreg_config = sigreg_config or {}
        self.regularize_context = regularize_context
        self.contextualized_targets = contextualized_targets

        self.cross_resolution = bool(cross_resolution)
        if self.cross_resolution and contextualized_targets:
            raise ValueError(
                "cross_resolution=True exige contextualized_targets=false : la "
                "cible contextualisée concatène [contexte‖cible] en pas "
                "d'échantillon, physiquement faux quand contexte et cible sont "
                "à des résolutions différentes. Poser "
                "training.contextualized_targets: false dans la config de l'arm."
            )

        self.error_signal = bool(error_signal)
        self.lambda_z = float(lambda_z)
        if self.error_signal and reconstruction_target:
            raise ValueError(
                "error_signal=True est incompatible avec "
                "reconstruction_target=True : l'arm reconstruction remplace "
                "l'objectif latent entier (_compute_loss court-circuite "
                "jepa_loss), une loss z par-dessus serait incohérente."
            )
        if self.error_signal and not getattr(model, 'error_signal', False):
            raise ValueError(
                "error_signal=True côté module mais le modèle a été construit "
                "sans model.error_signal — les clés z_predictions/z_targets "
                "n'existeraient pas. Les deux flags viennent de la même clé "
                "de config (model.error_signal), vérifier la plomberie."
            )

        # Reconstruction head: d_model -> patch_size * num_features, i.e. exactly
        # the inverse shape of the patch projection, so it is derived from the
        # model rather than re-declared (a second source of truth for the patch
        # geometry is how these two silently drift apart).
        # It is NOT part of JEPATST: the finetune loads the checkpoint by
        # component name (`online_encoder`, `predictor`, `patching`, `revin`), so
        # this head lands in the file and is ignored without a line of code.
        self.reconstruction_target = bool(reconstruction_target)
        if self.reconstruction_target:
            proj = model.patching.projection
            self.recon_head = nn.Linear(proj.out_features, proj.in_features)
            print("  ⚠️  ABLATION: reconstruction target (raw patches), NOT latent")

        self.context_lengths = list(context_lengths) if context_lengths else None
        self.p_random_context = float(p_random_context)
        self.horizon_lengths = list(horizon_lengths) if horizon_lengths else None
        self.p_random_horizon = float(p_random_horizon)

        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        
        # Scheduler params
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_scheduler_type = lr_scheduler
        self.min_lr = min_lr
        
        # Logging
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
        Sample the input geometry ONCE PER BATCH.

        Per-batch (not per-sample) keeps every tensor rectangular, so no padding
        or attention masking is needed — the encoder is length-agnostic (RoPE,
        no learned positional table) and simply sees a different patch count.

        Context is cropped from the LEFT (keep the most recent history, which is
        what a shorter context would actually contain at inference); the target
        is cropped from the right.
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
        Ablation: the same latents pushed through `recon_head` into value space,
        against the RAW future patches.

        Two details that matter and are easy to get wrong:
        * The raw patches are taken in the model's NORMALISED space. RevIN stores
          the context statistics on the module during `forward_pretrain`, and
          `forward_pretrain` normalises the target with those same statistics —
          scoring against un-normalised values would make the loss track each
          series' scale instead of its shape.
        * The patch spans are those of the standalone target window, which is
          also what sets `num_target_patches` on the JEPA path. So both arms
          predict the same number of patches covering the same timesteps, and
          `contextualized_targets` changes nothing here.
        """
        if not self.reconstruction_target:
            return outputs['predictions'], outputs['targets']

        patching = self.model.patching
        revin = self.model.revin
        x = (target - revin.mean) / revin.std if revin is not None else target

        # Mirrors Patching.forward's padding rule. Guarded by
        # test_reconstruction_patches_match_patching_geometry: if the two ever
        # drift, that test fails rather than the arm silently scoring a
        # different number of patches than the JEPA arm.
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
        Single entry point used by BOTH training_step and validation_step, so
        the two can never diverge again (see B8: validation_step used to omit
        vicreg_weights and silently score a different objective).
        """
        if self.reconstruction_target:
            # Deliberately NOT routed through jepa_loss: the anti-collapse terms
            # exist because latent targets are LEARNED and can degenerate. Raw
            # patches are fixed, so there is nothing to collapse — adding SIGReg
            # here would regularise against a non-existent failure mode.
            #
            # Huber rather than MSE, and this is not a detail. RevIN normalises
            # the target with the CONTEXT's statistics, and its scale is
            # sqrt(var + 1e-5): on a near-constant context (a flat sensor, an
            # off-hours counter, solar at night) that floors at 0.00316, so any
            # movement in the future window lands at thousands of sigma.
            # Measured on the first run: target_std spiking to 3000, target_var
            # to 4e7, against ~1 on a healthy batch.
            #
            # Under MSE one such batch outweighs tens of millions of normal ones
            # and the objective becomes whatever the degenerate windows say. The
            # JEPA arm never showed this because its targets are encoder outputs
            # — LayerNorm keeps them O(1) — so the pathology was always in the
            # data and was simply absorbed. Huber bounds each element's gradient
            # contribution, which is the standard answer for heavy-tailed
            # regression targets and keeps every window in the training set,
            # so both arms still see EXACTLY the same data.
            #
            # ⚠️ This does make the arm differ from JEPA in loss SHAPE as well as
            # target space. Stated in the log rather than hidden: an unbounded
            # value-space MSE is not a well-posed objective here, so "pure
            # like-for-like" was never actually on the table.
            loss = torch.nn.functional.smooth_l1_loss(predictions, targets)

            # Keep the pathology visible instead of letting the robust loss hide
            # it: if this climbs into the thousands, the corpus is feeding
            # degenerate windows and that is worth knowing when reading the run.
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

        # ESJEPA — loss z auxiliaire. smooth_l1 : les deux premières
        # composantes sont des log-échelles (le log rend le multiplicatif
        # additif et ~gaussien) et Huber borne la contribution des patches à
        # résidu extrême (bitbrains) — même raisonnement que l'arm recon
        # ci-dessus. Pas d'anti-collapse : la cible est FIXE (données) ; le
        # mode d'échec réel (z_pred → moyenne marginale) est surveillé par le
        # témoin esjepa/z_pred_std_ratio, pas régularisé.
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
        Training step with TRUE forecasting objective.
        
        Args:
            batch: Dictionary with 'context' (past) and 'target' (future)
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        # Get context (past) and target (future)
        context = batch['context']  # [B, context_length] or [B, context_length, C]
        target = batch['target']    # [B, prediction_length] or [B, prediction_length, C]
        
        # Add channel dimension if needed (univariate case)
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

        # Observabilité des augmentations d'entrée (demande utilisateur,
        # 2026-08-19) : plutôt que de deviner ce qui est actif, le run le dit.
        # `resolution_factor` est émis par le dataset depuis toujours mais
        # n'était consommé nulle part ; `w` n'existe que sur l'arm
        # inter-résolution. Moyennés sur l'epoch pour être lisibles dans wandb.
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
            # Témoins de l'arm xres — l'audit du 2026-08-20 (T1) a montré que
            # w_neq1_frac SEUL rassure à tort : il est dominé par les paires
            # k1=1<k2 (éligibles sur tous les morceaux 2048), alors que k1>1
            # exige des morceaux 8192 et que w<1 (k1>k2) n'existe QUE là. En
            # mode cross_resolution, `aug/multires_frac` (plus haut) = fraction
            # k1>1 ; `w_lt1_frac` est la moitié de la distribution que le FiLM
            # ne verrait jamais sans les morceaux longs.
            self.log('aug/w_neq1_frac', (w != 1).float().mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('aug/w_lt1_frac', (w < 1).float().mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('aug/w_mean', w.mean(),
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Forward pass - predict future representations
        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets,
            w=w,
        )

        predictions, targets = self._scored_pair(target, outputs)

        # Compute JEPA loss
        loss, components = self._compute_loss(predictions, targets, outputs)

        # Audit 2026-08-20 (C4) — LE témoin de convergence de l'arm xres : la
        # loss par item, conditionnée sur w. Si `wneq1` stagne pendant que `w1`
        # descend, les items inter-résolution ne convergent pas et l'arm échoue
        # de manière DIAGNOSTIQUÉE (au lieu de dégrader la moyenne en silence).
        # Coût : une MSE élément-par-élément sans réduction, négligeable.
        if w is not None and bool((w != 1).any()):
            with torch.no_grad():
                per_item = (predictions - targets).pow(2).mean(dim=(1, 2))
                mask = (w != 1)
                self.log('train_loss/wneq1', per_item[mask].mean(),
                         on_step=False, on_epoch=True, logger=True, sync_dist=True)
                if bool((~mask).any()):
                    self.log('train_loss/w1', per_item[~mask].mean(),
                             on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Logging
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

        # NOTE: validation deliberately uses the NATIVE geometry, never the
        # randomized one, so val_loss stays comparable across epochs and runs.
        # (Sur l'arm inter-résolution, le split de validation n'applique pas
        # les augmentations — w y vaut donc toujours 1 quand il existe.)
        w = batch.get('w') if self.cross_resolution else None
        outputs = self.model.forward_pretrain(
            context, target, contextualized_targets=self.contextualized_targets,
            w=(w.float() if w is not None else None),
        )

        predictions, targets = self._scored_pair(target, outputs)

        # Same objective as training — see _compute_loss
        loss, components = self._compute_loss(predictions, targets, outputs)

        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for key, value in components.items():
            if key == 'loss':
                continue
            self.log(f'val_loss/{key}', value, on_step=False, on_epoch=True,
                     logger=True, sync_dist=True)

        # Compute metrics
        metrics = compute_pretrain_metrics(
            predictions,
            targets,
            context_embeddings=outputs.get('context_embeddings')
        )
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Collapse is the failure mode this whole phase exists to prevent, so
        # surface it as a first-class number instead of burying it in metrics.
        ctx = outputs.get('context_embeddings')
        if ctx is not None:
            collapse = ctx.std(dim=0).mean()
            self.log('collapse/context_std', collapse, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            # Effective rank: a collapsed encoder concentrates all its energy in
            # a handful of directions. Computed on the first validation batch
            # only — it is a diagnostic, not a per-batch quantity, and running an
            # eigendecomposition on every batch is pure overhead.
            if batch_idx == 0:
                eff_rank = self._effective_rank(ctx)
                if eff_rank is not None:
                    self.log('collapse/effective_rank', eff_rank, on_step=False,
                             on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # ESJEPA — les deux témoins de l'arm (équivalents de aug/w_neq1_frac
        # pour xres : s'ils sont mauvais, le run ne mesure rien) :
        # * esjepa/z_corr : Spearman entre la log-RMS PRÉDITE (z_pred[...,0])
        #   et la log-RMS RÉALISÉE du résidu — LE témoin de non-stérilité.
        #   Prédiction P1 posée avant le run : > 0.3 ; ≈ 0 = l'hétéro-
        #   scédasticité n'est pas prévisible depuis le contexte, l'arm meurt
        #   AU PRETRAIN.
        # * esjepa/z_pred_std_ratio : std(z_pred)/std(z_target) — le témoin du
        #   collapse-vers-la-moyenne-marginale (l'analogue du pred_var 0.6 /
        #   target_var 0.95 qui a motivé tout l'arm).
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
        """Corrélation de Spearman (Pearson des rangs), sans scipy — les
        ex-aequo reçoivent des rangs d'ordre d'apparition, suffisant pour un
        témoin de monitoring."""
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

        Two deliberate choices:

        - Eigenvalues of the DxD covariance rather than SVD of the [B*N, D]
          matrix. Mathematically the same spectrum (up to squaring) but on a
          128x128 matrix instead of a 32000x128 one.

        - Wrapped in try/except. Iterative eigensolvers can fail to converge on
          near-degenerate input — which is *exactly* the collapsed case this
          metric exists to detect. A monitoring metric that crashes the run at
          the precise moment the thing it monitors happens would be worse than
          useless.
        """
        try:
            # autocast must be disabled explicitly. Casting to float32 is not
            # enough: the matmul below sits inside the bf16-mixed autocast
            # region, so PyTorch casts it straight back to bfloat16 and
            # eigvalsh then fails with
            #     "linalg_eigh_cuda" not implemented for 'BFloat16'
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
        except Exception as e:  # noqa: BLE001 - diagnostics must never kill a run
            logger.warning(f"effective_rank unavailable this step: {e}")
            return None
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler_type == 'constant':
            return optimizer
        
        # Calculate number of training steps
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.max_epochs * steps_per_epoch
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
        print(f"✅ Saved pretrained encoder to {save_path}")