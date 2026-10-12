"""
JEPA-TST: Joint-Embedding Predictive Architecture for Time Series Forecasting.

This model learns to predict future representations from past context:
- Pretrain: Context -> Encoder -> Predictor -> Predicted future repr
            Target -> Target Encoder (EMA) -> Target repr
            Loss: MSE(Predicted, Target)
- Finetune: Context -> Encoder -> Predictor -> Decoder -> Forecast values
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Literal

from .components.revin import RevIN
from .components.patching import Patching
from .encoders.bare_encoder import BareTransformerEncoder
from .encoders.target_encoder import TargetEncoder
from .predictors.transformer_predictor import TransformerPredictor, MLPPredictor
from .decoders.linear_decoder import ForecastingHead
from .components.robust_scale import RobustScale

import logging

logger = logging.getLogger(__name__)


def filter_loadable(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> tuple:
    """
    Drop checkpoint entries whose shape does not match the target model.

    `load_state_dict(strict=False)` tolerates missing and unexpected keys but
    still raises on shape mismatches. A point decoder and the quantile head
    both own `decoder.decoder.unpatching.projection` with different sizes, so
    reusing a pretrained encoder under a new head needs this filter.

    Returns:
        (filtered_state_dict, list of (key, checkpoint_shape, model_shape))
    """
    target = model.state_dict()
    filtered, dropped = {}, []
    for key, value in state_dict.items():
        if key in target and hasattr(value, "shape") and value.shape != target[key].shape:
            dropped.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        filtered[key] = value
    return filtered, dropped


def grow_future_query_table(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    key: str = "predictor.future_position_embedding",
) -> Dict[str, torch.Tensor]:
    """
    Merge the future-query table of a shorter checkpoint into a longer model.

    `predictor.future_position_embedding` ([1, max_target_patches, d_model])
    is the only parameter whose shape depends on `prediction_length`. Without
    this merge `filter_loadable` would drop the key and the predictor would
    restart from a random table. The checkpoint rows are copied into a copy of
    the model's table, whose init provides the new rows. Raises ValueError if
    the checkpoint table is longer or d_model differs; no-op when shapes match.

    Opt-in only: called by FinetuneModule when `extend_horizon_queries=true`.
    """
    if key not in state_dict:
        return state_dict
    model_table = dict(model.state_dict()).get(key)
    if model_table is None:
        return state_dict
    ckpt_table = state_dict[key]
    if tuple(ckpt_table.shape) == tuple(model_table.shape):
        return state_dict
    if ckpt_table.shape[-1] != model_table.shape[-1]:
        raise ValueError(
            f"{key}: d_model {ckpt_table.shape[-1]} (checkpoint) != "
            f"{model_table.shape[-1]} (model) - cannot merge."
        )
    if ckpt_table.shape[1] > model_table.shape[1]:
        raise ValueError(
            f"{key}: checkpoint table ({ckpt_table.shape[1]} rows) is "
            f"longer than the model's ({model_table.shape[1]}); truncating "
            f"would drop learned queries. Shrinking the horizon needs no "
            f"merge: the predictor slices its own table."
        )
    merged = model_table.detach().clone()
    n = ckpt_table.shape[1]
    merged[:, :n, :] = ckpt_table
    out = dict(state_dict)
    out[key] = merged
    logger.info(
        f"  {key}: table extended {tuple(ckpt_table.shape)} -> "
        f"{tuple(model_table.shape)} ({n} pretrained rows copied, "
        f"{model_table.shape[1] - n} new rows at model init)"
    )
    return out


class JEPATST(nn.Module):
    """
    JEPA-TST model for time series forecasting.

    Architecture:
        Pretrain (True Forecasting JEPA):
            Context [B, L_ctx, C] -> RevIN -> Patching -> Online Encoder -> context_repr
            Target [B, L_tgt, C] -> RevIN (same stats) -> Patching -> Target Encoder (EMA) -> target_repr
            Predictor(context_repr) -> pred_repr
            Loss: MSE(pred_repr, target_repr.detach())
        
        Finetune:
            Context [B, L_ctx, C] -> RevIN -> Patching -> Encoder -> Predictor -> Decoder -> Forecast
            Loss: MSE(Forecast, Ground Truth)
    """

    def __init__(
        self,
        # Data params
        input_length: int = 384,
        prediction_length: int = 96,
        num_features: int = 1,
        
        # Patching
        patch_size: int = 16,
        stride: int = 8,
        
        # Encoder
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        
        # Predictor
        predictor_type: Literal['transformer', 'mlp'] = 'transformer',
        predictor_num_layers: int = 2,
        predictor_num_heads: int = 4,
        predictor_d_ff: int = 512,
        
        # Decoder
        decoder_type: Literal['linear', 'mlp', 'attentive'] = 'attentive',
        
        # EMA
        ema_tau_base: float = 0.996,
        ema_tau_end: float = 1.0,
        
        # RevIN
        use_revin: bool = True,

        # Cross-resolution: builds the predictor with the scale-conditioning
        # FiLM (w = k2/k1). Opt-in: at False no extra parameter exists.
        cross_resolution: bool = False,

        # Robust arcsinh scaling composed around RevIN (see
        # components/robust_scale.py). Opt-in: at False no attribute, no
        # compute path.
        robust_scale: bool = False,

        # Error signal: a second latent path predicts the statistics of the
        # smoothing residual (conditional heteroscedasticity). z_target is
        # computed from the data (causal EWMA), z_pred comes from a head on
        # the predictor trunk, and at finetune z modulates the quantile fan's
        # spread through a zero-init gate. Opt-in: at False no attribute, no
        # compute path.
        error_signal: bool = False,

        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()

        if error_signal and predictor_type != 'transformer':
            raise ValueError(
                "error_signal=True requires predictor_type='transformer' - "
                "the z head lives on the predictor trunk (MLPPredictor has "
                "neither queries nor target tokens to attach it to)."
            )
        if error_signal and num_features != 1:
            raise ValueError(
                "error_signal=True is only implemented univariate "
                "(num_features=1) - per-patch residual stats are computed "
                "with channels folded."
            )

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        self.robust_scaler = RobustScale() if robust_scale else None
        self.error_signal = error_signal
        
        # Calculate number of patches for context and target
        self.num_patches = (input_length - patch_size) // stride + 1
        self.num_target_patches = (prediction_length - patch_size) // stride + 1
        
        # ===== RevIN Normalization =====
        if use_revin:
            self.revin = RevIN(
                num_features=num_features,
                affine=affine,
                subtract_last=subtract_last
            )
        else:
            self.revin = None
        
        # ===== Patching Layer (shared for context and target) =====
        self.patching = Patching(
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            num_features=num_features
        )
        
        # ===== Online Encoder =====
        self.online_encoder = BareTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # ===== Target Encoder (EMA of online encoder) =====
        self.target_encoder = TargetEncoder(
            encoder=self.online_encoder,
            ema_decay=ema_tau_base,
            ema_decay_end=ema_tau_end
        )
        
        # ===== Predictor =====
        if predictor_type == 'transformer':
            # Size the future-query table from the prediction length, with
            # headroom (see TransformerPredictor._future_queries).
            max_target_patches = max(16, int(self.num_target_patches * 1.5) + 4)
            self.predictor = TransformerPredictor(
                d_model=d_model,
                num_layers=predictor_num_layers,
                num_heads=predictor_num_heads,
                d_ff=predictor_d_ff,
                dropout=dropout,
                activation=activation,
                max_target_patches=max_target_patches,
                use_w_film=cross_resolution,
                error_signal=error_signal,
            )
        elif predictor_type == 'mlp':
            self.predictor = MLPPredictor(
                d_model=d_model,
                num_layers=predictor_num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")
        
        # ===== Decoder (for finetuning) =====
        # `stride` must be forwarded: with the ForecastingHead default of 8,
        # UnPatching reassembles the forecast on the wrong grid and silently
        # truncates it.
        self.decoder = ForecastingHead(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            prediction_length=prediction_length,
            num_features=num_features,
            decoder_type=decoder_type,
            revin=self.revin,
            error_signal=error_signal and decoder_type == 'quantile',
        )
        
        # Model state
        self._pretrain_mode = True
        
        print(f"JEPATST initialized:")
        print(f"  Context: {input_length} tp -> {self.num_patches} patches")
        print(f"  Target: {prediction_length} tp -> {self.num_target_patches} patches")
        
    def set_pretrain_mode(self, mode: bool = True):
        """Set whether model is in pretrain or finetune mode."""
        self._pretrain_mode = mode
        
    def is_pretrain_mode(self) -> bool:
        """Check if model is in pretrain mode."""
        return self._pretrain_mode

    def forward_pretrain(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        contextualized_targets: bool = True,
        w: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA pretraining: predict the representations of
        future timesteps from the context.

        Args:
            context: Past time series [B, context_length, C] (e.g., [B, 384, 1])
            target: Future time series [B, prediction_length, C] (e.g., [B, 96, 1])
        
        Returns:
            Dictionary with:
                - 'predictions': Predicted future representations [B, num_target_patches, d_model]
                - 'targets': Target encoder representations [B, num_target_patches, d_model]
                - 'context_embeddings': Context embeddings [B, num_context_patches, d_model]
        """
        # 0. Robust compression with context statistics, before RevIN.
        if self.robust_scaler is not None:
            self.robust_scaler.fit(context)
            context = self.robust_scaler.transform(context)
            target = self.robust_scaler.transform(target)

        # 1. RevIN normalization; the target uses the context statistics.
        if self.revin is not None:
            context_norm = self.revin(context, mode='norm')
            target_norm = (target - self.revin.mean) / self.revin.std
        else:
            context_norm = context
            target_norm = target
        
        # 2. Patch context and target
        context_patches = self.patching(context_norm)  # [B, num_patches, d_model]
        target_patches = self.patching(target_norm)    # [B, num_target_patches, d_model]

        num_target_patches = target_patches.shape[1]

        # 3. Encode context with online encoder (gets gradients)
        context_embeddings = self.online_encoder(context_patches)
        # [B, num_patches, d_model]

        # 4. Encode target with target encoder (NO gradients - EMA updated)
        with torch.no_grad():
            if contextualized_targets:
                # I-JEPA-style targets: encode the full window and slice out
                # the target positions. Encoding the future window alone gives
                # the target encoder far fewer patches than the online encoder
                # and yields nearly context-free targets. The last
                # num_target_patches patches of the concatenated window span
                # the same positions as the standalone target patches.
                full_norm = torch.cat([context_norm, target_norm], dim=1)
                full_patches = self.patching(full_norm)
                full_embeddings = self.target_encoder(full_patches)
                target_embeddings = full_embeddings[:, -num_target_patches:, :]
            else:
                target_embeddings = self.target_encoder(target_patches)
            # [B, num_target_patches, d_model]

        # 5. Predict target representations from context embeddings
        if self.error_signal:
            # The same trunk pass also produces z_pred [B, N, 4].
            predictions, z_predictions = self.predictor.forward_simple(
                context_embeddings=context_embeddings,
                num_targets=num_target_patches,
                w=w,
                return_z=True,
            )
        else:
            predictions = self.predictor.forward_simple(
                context_embeddings=context_embeddings,
                num_targets=num_target_patches,
                # Per-item scale ratio; None outside cross-resolution.
                w=w,
            )
        # [B, num_target_patches, d_model]

        out = {
            'predictions': predictions,
            'targets': target_embeddings.detach(),
            'context_embeddings': context_embeddings,
        }
        if self.error_signal:
            # Keys absent with the flag off.
            out['z_predictions'] = z_predictions
            out['z_targets'] = self._residual_stats(
                context_norm, target_norm, num_target_patches)
        return out

    @torch.no_grad()
    def _residual_stats(
        self,
        context_norm: torch.Tensor,
        target_norm: torch.Tensor,
        num_target_patches: int,
    ) -> torch.Tensor:
        """
        z_target: deterministic per-patch statistics of the target window's
        smoothing residual, [B, num_target_patches, 4].

        Computed from the normalized data, independent of any encoder. The
        smoothing is a causal EWMA (halflife = patch_size/2) as a convolution
        with a truncated exponential kernel initialized on the context tail,
        so the statistic of patch t only depends on values up to its end.

        The 4 components per residual patch r:
          0. log(RMS(r) + 1e-3): noise scale
          1. log(MAD(r)*1.4826 + 1e-3): robust scale
          2. tanh(mean(r^3)/RMS^3): bounded asymmetry
          3. lag-1 autocorrelation: white vs structured noise
        The 1e-3 floor handles flat patches.
        """
        B, target_len, C = target_norm.shape
        # Exponential kernel truncated at K steps (over 99% of the mass at
        # halflife=8 for K=64), never longer than the available context.
        halflife = self.patch_size / 2.0
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        K = min(context_norm.shape[1] + 1, 4 * self.patch_size)
        lags = torch.arange(K, device=target_norm.device,
                            dtype=target_norm.dtype)
        kern = alpha * (1.0 - alpha) ** lags
        kern = (kern / kern.sum()).flip(0).reshape(1, 1, K)

        # Window = context tail (K-1 steps) + target; "valid" conv yields
        # exactly target_len causal smoothings.
        tail = torch.cat([context_norm[:, -(K - 1):], target_norm], dim=1)
        x = tail.permute(0, 2, 1).reshape(B * C, 1, -1)
        smooth = F.conv1d(x, kern).reshape(B, C, target_len).permute(0, 2, 1)
        resid = (target_norm - smooth).squeeze(-1)          # [B, target_len]

        # Per-patch split on the Patching grid. If the geometry leaves a
        # padding patch, the last statistic is replicated, as Patching does.
        patches = resid.unfold(1, self.patch_size, self.stride)  # [B, N', P]
        rms = patches.pow(2).mean(-1).sqrt()
        mad = (patches - patches.median(-1, keepdim=True).values).abs() \
            .median(-1).values * 1.4826
        asym = torch.tanh(patches.pow(3).mean(-1) / rms.pow(3).clamp_min(1e-9))
        ac1 = (patches[..., :-1] * patches[..., 1:]).mean(-1) / \
            patches.pow(2).mean(-1).clamp_min(1e-9)
        z = torch.stack([
            (rms + 1e-3).log(),
            (mad + 1e-3).log(),
            asym,
            ac1.clamp(-1.0, 1.0),
        ], dim=-1)                                          # [B, N', 4]

        if z.shape[1] < num_target_patches:
            pad = z[:, -1:, :].expand(-1, num_target_patches - z.shape[1], -1)
            z = torch.cat([z, pad], dim=1)
        return z[:, :num_target_patches, :].detach()

    def forward_finetune(
        self,
        context: torch.Tensor,
        return_representations: bool = False,
        skip_revin: bool = False,
        w: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised finetuning: encode, predict future
        representations, decode them to forecast values.

        Args:
            context: Past time series [B, context_length, C]
            return_representations: If True, also return intermediate representations

        Returns:
            Dictionary with 'forecast' and 'forecast_denorm'
        """
        # 0. Robust compression. As for RevIN, skip_revin=True means the
        # context is already in the transformed frame.
        if self.robust_scaler is not None and not skip_revin:
            self.robust_scaler.fit(context)
            context = self.robust_scaler.transform(context)

        # 1. RevIN normalization
        if skip_revin:
            context_norm = context
        elif self.revin is not None:
            context_norm = self.revin(context, mode='norm')
        else:
            context_norm = context
        
        # 2. Patch context
        context_patches = self.patching(context_norm)  # [B, num_patches, d_model]
        
        # 3. Encode context with online encoder
        context_embeddings = self.online_encoder(context_patches)
        # [B, num_patches, d_model]
        
        # 4. Predict future representations. `w` lets cross-resolution
        # finetuning pass the batch's scale ratio and RateIN pass w=1/k. With
        # w=None, the FiLM (when present) is applied at w=1: its bias is
        # trained, so skipping it would not be the identity.
        if w is not None:
            w_eff = w.to(context_embeddings.device).reshape(-1)
        elif hasattr(self.predictor, 'w_film'):
            w_eff = torch.ones(context_embeddings.shape[0],
                               device=context_embeddings.device)
        else:
            w_eff = None

        z_pred = None
        if self.error_signal:
            # z_pred comes from the same pass and modulates the quantile
            # fan's spread; recomputed at every roll of the rollout.
            predictions, z_pred = self.predictor.forward_simple(
                context_embeddings=context_embeddings,
                num_targets=self.num_target_patches,
                w=w_eff,
                return_z=True,
            )
        else:
            predictions = self.predictor.forward_simple(
                context_embeddings=context_embeddings,
                num_targets=self.num_target_patches,
                w=w_eff,
            )
        # [B, num_target_patches, d_model]

        # 5. Decode to forecast values. With skip_revin, norm == denorm.
        #
        # `context_embeddings` is forwarded unconditionally: point decoders
        # ignore it, the quantile head cross-attends to it. The predictor
        # output is a conditional mean under MSE, so a head that only sees it
        # cannot separate two contexts with the same expected future but
        # different volatility.
        forecast, forecast_denorm = self.decoder(
            predictions, skip_revin=skip_revin,
            context_embeddings=context_embeddings,
            # None without error_signal; the quantile head refuses a missing z
            # and a point decoder refuses a provided one.
            z=z_pred,
        )
        # Point decoders: [B, prediction_length, C]
        # Quantile head:  [B, prediction_length, Q], sorted along Q

        # Back to raw space for denormalized outputs. sinh is monotonic, so
        # the quantile order is preserved and the median commutes with it.
        if self.robust_scaler is not None and not skip_revin:
            forecast_denorm = self.robust_scaler.inverse(forecast_denorm)

        result = {
            'forecast': forecast,
            'forecast_denorm': forecast_denorm
        }

        # A probabilistic head returns the full fan; expose the median as the
        # point forecast so downstream callers keep working unchanged. The
        # median is also the MAE-optimal point estimate, which MASE scores.
        if getattr(self.decoder, 'is_probabilistic', False):
            head = self.decoder.decoder
            result['quantiles'] = forecast
            result['quantiles_denorm'] = forecast_denorm
            result['quantile_levels'] = head.quantile_levels
            result['forecast'] = head.median(forecast)
            result['forecast_denorm'] = head.median(forecast_denorm)

        if z_pred is not None:
            # Per-patch confidence read; key absent without error_signal.
            result['z'] = z_pred

        if return_representations:
            result['context_embeddings'] = context_embeddings
            result['future_representations'] = predictions
            # The normalized context, so finetune terms can encode candidates
            # in the head's frame without refitting a scaler.
            result['context_norm'] = context_norm

        return result

    def forecast(
        self,
        context: torch.Tensor,
        n: Optional[int] = None,
        return_representations: bool = False,
        skip_revin: bool = False,
        sample_paths: bool = True,
        w: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forecast n steps ahead, rolling when n exceeds prediction_length.

        With `skip_revin=False` the whole rollout happens in one
        instance-normalized frame: RevIN statistics are computed once on the
        real context, pinned via `revin.freeze()`, and the result is
        denormalized once at the end. `skip_revin=True` means the caller
        guarantees the context is already in that frame; globally z-scored
        benchmark data does not qualify.

        Args:
            context: Past time series [B, context_length, C]
            n: Number of steps to forecast. Defaults to self.prediction_length.
               Above it, ceil(n / prediction_length) rolling passes.
            return_representations: If True, return intermediate representations
                (only when n <= prediction_length)
            skip_revin: Only if the context is already instance-normalized.

        Returns:
            Dictionary with:
                - 'forecast': Forecast in the normalized frame [B, n, C]
                - 'forecast_denorm': Forecast in the input's original scale [B, n, C]
                - 'context_embeddings', 'future_representations' (if return_representations
                  and n <= prediction_length)
        """
        if n is None:
            n = self.prediction_length

        # ---- Case 1: single-shot forecast ----
        if n <= self.prediction_length:
            result = self.forward_finetune(
                context,
                return_representations=return_representations,
                skip_revin=skip_revin,
                w=w,
            )
            # Every horizon-shaped tensor is truncated, the quantile fan
            # included, so they stay aligned downstream.
            for key in ('forecast', 'forecast_denorm', 'quantiles', 'quantiles_denorm'):
                if key in result:
                    result[key] = result[key][:, :n]
            return result

        # ---- Case 2: rolling forecast ----
        # On a robust_scale model, skip_revin=True would also bypass the
        # arcsinh compression and emit wrong numbers. Refuse.
        if self.robust_scaler is not None and skip_revin:
            raise ValueError(
                "skip_revin=True is incompatible with a robust_scale model: "
                "the arcsinh compression would be bypassed and the numbers "
                "would be silently wrong (eval_skip_revin is a pre-P0 legacy "
                "mode, never valid on an arcsinh checkpoint)."
            )

        # Robust compression at the rollout's single entry point: statistics
        # from the true context, the whole rollout lives in compressed space,
        # the inverse only applies to *_denorm outputs.
        if self.robust_scaler is not None and not skip_revin:
            self.robust_scaler.fit(context)
            context = self.robust_scaler.transform(context)

        use_revin = (not skip_revin) and (self.revin is not None)

        if use_revin:
            # Instance statistics computed once on the true context, then
            # pinned so every roll stays in the same frame.
            current_context = self.revin(context, mode='norm')
            self.revin.freeze()
        else:
            current_context = context.clone()

        try:
            num_rolls = (n + self.prediction_length - 1) // self.prediction_length

            # With a probabilistic head, propagate the fan through the rolls
            # instead of the median: median feedback makes later rolls see a
            # context smoother than real data, so intervals shrink with the
            # horizon while the true uncertainty widens.
            if sample_paths and getattr(self.decoder, 'is_probabilistic', False):
                forecast_norm, quantiles_norm, quantile_levels = \
                    self._rolling_forecast_sampled(current_context, n, num_rolls, use_revin, w=w)
                if use_revin:
                    forecast_denorm = self.revin.denormalize_target_space(forecast_norm)
                    quantiles_denorm = self.revin.denormalize_target_space(quantiles_norm)
                else:
                    forecast_denorm = forecast_norm
                    quantiles_denorm = quantiles_norm
                if self.robust_scaler is not None and not skip_revin:
                    forecast_denorm = self.robust_scaler.inverse(forecast_denorm)
                    quantiles_denorm = self.robust_scaler.inverse(quantiles_denorm)
                return {
                    "forecast": forecast_norm,
                    "forecast_denorm": forecast_denorm,
                    "quantiles": quantiles_norm,
                    "quantiles_denorm": quantiles_denorm,
                    "quantile_levels": quantile_levels,
                }

            all_forecasts_norm = []
            all_quantiles_norm = []
            quantile_levels = None

            for roll_idx in range(num_rolls):
                # `current_context` is already in the normalized frame, so the
                # inner call must not re-normalize it.
                result = self.forward_finetune(
                    current_context,
                    return_representations=False,
                    skip_revin=True,
                    w=w,
                )

                forecast_norm = result['forecast']  # [B, pred_len, C] (median if probabilistic)
                all_forecasts_norm.append(forecast_norm)

                # Keep each roll's quantile fan. Every roll is conditioned on
                # the median path fed back to it, so the fan reflects that
                # roll alone and interval widths at long horizons are too
                # narrow.
                if 'quantiles' in result:
                    all_quantiles_norm.append(result['quantiles'])
                    quantile_levels = result['quantile_levels']

                if roll_idx < num_rolls - 1:
                    # `current_context` is in the encoder input frame (affine
                    # applied); `forecast_norm` is in the target frame. Realign
                    # before feeding the prediction back in.
                    feedback = (
                        self.revin.to_input_frame(forecast_norm)
                        if use_revin else forecast_norm
                    )
                    current_context = torch.cat(
                        [
                            current_context[:, self.prediction_length:, :],
                            feedback
                        ],
                        dim=1
                    )

            # ---- Concatenate & truncate ----
            forecast_norm = torch.cat(all_forecasts_norm, dim=1)[:, :n]
            quantiles_norm = (
                torch.cat(all_quantiles_norm, dim=1)[:, :n]
                if all_quantiles_norm else None
            )

            # ---- Denormalize ONCE, with the pinned statistics ----
            if use_revin:
                forecast_denorm = self.revin.denormalize_target_space(forecast_norm)
                quantiles_denorm = (
                    self.revin.denormalize_target_space(quantiles_norm)
                    if quantiles_norm is not None else None
                )
            else:
                forecast_denorm = forecast_norm
                quantiles_denorm = quantiles_norm

            if self.robust_scaler is not None and not skip_revin:
                forecast_denorm = self.robust_scaler.inverse(forecast_denorm)
                if quantiles_denorm is not None:
                    quantiles_denorm = self.robust_scaler.inverse(quantiles_denorm)
        finally:
            if use_revin:
                self.revin.unfreeze()

        out = {
            "forecast": forecast_norm,
            "forecast_denorm": forecast_denorm,
        }
        if quantiles_norm is not None:
            out["quantiles"] = quantiles_norm
            out["quantiles_denorm"] = quantiles_denorm
            out["quantile_levels"] = quantile_levels
        return out


    def _rolling_forecast_sampled(
        self,
        current_context: torch.Tensor,
        n: int,
        num_rolls: int,
        use_revin: bool,
        w: 'Optional[torch.Tensor]' = None,
    ):
        """
        Quantile-path rollout: propagate the whole fan rather than the median.

        Roll 1 is one exact forward. Later rolls expand the batch Q-fold, one
        copy per quantile level, and copy k is fed the level-k trajectory of
        the previous roll (comonotonic coupling). The marginal fan at each
        timestep is the sort of the Q paths, which also restores monotonicity.
        This assumes perfect rank dependence across rolls, an upper bound on
        the spread, and is deterministic. Rolls after the first run at batch
        B*Q.

        Returns (forecast_norm [B,n,C], quantiles_norm [B,n,Q], levels).
        """
        head = self.decoder.decoder
        mid = head.median_idx
        pred_len = self.prediction_length
        batch = current_context.shape[0]

        # ---- Roll 1: exact, on the true context ----
        first = self.forward_finetune(current_context, skip_revin=True, w=w)
        fan = first['quantiles']                              # [B, H, Q]
        levels = first['quantile_levels']
        n_q = fan.shape[-1]

        fans = [fan]

        if num_rolls > 1:
            # One batch copy per quantile level: rows are ordered
            # (b0,k0)..(b0,kQ-1),(b1,k0)... so the level of row i is i % Q.
            ctx = current_context.repeat_interleave(n_q, dim=0)     # [B*Q, L, C]
            paths = fan.permute(0, 2, 1).reshape(batch * n_q, pred_len, 1)
            level_idx = torch.arange(n_q, device=fan.device).repeat(batch)

            for _ in range(1, num_rolls):
                feedback = self.revin.to_input_frame(paths) if use_revin else paths
                ctx = torch.cat([ctx[:, pred_len:, :], feedback], dim=1)

                rolled = self.forward_finetune(
                    ctx, skip_revin=True,
                    w=(w.repeat_interleave(n_q) if w is not None else None))
                fan_k = rolled['quantiles']                    # [B*Q, H, Q]

                # Comonotonic continuation: copy k follows its own level k
                paths = fan_k.gather(
                    -1, level_idx.view(-1, 1, 1).expand(-1, pred_len, 1)
                )                                              # [B*Q, H, 1]

                # Marginal fan = per-timestep order statistics of the Q paths
                roll_fan = paths.view(batch, n_q, pred_len).permute(0, 2, 1)
                roll_fan, _ = roll_fan.sort(dim=-1)            # [B, H, Q]
                fans.append(roll_fan)

        quantiles_norm = torch.cat(fans, dim=1)[:, :n]
        forecast_norm = quantiles_norm[..., mid:mid + 1]
        return forecast_norm, quantiles_norm, levels

    def forward(
        self,
        context: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass (dispatches to pretrain or finetune).
        
        Args:
            context: Past time series [B, context_length, C]
            target: Future time series [B, prediction_length, C] (required for pretrain)
        
        Returns:
            Output dictionary (depends on mode)
        """
        if self._pretrain_mode:
            if target is None:
                raise ValueError("target is required in pretrain mode")
            return self.forward_pretrain(context, target)
        else:
            return self.forward_finetune(context, **kwargs)

    # ===== Freezing/Unfreezing methods =====
    
    def freeze_encoder(self):
        """Freeze encoder parameters (for finetuning)."""
        for param in self.online_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.online_encoder.parameters():
            param.requires_grad = True

    def freeze_target_encoder(self):
        """Freeze target encoder (should always be frozen)."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def freeze_predictor(self):
        """Freeze predictor for finetuning."""
        for param in self.predictor.parameters():
            param.requires_grad = False

    def unfreeze_predictor(self):
        """Unfreeze the predictor."""
        for param in self.predictor.parameters():
            param.requires_grad = True

    def freeze_patching(self):
        """Freeze patching layers."""
        for param in self.patching.parameters():
            param.requires_grad = False
    
    def unfreeze_patching(self):
        """Unfreeze patching layer."""
        for param in self.patching.parameters():
            param.requires_grad = True

    def freeze_revin(self):
        """Freeze revin layers."""
        if self.revin is not None:
            for param in self.revin.parameters():
                param.requires_grad = False

    def unfreeze_revin(self):
        """Unfreeze revin layers."""
        if self.revin is not None:
            for param in self.revin.parameters():
                param.requires_grad = True

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'online_encoder': sum(p.numel() for p in self.online_encoder.parameters()),
            'target_encoder': sum(p.numel() for p in self.target_encoder.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'patching': sum(p.numel() for p in self.patching.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    @torch.no_grad()
    def update_target_encoder(self, step: int, max_steps: int):
        """Update target encoder with EMA."""
        self.target_encoder.update(self.online_encoder, step, max_steps)

    def save_pretrained_encoder(self, save_path: str):
        """Save encoder and predictor weights for finetuning."""
        state = {
            'online_encoder': self.online_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'patching': self.patching.state_dict(),
        }
        if self.revin is not None:
            state['revin'] = {
                k: v for k, v in self.revin.state_dict().items()
                if not k.endswith('.mean') and not k.endswith('.std')
            }
        torch.save(state, save_path)
        print(f"Saved pretrained encoder + predictor to {save_path}")