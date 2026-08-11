# src/timejepa/models/jepa_tst.py
"""
JEPA-TST: Joint-Embedding Predictive Architecture for Time Series Forecasting.

This model learns to predict future representations from past context:
- Pretrain: Context → Encoder → Predictor → Predicted future repr
            Target → Target Encoder (EMA) → Target repr
            Loss: MSE(Predicted, Target)
- Finetune: Context → Encoder → Predictor → Decoder → Forecast values
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal

from .components.revin import RevIN
from .components.patching import Patching
from .encoders.bare_encoder import BareTransformerEncoder
from .encoders.target_encoder import TargetEncoder
from .predictors.transformer_predictor import TransformerPredictor, MLPPredictor
from .decoders.linear_decoder import ForecastingHead


def filter_loadable(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> tuple:
    """
    Drop checkpoint entries whose shape does not match the target model.

    `load_state_dict(strict=False)` tolerates missing and unexpected keys but
    NOT shape mismatches — those still raise. That bites whenever a component is
    swapped for one that occupies the same key path with a different shape.

    The concrete case: a point decoder and the quantile head both own
    `decoder.decoder.unpatching.projection`, sized `patch_size * 1` and
    `patch_size * n_quantiles` respectively. Loading a pre-quantile checkpoint
    into a quantile model therefore aborted, which is precisely the intended
    workflow — reuse a pretrained encoder, re-learn the head.

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


class JEPATST(nn.Module):
    """
    JEPA-TST model for time series forecasting.

    Architecture:
        Pretrain (True Forecasting JEPA):
            Context [B, L_ctx, C] → RevIN → Patching → Online Encoder → context_repr
            Target [B, L_tgt, C] → RevIN (same stats) → Patching → Target Encoder (EMA) → target_repr
            Predictor(context_repr) → pred_repr
            Loss: MSE(pred_repr, target_repr.detach())
        
        Finetune:
            Context [B, L_ctx, C] → RevIN → Patching → Encoder → Predictor → Decoder → Forecast
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
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        
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
            # Size the future-query table from the actual prediction length,
            # with headroom. The old default of 16 silently truncated any
            # configuration needing more target patches, substituting context
            # embeddings for predictions (see TransformerPredictor._future_queries).
            max_target_patches = max(16, int(self.num_target_patches * 1.5) + 4)
            self.predictor = TransformerPredictor(
                d_model=d_model,
                num_layers=predictor_num_layers,
                num_heads=predictor_num_heads,
                d_ff=predictor_d_ff,
                dropout=dropout,
                activation=activation,
                max_target_patches=max_target_patches,
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
        # `stride` MUST be forwarded. Without it ForecastingHead fell back to
        # its default of 8, so UnPatching reassembled the forecast on the wrong
        # grid: with patch_size=32 the decoder emitted 80 timesteps instead of
        # 128, silently truncated, with no error anywhere.
        #
        # This stayed invisible because train.py and evaluate.py both replace
        # model.decoder with a correctly-strided one right after construction —
        # but anything using JEPATST directly (notably the packaged
        # `model.forecast(...)` API) got the broken head.
        self.decoder = ForecastingHead(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            prediction_length=prediction_length,
            num_features=num_features,
            decoder_type=decoder_type,
            revin=self.revin
        )
        
        # Model state
        self._pretrain_mode = True
        
        print(f"JEPATST initialized:")
        print(f"  Context: {input_length} tp → {self.num_patches} patches")
        print(f"  Target: {prediction_length} tp → {self.num_target_patches} patches")
        
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA pretraining with TRUE forecasting objective.
        
        The model learns to predict representations of FUTURE timesteps,
        not masked patches within the same context window.
        
        Args:
            context: Past time series [B, context_length, C] (e.g., [B, 384, 1])
            target: Future time series [B, prediction_length, C] (e.g., [B, 96, 1])
        
        Returns:
            Dictionary with:
                - 'predictions': Predicted future representations [B, num_target_patches, d_model]
                - 'targets': Target encoder representations [B, num_target_patches, d_model]
                - 'context_embeddings': Context embeddings [B, num_context_patches, d_model]
        """
        # 1. RevIN normalization
        # IMPORTANT: Normalize target with SAME statistics as context
        if self.revin is not None:
            context_norm = self.revin(context, mode='norm')
            # Apply same normalization to target (using stored mean/std from context)
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
                # I-JEPA-style targets: encode the FULL window and slice out the
                # target positions, rather than encoding the future window in
                # isolation.
                #
                # Encoding it alone means the target encoder sees ~11 patches
                # while the online encoder sees ~47 — a distribution shift
                # between two networks that are supposed to be an EMA pair. It
                # also makes targets nearly context-free (a 96-step window in
                # isolation is little more than local statistics), which is a
                # weak thing to ask the predictor to match.
                #
                # Patch geometry works out exactly: with L_ctx=384, L_tgt=96,
                # patch=16, stride=8, the concatenated window yields 59 patches
                # whose last 11 start at 384, 392, ..., 464 — the same spans as
                # the 11 standalone target patches, now contextualized.
                full_norm = torch.cat([context_norm, target_norm], dim=1)
                full_patches = self.patching(full_norm)
                full_embeddings = self.target_encoder(full_patches)
                target_embeddings = full_embeddings[:, -num_target_patches:, :]
            else:
                target_embeddings = self.target_encoder(target_patches)
            # [B, num_target_patches, d_model]

        # 5. Predict target representations from context embeddings
        predictions = self.predictor.forward_simple(
            context_embeddings=context_embeddings,
            num_targets=num_target_patches
        )
        # [B, num_target_patches, d_model]
        
        return {
            'predictions': predictions,
            'targets': target_embeddings.detach(),
            'context_embeddings': context_embeddings,
        }

    def forward_finetune(
        self,
        context: torch.Tensor,
        return_representations: bool = False,
        skip_revin: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised finetuning (forecasting).
        
        Uses the learned encoder and predictor to generate future representations,
        then decodes them to actual forecast values.
        
        Args:
            context: Past time series [B, context_length, C]
            return_representations: If True, also return intermediate representations
        
        Returns:
            Dictionary with 'forecast' and 'forecast_denorm'
        """
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
        
        # 4. Predict future representations
        predictions = self.predictor.forward_simple(
            context_embeddings=context_embeddings,
            num_targets=self.num_target_patches
        )
        # [B, num_target_patches, d_model]
        
        # 5. Decode to forecast values
        # If using skip_revin, it assumes data is already normalized
        # Useful for evaluations of norm data, but otherwise keep it false
        # If skip_revin, norm = denorm
        #
        # `context_embeddings` is forwarded unconditionally: the point decoders
        # ignore it, the quantile head cross-attends to it. The predictor output
        # is E[z_target | z_context] under MSE — a conditional mean — so a head
        # that only sees it cannot separate two contexts with the same expected
        # future but different volatility. Giving it the context restores that.
        forecast, forecast_denorm = self.decoder(
            predictions, skip_revin=skip_revin, context_embeddings=context_embeddings
        )
        # Point decoders: [B, prediction_length, C]
        # Quantile head:  [B, prediction_length, Q], sorted along Q

        result = {
            'forecast': forecast,
            'forecast_denorm': forecast_denorm
        }

        # A probabilistic head returns the full fan; expose the median as the
        # point forecast so every downstream caller (metrics, plots, rollout)
        # keeps working unchanged. The median is also the MAE-optimal point
        # estimate, which is what MASE scores.
        if getattr(self.decoder, 'is_probabilistic', False):
            head = self.decoder.decoder
            result['quantiles'] = forecast
            result['quantiles_denorm'] = forecast_denorm
            result['quantile_levels'] = head.quantile_levels
            result['forecast'] = head.median(forecast)
            result['forecast_denorm'] = head.median(forecast_denorm)

        if return_representations:
            result['context_embeddings'] = context_embeddings
            result['future_representations'] = predictions

        return result

    def forecast(
        self,
        context: torch.Tensor,
        n: Optional[int] = None,
        return_representations: bool = False,
        skip_revin: bool = False,
        sample_paths: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forecast n steps ahead with automatic rolling if needed.

        Normalization contract
        ----------------------
        With `skip_revin=False` (the default, and the regime the model was
        TRAINED in), the whole rollout happens in a single instance-normalized
        frame: RevIN statistics are computed once on the real context, pinned
        via `revin.freeze()`, and every roll operates in that frame. The result
        is denormalized once at the end. Mixing spaces mid-rollout — feeding a
        normalized forecast back into a raw-space context — silently produces
        garbage, which is what the previous implementation did.

        `skip_revin=True` means "the caller guarantees the context is already in
        the model's normalized frame". It is NOT the right flag for globally
        z-scored benchmark data (Nixtla/ETT): a global z-score is not a per-window
        instance normalization, and passing such data with skip_revin=True feeds
        the encoder an out-of-distribution input.

        Args:
            context: Past time series [B, context_length, C]
            n: Number of steps to forecast. Defaults to self.prediction_length.
               - If n <= prediction_length: single forward pass, truncate output
               - If n > prediction_length: rolling forecast with m passes where
                 m = ceil(n / prediction_length)
            return_representations: If True, return intermediate representations
                (only meaningful when n <= prediction_length)
            skip_revin: Only use if the context is already in the model's
                instance-normalized frame.

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
                skip_revin=skip_revin
            )
            # Every horizon-shaped tensor must be truncated, the quantile fan
            # included — leaving it at prediction_length while the point forecast
            # is cut to n silently mismatches them downstream.
            for key in ('forecast', 'forecast_denorm', 'quantiles', 'quantiles_denorm'):
                if key in result:
                    result[key] = result[key][:, :n]
            return result

        # ---- Case 2: rolling forecast ----
        use_revin = (not skip_revin) and (self.revin is not None)

        if use_revin:
            # Compute the instance statistics ONCE on the true context, then pin
            # them so every roll stays in the same frame.
            current_context = self.revin(context, mode='norm')
            self.revin.freeze()
        else:
            current_context = context.clone()

        try:
            num_rolls = (n + self.prediction_length - 1) // self.prediction_length

            # With a probabilistic head, propagate the FAN through the rolls
            # instead of the median. Median feedback makes every later roll see
            # a context smoother than real data, so intervals shrink with
            # horizon when the truth widens — measured on exchange h720: width
            # 0.267 against a sqrt(h)-growing true uncertainty.
            if sample_paths and getattr(self.decoder, 'is_probabilistic', False):
                forecast_norm, quantiles_norm, quantile_levels = \
                    self._rolling_forecast_sampled(current_context, n, num_rolls, use_revin)
                if use_revin:
                    forecast_denorm = self.revin.denormalize_target_space(forecast_norm)
                    quantiles_denorm = self.revin.denormalize_target_space(quantiles_norm)
                else:
                    forecast_denorm = forecast_norm
                    quantiles_denorm = quantiles_norm
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
                )

                forecast_norm = result['forecast']  # [B, pred_len, C] (median if probabilistic)
                all_forecasts_norm.append(forecast_norm)

                # Keep each roll's quantile fan. Note what this is and is not:
                # every roll is conditioned on the MEDIAN path fed back to it, so
                # the fan reflects the uncertainty of that roll alone and does
                # NOT accumulate the uncertainty of the rolls before it. Interval
                # widths at long horizons are therefore systematically too
                # narrow. Propagating properly would need sampling or an
                # analytic convolution across rolls; this is the honest
                # approximation, and better than reporting no fan at all.
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
    ):
        """
        Quantile-path rollout: propagate the whole fan, not the median.

        Mechanism
        ---------
        Roll 1 is a single exact forward. For later rolls the batch is expanded
        Q-fold — one copy per quantile level — and copy k is fed the level-k
        trajectory of the previous roll. Each copy then continues at its own
        level (a comonotonic coupling), and the marginal fan at every timestep
        is the per-timestep SORT of the Q paths (rearrangement, which also
        restores monotonicity where paths cross).

        What this assumes, honestly: perfect rank dependence across rolls — a
        series running at its q90 keeps running at its q90. True temporal
        dependence is weaker, so this bounds the spread from above the way
        independence would bound it from below. It replaces a systematic bias
        (median feedback shrinks intervals as the horizon grows, because the
        median path is smoother than any real trajectory) with a much smaller,
        sign-known one. Deterministic — the levels ARE the stratified sample,
        no RNG involved.

        Cost: rolls after the first run at batch B*Q (Q=9 by default). Only the
        rolling path pays it, and only for probabilistic decoders.

        Returns (forecast_norm [B,n,C], quantiles_norm [B,n,Q], levels).
        """
        head = self.decoder.decoder
        mid = head.median_idx
        pred_len = self.prediction_length
        batch = current_context.shape[0]

        # ---- Roll 1: exact, on the true context ----
        first = self.forward_finetune(current_context, skip_revin=True)
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

                rolled = self.forward_finetune(ctx, skip_revin=True)
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

    # ===== Freezing/Unfreezing methods (unchanged) =====
    
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
        print(f"✅ Saved pretrained encoder + predictor to {save_path}")