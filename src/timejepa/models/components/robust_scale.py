"""
Robust arcsinh scaling (G8.4) - COMPOSED around RevIN, never in its place.

The contract, decided 2026-08-20 (PLAN.md G8.4):

    input:  x' = arcsinh((x - median(ctx)) / MAD_scaled(ctx))
    output: x  = sinh(x') * MAD_scaled + median

RevIN and its whole denormalization contract (freeze / to_input_frame /
denormalize_target_space - the B2/B3/B10 scars) stay INTACT: they simply
operate in compressed space. The entire rollout, the finetune loss (pinball in
normalized space) and the predictor live in that space; only the `*_denorm`
outputs go back to raw space through the inverse.

Why it works where the z-score breaks (measured, not assumed):
* Epsilon floor (G6): a near-constant context puts the RevIN scale at ~0.003
  and projects the target to THOUSANDS of sigma. arcsinh is logarithmic in the
  tail: arcsinh(6300) ~ 9.4 - the outlier becomes an ordinary number instead
  of eating the gradient.
* Heavy tails (E17, the "domain" half): a x100 spike in a VM trace inflates
  the std and crushes the rest of the signal to zero. The MAD ignores the
  spike; arcsinh compresses the spike itself. bitbrains/car_parts/bizitobs -
  our worst configs - are exactly this regime. Precedent: Toto (Datadog, born
  from cloudops) uses a "robust arcsinh scaler".

Properties that make the composition safe:
* arcsinh is STRICTLY MONOTONIC -> quantiles are equivariant: the
  probabilistic head predicts in compressed space and the pointwise inverse
  yields valid, ordered quantiles in raw space. The median commutes with the
  inverse (median(sinh(q)) = sinh(median(q))); we never use the mean, which
  would not commute.
* MAD * 1.4826 is a consistent sigma estimator for a Gaussian, and
  arcsinh(z) ~ z for |z| <~ 1: on a well-behaved window the transformation is
  nearly the z-score identity - existing behavior is preserved where it
  worked, only the tails change.
* NO learned parameter here. The statistics are runtime attributes (like
  revin.mean/std) - but a marker buffer is registered so checkpoints DECLARE
  themselves: loading an arcsinh checkpoint into a bare model (or the reverse)
  would give silently wrong numbers; the marker makes the mismatch detectable
  by the P3.2 refusal contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# MAD -> equivalent standard deviation for a Gaussian. Keeps arcsinh ~ z-score
# on well-behaved windows.
MAD_TO_SIGMA = 1.4826


class RobustScale(nn.Module):
    """
    Per-instance, per-channel robust transform, CONTEXT statistics.

    Same lifecycle semantics as RevIN: `fit(context)` computes and stores
    (median, scale); `transform`/`inverse` reuse them - the target and every
    output are handled in the context's frame, never their own (the target's
    frame would leak the future).
    """

    # Scale fallback and floor - fix of 2026-08-22, measured on the mix
    # finetune (CRPS at 10^10..inf on bitbrains/kdd at 5-10% of an epoch).
    #
    # The pathology: on a "flat + spikes" window (idle VM - 29% of
    # bitbrains_rnd contexts have MAD EXACTLY 0), the old floor eps=1e-8 gave
    # a 1e-8 scale, hence a compressed frame offset by ln(1e8) ~ 18: the
    # target landed at |arcsinh| ~ 20-38, and the sinh INVERSE re-amplified
    # exponentially - sinh(34) ~ 3e14, float32 overflow at ~89. Even a fan
    # perfectly trained in that frame inverted into astronomical raw
    # intervals: structural, not transient. This is the G6 epsilon-floor
    # pathology, resurrected one level up.
    #
    # The fix exploits an asymmetry: a TOO LARGE scale is benign (arcsinh
    # becomes near linear and RevIN - which follows in the composition -
    # renormalizes: graceful degradation to z-score behavior); a too small
    # scale is catastrophic (the logarithmic offset explodes at the inverse).
    # BUT the fallback must be CONDITIONAL, not a max: an unconditional
    # max(MAD, 0.1*std) would let a lone spike reinflate the scale on a
    # healthy window - exactly the E17 pathology the MAD exists to ignore
    # (caught by the existing spike test). So:
    #   scale = MAD*1.4826                     if MAD*1.4826 > 0.01*std
    #         = max(0.1*std, eps)              otherwise (MAD collapsed)
    # * healthy windows (MAD ~ std, spike or not): MAD stays in charge,
    #   behavior STRICTLY identical to before the fix;
    # * flat + spikes (MAD ~ 0 against the std): 0.1*std takes over - the
    #   only scale estimator still available in that regime;
    # * strict constants (std = 0 too): eps = 1e-3 bounds the frame at
    #   arcsinh(X/1e-3) ~ ln(2000*X) - finite and log-bounded, never +18
    #   again.
    STD_FALLBACK = 0.1
    MAD_COLLAPSE_GATE = 0.01

    # Forecast envelope relative to the context - G8.4b fix of 2026-08-23,
    # measured on the mix_zs_1ep3e4 run at 15% of an epoch:
    # bitbrains_fast_storage/H/short showed a CRPS of 18,305,724 (the config
    # sits at 0.62-0.67 on all neighboring evals), i.e. x1.19 on the geomean
    # aggregate of the 97 configs ON ITS OWN. Diagnosis: NOT the floor scale
    # (the frame was bounded) - the half-trained quantile head emits a tail
    # quantile |z| ~ 15 in compressed space, which sinh re-amplifies into
    # sinh(15)*scale ~ 10^6*scale. The floor cannot stop a rogue z: the right
    # guard is DOWNSTREAM, on the inverse.
    #
    # The encoded prior: a forecast does not leave
    #     [min(ctx) - K*w, max(ctx) + K*w],  w = max(range(ctx), scale)
    # with K = 10 - ten times the context range beyond its bounds, far above
    # any plausible benchmark future, far below sinh accidents. Structural
    # precedent: the Chronos vocabulary bounds its outputs at +-15 sigma by
    # construction. The clamp is monotonic (quantile order survives) and
    # INACTIVE on any reasonable forecast - only tail accidents are touched.
    # `w` is protected by the scale for degenerate contexts (range 0).
    # Expected measurable bonus: also bounds the x10-30 upward bias of
    # near-zero windows (the london_smart_meters observation that opened
    # G8.4b).
    FORECAST_ENVELOPE = 10.0

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # Checkpoint self-description marker (see docstring). No gradient, no
        # influence on the computation.
        self.register_buffer("is_robust", torch.ones(1))
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.ctx_min: torch.Tensor | None = None
        self.ctx_max: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        """context: [B, L, C] - stats over L, per instance and per channel."""
        med = context.median(dim=1, keepdim=True).values                # [B,1,C]
        mad = (context - med).abs().median(dim=1, keepdim=True).values  # [B,1,C]
        std = context.std(dim=1, keepdim=True)                          # [B,1,C]
        mad_sigma = mad * MAD_TO_SIGMA
        fallback = (self.STD_FALLBACK * std).clamp_min(self.eps)
        self.median = med.detach()
        self.scale = torch.where(
            mad_sigma > self.MAD_COLLAPSE_GATE * std, mad_sigma, fallback
        ).clamp_min(self.eps).detach()
        # Context bounds for the forecast envelope (see FORECAST_ENVELOPE).
        self.ctx_min = context.amin(dim=1, keepdim=True).detach()
        self.ctx_max = context.amax(dim=1, keepdim=True).detach()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.median is not None, "call fit(context) first"
        return torch.asinh((x - self.median) / self.scale)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pointwise inverse, bounded by the context envelope (see
        FORECAST_ENVELOPE). For a quantile tensor [B, n, Q], the monotonicity
        of sinh AND of the clamp preserves the level order - no re-sort
        needed. Any value whose exact inverse falls INSIDE the envelope (i.e.
        any reasonable forecast, and any transform->inverse round trip of real
        data) is restored exactly.
        """
        assert self.median is not None, "call fit(context) first"
        med, scale = self.median, self.scale
        if x.dim() == med.dim() and x.shape[-1] != med.shape[-1]:
            # quantile outputs [B, n, Q] against stats [B, 1, C=1]: the stats
            # broadcast over the Q dimension (univariate, C=1).
            pass  # the [B,1,1] -> [B,n,Q] broadcast is already correct
        raw = torch.sinh(x) * scale + med
        half = self.FORECAST_ENVELOPE * torch.maximum(self.ctx_max - self.ctx_min, scale)
        return raw.clamp(self.ctx_min - half, self.ctx_max + half)
