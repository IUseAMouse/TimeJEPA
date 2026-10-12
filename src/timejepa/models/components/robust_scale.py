"""
Robust arcsinh scaling, composed around RevIN (never in its place):

    input:  x' = arcsinh((x - median(ctx)) / MAD_scaled(ctx))
    output: x  = sinh(x') * MAD_scaled + median

RevIN, the rollout, the finetune loss and the predictor all operate in the
compressed space; only the `*_denorm` outputs go back to raw space.

Why: a near-constant context puts the RevIN scale near zero and projects the
target to thousands of sigma, and a heavy-tailed spike inflates the std and
crushes the rest of the signal. The MAD ignores the spike and arcsinh is
logarithmic in the tail. Precedent: the robust arcsinh scaler of Toto.

arcsinh is strictly monotonic, so quantiles predicted in compressed space
invert to valid, ordered quantiles and the median commutes with the inverse.
MAD * 1.4826 estimates sigma for a Gaussian and arcsinh(z) ~ z for |z| < 1,
so well-behaved windows stay close to the z-score. There is no learned
parameter; a marker buffer lets checkpoints declare the transform so the
loader can refuse a mismatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# MAD to standard deviation for a Gaussian.
MAD_TO_SIGMA = 1.4826


class RobustScale(nn.Module):
    """
    Per-instance, per-channel robust transform with context statistics.

    Same lifecycle as RevIN: `fit(context)` computes and stores (median,
    scale); `transform`/`inverse` reuse them, so the target and every output
    are handled in the context's frame (the target's frame would leak the
    future).
    """

    # Scale fallback and floor. On a flat window with spikes the MAD is
    # exactly 0; a tiny scale offsets the compressed frame by a large
    # logarithm and the sinh inverse then overflows. A too large scale is
    # benign (arcsinh becomes near linear and RevIN renormalizes), so the
    # fallback is conditional rather than a max, which would let a lone
    # spike reinflate the scale on a healthy window:
    #   scale = MAD*1.4826          if MAD*1.4826 > 0.01*std
    #         = max(0.1*std, eps)   otherwise (MAD collapsed)
    # For strict constants eps = 1e-3 keeps the frame log-bounded.
    STD_FALLBACK = 0.1
    MAD_COLLAPSE_GATE = 0.01

    # Forecast envelope relative to the context. A half-trained quantile head
    # can emit a tail quantile around |z| = 15 in compressed space, which
    # sinh amplifies to 10^6 times the scale; the floor cannot stop that, so
    # the inverse is clamped to
    #     [min(ctx) - K*w, max(ctx) + K*w],  w = max(range(ctx), scale)
    # with K = 10. The clamp is monotonic (quantile order survives) and
    # inactive on any reasonable forecast. Precedent: the Chronos vocabulary
    # bounds its outputs at 15 sigma by construction.
    FORECAST_ENVELOPE = 10.0

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        # Checkpoint marker; no influence on the computation.
        self.register_buffer("is_robust", torch.ones(1))
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.ctx_min: torch.Tensor | None = None
        self.ctx_max: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        """context: [B, L, C]; statistics over L, per instance and per channel."""
        med = context.median(dim=1, keepdim=True).values                # [B,1,C]
        mad = (context - med).abs().median(dim=1, keepdim=True).values  # [B,1,C]
        # A one-point context has no unbiased std (torch returns NaN); use
        # zero and let the eps floor take over.
        if context.shape[1] > 1:
            std = context.std(dim=1, keepdim=True)                      # [B,1,C]
        else:
            std = torch.zeros_like(med)
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
        FORECAST_ENVELOPE). sinh and the clamp are both monotonic, so a
        quantile tensor [B, n, Q] keeps its level order. Values whose exact
        inverse falls inside the envelope are restored exactly.
        """
        assert self.median is not None, "call fit(context) first"
        med, scale = self.median, self.scale
        if x.dim() == med.dim() and x.shape[-1] != med.shape[-1]:
            # Quantile outputs [B, n, Q] against stats [B, 1, 1]: the
            # broadcast over Q is already correct.
            pass
        raw = torch.sinh(x) * scale + med
        half = self.FORECAST_ENVELOPE * torch.maximum(self.ctx_max - self.ctx_min, scale)
        return raw.clamp(self.ctx_min - half, self.ctx_max + half)
