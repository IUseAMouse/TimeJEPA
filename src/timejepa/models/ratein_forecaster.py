"""
RateINForecaster: the RateIN layer composed in front of a frozen forecaster,
behind the forecaster interface itself.

The wrapper is what the training stack and the GIFT harness see: it has the
duck-typed attributes and hooks of JEPATST / SSMForecaster (`forecast`,
`robust_scaler`, `revin`, `decoder`, `patching`, `predictor`, `freeze_*`,
`set_pretrain_mode`, `prediction_length`, `delta_range`, `core_prefixes`),
so `FinetuneModule` / `SSMFinetuneModule` train it unchanged (only the
layer's selector has trainable parameters) and `evaluate_gift.py` scores it
through its ordinary path (`+ratein` off: the wrapper owns the rate).

    forecast(context [B, L, 1], n, w)
        log k = log k_prior + g(context)            selector, bounded
        y     = resample_context(context, k)        fixed model length, differentiable in k
        fan   = inner.forecast(y, ceil(n / min k), w * k)     frozen
        out   = fan_to_native(fan, k, n)            native horizon

Frames. The inner fits its robust scaler and RevIN on the resampled context
`y`; `_forward_and_loss` normalizes the native target with those statistics
and takes the pinball against `quantiles`, which is `fan_to_native` of the
model-grid normalized fan. RevIN is affine per series and commutes with the
linear `fan_to_native`; the robust scaler is a pointwise monotone arcsinh,
so target and fan sit in the same frame. `quantiles_denorm` is interpolated
in raw units (parity with the published decimate-and-reinterpolate path);
it differs from denormalizing `quantiles` by a second-order term for k != 1
and not at all at k = 1.

Rate knob. With an inner that exposes `rate_knob == "delta"`, the knob
follows the resampling as `+freq_delta` does: `w_inner = clamp(w * k)`,
`w` being the caller's base scale (`_tie_delta` in training,
`freq_delta_w(model, season, 1)` in the harness, 1 otherwise). k is NOT
detached and NOT quantized: the inner reads the period from w (it was
trained with w = 24 / season exactly) and a w off by 3 % shifts its phase
by half a sample per period, which ruins a 16-period horizon (measured
2026-10-15: loss 0.13 with the exact w against 0.16-0.22 with k rounded
to 1/8 octave, at k = 2.5-3.5); and the SSM kernel is differentiable in
its scale, so this is the main path of the gradient to k, the resampling
alone being rough at the scale of its interpolation kinks. A batch of B
distinct rates costs B kernels per SSM layer, checkpointed under autograd
(`S4DLayer._kernels`). `w_inner == w` exactly at k = 1.

Rate gradient. `rate_grad = "exact"` backpropagates the loss through the
resampling and the inner's kernel to log k. `rate_grad = "fd"` (training
only) does not: the selector's log k is detached for the forward, and the
module adds `fd_surrogate`, the straight-through term
    mean_i( log k_i * (L_i(log k_i + d) - L_i(log k_i - d)) / 2d )
with the per-item loss at the two shifted rates under no_grad. Choice: the
exact gradient is the slope of the loss at the scale of the interpolation
kinks (block j of the resampled context moves by (1023 - j) dk samples),
of order 10 and of random sign on real windows, while the slope over a
third of an octave is of order 0.03 (measured 2026-10-15): the exact
gradient is noise around the signal that matters. The fd term is the
gradient of the loss box-smoothed over +-d in log k, the two-point
version of the curve mode's kernel-smoothed grid; it costs two no_grad
forwards instead of a backward through the inner.

Left padding. `lengths` is not used: the datamodule's windows and the
harness's prepared contexts are rectangular and fully valid; the flat
prefix of a padded corpus row is history to the inner already and equals
the layer's own edge fill (the block mean of a constant is that constant).
A native past shorter than model_len * k is handed to the inner as its
valid blocks only (the longest valid suffix of the batch), so a short
series at k = 1 reaches the inner unchanged.
"""

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ..ratein_layer import K_MAX, K_MIN, MODEL_LEN, RateINLayer, RateSelector, fan_to_native, resample_context

_FAN_KEYS = ("quantiles", "quantiles_denorm", "forecast", "forecast_denorm")


class RateINForecaster(nn.Module):
    core_prefixes = ("inner.", "layer.")
    handles_nan = False

    def __init__(self, inner: nn.Module, layer: RateINLayer, log_k_prior: float = 0.0,
                 name: str = "ratein", rate_grad: str = "exact", fd_delta: float = 0.35):
        super().__init__()
        self.inner = inner
        self.layer = layer
        for p in self.inner.parameters():
            p.requires_grad_(False)
        self.inner.eval()
        self.log_k_prior = float(log_k_prior)
        self.name = str(name)
        if rate_grad not in ("exact", "fd"):
            raise ValueError(f"rate_grad must be 'exact' or 'fd', got {rate_grad!r}")
        if not float(fd_delta) > 0:
            raise ValueError("fd_delta must be positive (log k half-width of the finite difference)")
        self.rate_grad = str(rate_grad)
        self.fd_delta = float(fd_delta)
        self._fd_log_k = None
        self.last_fd_slope = None
        self.prediction_length = int(getattr(inner, "prediction_length", 0))
        self.input_length = int(layer.model_len * K_MAX)       # the harness hands the whole native past
        self.rate_knob = getattr(inner, "rate_knob", None)
        self.delta_range = getattr(inner, "delta_range", None)
        self.expects_frequency = bool(getattr(inner, "expects_frequency", False))
        self._pretrain_mode = False
        self.last_log_k = None
        self.last_k = None
        self.last_model_horizon = None
        self._last_n_unique_w = None

    # ------------------------------------------------------------ modes
    def train(self, mode: bool = True):
        """The inner stays in eval whatever Lightning asks (dropout off,
        checkpointing off); only the layer follows the mode."""
        super().train(mode)
        self.inner.eval()
        return self

    # ------------------------------------------------------- delegation
    @property
    def robust_scaler(self):
        return getattr(self.inner, "robust_scaler", None)

    @property
    def revin(self):
        return getattr(self.inner, "revin", None)

    @property
    def decoder(self):
        return getattr(self.inner, "decoder", None)

    @property
    def patching(self):
        return getattr(self.inner, "patching", None)

    @property
    def predictor(self):
        return getattr(self.inner, "predictor", None)

    def set_pretrain_mode(self, mode: bool = True):
        if mode:
            raise ValueError("RateINForecaster has no pretrain mode")
        if hasattr(self.inner, "set_pretrain_mode"):
            self.inner.set_pretrain_mode(False)
        self._pretrain_mode = False

    def is_pretrain_mode(self) -> bool:
        return False

    def forward_pretrain(self, *args, **kwargs):
        raise NotImplementedError("RateINForecaster wraps a frozen forecaster: no pretrain path")

    # The inner is frozen by construction and the layer always trains:
    # the finetune strategies have nothing to switch.
    def freeze_encoder(self):
        pass

    def unfreeze_encoder(self):
        pass

    def freeze_predictor(self):
        pass

    def unfreeze_predictor(self):
        pass

    def freeze_patching(self):
        pass

    def unfreeze_patching(self):
        pass

    def freeze_target_encoder(self):
        pass

    def freeze_revin(self):
        pass

    def unfreeze_revin(self):
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def get_num_params(self) -> Dict[str, int]:
        parts = dict(self.inner.get_num_params()) if hasattr(self.inner, "get_num_params") else {}
        parts["selector"] = sum(p.numel() for p in self.layer.parameters())
        parts["total"] = self.count_parameters(trainable_only=False)
        return parts

    # ---------------------------------------------------------- forecast
    def _inner_w(self, w, k: torch.Tensor):
        """The inner's rate knob: the caller's base scale times k, clamped
        to the inner's range, with the gradient to k kept. Without a knob
        the caller's w passes through. Returns None when nothing changes a
        missing w (every k == 1), so the inner sees exactly the call it
        would get without the layer."""
        if self.rate_knob != "delta":
            return w
        k = k.float()
        if w is None:
            if bool((k.detach() == 1.0).all()):
                self._last_n_unique_w = 1
                return None
            base = torch.ones_like(k)
        else:
            base = torch.as_tensor(w, dtype=torch.float32, device=k.device).reshape(-1)
            if base.numel() == 1:
                base = base.expand(k.shape[0])
        wi = base * k
        if self.delta_range is not None:
            wi = wi.clamp(float(self.delta_range[0]), float(self.delta_range[1]))
        self._last_n_unique_w = int(torch.unique(wi.detach()).numel())
        return wi

    def _forecast_at(self, x: torch.Tensor, lk: torch.Tensor, h: int, w=None, **kwargs) -> Dict[str, torch.Tensor]:
        """Resample at exp(lk), run the inner, bring the fan back to the
        native grid. `lk` [B] is the (bounded) log rate; its graph, if any,
        carries the exact gradient."""
        k = torch.exp(lk)
        y, valid = resample_context(x, k, self.layer.model_len)
        # A native past shorter than model_len * k fills only part of the
        # blocks: hand the inner the valid suffix (the longest one of the
        # batch), not an edge-filled prefix. At k = 1 the inner then sees a
        # short series exactly as it does without the layer.
        keep = int(valid.sum(1).max())
        if keep < y.shape[1]:
            y = y[:, -keep:]
        n_m = self.layer.model_horizon(h, k)
        out = self.inner.forecast(y.unsqueeze(-1), n=n_m, w=self._inner_w(w, k), **kwargs)
        res = {}
        for key in _FAN_KEYS:
            fan = out.get(key)
            if fan is None:
                continue
            if fan.dim() == 2:
                fan = fan.unsqueeze(-1)
            if fan.dim() == 4:
                fan = fan[..., 0]
            res[key] = fan_to_native(fan.float(), k, h)
        if "quantile_levels" in out:
            res["quantile_levels"] = out["quantile_levels"]
        self.last_log_k = lk.detach()
        self.last_k = k.detach()
        self.last_model_horizon = n_m
        return res

    def forecast(self, context: torch.Tensor, n: Optional[int] = None,
                 return_representations: bool = False, w=None, **kwargs) -> Dict[str, torch.Tensor]:
        if return_representations:
            raise NotImplementedError("RateINForecaster returns no representations")
        x = context[..., 0] if context.dim() == 3 else context
        h = int(n or self.prediction_length)
        lk = self.layer.log_k(x, None, self.log_k_prior)
        self._fd_log_k = None
        if self.training and self.rate_grad == "fd":
            # The forward carries no graph to the selector: the module adds
            # fd_surrogate(), which holds the selector's log k.
            self._fd_log_k = lk
            lk = lk.detach()
        return self._forecast_at(x, lk, h, w, **kwargs)

    def fd_surrogate(self, context: torch.Tensor, target, w, h: int, per_item_loss) -> torch.Tensor:
        """The straight-through term of the fd mode, after a training
        forward: `per_item_loss(results, target) -> [B]` is the module's own
        loss evaluated per item (normalized frame of the shifted forward),
        called twice under no_grad at log k +- fd_delta (bounded as the
        selector's output is). Returns mean(log k * slope) whose gradient
        w.r.t. the selector is the finite-difference slope."""
        if self._fd_log_k is None:
            raise RuntimeError("fd_surrogate needs a training forward in rate_grad='fd' first")
        x = context[..., 0] if context.dim() == 3 else context
        lk = self._fd_log_k
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
        kept = (self.last_log_k, self.last_k, self.last_model_horizon, self._last_n_unique_w)
        with torch.no_grad():
            losses = []
            for sign in (1.0, -1.0):
                shifted = self.layer.bound_log_k(lk.detach() + sign * self.fd_delta, lengths)
                losses.append(per_item_loss(self._forecast_at(x, shifted, h, w), target))
            slope = (losses[0] - losses[1]) / (2.0 * self.fd_delta)
        # The diagnostics describe the selector's forward, not the shifted ones.
        self.last_log_k, self.last_k, self.last_model_horizon, self._last_n_unique_w = kept
        self.last_fd_slope = slope.detach()
        self._fd_log_k = None
        return (lk * slope).mean()

    def forward(self, context: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.forecast(context, **kwargs)

    def rate_diagnostics(self) -> Dict[str, float]:
        """Statistics of the last forecast's rates, for the trainer's and
        the harness's logs; empty before the first call."""
        if self.last_log_k is None:
            return {}
        k, g = self.last_k, self.last_log_k - self.log_k_prior
        out = {"k_median": float(k.median()), "k_min": float(k.min()), "k_max": float(k.max()),
               "frac_k_gt1": float((k > 1.0001).float().mean()),
               "g_mean": float(g.mean()), "g_std": float(g.std()) if g.numel() > 1 else 0.0,
               "model_horizon": float(self.last_model_horizon)}
        if self._last_n_unique_w is not None:
            out["n_unique_w"] = float(self._last_n_unique_w)
        if self.last_fd_slope is not None:
            out["fd_slope_mean"] = float(self.last_fd_slope.mean())
            out["fd_slope_abs"] = float(self.last_fd_slope.abs().mean())
        return out


# ------------------------------------------------------------------ builder
def build_from_config(cfg: DictConfig) -> RateINForecaster:
    """`model.builder: "timejepa.models.ratein_forecaster:build_from_config"`.

    Reads `cfg.model.ratein_layer`:
      inner_builder    the inner's own builder ("module:function"), required
      inner_checkpoint weights of the inner for training (loaded through
                       `load_checkpoint`, the inner's refusal contract);
                       null at evaluation, the Lightning checkpoint of the
                       wrapper carries the inner
      model_len, n_modes, hidden, min_period, max_period, q, min_blocks,
      k_min, k_max, k_prior, rate_grad (exact | fd), fd_delta
    The inner is built from a copy of the config whose `model.builder` is
    `inner_builder` and whose `model.seq_length` is `model_len`.
    `training.pretrained_encoder_path` must stay null for the wrapper: it
    loads unprefixed keys into `self.model`, which here are `inner.*`.
    """
    from ..evaluation.loading import create_model_from_config, load_checkpoint
    rl = cfg.model.get("ratein_layer")
    if rl is None or not rl.get("inner_builder"):
        raise ValueError("model.ratein_layer.inner_builder is required ('module:function' of the inner)")
    model_len = int(rl.get("model_len", MODEL_LEN))
    inner_cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(inner_cfg, False)
    inner_cfg.model.builder = str(rl.inner_builder)
    inner_cfg.model.seq_length = model_len
    inner = create_model_from_config(inner_cfg)
    if rl.get("inner_checkpoint"):
        inner = load_checkpoint(inner, str(rl.inner_checkpoint), torch.device("cpu"))
    selector = RateSelector(int(rl.get("n_modes", 48)), int(rl.get("hidden", 32)),
                            float(rl.get("min_period", 2.0)), float(rl.get("max_period", 2048.0)),
                            float(rl.get("q", 4.0)))
    layer = RateINLayer(selector, model_len=model_len, k_min=float(rl.get("k_min", K_MIN)),
                        k_max=float(rl.get("k_max", K_MAX)), min_blocks=int(rl.get("min_blocks", 32)))
    if "k_quant_octave" in rl:
        raise ValueError("model.ratein_layer.k_quant_octave is gone: w follows k exactly (the inner reads "
                         "the period from w; a rounded w shifts its phase)")
    return RateINForecaster(inner, layer, log_k_prior=math.log(float(rl.get("k_prior", 1.0))),
                            name=str(cfg.model.get("name", "ratein")),
                            rate_grad=str(rl.get("rate_grad", "exact")), fd_delta=float(rl.get("fd_delta", 0.35)))
