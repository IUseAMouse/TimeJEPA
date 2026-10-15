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
        fan   = inner.forecast(y, ceil(n / min k), w * k_q)   frozen
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
follows the resampling as `+freq_delta` does: `w_inner = clamp(w * k_q)`,
`w` being the caller's base scale (`_tie_delta` in training,
`freq_delta_w(model, season, 1)` in the harness, 1 otherwise). Choice: k_q
is k quantized to 1/8 octave, detached, so a batch of continuous rates costs
a handful of SSM kernels (`S4DLayer._scales` computes one per distinct
scale) and `w_inner == w` exactly at k = 1. The gradient reaches k through
the resampling only.

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
                 k_quant_octave: int = 8, name: str = "ratein"):
        super().__init__()
        self.inner = inner
        self.layer = layer
        for p in self.inner.parameters():
            p.requires_grad_(False)
        self.inner.eval()
        self.log_k_prior = float(log_k_prior)
        self.k_quant_octave = int(k_quant_octave)
        self.name = str(name)
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
        """The inner's rate knob: the caller's base scale times k quantized
        to 1/k_quant_octave octave, clamped to the inner's range. Without a
        knob the caller's w passes through. Returns None when nothing
        changes a missing w (every k_q == 1), so the inner sees exactly the
        call it would get without the layer."""
        if self.rate_knob != "delta":
            return w
        step = math.log(2.0) / self.k_quant_octave
        kq = torch.exp(torch.round(torch.log(k.detach().float()) / step) * step)
        if w is None:
            if bool((kq == 1.0).all()):
                self._last_n_unique_w = 1
                return None
            base = torch.ones_like(kq)
        else:
            base = torch.as_tensor(w, dtype=torch.float32, device=k.device).reshape(-1)
            if base.numel() == 1:
                base = base.expand(kq.shape[0])
        wi = base * kq
        if self.delta_range is not None:
            wi = wi.clamp(float(self.delta_range[0]), float(self.delta_range[1]))
        self._last_n_unique_w = int(torch.unique(wi).numel())
        return wi

    def forecast(self, context: torch.Tensor, n: Optional[int] = None,
                 return_representations: bool = False, w=None, **kwargs) -> Dict[str, torch.Tensor]:
        if return_representations:
            raise NotImplementedError("RateINForecaster returns no representations")
        x = context[..., 0] if context.dim() == 3 else context
        h = int(n or self.prediction_length)
        lk = self.layer.log_k(x, None, self.log_k_prior)
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
      k_min, k_max, k_prior, k_quant_octave
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
    return RateINForecaster(inner, layer, log_k_prior=math.log(float(rl.get("k_prior", 1.0))),
                            k_quant_octave=int(rl.get("k_quant_octave", 8)),
                            name=str(cfg.model.get("name", "ratein")))
