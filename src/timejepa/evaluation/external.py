"""
Third-party forecasters behind the GIFT harness contract (2026-09-13).

RateIN is a model-agnostic inference layer; to say so in a paper it has to be
measured on models people already use, through the SAME harness, with the
SAME instances, flags and caches as TimeJEPA and TimeSSM. Each adapter here
exposes what `scripts/evaluate_gift.py` reads from a model:

    input_length            context cap (prepare_context keeps the last L points)
    patching.{patch_size, stride}   1 / 1: the harness never truncates or pads
    predictor.w_film = None         +ratein_w refused
    rate_knob = None                +ratein=delta refused
    forecast(batch [B, L, 1], n, w=None) -> {forecast_denorm [B, n, 1],
                                             quantiles_denorm [B, n, Q],
                                             quantile_levels}

The models are built by `build(cfg)` from `cfg.model.external` (read through
TimeJEPA's `model.builder` dispatch), never loaded from a checkpoint: the
harness keys their cache on the Hugging Face id instead.

Kinds:
  chronos   Chronos-Bolt (tiny 9M, small 48M, base 205M) and Chronos-2 (120M),
            `chronos-forecasting` >= 2.3. Native 9-quantile fan.
  t0        t0-alpha (The Forecasting Company, 102M), `tfc-t0`. Native
            quantiles 0.1/0.25/0.5/0.75/0.9, the others interpolated by the
            package.
  ttm       TinyTimeMixer (IBM Granite, 1-5M), `granite-tsfm`. Point-only:
            the fan is the point repeated (CRPS = MAE-like, as in the
            2026-09-06 paired reading). Autoregressive beyond its horizon.

Install: uv pip install -e ".[external]"  (chronos + t0; ttm is the `ttm`
extra). toto-ts is NOT included: its pins downgrade torch to 2.7 and
transformers to 4.x, incompatible with this environment - run it in its own
venv if ever.
"""

from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch

LEVELS = tuple(round(0.1 * j, 1) for j in range(1, 10))


class ExternalForecaster:
    """Harness contract for a model that is not a JEPATST."""
    rate_knob = None
    kind = "external"

    def __init__(self, name: str, hf_id: str, context_length: int, device):
        self.name = name
        self.hf_id = hf_id
        self.input_length = int(context_length)
        self.patching = SimpleNamespace(patch_size=1, stride=1)
        self.predictor = SimpleNamespace(w_film=None)
        self.device = device
        self.quantile_levels = list(LEVELS)
        self.median_idx = self.quantile_levels.index(0.5)

    # ------------------------------------------------------------ contract
    def _predict(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        """ctx [B, L] float32 (cpu) -> quantiles [B, n, Q] float32 (cpu),
        Q = len(self.quantile_levels), sorted along Q."""
        raise NotImplementedError

    @torch.no_grad()
    def forecast(self, batch: torch.Tensor, n: Optional[int] = None,
                 w=None, **kwargs) -> dict:
        if w is not None and bool((torch.as_tensor(w).float() != 1.0).any()):
            raise ValueError(f"{self.name} has no rate knob: w must be None or 1")
        if n is None:
            raise ValueError("external forecasters need an explicit horizon n")
        x = batch[..., 0] if batch.dim() == 3 else batch
        x = x.detach().float().cpu()[:, -self.input_length:]
        q = self._predict(x, int(n)).float().cpu()
        if q.shape != (x.shape[0], int(n), len(self.quantile_levels)):
            raise RuntimeError(f"{self.name}: quantiles shape {tuple(q.shape)}, expected "
                               f"{(x.shape[0], int(n), len(self.quantile_levels))}")
        q, _ = torch.sort(q, dim=-1)                      # monotone fan, no crossing
        med = q[:, :, self.median_idx].unsqueeze(-1)
        return {"forecast": med, "forecast_denorm": med,
                "quantiles": q, "quantiles_denorm": q,
                "quantile_levels": list(self.quantile_levels)}

    # no-ops so generic code can call them
    def eval(self):
        return self

    def to(self, device):
        return self


# --------------------------------------------------------------- Chronos
class ChronosForecaster(ExternalForecaster):
    """Chronos-Bolt and Chronos-2 through `chronos.BaseChronosPipeline`.
    The pipeline truncates to its own context length and rolls out beyond
    its native horizon (Bolt: 64 steps) by feeding its quantiles back."""
    kind = "chronos"

    def __init__(self, hf_id: str, device, context_length: int = 2048,
                 name: Optional[str] = None, torch_dtype: Optional[str] = None):
        from chronos import BaseChronosPipeline
        super().__init__(name or hf_id.split("/")[-1], hf_id, context_length, device)
        dtype = (getattr(torch, torch_dtype) if torch_dtype
                 else (torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32))
        self.pipeline = BaseChronosPipeline.from_pretrained(
            hf_id, device_map=str(device), torch_dtype=dtype)
        native = getattr(self.pipeline, "model_context_length", None)
        if native is not None:
            self.input_length = min(self.input_length, int(native))

    def _predict(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        quantiles, _ = self.pipeline.predict_quantiles(
            ctx, prediction_length=n, quantile_levels=list(self.quantile_levels))
        if isinstance(quantiles, (list, tuple)):          # Chronos-2: one per series
            quantiles = torch.stack([q[0] for q in quantiles])
        return quantiles


# --------------------------------------------------------------- t0-alpha
class T0Forecaster(ExternalForecaster):
    """t0-alpha (The Forecasting Company) through `t0.T0Forecaster.predict`;
    quantiles the model was not trained on are interpolated by the package."""
    kind = "t0"

    def __init__(self, hf_id: str, device, context_length: int = 2048,
                 name: Optional[str] = None):
        from t0 import T0Forecaster as _T0
        super().__init__(name or hf_id.split("/")[-1], hf_id, context_length, device)
        try:
            self.model = _T0.from_pretrained(hf_id).eval().to(device)
        except TypeError as e:
            # huggingface_hub instantiates the class WITHOUT its config when
            # config.json could not be fetched - the repo is gated (measured
            # 2026-09-13: 401 on config.json). Say so instead of the
            # "missing 8 positional arguments" it produces.
            raise RuntimeError(
                f"{hf_id} could not be loaded: the repo is gated. Accept its "
                "license on huggingface.co and authenticate (`hf auth login`), "
                f"then retry. Underlying error: {e}") from e

    def _predict(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        out = self.model.predict(ctx.to(self.device), horizon=n,
                                 quantiles=list(self.quantile_levels))
        return out.quantiles


# --------------------------------------------------------------- TTM
class TTMForecaster(ExternalForecaster):
    """TinyTimeMixer (granite-tsfm). Point forecaster: the fan is the point
    repeated over the 9 levels. Contexts shorter than its window are
    left-padded with their first value (the model's own convention);
    horizons beyond its window are rolled out autoregressively."""
    kind = "ttm"

    def __init__(self, hf_id: str, device, revision: str = "main",
                 name: Optional[str] = None, context_length: Optional[int] = None):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        model = TinyTimeMixerForPrediction.from_pretrained(hf_id, revision=revision)
        model.to(device).eval()
        ctx_len = int(model.config.context_length)
        super().__init__(name or f"{hf_id.split('/')[-1]}-{revision}", hf_id,
                         context_length or ctx_len, device)
        self.model = model
        self.ctx_len = ctx_len
        self.pred_len = int(model.config.prediction_length)

    def _point(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        cur = ctx
        outs = []
        remaining = n
        while remaining > 0:
            base = cur[:, -self.ctx_len:]
            if base.shape[1] < self.ctx_len:
                pad = base[:, :1].expand(-1, self.ctx_len - base.shape[1])
                base = torch.cat([pad, base], dim=1)
            pred = self.model(past_values=base.unsqueeze(-1).to(self.device)).prediction_outputs
            pred = pred[:, :, 0].float().cpu()                     # [B, P]
            step = pred[:, :min(self.pred_len, remaining)]
            outs.append(step)
            cur = torch.cat([cur, step], dim=1)
            remaining -= step.shape[1]
        return torch.cat(outs, dim=1)

    def _predict(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        pts = self._point(ctx, n)
        return pts.unsqueeze(-1).expand(-1, -1, len(self.quantile_levels)).clone()


# --------------------------------------------------------------- builder
KINDS = {"chronos": ChronosForecaster, "t0": T0Forecaster, "ttm": TTMForecaster}


def build(cfg):
    """`model.builder: timejepa.evaluation.external:build` - reads
    cfg.model.external = {kind, hf_id, context_length?, revision?, dtype?}."""
    ext = cfg.model.get("external")
    if not ext:
        raise ValueError("model.external is required for the external builder")
    kind = str(ext.get("kind", "")).lower()
    if kind not in KINDS:
        raise ValueError(f"unknown external kind {kind!r} (choose from {sorted(KINDS)})")
    device = torch.device(str(ext.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    kwargs = {"hf_id": str(ext.hf_id), "device": device, "name": str(cfg.model.name)}
    if ext.get("context_length") is not None:
        kwargs["context_length"] = int(ext.context_length)
    if kind == "ttm" and ext.get("revision") is not None:
        kwargs["revision"] = str(ext.revision)
    if kind == "chronos" and ext.get("dtype") is not None:
        kwargs["torch_dtype"] = str(ext.dtype)
    return KINDS[kind](**kwargs)


def run_identity(hf_id: str, revision: Optional[str] = None) -> tuple:
    """(cache stem, fingerprint) for a model without a checkpoint file: the
    Hugging Face id (and revision) name the run; the fingerprint is the id,
    so a different id in the same directory is refused like an overwritten
    checkpoint would be."""
    stem = hf_id.replace("/", "__") + (f"@{revision}" if revision else "")
    return stem, f"hf:{hf_id}" + (f"@{revision}" if revision else "")
