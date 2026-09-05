"""
GIFT-Eval harness - evaluates a TimeJEPA checkpoint on the official 97 configs.

    # 1. download the benchmark data (once, ~a few GB)
    make gift-download

    # 2. run (CPU is fine for the tiny models - that is their point)
    python scripts/evaluate_gift.py --config-name lotsa_tiny_eval \\
        +checkpoint_path=checkpoints/timejepa_lotsa_tiny_zs/pretrain_False/<ckpt>

    # useful extras
    #   +gift_configs='electricity/H/short,us_births/D/short'  restrict the set
    #   +gift_terms=short                                      one term only
    #   +gift_max_series=50                                    debug subsample

The protocol lives in src/timejepa/evaluation/gift.py, transcribed constant by
constant from the official harness - read its module docstring before touching
anything here. This script only does the plumbing: batching contexts, calling
`model.forecast`, streaming instances into the metric accumulators, and writing
three artifacts under evaluation/<model>/<ckpt>/gift/:

  per_config/<config>.json   one file per config, written as soon as the config
                             finishes -> a killed run RESUMES for free (the file
                             acts as the marker; delete it to force a re-run)
  all_results.csv            the official leaderboard row format, directly
                             comparable to results/* of the gift-eval repo
  summary.json               the two leaderboard aggregates: geometric mean of
                             per-config ratios vs Seasonal Naive, computed both
                             against the OFFICIAL SN numbers and against a SN
                             run through this very code path (any gap between
                             the two normalizations measures our convention
                             drift vs gluonts, so it is printed, not hidden)

MSIS is reported as NaN: it needs the 0.025/0.975 quantiles and the decoder's
grid is the leaderboard's nine 0.1..0.9 levels. The leaderboard aggregate does
not use MSIS.
"""

import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift  # noqa: E402
from timejepa.evaluation import ratein as ratein_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_gift")

# Same silencing as prepare_lotsa.py: HF logs every arrow file at INFO.
for _noisy in ("datasets", "huggingface_hub", "fsspec", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Context preparation
# ---------------------------------------------------------------------------

def prepare_context(past: np.ndarray, max_len: int, stride: int,
                    min_len: int) -> np.ndarray:
    """
    Turn a raw past into a model-ready context.

    * keep the most recent `max_len` points (the model's trained maximum);
    * NaNs are linearly interpolated (edge-filled at the ends) - model INPUT
      only, targets are never imputed;
    * truncate FROM THE LEFT to a multiple of `stride`: Patching would
      otherwise right-pad by repeating the final value, i.e. fabricate
      observations after the true last point, exactly where they hurt most;
    * series shorter than `min_len` (one patch) are edge-padded on the left.
    """
    ctx = np.asarray(past[-max_len:], dtype=np.float32)

    if np.isnan(ctx).any():
        idx = np.arange(len(ctx))
        finite = ~np.isnan(ctx)
        if not finite.any():
            return None
        ctx = np.interp(idx, idx[finite], ctx[finite]).astype(np.float32)

    if len(ctx) > min_len:
        ctx = ctx[len(ctx) % stride:]
    if len(ctx) < min_len:
        ctx = np.concatenate([np.full(min_len - len(ctx), ctx[0],
                                      dtype=np.float32), ctx])
    return ctx


# ---------------------------------------------------------------------------
# TTA - inference procedures UNIFORM across the 97 configs (roadmap S1.5,
# decision 2026-08-24: TTA yes / multi-checkpoints no).
# Single checkpoint; each variant is a deterministic transform of the context,
# identical for all configs - no per-config adaptation.
# ---------------------------------------------------------------------------

def _negated(out: dict) -> dict:
    """Sign-flip mirror: q_k(-X) = -q_{1-k}(X). The median negates; the fan
    negates AND reverses along the level axis (ascending order preserved).
    Precedents: TimesFM `force_flip_invariance`, YingLong. Well defined under
    arcsinh/RevIN: both frames are odd around their center (median/mean of the
    negated context = negation of the center)."""
    res = {"forecast_denorm": -out["forecast_denorm"]}
    q = out.get("quantiles_denorm")
    if q is not None:
        qdim = -2 if q.dim() == 4 else -1
        res["quantiles_denorm"] = torch.flip(-q, dims=(qdim,))
    if "quantile_levels" in out:
        res["quantile_levels"] = out["quantile_levels"]
    return res


def tta_forecast(model, batch: torch.Tensor, h: int,
                 lookbacks=None, flip: bool = False, shifts=None,
                 w=None) -> dict:
    """
    Average of inference variants of the SAME checkpoint:
      * multi-lookback: context truncated to L' for each L' in `lookbacks`
        (capped to available length, stride-aligned);
      * optional sign-flip mirror on each variant;
      * `shifts` (2026-08-25, translation equivariance): for each
        s, a variant forecasts from origin t-s (the s most recent points
        removed, length re-aligned on the stride, so the patch-grid PHASE
        changes); its forecast is REALIGNED onto the horizon (variant index
        p+s-1 = horizon position p) and averaged per position with a coverage
        mask (the last s positions are only covered by less-shifted variants).
        Recommended: small s (1..7 = less than one patch of staleness).
    Theory note (tested, not just asserted): the scale variant f(kx)/k is an
    EXACT NO-OP under RobustScale+RevIN (median and MAD are 1-homogeneous, so
    the normalized input is bit-identical) - the only affine-group element not
    quotiented out by normalization is the SIGN, hence the flip.
    Fans are averaged level by level (a weighted mean of sorted vectors stays
    sorted); the median is re-read from the averaged fan.
    With no options: strictly equivalent to model.forecast.
    """
    L = batch.shape[1]
    stride = model.patching.stride
    looks = sorted({min(int(lb), L) - (min(int(lb), L) % stride)
                    for lb in (lookbacks or [L])
                    if min(int(lb), L) >= model.patching.patch_size})
    if not looks:
        looks = [L]
    # s >= h: the shifted variant would cover NO horizon position at all
    # (measured: m4_yearly has h=6 - a shift of 7 broke the alignment).
    shift_list = [0] + sorted({int(s) for s in (shifts or [])
                               if 0 < int(s) < h})

    outs = []                                  # (output, shift s)
    for lb in looks:
        for s in shift_list:
            ctx = batch[:, -lb:L - s if s else None] if s else batch[:, -lb:]
            # stride re-alignment: trim on the LEFT (never on the right -
            # Patching's padding would fabricate points after the origin)
            trim = ctx.shape[1] % stride
            if trim:
                ctx = ctx[:, trim:]
            if ctx.shape[1] < model.patching.patch_size:
                continue
            # w (RateIN x w): the rate is invariant under negation -> same w
            # on the mirrored variant. Only forwarded when set, so test
            # doubles without a w kwarg keep working.
            kw = {} if w is None else {"w": w}
            outs.append((model.forecast(ctx, n=h, **kw), s))
            if flip:
                outs.append((_negated(model.forecast(-ctx, n=h, **kw)), s))

    if len(outs) == 1:
        return outs[0][0]

    def _aligned_mean(key):
        vals = [(o.get(key), s) for o, s in outs]
        if any(v is None for v, _ in vals):
            return None
        ref = vals[0][0]
        acc = torch.zeros_like(ref)
        cnt = torch.zeros(h, device=ref.device)
        for v, s in vals:
            if s == 0:
                acc += v
                cnt += 1.0
            else:
                # variant indices s..h-1 map to horizon positions 0..h-s-1
                acc[:, :h - s] += v[:, s:]
                cnt[:h - s] += 1.0
        shape = [1, h] + [1] * (ref.dim() - 2)
        return acc / cnt.reshape(shape)

    res = {"forecast_denorm": _aligned_mean("forecast_denorm")}
    fan = _aligned_mean("quantiles_denorm")
    if fan is not None:
        res["quantiles_denorm"] = fan
        levels = next((o["quantile_levels"] for o, _ in outs
                       if "quantile_levels" in o), None)
        if levels is not None:
            mid = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
            res["forecast_denorm"] = fan.select(-2, mid).unsqueeze(-2) \
                if fan.dim() == 4 else fan[..., mid:mid + 1]
            res["quantile_levels"] = levels
    return res


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def apply_quantile_gamma(out: dict, gamma) -> dict:
    """
    G4.2 - uniform conformal calibration: q'_k = med + gamma_k * (q_k - med).

    gamma [Q] comes from scripts/calibrate_quantiles.py (FINETUNE corpus,
    never GIFT - one vector for all 97 configs, single-checkpoint doctrine).
    The median is untouched by construction (gamma_0.5 = 1 and the point
    forecast is unchanged): MASE bit-identical, any delta is CRPS.
    gamma > 0 preserves quantile monotonicity. Applied AFTER the TTA average
    (calibration was done under the same procedure); the G8.4b envelope
    already bounded the quantiles upstream - widening can exceed it slightly
    (gamma ~< 2 against a +-10*range envelope: accepted).
    """
    q = out.get("quantiles_denorm")
    if q is None or gamma is None:
        return out
    med = out["forecast_denorm"]                       # [B, h, 1]
    g = gamma.to(q.device, q.dtype)
    if q.dim() == 4:                                   # [B, h, Q, 1]
        q = med.unsqueeze(-2) + g.view(1, 1, -1, 1) * (q - med.unsqueeze(-2))
    else:                                              # [B, h, Q]
        q = med + g.view(1, 1, -1) * (q - med)
    res = dict(out)
    res["quantiles_denorm"] = q
    return res


def _pinball_np(fan: np.ndarray, y: np.ndarray, levels) -> float:
    """Mean pinball loss (GluonTS x2 convention, no effect on the argmin)."""
    d = y[:, None] - fan
    q = np.asarray(levels, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def _backtest_series_k(model, series, h: int, windows: int, max_len: int,
                       stride: int, patch: int, device,
                       batch_size: int, pooled: bool = False) -> tuple:
    """RateIN v2 (2026-09-01) - per-series k chosen by CAUSAL BACKTEST.

    Oracle verdict 2026-08-31: the mechanism is worth up to +57% per config
    (34/91 > 5%), but the v1 FFT detector misses coarse frequencies (it
    decimates D/W/M configs the oracle leaves at k=1) and the oracle reveals
    a SECOND mechanism the period cannot see (rollout collapse: bizitobs/H
    gains 40% at k=16 on a cycle of 24). The backtest captures both WITHOUT
    metadata: for each series, drop the last h_bt steps of its past (never
    the test - the replayed window precedes the first eval target), forecast
    each candidate k, keep the best pinball. Same logic as in-context
    T-calibration (E18h).

    v2.1 (2026-09-01, v2 run verdict: captures 30% of the oracle):
    * h_bt = REAL h (no more 256 cap) - the cap made rollout collapse
      INVISIBLE to selection: at h_bt<=256, k=1 never needs rollout in the
      backtest, so the biggest oracle gains (bizitobs +40-56% at k=16-24,
      h=480) could not be seen. Fallback: h_bt reduced on short histories.
    * up to 2 averaged windows (variance reduction, x2 cost on a mini-pass);
    * 5% no-op margin (1-SE-rule spirit): pinball on 1-2 windows of 14 steps
      is noisy (m4_daily 3.89 vs flip 3.48 in v2) - k>1 must beat k=1 by at
      least REL_MARGIN; real oracle gains are +20-50%, far above.

    v3 (2026-09-01, v2.1 diagnosis: per-series winner's curse): k PER
    CONFIG, scores pooled across series. The per-series argmin over 11
    candidates with 1-2 windows picks lucky draws (jena/D: 14/42 instances
    at k=16, the WORST k in the oracle table, +163%) and mixed k
    underperforms uniform k on rough landscapes (bitbrains 0.896 with a mix
    vs <=0.837 for the WHOLE oracle column). Pooling: pinball ratio k/k=1
    per series, geomean across series (normalization removes
    scale/difficulty), variance / n_series, and the granularity becomes the
    ORACLE's (per config), which bounds the capture. A k scored on fewer
    than 2/3 of the base series is disqualified (biased subset - typically
    large k on short series). The per-instance guard (decimated history <
    patch -> k=1) stays as the safety net.
    """
    REL_MARGIN = 0.05
    N_BT_WINDOWS = 2
    ks = {}
    entries = []                                    # (idx, sub_hist, known)
    for idx, y in enumerate(series):
        past = y[:len(y) - windows * h]             # before ANY test target
        avail = len(past) - 4 * patch
        h_bt = min(h, avail)
        if h_bt < 16:
            ks[idx] = 1
            continue
        got = 0
        for j in range(1, min(N_BT_WINDOWS, avail // h_bt) + 1):
            lo = len(past) - j * h_bt
            sub_hist, known = past[:lo], past[lo:lo + h_bt]
            if not np.isfinite(known).all():
                continue
            entries.append((idx, sub_hist, known))
            got += 1
        if got == 0:
            ks[idx] = 1

    scores = defaultdict(lambda: defaultdict(list))
    for k in ratein_mod.K_CANDIDATES:
        buckets = defaultdict(list)
        for idx, sub_hist, known in entries:
            hist = (ratein_mod.decimate(sub_hist[-(max_len * k):], k)
                    if k > 1 else sub_hist)
            if len(hist) < patch:
                continue
            ctx = prepare_context(hist, max_len, stride, patch)
            if ctx is None:
                continue
            # h_bt varies per series (short-history fallback) -> the bucket
            # also carries h_fc so each batch stays homogeneous.
            buckets[(len(ctx), -(-len(known) // k))].append((idx, ctx, known))
        for (length, h_fc), items in buckets.items():
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                batch = torch.from_numpy(np.stack([c[1] for c in chunk]))
                batch = batch.unsqueeze(-1).to(device)
                with torch.no_grad():
                    out = model.forecast(batch, n=h_fc)
                q = out.get("quantiles_denorm")
                if q is None:
                    continue
                levels = list(out.get("quantile_levels",
                                      [0.1 * j for j in range(1, 10)]))
                q = q.cpu().numpy()
                if q.ndim == 4:
                    q = q[..., 0]
                for b, (idx, _, known) in enumerate(chunk):
                    fan_nat = ratein_mod.reinterp_fan(q[b], len(known), k)
                    sc = _pinball_np(fan_nat, known, levels)
                    if np.isfinite(sc):
                        scores[idx][k].append(sc)
    ratios, n_base = _pool_ratios(scores, pooled)
    K = 1
    if ratios:
        kbest = min(ratios, key=ratios.get)
        if ratios[kbest] < 1.0 - REL_MARGIN:
            K = kbest
    # Selection diagnostic (2026-09-05): the pooled ratio table is what the
    # selector saw. Cached per config so ratein_selection_gap.py can split
    # the backtest-vs-oracle residual into margin no-ops, wrong k and
    # coverage disqualifications without re-running anything.
    diag = {"K": K, "margin": REL_MARGIN, "n_base": n_base,
            "pooling": "crps" if pooled else "geomean",
            "ratios": {str(k): round(r, 5) for k, r in sorted(ratios.items())}}
    return {idx: K for idx in range(len(series))}, diag


def _pool_ratios(scores: dict, pooled: bool) -> tuple:
    """Per-config ratio table k -> score(k)/score(1) from per-series scores.

    scores[idx][k] = list of window scores of series idx at rate k.
    geomean (v3): equal weight per series. pooled (2026-09-06, "objective
    alignment"): sum over series, i.e. the leaderboard's own weighting - CRPS
    is sum(2 QL) / sum(|y|) over the whole config, so high-amplitude series
    dominate the metric and must dominate the selection too. In both cases
    a k scored on fewer than 2/3 of the base series is disqualified.
    """
    base_scored = [i for i in scores
                   if scores[i].get(1) and np.mean(scores[i][1]) > 0]
    ratios = {}
    for k in ratein_mod.K_CANDIDATES:
        if k == 1:
            continue
        idx = [i for i in base_scored if scores[i].get(k)]
        if not idx or len(idx) < (2 * len(base_scored)) // 3:
            continue                                # biased subset
        num = np.array([np.mean(scores[i][k]) for i in idx])
        den = np.array([np.mean(scores[i][1]) for i in idx])
        if pooled:
            ratios[k] = float(num.sum() / max(den.sum(), 1e-12))
        else:
            ratios[k] = float(np.exp(np.mean(np.log(np.maximum(num / den, 1e-12)))))
    return ratios, len(base_scored)


@torch.no_grad()
def _energy_series_k(judge, series, h: int, windows: int, max_len: int,
                     stride: int, patch: int, device, batch_size: int,
                     pooled: bool = False) -> tuple:
    """RateIN-energy (2026-09-06, user idea): the rate at which the series is
    most PREDICTABLE for the pretrain, no forecast involved.

    For each candidate k the past (before any test target) is decimated, its
    last `n_e` steps (the judge's native target span) play the future and the
    preceding steps the context; the energy is the JEPA readout of the
    hybrid judge (E18/G12 recipe, online encoder both sides): 1 - cos between
    the predictor's latent for the future and the encoder's latent of the
    actual future. The k minimizing the pooled energy ratio vs k=1 wins - no
    margin (argmin), same 2/3 coverage rule as the backtest. Sees cycle
    canonicalization (what the pretrain learned), not rollout collapse.
    Cost: one encoder + predictor pass per (series, k), no rollout.
    """
    n_e = int(judge.prediction_length)
    P = patch
    entries = []
    for idx, y in enumerate(series):
        past = y[:len(y) - windows * h]
        if len(past) < n_e + P:
            continue
        entries.append((idx, past))
    scores = defaultdict(lambda: defaultdict(list))
    n_tgt = (n_e - P) // stride + 1
    w_one = (torch.ones(1, device=device)
             if getattr(judge.predictor, "w_film", None) is not None else None)
    for k in ratein_mod.K_CANDIDATES:
        buckets = defaultdict(list)
        for idx, past in entries:
            hist = (ratein_mod.decimate(past[-((max_len + n_e) * k):], k)
                    if k > 1 else past[-(max_len + n_e):])
            if len(hist) < n_e + P or not np.isfinite(hist).all():
                continue
            ctx = prepare_context(hist[:-n_e], max_len, stride, P)
            if ctx is None:
                continue
            buckets[len(ctx)].append((idx, ctx, hist[-n_e:]))
        for length, items in buckets.items():
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                x_ctx = torch.from_numpy(np.stack([c[1] for c in chunk]))
                x_fut = torch.from_numpy(np.stack([c[2] for c in chunk]))
                x_ctx = x_ctx.unsqueeze(-1).to(device)
                x_fut = x_fut.unsqueeze(-1).to(device)
                if judge.robust_scaler is not None:
                    judge.robust_scaler.fit(x_ctx)
                    x_ctx = judge.robust_scaler.transform(x_ctx)
                    x_fut = judge.robust_scaler.transform(x_fut)
                if judge.revin is not None:
                    ctx_norm = judge.revin(x_ctx, mode='norm')
                    fut_norm = (x_fut - judge.revin.mean) / judge.revin.std
                else:
                    ctx_norm, fut_norm = x_ctx, x_fut
                ctx_emb = judge.online_encoder(judge.patching(ctx_norm))
                z_pred = judge.predictor.forward_simple(
                    context_embeddings=ctx_emb,
                    num_targets=judge.num_target_patches,
                    w=(w_one.expand(ctx_emb.shape[0]) if w_one is not None
                       else None))[:, :n_tgt, :]
                full = torch.cat([ctx_norm, fut_norm], dim=1)
                z_true = judge.online_encoder(judge.patching(full))[:, -n_tgt:, :]
                e = 1.0 - torch.nn.functional.cosine_similarity(
                    z_true.flatten(1), z_pred.flatten(1), dim=1)
                for b, (idx, _, _) in enumerate(chunk):
                    val = float(e[b])
                    if np.isfinite(val):
                        scores[idx][k].append(val)
    ratios, n_base = _pool_ratios(scores, pooled)
    K = 1
    if ratios:
        kbest = min(ratios, key=ratios.get)
        if ratios[kbest] < 1.0:
            K = kbest
    diag = {"K": K, "margin": 0.0, "n_base": n_base,
            "pooling": "crps" if pooled else "geomean", "judge_span": n_e,
            "ratios": {str(k): round(r, 5) for k, r in sorted(ratios.items())}}
    return {idx: K for idx in range(len(series))}, diag


# RateIN-mix (2026-09-05). The selection-gap decomposition on the head8
# champion split the 2.43-pt residual three ways (missed 32%, wrong k 38%,
# false positives 29%), so no margin setting can win: a harder margin trades
# missed for false positives. Mixing addresses all three at once - no
# threshold, no argmin, k=1 keeps weight. The temperature equals the old
# margin: a k that beats k=1 by exactly the margin gets weight e vs 1.
MIX_TAU = 0.05
MIX_MIN_WEIGHT = 0.02
MIX_MAX_COMPONENTS = 4


def _mix_weights(ratios: dict, tau: float = MIX_TAU,
                 min_weight: float = MIX_MIN_WEIGHT,
                 max_components: int = MIX_MAX_COMPONENTS) -> dict:
    """Softmax over -log(pooled pinball ratio)/tau, k=1 included at ratio 1.

    `ratios` is the selector's table ({str(k): ratio}); k's the selector
    disqualified (coverage < 2/3) are absent and get no weight. Components
    below `min_weight` are dropped, the largest `max_components` kept, and
    the result renormalized (deterministic, sorted by k).
    """
    cand = {1: 1.0}
    cand.update({int(k): float(r) for k, r in ratios.items()})
    logits = {k: -math.log(max(r, 1e-6)) / tau for k, r in cand.items()}
    top = max(logits.values())
    w = {k: math.exp(v - top) for k, v in logits.items()}
    z = sum(w.values())
    w = {k: v / z for k, v in w.items() if v / z >= min_weight}
    w = dict(sorted(w.items(), key=lambda kv: -kv[1])[:max_components])
    z = sum(w.values())
    return {k: v / z for k, v in sorted(w.items())}


def evaluate_config(model, config: str, gift_root: Path, device,
                    batch_size: int, max_series: int = 0,
                    max_context: int = 0, tta_lookbacks=None,
                    tta_flip: bool = False, tta_shifts=None,
                    quantile_gamma=None,
                    ratein_mode: str = "off", forced_k: int = 0,
                    ratein_w: bool = False, ratein_w_max_k: int = 4,
                    ratein_pool: bool = False, energy_judge=None) -> dict:
    h = gift.prediction_length(config)
    freq = config.split("/")[1]
    m = gift.seasonality(freq)

    series = gift.load_series(gift_root, config)
    if max_series:
        series = series[:max_series]
    windows = gift.num_windows(config, min(len(s) for s in series))

    model_acc = gift.MetricAccumulator()
    sn_acc = gift.MetricAccumulator()

    # Bucket instances by prepared-context length so each forward pass gets a
    # rectangular batch. Within a config almost everything lands in one bucket
    # (most series are longer than the 1024-step cap).
    buckets = defaultdict(list)
    # G9.1 - max_context may EXCEED model.input_length: the encoder is RoPE
    # (length-agnostic), the predictor only depends on the number of target
    # queries. This is the "2048 context at inference only" test.
    max_len = max_context or (model.input_length
                              if hasattr(model, "input_length") else 1024)
    stride = model.patching.stride
    n_inst = 0
    k_hist = Counter()
    # RateIN v2 - per-series k chosen by causal backtest (see the helper);
    # computed once before the loop, batched.
    bt_ks, bt_diag, mix_weights = None, None, None
    if ratein_mode in ("backtest", "mix") and not forced_k:
        bt_ks, bt_diag = _backtest_series_k(model, series, h, windows, max_len,
                                            stride, model.patching.patch_size,
                                            device, batch_size,
                                            pooled=ratein_pool)
        if ratein_mode == "mix":
            # The hard per-series choice is replaced by per-config weights.
            mix_weights, bt_ks = _mix_weights(bt_diag["ratios"]), None
    if ratein_mode == "energy" and not forced_k:
        bt_ks, bt_diag = _energy_series_k(energy_judge, series, h, windows,
                                          max_len, stride,
                                          model.patching.patch_size, device,
                                          batch_size, pooled=ratein_pool)
    # RateIN-mix state: per instance, the weight-summed native fan/median of
    # the components that survived the guards (finalized after the loop).
    mix_state, mix_mid = {}, 0
    for inst in gift.iter_test_instances(series, h, windows):
        if mix_weights is not None:
            comps = []
            for kk, wk in mix_weights.items():
                hist = (ratein_mod.decimate(inst.context[-(max_len * kk):], kk)
                        if kk > 1 else inst.context)
                if kk > 1 and len(hist) < model.patching.patch_size:
                    continue                    # same guard as the hard path
                ctx = prepare_context(hist, max_len, stride,
                                      model.patching.patch_size)
                if ctx is None:
                    continue
                comps.append((kk, wk, ctx))
            if not comps:
                continue
            scale = gift.seasonal_error(inst.context, m)
            iid = n_inst
            mix_state[iid] = {"target": inst.target, "past": inst.context,
                              "scale": scale, "fan": None, "med": None, "w": 0.0}
            for kk, wk, ctx in comps:
                buckets[(len(ctx), kk)].append(
                    (ctx, inst.target, inst.context, scale, iid, wk))
            k_hist[max(comps, key=lambda c: c[1])[0]] += 1
            n_inst += 1
            continue
        # RateIN (2026-08-31, G9.3 verdict): k is a causal statistic of the
        # past, one uniform rule across the 97 configs (same status as
        # median/MAD). forced_k = ORACLE mode (diagnostic, never official).
        if forced_k:
            k = forced_k
        elif bt_ks is not None:
            k = bt_ks.get(inst.series_idx, 1)
        elif ratein_mode == "fft":
            k = ratein_mod.choose_k(ratein_mod.detect_period(inst.context))
        else:
            k = 1
        hist = (ratein_mod.decimate(inst.context[-(max_len * k):], k)
                if k > 1 else inst.context)
        if k > 1 and len(hist) < model.patching.patch_size:
            # Guard (oracle crash 2026-08-31, IndexError on 6 configs): a
            # short history decimated by a large k goes empty - fall back k=1.
            k, hist = 1, inst.context
        ctx = prepare_context(hist, max_len, stride, model.patching.patch_size)
        if ctx is None:
            continue
        # MASE scale uses the FULL past, not the capped context - gluonts
        # computes the seasonal error on the entire history of the series.
        scale = gift.seasonal_error(inst.context, m)
        buckets[(len(ctx), k)].append((ctx, inst.target, inst.context, scale))
        k_hist[k] += 1
        n_inst += 1

    for (length, k), items in sorted(buckets.items()):
        # RateIN x w (G9.3 synergy, gated flag): context decimated by k, fan
        # requested DIRECTLY at native rate via w = 1/k - zero
        # re-interpolation (the one loss even the oracle cannot avoid).
        # Extrapolation guard: the FiLM only saw log2(w) within the training
        # factor range ([1,2,4] -> [-2,2]); beyond ratein_w_max_k, fall back
        # to the standard decimate+reinterp path.
        use_w = ratein_w and 1 < k <= ratein_w_max_k
        # Decimated grid: h' = ceil(h/k) steps cover the native horizon
        # (measurable bonus: fewer rollouts on long-term 10S/5T).
        h_fc = h if use_w else -(-h // k)
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            batch = torch.from_numpy(np.stack([c[0] for c in chunk]))
            batch = batch.unsqueeze(-1).to(device)          # [B, L, 1]

            w_vec = (torch.full((batch.shape[0],), 1.0 / k, device=device)
                     if use_w else None)
            with torch.no_grad():
                out = tta_forecast(model, batch, h_fc,
                                   lookbacks=tta_lookbacks, flip=tta_flip,
                                   shifts=tta_shifts, w=w_vec)
            out = apply_quantile_gamma(out, quantile_gamma)

            median = out["forecast_denorm"].squeeze(-1).cpu().numpy()  # [B, h']
            quants = out.get("quantiles_denorm")
            if quants is not None:
                quants = quants.cpu().numpy()
                if quants.ndim == 4:                        # [B, h', Q, 1]
                    quants = quants[..., 0]
                # -> [B, h', Q]
            levels = out.get("quantile_levels")
            mid = (min(range(len(levels)), key=lambda j: abs(levels[j] - 0.5))
                   if levels is not None and quants is not None
                   else (quants.shape[-1] // 2 if quants is not None else 0))

            if quants is not None:
                mix_mid = mid
            for b, item in enumerate(chunk):
                ctx, target, past, scale = item[:4]
                if k > 1 and not use_w:
                    if quants is not None:
                        fan_nat = ratein_mod.reinterp_fan(quants[b], h, k)
                        med_nat = fan_nat[:, mid]
                    else:
                        fan_nat = None
                        med_nat = ratein_mod.reinterp_fan(
                            median[b][:, None], h, k)[:, 0]
                else:
                    fan_nat = quants[b] if quants is not None else None
                    med_nat = median[b]
                if len(item) == 6:                      # RateIN-mix component
                    st = mix_state[item[4]]
                    wk = item[5]
                    if fan_nat is not None:
                        st["fan"] = (wk * fan_nat if st["fan"] is None
                                     else st["fan"] + wk * fan_nat)
                    st["med"] = (wk * med_nat if st["med"] is None
                                 else st["med"] + wk * med_nat)
                    st["w"] += wk
                    continue
                model_acc.add(target, med_nat, fan_nat, scale)
                sn = gift.seasonal_naive_forecast(past, h, m)
                sn_acc.add(target, sn, None, scale)

    # RateIN-mix: quantile averaging (Vincentization) of the surviving
    # components - a convex combination of monotone fans stays monotone and
    # keeps sharpness, unlike a mixture of distributions.
    for st in mix_state.values():
        if st["w"] <= 0:
            continue
        fan = None if st["fan"] is None else st["fan"] / st["w"]
        med = fan[:, mix_mid] if fan is not None else st["med"] / st["w"]
        model_acc.add(st["target"], med, fan, st["scale"])
        sn_acc.add(st["target"], gift.seasonal_naive_forecast(st["past"], h, m),
                   None, st["scale"])

    res = {"config": config, "prediction_length": h, "seasonality": m,
           "windows": windows, "n_series": len(series), "n_instances": n_inst,
           "model": model_acc.result(), "seasonal_naive_local": sn_acc.result()}
    if ratein_mode != "off" or forced_k:
        res["ratein"] = {
            "k_hist": {str(kk): n for kk, n in sorted(k_hist.items())},
            "frac_k_gt1": (sum(n for kk, n in k_hist.items() if kk > 1)
                           / max(1, sum(k_hist.values()))),
        }
        if bt_diag is not None:
            res["ratein"]["energy" if ratein_mode == "energy" else "backtest"] = bt_diag
        if mix_weights is not None:
            res["ratein"]["mix"] = {
                "tau": MIX_TAU,
                "weights": {str(k): round(w, 4) for k, w in mix_weights.items()}}
    return res


# ---------------------------------------------------------------------------
# Official CSV row
# ---------------------------------------------------------------------------

CSV_HEADER = ("dataset,model,eval_metrics/MSE[mean],eval_metrics/MSE[0.5],"
              "eval_metrics/MAE[0.5],eval_metrics/MASE[0.5],eval_metrics/MAPE[0.5],"
              "eval_metrics/sMAPE[0.5],eval_metrics/MSIS,eval_metrics/RMSE[mean],"
              "eval_metrics/NRMSE[mean],eval_metrics/ND[0.5],"
              "eval_metrics/mean_weighted_sum_quantile_loss,domain,num_variates")


def csv_row(config: str, model_name: str, r: dict) -> str:
    m = r["model"]
    # The point forecast is the median, so MSE[mean] == MSE[0.5] here; models
    # with a distinct mean head fill them differently, ours does not have one.
    return (f"{config},{model_name},{m['MSE']},{m['MSE']},{m['MAE']},"
            f"{m['MASE']},{m['MAPE']},{m['sMAPE']},nan,{m['RMSE']},"
            f"{m['NRMSE']},{m['ND']},{m['CRPS']},,")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("pass +checkpoint_path=<...>")

    gift_root = Path(cfg.get("gift_data_dir", "data/gift_eval"))
    terms = str(cfg.get("gift_terms", "")).split(",") if cfg.get("gift_terms") else []
    only = ([c.strip() for c in str(cfg.gift_configs).split(",")]
            if cfg.get("gift_configs") else [])
    batch_size = int(cfg.get("gift_batch_size", 64))
    max_series = int(cfg.get("gift_max_series", 0))

    # TTA / long context (roadmap S1) - opt-in, defaults = exact previous
    # behavior. Examples:
    #   +max_context=2048                       (G9.1, inference only)
    #   +tta_lookbacks='512,1024'               (averaged multi-lookback)
    #   +tta_flip=true                          (sign-flip mirror)
    max_context = int(cfg.get("max_context", 0))
    #   +ratein=fft   (alias true)             period detector v1
    #   +ratein=backtest                        per-series k, causal backtest (v2)
    #   +ratein=oracle                          k sweep - DIAGNOSTIC, looks at
    #                                           the test set, never official
    ratein_raw = str(cfg.get("ratein", "")).lower()
    #   +ratein=mix                             per-config weighted mix of k
    #                                           (quantile averaging, causal)
    #   +ratein=energy +energy_ckpt=<pretrain>  k = argmin of the pretrain's
    #                                           JEPA energy on the decimated
    #                                           past (no forecast); optional
    #                                           +energy_config=<eval config>
    #   +ratein_pool=true                       ratio table pooled like the
    #                                           CRPS (sum over series) instead
    #                                           of a per-series geomean
    ratein_mode_val = {"true": "fft", "1": "fft", "on": "fft", "fft": "fft",
                       "backtest": "backtest", "bt": "backtest",
                       "mix": "mix", "energy": "energy"}.get(ratein_raw, "off")
    if ratein_raw and ratein_raw not in ("off", "false", "0", "oracle") \
            and ratein_mode_val == "off":
        # An unknown mode must not fall back to "off": it would land in the
        # plain cache directory and silently re-read another procedure's
        # numbers (seen 2026-09-06 with a stale checkout).
        raise ValueError(f"unknown +ratein={ratein_raw!r} (fft, backtest, mix, "
                         "energy, oracle)")
    ratein_on = ratein_mode_val != "off"
    ratein_oracle = ratein_raw == "oracle"
    ratein_pool = str(cfg.get("ratein_pool", "")).lower() in ("true", "1", "on")
    if ratein_pool and ratein_mode_val not in ("backtest", "mix", "energy"):
        raise ValueError("+ratein_pool needs +ratein=backtest/mix/energy")
    if ratein_mode_val == "energy" and not cfg.get("energy_ckpt"):
        raise ValueError("+ratein=energy needs +energy_ckpt=<pretrain checkpoint>")
    # RateIN x w (gated): +ratein_w=true - fan at native rate via w=1/k on
    # decimated buckets (k <= 4). Requires a cross_resolution model AND an
    # active ratein mode (without decimation, w=1 everywhere = misleading no-op).
    ratein_w = str(cfg.get("ratein_w", "")).lower() in ("true", "1", "on")
    if ratein_w and not (ratein_on or ratein_oracle):
        raise ValueError("+ratein_w requires +ratein=backtest/fft/oracle "
                         "(without decimation, w would be 1 everywhere)")
    tta_lookbacks = ([int(x) for x in str(cfg.tta_lookbacks).split(",")]
                     if cfg.get("tta_lookbacks") else None)
    tta_flip = bool(cfg.get("tta_flip", False))
    #   +tta_shifts='2,4,6'  (translation: origins t-s realigned, s < stride
    #                         recommended - less than one patch of staleness)
    tta_shifts = ([int(x) for x in str(cfg.tta_shifts).split(",")]
                  if cfg.get("tta_shifts") else None)
    #   +quantile_gamma='evaluation/calibration/gamma_<ckpt>.json'
    #   (G4.2: per-level widening factors, calibrated on the FINETUNE corpus
    #    by scripts/calibrate_quantiles.py - status: paper ablation, user
    #    decision 2026-08-25, not necessarily the official number)
    quantile_gamma, gamma_tag = None, ""
    if cfg.get("quantile_gamma"):
        gpath = Path(str(cfg.quantile_gamma))
        gdata = json.loads(gpath.read_text())
        quantile_gamma = torch.tensor(gdata["gamma"], dtype=torch.float32)
        gamma_tag = "_gamma-" + gpath.stem
        logger.info(f"quantile_gamma: {gdata['gamma']} ({gpath})")

    configs = [c for c in gift.GIFT_CONFIGS
               if (not only or c in only)
               and (not terms or c.split("/")[2] in terms)]
    logger.info(f"{len(configs)} configs to evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, checkpoint_path, device)

    # Each inference variant gets its OWN directory: the per_config JSONs act
    # as resume markers, and mixing two procedures in one cache would re-read
    # one procedure's numbers as if they measured the other.
    tag = ""
    if max_context:
        tag += f"_ctx{max_context}"
    if tta_lookbacks:
        tag += "_lb" + "-".join(str(x) for x in tta_lookbacks)
    if tta_flip:
        tag += "_flip"
    if ratein_w and getattr(model.predictor, "w_film", None) is None:
        raise ValueError("+ratein_w requires a cross_resolution model "
                         "(no w FiLM in the predictor - xres config)")
    if ratein_mode_val == "fft":
        tag += "_ratein"
    elif ratein_mode_val == "backtest":
        tag += "_ratein-bt"
    elif ratein_mode_val == "mix":
        tag += "_ratein-mix"
    elif ratein_mode_val == "energy":
        tag += "_ratein-energy"
    elif ratein_oracle:
        tag += "_ratein-oracle"
    if ratein_pool:
        tag += "-pool"
    energy_judge = None
    if ratein_mode_val == "energy":
        # The judge is the PRETRAIN checkpoint (its predictor never saw a
        # forecasting loss); a finetuned model's own trunk would do, but the
        # pretrain is the hypothesis: "the rate the pretrain finds natural".
        from hydra import compose as _compose
        ecfg = (_compose(config_name=str(cfg.energy_config))
                if cfg.get("energy_config") else cfg)
        energy_judge = create_model_from_config(ecfg)
        energy_judge = load_checkpoint(energy_judge, str(cfg.energy_ckpt), device,
                                       allow_partial=True).eval()
        logger.info(f"energy judge: {cfg.energy_ckpt}")
    if ratein_w:
        tag += "-w"
    if tta_shifts:
        tag += "_sh" + "-".join(str(x) for x in tta_shifts)
    tag += gamma_tag
    out_dir = (Path("evaluation") / cfg.model.name
               / Path(checkpoint_path).stem / f"gift{tag}")
    per_config = out_dir / "per_config"
    per_config.mkdir(parents=True, exist_ok=True)

    # Measured resume trap (2026-08-20): the directory is keyed on the
    # checkpoint NAME, and `last.ckpt` gets overwritten during a run - a
    # re-eval of `last` re-read the old checkpoint's JSONs while printing
    # "already done" on 97 configs, i.e. an aggregate of the OLD model
    # presented as a measure of the new one. The fingerprint (mtime+size)
    # makes the case loud. Refuse rather than clean: nothing is ever deleted here.
    ckpt_stat = Path(checkpoint_path).stat()
    fingerprint = f"{ckpt_stat.st_mtime_ns}:{ckpt_stat.st_size}"
    fp_file = out_dir / "checkpoint_fingerprint.txt"
    if fp_file.exists() and fp_file.read_text().strip() != fingerprint:
        raise RuntimeError(
            f"{out_dir} holds results from a DIFFERENT version of "
            f"{Path(checkpoint_path).name} (fingerprint mismatch - the file was "
            f"overwritten since, typically a last.ckpt from a running job). "
            f"Resuming would re-read the old JSONs as if they measured the new "
            f"checkpoint. Move or rename this directory, then relaunch."
        )
    fp_file.write_text(fingerprint)

    results = {}
    for i, config in enumerate(configs, 1):
        marker = per_config / (config.replace("/", "__") + ".json")
        if marker.exists():
            results[config] = json.loads(marker.read_text())
            logger.info(f"[{i}/{len(configs)}] {config}: already done, skipped")
            continue
        t0 = time.time()
        try:
            if ratein_oracle:
                # Per-config k sweep - the upper bound on the gain reachable
                # by rate canonicalization. Settles the P-RIN diagnostic
                # failure: if even the oracle gains nowhere, scale geometry
                # is not the mechanism.
                per_k, best = {}, None
                for kk in ratein_mod.K_CANDIDATES:
                    r_k = evaluate_config(model, config, gift_root, device,
                                          batch_size, max_series,
                                          max_context=max_context,
                                          tta_lookbacks=tta_lookbacks,
                                          tta_flip=tta_flip,
                                          tta_shifts=tta_shifts,
                                          quantile_gamma=quantile_gamma,
                                          forced_k=kk, ratein_w=ratein_w)
                    per_k[str(kk)] = r_k["model"]["CRPS"]
                    if best is None or r_k["model"]["CRPS"] < best["model"]["CRPS"]:
                        best, best_k = r_k, kk
                res = best
                res["oracle"] = {"per_k_crps": per_k, "best_k": best_k,
                                 "gain_vs_k1": 1.0 - per_k[str(best_k)]
                                 / per_k["1"] if per_k["1"] > 0 else 0.0}
            else:
                res = evaluate_config(model, config, gift_root, device,
                                      batch_size, max_series,
                                      max_context=max_context,
                                      tta_lookbacks=tta_lookbacks,
                                      tta_flip=tta_flip,
                                      tta_shifts=tta_shifts,
                                      quantile_gamma=quantile_gamma,
                                      ratein_mode=ratein_mode_val,
                                      ratein_w=ratein_w,
                                      ratein_pool=ratein_pool,
                                      energy_judge=energy_judge)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return
        except Exception as exc:  # one broken config must not kill 96 others
            logger.error(f"[{i}/{len(configs)}] {config} FAILED: "
                         f"{type(exc).__name__}: {exc}")
            continue
        marker.write_text(json.dumps(res, indent=1))
        results[config] = res
        mm = res["model"]
        extra = ""
        if "ratein" in res and not ratein_oracle:
            extra = f" k>1: {res['ratein']['frac_k_gt1']:.0%}"
            if "mix" in res["ratein"]:
                extra += " mix " + " ".join(
                    f"k{k}:{w:.2f}" for k, w in res["ratein"]["mix"]["weights"].items())
        if "oracle" in res:
            extra = (f" best_k={res['oracle']['best_k']} "
                     f"(gain {res['oracle']['gain_vs_k1']:+.1%} vs k=1)")
        logger.info(
            f"[{i}/{len(configs)}] {config}: MASE {mm['MASE']:.3f} "
            f"CRPS {mm['CRPS']:.3f} ({res['n_instances']} inst, "
            f"{time.time() - t0:.0f}s){extra}")

    # ---- official-format CSV + leaderboard aggregates ----
    rows = [CSV_HEADER] + [csv_row(c, cfg.model.name, results[c])
                           for c in gift.GIFT_CONFIGS if c in results]
    (out_dir / "all_results.csv").write_text("\n".join(rows) + "\n")

    model_metrics = {c: results[c]["model"] for c in results}
    local_sn = {c: results[c]["seasonal_naive_local"] for c in results}
    summary = {
        "checkpoint": str(checkpoint_path),
        "n_configs": len(results),
        "vs_official_seasonal_naive":
            gift.aggregate(model_metrics, gift.official_seasonal_naive()),
        "vs_local_seasonal_naive":
            gift.aggregate(model_metrics, local_sn),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"\n{'=' * 60}\nLEADERBOARD AGGREGATES ({len(results)} configs)")
    for key in ("vs_official_seasonal_naive", "vs_local_seasonal_naive"):
        a = summary[key]
        logger.info(f"  {key}: MASE ratio {a['geomean_MASE_ratio']:.4f} | "
                    f"CRPS ratio {a['geomean_CRPS_ratio']:.4f}")
    # Mean empirical coverage (E21 instrument: nominal q10=0.10, q90=0.90
    # => 80% interval; the measured zero-shot under-coverage is shift, G4.2 -
    # the legitimate lever is the ESJEPA z gate).
    covs = [r["model"].get("coverage") for r in results.values()
            if r["model"].get("coverage")]
    if covs:
        c10 = float(np.mean([c["0.1"] for c in covs]))
        c90 = float(np.mean([c["0.9"] for c in covs]))
        logger.info(f"  coverage (mean over {len(covs)} configs): q10 {c10:.3f} (nominal 0.100) | "
                    f"q90 {c90:.3f} (nominal 0.900) | 80% interval -> {c90 - c10:.3f}")
    if ratein_on:
        fr = [r["ratein"]["frac_k_gt1"] for r in results.values()
              if "ratein" in r]
        n_active = sum(1 for f in fr if f > 0.5)
        logger.info(f"  RateIN: {n_active}/{len(fr)} configs mostly "
                    f"decimated | share of k>1 instances (mean): "
                    f"{float(np.mean(fr)):.1%}")
    if ratein_oracle:
        # THE reading of the P-RIN diagnostic failure: how many configs gain
        # > 5% even with the best k chosen by cheating.
        gains = {c: r["oracle"]["gain_vs_k1"] for c, r in results.items()
                 if "oracle" in r}
        big = sorted(((g, c) for c, g in gains.items() if g > 0.05),
                     reverse=True)
        logger.info(f"  ORACLE-k (diagnostic, never official): "
                    f"{len(big)}/{len(gains)} configs gain > 5% vs k=1")
        for g, c in big[:8]:
            logger.info(f"    {c}: {g:+.1%} (best_k="
                        f"{results[c]['oracle']['best_k']})")
        if not big:
            logger.info("    NONE -> P-RIN diagnostic failure: scale "
                        "geometry is not the tail mechanism; G9.3/xres "
                        "not funded.")
    logger.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
