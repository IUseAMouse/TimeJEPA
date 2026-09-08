"""
BiasIN (2026-09-08): causal level-bias correction of the forecast center.

Where it comes from. The refinement ceiling (a per-step translation of the
fan's center toward the TRUE target) is worth 17 CRPS points at a 0.4 sigma
box, but two trained judges (S6 critic, S6-b score matching) could not read
the direction of the fan's error from the context: a forecast is consistent
with its context by construction, its error is what the context does not
determine. And the ceiling itself mixed two things - a systematic bias and
the realized noise no one can predict.

What survives is the part of the bias that PERSISTS from one window to the
next. That part is measurable causally: on the backtest windows already used
by RateIN (the last h steps of the past, strictly before any test target),
forecast, take the residual of the median, express it in units of the local
scale of the context, and shift the test fan by a SHRUNK fraction of it. The
shrinkage is chosen by the backtest too: the bias estimated on the older
window is applied to the forecast of the recent window, and the shrink that
lowers the pooled pinball is kept (5 percent margin, like the k selector);
nothing lowers it => no-op. Same doctrine as RateIN: a causal statistic of
the past, one uniform rule, per-config decision, model-agnostic.

Everything here is numpy and free of the model; the harness feeds it fans,
targets and contexts.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_LAMBDAS = (0.25, 0.5, 1.0)
REL_MARGIN = 0.05
SCALE_TAIL = 256


def level_scale(context: np.ndarray, tail: int = SCALE_TAIL, floor: float = 1e-8) -> float:
    """Robust scale of the recent context (MAD x 1.4826 on the last `tail`
    points, std fallback), the unit in which a level bias is expressed so it
    transfers from the backtest window to the test window."""
    x = np.asarray(context, dtype=np.float64)[-tail:]
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad <= floor:
        sd = float(x.std())
        return max(sd, floor) if sd > floor else 1.0
    return float(mad)


def level_bias(known: np.ndarray, median: np.ndarray, scale: float) -> float:
    """beta = MEDIAN(known - median) / scale over the finite steps; nan when
    nothing is finite. Median, not mean: MASE and the pinball are L1 losses,
    whose optimal constant shift is the median of the residual; on heavy
    tails (bitbrains) the mean is dragged by spikes and the shift lands
    anywhere (2026-09-08, the first oracle run degraded bitbrains by 50%)."""
    d = np.asarray(known, dtype=np.float64) - np.asarray(median, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.median(d) / max(scale, 1e-12))


def shift_fan(fan: Optional[np.ndarray], median: np.ndarray, shift: float):
    """Translate the whole fan (every quantile) and the median by `shift`
    (native units): the fan's shape and monotonicity are untouched."""
    med = np.asarray(median, dtype=np.float64) + shift
    if fan is None:
        return None, med
    return np.asarray(fan, dtype=np.float64) + shift, med


def pinball(fan: np.ndarray, y: np.ndarray, levels: Sequence[float]) -> float:
    """Mean pinball (x2 convention is irrelevant to ratios); NaN targets masked."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan")
    d = y[ok, None] - np.asarray(fan, dtype=np.float64)[ok]
    q = np.asarray(levels, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def choose_shrink(records: List[dict], levels: Sequence[float],
                  lambdas: Sequence[float] = DEFAULT_LAMBDAS,
                  margin: float = REL_MARGIN) -> dict:
    """Per-config causal validation of the correction.

    records: one per series with two backtest windows,
        {"fan": fan of the RECENT window [h, Q], "known": its truth [h],
         "beta_old": bias estimated on the OLDER window, "scale": scale of the
         recent window's context}.
    For each lambda the recent-window fan is shifted by lambda * beta_old *
    scale and the pinball is POOLED over series (sum, the leaderboard's own
    weighting); ratio to the unshifted pooled pinball. The best lambda is
    kept if its ratio is below 1 - margin, else 0 (no-op).
    Returns {"lambda", "ratios", "n_val"}.
    """
    valid = [r for r in records if np.isfinite(r["beta_old"]) and r["fan"] is not None]
    if not valid:
        return {"lambda": 0.0, "ratios": {}, "n_val": 0}
    base = sum(pinball(r["fan"], r["known"], levels) for r in valid)
    ratios = {}
    for lam in lambdas:
        tot = 0.0
        for r in valid:
            fan_s, _ = shift_fan(r["fan"], r["fan"][:, 0], lam * r["beta_old"] * r["scale"])
            tot += pinball(fan_s, r["known"], levels)
        ratios[float(lam)] = float(tot / max(base, 1e-12))
    best = min(ratios, key=ratios.get)
    lam_star = best if ratios[best] < 1.0 - margin else 0.0
    return {"lambda": float(lam_star), "ratios": {f"{k:g}": round(v, 5) for k, v in ratios.items()},
            "n_val": len(valid)}


def series_beta(betas: Sequence[float]) -> float:
    """The bias carried to the test window: mean of the finite window
    estimates (recent and older), nan if none."""
    b = np.asarray([x for x in betas if np.isfinite(x)], dtype=np.float64)
    return float(b.mean()) if b.size else float("nan")


def test_shift(beta: float, lam: float, context: np.ndarray) -> float:
    """Native-unit shift for a test instance: lambda * beta * scale(context)."""
    if not np.isfinite(beta) or lam == 0.0:
        return 0.0
    return float(lam * beta * level_scale(context))


def oracle_shift(target: np.ndarray, median: np.ndarray) -> float:
    """DIAGNOSTIC (never official): the constant per-instance level shift
    that a target-aware oracle would apply - bounds the SYSTEMATIC level
    bias, unlike the per-step ceiling which also fits the realized noise.
    The median of the residual (L1-optimal), never the mean."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(median, dtype=np.float64)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 0.0      # L1-optimal constant
