"""
Diagnostic: is the ETTm failure about patch size, or about frequency coverage?

Hypothesis A (patch size)
    patch=16 / stride=8 cannot resolve a 96-step daily cycle, so the model
    fails on 15-minute data by construction.

Hypothesis B (frequency coverage)
    The pretrain corpus is overwhelmingly hourly / half-hourly and contains NO
    15-minute data. In patch-position units (stride 8) a daily cycle is:
        hourly       24/8  =  3 positions
        half-hourly  48/8  =  6 positions
        10-minute   144/8  = 18 positions   <- solar-10-minute IS in pretrain
        15-minute    96/8  = 12 positions   <- never seen
    The model would then fail on ETTm not because 16/8 is too coarse, but
    because it has never seen a daily cycle at that scale.

The two hypotheses make opposite predictions under resampling:

    A predicts: downsampling ETTm1 by 4 (-> hourly, cycle = 24 steps = 3
       positions) does NOT help, because the patch is now *coarser* relative
       to the signal, which is what A blames.
    B predicts: downsampling ETTm1 by 4 moves it onto the pretrain frequency
       manifold and skill should jump.

Control, in the other direction: upsampling electricity (hourly, a dataset the
model wins on) by 4 puts its daily cycle at 96 steps = 12 positions, the exact
ETTm regime. B predicts skill collapses; A predicts it improves (finer patches
relative to the cycle).

Usage:
    python scripts/diagnose_ettm.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from timejepa.training.utils.baselines import seasonal_naive_forecast   # noqa: E402
from timejepa.training.utils.metrics import mase                        # noqa: E402
from reevaluate_checkpoints import build_model, load_weights, discover_pairs  # noqa: E402

CTX = 384
HORIZON = 96
MAX_WINDOWS = 800


def make_windows(series_2d, ctx, horizon, stride):
    """series_2d: (n_series, T) -> (N, ctx), (N, horizon)"""
    ctxs, tgts = [], []
    n_series, T = series_2d.shape
    need = ctx + horizon
    for s in range(n_series):
        for start in range(0, T - need + 1, stride):
            ctxs.append(series_2d[s, start:start + ctx])
            tgts.append(series_2d[s, start + ctx:start + need])
            if len(ctxs) >= MAX_WINDOWS:
                break
        if len(ctxs) >= MAX_WINDOWS:
            break
    return (
        torch.from_numpy(np.asarray(ctxs)).float(),
        torch.from_numpy(np.asarray(tgts)).float(),
    )


def resample(series_2d: np.ndarray, factor: int) -> np.ndarray:
    """factor>1 downsamples (decimate); factor<-1 upsamples by linear interp."""
    if factor == 1:
        return series_2d
    if factor > 1:
        return series_2d[:, ::factor]
    up = -factor
    t = torch.from_numpy(series_2d).float().unsqueeze(1)      # (S,1,T)
    out = torch.nn.functional.interpolate(
        up_ := t, size=series_2d.shape[1] * up, mode="linear", align_corners=True
    )
    return out.squeeze(1).numpy()


@torch.no_grad()
def score(model, ctxs, tgts, season):
    preds = []
    for i in range(0, len(ctxs), 128):
        c = ctxs[i:i + 128].unsqueeze(-1)
        preds.append(model.forecast(c, n=HORIZON)["forecast_denorm"].squeeze(-1))
    pred = torch.cat(preds)
    sn = seasonal_naive_forecast(ctxs, HORIZON, season)
    m_model = mase(pred, tgts, ctxs, season).item()
    m_sn = mase(sn, tgts, ctxs, season).item()
    return m_model, m_sn, 1.0 - m_model / m_sn


def main():
    device = torch.device("cpu")
    pairs = [p for p in discover_pairs()
             if p["ckpt_name"] == "best-unfreeze-1-stride-48-full-datasets"]
    if not pairs:
        print("checkpoint not found")
        return
    cfg = OmegaConf.load(pairs[0]["cfg_path"])
    model = build_model(cfg, device)
    load_weights(model, pairs[0]["ckpt_path"], device)
    print(f"model ctx={cfg.model.seq_length} native_h={cfg.model.prediction_length}\n")

    nix = REPO_ROOT / "data" / "processed" / "nixtla"

    # (dataset, resample factor, resulting seasonal period, label)
    cases = [
        ("ettm1", 1, 96, "ETTm1 native 15min      cycle=96  = 12 patch-pos"),
        ("ettm1", 4, 24, "ETTm1 downsampled x4    cycle=24  =  3 patch-pos"),
        ("ettm1", 2, 48, "ETTm1 downsampled x2    cycle=48  =  6 patch-pos"),
        ("electricity", 1, 24, "ECL native hourly       cycle=24  =  3 patch-pos"),
        ("electricity", -4, 96, "ECL upsampled x4        cycle=96  = 12 patch-pos"),
    ]

    print(f"{'case':<48} {'MASE':>7} {'SN':>7} {'skill':>8}")
    print("-" * 74)
    for ds, factor, season, label in cases:
        path = nix / f"nixtla_{ds}_test.npy"
        if not path.exists():
            print(f"{label:<48}  missing {path.name}")
            continue
        data = np.load(path).astype(np.float32)
        data = resample(data, factor)
        if data.shape[1] < CTX + HORIZON:
            print(f"{label:<48}  too short after resample")
            continue
        ctxs, tgts = make_windows(data, CTX, HORIZON, stride=HORIZON)
        m, sn, skill = score(model, ctxs, tgts, season)
        print(f"{label:<48} {m:7.3f} {sn:7.3f} {skill:+8.1%}   (n={len(ctxs)})")

    # -----------------------------------------------------------------
    # Resampling changes TWO things at once: patch-positions per cycle AND
    # the number of full cycles that fit in the context. Vary the context
    # length instead to separate them: patch-positions per cycle stays at 12,
    # only the cycle count moves. The encoder is length-agnostic (RoPE, no
    # learned positional table), so it accepts any context.
    # -----------------------------------------------------------------
    print("\n\nDisentangling: hold patch-positions/cycle FIXED, vary cycles in context")
    print(f"{'case':<48} {'MASE':>7} {'SN':>7} {'skill':>8}")
    print("-" * 74)

    for ds, season, ctx_lens in [("ettm1", 96, [384, 768, 1536]),
                                 ("electricity", 24, [96, 192, 384, 768])]:
        path = nix / f"nixtla_{ds}_test.npy"
        if not path.exists():
            continue
        data = np.load(path).astype(np.float32)
        for ctx_len in ctx_lens:
            if data.shape[1] < ctx_len + HORIZON:
                continue
            ctxs, tgts = make_windows(data, ctx_len, HORIZON, stride=HORIZON)
            m, sn, skill = score(model, ctxs, tgts, season)
            cycles = ctx_len / season
            label = f"{ds} ctx={ctx_len:<5} cycle={season:<4} -> {cycles:4.1f} cycles in ctx"
            print(f"{label:<48} {m:7.3f} {sn:7.3f} {skill:+8.1%}   (n={len(ctxs)})")

    print("\nReading:")
    print("  Skill rising with context length, at constant patch-positions/cycle,")
    print("  means the driver is HOW MANY CYCLES FIT IN THE CONTEXT — not patch size,")
    print("  and not which frequencies were in the pretrain corpus.")


if __name__ == "__main__":
    main()
