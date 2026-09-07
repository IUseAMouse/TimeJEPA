"""
Energy sensitivity to a LEVEL SHIFT of the candidate (2026-09-08).

The refinement translates the fan's center by a few hundredths of a sigma
and asks the judge to point the way. The ceiling proved that translation
alone is worth 17 points; the trained critic moved the pinball by 0.1
percent. Hypothesis: the standalone candidate encoder (patch projection,
then LayerNorm blocks) is close to invariant to a constant offset, so the
judge is blind to exactly the move it must guide. E18b ranked candidates of
different SHAPES, never candidates shifted by 0.1 sigma.

For each instance: the model's own fan, then the center shifted by c
(normalized units) for c on a grid; E(c) = 1 - cos(z_pred, enc(center + c))
under the STANDALONE encoding (the critic arm's default) and under the
CONTEXTUALIZED encoding ([ctx || candidate], the junction makes an offset
visible). Also the "oracle direction": the sign of c that lowers E vs the
sign of the true residual mean(y - center).

Reports, per encoding: the mean |E(c) - E(0)| per c (flat = blind), the
share of instances where argmin_c E(c) has the sign of the true residual
(0.5 = coin flip), and the Spearman between E(c) and |c - c*| where c* is
the residual mean (does the energy valley sit on the truth?).

Usage:
    python scripts/probe_energy_shift.py --checkpoint <ckpt> \
        --model-config lotsa_mini_v3_head8_eval \
        --configs m_dense/D/short,loop_seattle/H/short --instances 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hydra import compose, initialize_config_dir                   # noqa: E402

from timejepa.evaluation import gift                                # noqa: E402
from timejepa.evaluation import refine as refine_mod                # noqa: E402
from timejepa.evaluation.loading import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.training import critic                                # noqa: E402
from evaluate_gift import prepare_context, tta_forecast             # noqa: E402

GRID = (-0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def energies_for_shift(model, ctx_norm, center_norm, z_pred, c, contextualized):
    cand = center_norm + c
    z_y = critic.encode_candidate(model, ctx_norm, cand, contextualized=contextualized)
    return critic.energy_per_item(z_y, z_pred, "cos")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="lotsa_tiny_mix_eval")
    ap.add_argument("--configs", default="m_dense/D/short,loop_seattle/H/short")
    ap.add_argument("--instances", type=int, default=64)
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()
    can_ctx = getattr(model.predictor, "w_film", None) is None

    grid = np.array(GRID, dtype=np.float32)
    results = {}
    for config in [c.strip() for c in args.configs.split(",")]:
        h = gift.prediction_length(config)
        series = gift.load_series(Path(args.gift_root), config)
        windows = gift.num_windows(config, min(len(s) for s in series))
        E_sa, E_cx, resid = [], [], []
        for inst in gift.iter_test_instances(series, h, windows):
            if len(resid) >= args.instances:
                break
            if np.isnan(inst.target).any():
                continue
            ctx = prepare_context(inst.context, model.input_length,
                                  model.patching.stride, model.patching.patch_size)
            if ctx is None:
                continue
            x = torch.as_tensor(ctx, dtype=torch.float32, device=device).view(1, -1, 1)
            with torch.no_grad():
                out = tta_forecast(model, x, h)
                fan = out["quantiles_denorm"]
                if fan.ndim == 4:
                    fan = fan[..., 0]
                ctx_norm, fan_norm = refine_mod.normalize_with_context(model, x, fan)
                y = torch.as_tensor(inst.target, dtype=torch.float32, device=device).view(1, -1, 1)
                _, y_norm = refine_mod.normalize_with_context(model, x, y)
                mid = fan_norm.shape[-1] // 2
                center = fan_norm[..., mid:mid + 1]
                hj = min(h, int(model.prediction_length))
                center, y_norm = center[:, :hj], y_norm[:, :hj]
                z_pred = critic.predict_latent(model, ctx_norm, None)[0]
                e_sa = [float(energies_for_shift(model, ctx_norm, center, z_pred, float(c), False))
                        for c in grid]
                e_cx = ([float(energies_for_shift(model, ctx_norm, center, z_pred, float(c), True))
                         for c in grid] if can_ctx else [float("nan")] * len(grid))
            E_sa.append(e_sa)
            E_cx.append(e_cx)
            resid.append(float((y_norm - center).mean()))
        if not resid:
            continue
        E_sa, E_cx, resid = np.array(E_sa), np.array(E_cx), np.array(resid)
        i0 = int(np.where(grid == 0.0)[0][0])

        def summarize(E):
            if not np.isfinite(E).all():
                return None
            dE = np.abs(E - E[:, i0:i0 + 1]).mean(axis=0)
            argmin = grid[E.argmin(axis=1)]
            nz = np.abs(resid) > 0.02
            sign_ok = float((np.sign(argmin[nz]) == np.sign(resid[nz])).mean()) if nz.any() else float("nan")
            stay = float((argmin == 0.0).mean())
            sp = [spearman(E[i], np.abs(grid - resid[i])) for i in range(len(E))]
            return {"mean_abs_dE_per_c": {f"{c:+.2f}": round(float(d), 5) for c, d in zip(grid, dE)},
                    "argmin_sign_matches_residual": round(sign_ok, 3),
                    "argmin_stays_at_zero": round(stay, 3),
                    "spearman_E_vs_dist_to_truth_mean": round(float(np.nanmean(sp)), 3),
                    "n": int(len(E))}

        results[config] = {"residual_mean_abs": round(float(np.abs(resid).mean()), 4),
                           "standalone": summarize(E_sa),
                           "contextualized": summarize(E_cx)}
        print(f"\n== {config} (n={len(resid)}, mean |residual| {np.abs(resid).mean():.3f} sigma)")
        for name in ("standalone", "contextualized"):
            r = results[config][name]
            if r is None:
                print(f"  {name}: n/a"); continue
            print(f"  {name}: |dE| per c " + " ".join(f"{k}:{v:.4f}" for k, v in r["mean_abs_dE_per_c"].items()))
            print(f"  {name}: argmin sign = residual sign {r['argmin_sign_matches_residual']:.2f} | "
                  f"argmin at 0: {r['argmin_stays_at_zero']:.2f} | spearman(E, |c-c*|) {r['spearman_E_vs_dist_to_truth_mean']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
