"""
Prototype v0 - ENERGY-based forecasting on the local Nixtla benchmark.

    python scripts/evaluate_energy.py \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_full/last.ckpt \\
        --decoder-checkpoint checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/last.ckpt

The propose-judge-weight readout of the JEPA-energy doc, end to end, with NO
decoder and NO finetune: the PRETRAIN checkpoint alone.

  1. propose: K block-bootstraps of the history + seasonal naive + drift;
  2. judge:   E_k = 1 - cos(z_pred, enc([ctx||cand_k])) - CONTEXTUALIZED
              encoding (E18c conclusion: a READOUT choice, valid even on a
              lineage trained standalone);
  3. weight:  w ~ exp(-(E-mean_E)/std_E) - softmax over per-instance
              standardized energies. v0 has no free temperature on purpose:
              standardizing makes the weight scale-invariant, so NO
              hyperparameter tuned on the test. In-context T calibration
              (conformal) is the next step, not this one;
  4. read:    weighted quantiles (9 GIFT levels) per time step;
              point forecast = weighted median.

Comparability: windows, seasonality, MASE (pooled, repo helper) and WQL
(GluonTS convention, repo helper) are STRICTLY shared between:
    energy    the prototype above (pretrain only, zero downstream training)
    decoder   the existing generative path (FINETUNED ckpt, --decoder-checkpoint)
    snaive    seasonal naive (point forecast -> its WQL collapses to ND, expected)
Compare energy to the experiment log via RATIOS vs snaive from the same run,
never via absolute values from another harness. Known limit: the bootstrap
only proposes recombinations of the past (no extrapolation outside the
envelope) - the "decoder trajectories" third of the hybrid is NOT in this v0.

Window protocol: non-overlapping (stride = h) on the converted test split
(data/processed/nixtla/), context 1024, h = 96 (one predictor pass, no
rolling - the energy readout is single-shot by construction).
Read-only; console output + JSON under evaluation/energy_nixtla/.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                   # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.data.nixtla import NIXTLA_REGISTRY                   # noqa: E402
from timejepa.training.utils.metrics import mase, weighted_quantile_loss  # noqa: E402
from timejepa.training.utils.baselines import get_seasonality      # noqa: E402
from probe_energy import block_bootstrap                           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_energy")

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_DATASETS = ["ettm1", "ettm2", "etth1", "etth2", "weather", "exchange"]


# ---------------------------------------------------------------------------
# Energy readout
# ---------------------------------------------------------------------------

@torch.no_grad()
def candidate_energies(model, ctx: np.ndarray, history: np.ndarray, h: int,
                       m: int, K: int, rng, device,
                       extra_cands: np.ndarray = None,
                       refine_steps: int = 0, refine_lr: float = 0.05,
                       h_judge: int = None, centered: bool = False):
    """
    Build the candidate pool and compute their energies -> (cands, e).

    extra_cands [N, h]: extra candidates submitted to the SAME judge - the
    "hybrid" mode (user protocol 2026-08-21) puts the FINETUNED decoder's fan
    trajectories there: the decoder proposes (it can extrapolate outside the
    historical envelope, which the bootstrap cannot - the E18d exchange
    failure), the pretrain judges (its energy alignment is intact, E18b).
    Two checkpoints of one lineage in tandem, no new model.

    h_judge: judging window (G12-on-GIFT bug, 2026-08-31). The energy readout
    is SINGLE-SHOT by construction (z_pred = num_target_patches, native h
    256); on GIFT, h ranges from 6 to 720. When h_judge is given, the energy
    is computed on the FIRST h_judge steps of each candidate (returned
    candidates stay full length for the quantile readout). Honest readout:
    TTM path diversity is born in the first segment (first-segment jitter
    only), so judging the start captures their spread; for h < patch_size the
    candidate is edge-padded FOR ENCODING ONLY (the transform is pointwise,
    so padding after normalization = normalizing the padded). None =
    bit-identical to the existing behavior (Nixtla, h=96).
    """
    drift = ctx[-1] + (ctx[-1] - ctx[max(0, len(ctx) - m - 1)]) / max(m, 1) \
        * np.arange(1, h + 1, dtype=np.float32)
    sn = np.tile(ctx[-m:], (h + m - 1) // m + 1)[:h].astype(np.float32)
    # Two block scales: the full seasonality (cycle structure) and a
    # sub-block (local textures) - diversifies the pool without learning.
    blocks = [max(8, min(m, h)), max(8, min(m, h) // 3)]
    if centered and extra_cands is not None:
        # G12c (2026-08-31) - bootstrap CENTERED ON THE PROPOSER. The dilution
        # diagnosis (4 replications) is a CENTER problem, not a spread
        # problem: a bootstrap of the raw history is centered on the PAST,
        # and every gram of weight it gets pulls the weighted median off the
        # proposer's trend. Here: candidates = the proposer's own path +
        # resampled blocks of the seasonal INNOVATIONS (y_t - y_{t-m}) - all
        # candidates share the center, center dilution is impossible BY
        # CONSTRUCTION, and the judge only arbitrates what a verifier should
        # touch: texture and spread. Drift (the anchor that poisoned the
        # ttmonly run on D/W configs) leaves the pool; sn stays as a
        # legitimate alternative center.
        center = extra_cands[0].astype(np.float32)
        if len(history) > m:
            resid = (history[m:] - history[:-m]).astype(np.float32)
        else:
            resid = np.diff(history).astype(np.float32)
        pool = [sn] + [c.astype(np.float32) for c in extra_cands] \
            + [center + block_bootstrap(resid, h, blocks[i % 2], rng)
               for i in range(K)]
    else:
        pool = [sn, drift] + [block_bootstrap(history, h, blocks[i % 2], rng)
                              for i in range(K)]
        if extra_cands is not None:
            pool += [c.astype(np.float32) for c in extra_cands]
    cands = np.stack(pool)

    x_ctx = torch.from_numpy(ctx).reshape(1, -1, 1).to(device)
    xc = torch.from_numpy(cands).unsqueeze(-1).to(device)

    if model.robust_scaler is not None:
        model.robust_scaler.fit(x_ctx)
        x_ctx = model.robust_scaler.transform(x_ctx)
        xc = model.robust_scaler.transform(xc)
    ctx_norm = model.revin(x_ctx, mode='norm') if model.revin is not None else x_ctx
    xc_norm = (xc - model.revin.mean) / model.revin.std if model.revin is not None else xc

    # Judging window: the energy is computed on the first hj steps
    # (edge-padded if shorter than a patch); cands (full length) carries the
    # quantile readout. The slice itself is taken AFTER the refine block
    # (which reassigns xc_norm).
    hj = h if h_judge is None else min(h_judge, h)
    if refine_steps > 0 and hj < h:
        raise ValueError("refine_steps with h_judge < h would refine "
                         "candidates through a truncated energy - undefined.")
    P = model.patching.patch_size
    hj_enc = max(hj, P)
    n_tgt = (hj_enc - P) // model.patching.stride + 1
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb, num_targets=model.num_target_patches,
        w=(torch.ones(1, device=device)
           if hasattr(model.predictor, 'w_film') else None))[:, :n_tgt, :]

    # Gradient refinement ("planning by backprop"): the energy is
    # differentiable in y - a few descent steps ON THE CANDIDATES themselves
    # slide them toward the nearest valley of the landscape. Pure test-time
    # compute, no weight modified. Goodhart guard: few steps, small lr - too
    # much optimization would craft adversarial candidates that minimize E
    # without looking like a future.
    if refine_steps > 0:
        with torch.enable_grad():
            xc_ref = xc_norm.detach().clone().requires_grad_(True)
            opt = torch.optim.SGD([xc_ref], lr=refine_lr)
            for _ in range(refine_steps):
                opt.zero_grad()
                full_r = torch.cat(
                    [ctx_norm.expand(xc_ref.shape[0], -1, -1), xc_ref], dim=1)
                z_r = model.online_encoder(model.patching(full_r))[:, -n_tgt:, :]
                e_r = (1.0 - torch.nn.functional.cosine_similarity(
                    z_r.flatten(1), z_pred.expand_as(z_r).flatten(1), dim=1)).sum()
                e_r.backward()
                opt.step()
        xc_norm = xc_ref.detach()
        # Back to raw space for the quantile readout: inverse RevIN (context
        # stats) then inverse arcsinh if the checkpoint carries it.
        raw = xc_norm * model.revin.std + model.revin.mean \
            if model.revin is not None else xc_norm
        if model.robust_scaler is not None:
            raw = model.robust_scaler.inverse(raw)
        cands = raw[..., 0].cpu().numpy()

    xc_judge = xc_norm[:, :hj]
    if hj < P:
        xc_judge = torch.cat(
            [xc_judge, xc_judge[:, -1:].expand(-1, P - hj, -1)], dim=1)
    full = torch.cat([ctx_norm.expand(xc_judge.shape[0], -1, -1), xc_judge], dim=1)
    z_cand = model.online_encoder(model.patching(full))[:, -n_tgt:, :]
    e = 1.0 - torch.nn.functional.cosine_similarity(
        z_cand.flatten(1), z_pred.expand_as(z_cand).flatten(1), dim=1)
    return cands, e.cpu().numpy()


def fan_from_energies(cands: np.ndarray, e: np.ndarray,
                      temperature: float = 1.0) -> np.ndarray:
    """
    Energies -> Gibbs weights -> weighted quantiles [h, 9]. T applies to the
    STANDARDIZED energies (v0's scale invariance is preserved): T=1 = v0
    behavior; SMALL T = contrasted judge, mass concentrates on the best
    candidates - the cure for the dilution measured three times (E18e / e-v2
    / g, always weather+exchange); large T = near-uniform pool.
    """
    # Degeneracy guard (measured, ttmonly run 2026-08-31): a pool of
    # near-identical candidates gives near-identical energies, and dividing
    # by a tiny std turns the softmax into a noise amplifier (mass dropped on
    # an arbitrary anchor - ett1/D MASE 1.63->5.29). Below the threshold the
    # energy signal is noise: uniform weights.
    if e.std() < 1e-4:
        w = np.full(len(e), 1.0 / len(e))
    else:
        z = (e - e.mean()) / e.std()
        w = np.exp(-z / max(temperature, 1e-3)); w /= w.sum()

    h = cands.shape[1]
    fan = np.empty((h, len(QUANTILE_LEVELS)), dtype=np.float32)
    order = np.argsort(cands, axis=0)                       # [Nc, h]
    for t in range(h):
        idx = order[:, t]
        cum = np.cumsum(w[idx])
        vals = cands[idx, t]
        for qi, q in enumerate(QUANTILE_LEVELS):
            fan[t, qi] = vals[np.searchsorted(cum, q, side='left').clip(0, len(vals) - 1)]
    return fan


@torch.no_grad()
def energy_readout(model, ctx, history, h, m, K, rng, device,
                   extra_cands=None, refine_steps=0, refine_lr=0.05,
                   temperature: float = 1.0, centered: bool = False) -> np.ndarray:
    """The full pipeline: pool + energies + weighted readout at a given T."""
    cands, e = candidate_energies(model, ctx, history, h, m, K, rng, device,
                                  extra_cands=extra_cands,
                                  refine_steps=refine_steps, refine_lr=refine_lr,
                                  centered=centered)
    return fan_from_energies(cands, e, temperature)


T_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def pinball_np(fan: np.ndarray, target: np.ndarray) -> float:
    diff = target[:, None] - fan                            # [h, 9]
    q = np.asarray(QUANTILE_LEVELS)
    return float(np.maximum(q * diff, (q - 1.0) * diff).mean())


@torch.no_grad()
def calibrate_temperature(model, ctx: np.ndarray, h: int, m: int, K: int,
                          device, n_cal: int = 2, proposer_fn=None,
                          seed: int = 1000, centered: bool = False) -> float:
    """
    IN-CONTEXT T calibration - the G12(a) prerequisite, and the lever pointed
    at by three replications of the dilution signature. Replay the FULL
    pipeline (real pool, proposer included) on n_cal past sub-windows of the
    context whose continuation is known; keep the T minimizing mean pinball.
    Energies are computed ONCE per sub-window, sweeping the grid only costs
    softmaxes. Zero training, zero look at the test; dedicated rng (offset
    seed) so the main path's draws stay bit-paired with uncalibrated runs.
    """
    rng = np.random.default_rng(seed)
    scores = {T: [] for T in T_GRID}
    for j in range(n_cal):
        cut = len(ctx) - (j + 1) * h
        if cut < 512:
            break
        sub_ctx, known = ctx[:cut].copy(), ctx[cut:cut + h]
        if not (np.isfinite(sub_ctx).all() and np.isfinite(known).all()):
            continue
        extra = proposer_fn(sub_ctx) if proposer_fn is not None else None
        cands, e = candidate_energies(model, sub_ctx, sub_ctx, h, m, K, rng,
                                      device, extra_cands=extra,
                                      centered=centered)
        for T in T_GRID:
            scores[T].append(pinball_np(fan_from_energies(cands, e, T), known))
    valid = {T: sum(v) / len(v) for T, v in scores.items() if v}
    return min(valid, key=valid.get) if valid else 1.0


@torch.no_grad()
def mc_dropout_paths(dec_model, ctx: np.ndarray, h: int, n: int, device) -> np.ndarray:
    """
    n epistemic trajectories from the FINETUNED decoder: the model's Dropout
    modules are switched to train mode FOR THE FORWARDS ONLY (the rest -
    norm, EMA - stays in eval), then restored. Each stochastic forward gives
    a different median = a temporally COHERENT trajectory, unlike quantile
    paths (marginals). Script-only - the core code is untouched.
    """
    x = torch.from_numpy(ctx).reshape(1, -1, 1).to(device)
    drops = [mod for mod in dec_model.modules()
             if isinstance(mod, torch.nn.Dropout) and mod.p > 0]
    for mod in drops:
        mod.train()
    try:
        paths = [dec_model.forecast(x, n=h)["forecast_denorm"][0, :, 0].cpu().numpy()
                 for _ in range(n)]
    finally:
        for mod in drops:
            mod.eval()
    return np.stack(paths)


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------

class TTMProposer:
    """
    External proposer G12(b): TTM-R3 (IBM Granite, ~1.4M - the size-class
    rival, CRPS 0.520 on GIFT). Point forecast only -> diversity comes from N
    jittered contexts (noise 0.05*std) plus the clean path. Lazily loaded:
    the script stays usable without granite-tsfm installed. The G12 measure
    is the UPLIFT: reader `ttm` alone vs `hybrid_ttm` (bootstrap + SN + drift
    + TTM paths, judged by OUR pretrain).
    """

    def __init__(self, model_id: str, device, revision: str = "main"):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        # Warning: the TTM repo hosts its variants by REVISION; `main` is a
        # horizon-30 variant that loads with REINITIALIZED head weights
        # (MISSING warning measured) - always pass the exact revision.
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_id, revision=revision)
        self.model.to(device).eval()
        self.ctx_len = self.model.config.context_length
        self.pred_len = self.model.config.prediction_length
        self.device = device

    @torch.no_grad()
    def paths(self, ctx: np.ndarray, h: int, n_jitter: int, rng) -> np.ndarray:
        assert h <= self.pred_len, f"h={h} > TTM horizon {self.pred_len}"
        base = ctx[-self.ctx_len:].astype(np.float32)
        if len(base) < self.ctx_len:                     # left edge-pad
            base = np.concatenate([np.full(self.ctx_len - len(base), base[0],
                                           dtype=np.float32), base])
        ctxs = [base] + [base + rng.normal(0, 0.05 * max(base.std(), 1e-8),
                                           size=base.shape).astype(np.float32)
                         for _ in range(n_jitter)]
        x = torch.from_numpy(np.stack(ctxs)).unsqueeze(-1).to(self.device)
        out = self.model(past_values=x).prediction_outputs                # [N, P, 1]
        return out[:, :h, 0].cpu().numpy()


def iter_windows(series: np.ndarray, ctx_len: int, h: int, max_windows: int):
    """Non-overlapping windows (stride=h), spread across the whole test split."""
    starts = list(range(ctx_len, len(series) - h + 1, h))
    if len(starts) > max_windows:
        starts = [starts[i] for i in
                  np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    for s in starts:
        yield series[s - ctx_len:s].astype(np.float32), series[s:s + h].astype(np.float32), s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True, help="PRETRAIN checkpoint (energy readout)")
    ap.add_argument("--decoder-checkpoint", default=None,
                    help="FINETUNED checkpoint (generative reference, same harness)")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="gradient-descent steps of E on the candidates (planning by backprop)")
    ap.add_argument("--refine-lr", type=float, default=0.05)
    ap.add_argument("--decoder-samples", type=int, default=0,
                    help="decoder MC-dropout paths added to the hybrid pool")
    ap.add_argument("--proposer-ttm", default=None, const="ibm-granite/granite-timeseries-ttm-r3",
                    nargs="?", help="enable the external TTM proposer (optional HF id)")
    ap.add_argument("--ttm-revision", default="1024-96-r3",
                    help="HF revision (context-horizon) - main = reinitialized head!")
    ap.add_argument("--calibrate-T", action="store_true",
                    help="calibrate T per series on past context sub-windows (G12a)")
    ap.add_argument("--cal-windows", type=int, default=2)
    ap.add_argument("--centered-bootstrap", action="store_true",
                    help="G12c: hybrid_ttm pool centered on the proposer - "
                         "seasonal-innovation bootstrap GLUED onto the TTM "
                         "path (anti-dilution by construction), drift out of "
                         "the pool")
    ap.add_argument("--ttm-jitter", type=int, default=4,
                    help="jittered contexts per window to diversify TTM")
    ap.add_argument("--max-windows", type=int, default=40, help="per series")
    ap.add_argument("--max-series", type=int, default=21)
    ap.add_argument("--model-config", default="lotsa_tiny_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    ttm = (TTMProposer(args.proposer_ttm, device, revision=args.ttm_revision)
           if args.proposer_ttm else None)
    if ttm:
        logger.info(f"TTM proposer: {args.proposer_ttm} (ctx {ttm.ctx_len}, h {ttm.pred_len})")

    dec_model = None
    if args.decoder_checkpoint:
        dec_model = create_model_from_config(cfg)
        load_checkpoint(dec_model, args.decoder_checkpoint, device)
        dec_model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    h, ctx_len = args.horizon, cfg.model.seq_length
    results = {}

    for name in args.datasets.split(","):
        name = name.strip()
        info = NIXTLA_REGISTRY[name]
        m = get_seasonality(freq=info.freq)
        data = np.load(f"data/processed/nixtla/nixtla_{name}_test.npy")[:args.max_series]

        acc = {r: {"fan": [], "tgt": [], "ctx": []}
               for r in (("energy", "snaive")
                         + (("decoder", "hybrid") if dec_model else ())
                         + (("ttm", "hybrid_ttm") if ttm else ()))}
        T_used = {"energy": [], "hybrid": [], "hybrid_ttm": []}
        for series in data:
            T = {"energy": 1.0, "hybrid": 1.0, "hybrid_ttm": 1.0}
            if args.calibrate_T:
                # one calibration PER SERIES and PER POOL COMPOSITION (the
                # energy / hybrid_ttm pools do not have the same contrast), on
                # the first window's context - amortized over ~max_windows.
                first = next(iter_windows(series, ctx_len, h, args.max_windows), None)
                if first is not None:
                    c0 = first[0]
                    T["energy"] = calibrate_temperature(
                        model, c0, h, m, args.candidates, device,
                        n_cal=args.cal_windows, seed=args.seed + 1000)
                    if ttm is not None:
                        T["hybrid_ttm"] = calibrate_temperature(
                            model, c0, h, m, args.candidates, device,
                            n_cal=args.cal_windows, seed=args.seed + 1000,
                            centered=args.centered_bootstrap,
                            proposer_fn=lambda sc: ttm.paths(sc, h, args.ttm_jitter,
                                                             np.random.default_rng(args.seed + 2000)))
                    T["hybrid"] = T["energy"]
                    for k in T_used:
                        T_used[k].append(T[k])
            for ctx, tgt, _ in iter_windows(series, ctx_len, h, args.max_windows):
                if not (np.isfinite(ctx).all() and np.isfinite(tgt).all()):
                    continue
                fan = energy_readout(model, ctx, ctx, h, m, args.candidates, rng, device,
                                     refine_steps=args.refine_steps,
                                     refine_lr=args.refine_lr,
                                     temperature=T["energy"])
                acc["energy"]["fan"].append(fan)
                sn = np.tile(ctx[-m:], (h + m - 1) // m + 1)[:h]
                acc["snaive"]["fan"].append(np.repeat(sn[:, None], 9, axis=1))
                if dec_model is not None:
                    with torch.no_grad():
                        out = dec_model.forecast(
                            torch.from_numpy(ctx).reshape(1, -1, 1).to(device), n=h)
                    q = out["quantiles_denorm"][0].cpu().numpy()
                    q = q[..., 0] if q.ndim == 3 else q            # [h, 9]
                    acc["decoder"]["fan"].append(q)
                    # Hybrid: quantile trajectories + decoder MC-dropout paths
                    # enter the pool; the pretrain judges everything.
                    dec_paths = q.T
                    if args.decoder_samples > 0:
                        dec_paths = np.concatenate(
                            [dec_paths,
                             mc_dropout_paths(dec_model, ctx, h,
                                              args.decoder_samples, device)])
                    acc["hybrid"]["fan"].append(energy_readout(
                        model, ctx, ctx, h, m, args.candidates, rng, device,
                        extra_cands=dec_paths,
                        refine_steps=args.refine_steps,
                        refine_lr=args.refine_lr,
                        temperature=T["hybrid"]))
                if ttm is not None:
                    tp = ttm.paths(ctx, h, args.ttm_jitter, rng)          # [N, h]
                    # `ttm` alone = its clean path (point -> repeated fan, WQL=ND)
                    acc["ttm"]["fan"].append(np.repeat(tp[:1].T, 9, axis=1))
                    acc["hybrid_ttm"]["fan"].append(energy_readout(
                        model, ctx, ctx, h, m, args.candidates, rng, device,
                        extra_cands=tp,
                        refine_steps=args.refine_steps, refine_lr=args.refine_lr,
                        temperature=T["hybrid_ttm"],
                        centered=args.centered_bootstrap))
                for r in acc:
                    acc[r]["tgt"].append(tgt); acc[r]["ctx"].append(ctx)

        results[name] = {}
        for r, d in acc.items():
            fans = torch.from_numpy(np.stack(d["fan"]))            # [B, h, 9]
            tgts = torch.from_numpy(np.stack(d["tgt"]))            # [B, h]
            ctxs = torch.from_numpy(np.stack(d["ctx"]))            # [B, L]
            med = fans[..., 4]
            res = {
                "mase": float(mase(med, tgts, ctxs, season_length=m)),
                "wql": float(weighted_quantile_loss(
                    fans.permute(2, 0, 1), tgts, QUANTILE_LEVELS)),
                "n_windows": int(fans.shape[0]),
            }
            results[name][r] = res
        if args.calibrate_T and T_used["energy"]:
            import collections
            for k in ("energy", "hybrid_ttm"):
                if T_used[k]:
                    cnt = collections.Counter(T_used[k])
                    logger.info(f"  calibrated T [{k}] {name}: "
                                + ", ".join(f"{t}x{n}" for t, n in sorted(cnt.items())))
        sn_ref = results[name]["snaive"]
        line = f"{name:12s} (n={results[name]['energy']['n_windows']:4d}, m={m:3d})"
        for r in acc:
            res = results[name][r]
            line += (f" | {r}: MASE {res['mase']:.3f}"
                     f" ({res['mase'] / sn_ref['mase']:.2f}x) "
                     f"WQL {res['wql']:.3f} ({res['wql'] / sn_ref['wql']:.2f}x)")
        logger.info(line)

    out_dir = Path("evaluation/energy_nixtla"); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.checkpoint).stem}_h{h}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    logger.info(f"JSON: {out}")


if __name__ == "__main__":
    main()
