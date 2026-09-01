# ARCHIVED - not wired to live code, do not import (see scripts/archive/README.md).
"""
Is forecast uncertainty already encoded in the representations?

The question decides where a probabilistic head can live. The predictor is
trained under MSE against the target latent, so it converges to
E[z_target | z_context] - a conditional MEAN by construction (measured:
pred_var 0.6 against target_var 0.95). The spread lives in the residual between
predicted and actual target latent, which the model never sees at inference.

So a decoder-side quantile head can only work if the *inputs it receives* still
carry the information "this window is volatile". This script measures that
directly, on an existing checkpoint, with no training.

Four feature sets are probed against the per-window forecast error, each
answering a different design question:

    ctx_std     one scalar, the context's own standard deviation.
                The free baseline. If this matches the learned features, the
                encoder contributes nothing and a hand-crafted feature suffices.

    z_pred      the predicted target latents (mean+std pooled).
                This is exactly what a latent-only decoder head sees.

    z_ctx       the context embeddings (mean+std pooled).
                This is what wiring the context into the head adds.

    z_ctx+z_pred    both - the full context-fed input.

Read it as: if z_ctx clearly beats z_pred, wiring the context into the decoder
is worth it (context-fed over latent-only). If neither beats ctx_std by much, the encoder is
not representing volatility and a decoder-side head will be limited whatever we
feed it - which is the case for revisiting the pretraining objective (option C).

Usage:
    python scripts/probe_uncertainty.py +checkpoint_path=path/to/finetuned.ckpt
    python scripts/probe_uncertainty.py +checkpoint_path=... +nixtla=[ettm1,traffic,exchange]
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timejepa.data.datamodule import MonashDataModule          # noqa: E402
from timejepa.data.nixtla import download_and_convert, NIXTLA_REGISTRY  # noqa: E402

from evaluate import create_model_from_config, load_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("timejepa").setLevel(logging.ERROR)
logger = logging.getLogger("probe")


# =============================================================================
# RIDGE PROBE
# =============================================================================

def ridge_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
             train_frac: float = 0.7, seed: int = 0) -> Dict[str, float]:
    """
    Closed-form ridge with a held-out split, returning out-of-sample R^2.

    Ridge rather than plain least squares because the pooled embeddings are
    high-dimensional and correlated; held-out rather than in-sample because a
    256-feature probe on a few thousand windows would otherwise report a high
    R^2 purely from overfitting.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * train_frac)
    tr, te = idx[:cut], idx[cut:]

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # Standardize on train statistics only
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])

    d = Xtr.shape[1]
    reg = alpha * np.eye(d)
    reg[-1, -1] = 0.0                      # never penalize the intercept
    w = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)

    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    corr = float(np.corrcoef(pred, yte)[0, 1]) if len(yte) > 2 else float("nan")
    return {"r2": float(r2), "corr": corr, "n_test": int(len(yte)), "n_feat": int(d - 1)}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def collect(model, loader, horizon: int, device, max_windows: int) -> Dict[str, np.ndarray]:
    """
    Run the model and record, per window, the features and the realised error.

    The error is measured in the NORMALIZED space - the same space the finetune
    loss uses - so that windows of wildly different scale stay comparable.
    """
    feats = {"ctx_std": [], "z_ctx": [], "z_pred": []}
    errors = []
    seen = 0

    model.eval()
    for batch in loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        target = target[:, :horizon]

        out = model.forward_finetune(context, return_representations=True)
        pred = out["forecast"][:, :horizon]                  # normalized frame

        # Same normalization the finetune loss applies to the target
        if model.revin is not None:
            target_n = (target - model.revin.mean) / model.revin.std
        else:
            target_n = target

        err = (pred - target_n).abs().mean(dim=(1, 2))       # [B] per-window MAE

        z_ctx = out["context_embeddings"]                    # [B, N_ctx, D]
        z_pred = out["future_representations"]               # [B, N_tgt, D]

        feats["ctx_std"].append(context.std(dim=1).squeeze(-1).unsqueeze(-1).cpu())
        # mean AND std over patches: the spread across patch positions is itself
        # a plausible volatility signal, and pooling by mean alone would erase it
        feats["z_ctx"].append(torch.cat([z_ctx.mean(1), z_ctx.std(1)], -1).cpu())
        feats["z_pred"].append(torch.cat([z_pred.mean(1), z_pred.std(1)], -1).cpu())
        errors.append(err.cpu())

        seen += len(context)
        if seen >= max_windows:
            break

    return {
        **{k: torch.cat(v).float().numpy() for k, v in feats.items()},
        "err": torch.cat(errors).float().numpy(),
    }


def probe_dataset(data: Dict[str, np.ndarray], alpha: float) -> Dict[str, Dict]:
    """Probe log-error, which is far better conditioned than raw error."""
    y = np.log(data["err"] + 1e-6)
    sets = {
        "ctx_std": data["ctx_std"],
        "z_pred": data["z_pred"],
        "z_ctx": data["z_ctx"],
        "z_ctx+z_pred": np.hstack([data["z_ctx"], data["z_pred"]]),
    }
    return {name: ridge_r2(X, y, alpha=alpha) for name, X in sets.items()}


# =============================================================================
# MAIN
# =============================================================================

@hydra.main(version_base=None, config_path="../configs/model", config_name="tiny")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = cfg.get("checkpoint_path")
    if ckpt is None:
        raise ValueError("Specify with: +checkpoint_path=path/to/finetuned.ckpt")

    model = create_model_from_config(cfg)
    model = load_checkpoint(model, ckpt, device)
    model.set_pretrain_mode(False)

    datasets = cfg.get("nixtla") or ["ettm1", "ettm2", "traffic", "exchange"]
    if isinstance(datasets, ListConfig):
        datasets = list(datasets)
    horizon = int(cfg.get("probe_horizon", 96))
    max_windows = int(cfg.get("probe_max_windows", 4000))
    alpha = float(cfg.get("probe_alpha", 10.0))

    nixtla_cache = Path(cfg.data.data_dir) / "nixtla"
    all_rows: List[Dict] = []

    print("\n" + "=" * 84)
    print("UNCERTAINTY PROBE - can the realised forecast error be predicted from")
    print("the representations the decoder already receives?")
    print("=" * 84)
    print(f"context={cfg.model.seq_length}  horizon={horizon}  ridge alpha={alpha}")

    for name in datasets:
        if name.lower() not in NIXTLA_REGISTRY:
            logger.warning(f"skip unknown dataset {name}")
            continue
        try:
            path = download_and_convert(name, nixtla_cache, split="test")
            dm = MonashDataModule(
                data_path=path,
                context_length=cfg.model.seq_length,
                prediction_length=horizon,
                batch_size=cfg.data.batch_size,
                stride=horizon,
                normalize_mode="per_series",
                normalizer_type="identity",
                clip_outliers=False,
                train_val_test_split=(0.0, 0.0, 1.0),
                num_workers=0,
            )
            dm.prepare_data()
            dm.setup("fit")

            data = collect(model, dm.test_dataloader(), horizon, device, max_windows)
            res = probe_dataset(data, alpha)

            print(f"\n-- {name}   ({len(data['err'])} windows, "
                  f"median error {np.median(data['err']):.3f})")
            print(f"   {'features':<16}{'dim':>6}{'out-of-sample R2':>22}{'corr':>9}")
            print("   " + "-" * 51)
            base = res["ctx_std"]["r2"]
            for k, v in res.items():
                gain = "" if k == "ctx_std" else f"   ({v['r2'] - base:+.3f} vs ctx_std)"
                print(f"   {k:<16}{v['n_feat']:>6}{v['r2']:>22.3f}{v['corr']:>9.3f}{gain}")

            all_rows.append({"dataset": name, **{k: v["r2"] for k, v in res.items()}})

        except Exception as e:
            logger.error(f"{name}: {e}")

    if not all_rows:
        return

    print("\n" + "=" * 84)
    print("SUMMARY - out-of-sample R2 on log|error|")
    print("=" * 84)
    keys = ["ctx_std", "z_pred", "z_ctx", "z_ctx+z_pred"]
    print(f"{'dataset':<14}" + "".join(f"{k:>16}" for k in keys))
    print("-" * 78)
    for r in all_rows:
        print(f"{r['dataset']:<14}" + "".join(f"{r[k]:>16.3f}" for k in keys))
    means = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
    print("-" * 78)
    print(f"{'mean':<14}" + "".join(f"{means[k]:>16.3f}" for k in keys))

    print("\nReading:")
    d_ctx = means["z_ctx"] - means["z_pred"]
    d_free = means["z_ctx+z_pred"] - means["ctx_std"]
    if means["z_ctx+z_pred"] < 0.05:
        print("  No feature set predicts the error. Uncertainty is NOT encoded:")
        print("  a decoder-side head will be limited whatever it is fed.")
        print("  -> the case where option C (probabilistic predictor) is justified.")
    else:
        if d_free < 0.03:
            print(f"  Representations add almost nothing over ctx_std ({d_free:+.3f}).")
            print("  The encoder does not represent volatility beyond a trivial statistic.")
        else:
            print(f"  Representations beat ctx_std by {d_free:+.3f} R2: uncertainty IS encoded.")
        if d_ctx > 0.03:
            print(f"  z_ctx exceeds z_pred by {d_ctx:+.3f}: wiring the context into the decoder")
            print("  adds information the predicted latent lost -> feed the context.")
        else:
            print(f"  z_ctx only adds {d_ctx:+.3f} over z_pred: the predicted latent suffices.")


if __name__ == "__main__":
    main()
