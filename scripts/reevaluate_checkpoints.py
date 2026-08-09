"""
Re-evaluate existing checkpoints under the corrected protocol (P0.7).

Purpose
-------
Every number in ../TimeJEPA_2ndbatch_results/ was produced with `skip_revin=True`
on the Nixtla long-horizon benchmarks. Those datasets are z-scored GLOBALLY with
train statistics, which is NOT the per-window instance normalization the model
was trained under. The encoder therefore saw out-of-distribution inputs and its
output was scored in the wrong space — visible as a constant level offset in the
h96 forecast plots.

This script re-runs the same checkpoints on the same data in BOTH modes:

    legacy : skip_revin=True   (reproduces the old numbers, sanity check)
    fixed  : skip_revin=False  (RevIN on, the regime the model was trained in)

...and scores both against seasonal-naive / naive-last / context-mean baselines,
so the result is finally interpretable.

Each checkpoint's architecture is read from the `eval_config.yaml` that was saved
alongside its original evaluation, so no config guessing is involved.

Usage
-----
    python scripts/reevaluate_checkpoints.py
    python scripts/reevaluate_checkpoints.py --datasets ettm1 etth1 --horizons 96
    python scripts/reevaluate_checkpoints.py --only tiny --max-samples 2000
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from timejepa.models import JEPATST                                    # noqa: E402
from timejepa.models.decoders import ForecastingHead                   # noqa: E402
from timejepa.data.datamodule import MonashDataModule                  # noqa: E402
from timejepa.data.nixtla import download_and_convert, NIXTLA_REGISTRY  # noqa: E402
from timejepa.training.utils.baselines import get_seasonality          # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("timejepa").setLevel(logging.ERROR)
logger = logging.getLogger("reeval")
logger.setLevel(logging.INFO)

RESULTS_ROOT = REPO_ROOT.parent / "TimeJEPA_2ndbatch_results"
CKPT_ROOT = RESULTS_ROOT / "checkpoints"
EVAL_ROOT = RESULTS_ROOT / "lightning" / "evaluation"

DEFAULT_DATASETS = ["ettm1", "ettm2", "etth1", "etth2", "electricity", "exchange", "traffic", "weather"]
DEFAULT_HORIZONS = [96, 192, 336, 720]

# Nixtla's LongHorizon ships a single 'OT' series for these groups, while the
# published benchmark tables average over all 7 ETT channels.
UNIVARIATE_ONLY = {"etth1", "etth2"}


# =============================================================================
# DISCOVERY
# =============================================================================

def discover_pairs() -> List[Dict]:
    """Find (checkpoint, eval_config) pairs that can be re-evaluated."""
    pairs = []
    if not EVAL_ROOT.exists():
        logger.error(f"Results dir not found: {EVAL_ROOT}")
        return pairs

    for eval_dir in sorted(EVAL_ROOT.glob("*/*")):
        if not eval_dir.is_dir():
            continue
        cfg_path = eval_dir / "eval_config.yaml"
        if not cfg_path.exists():
            continue

        model_name = eval_dir.parent.name
        ckpt_name = eval_dir.name

        matches = list(CKPT_ROOT.glob(f"*/*/{ckpt_name}.ckpt"))
        if not matches:
            continue

        pairs.append({
            "model_name": model_name,
            "ckpt_name": ckpt_name,
            "ckpt_path": matches[0],
            "cfg_path": cfg_path,
            "old_results": eval_dir / "nixtla_results.json",
        })
    return pairs


# =============================================================================
# MODEL
# =============================================================================

def build_model(cfg, device: torch.device) -> JEPATST:
    """Rebuild the exact architecture described by a saved eval_config.yaml."""
    model = JEPATST(
        input_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        d_model=cfg.model.encoder.d_model,
        num_layers=cfg.model.encoder.n_layers,
        num_heads=cfg.model.encoder.n_heads,
        d_ff=cfg.model.encoder.d_ff,
        dropout=cfg.model.encoder.dropout,
        activation=cfg.model.encoder.activation,
        predictor_type=cfg.model.predictor.type,
        predictor_num_layers=cfg.model.predictor.n_layers,
        predictor_num_heads=cfg.model.predictor.n_heads,
        predictor_d_ff=cfg.model.predictor.d_ff,
        decoder_type=cfg.model.decoder.type,
        ema_tau_base=cfg.model.target_encoder.momentum_base,
        ema_tau_end=cfg.model.target_encoder.momentum_final,
        use_revin=cfg.model.encoder.use_revin,
    )
    # Match train.py / evaluate.py: the decoder is rebuilt with the right stride
    model.decoder = ForecastingHead(
        d_model=cfg.model.decoder.d_model,
        patch_size=cfg.model.patch_length,
        stride=cfg.model.stride,
        prediction_length=cfg.model.prediction_length,
        num_features=cfg.model.num_channels,
        decoder_type=cfg.model.decoder.type,
        revin=model.revin,
    )
    return model.to(device)


def load_weights(model: JEPATST, ckpt_path: Path, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state.items():
        key = k.replace("model.", "").replace("_orig_mod.", "")
        if "target_encoder" in key:
            continue
        if "revin" in key and (key.endswith(".mean") or key.endswith(".std")):
            continue
        cleaned[key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    critical = [k for k in missing if "target_encoder" not in k and not k.endswith((".mean", ".std"))]
    model.eval()
    return {"missing_critical": critical, "unexpected": list(unexpected)[:5]}


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def run_one(model, loader, horizon, device, skip_revin, max_samples):
    ctxs, preds, tgts = [], [], []
    n = 0
    for batch in loader:
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        if context.ndim == 2:
            context = context.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        out = model.forecast(context, n=horizon, skip_revin=skip_revin)
        prediction = out["forecast_denorm"]
        target = target[:, :horizon]

        if context.shape[-1] == 1:
            context, prediction, target = context.squeeze(-1), prediction.squeeze(-1), target.squeeze(-1)

        ctxs.append(context.cpu())
        preds.append(prediction.cpu())
        tgts.append(target.cpu())
        n += len(context)
        if max_samples and n >= max_samples:
            break

    return torch.cat(ctxs), torch.cat(preds), torch.cat(tgts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--only", type=str, default=None, help="Substring filter on checkpoint name")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="lightning/reevaluation")
    ap.add_argument("--no-resume", action="store_true",
                    help="Recompute everything instead of reusing saved cells")
    args = ap.parse_args()

    # Imported late so the corrected evaluate.py helpers are picked up
    from evaluate import evaluate_with_baselines  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs()
    if args.only:
        pairs = [p for p in pairs if args.only in p["ckpt_name"] or args.only in p["model_name"]]

    print("=" * 100)
    print(f"RE-EVALUATION — {len(pairs)} checkpoint(s), device={device}")
    print("=" * 100)
    for p in pairs:
        print(f"  • {p['model_name']:<14} {p['ckpt_name']}")
    print()

    # Materialize the test splits once (shared across checkpoints)
    nixtla_cache = REPO_ROOT / "data" / "processed" / "nixtla"
    data_paths = {}
    for ds in args.datasets:
        if ds.lower() not in NIXTLA_REGISTRY:
            logger.warning(f"skip unknown dataset {ds}")
            continue
        try:
            data_paths[ds] = download_and_convert(ds, nixtla_cache, split="test")
        except Exception as e:
            logger.error(f"could not prepare {ds}: {e}")

    # Load every model up front. Building a MonashDataModule is by far the most
    # expensive step (traffic alone generates ~25k window indices over 862
    # series), so the loop order is dataset -> horizon -> checkpoint: each
    # DataModule is constructed ONCE and reused across all checkpoints, instead
    # of being rebuilt per checkpoint.
    models = []
    for pair in pairs:
        name = f"{pair['model_name']}/{pair['ckpt_name']}"
        cfg = OmegaConf.load(pair["cfg_path"])
        try:
            model = build_model(cfg, device)
            info = load_weights(model, pair["ckpt_path"], device)
            if info["missing_critical"]:
                print(f"  ⚠️  {name}: missing keys {info['missing_critical'][:6]}")
            models.append({
                "name": name,
                "model": model,
                "ctx_len": int(cfg.model.seq_length),
                "native_h": int(cfg.model.prediction_length),
                "decoder": str(cfg.model.decoder.type),
            })
            print(f"  ✓ {name:<62} ctx={cfg.model.seq_length} "
                  f"native_h={cfg.model.prediction_length} dec={cfg.model.decoder.type}",
                  flush=True)
        except Exception as e:
            print(f"  ❌ {name}: could not build/load: {e}")
            traceback.print_exc()

    if not models:
        print("No models could be loaded.")
        return

    # Context lengths differ across checkpoints (384 vs 512), and the DataModule
    # depends on it, so group by context length.
    ctx_lengths = sorted({m["ctx_len"] for m in models})
    print(f"\n  context lengths present: {ctx_lengths}\n", flush=True)

    # Resume: reload anything a previous (possibly interrupted) run computed.
    # A full grid takes hours on CPU, so recomputing finished cells is wasteful.
    all_out = {}
    resumed = 0
    for entry in models:
        safe = entry["name"].replace("/", "__")
        prev = out_root / f"{safe}.json"
        if prev.exists() and not args.no_resume:
            try:
                blob = json.loads(prev.read_text())
                all_out[entry["name"]] = blob
                resumed += sum(
                    1 for hs in blob.values() for r in hs.values() if "fixed" in r
                )
                continue
            except Exception:
                pass
        all_out[entry["name"]] = {}
    if resumed:
        print(f"  ↻ resumed {resumed} already-computed (dataset, horizon) cells\n", flush=True)

    def already_done(name: str, ds: str, h: int) -> bool:
        return "fixed" in all_out.get(name, {}).get(ds, {}).get(str(h), {})

    def flush_results():
        for entry in models:
            safe = entry["name"].replace("/", "__")
            with open(out_root / f"{safe}.json", "w") as f:
                json.dump(all_out[entry["name"]], f, indent=2)
        with open(out_root / "all_reevaluation.json", "w") as f:
            json.dump(all_out, f, indent=2)

    for ds, path in data_paths.items():
        season = get_seasonality(ds)
        flag = "  [univariate OT only — NOT comparable to published tables]" if ds in UNIVARIATE_ONLY else ""
        print("\n" + "=" * 100)
        print(f"▶ {ds}  (m={season}){flag}", flush=True)
        print("=" * 100)

        for h in args.horizons:
            for ctx_len in ctx_lengths:
                group = [
                    m for m in models
                    if m["ctx_len"] == ctx_len and not already_done(m["name"], ds, h)
                ]
                if not group:
                    continue
                try:
                    dm = MonashDataModule(
                        data_path=path,
                        context_length=ctx_len,
                        prediction_length=h,
                        batch_size=args.batch_size,
                        stride=h,
                        normalize_mode="per_series",
                        normalizer_type="identity",
                        clip_outliers=False,
                        train_val_test_split=(0.0, 0.0, 1.0),
                        num_workers=0,
                    )
                    dm.prepare_data()
                    dm.setup("fit")
                    loader = dm.test_dataloader()
                except Exception as e:
                    print(f"  h={h:<4} ctx={ctx_len} ❌ dataloader: {e}", flush=True)
                    for entry in group:
                        all_out[entry["name"]].setdefault(ds, {})[str(h)] = {"error": str(e)}
                    continue

                for entry in group:
                    name = entry["name"]
                    try:
                        row = {}
                        for mode, skip in (("legacy", True), ("fixed", False)):
                            c, p, t = run_one(
                                entry["model"], loader, h, device, skip, args.max_samples
                            )
                            scored = evaluate_with_baselines(c, p, t, season)
                            row[mode] = scored["timejepa"]
                            row[mode].update(scored.get("_skill", {}))
                            if mode == "fixed":
                                row["baselines"] = {
                                    k: v for k, v in scored.items()
                                    if k not in ("timejepa", "_skill")
                                }

                        lg, fx = row["legacy"], row["fixed"]
                        bl = row["baselines"]["seasonal_naive"]
                        print(
                            f"  h={h:<4} {name.split('/')[-1][:44]:<44} "
                            f"MSE {lg['mse']:7.3f} → {fx['mse']:7.3f}  "
                            f"MASE {lg.get('mase', float('nan')):6.2f} → {fx.get('mase', float('nan')):6.2f}  "
                            f"| SN {bl.get('mase', float('nan')):5.2f}  "
                            f"skill {fx.get('skill_vs_seasonal_naive', float('nan')):+6.1%}",
                            flush=True,
                        )
                        all_out[name].setdefault(ds, {})[str(h)] = row

                    except Exception as e:
                        print(f"  h={h:<4} {name} ❌ {e}", flush=True)
                        all_out[name].setdefault(ds, {})[str(h)] = {"error": str(e)}

                del dm, loader

            # Persist after every (dataset, horizon) so an interrupted CPU run
            # keeps everything computed so far.
            flush_results()

    flush_results()
    print("\n" + "=" * 100)
    print(f"✅ Done. Results → {out_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
