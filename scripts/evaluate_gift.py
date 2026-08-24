#!/usr/bin/env python
"""
GIFT-Eval harness — evaluates a TimeJEPA checkpoint on the official 97 configs.

    # 1. download the benchmark data (once, ~a few GB)
    make gift-download

    # 2. run (CPU is fine for the tiny models — that is their point)
    python scripts/evaluate_gift.py --config-name lotsa_tiny_eval \\
        +checkpoint_path=checkpoints/timejepa_lotsa_tiny_zs/pretrain_False/<ckpt>

    # useful extras
    #   +gift_configs='electricity/H/short,us_births/D/short'  restrict the set
    #   +gift_terms=short                                      one term only
    #   +gift_max_series=50                                    debug subsample

The protocol lives in src/timejepa/evaluation/gift.py, transcribed constant by
constant from the official harness — read its module docstring before touching
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
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift  # noqa: E402

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
    * NaNs are linearly interpolated (edge-filled at the ends) — model INPUT
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
# TTA — procédures d'inférence UNIFORMES sur les 97 configs (roadmap S1.5,
# décision utilisateur 2026-08-24 : TTA oui / multi-checkpoints non).
# Un seul checkpoint ; chaque variante est une transformation déterministe du
# contexte, identique pour toutes les configs — pas d'adaptation par config.
# ---------------------------------------------------------------------------

def _negated(out: dict) -> dict:
    """Miroir par inversion de signe : q_k(-X) = -q_{1-k}(X) — la médiane se
    nie, le fan se nie ET se renverse sur l'axe des niveaux (l'ordre croissant
    est préservé). Précédents : TimesFM `force_flip_invariance`, YingLong.
    Bien défini sous arcsinh/RevIN : les deux repères sont impairs autour de
    leur centre (médiane/moyenne du contexte nié = négation du centre)."""
    res = {"forecast_denorm": -out["forecast_denorm"]}
    q = out.get("quantiles_denorm")
    if q is not None:
        qdim = -2 if q.dim() == 4 else -1
        res["quantiles_denorm"] = torch.flip(-q, dims=(qdim,))
    if "quantile_levels" in out:
        res["quantile_levels"] = out["quantile_levels"]
    return res


def tta_forecast(model, batch: torch.Tensor, h: int,
                 lookbacks=None, flip: bool = False) -> dict:
    """
    Moyenne de variantes d'inférence du MÊME checkpoint :
      * multi-lookback : le contexte tronqué à L' pour chaque L' de `lookbacks`
        (borné à la longueur disponible, aligné sur le stride) ;
      * miroir sign-flip optionnel sur chaque lookback.
    Les fans de quantiles sont moyennés niveau à niveau (la moyenne de
    vecteurs triés reste triée) ; la médiane est relue dans le fan moyenné.
    Sans lookbacks ni flip : strictement équivalent à model.forecast.
    """
    L = batch.shape[1]
    stride = model.patching.stride
    looks = sorted({min(int(lb), L) - (min(int(lb), L) % stride)
                    for lb in (lookbacks or [L])
                    if min(int(lb), L) >= model.patching.patch_size})
    if not looks:
        looks = [L]

    outs = []
    for lb in looks:
        ctx = batch[:, -lb:]
        outs.append(model.forecast(ctx, n=h))
        if flip:
            outs.append(_negated(model.forecast(-ctx, n=h)))

    if len(outs) == 1:
        return outs[0]

    res = {"forecast_denorm":
           torch.stack([o["forecast_denorm"] for o in outs]).mean(0)}
    fans = [o.get("quantiles_denorm") for o in outs]
    if all(f is not None for f in fans):
        fan = torch.stack(fans).mean(0)
        res["quantiles_denorm"] = fan
        levels = next((o["quantile_levels"] for o in outs
                       if "quantile_levels" in o), None)
        if levels is not None:
            mid = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
            qdim = -2 if fan.dim() == 4 else -1
            res["forecast_denorm"] = fan.select(qdim, mid).unsqueeze(qdim) \
                if fan.dim() == 4 else fan[..., mid:mid + 1]
            res["quantile_levels"] = levels
    return res


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(model, config: str, gift_root: Path, device,
                    batch_size: int, max_series: int = 0,
                    max_context: int = 0, tta_lookbacks=None,
                    tta_flip: bool = False) -> dict:
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
    # G9.1 — max_context peut DÉPASSER model.input_length : l'encodeur est
    # RoPE (agnostique à la longueur), le prédicteur ne dépend que du nombre
    # de requêtes cibles. C'est le test « contexte 2048 à l'inférence seule ».
    max_len = max_context or (model.input_length
                              if hasattr(model, "input_length") else 1024)
    stride = model.patching.stride
    n_inst = 0
    for inst in gift.iter_test_instances(series, h, windows):
        ctx = prepare_context(inst.context, max_len, stride, model.patching.patch_size)
        if ctx is None:
            continue
        # MASE scale uses the FULL past, not the capped context — gluonts
        # computes the seasonal error on the entire history of the series.
        scale = gift.seasonal_error(inst.context, m)
        buckets[len(ctx)].append((ctx, inst.target, inst.context, scale))
        n_inst += 1

    for length, items in sorted(buckets.items()):
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            batch = torch.from_numpy(np.stack([c[0] for c in chunk]))
            batch = batch.unsqueeze(-1).to(device)          # [B, L, 1]

            with torch.no_grad():
                out = tta_forecast(model, batch, h,
                                   lookbacks=tta_lookbacks, flip=tta_flip)

            median = out["forecast_denorm"].squeeze(-1).cpu().numpy()  # [B, h]
            quants = out.get("quantiles_denorm")
            if quants is not None:
                quants = quants.cpu().numpy()
                if quants.ndim == 4:                        # [B, h, Q, 1]
                    quants = quants[..., 0]
                # -> [B, h, Q]

            for b, (ctx, target, past, scale) in enumerate(chunk):
                model_acc.add(target, median[b],
                              quants[b] if quants is not None else None, scale)
                sn = gift.seasonal_naive_forecast(past, h, m)
                sn_acc.add(target, sn, None, scale)

    res = {"config": config, "prediction_length": h, "seasonality": m,
           "windows": windows, "n_series": len(series), "n_instances": n_inst,
           "model": model_acc.result(), "seasonal_naive_local": sn_acc.result()}
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

    # TTA / contexte long (roadmap S1) — opt-in, défauts = comportement exact
    # d'avant. Exemples :
    #   +max_context=2048                       (G9.1, inférence seule)
    #   +tta_lookbacks='512,1024'               (multi-lookback moyenné)
    #   +tta_flip=true                          (miroir sign-flip)
    max_context = int(cfg.get("max_context", 0))
    tta_lookbacks = ([int(x) for x in str(cfg.tta_lookbacks).split(",")]
                     if cfg.get("tta_lookbacks") else None)
    tta_flip = bool(cfg.get("tta_flip", False))

    configs = [c for c in gift.GIFT_CONFIGS
               if (not only or c in only)
               and (not terms or c.split("/")[2] in terms)]
    logger.info(f"{len(configs)} configs to evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, checkpoint_path, device)

    # Chaque variante d'inférence a son PROPRE répertoire : les JSON per_config
    # servent de marqueurs de reprise, mélanger deux procédures dans le même
    # cache relirait les chiffres de l'une comme s'ils mesuraient l'autre.
    tag = ""
    if max_context:
        tag += f"_ctx{max_context}"
    if tta_lookbacks:
        tag += "_lb" + "-".join(str(x) for x in tta_lookbacks)
    if tta_flip:
        tag += "_flip"
    out_dir = (Path("evaluation") / cfg.model.name
               / Path(checkpoint_path).stem / f"gift{tag}")
    per_config = out_dir / "per_config"
    per_config.mkdir(parents=True, exist_ok=True)

    # Piège de reprise mesuré (2026-08-20) : le répertoire est indexé sur le NOM
    # du checkpoint, et `last.ckpt` est écrasé au fil du run — une ré-éval de
    # `last` relisait les JSON de l'ancien checkpoint en affichant « already
    # done » sur 97 configs, c'est-à-dire un agrégat de l'ANCIEN modèle présenté
    # comme une mesure du nouveau. L'empreinte (mtime+taille) rend le cas
    # bruyant. Refus plutôt que nettoyage : rien n'est jamais supprimé ici.
    ckpt_stat = Path(checkpoint_path).stat()
    fingerprint = f"{ckpt_stat.st_mtime_ns}:{ckpt_stat.st_size}"
    fp_file = out_dir / "checkpoint_fingerprint.txt"
    if fp_file.exists() and fp_file.read_text().strip() != fingerprint:
        raise RuntimeError(
            f"{out_dir} contient les résultats d'une AUTRE version de "
            f"{Path(checkpoint_path).name} (empreinte différente — le fichier a "
            f"été écrasé depuis, typiquement un last.ckpt en cours de run). "
            f"Reprendre relirait les anciens JSON comme s'ils mesuraient le "
            f"nouveau checkpoint. Déplacer ou renommer ce répertoire, puis relancer."
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
            res = evaluate_config(model, config, gift_root, device,
                                  batch_size, max_series,
                                  max_context=max_context,
                                  tta_lookbacks=tta_lookbacks,
                                  tta_flip=tta_flip)
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
        logger.info(
            f"[{i}/{len(configs)}] {config}: MASE {mm['MASE']:.3f} "
            f"CRPS {mm['CRPS']:.3f} ({res['n_instances']} inst, "
            f"{time.time() - t0:.0f}s)")

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
    logger.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
