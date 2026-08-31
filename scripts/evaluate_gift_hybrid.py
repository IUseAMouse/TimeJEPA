#!/usr/bin/env python
"""
G12 sur GIFT — TTM-R3 brut vs hybride « TTM propose, TimeJEPA juge ».

    python scripts/evaluate_gift_hybrid.py \\
        --judge-checkpoint checkpoints/timejepa_lotsa_tiny_v3/pretrain_True/<ckpt> \\
        --judge-config lotsa_tiny_v3_eval

STATUT : expérience de PAPIER (chapitre G12), JAMAIS un chiffre officiel — un
hybride bi-modèle est au-delà de la ligne G11 (décision utilisateur 2026-08-24).
Conçu 2026-08-28 (question utilisateur : porter le duo evaluate_energy/TTM sur
GIFT). Le harnais leaderboard (evaluate_gift.py) n'est PAS touché.

Trois lectures, trois statuts :
  * `ttm`        — TTM-R3 brut sur les fenêtres GIFT. Sa MASE ABSOLUE se
                   compare au CSV officiel TTM-R3-PT vendoré (validation
                   croisée du harnais + part « pipeline » de leur score).
                   ⚠️ son CRPS est un point-forecast effondré : NE PAS le citer.
  * `hybrid_ttm` — pool bootstrap + SN + drift + chemins TTM jitterés, jugé
                   par NOTRE pretrain (encodage contextualisé, E18c), lu en
                   quantiles pondérés. LA mesure : un point forecaster externe
                   devient probabiliste (fan + couverture) par juge latent.
  * référence champion : PAS recalculée ici — citer les évals complètes
                   existantes (meilleure statistique que nos fenêtres plafonnées).

Appariement : les DEUX readers voient exactement les mêmes instances
(sous-échantillonnage régulier, --instances par config) — les comparaisons
internes sont appariées ; les valeurs absolues sur fenêtres plafonnées ne
remplacent pas une éval complète.

Rollout TTM pour h > 96 : réinjection autorégressive segment par segment
(jitter sur le PREMIER segment seulement — la diversité naît au départ, les
segments suivants prolongent chaque chemin de façon déterministe).

Choix du juge (à déclarer dans le papier) :
  * primaire   = checkpoint FINAL nommé du pretrain (zéro sélection) ;
  * secondaire = juge sélectionné par probe (loi pic-tôt) — le probe lit des
    données GIFT, donc cette colonne est étiquetée « sélection déclarée ».

Sorties : evaluation/gift_hybrid/<judge>/{per_config/*.json, summary.json}
(même format que le harnais officiel, couverture incluse).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift                                # noqa: E402
from evaluate_gift import prepare_context                           # noqa: E402
from evaluate_energy import (TTMProposer, candidate_energies,       # noqa: E402
                             fan_from_energies)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gift_hybrid")


def ttm_rollout(prop: TTMProposer, ctx: np.ndarray, h: int,
                n_jitter: int, rng) -> np.ndarray:
    """[1+n_jitter, h] — chemins TTM, autorégressifs par segments de pred_len."""
    ctxs, segs, rem = None, [], h
    while rem > 0:
        step = min(prop.pred_len, rem)
        if ctxs is None:
            p = prop.paths(ctx, step, n_jitter, rng)          # [N, step]
            ctxs = [np.concatenate([ctx, p[i]]) for i in range(len(p))]
        else:
            p = np.stack([prop.paths(c, step, 0, rng)[0] for c in ctxs])
            ctxs = [np.concatenate([ctxs[i], p[i]]) for i in range(len(p))]
        segs.append(p)
        rem -= step
    return np.concatenate(segs, axis=1)


def evaluate_config(config, judge, prop, gift_root, device, rng,
                    max_inst, K, n_jitter, centered=False):
    h = gift.prediction_length(config)
    m = gift.seasonality(config.split("/")[1])
    series = gift.load_series(gift_root, config)
    windows = gift.num_windows(config, min(len(s) for s in series))

    total = sum(1 for _ in gift.iter_test_instances(series, h, windows))
    stride = max(1, total // max_inst)

    accs = {r: gift.MetricAccumulator() for r in ("ttm", "hybrid_ttm")}
    sn_acc = gift.MetricAccumulator()
    n_used, n_ttm_nonfinite = 0, 0
    for i, inst in enumerate(gift.iter_test_instances(series, h, windows)):
        if i % stride or n_used >= max_inst:
            continue
        if np.isnan(inst.target).any():
            continue
        ctx = prepare_context(inst.context, judge.input_length,
                              judge.patching.stride, judge.patching.patch_size)
        if ctx is None:
            continue
        scale = gift.seasonal_error(inst.context, m)

        tp = ttm_rollout(prop, inst.context.astype(np.float32), h, n_jitter, rng)
        # TTM émet des NaN sur certains contextes extrêmes (variance ~0,
        # bitbrains) : chemin propre non fini -> instance sautée pour les DEUX
        # readers (l'appariement prime) ; jitters non finis simplement écartés.
        if not np.isfinite(tp[0]).all():
            n_ttm_nonfinite += 1
            continue
        tp = tp[np.isfinite(tp).all(axis=1)]
        accs["ttm"].add(inst.target, tp[0], None, scale)      # chemin propre, point

        # h_judge : l'énergie se lit sur les premiers pas <= horizon natif du
        # juge (single-shot par construction) ; les quantiles sur h complet.
        cands, e = candidate_energies(judge, ctx, inst.context, h, m, K,
                                      rng, device, extra_cands=tp,
                                      h_judge=judge.prediction_length,
                                      centered=centered)
        fan = fan_from_energies(cands, e)                     # [h, 9]
        accs["hybrid_ttm"].add(inst.target, fan[:, 4], fan, scale)

        sn_acc.add(inst.target, gift.seasonal_naive_forecast(inst.context, h, m),
                   None, scale)
        n_used += 1

    out = {"config": config, "h": h, "n_instances": n_used,
           "n_ttm_nonfinite_skipped": n_ttm_nonfinite,
           "seasonal_naive_local": sn_acc.result()}
    for r, acc in accs.items():
        out[r] = acc.result()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--judge-checkpoint", required=True)
    ap.add_argument("--judge-config", default="lotsa_tiny_v3_eval")
    ap.add_argument("--ttm-model", default="ibm-granite/granite-timeseries-ttm-r3")
    ap.add_argument("--ttm-revision", default="1024-96-r3")
    ap.add_argument("--ttm-jitter", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=24, help="bootstraps du pool")
    ap.add_argument("--instances", type=int, default=150, help="par config, appariées")
    ap.add_argument("--configs", default=None,
                    help="sous-ensemble GIFT 'a/b/c,d/e/f' (défaut : les 97)")
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centered-bootstrap", action="store_true",
                    help="G12c : bootstrap des innovations saisonnières recollé "
                         "sur le chemin TTM (anti-dilution par construction)")
    ap.add_argument("--tag", default=None,
                    help="suffixe du répertoire de sortie — OBLIGATOIRE pour "
                         "une variante de pool (sinon collision de marqueurs "
                         "avec le run de base du même juge)")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.judge_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge = create_model_from_config(cfg)
    judge = load_checkpoint(judge, args.judge_checkpoint, device)
    judge.to(device).eval()
    prop = TTMProposer(args.ttm_model, device, revision=args.ttm_revision)

    configs = ([c.strip() for c in args.configs.split(",")] if args.configs
               else list(gift.GIFT_CONFIGS))
    rng = np.random.default_rng(args.seed)
    out_dir = (Path("evaluation") / "gift_hybrid"
               / (Path(args.judge_checkpoint).stem
                  + (f"_{args.tag}" if args.tag else "")))
    (out_dir / "per_config").mkdir(parents=True, exist_ok=True)

    results = {}
    for i, config in enumerate(configs, 1):
        marker = out_dir / "per_config" / (config.replace("/", "__") + ".json")
        if marker.exists():
            results[config] = json.loads(marker.read_text())
            logger.info(f"[{i}/{len(configs)}] {config}: déjà fait, sauté")
            continue
        t0 = time.time()
        try:
            res = evaluate_config(config, judge, prop, Path(args.gift_root),
                                  device, rng, args.instances,
                                  args.candidates, args.ttm_jitter,
                                  centered=args.centered_bootstrap)
        except Exception as exc:   # une config cassée ne tue pas les 96 autres
            logger.error(f"[{i}/{len(configs)}] {config} FAILED: "
                         f"{type(exc).__name__}: {exc}")
            continue
        marker.write_text(json.dumps(res, indent=1))
        results[config] = res
        t, hy, sn = res["ttm"], res["hybrid_ttm"], res["seasonal_naive_local"]
        cov = hy.get("coverage") or {}
        logger.info(
            f"[{i}/{len(configs)}] {config}: ttm MASE {t['MASE']:.3f} | "
            f"hybrid MASE {hy['MASE']:.3f} CRPS {hy['CRPS']:.3f} "
            f"couv80 {(cov.get('0.9', 0) - cov.get('0.1', 0)):.2f} "
            f"({res['n_instances']} inst, {time.time() - t0:.0f}s)")

    # Agrégats : geomean des ratios vs SN LOCAL (mêmes fenêtres — apparié).
    def geomean_ratio(metric, reader):
        ratios = [results[c][reader][metric] / results[c]["seasonal_naive_local"][metric]
                  for c in results
                  if results[c]["seasonal_naive_local"][metric] > 0]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")

    summary = {"judge": args.judge_checkpoint, "ttm": args.ttm_model,
               "revision": args.ttm_revision, "instances_cap": args.instances,
               "aggregates_vs_local_sn": {
                   r: {m: geomean_ratio(m, r) for m in ("MASE", "CRPS")}
                   for r in ("ttm", "hybrid_ttm")}}
    covs = [results[c]["hybrid_ttm"].get("coverage") for c in results
            if results[c]["hybrid_ttm"].get("coverage")]
    if covs:
        summary["hybrid_coverage80_mean"] = float(
            np.mean([c["0.9"] - c["0.1"] for c in covs]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    for r in ("ttm", "hybrid_ttm"):
        a = summary["aggregates_vs_local_sn"][r]
        note = "  (CRPS point effondré — ne pas citer)" if r == "ttm" else ""
        logger.info(f"  {r:11s} MASE ratio {a['MASE']:.4f} | "
                    f"CRPS ratio {a['CRPS']:.4f}{note}")
    if covs:
        logger.info(f"  couverture 80% hybride : "
                    f"{summary['hybrid_coverage80_mean']:.3f} (nominal 0.800)")
    logger.info(f"Sorties : {out_dir}")


if __name__ == "__main__":
    main()
