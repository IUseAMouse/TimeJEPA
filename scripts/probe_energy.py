#!/usr/bin/env python
"""
Sonde « lecture par énergie » du JEPA — le veto avant toute construction.

    python scripts/probe_energy.py \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/last.ckpt

Hypothèse testée
----------------
Un JEPA pré-entraîné est une fonction d'ÉNERGIE entraînée : le pretrain minimise
distance(pred(ctx), enc(futur_vrai)) pendant que SIGReg maintient les encodages
des autres futurs dispersés. Si cette énergie discrimine, on peut lire le modèle
SANS décodeur : proposer K futurs candidats réalistes (block-bootstrap de
l'historique de la série — vraies queues, vrais zéros, vraies rafales), les
encoder, et les classer par distance latente au futur prédit. Intervalle de
confiance = quantiles pondérés par softmax(-E/T). Aucun entraînement.

Le test (une après-midi, CPU) : pour des instances GIFT, générer K candidats
bootstrap + le seasonal naive + LE FUTUR VRAI, et mesurer le RANG du vrai dans
le classement d'énergie.
  * rang normalisé ~0.5 = l'énergie classe au hasard -> dossier clos, un jour.
  * rang systématiquement bas = le latent JEPA « sait » des choses que le
    décodeur ne lit pas -> l'édifice (re-notation, intervalles) mérite d'être
    construit. C'est aussi l'argument central du papier, mesuré.
Second témoin : Spearman(énergie, MAE(candidat, vérité)) parmi les bootstraps —
l'énergie doit être corrélée à la proximité réelle, pas seulement repérer le
vrai par un artefact.

Protocole d'encodage — répliqué du pretrain, pas réinventé
----------------------------------------------------------
Les cibles du pretrain (lignée tiny-full : contextualized_targets=true) sont
encodées comme [ctx‖cible] normalisés AUX STATS DU CONTEXTE puis tranchées
(jepa_tst.forward_pretrain). La sonde fait exactement pareil pour chaque
candidat, robust_scale compris quand le checkpoint le porte. L'énergie est
mesurée sur les n premiers patches (l'horizon GIFT est plus court que les 256
pas du prédicteur — même troncature que forecast()).

Deux approximations, dites plutôt que cachées :
  * l'encodeur ONLINE remplace le target encoder (le chargeur d'éval saute
    l'EMA) — en fin de pretrain les deux sont proches ;
  * sur un checkpoint de FINETUNE, plus rien n'ancre pred(ctx) ~ enc(futur)
    (la pinball seule entraîne) : un rang dégradé pretrain -> finetune mesure
    le DRIFT du full finetune, pas un échec du pretrain. Comparer les deux
    checkpoints est le sous-résultat le plus intéressant de la sonde.

Sorties : tableau console + JSON sous evaluation/probe_energy/.
Ne modifie rien : aucun code d'entraînement touché, lecture seule.
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
from timejepa.evaluation import gift                               # noqa: E402
from evaluate_gift import prepare_context                          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("probe_energy")

# Trois configs de la queue E18 (l'énergie doit-elle reconnaître les
# morphologies que le décodeur ne sait pas produire ?) + trois où le modèle
# est bon (l'énergie doit y être franchement discriminante, sinon rien à lire).
DEFAULT_CONFIGS = [
    "solar/10T/short",
    "bizitobs_l2c/5T/short",
    "electricity/15T/short",
    "electricity/H/short",
    "jena_weather/H/short",
    "sz_taxi/15T/short",
]


def block_bootstrap(history: np.ndarray, h: int, block: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Un futur candidat : blocs contigus de l'historique, recollés jusqu'à h."""
    finite = history[np.isfinite(history)]
    if len(finite) < block + 1:
        finite = np.resize(finite, block + 1)
    out = np.empty(h, dtype=np.float32)
    pos = 0
    while pos < h:
        start = rng.integers(0, len(finite) - block)
        take = min(block, h - pos)
        out[pos:pos + take] = finite[start:start + take]
        pos += take
    return out


@torch.no_grad()
def probe_instance(model, ctx: np.ndarray, candidates: np.ndarray, device,
                   standalone: bool = False):
    """
    ctx        [L]        contexte préparé (même chemin que l'éval GIFT)
    candidates [Nc, h]    futurs candidats, le VRAI en position 0
    standalone : encoder le candidat SEUL (convention des lignées
                 contextualized_targets=false — mix/xres) au lieu de la
                 tranche de [ctx‖cand]. Utiliser la convention du pretrain
                 du checkpoint sondé, sinon l'énergie est interrogée hors
                 de son régime d'entraînement.
    Retourne (energies_mse, energies_cos) [Nc].
    """
    h = candidates.shape[1]
    n_tgt = (h - model.patching.patch_size) // model.patching.stride + 1

    x_ctx = torch.from_numpy(ctx).reshape(1, -1, 1).to(device)
    cands = torch.from_numpy(candidates).unsqueeze(-1).to(device)   # [Nc, h, 1]

    # G8.4 — mêmes stats (celles du contexte) pour le contexte ET les candidats.
    if model.robust_scaler is not None:
        model.robust_scaler.fit(x_ctx)
        x_ctx = model.robust_scaler.transform(x_ctx)
        cands = model.robust_scaler.transform(cands)

    # RevIN : stats du contexte, appliquées aux candidats — la convention
    # exacte de forward_pretrain (cible normalisée aux stats du contexte).
    ctx_norm = model.revin(x_ctx, mode='norm') if model.revin is not None else x_ctx
    if model.revin is not None:
        cands_norm = (cands - model.revin.mean) / model.revin.std
    else:
        cands_norm = cands

    # z_pred : le chemin de forward_finetune, tronqué à l'horizon du candidat.
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb,
        num_targets=model.num_target_patches,
        w=(torch.ones(1, device=device)
           if hasattr(model.predictor, 'w_film') else None),
    )[:, :n_tgt, :]                                                  # [1, n, D]

    # z_cand : selon la convention de cible du pretrain sondé.
    if standalone:
        z_cand = model.online_encoder(model.patching(cands_norm))
    else:
        full = torch.cat([ctx_norm.expand(cands_norm.shape[0], -1, -1),
                          cands_norm], dim=1)                        # [Nc, L+h, 1]
        z_cand = model.online_encoder(model.patching(full))[:, -n_tgt:, :]

    diff = z_cand - z_pred                                           # [Nc, n, D]
    e_mse = diff.pow(2).mean(dim=(1, 2))
    cos = torch.nn.functional.cosine_similarity(
        z_cand.flatten(1), z_pred.expand_as(z_cand).flatten(1), dim=1)
    return e_mse.cpu().numpy(), (1.0 - cos).cpu().numpy()


def normalized_rank(energies: np.ndarray) -> float:
    """Rang du vrai (position 0) dans le classement, dans [0, 1] ; 0 = meilleur."""
    return float((energies < energies[0]).sum()) / (len(energies) - 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="configs GIFT, séparées par des virgules")
    ap.add_argument("--instances", type=int, default=100, help="max par config")
    ap.add_argument("--candidates", type=int, default=32, help="bootstraps par instance")
    ap.add_argument("--gift-root", default="data/gift_eval")
    ap.add_argument("--model-config", default="lotsa_tiny_eval")
    ap.add_argument("--standalone-targets", action="store_true",
                    help="convention des lignées contextualized_targets=false (mix/xres)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    gift_root = Path(args.gift_root)
    results = {}

    for config in args.configs.split(","):
        config = config.strip()
        h = gift.prediction_length(config)
        m = gift.seasonality(config.split("/")[1])
        block = max(8, min(m, h))
        series = gift.load_series(gift_root, config)
        windows = gift.num_windows(config, min(len(s) for s in series))

        ranks_mse, ranks_cos, spears = [], [], []
        for inst in gift.iter_test_instances(series, h, windows):
            if len(ranks_mse) >= args.instances:
                break
            if np.isnan(inst.target).any() or len(inst.context) < 2 * block:
                continue
            ctx = prepare_context(inst.context, model.input_length,
                                  model.patching.stride, model.patching.patch_size)
            if ctx is None:
                continue

            cands = np.stack(
                [inst.target.astype(np.float32)]
                + [gift.seasonal_naive_forecast(inst.context, h, m).astype(np.float32)]
                + [block_bootstrap(inst.context, h, block, rng)
                   for _ in range(args.candidates)])
            if not np.isfinite(cands).all():
                continue

            e_mse, e_cos = probe_instance(model, ctx, cands, device,
                                          standalone=args.standalone_targets)
            ranks_mse.append(normalized_rank(e_mse))
            ranks_cos.append(normalized_rank(e_cos))
            # L'énergie suit-elle la proximité réelle ? (bootstraps seulement)
            mae = np.abs(cands[2:] - cands[0]).mean(axis=1)
            spears.append(spearman(e_cos[2:], mae))

        r_mse, r_cos = np.array(ranks_mse), np.array(ranks_cos)
        results[config] = {
            "n_instances": len(r_mse),
            "mean_rank_mse": float(r_mse.mean()),
            "mean_rank_cos": float(r_cos.mean()),
            "median_rank_cos": float(np.median(r_cos)),
            "frac_truth_top20pct_cos": float((r_cos <= 0.2).mean()),
            "spearman_energy_vs_mae": float(np.mean(spears)),
        }
        r = results[config]
        logger.info(
            f"{config:28s} n={r['n_instances']:3d} | rang vrai (cos) "
            f"moy {r['mean_rank_cos']:.3f} méd {r['median_rank_cos']:.3f} "
            f"top20% {r['frac_truth_top20pct_cos']:.2f} | mse {r['mean_rank_mse']:.3f} "
            f"| rho(E,MAE) {r['spearman_energy_vs_mae']:.3f}   [hasard: 0.50 / 0.20 / 0.00]")

    agg = {k: float(np.mean([r[k] for r in results.values()]))
           for k in ("mean_rank_cos", "mean_rank_mse",
                     "frac_truth_top20pct_cos", "spearman_energy_vs_mae")}
    logger.info(f"\nAGRÉGAT {len(results)} configs : rang vrai (cos) "
                f"{agg['mean_rank_cos']:.3f} | top20% {agg['frac_truth_top20pct_cos']:.2f} "
                f"| rho(E,MAE) {agg['spearman_energy_vs_mae']:.3f}")

    out_dir = Path("evaluation/probe_energy")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_standalone" if args.standalone_targets else ""
    out = out_dir / f"{Path(args.checkpoint).stem}{suffix}.json"
    out.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "candidates": args.candidates,
         "per_config": results, "aggregate": agg}, indent=2))
    logger.info(f"JSON : {out}")


if __name__ == "__main__":
    main()
