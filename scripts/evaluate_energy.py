#!/usr/bin/env python
"""
Prototype v0 — forecast par ÉNERGIE sur le benchmark Nixtla local.

    python scripts/evaluate_energy.py \\
        --checkpoint checkpoints/timejepa_lotsa_tiny_full/last.ckpt \\
        --decoder-checkpoint checkpoints/timejepa_lotsa_tiny_full_zs/pretrain_False/last.ckpt

Ce que le script mesure
-----------------------
La lecture « proposer-juger-pondérer » du doc JEPA-énergie, de bout en bout,
SANS décodeur ni finetune : le checkpoint de PRETRAIN seul.

  1. proposer : K block-bootstraps de l'historique + seasonal naive + drift ;
  2. juger    : E_k = 1 − cos(ẑ, enc([ctx‖cand_k])) — encodage CONTEXTUALISÉ
                (conclusion E18c : c'est un choix de LECTURE, valable même sur
                une lignée entraînée standalone) ;
  3. pondérer : w ∝ exp(−(E−μ_E)/σ_E) — softmax sur énergies standardisées par
                instance. v0 délibérément sans température libre : standardiser
                rend le poids invariant d'échelle, donc AUCUN hyperparamètre
                réglé sur le test. La calibration de T en contexte (conformal)
                est l'étape suivante, pas celle-ci ;
  4. lire     : quantiles pondérés (9 niveaux GIFT) par pas de temps ;
                point forecast = médiane pondérée.

Comparabilité — trois lecteurs dans le MÊME harnais
---------------------------------------------------
Les fenêtres, la saisonnalité, la MASE (poolée, helper du repo) et la WQL
(convention GluonTS, helper du repo) sont STRICTEMENT partagées entre :
    energy    le prototype ci-dessus (pretrain seul, zéro entraînement aval)
    decoder   la voie générative existante (checkpoint FINETUNÉ, --decoder-checkpoint)
    snaive    seasonal naive (point -> sa WQL s'effondre en ND, c'est attendu)
Comparer energy au registre expérimental se fait donc via les RATIOS vs snaive
du même run — jamais via les valeurs absolues d'un autre harnais.

Attentes honnêtes, écrites avant le run : le décodeur finetuné devrait gagner
en point forecast (il a vu un epoch de finetune, l'énergie zéro) ; la question
ouverte est la WQL — si les intervalles pondérés par énergie approchent ou
battent le fan du décodeur sans AUCUN entraînement aval, la lecture énergie a
prouvé sa valeur. Limite connue : le bootstrap ne propose que des
recombinaisons du passé (pas d'extrapolation hors enveloppe) — le tiers
« trajectoires du décodeur » de l'hybride n'est PAS dans cette v0.

Protocole fenêtres : non chevauchantes (stride = h) sur le split test converti
(data/processed/nixtla/), contexte 1024, h = 96 (une passe du prédicteur, pas
de rolling — la lecture énergie est single-shot par construction).
Lecture seule ; sorties console + JSON sous evaluation/energy_nixtla/.
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
# Lecture énergie
# ---------------------------------------------------------------------------

@torch.no_grad()
def energy_readout(model, ctx: np.ndarray, history: np.ndarray, h: int,
                   m: int, K: int, rng, device,
                   extra_cands: np.ndarray = None) -> np.ndarray:
    """
    Retourne le fan [h, 9] : quantiles pondérés par énergie.

    extra_cands [N, h] : candidats supplémentaires soumis au MÊME juge — le
    mode « hybride » (protocole utilisateur 2026-08-21) y met les trajectoires
    du fan du décodeur FINETUNÉ : le décodeur propose (il sait extrapoler hors
    de l'enveloppe historique, ce que le bootstrap ne sait pas — l'échec
    exchange d'E18d), le pretrain juge (son alignement énergie est intact,
    E18b). Deux checkpoints d'une même lignée en tandem, aucun modèle nouveau.
    """
    drift = ctx[-1] + (ctx[-1] - ctx[max(0, len(ctx) - m - 1)]) / max(m, 1) \
        * np.arange(1, h + 1, dtype=np.float32)
    sn = np.tile(ctx[-m:], (h + m - 1) // m + 1)[:h].astype(np.float32)
    block = max(8, min(m, h))
    pool = [sn, drift] + [block_bootstrap(history, h, block, rng) for _ in range(K)]
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

    n_tgt = (h - model.patching.patch_size) // model.patching.stride + 1
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb, num_targets=model.num_target_patches,
        w=(torch.ones(1, device=device)
           if hasattr(model.predictor, 'w_film') else None))[:, :n_tgt, :]

    full = torch.cat([ctx_norm.expand(xc_norm.shape[0], -1, -1), xc_norm], dim=1)
    z_cand = model.online_encoder(model.patching(full))[:, -n_tgt:, :]
    e = 1.0 - torch.nn.functional.cosine_similarity(
        z_cand.flatten(1), z_pred.expand_as(z_cand).flatten(1), dim=1)
    e = e.cpu().numpy()

    # Softmax sur énergies standardisées : invariant d'échelle, zéro réglage.
    z = (e - e.mean()) / max(e.std(), 1e-8)
    w = np.exp(-z); w /= w.sum()

    # Quantiles pondérés par pas de temps.
    fan = np.empty((h, len(QUANTILE_LEVELS)), dtype=np.float32)
    order = np.argsort(cands, axis=0)                       # [Nc, h]
    for t in range(h):
        idx = order[:, t]
        cum = np.cumsum(w[idx])
        vals = cands[idx, t]
        for qi, q in enumerate(QUANTILE_LEVELS):
            fan[t, qi] = vals[np.searchsorted(cum, q, side='left').clip(0, len(vals) - 1)]
    return fan


# ---------------------------------------------------------------------------
# Harnais commun
# ---------------------------------------------------------------------------

def iter_windows(series: np.ndarray, ctx_len: int, h: int, max_windows: int):
    """Fenêtres non chevauchantes (stride=h), réparties sur tout le split test."""
    starts = list(range(ctx_len, len(series) - h + 1, h))
    if len(starts) > max_windows:
        starts = [starts[i] for i in
                  np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    for s in starts:
        yield series[s - ctx_len:s].astype(np.float32), series[s:s + h].astype(np.float32), s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True, help="checkpoint de PRETRAIN (lecture énergie)")
    ap.add_argument("--decoder-checkpoint", default=None,
                    help="checkpoint FINETUNÉ (référence générative, même harnais)")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--max-windows", type=int, default=40, help="par série")
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
                         + (("decoder", "hybrid") if dec_model else ()))}
        for series in data:
            for ctx, tgt, _ in iter_windows(series, ctx_len, h, args.max_windows):
                if not (np.isfinite(ctx).all() and np.isfinite(tgt).all()):
                    continue
                fan = energy_readout(model, ctx, ctx, h, m, args.candidates, rng, device)
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
                    # Hybride : les 9 trajectoires-quantiles du décodeur entrent
                    # dans le pool et le pretrain les juge avec les autres.
                    acc["hybrid"]["fan"].append(energy_readout(
                        model, ctx, ctx, h, m, args.candidates, rng, device,
                        extra_cands=q.T))
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
    logger.info(f"JSON : {out}")


if __name__ == "__main__":
    main()
