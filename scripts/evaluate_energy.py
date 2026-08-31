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
def candidate_energies(model, ctx: np.ndarray, history: np.ndarray, h: int,
                       m: int, K: int, rng, device,
                       extra_cands: np.ndarray = None,
                       refine_steps: int = 0, refine_lr: float = 0.05,
                       h_judge: int = None, centered: bool = False):
    """
    Construit le pool de candidats et calcule leurs énergies -> (cands, e).

    extra_cands [N, h] : candidats supplémentaires soumis au MÊME juge — le
    mode « hybride » (protocole utilisateur 2026-08-21) y met les trajectoires
    du fan du décodeur FINETUNÉ : le décodeur propose (il sait extrapoler hors
    de l'enveloppe historique, ce que le bootstrap ne sait pas — l'échec
    exchange d'E18d), le pretrain juge (son alignement énergie est intact,
    E18b). Deux checkpoints d'une même lignée en tandem, aucun modèle nouveau.

    h_judge : fenêtre de jugement (bug G12-sur-GIFT, 2026-08-31). La lecture
    énergie est SINGLE-SHOT par construction (ẑ = num_target_patches, h natif
    256) ; sur GIFT, h va de 6 à 720. Quand h_judge est fourni, l'énergie est
    calculée sur les h_judge PREMIERS pas de chaque candidat (les candidats
    retournés restent pleine longueur pour la lecture des quantiles). Lecture
    honnête : la diversité des chemins TTM naît au premier segment (jitter
    premier segment seulement), donc juger le début capture leur écart ; pour
    h < patch_size, le candidat est paddé au bord POUR L'ENCODAGE SEUL (la
    transformation étant ponctuelle, padder après normalisation = normaliser
    le paddé). None = comportement bit-identique à l'existant (Nixtla, h=96).
    """
    drift = ctx[-1] + (ctx[-1] - ctx[max(0, len(ctx) - m - 1)]) / max(m, 1) \
        * np.arange(1, h + 1, dtype=np.float32)
    sn = np.tile(ctx[-m:], (h + m - 1) // m + 1)[:h].astype(np.float32)
    # Deux échelles de blocs : la saisonnalité entière (structure de cycle) et
    # un sous-bloc (textures locales) — diversifie le pool sans rien apprendre.
    blocks = [max(8, min(m, h)), max(8, min(m, h) // 3)]
    if centered and extra_cands is not None:
        # G12c (2026-08-31) — bootstrap CENTRÉ SUR LE PROPOSEUR. Le diagnostic
        # de la dilution (4 réplications) est un problème de CENTRE, pas
        # d'étalement : un bootstrap de l'historique brut est centré sur le
        # PASSÉ, et chaque gramme de poids qu'il reçoit tire la médiane
        # pondérée hors de la tendance du proposeur. Ici : candidats = chemin
        # propre du proposeur + blocs rééchantillonnés des INNOVATIONS
        # saisonnières (y_t − y_{t−m}) — tous les candidats partagent le
        # centre, la dilution du centre est impossible PAR CONSTRUCTION, et le
        # juge n'arbitre que ce qu'un vérificateur doit toucher : texture et
        # étalement. Le drift (l'ancre qui a empoisonné le run ttmonly sur les
        # configs D/W) sort du pool ; sn reste comme centre alternatif légitime.
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

    # Fenêtre de jugement : l'énergie se calcule sur les hj premiers pas
    # (padde au bord si plus court qu'un patch) ; cands (pleine longueur)
    # porte la lecture des quantiles. La tranche elle-même est prise APRÈS le
    # bloc refine (qui réassigne xc_norm).
    hj = h if h_judge is None else min(h_judge, h)
    if refine_steps > 0 and hj < h:
        raise ValueError("refine_steps avec h_judge < h raffinerait des "
                         "candidats via une énergie tronquée — non défini.")
    P = model.patching.patch_size
    hj_enc = max(hj, P)
    n_tgt = (hj_enc - P) // model.patching.stride + 1
    ctx_emb = model.online_encoder(model.patching(ctx_norm))
    z_pred = model.predictor.forward_simple(
        context_embeddings=ctx_emb, num_targets=model.num_target_patches,
        w=(torch.ones(1, device=device)
           if hasattr(model.predictor, 'w_film') else None))[:, :n_tgt, :]

    # Raffinement par gradient (« planning by backprop ») : l'énergie est
    # différentiable en y — quelques pas de descente SUR LES CANDIDATS
    # eux-mêmes les glissent vers la vallée la plus proche du paysage. C'est
    # du test-time compute pur, aucun poids modifié. Garde-fou Goodhart : peu
    # de pas, petit lr — trop d'optimisation fabriquerait des candidats
    # adversariaux qui minimisent E sans ressembler à un futur.
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
        # Retour à l'espace brut pour la lecture des quantiles : inverse RevIN
        # (stats du contexte) puis inverse arcsinh si le checkpoint le porte.
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
    Énergies -> poids de Gibbs -> quantiles pondérés [h, 9]. T s'applique aux
    énergies STANDARDISÉES (l'invariance d'échelle de la v0 est conservée) :
    T=1 = comportement v0 ; T PETIT = juge contrasté, la masse se concentre sur
    les meilleurs candidats — le remède à la dilution mesurée trois fois
    (E18e / e-v2 / g, toujours weather+exchange) ; T grand = pool ~uniforme.
    """
    # Garde de dégénérescence (mesuré, run ttmonly 2026-08-31) : un pool de
    # candidats quasi identiques donne des énergies quasi identiques, et la
    # division par un std minuscule transforme le softmax en amplificateur de
    # bruit (masse posée sur une ancre arbitraire — ett1/D MASE 1.63→5.29).
    # Sous le seuil, l'information d'énergie est du bruit : poids uniformes.
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
    """Le pipeline complet : pool + énergies + lecture pondérée à T donné."""
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
    Calibration de T EN CONTEXTE — le prérequis G12(a), et le levier désigné
    par trois réplications de la signature de dilution. On rejoue le pipeline
    COMPLET (pool réel, proposeur compris) sur n_cal sous-fenêtres passées du
    contexte dont la suite est connue ; le T retenu minimise la pinball
    moyenne. Les énergies se calculent UNE fois par sous-fenêtre, balayer la
    grille ne coûte que des softmax. Zéro entraînement, zéro regard sur le
    test ; rng dédié (seed décalée) pour que les tirages du chemin principal
    restent appariés au bit près avec les runs non calibrés.
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
    n trajectoires épistémiques du décodeur FINETUNÉ : les modules Dropout du
    modèle sont basculés en mode train LE TEMPS DES FORWARDS (le reste — norm,
    EMA — reste en eval), puis restaurés. Chaque forward stochastique donne un
    médian différent = une trajectoire COHÉRENTE temporellement, contrairement
    aux chemins-quantiles (marginales). Uniquement dans le script — le cœur du
    code n'est pas touché.
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
# Harnais commun
# ---------------------------------------------------------------------------

class TTMProposer:
    """
    Proposeur externe G12(b) : TTM-R3 (IBM Granite, ~1.4M — le rival de classe
    de taille, CRPS 0.520 sur GIFT). Point forecast only -> la diversité vient
    de N contextes jitterés (bruit 0.05·std) en plus du chemin propre. Chargé
    paresseusement : le script reste utilisable sans granite-tsfm installé.
    La mesure G12 est l'UPLIFT : reader `ttm` seul vs `hybrid_ttm` (bootstrap
    + SN + drift + chemins TTM, jugés par NOTRE pretrain).
    """

    def __init__(self, model_id: str, device, revision: str = "main"):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        # ⚠️ le dépôt TTM héberge ses variantes par RÉVISION ; `main` est une
        # variante horizon-30 qui charge avec des poids de tête RÉINITIALISÉS
        # (avertissement MISSING mesuré) — toujours passer la révision exacte.
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_id, revision=revision)
        self.model.to(device).eval()
        self.ctx_len = self.model.config.context_length
        self.pred_len = self.model.config.prediction_length
        self.device = device

    @torch.no_grad()
    def paths(self, ctx: np.ndarray, h: int, n_jitter: int, rng) -> np.ndarray:
        assert h <= self.pred_len, f"h={h} > horizon TTM {self.pred_len}"
        base = ctx[-self.ctx_len:].astype(np.float32)
        if len(base) < self.ctx_len:                     # pad gauche par bord
            base = np.concatenate([np.full(self.ctx_len - len(base), base[0],
                                           dtype=np.float32), base])
        ctxs = [base] + [base + rng.normal(0, 0.05 * max(base.std(), 1e-8),
                                           size=base.shape).astype(np.float32)
                         for _ in range(n_jitter)]
        x = torch.from_numpy(np.stack(ctxs)).unsqueeze(-1).to(self.device)
        out = self.model(past_values=x).prediction_outputs                # [N, P, 1]
        return out[:, :h, 0].cpu().numpy()


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
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="pas de descente de gradient de E sur les candidats (planning by backprop)")
    ap.add_argument("--refine-lr", type=float, default=0.05)
    ap.add_argument("--decoder-samples", type=int, default=0,
                    help="chemins MC-dropout du décodeur ajoutés au pool hybride")
    ap.add_argument("--proposer-ttm", default=None, const="ibm-granite/granite-timeseries-ttm-r3",
                    nargs="?", help="active le proposeur externe TTM (id HF optionnel)")
    ap.add_argument("--ttm-revision", default="1024-96-r3",
                    help="révision HF (contexte-horizon) — main = tête réinitialisée !")
    ap.add_argument("--calibrate-T", action="store_true",
                    help="calibre T par série sur des sous-fenêtres du contexte (G12a)")
    ap.add_argument("--cal-windows", type=int, default=2)
    ap.add_argument("--centered-bootstrap", action="store_true",
                    help="G12c : pool hybrid_ttm centré sur le proposeur — "
                         "bootstrap des innovations saisonnières RECOLLÉ sur "
                         "le chemin TTM (anti-dilution par construction), "
                         "drift hors du pool")
    ap.add_argument("--ttm-jitter", type=int, default=4,
                    help="contextes jitterés par fenêtre pour diversifier TTM")
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

    ttm = (TTMProposer(args.proposer_ttm, device, revision=args.ttm_revision)
           if args.proposer_ttm else None)
    if ttm:
        logger.info(f"proposeur TTM : {args.proposer_ttm} (ctx {ttm.ctx_len}, h {ttm.pred_len})")

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
                # une calibration PAR SÉRIE et PAR COMPOSITION DE POOL (les
                # pools energy / hybrid_ttm n'ont pas le même contraste), sur
                # le contexte de la première fenêtre — amorti sur ~max_windows.
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
                    # Hybride : trajectoires-quantiles + chemins MC-dropout du
                    # décodeur entrent dans le pool ; le pretrain juge tout.
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
                    # `ttm` seul = son chemin propre (point -> fan répété, WQL=ND)
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
                    logger.info(f"  T calibrés [{k}] {name}: "
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
    logger.info(f"JSON : {out}")


if __name__ == "__main__":
    main()
