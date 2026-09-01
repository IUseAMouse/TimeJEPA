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
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from timejepa.evaluation import gift  # noqa: E402
from timejepa.evaluation import ratein as ratein_mod  # noqa: E402

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
                 lookbacks=None, flip: bool = False, shifts=None,
                 w=None) -> dict:
    """
    Moyenne de variantes d'inférence du MÊME checkpoint :
      * multi-lookback : le contexte tronqué à L' pour chaque L' de `lookbacks`
        (borné à la longueur disponible, aligné sur le stride) ;
      * miroir sign-flip optionnel sur chaque variante ;
      * `shifts` (2026-08-25, idée utilisateur — équivariance de translation) :
        pour chaque s, une variante prévoit depuis l'origine t−s (les s points
        les plus récents retirés, longueur ré-alignée sur le stride → la PHASE
        du grid de patchs change) ; sa prévision est RÉALIGNÉE sur l'horizon
        (l'indice p+s−1 de la variante = la position p de l'horizon) et
        moyennée par position avec masque de couverture (les s dernières
        positions ne sont couvertes que par les variantes moins décalées).
        Recommandé : s petits (1..7 = moins d'un patch de péremption).
    NB théorique (testé, pas seulement affirmé) : la variante d'échelle
    f(kx)/k est un NO-OP EXACT sous RobustScale+RevIN (médiane et MAD sont
    1-homogènes → entrée normalisée bit-identique) — le seul élément du groupe
    affine non quotienté par la normalisation est le SIGNE, c'est le flip.
    Les fans sont moyennés niveau à niveau (la moyenne pondérée de vecteurs
    triés reste triée) ; la médiane est relue dans le fan moyenné.
    Sans options : strictement équivalent à model.forecast.
    """
    L = batch.shape[1]
    stride = model.patching.stride
    looks = sorted({min(int(lb), L) - (min(int(lb), L) % stride)
                    for lb in (lookbacks or [L])
                    if min(int(lb), L) >= model.patching.patch_size})
    if not looks:
        looks = [L]
    # s >= h : la variante décalée ne couvrirait AUCUNE position de l'horizon
    # (mesuré : m4_yearly a h=6 — un shift de 7 cassait l'alignement).
    shift_list = [0] + sorted({int(s) for s in (shifts or [])
                               if 0 < int(s) < h})

    outs = []                                  # (sortie, décalage s)
    for lb in looks:
        for s in shift_list:
            ctx = batch[:, -lb:L - s if s else None] if s else batch[:, -lb:]
            # ré-alignement stride : tronquer À GAUCHE (jamais à droite — le
            # padding de Patching fabriquerait des points après l'origine)
            trim = ctx.shape[1] % stride
            if trim:
                ctx = ctx[:, trim:]
            if ctx.shape[1] < model.patching.patch_size:
                continue
            # w (RateIN x w) : le taux est invariant par négation -> même w
            # sur la variante miroir.
            outs.append((model.forecast(ctx, n=h, w=w), s))
            if flip:
                outs.append((_negated(model.forecast(-ctx, n=h, w=w)), s))

    if len(outs) == 1:
        return outs[0][0]

    def _aligned_mean(key):
        vals = [(o.get(key), s) for o, s in outs]
        if any(v is None for v, _ in vals):
            return None
        ref = vals[0][0]
        acc = torch.zeros_like(ref)
        cnt = torch.zeros(h, device=ref.device)
        for v, s in vals:
            if s == 0:
                acc += v
                cnt += 1.0
            else:
                # indices s..h−1 de la variante ↔ positions 0..h−s−1
                acc[:, :h - s] += v[:, s:]
                cnt[:h - s] += 1.0
        shape = [1, h] + [1] * (ref.dim() - 2)
        return acc / cnt.reshape(shape)

    res = {"forecast_denorm": _aligned_mean("forecast_denorm")}
    fan = _aligned_mean("quantiles_denorm")
    if fan is not None:
        res["quantiles_denorm"] = fan
        levels = next((o["quantile_levels"] for o, _ in outs
                       if "quantile_levels" in o), None)
        if levels is not None:
            mid = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
            res["forecast_denorm"] = fan.select(-2, mid).unsqueeze(-2) \
                if fan.dim() == 4 else fan[..., mid:mid + 1]
            res["quantile_levels"] = levels
    return res


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def apply_quantile_gamma(out: dict, gamma) -> dict:
    """
    G4.2 — calibration conforme uniforme : q'_k = med + gamma_k · (q_k − med).

    gamma [Q] vient de scripts/calibrate_quantiles.py (corpus de FINETUNE,
    jamais GIFT — un seul vecteur pour les 97 configs, doctrine mono-checkpoint).
    La médiane est intouchable par construction (gamma_0.5 = 1 et le point
    forecast n'est pas modifié) : MASE bit-identique, tout delta est du CRPS.
    gamma > 0 préserve la monotonie des quantiles. Appliqué APRÈS la moyenne
    TTA (la calibration est faite sous la même procédure) ; l'enveloppe G8.4b
    a déjà borné les quantiles en amont — l'élargissement peut la dépasser
    marginalement (gamma ≲ 2 contre une enveloppe à ±10·range : accepté).
    """
    q = out.get("quantiles_denorm")
    if q is None or gamma is None:
        return out
    med = out["forecast_denorm"]                       # [B, h, 1]
    g = gamma.to(q.device, q.dtype)
    if q.dim() == 4:                                   # [B, h, Q, 1]
        q = med.unsqueeze(-2) + g.view(1, 1, -1, 1) * (q - med.unsqueeze(-2))
    else:                                              # [B, h, Q]
        q = med + g.view(1, 1, -1) * (q - med)
    res = dict(out)
    res["quantiles_denorm"] = q
    return res


def _pinball_np(fan: np.ndarray, y: np.ndarray, levels) -> float:
    """Pinball moyenne (convention GluonTS x2 sans effet sur l'argmin)."""
    d = y[:, None] - fan
    q = np.asarray(levels, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def _backtest_series_k(model, series, h: int, windows: int, max_len: int,
                       stride: int, patch: int, device,
                       batch_size: int) -> dict:
    """RateIN v2 (2026-09-01) — le k PAR SÉRIE, choisi par BACKTEST CAUSAL.

    Verdict oracle du 2026-08-31 : le mécanisme vaut jusqu'à +57 % par config
    (34/91 > 5 %), mais le détecteur FFT v1 le rate sur les fréquences
    grossières (il décime des D/W/M que l'oracle laisse à k=1) ET l'oracle
    révèle un SECOND mécanisme que la période ne voit pas (collapse de
    rollout : bizitobs/H gagne 40 % à k=16 sur un cycle de 24). Le backtest
    capture les deux SANS métadonnée : pour chaque série, on retire les h_bt
    derniers pas de son passé (jamais le test — la fenêtre rejouée précède la
    première cible d'éval), on forecast chaque k candidat, et on garde le
    meilleur pinball. Même logique que la calibration-T en contexte (E18h).

    v2.1 (2026-09-01, verdict du run v2 : capture 30 % de l'oracle) :
    * h_bt = h RÉEL (plus de cap à 256) — le cap rendait le collapse de
      rollout INVISIBLE à la sélection : à h_bt<=256, k=1 n'a jamais besoin
      de rollout dans le backtest, donc les plus gros gains oracle (bizitobs
      +40-56 % à k=16-24, h=480) ne pouvaient pas être vus. Le backtest doit
      refléter la vraie tâche. Repli : h_bt réduit si l'historique est court.
    * jusqu'à 2 fenêtres moyennées (réduction de variance, coût x2 sur une
      mini-passe) ;
    * marge no-op de 5 % (esprit règle 1-SE) : une pinball sur 1-2 fenêtres
      de 14 pas est bruitée (m4_daily 3.89 vs flip 3.48 en v2) — k>1 doit
      battre k=1 d'au moins REL_MARGIN ; les vrais gains oracle sont à
      +20-50 %, très au-dessus.

    v3 (2026-09-01, diagnostic v2.1 : winner's curse par série) : k PAR
    CONFIG, scores poolés entre séries. L'argmin PAR SÉRIE sur 11 candidats
    avec 1-2 fenêtres sélectionne les coups de chance (jena/D : 14/42
    instances à k=16, le PIRE k de la table oracle, +163 %) et les mélanges
    de k sous-performent le k uniforme sur les paysages accidentés
    (bitbrains 0.896 avec un mélange vs <=0.837 pour TOUTE la colonne
    oracle). Pooling : ratio pinball k/k=1 par série, geomean entre séries
    (la normalisation ôte l'échelle/difficulté), variance / n_séries, et la
    granularité devient CELLE DE L'ORACLE (par config) qui borne la capture.
    Un k coté sur moins de 2/3 des séries de base est disqualifié
    (sous-ensemble biaisé — typiquement grand k sur séries courtes). La
    garde par instance (historique décimé < patch -> k=1) reste le filet.
    """
    REL_MARGIN = 0.05
    N_BT_WINDOWS = 2
    ks = {}
    entries = []                                    # (idx, sub_hist, known)
    for idx, y in enumerate(series):
        past = y[:len(y) - windows * h]             # avant TOUTE cible de test
        avail = len(past) - 4 * patch
        h_bt = min(h, avail)
        if h_bt < 16:
            ks[idx] = 1
            continue
        got = 0
        for j in range(1, min(N_BT_WINDOWS, avail // h_bt) + 1):
            lo = len(past) - j * h_bt
            sub_hist, known = past[:lo], past[lo:lo + h_bt]
            if not np.isfinite(known).all():
                continue
            entries.append((idx, sub_hist, known))
            got += 1
        if got == 0:
            ks[idx] = 1

    scores = defaultdict(lambda: defaultdict(list))
    for k in ratein_mod.K_CANDIDATES:
        buckets = defaultdict(list)
        for idx, sub_hist, known in entries:
            hist = (ratein_mod.decimate(sub_hist[-(max_len * k):], k)
                    if k > 1 else sub_hist)
            if len(hist) < patch:
                continue
            ctx = prepare_context(hist, max_len, stride, patch)
            if ctx is None:
                continue
            # h_bt varie par série (repli historiques courts) -> le bucket
            # porte aussi h_fc pour que le batch soit homogène.
            buckets[(len(ctx), -(-len(known) // k))].append((idx, ctx, known))
        for (length, h_fc), items in buckets.items():
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                batch = torch.from_numpy(np.stack([c[1] for c in chunk]))
                batch = batch.unsqueeze(-1).to(device)
                with torch.no_grad():
                    out = model.forecast(batch, n=h_fc)
                q = out.get("quantiles_denorm")
                if q is None:
                    continue
                levels = list(out.get("quantile_levels",
                                      [0.1 * j for j in range(1, 10)]))
                q = q.cpu().numpy()
                if q.ndim == 4:
                    q = q[..., 0]
                for b, (idx, _, known) in enumerate(chunk):
                    fan_nat = ratein_mod.reinterp_fan(q[b], len(known), k)
                    sc = _pinball_np(fan_nat, known, levels)
                    if np.isfinite(sc):
                        scores[idx][k].append(sc)
    base_scored = [i for i in scores
                   if scores[i].get(1) and np.mean(scores[i][1]) > 0]
    ratios = {}
    for k in ratein_mod.K_CANDIDATES:
        if k == 1:
            continue
        rs = [np.mean(scores[i][k]) / np.mean(scores[i][1])
              for i in base_scored if scores[i].get(k)]
        if not rs or len(rs) < (2 * len(base_scored)) // 3:
            continue                                # sous-ensemble biaisé
        ratios[k] = float(np.exp(np.mean(np.log(np.maximum(rs, 1e-12)))))
    K = 1
    if ratios:
        kbest = min(ratios, key=ratios.get)
        if ratios[kbest] < 1.0 - REL_MARGIN:
            K = kbest
    return {idx: K for idx in range(len(series))}


def evaluate_config(model, config: str, gift_root: Path, device,
                    batch_size: int, max_series: int = 0,
                    max_context: int = 0, tta_lookbacks=None,
                    tta_flip: bool = False, tta_shifts=None,
                    quantile_gamma=None,
                    ratein_mode: str = "off", forced_k: int = 0,
                    ratein_w: bool = False, ratein_w_max_k: int = 4) -> dict:
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
    k_hist = Counter()
    # RateIN v2 — le k par SÉRIE choisi par backtest causal (voir le helper) ;
    # calculé une fois avant la boucle, batché.
    bt_ks = None
    if ratein_mode == "backtest" and not forced_k:
        bt_ks = _backtest_series_k(model, series, h, windows, max_len, stride,
                                   model.patching.patch_size, device,
                                   batch_size)
    for inst in gift.iter_test_instances(series, h, windows):
        # RateIN — canonicalisation du taux (2026-08-31, verdict G9.3) : k
        # est une statistique CAUSALE du passé (règle uniforme sur les 97
        # configs — même statut que médiane/MAD). forced_k = mode ORACLE
        # (diagnostic, jamais officiel).
        if forced_k:
            k = forced_k
        elif bt_ks is not None:
            k = bt_ks.get(inst.series_idx, 1)
        elif ratein_mode == "fft":
            k = ratein_mod.choose_k(ratein_mod.detect_period(inst.context))
        else:
            k = 1
        hist = (ratein_mod.decimate(inst.context[-(max_len * k):], k)
                if k > 1 else inst.context)
        if k > 1 and len(hist) < model.patching.patch_size:
            # Garde (crash oracle 2026-08-31, IndexError sur 6 configs) : un
            # historique court décimé par un grand k devient vide — repli k=1.
            k, hist = 1, inst.context
        ctx = prepare_context(hist, max_len, stride, model.patching.patch_size)
        if ctx is None:
            continue
        # MASE scale uses the FULL past, not the capped context — gluonts
        # computes the seasonal error on the entire history of the series.
        scale = gift.seasonal_error(inst.context, m)
        buckets[(len(ctx), k)].append((ctx, inst.target, inst.context, scale))
        k_hist[k] += 1
        n_inst += 1

    for (length, k), items in sorted(buckets.items()):
        # RateIN x w (synergie G9.3, flag gated) : contexte décimé par k, fan
        # demandé DIRECTEMENT au taux natif via w = 1/k — zéro
        # ré-interpolation (la seule perte que même l'oracle ne peut éviter).
        # Garde d'extrapolation : la FiLM n'a vu que log2(w) dans la gamme
        # des facteurs d'entraînement ([1,2,4] -> [-2,2]) ; au-delà de
        # ratein_w_max_k, repli sur le chemin standard décimé+reinterp.
        use_w = ratein_w and 1 < k <= ratein_w_max_k
        # Grille décimée : h' = ceil(h/k) pas suffisent à couvrir l'horizon
        # natif (bonus mesurable : moins de rollouts sur 10S/5T long-terme).
        h_fc = h if use_w else -(-h // k)
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            batch = torch.from_numpy(np.stack([c[0] for c in chunk]))
            batch = batch.unsqueeze(-1).to(device)          # [B, L, 1]

            w_vec = (torch.full((batch.shape[0],), 1.0 / k, device=device)
                     if use_w else None)
            with torch.no_grad():
                out = tta_forecast(model, batch, h_fc,
                                   lookbacks=tta_lookbacks, flip=tta_flip,
                                   shifts=tta_shifts, w=w_vec)
            out = apply_quantile_gamma(out, quantile_gamma)

            median = out["forecast_denorm"].squeeze(-1).cpu().numpy()  # [B, h']
            quants = out.get("quantiles_denorm")
            if quants is not None:
                quants = quants.cpu().numpy()
                if quants.ndim == 4:                        # [B, h', Q, 1]
                    quants = quants[..., 0]
                # -> [B, h', Q]
            levels = out.get("quantile_levels")
            mid = (min(range(len(levels)), key=lambda j: abs(levels[j] - 0.5))
                   if levels is not None and quants is not None
                   else (quants.shape[-1] // 2 if quants is not None else 0))

            for b, (ctx, target, past, scale) in enumerate(chunk):
                if k > 1 and not use_w:
                    if quants is not None:
                        fan_nat = ratein_mod.reinterp_fan(quants[b], h, k)
                        med_nat = fan_nat[:, mid]
                    else:
                        fan_nat = None
                        med_nat = ratein_mod.reinterp_fan(
                            median[b][:, None], h, k)[:, 0]
                    model_acc.add(target, med_nat, fan_nat, scale)
                else:
                    model_acc.add(target, median[b],
                                  quants[b] if quants is not None else None,
                                  scale)
                sn = gift.seasonal_naive_forecast(past, h, m)
                sn_acc.add(target, sn, None, scale)

    res = {"config": config, "prediction_length": h, "seasonality": m,
           "windows": windows, "n_series": len(series), "n_instances": n_inst,
           "model": model_acc.result(), "seasonal_naive_local": sn_acc.result()}
    if ratein_mode != "off" or forced_k:
        res["ratein"] = {
            "k_hist": {str(kk): n for kk, n in sorted(k_hist.items())},
            "frac_k_gt1": (sum(n for kk, n in k_hist.items() if kk > 1)
                           / max(1, sum(k_hist.values()))),
        }
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
    #   +ratein=fft   (alias true)             détecteur de période v1
    #   +ratein=backtest                        k par série, backtest causal (v2)
    #   +ratein=oracle                          balayage de k — DIAGNOSTIC,
    #                                           regarde le test, jamais officiel
    ratein_raw = str(cfg.get("ratein", "")).lower()
    ratein_mode_val = {"true": "fft", "1": "fft", "on": "fft", "fft": "fft",
                       "backtest": "backtest", "bt": "backtest"}.get(
                           ratein_raw, "off")
    ratein_on = ratein_mode_val != "off"
    ratein_oracle = ratein_raw == "oracle"
    # RateIN x w (gated) : +ratein_w=true — fan au taux natif via w=1/k sur
    # les buckets décimés (k <= 4). Exige un modèle cross_resolution ET un
    # mode ratein actif (sans décimation, w=1 partout = no-op trompeur).
    ratein_w = str(cfg.get("ratein_w", "")).lower() in ("true", "1", "on")
    if ratein_w and not (ratein_on or ratein_oracle):
        raise ValueError("+ratein_w exige +ratein=backtest/fft/oracle "
                         "(sans décimation, w vaudrait 1 partout)")
    tta_lookbacks = ([int(x) for x in str(cfg.tta_lookbacks).split(",")]
                     if cfg.get("tta_lookbacks") else None)
    tta_flip = bool(cfg.get("tta_flip", False))
    #   +tta_shifts='2,4,6'  (translation : origines t−s réalignées, s < stride
    #                         recommandé — moins d'un patch de péremption)
    tta_shifts = ([int(x) for x in str(cfg.tta_shifts).split(",")]
                  if cfg.get("tta_shifts") else None)
    #   +quantile_gamma='evaluation/calibration/gamma_<ckpt>.json'
    #   (G4.2 : facteurs d'élargissement par niveau, calibrés sur le corpus de
    #    FINETUNE par scripts/calibrate_quantiles.py — statut : ablation papier,
    #    décision utilisateur 2026-08-25, pas nécessairement le chiffre officiel)
    quantile_gamma, gamma_tag = None, ""
    if cfg.get("quantile_gamma"):
        gpath = Path(str(cfg.quantile_gamma))
        gdata = json.loads(gpath.read_text())
        quantile_gamma = torch.tensor(gdata["gamma"], dtype=torch.float32)
        gamma_tag = "_gamma-" + gpath.stem
        logger.info(f"quantile_gamma: {gdata['gamma']} ({gpath})")

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
    if ratein_w and getattr(model.predictor, "w_film", None) is None:
        raise ValueError("+ratein_w exige un modèle cross_resolution "
                         "(FiLM w absent du prédicteur — config xres)")
    if ratein_mode_val == "fft":
        tag += "_ratein"
    elif ratein_mode_val == "backtest":
        tag += "_ratein-bt"
    elif ratein_oracle:
        tag += "_ratein-oracle"
    if ratein_w:
        tag += "-w"
    if tta_shifts:
        tag += "_sh" + "-".join(str(x) for x in tta_shifts)
    tag += gamma_tag
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
            if ratein_oracle:
                # Balayage de k PAR CONFIG — la borne supérieure du gain
                # atteignable par canonicalisation du taux. Tranche
                # l'échec-diagnostic de P-RIN : si même l'oracle ne gagne
                # nulle part, la géométrie d'échelle n'est pas le mécanisme.
                per_k, best = {}, None
                for kk in ratein_mod.K_CANDIDATES:
                    r_k = evaluate_config(model, config, gift_root, device,
                                          batch_size, max_series,
                                          max_context=max_context,
                                          tta_lookbacks=tta_lookbacks,
                                          tta_flip=tta_flip,
                                          tta_shifts=tta_shifts,
                                          quantile_gamma=quantile_gamma,
                                          forced_k=kk, ratein_w=ratein_w)
                    per_k[str(kk)] = r_k["model"]["CRPS"]
                    if best is None or r_k["model"]["CRPS"] < best["model"]["CRPS"]:
                        best, best_k = r_k, kk
                res = best
                res["oracle"] = {"per_k_crps": per_k, "best_k": best_k,
                                 "gain_vs_k1": 1.0 - per_k[str(best_k)]
                                 / per_k["1"] if per_k["1"] > 0 else 0.0}
            else:
                res = evaluate_config(model, config, gift_root, device,
                                      batch_size, max_series,
                                      max_context=max_context,
                                      tta_lookbacks=tta_lookbacks,
                                      tta_flip=tta_flip,
                                      tta_shifts=tta_shifts,
                                      quantile_gamma=quantile_gamma,
                                      ratein_mode=ratein_mode_val,
                                      ratein_w=ratein_w)
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
        extra = ""
        if "ratein" in res and not ratein_oracle:
            extra = f" k>1: {res['ratein']['frac_k_gt1']:.0%}"
        if "oracle" in res:
            extra = (f" best_k={res['oracle']['best_k']} "
                     f"(gain {res['oracle']['gain_vs_k1']:+.1%} vs k=1)")
        logger.info(
            f"[{i}/{len(configs)}] {config}: MASE {mm['MASE']:.3f} "
            f"CRPS {mm['CRPS']:.3f} ({res['n_instances']} inst, "
            f"{time.time() - t0:.0f}s){extra}")

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
    # Couverture empirique moyenne (instrument E21 : nominal q10=0.10, q90=0.90
    # => intervalle 80 % ; la sous-couverture zero-shot mesurée est du shift,
    # G4.2 — le levier légitime est le gate z d'ESJEPA).
    covs = [r["model"].get("coverage") for r in results.values()
            if r["model"].get("coverage")]
    if covs:
        c10 = float(np.mean([c["0.1"] for c in covs]))
        c90 = float(np.mean([c["0.9"] for c in covs]))
        logger.info(f"  coverage (moyenne {len(covs)} configs): q10 {c10:.3f} (nominal 0.100) | "
                    f"q90 {c90:.3f} (nominal 0.900) | intervalle 80% -> {c90 - c10:.3f}")
    if ratein_on:
        fr = [r["ratein"]["frac_k_gt1"] for r in results.values()
              if "ratein" in r]
        n_active = sum(1 for f in fr if f > 0.5)
        logger.info(f"  RateIN : {n_active}/{len(fr)} configs majoritairement "
                    f"décimées | part d'instances k>1 (moyenne) : "
                    f"{float(np.mean(fr)):.1%}")
    if ratein_oracle:
        # LA lecture de l'échec-diagnostic P-RIN : combien de configs
        # gagnent > 5 % même avec le meilleur k choisi en trichant.
        gains = {c: r["oracle"]["gain_vs_k1"] for c, r in results.items()
                 if "oracle" in r}
        big = sorted(((g, c) for c, g in gains.items() if g > 0.05),
                     reverse=True)
        logger.info(f"  ORACLE-k (diagnostic, jamais officiel) : "
                    f"{len(big)}/{len(gains)} configs gagnent > 5% vs k=1")
        for g, c in big[:8]:
            logger.info(f"    {c}: {g:+.1%} (best_k="
                        f"{results[c]['oracle']['best_k']})")
        if not big:
            logger.info("    AUCUNE -> échec-diagnostic P-RIN : la géométrie "
                        "d'échelle n'est pas le mécanisme de la queue ; "
                        "G9.3/xres non financé.")
    logger.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
