#!/usr/bin/env python
"""
G4.2 — calibration conforme UNIFORME des quantiles (split conformal, CQR-style).

    python scripts/calibrate_quantiles.py \\
        --checkpoint checkpoints/champions/ration45_mase0.8702_crps0.5959.ckpt \\
        --config-name lotsa_tiny_mix_zeroshot --flip

Statut (décision utilisateur 2026-08-25) : ABLATION EXPÉRIMENTALE pour le
papier — pas nécessairement le chiffre officiel communiqué.

Le problème mesuré : le fan est trop étroit — couverture [q10, q90] de 42-72 %
selon les datasets contre 80 % nominal, biais dans la MÊME direction partout.

La correction : un facteur d'élargissement PAR NIVEAU de quantile, autour de
la médiane —  q'_k = med + gamma_k · (q_k − med)  — un seul vecteur de 9
scalaires pour les 97 configs GIFT (doctrine « un checkpoint, zéro adaptation
par config »). MASE invariante par construction (la médiane ne bouge pas).

La calibration : sur les fenêtres de VALIDATION du corpus de FINETUNE — jamais
GIFT (le test de l'éval à l'aveugle passe : gamma se fige avant de soumettre,
il fait partie du modèle, comme la température d'un classifieur). Pour chaque
niveau k, gamma_k est le quantile empirique du ratio r = (y−med)/(q_k−med) qui
rétablit la couverture nominale (Romano et al., CQR — variante multiplicative,
invariante d'échelle par fenêtre). Agrégation : MÉDIANE des gamma par dataset
(un dataset = une voix, les gros corpus ne votent pas plus).

Limites assumées (dites, pas cachées) :
  * la couverture est invariante par transformation monotone, mais gamma est
    fitté dans l'espace des fenêtres du datamodule et appliqué dans l'espace
    dénormalisé GIFT — l'écart (arcsinh) est de second ordre devant un
    miscalibrage de 8-38 points ;
  * gamma est supposé indépendant de l'horizon (calibré à h=256, appliqué à
    h∈[6, 720]) ;
  * calibrer AVEC --flip si la procédure officielle est ×flip (la moyenne de
    deux fans RESSERRE l'étalement — la correction doit voir la procédure
    qu'elle corrige).

Sortie : evaluation/calibration/gamma_<ckpt>[<tags>].json
         {levels, gamma, coverage_before/after par dataset, meta}
Consommé par : evaluate_gift.py  +quantile_gamma=<ce json>.
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

from hydra import compose, initialize_config_dir                    # noqa: E402

from timejepa.data.datamodule import MultiDatasetMonashDataModule   # noqa: E402
from timejepa.evaluation import create_model_from_config, load_checkpoint  # noqa: E402
from evaluate_gift import tta_forecast                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_quantiles")

GAMMA_CLAMP = (0.25, 4.0)   # garde-fou : jamais un facteur dégénéré


def gamma_for_level(r: np.ndarray, k: float) -> float:
    """gamma rétablissant P(couverture) = k pour un niveau k, à partir des
    ratios r = (y−med)/(q_k−med).  Niveau haut (den>0) : P(r<=g)=k ;
    niveau bas (den<0, l'inégalité se renverse) : P(r>=g)=k."""
    if len(r) < 50:
        return 1.0
    q = k if k > 0.5 else 1.0 - k
    return float(np.clip(np.quantile(r, q), *GAMMA_CLAMP))


def calibrate_dataset(model, get_item, idx, h: int, batch_size: int,
                      flip: bool, device):
    """Collecte les ratios et couvertures d'UN dataset et rend
    (levels, {gamma, coverage_before, n_windows}) — accumulateurs locaux à la
    fonction : l'état ne peut plus fuir d'un dataset à l'autre (bug du
    2026-08-26 : init couplée à la découverte des levels, seul le premier
    dataset était initialisé)."""
    levels, ratios, covered = None, None, None
    for i0 in range(0, len(idx), batch_size):
        chunk = idx[i0:i0 + batch_size]
        items = [get_item(j) for j in chunk]
        ctx = torch.stack([torch.as_tensor(it['context'], dtype=torch.float32).reshape(-1)
                           for it in items]).unsqueeze(-1).to(device)  # [B, L, 1]
        y = torch.stack([torch.as_tensor(it['target'], dtype=torch.float32).reshape(-1)
                         for it in items]).cpu().numpy()               # [B, h]
        with torch.no_grad():
            out = tta_forecast(model, ctx, h, flip=flip)
        med = out["forecast_denorm"].squeeze(-1).cpu().numpy()         # [B, h]
        q = out["quantiles_denorm"].cpu().numpy()
        if q.ndim == 4:
            q = q[..., 0]                                              # [B, h, Q]
        if levels is None:
            levels = [float(x) for x in out["quantile_levels"]]
            ratios = [[] for _ in levels]
            covered = [[] for _ in levels]
        den = q - med[..., None]                                       # [B, h, Q]
        num = (y - med).ravel()                        # commun à tous les niveaux
        for ki in range(len(levels)):
            d_k = den[..., ki].ravel()
            ok = np.abs(d_k) > 1e-8
            ratios[ki].append((num[ok] / d_k[ok]).astype(np.float32))
            covered[ki].append((y.ravel() <= q[..., ki].ravel()).astype(np.float32))
    stats = {
        "gamma": [gamma_for_level(np.concatenate(ratios[ki]), levels[ki])
                  if levels[ki] != 0.5 else 1.0
                  for ki in range(len(levels))],
        "coverage_before": [float(np.concatenate(covered[ki]).mean())
                            for ki in range(len(levels))],
        "n_windows": int(len(idx)),
    }
    return levels, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="lotsa_tiny_mix_zeroshot",
                    help="config du FINETUNE (fixe corpus + géométrie)")
    ap.add_argument("--flip", action="store_true",
                    help="calibrer sous la procédure officielle ×flip")
    ap.add_argument("--per-dataset", type=int, default=192,
                    help="fenêtres de val échantillonnées par dataset")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="evaluation/calibration")
    args = ap.parse_args()

    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "model")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_from_config(cfg)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dm = MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get('datasets_finetune'),
        dataset_pattern=cfg.data.get('dataset_pattern', '*.npy'),
        combine_mode=cfg.data.get('combine_mode', 'concatenate'),
        balanced_sampling=cfg.data.balanced_sampling,
        sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio,
        batch_size=args.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        seed=cfg.data.seed,
        num_workers=0,
        use_mmap=bool(cfg.data.get('use_mmap', False)),
    )
    # Hors Lightning, prepare_data (découverte des .npy) doit être appelé
    # explicitement — setup('fit') seul voit un dataset_files vide et conclut
    # « every dataset skipped » (mesuré 2026-08-26).
    dm.prepare_data()
    dm.setup('fit')

    # Le val_dataset est la concaténation ordonnée des datasets : les tailles
    # par dataset donnent les bornes — échantillonnage régulier dans chacune.
    names = dm.dataset_names_order
    sizes = dm.val_dataset_sizes
    bounds = np.cumsum([0] + list(sizes))
    h = int(cfg.model.prediction_length)
    levels = None
    per_ds = {}

    for d, name in enumerate(names):
        lo, hi = int(bounds[d]), int(bounds[d + 1])
        if hi - lo < 8:
            logger.info(f"  ⏭️  {name}: {hi - lo} fenêtres val, ignoré")
            continue
        idx = np.linspace(lo, hi - 1, min(args.per_dataset, hi - lo)).astype(int)
        levels, stats = calibrate_dataset(
            model, lambda j: dm.val_dataset[int(j)], idx, h,
            args.batch_size, args.flip, device)
        per_ds[name] = stats
        g, cb = stats["gamma"], stats["coverage_before"]
        logger.info(f"  {name:32s} n={len(idx):4d} "
                    f"couv q10/q90 avant {cb[0]:.2f}/{cb[-1]:.2f} "
                    f"gamma q10/q90 {g[0]:.2f}/{g[-1]:.2f}")

    if not per_ds:
        raise RuntimeError("aucun dataset calibrable")

    # Un dataset = une voix.
    gamma = [1.0 if levels[ki] == 0.5 else
             float(np.median([per_ds[n]["gamma"][ki] for n in per_ds]))
             for ki in range(len(levels))]

    logger.info("\nNiveau   gamma (médiane inter-datasets)")
    for lv, g in zip(levels, gamma):
        logger.info(f"  {lv:.1f}    {g:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem + ("_flip" if args.flip else "")
    payload = {
        "levels": levels, "gamma": gamma,
        "checkpoint": args.checkpoint, "config_name": args.config_name,
        "flip": args.flip, "per_dataset": per_ds,
        "doctrine": ("uniforme 97 configs, calibré corpus de finetune, "
                     "jamais GIFT — statut ablation papier 2026-08-25"),
    }
    path = out_dir / f"gamma_{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"\nJSON : {path}")
    logger.info("Éval : python scripts/evaluate_gift.py --config-name "
                f"lotsa_tiny_mix_eval '+checkpoint_path=\"{args.checkpoint}\"' "
                + ("+tta_flip=true " if args.flip else "")
                + f"+quantile_gamma='{path}'")


if __name__ == "__main__":
    main()
