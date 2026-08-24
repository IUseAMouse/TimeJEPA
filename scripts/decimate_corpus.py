#!/usr/bin/env python
"""
Décimation de familles denses vers des grilles plus grossières (corpus v3, S2).

    # fabriquer du 10T et du 15T depuis les familles 5T
    python scripts/decimate_corpus.py --src data/processed/lotsa_full \\
        --dst data/processed/decimated --factors 2,3 \\
        --families largest_2017 largest_2018 largest_2019 largest_2021

Le tour de main d'IBM/TTM chiffré par leurs ablations : fabriquer la couverture
fréquentielle à partir des mêmes octets. Un morceau 5T mean-poolé par 2 devient
un morceau 10T LÉGITIME (l'agrégation vers le bas est physiquement correcte —
c'est l'interpolation vers le haut qui fabriquerait de l'information). Cible
E19 : renforcer 10T/15T (electricity/15T/long encore à 1.206) avec du RÉEL en
plus du synthétique.

Contrats :
* mean-pooling par blocs de `factor` (troncature du reste) ;
* la longueur décimée doit rester >= min_len (défaut 1280 = ctx+pred) sinon la
  famille est SAUTÉE pour ce facteur (dit, pas silencieux) ;
* sortie `<famille>_dec<f>.npy` dans un répertoire SÉPARÉ — le mélange se fait
  par symlink + audit d'équilibre, jamais en générant dans un corpus en place
  (même règle que generate_synthetic.py) ;
* jamais d'écrasement : fichier existant = sauté.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("decimate_corpus")


def decimate_file(src: Path, dst_dir: Path, factor: int, min_len: int) -> bool:
    out_path = dst_dir / f"{src.stem}_dec{factor}.npy"
    if out_path.exists():
        logger.info(f"  {out_path.name}: déjà présent, sauté")
        return False
    arr = np.load(src, mmap_mode="r")
    if arr.ndim != 2:
        logger.warning(f"  {src.name}: forme {arr.shape} != [chunks, L], sauté")
        return False
    n, L = arr.shape
    L_dec = (L // factor)
    if L_dec < min_len:
        logger.warning(f"  {src.name} ÷{factor}: {L_dec} < {min_len} pas — "
                       f"sauté (morceau trop court après décimation)")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = np.empty((n, L_dec), dtype=np.float32)
    # par lots pour rester en RAM constante sur les gros fichiers memmappés
    step = max(1, (1 << 26) // max(L, 1))
    for i in range(0, n, step):
        block = np.asarray(arr[i:i + step, :L_dec * factor], dtype=np.float32)
        out[i:i + step] = block.reshape(block.shape[0], L_dec, factor).mean(axis=2)
    np.save(out_path, out)
    logger.info(f"✓ {out_path.name}: {n:,} morceaux x {L_dec} (÷{factor})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="répertoire du corpus source")
    ap.add_argument("--dst", type=Path, required=True, help="répertoire de sortie (SÉPARÉ)")
    ap.add_argument("--factors", default="2,3", help="facteurs de décimation, ex. '2,3'")
    ap.add_argument("--families", nargs="*", default=None,
                    help="préfixes de fichiers à décimer (défaut : tous les .npy)")
    ap.add_argument("--min-len", type=int, default=1280,
                    help="longueur minimale après décimation (ctx 1024 + pred 256)")
    args = ap.parse_args()

    factors = [int(f) for f in args.factors.split(",")]
    files = sorted(args.src.glob("*.npy"))
    if args.families:
        files = [f for f in files
                 if any(f.stem.startswith(fam) for fam in args.families)]
    if not files:
        sys.exit(f"✗ aucun .npy correspondant dans {args.src}")

    logger.info(f"{len(files)} fichiers x facteurs {factors} -> {args.dst}")
    done = 0
    for f in files:
        for k in factors:
            done += decimate_file(f, args.dst, k, args.min_len)
    print(f"\nTerminé : {done} fichiers décimés dans {args.dst}")
    print("Mélange : symlinks dans le corpus visé + audit d'équilibre "
          "(audit_batch_schedule.py) — le ratio est une décision d'expérience.")


if __name__ == "__main__":
    main()
