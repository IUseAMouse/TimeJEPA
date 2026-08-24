#!/usr/bin/env python
"""
Génère le corpus synthétique (G8 / P2.5) au format des corpus convertis.

    # les trois familles par défaut, ~1.6 Md d'observations, ~6.5 Go
    python scripts/generate_synthetic.py --out data/processed/synthetic

    # dimensionner autrement
    python scripts/generate_synthetic.py --out data/processed/synthetic \\
        --chunks-per-family 50000 --seed 7

Trois familles, une par trou mesuré (voir src/timejepa/data/synthetic.py) :
  synthetic_subhourly   périodes 24-150 pas, morceaux 8192  -> trou 10T/15T (E17)
  synthetic_broadband   périodes 4-2048,     morceaux 8192  -> fond de diversité
  synthetic_lowfreq     périodes 4-52,       morceaux 1280  -> séries courtes (G7.1)

⚠️ Le répertoire de sortie est SÉPARÉ des corpus réels, exprès : mélanger se
fait en déposant/symlinkant les .npy dans le répertoire du corpus visé, jamais
en générant dedans — un corpus d'un run en cours ne doit pas changer sous ses
pieds, et le ratio réel/synthétique est une DÉCISION d'expérience à consigner,
pas un état de répertoire implicite.

Les morceaux de 8192 sont aussi la première donnée compatible avec la
multi-résolution (f jusqu'à 6 : 1280·6 = 7680 ≤ 8192) et donc avec le JEPA
inter-résolution (G9.2) — les morceaux réels de 2048 bloquent f=2.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.synthetic import (  # noqa: E402
    DEFAULT_FAMILIES,
    V3_FAMILIES,
    write_synthetic_family,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("generate_synthetic")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    ap.add_argument("--chunks-per-family", type=int, default=25_000,
                    help="Morceaux par famille. À 25k : ~0.4 Md d'observations "
                         "sur les familles 8192, ~30 s/famille de génération.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Graine racine ; chaque famille dérive la sienne.")
    ap.add_argument("--families", nargs="*", default=None,
                    help="Restreindre à ces familles (défaut : toutes celles du set).")
    ap.add_argument("--set", choices=["default", "v3"], default="default",
                    help="v3 = les 3 familles v1 + ops_bursty (bizitobs/E19) "
                         "+ intermittent (car_parts). Roadmap S2 2026-08-24.")
    args = ap.parse_args()

    bank = V3_FAMILIES if args.set == "v3" else DEFAULT_FAMILIES
    fams = [f for f in bank
            if args.families is None or f.name in args.families]
    if not fams:
        known = ", ".join(f.name for f in bank)
        raise SystemExit(f"aucune famille ne correspond — connues : {known}")

    total_obs = 0
    for i, spec in enumerate(fams):
        out_path = args.out / f"{spec.name}.npy"
        if out_path.exists():
            logger.info(f"{out_path.name}: déjà présent, sauté (supprimer pour régénérer)")
            continue
        t0 = time.time()
        write_synthetic_family(out_path, spec, args.chunks_per_family,
                               seed=args.seed * 1000 + i)
        total_obs += args.chunks_per_family * spec.chunk_length
        logger.info(f"  ({time.time() - t0:.0f} s)")

    print()
    print(f"Terminé : {total_obs / 1e9:.2f} Md d'observations dans {args.out}")
    print()
    print("Pour mélanger à un corpus (exemple, corpus complet) :")
    print("  ln -s $(pwd)/data/processed/synthetic/*.npy data/processed/lotsa_full/")
    print("  # puis relancer l'audit d'équilibre — le synthétique y apparaît")
    print("  # comme trois familles de plus, et la décision du ratio se lit dedans.")


if __name__ == "__main__":
    main()
