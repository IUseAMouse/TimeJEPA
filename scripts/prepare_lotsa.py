#!/usr/bin/env python
"""
Convertit LOTSA (HuggingFace `Salesforce/lotsa_data`) au format TimeJEPA.

    # inventaire seul : liste les sous-ensembles et ce qui serait exclu
    python scripts/prepare_lotsa.py --list

    # conversion (streaming, RAM constante)
    python scripts/prepare_lotsa.py --out data/processed/lotsa \\
        --chunk-length 8192 --max-chunks-per-subset 200000

    # reprise : les sous-ensembles déjà convertis sont sautés
    python scripts/prepare_lotsa.py --out data/processed/lotsa --resume

Ce que le script garantit, et pourquoi
--------------------------------------
* **Sortie dense float32 uniquement**, par segmentation des séries en morceaux de
  longueur fixe. Les tableaux `object` cassent le copy-on-write de fork et ont
  déjà fait exploser la RAM du projet (B19) ; à l'échelle de LOTSA ce serait
  rédhibitoire. Corollaire : les fichiers produits sont memmappables
  (`data.use_mmap: true`).
* **Exclusion des datasets d'évaluation** par sous-chaîne
  (`EVAL_OVERLAP_PATTERNS`). Le script IMPRIME la liste des exclus : à relire
  avant tout pretrain, les noms LOTSA n'étant pas figés dans le temps.
* **Plafond par sous-ensemble** (`--max-chunks-per-subset`). E10 a mesuré que
  deux datasets pesaient 48,7 % du batch de pretrain ; le plafond empêche LOTSA
  de reproduire ce déséquilibre à plus grande échelle.
* **Reprise** : chaque sous-ensemble est un fichier indépendant.

Dépendances : `datasets` et `huggingface_hub`, déclarées dans pyproject.toml.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.lotsa import (  # noqa: E402
    EVAL_OVERLAP_PATTERNS,
    is_eval_overlap,
    iter_dense_chunks,
    write_dense_npy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_lotsa")

# httpx et huggingface_hub loggent CHAQUE requête en INFO. Sur une conversion
# complète cela fait des dizaines de milliers de lignes qui noient la seule
# sortie qui compte — la liste des exclus et l'avancement. On les remonte à
# WARNING : les erreurs réseau restent visibles.
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets",
               "fsspec", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

REPO_ID = "Salesforce/lotsa_data"

def list_subsets():
    """Noms des sous-ensembles LOTSA, via l'API HuggingFace."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO_ID, repo_type="dataset")
    names = sorted({f.split("/")[0] for f in files if "/" in f})
    return names


def series_iter(subset: str):
    """
    Flux de séries 1-D d'un sous-ensemble LOTSA, en streaming.

    LOTSA stocke la série dans la colonne `target`, qui est soit 1-D (univarié)
    soit 2-D (multivarié : une ligne par canal). Le projet étant univarié par
    choix, chaque canal devient une série indépendante — ce qui est exactement
    le traitement déjà appliqué aux datasets multivariés du corpus actuel.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO_ID, subset, split="train", streaming=True)
    for row in ds:
        target = row.get("target")
        if target is None:
            continue
        arr = np.asarray(target, dtype=np.float32)
        if arr.ndim == 1:
            yield arr
        elif arr.ndim == 2:
            for channel in arr:
                yield channel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/lotsa"))
    ap.add_argument("--chunk-length", type=int, default=8192,
                    help="Longueur des morceaux denses. Doit dépasser largement "
                         "context+prediction (1280 pour tiny_geo).")
    ap.add_argument("--min-length", type=int, default=1280,
                    help="Séries plus courtes ignorées (= fenêtre requise).")
    ap.add_argument("--max-chunks-per-subset", type=int, default=200_000,
                    help="Plafond anti-domination (E10).")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Restreindre à ces sous-ensembles (défaut : tous).")
    ap.add_argument("--list", action="store_true", help="Inventaire seul.")
    ap.add_argument("--resume", action="store_true",
                    help="Sauter les sous-ensembles déjà convertis.")
    args = ap.parse_args()

    logger.info(f"Sous-ensembles LOTSA depuis {REPO_ID}…")
    names = args.subsets or list_subsets()

    kept, excluded = [], []
    for n in names:
        (excluded if is_eval_overlap(n) else kept).append(n)

    print()
    print("=" * 72)
    print(f"EXCLUS pour recoupement avec l'évaluation ({len(excluded)}) — À RELIRE")
    print("=" * 72)
    for n in excluded:
        print(f"  ✗ {n}")
    print(f"\n  motifs : {', '.join(EVAL_OVERLAP_PATTERNS)}")
    print()
    print("=" * 72)
    print(f"RETENUS pour le pretrain ({len(kept)})")
    print("=" * 72)
    for n in kept:
        print(f"  ✓ {n}")
    print()

    if args.list:
        print("--list : aucune conversion effectuée.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, subset in enumerate(kept, 1):
        out_path = args.out / f"{subset}.npy"
        if args.resume and out_path.exists():
            logger.info(f"[{i}/{len(kept)}] {subset}: déjà présent, sauté")
            continue

        logger.info(f"[{i}/{len(kept)}] {subset} → {out_path.name}")
        try:
            chunks = iter_dense_chunks(
                series_iter(subset),
                chunk_length=args.chunk_length,
                min_length=args.min_length,
                max_chunks=args.max_chunks_per_subset,
            )
            written = write_dense_npy(
                chunks, out_path,
                chunk_length=args.chunk_length,
                max_chunks=args.max_chunks_per_subset,
            )
            total += written
        except Exception as exc:  # un sous-ensemble cassé ne doit pas tuer le run
            logger.error(f"  ✗ {subset} échoué : {type(exc).__name__}: {exc}")

    print()
    print(f"Terminé : {total:,} morceaux de {args.chunk_length} pas "
          f"≈ {total * args.chunk_length / 1e9:.2f} Md d'observations")
    print(f"Sortie : {args.out}")
    print()
    print("Prochaine étape — pretrain :")
    print("  python scripts/train.py --config-name lotsa_tiny")


if __name__ == "__main__":
    main()
