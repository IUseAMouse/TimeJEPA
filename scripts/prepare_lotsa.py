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
from collections import Counter
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.lotsa import (  # noqa: E402
    EVAL_OVERLAP_PATTERNS,
    ChunkStats,
    choose_chunk_length,
    family_of,
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
    ap.add_argument("--chunk-length", type=int, default=2048,
                    help="Longueur des morceaux denses. Arbitrage : plus c'est "
                         "long, plus il y a de positions de fenêtre par morceau, "
                         "mais toute série plus courte est PERDUE. À 8192, des "
                         "sous-ensembles entiers (métros, traces de VM) ne "
                         "produisaient rien. 2048 garde ~13 positions par morceau "
                         "pour une fenêtre de 1280, et bien plus de séries.")
    ap.add_argument("--min-length", type=int, default=1280,
                    help="Séries plus courtes ignorées (= fenêtre requise).")
    ap.add_argument("--pad-to", type=int, default=None,
                    help="Corpus v3, séries courtes (G7.1/S2) : rembourre À "
                         "GAUCHE tout morceau accepté jusqu'à cette longueur "
                         "(répétition de la première valeur) — la cible reste "
                         "réelle, le contexte 'plat puis données' est la "
                         "condition d'éval des séries courtes. Réglage "
                         "recommandé : --min-length 384 --pad-to 1280 "
                         "--chunk-length 1280 sur m1/m3/tourism/nn5.")
    ap.add_argument("--max-chunks-per-subset", type=int, default=200_000,
                    help="Plafond par sous-ensemble.")
    ap.add_argument("--max-chunks-per-family", type=int, default=None,
                    help="Plafond par FAMILLE, réparti entre ses membres. Défaut : "
                         "3x le plafond par sous-ensemble. Indispensable parce que "
                         "cmip6 (33 tranches annuelles) et era5 (30) sont 63 des "
                         "123 sous-ensembles : sans ce plafond, la moitié du corpus "
                         "serait de la réanalyse climatique. E10 avait déjà mesuré "
                         "ce déséquilibre à plus petite échelle.")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Restreindre à ces sous-ensembles (défaut : tous).")
    ap.add_argument("--list", action="store_true", help="Inventaire seul.")
    ap.add_argument("--resume", action="store_true",
                    help="Sauter les sous-ensembles déjà convertis.")
    ap.add_argument("--max-nan-fraction", type=float, default=0.05,
                    help="Fraction de valeurs manquantes tolérée dans un morceau, "
                         "comblée par interpolation linéaire. Au-delà, le morceau "
                         "est renoncé plutôt qu'inventé. Rejeter tout morceau "
                         "contenant un NaN coûtait 100 %% de HZMETRO et SHMETRO.")
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

    # Budget par sous-ensemble, réduit pour les familles nombreuses. Calculé
    # AVANT le retour de --list : c'est exactement ce qu'on veut inspecter avant
    # de convertir quoi que ce soit.
    family_cap = args.max_chunks_per_family or 3 * args.max_chunks_per_subset
    members = Counter(family_of(n) for n in kept)
    budget = {
        n: min(args.max_chunks_per_subset,
               max(1, family_cap // members[family_of(n)]))
        for n in kept
    }
    shared = {f: c for f, c in members.items() if c > 1}
    if shared:
        print("=" * 72)
        print(f"FAMILLES partageant un budget (plafond {family_cap:,} morceaux chacune)")
        print("=" * 72)
        for f, c in sorted(shared.items(), key=lambda kv: -kv[1]):
            example = next(n for n in kept if family_of(n) == f)
            print(f"  {f:<28} {c:>3} tranches → {budget[example]:,} morceaux chacune")
        print()

    if args.list:
        print("--list : aucune conversion effectuée.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0
    for i, subset in enumerate(kept, 1):
        out_path = args.out / f"{subset}.npy"
        if args.resume and out_path.exists():
            logger.info(f"[{i}/{len(kept)}] {subset}: déjà présent, sauté")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(kept)}] {subset} → {out_path.name}")
        try:
            # `--chunk-length` est un MAXIMUM. On échantillonne les premières
            # séries pour choisir la longueur effective du sous-ensemble : chaque
            # fichier étant un tableau dense indépendant, rien n'oblige deux
            # sous-ensembles à partager la même. Sans ça, BEIJING_SUBWAY_30MIN
            # (552 séries de 1572 pas) ne produisait rien face à un chunk de 2048.
            stream = series_iter(subset)
            sample = []
            for series in stream:
                sample.append(series)
                if len(sample) >= 200:
                    break

            effective = choose_chunk_length(
                [len(x) for x in sample], args.chunk_length, args.min_length
            )
            if effective is None:
                logger.warning(
                    f"    séries trop courtes (médiane < {args.min_length}) — "
                    f"sous-ensemble inutilisable pour cette géométrie"
                )
                continue
            if effective != args.chunk_length:
                logger.info(f"    longueur de morceau adaptée : {effective}")

            def _chained(buffered=sample, rest=stream):
                yield from buffered
                yield from rest

            emit_len = max(effective, args.pad_to) if args.pad_to else effective
            if emit_len != effective:
                logger.info(f"    rembourrage-bord gauche : {effective} -> {emit_len}")
            stats = ChunkStats()
            chunks = iter_dense_chunks(
                _chained(),
                chunk_length=effective,
                min_length=args.min_length,
                max_chunks=budget[subset],
                max_nan_fraction=args.max_nan_fraction,
                stats=stats,
                pad_to=args.pad_to,
            )
            written = write_dense_npy(
                chunks, out_path,
                chunk_length=emit_len,
                max_chunks=budget[subset],
            )
            # Toujours loggué, y compris (surtout) quand rien n'est écrit :
            # c'est la seule façon de savoir POURQUOI un sous-ensemble est vide.
            logger.info(f"    {stats.summary(effective, args.min_length)}")
            total += written
        except Exception as exc:  # un sous-ensemble cassé ne doit pas tuer le run
            logger.error(f"  ✗ {subset} échoué : {type(exc).__name__}: {exc}")

    print()
    print(f"Terminé : {total:,} morceaux ÉCRITS ce run "
          f"≈ {total * args.chunk_length / 1e9:.2f} Md d'observations")
    print(f"Sortie : {args.out}")

    # `total` ne compte QUE les fichiers écrits pendant ce run. Avec --resume et
    # un corpus déjà converti, il vaut 0 et se lit comme un échec alors que rien
    # n'a échoué — c'est exactement ce qui s'est produit en relançant avec un
    # plafond plus haut, et le message a coûté un aller-retour de diagnostic.
    if skipped:
        print()
        print(f"⚠️  {skipped} sous-ensembles SAUTÉS (--resume) : leurs fichiers")
        print("    existaient déjà et n'ont PAS été reconvertis.")
        if total == 0:
            print()
            print("    Aucun fichier écrit : le corpus est inchangé, à sa taille")
            print("    d'origine. Si le but était de l'AGRANDIR (plafond relevé),")
            print("    --resume l'en empêche — reconvertir vers un autre --out.")
    print()
    print("Prochaine étape — pretrain :")
    print("  python scripts/train.py --config-name lotsa_tiny")


if __name__ == "__main__":
    main()
