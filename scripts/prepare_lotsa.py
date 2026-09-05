"""
Converts LOTSA (HuggingFace `Salesforce/lotsa_data`) to the TimeJEPA format.

    # inventory only: list subsets and what would be excluded
    python scripts/prepare_lotsa.py --list

    # conversion (streaming, constant RAM)
    python scripts/prepare_lotsa.py --out data/processed/lotsa \\
        --chunk-length 8192 --max-chunks-per-subset 200000

    # resume: already-converted subsets are skipped
    python scripts/prepare_lotsa.py --out data/processed/lotsa --resume

What the script guarantees, and why:
* **Dense float32 output only**, by segmenting series into fixed-length
  chunks. `object` arrays break fork's copy-on-write and already blew up the
  project's RAM (B19); at LOTSA scale that would be fatal. Corollary: the
  produced files are memmappable (`data.use_mmap: true`).
* **Exclusion of evaluation datasets** by substring (`EVAL_OVERLAP_PATTERNS`).
  The script PRINTS the excluded list: re-read it before any pretrain, since
  LOTSA names are not frozen in time.
* **Per-subset cap** (`--max-chunks-per-subset`). E10 measured two datasets
  weighing 48.7% of the pretrain batch; the cap keeps LOTSA from reproducing
  that imbalance at larger scale.
* **Resume**: each subset is an independent file.

Dependencies: `datasets` and `huggingface_hub`, declared in pyproject.toml.
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

# httpx and huggingface_hub log EVERY request at INFO. On a full conversion
# that is tens of thousands of lines drowning the only output that matters -
# the excluded list and the progress. Raise them to WARNING: network errors
# stay visible.
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets",
               "fsspec", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

REPO_ID = "Salesforce/lotsa_data"

def list_subsets():
    """LOTSA subset names, via the HuggingFace API."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO_ID, repo_type="dataset")
    names = sorted({f.split("/")[0] for f in files if "/" in f})
    return names


def series_iter(subset: str):
    """
    Stream of 1-D series from one LOTSA subset.

    LOTSA stores the series in the `target` column, either 1-D (univariate)
    or 2-D (multivariate: one row per channel). The project is univariate by
    choice, so each channel becomes an independent series - exactly the
    treatment already applied to the current corpus's multivariate datasets.
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
                    help="Dense chunk length. Trade-off: longer means more "
                         "window positions per chunk, but any shorter series "
                         "is LOST. At 8192, whole subsets (metros, VM traces) "
                         "produced nothing. 2048 keeps ~13 positions per "
                         "chunk for a 1280 window, and far more series.")
    ap.add_argument("--min-length", type=int, default=1280,
                    help="Shorter series ignored (= required window).")
    ap.add_argument("--pad-to", type=int, default=None,
                    help="v3 corpus, short series (G7.1/S2): LEFT-pads every "
                         "accepted chunk up to this length (first value "
                         "repeated) - the target stays real, the 'flat then "
                         "data' context is the short-series eval condition. "
                         "Recommended setting: --min-length 384 --pad-to 1280 "
                         "--chunk-length 1280 on m1/m3/tourism/nn5.")
    ap.add_argument("--max-chunks-per-subset", type=int, default=200_000,
                    help="Per-subset cap.")
    ap.add_argument("--max-chunks-per-family", type=int, default=None,
                    help="Per-FAMILY cap, split among its members. Default: "
                         "3x the per-subset cap. Essential because cmip6 (33 "
                         "yearly slices) and era5 (30) are 63 of the 123 "
                         "subsets: without this cap, half the corpus would be "
                         "climate reanalysis. E10 already measured this "
                         "imbalance at smaller scale.")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Restrict to these subsets (default: all).")
    ap.add_argument("--list", action="store_true", help="Inventory only.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-converted subsets.")
    ap.add_argument("--max-nan-fraction", type=float, default=0.05,
                    help="Missing-value fraction tolerated in a chunk, filled "
                         "by linear interpolation. Beyond it, the chunk is "
                         "given up rather than invented. Rejecting any chunk "
                         "with a NaN cost 100 %% of HZMETRO and SHMETRO.")
    args = ap.parse_args()

    logger.info(f"LOTSA subsets from {REPO_ID}...")
    names = args.subsets or list_subsets()

    kept, excluded = [], []
    for n in names:
        (excluded if is_eval_overlap(n) else kept).append(n)

    print()
    print("=" * 72)
    print(f"EXCLUDED for evaluation overlap ({len(excluded)}) - RE-READ THIS")
    print("=" * 72)
    for n in excluded:
        print(f"  x {n}")
    print(f"\n  patterns: {', '.join(EVAL_OVERLAP_PATTERNS)}")
    print()
    print("=" * 72)
    print(f"KEPT for pretrain ({len(kept)})")
    print("=" * 72)
    for n in kept:
        print(f"  + {n}")
    print()

    # Per-subset budget, reduced for large families. Computed BEFORE the
    # --list return: this is exactly what we want to inspect before
    # converting anything.
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
        print(f"FAMILIES sharing a budget (cap {family_cap:,} chunks each)")
        print("=" * 72)
        for f, c in sorted(shared.items(), key=lambda kv: -kv[1]):
            example = next(n for n in kept if family_of(n) == f)
            print(f"  {f:<28} {c:>3} slices -> {budget[example]:,} chunks each")
        print()

    if args.list:
        print("--list: no conversion performed.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0
    for i, subset in enumerate(kept, 1):
        out_path = args.out / f"{subset}.npy"
        if args.resume and out_path.exists():
            logger.info(f"[{i}/{len(kept)}] {subset}: already present, skipped")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(kept)}] {subset} -> {out_path.name}")
        try:
            # `--chunk-length` is a MAXIMUM. Sample the first series to pick
            # the subset's effective length: each file is an independent
            # dense array, nothing forces two subsets to share one. Without
            # this, BEIJING_SUBWAY_30MIN (552 series of 1572 steps) produced
            # nothing against a 2048 chunk.
            stream = series_iter(subset)
            sample = []
            for series in stream:
                sample.append(series)
                if len(sample) >= 200:
                    break

            lengths = [len(x) for x in sample]
            if args.pad_to:
                # Padded block (corpus v3/v4): NO median adaptation. Adapting
                # to the median and then segmenting kept only the LEADING
                # chunk of every series longer than the median, i.e. dropped
                # its most recent steps (v4 short block, 2026-09-05:
                # monash_m3_quarterly 24-72 steps cut at 44). With padding a
                # series is either kept whole (>= min_length) or rejected.
                effective = (args.chunk_length
                             if max(lengths, default=0) >= args.min_length
                             else None)
            else:
                effective = choose_chunk_length(
                    lengths, args.chunk_length, args.min_length)
            if effective is None:
                logger.warning(
                    f"    series too short (< {args.min_length}) - "
                    f"subset unusable for this geometry"
                )
                continue
            if effective != args.chunk_length:
                logger.info(f"    chunk length adapted: {effective}")

            def _chained(buffered=sample, rest=stream):
                yield from buffered
                yield from rest

            emit_len = max(effective, args.pad_to) if args.pad_to else effective
            if emit_len != effective:
                logger.info(f"    left edge-padding: {effective} -> {emit_len}")
            stats = ChunkStats()
            # Corpus v4: with padding, record the real length of every row in
            # a sidecar under _reallen/ (a subfolder, so the dataset glob
            # '*.npy' never mistakes it for a series file).
            real_lens = [] if args.pad_to else None
            chunks = iter_dense_chunks(
                _chained(),
                chunk_length=effective,
                min_length=args.min_length,
                max_chunks=budget[subset],
                max_nan_fraction=args.max_nan_fraction,
                stats=stats,
                pad_to=args.pad_to,
                real_lens=real_lens,
            )
            written = write_dense_npy(
                chunks, out_path,
                chunk_length=emit_len,
                max_chunks=budget[subset],
            )
            if real_lens is not None and written > 0:
                side = Path(out_path).parent / "_reallen" / Path(out_path).name
                side.parent.mkdir(parents=True, exist_ok=True)
                np.save(side, np.asarray(real_lens[:written], dtype=np.int32))
            # Always logged, including (especially) when nothing is written:
            # the only way to know WHY a subset is empty.
            logger.info(f"    {stats.summary(effective, args.min_length)}")
            total += written
        except Exception as exc:  # one broken subset must not kill the run
            logger.error(f"  x {subset} failed: {type(exc).__name__}: {exc}")

    print()
    print(f"Done: {total:,} chunks WRITTEN this run "
          f"~ {total * args.chunk_length / 1e9:.2f}B observations")
    print(f"Output: {args.out}")

    # `total` only counts files written during this run. With --resume and an
    # already-converted corpus it is 0 and reads like a failure when nothing
    # failed - exactly what happened when relaunching with a higher cap, and
    # the message cost a diagnostic round-trip.
    if skipped:
        print()
        print(f"WARNING: {skipped} subsets SKIPPED (--resume): their files")
        print("    already existed and were NOT reconverted.")
        if total == 0:
            print()
            print("    No file written: the corpus is unchanged, at its original")
            print("    size. If the goal was to GROW it (raised cap), --resume")
            print("    prevents that - reconvert to a different --out.")
    print()
    print("Next step - pretrain:")
    print("  python scripts/train.py --config-name lotsa_tiny")


if __name__ == "__main__":
    main()
