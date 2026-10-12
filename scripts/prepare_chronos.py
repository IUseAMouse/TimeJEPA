"""
Converts the useful subsets of the Chronos corpus to the TimeJEPA format.

    # inventory: what is admitted, what is excluded and why
    python scripts/prepare_chronos.py --list

    # conversion (4 datasets, a few minutes)
    python scripts/prepare_chronos.py --out data/processed/chronos_extras

Of the Chronos corpus (`autogluon/chronos_datasets`, 53 subsets), everything
that is neither a LOTSA duplicate nor an evaluation leak fits in a four-name
allowlist, so the script enumerates what it admits rather than what it
refuses.

`--chunk-length 8192` by default: multi-resolution decimation requires
`1280*f <= chunk_length`, which the 2048-step LOTSA chunks never satisfy for
f >= 2. `choose_chunk_length` shortens automatically when series are too
short.

Mixing with an existing corpus is done by symlinking the .npy files into the
target corpus directory, never by generating inside it (see
generate_synthetic.py). Files are prefixed `chronos_` for readable provenance
and no name collision.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.lotsa import convert_subset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_chronos")
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets",
               "fsspec", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

CHRONOS_REPO = "autogluon/chronos_datasets"
# Revision the corpus v3 extras were built from.
CHRONOS_REVISION_V3 = "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"
_REVISION = None

# The only subsets both new relative to LOTSA and free of overlap with the
# project's three evaluation suites (GIFT-Eval, Nixtla, local Monash),
# checked one by one.
CHRONOS_ALLOWLIST = ("dominick", "ercot", "mexico_city_bikes", "ushcn_daily")

# The rest of the corpus, with the reason, printed by --list for audit.
# '*' patterns cover whole groups.
CHRONOS_EXCLUDED = {
    "exchange_rate": "IS the Nixtla `exchange` eval",
    "electricity_15min": "source of the Nixtla `electricity` (UCI)",
    "wiki_daily_100k": "overlaps the local Monash eval wikipedia-web-traffic",
    "solar* (solar, solar_1h)": "overlaps the local Monash eval solar-10-minute",
    "m4_*": "GIFT eval datasets",
    "monash_*": "LOTSA duplicates and/or eval datasets (traffic, weather, hospital...)",
    "taxi_*": "already in LOTSA (taxi_30min)",
    "uber_tlc_*": "already in LOTSA",
    "m5": "already in LOTSA",
    "nn5": "already in LOTSA (readmitted there)",
    "weatherbench_*": "climate - era5+cmip6 already weigh ~30% of the batch (G7.1 audit)",
    "wind_farms_*": "already in LOTSA (wind_farms_with_missing)",
    "training_corpus": "TSMixup of the whole Chronos corpus: a mix of the "
                       "sources above, so leakage by construction",
}

# ushcn_daily has no `target` column: each weather variable is a column and
# becomes an independent series, the project's standard univariate treatment.
USHCN_VALUE_COLUMNS = ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN")


def series_iter_chronos(subset: str):
    """Stream of 1-D float32 series from one Chronos subset."""
    from datasets import load_dataset

    ds = load_dataset(CHRONOS_REPO, subset, split="train", streaming=True, revision=_REVISION)
    for row in ds:
        if subset == "ushcn_daily":
            for col in USHCN_VALUE_COLUMNS:
                v = row.get(col)
                if v is None:
                    continue
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1:
                    yield arr
            continue
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
    ap.add_argument("--out", type=Path, default=Path("data/processed/chronos_extras"))
    ap.add_argument("--chunk-length", type=int, default=8192,
                    help="Maximum; choose_chunk_length shortens per subset. "
                         "8192 = decimation possible up to f=6.")
    ap.add_argument("--min-length", type=int, default=1280,
                    help="Required window (ctx 1024 + horizon 256).")
    ap.add_argument("--max-chunks-per-subset", type=int, default=200_000)
    ap.add_argument("--max-nan-fraction", type=float, default=0.05)
    ap.add_argument("--revision", default=None, help="HF dataset revision (sha or tag).")
    ap.add_argument("--list", action="store_true", help="Inventory only.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-converted subsets.")
    args = ap.parse_args()
    global _REVISION
    _REVISION = args.revision
    print(f"HF revision: {args.revision or 'main (unpinned)'}")

    print()
    print("=" * 72)
    print(f"ADMITTED ({len(CHRONOS_ALLOWLIST)}) - explicit allowlist")
    print("=" * 72)
    for n in CHRONOS_ALLOWLIST:
        print(f"  + {n}")
    print()
    print("=" * 72)
    print(f"EXCLUDED ({len(CHRONOS_EXCLUDED)} groups) - RE-READ THIS")
    print("=" * 72)
    for n, why in CHRONOS_EXCLUDED.items():
        print(f"  x {n:<28} {why}")
    print()

    if args.list:
        print("--list: no conversion performed.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, subset in enumerate(CHRONOS_ALLOWLIST, 1):
        out_path = args.out / f"chronos_{subset}.npy"
        if args.resume and out_path.exists():
            logger.info(f"[{i}/{len(CHRONOS_ALLOWLIST)}] {subset}: already present, skipped")
            continue
        logger.info(f"[{i}/{len(CHRONOS_ALLOWLIST)}] {subset} -> {out_path.name}")
        t0 = time.time()
        try:
            written, stats, effective = convert_subset(
                series_iter_chronos(subset), out_path,
                chunk_length=args.chunk_length,
                min_length=args.min_length,
                max_chunks=args.max_chunks_per_subset,
                max_nan_fraction=args.max_nan_fraction,
            )
            if effective is None:
                logger.warning(f"    series too short (median < {args.min_length}) "
                               f"- unusable for this geometry")
                continue
            if effective != args.chunk_length:
                logger.info(f"    chunk length adapted: {effective}")
            logger.info(f"    {stats.summary(effective, args.min_length)} "
                        f"({time.time() - t0:.0f} s)")
            total += written
        except Exception as exc:  # one broken subset must not kill the others
            logger.error(f"  x {subset} failed: {type(exc).__name__}: {exc}")

    print()
    print(f"Done: {total:,} chunks WRITTEN this run")
    print(f"Output: {args.out}")
    print()
    print("Mixing into a corpus (example):")
    print("  mkdir -p data/processed/lotsa_chronos")
    print("  ln -s $(pwd)/data/processed/lotsa/*.npy data/processed/lotsa_chronos/")
    print("  ln -s $(pwd)/data/processed/chronos_extras/*.npy data/processed/lotsa_chronos/")
    print("  # then the balance audit on the mixed directory")


if __name__ == "__main__":
    main()
