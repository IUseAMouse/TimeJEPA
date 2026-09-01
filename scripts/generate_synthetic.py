"""
Generates the synthetic corpus (G8 / P2.5) in the converted-corpus format.

    # the three default families, ~1.6B observations, ~6.5 GB
    python scripts/generate_synthetic.py --out data/processed/synthetic

    # size it differently
    python scripts/generate_synthetic.py --out data/processed/synthetic \\
        --chunks-per-family 50000 --seed 7

Three families, one per measured gap (see src/timejepa/data/synthetic.py):
  synthetic_subhourly   periods 24-150 steps, 8192 chunks  -> 10T/15T gap (E17)
  synthetic_broadband   periods 4-2048,       8192 chunks  -> diversity floor
  synthetic_lowfreq     periods 4-52,         1280 chunks  -> short series (G7.1)

The output directory is SEPARATE from the real corpora on purpose: mixing is
done by dropping/symlinking the .npy files into the target corpus directory,
never by generating inside it - a corpus of a running job must not change
under its feet, and the real/synthetic ratio is an experiment DECISION to
record, not an implicit directory state.

The 8192 chunks are also the first data compatible with multi-resolution
(f up to 6: 1280*6 = 7680 <= 8192) and therefore with the cross-resolution
JEPA (G9.2) - the real 2048 chunks block f=2.
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
                    help="Chunks per family. At 25k: ~0.4B observations on "
                         "the 8192 families, ~30 s/family to generate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Root seed; each family derives its own.")
    ap.add_argument("--families", nargs="*", default=None,
                    help="Restrict to these families (default: all in the set).")
    ap.add_argument("--set", choices=["default", "v3"], default="default",
                    help="v3 = the 3 v1 families + ops_bursty (bizitobs/E19) "
                         "+ intermittent (car_parts). Roadmap S2 2026-08-24.")
    ap.add_argument("--suffix", default="",
                    help="Filename suffix (v3 sharding, 2026-08-27): the "
                         "T=0.5 sampler weighs in sqrt PER FILE, so the "
                         "synthetic batch share is sized by the NUMBER of "
                         "shards (corpus precedent: era5_*/cmip6_*/largest_*). "
                         "E.g. --suffix _s1 -> synthetic_ops_bursty_s1.npy. "
                         "Always pair with a distinct --seed per shard.")
    args = ap.parse_args()

    bank = V3_FAMILIES if args.set == "v3" else DEFAULT_FAMILIES
    fams = [f for f in bank
            if args.families is None or f.name in args.families]
    if not fams:
        known = ", ".join(f.name for f in bank)
        raise SystemExit(f"no matching family - known: {known}")

    total_obs = 0
    for i, spec in enumerate(fams):
        out_path = args.out / f"{spec.name}{args.suffix}.npy"
        if out_path.exists():
            logger.info(f"{out_path.name}: already present, skipped (delete to regenerate)")
            continue
        t0 = time.time()
        write_synthetic_family(out_path, spec, args.chunks_per_family,
                               seed=args.seed * 1000 + i)
        total_obs += args.chunks_per_family * spec.chunk_length
        logger.info(f"  ({time.time() - t0:.0f} s)")

    print()
    print(f"Done: {total_obs / 1e9:.2f}B observations in {args.out}")
    print()
    print("To mix into a corpus (example, full corpus):")
    print("  ln -s $(pwd)/data/processed/synthetic/*.npy data/processed/lotsa_full/")
    print("  # then rerun the balance audit - the synthetic data shows up")
    print("  # as three more families, and the ratio decision reads from it.")


if __name__ == "__main__":
    main()
