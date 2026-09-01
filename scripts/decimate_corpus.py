"""
Decimation of dense families to coarser grids (v3 corpus, S2).

    # make 10T and 15T from the 5T families
    python scripts/decimate_corpus.py --src data/processed/lotsa_full \\
        --dst data/processed/decimated --factors 2,3 \\
        --families largest_2017 largest_2018 largest_2019 largest_2021

The IBM/TTM trick, quantified by their ablations: build frequency coverage
from the same bytes. A 5T chunk mean-pooled by 2 becomes a LEGITIMATE 10T
chunk (downward aggregation is physically correct - upward interpolation is
what would fabricate information). E19 target: reinforce 10T/15T
(electricity/15T/long still at 1.206) with REAL data on top of the synthetic.

Contracts:
* mean-pooling in blocks of `factor` (remainder truncated);
* the decimated length must stay >= min_len (default 1280 = ctx+pred),
  otherwise the family is SKIPPED for that factor (stated, not silent);
* output `<family>_dec<f>.npy` in a SEPARATE directory - mixing is done by
  symlink + balance audit, never by generating into a corpus in place (same
  rule as generate_synthetic.py);
* never overwrite: existing file = skipped.
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
        logger.info(f"  {out_path.name}: already present, skipped")
        return False
    arr = np.load(src, mmap_mode="r")
    if arr.ndim != 2:
        logger.warning(f"  {src.name}: shape {arr.shape} != [chunks, L], skipped")
        return False
    n, L = arr.shape
    L_dec = (L // factor)
    if L_dec < min_len:
        logger.warning(f"  {src.name} /{factor}: {L_dec} < {min_len} steps, "
                       f"skipped (chunk too short after decimation)")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = np.empty((n, L_dec), dtype=np.float32)
    # in batches to keep RAM constant on large memmapped files
    step = max(1, (1 << 26) // max(L, 1))
    for i in range(0, n, step):
        block = np.asarray(arr[i:i + step, :L_dec * factor], dtype=np.float32)
        out[i:i + step] = block.reshape(block.shape[0], L_dec, factor).mean(axis=2)
    np.save(out_path, out)
    logger.info(f"{out_path.name}: {n:,} chunks x {L_dec} (/{factor})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="source corpus directory")
    ap.add_argument("--dst", type=Path, required=True, help="output directory (SEPARATE)")
    ap.add_argument("--factors", default="2,3", help="decimation factors, e.g. '2,3'")
    ap.add_argument("--families", nargs="*", default=None,
                    help="file prefixes to decimate (default: all .npy)")
    ap.add_argument("--min-len", type=int, default=1280,
                    help="minimum length after decimation (ctx 1024 + pred 256)")
    args = ap.parse_args()

    factors = [int(f) for f in args.factors.split(",")]
    files = sorted(args.src.glob("*.npy"))
    if args.families:
        files = [f for f in files
                 if any(f.stem.startswith(fam) for fam in args.families)]
    if not files:
        sys.exit(f"no matching .npy in {args.src}")

    logger.info(f"{len(files)} files x factors {factors} -> {args.dst}")
    done = 0
    for f in files:
        for k in factors:
            done += decimate_file(f, args.dst, k, args.min_len)
    print(f"\nDone: {done} files decimated into {args.dst}")
    print("Mixing: symlinks into the target corpus + balance audit "
          "(audit_batch_schedule.py) - the ratio is an experiment decision.")


if __name__ == "__main__":
    main()
