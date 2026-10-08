"""
Seasons of the synthetic corpus files, read back by replaying their seeds
(2026-10-08). The generator is deterministic: replaying a file's seed gives
the same series, and `sample_series(..., info=)` reports the period of each
series' reference cycle as it is drawn, without an extra draw. Each replayed
row is compared with the file on disk; a single mismatch aborts the file and
nothing is written. The result is a sidecar `_season/<stem>.npy` (float32,
one value per row, 0.0 = no cycle) next to the corpus, which the dataset reads
when the frequency table says `{season: per_row}`. The .npy files are not
touched.

    python scripts/build_season_sidecars.py --corpus-dir data/processed/lotsa_v3 [--jobs 4] [--only stem ...]
    python scripts/build_season_sidecars.py --corpus-dir data/processed/lotsa_v3 --quantize 2

`--quantize N` rounds the seasons already written so that the rate scale
w = 24 / season falls on a grid of N levels per octave (no replay). Reason
(2026-10-08): continuous seasons give every synthetic series its own scale,
a batch then carries ~40 distinct scales and the S4D layer, which computes
one kernel per scale, runs three times slower than with ~15. Two levels per
octave bound the rounding of a cycle to 19% and the batch to ~20 scales.

File -> (family, seed), from scripts/generate_synthetic.py and build_corpus_v3.sh:
    synthetic_<family>            v1 run: seed = index of the family in DEFAULT_FAMILIES
    synthetic_<family>_s<N>       v3 shard: seed = N * 1000
    <stem>_dec<K>                 the parent's seasons divided by K (rows are kept by decimate_corpus)
"""

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.synthetic import DEFAULT_FAMILIES, V3_FAMILIES, sample_series  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_season_sidecars")
SPECS = {f.name: f for f in V3_FAMILIES}


def family_and_seed(stem: str):
    """('synthetic_subhourly', seed) for a generated file, or None."""
    m = re.fullmatch(r"(synthetic_[a-z_]+?)(?:_s(\d+))?", stem)
    if not m or m.group(1) not in SPECS:
        return None
    if m.group(2) is not None:
        return m.group(1), int(m.group(2)) * 1000
    return m.group(1), [f.name for f in DEFAULT_FAMILIES].index(m.group(1))


def replay(path: Path) -> np.ndarray:
    """Seasons of one generated file, after checking every row against the file."""
    stem = path.stem
    family, seed = family_and_seed(stem)
    arr = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    seasons = np.zeros(arr.shape[0], dtype=np.float32)
    info = {}
    for i in range(arr.shape[0]):
        row = sample_series(SPECS[family], rng, info)
        if not np.array_equal(row, arr[i]):
            raise RuntimeError(f"{stem}: replayed row {i} differs from the file (family {family}, "
                               f"seed {seed}); the generator or the seed rule changed, nothing written")
        seasons[i] = info["season"]
    return seasons


def build_one(path: Path, out_dir: Path) -> str:
    seasons = replay(path)
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / path.name, seasons)
    has = seasons > 0
    return (f"{path.stem:34s} {len(seasons):6d} rows, {100 * has.mean():5.1f}% with a cycle, "
            f"season median {np.median(seasons[has]) if has.any() else 0:7.1f}")


def quantize_seasons(seasons: np.ndarray, levels_per_octave: int, base: float = 24.0) -> np.ndarray:
    """Round each season so that base / season sits on a 2^(k / levels) grid;
    0 (no cycle) stays 0."""
    out = seasons.astype(np.float32).copy()
    has = out > 0
    k = np.round(levels_per_octave * np.log2(base / out[has]))
    out[has] = (base / np.power(2.0, k / levels_per_octave)).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quantize", type=int, default=0, metavar="N",
                    help="round the seasons already in _season/ to N rate levels per octave (no replay)")
    args = ap.parse_args()
    corpus = Path(args.corpus_dir)
    out_dir = corpus / "_season"
    if args.quantize:
        raw_dir = corpus / "_season_raw"
        raw_dir.mkdir(exist_ok=True)
        for p in sorted(out_dir.glob("synthetic_*.npy")):
            raw = raw_dir / p.name
            if not raw.exists():
                raw.write_bytes(p.read_bytes())          # the exact seasons are kept aside, never deleted
            seasons = quantize_seasons(np.load(raw), args.quantize)
            np.save(p, seasons)
            logger.info(f"{p.stem:34s} {int(np.unique(seasons[seasons > 0]).size):3d} distinct seasons after rounding")
        logger.info(f"rounded {args.quantize} levels per octave; exact values in {raw_dir}")
        return
    files = sorted(p for p in corpus.glob("synthetic_*.npy") if args.only is None or p.stem in args.only)
    generated = [p for p in files if "_dec" not in p.stem]
    decimated = [p for p in files if "_dec" in p.stem]
    for p in generated:
        if family_and_seed(p.stem) is None:
            raise SystemExit(f"{p.stem}: unknown family or seed rule")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(build_one, generated, [out_dir] * len(generated)):
            logger.info(line)
    for p in decimated:
        m = re.fullmatch(r"(.+)_dec(\d+)", p.stem)
        parent = out_dir / f"{m.group(1)}.npy"
        if not parent.exists():
            raise SystemExit(f"{p.stem}: parent sidecar {parent.name} missing")
        seasons = np.load(parent) / int(m.group(2))
        n = np.load(p, mmap_mode="r").shape[0]
        if seasons.shape[0] != n:
            raise SystemExit(f"{p.stem}: {n} rows but the parent has {seasons.shape[0]}")
        np.save(out_dir / p.name, seasons.astype(np.float32))
        logger.info(f"{p.stem:34s} {n:6d} rows, parent / {m.group(2)}")
    logger.info(f"written: {out_dir} ({len(files)} files)")


if __name__ == "__main__":
    main()
