"""
Build the corpus frequency table (2026-10-05): one line per .npy file of the
corpus, file stem -> sampling frequency, for timejepa.data.frequency.

The .npy files carry no metadata, so the frequency is read where it is
declared: the `freq` field of the source rows on Hugging Face, at the revision
the corpus was built from.

    python scripts/build_frequency_table.py --corpus-dir data/processed/lotsa_v3 \
        --out configs/corpus_v3_frequencies.yaml

Rules, in order:
  synthetic_*            no physical frequency -> null (kept as an augmentation)
  <stem>_dec<K>          the source stem's frequency, K times coarser
  chronos_<subset>       CHRONOS_FREQ below (four subsets, checked by hand)
  anything else          a LOTSA subset: `freq` of its first row
A daily file needs one more decision the source does not carry: does the
series follow a weekly rhythm (FlowState's rule: transport, healthcare, sales)?
Daily files are written with `weekly: REVIEW`, which the loader refuses, so the
table cannot be used before a human has replaced each REVIEW by true or false.
A stem that cannot be resolved stops the script: nothing is guessed.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.frequency import parse_frequency, season_length  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_frequency_table")
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets", "fsspec", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

LOTSA_REPO = "Salesforce/lotsa_data"
LOTSA_REVISION_V3 = "8191fd29eb5cf906ec55effca44d8059888b615d"
CHRONOS_FREQ = {"dominick": "W", "ercot": "H", "mexico_city_bikes": "H", "ushcn_daily": "D"}


def coarser(freq: str, k: int) -> str:
    """'5T', 3 -> '15T'; 'H', 2 -> '2H'."""
    unit, n = parse_frequency(freq)
    return f"{n * k}{unit}"


def lotsa_frequency(subset: str, revision: str) -> str:
    from datasets import load_dataset

    ds = load_dataset(LOTSA_REPO, subset, split="train", streaming=True, revision=revision)
    row = next(iter(ds))
    freq = row.get("freq")
    if not freq or parse_frequency(freq) is None:
        raise ValueError(f"{subset}: no usable `freq` in the source row (got {freq!r}, "
                         f"columns {sorted(row)})")
    return str(freq)


def resolve(stem: str, revision: str, cache: dict) -> tuple:
    """(frequency or None, how it was found)."""
    if stem.startswith("synthetic_"):
        return None, "synthetic"
    m = re.fullmatch(r"(.+)_dec(\d+)", stem)
    if m:
        base, how = resolve(m.group(1), revision, cache)
        return (None if base is None else coarser(base, int(m.group(2)))), f"decimated ({how})"
    if stem.startswith("chronos_"):
        return CHRONOS_FREQ[stem[len("chronos_"):]], "chronos"
    if stem not in cache:
        cache[stem] = lotsa_frequency(stem, revision)
    return cache[stem], "lotsa"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=LOTSA_REVISION_V3)
    args = ap.parse_args()

    stems = sorted(p.stem for p in Path(args.corpus_dir).glob("*.npy"))
    if not stems:
        raise SystemExit(f"no .npy file in {args.corpus_dir}")
    rows, failed, cache = [], [], {}
    for stem in stems:
        try:
            freq, how = resolve(stem, args.revision, cache)
        except Exception as e:                       # network, unknown subset, missing field
            failed.append(f"{stem}: {type(e).__name__}: {e}")
            continue
        rows.append((stem, freq, how))
        logger.info(f"{stem:44s} {str(freq):>6s}  {how}")
    if failed:
        raise SystemExit(f"{len(failed)} file(s) not resolved, nothing written:\n  " + "\n  ".join(failed))

    lines = ["# Corpus frequency table: file stem -> sampling frequency (timejepa.data.frequency).",
             f"# Built by scripts/build_frequency_table.py from {args.corpus_dir} ({len(rows)} files),",
             f"# LOTSA revision {args.revision}. `null` = no physical frequency.",
             "# Daily files: replace each `weekly: REVIEW` by true (human rhythm: transport, health,",
             "# sales) or false before use; the loader refuses REVIEW."]
    daily = []
    for stem, freq, how in rows:
        if freq is None:
            lines.append(f"{stem}: null          # {how}")
            continue
        unit, n = parse_frequency(freq)
        if unit == "D":
            daily.append(stem)
            lines.append(f"{stem}: {{freq: \"{freq}\", weekly: REVIEW}}     # {how}")
        else:
            lines.append(f"{stem}: {{freq: \"{freq}\"}}     # {how}, season {season_length(freq):g}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    logger.info(f"written: {args.out} ({len(rows)} files, {sum(f is None for _, f, _ in rows)} without a frequency)")
    if daily:
        logger.info(f"{len(daily)} daily file(s) to review (weekly cycle or not): {', '.join(daily)}")


if __name__ == "__main__":
    main()
