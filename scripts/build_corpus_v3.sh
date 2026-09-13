#!/usr/bin/env bash
# Rebuild the corpus v3 from scratch on a new machine (2026-09-13), the exact
# recipe of docs/RUNBOOK_V3.md and of the xres / mix config headers, chained
# and checked. Every step is idempotent (existing outputs are kept, the
# converters resume), nothing is ever deleted, every source lives in its own
# directory and lotsa_v3/ is a directory of symlinks.
#
#   scripts/build_corpus_v3.sh                 # full build (~half a day + download, ~80 GB)
#   scripts/build_corpus_v3.sh --check         # only the final audit against the reference
#
# Reference (measured on the pod, 2026-09-13): 106 files, 15.85 B observations.
# The HF datasets are pinned to the revisions the v3 corpus was built from
# (prepare_lotsa.py LOTSA_REVISION_V3, prepare_chronos.py CHRONOS_REVISION_V3);
# a different revision is the first suspect if the audit disagrees.
#
# Steps (RUNBOOK_V3 numbering in brackets):
#   1. lotsa_full        LOTSA converted, per-subset cap 1e6 chunks of 2048   (G7.1)
#   2. chronos_extras    the 4 admitted Chronos datasets, 8192 chunks         (G9.2)
#   3. synthetic         v1 families, 20k chunks each                          (P2.5b)
#   4. lotsa_xres        symlinks of 1 + 2 + 3                                 (xres header)
#   5. synthetic_v3      23 seeded shards of the v3 families                   [2]
#   6. lotsa_short       short real series padded to 1280; lotsa_solar         [3]
#   7. decimated         5T -> 10T/15T (factors 2, 3) from lotsa_xres          [4]
#   8. lotsa_v3          symlinks of 4 + 5 + 6 + 7                             [5]
#   9. audit             file count and observations vs the reference         [6]
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/processed
LOTSA_REV=$(python -c "import sys; sys.path.insert(0,'scripts'); import prepare_lotsa as p; print(p.LOTSA_REVISION_V3)")
CHRONOS_REV=$(python -c "import sys; sys.path.insert(0,'scripts'); import prepare_chronos as p; print(p.CHRONOS_REVISION_V3)")
REF_FILES=106
REF_OBS=15.85

audit() {
  python - "$D/lotsa_v3" "$REF_FILES" "$REF_OBS" <<'EOF'
import glob, sys
import numpy as np
d, ref_files, ref_obs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
files = sorted(glob.glob(f"{d}/*.npy"))
tot, series = 0, 0
for f in files:
    a = np.load(f, mmap_mode="r", allow_pickle=True)
    if a.dtype == object:
        tot += sum(len(s) for s in a); series += len(a)
    else:
        tot += a.size; series += a.shape[0]
obs = tot / 1e9
print(f"lotsa_v3: {len(files)} files, {series:,} series, {obs:.2f} B observations "
      f"(reference {ref_files} files, {ref_obs:.2f} B)")
ok = len(files) == ref_files and abs(obs - ref_obs) < 0.05
print("AUDIT", "OK" if ok else "MISMATCH - check the HF revisions and the step logs")
sys.exit(0 if ok else 3)
EOF
}

if [ "${1:-}" = "--check" ]; then audit; exit $?; fi

echo "== corpus v3 rebuild, LOTSA @ $LOTSA_REV, Chronos @ $CHRONOS_REV"
df -h "$D" 2>/dev/null || mkdir -p "$D"

echo "== 1. lotsa_full (streaming from HF, resume on)"
python scripts/prepare_lotsa.py --out "$D/lotsa_full" --chunk-length 2048 \
  --max-chunks-per-subset 1000000 --resume --revision "$LOTSA_REV" 2>&1 | tee logs/corpus_1_lotsa_full.log
grep -q "EXCLUDED for evaluation overlap" logs/corpus_1_lotsa_full.log || { echo "exclusion list not printed: STOP"; exit 2; }

echo "== 2. chronos_extras"
python scripts/prepare_chronos.py --out "$D/chronos_extras" --resume --revision "$CHRONOS_REV" 2>&1 | tee logs/corpus_2_chronos.log

echo "== 3. synthetic v1 (20k chunks per family)"
[ -d "$D/synthetic" ] && [ "$(ls "$D/synthetic"/*.npy 2>/dev/null | wc -l)" -gt 0 ] \
  || python scripts/generate_synthetic.py --out "$D/synthetic" --chunks-per-family 20000

echo "== 4. lotsa_xres (symlinks)"
mkdir -p "$D/lotsa_xres"
for src in lotsa_full synthetic chronos_extras; do
  for f in "$D/$src"/*.npy; do ln -sfn "$(realpath "$f")" "$D/lotsa_xres/$(basename "$f")"; done
done

echo "== 5. synthetic_v3 (23 seeded shards)"
OUT="$D/synthetic_v3"
for s in $(seq 1 10);  do python scripts/generate_synthetic.py --set v3 --out $OUT --families synthetic_ops_bursty   --suffix _s$s --seed $s; done
for s in $(seq 11 15); do python scripts/generate_synthetic.py --set v3 --out $OUT --families synthetic_intermittent --suffix _s$s --seed $s; done
for s in 16 17 18;     do python scripts/generate_synthetic.py --set v3 --out $OUT --families synthetic_subhourly    --suffix _s$s --seed $s; done
for s in 19 20 21;     do python scripts/generate_synthetic.py --set v3 --out $OUT --families synthetic_broadband    --suffix _s$s --seed $s; done
for s in 22 23;        do python scripts/generate_synthetic.py --set v3 --out $OUT --families synthetic_lowfreq      --suffix _s$s --seed $s; done
[ "$(ls $OUT/*.npy | wc -l)" -eq 23 ] || { echo "synthetic_v3: expected 23 shards"; exit 2; }

echo "== 6. lotsa_short (padded short real series) and lotsa_solar"
python scripts/prepare_lotsa.py --out "$D/lotsa_short" --revision "$LOTSA_REV" --resume \
  --subsets m1_monthly m1_quarterly m1_yearly monash_m3_monthly monash_m3_other monash_m3_quarterly \
  monash_m3_yearly tourism_monthly tourism_quarterly tourism_yearly nn5_daily_with_missing nn5_weekly \
  --min-length 384 --chunk-length 1280 --pad-to 1280 2>&1 | tee logs/corpus_6_short.log
python scripts/prepare_lotsa.py --out "$D/lotsa_solar" --revision "$LOTSA_REV" --resume \
  --subsets solar_power 2>&1 | tee logs/corpus_6_solar.log

echo "== 7. decimated (5T -> 10T/15T)"
python scripts/decimate_corpus.py --src "$D/lotsa_xres" --dst "$D/decimated" --factors 2,3 2>&1 | tee logs/corpus_7_decimated.log

echo "== 8. lotsa_v3 (symlinks; runbook: lowfreq_dec3 and broadband_dec3 removed after audit 1)"
mkdir -p "$D/lotsa_v3"
for src in lotsa_xres synthetic_v3 lotsa_short lotsa_solar decimated; do
  for f in "$D/$src"/*.npy; do
    b=$(basename "$f")
    case "$b" in *lowfreq*_dec3*|*broadband*_dec3*) continue ;; esac
    ln -sfn "$(realpath "$f")" "$D/lotsa_v3/$b"
  done
done

echo "== 9. audit"
audit
