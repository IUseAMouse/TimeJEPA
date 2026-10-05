#!/usr/bin/env bash
# MASE fix of 2026-10-05 (pooled over valid observations, as gluonts does): the
# cached per-config results of the 15 configs whose targets are partly NaN
# carry the old MASE. This moves them aside (never deletes) so the next
# evaluate_gift.py run on the same directory recomputes those 15 and keeps the
# other 82 from the cache. CRPS and every NaN-free config are unaffected.
#   scripts/requeue_nan_configs.sh <run_dir> [more run dirs...]     # .../gift<tag>/
set -eu
CONFIGS="bitbrains_fast_storage/5T/long bitbrains_fast_storage/5T/medium bitbrains_fast_storage/5T/short
bitbrains_fast_storage/H/short bitbrains_rnd/5T/long bitbrains_rnd/5T/medium bitbrains_rnd/5T/short
bitbrains_rnd/H/short hierarchical_sales/D/short kdd_cup_2018/D/short kdd_cup_2018/H/long
kdd_cup_2018/H/medium kdd_cup_2018/H/short restaurant/D/short temperature_rain/D/short"
[ $# -ge 1 ] || { echo "usage: $0 <run_dir> [more run dirs...]" >&2; exit 2; }
for run in "$@"; do
  [ -d "$run/per_config" ] || { echo "$run: no per_config directory, skipped"; continue; }
  keep="$run/per_config_before_2026-10-05"
  mkdir -p "$keep"
  moved=0
  for c in $CONFIGS; do
    f="$run/per_config/$(echo "$c" | sed 's#/#__#g').json"
    if [ -f "$f" ]; then mv "$f" "$keep/"; moved=$((moved + 1)); fi
  done
  echo "$run: $moved configs moved to $(basename "$keep")/"
done
