#!/usr/bin/env bash
# Evaluate every checkpoint of a finetune run on GIFT-Eval with the official
# stack (flip + RateIN mix + pool), one after the other, and keep a digest.
#
#   scripts/eval_checkpoints.sh <checkpoint_dir> <eval_config> [extra hydra flags...]
#
#   scripts/eval_checkpoints.sh \
#       checkpoints/timejepa_lotsa_mini_v3_head8_anneal30_zs/pretrain_False \
#       lotsa_mini_v3_head8_eval
#   scripts/eval_checkpoints.sh <dir> <cfg> +ttt=norm          # stack + a layer
#
# Checkpoints are taken in creation order (oldest first). evaluate_gift caches
# per checkpoint stem, so re-running the script only evaluates what is new.
# Digest: logs/eval_<run>.log at the repo root (one block per checkpoint, then
# a table), full per-checkpoint logs in logs/eval_<run>_<stem>.log.
# Selection doctrine (G7.3c): the GIFT eval picks the champion, never val_loss.
set -u
if [ $# -lt 2 ]; then
  echo "usage: $0 <checkpoint_dir> <eval_config> [extra hydra flags...]" >&2
  exit 2
fi
DIR="$1"; CFG="$2"; shift 2
EXTRA=("$@")
RUN=$(basename "$(dirname "$DIR")")
[ "$RUN" = "checkpoints" ] && RUN=$(basename "$DIR")
mkdir -p logs
DIGEST="logs/eval_${RUN}.log"
STACK=(+tta_flip=true +ratein=mix +ratein_pool=true)
echo "== $(date '+%F %T') run=$RUN cfg=$CFG flags: ${STACK[*]} ${EXTRA[*]:-}" | tee -a "$DIGEST"

mapfile -t CKPTS < <(ls -tr "$DIR"/*.ckpt 2>/dev/null | grep -v '/last\.ckpt$')
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "no checkpoint in $DIR" | tee -a "$DIGEST"; exit 1
fi
declare -a ROWS=()
for CK in "${CKPTS[@]}"; do
  STEM=$(basename "$CK" .ckpt)
  LOG="logs/eval_${RUN}_${STEM}.log"
  echo "-- $STEM ($(date '+%T'))" | tee -a "$DIGEST"
  PYTHONUNBUFFERED=1 python scripts/evaluate_gift.py --config-name "$CFG" \
    "+checkpoint_path=$CK" "${STACK[@]}" "${EXTRA[@]}" > "$LOG" 2>&1
  RC=$?
  grep -E "vs_official_seasonal_naive|vs_local_seasonal_naive|coverage \(mean|RateIN:|TTT\[|SPREAD\[|BIAS\[|Results:" "$LOG" \
    | sed -E 's/.*INFO\] - //' | tee -a "$DIGEST"
  [ $RC -ne 0 ] && echo "   exit $RC (see $LOG)" | tee -a "$DIGEST"
  LINE=$(grep "vs_official_seasonal_naive" "$LOG" | tail -1 | sed -E 's/.*MASE ratio ([0-9.]+) \| CRPS ratio ([0-9.]+).*/\1 \2/')
  COV=$(grep "coverage (mean" "$LOG" | tail -1 | sed -E 's/.*-> ([0-9.]+).*/\1/')
  ROWS+=("$STEM ${LINE:-nan nan} ${COV:-nan}")
done
{
  echo "== table run=$RUN ($(date '+%F %T'))"
  printf "%-36s %8s %8s %8s\n" checkpoint MASE CRPS cov80
  for r in "${ROWS[@]}"; do
    set -- $r
    printf "%-36s %8s %8s %8s\n" "$1" "$2" "$3" "$4"
  done
  echo "reference head8 champion stack: 0.7842 0.5340 0.756"
} | tee -a "$DIGEST"
