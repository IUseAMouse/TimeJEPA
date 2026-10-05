#!/usr/bin/env bash
# RateIN on third-party forecasters (2026-10-05, P-RateIN-ext in the registry):
# each model goes through the SAME harness and instances as our checkpoints,
# three evaluations each - bare, flip, and the champion stack (flip + mix +
# pool + RateIN-up). One lane per GPU, models one after the other inside a
# lane; a model whose smoke test fails is skipped, the lane goes on.
#
#   nohup scripts/queue_external.sh > logs/queue_ext.out 2>&1 &
#   PY=/workspace/venv-ext/bin/python nohup scripts/queue_external.sh > logs/queue_ext.out 2>&1 &
#   LANES="chronos_bolt_tiny|ttm_r3|chronos_bolt_small" ...   one lane per GPU, models separated by spaces
#
# Needs the `external` extra (chronos-forecasting, tfc-t0) and, for ttm_r3,
# the `ttm` extra (granite-tsfm); t0_alpha needs an accepted licence and a
# Hugging Face login. Results: evaluation/<model.name>/<hf id>/gift<tag>/.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LANES=${LANES:-"chronos_bolt_tiny t0_alpha|ttm_r3 chronos2|chronos_bolt_small"}
BATCH=${BATCH:-32}
UP="+tta_flip=true +ratein=mix +ratein_pool=true +ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4"
mkdir -p logs

run() {  # run <gpu> <model> <label> [flags...]
  local gpu=$1 model=$2 label=$3; shift 3
  local log="logs/ext_${model}_${label}.out"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" scripts/evaluate_gift.py \
    --config-name "ext_${model}_eval" "+gift_batch_size=$BATCH" "$@" > "$log" 2>&1
  local status=$?
  echo "[gpu $gpu] $(date '+%T') $model $label: $(grep -h 'vs_official' "$log" | tail -n 1 | sed 's/^ *//') $(grep -h 'configs:' "$log" | tail -n 1 | sed 's/^ *//')"
  return $status
}

lane() {  # lane <gpu> <models...>
  local gpu=$1; shift
  for model in "$@"; do
    [ -f "configs/model/ext_${model}_eval.yaml" ] || { echo "[gpu $gpu] $model: no config, skipped"; continue; }
    if ! run "$gpu" "$model" smoke "+gift_configs=m_dense/D/short"; then
      echo "[gpu $gpu] $model: smoke test FAILED (tail of logs/ext_${model}_smoke.out below), skipped"
      tail -n 5 "logs/ext_${model}_smoke.out"
      continue
    fi
    run "$gpu" "$model" nu
    run "$gpu" "$model" flip +tta_flip=true
    # shellcheck disable=SC2086
    run "$gpu" "$model" up $UP
  done
}

echo "== $(date '+%F %T') lanes: $LANES (python: $PY)"
IFS='|' read -r -a LANE_ARR <<< "$LANES"
for gpu in "${!LANE_ARR[@]}"; do
  # shellcheck disable=SC2086
  lane "$gpu" ${LANE_ARR[$gpu]} &
done
wait
echo "== $(date '+%F %T') external queue done"
