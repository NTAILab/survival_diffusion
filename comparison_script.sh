#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset> [device]}"
DEVICE="${2:-cuda:0}"

MODELS=(
  sdpm_mlp
  rsf
  deepsurv
  deephit
  gbm_wb
  gbm_km
)

for model in "${MODELS[@]}"; do
  if [[ "$model" == "sdpm_mlp" ]]; then
    model_device="$DEVICE"
  else
    model_device="cpu"
  fi

  echo "Running $model on dataset=$DATASET device=$model_device"

  python -m sdpm.experiments.comparison \
    -data "$DATASET" \
    -model "$model" \
    -device "$model_device" \
    -table_filename "sdpm/experiments/results/results.csv" \
    -threads 16
done
