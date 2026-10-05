#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT [GPU] [BATCH_SIZE]"
  exit 2
fi

config=$1
checkpoint=$2
gpu=${3:-0}
batch_size=${4:-}

# These defaults can be overridden when invoking the script, e.g.
# META_KINDS=noise META_NOISE_STDS="0.1 0.3 0.5" bash ...
missing_rate=${META_MISSING_RATE:-0.5}
noise_rate=${META_NOISE_RATE:-1.0}
read -r -a kinds <<< "${META_KINDS:-missing noise}"
read -r -a noise_stds <<< "${META_NOISE_STDS:-0.3}"
read -r -a groups <<< "${META_GROUPS:-geo road poi all}"
read -r -a seeds <<< "${META_SEEDS:-47 48 49}"

run_evaluation() {
  local kind=$1
  local group=$2
  local rate=$3
  local seed=$4
  local noise_std=$5
  local command=(
    python experiments/evaluate.py
    -cfg "$config"
    -ckpt "$checkpoint"
    -g "$gpu"
  )
  if [[ -n "$batch_size" ]]; then
    command+=(-b "$batch_size")
  fi

  echo "Running kind=$kind group=$group rate=$rate seed=$seed noise_std=$noise_std"
  SEMPROTO_META_KIND="$kind" \
  SEMPROTO_META_GROUP="$group" \
  SEMPROTO_META_RATE="$rate" \
  SEMPROTO_META_NOISE_STD="$noise_std" \
  SEMPROTO_META_SEED="$seed" \
  "${command[@]}"
}

for kind in "${kinds[@]}"; do
  case "$kind" in
    missing)
      for group in "${groups[@]}"; do
        for seed in "${seeds[@]}"; do
          run_evaluation missing "$group" "$missing_rate" "$seed" 0.0
        done
      done
      ;;
    noise)
      for noise_std in "${noise_stds[@]}"; do
        for group in "${groups[@]}"; do
          for seed in "${seeds[@]}"; do
            run_evaluation noise "$group" "$noise_rate" "$seed" "$noise_std"
          done
        done
      done
      ;;
    *)
      echo "Unknown META_KINDS entry: $kind (expected missing or noise)"
      exit 2
      ;;
  esac
done
