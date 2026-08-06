#!/usr/bin/env bash
# v16: v15 spatial model + stability fixes (latent RMS norm, capped rollout, no default retarget).
#
# Usage: scripts/p2_v16_train_eval.sh
# Env: OUTPUT_DIR (default runs/p2/v16), DEVICE
# Optional: LESSONS=..., RETARGET=1 to append retarget stress lesson
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

OUTPUT_DIR="${OUTPUT_DIR:-$(p2_run_dir v16)}"
DEVICE="${DEVICE:-cuda}"
LESSONS="${LESSONS:-dynamics,exploration,sequential,q_calibration,falsification}"
if [[ "${RETARGET:-0}" == "1" ]]; then
  LESSONS="${LESSONS},retarget"
fi
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline-$(date +%Y%m%dT%H%M%S).log"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

run_train() {
  local -a train=(
    cargo run --release --features cudnn -- p2-train
    --device "$DEVICE"
    --lessons "$LESSONS"
    --hidden-dim 64 --action-dim 32
    --physical-batch 1024 --grad-accum 1
    --steps-per-lesson 4096
    --sigreg-weight 0.003
    --checkpoint-every-steps 100
    --output-dir "$OUTPUT_DIR"
  )
  if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
    train+=(--resume "$OUTPUT_DIR/checkpoints")
  fi
  "${train[@]}"
}

run_eval() {
  local ckpt q
  if [[ -f "$OUTPUT_DIR/model.best.safetensors" ]]; then
    ckpt="$OUTPUT_DIR/model.best.safetensors"
  else
    ckpt="$OUTPUT_DIR/model.safetensors"
  fi
  q="$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/config.json'))['q_mse_threshold'])")"
  cargo run --release --features cudnn -- p2-eval \
    --device "$DEVICE" --physical-batch 64 \
    --checkpoint "$ckpt" \
    --train-config "$OUTPUT_DIR/config.json" \
    --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
    --q-mse-threshold "$q" \
    --output "$OUTPUT_DIR/eval_report_64ep_v5.json"
}

log "v16 train+eval → $OUTPUT_DIR lessons=$LESSONS (log: $LOG_FILE)"
{
  run_train
  run_eval
} 2>&1 | tee "$LOG_FILE"
