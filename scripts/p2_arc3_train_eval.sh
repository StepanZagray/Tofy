#!/usr/bin/env bash
# ARC-AGI-3 aligned train → eval (Stage 1b curriculum, ArcPad + patch encoder).
#
# Usage: scripts/p2_arc3_train_eval.sh
# Env: OUTPUT_DIR (default runs/p2/v14), DEVICE
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

OUTPUT_DIR="${OUTPUT_DIR:-$(p2_run_dir v14)}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline-$(date +%Y%m%dT%H%M%S).log"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

run_pipeline() {
  local -a train=(
    cargo run --release --features cudnn -- p2-train
    --device "$DEVICE"
    --hidden-dim 128 --action-dim 32
    --physical-batch 1024 --grad-accum 1
    --steps-per-lesson 4096
    --checkpoint-every-steps 100
    --output-dir "$OUTPUT_DIR"
  )
  if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
    train+=(--resume "$OUTPUT_DIR/checkpoints")
  fi
  "${train[@]}"
  local q
  q="$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/config.json'))['q_mse_threshold'])")"
  cargo run --release --features cudnn -- p2-eval \
    --device "$DEVICE" --physical-batch 64 \
    --checkpoint "$OUTPUT_DIR/model.safetensors" \
    --train-config "$OUTPUT_DIR/config.json" \
    --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
    --q-mse-threshold "$q" \
    --output "$OUTPUT_DIR/eval_report_64ep_v4.json"
}

log "ARC-AGI-3 v14 train+eval → $OUTPUT_DIR (log: $LOG_FILE)"
run_pipeline 2>&1 | tee "$LOG_FILE"
test "${PIPESTATUS[0]}" -eq 0
