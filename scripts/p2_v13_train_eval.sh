#!/usr/bin/env bash
# Single v13 train → eval (v11 arch + q_mse_threshold=0.05). On failure, optional Grok repair.
#
# Usage: scripts/p2_v13_train_eval.sh
# Env: OUTPUT_DIR (default runs/p2/v13), DEVICE, AGENT_BIN, AGENT_MODEL, MAX_REPAIR_ATTEMPTS
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

OUTPUT_DIR="${OUTPUT_DIR:-$(p2_run_dir v13)}"
DEVICE="${DEVICE:-cuda}"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-3}"
AGENT_BIN="${AGENT_BIN:-cursor-agent}"
AGENT_MODEL="${AGENT_MODEL:-cursor-grok-4.5-high}"
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
    --output "$OUTPUT_DIR/eval_report_64ep_v3.json"
}

repair() {
  local rc="$1" attempt="$2"
  local tail="$LOG_DIR/failure-${attempt}.txt"
  tail -n 400 -- "$LOG_FILE" >"$tail"
  log "repair attempt $attempt via $AGENT_MODEL"
  "$AGENT_BIN" --print --trust --force --sandbox disabled \
    --workspace "$repo_root" --model "$AGENT_MODEL" \
    "Fix P2 v13 train/eval failure (exit $rc). Log: $LOG_FILE tail: $tail. Output: $OUTPUT_DIR. Minimal fix; cargo test + clippy after."
}

attempt=0
while true; do
  set +e
  run_pipeline 2>&1 | tee -a -- "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  set -e
  [[ "$rc" -eq 0 ]] && { log "done: $OUTPUT_DIR/eval_report_64ep_v3.json"; exit 0; }
  attempt=$((attempt + 1))
  [[ "$attempt" -gt "$MAX_REPAIR_ATTEMPTS" ]] && exit "$rc"
  repair "$rc" "$attempt"
done
