#!/usr/bin/env bash
# ADR 0003 foundation-v2 launcher: deterministic mixed-stream training with
# in-trainer gates, followed by held-out synthetic and live ARC-AGI-3 eval.
#
# Usage: scripts/foundation_v2_train.sh <output-dir> [physical-batch] [steps]
# Env:   DEVICE (default cuda), SEED (default 2).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_DIR="${1:?usage: foundation_v2_train.sh <output-dir> [physical-batch] [steps]}"
PHYSICAL_BATCH="${2:-2048}"
STEPS="${3:-24576}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-2}"

cargo build --release --locked --features cudnn
BIN=target/release/tofy

RESUME_ARGS=()
if [[ -f "$RUN_DIR/checkpoints/latest.json" ]]; then
  RESUME_ARGS=(--resume "$RUN_DIR/checkpoints")
fi

"$BIN" p2-train \
  --recipe foundation-v2 --device "$DEVICE" \
  --seed "$SEED" --init-seed "$SEED" \
  --physical-batch "$PHYSICAL_BATCH" --steps "$STEPS" \
  --checkpoint-every-steps 256 \
  --output-dir "$RUN_DIR" \
  "${RESUME_ARGS[@]}"

CHECKPOINT="$RUN_DIR/checkpoints/best/ema.safetensors"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/model.safetensors"
fi

"$BIN" p2-eval \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --train-config "$RUN_DIR/config.json" \
  --physical-batch 64 --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
  --seed 1000002 --iid-seed 1000003 \
  --output "$RUN_DIR/eval_report.json"

"$BIN" p2-arc3-live-eval \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --train-config "$RUN_DIR/config.json" \
  --output "$RUN_DIR/arc3_live_report.json"
