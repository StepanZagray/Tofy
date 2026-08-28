#!/usr/bin/env bash
# Run the complete frozen-checkpoint evaluation campaign after a sealed P2
# training run. Public ARC live play takes precedence over recording replay.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_DIR="${1:?usage: p2_post_train_eval.sh RUN_DIR}"
TRAIN_REPORT="$RUN_DIR/train_report.json"
EVAL_DIR="$RUN_DIR/eval"
DEVICE="${DEVICE:-cuda}"
BIN="${TOFY_BIN:-target/release/tofy}"

if [[ ! -f "$TRAIN_REPORT" ]] || ! jq -e '.status == "completed"' "$TRAIN_REPORT" >/dev/null; then
  printf 'ERROR: refusing post-training evaluation: %s must exist and have status=completed\n' \
    "$TRAIN_REPORT" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  printf 'ERROR: evaluator binary %s is missing; build the reviewed revision first\n' "$BIN" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"
CHECKPOINT="$RUN_DIR/checkpoints/best/ema.safetensors"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/model.safetensors"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  printf 'ERROR: no evaluation checkpoint found under %s\n' "$RUN_DIR" >&2
  exit 1
fi

"$BIN" p2-eval \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --train-config "$RUN_DIR/config.json" \
  --physical-batch "${EVAL_PHYSICAL_BATCH:-64}" \
  --synthetic-episodes "${EVAL_SYNTHETIC_EPISODES:-64}" \
  --ptrm-k "${EVAL_PTRM_K:-1,2,4,8}" \
  --seed "${EVAL_SEED:-1000002}" \
  --iid-seed "${EVAL_IID_SEED:-1000003}" \
  --eval-mode full \
  --profile-eval true \
  --output "$EVAL_DIR/eval_report.json" \
  2>&1 | tee "$EVAL_DIR/p2-eval.log"

if [[ -n "${ARC_API_KEY:-${ARC_AGI_3_API_KEY:-}}" ]]; then
  "$BIN" p2-arc3-live-eval \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --train-config "$RUN_DIR/config.json" \
    --recordings-dir "$EVAL_DIR/recordings" \
    --profile-eval true \
    --output "$EVAL_DIR/arc3_live_report.json" \
    2>&1 | tee "$EVAL_DIR/p2-arc3-live-eval.log"
  exit 0
fi

ARC_RECORDINGS="${RECORDINGS_DIR:-$RUN_DIR/../arc3-recordings}"
if [[ -d "$ARC_RECORDINGS" ]] && \
  [[ -n "$(find "$ARC_RECORDINGS" -type f -name '*.jsonl' -print -quit)" ]]; then
  "$BIN" p2-arc3-eval \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --train-config "$RUN_DIR/config.json" \
    --physical-batch "${ARC3_PHYSICAL_BATCH:-128}" \
    --arc-recordings-dir "$ARC_RECORDINGS" \
    --profile-eval true \
    --output "$EVAL_DIR/arc3_eval_report.json" \
    2>&1 | tee "$EVAL_DIR/p2-arc3-eval.log"
  exit 0
fi

printf '%s\n' \
  'SKIPPED ARC-AGI-3 EVAL: no API key or non-empty recordings directory was found.' \
  'Set ARC_API_KEY (or ARC_AGI_3_API_KEY) for live play, or set RECORDINGS_DIR' \
  "to a JSONL recording tree (default checked: $RUN_DIR/../arc3-recordings)." \
  | tee "$EVAL_DIR/p2-arc3-skipped.log"
