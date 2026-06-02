#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="${PROFILE:-80gb}"
RUN_ID="${RUN_ID:-}"
LOG_DIR="${LOG_DIR:-/workspace}"

cd "$REPO_DIR"
source "$HOME/.cargo/env"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(ls -td runs/code_poc_* | head -n1 | xargs -r basename)"
fi
test -n "$RUN_ID"

source scripts/tofy_pi_runtime_env.sh "runs/${RUN_ID}" "$PROFILE"

./target/release/jepa_ai --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl

./target/release/jepa_ai --eval-code-assistant \
  "$TOFY_WORLD_ENCODER_MODEL" \
  "$TOFY_ENCODER_VOCAB" \
  "$TOFY_WORLD_MODEL" \
  eval/code_assistant_go_hard.jsonl \
  384 "$TOFY_PROFILE_DIM" "$TOFY_PROFILE_MAX_SEQ" "$TOFY_PROFILE_LAYERS" "$TOFY_PROFILE_HEADS" "$TOFY_PROFILE_BRIDGE_DIM" "$TOFY_PROFILE_CONTEXT_SLOTS" \
  --high-world-model "$TOFY_HIGH_WORLD_MODEL" \
  --code-decoder "$JEPA_CANDLE_DECODER" \
  --code-decoder-vocab "$JEPA_CANDLE_DECODER_VOCAB" \
  --pi-agent-env \
  --go-timeout-sec 6 \
  2>&1 | tee "${LOG_DIR}/tofy-go-eval-${RUN_ID}.log"
