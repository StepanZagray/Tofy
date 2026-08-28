#!/usr/bin/env bash
# Run the complete frozen-checkpoint evaluation campaign after a sealed P2
# training run. Local toolkit play precedes live API and recording replay.
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

if [[ -d /workspace ]]; then
  VENV="${TOFY_ARC3_VENV:-/workspace/.runpod-agent/arcagi-venv}"
else
  VENV="${TOFY_ARC3_VENV:-$HOME/.cache/tofy/arcagi-venv}"
fi

ensure_arcagi_venv() {
  if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c 'import arc_agi' >/dev/null 2>&1; then
    printf 'ARC-AGI-3 local toolkit ready: %s\n' "$VENV"
    return 0
  fi

  local interpreter=""
  if command -v python3.12 >/dev/null 2>&1; then
    interpreter="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1 && \
    python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
    interpreter="$(command -v python3)"
  fi

  if [[ -z "$interpreter" ]] && command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3.12-venv || true
    if command -v python3.12 >/dev/null 2>&1; then
      interpreter="$(command -v python3.12)"
    fi
  fi
  if [[ -z "$interpreter" ]]; then
    printf 'ERROR: ARC-AGI-3 local toolkit provisioning skipped: no Python >=3.12 interpreter is available\n' >&2
    return 1
  fi

  if ! mkdir -p "$(dirname -- "$VENV")"; then
    printf 'ERROR: ARC-AGI-3 local toolkit provisioning skipped: cannot create parent directory for %s\n' "$VENV" >&2
    return 1
  fi
  if ! "$interpreter" -m venv "$VENV"; then
    printf 'ERROR: ARC-AGI-3 local toolkit provisioning skipped: could not create venv %s\n' "$VENV" >&2
    return 1
  fi
  if ! "$VENV/bin/pip" install 'arc-agi>=0.9.9'; then
    printf 'ERROR: ARC-AGI-3 local toolkit provisioning skipped: arc-agi installation failed in %s\n' "$VENV" >&2
    return 1
  fi
  if ! "$VENV/bin/python" -c 'import arc_agi' >/dev/null 2>&1; then
    printf 'ERROR: ARC-AGI-3 local toolkit provisioning skipped: arc_agi import failed in %s\n' "$VENV" >&2
    return 1
  fi
  printf 'ARC-AGI-3 local toolkit provisioned: %s\n' "$VENV"
}

if ensure_arcagi_venv 2>&1 | tee "$EVAL_DIR/p2-arc3-local-provision.log"; then
  ARC3_LOCAL_ENVIRONMENTS="${ARC3_ENVIRONMENTS_DIR:-$(dirname -- "$VENV")/arcagi-environments}"
  if "$VENV/bin/python" python/tofy_arc3/run_local.py \
    --bin "$BIN" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --train-config "$RUN_DIR/config.json" \
    --games "${ARC3_GAMES:-all}" \
    --environments-dir "$ARC3_LOCAL_ENVIRONMENTS" \
    --allow-download \
    --output-dir "$EVAL_DIR/arc3_local" \
    --profile-eval true \
    2>&1 | tee "$EVAL_DIR/p2-arc3-local-eval.log"; then
    exit 0
  fi
  printf 'ERROR: ARC-AGI-3 local toolkit evaluation failed; trying the next evaluation source\n' \
    | tee -a "$EVAL_DIR/p2-arc3-local-eval.log" >&2
fi

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
  'SKIPPED ARC-AGI-3 EVAL: local toolkit, live API, and recordings replay were unavailable.' \
  'Set TOFY_ARC3_VENV to a usable arc-agi venv (or allow automatic provisioning),' \
  'set ARC_API_KEY (or ARC_AGI_3_API_KEY) for live play, or set RECORDINGS_DIR' \
  "to a JSONL recording tree (default checked: $RUN_DIR/../arc3-recordings)." \
  | tee "$EVAL_DIR/p2-arc3-skipped.log"
