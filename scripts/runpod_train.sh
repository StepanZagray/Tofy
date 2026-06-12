#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODE="${1:-train}"
PROFILE="${PROFILE:-80gb}"
RESUME_TARGET="${RESUME_TARGET:-latest}"
SESSION="${TMUX_SESSION:-tofy-${MODE}}"
STOP_WRAPPER="${TOFY_RUNPOD_STOP_WRAPPER:-/workspace/run-tofy-and-stop.sh}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
  echo "usage: $0 [train|resume]" >&2
  exit 2
fi
if [[ -z "${LOG_PATH:-}" ]]; then
  if [[ "$MODE" == "resume" ]]; then
    LOG_PATH="/workspace/tofy-train-${PROFILE}-resume.log"
  else
    LOG_PATH="/workspace/tofy-train-${PROFILE}.log"
  fi
fi

if [[ "${TOFY_RUNPOD_TMUX_CHILD:-0}" != "1" && -z "${TMUX:-}" ]]; then
  exec tmux new -s "$SESSION" \
    "TOFY_RUNPOD_TMUX_CHILD=1 PROFILE='$PROFILE' RESUME_TARGET='$RESUME_TARGET' SKIP_TRAINED_STAGES='${SKIP_TRAINED_STAGES:-}' LOG_PATH='$LOG_PATH' TOFY_REPO_DIR='$REPO_DIR' SKIP_GIT_PULL='${SKIP_GIT_PULL:-0}' TOFY_AUTO_STOP='${TOFY_AUTO_STOP:-1}' TOFY_RUNPOD_STOP_WRAPPER='$STOP_WRAPPER' bash '$0' '$MODE'"
fi

cd "$REPO_DIR"
source "$HOME/.cargo/env"

if [[ "${SKIP_GIT_PULL:-0}" != "1" && -d .git ]]; then
  git fetch origin
  git pull --ff-only
fi
cargo build --release

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
export TOFY_REQUIRE_PREPARED_CACHE="${TOFY_REQUIRE_PREPARED_CACHE:-1}"
export TOFY_GO_MODEL_FEEDBACK_ROWS="${TOFY_GO_MODEL_FEEDBACK_ROWS:-0}"

cmd=(./target/release/jepa_ai train "$PROFILE")
if [[ "$MODE" == "resume" ]]; then
  cmd+=(--resume "$RESUME_TARGET")
fi
if [[ -n "${SKIP_TRAINED_STAGES:-}" ]]; then
  cmd+=(--skip-trained "$SKIP_TRAINED_STAGES")
fi

if [[ "${TOFY_AUTO_STOP:-1}" == "1" && -x "$STOP_WRAPPER" ]]; then
  "$STOP_WRAPPER" "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
else
  "${cmd[@]}" 2>&1 | tee "$LOG_PATH"
fi
