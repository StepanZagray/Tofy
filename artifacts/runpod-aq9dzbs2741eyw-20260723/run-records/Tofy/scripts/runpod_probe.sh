#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="${PROFILE:-80gb}"
STAGE="${PROBE_STAGE:-all}"
SESSION="${TMUX_SESSION:-tofy-probe}"
PROBE_DIR="${PROBE_DIR:-/workspace/tofy-vram-probe-${PROFILE}}"
LOG_PATH="${LOG_PATH:-/workspace/tofy-vram-probe-${PROFILE}.log}"

if [[ "${TOFY_RUNPOD_TMUX_CHILD:-0}" != "1" && -z "${TMUX:-}" ]]; then
  exec tmux new -s "$SESSION" \
    "TOFY_RUNPOD_TMUX_CHILD=1 PROFILE='$PROFILE' PROBE_STAGE='$STAGE' PROBE_DIR='$PROBE_DIR' LOG_PATH='$LOG_PATH' TOFY_REPO_DIR='$REPO_DIR' bash '$0'"
fi

cd "$REPO_DIR"
source "$HOME/.cargo/env"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

./target/release/jepa_ai --max-vram-probe --profile "$PROFILE" --stage "$STAGE" --probe-dir "$PROBE_DIR" \
  2>&1 | tee "$LOG_PATH"
