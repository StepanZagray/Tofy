#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODE="${1:-train}"
RESUME_TARGET="${RESUME_TARGET:-latest}"
SESSION="${TMUX_SESSION:-tofy-${MODE}}"

if [[ "$MODE" != "train" && "$MODE" != "resume" ]]; then
  echo "usage: $0 [train|resume]" >&2
  exit 2
fi
if [[ -z "${LOG_PATH:-}" ]]; then
  if [[ "$MODE" == "resume" ]]; then
    LOG_PATH="/workspace/tofy-train-minimal-resume.log"
  else
    LOG_PATH="/workspace/tofy-train-minimal.log"
  fi
fi

is_sensitive_env_name() {
  local upper_name="${1^^}"
  case "$upper_name" in
    TOKEN|API_KEY|API_TOKEN|ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN|BEARER_TOKEN|\
    PASSWORD|PASSWD|SECRET|CLIENT_SECRET|CREDENTIAL|CREDENTIALS|COOKIE|\
    PRIVATE_KEY|SSH_KEY|\
    *_KEY|*_TOKEN|*_API_KEY|*_API_TOKEN|*_ACCESS_TOKEN|*_REFRESH_TOKEN|\
    *_AUTH|*_AUTH_TOKEN|\
    *_BEARER_TOKEN|*_PASSWORD|*_PASSWD|*_SECRET|*_CLIENT_SECRET|\
    *_CREDENTIAL|*_CREDENTIALS|*_COOKIE|*_PRIVATE_KEY|*_SSH_KEY)
      return 0
      ;;
  esac
  return 1
}

is_sensitive_env_value() {
  local value="$1"
  if [[ "$value" =~ ^([Bb]earer|[Bb]asic)[[:space:]]+[^[:space:]] ]] ||
    [[ "$value" =~ ^sk-[[:alnum:]_-]{16,}$ ]] ||
    [[ "$value" =~ ^hf_[[:alnum:]]{20,}$ ]] ||
    [[ "$value" =~ ^(ghp_|github_pat_)[[:alnum:]_]{20,}$ ]] ||
    [[ "$value" =~ ^xox[baprs]-[[:alnum:]-]{10,}$ ]] ||
    [[ "$value" =~ ^AKIA[[:alnum:]]{16}$ ]] ||
    [[ "$value" == *"-----BEGIN "*"PRIVATE KEY-----"* ]]; then
    return 0
  fi
  return 1
}

if [[ "${TOFY_RUNPOD_TMUX_CHILD:-0}" != "1" && -z "${TMUX:-}" ]]; then
  tmux_env=(
    -e "TOFY_RUNPOD_TMUX_CHILD=1"
    -e "RESUME_TARGET=$RESUME_TARGET"
    -e "LOG_PATH=$LOG_PATH"
    -e "TOFY_REPO_DIR=$REPO_DIR"
  )
  while IFS= read -r name; do
    case "$name" in
      TOFY_RUNPOD_TMUX_CHILD|TOFY_REPO_DIR)
        continue
        ;;
      TOFY_*|SKIP_*)
        ;;
      *)
        continue
        ;;
    esac
    if is_sensitive_env_name "$name"; then
      printf 'Not forwarding credential-like variable %s into tmux; put it in /workspace/tofy-runpod.env instead.\n' "$name" >&2
      continue
    fi
    declaration="$(declare -p "$name" 2>/dev/null || true)"
    if [[ "$declaration" == "declare -a "* || "$declaration" == "declare -A "* ]]; then
      continue
    fi
    value="${!name}"
    if is_sensitive_env_value "$value"; then
      printf 'Not forwarding variable %s because its value looks like a credential; put it in /workspace/tofy-runpod.env instead.\n' "$name" >&2
      continue
    fi
    tmux_env+=(-e "$name=$value")
  done < <(compgen -v)

  # Keep each assignment as its own argv item. Besides preserving whitespace,
  # this makes the forwarding contract testable with a mocked `tmux` function.
  exec tmux new-session -s "$SESSION" "${tmux_env[@]}" bash "$0" "$MODE"
fi

cd "$REPO_DIR"
source "$HOME/.cargo/env"
if [[ -f /workspace/tofy-runpod.env ]]; then
  # shellcheck disable=SC1091
  source /workspace/tofy-runpod.env
fi
BINARY="${TOFY_BINARY:-./target/release/tofy}"

if [[ -z "${SKIP_GIT_PULL:-}" ]]; then
  if [[ "$MODE" == "resume" ]]; then
    SKIP_GIT_PULL=1
  else
    SKIP_GIT_PULL=0
  fi
fi
if [[ "$SKIP_GIT_PULL" != "1" && -d .git ]]; then
  git fetch origin
  git pull --ff-only
fi
cargo build --release

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
# Veclab pipeline builds its own vocab in stage 1. Set to 1 only when restoring
# a prepared HF cache handoff that must be treated as authoritative.
export TOFY_REQUIRE_PREPARED_CACHE="${TOFY_REQUIRE_PREPARED_CACHE:-0}"

cmd=("$BINARY" train minimal)
if [[ "$MODE" == "resume" ]]; then
  cmd+=(--resume "$RESUME_TARGET")
fi
if [[ -n "${SKIP_TRAINED_STAGES:-}" ]]; then
  cmd+=(--skip-trained "$SKIP_TRAINED_STAGES")
fi

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"
