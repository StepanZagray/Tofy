#!/usr/bin/env bash
# Run P2 train → eval → arc3-eval overnight. On non-zero exit, start Cursor CLI
# agent to repair, then retry (resuming train when checkpoints exist).
#
# Usage:
#   scripts/p2_night_with_repair.sh run
#   scripts/p2_night_with_repair.sh watch-pid <pid> [log-file]
#   scripts/p2_night_with_repair.sh watch-tmux [session]
#
# Env:
#   MAX_REPAIR_ATTEMPTS  max agent repair+retry loops (default: 3)
#   AGENT_BIN            cursor agent binary (default: agent)
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

OUTPUT_DIR="${OUTPUT_DIR:-$(p2_run_dir v2)}"
ARC_DIR="${ARC_DIR:-$(p2_run_dir arc-recordings)}"
DEVICE="${DEVICE:-cuda}"
HIDDEN_DIM=128
ACTION_DIM=32
PHYSICAL_BATCH=1024
GRAD_ACCUM=1
STEPS_PER_LESSON=4096
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-3}"
AGENT_BIN="${AGENT_BIN:-agent}"
TMUX_SESSION="${TMUX_SESSION:-p2-night}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TOFY_MALLOC_TRIM_EVERY="${TOFY_MALLOC_TRIM_EVERY:-100}"

LOG_DIR="$OUTPUT_DIR/repair-logs"
mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR" "$ARC_DIR"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

preflight_profile_view() {
  if [[ ! -f "$OUTPUT_DIR/profile.jsonl" ]]; then
    log "skip pre-flight profile view (no profile.jsonl yet)"
    return 0
  fi
  log "pre-flight profile HTML -> $OUTPUT_DIR/model.html"
  cargo p2-view "$OUTPUT_DIR/profile.jsonl" --output "$OUTPUT_DIR/model.html"
}

train_cmd() {
  local -a cmd=(
    cargo run --release --features cudnn -- p2-train
    --device "$DEVICE"
    --hidden-dim "$HIDDEN_DIM"
    --action-dim "$ACTION_DIM"
    --physical-batch "$PHYSICAL_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --steps-per-lesson "$STEPS_PER_LESSON"
    --output-dir "$OUTPUT_DIR"
  )
  if [[ -d "$OUTPUT_DIR/checkpoints" ]] || [[ -f "$OUTPUT_DIR/checkpoints/latest.json" ]]; then
    cmd+=(--resume "$OUTPUT_DIR/checkpoints")
  fi
  "${cmd[@]}"
}

eval_cmd() {
  cargo run --release --features cudnn -- p2-eval \
    --device "$DEVICE" \
    --physical-batch "$PHYSICAL_BATCH" \
    --checkpoint "$OUTPUT_DIR/model.safetensors" \
    --train-config "$OUTPUT_DIR/config.json" \
    --output "$OUTPUT_DIR/eval_report.json"
}

arc3_cmd() {
  cargo run --release --features cudnn -- p2-arc3-eval \
    --device "$DEVICE" \
    --physical-batch "$PHYSICAL_BATCH" \
    --checkpoint "$OUTPUT_DIR/model.safetensors" \
    --train-config "$OUTPUT_DIR/config.json" \
    --arc-recordings-dir "$ARC_DIR" \
    --output "$OUTPUT_DIR/arc3_eval_report.json"
}

run_pipeline() {
  local log_file="$1"
  {
    echo "=== pipeline start $(date -Iseconds) ==="
    preflight_profile_view
    train_cmd
    echo "=== train done $(date -Iseconds) ==="
    eval_cmd
    echo "=== eval done $(date -Iseconds) ==="
    arc3_cmd
    echo "=== arc3-eval done $(date -Iseconds) ==="
  } 2>&1 | tee -a -- "$log_file"
}

invoke_repair_agent() {
  local exit_code="$1"
  local log_file="$2"
  local attempt="$3"
  local agent_log="$LOG_DIR/agent-repair-${attempt}-$(date +%Y%m%dT%H%M%S).log"
  local tail_file="$LOG_DIR/failure-tail-${attempt}.txt"
  local profile_trace="$OUTPUT_DIR/profile.jsonl"

  tail -n 400 -- "$log_file" >"$tail_file"

  log "runtime failure exit=$exit_code; starting Cursor agent (attempt $attempt/$MAX_REPAIR_ATTEMPTS)"
  log "failure tail: $tail_file"
  log "agent log: $agent_log"

  local profile_block=""
  if [[ -f "$profile_trace" ]]; then
    profile_block="$(cat <<PROFILE

Execution profile trace (candle-graph trace/4):
- file: $profile_trace
- view: cargo p2-view $profile_trace --output $OUTPUT_DIR/model.html
- query: cargo candle-graph query $profile_trace --kind slowest
PROFILE
)"
  fi

  local prompt
  prompt="$(cat <<EOF
You are repairing an overnight P2 run in this workspace that failed at runtime.

Hard constraints:
- Keep these settings exactly (do not change them):
  hidden_dim=$HIDDEN_DIM
  action_dim=$ACTION_DIM
  physical_batch=$PHYSICAL_BATCH
  grad_accum=$GRAD_ACCUM
  steps_per_lesson=$STEPS_PER_LESSON
  device=$DEVICE
  output_dir=$OUTPUT_DIR
  arc_recordings_dir=$ARC_DIR
- Do not invent new hyperparameters.
- Fix only what is needed for the failed command to run.
- Prefer resuming train from $OUTPUT_DIR/checkpoints when present.
- After fixing, run the smallest compile/check that proves the failure is addressed
  (e.g. cargo check / targeted cargo test, or a short dry run if appropriate).
- Do not commit or push unless asked.
${profile_block}

Failure context:
- pipeline exit code: $exit_code
- full log: $log_file
- last 400 lines also in: $tail_file
- repo: $repo_root

Read the failure log, identify the runtime error, apply a minimal fix, and leave the
tree ready so the overnight script can retry the same train→eval→arc3-eval chain.
EOF
)"

  if ! command -v "$AGENT_BIN" >/dev/null 2>&1; then
    log "ERROR: agent binary not found: $AGENT_BIN"
    return 127
  fi

  # Headless repair: print mode + trust + force (unattended overnight).
  "$AGENT_BIN" --print --trust --force --sandbox disabled \
    --workspace "$repo_root" \
    "$prompt" 2>&1 | tee -a -- "$agent_log"
}

cmd_run() {
  local attempt=0
  local log_file="$LOG_DIR/pipeline-$(date +%Y%m%dT%H%M%S).log"
  log "pipeline log: $log_file"

  while true; do
    set +e
    run_pipeline "$log_file"
    local rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      log "pipeline completed successfully"
      exit 0
    fi

    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$MAX_REPAIR_ATTEMPTS" ]]; then
      log "ERROR: still failing after $MAX_REPAIR_ATTEMPTS repair attempts; giving up (exit $rc)"
      exit "$rc"
    fi

    set +e
    invoke_repair_agent "$rc" "$log_file" "$attempt"
    local agent_rc=$?
    set -e
    if [[ "$agent_rc" -ne 0 ]]; then
      log "ERROR: Cursor agent repair failed with exit $agent_rc"
      exit "$agent_rc"
    fi
    log "repair agent finished; retrying pipeline"
  done
}

cmd_watch_pid() {
  local pid="${1:?pid required}"
  local log_file="${2:-}"
  if [[ -z "$log_file" ]]; then
    log_file="$LOG_DIR/watched-pid-${pid}.log"
    if [[ -r "/proc/$pid/fd/1" ]]; then
      # Best-effort: cannot always tee an existing process; capture tmux instead if used.
      :
    fi
  fi

  log "watching pid=$pid"
  set +e
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  set -e

  # If we weren't the parent, wait may fail; fall back to 1 on unexplained death.
  if [[ "$rc" -eq 127 ]]; then
    rc=1
  fi

  if [[ "$rc" -eq 0 ]]; then
    log "watched pid exited 0; no repair"
    exit 0
  fi

  if [[ ! -f "$log_file" ]]; then
    log_file="$LOG_DIR/watched-pid-${pid}-$(date +%Y%m%dT%H%M%S).log"
    {
      echo "watched pid $pid exited $rc at $(date -Iseconds)"
      ps -p "$pid" -o pid,etime,cmd 2>&1 || true
      if command -v tmux >/dev/null 2>&1; then
        tmux capture-pane -t "$TMUX_SESSION" -p -S -400 2>&1 || true
      fi
    } >"$log_file"
  fi

  invoke_repair_agent "$rc" "$log_file" 1
}

cmd_watch_tmux() {
  local session="${1:-$TMUX_SESSION}"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    log "ERROR: tmux session not found: $session"
    exit 2
  fi

  local pane_pid
  pane_pid="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -n1)"
  local child
  child="$(pgrep -P "$pane_pid" -a | awk '/target\/release\/tofy|cargo run/ {print $1; exit}')"
  local watch_pid="${child:-$pane_pid}"
  local log_file="$LOG_DIR/tmux-${session}-$(date +%Y%m%dT%H%M%S).log"

  log "tmux session=$session pane_pid=$pane_pid watch_pid=$watch_pid"
  log "snapshotting pane to $log_file while waiting"

  (
    while kill -0 "$watch_pid" 2>/dev/null; do
      tmux capture-pane -t "$session" -p -S -200 >>"$log_file" || true
      printf '\n--- snapshot %s ---\n' "$(date -Iseconds)" >>"$log_file"
      sleep 30
    done
  ) &
  local snap_pid=$!

  set +e
  while kill -0 "$watch_pid" 2>/dev/null; do
    sleep 5
  done
  set -e
  kill "$snap_pid" 2>/dev/null || true
  wait "$snap_pid" 2>/dev/null || true

  tmux capture-pane -t "$session" -p -S -400 >>"$log_file" || true

  # Infer failure: look for success markers / non-zero cargo/tofy hints in pane.
  if grep -Eq 'pipeline completed|p2-arc3-eval smoke complete|arc3-eval done' "$log_file"; then
    log "tmux output looks successful; no repair"
    exit 0
  fi
  if grep -Eqi 'error:|panic|SIGKILL|CUDA error|out of memory|FAILED|Address already|No such file' "$log_file"; then
    invoke_repair_agent 1 "$log_file" 1
    exit $?
  fi

  log "watched process ended without clear success; treating as failure"
  invoke_repair_agent 1 "$log_file" 1
}

usage() {
  cat <<EOF
Usage:
  $0 run
  $0 watch-pid <pid> [log-file]
  $0 watch-tmux [session]

run         Execute train→eval→arc3-eval; on failure, Cursor agent repairs and retries.
watch-pid   Wait for an existing PID; on non-zero/death, start repair agent.
watch-tmux  Watch tmux session (default: $TMUX_SESSION) and repair on apparent failure.
EOF
}

main() {
  local mode="${1:-}"
  case "$mode" in
    run) shift; cmd_run "$@" ;;
    watch-pid) shift; cmd_watch_pid "$@" ;;
    watch-tmux) shift; cmd_watch_tmux "$@" ;;
    -h|--help|help|"") usage; exit 2 ;;
    *) log "unknown mode: $mode"; usage; exit 2 ;;
  esac
}

main "$@"
