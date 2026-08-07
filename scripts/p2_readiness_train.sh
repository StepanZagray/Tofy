#!/usr/bin/env bash
# Readiness train → eval with Cursor-agent repair + checkpoint resume on failure.
#
# Usage:
#   scripts/p2_readiness_train.sh run [output-dir]
#   scripts/p2_readiness_train.sh watch-pid <pid> [output-dir]
#   scripts/p2_readiness_train.sh watch-tmux [session] [output-dir]
#
# Env:
#   MAX_REPAIR_ATTEMPTS  (default: 5)
#   AGENT_BIN            (default: agent)
#   P2_DEVICE            (default: cuda)
#   P2_OUTPUT_DIR        (default: runs/p2/readiness-v2)
#   P2_RUNS_ROOT         (default: runs/p2)
#   P2_PHYSICAL_BATCH    (default: 128)
#   P2_GRAD_ACCUM        (default: 4)
#   TMUX_SESSION         (default: p2-readiness)
#
# Legacy root dirs (p2-output-*): mv p2-output-foo runs/p2/foo before resume.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/p2_runs.sh
source "$script_dir/p2_runs.sh"
cd -- "$repo_root"

MODE="${1:-run}"
shift || true

# v3: the depth-sampling, Q-target and SIGReg-row changes all alter the resume
# contract, so v2's completed checkpoints cannot (and must not) be resumed into
# this run. Keeping a new directory also preserves v2 for comparison.
OUT="${P2_OUTPUT_DIR:-$(p2_migrate_legacy_run_dir readiness-v3)}"
case "$MODE" in
  run)
    if [[ $# -ge 1 && "$1" != --* ]]; then
      OUT="$1"
      shift
    fi
    ;;
esac

DEVICE="${P2_DEVICE:-cuda}"
PHYSICAL_BATCH="${P2_PHYSICAL_BATCH:-1024}"
GRAD_ACCUM="${P2_GRAD_ACCUM:-1}"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-5}"
if [[ -z "${AGENT_BIN:-}" ]]; then
  if [[ -x "${HOME}/.local/bin/agent" ]]; then
    AGENT_BIN="${HOME}/.local/bin/agent"
  elif command -v agent >/dev/null 2>&1; then
    AGENT_BIN="agent"
  elif command -v cursor-agent >/dev/null 2>&1; then
    AGENT_BIN="cursor-agent"
  else
    AGENT_BIN="agent"
  fi
fi
TMUX_SESSION="${TMUX_SESSION:-p2-readiness}"
# Glibc's arena count is left at its default on purpose. Capping it to 2 serialised
# every malloc across the ~100 episode-generation threads: measured on an L40S at
# physical_batch=512 that cost 15x throughput (1.9 vs 28.6 steps/min) and left the GPU
# idle ~90% of the time. RSS is held down by the periodic malloc_trim below instead.
# Periodic malloc_trim in the train loop (optimizer steps); 0 disables.
export TOFY_MALLOC_TRIM_EVERY="${TOFY_MALLOC_TRIM_EVERY:-100}"
LOG_DIR="$OUT/repair-logs"
mkdir -p -- "$OUT" "$LOG_DIR"

FEATURES=""
if command -v nvidia-smi >/dev/null 2>&1; then
  FEATURES="--features cudnn"
fi

TOFY_BIN="${TOFY_BIN:-}"
if [[ -z "$TOFY_BIN" && -x "$repo_root/target/release/tofy" ]]; then
  TOFY_BIN="$repo_root/target/release/tofy"
fi

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$OUT/run.log"; }

find_training_pid() {
  local pid_file="$OUT/train.pid"
  if [[ -f "$pid_file" ]]; then
    local tp
    tp="$(tr -d '[:space:]' <"$pid_file")"
    if [[ -n "$tp" ]] && kill -0 "$tp" 2>/dev/null; then
      if tr '\0' ' ' </proc/"$tp"/cmdline 2>/dev/null | grep -q 'p2-train'; then
        echo "$tp"
        return 0
      fi
    fi
  fi
  pgrep -af "target/release/tofy p2-train.*--output-dir $OUT" 2>/dev/null \
    | awk 'NR==1 {print $1}'
}

training_still_running() {
  [[ -n "$(find_training_pid)" ]]
}

resolve_ckpt() {
  python3 - <<'PY' "$OUT"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
latest = out / "checkpoints" / "latest.json"
if latest.is_file():
    d = json.loads(latest.read_text())["directory"]
    print(out / "checkpoints" / d / "model.safetensors")
else:
    print(out / "model.safetensors")
PY
}

cargo_p2() {
  if [[ -n "$TOFY_BIN" ]]; then
    "$TOFY_BIN" "$@"
  else
    cargo run --release $FEATURES -- "$@"
  fi
}

train_cmd() {
  wait_for_gpu_idle
  local stale_eval
  stale_eval="$(pgrep -af "tofy p2-eval.*$OUT" 2>/dev/null | awk '{print $1}' || true)"
  for ep in $stale_eval; do
    if [[ -n "$ep" ]] && kill -0 "$ep" 2>/dev/null; then
      log "stopping stale p2-eval pid $ep before train"
      kill "$ep" 2>/dev/null || true
      sleep 1
    fi
  done
  local -a cmd=(
    cargo_p2 p2-train
    --device "$DEVICE"
    --seed 1
    --lessons dynamics,exploration,sequential,q_calibration,falsification
    --steps-per-lesson 4096
    --physical-batch "$PHYSICAL_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --checkpoint-every-steps 500
    --randomize-depth
    --ptrm-rank-every 16
    --supervise-last-outer-only
    --sigreg-spatial-pool
    --residual-y-update
    --warm-start-y
    --sigreg-spatial
    --shuffled-episodes
    --prefix-weight 0.1
    --reliability-weight 0.1
    # Q labels are a threshold on the model's own latent error. With a fixed
    # absolute threshold that label collapses to all-ones as training fits
    # (falsification reached next_latent 0.0022 against a 0.05 threshold), so Q
    # scored ~0 loss by always predicting "reliable" and was uncalibrated on
    # held-out data. The median-quantile target keeps both classes populated.
    --q-quantile-targets
    # SIGReg estimates a distributional statistic, so it needs rows; the full
    # pooled stack at batch 1024 is 32768 rows and costs ~17 MiB.
    --sigreg-max-rows 32768
    --outer-steps 8
    --inner-steps 2
    --hidden-dim 128
    --output-dir "$OUT"
  )
  if [[ -f "$OUT/checkpoints/latest.json" ]] || [[ -d "$OUT/checkpoints" ]]; then
    cmd+=(--resume "$OUT/checkpoints")
    log "resuming from $OUT/checkpoints"
  fi
  "${cmd[@]}"
}

wait_for_gpu_idle() {
  local pid
  pid="$(find_training_pid)"
  if [[ -n "$pid" ]]; then
    log "waiting for training pid $pid before GPU work"
    while kill -0 "$pid" 2>/dev/null; do sleep 2; done
  fi
  local eval_pids
  eval_pids="$(pgrep -af "tofy p2-eval.*$OUT" 2>/dev/null | awk '{print $1}' || true)"
  for ep in $eval_pids; do
    if [[ -n "$ep" ]] && kill -0 "$ep" 2>/dev/null; then
      log "waiting for stale p2-eval pid $ep"
      while kill -0 "$ep" 2>/dev/null; do sleep 2; done
    fi
  done
}

eval_cmd() {
  wait_for_gpu_idle
  local ckpt
  ckpt="$(resolve_ckpt)"
  cargo_p2 p2-eval \
    --checkpoint "$ckpt" \
    --train-config "$OUT/config.json" \
    --device "$DEVICE" \
    --seed 2 \
    --synthetic-episodes 32 \
    --physical-batch 8 \
    --ensemble-members 8 \
    --episode-jsonl "$OUT/episodes.jsonl" \
    --output "$OUT/eval_report.json"
}

run_pipeline() {
  local log_file="$1"
  local stage_log rc
  stage_log="$(mktemp)"
  set -o pipefail
  {
    echo "=== readiness pipeline start $(date -Iseconds) ==="
    if [[ -z "$TOFY_BIN" ]]; then
      cargo build --release $FEATURES
    else
      echo "using prebuilt binary: $TOFY_BIN"
    fi
    train_cmd
  } >"$stage_log" 2>&1
  rc=$?
  cat "$stage_log" | tee -a "$log_file"
  rm -f "$stage_log"
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
  {
    echo "=== train done $(date -Iseconds) ==="
    eval_cmd
    echo "=== eval done $(date -Iseconds) ==="
  } >"$stage_log" 2>&1
  rc=$?
  cat "$stage_log" | tee -a "$log_file"
  rm -f "$stage_log"
  return "$rc"
}

invoke_repair_agent() {
  local exit_code="$1"
  local log_file="$2"
  local attempt="$3"
  local agent_log="$LOG_DIR/agent-repair-${attempt}-$(date +%Y%m%dT%H%M%S).log"
  local tail_file="$LOG_DIR/failure-tail-${attempt}.txt"
  local profile_trace="$OUT/profile.jsonl"
  local latest_ckpt="none"
  if [[ -f "$OUT/checkpoints/latest.json" ]]; then
    latest_ckpt="$(cat "$OUT/checkpoints/latest.json")"
  fi

  tail -n 400 -- "$log_file" >"$tail_file"
  log "failure exit=$exit_code; Cursor agent repair attempt $attempt/$MAX_REPAIR_ATTEMPTS"
  log "tail: $tail_file agent log: $agent_log"

  local profile_block=""
  if [[ -f "$profile_trace" ]]; then
    profile_block="$(cat <<PROFILE

Profile trace:
- $profile_trace
- cargo p2-view $profile_trace --output $OUT/model.html
PROFILE
)"
  fi

  local prompt
  prompt="$(cat <<EOF
Repair a failed P2 readiness training run in this workspace.

Constraints:
- output_dir=$OUT (resume from $OUT/checkpoints when present; latest: $latest_ckpt)
- Keep: physical_batch=128 grad_accum=4 steps_per_lesson=4096 device=$DEVICE
- Build with: cargo build --release --features cudnn
- Fix the runtime failure minimally (NaN loss, CUDA OOM, cudnn error, compile error, hang on exit, etc.).
- Lesson-gate prefix/reliability: dynamics/exploration must NOT train prefix or reliability.
- After fixing, run cargo test --lib p2::train and a short sanity p2-train step if needed.
- Do not commit or push.

Failure:
- exit code: $exit_code
- log: $log_file
- tail: $tail_file
${profile_block}

Leave the tree ready for scripts/p2_readiness_train.sh to resume train from checkpoint and finish eval.
EOF
)"

  if ! command -v "$AGENT_BIN" >/dev/null 2>&1; then
    log "ERROR: agent binary not found: $AGENT_BIN (install Cursor CLI or set AGENT_BIN)"
    return 127
  fi

  "$AGENT_BIN" --print --trust --force --sandbox disabled \
    --workspace "$repo_root" \
    "$prompt" 2>&1 | tee -a "$agent_log"
}

capture_failure_log() {
  local log_file="$1"
  local pid="${2:-}"
  {
    echo "=== failure capture $(date -Iseconds) ==="
    if [[ -n "$pid" ]]; then
      echo "watched pid=$pid"
      ps -p "$pid" -o pid,etime,stat,cmd 2>&1 || true
    fi
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      echo "=== tmux session $TMUX_SESSION (last 400 lines) ==="
      tmux capture-pane -t "$TMUX_SESSION" -p -S -400 2>&1 || true
    fi
    if [[ -f "$OUT/run.log" ]]; then
      echo "=== run.log tail ==="
      tail -n 200 "$OUT/run.log" || true
    fi
    if [[ -f "$OUT/train_report.json" ]]; then
      echo "=== train_report.json ==="
      cat "$OUT/train_report.json"
    fi
  } >"$log_file"
}

repair_retry_loop() {
  local initial_rc="${1:-1}"
  local log_file="$2"
  local attempt=0
  local rc="$initial_rc"

  while [[ "$rc" -ne 0 ]]; do
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$MAX_REPAIR_ATTEMPTS" ]]; then
      log "ERROR: failed after $MAX_REPAIR_ATTEMPTS repair attempts (exit $rc)"
      return "$rc"
    fi

    set +e
    invoke_repair_agent "$rc" "$log_file" "$attempt"
    local agent_rc=$?
    set -e
    if [[ "$agent_rc" -ne 0 ]]; then
      log "ERROR: repair agent exit $agent_rc"
      return "$agent_rc"
    fi

    log "repair finished; retrying pipeline (will resume checkpoints if present)"
    set +e
    run_pipeline "$log_file"
    rc=$?
    set -e
  done

  log "readiness pipeline completed: $OUT/eval_report.json"
  return 0
}

cmd_run() {
  local pipeline_log="$LOG_DIR/pipeline-$(date +%Y%m%dT%H%M%S).log"
  log "pipeline log: $pipeline_log"
  log "repair watcher: failures spawn $AGENT_BIN (max $MAX_REPAIR_ATTEMPTS attempts)"

  set +e
  run_pipeline "$pipeline_log"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    log "readiness pipeline completed: $OUT/eval_report.json"
    exit 0
  fi

  repair_retry_loop "$rc" "$pipeline_log"
}

cmd_watch_pid() {
  local pid="${1:?pid required}"
  local pipeline_log="$LOG_DIR/watched-pid-${pid}-$(date +%Y%m%dT%H%M%S).log"
  log "watching training pid=$pid output_dir=$OUT"
  log "on failure: spawn $AGENT_BIN then resume train+eval (max $MAX_REPAIR_ATTEMPTS repairs)"
  log "pipeline log: $pipeline_log"

  set +e
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  set -e

  # wait(1) returns 127 when the pid is not our child (common for watch-pid).
  if [[ "$rc" -eq 127 ]] && training_still_running; then
    local new_pid
    new_pid="$(find_training_pid)"
    log "watched pid $pid is not our child but training still running (pid=$new_pid); re-attaching"
    cmd_watch_pid "$new_pid"
    return $?
  fi

  if [[ "$rc" -eq 127 || "$rc" -eq 1 ]] && ! kill -0 "$pid" 2>/dev/null; then
    # Not our child: infer from train_report / tmux if process died abnormally.
    if training_still_running; then
      local new_pid
      new_pid="$(find_training_pid)"
      log "watched pid $pid exited (rc=$rc) but training still running (pid=$new_pid); re-attaching"
      cmd_watch_pid "$new_pid"
      return $?
    fi
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      if tmux capture-pane -t "$TMUX_SESSION" -p -S -80 2>/dev/null | grep -Eqi 'error:|panic|CudnnError|CUDA error|out of memory'; then
        rc=1
      elif tmux capture-pane -t "$TMUX_SESSION" -p -S -20 2>/dev/null | grep -Eq 'p2-train status=Completed'; then
        rc=0
      fi
    fi
  fi

  capture_failure_log "$pipeline_log" "$pid"

  if [[ "$rc" -eq 0 ]]; then
    log "watched pid $pid exited cleanly; running eval if train finished"
    set +e
    {
      echo "=== post-watch eval $(date -Iseconds) ==="
      eval_cmd
    } >>"$pipeline_log" 2>&1
    local eval_rc=$?
    cat "$pipeline_log" | tee -a "$OUT/run.log"
    set -e
    if [[ "$eval_rc" -eq 0 ]]; then
      log "train+eval complete"
      exit 0
    fi
    repair_retry_loop "$eval_rc" "$pipeline_log"
    exit $?
  fi

  log "watched pid $pid failed (rc=$rc); starting repair loop"
  repair_retry_loop "$rc" "$pipeline_log"
}

cmd_watch_tmux() {
  local session="${1:-$TMUX_SESSION}"
  if ! command -v tmux >/dev/null 2>&1; then
    log "ERROR: tmux not installed"
    exit 2
  fi
  if ! tmux has-session -t "$session" 2>/dev/null; then
    log "ERROR: tmux session not found: $session"
    exit 2
  fi

  local pane_pid child watch_pid
  pane_pid="$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -n1)"
  child="$(pgrep -P "$pane_pid" -a 2>/dev/null | awk '/target\/release\/tofy|cargo run/ {print $1; exit}')"
  watch_pid="${child:-$pane_pid}"
  log "tmux session=$session watch_pid=$watch_pid"
  cmd_watch_pid "$watch_pid"
}

usage() {
  cat <<EOF
Usage:
  $0 run [output-dir]
  $0 watch-pid <pid> [output-dir]
  $0 watch-tmux [session] [output-dir]

run         Train → eval; on failure spawn Cursor agent and retry (checkpoint resume).
watch-pid   Attach to a running tofy PID; on exit/error, repair and resume pipeline.
watch-tmux  Same as watch-pid for the training process in a tmux session.

Env: P2_OUTPUT_DIR P2_DEVICE AGENT_BIN MAX_REPAIR_ATTEMPTS TMUX_SESSION TOFY_BIN
EOF
}

case "$MODE" in
  run) cmd_run "$@" ;;
  watch-pid)
    pid="${1:?pid required}"
    shift
    if [[ $# -ge 1 ]]; then OUT="$1"; shift; fi
    cmd_watch_pid "$pid" "$@"
    ;;
  watch-tmux)
    session="${1:-$TMUX_SESSION}"
    shift || true
    if [[ $# -ge 1 ]]; then OUT="$1"; shift; fi
    cmd_watch_tmux "$session" "$@"
    ;;
  -h|--help|help) usage; exit 0 ;;
  *) log "unknown mode: $MODE"; usage; exit 2 ;;
esac
