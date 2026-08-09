#!/usr/bin/env bash
# Run one preregistered SIGReg/action-conditioning or geometry arm to updates 1,000 and 2,000.
# Usage: scripts/p2_sigreg_action_ab.sh <control|projector|pre-rms-spatial> <training-seed>
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
ab_root="${P2_AB_ROOT:-$repo_root/runs/p2/ab-sigreg-action-v1}"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
eval_seed="${P2_AB_EVAL_SEED:-424242}"
target_update="${P2_AB_TARGET_UPDATE:-2000}"
experiment="${P2_AB_EXPERIMENT:-action}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
recovery_only_update="${P2_AB_RECOVERY_ONLY_UPDATE:-}"
repeat_eval_update="${P2_AB_REPEAT_EVAL_UPDATE:-}"
gpu_sample_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

arm="${1:?usage: $0 <control|projector|pre-rms-spatial> <training-seed>}"
seed="${2:?usage: $0 <control|projector> <training-seed>}"
case "$arm" in
  control|projector|pre-rms-spatial) ;;
  *) printf 'invalid arm: %s\n' "$arm" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || { printf 'seed must be a positive integer\n' >&2; exit 2; }
[[ "$target_update" == 2000 || "$target_update" == 4000 ]] || {
  printf 'P2_AB_TARGET_UPDATE must be 2000 or 4000\n' >&2
  exit 2
}
[[ -z "$recovery_only_update" || "$recovery_only_update" =~ ^(1000|2000|4000)$ ]] || {
  printf 'P2_AB_RECOVERY_ONLY_UPDATE must be empty, 1000, 2000, or 4000\n' >&2
  exit 2
}
[[ -z "$repeat_eval_update" || "$repeat_eval_update" =~ ^(1000|2000|4000)$ ]] || {
  printf 'P2_AB_REPEAT_EVAL_UPDATE must be empty, 1000, 2000, or 4000\n' >&2
  exit 2
}
[[ "$gpu_sample_interval" =~ ^([1-9][0-9]*([.][0-9]+)?|0[.][0-9]*[1-9][0-9]*)$ ]] || {
  printf 'P2_GPU_SAMPLE_INTERVAL must be a positive number\n' >&2
  exit 2
}
[[ "$experiment" == action || "$experiment" == geometry ]] || {
  printf 'P2_AB_EXPERIMENT must be action or geometry\n' >&2
  exit 2
}
case "$experiment:$arm" in
  action:control|action:projector|geometry:control|geometry:pre-rms-spatial) ;;
  *) printf 'arm %s is invalid for %s experiment\n' "$arm" "$experiment" >&2; exit 2 ;;
esac
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }
for required in git jq nvidia-smi sha256sum awk tee python3 realpath; do
  command -v "$required" >/dev/null || { printf 'missing required command: %s\n' "$required" >&2; exit 2; }
done
: "${P2_EXPECTED_SHA:?set P2_EXPECTED_SHA to the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set P2_EXPECTED_CANDLE_SHA to the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set P2_EXPECTED_BINARY_SHA to the reviewed tofy binary hash}"
[[ -d "$candle_root/.git" ]] || { printf 'missing candle_graph checkout: %s\n' "$candle_root" >&2; exit 2; }

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "$git_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'HEAD %s does not match P2_EXPECTED_SHA %s\n' "$git_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi
candle_git_sha="$(git -C "$candle_root" rev-parse HEAD)"
if [[ "$candle_git_sha" != "$P2_EXPECTED_CANDLE_SHA" ]]; then
  printf 'candle_graph HEAD %s does not match P2_EXPECTED_CANDLE_SHA %s\n' \
    "$candle_git_sha" "$P2_EXPECTED_CANDLE_SHA" >&2
  exit 2
fi
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || {
  printf 'Tofy tracked worktree is dirty; refusing an attributed experiment\n' >&2; exit 2;
}
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || {
  printf 'candle_graph tracked worktree is dirty; refusing an attributed experiment\n' >&2; exit 2;
}
binary_sha256="$(sha256sum "$tofy_bin" | awk '{print $1}')"
if [[ "$binary_sha256" != "$P2_EXPECTED_BINARY_SHA" ]]; then
  printf 'binary SHA-256 %s does not match P2_EXPECTED_BINARY_SHA %s\n' \
    "$binary_sha256" "$P2_EXPECTED_BINARY_SHA" >&2
  exit 2
fi

mkdir -p -- "$ab_root"
ab_root="$(realpath "$ab_root")"
arm_dir="$ab_root/seed-$seed/$arm"
mkdir -p -- "$arm_dir" "$arm_dir/telemetry"
commands_log="$arm_dir/commands.log"
phase_log="$arm_dir/phases.jsonl"

artifact_ref() {
  realpath --relative-to="$ab_root" "$1"
}

verify_sha_manifest() {
  python3 "$script_dir/p2_artifacts.py" --root "$ab_root" --sha-file "$1"
}

record_command() {
  local stage="$1"
  shift
  {
    printf '[%s] %s' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage"
    printf ' %q' "$@"
    printf '\n'
  } >>"$commands_log"
}

record_phase() {
  local stage="$1" started="$2" finished="$3" status="$4" attempt="$5" log_path="$6" time_path="$7"
  log_path="$(artifact_ref "$log_path")"
  time_path="$(artifact_ref "$time_path")"
  jq -nc \
    --arg stage "$stage" \
    --arg started_utc "$started" \
    --arg finished_utc "$finished" \
    --arg status "$status" \
    --argjson attempt "$attempt" \
    --arg log_path "$log_path" \
    --arg time_path "$time_path" \
    --arg git_sha "$git_sha" \
    --arg candle_git_sha "$candle_git_sha" \
    --arg binary_sha256 "$binary_sha256" \
    --arg arm "$arm" \
    --argjson seed "$seed" \
    '{stage:$stage,attempt:$attempt,started_utc:$started_utc,finished_utc:$finished_utc,status:$status,git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,arm:$arm,seed:$seed,log_path:$log_path,time_path:$time_path}' \
    >>"$phase_log"
}

run_phase() {
  local stage="$1" log_path="$2"
  shift 2
  local started finished started_epoch finished_epoch rc attempt resolved_log time_path
  attempt=1
  resolved_log="$log_path"
  time_path="$arm_dir/$stage.time"
  while [[ -e "$resolved_log" || -e "$time_path" ]]; do
    attempt=$((attempt + 1))
    resolved_log="${log_path%.log}.attempt-$attempt.log"
    time_path="$arm_dir/$stage.attempt-$attempt.time"
  done
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  started_epoch="$(date -u +%s)"
  record_command "$stage" "$@"
  set +e
  "$@" > >(tee "$resolved_log") 2>&1
  rc=$?
  set -e
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  finished_epoch="$(date -u +%s)"
  printf 'wall_seconds=%s exit_status=%s\n' \
    "$((finished_epoch - started_epoch))" "$rc" >"$time_path"
  record_phase "$stage" "$started" "$finished" "$rc" "$attempt" "$resolved_log" "$time_path"
  return "$rc"
}

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_sample_interval"
  done
}

sampler_pid=""
stop_sampler() {
  if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap stop_sampler EXIT INT TERM
sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
sampler_pid=$!

common_train=(
  "$tofy_bin" p2-train
  --device cuda
  --seed "$seed"
  --lessons dynamics
  # Dynamics receives 2× this base budget: 4,000 total updates are fixed in
  # the exact contract so a preregistered extension needs no schedule mutation.
  --steps-per-lesson 2000
  --physical-batch 1024
  --grad-accum 1
  --checkpoint-every-steps 500
  --randomize-depth
  --supervise-last-outer-only
  --residual-y-update
  --warm-start-y
  --sigreg-max-rows 32768
  --shuffled-episodes
  --outer-steps 8
  --inner-steps 2
  --hidden-dim 128
  --action-dim 8
  --event-weight 0
  --q-weight 0
  --rollout-weight 0
  --prefix-weight 0
  --reliability-weight 0
  --ensemble-members 1
  --output-dir "$arm_dir"
)
case "$arm" in
  control) common_train+=(--sigreg-spatial --sigreg-spatial-pool) ;;
  projector) common_train+=(--sigreg-spatial --sigreg-spatial-pool --sigreg-projector --sigreg-projector-dim 128) ;;
  pre-rms-spatial) common_train+=(--sigreg-pre-rms-spatial) ;;
esac

eval_update() {
  local update="$1" checkpoint_dir eval_dir eval_stage sha_path repeat_marker baseline_path baseline_sha_path baseline_sha repeat_sha
  printf -v checkpoint_dir '%s/checkpoints/step-%012d' "$arm_dir" "$update"
  printf -v eval_dir '%s/eval-update-%04d' "$arm_dir" "$update"
  eval_stage="eval-$update"
  sha_path="$eval_dir/sha256.txt"
  repeat_marker="$eval_dir/repeat-required"
  baseline_path="$eval_dir/eval_report.repeat-baseline.json"
  baseline_sha_path="$eval_dir/sha256.repeat-baseline.txt"
  if [[ -e "$repeat_marker" && "$repeat_eval_update" != "$update" ]]; then
    printf 'evaluation repeat is incomplete; rerun with P2_AB_REPEAT_EVAL_UPDATE=%s: %s\n' \
      "$update" "$eval_dir" >&2
    return 1
  fi
  if [[ "$repeat_eval_update" != "$update" \
    && -f "$eval_dir/eval_report.json" && -f "$sha_path" && -f "$phase_log" ]] \
    && verify_sha_manifest "$sha_path" >/dev/null \
    && jq -s -e --arg stage "$eval_stage" \
      '[.[] | select(.stage == $stage)] | length > 0 and .[-1].status == "0"' \
      "$phase_log" >/dev/null; then
    printf 'preserving verified evaluation: %s\n' "$eval_dir/eval_report.json"
    return 0
  fi
  baseline_sha=""
  if [[ "$repeat_eval_update" == "$update" ]]; then
    if [[ -e "$repeat_marker" && -s "$baseline_path" && -s "$baseline_sha_path" ]]; then
      cp -- "$baseline_path" "$eval_dir/eval_report.json"
      cp -- "$baseline_sha_path" "$sha_path"
    fi
    if [[ ! -s "$eval_dir/eval_report.json" || ! -s "$sha_path" ]] \
      || ! verify_sha_manifest "$sha_path" >/dev/null; then
      printf 'cannot repeat an unverified evaluation: %s\n' "$eval_dir" >&2
      return 1
    fi
    baseline_sha="$(sha256sum "$eval_dir/eval_report.json" | awk '{print $1}')"
    cp -- "$eval_dir/eval_report.json" "$baseline_path"
    cp -- "$sha_path" "$baseline_sha_path"
    : >"$repeat_marker"
  fi
  if [[ -e "$eval_dir/eval_report.json" || -e "$sha_path" ]]; then
    printf 'retrying incomplete or invalid evaluation: %s\n' "$eval_dir"
  fi
  mkdir -p -- "$eval_dir"
  local eval_cmd=(
    "$tofy_bin" p2-eval
    --checkpoint "$checkpoint_dir/model.safetensors"
    --train-config "$arm_dir/config.json"
    --device cuda
    --seed "$eval_seed"
    --synthetic-episodes 64
    --physical-batch 1024
    --ptrm-k 1
    --ptrm-noise 0
    --ensemble-members 1
    --episode-jsonl "$eval_dir/episodes.jsonl"
    --output "$eval_dir/eval_report.json"
  )
  run_phase "$eval_stage" "$eval_dir/eval.log" "${eval_cmd[@]}"
  (
    cd "$ab_root"
    sha256sum \
      "$(artifact_ref "$checkpoint_dir/model.safetensors")" \
      "$(artifact_ref "$checkpoint_dir/optimizer.safetensors")" \
      "$(artifact_ref "$checkpoint_dir/trainer_state.json")" \
      "$(artifact_ref "$arm_dir/config.json")" \
      "$(artifact_ref "$eval_dir/eval_report.json")"
  ) >"$sha_path"
  if [[ -n "$baseline_sha" ]]; then
    repeat_sha="$(sha256sum "$eval_dir/eval_report.json" | awk '{print $1}')"
    jq -nc \
      --arg schema p2.eval_repeat_verification.v1 \
      --argjson update "$update" \
      --arg baseline_sha256 "$baseline_sha" \
      --arg repeat_sha256 "$repeat_sha" \
      --argjson identical "$(if [[ "$baseline_sha" == "$repeat_sha" ]]; then printf true; else printf false; fi)" \
      '{schema:$schema,update:$update,baseline_sha256:$baseline_sha256,repeat_sha256:$repeat_sha256,identical:$identical}' \
      >"$eval_dir/repeat-verification.json"
    if [[ "$baseline_sha" != "$repeat_sha" ]]; then
      printf 'repeated evaluation checksum mismatch: %s != %s\n' \
        "$baseline_sha" "$repeat_sha" >&2
      return 1
    fi
    rm -f -- "$repeat_marker"
  fi
}

finish_recovery_only() {
  local update="$1"
  if [[ "$recovery_only_update" == "$update" ]]; then
    stop_sampler
    trap - EXIT INT TERM
    printf 'completed isolated evaluation recovery for %s seed %s update %s\n' \
      "$arm" "$seed" "$update"
    exit 0
  fi
}

checkpoint_1000="$arm_dir/checkpoints/step-000000001000/model.safetensors"
if [[ ! -f "$checkpoint_1000" ]]; then
  run_phase train-1000 "$arm_dir/train-1000.log" \
    "${common_train[@]}" --max-steps-this-run 1000
fi
eval_update 1000
finish_recovery_only 1000

checkpoint_2000="$arm_dir/checkpoints/step-000000002000/model.safetensors"
if [[ ! -f "$checkpoint_2000" ]]; then
  run_phase train-2000 "$arm_dir/train-2000.log" \
    "${common_train[@]}" --resume "$arm_dir/checkpoints" --max-steps-this-run 1000
fi
eval_update 2000
finish_recovery_only 2000

if [[ "$target_update" == 4000 ]]; then
  checkpoint_4000="$arm_dir/checkpoints/step-000000004000/model.safetensors"
  if [[ ! -f "$checkpoint_4000" ]]; then
    run_phase train-4000 "$arm_dir/train-4000.log" \
      "${common_train[@]}" --resume "$arm_dir/checkpoints"
  fi
  eval_update 4000
  finish_recovery_only 4000
fi

jq -nc \
  --arg schema "$(if [[ "$experiment" == geometry ]]; then printf p2.sigreg_geometry_arm.v1; else printf p2.sigreg_action_arm.v1; fi)" \
  --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_git_sha" \
  --arg binary_sha256 "$binary_sha256" \
  --arg arm "$arm" \
  --argjson seed "$seed" \
  --argjson eval_seed "$eval_seed" \
  --arg output_dir "$(artifact_ref "$arm_dir")" \
  --arg config_sha256 "$(sha256sum "$arm_dir/config.json" | awk '{print $1}')" \
  --argjson complete_through_update "$target_update" \
  '{schema:$schema,git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,arm:$arm,seed:$seed,eval_seed:$eval_seed,physical_batch:1024,grad_accum:1,sigreg_geometry:(if $arm=="projector" then "pre_rms_global_pool_linear_projector" elif $arm=="pre-rms-spatial" then "pre_rms_unpooled_spatial_cells" else "post_rms_pooled_spatial_cells" end),projector_dim:(if $arm=="projector" then 128 else null end),output_dir:$output_dir,config_sha256:$config_sha256,complete_through_update:$complete_through_update}' \
  >"$arm_dir/manifest.json"

stop_sampler
trap - EXIT INT TERM
printf 'completed %s seed %s at %s\n' "$arm" "$seed" "$arm_dir"
