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
[[ "$experiment" == action || "$experiment" == geometry ]] || {
  printf 'P2_AB_EXPERIMENT must be action or geometry\n' >&2
  exit 2
}
case "$experiment:$arm" in
  action:control|action:projector|geometry:control|geometry:pre-rms-spatial) ;;
  *) printf 'arm %s is invalid for %s experiment\n' "$arm" "$experiment" >&2; exit 2 ;;
esac
[[ -x "$tofy_bin" ]] || { printf 'missing reviewed binary: %s\n' "$tofy_bin" >&2; exit 2; }
for required in git jq nvidia-smi sha256sum awk tee python3; do
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

arm_dir="$ab_root/seed-$seed/$arm"
mkdir -p -- "$arm_dir" "$arm_dir/telemetry"
commands_log="$arm_dir/commands.log"
phase_log="$arm_dir/phases.jsonl"

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
    sleep 15
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
  local update="$1" checkpoint_dir eval_dir eval_stage sha_path
  printf -v checkpoint_dir '%s/checkpoints/step-%012d' "$arm_dir" "$update"
  printf -v eval_dir '%s/eval-update-%04d' "$arm_dir" "$update"
  eval_stage="eval-$update"
  sha_path="$eval_dir/sha256.txt"
  if [[ -f "$eval_dir/eval_report.json" && -f "$sha_path" && -f "$phase_log" ]] \
    && sha256sum --check --status "$sha_path" \
    && jq -s -e --arg stage "$eval_stage" \
      '[.[] | select(.stage == $stage)] | length > 0 and .[-1].status == "0"' \
      "$phase_log" >/dev/null; then
    printf 'preserving verified evaluation: %s\n' "$eval_dir/eval_report.json"
    return 0
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
  sha256sum \
    "$checkpoint_dir/model.safetensors" \
    "$checkpoint_dir/optimizer.safetensors" \
    "$checkpoint_dir/trainer_state.json" \
    "$arm_dir/config.json" \
    "$eval_dir/eval_report.json" \
    >"$sha_path"
}

checkpoint_1000="$arm_dir/checkpoints/step-000000001000/model.safetensors"
if [[ ! -f "$checkpoint_1000" ]]; then
  run_phase train-1000 "$arm_dir/train-1000.log" \
    "${common_train[@]}" --max-steps-this-run 1000
fi
eval_update 1000

checkpoint_2000="$arm_dir/checkpoints/step-000000002000/model.safetensors"
if [[ ! -f "$checkpoint_2000" ]]; then
  run_phase train-2000 "$arm_dir/train-2000.log" \
    "${common_train[@]}" --resume "$arm_dir/checkpoints" --max-steps-this-run 1000
fi
eval_update 2000

if [[ "$target_update" == 4000 ]]; then
  checkpoint_4000="$arm_dir/checkpoints/step-000000004000/model.safetensors"
  if [[ ! -f "$checkpoint_4000" ]]; then
    run_phase train-4000 "$arm_dir/train-4000.log" \
      "${common_train[@]}" --resume "$arm_dir/checkpoints"
  fi
  eval_update 4000
fi

jq -nc \
  --arg schema "$(if [[ "$experiment" == geometry ]]; then printf p2.sigreg_geometry_arm.v1; else printf p2.sigreg_action_arm.v1; fi)" \
  --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_git_sha" \
  --arg binary_sha256 "$binary_sha256" \
  --arg arm "$arm" \
  --argjson seed "$seed" \
  --argjson eval_seed "$eval_seed" \
  --arg output_dir "$arm_dir" \
  --arg config_sha256 "$(sha256sum "$arm_dir/config.json" | awk '{print $1}')" \
  --argjson complete_through_update "$target_update" \
  '{schema:$schema,git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,arm:$arm,seed:$seed,eval_seed:$eval_seed,physical_batch:1024,grad_accum:1,sigreg_geometry:(if $arm=="projector" then "pre_rms_global_pool_linear_projector" elif $arm=="pre-rms-spatial" then "pre_rms_unpooled_spatial_cells" else "post_rms_pooled_spatial_cells" end),projector_dim:(if $arm=="projector" then 128 else null end),output_dir:$output_dir,config_sha256:$config_sha256,complete_through_update:$complete_through_update}' \
  >"$arm_dir/manifest.json"

stop_sampler
trap - EXIT INT TERM
printf 'completed %s seed %s at %s\n' "$arm" "$seed" "$arm_dir"
