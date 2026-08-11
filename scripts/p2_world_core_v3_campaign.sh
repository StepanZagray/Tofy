#!/usr/bin/env bash
# Deadline-bounded causal V3 campaign: four seed-1 arms plus matched controls at seeds 2 and 3.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_V3_RUN_ROOT:-$repo_root/runs/p2/world-core-v3-v1}"
physical_batch="${P2_V3_PHYSICAL_BATCH:-1024}"
eval_batch="${P2_V3_EVAL_BATCH:-1024}"
steps_per_lesson="${P2_V3_STEPS_PER_LESSON:-100}"
max_seconds="${P2_V3_MAX_SECONDS:-36000}"
minimum_arm_seconds="${P2_V3_MINIMUM_ARM_SECONDS:-5400}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set reviewed tofy binary hash}"
: "${P2_V3_BATCH_PROBE:?set the passed V3 probe report}"
for command in git jq nvidia-smi sha256sum awk realpath timeout; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
for value in "$physical_batch" "$eval_batch" "$steps_per_lesson" "$max_seconds" "$minimum_arm_seconds" "$gpu_interval"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid numeric campaign setting\n' >&2; exit 2; }
done
((physical_batch % 4 == 0)) || { printf 'physical batch must preserve four-branch groups\n' >&2; exit 2; }
[[ -x "$tofy_bin" && -d "$candle_root/.git" ]] || exit 2

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" && "$gpu_name" == "NVIDIA A40" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" && -z "$(git -C "$candle_root" status --porcelain)" ]] || exit 2

probe_dir="$(cd -- "$(dirname -- "$P2_V3_BATCH_PROBE")" && pwd)"
probe_config="$probe_dir/config.json"
probe_report="$probe_dir/train_report.json"
jq -e --arg git_sha "$git_sha" --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" \
  --arg gpu_name "$gpu_name" --argjson physical "$physical_batch" \
  '.schema == "p2.world_core_v3_batch_probe.v1" and .status == "passed"
   and .world_core_schema == "world_core_v3" and .physical_batch == $physical
   and .grad_accum == 1 and .global_step == 2 and .git_sha == $git_sha
   and .candle_git_sha == $candle_sha and .binary_sha256 == $binary_sha and .gpu_name == $gpu_name' \
  "$P2_V3_BATCH_PROBE" >/dev/null
jq -e --argjson physical "$physical_batch" '.world_core_v3 == true and .spatial_action_field == true
  and .spatial_action_residual == true and .spatial_action_residual_scale == 0.25
  and .branch_learning.displacement_health.variance_weight == 0.3
  and .branch_learning.displacement_health.covariance_weight == 0.03
  and .branch_learning.displacement_norm_floor == 0.05
  and .physical_batch == $physical and .grad_accum == 1' "$probe_config" >/dev/null
jq -e --argjson ratio "$(jq -er '.gradient_pressure.displacement_health_to_next_ratio' "$probe_report")" \
  '.displacement_health_to_next_ratio == $ratio
   and .displacement_health_to_next_ratio >= 0.01
   and .displacement_health_to_next_ratio <= 0.5' "$P2_V3_BATCH_PROBE" >/dev/null
[[ "$(sha256sum "$probe_config" | awk '{print $1}')" == "$(jq -r .config_sha256 "$P2_V3_BATCH_PROBE")" \
  && "$(sha256sum "$probe_report" | awk '{print $1}')" == "$(jq -r .report_sha256 "$P2_V3_BATCH_PROBE")" ]] || exit 2

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root already exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
started_epoch="$(date +%s)"
deadline_epoch=$((started_epoch + max_seconds))
final_update=$((steps_per_lesson * 5))
jq -nc --arg schema p2.world_core_v3_campaign.v1 --arg status running \
  --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" \
  --argjson deadline_epoch "$deadline_epoch" --argjson max_seconds "$max_seconds" \
  --argjson physical_batch "$physical_batch" --argjson final_update "$final_update" \
  '{schema:$schema,status:$status,started_utc:$started_utc,deadline_epoch:$deadline_epoch,
    max_seconds:$max_seconds,git_sha:$git_sha,candle_git_sha:$candle_git_sha,
    binary_sha256:$binary_sha256,physical_batch:$physical_batch,grad_accum:1,
    final_update:$final_update,factual_eval_groups:256,
    design:"seed1 four-arm causal screen; seeds2-3 predeclared matched global/residual replication",
    promotion:"locked_pending_analysis"}' >"$run_root/campaign.json"

remaining_seconds() { printf '%s\n' "$((deadline_epoch - $(date +%s)))"; }
run_before_deadline() {
  local remaining budget
  remaining="$(remaining_seconds)"
  budget=$((remaining - 300))
  ((budget > 0)) || return 124
  timeout --signal=TERM --kill-after=240 "${budget}s" "$@"
}
sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}

run_arm() {
  local seed="$1" arm="$2" residual="$3" variance="$4" covariance="$5"
  local arm_dir="$run_root/seed-$seed/$arm" sampler_pid checkpoint eval_dir
  local -a residual_args=() health_args=()
  [[ "$residual" == true ]] && residual_args+=(--spatial-action-field --spatial-action-residual --spatial-action-residual-scale 0.25)
  if [[ "$variance" != 0 || "$covariance" != 0 ]]; then
    health_args+=(--displacement-variance-weight "$variance" --displacement-covariance-weight "$covariance" \
      --displacement-norm-floor 0.05 --health-minimum-std 0.1 --health-maximum-rows 16384)
  fi
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc --arg schema p2.world_core_v3_arm.v1 --arg arm "$arm" --argjson seed "$seed" \
    --argjson residual "$residual" --argjson variance "$variance" --argjson covariance "$covariance" \
    --argjson final_update "$final_update" \
    '{schema:$schema,arm:$arm,seed:$seed,world_core_schema:"world_core_v3",
      spatial_action_residual:$residual,displacement_variance_weight:$variance,
      displacement_covariance_weight:$covariance,final_update:$final_update}' >"$arm_dir/run.json"
  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT INT TERM

  run_before_deadline "$tofy_bin" p2-train --device cuda:0 --seed "$seed" --world-core-v3 \
    "${residual_args[@]}" "${health_args[@]}" --shuffled-episodes \
    --lessons factual_branches,dynamics,sequential --steps-per-lesson "$steps_per_lesson" \
    --physical-batch "$physical_batch" --grad-accum 1 --checkpoint-every-steps 250 \
    --profile-update 2 --sigreg-weight 0 --outcome-pull-weight 0.05 \
    --outcome-push-weight 0.05 --outcome-margin 0.5 --action-recovery-weight 0.05 \
    --coordinate-recovery-weight 0.05 --changed-margin-weight 0.05 --changed-margin 0.1 \
    --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0 --rollout-weight 0.1 --prefix-weight 0.05 \
    --reliability-weight 0 --ensemble-members 1 --output-dir "$arm_dir" \
    >"$arm_dir/train.log" 2>&1
  jq -e --argjson final "$final_update" \
    '.world_core_schema == "world_core_v3" and .global_step == $final and .status == "completed"' \
    "$arm_dir/train_report.json" >/dev/null
  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$final_update"
  eval_dir="$arm_dir/eval-final"
  mkdir -p -- "$eval_dir"
  run_before_deadline "$tofy_bin" p2-eval --checkpoint "$checkpoint" \
    --train-config "$arm_dir/config.json" --device cuda:0 --seed 424242 \
    --synthetic-episodes 64 --physical-batch "$eval_batch" --ptrm-k 1 --ptrm-noise 0 \
    --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" >"$eval_dir/eval.log" 2>&1
  jq -e '.schema == "p2.eval_report.v12" and .factual_branches.groups == 256
    and .factual_branches.board_probe != null' "$eval_dir/eval_report.json" >/dev/null
  sha256sum "$checkpoint" "$arm_dir/config.json" "$arm_dir/train_report.json" \
    "$eval_dir/eval_report.json" "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  trap - EXIT INT TERM
}

arms=(
  '1 global-control false 0 0'
  '1 spatial-residual true 0 0'
  '1 displacement-variance true 0.3 0'
  '1 displacement-decorrelated true 0.3 0.03'
  '2 global-control false 0 0'
  '2 spatial-residual true 0 0'
  '3 global-control false 0 0'
  '3 spatial-residual true 0 0'
)
failed=0
skipped=0
for definition in "${arms[@]}"; do
  read -r seed arm residual variance covariance <<<"$definition"
  if (( $(remaining_seconds) < minimum_arm_seconds )); then
    printf '{"seed":%s,"arm":"%s","status":"skipped_deadline_admission"}\n' "$seed" "$arm" >>"$run_root/arms.jsonl"
    skipped=$((skipped + 1))
    continue
  fi
  if (set -euo pipefail; run_arm "$seed" "$arm" "$residual" "$variance" "$covariance"); then
    status=passed
  else
    status=failed
    failed=$((failed + 1))
  fi
  jq -nc --argjson seed "$seed" --arg arm "$arm" --arg status "$status" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{seed:$seed,arm:$arm,status:$status,finished_utc:$finished_utc}' >>"$run_root/arms.jsonl"
done

status=complete
((failed == 0)) || status=partial_failure
((skipped == 0)) || status=deadline_limited
((failed == 0 || skipped == 0)) || status=partial_failure_deadline_limited
jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson failed_arms "$failed" --argjson skipped_arms "$skipped" \
  '.status=$status | .finished_utc=$finished_utc | .failed_arms=$failed_arms | .skipped_arms=$skipped_arms' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
((failed == 0 && skipped == 0))
