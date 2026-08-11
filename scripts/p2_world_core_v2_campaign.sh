#!/usr/bin/env bash
# Sequential causal campaign for the action-faithful world-core-v2.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_V2_RUN_ROOT:-$repo_root/runs/p2/world-core-v2-v1}"
physical_batch="${P2_V2_PHYSICAL_BATCH:-1024}"
grad_accum="${P2_V2_GRAD_ACCUM:-1}"
steps_per_lesson="${P2_V2_STEPS_PER_LESSON:-100}"
eval_batch="${P2_V2_EVAL_BATCH:-1024}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set reviewed tofy binary hash}"
: "${P2_V2_BATCH_PROBE:?set the passed worst-case world-core-v2 probe report}"

for command in git jq nvidia-smi sha256sum awk realpath tee; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" && -d "$candle_root/.git" ]] || exit 2
[[ "$physical_batch" =~ ^[1-9][0-9]*$ && "$grad_accum" == 1 ]] || {
  printf 'world-core-v2 requires a positive physical batch and grad_accum=1\n' >&2; exit 2;
}
[[ "$steps_per_lesson" =~ ^[1-9][0-9]*$ && "$eval_batch" =~ ^[1-9][0-9]*$ \
  && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid numeric campaign setting\n' >&2; exit 2; }
((physical_batch % 4 == 0)) || { printf 'physical batch must preserve four-branch groups\n' >&2; exit 2; }

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'reviewed SHA mismatch\n' >&2; exit 2; }
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, found %s\n' "$gpu_name" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" && -z "$(git -C "$candle_root" status --porcelain)" ]] || {
  printf 'campaign requires clean reviewed worktrees\n' >&2; exit 2;
}
probe_dir="$(cd -- "$(dirname -- "$P2_V2_BATCH_PROBE")" && pwd)"
probe_config="$probe_dir/config.json"
probe_report="$probe_dir/train_report.json"
[[ -s "$probe_config" && -s "$probe_report" ]] || { printf 'batch probe artifacts missing\n' >&2; exit 2; }
jq -e --arg git_sha "$git_sha" --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" \
  --arg gpu_name "$gpu_name" --argjson physical "$physical_batch" \
  '.schema == "p2.world_core_v2_batch_probe.v1" and .status == "passed"
   and .world_core_schema == "world_core_v2" and .physical_batch == $physical
   and .grad_accum == 1 and .global_step >= 2 and .git_sha == $git_sha
   and .candle_git_sha == $candle_sha and .binary_sha256 == $binary_sha
   and .gpu_name == $gpu_name' "$P2_V2_BATCH_PROBE" >/dev/null || {
  printf 'batch probe does not authorize this physical population\n' >&2; exit 2;
}
jq -e --argjson physical "$physical_batch" \
  '.world_core_v2 == true and .spatial_action_field == true and .physical_batch == $physical
   and .grad_accum == 1 and .sigreg_weight == 0
   and .branch_learning.spatial_health != null and .branch_learning.pooled_health != null' \
  "$probe_config" >/dev/null || { printf 'batch probe was not worst-case dual health\n' >&2; exit 2; }
[[ "$(sha256sum "$probe_config" | awk '{print $1}')" == "$(jq -r .config_sha256 "$P2_V2_BATCH_PROBE")" \
  && "$(sha256sum "$probe_report" | awk '{print $1}')" == "$(jq -r .report_sha256 "$P2_V2_BATCH_PROBE")" ]] || {
  printf 'batch probe hashes do not match retained artifacts\n' >&2; exit 2;
}

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root already exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
final_update=$((steps_per_lesson * 5))
jq -nc --arg schema p2.world_core_v2_campaign.v1 --arg status running \
  --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg candle_git_sha "$candle_sha" --arg binary_sha256 "$binary_sha" \
  --arg batch_probe_sha256 "$(sha256sum "$P2_V2_BATCH_PROBE" | awk '{print $1}')" \
  --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  --argjson steps_per_lesson "$steps_per_lesson" --argjson final_update "$final_update" \
  '{schema:$schema,status:$status,started_utc:$started_utc,seed:1,
    arms:["branch-global","branch-spatial","spatial-health","dual-health"],
    git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,
    batch_probe_sha256:$batch_probe_sha256,physical_batch:$physical_batch,
    grad_accum:$grad_accum,steps_per_lesson:$steps_per_lesson,final_update:$final_update,
    promotion:"locked_pending_analysis"}' >"$run_root/campaign.json"

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}

run_arm() {
  local arm="$1" spatial="$2" spatial_var="$3" spatial_cov="$4" pooled_var="$5" pooled_cov="$6"
  local arm_dir="$run_root/seed-1/$arm" sampler_pid started finished checkpoint eval_dir
  local -a spatial_args=()
  [[ "$spatial" == true ]] && spatial_args+=(--spatial-action-field)
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc --arg schema p2.world_core_v2_arm.v1 --arg arm "$arm" --argjson spatial "$spatial" \
    --argjson spatial_var "$spatial_var" --argjson spatial_cov "$spatial_cov" \
    --argjson pooled_var "$pooled_var" --argjson pooled_cov "$pooled_cov" \
    --argjson physical_batch "$physical_batch" --argjson final_update "$final_update" \
    '{schema:$schema,arm:$arm,world_core_schema:"world_core_v2",spatial_action_field:$spatial,
      physical_batch:$physical_batch,grad_accum:1,final_update:$final_update,
      spatial_variance_weight:$spatial_var,spatial_covariance_weight:$spatial_cov,
      pooled_variance_weight:$pooled_var,pooled_covariance_weight:$pooled_cov}' >"$arm_dir/run.json"

  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT INT TERM
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$tofy_bin" p2-train --device cuda:0 --seed 1 \
    --world-core-v2 "${spatial_args[@]}" \
    --lessons factual_branches,dynamics,sequential --steps-per-lesson "$steps_per_lesson" \
    --physical-batch "$physical_batch" --grad-accum 1 --checkpoint-every-steps "$final_update" \
    --profile-update 2 --sigreg-weight 0 --outcome-pull-weight 0.05 \
    --outcome-push-weight 0.05 --outcome-margin 0.5 --action-recovery-weight 0.05 \
    --coordinate-recovery-weight 0.05 --changed-margin-weight 0.05 --changed-margin 0.1 \
    --spatial-variance-weight "$spatial_var" --spatial-covariance-weight "$spatial_cov" \
    --pooled-variance-weight "$pooled_var" --pooled-covariance-weight "$pooled_cov" \
    --health-minimum-std 1 --health-maximum-rows 16384 \
    --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y \
    --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8 \
    --event-weight 0 --q-weight 0 --rollout-weight 0.1 --prefix-weight 0.05 \
    --reliability-weight 0 --ensemble-members 1 --output-dir "$arm_dir" \
    > >(tee "$arm_dir/train.log") 2>&1
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -nc --arg stage train --arg started_utc "$started" --arg finished_utc "$finished" \
    '{stage:$stage,started_utc:$started_utc,finished_utc:$finished_utc,status:"passed"}' \
    >>"$arm_dir/phases.jsonl"

  printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$final_update"
  eval_dir="$arm_dir/eval-final"
  mkdir -p -- "$eval_dir"
  "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$arm_dir/config.json" \
    --device cuda:0 --seed 424242 --synthetic-episodes 64 --physical-batch "$eval_batch" \
    --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
    --episode-jsonl "$eval_dir/episodes.jsonl" --output "$eval_dir/eval_report.json" \
    > >(tee "$eval_dir/eval.log") 2>&1
  sha256sum "$checkpoint" "$arm_dir/config.json" "$arm_dir/train_report.json" \
    "$eval_dir/eval_report.json" "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  trap - EXIT INT TERM
}

arms=(
  'branch-global false 0 0 0 0'
  'branch-spatial true 0 0 0 0'
  'spatial-health true 0.05 0.005 0 0'
  'dual-health true 0.05 0.005 0.05 0.005'
)
failed=0
for definition in "${arms[@]}"; do
  read -r arm spatial spatial_var spatial_cov pooled_var pooled_cov <<<"$definition"
  if (set -euo pipefail; run_arm "$arm" "$spatial" "$spatial_var" "$spatial_cov" "$pooled_var" "$pooled_cov"); then
    status=passed
  else
    status=failed
    failed=$((failed + 1))
  fi
  jq -nc --arg arm "$arm" --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{arm:$arm,status:$status,finished_utc:$finished_utc}' >>"$run_root/arms.jsonl"
done

jq --arg status "$([[ "$failed" -eq 0 ]] && printf complete || printf partial_failure)" \
  --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson failed_arms "$failed" \
  '.status=$status | .finished_utc=$finished_utc | .failed_arms=$failed_arms' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
exit "$([[ "$failed" -eq 0 ]] && printf 0 || printf 1)"
