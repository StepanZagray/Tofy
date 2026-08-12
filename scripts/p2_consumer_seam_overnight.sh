#!/usr/bin/env bash
# Seed-1 controlled test of temporal-QQ pressure at the spatial/cell versus
# global-pooled representation seam. GPU-heavy work is strictly sequential;
# promotion remains locked until the completed campaign is analyzed.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_CONSUMER_RUN_ROOT:-$repo_root/runs/p2/consumer-seam-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
train_steps="${P2_CONSUMER_STEPS:-1000}"
eval_batch="${P2_CONSUMER_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath tee; do
  command -v "$command" >/dev/null || { printf 'missing required command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$train_steps" == 1000 && "$eval_batch" =~ ^[1-9][0-9]*$ \
  && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || {
  printf 'campaign requires 1000 updates and positive eval/telemetry sizes\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" \
  && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || {
  printf 'reviewed SHA mismatch\n' >&2; exit 2;
}
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
((initial_memory <= 1024 && initial_utilization == 0)) || {
  printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' \
    "$initial_memory" "$initial_utilization" >&2; exit 2;
}
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root already exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
started_epoch="$(date +%s)"
campaign_finalized=false
mark_unhandled_failure() {
  local rc="$?"
  if [[ "$campaign_finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg status interrupted_or_unhandled_failure \
      --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson exit_code "$rc" \
      '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null \
      && mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap mark_unhandled_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
jq -nc --arg schema p2.consumer_seam_campaign.v1 --arg status running \
  --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" \
  --arg binary_sha256 "$binary_sha" --arg gpu_name "$gpu_name" \
  --argjson train_steps "$train_steps" \
  '{schema:$schema,status:$status,started_utc:$started_utc,seed:1,
    git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,
    gpu_name:$gpu_name,train_steps:$train_steps,evaluation_seed:424242,
    synthetic_episodes:64,
    sequence:["batch_probe","no_sigreg","temporal_qq_cell","temporal_qq_mix","temporal_qq_global"],
    invariant:{lessons:["sequential"],temporal_window:8,statistic:"quantile",
      spatial_population:"post_rms_2x2_pooled_cells",hidden_dim:128,
      outer_steps:8,inner_steps:2,effective_batch:1024},
    treatment:{variable:"sigreg_global_mix",levels:[0,0.5,1],sigreg_weight:0.003},
    gates:{pooled_rank_fraction_min:0.10,pooled_variance_min:0.0001,
      changed_improvement_ci95_low_strictly_above:0,
      random_action_ratio_ci95_low_strictly_above:1,
      h8_finite_required:true,h8_normalized_max:1},
    run_order_confounded:true,
    promotion:"locked_pending_completed_campaign_analysis"}' >"$run_root/campaign.json"

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,finished_utc:$finished_utc}' \
    >>"$run_root/stages.jsonl"
}

finish_campaign() {
  local status="$1"
  jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
    '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  campaign_finalized=true
}

common_model_args=(
  --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0 --reliability-weight 0 --ensemble-members 1
  --rollout-weight 0 --prefix-weight 0
)

# The preceding full campaign established 1024x1 as stable on this A40. Recheck
# the worst-case mixed objective, but fail closed rather than changing the
# nonlinear objective population through accumulation.
physical_batch=""
grad_accum=""
for pair in "1024 1"; do
  read -r candidate accum <<<"$pair"
  probe_dir="$run_root/batch-probe-$candidate-$accum"
  if "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons sequential \
      --steps-per-lesson "$train_steps" --max-steps-this-run 2 \
      --physical-batch "$candidate" --grad-accum "$accum" \
      --checkpoint-every-steps 0 --profile-update 2 --sigreg-weight 0.003 \
      --sigreg-target temporal-residual --sigreg-statistic quantile \
      --sigreg-temporal-window 8 --sigreg-global-mix 0.5 \
      --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
      "${common_model_args[@]}" --output-dir "$probe_dir" \
      >"$probe_dir.log" 2>&1; then
    physical_batch="$candidate"
    grad_accum="$accum"
    record_stage batch_probe passed "physical_batch=$candidate grad_accum=$accum effective_batch=1024 statistic=quantile mix=0.5"
    break
  fi
  record_stage "batch_probe_$candidate" failed "see $probe_dir.log"
done
if [[ -z "$physical_batch" ]]; then
  record_stage batch_probe failed all_candidates
  finish_campaign failed_batch_probe
  exit 1
fi
probe_dir="$(realpath "$run_root/batch-probe-$physical_batch-$grad_accum")"
jq -e --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  '.global_step == 2 and .status == "paused" and .physical_batch == $physical_batch
    and .grad_accum == $grad_accum' "$probe_dir/train_report.json" >/dev/null
jq -e '.sigreg_weight == 0.003 and .sigreg_target == "temporal_residual"
    and .sigreg_statistic == "quantile" and .sigreg_temporal_window == 8
    and .sigreg_global_mix == 0.5 and .steady_gpu == true' \
  "$probe_dir/config.json" >/dev/null
sha256sum "$probe_dir/config.json" "$probe_dir/train_report.json" \
  "$probe_dir/model.safetensors" >"$probe_dir/artifacts.sha256"
jq -nc --arg schema p2.consumer_seam_batch_probe.v1 --arg status passed \
  --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" \
  --arg binary_sha256 "$binary_sha" --arg gpu_name "$gpu_name" \
  --arg artifacts_sha256 "$(sha256sum "$probe_dir/artifacts.sha256" | awk '{print $1}')" \
  --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  '{schema:$schema,status:$status,git_sha:$git_sha,candle_git_sha:$candle_git_sha,
    binary_sha256:$binary_sha256,gpu_name:$gpu_name,physical_batch:$physical_batch,
    grad_accum:$grad_accum,effective_batch:1024,sigreg_weight:0.003,
    sigreg_target:"temporal_residual",sigreg_statistic:"quantile",
    sigreg_temporal_window:8,sigreg_global_mix:0.5,completed_updates:2,
    artifacts_sha256:$artifacts_sha256}' >"$probe_dir/probe.json"
jq --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  --arg batch_probe "$probe_dir/probe.json" \
  '.physical_batch=$physical_batch | .grad_accum=$grad_accum | .effective_batch=1024
    | .batch_probe=$batch_probe' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval"
  done
}

run_eval() {
  local checkpoint="$1" config="$2" eval_dir="$3"
  mkdir -p -- "$eval_dir"
  "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" \
    --device cuda:0 --seed 424242 --synthetic-episodes 64 \
    --physical-batch "$eval_batch" --ptrm-k 1 --ptrm-noise 0 \
    --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" >"$eval_dir/eval.log" 2>&1
  sha256sum "$checkpoint" "$config" "$eval_dir/eval_report.json" \
    "$eval_dir/episodes.jsonl" >"$eval_dir/sha256.txt"
}

write_decision() {
  local report="$1" destination="$2"
  jq '{
    schema:"p2.consumer_seam_decision.v1",
    pooled_representation:.synthetic_dynamics.representation,
    seams:.synthetic_dynamics.representation_seams,
    changed_transitions:.synthetic_dynamics.changed_transitions,
    random_one_step_action:.synthetic_dynamics.action_diagnostics.by_source.random_one_step.shuffle,
    h8:.synthetic_dynamics.rollout.h8,
    gates:{
      pooled_rank:((.synthetic_dynamics.representation.effective_rank_fraction // 0) >= 0.10),
      pooled_variance:((.synthetic_dynamics.representation.mean_encoder_variance // 0) >= 0.0001),
      changed_copy:((.synthetic_dynamics.changed_transitions.improvement_ci95_low // -1) > 0),
      random_action:((.synthetic_dynamics.action_diagnostics.by_source.random_one_step.shuffle.ratio_ci95_low // 0) > 1),
      h8_finite:((.synthetic_dynamics.rollout.h8.finite_n // 0) == (.synthetic_dynamics.rollout.h8.n // -1)),
      h8_no_regression:((.synthetic_dynamics.rollout.h8.normalized_mean // 1e300) <= 1)
    }
  } | .absolute_gate_pass=(.gates | [.pooled_rank,.pooled_variance,.changed_copy,.random_action,.h8_finite,.h8_no_regression] | all)' \
    "$report" >"$destination"
}

run_arm() {
  local arm="$1" sigreg_weight="$2" global_mix="$3"
  local arm_dir="$run_root/seed-1/$arm" sampler_pid="" update checkpoint eval_dir
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc --arg schema p2.consumer_seam_arm.v1 --arg arm "$arm" \
    --arg git_sha "$git_sha" --arg candle_git_sha "$candle_sha" \
    --arg binary_sha256 "$binary_sha" --argjson sigreg_weight "$sigreg_weight" \
    --argjson global_mix "$global_mix" --argjson physical_batch "$physical_batch" \
    --argjson grad_accum "$grad_accum" \
    '{schema:$schema,arm:$arm,seed:1,sigreg_weight:$sigreg_weight,
      sigreg_target:"temporal_residual",sigreg_statistic:"quantile",
      sigreg_global_mix:$global_mix,sigreg_temporal_window:8,
      physical_batch:$physical_batch,grad_accum:$grad_accum,effective_batch:1024,
      git_sha:$git_sha,candle_git_sha:$candle_git_sha,binary_sha256:$binary_sha256,
      checkpoints:[250,500,750,1000],evaluation_updates:[250,500,750,1000],
      promotion:"locked"}' >"$arm_dir/run.json"

  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
  trap 'exit 130' INT TERM
  "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons sequential \
    --steps-per-lesson "$train_steps" --physical-batch "$physical_batch" \
    --grad-accum "$grad_accum" --checkpoint-every-steps 250 --profile-update 250 \
    --sigreg-weight "$sigreg_weight" --sigreg-target temporal-residual \
    --sigreg-statistic quantile --sigreg-temporal-window 8 \
    --sigreg-global-mix "$global_mix" --sigreg-spatial --sigreg-spatial-pool \
    --sigreg-max-rows 32768 "${common_model_args[@]}" \
    --output-dir "$arm_dir" > >(tee "$arm_dir/train.log") 2>&1
  jq -e --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
    '.global_step == 1000 and .status == "completed"
      and .physical_batch == $physical_batch and .grad_accum == $grad_accum
      and .profile.published.update == 250' "$arm_dir/train_report.json" >/dev/null
  jq -e --argjson sigreg_weight "$sigreg_weight" --argjson global_mix "$global_mix" \
    '.sigreg_weight == $sigreg_weight and .sigreg_target == "temporal_residual"
      and .sigreg_statistic == "quantile" and .sigreg_temporal_window == 8
      and .sigreg_global_mix == $global_mix and .steady_gpu == true
      and .physical_batch == 1024 and .grad_accum == 1' \
    "$arm_dir/config.json" >/dev/null

  for update in 250 500 750 1000; do
    if ((update > train_steps)); then
      continue
    fi
    printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$update"
    eval_dir="$arm_dir/eval-update-$update"
    [[ -s "$checkpoint" ]] || { printf 'missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
    run_eval "$checkpoint" "$arm_dir/config.json" "$eval_dir"
  done
  eval_dir="$arm_dir/eval-update-$train_steps"
  [[ -s "$eval_dir/eval_report.json" ]] || {
    run_eval "$arm_dir/model.safetensors" "$arm_dir/config.json" "$eval_dir"
  }
  write_decision "$eval_dir/eval_report.json" "$arm_dir/decision.json"
  kill -0 "$sampler_pid"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  (( $(awk 'END {print NR}' "$arm_dir/telemetry/gpu.csv") >= 10 ))
  sha256sum "$arm_dir/config.json" "$arm_dir/train_report.json" \
    "$arm_dir/decision.json" >"$arm_dir/artifacts.sha256"
  for update in 250 500 750 1000; do
    sha256sum "$arm_dir/eval-update-$update/eval_report.json" \
      "$arm_dir/eval-update-$update/episodes.jsonl" >>"$arm_dir/artifacts.sha256"
  done
  trap - EXIT INT TERM
}

arms=(
  'no_sigreg 0 0'
  'temporal_qq_cell 0.003 0'
  'temporal_qq_mix 0.003 0.5'
  'temporal_qq_global 0.003 1'
)
failed=0
for definition in "${arms[@]}"; do
  read -r arm sigreg_weight global_mix <<<"$definition"
  arm_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if (set -euo pipefail; run_arm "$arm" "$sigreg_weight" "$global_mix"); then
    arm_status=passed
  else
    arm_status=failed
    failed=$((failed + 1))
  fi
  jq -nc --arg arm "$arm" --arg status "$arm_status" \
    --arg started_utc "$arm_started" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{arm:$arm,status:$status,started_utc:$started_utc,finished_utc:$finished_utc}' \
    >>"$run_root/arms.jsonl"
  record_stage "$arm" "$arm_status"
done

if ((failed == 0)); then
  finish_campaign complete_pending_analysis
else
  jq --argjson failed_arms "$failed" '.failed_arms=$failed_arms' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  finish_campaign partial_failure
fi
printf 'consumer-seam campaign finished with %s failed arms; promotion remains locked\n' "$failed"
exit "$([[ "$failed" -eq 0 ]] && printf 0 || printf 1)"
