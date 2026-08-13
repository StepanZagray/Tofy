#!/usr/bin/env bash
# Preregistered seed-1 A/B at the active planning-head consumer seam.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_READOUT_RUN_ROOT:-$repo_root/runs/p2/consumer-readout-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
train_steps="${P2_READOUT_STEPS:-1000}"
eval_batch="${P2_READOUT_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath tee cmp; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$train_steps" == 1000 && "$eval_batch" =~ ^[1-9][0-9]*$ && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || {
  printf 'campaign requires 1000 updates and positive eval/telemetry sizes\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || {
  printf 'reviewed SHA mismatch\n' >&2; exit 2;
}
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
(( initial_memory <= 1024 && initial_utilization == 0 )) || {
  printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' "$initial_memory" "$initial_utilization" >&2; exit 2;
}
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
started_epoch="$(date +%s)"
campaign_finalized=false
mark_unhandled_failure() {
  local rc="$?"
  if [[ "$campaign_finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg status interrupted_or_unhandled_failure --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson exit_code "$rc" '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null \
      && mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap mark_unhandled_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg candle_sha "$candle_sha" --arg binary_sha "$binary_sha" --arg gpu "$gpu_name" \
  '{schema:"p2.consumer_readout_campaign.v1",status:"running",started_utc:$started_utc,
    git_sha:$git_sha,candle_git_sha:$candle_sha,binary_sha256:$binary_sha,gpu_name:$gpu,
    seed:1,evaluation_seed:424242,train_steps:1000,synthetic_episodes:64,
    sequence:["capacity_probe","integrity_smoke_spatial_query","integrity_smoke_global_mean",
      "consumer_readout_spatial_query","consumer_readout_global_mean"],
    treatment:{variable:"consumer_readout",levels:["spatial_query","global_mean"]},
    invariant:{lesson:"q_calibration",shuffled_episodes:true,temporal_qq:"cell",qq_weight:0.003,
      q_weight:0.1,ptrm_rank:false,temporal_window:8,statistic:"quantile",effective_batch:1024,
      hidden_dim:128,inner_steps:2,outer_steps:8},
    checkpoints:[250,500,750,1000],efficacy_early_stop:false,
    run_order_confounded:true,
    promotion:"locked_pending_completed_campaign_analysis"}' >"$run_root/campaign.json"

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,finished_utc:$finished_utc}' >>"$run_root/stages.jsonl"
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

common_args=(
  --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0.1 --reliability-weight 0 --rollout-weight 0 --prefix-weight 0
  --ptrm-rank-every 0 --ensemble-members 1 --shuffled-episodes
  --sigreg-weight 0.003 --sigreg-target temporal-residual --sigreg-statistic quantile
  --sigreg-temporal-window 8 --sigreg-global-mix 0 --sigreg-spatial --sigreg-spatial-pool
  --sigreg-max-rows 32768
)

physical_batch=""
grad_accum=""
for pair in "1024 1" "512 2" "256 4"; do
  read -r candidate accum <<<"$pair"
  probe_dir="$run_root/capacity-$candidate-$accum"
  if "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons q_calibration \
      --steps-per-lesson "$train_steps" --max-steps-this-run 3 \
      --physical-batch "$candidate" --grad-accum "$accum" --checkpoint-every-steps 0 \
      --profile-update 3 --consumer-readout spatial-query "${common_args[@]}" \
      --output-dir "$probe_dir" >"$probe_dir.log" 2>&1; then
    physical_batch="$candidate"
    grad_accum="$accum"
    record_stage capacity_probe passed "physical_batch=$candidate grad_accum=$accum effective_batch=1024"
    break
  fi
  record_stage "capacity_$candidate" failed "see $probe_dir.log"
done
if [[ -z "$physical_batch" ]]; then
  finish_campaign failed_capacity_probe
  exit 1
fi

run_smoke() {
  local topology="$1" smoke_dir="$run_root/integrity-$topology"
  "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons q_calibration \
    --steps-per-lesson "$train_steps" --max-steps-this-run 3 \
    --physical-batch "$physical_batch" --grad-accum "$grad_accum" --checkpoint-every-steps 0 \
    --profile-update 3 --consumer-readout "$topology" "${common_args[@]}" \
    --output-dir "$smoke_dir" >"$smoke_dir.log" 2>&1
  jq -e --arg topology "${topology//-/_}" --argjson batch "$physical_batch" --argjson accum "$grad_accum" \
    '.global_step == 3 and .status == "paused" and .physical_batch == $batch and .grad_accum == $accum
      and .experiment.consumer_readout == $topology and .training_population_rows > 0' \
    "$smoke_dir/train_report.json" >/dev/null
  sha256sum "$smoke_dir/config.json" "$smoke_dir/train_report.json" "$smoke_dir/model.safetensors" \
    >"$smoke_dir/artifacts.sha256"
}
run_smoke spatial-query
record_stage integrity_smoke_spatial_query passed
run_smoke global-mean
record_stage integrity_smoke_global_mean passed

spatial_fingerprint="$(jq -r .training_population_fingerprint "$run_root/integrity-spatial-query/train_report.json")"
global_fingerprint="$(jq -r .training_population_fingerprint "$run_root/integrity-global-mean/train_report.json")"
spatial_rows="$(jq -r .training_population_rows "$run_root/integrity-spatial-query/train_report.json")"
global_rows="$(jq -r .training_population_rows "$run_root/integrity-global-mean/train_report.json")"
[[ "$spatial_fingerprint" == "$global_fingerprint" && "$spatial_rows" == "$global_rows" ]] || {
  record_stage integrity_population failed "fingerprint or row count mismatch"
  finish_campaign failed_integrity
  exit 1
}
jq -S 'del(.output_dir,.consumer_readout)' "$run_root/integrity-spatial-query/config.json" \
  >"$run_root/integrity-spatial-query/normalized-config.json"
jq -S 'del(.output_dir,.consumer_readout)' "$run_root/integrity-global-mean/config.json" \
  >"$run_root/integrity-global-mean/normalized-config.json"
cmp -s "$run_root/integrity-spatial-query/normalized-config.json" \
  "$run_root/integrity-global-mean/normalized-config.json" || {
  record_stage integrity_normalized_config failed
  finish_campaign failed_integrity
  exit 1
}
jq --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  --arg fingerprint "$spatial_fingerprint" --argjson smoke_rows "$spatial_rows" \
  '.physical_batch=$physical_batch | .grad_accum=$grad_accum | .effective_batch=1024
    | .integrity_smoke_population_fingerprint=$fingerprint | .integrity_smoke_rows=$smoke_rows' \
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
  "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" --device cuda:0 \
    --seed 424242 --synthetic-episodes 64 --physical-batch "$eval_batch" --ptrm-k 1 \
    --ptrm-noise 0 --ensemble-members 1 --episode-jsonl "$eval_dir/episodes.jsonl" \
    --output "$eval_dir/eval_report.json" >"$eval_dir/eval.log" 2>&1
  sha256sum "$checkpoint" "$config" "$eval_dir/eval_report.json" "$eval_dir/episodes.jsonl" \
    >"$eval_dir/sha256.txt"
}

write_decision() {
  local report="$1" train_report="$2" destination="$3"
  jq --slurpfile train "$train_report" '
    def seam($name): (.synthetic_dynamics.representation_seams[$name] // {});
    def h8raw: if ((.synthetic_dynamics.copy_forward.h8.mean // 0) > 0)
      then ((.synthetic_dynamics.rollout.h8.mean // 1e300) /
        .synthetic_dynamics.copy_forward.h8.mean) else null end;
    {schema:"p2.consumer_readout_decision.v1",
      topology:$train[0].experiment.consumer_readout,
      training_population_fingerprint:$train[0].training_population_fingerprint,
      training_population_rows:$train[0].training_population_rows,
      gradient_pressure:$train[0].gradient_pressure,
      q:.synthetic_planner.q,
      pooled_representation:.synthetic_dynamics.representation,
      consumer_readout_representation:seam("prediction_final_consumer_readout"),
      spatial_representation:seam("prediction_final_spatial"),
      board_probe:.board_probe,
      changed_transitions:.synthetic_dynamics.changed_transitions,
      action_intervention:.synthetic_dynamics.action_diagnostics.by_source.random_one_step.shuffle,
      h8_open:.synthetic_dynamics.rollout.h8,
      h8_copy:.synthetic_dynamics.copy_forward.h8,
      gates:{
        consumer_readout_rank:((seam("prediction_final_consumer_readout").effective_rank_fraction // 0) >= 0.10),
        spatial_rank:((seam("prediction_final_spatial").effective_rank_fraction // 0) >= 0.10),
        board_trusted:(.board_probe.metrics.trusted // false),
        board_beats_copy:((.board_probe.metrics.improvement_fraction // -1) > 0),
        board_changed_f1:((.board_probe.metrics.changed_patch_f1 // 0) >= 0.5),
        changed_copy:((.synthetic_dynamics.changed_transitions.improvement_ci95_low // -1) > 0),
        action_intervention:((.synthetic_dynamics.action_diagnostics.by_source.random_one_step.shuffle.ratio_ci95_low // 0) > 1),
        q_not_saturated:((.synthetic_planner.q.saturated // true) == false),
        q_both_classes:(((.synthetic_planner.q.positive_label_rate // -1) >= 0.1)
          and ((.synthetic_planner.q.positive_label_rate // 2) <= 0.9)),
        q_balanced:((.synthetic_planner.q.balanced_accuracy // 0) > 0.5),
        q_brier:((.synthetic_planner.q.brier // 1) < 0.25),
        h8_finite:((.synthetic_dynamics.rollout.h8.finite_n // 0) == (.synthetic_dynamics.rollout.h8.n // -1)),
        h8_raw_aggregate:((h8raw // 1e300) <= 1),
        h8_normalized_median:((.synthetic_dynamics.rollout.h8.normalized_median // 1e300) <= 1),
        h8_normalized_p95:((.synthetic_dynamics.rollout.h8.normalized_p95 // 1e300) <= 10),
        h8_normalized_cvar95:((.synthetic_dynamics.rollout.h8.normalized_cvar95 // 1e300) <= 100),
        h8_fraction_beating_copy:((.synthetic_dynamics.rollout.h8.fraction_beating_copy // 0) >= 0.5),
        h8_tail_alarm:((.synthetic_dynamics.rollout.h8.normalized_mean // 1e300) <= 1)
      }} | .absolute_gate_pass=(.gates | [.[]] | all)' "$report" >"$destination"
}

run_arm() {
  local topology="$1" arm="consumer_readout_${topology//-/_}"
  local arm_dir="$run_root/seed-1/$arm" sampler_pid update checkpoint eval_dir
  mkdir -p -- "$arm_dir/telemetry"
  jq -nc --arg topology "${topology//-/_}" --arg git_sha "$git_sha" --arg binary_sha "$binary_sha" \
    --argjson batch "$physical_batch" --argjson accum "$grad_accum" \
    '{schema:"p2.consumer_readout_arm.v1",seed:1,consumer_readout:$topology,
      physical_batch:$batch,grad_accum:$accum,effective_batch:1024,git_sha:$git_sha,
      binary_sha256:$binary_sha,checkpoints:[250,500,750,1000],promotion:"locked"}' \
    >"$arm_dir/run.json"
  sample_gpu >>"$arm_dir/telemetry/gpu.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
  "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons q_calibration \
    --steps-per-lesson "$train_steps" --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 250 --profile-update 250 --consumer-readout "$topology" \
    "${common_args[@]}" --output-dir "$arm_dir" > >(tee "$arm_dir/train.log") 2>&1
  jq -e --arg topology "${topology//-/_}" --argjson batch "$physical_batch" --argjson accum "$grad_accum" \
    '.global_step == 1000 and .status == "completed" and .physical_batch == $batch
      and .grad_accum == $accum and .experiment.consumer_readout == $topology
      and .profile.published.update == 250 and .training_population_rows > 0
      and .gradient_pressure.encoder_readout_weighted_l2 != null' "$arm_dir/train_report.json" >/dev/null
  jq -e --arg topology "${topology//-/_}" --argjson batch "$physical_batch" --argjson accum "$grad_accum" '
    .consumer_readout == $topology and .physical_batch == $batch and .grad_accum == $accum
    and .lessons == ["q_calibration"] and .shuffled_episodes == true
    and .sigreg_weight == 0.003 and .sigreg_target == "temporal_residual"
    and .sigreg_statistic == "quantile" and .sigreg_temporal_window == 8
    and .sigreg_global_mix == 0 and .sigreg_spatial == true and .sigreg_spatial_pool == true
    and .q_weight == 0.1 and .stop_grad_q_y == false and .ptrm_rank_every == 0
    and .event_weight == 0 and .reliability_weight == 0 and .rollout_weight == 0
    and .prefix_weight == 0 and .world_core_v2 == false and .world_core_v3 == false' \
    "$arm_dir/config.json" >/dev/null
  for update in 250 500 750 1000; do
    printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$arm_dir" "$update"
    eval_dir="$arm_dir/eval-update-$update"
    [[ -s "$checkpoint" ]] || { printf 'missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
    run_eval "$checkpoint" "$arm_dir/config.json" "$eval_dir"
  done
  write_decision "$arm_dir/eval-update-1000/eval_report.json" "$arm_dir/train_report.json" \
    "$arm_dir/decision.json"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  (( $(awk 'END {print NR}' "$arm_dir/telemetry/gpu.csv") >= 10 ))
  sha256sum "$arm_dir/config.json" "$arm_dir/train_report.json" "$arm_dir/decision.json" \
    >"$arm_dir/artifacts.sha256"
  trap - EXIT
}

failed=0
for topology in spatial-query global-mean; do
  arm="consumer_readout_${topology//-/_}"
  arm_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if (set -euo pipefail; run_arm "$topology"); then status=passed; else status=failed; failed=$((failed + 1)); fi
  jq -nc --arg arm "$arm" --arg status "$status" --arg started "$arm_started" \
    --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{arm:$arm,status:$status,started_utc:$started,finished_utc:$finished}' >>"$run_root/arms.jsonl"
  record_stage "$arm" "$status"
done

if (( failed == 0 )); then
  spatial="$run_root/seed-1/consumer_readout_spatial_query"
  global="$run_root/seed-1/consumer_readout_global_mean"
  [[ "$(jq -r .training_population_fingerprint "$spatial/train_report.json")" == \
     "$(jq -r .training_population_fingerprint "$global/train_report.json")" ]] || {
    record_stage final_population_integrity failed
    finish_campaign failed_integrity
    exit 1
  }
  [[ "$(jq -r .training_population_rows "$spatial/train_report.json")" == \
     "$(jq -r .training_population_rows "$global/train_report.json")" ]] || {
    record_stage final_training_row_integrity failed
    finish_campaign failed_integrity
    exit 1
  }
  jq -S 'del(.output_dir,.consumer_readout)' "$spatial/config.json" >"$spatial/normalized-config.json"
  jq -S 'del(.output_dir,.consumer_readout)' "$global/config.json" >"$global/normalized-config.json"
  cmp -s "$spatial/normalized-config.json" "$global/normalized-config.json" || {
    record_stage final_normalized_config_integrity failed
    finish_campaign failed_integrity
    exit 1
  }
  [[ "$(jq -r .board_probe.population_fingerprint "$spatial/eval-update-1000/eval_report.json")" == \
     "$(jq -r .board_probe.population_fingerprint "$global/eval-update-1000/eval_report.json")" ]] || {
    record_stage final_eval_population_integrity failed
    finish_campaign failed_integrity
    exit 1
  }
  finish_campaign complete_pending_analysis
else
  jq --argjson failed "$failed" '.failed_arms=$failed' "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  finish_campaign partial_failure
fi
printf 'consumer-readout-v1 finished with %s failed arms; promotion remains locked\n' "$failed"
exit "$([[ "$failed" -eq 0 ]] && printf 0 || printf 1)"
