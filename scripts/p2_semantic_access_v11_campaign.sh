#!/usr/bin/env bash
# Fail-closed six-checkpoint Semantic Access V1.1 seam campaign.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
parent_root="${P2_AUDIT_PARENT_ROOT:?set the pressure-grounding parent root}"
run_root="${P2_SEMANTIC_V11_ROOT:-$repo_root/runs/p2/semantic-access-v1_1-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_SEMANTIC_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
previous_root="${P2_PREVIOUS_SEMANTIC_ROOT:?set the checksum-verified V1 audit root}"

: "${P2_EXPECTED_SHA:?set the reviewed V1.1 commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed V1.1 binary hash}"
: "${P2_EXPECTED_PARENT_SHA:?set the frozen parent commit}"
: "${P2_EXPECTED_PARENT_BINARY_SHA:?set the frozen parent binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath seq sleep tee timeout; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
for value in "$eval_batch" "$gpu_interval"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid positive integer: %s\n' "$value" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing V1.1 binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ -d "$parent_root" ]] || { printf 'missing parent root: %s\n' "$parent_root" >&2; exit 2; }
parent_root="$(realpath "$parent_root")"
previous_root="$(realpath "$previous_root")"
run_root="$(realpath -m "$run_root")"

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
parent_sha="$(jq -er '.git_sha' "$parent_root/campaign.json")"
[[ "$parent_sha" == "$P2_EXPECTED_PARENT_SHA" ]] || { printf 'parent SHA mismatch\n' >&2; exit 2; }
[[ "$(jq -er '.binary_sha256' "$parent_root/campaign.json")" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || {
  printf 'parent binary SHA mismatch\n' >&2; exit 2;
}
sha256sum -c "$previous_root/SHA256SUMS" >/dev/null
[[ "$(jq -er '.schema' "$previous_root/summary.json")" == "p2.semantic_access_summary.v1" ]] || {
  printf 'previous semantic audit schema mismatch\n' >&2; exit 2;
}
previous_fingerprint="$(jq -er '.population_fingerprint' "$previous_root/summary.json")"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
(( initial_memory <= 1024 && initial_utilization == 0 )) || {
  printf 'GPU is not idle (memory=%s MiB utilization=%s%%)\n' "$initial_memory" "$initial_utilization" >&2
  exit 2
}

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root"
started_epoch="$(date +%s)"
telemetry_pid=""
finalized=false

cleanup() {
  local rc="$?"
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$rc" \
      '.status="failed_integrity_or_infrastructure" | .finished_utc=$at | .exit_code=$rc' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg binary_sha "$binary_sha" --arg parent_root "$parent_root" --arg parent_sha "$parent_sha" \
  --arg parent_binary_sha "$P2_EXPECTED_PARENT_BINARY_SHA" --arg gpu "${gpu_names[0]}" \
  --arg previous "$previous_fingerprint" --arg previous_root "$previous_root" \
  --argjson eval_batch "$eval_batch" \
  '{schema:"p2.semantic_access_campaign.v1_1_stage_b1b",status:"running",started_utc:$started,
    git_sha:$git_sha,binary_sha256:$binary_sha,parent_root:$parent_root,parent_git_sha:$parent_sha,
    parent_binary_sha256:$parent_binary_sha,
    gpu_name:$gpu,arms:["S0G0","ScalG0","ScurG0","S0G1","ScalG1","ScurG1"],
    previous_semantic_root:$previous_root,previous_population_fingerprint:$previous,
    frozen_protocol:{checkpoint_update:500,population_seed:424244,episodes_per_source:64,
      physical_batch:$eval_batch,decoder_batch:4096,decoder_gradient_accumulation:1,decoder_hidden:64,
      max_optimizer_steps:4800,evaluate_every_steps:25,patience_evaluations:8,
      target_routes:["true_next_encoder_fit","target_fit_transfer_to_predicted_next","predicted_next_refit"],
      inference:"descriptive_only",final_used_for_selection:false},
    promotion:"locked_pending_analysis"}' >"$run_root/campaign.json"

sample_gpu() {
  local sampler_sleep_pid=""
  trap 'if [[ -n "$sampler_sleep_pid" ]]; then kill "$sampler_sleep_pid" 2>/dev/null || true; fi' EXIT INT TERM
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval" &
    sampler_sleep_pid=$!
    wait "$sampler_sleep_pid" || exit
    sampler_sleep_pid=""
  done
}
sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"
: >"$run_root/INPUTS.sha256"

wait_for_idle_gpu() {
  local attempt used utilization
  for attempt in $(seq 1 120); do
    read -r used utilization < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
    )
    if (( used <= 1024 && utilization == 0 )); then return 0; fi
    sleep 5
  done
  return 1
}

audit_arm() {
  local arm="$1" phase="$2" arm_dir checkpoint config manifest primary_dir recovery_dir selected
  [[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || return 1
  [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || return 1
  arm_dir="$parent_root/seed-1/$arm"
  checkpoint="$arm_dir/checkpoints/step-000000000500/model.safetensors"
  config="$arm_dir/config.json"
  manifest="$arm_dir/train-500.sha256"
  [[ -s "$checkpoint" && -s "$config" && -s "$manifest" ]] || return 1
  sha256sum -c "$manifest" >/dev/null || return 1
  awk -v path="$checkpoint" '$2==path {found=1} END {exit !found}' "$manifest" || return 1
  awk -v path="$config" '$2==path {found=1} END {exit !found}' "$manifest" || return 1
  sha256sum "$checkpoint" "$config" "$manifest" >>"$run_root/INPUTS.sha256" || return 1
  primary_dir="$run_root/$phase/$arm"
  mkdir -p -- "$primary_dir"
  local -a command=("$tofy_bin" p2-semantic-access-v11-audit
    --checkpoint "$checkpoint" --train-config "$config" --device cuda:0
    --physical-batch "$eval_batch" --forbid-population-fingerprint "$previous_fingerprint")
  if [[ "$phase" == selection ]]; then
    command+=(--selection-only)
  else
    command+=(--selection-reference "$(jq -er '.selected_report' "$run_root/selection/$arm/receipt.json")")
  fi
  if timeout --foreground 12h "${command[@]}" --output "$primary_dir/semantic.json" \
    >"$primary_dir/audit.log" 2>&1; then
    selected="$primary_dir/semantic.json"
    jq -nc --arg arm "$arm" --arg status primary_succeeded --arg report "$selected" \
      '{arm:$arm,status:$status,selected_report:$report}' >"$primary_dir/receipt.json"
  else
    wait_for_idle_gpu || return 1
    recovery_dir="$run_root/recoveries/$phase/$arm/attempt-1"
    mkdir -p -- "$recovery_dir"
    if CUDA_LAUNCH_BLOCKING=1 timeout --foreground 12h "${command[@]}" \
      --output "$recovery_dir/semantic.json" >"$recovery_dir/audit.log" 2>&1; then
      selected="$recovery_dir/semantic.json"
      jq -nc --arg arm "$arm" --arg status recovered_after_primary_failure --arg report "$selected" \
        --arg primary_log "$primary_dir/audit.log" \
        '{arm:$arm,status:$status,selected_report:$report,primary_failure_log:$primary_log}' \
        >"$primary_dir/receipt.json"
    else
      jq -nc --arg arm "$arm" --arg status recovery_failed \
        '{arm:$arm,status:$status}' >"$primary_dir/receipt.json"
      return 1
    fi
  fi
  jq -e '
    .schema=="p2.semantic_access.v1_1_stage_b1b"
    and .population_seed==424244 and .synthetic_episodes_per_source==64
    and .model_weights_updated==false and .final_partition_used_for_decoder_selection==false
    and .protocol.inferential_claims_enabled==false and .protocol.model_weights_frozen==true
    and .protocol.max_optimizer_steps==4800 and .protocol.decoder_hidden==64
    and .protocol.decoder_batch==4096 and .protocol.decoder_gradient_accumulation==1
    and .protocol.physical_batch==$eval_batch
    and .protocol.evaluate_every_steps==25 and .protocol.patience_evaluations==8
    and .protocol.learning_rate==0.001 and .protocol.weight_decay==0.0001
    and .protocol.parameter_cap==100000
    and .protocol.observable_control_mse_ceiling==0.04
    and .protocol.observable_control_min_fractional_reduction==0.90
    and .protocol.observable_control_min_absolute_improvement==0.01
    and .target=="descriptive_16_colour_counts_per_8x8_patch_status_row_excluded"
    and (.population_fingerprint|startswith("sha256:"))
    and .model_level_conclusion_permitted==false
    and (.target_latents_sha256|startswith("sha256:"))
    and (.predicted_latents_sha256|startswith("sha256:"))
    and (.evaluator_status=="qualified" or .evaluator_status=="control_invalid"
      or .evaluator_status=="route_selection_budget_censored")
    and (.families|length)==2
    and ([.families[].name]|sort)==["contextual_3x3_global","local"]
    and ([.families[].qualification.passed]|all(type=="boolean"))
    and (if .evaluator_status=="qualified" then
        ([.families[].qualification.passed]|all(.==true))
        and ([.families[].route_selection_diagnostics|length]|all(.==2))
        and ([.families[].route_selection_diagnostics[].converged_before_budget]|all(.==true))
      elif .evaluator_status=="control_invalid" then
        ([.families[].qualification.passed]|any(.==false))
        and ([.families[].route_selection_diagnostics|length]|all(.==0))
        and ([.families[].routes|length]|all(.==0))
        and .descriptive_seam_interpretation_permitted==false
      else
        ([.families[].qualification.passed]|all(.==true))
        and ([.families[].route_selection_diagnostics|length]|all(.==2))
        and ([.families[].route_selection_diagnostics[].converged_before_budget]|any(.==false))
        and ([.families[].routes|length]|all(.==0))
        and .descriptive_seam_interpretation_permitted==false
      end)
    and (if $phase=="selection" then
        .execution_phase=="selection_only" and .final_partition_scored==false
        and .descriptive_seam_interpretation_permitted==false
        and ([.families[].routes|length]|all(.==0))
      else
        .execution_phase=="final_score"
        and (if .evaluator_status=="qualified" then
          .final_partition_scored==true and .descriptive_seam_interpretation_permitted==true
          and ([.families[].routes|length]|all(.==3))
          and ([.families[].routes[].route]|unique|sort)==
            ["predicted_next_refit","target_fit_transfer_to_predicted_next","true_next_encoder_fit"]
          and ([.families[].routes[].ridge_final_mse,.families[].routes[].residual_final_mse]
            | flatten | all(type=="number" and .>=0 and .<1e300))
        else .final_partition_scored==false and ([.families[].routes|length]|all(.==0)) end)
      end)' --arg phase "$phase" --argjson eval_batch "$eval_batch" "$selected" >/dev/null || return 1
  [[ "$(jq -r '.population_fingerprint' "$selected")" != "$previous_fingerprint" ]] || return 1
  [[ "$(jq -r '.checkpoint_sha256' "$selected")" == "$(sha256sum "$checkpoint" | awk '{print $1}')" ]] || return 1
  [[ "$(jq -r '.train_config_sha256' "$selected")" == "$(sha256sum "$config" | awk '{print $1}')" ]] || return 1
  sha256sum "$checkpoint" "$config" "$selected" "$primary_dir/receipt.json" >"$primary_dir/SHA256SUMS"
}

# Phase 1: every arm must qualify without scoring any final row.
for arm in ScalG0 ScurG1 S0G0 ScalG1 ScurG0 S0G1; do
  kill -0 "$telemetry_pid" 2>/dev/null || { printf 'telemetry process failed\n' >&2; exit 1; }
  audit_arm "$arm" selection
done
sha256sum -c "$run_root/INPUTS.sha256" >/dev/null
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || exit 1
[[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || exit 1

mapfile -t selection_reports < <(
  for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do
    jq -er '.selected_report' "$run_root/selection/$arm/receipt.json"
  done
)
selection_fingerprint="$(jq -r '.population_fingerprint' "${selection_reports[0]}")"
for report in "${selection_reports[@]}"; do
  [[ "$(jq -r '.population_fingerprint' "$report")" == "$selection_fingerprint" ]] || {
    printf 'selection population fingerprint mismatch\n' >&2; exit 1;
  }
done
if ! jq -e -s 'all(.evaluator_status=="qualified")' "${selection_reports[@]}" >/dev/null; then
  jq -s '{schema:"p2.semantic_access_summary.v1_1_stage_b1b",phase:"selection_only",
    evaluator_status:"invalid",decision:"selector_invalid_no_final_partition_scored",
    arms:map({checkpoint:.checkpoint,evaluator_status,families})}' \
    "${selection_reports[@]}" >"$run_root/summary.json"
  jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg decision "selector_invalid_no_final_partition_scored" \
    --arg fingerprint "$selection_fingerprint" \
    --argjson elapsed "$(( $(date +%s) - started_epoch ))" \
    '.status="complete_pending_analysis" | .finished_utc=$finished | .elapsed_seconds=$elapsed
      | .decision=$decision | .population_fingerprint=$fingerprint' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=""
  sha256sum "$run_root/campaign.json" "$run_root/summary.json" "$run_root/INPUTS.sha256" \
    "$run_root/gpu.csv" "${selection_reports[@]}" "$run_root"/selection/*/receipt.json \
    "$run_root"/selection/*/audit.log >"$run_root/SHA256SUMS"
  while IFS= read -r log; do sha256sum "$log"; done \
    < <(compgen -G "$run_root/recoveries/*/*/attempt-1/audit.log" || true) >>"$run_root/SHA256SUMS"
  finalized=true
  printf 'Semantic Access V1.1 selection invalid; no final rows scored; root=%s\n' "$run_root"
  exit 0
fi

# Phase 2 is allowed only after all six phase-1 selectors qualified.
for arm in ScurG1 ScalG0 S0G1 ScurG0 ScalG1 S0G0; do
  kill -0 "$telemetry_pid" 2>/dev/null || { printf 'telemetry process failed\n' >&2; exit 1; }
  audit_arm "$arm" final
done
sha256sum -c "$run_root/INPUTS.sha256" >/dev/null
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || exit 1
[[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || exit 1

mapfile -t selected_reports < <(
  for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do
    jq -er '.selected_report' "$run_root/final/$arm/receipt.json"
  done
)
fingerprint="$(jq -r '.population_fingerprint' "${selected_reports[0]}")"
[[ "$fingerprint" != "$previous_fingerprint" ]] || {
  printf 'fresh-population check failed\n' >&2; exit 1;
}
for report in "${selected_reports[@]}"; do
  [[ "$(jq -r '.population_fingerprint' "$report")" == "$fingerprint" ]] || {
    printf 'population fingerprint mismatch\n' >&2; exit 1;
  }
done

jq -s '
  map({arm:.arm,receipt_status:.receipt.status,evaluator_status:.report.evaluator_status,
    model_level_conclusion_permitted:.report.model_level_conclusion_permitted,
    qualifications:(.report.families|map({name,qualification})),
    routes:(.report.families|map({name,routes}))}) as $rows
  | {schema:"p2.semantic_access_summary.v1_1_stage_b1b",arms:$rows,
      evaluator_status:(if ($rows|all(.evaluator_status=="qualified")) then "qualified" else "invalid" end),
      decision:(if ($rows|all(.evaluator_status=="qualified"))
        then "qualified_seam_matrix_ready_for_analysis"
        else "evaluator_invalid_no_model_conclusion" end)}' \
  <(
    for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do
      receipt="$run_root/final/$arm/receipt.json"
      report="$(jq -er '.selected_report' "$receipt")"
      jq -nc --arg arm "$arm" --slurpfile receipt "$receipt" --slurpfile report "$report" \
        '{arm:$arm,receipt:$receipt[0],report:$report[0]}'
    done
  ) >"$run_root/summary.json.tmp"
mv -- "$run_root/summary.json.tmp" "$run_root/summary.json"

jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg decision "$(jq -r '.decision' "$run_root/summary.json")" \
  --arg fingerprint "$fingerprint" --argjson elapsed "$(( $(date +%s) - started_epoch ))" \
  '.status="complete_pending_analysis" | .finished_utc=$finished | .elapsed_seconds=$elapsed
    | .decision=$decision | .population_fingerprint=$fingerprint' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
telemetry_pid=""
sha256sum "$run_root/campaign.json" "$run_root/summary.json" "$run_root/INPUTS.sha256" \
  "$run_root/gpu.csv" "${selection_reports[@]}" "${selected_reports[@]}" \
  "$run_root"/selection/*/receipt.json "$run_root"/selection/*/audit.log \
  "$run_root"/final/*/receipt.json "$run_root"/final/*/audit.log \
  >"$run_root/SHA256SUMS"
while IFS= read -r log; do sha256sum "$log"; done \
  < <(compgen -G "$run_root/recoveries/*/*/attempt-1/audit.log" || true) >>"$run_root/SHA256SUMS"
finalized=true
printf 'Semantic Access V1.1 complete; decision=%s root=%s\n' \
  "$(jq -r '.decision' "$run_root/summary.json")" "$run_root"
