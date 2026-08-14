#!/usr/bin/env bash
# Fail-closed six-arm deterministic nonlinear coarse semantic-access campaign.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
parent_root="${P2_AUDIT_PARENT_ROOT:?set the pressure-grounding parent root}"
previous_root="${P2_PREVIOUS_SEMANTIC_ROOT:?set the checksum-verified B1b root}"
run_root="${P2_SEMANTIC_FIXED_ROOT:-$repo_root/runs/p2/semantic-access-fixed-coarse-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_SEMANTIC_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_SHA:?set the reviewed fixed-probe commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed fixed-probe binary hash}"
: "${P2_EXPECTED_PARENT_SHA:?set the frozen parent commit}"
: "${P2_EXPECTED_PARENT_BINARY_SHA:?set the frozen parent binary hash}"
for command in git jq nvidia-smi sha256sum awk realpath seq sleep timeout; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ "$eval_batch" == 256 ]] || { printf 'physical batch must remain sealed at 256\n' >&2; exit 2; }
[[ "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$tofy_bin" && -d "$parent_root" && -d "$previous_root" ]] || exit 2
parent_root="$(realpath "$parent_root")"
previous_root="$(realpath "$previous_root")"
run_root="$(realpath -m "$run_root")"

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" && "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
parent_sha="$(jq -er '.git_sha' "$parent_root/campaign.json")"
[[ "$parent_sha" == "$P2_EXPECTED_PARENT_SHA" ]] || exit 2
[[ "$(jq -er '.binary_sha256' "$parent_root/campaign.json")" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || exit 2
sha256sum -c "$previous_root/SHA256SUMS" >/dev/null
[[ "$(jq -er '.schema' "$previous_root/summary.json")" == p2.semantic_access_summary.v1_1_stage_b1b ]] || exit 2
[[ "$(jq -er '.phase' "$previous_root/summary.json")" == selection_only ]] || exit 2
[[ "$(jq -er '.decision' "$previous_root/summary.json")" == selector_invalid_no_final_partition_scored ]] || exit 2
expected_fingerprint="$(jq -er '.population_fingerprint // empty' "$previous_root/campaign.json")"
[[ "$expected_fingerprint" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
previous_manifest_sha="$(sha256sum "$previous_root/SHA256SUMS" | awk '{print $1}')"

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
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$rc" \
      '.status="failed_integrity_or_infrastructure"|.finished_utc=$at|.exit_code=$rc' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg binary_sha "$binary_sha" --arg parent_root "$parent_root" --arg parent_sha "$parent_sha" \
  --arg parent_binary "$P2_EXPECTED_PARENT_BINARY_SHA" --arg previous_root "$previous_root" \
  --arg previous_manifest_sha "$previous_manifest_sha" --arg fingerprint "$expected_fingerprint" \
  --arg gpu "${gpu_names[0]}" \
  '{schema:"p2.semantic_access.fixed_coarse_campaign.v1",status:"running",started_utc:$started,
    git_sha:$git_sha,binary_sha256:$binary_sha,parent_root:$parent_root,parent_git_sha:$parent_sha,
    parent_binary_sha256:$parent_binary,previous_semantic_root:$previous_root,
    previous_root_manifest_sha256:$previous_manifest_sha,expected_population_fingerprint:$fingerprint,
    gpu_name:$gpu,arms:["S0G0","ScalG0","ScurG0","S0G1","ScalG1","ScurG1"],
    frozen_protocol:{population_seed:424244,episodes_per_source:64,physical_batch:256,
      feature_map:"rand-0.9_chacha8_seeded_mixed_sparse_relu_quadratic_v2",feature_width:256,
      inputs_per_feature:8,feature_seeds:[4302768118693889,4302768118693890,4302768118693891],
      seed_aggregation:"arithmetic_mean_predictions_no_seed_selection",ridge:0.01,
      model_weights_frozen:true,optimizer_used:false,
      target_routes:["true_next_encoder_fit","target_fit_transfer_to_predicted_next","predicted_next_refit"],
      inference:"descriptive_only",final_used_for_selection:false},promotion:"locked_pending_analysis"}' \
  >"$run_root/campaign.json"

sample_gpu() {
  local sleeper=""
  trap 'if [[ -n "$sleeper" ]]; then kill "$sleeper" 2>/dev/null || true; fi' EXIT INT TERM
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
    sleep "$gpu_interval" & sleeper=$!; wait "$sleeper" || exit; sleeper=""
  done
}
sample_gpu >>"$run_root/gpu.csv" & telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"
: >"$run_root/INPUTS.sha256"

validate_report() {
  local report="$1" phase="$2"
  jq -e --arg phase "$phase" --arg fingerprint "$expected_fingerprint" '
    .schema=="p2.semantic_access.fixed_coarse.v1"
    and .population_seed==424244 and .synthetic_episodes_per_source==64
    and .population_fingerprint==$fingerprint and (.target_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
    and (.predicted_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
    and .protocol.physical_batch==256 and .protocol.feature_width==256
    and .protocol.inputs_per_feature==8
    and .protocol.feature_seeds==[4302768118693889,4302768118693890,4302768118693891]
    and .protocol.seed_aggregation=="arithmetic_mean_predictions_no_seed_selection"
    and .protocol.ridge==0.01 and .protocol.learned_parameter_count_per_family==14400
    and .protocol.fixed_nonzero_coefficient_count_per_family==6912
    and .protocol.observable_control_interaction_scale==32
    and .protocol.model_weights_frozen==true and .protocol.optimizer_used==false
    and .protocol.inferential_claims_enabled==false and .model_weights_updated==false
    and .final_partition_used_for_decoder_selection==false and .model_level_conclusion_permitted==false
    and (.evaluator_status=="qualified" or ($phase=="selection" and .evaluator_status=="control_invalid"))
    and (.families|length)==2
    and ([.families[].name]|sort)==["contextual_3x3_global","local"]
    and ([.families[].input_dim]|sort)==[128,384]
    and ([.families[].learned_parameter_count]|all(.==14400))
    and ([.families[].fixed_nonzero_coefficient_count]|all(.==6912))
    and ([.families[].qualification.per_seed|length]|all(.==3))
    and ([.families[].qualification.per_seed[].feature_map_sha256]|all(test("^sha256:[0-9a-f]{64}$")))
    and (if .evaluator_status=="qualified" then
      ([.families[].route_selection_diagnostics|length]|all(.==2))
      else ([.families[].route_selection_diagnostics|length]|all(.==0)) end)
    and ([.families[].route_selection_diagnostics[].per_seed|length]|all(.==3))
    and (if $phase=="selection" then .execution_phase=="selection_only"
      and .final_partition_scored==false and .descriptive_seam_interpretation_permitted==false
      and ([.families[].routes|length]|all(.==0))
    else .execution_phase=="final_score" and .final_partition_scored==true
      and .descriptive_seam_interpretation_permitted==true
      and ([.families[].routes|length]|all(.==3))
      and ([.families[].routes[].route]|unique|sort)==
        ["predicted_next_refit","target_fit_transfer_to_predicted_next","true_next_encoder_fit"]
      and ([.families[].routes[].ensemble_final_mse,.families[].routes[].ridge_final_mse]
        |flatten|all(type=="number" and .>=0 and .<1e300))
      and ([.families[].routes[].per_seed_final|length]|all(.==3)) end)' "$report" >/dev/null
}

audit_arm() {
  local arm="$1" phase="$2" arm_dir checkpoint config manifest out_dir report selection_report selection_sha
  [[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || return 1
  [[ "$(sha256sum "$tofy_bin"|awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || return 1
  arm_dir="$parent_root/seed-1/$arm"
  checkpoint="$arm_dir/checkpoints/step-000000000500/model.safetensors"
  config="$arm_dir/config.json"
  manifest="$arm_dir/train-500.sha256"
  [[ -s "$checkpoint" && -s "$config" && -s "$manifest" ]] || return 1
  sha256sum -c "$manifest" >/dev/null || return 1
  sha256sum "$checkpoint" "$config" "$manifest" >>"$run_root/INPUTS.sha256"
  out_dir="$run_root/$phase/$arm"; mkdir -p -- "$out_dir"; report="$out_dir/semantic.json"
  local -a command=("$tofy_bin" p2-semantic-access-fixed-audit --checkpoint "$checkpoint"
    --train-config "$config" --device cuda:0 --physical-batch 256
    --require-population-fingerprint "$expected_fingerprint")
  if [[ "$phase" == selection ]]; then
    command+=(--selection-only)
  else
    selection_report="$(jq -er '.selected_report' "$run_root/selection/$arm/receipt.json")"
    selection_sha="$(sha256sum "$selection_report"|awk '{print $1}')"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) $arm $selection_sha" >"$out_dir/final_access_started"
    command+=(--selection-reference "$selection_report" --selection-reference-sha256 "$selection_sha")
  fi
  timeout --foreground 12h "${command[@]}" --output "$report" >"$out_dir/audit.log" 2>&1
  validate_report "$report" "$phase"
  [[ "$(jq -r '.checkpoint_sha256' "$report")" == "$(sha256sum "$checkpoint"|awk '{print $1}')" ]] || return 1
  [[ "$(jq -r '.train_config_sha256' "$report")" == "$(sha256sum "$config"|awk '{print $1}')" ]] || return 1
  jq -nc --arg arm "$arm" --arg report "$report" --arg sha "$(sha256sum "$report"|awk '{print $1}')" \
    '{arm:$arm,status:"primary_succeeded",selected_report:$report,selected_report_sha256:$sha}' >"$out_dir/receipt.json"
  sha256sum "$checkpoint" "$config" "$report" "$out_dir/receipt.json" >"$out_dir/SHA256SUMS"
}

for arm in ScalG0 ScurG1 S0G0 ScalG1 ScurG0 S0G1; do audit_arm "$arm" selection; done
sha256sum -c "$run_root/INPUTS.sha256" >/dev/null
mapfile -t selection_reports < <(for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do jq -er '.selected_report' "$run_root/selection/$arm/receipt.json"; done)
if ! jq -e -s 'all(.evaluator_status=="qualified")' "${selection_reports[@]}" >/dev/null; then
  jq -s '{schema:"p2.semantic_access.fixed_coarse_summary.v1",phase:"selection_only",
    evaluator_status:"invalid",decision:"fixed_control_invalid_no_final_partition_scored",
    arms:map({checkpoint,evaluator_status,families})}' "${selection_reports[@]}" >"$run_root/summary.json"
  jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson elapsed "$(( $(date +%s)-started_epoch ))" \
    '.status="complete_pending_analysis"|.finished_utc=$finished|.elapsed_seconds=$elapsed
      |.decision="fixed_control_invalid_no_final_partition_scored"' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; telemetry_pid=""
  sha256sum "$run_root/campaign.json" "$run_root/summary.json" "$run_root/INPUTS.sha256" "$run_root/gpu.csv" \
    "${selection_reports[@]}" "$run_root"/selection/*/receipt.json "$run_root"/selection/*/audit.log \
    >"$run_root/SHA256SUMS"
  finalized=true
  printf 'fixed control invalid; no final rows scored; root=%s\n' "$run_root"
  exit 0
fi

# No final invocation may be retried: a failure after the marker invalidates the one-shot final phase.
for arm in ScurG1 ScalG0 S0G1 ScurG0 ScalG1 S0G0; do audit_arm "$arm" final; done
mapfile -t final_reports < <(for arm in S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1; do jq -er '.selected_report' "$run_root/final/$arm/receipt.json"; done)
jq -s '{schema:"p2.semantic_access.fixed_coarse_summary.v1",evaluator_status:"qualified",
  decision:"fixed_coarse_seam_matrix_ready_for_analysis",
  arms:map({checkpoint,evaluator_status,families})}' "${final_reports[@]}" >"$run_root/summary.json"
jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson elapsed "$(( $(date +%s)-started_epoch ))" \
  '.status="complete_pending_analysis"|.finished_utc=$finished|.elapsed_seconds=$elapsed
    |.decision="fixed_coarse_seam_matrix_ready_for_analysis"' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; telemetry_pid=""
sha256sum "$run_root/campaign.json" "$run_root/summary.json" "$run_root/INPUTS.sha256" "$run_root/gpu.csv" \
  "${selection_reports[@]}" "${final_reports[@]}" "$run_root"/selection/*/receipt.json \
  "$run_root"/selection/*/audit.log "$run_root"/final/*/receipt.json "$run_root"/final/*/audit.log \
  "$run_root"/final/*/final_access_started >"$run_root/SHA256SUMS"
finalized=true
printf 'fixed coarse semantic campaign complete; root=%s\n' "$run_root"
