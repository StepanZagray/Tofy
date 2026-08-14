#!/usr/bin/env bash
# Fail-closed deterministic nonlinear semantic-access campaign with sealed fitted state.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
parent_root="${P2_AUDIT_PARENT_ROOT:?set the pressure-grounding parent root}"
previous_root="${P2_PREVIOUS_SEMANTIC_ROOT:?set the checksum-verified B1b root}"
run_root="${P2_SEMANTIC_FIXED_ROOT:-$repo_root/runs/p2/semantic-access-fixed-coarse-v4-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_SEMANTIC_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
final_seeds=(424245 424246 424247)
all_arms=(S0G0 ScalG0 ScurG0 S0G1 ScalG1 ScurG1)
final_arms=(S0G1 ScurG1)

: "${P2_EXPECTED_SHA:?set the reviewed fixed-probe commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed fixed-probe binary hash}"
: "${P2_EXPECTED_PARENT_SHA:?set the frozen parent commit}"
: "${P2_EXPECTED_PARENT_BINARY_SHA:?set the frozen parent binary hash}"
for command in awk cmp find git jq nvidia-smi realpath sha256sum sort timeout; do
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

validate_runtime_identity() {
  [[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || return 1
  [[ -z "$(git -C "$repo_root" status --porcelain)" ]] || return 1
  [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || return 1
}
parent_sha="$(jq -er '.git_sha' "$parent_root/campaign.json")"
[[ "$parent_sha" == "$P2_EXPECTED_PARENT_SHA" ]] || exit 2
[[ "$(jq -er '.binary_sha256' "$parent_root/campaign.json")" == "$P2_EXPECTED_PARENT_BINARY_SHA" ]] || exit 2
sha256sum -c "$previous_root/SHA256SUMS" >/dev/null
[[ "$(jq -er '.schema' "$previous_root/summary.json")" == p2.semantic_access_summary.v1_1_stage_b1b ]] || exit 2
[[ "$(jq -er '.phase' "$previous_root/summary.json")" == selection_only ]] || exit 2
[[ "$(jq -er '.decision' "$previous_root/summary.json")" == selector_invalid_no_final_partition_scored ]] || exit 2
expected_source_fingerprint="$(jq -er '.population_fingerprint // empty' "$previous_root/campaign.json")"
[[ "$expected_source_fingerprint" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
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

stop_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
    telemetry_pid=""
  fi
}

write_root_manifest() {
  local staging="$run_root/SHA256SUMS.staging" file
  : >"$staging"
  while IFS= read -r -d '' file; do
    sha256sum "$file" >>"$staging"
  done < <(find "$run_root" -type f ! -name SHA256SUMS ! -name SHA256SUMS.staging ! -name '*.tmp' ! -name '*.staging' -print0 | sort -z)
  mv -- "$staging" "$run_root/SHA256SUMS"
  sha256sum -c "$run_root/SHA256SUMS" >/dev/null
}

cleanup() {
  local rc="$?"
  set +e
  stop_telemetry
  if [[ "$finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson rc "$rc" \
      '.status="failed_integrity_or_infrastructure"|.finished_utc=$at|.exit_code=$rc' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  fi
  if [[ "$finalized" != true && -d "$run_root" ]]; then write_root_manifest; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg git_sha "$git_sha" \
  --arg binary_sha "$binary_sha" --arg parent_root "$parent_root" --arg parent_sha "$parent_sha" \
  --arg parent_binary "$P2_EXPECTED_PARENT_BINARY_SHA" --arg previous_root "$previous_root" \
  --arg previous_manifest_sha "$previous_manifest_sha" --arg fingerprint "$expected_source_fingerprint" \
  --arg gpu "${gpu_names[0]}" \
  '{schema:"p2.semantic_access.fixed_coarse_campaign.v2",status:"running",started_utc:$started,
    git_sha:$git_sha,binary_sha256:$binary_sha,parent_root:$parent_root,parent_git_sha:$parent_sha,
    parent_binary_sha256:$parent_binary,previous_semantic_root:$previous_root,
    previous_root_manifest_sha256:$previous_manifest_sha,expected_source_population_fingerprint:$fingerprint,
    gpu_name:$gpu,selection_arms:["S0G0","ScalG0","ScurG0","S0G1","ScalG1","ScurG1"],
    final_arms:["S0G1","ScurG1"],final_population_seeds:[424245,424246,424247],
    frozen_protocol:{selection_population_seed:424244,episodes_per_source:64,physical_batch:256,
      selection_replays:2,feature_map:"rand-0.9_chacha8_seeded_mixed_sparse_relu_quadratic_v2",
      feature_width:256,inputs_per_feature:8,feature_seeds:[4302768118693889,4302768118693890,4302768118693891],
      seed_aggregation:"arithmetic_mean_predictions_no_seed_selection",ridge:0.01,
      model_weights_frozen:true,optimizer_used:false,
      target_routes:["true_next_encoder_fit","target_fit_transfer_to_predicted_next","predicted_next_refit"],
      primary_family:"local",contextual_role:"secondary_sensitivity",
      inference:"descriptive_paired_episode_macro_only",final_used_for_selection:false},
    promotion:"locked_pending_analysis"}' >"$run_root/campaign.json"

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
  local report="$1" phase="$2" final_seed="${3:-}"
  jq -e --arg phase "$phase" --arg source_fp "$expected_source_fingerprint" --argjson final_seed "${final_seed:-0}" '
    .schema=="p2.semantic_access.fixed_coarse.v2"
    and .population_seed==424244 and .synthetic_episodes_per_source==64
    and .source_population_fingerprint==$source_fp
    and (.population_fingerprint|test("^sha256:[0-9a-f]{64}$"))
    and (.target_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
    and (.predicted_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
    and .protocol.physical_batch==256 and .protocol.feature_width==256
    and .protocol.inputs_per_feature==8
    and .protocol.feature_seeds==[4302768118693889,4302768118693890,4302768118693891]
    and .protocol.seed_aggregation=="arithmetic_mean_predictions_no_seed_selection"
    and .protocol.ridge==0.01
    and .protocol.learned_parameter_count_per_family==14400
    and .protocol.fixed_nonzero_coefficient_count_per_family==6912
    and .protocol.observable_control_interaction_scale==32
    and .protocol.model_weights_frozen==true and .protocol.optimizer_used==false
    and .protocol.inferential_claims_enabled==false
    and .model_weights_updated==false and .final_partition_used_for_decoder_selection==false
    and .model_level_conclusion_permitted==false and .evaluator_status=="qualified"
    and (.families|length)==2 and ([.families[].name]|sort)==["contextual_3x3_global","local"]
    and ([.families[].input_dim]|sort)==[128,384]
    and ([.families[].learned_parameter_count]|all(.==14400))
    and ([.families[].fixed_nonzero_coefficient_count]|all(.==6912))
    and ([.families[].qualification.passed]|all(.==true))
    and ([.families[].qualification.per_seed|length]|all(.==3))
    and ([.families[].route_selection_diagnostics|length]|all(.==2))
    and ([.families[].route_selection_diagnostics[].per_seed|length]|all(.==3))
    and (if $phase=="selection" then
      .execution_phase=="selection_only" and .final_partition_scored==false
      and .descriptive_seam_interpretation_permitted==false and .split.final_frames==0
      and (.fitted_state_sha256|test("^[0-9a-f]{64}$")) and .final_population_seed==null
      and ([.families[].routes|length]|all(.==0))
    else
      .execution_phase=="final_score" and .final_partition_scored==true
      and .descriptive_seam_interpretation_permitted==true and .final_population_seed==$final_seed
      and (.final_population_fingerprint|test("^sha256:[0-9a-f]{64}$"))
      and (.final_target_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
      and (.final_predicted_latents_sha256|test("^sha256:[0-9a-f]{64}$"))
      and ([.families[].routes|length]|all(.==3))
      and ([.families[].routes[].route]|unique|sort)==
        ["predicted_next_refit","target_fit_transfer_to_predicted_next","true_next_encoder_fit"]
    end)' "$report" >/dev/null
}

selection_arm() {
  local arm="$1" replay="$2" arm_dir checkpoint config manifest out_dir report state
  validate_runtime_identity
  arm_dir="$parent_root/seed-1/$arm"
  checkpoint="$arm_dir/checkpoints/step-000000000500/model.safetensors"
  config="$arm_dir/config.json"
  manifest="$arm_dir/train-500.sha256"
  [[ -s "$checkpoint" && -s "$config" && -s "$manifest" ]] || return 1
  sha256sum -c "$manifest" >/dev/null
  if [[ "$replay" == primary ]]; then sha256sum "$checkpoint" "$config" "$manifest" >>"$run_root/INPUTS.sha256"; fi
  out_dir="$run_root/selection_$replay/$arm"; mkdir -p -- "$out_dir"
  report="$out_dir/semantic.json"; state="$out_dir/fitted_state.json"
  timeout --foreground 12h "$tofy_bin" p2-semantic-access-fixed-audit \
    --checkpoint "$checkpoint" --train-config "$config" --device cuda:0 --physical-batch 256 \
    --require-population-fingerprint "$expected_source_fingerprint" --selection-only \
    --fitted-state-output "$state" --output "$report" >"$out_dir/audit.log" 2>&1
  validate_runtime_identity
  validate_report "$report" selection
  [[ "$(jq -r '.checkpoint_sha256' "$report")" == "$(sha256sum "$checkpoint"|awk '{print $1}')" ]] || return 1
  [[ "$(jq -r '.train_config_sha256' "$report")" == "$(sha256sum "$config"|awk '{print $1}')" ]] || return 1
  [[ "$(jq -r '.fitted_state_sha256' "$report")" == "$(sha256sum "$state"|awk '{print $1}')" ]] || return 1
  jq -nc --arg arm "$arm" --arg report "$report" --arg report_sha "$(sha256sum "$report"|awk '{print $1}')" \
    --arg state "$state" --arg state_sha "$(sha256sum "$state"|awk '{print $1}')" \
    --arg checkpoint_sha "$(sha256sum "$checkpoint"|awk '{print $1}')" --arg config_sha "$(sha256sum "$config"|awk '{print $1}')" \
    --arg replay "$replay" \
    '{arm:$arm,replay:$replay,status:"primary_succeeded",selected_report:$report,selected_report_sha256:$report_sha,
      fitted_state:$state,fitted_state_sha256:$state_sha,checkpoint_sha256:$checkpoint_sha,
      train_config_sha256:$config_sha,evaluator_status:"qualified"}' >"$out_dir/receipt.json"
  sha256sum "$checkpoint" "$config" "$report" "$state" "$out_dir/receipt.json" >"$out_dir/SHA256SUMS"
}

write_json_diff() {
  local expected="$1" actual="$2" output="$3"
  jq -n --slurpfile expected "$expected" --slurpfile actual "$actual" '
    [$expected[0] | paths(scalars) as $path
      | select(getpath($path) != ($actual[0] | getpath($path)))
      | {path:$path,expected:getpath($path),actual:($actual[0] | getpath($path))}]' >"$output"
}

for arm in "${all_arms[@]}"; do selection_arm "$arm" primary; done
for arm in "${all_arms[@]}"; do selection_arm "$arm" replay; done
sha256sum -c "$run_root/INPUTS.sha256" >/dev/null
for arm in "${all_arms[@]}"; do
  cmp --silent "$run_root/selection_primary/$arm/semantic.json" "$run_root/selection_replay/$arm/semantic.json" || {
    write_json_diff "$run_root/selection_primary/$arm/semantic.json" "$run_root/selection_replay/$arm/semantic.json" "$run_root/selection_replay/$arm/report_replay_diff.json"
    printf 'selection report replay mismatch: %s\n' "$arm" >&2; exit 1;
  }
  cmp --silent "$run_root/selection_primary/$arm/fitted_state.json" "$run_root/selection_replay/$arm/fitted_state.json" || {
    write_json_diff "$run_root/selection_primary/$arm/fitted_state.json" "$run_root/selection_replay/$arm/fitted_state.json" "$run_root/selection_replay/$arm/state_replay_diff.json"
    printf 'fitted-state replay mismatch: %s\n' "$arm" >&2; exit 1;
  }
done

mapfile -t selection_receipts < <(for replay in primary replay; do for arm in "${all_arms[@]}"; do printf '%s\n' "$run_root/selection_$replay/$arm/receipt.json"; done; done)
jq -s '{schema:"p2.semantic_access.fixed_coarse.selection_seal.v1",
  final_arms:["S0G1","ScurG1"],final_population_seeds:[424245,424246,424247],
  synthetic_episodes_per_source:64,
  arms:(sort_by(.arm)|group_by(.arm)|map(
    (map(select(.replay=="primary"))[0]) as $primary |
    (map(select(.replay=="replay"))[0]) as $replay |
    {key:$primary.arm,value:{selection_report_sha256:$primary.selected_report_sha256,
      fitted_state_sha256:$primary.fitted_state_sha256,
      replay_selection_report_sha256:$replay.selected_report_sha256,
      replay_fitted_state_sha256:$replay.fitted_state_sha256,
      checkpoint_sha256:$primary.checkpoint_sha256,train_config_sha256:$primary.train_config_sha256,
      evaluator_status:$primary.evaluator_status}})|from_entries)}' \
  "${selection_receipts[@]}" >"$run_root/selection_seal.json"
selection_seal_sha="$(sha256sum "$run_root/selection_seal.json"|awk '{print $1}')"
find "$run_root/selection_primary" "$run_root/selection_replay" -type f -print0 | sort -z | while IFS= read -r -d '' file; do sha256sum "$file"; done >"$run_root/SELECTION.sha256"
sha256sum "$run_root/selection_seal.json" >>"$run_root/SELECTION.sha256"
sha256sum -c "$run_root/SELECTION.sha256" >/dev/null

final_arm_seed() {
  local arm="$1" final_seed="$2" arm_dir checkpoint config receipt report_path report_sha state_path state_sha out_dir report marker
  validate_runtime_identity
  sha256sum -c "$run_root/SELECTION.sha256" >/dev/null
  arm_dir="$parent_root/seed-1/$arm"; checkpoint="$arm_dir/checkpoints/step-000000000500/model.safetensors"; config="$arm_dir/config.json"
  receipt="$run_root/selection_primary/$arm/receipt.json"
  report_path="$(jq -er '.selected_report' "$receipt")"; report_sha="$(jq -er '.selected_report_sha256' "$receipt")"
  state_path="$(jq -er '.fitted_state' "$receipt")"; state_sha="$(jq -er '.fitted_state_sha256' "$receipt")"
  [[ "$(sha256sum "$report_path"|awk '{print $1}')" == "$report_sha" ]] || return 1
  [[ "$(sha256sum "$state_path"|awk '{print $1}')" == "$state_sha" ]] || return 1
  out_dir="$run_root/final/seed-$final_seed/$arm"; mkdir -p -- "$out_dir"
  report="$out_dir/semantic.json"; marker="$out_dir/final_access_started.json"
  timeout --foreground 12h "$tofy_bin" p2-semantic-access-fixed-audit \
    --checkpoint "$checkpoint" --train-config "$config" --device cuda:0 --physical-batch 256 \
    --selection-reference "$report_path" --selection-reference-sha256 "$report_sha" \
    --fitted-state-reference "$state_path" --fitted-state-reference-sha256 "$state_sha" \
    --campaign-selection-seal "$run_root/selection_seal.json" \
    --campaign-selection-seal-sha256 "$selection_seal_sha" --arm "$arm" \
    --final-population-seed "$final_seed" --final-access-marker "$marker" \
    --output "$report" >"$out_dir/audit.log" 2>&1
  validate_runtime_identity
  validate_report "$report" final "$final_seed"
  [[ -s "$marker" ]] || return 1
  jq -nc --arg arm "$arm" --argjson seed "$final_seed" --arg report "$report" \
    --arg sha "$(sha256sum "$report"|awk '{print $1}')" \
    '{arm:$arm,final_population_seed:$seed,status:"primary_succeeded",selected_report:$report,selected_report_sha256:$sha}' \
    >"$out_dir/receipt.json"
  sha256sum "$checkpoint" "$config" "$report" "$marker" "$out_dir/receipt.json" >"$out_dir/SHA256SUMS"
}

for final_seed in "${final_seeds[@]}"; do
  for arm in "${final_arms[@]}"; do final_arm_seed "$arm" "$final_seed"; done
done
mapfile -t final_reports < <(for final_seed in "${final_seeds[@]}"; do for arm in "${final_arms[@]}"; do jq -er '.selected_report' "$run_root/final/seed-$final_seed/$arm/receipt.json"; done; done)
jq -s '{schema:"p2.semantic_access.fixed_coarse_summary.v2",evaluator_status:"qualified",
  decision:"paired_fresh_population_transfer_matrix_ready_for_analysis",
  arms:map({checkpoint,final_population_seed,final_population_fingerprint,evaluator_status,families})}' \
  "${final_reports[@]}" >"$run_root/summary.json"
jq --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson elapsed "$(( $(date +%s)-started_epoch ))" \
  '.status="complete_pending_analysis"|.finished_utc=$finished|.elapsed_seconds=$elapsed
    |.decision="paired_fresh_population_transfer_matrix_ready_for_analysis"' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
stop_telemetry
write_root_manifest
finalized=true
printf 'fixed coarse semantic campaign complete; root=%s\n' "$run_root"
