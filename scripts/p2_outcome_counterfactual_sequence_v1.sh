#!/usr/bin/env bash
# Sequential, preregistered frozen-checkpoint outcome-counterfactual evaluation.
#
# Runtime estimate basis: 815 seconds per full evidence cell, measured from the
# exact prior source runs sigreg-cell-dose-response-v1-20260815T071547Z seed-2
# and seed-3 evaluation cells. The registered target is 25,200--28,800 seconds;
# the hard sequence maximum is 32,400 seconds. No training is launched here.
set -euo pipefail

if [[ "${1:-}" == "--internal-gpu-telemetry" ]]; then
  shift
  (($# == 2)) || exit 2
  telemetry_output="$1"
  telemetry_interval="$2"
  trap 'exit 0' TERM INT
  while true; do
    {
      printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    } >>"$telemetry_output"
    sleep "$telemetry_interval" &
    delay_pid=$!
    wait "$delay_pid" || true
  done
fi

if [[ "${1:-}" != "--internal-hard-runtime" ]]; then
  exec timeout --signal=TERM --kill-after=60s 32400s \
    bash "${BASH_SOURCE[0]}" --internal-hard-runtime "$@"
fi
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
eval_validator="$script_dir/p2_validate_eval.sh"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
source_run="${P2_OUTCOME_COUNTERFACTUAL_SOURCE_RUN:?set the sealed source run root}"
sequence_root="${P2_OUTCOME_COUNTERFACTUAL_SEQUENCE_ROOT:?set a never-used sequence root}"
binary_build_command="${P2_BINARY_BUILD_COMMAND:?set the exact binary build command}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
smoke_seed="${P2_OUTCOME_COUNTERFACTUAL_SMOKE_SEED:-424254}"
dry_run="${P2_OUTCOME_COUNTERFACTUAL_DRY_RUN:-0}"
test_fail_before_d="${P2_OUTCOME_COUNTERFACTUAL_TEST_FAIL_BEFORE_D:-0}"

: "${P2_EXPECTED_LAUNCHER_SHA:?set the reviewed launcher commit}"
: "${P2_EXPECTED_SOURCE_SHA:?set the source campaign commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SOURCE_SHA:?set the Tofy commit used to build the binary}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"

readonly source_run_id="sigreg-cell-dose-response-v1-20260815T071547Z"
readonly evaluation_count=32
readonly physical_batch=256
readonly synthetic_episodes=64
readonly baseline_cell_seconds=815
readonly target_runtime_low_seconds=25200
readonly target_runtime_high_seconds=28800
readonly hard_max_seconds=32400
readonly runtime_reserve_seconds=900
readonly per_cell_hard_guard_seconds=2100
readonly material_matching_advantage=0.10
readonly group_bootstrap_resamples=10000

[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || { printf 'invalid dry-run value\n' >&2; exit 2; }
[[ "$test_fail_before_d" == 0 || "$test_fail_before_d" == 1 ]] || exit 2
[[ "$gpu_interval" =~ ^[1-9][0-9]*$ && "$smoke_seed" =~ ^[0-9]+$ ]] || exit 2
[[ "$smoke_seed" != 424250 && "$smoke_seed" != 424251 \
  && "$smoke_seed" != 424252 && "$smoke_seed" != 424253 ]] || {
  printf 'smoke seed overlaps an evidence evaluator seed\n' >&2
  exit 2
}
[[ "$binary_build_command" == "cargo build --release --locked --features cudnn" ]] || {
  printf 'unexpected binary build command\n' >&2
  exit 2
}

for command in awk bash cargo date find git install jq mkdir mv pgrep realpath setsid sha256sum sleep sort timeout wc xargs; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done

sequence_parent="$(dirname -- "$sequence_root")"
mkdir -p -- "$sequence_parent"
sequence_parent="$(realpath "$sequence_parent")"
sequence_root="$sequence_parent/$(basename -- "$sequence_root")"
experiment_a_root="$sequence_root/experiment-A-seed-424250"
experiment_b_root="$sequence_root/experiment-B-seed-424251"
experiment_c_root="$sequence_root/experiment-C-seed-424252"
experiment_d_root="$sequence_root/experiment-D-seed-424253"
smoke_root="$sequence_root/implementation-smoke-seed-$smoke_seed"

roots=("$sequence_root" "$smoke_root" "$experiment_a_root" "$experiment_b_root" \
  "$experiment_c_root" "$experiment_d_root")
[[ "$(printf '%s\n' "${roots[@]}" | sort -u | wc -l)" == "${#roots[@]}" ]] || {
  printf 'sequence and experiment roots must be unique\n' >&2
  exit 2
}
for root in "${roots[@]}"; do
  [[ ! -e "$root" && ! -e "${root}.ROOT_SHA256SUMS.sha256" ]] || {
    printf 'root or external seal already exists: %s\n' "$root" >&2
    exit 2
  }
done
[[ ! -e "${sequence_root}.PREREGISTRATION.sha256" ]] || exit 2

build_cells() {
  local experiment eval_seed training_seed arm update ordinal=0
  local -a ascending=(S0 w004 w008 w016 w0323)
  local -a descending=(w0323 w016 w008 w004 S0)
  for experiment in A B; do
    if [[ "$experiment" == A ]]; then eval_seed=424250; else eval_seed=424251; fi
    for arm in "${ascending[@]}"; do
      ordinal=$((ordinal + 1))
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ordinal" "$experiment" "$eval_seed" 2 "$arm" 250
    done
    for arm in "${descending[@]}"; do
      ordinal=$((ordinal + 1))
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ordinal" "$experiment" "$eval_seed" 3 "$arm" 250
    done
  done
  eval_seed=424252
  for arm in S0 w0323; do
    for update in 125 250; do
      ordinal=$((ordinal + 1))
      printf '%s\tC\t%s\t2\t%s\t%s\n' "$ordinal" "$eval_seed" "$arm" "$update"
    done
  done
  for arm in w0323 S0; do
    for update in 125 250; do
      ordinal=$((ordinal + 1))
      printf '%s\tC\t%s\t3\t%s\t%s\n' "$ordinal" "$eval_seed" "$arm" "$update"
    done
  done
  eval_seed=424253
  for arm in S0 w0323; do
    ordinal=$((ordinal + 1))
    printf '%s\tD\t%s\t2\t%s\t250\n' "$ordinal" "$eval_seed" "$arm"
  done
  for arm in w0323 S0; do
    ordinal=$((ordinal + 1))
    printf '%s\tD\t%s\t3\t%s\t250\n' "$ordinal" "$eval_seed" "$arm"
  done
}

mkdir -- "$sequence_root"
started_epoch="$(date +%s)"
started_ns="$(date +%s%N)"
sequence_finalized=false
current_experiment_root=""
current_experiment_finalized=true
active_pid=""
active_pgid=""
telemetry_pid=""
telemetry_pgid=""
binary_sha=""
preregistration_digest_record="${sequence_root}.PREREGISTRATION.sha256"

seal_root() {
  local root="$1"
  [[ -d "$root" && ! -e "$root/ROOT_SHA256SUMS" ]] || return 1
  (
    cd "$root"
    find . -type f ! -name ROOT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
      >ROOT_SHA256SUMS
    sha256sum --quiet -c ROOT_SHA256SUMS
  )
  sha256sum "$root/ROOT_SHA256SUMS" >"${root}.ROOT_SHA256SUMS.sha256"
  sha256sum --quiet -c "${root}.ROOT_SHA256SUMS.sha256"
}

set_json_status() {
  local file="$1" status="$2" exit_code="${3:-0}"
  [[ -s "$file" ]] || return 0
  jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson finished_ns "$(date +%s%N)" --argjson exit_code "$exit_code" \
    '.status=$status | .finished_utc=$finished_utc | .finished_epoch_ns=$finished_ns
      | .exit_code=$exit_code
      | if $status=="failed_integrity_or_evaluation" then
          .evidence_class="failed_infrastructure_or_integrity"
        elif $status=="complete_pending_analysis"
          and .evidence_class!="implementation_smoke" then
          .evidence_class="completed_evidence"
        else . end' "$file" >"${file}.tmp"
  mv -- "${file}.tmp" "$file"
}

group_is_alive() {
  local pgid="$1"
  [[ -n "$pgid" ]] && pgrep -g "$pgid" >/dev/null 2>&1
}

stop_process_group() {
  local pid="$1" pgid="$2" label="$3"
  [[ -n "$pgid" ]] || return 0
  if group_is_alive "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..50}; do
      group_is_alive "$pgid" || break
      sleep 0.1
    done
  fi
  if group_is_alive "$pgid"; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
    for _ in {1..50}; do
      group_is_alive "$pgid" || break
      sleep 0.1
    done
  fi
  [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  if group_is_alive "$pgid"; then
    printf '%s process group survived cleanup: %s\n' "$label" "$pgid" >&2
    return 1
  fi
}

stop_telemetry() {
  local rc=0
  if [[ -n "$telemetry_pgid" ]]; then
    stop_process_group "$telemetry_pid" "$telemetry_pgid" telemetry || rc=$?
  fi
  telemetry_pid=""
  telemetry_pgid=""
  return "$rc"
}

assert_binary_unchanged() {
  [[ "$dry_run" == 1 ]] && return 0
  [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$binary_sha" ]] || {
    printf 'launch binary changed during sequence\n' >&2
    return 1
  }
}

assert_preregistration_unchanged() {
  sha256sum --quiet -c "$preregistration_digest_record"
}

assert_source_ledger_unchanged() {
  [[ "$(sha256sum "$source_artifact_ledger" | awk '{print $1}')" == \
    "$source_artifact_ledger_sha" ]]
}

assert_source_manifest_unchanged() {
  [[ "$dry_run" == 1 ]] && return 0
  [[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$source_manifest_sha" ]]
}

cleanup() {
  local rc="$?" cleanup_rc=0
  trap - EXIT INT TERM
  if [[ -n "$active_pgid" ]]; then
    stop_process_group "$active_pid" "$active_pgid" evaluation || cleanup_rc=$?
  fi
  active_pid=""; active_pgid=""
  stop_telemetry || cleanup_rc=$?
  (( cleanup_rc == 0 )) || rc=125
  if [[ "$current_experiment_finalized" != true && -n "$current_experiment_root" \
    && -d "$current_experiment_root" ]]; then
    set_json_status "$current_experiment_root/campaign.json" failed_integrity_or_evaluation "$rc" || true
    seal_root "$current_experiment_root" || true
    current_experiment_finalized=true
  fi
  if [[ "$sequence_finalized" != true && -d "$sequence_root" ]]; then
    set_json_status "$sequence_root/campaign.json" failed_integrity_or_evaluation "$rc" || true
    if [[ ! -e "$sequence_root/ROOT_SHA256SUMS" ]]; then seal_root "$sequence_root" || true; fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson started_ns "$started_ns" \
  '{schema:"p2.outcome_counterfactual_sequence.v1",status:"running",
    evidence_class:"pending_evidence",training:false,
    lifecycle_note:"preflight_before_preregistration",started_utc:$started_utc,
    started_epoch_ns:$started_ns}' >"$sequence_root/campaign.json"

cells_tsv="$sequence_root/preregistered-cells.tsv"
build_cells >"$cells_tsv"
[[ "$(wc -l <"$cells_tsv")" == "$evaluation_count" ]] || exit 1

source_artifact_ledger="$sequence_root/source-artifacts.tsv"
if [[ "$dry_run" == 0 ]]; then
  [[ "$(basename -- "$source_run")" == "$source_run_id" ]] || {
    printf 'unexpected source run id\n' >&2; exit 2;
  }
  source_run="$(realpath "$source_run")"
  source_manifest="$source_run/root-sha256.txt"
  [[ -s "$source_manifest" && -s "$source_run/campaign.json" ]] || {
    printf 'source internal seal is missing\n' >&2; exit 2;
  }
  (cd "$source_run" && sha256sum --quiet -c root-sha256.txt)
  source_manifest_sha="$(sha256sum "$source_manifest" | awk '{print $1}')"
  jq -e --arg source_sha "$P2_EXPECTED_SOURCE_SHA" '
    .schema=="p2.sigreg_cell_dose_response.v1"
    and (.status=="complete" or .status=="complete_pending_analysis")
    and .git_sha==$source_sha' "$source_run/campaign.json" >/dev/null
else
  source_manifest_sha="dry-run-source-manifest-sha"
fi

while IFS=$'\t' read -r ordinal experiment eval_seed training_seed arm update; do
  printf -v checkpoint '%s/seed-%s/%s/checkpoints/step-%012d/model.safetensors' \
    "$source_run" "$training_seed" "$arm" "$update"
  config="$source_run/seed-$training_seed/$arm/config.json"
  if [[ "$dry_run" == 0 ]]; then
    [[ -s "$checkpoint" && -s "$config" ]] || exit 2
    checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
    config_sha="$(sha256sum "$config" | awk '{print $1}')"
  else
    checkpoint_sha="dry-checkpoint-$training_seed-$arm-$update"
    config_sha="dry-config-$training_seed-$arm"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ordinal" "$experiment" "$eval_seed" \
    "$training_seed" "$arm" "$update" "$checkpoint_sha" "$config_sha"
done <"$cells_tsv" >"$source_artifact_ledger"
source_artifact_ledger_sha="$(sha256sum "$source_artifact_ledger" | awk '{print $1}')"

jq -Rn \
  --arg source_run_id "$source_run_id" \
  --arg registered_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson material "$material_matching_advantage" \
  --argjson resamples "$group_bootstrap_resamples" \
  --arg source_manifest_sha "$source_manifest_sha" \
  --arg source_artifact_ledger_sha "$source_artifact_ledger_sha" \
  '[inputs | split("\t") |
    {ordinal:(.[0]|tonumber),experiment:.[1],evaluation_seed:(.[2]|tonumber),
     training_seed:(.[3]|tonumber),arm:.[4],update:(.[5]|tonumber)}] |
   {schema:"p2.outcome_counterfactual_sequence_preregistration.v1",
    registered_utc:$registered_utc,observed_sequence_results_before_registration:false,
    source_campaign_preexisted:true,
    source_run_id:$source_run_id,training:false,cells:.,cell_count:length,
    source_manifest_sha256:$source_manifest_sha,
    source_artifact_ledger_sha256:$source_artifact_ledger_sha,
    material_matching_advantage_threshold:$material,
    controls:{required_all_true:true,
      registered:["pixel_oracle_exactly_one","latent_oracle_at_least_0_99",
        "target_collapse_failure_false","swapped_oracle_at_most_negative_0_99",
        "action_masked_max_abs_at_most_1e_6","identity_max_abs_at_most_1e_6",
        "outcome_equivalent_max_abs_at_most_1e_6","pair_ledger_reconciled"],
      state_scrambled_same_action_template:"diagnostic_only"},
    population_gates:{required_all_true_for_evidence_cells:true,
      registered:["eligible_simulator_groups_at_least_100",
        "each_movement_action_at_least_16_changed_and_16_unchanged",
        "simulator_changed_changed_pairs_at_least_100","target_collapse_failure_false"]},
    primary_estimand:"exact-simulator movement-group matching advantage",
    primary_rule:"movement estimate > 0.10 and movement lower 95% bound > 0.10",
    coordinate_groups_role:"separate synthetic diagnostic; excluded from primary claim",
    bootstrap:{unit:"whole_branch_group",group_resamples:$resamples},
    per_cell_intervals:[{coverage_percent:95},{coverage_percent:98.75}],
    outcome_classes:["simulator_outcome_equivalent","simulator_outcome_changing"],
    experiment_roles:{A:"complete five-arm panel, evaluator seed 424250",
      B:"complete five-arm replication, evaluator seed 424251",
      C:"endpoint temporal panel at updates 125 and 250, evaluator seed 424252",
      D:"fixed mature endpoints, evaluator seed 424253; only after A/B/C seal and pass"},
    multiplicity_policy:"report both per-cell 95% and 98.75% group-bootstrap intervals; no cell, metric, seed, checkpoint, or outcome-class selection after observation",
    stop_rule:"run exactly 32 registered frozen-checkpoint evaluations sequentially unless provenance, runtime, integrity, evaluation, or sealing fails",
    handoff_rule:"D may start only after A, B, and C have complete_pending_analysis status and verified internal and external seals",
    no_training:true}' "$cells_tsv" >"$sequence_root/preregistration.json"
preregistration_sha="$(sha256sum "$sequence_root/preregistration.json" | awk '{print $1}')"
sha256sum "$sequence_root/preregistration.json" >"$preregistration_digest_record"
assert_preregistration_unchanged
assert_source_ledger_unchanged
assert_source_manifest_unchanged

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson started_ns "$started_ns" --arg source_run "$source_run" \
  --arg source_run_id "$source_run_id" --arg preregistration_sha "$preregistration_sha" \
  --arg a "$experiment_a_root" --arg b "$experiment_b_root" --arg c "$experiment_c_root" \
  --arg d "$experiment_d_root" --arg smoke "$smoke_root" \
  --arg launcher_sha "$P2_EXPECTED_LAUNCHER_SHA" --arg source_sha "$P2_EXPECTED_SOURCE_SHA" \
  --arg candle_sha "$P2_EXPECTED_CANDLE_SHA" \
  --arg binary_source_sha "$P2_EXPECTED_BINARY_SOURCE_SHA" \
  --arg binary_sha "$P2_EXPECTED_BINARY_SHA" --arg build "$binary_build_command" \
  --argjson dry_run "$dry_run" --argjson cells "$evaluation_count" \
  --argjson baseline "$baseline_cell_seconds" --argjson target_low "$target_runtime_low_seconds" \
  --argjson target_high "$target_runtime_high_seconds" --argjson hard_max "$hard_max_seconds" \
  --argjson reserve "$runtime_reserve_seconds" \
  '{schema:"p2.outcome_counterfactual_sequence.v1",status:"running",
    started_utc:$started_utc,started_epoch_ns:$started_ns,source_run:$source_run,
    source_run_id:$source_run_id,launcher_git_sha:$launcher_sha,source_git_sha:$source_sha,
    candle_git_sha:$candle_sha,binary_source_git_sha:$binary_source_sha,
    binary_sha256:$binary_sha,binary_build_command:$build,binary_features:["cudnn"],
    preregistration_sha256:$preregistration_sha,evaluator_schema:"p2.eval_report.v14",
    evidence_class:"pending_evidence",training:false,evaluation_count:$cells,
    physical_evaluation_batch:256,synthetic_episodes:64,ptrm_k:[1],ptrm_noise:0,
    ensemble_members:1,cell_timeout_seconds:1800,
    experiment_roots:{A:$a,B:$b,C:$c,D:$d},smoke_root:$smoke,
    runtime:{baseline_seconds_per_cell:$baseline,
      baseline_source_run_ids:["sigreg-cell-dose-response-v1-20260815T071547Z/seed-2",
        "sigreg-cell-dose-response-v1-20260815T071547Z/seed-3"],
      target_low_seconds:$target_low,target_high_seconds:$target_high,
      hard_max_seconds:$hard_max,reserve_seconds:$reserve,artificial_padding:false},
    dry_run:($dry_run==1)}' >"$sequence_root/campaign.json"
: >"$sequence_root/runtime-estimates.jsonl"

if [[ "$dry_run" == 0 ]]; then
  [[ -r "$eval_validator" && -d "$source_run" ]] || exit 2
  [[ -d "$candle_root" ]] || exit 2
  git_sha="$(git -C "$repo_root" rev-parse HEAD)"
  candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
  [[ "$git_sha" == "$P2_EXPECTED_LAUNCHER_SHA" ]] || { printf 'launcher SHA mismatch\n' >&2; exit 2; }
  [[ "$git_sha" == "$P2_EXPECTED_BINARY_SOURCE_SHA" ]] || {
    printf 'binary source SHA must equal the checked-out evaluator/launcher commit\n' >&2
    exit 2
  }
  [[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle SHA mismatch\n' >&2; exit 2; }
  [[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
  [[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle checkout\n' >&2; exit 2; }

  git -C "$repo_root" fetch origin main
  [[ "$(git -C "$repo_root" rev-parse refs/remotes/origin/main)" == "$P2_EXPECTED_LAUNCHER_SHA" ]] || {
    printf 'origin/main is not the exact launcher commit\n' >&2; exit 2;
  }
  git -C "$repo_root" cat-file -e "${P2_EXPECTED_SOURCE_SHA}^{commit}"
  git -C "$repo_root" cat-file -e "${P2_EXPECTED_BINARY_SOURCE_SHA}^{commit}"
  git -C "$candle_root" fetch origin main
  [[ "$(git -C "$candle_root" rev-parse refs/remotes/origin/main)" == "$P2_EXPECTED_CANDLE_SHA" ]] || {
    printf 'candle origin/main is not the exact expected commit\n' >&2; exit 2;
  }

  (cd "$repo_root" && cargo build --release --locked --features cudnn)
  built_binary="$repo_root/target/release/tofy"
  [[ -x "$built_binary" ]] || { printf 'exact build produced no executable\n' >&2; exit 2; }
  built_binary_sha="$(sha256sum "$built_binary" | awk '{print $1}')"
  [[ "$built_binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || {
    printf 'fresh exact-build binary SHA mismatch\n' >&2; exit 2;
  }
  [[ "$tofy_bin" != "$built_binary" && ! -e "$tofy_bin" ]] || {
    printf 'TOFY_BIN must be a new dedicated path distinct from target/release/tofy\n' >&2
    exit 2
  }
  mkdir -p -- "$(dirname -- "$tofy_bin")"
  install -m 0755 -- "$built_binary" "$tofy_bin"
  binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
  [[ "$binary_sha" == "$built_binary_sha" ]] || exit 2

  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  (( ${#gpu_names[@]} == 1 )) || { printf 'requires one visible GPU\n' >&2; exit 2; }
  [[ "${gpu_names[0]}" == "NVIDIA A40" ]] || { printf 'requires NVIDIA A40\n' >&2; exit 2; }
  read -r gpu_memory gpu_utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
  )
  (( gpu_memory <= 1024 && gpu_utilization == 0 )) || {
    printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' "$gpu_memory" "$gpu_utilization" >&2
    exit 2
  }
  jq --arg source_manifest_sha "$source_manifest_sha" --arg gpu "${gpu_names[0]}" \
    '.source_root_manifest_sha256=$source_manifest_sha | .gpu_name=$gpu' \
    "$sequence_root/campaign.json" >"$sequence_root/campaign.json.tmp"
  mv -- "$sequence_root/campaign.json.tmp" "$sequence_root/campaign.json"
else
  binary_sha="$P2_EXPECTED_BINARY_SHA"
fi

outcome_object_filter='.outcome_counterfactuals'

validate_outcome_report() {
  local report="$1" require_population="$2"
  jq -e --argjson require_population "$require_population" '
    def finite: type=="number" and .>-1e300 and .<1e300;
    def interval: (.estimate|finite)
      and (.lower_95|finite) and (.upper_95|finite)
      and (.lower_98_75|finite) and (.upper_98_75|finite)
      and .lower_98_75<=.lower_95 and .lower_95<=.estimate
      and .estimate<=.upper_95 and .upper_95<=.upper_98_75
      and .groups>0 and .pairs>0 and .resamples==10000
      and .unit=="whole_branch_group";
    .schema=="p2.eval_report.v14"
    and (.outcome_counterfactuals as $o | ($o|type)=="object"
      and $o.ledger_reconciled==true
      and ($o.pair_ledger|type)=="array" and ($o.pair_ledger|length)>0
      and $o.unordered_pairs==($o.pair_ledger|length)
      and $o.eligible_pairs+$o.outcome_equivalent_pairs==$o.unordered_pairs
      and ($o.movement|interval)
      and $o.action_separation_pass
        ==($o.movement.estimate>0.10 and $o.movement.lower_95>0.10)
      and $o.controls.pixel_oracle_exactly_one==true
      and $o.controls.latent_oracle_at_least_0_99==true
      and $o.controls.target_collapse_failure==false
      and $o.controls.swapped_oracle_at_most_negative_0_99==true
      and $o.controls.action_masked_max_abs_at_most_1e_6==true
      and $o.controls.identity_max_abs_at_most_1e_6==true
      and $o.controls.outcome_equivalent_max_abs_at_most_1e_6==true
      and $o.controls.required_controls_pass==true
      and (if $require_population then
        ($o.overall|interval) and ($o.coordinate|interval)
        and ($o.changed_changed|interval) and ($o.changed_unchanged|interval)
        and $o.population_gates.eligible_simulator_groups_at_least_100==true
        and $o.population_gates.each_movement_action_at_least_16_changed_and_16_unchanged==true
        and $o.population_gates.simulator_changed_changed_pairs_at_least_100==true
        and $o.population_gates.target_collapse_failure==false
        and $o.population_gates.population_pass==true
        and $o.population_gates.simulator_changed_changed_pairs>0
      else true end))' "$report" >/dev/null
}

extract_pair_ledger() {
  local report="$1" output="$2"
  jq -c "$outcome_object_filter | .pair_ledger[]" "$report" >"$output"
  local expected
  expected="$(jq "$outcome_object_filter | .unordered_pairs" "$report")"
  [[ "$(wc -l <"$output")" == "$expected" && "$expected" -gt 0 ]]
}

outcome_fingerprint() {
  jq -er "$outcome_object_filter | .population_fingerprint
    | select(type==\"string\" and test(\"^sha256:[0-9a-f]{64}$\"))" "$1"
}

run_tracked() {
  local log_path="$1"; shift
  local rc
  assert_binary_unchanged
  assert_preregistration_unchanged
  assert_source_ledger_unchanged
  assert_source_manifest_unchanged
  setsid "$@" >"$log_path" 2>&1 &
  active_pid=$!
  active_pgid="$active_pid"
  printf '%s\n' "$active_pid" >"${log_path}.pid"
  if wait "$active_pid"; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" >"${log_path}.exit-code"
  if group_is_alive "$active_pgid"; then
    printf 'evaluation process group survived wait: %s\n' "$active_pgid" >&2
    return 125
  fi
  active_pid=""; active_pgid=""
  assert_binary_unchanged
  assert_preregistration_unchanged
  assert_source_ledger_unchanged
  assert_source_manifest_unchanged
  return "$rc"
}

start_telemetry() {
  local root="$1"
  [[ "$dry_run" == 1 ]] && return 0
  setsid bash "$script_dir/$(basename -- "${BASH_SOURCE[0]}")" \
    --internal-gpu-telemetry "$root/gpu.csv" "$gpu_interval" &
  telemetry_pid=$!
  telemetry_pgid="$telemetry_pid"
  printf '%s\n' "$telemetry_pid" >"$root/telemetry.pid"
  kill -0 "$telemetry_pid"
}

write_dry_report() {
  local report="$1" fingerprint="$2"
  fingerprint="sha256:$(printf '%s' "$fingerprint" | sha256sum | awk '{print $1}')"
  jq -nc --arg fingerprint "$fingerprint" '
    def interval: {estimate:0.05,lower_95:0.01,upper_95:0.09,
      lower_98_75:0.0,upper_98_75:0.10,groups:1,pairs:1,
      resamples:10000,unit:"whole_branch_group"};
    {schema:"p2.eval_report.v14",mode:"full",seed:0,outcome_counterfactuals:{
      population_fingerprint:$fingerprint,unordered_pairs:1,eligible_pairs:1,
      outcome_equivalent_pairs:0,overall:interval,movement:interval,
      coordinate:interval,changed_changed:interval,changed_unchanged:interval,
      action_separation_pass:false,ledger_reconciled:true,
      controls:{pixel_oracle_exactly_one:true,latent_oracle_at_least_0_99:true,
        target_collapse_failure:false,swapped_oracle_at_most_negative_0_99:true,
        action_masked_max_abs_at_most_1e_6:true,identity_max_abs_at_most_1e_6:true,
        outcome_equivalent_max_abs_at_most_1e_6:true,required_controls_pass:true},
      population_gates:{eligible_simulator_groups_at_least_100:true,
        each_movement_action_at_least_16_changed_and_16_unchanged:true,
        simulator_changed_changed_pairs:1,
        simulator_changed_changed_pairs_at_least_100:true,
        target_collapse_failure:false,population_pass:true},
      pair_ledger:[{dry_run:true}]}}' >"$report"
}

run_smoke() {
  current_experiment_root="$smoke_root"
  current_experiment_finalized=false
  mkdir -- "$smoke_root"
  local smoke_started_ns smoke_finished_ns checkpoint config checkpoint_sha config_sha
  smoke_started_ns="$(date +%s%N)"
  checkpoint="$source_run/seed-2/S0/checkpoints/step-000000000250/model.safetensors"
  config="$source_run/seed-2/S0/config.json"
  if [[ "$dry_run" == 0 ]]; then
    [[ -s "$checkpoint" && -s "$config" ]] || return 1
    checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
    config_sha="$(sha256sum "$config" | awk '{print $1}')"
    read -r registered_checkpoint_sha registered_config_sha < <(
      awk -F '\t' '$1==1 {print $7, $8}' "$source_artifact_ledger"
    )
    [[ "$checkpoint_sha" == "$registered_checkpoint_sha"
      && "$config_sha" == "$registered_config_sha" ]] || {
      printf 'smoke source artifact changed after preregistration\n' >&2
      return 1
    }
  else
    checkpoint_sha="dry-run-checkpoint-sha"; config_sha="dry-run-config-sha"
  fi
  jq -nc --argjson seed "$smoke_seed" --arg checkpoint "$checkpoint" --arg config "$config" \
    --arg checkpoint_sha "$checkpoint_sha" --arg config_sha "$config_sha" \
    --arg binary_sha "$binary_sha" --argjson started_ns "$smoke_started_ns" \
    '{schema:"p2.outcome_counterfactual_smoke.v1",status:"running",
      evidence_class:"implementation_smoke",seed:$seed,synthetic_episodes:2,
      checkpoint:$checkpoint,config:$config,checkpoint_sha256:$checkpoint_sha,
      config_sha256:$config_sha,binary_sha256:$binary_sha,physical_evaluation_batch:256,
      ptrm_k:[1],ptrm_noise:0,ensemble_members:1,training:false,started_epoch_ns:$started_ns}' \
    >"$smoke_root/campaign.json"
  : >"$smoke_root/stages.jsonl"
  start_telemetry "$smoke_root"
  if [[ "$dry_run" == 1 ]]; then
    write_dry_report "$smoke_root/eval_report.json" "content:smoke-$smoke_seed"
    printf '{"schema":"p2.episode_rollout.v2","dry_run":true}\n' >"$smoke_root/episodes.jsonl"
  else
    run_tracked "$smoke_root/eval.log" timeout --signal=TERM --kill-after=60s 30m \
      "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" \
      --device cuda:0 --seed "$smoke_seed" --synthetic-episodes 2 \
      --physical-batch "$physical_batch" --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
      --episode-jsonl "$smoke_root/episodes.jsonl" --output "$smoke_root/eval_report.json"
    bash "$eval_validator" "$smoke_root/eval_report.json" "$smoke_root/episodes.jsonl" \
      "$smoke_seed" false
  fi
  validate_outcome_report "$smoke_root/eval_report.json" false
  extract_pair_ledger "$smoke_root/eval_report.json" "$smoke_root/outcome-pairs.jsonl"
  stop_telemetry
  smoke_finished_ns="$(date +%s%N)"
  jq -nc --arg stage smoke --argjson started_ns "$smoke_started_ns" \
    --argjson finished_ns "$smoke_finished_ns" \
    '{stage:$stage,status:"passed",started_epoch_ns:$started_ns,
      finished_epoch_ns:$finished_ns,duration_milliseconds:(($finished_ns-$started_ns)/1000000|floor),
      evaluator_schema_v14:true,algebraic_controls:true,ledger_reconciled:true,
      full_population_gates_required:false,training:false}' >>"$smoke_root/stages.jsonl"
  set_json_status "$smoke_root/campaign.json" complete_pending_analysis 0
  seal_root "$smoke_root"
  current_experiment_finalized=true
}

completed_cells=0
completed_duration_ms=0

record_runtime_prediction() {
  local ordinal="$1" experiment="$2" elapsed rate remaining predicted
  elapsed=$(( $(date +%s) - started_epoch ))
  if (( completed_cells > 0 )); then
    rate=$(( (completed_duration_ms + completed_cells * 1000 - 1) / (completed_cells * 1000) ))
    (( rate >= baseline_cell_seconds )) || rate="$baseline_cell_seconds"
  else
    rate="$baseline_cell_seconds"
  fi
  remaining=$(( evaluation_count - completed_cells ))
  predicted=$(( elapsed + remaining * rate + runtime_reserve_seconds ))
  jq -nc --argjson ordinal "$ordinal" --arg experiment "$experiment" \
    --arg at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson completed "$completed_cells" \
    --argjson elapsed "$elapsed" --argjson rate "$rate" --argjson remaining "$remaining" \
    --argjson predicted "$predicted" --argjson hard "$hard_max_seconds" \
    '{before_cell_ordinal:$ordinal,experiment:$experiment,at_utc:$at_utc,
      completed_evidence_cells:$completed,elapsed_seconds:$elapsed,
      conservative_seconds_per_remaining_cell:$rate,remaining_cells_including_next:$remaining,
      reserve_seconds:900,predicted_total_seconds:$predicted,hard_max_seconds:$hard,
      permitted:($predicted<=$hard)}' >>"$sequence_root/runtime-estimates.jsonl"
  (( predicted <= hard_max_seconds )) || {
    printf 'runtime prediction exceeds hard maximum before cell %s: %s > %s\n' \
      "$ordinal" "$predicted" "$hard_max_seconds" >&2
    return 1
  }
  (( elapsed + per_cell_hard_guard_seconds <= hard_max_seconds )) || {
    printf 'insufficient hard-runtime budget for one bounded cell plus validation\n' >&2
    return 1
  }
}

experiment_root_for() {
  case "$1" in
    A) printf '%s\n' "$experiment_a_root" ;;
    B) printf '%s\n' "$experiment_b_root" ;;
    C) printf '%s\n' "$experiment_c_root" ;;
    D) printf '%s\n' "$experiment_d_root" ;;
    *) return 2 ;;
  esac
}

run_experiment() {
  local experiment="$1" eval_seed="$2" expected_cells="$3"
  local root ordinal ignored seed arm update checkpoint config output
  local checkpoint_sha config_sha cell_started_ns cell_finished_ns duration_ms fingerprint
  root="$(experiment_root_for "$experiment")"
  current_experiment_root="$root"
  current_experiment_finalized=false
  mkdir -- "$root"
  jq -nc --arg experiment "$experiment" --argjson eval_seed "$eval_seed" \
    --arg source_run "$source_run" --arg preregistration_sha "$preregistration_sha" \
    --arg binary_sha "$binary_sha" --argjson expected_cells "$expected_cells" \
    --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson started_ns "$(date +%s%N)" \
    '{schema:"p2.outcome_counterfactual_experiment.v1",status:"running",
      experiment:$experiment,evaluation_seed:$eval_seed,source_run:$source_run,
      evidence_class:"pending_evidence",
      preregistration_sha256:$preregistration_sha,binary_sha256:$binary_sha,
      evaluator_schema:"p2.eval_report.v14",expected_cells:$expected_cells,
      physical_evaluation_batch:256,synthetic_episodes:64,ptrm_k:[1],ptrm_noise:0,
      ensemble_members:1,cell_timeout_seconds:1800,training:false,
      started_utc:$started_utc,started_epoch_ns:$started_ns}' >"$root/campaign.json"
  : >"$root/stages.jsonl"; : >"$root/metrics.jsonl"
  start_telemetry "$root"

  while IFS=$'\t' read -r ordinal ignored ignored seed arm update; do
    [[ "$ignored" == "$eval_seed" ]] || return 1
    record_runtime_prediction "$ordinal" "$experiment"
    output="$root/cell-$(printf '%02d' "$ordinal")-seed-$seed-$arm-update-$update"
    mkdir -- "$output"
    checkpoint="$source_run/seed-$seed/$arm/checkpoints/step-$(printf '%012d' "$update")/model.safetensors"
    config="$source_run/seed-$seed/$arm/config.json"
    if [[ "$dry_run" == 0 ]]; then
      [[ -s "$checkpoint" && -s "$config" ]] || return 1
      checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
      config_sha="$(sha256sum "$config" | awk '{print $1}')"
      read -r registered_checkpoint_sha registered_config_sha < <(
        awk -F '\t' -v ordinal="$ordinal" '$1==ordinal {print $7, $8}' "$source_artifact_ledger"
      )
      [[ -n "$registered_checkpoint_sha" && "$checkpoint_sha" == "$registered_checkpoint_sha"
        && "$config_sha" == "$registered_config_sha" ]] || {
        printf 'source artifact changed after preregistration for cell %s\n' "$ordinal" >&2
        return 1
      }
    else
      checkpoint_sha="dry-checkpoint-$seed-$arm-$update"
      config_sha="dry-config-$seed-$arm"
    fi
    cell_started_ns="$(date +%s%N)"
    if [[ "$dry_run" == 1 ]]; then
      write_dry_report "$output/eval_report.json" "content:evaluator-$eval_seed"
      printf '{"schema":"p2.episode_rollout.v2","dry_run":true}\n' >"$output/episodes.jsonl"
    else
      run_tracked "$output/eval.log" timeout --signal=TERM --kill-after=60s 30m \
        "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" \
        --device cuda:0 --seed "$eval_seed" --synthetic-episodes "$synthetic_episodes" \
        --physical-batch "$physical_batch" --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
        --episode-jsonl "$output/episodes.jsonl" --output "$output/eval_report.json"
      bash "$eval_validator" "$output/eval_report.json" "$output/episodes.jsonl" "$eval_seed"
    fi
    validate_outcome_report "$output/eval_report.json" true
    extract_pair_ledger "$output/eval_report.json" "$output/outcome-pairs.jsonl"
    fingerprint="$(outcome_fingerprint "$output/eval_report.json")"
    cell_finished_ns="$(date +%s%N)"
    duration_ms=$(( (cell_finished_ns - cell_started_ns) / 1000000 ))
    (( $(date +%s) - started_epoch <= hard_max_seconds )) || {
      printf 'hard runtime exceeded after cell %s\n' "$ordinal" >&2
      return 1
    }
    completed_cells=$((completed_cells + 1))
    completed_duration_ms=$((completed_duration_ms + duration_ms))
    jq -cn --argjson ordinal "$ordinal" --arg experiment "$experiment" \
      --argjson evaluation_seed "$eval_seed" --argjson training_seed "$seed" \
      --arg arm "$arm" --argjson update "$update" --arg checkpoint "$checkpoint" \
      --arg config "$config" --arg checkpoint_sha "$checkpoint_sha" --arg config_sha "$config_sha" \
      --arg fingerprint "$fingerprint" --argjson started_ns "$cell_started_ns" \
      --argjson finished_ns "$cell_finished_ns" --argjson duration_ms "$duration_ms" \
      --slurpfile report "$output/eval_report.json" \
      "\$report[0] | $outcome_object_filter as \$o |
       {ordinal:\$ordinal,experiment:\$experiment,evaluation_seed:\$evaluation_seed,
        training_seed:\$training_seed,arm:\$arm,update:\$update,checkpoint:\$checkpoint,
        config:\$config,checkpoint_sha256:\$checkpoint_sha,config_sha256:\$config_sha,
        content_only_population_fingerprint:\$fingerprint,started_epoch_ns:\$started_ns,
        finished_epoch_ns:\$finished_ns,duration_milliseconds:\$duration_ms,
        physical_evaluation_batch:256,training:false,outcome_counterfactuals:\$o}" \
      >>"$root/metrics.jsonl"
    jq -nc --arg stage "cell-$ordinal" --argjson ordinal "$ordinal" \
      --argjson started_ns "$cell_started_ns" --argjson finished_ns "$cell_finished_ns" \
      --argjson duration_ms "$duration_ms" \
      '{stage:$stage,ordinal:$ordinal,status:"passed",started_epoch_ns:$started_ns,
        finished_epoch_ns:$finished_ns,duration_milliseconds:$duration_ms,
        evaluator_validation:true,outcome_counterfactual_validation:true,
        pair_ledger_reconciled:true,all_controls_true:true,all_population_gates_true:true,
        group_bootstrap_resamples:10000,physical_evaluation_batch:256,training:false}' \
      >>"$root/stages.jsonl"
    (
      cd "$root"
      sha256sum "$(realpath --relative-to="$root" "$output/eval_report.json")" \
        "$(realpath --relative-to="$root" "$output/episodes.jsonl")" \
        "$(realpath --relative-to="$root" "$output/outcome-pairs.jsonl")" \
        >"$output/SHA256SUMS"
      sha256sum --quiet -c "$output/SHA256SUMS"
    )
  done < <(awk -F '\t' -v e="$experiment" '$2==e {print}' "$cells_tsv")

  [[ "$(wc -l <"$root/stages.jsonl")" == "$expected_cells" \
    && "$(wc -l <"$root/metrics.jsonl")" == "$expected_cells" ]] || return 1
  [[ "$(jq -r .content_only_population_fingerprint "$root/metrics.jsonl" | sort -u | wc -l)" == 1 ]] || {
    printf 'experiment %s has non-identical content populations\n' "$experiment" >&2
    return 1
  }
  stop_telemetry
  set_json_status "$root/campaign.json" complete_pending_analysis 0
  seal_root "$root"
  current_experiment_finalized=true
}

verify_experiment_seal() {
  local root="$1"
  jq -e '.status=="complete_pending_analysis" and .training==false' "$root/campaign.json" >/dev/null
  (cd "$root" && sha256sum --quiet -c ROOT_SHA256SUMS)
  sha256sum --quiet -c "${root}.ROOT_SHA256SUMS.sha256"
}

run_smoke
run_experiment A 424250 10
run_experiment B 424251 10
run_experiment C 424252 8

verify_experiment_seal "$experiment_a_root"
verify_experiment_seal "$experiment_b_root"
verify_experiment_seal "$experiment_c_root"
if [[ "$test_fail_before_d" == 1 ]]; then
  printf 'test hook: failing closed before D handoff\n' >&2
  exit 97
fi
run_experiment D 424253 4
verify_experiment_seal "$experiment_d_root"

[[ "$completed_cells" == "$evaluation_count" ]] || exit 1
mapfile -t population_fingerprints < <(
  for root in "$experiment_a_root" "$experiment_b_root" "$experiment_c_root" "$experiment_d_root"; do
    jq -r .content_only_population_fingerprint "$root/metrics.jsonl" | sort -u
  done
)
[[ "${#population_fingerprints[@]}" == 4 \
  && "$(printf '%s\n' "${population_fingerprints[@]}" | sort -u | wc -l)" == 4 ]] || {
  printf 'content populations are not distinct across the four evaluator seeds\n' >&2
  exit 1
}

assert_binary_unchanged
assert_preregistration_unchanged
for root in "$smoke_root" "$experiment_a_root" "$experiment_b_root" \
  "$experiment_c_root" "$experiment_d_root"; do
  verify_experiment_seal "$root"
done
set_json_status "$sequence_root/campaign.json" complete_pending_analysis 0
jq -nc --arg a "${population_fingerprints[0]}" --arg b "${population_fingerprints[1]}" \
  --arg c "${population_fingerprints[2]}" --arg d "${population_fingerprints[3]}" \
  '{evaluation_populations:{"424250":$a,"424251":$b,"424252":$c,"424253":$d},
    identical_within_each_evaluator_seed:true,distinct_across_evaluator_seeds:true}' \
  >"$sequence_root/populations.json"
assert_binary_unchanged
assert_preregistration_unchanged
assert_source_ledger_unchanged
assert_source_manifest_unchanged
(( $(date +%s) - started_epoch <= hard_max_seconds )) || exit 1
seal_root "$sequence_root"
sequence_finalized=true
printf 'outcome-counterfactual sequence complete: %s\n' "$sequence_root"
