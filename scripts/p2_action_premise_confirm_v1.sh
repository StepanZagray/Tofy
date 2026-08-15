#!/usr/bin/env bash
# Held-out evaluator-population confirmation for the fixed action-premise endpoints.
# This evaluates frozen checkpoints only; it performs no training or arm selection.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
eval_validator="$script_dir/p2_validate_eval.sh"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
parent_run="${P2_ACTION_CONFIRM_PARENT_RUN:?set the completed exploratory parent root}"
run_root="${P2_ACTION_CONFIRM_ROOT:-$repo_root/runs/p2/action-premise-confirm-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_ACTION_CONFIRM_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
binary_build_command="${P2_BINARY_BUILD_COMMAND:?set the exact binary build command}"
parent_eval_seed=424248
eval_seed=424249
update=250
evaluation_count=4

: "${P2_EXPECTED_SHA:?set the reviewed Tofy launcher commit}"
: "${P2_EXPECTED_PARENT_SHA:?set the reviewed exploratory-parent commit}"
: "${P2_EXPECTED_SOURCE_SHA:?set the source campaign Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"
: "${P2_EXPECTED_BINARY_SOURCE_SHA:?set the Tofy commit used to build the binary}"
for command in awk bash date find git grep jq mkdir mv nvidia-smi realpath setsid sha256sum sleep sort timeout wc xargs; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ "$eval_batch" =~ ^[1-9][0-9]*$ && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$binary_build_command" == "cargo build --release --locked --features cudnn" ]] || {
  printf 'unexpected binary build command\n' >&2; exit 2;
}
[[ -x "$tofy_bin" && -r "$eval_validator" && -d "$parent_run" ]] || exit 2

parent_run="$(realpath "$parent_run")"
parent_manifest="$parent_run/ROOT_SHA256SUMS"
parent_manifest_digest_record="${parent_run}.ROOT_SHA256SUMS.sha256"
[[ -s "$parent_manifest" ]] || { printf 'missing parent root manifest\n' >&2; exit 2; }
[[ -s "$parent_manifest_digest_record" ]] || {
  printf 'missing external parent-manifest digest\n' >&2; exit 2;
}
parent_manifest_sha="$(sha256sum "$parent_manifest" | awk '{print $1}')"
parent_status="$(jq -r .status "$parent_run/campaign.json")"
parent_sha="$(jq -r .git_sha "$parent_run/campaign.json")"
parent_binary_sha="$(jq -r .binary_sha256 "$parent_run/campaign.json")"
parent_binary_source_sha="$(jq -r .binary_source_git_sha "$parent_run/campaign.json")"
source_run="$(realpath "$(jq -r .source_run "$parent_run/campaign.json")")"
device_smoke_run="$(realpath "$(jq -r .device_smoke_run "$parent_run/campaign.json")")"
source_manifest="$source_run/root-sha256.txt"
[[ -s "$source_manifest" ]] || { printf 'missing source root manifest\n' >&2; exit 2; }
source_manifest_sha="$(sha256sum "$source_manifest" | awk '{print $1}')"
[[ "$(jq -r .source_root_manifest_sha256 "$parent_run/campaign.json")" == "$source_manifest_sha" ]] || {
  printf 'source manifest continuity mismatch\n' >&2; exit 2;
}

git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
source_sha="$(jq -r .git_sha "$source_run/campaign.json")"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy launcher SHA mismatch\n' >&2; exit 2; }
[[ "$parent_sha" == "$P2_EXPECTED_PARENT_SHA" ]] || { printf 'parent SHA mismatch\n' >&2; exit 2; }
[[ "$source_sha" == "$P2_EXPECTED_SOURCE_SHA" ]] || { printf 'source SHA mismatch\n' >&2; exit 2; }
[[ "$(jq -r .source_git_sha "$parent_run/campaign.json")" == "$source_sha" ]] || {
  printf 'source revision continuity mismatch\n' >&2; exit 2;
}
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" && "$parent_binary_sha" == "$binary_sha" ]] || {
  printf 'binary SHA mismatch\n' >&2; exit 2;
}
[[ "$parent_binary_source_sha" == "$P2_EXPECTED_BINARY_SOURCE_SHA" ]] || {
  printf 'parent binary-source SHA mismatch\n' >&2; exit 2;
}
smoke_report="$device_smoke_run/seed-2/S0/eval-update-125/eval_report.json"
smoke_log="$device_smoke_run/seed-2/S0/eval-update-125/eval.log"
[[ -s "$smoke_report" && -s "$smoke_log" ]] || { printf 'incomplete device smoke\n' >&2; exit 2; }
smoke_report_sha="$(sha256sum "$smoke_report" | awk '{print $1}')"
smoke_log_sha="$(sha256sum "$smoke_log" | awk '{print $1}')"
[[ "$(jq -r .device_smoke_report_sha256 "$parent_run/campaign.json")" == "$smoke_report_sha" \
  && "$(jq -r .device_smoke_log_sha256 "$parent_run/campaign.json")" == "$smoke_log_sha" ]] || {
  printf 'device smoke artifact continuity mismatch\n' >&2; exit 2;
}
[[ "$(jq -r .git_sha "$device_smoke_run/campaign.json")" == "$P2_EXPECTED_BINARY_SOURCE_SHA" \
  && "$(jq -r .binary_sha256 "$device_smoke_run/campaign.json")" == "$binary_sha" ]] || {
  printf 'device smoke provenance mismatch\n' >&2; exit 2;
}
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }
[[ "$parent_status" == complete || "$parent_status" == complete_pending_analysis ]] || {
  printf 'parent campaign is not complete: %s\n' "$parent_status" >&2; exit 2;
}
jq -e --argjson parent_eval_seed "$parent_eval_seed" '
  .schema=="p2.action_premise_rescore.v1"
  and .evidence_class=="exploratory_evaluator_calibration"
  and .evaluation_seed==$parent_eval_seed
  and .training==false
  and .training_seeds==[2,3]
  and .arms==["S0","w004","w008","w016","w0323"]
  and .checkpoints==[250]
  and .evaluations==10' "$parent_run/campaign.json" >/dev/null
[[ "$(wc -l <"$parent_run/metrics.jsonl")" == 10 ]] || exit 2
[[ "$(wc -l <"$parent_run/stages.jsonl")" == 10 ]] || exit 2
(cd "$parent_run" && sha256sum --quiet -c ROOT_SHA256SUMS)
sha256sum --quiet -c "$parent_manifest_digest_record"
(cd "$source_run" && sha256sum --quiet -c root-sha256.txt)

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == 1 )) || { printf 'campaign requires one visible GPU\n' >&2; exit 2; }
gpu_name="${gpu_names[0]}"
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }
read -r initial_memory initial_utilization < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
(( initial_memory <= 1024 && initial_utilization == 0 )) || {
  printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' "$initial_memory" "$initial_utilization" >&2
  exit 2
}

mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root" || { printf 'run root exists: %s\n' "$run_root" >&2; exit 2; }
run_root="$(realpath "$run_root")"
started_epoch="$(date +%s)"
campaign_finalized=false
telemetry_pid=""
active_pid=""
active_pgid=""

assert_binary_unchanged() {
  [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$binary_sha" ]] || {
    printf 'launch binary changed during campaign\n' >&2
    return 1
  }
}

assert_preregistration_unchanged() {
  sha256sum --quiet -c "$preregistration_digest_record"
}

finish_campaign() {
  local status="$1"
  jq --arg status "$status" --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
    '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
}

sample_gpu() {
  local delay_pid=""
  trap 'if [[ -n "$delay_pid" ]] && kill -0 "$delay_pid" 2>/dev/null; then
    kill "$delay_pid" 2>/dev/null || true
    wait "$delay_pid" 2>/dev/null || true
  fi
  exit 0' TERM INT
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "$gpu_interval" &
    delay_pid=$!
    wait "$delay_pid" || true
    delay_pid=""
  done
}

stop_telemetry() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
}

run_tracked() {
  local log_path="$1" rc
  shift
  assert_binary_unchanged
  assert_preregistration_unchanged
  setsid "$@" >"$log_path" 2>&1 &
  active_pid=$!
  active_pgid="$active_pid"
  printf '%s\n' "$active_pid" >"${log_path}.pid"
  if wait "$active_pid"; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" >"${log_path}.exit-code"
  if kill -0 -- "-$active_pgid" 2>/dev/null; then
    printf 'evaluation process group still active after wait: %s\n' "$active_pgid" >&2
    return 1
  fi
  active_pid=""
  active_pgid=""
  assert_binary_unchanged
  assert_preregistration_unchanged
  return "$rc"
}

cleanup() {
  local rc="$?"
  if [[ -n "$active_pgid" ]] && kill -0 -- "-$active_pgid" 2>/dev/null; then
    kill -TERM -- "-$active_pgid" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 -- "-$active_pgid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-$active_pgid" 2>/dev/null; then
      kill -KILL -- "-$active_pgid" 2>/dev/null || true
      for _ in {1..50}; do
        kill -0 -- "-$active_pgid" 2>/dev/null || break
        sleep 0.1
      done
    fi
    wait "$active_pid" 2>/dev/null || true
    if kill -0 -- "-$active_pgid" 2>/dev/null; then
      printf 'failed to terminate evaluation process group: %s\n' "$active_pgid" >&2
      rc=125
    fi
  fi
  active_pid=""
  active_pgid=""
  stop_telemetry
  if [[ "$campaign_finalized" != true && -s "$run_root/campaign.json" ]]; then
    jq --arg status failed_integrity_or_evaluation \
      --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson exit_code "$rc" --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
      '.status=$status | .finished_utc=$finished_utc | .exit_code=$exit_code
        | .elapsed_seconds=$elapsed_seconds' \
      "$run_root/campaign.json" >"$run_root/campaign.json.tmp" 2>/dev/null &&
      mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

jq -nc --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg git_sha "$git_sha" --arg parent_git_sha "$parent_sha" --arg source_git_sha "$source_sha" \
  --arg parent_run "$parent_run" --arg source_run "$source_run" \
  --arg device_smoke_run "$device_smoke_run" --arg candle_sha "$candle_sha" \
  --arg parent_manifest_sha "$parent_manifest_sha" --arg source_manifest_sha "$source_manifest_sha" \
  --arg binary_sha "$binary_sha" --arg binary_source_sha "$P2_EXPECTED_BINARY_SOURCE_SHA" \
  --arg binary_build_command "$binary_build_command" \
  --arg gpu "$gpu_name" --argjson eval_batch "$eval_batch" '
  {schema:"p2.action_premise_confirmation.v1",status:"running",started_utc:$started_utc,
   git_sha:$git_sha,parent_git_sha:$parent_git_sha,source_git_sha:$source_git_sha,
   parent_run:$parent_run,source_run:$source_run,device_smoke_run:$device_smoke_run,
   evidence_class:"confirmatory_heldout_evaluation_seed_postselection",
   parent_root_manifest_sha256:$parent_manifest_sha,
   source_root_manifest_sha256:$source_manifest_sha,candle_git_sha:$candle_sha,
   binary_sha256:$binary_sha,binary_source_git_sha:$binary_source_sha,
   binary_build_command:$binary_build_command,binary_features:["cudnn"],
   evaluator_schema:"p2.eval_report.v13",gpu_name:$gpu,
   parent_evaluation_seed:424248,evaluation_seed:424249,physical_evaluation_batch:$eval_batch,
   training_seeds:[2,3],arms:["S0","w0323"],checkpoints:[250],evaluations:4,
   training:false,eligible_rows_metric:"changed action tuple",outcome_changing_rows_measured:false,
   selection_status:"fixed endpoint cells and update were registered before the complete parent panel; exact reference Booleans are registered after the parent and before held-out evaluation",
   preregistered_success_rule:"all four held-out changed-only ratio CI upper bounds are below the registered material-sensitivity ratio 1.10; classification concordance with the parent is reported separately",
   multiplicity_policy:"conjunctive four-cell bound; any held-out CI upper bound at or above 1.10 rejects the bounded claim",
   stop_rule:"evaluate exactly four fixed cells unless integrity or evaluation fails",
   scope_limit:"tests whether a 10% action-tuple sensitivity effect is excluded at fixed endpoints on one fresh evaluator seed; does not prove zero sensitivity, action correctness, causal grounding, planning validity, or ARC-AGI-3 performance"}' \
  >"$run_root/campaign.json"
: >"$run_root/stages.jsonl"
: >"$run_root/metrics.jsonl"

: >"$run_root/preregistered-cells.jsonl"
for seed in 2 3; do
  for arm in S0 w0323; do
    parent_report="$parent_run/seed-$seed/$arm/eval-update-$update/eval_report.json"
    jq -c --argjson training_seed "$seed" --arg arm "$arm" --argjson update "$update" '
      .synthetic_dynamics.action_diagnostics.changed_conditioning_only as $m
      | {training_seed:$training_seed,arm:$arm,update:$update,
         metric_path:"synthetic_dynamics.action_diagnostics.changed_conditioning_only",
         reference_action_conditioning_pass:$m.action_conditioning_pass,
         reference_ratio:$m.ratio,reference_ratio_ci95_low:$m.ratio_ci95_low,
         reference_ratio_ci95_high:$m.ratio_ci95_high}' "$parent_report" \
      >>"$run_root/preregistered-cells.jsonl"
  done
done
jq -s --arg registered_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg parent_manifest_sha "$parent_manifest_sha" '
  {schema:"p2.action_premise_confirmation_preregistration.v1",
   registered_utc:$registered_utc,heldout_evaluator_seed:424249,
   heldout_observed:false,parent_root_manifest_sha256:$parent_manifest_sha,
   cells:.,material_sensitivity_ratio:1.10,
   success_rule:"all four held-out changed-only ratio_ci95_high values must be below 1.10; parent classification concordance is descriptive replication evidence",
   selection_status:"post-selection confirmation of fixed endpoint cells; exact reference values were observed in the parent before this held-out registration",
   multiplicity_policy:"conjunctive four-cell bound; any held-out CI upper bound at or above 1.10 rejects the bounded claim",
   stop_rule:"exactly four fixed cells unless integrity or evaluation fails",
   no_replacement_seed:true}' "$run_root/preregistered-cells.jsonl" \
  >"$run_root/preregistration.json"
preregistration_sha="$(sha256sum "$run_root/preregistration.json" | awk '{print $1}')"
preregistration_digest_record="${run_root}.PREREGISTRATION.sha256"
sha256sum "$run_root/preregistration.json" >"$preregistration_digest_record"
sha256sum --quiet -c "$preregistration_digest_record"
jq --arg preregistration_sha "$preregistration_sha" \
  '.preregistration_sha256=$preregistration_sha' "$run_root/campaign.json" \
  >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
printf 'preregistration sealed before held-out evaluation: %s\n' "$preregistration_sha"

sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"

for seed in 2 3; do
  for arm in S0 w0323; do
    source_arm="$source_run/seed-$seed/$arm"
    printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$source_arm" "$update"
    config="$source_arm/config.json"
    parent_report="$parent_run/seed-$seed/$arm/eval-update-$update/eval_report.json"
    output="$run_root/seed-$seed/$arm/eval-update-$update"
    mkdir -p -- "$output"
    [[ -s "$checkpoint" && -s "$config" && -s "$parent_report" ]] || exit 1
    jq -e '.synthetic_dynamics.action_diagnostics.changed_conditioning_only
      | .n>0 and .changed_conditionings==.n and .changed_fraction==1
        and (.action_conditioning_pass|type=="boolean")' "$parent_report" >/dev/null

    run_tracked "$output/eval.log" timeout --signal=TERM --kill-after=60s 30m \
      "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" \
      --device cuda:0 --seed "$eval_seed" --synthetic-episodes 64 \
      --physical-batch "$eval_batch" --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
      --episode-jsonl "$output/episodes.jsonl" --output "$output/eval_report.json"
    grep -q 'p2-eval smoke complete' "$output/eval.log"
    bash "$eval_validator" "$output/eval_report.json" "$output/episodes.jsonl" "$eval_seed"
    jq -e '
      def finite_number: type=="number" and .>-1e300 and .<1e300;
      .synthetic_dynamics.action_diagnostics.changed_conditioning_only as $m
      | ($m.n|type=="number" and .>0 and floor==.)
        and $m.changed_conditionings==$m.n and $m.changed_fraction==1
        and ($m.true_action_mse|finite_number and .>=0)
        and ($m.shuffled_action_mse|finite_number and .>=0)
        and ($m.ratio|finite_number) and ($m.ratio_ci95_low|finite_number)
        and ($m.ratio_ci95_high|finite_number)
        and $m.true_action_mse>0
        and (($m.ratio-($m.shuffled_action_mse/$m.true_action_mse))|fabs)<1e-9
        and $m.ratio_ci95_low<=$m.ratio and $m.ratio<=$m.ratio_ci95_high
        and ($m.action_conditioning_pass|type=="boolean")
        and ($m.action_conditioning_pass
          == ($m.ratio>=1.1 and $m.ratio_ci95_low>1.0))' "$output/eval_report.json" >/dev/null

    checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
    config_sha="$(sha256sum "$config" | awk '{print $1}')"
    jq -cn --argjson training_seed "$seed" --arg arm "$arm" --argjson update "$update" \
      --arg checkpoint_sha "$checkpoint_sha" --arg config_sha "$config_sha" \
      --slurpfile confirmation "$output/eval_report.json" --slurpfile parent "$parent_report" '
      $confirmation[0].synthetic_dynamics.action_diagnostics.changed_conditioning_only as $c
      | $parent[0].synthetic_dynamics.action_diagnostics.changed_conditioning_only as $p
      | {training_seed:$training_seed,arm:$arm,update:$update,
         checkpoint_sha256:$checkpoint_sha,config_sha256:$config_sha,
         eligible_rows:$c.n,genuinely_changed_tuples:$c.changed_conditionings,
         outcome_changing_tuples:null,
         parent_changed_conditioning_only:$p,
         confirmation_changed_conditioning_only:$c,
         classification_match:($p.action_conditioning_pass==$c.action_conditioning_pass),
         confirmation_population_fingerprint:$confirmation[0].board_probe.population_fingerprint}' \
      >>"$run_root/metrics.jsonl"
    (
      cd "$run_root"
      sha256sum \
        "$(realpath --relative-to="$run_root" "$checkpoint")" \
        "$(realpath --relative-to="$run_root" "$config")" \
        "$(realpath --relative-to="$run_root" "$output/eval_report.json")" \
        "$(realpath --relative-to="$run_root" "$output/episodes.jsonl")"
    ) >"$output/SHA256SUMS"
    jq -nc --arg stage "seed-$seed/$arm/update-$update" \
      --arg at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{stage:$stage,status:"passed",at_utc:$at_utc,
        evaluator_validation:true,fixed_before_heldout_observation:true,
        training:false,outcome_changing_tuples_measured:false}' >>"$run_root/stages.jsonl"
  done
done

[[ "$(wc -l <"$run_root/metrics.jsonl")" == "$evaluation_count" ]] || exit 1
[[ "$(wc -l <"$run_root/stages.jsonl")" == "$evaluation_count" ]] || exit 1
confirmation_fingerprint="$(jq -r .confirmation_population_fingerprint "$run_root/metrics.jsonl" | sort -u)"
[[ "$(printf '%s\n' "$confirmation_fingerprint" | wc -l)" == 1 ]] || exit 1
parent_fingerprint="$(jq -r .board_probe.population_fingerprint \
  "$parent_run/seed-2/S0/eval-update-$update/eval_report.json")"
[[ "$confirmation_fingerprint" != "$parent_fingerprint" ]] || {
  printf 'held-out population fingerprint unexpectedly matches parent\n' >&2; exit 1;
}
jq -nc --arg parent "$parent_fingerprint" --arg confirmation "$confirmation_fingerprint" \
  '{parent_evaluation_seed:424248,parent_population_fingerprint:$parent,
    confirmation_evaluation_seed:424249,confirmation_population_fingerprint:$confirmation,
    populations_distinct:($parent!=$confirmation)}' >"$run_root/population.json"
jq -s '
  {schema:"p2.action_premise_confirmation_decision.v1",cells:length,
   registered_material_sensitivity_ratio:1.10,
   threshold_classification_reproduced:all(.[];.classification_match),
   all_heldout_ci_upper_bounds_below_material_ratio:
     all(.[];.confirmation_changed_conditioning_only.ratio_ci95_high<1.10),
   outcome:(if all(.[];.confirmation_changed_conditioning_only.ratio_ci95_high<1.10)
     then "heldout_row_bootstrap_bound_below_1.10"
     else "material_sensitivity_not_excluded" end),
   zero_sensitivity_proved:false,
   interpretation_limit:"fixed endpoint cells at one held-out evaluator seed; row bootstrap does not account for within-episode dependence and changed tuples need not change simulator outcomes"}' \
  "$run_root/metrics.jsonl" >"$run_root/decision.json"

(cd "$parent_run" && sha256sum --quiet -c ROOT_SHA256SUMS)
sha256sum --quiet -c "$parent_manifest_digest_record"
[[ "$(sha256sum "$parent_manifest" | awk '{print $1}')" == "$parent_manifest_sha" ]] || exit 1
(cd "$source_run" && sha256sum --quiet -c root-sha256.txt)
[[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$source_manifest_sha" ]] || exit 1
assert_binary_unchanged
assert_preregistration_unchanged
[[ "$(sha256sum "$smoke_report" | awk '{print $1}')" == "$smoke_report_sha" \
  && "$(sha256sum "$smoke_log" | awk '{print $1}')" == "$smoke_log_sha" ]] || exit 1
stop_telemetry
finish_campaign complete_pending_analysis
(
  cd "$run_root"
  find . -type f ! -name ROOT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >ROOT_SHA256SUMS
  sha256sum --quiet -c ROOT_SHA256SUMS
)
assert_binary_unchanged
assert_preregistration_unchanged
sha256sum "$run_root/ROOT_SHA256SUMS" >"${run_root}.ROOT_SHA256SUMS.sha256"
assert_binary_unchanged
assert_preregistration_unchanged
campaign_finalized=true
printf 'action-premise confirmation complete: %s\n' "$run_root"
