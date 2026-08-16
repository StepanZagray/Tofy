#!/usr/bin/env bash
# Fail-closed A40 reliability preflight followed by one fresh, preregistered
# w0323 outcome-counterfactual prerequisite evaluation. No training occurs.
set -euo pipefail

readonly hard_runtime_seconds=9000

if [[ "${1:-}" == "--internal-gpu-telemetry" ]]; then
  shift
  (($# == 2)) || exit 2
  trap 'exit 0' TERM INT
  while true; do
    {
      printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    } >>"$1"
    sleep "$2" &
    delay_pid=$!
    wait "$delay_pid" || true
  done
fi

if [[ "${1:-}" != "--internal-hard-runtime" ]]; then
  exec timeout --signal=TERM --kill-after=60s "${hard_runtime_seconds}s" \
    bash "${BASH_SOURCE[0]}" --internal-hard-runtime "$@"
fi
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
validator="$script_dir/p2_validate_eval.sh"
source_run="${P2_OUTCOME_PREREQUISITE_SOURCE_RUN:?set the sealed source run root}"
sequence_root="${P2_OUTCOME_PREREQUISITE_ROOT:?set a never-used sequence root}"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy-outcome-prerequisite-v2}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
dry_run="${P2_OUTCOME_PREREQUISITE_DRY_RUN:-0}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"

: "${P2_EXPECTED_LAUNCHER_SHA:?set the reviewed launcher commit}"
: "${P2_EXPECTED_SOURCE_SHA:?set the source campaign commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"

readonly source_run_id="sigreg-cell-dose-response-v1-20260815T071547Z"
readonly reliability_seed=424250
readonly evidence_seed=424255
readonly physical_batch=256
readonly synthetic_episodes=64
readonly material_threshold=0.10
readonly build_command="cargo build --release --locked --features cudnn"
readonly per_evaluation_guard_seconds=1860
readonly evidence_and_sealing_reserve_seconds=2100

[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || exit 2
[[ "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || exit 2
for command in awk bash cargo date find git install jq mkdir mktemp mv nvidia-smi pgrep \
  realpath rm rmdir setsid sha256sum sleep sort timeout wc xargs; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done

sequence_parent="$(realpath -m "$(dirname -- "$sequence_root")")"
mkdir -p -- "$sequence_parent"
sequence_root="$sequence_parent/$(basename -- "$sequence_root")"
[[ ! -e "$sequence_root" && ! -e "${sequence_root}.ROOT_SHA256SUMS.sha256" ]] || {
  printf 'sequence root or external seal already exists: %s\n' "$sequence_root" >&2
  exit 2
}
mkdir -- "$sequence_root"

active_pid=""
active_pgid=""
active_label=""
telemetry_pid=""
telemetry_pgid=""
sequence_finalized=false
seal_tmp=""
started_epoch="$(date +%s)"

group_is_alive() {
  [[ -n "$1" ]] && pgrep -g "$1" >/dev/null 2>&1
}

stop_group() {
  local pid="$1" pgid="$2" label="$3"
  [[ -n "$pgid" ]] || return 0
  if group_is_alive "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..50}; do group_is_alive "$pgid" || break; sleep 0.1; done
  fi
  if group_is_alive "$pgid"; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
    for _ in {1..50}; do group_is_alive "$pgid" || break; sleep 0.1; done
  fi
  [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  group_is_alive "$pgid" && { printf '%s process group survived cleanup\n' "$label" >&2; return 1; }
  return 0
}

seal_root() {
  [[ -d "$sequence_root" && ! -e "$sequence_root/ROOT_SHA256SUMS" ]] || return 1
  local manifest_hash
  seal_tmp="$(mktemp -d "${sequence_parent}/.outcome-prerequisite-seal.XXXXXX")"
  (
    cd "$sequence_root"
    find . -type f ! -name ROOT_SHA256SUMS -print0 | sort -z | \
      xargs -0 sha256sum >"$seal_tmp/ROOT_SHA256SUMS"
    sha256sum --quiet -c "$seal_tmp/ROOT_SHA256SUMS"
  )
  manifest_hash="$(sha256sum "$seal_tmp/ROOT_SHA256SUMS" | awk '{print $1}')"
  printf '%s  %s/ROOT_SHA256SUMS\n' "$manifest_hash" "$sequence_root" \
    >"$seal_tmp/ROOT_SHA256SUMS.sha256"
  [[ "$(awk '{print $1}' "$seal_tmp/ROOT_SHA256SUMS.sha256")" == "$manifest_hash" ]]
  mv -- "$seal_tmp/ROOT_SHA256SUMS" "$sequence_root/ROOT_SHA256SUMS"
  mv -- "$seal_tmp/ROOT_SHA256SUMS.sha256" "${sequence_root}.ROOT_SHA256SUMS.sha256"
  rmdir -- "$seal_tmp"
  seal_tmp=""
  sha256sum --quiet -c "${sequence_root}.ROOT_SHA256SUMS.sha256"
}

run_tracked_process() {
  local label="$1" log="$2" rc=0
  shift 2
  setsid "$@" >"$log" 2>&1 &
  active_pid=$!
  active_pgid="$active_pid"
  active_label="$label"
  printf '%s\n' "$active_pid" >"${log}.pid"
  wait "$active_pid" || rc=$?
  printf '%s\n' "$rc" >"${log}.exit-code"
  if group_is_alive "$active_pgid"; then
    printf '%s process group survived wait: %s\n' "$label" "$active_pgid" >&2
    return 125
  fi
  active_pid=""; active_pgid=""
  active_label=""
  return "$rc"
}

require_runtime_budget() {
  local before_stage="$1" required_seconds="$2" elapsed remaining
  elapsed=$(( $(date +%s) - started_epoch ))
  remaining=$(( hard_runtime_seconds - elapsed ))
  jq -nc --arg before_stage "$before_stage" --argjson elapsed "$elapsed" \
    --argjson remaining "$remaining" --argjson required "$required_seconds" \
    '{before_stage:$before_stage,elapsed_seconds:$elapsed,remaining_seconds:$remaining,
      required_remaining_seconds:$required,permitted:($remaining>=$required)}' \
    >>"$sequence_root/runtime-gates.jsonl"
  (( remaining >= required_seconds )) || {
    printf 'insufficient runtime before %s: %s seconds remain, %s required\n' \
      "$before_stage" "$remaining" "$required_seconds" >&2
    return 1
  }
}

set_status() {
  local status="$1" exit_code="$2" evidence_class="$3"
  [[ -s "$sequence_root/campaign.json" ]] || return 0
  jq --arg status "$status" --argjson exit_code "$exit_code" \
    --arg evidence_class "$evidence_class" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '.status=$status | .exit_code=$exit_code | .evidence_class=$evidence_class
      | .finished_utc=$finished_utc' "$sequence_root/campaign.json" \
      >"$sequence_root/campaign.json.tmp"
  mv -- "$sequence_root/campaign.json.tmp" "$sequence_root/campaign.json"
}

cleanup() {
  local rc="$?" cleanup_rc=0
  trap - EXIT INT TERM
  stop_group "$active_pid" "$active_pgid" "${active_label:-active-stage}" || cleanup_rc=$?
  stop_group "$telemetry_pid" "$telemetry_pgid" telemetry || cleanup_rc=$?
  (( cleanup_rc == 0 )) || rc=125
  if [[ "$sequence_finalized" != true ]]; then
    if [[ -n "$seal_tmp" && -d "$seal_tmp" ]]; then
      rm -f -- "$seal_tmp/ROOT_SHA256SUMS" "$seal_tmp/ROOT_SHA256SUMS.sha256"
      rmdir -- "$seal_tmp" 2>/dev/null || true
      seal_tmp=""
    fi
    rm -f -- "$sequence_root/ROOT_SHA256SUMS" "${sequence_root}.ROOT_SHA256SUMS.sha256"
    set_status failed_integrity_or_evaluation "$rc" failed_infrastructure_or_integrity || true
    seal_root || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_ns="$(date +%s%N)"
jq -nc --arg started_utc "$started_utc" --argjson started_ns "$started_ns" \
  --arg source_run_id "$source_run_id" --arg launcher_sha "$P2_EXPECTED_LAUNCHER_SHA" \
  --arg source_sha "$P2_EXPECTED_SOURCE_SHA" --arg candle_sha "$P2_EXPECTED_CANDLE_SHA" \
  --arg binary_sha "$P2_EXPECTED_BINARY_SHA" --arg build "$build_command" \
  --argjson dry_run "$dry_run" \
  '{schema:"p2.outcome_counterfactual_prerequisite.v2",status:"running",
    evidence_class:"pending_evidence",training:false,started_utc:$started_utc,
    started_epoch_ns:$started_ns,source_run_id:$source_run_id,
    launcher_git_sha:$launcher_sha,source_git_sha:$source_sha,candle_git_sha:$candle_sha,
    binary_source_git_sha:$launcher_sha,binary_sha256:$binary_sha,
    binary_build_command:$build,binary_features:["cudnn"],gpu_required:"NVIDIA A40",
    physical_evaluation_batch:256,synthetic_episodes:64,
    reliability:{checkpoint:"seed-2/S0/update-250",evaluation_seed:424250,
      cells:["normal-1","normal-2","cuda-launch-blocking"],
      gate:"all evaluator gates pass and canonical outcome objects are byte-identical"},
    evidence:{checkpoint:"seed-2/w0323/update-250",evaluation_seed:424255,
      primary_estimand:"exact-simulator movement-group matching advantage",
      promotion_rule:"movement estimate > 0.10 and movement lower 95% bound > 0.10",
      next_if_pass:"matched S0 paired contrast and independent-seed replication",
      next_if_fail:"reject w0323 prerequisite; do not spend compute on panel"},
    dry_run:($dry_run==1)}' >"$sequence_root/campaign.json"

: >"$sequence_root/stages.jsonl"

source_run="$(realpath "$source_run")"
[[ "$(basename -- "$source_run")" == "$source_run_id" ]] || exit 2
source_manifest="$source_run/root-sha256.txt"
if [[ "$dry_run" == 0 ]]; then
  [[ -s "$source_manifest" && -s "$source_run/campaign.json" ]] || exit 2
  (cd "$source_run" && sha256sum --quiet -c root-sha256.txt)
  jq -e --arg sha "$P2_EXPECTED_SOURCE_SHA" \
    '(.status=="complete" or .status=="complete_pending_analysis") and .git_sha==$sha' \
    "$source_run/campaign.json" >/dev/null
  source_manifest_sha="$(sha256sum "$source_manifest" | awk '{print $1}')"
  jq --arg source_run "$source_run" --arg source_manifest_sha "$source_manifest_sha" \
    '.source_run=$source_run | .source_root_manifest_sha256=$source_manifest_sha' \
    "$sequence_root/campaign.json" >"$sequence_root/campaign.json.tmp"
  mv -- "$sequence_root/campaign.json.tmp" "$sequence_root/campaign.json"
else
  source_manifest_sha="dry"
fi

s0_checkpoint="$source_run/seed-2/S0/checkpoints/step-000000000250/model.safetensors"
s0_config="$source_run/seed-2/S0/config.json"
w_checkpoint="$source_run/seed-2/w0323/checkpoints/step-000000000250/model.safetensors"
w_config="$source_run/seed-2/w0323/config.json"
if [[ "$dry_run" == 0 ]]; then
  for artifact in "$s0_checkpoint" "$s0_config" "$w_checkpoint" "$w_config"; do
    [[ -s "$artifact" ]] || { printf 'missing source artifact: %s\n' "$artifact" >&2; exit 2; }
  done
  s0_checkpoint_sha="$(sha256sum "$s0_checkpoint" | awk '{print $1}')"
  s0_config_sha="$(sha256sum "$s0_config" | awk '{print $1}')"
  w_checkpoint_sha="$(sha256sum "$w_checkpoint" | awk '{print $1}')"
  w_config_sha="$(sha256sum "$w_config" | awk '{print $1}')"
else
  s0_checkpoint_sha=dry
  s0_config_sha=dry
  w_checkpoint_sha=dry
  w_config_sha=dry
fi
jq -nc --arg s0_checkpoint "$s0_checkpoint" --arg s0_config "$s0_config" \
  --arg w_checkpoint "$w_checkpoint" --arg w_config "$w_config" \
  --arg s0_checkpoint_sha "$s0_checkpoint_sha" --arg s0_config_sha "$s0_config_sha" \
  --arg w_checkpoint_sha "$w_checkpoint_sha" --arg w_config_sha "$w_config_sha" \
  '{S0:{checkpoint:$s0_checkpoint,config:$s0_config,checkpoint_sha256:$s0_checkpoint_sha,
      config_sha256:$s0_config_sha},w0323:{checkpoint:$w_checkpoint,config:$w_config,
      checkpoint_sha256:$w_checkpoint_sha,config_sha256:$w_config_sha}}' \
  >"$sequence_root/source-artifacts.json"

jq -nc --arg registered_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_manifest_sha "$source_manifest_sha" \
  --arg s0_checkpoint_sha "$s0_checkpoint_sha" --arg s0_config_sha "$s0_config_sha" \
  --arg w_checkpoint_sha "$w_checkpoint_sha" --arg w_config_sha "$w_config_sha" \
  --arg binary_sha "$P2_EXPECTED_BINARY_SHA" \
  '{schema:"p2.outcome_counterfactual_prerequisite_preregistration.v2",
    registered_utc:$registered_utc,training:false,
    observed_evidence_seed_results_before_registration:false,
    claim:{distribution:"64 deterministic held-out-composition synthetic episodes at seed 424255",
      objective:"frozen-checkpoint outcome-counterfactual action sensitivity",
      comparator_class:"registered material threshold; matched S0 is deferred unless prerequisite passes",
      compute_budget:"one w0323 seed-2 update-250 evaluation after three diagnostic reliability cells",
      better:"movement estimate and lower 95% whole-group bootstrap bound both exceed 0.10",
      scope:"bounded empirical prerequisite, not proof of global optimality or ARC-AGI-3 performance"},
    intervention:{training_seed:2,arm:"w0323",update:250,evaluation_seed:424255,
      checkpoint_sha256:$w_checkpoint_sha,config_sha256:$w_config_sha},
    exact_artifacts:{source_root_manifest_sha256:$source_manifest_sha,
      evaluator_binary_sha256:$binary_sha,
      reliability_checkpoint_sha256:$s0_checkpoint_sha,
      reliability_config_sha256:$s0_config_sha},
    invariants:{source_campaign:"sigreg-cell-dose-response-v1-20260815T071547Z",
      physical_batch:256,synthetic_episodes:64,ptrm_k:[1],ptrm_noise:0,
      ensemble_members:1,device:"cuda:0",bootstrap_unit:"whole_branch_group",
      bootstrap_resamples:10000,hard_runtime_seconds:9000,
      per_evaluation_guard_seconds:1860,evidence_and_sealing_reserve_seconds:2100},
    reliability_gate:{checkpoint:{training_seed:2,arm:"S0",update:250},
      evaluation_seed:424250,repeats:["normal-1","normal-2","cuda-launch-blocking"],
      require_exact_canonical_outcome_parity:true,require_all_evaluator_gates:true},
    stop_rule:"stop before evidence on any provenance, CUDA, validation, or parity failure; after evidence stop regardless of result",
    branch_rule:{pass:"queue matched S0 paired contrast and independent-seed replication only after analysis",
      fail:"reject w0323 prerequisite and do not run the former 32-cell panel"},
    multiplicity_policy:"one preregistered evidence cell and one primary estimand; no post-observation selection"}' \
  >"$sequence_root/preregistration.json"
preregistration_sha="$(sha256sum "$sequence_root/preregistration.json" | awk '{print $1}')"
sha256sum "$sequence_root/preregistration.json" >"${sequence_root}.PREREGISTRATION.sha256"

binary_sha="$P2_EXPECTED_BINARY_SHA"
if [[ "$dry_run" == 0 ]]; then
  [[ -d "$candle_root" && -r "$validator" ]] || exit 2
  [[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_LAUNCHER_SHA" ]] || exit 2
  [[ "$(git -C "$candle_root" rev-parse HEAD)" == "$P2_EXPECTED_CANDLE_SHA" ]] || exit 2
  [[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
  [[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle checkout\n' >&2; exit 2; }
  git -C "$repo_root" fetch origin main
  [[ "$(git -C "$repo_root" rev-parse refs/remotes/origin/main)" == "$P2_EXPECTED_LAUNCHER_SHA" ]] || exit 2
  git -C "$repo_root" cat-file -e "${P2_EXPECTED_SOURCE_SHA}^{commit}"
  git -C "$candle_root" fetch origin main
  [[ "$(git -C "$candle_root" rev-parse refs/remotes/origin/main)" == "$P2_EXPECTED_CANDLE_SHA" ]] || exit 2
  run_tracked_process build "$sequence_root/build.log" timeout --signal=TERM --kill-after=60s \
    20m bash -c \
    'cd -- "$1" && exec cargo build --release --locked --features cudnn' bash "$repo_root"
  built_binary="$repo_root/target/release/tofy"
  [[ "$(sha256sum "$built_binary" | awk '{print $1}')" == "$P2_EXPECTED_BINARY_SHA" ]] || exit 2
  [[ "$tofy_bin" != "$built_binary" && ! -e "$tofy_bin" ]] || exit 2
  install -m 0755 -- "$built_binary" "$tofy_bin"
  binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  [[ "${#gpu_names[@]}" == 1 && "${gpu_names[0]}" == "NVIDIA A40" ]] || exit 2
  read -r gpu_memory gpu_utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, 'NR==1 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
  )
  (( gpu_memory <= 1024 && gpu_utilization == 0 )) || {
    printf 'A40 is not idle (memory=%s MiB utilization=%s%%)\n' "$gpu_memory" "$gpu_utilization" >&2
    exit 2
  }
  IFS=, read -r gpu_name gpu_uuid driver_version gpu_memory_total < <(
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader,nounits |
      awk -F, 'NR==1 {for(i=1;i<=NF;i++) gsub(/^ +| +$/,"",$i); print $1 "," $2 "," $3 "," $4}'
  )
  jq -nc --arg gpu_name "$gpu_name" --arg gpu_uuid "$gpu_uuid" \
    --arg driver_version "$driver_version" --argjson gpu_memory_mib "$gpu_memory_total" \
    --arg rustc "$(rustc --version)" --arg cargo "$(cargo --version)" \
    --arg kernel "$(uname -srmo)" \
    '{gpu:{name:$gpu_name,uuid:$gpu_uuid,driver_version:$driver_version,
      memory_total_mib:$gpu_memory_mib},software:{rustc:$rustc,cargo:$cargo,kernel:$kernel}}' \
    >"$sequence_root/environment.json"

  assert_invariants() {
    sha256sum --quiet -c "${sequence_root}.PREREGISTRATION.sha256"
    [[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$source_manifest_sha" ]]
    [[ "$(sha256sum "$s0_checkpoint" | awk '{print $1}')" == "$s0_checkpoint_sha" ]]
    [[ "$(sha256sum "$s0_config" | awk '{print $1}')" == "$s0_config_sha" ]]
    [[ "$(sha256sum "$w_checkpoint" | awk '{print $1}')" == "$w_checkpoint_sha" ]]
    [[ "$(sha256sum "$w_config" | awk '{print $1}')" == "$w_config_sha" ]]
    [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$binary_sha" ]]
  }
  assert_invariants
  nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER >"$sequence_root/gpu-health-before.txt"
  stress_started="$(date +%s%N)"
  run_tracked_process collector-stress-normal "$sequence_root/collector-stress-normal.log" \
    timeout --signal=TERM --kill-after=60s 10m env TOFY_CUDA_COLLECTOR_STRESS_ITERS=250 bash -c \
    'cd -- "$1" && exec cargo test --release --locked --features cudnn \
      --test cuda_representation_collector -- --ignored --nocapture' bash "$repo_root"
  jq -nc --arg stage collector-cuda-stress-normal --argjson iterations 250 \
    --argjson started_ns "$stress_started" --argjson finished_ns "$(date +%s%N)" \
    '{stage:$stage,status:"passed",evidence_class:"implementation_diagnostic",
      iterations:$iterations,started_epoch_ns:$started_ns,finished_epoch_ns:$finished_ns}' \
    >>"$sequence_root/stages.jsonl"
  stress_started="$(date +%s%N)"
  run_tracked_process collector-stress-blocking "$sequence_root/collector-stress-blocking.log" \
    timeout --signal=TERM --kill-after=60s 10m env CUDA_LAUNCH_BLOCKING=1 \
    TOFY_CUDA_COLLECTOR_STRESS_ITERS=250 bash -c \
    'cd -- "$1" && exec cargo test --release --locked --features cudnn \
      --test cuda_representation_collector -- --ignored --nocapture' bash "$repo_root"
  jq -nc --arg stage collector-cuda-stress-blocking --argjson iterations 250 \
    --argjson started_ns "$stress_started" --argjson finished_ns "$(date +%s%N)" \
    '{stage:$stage,status:"passed",evidence_class:"implementation_diagnostic",
      cuda_launch_blocking:true,iterations:$iterations,started_epoch_ns:$started_ns,
      finished_epoch_ns:$finished_ns}' >>"$sequence_root/stages.jsonl"
fi

if [[ "$dry_run" == 0 ]]; then
  setsid bash "$script_dir/$(basename -- "${BASH_SOURCE[0]}")" \
    --internal-gpu-telemetry "$sequence_root/gpu.csv" "$gpu_interval" &
  telemetry_pid=$!
  telemetry_pgid="$telemetry_pid"
  printf '%s\n' "$telemetry_pid" >"$sequence_root/telemetry.pid"
fi

validate_report() {
  local report="$1"
  jq -e '
    .schema=="p2.eval_report.v14"
    and .outcome_counterfactuals.ledger_reconciled==true
    and .outcome_counterfactuals.controls.required_controls_pass==true
    and .outcome_counterfactuals.population_gates.population_pass==true
    and .outcome_counterfactuals.movement.resamples==10000
    and .outcome_counterfactuals.movement.unit=="whole_branch_group"' "$report" >/dev/null
}

write_dry_report() {
  local report="$1" seed="$2" estimate="$3" lower="$4"
  jq -nc --argjson seed "$seed" --argjson estimate "$estimate" --argjson lower "$lower" '
    {schema:"p2.eval_report.v14",seed:$seed,outcome_counterfactuals:{ledger_reconciled:true,
      population_fingerprint:"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      controls:{required_controls_pass:true},population_gates:{population_pass:true},
      movement:{estimate:$estimate,lower_95:$lower,resamples:10000,unit:"whole_branch_group"}}}' \
    >"$report"
}

run_eval() {
  local label="$1" checkpoint="$2" config="$3" seed="$4" classification="$5" blocking="$6"
  local root="$sequence_root/$label" started finished rc=0
  mkdir -- "$root"
  started="$(date +%s%N)"
  if [[ "$dry_run" == 1 ]]; then
    write_dry_report "$root/eval_report.json" "$seed" 0.05 0.01
    printf '{"schema":"p2.episode_rollout.v2","dry_run":true}\n' >"$root/episodes.jsonl"
  else
    assert_invariants
    local -a command=(timeout --signal=TERM --kill-after=60s 30m "$tofy_bin" p2-eval
      --checkpoint "$checkpoint" --train-config "$config" --device cuda:0 --seed "$seed"
      --synthetic-episodes "$synthetic_episodes" --physical-batch "$physical_batch"
      --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1
      --episode-jsonl "$root/episodes.jsonl" --output "$root/eval_report.json")
    if [[ "$blocking" == true ]]; then command=(env CUDA_LAUNCH_BLOCKING=1 "${command[@]}"); fi
    run_tracked_process evaluation "$root/eval.log" "${command[@]}" || rc=$?
    (( rc == 0 )) || return "$rc"
    [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$binary_sha" ]] || return 1
    assert_invariants
    bash "$validator" "$root/eval_report.json" "$root/episodes.jsonl" "$seed"
  fi
  validate_report "$root/eval_report.json"
  jq -S -c '.outcome_counterfactuals' "$root/eval_report.json" >"$root/outcome-canonical.json"
  sha256sum "$root/outcome-canonical.json" | awk '{print $1}' >"$root/outcome.sha256"
  finished="$(date +%s%N)"
  jq -nc --arg stage "$label" --arg class "$classification" --argjson seed "$seed" \
    --argjson started_ns "$started" --argjson finished_ns "$finished" --argjson blocking "$blocking" \
    '{stage:$stage,status:"passed",evidence_class:$class,evaluation_seed:$seed,
      cuda_launch_blocking:$blocking,started_epoch_ns:$started_ns,finished_epoch_ns:$finished_ns,
      duration_milliseconds:(($finished_ns-$started_ns)/1000000|floor)}' \
    >>"$sequence_root/stages.jsonl"
}

require_runtime_budget reliability-normal-1 \
  $((3 * per_evaluation_guard_seconds + evidence_and_sealing_reserve_seconds))
run_eval reliability-normal-1 "$s0_checkpoint" "$s0_config" "$reliability_seed" implementation_diagnostic false
require_runtime_budget reliability-normal-2 \
  $((2 * per_evaluation_guard_seconds + evidence_and_sealing_reserve_seconds))
run_eval reliability-normal-2 "$s0_checkpoint" "$s0_config" "$reliability_seed" implementation_diagnostic false
require_runtime_budget reliability-cuda-launch-blocking \
  $((per_evaluation_guard_seconds + evidence_and_sealing_reserve_seconds))
run_eval reliability-cuda-launch-blocking "$s0_checkpoint" "$s0_config" "$reliability_seed" implementation_diagnostic true

mapfile -t reliability_hashes < <(for label in reliability-normal-1 reliability-normal-2 \
  reliability-cuda-launch-blocking; do cat "$sequence_root/$label/outcome.sha256"; done)
[[ "$(printf '%s\n' "${reliability_hashes[@]}" | sort -u | wc -l)" == 1 ]] || {
  printf 'reliability outcome parity failed\n' >&2
  exit 1
}
jq -nc --arg outcome_sha "${reliability_hashes[0]}" \
  '{status:"passed",canonical_outcome_sha256:$outcome_sha,exact_parity:true,
    normal_repeats:2,cuda_launch_blocking_repeats:1}' >"$sequence_root/reliability-gate.json"

require_runtime_budget evidence-w0323-seed-424255 "$evidence_and_sealing_reserve_seconds"
run_eval evidence-w0323-seed-424255 "$w_checkpoint" "$w_config" "$evidence_seed" completed_evidence false
estimate="$(jq -r '.outcome_counterfactuals.movement.estimate' \
  "$sequence_root/evidence-w0323-seed-424255/eval_report.json")"
lower="$(jq -r '.outcome_counterfactuals.movement.lower_95' \
  "$sequence_root/evidence-w0323-seed-424255/eval_report.json")"
promotion="$(jq -n --argjson estimate "$estimate" --argjson lower "$lower" \
  --argjson threshold "$material_threshold" '$estimate>$threshold and $lower>$threshold')"
jq -nc --argjson estimate "$estimate" --argjson lower_95 "$lower" \
  --argjson threshold "$material_threshold" --argjson promotion "$promotion" \
  '{movement_estimate:$estimate,movement_lower_95:$lower_95,threshold:$threshold,
    promotion_gate_pass:$promotion,
    decision:(if $promotion then "queue_matched_S0_and_replication" else "reject_w0323_prerequisite" end)}' \
  >"$sequence_root/decision.json"

if [[ "$dry_run" == 0 ]]; then
  nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER >"$sequence_root/gpu-health-after.txt"
  assert_invariants
fi

stop_group "$telemetry_pid" "$telemetry_pgid" telemetry
telemetry_pid=""; telemetry_pgid=""
sha256sum --quiet -c "${sequence_root}.PREREGISTRATION.sha256"
if [[ "$dry_run" == 0 ]]; then
  [[ "$(sha256sum "$tofy_bin" | awk '{print $1}')" == "$binary_sha" ]] || exit 1
fi
set_status complete_pending_analysis 0 completed_evidence
jq --arg preregistration_sha "$preregistration_sha" --arg binary_sha "$binary_sha" \
  '.preregistration_sha256=$preregistration_sha | .binary_sha256=$binary_sha' \
  "$sequence_root/campaign.json" >"$sequence_root/campaign.json.tmp"
mv -- "$sequence_root/campaign.json.tmp" "$sequence_root/campaign.json"
seal_root
sequence_finalized=true
printf 'outcome-counterfactual prerequisite sequence complete: %s\n' "$sequence_root"
