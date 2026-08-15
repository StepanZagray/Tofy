#!/usr/bin/env bash
# Re-evaluate the completed cell-TC-QQ dose campaign with the deconfounded
# changed-conditioning-only action metric. This is a frozen-checkpoint premise audit;
# it performs no training and must reproduce every pre-existing report field exactly.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
eval_validator="$script_dir/p2_validate_eval.sh"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
source_run="${P2_ACTION_PREMISE_SOURCE_RUN:?set the completed dose-response run root}"
run_root="${P2_ACTION_PREMISE_ROOT:-$repo_root/runs/p2/action-premise-rescore-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
eval_batch="${P2_ACTION_PREMISE_EVAL_BATCH:-256}"
gpu_interval="${P2_GPU_SAMPLE_INTERVAL:-15}"
eval_seed=424248

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_SOURCE_SHA:?set the source campaign Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
: "${P2_EXPECTED_BINARY_SHA:?set the reviewed release binary SHA-256}"
for command in awk bash cmp date find git jq mkdir mv nvidia-smi realpath sha256sum sleep sort timeout wc xargs; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ "$eval_batch" =~ ^[1-9][0-9]*$ && "$gpu_interval" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$tofy_bin" && -r "$eval_validator" && -d "$source_run" ]] || exit 2

source_run="$(realpath "$source_run")"
git_sha="$(git -C "$repo_root" rev-parse HEAD)"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
source_sha="$(jq -r .git_sha "$source_run/campaign.json")"
source_status="$(jq -r .status "$source_run/campaign.json")"
source_manifest="$source_run/root-sha256.txt"
[[ -s "$source_manifest" ]] || { printf 'missing source root manifest\n' >&2; exit 2; }
source_manifest_sha="$(sha256sum "$source_manifest" | awk '{print $1}')"
[[ "$git_sha" == "$P2_EXPECTED_SHA" ]] || { printf 'Tofy SHA mismatch\n' >&2; exit 2; }
[[ "$source_sha" == "$P2_EXPECTED_SOURCE_SHA" ]] || { printf 'source SHA mismatch\n' >&2; exit 2; }
[[ "$candle_sha" == "$P2_EXPECTED_CANDLE_SHA" ]] || { printf 'candle_graph SHA mismatch\n' >&2; exit 2; }
[[ "$binary_sha" == "$P2_EXPECTED_BINARY_SHA" ]] || { printf 'binary SHA mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }
[[ "$source_status" == complete || "$source_status" == complete_pending_analysis ]] || {
  printf 'source campaign is not complete: %s\n' "$source_status" >&2; exit 2;
}
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
  "$@" >"$log_path" 2>&1 &
  active_pid=$!
  if wait "$active_pid"; then rc=0; else rc=$?; fi
  active_pid=""
  return "$rc"
}

cleanup() {
  local rc="$?"
  if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
    kill "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  active_pid=""
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
  --arg git_sha "$git_sha" --arg source_git_sha "$source_sha" \
  --arg source_run "$source_run" --arg candle_sha "$candle_sha" \
  --arg source_manifest_sha "$source_manifest_sha" \
  --arg binary_sha "$binary_sha" --arg gpu "$gpu_name" --argjson eval_batch "$eval_batch" \
  '{schema:"p2.action_premise_rescore.v1",status:"running",started_utc:$started_utc,
    git_sha:$git_sha,source_git_sha:$source_git_sha,source_run:$source_run,
    source_root_manifest_sha256:$source_manifest_sha,candle_git_sha:$candle_sha,
    binary_sha256:$binary_sha,gpu_name:$gpu,evaluation_seed:424248,
    physical_evaluation_batch:$eval_batch,
    training_seeds:[2,3],arms:["S0","w004","w008","w016","w0323"],
    checkpoints:[125,250],evaluations:20,training:false,
    question:"Does action sensitivity remain absent after excluding shuffled rows whose full action tuple did not change?",
    decision:"If changed-only CI remains near one, test same-state counterfactual separation; do not tune QQ dose."}' \
  >"$run_root/campaign.json"
: >"$run_root/stages.jsonl"
: >"$run_root/metrics.jsonl"

sample_gpu >>"$run_root/gpu.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" >"$run_root/telemetry.pid"

for seed in 2 3; do
  for arm in S0 w004 w008 w016 w0323; do
    for update in 125 250; do
      source_arm="$source_run/seed-$seed/$arm"
      printf -v checkpoint '%s/checkpoints/step-%012d/model.safetensors' "$source_arm" "$update"
      config="$source_arm/config.json"
      source_eval="$source_arm/eval-update-$update"
      output="$run_root/seed-$seed/$arm/eval-update-$update"
      mkdir -p -- "$output"
      [[ -s "$checkpoint" && -s "$config" && -s "$source_eval/eval_report.json" \
        && -s "$source_eval/episodes.jsonl" ]] || exit 1

      run_tracked "$output/eval.log" timeout --signal=TERM --kill-after=60s 30m \
        "$tofy_bin" p2-eval --checkpoint "$checkpoint" --train-config "$config" \
        --device cuda:0 --seed "$eval_seed" --synthetic-episodes 64 \
        --physical-batch "$eval_batch" --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
        --episode-jsonl "$output/episodes.jsonl" --output "$output/eval_report.json"
      bash "$eval_validator" "$output/eval_report.json" "$output/episodes.jsonl" "$eval_seed"
      jq -e '
        def finite_number: type=="number" and .>-1e300 and .<1e300;
        .synthetic_dynamics.action_diagnostics.changed_conditioning_only as $m
        | ($m.n|type=="number" and .>0 and floor==.)
          and $m.changed_conditionings==$m.n and $m.changed_fraction==1
          and ($m.true_action_mse|finite_number) and ($m.shuffled_action_mse|finite_number)
          and ($m.ratio|finite_number) and ($m.ratio_ci95_low|finite_number)
          and ($m.ratio_ci95_high|finite_number)
          and ($m.action_conditioning_pass|type=="boolean")' \
        "$output/eval_report.json" >/dev/null

      jq -S 'del(.synthetic_dynamics.action_diagnostics.changed_conditioning_only,
        .synthetic_planner.action_diagnostics.changed_conditioning_only)' \
        "$output/eval_report.json" >"$output/report-without-new-metric.json"
      jq -S . "$source_eval/eval_report.json" >"$output/source-report-normalized.json"
      cmp -s "$output/report-without-new-metric.json" "$output/source-report-normalized.json"
      cmp -s "$output/episodes.jsonl" "$source_eval/episodes.jsonl"

      jq -c --argjson seed "$seed" --arg arm "$arm" --argjson update "$update" '
        .synthetic_dynamics as $d
        | {seed:$seed,arm:$arm,update:$update,
           changed_conditioning_only:$d.action_diagnostics.changed_conditioning_only,
           aggregate:$d.action_diagnostics.aggregate.shuffle,
           changed_transition:$d.action_diagnostics.by_transition_kind.changed_transition,
           effective_rank_fraction:$d.representation.effective_rank_fraction,
           changed_transition_improvement:$d.changed_transitions.improvement_fraction,
           h4_normalized_mean:$d.rollout.h4.normalized_mean,
           h8_normalized_mean:$d.rollout.h8.normalized_mean}' \
        "$output/eval_report.json" >>"$run_root/metrics.jsonl"
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
          legacy_report_exact:true,episode_stream_exact:true}' >>"$run_root/stages.jsonl"
    done
  done
done

[[ "$(wc -l <"$run_root/metrics.jsonl")" == 20 ]] || exit 1
[[ "$(wc -l <"$run_root/stages.jsonl")" == 20 ]] || exit 1
reference_fingerprint="$(jq -r .board_probe.population_fingerprint \
  "$run_root/seed-2/S0/eval-update-125/eval_report.json")"
while IFS= read -r report; do
  [[ "$(jq -r .board_probe.population_fingerprint "$report")" == "$reference_fingerprint" ]] || exit 1
done < <(find "$run_root" -name eval_report.json -type f | sort)

(cd "$source_run" && sha256sum --quiet -c root-sha256.txt)
[[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$source_manifest_sha" ]] || exit 1
stop_telemetry
finish_campaign complete_pending_analysis
(
  cd "$run_root"
  find . -type f ! -name ROOT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >ROOT_SHA256SUMS
  sha256sum --quiet -c ROOT_SHA256SUMS
)
campaign_finalized=true
printf 'action-premise rescore complete: %s\n' "$run_root"
