#!/usr/bin/env bash
# Sequential causal campaign for the architecture rewrite. GPU-heavy stages
# never overlap; failed seed-1 hard gates prevent downstream promotion.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tofy_bin="${TOFY_BIN:-$repo_root/target/release/tofy}"
candle_root="${P2_CANDLE_ROOT:-$repo_root/../candle_graph}"
run_root="${P2_ARCH_RUN_ROOT:-$repo_root/runs/p2/architecture-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
geometry_steps="${P2_ARCH_GEOMETRY_STEPS:-1000}"
action_steps="${P2_ARCH_ACTION_STEPS:-100}"
eval_batch="${P2_ARCH_EVAL_BATCH:-256}"

: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_CANDLE_SHA:?set the reviewed candle_graph commit}"
for command in git jq nvidia-smi sha256sum awk; do
  command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 2; }
done
[[ -x "$tofy_bin" ]] || { printf 'missing release binary: %s\n' "$tofy_bin" >&2; exit 2; }
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$P2_EXPECTED_SHA" ]] || exit 2
[[ "$(git -C "$candle_root" rev-parse HEAD)" == "$P2_EXPECTED_CANDLE_SHA" ]] || exit 2
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'dirty Tofy checkout\n' >&2; exit 2; }
[[ -z "$(git -C "$candle_root" status --porcelain)" ]] || { printf 'dirty candle_graph checkout\n' >&2; exit 2; }
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
[[ "$gpu_name" == "NVIDIA A40" ]] || { printf 'expected NVIDIA A40, got %s\n' "$gpu_name" >&2; exit 2; }

mkdir -p -- "$run_root"
binary_sha="$(sha256sum "$tofy_bin" | awk '{print $1}')"
candle_sha="$(git -C "$candle_root" rev-parse HEAD)"
started_epoch="$(date +%s)"
jq -nc --arg schema p2.architecture_campaign.v1 --arg status running \
  --arg git_sha "$P2_EXPECTED_SHA" --arg candle_git_sha "$candle_sha" \
  --arg binary_sha256 "$binary_sha" --arg gpu_name "$gpu_name" \
  --arg started_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson geometry_steps "$geometry_steps" --argjson action_steps "$action_steps" \
  '{schema:$schema,status:$status,git_sha:$git_sha,candle_git_sha:$candle_git_sha,
    binary_sha256:$binary_sha256,gpu_name:$gpu_name,started_utc:$started_utc,
    geometry_steps:$geometry_steps,action_steps_per_lesson:$action_steps,
    sequence:["batch_probe","marginal_ep","temporal_ep","temporal_qq",
      "action_global","action_spatial_residual","dense_horizon_if_promoted"]}' \
  >"$run_root/campaign.json"

record_stage() {
  jq -nc --arg stage "$1" --arg status "$2" --arg detail "${3:-}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{stage:$stage,status:$status,detail:$detail,finished_utc:$finished_utc}' \
    >>"$run_root/stages.jsonl"
}

fail_campaign() {
  local stage="$1" detail="$2"
  jq --arg status failed --arg failed_stage "$stage" --arg failure_detail "$detail" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '.status=$status | .failed_stage=$failed_stage | .failure_detail=$failure_detail
      | .finished_utc=$finished_utc' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
}

common_model_args=(
  --steady-gpu --supervise-last-outer-only --residual-y-update --warm-start-y
  --outer-steps 8 --inner-steps 2 --hidden-dim 128 --action-dim 8
  --event-weight 0 --q-weight 0 --reliability-weight 0 --ensemble-members 1
)

# Largest allowed physical population for the preregistered effective batch is
# tried first. Accumulation increases only on OOM/failure.
physical_batch=""
grad_accum=""
for pair in "1024 1" "512 2" "256 4"; do
  read -r candidate accum <<<"$pair"
  probe_dir="$run_root/batch-probe-$candidate-$accum"
  if "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons sequential \
      --steps-per-lesson 1000 --max-steps-this-run 2 \
      --physical-batch "$candidate" --grad-accum "$accum" \
      --checkpoint-every-steps 0 --profile-update 2 \
      --sigreg-target temporal-residual --sigreg-statistic epps-pulley \
      --sigreg-temporal-window 8 --sigreg-spatial --sigreg-spatial-pool \
      --sigreg-max-rows 32768 --rollout-weight 0 --prefix-weight 0 \
      "${common_model_args[@]}" --output-dir "$probe_dir" \
      >"$probe_dir.log" 2>&1; then
    physical_batch="$candidate"
    grad_accum="$accum"
    record_stage batch_probe passed "physical_batch=$candidate grad_accum=$accum effective_batch=1024"
    break
  fi
  record_stage "batch_probe_$candidate" failed "see $probe_dir.log"
done
[[ -n "$physical_batch" ]] || {
  record_stage batch_probe failed all_candidates
  fail_campaign batch_probe all_candidates_failed
  exit 1
}
jq --argjson physical_batch "$physical_batch" --argjson grad_accum "$grad_accum" \
  '.physical_batch=$physical_batch | .grad_accum=$grad_accum | .effective_batch=1024' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"

run_eval() {
  local arm_dir="$1"
  "$tofy_bin" p2-eval --checkpoint "$arm_dir/model.safetensors" \
    --train-config "$arm_dir/config.json" --device cuda:0 --seed 424242 \
    --synthetic-episodes 64 --physical-batch "$eval_batch" --ptrm-k 1 --ptrm-noise 0 \
    --ensemble-members 1 --episode-jsonl "$arm_dir/episodes.jsonl" \
    --output "$arm_dir/eval_report.json" >"$arm_dir/eval.log" 2>&1
}

run_geometry_arm() {
  local name="$1" target="$2" statistic="$3" arm_dir="$run_root/geometry/$name"
  mkdir -p -- "$arm_dir"
  if ! "$tofy_bin" p2-train --device cuda:0 --seed 1 --lessons sequential \
      --steps-per-lesson "$geometry_steps" --physical-batch "$physical_batch" \
      --grad-accum "$grad_accum" --checkpoint-every-steps 250 --profile-update 2 \
      --sigreg-target "$target" --sigreg-statistic "$statistic" \
      --sigreg-temporal-window 8 --sigreg-spatial --sigreg-spatial-pool \
      --sigreg-max-rows 32768 --rollout-weight 0 --prefix-weight 0 \
      "${common_model_args[@]}" --output-dir "$arm_dir" >"$arm_dir/train.log" 2>&1; then
    record_stage "$name" failed training
    return 1
  fi
  run_eval "$arm_dir" || { record_stage "$name" failed evaluation; return 1; }
  if jq -e '.synthetic_dynamics.representation.noncollapse_pass == true
      and .synthetic_dynamics.representation.effective_rank_fraction >= 0.10
      and (.synthetic_dynamics.rollout.h8.finite_n // 0) > 0' \
      "$arm_dir/eval_report.json" >/dev/null; then
    record_stage "$name" passed seed1_hard_representation_gate
    return 0
  fi
  record_stage "$name" rejected seed1_hard_representation_gate
  return 3
}

geometry_passes=0
for definition in \
  "marginal_ep marginal epps-pulley" \
  "temporal_ep temporal-residual epps-pulley" \
  "temporal_qq temporal-residual quantile"; do
  read -r name target statistic <<<"$definition"
  set +e
  run_geometry_arm "$name" "$target" "$statistic"
  rc=$?
  set -e
  ((rc == 0)) && geometry_passes=$((geometry_passes + 1))
  ((rc == 1)) && { fail_campaign "$name" training_or_evaluation_failed; exit 1; }
done

if ((geometry_passes == 0)); then
  record_stage downstream skipped no_geometry_arm_passed_seed1_gate
  jq --arg status complete_no_promotion --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
    '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
    "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
  mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
  exit 0
fi

run_action_arm() {
  local name="$1" residual="$2" arm_dir="$run_root/action/$name"
  local -a residual_args=()
  [[ "$residual" == true ]] && residual_args=(--spatial-action-field --spatial-action-residual --spatial-action-residual-scale 0.25)
  mkdir -p -- "$arm_dir"
  if ! "$tofy_bin" p2-train --device cuda:0 --seed 1 --world-core-v3 \
    "${residual_args[@]}" --shuffled-episodes \
    --lessons factual_branches,dynamics,sequential --steps-per-lesson "$action_steps" \
    --physical-batch "$physical_batch" --grad-accum "$grad_accum" \
    --checkpoint-every-steps 100 --profile-update 2 --sigreg-weight 0 \
    --outcome-pull-weight 0.05 --outcome-push-weight 0.05 --outcome-margin 0.5 \
    --action-recovery-weight 0.05 --coordinate-recovery-weight 0.05 \
    --changed-margin-weight 0.05 --changed-margin 0.1 \
    --rollout-weight 0.1 --prefix-weight 0.05 "${common_model_args[@]}" \
    --output-dir "$arm_dir" >"$arm_dir/train.log" 2>&1; then
    record_stage "$name" failed training
    return 1
  fi
  run_eval "$arm_dir" || { record_stage "$name" failed evaluation; return 1; }
  if jq -e '.factual_branches.rows_reconciled == true
      and ([.factual_branches.by_action_id["1","2","3","4"]
        | (.changed > 0 and .unchanged > 0)] | all)
      and .factual_branches.board_probe.trusted == true
      and (.factual_branches.unique_changed_effect_action_top1 // 0)
        > (.factual_branches.majority_action_baseline_top1 // 1)' \
      "$arm_dir/eval_report.json" >/dev/null; then
    record_stage "$name" passed seed1_action_semantic_gate
    return 0
  fi
  record_stage "$name" rejected seed1_action_semantic_gate
  return 3
}

action_passes=0
for definition in "action_global false" "action_spatial_residual true"; do
  read -r name residual <<<"$definition"
  set +e
  run_action_arm "$name" "$residual"
  rc=$?
  set -e
  ((rc == 0)) && action_passes=$((action_passes + 1))
  ((rc == 1)) && { fail_campaign "$name" training_or_evaluation_failed; exit 1; }
done

if ((action_passes == 0)); then
  record_stage dense_horizon skipped no_action_arm_passed_seed1_gate
else
  # Dense horizons remain a separate phase; existing true H1/H4/H8 evaluation
  # is retained, but no planner is promoted merely from a passing latent loss.
  record_stage dense_horizon prepared_requires_three_seed_action_promotion
fi
jq --arg status complete --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson elapsed_seconds "$(( $(date +%s) - started_epoch ))" \
  '.status=$status | .finished_utc=$finished_utc | .elapsed_seconds=$elapsed_seconds' \
  "$run_root/campaign.json" >"$run_root/campaign.json.tmp"
mv -- "$run_root/campaign.json.tmp" "$run_root/campaign.json"
