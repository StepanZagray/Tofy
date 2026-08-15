#!/usr/bin/env bash
# Validate and reconcile one P2 evaluation report with its episode JSONL stream.
set -euo pipefail

if (($# != 3)); then
  printf 'usage: %s REPORT EPISODES EXPECTED_SEED\n' "$0" >&2
  exit 2
fi

report="$1"
episodes="$2"
expected_seed="$3"

[[ "$expected_seed" =~ ^[0-9]+$ ]] || {
  printf 'expected seed must be a non-negative integer\n' >&2
  exit 2
}
[[ -s "$report" ]] || { printf 'missing evaluation report: %s\n' "$report" >&2; exit 1; }
[[ -s "$episodes" ]] || { printf 'missing episode stream: %s\n' "$episodes" >&2; exit 1; }

jq -e -R 'fromjson | type=="object"' "$episodes" >/dev/null
jq -e --argjson expected_seed "$expected_seed" --slurpfile episodes "$episodes" '
  def finite_number: type=="number" and .>-1e300 and .<1e300;
  def finite_or_null: .==null or finite_number;
  def nonnegative_integer: type=="number" and .>=0 and floor==.;
  def valid_episode:
    .schema=="p2.episode_rollout.v2"
    and (.source=="synthetic_dynamics" or .source=="synthetic_planner")
    and .seed==$expected_seed
    and (.episode_id|nonnegative_integer)
    and (.families_through_horizon|type=="array" and length>0
      and all(.[]; type=="string" and length>0))
    and (.horizon==4 or .horizon==8 or .horizon==16)
    and (.open_mse|finite_number)
    and (.closed_mse|finite_number)
    and (.copy_forward_mse|finite_number)
    and (.normalized_open_mse|finite_or_null);
  def reconciles($source; $rollout):
    [4,8,16] | all(.[];
      . as $h
      | ($rollout["n\($h)"]|nonnegative_integer)
      and ($rollout["h\($h)"].n==$rollout["n\($h)"])
      and ($rollout["h\($h)"].finite_n==$rollout["n\($h)"])
      and ([$episodes[] | select(.source==$source and .horizon==$h)] | length)
        == $rollout["n\($h)"]);
  .schema=="p2.eval_report.v13" and .seed==$expected_seed
  and (.board_probe.population_fingerprint|type=="string" and length>0)
  and (.board_probe.metrics.trusted|type=="boolean")
  and (.synthetic_dynamics.n_samples|type=="number" and .>0)
  and (.synthetic_dynamics.representation.effective_rank_fraction|finite_number)
  and (.synthetic_dynamics.representation.noncollapse_pass|type=="boolean")
  and (.synthetic_dynamics.changed_transitions.improvement_fraction|finite_number)
  and (.synthetic_dynamics.changed_transitions.improvement_ci95_low|finite_number)
  and (.synthetic_dynamics.changed_transitions.ten_percent_improvement_pass|type=="boolean")
  and (.synthetic_dynamics.action_diagnostics.aggregate.shuffle.ratio|finite_number)
  and (.synthetic_dynamics.action_diagnostics.aggregate.shuffle.ratio_ci95_low|finite_number)
  and (.synthetic_dynamics.action_diagnostics.aggregate.shuffle.action_conditioning_pass|type=="boolean")
  and (.synthetic_dynamics.rollout.h4.n==128)
  and (.synthetic_dynamics.rollout.h4.finite_n==.synthetic_dynamics.rollout.h4.n)
  and (.synthetic_dynamics.rollout.h8.n==64)
  and (.synthetic_dynamics.rollout.h8.finite_n==.synthetic_dynamics.rollout.h8.n)
  and (.synthetic_dynamics.rollout.h4.normalized_mean|finite_number)
  and (.synthetic_dynamics.rollout.h4.normalized_cvar95|finite_number)
  and (.synthetic_dynamics.rollout.h8.normalized_mean|finite_number)
  and (.synthetic_dynamics.rollout.h8.normalized_cvar95|finite_number)
  and (.synthetic_dynamics.rollout.h8.fraction_beating_copy|finite_number)
  and (.synthetic_dynamics.q.n|type=="number" and .>0)
  and (.synthetic_dynamics.q.brier|finite_number)
  and (.synthetic_dynamics.q.positive_label_rate|finite_number)
  and (.synthetic_dynamics.q.saturated|type=="boolean")
  and ((.synthetic_dynamics.q.balanced_accuracy==null)
    or (.synthetic_dynamics.q.balanced_accuracy|finite_number))
  and ($episodes|length)>0
  and all($episodes[]; valid_episode)
  and ([$episodes[] | [.source,.seed,.episode_id,.horizon]] | unique | length)
    == ($episodes|length)
  and reconciles("synthetic_dynamics"; .synthetic_dynamics.rollout)
  and reconciles("synthetic_planner"; .synthetic_planner.rollout)
' "$report" >/dev/null
