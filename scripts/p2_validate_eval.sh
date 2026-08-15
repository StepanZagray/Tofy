#!/usr/bin/env bash
# Validate and reconcile one P2 evaluation report with its episode JSONL stream.
set -euo pipefail

if (($# < 3 || $# > 4)); then
  printf 'usage: %s REPORT EPISODES EXPECTED_SEED [REQUIRE_FULL_POPULATION]\n' "$0" >&2
  exit 2
fi

report="$1"
episodes="$2"
expected_seed="$3"
require_full_population="${4:-true}"

[[ "$expected_seed" =~ ^[0-9]+$ ]] || {
  printf 'expected seed must be a non-negative integer\n' >&2
  exit 2
}
[[ "$require_full_population" == true || "$require_full_population" == false ]] || {
  printf 'REQUIRE_FULL_POPULATION must be true or false\n' >&2
  exit 2
}
[[ -s "$report" ]] || { printf 'missing evaluation report: %s\n' "$report" >&2; exit 1; }
[[ -s "$episodes" ]] || { printf 'missing episode stream: %s\n' "$episodes" >&2; exit 1; }

jq -e -R 'fromjson | type=="object"' "$episodes" >/dev/null
jq -e --argjson expected_seed "$expected_seed" \
  --argjson require_full_population "$require_full_population" \
  --slurpfile episodes "$episodes" '
  def finite_number: type=="number" and .>-1e300 and .<1e300;
  def finite_or_null: .==null or finite_number;
  def nonnegative_integer: type=="number" and .>=0 and floor==.;
  def close($left; $right): (($left-$right)|fabs)<1e-9;
  def equal_weight_group_mean:
    group_by(.group.group_index)
    | map(map(.margin)|add/length)
    | add/length;
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
  def sha256: type=="string" and test("^sha256:[0-9a-f]{64}$");
  def cf_interval:
    (.estimate|finite_number)
    and (.lower_95|finite_number) and (.upper_95|finite_number)
    and (.lower_98_75|finite_number) and (.upper_98_75|finite_number)
    and .lower_98_75<=.lower_95 and .lower_95<=.estimate
    and .estimate<=.upper_95 and .upper_95<=.upper_98_75
    and (.groups|nonnegative_integer) and .groups>0
    and (.pairs|nonnegative_integer) and .pairs>0
    and (.resamples|nonnegative_integer) and .resamples>=10000
    and .unit=="whole_branch_group";
  def cf_pair:
    (.group.group_index|nonnegative_integer)
    and (.group.family|type=="string" and length>0)
    and (.group.population=="movement" or .group.population=="coordinate")
    and (.group.content_fingerprint|sha256)
    and (.group.current_sha256|sha256)
    and (.group.next_sha256|type=="array" and length==4 and all(.[]; sha256))
    and (.group.actions|type=="array" and length==4
      and all(.[]; (.id|nonnegative_integer) and .id>=1 and .id<=7))
    and (.left_branch_index|nonnegative_integer)
    and (.right_branch_index|nonnegative_integer)
    and .left_branch_index<.right_branch_index
    and (.left_action|type=="object") and (.right_action|type=="object")
    and (.left_outcome_class|nonnegative_integer)
    and (.right_outcome_class|nonnegative_integer)
    and (.left_changed|type=="boolean") and (.right_changed|type=="boolean")
    and (.left_changed_cells|type=="array" and all(.[]; nonnegative_integer))
    and (.right_changed_cells|type=="array" and all(.[]; nonnegative_integer))
    and (.target_pair_mse|finite_number) and .target_pair_mse>=0
    and (.concordant_loss|finite_number) and .concordant_loss>=0
    and (.crossed_loss|finite_number) and .crossed_loss>=0
    and (.margin|finite_number) and .margin>=-1 and .margin<=1
    and (.eligible|type=="boolean")
    and ((.eligible and .reason=="distinct_canonical_board_outcomes"
          and .left_outcome_class!=.right_outcome_class)
      or ((.eligible|not) and .reason=="outcome_equivalent"
          and .left_outcome_class==.right_outcome_class));
  .schema=="p2.eval_report.v14" and .mode=="full" and .seed==$expected_seed
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
  and (if $require_full_population then .synthetic_dynamics.rollout.h4.n==128
    else .synthetic_dynamics.rollout.h4.n>0 end)
  and (.synthetic_dynamics.rollout.h4.finite_n==.synthetic_dynamics.rollout.h4.n)
  and (if $require_full_population then .synthetic_dynamics.rollout.h8.n==64
    else .synthetic_dynamics.rollout.h8.n>0 end)
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
  and (.outcome_counterfactuals as $oc
    | ($oc.population_fingerprint|sha256)
    and ($oc.groups|nonnegative_integer) and $oc.groups>0
    and ($oc.movement_groups|nonnegative_integer) and $oc.movement_groups>0
    and ($oc.coordinate_groups|nonnegative_integer) and $oc.coordinate_groups>0
    and $oc.groups==$oc.movement_groups+$oc.coordinate_groups
    and ($oc.unordered_pairs|nonnegative_integer)
    and $oc.unordered_pairs==$oc.groups*6
    and ($oc.eligible_pairs|nonnegative_integer)
    and ($oc.outcome_equivalent_pairs|nonnegative_integer)
    and $oc.eligible_pairs+$oc.outcome_equivalent_pairs==$oc.unordered_pairs
    and ($oc.changed_changed_pairs|nonnegative_integer)
    and ($oc.changed_unchanged_pairs|nonnegative_integer)
    and $oc.changed_changed_pairs+$oc.changed_unchanged_pairs==$oc.eligible_pairs
    and ($oc.epsilon|finite_number) and $oc.epsilon>0
    and $oc.material_threshold==0.10
    and ($oc.overall|cf_interval)
    and ($oc.movement|cf_interval)
    and ($oc.coordinate|cf_interval)
    and ($oc.changed_changed|cf_interval)
    and ($oc.changed_unchanged|cf_interval)
    and ($oc.action_separation_pass|type=="boolean")
    and $oc.action_separation_pass
      ==($oc.movement.estimate>0.10 and $oc.movement.lower_95>0.10)
    and ($oc.pair_ledger|type=="array" and length==$oc.unordered_pairs
      and all(.[]; cf_pair))
    and all($oc.pair_ledger[];
      . as $row
      | $row.group.actions[$row.left_branch_index]==$row.left_action
      and $row.group.actions[$row.right_branch_index]==$row.right_action
      and close($row.margin;
        (($row.crossed_loss-$row.concordant_loss)
          /($row.crossed_loss+$row.concordant_loss+$oc.epsilon))))
    and ([$oc.pair_ledger|group_by(.group.group_index)[]
      | length==6
        and ([.[]|[.left_branch_index,.right_branch_index]]|unique|length)==6
        and ([.[].group]|unique|length)==1] | all)
    and ([$oc.pair_ledger[].group.group_index]|unique|length)==$oc.groups
    and ([$oc.pair_ledger[]|select(.group.population=="movement")
          |.group.group_index]|unique|length)==$oc.movement_groups
    and ([$oc.pair_ledger[]|select(.group.population=="coordinate")
          |.group.group_index]|unique|length)==$oc.coordinate_groups
    and ([$oc.pair_ledger[]|select(.eligible)]|length)==$oc.eligible_pairs
    and ([$oc.pair_ledger[]|select(.eligible|not)]|length)==$oc.outcome_equivalent_pairs
    and ([$oc.pair_ledger[]|select(.eligible and .left_changed and .right_changed)]|length)
      ==$oc.changed_changed_pairs
    and ([$oc.pair_ledger[]|select(.eligible and (.left_changed!=.right_changed))]|length)
      ==$oc.changed_unchanged_pairs
    and $oc.overall.pairs==$oc.eligible_pairs
    and $oc.movement.pairs
      ==([$oc.pair_ledger[]|select(.eligible and .group.population=="movement")]|length)
    and $oc.coordinate.pairs
      ==([$oc.pair_ledger[]|select(.eligible and .group.population=="coordinate")]|length)
    and $oc.changed_changed.pairs==$oc.changed_changed_pairs
    and $oc.changed_unchanged.pairs==$oc.changed_unchanged_pairs
    and close(([$oc.pair_ledger[]|select(.eligible)]|equal_weight_group_mean);
      $oc.overall.estimate)
    and close(([$oc.pair_ledger[]|select(.eligible and .group.population=="movement")]
      |equal_weight_group_mean);$oc.movement.estimate)
    and close(([$oc.pair_ledger[]|select(.eligible and .group.population=="coordinate")]
      |equal_weight_group_mean);$oc.coordinate.estimate)
    and close(([$oc.pair_ledger[]|select(.eligible and .left_changed and .right_changed)]
      |equal_weight_group_mean);$oc.changed_changed.estimate)
    and close(([$oc.pair_ledger[]|select(.eligible and (.left_changed!=.right_changed))]
      |equal_weight_group_mean);$oc.changed_unchanged.estimate)
    and $oc.ledger_reconciled==true
    and $oc.controls.pixel_oracle_estimate==1
    and $oc.controls.pixel_oracle_exactly_one==true
    and ($oc.controls.latent_oracle_estimate|finite_number)
    and $oc.controls.latent_oracle_at_least_0_99==true
    and $oc.controls.target_collapse_failure==false
    and $oc.controls.target_collapsed_pairs==0
    and ($oc.controls.swapped_oracle_estimate|finite_number)
    and $oc.controls.swapped_oracle_estimate<=-0.99
    and $oc.controls.swapped_oracle_at_most_negative_0_99==true
    and ($oc.controls.action_masked_max_abs_margin|finite_number)
    and $oc.controls.action_masked_max_abs_margin<=1e-6
    and $oc.controls.action_masked_max_abs_at_most_1e_6==true
    and ($oc.controls.identity_max_abs_margin|finite_number)
    and $oc.controls.identity_max_abs_margin<=1e-6
    and $oc.controls.identity_max_abs_at_most_1e_6==true
    and $oc.controls.outcome_equivalent_pairs==$oc.outcome_equivalent_pairs
    and ($oc.controls.outcome_equivalent_max_abs_margin|finite_number)
    and $oc.controls.outcome_equivalent_max_abs_margin<=1e-6
    and $oc.controls.outcome_equivalent_max_abs_at_most_1e_6==true
    and ($oc.controls.state_scrambled_same_action_template.available|type=="boolean")
    and (($oc.controls.state_scrambled_same_action_template.available==true)
      or ($oc.controls.state_scrambled_same_action_template.reason
          |type=="string" and length>0))
    and $oc.controls.required_controls_pass==true
    and (if $oc.controls.state_scrambled_same_action_template.available then
      ($oc.controls.state_scrambled_same_action_template.estimate|finite_number)
      and $oc.controls.state_scrambled_same_action_template.groups>0
      and $oc.controls.state_scrambled_same_action_template.pairs>0
    else ($oc.controls.state_scrambled_same_action_template.reason
      |type=="string" and length>0) end)
    and (if $require_full_population then
      $oc.population_gates.eligible_simulator_groups>=100
      and $oc.population_gates.eligible_simulator_groups_at_least_100==true
      and ([1,2,3,4]|all(.[]; . as $action
        | $oc.population_gates.movement_action_anchors[$action|tostring].changed>=16
        and $oc.population_gates.movement_action_anchors[$action|tostring].unchanged>=16))
      and $oc.population_gates.each_movement_action_at_least_16_changed_and_16_unchanged==true
      and $oc.population_gates.simulator_changed_changed_pairs>=100
      and $oc.population_gates.simulator_changed_changed_pairs_at_least_100==true
      and $oc.population_gates.target_collapse_failure==false
      and $oc.population_gates.population_pass==true
    else true end))
' "$report" >/dev/null
