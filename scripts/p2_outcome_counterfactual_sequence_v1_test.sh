#!/usr/bin/env bash
# Process-free orchestration test for p2_outcome_counterfactual_sequence_v1.sh.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/p2_outcome_counterfactual_sequence_v1.sh"
fixture="$(mktemp -d "${TMPDIR:-/tmp}/p2-outcome-cf-sequence-test.XXXXXX")"
trap 'rm -rf -- "$fixture"' EXIT

run_dry() {
  local root="$1" fail_before_d="${2:-0}"
  P2_OUTCOME_COUNTERFACTUAL_SOURCE_RUN="$fixture/sigreg-cell-dose-response-v1-20260815T071547Z" \
  P2_OUTCOME_COUNTERFACTUAL_SEQUENCE_ROOT="$root" \
  P2_BINARY_BUILD_COMMAND='cargo build --release --locked --features cudnn' \
  P2_EXPECTED_LAUNCHER_SHA=1111111111111111111111111111111111111111 \
  P2_EXPECTED_SOURCE_SHA=2222222222222222222222222222222222222222 \
  P2_EXPECTED_CANDLE_SHA=3333333333333333333333333333333333333333 \
  P2_EXPECTED_BINARY_SOURCE_SHA=4444444444444444444444444444444444444444 \
  P2_EXPECTED_BINARY_SHA=5555555555555555555555555555555555555555555555555555555555555555 \
  P2_OUTCOME_COUNTERFACTUAL_DRY_RUN=1 \
  P2_OUTCOME_COUNTERFACTUAL_TEST_FAIL_BEFORE_D="$fail_before_d" \
    bash "$launcher"
}

success_root="$fixture/success/sequence"
mkdir -p -- "$(dirname -- "$success_root")"
run_dry "$success_root"

[[ "$(wc -l <"$success_root/preregistered-cells.tsv")" == 32 ]]
expected="$fixture/expected-order.tsv"
{
  ordinal=0
  for experiment_seed in 'A 424250' 'B 424251'; do
    read -r experiment eval_seed <<<"$experiment_seed"
    for arm in S0 w004 w008 w016 w0323; do
      ordinal=$((ordinal + 1)); printf '%s\t%s\t%s\t2\t%s\t250\n' "$ordinal" "$experiment" "$eval_seed" "$arm"
    done
    for arm in w0323 w016 w008 w004 S0; do
      ordinal=$((ordinal + 1)); printf '%s\t%s\t%s\t3\t%s\t250\n' "$ordinal" "$experiment" "$eval_seed" "$arm"
    done
  done
  for arm in S0 w0323; do for update in 125 250; do
    ordinal=$((ordinal + 1)); printf '%s\tC\t424252\t2\t%s\t%s\n' "$ordinal" "$arm" "$update"
  done; done
  for arm in w0323 S0; do for update in 125 250; do
    ordinal=$((ordinal + 1)); printf '%s\tC\t424252\t3\t%s\t%s\n' "$ordinal" "$arm" "$update"
  done; done
  for arm in S0 w0323; do
    ordinal=$((ordinal + 1)); printf '%s\tD\t424253\t2\t%s\t250\n' "$ordinal" "$arm"
  done
  for arm in w0323 S0; do
    ordinal=$((ordinal + 1)); printf '%s\tD\t424253\t3\t%s\t250\n' "$ordinal" "$arm"
  done
} >"$expected"
cmp "$expected" "$success_root/preregistered-cells.tsv"

jq -e '
  .observed_sequence_results_before_registration==false
  and .source_campaign_preexisted==true and .training==false and .no_training==true
  and .cell_count==32 and (.cells|length)==32
  and .material_matching_advantage_threshold==0.10
  and .bootstrap.unit=="whole_branch_group" and .bootstrap.group_resamples==10000
  and .primary_estimand=="exact-simulator movement-group matching advantage"
  and (.primary_rule|contains("movement lower 95%"))
  and (.source_manifest_sha256|length)>0
  and (.source_artifact_ledger_sha256|length)>0
  and .per_cell_intervals==[{coverage_percent:95},{coverage_percent:98.75}]
  and .outcome_classes==["simulator_outcome_equivalent","simulator_outcome_changing"]
  and .controls.required_all_true==true
  and .population_gates.required_all_true_for_evidence_cells==true
  and (.handoff_rule|contains("A, B, and C"))' "$success_root/preregistration.json" >/dev/null

mapfile -t experiment_roots < <(jq -r '.experiment_roots[]' "$success_root/campaign.json")
[[ "${#experiment_roots[@]}" == 4 \
  && "$(printf '%s\n' "${experiment_roots[@]}" | sort -u | wc -l)" == 4 ]]
for root in "${experiment_roots[@]}" "$success_root/implementation-smoke-seed-424254"; do
  [[ -s "$root/ROOT_SHA256SUMS" && -s "${root}.ROOT_SHA256SUMS.sha256" ]]
  (cd "$root" && sha256sum --quiet -c ROOT_SHA256SUMS)
  sha256sum --quiet -c "${root}.ROOT_SHA256SUMS.sha256"
done
[[ -s "$success_root/ROOT_SHA256SUMS" && -s "${success_root}.ROOT_SHA256SUMS.sha256" ]]
[[ "$(wc -l <"$success_root/source-artifacts.tsv")" == 32 ]]
sha256sum --quiet -c "${success_root}.PREREGISTRATION.sha256"
jq -e '.status=="complete_pending_analysis" and .evaluation_count==32 and .training==false' \
  "$success_root/campaign.json" >/dev/null
[[ "$(find "$success_root" -type f -name eval_report.json | wc -l)" == 33 ]]

failure_root="$fixture/failure/sequence"
mkdir -p -- "$(dirname -- "$failure_root")"
set +e
run_dry "$failure_root" 1 >"$fixture/failure.stdout" 2>"$fixture/failure.stderr"
failure_rc=$?
set -e
[[ "$failure_rc" == 97 ]]
jq -e '.status=="failed_integrity_or_evaluation" and .exit_code==97' \
  "$failure_root/campaign.json" >/dev/null
for experiment in A-seed-424250 B-seed-424251 C-seed-424252; do
  root="$failure_root/experiment-$experiment"
  jq -e '.status=="complete_pending_analysis"' "$root/campaign.json" >/dev/null
  [[ -s "$root/ROOT_SHA256SUMS" && -s "${root}.ROOT_SHA256SUMS.sha256" ]]
done
[[ ! -e "$failure_root/experiment-D-seed-424253" ]]
[[ -s "$failure_root/ROOT_SHA256SUMS" && -s "${failure_root}.ROOT_SHA256SUMS.sha256" ]]
grep -q 'failing closed before D handoff' "$fixture/failure.stderr"

printf 'PASS: 32-cell order, preregistration, unique roots/seals, and fail-closed D handoff\n'
