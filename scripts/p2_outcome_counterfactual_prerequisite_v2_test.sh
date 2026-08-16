#!/usr/bin/env bash
# Process-free orchestration test for the prerequisite sequence.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/p2_outcome_counterfactual_prerequisite_v2.sh"
fixture="$(mktemp -d "${TMPDIR:-/tmp}/p2-outcome-prerequisite-test.XXXXXX")"
trap 'rm -rf -- "$fixture"' EXIT

source_run="$fixture/sigreg-cell-dose-response-v1-20260815T071547Z"
root="$fixture/sequence"
mkdir -p -- "$source_run"

P2_OUTCOME_PREREQUISITE_SOURCE_RUN="$source_run" \
P2_OUTCOME_PREREQUISITE_ROOT="$root" \
P2_EXPECTED_LAUNCHER_SHA=1111111111111111111111111111111111111111 \
P2_EXPECTED_SOURCE_SHA=2222222222222222222222222222222222222222 \
P2_EXPECTED_CANDLE_SHA=3333333333333333333333333333333333333333 \
P2_EXPECTED_BINARY_SHA=4444444444444444444444444444444444444444444444444444444444444444 \
P2_OUTCOME_PREREQUISITE_DRY_RUN=1 \
  bash "$launcher"

jq -e '.status=="complete_pending_analysis" and .training==false
  and .evidence_class=="completed_evidence"
  and .reliability.cells==["normal-1","normal-2","cuda-launch-blocking"]
  and .evidence.evaluation_seed==424255' "$root/campaign.json" >/dev/null
jq -e '.observed_evidence_seed_results_before_registration==false
  and .intervention.training_seed==2 and .intervention.arm=="w0323"
  and .intervention.update==250 and .intervention.evaluation_seed==424255
  and .intervention.checkpoint_sha256=="dry" and .intervention.config_sha256=="dry"
  and .reliability_gate.require_exact_canonical_outcome_parity==true
  and .exact_artifacts.source_root_manifest_sha256=="dry"
  and .exact_artifacts.evaluator_binary_sha256=="4444444444444444444444444444444444444444444444444444444444444444"
  and (.stop_rule|contains("stop before evidence"))' "$root/preregistration.json" >/dev/null
jq -e '.status=="passed" and .exact_parity==true and .normal_repeats==2
  and .cuda_launch_blocking_repeats==1' "$root/reliability-gate.json" >/dev/null
jq -e '.promotion_gate_pass==false and .decision=="reject_w0323_prerequisite"' \
  "$root/decision.json" >/dev/null
[[ "$(wc -l <"$root/stages.jsonl")" == 4 ]]
[[ "$(wc -l <"$root/runtime-gates.jsonl")" == 4 ]]
jq -e '.permitted==true' "$root/runtime-gates.jsonl" >/dev/null
[[ ! -e "$root/gpu.csv" && ! -e "$root/telemetry.pid" ]]
sha256sum --quiet -c "${root}.PREREGISTRATION.sha256"
(cd "$root" && sha256sum --quiet -c ROOT_SHA256SUMS)
sha256sum --quiet -c "${root}.ROOT_SHA256SUMS.sha256"

set +e
P2_OUTCOME_PREREQUISITE_SOURCE_RUN="$source_run" \
P2_OUTCOME_PREREQUISITE_ROOT="$root" \
P2_EXPECTED_LAUNCHER_SHA=1111111111111111111111111111111111111111 \
P2_EXPECTED_SOURCE_SHA=2222222222222222222222222222222222222222 \
P2_EXPECTED_CANDLE_SHA=3333333333333333333333333333333333333333 \
P2_EXPECTED_BINARY_SHA=4444444444444444444444444444444444444444444444444444444444444444 \
P2_OUTCOME_PREREQUISITE_DRY_RUN=1 \
  bash "$launcher" >"$fixture/reuse.stdout" 2>"$fixture/reuse.stderr"
reuse_rc=$?
set -e
[[ "$reuse_rc" == 2 ]]
grep -q 'sequence root or external seal already exists' "$fixture/reuse.stderr"

printf 'PASS: reliability parity, fresh evidence gate, provenance, seals, and root reuse rejection\n'
