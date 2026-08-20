#!/usr/bin/env bash
# Resume only the six Full V4 boundary evaluations after a failed evaluator.
set -euo pipefail

: "${OUTPUT_DIR:?set the existing failed Full V4 run root}"
: "${TOFY_BIN:?set the reviewed replacement evaluator binary}"
: "${P2_EXPECTED_SHA:?set the reviewed evaluator commit}"
: "${P2_EXPECTED_BINARY_SHA256:?set the reviewed evaluator binary SHA-256}"
: "${P2_REVIEWED_REMOTE_REF:?set the remote branch or tag containing the evaluator commit}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'refusing evaluation resume: checkout is not clean\n' >&2
  exit 2
fi
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'refusing evaluation resume: checkout %s != reviewed %s\n' \
    "$actual_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi
git fetch --no-tags origin "$P2_REVIEWED_REMOTE_REF"
remote_sha="$(git rev-parse FETCH_HEAD)"
if [[ "$remote_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'refusing evaluation resume: origin/%s resolves to %s, not reviewed %s\n' \
    "$P2_REVIEWED_REMOTE_REF" "$remote_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi
actual_binary_sha="$(sha256sum "$TOFY_BIN" | awk '{print $1}')"
if [[ "$actual_binary_sha" != "$P2_EXPECTED_BINARY_SHA256" ]]; then
  printf 'refusing evaluation resume: binary hash %s != reviewed %s\n' \
    "$actual_binary_sha" "$P2_EXPECTED_BINARY_SHA256" >&2
  exit 2
fi

test -d "$OUTPUT_DIR/checkpoints"
jq -e '.status == "completed" and .world_core_schema == "world_core_v4_full_training"' \
  "$OUTPUT_DIR/train_report.json" >/dev/null
previous_lifecycle="$(jq -er '.state' "$OUTPUT_DIR/lifecycle.json")"
if [[ "$previous_lifecycle" != failed_integrity_or_evaluation ]]; then
  printf 'refusing evaluation resume: lifecycle is %s, expected failed_integrity_or_evaluation\n' \
    "$previous_lifecycle" >&2
  exit 2
fi

OUTPUT_FILE_MANIFEST="${OUTPUT_DIR}.files.sha256"
OUTPUT_MANIFEST_DIGEST="${OUTPUT_FILE_MANIFEST}.sha256"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p -- "$LOG_DIR" "$OUTPUT_DIR/evaluations"
LOG_FILE="$LOG_DIR/eval-resume-$(date +%Y%m%dT%H%M%S).log"
resume_complete=0

write_lifecycle() {
  local state="$1"
  jq -n \
    --arg state "$state" \
    --arg timestamp "$(date -Iseconds)" \
    --arg evaluator_revision "$P2_EXPECTED_SHA" \
    '{state: $state, phase: "evaluation", timestamp: $timestamp,
      evaluator_revision: $evaluator_revision}' \
    >"$OUTPUT_DIR/lifecycle.json.tmp"
  mv -- "$OUTPUT_DIR/lifecycle.json.tmp" "$OUTPUT_DIR/lifecycle.json"
}

mark_failed_if_needed() {
  if [[ "$resume_complete" != 1 ]]; then
    write_lifecycle failed_integrity_or_evaluation
    rm -f -- \
      "$OUTPUT_FILE_MANIFEST" "$OUTPUT_MANIFEST_DIGEST" \
      "${OUTPUT_FILE_MANIFEST}.tmp.$$" "${OUTPUT_MANIFEST_DIGEST}.tmp.$$"
  fi
}
trap mark_failed_if_needed EXIT

source_run_revision="$(jq -er '.reviewed_commit' "$OUTPUT_DIR/launch-manifest.json")"
jq -n \
  --arg schema p2.full_v4_evaluation_resume.v1 \
  --arg timestamp "$(date -Iseconds)" \
  --arg source_run_revision "$source_run_revision" \
  --arg evaluator_revision "$P2_EXPECTED_SHA" \
  --arg evaluator_binary "$TOFY_BIN" \
  --arg evaluator_binary_sha256 "$actual_binary_sha" \
  --arg previous_lifecycle "$previous_lifecycle" \
  '{schema: $schema, timestamp: $timestamp, source_run_revision: $source_run_revision,
    evaluator_revision: $evaluator_revision, evaluator_binary: $evaluator_binary,
    evaluator_binary_sha256: $evaluator_binary_sha256,
    previous_lifecycle: $previous_lifecycle, physical_batch: 64,
    synthetic_episodes: 64, ood_seed: 1000002, iid_seed: 1000003,
    boundary_steps: [0, 8192, 16384, 20480, 24576, 28672]}' \
  >"$OUTPUT_DIR/evaluation-resume-manifest.json"
write_lifecycle running

run_evaluations() {
  local q step checkpoint bundle_config report
  local -a boundary_steps=(0 8192 16384 20480 24576 28672)
  q="$(jq -er '.q_mse_threshold' "$OUTPUT_DIR/config.json")"
  for step in "${boundary_steps[@]}"; do
    checkpoint="$OUTPUT_DIR/checkpoints/step-$(printf '%012d' "$step")/model.safetensors"
    bundle_config="$(dirname -- "$checkpoint")/config.json"
    report="$OUTPUT_DIR/evaluations/step-$(printf '%012d' "$step").json"
    test -s "$checkpoint"
    test -s "$bundle_config"
    test -s "$(dirname -- "$checkpoint")/bundle-manifest.json"
    if [[ -e "$report" || -e "$report.sha256" ]]; then
      printf 'refusing evaluation resume: report path already exists: %s\n' "$report" >&2
      return 2
    fi
    "$TOFY_BIN" p2-eval \
      --device cuda --physical-batch 64 \
      --checkpoint "$checkpoint" \
      --train-config "$bundle_config" \
      --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
      --seed 1000002 --iid-seed 1000003 \
      --q-mse-threshold "$q" \
      --output "$report"
    jq -e '
      .schema == "p2.eval_report.v15" and
      .research_claim == false and
      .synthetic_dynamics.semantic.schema == "p2.semantic_eval.v1" and
      .synthetic_iid_dynamics.semantic.schema == "p2.semantic_eval.v1" and
      .factual_branches.semantic_outcome_retrieval_n > 0 and
      .synthetic_dynamics.semantic_rollout.open["4"] != null and
      .synthetic_dynamics.semantic_rollout.open["8"] != null and
      .identity.checkpoint_sha256 != "" and
      .identity.population_sha256 != {} and
      .q_label_definition ==
        "exact_gameplay_pixels:overall>=0.99,changed>=0.90,status_row_excluded"
    ' "$report" >/dev/null
    sha256sum "$report" >"$report.sha256"
    sha256sum -c "$report.sha256" >/dev/null
  done
}

seal_tree() {
  local file_manifest_tmp="${OUTPUT_FILE_MANIFEST}.tmp.$$"
  local manifest_digest_tmp="${OUTPUT_MANIFEST_DIGEST}.tmp.$$"
  local file_manifest_tmp_abs
  file_manifest_tmp_abs="$(realpath -m "$file_manifest_tmp")"
  (
    cd -- "$OUTPUT_DIR"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum -c "$file_manifest_tmp_abs" >/dev/null
  ) >"$file_manifest_tmp"
  mv -- "$file_manifest_tmp" "$OUTPUT_FILE_MANIFEST"
  sha256sum "$OUTPUT_FILE_MANIFEST" >"$manifest_digest_tmp"
  sha256sum -c "$manifest_digest_tmp" >/dev/null
  mv -- "$manifest_digest_tmp" "$OUTPUT_MANIFEST_DIGEST"
}

printf '[%s] resuming Full V4 evaluations for %s (log: %s)\n' \
  "$(date -Iseconds)" "$OUTPUT_DIR" "$LOG_FILE"
run_evaluations 2>&1 | tee "$LOG_FILE"
test "${PIPESTATUS[0]}" -eq 0
write_lifecycle complete_pending_analysis
seal_tree
resume_complete=1
trap - EXIT
