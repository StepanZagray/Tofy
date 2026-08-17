#!/usr/bin/env bash
# Full V4 ARC-AGI-3 aligned train → eval.
#
# Usage (after measuring the stable batch on this exact hardware):
#   OUTPUT_DIR=runs/p2/full-v4-<unique> PHYSICAL_BATCH=<measured> \
#   P2_BATCH_MEASURED=1 P2_EXPECTED_SHA=<reviewed> \
#   P2_BATCH_MEASUREMENT_EVIDENCE=<file> \
#   P2_REVIEWED_REMOTE_REF=<branch-or-tag> \
#   P2_EXPECTED_BINARY_SHA256=<reviewed> scripts/p2_arc3_train_eval.sh
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_root"

: "${OUTPUT_DIR:?set a new never-reused Full V4 output root}"
: "${PHYSICAL_BATCH:?set the largest stable physical batch measured on this hardware}"
: "${P2_EXPECTED_SHA:?set the reviewed Tofy commit}"
: "${P2_EXPECTED_BINARY_SHA256:?set the reviewed locked CUDA binary SHA-256}"
: "${P2_BATCH_MEASUREMENT_EVIDENCE:?set the batch-capacity evidence file}"
: "${P2_REVIEWED_REMOTE_REF:?set the remote branch or tag containing the reviewed commit}"
if [[ "$OUTPUT_DIR" == */ ]]; then
  printf 'refusing launch: OUTPUT_DIR must not end with a slash\n' >&2
  exit 2
fi
if [[ "${P2_BATCH_MEASURED:-0}" != 1 ]]; then
  printf 'refusing launch: set P2_BATCH_MEASURED=1 only after a batch-capacity measurement\n' >&2
  exit 2
fi
if [[ ! -s "$P2_BATCH_MEASUREMENT_EVIDENCE" ]]; then
  printf 'refusing launch: batch-capacity evidence is missing or empty: %s\n' \
    "$P2_BATCH_MEASUREMENT_EVIDENCE" >&2
  exit 2
fi
DEVICE="${DEVICE:-cuda}"
STEPS_PER_LESSON="${STEPS_PER_LESSON:-4096}"
if [[ "$STEPS_PER_LESSON" != 4096 ]]; then
  printf 'refusing launch: Full V4 diagnostic contract requires STEPS_PER_LESSON=4096\n' >&2
  exit 2
fi
TRAIN_SEED="${TRAIN_SEED:-2}"
INIT_SEED="${INIT_SEED:-2}"
OOD_EVAL_SEED="${OOD_EVAL_SEED:-1000002}"
IID_EVAL_SEED="${IID_EVAL_SEED:-1000003}"
P2_PREFLIGHT_EVAL_BATCH="${P2_PREFLIGHT_EVAL_BATCH:-64}"
PREFLIGHT_DIR="${OUTPUT_DIR}.preflight"
OUTPUT_FILE_MANIFEST="${OUTPUT_DIR}.files.sha256"
OUTPUT_MANIFEST_DIGEST="${OUTPUT_FILE_MANIFEST}.sha256"
PREFLIGHT_FILE_MANIFEST="${PREFLIGHT_DIR}.files.sha256"
PREFLIGHT_MANIFEST_DIGEST="${PREFLIGHT_FILE_MANIFEST}.sha256"
if [[ -e "$OUTPUT_DIR" || -e "$PREFLIGHT_DIR" || \
      -e "$OUTPUT_FILE_MANIFEST" || -e "$OUTPUT_MANIFEST_DIGEST" || \
      -e "$PREFLIGHT_FILE_MANIFEST" || -e "$PREFLIGHT_MANIFEST_DIGEST" ]]; then
  printf 'refusing launch: output or preflight root already exists\n' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'refusing launch: checkout is not clean\n' >&2
  exit 2
fi
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'refusing launch: checkout %s != reviewed %s\n' "$actual_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi
git fetch --no-tags origin "$P2_REVIEWED_REMOTE_REF"
remote_sha="$(git rev-parse FETCH_HEAD)"
if [[ "$remote_sha" != "$P2_EXPECTED_SHA" ]]; then
  printf 'refusing launch: origin/%s resolves to %s, not reviewed %s\n' \
    "$P2_REVIEWED_REMOTE_REF" "$remote_sha" "$P2_EXPECTED_SHA" >&2
  exit 2
fi

cargo build --release --locked --features cudnn
TOFY_BIN="${TOFY_BIN:-$repo_root/target/release/tofy}"
actual_binary_sha="$(sha256sum "$TOFY_BIN" | awk '{print $1}')"
if [[ "$actual_binary_sha" != "$P2_EXPECTED_BINARY_SHA256" ]]; then
  printf 'refusing launch: binary hash %s != reviewed %s\n' \
    "$actual_binary_sha" "$P2_EXPECTED_BINARY_SHA256" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'refusing launch: nvidia-smi is unavailable for accelerator provenance\n' >&2
  exit 2
fi
gpu_identity="$(nvidia-smi \
  --query-gpu=uuid,name,memory.total,driver_version \
  --format=csv,noheader)"
if [[ -z "$gpu_identity" ]]; then
  printf 'refusing launch: accelerator identity query returned no devices\n' >&2
  exit 2
fi

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p -- "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline-$(date +%Y%m%dT%H%M%S).log"
BATCH_EVIDENCE_COPY="$OUTPUT_DIR/batch-measurement-evidence"
cp -- "$P2_BATCH_MEASUREMENT_EVIDENCE" "$BATCH_EVIDENCE_COPY"
batch_evidence_sha="$(sha256sum "$BATCH_EVIDENCE_COPY" | awk '{print $1}')"
pipeline_complete=0

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

write_lifecycle() {
  local root="$1"
  local state="$2"
  [[ -d "$root" ]] || return 0
  jq -n --arg state "$state" --arg timestamp "$(date -Iseconds)" \
    '{state: $state, timestamp: $timestamp}' >"$root/lifecycle.json.tmp"
  mv -- "$root/lifecycle.json.tmp" "$root/lifecycle.json"
}

seal_tree() {
  local root="$1"
  local file_manifest="$2"
  local manifest_digest="$3"
  local file_manifest_tmp="${file_manifest}.tmp.$$"
  local manifest_digest_tmp="${manifest_digest}.tmp.$$"
  local file_manifest_tmp_abs
  file_manifest_tmp_abs="$(realpath -m "$file_manifest_tmp")"
  (
    cd -- "$root"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum -c "$file_manifest_tmp_abs" >/dev/null
  ) >"$file_manifest_tmp"
  mv -- "$file_manifest_tmp" "$file_manifest"
  sha256sum "$file_manifest" >"$manifest_digest_tmp"
  sha256sum -c "$manifest_digest_tmp" >/dev/null
  mv -- "$manifest_digest_tmp" "$manifest_digest"
}

mark_failed_if_needed() {
  local preflight_file_manifest_abs
  preflight_file_manifest_abs="$(realpath -m "$PREFLIGHT_FILE_MANIFEST")"
  if [[ "$pipeline_complete" != 1 ]]; then
    if jq -e '.state == "implementation_smoke_complete"' \
      "$PREFLIGHT_DIR/lifecycle.json" >/dev/null 2>&1 && \
      [[ -s "$PREFLIGHT_FILE_MANIFEST" && -s "$PREFLIGHT_MANIFEST_DIGEST" ]] && \
      sha256sum -c "$PREFLIGHT_MANIFEST_DIGEST" >/dev/null 2>&1 && \
      (cd -- "$PREFLIGHT_DIR" && \
        sha256sum -c "$preflight_file_manifest_abs" >/dev/null 2>&1); then
      :
    else
      write_lifecycle "$PREFLIGHT_DIR" failed_integrity_or_evaluation
      rm -f -- "$PREFLIGHT_FILE_MANIFEST" "$PREFLIGHT_MANIFEST_DIGEST"
    fi
    write_lifecycle "$OUTPUT_DIR" failed_integrity_or_evaluation
    rm -f -- \
      "$OUTPUT_FILE_MANIFEST" "$OUTPUT_MANIFEST_DIGEST" \
      "${OUTPUT_FILE_MANIFEST}.tmp.$$" "${OUTPUT_MANIFEST_DIGEST}.tmp.$$" \
      "${PREFLIGHT_FILE_MANIFEST}.tmp.$$" "${PREFLIGHT_MANIFEST_DIGEST}.tmp.$$"
  fi
}
trap mark_failed_if_needed EXIT

run_pipeline() {
  "$TOFY_BIN" p2-train \
    --recipe full-v4 --device "$DEVICE" \
    --seed "$TRAIN_SEED" --init-seed "$INIT_SEED" \
    --physical-batch "$PHYSICAL_BATCH" --grad-accum 1 \
    --steps-per-lesson 2 --checkpoint-every-steps 0 \
    --profile-update 999999 --output-dir "$PREFLIGHT_DIR"
  jq -e '
    .status == "completed" and
    .world_core_schema == "world_core_v4_full_training" and
    ([.lessons[].lesson] == ["dynamics","exploration","sequential","q_calibration","falsification"])
  ' "$PREFLIGHT_DIR/train_report.json" >/dev/null
  local preflight_q preflight_checkpoint_dir
  preflight_q="$(jq -er '.q_mse_threshold' "$PREFLIGHT_DIR/config.json")"
  preflight_checkpoint_dir="$PREFLIGHT_DIR/checkpoints/$(jq -er '.directory' "$PREFLIGHT_DIR/checkpoints/latest.json")"
  "$TOFY_BIN" p2-eval \
    --device "$DEVICE" --physical-batch "$P2_PREFLIGHT_EVAL_BATCH" \
    --checkpoint "$preflight_checkpoint_dir/model.safetensors" \
    --train-config "$preflight_checkpoint_dir/config.json" \
    --synthetic-episodes 1 --ptrm-k 1 --ensemble-members 1 \
    --seed "$OOD_EVAL_SEED" --iid-seed "$IID_EVAL_SEED" \
    --q-mse-threshold "$preflight_q" \
    --output "$PREFLIGHT_DIR/eval_report.json"
  jq -e '
    .schema == "p2.eval_report.v15" and
    .research_claim == false and
    .factual_branches.semantic_outcome_retrieval_n > 0 and
    .synthetic_dynamics.semantic_rollout.open["4"] != null and
    .synthetic_dynamics.semantic_rollout.open["8"] != null and
    .q_label_definition ==
      "exact_gameplay_pixels:overall>=0.99,changed>=0.90,status_row_excluded"
  ' "$PREFLIGHT_DIR/eval_report.json" >/dev/null

  local preflight_train_sha preflight_eval_sha rustc_version cargo_version
  preflight_train_sha="$(sha256sum "$PREFLIGHT_DIR/train_report.json" | awk '{print $1}')"
  preflight_eval_sha="$(sha256sum "$PREFLIGHT_DIR/eval_report.json" | awk '{print $1}')"
  rustc_version="$(rustc --version)"
  cargo_version="$(cargo --version)"
  jq -n \
    --arg schema p2.full_v4_launch_manifest.v1 \
    --arg created_at "$(date -Iseconds)" \
    --arg reviewed_commit "$P2_EXPECTED_SHA" \
    --arg remote_ref "$P2_REVIEWED_REMOTE_REF" \
    --arg binary "$TOFY_BIN" \
    --arg binary_sha256 "$actual_binary_sha" \
    --arg batch_evidence "$BATCH_EVIDENCE_COPY" \
    --arg batch_evidence_sha256 "$batch_evidence_sha" \
    --arg gpu_identity "$gpu_identity" \
    --arg cuda_visible_devices "${CUDA_VISIBLE_DEVICES:-unset}" \
    --arg device "$DEVICE" \
    --arg rustc "$rustc_version" \
    --arg cargo "$cargo_version" \
    --arg preflight_train_sha256 "$preflight_train_sha" \
    --arg preflight_eval_sha256 "$preflight_eval_sha" \
    --argjson physical_batch "$PHYSICAL_BATCH" \
    --argjson preflight_eval_batch "$P2_PREFLIGHT_EVAL_BATCH" \
    --argjson steps_per_lesson "$STEPS_PER_LESSON" \
    --argjson train_seed "$TRAIN_SEED" \
    --argjson init_seed "$INIT_SEED" \
    --argjson ood_eval_seed "$OOD_EVAL_SEED" \
    --argjson iid_eval_seed "$IID_EVAL_SEED" \
    '{
      schema: $schema,
      created_at: $created_at,
      reviewed_commit: $reviewed_commit,
      reviewed_commit_fetch_verified: true,
      remote: "origin",
      remote_ref: $remote_ref,
      build: {
        command: "cargo build --release --locked --features cudnn",
        features: ["cudnn"],
        binary: $binary,
        binary_sha256: $binary_sha256,
        rustc: $rustc,
        cargo: $cargo
      },
      training: {
        recipe: "full-v4",
        device: $device,
        physical_batch: $physical_batch,
        grad_accum: 1,
        steps_per_lesson: $steps_per_lesson,
        train_seed: $train_seed,
        init_seed: $init_seed,
        ood_eval_seed: $ood_eval_seed,
        iid_eval_seed: $iid_eval_seed
      },
      batch_measurement: {
        attested: true,
        evidence_copy: $batch_evidence,
        evidence_sha256: $batch_evidence_sha256
      },
      hardware: {
        nvidia_smi: $gpu_identity,
        cuda_visible_devices: $cuda_visible_devices
      },
      preflight: {
        train_report_sha256: $preflight_train_sha256,
        eval_report_sha256: $preflight_eval_sha256,
        eval_physical_batch: $preflight_eval_batch
      }
    }' >"$OUTPUT_DIR/launch-manifest.json.tmp"
  mv -- "$OUTPUT_DIR/launch-manifest.json.tmp" "$OUTPUT_DIR/launch-manifest.json"
  write_lifecycle "$PREFLIGHT_DIR" implementation_smoke_complete
  seal_tree "$PREFLIGHT_DIR" "$PREFLIGHT_FILE_MANIFEST" "$PREFLIGHT_MANIFEST_DIGEST"
  write_lifecycle "$OUTPUT_DIR" running

  local -a train=(
    "$TOFY_BIN" p2-train
    --recipe full-v4
    --device "$DEVICE"
    --seed "$TRAIN_SEED" --init-seed "$INIT_SEED"
    --physical-batch "$PHYSICAL_BATCH" --grad-accum 1
    --steps-per-lesson "$STEPS_PER_LESSON"
    --checkpoint-every-steps 100
    --output-dir "$OUTPUT_DIR"
  )
  "${train[@]}"
  jq -e '
    .status == "completed" and
    .world_core_schema == "world_core_v4_full_training" and
    ([.lessons[].lesson] == ["dynamics","exploration","sequential","q_calibration","falsification"])
  ' "$OUTPUT_DIR/train_report.json" >/dev/null
  local sequential_manifest="$OUTPUT_DIR/checkpoints/step-000000020480/bundle-manifest.json"
  local q_manifest="$OUTPUT_DIR/checkpoints/step-000000024576/bundle-manifest.json"
  local falsification_manifest="$OUTPUT_DIR/checkpoints/step-000000028672/bundle-manifest.json"
  local sequential_frozen q_frozen falsification_frozen
  sequential_frozen="$(jq -cS '.parameter_groups | del(.observers)' "$sequential_manifest")"
  q_frozen="$(jq -cS '.parameter_groups | del(.observers)' "$q_manifest")"
  falsification_frozen="$(jq -cS '.parameter_groups | del(.observers)' "$falsification_manifest")"
  if [[ "$sequential_frozen" != "$q_frozen" || "$q_frozen" != "$falsification_frozen" ]]; then
    printf 'observer-stage invariant failed: a frozen parameter group changed after update 20480\n' >&2
    exit 1
  fi
  jq -n \
    --arg schema p2.observer_freeze_verification.v1 \
    --argjson frozen_parameter_groups "$q_frozen" \
    --arg sequential_manifest_sha256 "$(sha256sum "$sequential_manifest" | awk '{print $1}')" \
    --arg q_manifest_sha256 "$(sha256sum "$q_manifest" | awk '{print $1}')" \
    --arg falsification_manifest_sha256 "$(sha256sum "$falsification_manifest" | awk '{print $1}')" \
    '{schema: $schema, verified: true, invariant: "all non-observer parameter groups unchanged at 20480/24576/28672", frozen_parameter_groups: $frozen_parameter_groups, manifests: [$sequential_manifest_sha256, $q_manifest_sha256, $falsification_manifest_sha256]}' \
    >"$OUTPUT_DIR/observer-freeze-verification.json"
  local q
  q="$(jq -er '.q_mse_threshold' "$OUTPUT_DIR/config.json")"
  local eval_dir="$OUTPUT_DIR/evaluations"
  mkdir -p -- "$eval_dir"
  local -a boundary_steps=(0 8192 16384 20480 24576 28672)
  local step checkpoint bundle_config report
  for step in "${boundary_steps[@]}"; do
    checkpoint="$OUTPUT_DIR/checkpoints/step-$(printf '%012d' "$step")/model.safetensors"
    bundle_config="$(dirname -- "$checkpoint")/config.json"
    report="$eval_dir/step-$(printf '%012d' "$step").json"
    test -s "$checkpoint"
    test -s "$bundle_config"
    test -s "$(dirname -- "$checkpoint")/bundle-manifest.json"
    "$TOFY_BIN" p2-eval \
      --device "$DEVICE" --physical-batch 64 \
      --checkpoint "$checkpoint" \
      --train-config "$bundle_config" \
      --synthetic-episodes 64 --ptrm-k 1,2,4,8 \
      --seed "$OOD_EVAL_SEED" --iid-seed "$IID_EVAL_SEED" \
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
  write_lifecycle "$OUTPUT_DIR" complete_pending_analysis
}

log "ARC-AGI-3 Full V4 train+eval → $OUTPUT_DIR (log: $LOG_FILE)"
run_pipeline 2>&1 | tee "$LOG_FILE"
test "${PIPESTATUS[0]}" -eq 0
seal_tree "$OUTPUT_DIR" "$OUTPUT_FILE_MANIFEST" "$OUTPUT_MANIFEST_DIGEST"
pipeline_complete=1
trap - EXIT
