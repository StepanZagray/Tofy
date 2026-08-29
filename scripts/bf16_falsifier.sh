#!/usr/bin/env bash
# Staged recurrent-core BF16 falsifier. RUN_DIR is read-only; evidence is
# written to its sibling bf16-falsifier directory.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_DIR="${1:?usage: scripts/bf16_falsifier.sh RUN_DIR}"
RUN_PARENT="$(dirname -- "$RUN_DIR")"
OUT_DIR="${BF16_FALSIFIER_OUT:-$RUN_PARENT/bf16-falsifier}"
DEVICE="${DEVICE:-cuda}"
TOFY_BIN="${TOFY_BIN:-target/release/tofy}"
KERNEL_PROBE_BIN="${BF16_KERNEL_PROBE_BIN:-target/release/examples/conv_kernel_probe}"
DRIFT_BATCH="${BF16_DRIFT_BATCH:-128}"
WARMUP_UPDATES="${BF16_WARMUP_UPDATES:-20}"
MEASURED_UPDATES="${BF16_MEASURED_UPDATES:-100}"

if [[ ! -x "$TOFY_BIN" ]]; then
  printf 'FAIL: Tofy binary is missing or not executable: %s\n' "$TOFY_BIN" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/config.json" ]]; then
  printf 'FAIL: training config is missing: %s/config.json\n' "$RUN_DIR" >&2
  exit 1
fi

CHECKPOINT="$RUN_DIR/checkpoints/best/ema.safetensors"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/model.safetensors"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  printf 'FAIL: neither best EMA nor fallback model checkpoint exists under %s\n' "$RUN_DIR" >&2
  exit 1
fi
if [[ -e "$OUT_DIR" ]]; then
  printf 'FAIL: refusing to reuse falsifier evidence root: %s\n' "$OUT_DIR" >&2
  exit 1
fi
mkdir -p -- "$OUT_DIR"

KERNEL_STATUS=SKIP
if [[ -x "$KERNEL_PROBE_BIN" ]]; then
  if "$KERNEL_PROBE_BIN" >"$OUT_DIR/kernel-probe.log" 2>&1; then
    KERNEL_STATUS=PASS
  else
    KERNEL_STATUS=FAIL
  fi
else
  printf 'probe missing, skipped: %s\n' "$KERNEL_PROBE_BIN" | tee "$OUT_DIR/kernel-probe.log"
fi
printf 'KERNEL-PROBE %s\n' "$KERNEL_STATUS"

"$TOFY_BIN" p2-bf16-drift \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --train-config "$RUN_DIR/config.json" \
  --batch-size "$DRIFT_BATCH" \
  --output "$OUT_DIR/drift.json" \
  | tee "$OUT_DIR/drift.log"

if jq -e '
  (.latent_max_abs_drift >= 0) and
  (.logit_max_abs_drift >= 0) and
  (.changed_pixel_prediction_flip_rate >= 0) and
  (.changed_pixel_prediction_flip_rate <= 1) and
  (.composed_decode_flip_rate >= 0) and
  (.composed_decode_flip_rate <= 1) and
  (.f32_rollout_loss > 0) and
  (.bf16_rollout_loss > 0) and
  (.f32_rollout_fragments >= 16) and
  (.bf16_rollout_fragments >= 16)
' "$OUT_DIR/drift.json" >/dev/null; then
  DRIFT_STATUS=PASS
else
  DRIFT_STATUS=FAIL
fi
printf 'DRIFT/H2 %s latent_max=%s logit_max=%s changed_flip_rate=%s composed_flip_rate=%s\n' \
  "$DRIFT_STATUS" \
  "$(jq -r '.latent_max_abs_drift' "$OUT_DIR/drift.json")" \
  "$(jq -r '.logit_max_abs_drift' "$OUT_DIR/drift.json")" \
  "$(jq -r '.changed_pixel_prediction_flip_rate' "$OUT_DIR/drift.json")" \
  "$(jq -r '.composed_decode_flip_rate' "$OUT_DIR/drift.json")"

"$TOFY_BIN" p2-bf16-bench \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --train-config "$RUN_DIR/config.json" \
  --warmup-updates "$WARMUP_UPDATES" \
  --measured-updates "$MEASURED_UPDATES" \
  --output "$OUT_DIR/benchmark.json" \
  | tee "$OUT_DIR/benchmark.log"

SPEEDUP="$(jq -r '.speedup' "$OUT_DIR/benchmark.json")"
if jq -e '.warmup_updates >= 20 and .measured_updates >= 100 and .speedup >= 1.20' \
  "$OUT_DIR/benchmark.json" >/dev/null; then
  THROUGHPUT_STATUS=PASS
else
  THROUGHPUT_STATUS=FAIL
fi
printf 'THROUGHPUT %s speedup=%sx threshold=1.20x\n' "$THROUGHPUT_STATUS" "$SPEEDUP"
printf 'QUALITY NEEDS-3-SEED (no material-regression claim is made by frozen drift or timing)\n'

OVERALL=PASS
if [[ "$KERNEL_STATUS" == FAIL || "$DRIFT_STATUS" != PASS || "$THROUGHPUT_STATUS" != PASS ]]; then
  OVERALL=FAIL
fi
jq -n \
  --arg schema 'p2.bf16_recurrent_core_falsifier.v1' \
  --arg run_dir "$RUN_DIR" \
  --arg checkpoint "$CHECKPOINT" \
  --arg kernel "$KERNEL_STATUS" \
  --arg drift "$DRIFT_STATUS" \
  --arg throughput "$THROUGHPUT_STATUS" \
  --arg quality 'NEEDS-3-SEED' \
  --arg overall "$OVERALL" \
  --argjson speedup "$SPEEDUP" \
  '{schema: $schema, source_run: $run_dir, checkpoint: $checkpoint,
    kernel_probe: $kernel, drift_h2: $drift, throughput: $throughput,
    quality: $quality, speedup: $speedup, overall: $overall}' \
  >"$OUT_DIR/summary.json"
printf 'BF16-FALSIFIER %s evidence=%s\n' "$OVERALL" "$OUT_DIR"

[[ "$OVERALL" == PASS ]]
