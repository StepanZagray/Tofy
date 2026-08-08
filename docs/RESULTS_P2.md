# Results P2

P2 is implemented as a recursive latent world-model experiment. The completed
`readiness-v2` run is recorded below as a negative diagnostic result; implementation
smoke tests must not be promoted to research results.

## Frozen fields for the first experimental run

Before inspecting a full run, record here:

- training and held-out seeds;
- curriculum lesson lengths;
- model dimensions and recursion schedule;
- physical batch size and gradient accumulation;
- optimizer, learning rate, and SIGReg/event/Q weights;
- PTRM `K` values and latent-noise sweep;
- checkpoint-selection metric;
- exact train/evaluation commands.

## Readiness training

VRAM/GPU/Muon integration landed 2026-08-05:

- **Batch:** physical `128` × grad_accum `4` (effective 512) on RTX 5060 8GB (~6.4 GiB peak). Resume may migrate microbatch schedule when effective batch stays 512 (e.g. `512×1` on L40S).
- **Input:** palette index tensors + `pixel_emb` (replaces 16-ch one-hot staging)
- **SIGReg:** spatial with row subsample cap `4096`
- **Optimizer:** DeepSeek-V4 hybrid — Muon on hidden conv/proj weights (≥2×2); AdamW on embeddings, auxiliary heads (`event_head`, `q_head`, …), biases
- **Pipeline:** batch prefetch on shuffled episodes + GPU-side grad clip

```bash
bash scripts/p2_readiness_train.sh run
# default output: runs/p2/readiness-v2 (override: P2_OUTPUT_DIR=...)
```

Resume checkpoint: `runs/p2/readiness-v2/checkpoints/` (see `latest.json`).

### Readiness-v2 action-conditioning diagnostic (2026-08-07)

The report-schema-v8 action shuffle holds frames, next-frame targets, and goals fixed
while deranging the complete action tuple (including ACTION6 coordinates) within each
curriculum source. A ratio `shuffled MSE / true-action MSE <= 1.1` is
action-marginalized. Every source with at least two distinct action
conditionings failed:

| Source | Shuffle ratio | Coverage note |
|--------|--------------:|---------------|
| dynamics aggregate | 1.0023 | all ACTION1..7 represented |
| `random_one_step` | 1.0078 | ACTION5=25%, ACTION6=25%; 127 distinct coordinates |
| `exploration` | 1.0001 | ACTION1..4,7; 40.2% no-op |
| `hazard_one_step` | 1.0012 | ACTION3/4 only |
| planner aggregate | 1.0005 | ACTION2=60.0% |
| `sequential` | 0.9999 | ACTION1..4 |
| `hypothesis_probe` | 1.0023 | ACTION2=62.0% |
| `p1c_falsification` | n/a | ACTION2 only; shuffling cannot intervene |
| `p1c_hard_retarget` | 1.0018 | ACTION1..4 |

This confirms action-marginalized dynamics, but does not support narrow planner
coverage as the sole cause: the deliberately broad `random_one_step` source also
fails by a wide margin. The live differentiable target-encoder collapse hypothesis
therefore moves ahead of curriculum coverage for the next A/B.

```bash
cargo run --release --features cudnn -- p2-eval \
  --checkpoint runs/p2/readiness-v2/checkpoints/step-000000028672/model.safetensors \
  --train-config runs/p2/readiness-v2/config.json \
  --seed 2 --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1 --ptrm-noise 0.1 --q-mse-threshold 0.05 --ensemble-members 1 \
  --output runs/p2/readiness-v2/eval_report_action_shuffle.json
```

### Readiness-v3 incomplete NaN run (2026-08-08)

The L40S run at commit `2ae5a3ac` used physical batch `1024` and
gradient accumulation `1`. It started at `2026-08-07T13:12:39Z` and exited at
`2026-08-08T08:11:00Z` with `Error: total is not finite: NaN`. The last durable
checkpoint is step `18,500 / 28,672` (64.5%), at sequential step `2,116 / 4,096`;
the failure occurred before the step-19,000 checkpoint. There is no completed
`train_report.json`, synthetic eval report, or ARC-AGI-3 eval report.

Every saved checkpoint reports SIGReg exactly at its `10,000` clamp, including the
first 500 steps. At step 18,500 the sequential lesson's accumulated means were
`total=34.5368`, `next_latent=0.04187`, `rollout=0.01722`, `sigreg=10000`, and
`prefix=0.01066`. The saturated SIGReg term contributed a constant `30` to total
loss and did not provide a useful training signal. The terminal component cannot be
identified from the captured error: the trainer checks `total` before its named
constituents, and multi-horizon prefix plus PTRM rank are not reported separately.

```bash
P2_DEVICE=cuda \
P2_OUTPUT_DIR=runs/p2/readiness-v3 \
P2_PHYSICAL_BATCH=1024 \
P2_GRAD_ACCUM=1 \
RAYON_NUM_THREADS=64 \
TOFY_P2_PREFETCH_WORKERS=32 \
TOFY_P2_PREFETCH_LOOKAHEAD=16 \
TOFY_P2_PREFETCH_QUEUE_DEPTH=32 \
MAX_REPAIR_ATTEMPTS=0 \
bash scripts/p2_readiness_train.sh run
```

#### Readiness-v3 repair evidence

Fast CPU regressions isolated the stability chain before resume:

- the 16-step prefix recurrence reached RMS `4083.7` and `19335.5` instead of
  remaining on the encoder's unit-RMS latent support;
- `clip_gradients_gpu` returned success for a NaN global norm, allowing a bad
  gradient to reach the optimizer;
- the hard SIGReg clamp had exactly zero gradient above `10,000`; and
- aggregate `total` was checked before its constituents, while Q surprise, PTRM rank,
  and multi-horizon prefix were not named separately.

The repair unit-normalizes every prefix prediction, uses bounded smooth losses that
retain gradients, rejects non-finite gradient norms before the optimizer, and checks
every named constituent before `total`. All four regression tests fail on the old
behavior and pass after the repair. A one-episode CUDA eval also loaded the step-18,500
checkpoint successfully (`dyn_mse=0.000574`, `plan_mse=0.000471`), confirming the
durable checkpoint itself is finite.

```bash
cargo test --lib p2::model::tests::prefix_rollout_stays_on_unit_rms_latent_support -- --exact
cargo test --lib p2::optimizer::tests::gradient_clip_rejects_non_finite_norm -- --exact
cargo test --lib p2::train::tests::sigreg_cap_retains_gradient_above_reported_limit -- --exact
cargo test --lib p2::train::tests::loss_check_reports_constituent_before_non_finite_total -- --exact
```

## Best So Far

**Rollout dynamics (held-out synthetic, 64 episodes, eval v3):**

| Metric | Value | Run |
|--------|-------|-----|
| rollout MSE @ 8 | **0.17** | `p2-output-v11-control` |
| rollout MSE @ 4 | **0.084** | `p2-output-v11-control` |
| one-step latent MSE | **0.026** | `p2-output-v11-control` |
| events accuracy | 0.906 | `p2-output-v11-control` |

v11-control replayed the v8 curriculum (no exploration, `q_mse_threshold=0.25`) on
pre-v12 architecture; it restored multi-step rollout after v9/v10 Stage 1b collapse.

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 32 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --lessons dynamics,sequential,q_calibration,falsification,retarget \
  --q-mse-threshold 0.25 --checkpoint-every-steps 100 \
  --output-dir p2-output-v11-control

cargo run --release --features cudnn -- p2-eval \
  --checkpoint p2-output-v11-control/model.safetensors \
  --train-config p2-output-v11-control/config.json \
  --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --q-mse-threshold 0.25 \
  --output p2-output-v11-control/eval_report_64ep_v3.json
```

**In progress:** v12 architecture (dual TRM blocks, dual-pool encoder, delta/stop-grad
defaults) + 7-experiment chain — `scripts/p2_experiment_chain.sh run` →
`p2-output-v12/`. See `docs/P2_V12.md`.

## Implementation validation (not a result)

The CPU smoke path completed the four ordered lessons, wrote a safetensors
checkpoint/config/report plus a `candle-graph/runtime/1` trace, evaluated held-out
synthetic transitions with deterministic `K=1` and stochastic `K=2`, and imported
the trace into `candle_graph`. The smoke used physical batch `2` and gradient
accumulation `1`; it is not the required accelerator batch-capacity measurement for
the first real run.

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --steps-per-lesson 1 --physical-batch 2 \
  --hidden-dim 16 --action-dim 4 \
  --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 4 --sigreg-knots 3 \
  --output-dir /tmp/tofy-p2-final-smoke-v3

cargo run --release -- p2-eval \
  --checkpoint /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  --train-config /tmp/tofy-p2-final-smoke-v3/config.json \
  --synthetic-episodes 1 --physical-batch 2 \
  --ptrm-k 1,2 --ptrm-noise 0.1 \
  --output /tmp/tofy-p2-final-smoke-v3/eval.json

scripts/audit_p2.sh \
  /tmp/tofy-p2-final-smoke-v3/analyzer \
  /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  /tmp/tofy-p2-final-smoke-v3/runtime.json
```

All generated reports set `research_claim=false`; without `--scorecard-json` the local
evaluator leaves `official_rhae=null` and records `public_data_used_for_fitting=false`.
Pass `--scorecard-json` with a closed official scorecard to populate RHAE per
https://docs.arcprize.org/methodology .

Exact pause/resume was added and validated without recording a model-quality result.
The equivalence test compares an uninterrupted two-update run with a one-update
pause plus resume and requires exact equality of every final model tensor, both
named AdamW moment tensors, the optimizer/global step, curriculum cursor, and lesson
metrics. Separate checks reject a changed training contract and a missing optimizer
file. A command-line smoke paused at step 1, resumed through the `checkpoints`
directory, completed at step 2, and passed the external analyzer audit. A separate
PTY smoke sent an actual `SIGINT`; the trainer finished its in-flight update, wrote
`step-000000000012`, reported `status=Paused`, and exited with code 0. Resuming that
run directory advanced cleanly to a new `step-000000000013` bundle.

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings

cargo run -- p2-train \
  --lessons dynamics --steps-per-lesson 2 --physical-batch 2 \
  --hidden-dim 8 --action-dim 4 --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 2 --sigreg-knots 3 \
  --checkpoint-every-steps 0 --max-steps-this-run 1 \
  --output-dir /tmp/tofy-p2-resume-smoke.2y6nB5

cargo run -- p2-train \
  --lessons dynamics --steps-per-lesson 2 --physical-batch 2 \
  --hidden-dim 8 --action-dim 4 --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 2 --sigreg-knots 3 \
  --checkpoint-every-steps 0 \
  --output-dir /tmp/tofy-p2-resume-smoke.2y6nB5 \
  --resume /tmp/tofy-p2-resume-smoke.2y6nB5/checkpoints

scripts/audit_p2.sh \
  /tmp/tofy-p2-resume-smoke.2y6nB5/analyzer \
  /tmp/tofy-p2-resume-smoke.2y6nB5/model.safetensors \
  /tmp/tofy-p2-resume-smoke.2y6nB5/runtime.json
```

### Training throughput (not a quality result)

Backend choice is measurement-driven: use `--features cudnn` only while the vendored
candle patches make it faster than `--features cuda`. Stock candle cudnn regresses
conv backward; with `vendor/candle-core` (real `ConvBackwardFilter`/`Data` + skip
dead leaf input grads) it wins.

Same dynamics microbench (hidden 128, action 32, physical batch 1024, RTX 5060
Laptop, steps 11–15 steady after fused 2B encoder):

| build | steady ms/step | forward | backward |
|-------|----------------|---------|----------|
| `--features cuda` | ~327 | ~71 | ~228 |
| `--features cudnn` + patches | ~120 | ~24 | ~69 |

Layer microbench (`tests/cuda_conv_probe.rs`, encoder c1 leaf input): full fwd+bwd
~79 ms (cuda) → ~28 ms (cudnn+patches); patched vs rewrite weight-grad max abs
diff `1.4e-5`.

```bash
scripts/p2_bench_backends.sh
cargo test --release --features cudnn --test cuda_conv_probe -- --ignored --nocapture
TOFY_P2_STEP_PROFILE=5 cargo run --release --features cudnn -- p2-train \
  --device cuda --lessons dynamics --steps-per-lesson 15 \
  --physical-batch 1024 --hidden-dim 128 --action-dim 32 \
  --checkpoint-every-steps 0 --output-dir /tmp/tofy-p2-cudnn-profile
```

Perf tooling: `TOFY_P2_STEP_PROFILE=N` (phase ms), `TOFY_PERF_TRACE=path` with
`--features profiling` (Chrome/Perfetto), and NVIDIA Nsight Systems (`nsys`) for
unified CPU+GPU timelines on CUDA. See `src/perf.rs`.

## Update rule

When a P2 metric is reported, include the exact command and separate synthetic
oracle-normalized efficiency from official ARC-AGI-3 RHAE. Public ARC games are
held-out transfer evaluation and must not be used for checkpoint selection or
hyperparameter tuning.
