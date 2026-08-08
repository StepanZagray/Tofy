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

## SIGReg/action-conditioning A/B v1 (experimental; not a positive claim)

The controlled experiment is preregistered under
`runs/p2/ab-sigreg-action-v1/` and run by
`scripts/p2_sigreg_action_ab.sh`. Both arms use fresh initialization, physical
batch `1024`, accumulation `1`, dynamics only, shuffled episodes, randomized
recursion up to `8` outer / `2` inner steps, final-outer-only supervision, seed-fixed
data order, and checkpoints every 500 updates. Event, Q, reliability, prefix,
rollout, PTRM, ensemble, and live-policy signals are disabled or ignored for this
representation experiment. Held-out evaluation uses 64 synthetic episodes,
deterministic `K=1`, ensemble `1`, and evaluation seed `424242` at updates 1,000
and 2,000.

- `control` retains the existing RMS-normalized, 2×2 spatial-pooled cell-vector
  SIGReg path unchanged.
- `projector` globally pools the encoder's pre-RMS `B×128×8×8` features, applies
  one learned linear `128→128` projector without output normalization, and supplies
  current/next embeddings to SIGReg as `2×B×128` (`T×B×D`). A linear width-preserving
  projector is the smallest treatment that tests the audited primary-source
  constraints: the official statistic supports time×batch embeddings, while the
  LeWorldModel report says a projector was needed when final encoder normalization
  impeded the anti-collapse objective. The unchanged width avoids adding a capacity
  or bottleneck confound.

Before any result is read, representation collapse is defined as mean variance below
`1e-4` or covariance participation-ratio effective rank below `10%` of encoder
dimension. SIGReg is considered near-pinned at `>=99%` of its `10,000` smooth bound.
Action gates require both dynamics aggregate and `random_one_step` shuffled/true MSE
ratios `>=1.10` with paired-bootstrap 95% lower bounds above `1.0`. Genuinely changed
transitions must show at least 10% lower learned one-step MSE than copy-forward.
Across three seeds, the median lower confidence bound for each action ratio must
also exceed `1.0`. A candidate arm has a preregistered severe relative regression
if any paired seed has more than 25% worse horizon-8 rollout MSE, more than 25%
lower encoder variance, or more than 25% lower effective-rank fraction. Such an
arm is not promoted. Any arm with the preregistered credible monotonic approach
advances both arms to seeds 2 and 3.

```bash
# Run one arm/seed through the 1,000 and 2,000 update pauses and evaluations.
P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
P2_EXPECTED_SHA=a4cd11213e7aec91ec744012223d36b73848741c \
TOFY_BIN=/workspace/Personal/Tofy-p2-ab/target/release/tofy \
bash scripts/p2_sigreg_action_ab.sh control 1

P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
P2_EXPECTED_SHA=a4cd11213e7aec91ec744012223d36b73848741c \
TOFY_BIN=/workspace/Personal/Tofy-p2-ab/target/release/tofy \
bash scripts/p2_sigreg_action_ab.sh projector 1

python3 scripts/p2_ab_gate.py \
  --root /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
  --seeds 1 \
  --output-json /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/pilot-gates.json \
  --output-md /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/pilot-gates.md

# Only after branch E has been recorded and all six runs reach 4,000:
python3 scripts/p2_ab_gate.py \
  --root /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
  --seeds 1 2 3 --final-update 4000 \
  --output-json /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/final-4000-gates.json \
  --output-md /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/final-4000-gates.md
```

### Pilot result (2026-08-08)

Both seed-1 arms completed four status-0 phases from fresh initialization at the
reviewed experiment SHA `a4cd11213e7aec91ec744012223d36b73848741c`.
Every update-1,000/update-2,000 checkpoint/report SHA-256 manifest verifies. The
control config hash is
`bcf4b3e456227f1441e680857327a02f292bd323f741d7dfefab5eb7dd819f09`;
the projector config hash is
`0994064d59d7b7b1e3bf9dc6784b7e5016142c45f74cb5d026a099fa6d77cdf2`.

| Arm | Update | Aggregate shuffled/true [95% CI] | `random_one_step` [95% CI] | Changed learned-vs-copy improvement [95% CI] | Variance | Effective-rank fraction | Hard pass |
|---|---:|---|---|---|---:|---:|---|
| control | 1,000 | 1.0396 [1.0219, 1.0607] | 1.1536 [1.0777, 1.2383] | 0.6607 [0.6123, 0.7000] | 0.002760 | 0.0140 | no |
| control | 2,000 | 1.2073 [1.1535, 1.2681] | 1.7291 [1.5048, 1.9963] | 0.4955 [0.4448, 0.5418] | 0.006791 | 0.0245 | no |
| projector | 1,000 | 1.0000 [0.9999, 1.0001] | 1.0000 [0.9998, 1.0003] | -98.4505 [-102.1645, -94.7269] | 1.50e-7 | 0.0157 | no |
| projector | 2,000 | 1.0000 [0.9999, 1.0001] | 1.0001 [0.9999, 1.0002] | -159.3740 [-171.4533, -146.9218] | 3.92e-8 | 0.0144 | no |

At update 2,000, raw/bounded SIGReg was `927.069 / 841.820` for
control and `339.291 / 326.971` for projector; neither was near the 10,000
bound. Control passed both action gates and the changed-transition gate, but
failed noncollapse (`0.0245 < 0.10` rank fraction) and deteriorated over time:
one-step MSE rose `0.0246→0.1087` and horizon-8 open-loop MSE rose
`0.106→0.752`. Its exploration and hazard-source shuffle ratios remained near
1.0. Projector variance collapsed, its action-shuffle ratios stayed at 1.0, and
its learned changed-transition MSE was about 160 times copy-forward at update
2,000; its low absolute latent MSE is therefore degenerate, not a positive
result.

The unchanged preregistered gate selected terminal branch A
(`stop_after_pilot`): neither arm hard-passed or showed the defined credible
monotonic approach, and projector did not materially improve control. Seeds 2
and 3 and the 4,000-update extension were not run. No arm is promoted, and a
full 28,672-update curriculum restart is not recommended. The smallest next
hypothesis is a fresh geometric-isolation arm: pre-RMS spatial-cell SIGReg
without global pooling or a learned projector, with all other control fields
fixed. Exact expanded phase commands and immutable reports live under
`runs/p2/ab-sigreg-action-v1/`; the decision is in `pilot-gates.{json,md}`.

These commands and any interim smoke metrics are experimental diagnostics only.
They do not set `research_claim=true`, use public ARC games, or justify a model claim.

## Readiness training

VRAM/GPU/Muon integration landed 2026-08-05:

- **Batch:** physical `128` × grad_accum `4` (effective 512) on RTX 5060 8GB (~6.4 GiB peak). An equal-effective-batch physical/accumulation change is a trajectory migration, not exact resume, and now requires the explicit migration flag and durable label.
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

#### Preserved readiness-v3 pause and evaluation (2026-08-08)

After the reviewed correctness commit was ready, PID 88711 was re-resolved from
`runs/p2/readiness-v3/train.pid`; its full command, cwd, parent wrapper, and sole
GPU ownership were verified. Exactly one `SIGINT` was sent at
`2026-08-08T11:08:17Z`. The trainer finished its optimizer update and atomically
paused at step 20,557. The wrapper's post-train evaluator then completed normally
at `2026-08-08T11:18:01Z` without interference.

- Preserved checkpoint:
  `runs/p2/readiness-v3/checkpoints/step-000000020557/`.
- Preserved synthetic evaluation: `runs/p2/readiness-v3/eval_report.json` and
  `episodes.jsonl` (`research_claim=false`, no official scorecard).
- Trainer/evaluator/wrapper PIDs exited and the GPU lock was released.

This remains an inherited recovery/stability run, not a clean result, and it was
never used to initialize the SIGReg/action A/B.

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
