# Results P2

P2 is implemented as a recursive latent world-model experiment. No trained metric is
recorded yet; implementation smoke tests must not be promoted to a research result.

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

## Readiness training (in progress)

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
