# Tofy

Clean-slate hidden-objective planning research. **P2** (this branch) learns a recursive
latent world model from synthetic exact-simulator lessons. Official ARC-AGI-3
recordings are held-out transfer data only.

The P1 exact-simulator feasibility harness (`p1a` / `p1b` / `p1c` / `p1c-hard`) lives
only on the `p1` git branch. Archived P1 metrics remain in
[`docs/RESULTS_P1.md`](docs/RESULTS_P1.md).

The previous VecLab → LeJEPA → world → Qwen bridge experiment is **archived in Git
history as a negative result**.

## Phase status

P0 was skipped. P1 established exploratory mechanisms on the `p1` branch. P2 now
learns goal-free pixel dynamics from those synthetic lessons. Public candidate
features condition only the auxiliary predicate head; the hidden goal/index never
enters the dynamics model.

## Layout

| Module | Role |
|---|---|
| `src/domain.rs` | Exact grid MDP: goals, state, simulator |
| `src/generator.rs` | Deterministic Train / structurally held-out compositions |
| `src/search.rs` | Exact BFS / shortest paths for synthetic labels |
| `src/experiment.rs` | clap CLI for P2 commands |
| `src/p2/model.rs` | categorical pixel encoder, shared recursive block, event/Q heads, PTRM |
| `src/p2/sigreg.rs` | LeWorldModel Epps--Pulley SIGReg loss |
| `src/p2/data.rs` | synthetic lesson transitions and ARC-shaped tensors |
| `src/p2/arc3.rs` | official-toolkit recording JSONL importer |
| `src/p2/train.rs` | ordered curriculum trainer and runtime-gradient trace |
| `src/p2/eval.rs` | held-out dynamics, rollout, calibration, and PTRM metrics |

## Commands

The defaults are deliberately tiny smoke settings and always write
`research_claim=false`:

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --checkpoint-every-steps 100 \
  --output-dir runs/p2/smoke

cargo run --release -- p2-eval \
  --checkpoint runs/p2/smoke/model.safetensors \
  --train-config runs/p2/smoke/config.json \
  --output runs/p2/smoke/eval_report.json

cargo run --release -- p2-arc3-eval \
  --checkpoint runs/p2/smoke/model.safetensors \
  --train-config runs/p2/smoke/config.json \
  --arc-recordings-dir /path/to/official-toolkit-recordings \
  --output runs/p2/smoke/arc3_eval_report.json

# Optional: attach official RHAE from a closed scorecard JSON
cargo run --release -- p2-eval \
  --checkpoint runs/p2/smoke/model.safetensors \
  --train-config runs/p2/smoke/config.json \
  --scorecard-json /path/to/scorecard.json \
  --output runs/p2/smoke/eval_with_rhae.json

cargo p2-view runs/p2/smoke/profile.jsonl --output runs/p2/smoke/model.html
```

`p2-train` treats `SIGINT`/`SIGTERM` as a clean pause request. It finishes the
current optimizer update and prints the complete checkpoint bundle. Resume from that
bundle, its `checkpoints` parent, or the run directory:

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --output-dir runs/p2/smoke \
  --resume runs/p2/smoke/checkpoints
```

All trajectory-defining options must match. Checkpoint cadence and
`--max-steps-this-run` are operational controls and may change across invocations.

Accelerator builds (prefer `cudnn` — see [`vendor/candle-core/TOFY_PATCH.md`](vendor/candle-core/TOFY_PATCH.md)):

```bash
cargo run --release --features cudnn -- p2-train --device cuda ...
```

Optional timings: `TOFY_P2_STEP_PROFILE=20`, or Chrome Trace via
`TOFY_PERF_TRACE=/tmp/tofy-perf.json` with `--features profiling,cudnn`.
For unified CPU+GPU timelines on CUDA, wrap the same binary with NVIDIA
Nsight Systems (`nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu
...`); see [`src/perf.rs`](src/perf.rs).

Validation:

```bash
cargo fmt
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

See [`docs/P2.md`](docs/P2.md) for the contract and
[`docs/ARC_AGI_3_PLAN.md`](docs/ARC_AGI_3_PLAN.md) for the staged path to the official
evaluation. See
[`docs/CANDLE_GRAPH.md`](docs/CANDLE_GRAPH.md) for the full
candle-graph guide (HTML visualizer, train/infer graphs, audit, v10 artifacts). Track metrics in [`docs/RESULTS_P2.md`](docs/RESULTS_P2.md) (index:
[`docs/RESULTS.md`](docs/RESULTS.md)).
