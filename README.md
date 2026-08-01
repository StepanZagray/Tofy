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

scripts/audit_p2.sh \
  runs/p2/smoke/analyzer \
  runs/p2/smoke/model.safetensors \
  runs/p2/smoke/runtime.json
```

Validation:

```bash
cargo fmt
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

See [`docs/P2.md`](docs/P2.md) for the contract and
[`docs/CANDLE_MODEL_ANALYZER.md`](docs/CANDLE_MODEL_ANALYZER.md) for the external
audit boundary. Track metrics in [`docs/RESULTS_P2.md`](docs/RESULTS_P2.md) (index:
[`docs/RESULTS.md`](docs/RESULTS.md)).
