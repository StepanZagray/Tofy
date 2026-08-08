# P2 ARC-AGI-3 aligned training

Synthetic training matches the official environment contract as closely as possible
without fitting on public ARC recordings.

## Pixel contract

- **64×64 categorical frames**, 16 colors, native grid copied to top-left (ArcPad).
- **Status UI** on the bottom canvas row (remaining action budget bar).
- **8×8 patch encoder** (8×8 spatial grid) instead of legacy 4×4 conv pool.
- **ACTION1–5, ACTION6 (xy), ACTION7 (undo)** mixed in `dynamics` one-step batches.

## Curriculum (default)

| Lesson | Steps | ARC alignment |
|--------|-------|----------------|
| `dynamics` | 8192 | Goal-free one-step + coordinate + interact + hazard |
| `exploration` | 8192 | 8–12 step random walks, zero goal features, masked events |
| `sequential` | 4096 | Multi-step plans + open-loop rollout ramp |
| `q_calibration` | 4096 | Warm Q/event heads |
| `falsification` | 4096 | Multi-candidate probes + PTRM rank @ k=4 |
The default run stops after `falsification`; `retarget` remains an optional ablation.

Total default: 28,672 optimizer steps (`dynamics` and `exploration` at 8,192 each,
then three 4,096-step lessons).

## Train

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 8 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --checkpoint-every-steps 100 --output-dir p2-output-v14
```

Or: `scripts/p2_arc3_train_eval.sh`

## Eval (schema `p2.eval_report.v9`)

Split held-out probes aligned with training stages:

- **`synthetic_dynamics`**: `random_one_step`, `exploration`, hazard (open-loop rollout on dynamics traces).
- **`synthetic_planner`**: `sequential`, `hypothesis_probe`, falsification, retarget.
- **`arc3_transfer`**: optional `--arc-recordings-dir` (never used for training).

## Not in synthetic training

- Public ARC game recordings (transfer eval only).
- `available_actions` API masking (inference harness).
- Animation frame sequences (importer uses settled last frame, same as `arc3`).
