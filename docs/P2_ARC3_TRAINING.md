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
| `falsification` | 4096 | Multi-candidate observer probes; world core frozen |
The default run stops after `falsification`; `retarget` remains an optional ablation.

Total default: 28,672 optimizer steps (`dynamics` and `exploration` at 8,192 each,
then three 4,096-step lessons).

## Train

Use `scripts/p2_arc3_train_eval.sh` only after measuring the largest stable
physical batch on the exact reviewed CUDA binary. The script requires a clean
reviewed commit, a matching binary hash, explicit batch-measurement attestation,
preserved batch-measurement evidence, a fetchable named remote ref, a new run
root, and successful five-lesson training plus one-episode evaluation smokes
before it starts the full campaign. It persists a launch manifest containing
the exact source, binary, build, hardware, batch, and preflight hashes. Full V4
rejects gradient accumulation above one so the EP population matches the
physical batch.

## Eval (schema `p2.eval_report.v15`)

Split held-out probes aligned with training stages:

- **`synthetic_dynamics`**: `random_one_step`, `exploration`, hazard (open-loop rollout on dynamics traces).
- **`synthetic_planner`**: `sequential`, `hypothesis_probe`, falsification, retarget.
- **`synthetic_iid_dynamics` / `synthetic_iid_planner`**: the same families on
  an unseen seed from the 7×7 training-composition distribution. The original
  fields remain the 8×8 held-out-composition OOD populations.
- **`arc3_transfer`**: optional `--arc-recordings-dir` (never used for training).

Full V4 reports exact-decoder CE/NLL, pixel accuracy, equal-transition accuracy,
and exact-board accuracy for content, padding, foreground, changed, unchanged,
and changed-content masks. Current/target reconstruction, one-step prediction,
learned/hard copy, zero, direct-target, action-masked, and within-source
action-shuffled controls share the same labels and population. Semantic
open-loop endpoints are persisted at H4 and H8; H1 is the one-step report.
Visible-input collision ceilings and same-state factual-outcome retrieval are
reported separately from action-ID recovery.

Every transition carries explicit content dimensions, source kind, and a stable
trajectory ID. The bottom status row is counted separately and excluded from
the exact decoder. ACTION5 and ACTION6 retain distinct source kinds.

## Diagnostic checkpoints and provenance

Fresh Full V4 runs publish complete bundles at updates `0`, `8192`, `16384`,
`20480`, `24576`, and `28672` (plus the configured periodic cadence). Each
bundle contains model, optimizer, trainer state, persisted config, a SHA-256
artifact manifest, and named parameter-group hashes. Resume and boundary eval
verify these hashes before loading.

The production launcher evaluates all six boundaries with explicit train,
initialization, IID, and OOD seeds. It verifies that every non-observer
parameter-group hash is identical at updates 20480/24576/28672, hashes each
evaluation report, and only marks the pipeline complete after the final tree
seal succeeds.

## Not in synthetic training

- Public ARC game recordings (transfer eval only).
- `available_actions` API masking (inference harness).
- Animation frame sequences (importer uses settled last frame, same as `arc3`).
