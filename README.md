# Tofy

Tofy tests whether a world model can transmit fictional API knowledge to a
frozen **Qwen3-1.7B-Base** decoder through learned latent conditioning. The
experiment uses the deterministic `veclab` Go library, so every answer can be
compiled and tested without relying on benchmark contamination.

The current pipeline is:

`veclab preparation → LeJEPA encoder → world-knowledge model → Qwen bridge → Go evaluation`

## Quick start

Prerequisites: a Rust toolchain, Go (for corpus and evaluation verification),
a CUDA-capable GPU for training, and a local Qwen3-1.7B-Base model directory.

```bash
export TOFY_QWEN_DIR=models/qwen3-1.7b-base
cargo run --release -- train minimal
```

`minimal` targets the 96 GiB RTX PRO 6000 Blackwell pod. It uses 20,000
encoder, world, and bridge steps. Physical batch/accumulation pairs are
`16/8`, `8/32`, and `8/16` respectively, giving effective batches `128`,
`256`, and `128`. Encoder and world SIGReg statistics are pooled over the full
effective batch with memory-bounded replay, rather than being estimated
separately on each physical microbatch. These revised training paths require a
sustained hardware qualification. Smaller GPUs require a fresh capacity
qualification.
`minimal` is the only supported profile; its exact shape is in
[`config/model_profiles.json`](config/model_profiles.json).

Prepare the deterministic corpus and encoder cache without starting training:

```bash
cargo run --release -- prepare cache minimal
```

Generate and verify only the VecLab corpus:

```bash
cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional --print-split-stats
```

Resume a pipeline run:

```bash
cargo run --release -- train minimal --resume latest
```

The pipeline writes all artifacts below `runs/code_poc_<timestamp>/`:
`latent/`, `world/`, `bridge/`, and `eval/`.

For the `minimal` profile, training automatically recovers from a confirmed
CUDA allocation OOM. It halves the physical batch and doubles gradient
accumulation (`16/8 → 8/16 → 4/32`, for example), so the effective batch and
optimizer-step schedule stay unchanged. Each attempted and selected pair is
atomically recorded in `adaptive_batches.json` in the run root. A retry resumes
from the latest complete checkpoint tuple when one exists, otherwise it
restarts that stage cleanly. Non-CUDA failures and bridge nonqualification are
never treated as OOMs. Set `TOFY_AUTO_BATCH_OOM_RECOVERY=false` to disable this
behavior.

## Experiment safeguards

The bridge task prompt and encoder input never contain the `[fn:NNN]` sampling
tag. Batch-one bridge training uses a real different-function conditioning
negative, and a checkpoint is eligible only when its wrong-conditioning loss
exceeds its matched-conditioning loss by at least
`TOFY_BRIDGE_MIN_SEMANTIC_GAP` (default `0.02`).

Do **not** rerun the decoder-only floor for the current VecLab experiment.
That control already scored zero seen and held-out suite passes; see
[`docs/RESULTS.md`](docs/RESULTS.md). Training never schedules it. Evaluate
only qualifying bridge checkpoints with matched, wrong, and zeroed
conditioning. To reopen the floor after a deliberate base-model / tokenizer /
prompt / suite change, run
`scripts/run_veclab_decoder_floor.sh <runs/code_poc_<id>>` against an
existing run directory (it loads bridge artifacts and sets
`TOFY_EVAL_MODE=floor`).

`code_poc_1783547471` is an invalidated baseline: its matched conditioning did
not outperform shuffled conditioning. Do not resume it or treat its pass rates
as evidence of knowledge transmission. See the results record for exact
metrics and commands.

## Documentation

- [Documentation index](docs/README.md) — current workflow and source layout
- [RunPod guide](docs/RUNPOD.md) — pod setup, training, resume, and recovery
- [VecLab data specification](docs/VECLAB_DATA_SPEC.md) — generated corpus and split controls
- [World-to-Qwen architecture rewrite](docs/MODEL_REWRITE_2026-07.md) — failed-run evidence and module-by-module replacement
- [Qwen knowledge-injection specification](docs/QWEN_KNOWLEDGE_INJECTION_SPEC.md) — experiment and causal controls
- [Results](docs/RESULTS.md) — best metrics, invalidated runs, and exact commands

Use `cargo run --release --` with no mode to print the complete CLI usage.
