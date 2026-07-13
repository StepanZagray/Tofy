# Tofy

Tofy tests whether a world model can transmit fictional API knowledge to a
frozen **Qwen3-1.7B-Base** decoder through learned latent conditioning. The
experiment uses the deterministic `veclab` Go library, so every answer can be
compiled and tested without relying on benchmark contamination.

The current pipeline is:

`veclab preparation → LeJEPA encoder → world-knowledge model → Qwen bridge → Go evaluation`

The old Candle text/code-decoder and Go-assistant pipeline is retired. Its
documentation is retained under `docs/` only as historical context.

## Quick start

Prerequisites: a Rust toolchain, Go (for corpus and evaluation verification),
a CUDA-capable GPU for training, and a local Qwen3-1.7B-Base model directory.

```bash
export TOFY_QWEN_DIR=models/qwen3-1.7b-base
cargo run --release -- train minimal
```

`minimal` is the L40S/RTX 6000 Ada-safe profile. It uses 20,000 encoder,
world, and bridge steps, with a batch-one bridge and gradient accumulation.
The other profiles are `48gb` and `80gb`; their exact shapes and schedules are
in [`config/model_profiles.json`](config/model_profiles.json).

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

## Experiment safeguards

The bridge task prompt and encoder input never contain the `[fn:NNN]` sampling
tag. Batch-one bridge training uses a real different-function conditioning
negative, and a checkpoint is eligible only when its wrong-conditioning loss
exceeds its matched-conditioning loss by at least
`TOFY_BRIDGE_MIN_SEMANTIC_GAP` (default `0.02`).

`code_poc_1783547471` is an invalidated baseline: its matched conditioning did
not outperform shuffled conditioning. Do not resume it or treat its pass rates
as evidence of knowledge transmission. See the results record for exact
metrics and commands.

## Documentation

- [Documentation index](docs/README.md) — current workflow and source layout
- [RunPod guide](docs/RUNPOD.md) — pod setup, training, resume, and recovery
- [VecLab data specification](docs/VECLAB_DATA_SPEC.md) — generated corpus and split controls
- [Qwen knowledge-injection specification](docs/QWEN_KNOWLEDGE_INJECTION_SPEC.md) — design and July 2026 causal-control repair
- [Results](docs/RESULTS.md) — best metrics, invalidated runs, and exact commands

Use `cargo run --release --` with no mode to print the complete CLI usage.
