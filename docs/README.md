# Documentation index

Start here for the **current VecLab + Qwen bridge experiment**. The experiment
tests latent knowledge transfer into a frozen Qwen3 decoder.

| Doc | Purpose |
|-----|--------|
| [RUNPOD.md](RUNPOD.md) | **Cloud training:** pod setup, Qwen download, `train minimal`, resume, artifact recovery |
| [VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md) | Fictional Go library corpus generator and split rules |
| [MODEL_REWRITE_2026-07.md](MODEL_REWRITE_2026-07.md) | **Current architecture:** failed-run evidence and module-by-module rewrite |
| [QWEN_KNOWLEDGE_INJECTION_SPEC.md](QWEN_KNOWLEDGE_INJECTION_SPEC.md) | Experimental claim, ladder, and causal controls |
| [DATA_FORMATS.md](DATA_FORMATS.md) | Pair / world / cache formats |
| [TRAINING_INFRA_FIXES_SPEC.md](TRAINING_INFRA_FIXES_SPEC.md) | Archived training-loop fix record (completed) |
| [RESULTS.md](RESULTS.md) | Best metrics and commands |

## Canonical commands (local or pod)

```bash
# Full pipeline (minimal profile, 96 GiB-class GPU)
export TOFY_QWEN_DIR=models/qwen3-1.7b-base
cargo run --release -- train minimal

# Prepare encoder vocab cache only (optional handoff)
cargo run --release -- prepare cache minimal

# Generate veclab corpus only
cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional
```

The sole supported profile is `minimal`, defined in
`config/model_profiles.json`.

Pipeline stages: veclab prep → encoder → world knowledge → mandatory
seen/held-out RAG ceiling → Qwen bridge → veclab eval.

## Source layout (current)

- [`src/tasks/pipeline.rs`](../src/tasks/pipeline.rs) — `train` / `prepare cache`
- [`src/tasks/prepare_veclab.rs`](../src/tasks/prepare_veclab.rs) — VecLab corpus generator
- [`src/tasks/veclab.rs`](../src/tasks/veclab.rs) — `--prepare-veclab` CLI + split helpers
- [`src/tasks/latent.rs`](../src/tasks/latent.rs) — LeJEPA encoder training
- [`src/tasks/knowledge.rs`](../src/tasks/knowledge.rs) — `--train-world-knowledge`
- [`src/tasks/world_context.rs`](../src/tasks/world_context.rs) — world batch / context assembly
- [`src/tasks/bridge.rs`](../src/tasks/bridge.rs) — `--train-bridge`
- [`src/tasks/eval.rs`](../src/tasks/eval.rs) — `--eval-bridge`
- [`src/model/leworld.rs`](../src/model/leworld.rs) — LeWorldModel projectors + predictor core
- [`src/model/decoders/qwen3_bridge.rs`](../src/model/decoders/qwen3_bridge.rs) — gated cross-attn on Qwen3

Root [../README.md](../README.md) provides the current quick start; use this
index and [RUNPOD.md](RUNPOD.md) for the full workflow.
