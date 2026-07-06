# Documentation index

Start here for the **current veclab + Qwen bridge experiment**:

| Doc | Purpose |
|-----|--------|
| [RUNPOD.md](RUNPOD.md) | **Cloud training:** pod setup, Qwen download, `train minimal`, resume, artifact recovery |
| [VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md) | Fictional Go library corpus generator and split rules |
| [BRIDGE_EXPERIMENT_FIXES_SPEC.md](BRIDGE_EXPERIMENT_FIXES_SPEC.md) | Implementation fixes and experiment ladder |
| [TRAINING_INFRA_FIXES_SPEC.md](TRAINING_INFRA_FIXES_SPEC.md) | Training-loop bugs, observability |
| [QWEN_KNOWLEDGE_INJECTION_SPEC.md](QWEN_KNOWLEDGE_INJECTION_SPEC.md) | Original experiment design (partially superseded by fixes spec) |
| [RESULTS.md](RESULTS.md) | Best metrics and commands |

## Canonical commands (local or pod)

```bash
# Full pipeline (minimal profile, 48 GB-class GPU)
export TOFY_QWEN_DIR=models/qwen3-1.7b-base
cargo run --release -- train minimal

# Prepare encoder vocab cache only (optional handoff)
cargo run --release -- prepare cache minimal

# Generate veclab corpus only
cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional
```

Profiles: `minimal`, `48gb`, `80gb` in `config/model_profiles.json`.

Pipeline stages: veclab prep → encoder → world knowledge → Qwen bridge → veclab eval.

## Source layout (current)

- [`src/tasks/pipeline.rs`](../src/tasks/pipeline.rs) — `train` / `prepare cache`
- [`src/tasks/knowledge.rs`](../src/tasks/knowledge.rs) — `--train-world-knowledge`
- [`src/tasks/bridge.rs`](../src/tasks/bridge.rs) — `--train-bridge`, `--eval-bridge`
- [`src/tasks/prepare_veclab.rs`](../src/tasks/prepare_veclab.rs) — `--prepare-veclab`
- [`src/model/decoders/qwen3_bridge.rs`](../src/model/decoders/qwen3_bridge.rs) — gated cross-attn on Qwen3

## Older docs (pre-veclab refactor)

These still describe the removed Candle code/text decoder + `--serve` stack.
Use only for historical context:

- [RUNBOOK.md](RUNBOOK.md) — being updated incrementally
- [DECODER_RUNTIME.md](DECODER_RUNTIME.md) — obsolete (GGUF/Candle decoders)
- [ARCHITECTURE_AND_CAPACITY.md](ARCHITECTURE_AND_CAPACITY.md) — concepts still useful; pipeline section stale
- [CODE_DATA.md](CODE_DATA.md), [OPENCODE.md](OPENCODE.md) — old Go POC data path

Root [../README.md](../README.md) quick start is also stale; prefer this index + RUNPOD.md.
