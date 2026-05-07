# Documentation index

All docs live in this directory. Start with [RUNBOOK.md](RUNBOOK.md) for copy-paste commands from setup to inference.

| Doc | Purpose |
|-----|--------|
| [RUNBOOK.md](RUNBOOK.md) | End-to-end: clone → HF CLI → data → `train 8gb|48gb|80gb` (or per-stage flags) → eval → serve. Includes TensorBoard. |
| [OPENCODE.md](OPENCODE.md) | Run Tofy in OpenCode; auth and provider config; code-specialist and text-generalist decoders. |
| [ARCHITECTURE_AND_CAPACITY.md](ARCHITECTURE_AND_CAPACITY.md) | Current planner-memory architecture, cross-action awareness, capacity, and multi-step replies. |
| [CODE_DATA.md](CODE_DATA.md) | Code training data: CLI `--prepare-*` generators, Rust-by-Practice corpus, pair formats. |
| [DATA_FORMATS.md](DATA_FORMATS.md) | Data formats for LeJEPA/planner-world/decoder training; hub caching; expert/technical data. |
| [DECODER_RUNTIME.md](DECODER_RUNTIME.md) | Decoder backends (GGUF/llama.cpp, Candle code/text decoders) and env vars. |
| [OOM_TESTING.md](OOM_TESTING.md) | Sustained CUDA OOM probes for batch/VRAM decisions. |
| [RESULTS.md](RESULTS.md) | Best metrics and commands (updated when runs improve). |

Root [../README.md](../README.md) has quick start, argument order, and project structure.

## Current Source Layout

Recent refactors split command parsing and runtime code more explicitly:

- [`src/main.rs`](../src/main.rs) is now a thin command-dispatch entrypoint.
- [`src/cli.rs`](../src/cli.rs) owns shared CLI helpers such as hub-path resolution and usage text.
- [`src/config/latent.rs`](../src/config/latent.rs) and [`src/config/world.rs`](../src/config/world.rs) hold typed configs for latent, world, decoder, eval, and serve commands.
- [`src/tasks/latent.rs`](../src/tasks/latent.rs) owns latent training and JEPA evaluation.
- [`src/tasks/pipeline.rs`](../src/tasks/pipeline.rs) owns the canonical full `train <8gb|48gb|80gb>` multi-stage pipeline (data prep through code eval).
- [`src/tasks/world.rs`](../src/tasks/world.rs) owns world/high-world/orchestrator/decoder training plus agent runtime.
- [`src/tasks/world_support.rs`](../src/tasks/world_support.rs) holds shared world/decoder metrics, masking, and evaluation helpers extracted from `world.rs` for readability.
