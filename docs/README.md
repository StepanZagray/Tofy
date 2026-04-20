# Documentation index

All docs live in this directory. Start with [RUNBOOK.md](RUNBOOK.md) for copy-paste commands from setup to inference.

| Doc | Purpose |
|-----|--------|
| [RUNBOOK.md](RUNBOOK.md) | End-to-end: clone → HF CLI → data → train (LeJEPA, planner/world, decoder) → eval → serve. Includes TensorBoard. |
| [OPENCODE.md](OPENCODE.md) | Run Tofy in OpenCode; auth and provider config; code-specialist and text-generalist decoders. |
| [ARCHITECTURE_AND_CAPACITY.md](ARCHITECTURE_AND_CAPACITY.md) | Current planner-memory architecture, cross-action awareness, capacity, and multi-step replies. |
| [CODE_DATA.md](CODE_DATA.md) | Code training data: GitHub Top Code script, Rust-by-Practice script, formats. |
| [DATA_FORMATS.md](DATA_FORMATS.md) | Data formats for LeJEPA/planner-world/decoder training; hub caching; expert/technical data. |
| [DECODER_RUNTIME.md](DECODER_RUNTIME.md) | Decoder backends (GGUF/llama.cpp, Candle code/text decoders) and env vars. |
| [OOM_TESTING.md](OOM_TESTING.md) | Sustained CUDA OOM probes for batch/VRAM decisions. |
| [RESULTS.md](RESULTS.md) | Best metrics and commands (updated when runs improve). |

Root [../README.md](../README.md) has quick start, argument order, and project structure.
