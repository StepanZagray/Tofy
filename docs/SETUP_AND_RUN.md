# Setup and Run

## Prerequisites

- Rust toolchain (`cargo`)
- CUDA-capable GPU for default build (optional; CPU fallback with `--no-default-features`)
- For downloading GGUF/UltraChat: Hugging Face CLI (`hf`) — install globally with pipx (see RUNBOOK)
- For agent inference: `llama-cli` from llama.cpp on PATH

## Build modes

- Default (CUDA): `cargo run --release -- <args>`
- CPU-only: `cargo run --release --no-default-features -- <args>`

## Typical workflow (agent only)

1. Install `hf` globally (pipx).
2. Prepare world data: `--prepare-ultrachat`.
3. Train JEPA encoder: `--latent` (e.g. Wikipedia with `JEPA_WIKI_MAX_FILES=1`).
4. Train world model: `--train-world ... --init-encoder <encoder>`.
5. Run agent: `--infer-agent` with a GGUF decoder.

## Commands

### Prepare UltraChat (world-model pairs)

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

### JEPA latent training

```bash
JEPA_WIKI_MAX_FILES=1 cargo run --release -- --latent hub:wikimedia/wikipedia 25000 32 368 128 4 8 8000
```

### JEPA evaluation

```bash
cargo run --release -- --eval-jepa model_latent_9.51M.safetensors vocab_latent.txt hub:wikimedia/wikipedia 200 32 368 128 4 8
```

### World-model training

```bash
cargo run --release -- --train-world data/ultrachat_pairs.txt 40000 32 368 128 4 8 8000 256 --init-encoder model_latent_9.51M.safetensors
```

### World-model evaluation

```bash
cargo run --release -- --eval-world model_world_*.safetensors vocab_world.txt data/ultrachat_pairs.txt 200 32 368 128 4 8 256
```

### Agent inference

```bash
export JEPA_DECODER_MODEL=./models/your_model.gguf
cargo run --release -- --infer-agent model_world_*.safetensors vocab_world.txt "your prompt" 368 128 4 8 256 64
```

See [RUNBOOK.md](RUNBOOK.md) for full step-by-step and [DECODER_RUNTIME.md](DECODER_RUNTIME.md) for decoder env vars.
