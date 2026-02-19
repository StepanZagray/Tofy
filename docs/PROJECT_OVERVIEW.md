# Project Overview

Tofy is a Rust (Candle) project that implements a **JEPA-style world-model agent**:

- **JEPA latent pretraining** (`--latent`, `--eval-jepa`): encoder + predictor + EMA teacher; VICReg-style regularization.
- **World-model agent** (`--train-world`, `--eval-world`, `--infer-agent`): fixed `reply` action, transition head, decoder bridge, local text generation via llama.cpp + GGUF.

There is **no classification**; the model is an agent only.

## Design goals

- Keep training and inference local and reproducible.
- Simple, explicit CLI workflows.
- Lightweight encoder-side modeling.
- External decoder backends (e.g. Llama 1B, Qwen 1.5B GGUF) without changing world-model checkpoints.

## Current architecture status

- Encoder and JEPA training are implemented and used for pretraining.
- World-model path: encode context → transition(reply) → bridge → conditioning for decoder.
- Local decoder: llama.cpp CLI (`llama-cli`) with GGUF; stub fallback when unavailable.

## Stable vs evolving

- **Stable:** JEPA latent training/eval; world-model train/eval/infer with fixed `reply`; UltraChat prep in Rust; decoder runtime interface.
- **Evolving:** Decoder backend tuning; **true conditioning** (use bridge output as prefix embeddings or adapter input so JEPA steers the decoder by condition, not only by prompt text); optional multi-action policy.
