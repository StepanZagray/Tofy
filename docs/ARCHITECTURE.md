# Architecture

## High-level modules

- **`src/main.rs`:** CLI dispatch and JEPA latent training/eval loops.
- **`src/tasks/world.rs`:** World-model train, eval, and agent inference.
- **`src/model/*`:** Encoder, predictor, world transition, decoder bridge, decoder runtime (llama-cli).
- **`src/data/*`:** Tokenization, batching, hub caching, UltraChat pair preparation.

There is no classification path; the codebase is agent-only.

## JEPA latent path

1. Build vocab and sequence pairs (or use hub cache).
2. Generate masked context/target batches.
3. Encode context with online encoder, target with EMA teacher.
4. Predict target latents via predictor MLP.
5. Optimize alignment + VICReg-style variance/covariance terms.
6. Save best online and teacher checkpoints.

## World-model path (fixed action `reply`)

1. Encode context into state latent.
2. Predict next latent with `WorldTransition` conditioned on fixed reply action embedding.
3. Train against teacher-encoded next-turn latent.
4. Project latent through `DecoderBridge` to conditioning vector.
5. At inference: call local decoder runtime with prompt + conditioning (or zeroed for ablation).

## Decoder runtime

- **Primary:** llama.cpp CLI (`llama-cli`) with a GGUF model (e.g. Llama 3.2 1B, Qwen 1.5B).
- **Fallback:** Stub string output when decoder binary or model is missing.
- Selection is automatic in `--infer-agent`.
