# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

*(No entries yet. Add metric, dataset/split, and full command when you have a run to record.)*

Example:

- **JEPA eval (cos_ema):** 0.42 — `cargo run --release -- --eval-jepa local_models/model_latent_48.00M.safetensors local_models/vocabs/vocab_encoder.txt hub:wikimedia/wikipedia 500 32 768 128 6 8`
- **World eval (transition_cos):** 0.38 — `cargo run --release -- --eval-world ...`
