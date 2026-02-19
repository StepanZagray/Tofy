# Decoder Runtime

## Goal

Provide local text generation for `--infer-agent` without coupling decoder internals to world-model training code.

## Backends

- Preferred: `LlamaCppDecoder` (calls `llama-cli`)
- Fallback: `StubLocalDecoder`

## Environment variables

- `JEPA_DECODER_MODEL`: explicit path to `.gguf` model
- `JEPA_DECODER_BIN`: decoder binary name/path (default `llama-completion`). Use `llama-completion` for non-interactive completion; if you only have `llama-cli` and it errors on flags, set `JEPA_DECODER_BIN=llama-cli` and ensure your build supports the same flags, or install/build `llama-completion`.
- `JEPA_DECODER_CTX`: context window size (default `4096`)
- `JEPA_DECODER_NGL`: number of GPU layers (default `99`)
- `JEPA_DECODER_TEMP`: sampling temperature (default `0.7`)
- `JEPA_DECODER_REPEAT_PENALTY`: repeat penalty (default `1.12`)
- `JEPA_DECODER_COND_FORMAT`: how to encode the conditioning vector in the prompt:
  - `chunks` (default): 8 chunk means as `chunks=[a,b,c,d,e,f,g,h]` — short, stable signal for the LM to use.
  - `stats`: legacy mean/std/l2/head summary.
  - `both`: chunks plus stats (longer prompt).

## Conditioning vs context (current limitation)

Right now the JEPA/world-model output is **not** used to condition the decoder’s forward pass. It is only turned into **text** (e.g. `chunks=[...]`) and placed in the prompt. So the LLM is steered by **extra context**, not by a continuous conditioning vector. For the JEPA model to **actually steer** the decoder by condition you’d need one of:

- **Prefix / embedding conditioning**: a decoder backend that accepts a vector (or a sequence of embedding vectors) and uses it as a prefix or bias in the forward pass (e.g. “prompt cache” of embeddings, or a conditioning embedding that is prepended or added to hidden states). Then the bridge output would be passed as that vector (or mapped to a prefix of tokens/embeddings) instead of being summarized as text.
- **Adapter**: a small module that takes the conditioning vector and modulates the decoder (e.g. scale/shift per layer, or extra cross-attention from the condition). That would require running the decoder in a setting where you can inject such a module (not the plain llama.cpp CLI).
- **Decoder trained with conditioning**: a decoder that was fine-tuned so that an auxiliary conditioning input (e.g. a single embedding or a short prefix) is part of its training; then at inference you pass the bridge output as that input.

Until one of these is implemented, the world-model output only affects the decoder via the prompt text, not as true conditioning.

## Running with OpenCode (or other OpenAI-compatible clients)

Use **`--serve`** to run an OpenAI-compatible HTTP server so OpenCode (or any compatible client) can use Tofy as the backend:

```bash
cargo run --release -- --serve model_world_10.00M.safetensors vocab_world.txt 0.0.0.0:8080 368 128 4 8 256
```

Then in OpenCode: set Base URL to `http://localhost:8080`, model to `tofy`. The server exposes `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`. Default `max_tokens` per request is 4096.


## Improving with conditioning

The decoder only sees conditioning as **text** in the prompt (not as embeddings). Default format is `chunks`: eight numbers that summarize the 256-d vector so the model can use them to guide tone/depth/focus. If responses are better with `--ablate-conditioning`, try: (1) `JEPA_DECODER_COND_FORMAT=chunks` (default), (2) a clearer system instruction is already used (“use the latent condition to guide your reply”). For stronger improvement, you’d need a decoder that accepts prefix/embedding conditioning or fine-tuning on (condition, reply) pairs.

## Model discovery behavior

If `JEPA_DECODER_MODEL` is not set, runtime searches:

1. `./models/*.gguf`
2. `./models/**/*.gguf`

First sorted match is used.

## Example

```bash
export JEPA_DECODER_MODEL=./models/your_model.gguf
cargo run --release -- --infer-agent model_world_10.00M.safetensors vocab_world.txt "write a short gpu recommendation" 368 128 4 8 256 4096
```

## Troubleshooting

- If output says stub backend is used:
  - install/build `llama-cli`, or
  - set `JEPA_DECODER_BIN` to full binary path.
- If decoder process fails:
  - run decoder command manually to verify model path and flags.
- If VRAM is tight:
  - reduce `JEPA_DECODER_CTX`,
  - reduce `JEPA_DECODER_NGL`,
  - use a smaller or lower-precision GGUF.
