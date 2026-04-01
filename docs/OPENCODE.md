# Running Tofy in OpenCode

Use [OpenCode](https://opencode.ai) with your local Tofy server so the coding agent runs the current planner-memory pipeline.

## Prerequisites

1. **Tofy server is running** (see [RUNBOOK.md](RUNBOOK.md) section 11 — Serve). Default: `http://localhost:8080`.
2. **OpenCode** installed (e.g. from [opencode.ai](https://opencode.ai) or your package manager).

## Quick walkthrough

### 1. Start the Tofy server (if not already)

From your Tofy repo. Use either **GGUF** or the in-repo Candle decoders.

**With GGUF:**
```bash
export JEPA_DECODER_MODEL=./models/<your_model>.gguf
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

**With Candle code + text decoders (no GGUF):**  
Trained models live in `local_models/`; vocabs in `local_models/vocabs/`. Set both code and text decoders so OpenCode can do code and chat.
```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=./local_models/code_decoder_90M.safetensors
export JEPA_USE_TEXT_DECODER=1
export JEPA_TEXT_DECODER=./local_models/text_decoder_90M.safetensors
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

You should see: `Tofy OpenAI-compatible server listening on http://0.0.0.0:8080`.

### 2. Configure OpenCode to use Tofy

OpenCode talks to backends via **providers**. Add a provider that points at your Tofy server.

**Auth** (so OpenCode can call the API): create or edit `~/.local/share/opencode/auth.json` (paths may differ by OS; check OpenCode docs):

```json
{
  "tofy": {
    "type": "api",
    "key": "sk-local"
  }
}
```

The key can be any placeholder (e.g. `sk-local`); the Tofy server does not validate API keys.

**Provider and model**: edit `~/.config/opencode/opencode.json` (paths may differ by OS). You must have **one** top-level JSON object. Add the `provider` and `model` keys **inside that same object** (alongside your existing `$schema`, `theme`, etc.). Do not add an extra `{ ... }` around provider/model.

Example of a **valid** full file (if you already have `$schema` and `theme`, keep them and add `provider` and `model`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "theme": "system",
  "provider": {
    "tofy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Tofy",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "tofy": {
          "name": "Tofy (world-model agent)"
        }
      }
    }
  },
  "model": "tofy/tofy"
}
```

- **baseURL**: must end with `/v1`. If the server runs on another host or port, use that (e.g. `http://192.168.1.10:8080/v1`).
- **model id**: `tofy` (returned by the Tofy server; see `GET /v1/models`).

### 3. Restart OpenCode and select the model

Restart OpenCode so it picks up the new provider. In the UI, use the model switcher (often `/model` or the model dropdown) and choose **Tofy** or `tofy/tofy`.

### 4. Use Tofy as your coding agent

Chat or run agent tasks as usual. Requests go to your local Tofy server, which runs:

`encoder -> planner memory -> router/orchestrator -> decoder adapter -> decoder`

**How to test:** Open a new chat in OpenCode, select the **Tofy** model (e.g. `tofy/tofy`), and send a message. The first reply may be slow (encoder + planner/world + decoder). For code: ask for a snippet or “write a function …”; for chat: ask a normal question. Check the Tofy server terminal for logs and any errors. To confirm the API: `curl http://localhost:8080/health` and `curl http://localhost:8080/v1/models`.

**Token stream / generation animation:** The Tofy server supports **streaming** (`stream: true`). When OpenCode sends that, the server returns Server-Sent Events (SSE) so tokens appear as they’re generated and you get the usual typing animation. If you previously saw no animation and no response, ensure your OpenCode client sends `stream: true` (many do by default); the server now responds with SSE in that case.

## Verifying the connection

- **Health**: `curl http://localhost:8080/health` → `ok`
- **Models**: `curl http://localhost:8080/v1/models` → list including `tofy` with `owned_by: "Tofy"`

## Troubleshooting

- **Connection refused**: Ensure the Tofy server is running and the port in `baseURL` matches (default 8080).
- **Wrong base URL**: Use `http://localhost:8080/v1` (with `/v1`). Do not use a trailing slash after `v1`.
- **Model not listed**: Confirm `GET /v1/models` returns `tofy`; then check that the `models` key in `opencode.json` matches (e.g. `"tofy": { "name": "..." }`).
- **Paths differ**: OpenCode may use different config paths on your OS; see [OpenCode docs](https://opencode.ai/docs/) for the correct locations of `auth.json` and `opencode.json`.
- **No response / no generation animation**: (1) The server supports **streaming** (`stream: true`); if your client sends that, you should get SSE and a token-by-token animation. (2) The first reply can be slow (encoder + planner/world + decoder); wait for the decoder to finish. (3) Check the server terminal for errors.
- **Empty response in OpenCode**: (1) Ensure **both** decoders are set (`JEPA_USE_CANDLE_DECODER` + `JEPA_CANDLE_DECODER` and `JEPA_USE_TEXT_DECODER` + `JEPA_TEXT_DECODER`) so the server can do text reply. (2) The orchestrator head may predict Done on the first step; the server now forces at least one TextReply when no content has been generated yet. (3) Run with `--debug` or `JEPA_DEBUG=1` and watch the server terminal for errors or decoder output.
- **GPU not utilized**: The **decoder** (llama-completion / llama-cli) uses the GPU via `-ngl`. Set `JEPA_DECODER_NGL=99` (default) so all layers run on GPU. Ensure `JEPA_DECODER_MODEL` points to your GGUF and the decoder binary is built with GPU support.
- **`--no-conversation is not supported` / broken output**: The default decoder binary is **llama-completion** (non-interactive). If you see this error or banner/prompt echo in the response, set `JEPA_DECODER_BIN=llama-completion` and restart the server. If your llama.cpp build only provides `llama-cli`, use a build that includes `llama-completion` or see [DECODER_RUNTIME.md](DECODER_RUNTIME.md).

- **Speed stats**: Add **`--debug`** to the serve command so decoder stderr is shown on the server terminal (llama.cpp usually prints e.g. `Prompt: 116.1 t/s | Generation: 33.7 t/s` there). Response content is unchanged.

## Code path vs text path

When the server runs for OpenCode, generation uses two decoders:

- **Code path** → **codeDecoder**  
  The server turns planner-memory slots into code-decoder conditioning via the code decoder adapter. Train this decoder on **code** data (`code_prefix\tcompletion` pairs). Set `JEPA_USE_CANDLE_DECODER=1` and `JEPA_CANDLE_DECODER=<path>`. The decoder loads its own sibling vocab file or `JEPA_CANDLE_DECODER_VOCAB`.

- **Text path** → **textDecoder**  
  The server turns planner-memory slots into text-decoder conditioning via the text decoder adapter. The **textDecoder** generates chat replies. It shares the same decoder family as the code decoder, but is trained on **text** data. Set `JEPA_USE_TEXT_DECODER=1` and `JEPA_TEXT_DECODER=<path>`. The decoder loads its own sibling vocab file or `JEPA_TEXT_DECODER_VOCAB`.

**Action selection:** The server chooses Code vs Text from the **prompt**: if the prompt looks like code (e.g. contains `fn `, `impl `, `def `, `struct `, `->`, `::`), it uses the code path and codeDecoder; otherwise it uses the text path and textDecoder. You can run with one or both decoders; if a decoder is not set, the server falls back to llama.cpp or the stub for that action.

## Decoder size for your hardware

The in-repo decoder (codeDecoder / textDecoder) is a small transformer (embed + blocks + lm_head). Choose a preset that fits your GPU or CPU:

| Preset   | dim | layers | heads | ff_dim | ~params | ~VRAM (f32) | When to use |
|----------|-----|--------|-------|--------|---------|-------------|--------------|
| **Small**  | 256 | 2 | 4 | 1024 | ~7M  | ~30 MB  | 4GB GPU, or CPU-only; fast, lower quality. |
| **Medium** | 512 | 4 | 8 | 2048 | ~26M | ~100 MB | **Default.** 6–8GB GPU; good tradeoff. |
| **Large**  | 768 | 6 | 12 | 3072 | ~80M | ~320 MB | 8GB+ GPU; better quality, slower. |

The current build uses the **large** preset (~90M decoder: dim 768, 8 layers; see `DECODER_DIM`, `DECODER_LAYERS` in `src/tasks/world.rs`). To use small or medium, change those constants and re-train; inference VRAM is dominated by decoder + world model.

## Using the Candle decoders

You can use the in-repo decoders instead of llama.cpp. Both use cross-attention, but each has its own decoder adapter and decoder kind.

### 1. Train the world model (if not already)

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/ultrachat_pairs.txt 40000 32 768 128 6 8 256 64 --lambda 0.2
```

Replace encoder path with yours (see [RUNBOOK.md](RUNBOOK.md)). The world checkpoint includes planner memory, world transition, and the orchestrator head, but not the encoder or its vocab.

### 2. Train the decoder(s)

Same data format: one line per example, `context\treply`. The decoder is trained to predict the reply given the context and planner-derived conditioning from the frozen world model.

- **textDecoder** (chat): train on **ultrachat_pairs** (or similar dialog data); saved under `local_models/`:
  ```bash
  cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 16 128 --decoder-kind text --decoder-output local_models/text_decoder_90M.safetensors
  ```

- **codeDecoder** (code): train on **code prefix \t completion** pairs; default save path is `local_models/code_decoder_<size>.safetensors`:
  ```bash
  cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 16 128 --decoder-kind code --decoder-output local_models/code_decoder_90M.safetensors
  ```

Optional: `[steps] [batch] [max_seq] [dim] [num_layers] [num_heads] [planner_dim] [num_planner_slots]`, `--lr <float>`, `--init-decoder <path>`, `--decoder-output <path>`, `--decoder-vocab <path>`.

### 3. Serve with code and/or text decoders

Set env for the decoders you trained (one or both; no GGUF required if both are set):

```bash
# Code path (code completion)
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=./local_models/code_decoder_90M.safetensors
# optional: JEPA_DECODER_TEMP=0.7

# Text path (chat) — set both so OpenCode can do code and chat
export JEPA_USE_TEXT_DECODER=1
export JEPA_TEXT_DECODER=./local_models/text_decoder_90M.safetensors
# optional: JEPA_TEXT_DECODER_TEMP=0.7

cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

Then configure OpenCode as in §2–4 above. The server uses **codeDecoder** for code-like prompts and **textDecoder** for chat; if a decoder is not set for that action, it falls back to llama.cpp (if `JEPA_DECODER_MODEL` is set) or the stub.

### Testing in OpenCode

1. Start the server with both `JEPA_USE_TEXT_DECODER`/`JEPA_TEXT_DECODER` and optionally `JEPA_USE_CANDLE_DECODER`/`JEPA_CANDLE_DECODER` set.
2. **Chat:** Send a natural-language prompt (e.g. “Explain what a for loop does.”). The server should use the text path and textDecoder.
3. **Code:** Send a code fragment (e.g. “fn main() {”). The server should use the code path and codeDecoder.
4. Check the server log for errors; unset the Candle env vars to fall back to GGUF for that action.
