# Running Tofy in OpenCode

Use [OpenCode](https://opencode.ai) with your local Tofy server so the coding agent runs your full pipeline (world model + decoder).

## Prerequisites

1. **Tofy server is running** (see [RUNBOOK.md](RUNBOOK.md) §11 for the start command). Default: `http://localhost:8080`.
2. **OpenCode** installed (e.g. from [opencode.ai](https://opencode.ai) or your package manager).

## Quick walkthrough

### 1. Start the Tofy server (if not already)

From your Tofy repo, with a GGUF decoder available:

```bash
export JEPA_DECODER_MODEL=./models/<your_model>.gguf
cargo run --release -- --serve model_world_10.00M.safetensors vocab_world.txt 0.0.0.0:8080 368 128 4 8 256
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

Chat or run agent tasks as usual. Requests go to your local Tofy server; the decoder is instructed that it is **Tofy**, a JEPA-style world-model agent.

**Token stream / generation animation:** The Tofy server supports **streaming** (`stream: true`). When OpenCode sends that, the server returns Server-Sent Events (SSE) so tokens appear as they’re generated and you get the usual typing animation. If you previously saw no animation and no response, ensure your OpenCode client sends `stream: true` (many do by default); the server now responds with SSE in that case.

## Verifying the connection

- **Health**: `curl http://localhost:8080/health` → `ok`
- **Models**: `curl http://localhost:8080/v1/models` → list including `tofy` with `owned_by: "Tofy"`

## Troubleshooting

- **Connection refused**: Ensure the Tofy server is running and the port in `baseURL` matches (default 8080).
- **Wrong base URL**: Use `http://localhost:8080/v1` (with `/v1`). Do not use a trailing slash after `v1`.
- **Model not listed**: Confirm `GET /v1/models` returns `tofy`; then check that the `models` key in `opencode.json` matches (e.g. `"tofy": { "name": "..." }`).
- **Paths differ**: OpenCode may use different config paths on your OS; see [OpenCode docs](https://opencode.ai/docs/) for the correct locations of `auth.json` and `opencode.json`.
- **No response / no generation animation**: (1) The server now supports **streaming** (`stream: true`); if your client sends that, you should get SSE and a token-by-token animation. (2) The first reply can be slow (encoder + world model + decoder); wait for the decoder (llama-cli) to finish. (3) Check the server terminal for errors.
- **GPU not utilized**: The **decoder** (llama-completion / llama-cli) uses the GPU via `-ngl`. Set `JEPA_DECODER_NGL=99` (default) so all layers run on GPU. Ensure `JEPA_DECODER_MODEL` points to your GGUF and the decoder binary is built with GPU support.
- **`--no-conversation is not supported` / broken output**: The default decoder binary is **llama-completion** (non-interactive). If you see this error or banner/prompt echo in the response, set `JEPA_DECODER_BIN=llama-completion` and restart the server. If your llama.cpp build only provides `llama-cli`, use a build that includes `llama-completion` or see [DECODER_RUNTIME.md](DECODER_RUNTIME.md).

- **Speed stats**: Add **`--debug`** to the serve command so decoder stderr is shown on the server terminal (llama.cpp usually prints e.g. `Prompt: 116.1 t/s | Generation: 33.7 t/s` there). Response content is unchanged.
