# Decoder runtime and environment variables

Decoder backends and env vars for `--serve`. The server picks the decoder by action (code vs text) after routing from planner memory. See [RUNBOOK.md](RUNBOOK.md) for serve commands and [OPENCODE.md](OPENCODE.md) for OpenCode setup.

## Backends

| Backend | Use case | Env vars |
|--------|----------|----------|
| **GGUF (llama.cpp)** | General text/code via a GGUF model | `JEPA_DECODER_MODEL`, optional: `JEPA_DECODER_BIN`, `JEPA_DECODER_CTX`, `JEPA_DECODER_NGL`, `JEPA_DECODER_TEMP`, `JEPA_DECODER_REPEAT_PENALTY` |
| **Candle codeDecoder** | In-repo code-specialist decoder | `JEPA_USE_CANDLE_DECODER=1`, `JEPA_CANDLE_DECODER=<path>`, optional `JEPA_CANDLE_DECODER_VOCAB=<path>`, optional `JEPA_DECODER_TEMP` |
| **Candle textDecoder** | In-repo text-generalist decoder | `JEPA_USE_TEXT_DECODER=1`, `JEPA_TEXT_DECODER=<path>`, optional `JEPA_TEXT_DECODER_VOCAB=<path>`, optional `JEPA_TEXT_DECODER_TEMP` |

You can use GGUF only, Candle only, or both; the server chooses the decoder from the prompt/action.

## GGUF / llama.cpp

- **`JEPA_DECODER_MODEL`** — Path to the `.gguf` file. If unset, the binary looks under `./models/` for a GGUF.
- **`JEPA_DECODER_BIN`** — Binary name (default `llama-completion`). Use `llama-completion` for chat; if you only have `llama-cli`, use a build that provides `llama-completion` or see OpenCode/server logs for alternatives.
- **`JEPA_DECODER_CTX`** — Context size (optional).
- **`JEPA_DECODER_NGL`** — GPU layers (optional).
- **`JEPA_DECODER_TEMP`** — Sampling temperature (optional).
- **`JEPA_DECODER_REPEAT_PENALTY`** — Repeat penalty (optional).
- **`JEPA_DECODER_COND_FORMAT`** — Conditioning format for the server (default `chunks`).

If the decoder model is not found, the server prints a message asking you to set `JEPA_DECODER_MODEL` or place a `.gguf` under `./models`.

## Candle decoders

Train with `--train-decoder --decoder-kind code` or `--train-decoder --decoder-kind text` (see [RUNBOOK.md](RUNBOOK.md)).

Each decoder now owns its own vocab file:

- `code_decoder_*.safetensors` -> `code_decoder_*.vocab.txt`
- `text_decoder_*.safetensors` -> `text_decoder_*.vocab.txt`

At serve time, the server loads:

- encoder checkpoint + encoder vocab
- pure world-model checkpoint
- selected decoder checkpoint + decoder vocab

If a decoder vocab env var is not set explicitly, Tofy looks for a sibling `.vocab.txt` file next to the decoder checkpoint.
