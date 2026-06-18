# Decoder runtime and environment variables

Decoder backends and env vars for `--serve`. The server picks the decoder by action (code vs text) after routing from context compressor. See [RUNBOOK.md](RUNBOOK.md) for serve commands and [OPENCODE.md](OPENCODE.md) for OpenCode setup.

## Backends

| Backend | Use case | Env vars |
|--------|----------|----------|
| **GGUF (llama.cpp)** | General text/code via a GGUF model; no latent-vector conditioning | `JEPA_DECODER_MODEL`, optional: `JEPA_DECODER_BIN`, `JEPA_DECODER_CTX`, `JEPA_DECODER_NGL`, `JEPA_DECODER_TEMP`, `JEPA_DECODER_REPEAT_PENALTY` |
| **Candle codeDecoder** | In-repo code-specialist decoder | `JEPA_USE_CANDLE_DECODER=1`, `JEPA_CANDLE_DECODER=<path>`, optional `JEPA_CANDLE_DECODER_VOCAB=<path>`, optional `JEPA_DECODER_TEMP`, `JEPA_DECODER_REPEAT_PENALTY`, `JEPA_DECODER_REPEAT_LAST_N`, `JEPA_DECODER_TOP_K`, `JEPA_DECODER_TOP_P`, `JEPA_CANDLE_DECODER_CTX` |
| **Candle textDecoder** | In-repo text-generalist decoder | `JEPA_USE_TEXT_DECODER=1`, `JEPA_TEXT_DECODER=<path>`, optional `JEPA_TEXT_DECODER_VOCAB=<path>`, optional `JEPA_TEXT_DECODER_TEMP`, `JEPA_DECODER_REPEAT_PENALTY`, `JEPA_DECODER_REPEAT_LAST_N`, `JEPA_DECODER_TOP_K`, `JEPA_DECODER_TOP_P`, `JEPA_CANDLE_DECODER_CTX` |

You can use GGUF only, Candle only, or both; the server chooses the decoder from the prompt/action.

## GGUF / llama.cpp

- **`JEPA_DECODER_MODEL`** — Path to the `.gguf` file. If unset, the binary looks under `./models/` for a GGUF.
- **`JEPA_DECODER_BIN`** — Binary name (default `llama-completion`). Use `llama-completion` for chat; if you only have `llama-cli`, use a build that provides `llama-completion` or see OpenCode/server logs for alternatives.
- **`JEPA_DECODER_CTX`** — Context size (optional).
- **`JEPA_DECODER_NGL`** — GPU layers (optional).
- **`JEPA_DECODER_TEMP`** — Sampling temperature (optional).
- **`JEPA_DECODER_REPEAT_PENALTY`** — Repeat penalty (optional).
GGUF decoding no longer receives context/state vectors as textual latent summaries. Use the Candle decoder path for latent conditioning; the GGUF backend is now a plain prompt-only fallback. If the decoder model is not found, the server prints a message asking you to set `JEPA_DECODER_MODEL` or place a `.gguf` under `./models`.

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

Additional Candle decoding controls:

- **`JEPA_DECODER_REPEAT_PENALTY`** — Repetition penalty applied over recent generated/prompt tokens. Default: code `1.12`, text `1.08`.
- **`JEPA_DECODER_REPEAT_LAST_N`** — Number of recent tokens to consider for repetition penalty. Default: code `160`, text `96`.
- **`JEPA_DECODER_TOP_K`** — Keep only the top-k logits before sampling. Default: code `40`, text `0`.
- **`JEPA_DECODER_TOP_P`** — Nucleus sampling cutoff. Default: code `0.92`, text `1.0`.
- **`JEPA_CANDLE_DECODER_CTX`** — Maximum prompt tokens kept by the Candle runtime before generation.
- **`TOFY_DECODER_CONDITION_BUDGET`** / **`JEPA_DECODER_CONDITION_BUDGET`** — Keep only the most recent N context-conditioning slots before the decoder conditioning adapter. `0` keeps the shape but zeros conditioning, which is useful for ablations.
- **`TOFY_DECODER_CROSS_ATTN_SCHEDULE`** — Controls which decoder layers use context/state cross-attention. Supported values: `all`, `every-2nd`, `every-3rd`, `last-only`.
- **`TOFY_DECODER_CSA_TOPK`** — Number of compressed long-range blocks kept per query by the on-device DSA-style top-k mask in compressed sparse self-attention. Standalone default: `8`; pipeline default: `16`.
- **`TOFY_DECODER_LATENT_PREFIX`** — Enables prefix-LM-style latent memory tokens in decoder self-attention. Default: `1`; set `0` to compare against cross-attention-only conditioning.

The Candle path now uses:

- batched prompt prefill over the full prompt once before incremental decode
- action-aware local latent-plan conditioning in the decoder conditioning adapter: exact recent context slots, compressed older context blocks, query-adaptive cross-attention retrieval, action embeddings, and gated latent injection
- prefix-LM-style latent self-attention memory: decoder-facing context slots are projected into prefix tokens visible to prompt/generated tokens, while generated text remains causal
- adaptive recurrent latent test-time reasoning before decoder conditioning: the runtime repeatedly applies the action-conditioned transition, anchors proposals near the selected next-action latent, and keeps the best goal/route/stability-scored state
- DeepSeek-V4-inspired hybrid self-attention: sliding local attention, compressed sparse anchor layers with query-selected compressed blocks, and separate heavily compressed global-summary layers
- incremental self-attention KV cache with exact recent tokens plus compressed older blocks for compressed schedules
- precomputed cross-attention K/V for the fixed context/state latent
- repeat penalty and optional top-k/top-p sampling controls

Decoder checkpoints written after this change include `conditioner=action_aware_local_plan_v1` in the sibling `.meta.txt` file. Older decoder checkpoints that do not contain the action-aware conditioner weights should be retrained or used only with matching older binaries.

## Latent test-time reasoning

Before the Candle or GGUF decoder receives conditioning, Tofy can refine the planned context slots in latent space. This follows the recurrent-depth test-time compute direction: spend compute on latent state updates, then decode once. Code eval disables this by default unless `TOFY_LATENT_REASONING` is already set, so eval remains comparable to direct decoder training. Pass `--pi-agent-env` to `--eval-code-assistant` to use the same default as serve/Pi.

Useful controls:

- `TOFY_LATENT_REASONING=0|1` enables adaptive recurrent latent refinement, default `1`.
- `TOFY_LATENT_REASONING_STEPS=<int>` sets the maximum refinement depth, default `8` for code-like requests and `3` for text.
- `TOFY_LATENT_REASONING_MIN_STEPS=<int>` sets the minimum depth before early stopping, default `2` for code-like requests and `1` for text.
- `TOFY_LATENT_REASONING_PATIENCE=<int>` stops after this many non-improving steps beyond the minimum, default `2`.
- `TOFY_LATENT_REASONING_ALPHA=<float>` blends each recurrent proposal with the selected next-action latent anchor, default `0.35`.
- `TOFY_LATENT_REASONING_GOAL_WEIGHT`, `TOFY_LATENT_REASONING_ROUTE_WEIGHT`, and `TOFY_LATENT_REASONING_STABILITY_WEIGHT` tune selection between goal alignment, route confidence, and drift control.

## Recursive decoder

Decoder backends can be wrapped with the generic recursive decoder scaffold (`TOFY_DECODER_RLM=1`, default). The wrapper keeps the full request as external environment state, splits long prompts into semantic work units, executes a small RLM command program, invokes `SUB_RLM` recursively on bounded snippets, and joins stored sub-call outputs as the final response. Short code prompts no longer recurse just because the action is `code`; they recurse only when they meet `TOFY_DECODER_RLM_MIN_CHARS`. Code eval also defaults `TOFY_DECODER_RLM=0` unless explicitly overridden or run with `--pi-agent-env`.

Useful controls:

- `TOFY_DECODER_RLM=0` disables the recursive decoder wrapper.
- `TOFY_DECODER_RLM_ACTIONS=<csv>` selects wrapped actions, default `code,text,text_reply`.
- `TOFY_DECODER_RLM_MIN_CHARS=<n>` is the prompt length threshold for recursive decoding, default `3600`.
- `TOFY_DECODER_RLM_CHUNK_CHARS=<n>` sets semantic work-unit size, default `2400`.
- `TOFY_DECODER_RLM_MAX_UNITS=<n>` caps root work units, default `8`.
- `TOFY_DECODER_RLM_MAX_DEPTH=<n>` caps recursive `SUB_RLM` depth, default `3`.
- `TOFY_DECODER_RLM_MAX_OPS=<n>` caps root command execution.
- `TOFY_DECODER_RLM_LEAF_TOKENS=<tokens>` sets each leaf generation budget, default `256`.
- `TOFY_DECODER_RLM_MODEL_PROGRAM=1` asks the decoder to draft the root command program; default `0` uses the deterministic semantic chunk program.
- `TOFY_DECODER_RLM_PROGRAM_TOKENS=<tokens>` sets the root program generation budget, default `192`.

## Pi-Style Tool-Calling Decoder

Tofy can run an agentic decoder loop around the existing text/code decoder. The decoder emits one structured tool call, Tofy executes the matching bash command from a Pi-style Markdown skill, appends the `<tool_result>` to the transcript, re-encodes that expanded transcript, and repeats until the decoder returns a final answer or the step budget is exhausted.

Enable it with either:

```bash
TOFY_AGENTIC_DECODER=1
TOFY_TOOL_FILE=.pi/skills/read-file/SKILL.md
```

or point at a directory:

```bash
TOFY_AGENTIC_DECODER=1
TOFY_TOOL_DIR=.pi/skills
```

Discovered default locations follow Pi conventions first:

- `.pi/skills`
- `.agents/skills`
- `~/.pi/agent/skills`
- `~/.agents/skills`

Compatibility fallbacks are also checked: `TOOLS.md`, `.tofy/tools.md`, `.tofy/tools`, `.pi/prompts`, and `docs/TOOLS.md`.

Supported Pi skill shape:

````markdown
---
name: read-file
description: Read a file from the current repository when file contents are needed.
---

# Read File

## Usage

```bash
sed -n '1,220p' "$TOFY_ARG_PATH"
```
````

The tool call emitted by the decoder should be:

```xml
<tool_call>{"tool":"read-file","args":{"path":"src/lib.rs"}}</tool_call>
```

Arguments are exposed to bash as `TOFY_ARG_<UPPER_KEY>`. Tofy also shell-quotes and substitutes `{{key}}` and `<key>` placeholders. Commands run from the skill file's directory with `TOFY_SKILL_DIR` set there and `TOFY_WORKSPACE` set to the served repository root, so Pi-style relative `./scripts/...` commands work.

Useful controls:

- `TOFY_AGENTIC_MAX_STEPS=<n>` caps tool-call rounds, default `4`.
- `TOFY_AGENTIC_STEP_TOKENS=<n>` sets each tool-call decoding budget, default `384`.
- `TOFY_TOOL_TIMEOUT_MS=<n>` caps each bash tool process, default `10000`.
- `TOFY_TOOL_RESULT_CHARS=<n>` caps stdout/stderr returned to the next encoder pass, default `12000`.
