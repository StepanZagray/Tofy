# Architecture and using model capacity

How cross-action awareness works and how encoder/decoder size is chosen. See [RUNBOOK.md](RUNBOOK.md) for commands.

Current stack:

`strict LeJEPA encoder -> action-conditioned LeJEPA world transition -> planner memory -> downstream decoder adapter -> decoder`

## Current Code Layout

The runtime architecture above is now reflected more directly in the crate structure:

- [`src/main.rs`](../src/main.rs) only initializes tracing and dispatches modes.
- [`src/cli.rs`](../src/cli.rs) contains shared command-line/path helpers.
- [`src/config/latent.rs`](../src/config/latent.rs) contains latent train/eval configs.
- [`src/config/world.rs`](../src/config/world.rs) contains world, orchestrator, decoder, eval, and serve configs.
- [`src/tasks/latent.rs`](../src/tasks/latent.rs) implements the latent train/eval path.
- [`src/tasks/pipeline.rs`](../src/tasks/pipeline.rs) wires the canonical `train` pipeline: data prep and caches, encoder, world, high-world, code decoder (+ polish), and the hard Rust code eval stage.
- [`src/tasks/world.rs`](../src/tasks/world.rs) implements the world/high-world/orchestrator/decoder train paths and runtime engine.
- [`src/tasks/world_support.rs`](../src/tasks/world_support.rs) contains shared world/decoder metric and masking helpers that used to live inline in `world.rs`.

This means the codebase now separates:

- command parsing and path resolution
- typed stage configuration
- stage orchestration
- shared world/decoder helper logic

more cleanly than the earlier single-file entrypoint approach.

Training defaults follow the LeJEPA/LeWorldModel direction: encoder training uses online masked-view prediction plus SIGReg without EMA target updates, stop-gradient teacher targets, contrastive loss, or predictor heads; world training uses action-conditioned next-latent prediction plus SIGReg without router/inverse auxiliary losses. The autoregressive decoders remain downstream code/text emitters, not the world model.

## How Context Works

In this project, "context" means three different things:

1. **Conversation context at runtime**
2. **Training-pair context in dataset rows**
3. **Decoder autoregressive context during generation**

They are related, but they are not the same object.

### 1. Conversation context at runtime

When serving, the API takes the full `messages` array and turns it into one prompt string:

- `System: ...`
- `User: ...`
- `Assistant: ...`

So the encoder sees the whole conversation text, not only the last user turn.

Important detail: the encoder still has a per-forward-pass window of `max_seq`, but runtime context no longer has to be a pure hard truncation. The current serve/eval path can retain multiple encoder segments:

- the newest segment keeps full token-level encoder memory
- older segments are re-encoded one segment at a time and compressed into chunk/global/planner summaries
- the planner then attends over the concatenation of compressed older memory and full recent memory

So the runtime path keeps up to `TOFY_ENCODER_CONTEXT_SEGMENTS * max_seq` encoder tokens, with only the newest `TOFY_ENCODER_RECENT_FULL_SEGMENTS` segments preserved at full token resolution.

Recent encoder changes make that window cheaper to scale:

- token-level local attention now runs as a true sliding window instead of a dense masked `seq x seq` attention map
- chunk size grows with sequence length so the global latent path stays near a fixed number of chunk states instead of growing linearly with every extra token
- latent training can now sample multiple source segments with `TOFY_LATENT_CONTEXT_SEGMENTS`, keeping recent segments dense and reserving part of the window for older-history tokens via `TOFY_LATENT_HISTORY_RATIO`
- latent training can also run in `bf16` / `f16` with `TOFY_TRAIN_DTYPE`, which is mainly there to buy more useful context under the same VRAM budget

That means:

- the model is **conversation-aware**
- but its active runtime memory is still bounded by the encoder context window
- "context length" here means **encoder-token count**, not characters or words

The world/planner path can now also fold multiple state segments recurrently in planner space:

- each retained segment is encoded separately
- each segment is converted into planner slots
- those segment-level planner slots are folded with a recency-biased recurrent update when `TOFY_RECURSIVE_PLANNER_MEMORY=1`

That is cheaper than carrying every older token state forward at full resolution, and it is closer to a recurrent latent-memory setup than the earlier one-shot concatenation path.

### 2. Planner/world context

The encoder output is not passed directly to the decoders. Instead:

1. the encoder produces token/chunk/global states
2. planner memory compresses them into `num_latent_tokens` planner slots
3. the low-level action-conditioned transition model predicts the **next** planner state for a candidate primitive action
4. a high-level world model predicts longer-range planner states from macro-actions encoded from primitive action spans
5. optional downstream router/orchestrator heads can choose `TextReply`, `Code`, or `Done` for compatibility, but they are not part of the strict world objective

So after the encoder stage, the conversation is represented as a small latent memory rather than a long token sequence.

This is important because the model does **not** work like a normal single LLM with one giant KV-cache over the whole dialog. The shared conversation context is compressed into planner slots first.

The integrated high-level world model follows the Hierarchical Planning with
Latent World Models idea: it operates in the same planner-slot latent space as
the low-level world model, but conditions on a learned macro-action vector
instead of one primitive action id. At inference, HWM planning chooses a
macro-action subgoal first, then uses the low-level transition model to choose
the first primitive action toward that subgoal.

### 3. Decoder context at inference

The selected decoder gets context from **two places**:

- the raw prompt text tokenized with the decoder's own vocab
- the predicted planner slots through cross-attention

So the decoder is not generating from the latent alone. It still autoregresses over prompt/output tokens, but it is additionally conditioned by planner memory.

For Candle decoders:

- text decoder uses its own text vocab/tokenizer
- code decoder uses its own code-aware vocab/tokenizer

Because the tokenizers differ, `max_seq = 256` in the encoder and `max_seq = 128` or `192` in the code decoder do **not** mean the same exact amount of source text. Each module counts its own tokens.

### 4. Decoder context during training

For decoder training, each row is a pair:

`state<TAB>next`

Here:

- `state` = conditioning side / prior context
- `next` = target continuation

Teacher forcing is built as:

- `input = state + shifted(next)`
- `target = shifted(state) + next`

So the decoder training sequence length is effectively **`2 * max_seq`**, even though `max_seq` is still the configuration knob for each side individually.

This means `max_seq = 128` for the code decoder really means:

- up to `128` tokens from the left side
- up to `128` tokens from the right side
- up to `256` autoregressive positions inside the decoder loss

### 5. Dataset context

In the training data, "context" usually refers to the **left side** of the pair file.

Examples:

- chat pair: previous conversation or user request on the left, assistant reply on the right
- code pair: code prefix on the left, code continuation on the right

That dataset-level context is what teaches the world model and decoders what "continue from this state" means.

### 6. What gets forgotten first

When prompts get too long, the model forgets in this order:

1. the oldest prompt segments are dropped first once the segment budget is exceeded
2. within the retained budget, older segments are compressed before the newest segment
3. the retained memory is compressed into planner slots
4. the chosen decoder then generates using its own prompt tokens plus planner conditioning

So the bottleneck is not only decoder context. The main bottleneck is usually the **encoder window**, because everything downstream depends on what survived encoder truncation.

### 7. Practical implications

- If you want better long-chat consistency, increasing encoder `max_seq` matters more than only increasing decoder output length.
- If you want better code continuation quality, increasing code-decoder `max_seq` helps because the decoder teacher-forcing path sees longer prefix/continuation pairs.
- Since decoders have separate tokenizers, compare context lengths in **tokens per module**, not by raw text length.
- The planner slots are a compressed memory, so the system can preserve high-level intent even when exact token-level detail is partially lost.

## How each action model can know what the others do

The text model (when replying in chat) and the code model (when generating code) can correctly refer to what the other did **only if they share the same context**. In this stack that happens in one place: **the encoder**.

- **Encoder** sees a single string (the prompt) and produces token states.
- The encoder is hierarchical: local token blocks feed chunk states, chunk states exchange information globally, and learned global latent tokens hold whole-sequence summaries.
- **Planner memory** resamples those token states into planner slots.
- **World transition + orchestrator** operate on those planner slots.
- **Decoder adapters** transform planner slots into decoder-specific conditioning for code or text.

So:

- **If the encoder sees only the last user message**  
  The latent only encodes that message. The code decoder has no notion of “what the text assistant said before,” and the text decoder has no notion of “what code was just generated.” So they cannot correctly describe or refer to each other.

- **If the encoder sees the full conversation**  
  The latent encodes the whole dialog: prior user messages, prior assistant text replies, and prior assistant code outputs. Then:
  - The **text** decoder (when we use it) is conditioned on a latent that already includes “what code was generated” → it can correctly tell the user what the code does.
  - The **code** decoder is conditioned on a latent that includes “what the text assistant said” → it can stay consistent with prior explanations or instructions.

So **cross-action awareness** is achieved by:

1. **Feeding the full conversation into the encoder** (all messages so far, including every assistant code and text reply).
2. **Using the same encoded context** for the planner/router/decoder path, for whichever action is chosen.

The shared state now lives in planner-memory slots rather than a single bridge/code-world latent. **Current behavior:** the HTTP server (`--serve`) uses full-conversation encoding: it builds the prompt from the full `messages` array and passes that to the engine. For long chats, the engine can retain several encoder segments, compressing older segments into chunk/global/planner summaries so the planner memory reflects more than just the last raw `max_seq` tokens.

## Encoder smaller than decoder — is that okay?

Yes. The encoder and planner memory compress the conversation into a compact internal state; the decoder does the actual token generation. It is normal for the decoder to be larger than the encoder because generation capacity usually dominates vocabulary, syntax, and long-range output quality.

## Fit with modern hardware

- **Shared encoder + full-conversation context** uses one forward pass over the full context; GPUs are good at that. Scaling encoder size (dim, layers, context length) and training on long dialogs lets you use more capacity where it matters most (one representation for all actions).
- **Separate decoders per action** (code vs text) keep each decoder focused and smaller; you can scale each to the capacity you want for that modality.
- **Single world model** (planner memory + transition + orchestrator) keeps the next internal state in one place; decoder-specific behavior sits in the adapters and decoder checkpoints.

Summary: cross-action awareness comes from shared full-context encoding plus shared planner memory, while decoder-specific behavior is isolated in each adapter/decoder pair.

## Why the new encoder is more JEPA-like

- It predicts target representations through dedicated predictor heads instead of matching raw context states directly.
- It trains with multiscale losses over masked token targets, chunk latents, and global latent summaries.
- It uses structured masking so code examples more often hide meaningful regions such as identifiers, comments, and block boundaries rather than only random spans.

## Current Router Scope

The **orchestrator** currently predicts only actions that the runtime can actually execute:

1. `TextReply`
2. `Code`
3. `Done`

This is intentional. The runtime no longer pretends that file-writing or CLI execution are learned actions. `Done` is the only non-decoder action, and it is trained from explicit terminal rows in the world/orchestrator data.

Current reply flow:

1. Encode the full conversation prompt.
2. Planner memory compresses it into private slots.
3. The orchestrator picks `TextReply`, `Code`, or `Done`.
4. The transition model conditions the planner state on that action.
5. The matching decoder generates the response.

## Current memory levers

What is available now:

- true sliding-window local attention in the encoder
- adaptive chunk/global hierarchy for longer encoder context
- `--grad-accum <int>` for latent, world, and decoder training
- Candle decoder inference KV cache for incremental self-attention
- precomputed cross-attention K/V for the fixed planner/world latent during Candle decoding

What is not implemented yet:

- true activation checkpointing / recomputation wrappers

Why activation checkpointing is still missing:

- the current Candle stack in this repo does not expose a simple built-in checkpoint wrapper for these transformer blocks
- adding it cleanly would require a more invasive custom recompute path through encoder and decoder blocks, not just a config flag

Why KV-cache is still only partial:

- the GGUF / `llama.cpp` runtime has its own cache system
- the in-repo Candle decoder now caches self-attention and precomputes world-latent cross-attention K/V
- but there is still no trained recurrent-memory module, so runtime conversation context is bounded first by the configured encoder segment budget rather than being truly unbounded
