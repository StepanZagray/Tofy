# Architecture and using model capacity

How cross-action awareness works and how encoder/decoder size is chosen. See [RUNBOOK.md](RUNBOOK.md) for commands.

Current stack:

`strict LeJEPA encoder -> LeJEPA action-state transition -> context compressor -> downstream decoder conditioning adapter -> decoder`

## Current Code Layout

The runtime architecture above is now reflected more directly in the crate structure:

- [`src/main.rs`](../src/main.rs) only initializes tracing and dispatches modes.
- [`src/cli.rs`](../src/cli.rs) contains shared command-line/path helpers.
- [`src/config/latent.rs`](../src/config/latent.rs) contains latent train/eval configs.
- [`src/config/world.rs`](../src/config/world.rs) contains world, action classifier, decoder, eval, and serve configs.
- [`src/tasks/latent.rs`](../src/tasks/latent.rs) implements the latent train/eval path.
- [`src/tasks/pipeline.rs`](../src/tasks/pipeline.rs) wires the canonical `train` pipeline: data prep and caches, encoder, world, high-world, base code decoder, Go execution-feedback decoder, and hard Go compile/test eval for decoder promotion.
- [`src/tasks/world.rs`](../src/tasks/world.rs) implements the world/high-world/action classifier/decoder train paths and runtime engine.
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
- older segments are re-encoded one segment at a time and compressed into chunk/global/context summaries
- the context compressor then attends over the concatenation of compressed older memory and full recent memory

For multi-segment context, the context compressor can also use hybrid memory before
slot cross-attention. The newest memory tail remains exact, older memory is
compressed into learned salience-pooled block summaries, a small set of high
salience old tokens is carried forward exactly, and the compressor's own slot
queries retrieve a small set of old-memory summaries. This keeps the shared
world state bounded without forcing every old detail through one fixed average.
The knobs are `TOFY_CONTEXT_HYBRID_MEMORY`,
`TOFY_CONTEXT_HYBRID_EXACT_TAIL`, `TOFY_CONTEXT_HYBRID_BLOCK_SIZE`, and
`TOFY_CONTEXT_RETRIEVAL_SLOTS`. `TOFY_CONTEXT_EXACT_OLD_TOKENS` controls the
learned exact old-token carry-forward budget.

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

The context/state path can now also fold multiple state segments recurrently in context-slot space:

- each retained segment is encoded separately
- each segment is converted into context slots
- those segment-level context slots are folded with a recency-biased recurrent update when `TOFY_RECURSIVE_CONTEXT_COMPRESSION=1`

That is cheaper than carrying every older token state forward at full resolution, and it is closer to a recurrent latent-memory setup than the earlier one-shot concatenation path.

### 2. Planner/world context

The encoder output is not passed directly to the decoders. Instead:

1. the encoder produces token/chunk/global states
2. context compressor compresses them into `num_latent_tokens` context slots
3. the low-level action-conditioned transition model predicts the **next** context state for a candidate primitive action using a 6-block, 16-head LeWorldModel-style predictor with per-block action-conditioned normalization
4. a macro-action state transition predicts longer-range context states from macro-actions encoded from primitive action spans
5. optional downstream router/action classifiers can choose `TextReply`, `Code`, or `Done` , but they are not part of the strict world objective

So after the encoder stage, the conversation is represented as a small latent memory rather than a long token sequence.

This is important because the model does **not** work like a normal single LLM with one giant KV-cache over the whole dialog. The shared conversation context is compressed into context slots first.

The integrated macro-action state transition follows the Hierarchical Planning with
Latent World Models idea: it operates in the same context-slot latent space as
the low-level world model, but conditions on a learned macro-action vector
instead of one primitive action id. At inference, HWM planning chooses a
macro-action subgoal first, then uses the low-level transition model to choose
the first primitive action toward that subgoal.

After the first action is selected, runtime can spend extra test-time compute in
latent space before decoding. `TOFY_LATENT_REASONING=1` enables a recurrent
depth pass that repeatedly applies the action-conditioned transition, anchors the
state near the selected next-action latent, and keeps the best intermediate state
under a goal/route/stability score. Code requests get a deeper default latent
budget than text replies, so hard code paths can deliberate without emitting
extra chain-of-thought tokens.

### 3. Decoder context at inference

The selected decoder gets context from **two places**:

- the raw prompt text tokenized with the decoder's own vocab
- the predicted context slots through cross-attention

So the decoder is not generating from the latent alone. It still autoregresses over prompt/output tokens, but it is additionally conditioned by context compressor.

For Candle decoders:

- text decoder uses its own text vocab/tokenizer
- code decoder uses its own code-aware vocab/tokenizer
- decoder conditioning adapters do not turn context vectors into text. They build a latent memory from exact recent context slots plus compressed older context blocks, then use learned query/cross-attention weights and a gated latent path to produce decoder-facing conditioning slots.
- decoder-facing context slots are also projected into prefix-LM-style latent memory tokens when `TOFY_DECODER_LATENT_PREFIX=1`: prompt/generated tokens can attend to those latent prefix tokens, while generated text remains causal.
- decoder self-attention uses RoPE plus a hybrid long-context schedule inspired by DeepSeek-V4: the first layer is sliding local attention, anchor/final layers use compressed sparse long-range attention with query-selected compressed blocks, and the remaining layers use separate heavily compressed global-summary attention
- context/state cross-attention can be budgeted and scheduled at eval/runtime with `TOFY_DECODER_CONDITION_BUDGET` and `TOFY_DECODER_CROSS_ATTN_SCHEDULE`

Because the tokenizers differ, `max_seq = 256` in the encoder and `max_seq = 192` in the 8 GB code decoder do **not** mean the same exact amount of source text. Each module counts its own tokens.

### 4. Decoder context during training

For decoder training, each row is a pair:

`state<TAB>next`

Here:

- `state` = conditioning side / prior context
- `next` = target continuation

Teacher forcing is built as:

- `input = state + shifted(next)`
- `target = shifted(state) + next`

The decoder objective can include a conditioning-margin term. It compares the matched context/state latent against configurable negative conditioning (`TOFY_DECODER_CONDITIONING_NEGATIVES`, default `zero,shuffle`) and logs `zero_gain`, `shuffle_gain`, and `hard_negative_gain` when ablation metrics are enabled. The mismatched negatives matter because they check whether the decoder is using the right latent, not merely reacting to any nonzero conditioning vector.

So the decoder training sequence length is effectively **`2 * max_seq`**, even though `max_seq` is still the configuration knob for each side individually.

This means `max_seq = 192` for the current 8 GB code-decoder profile really means:

- up to `192` tokens from the left side
- up to `192` tokens from the right side
- up to `384` autoregressive positions inside the decoder loss

If either side is longer than its per-side budget, training keeps the tail of that side. The newest prompt tokens and the end of the target completion are usually where instructions, compiler feedback, return values, and closing braces live.

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
3. the retained memory is compressed into context slots
4. the chosen decoder then generates using its own prompt tokens plus context conditioning

So the bottleneck is not only decoder context. The main bottleneck is usually the **encoder window**, because everything downstream depends on what survived encoder truncation.

### 7. Practical implications

- If you want better long-chat consistency, increasing encoder `max_seq` matters more than only increasing decoder output length.
- If you want better code continuation quality, increasing code-decoder `max_seq` helps because the decoder teacher-forcing path sees longer prefix/continuation pairs.
- Since decoders have separate tokenizers, compare context lengths in **tokens per module**, not by raw text length.
- The context slots are a compressed memory, so the system can preserve high-level intent even when exact token-level detail is partially lost.

## How each action model can know what the others do

The text model (when replying in chat) and the code model (when generating code) can correctly refer to what the other did **only if they share the same context**. In this stack that happens in one place: **the encoder**.

- **Encoder** sees a single string (the prompt) and produces token states.
- The encoder is hierarchical: local token blocks feed chunk states, chunk states exchange information globally, and learned global latent tokens hold whole-sequence summaries.
- **Context compressor** resamples those token states into context slots.
- **Action-conditioned state transition + action classifier** operate on those context slots.
- **Decoder conditioning adapters** transform context slots into decoder-specific conditioning for code or text.

So:

- **If the encoder sees only the last user message**  
  The latent only encodes that message. The code decoder has no notion of “what the text assistant said before,” and the text decoder has no notion of “what code was just generated.” So they cannot correctly describe or refer to each other.

- **If the encoder sees the full conversation**  
  The latent encodes the whole dialog: prior user messages, prior assistant text replies, and prior assistant code outputs. Then:
  - The **text** decoder (when we use it) is conditioned on a latent that already includes “what code was generated” → it can correctly tell the user what the code does.
  - The **code** decoder is conditioned on a latent that includes “what the text assistant said” → it can stay consistent with prior explanations or instructions.

So **cross-action awareness** is achieved by:

1. **Feeding the full conversation into the encoder** (all messages so far, including every assistant code and text reply).
2. **Using the same encoded context** for the context compressor/router/decoder path, for whichever action is chosen.

The shared state now lives in context-compressor slots rather than a single bridge/code-world latent. **Current behavior:** the HTTP server (`--serve`) uses full-conversation encoding: it builds the prompt from the full `messages` array and passes that to the engine. For long chats, the engine can retain several encoder segments, compressing older segments into chunk/global/context summaries so the context compressor reflects more than just the last raw `max_seq` tokens.

## Encoder smaller than decoder — is that okay?

Yes. The encoder and context compressor compress the conversation into a compact internal state; the decoder does the actual token generation. It is normal for the decoder to be larger than the encoder because generation capacity usually dominates vocabulary, syntax, and long-range output quality.

## Fit with modern hardware

- **Shared encoder + full-conversation context** uses one forward pass over the full context; GPUs are good at that. Scaling encoder size (dim, layers, context length) and training on long dialogs lets you use more capacity where it matters most (one representation for all actions).
- **Separate decoders per action** (code vs text) keep each decoder focused and smaller; you can scale each to the capacity you want for that modality.
- **Single world model** (context compressor + transition + action classifier) keeps the next internal state in one place; decoder-specific behavior sits in the adapters and decoder checkpoints.

Summary: cross-action awareness comes from shared full-context encoding plus shared context compressor, while decoder-specific behavior is isolated in each adapter/decoder pair.

## Why the new encoder is more JEPA-like

- It predicts target representations through dedicated predictor heads instead of matching raw context states directly.
- It trains with multiscale losses over masked token targets, chunk latents, and global latent summaries.
- It uses structured masking so code examples more often hide meaningful regions such as identifiers, comments, and block boundaries rather than only random spans.

## Current Router Scope

The **action classifier** currently predicts only actions that the runtime can actually execute:

1. `TextReply`
2. `Code`
3. `Done`

This is intentional. The runtime no longer pretends that file-writing or CLI execution are learned actions. `Done` is the only non-decoder action, and it is trained from explicit terminal rows in the world/action classifier data.

Current reply flow:

1. Encode the full conversation prompt.
2. Context compressor compresses it into private slots.
3. The action classifier picks `TextReply`, `Code`, or `Done`.
4. The transition model conditions the context state on that action.
5. The matching decoder generates the response.

## Current memory levers

What is available now:

- true sliding-window local attention in the encoder
- adaptive chunk/global hierarchy for longer encoder context
- `--grad-accum <int>` for latent, world, and decoder training
- Candle decoder inference KV cache for incremental self-attention
- Candle decoder hybrid local/compressed self-attention for longer autoregressive context
- precomputed cross-attention K/V for the fixed context/state latent during Candle decoding
- learned salience pooling for old context-compressor memory blocks
- exact high-salience old-token carry-forward before context slot cross-attention
- world-model auxiliary post-state and chained rollout losses for multi-step latent stability
- adaptive recurrent latent test-time reasoning before decoder conditioning

What is not implemented yet:

- true activation checkpointing / recomputation wrappers

Why activation checkpointing is still missing:

- the current Candle stack in this repo does not expose a simple built-in checkpoint wrapper for these transformer blocks
- adding it cleanly would require a more invasive custom recompute path through encoder and decoder blocks, not just a config flag

Why KV-cache is still only partial:

- the GGUF / `llama.cpp` runtime has its own cache system
- the in-repo Candle decoder now caches self-attention and precomputes world-latent cross-attention K/V
- but there is still no unbounded persistent memory store, so runtime conversation context is bounded first by the configured encoder segment budget rather than being truly unbounded
