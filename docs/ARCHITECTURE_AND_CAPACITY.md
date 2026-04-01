# Architecture and using model capacity

How cross-action awareness works and how encoder/decoder size is chosen. See [RUNBOOK.md](RUNBOOK.md) for commands.

Current stack:

`encoder -> planner memory -> router/orchestrator -> decoder-specific adapter -> decoder`

## How each action model can know what the others do

The text model (when replying in chat) and the code model (when generating code) can correctly refer to what the other did **only if they share the same context**. In this stack that happens in one place: **the encoder**.

- **Encoder** sees a single string (the prompt) and produces token states.
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

The shared state now lives in planner-memory slots rather than a single bridge/code-world latent. **Current behavior:** the HTTP server (`--serve`) uses full-conversation encoding: it builds the prompt from the full `messages` array and passes that to the engine. For long chats, the engine keeps the last `max_seq` tokens so the planner memory reflects the most recent context.

## Encoder smaller than decoder — is that okay?

Yes. The encoder and planner memory compress the conversation into a compact internal state; the decoder does the actual token generation. It is normal for the decoder to be larger than the encoder because generation capacity usually dominates vocabulary, syntax, and long-range output quality.

## Fit with modern hardware

- **Shared encoder + full-conversation context** uses one forward pass over the full context; GPUs are good at that. Scaling encoder size (dim, layers, context length) and training on long dialogs lets you use more capacity where it matters most (one representation for all actions).
- **Separate decoders per action** (code vs text) keep each decoder focused and smaller; you can scale each to the capacity you want for that modality.
- **Single world model** (planner memory + transition + orchestrator) keeps the next internal state in one place; decoder-specific behavior sits in the adapters and decoder checkpoints.

Summary: cross-action awareness comes from shared full-context encoding plus shared planner memory, while decoder-specific behavior is isolated in each adapter/decoder pair.

## Multi-step replies and one action at a time

The **orchestrator** does **not** decide once per user message. It decides **per step** inside a single reply: the model can interleave text, code, writing a file, and running a CLI in one reply (e.g. brief text → code block → write file → run test). Only **one action is active at a time** (no parallel tool calls), so memory stays bounded.

**Flow:**

1. **Reply loop** (inside one assistant reply): we maintain `assistant_content` (what the assistant has produced so far). For each step (up to `MAX_ACTIONS_PER_REPLY`):
   - Build **current prompt** = full conversation + `"\nAssistant: "` + `assistant_content`.
   - **Orchestrator** returns the next action: `TextReply`, `Code`, `WriteFile`, `RunCli`, or `Done`.
   - If `Done`, we finish and return `assistant_content`.
  - If **decoder** action (`TextReply` or `Code`): encode current prompt → planner memory → transition → decoder adapter → run the appropriate decoder with a chunk of tokens; append the decoder output to `assistant_content`.
   - If **tool** action (`WriteFile` or `RunCli`): execute the tool (or stub); append the result (e.g. `[Wrote file]`, `[Ran command]`) to `assistant_content`.
2. **One action at a time**: we never run two decoders or two tools in parallel. We do one action → append result → decide next action. That keeps memory and control flow simple.
3. **Orchestrator as the router:** The orchestrator is a trained action head on top of the predicted planner state. It outputs logits over 5 actions (`TextReply`, `Code`, `WriteFile`, `RunCli`, `Done`). At inference, if the checkpoint has the head, its prediction is used; otherwise the engine falls back to a fixed step policy.

**Actions:** `TextReply` (chat), `Code` (code block), `WriteFile` (write code to file; stub for now), `RunCli` (run command; stub for now), `Done`. Decoder actions use the existing text/code decoders; tool actions use stubs until real file/CLI execution is wired.

**Memory:** The encoder sees `current_prompt` each step; if it exceeds `max_seq`, the existing truncation keeps the last `max_seq` tokens. No extra state beyond the growing `assistant_content` string.
