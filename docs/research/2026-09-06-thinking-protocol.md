# Thinking-mode strict-schema protocol qualification

The earlier thinking-mode failure at `fe5571d3` did not use the current strict
`response_format`, so it does not establish whether finite server-side thinking
and the repaired strict action protocol interoperate. This qualification tests
that prerequisite only. It makes no gameplay, reasoning-quality, or public ARC
performance claim.

Claim: Qwen3-8B Q4_K_M served by the pinned llama.cpp build can produce strict
valid actions under the production JSON schema with `reasoning=on` and a
256-token reasoning budget. The fixed population is the first five sealed
non-ARC capacity observations, in order, for model seeds 0 and 1. Each seed gets
a fresh server and all five sequential decisions. The only configuration change
from the qualified reference is `reasoning_mode: on`; sampler 0.7/0.8/20/0,
maximum output 1024, context 16384, compact lossless frames, history four,
prompt/physical batches 1024/1024, and full CUDA offload remain fixed.

Pass requires all ten completion responses to end with `finish_reason=stop`, all
ten actions to satisfy the production strict schema, valid nonnegative integer
prompt/completion usage, estimated versus actual prompt counts within 128, and
the context bound `actual + 1024 + 128 <= 16384`. At least one response per seed
must contain nonempty `reasoning_content`, proving that thinking mode was active.
Missing reasoning-token detail is recorded rather than inferred. Any protocol,
resource, deadline, identity, or cleanup failure fails the qualification.

The run has a 600-second global deadline, 120 seconds per server startup, and 60
seconds per decision. It requires clean reviewed pushed source whose production
core bytes equal qualified revision `92cce245`, the pinned model/server/config/
fixture/helper identities, 37/37 CUDA offload, no unrelated GPU compute process,
peak VRAM at most 95%, complete loopback request/response capture, exact owned
PID/process-group cleanup, and a verified manifest plus external digest.

This is implementation-smoke evidence. Failure falsifies the bounded protocol
claim and does not measure behavior. Passing permits later separately registered
thinking-policy evaluation; it does not enable automatic promotion or public
interaction. The run waits until the GPU is free and never reads game assets,
public observations, game source, or solutions.
