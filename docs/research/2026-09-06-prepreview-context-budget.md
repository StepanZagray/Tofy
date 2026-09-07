# Token-budgeted local reference repair

The ae4bc401 reference failed on seed0's bp35 third request:17063 prompt
tokens exceeded16384 context. The preceding ar25 episode completed128 actions,
all identical ineffective clicks,0/8 levels. Seed1 and cd82 never ran. Keep the
failed suite sealed; it is infrastructure evidence with a partial negative
behavior observation, not a completed paired reference. The original five
non-ARC requests and two-action ar25 smoke did not cover this context boundary.

Claim: exact server token counting plus eviction of oldest complete historical
turns prevents context overflow when the current complete observation fits.
Assumptions: the same server applies the same chat template/tokenizer as the
completion route, and tokenization reports are correct. The local inequality
prompt_tokens + max_output_tokens +128 <= context_size guarantees room under
those assumptions; actual endpoint parity must be measured. This does not prove
arbitrary observations fit, improve reasoning, or prevent ineffective actions.

Intervention only: before completion, apply the serving chat template and
tokenize the exact messages through loopback endpoints. Preserve the system
message and full current observation, including every animation layer/pixel.
Remove oldest whole historical observation/action/terminal-feedback groups
until the inequality holds. No row/layer truncation, hidden state access,
response repair or fallback action. If the current observation alone is too
large, fail explicitly before generation. Record retained/evicted history,
estimated prompt tokens, actual usage, preflight latency and completion cost.
Fitting completion payloads must remain identical in CPU parity tests.

Owning interface documentation: llama.cpp server POST /apply-template and
/tokenize, retrieved2026-09-06. Installed build10760/commit e2f1e6f needs its own
device check; current documentation is not proof of installed behavior.
[Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Qualification is engineering smoke, no gameplay promotion. Use unchanged
official Qwen3-8B Q4_K_M model/server hashes and resource/decoding settings from
the ae4bc401 qualification: context16384, prompt/physical1024/1024, all layers
CUDA, F16 KV, nonthinking, temperature0.7/top_p0.8/top_k20/min_p0,1024 output,
seed0, at most four prior turns. No optimization/training or concurrent GPU job.
Use all five original deterministic non-ARC fixtures, followed by five copies
of the largest original fixture with two animation layers (the second layer
changes one categorical pixel). This deterministic extension forces budget
pressure without using public frames. Require at least one history eviction,
ten strict valid actions, current frames intact, actual prompt count within128
tokens of estimate, and all requests below the context/output bound. Include a
CPU negative control whose current-only token count is too large; require zero
completion calls. Maximum120 seconds startup,60 per decision,900 total. A fail
blocks native qualification; do not select a lucky seed or mutate settings.

Then run a fresh bp35-only native smoke for four charged actions (no voluntary
RESET), seed0, maximum180 seconds/game,240 total,60 per decision,20 retries.
This exercises the previous third-request failure. It remains implementation
smoke. Use a dedicated exact one-game cache and the same corrected native driver.
Require protocol/token-budget integrity and native action reconciliation. No
game source or solutions are read; the four permitted observations/actions are
development interaction and are not training data.

Only after both smokes pass, repeat both full fixed seed0/1 development suites
under the original public-reference registration's population, caps and success
gate. The sole changed policy is available history under token pressure; this
can affect gameplay. Do not claim an isolated performance gain against the
incomplete old run. ar25's seed0 fitting prompts are expected to remain unchanged
and its repeated-click failure may persist. Outcomes on the other games and
seed1 remain unknown. Both suites, failures and costs must be reported;22 games
remain reserved. No longer budget is authorized by this registration.

Launch from a reviewed pushed clean revision, record dependency/model/server/
script/config/cache hashes, use never-reused roots and exact owned PID cleanup.
Verify and seal finalized trees; save external manifest digests. Classify a
completed fixed paired reference as completed development evidence in its outer
analysis, while naming the legacy driver's exploratory wire label. No seed,
checkpoint or threshold selection is allowed. A failed paired reference remains
failed infrastructure evidence. Campaign continuation requires analyzing both
context and repeated-action behavior before the next registered intervention.
