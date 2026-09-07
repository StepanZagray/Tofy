# Prepreview local agent: protocol repair qualification

The original capacity run atfe5571d3 loaded Qwen3-8B Q4_K_M and used7120MiB
sampled GPU memory at context16384 and prompt batch1024. The first response was
valid. The second exhausted1024 generated tokens while continuing prose after
the256-token reasoning cutoff. The strict parser rejected it; no public game
ran. Default server logs also failed to establish all-layer offload. Preserve
that failed run and its manifest; it cannot satisfy a qualification gate.

This new engineering test changes three explicitly named settings: supported
nonthinking mode (`--reasoning off`), constrained decoding with a schema for
exactly the currently available action/coordinate alternatives, and verbose
startup logging sufficient to establish layer placement. The first two can
change generated behavior; this is a combined protocol repair, not an isolated
causal performance experiment. Nonthinking mode removes the explicit thinking
trace and can reduce reasoning ability; it is an initial reference only.

Qwen's official documentation supports disabling thinking at the template
level and warns that greedy decoding in thinking mode can repeat indefinitely.
That warning applies to the original temperature0 setup. The precise local
interaction of forced cutoff, parser and grammar is not proved from that trace.
Related upstream reports describe thinking/grammar failures, but do not certify
this installed build. The installed help documents `--reasoning off` and
`--log-verbosity 5`. [Qwen guide](https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html),
[llama.cpp grammar issue](https://github.com/ggml-org/llama.cpp/issues/20345).

All other qualification settings and boundaries remain those in the
[original registration](2026-09-06-prepreview-agent-capacity.md): same official
model revision/whole-file hash, same server executable hash, all layers on CUDA,
F16 KV, flash on/fit off, one slot, max1024 output, temperature0, seed0, four
prior turns and compact lossless frames. The configured256 reasoning budget
is inactive in nonthinking mode; require no nonempty reasoning-content trace.
No training, game source/assets, public interaction, code tools or fallback.

Use the same five deterministic non-ARC frames in the same order. Reset the
model process and history for this new run. The initial candidate remains
context16384/physical1024 (logical1024), with the original fixed resource-only
fallback order. All five actions must pass the unchanged strict local parser,
including available action IDs and mandatory/bounded click coordinates. The
request schema prevents extra properties and excludes click fields for other
actions. Do not repair responses, choose a lucky sample or use random fallback.
The fifth request must contain all four factual prior turns and no private
fixture identifiers. Record exact request schemas and raw outputs.

Require server logs proving all layers on CUDA and the registered memory cap,
plus hard120s startup/60s decision/450s candidate/1200s global limits. Retain
and verify exact process/telemetry ownership and cleanup before sealing. Record
new clean reviewed pushed source, exact script/registration snapshots, commands,
package/model/server hashes and external manifest digest. Never reuse the failed
root. No simultaneous GPU training or model load.

PASS allows a separately registered native-engine development run; it does not
establish game reasoning, optimal decoding or a prize score. FAIL names the
remaining infrastructure/protocol problem before any new intervention. These
five prompts do not establish that all animation lengths fit the context.
The22 reserved public games remain untouched and absent from pretraining.
