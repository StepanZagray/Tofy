# Prepreview local language agent: resource and protocol qualification

This is a preparation test, not ARC performance evidence. The independent
language-model path does not depend on the online effect CNN. Hypothesis:
the pinned Qwen3-8B Q4_K_M model can serve the existing strict local-chat action
protocol with compact lossless observations and four factual prior turns on
the local 8GiB GPU. No optimizer runs and no public game data is read here.

Use official Qwen/Qwen3-8B-GGUF revision
`7c41481f57cb95916b40956ab2f0b139b296d974`, file
`Qwen3-8B-Q4_K_M.gguf`, bytes5027783488, SHA256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
The May2025 artifact predates the July17,2025 public preview; this is chronology
evidence against public-level pretraining, not an exhaustive corpus audit.
Verify the download identity and whole-file hash before serving. Record actual
llama-server executable hash/version, source revision, package versions, GPU,
command/configuration, process IDs, latency, token use and sampled device memory.
Launch from reviewed pushed clean source, create new roots, and seal artifacts
only after the exact server/telemetry processes have stopped.

Inference configuration: all layers on CUDA, flash attention on, automatic fit
off, one slot, F16 KV cache (the server default), logical prompt batch1024,
reasoning budget256, max output1024, temperature0, seed0, compact frame format,
four prior observation/action pairs. There is no Python tool, search procedure,
external API, persistent learned memory or trained goal/value module. A local
client action uses model inference; the connected-component summaries are
generic deterministic geometry, not game-specific objects or goals.

Resource candidates in fixed order: context16384 with physical prompt batch1024,
then16384/512, then8192/1024, then8192/512. Stop selection at the first candidate
that meets all preparation gates; the batch candidates are bounded by logical1024.
The context is smaller than the server's32768 default because F16 KV allocation
at32768 plus these weights is expected to exceed this GPU's practical headroom.
Selection is based on resources/protocol only; discard any model-quality inference.
Retry a new candidate only after server OOM/startup failure, excess memory, or
context overflow. Invalid action/response is a protocol failure requiring analysis,
not a reason to pick a lucky candidate. Preserve every failed candidate root.

For each candidate, first load and verify the model alias. Then issue five
sequential, deterministic non-ARC observations: uniform background; a small
geometric arrangement of distinct palette blocks; a shifted version of those
blocks; a dense categorical checker pattern; and a second dense pattern. All
are64x64 and contain only one animation layer. Available inputs are1..7;
there is no synthetic goal or simulator and no performance label. These calls
populate the full four-turn history and exercise dense-frame encoding. Require
all responses to pass the unchanged strict available-action/coordinate parser,
finite positive recorded latency, and no context overflow. This establishes
only that those exact prompt sizes fit; longer animations can still exceed them.

Maximum120 seconds startup,60 seconds per decision,450 seconds/candidate,
20 minutes total including integrity checks. A200ms GPU sampler must show peak
memory below95% of8151MiB and no unrelated concurrent compute process. Retain
sampled peaks as a lower bound on instantaneous peaks, not allocator proof.
Do not change the desktop, clocks or power limits. No training effective batch
or accumulation applies; report the selected inference prompt batch explicitly.

PASS allows a separately registered engine-backed development screen; it does
not imply useful gameplay or a competitive score. FAIL requires analysis of
the named resource/protocol failure. Both outcomes leave the22 reserved public
games untouched. The campaign continues after this preparation stage.
