# Factual transition ledger readout diagnostic registration

Date: September 6, 2026 CDT

## Claim and decision

On the fixed non-ARC readout population below, adding the production factual
transition ledger to unchanged raw chronological history may improve selection
of the previously observed board-changing coordinate action. “Better” means
that, on each of model seeds 0 and 1 separately, ledger ON scores at least 21/24
and exceeds ledger OFF by at least 3/24. Before effect scoring, every grounding
cell must score at least 11/12. This is an empirical prerequisite for this exact
prompt representation. It is not evidence of ARC play, hidden-goal inference,
general memory quality, or public performance.

## Frozen population and arms

- Fixture generator: production `python/tofy_arc3/readout_fixtures.py` from the
  reviewed launch revision.
- Fresh layout seed: `2060908`.
- Exact visible-plus-oracle fixture SHA-256:
  `f8e5d8151eab3f788469819b8079d7d08f1dbb19cb317f818119bced0e3a2e31`.
- Model seeds: `0`, `1`; no seed or item exclusion.
- Ledger OFF: raw chronological history, `factual_transition_ledger=false`.
- Ledger ON: byte-identical raw chronological history plus the default
  production factual transition ledger, `factual_transition_ledger=true`.
- `reasoning_mode=off` in both arms. All model, server, sampling, context, batch,
  frame-format, output-token and action-schema settings come unchanged from the
  successful selected-agent configuration.
- Fresh server order within each stage: seed 0 OFF then ON; seed 1 ON then OFF.
  Every stage/arm/seed cell starts a fresh server.

The ledger is derived exclusively from each model-visible `raw_history` triple:
visible before observation, canonical selected action, and confirmed visible
next observation. The runner must use the production pending-transition/fold
seam. `summary_history`, oracle fields, fixture IDs, target actions, hidden
state, inferred progress, recommendations, filters, public traces, game source
and assets are forbidden inputs. An independent literal calculation from the
same visible frames must match every ordered ledger row, including both changed
and no-op transitions, changed-pixel count, inclusive bounding box, visible
state, level delta, reset flag and before/after animation-frame counts.

The fixed serving configuration is the prepreview Qwen3-8B Q4_K_M artifact:
16,384-token F16 KV context, all 37 layers on CUDA, prompt/physical batch
1024/1024, maximum output 1024, temperature 0.7, top-p 0.8, top-k 20 and min-p 0.
The existing reasoning budget remains configured at 256 but is inactive because
reasoning is OFF. No optimizer or gradient accumulation is used. The pinned
helper and model/server hashes enforce the same qualified configuration.

## Stages, gates and stopping rule

Stage A runs 48 grounding requests: 12 items times two ledger arms times two
model seeds. Every arm/seed cell must achieve at least 11/12 exact centroids.
Any failure ends the diagnostic with stage B not run by design.

If all four grounding cells pass, stage B runs 96 effect requests: 24 items
times two ledger arms times two model seeds. The readout gate requires both of these
conditions on each seed separately:

- ledger ON is at least 21/24; and
- ledger ON minus ledger OFF is at least 3/24.

The maximum is 144 completions. All rows and both seeds are scored. There is no
checkpoint, seed, item or threshold selection, no confidence/population claim,
and no ceiling exception. A malformed/extra/missing/duplicate response, failed
strict schema, non-`stop` finish, nonempty reasoning, token-accounting failure,
identity drift, unexpected GPU process, resource breach or unproved cleanup
fails the run. Failed roots remain preserved and are never reused.

## Matching and integrity controls

- OFF and ON receive identical raw user/assistant history messages, system
  prompt, current base observation and every completion-request field. The only
  allowed prompt difference is the exact factual-ledger suffix on the current
  user message in ON. Stage A requests must be wholly identical across arms.
- Before inference, an independent literal helper visits every coordinate of
  each validated 64x64 before/after frame exactly once, counts cell inequalities,
  and computes the inclusive minimum/maximum bounds of changed coordinates. Its
  result must equal the production ledger. This establishes observed visible
  pixel-difference arithmetic only; it does not establish action causality,
  progress, or useful effects. Counterfactual pairs reverse which history
  position has a nonzero change; rectangle areas and counts need not be equal.
- Capture complete apply-template, tokenize and completion requests/responses,
  strict action schema, finish reason, raw assistant content, token usage,
  context estimates and latency. Reasoning content must be absent or empty for
  all completions.
- Bind a supplied clean, pushed, reviewed full Git revision. Before inference,
  hash the registration, runner, helper, selected configuration, model, server,
  fixture module, and the actual `local_chat.py`, `run_baseline.py`,
  `frame_memory.py`, and `readout_fixtures.py` bytes. Recheck all mutable source,
  registration, model and server identities after the run.
- Require exact reviewed source hashes: `local_chat.py`
  `83ac26b50b530872a4f6d7356e84bb72374a3d5c75599d6b543a4607f41e2721`,
  `run_baseline.py`
  `20f7b0b02aaacfcd5b07f1a59f2e822e4c8bba3dc5eba53dacbbb739112dba7b`,
  `frame_memory.py`
  `04eda0491a80cb7eb24dae1d91b6f94efc7123537e79db52ebb569c9dfbf3a74`,
  and `readout_fixtures.py`
  `db010a29279f9a39b6dd5e30ee62827f479ddc55cc89549abdc981adb104cb36`.
- Require the sealed context qualification manifest
  `9b224eb53cb25e7672d75a11576d41699fa495fd0860b83f688b558f799435e3`.
- Require the completed-negative thinking readout root
  `thinking-readout-20260906T215343510535-0500-pid373838`, manifest
  `6675e3652f79b3104b363491dd545304fb6a02c00c9d103d870cc5c3655343df`,
  plus its independent verification. Its outcome is
  `completed_negative_B`; it is a negative prior diagnostic, not a passed
  thinking or behavior gate.
- Require the sealed CPU/no-network default-OFF replay rooted at
  `off-replay-20260906T221447680131-0500-pid389015`, manifest
  `e477a6889387c9f91a9be2b959cbf78aa09372d00bd90899b344c9bc9d2c1d2f`
  and report
  `8dde381e2d99a98ea1fbd756887243e79f29f6aac79c8b4570becb919b5e8113`.
  It establishes exact request/action replay for all 144 prior rows when the
  default-off feature is present; it supplies no ledger-ON behavioral evidence.
- Require the one expected GPU to be idle before launch, record hardware,
  Python/package and server-version identity, prove 37/37 layer offload, reject
  any compute PID outside the exactly owned fresh servers, and keep sampled
  VRAM below 95% of 8,151 MiB.

## Budgets and lifecycle

Startup has a 120-second hard wall clock, each decision a 60-second hard wall
clock, and the whole run a 600-second hard wall clock. The expected runtime is
about three minutes based on prior nonthinking timings; this estimate is not a
guarantee. This is a controlled ledger intervention with extra prompt-token
compute in ON, not a matched token-compute comparison.

The runner owns every server and telemetry process by exact PID and process
group. On normal exit, exception, timeout, SIGTERM or SIGINT it must TERM/wait,
KILL/wait if needed, and prove every owned `/proc` entry and group absent before
sealing. The sealed run root contains full provenance, fixtures, captures,
predictions, score and cleanup evidence. Its final manifest digest is written
outside the sealed root. This runner does not auto-launch public evaluation;
passing only admits a separately registered public development screen.

## Launch template after repository registration, review and push

Replace both zero placeholders with the reviewed full 40-hex revision and final
64-hex repository-registration SHA-256. A literal command with zeros must fail
closed.

```bash
CHECKOUT=/home/stepan/Coding/Personal/.tofy-build/factual-transition-ledger
REVISION=0000000000000000000000000000000000000000
REGISTRATION=docs/research/2026-09-06-factual-ledger-readout.md
REGISTRATION_SHA256=0000000000000000000000000000000000000000000000000000000000000000
/home/stepan/Coding/Personal/.tofy-build/arcagi-0.9.9-venv/bin/python \
  /home/stepan/Coding/Personal/.tofy-build/factual-ledger-readout-preparation-20260906T2211-CDT/run_ledger_readout.py \
  --checkout "$CHECKOUT" \
  --revision "$REVISION" \
  --registration "$REGISTRATION" \
  --registration-sha256 "$REGISTRATION_SHA256"
```

The registered result may support only this fixed synthetic ledger-readout
claim. It cannot change an ARC Best So Far metric, justify public-level fitting,
or establish that the model can infer a game's goal or plan useful actions.
