# P2 legacy root-output archive

This document preserves the durable evidence from the legacy root-level
`p2-output-*` directories before their removal on 2026-08-08. New experiments belong
under `runs/p2/`; this is a historical development record, not a set of research
claims.

## Cleanup scope

- Removed 21 ignored `p2-output-*` directories and the empty ignored
  `p2-arc-recordings/` directory.
- Reclaimed 18,415,917,104 bytes (about 17.15 GiB).
- Checkpoint bundles occupied about 99.7% of those bytes. Individual exported models
  were only about 0.7--2.6 MB, so frequent full optimizer checkpoints, not final model
  weights or reports, caused almost all of the storage cost.
- No removed path was tracked by Git. The relevant commands, decisions, and selected
  metrics remain in this file, `RESULTS_P2.md`, the `P2_V*.md` notes, and
  `docs/research/`.

The largest single trees were v12 (7.25 GB across seven experiment chains), v14
(2.58 GB), and v17-stable (0.78 GB). `p2-output-readiness-v2/` contained only a
launcher log; the durable readiness-v2 result is already recorded in
`RESULTS_P2.md` and its canonical run path is `runs/p2/readiness-v2`.

## Comparable historical metrics

The table uses held-out synthetic dynamics from the report stored with each
checkpoint. Compare horizon growth within a row. Absolute latent MSE is not safely
comparable across architecture families because their representations and report
schemas changed.

| Run | One-step MSE | Open @4 | Open @8 | Closed @8 | Durable conclusion |
|---|---:|---:|---:|---:|---|
| v8 | 0.0565 | 0.4316 | 1.0188 | n/a | Stable predecessor, but Q was saturated. |
| v9 | 0.0694 | 5.38e3 | 6.12e11 | n/a | Exploration plus the tighter Q threshold coincided with catastrophic compounding. |
| v10 | 0.0536 | 2.83e4 | 3.45e12 | n/a | The v10 fixes recovered one-step error, not open-loop stability. |
| v11-control | 0.0258 | **0.0843** | **0.1725** | n/a | Best completed open-loop baseline; removing exploration and restoring `q_mse_threshold=0.25` recovered rollout. |
| v13 | **0.0240** | 0.1185 | 0.3204 | n/a | Slightly better one-step error than v11, but worse rollout and saturated Q; it failed its promotion gate. |
| v14 | 0.0679 | 2.19e3 | 4.03e11 | n/a | The first spatial/ARC-oriented family was violently unstable when self-fed. |
| v15 final | 0.8035 | 1.16e3 | 1.50e11 | 2.7420 | Closed-loop feedback bounded error while open-loop latent rollout exploded. |
| v15 step 28,700 | 0.0530 | 3.49e3 | 3.74e11 | 0.0465 | A good closed-loop checkpoint can still be unusable for open-loop planning; final-checkpoint selection also materially changed quality. |
| v17 step 16,300 | 0.2533 | 0.4268 | 1.1874 | 0.0696 | RMS normalization/residual updates removed the v14/v15 `1e11`-scale explosion, but compounding remained. This checkpoint was partial and had not reached rollout/Q lessons. |

The v12 seven-run chain uniformly failed its rollout gate. One-step MSE ranged from
0.0947 to 2.27 and rollout-8 MSE from 6.68e4 to 1.36e24. The least bad variant was
`exp-q005` (one-step 0.1177, open @4 8.15, open @8 6.68e4), still far behind
v11-control. The bundled dual-block/dual-pool/stop-gradient architecture was therefore
reverted. `P2_V12.md` and `docs/research/fable5/02-delta-vs-absolute-latents.md`
explain why this chain should not be interpreted as a clean test of residual-state
dynamics.

Early v2--v7 reports are retained only as development history. Their evaluator
schemas and architecture contracts changed, and two reports for the same v2 export
even disagree by orders of magnitude after later re-evaluation. v5 and v6 clearly
diverged numerically; v3, v4, and v7 also had severe horizon compounding. They should
not be mixed into current leaderboards.

## Durable engineering conclusions

1. **One-step accuracy is not a rollout gate.** Several checkpoints have one-step MSE
   near 0.05 while open-loop horizons explode by many orders of magnitude. Promotion
   must require frozen open-loop metrics at every intended planning horizon.
2. **v11-control remains the rollout reference.** v13 owns the marginally better
   one-step number, but no later completed legacy run beat v11 at horizons 4 and 8.
3. **Closed-loop metrics diagnose self-feeding failure.** The enormous v15
   open/closed gap localizes much of the failure to feeding predicted latent states
   back into the transition operator, rather than to a single observed-state step.
4. **Q/PTRM ranking was not demonstrated.** v11 and v13 saturated their fixed-threshold
   Q labels. In v11 and v15, `q_oracle_rank_accuracy` was approximately chance
   (`1/k`); v15 still had a 45.7% confident-error rate. Do not treat noisy best-of-K
   results as evidence until Q receives direct differentiable ranking supervision and
   calibration is evaluated on a frozen split.
5. **Larger or deeper models did not cure contraction failure.** The v12 hidden-256
   and deeper-recursion variants were worse, and the spatial v14/v15 models amplified
   the problem. Control latent scale and transition gain before spending on capacity.
6. **Action conditioning must be an explicit diagnostic.** The completed
   readiness-v2 action-shuffle ratios were all approximately 1.0, including the broad
   `random_one_step` curriculum. Coverage alone was not the explanation; target-encoder
   collapse/action marginalization must be fixed before planner claims.
7. **Capacity probes must use worst-case recursion depth.** The legacy v17 launcher
   OOMed at physical batches 1024, 768, and 512 on the 8 GiB RTX 5060, while shallow
   randomized probes could pass. Later readiness work established physical 128 with
   accumulation 4 as the stable local schedule for effective batch 512. Hardware and
   architecture differ, so record the measured batch/accumulation pair per run.
8. **NaN diagnostics need named constituent checks.** The old root readiness run
   reached only step 2,691 and repeatedly failed with non-finite total loss; a resume
   at physical 256 also OOMed. The subsequent readiness-v3 investigation traced a
   stronger failure chain to runaway prefix recurrence, non-finite gradient clipping,
   and a zero-gradient hard SIGReg clamp; the repairs and exact tests are preserved in
   `RESULTS_P2.md`.

## Artifact policy for future runs

- Use `runs/p2/<run-name>` and keep root clean.
- Before pruning a run, preserve: Git commit, hardware, physical batch and gradient
  accumulation, exact train/eval commands, `config.json`, final/selected-step report,
  report schema, and the reason the checkpoint was selected.
- Retain periodic optimizer checkpoints only while resume or checkpoint selection is
  active. After a run is summarized, keep the selected checkpoint only when it is
  needed for a planned comparison; otherwise retain the report and delete binaries.
- Never promote `research_claim=false`, `official_rhae=null` reports to official
  ARC-AGI-3 results. All removed legacy reports had those limitations, and the ARC
  recording directory was empty.

