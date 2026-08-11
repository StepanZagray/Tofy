# P2 world-core-v2 causal campaign

Date: 2026-08-11
Research scope: `ml/tofy`
Status: preregistered; promotion locked pending analysis

## Question

Does factual same-state experience plus spatial action conditioning make P2 measurably action-faithful, and do consumer-latent variance/covariance objectives improve representation health without sacrificing transition quality?

The campaign evaluates architectural direction, not the claim that ARC-AGI-3 is solved. No current evidence supports a path to 100% accuracy from a single intervention.

## Fixed contract

- seed: `1`
- lessons: `factual_branches,dynamics,sequential`
- physical population: largest stable A40 batch, tested first at `1024 × 1`
- `grad_accum=1` is mandatory for nonlinear variance/covariance objectives
- recursion: fixed `inner=2`, `outer=8`, warm-started residual dynamics
- final-outer supervision
- legacy SIGReg disabled
- same V2 parameter topology and deterministic initialization conventions in every arm
- one final synthetic evaluation with seed `424242`, 64 episodes, `ptrm-k=1`
- no automatic seed promotion

The factual lesson alternates four-action exact-simulator Branch Groups with four-coordinate, marker-free ACTION6 Branch Groups. Board Effect excludes the deterministic bottom status row, and world-core-v2 replaces that strip with EMPTY before encoding so unchanged-board supervision cannot conflict with action-budget progression.

## Sequential arms

1. `branch-global`: factual branch objectives with legacy global coordinate broadcast. This is the spatial-conditioning control.
2. `branch-spatial`: adds ACTION6 impulse, relative x/y, active mask, and spatial prefix prediction.
3. `spatial-health`: adds spatial Consumer Latent variance (`0.05`) and covariance (`0.005`).
4. `dual-health`: also applies the same weights to the pooled Consumer Latent used by downstream heads.

Every arm uses outcome pull/push, action-ID recovery, ACTION6 coordinate recovery, and Changed Transition/copy margins at weight `0.05`.

## Primary evidence

- action-shuffle error ratio and confidence interval, aggregate and by source;
- changed-transition improvement over copy-forward;
- factual-lesson action recovery and coordinate recovery losses;
- equivalent/distinct Board Effect pair counts;
- spatial and pooled variance/covariance terms and exact population rows;
- named representation-seam variance and effective-rank fractions;
- one-step and open-loop transition errors;
- stability, GPU utilization, peak memory, update throughput, and finite gradients.

## Decision rules

Spatial conditioning is promising only if `branch-spatial` improves changed-transition and ACTION6 action diagnostics over `branch-global` without a material transition-error regression.

A health treatment is promising only if its targeted seam improves variance/effective-rank evidence and does not erase the spatial arm's action gains. A lower regularizer value alone is not success.

Any non-finite loss, missing factual-pair population, mismatched architecture/binary provenance, incomplete artifact hashes, or representation objective evaluated with `grad_accum>1` invalidates the arm.

Seed 2/3 promotion requires analysis after seed 1. Promotion is not automatic.

## Execution

Run `scripts/p2_world_core_v2_campaign.sh` only after:

- checking out the reviewed Git commit on the pod;
- building the exact release binary;
- running the worst-case dual-health batch probe;
- setting its report in `P2_V2_BATCH_PROBE`;
- setting the expected Tofy, candle_graph, and binary hashes.

`scripts/p2_world_core_v2_probe.sh` creates the required worst-case probe and its hashed provenance manifest.

`P2_V2_STEPS_PER_LESSON` controls duration. Total optimizer updates per arm are five times this value because factual and dynamics lessons receive double duration.
