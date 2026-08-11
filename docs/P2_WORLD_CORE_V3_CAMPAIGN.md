# P2 world-core-v3 campaign

## Why this run

The seed-1 V2 campaign rejected every treatment. Replacing global ACTION6
conditioning with a spatial field reduced action sensitivity, absolute latent
health increased scale without enough independent dimensions, and the existing
evaluation could not tell whether the factual auxiliary tasks generalized.

V3 makes three targeted changes while retaining the world model as the core:

1. ACTION6 spatial conditioning is a bounded residual on the proven global
   coordinate path. Its projection is zero-initialized, so treatment and control
   begin with exactly the same global behavior.
2. Optional health acts on same-state, group-centered, action-balanced factual
   displacements. A differentiable scalar RMS above a fixed floor removes scale
   mismatch without adding a radial gradient; covariance is measured as
   correlation so feature rescaling cannot fake decorrelation.
3. Evaluation adds a frozen held-out factual population and an eval-only board
   probe. It reports relation retrieval, action/coordinate recovery,
   changed-versus-unchanged displacement, and board-patch prediction relative to
   literal copy-forward.

These are diagnostics, not evidence of ARC-AGI-3 transfer or a path to guaranteed
100% accuracy.

## Frozen design

Every arm starts fresh, uses physical batch 1024 with accumulation 1, shuffles
episode starts, and runs exactly 500 optimizer updates (`100` per generated lesson
stage). The final evaluator uses 64 generic synthetic episodes and 256 held-out
same-state factual groups.

| Seed | Arm | Spatial residual | Displacement variance | Correlation penalty | Purpose |
|---:|---|:---:|---:|---:|---|
| 1 | global-control | no | 0 | 0 | fresh causal baseline |
| 1 | spatial-residual | yes | 0 | 0 | isolate residual conditioning |
| 1 | displacement-variance | yes | 0.02 | 0 | isolate scale-normalized spread |
| 1 | displacement-decorrelated | yes | 0.02 | 0.002 | test independent displacement dimensions |
| 2, 3 | global-control | no | 0 | 0 | matched baseline replication |
| 2, 3 | spatial-residual | yes | 0 | 0 | matched treatment replication |

The seed-2/3 pairs are predeclared replications of the architectural treatment,
not automatic promotion based on seed 1. The health variants remain exploratory
until their gradient pressure and factual metrics are inspected.

## Runtime and integrity controls

- `scripts/p2_world_core_v3_probe.sh` verifies the exact Tofy revision,
  `candle_graph` revision, release-binary hash, clean worktrees, A40 hardware,
  batch 1024, and displacement-health gradient pressure before authorizing the run.
- `scripts/p2_world_core_v3_campaign.sh` verifies that probe and enforces one
  absolute 36,000-second deadline. A running stage receives `TERM` five minutes
  before the deadline and `KILL` four minutes later; a new arm is not admitted
   when less than 90 minutes remains.
- Each arm retains configuration, training report, step-250 and step-500
  checkpoints, evaluation report, episode rows, GPU telemetry, and SHA-256 hashes.
- Scientific promotion remains locked until all artifacts are copied locally and
  analyzed against the same factual-population fingerprint and per-action/source
  strata.

## Analysis priorities

The primary causal estimand is the within-seed difference between
`spatial-residual` and `global-control`. Do not average away seed disagreement.
Reject a treatment with invalid reports, non-finite loss, untrusted board probe,
or factual regressions hidden by aggregate dynamics metrics. For health arms,
first check that weighted health-to-next gradient pressure is finite and no more
than `0.5`; the desired diagnostic interval is `0.05–0.25`.

Board-probe trust only means that the held-out target latent can be decoded under
the fixed probe contract. It does not itself prove good transition prediction.
Once trusted, compare predicted-next histogram MSE against literal copy-forward,
with changed-patch recall and unchanged-patch MSE reported separately.
