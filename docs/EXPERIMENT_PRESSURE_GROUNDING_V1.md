# Pressure × grounding V1 preregistration

## Question and estimand

Does reducing SIGReg encoder-gradient pressure, adding an observable-space patch grounding bundle, or their interaction repair the representation and dynamics failures observed in Consumer Readout V1?

This seed-1 screen is descriptive. It does not estimate training-seed uncertainty and cannot establish progress on the official hidden ARC-AGI-3 benchmark. Q remains diagnostic because its labels depend on each checkpoint's learned latent error.

The grounding factor is one frozen bundled intervention: one shared linear `128 → 16` head predicts the normalized colour histogram of each aligned 8×8 gameplay patch from both the target and predicted 8×8 latents. Target and prediction cross-entropies receive equal weight. Changed and unchanged patches receive equal class mass. The synthetic status row is excluded. This bundle tests fixed-space state anchoring plus direct prediction supervision; a positive result alone does not separate those mechanisms. The head exists with identical initialization and optimizer routing in every arm and is never used by evaluation.

## Calibration and arms

Before efficacy training, eight one-update probes use data seeds 1001–1008 and initialization seed 1. Gradient pressure is measured before each probe's first optimizer step. The calibrated coefficient for auxiliary loss `j` is:

`w_j = min(0.275 / median(r_j), 0.50 / max(r_j))`

where `r_j` is the unweighted auxiliary/next-latent encoder-gradient norm ratio. Calibration aborts unless the weighted median lies in `[0.20, 0.35]`, the maximum is at most `0.50`, the eight full-content SHA-256 fingerprints are distinct, and calibrated SIGReg differs from `0.003` by at least 20%.

The factorial cells are `S0G0`, `S0G1`, `ScalG0`, `ScalG1`, `ScurG0`, and `ScurG1`, where SIGReg is zero, calibrated, or current `0.003`, and grounding is off or calibrated. All cells use seed/init seed 1, global-mean readout, current Q definition, global additive action conditioning, identical recurrence/data/optimizer settings, and 500 updates.

The frozen order is:

1. Train 0→250: `S0G1`, `ScalG0`, `ScalG1`, `S0G0`, `ScurG1`, `ScurG0`.
2. Evaluate 250 in reverse order on seed 424242.
3. Train 250→500 in reverse order.
4. Evaluate 500 in original order on the disjoint seed-424243 population.

No update-250 efficacy result changes continuation. Any missing/non-finite artifact, population/config/parameter mismatch, or non-exact resume invalidates the seed block and stops the launcher.

## Fairness contract

- Capacity-probe the worst-memory cell at `1024×1` and require it for all calibration probes and arms. Abort instead of falling back: gradient pressure is measured per physical microbatch, and a lower physical batch would change both the nonlinear SIGReg population and the calibration estimand.
- Match full ordered training tensors through a chained SHA-256 over current/next pixels, goals, labels, actions, coordinates, identities, and row order.
- Match parameter count and name-seeded initialization topology in every cell.
- Record encoder pressure at updates 1, 249, and 499 plus global pre-clip norm, mean clip scale, and clipped-update fraction.
- Checkpoint/evaluate every arm at 250 and 500. Training uses fixed updates and rows, not fixed wall time or FLOPs.
- Use the registered independent ridge probe for evaluation; the training decoder is inaccessible to evaluation.

Primary update-500 evidence is fixed observable-space board prediction versus literal copy, changed-transition improvement, action-shuffle sensitivity, H8 raw and tail-normalized error, and final consumer/spatial effective rank. Factor main effects and interactions are matched seed-1 descriptive contrasts. Only a passing cell justifies full counterbalanced replication at seeds 2 and 3.
