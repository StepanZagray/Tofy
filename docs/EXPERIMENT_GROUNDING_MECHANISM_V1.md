# Grounding mechanism V1 preregistration

## Question

Does the patch-histogram bundle help because it anchors encoded target state, because it directly supervises predicted next state, or because both branches interact? The active Pressure x Grounding V1 campaign cannot answer this because its grounding factor always averages both branches.

This follow-up is frozen before parent efficacy is inspected. It runs the exact `2 x 2` mechanism factorial at training/initialization seeds 2 and 3:

| Cell | Target-latent coefficient | Predicted-latent coefficient |
|---|---:|---:|
| `G00` | 0 | 0 |
| `GT0` | `0.038966035989056355` | 0 |
| `G0P` | 0 | `0.038966035989056355` |
| `GTP` | `0.038966035989056355` | `0.038966035989056355` |

`GTP` exactly reproduces the parent bundled coefficient: `0.07793207197811271 * 0.5 * (target CE + predicted CE)`. The single-branch cells select one unaveraged loss at half the parent scalar. This keeps each branch's coefficient fixed across the factorial, so its two main effects and interaction are identifiable. All cells use the initialization-pressure-targeted SIGReg coefficient `0.008883956672433376`; “calibrated” does not mean lower pressure throughout training.

## Predictions and decisions

- Target anchoring is supported if `GT0-G00` primarily improves independent target-state decoding and representation breadth.
- Direct prediction supervision is supported if `G0P-G00` primarily improves changed-board prediction, action influence, or H8 observable error.
- Complementarity is supported when `GTP-GT0-G0P+G00` is positive after every endpoint is oriented so larger means better (negate error/tail metrics before taking the contrast).
- Cancellation is established if the two single branches move a gate in opposite directions and explain a null bundled effect.
- If neither main effect nor their interaction improves independent semantics, action influence, and rollout behavior across the two seeds, reject this 16-bin patch-histogram mechanism as sufficient.

Q remains diagnostic. Effective rank and the learned grounding head are not promotion evidence by themselves. Primary evidence is the independent board probe versus literal copy, changed/unchanged error, same-population action sensitivity, H8 median/tails, and clipping/gradient-pressure trajectories.

## Fixed execution

The detached queue waits for the exact parent campaign to finish with final integrity, then runs all eight arms regardless of parent or interim efficacy.

- Seed 2 train 0->250: `G00, GT0, G0P, GTP`; train 250->500 in reverse.
- Seed 3 train 0->250: `GTP, G0P, GT0, G00`; train 250->500 in reverse.
- Every arm retains its update-250 checkpoint and is evaluated at update 500 on the parent's fixed seed-424243 population.

The complementary phase order makes every arm's two training positions sum to five and reverses calendar position across seeds. No result changes continuation, order, coefficients, or evaluation.

## Fairness and integrity

- All cells instantiate the same shared grounding decoder and differ only in which raw loss contributes gradients.
- Every arm uses physical batch `1024`, accumulation `1`, 500 updates, global-mean readout, global additive action conditioning, and the parent's recurrence, optimizer, data, and Q configuration.
- Full ordered training-content SHA-256 and row counts match within each seed and differ across seeds. Parameter count, normalized configuration, optimizer routing, exact resume, and evaluation-population fingerprints must match.
- Pressure is recorded at updates 1, 249, and 499. Global pre-clip norm, clip scale, and clipped-update fraction remain treatment mediators.
- Any parent, hardware, resume, non-finite, content, topology, evaluation, telemetry, or artifact failure stops the whole queue. No arm is substituted or retried silently.

Two seeds provide a robustness check, not a population-level guarantee. The targets are patch colour histograms, not exact cells/objects; a negative result does not reject richer structured state or multi-horizon transition grounding. Recent PSG-JEPA evidence motivates separating state and transition grounding, but transferring that robotics result to ARC-like patch targets is a Tofy hypothesis.
