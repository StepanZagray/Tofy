# ADR 0001: Action-faithful world-core-v2

Status: Accepted (2026-08-11)

## Context

P2 learns useful one-step dynamics but remains far from ARC-AGI-3 competence. The latest TC-SIGReg campaign did not improve the primary metrics, and its `512 × 2` schedule also changed the nonlinear population objective relative to `1024 × 1`. Independent action diagnostics show a more fundamental problem: the model can predict while using action information weakly. Two architectural shortcuts make that outcome unsurprising:

- action IDs and ACTION6 coordinates are projected to one spatially uniform bias;
- the prefix path pools the board, predicts one channel delta, and broadcasts it back to every cell.

The synthetic ACTION6 lesson also painted the selected target into the current frame. Sequential experience supplied only one expert action per state, so it did not identify counterfactual action effects. Representation regularization was partly attached to a loss-only projector instead of solely to latents consumed by dynamics and planning.

## Decision

Introduce an intentionally checkpoint-incompatible `world_core_v2` training architecture while retaining legacy checkpoints for evaluation only.

World-core-v2:

- trains on factual same-state Branch Groups, alternating exact simulator movement and marker-free ACTION6 coordinate groups;
- retains global action identity for every action;
- optionally adds an ACTION6 Spatial Action Field containing a localized impulse, relative x/y fields, and an active mask;
- uses a spatial convolutional prefix predictor instead of a pooled broadcast delta;
- learns action identity and ACTION6 coordinates from predicted Consumer Latent displacement;
- pulls equivalent Board Effects together, separates distinct effects, and applies Changed Transition/copy margins;
- defines Board Effect without the deterministic bottom-row status display;
- applies differentiable variance and off-diagonal covariance health losses directly to spatial and pooled Consumer Latents;
- requires `grad_accum = 1` whenever representation-health losses are active;
- persists all switches, weights, architecture identity, batch schedule, loss terms, and population counts.

Every intervention is independently switchable. The first causal campaign uses physical `1024 × 1` and compares action-only, spatial-health, and spatial+pooled-health arms from identical V2 initialization conventions.

## Consequences

Training cannot resume a legacy checkpoint because V2 adds action-decoder, coordinate-decoder, spatial-prefix, and optionally spatial-action-field parameters. This is intentional. Legacy evaluation remains supported.

Factual branches reduce the number of unique current states per transition count, but expose the relational evidence required to identify action effects. Reports therefore distinguish branches, groups, changed branches, equivalent/distinct pairs, ACTION6 branches, and representation population rows.

Covariance work is quadratic in channel width. Deterministic row caps bound the spatial population while preserving live gradients and a stable logical sample.

This architecture improves the validity and action faithfulness of the learning problem. It does not establish, imply, or guarantee 100% ARC-AGI-3 accuracy.
