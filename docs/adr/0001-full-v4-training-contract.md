# ADR 0001: Full V4 training contract

- Status: accepted for local implementation and accelerator preflight
- Date: 2026-08-16
- Successor schema: `world_core_v4_full_training`
- Training report: `p2.train_report.v11`
- Evaluation report: `p2.eval_report.v15`

## Claim and evidence boundary

Full V4 is a coherent baseline for testing whether Tofy's synthetic curriculum
can learn action-conditioned causal dynamics without the representation and
stage-maturity confounds in the earlier experiments. It is not a proof of global
optimality, an ARC-AGI-3 solution, or a guarantee of any competition score.

All model-level conclusions from checkpoints trained under the earlier setup are
non-transferable to V4. Those artifacts remain useful only for provenance,
evaluator debugging, and implementation-history audits.

## Decision

`p2-train --recipe full-v4` resolves one persisted recipe. Runtime/provenance
settings (seed, physical batch, steps, device, output, and checkpoint cadence)
remain caller controlled; model and loss switches do not.

The recipe fixes:

- hidden width 128, action embedding width 32, two inner and two outer steps;
- hybrid optimizer settings (`lr=1e-3`, weight decay `0.01`, Muon momentum
  `0.95`, repository RMS scale), with DeepSeek-V4-equivalent normalized-buffer
  Nesterov, simultaneous decoupled Muon decay, and checkpointed per-parameter
  Adam clocks; F32 convolution; and shuffled episodes;
- convolutional 64x64 palette encoder with the status row replaced before
  encoding;
- `SpatialQuery` as one canonical `B x C` state;
- that canonical state is RMS-normalized, consumed by the transition, used by
  observer heads, and is the sole Epps-Pulley population;
- spatial ACTION6 conditioning and action-conditioned input on every outer
  recurrence update;
- final-step spatial Huber plus canonical Huber prediction loss;
- exact per-pixel palette grounding of encoded current and target states only,
  excluding the bottom status row;
- marginal Epps-Pulley with 1024 seeded projections, the original 17-knot
  quadrature (the analytically zero `t=0` term is omitted from tensor work),
  weight 0.1, no row subsampling, and no smooth cap;
- sequential open-loop Huber training with the implemented `2 -> 4 -> 8 -> max`
  horizon progression;
- `q_calibration` and `falsification` as observer-only stages: world objectives
  are zero, world representations are detached, and only observer/goal heads
  update;
- Q/reliability labels derived from the frozen exact decoder: at least 99%
  gameplay-pixel accuracy overall and, for changed transitions, at least 90%
  accuracy on changed pixels.

The recipe excludes temporal centering, QQ/TC-QQ, factual-branch losses, PTRM
ranking loss, prefix losses, predicted-latent grounding, change decoders,
gradient-ratio calibration, stochastic depth, and latent noise. These may be
tested only after a mature V4 checkpoint fails a preregistered prerequisite.

## Paper-fidelity matrix

| Seam | Full V4 | Source relationship |
|---|---|---|
| Joint encoder gradients | Current and target states share one trainable encoder/readout; neither is EMA or stop-gradient during world stages | Compatible with LeWorldModel v3 |
| Prediction loss | Spatial and canonical Huber | Tofy adaptation; LeWorldModel uses next-embedding MSE |
| EP estimator | Marginal Epps-Pulley, 1024 seeded projections, original 17-knot weights with the identically-zero `t=0` term omitted from execution, weight 0.1, no row cap/cap function | Algebraically component-faithful to the audited LeWorldModel implementation; F32 reduction order remains an implementation detail |
| Projector/readout | Shared in-stream `SpatialQuery`, per-sample RMS, consumed by transition and heads | Tofy adaptation; LeWorldModel uses separate loss-side encoder/predictor projectors over a global CLS token |
| Temporal population | Marginal current/target rows | Deliberate baseline; TC-LeWM's centered residuals require valid ordered same-trajectory windows not present in `random_one_step` |
| Exact semantics | Current and target gameplay pixels, status row masked | Tofy/ARC adaptation motivated by PSG-JEPA-style static-state grounding; no source theorem transfers |
| Action sensitivity | Spatial action field in every outer update | Tofy adaptation; ActSWM separation losses are explicitly deferred |
| Observer stages | Detached canonical state, exact pixel-derived labels | Tofy training-contract correction |
| Environment | Synthetic Tofy curriculum, not the paper environments and not ARC public recordings | Material mismatch; empirical claims do not transfer |

Primary sources audited:

- [LeWorldModel v3](https://arxiv.org/html/2603.19312v3) and its
  [official implementation](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)
- [TC-LeWM v2](https://arxiv.org/html/2607.26924v2)
- [QQWorld v1](https://arxiv.org/html/2607.28415v1)
- [PSG-JEPA](https://arxiv.org/abs/2608.06799)
- [ActSWM](https://arxiv.org/html/2607.26712v1)
- [ARC-AGI-3 environment paper](https://arxiv.org/abs/2603.24621)

## Mathematical checks

The old patch-histogram target is non-injective: permuting pixels inside a patch
preserves its histogram, so zero histogram loss cannot imply exact state
reconstruction. Per-position palette targets remove that particular equivalence
class. This proves only that the new target distinguishes within-patch
permutations; it does not prove that the learned encoder or transition is
globally identifiable.

Temporal centering is undefined as a production dynamics objective when a batch
contains no eligible ordered same-trajectory window. The current one-step batch
generator can create exactly that condition. Marginal EP is therefore the only
valid production baseline until the data contract supplies such windows.

Multiplying a connected world loss by zero is insufficient for an optimizer
freeze because it may still create zero gradients and trigger weight decay.
Observer stages instead begin from a new disconnected zero scalar and attach
only detached canonical states to observer heads. A regression test asserts the
absence of encoder, recurrence, readout, and decoder gradients.

## Training-readiness gates

Before a full accelerator run:

1. build the exact reviewed revision with `--locked --features cudnn`;
2. measure the largest stable physical batch for the exact Full V4 binary,
   keeping `grad_accum=1`, and preserve the measurement evidence;
3. verify that the reviewed commit is fetchable from its named remote ref;
4. run a bounded device smoke through all five lessons and a one-episode
   evaluator smoke with the exact binary;
5. confirm finite losses, nonzero world gradients in world stages, observer-only
   gradients in the last two stages, exact-decoder status masking, and report
   schema/provenance;
6. only then start fresh full training. No pre-V4 checkpoint or optimizer state
   predating trainer-state schema v8 may be resumed into V4.

`scripts/p2_arc3_train_eval.sh` enforces the clean revision and binary hash,
requires preserved evidence for the explicitly measured physical batch, verifies
that the reviewed commit is fetchable, runs bounded five-lesson training and
evaluation smokes, persists hardware/build/preflight hashes, and refuses reused
roots before starting a full campaign. After evaluation it writes deterministic
file manifests and their digests outside both run roots; the preflight is sealed
before full training begins. These are point-in-time integrity records, not
immutable storage. A zero process exit is insufficient:
the full report must explicitly say `completed` and contain all five lessons
before evaluation or sealing. A completed preflight remains classified as an
implementation smoke if the later campaign fails.

## Diagnostic baseline amendment

The base-checkpoint campaign is evaluated at updates 0, 8192, 16384, 20480,
24576, and 28672. Each transition now preserves semantic content dimensions,
source kind, and trajectory identity, so exact-decoder metrics cannot count
7×7 padding or the status row as task content. Schema v15 adds unseen-seed IID
7×7 populations alongside the existing 8×8 held-out composition, exact
semantic H1/H4/H8 metrics and controls, visible-input collision ceilings, and
same-state factual-outcome retrieval. Action recovery remains a secondary
diagnostic because it does not establish correct next-state prediction.

Checkpoint bundles and evaluation identities are SHA-256 bound to their model,
optimizer, trainer/config state, evaluator binary and command, and generated
populations. Parameter-group hashes make the observer-stage freeze an artifact
invariant rather than an inference from loss configuration.
