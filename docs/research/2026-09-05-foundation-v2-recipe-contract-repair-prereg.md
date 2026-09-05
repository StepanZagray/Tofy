# Foundation-v2 recipe-contract repair preregistration (2026-09-05)

## Why this supersedes the next E2G launch

The sealed 2x2 E2 run requested physical batch 128 and gradient accumulation 2,
but the Foundation-v2 loop constructed one 128-row batch, called backward once,
and stepped once. Its 524,288 recorded population rows equal `4096 * 128`, not
the registered 1,048,576. The same loop's mixed batch could contain at most 15
sequential rows under the V6 schedule, while the rollout objective required 16
distinct adjacent fragments, so rollout was unreachable and its reported mean
was exactly zero. E2 is therefore exploratory checkpoint evidence, not a valid
test of its registered effective-batch-256 claim. E2G is paused until the
training recipe has a truthful executable contract.

The review's other two interpretations are not adopted as facts. A common
gradient clip scale does not equal an effective learning-rate multiplier under
AdamW moment normalization or Muon's Frobenius normalization, although E2's
variable scale and 100% clipped-update rate are serious conditioning signals.
Foundation-v2's inline separation, pull, and inverse-action losses were active
despite legacy `branch_learning.enabled=false`; their nonzero scalar losses do
not establish useful gradient pressure.

## Exact bounded claims

This change is an implementation repair and premise diagnostic, not an ARC or
model-quality experiment.

1. A Foundation-v2 launch cannot claim gradient accumulation unless the loop
   executes it. Until that implementation exists, every Foundation-v2 config
   with `grad_accum != 1` must fail validation before data generation or device
   work.
2. Every update with rollout enabled receives a separate deterministic 32-row
   batch containing exactly 16 distinct sequential units, each truncated to
   its first adjacent pair.
   The unweighted rollout loss must be finite and positive and its weighted
   gradient must reach the recurrent core before the first optimizer step.
3. At explicitly requested `pressure_updates`, read-only attribution reports
   the global, AdamW-routed, and Muon-routed gradient L2 norms for prediction,
   grounding, gate, latent, separation, pull, inverse action, EP, rollout, the
   detached-observer bundle, and the actual combined pre-clip gradient. Each
   component also reports its cosine to prediction where defined. Attribution
   backprops never reach optimizer state.
4. The main 128-row loss gradient and weighted dedicated-rollout gradient are
   summed before the existing one-time global L2 clip and optimizer step. The
   report and population hashes include both consumed batches.

## Implementation comparator and invariants

- Baseline: Tofy `e55313ee`, Foundation objective revision 4. It silently
  ignores `grad_accum > 1` and gives V6 physical-128 rollout zero support.
- Treatment: objective revision 5 with Foundation accumulation rejection,
  dedicated rollout batches, launch activation evidence, and read-only
  component attribution.
- Fixed: model architecture (V6 2x2), main mixed-stream schedule, loss weights,
  optimizer, WSD schedule, global max norm 1.0, EMA, gate evaluator, and public
  ARC data boundary.
- Explicitly unchanged: `CurrentDouble` split CE, learning rate, clip threshold,
  and inline action-objective weights. No one of these changes until attribution
  identifies a falsifiable intervention.
- Resume must fail across objective revisions and must reject a completed
  pressure target whose durable sample is missing.

## Cheapest decisive validation

Run tests first. Then run a one-update, fresh-initialization CUDA implementation
smoke on the exact reviewed binary with physical batch 128, accumulation 1,
V6 2x2, `CurrentDouble`, and `pressure_updates=[1]`. It uses synthetic training
data only and writes an unpromoted final checkpoint.

The smoke passes only if all of the following are machine-checkable:

- main rows are 128 and the separately reported rollout population contains
  exactly 32 rows and 16 eligible fragments;
- rollout loss is finite and strictly positive;
- weighted rollout recurrent-core gradient L2 is finite and strictly positive;
- combined pre-clip norm and clip scale are finite;
- every scheduled pressure component has finite route norms, with zero/undefined
  represented explicitly rather than omitted;
- prediction, the aggregate inline-action bundle, and rollout have nonzero
  global gradients; individual action components remain explicit diagnostics;
  and
- the report's total training-population rows equal main plus dedicated rows,
  not a fictitious `physical_batch * grad_accum` value.

Failure stops this branch before any longer run. PASS authorizes only analysis
of the pressure packet and a separately preregistered single-factor recipe
screen. It does not authorize E2G, a public-game policy change, a full
Foundation campaign, or model promotion.

## Decision rule after the smoke

- If rollout is still inert or does not reach `block.*`, repair only that path.
- If prediction dominates the combined gradient, compare legacy
  `CurrentDouble` against a normalized class-balanced CE on the same frozen
  initialization and batch; do not infer a fix from loss magnitudes alone.
- If an action component has zero or negligible weighted pressure, localize its
  population/mask/Jacobian before changing its weight.
- If route-wise pressure is healthy, retain max norm 1.0; universal clipping by
  itself is not evidence to raise the threshold or lower the configured LR.
- No branch enables legacy branch learning for V6.

## Resource and evidence contract

The one-update CUDA smoke has a 15-minute wall-time cap. Before it, use a clean,
reviewed, pushed commit; record exact Tofy and sibling revisions, locked build
features, binary SHA-256, device/driver identity, config, source and batch
hashes, physical batch/accumulation, and pressure update. Use a never-reused
root and seal it recursively. No public ARC data is read. Any subsequent
comparison needs a fresh preregistration with seeds, checkpoint selection,
uncertainty, multiplicity, and promotion/rejection rules.

## Post-run qualification (recorded 2026-09-05 after the smoke)

The smoke passed its active implementation gates, but the component list did
not contain the `grounding` item named in claim 3 because both Foundation
grounding weights are fixed to zero. The config and evidence manifest make that
disabled state explicit, so no active loss was hidden, but the prose and packet
were not exact. This mismatch is declared rather than edited away. The next
pressure contract must report a zero/null grounding row explicitly.

The one-step WSD schedule also selected final LR `0.0001`, not the stable
production LR `0.001`. That does not affect the pre-optimizer attribution used
by this smoke, but it prevents treating the optimizer update as representative.
