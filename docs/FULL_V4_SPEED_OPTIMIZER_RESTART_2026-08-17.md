# Full V4 speed/optimizer restart preregistration

- Status: registered before accelerator compute; local verification complete,
  accelerator preflight pending
- Date: 2026-08-17
- Task distribution: the fixed Full V4 five-lesson synthetic curriculum
- Initialization/data seed: 2/2
- Budget: 28,672 optimizer updates on one RunPod A40
- Checkpoints: 0, 8,192, 16,384, 20,480, 24,576, and 28,672

## Claim and comparator

The implementation should preserve the Full V4 objective while reducing avoidable
work per optimizer update. The speed comparator is the superseded seed-2 run at
revision `2da103621d8a136d4d5126d3c77b597d5b9e5fbe`, whose early observed wall time
was approximately 9.03 seconds per update at physical batch 2,048 and gradient
accumulation 1. This is a bounded engineering screen, not a causal model-quality
A/B: the corrected hybrid optimizer intentionally changes the parameter trajectory.

"Faster" means a lower median steady-state seconds/update over comparable dynamics
updates after initialization, excluding checkpoint, evaluation, and the selected
profiling update. Throughput is secondary to finite training and provenance gates;
no particular speedup is required to continue the already requested fresh run.

## Intervention

One reviewed revision combines four implementation changes:

1. CUDA-synchronized attribution around the representative update's broad phases;
2. one Full V4 objective module that skips excluded objectives and reuses encoded,
   canonical, and predicted representations;
3. exact omission of the identically zero `t=0` Epps-Pulley quadrature term;
4. corrected hybrid optimization: normalized-buffer Muon Nesterov, three-matmul
   Newton-Schulz iterations, simultaneous decoupled decay, and checkpointed
   per-parameter Adam clocks.

The EP quadrature shortcut is algebraic: at `t=0`, both the empirical characteristic
function and the standard-normal characteristic function equal one, so their squared
difference is exactly zero before multiplication by the quadrature weight. Removing
only that tensor evaluation leaves the mathematical objective unchanged.

## Invariants and preflight

- Full V4 recipe, lesson schedule, initialization, data order, update budget,
  effective batch, checkpoint selection, and evaluation seed remain fixed.
- The largest stable physical batch is rechecked for the exact CUDA binary. Gradient
  accumulation remains the minimum that preserves the effective batch and optimizer
  schedule.
- CPU regression tests must establish cached/uncached Full V4 output parity,
  zero-knot loss/gradient parity, profiler synchronization on errors, Muon update
  semantics, and sparse-gradient Adam clocks.
- The exact clean commit must be fetchable from the recorded remote ref. The release
  CUDA binary is SHA-256 bound to the run, and a bounded five-lesson plus evaluator
  smoke must pass before the full launch.
- The old trainer is stopped and its artifacts classified and sealed before the new
  never-reused run root enters `running` state.

## Metrics, uncertainty, and decision rule

Training integrity requires finite losses, expected lesson coverage, no unexpected
missing/non-finite gradients, observer-stage world-parameter freeze, checkpoint
hashes, and a completed evaluator report. Report seconds/update by lesson and for the
steady dynamics interval; retain the raw sequence so median and dispersion can be
recomputed. With one seed, any throughput or quality difference is a screen only.
There is no multiplicity-based promotion claim and no post-hoc checkpoint selection.

Stop on non-finite training, CUDA OOM, provenance/hash mismatch, failed lesson smoke,
failed evaluator smoke, or artifact-integrity failure. Otherwise run through update
28,672. Model-quality conclusions require the preregistered checkpoint evaluator and
subsequent multi-seed confirmation; this restart alone cannot promote the method.

## Next falsifiable experiment

After completion, compare the predeclared checkpoints and steady-state update times.
If speed does not improve, use the synchronized representative-update packet to pick
one measured dominant phase and test one isolated implementation change against this
revision. If quality changes materially, separate optimizer semantics from objective
execution in a fresh matched-seed A/B before attributing the cause.
