# Frozen gradient reconstruction failure capture

Status: reviewed and frozen for one telemetry-only preflight; outcome pending
Date: 2026-09-06 CDT
Evidence class: infrastructure characterization only
Training / model promotion / public ARC authority: none

## Question and known limit

At source `b8599aed5a50ef9cb868fd8d8db893c6f18f50ae`, the single repaired
mask-contract preflight failed with `gradient component reconstruction failed`
before a completed D1 cell. The two ReconstructionCheck objects were computed
but discarded by the generic guard. Which check/route failed, the residual
magnitude, and cause are unknown. The inference that some fixed reconstruction
criterion failed is supported; no CUDA algorithm or training defect is yet
identified.

Parent root: `/home/stepan/Coding/Personal/.tofy-build/v6-mask-contract-preflight-20260906T105609-CDT`.
Report SHA-256: `3b5f0e5e79a89c29e2b0a39e9a546c95843908cc049cf350e8785adf677b403a`.
External manifest SHA-256: `1cb732c54756a0b1cca3428c05a9dc901c2727be83f7e5d3bb9ff4cdad2d0eec`.
Parent binary: `9a422b4adceb65e58a88b7efd7eb22bb01ff01444926d2a103271429102794de`.

## Algebra before compute

For the same forward and trainable-parameter population, differentiation is
linear over the weighted scalar loss in exact arithmetic. Inspection of
`foundation_v2_training_loss_with_event_weights`, `gradient_cell`, and the
accumulation helpers finds matching prediction, gate, latent, encoder CE,
separation, pull, inverse-action, EP, event, Q, and reliability coefficients.
The in-forward rollout is disabled/zero; the dedicated rollout gradient is
added to both full-gradient constructions. Changed and unchanged prediction
terms have the same real-arithmetic coefficients and denominators.

This local algebra does not prove that differently grouped F32 CUDA backward
computations agree within `1e-5`. In particular the diagnostic combines
coefficient/count into one affine while the production positive term has two
affines. That is a candidate numerical confound, not a demonstrated cause.

## Fixed intervention and measurement

Add only observability to the existing module. Return the already-computed
reference L2 and absolute residual L2 for global, AdamW, and Muon routes from
`reconstruction_check`, alongside the existing relative-or-absolute values
and pass bit. Additive fields use serde defaults. On a failed reconstruction
guard, serialize BOTH full and prediction checks into the error that the
existing runner seals in `report.json`. Preserve the generic error prefix.
No additional forward, backward, norm reduction, or model update is permitted;
no objective coefficient, affine grouping, tolerance, ordering, population,
seed, checkpoint, evaluator, backend, batch, or projection parameter changes.

Keep the original preflight CLI and full `[0,1024,2048]` rescore / `[0,1024]`
D1 grid. This avoids a new driver or contradictory partial-spec identity.
If the first guard fails again, existing fail-closed execution stops there.
If it no longer fails, the original bounded preflight may finish; that is
`NOT_REPRODUCED` characterization and does not admit the full diagnostic.

## Exact launch and budget

One newly reviewed, committed, pushed, clean source revision; no dependency
change; exact `cargo build --release --locked --features cudnn` build command
embedded through TOFY_BUILD_COMMAND. Preserve/hash a separate binary. Before
launch, verify parent report/manifest digests above, G's sealed tree, source
and dependency identity, and exact GPU. Use a never-reused root and the same
registered G report, seed, physical 128, and no accumulation/update.

The parent failed in 18.69 seconds. The expected repeated failure is similarly
bounded, but the unchanged preflight retains its 300-second wall limit plus a
330-second external safety stop for sealing. This is intentionally a replay
of the existing grid with additional failure telemetry, not a new 60-second
partial-grid spec. No automatic retry or downstream launch.

## Frozen interpretation and next decision

- A repeated error with finite serialized checks identifies the failed
  reconstruction(s), optimizer route(s), absolute/relative residuals, and
  reference norms. Report every route without selecting the worst only.
- Missing/nonfinite fields or a different prior error are failed infrastructure
  evidence; no model decision follows.
- No repeated failure is NOT_REPRODUCED, not permission to loosen the guard.
- A large discrepancy does not by itself prove an omitted objective; a small
  discrepancy does not by itself prove harmless roundoff. Follow the captured
  values with a separately designed minimum algebra/backward control.
- The existing `1e-5` relative and `1e-6` near-zero absolute criteria, including
  the reference-norm branch threshold, remain unchanged. These numbers are
  integrity thresholds, not a theorem about F32 backward linearity.

The source change and this complete contract require an independent bounded coding review before compute. Opus 5 XHigh
timed out on both the broad and focused attempts without a verdict; Fable 5.1
High remains quota-blocked. GPT-5.6 Sol XHigh is the explicitly identified
available reviewer for this telemetry-only patch under the project delegation
policy. None of those failures is treated as approval. Host tests must demonstrate that deliberate opposing
gradients retain both reference/residual values and serialize both checks on
failure, while the pass/fail boundary is unchanged. Existing diagnostic tests,
compile, strict Clippy, and diff hygiene must pass.

Seal/recheck evidence, verify process cleanup, preserve all positive and
negative findings in ml/tofy, and stop for analysis.

## Source admission

GPT-5.6 Sol XHigh returned GO for exactly this bounded telemetry capture.
Review: `/home/stepan/Coding/Personal/.tofy-build/reviews/gradient-reconstruction-advice-20260906T110051/sol-final.md`.
Review SHA-256: `f13484da6205ef17b8616769f077092aad0a5651b7debdef5e943a7917cb49e7`. The source-excerpt review did not rerun tests;
primary validation passed `cargo check --all-targets --locked`,
`cargo clippy --all-targets --locked -- -D warnings`, all 30 focused diagnostic
tests, formatting and diff hygiene. No threshold or arithmetic change followed
the review. The earlier Opus timeouts and Fable quota rejection supplied no verdict.
