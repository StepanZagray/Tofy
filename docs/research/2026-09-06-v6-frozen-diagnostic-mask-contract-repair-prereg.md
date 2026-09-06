# V6 frozen diagnostic mask-contract repair (preregistered design)

Status: **design independently reviewed GO; implementation not started**
Date: 2026-09-06 CDT
Evidence class: **diagnostic implementation/integrity repair**
Research claim: **false until a separately admitted registered run completes**
Training authority: **none**
Public ARC authorization: **none**

Parent characterization source: `3d92157073d072fd81ae291074665201adc05dbd`

Parent characterization root:
`/home/stepan/Coding/Personal/.tofy-build/v6-frozen-seam-characterization-20260906T085541-CDT`

Parent report SHA-256:
`787d30d78c4c53dda54becb1a0222ace46ae1cee89f83ae2d21abc49b41b206c`

Parent external manifest SHA-256:
`0fe477652d7d3e34e347896ac0cc427db564084f40ccb584131ce77663eebbb3`

Parent bounded result: on G step-0 train batch 0, active inputs matched,
batch-128 training/evaluator logits were bit-identical, both execution shapes
repeated bitwise, and chunk-32 decode changed 216 argmax pixels across 107/128
rows. This proves execution shape is sufficient to reproduce the mismatch; the
failed run did not preserve its comparand and had a preceding rollout backward.

Independent design review history:

- Initial Opus 5 XHigh NO-GO:
  `/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-mask-contract-prereg-nogo-xhigh-20260906.md`,
  SHA-256 `7d3dcf80454f0cc1d33382786dd7769d71778912896bfb46c91c7e1e716d33e1`.
- Second Opus 5 XHigh NO-GO:
  `/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-mask-contract-prereg-second-nogo-xhigh-20260906.md`,
  SHA-256 `39e78bf09f5a70e79d773d5d94357f93d620946ec18e7a3a356f439e345124fa`.
- Third Opus 5 XHigh NO-GO:
  `/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-mask-contract-prereg-third-nogo-xhigh-20260906.md`,
  SHA-256 `7a93be235a6b86ec5acb602e71a9ccd234405424526167201611ae23cec34cc5`.
- Corrected Opus 5 XHigh GO:
  `/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-mask-contract-prereg-go-xhigh-20260906.md`,
  SHA-256 `bdb98977a85afd665758b79df8acebdbc603d158a39fdefaacdb8545839cdfbd`.

Fable 5.1 remained quota-blocked. No Opus result is attributed to Fable.

## 1. Exact defect and repair claim

The old diagnostic used chunk-32 raw-evaluator argmaxes to define a false-edit
mask for gradients computed by a batch-128 training forward, then required D1
false-edit counts to equal D2's G-bound chunk-32 counts. The characterization
falsified that cross-shape identity premise.

The bounded source claim to test is:

> A diagnostic can bind every D1 false-edit gradient to the exact argmax tensor
> captured from that same batch-128 training forward, make primary D2
> anatomy/persistence use the same batch-128 training seam in independent
> backward-free forwards, and retain the old chunk-32 G scorer only as a
> separately named legacy-parity control.

This is a diagnostic-contract repair, not evidence that any model treatment is
beneficial. No threshold relaxation is allowed. The frozen classifier
thresholds were registered on a chunk-32 false-edit population; applying them
to the repaired batch-128 population changes their population semantics. That
shift is accepted and must be disclosed with every verdict.

## 2. Frozen population semantics

Define two populations that must never be silently compared:

1. `primary_same_forward`: per snapshot step and train/held-out batch position,
   the ordered 128 rows and exact prediction logits captured from one
   Foundation-v2 training-seam forward at physical batch 128. Its argmax defines
   D1 masks, primary D2 scores, anatomy, per-batch anatomy, and persistence.
2. `legacy_g_chunk32`: the existing `raw_one_step_logits` scorer over the frozen
   selected union with decode chunks of 32. It exists only to reproduce every G
   control field exactly and quantify shape sensitivity. It cannot feed D1,
   anatomy, persistence, classification, or a treatment decision.

Every newly produced serialized score, anatomy, mask count, set, and verdict
input must carry one of those names. The sealed failed parent predates these
labels and deserializes additive label fields to empty by design. A report
consumer must not infer equality or causal comparability across populations.

Preserve every existing serialized field for sealed-parent parsing. Add new
`primary_*` fields and explicit population labels; do not rename
`SnapshotRescore.train`, `heldout`, `train_anatomy`, or other legacy fields.

## 3. Same-forward binding

For each D1 cell, execute the existing batch-128 Foundation-v2 training loss
once after the frozen rollout-prefix work. Capture
`losses.diagnostic_predicted_logits`; compute its host argmax exactly once; and
derive only the false-edit mask from that argmax. Changed/unchanged masks remain
frame/content-mask derived, including the existing `losses.changed_weights`
geometry cross-check. The logits used for prediction gradients and the
false-edit mask must share one recorded tensor fingerprint. Do not issue a
second evaluator or training forward to construct the D1 mask.

For primary D2 at every frozen rescore/anatomy step, run the same training-seam
forward independently per ordered 128-row batch with `rollout_enabled=false`,
no backward, and no parameter update. Capture logits, argmaxes, exact scores,
anatomy, and stable row/pixel keys. Record the execution-order caveat: this D2
forward does not reproduce D1's preceding backward allocator history.

At steps/batches shared by D1 and primary D2, report exact logit and argmax
disagreement descriptively. Because D2 is backward-free and D1 follows a
rollout backward, this comparison has no admission power and cannot substitute
for D1's same-tensor fingerprint binding. No tolerance is introduced.

Prepare held-out `PreparedFoundationV2BatchHost` values through the same generic
host path as train batches and require identical 128-row geometry plus the
frozen factual-group range `43..53`. Build union-level primary anatomy by one
`anatomy()` call over concatenated ordered per-batch predictions and F32 host
logits; do not combine per-batch quantiles.

Pin the primary-D2 `FoundationV2ObjectiveConfig` values exactly to D1's
per-step/per-position construction: frozen `ep_weight`, SIGReg seed/projections/
knots, Q threshold, split-CE weighting/budget, recursion config, and
`rollout_enabled=false`. These loss-only values must not drift even though the
captured prediction logits precede their reductions.

## 4. Legacy G control

Run the unchanged chunk-32 scorer and require exact parity through the existing
G bindings (`additive_matches`, `group_matches`, `partial_group_matches`, and
snapshot hashes), including ordered population, per-batch additive scores,
group routing, and ACTION6 controls. Separately extend the existing
preflight-to-registered control binding (`ensure_rescore_control_matches` and
`ensure_preflight_controls_bind`) to cover every new primary field.
Record its false-edit counts under `legacy_g_chunk32` only.

Tensor fingerprints and all hash-valued primary fields are intra-run bindings
only: in-cell construction plus completion-time recorded-value checks. Exclude
them from `ensure_gradient_control_matches` and
`ensure_rescore_control_matches`. Cross-process controls continue to compare
primary integer counts exactly and floats under the frozen tolerance.

Report same-forward versus legacy argmax disagreement pixels/rows, absolute
logit-delta summaries, and margins when both logits exist. Those fields are
shape-sensitivity diagnostics and cannot fail primary scientific admission
unless the legacy G parity itself fails.

## 5. Replaced invariant and classifier inputs

Delete both old cross-shape admission invariants:

1. the in-cell `training_raw_argmax_disagreement_pixels == 0` guard and its
   clean-completion re-assertion;
2. `D1 false_edit_pixels == legacy G chunk-32 D2 false_edit_pixels`.

Rename/redefine `GradientCellReport.training_raw_argmax_disagreement_pixels` as
a `primary_same_forward` versus `legacy_g_chunk32` shape-sensitivity field with
no admission or classifier power. Preserve it descriptively.

Repoint the four train/held-out union/per-batch anatomy false-edit assertions in
`score_frozen_snapshot` to primary same-forward `PerBatchScore` counts. Anatomy
is computed only from `primary_same_forward`; G defines no anatomy control. No
primary anatomy count may ever be compared with a `legacy_g_chunk32` score.
The existing `SnapshotRescore.train_anatomy`, `heldout_anatomy`, and
`*_anatomy_by_batch` fields now hold primary anatomy and carry an additive
population label; do not add parallel nullable anatomy fields.

Replace it with all of:

1. each D1 cell's false-edit count exactly reproduces from its own captured
   training argmax, current frame, next frame, and unchanged mask;
2. its mask/logit tensor fingerprint equals the fingerprint recorded by the
   gradient cell;
3. each shared D1/primary-D2 comparison is recorded but cannot gate admission;
4. primary D2 anatomy false-edit counts reproduce from its own argmax keys;
5. legacy D2 fields reproduce G exactly and are absent from `classify` inputs;
6. all persistence and prediction-quality conditions consumed by `classify`
   use `primary_same_forward` only.

The classifier's numeric thresholds, priority, loss definitions, route
geometry, steps, batch positions, optimizer routing, and next-action mapping
remain frozen. If removing legacy inputs makes any classifier field undefined,
implementation must stop for a new design review rather than invent a proxy.

## 6. Implementation tests before CUDA

Required direct tests:

- same captured logits produce both D1 prediction loss and false-edit mask;
- changing an unrelated second forward cannot change the bound D1 mask;
- both old cross-shape equality admission invariants are absent and their
  descriptive successor cannot reach `classify` or completion admission;
- primary and legacy population labels survive JSON round-trip;
- `classify` rejects or cannot receive legacy-population anatomy/persistence;
- shared D1/primary-D2 argmax mismatch is recorded without failing admission;
- legacy chunk-32 scoring and its existing chunking-control test remain
  unchanged;
- report completion recomputes every same-forward count/fingerprint and fails
  on one-bit or one-pixel corruption;
- no optimizer/EMA/checkpoint mutation occurs.

Reuse the existing `TensorFingerprint`/`tensor_fingerprint` and
`LogitComparisonReport`/`compare_logit_captures` primitives rather than adding
parallel definitions.

“Completion recomputes” means in-cell recomputation from the captured argmax,
then completion-time equality of recorded counts and mask/logit fingerprints.
Do not serialize full per-cell logits or argmax tensors merely to rerun device
computation at completion.

Run `cargo fmt --all`, `cargo check --all-targets`, strict default-feature
Clippy, focused diagnostic tests, unchanged evaluator chunking tests, CLI help,
and `git diff --check`. Resolve every warning/error.

## 7. Schema and sealed-parent compatibility

`FrozenDiagnosticSpec`, `FROZEN_DIAGNOSTIC_SCHEMA`, and
`diagnostic_identity` are byte-frozen so the sealed failed parent remains
parseable and identity-valid. Every new field on `FrozenDiagnosticReport`,
`SnapshotRescore`, `PerBatchScore`, `FalseEditAnatomy`, or
`GradientCellReport` is additive and carries `#[serde(default)]`. No dependency
change is allowed because G also pins `Cargo.lock`.

Retain the literal failed-parent error string
`training and raw prediction seams disagree` for
`bind_failed_frozen_preflight`; removing its live admission guard does not
authorize deleting the historical binding constant.

## 8. Independent review and provenance gate

Before any CUDA execution, obtain an independent Opus 5 XHigh implementation
GO (Fable 5.1 if it becomes available, clearly attributed). The reviewer must
judge population separation, exact same-forward binding, classifier dataflow,
legacy G parity, lifecycle/sealing, and tests. Record the review artifact path
and SHA-256 in this preregistration before committing implementation.

Commit and push a clean source revision. Build only with
`TOFY_BUILD_COMMAND='cargo build --release --locked --features cudnn' cargo build --release --locked --features cudnn`.
Record source/dependency revisions, binary/Cargo/config/checkpoint/population
hashes, exact GPU identity, command, and never-reused root. Fail closed on drift.

## 9. Execution ladder and stop rule

1. Run one unregistered CUDA preflight with the unchanged byte-frozen spec: D1
   at steps 0 and 1024 on train batch position 0; rescore steps 0, 1024, and
   2048; anatomy at 2048; train/held-out batch position 0 only. D1 retains its
   existing rollout-prefix ordering; primary D2 remains backward-free.
2. Require legacy G parity, exact D1 same-tensor mask/logit binding, finite
   route geometry, complete population labels, no mutation, valid runtime
   estimate, and a verified recursive seal. Record D1/primary-D2 disagreement
   descriptively without gating.
   A held-out training-seam objective rejection is a legitimate fail-closed
   preflight outcome; it is not authority to fall back to legacy scoring.
3. Stop for analysis. Do not automatically launch the registered diagnostic.
4. A registered diagnostic requires a fresh review of the sealed preflight and
   an explicit machine-checkable admission decision recorded in a new commit.

The preflight must measure both population paths and publish a corrected
25-cell D1 plus 56-rescore registered runtime forecast against the frozen
900-second admission cap; no cap is relaxed. Compute descriptive full-logit
comparisons outside the timed D1 cell and report their cost separately. The
additional primary forwards may legitimately make the forecast inadmissible;
that is a preflight result, not authority to relax the cap. Preflight maximum
wall time is 300 seconds. No automatic retry. Any failure root
is sealed as infrastructure/integrity evidence and cannot satisfy admission.

## 10. Decision boundary

A passing preflight establishes only that the repaired measurement contract is
internally coherent on the frozen step-0 controls. It does not restore the old
failed run, prove model quality, select prediction-only training, authorize a
long run, or permit public ARC evaluation. The registered diagnostic remains
the next evidence step only after separate analysis and review.
