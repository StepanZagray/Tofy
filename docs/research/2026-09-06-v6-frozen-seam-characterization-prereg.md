# V6 frozen train/raw seam characterization (preregistered)

Status: **implementation independently reviewed GO; exact run not started**
Date: 2026-09-06 CDT
Evidence class: **characterization-only implementation/integrity diagnostic**
Research claim: **false**
Promotion authority: **none**
Training authority: **none**
Public ARC authorization: **none**

Parent source revision:
`7907731b6fa69043089c55ddb92c573a79b2f29d`

Parent G identity:
`sha256:1d470fc1b5680e33efa07e32c51bf74fa0a73bea6e839828da09fd7922eee265`

Failed frozen preflight root:
`/home/stepan/Coding/Personal/.tofy-build/v6-frozen-diagnostic-preflight-20260906T041517-CDT`

Failed preflight report SHA-256:
`54cc1ddf07aca7e6f8ffd72cb3f216d4d1b079899f995610f807c00fbeb3cb74`

Failed preflight external manifest SHA-256:
`41705d4a9103702600523647475205301fea9383df0011b98a854e7a53b33dbb`

Independent design advice: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-seam-characterization-advice-20260906.md`,
SHA-256 `8c44d7b7eeea253eb22e81f5987a5d93170e98d3f983766682fbb1437fd66d98`.
Fable 5.1 remains unavailable because its account usage limit was reached;
no Opus statement is attributed to Fable.

Initial preregistration NO-GO review: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-seam-characterization-prereg-nogo-20260906.md`,
SHA-256 `ae5ee8407d94ce92aaa8b62fcd3e013954b4d8e6f13e2edc4248a58998e84f15`.
It found that staged `model_frames` differ by construction even though V6
forwards do not consume them. The active-input definition and every
nonblocking ambiguity from that review are corrected below.

Corrected preregistration GO review: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-seam-characterization-prereg-go-20260906.md`,
SHA-256 `8f426d132aff2a49ecdcaadc91eb5ff90a9b6a46493222b9ead319c55d2b2ee5`.
No blocking finding remains.

Implementation review NO-GO: Opus 5 XHigh, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-seam-characterization-implementation-nogo-xhigh-20260906.md`,
SHA-256 `f180848699216654095ab830d758a3b4dd63fa950d81a0df7d72e90748c2c388`.
Its two blockers required direct tests for logit argmax/accounting and the
inert operator-conditioning exception; both were added.

Corrected implementation GO review: Opus 5 XHigh, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-seam-characterization-implementation-go-xhigh-20260906.md`,
SHA-256 `09003d1a115091c884c0ca4a0404ad0125310d229f1ceb2921f491487cdd5359`.
No blocking finding remains. Fable 5.1 remained quota-blocked and is not
credited for either implementation verdict.

## 1. Exact question and bounded claim

The frozen diagnostic failed before its first gradient cell because the
attached training-path argmax and G-compatible raw evaluator argmax disagreed
on G checkpoint step 0, train batch position 0. The failed report did not
preserve the disagreement count or intermediate comparisons.

This characterization asks only:

> On those exact 128 ordered rows, does the disagreement arise from
> same-process nondeterminism, active input tensor construction, CUDA
> recursion/decode-forward shape (one batch of 128 versus four chunks of 32),
> or a residual same-shape
> train/eval seam difference?

The outcome is one machine-readable localization branch plus descriptive
numeric anatomy. It cannot establish model quality, auxiliary-gradient
competition, a valid replacement tolerance, or ARC performance. It cannot
authorize the original full diagnostic or any training run.

## 2. Source-derived premises

No device experiment is allocated to axes already closed by source:

1. `exact_gameplay_logits_detached(latents)` is exactly
   `exact_gameplay_logits_trainable(latents)?.detach()`. Detachment changes
   graph ownership, not values.
2. evaluator `run_recursion` is `run_latent_recursion` followed by observer
   heads. Goal projection feeds only those heads, after the predicted latent
   `y` consumed by the exact decoder.
3. `RecursionOpts::training(true)` and `training(false)` change probe and
   intermediate-step retention; both retain the same final `y`.
4. supplying `canonical=None` recomputes the same canonical representation of
   the same encoded current state used by the training seam.

These are local program properties, not a claim that two differently shaped
CUDA executions are bit-identical.

One known input non-identity is frozen explicitly: V6 training host preparation
uses `provenance.conditioning_operator()`, while generic evaluator batching uses
`provenance.operator`. The step-0 checkpoint must pass
`ensure_operator_projection_zero`, so this field cannot affect `y` in this
characterization. Its tensor equality is reported separately as
`operator_conditioning_equal`; it does not trigger the active-input branch.
Any future nonzero operator projection invalidates this bounded exception.

## 3. Frozen data, checkpoint, hardware, and execution

- parent: the exact sealed registered G report frozen by the parent diagnostic;
- checkpoint: G raw snapshot step 0 only;
- population: `MultibatchPopulation::compose(seed=5, physical_batch=128,
  data_contract_v6=true)` with exact census equality to G;
- rows: `population.train_main[0]`, exactly 128 rows in existing order;
- device: CUDA on the exact G GPU name, UUID, memory, and driver;
- binary: clean, pushed, locked release build with cuDNN and embedded command
  `cargo build --release --locked --features cudnn`;
- run count: one registered characterization, one seed, one checkpoint;
- wall cap: 120 seconds;
- optimizer steps: zero;
- backward calls: zero;
- EMA updates: zero;
- checkpoint writes: zero;
- public data reads: zero.

Use a fresh never-reused root, process guard, explicit lifecycle, recursive
manifest and external sidecar. Recheck the G binding, Cargo.lock, config,
population census, running binary hash, checkpoint hash, and GPU identity after
device work. Any mismatch is failed infrastructure/integrity evidence.

## 4. Exact input comparison

Construct both actual `BatchTensors` objects before any model forward:

- `T`: `prepare_foundation_v2_batch_host(train_main[0])` followed by
  `batch_from_foundation_v2_host`;
- `E`: `batch_from_samples` over the identical ordered 128 cloned
  `TransitionSample` rows.

For every tensor below, copy the flattened tensor to host in its native logical
type, hash the exact row-major bytes with domain-separated SHA-256, and report
shape, dtype, digest and exact equality:

- `frames`, `next_frames`, `model_frames`, `model_next_frames`;
- `actions`, `action_coords`, `goals`, `event_targets`, `event_mask`;
- `operator_conditioning` separately;
- when context is present on both sides: context `current`, `next`, `actions`,
  `coords`, and `valid`.

Require context presence, `k`, and tensor shapes to agree. Context `packed`,
`last_slot`, and `valid_host` are deterministically derived from `valid`, `k`
and shape, so equality of those owning inputs covers the private derivatives.
Define `active_inputs_equal` using only `frames`, `next_frames`, `actions`,
`action_coords`, and the five context tensors. Report `model_frames`,
`model_next_frames`, `goals`, `event_targets`, `event_mask`, and
`operator_conditioning` as separate descriptive flags: staged model frames are
not consumed under V6, goals feed only observer heads, event labels feed only
losses, and operator conditioning is inert only under the required zero
projection.

## 5. Frozen GPU variant order

Load the step-0 checkpoint exactly once and confirm the operator projection is
zero. Run exactly this order, with `sync_cuda_device` immediately after every
variant and before recording its duration:

1. `V5a`: existing `raw_one_step_logits` with frozen chunk size 32;
2. `V5b`: immediate repeat of `V5a`;
3. `V4`: the identical evaluator seam on evaluator-side `E` inputs with chunk
   size 128, forcing a single recursion/decode forward; both V4 and V5 encode
   all 128 rows together before this point;
4. `V1a`: the existing Foundation-v2 training loss forward on `T`, with
   `capture_pred_per_pixel=true`, G's step-0 EP weight and SIGReg seed, no
   rollout forward, and its captured attached prediction logits detached only
   for reporting;
5. `V1b`: immediate repeat of `V1a` with identical inputs and constants;
6. `V5c`: final repeat of `V5a` to expose order/workspace sensitivity.

Unlike the failed rail, this no-gradient characterization does not execute the
dedicated rollout forward and backward before `V1a`. That is intentional under
the zero-backward contract, but changes allocator/workspace history. Therefore
`NOT_REPRODUCED` localizes nothing beyond this lighter execution order and may
select only a newly preregistered allocator-history reproduction—not a verbatim
rerun of the failed rail.

The existing frozen `raw_one_step_logits` function remains unchanged. A new
crate-visible helper may parameterize the evaluator chunk size; the frozen raw
wrapper delegates to it with 32. Implementation may widen only
`batch_from_foundation_v2_host`, `encode_gate_support_population`, and the new
chunk helper to crate visibility. The training loss and model semantics remain
unchanged. No variant calls `backward`.

## 6. Frozen comparisons and descriptive metrics

For each tensor pair below, report:

- exact SHA-256 equality over F32 bit patterns;
- exact argmax disagreement pixels out of `128 * 64 * 64 = 524,288`;
- rows containing any argmax disagreement;
- maximum absolute logit delta;
- absolute logit-delta min, median, p90, p99 and max over all logits;
- on every argmax-disagreement pixel, the reference tensor's exact top-1 minus
  top-2 margin summarized by count, zero count, min, median, p90, p99 and max;
- all defined floats finite.

Pairs:

- repeatability: `V5a/V5b`, `V5a/V5c`, `V1a/V1b`;
- execution shape: `V4/V5a`;
- same-shape train/eval: `V1a/V4`;
- original failed relation: `V1a/V5a`.

Quantiles use the existing deterministic nearest-rank convention
`round(q * (n - 1))` after total ordering. Numeric summaries are descriptive
only; no float tolerance or margin threshold selects a branch.

## 7. Exact machine-checkable decision rule

Compute independent flags first, then the single branch in fixed priority:

1. `SELF_REPEAT_UNSTABLE` if any repeatability pair is not bit-identical.
2. `ACTIVE_INPUT_PREPARATION_DIFFERS` if `active_inputs_equal=false`.
3. Otherwise let `shape_flip = argmax_disagreements(V4,V5a) > 0` and
   `same_shape_flip = argmax_disagreements(V1a,V4) > 0`:
   - both true: `COMPOUND_SHAPE_AND_SAME_SHAPE`;
   - only `shape_flip`: `EXECUTION_SHAPE`;
   - only `same_shape_flip`: `SAME_SHAPE_TRAIN_EVAL`;
   - neither true: `NOT_REPRODUCED`.

Require `argmax_disagreements(V1a,V5a) > 0` unless the selected branch is
`NOT_REPRODUCED` or `SELF_REPEAT_UNSTABLE`. Under repeat instability, one draw
may agree by chance and does not invalidate the instability finding. In every
other branch, failure of this transitive integrity condition makes the report
failed integrity rather than assigning a scientific branch.

`operator_conditioning_equal=false` is emitted as
`ACCEPTED_INERT_OPERATOR_NON_IDENTITY` only while the projection-zero check
passes. It never overrides the branch. Numeric logit differences without an
argmax flip are emitted as independent descriptive flags and do not change the
branch.

No 100x floor, learned tolerance, near-tie cutoff, or row-permutation search is
used. The result selects only the next preregistration:

- repeat instability: design a deterministic-algorithm/repeatability repair;
- active input mismatch: unify the active batch-construction seam;
- execution shape: decide whether the diagnostic must use a same-shape D2 mask
  or whether the G evaluator itself needs a separately justified shape contract;
- same-shape or compound: bisect encoded state, conditioned input and final
  latent in a new diagnostic;
- not reproduced: preregister an allocator/workspace-history characterization
  that reproduces the original rollout-forward/backward prefix; do not rerun
  the original preflight from this lighter result.

This characterization never directly authorizes any of those actions.

## 8. Validation and stop rule

Before the registered CUDA run:

- unit tests cover tensor-bit hashing, argmax comparison, quantiles, branch
  priority, operator exception, and nonfinite rejection;
- the existing raw wrapper test proves its argmax is unchanged;
- all-target check, strict Clippy, formatting and `git diff --check` pass;
- implementation receives independent review;
- the reviewed source is committed, pushed and clean;
- the exact locked cuDNN binary opens the expected device in a bounded smoke.

Run once. Stop after sealing and analysis regardless of branch. Do not rerun the
failed diagnostic, implement the selected repair, or launch training in the
same experiment chain.
