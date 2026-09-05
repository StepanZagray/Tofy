# V6 CurrentDouble frozen EMA-lag audit replication

Date: 2026-09-05

Evidence class: retrospective deterministic audit / selection-only

Promotion authority: none

Training authorized: no

## Registration boundary

Fable 5.1 inspected these frozen artifacts before this replication contract
was written and reported candidate metrics and thresholds. This is therefore
not a blind preregistration and must not be described as fresh confirmation.
The purpose of the independently implemented replication is narrower: detect
calculation errors in that advice, preserve machine-readable evidence, and
decide whether an EMA implementation fault still warrants investigation.

No public ARC data, new model evaluation, optimizer update, or GPU work is
authorized. The result cannot promote a checkpoint or treatment. The raw
trajectory's failure already routes to a positive-control overfit regardless
of this audit's outcome; an indicated EMA fault would create a parallel repair
task, not make the failed raw model useful.

## Bounded claim and mathematical limit

The audit tests whether the available source and frozen checkpoint files show
evidence of an EMA initialization, binding, aliasing, reset, or wrong-decay
fault. Sparse raw checkpoints cannot certify the 2,048 historical recurrence
steps. A passing result means only `consistent_no_ema_fault_detected`.

For decay `d`, initial raw weights `w_0`, and the coded recurrence

```text
e_t = d e_(t-1) + (1-d) w_t,
```

the generally valid bound is

```text
||e_t-w_0|| <= (1-d^t) max_(1<=k<=t) ||w_k-w_0||.
```

The previously proposed replacement of the maximum with `||w_t-w_0||` is
invalid without monotone displacement. Counterexample for `d=0.999`,
`w_0=0`, `w_1=10`, `w_2=1`: the correct `e_2=0.01099`, so
`|e_2|/|w_2|=0.01099 > 1-d^2=0.001999`. That norm-ratio gate and the weak
step-2 projection-coefficient gate are excluded.

## Frozen inputs

- Checkpoint run:
  `/home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT`.
- Checkpoint source revision:
  `d0468a808d0d3cd2754dc1b31e4a85ab636fcc36`.
- Source-run recursive seal-manifest digest:
  `e686a783b13eadc9f099d2ba5bdce68ecc10923e686db6fc86cc9bb4c02abfc5`.
- Trajectory root:
  `/home/stepan/Coding/Personal/.tofy-build/current-double-checkpoint-trajectory-20260905T141030-CDT`.
- Trajectory preregistration revision:
  `aaa51d6eda03ba4316038b2ad90f44a9899bc350`.
- Trajectory recursive seal-manifest digest:
  `5a476f37c88624d37fb43d500f3e8c71dfd1fcedfe7e9315cf339b18a9627930`.
- Fable advice:
  `/home/stepan/Coding/Personal/.tofy-build/reviews/fable-ema-lag-audit-advice-20260905.md`,
  SHA-256
  `019559da1ad9cfd73b5319b11a11081a216207bbe8b6b1a0dab500e1e8274697`.
- EMA source SHA-256: `src/p2/optimizer.rs` =
  `18dc5c445cf6576f8d82ab99c6654affb4fdb2d1db3816820ac2eba79408e48f`.
- Training/checkpoint source SHA-256: `src/p2/train.rs` =
  `4a89175ba74ac0f78994895012f8489d01f21035de726c6472aecf2c6fa7905f`.
- Fixed steps: `0, 2, 256, 512, 768, 1024, 1280, 1536, 1792, 2048`.
- Fixed decay under test: `0.999`; wrong-decay controls:
  `0.9995, 0.998, 0.99`.

## Independent implementation

Write one standalone Python/NumPy script into a fresh, never-reused audit root.
It must parse safetensors directly, without installing packages or modifying
the repository. It writes one JSON report. Preserve the script, stdout/stderr,
launch/lifecycle records, SHA sidecars, Fable advice, and final verification in
the sealed root. Maximum runtime is one CPU minute.

The script must fail closed on malformed safetensors, unsupported dtype,
non-finite floating values, duplicate/missing tensor names, shape/dtype drift,
checkpoint-manifest hash mismatch, trainer-state step mismatch, or an input
source/hash mismatch.

## Exact gates

### E1: initialization identity

Step-0 `model.safetensors` and `ema.safetensors` must be byte-identical and
both hash to
`0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79`.

### E2: two-update Adam reconstruction

For each model tensor with saved `adam.step.<name> == 2`, use the saved step-2
first and second moments and raw `w_2` to invert the second AdamW update:

```text
mhat_2 = m_2 / (1-beta1^2)
vhat_2 = v_2 / (1-beta2^2)
w_1 = (w_2 + lr_2 mhat_2/(sqrt(vhat_2)+eps)) / (1-lr_2 wd)
```

Use `beta1=0.9`, `beta2=0.95`, the configured epsilon and weight decay, and
the logged `lr_2=4e-6`. Emulate in F32:

```text
e_1 = d w_0 + (1-d) w_1
e_2 = d e_1 + (1-d) w_2.
```

For `d=0.999`, require at least 95% of eligible elements bitwise equal to the
stored EMA and maximum deviation at most eight representable F32 values
(ULPs). Require every wrong-decay control to have less than 50% bitwise
equality and maximum deviation above 1,000 ULPs. Report eligible tensor and
element counts, exact matches, fractions, and maximum ULPs for all four
decays. These thresholds were selected after Fable observed the same files and
are deterministic replication tolerances, not statistical evidence.

### E3: source and checkpoint binding

Machine-check the frozen source digests and record bounded source evidence for:

- EMA initialization as an independent copy of the raw VarMap and default
  decay 0.999 in `src/p2/optimizer.rs:10,23-45`;
- exactly one EMA recurrence call after each successful optimizer step in
  `src/p2/train.rs:10474-10526`;
- raw and `ema.weights()` saved to different named artifacts in
  `src/p2/train.rs:8477-8491`;
- the checkpoint saved after evaluation restores live raw weights in
  `src/p2/train.rs:10578-10712`;
- the run's resume count is zero.

This is source/run binding evidence, not proof that every historical machine
instruction executed correctly.

## Numeric recurrence check on invariant tensors

Identify tensors whose `adam.step` stays zero and whose raw bytes remain
unchanged at every checkpoint. Replay 2,048 F32 EMA updates against that fixed
raw value and compare with each saved EMA boundary. Report bitwise fraction
and maximum ULP error. A maximum over 1,000 ULPs is an exact-gate failure;
smaller nonzero drift is documented as expected F32 recurrence rounding. Also
report any tensor with zero optimizer step whose raw value changed.

## Sparse-window consistency diagnostics

These gates assume an approximately linear raw path within each 256-update
window from step 256 through 2,048. They can detect gross wrong-decay/reset
behavior but cannot prove a recurrence fault.

For each of the seven windows:

1. Fit decay on a fixed grid from `0.99` through `0.9999` in increments of
   `0.00001`, minimizing whole-model squared residual under the linear-path
   recurrence. Require the fitted decay in `[0.9985, 0.9995]`.
2. At decay 0.999, compute the implied exponentially weighted raw mean
   `(e_b-d^D e_a)/(1-d^D)`. Compare its displacement from raw window start to
   the raw endpoint chord. Require cosine at least 0.85 and projected fraction
   in `[0.3, 0.8]`.
3. At every checkpoint at or after 256, every tensor whose raw bytes differ
   from initialization must have EMA bytes different from both initialization
   and same-step raw bytes.

Failure of only these diagnostics yields
`sparse_checkpoint_consistency_not_established`, not a proven fault.

## Fixed outcomes and decision

- Any integrity or exact-gate failure:
  `ema_recurrence_or_binding_fault_indicated`; preserve the root and open a
  parallel implementation investigation.
- Exact gates pass but a sparse-window diagnostic fails:
  `sparse_checkpoint_consistency_not_established`.
- All gates pass: `consistent_no_ema_fault_detected` with the required caveat
  that intermediate sparse updates are not certified.

Under every outcome, the raw trajectory remains a no-useful-learning result.
Proceed next to a separately preregistered, raw-weight-gated, same-row 2x2
positive-control overfit. Before launch, prove its target routes only through
parameters that received gradients in this recipe. EMA-only overfit gates are
forbidden because decay 0.999 retains about 61% of initialization after 500
updates.
