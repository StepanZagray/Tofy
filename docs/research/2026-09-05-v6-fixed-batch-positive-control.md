# V6 fixed-batch raw-weight positive control

Date: 2026-09-05

Evidence class: implementation / learnability diagnostic

Research claim: false

Promotion authority: none

Public ARC data authorized: no

## Registration boundary

This contract is frozen before implementation and before any new optimizer
update. It follows the failed seed-5 CurrentDouble baseline, the frozen
checkpoint trajectory, and the deterministic EMA audit. The baseline learned
no useful raw or EMA predictor, and the EMA implementation showed no detected
fault. The cheapest remaining prerequisite is therefore whether the exact
production update can fit one repeated, action-ambiguous 2x2 population at all.

Fable 5.1 advised the design in read-only mode. Its retained memo is
`/home/stepan/Coding/Personal/.tofy-build/reviews/fable-fixed-batch-positive-control-advice-20260905.md`,
SHA-256
`b7a7f4f1c9ad812da0897f6f5ccfd258f72af0f734ebebaa16740ec3a2ff8d58`.
The primary agent checked the cited baseline values and model tensor names
against the frozen artifacts before registering this document.

## Bounded empirical claim

On one deterministic 128-row training batch containing a complete factual
branch group with one shared current frame and multiple action-dependent next
boards, can the raw V6 model initialized exactly like the failed seed-5 run,
and updated through the complete failed Foundation-v2 recipe, both:

1. fit a majority of the batch's board-changing rows exactly; and
2. exactly emit at least two different changed outcomes for different actions
   applied to that one shared frame?

`PASS` supports only this fixed-population learnability claim. It does not
support generalization, ARC-AGI-3 performance, treatment promotion, or the
blocked A/C/D branches. A fixed-batch fit can be memorization. No result from
this diagnostic can be called globally optimal or paper-faithful evidence.

## Comparator and rationale

The primary comparator is the same raw initialization at update 0, plus copy,
background, and direct-target scorer controls on the same rows. The failed
streaming baseline's update-1 values bind implementation fidelity. That run
did not score its step-1,024 raw checkpoint on this fixed training population,
so no such comparator is claimed.

The earlier E2F control is insufficient: it optimized a two-row context-wiring
loss from a trained E2 EMA checkpoint under plain AdamW and constant learning
rate. It did not exercise fresh initialization, the CurrentDouble split loss,
hybrid Muon/AdamW, the full auxiliary mixture, dedicated rollout, WSD warmup,
EP control, or the action-conditioning route being tested here.

## Frozen inputs and population

- Baseline root:
  `/home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT`.
- Baseline source revision:
  `d0468a808d0d3cd2754dc1b31e4a85ab636fcc36`.
- Baseline recursive seal-manifest digest:
  `e686a783b13eadc9f099d2ba5bdce68ecc10923e686db6fc86cc9bb4c02abfc5`.
- Initialization: baseline
  `checkpoints/step-000000000000/model.safetensors`, loaded as raw weights by
  `load_varmap_exact`; SHA-256
  `0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79`.
- Frozen baseline `config.json` SHA-256:
  `874d53e53e68cfb5dbaada83bf25b5558f2874ae23f3af62997e13ec1263f3c1`.
- Optimizer and EMA state: fresh. EMA decay is the production default and EMA
  is descriptive only.
- Configuration: the frozen baseline `config.json`, including seed/init seed
  5, physical batch 128, accumulation 1, F32 recurrent core, hidden dimension
  128, action dimension 32, CurrentDouble weighting, learning rate `1e-3`,
  weight decay `0.01`, Muon momentum `0.95`, Muon RMS scale `0.2`, rollout
  weight `0.02`, `steps_per_lesson=2048`, 1,024 SIGReg projections, and 17
  SIGReg knots.
- Main population: `adaptation_v6_stream_schedule` and `data_contract_v6`,
  `compose_mixed_stream_batch` at progress `0.0`, batch index `0`, train split,
  physical batch 128. The same rows and order are repeated every update.
- Rollout population: seed `5 ^ 0xA011_0A77_0000_0002`, batch index `0`, train
  split, 16 complete fragments / 32 rows. The same rows and order are repeated
  every update.
- Record the exact serialized population files, content masks, provenance,
  source counts, factual ranges, content digests, and SHA-256 values before
  training.

Deterministic CPU generation before implementation recorded 128 rows, exactly
95 board-changing rows, one complete ten-row factual group, and eight distinct
changed next-board classes in that group. All ten group rows had identical
4,096-pixel current frames, empty and identical contexts, identical goal
features, and identical UNKNOWN-family operator-conditioning vectors. The
registered binary must reproduce those exact counts and identities from the
actual serialized host vectors. The frozen checkpoint does contain the
zero-initialized `operator_conditioning_proj.*` parameters; their presence is
required. If any assertion fails, stop invalid. Do not search a later batch;
doing so would break the frozen population and update-1 binding.

## Registered arms and update path

### P: full production-objective arm

Run exactly 1,024 updates on the fixed populations. Each update must reproduce
the production order:

1. dedicated rollout forward and weighted backward;
2. full Foundation-v2 main loss with `rollout_enabled=false` and the current EP
   weight;
3. EP controller measurement/update every 128th update before forming the
   effective main total;
4. main-total backward and accumulation with rollout gradients;
5. global gradient clipping at L2 norm 1;
6. hybrid optimizer step at
   `foundation_v2_wsd_learning_rate(update, 2048)`; and
7. one EMA update after a successful optimizer step.

Every update uses production SIGReg parameters: 1,024 projections, 17 knots,
and seed `5.wrapping_add(zero_based_update)`; thus update 1 uses seed 5 and
update 1,024 uses seed 1,028. Argument validation and tests must bind this
progression rather than permitting a fixed projection seed.

Start EP weight at `0.01`. Any non-finite loss or gradient is an integrity stop,
not a skipped update. This is a deliberate stricter deviation from production,
whose bounded policy can skip isolated non-finite updates. Save and score
boundaries `0, 1, 128, 256, 512, 768, 1024`. Raw and EMA snapshots are both
preserved, but every decision gate uses raw weights.

### Q: matched prediction-only arm, conditional

Do not launch Q initially. Launch it only if P's final class is `FAIL` after an
integrity-valid run. Q uses the identical initialization, populations, 1,024
updates, WSD schedule, optimizer, clipping, snapshots, and scoring, but
back-propagates `pred_ce` alone: no auxiliary, EP, or dedicated-rollout
gradient enters the optimizer, and the EP controller does not run. Other losses
may be computed for logging only. Hybrid-optimizer AdamW weight decay remains
active on parameters that receive prediction gradients. Q uses its own
never-reused root and cannot inherit P's weights or moments.

Q distinguishes failure caused by competing objectives from a failure in the
core prediction/optimizer/action path. It has the same outcome gates as P and
the same lack of promotion authority.

## Update-1 production binding

Before committing update 1, record the full-objective route gradients and
require the P arm to reproduce these baseline values with relative tolerance
`1e-6` and absolute tolerance `1e-6`:

| Quantity | Frozen value |
|---|---:|
| total loss | `175.38963317871094` |
| prediction CE | `155.89109802246094` |
| rollout loss | `1.3734819889068604` |
| pre-clip gradient L2 | `450.4061279296875` |
| learning rate | `0.000002` |
| changed content pixels | `176` |
| unchanged content pixels | `524112` |
| changed coefficient | `50` |
| coefficient mass | `51` |
| rollout fragments | `16` |

A mismatch makes P `INTEGRITY_INVALID` and stops before a long run. The
two-update CUDA smoke must pass this binding using the exact launch binary.
Q is intentionally not required to match the full-objective total or gradient
norm; it must match P's population, initialization prediction CE, schedule,
and update-1 `pred_ce`-only per-prefix gradient norms under the same `1e-6`
absolute/relative tolerance.

## Gradient-route premise

At the untrained boundary, use a `pred_ce`-only backward on the frozen main
population and record per-prefix L2 norms. Required finite and strictly
positive:

- `block.`;
- `exact_grounding_head.decoder.`;
- `action_emb.`;
- `action_proj.`;
- `action_film_gamma.weight`;
- `action_film_beta.weight`;
- `spatial_action_proj.weight`;
- `operator_conditioning_proj.weight`; and
- `encoder.`.

Required absent or exactly zero:

- `coord_proj.`;
- `grounding_head.decoder.`; and
- `prefix_head.`.

Repeat the positive-route check, including `spatial_action_proj.weight` and
`operator_conditioning_proj.weight` specifically, on the accumulated, clipped
P-arm gradient store at update 1. Any violation is `ROUTE_PREMISE_FAIL` and
stops the run. The zero prefixes are topology assertions, not evidence that
those components are healthy.

## Frozen raw-weight metrics

V6 has no status row: scoring uses all 64 rows and 4,096 pixels of every frame.
Let `C` be rows whose current and next boards differ; its registered size is
exactly 95 and values below 32 are independently invalid even if the exact
census check were accidentally weakened. Full exactness uses the established
raw `one_step_full_exact` scorer seam and means all 4,096 predicted pixels equal
their target. Changed exactness means every factually changed pixel equals its
target. Report both counts and denominators, copy/background/direct-target
controls, and the number of exact rows whose changed target pixels are not all
the configured empty/background colour.

For the factual group, canonicalize next-board classes from all 4,096 target
pixels. Record the maximum class multiplicity `m`, the raw-full-exact branch
count, and the set of distinct changed classes reproduced exactly. The frozen
premise proves all non-action inputs (frame, context, goal features, and
operator conditioning) are identical, so a deterministic action-blind model
emits one board and cannot exactly reproduce two distinct classes.

Also report, but never gate on, the sidecar-aware ACTION5/ACTION6 shuffle:
total rows, eligible rows, changed tuples, outcome-changing tuples,
counterfactual-target changed-pixel accuracy, and factual-target accuracy under
the shuffled action. This small fixed batch may not meet the existing
32-outcome-changing-row sensitivity floor.

## Gates and decision tree

Define `BF(t)` as raw full exactness on the 95 rows in `C` at update `t` being
at least `0.5`. Define `MONO` as raw changed-exact counts non-decreasing across
updates 512, 768, and 1,024. Define `AR(t)` as at least two distinct changed
factual-group outcome classes reproduced raw-full-exact on their own action
rows at update `t`. The `0.5` threshold is a registered engineering effect
size: it demands exact fitting of a majority, vastly above the failed baseline
and incapable of being passed by the registered copy/background controls. It
is not a confidence threshold and does not imply generalization.

Classify in this fixed priority order so every valid run has exactly one class:

- `PASS / same_row_action_conditioned_fit` if `BF(768) && BF(1024) && MONO &&
  AR(768) && AR(1024)`. Next: preregister a multi-batch generalization screen;
  do not run public ARC or promote a checkpoint.
- Else `PARTIAL_FIT` if `AR(1024)`. Next: preregister an otherwise identical
  extension to 2,048 updates.
- Else `FIT_WITHOUT_ACTION` if `BF(1024)`. Next: diagnose action projection,
  zero-initialized FiLM, and spatial action conditioning.
- Else `FAIL`. Next: launch conditional arm Q.
- `INTEGRITY_INVALID` or `ROUTE_PREMISE_FAIL`: no model conclusion. Repair the
  diagnostic and rerun under a new root and new registered revision.

No post-hoc checkpoint or threshold substitution is allowed. Under every
outcome, A/C/D and public ARC evaluation stay blocked.

## Compute, stop, and provenance contract

- P maximum: 1,024 updates, one seed, one fixed main/rollout population, 90
  minutes wall clock on the same RTX 5060 Laptop GPU. Expected runtime is
  approximately 45 minutes from measured baseline throughput; this is a
  diagnostic screen, not a promotion
  claim.
- Q maximum, only on P `FAIL`: the same cap in a separate launch.
- Physical batch 128 and accumulation 1 are frozen; no OOM-driven batch change
  is allowed. A failed launch is infrastructure evidence only.
- Before launch: clean reviewed pushed commit; exact dependency revisions;
  locked CUDA build; binary SHA-256; idle-device check; GPU UUID/name/memory and
  driver; two-update exact-binary CUDA smoke; exact checkpoint and population
  hashes; never-reused root; lifecycle `running`.
- Stop immediately on source/binary/checkpoint/population mismatch, update-1
  binding failure, route failure, non-finite values, wrong rollout fragment
  count, failed snapshot/hash write, external interruption, or 90-minute cap.
- Preserve configuration, source/dependency revisions, build command and
  features, Cargo.lock SHA-256, binary hash, GPU identity, command, timestamps,
  stdout/stderr, population files and digests, metrics, snapshots, lifecycle,
  Fable judgments, and a recursive manifest. Stop tracked child/telemetry
  processes before sealing, verify the tree, and record the manifest digest
  outside the root.

## Implementation and review acceptance

Implementation may add one isolated diagnostic module and CLI plus the minimum
`pub(crate)` helper exposure. Production trainer, benchmark, evaluator, model,
optimizer behavior, and fixed recipes must remain byte-for-byte behaviorally
unchanged. Required tests cover population premise, action-blind class ceiling,
outcome classification, route-prefix policy, SIGReg seed progression, and
argument validation.

Before any GPU launch, Fable 5.1 must independently inspect the diff against
this document, verify the production-update equivalence and fail-closed gates,
and return an explicit launch `GO`. Any blocking finding is repaired and
re-reviewed. After a completed arm, Fable must judge the machine-readable
report and artifact consistency before the result is interpreted.
