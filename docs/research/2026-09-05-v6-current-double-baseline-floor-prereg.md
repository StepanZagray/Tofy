# V6 CurrentDouble changed-exactness baseline-floor preregistration (2026-09-05)

## Decision and bounded claim

The split-CE mass mechanism is confirmed at initialization, but no experiment
shows that any weighting mode learns useful changed transitions. A matched
quality screen is uninformative if its CurrentDouble control still has zero or
near-zero changed exactness at the selected update budget.

This A-only pilot asks one question: under the exact Foundation-v2 synthetic V6
2x2 CurrentDouble recipe, does a fresh disjoint seed reach enough exactly
decoded changed transitions at a nominal 2,048-update budget to make a later
loss-of-function or coarse non-inferiority screen informative? It does not test
UnitMassBalanced, choose a model, or support generalization, planning, or ARC.

The only locally available multi-step V6 A run is integrity-invalid under the
repaired recipe. As non-admissible planning context, it decoded `8/382` changed
transitions exactly at update 1,024 and `10/382` at update 2,048. Those values
motivate the fixed count gates below but are not a baseline result.

## Why the budget is 2,048

The in-trainer gate runs only every 1,024 updates, so smaller budgets cannot
measure changed exactness. A nominal 1,024-step run spends 500 updates warming
up, begins WSD decay around update 870, and evaluates EMA weights that retain
about 36% initialization residue. At 2,048 updates, residue is about 13%, the
run supplies both 1,024 and 2,048 gate readings, and the latter uses the exact
schedule a 2,048-step quality screen would use.

Total steps are part of the WSD schedule and resume contract. Therefore this
pilot is a fresh nominal 2,048-step run with no resume or adaptive 1,024→2,048
ladder. Its selected budget, if any, is exactly 2,048. A failure does not
authorize a longer run; it requires a new preregistration.

## Runs and fixed sequence

### P0 — profile-publication implementation smoke

- Fresh seed/init seed 105, CurrentDouble, two total updates.
- V6 2x2, physical batch 128, accumulation 1, effective batch 128.
- `--pressure-updates 1`, `--profile-updates 2`, periodic checkpointing off.
- No gate or quality result is possible or interpreted.
- Pass only if the update-1 pressure packet is present, the update-2 full
  Candle Graph bundle publishes atomically, the report/evidence manifest names
  it, and a small representation-only `p2-eval` successfully loads the
  update-2 `ema.safetensors`. Source/binary/device provenance must pass, every
  artifact must rehash, and process/lock/GPU cleanup must complete.
- Maximum runtime 15 minutes. Failure blocks P1.

### P1 — A-only floor pilot

- Fresh seed/init seed 5, disjoint from later treatment-screen seeds 2/3/4.
- CurrentDouble, V6 2x2, physical batch 128, accumulation 1, effective batch
  128, exactly 2,048 nominal optimizer updates, no resume.
- Pressure updates `1,1024,2048`; full profile updates `2,1024,2048`.
- Complete checkpoints every 256 updates and permanent boundary at 2,048.
- Profile/pressure/gate coexistence at updates 1,024 and 2,048 is intentional.
  The measured profile region includes the additional component-pressure
  backward passes, so it is evidence-path validation rather than unbiased
  production throughput.
- Maximum runtime 3 hours. The planning estimate is about 80 minutes from the
  invalid historical run's cadence; the estimate is operational only.

Run P0, verify and seal it, then run P1 under the global GPU lock. Each gets a
never-reused root. No P0 artifact, weights, seed, or metric enters P1.

The exact training command templates are fixed below. `<BINARY>`, `<P0>`, and
`<P1>` are replaced only by paths recorded before launch.

```bash
<BINARY> p2-train \
  --recipe foundation-v2 --world-core-v6 --data-contract-v6 \
  --v6-recursion-steps 2 --device cuda --seed 105 --init-seed 105 \
  --steps 2 --physical-batch 128 --grad-accum 1 \
  --checkpoint-every-steps 0 --pressure-updates 1 --profile-updates 2 \
  --split-ce-weighting current-double --output-dir <P0>

<BINARY> p2-eval \
  --checkpoint <P0>/checkpoints/step-000000000002/ema.safetensors \
  --train-config <P0>/config.json \
  --seed 115 --iid-seed 106 --synthetic-episodes 1 --physical-batch 64 \
  --ptrm-k 1 --device cuda --eval-mode representation \
  --identifiability false --profile-eval false \
  --output <P0>/ema-load-smoke.json

<BINARY> p2-train \
  --recipe foundation-v2 --world-core-v6 --data-contract-v6 \
  --v6-recursion-steps 2 --device cuda --seed 5 --init-seed 5 \
  --steps 2048 --physical-batch 128 --grad-accum 1 \
  --checkpoint-every-steps 256 --pressure-updates 1,1024,2048 \
  --profile-updates 2,1024,2048 --split-ce-weighting current-double \
  --output-dir <P1>
```

No command passes `--split-ce-changed-budget` or the legacy singular profile
flag.

## Fixed source, device, and provenance

- Before launch, bind a reviewed, pushed, clean preregistration revision whose
  `src`, `Cargo.toml`, and `Cargo.lock` are byte-identical to `96f0c449`.
- Build once in a new target directory with embedded command
  `cargo build --release --locked --features cudnn`. Bind the exact binary
  SHA-256, Cargo.lock SHA-256, Candle Graph revision, GPU UUID, driver, feature
  set, and full commands; reuse the same binary for P0, P1, and P2 below.
- Synthetic data only; no public ARC recordings or scorecard.
- No changed-budget override; CurrentDouble must report coefficients `w/1`,
  mass `w+1`, and share `w/(w+1)` at every pressure sample.
- `research_claim=false`; no best or final checkpoint is promoted to a model
  decision. The pilot seed 5 can never be reused as a treatment-screen arm.
- Any source/runtime revision mismatch, dirty/unpushed dependency, unknown
  build command, hash mismatch, reused root, resume attempt, or device mismatch
  fails closed.
- GPU UUID, driver, Cargo.lock digest, global-lock state, and cleanup are
  captured in external launch/lifecycle records because generated Tofy
  provenance does not contain every one of those fields.

## P1 population, execution, and evidence gates

- The report seed/init seed must be 5. Training population/content
  fingerprints must differ from registered seed 2, 3, and 4 values.
- Both gate entries must use the same fixed 512-row selection-only population,
  population fingerprint, and content-mask fingerprint. Each must report at
  least 256 changed transitions and at least 32 shuffled-action
  outcome-changing tuples.
- Report and checkpoint state must contain gate entries at exactly 1,024 and
  2,048; pressure at exactly 1, 1,024, and 2,048; profiles at exactly 2, 1,024,
  and 2,048; and periodic checkpoints at the registered cadence.
- Each pressure sample must have 128 main plus 32 rollout rows, 16 rollout
  fragments, an active action bundle, explicit zero/null grounding, both pixel
  strata nonempty, and finite route norms/cosines. Record `w`, coefficients,
  changed share/mass, prediction/encoder/action/rollout pressure, combined
  direction, and clip scale at all three samples. The one-time rollout
  activation record must contain weighted recurrent-core gradient L2 at update
  1; at updates 1,024 and 2,048, use the pressure sample's `rollout` component
  global/AdamW/Muon route L2 because recurrent-core L2 is not emitted there.
- Every generated evidence artifact and surviving recursive file must rehash.
  Exact run processes and locks must be gone and the GPU idle before sealing.
- The only acceptable evidence caveat is that profiled timings include
  pressure instrumentation. Any missing profile/pressure artifact, non-finite
  skip on a registered update, abort marker, missing final checkpoint, or
  evaluator invariant failure makes P1 invalid and blocks interpretation.

At update 1,024, monitor only the integrity/count gates above. Do not stop or
select based on changed exactness. If population support is below its minimum,
stop for integrity without using the metric; otherwise complete update 2,048.
The built-in scientific abort cannot fire with only two gate evaluations.
The monitor reads
`checkpoints/step-000000001024/gate_history.json`; the population-support floor
is operator-enforced rather than a trainer gate. If outcome-changing tuples are
below 32, the trainer itself bails during evaluation before writing the
step-1,024 checkpoint, so the last surviving periodic bundle is step 768; this
is an integrity stop, not an unexplained crash.

## Primary metric and fixed decision rule

Use only the final update-2,048 gate-history entry, never `best_changed_exact`
or the best-checkpoint directory. Let `n` be `changed_transitions` and `p` be
`one_step_changed_exact`. Require `p*n` to be within `1e-9` of an integer and
define `k = round(p*n)`, the number of transitions whose factually changed
gameplay pixels are all exactly decoded.
If the trainer's semantic non-noop `changed_transitions` field ever diverges
from the metric's gameplay-changed denominator, the integrality/evaluator
invariant fails closed rather than silently changing `n`.

- **Pass:** `k >= 10`. Adopt 2,048 as the fixed candidate budget for a later
  coarse non-inferiority/loss-of-function screen.
- **Inconclusive:** `5 <= k <= 9`. The budget may support only a paired
  discordant-row loss-of-function design; it does not support ordinary
  non-inferiority. Stop for fresh design and Fable review.
- **Fail:** `0 <= k <= 4`. Select no budget and do not escalate within this
  preregistration.

The update-1,024 changed-exactness value is descriptive only. There is one
fixed pilot seed and no confidence interval or promotion test. Deterministic
cuDNN is not claimed; count bands are used rather than a zero/nonzero rule.

## P2 — fixed final-checkpoint offline diagnostic

After any integrity-valid P1 outcome, run one full synthetic `p2-eval` on the
explicit update-2,048 checkpoint, never the best directory:

```bash
<BINARY> p2-eval \
  --checkpoint <P1>/checkpoints/step-000000002048/ema.safetensors \
  --train-config <P1>/config.json \
  --seed 15 --iid-seed 6 --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --q-mse-threshold 0.05 --device cuda \
  --identifiability false --profile-eval true \
  --output <P1>/eval-step-000000002048.json \
  --episode-jsonl <P1>/eval-step-000000002048.episodes.jsonl
```

P2 validates full evaluator/profile publication and inventories fixed held-out
split denominators for the later quality design. Its composition, translation,
size, operator, rollout, action, non-collapse, and planner/Q metrics are
diagnostic only and cannot change the P1 decision. Maximum runtime 45 minutes;
any evaluator/hash/provenance failure is recorded but does not retroactively
change an already valid P1 count verdict.
The checkpoint is the same EMA weight set used by the in-trainer gate; P0's
EMA-load smoke is the fail-closed loader preflight.

For pressure reporting, “prediction/encoder/action/rollout” maps to emitted
components `pred_ce`, `enc_ce`, `action_bundle` plus `inverse_action`, and
`rollout`. If the update-1,024 one-step collapse gate disables rollout, the
update-2,048 rollout route may be zero; record the gate transition and do not
misclassify the finite zero as a missing packet.

## Budget, stop rule, and next decision

Maximum compute is one two-update smoke, one 2,048-update A pilot, and one
offline evaluation: 4 hours wall-clock total. Stop on any integrity failure,
runtime cap, or non-finite registered update. After P2, stop for complete
analysis, sealing, and Fable 5.1 judgment; no treatment run auto-launches.

If P1 passes, the next preregistration must compare at least:

- A — CurrentDouble;
- C — equal-means UnitMassBalanced, share 0.5;
- D — UnitMassBalanced with changed share `50/51`.

D is a faithful mass-only control only while A's batch-local `w` is at the cap
50; P1 pressure samples must establish whether that premise holds at all three
registered updates. The later screen must use a larger fixed held-out population
or an explicitly coarse margin, compare EMA models, freeze identical 2,048-step
WSD/LR/clip/data/evaluator contracts, and decide by per-seed conjunction. This
pilot licenses only a budget/evidence-path decision, never UnitMassBalanced
quality, model promotion, or ARC performance.
