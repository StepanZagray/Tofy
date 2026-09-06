# V6 multi-batch function-learning screen (preregistered)

Status: **design freeze; runnable implementation reviewed GO; no CUDA launch yet**
Date: 2026-09-05 CDT  
Evidence class: **selection-only single-seed screen**  
Research claim: **false until a complete registered root is analyzed**  
Promotion authority: **next-diagnostic branch selection only; no checkpoint promotion**  
Public ARC authorization: **none**  
Selected by: registered fixed-batch P outcome `same_row_action_conditioned_fit`  
Parent P root: `/home/stepan/Coding/Personal/.tofy-build/v6-fixed-batch-p-registered-20260905T170914-CDT`  
Parent P report SHA-256: `a837bb95be922376fee9d7ec9b0f9a5ec03aa002ef55188e4b591bca51e64e53`  
Parent P manifest SHA-256: `9867affe63584b5736c81d41833ab8259509a13fc056e88ae3036c2c65d7908f`  
Parent P identity: `sha256:521c077e4e7030697cabcfc1a4036d32b6e57e709d31625e05a413031fa89d7d`  
Parent outcome review: Fable 5.1 High, SHA-256
`408d9c3a71bb51b00849abb35313a31a5363c913bb1a3830e01c7c58442a34f1`  
Preregistration NO-GO review: Fable 5.1 High, SHA-256
`edeaab6097ae1276b8d991948e9c12926c664f0a237b4425813c0628459fc0f6`  
Corrected preregistration GO review: Fable 5.1 High, SHA-256
`60fa397ee99ed06b4d457838fa6e339713fb2cd12eac39b9a48ea13d6a0d30fd`

Host Slice 1 NO-GO review: Fable 5.1 High, SHA-256
`b9d2741c0fa45cc46a7b5ada213f4d3ce5651ce3e9a04a54ad671e8ec8ec5028`

Corrected host Slice 1 GO review: Fable 5.1 High, SHA-256
`ec13909147dbddc8000c6845257abd2c228e83bb1b26d7d8ba2ec91170500193`

Runnable Slice 2 NO-GO review: Fable 5.1 High, SHA-256
`a67e88512235a32818d98be4b0d2e9cf6f55ec2b1e889c5c5bc835575e3ffd89`

Corrected runnable Slice 2 GO review: Fable 5.1 High, SHA-256
`02df7fd5b68f64ba1998909fa9924168d365b6cfa31e370c4637aec1353786ab`

Pre-implementation census clarification (recorded before any preflight or G
checkpoint existed): the frozen `1006/1005` tuple counts include the sidecar
episode operator used for exact counterfactual replay; they are not the
operator conditioning visible to the V6 model. Recomputing the same keys with
the model-visible UNKNOWN operator gives `955/945` unique train/held-out
tuples. Both definitions have exactly zero train/held-out overlap. The
implementation freezes and checks both. This corrects terminology and adds a
strictly coarser leakage check; it changes no population, threshold, outcome,
or branch rule.

This document must be reviewed, committed, and pushed before implementation.
The implementation commit, exact locked CUDA binary, and its sealed preflight
must receive a separate GO before the registered run.

## 1. Question and bounded claim

Question: starting from the same seed-5 step-0 weights, can the exact
Foundation-v2 production update learn a next-board function over eight fixed
2x2 batches that transfers to eight deterministic, index-held-out batches and
to outcome-changing ACTION5/ACTION6 interventions?

The empirical hypothesis is:

> Cycling eight fixed same-distribution V6 batches for 2,048 full-objective
> updates produces both nontrivial exact prediction on never-trained batch
> indices and correct held-out action effects, including spatial coordinates.

The claim is limited to the exact synthetic generator, seed, batch indices,
initialization, objective, hardware/software identity, and finite populations
below. It is not a claim about streaming training, another seed, OOD
composition, planning, ARC-AGI-3, or architecture optimality.

## 2. Why this is the weakest remaining prerequisite

Registered P proved only that repeated updates on one 128-row batch can reach
`81/95` raw full-board exact and four exact trained discrete-action outcomes.
It did not fit any of four ACTION6 coordinate branches. On 27
outcome-changing shuffled ACTION5/ACTION6 tuples, final predictions followed
the counterfactual target on only `13/111` disagreement pixels and retained
the factual target on `89/111`. A frame-keyed lookup is therefore a live
counterexample to function learning.

No theorem derives generalization from P. A network can memorize all eight
training batches and fail every unseen frame or pairing. Conversely, failure
to fit the eight-batch train union would mean capacity/optimization does not
scale even before generalization is asked. One matched single-arm feasibility
screen is cheaper and more decisive than changing loss weights, recursion,
memory, planning, or public ARC behavior.

## 3. Frozen identity and parent admission

- Initialization checkpoint:
  `/home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/checkpoints/step-000000000000/model.safetensors`,
  SHA-256 `0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79`.
- Train config:
  `/home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/config.json`,
  SHA-256 `874d53e53e68cfb5dbaada83bf25b5558f2874ae23f3af62997e13ec1263f3c1`.
- Parent P must be reverified from its external manifest and report. Its
  registered outcome must be exactly `same_row_action_conditioned_fit`; all
  parent provenance, route, update-1, and identity checks must pass.
- Source must descend from parent source `55d3e691`, be reviewed, clean, pushed,
  and record exact Tofy and candle_graph revisions, Cargo.lock hash, locked
  cuDNN build command/features, binary SHA-256, runtime checkout, and GPU.
- Device is CUDA. The proven stable physical/effective batch is `128/128` with
  accumulation `1`; it remains fixed to preserve P's causal update contract.
- No public ARC data, recording, live endpoint, prior adapted weight, or
  synthetic shard may be read.

## 4. Frozen populations

All populations use `MixedStreamConfig { batch_size: 128, seed: 5,
schedule: adaptation_v6_stream_schedule, data_contract_v6: true, ..default }`,
`progress=0.0`, and `V5DataSplit::Train`. “Held-out” below means held out by
deterministic batch index from optimization; it is not the generator's
`HeldOutComposition` split and must not be described as OOD.

### Training main and rollout batches

- Main indices: `0..=7`.
- At zero-based update `u`, train only main index `u mod 8`.
- Rollout indices: `0..=7`, generated under seed
  `5 XOR 0xA0110A7700000002`, 16 adjacent fragments / 32 rows.
- At update `u`, use rollout index `u mod 8` in lockstep with the main batch.
- Exactly 2,048 optimizer updates give exactly 256 visits to every train batch.

### Index-held-out evaluation batches

- Main indices: `8..=15`.
- These 1,024 rows are serialized before training and are never supplied to a
  loss, gradient, optimizer, EMA update, EP controller, or checkpoint selector.
- Snapshot evaluation is read-only and uses raw weights for every gate.

### Preregistered host census

One-shot host generation at parent source `55d3e691` established these values
before this document was written. The preserved census artifact is
`/home/stepan/Coding/Personal/.tofy-build/reviews/v6-multibatch-host-census-55d3e691.json`,
SHA-256
`060937144393121c30ca469f2175e2e297801e2a8728b3afa7f6bfcba98130a1`:

| Batch indices | Rows | Changed rows | Factual groups | Global shuffle eligible / changed / outcome-changing | All-row ACTION6 changed | Changed factual ACTION6 |
|---|---:|---:|---:|---:|---:|---:|
| train `0..=7` | 1,024 | 725 | 8 | 218 / 217 / 193 | 112 | 24 |
| held-out `8..=15` | 1,024 | 726 | 8 | 219 / 217 / 200 | 108 | 25 |

Per-batch changed rows are
`[95,86,87,84,91,94,91,97,75,92,99,87,85,101,98,89]`.
Per-batch changed factual-group rows are
`[8,8,4,9,9,9,8,5,8,9,9,8,5,5,9,8]`.
Per-batch distinct changed factual outcome classes are
`[8,6,3,8,5,5,6,4,6,8,5,6,4,4,8,7]`.
Per-batch outcome-changing shuffle counts are
`[27,27,21,24,24,27,23,24,23,25,25,26,26,23,27,26]` when shuffled within
each batch; the registered gate uses the global-union counts in the table.
Held-out per-group counts of distinct changed ACTION6 target classes are
`[4,4,4,4,1,0,4,4]`, so six groups can satisfy the strengthened coordinate
gate below.

Train/held-out exact overlaps are zero for current frames, frame-action keys,
sidecar-operator input tuples, and model-visible UNKNOWN-operator input tuples.
Unique train/held-out counts are `701/706` frames, `871/876` frame-action keys,
`1006/1005` sidecar-operator tuples, and `955/945` model-visible tuples.
Population construction must reproduce every count or fail before CUDA
training. Serialize every batch, hash it, and bind an ordered union identity.

The eight train rollout batches contain 256 rows, 256 unique current frames,
and 256 unique frame-action keys. Their exact current-frame and frame-action
overlap with the held-out main union is zero. The preflight must recompute this
as well as main-train/held-out overlap and save the complete result as
`population/census.json` with a bound SHA-256.

## 5. Frozen update path and budget

Arm G is a single full-objective arm. It uses the same initialization, F32
model, `CurrentDouble` split CE (`50/1`, mass 51), Foundation-v2 loss,
separate 16-fragment rollout backward, parameter-gradient filtering,
accumulation order, global clip `1.0`, hybrid AdamW/Muon optimizer, EMA update,
and EP controller as parent P and production training.

- Updates: exactly `2,048`.
- WSD schedule denominator: `2,048`; LR therefore follows the production
  schedule, reaches `1e-3` at update 500, and decays from update 1,741 through
  2,048 toward `1e-4`.
- SIGReg: 1,024 Gaussian projections, 17 knots, update seed
  `5.wrapping_add(zero_based_update)`.
- EP controller: update before the main effective total on exact one-based step
  multiples of 128, using the active batch's encoder EP/prediction gradient
  norms. Because zero-based selection is `u mod 8`, every controller
  measurement falls on train batch index 7. This production-faithful
  concentration is a known limitation, not silently pooled evidence.
- Snapshots: raw and EMA at `0, 1, 256, 512, 1024, 1536, 2048`.
- Gate checkpoint: raw update 2,048 only. Earlier snapshots are trajectory
  reports, not candidates; EMA never selects the result.
- Maximum registered wall time: 90 minutes. No early success stop. Any
  non-finite value, input drift, census drift, hash mismatch, CUDA loss,
  manifest failure, or route/binding failure stops as failed integrity.

Parent P measured `1,145.93 s` for 1,024 updates. The preregistered expectation
is about 40 minutes plus snapshot scoring; this estimate does not alter the
90-minute cap or any gate.

## 6. Preflight and frozen update-1 binding

Before G, run an unregistered, same-binary eight-update CUDA preflight under a
fresh root. It cycles each train main/rollout batch exactly once, scores step
0/1/8, reproduces the full census and ordered population hashes, and seals its
manifest. Registered G must bind this preflight's exact binary, source,
configuration, populations, parent P identity, and report identity.

The preflight must report training time separately from all three union
snapshot times. Estimate registered runtime as
`(preflight_training_seconds / 8) * 2048 + median(preflight_snapshot_seconds) * 7`.
Admission requires this estimate to be at most 4,500 seconds (75 minutes),
leaving at least 15 minutes below the hard 90-minute cap.

Because update 1 uses main/rollout index 0 from the same initialization, it
must reproduce parent P within `max(1e-6, 1e-6 * abs(expected))`:

| Field | Expected |
|---|---:|
| total | `175.38963317871094` |
| prediction CE | `155.89109802246094` |
| rollout loss | `1.3734819889068604` |
| pre-clip global norm | `450.4061279296875` |
| learning rate | `0.000002` |
| changed / unchanged pixels | `176 / 524112` |
| changed coefficient / mass | `50 / 51` |
| rollout fragments | `16` |

The raw and EMA step-1 checkpoint hashes must equal parent P exactly:
`7c62689c8938a9e351468a8b56e53bb9314cbe84d351a9d948f0cb457d57a3da`
and
`834b63be6767a9d99538e37034d35dcb213361afec993a54e071ea004aa1d4c7`.

At update 1, unclipped prediction gradients and clipped combined gradients
must be finite and positive for each of:
`block.`, `exact_grounding_head.decoder.`, `action_emb.`, `action_proj.`,
`action_film_gamma.weight`, `action_film_beta.weight`,
`spatial_action_proj.weight`, `operator_conditioning_proj.weight`, and
`encoder.`. `coord_proj.`, `grounding_head.decoder.`, and `prefix_head.` must
exist in topology and have zero gradient under this route.

## 7. Frozen metrics and controls

Every snapshot scores train and index-held-out unions separately.

- **Changed exact:** all target-altered gameplay pixels are predicted exactly;
  false edits elsewhere do not affect this diagnostic count.
- **Full exact:** all 4,096 pixels over all 64 V6 rows equal the target,
  reported over board-changing rows. Row 63 is part of the V6 playfield and is
  included in the changed-row census and every scorer. Do not use the legacy
  4,032-pixel helper for outcome-changing counts. This is the main fit metric.
- **All-row exact and false-edit rate:** reported, not primary gates.
- **Controls:** copy, background, and direct target under the identical scorer.
  Fixed control census is train background changed-exact `2/725`, held-out
  `3/726`, background full exact `0` in both, copy changed/full exact `0`, and
  direct-target changed/full exact `725/725` and `726/726`.
- **Per-group action routing (AR):** a factual group passes when at least two
  distinct changed target classes are raw full-board exact. Report the number
  of passing groups among eight and the exact reproduced class hashes.
- **ACTION6 coordinate fit:** report raw full-board exact changed ACTION6 rows
  for all rows and factual groups. The coordinate gate counts held-out factual
  groups reproducing at least two distinct exact changed ACTION6 target
  classes; one coincidental branch cannot pass a coordinate-blind predictor.
- **Counterfactual action:** construct the sidecar-aware global-union cyclic
  shuffle over ACTION5/ACTION6 rows, replay every action under the target row's
  recorded operator, and report total, eligible, genuinely changed,
  outcome-changing, disagreement pixels, counterfactual-target accuracy, and
  factual-target accuracy. Also stratify disagreement counts/accuracies by
  resulting shuffled action id 5 versus 6. The held-out population must have
  exactly 200 outcome-changing tuples; otherwise fail before applying gates.

For the held-out union, the exact shuffle has 1,456 disagreement pixels.
Sensitivity controls under this identical pixel set are:

| Prediction control | Counterfactual correct | Factual correct |
|---|---:|---:|
| copy current | `643/1456` | `812/1456` |
| background | `571/1456` | `568/1456` |
| direct factual target | `0/1456` | `1456/1456` |
| counterfactual oracle | `1456/1456` | `0/1456` |

The implementation must recompute these exact control counts before training
and at every snapshot use the same shuffled tuples and disagreement mask for
the model. This supplies positive and negative sensitivity controls.

The deterministic finite population receives no sampling confidence interval.
No p-value or post-hoc checkpoint/metric selection is allowed. The composite
decision is one conjunction, so there is no multiplicity claim.

## 8. Preregistered gates at raw update 2,048

Let denominators be the preregistered changed-row counts above.

- `TRAIN_FIT`: train raw full exact `>= 0.50` (`>= 363/725`).
- `GEN_CHANGED`: held-out raw changed exact `>= 0.20` (`>= 146/726`) and
  strictly above background control.
- `GEN_FULL`: held-out raw full exact `>= 0.10` (`>= 73/726`) and strictly
  above background full exact.
- `GEN_FIT = GEN_CHANGED AND GEN_FULL`.
- `GROUP_AR`: at least `2/8` held-out factual groups pass AR.
- `CF_ACTION`: held-out counterfactual-target accuracy is `>= 0.50`, strictly
  greater than held-out factual-target accuracy, strictly greater than copy
  counterfactual accuracy `643/1456`, and strictly greater than background
  counterfactual accuracy `571/1456`, all on the identical disagreement pixels.
- `COORD`: at least two held-out factual groups each reproduce at least two
  distinct raw full-exact changed ACTION6 target classes.
- `GEN_ACTION = GROUP_AR AND CF_ACTION AND COORD`.

These thresholds were fixed before any G checkpoint existed. They are a
screening floor, not an estimate of acceptable production or ARC quality.

## 9. Fixed-priority decision rule

Apply exactly one class at update 2,048 in this order:

1. **GENERALIZES** when `TRAIN_FIT && GEN_FIT && GEN_ACTION`.
   - Conclusion: this finite seed-5 screen supports shared function learning on
     its same-distribution held-out indices.
   - Next: preregister a matched confirmation/streaming contrast; no direct
     checkpoint or ARC promotion.
2. **FRAME_GENERALIZES_ACTION_FAIL** when `TRAIN_FIT && GEN_FIT` but
   `!GEN_ACTION`.
   - Conclusion: exact board prediction transfers at the screening floor, but
     causal action/coordinate transfer does not.
   - Next: frozen-checkpoint action-FiLM/spatial-coordinate diagnosis; do not
     retrain first.
3. **FITS_NO_GENERALIZATION** when `TRAIN_FIT && !GEN_FIT`.
   - Conclusion: the scaled fixed set fits but behaves as lookup on unseen
     indices.
   - Next: frozen representation/readout comparison of train versus held-out;
     do not change loss or extend budget first.
4. **DOES_NOT_SCALE** when `!TRAIN_FIT`.
   - Conclusion: parent P's one-batch fit does not scale to eight batches under
     the fixed budget. This is confounded by visits per batch: P had zero raw
     full-exact rows after 256 visits and `70/95` after 512, while G gives each
     train batch exactly 256 visits.
   - Next: compute `EXTENSION_SIGNAL`, defined now as train raw full-exact
     fraction improving by at least `0.10` absolute from update 1,024 to update
     2,048. If true, preregister a matched 4,096-update extension giving 512
     visits per batch. If false, preregister the matched prediction-only
     discriminator. Do not launch either automatically.

All four outcomes keep A/C/D, memory, planning, broader public ARC evaluation,
and architecture-success claims blocked. Conditional P-arm Q is permanently
deselected because parent P did not FAIL. The simple one-batch 2,048-update
extension is permanently deselected because parent P was not PARTIAL.

## 10. Required evidence and sealing

Preserve the parent/preflight bindings; this preregistration; exact command and
local timestamps; source/dependency/Cargo/binary identities; device; config and
checkpoint; ordered serialized populations and per-batch hashes; overlap and
census records; update loss log; route and update-1 evidence; raw/EMA snapshot
hashes; all train/held-out metrics and controls; lifecycle/error; runtime; PID
and GPU guards; and an external recursive SHA-256 manifest plus sidecar.

Implementation must use a new module and report schema. Parent
`positive_control.rs` behavior must remain unchanged; only narrow helper
visibility may widen. Registered implementation review must recheck zero-based
`u mod 8` selection for both main and rollout, the controller's exact one-based
step multiples of 128, SIGReg seeds exactly `5..=2052`, that held-out rows
never reach any loss/gradient/EP/EMA path, and that every
census/overlap/control count fails closed before CUDA training.

Only a complete, same-binary, manifest-verified registered root may receive a
scientific classification. Failed launches remain failed infrastructure.
Analysis must preserve the result under `ml/tofy/insights/`, update
`docs/RESULTS_P2.md`, and obtain an independent Fable 5.1 judgment before any
dependent experiment.
