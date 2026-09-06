# Results P2

P2 is implemented as a recursive latent world-model experiment. The completed
`readiness-v2` run is recorded below as a negative diagnostic result; implementation
smoke tests must not be promoted to research results.

## Frozen fields for the first experimental run

Before inspecting a full run, record here:

- training and held-out seeds;
- curriculum lesson lengths;
- model dimensions and recursion schedule;
- physical batch size and gradient accumulation;
- optimizer, learning rate, and SIGReg/event/Q weights;
- PTRM `K` values and latent-noise sweep;
- checkpoint-selection metric;
- exact train/evaluation commands.

## V6 frozen-diagnostic seam prerequisite (2026-09-06; falsified)

The reviewed frozen-checkpoint diagnostic at pushed revision
`7907731b6fa69043089c55ddb92c573a79b2f29d` reached its unregistered CUDA
preflight with locked cuDNN binary
`sha256:22781d436df998f98213390e8770cffdbcea3f8ae53ca9080278469d8f4d3691`.
The fresh root
`v6-frozen-diagnostic-preflight-20260906T041517-CDT` reconstructed the exact G
population and completed one step-0 batch-0 train/held-out raw rescore, then
failed before its first gradient cell because the attached training-path and
chunk-32 raw-evaluator argmaxes disagreed. It took no optimizer or EMA step and
sealed after `8.138813185 s`; report SHA-256 is
`54cc1ddf07aca7e6f8ffd72cb3f216d4d1b079899f995610f807c00fbeb3cb74`
and external manifest SHA-256 is
`41705d4a9103702600523647475205301fea9383df0011b98a854e7a53b33dbb`.

This is failed integrity evidence, not a gradient or model result. The frozen
zero-disagreement rail is not weakened post hoc, and the full diagnostic and
prediction-only treatment remain blocked. A separate no-gradient seam
characterization is preregistered to distinguish same-process instability,
active input construction, batch-128 versus chunk-32 execution, and residual
same-shape train/eval differences. Two earlier roots failed before CUDA on
build-command and digest-representation provenance checks and are preserved as
failed infrastructure only.

## Foundation-v2 recipe repair smoke (2026-09-05; bounded PASS)

The exact pushed revision
`42bfe5b5c7d613f397b9e8345c659712d6eb11e2` repaired the executable
Foundation-v2 contract and passed a single-update CUDA implementation smoke on
the RTX 5060 Laptop GPU. The locked cuDNN release binary was
`sha256:5ac69bf19952de0d63b0764287b8b6a68ad31ffa973b10f275a41d8e07435b9b`.
The never-reused root
`foundation-v2-recipe-smoke-20260905T101459-CDT` completed at physical batch
128, accumulation 1, and truthful effective batch 128. It consumed 128 main
rows plus 32 dedicated rollout rows from exactly 16 adjacent fragments.
Rollout loss was `1.3553996` and its weighted recurrent-core gradient L2 was
`0.0526454`.

The route-pressure result is negative for objective balance: prediction global
L2 was `408.2888` against combined `411.9768` (ratio `0.9910`), and the combined
global/AdamW/Muon cosines to prediction were
`0.99709 / 0.99642 / 0.99836`. The complete action bundle was `19.3412` L2
(4.74% of prediction), while rollout was `0.16214` (0.040%). Clip scale was
`0.0024273`; this is a conditioning signal, not an effective-learning-rate
multiplier under AdamW/Muon. Fable 5.1 High independently judged the bounded
repair GO and selected a unit-mass split-CE falsifier.

Every evidence-manifest artifact and the final 26-file recursive seal verified;
the external seal-manifest digest is
`sha256:0d93e448d83cf246314fc852a7d4c8a48dbeb1a7770bc10cf48d36c158737752`.
The expected evidence gap is that no representative-update profile was
published. Also, the one-step WSD schedule used final LR `0.0001`, and the
pressure list omitted the named grounding row because both grounding weights
are zero. This remains an implementation smoke with `research_claim=false`,
not model evidence or an ARC result. It authorizes only the separately
preregistered split-CE pressure screen.

## V6 split-CE coefficient-mass screen (2026-09-05; support for confirmation)

The exact pushed revision
`96f0c449633b16930b840d92b6f8c4b36b26d085`, locked cuDNN release binary
`sha256:4c52ff7351e137f59d61c61b61026121ac559d07439c0e52544d4a0fb4340988`,
and RTX 5060 Laptop GPU ran three matched one-update V6 2x2 arms at data/init
seed 2. Each arm consumed 128 main plus 32 rollout rows with 16 fragments and
used identical population/content fingerprints. Every artifact hash passed;
the finalized parent recursive seal-manifest digest is
`sha256:386328081d13a5fbabd8830e28e1a43af966d9816d95a37b48c1c3825cc78ea8`.

| Arm | Split-CE coefficients | Prediction global L2 | Combined global cosine to prediction | Combined Muon cosine to prediction |
|---|---:|---:|---:|---:|
| A CurrentDouble | `50 / 1` (mass 51) | `408.288757` | `0.9970859` | `0.9983581` |
| B EqualMeans | `25.5 / 25.5` (mass 51) | `290.258240` | `0.9946433` | `0.9973892` |
| C UnitMassBalanced | `0.5 / 0.5` (mass 1) | `5.691411` | `0.1653788` | `0.2720963` |

A reproduced the prior packet. B kept both cosines above `0.95`, so changing
class ratio while preserving mass did not trigger the registered alternative
branch. C reduced prediction L2 `71.74x` versus A and passed every support gate.
The B/C prediction-norm ratio was `50.99935`, matching their exact 51x
coefficient-mass ratio. This supports legacy coefficient mass as sufficient
for initial prediction-direction dominance on the seed-2 batch.

The treatment is not prediction-only: `split_ce_weighting` also changes the
next-frame half of encoder CE. B/C scaling is mostly arithmetic; the meaningful
new-seed premise is non-collinearity of the other objectives. All arms remained
fully clipped, so clip scale is not an effective LR or update-magnitude result.
No learned-quality, planning, evaluator, or ARC outcome was measured, and unit
mass cuts the changed-pixel coefficient from 50 to 0.5. Fable 5.1 High judged
the evidence integrity-clean and authorized only a relative-gate A/B/C
confirmation at seeds 3 and 4 before any multi-step comparison.

## V6 split-CE mass multiseed confirmation (2026-09-05; CONFIRMED)

The preregistered six-arm confirmation at pushed revision
`02fc64a639997b0727680885bc52ccca8271bb18` used a fresh locked cuDNN binary
`sha256:ac2beae20f5f5a20e1046310c6ea820e7797d2326b06e8b9e8ebde0dd425e2e4`
whose source/Cargo inputs are byte-identical to implementation revision
`96f0c449`. The successful never-reused root is
`split-ce-mass-confirmation-20260905T112618-CDT`; its 150-file parent seal
manifest has digest
`sha256:d07efd41a3af4293e07900e0ac12f89bd20824551e551401ac20b2ce17273435`.

| Seed/arm | Prediction global L2 | Pred/combined share | Global cosine to prediction | Muon cosine to prediction | Result |
|---|---:|---:|---:|---:|---|
| 3A CurrentDouble | `325.6452` | `0.98008` | `0.99727` | `0.99881` | premise pass |
| 3B EqualMeans | `280.0593` | — | `0.99688` | `0.99854` | mass-control pass |
| 3C UnitMassBalanced | `5.49141` | — | `0.39275` | `0.51337` | support pass |
| 4A CurrentDouble | `411.8410` | `0.99223` | `0.99882` | `0.99923` | premise pass |
| 4B EqualMeans | `304.7730` | — | `0.99860` | `0.99888` | mass-control pass |
| 4C UnitMassBalanced | `5.97601` | — | `0.33649` | `0.46628` | support pass |

C reduced prediction L2 `59.30x / 68.92x` versus A. B/C prediction ratios
`50.99952 / 50.99937` matched coefficient mass ratio 51 within relative
`1.23e-5`; invariant route norms matched within `1.3e-7`, and rollout-core
norms matched exactly per seed. Every provenance, population, census, pressure,
hash, process, lock, and GPU-cleanup gate passed. Fable 5.1 High independently
recomputed the raw reports and returned CONFIRMED.

This confirms only that legacy shared split-CE mass is sufficient for initial
prediction-direction dominance at seeds 3 and 4. The treatment also changes
encoder next-frame CE; all arms were fully clipped; and no learned-quality,
planning, rollout-fidelity, or ARC metric was measured. An earlier root stopped
after 3A for an unknown embedded build command and is separately sealed as
failed infrastructure; no metric from it entered the confirmation.

The next prerequisite is not a two-arm treatment run. First establish an
informative CurrentDouble update budget with nonzero held-out changed exactness.
The eventual quality screen must then include a unit-mass changed-share
`50/51` control so total mass is separated from the candidate's 100x reduction
in changed-pixel coefficient.

## V6 CurrentDouble baseline floor (2026-09-05; registered FAIL)

The clean seed-5 CurrentDouble baseline at pushed preregistration/source
revision `d0468a808d0d3cd2754dc1b31e4a85ab636fcc36` completed all 2,048
updates with the locked cuDNN release binary
`sha256:aa4e2498a4f83295fe4f65be1a99ec1194f7f04e7a6a8926166d3d016b447611`
on the RTX 5060 Laptop GPU. The never-reused root
`baseline-floor-current-double-seed5-20260905T121849-CDT` passed its
population, provenance, checkpoint, profile, pressure, rollout, non-finite,
and recursive-manifest integrity gates; its 124-file external seal-manifest
digest is
`sha256:e686a783b13eadc9f099d2ba5bdce68ecc10923e686db6fc86cc9bb4c02abfc5`.

The preregistered final raw changed exact count was exactly `3/382`, below the
FAIL ceiling of four and far below the PASS floor of ten. Final composed
changed exactness and full exactness were both zero. The full final-EMA P2
evaluation used 683 synthetic episodes on the fixed seeds and found that
one-step predictions matched the background zero-control on all four
synthetic sets. Learned changed-transition latent MSE was approximately
`1.03--1.04`, versus about `0.003` for copy-forward; action shuffle,
rollout-beats-copy, non-collapse, Q, and planner gates all failed. The decoder
still reconstructed encoder latents, localizing the observed endpoint failure
to off-manifold world-core predictions rather than a dead decoder.

This is a registered negative diagnostic on synthetic data, not an ARC result
and not a proof that CurrentDouble can never work. It selected no informative
training budget, so the planned A/C/D quality comparison remained blocked.
Fable 5.1 High independently judged the evidence admissible and routed next to
frozen raw/EMA checkpoint rescoring before any new training.

## V6 CurrentDouble frozen-checkpoint trajectory (2026-09-05; diagnostic)

At pushed preregistration revision
`aaa51d6eda03ba4316038b2ad90f44a9899bc350`, twenty representation-only
rescans covered raw and EMA weights at steps
`0, 2, 256, 512, 768, 1024, 1280, 1536, 1792, 2048`. The never-reused root
`current-double-checkpoint-trajectory-20260905T141030-CDT` passed exact
endpoint replay, fixed-population identity, semantic-count, checkpoint-hash,
sidecar, finiteness, runtime, and process-cleanup gates. Its 67-file external
seal-manifest digest is
`sha256:5a476f37c88624d37fb43d500f3e8c71dfd1fcedfe7e9315cf339b18a9627930`.

The literal registered branch is `EMA_MASKING` at step 512: raw weights sit
inside both never-learned bands while EMA foreground accuracy is `0.028890`,
above the `0.01` band. This is chance-sensitive initialization residue, not
useful masked learning. No raw or EMA checkpoint beats the background control
by the registered `0.02`; the largest positive delta is only `0.002570`, about
9 of 3,502 changed pixels, and no foreground score reaches its `0.070006`
positive threshold. Raw weights move from worse than background to a
background-like decode, while learned latent MSE never approaches copy-forward.

EMA is initialized from raw weights and uses decay `0.999` without bias
correction, retaining about 77% and 60% of initialization at steps 256 and
512. Sparse checkpoints cannot reconstruct every intervening EMA update, so
this trajectory supports a lag interpretation but does not prove historical
recurrence correctness. The result has no promotion authority, uses no public
ARC data, and keeps A/C/D blocked. Fable 5.1 High independently reproduced the
integrity and literal classification; the next discriminator must avoid the
chance-sensitive foreground rule and test same-row controls plus a tiny
positive-control overfit before any longer retraining.

## V6 CurrentDouble frozen EMA-lag audit (2026-09-05; no fault detected)

The retrospective deterministic audit at pushed revision
`a94bc9e5b71561e06c697a2d565a8c6cc4671b49` used no GPU, training, evaluator,
or public ARC data. Its successful never-reused root
`current-double-ema-lag-audit-20260905T145928-CDT` completed in 0.34 seconds
and is sealed by external manifest
`sha256:29947fbbd8181fa849181487a2f03bf4b79c87faa1a1198804a8e6cf65c276bf`.
Two earlier script attempts exited before producing reports and are separately
sealed failed infrastructure; they contribute no evidence.

Step-0 raw and EMA weights were byte-identical. Saved step-2 Adam moments
allowed raw step 1 to be reconstructed for 37 tensors / 48,833 elements;
decay-0.999 F32 replay matched 48,614 elements exactly (99.5515%) with maximum
error 2 ULPs, while three wrong-decay controls matched only 22.26%, 19.05%,
and 2.24%. Six optimizer-step-zero tensors replayed all saved EMA boundaries
bit-for-bit. Seven linear-path sparse-window fits, 424 trained-tensor alias
checks, source binding, and all checkpoint manifest/schema/finiteness gates
passed. Fable 5.1 High independently recomputed every value and returned GO.

The admissible verdict is `consistent_no_ema_fault_detected`, not proof of all
historical updates. Roughly 1,790 updates are outside exact reconstruction, and
the 256-step window diagnostic assumes approximately linear raw drift. The
previous whole-model norm-ratio proposal is invalid: three correctly behaving
bias tensors violate it when raw weights return toward initialization. A dead
duplicate-key guard in the standalone parser is harmless for these Candle-
written, manifest-hashed inputs but must be fixed before script reuse.

EMA health does not change the raw no-learning result. The EMA branch is
closed; proceed to a separately preregistered raw-weight-gated, same-row 2x2
positive-control overfit after proving its target receives gradients. A/C/D
and ARC claims remain blocked.

## V6 fixed-batch positive control (2026-09-05; registered PASS)

The clean pushed source `55d3e691cbb03e8aa6a7ccea6190944fb8c75bad`
and locked cuDNN binary
`sha256:2a36dc76d0310e747d13edbad9ade71de6b66914708fc820c55ef672e6fe0e2f`
ran the registered full-objective P arm from the frozen seed-5 initialization.
The corrected same-binary two-update preflight is
`v6-fixed-batch-preflight-provenance-20260905T170403-CDT`; the registered root
is `v6-fixed-batch-p-registered-20260905T170914-CDT`. It completed 1,024
updates in 1,145.93 seconds on the RTX 5060 Laptop GPU. All 27 finalized files
verify under external manifest
`sha256:9867affe63584b5736c81d41833ab8259509a13fc056e88ae3036c2c65d7908f`.

| Raw snapshot | Changed exact | Full-board exact | Exact shared-frame outcome classes |
|---:|---:|---:|---:|
| 512 | `91/95` | `70/95` | `0` |
| 768 | `91/95` | `77/95` | `2` |
| 1,024 | `91/95` | `81/95` | `4` |

BF(768), BF(1024), MONO, AR(768), and AR(1024) all pass, yielding the
preregistered class `same_row_action_conditioned_fit`. All nine required
gradient routes were finite and positive before and after clipping, all three
intentionally bypassed routes were zero, and every frozen update-1 production
binding passed. Fable 5.1 High independently recomputed the artifact identity,
manifest, census, update log, and gates and returned PASS VERIFIED.

This is a one-seed, one-fixed-batch memorization result, not broad model
quality. The four final exact branches are action ids 1, 2, 5, and 7; none of
four ACTION6 coordinate branches is exact. On all 27 outcome-changing shuffled
ACTION5/ACTION6 tuples, predictions follow the counterfactual target on only
`13/111` disagreement pixels (`11.7%`) while retaining the factual target on
`89/111` (`80.2%`). The result therefore proves that the full objective and
discrete action path can fit directly supervised rows, but does not establish
counterfactual action rules, coordinate conditioning, held-out transfer,
checkpoint promotion, or ARC readiness.

Exact registered command:

```bash
/home/stepan/Coding/Personal/.tofy-build/target-fixed-batch-55d3e691-provenance/release/tofy p2-v6-fixed-batch-positive-control \
  --checkpoint /home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/checkpoints/step-000000000000/model.safetensors \
  --train-config /home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/config.json \
  --device cuda \
  --output-root /home/stepan/Coding/Personal/.tofy-build/v6-fixed-batch-p-registered-20260905T170914-CDT \
  --max-updates 1024 \
  --registered \
  --preflight-report /home/stepan/Coding/Personal/.tofy-build/v6-fixed-batch-preflight-provenance-20260905T170403-CDT/report.json
```

This selection-only diagnostic does not update “Best So Far.” Its PASS
deselects conditional Q and the simple one-batch 2,048-update extension. The
next allowed experiment is the separately preregistered eight-train/eight-index-
held-out function-learning screen; A/C/D and public ARC remain blocked.

## V6 multi-batch generalization screen (2026-09-06; DOES_NOT_SCALE)

The clean pushed source `dba110de8ed467b58dbaa2a936565f0dc8a7b794`
and locked cuDNN binary
`sha256:faf6fb74820582de8f1a62675392de19a729a9961a2573fc8bf8c96358631f44`
ran the registered eight-train/eight-index-held-out screen from the same seed-5
initialization. The same-binary eight-update preflight root is
`v6-multibatch-preflight-20260906T001650-CDT`; registered G is
`v6-multibatch-g-registered-20260906T002436-CDT`. G completed 2,048 finite
updates—256 visits per train batch—in 2,596.29 seconds wall time on the RTX
5060 Laptop GPU. Its report SHA-256 is
`03f645a5cccfbd4dcf72bf9927ac15589dc8fa579c6330f254101ed70c18789f`,
identity is
`sha256:1d470fc1b5680e33efa07e32c51bf74fa0a73bea6e839828da09fd7922eee265`,
and all 49 finalized files verify under external manifest
`sha256:900488b44e0f1623513234839f0f777d2c1d052ec3c46458fb714383868748d4`.

| Raw update-2,048 gate | Measured | Required | Pass |
|---|---:|---:|:---:|
| Train full exact | `2/725` | `>=363/725` | no |
| Held-out changed exact | `3/726` | `>=146/726` and above background `3` | no |
| Held-out full exact | `0/726` | `>=73/726` and above background `0` | no |
| Held-out action-routed groups | `0/8` | `>=2/8` | no |
| Held-out counterfactual pixels | `507/1456` | `>=728`, above factual `472`, copy `643`, and background `571` | no |
| Held-out coordinate groups | `0` | `>=2` | no |

Train changed-pixel exactness rose from `4/725` at step 0 to `709/725`, and
the train false-edit rate fell from `0.9407` to `0.00456`. That apparent
pixel-level fit still left 19,100 false-edit pixels across 1,022/1,024 train
rows, so only two changed boards were fully exact. Held-out changed exact
ended at `3/726`, exactly the background control; held-out full exact remained
zero. The frozen extension signal was also false: train full exact moved only
`0` to `2` between updates 1,024 and 2,048, below the preregistered `+73`
threshold. The fixed-priority verdict is therefore `DOES_NOT_SCALE`.

This literal verdict is visit-confounded: the extension window gives each
batch only 128 to 256 visits, while parent P also had zero full-exact rows at
256 visits and reached `70/95` only after 512. It binds the registered branch
but does not establish that a 512-visit extension would fail. The global LR
schedules also differ, the EP controller observes only batch 7, and this is a
single seed with deterministic same-distribution-by-index held-out data.
Fable 5.1 High independently rehashed and recomputed the artifact and returned
GO under review SHA-256
`0a70421897844ed80ad40dada18c80d1087fc638990281e84a9d7399ee8b89ed`.

The next allowed branch is frozen-checkpoint gradient/false-edit/per-batch
diagnosis followed by a separately preregistered matched eight-batch
prediction-only discriminator. No checkpoint, generalization, ARC, or
level-completion claim is promoted, and “Best So Far” is unchanged.

Exact registered command:

```bash
/home/stepan/Coding/Personal/.tofy-build/target-v6-multibatch-dba110de/release/tofy p2-v6-multibatch-generalization-screen \
  --checkpoint /home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/checkpoints/step-000000000000/model.safetensors \
  --train-config /home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT/config.json \
  --device cuda \
  --output-root /home/stepan/Coding/Personal/.tofy-build/v6-multibatch-g-registered-20260906T002436-CDT \
  --max-updates 2048 \
  --registered \
  --parent-p-report /home/stepan/Coding/Personal/.tofy-build/v6-fixed-batch-p-registered-20260905T170914-CDT/report.json \
  --preflight-report /home/stepan/Coding/Personal/.tofy-build/v6-multibatch-preflight-20260906T001650-CDT/report.json
```

## V6 2x2 E2 recipe-contract audit (2026-09-05; integrity invalid)

The independently sealed E2 root at Tofy `dadc3e5f` is no longer admissible as
the registered effective-batch-256 experiment. Its Foundation-v2 config records
physical batch 128 and `grad_accum=2`, but the specialized loop made one
128-row batch, one backward, and one optimizer step per update; it never entered
the generic accumulation loop. The report's 524,288 consumed rows are exactly
`4096 * 128`, rather than `4096 * 128 * 2`. The artifact's effective-batch-256
identity is therefore false even though its files and evaluator replay remain
internally useful.

The same source made rollout activation impossible: V6 allocated at most 15
sequential rows in a physical-128 mixed batch, but the objective required 16
distinct adjacent fragments. The observed mean rollout loss of `0.0` was inert
by construction. Consequently E2's K16-K0 delta `0.0`, changed exactness, loss,
and clipping telemetry are exploratory checkpoint diagnostics, not a valid
registered E2 FAIL. E2G and any production-style retrain are paused pending the
preregistered Foundation-v2 contract repair.

This audit does not invalidate the later local LP85 replay as a statement about
the frozen checkpoint's behavior: that checkpoint did complete level 1 again.
It does invalidate any inference that the checkpoint was trained with effective
batch 256 or with an active horizon-2 rollout objective.

## V6 2x2 E2F registered semantic confirmation (2026-09-05; PASS)

E2F passed its frozen single-pair semantic confirmation. It reused the exact
deterministic-backward E2E binary from pushed Tofy
`e2fb4b66c270f154d8f2c2ed81b561b5ca5974e8`, binary
`sha256:4bda8befc5e492031833c779a3fbc1f65a4652b16875ca24e88c3e3e3ae9114f`,
and bound the sealed E2E preflight under manifest
`sha256:e9bc531a58bab2908b4cad6150652b48a15de10314d9d7cf971a68c587a4bf5d`.
The registered root `v6-e2f-registered-20260905T071154-CDT` completed all 256
updates per arm in 72.02 s on NVIDIA GeForce RTX 5060 Laptop GPU
`GPU-216be468-8184-1801-0563-7c67555dbc45`. Its report is
`sha256:968d05d9dc15e85fb3179d26d9575b6f0dad6da9f22910e33eccd69432334a27`
and its recursively verified external manifest is
`sha256:c82d2e7e1c7b0be0fc008ef53ef8a8375aef0e05d4e51da22283f1a58627be0a`.
No public ARC data was read and no arm weights were saved.

| Checkpoint | D correct A/B | D swapped | Interaction A/B | L1 correct A/B | L1 swapped | Exact correct own/paired/K0 | Exact swapped own/paired/K0 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | +0.000329548 | -0.000068247 | 0.000397795 | 0.000444106 | 0.000145469 | 28/28/28 | 28/28/28 | fail |
| 128 | +13.5590 | -10.9841 | 24.5431 | 1.999877 | 1.999046 | 56/0/28 | 0/56/28 | pass |
| 256 | +15.2549 | -15.8516 | 31.1066 | 1.999990 | 1.999986 | 56/0/28 | 0/56/28 | pass |

The complete conjunction held at consecutive fixed checkpoints 128 and 256:
both independently trained correct replicas predicted all 56 disagreement
pixels with their own histories and none with the paired histories, while the
swapped arm exactly reversed that behavior. All same-shape and cross-shape
semantic K0 invariants passed at every checkpoint. Correct A/B update records,
parameter digests, and complete checkpoints remained bit-identical through
update 256; checkpoint-0 evaluator replay and exact preflight-versus-registered
evaluator/optimizer parity also passed.

This is the first integrity-clean evidence that the repaired 2x2 path can learn
and causally route a second, harder synthetic twin pair, rather than merely
reacting to a context-shaped bias. Its scope is deliberately narrow: one fixed
pair, one initialization, one GPU, and a direct pair-specific overfit objective.
It is not held-out population generalization, a promoted world model, planning,
or ARC-AGI level completion, and it does not update “Best So Far.” PASS
authorizes only the separately preregistered simultaneous multi-pair synthetic
screen.

## V6 2x2 E2E deterministic-backward premise (2026-09-05; implementation smoke)

E2E changed only vendored Candle's cuDNN convolution backward-filter and
backward-data choices from unfiltered fastest-heuristic selection to the
documented deterministic algorithm-1 variants; forward convolution and all
model, data, evaluator, loss, optimizer, ordering, and budget fields remained
unchanged. The reviewed, pushed Tofy revision was
`e2fb4b66c270f154d8f2c2ed81b561b5ca5974e8`, the exact locked cuDNN release
binary was
`sha256:4bda8befc5e492031833c779a3fbc1f65a4652b16875ca24e88c3e3e3ae9114f`,
and the dynamically linked cuDNN runtime was 9.25.0. The run used the same
NVIDIA GeForce RTX 5060 Laptop GPU
`GPU-216be468-8184-1801-0563-7c67555dbc45` as E2D.

The preregistered 8-update run,
`v6-e2e-preflight-20260905T070253-CDT`, completed in 5.43 s. Its report is
`sha256:68a74d7d2078ca35afc0e67861c542b783ea3b972d20b4ce574db9cf927a35aa`
and its recursively verified external manifest is
`sha256:e9bc531a58bab2908b4cad6150652b48a15de10314d9d7cf971a68c587a4bf5d`.
Both correct replicas matched bit-for-bit at checkpoint 0 and checkpoint 8 in
every update record, parameter digest, and complete checkpoint field. At
update 1 both recorded loss `2.607574462890625`, pre-clip norm
`1.6909115314483643`, clip scale `0.59139699588153`, and context-gradient norm
`0.1568803931726758`; at update 8 those fields were respectively
`1.9157449007034302`, `3.0478193759918213`, `0.3281034328599542`, and
`0.780906425882186`, with shared parameter digest
`sha256:e4c4f02a2ac644faf8841d095e9d56f560d49955ca21fb6f84178d5fd128a994`.
Fixed selection, E2W legacy parity, in-process evaluator replay, semantic K0
batch invariance, and the finite K0 bound also passed.

This supports the narrow causal claim that the pinned backward algorithms
remove the observed within-process E2D replica divergence for this exact
8-update workload. It does not prove global CUDA determinism or improve a
model: E2E was an unregistered infrastructure smoke, its update-8 model gates
were false by design, and its outcome cannot satisfy E2D. No public ARC data
was read and no weights were saved. Its PASS authorizes only the freshly
preregistered E2F registered semantic confirmation below.

## V6 2x2 E2D GPU replica-integrity failure (2026-09-05; failed preflight)

E2D produced no model verdict and no registered run was launched. A first
provenance-only attempt, `v6-e2d-preflight-20260905T063254-CDT`, failed before
opening CUDA because its detached build recorded `source_pushed: null`. It used
Tofy `559edf9997949b9eeb5dcac947e7cd057ac45eba`, binary
`sha256:d9fd55904c6183becf4b9255c5a37a7db7d3907bd287d2f1fefbc595d3af25cf`,
report
`sha256:66ac470e40e7a6fdc2d15e3824f3b89d17abd9855abebdd6d7b34fde96fae42e`,
and external manifest
`sha256:35702c3f83668443db015bd7ea31b2dff137517ff49b50b60e5d70f0e3a0c1b0`.
This root is failed infrastructure evidence only.

The valid exact-binary CUDA preflight,
`v6-e2d-preflight-20260905T063615-CDT`, used the same pushed, clean Tofy
revision, candle_graph `8e012f25e38f0c597c14268f0c705e504a5b5c28`, binary
`sha256:c5c46a5bcce87f02434224de103a8512036bf0beaab8ba6ef185611c01041b31`,
and NVIDIA GeForce RTX 5060 Laptop GPU
`GPU-216be468-8184-1801-0563-7c67555dbc45`. It stopped after 8 updates in
5.27 s and is sealed by external manifest
`sha256:cc8d40651aec5f2b3254961c41fe6ed3feb9821cf775d7bc55f4bafef1f4c660`;
the report digest is
`sha256:83a6de7b3dc8e76f32632c2d18133e69097f81130271fad30ac9dd4d390db318`.
Both recursive manifests and report sidecars verified. No public ARC data was
read and no arm weights were saved.

Checkpoint 0 passed every repaired evaluator premise: the fixed second twin
pair matched, sealed E2W legacy fields were bit-identical, all three arms had
identical initialization and canonical parameter order, and the repeated
complete evaluator digest was exactly
`24ba590629f904fd0c40a01a10e414fcc668124f2eb70a4e18c526588862d804`.
The batch-2 duplicate rows were bit-identical in latent, probability, log,
gate, context-summary, raw-argmax, and composed-argmax fields. Each retained
batch-1 K0 decode also matched its batch-2 row on all 4,096 raw and composed
argmax labels, and the finite control passed (`K0 = 16/56 <= 28`). The observed
batch-shape NLL deltas (`2.152e-6` primary, `4.621e-6` twin raw) remained
descriptive and were not used as a gate.

The identical `correct_a` and `correct_b` replicas then diverged in the first
backward update despite equal loss (`2.607574462890625`): pre-clip gradient
norms were `1.6910316944` versus `1.6910375357`. By update 8 their losses were
`1.9128700495` versus `1.9128057957` and parameter digests differed. The
preregistered exact update/state/checkpoint gate therefore failed and the root
closed as `failed_infrastructure_or_integrity`; update-8 model fields are not
admissible evidence.

Source inspection identifies a candidate cause, not a confirmed attribution:
vendored Candle asks cuDNN heuristics to choose both backward-filter and
backward-data convolution algorithms. The installed NVIDIA cuDNN 9.25 headers
classify backward-filter algorithms 0 and 3 and backward-data algorithm 0 as
nondeterministic, while both algorithm-1 variants are deterministic. Inference
replay was exact, so forward convolution remains unchanged. A one-factor,
8-update implementation smoke must pin only both backward algorithm-1 variants
and reproduce all E2D gates before any fresh registered semantic confirmation.

## V6 2x2 E2C batch-shape integrity failure (2026-09-05; failed preflight)

E2C produced no wiring result and no registered run was launched. Its exact
CUDA preflight stopped at checkpoint 0, before any optimizer update, because
the preregistered same-row K0 NLL bit-identity check compared a batch-1 decode
with a batch-2 decode. The first failed root,
`v6-e2c-preflight-20260905T050507-CDT`, used Tofy
`5fc8008982fb40931f7f262461a138ace97d3423`, cuDNN release binary
`sha256:cc5c5ba6c846787f149f33c64dcd7a4fc199002ee54ae4ed58b91b40c5669a9e`,
and is sealed by manifest
`sha256:9338ef26e42d4b24d90f15ee54459383df7ec66fe9803d4fdeecacf992424b6f`.
It failed in 2.51 s with zero updates.

A failure-forensics-only rerun,
`v6-e2c-preflight-diagnostic-20260905T051337-CDT`, used Tofy
`18864730541323b8360b6f73aac30dfe8f9e1036`, binary
`sha256:4bb976aa130774154118c24d2408a95b16365cc4bcc756e196261a6963f4f028`,
and is sealed by manifest
`sha256:75022190ecdb765911ddaa49c48cc4a0fd76fa236089d7d8ec91b5f92320d575`.
It reproduced the same checkpoint-0 failure in 2.56 s and quantified the
batch-shape deltas:

| Direction | Raw NLL batch 1 | Raw NLL batch 2 | Absolute delta | Unimix absolute delta |
|---|---:|---:|---:|---:|
| primary | 2.523148322 | 2.523150474 | 2.152e-6 | 2.233e-6 |
| twin | 2.722665046 | 2.722660424 | 4.621e-6 | 4.633e-6 |

Every within-batch-2 K0 identity field passed: latent, raw probabilities, log
probabilities, copy gate, raw argmax, and composed argmax had zero differing
elements between the two identical query rows. The finite control also passed
(`K0 = 16/56 <= m = 28`). The sealed E2W meta-episode-1 checkpoint-0
legacy parity passed bit-for-bit in both arms. E2C selected the preregistered
meta-episode 2, position 20, with 28 disagreement pixels and mask digest
`sha256:a725cdaaf2cb9101b2987fca0fcd328c6e40a6e2deef5e883c61e3feb52a818e`.
No public ARC data was read.

This is a failed infrastructure/evaluator-integrity artifact, not
`reject_second_pair_exact_wiring_by_update_256` and not evidence that the model
cannot route context. The report's top-level `evidence_class` retains the
implementation-smoke schema value, but its lifecycle correctly classifies the
artifact as `failed_infrastructure_or_integrity`; both roots are excluded from
model evidence. Choosing a tolerance from these observed deltas would be
post-hoc, so the E2C gate is not relaxed. A freshly preregistered successor must
keep one canonical scoring batch shape and use a discrete cross-shape semantic
invariant before the three-arm comparison can resume.

## V6 2x2 E2W context-wiring diagnostic (2026-09-05; implementation smoke)

The registered E2W diagnostic used Tofy
`70ed5710d224336e019ecacb50484fe1f855f05d`, candle_graph
`8e012f25e38f0c597c14268f0c705e504a5b5c28`, and cuDNN release binary
`sha256:5c967c8eb4e50611414e272ae84f5a3065e46a209b0d45d541bd9db0324b7bdd`.
It loaded the sealed E2 step-4096 EMA, scanned the fixed 256-pair population,
and selected meta-episode 1 at position 20 with two target-disagreement pixels.
The exact-binary 8-update CUDA preflight completed in 4.41 s and is sealed by
manifest `sha256:e56e0674d7e9117e0a02b360c1a4b28b9114c6ecc7f53094eb3f36086af488fc`.
The 256-update registered run completed in 49.68 s and is sealed by manifest
`sha256:af133734e0e5d886e29427b4f8e06a8e94337e2d0ae8daeec861fe77321a403d`.
No public ARC data was read.

| Update | D correct | D swapped | Interaction | Correct L1 | Swapped L1 | Wiring | Promotion |
|---:|---:|---:|---:|---:|---:|---|---|
| 64 | +0.000234842 | -0.000284314 | 0.000519156 | 0.000234476 | 0.000283901 | pass | fail |
| 128 | +17.2984 | -10.6980 | 27.9964 | 1.999999 | 1.999527 | pass | fail |
| 256 | +20.0497 | -21.2306 | 41.2803 | 2.000000 | 2.000000 | pass | fail |

At updates 128 and 256, the correct arm scored own context `4/4`, paired
context `0/4`, and aggregate K0 `2/4`; the swapped arm exactly reversed own
and paired. Every mixed-batch K0 bit-identity check passed. This establishes
only that the repaired 2x2 path is locally trainable on the selected pair.

The registered outcome is `wiring_only_no_promotion`, so it cannot promote a
model, authorize E3, or establish planning/ARC performance. The promotion gate
failed because it required strict own-over-K0 improvement in each direction:
K0 was already `2/2` on the twin target, making `own > K0` mathematically
impossible there even though own context was perfect on both targets and paired
context was wrong on both. Aggregate own-over-K0 would pass, but it was not the
registered metric and is not substituted post hoc. The preflight and registered
update-8 trajectories also differed despite identical recorded inputs. Source
inspection found that E2W constructed AdamW variables and the global clip
reduction from randomized `VarMap` `HashMap` iteration; floating reduction order
is therefore an avoidable cross-arm/cross-process confound in addition to any
CUDA nondeterminism. A fresh confirmation must sort parameters and require
exact preflight-versus-registered update-8 parity.

```bash
./target/release/tofy p2-context-wiring \
  --checkpoint <sealed-e2-root>/checkpoints/step-000000004096/ema.safetensors \
  --train-config <sealed-e2-root>/config.json \
  --parent-root <sealed-e2-root> \
  --parent-manifest <sealed-e2-manifest> \
  --preflight-report <sealed-e2w-preflight>/report.json \
  --output-root <never-reused-e2w-root> \
  --device cuda --pairs 256 --max-updates 256 --registered
```

## SIGReg/action-conditioning A/B v1 (experimental; not a positive claim)

The controlled experiment is preregistered under
`runs/p2/ab-sigreg-action-v1/` and run by
`scripts/p2_sigreg_action_ab.sh`. Both arms use fresh initialization, physical
batch `1024`, accumulation `1`, dynamics only, shuffled episodes, randomized
recursion up to `8` outer / `2` inner steps, final-outer-only supervision, seed-fixed
data order, and checkpoints every 500 updates. Event, Q, reliability, prefix,
rollout, PTRM, ensemble, and live-policy signals are disabled or ignored for this
representation experiment. Held-out evaluation uses 64 synthetic episodes,
deterministic `K=1`, ensemble `1`, and evaluation seed `424242` at updates 1,000
and 2,000.

- `control` retains the existing RMS-normalized, 2×2 spatial-pooled cell-vector
  SIGReg path unchanged.
- `projector` globally pools the encoder's pre-RMS `B×128×8×8` features, applies
  one learned linear `128→128` projector without output normalization, and supplies
  current/next embeddings to SIGReg as `2×B×128` (`T×B×D`). A linear width-preserving
  projector is the smallest treatment that tests the audited primary-source
  constraints: the official statistic supports time×batch embeddings, while the
  LeWorldModel report says a projector was needed when final encoder normalization
  impeded the anti-collapse objective. The unchanged width avoids adding a capacity
  or bottleneck confound.

Before any result is read, representation collapse is defined as mean variance below
`1e-4` or covariance participation-ratio effective rank below `10%` of encoder
dimension. SIGReg is considered near-pinned at `>=99%` of its `10,000` smooth bound.
Action gates require both dynamics aggregate and `random_one_step` shuffled/true MSE
ratios `>=1.10` with paired-bootstrap 95% lower bounds above `1.0`. Genuinely changed
transitions must show at least 10% lower learned one-step MSE than copy-forward.
Across three seeds, the median lower confidence bound for each action ratio must
also exceed `1.0`. A candidate arm has a preregistered severe relative regression
if any paired seed has more than 25% worse horizon-8 rollout MSE, more than 25%
lower encoder variance, or more than 25% lower effective-rank fraction. Such an
arm is not promoted. Any arm with the preregistered credible monotonic approach
advances both arms to seeds 2 and 3.

```bash
# Run one arm/seed through the 1,000 and 2,000 update pauses and evaluations.
P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
P2_EXPECTED_SHA=a4cd11213e7aec91ec744012223d36b73848741c \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-tofy-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy-p2-ab/target/release/tofy \
bash scripts/p2_sigreg_action_ab.sh control 1

P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
P2_EXPECTED_SHA=a4cd11213e7aec91ec744012223d36b73848741c \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-tofy-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy-p2-ab/target/release/tofy \
bash scripts/p2_sigreg_action_ab.sh projector 1

python3 scripts/p2_ab_gate.py \
  --root /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
  --seeds 1 \
  --output-json /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/pilot-gates.json \
  --output-md /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/pilot-gates.md

# Only after branch E has been recorded and all six runs reach 4,000:
python3 scripts/p2_ab_gate.py \
  --root /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1 \
  --seeds 1 2 3 --final-update 4000 \
  --output-json /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/final-4000-gates.json \
  --output-md /workspace/Personal/Tofy/runs/p2/ab-sigreg-action-v1/final-4000-gates.md
```

### Pilot result (2026-08-08)

Both seed-1 arms completed four status-0 phases from fresh initialization at the
reviewed experiment SHA `a4cd11213e7aec91ec744012223d36b73848741c`.
Every update-1,000/update-2,000 checkpoint/report SHA-256 manifest verifies. The
control config hash is
`bcf4b3e456227f1441e680857327a02f292bd323f741d7dfefab5eb7dd819f09`;
the projector config hash is
`0994064d59d7b7b1e3bf9dc6784b7e5016142c45f74cb5d026a099fa6d77cdf2`.

| Arm | Update | Aggregate shuffled/true [95% CI] | `random_one_step` [95% CI] | Changed learned-vs-copy improvement [95% CI] | Variance | Effective-rank fraction | Hard pass |
|---|---:|---|---|---|---:|---:|---|
| control | 1,000 | 1.0396 [1.0219, 1.0607] | 1.1536 [1.0777, 1.2383] | 0.6607 [0.6123, 0.7000] | 0.002760 | 0.0140 | no |
| control | 2,000 | 1.2073 [1.1535, 1.2681] | 1.7291 [1.5048, 1.9963] | 0.4955 [0.4448, 0.5418] | 0.006791 | 0.0245 | no |
| projector | 1,000 | 1.0000 [0.9999, 1.0001] | 1.0000 [0.9998, 1.0003] | -98.4505 [-102.1645, -94.7269] | 1.50e-7 | 0.0157 | no |
| projector | 2,000 | 1.0000 [0.9999, 1.0001] | 1.0001 [0.9999, 1.0002] | -159.3740 [-171.4533, -146.9218] | 3.92e-8 | 0.0144 | no |

At update 2,000, raw/bounded SIGReg was `927.069 / 841.820` for
control and `339.291 / 326.971` for projector; neither was near the 10,000
bound. Control passed both action gates and the changed-transition gate, but
failed noncollapse (`0.0245 < 0.10` rank fraction) and deteriorated over time:
one-step MSE rose `0.0246→0.1087` and horizon-8 open-loop MSE rose
`0.106→0.752`. Its exploration and hazard-source shuffle ratios remained near
1.0. Projector variance collapsed, its action-shuffle ratios stayed at 1.0, and
its learned changed-transition MSE was about 160 times copy-forward at update
2,000; its low absolute latent MSE is therefore degenerate, not a positive
result.

The unchanged preregistered gate selected terminal branch A
(`stop_after_pilot`): neither arm hard-passed or showed the defined credible
monotonic approach, and projector did not materially improve control. Seeds 2
and 3 and the 4,000-update extension were not run. No arm is promoted, and a
full 28,672-update curriculum restart is not recommended. The smallest next
hypothesis is a fresh geometric-isolation arm: pre-RMS spatial-cell SIGReg
without global pooling or a learned projector, with all other control fields
fixed. Exact expanded phase commands and immutable reports live under
`runs/p2/ab-sigreg-action-v1/`; the decision is in `pilot-gates.{json,md}`.

### SIGReg geometry-isolation A/B v2 preregistration and completed pilot

The next pilot uses fresh initialization for both arms at one reviewed Git SHA.
It retains the v1 seed-1 dynamics-only schedule, physical batch `1024`, accumulation
`1`, row cap `32768`, seed-fixed shuffled episodes, randomized `8×2` recursion,
final-outer-only supervision, update-1,000/update-2,000 pauses, evaluation seed
`424242`, and all v1 action/noncollapse gates.

- `control` remains post-RMS, 2×2-pooled spatial-cell SIGReg: all
  `2×1024×4×4 = 32768` rows are used.
- `pre-rms-spatial` adds no parameters. Dynamics still use RMS-normalized latents;
  SIGReg uses unpooled pre-RMS encoder cells, deterministically subsampling
  `32768` rows from `2×1024×8×8 = 131072`.

The treatment intentionally isolates the proposed raw local-cell construction
from the failed global-pool/linear-projector path, but it changes both normalization
placement and pooling relative to control. Existing noncollapse gates continue to
measure globally pooled normalized encoder output, so they test downstream representation
health rather than the exact raw cell population regularized by the treatment.

Run seed 1 only, then apply the unchanged pilot decision. Stop after 2,000 when
neither arm hard-passes nor shows the preregistered credible monotonic approach;
otherwise replicate both arms at seeds 2 and 3. Do not extend to 4,000 or restart
the full curriculum without the subsequent three-seed gate.

```bash
P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-geometry-v2 \
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-tofy-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_sigreg_geometry_ab.sh control 1

P2_AB_ROOT=/workspace/Personal/Tofy/runs/p2/ab-sigreg-geometry-v2 \
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-tofy-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_sigreg_geometry_ab.sh pre-rms-spatial 1

python3 scripts/p2_ab_gate.py \
  --root /workspace/Personal/Tofy/runs/p2/ab-sigreg-geometry-v2 \
  --seeds 1 --treatment-arm pre-rms-spatial \
  --output-json /workspace/Personal/Tofy/runs/p2/ab-sigreg-geometry-v2/pilot-gates.json \
  --output-md /workspace/Personal/Tofy/runs/p2/ab-sigreg-geometry-v2/pilot-gates.md

# Run the arms serially and adapt through the pilot, three-seed replication,
# and preregistered extension gates:
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-tofy-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_sigreg_geometry_overnight.sh
```

These commands and any interim smoke metrics are experimental diagnostics only.
They do not set `research_claim=true`, use public ARC games, or justify a model claim.

#### Completed recovery pilot (2026-08-09)

The serialized recovery queue completed all eight seed-1 train/evaluation phases at
Tofy commit `17fbfdff`, `candle_graph` commit `c9fa15ee`, and binary SHA-256
`6540482ef354...`. Every phase exited zero, every checkpoint/report manifest
verified, and the supervisor exited zero. A local replay against the transferred
artifacts reproduced the unchanged gate decision.

| Arm | Update | Aggregate shuffled/true [95% CI] | `random_one_step` [95% CI] | Changed improvement [95% CI] | Variance | Rank fraction | Dynamics H8 | Hard pass |
|---|---:|---|---|---|---:|---:|---:|---|
| control | 1,000 | 1.0402 [1.0225, 1.0620] | 1.1547 [1.0792, 1.2378] | 0.6665 [0.6187, 0.7054] | 0.002772 | 0.01383 | 0.1094 | no |
| control | 2,000 | 1.2095 [1.1567, 1.2695] | 1.7365 [1.5164, 1.9980] | 0.4661 [0.4133, 0.5153] | 0.006128 | 0.02529 | 1.8082 | no |
| pre-RMS spatial | 1,000 | 1.0000 [0.9997, 1.0003] | 0.9999 [0.9994, 1.0005] | -49.2690 [-52.4334, -45.9926] | 2.18e-6 | 0.01128 | 0.00465 | no |
| pre-RMS spatial | 2,000 | 1.0004 [0.9992, 1.0018] | 1.0016 [0.9973, 1.0064] | -6.4256 [-6.6567, -6.1754] | 1.95e-5 | 0.00839 | 0.03124 | no |

The unchanged gate selected terminal branch A (`stop_after_pilot`). Control acquired
stronger action dependence but failed noncollapse and deteriorated sharply in
one-step and rollout error. Treatment's much smaller raw latent MSE was degenerate:
it remained action-marginalized, failed both downstream noncollapse floors, and its
learned changed-transition MSE was `7.43x` copy-forward at update 2,000. Neither arm
met the credible monotonic-approach rule; seeds 2/3 and the 4,000-update extension
were not run, and no arm is promoted.

Training used physical `1024` / accumulation `1` stably. Mean sampled training GPU
utilization was `90.46%` control and `93.06%` treatment; peak memory was 14,866 and
14,994 MiB. Evaluations remained CPU-heavy (`3.40%` and `4.16%` mean sampled GPU
utilization). Trusted synchronized candle-graph updates were `4720.71 ms` control and
`4760.75 ms` treatment, so the geometry change had negligible runtime cost. Operation,
allocation/device-memory, and Nsight evidence remain gaps.

The completed pilot, preserved A40 handoff, full metric interpretation, runtime
analysis, and next recommendations are in
[`P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md`](P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md).

#### Phase-0 repaired re-evaluation (2026-08-10)

The real-episode rollout, bounded seam-diagnostic, and at-most-once ARC action
repairs were integrated through commit `11686b82`. The four preserved geometry-v2
checkpoints were re-evaluated with `p2.eval_report.v10` in separate representation
and rollout modes. All 1,776 `p2.episode_rollout.v2` rows reconcile, all named seams
have zero non-finite rows, prior v9 scientific metrics reproduce, and a repeated
control/update-1,000 representation report is byte-identical.

The seam map localizes the dominant control bottleneck to global spatial pooling:
at update 2,000, dynamics rank fraction falls from `0.08175` at post-RMS spatial
cells to `0.02392` after pooling. The treatment is already collapsed before pooling
(`0.02639` spatial rank), then falls to `0.00844`. Recursion preserves each arm's
already-limited spatial rank rather than creating a new abrupt collapse. No metric
is promoted to "Best So Far". Full provenance, commands, tables, and validation are
in [`P2_PHASE0_REPAIR_REEVAL_2026-08-10.md`](P2_PHASE0_REPAIR_REEVAL_2026-08-10.md).

#### Phase-1B paired TC-SIGReg seed-1 pilot

The next authorized run is a serialized, fresh seed-1 comparison of unchanged
marginal SIGReg against temporally centered residual SIGReg over ordered contiguous
`W=8` windows. Both arms retain the same post-RMS, 2x2-pooled geometry, model,
optimizer schedule, and effective batch 1,024. Checkpoints and frozen evaluations
occur at updates 250, 500, 750, and 1,000. The physical/accumulation pair must come
from the largest stable maximum-depth A40 probe for the exact Git, `candle_graph`,
and binary hashes. The launcher cannot run seeds 2/3; promotion remains locked until
the update-1,000 gate is analyzed. Factual-graph/executable-world-model work is not
part of this experiment.

#### Architecture-v1 EP/TC/QQ seed-1 result (2026-08-12)

The corrected three-arm campaign completed at Tofy revision
`229052e8a18a19cf62ab31285488f83ee2cb8bcf`, `candle_graph` revision
`6b79028c72d76c8e861849842b1f0717c9e2d88a`, and binary SHA-256
`b52bdf02260d2725e6d58a48bace57ba14250f1caff615bfa223f1f2ecdbe8e3`.
All arms used seed 1, 1,000 sequential updates, physical batch `1024`, accumulation
`1`, hidden width 128, `8x2` recurrence, SIGReg weight `0.003`, held-out seed
`424242`, and 64 synthetic episodes. The A40 probe selected the largest tested
stable physical population, and all training/evaluation processes exited cleanly.

| Arm | Pooled variance | Rank fraction | Action shuffle [95% CI] | Changed improvement [95% CI] | H8 normalized | Promotion |
|---|---:|---:|---:|---:|---:|---|
| marginal EP | 0.000450 | 0.01261 | 0.999 [0.999, 0.999] | -7.095 [-7.860, -6.455] | 5.045 | rejected |
| temporal EP | 0.012123 | 0.01584 | 1.217 [1.107, 1.330] | 0.922 [0.910, 0.932] | 18.356 | rejected |
| temporal QQ | 0.014409 | 0.01616 | 1.404 [1.258, 1.550] | 0.867 [0.852, 0.884] | 19.450 | rejected |

All three passed the variance floor and produced 64/64 finite H8 rows. All failed
the same preregistered `>=0.10` effective-rank requirement: the best arm had only
`2.07/128` effective dimensions. Temporal centering therefore repaired scale but
not dimensional breadth. Marginal EP's low raw MSE was degenerate: on changed
transitions it was `8.10x` worse than literal copy-forward and shuffled actions did
not hurt prediction. Temporal EP and QQ learned real changed/action signal, but
their open-loop H8 predictions remained `18.36x` and `19.45x` copy-forward.

QQ is the most interesting negative arm: it had the strongest aggregate and
random-one-step action intervention (`1.135`, lower CI `1.076`) and retained more
spatial rank than temporal EP, while still collapsing after pooling and failing at
H8. This does not promote QQ. It rejects another statistic-only or longer-training
sweep in this shared legacy topology and moves the next falsifiable test to the
spatial-to-pooled consumer seam with explicit fixed-space/semantic grounding.

The automatic gate correctly recorded `complete_no_promotion` and skipped V3 action
and dense-horizon stages. No factual-branch, board-probe, ARC transfer, official
scorecard, H16, or multi-seed result exists for these arms. Exact artifacts are at
`runs/p2/architecture-v1-20260812T125926Z/` on the named RunPod workspace; the
durable analysis is indexed under `ml/tofy/insights/architecture-v1-geometry.md`.

#### Prior failed overnight attempt (2026-08-08/09; incomplete)

The seed-1 pilot did not reach its decision gate. Both arms trained from fresh
initialization through update 1,000 at Tofy commit `0a3f8205`, `candle_graph`
commit `c9fa15ee`, and binary SHA-256 `682dc1e9e783...`. The control completed
evaluation at updates 1,000 and 2,000. `pre-rms-spatial` aborted during its first
evaluation with CUDA device assertion exit 134 and has no held-out report or arm
manifest. There is therefore no valid A/B result, terminal branch, or promoted arm.

Control alone moved toward stronger aggregate/random-one-step action dependence at
update 2,000 (aggregate ratio `1.1817 [1.1308, 1.2405]`, random-one-step
`1.6625 [1.4453, 1.9146]`) but
still failed noncollapse (`0.02341 < 0.10` effective-rank fraction). Its true-action
MSE worsened `0.02411 -> 0.10886` and horizon-8 MSE worsened
`0.10303 -> 0.80915`; normalized H8 worsened `1.441 -> 4.452`. Because each
checkpoint's encoder defines the latent target, the raw cross-checkpoint MSE change
mixes dynamics and representation drift. This is a control diagnostic, not evidence
for or against the treatment.

The two concurrent training arms used 29,798 MiB peak and delivered only 2.36%
more aggregate update throughput than the observed single-arm phase. Concurrent
evaluation peaked at 34,754 / 46,068 MiB, so simple OOM is not supported. An
isolated small CUDA replay with the exact binary and treatment checkpoint succeeded,
making shared-device concurrency, asynchronous CUDA fault reporting, and original
batch/runtime state important discriminants, not established causes. The later glibc
heap-corruption message also keeps FFI/driver/allocator memory safety in scope. Future
arms should be serialized for isolation, and the treatment checkpoint should be
evaluated alone before any retraining.

The supervisor's fail-stop behavior left the pod idle for approximately 8h31m
between the completed control evaluation and artifact capture. Recovery queues are
therefore required for future unattended runs.

A non-preregistered local diagnostic then evaluated the exact treatment checkpoint
and exact pod binary on all 64 held-out episodes at physical batch 256. It completed
normally, which strongly disfavors a corrupt checkpoint or sample-specific rollout
fault. Its aggregate/random action ratios were `0.99995 / 0.99978`, variance was
`1.93e-6`, effective-rank fraction was `0.01204`, and learned changed-transition MSE
was 40.4x copy-forward. The treatment was therefore directionally collapsed at
update 1,000; these metrics do not replace the missing frozen L40S report or satisfy
the pilot gate.

The full forensic report, artifact inventory, timing/telemetry analysis, ranked
hypotheses, and recovery sequence are in
[`P2_OVERNIGHT_GEOMETRY_V2_ANALYSIS.md`](P2_OVERNIGHT_GEOMETRY_V2_ANALYSIS.md).

Local remediation now makes artifact validation root-authoritative and relocatable,
serializes complete arms, runs a default 64-episode/batch-1,024 CUDA preflight for
both geometries, and performs one isolated synchronous-CUDA recovery plus an exact
checksum-verified repeat before resuming a failed evaluation. Rollout failures include
source, loop mode, seed, episode, transition, and operation context. These are
implementation safeguards only; they do not change the incomplete pilot's status or
produce a model-quality result.

## Readiness training

VRAM/GPU/Muon integration landed 2026-08-05:

- **Batch:** physical `128` × grad_accum `4` (effective 512) on RTX 5060 8GB (~6.4 GiB peak). An equal-effective-batch physical/accumulation change is a trajectory migration, not exact resume, and now requires the explicit migration flag and durable label.
- **Input:** palette index tensors + `pixel_emb` (replaces 16-ch one-hot staging)
- **SIGReg:** spatial with row subsample cap `4096`
- **Optimizer:** DeepSeek-V4 hybrid — Muon on hidden conv/proj weights (≥2×2); AdamW on embeddings, auxiliary heads (`event_head`, `q_head`, …), biases
- **Pipeline:** batch prefetch on shuffled episodes + GPU-side grad clip

```bash
bash scripts/p2_readiness_train.sh run
# default output: runs/p2/readiness-v2 (override: P2_OUTPUT_DIR=...)
```

Resume checkpoint: `runs/p2/readiness-v2/checkpoints/` (see `latest.json`).

### Readiness-v2 action-conditioning diagnostic (2026-08-07)

The report-schema-v8 action shuffle holds frames, next-frame targets, and goals fixed
while deranging the complete action tuple (including ACTION6 coordinates) within each
curriculum source. A ratio `shuffled MSE / true-action MSE <= 1.1` is
action-marginalized. Every source with at least two distinct action
conditionings failed:

| Source | Shuffle ratio | Coverage note |
|--------|--------------:|---------------|
| dynamics aggregate | 1.0023 | all ACTION1..7 represented |
| `random_one_step` | 1.0078 | ACTION5=25%, ACTION6=25%; 127 distinct coordinates |
| `exploration` | 1.0001 | ACTION1..4,7; 40.2% no-op |
| `hazard_one_step` | 1.0012 | ACTION3/4 only |
| planner aggregate | 1.0005 | ACTION2=60.0% |
| `sequential` | 0.9999 | ACTION1..4 |
| `hypothesis_probe` | 1.0023 | ACTION2=62.0% |
| `p1c_falsification` | n/a | ACTION2 only; shuffling cannot intervene |
| `p1c_hard_retarget` | 1.0018 | ACTION1..4 |

This confirms action-marginalized dynamics, but does not support narrow planner
coverage as the sole cause: the deliberately broad `random_one_step` source also
fails by a wide margin. The live differentiable target-encoder collapse hypothesis
therefore moves ahead of curriculum coverage for the next A/B.

```bash
cargo run --release --features cudnn -- p2-eval \
  --checkpoint runs/p2/readiness-v2/checkpoints/step-000000028672/model.safetensors \
  --train-config runs/p2/readiness-v2/config.json \
  --seed 2 --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1 --ptrm-noise 0.1 --q-mse-threshold 0.05 --ensemble-members 1 \
  --output runs/p2/readiness-v2/eval_report_action_shuffle.json
```

### Readiness-v3 incomplete NaN run (2026-08-08)

The L40S run at commit `2ae5a3ac` used physical batch `1024` and
gradient accumulation `1`. It started at `2026-08-07T13:12:39Z` and exited at
`2026-08-08T08:11:00Z` with `Error: total is not finite: NaN`. The last durable
checkpoint is step `18,500 / 28,672` (64.5%), at sequential step `2,116 / 4,096`;
the failure occurred before the step-19,000 checkpoint. There is no completed
`train_report.json`, synthetic eval report, or ARC-AGI-3 eval report.

Every saved checkpoint reports SIGReg exactly at its `10,000` clamp, including the
first 500 steps. At step 18,500 the sequential lesson's accumulated means were
`total=34.5368`, `next_latent=0.04187`, `rollout=0.01722`, `sigreg=10000`, and
`prefix=0.01066`. The saturated SIGReg term contributed a constant `30` to total
loss and did not provide a useful training signal. The terminal component cannot be
identified from the captured error: the trainer checks `total` before its named
constituents, and multi-horizon prefix plus PTRM rank are not reported separately.

```bash
P2_DEVICE=cuda \
P2_OUTPUT_DIR=runs/p2/readiness-v3 \
P2_PHYSICAL_BATCH=1024 \
P2_GRAD_ACCUM=1 \
RAYON_NUM_THREADS=64 \
TOFY_P2_PREFETCH_WORKERS=32 \
TOFY_P2_PREFETCH_LOOKAHEAD=16 \
TOFY_P2_PREFETCH_QUEUE_DEPTH=32 \
MAX_REPAIR_ATTEMPTS=0 \
bash scripts/p2_readiness_train.sh run
```

#### Readiness-v3 repair evidence

Fast CPU regressions isolated the stability chain before resume:

- the 16-step prefix recurrence reached RMS `4083.7` and `19335.5` instead of
  remaining on the encoder's unit-RMS latent support;
- `clip_gradients_gpu` returned success for a NaN global norm, allowing a bad
  gradient to reach the optimizer;
- the hard SIGReg clamp had exactly zero gradient above `10,000`; and
- aggregate `total` was checked before its constituents, while Q surprise, PTRM rank,
  and multi-horizon prefix were not named separately.

The repair unit-normalizes every prefix prediction, uses bounded smooth losses that
retain gradients, rejects non-finite gradient norms before the optimizer, and checks
every named constituent before `total`. All four regression tests fail on the old
behavior and pass after the repair. A one-episode CUDA eval also loaded the step-18,500
checkpoint successfully (`dyn_mse=0.000574`, `plan_mse=0.000471`), confirming the
durable checkpoint itself is finite.

```bash
cargo test --lib p2::model::tests::prefix_rollout_stays_on_unit_rms_latent_support -- --exact
cargo test --lib p2::optimizer::tests::gradient_clip_rejects_non_finite_norm -- --exact
cargo test --lib p2::train::tests::sigreg_cap_retains_gradient_above_reported_limit -- --exact
cargo test --lib p2::train::tests::loss_check_reports_constituent_before_non_finite_total -- --exact
```

#### Preserved readiness-v3 pause and evaluation (2026-08-08)

After the reviewed correctness commit was ready, PID 88711 was re-resolved from
`runs/p2/readiness-v3/train.pid`; its full command, cwd, parent wrapper, and sole
GPU ownership were verified. Exactly one `SIGINT` was sent at
`2026-08-08T11:08:17Z`. The trainer finished its optimizer update and atomically
paused at step 20,557. The wrapper's post-train evaluator then completed normally
at `2026-08-08T11:18:01Z` without interference.

- Preserved checkpoint:
  `runs/p2/readiness-v3/checkpoints/step-000000020557/`.
- Preserved synthetic evaluation: `runs/p2/readiness-v3/eval_report.json` and
  `episodes.jsonl` (`research_claim=false`, no official scorecard).
- Trainer/evaluator/wrapper PIDs exited and the GPU lock was released.

This remains an inherited recovery/stability run, not a clean result, and it was
never used to initialize the SIGReg/action A/B.

### Pressure × Grounding V1 negative screen and exact-logical recovery (2026-08-14)

The seed-1 parent at revision `ccc87452a0a0cac4dd9358bc689a2d3d85691b6b`
trained all six `SIGReg dose × bundled grounding` arms for 500 updates at physical
batch `1024`, accumulation `1`. Its registered final `ScurG0` evaluation aborted with
a CUDA `index_select` device assertion and allocator corruption, so the historical
campaign remains `failed_integrity_or_infrastructure`. A separately recorded
exact-logical-endpoint retry used the registered binary/config/checkpoint, seed `424243`,
64 episodes, eval batch `256`, and diagnostic `CUDA_LAUNCH_BLOCKING=1` exited zero
and passed fresh hashes, population identity, and finite-H8 checks. The child
Grounding Mechanism V1 queue stopped on the parent status and trained zero arms.

No cell is promoted. Calibrated-SIGReg cells acquired update-500 learned-latent action
ratios `1.2389` and `1.2875` with lower confidence bounds above one, and normalized
H8 medians `0.8964` and `0.8674`. Those cells still had normalized H8 p95 `146.3`
and `140.3`, consumer-rank fractions about `0.0178`, and target-decoder MSE `0.7928`
and `0.6817` against the frozen `0.001` semantic-trust ceiling. Every predicted-board
histogram was vastly worse than literal board copy.

Initialization calibration did not control training pressure. Calibrated SIGReg
later reached `5.33–8.97×` the next-latent encoder gradient and current SIGReg reached
`1.73–4.99×`; every nonzero-SIGReg arm clipped 100% of optimizer updates, with mean
clip scales about `0.0125` and `0.038`. Bundled grounding lowered target-decoder MSE
in every matched pair but never passed the registered coarse-probe trust gate, and
its predicted/H8 effects reversed by pressure stratum. The fixed mechanism queue
must not be relaunched unchanged inside
this clipping regime. The next highest-information experiment is an all-six-checkpoint
frozen decoder audit: local ridge, fixed local MLP, and a bounded contextual decoder,
with nested validation, exact stratified permutations, and positive/marginal controls.
Its result selects between a richer state target and a prospective pressure-controlled
SIGReg experiment without pretending a small local decoder proves state absence.
Exact analysis is preserved at
`ml/tofy/insights/pressure-grounding-v1-results.md` in the research library.

### Semantic Access V1 frozen audit (2026-08-14)

The recommended six-checkpoint audit completed on the A40 at revision
`997883fe94e4b191a0b5d6f35dee6c58f92c6817`, binary SHA-256
`7d9dcc22e986d3609d32837fb9b3db930b54f80d312259e84d7a0646417ed2ab`,
under `runs/p2/semantic-access-v1-20260814T092508Z`. All six primary CUDA paths
exited successfully, every root/per-arm checksum verified, and no model weights were
updated. The common population used seed `424243`, 64 episodes per source, evaluation
batch `256`, decoder batch `4096`, 39 registered episode derangements, and an
episode-disjoint 71/37/214 train/selection/final split.

| Arm | Descriptive ridge MSE | Local MLP MSE / rank | Contextual MLP MSE / rank | Registered trust |
|---|---:|---:|---:|---|
| `S0G0` | 0.02052 | 4.26364 / 0.050 | 1.48385 / 0.025 | no |
| `S0G1` | 0.01301 | 4.35486 / 0.025 | 1.09325 / 0.025 | no |
| `ScalG0` | 0.79255 | 73.44678 / 0.875 | 6.54460 / 0.025 | no |
| `ScalG1` | 0.68159 | 71.23297 / 0.975 | 6.30179 / 1.000 | no |
| `ScurG0` | 0.88550 | 41.54501 / 1.000 | 4.30441 / 0.025 | no |
| `ScurG1` | 0.76245 | 30.58390 / 0.625 | 4.38978 / 0.050 | no |

The registered all-negative gate is valid and the preserved launcher decision is
`richer_exact_semantic_grounding`; no metric is promoted to Best So Far. Scientifically,
that string is not a causal result. Every MLP selected its maximum 40 epochs and was
far worse than ridge, while the positive control used target histograms through ridge
rather than the MLP path. The derangement ranks are not demonstrated exact permutation
p-values, and the target is coarse patch composition from the encoded true-next frame,
not predicted-next state. Descriptively, no-SIGReg ridge is 38.6–58.6 times better than
matched nonzero-SIGReg checkpoints, and grounding improves all three ridge pairs by
36.6%, 14.0%, and 13.9%.

The next experiment is evaluation-only Semantic Access V1.1 on a fresh population.
Stage B1 first qualifies a ridge-nested residual MLP with target normalization, a
same-fitting-path observable control, matched optimizer-step budgets, and fail-closed
convergence checks. Only after every selector converges does it score true-next encoder,
target-decoder transfer to predicted-next, and predicted-only decoder seams for the
coarse patch target. This is a descriptive localization stage: it has no null p-values
and cannot authorize a model-level conclusion. If it qualifies, the preregistered next
stage is the coordinate-aware exact-cell target; otherwise only the evaluator is repaired.
The campaign enforces this globally: it runs selection-only on all six checkpoints first,
and no checkpoint scores the final partition unless all six qualify. Final invocations
must reproduce their checksum-verified selection diagnostics exactly before scoring.
Together the gated stages choose among probe repair, trajectory-pressure control,
predictor alignment, exact-cell grounding, and action/rollout work before another
model-training campaign.

```bash
TOFY_BIN=/workspace/Personal/Tofy/target/semantic-access-v1/release/tofy \
P2_AUDIT_PARENT_ROOT=/workspace/Personal/Tofy/runs/p2/pressure-grounding-v1-20260813T200848Z \
P2_SEMANTIC_AUDIT_ROOT=/workspace/Personal/Tofy/runs/p2/semantic-access-v1-20260814T092508Z \
P2_EXPECTED_SHA=997883fe94e4b191a0b5d6f35dee6c58f92c6817 \
P2_EXPECTED_BINARY_SHA=7d9dcc22e986d3609d32837fb9b3db930b54f80d312259e84d7a0646417ed2ab \
P2_EXPECTED_PARENT_SHA=ccc87452a0a0cac4dd9358bc689a2d3d85691b6b \
P2_EXPECTED_PARENT_BINARY_SHA=cb1e7bded0da2fcfc645521283251db4e8fa3477eec2f57429365c88b9dacfec \
P2_SEMANTIC_EVAL_BATCH=256 P2_SEMANTIC_DECODER_BATCH=4096 \
P2_SEMANTIC_PERMUTATIONS=39 \
bash scripts/p2_semantic_access_campaign.sh
```

### Semantic Access V1.1 Stage B1 selector calibration (2026-08-14)

Stage B1 completed selection-only on the A40 at revision
`05a51f54b00fac1ede6cde4483f3be062bf74e6c`, binary SHA-256
`d8136923064bbfb7aa96c111c7ff81d307507fee9c8582d3a153b494d8f27ebe`,
under `runs/p2/semantic-access-v1_1-stage-b1-20260814T110814Z`. All 22 campaign
artifacts verify. The fresh population fingerprint was
`sha256:b3eb7968bcb9da8e7d97403e87f6492b2f545e9f3b28336b323e03ae3c207f42`.
All 12 same-path nonlinear controls qualified at step 150, with local and contextual
selection MSE `0.03893` and `0.03151` (93.0% and 94.3% reductions). This repairs the
V1 calibration ambiguity: the registered residual-MLP fitting path can learn a known
nonlinear observable.

The real route selectors were budget-censored. Only the `S0G1` contextual target fit
converged before the 1,200-step cap (selected step 275, stopped step 475). The other
23 fits did not meet the predeclared plateau rule; most semantic-pressure fits still
improved by roughly 5--9% over their final 200 optimizer steps. The global fail-closed
gate therefore recorded `selector_invalid_no_final_partition_scored`; no final row was
scored and no seam or model-level result is claimed. The negative result is about the
selector budget, not semantic accessibility.

Stage B1b changes only the maximum decoder budget from 1,200 to 4,800 optimizer steps.
It preserves the frozen model/population/splits, decoder architecture, AdamW schedule,
physical evaluation batch `256`, decoder batch `4096`, accumulation `1`, minibatch order,
25-step evaluation cadence, eight-evaluation patience, controls, and all-six-arm gate.
This post-B1 calibration uses only the selection partition; the final partition remains
sealed until every B1b selector converges.

```bash
TOFY_BIN=/workspace/Personal/Tofy/target/semantic-access-v1_1-stage-b1b/release/tofy \
P2_AUDIT_PARENT_ROOT=/workspace/Personal/Tofy/runs/p2/pressure-grounding-v1-20260813T200848Z \
P2_PREVIOUS_SEMANTIC_ROOT=/workspace/Personal/Tofy/runs/p2/semantic-access-v1-20260814T092508Z \
P2_SEMANTIC_V11_ROOT=/workspace/Personal/Tofy/runs/p2/semantic-access-v1_1-stage-b1b-<UTC> \
P2_EXPECTED_SHA=<reviewed-b1b-commit> \
P2_EXPECTED_BINARY_SHA=<reviewed-b1b-binary-sha256> \
P2_EXPECTED_PARENT_SHA=ccc87452a0a0cac4dd9358bc689a2d3d85691b6b \
P2_EXPECTED_PARENT_BINARY_SHA=cb1e7bded0da2fcfc645521283251db4e8fa3477eec2f57429365c88b9dacfec \
P2_SEMANTIC_EVAL_BATCH=256 \
bash scripts/p2_semantic_access_v11_campaign.sh
```

### Semantic Access deterministic fixed-feature coarse probe (2026-08-14)

B1b completed selection-only at revision `0a721caa78799ea413ab4cfba983ef820e618271`
under `runs/p2/semantic-access-v1_1-stage-b1b-20260814T112232Z`. Its 22 recorded
artifacts verify, all 12 nonlinear controls passed, and 7/24 real selectors converged;
the other 17 remained censored at the preregistered 4,800-step ceiling. The global gate
therefore did not access the final scoring partition. This is evidence that the iterative
MLP selector remains a measurement bottleneck, not evidence that the corresponding model
representations lack the coarse target.

The initial fixed-map control calibration ran at revision
`d445b9ff9b35850db75bb59245d96ef5f4de3802`, binary SHA-256
`9fcb16e17f6050b335f3680a54b5d4657407423e6c64ce2bde4d9d3233a58c0b`, under
`runs/p2/semantic-access-fixed-coarse-20260814T130337Z`. All artifacts verify and
no final score was accessed. Its deterministic results were identical on all six arms:
the 64-wide map reduced control error by 89.0% locally and 77.1% contextually, below the
sealed 90% gate, while the original absolute-improvement threshold exceeded the control's
entire ridge error. This is a negative evaluator-calibration result, not a model result.

The width/scale calibration then ran at revision
`0c6e5c661ce3f805b6808d4d13e7c510893b94b8`, binary SHA-256
`4a67c418431f8f6e69389402c9c779ac4de86eda37338f2fd2de00f0d18837bc`, under
`runs/p2/semantic-access-fixed-coarse-v2-20260814T131119Z`. Its manifest verifies and
no final score was accessed. Contextual control passed at 90.02%, but local control fell
to 86.47%, confirming unstable pure-ReLU random-feature approximation rather than a
capacity-only problem. It also exposed that a passing family could emit selection diagnostics
before the global two-family control gate. Those diagnostics are treated as calibration-tainted
and are not used to design the next map; the code now gates all real diagnostics globally.

The third campaign uses a deterministic evaluator: a fixed 256-wide mixed sparse ReLU and
quadratic sketch at each of three sealed seeds, a closed-form ridge
readout per seed, and an arithmetic mean of predictions with no seed selection. Local and
contextual families have the same 14,400 learned coefficients (and 6,912 fixed nonzero map
coefficients), while contextual projection compute remains larger because its input is 3x
wider. Tofy weights are frozen and no optimizer is used. The same B1b population fingerprint,
episode-disjoint split, six counterbalanced arms, local/contextual families, target/predicted
fits, transfer seam, all-arm qualification gate, and one-shot final policy are retained.
This isolates whether B1b's result was caused by optimizer/patience censoring. It does not
test whether this decoder is optimal, and it cannot by itself support a model-level claim.

Only the model-independent control was used to choose the mixed quadratic map; the accidentally
emitted contextual diagnostics are explicitly excluded. Width remains 256 and interaction scale
32 so the unchanged absolute threshold is measurable. No final metric was observed. The final
invocation is bound to the
exact selection-report SHA-256. The nonlinear control
coordinates are derived from row-local observable content rather than global row order. All
three seeds and the ensemble must remain finite; the aggregate control must meet the sealed
MSE and improvement thresholds. If any arm fails qualification, no final invocation begins.

```bash
TOFY_BIN=/workspace/Personal/Tofy/target/semantic-access-fixed-coarse/release/tofy \
P2_AUDIT_PARENT_ROOT=/workspace/Personal/Tofy/runs/p2/pressure-grounding-v1-20260813T200848Z \
P2_PREVIOUS_SEMANTIC_ROOT=/workspace/Personal/Tofy/runs/p2/semantic-access-v1_1-stage-b1b-20260814T112232Z \
P2_SEMANTIC_FIXED_ROOT=/workspace/Personal/Tofy/runs/p2/semantic-access-fixed-coarse-<UTC> \
P2_EXPECTED_SHA=<reviewed-fixed-probe-commit> \
P2_EXPECTED_BINARY_SHA=<reviewed-fixed-probe-binary-sha256> \
P2_EXPECTED_PARENT_SHA=ccc87452a0a0cac4dd9358bc689a2d3d85691b6b \
P2_EXPECTED_PARENT_BINARY_SHA=cb1e7bded0da2fcfc645521283251db4e8fa3477eec2f57429365c88b9dacfec \
P2_SEMANTIC_EVAL_BATCH=256 \
bash scripts/p2_semantic_access_fixed_campaign.sh
```

V3 ran at revision `6c4f24d702c943df2d54530be8189cdc7c11f47d` under
`runs/p2/semantic-access-fixed-coarse-v3-20260814T131829Z`. All six arms passed
the deterministic control and completed selection, but the first final invocation
failed exact selection replay before scoring. V3 is therefore selection-only and its
consumed final population is excluded.

The repaired V4 campaign completed at revision
`403eff1d8fef8aca09dfec0580f4b3e167344947`, binary SHA-256
`472d88fab4dd10551ae0450114355e3daa5c6922e5cde4f7dd2ca665a9304d78`, under
`runs/p2/semantic-access-fixed-coarse-v4-20260814T144249Z`. Root, external-input,
selection, and per-arm manifests verify. Primary and replay selection reports and
fitted states are byte-identical for all six arms. Six sealed final reports cover
`S0G1` and `ScurG1` on fresh paired population seeds 424245--424247, each with 1,408
frames and 90,112 patch rows; final data were not used to select or fit the evaluator.

The preregistered episode-macro result is a strong checkpoint-conditional negative for
adding current SIGReg (`0.003` versus `0.0`) at fixed G1, training seed 1, and update 500.
Relative to `S0G1`, `ScurG1` MSE is `60.20x` worse for the primary local true-next
route, `9.70x` worse after a predicted-latent-specific refit,
and `3.62x` worse when transferring the target fit to predicted latents. The two
contextual within-domain ratios are also adverse (`34.64x` true-next and `10.53x`
predicted refit). These orderings hold on all 966 paired seed-episodes. Contextual
target-to-predicted transfer reverses (`0.250x`), but the nonlinear evaluator makes
`S0G1` dramatically worse than its own ridge baseline on that cross-domain route;
it is treated as transfer-distribution-shift evidence, not a `ScurG1` advantage.

This frozen audit trained no Tofy weights. It covers one training initialization at
update 500, while the three seeds resample only evaluation populations. It supports a
descriptive simple effect of the tested SIGReg setting at G1, not training-seed,
checkpoint-time, mechanism-level, or model-level generalization. In particular, 100%
clipping in the nonzero-SIGReg parent arms is a plausible optimization mediator. The next
frozen test is a sealed, coordinate-aware exact-cell sentinel with new selection/final
seeds and the same three
representation routes; only after that sentinel should a multi-training-seed,
pressure-controlled SIGReg experiment be considered.

### SIGReg pressure × population-geometry V1 preregistration (2026-08-14)

The next serial A40 campaign does not claim to reproduce LeWorldModel, TC-SIGReg, or
QQWorld. It tests the Tofy-specific temporally centered QQ objective while separating
the known optimization-pressure mediator from the population construction. Every arm
uses physical batch `1024`, accumulation `1`, fixed `8×2` recursion, the same
`q_calibration` lesson and data order within training seed, 250 fresh updates, and
evaluations at updates 125 and 250.
Recent paired runs took 4.72--4.76 seconds per update on this A40, so the five
training arms alone are expected to take about 99 minutes; calibration, checkpoint
evaluation, and integrity checks extend the wall-clock runtime beyond that.

- `S0`: no SIGReg.
- `cell-high` and `global-high`: the cell and global populations at the historical
  weight `0.003`; seed 1 only, completing the nominal-dose population contrast while
  replicating the known cell-pressure condition.
- `cell-matched`: the same cell population with a per-initialization weight calibrated
  across eight independent data batches to a shared initial median SIGReg/next-latent
  encoder gradient ratio at most `0.01` and maximum `0.02`.
- `global-matched`: globally pooled frame rows with exactly the same achieved median
  gradient-pressure target and maximum bound. It is
  closer to frame-level population semantics but remains post-RMS and lacks a consumed
  BN-MLP projector, so it is not labeled paper-faithful.

Seed 1 runs the control plus the complete `population × low/current pressure` panel.
SIGReg pressure is sampled at updates 1, 124, and 249; every checkpoint, optimizer state,
trainer state, configuration, evaluation, and episode file is checksummed. If lower
pressure repairs the cell arm, the historical negative is primarily dose/clipping-mediated.
If the pressure-matched global arm beats the cell arm, population geometry is the stronger
seed-1 candidate. If both remain adverse without sustained clipping, the next change must
be a consumed pre-RMS frame embedding rather than another dose sweep. No seed-1 arm is
promoted; replication waits for the pressure gate and the registered exact-cell sentinel.

Pressure attribution is valid only when the matched arms clip on at most 25% of updates,
the current-dose arms clip on at least 75%, and the within-population difference is at
least 50 percentage points. Otherwise the geometry metrics remain descriptive and the
pressure mechanism is not identified.

```bash
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_sigreg_pressure_geometry_v1.sh
```

### SIGReg cell dose-response V1 preregistration (2026-08-14)

The completed pressure × geometry screen showed that one-update calibration does not
control the later optimization trajectory. Both nominally matched arms clipped every
update, all five arms failed effective rank and action conditioning, and the
lower-dose cell arm developed catastrophic H8 tails from update 125 to 250. The next
serial A40 campaign therefore does not make a geometry or clipping-mediation claim.
It estimates a replicated cell-TC-QQ dose response and the location of the realized
pressure/clipping transition.

Seeds 2 and 3 each train `S0` plus weights `0.00004`, `0.00008`, `0.00016`, and
`0.00032305295536180014` for 250 fresh updates, with evaluations at 125 and 250.
The arm order ascends for seed 2 and descends for seed 3. Every arm uses physical
batch `1024`, accumulation `1`, identical topology and lesson settings, and nine
pressure samples at updates `1,31,62,93,124,155,186,217,249`. Seed-local content,
rows, parameter counts, normalized configs, evaluation populations, and all checkpoint
artifacts are verified; every arm uses the same fresh evaluation seed `424248`, and
cross-seed configs must differ only in seed/output identity.

The primary analysis relates configured dose and normalized trapezoidal realized
pressure-trajectory AUC to
effective-rank fraction, changed-transition improvement, H4/H8 normalized mean and
CVaR95, fraction beating copy, action-shuffle ratio, Q balance, and board-probe trust.
Clipping is interpreted relative to same-seed S0 rather than an absolute `25%` threshold.
A low-pressure candidate must clip on at most 10 percentage points more than S0, retain
at least 75% of S0's mean clip scale, have pressure AUC/median at most `0.10`, and have
maximum sampled pressure at most `0.25`. No weight is selected unless direction and
failure mode agree across both seeds. No arm is promoted unless both seeds at both
checkpoints pass non-collapse, changed-transition improvement, action conditioning,
non-saturated balanced Q (`balanced_accuracy > 0.5`), H4/H8 normalized mean and CVaR95
at most `1`, and semantic trust. Automatic promotion remains locked pending causal
analysis. Each train is bounded to 90 minutes and each evaluation attempt to 30 minutes;
the launcher fails closed rather than automatically resuming a partially completed arm.
This experiment may select a future controller target; it cannot by itself prove
clipping mediation or paper fidelity.

The preceding five-arm campaign averaged about 52.3 minutes per arm including evaluation.
Ten arms plus six additional pressure probes per arm are expected to require roughly
8 hours 45 minutes to 9 hours on the A40.

```bash
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_sigreg_cell_dose_response_v1.sh
```

### Action-identifiability premise rescore V1 (2026-08-15)

The cell-dose campaign's action-shuffle aggregate includes source-local derangements
whose shuffled full `(id,x,y)` tuple can equal the true tuple. Those rows are valid for
artifact accounting but dilute the intervention ratio toward one. Before any new
training, the premise rescore evaluates all 20 existing `seed × dose × checkpoint`
artifacts with a dedicated `changed_conditioning_only` stratum. It includes exactly
the paired rows whose full action tuple changed, requires its `n` and
`changed_conditionings` to agree, and reports a paired bootstrap ratio interval.

This is a frozen-checkpoint evaluator experiment, not a training campaign. The new
binary must reproduce the legacy report and episode structure, all non-numeric
decisions and identity/count fields exactly, and all numeric outputs within the
registered cross-binary CUDA replay envelope after deleting only the new metric.
Checkpoint/config hashes, evaluation population, seed `424248`, synthetic episode
count, physical evaluation batch, and PTRM settings remain fixed. If the
changed-only confidence interval remains near one, the next causal experiment
crosses the selected TC-QQ dose with same-state counterfactual action separation.
If it shows real action sensitivity, the prior aggregate was confounded and no
action-loss experiment is justified yet. This rescore cannot establish semantic,
Q, rollout, or ARC optimality.

```bash
P2_ACTION_PREMISE_SOURCE_RUN=<completed-dose-run> \
P2_ACTION_PREMISE_DEVICE_SMOKE_RUN=<exact-binary-device-smoke-run> \
P2_EXPECTED_SOURCE_SHA=1aa235ceb5bd1ec8cf60a9554ed31d738f7cd96b \
P2_EXPECTED_SHA=<reviewed-sha> \
P2_EXPECTED_CANDLE_SHA=<reviewed-candle-sha> \
P2_EXPECTED_BINARY_SHA=<reviewed-binary-sha256> \
TOFY_BIN=/workspace/Personal/Tofy/target/release/tofy \
bash scripts/p2_action_premise_rescore_v1.sh
```

The first CUDA/cuDNN attempt at root
`action-premise-rescore-v1-20260815T160107Z` completed one evaluation but failed
closed before accepting a stage: the rebuilt binary reproduced the population and
report structure but not floating outputs bit-for-bit. Representative MSE drift was
sub-percent, so exact `cmp` is not a valid cross-binary CUDA reproducibility gate.
The repaired launcher instead requires identical JSON shape and non-numeric
decisions, exact registered identity/count fields and numeric paths, and per-value
numeric drift no greater than `1e-6 + r * abs(source_value)`; it records the maximum
drift for each evaluation. The exploratory calibration uses `r=10%` for the full
legacy report, `r=1%` for episodes, and the much tighter `r=0.1%` for the action
diagnostics on which the new stratum depends. Exact fields are leaves named `n`,
count/counts forms, `changed_conditionings`, seed forms, bare or suffixed ID/index
forms, `update`, `step`, `episodes`, `horizon`, `members`, `x`, or `y`. Root
`action-premise-rescore-v1-20260815T155759Z` remains
excluded as a CPU-only build failure, and `...T160107Z` remains excluded as an
integrity-preflight failure. Neither is model evidence.

Because the numeric envelope was selected after inspecting only the failed legacy
replay difference, the repaired panel is classified
`exploratory_evaluator_calibration`; it cannot satisfy a promotion gate. Any model
conclusion requires a fresh held-out evaluation-seed confirmation. The failed
`...T160107Z` root is nevertheless the exact-binary CUDA device smoke: its binary
hash matches, `cuda:0` opened, and one complete report was produced before the old
bitwise replay assertion rejected the stage.

Unrelated legacy summary fields in `metrics.jsonl` are copied from the sealed source
reports, not from the rebuilt evaluator. Only `changed_conditioning_only` comes from
the exploratory binary. This prevents known cross-build rank drift from being
misread as a treatment result while retaining a gross full-report change detector
and a tight parity gate on the action quantities used by the new metric.

The deadline-bounded repaired launch may set `P2_ACTION_PREMISE_UPDATES=250` to
evaluate the mature checkpoint for all five arms and both training seeds (10
evaluations). This scope was selected after the parity preflight failed but before
any changed-only treatment metric was inspected. It preserves the decisive final
dose-by-seed comparison while dropping the secondary update-125 trajectory question;
the run must record this scope note and cannot support a learning-trajectory claim.

## Best So Far

**Rollout dynamics (held-out synthetic, 64 episodes, eval v3):**

| Metric | Value | Run |
|--------|-------|-----|
| rollout MSE @ 8 | **0.17** | `p2-output-v11-control` |
| rollout MSE @ 4 | **0.084** | `p2-output-v11-control` |
| one-step latent MSE | **0.024** | `p2-output-v13` |
| events accuracy | 0.906 | `p2-output-v11-control` |

v11-control replayed the v8 curriculum (no exploration, `q_mse_threshold=0.25`) on
pre-v12 architecture; it restored multi-step rollout after v9/v10 Stage 1b collapse.
v13 tightened `q_mse_threshold` to 0.05 on the same architecture and improved
one-step MSE from 0.0258 to 0.0240, but rollout regressed to 0.1185 @4 and 0.3204 @8
and Q saturated. It did not replace v11 as the rollout baseline.

**Implementation hot path (2026-08-08; not a model-quality result):** direct transition-frame
packing reduced CPU `batch_from_samples` at batch 1024 to a Criterion median of **778.75 µs**
(`762.43–795.42 µs`), a measured **72.29%** improvement over the immediately preceding baseline.
The release step probe measured `1.2–1.6 ms` across sequential, random-one-step, and falsification
sources. Exact commands:

```bash
cargo bench --bench p2_hotpath -- --noplot
cargo test --release --test p2_step_profile -- --ignored --nocapture --test-threads=1
```

The full-depth CUDA capacity gate passed both falsification (K=4 PTRM) and retarget
(PTRM plus open-loop) on the RTX 5060 Laptop 8 GiB at physical **64** × accumulation **8**
(effective 512). Falsification at 128×4 OOMed, so 64×8 is the largest tested stable physical
batch for the worst branch on this machine. The gate uses hidden 128, inner/outer depth 2×8,
spatial SIGReg, active auxiliaries, and fixed worst-case recursion. After the final synchronization
and Muon changes, falsification completed in **13.47 s**. After the final shared-prefix and
batched-target change, retarget completed in **7.56 s**. Exact commands:

```bash
TOFY_VRAM_PROBE=1 TOFY_VRAM_PHYSICAL_BATCH=64 TOFY_VRAM_GRAD_ACCUM=8 \
TOFY_VRAM_LESSON=falsification \
  cargo test --release --features cudnn --test p2_vram_probe -- \
  --ignored --nocapture --test-threads=1

TOFY_VRAM_PROBE=1 TOFY_VRAM_PHYSICAL_BATCH=64 TOFY_VRAM_GRAD_ACCUM=8 \
TOFY_VRAM_LESSON=retarget \
  cargo test --release --features cudnn --test p2_vram_probe -- \
  --ignored --nocapture --test-threads=1
```

The current L40S acceptance command remains physical 512×1. Prior L40S training reached update
18,500 at physical 1024×1 and stopped on a numerical failure rather than OOM; because
falsification 512×K4 has the same aggregate PTRM trajectory batch as sequential 1024×K2, this is
strong fit evidence, but not a substitute for rerunning the current capacity gate on that card.

A final warm-update candle-graph capture used the same local 64×8 effective-batch-512 schedule.
The evidence packet was **TRUSTED** with 37/37 closed spans, one measured root, forward/backward/
optimizer coverage, 28 gradient facts, and `root_device_synchronized=true`. The trace correctly
labels its nested semantic spans as host enqueue timing; the synchronized full-update root was
**2425.29 ms**. Nsight, allocation events, and device-memory samples were not captured, so no
kernel-level attribution or VRAM high-water claim is made from this packet.

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --seed 7 --lessons dynamics --steps-per-lesson 2 \
  --physical-batch 64 --grad-accum 8 --hidden-dim 128 --action-dim 32 \
  --inner-steps 2 --outer-steps 8 --supervise-last-outer-only \
  --sigreg-spatial --sigreg-spatial-pool --sigreg-max-rows 32768 \
  --residual-y-update --warm-start-y --shuffled-episodes \
  --checkpoint-every-steps 0 --profile-update 2 \
  --output-dir /tmp/tofy-architecture-profile-final.WtuJcm
```

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 32 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --lessons dynamics,sequential,q_calibration,falsification,retarget \
  --q-mse-threshold 0.25 --checkpoint-every-steps 100 \
  --output-dir p2-output-v11-control

cargo run --release --features cudnn -- p2-eval \
  --checkpoint p2-output-v11-control/model.safetensors \
  --train-config p2-output-v11-control/config.json \
  --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --q-mse-threshold 0.25 \
  --output p2-output-v11-control/eval_report_64ep_v3.json
```

The v13 one-step result used:

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 32 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --checkpoint-every-steps 100 \
  --output-dir p2-output-v13

cargo run --release --features cudnn -- p2-eval \
  --checkpoint p2-output-v13/model.safetensors \
  --train-config p2-output-v13/config.json \
  --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --q-mse-threshold 0.05 \
  --output p2-output-v13/eval_report_64ep_v3.json
```

The superseded v12--v17 root-output experiments and their negative results are
summarized in [`P2_LEGACY_ROOT_OUTPUTS.md`](P2_LEGACY_ROOT_OUTPUTS.md). The generated
root directories were removed after archival; new run paths use `runs/p2/`.

## Implementation validation (not a result)

The CPU smoke path completed the four ordered lessons, wrote a safetensors
checkpoint/config/report plus a `candle-graph/runtime/1` trace, evaluated held-out
synthetic transitions with deterministic `K=1` and stochastic `K=2`, and imported
the trace into `candle_graph`. The smoke used physical batch `2` and gradient
accumulation `1`; it is not the required accelerator batch-capacity measurement for
the first real run.

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --steps-per-lesson 1 --physical-batch 2 \
  --hidden-dim 16 --action-dim 4 \
  --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 4 --sigreg-knots 3 \
  --output-dir /tmp/tofy-p2-final-smoke-v3

cargo run --release -- p2-eval \
  --checkpoint /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  --train-config /tmp/tofy-p2-final-smoke-v3/config.json \
  --synthetic-episodes 1 --physical-batch 2 \
  --ptrm-k 1,2 --ptrm-noise 0.1 \
  --output /tmp/tofy-p2-final-smoke-v3/eval.json

scripts/audit_p2.sh \
  /tmp/tofy-p2-final-smoke-v3/analyzer \
  /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  /tmp/tofy-p2-final-smoke-v3/runtime.json
```

All generated reports set `research_claim=false`; without `--scorecard-json` the local
evaluator leaves `official_rhae=null` and records `public_data_used_for_fitting=false`.
Pass `--scorecard-json` with a closed official scorecard to populate RHAE per
https://docs.arcprize.org/methodology .

Pause/resume was added and validated without recording a model-quality result.
The equivalence test compares an uninterrupted two-update run with a one-update
pause plus resume and requires model weights, named optimizer moments, lesson metrics,
and active sums to agree within `1e-5` relative tolerance. Candle's CPU reductions are
parallel and are not bitwise repeatable; the test name and contract now state that limitation.
Optimizer/global step and curriculum cursor remain exact. Separate checks reject a changed
training contract and a missing optimizer file. A command-line smoke paused at step 1, resumed through the `checkpoints`
directory, completed at step 2, and passed the external analyzer audit. A separate
PTY smoke sent an actual `SIGINT`; the trainer finished its in-flight update, wrote
`step-000000000012`, reported `status=Paused`, and exited with code 0. Resuming that
run directory advanced cleanly to a new `step-000000000013` bundle.

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings

cargo run -- p2-train \
  --lessons dynamics --steps-per-lesson 2 --physical-batch 2 \
  --hidden-dim 8 --action-dim 4 --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 2 --sigreg-knots 3 \
  --checkpoint-every-steps 0 --max-steps-this-run 1 \
  --output-dir /tmp/tofy-p2-resume-smoke.2y6nB5

cargo run -- p2-train \
  --lessons dynamics --steps-per-lesson 2 --physical-batch 2 \
  --hidden-dim 8 --action-dim 4 --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 2 --sigreg-knots 3 \
  --checkpoint-every-steps 0 \
  --output-dir /tmp/tofy-p2-resume-smoke.2y6nB5 \
  --resume /tmp/tofy-p2-resume-smoke.2y6nB5/checkpoints

scripts/audit_p2.sh \
  /tmp/tofy-p2-resume-smoke.2y6nB5/analyzer \
  /tmp/tofy-p2-resume-smoke.2y6nB5/model.safetensors \
  /tmp/tofy-p2-resume-smoke.2y6nB5/runtime.json
```

### Training throughput (not a quality result)

Backend choice is measurement-driven: use `--features cudnn` only while the vendored
candle patches make it faster than `--features cuda`. Stock candle cudnn regresses
conv backward; with `vendor/candle-core` (real `ConvBackwardFilter`/`Data` + skip
dead leaf input grads) it wins.

Same dynamics microbench (hidden 128, action 32, physical batch 1024, RTX 5060
Laptop, steps 11–15 steady after fused 2B encoder):

| build | steady ms/step | forward | backward |
|-------|----------------|---------|----------|
| `--features cuda` | ~327 | ~71 | ~228 |
| `--features cudnn` + patches | ~120 | ~24 | ~69 |

Layer microbench (`tests/cuda_conv_probe.rs`, encoder c1 leaf input): full fwd+bwd
~79 ms (cuda) → ~28 ms (cudnn+patches); patched vs rewrite weight-grad max abs
diff `1.4e-5`.

```bash
scripts/p2_bench_backends.sh
cargo test --release --features cudnn --test cuda_conv_probe -- --ignored --nocapture
TOFY_P2_STEP_PROFILE=5 cargo run --release --features cudnn -- p2-train \
  --device cuda --lessons dynamics --steps-per-lesson 15 \
  --physical-batch 1024 --hidden-dim 128 --action-dim 32 \
  --checkpoint-every-steps 0 --output-dir /tmp/tofy-p2-cudnn-profile
```

Perf tooling: `TOFY_P2_STEP_PROFILE=N` (phase ms), `TOFY_PERF_TRACE=path` with
`--features profiling` (Chrome/Perfetto), and NVIDIA Nsight Systems (`nsys`) for
unified CPU+GPU timelines on CUDA. See `src/perf.rs`.

## Update rule

When a P2 metric is reported, include the exact command and separate synthetic
oracle-normalized efficiency from official ARC-AGI-3 RHAE. Public ARC games are
held-out transfer evaluation and must not be used for checkpoint selection or
hyperparameter tuning.
