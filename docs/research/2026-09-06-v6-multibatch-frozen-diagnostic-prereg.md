# V6 multi-batch frozen-checkpoint diagnostic (preregistered)

Status: **implementation independently reviewed GO; no CUDA run yet**
Date: 2026-09-06 CDT
Evidence class: **selection-only single-seed frozen-checkpoint diagnostic**
Research claim: **false**
Promotion authority: **none; treatment-design selection only**
Public ARC authorization: **none**
Selected by: registered G outcome `DOES_NOT_SCALE` with
`EXTENSION_SIGNAL=false`
Parent G root:
`/home/stepan/Coding/Personal/.tofy-build/v6-multibatch-g-registered-20260906T002436-CDT`
Parent G report SHA-256:
`03f645a5cccfbd4dcf72bf9927ac15589dc8fa579c6330f254101ed70c18789f`
Parent G manifest SHA-256:
`900488b44e0f1623513234839f0f777d2c1d052ec3c46458fb714383868748d4`
Parent G identity:
`sha256:1d470fc1b5680e33efa07e32c51bf74fa0a73bea6e839828da09fd7922eee265`
Design advice: Fable 5.1 High, SHA-256
`e065f4779c77001e0c10af621d1231fe2b19fa1cc4d188b5ae2c07e95c9cfb3e`

Preregistration NO-GO review: Opus 5 High fallback, SHA-256
`369c7f8eb344d493c189e5267815dc1524f6a936c4e8cff7bf3ac09f280a5b94`

Corrected preregistration GO review: Opus 5 High fallback, SHA-256
`6e1df6e1ef5cbf351e904d8f3ca95ed20759955409742738fca0c5ba283b84a7`

Initial implementation NO-GO review: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-frozen-diagnostic-implementation-nogo-20260906.md`,
SHA-256 `a3afaae6d6632ebb830f4029d0ef87ef1bd7cd21cf0d2c262455d14977fd35b1`.
It found cross-process bit-exact float comparison and mismatched preflight D2
coverage; both require correction and a fresh implementation review before
preflight. Fable 5.1 High was attempted twice first and remained unavailable
because of its account usage limit; that failure is recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/fable-v6-frozen-diagnostic-implementation-unavailable-20260906.md`.
Its SHA-256 is
`596de3830cdbb35a7cc94f58697d9712d48673649a01d76fc84d7df175c9bd6d`.

Post-fix implementation NO-GO review: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-frozen-diagnostic-postfix-nogo-20260906.md`,
SHA-256 `e253cd81f3e60412a21c5c8fc5189ef79c49731ee953ce1ff9c563019318badc`.
It found registered-only snapshot lookups, absent step-1,024 preflight anatomy,
and invalid cross-shape margin-float admission. All three are corrected below
and required a fresh review verdict before commit/build.

Final implementation review: Opus 5 High fallback, recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-frozen-diagnostic-final-go-20260906.md`,
SHA-256 `a5d2cdbf796a22c5b5dc14c72b923e067e75ac910224e8556e8b1669356b400d`.
Verdict: GO with no blocking findings. The primary agent nevertheless corrected
its one medium fail-closed tolerance risk and four low contract weaknesses.
The bounded post-correction review is recorded at
`/home/stepan/Coding/Personal/.tofy-build/reviews/opus-v6-frozen-diagnostic-postcorrection-go-20260906.md`,
SHA-256 `6dcabc5817c3bbc464f32d887d12fba6157f3efc988c1d275d0c50074637c5b1`.
Verdict: GO; all five corrections were confirmed and no blocking regression
was found.

Fable 5.1 was requested first for the document review, but both the initial
call and required retry were rejected by its account usage limit. Opus is
identified explicitly as the fallback; its verdict is not attributed to
Fable.

This document must receive independent review, then be committed and pushed
before diagnostic implementation. The implementation, exact locked CUDA
binary, and same-binary preflight require separate review before the full
diagnostic.

## 1. Question and bounded claim

Question: on G's frozen raw trajectory, are non-prediction objectives
materially changing the full-objective gradient away from prediction CE, or
is the retained CurrentDouble prediction objective itself nearly blind to the
false edits that prevent full-board exactness?

The diagnostic hypothesis is:

> Across raw updates 1,024, 1,536, and 2,048 and all eight fixed train
> batches, either auxiliary gradients produce a registered conflict with
> prediction CE, or the false-edit contribution to prediction CE is small
> enough to make a matched `pred_ce`-only arm causally ambiguous.

This is a first-order measurement on G's own single-seed trajectory. It does
not observe the counterfactual treatment trajectory, saved AdamW/Muon optimizer
state, a 512-visits-per-batch budget, streaming data, ARC tasks, or planning.
It may forecast whether the selected arm is likely null; it cannot prove the
arm's result.

## 2. Analytic premise and counterexamples

For one frozen cell, let `g_pred` be the CurrentDouble prediction-CE gradient,
`g_aux` the exact sum of every other production-weighted gradient, and
`g_full = g_pred + g_aux`. If `||g_aux|| / ||g_full|| <= 0.10` and the
full/prediction cosine is at least 0.95 on every cell and optimizer route,
removing auxiliaries is a small first-order directional intervention. This is
a local property only: AdamW preconditioning, Muon momentum, clipping history,
and trajectory divergence can still amplify it.

Conversely, a large or opposing auxiliary gradient does not prove harmful
competition. It may be useful regularization or lie in parameters irrelevant
to the exact decoder. Therefore `AUX_CONFLICTS` means the prediction-only arm
is causally informative, not that it will improve.

A second counterexample is specific to this objective. CurrentDouble uses
`50 * mean(changed CE) + mean(unchanged CE)`. A train cell has roughly 524,000
unchanged pixels, so each false edit can have very small optimizer pressure
even after changed pixels fit. Removing auxiliaries leaves that imbalance
unchanged. The diagnostic consequently separates the contribution from
currently false-edited unchanged pixels and the changed-versus-unchanged
gradient geometry.

## 3. Frozen identities and populations

The diagnostic must bind and reverify the sealed G report, external manifest
and sidecar, all 49 entries, lifecycle, source/binary/Cargo/dependency/config/
initial-checkpoint/GPU identities, population census, and raw snapshot hashes.
No model file may be written.

The exact G source is `dba110de8ed467b58dbaa2a936565f0dc8a7b794`,
its locked cuDNN binary is
`sha256:faf6fb74820582de8f1a62675392de19a729a9961a2573fc8bf8c96358631f44`,
and its Cargo.lock is
`b3d7c2e65ee49f07e5fb8c0ba5d3e183bb839c9f0117ef5e7ff820d80bc367cc`.
Diagnostic source must descend from G, be reviewed, clean, and pushed. Its own
locked cuDNN binary, Cargo.lock, source and sibling dependency must be recorded.

Recompose the exact `MultibatchPopulation` at seed 5, physical/effective batch
`128/128`, accumulation 1, V6 data contract, train main/rollout indices 0..7,
and index-held-out main indices 8..15. Every frozen census and serialized
per-batch hash must equal G. The diagnostic may read G's serialized batches for
binding but must use the recomposed in-memory population for model input.

Raw snapshot hashes are fixed:

| Step | Raw SHA-256 | Recorded EP weight |
|---:|---|---:|
| 0 | `0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79` | `0.01` |
| 1 | `7c62689c8938a9e351468a8b56e53bb9314cbe84d351a9d948f0cb457d57a3da` | `0.01` |
| 256 | `72d4b26e00205469b1bf1f9c31eaaf2dbec4f3e61a01b929c947df3b0d473e40` | `0.1` |
| 512 | `8ad2790b3d85db4c6d3d4485ded3ddb34645dc242e8f00d64fa92538f0f8dd66` | `0.039635810803259175` |
| 1,024 | `d27eb12ad58f9b9788780f713c7cc009e816e09f92ddabf7ba3428182a9beef5` | `0.018894249544366932` |
| 1,536 | `0738ba97ae70d74d7e425b5c18c460d5a032bab740aa39f033d6a0335218c787` | `0.1` |
| 2,048 | `07ef8d1b3b79d99fad61c3fb9ae4de193ff2a2fc77ba8a4a1e5b33bcbdb70b1a` | `0.0002540643898500332` |

EMA snapshots are neither loaded nor scored.

## 4. D1: frozen gradient anatomy

The registered gradient grid is raw steps `[1024,1536,2048]` by train batch
positions `[0,1,2,3,4,5,6,7]`, exactly 24 cells. Each position uses its
lockstep main and rollout batch. At snapshot `S`, position `p` uses the
next-cycle SIGReg seed `5 + S + p` and the EP weight stored in that snapshot.
This defines a common forward objective at each frozen boundary; it does not
claim that step `S+p+1` existed when `S=2048`.

For every cell, compute but never apply these production-weighted gradients:

| Component | Weight |
|---|---:|
| `pred_ce` | `1.0` |
| `gate` | `0.5` |
| `latent` | `0.25` |
| `enc_ce` | `0.1` |
| `separation` | `0.2` |
| `pull` | `0.1` |
| `inverse_action` | `0.1` |
| `ep` | snapshot-recorded weight |
| `event` | `0.1` |
| `q` | `0.1` |
| `reliability` | `0.1` |
| dedicated rollout | `0.02` |

Also construct `g_aux` as the exact sum of every row except `pred_ce` and
`g_full = g_pred + g_aux`. The direct reference must use G's exact two-store
accumulation: `main_total.backward()` plus the separately weighted dedicated-
rollout store. For every global/AdamW/Muon route, fail unless
`||g_full_direct - g_reconstructed|| / ||g_full_direct|| <= 1e-5`; when the
reference route norm is below `1e-6`, require absolute residual norm `<=1e-6`
instead. Norm equality or cosine alone is not an adequate reconstruction test.

Capture the per-pixel prediction CE only for this diagnostic. With the same
original CurrentDouble denominators, compute:

- `g_ch`: `50 * sum(changed CE) / N_changed`;
- `g_un`: `sum(unchanged CE) / N_unchanged`;
- `g_fe`: `sum(unchanged CE * false_edit_mask) / N_unchanged`, where the mask
  is the same checkpoint's factual raw argmax error and the denominator is not
  renormalized to the number of false edits.

Require `g_pred = g_ch + g_un` under the same per-route direct-residual rule.
Require the observed CurrentDouble changed coefficient to be exactly `50.0`
in every cell before applying that identity.

For every component, `g_aux`, and `g_full`, record:

- global, AdamW-route, and Muon-route L2;
- L2 divided by full-gradient L2 on the same route;
- cosine to `g_pred` on the same route;
- `kappa = dot(g_component,g_pred) / ||g_pred||^2` on the same route;
- L2 on each of the registered nine positive and three zero parameter-prefix
  routes from parent P.

For `g_full`, also record
`cos(NS(g_full), NS(g_pred))` restricted to Muon-routed parameters. For each
parameter use the exact memoryless production pipeline
`matrix_view -> hybrid_newton_schulz -> muon_shape_rescale(0.2)`, substitute
zero momentum, flatten the transformed matrices in parameter-name order, and
then take one cosine. This excludes unavailable saved momentum and is
explicitly not an optimizer-state reconstruction. The exact grounding decoder
and copy-gate heads are AdamW-routed by the production name rule, so AdamW is
the route closest to the false-edit surface; Muon/NS-Muon describe the
recurrent core. Record global
`cos(g_fe,g_aux)`, `cos(g_fe,g_pred)`, `cos(g_ch,g_un)`, and
`share_fe = ||g_fe|| / ||g_pred||`.

The D1 false-edit mask is exactly the D2 union-sliced raw-seam argmax error for
the same checkpoint and 128 rows. Also report the number of pixels on which
the attached training-path argmax disagrees with that raw-seam argmax; this is
an integrity control and must be zero for all 24 classification cells and the
step-0 loss-path control.

The snapshot-recorded EP weight intentionally varies from `0.1` at step 1,536
to `0.0002540643898500332` at step 2,048. Therefore the eight final cells are
almost EP-off by controller state; low final auxiliary pressure is not evidence
that the production objective is generally auxiliary-free.

A cosine is undefined when either norm is below `1e-6`; preserve it as null.
Every registered zero route must exist in topology and be exactly zero in every
component and combined store. Prediction-positive routes are required at the
step-0 control cell; later near-floor cells may be zero but must be reported.

## 5. D2: exact per-batch rescoring

For all seven raw snapshots and both train and held-out sets, obtain one union
prediction through the same context-aware, chunked raw seam used by G and slice
it into the eight ordered 128-row batches. Report per batch:

- rows and changed rows;
- changed exact, full exact, and all-row exact;
- false-edit pixels, false-edit rows, and unchanged-pixel denominator;
- factual-group exact branches, reproduced distinct changed classes, and AR.

For every snapshot and split, summing the eight blocks must reproduce G's
sealed additive raw fields exactly: rows, changed rows, changed/full/all-row
exact, unchanged-pixel denominator, false-edit pixels, and false-edit rows.
Bind group-routing and ACTION6 records element-wise by factual-group index,
not by summing distinct-class counts. The global counterfactual shuffle is not
batch-aligned and is explicitly excluded from D2; G's already sealed
counterfactual metrics remain unchanged. Step 0 and step 2,048 are mandatory
positive controls, and any specified mismatch fails before classification.

## 6. D3: false-edit anatomy

At raw steps 1,024, 1,536, and 2,048, retain detached exact-gameplay logits for
train and held-out rows. For every factually unchanged pixel whose raw argmax
is wrong, define margin as `wrong_top_logit - target_logit` and record only
aggregates:

- margin bins `[0,0.5]`, `(0.5,1]`, `(1,2]`, `(2,4]`, `>4`, plus min,
  median, p90, p99, and max;
- exclusive location strata `row63`, `other_border`, and `interior`;
- target-to-predicted palette-class counts;
- whether the predicted class appears among that row's genuinely changed
  target pixels;
- Chebyshev distance to the nearest genuinely changed target pixel, with
  `none` for no changed pixels and bins `1`, `2-3`, `4-7`, `>=8`;
- per-row false-edit histogram bins `0`, `1`, `2-3`, `4-7`, `8-15`, `16-31`,
  `32-63`, `>=64`, split by changed versus no-change rows;
- among changed-exact rows, near-miss full-board counts at at most 1, 2, 4,
  and 8 false edits.

Use stable keys `(split,batch_position,row_in_batch,pixel)` to report set
intersection/union and pairwise new/resolved counts across the three snapshots.
`persistence_1536_to_2048` is the fraction of step-2,048 false-edit keys also
present at step 1,536. Also report the fraction present at all three steps.

No anatomy field selects a checkpoint or becomes a model-quality gate.

## 7. D4: loss, route, and execution bindings

Four cells are non-classifying controls:

- step-0 checkpoint / batch 0 with seed 5 and EP weight 0.01 must reproduce
  G's update-1 13 losses, rollout fragments, pre-clip norm, and prediction
  route premise;
- step-1,024 / batch 0 with seed 1,029 must reproduce loss-log step 1,025;
- step-1,536 / batch 0 with seed 1,541 must reproduce loss-log step 1,537;
- step-2,048 / batch 0 has no next G update and therefore has no loss-log
  equality claim.

Scalar tolerance is `max(1e-6, 1e-6 * abs(expected))`, except pre-clip global
norm uses relative tolerance `1e-5` plus absolute `1e-6`. Update-1 prediction
prefix norms use the same norm tolerance. Every loss, gradient, margin, norm,
cosine, kappa, and count must be finite when mathematically defined.

The exact diagnostic binary first runs an unregistered preflight containing
the step-0 and step-1,024 batch-0 control cells, batch-0 D2 rescoring at steps
0 and 1,024 so both D1 masks come from the required union-sliced seam,
step-2,048 train/held-out batch-0 rescoring, and one logit-anatomy pass. It
takes no optimizer step. The
full registered run must bind that preflight's source, binary, Cargo,
dependency, G parent, population, GPU, control values, and report identity.

The zero training-vs-raw argmax-disagreement requirement is retained after
implementation review as a cheap fail-closed preflight falsifier. Backend
batch-shape numerics could make it fail despite mathematical seam equality; if
that happens, the preflight is failed evidence and the rail will not be
weakened after observing the result. Any revised rail requires a new frozen
registration and independent review before another CUDA attempt.

Cross-process D2 controls are compared only at the preflight's frozen steps
`[0,1024,2048]`; registered-only steps are not required to appear in the
preflight. At step 1,024 the registered run additionally computes D3 anatomy,
while the preflight deliberately does not, so only the shared D2 batch-0
record is compared. At step 2,048 the 128-row preflight and 1,024-row
registered union use different encoder batch shapes. The cross-process D3
binding therefore compares the full integer anatomy (counts, bins, locations,
class pairs, distance and row histograms) but not the floating margin
quantiles. D2 integer fields still fail on any argmax/count change. Margin
floats remain required finite and are reported within each run, but a
cross-shape `~1e-5` logit drift cannot invalidate an otherwise identical
preflight under the §7 same-shape `1e-6` loss tolerance. This exclusion is
frozen before CUDA and may not be changed after observing preflight output.

Both runs require CUDA, the exact G GPU name, UUID, memory, and driver, a fresh
never-reused root, a process guard, no public data, explicit lifecycle, and a
sealed recursive manifest plus external sidecar. Input hashes are rechecked
after device work and the GPU/process guard must be clean before sealing.

## 8. Frozen thresholds and classifications

All 24 gradient cells participate. Any quotient or cosine whose denominator
norm is below `1e-6` is null. Null values fail every “negligible” conjunction
but do not themselves satisfy a conflict threshold.

| Quantity | Negligible threshold | Conflict threshold |
|---|---:|---:|
| `cos(g_full,g_pred)` on global, AdamW, Muon, and memoryless NS-Muon routes | `>=0.95` on all 24 cells | `<0.80` |
| `aux_share = ||g_aux|| / ||g_full||` globally | `<=0.10` on all 24 cells | `>0.30` |
| global `kappa(g_aux)` | `>=-0.10` on all 24 cells | `<=-0.25` |
| global `cos(g_fe,g_aux)` | `>=-0.10` on all 24 cells | `<=-0.25` |

Apply exactly one auxiliary class in fixed priority:

1. `AUX_CONFLICTS` if the union of cells satisfying at least one conflict
   threshold contains at least 4/24 cells or at least 2/8 step-2,048 cells.
2. `AUX_NEGLIGIBLE` if every negligible threshold is satisfied on all 24
   cells.
3. `AUX_MIXED` otherwise.

Compute independent flags:

- `PRED_BLIND` if `share_fe < 0.05` on all 16 cells at steps 1,536 and 2,048;
- `INTERNAL_CONFLICT` if global `cos(g_ch,g_un) <= -0.25` on at least 8/24
  cells;
- descriptive `ATTRACTOR` if train `persistence_1536_to_2048 >= 0.60` and the
  step-2,048 train false-edit median margin is at least `1.0`.

These are engineering effect sizes on a deterministic finite population, not
confidence levels. There is no seed CI, p-value, multiplicity claim, or
post-hoc subset. In particular, the previously suggested batches 0/1/3/5 are
not a decision subset because their apparent lag is known from G.

The step-2,048 auxiliary measurements receive special two-cell trigger power
even though those cells are almost EP-off. Preserve that controller-state
caveat in every interpretation; the class describes these frozen cells, not a
stationary full-objective mixture.

A conflict triggered only by memoryless NS-Muon may reflect whitening rather
than objective competition and must be labeled as such. Likewise,
`aux_share` and auxiliary kappa thresholds are global even though the exact
decoder is AdamW-routed; the required per-route ratios bound this limitation
but do not silently replace the frozen global thresholds.

## 9. Fixed-priority treatment-design decision

After an integrity-valid report:

1. If `PRED_BLIND` or `INTERNAL_CONFLICT`, the selected matched
   prediction-only arm cannot cleanly distinguish auxiliary competition from
   the retained CurrentDouble weighting. Still preregister the exact matched
   prediction-only arm required by G, but mark it causally non-decisive and do
   not launch it. Then preregister a matched changed-versus-unchanged
   prediction-weight discriminator before further training and record why the
   originally selected arm is insufficient.
2. Else if `AUX_NEGLIGIBLE`, preregister the prediction-only arm but disclose
   that it is first-order predicted near-null before asking for authority to
   spend the approximately 43-minute run.
3. Else (`AUX_CONFLICTS` or `AUX_MIXED`), preregister the exact matched
   prediction-only arm as the weakest direct discriminator. This diagnostic
   still does not authorize its automatic launch.

All three branches therefore preregister the same G-mandated prediction-only
arm and launch nothing. The diagnostic changes the immediate treatment plan
only if `PRED_BLIND` or `INTERNAL_CONFLICT` adds the weighting discriminator;
that single binary is what the device budget buys. The auxiliary class mainly
changes the causal forecast and required disclosure.

In every branch, freeze the later treatment's exact false-edit relief ratio,
full-exact threshold, visit budget, schedule, and null-result successor before
implementation. Do not revive a 4,096-update extension, A/C/D, memory,
planning, public ARC evaluation, or checkpoint promotion from this result.

## 10. Budget, implementation boundary, and sealing

Expected device work is 24 cells times component backpropagation plus seven
snapshot rescoring passes over both unions. Preflight wall cap is 5 minutes;
full diagnostic wall cap is 20 minutes. The preflight must separately time one
complete D1 cell and one train-plus-held-out D2/D3 batch pair. Estimate device
work as `24 * D1_cell_seconds + 7 * 8 * D2D3_pair_seconds`; admission requires
that estimate to be at most 900 seconds, preserving five minutes for complete
integrity checks and sealing.

The registered run also recomputes one non-classifying step-0 D4 gradient
control. The frozen admission formula intentionally counts only the 24-cell D1
classification grid; the five-minute integrity reserve covers that 25th cell,
checkpoint hashing, population verification, and sealing. This is an explicit
conservative-budget limitation, not an unreported 24-cell execution claim.

The implementation is limited to:

- a new `p2-v6-multibatch-frozen-diagnostic` module/CLI/report schema;
- one crate-visible detached-logit seam underlying the unchanged G argmax
  scorer;
- diagnostic-only optional capture of per-pixel prediction CE, false for all
  existing production callers; the captured tensor must remain attached so
  the three diagnostic sub-losses can be differentiated;
- the minimum visibility widening for existing gradient route/cosine helpers,
  plus an absent-tolerant per-component prefix-L2 helper because observer-only
  components legitimately have no entry on prediction routes. Its report type
  must distinguish a structurally absent entry from a present gradient whose
  computed norm is zero.

Registered G and parent-P execution behavior must remain bit-identical. Tests
must cover thresholds and priority, masks/denominators, histograms,
persistence, component reconstruction, undefined cosines, preflight binding,
source/G/checkpoint/census drift, report identity, manifest validation, and
the unchanged raw prediction seam. Strict all-target Clippy and focused
release tests must pass before a locked CUDA build.
