# V6 split-CE coefficient-mass multiseed confirmation preregistration (2026-09-05)

## Decision and bounded claim

The seed-2 implementation screen at Tofy `96f0c449` found that equal-means
geometry with legacy coefficient mass 51 retained prediction-direction
dominance, while the same geometry with unit mass reduced prediction L2
`71.74x` and combined global/Muon cosines from `0.99464 / 0.99739` to
`0.16538 / 0.27210`. This confirmation asks whether the same attribution holds
independently at data/init seeds 3 and 4.

The bounded empirical claim is: for the Foundation-v2 synthetic V6 2x2
distribution, comparator class {CurrentDouble, EqualMeans}, and the exact
one-update budget below, legacy shared split-CE coefficient mass is sufficient
to dominate the initial combined gradient direction at each registered seed.
“Confirmed” requires the full conjunction at both seeds. It does not mean the
unit-mass objective learns better, generalizes, plans, or improves ARC-AGI.

## Mathematical premise and affected objectives

For populated changed and unchanged strata, CurrentDouble uses coefficients
`w / 1`, EqualMeans uses `(w+1)/2 / (w+1)/2`, and UnitMassBalanced uses
`0.5 / 0.5`. Thus B and C have identical class geometry and their prediction
gradients should differ by the scalar mass `w+1`; this identity is an
implementation gate, not the scientific result. The nontrivial premise is that
other objective gradients remain sufficiently non-collinear with prediction
for removing legacy mass to change the combined direction.

`split_ce_weighting` is a shared treatment: it changes both prediction CE and
the next-frame half of encoder CE. Encoder CE is therefore an explicitly
affected secondary component, not an invariant. The experiment attributes a
shared split-CE mass mechanism; it does not isolate the decoder alone.
At seed 2, the secondary change shrank a non-prediction component, so the
observed cosine reduction was not created by adding extra non-prediction mass;
it remains intervention leakage and is reported separately at every arm.

## Arms, seeds, and sequence

For each of seed/init seed 3 and seed/init seed 4:

1. **A — CurrentDouble:** establish the seed-local dominance premise.
2. **B — EqualMeans:** preserve coefficient mass while changing class ratio.
3. **C — UnitMassBalanced:** keep B geometry and reduce coefficient mass to 1.

Run sequentially under the global GPU lock in order 3A, 3B, 3C, 4A, 4B, 4C.
There is no seed or checkpoint selection, replacement, averaging, or best-of
analysis. A seed is a separate confirmation replicate, not a sample chosen
after observing results.

## Fixed execution and provenance contract

- Synthetic Foundation-v2 V6 data only; no public ARC data.
- World core V6, data contract V6, recursion 2x2.
- Physical batch 128, accumulation 1, effective batch 128.
- One optimizer step; pass `--pressure-updates 1` explicitly and disable
  periodic checkpointing. Absence of a pressure sample at update 1 is an
  integrity failure.
- Same architecture, initialization algorithm, main stream, dedicated
  32-row/16-fragment rollout stream, loss weights, optimizer, WSD schedule,
  global max norm 1.0, EMA, and event population across every arm.
- The split-CE changed-budget override is unset in every arm.
- At launch, bind one reviewed, pushed, clean preregistration revision. Its
  `src` tree, `Cargo.toml`, and `Cargo.lock` must be byte-identical to the
  implementation-screen revision `96f0c449`, verified by an empty
  `git diff --stat 96f0c449 HEAD -- src Cargo.toml Cargo.lock`.
- Build a fresh locked cuDNN release binary from that bound revision. Record
  its SHA-256, the exact Candle Graph revision, Cargo.lock SHA-256, one GPU UUID
  and driver, and every full command. The binary's embedded source revision
  and runtime checkout must both equal the bound revision. The same identities
  must hold for all six arms.
- Each arm receives a never-reused root with lifecycle, launch metadata,
  evidence manifest, exact process/lock/GPU cleanup, artifact rehash, and an
  external recursive seal. Any mismatch fails closed.
- `research_claim=false`; no checkpoint is promoted.
- No representative-update profile is expected in a one-step run because the
  Foundation profile-update list is empty. Do not pass `--profile-update 0`:
  zero is invalid. Record the manifest gap as expected and do not infer runtime
  representativeness or optimizer-update magnitude.

## Within-seed population and isolation gates

Before interpreting treatment metrics for a seed:

- A/B/C population and content fingerprints must be exactly equal within each
  seed. Actual values are recorded in the campaign's machine verification,
  not in this document, which is not amended during the campaign. Each seed's
  fingerprints must differ from seed 2's `fnv1a64:478eaf5a1d639824` /
  `sha256:d02bdb06f24b8019c776c781721d84cf0bcdb386febab1c51d4ad1a6e814b0be`
  and from the other seed's; equality with either is an integrity failure. The
  report's seed field must equal 3 or 4 respectively.
- Changed pixels, unchanged pixels, and resolved changed weight must match
  exactly across arms. A and B coefficient mass must equal `w+1`; C mass must
  equal exactly 1. A changed-coefficient share must equal `w/(w+1)`; B and C
  shares must equal `0.5`. Changed and unchanged pixel counts must each be at
  least 1.
- Main/rollout/total rows must be `128 / 32 / 160`, with 16 rollout fragments.
  Action bundle must be active. Grounding must serialize zero global/AdamW/Muon
  norm and null cosines.
- For values `x,y`, relative difference means
  `|x-y| / max(|x|,|y|,1e-12)`. Every component global, AdamW, and Muon route
  norm other than prediction and encoder CE must match for B/A and C/A within
  relative `1e-5`; rollout weighted recurrent-core gradient L2 must also match
  within `1e-5`. Prediction and encoder CE are excluded because the treatment
  changes both.
- B/C prediction global-L2 ratio must equal the B/C coefficient-mass ratio
  within relative `1e-3` using the same denominator rule.
- Every reported scalar must be finite where the schema requires a number.

Any provenance, hash, fingerprint, census, row, rollout, grounding, isolation,
scalar-identity, non-finite, process, lock, or GPU-cleanup failure stops every
remaining launch and yields `failed_integrity_or_evaluation` with no treatment
interpretation.

## Preregistered scientific gates

All gates use weighted pre-clip parameter gradients from update 1 and are
evaluated independently per seed.

- **A dominance premise:** combined global and Muon cosine to prediction must
  each be at least `0.95`, and prediction global L2 divided by combined global
  L2 must be at least `0.90`. Failure is inconclusive for the mass hypothesis:
  stop before further seeds and redesign around the missing premise.
- **B mass-preserving control:** if either combined global or Muon cosine to
  prediction is below `0.95`, the class-ratio branch triggers. Stop; design a
  separate ratio experiment and make no pure-mass claim.
- **C decision:** reject if either combined global or Muon cosine to prediction
  exceeds `0.95`. C supports only if all three hold: prediction global L2 is at
  most A prediction global L2 divided by 10, global cosine is at most `0.90`,
  and Muon cosine is at most `0.95`. Any other outcome is inconclusive.
  Rejection and inconclusive outcomes both stop remaining launches.

Confirmation requires every integrity and scientific conjunct at both seeds,
without averaging. A rejection at either seed rejects. Only a complete pass at
seed 3 authorizes launching seed 4.

## Metrics, uncertainty, and interpretation boundary

Primary metrics are prediction/combined global L2, prediction-to-combined
global and Muon cosines, and B/C scalar identity. Report encoder CE as an
affected secondary component, including its norm and cosine to prediction;
also report all component route norms/cosines,
combined norm, coefficient geometry/census, clip scale, losses, and active
population counts. With two fixed deterministic seeds, no confidence interval
or post-hoc multiplicity correction is claimed; conjunction across both seeds
is the conservative decision rule.

All arms are expected to clip at max norm 1. Clip scale and identical post-clip
norm do not establish effective LR, optimizer update size, or quality. The
one-step WSD rate is not representative training. A favorable result only
confirms initialization attribution under this distribution.

## Compute budget, stop rule, and next decision

Each arm has a 15-minute wall-time cap. The maximum is six optimizer updates
and 90 minutes, though registered early stops should reduce waste. After the
last valid arm, stop for machine verification, recursive sealing, research
analysis, and Fable 5.1 judgment.

- **Both seeds pass:** preregister, but do not yet launch, a bounded matched
  multi-step A-versus-C quality screen. It must measure changed-pixel exactness,
  rollout fidelity, action use, semantic grounding, non-collapse, and planner/Q
  validity separately. Consider a unit-mass changed budget above 0.5 because
  this treatment cuts the legacy changed coefficient 100x.
- **A fails:** dominance is not universal across the registered seeds; redesign
  the screen with a justified seed set before changing the objective.
- **B triggers:** isolate ratio geometry before returning to mass.
- **C rejects:** localize decoder/encoder Jacobian pressure by route and layer.
- **Integrity failure:** repair evidence or execution only; no scientific
  conclusion and no reuse of the failed root.

No branch authorizes LR, clipping, auxiliary-weight, architecture, public-ARC,
or production-training changes.
