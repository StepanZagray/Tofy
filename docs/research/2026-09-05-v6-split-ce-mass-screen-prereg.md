# V6 split-CE coefficient-mass screen preregistration (2026-09-05)

## Decision and bounded claim

The objective-revision-5 smoke at Tofy `42bfe5b5` showed that weighted
prediction CE supplies `408.2888 / 411.9768 = 0.9910` of the combined global
gradient norm and that the combined global/Muon directions have cosine
`0.99709 / 0.99836` to prediction at seed 2 initialization. This screen asks
only whether legacy split-CE coefficient mass is sufficient to explain that
initial pressure dominance on the exact synthetic batch.

It does not ask whether any arm trains better, generalizes, plans, or improves
ARC-AGI. “Support” below authorizes multi-seed confirmation only; “reject” means
this coefficient-mass intervention is insufficient at the measured seam.

## Mathematical premise and counterexample

For changed and unchanged mean losses `L+` and `L-`, CurrentDouble computes
`w L+ + L-`, where `w = clamp((1-p)/p, 1, 50)`. Its nominal coefficient mass
is `w+1`. EqualMeans computes `(w+1)(L+ + L-)/2`; PooledPerPixel also preserves
`w+1`. They change within-loss geometry but do not remove legacy mass.

The new UnitMassBalanced arm computes `(L+ + L-)/2`, or the sole observed mean
with coefficient 1 when a stratum is empty. With a configured changed-budget
override `b`, it computes `b L+ + (1-b)L-`; total mass remains 1. A counterexample
to the hypothesis is possible: even after unit normalization, the decoder
Jacobian or alignment with other objectives could leave the combined direction
nearly collinear with prediction. Therefore the claim is empirical and local.

## Arms and single causal factor

All three arms start fresh and differ only in `split_ce_weighting`:

1. **A — CurrentDouble:** reproduction control.
2. **B — EqualMeans:** changed/unchanged ratio control while preserving mass.
3. **C — UnitMassBalanced:** equal class means with total coefficient mass 1.

The implementation also adds a zero-weight `grounding` pressure row to repair
the prior evidence-list mismatch. That reporting-only change must be identical
across all arms and has no gradient.

## Fixed population, model, and execution contract

- Synthetic Foundation-v2 V6 data only; no public ARC data.
- World core V6, data contract V6, recursion 2x2.
- Data seed 2 and initialization seed 2.
- Physical batch 128, accumulation 1, effective batch 128.
- One optimizer step; pressure update 1; periodic checkpointing disabled.
- Existing main stream, dedicated 32-row/16-fragment rollout population,
  architecture, initialization, all loss weights, optimizer, WSD schedule,
  global max norm 1.0, EMA, and event population remain fixed.
- Exact same reviewed, pushed, clean source revision, locked cuDNN release
  binary, GPU UUID, driver, and sibling revision across arms.
- Each arm gets a never-reused root, lifecycle state, exact launch metadata,
  completed process/lock cleanup, evidence verification, and recursive seal.
- All three training-population and content fingerprints must match each other.
  Arm A must also match the prior smoke fingerprints
  `fnv1a64:478eaf5a1d639824` and
  `sha256:d02bdb06f24b8019c776c781721d84cf0bcdb386febab1c51d4ad1a6e814b0be`.

The new pressure schema must record changed/unchanged pixel counts, the resolved
changed weight/share, and total split-CE coefficient mass so the intervention
is machine-checkable. CurrentDouble must remain bit-identical in a CPU tensor
test. UnitMassBalanced must have coefficient sum 1 for both populated and
single-stratum cases. Disabled grounding must serialize global/AdamW/Muon norm
zero and cosine `null`.

## Metrics and preregistered thresholds

All comparisons use weighted pre-clip parameter gradients from update 1.

- **Integrity stop:** stop without interpretation on any non-finite value,
  provenance/hash mismatch, population-fingerprint mismatch, missing component,
  nonzero grounding row, inactive rollout/action bundle, or failed cleanup.
- **A reproduction:** prediction global L2 and combined global cosine must be
  within relative `1e-3` of `408.2887573` and `0.9970859580`. Failure stops B/C
  interpretation because the comparator is not matched.
- **B ratio branch:** if EqualMeans alone lowers either combined global or Muon
  cosine below `0.95`, class ratio is already a major driver and the pure mass
  attribution is rejected. Otherwise B is the mass-preserving control.
- **C support for confirmation:** UnitMassBalanced prediction global L2 must be
  at most `41.0` (at least a tenfold reduction from A), combined global cosine
  at most `0.90`, and combined Muon cosine at most `0.95`.
- **C rejection:** if either combined global or Muon cosine remains above
  `0.95`, unit mass is insufficient to remove prediction-direction dominance.
- Values between the support and rejection regions are inconclusive.

Action bundle, rollout, encoder CE, every individual action term, global/AdamW/
Muon norms and cosines, combined norm, and clip scale remain descriptive
secondary metrics. No checkpoint or metric selects a model. There is no
multiplicity-adjusted promotion test because this is a deterministic one-seed
mechanism screen; every conjunct above is required for its named branch.

## Budget, stop rule, and next decision

Compile and unit-test before GPU use. Each arm has a 15-minute wall-time cap;
the maximum screen budget is three updates and 45 minutes, run sequentially
under the global GPU lock. Any integrity failure stops all remaining launches.
After all valid arms, stop for analysis and Fable 5.1 review.

If C meets every support threshold and B stays in the mass-preserving branch,
preregister confirmation on init/data seeds 3 and 4 before any multi-step run.
If C is rejected, localize decoder-Jacobian pressure rather than changing the
clip threshold, LR, or auxiliary weights. If B triggers its ratio branch,
design a separate ratio experiment. Pull-pressure localization remains a
separate future experiment and cannot be mixed into this screen.
