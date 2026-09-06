# Charged-action evaluator confirmation

Registered after the first exploratory screen, before any corrected-run action
on September 6, 2026 CDT. This supersedes only the action-accounting definition
and implementation acceptance in the earlier reference-screen preregistration.
The older result and preregistration remain intact.

## Trigger and evidence boundary

The c2dfa009 three-game Tofy run selected 128 non-reset actions and issued one
retry reset per game. The official arc-agi0.9.9 scorecard charged 129 per game,
387 total. Its engine score was 0 and all 23 levels were unsolved. Thus the claimed
128-environment-action cap was exceeded by 1 per game; retain that run as exploratory
with a budget deviation, not a passed registered screen. No model change follows.

## Claim, intervention and comparator

Claim: counting charged retries in the same budget as policy steps makes the
runner stop at exactly 128 engine-charged actions per game, and its report agrees
with the canonical engine count. This arithmetic premise follows if one retry
is charged as one action, confirmed by the existing scorecard and a new fixture.
A real-engine run checks whether that prerequisite holds for these versions.

Only the evaluator's counters/report reconciliation and failure-message wording
change. Model code, raw G step 2048 weights, greedy policy, seed 0, context defaults,
no adaptation, public game versions/order, and all numerical budget values remain
as previously registered. Comparator: the complete recorded prefix from
`prize-tofy-public-screen-20260906T162700-CDT` at c2dfa009. Both source revisions
and binary hashes must be captured; do not claim evaluator-independent model
improvement. The corrected run is implementation smoke, not quality promotion.

## Frozen contracts and gates

The contract now hashes action-accounting version
`engine_charged_including_retry_reset_v1` as well as population/limits.

- Three-game 128-action/180-second gameplay/600-second suite digest:
  `905416cf8ac752e34628cd3f9bed796e503cf7f5ad96fcd0313c1c61d72f8a76`.
- One-game 2-action/60-second gameplay/120-second suite smoke digest:
  `daa919af40adcd8c5ac78303629acb34df831176cc45278d76a7fecf9c967a9d`.
- Games: ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d, seed 0.
- Checkpoint SHA-256:
  `07ef8d1b3b79d99fad61c3fb9ae4de193ff2a2fc77ba8a4a1e5b33bcbdb70b1a`.

Before launch: 21 Python fixtures plus lint/compile checks; clean reviewed pushed
source; same frozen model bytes and dependency; freshly hashed CUDA/cuDNN binary;
two-action exact-device/engine smoke. Then at most one fresh three-game run,
which must use a new root and pass artifact, contract and process-cleanup checks.

PASS: report counts equal engine counts per game and overall; no game exceeds 128;
compare actions, coordinates and frame/state/level outcomes with the corresponding
prefix of the original run (ignore GUIDs and timing). Prefix mismatch must be
reported and investigated before claiming evaluation parity. The final shorter
prefix is expected; unchanged zero scores alone would not demonstrate parity.
No confidence intervals, seed selection or model-quality promotion.

STOP on the first integrity or mismatch failure. Do not extend the population,
train or retry within a root. Max additional GPU runtime 5 minutes including smoke;
600 seconds remains the runner's inherited suite ceiling, but this confirmation
has an external 300-second process limit. Qwen remains blocked before inference;
this confirmation cannot satisfy the earlier two-arm PASS requirement.

Next decision: if the evaluator passes, retain the bounded Tofy reference and
complete model asset preparation plus non-game device/context validation before
opening a separate pretrained-model screen. Do not tune Tofy from these three games.
