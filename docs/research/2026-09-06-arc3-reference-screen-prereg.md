# Three-game offline reference screen

Registered before model inference or a baseline game action on September 6,
2026 CDT. Download/bootstrap engine actions are preparation only and excluded.

## Claim and decision

Claim: fixed Tofy and local pretrained reference configurations can be evaluated
through a common official offline engine loop, with complete interaction and
runtime records, correctly preserved engine scores and bounded resource use.
This is a descriptive single-seed exploratory screen, not an improvement test,
architecture proof, generalization promotion or prediction of prize success.
No mathematical derivation can determine either checkpoint's policy on these
unobserved action sequences. Source/fixture tests instead establish prerequisite
properties: one initialization, legal actions, reset order and score preservation.

The known Tofy G checkpoint fails held-out synthetic transition gates and its
greedy controller has no trained task-value objective. A low public score is
plausible. The selected Qwen text-only baseline lacks vision, code tools and
explicit rule memory. This run can establish reference behavior; it cannot
answer whether the planned full hybrid agent will win the prize.

## Frozen population and arms

- Games: `ar25-0c556536`, `bp35-0a0ad940`, `cd82-fb555c5d`, in that order.
  These are the first three alphabetically in the existing frozen public list;
  selection is independent of new model outcomes. No public game source or
  score baselines enter an agent prompt or policy.
- Engine seed 0 for every arm/game; model sampling seed 0, temperature 0 for
  Qwen. Single seed is a screen only. Environment versions and file hashes,
  Python/toolkit versions, source revisions, inference binaries and weight
  hashes must be written into final configs before opening the run.
- Tofy reference: registered G final raw step-2048 checkpoint,
  SHA-256 `07ef8d1b3b79d99fad61c3fb9ae4de193ff2a2fc77ba8a4a1e5b33bcbdb70b1a`;
  greedy, context defaults from v6, no adaptation. Choosing the final raw state
  follows the already-reported synthetic diagnostic; no public selection occurs.
- Pretrained reference: Qwen3.5-9B, Unsloth Q4_K_M at revision
  `3885219b6810b007914f3a7950a8d1b469d598a5`, SHA-256
  `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`.
  Lossless palette text; four recent turns; max 1024 output tokens per call;
  no vision or executable tools. Inference context/sampling/server flags must
  be frozen after a non-game fit/tokenizer smoke, before actual game inference.

## Budgets, comparison and gates

Both arms use 128 environment actions per game and per level, at most three
GAME_OVER retries per level, 180 seconds per game, 600 seconds per arm and a
30-second decision deadline. Fresh game/agent state per game; no cross-arm
trajectory access. Each arm launches a fresh owned model process per game.
The 180-second gameplay cap starts after readiness; startup, gameplay and total
elapsed times are recorded separately. The 600-second suite budget includes
startup, gameplay and cleanup. A shared contract hash binds population and
limits, and environment hashes bind the actual cached game implementations. Tofy inference batches are 128 candidates; no training or
gradient accumulation occurs. Qwen's physical/logical prompt batches are pinned
after safe device fit; this is not a training batch comparison.

The exact full-screen contract SHA-256 is
`2656cdff54be80460e010872e1dcfbb7fdd6775281a10cc5e40d71f40a041f2a`; the two-action smoke
contract is `f02ad18a7e3f3d148d0b101b1d376a9fc7fc78a570bdd0016c4182d9a2de67e6`
(one game, 60 seconds gameplay, 120 seconds suite; otherwise the same fields).
Configurations must match these independently recorded digests before launch.
Final observation ingestion may consume at most five seconds of cleanup after
the action-selection window; no extra action is selected or executed. It is
included in recorded gameplay wall time and the hard suite acceptance cap.

Primary recorded outcome: engine native root score for exactly these games.
Also report every game's level completion, actual actions, resets, wall time,
stop reason, errors, input/output tokens and model limitations. No significance
test or confidence interval is justified by one fixed-seed screen; no
multiple-comparison or checkpoint selection is performed. No promotion threshold.

PASS for the implementation claim: both configurations finish the registered
population, close their scorecard, stop their processes, produce complete
trajectories and reports, and pass artifact/source/score integrity checks. Zero
levels can pass plumbing only, never model-quality promotion.

FAIL/BLOCKED: any wrong artifact/version, invalid action, protocol/score error,
unexpected whole-game reset, unclosed process or unrecoverable resource failure.
Retain failed evidence separately. Do not silently retry, tune or extend the
same run root. A model decision timeout within its remaining time budget must
be recorded distinctly from a server crash; determine the next step from the
failure evidence before a new registered screen. Expiry of the overall game
budget during a decision is an ordinary `time_limit` stop with no returned
action executed. Expiry of the shorter decision deadline while game time
remains is an evaluation failure. This distinction is fixed before inference.

## Premise checks and stop rule

Before the screen: source/fixture checks; exact binary/device load and one
non-game chat response; one bounded real-engine integration smoke (at most
two actions on the first registered game) for each arm under separate roots.
Smokes are implementation evidence only. Stop after the two registered screens
or the first integrity blocker for an arm; no automatic training or expanded
public evaluation. Maximum experimental GPU runtime 25 minutes including fit,
smokes and both arms. Asset downloading is separate preparation, capped at
15 minutes per bounded attempt and may not be reported as model evidence.

Next decision: analyze outcome and error categories before selecting a larger
reference evaluation or one-factor memory/representation intervention. Neither
the screen nor its public data changes the older registered synthetic treatment.
