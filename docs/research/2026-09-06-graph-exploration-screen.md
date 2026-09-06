# Graph exploration versus random input: registered development screen

Claim (empirical): the existing uncalibrated observed-state frontier policy may
complete public development levels that G's confidence/effect ranking failed
to complete. Random input is an absolute exploration reference. This comparison
does not isolate learning: the graph arm deliberately makes zero model calls
without calibration, though its bridge still loads the frozen G checkpoint.
It cannot establish a trained-model improvement or generalization to private games.

Rationale: the sealed charged-confirmation trajectories at source 9d783234 show
127 available-action alternatives in each game; excluding row 63, bp35 has
125 repeated state/action tuples and cd82 has 120. Whole-frame hashes have zero
repeated tuples. A counter/timer may therefore defeat exact graph identity too.
No row is excluded in the intervention. This sensitivity check is not evidence
that a discarded row is irrelevant, nor that repeating an action is always wrong.

Population: ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d, in that order.
Seeds: 0 and 1. Both arms use the official offline arc-agi 0.9.9 / arcengine
0.9.3 evaluator, identical versioned assets and 2,000 charged actions per
game and per level, at most 20 GAME_OVER retries, 180 seconds per game,
600 seconds per three-game suite and 30 seconds per decision. No tuning,
extension, best-seed choice or checkpoint selection after observing results.

Graph: phase-a, fail-closed calibration, exact visible-state identity,
unchanged generic coordinate candidates, reviewed confirmed-transition fix.
Random: sample uniformly among available non-reset action IDs, then uniformly
over 64x64 coordinates for ACTION6. This deliberately simple reference has a
different coordinate proposal distribution from the graph; report that confound.
Both arms use evaluator-owned GAME_OVER resets. Agent-requested reset is not
implemented in this screen, and that limits recovery from nonterminal dead ends.

Metrics: native SDK root and per-game score, completed levels, charged actions,
retry count, stop reasons, unique settled-frame/action tuples, and wall time.
SDK level score is min(115,100*(human_actions/agent_actions)^2), averaged with
level-index weights and capped by the weighted completed-level fraction; root
score averages games. Completion and efficiency therefore remain distinct.
No inferential confidence intervals from two fixed development seeds; report
both separately. Any later promotion requires reserved-game confirmation.

Decision: if graph completes at least one level on at least two development
games in each seed and random does not, retain graph as the promising control
for online learning. Otherwise graph remains a control only. Zero completions
reject this particular standalone graph/budget configuration, not all exploration
methods or the possibility of a model-based solution. The CNN prerequisite and
non-game language-model capacity check can continue independently.

Runtime ceiling: 45 minutes total including exact-binary device smoke, four
screens, integrity checks and analysis. Source/binary/config hashes are captured
in each new run root; source must be reviewed, pushed and clean. A two-action
graph smoke and a two-action random smoke precede the registered screens.
An integrity failure stops the affected run; analyze before making a new root.
All evidence is exploratory. No public trajectory enters pretraining. The 22
other public games remain reserved for the first locked confirmation.

## Frozen accounting contracts

- Seed 0 screen contract: `92530ce590c27c5615d9f5a8b0d27a5fa0c3c1988ffd13fc294c411c84967c76`.
- Seed 1 screen contract: `ea4b5bf9590b4577eeeac4460737ee1a486526770fb9735cd128e949feef4119`.
- Two-action smoke contract: `b4ac48404daaea53a86b1a069c69864909acc2cf84a14b9c560c15190d4c16e1`.
