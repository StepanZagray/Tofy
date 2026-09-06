# Online effect learner: prerequisite registration

Registered before implementation or training. This is a new empirical
hypothesis, independent of the failed G generalization treatment.

Claim: a small action-conditioned CNN, freshly initialized for each environment,
can learn to distinguish action-caused visible changes from noops on independent
procedural transition streams. This is a prerequisite for efficient exploration;
it does not establish goal understanding, public transfer, or a prize score.
Correct transition bookkeeping is a separate, locally testable invariant: every
confirmed transition must be folded exactly once before terminal/reset callbacks.

No theorem implies this system-level claim. Counterexamples include hidden-state
aliasing, irreversible traps, sparse goals, timers independent of actions, and
games requiring repeated apparently identical actions. More predicted change is
not necessarily useful. Full-frame copy accuracy cannot establish this claim.

Implementation scope: separate four-convolution change-mask predictor using
categorical palette planes and explicit action/coordinate conditioning. Fresh
deterministic initialization, bounded replay, stable binary-logit loss, finite
gradient checks, and pre-update diagnostics. Do not update or depend on G.
The official ARC-AGI-3 technical report, section 6.1, motivates online change
prediction, but this is an adaptation: architecture details, objectives, replay,
action coverage, control, hardware and datasets differ from StochasticGoose.

Before any long training: compile and deterministic non-ARC unit fixtures;
then freeze a procedural generator and independent seed split; register its exact
population, changed/unchanged counts, physical batch sweep, optimizer schedule,
training/evaluation seeds, runtime and promotion thresholds in a follow-up
registration. Unit fixtures are implementation checks, not model evidence.
No public frame, mechanic, identifier, solution or trajectory enters those data.
Public experience may subsequently update a fresh learner inside that game only;
all weights/replay are discarded between games and evaluation seeds.

The controller must remain graph-only until prequential effect predictions beat
copy and action-independent controls on genuinely changed examples. Novelty,
effect discrimination and level completion are independent gates. Any public
comparison will use the same frozen candidate generator and budgets for both
arms, and will report absolute native score and every game's completed levels.

Development population is the three previously inspected versioned games:
ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d. The other 22 games are reserved
for a first locked confirmation before subsequent full-public evaluation.
Seeds 0 and 1 are fixed for confirmation, with no best-seed aggregation.
The campaign target remains at least 50% levels and 20/100 native score on
each full-public seed; first-level diagnostics do not satisfy that target.

Stop this implementation stage on a failed integrity/finite-gradient check or
after the prerequisite module and tests are reviewed. Analyze before launching
an experiment; this stage's stop rule does not end the autonomous campaign.
