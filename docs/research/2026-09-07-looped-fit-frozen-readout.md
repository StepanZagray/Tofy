# Frozen readout after the fixed-set fitting test

Registered September 7 before observing the completed fitting result. The resumed
fit is running from b1e81e3a with unchanged model/task bytes and periodic weight
checkpoints. This diagnostic uses the existing evaluator; no optimizer step,
checkpoint search or public-game data is introduced.

## Questions and interpretation

Does the final policy respond to new query layouts, observed control rules, both,
or neither? Can a fitting solution generalize to fresh layouts and held-out rule
combinations? Small input-dependent effects cannot alone locate a defective
module or establish useful control. A fit failure remains a failure of the tested
optimization recipe, not an expressivity impossibility theorem.

Run the initial and registered final/first-passing checkpoints with the same
b1e81e3a binary, mode inspect, CUDA device 0, batch/effective batch one, four loops,
32 evaluation layouts, data seed 9173, and a 300-second cap per checkpoint. Keep
the exact initial/final checkpoint hashes, evaluator revision, binary hash,
input hashes and artifact manifests. If the parent fails integrity, do not treat
its weights as completed fitting evidence; any rescore must retain that failure.
The initial checkpoint may be inspected while the parent is running only on CPU;
the main initial/final comparison must use the same device and exact inputs.

The paired one-step rule probe uses all 16 training and eight held-out mappings
per layout: 768 rows, each with true, cleared and valid wrong support. Its query
seed domain differs from the training domain. Cleared inputs must be identical
within each layout; label balance implies exactly 25% selected-action accuracy.
All valid wrong-support tuples must change the unique correct action.

## Fixed measurements

- Absolute true-context, cleared and follows-presented-wrong-rule accuracies, by
  split. Bootstrap whole layouts 10,000 times with seed 1907 for paired
  true-minus-cleared and true-minus-wrong-under-real-rule accuracy intervals.
- Across-layout range of each cleared policy probability, and within-layout
  true-versus-wrong maximum probability difference; report median, 95th percentile
  and maximum across rows, plus the corresponding winning probability margins.
  Use probabilities as primary; raw logit shifts shared by all classes have no
  effect on decisions. Save raw vectors and input hashes.
- Per-action successor exactness versus copy, vacated/destination correctness,
  nonuniform patch count, reward precision/recall counts, and value MSE versus
  the existing fixed comparator. These are separate gates, not policy evidence.
- Initial/final input hashes and every row's split/layout/rule/oracle labels must
  match. Frozen runs must have zero optimizer updates and unchanged checkpoint
  hashes. Stop and preserve invalid provenance rather than relax a gate.

This is an exploratory single-seed diagnostic. A useful generalization screen
requires at least 75% true and follows-presented-rule accuracy on each split,
with the lower paired true-minus-cleared 95% interval above 25 percentage points.
Passing warrants fresh-seed confirmation, not ARC deployment. Failing does not
authorize a longer run automatically. Choose one subsequent experiment from the
fit and frozen results; match the relevant training recipe and preregister it
before treatment outcomes. Representation changes may help for several reasons;
their success cannot uniquely prove attention dilution.
