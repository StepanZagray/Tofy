# Frozen checkpoint diagnostic registration

The first screen at source `2d194f894b78da1edc0bf323f1b075752e34c93f`
failed every learning gate. Its 768 paired queries all selected action 2. This
establishes constant decisions on that population, not constant probabilities or
internal activations. No new optimization is authorized by this registration.

Claim: the same frozen checkpoint and exact inputs produce numerically comparable
CPU/CUDA readouts, and changing only valid observed support has measurable or
negligible influence on policy probabilities. This local check cannot distinguish
global expressivity from optimizer reachability. It can expose a backend mismatch
or quantify information discarded by argmax before spending on fixed-set fitting.

Use the first screen's final checkpoint, with its recorded SHA-256 verified before
each run. Keep the exact model code, weights, seed 9173 and four inference loops.
One existing evaluation layout under all 16 training and 8 held-out rules, with
true, cleared and direction-shifted valid support, gives 72 policy readouts per
device. No hidden rule enters model inputs. Compare the same rows on CPU and CUDA,
using batch one for both. Use a newly hashed binary from reviewed pushed clean
source; adding diagnostic logging necessarily changes binary identity.

Record complete four-action policy logits/probabilities, selected actions, value,
and reward probabilities in the diagnostic artifact. Separately decompose exact
changed-pixel correctness into vacated cursor pixels and destination pixels using
simulator targets; verify these exhaust changed pixels. This tests, rather than
assumes, the suggested perfect-origin/chance-destination explanation for 62.5%
changed accuracy. This decomposition uses the first prediction row of each split;
it is not a rescore of all original 128 action tuples. Count within-patch
categorical inconsistency without assigning
it an after-the-fact promotion threshold.

Backend gate: all selected actions equal and maximum policy probability difference
at most 0.001 on paired CPU/CUDA rows. Report absolute and relative logit differences
without silently changing tolerances. A failure calls for numerical/backend
diagnosis before training; it does not automatically identify a layout bug.

Sensitivity: report per-layout/action logit and probability ranges over true rules,
true-versus-wrong-support differences, and top-action margins. Classify probability
range at most 0.00001 as numerically negligible under this diagnostic; larger
ranges with constant decisions show subdecision sensitivity, not correct inference.
Do not infer zero gradients or globally constant function from an unchanged argmax.

One CUDA and one CPU run, each with a five-minute hard cap, fresh roots, no search
or closed-loop navigation. Record all failures, input/checkpoint/provenance hashes,
device identity and timing; stop processes before final artifact verification.
No checkpoint, metric, threshold or rule subset selection after seeing results.

Next decision: with backend agreement, register a balanced fixed-set fit test to
check whether optimization can reach history-dependent choices. A success would
show fitting capacity only, not explain the original failure solely by budget or
establish generalization. A failure would leave representation and optimization
both possible. Advisor suggestions remain hypotheses requiring these bounds.

Normalization correction: the first screen's prose incorrectly named LayerNorm.
The exact committed implementation uses RMS normalization with a clamped second
moment and no learned normalization gain. This is a documented registration
deviation; retain the original negative result as exploratory, with no promotion
claim. Correct the description, not the checkpoint or the historical artifacts.
