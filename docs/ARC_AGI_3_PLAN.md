# Revised path to ARC-AGI-3

## End goal

Build an online agent that can enter an unseen ARC-AGI-3 game, infer its hidden
objective from pixels and public interaction metadata, and solve levels with high
official reward-per-action efficiency. The agent must not receive a hidden goal
index, instructions, simulator state, or an oracle terminal predicate. Only real
environment observations may update its belief about what the game requires.

The target is the official ARC-AGI-3 scorecard. Synthetic success and
oracle-normalized efficiency are development proxies, not RHAE and not final
evidence.

## Decisions carried forward from P1

P1 was a mechanism study on the `p1` branch, not a learned ARC agent. Its useful
conclusion is a control policy for ambiguity:

- when one hypothesis is clearly ahead, plan sequentially toward it;
- while several hypotheses remain credible, choose safe actions that falsify as
  many of them as possible;
- after evidence changes the ranking, retarget instead of defending the first
  commitment.

The first learned planner will therefore use **sequential pursuit plus
falsification-only parallel planning**. Broad-progress planning is not part of the
first integration. It can return later only if a frozen ablation shows that it adds
value beyond those two modes.

P1's exact simulator is not part of `main`. P2 must re-establish every behavior
through learned predictions and real observations.

## Non-negotiable boundaries

1. Dynamics are goal-free. Candidate features may condition predicate and planning
   heads, never the transition representation itself.
2. A model rollout is a proposal, not evidence. Hidden-objective beliefs change only
   after a real action and observation.
3. Public ARC data is held-out transfer evaluation. It is not used to train weights,
   tune thresholds, select checkpoints, or edit the curriculum after results are
   inspected.
4. Synthetic proxy metrics and official RHAE are reported separately.
5. Each promotion uses a frozen config, exact command, Git revision, physical batch
   and gradient-accumulation pair, checkpoint, and analyzer report.

## Stage 0 — make experiments recoverable

Status: implemented; validation remains part of every run.

- Atomic checkpoints contain model weights, AdamW moments and bias-correction step,
  the next curriculum cursor, partial lesson sums, completed lesson reports, and the
  first-step runtime trace.
- `SIGINT` and `SIGTERM` finish the current optimizer update, publish the bundle, and
  exit as `paused`.
- Resume rejects changes to any trajectory-defining training field, including the
  device used for exact continuation.
- An uninterrupted CPU run and a pause/resume run must end with identical parameters,
  optimizer state, cursor, and lesson metrics.
- Before the first accelerator experiment, measure the largest stable physical batch
  and minimize accumulation while preserving the intended effective batch and
  optimizer schedule.

Gate: recovery equivalence, corruption rejection, tests, clippy, and
`candle_graph` audit all pass.

## Stage 1 — establish a trustworthy world model

Train the recursive LeWorldModel/PTRM model through ordered lessons rather than a
single stationary mixture:

1. local categorical dynamics, actions, coordinate actions, and no-ops;
2. multi-step sequential pursuit;
3. safe falsification actions shared across competing hypotheses;
4. false leads and repeated retargeting;
5. later, arbitration examples where uncertainty must choose between falsification
   and sequential pursuit;
6. later, model-error lessons emphasizing irreversible hazards and long rollouts.

Evaluate only on structurally held-out synthetic compositions. Freeze numeric gates
before the full run for one-, four-, and eight-step latent error, public predicate
errors (especially dangerous false negatives), Q calibration, and invalid-input
fail-closed behavior.

Gate: the world model passes every frozen held-out threshold. Passing training loss
or one-step smoke evaluation is insufficient.

## Stage 2 — justify recursion and probabilistic inference

Using the same checkpoint, compare:

- deterministic `K=1`;
- PTRM `K=2,4,8` with a frozen noise sweep;
- extra deterministic recursion at approximately matched compute.

Measure pass-at-K, best-Q-at-K, disagreement, calibration, latency, and memory. PTRM
is enabled only where uncertainty or irreversibility warrants its cost; it is not a
default multiplier on every imagined transition.

Gate: recursive/probabilistic inference must improve a frozen held-out metric at an
acceptable measured compute cost. Otherwise use the simpler deterministic model.

## Stage 3 — learned-model hidden-objective planner

Add a closed-loop planner that maintains probabilities over public candidate
hypotheses and has two planning modes:

- **falsification mode:** while multiple hypotheses are credible, score actions by
  expected safe elimination and information gain under the model's credible
  transition set;
- **sequential mode:** once a frozen confidence/margin rule identifies one clear
  winner, plan toward that hypothesis.

Every real observation re-scores all surviving hypotheses and can switch the mode or
target. Tasks must contain convincing false leads and require several retargets, so a
greedy early commitment has a measurable cost. Wrong commitments must sometimes be
dangerous or action-expensive; otherwise the experiment cannot distinguish research
from cheap trial-and-error.

Compare at least:

- always-sequential pursuit;
- falsification-only until the confidence switch, then sequential pursuit;
- an oracle-model upper bound and a random/legal-action lower bound.

Report success, actions, irreversible failures, number of retargets, model calls,
and wall-clock cost. Synthetic oracle-normalized efficiency remains explicitly
non-official.

Gate: falsification-first planning beats always-sequential on the frozen hard
retarget split without losing ordinary-task success, and the gain survives multiple
seeds.

## Stage 4 — held-out public ARC transfer

Freeze the model, planner, thresholds, noise schedule, and checkpoint before opening
public ARC recordings. Use the existing official-toolkit importer to test observation
and action compatibility, rollout calibration, no-op handling, terminal metadata,
and runtime limits. Do not fit or select anything from this set.

Gate: the frozen system operates fail-closed on the public interface and produces a
complete transfer report. Any design change discovered here creates a new
pre-registered run; it does not rewrite the result already observed.

## Stage 5 — official online evaluation

Integrate the same observation/action loop with the official evaluator. Start with a
small, logged validation allocation, then run the frozen evaluation budget. Record
the official score and RHAE exactly as returned, together with failures, action
traces, latency, compute, checkpoint identity, and analyzer artifacts.

Only this stage answers whether Tofy works on ARC-AGI-3. Earlier stages answer why a
specific component is safe and useful enough to carry forward.

## Immediate next work

1. Run Stage 1b curriculum (`docs/P2_STAGE1B.md`): exploration + hypothesis probes,
   tighter Q threshold, PTRM rank on falsification.
2. Freeze Stage 1 numeric thresholds before inspecting a full result.
2. Measure the largest stable accelerator batch, record the batch/accumulation pair,
   and freeze the full run configuration.
3. Run the world-model gate and same-checkpoint PTRM ablation.
4. Implement the two-mode learned planner only after those gates pass.
