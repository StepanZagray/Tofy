# Tofy

Clean-slate hidden-objective planning research. P1 supplies an exact-simulator
feasibility harness; P2 is the active Candle implementation of a recursive latent
world model, sequential curriculum, falsification lessons, and probabilistic TRM
rollouts.

The previous VecLab → LeJEPA → world → Qwen bridge experiment is **archived in Git
history as a negative result**. Do not resurrect that stack for P1.

## Phase status

P0 was skipped at the user's direction. P1 established three exploratory mechanisms:

1. Test whether candidate-goal discrimination beats simple exploration baselines
   when the objective is hidden (P1A).
2. Test whether planning beats reactive one-step control when the objective is known
   (P1B).
3. Test whether one probe can falsify several hidden-goal hypotheses more efficiently
   than testing candidate goals one at a time (P1C).

P2 now learns goal-free pixel dynamics from those synthetic lessons. Public candidate
features condition only the auxiliary predicate head; the hidden goal/index never
enters the dynamics model. Official ARC-AGI-3 recordings are held-out transfer data,
not training, checkpoint-selection, or tuning data.

## Layout

| Module | Role |
|---|---|
| `src/domain.rs` | Exact grid MDP: goals, state, simulator |
| `src/generator.rs` | Deterministic Train / structurally held-out compositions |
| `src/search.rs` | Exact BFS / beam / heuristics (no value head) |
| `src/agents.rs` | P1A / P1B / P1C controllers |
| `src/report.rs` | Versioned JSON reports + exploratory gates |
| `src/experiment.rs` | clap CLI for P1 and P2 commands |
| `src/p2/model.rs` | categorical pixel encoder, shared recursive block, event/Q heads, PTRM |
| `src/p2/sigreg.rs` | LeWorldModel Epps--Pulley SIGReg loss |
| `src/p2/data.rs` | synthetic lesson transitions and ARC-shaped tensors |
| `src/p2/arc3.rs` | official-toolkit recording JSONL importer |
| `src/p2/train.rs` | ordered curriculum trainer and runtime-gradient trace |
| `src/p2/eval.rs` | held-out dynamics, rollout, calibration, and PTRM metrics |

## Conditions

### P1A — hidden-objective discovery

Same generated episodes for every agent. Only `oracle_objective` receives the true goal.

Every scenario exposes candidate predicates from six families: reach a marker,
collect all objects, activate switches in order, preserve a resource while reaching a
marker, reach a marker while avoiding a named hazard, and trigger one of several
terminal pads. Hidden avoid-goal episodes end irreversibly when the forbidden hazard
is touched, so a wrong commitment can be catastrophic. Hidden families are assigned
round-robin by episode ID rather than left to random sampling.

| Agent | Role |
|---|---|
| `random` | Uniform legal actions |
| `novel_state` | Prefer unvisited state keys |
| `greedy_apparent_progress` | Greedy on a public proxy goal |
| `candidate_goal_discrimination` | Exact test plans over the candidate set |
| `oracle_objective` | Shortest path to the true hidden goal |

### P1B — planning necessity

Same episodes. **All** methods know the true objective: non-oracles via
`AgentConfig.planning_goal`; `oracle_optimal` via the `true_goal` argument only.

| Agent | Role |
|---|---|
| `reactive` | One-step greedy on the true goal |
| `pause_compute` | Same action choice + extra non-rollout evals |
| `best_of_k` | Sample K actions, pick best heuristic |
| `beam_search` | Limited-width beam / best-first |
| `oracle_optimal` | Exact shortest path on the simulator |

### P1C — set-aware hidden-objective planning

The objective is hidden exactly as in P1A. P1C compares sequential discrimination
with two parallel planning primitives and five routing rules on identical paired
episodes. None receives the hidden goal or hidden index. P1 has no graded prior, so
each of the `n` exact-live candidates has explicit probability `1/n`: `n=1` is a clear
winner, `n>=4` is broad uncertainty, and `n=2..3` is a narrow several-winner tie.

The shared-progress primitive takes one jointly safe action only when at least two
and a strict majority of canonical exact plans recommend it. The falsification
primitive commits to a safe exact probe whose endpoint makes at least two live goals
predict terminal success; a nonterminal observation falsifies all those predictions.
Both reject live catastrophic failures and irreversible viability loss.

Mixed routers try their preferred primitive, then the other if the preferred method
is unavailable, then a safe single-goal probe. With one candidate they solve it
directly. If no complete plan is jointly safe, they take at most one jointly safe
step and reassess; if no such step exists, they stop.

| Agent | Role |
|---|---|
| `candidate_goal_discrimination` | P1A sequential one-candidate test-plan baseline |
| `set_aware_parallel_planning` | Falsification-only parallel method |
| `shared_progress_planning` | Shared-progress-only parallel method |
| `broad_progress_narrow_falsify` | Progress for `n>=4`; falsify for `n=2..3` |
| `broad_falsify_narrow_progress` | Reversed candidate-count routing |
| `alternating_parallel_planning` | Alternate preferred methods at each ambiguous planning decision |
| `cost_aware_parallel_planning` | Compare falsifications/action with exact net distance reduction/action; ties falsify |
| `capped_broad_progress_planning` | Broad-progress routing, but force a falsification attempt after two consecutive progress actions |
| `oracle_objective` | Shortest path to the true hidden goal |

### P1C hard — research before commitment

This separate adversarial challenge contains only sequential discrimination,
falsification-only planning, and broad-falsify/narrow-progress planning. Its forked
maps contain a deep recoverable false lead and a safe multi-goal endpoint probe. A
deterministic attempt stream keeps a scenario only when sequential discrimination
retargets after at least three falsified commitments. Parallel-policy results are
never used to select tasks, and all six hidden families remain round-robin.

Because P1 has no graded prior, this tests greedy commitment to the easiest-looking
live hypothesis, not literal maximum-posterior planning. The hard set is intentionally
selected against that sequential behavior, so it is a developmental stress test for
ARC-like delayed disambiguation rather than an unbiased estimate of general lift.

## Metrics

Primary proxy: **oracle-normalized efficiency**

\[
\min\!\left(1.15,\left(\frac{a_{\mathrm{oracle}}}{a_{\mathrm{agent}}}\right)^{2}\right)
\]

on success, else `0`. This is **not** RHAE.

Also recorded: success, scored environment actions, internal expansions/evaluations,
terminal failure, exhaustion, oracle optimal actions, and (P1A discrimination)
correct objective identification / actions-to-identification / incorrect commitments.
P1C additionally records shared-progress actions, actions spent in multi-goal probes,
candidates removed by evidence, and switches between parallel methods. These are
diagnostics; every strategy gate still uses success and oracle-normalized
environment-action efficiency.
An unsolvable oracle is reported explicitly; the action budget is never substituted
as a fake normalizer.

Reports are versioned, sorted, and free of wall-clock / platform fields so identical
commands produce byte-identical JSON.

## Exploratory gates

The P0 research contract was not frozen, so gates remain **exploratory**. They use
deterministic paired, goal-family-stratified bootstrap intervals and require the lower
95% confidence bound to clear both configured lift thresholds. Success and efficiency
are compared with their independently strongest baselines. Missing pairs, non-finite
values, fewer than two seeds, or fewer than two observations from any of the six goal
families fail closed.

- **P1A:** `candidate_goal_discrimination` vs best non-oracle exploration baseline on
  `HeldOutComposition` (success + efficiency lifts).
- **P1B:** (a) `oracle_optimal` vs `reactive` (planning necessity);
  (b) `beam_search` vs strongest one-step control (`reactive` / `pause_compute` /
  `best_of_k`).
- **P1C:** one gate per parallel strategy versus sequential
  `candidate_goal_discrimination` on held-out success and efficiency. Separate
  diagnostics verify which primitive each router actually used.
- **P1C hard:** both research-first policies versus sequential discrimination, plus
  broad-falsify/narrow-progress versus falsification-only. The same paired metrics are
  used; sequential retarget count verifies the challenge contract.

## Commands

Quick local smoke (small enough for a laptop). These commands validate execution and
are expected to fail the statistical gates because they use one seed and incomplete
family replication:

```bash
cargo run --release -- p1a --seed 1 --episodes-per-split 2 --output runs/p1/p1a.json
cargo run --release -- p1b --seed 1 --episodes-per-split 2 --output runs/p1/p1b.json
cargo run --release -- p1c --seed 1 --episodes-per-split 2 --output runs/p1/p1c.json
cargo run --release -- p1c-hard --seed 1 --episodes-per-split 2 --output runs/p1/p1c_hard.json
cargo run --release -- all --seed 1 --episodes-per-split 2 --output runs/p1/all.json
```

Useful knobs:

```bash
cargo run --release -- all \
  --seeds 1,2 \
  --episodes-per-split 6 \
  --beam-width 8 \
  --beam-horizon 24 \
  --best-of-k 4 \
  --pause-extra-evals 16 \
  --bootstrap-seed 12684203258873380865 \
  --bootstrap-samples 999 \
  --min-success-lift 0.05 \
  --min-efficiency-lift 0.05 \
  --output runs/p1/all.json
```

This multi-seed command is still implementation validation, not a recorded P1
experiment. Experimental settings must be chosen before examining their output.
Standalone `p1a` and `p1b` use the general generator. Standalone `p1c` uses the
multi-goal-probe generator; `p1c-hard` adds sequential-only hardness selection and
runs exactly three agents. `all` uses the regular P1C-qualified scenario set for every
agent so every comparison inside the combined report remains paired.

Validation:

```bash
cargo fmt
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

## P2 commands

The defaults are deliberately tiny smoke settings and always write
`research_claim=false`:

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --output-dir runs/p2/smoke

cargo run --release -- p2-eval \
  --checkpoint runs/p2/smoke/model.safetensors \
  --train-config runs/p2/smoke/config.json \
  --output runs/p2/smoke/eval_report.json

cargo run --release -- p2-arc3-eval \
  --checkpoint runs/p2/smoke/model.safetensors \
  --train-config runs/p2/smoke/config.json \
  --arc-recordings-dir /path/to/official-toolkit-recordings \
  --output runs/p2/smoke/arc3_eval_report.json

scripts/audit_p2.sh \
  runs/p2/smoke/analyzer \
  runs/p2/smoke/model.safetensors \
  runs/p2/smoke/runtime.json
```

See [`docs/P2.md`](docs/P2.md) for the contract and
[`docs/CANDLE_MODEL_ANALYZER.md`](docs/CANDLE_MODEL_ANALYZER.md) for the external
audit boundary. Track metrics in [`docs/RESULTS_P2.md`](docs/RESULTS_P2.md) (index:
[`docs/RESULTS.md`](docs/RESULTS.md)).
