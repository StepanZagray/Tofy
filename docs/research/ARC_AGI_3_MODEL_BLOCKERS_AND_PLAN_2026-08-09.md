# ARC-AGI-3 model blockers and evidence-driven plan

Date: 2026-08-09

Scope: current `main` at `17fbfdffba917bcef23f5bdea86aa83561524272`, the
completed `ab-sigreg-geometry-v2` seed-1 pilot, current P2 implementation, and
primary sources available on this date.

## Decision

Do not run seeds 2/3 of either failed geometry arm and do not restart the full
28,672-update curriculum. The next useful work is an evaluator slice that reports
representation health at the exact pre-RMS, post-RMS, pooled/spatial, target, and
predicted seams. After that lands, run a fresh, short, one-change pilot of temporally
centered SIGReg on ordered windows. In parallel with the learned-world-model path,
build the factual ledger, state graph, and fixed-interface executable-model replay
verifier. Those interfaces should precede a local reasoner, but the reasoner/verified
world-model track must no longer be deferred behind completion of all neural pilots:
it has substantially stronger direct ARC-AGI-3 evidence than the current latent-only
policy.

No plan can honestly promise 100%. ARC-AGI-3 gives 100% only when every level is
completed and every game's capped, level-weighted score reaches 100%. Individual
level ratios are squared and capped at 115%, so faster levels can offset some slower
ones; incomplete later levels cap the game score. This makes 100% a system-level
target, not a loss threshold for the current model.
[Official scoring methodology](https://docs.arcprize.org/methodology)

## What 100% requires

The benchmark is explicitly about four coupled abilities: exploration, modeling,
goal-setting, and planning/execution in unseen, instruction-free, turn-based
environments. Its official report says humans solve all environments while frontier
systems were below 1% in March 2026.
[ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621)

The June 2026 milestone winners are important evidence about the missing system
layer. The first-place system used a local 27B multimodal model that wrote and ran
Python in a live REPL; the second- and third-place systems used local 31B VLM policies,
recent-frame history, reflection memory, legal-action guards, and structured action
output. These results do not prove that this architecture reaches 100%, but they do
show that the current competitive path combines visual reasoning, memory, and an
interactive harness rather than relying on a small latent predictor alone.
[ARC Prize Milestone 1 report](https://arcprize.org/blog/arc-prize-2026-milestone-1)

The competition evaluation is offline, so any such reasoning model and tools must be
local at evaluation time.
[ARC-AGI-3 competition rules](https://arcprize.org/competitions/2026/arc-agi-3)

## Current evidence

The completed geometry pilot is operationally trustworthy: all eight seed-1 phases
exited zero, manifests verified, and the frozen gate reproduced locally. It rejected
both arms.
[Completed pilot analysis](../P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md)

| Arm at update 2,000 | Aggregate action ratio | Random-one-step ratio | Changed-transition improvement | Variance | Rank fraction | Dynamics H8 |
|---|---:|---:|---:|---:|---:|---:|
| control | 1.2095 | 1.7365 | 0.4661 | 0.006128 | 0.02529 | 1.8082 |
| pre-RMS spatial | 1.0004 | 1.0016 | -6.4256 | 1.95e-5 | 0.00839 | 0.03124 |

The control learned measurable action dependence but remained far below the frozen
`0.10` effective-rank floor and its horizon-8 error worsened `16.5x` from update
1,000. The treatment produced very small raw latent MSE by shrinking/aliasing the
representation: actions had no measurable effect and learned changed-transition
prediction was `7.43x` worse than copy-forward. Action sensitivity, non-collapse,
and rollout stability are therefore separate gates.

Several engineering problems identified in the August 5 readiness report have since
landed: real gradient accumulation, deterministic shuffled episode scheduling,
warm-start-consistent inference adapters, differentiable PTRM ranking labels, stable
transition indices, a held-out identifiability bridge, the official per-game RHAE cap,
and multi-horizon prefix loss infrastructure are present in current code. They should
not be reimplemented.

However, the current prefix path is not the direct action-prefix architecture from
Fast LeWorldModel. `WorldModel::prefix_predict` consumes one action and adds a spatially
uniform channel delta; `prefix_multi_horizon_loss` repeatedly feeds each prediction
back into the same one-action head. It is still autoregressive and therefore still
exposed to compounding.

The live ARC policy also lacks a learned goal or action-value signal. It ranks legal
actions by transition fidelity, reliability, no-op probability, and latent action
effect, so even a perfect next-state predictor would not by itself know what winning
state to seek.
[Live policy scope](../P2_ARC3_LIVE_EVAL.md#policy-scope)

Two evidence/serving gaps remain even though many earlier evaluator repairs landed.
The optional episode JSONL path currently emits two synthetic aggregate rows with
placeholder episode IDs rather than the real episode/horizon observations, so it
cannot support the intended episode-level confidence analysis. The live HTTP retry
path also needs an idempotency decision: retrying a mutating action after an ambiguous
transport failure can repeat the action and damage RHAE even when the model chose
correctly. Neither issue explains representation collapse, but both must be fixed
before expensive multi-seed or official claims.

## Evidence classes and responses

The order below is not a causal or numerical ranking. It separates measured neural
failures, benchmark-required but absent capabilities, correctness/safety defects, and
cross-domain research candidates. No available experiment compares their relative
importance end to end.

### Measured neural failure: representation collapse and task/state aliasing

The hard evidence is the low downstream effective-rank fraction in both arms. The
directly relevant primary result finds that applying SIGReg to a multi-task
latent marginal compresses task-cluster separation relative to within-cluster
variation. It proposes applying the same regularizer to temporally centered residuals
`r_t = z_t - mean(z_s, s in W_t)` instead. This removes direct Gaussianization
pressure from slowly varying task/state centers while still penalizing a temporally
constant representation. The paper used `W=8` by default and found all non-degenerate
tested windows substantially better than raw marginal SIGReg on its benchmark.
[Temporally Centered SIGReg](https://arxiv.org/abs/2607.26924)

This is a strong hypothesis for Tofy, not a guaranteed transfer: the source benchmark
is LIBERO, its encoder is a ViT, and its downstream task is behavior cloning. Tofy must
therefore run a one-change causal pilot using ordered synthetic episodes and its own
action/non-collapse/rollout gates.

Response:

1. Instrument every relevant seam before changing training.
2. Preserve the current downstream normalized latent and change only the SIGReg target
   from marginal latents to centered residuals from ordered windows.
3. Start with `W=8`; include `W=4` only as a small preregistered sensitivity arm if the
   first pilot is positive. Never use `W=1`, whose centered residual is identically zero.
4. Keep spatial geometry, row cap, optimizer schedule, seed, and evaluation frozen.

### Measured neural failure: actions are not uniformly identifiable

The control's aggregate and random-one-step action gates passed, but the completed
analysis shows that this did not rescue rank or rollout. Per-family gates remain
necessary because a model can learn easy directional effects while marginalizing
coordinate, undo, hazard, or exploration actions.

Action-conditioned world-model research provides useful structural probes: identity,
inverse, and composition consistency can expose a model whose visually plausible
dynamics are not actually governed by actions. These constraints apply only where
Tofy's synthetic action family genuinely has the relevant algebra; coordinate clicks
and irreversible hazards must not be forced into a false group structure.
[World Models as Group Actions](https://arxiv.org/abs/2605.24578)

Response:

- retain shuffled-action and changed-vs-copy gates by source;
- add no-op identity, valid directional inverse/undo, and composition-equivalence
  probes on synthetic families with known semantics;
- reject any representation arm whose action lower confidence bound returns to 1.0,
  even if raw latent MSE improves.

### Cross-domain candidate for a measured failure: direct action-prefix prediction

The control's horizon-8 deterioration is direct evidence. Fast LeWorldModel identifies
the same architectural problem in repeated one-step latent rollout and replaces it
with a causal action-prefix encoder that predicts future latents directly from the
latest real anchor and every prefix of a candidate action sequence in parallel. In its
reported experiments, this reduced error growth, improved average planning success
from 85.8% to 90.5%, and reduced the dynamics-module time by 3.9x.
[Fast LeWorldModel](https://arxiv.org/abs/2606.26217)

Response:

- do not label the existing autoregressive `prefix_head` as Fast-LeWM;
- after representation health passes, add a sequence/prefix encoder whose horizon
  predictions are all conditioned on the same real anchor latent;
- supervise horizons `1,2,4,8,16` against encoded real future frames;
- compare direct-prefix and composed one-step predictions, but let planning use the
  direct path only if it passes frozen downstream gates.

### Benchmark-required capability: a verified executable mechanics model is absent

The strongest direct ARC-AGI-3 architecture evidence found in this run comes from a
July 2026 nested coding-agent study. Its complete fixed-interface treatment required
an executable transition engine, initial-state reconstruction, a renderer, a planner,
scheduled simplification, and exact replay of every recorded settled observation.
That treatment ranked first in all four controlled model/effort settings. A follow-up
with GPT-5.6 Sol solved all 183 public levels and scored 98.97 RHAE at `xhigh`.
[Controlled executable-world-model study](https://arxiv.org/abs/2607.15439),
[author repository](https://github.com/astroseger/arc-3-agents-baseline1)

This evidence must be narrowed carefully. The experiment does not isolate verification
from fixed interfaces, tools, templates, and simplification. A persistent executable
deliverable alone sometimes performed worse than a textual model. GPT-5.6 postdates
the public games, and no semi-private or private result was available, so 98.97 is
public-set saturation rather than evidence that hidden ARC-AGI-3 is solved.

Response:

1. Keep the learned neural world model as the predictive core, but add a separate
   executable hypothesis model with fixed reconstruct/step/render/plan interfaces.
2. Replay every factual attempt and reject multi-step use of any executable model that
   does not reproduce public status and settled frames exactly.
3. Preserve textual mechanics hypotheses and explicit modeling debt; exact replay is
   consistency with observed history, not proof on unseen states.
4. Execute at most one real action before comparing prediction and observation and
   replanning.
5. Ablate textual, executable, and verified variants under equal local compute and
   action budgets; do not attribute the bundled treatment's gain to verification alone.

Keeping the neural model central is a Tofy design constraint, not a conclusion of the
coding-agent study. The hybrid interface must therefore make the neural contribution
ablatable; hidden/public agent performance, not architectural preference, decides how
much planning authority it ultimately receives.

### Benchmark-required capability: goal discovery and memory are absent

ARC-AGI-3 provides no instructions or target. The official benchmark therefore cannot
be solved by transition accuracy alone. A strong non-neural primary baseline segments
frames, masks likely status UI, stores a directed graph of observed states and tested
actions, and returns to frontier states with untested actions. It solved 12/25 private
preview levels in the official evaluation, and the authors report better post-fix
results. Its limitations—large state spaces, partial observability, non-determinism,
and brittle status masking—are also exactly where learned abstraction can add value.
[Graph exploration paper](https://arxiv.org/abs/2512.24156),
[reference implementation](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)

Response:

1. Add deterministic palette/object/component/diff features and state hashing.
2. Maintain episodic memory of real `(observation, action, effect, event)` tuples and a
   state/action frontier graph. Imagined transitions may propose actions but may never
   update this factual graph.
3. Establish graph exploration as a mandatory baseline before crediting the neural
   planner.
4. Add a local multimodal/code-reasoning adapter that can propose mechanics and goals,
   summarize durable reflection memory, and emit only legal actions. Keep it behind a
   narrow interface so model families can be ablated.
5. Switch from information-seeking to efficient goal pursuit only after a frozen
   confidence/margin rule, and rescore after every real observation.

### Correctness discipline: public evaluation cannot be the optimizer

The official toolkit supports local/API interaction, recordings, and scorecards.
[Official ARC-AGI toolkit](https://github.com/arcprize/ARC-AGI)
The repository's existing policy treats public recordings as held-out transfer data,
not training, tuning, threshold selection, or checkpoint selection. Preserve that
boundary. Public score improvements without a frozen synthetic decision rule are not
evidence of generalization.

## Implementation and experiment plan

### Phase 0 — representation-seam observability

Add layer-aligned representation diagnostics to `src/p2/eval.rs` without changing
training or model weights. Report variance, participation-ratio effective rank, row
count, and dimension for:

- pre-RMS pooled current/next encoder features;
- post-RMS pooled current/next encoder features (the existing gate);
- pre-RMS spatial cells;
- post-RMS spatial cells;
- predicted next latent, pooled and spatial;
- target next latent, pooled and spatial.

Use a deterministic row cap for spatial diagnostics, preserve the existing downstream
gate fields consumed by `scripts/p2_ab_gate.py`, and add a named seam map. Add unit
tests for shape accounting, deterministic capping, collapsed/diverse summaries, and
JSON schema. Re-evaluate the four existing 1,000/2,000 checkpoints; do not retrain.

Gate: the same checkpoint and seed reproduce identical seam metrics, and the report
localizes whether collapse first appears before normalization, after pooling, or in
the predictor.

Then replace the placeholder episode JSONL writer with real per-episode horizon-4/8/16
rows. Preserve `(seed, episode_id)` as identity because retarget traces can change
family; record the sorted family provenance contributing through each horizon instead
of splitting an episode by family. Test mixed lengths, invalid normalization
denominators, deterministic ordering, aggregate reconciliation, failure durability,
and repeated-run equality. Complete both Phase-0 slices before the next training pilot.

### Phase 1 — temporally centered SIGReg pilot

Add an explicit `sigreg_target = marginal | temporal_residual` contract. Build
non-overlapping ordered windows from one episode/family, compute each local mean and
`r_t`, and apply the unchanged SIGReg statistic to those residual rows. Do not combine
this with a normalization, pooling, projector, model-width, or sampler change.

Pilot control and treatment from fresh initialization at seed 1. Evaluate at updates
250, 500, 750, and 1,000 because the prior control gained action dependence while
rollout degraded between the sparse 1,000-update checkpoints. Promote to seeds 2/3
only if, at update 1,000:

- downstream mean variance is at least `1e-4`;
- downstream effective-rank fraction is at least `0.10`;
- aggregate and random-one-step shuffled/true ratios are at least `1.10`, with lower
  95% confidence bounds above `1.0`;
- changed-transition learned error is at least 10% better than copy-forward;
- normalized H8 does not regress more than 25% against paired control; and
- no seam shows a new collapse hidden by a downstream aggregate.

Stop immediately if the treatment reproduces action ratios near `1.0` or makes the
learned changed-transition predictor worse than copy-forward.

### Phase 2 — action-structure pilot

On the best non-collapsed representation only, add identity/inverse/composition losses
for synthetic families where those relations are exact. First add evaluation-only
GAC-style probes, then train one relation at a time. Promotion requires per-source
action gates and a powered H8 improvement without harming ordinary one-step accuracy.

### Phase 3 — true direct action-prefix prediction

Replace the current autoregressive prefix experiment with an action-sequence encoder
and direct anchor-to-prefix predictions. Keep the one-step transition as a baseline.
Promotion requires:

- one-step normalized error no more than 5% worse;
- H8 at least 30% better and no more than `2x` closed-loop H8;
- finite H16 with p95/median normalized error at most 5;
- a seed-stratified 95% interval for H8 improvement excluding zero; and
- measured planning success and latency improvement, not latent MSE alone.

### Phase 4 — factual graph memory and executable replay verification

Implement canonical observation features, state hashes, tested-action edges, reversible
return paths, frontier choice, dead-action signatures, and level-to-level memory. Keep
exact settled-frame identity separate from any learned/perceptual equivalence. Add a
per-game executable workspace with reconstruct/step/render/plan interfaces and an exact
attempt replay verifier. Compare random/legal, graph-only, textual-mechanics,
unverified-executable, verified-executable, graph-plus-learned-novelty, and oracle
transition baselines on frozen synthetic generators. Optimize completion first, then
action efficiency; report both.

### Phase 5 — local multimodal/code goal reasoner

Add a narrow offline adapter:

```text
observe(real_history, graph_summary, legal_actions)
  -> mechanics_hypotheses, goal_hypotheses, next_action_or_probe, confidence
```

The adapter may inspect image, categorical grid, component/diff features, the factual
ledger, and a fail-closed per-game Python workspace. Generated code runs only in a
no-network bubblewrap namespace with a pinned read-only Python root, sanitized files
and environment, hard process/memory/CPU/file/output limits, and whole-process-tree
termination; if that boundary cannot be established, generated Python is disabled.
It proposes mechanics/goal
hypotheses and executable-model patches; the controller owns replay verification and
the real action call. It may not alter the factual graph except through a confirmed
real action and observation. Compare JSON-policy, textual-world-model, code/REPL,
verified-executable, reflection-memory, and no-reasoner ablations under the same local
compute/action budget. Codex CLI is a development backend only; official competition
mode must use a local offline-capable model.

### Phase 6 — frozen transfer and official scorecard

Freeze code, local model, checkpoint, thresholds, and action budget before public or
private evaluation. Use the official toolkit and closed scorecard. Report per-game and
per-level completion, action count, RHAE, latency, failure class, and full provenance.
Resolve ambiguous mutating-request retries before spending an official action budget;
do not blindly repeat an action unless the protocol proves the first attempt was not
applied.
Only this phase measures progress toward 100%.

## Pod handoff: first bounded coding slice

The first pod agent should implement **Phase 0-A only**: real episode rows and aggregate
reconciliation. This is a measurement-integrity prerequisite, not the architecture
decision test; named representation seams are the following P0-B task. It must not change training,
add TC-SIGReg, start experiments, use API keys, or alter result documents.

Acceptance criteria:

1. Modify only `src/p2/eval.rs` and focused schema/tests required for the episode rows.
2. Preserve real `(seed, episode_id)` identity; record changing family labels as sorted
   `families_through_horizon` provenance.
3. Derive both JSONL and aggregate rollout metrics from the same H4/H8/H16 rows.
4. Add deterministic ordering, guarded copy-forward normalization, and a durable
   same-directory staging/sync/rename/parent-sync writer that preserves an old final
   file on pre-rename failure.
5. Test retargeting families, mixed lengths and absent horizons, invalid denominators,
   aggregate weighting/reconciliation, byte determinism, disabled-JSONL report parity,
   failure preservation, and parent-directory durability.
6. Run `cargo fmt --check`, focused tests, `cargo test`, and `cargo clippy --all-targets
   --all-features -- -D warnings` when the pod toolchain supports them.
7. Commit on a dedicated pod branch and stop. Do not launch GPU training.

This slice makes rollout evidence auditable. The separately reviewed P0-B seam slice
will decide the exact Phase-1 geometry and prevent another pilot from changing two
mechanisms without knowing where the representation failed.

## Stop rules

- Do not replicate a seed-1 arm that fails the frozen hard gate.
- Do not interpret lower raw latent MSE as progress without scale-normalized,
  non-collapse, action-intervention, and copy-forward gates.
- Do not add PTRM/recursion complexity until the deterministic representation and
  direct-prefix baseline pass.
- Do not tune on public ARC scorecards.
- Do not claim progress toward 100% from synthetic model metrics alone.
