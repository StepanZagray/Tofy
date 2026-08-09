# P2 ARC-AGI-3 world-model-core redesign

Status: implementation specification

Date: 2026-08-09

Source revision: `17fbfdffba917bcef23f5bdea86aa83561524272`
Research run: `~/Research/Knowledge/_runs/2026-08-09T172003Z-tofy-arc-agi-3-world-model-redesign/`

## 1. Objective and decision

Build an offline-capable ARC-AGI-3 agent that keeps a world model at the center of
perception, hypothesis testing, and planning. The architecture is hybrid because the
current evidence rules out treating one latent predictor as a complete agent:

- the learned neural world model supplies compact state, action-conditioned prediction,
  direct multi-horizon proposals, model-error estimates, and learned novelty;
- an executable hypothesis model represents candidate mechanics and goals in inspectable
  code;
- exact replay verification rejects executable hypotheses that contradict factual
  observations;
- an append-only factual ledger and state-action graph contain only real interactions;
- a local reasoning backend proposes hypotheses, model patches, and plans, but cannot
  directly mutate factual memory or bypass the action executor.

Keeping the world model central is a project constraint. Retaining the existing learned
neural model as the predictive core is an architectural choice to be tested, not a claim
established by the ARC coding-agent studies; those studies evaluated textual and
programmatic models, not Tofy's neural model. The fixed interfaces below keep the neural,
textual, and executable contributions separately ablatable.

The target is strong hidden-set generalization and official Relative Human Action
Efficiency (RHAE). This specification does not promise 100%. The current official
methodology requires completion of every level and a capped, level-weighted score of
100% in every game; faster-than-human levels may offset some slower levels. Public-set
saturation is not evidence of that hidden-set result.

## 2. Evidence that determines the design

1. The completed geometry-v2 pilot rejected both arms. The control learned action
   sensitivity but remained collapsed and its horizon-8 rollout degraded; the treatment
   achieved small raw error through an action-marginalized, low-scale representation.
   Action dependence, representation health, and normalized rollout are independent
   gates.
2. Current `prefix_predict` consumes one action and broadcasts one channel delta over
   the spatial grid. `prefix_multi_horizon_loss` recursively feeds those predictions
   back into the same head. It is not direct action-prefix prediction.
3. Current `episodes.jsonl` contains two aggregate placeholder rows, not episode rows.
4. Current live `ACTION*` POSTs share a generic retry loop with reads. The official API
   does not document an idempotency key or reconciliation endpoint, so ambiguous
   mutations must not be replayed automatically.
5. Current Q/reliability signals estimate transition error, not reward, goal progress,
   or official action utility. The live policy is not a hidden-goal solver.
6. A July 2026 nested ARC-AGI-3 study found that a complete fixed-interface,
   simplification, and exact-replay-verification treatment ranked first in every tested
   model/effort setting. Its GPT-5.6 Sol follow-up reached 98.97 RHAE on all public
   levels, but the authors explicitly limit this to public-set saturation and do not
   isolate verification from the bundled workspace and interfaces.
7. Temporally centered SIGReg and Fast LeWorldModel are mechanistically motivated
   cross-domain candidates. Neither is an accepted architecture component until a
   Tofy-specific controlled test passes.

## 3. Non-negotiable invariants

### 3.1 Evidence provenance

- Only a real environment response creates an `ObservedTransition`.
- A neural rollout creates a `NeuralPrediction`; executable simulation creates an
  `ExecutablePrediction`. Neither type can be inserted into the factual ledger or graph.
- Hidden-goal belief may cite only observed transition IDs as positive or negative
  evidence. Predictions may rank the next experiment but may not update belief as if
  they happened.
- Every real action is followed by observation, ledger append, model mismatch checks,
  belief revision, and replanning. A queued plan never commits multiple remote actions
  without intervening verification.

### 3.2 World-model boundaries

- Learned dynamics remain goal-free: `predict(state, action)` never receives a goal
  hypothesis.
- Goal hypotheses may condition planners and observer predicates, not the neural state
  transition.
- The executable model is a hypothesis about mechanics, not an authority. It is usable
  for multi-step planning only after replay verification against all applicable factual
  transitions.
- The current `q_head` is transition-quality/ranking, not task value. New code and report
  fields must use `trajectory_quality`; do not label it reward, value, or utility.

### 3.3 Evaluation and safety

- Public ARC games remain held-out evaluation. They may not train weights, tune
  thresholds, select checkpoints, or choose architecture after inspection.
- Official/competition mode must run without internet-dependent inference.
- Mutating ARC requests are at-most-once unless a future official protocol explicitly
  provides idempotency or unambiguous reconciliation.
- A lower raw learned-latent MSE is never a promotion criterion without scale,
  non-collapse, action-intervention, copy-forward-normalized, and fixed-anchor gates.
- Do not restore the P1 exact simulator or P1 CLI on `main`.

## 4. Target architecture

```text
real observation
      |
      v
ObservationAdapter ---> FactualLedger ---> StateActionGraph
      |                       |                    |
      v                       +---------+----------+
PerceptionState                        |
      |                                v
      +--> NeuralWorldModelCore --> Candidate predictions + uncertainty
      |                                |
      +--> ExecutableModelBank --> replay-verified predictions
      |                                |
      +--> GoalBelief <----- real evidence only
                                       |
                                       v
                         RiskAwarePlanner / ProbeSelector
                                       |
                              one legal action only
                                       |
                                       v
                           AtMostOnceActionExecutor
                                       |
                              next real observation
```

`NeuralWorldModelCore` remains the learned predictive core. `ExecutableModelBank`
provides a second, inspectable hypothesis representation. The planner may use both,
but the factual ledger is the sole source of historical truth.

The planner must not depend directly on `WorldModel`. Define a replaceable provider
seam so neural centrality is measurable rather than assumed:

```rust
pub trait TransitionProvider {
    fn provider_id(&self) -> &str;
    fn predict(&mut self, request: TransitionRequest<'_>) -> Result<TransitionProposal>;
}
```

Implementations are `NeuralTransitionProvider`, `ExecutableTransitionProvider`, and
`GraphTransitionProvider`; tests use `NoPredictionProvider`. Every action decision
report records which providers contributed and their individual scores. Frozen
baselines must include graph/executable planning with the neural provider disabled and
the neural provider's incremental gain under the same action/compute budget.

## 5. Code layout

Keep existing P2 training code operational while adding narrow modules. Do not place
agent state or public ARC dependencies in `src/p2/train.rs`.

```text
src/p2/
  model.rs                  # neural encoder, recursive one-step model, direct prefix model
  train.rs                  # synthetic-only training, TC-SIGReg, direct-prefix loss
  eval.rs                   # synthetic/recording evaluation and raw episode rows
  representation.rs         # NEW: seam capture and representation summaries
  prefix.rs                 # NEW: action-prefix tensors/encoder/predictor helpers
  arc3_live.rs              # HTTP adapter and legacy frozen-checkpoint evaluation
  arc3_agent/
    mod.rs                  # public facade; no training imports
    types.rs                # observed/predicted provenance types
    memory.rs               # append-only ledger and exact state-action graph
    perception.rs           # palette/component/diff features and hashes
    executable.rs           # executable-model workspace contract
    verifier.rs             # exact replay and planner verification
    reasoner.rs             # swappable local/development reasoning backend
    belief.rs               # goal/mechanics hypotheses with factual citations
    planner.rs              # probe/pursuit selection and action cost/risk
    controller.rs           # one-action closed-loop state machine
    report.rs               # durable schemas and atomic writers
scripts/
  p2_tc_sigreg_ab.sh        # NEW only after Phase 1 code passes local validation
  p2_direct_prefix_ab.sh    # NEW only after Phase 2 code passes local validation
```

If a phase does not require a new module, prefer changing the existing owner rather
than introducing a forwarding wrapper.

## 6. Phase 0: measurement correctness and action safety

Phase 0 changes no training objective, model weights, or experiment result. It must
land and re-evaluate existing checkpoints before any new training.

### 6.1 Named representation seams

Add `src/p2/representation.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationSeam {
    EncoderPreRmsPooled,
    EncoderPostRmsPooled,
    EncoderPreRmsSpatial,
    EncoderPostRmsSpatial,
    ActionConditionedInputSpatial,
    RecursionOuterOneSpatial,
    PredictionFinalPooled,
    PredictionFinalSpatial,
    TargetPostRmsPooled,
    TargetPostRmsSpatial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationSeamMetrics {
    pub rows_seen: usize,
    pub rows_used: usize,
    pub dimension: usize,
    pub mean_rms: Option<f64>,
    pub mean_variance: Option<f64>,
    pub effective_rank: Option<f64>,
    pub effective_rank_fraction: Option<f64>,
}

pub type RepresentationSeamMap = BTreeMap<RepresentationSeam, RepresentationSeamMetrics>;
```

Requirements:

- Expose one diagnostic forward path from `WorldModel`; do not make internal trainable
  modules public individually.
- The diagnostic path returns detached tensors for all named seams from one batch.
- `EncoderPostRmsPooled` remains the population used by existing top-level
  `SplitEval.representation` fields so `scripts/p2_ab_gate.py` remains readable.
- Spatial populations flatten each cell as one row. Pooled populations use one row per
  sample. Report the exact row count and feature dimension.
- Apply deterministic row capping after row construction. The sampled index set is a
  pure function of `(eval_seed, seam, rows_seen, cap)` and is independent of physical
  evaluation batch size.
- Summaries must operate in F64 on the host after one bounded device transfer. Do not
  issue one GPU synchronization per scalar.
- Add `representation_seams` to a new `p2.eval_report.v10` schema. Do not silently emit
  v9 with new semantics.

Tests in `src/p2/representation.rs` and `src/p2/eval.rs`:

- a constant matrix fails variance and effective-rank gates;
- orthogonal/diverse rows have high rank fraction;
- pooled/spatial row and dimension accounting is exact;
- row selection is deterministic and batch-partition invariant;
- v10 serialization uses stable snake-case seam keys;
- the existing top-level representation summary equals the post-RMS pooled seam.

### 6.2 Real episode-level rollout rows

Replace `RolloutSextuple` with a named result carrying factual identity:

```rust
#[derive(Debug, Clone)]
struct EpisodeRolloutResult {
    seed: u64,
    episode_id: u64,
    families_through_horizon: Vec<String>,
    horizon: usize,
    open_mse: Option<f64>,
    closed_mse: Option<f64>,
    copy_forward_mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRolloutRow {
    pub schema: String,                 // "p2.episode_rollout.v2"
    pub source: String,                 // synthetic_dynamics | synthetic_planner
    pub seed: u64,
    pub episode_id: u64,
    pub families_through_horizon: Vec<String>,
    pub horizon: usize,                 // 4 | 8 | 16
    pub open_mse: Option<f64>,
    pub closed_mse: Option<f64>,
    pub copy_forward_mse: Option<f64>,
    pub normalized_open_mse: Option<f64>,
}
```

Requirements:

- Preserve the existing real episode grouping `(seed, episode_id)`. A synthetic episode
  may retarget and therefore change family within one trace; `family` is not part of the
  episode identity.
- Emit one row for every actual `(source, seed, episode_id, horizon)` that has enough
  transitions. Never invent episode IDs. `families_through_horizon` is the sorted,
  deduplicated set of transition-family labels contributing from the episode anchor
  through that horizon.
- Sort lexicographically by `source`, `seed`, `episode_id`, `horizon` before writing.
- `normalized_open_mse = open_mse / copy_forward_mse` only when the denominator is
  finite and greater than `1e-8`; otherwise `None`.
- Aggregate `RolloutMetrics` from these rows. The JSONL and report therefore share one
  calculation instead of two paths.
- Write through a unique staging file in the destination directory. Flush and
  `sync_all` the staging file, atomically rename it over the destination, then open and
  `sync_all` the parent directory. On any pre-rename failure, preserve an existing
  destination byte-for-byte and remove the staging file. There is no non-atomic
  fallback.

Tests:

- exact row identities and horizons for a two-episode fixture;
- a family change inside one episode remains one episode and produces the correct
  sorted `families_through_horizon` provenance;
- mixed episode lengths omit unavailable horizons instead of emitting empty rows;
- deterministic ordering independent of input order;
- row means reconcile exactly with `RolloutMetrics` for every horizon/mode;
- aggregate weighting is one weight per emitted episode/horizon row, not per family or
  transition;
- zero, negative, NaN, and infinite copy-forward denominators yield
  `normalized_open_mse = None`;
- repeated evaluation produces byte-identical JSONL;
- a forced write/sync failure preserves an existing final file and cleans staging;
- the successful path synchronizes the parent directory;
- when episode JSONL output is disabled, aggregate report values remain unchanged.

### 6.3 Explicit evaluation modes

Add:

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum EvalMode {
    Full,
    Representation,
    Rollout,
}
```

- `Full` preserves the complete v10 evaluation graph.
- `Representation` runs one-step/action/representation/seam diagnostics and skips
  PTRM, stochastic disagreement, event/Q reports, and rollouts.
- `Rollout` runs open, closed, and copy-forward episode rows plus the minimal encodes
  needed for them; it skips action shuffle, PTRM, and unrelated heads.
- Reports include `mode` and set unavailable fields to `None`; never synthesize zeroes.
- Add `--eval-mode` to `P2EvalArgs` with default `full`.

### 6.4 At-most-once live mutation

Refactor `HttpArcApi`:

```rust
#[derive(Debug, Clone, Copy)]
enum RetryClass {
    IdempotentRead,
    AtMostOnceMutation,
}

#[derive(Debug)]
pub struct AmbiguousMutation {
    pub operation: String,
    pub game_id: Option<String>,
    pub guid: Option<String>,
    pub action: Option<ArcAction>,
    pub cause: String,
}
```

- `GET /api/games` may retry `429`, `5xx`, and transport errors with bounded backoff.
- `ACTION*` sends exactly once. On timeout, connection reset, response-body failure,
  `429`, or `5xx`, return `AmbiguousMutation`; do not submit the action again.
- Apply the same conservative single-send rule to other POST mutations unless official
  documentation explicitly guarantees idempotency for that endpoint.
- `run_public_suite` stops the affected game with `stop_reason="ambiguous_mutation"`,
  preserves the attempted action separately from confirmed trace entries, closes the
  scorecard once if possible, and writes the report.
- Update `docs/P2_ARC3_LIVE_EVAL.md`; remove the claim that all `429`/`5xx` requests are
  retried.

Tests use a deterministic fake transport that counts sends:

- idempotent GET retries and eventually succeeds;
- ACTION transport failure has send count one;
- ACTION `500` has send count one;
- ambiguous attempted action is present in the report but absent from confirmed factual
  transitions;
- non-retryable `4xx` fails once.

### 6.5 Phase-0 acceptance

Run:

```bash
cargo fmt --check
cargo test --lib p2::representation
cargo test --lib p2::eval
cargo test --lib p2::arc3_live
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Then run `representation` and `rollout` modes on the four preserved update-1,000 and
update-2,000 geometry-v2 checkpoints. Publish the exact command and confirm:

- existing aggregate fields reproduce within floating-point tolerance;
- the seam map identifies where variance/rank is first lost;
- JSONL aggregate reconciliation passes;
- no model-quality promotion is claimed.

## 7. Phase 1A: factual memory and verified executable hypothesis models

This track has the strongest direct ARC-AGI-3 system evidence and can proceed after
Phase 0 without waiting for a new neural checkpoint.

### 7.1 Provenance-safe types

In `src/p2/arc3_agent/types.rs`:

```rust
pub struct ObservationId(pub [u8; 32]);
pub struct TransitionId(pub [u8; 32]);

pub struct ObservedState {
    pub id: ObservationId,
    pub game_id: String,
    pub guid: String,
    pub level: u16,
    pub frame: ArcFrame,
    pub status: ArcStatus,
    pub legal_actions: BTreeSet<u8>,
    pub components: Vec<Component>,
}

pub struct ObservedTransition {
    pub id: TransitionId,
    pub from: ObservationId,
    pub action: ArcAction,
    pub to: ObservationId,
    pub level_before: u16,
    pub level_after: u16,
    pub status_after: ArcStatus,
    pub frame_changed: bool,
}

pub struct NeuralPrediction { /* no ObservedTransition conversion */ }
pub struct ExecutablePrediction { /* no ObservedTransition conversion */ }
```

There must be no `From<NeuralPrediction>` or `From<ExecutablePrediction>` for
`ObservedTransition`. `FactualLedger::append` accepts only `ObservedTransition`.

### 7.2 Factual ledger and graph

- Ledger files are append-only JSONL with schema, session ID, monotonically increasing
  sequence, SHA-256 link to the preceding row, and atomic flush after every confirmed
  action.
- Graph nodes use exact settled-frame hash plus public status/level/action mask. A
  perceptual or learned embedding may be stored for search but may not merge exact nodes.
- Edges record confirmed actions and effects. Tested/no-op/dead-action signatures are
  scoped to `(game, level, exact state)` unless a reasoner proposes a broader rule and
  replay verification supports it.
- Imagined edges live in a separate transient search arena and are discarded or
  re-anchored after every observation.

### 7.3 Perception features

`perception.rs` deterministically produces:

- the raw `64x64` palette grid;
- connected components per palette value, including bounding box, area, centroid,
  border contact, and a normalized shape bitmap;
- frame delta against the preceding real observation;
- candidate ACTION6 coordinates from component representatives and a bounded fallback
  grid;
- status-region hypotheses as annotations only. Do not mask pixels out of exact hashes
  until repeated factual evidence proves a region is display-only.

### 7.4 Executable workspace contract

Each game gets a durable workspace containing:

```text
attempts.jsonl             # controller-owned, read-only to reasoner
world_model.md             # concise mechanics and uncertainty
world_model.py             # executable hypothesis
planner.py                 # model-based planner
hypotheses.json            # structured mechanics/goal hypotheses
verification.json          # controller-owned latest verifier result
```

`world_model.py` must expose:

```python
def reconstruct(initial_observation: dict) -> object: ...
def step(state: object, action: dict) -> tuple[object, str]: ...
def render(state: object) -> list[list[int]]: ...       # exactly 64x64, values 0..15
```

`planner.py` must expose:

```python
def plan(state: object, goal_id: str, legal_actions: list[dict], budget: int) -> list[dict]: ...
```

Arbitrary reasoner-generated Python is untrusted. It may run only through this fail-closed
interface:

```rust
pub trait ExecutableSandbox {
    fn run(&self, request: SandboxRequest) -> Result<SandboxResult>;
}

pub struct BubblewrapSandbox { /* pinned rootfs digest and resource policy */ }
```

The production implementation is Linux `bubblewrap` with a versioned, read-only rootfs
containing only the fixed Python interpreter and pinned dependencies; `pip`, package
installation, compilers, and network clients are absent. The controller copies only
validated regular files into a fresh execution directory, rejects symlinks and path
escapes, and binds that sanitized directory read-only at `/work`. Launch with the
equivalent of:

```text
bwrap --unshare-all --new-session --die-with-parent \
  --ro-bind <pinned-rootfs> / --ro-bind <sanitized-run-dir> /work \
  --tmpfs /tmp --proc /proc --dev /dev --chdir /work --clearenv \
  --setenv PATH /usr/bin --setenv PYTHONHASHSEED 0 -- <fixed-python-command>
```

The outer launcher also applies `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NPROC`,
`RLIMIT_FSIZE`, and `RLIMIT_NOFILE`; enforces a wall-clock deadline; starts a separate
process group/session; kills and reaps the whole tree on timeout; and caps captured
stdout/stderr. No host environment variable is inherited. Controller-owned attempts,
observations, scorecards, and verification records are mounted read-only or passed as
bounded serialized input; the subprocess cannot edit them. If bubblewrap, user
namespaces, the pinned rootfs digest, or any limit cannot be established, execution is
reported as unsupported and no generated Python is run. There is no insecure fallback.

Adversarial tests must cover `../` traversal, absolute paths, symlink escape, network
access, subprocess creation, fork bombs, oversized files/output, memory exhaustion,
wall/CPU timeout, signal handling, and attempted environment-secret reads. They must
also prove an existing controller-owned file is unchanged after every case.

### 7.5 Replay verifier

For every recorded attempt applicable to a model version:

1. reconstruct the initial state;
2. require exact `64x64` settled-frame reproduction;
3. replay every recorded action;
4. require exact public status after every step;
5. for each nonterminal step, require exact settled-frame reproduction;
6. record the first mismatch with transition ID and a bounded pixel diff;
7. on completed levels, require the planner to reach `WIN`/level completion in the
   executable model from the recorded initial state.

```rust
pub enum VerificationStatus {
    Verified {
        attempts: usize,
        transitions: usize,
        planner_levels: usize,
    },
    Rejected { first_mismatch: VerificationMismatch },
    Incomplete { modeling_debt: Vec<ModelingDebt> },
}
```

Only `Verified` models may authorize multi-step model planning. `Incomplete` and
`Rejected` models may suggest a single safe probe, subject to the planner's risk rule.

### 7.6 Phase-1A acceptance

- ledger hash-chain corruption is detected;
- predicted types cannot enter factual memory (compile-fail or API visibility test);
- exact graph identity is deterministic;
- verifier catches wrong frame, wrong status, missing latent state, and planner failure;
- a known deterministic synthetic episode replays exactly;
- one-step online plan execution aborts immediately on model/real mismatch;
- no API key or environment secret reaches the Python subprocess.

## 8. Phase 1B: temporally centered SIGReg pilot

This track repairs the neural core and runs independently from Phase 1A.

### 8.1 Configuration

Add serialized enums and include every field in `TrainingContract`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum SigregTarget {
    Marginal,
    TemporalResidual,
}

pub struct TrainConfig {
    // existing fields...
    pub sigreg_target: SigregTarget,
    pub sigreg_temporal_window: usize, // default 8; ignored for Marginal
}
```

Validation:

- `TemporalResidual` requires `window >= 2`;
- ordered windows may not cross `(seed, episode_id, family)`;
- transition indices must be contiguous;
- exact resume rejects any target/window change.

### 8.2 Matched data path

- Build a dedicated ordered representation trace for both control and treatment.
- Control and treatment consume the same frames, rows, cap, physical batch, optimizer
  schedule, and number of SIGReg calls.
- Encode a window to `T x B x C x H x W` (or an equivalent documented layout).
- For `TemporalResidual`, compute `mean` over `T` for each episode and spatial cell,
  then `residual_t = z_t - mean`.
- Apply the unchanged current control geometry after centering: post-RMS, 2x2 spatial
  pooling, then cell rows. The first pilot changes only the regularization target.
- `Marginal` applies SIGReg to the same ordered-window latents without centering.
- Never use `window=1`; it is identically zero after centering.

### 8.3 Pilot contract

Fresh seed 1, paired control/treatment, checkpoints/evaluation at 250, 500, 750, and
1,000 updates. Serialize arms; do not run concurrent evaluator/trainer processes.
Measure the largest stable A40 physical batch, then minimize accumulation while keeping
the intended effective batch and optimizer schedule. Record the pair.

Promote to seeds 2 and 3 only when update 1,000 satisfies every gate:

- post-RMS pooled variance `>= 1e-4`;
- post-RMS pooled effective-rank fraction `>= 0.10`;
- aggregate and `random_one_step` shuffled/true ratios `>= 1.10`, each 95% lower bound
  above `1.0`;
- changed-transition prediction at least 10% better than copy-forward;
- normalized H8 no more than 25% worse than paired control;
- no named seam shows a new hidden collapse;
- all episode-row reconciliation and finite-value checks pass.

Immediate stop: treatment action ratio returns to approximately `1.0`, learned changed
transition is worse than copy-forward, or any variance/rank hard gate fails at update
1,000. Do not run seeds 2/3 merely because raw MSE falls.

## 9. Phase 2: true direct action-prefix prediction

Begin only with a non-collapsed Phase-1B representation.

### 9.1 API and tensor contract

In `src/p2/prefix.rs`:

```rust
pub struct ActionPrefixBatch {
    pub action_ids: Tensor,      // U32 [B,H]
    pub action_coords: Tensor,   // F32 [B,H,2]
    pub valid: Tensor,           // U8/F32 [B,H]
}

pub struct PrefixPrediction {
    pub latents: Tensor,         // F32 [B,H,C,8,8]
    pub reliability_logits: Tensor, // F32 [B,H,1]
}

impl WorldModel {
    pub fn predict_action_prefixes(
        &self,
        anchor: &Tensor,         // real encoded state [B,C,8,8]
        prefixes: &ActionPrefixBatch,
    ) -> Result<PrefixPrediction>;
}
```

Requirements:

- All horizon outputs are conditioned on the same real `anchor`.
- No predicted horizon latent is an input to another horizon prediction.
- A causal prefix encoder prevents future actions from affecting earlier outputs.
- Action ID, coordinate, position, and validity mask are encoded per step.
- The predictor produces spatial deltas; it must not broadcast one `B x C` delta over
  every cell.
- Implement horizons in one batched/parallel call. A loop that invokes the one-step
  predictor H times is not accepted.
- Keep the current recursive one-step model as a baseline and for single-step fallback.

The first implementation should use a small causal prefix encoder (maximum H=16) and a
spatial FiLM/residual predictor. Exact layer count and width are config fields and part
of the checkpoint/training contract. Do not add a large transformer before the small
version has a measured failure.

### 9.2 Loss and reporting

Replace the semantic role of `prefix_multi_horizon_loss`:

- one forward call produces horizons `1..H`;
- targets are real encoded frames from the same ordered episode;
- report separate H1/H2/H4/H8/H16 losses and a weighted total;
- retain an explicitly named `autoregressive_prefix_baseline` evaluator for comparison;
- causal-leakage unit test: changing actions after horizon `k` must leave predictions
  through `k` bit-identical on CPU;
- anchor test: poison an earlier predicted output and prove later direct outputs are
  unchanged.

### 9.3 Promotion gates

Across three seeds:

- one-step normalized error no more than 5% worse than the accepted neural baseline;
- H8 normalized error at least 30% better than autoregressive rollout;
- open H8 no more than 2x closed H8;
- H16 finite and p95/median normalized error no greater than 5;
- seed-stratified 95% interval for H8 improvement excludes zero;
- measured candidate-planning latency improves and synthetic planner completion does
  not regress.

Do not promote from latent MSE alone.

## 10. Phase 3: reasoner, goal belief, and risk-aware planning

### 10.1 Reasoner interface

```rust
pub trait ReasonerBackend {
    fn propose(&mut self, request: ReasonerRequest) -> Result<ReasonerProposal>;
}

pub struct ReasonerRequest {
    pub recent_observations: Vec<ObservedStateSummary>,
    pub graph_summary: GraphSummary,
    pub verification: VerificationStatus,
    pub hypotheses: Vec<HypothesisSummary>,
    pub legal_actions: Vec<ArcAction>,
    pub action_budget_remaining: usize,
}

pub struct ReasonerProposal {
    pub mechanics_updates: Vec<MechanicsHypothesis>,
    pub goal_updates: Vec<GoalHypothesisProposal>,
    pub workspace_patch: Option<WorkspacePatch>,
    pub candidate_actions: Vec<ActionProposal>,
}
```

Backends:

- `CodexCliReasoner`: development/public-harness research only; runs in the per-game
  workspace and is never enabled in official offline mode.
- `LocalOpenAiCompatibleReasoner`: talks only to a loopback local server and is the
  intended competition backend.
- `ScriptedReasoner`: deterministic tests and graph-only baseline.

The backend never receives API keys and never calls `ArcApi`. The controller validates
workspace patches, runs verification, and selects/submits actions.

### 10.2 Goal-hypothesis evidence ledger (experimental)

```rust
pub struct GoalHypothesis {
    pub id: String,
    pub predicate: GoalPredicateRef,
    pub status: HypothesisStatus,
    pub supporting: BTreeSet<TransitionId>,
    pub contradicting: BTreeSet<TransitionId>,
    pub irreversible_risk: f64,
    pub supersedes: BTreeSet<String>,
}

pub enum HypothesisStatus { Proposed, Supported, Contradicted, Confirmed }

pub struct GoalHypothesisSet {
    pub hypotheses: BTreeMap<String, GoalHypothesis>,
    pub unknown_goal_active: bool,
}
```

- `GoalPredicateRef` names a deterministic predicate exported by a replay-verified
  executable model version. The predicate returns a typed Boolean/result score and is
  separately exercised by verifier fixtures; free-form strings are never executed.
- Reasoners may propose hypotheses, but only the controller may change status, and each
  `Supported`, `Contradicted`, or `Confirmed` transition must cite a confirmed real
  transition ID present in the factual ledger. Predictions never supply evidence.
- `unknown_goal_active` starts true and remains available until the real environment
  confirms completion. Do not force probabilities over an incomplete hypothesis set.
- Enter pursuit only when exactly one non-contradicted supported/confirmed hypothesis
  has a replay-verified plan and its next action passes the risk rule. Otherwise probe.
  Return to probe immediately after contrary evidence, model mismatch, or verifier
  invalidation.
- Tests cover citation referential integrity, duplicate/canonical predicate IDs,
  contradictory evidence, supersession cycles, unknown-goal preservation, deterministic
  replay, and rejection of prediction IDs as evidence.

Calibrated posterior weights and information gain are deferred until a likelihood model
and calibration protocol are specified and validated. This first implementation is an
auditable evidence ledger, not a probabilistic belief claim.

### 10.3 Planner contract

Planner modes:

```rust
pub enum PlannerMode { Probe, Pursuit, Recovery }
```

Candidate score:

```text
verified_goal_progress
+ beta_hypothesis_discrimination
+ gamma_unvisited_frontier
- lambda_failure_risk
- mu_model_disagreement
- one_action_cost
```

- `hypothesis_discrimination` is a bounded count/weight of currently viable hypotheses
  for which verified providers predict different observable outcomes; it is zero when
  no such verified comparison exists. It is not entropy or information gain.
- Terms unavailable from verified/factual evidence are omitted, not fabricated.
- Known illegal actions are impossible candidates.
- Known exact-state no-ops/dead actions are removed unless a changed context invalidates
  the signature.
- High-risk actions require either verified goal progress or a uniquely valuable
  falsification result.
- After every real action, discard stale imagined suffixes and replan.

### 10.4 Baselines and gates

Evaluate under identical action/compute budgets:

1. random legal;
2. deterministic factual graph only;
3. graph plus neural novelty/error signals;
4. graph plus textual hypotheses;
5. full verified executable hybrid;
6. oracle synthetic simulator upper bound.

Report completion, actions, irreversible failures, resets, model mismatches, verifier
coverage, reasoner calls/tokens, wall time, and official RHAE only when supplied by the
official scorecard. Require multi-seed gains over graph-only on frozen synthetic tasks
before public evaluation.

## 11. Deferred candidates

These are not part of the first implementation:

- resumable `(y,z)` rolling-horizon caches: implement only after direct prefixes and
  action conditioning pass; stale caches require versioning, repair, and reset;
- group-action losses: first add evaluation probes, and train only on synthetic action
  families with proven identity/inverse/composition semantics;
- full object-slot replacement: begin with deterministic components and action-local
  features; the rejected spatial SIGReg arm is not evidence for a slot architecture;
- independent bootstrap neural ensembles: current `ensemble_members` is same-model PTRM
  noise and must be relabeled `stochastic_members`; train independent members only if
  disagreement adds calibrated decision value beyond reliability and verifier mismatch;
- test-time weight updates on public/private games: prohibited until the evaluation
  contract explicitly allows them and a no-leakage protocol exists.
- probabilistic goal-belief weights/information gain: require an explicit likelihood
  model, calibration set, and frozen promotion thresholds; the initial evidence ledger
  intentionally avoids pseudo-posteriors.

## 12. Schema and compatibility policy

- Bump evaluation report to `p2.eval_report.v10` for seam maps/modes.
- Bump episode rows to `p2.episode_rollout.v2`.
- Introduce `p2.arc3_agent_report.v1`, `p2.factual_ledger.v1`, and
  `p2.executable_verification.v1`.
- Old checkpoints remain loadable only for Phase-0 evaluation. New TC-SIGReg or prefix
  architecture checkpoints use a new model/training contract and are not resumed from
  old optimizer state.
- Do not add compatibility aliases that mislabel stochastic PTRM members as a bootstrap
  ensemble or trajectory quality as task value.

## 13. Implementation order for delegated agents

Each delegated coding task must use a dedicated branch/worktree, commit its result, and
stop. It must not start GPU training unless its task explicitly includes a reviewed
experiment command.

1. **P0-A:** real episode JSONL rows and aggregate reconciliation only. This is a
   measurement-integrity prerequisite, not an architecture decision test.
2. **P0-B:** named representation seams and `EvalMode` only.
3. **P0-C:** at-most-once live mutation only.
4. Primary agent integrates and runs all Phase-0 validation/re-evaluation.
5. **P1A-A:** provenance types, ledger, and graph.
6. **P1A-B:** executable workspace and replay verifier.
7. **P1B-A:** TC-SIGReg ordered-window implementation and unit tests.
8. Primary agent reviews/preregisters the paired TC-SIGReg pilot; only then may the pod
   run training.
9. **P2-A:** true direct-prefix module/loss/tests.
10. **P3-A:** reasoner interface, belief state, and planner after lower layers pass.

The immediate pod handoff is task 1 only. It may edit `src/p2/eval.rs` and focused
tests, plus schema documentation if needed. It must not edit training, model
architecture, scripts, live ARC code, results documents, or run experiments.

## 14. Immediate handoff acceptance criteria: P0-A

The coding agent must:

1. replace tuple rollout results with a named per-episode result preserving real
   `(seed, episode_id)` identity and recording changing family labels separately as
   sorted `families_through_horizon` provenance;
2. emit actual H4/H8/H16 rows for both synthetic sources;
3. derive aggregate rollout metrics from the same rows;
4. add deterministic ordering, normalized error, staging write, `sync_all`, and atomic
   rename;
5. bump the episode-row schema without changing `p2.eval_report.v9` yet;
6. add the focused tests enumerated in section 6.2, including retargeting families,
   missing horizons, invalid normalization denominators, aggregate weighting, disabled
   JSONL parity, prior-file preservation, and parent-directory durability;
7. run formatting, focused tests, full tests, and clippy with warnings denied;
8. commit on its branch and stop without training or touching existing run artifacts.

P0-A repairs evidence integrity. It must not be used to choose an architecture. P0-B's
named seam diagnostics are the first Phase-0 result allowed to falsify a representation
decision.

## 15. Primary sources

- [ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621), 2026-03-24.
- [Official ARC-AGI-3 methodology](https://docs.arcprize.org/methodology), retrieved
  2026-08-09.
- [Official 2026 competition](https://arcprize.org/competitions/2026/arc-agi-3),
  retrieved 2026-08-09.
- [Official Milestone 1 report](https://arcprize.org/blog/arc-prize-2026-milestone-1),
  2026-07-06.
- [Do Coding Agents Need Executable World Models, Simplification, and Verification to
  Solve ARC-AGI-3?](https://arxiv.org/abs/2607.15439), 2026-07-16, and
  [author repository](https://github.com/astroseger/arc-3-agents-baseline1).
- [Executable World Models for ARC-AGI-3 in the Era of Coding Agents](https://arxiv.org/abs/2605.05138),
  2026-05-06.
- [Graph-Based Exploration for ARC-AGI-3](https://arxiv.org/abs/2512.24156),
  2025-12-30, and [author code](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore).
- [Temporally Centered SIGReg](https://arxiv.org/abs/2607.26924), 2026-07-29.
- [Fast LeWorldModel](https://arxiv.org/abs/2606.26217), 2026-06-24.
- [World Models as Group Actions](https://arxiv.org/abs/2605.24578), 2026-05-23.
