# ADR 0005: World-core v6 — adaptation-first training and test-time adaptation contract

- Status: proposed
- Date: 2026-09-03
- Amends: [ADR 0003](0003-world-core-v5-foundation-v2.md) §1 data contract, §2 model,
  §3 objective, §6 evaluation, §7 deferred list; makes [ADR 0004](0004-latent-planning-contract.md)
  Phase C ("prequential fast adapters") concrete.
- Basis: research run `~/Research/_runs/2026-09-03T033950Z-tofy-zero-shot-transfer-training-design`
  (synthesis.md and seven verified findings), plus the local audit
  `findings/local-generator-vs-interface-audit.md` against worktree `7eddfd21`.

## Decision

Tofy stops treating an unseen ARC-AGI-3 game as an in-distribution query and
starts treating it as a task to be adapted to. Version 6 of the world core:

1. **Fixes the interface** so that the tuple the live policy actually sends
   is a tuple the model was trained on (whole-frame content, free background
   colour, no reserved status row, unknown operator with an all-zero goal).
2. **Trains for adaptation**: every training row may carry a context window of
   the most recent factual transitions from the same hidden-rule episode, the
   generator is made *mutually exclusive* (identical frames, different rules),
   and histories are exploratory learning histories spanning level boundaries.
3. **Adds a context channel to the model** (CaDM-style in-context adaptation,
   no gradients) and names a **fast-weight subset** for gradient adaptation.
4. **Adds a bounded per-level test-time adaptation loop** at inference, with
   snapshot/revert safeguards, that never touches the pretrained checkpoint.
5. **Preregisters cheap falsifiers** that must pass before GPU-hours are spent.

## Bounded claim and honest expectation

The evidence supports each mechanism separately in other domains; no primary
source shows a ~0.5M-parameter neural world model adapting profitably within
the 5x-human action budget on unseen ARC-AGI-3 games. The Twin and Tycho
results say goal inference, not dynamics accuracy, is the score bottleneck.
This ADR therefore claims only: (a) the live query becomes in-distribution,
(b) the model gains a measurable ability to use in-episode evidence, (c) the
adaptation loop is safe (cannot make the frozen model worse than the prior).
It does not claim a hidden-set score. The defensible target remains the
Kaggle milestone regime; the grand prize is not a projected outcome.

## Non-negotiable invariants (retained from ADR 0003)

- Training data is synthetic only. Public ARC-AGI-3 levels are evaluation
  only, enforced by `training_source_cannot_depend_on_live_or_recording_modules`.
  Test-time adaptation at evaluation time uses only the current game's own
  factual transitions, and its adapted weights are discarded at the end of
  the game; the pretrained checkpoint is never updated from public games.
- Deterministic seeds; manifested, hashed checkpoints; fail-closed loaders.
- Predictions never enter Factual Memory; adaptation trains on factual
  transitions only, never on imagined rollouts.
- **No state crosses a game boundary** (owner decision, 2026-09-03). Fast
  weights, context windows, factual buffers, the observed-state graph and any
  calibration statistics are per game and are discarded when the game ends.
  Experience across games lives only in the pretrained checkpoint. The Kaggle
  rules would permit cross-environment persistence; Tofy does not use it,
  because games differ enough that carried state is judged more likely to
  mislead than to help, and no evidence shows a benefit at this scale.

## 1. Interface contract v6

All items apply to `world_core_v6` models and the `v6` generator; legacy
model flags are unchanged so old checkpoints remain loadable.

1.1 **Whole-frame content.** Every pixel of the 64x64 frame is board content.
The content mask of a v6 row is all ones; `ContentRect` survives only as
augmentation geometry (D4 conjugation of coordinates) and as a per-source
metric stratum. Losses, exactness metrics, no-op labels, board effects and
outcome equivalence are computed over all 4096 pixels.

1.2 **Free background colour.** The exact simulator keeps semantic index 0 for
EMPTY; rendering maps it through the colour permutation like every other
index. The v6 colour permutation is a uniformly random permutation of all
16 colours. Padding outside the native board is rendered with the
permuted EMPTY colour, so live frames with arbitrary backgrounds and
synthetic frames are the same distribution. No model path may treat
rendered index 0 specially (`embed_frames` status shortcut, decoder priors,
`dominant_color` background inference in the live driver).

1.3 **No reserved status row.** The v6 generator paints no synthetic budget
bar; `V5_PLAYFIELD_HEIGHT` is not used by v6 layouts (content may occupy row
63). `embed_frames` does not replace row 63 for `world_core_v6`. The live
tried-action key hashes all 64 rows; ACTION6 proposals may target row 63.

1.4 **Live query tuple has training support.** v6 rows always use the
UNKNOWN operator-conditioning token with neutral colours. Rule identity is
conveyed only by the context window (1.5) and never by conditioning.
The 30% goal dropout is retained, so (all-zero goal, UNKNOWN) is a
first-class training tuple. The `operator_conditioning_proj` parameters are
retained for checkpoint compatibility and receive only UNKNOWN in v6.

1.5 **Context window.** A v6 row carries `context: Vec<ContextTransition>`,
`0 <= len <= CONTEXT_WINDOW_MAX = 16`, each `(current, action, next)` frame
triple drawn from the same `(seed, meta_episode_id)` episode, strictly
earlier in chronological order than the row, under the same D4/colour
augmentation as the row. Context may cross level boundaries within the
episode. The live policy fills the context with the most recent factual
transitions from Factual Memory under `--context-scope {level,game}`
(default `game`, so the live window crosses level boundaries exactly as the
training rows' windows do; `level` restricts it to the current level). The
scope is a Channel A knob, independent of `--adapt` / `--adapt-carry`; a
game boundary always empties Factual Memory (§6.3).

1.6 **Available actions.** Rows record an 8-bit `available_actions` mask
(RESET, ACTION1..7). The Rust generator emits all-available; ARCEngine shards
emit the game's mask; the live driver masks proposals by it.

## 2. Data contract v6

2.1 **Streams.** ADR 0003 §1.1 streams are retained, rendered under §1
rules, as the "legacy" mixture. A new stream `LearningHistories` is added.
Schedule (fraction of physical rows): `LearningHistories` ramps linearly
from 0.25 at progress 0 to 0.50 at progress 1; legacy streams share the
remainder in their ADR 0003 ratios.

2.2 **Learning histories.** A `LearningHistories` unit is one `MetaEpisode`
with `levels in {2,3,4}` and a stable hidden rule (operator family + colour
bindings + goal family). Its transitions come from an epsilon-decaying
policy over the existing scripted solvers: `epsilon = 1.0` at the first
transition of the episode decaying linearly to `0.2` at the last, so early
context is exploratory and late context is competent (Algorithm
Distillation shape: histories that improve, not expert data). Each emitted
row is the transition at index `t` with `context = transitions[t-K..t]`,
`K ~ Uniform{0..16}` with `P(K=0) >= 0.10` so the no-context prior keeps
training. Goal-satisfaction, failure and exhausted labels follow ADR 0003.

2.3 **Mutual exclusivity (the non-negotiable data property).** The hidden
rule must not be inferable from a single frame. The generator enforces this
by construction: every meta-episode is emitted together with a **twin** that
shares the byte-identical level-0 initial frame and initial policy prefix
but is bound to a different hidden rule (different operator family and/or
different goal family and/or different colour binding), so identical frames
map to different next frames across the two episodes. The goal family is a
seeded draw independent of `episode_id` (the `episode_id % 6` shortcut is
removed). The `census_rule_identifiability` machinery is extended with a
`single_frame_rule_identifiable` count that must be 0 on the twin pairs.

2.4 **Provenance.** `TransitionProvenance` gains `rule_id: u64` (hash of the
hidden rule), `level_index: u16`, `available_actions: u8`, and
`context_len: u8`. `TransitionSample` gains `context: Vec<ContextTransition>`
(serde default empty).

2.5 **Counterfactual sidecars** for gate 2 are retained for every stream.

2.6 **ARCEngine shard stream (weight 0 in this ADR).** A `SyntheticShards`
stream may load safetensors shards produced by `python/tofy_arc3/synth`
(games authored against `arcengine.ARCBaseGame`, the same engine Kaggle
runs). Shard contract: tensors `frames u8[N,64,64]`, `next_frames u8[N,64,64]`,
`actions u8[N,3]` (id, x, y; 255 = none), `available_actions u8[N]`,
`episode u32[N]`, `level u16[N]`, `transition_index u32[N]`,
`rule_id u64[N]` (as two u32 halves `rule_id_lo/hi`), `level_completed u8[N]`,
`game_over u8[N]`; sidecar `manifest.json` with `source: "tofy_synth_arcengine"`,
generator revision, seed, rule census, and a `public_game_ids_excluded`
attestation. The loader rejects any manifest whose game ids intersect the
public game list. The stream weight stays 0 until the shard generator passes
the memorization diagnostic (§5.1) on its own held-out twins.

## 3. Model contract v6 (`world_core_v6`)

3.1 v6 is v5 plus a **context channel**. Each context transition is encoded
by the shared frame encoder applied to `current` and `next`, the action
embedding (with spatial coordinate field for ACTION6), and the pooled latent
difference; a small MLP maps the concatenation to `hidden_dim`. The `K`
embeddings are aggregated by mean pooling plus the last element
(order-aware summary), giving `c in R^hidden`; `K = 0` gives `c = 0`.
`c` enters the dynamics block through **context FiLM**
(`context_film_gamma`, `context_film_beta`, zero-initialised so the v5
computation is recovered exactly at init) applied next to action FiLM.

3.2 Parameter budget: <= 650k parameters total (v5 is ~467k).

3.5 **Recursion depth 3x3** (owner decision, 2026-09-03; arithmetic corrected
2026-09-04 after audit). v6 applies `inner_steps = outer_steps = 3` instead of
v5's 2x2. Each outer step runs `inner_steps + 1` applications of the residual
block (`deep_step`), and each block holds two 3x3 convolutions, so:

| depth | blocks | 3x3 convs | receptive field on the 16x16 latent grid |
|---|---|---|---|
| 2x2 (v5) | 6 | 12 | 25 cells |
| 3x3 (v6) | 12 | 24 | 49 cells |

The earlier text (4 vs 9 blocks, 17 vs 25 cells) was wrong: 2x2 already
covers the grid; 3x3 doubles the dynamics compute (measured +10% wall clock on
the host-bound local step) for margin that is not evidenced. RecurTrace
(2026, in the 2026-09-04 literature map) reports that extra recurrent loops
can hurt. The depth is therefore an owner choice under test, not a
receptive-field necessity; the preregistered 2x2-vs-3x3 ablation is the
arbiter. v5 keeps 2x2 so legacy checkpoints stay reproducible.

## 4. Objective v6

ADR 0003 §3 is retained unchanged in form; every loss is evaluated over
whole-frame content (§1.1). Context rows use the same losses with the
context supplied. No inner-loop meta-objective is added (Miranda et al.:
effect size < 0.2 over multi-task pretraining; deferred, §8).

## 5. Preregistered gates and falsifiers (must run before pod GPU-hours)

5.1 **Memorization diagnostic** (Yin et al. 2020). On 512 held-out twin-pair
meta-episodes, measure changed-exact on the row after the context with
`K = 0` versus `K = 16`. Promotion of the data contract requires
`delta >= 0.05` absolute by the first evaluation after step 4096 on a local
run; a smaller delta is a **data** failure (the generator is not mutually
exclusive) and blocks any pod run.

5.2 **Adaptation falsifier.** Channel A (context only) vs Channel A+B
(context plus fast-weight updates) on held-out synthetic meta-episodes,
prequential: adapt on transitions `1..t`, score `t+1..t+4`. Channel B is
promoted only if it improves prequential changed-exact by >= 0.02 absolute
AND the adapted-then-frozen model is not worse than the prior on the same
rows (Tempora failure mode). Otherwise Channel B ships disabled.

5.3 **Residual-vs-reliability AUROC** on the latest local v5 checkpoint:
one-step residual on real ARC frames (toolkit local recordings, evaluation
only) versus in-distribution synthetic frames. If residual separates and
the reliability head does not, Phase A trust uses residual-derived
calibration until a v6 checkpoint exists.

5.4 Existing ADR 0003 §5 gates and the v6 gate-policy schema are retained.

## 6. Test-time adaptation contract (`src/p2/adaptation.rs`)

6.1 **Channel A is always on** for v6 policies: the live driver keeps the
last `CONTEXT_WINDOW_MAX` factual transitions of the `--context-scope`
(§1.5; default the current game, `level` the current level) and passes them
as context to every model call, including Phase A screening and
verification. The scope is recorded per decision (`context_scope`) next to
`context_len`.

6.2 **Channel B (`--adapt`)** runs after every observed factual transition
once the current level has >= 8 transitions:
- optimizer AdamW on `FAST_WEIGHT_PREFIXES` only, lr `1e-4`, weight decay 0,
  global grad-norm clip 1.0, no warm-up;
- at most 4 gradient steps per new transition; cumulative steps per level
  <= 8 x (number of unique transitions in the level);
- batch `min(32, buffer)` drawn by reservoir sampling from the level-tagged
  factual buffer (the buffer persists across levels; fast weights reset to
  theta_0 at every level boundary in the default arm; the preregistered
  `carry` arm keeps them; each stored row's own context follows
  `--context-scope`, not the arm);
- loss = the ADR 0003 exact-decoder next-frame loss on factual rows with
  the row's own context, plus L2-SP `1e-3 * ||theta - theta_0||^2` on the
  fast subset;
- prequential guard: loss on the newest 4 transitions is measured before
  and after each update; two consecutive worsenings revert to theta_best;
- collapse guard: an update whose grad-norm exceeds 3x the running mean
  is skipped (SAR precursor);
- goal/terminal/reliability readouts consumed by Phase A trust and by the
  greedy scorer are computed from the prior weights (theta_0) end to end: the
  adapter exposes `with_prior_weights` (a shared `PriorWeights` handle with
  theta_0 cached on device), and the policies swap the fast subset back to
  theta_0 bitwise for a second encode + forward whose q, reliability, no-op
  and per-goal event heads are the only ones read; Phase A chains these
  readouts on a theta_0 latent (`PhaseALatent::prior`) so horizon-2
  verification never reads adapted dynamics either. The adapted weights are
  restored after every readout (also on error) and serve only the decoded
  next frame, the predicted effect and the search. An un-drifted model skips
  the swap. Phase A calibration is unchanged by adaptation and must be fitted
  from synthetic held-out data only (`p2-eval --emit-phase-a-calibration`,
  `source: synthetic_holdout`; any other declared source fails closed at load);
- every update, skip and revert is recorded in `ActionDecision` telemetry.

6.3 Adapted weights are discarded at the end of a game. Nothing here
modifies a checkpoint on disk.

## 7. Evaluation additions

- `p2-eval` reports v6 metrics stratified by `context_len in {0, 1-4, 5-16}`.
- `p2-arc3-live-eval` / bridge report per-level adaptation telemetry
  (updates, skips, reverts, prequential loss before/after).
- The memorization diagnostic and adaptation falsifier are `p2-eval`
  modes with fixed seeds and are `selection_only` until a fresh population
  confirms them.

## 8. Explicitly deferred

MAML/MAML++ inner loop and multi-step loss; attention (Transformer-XL)
memory over the context; mixture-of-adapted-models (MOLe); ARCEngine
shard stream weight > 0; ADR 0004 Phase B heads; anything that changes the
19-dim goal vector or the six goal-family slots consumed by Phase A.

## 9. Consequences

- A v6 generator is trajectory-changing: no exact resume from any v5 run.
- Old checkpoints load under their own flags; v6 metrics are not comparable
  to v5 metrics because the content population changes (whole frame).
- The `RuleIdentifiabilityCensus` becomes a hard data gate, not a report.

## Implementation record (2026-09-03, branch `feat/adaptation-v6`)

Implemented across five reviewed branches and merged; full library test suite green at
each merge. Deviations from the text above, recorded so nobody rediscovers them:

- §3.3 fast-weight subset: the first encoder conv is `encoder.patch`, not `encoder.c1`.
  `FAST_WEIGHT_PREFIXES = ["action_film_", "context_film_", "pixel_emb", "encoder.patch",
  "exact_grounding_head"]`. Parameter count: v5 467,665 → v6 571,217 (≤ 650k).
- §3.1 `ContextBatch` frames are U8 (widened at the gather) and only valid slots are encoded
  (packed), for device memory; `encode_state` is not context-conditioned (context acts only
  through dynamics FiLM). Warm start (§3.4) zeroes only `context_film_*`; the context MLP keeps
  its seeded init because zeroing it would make the channel untrainable.
- §2.2 Learning Histories: operators act on frames, not simulator state, so each level is 6
  epsilon-greedy movement rows followed by one realized ACTION5/ACTION6 row (+4 same-state
  counterfactual sidecar rows sharing its Context Window); rule evidence in a context comes from
  earlier levels' operator rows. `LEARNING_HISTORY_STEPS_PER_LEVEL = 6` is a constant.
- §2.3 Twins differ by operator family only (colour binding and goal family shared) so the
  realized operator row diverges; the census records non-divergent pairs (0 of 64 in tests).
- §1.6 `available_actions = 0xFF` only on v6 rows; legacy rows keep 0 for byte-identity.
- §1.1 A v6 model with a legacy 63-row data contract fails closed at loss time; model-free
  eval helpers (`semantic_eval` censuses, shuffled-control outcome compare) keep 63-row
  semantics because no model config is in scope there.
- §6 Channel A and Channel B keep separate `FactualBuffer`s (Arc-shared frames). The context
  batch is rebuilt per physical chunk at inference (no per-decision summary cache yet). Goal
  and terminal heads are frozen under adaptation, but they read latents produced by the
  adapted dynamics; only heads and calibration are guaranteed pristine.
- §5.3 ran on the 2026-08-27 foundation-v2 checkpoint (the s8 model was unreachable):
  residual AUROC 0.905 vs reliability 0.634; the preregistered switch rule was narrowly missed
  on the reliability side (see `docs/research/2026-09-03-v6-local-falsifiers-prereg.md`).
- 2026-09-04 fixes from code review and the first local v6 run: the in-trainer
  gate forward is chunked (32 rows) AND its outputs are detached. Chunking
  alone did not help (measured: identical OOM at step 1024 with 128- and
  32-row chunks) because candle keeps every op's inputs alive as autograd
  history while the output lives, so concatenated chunk outputs pinned all
  chunks' activations at once. With `detach`, a resume across step 1024 peaked
  at 5.7 GB (no spike above training) on the 8 GB laptop GPU;
  twins on the singleton held-out split draw the alternative family from all
  families (the primary keeps the held-out rule); Phase A retrodiction and
  goal evaluation treat row 63 as board content for v6; the §5.1 threshold is
  applied to the `5-16` context-length stratum (prereg amendment) because the
  implemented ablation scores the mixed held-out population, not K=0 vs K=16
  twin pairs.
- §2.6 the ARCEngine shard generator exists (`python/tofy_arc3/synth`, 24 tests) and its
  seed-1 census passes the twin and random-play gates; the Rust `SyntheticShards` loader is
  not yet written (stream weight remains 0).
