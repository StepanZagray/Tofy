# P2 world-model inference surface at `c2ed65dc`

## 1. Heads and forward outputs — `/home/stepan/Coding/Personal/Tofy/src/p2/model.rs`

Latent convention throughout: spatial latent `y`/`z`/`state` is `B×C×8×8` where `C = hidden_dim` (default 128), grid `LATENT_GRID = FRAME_SIDE/PATCH_SIZE = 64/8 = 8` (:255-257). Frames are `B×1×64×64` palette indices or `B×8×64×64` embedded (`PIXEL_EMB_DIM=8`, `PALETTE_SIZE=16`).

| Component | Location | Signature / shape |
|---|---|---|
| Encoder (obs → latent) | `WorldModel::encode_state` :659 | `(&self, frames: &Tensor) -> Result<Tensor>`; `B×1×64×64` → RMS-normed `B×C×8×8`. Pair variant `encode_state_pair` :686, training variant `encode_state_pair_for_training` :697 → `TrainingEncodedPair` (:48) |
| Canonical latent (the `B×C` "single consumed state" in V4) | `canonical_representation` :654 | `(&self, spatial: &Tensor) -> Result<Tensor>`; `B×C×8×8` → `rms_norm(consumer_readout(spatial))` = `B×C` |
| Readout adapter | `/home/stepan/Coding/Personal/Tofy/src/p2/consumer_readout.rs:45` `ConsumerReadout::forward` | `B×C×8×8` → `B×C`; `GlobalMean` (mean pool) or `SpatialQuery` (softmax attention over 64 position-augmented tokens) |
| Dynamics / transition core | `deep_step` :1099, `run_latent_recursion` :1142 | No separate "head": a shared `GridResidualBlock` (:324) recursed `inner_steps` z-updates + 1 y-update per outer step, `outer_steps` times. `y` is RMS-normed and clamped to ±32 each outer step (:1192-1195). Latent-in → latent-out, `B×C×8×8` |
| Action conditioning | `add_action` :841 / `add_action_with_canonical` :853; `encode_x` :985 | `(state B×C×8×8, actions [B] or [B,1] u32, action_coords B×2 in [0,1])` → `x: B×C×8×8` |
| Event head (includes noop) | field :508, built :582, applied in `heads` :1028 | `Linear(hidden_dim*2 → num_events=4)`; input `cat([readout B×C, goal_h B×C])` → `B×4`. Slots: `EVENT_NOOP=0`, `EVENT_GOAL_SATISFIED=1`, `EVENT_GOAL_FAILED=2`, `EVENT_EXHAUSTED=3` (:36-39). **There is no standalone noop head** — noop is channel 0 of the event head. Seams: `event_logits_from` :1042, `event_logits_from_canonical` :1055 |
| Q head | field :510, built :583 | `Linear(hidden_dim → 1)` on the readout → `B×1`. Seams: `q_logit_from_y` :1258, `q_logit_from_canonical` :1065. Documented as a *PTRM trajectory ranking score*, not a value/reward function |
| Reliability head | field :512, built :584 | `Linear(hidden_dim → 1)` → `B×1` (calibrated error prediction, "Phase D"). Seams: `reliability_logit_from_y` :1267, `reliability_logit_from_canonical` :1069 |
| Decoder (latent → pixels) | `exact_gameplay_logits` :632 → `/home/stepan/Coding/Personal/Tofy/src/p2/grounding.rs:61` `ExactPatchGrounding::gameplay_logits` | `Linear(hidden_dim → 8*8*16)` per token; `B×C×8×8` → `B×63×64×16` palette logits (status row dropped). **world_core_v4 only**; otherwise `PatchHistogramGrounding` (training-loss only, no decode API) |
| Prefix / direct one-step delta | `prefix_predict` :1277 | `(&self, state: &Tensor, actions: &Tensor, action_coords: &Tensor) -> Result<Tensor>` → `B×C×8×8`, `rms_norm(state + delta)`. Bypasses recursion |
| Inverse heads (v2 only) | `decode_action_displacement` :1310 | `B×hidden_dim` displacement → `(action_logits B×ACTION_VOCAB, coords B×2 sigmoid)` |
| Goal projection | `project_goal` :995 | `B×goal_dim(19)` → `B×hidden_dim` |

Output structs: `StepOutput` :408 `{y, event_logits: Option, q_logit: Option}`; `ForwardOutput` :415 `{steps: Vec<StepOutput>, y, event_logits, q_logit, reliability_logit, recursion_probes}`; `LatentRecursionOutput` :427 (headless); `PtrmOutput`/`PtrmTrajectory` :456/:442; `RepresentationDiagnosticOutput` :437.

Head attachment is centralized in `heads` :1028 → `(event_logits, q_logit, reliability_logit)` and `attach_heads` :1230. Note in `attach_heads`: event/Q logits are attached **only to the last outer step**; intermediate `StepOutput`s carry `None`.

## 2. One-step imagination (latent-in → latent-out)

Yes, and it is fully recursive — no observation is required after the first encode.

- `WorldModel::forward_from_latent` :1516
  `pub fn forward_from_latent(&self, state: &Tensor, actions: &Tensor, action_coords: &Tensor, goal_features: &Tensor) -> Result<ForwardOutput>`
- `WorldModel::forward_from_latent_with_depth` :1532 (same + `depth: RecursionDepth`)
- Goal-free variant: `WorldModel::predict_latent_with_depth` :1559
  `pub fn predict_latent_with_depth(&self, state: &Tensor, actions: &Tensor, action_coords: &Tensor, depth: RecursionDepth) -> Result<Tensor>`
- Obs-shortcutting variant used by the live policy: `forward_from_encoded_state` :1391 (takes a pre-encoded `cur_state`; `_frames` argument is unused).
- Cheap non-recursive alternative: `prefix_predict` :1277.

`state` in and `output.y` out are both `B×C×8×8` and both RMS-normed, so `y` can be fed straight back in. Confirmed by `eval.rs` open-loop rollout (below) and the test `prefix_rollout_stays_on_unit_rms_latent_support` :2031.

## 3. Play-time action selection

All of it lives in `/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs`. **`eval.rs` contains no policy** — it is teacher-forced metrics only ("synthetic_planner" there is a dataset family name, not a search planner).

- `ModelPolicy::score_candidates` :620 — `fn score_candidates(&self, frame: &ArcFrame, candidates: &[ArcAction]) -> Result<Vec<ActionScore>>`
  - Encodes the current frame once (:623), broadcasts that latent across the candidate batch, and calls `forward_from_encoded_state(..., RecursionDepth::from_config, 0.0, None, RecursionOpts::EVAL)` :651-661.
  - Uses four signals: `sigmoid(q_logit)` :663, `sigmoid(reliability_logit)` :666, `sigmoid(event_logits[:, EVENT_NOOP])` :669, and `latent_mse_per_sample(output.y, state)` :672 as "predicted effect".
  - Scalarization (:678-682), exact:
    ```rust
    let score = 0.25 * f64::from(q[index])
        + 0.30 * f64::from(reliability[index])
        + 0.30 * (1.0 - f64::from(noop[index]))
        + 0.15 * effect_unit;
    ```
    where `effect_unit = e/(1+e)`.
- `impl LivePolicy for ModelPolicy::choose_action` :697 — enumerate → score → penalize already-tried actions by `-1.0` (:722) keyed on `observation_hash` :736 / `action_key` :746, with a reset when all candidates are exhausted (:715-717) → sort descending with deterministic tiebreak (:726-733) → take `.first()` (greedy argmax).
- Candidate enumeration: `enumerate_actions` :755 — all non-ACTION6 available ids, plus ACTION6 at coordinates from `action6_coordinates` :784 (non-background color centroids/grid, capped by `action6_max_candidates`, strided by `action6_grid_stride`).

**Lookahead: none.** Depth is exactly one environment step; the only "depth" is the model's internal recursion (`outer_steps`, default 2). No tree, no beam, no rollout, no re-planning across steps, no goal features (goals are passed as zeros, :647). The file states this explicitly at :38 (`POLICY_LIMITATION`): the checkpoint "has no trained reward/value head. This exploratory policy is not a hidden-goal solver."

## 4. Hypothesis / uncertainty / disagreement machinery

Present but diagnostic-only — nothing feeds back into action selection.

- **Reliability head** (model.rs :512) is the only per-action uncertainty estimate that the live policy consumes, and it is consumed as a bonus term, not as a gate.
- **PTRM stochastic ensemble** — `forward_ptrm` :1577 / `forward_ptrm_with_depth` :1595 / `forward_ptrm_prepared` :1616, `PtrmConfig{k, sigma, seed}` :464. K trajectories with RMS-relative Gaussian noise injected into `z` at every inner step (`maybe_noise_z` :1082); trajectory 0 is forced deterministic. Selection across trajectories via `best_q_indices` :471 (per-sample argmax over Q).
- **Ensemble disagreement metric** — `/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:2650-2686`: mean pairwise L2 between the K PTRM `y` latents, accumulated into `ensemble_disagreement` / `ensemble_n`. `cfg.ensemble_members` defaults to 8 (eval.rs :117, :146). Purely reported.
- **pass@k** — `ptrm_metrics` eval.rs :2436.
- **Calibration** — `/home/stepan/Coding/Personal/Tofy/src/p2/calibration.rs`: `expected_calibration_error` :4, `binary_auroc` :39, `risk_coverage_buckets` :70, applied to reliability probs at eval.rs :3509-3518.
- **Recursion probes** — `RecursionStepProbe` model.rs :173 / `probe_step` :1332: residual norm, latent norm, amplification per outer step (a convergence diagnostic, not a belief).
- **Rollout error decomposition** — `eval_rollout_group` eval.rs :3000: open-loop (recursive latent, `forward_from_latent` at :3026 with `open_latent = open_pred`), closed-loop (re-encode real frame each step, :3037-3045), and copy-forward baseline. `eval_contrastive_probes` :3647 computes `action_effect_mse / noop_identity_mse`.
- **Explicitly absent**: no belief/hypothesis state anywhere. `/home/stepan/Coding/Personal/Tofy/src/p2/agent_session.rs:3` says the native-agent slice "deliberately contains no hypothesis scoring or planning. It records only what the environment actually returned." The only cross-step memory in the live policy is the `tried: BTreeMap<u64, BTreeSet<String>>` novelty set (arc3_live.rs :599).

### Gaps a planner design would have to fill
- No reward/value head exists; `q_head` is a trajectory-ranking score trained (outside FullV4) against latent-MSE oracles, and under FullV4 against exact gameplay-pixel correctness — check which before treating it as a value.
- Goal features are available to the event head (`goal_dim=19`) but the live policy passes zeros.
- Recursive imagination is supported at the API level, but the only measured multi-step numbers are open-loop MSE at horizons 4/8/16, and semantic decode (`exact_gameplay_logits`) is gated on `world_core_v4`.