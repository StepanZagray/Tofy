Restating the coordinator's update: the eval subagent's findings are confirmed and relayed — all-or-nothing changed-mask metric, no gradient path grounding predicted latents, a collapse (0.166→0.034) during the sequential lesson followed by a hard freeze, the live-policy score composition, and four concrete eval bugs. Combining that with my own reading of the loss/model code, the run artifacts from the 61h A40 campaign, and the generator/optimizer audits, here is the final diagnosis.

All paths relative to `/home/stepan/Coding/Personal/Tofy`.

---

# Why full-v4 sits at 2–5% changed-pixel exact accuracy

**Headline correction: it is not a plateau — it is a collapse followed by a freeze.** Archived run `runs/p2/_pod_handoffs/qta3x0itzt4g33-20260820-full-v4/.../evaluations/`: changed-exact = 0.166 @8192 (end of dynamics) → 0.136 @16384 → **0.034 @20480** (during `sequential`, the only lesson with rollout loss) → bit-identical thereafter, because `q_calibration`/`falsification` set world weight to 0 (`src/p2/train.rs:2055-2103`), freezing the damaged world core for the final 8,192 of 28,672 steps (29% of the budget).

**Metric being failed:** all-or-nothing exact match over the changed mask (current≠target on the 63×64 gameplay grid), no-change transitions excluded (`src/p2/semantic_eval.rs:141-181,254-258`); the Q-label variant additionally requires ≥99% overall and ≥90% changed-pixel accuracy (`src/p2/grounding.rs:109-166`).

## (a) Ranked root causes

**1. No gradient path optimizes what the metric measures: predicted latents are never grounded in pixels.**
Evidence: the exact CE decoder is trained only on *encoded* current/next states (`src/p2/train.rs:3081-3088`); the predicted latent `out.y` reaches the decoder only detached, to make observer labels (`train.rs:3172-3176`, `grounding.rs:115-119`); ADR `docs/adr/0001-full-v4-training-contract.md:57` *explicitly excludes* "predicted-latent grounding." The prediction loss is latent Huber (`train.rs:3071-3074`), which for unit-RMS latents and delta=1 is exactly 0.5·MSE averaged over 8,192 dims (candle-nn 0.11 `loss.rs` `mean_all`; the robust branch never triggers). Distance minimization does not enforce decoder decision-region membership, and predicted latents are off the decoder's training manifold: at step 28672, prediction foreground pixel accuracy is **8.5%** vs 31.9% for decoding the *encoded* current state (`learned_copy_control`) — and prediction changed-pixel accuracy (45.9%) barely beats the predict-background `zero_control` (43.9%). The model decodes to "background" at changed locations.
Minimal intervention: add the exact per-pixel CE of `gameplay_logits(out.y)` against `next_frames` to the world loss (the machinery already exists — one extra call to `exact_grounding_loss(&out.y, &self.batch.next_frames)` at `train.rs:3077-3089`), weighted toward changed/foreground pixels. This is the single change most directly aimed at the metric.

**2. Epps–Pulley dominates loss and gradient; the encoder collapses the transition signal to satisfy it.**
Evidence from the run's own telemetry (`train_report.json`): dynamics lesson mean `sigreg_raw` = 80.0 (weighted 8.0) vs `next_latent` = 0.116 → EP is ~70× the prediction term; gradient-pressure at update 1: encoder EP-gradient L2 = 778 vs prediction-gradient 11.1 (**ratio 70:1**, cosine −0.027 — orthogonal, pure noise pressure); 100% of world-lesson updates hit the global clip `MAX_GRAD_NORM=1.0` (`train.rs:74`) at scale 0.18–0.41, so the prediction gradient is throttled 2.5–5× while EP consumes the clip budget. Consequence measured directly: `changed_transitions.copy_forward_mse` (latent distance between encoded current and next on *changed* transitions) shrank 0.038 → 0.0066 across training — the encoder became nearly transition-invariant, erasing exactly the information the metric needs — while `improvement_fraction` fell to **−10.5** (the learned prediction is 11× *worse* than copying the current encoding).
Minimal intervention: cut `sigreg_weight` ~10× (or normalize the EP gradient to a fixed fraction of the prediction gradient), and/or clip EP separately instead of sharing one global clip. Gate future runs on `improvement_fraction > 0`, which was negative at every checkpoint and never tripped anything.

**3. Extreme class imbalance with zero changed-content weighting anywhere in the V4 objective.**
Quantified from the eval populations: ≤2 changed pixels per transition out of 4,032 gameplay pixels (**0.05%**; 1,918 changed px across 1,408 transitions); foreground is ~0.6% of pixels; and the whole 7×7 training board occupies **one of 64 latent tokens** (8×8-stride patch encoder, `src/p2/model.rs:255-257,362-374`), the other 63 encoding constant padding whose value equals EMPTY. In the latent Huber, the copy attractor is within ~0.003 of the optimum (0.5×`copy_forward_mse`) against a total objective of 0.7–8.1 — a 10³–10⁴:1 dilution. The only balanced changed/unchanged weighting in the codebase lives in the *disabled* legacy histogram head (`grounding.rs:301-321`; `patch_grounding_weight=0.0`, `train.rs:603`).
Minimal intervention: changed-pixel upweighting in the new predicted-latent CE (cause 1); a change-masked term on the latent loss; and denser curricula — more changing pixels per transition, boards that span multiple tokens (or a finer latent grid, e.g. patch 4 → 16×16 tokens).

**4. The encoder→decoder pipeline alone caps the metric at 60%.**
`target_reconstruction` (encode the *true next frame*, decode it — a perfect-transition ceiling) scores only **0.600** changed-exact and 26.4% foreground pixel accuracy at step 28672. Even a flawless transition model under this contract could not exceed 60%. Cause: the exact CE at weight 0.1 is unweighted over pixels that are 99.4% background, and the ADR's 99%-overall reconstruction gate (ADR 0001:51-53) passes trivially on near-empty boards while foreground is lost.
Minimal intervention: foreground/changed-weighted CE and a *foreground* reconstruction gate; optionally raise `exact_grounding_weight`.

**5. The model is action-blind, and the data cannot force it not to be.**
The curriculum provides exactly one action per state — branch groups exist in code but are excluded from V4 (ADR 0001:57; `factual_branches` is not in `DEFAULT_LESSONS`, `train.rs:45-51`). With no counterfactual pairs, action-invariant prediction is loss-optimal whenever the next state is predictable from the current state. Measured: shuffled-action latent-MSE ratio **0.975** (CI spans 1.0), `action_conditioning_pass: false`; changed-pixel accuracy 45.9% (true action) vs 44.3% (shuffled) vs 41.2% (masked). For the no-harness goal this is fatal independently of accuracy: the greedy policy ranks *actions*, and an action-invariant model ranks them identically.
Minimal intervention: reinstate factual branch groups (several actions from the same state) in the world lessons — the FactualBatch machinery from ADR 0002 already exists — or add an action-contrastive/branch-separation term (the deferred ActSWM-style loss).

**6. The sequential lesson destroyed one-step accuracy and the observer freeze locked it in.**
Timing is exact: collapse happens 16384→20480, the only interval where the rollout loss is on (`train.rs:2044-2054`) — computed on a **single trajectory** (batch 1: `open_loop_latent_loss` narrows to one trace, `train.rs:3814-3816`) with horizon ramping 2→4→8 (`train.rs:2155-2174`) and teacher-forcing annealed to 0 (`train.rs:2177-2182`). High-variance long-horizon gradients from one trace, added to an already clip-saturated update, degraded the one-step map; then 8,192 observer-only steps froze it. Checkpoint selection did not protect the 16.6% model.
Minimal intervention: batch the rollout traces; gate each horizon increase on non-regression of one-step changed-exact; keep a small world/grounding loss alive in observer lessons or at least select `model.best` by changed-exact before the freeze.

**7. The observer heads that the live policy actually uses are trained on ~97%-negative labels and are below chance.**
Q/reliability labels = `transition_correctness` of the model's own predictions (≥99% overall ∧ ≥90% changed, `grounding.rs:109-128`; confirmed by the report's `q_label_definition`). With 1.6–3.4% correctness the positive rate is ~3%, and measured Q quality is: accuracy 63.6%, **balanced accuracy 47.2% (below chance)**, Brier 0.318, all calibration gates false. The live greedy score is 0.25·σ(q) + 0.30·σ(rel) + 0.30·(1−σ(noop)) + 0.15·effect (`src/p2/arc3_live.rs:678-681`) — 85% of it is these heads, with goals fed as zeros (OOD vs training's goal features, `arc3_live.rs:649`) and the effect term ≈ constant for an action-blind model. The policy is structurally noise.
Minimal intervention: fix causes 1–5 first (labels are downstream); switch labels to graded pixel accuracy instead of a threshold that fires 3% of the time; align live goal features with training.

**8. Optimizer schedule (secondary).** Constant lr 1e-3 with no warmup, decay, or EMA anywhere; Muon's normalized-buffer updates inject fixed-magnitude (~1.8e-4 RMS) whitened steps even at small gradients — a refinement noise floor that pixel-exact decoding likely cannot cross at lr 1e-3; Muon also orthogonalizes degenerate 128×2/128×4 projections (`coord_proj`, `spatial_action_proj`).
Minimal intervention: cosine decay or a late-lesson lr drop; route tiny matrices to Adam; consider eval-time EMA.

## (b) Priority of interventions

1. Predicted-latent exact CE with changed-pixel weighting (causes 1+3) — the metric finally gets a gradient.
2. EP down-weighting / gradient normalization + `improvement_fraction>0` gate (cause 2).
3. Foreground-weighted decoder CE + foreground reconstruction gate (cause 4).
4. Branch groups in world lessons (cause 5).
5. Rollout batching + horizon gating + best-checkpoint selection by changed-exact (cause 6).
6. Then relabel and retrain observers; fix live goal features (cause 7).

The in-flight foundation-v1 rerun reproduces the same contract, so it should be expected to reproduce the same collapse; the eval-at-8192 checkpoint will be its best world model.

## (c) Outright bugs found

1. **`src/p2/eval.rs:2591-2598`** — changed-transition stratification compares full frames *including status row 63*, which advances every action, so 1,396/1,408 transitions count as "changed"; the `changed_transitions.*` latent metrics are contaminated (semantic metrics are not).
2. **`src/p2/semantic_eval.rs:420-437`** — `shuffled_samples` builds spans from `source_lengths`, not `provenance.source_kind`; arc3_transfer collapses into one span, weakening the shuffled-action control there.
3. **`src/p2/semantic_eval.rs:470`** — the action-masked control uses action id 0, an *untrained* embedding row (training range is 1..=7, `train.rs:1587`), so the "null action" control is actually an OOD-embedding probe.
4. **Status-row train/live mismatch** — training data always paints a synthetic budget bar in row 63 while live ARC frames carry real row-63 content; `embed_frames` masks the row before encoding (`model.rs:830-838`) so the encoder path is safe, but raw-frame consumers (the tried-action exact-frame-hash penalty in `arc3_live.rs`, event labels) see different distributions.
5. **`arc3_live.rs:649`** — live policy passes all-zero goal features where training used populated goal vectors: every live head readout is off-distribution.
6. **`src/p2/train.rs:3814-3816`** — rollout loss batch is 1 trajectory per optimizer step at batch 2048; combined with 100%-clipped updates this is the proximate destabilizer of the sequential lesson.
7. Cosmetic: the ±32 latent clamp after per-sample RMS norm (`model.rs:1195`) can never bind (RMS=1); and `DEFAULT_MAX_ROLLOUT_HORIZON=8` means the documented "2→4→8→max(16)" final stage never occurs.

Sanity check performed: every quantitative claim above was re-read from either the cited source lines or the archived run bundle (`train_report.json`, `evaluations/step-*.json`) rather than taken from subagent summaries where the two could be cross-checked; the three subagent reports (generator/data, eval/live, optimizer) agreed with my direct reads at every overlap point.

result: full-v4's 2-5% changed-exact is a sequential-lesson collapse (16.6%→3.4%) then observer freeze, rooted in a metric with no gradient path (predicted latents never pixel-grounded), 70:1 Epps-Pulley gradient dominance collapsing the transition signal, 0.05% changed-pixel imbalance, a 60% decoder ceiling, and action-blind data — with predicted-latent weighted CE grounding as the highest-leverage fix.