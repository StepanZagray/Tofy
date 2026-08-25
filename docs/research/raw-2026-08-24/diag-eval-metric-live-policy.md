I have everything I need. Here are the conclusions.

## 1. Exact definition of "changed-pixel exact transition accuracy"

The headline number is `semantic.<split>.overall.variants["one_step_prediction"].masks["changed"].exact_transition_accuracy`.

**Definition: all-or-nothing per transition, over the changed-pixel set only.**
- `MaskAccum::add` (`src/p2/semantic_eval.rs:141-170`) walks only mask-selected pixels and does `exact &= is_correct` (`:152`). One wrong pixel in the mask ⇒ the whole transition scores 0.
- `exact_transition_accuracy = exact / transitions` (`src/p2/semantic_eval.rs:178-179`); the per-pixel version is reported separately as `pixel_accuracy` and `mean_transition_accuracy` (`:177`, `:180-181`).
- The `changed` mask is `current != target` over the 63×64 gameplay grid (`src/p2/semantic_eval.rs:254-258`); row 63 (synthetic status/budget bar) is stripped by `gameplay()` (`:327-330`).
- **No-change transitions are dropped entirely**, not counted as correct: `if pixels == 0 { return; }` before `transitions += 1` (`src/p2/semantic_eval.rs:158-163`). So the denominator is "transitions with ≥1 changed gameplay pixel".
- Aggregation across rows is transition-weighted, not pixel-weighted (`src/p2/semantic_eval.rs:76-83` passes `weights_pixels = false`).

**Decoder: a learned linear per-token exact decoder, argmax over 16 palette logits.**
- `ExactPatchGrounding` = one `Linear(hidden_dim → 8*8*16)` per spatial token (`src/p2/grounding.rs:44-57`), rearranged to `B×63×64×16` with the status row narrowed off (`:61-86`).
- Eval decodes via `model.exact_gameplay_logits(latent)` then `argmax(D::Minus1)` (`src/p2/semantic_eval.rs:280-295`, `:302`, `:490-515`). Not nearest-neighbour, not a frozen oracle decoder.

**Beware: there is a second, different "changed-pixel" definition in the codebase.** `grounding.rs:109-166` (`transition_correctness`) is a *threshold* rule: ≥99% of all gameplay pixels correct AND ≥90% of changed pixels correct, with no-change transitions auto-passing the changed clause (`:159-164`). That one is the Q/reliability **label**, not the reported accuracy — see `q_label_definition: "exact_gameplay_pixels:overall>=0.99,changed>=0.90,status_row_excluded"` in the run reports.

## 2. Is there any gradient path rewarding pixel-exactness of *predicted* states? No.

- The exact decoder is trained **only on encoded states**: `exact_grounding_loss(&encoded.current, frames)` and `exact_grounding_loss(&encoded.next, next_frames)` (`src/p2/train.rs:3081-3088`; duplicate path `:3390-3392`). `out.y` (the prediction) is never passed to the decoder in any loss.
- The only prediction loss is latent Huber to the encoded next state plus canonical Huber (`src/p2/train.rs:3066-3073`).
- Where the decoder does touch `out.y`, it is explicitly detached and used only to manufacture observer labels: `exact_transition_correctness(&out.y.detach(), …)` (`src/p2/train.rs:3172-3176`, `:3471-3473`), and inside `transition_correctness` the logits are detached again (`src/p2/grounding.rs:115-118`).
- `PatchGroundingMode::Predicted` exists (`src/p2/grounding.rs:16-25`) but applies to the *histogram* head, and the Full V4 recipe sets `patch_grounding_weight = 0.0`, `exact_grounding_weight = 0.1` (`src/p2/train.rs:678-679`).
- Additionally the grounding term is scaled by `weights.world` (`src/p2/train.rs:3111`, `:3569`), which is **0.0** in the `q_calibration` and `falsification` lessons for FullV4 (`src/p2/train.rs:2055-2066`, `:2085-2096`).

**Conclusion:** the pipeline optimizes latent-space Huber for the predictor and pixel CE for the decoder-on-encodings. Nothing in the loss ever asks "does the *predicted* latent decode to the right pixels". The metric measures a composition that was never jointly trained.

The archived run confirms the size of that seam (`runs/p2/_pod_handoffs/qta3x0itzt4g33-20260820-full-v4/.../evaluations/step-000000028672.json`, `synthetic_dynamics.semantic.overall.variants`):

| variant | changed exact | changed pixel acc | foreground pixel acc |
|---|---|---|---|
| `target_reconstruction` (decode true next encoding) | 0.600 | 0.784 | 0.264 |
| `one_step_prediction` | **0.034** | 0.459 | 0.085 |
| `action_shuffled_prediction` | 0.016 | 0.443 | 0.118 |
| `zero_control` (predict EMPTY everywhere) | 0.000 | **0.439** | 0.000 |

Two things fall out: (a) the decoder itself caps the metric at ~0.60 even given the true encoded next state; (b) the predicted-latent decode is essentially the all-background control — 0.459 vs 0.439 changed-pixel accuracy, and foreground accuracy 0.085 vs 0.264 for the same decoder on an encoded state. Most transitions have exactly 2 changed pixels (agent leaves cell A, enters cell B — see per-source `pixels/transitions` = 2.0), and 0.5 changed-pixel accuracy is exactly what "paint background at both changed cells" yields. `action5_interact` (1 changed pixel) scores 0.000 for the model and 0.023 even for `target_reconstruction`.

## 3. `arc3_live.rs` greedy policy: it never decodes pixels

`ModelPolicy::score_candidates` (`src/p2/arc3_live.rs:620-693`) encodes the current frame once, broadcasts it across candidates, and runs `forward_from_encoded_state` per candidate. The score (`:678-681`):

```
0.25 * sigmoid(q) + 0.30 * sigmoid(reliability) + 0.30 * (1 - sigmoid(noop)) + 0.15 * effect/(1+effect)
```

where `effect = latent_mse_per_sample(output.y, state)` — the magnitude of the predicted latent displacement (`:672-674`). Then `choose_action` subtracts 1.0 for already-tried actions at this observation hash and takes the argmax (`:697-733`).

Consequences:
- **No goal distance, no value head, no novelty over states, no decoding.** Pixel exactness is irrelevant to action selection; only the *relative ranking* of scalar head outputs across candidate actions matters.
- 0.55 of the weight is Q + reliability, and **both heads are trained on the same `exact_transition_correctness` labels** (`src/p2/train.rs:3167-3178` for Q, `:3210-3224` and `:3507-3524` for reliability). So the policy preferentially picks actions **the model believes it can predict accurately**, which is a confidence signal, not a progress signal. Under a collapsed predictor this actively selects boring/no-op-like actions.
- 0.30 is `1 - P(noop)`, and the event head is the only head that consumes goal features (`src/p2/model.rs:1028-1039`). Live passes `goals = zeros` (`src/p2/arc3_live.rs:649`), while training uses a one-hot goal family + params (`src/p2/data.rs:192-206`, used at `data.rs:746`). The noop term is therefore evaluated at an out-of-distribution goal vector.
- Goal conditioning does not affect `y` at all (`run_recursion` → `run_latent_recursion` then `attach_heads`, `src/p2/model.rs:1213-1225`), so the transition itself is goal-free.
- Exploration is purely the tabular `tried` set keyed by an exact frame hash (`:736-744`), which never generalizes across near-identical frames.

## 4. Eval-time vs training-time normalization

Substantively identical; I found no eval-only RMS step.
- `encode_state` / `encode_state_pair` / `encode_state_pair_for_training` all apply `rms_norm_latent` (`src/p2/model.rs:659-720`), and the recursion re-normalizes and clamps `y` each outer step (`src/p2/model.rs:1189-1196`).
- Training conditions on an explicitly-passed `current_canonical` (`full_v4_training_latents_from_encoded_state`, `src/p2/model.rs:1446-1470`); eval's `forward_from_latent` / `forward_from_encoded_state` pass `None` and `add_action_with_canonical` recomputes it from the same state (`src/p2/model.rs:841-848`, `:900-910`) — numerically the same node.
- `RecursionOpts` differs (`EVAL` vs `training(true)`, `src/p2/model.rs:74-86`) but only toggles probe recording and step retention, not `y`.
- Depth: eval uses `RecursionDepth::from_config`; this run has `randomize_depth: false`, so they match. With `randomize_depth: true` (`src/p2/train.rs:1936-1958`) training would see 1..=N and eval always N — a real mismatch for other configs.
- One genuine asymmetry: eval's open-loop rollout uses `forward_from_latent` (goal-projected head path, `src/p2/eval.rs:3025-3033`), while the training rollout loss uses the goal-free `predict_latent_with_depth` (`src/p2/train.rs:3821-3826`, `src/p2/model.rs:1559-1571`). Harmless for `y`, but it means the two paths are not the same function call and can drift.

## 5. Open-loop horizons and one-step vs multi-step reporting

`eval_rollout_group` (`src/p2/eval.rs:3000-3086`):
- Encodes `z0` once, then feeds each prediction back (`open_latent = open_pred`, `:3034`). Trajectories are validated contiguous by `group_rollouts` (`:3254-3275`).
- Three curves per step: open-loop MSE, closed-loop MSE (re-encode the true frame each step), copy-forward MSE (`z0` vs target — a *no-dynamics* baseline, `:3066-3073`).
- Rows are emitted only at `horizon ∈ {4, 8, 16}` (`:3074-3077`); **semantic decoder metrics only at `{4, 8}`** and only under `world_core_v4` (`:3078-3082`), aggregated by `aggregate_rollout_semantics` over `[4, 8]` (`:3103-3119`).
- `attach_rollout_metrics` (`:3121-3155`) fills `rollout` / `closed_loop` / `copy_forward` plus `open_closed_ratio_8`; horizon stats via `summarize_horizon` (`:1727-1767`) with copy-normalized mean/median/p95/CVaR95 and `fraction_beating_copy`.
- **There is no horizon-1 semantic row**, and the one-step semantic table comes from a *different population* (`eval_sample_set` samples: `random_one_step` + `exploration` + `hazard_one_step`) than the rollout groups (`collect_dynamics_rollout_samples`, `:3313-3336`, only `random_one_step` + `exploration`, then filtered to trajectories ≥4 steps). So one-step vs multi-step numbers in the report are **not on a common population** and shouldn't be read as a degradation curve. In the archived run `n16 = 0` — h16 is never populated.

## 6. Bugs and inconsistencies

**a. Status-row leak in the changed-transition stratification (real mask inconsistency).** `src/p2/eval.rs:2591-2598` selects "changed" rows with `if sample.current != sample.next`, comparing **full 64×64 frames including row 63**. The budget bar advances with actions (`src/p2/data.rs:791-802`), so nearly every transition qualifies (n = 1396 of 1408 in the archived report) even when the board is untouched. Everything else in the codebase excludes row 63: `gameplay_logits` (`grounding.rs:84`), `patch_targets` (`grounding.rs:274`), `semantic_masks`/`gameplay` (`semantic_eval.rs:240`, `:327-330`), and `BoardEffect` (`data.rs:498-518`). The correct predicate here is `sample.noop == Some(false)` (as `summarize_action_strata` already uses, `eval.rs:1524-1527`) or a status-excluded pixel comparison.

**b. `changed_transitions.improvement_fraction = -10.5` is being reported without a guard.** `summarize_changed_transitions` (`eval.rs:2090-2115`) computes `1 - learned/copy`. In the archived run learned = 0.0760, copy = 0.0066: the predictor is **11× worse in latent space than doing nothing**, on unit-RMS latents where `‖encode(s_t) − encode(s_{t+1})‖² = 0.0066` (i.e. the encoder is nearly invariant to the transition). This is the more diagnostic number than the pixel metric, and the `ten_percent_improvement_pass` gate silently reports `false` rather than failing loudly.

**c. Metric peaks then collapses, and the collapse coincides with the `sequential` lesson.** Lesson lengths are not uniform — `dynamics` and `exploration` get 2× (`src/p2/train.rs:97-102`), so with `steps_per_lesson: 4096` the boundaries are dynamics 0–8192, exploration 8192–16384, sequential 16384–20480, q_calibration 20480–24576, falsification 24576–28672. The changed-exact trajectory: 0.166 (8192) → 0.136 (16384) → **0.034 (20480)** → 0.034 → 0.034. `one_step_latent_mse` is bit-identical from 20480 on, matching `observer-freeze-verification.json` (world/exact_decoder/auxiliary_decoders frozen at 20480/24576/28672). So: the 4× degradation happens entirely inside `sequential`, the only lesson that enables `rollout_weight * ramp` and `ptrm_rank` (`src/p2/train.rs:2044-2054`, `rollout_weight: 0.1`); and the "plateau" after 20480 is not a plateau at all — the world core is deliberately frozen and only observer heads train (`world: 0.0` at `:2055-2066` and `:2085-2096`). Any world-model improvement must come from steps < 20480.

**d. The `content` mask is not comparable across sources.** It is built from `provenance.content_width/height` (`semantic_eval.rs:242-248`), which is the true board rect for simulator samples but the whole 64×63 frame for `TransitionProvenance::full_frame` (`src/p2/data.rs:284-292`). Since `pixel_accuracy` aggregates pixel-weighted (`semantic_eval.rs:74-75`), the overall `content` numbers are dominated by all-background padding from full-frame sources (0.97 in the one-step table vs 0.54–0.58 in the rollout table on the same model). The `changed` mask is unaffected, so the headline metric is safe, but `content` should not be quoted.

**e. `shuffled_samples` rotates actions using `source_lengths`, not `provenance.source_kind`.** `src/p2/semantic_eval.rs:420-437` partitions by the caller-supplied `(name, len)` spans, while `by_source` reporting partitions by `sample.provenance.source_kind` (`:332-337`). For `arc3_transfer` the fallback is a single span covering everything (`eval.rs:3560`, `:3593`), so the "shuffled action" control can donate an action from a completely different game family — it is a weaker (more easily beaten) control than it reads.

**f. `action_masked_prediction` uses action id 0** (`semantic_eval.rs:470`), which is outside the trained 1..=7 range (`arc3_live.rs:123`) — an untrained embedding row rather than a learned null action. Its score (0.019 changed-exact vs 0.034 for the true action) therefore mixes "no action information" with "OOD embedding", which understates how much the true action actually contributes.

**g. Live/train frame mismatch on row 63.** Training always paints a synthetic budget bar on row 63 and then excludes it from every loss and metric (`data.rs:695-699`, `:791-802`; `grounding.rs:84`). Live frames are the raw API grid padded to 64×64 (`arc3_live.rs:156-174`), so row 63 is real content or padding — but the encoder consumes all 64 rows (`model.rs:659-662`) and has only ever seen a monotone status bar there.

No off-by-one errors found in the decoder rearrangement (`grounding.rs:72-85` — the `(0,1,3,2,4,5)` permute maps `patch_y*8+dy → y`, `patch_x*8+dx → x` correctly), in the rollout horizon indexing (`eval.rs:3074`, `horizon = idx + 1` with `open_pred` compared against `steps[idx].next`), or in the NLL flat indexing (`semantic_eval.rs:155`).