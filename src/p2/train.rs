//! P2 LeWorld / TRM training on synthetic curriculum only.

use crate::domain::Split;
use crate::gpu_lock::{GpuSessionGuard, TrainPidGuard};
use crate::p2::branch_learning::{branch_learning_loss, BranchLearningAudit, BranchLearningConfig};
use crate::p2::cg_profile::{
    reconcile_profile_bundle, CaptureSpec, GradientClipState, ProfileRange, ProfileState,
    RepresentativeUpdateCapture, PROFILE_ENTRYPOINT,
};
use crate::p2::consumer_transition::ConsumerTransition;
use crate::p2::data::{
    adaptation_v6_stream_schedule, compose_mixed_stream_batch, foundation_v2_stream_schedule,
    gameplay_rows, generate_curriculum, ArcFrame, ContentMask, ContentRect, EventLabelCensus,
    FactualBatch, MixedStreamBatch, MixedStreamConfig, MixedStreamKind, OperatorFamily,
    TransitionSample, V5DataSplit, V5Sample, V5SampleProvenance, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::eval::{
    evaluate_gate_support_with_v5_provenance, GateSupportMetrics,
    MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS,
};
use crate::p2::experiment::{
    ConsumerReadoutTopology, ExperimentRequest, ResolvedExperiment, SigregPopulation,
    SigregStatistic, TrainingRecipe,
};
use crate::p2::grounding::{DecodeComposition, PatchGroundingMode};
use crate::p2::model::{
    flatten_latent, init_copy_bypass_gate, latent_mse_per_sample, restore_copy_gate_bias_prior,
    zero_action_film_projections, zero_context_film_projections,
    zero_operator_conditioning_projection, ContextBatch, ContextBatchHost, ModelConfig, PtrmConfig,
    RecursionDepth, RecursionOpts, TrainingEncodedPair, WorldModel, ACTION_VOCAB,
    CONTEXT_PARAMETER_PREFIX, DEFAULT_NUM_EVENTS, LEGACY_PATCH_SIZE, OPERATOR_CONDITION_DIM,
    OPERATOR_FAMILY_UNKNOWN, OPERATOR_FAMILY_VOCAB, PALETTE_SIZE, PATCH_SIZE, PREFIX_HORIZONS,
};
use crate::p2::muon::MUON_RMS_SCALE;
use crate::p2::optimizer::{
    accumulate_parameter_gradients, clip_gradients_gpu_with_stats,
    try_clip_gradients_gpu_with_stats, CheckpointHybridOptimizer, ModelEma,
};
use crate::p2::prefetch::{
    BatchPrefetcher, MixedStreamBatchPrefetcher, PrefetchRequest, PrefetchScope,
};
use crate::p2::sigreg::{sigreg_epps_pulley_seeded, sigreg_quantile_seeded};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var, D};
use candle_graph::{
    CampaignManifest, ExecutionStep, PlannedCapture, SpanId, SpanKind, CAMPAIGN_SCHEMA,
};
use candle_nn::init::FanInOut;
use candle_nn::optim::ParamsAdamW;
use candle_nn::{VarBuilder, VarMap};
use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::OnceLock;
use std::thread;

/// ARC-AGI-3 aligned curriculum: goal-free early play, then planning/falsification.
pub const DEFAULT_LESSONS: &[&str] = &[
    "dynamics",
    "exploration",
    "sequential",
    "q_calibration",
    "falsification",
];

/// Weight on PTRM best-Q ranking loss (sequential/falsification/retarget).
const PTRM_RANK_WEIGHT: f64 = 0.05;
/// Default PTRM trajectories for ranking loss outside falsification.
const PTRM_RANK_K_DEFAULT: usize = 2;
/// Falsification lesson uses more trajectories to match eval q_oracle_rank@4.
const PTRM_RANK_K_FALSIFICATION: usize = 4;
/// Retarget open-loop loss is capped — full weight destabilized v9 rollouts.
const RETARGET_ROLLOUT_SCALE: f64 = 0.25;
/// Max open-loop imagination horizon during sequential training (v11-stable).
const DEFAULT_MAX_ROLLOUT_HORIZON: usize = 8;
/// Retarget uses a shorter cap than sequential (v10 stability fix).
const RETARGET_MAX_ROLLOUT_HORIZON: usize = 4;
/// Huber cap per open-loop step so runaway latents do not dominate the optimizer.
const ROLLOUT_STEP_LOSS_CAP: f64 = 10.0;
/// Reset open-loop state to the encoded real frame when step error exceeds this.
const ROLLOUT_ERROR_RESET: f32 = 5.0;
/// Penalize high Q when latent error is large (anti-hallucination).
const Q_SURPRISE_WEIGHT: f64 = 0.1;
/// Smooth forward bound for SIGReg while retaining gradients above the limit.
const SIGREG_LOSS_CAP: f64 = 10_000.0;
/// Global gradient L2 clip. A safety rail is the intended policy, but keep 1.0
/// until post-conditioning run telemetry establishes the typical norm;
/// pre-fix, almost-always-clipped norms do not justify raising the threshold.
const MAX_GRAD_NORM: f64 = 1.0;
/// Caps rare-change amplification while leaving the observed 812/13_341
/// changed-pixel regime (`(1-p)/p = 15.43`) unchanged.
const FOUNDATION_V2_CHANGED_STRATUM_AMPLIFICATION_MAX: f64 = 50.0;
/// Smooth displacement-normalization radius. For
/// `d / sqrt(||d||^2 + eps^2)`, the Jacobian operator norm is at most 1/eps.
const FOUNDATION_V2_DISPLACEMENT_NORM_EPS: f64 = 1e-3;
const FOUNDATION_V2_EP_MIN_WEIGHT: f64 = 1e-4;
const FOUNDATION_V2_EP_MAX_WEIGHT: f64 = 0.1;
const FOUNDATION_V2_EP_GRADIENT_BUDGET: f64 = 0.3;
const FOUNDATION_V2_GATE_EVERY: u64 = 1_024;
/// Step at which the shuffled-action and composed-collapse gates arm. Values
/// measured before this step are warmup telemetry and must not seed any
/// collapse floor: an unarmed warmup peak is not evidence of attained quality.
const FOUNDATION_V2_GATE_WARMUP_STEPS: u64 = 4_096;
const FOUNDATION_V2_PERMANENT_EVERY: u64 = 2_048;
const FOUNDATION_V2_GATE_ROWS: usize = 512;
/// ADR 0005 §3.5 recursion depth (inner and outer) for world-core v6.
pub const V6_RECURSION_STEPS: usize = 2;
fn default_v6_recursion_steps() -> usize {
    V6_RECURSION_STEPS
}
const FOUNDATION_V2_ABORT_MARKER: &str = "foundation_v2_abort.json";
const LEGACY_CHECKPOINT_ARTIFACTS: &[&str] = &[
    "model.safetensors",
    "optimizer.safetensors",
    "trainer_state.json",
    "config.json",
];
const FOUNDATION_V2_CHECKPOINT_ARTIFACTS: &[&str] = &[
    "model.safetensors",
    "optimizer.safetensors",
    "trainer_state.json",
    "config.json",
    "ema.safetensors",
    "gate_history.json",
];
pub(crate) const FOUNDATION_V2_GATE_SEED: u64 = 0xF0A2_DA7A_0000_0005;
const FOUNDATION_V2_MIN_ROLLOUT_FRAGMENTS: usize = 16;
/// ADR 0003 §3.5 hinge margin on the L2 distance of normalized displacements.
const FOUNDATION_V2_SEPARATION_MARGIN: f64 = 0.3;
/// Per-event-slot multipliers: noop, satisfied, failed, exhausted. The
/// `exhausted` weight is deliberately dead for generated V5 data: its
/// action-budget premise is absent from the event-head input, so
/// `augment_v5_transition` force-masks the label (ADR 0003 corrections);
/// do not "fix" the weight instead of the premise.
const EVENT_SLOT_WEIGHTS: [f32; 4] = [1.0, 1.0, 4.0, 2.0];
pub const TRAIN_REPORT_SCHEMA: &str = "p2.train_report.v13";
pub const TRAINER_STATE_SCHEMA: &str = "p2.trainer_state.v10";
/// Revision of the foundation-v2 objective *implementation* (masks, loss
/// construction, reductions, gates). Bump on any semantic change so a
/// checkpoint trained under an older objective cannot silently resume under a
/// newer one; the resume contract carries this value without a serde default.
/// 1 = pre-content-mask objective; 2 = 2026-08-27 content-masked objective
/// with the reachable separation hinge and budget-exact EP controller;
/// 3 = bounded split-CE amplification, conditioned displacement norm, and the
/// nonzero copy-bypass alpha init that reopens the candidate gradient path,
/// plus unchanged-target copy-gate supervision on gameplay PAD pixels;
/// 4 = episode-operator family and permuted-color conditioning at the action
/// seam, and reliability observes thresholded factual latent prediction error
/// while Q continues to observe graded composed-pixel accuracy.
pub const FOUNDATION_OBJECTIVE_REVISION: u32 = 4;
const FNV1A64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

pub type SigregTarget = SigregPopulation;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChangedPixelWeights {
    pub content_pixels: usize,
    pub changed_pixels: usize,
    pub unchanged_pixels: usize,
    pub changed_fraction: f64,
    pub changed_weight: f64,
    pub unchanged_weight: f64,
}

/// Resolve ADR 0003's changed/unchanged weighting from gameplay masks. PAD is
/// excluded by `content_mask`; semantic palette zero remains a valid content
/// pixel and is never used as a padding proxy.
pub fn foundation_v2_loss_weights_from_masks(
    current: &[u8],
    target: &[u8],
    content_mask: &[u8],
) -> Result<ChangedPixelWeights> {
    if current.len() != target.len() || current.len() != content_mask.len() {
        bail!("foundation-v2 loss masks must have identical lengths");
    }
    let content_pixels = content_mask.iter().filter(|&&value| value != 0).count();
    if content_pixels == 0 {
        bail!("foundation-v2 loss requires at least one content pixel");
    }
    let changed_pixels = current
        .iter()
        .zip(target)
        .zip(content_mask)
        .filter(|((before, after), content)| **content != 0 && before != after)
        .count();
    let unchanged_pixels = content_pixels - changed_pixels;
    let changed_fraction = changed_pixels as f64 / content_pixels as f64;
    let ratio = if changed_pixels == 0 {
        f64::INFINITY
    } else {
        (1.0 - changed_fraction) / changed_fraction
    };
    Ok(ChangedPixelWeights {
        content_pixels,
        changed_pixels,
        unchanged_pixels,
        changed_fraction,
        changed_weight: ratio.clamp(1.0, FOUNDATION_V2_CHANGED_STRATUM_AMPLIFICATION_MAX),
        unchanged_weight: 1.0,
    })
}

/// Direct-target EP controller: the returned weight satisfies
/// `weight * ||g_ep|| = 0.3 * ||g_pred||` exactly in the interior (the
/// previous weight participates only on the invalid-input rail), then
/// applies the ADR bounds. When
/// the budget demands a weight below the positive floor (including a zero
/// prediction gradient), EP is disabled outright with weight zero rather than
/// held at a floor that would violate the `<= 0.3x` bound.
pub fn foundation_v2_ep_weight_update(
    current_weight: f64,
    ep_gradient_l2: f64,
    prediction_gradient_l2: f64,
) -> f64 {
    let current = current_weight.clamp(0.0, FOUNDATION_V2_EP_MAX_WEIGHT);
    if !(ep_gradient_l2.is_finite() && prediction_gradient_l2.is_finite())
        || ep_gradient_l2 < 0.0
        || prediction_gradient_l2 < 0.0
    {
        return current;
    }
    if ep_gradient_l2 == 0.0 {
        return FOUNDATION_V2_EP_MAX_WEIGHT;
    }
    let target = FOUNDATION_V2_EP_GRADIENT_BUDGET * prediction_gradient_l2 / ep_gradient_l2;
    if target < FOUNDATION_V2_EP_MIN_WEIGHT {
        return 0.0;
    }
    target.clamp(FOUNDATION_V2_EP_MIN_WEIGHT, FOUNDATION_V2_EP_MAX_WEIGHT)
}

/// ADR 0003 §3.5 separation term: hinge of `margin` on the L2 distance
/// between two displacement rows, in the same units as the L2-normalized
/// displacements (an RMS distance over 128 dims caps at `2/sqrt(128) ≈ 0.177`
/// and can never satisfy the 0.3 margin). The epsilon offset keeps the
/// gradient finite at exact displacement equality, where a bare
/// `sqrt(0)` backward is non-finite.
pub fn separation_hinge_term(left: &Tensor, right: &Tensor, margin: f64) -> Result<Tensor> {
    const EPS: f64 = 1e-12;
    let distance = left
        .sub(right)?
        .sqr()?
        .sum_all()?
        .affine(1.0, EPS)?
        .sqrt()?
        .affine(1.0, -EPS.sqrt())?;
    Ok(distance.affine(-1.0, margin)?.relu()?)
}

/// WSD schedule used by foundation-v2. `step` is the number of the update
/// about to execute (0 is the untrained boundary, `total_steps` is final).
pub fn foundation_v2_wsd_learning_rate(step: usize, total_steps: usize) -> f64 {
    const PEAK: f64 = 1e-3;
    const FINAL: f64 = 1e-4;
    const WARMUP: usize = 500;
    if total_steps == 0 {
        return FINAL;
    }
    let step = step.min(total_steps);
    // Preserve the 500-update production warmup, but keep smoke schedules
    // well-defined and guarantee the requested final LR at their last step.
    let warmup_steps = WARMUP.min(total_steps.saturating_sub(1));
    if warmup_steps > 0 && step < warmup_steps {
        return PEAK * step as f64 / warmup_steps as f64;
    }
    let decay_steps = (((total_steps as f64) * 0.15).ceil() as usize)
        .max(1)
        .min(total_steps.saturating_sub(warmup_steps).max(1));
    let decay_start = total_steps.saturating_sub(decay_steps);
    if step <= decay_start {
        return PEAK;
    }
    let progress = (step - decay_start) as f64 / decay_steps as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    FINAL + (PEAK - FINAL) * cosine
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2GateResult {
    pub name: String,
    pub passed: bool,
    pub measured: Option<f64>,
    pub threshold: String,
    #[serde(default)]
    pub abort_exempt: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abort_exemption_reason: Option<String>,
    /// Structured floor for abort significance (gate policy v6). Absent on
    /// warmup entries and on histories written before v6; absent means the
    /// legacy fail-closed abort accounting applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<f64>,
    /// One-sided 95% binomial noise margin of the floor on this gate's
    /// denominator. A failure must violate the floor by more than this to
    /// count toward an abort; recorded so replayed histories are auditable
    /// and recomputed accounting cannot be forged through `abort_exempt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_margin: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2GateDiagnostic {
    pub name: String,
    pub measured: Option<f64>,
    pub interpretation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2GateEvaluation {
    pub step: u64,
    pub metrics: GateSupportMetrics,
    pub running_best_before: Option<f64>,
    pub running_best_after: Option<f64>,
    pub gates: Vec<FoundationV2GateResult>,
    /// Non-enforcing measurements. The default accepts gate histories written
    /// before diagnostics were separated from the enforceable gate vector.
    #[serde(default)]
    pub diagnostics: Vec<FoundationV2GateDiagnostic>,
}

/// Checkpoint-promotion selection metric. `ChangedExact` is the historical
/// default; `FullExact` selects on full-transition exactness (unchanged
/// pixels included) without touching the gate or collapse-floor semantics.
/// `ComposedExactGuarded` selects lexicographically on composed changed-row
/// exactness and then composed all-row exactness. Changed-row exactness is the
/// primary key because an all-row score can be dominated by no-op copies; a
/// zero-changed-exact copy model therefore cannot freeze out a candidate that
/// predicts any changed row exactly. Both false-edit rates remain regression
/// guards with 10% relative plus 0.1 percentage-point absolute tolerance, so
/// quantization-scale movement away from a near-zero incumbent does not make
/// the first useful edit impossible while material hallucination still fails
/// closed. Note: `PromotionMetric` names in-run best-checkpoint *election* (a
/// selection_only mechanism per ADR 0003 §6), not the promotion-evidence
/// class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PromotionMetric {
    #[default]
    ChangedExact,
    FullExact,
    ComposedExactGuarded,
}

/// Value of the configured promotion metric on one gate measurement.
pub fn foundation_v2_promotion_value(
    metric: PromotionMetric,
    metrics: &GateSupportMetrics,
) -> Option<f64> {
    match metric {
        PromotionMetric::ChangedExact => metrics.one_step_changed_exact,
        PromotionMetric::FullExact => metrics.one_step_full_exact,
        PromotionMetric::ComposedExactGuarded => metrics.one_step_composed_changed_exact,
    }
}

/// Guard for `ComposedExactGuarded`: a candidate may not regress either
/// false-edit rate beyond a small relative-plus-absolute tolerance versus the
/// incumbent best. All four selection measurements must be present and
/// finite on the candidate; incomplete evaluations fail conservatively.
fn foundation_v2_false_edit_guard(
    incumbent: &GateSupportMetrics,
    candidate: &GateSupportMetrics,
) -> bool {
    const RELATIVE_TOLERANCE: f64 = 0.10;
    const ABSOLUTE_TOLERANCE: f64 = 0.001;

    fn not_regressed(incumbent: Option<f64>, candidate: Option<f64>) -> bool {
        match (incumbent, candidate) {
            (None, Some(current)) => current.is_finite(),
            (None, None) => false,
            (Some(_), None) => false,
            (Some(prior), Some(current)) => {
                prior.is_finite()
                    && current.is_finite()
                    && current <= prior * (1.0 + RELATIVE_TOLERANCE) + ABSOLUTE_TOLERANCE
            }
        }
    }
    not_regressed(incumbent.false_edit_rate, candidate.false_edit_rate)
        && not_regressed(
            incumbent.padding_false_edit_rate,
            candidate.padding_false_edit_rate,
        )
}

fn foundation_v2_composed_selection_complete(metrics: &GateSupportMetrics) -> bool {
    [
        metrics.one_step_composed_changed_exact,
        metrics.one_step_all_rows_exact,
        metrics.false_edit_rate,
        metrics.padding_false_edit_rate,
    ]
    .into_iter()
    .all(|metric| metric.is_some_and(f64::is_finite))
}

/// Whether `candidate` replaces `incumbent` under the configured promotion
/// metric. `ComposedExactGuarded` uses a strict lexicographic improvement on
/// composed changed-row exactness and all-row exactness, subject to the
/// false-edit guard.
pub fn foundation_v2_candidate_improves(
    metric: PromotionMetric,
    incumbent: Option<&GateSupportMetrics>,
    candidate: &GateSupportMetrics,
) -> bool {
    let Some(value) = foundation_v2_promotion_value(metric, candidate) else {
        return false;
    };
    match metric {
        PromotionMetric::ComposedExactGuarded => {
            if !foundation_v2_composed_selection_complete(candidate) {
                return false;
            }
            // A model with zero composed changed-exact never edits anything;
            // its trivially tiny false-edit rate would otherwise become an
            // unbeatable incumbent under the non-regression guard and latch
            // promotion forever (run s8 exported its untrained step-1024
            // checkpoint this way). The collapse gate already treats
            // composed == 0 as catastrophic once armed; selection must not
            // crown what the gate condemns.
            if candidate
                .one_step_composed_changed_exact
                .is_some_and(|value| value <= 0.0)
            {
                return false;
            }
            incumbent.is_none_or(|metrics| {
                let better = match (
                    metrics.one_step_composed_changed_exact,
                    metrics.one_step_all_rows_exact,
                ) {
                    (Some(best_changed), Some(best_all_rows)) => {
                        value > best_changed
                            || value == best_changed
                                && candidate
                                    .one_step_all_rows_exact
                                    .is_some_and(|current| current > best_all_rows)
                    }
                    _ => true,
                };
                better && foundation_v2_false_edit_guard(metrics, candidate)
            })
        }
        _ => incumbent
            .and_then(|metrics| foundation_v2_promotion_value(metric, metrics))
            .is_none_or(|best| value > best),
    }
}

/// Whether a named gate is enforced at this evaluation step. Historical
/// `positive_improvement` entries remain diagnostic when old gate-history
/// JSON is replayed.
pub fn foundation_v2_gate_is_armed(evaluation: &FoundationV2GateEvaluation, name: &str) -> bool {
    match name {
        "positive_improvement" => false,
        "shuffled_action_ratio" | "composed_changed_exact_collapse" => {
            evaluation.step >= FOUNDATION_V2_GATE_WARMUP_STEPS
        }
        "foreground_reconstruction" => evaluation.step >= 8_192,
        "one_step_collapse" => true,
        // Unknown gates fail closed instead of silently becoming diagnostics.
        _ => true,
    }
}

/// Look up a gate by stable name rather than its position in the serialized
/// vector. Missing gates fail closed for consumers that depend on them.
pub fn foundation_v2_named_gate_passed(
    evaluation: &FoundationV2GateEvaluation,
    name: &str,
) -> bool {
    evaluation
        .gates
        .iter()
        .find(|gate| gate.name == name)
        .is_some_and(|gate| gate.passed)
}

/// Promotion is allowed only when every gate armed for the same evaluation
/// passed. Warmup PASS-by-fiat results are explicitly not evidence here.
pub fn foundation_v2_armed_gates_passed(evaluation: &FoundationV2GateEvaluation) -> bool {
    evaluation
        .gates
        .iter()
        .filter(|gate| foundation_v2_gate_is_armed(evaluation, &gate.name))
        .all(|gate| gate.passed)
}

/// The gate evaluation currently holding the promotion under `metric`,
/// obtained by replaying the strict-improvement scan over the history.
pub fn foundation_v2_best_evaluation(
    metric: PromotionMetric,
    gate_history: &[FoundationV2GateEvaluation],
) -> Option<&FoundationV2GateEvaluation> {
    let mut best: Option<&FoundationV2GateEvaluation> = None;
    for evaluation in gate_history {
        if foundation_v2_armed_gates_passed(evaluation)
            && foundation_v2_candidate_improves(
                metric,
                best.map(|evaluation| &evaluation.metrics),
                &evaluation.metrics,
            )
        {
            best = Some(evaluation);
        }
    }
    best
}

/// Whether the new measurement beats the running best of the configured
/// promotion metric. Gate history is replayed through the same strict rule
/// used for selection; `best_changed_exact` is retained only as a compatibility
/// fallback for old checkpoints whose gate history is empty.
pub fn foundation_v2_promotion_improved(
    metric: PromotionMetric,
    best_changed_exact: Option<f64>,
    gate_history: &[FoundationV2GateEvaluation],
    metrics: &GateSupportMetrics,
) -> bool {
    if metric == PromotionMetric::ChangedExact && gate_history.is_empty() {
        return foundation_v2_promotion_value(metric, metrics)
            .is_some_and(|current| best_changed_exact.is_none_or(|best| current > best));
    }
    foundation_v2_candidate_improves(
        metric,
        foundation_v2_best_evaluation(metric, gate_history).map(|evaluation| &evaluation.metrics),
        metrics,
    )
}

/// Gate-aware promotion decision for one complete evaluation.
pub fn foundation_v2_evaluation_improves(
    metric: PromotionMetric,
    best_changed_exact: Option<f64>,
    gate_history: &[FoundationV2GateEvaluation],
    evaluation: &FoundationV2GateEvaluation,
) -> bool {
    foundation_v2_armed_gates_passed(evaluation)
        && foundation_v2_promotion_improved(
            metric,
            best_changed_exact,
            gate_history,
            &evaluation.metrics,
        )
}

/// The step whose gate evaluation last strictly improved the configured
/// promotion metric — i.e. the step whose bundle `publish_best_checkpoint`
/// promoted into `checkpoints/best`. `None` when no evaluation produced a
/// metric value.
pub fn foundation_v2_selected_best_step(
    metric: PromotionMetric,
    gate_history: &[FoundationV2GateEvaluation],
) -> Option<u64> {
    foundation_v2_best_evaluation(metric, gate_history).map(|evaluation| evaluation.step)
}

pub fn foundation_v2_gate_evaluation(
    step: u64,
    metrics: GateSupportMetrics,
    running_best: Option<f64>,
    composed_running_best: Option<f64>,
    gate_history: &[FoundationV2GateEvaluation],
) -> Result<FoundationV2GateEvaluation> {
    let outcome_changing_rows =
        metrics
            .shuffled_action_outcome_changing_tuples
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "foundation-v2 shuffled-action gate has no counterfactual outcome coverage; \
                 evaluate it with aligned V5 operator provenance"
                )
            })?;
    if outcome_changing_rows < MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS {
        bail!(
            "foundation-v2 shuffled-action gate has only {outcome_changing_rows} \
             outcome-changing tuples; at least {MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS} \
             are required to certify action sensitivity"
        );
    }
    let current_exact = metrics.one_step_changed_exact;
    let running_best_after = match (running_best, current_exact) {
        (Some(best), Some(current)) => Some(best.max(current)),
        (None, Some(current)) => Some(current),
        (best, None) => best,
    };
    let collapse_floor = running_best_after.map(|best| best * 0.8);
    let current_composed_exact = metrics.one_step_composed_changed_exact;
    let composed_running_best_after = match (composed_running_best, current_composed_exact) {
        (Some(best), Some(current)) => Some(best.max(current)),
        (None, Some(current)) => Some(current),
        (best, None) => best,
    };
    let composed_collapse_floor = composed_running_best_after.map(|best| best * 0.8);
    // Absolute-quality gates get the same warmup grace as foreground
    // reconstruction: before step 4096 the model cannot yet be expected to
    // beat latent copy or show action sensitivity, so those gates are
    // measured and logged but PASS by fiat. The collapse detector stays
    // active from the first evaluation because it is relative to the run's
    // own best.
    let warmup_done = step >= FOUNDATION_V2_GATE_WARMUP_STEPS;
    // Foreground reconstruction ramps slowly under the changed-pixel-weighted
    // CE (0.086 -> 0.639 over the first 4096 steps of the first launch);
    // enforce it once the decoder has had half the pre-decay run to mature.
    let foreground_active = step >= 8_192;
    let shuffled_passed = !warmup_done
        || metrics
            .shuffled_action_changed_pixel_ratio
            .is_some_and(|value| value <= 0.95);
    let shuffled_abort_exemption = (!shuffled_passed)
        .then(|| {
            foundation_v2_floor_gate_abort_exemption(
                "shuffled_action_ratio",
                metrics.shuffled_action_changed_pixel_ratio,
                gate_history,
            )
        })
        .flatten();
    let foreground_passed = !foreground_active
        || metrics
            .foreground_reconstruction_accuracy
            .is_some_and(|value| value >= 0.60);
    let foreground_abort_exemption = (!foreground_passed)
        .then(|| {
            foundation_v2_floor_gate_abort_exemption(
                "foreground_reconstruction",
                metrics.foreground_reconstruction_accuracy,
                gate_history,
            )
        })
        .flatten();
    // Gate policy v6: structured floors + one-sided 95% binomial noise
    // margins on each gate's real denominator. `passed` stays the plain
    // threshold comparison (promotion bar unchanged); only the abort
    // accounting uses the margins.
    let shuffled_floor = warmup_done.then_some(0.95);
    let shuffled_margin =
        shuffled_floor.map(|floor| foundation_v2_noise_margin(floor, outcome_changing_rows));
    let foreground_floor = foreground_active.then_some(0.60);
    let foreground_margin =
        foreground_floor.map(|floor| foundation_v2_noise_margin(floor, metrics.foreground_pixels));
    let one_step_margin =
        collapse_floor.map(|floor| foundation_v2_noise_margin(floor, metrics.changed_transitions));
    let composed_floor_field = if warmup_done {
        composed_collapse_floor
    } else {
        None
    };
    let composed_margin = composed_floor_field
        .map(|floor| foundation_v2_noise_margin(floor, metrics.changed_transitions));
    let diagnostics = vec![FoundationV2GateDiagnostic {
        // Latent-MSE improvement over copy measures proximity in a space the
        // v5 objective does not optimize; it cannot be represented as a
        // passing enforceable gate.
        name: "positive_improvement".into(),
        measured: metrics.improvement_fraction,
        interpretation: "latent-MSE diagnostic; superseded by pixel-space gates".into(),
    }];
    let gates = vec![
        FoundationV2GateResult {
            name: "shuffled_action_ratio".into(),
            passed: shuffled_passed,
            measured: metrics.shuffled_action_changed_pixel_ratio,
            threshold: if warmup_done {
                "<= 0.95 on outcome-changing counterfactual rows (action-blind ~= 1.0)".into()
            } else {
                "warmup PASS until step 4096".into()
            },
            abort_exempt: shuffled_abort_exemption.is_some(),
            abort_exemption_reason: shuffled_abort_exemption,
            floor: shuffled_floor,
            noise_margin: shuffled_margin,
        },
        FoundationV2GateResult {
            // Under changed-pixel-weighted CE the encoded-state foreground
            // reconstruction asymptotes near 0.67 (first launch measured
            // 0.639 -> 0.675 over steps 4096..9216 while changed-exact kept
            // climbing). The gate exists to catch decoder collapse, so it is
            // a regression floor below the observed asymptote, not an
            // aspirational target.
            name: "foreground_reconstruction".into(),
            passed: foreground_passed,
            measured: metrics.foreground_reconstruction_accuracy,
            threshold: if foreground_active {
                ">= 0.60 (collapse floor)".into()
            } else {
                "warmup PASS until step 8192".into()
            },
            abort_exempt: foreground_abort_exemption.is_some(),
            abort_exemption_reason: foreground_abort_exemption,
            floor: foreground_floor,
            noise_margin: foreground_margin,
        },
        FoundationV2GateResult {
            name: "one_step_collapse".into(),
            passed: current_exact
                .zip(collapse_floor)
                .is_some_and(|(current, floor)| current >= floor),
            measured: current_exact,
            threshold: collapse_floor
                .map(|floor| format!(">= {floor:.8} (0.8 x running best)"))
                .unwrap_or_else(|| "metric required".into()),
            abort_exempt: false,
            abort_exemption_reason: None,
            floor: collapse_floor,
            noise_margin: one_step_margin,
        },
        FoundationV2GateResult {
            name: "composed_changed_exact_collapse".into(),
            passed: !warmup_done
                || current_composed_exact
                    .zip(composed_collapse_floor)
                    .is_some_and(|(current, floor)| current > 0.0 && current >= floor),
            measured: current_composed_exact,
            threshold: if warmup_done {
                composed_collapse_floor
                    .map(|floor| format!("> 0 and >= {floor:.8} (0.8 x armed running best)"))
                    .unwrap_or_else(|| "> 0; metric required".into())
            } else {
                "warmup PASS until step 4096".into()
            },
            abort_exempt: false,
            abort_exemption_reason: None,
            floor: composed_floor_field,
            noise_margin: composed_margin,
        },
    ];
    Ok(FoundationV2GateEvaluation {
        step,
        metrics,
        running_best_before: running_best,
        running_best_after,
        gates,
        diagnostics,
    })
}

fn foundation_v2_floor_gate_abort_exemption(
    name: &str,
    measured: Option<f64>,
    history: &[FoundationV2GateEvaluation],
) -> Option<String> {
    let current = measured?;
    let prior = history
        .last()?
        .gates
        .iter()
        .find(|gate| gate.name == name)?;
    let prior = prior.measured?;
    if ![prior, current].into_iter().all(f64::is_finite) {
        return None;
    }
    // Gate policy v5: one strict improvement on the latest interval suffices.
    // Arm s7 aborted at 0.9567 against a 0.95 floor while improving from
    // 0.9904, because a single warmup-window blip broke the former
    // two-consecutive-improvements requirement. The relaxed rule still aborts
    // plateaus and declines, and an oscillating metric aborts on its first
    // worsening evaluation because that failure is not exempt while the prior
    // failure still counts.
    let (improving, direction) = match name {
        "foreground_reconstruction" => (current > prior, "higher is better"),
        "shuffled_action_ratio" => (current < prior, "lower is better"),
        _ => return None,
    };
    improving.then(|| {
        format!(
            "strict improvement on the latest interval ({prior:.8} -> {current:.8}; {direction})"
        )
    })
}

/// Abort = sustained, significant degradation (gate policy v6): the same
/// named gate must fail in the three latest evaluations, and each failure
/// must violate its recorded floor by more than its one-sided 95% binomial
/// noise margin. Failures within measurement noise still record
/// `passed: false` (they block promotion) but cannot kill the run. Entries
/// without structured floors — legacy histories — count as significant so
/// old evidence still fails closed.
pub const FOUNDATION_V2_ABORT_PATIENCE: usize = 3;

pub fn foundation_v2_gate_history_aborts(history: &[FoundationV2GateEvaluation]) -> bool {
    if history.len() < FOUNDATION_V2_ABORT_PATIENCE {
        return false;
    }
    let window = &history[history.len() - FOUNDATION_V2_ABORT_PATIENCE..];
    let latest = window.last().expect("non-empty abort window");
    latest.gates.iter().any(|gate| {
        window.iter().all(|evaluation| {
            evaluation
                .gates
                .iter()
                .find(|candidate| candidate.name == gate.name)
                .is_some_and(foundation_v2_gate_failure_counts_toward_abort)
        })
    })
}

/// Whether one recorded gate result contributes to the abort window.
/// Significance is recomputed from the structured floor and margin rather
/// than trusted from `abort_exempt`, so a forged exemption flag on a
/// collapse gate still fails closed; the trend exemption remains honored
/// only for the two absolute floor gates that legitimately record it.
fn foundation_v2_gate_failure_counts_toward_abort(gate: &FoundationV2GateResult) -> bool {
    if gate.passed {
        return false;
    }
    if matches!(
        gate.name.as_str(),
        "foreground_reconstruction" | "shuffled_action_ratio"
    ) && gate.abort_exempt
    {
        return false;
    }
    let (Some(floor), Some(margin), Some(measured)) =
        (gate.floor, gate.noise_margin, gate.measured)
    else {
        return true;
    };
    // Structural copy collapse is catastrophic regardless of noise.
    if gate.name == "composed_changed_exact_collapse" && measured == 0.0 {
        return true;
    }
    let violation = match gate.name.as_str() {
        "shuffled_action_ratio" => measured - floor,
        _ => floor - measured,
    };
    violation > margin
}

/// One-sided 95% binomial noise margin for a proportion floor measured on
/// `n` effectively independent rows of the fixed gate population. Gate
/// metrics are sample proportions; a violation smaller than ~1.645 standard
/// errors is indistinguishable from evaluation noise (compounded by the
/// platform's non-reproducible cuDNN f32 kernels) and must not abort a run.
fn foundation_v2_noise_margin(floor: f64, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    let p = floor.clamp(0.0, 1.0);
    1.645 * (p * (1.0 - p) / n as f64).sqrt()
}

fn default_sigreg_target() -> SigregTarget {
    SigregTarget::Marginal
}

fn default_sigreg_temporal_window() -> usize {
    8
}

fn default_sigreg_global_mix() -> f64 {
    0.0
}

/// Optimizer steps for a lesson (`dynamics` / `exploration` get 2× base steps).
pub fn steps_for_lesson(cfg: &TrainConfig, lesson: &str) -> usize {
    match lesson {
        "dynamics" | "exploration" | "factual_branches" => cfg.steps_per_lesson.saturating_mul(2),
        _ => cfg.steps_per_lesson,
    }
}

pub fn resolved_lesson_steps(cfg: &TrainConfig) -> Vec<usize> {
    cfg.lessons
        .iter()
        .map(|lesson| steps_for_lesson(cfg, lesson))
        .collect()
}

pub fn global_step_from_cursor(
    lesson_steps: &[usize],
    lesson_index: usize,
    step_in_lesson: usize,
) -> u64 {
    let prior: usize = lesson_steps.iter().take(lesson_index).sum();
    (prior + step_in_lesson) as u64
}

static PAUSE_REQUESTED: AtomicBool = AtomicBool::new(false);
static PAUSE_HANDLER: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default)]
    pub recipe: TrainingRecipe,
    pub seed: u64,
    pub lessons: Vec<String>,
    pub steps_per_lesson: usize,
    pub physical_batch: usize,
    /// Must be recorded. Microbatches are averaged; one Adam step per effective batch.
    pub grad_accum: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub sigreg_projections: usize,
    pub sigreg_knots: usize,
    pub sigreg_weight: f64,
    /// Training-only shared patch-histogram grounding objective.
    #[serde(default)]
    pub patch_grounding_weight: f64,
    /// Selects target anchoring, predicted-next supervision, or their original
    /// equal mixture without changing model topology.
    #[serde(default)]
    pub patch_grounding_mode: PatchGroundingMode,
    /// Full V4 exact gameplay-pixel grounding on encoded current/next states.
    #[serde(default)]
    pub exact_grounding_weight: f64,
    /// Model initialization seed. None resolves to `seed`.
    #[serde(default)]
    pub init_seed: Option<u64>,
    pub event_weight: f64,
    pub q_weight: f64,
    /// Weight for open-loop latent error on sequential/retarget lessons.
    pub rollout_weight: f64,
    /// Frozen MSE threshold for Q-correctness targets.
    pub q_mse_threshold: f64,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    /// `"cpu"` or `"cuda"` / `"cuda:N"`.
    pub device: String,
    pub output_dir: PathBuf,
    /// Optional complete checkpoint bundle (or run/checkpoints directory) to resume.
    pub resume: Option<PathBuf>,
    /// Explicitly permit continuing a foundation-v2 run whose durable state
    /// records a gate abort.
    #[serde(default)]
    pub resume_after_abort: bool,
    /// Explicitly migrate an equal-effective-batch physical/accumulation schedule.
    /// This is trajectory-changing and is recorded; it is never an exact resume.
    #[serde(default)]
    pub allow_batch_schedule_migration: bool,
    /// Save a complete resumable checkpoint every N optimizer updates. Zero disables it.
    pub checkpoint_every_steps: usize,
    /// Stop cleanly after this many updates in this invocation (scheduler/testing hook).
    pub max_steps_this_run: Option<usize>,
    /// One-based representative optimizer update captured by candle-graph.
    pub profile_update: u64,
    /// One-based read-only loss-gradient attribution probes.
    #[serde(default)]
    pub pressure_updates: Vec<u64>,
    /// Run PTRM ranking loss every N optimizer steps (`1` = every step, `0` = disabled).
    #[serde(default = "default_ptrm_rank_every")]
    pub ptrm_rank_every: usize,
    /// Sample inner/outer recursion depth uniformly in `1..=configured` each optimizer step.
    #[serde(default = "default_randomize_depth")]
    pub randomize_depth: bool,
    /// Fixed recursion depth every step (ignores `randomize_depth`) for steadier GPU load.
    #[serde(default)]
    pub steady_gpu: bool,
    /// Supervise only the final outer step for next-latent MSE (lower VRAM than TRM deep supervision).
    #[serde(default)]
    pub supervise_last_outer_only: bool,
    /// Lesson-scoped loss schedule (dynamics → rollout → events/Q → PTRM).
    #[serde(default = "default_phased_training")]
    pub phased_training: bool,
    /// Stop-gradient on predicted `y` for event loss only (Q keeps full gradients).
    #[serde(default = "default_stop_grad_event_y")]
    pub stop_grad_event_y: bool,
    /// Pre-LN residual dynamics update (see `ModelConfig.residual_y_update`).
    #[serde(default)]
    pub residual_y_update: bool,
    /// Warm-start recursion `y` from encoded state (see `ModelConfig.warm_start_y`).
    #[serde(default)]
    pub warm_start_y: bool,
    /// Apply SIGReg on per-grid-cell channel vectors `(B·H·W)×C` instead of flattened latent.
    #[serde(default)]
    pub sigreg_spatial: bool,
    /// 2×2 avg-pool latents before spatial SIGReg (4× fewer rows; keeps local geometry).
    #[serde(default = "default_sigreg_spatial_pool")]
    pub sigreg_spatial_pool: bool,
    /// Apply SIGReg directly to unpooled pre-RMS encoder cells without learned parameters.
    #[serde(default)]
    pub sigreg_pre_rms_spatial: bool,
    /// Experimental pre-RMS pooled encoder projector with `T×B×D` SIGReg geometry.
    #[serde(default)]
    pub sigreg_projector: bool,
    #[serde(default = "default_sigreg_projector_dim")]
    pub sigreg_projector_dim: usize,
    /// Stop-gradient on `y` for Q BCE and surprise (Q becomes a pure observer).
    #[serde(default)]
    pub stop_grad_q_y: bool,
    /// Label Q positives as transitions below the batch median latent MSE (threshold-free).
    #[serde(default)]
    pub q_quantile_targets: bool,
    /// Gaussian noise on `z` during training forwards (0 = disabled). Applied on ~50% of steps.
    #[serde(default)]
    pub train_z_noise: f64,
    /// Deterministic shuffled episode IDs instead of the sliding `global_step` window.
    #[serde(default)]
    pub shuffled_episodes: bool,
    /// Force D=1 residual baseline (no randomized depth / PTRM in causal runs).
    #[serde(default)]
    pub baseline_d1: bool,
    /// Weight for direct action-prefix prediction loss (Phase C).
    #[serde(default)]
    pub prefix_weight: f64,
    /// Weight for reliability-head BCE (Phase D).
    #[serde(default)]
    pub reliability_weight: f64,
    /// BF16 conv encoder path (Phase B).
    #[serde(default)]
    pub bf16_conv: bool,
    /// Preregistered recurrent-core-only BF16 convolution treatment. F32
    /// master parameters and all recurrent state boundaries remain F32.
    #[serde(default)]
    pub bf16_recurrent_core: bool,
    /// Bootstrap ensemble size for eval uncertainty (Phase D).
    #[serde(default = "default_ensemble_members")]
    pub ensemble_members: usize,
    #[serde(default = "default_muon_momentum")]
    pub muon_momentum: f64,
    #[serde(default = "default_muon_rms_scale")]
    pub muon_rms_scale: f64,
    /// Cap SIGReg row count (0 = no cap). Reduces VRAM for spatial SIGReg.
    #[serde(default = "default_sigreg_max_rows")]
    pub sigreg_max_rows: usize,
    /// Marginal control or per-window temporally centered residual population.
    #[serde(default = "default_sigreg_target")]
    pub sigreg_target: SigregTarget,
    /// Distribution-matching statistic applied to the resolved SIGReg population.
    #[serde(default)]
    pub sigreg_statistic: SigregStatistic,
    /// Ordered transition window size. Ignored by the legacy marginal fallback.
    #[serde(default = "default_sigreg_temporal_window")]
    pub sigreg_temporal_window: usize,
    /// Convex weight on a global-spatial-mean temporal-residual population.
    /// Zero preserves the original 2x2-pooled cell-row TC objective exactly.
    #[serde(default = "default_sigreg_global_mix")]
    pub sigreg_global_mix: f64,
    /// Overlap CPU batch generation with GPU work.
    #[serde(default = "default_prefetch_batches")]
    pub prefetch_batches: bool,
    /// CPU workers allowed to compose foundation-v2 batches ahead of the GPU.
    #[serde(default = "default_data_workers")]
    pub data_workers: usize,
    /// ADR 0005 data contract: whole-frame content, free background colour, no
    /// status row, UNKNOWN conditioning, and the `LearningHistories` stream.
    /// Off keeps every legacy data path byte-identical.
    #[serde(default)]
    pub data_contract_v6: bool,
    /// Intentionally checkpoint-incompatible action-faithful world core.
    #[serde(default)]
    pub world_core_v2: bool,
    /// V3 experiment schema: V2 topology plus residual spatial conditioning
    /// and scale-normalized factual displacement health.
    #[serde(default)]
    pub world_core_v3: bool,
    #[serde(default)]
    pub world_core_v4: bool,
    /// ADR 0005 world-core v6: foundation-v2 topology plus the context
    /// channel. Trajectory-changing; recorded in the training contract.
    #[serde(default)]
    pub world_core_v6: bool,
    /// ADR 0005 §3.5 recursion depth (inner and outer) applied by the
    /// foundation-v2 recipe when `world_core_v6` is set. Default 2; the
    /// deferred depth treatment sets 3. Reaches the contract through
    /// `inner_steps`/`outer_steps`.
    #[serde(default = "default_v6_recursion_steps")]
    pub v6_recursion_steps: usize,
    /// ADR 0005 §3.4 warm start: v5 checkpoint (bundle directory or
    /// `model.safetensors`) whose tensors initialize every non-context
    /// parameter of a fresh v6 run. Without it a v6 config on a v5 checkpoint
    /// fails closed.
    #[serde(default)]
    pub init_context_from_v5: Option<PathBuf>,
    /// Planning-head aggregation at the final spatial prediction seam.
    #[serde(default)]
    pub consumer_readout: ConsumerReadoutTopology,
    /// Localized ACTION6 conditioning, independently switchable inside V2.
    #[serde(default)]
    pub spatial_action_field: bool,
    #[serde(default)]
    pub spatial_action_residual: bool,
    #[serde(default = "default_spatial_action_residual_scale")]
    pub spatial_action_residual_scale: f64,
    /// Foundation-v2 split-CE weighting construction (objective isolation).
    /// Old configs deserialize to `current_double`, the exact ADR 0003 path.
    #[serde(default)]
    pub split_ce_weighting: SplitCeWeighting,
    /// Optional fixed aggregate loss-coefficient share, strictly inside
    /// `(0, 1)`, for the changed stratum. This is not a claim about measured
    /// gradient share. Overrides the active mode's changed weight.
    #[serde(default)]
    pub split_ce_changed_budget: Option<f64>,
    /// Checkpoint-promotion selection metric. Old configs deserialize to
    /// `changed_exact`, the exact historical promotion behavior.
    #[serde(default)]
    pub promotion_metric: PromotionMetric,
    #[serde(default)]
    pub branch_learning: BranchLearningConfig,
    /// Preregistered model-treatment arms for the next matched runs. All
    /// default off; each is recorded in the training contract, and at most
    /// one should be enabled per arm to preserve causal attribution.
    /// Copy-bypass gated outer update (`y' = y + a*(l - y)`, small nonzero `a` init).
    #[serde(default)]
    pub copy_bypass_gate: bool,
    /// Copy-gate bias initialized to `logit(p)` for this changed-pixel prior.
    #[serde(default)]
    pub copy_gate_bias_prior: Option<f64>,
    /// Grid-scaled ACTION6 Gaussian impulse (sigma = one latent cell).
    #[serde(default)]
    pub grid_scaled_action_impulse: bool,
    /// Deployed decode composition (legacy hard gate vs joint copy mixture).
    #[serde(default)]
    pub decode_composition: DecodeComposition,
    /// Native-grid positional-value canonical readout (adds 57,344 params).
    #[serde(default)]
    pub positional_value_readout: bool,
    /// Explicit acknowledgement that this run intentionally enables more than
    /// one model treatment, giving up single-factor causal attribution.
    /// Without it, foundation-v2 validation rejects multi-treatment arms.
    #[serde(default)]
    pub allow_multi_treatment_arm: bool,
    /// One-based optimizer updates captured as full candle-graph evidence
    /// bundles during foundation-v2 training (mechanism observability for
    /// bundle runs). Empty disables periodic capture.
    #[serde(default)]
    pub profile_updates: Vec<u64>,
}

pub fn effective_batch(cfg: &TrainConfig) -> usize {
    cfg.physical_batch.saturating_mul(cfg.grad_accum.max(1))
}

fn effective_batch_contract(contract: &TrainingContract) -> usize {
    contract
        .physical_batch
        .saturating_mul(contract.grad_accum.max(1))
}

fn default_phased_training() -> bool {
    true
}

fn default_stop_grad_event_y() -> bool {
    true
}

fn default_ptrm_rank_every() -> usize {
    4
}

fn default_randomize_depth() -> bool {
    false
}

fn default_ensemble_members() -> usize {
    8
}

fn default_muon_momentum() -> f64 {
    0.95
}

fn default_muon_rms_scale() -> f64 {
    MUON_RMS_SCALE
}

fn default_sigreg_max_rows() -> usize {
    4096
}

fn default_sigreg_spatial_pool() -> bool {
    true
}

fn default_sigreg_projector_dim() -> usize {
    128
}

/// Cap SIGReg rows for tight VRAM (8GB + batch 128 + steady full-depth).
pub fn effective_sigreg_max_rows(cfg: &TrainConfig) -> usize {
    let cap = cfg.sigreg_max_rows;
    if cap == 0 {
        return 0;
    }
    if !cfg.sigreg_spatial {
        return cap;
    }
    // Spatial stack is (B·H·W·2)×C; pool halves H/W when enabled.
    let cells = if cfg.sigreg_spatial_pool { 16 } else { 64 };
    let spatial_rows = cfg.physical_batch.saturating_mul(cells).saturating_mul(2);
    // `sigreg_max_rows` is authoritative. A previous batch-keyed clamp silently
    // pinned this to 1024 rows for any physical_batch >= 128, so raising the
    // batch bought SIGReg no extra samples even though the statistic is what
    // batch size is supposed to improve. The rows are cheap enough that the
    // clamp protected nothing: the full pooled stack at batch 1024 is
    // 32768x128 f32 = 16.8 MiB, against ~6.8 GiB of retained recursion graph.
    cap.min(spatial_rows)
}

fn default_prefetch_batches() -> bool {
    true
}

pub fn default_data_workers() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| (parallelism.get() / 2).clamp(1, 8))
        .unwrap_or(1)
}

fn default_spatial_action_residual_scale() -> f64 {
    0.25
}

pub(crate) fn sync_cuda_device(device: &Device) -> Result<()> {
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(())
}

/// Config fields safe to persist for resume/eval (omit per-run hooks).
fn persist_train_config(cfg: &TrainConfig) -> TrainConfig {
    let mut persisted = cfg.clone();
    persisted.resume = None;
    persisted.resume_after_abort = false;
    persisted.max_steps_this_run = None;
    persisted.allow_batch_schedule_migration = false;
    persisted
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            recipe: TrainingRecipe::LegacyExperimental,
            seed: 1,
            lessons: DEFAULT_LESSONS.iter().map(|s| (*s).to_string()).collect(),
            steps_per_lesson: 2,
            physical_batch: 2,
            grad_accum: 1,
            lr: 1e-3,
            weight_decay: 0.01,
            sigreg_projections: 8,
            sigreg_knots: 5,
            sigreg_weight: 0.003,
            patch_grounding_weight: 0.0,
            patch_grounding_mode: PatchGroundingMode::Both,
            exact_grounding_weight: 0.0,
            init_seed: None,
            event_weight: 0.1,
            q_weight: 0.1,
            rollout_weight: 0.1,
            q_mse_threshold: 0.05,
            hidden_dim: 128,
            action_dim: 8,
            inner_steps: 2,
            outer_steps: 2,
            device: "cpu".into(),
            output_dir: PathBuf::from("runs/p2/smoke"),
            resume: None,
            resume_after_abort: false,
            allow_batch_schedule_migration: false,
            checkpoint_every_steps: 100,
            max_steps_this_run: None,
            profile_update: 2,
            pressure_updates: Vec::new(),
            ptrm_rank_every: 4,
            randomize_depth: false,
            steady_gpu: false,
            supervise_last_outer_only: false,
            phased_training: true,
            stop_grad_event_y: true,
            residual_y_update: false,
            warm_start_y: false,
            sigreg_spatial: false,
            sigreg_spatial_pool: true,
            sigreg_pre_rms_spatial: false,
            sigreg_projector: false,
            sigreg_projector_dim: default_sigreg_projector_dim(),
            stop_grad_q_y: false,
            q_quantile_targets: false,
            train_z_noise: 0.0,
            shuffled_episodes: false,
            baseline_d1: false,
            prefix_weight: 0.0,
            reliability_weight: 0.0,
            bf16_conv: false,
            bf16_recurrent_core: false,
            ensemble_members: 8,
            muon_momentum: 0.95,
            muon_rms_scale: MUON_RMS_SCALE,
            sigreg_max_rows: 4096,
            sigreg_target: SigregTarget::Marginal,
            sigreg_statistic: SigregStatistic::EppsPulley,
            sigreg_temporal_window: 8,
            sigreg_global_mix: 0.0,
            prefetch_batches: true,
            data_workers: default_data_workers(),
            data_contract_v6: false,
            world_core_v2: false,
            world_core_v3: false,
            world_core_v4: false,
            world_core_v6: false,
            v6_recursion_steps: V6_RECURSION_STEPS,
            init_context_from_v5: None,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: default_spatial_action_residual_scale(),
            split_ce_weighting: SplitCeWeighting::CurrentDouble,
            split_ce_changed_budget: None,
            promotion_metric: PromotionMetric::ChangedExact,
            branch_learning: BranchLearningConfig::default(),
            copy_bypass_gate: false,
            copy_gate_bias_prior: None,
            grid_scaled_action_impulse: false,
            decode_composition: DecodeComposition::default(),
            positional_value_readout: false,
            allow_multi_treatment_arm: false,
            profile_updates: Vec::new(),
        }
    }
}

impl TrainConfig {
    pub fn resolved_experiment(&self) -> Result<ResolvedExperiment> {
        ResolvedExperiment::resolve(ExperimentRequest {
            recipe: self.recipe,
            world_core_v2: self.world_core_v2,
            world_core_v3: self.world_core_v3,
            world_core_v4: self.world_core_v4,
            world_core_v6: self.world_core_v6,
            spatial_action_field: self.spatial_action_field,
            spatial_action_residual: self.spatial_action_residual,
            spatial_action_residual_scale: self.spatial_action_residual_scale,
            consumer_readout: self.consumer_readout,
            branch_learning_enabled: self.branch_learning.enabled,
            displacement_health_enabled: self.branch_learning.displacement_health.is_some(),
            sigreg_weight: self.sigreg_weight,
            patch_grounding_weight: self.patch_grounding_weight,
            patch_grounding_mode: self.patch_grounding_mode,
            exact_grounding_weight: self.exact_grounding_weight,
            sigreg_statistic: self.sigreg_statistic,
            sigreg_population: self.sigreg_target,
            sigreg_temporal_window: self.sigreg_temporal_window,
            sigreg_global_mix: self.sigreg_global_mix,
            sigreg_spatial: self.sigreg_spatial,
            sigreg_spatial_pool: self.sigreg_spatial_pool,
            sigreg_pre_rms_spatial: self.sigreg_pre_rms_spatial,
            sigreg_projector: self.sigreg_projector,
            sigreg_projector_dim: self.sigreg_projector_dim,
            lessons: &self.lessons,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.steps_per_lesson == 0 {
            bail!("steps_per_lesson must be > 0");
        }
        if let Some(budget) = self.split_ce_changed_budget {
            if !(budget > 0.0 && budget < 1.0) {
                bail!("split_ce_changed_budget must lie strictly inside (0, 1)");
            }
        }
        if self.physical_batch < 2 {
            bail!("physical_batch must be >= 2 (SIGReg needs batch >= 2)");
        }
        if self.grad_accum == 0 {
            bail!("grad_accum must be >= 1");
        }
        if self.data_workers == 0 {
            bail!("data_workers must be >= 1");
        }
        if self.recipe == TrainingRecipe::FoundationV2 {
            if !self.lessons.is_empty() {
                bail!("foundation-v2 does not use lesson staging");
            }
        } else if self.lessons.is_empty() {
            bail!("at least one lesson is required");
        }
        if self.max_steps_this_run == Some(0) {
            bail!("max_steps_this_run must be > 0 when provided");
        }
        if self.profile_update == 0 {
            bail!("profile_update is one-based and must be > 0");
        }
        if !self.profile_updates.is_empty() {
            if self.recipe != TrainingRecipe::FoundationV2 {
                bail!(
                    "profile_updates is consumed only by the foundation-v2 loop; \
                     setting it elsewhere would silently capture nothing"
                );
            }
            let mut sorted = self.profile_updates.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.first() == Some(&0) || sorted.len() != self.profile_updates.len() {
                bail!("profile_updates must be unique, one-based update numbers");
            }
        }
        if self.pressure_updates.contains(&0) {
            bail!("pressure_updates are one-based and must be > 0");
        }
        if self.recipe != TrainingRecipe::FoundationV2 {
            for lesson in &self.lessons {
                lesson_to_curriculum(lesson)?;
            }
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            bail!("lr must be finite and > 0");
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            bail!("weight_decay must be finite and >= 0");
        }
        if !self.muon_momentum.is_finite() || !(0.0..1.0).contains(&self.muon_momentum) {
            bail!("muon_momentum must be finite and in [0,1)");
        }
        if !self.muon_rms_scale.is_finite() || self.muon_rms_scale <= 0.0 {
            bail!("muon_rms_scale must be finite and > 0");
        }
        for (name, weight) in [
            ("sigreg_weight", self.sigreg_weight),
            ("patch_grounding_weight", self.patch_grounding_weight),
            ("exact_grounding_weight", self.exact_grounding_weight),
            ("event_weight", self.event_weight),
            ("q_weight", self.q_weight),
            ("rollout_weight", self.rollout_weight),
        ] {
            if !(weight.is_finite() && weight >= 0.0) {
                bail!("{name} must be finite and >= 0");
            }
        }
        if self.sigreg_projections == 0 || self.sigreg_knots < 3 {
            bail!("sigreg_projections >= 1 and sigreg_knots >= 3 required");
        }
        if !(self.q_mse_threshold.is_finite() && self.q_mse_threshold >= 0.0) {
            bail!("q_mse_threshold must be finite and >= 0");
        }
        if !self.train_z_noise.is_finite() || self.train_z_noise < 0.0 {
            bail!("train_z_noise must be finite and >= 0");
        }
        self.branch_learning.validate(self.grad_accum)?;
        let enabled_treatments = usize::from(self.bf16_recurrent_core)
            + usize::from(self.copy_bypass_gate)
            + usize::from(self.copy_gate_bias_prior.is_some())
            + usize::from(self.grid_scaled_action_impulse)
            + usize::from(self.decode_composition != DecodeComposition::LegacyHardGate)
            + usize::from(self.positional_value_readout);
        if enabled_treatments > 1 && !self.allow_multi_treatment_arm {
            bail!(
                "{enabled_treatments} model treatments are enabled; single-factor \
                 attribution requires one per arm (set allow_multi_treatment_arm \
                 to intentionally give that up)"
            );
        }
        if (self.world_core_v6 || self.init_context_from_v5.is_some())
            && self.recipe != TrainingRecipe::FoundationV2
        {
            bail!("world_core_v6 / init_context_from_v5 require the foundation-v2 recipe");
        }
        if self.init_context_from_v5.is_some() && !self.world_core_v6 {
            bail!("init_context_from_v5 is only meaningful for a world_core_v6 run");
        }
        if !(1..=4).contains(&self.v6_recursion_steps) {
            bail!(
                "v6_recursion_steps must be in 1..=4, got {}",
                self.v6_recursion_steps
            );
        }
        if self.v6_recursion_steps != V6_RECURSION_STEPS && !self.world_core_v6 {
            bail!("v6_recursion_steps applies only to world_core_v6 runs");
        }
        if self.data_contract_v6 && !self.world_core_v6 {
            bail!(
                "data_contract_v6 requires world_core_v6: the v6 data contract supervises \
                 all 64 rows and only a v6 model decodes them (ADR 0005 §1.1)"
            );
        }
        if self.world_core_v6 && !self.data_contract_v6 && self.init_context_from_v5.is_none() {
            bail!(
                "world_core_v6 without data_contract_v6 is only allowed as an \
                 init_context_from_v5 warm-start smoke; a v6 model trained on legacy \
                 63-row data violates the whole-frame contract (ADR 0005 §1.1). Set \
                 data_contract_v6 or supply init_context_from_v5"
            );
        }
        if self.recipe == TrainingRecipe::FullV4 {
            self.validate_full_v4()?;
        } else if self.recipe == TrainingRecipe::FoundationV2 {
            self.validate_foundation_v2()?;
        }
        let resolved = self.resolved_experiment()?;
        if self.recipe != TrainingRecipe::FoundationV2
            && resolved.factual_learning
            && !self
                .physical_batch
                .is_multiple_of(crate::p2::data::FACTUAL_BRANCHES_PER_GROUP)
        {
            bail!(
                "action-faithful physical_batch must be a multiple of {} so factual groups cannot be truncated",
                crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
            );
        }
        Ok(())
    }

    pub fn model_config(&self) -> ModelConfig {
        ModelConfig {
            frame_side: FRAME_SIDE,
            patch_size: if self.recipe == TrainingRecipe::FoundationV2 {
                PATCH_SIZE
            } else {
                LEGACY_PATCH_SIZE
            },
            hidden_dim: self.hidden_dim,
            action_dim: self.action_dim,
            goal_dim: GOAL_FEATURES_DIM,
            inner_steps: self.inner_steps,
            outer_steps: self.outer_steps,
            num_events: DEFAULT_NUM_EVENTS,
            residual_y_update: self.residual_y_update,
            warm_start_y: self.warm_start_y,
            bf16_conv: self.bf16_conv,
            bf16_recurrent_core: self.bf16_recurrent_core,
            sigreg_projector: self.sigreg_projector,
            sigreg_projector_dim: self.sigreg_projector_dim,
            spatial_action_field: self.spatial_action_field,
            spatial_action_residual: self.spatial_action_residual,
            spatial_action_residual_scale: self.spatial_action_residual_scale,
            world_core_v2: self.world_core_v2,
            world_core_v3: self.world_core_v3,
            world_core_v4: self.world_core_v4,
            world_core_v5: self.recipe == TrainingRecipe::FoundationV2,
            world_core_v6: self.world_core_v6,
            consumer_readout: self.consumer_readout,
            copy_bypass_gate: self.copy_bypass_gate,
            copy_gate_bias_prior: self.copy_gate_bias_prior,
            grid_scaled_action_impulse: self.grid_scaled_action_impulse,
            decode_composition: self.decode_composition,
            positional_value_readout: self.positional_value_readout,
        }
    }

    /// Resolve the one supported V4 training contract. Runtime/provenance
    /// values (seed, steps, batch, device, output, checkpoints) remain caller
    /// controlled; model and objective choices do not.
    pub fn apply_full_v4_recipe(&mut self) {
        self.recipe = TrainingRecipe::FullV4;
        self.lessons = DEFAULT_LESSONS.iter().map(|s| (*s).to_string()).collect();
        // `grad_accum` is a runtime batch-schedule field (caller-owned, recorded
        // in the contract); recipes must not overwrite it (2026-09-04 audit).
        self.sigreg_projections = 1024;
        self.sigreg_knots = 17;
        self.sigreg_weight = 0.1;
        self.sigreg_max_rows = 0;
        self.sigreg_target = SigregTarget::Marginal;
        self.sigreg_statistic = SigregStatistic::EppsPulley;
        self.sigreg_global_mix = 0.0;
        self.sigreg_spatial = false;
        self.sigreg_spatial_pool = false;
        self.sigreg_pre_rms_spatial = false;
        self.sigreg_projector = false;
        self.patch_grounding_weight = 0.0;
        self.exact_grounding_weight = 0.1;
        self.event_weight = 0.1;
        self.q_weight = 0.1;
        self.rollout_weight = 0.1;
        self.reliability_weight = 0.1;
        self.prefix_weight = 0.0;
        self.ptrm_rank_every = 0;
        self.randomize_depth = false;
        self.supervise_last_outer_only = true;
        self.phased_training = true;
        self.stop_grad_event_y = true;
        self.stop_grad_q_y = true;
        self.q_quantile_targets = false;
        self.train_z_noise = 0.0;
        self.baseline_d1 = false;
        self.residual_y_update = true;
        self.warm_start_y = true;
        self.hidden_dim = 128;
        self.action_dim = 32;
        self.inner_steps = 2;
        self.outer_steps = 2;
        self.lr = 1e-3;
        self.weight_decay = 0.01;
        self.muon_momentum = 0.95;
        self.muon_rms_scale = MUON_RMS_SCALE;
        self.bf16_conv = false;
        self.bf16_recurrent_core = false;
        self.shuffled_episodes = true;
        self.world_core_v2 = false;
        self.world_core_v3 = false;
        self.world_core_v4 = true;
        self.consumer_readout = ConsumerReadoutTopology::SpatialQuery;
        self.spatial_action_field = true;
        self.spatial_action_residual = false;
        self.branch_learning = BranchLearningConfig::default();
    }

    /// Resolve ADR 0003's fixed model/objective choices. Runtime controls —
    /// seed, total steps (`steps_per_lesson` storage for compatibility), batch,
    /// device, output, and checkpoint cadence — remain caller-owned.
    pub fn apply_foundation_v2_recipe(&mut self) {
        self.recipe = TrainingRecipe::FoundationV2;
        self.lessons.clear();
        // `grad_accum` is a runtime batch-schedule field (caller-owned, recorded
        // in the contract); recipes must not overwrite it (2026-09-04 audit).
        self.sigreg_projections = 1024;
        self.sigreg_knots = 17;
        self.sigreg_weight = 0.01;
        self.sigreg_max_rows = 0;
        self.sigreg_target = SigregTarget::Marginal;
        self.sigreg_statistic = SigregStatistic::EppsPulley;
        self.sigreg_global_mix = 0.0;
        self.sigreg_spatial = false;
        self.sigreg_spatial_pool = false;
        self.sigreg_pre_rms_spatial = false;
        self.sigreg_projector = false;
        self.patch_grounding_weight = 0.0;
        self.exact_grounding_weight = 0.0;
        self.event_weight = 0.1;
        self.q_weight = 0.1;
        self.rollout_weight = 0.02;
        self.reliability_weight = 0.1;
        self.prefix_weight = 0.0;
        self.ptrm_rank_every = 0;
        self.randomize_depth = false;
        self.supervise_last_outer_only = true;
        self.phased_training = false;
        self.stop_grad_event_y = true;
        self.stop_grad_q_y = true;
        self.q_quantile_targets = false;
        self.train_z_noise = 0.0;
        self.baseline_d1 = false;
        self.residual_y_update = true;
        self.warm_start_y = true;
        self.hidden_dim = 128;
        self.action_dim = 32;
        // ADR 0005 §3.5: each outer iteration executes `inner_steps + 1`
        // applications of the two-convolution residual block. Thus 2x2 is 6
        // blocks / 12 convolutions / receptive field 25, while 3x3 is 12 / 24
        // / 49. Both already cover the 16x16 latent grid; 2x2 is the v6
        // baseline contract and 3x3 remains an explicit treatment arm, not a
        // proven receptive-field necessity. v5 stays 2x2 for reproducibility.
        let depth = if self.world_core_v6 {
            self.v6_recursion_steps
        } else {
            2
        };
        self.inner_steps = depth;
        self.outer_steps = depth;
        self.lr = 1e-3;
        self.weight_decay = 0.01;
        self.muon_momentum = 0.95;
        self.muon_rms_scale = MUON_RMS_SCALE;
        self.bf16_conv = false;
        self.shuffled_episodes = false;
        self.world_core_v2 = false;
        self.world_core_v3 = false;
        self.world_core_v4 = true;
        self.consumer_readout = ConsumerReadoutTopology::SpatialQuery;
        self.spatial_action_field = true;
        self.spatial_action_residual = false;
        self.branch_learning = BranchLearningConfig::default();
    }

    fn validate_foundation_v2(&self) -> Result<()> {
        if self.seed == FOUNDATION_V2_GATE_SEED {
            bail!("foundation-v2 training seed is reserved by the frozen gate population");
        }
        let mut canonical = self.clone();
        canonical.apply_foundation_v2_recipe();
        // Preserve the runtime/provenance fields explicitly left caller-owned.
        canonical.seed = self.seed;
        canonical.steps_per_lesson = self.steps_per_lesson;
        canonical.physical_batch = self.physical_batch;
        canonical.grad_accum = self.grad_accum;
        canonical.device = self.device.clone();
        canonical.output_dir = self.output_dir.clone();
        canonical.resume = self.resume.clone();
        canonical.resume_after_abort = self.resume_after_abort;
        canonical.allow_batch_schedule_migration = self.allow_batch_schedule_migration;
        canonical.checkpoint_every_steps = self.checkpoint_every_steps;
        canonical.max_steps_this_run = self.max_steps_this_run;
        canonical.init_seed = self.init_seed;
        canonical.profile_update = self.profile_update;
        canonical.pressure_updates = self.pressure_updates.clone();
        canonical.profile_updates = self.profile_updates.clone();
        canonical.prefetch_batches = self.prefetch_batches;
        canonical.data_workers = self.data_workers;
        // Preregistered model-treatment arms: caller-owned by design, recorded
        // in the training contract so a resume across arms fails closed. At
        // most one should be enabled per matched run.
        canonical.bf16_recurrent_core = self.bf16_recurrent_core;
        canonical.copy_bypass_gate = self.copy_bypass_gate;
        canonical.copy_gate_bias_prior = self.copy_gate_bias_prior;
        canonical.grid_scaled_action_impulse = self.grid_scaled_action_impulse;
        canonical.decode_composition = self.decode_composition;
        canonical.positional_value_readout = self.positional_value_readout;
        canonical.allow_multi_treatment_arm = self.allow_multi_treatment_arm;
        canonical.world_core_v6 = self.world_core_v6;
        canonical.init_context_from_v5 = self.init_context_from_v5.clone();
        if self != &canonical {
            bail!("foundation-v2 recipe contains a caller-overridden fixed model/loss switch");
        }
        if self.physical_batch < crate::p2::data::FACTUAL_BRANCHES_PER_GROUP {
            bail!(
                "foundation-v2 physical_batch must be at least {}",
                crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
            );
        }
        Ok(())
    }

    fn validate_full_v4(&self) -> Result<()> {
        let expected_lessons = DEFAULT_LESSONS
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>();
        ensure_full_v4(self.lessons == expected_lessons, "canonical lesson order")?;
        ensure_full_v4(self.grad_accum == 1, "grad_accum=1")?;
        ensure_full_v4(
            self.sigreg_projections == 1024
                && self.sigreg_knots == 17
                && self.sigreg_weight == 0.1
                && self.sigreg_max_rows == 0
                && self.sigreg_target == SigregTarget::Marginal
                && self.sigreg_statistic == SigregStatistic::EppsPulley
                && self.sigreg_global_mix == 0.0
                && !self.sigreg_spatial
                && !self.sigreg_spatial_pool
                && !self.sigreg_pre_rms_spatial
                && !self.sigreg_projector,
            "uncapped marginal EP(1024 projections, 17 knots, weight 0.1)",
        )?;
        ensure_full_v4(
            self.hidden_dim == 128
                && self.action_dim == 32
                && self.inner_steps == 2
                && self.outer_steps == 2,
            "hidden=128, action=32, inner=2, outer=2 architecture",
        )?;
        ensure_full_v4(
            self.lr == 1e-3
                && self.weight_decay == 0.01
                && self.muon_momentum == 0.95
                && self.muon_rms_scale == MUON_RMS_SCALE
                && !self.bf16_conv
                && !self.bf16_recurrent_core
                && self.shuffled_episodes,
            "fixed optimizer, precision, and episode-order contract",
        )?;
        ensure_full_v4(
            self.patch_grounding_weight == 0.0 && self.exact_grounding_weight == 0.1,
            "exact current/target grounding at weight 0.1",
        )?;
        ensure_full_v4(
            self.event_weight == 0.1
                && self.q_weight == 0.1
                && self.rollout_weight == 0.1
                && self.reliability_weight == 0.1
                && self.phased_training,
            "fixed auxiliary coefficients and phased schedule",
        )?;
        ensure_full_v4(
            self.world_core_v4
                && !self.world_core_v2
                && !self.world_core_v3
                && self.consumer_readout == ConsumerReadoutTopology::SpatialQuery
                && self.spatial_action_field
                && !self.spatial_action_residual,
            "exclusive V4 topology with SpatialQuery and spatial action conditioning",
        )?;
        ensure_full_v4(
            self.residual_y_update
                && self.warm_start_y
                && self.supervise_last_outer_only
                && !self.randomize_depth
                && self.train_z_noise == 0.0,
            "fixed deterministic residual recurrence",
        )?;
        ensure_full_v4(
            self.stop_grad_event_y
                && self.stop_grad_q_y
                && !self.q_quantile_targets
                && self.ptrm_rank_every == 0
                && self.prefix_weight == 0.0
                && !self.branch_learning.enabled,
            "frozen observer stages and excluded experimental objectives",
        )?;
        Ok(())
    }
}

fn ensure_full_v4(condition: bool, invariant: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        bail!("full-v4 recipe invariant violated: {invariant}")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LessonLossMeans {
    pub total: f64,
    pub next_latent: f64,
    pub rollout: f64,
    pub sigreg_raw: f64,
    pub sigreg_bounded: f64,
    #[serde(default)]
    pub patch_grounding: f64,
    #[serde(default)]
    pub grounding_changed_patches: f64,
    #[serde(default)]
    pub grounding_unchanged_patches: f64,
    #[serde(default)]
    pub pre_clip_gradient_norm: f64,
    #[serde(default)]
    pub gradient_clip_scale: f64,
    #[serde(default)]
    pub clipped_updates: f64,
    pub event: f64,
    pub q: f64,
    #[serde(default)]
    pub prefix: f64,
    #[serde(default)]
    pub reliability: f64,
    #[serde(default)]
    pub branch_total: f64,
    #[serde(default)]
    pub outcome_pull: f64,
    #[serde(default)]
    pub outcome_push: f64,
    #[serde(default)]
    pub action_recovery: f64,
    #[serde(default)]
    pub coordinate_recovery: f64,
    #[serde(default)]
    pub changed_margin: f64,
    #[serde(default)]
    pub spatial_variance: f64,
    #[serde(default)]
    pub spatial_covariance: f64,
    #[serde(default)]
    pub pooled_variance: f64,
    #[serde(default)]
    pub pooled_covariance: f64,
    #[serde(default)]
    pub displacement_variance: f64,
    #[serde(default)]
    pub displacement_covariance: f64,
    #[serde(default)]
    pub branch_groups: f64,
    #[serde(default)]
    pub changed_branches: f64,
    #[serde(default)]
    pub equivalent_pairs: f64,
    #[serde(default)]
    pub distinct_pairs: f64,
    #[serde(default)]
    pub action6_branches: f64,
    #[serde(default)]
    pub action_recovery_branches: f64,
    #[serde(default)]
    pub spatial_population_rows: f64,
    #[serde(default)]
    pub pooled_population_rows: f64,
    #[serde(default)]
    pub displacement_population_rows: f64,
    #[serde(default)]
    pub unique_changed_outcomes: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessonReport {
    pub lesson: String,
    pub curriculum: String,
    pub steps: usize,
    pub mean_losses: LessonLossMeans,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2LossMeans {
    pub total: f64,
    pub pred_ce: f64,
    pub gate: f64,
    pub latent: f64,
    pub enc_ce: f64,
    pub separation: f64,
    pub pull: f64,
    pub inverse_action: f64,
    pub ep: f64,
    pub rollout: f64,
    pub event: f64,
    pub q: f64,
    pub reliability: f64,
    pub pre_clip_gradient_norm: f64,
    pub gradient_clip_scale: f64,
    #[serde(default)]
    pub clipped_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2EpGradientSample {
    pub step: u64,
    pub encoder_ep_gradient_l2: f64,
    pub encoder_prediction_gradient_l2: f64,
    pub ep_weight: f64,
    /// Achieved `weight * ||g_ep|| / ||g_pred||`; `None` when undefined.
    #[serde(default)]
    pub weighted_budget_ratio: Option<f64>,
    /// Whether the achieved ratio satisfies the `<= 0.3` ADR bound.
    #[serde(default)]
    pub budget_met: Option<bool>,
    /// Which controller rail produced the weight, if any.
    #[serde(default)]
    pub rail: Option<FoundationV2EpRail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoundationV2EpRail {
    DisabledForBudget,
    LowerBound,
    UpperBound,
    Interior,
}

/// Classify the controller output and its achieved budget compliance.
pub fn foundation_v2_ep_budget_status(
    weight: f64,
    ep_gradient_l2: f64,
    prediction_gradient_l2: f64,
) -> (Option<f64>, Option<bool>, FoundationV2EpRail) {
    let rail = if weight == 0.0 {
        FoundationV2EpRail::DisabledForBudget
    } else if weight <= FOUNDATION_V2_EP_MIN_WEIGHT {
        FoundationV2EpRail::LowerBound
    } else if weight >= FOUNDATION_V2_EP_MAX_WEIGHT {
        FoundationV2EpRail::UpperBound
    } else {
        FoundationV2EpRail::Interior
    };
    let weighted = weight * ep_gradient_l2;
    if prediction_gradient_l2 > 0.0 && weighted.is_finite() {
        let ratio = weighted / prediction_gradient_l2;
        let met = ratio <= FOUNDATION_V2_EP_GRADIENT_BUDGET * (1.0 + 1e-9);
        (Some(ratio), Some(met), rail)
    } else {
        // A zero prediction gradient permits only zero EP pressure.
        (None, Some(weighted == 0.0), rail)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2MechanismSample {
    pub step: u64,
    pub copy_bypass_alpha: Option<f64>,
    pub outer_step_cosines: Vec<f64>,
    pub gate_open_rate: f64,
    pub gate_mean_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2TrainingReport {
    pub total_steps: usize,
    pub mean_losses: FoundationV2LossMeans,
    pub ep_weight: f64,
    pub ep_gradient_budget: Vec<FoundationV2EpGradientSample>,
    pub gate_history: Vec<FoundationV2GateEvaluation>,
    pub best_changed_exact: Option<f64>,
    #[serde(default)]
    pub promotion_metric: PromotionMetric,
    #[serde(default)]
    pub best_promotion_value: Option<f64>,
    pub best_checkpoint: Option<PathBuf>,
    pub rollout_enabled: bool,
    pub permanent_checkpoints: Vec<PathBuf>,
    /// Exact support observed in consumed generated rows, ordered as
    /// noop/satisfied/failed/exhausted.
    #[serde(default)]
    pub event_label_census: EventLabelCensus,
    /// False for checkpoints created before event-census tracking began; in
    /// that case the counts cover only post-resume rows and must not be read as
    /// the complete consumed population.
    #[serde(default)]
    pub event_label_census_complete: bool,
    /// Cheap EMA-weight mechanism diagnostics sampled with each trainer gate.
    #[serde(default)]
    pub mechanism_history: Vec<FoundationV2MechanismSample>,
    /// Atomically published candle-graph bundles for preregistered updates.
    #[serde(default)]
    pub profile_bundles: Vec<PathBuf>,
    pub clip_strategy: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStatus {
    Completed,
    Paused,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReport {
    pub schema: String,
    #[serde(default = "default_legacy_world_core_schema")]
    pub world_core_schema: String,
    #[serde(default)]
    pub experiment: ResolvedExperiment,
    pub seed: u64,
    pub physical_batch: usize,
    pub grad_accum: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub parameter_count: usize,
    /// Legacy FNV-1a provenance fingerprint of every generated training row
    /// consumed by completed optimizer updates. It deliberately excludes
    /// content masks and content origins; use `training_content_fingerprint`
    /// when those objective inputs must be bound.
    pub training_population_fingerprint: String,
    /// Cryptographic chain over provenance, exact content masks, and complete
    /// current/next frames.
    #[serde(default)]
    pub training_content_fingerprint: String,
    pub training_population_rows: u64,
    pub device: String,
    pub lessons: Vec<LessonReport>,
    pub status: TrainStatus,
    /// Number of completed optimizer updates across this run and all resumes.
    pub global_step: u64,
    /// Complete bundle from which training can resume exactly.
    pub latest_checkpoint: PathBuf,
    pub resumed_from: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_schedule_migrations: Vec<BatchScheduleMigration>,
    pub checkpoint: PathBuf,
    /// Weights exported for eval when a pre-retarget snapshot exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_checkpoint: Option<PathBuf>,
    pub config_path: PathBuf,
    /// Published representative-update evidence, if the configured update completed.
    pub profile: ProfileState,
    /// One read-only attribution probe immediately before `profile_update`.
    /// Its gradients are discarded and never reach the optimizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_pressure: Option<GradientPressureDiagnostics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gradient_pressure_samples: Vec<GradientPressureDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foundation_v2: Option<FoundationV2TrainingReport>,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
    /// Launches of this run root after the fresh start (each one is a resume).
    #[serde(default)]
    pub resume_count: u64,
    /// Every launch of this run root, oldest first, with the source revision
    /// and binary that produced it and any loss-log repair it performed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub run_attempts: Vec<RunAttempt>,
}

/// One launch of a run root. Persisted to `run_attempts.jsonl` before any
/// training happens so attempts that die before their first checkpoint are
/// still on record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAttempt {
    /// One-based launch index within the run root.
    pub attempt: u64,
    pub kind: RunAttemptKind,
    pub started_unix_secs: u64,
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_from: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_step: Option<u64>,
    #[serde(flatten)]
    pub provenance: crate::p2::evidence::LaunchProvenance,
    /// Durable repair-journal state. A pending state in a later report proves
    /// the process stopped after its start record but before its result event.
    #[serde(default = "default_completed_repair_state")]
    pub repair_state: RunAttemptRepairState,
    /// Loss-log reconciliation performed by this resume, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_log_repair: Option<LossLogRepair>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_failure: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunAttemptKind {
    Fresh,
    Resume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunAttemptRepairState {
    Pending,
    Completed,
    Failed,
}

fn default_completed_repair_state() -> RunAttemptRepairState {
    // Legacy attempt rows were appended only after repair had completed.
    RunAttemptRepairState::Completed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
// Journal events are serialized and immediately discarded; keeping the
// schema-shaped record inline is clearer than adding allocation solely to
// equalize transient enum variant sizes.
#[allow(clippy::large_enum_variant)]
enum RunAttemptJournalEvent {
    Started {
        record: RunAttempt,
    },
    RepairCompleted {
        attempt: u64,
        loss_log_repair: Option<LossLogRepair>,
    },
    RepairFailed {
        attempt: u64,
        error: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RunAttemptJournalLine {
    Event(RunAttemptJournalEvent),
    Legacy(RunAttempt),
}

const RUN_ATTEMPTS_FILE: &str = "run_attempts.jsonl";

pub(crate) fn read_run_attempts(output_dir: &Path) -> Result<Vec<RunAttempt>> {
    let path = output_dir.join(RUN_ATTEMPTS_FILE);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let mut attempts: Vec<RunAttempt> = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        match serde_json::from_str::<RunAttemptJournalLine>(line)
            .with_context(|| format!("parse {}", path.display()))?
        {
            RunAttemptJournalLine::Legacy(attempt) => {
                if attempt.attempt != attempts.len() as u64 + 1 {
                    bail!(
                        "non-contiguous legacy attempt {} in {}",
                        attempt.attempt,
                        path.display()
                    );
                }
                attempts.push(attempt);
            }
            RunAttemptJournalLine::Event(RunAttemptJournalEvent::Started { record }) => {
                if record.attempt != attempts.len() as u64 + 1
                    || record.repair_state != RunAttemptRepairState::Pending
                {
                    bail!(
                        "invalid attempt start {} in {}",
                        record.attempt,
                        path.display()
                    );
                }
                attempts.push(record);
            }
            RunAttemptJournalLine::Event(RunAttemptJournalEvent::RepairCompleted {
                attempt,
                loss_log_repair,
            }) => {
                let record = attempts
                    .iter_mut()
                    .find(|record| record.attempt == attempt)
                    .ok_or_else(|| anyhow::anyhow!("completion for unknown attempt {attempt}"))?;
                if record.repair_state != RunAttemptRepairState::Pending {
                    bail!("duplicate repair result for attempt {attempt}");
                }
                record.repair_state = RunAttemptRepairState::Completed;
                record.loss_log_repair = loss_log_repair;
            }
            RunAttemptJournalLine::Event(RunAttemptJournalEvent::RepairFailed {
                attempt,
                error,
            }) => {
                let record = attempts
                    .iter_mut()
                    .find(|record| record.attempt == attempt)
                    .ok_or_else(|| anyhow::anyhow!("failure for unknown attempt {attempt}"))?;
                if record.repair_state != RunAttemptRepairState::Pending {
                    bail!("duplicate repair result for attempt {attempt}");
                }
                record.repair_state = RunAttemptRepairState::Failed;
                record.repair_failure = Some(error);
            }
        }
    }
    Ok(attempts)
}

fn append_run_attempt_event(output_dir: &Path, event: &RunAttemptJournalEvent) -> Result<()> {
    let path = output_dir.join(RUN_ATTEMPTS_FILE);
    let mut journal = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    if !journal.is_empty() && !journal.ends_with(b"\n") {
        bail!(
            "refusing to append to torn attempt journal {}",
            path.display()
        );
    }
    let mut line = serde_json::to_vec(event).context("serialize run attempt event")?;
    line.push(b'\n');
    journal.extend_from_slice(&line);
    let tmp = unused_checkpoint_sibling(output_dir, "run_attempts.jsonl.append");
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        file.write_all(&journal)
            .and_then(|_| file.sync_all())
            .with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, &path)
            .with_context(|| format!("publish attempt journal {}", path.display()))?;
        File::open(output_dir)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Durably record the attempt before any loss-log mutation.
fn begin_run_attempt(
    output_dir: &Path,
    resumed_from: Option<&Path>,
    resumed_step: Option<u64>,
) -> Result<u64> {
    let attempts = read_run_attempts(output_dir)?;
    let attempt = RunAttempt {
        attempt: attempts.len() as u64 + 1,
        kind: if resumed_from.is_some() {
            RunAttemptKind::Resume
        } else {
            RunAttemptKind::Fresh
        },
        started_unix_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default(),
        pid: std::process::id(),
        resumed_from: resumed_from.map(Path::to_path_buf),
        resumed_step,
        provenance: crate::p2::evidence::launch_provenance().clone(),
        repair_state: RunAttemptRepairState::Pending,
        loss_log_repair: None,
        repair_failure: None,
    };
    let attempt_number = attempt.attempt;
    append_run_attempt_event(
        output_dir,
        &RunAttemptJournalEvent::Started { record: attempt },
    )?;
    Ok(attempt_number)
}

fn complete_run_attempt_repair(
    output_dir: &Path,
    attempt: u64,
    loss_log_repair: Option<LossLogRepair>,
) -> Result<Vec<RunAttempt>> {
    append_run_attempt_event(
        output_dir,
        &RunAttemptJournalEvent::RepairCompleted {
            attempt,
            loss_log_repair,
        },
    )?;
    read_run_attempts(output_dir)
}

fn fail_run_attempt_repair(output_dir: &Path, attempt: u64, error: &anyhow::Error) -> Result<()> {
    append_run_attempt_event(
        output_dir,
        &RunAttemptJournalEvent::RepairFailed {
            attempt,
            error: format!("{error:#}"),
        },
    )
}

fn resume_count(attempts: &[RunAttempt]) -> u64 {
    attempts
        .iter()
        .filter(|attempt| attempt.kind == RunAttemptKind::Resume)
        .count() as u64
}

/// Outcome of reconciling `loss_log.jsonl` with the checkpoint a resume
/// starts from. Rows the resumed trajectory does not own are moved, in their
/// original order, to `loss_log.jsonl.attempt-N` (never deleted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossLogRepair {
    pub resumed_step: u64,
    pub rows_before: usize,
    pub rows_kept: usize,
    pub rows_removed: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_rows_path: Option<PathBuf>,
}

/// Move only a strictly increasing suffix beyond `resumed_step` out of the
/// active log. Duplicate, nonmonotonic, or malformed histories do not encode
/// checkpoint ancestry, so they fail closed without rewriting either file.
fn repair_loss_log_for_resume(
    output_dir: &Path,
    resumed_step: u64,
    attempt: u64,
) -> Result<Option<LossLogRepair>> {
    let path = output_dir.join("loss_log.jsonl");
    if !path.is_file() {
        return Ok(None);
    }
    let contents = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let rows_before = lines.len();
    let mut steps = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        let step = serde_json::from_str::<serde_json::Value>(line)
            .with_context(|| format!("parse {} row {} before repair", path.display(), index + 1))?
            .get("global_step")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} row {} has no unsigned global_step",
                    path.display(),
                    index + 1
                )
            })?;
        if let Some(previous) = steps.last() {
            if step <= *previous {
                bail!(
                    "refusing ambiguous loss-log repair for resume step {resumed_step}: {} row {} has global_step {step} after {previous}; duplicate/nonmonotonic ordering does not establish checkpoint lineage",
                    path.display(),
                    index + 1
                );
            }
        }
        steps.push(step);
    }
    let rows_kept = steps.partition_point(|step| *step <= resumed_step);
    let rows_removed = rows_before.saturating_sub(rows_kept);
    if rows_removed == 0 {
        return Ok(Some(LossLogRepair {
            resumed_step,
            rows_before,
            rows_kept,
            rows_removed: 0,
            removed_rows_path: None,
        }));
    }
    let sidecar = output_dir.join(format!("loss_log.jsonl.attempt-{attempt}"));
    let mut removed_text = String::new();
    for line in &lines[rows_kept..] {
        removed_text.push_str(line);
        removed_text.push('\n');
    }
    // Sidecar first (create_new: never clobber), then the repaired log
    // atomically; a crash between the two leaves the original log intact.
    let mut sidecar_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&sidecar)
        .with_context(|| format!("create {}", sidecar.display()))?;
    sidecar_file
        .write_all(removed_text.as_bytes())
        .and_then(|_| sidecar_file.sync_all())
        .with_context(|| format!("write {}", sidecar.display()))?;
    let mut kept_text = String::new();
    for line in &lines[..rows_kept] {
        kept_text.push_str(line);
        kept_text.push('\n');
    }
    let tmp = unused_checkpoint_sibling(output_dir, "loss_log.jsonl.repair");
    fs::write(&tmp, kept_text.as_bytes()).with_context(|| format!("write {}", tmp.display()))?;
    File::open(&tmp)?.sync_all()?;
    fs::rename(&tmp, &path)
        .with_context(|| format!("publish repaired loss log {}", path.display()))?;
    File::open(output_dir)?.sync_all()?;
    tracing::warn!(
        "loss log {} reconciled with resume step {resumed_step}: kept {} rows, moved {} stale rows to {}",
        path.display(),
        rows_kept,
        rows_removed,
        sidecar.display()
    );
    Ok(Some(LossLogRepair {
        resumed_step,
        rows_before,
        rows_kept,
        rows_removed,
        removed_rows_path: Some(sidecar),
    }))
}

fn default_legacy_world_core_schema() -> String {
    "legacy_p2_eval_compatible".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientPressureDiagnostics {
    pub update: u64,
    pub encoder_next_latent_l2: f64,
    pub encoder_sigreg_weighted_l2: f64,
    pub sigreg_to_next_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_grounding_weighted_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding_to_next_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding_head_weighted_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigreg_next_cosine: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding_next_cosine: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding_sigreg_cosine: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_readout_weighted_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readout_to_next_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_next_latent_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displacement_health_weighted_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displacement_health_to_next_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrainingContract {
    // Deliberately no serde default: an older state lacking the revision must
    // fail deserialization rather than hydrate into a changed objective.
    foundation_objective_revision: u32,
    #[serde(default)]
    recipe: TrainingRecipe,
    seed: u64,
    lessons: Vec<String>,
    steps_per_lesson: usize,
    #[serde(default)]
    lesson_steps: Vec<usize>,
    physical_batch: usize,
    grad_accum: usize,
    profile_update: u64,
    /// `None` identifies checkpoints written before this field existed. On
    /// load, only that legacy case adopts the configured preregistration list;
    /// newly written contracts compare the full list exactly on resume.
    #[serde(default)]
    profile_updates: Option<Vec<u64>>,
    #[serde(default)]
    pressure_updates: Vec<u64>,
    lr: f64,
    weight_decay: f64,
    sigreg_projections: usize,
    sigreg_knots: usize,
    sigreg_weight: f64,
    #[serde(default)]
    patch_grounding_weight: f64,
    #[serde(default)]
    patch_grounding_mode: PatchGroundingMode,
    #[serde(default)]
    exact_grounding_weight: f64,
    #[serde(default)]
    init_seed: Option<u64>,
    #[serde(default)]
    experiment: Option<ResolvedExperiment>,
    event_weight: f64,
    q_weight: f64,
    rollout_weight: f64,
    q_mse_threshold: f64,
    hidden_dim: usize,
    action_dim: usize,
    inner_steps: usize,
    outer_steps: usize,
    ptrm_rank_every: usize,
    randomize_depth: bool,
    #[serde(default)]
    steady_gpu: bool,
    #[serde(default)]
    world_core_v6: bool,
    #[serde(default)]
    init_context_from_v5: Option<PathBuf>,
    /// ADR 0005 §2: selects the stream schedule and whole-frame rendering.
    #[serde(default)]
    data_contract_v6: bool,
    #[serde(default)]
    supervise_last_outer_only: bool,
    phased_training: bool,
    #[serde(default = "default_stop_grad_event_y")]
    stop_grad_event_y: bool,
    #[serde(default)]
    residual_y_update: bool,
    #[serde(default)]
    warm_start_y: bool,
    #[serde(default)]
    sigreg_spatial: bool,
    #[serde(default = "default_sigreg_spatial_pool")]
    sigreg_spatial_pool: bool,
    #[serde(default)]
    sigreg_pre_rms_spatial: bool,
    sigreg_projector: bool,
    sigreg_projector_dim: usize,
    #[serde(default)]
    stop_grad_q_y: bool,
    #[serde(default)]
    q_quantile_targets: bool,
    #[serde(default)]
    train_z_noise: f64,
    #[serde(default)]
    shuffled_episodes: bool,
    baseline_d1: bool,
    prefix_weight: f64,
    reliability_weight: f64,
    bf16_conv: bool,
    #[serde(default)]
    bf16_recurrent_core: bool,
    sigreg_max_rows: usize,
    sigreg_target: SigregTarget,
    sigreg_temporal_window: usize,
    sigreg_global_mix: f64,
    #[serde(default)]
    world_core_v2: bool,
    #[serde(default)]
    spatial_action_field: bool,
    #[serde(default)]
    world_core_v3: bool,
    #[serde(default)]
    world_core_v4: bool,
    /// Foundation-v2 topology includes the operator-conditioning projection.
    #[serde(default)]
    operator_conditioning: bool,
    #[serde(default)]
    spatial_action_residual: bool,
    #[serde(default = "default_spatial_action_residual_scale")]
    spatial_action_residual_scale: f64,
    #[serde(default)]
    split_ce_weighting: SplitCeWeighting,
    #[serde(default)]
    split_ce_changed_budget: Option<f64>,
    #[serde(default)]
    promotion_metric: PromotionMetric,
    #[serde(default)]
    branch_learning: BranchLearningConfig,
    #[serde(default)]
    copy_bypass_gate: bool,
    #[serde(default)]
    copy_gate_bias_prior: Option<f64>,
    #[serde(default)]
    grid_scaled_action_impulse: bool,
    #[serde(default)]
    decode_composition: DecodeComposition,
    #[serde(default)]
    positional_value_readout: bool,
    /// The multi-treatment attribution waiver is provenance: a resume that
    /// flips it must fail the contract comparison.
    #[serde(default)]
    allow_multi_treatment_arm: bool,
    device: String,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_eps: f64,
    #[serde(default = "default_muon_momentum")]
    muon_momentum: f64,
    #[serde(default = "default_muon_rms_scale")]
    muon_rms_scale: f64,
}

impl From<&TrainConfig> for TrainingContract {
    fn from(cfg: &TrainConfig) -> Self {
        let adam = adam_params(cfg);
        Self {
            foundation_objective_revision: FOUNDATION_OBJECTIVE_REVISION,
            recipe: cfg.recipe,
            seed: cfg.seed,
            lessons: cfg.lessons.clone(),
            steps_per_lesson: cfg.steps_per_lesson,
            lesson_steps: resolved_lesson_steps(cfg),
            physical_batch: cfg.physical_batch,
            grad_accum: cfg.grad_accum,
            profile_update: cfg.profile_update,
            profile_updates: Some(cfg.profile_updates.clone()),
            pressure_updates: cfg.pressure_updates.clone(),
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            sigreg_projections: cfg.sigreg_projections,
            sigreg_knots: cfg.sigreg_knots,
            sigreg_weight: cfg.sigreg_weight,
            patch_grounding_weight: cfg.patch_grounding_weight,
            patch_grounding_mode: cfg.patch_grounding_mode,
            exact_grounding_weight: cfg.exact_grounding_weight,
            init_seed: cfg.init_seed,
            experiment: Some(
                cfg.resolved_experiment()
                    .expect("validated training config resolves an experiment"),
            ),
            event_weight: cfg.event_weight,
            q_weight: cfg.q_weight,
            rollout_weight: cfg.rollout_weight,
            q_mse_threshold: cfg.q_mse_threshold,
            hidden_dim: cfg.hidden_dim,
            action_dim: cfg.action_dim,
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
            ptrm_rank_every: cfg.ptrm_rank_every,
            randomize_depth: cfg.randomize_depth,
            steady_gpu: cfg.steady_gpu,
            world_core_v6: cfg.world_core_v6,
            init_context_from_v5: cfg.init_context_from_v5.clone(),
            data_contract_v6: cfg.data_contract_v6,
            supervise_last_outer_only: cfg.supervise_last_outer_only,
            phased_training: cfg.phased_training,
            stop_grad_event_y: cfg.stop_grad_event_y,
            residual_y_update: cfg.residual_y_update,
            warm_start_y: cfg.warm_start_y,
            sigreg_spatial: cfg.sigreg_spatial,
            sigreg_spatial_pool: cfg.sigreg_spatial_pool,
            sigreg_pre_rms_spatial: cfg.sigreg_pre_rms_spatial,
            sigreg_projector: cfg.sigreg_projector,
            sigreg_projector_dim: cfg.sigreg_projector_dim,
            stop_grad_q_y: cfg.stop_grad_q_y,
            q_quantile_targets: cfg.q_quantile_targets,
            train_z_noise: cfg.train_z_noise,
            shuffled_episodes: cfg.shuffled_episodes,
            baseline_d1: cfg.baseline_d1,
            prefix_weight: cfg.prefix_weight,
            reliability_weight: cfg.reliability_weight,
            bf16_conv: cfg.bf16_conv,
            bf16_recurrent_core: cfg.bf16_recurrent_core,
            sigreg_max_rows: cfg.sigreg_max_rows,
            sigreg_target: cfg.sigreg_target,
            sigreg_temporal_window: cfg.sigreg_temporal_window,
            sigreg_global_mix: cfg.sigreg_global_mix,
            world_core_v2: cfg.world_core_v2,
            world_core_v3: cfg.world_core_v3,
            world_core_v4: cfg.world_core_v4,
            operator_conditioning: cfg.recipe == TrainingRecipe::FoundationV2,
            spatial_action_field: cfg.spatial_action_field,
            spatial_action_residual: cfg.spatial_action_residual,
            spatial_action_residual_scale: cfg.spatial_action_residual_scale,
            split_ce_weighting: cfg.split_ce_weighting,
            split_ce_changed_budget: cfg.split_ce_changed_budget,
            promotion_metric: cfg.promotion_metric,
            branch_learning: cfg.branch_learning.clone(),
            copy_bypass_gate: cfg.copy_bypass_gate,
            copy_gate_bias_prior: cfg.copy_gate_bias_prior,
            grid_scaled_action_impulse: cfg.grid_scaled_action_impulse,
            decode_composition: cfg.decode_composition,
            positional_value_readout: cfg.positional_value_readout,
            allow_multi_treatment_arm: cfg.allow_multi_treatment_arm,
            device: cfg.device.clone(),
            adam_beta1: adam.beta1,
            adam_beta2: adam.beta2,
            adam_eps: adam.eps,
            muon_momentum: cfg.muon_momentum,
            muon_rms_scale: cfg.muon_rms_scale,
        }
    }
}

fn adam_params(cfg: &TrainConfig) -> ParamsAdamW {
    ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        beta2: 0.95,
        ..ParamsAdamW::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainerState {
    schema: String,
    contract: TrainingContract,
    global_step: u64,
    lesson_index: usize,
    step_in_lesson: usize,
    optimizer_step: usize,
    completed_lessons: Vec<LessonReport>,
    active_sums: LessonLossMeans,
    parameter_names: Vec<String>,
    #[serde(default = "default_training_population_hash")]
    training_population_hash: u64,
    #[serde(default)]
    training_content_hash: [u8; 32],
    #[serde(default)]
    training_population_rows: u64,
    #[serde(default)]
    batch_schedule_migrations: Vec<BatchScheduleMigration>,
    /// Global steps whose optimizer update was skipped because the gradient
    /// norm was non-finite (data/state-specific NaN in backward). Skipped
    /// steps advance `global_step` but not `optimizer_step`; the resume
    /// cursor check accounts for this list, keeping it auditable.
    #[serde(default)]
    nonfinite_skipped_updates: Vec<u64>,
    profile: ProfileState,
    #[serde(default)]
    gradient_pressure: Option<GradientPressureDiagnostics>,
    #[serde(default)]
    gradient_pressure_samples: Vec<GradientPressureDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    foundation_v2: Option<FoundationV2TrainerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FoundationV2TrainerState {
    total_steps: usize,
    ep_weight: f64,
    ep_gradient_budget: Vec<FoundationV2EpGradientSample>,
    gate_history: Vec<FoundationV2GateEvaluation>,
    best_changed_exact: Option<f64>,
    rollout_enabled: bool,
    loss_sums: FoundationV2LossMeans,
    loss_steps: u64,
    permanent_checkpoints: Vec<PathBuf>,
    #[serde(default)]
    event_label_census: EventLabelCensus,
    #[serde(default)]
    event_label_census_complete: bool,
    #[serde(default)]
    mechanism_history: Vec<FoundationV2MechanismSample>,
    /// One-based preregistered updates whose evidence bundles were published.
    #[serde(default)]
    profiles_published: Vec<u64>,
    /// Consecutive completed updates whose enabled realized rollout loss was
    /// exactly zero.
    #[serde(default)]
    rollout_zero_loss_consecutive_steps: u64,
    /// Frozen identity of the fixed gate population and gate policy. A resume
    /// must regenerate the exact same rows/masks under the same policy or
    /// fail closed before any optimizer update: best/collapse comparisons
    /// span runs and are meaningless across changed populations.
    #[serde(default)]
    gate_population_identity: Option<GatePopulationIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatePopulationIdentity {
    pub rows_sha256: String,
    pub masks_sha256: String,
    #[serde(default)]
    pub provenance_sha256: String,
    pub policy_schema: String,
}

/// Version of the in-trainer gate policy (thresholds, warmups, abort rule,
/// shuffle construction). Bump on any change so a resumed run cannot compare
/// new measurements against bests recorded under a different policy.
const FOUNDATION_V2_GATE_POLICY_SCHEMA: &str = "p2.gate_policy.v6";

fn foundation_v2_gate_population_identity(
    samples: &[TransitionSample],
    masks: &[ContentMask],
    provenance: &[V5SampleProvenance],
) -> Result<GatePopulationIdentity> {
    if samples.len() != masks.len() || samples.len() != provenance.len() {
        bail!("gate population identity rows, masks, and provenance must align");
    }
    let mut rows = Sha256::new();
    for sample in samples {
        let bytes = serde_json::to_vec(sample)?;
        rows.update((bytes.len() as u64).to_le_bytes());
        rows.update(&bytes);
    }
    let mut mask_digest = Sha256::new();
    for mask in masks {
        let bytes = serde_json::to_vec(mask)?;
        mask_digest.update((bytes.len() as u64).to_le_bytes());
        mask_digest.update(&bytes);
    }
    let mut provenance_digest = Sha256::new();
    for provenance in provenance {
        let bytes = serde_json::to_vec(provenance)?;
        provenance_digest.update((bytes.len() as u64).to_le_bytes());
        provenance_digest.update(&bytes);
    }
    Ok(GatePopulationIdentity {
        rows_sha256: format!("sha256:{:x}", rows.finalize()),
        masks_sha256: format!("sha256:{:x}", mask_digest.finalize()),
        provenance_sha256: format!("sha256:{:x}", provenance_digest.finalize()),
        policy_schema: FOUNDATION_V2_GATE_POLICY_SCHEMA.into(),
    })
}

fn reconcile_foundation_v2_gate_population_identity(
    foundation: &mut FoundationV2TrainerState,
    identity: GatePopulationIdentity,
) -> Result<()> {
    // Documented one-way migration (ADR 0003, gate policy v6): accept a
    // stored v4/v5 identity when the population digests are bit-identical —
    // only the abort accounting became noise-aware. The stored identity
    // adopts the new schema so the migration is durably visible.
    if let Some(stored) = &foundation.gate_population_identity {
        if stored.rows_sha256 == identity.rows_sha256
            && stored.masks_sha256 == identity.masks_sha256
            && stored.provenance_sha256 == identity.provenance_sha256
            && matches!(
                stored.policy_schema.as_str(),
                "p2.gate_policy.v4" | "p2.gate_policy.v5"
            )
            && identity.policy_schema == FOUNDATION_V2_GATE_POLICY_SCHEMA
        {
            tracing::warn!(
                "gate policy migration on resume: {} -> {}",
                stored.policy_schema,
                identity.policy_schema
            );
            foundation.gate_population_identity = Some(identity);
            return Ok(());
        }
    }
    match &foundation.gate_population_identity {
        Some(stored) if *stored != identity => bail!(
            "resumed gate population/policy identity mismatch: stored {:?} vs regenerated {:?}; best/collapse comparisons would span incomparable populations",
            stored,
            identity
        ),
        Some(_) => {}
        None if foundation.gate_history.is_empty() => {
            foundation.gate_population_identity = Some(identity)
        }
        None => bail!(
            "resumed checkpoint has foundation-v2 gate history but no gate population identity; refusing to compare promotions or collapse gates across an unknown population"
        ),
    }
    Ok(())
}

/// Initial state for the compatibility FNV fingerprint, which intentionally
/// excludes content masks and content origins.
fn default_training_population_hash() -> u64 {
    FNV1A64_OFFSET
}

/// Feed one ordered row into a cryptographic training-content digest.
///
/// Objective revision 3 binds the exact mask sidecar and content origin, so
/// published identities produced here differ from revision-2 runs even when
/// their serialized `TransitionSample` rows are otherwise identical. The
/// foundation-v2 loop now feeds this unchanged per-row byte layout into one
/// digest per delivered batch, then chains that digest with
/// `training_content_hash_append`. This intentionally changes its published
/// identity once from a chain of rows to a chain of batches while preserving
/// deterministic row order and mask/origin binding. Objective revision 4 also
/// binds the permuted episode operator consumed by the model, so published
/// identities differ when conditioning differs even if the rendered
/// transition happens to be unchanged.
fn training_content_digest_update(
    digest: &mut Sha256,
    sample: &TransitionSample,
    content_mask: &ContentMask,
) {
    digest.update(sample.seed.to_le_bytes());
    digest.update(sample.episode_id.to_le_bytes());
    digest.update(sample.transition_index.to_le_bytes());
    digest.update((sample.family.len() as u64).to_le_bytes());
    digest.update(sample.family.as_bytes());
    digest.update(sample.provenance.content_width.to_le_bytes());
    digest.update(sample.provenance.content_height.to_le_bytes());
    digest.update(sample.provenance.content_x.to_le_bytes());
    digest.update(sample.provenance.content_y.to_le_bytes());
    digest.update((content_mask.values.len() as u64).to_le_bytes());
    digest.update(&content_mask.values);
    digest.update((sample.provenance.source_kind.len() as u64).to_le_bytes());
    digest.update(sample.provenance.source_kind.as_bytes());
    digest.update((sample.provenance.trajectory_id.len() as u64).to_le_bytes());
    digest.update(sample.provenance.trajectory_id.as_bytes());
    match sample.provenance.operator {
        Some(operator) => digest.update([
            operator.family.conditioning_token() as u8,
            operator.agent_color,
            operator.primary_color,
            operator.secondary_color,
        ]),
        None => digest.update([0, 0, 0, 0]),
    }
    digest.update([sample.action.id]);
    digest.update([
        sample.action.x.unwrap_or(u8::MAX),
        sample.action.y.unwrap_or(u8::MAX),
    ]);
    for value in sample.goal_features.values {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([option_bool_byte(sample.noop)]);
    digest.update([option_bool_byte(sample.goal_satisfied)]);
    digest.update([option_bool_byte(sample.goal_failed)]);
    digest.update([option_bool_byte(sample.exhausted)]);
    digest.update((sample.current.pixels.len() as u64).to_le_bytes());
    digest.update(&sample.current.pixels);
    digest.update((sample.next.pixels.len() as u64).to_le_bytes());
    digest.update(&sample.next.pixels);
}

fn training_content_row_hash_append(
    previous: [u8; 32],
    sample: &TransitionSample,
    content_mask: &ContentMask,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(previous);
    training_content_digest_update(&mut digest, sample, content_mask);
    digest.finalize().into()
}

pub(crate) fn training_content_batch_digest<'a>(
    samples: impl ExactSizeIterator<Item = &'a TransitionSample>,
    content_masks: impl ExactSizeIterator<Item = &'a ContentMask>,
) -> Result<[u8; 32]> {
    if samples.len() != content_masks.len() {
        bail!("training population rows and content masks differ in length");
    }
    let mut digest = Sha256::new();
    for (sample, content_mask) in samples.zip(content_masks) {
        training_content_digest_update(&mut digest, sample, content_mask);
    }
    Ok(digest.finalize().into())
}

/// Fold one delivered foundation-v2 batch into the persisted content chain.
/// Only the previous chain and the worker-computed 32-byte batch digest are
/// hashed on the training thread. Resuming continues from the persisted
/// `previous` value, and ordered delivery makes the result independent of
/// worker completion timing.
pub(crate) fn training_content_hash_append(previous: [u8; 32], batch_digest: [u8; 32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(previous);
    digest.update(batch_digest);
    digest.finalize().into()
}

fn update_training_population<'a>(
    state: &mut TrainerState,
    samples: impl ExactSizeIterator<Item = &'a TransitionSample>,
    content_masks: impl ExactSizeIterator<Item = &'a ContentMask>,
    content_batch_digest: Option<[u8; 32]>,
) -> Result<()> {
    if samples.len() != content_masks.len() {
        bail!("training population rows and content masks differ in length");
    }

    fn bytes(hash: &mut u64, value: &[u8]) {
        for byte in value {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV1A64_PRIME);
        }
    }

    if let Some(batch_digest) = content_batch_digest {
        state.training_content_hash =
            training_content_hash_append(state.training_content_hash, batch_digest);
    }
    for (sample, content_mask) in samples.zip(content_masks) {
        if content_batch_digest.is_none() {
            state.training_content_hash =
                training_content_row_hash_append(state.training_content_hash, sample, content_mask);
        }
        bytes(
            &mut state.training_population_hash,
            &sample.seed.to_le_bytes(),
        );
        bytes(
            &mut state.training_population_hash,
            &sample.episode_id.to_le_bytes(),
        );
        bytes(
            &mut state.training_population_hash,
            &sample.transition_index.to_le_bytes(),
        );
        bytes(
            &mut state.training_population_hash,
            sample.family.as_bytes(),
        );
        bytes(
            &mut state.training_population_hash,
            &[0xff, sample.action.id],
        );
        bytes(
            &mut state.training_population_hash,
            &[
                sample.action.x.unwrap_or(u8::MAX),
                sample.action.y.unwrap_or(u8::MAX),
            ],
        );
        state.training_population_rows += 1;
    }
    Ok(())
}

fn training_content_mask_from_provenance(sample: &TransitionSample) -> Result<ContentMask> {
    ContentMask::from_rect(ContentRect {
        x: u8::try_from(sample.provenance.content_x).context("content x does not fit u8")?,
        y: u8::try_from(sample.provenance.content_y).context("content y does not fit u8")?,
        width: u8::try_from(sample.provenance.content_width)
            .context("content width does not fit u8")?,
        height: u8::try_from(sample.provenance.content_height)
            .context("content height does not fit u8")?,
    })
}

fn option_bool_byte(value: Option<bool>) -> u8 {
    match value {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchScheduleMigration {
    pub from_physical_batch: usize,
    pub from_grad_accum: usize,
    pub to_physical_batch: usize,
    pub to_grad_accum: usize,
    pub effective_batch: usize,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatestCheckpoint {
    schema: String,
    directory: String,
    global_step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointArtifactDigest {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointBundleManifest {
    schema: String,
    global_step: u64,
    artifacts: Vec<CheckpointArtifactDigest>,
    /// Hashes over named raw safetensor payloads. Comparing `world` between
    /// lesson-boundary bundles proves whether an observer-only stage moved it.
    parameter_groups: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FoundationV2AbortMarker {
    schema: String,
    global_step: u64,
    checkpoint: PathBuf,
}

/// Map a lesson name to a `generate_curriculum` kind. Never ARC recordings.
pub fn lesson_to_curriculum(lesson: &str) -> Result<&'static str> {
    match lesson {
        "factual_branches" => Ok("factual_branches"),
        "dynamics" => Ok("random_one_step"),
        "exploration" => Ok("exploration"),
        "sequential" | "q_calibration" | "events" => Ok("sequential"),
        "falsification" => Ok("p1c_falsification"),
        "retarget" => Ok("p1c_hard_retarget"),
        other => bail!("unknown lesson {other}"),
    }
}

pub fn resolve_device(spec: &str) -> Result<Device> {
    let spec = spec.trim();
    if spec == "cpu" {
        return Ok(Device::Cpu);
    }
    if spec == "cuda" {
        return Device::new_cuda(0).context("open cuda:0");
    }
    if let Some(rest) = spec.strip_prefix("cuda:") {
        let ordinal: usize = rest.parse().context("parse cuda ordinal")?;
        return Device::new_cuda(ordinal).with_context(|| format!("open cuda:{ordinal}"));
    }
    bail!("unsupported device {spec:?}; use cpu, cuda, or cuda:N");
}

fn stable_name_seed(master: u64, name: &str) -> u64 {
    let mut h = master ^ 0x9E37_79B9_7F4A_7C15;
    for &b in name.as_bytes() {
        h = h
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(b));
    }
    h
}

fn xavier_uniform_vec(shape: &[usize], seed: u64) -> Vec<f32> {
    let shape_obj = candle_core::Shape::from(shape.to_vec());
    let fan_in = FanInOut::FanIn.for_shape(&shape_obj).max(1);
    let fan_out = FanInOut::FanOut.for_shape(&shape_obj).max(1);
    let bound = (6.0f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = shape.iter().product::<usize>();
    (0..n).map(|_| rng.random_range(-bound..=bound)).collect()
}

/// Deterministic reinitialization: zero biases, Xavier-like weights from
/// `hash(name) ⊕ master_seed`. Works on CPU where `Device::set_seed` is unsupported.
pub fn reinit_varmap_deterministic(varmap: &VarMap, master_seed: u64) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    for name in names {
        let var = data
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?;
        let shape = var.shape().dims().to_vec();
        let n = var.elem_count();
        let seed = stable_name_seed(master_seed, &name);
        let is_bias = name.rsplit('.').next() == Some("bias") || name.ends_with("bias");
        let values = if is_bias {
            vec![0f32; n]
        } else {
            xavier_uniform_vec(&shape, seed)
        };
        let t = Tensor::from_vec(values, shape.as_slice(), var.device())?.to_dtype(var.dtype())?;
        var.set(&t)?;
    }
    Ok(())
}

/// V3 starts as the exact global-coordinate control: the shared spatial
/// residual projection is zero in every arm and learns only when its gate is
/// enabled. This removes an initialization-scale shock from the intervention.
fn zero_v3_spatial_residual(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut matched = 0usize;
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.starts_with("spatial_action_proj."))
    {
        let zero = Tensor::zeros(var.shape(), var.dtype(), var.device())?;
        var.set(&zero)
            .with_context(|| format!("zero V3 residual parameter {name}"))?;
        matched += 1;
    }
    if matched == 0 {
        bail!("V3 residual initialization found no spatial_action_proj parameters");
    }
    Ok(())
}

/// Load an exact model checkpoint after validating every name, shape, and dtype.
pub fn load_varmap_exact(varmap: &VarMap, path: &Path) -> Result<()> {
    let device = varmap
        .all_vars()
        .first()
        .map(|v| v.device().clone())
        .ok_or_else(|| anyhow::anyhow!("empty varmap"))?;
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    let expected: Vec<(String, Var)> = {
        let data = varmap.data().lock().unwrap();
        let mut vars: Vec<_> = data
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        vars
    };
    let expected_names: Vec<_> = expected.iter().map(|(name, _)| name.clone()).collect();
    let mut checkpoint_names: Vec<_> = mmap.tensors().into_iter().map(|(name, _)| name).collect();
    checkpoint_names.sort();
    if checkpoint_names != expected_names {
        let missing: Vec<_> = expected_names
            .iter()
            .filter(|name| !checkpoint_names.contains(name))
            .cloned()
            .collect();
        let extra: Vec<_> = checkpoint_names
            .iter()
            .filter(|name| !expected_names.contains(name))
            .cloned()
            .collect();
        bail!("model checkpoint tensor names mismatch: missing={missing:?} extra={extra:?}");
    }

    let mut loaded = Vec::with_capacity(expected.len());
    for (name, var) in expected {
        let tensor = mmap
            .load(&name, &device)
            .with_context(|| format!("load model tensor {name}"))?;
        if tensor.dims() != var.shape().dims() {
            bail!(
                "model checkpoint shape mismatch for {name}: checkpoint={:?} model={:?}",
                tensor.dims(),
                var.shape().dims()
            );
        }
        if tensor.dtype() != var.dtype() {
            bail!(
                "model checkpoint dtype mismatch for {name}: checkpoint={:?} model={:?}",
                tensor.dtype(),
                var.dtype()
            );
        }
        loaded.push((var, tensor));
    }
    for (var, tensor) in loaded {
        var.set(&tensor)?;
    }
    Ok(())
}

/// ADR 0005 §3.4: load a v5 checkpoint into a `world_core_v6` varmap. Every
/// checkpoint tensor must match a model tensor by name/shape/dtype and the
/// only tensors allowed to be absent from the checkpoint are the context
/// channel's (`context_*`), which keep their fresh initialization; context
/// FiLM is then zeroed so the loaded model computes exactly the v5 function.
/// Accepts a bundle directory or a `model.safetensors` path.
pub fn load_varmap_warm_start_context(varmap: &VarMap, source: &Path) -> Result<()> {
    let path = if source.is_dir() {
        source.join("model.safetensors")
    } else {
        source.to_path_buf()
    };
    let device = varmap
        .all_vars()
        .first()
        .map(|v| v.device().clone())
        .ok_or_else(|| anyhow::anyhow!("empty varmap"))?;
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path)? };
    let expected: BTreeMap<String, Var> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    let checkpoint_names = mmap
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>();
    let extra: Vec<_> = checkpoint_names
        .iter()
        .filter(|name| !expected.contains_key(*name))
        .cloned()
        .collect();
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !checkpoint_names.contains(*name))
        .cloned()
        .collect();
    let non_context_missing: Vec<_> = missing
        .iter()
        .filter(|name| !name.starts_with(CONTEXT_PARAMETER_PREFIX))
        .cloned()
        .collect();
    if !extra.is_empty() || !non_context_missing.is_empty() || missing.is_empty() {
        bail!(
            "v5 warm start requires a checkpoint missing exactly the context parameters: \
             missing={missing:?} extra={extra:?}"
        );
    }
    let mut loaded = Vec::with_capacity(checkpoint_names.len());
    for name in &checkpoint_names {
        let var = &expected[name];
        let tensor = mmap
            .load(name, &device)
            .with_context(|| format!("load model tensor {name}"))?;
        if tensor.dims() != var.shape().dims() || tensor.dtype() != var.dtype() {
            bail!(
                "model checkpoint shape/dtype mismatch for {name}: checkpoint={:?}/{:?} model={:?}/{:?}",
                tensor.dims(),
                tensor.dtype(),
                var.shape().dims(),
                var.dtype()
            );
        }
        loaded.push((var.clone(), tensor));
    }
    for (var, tensor) in loaded {
        var.set(&tensor)?;
    }
    zero_context_film_projections(varmap)
}

pub fn parameter_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.elem_count()).sum()
}

/// Palette indices `B×1×64×64` on device.
pub fn frames_to_indices(frames: &[ArcFrame], device: &Device) -> Result<Tensor> {
    let b = frames.len();
    if b == 0 {
        bail!("frames_to_indices requires at least one frame");
    }
    let pixels = FRAME_SIDE * FRAME_SIDE;
    let mut indices = vec![0u8; b * pixels];
    indices
        .par_chunks_mut(pixels)
        .zip(frames.par_iter())
        .try_for_each(|(slot, frame)| -> Result<()> {
            ensure_fixed_frame(frame)?;
            if let Some(&pix) = frame.pixels.iter().find(|&&p| p as usize >= PALETTE_SIZE) {
                bail!("palette value {pix} out of 0..{PALETTE_SIZE}");
            }
            slot.copy_from_slice(&frame.pixels);
            Ok(())
        })?;
    Tensor::from_vec(indices, (b, 1, FRAME_SIDE, FRAME_SIDE), device).map_err(Into::into)
}

pub(crate) trait TransitionSampleView: Sync {
    fn transition_sample(&self) -> &TransitionSample;
}

impl TransitionSampleView for TransitionSample {
    fn transition_sample(&self) -> &TransitionSample {
        self
    }
}

impl TransitionSampleView for V5Sample {
    fn transition_sample(&self) -> &TransitionSample {
        self.transition()
    }
}

fn sample_frames_to_indices<T: TransitionSampleView>(
    samples: &[T],
    next: bool,
    empty_status: bool,
    device: &Device,
) -> Result<Tensor> {
    if samples.is_empty() {
        bail!("sample frame batch requires at least one transition");
    }
    let pixels = FRAME_SIDE * FRAME_SIDE;
    let mut indices = vec![0u8; samples.len() * pixels];
    indices
        .par_chunks_mut(pixels)
        .zip(samples.par_iter())
        .try_for_each(|(slot, sample)| -> Result<()> {
            let sample = sample.transition_sample();
            let frame = if next { &sample.next } else { &sample.current };
            ensure_fixed_frame(frame)?;
            if let Some(&pix) = frame.pixels.iter().find(|&&p| p as usize >= PALETTE_SIZE) {
                bail!("palette value {pix} out of 0..{PALETTE_SIZE}");
            }
            slot.copy_from_slice(&frame.pixels);
            if empty_status {
                slot[(FRAME_SIDE - 1) * FRAME_SIDE..].fill(0);
            }
            Ok(())
        })?;
    Tensor::from_vec(indices, (samples.len(), 1, FRAME_SIDE, FRAME_SIDE), device)
        .map_err(Into::into)
}

fn sample_frame_pair_to_indices<T: TransitionSampleView>(
    samples: &[T],
    empty_status: bool,
    device: &Device,
) -> (Result<Tensor>, Result<Tensor>) {
    // CUDA tensor construction shares one device stream. Building both frame
    // tensors from separate Rayon workers can race their host-to-device copies,
    // leaving otherwise validated palette indices corrupted by the time the
    // embedding kernel reads them. CPU construction is independent and keeps
    // the parallel path.
    if device.is_cuda() {
        (
            sample_frames_to_indices(samples, false, empty_status, device),
            sample_frames_to_indices(samples, true, empty_status, device),
        )
    } else {
        rayon::join(
            || sample_frames_to_indices(samples, false, empty_status, device),
            || sample_frames_to_indices(samples, true, empty_status, device),
        )
    }
}

fn ensure_fixed_frame(frame: &ArcFrame) -> Result<()> {
    if frame.width as usize != FRAME_SIDE || frame.height as usize != FRAME_SIDE {
        bail!(
            "expected {FRAME_SIDE}x{FRAME_SIDE} frame, got {}x{}",
            frame.width,
            frame.height
        );
    }
    if frame.pixels.len() != FRAME_SIDE * FRAME_SIDE {
        bail!("frame pixel length mismatch");
    }
    Ok(())
}

/// Event targets (`B×4`) and mask (`B×4`) from `Option<bool>` labels.
pub fn event_targets_and_mask(
    samples: &[TransitionSample],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    event_targets_and_mask_from_rows(samples, device)
}

fn event_targets_and_mask_from_rows<T: TransitionSampleView>(
    samples: &[T],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let b = samples.len();
    let mut targets = vec![0f32; b * DEFAULT_NUM_EVENTS];
    let mut mask = vec![0f32; b * DEFAULT_NUM_EVENTS];
    for (i, s) in samples.iter().enumerate() {
        let s = s.transition_sample();
        let row = i * DEFAULT_NUM_EVENTS;
        for (j, opt) in [s.noop, s.goal_satisfied, s.goal_failed, s.exhausted]
            .into_iter()
            .enumerate()
        {
            if let Some(v) = opt {
                targets[row + j] = if v { 1.0 } else { 0.0 };
                mask[row + j] = 1.0;
            }
        }
    }
    let targets = Tensor::from_vec(targets, (b, DEFAULT_NUM_EVENTS), device)?;
    let mask = Tensor::from_vec(mask, (b, DEFAULT_NUM_EVENTS), device)?;
    Ok((targets, mask))
}

pub struct BatchTensors {
    pub frames: Tensor,
    pub next_frames: Tensor,
    /// Encoder inputs with the synthetic status row staged as EMPTY.
    pub model_frames: Tensor,
    pub model_next_frames: Tensor,
    pub actions: Tensor,
    /// Normalized `(x,y)` for ACTION6, zeros for simple actions.
    pub action_coords: Tensor,
    /// Family one-hot plus three color one-hots. UNKNOWN rows keep only the
    /// family bit and use an all-zero (neutral) color triple.
    pub operator_conditioning: Tensor,
    pub goals: Tensor,
    pub event_targets: Tensor,
    pub event_mask: Tensor,
    pub factual: Option<FactualBatch>,
    /// ADR 0005 context windows; `None` when no row carries context.
    pub context: Option<ContextBatch>,
}

/// All CPU-only work needed for one foundation-v2 batch.
///
/// This is constructed by `MixedStreamBatchPrefetcher` workers and intentionally
/// contains only ordinary host vectors. Candle CUDA tensor construction remains
/// serialized on the training thread; constructing those tensors in workers has
/// previously raced host-to-device copies.
#[derive(Debug)]
pub struct PreparedFoundationV2BatchHost {
    batch_size: usize,
    frames: Vec<u8>,
    next_frames: Vec<u8>,
    model_frames: Vec<u8>,
    model_next_frames: Vec<u8>,
    actions: Vec<u32>,
    action_coords: Vec<f32>,
    operator_conditioning: Vec<f32>,
    goals: Vec<f32>,
    event_targets: Vec<f32>,
    event_mask: Vec<f32>,
    latent_content_mask: Vec<f32>,
    content_values: Vec<f32>,
    changed_values: Vec<f32>,
    unchanged_values: Vec<f32>,
    foreground_values: Vec<f32>,
    changed_weights: ChangedPixelWeights,
    foreground_count: usize,
    background_count: usize,
    foreground_weight: f64,
    context: Option<ContextBatchHost>,
    /// Supervised board rows: 63 for legacy rows, 64 under ADR 0005 §1.1.
    gameplay_rows: usize,
}

impl PreparedFoundationV2BatchHost {
    pub fn context(&self) -> Option<&ContextBatchHost> {
        self.context.as_ref()
    }
}

fn validate_frame_pixels(frame: &ArcFrame) -> Result<()> {
    ensure_fixed_frame(frame)?;
    if let Some(&pixel) = frame
        .pixels
        .iter()
        .find(|&&pixel| pixel as usize >= PALETTE_SIZE)
    {
        bail!("palette value {pixel} out of 0..{PALETTE_SIZE}");
    }
    Ok(())
}

/// Expand a deterministic mixed batch without creating tensors. The vector
/// order exactly follows `MixedStreamBatch::samples`, so ordered prefetching
/// preserves both content digests and the training RNG stream.
pub(crate) fn prepare_foundation_v2_batch_host(
    mixed: &MixedStreamBatch,
) -> Result<PreparedFoundationV2BatchHost> {
    let samples = mixed.samples();
    if samples.is_empty() {
        bail!("empty batch");
    }
    let batch_size = samples.len();
    let frame_pixels = FRAME_SIDE * FRAME_SIDE;
    let whole_frame = mixed.contract_v6();
    let rows = gameplay_rows(whole_frame);
    let gameplay_pixels = rows * FRAME_SIDE;
    let mut background_colors = Vec::with_capacity(batch_size);
    let mut frames = Vec::with_capacity(batch_size * frame_pixels);
    let mut next_frames = Vec::with_capacity(batch_size * frame_pixels);
    let mut actions = Vec::with_capacity(batch_size);
    let mut action_coords = Vec::with_capacity(batch_size * 2);
    let mut operator_conditioning = vec![0.0; batch_size * OPERATOR_CONDITION_DIM];
    let mut goals = Vec::with_capacity(batch_size * GOAL_FEATURES_DIM);
    let mut event_targets = vec![0.0; batch_size * DEFAULT_NUM_EVENTS];
    let mut event_mask = vec![0.0; batch_size * DEFAULT_NUM_EVENTS];
    let mut current_pixels = Vec::with_capacity(batch_size * gameplay_pixels);
    let mut target_pixels = Vec::with_capacity(batch_size * gameplay_pixels);
    let mut content_mask_u8 = Vec::with_capacity(batch_size * gameplay_pixels);
    let mut latent_content_mask = Vec::with_capacity(batch_size * (FRAME_SIDE / PATCH_SIZE).pow(2));

    for (row, sample) in samples.iter().enumerate() {
        let transition = sample.transition();
        transition.provenance.validate()?;
        validate_frame_pixels(&transition.current)?;
        validate_frame_pixels(&transition.next)?;
        if sample.content_mask.values.len() != frame_pixels {
            bail!("content mask is not fixed 64x64");
        }
        frames.extend_from_slice(&transition.current.pixels);
        next_frames.extend_from_slice(&transition.next.pixels);
        current_pixels.extend_from_slice(&transition.current.pixels[..gameplay_pixels]);
        target_pixels.extend_from_slice(&transition.next.pixels[..gameplay_pixels]);
        content_mask_u8.extend_from_slice(&sample.content_mask.values[..gameplay_pixels]);
        background_colors.push(transition.provenance.background_color);

        let id = u32::from(transition.action.id);
        if id >= ACTION_VOCAB as u32 {
            bail!("action id {id} out of official range 0..{ACTION_VOCAB}");
        }
        match (id, transition.action.x, transition.action.y) {
            (6, Some(x), Some(y)) => {
                actions.push(id);
                action_coords.extend([f32::from(x) / 63.0, f32::from(y) / 63.0]);
            }
            (0..=5, None, None) | (7, None, None) => {
                actions.push(id);
                action_coords.extend([0.0, 0.0]);
            }
            (6, _, _) => bail!("ACTION6 requires a complete coordinate pair"),
            (_, Some(_), _) | (_, _, Some(_)) => bail!("coordinates are only valid for ACTION6"),
            _ => bail!("invalid action conditioning"),
        }

        let base = row * OPERATOR_CONDITION_DIM;
        // v6 rows always condition as UNKNOWN (ADR 0005 §1.4).
        let operator = sample
            .provenance
            .conditioning_operator()
            .filter(|operator| {
                matches!(
                    operator.family,
                    OperatorFamily::Teleport
                        | OperatorFamily::Toggle
                        | OperatorFamily::Paint
                        | OperatorFamily::PushLine
                )
            });
        let family_token = operator
            .map(|operator| operator.family.conditioning_token())
            .unwrap_or(OPERATOR_FAMILY_UNKNOWN);
        operator_conditioning[base + family_token] = 1.0;
        if let Some(operator) = operator {
            for (slot, color) in [
                operator.agent_color,
                operator.primary_color,
                operator.secondary_color,
            ]
            .into_iter()
            .enumerate()
            {
                operator_conditioning
                    [base + OPERATOR_FAMILY_VOCAB + slot * PALETTE_SIZE + color as usize] = 1.0;
            }
        }
        goals.extend_from_slice(&transition.goal_features.values);
        for (slot, label) in [
            transition.noop,
            transition.goal_satisfied,
            transition.goal_failed,
            transition.exhausted,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(label) = label {
                event_targets[row * DEFAULT_NUM_EVENTS + slot] = f32::from(label);
                event_mask[row * DEFAULT_NUM_EVENTS + slot] = 1.0;
            }
        }

        let latent_side = FRAME_SIDE / PATCH_SIZE;
        for latent_y in 0..latent_side {
            for latent_x in 0..latent_side {
                let occupied = (0..PATCH_SIZE).any(|dy| {
                    (0..PATCH_SIZE).any(|dx| {
                        sample.content_mask.values
                            [(latent_y * PATCH_SIZE + dy) * FRAME_SIDE + latent_x * PATCH_SIZE + dx]
                            != 0
                    })
                });
                latent_content_mask.push(f32::from(occupied));
            }
        }
    }
    let changed_weights =
        foundation_v2_loss_weights_from_masks(&current_pixels, &target_pixels, &content_mask_u8)?;
    let content_values = content_mask_u8
        .iter()
        .map(|&value| f32::from(value))
        .collect::<Vec<_>>();
    let changed_values = current_pixels
        .iter()
        .zip(&target_pixels)
        .zip(&content_mask_u8)
        .map(|((before, after), content)| f32::from(*content != 0 && before != after))
        .collect::<Vec<_>>();
    let unchanged_values = changed_values
        .iter()
        .zip(&content_mask_u8)
        .map(|(changed, content)| f32::from(*content != 0) - changed)
        .collect::<Vec<_>>();
    // Foreground is every content pixel that is not the row's rendered EMPTY
    // colour (ADR 0005 §1.2); legacy rows render EMPTY as index 0.
    let foreground_values = current_pixels
        .chunks_exact(gameplay_pixels)
        .zip(content_mask_u8.chunks_exact(gameplay_pixels))
        .zip(&background_colors)
        .flat_map(|((pixels, content), background)| {
            pixels
                .iter()
                .zip(content)
                .map(move |(pixel, content)| f32::from(*content != 0 && pixel != background))
        })
        .collect::<Vec<_>>();
    let foreground_count = foreground_values
        .iter()
        .filter(|&&value| value != 0.0)
        .count();
    let background_count = changed_weights.content_pixels - foreground_count;
    let foreground_weight = if foreground_count == 0 {
        64.0
    } else {
        ((background_count as f64) / foreground_count as f64).clamp(1.0, 64.0)
    };
    let mut model_frames = frames.clone();
    let mut model_next_frames = next_frames.clone();
    // Legacy rows carry a synthetic status strip on row 63 that the model must
    // not see; v6 rows have no status row (ADR 0005 §1.3), so row 63 stays.
    if !whole_frame {
        for pixels in model_frames.chunks_exact_mut(frame_pixels) {
            pixels[gameplay_pixels..].fill(0);
        }
        for pixels in model_next_frames.chunks_exact_mut(frame_pixels) {
            pixels[gameplay_pixels..].fill(0);
        }
    }
    Ok(PreparedFoundationV2BatchHost {
        batch_size,
        frames,
        next_frames,
        model_frames,
        model_next_frames,
        actions,
        action_coords,
        operator_conditioning,
        goals,
        event_targets,
        event_mask,
        latent_content_mask,
        content_values,
        changed_values,
        unchanged_values,
        foreground_values,
        changed_weights,
        foreground_count,
        background_count,
        foreground_weight,
        context: ContextBatchHost::from_rows(mixed.transitions())?,
        gameplay_rows: rows,
    })
}

fn batch_from_foundation_v2_host(
    host: &PreparedFoundationV2BatchHost,
    device: &Device,
) -> Result<BatchTensors> {
    let shape = (host.batch_size, 1, FRAME_SIDE, FRAME_SIDE);
    let context = host
        .context
        .as_ref()
        .map(|context| ContextBatch::from_host(context, device))
        .transpose()?;
    Ok(BatchTensors {
        context,
        frames: Tensor::from_vec(host.frames.clone(), shape, device)?,
        next_frames: Tensor::from_vec(host.next_frames.clone(), shape, device)?,
        model_frames: Tensor::from_vec(host.model_frames.clone(), shape, device)?,
        model_next_frames: Tensor::from_vec(host.model_next_frames.clone(), shape, device)?,
        actions: Tensor::from_vec(host.actions.clone(), (host.batch_size,), device)?,
        action_coords: Tensor::from_vec(host.action_coords.clone(), (host.batch_size, 2), device)?,
        operator_conditioning: Tensor::from_vec(
            host.operator_conditioning.clone(),
            (host.batch_size, OPERATOR_CONDITION_DIM),
            device,
        )?,
        goals: Tensor::from_vec(
            host.goals.clone(),
            (host.batch_size, GOAL_FEATURES_DIM),
            device,
        )?,
        event_targets: Tensor::from_vec(
            host.event_targets.clone(),
            (host.batch_size, DEFAULT_NUM_EVENTS),
            device,
        )?,
        event_mask: Tensor::from_vec(
            host.event_mask.clone(),
            (host.batch_size, DEFAULT_NUM_EVENTS),
            device,
        )?,
        factual: None,
    })
}

/// Tensorize observable train-family operators. Held-out families and rows
/// without provenance receive UNKNOWN and a neutral all-zero color triple.
pub(crate) fn operator_conditioning_from_samples<T: TransitionSampleView>(
    samples: &[T],
    device: &Device,
) -> Result<Tensor> {
    let mut values = vec![0f32; samples.len() * OPERATOR_CONDITION_DIM];
    for (row, sample) in samples.iter().enumerate() {
        let sample = sample.transition_sample();
        let base = row * OPERATOR_CONDITION_DIM;
        let operator = sample.provenance.operator.filter(|operator| {
            matches!(
                operator.family,
                OperatorFamily::Teleport
                    | OperatorFamily::Toggle
                    | OperatorFamily::Paint
                    | OperatorFamily::PushLine
            )
        });
        let family_token = operator
            .map(|operator| operator.family.conditioning_token())
            .unwrap_or(OPERATOR_FAMILY_UNKNOWN);
        values[base + family_token] = 1.0;
        if let Some(operator) = operator {
            for (slot, color) in [
                operator.agent_color,
                operator.primary_color,
                operator.secondary_color,
            ]
            .into_iter()
            .enumerate()
            {
                values[base + OPERATOR_FAMILY_VOCAB + slot * PALETTE_SIZE + color as usize] = 1.0;
            }
        }
    }
    Tensor::from_vec(values, (samples.len(), OPERATOR_CONDITION_DIM), device).map_err(Into::into)
}

pub struct OrderedTraceTensors {
    pub frames: Tensor,
    pub next_frames: Tensor,
    pub actions: Tensor,
    pub action_coords: Tensor,
}

/// Non-overlapping, time-major rows selected from one deterministic batch.
///
/// Every window belongs to one `(seed, episode_id, family)` and has exactly
/// consecutive transition indices. `row_indices` is laid out `[time, window]`,
/// so it can be gathered directly into `T × B × C × H × W` encoder latents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedSigregWindows {
    pub window: usize,
    pub windows: usize,
    pub row_indices: Vec<usize>,
}

/// Select complete ordered SIGReg windows without reordering generated samples.
/// Broken episode runs and short tails are deliberately excluded rather than joined.
pub fn ordered_sigreg_windows(
    samples: &[TransitionSample],
    window: usize,
) -> Result<Option<OrderedSigregWindows>> {
    if window < 2 {
        bail!("ordered SIGReg window must be >= 2");
    }

    let mut complete = Vec::<Vec<usize>>::new();
    let mut run_start = 0;
    while run_start < samples.len() {
        let first = &samples[run_start];
        let mut run_end = run_start + 1;
        while run_end < samples.len() {
            let previous = &samples[run_end - 1];
            let next = &samples[run_end];
            let same_trace = next.seed == first.seed
                && next.episode_id == first.episode_id
                && next.family == first.family;
            let contiguous = next.transition_index == previous.transition_index.saturating_add(1);
            if !same_trace || !contiguous {
                break;
            }
            run_end += 1;
        }
        for chunk in (run_start..run_end)
            .collect::<Vec<_>>()
            .chunks_exact(window)
        {
            complete.push(chunk.to_vec());
        }
        run_start = run_end;
    }
    if complete.is_empty() {
        return Ok(None);
    }

    let mut row_indices = Vec::with_capacity(complete.len() * window);
    for time in 0..window {
        for trace in &complete {
            row_indices.push(trace[time]);
        }
    }
    Ok(Some(OrderedSigregWindows {
        window,
        windows: complete.len(),
        row_indices,
    }))
}

pub fn action_tensors_from_samples(
    samples: &[TransitionSample],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    action_tensors_from_rows(samples, device)
}

fn action_tensors_from_rows<T: TransitionSampleView>(
    samples: &[T],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let actions: Vec<u32> = samples
        .par_iter()
        .map(|sample| {
            let sample = sample.transition_sample();
            let id = sample.action.id as u32;
            if id >= ACTION_VOCAB as u32 {
                bail!("action id {id} out of official range 0..{ACTION_VOCAB}");
            }
            match (id, sample.action.x, sample.action.y) {
                (6, Some(_), Some(_)) | (0..=5, None, None) | (7, None, None) => Ok(id),
                (6, _, _) => bail!("ACTION6 requires a complete coordinate pair"),
                (_, Some(_), _) | (_, _, Some(_)) => {
                    bail!("coordinates are only valid for ACTION6")
                }
                _ => bail!("invalid action conditioning"),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let actions = Tensor::from_vec(actions, (samples.len(),), device)?;
    let coords: Vec<f32> = samples
        .par_iter()
        .map(|sample| {
            let sample = sample.transition_sample();
            match (sample.action.x, sample.action.y) {
                (Some(x), Some(y)) => Ok([f32::from(x) / 63.0, f32::from(y) / 63.0]),
                (None, None) => Ok([0.0, 0.0]),
                _ => Err(anyhow::anyhow!("action coordinate pair is incomplete")),
            }
        })
        .collect::<Result<Vec<[f32; 2]>>>()?
        .into_iter()
        .flatten()
        .collect();
    let action_coords = Tensor::from_vec(coords, (samples.len(), 2), device)?;
    Ok((actions, action_coords))
}

pub fn batch_from_samples(samples: &[TransitionSample], device: &Device) -> Result<BatchTensors> {
    if samples.is_empty() {
        bail!("empty batch");
    }
    for sample in samples {
        sample.provenance.validate()?;
    }
    let factual = samples
        .iter()
        .all(|sample| sample.family.starts_with("factual_"))
        .then(|| FactualBatch::from_rows(samples))
        .transpose()?;
    if let Some(factual) = factual {
        let mut batch = batch_from_rows(factual.rows(), device)?;
        batch.factual = Some(factual);
        return Ok(batch);
    }
    batch_from_rows(samples, device)
}

fn batch_from_rows<T: TransitionSampleView>(
    samples: &[T],
    device: &Device,
) -> Result<BatchTensors> {
    if samples.is_empty() {
        bail!("empty batch");
    }
    for sample in samples {
        sample.transition_sample().provenance.validate()?;
    }
    let (frames, next_frames) = sample_frame_pair_to_indices(samples, false, device);
    let frames = frames?;
    let next_frames = next_frames?;
    let (model_frames, model_next_frames) = sample_frame_pair_to_indices(samples, true, device);
    let model_frames = model_frames?;
    let model_next_frames = model_next_frames?;
    let (actions, action_coords) = action_tensors_from_rows(samples, device)?;
    let operator_conditioning = operator_conditioning_from_samples(samples, device)?;
    let goals: Vec<f32> = samples
        .iter()
        .flat_map(|sample| {
            sample
                .transition_sample()
                .goal_features
                .values
                .iter()
                .copied()
        })
        .collect();
    let goals = Tensor::from_vec(goals, (samples.len(), GOAL_FEATURES_DIM), device)?;
    let (event_targets, event_mask) = event_targets_and_mask_from_rows(samples, device)?;
    let context =
        ContextBatchHost::from_rows(samples.iter().map(TransitionSampleView::transition_sample))?
            .map(|host| ContextBatch::from_host(&host, device))
            .transpose()?;
    Ok(BatchTensors {
        context,
        frames,
        next_frames,
        model_frames,
        model_next_frames,
        actions,
        action_coords,
        operator_conditioning,
        goals,
        event_targets,
        event_mask,
        factual: None,
    })
}

pub fn ordered_trace_from_samples(
    samples: &[TransitionSample],
    device: &Device,
) -> Result<OrderedTraceTensors> {
    if samples.len() < 2 {
        bail!("ordered trace requires at least two transitions");
    }
    let (frames, next_frames) = sample_frame_pair_to_indices(samples, false, device);
    let (actions, action_coords) = action_tensors_from_samples(samples, device)?;
    Ok(OrderedTraceTensors {
        frames: frames?,
        next_frames: next_frames?,
        actions,
        action_coords,
    })
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Episode id for microbatch `micro` of optimizer step `effective_step`.
pub fn scheduled_episode_start(
    seed: u64,
    effective_step: u64,
    micro: usize,
    grad_accum: usize,
    shuffled: bool,
) -> u64 {
    if !shuffled {
        return effective_step
            .wrapping_mul(grad_accum as u64)
            .wrapping_add(micro as u64);
    }
    let slot = effective_step
        .wrapping_mul(grad_accum as u64)
        .wrapping_add(micro as u64);
    splitmix64(seed ^ 0x5EED_E001 ^ slot)
}

fn legacy_sigreg_projection_seed(
    seed: u64,
    effective_step: u64,
    grad_accum: usize,
    micro: usize,
) -> u64 {
    seed.wrapping_add(effective_step.wrapping_mul(grad_accum as u64))
        .wrapping_add(micro as u64)
}

pub(crate) fn collect_batch_uncached(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
    batch: usize,
    split: Split,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::with_capacity(batch);
    let mut ep = start_episode;
    let limit = start_episode.saturating_add(50_000);
    let threads = rayon::current_num_threads().max(1) as u64;
    let mut wave = threads;

    while out.len() < batch {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            bail!("batch collection cancelled");
        }
        if ep > limit {
            bail!("failed to collect batch={batch} from curriculum {curriculum}");
        }
        let generated: Vec<Vec<TransitionSample>> = (0..wave)
            .into_par_iter()
            .map(|offset| generate_curriculum(curriculum, seed, ep.wrapping_add(offset), split))
            .collect::<Result<_>>()?;
        let produced: usize = generated.iter().map(Vec::len).sum();
        for sample in generated.into_iter().flatten() {
            out.push(sample);
            if out.len() == batch {
                break;
            }
        }
        ep = ep.wrapping_add(wave);
        if produced > 0 {
            let per_episode = produced as f64 / wave as f64;
            let need = batch.saturating_sub(out.len());
            wave = ((need as f64 / per_episode).ceil() as u64)
                .max(1)
                .div_ceil(threads)
                * threads;
        }
    }
    Ok(out)
}

/// Sliding window of generated episodes, reused across steps.
///
/// Step `N` draws episodes `[N, N+k)` and step `N+1` draws `[N+1, N+k+1)`, so consecutive
/// batches differ by a single episode — at `physical_batch=1024` that is a 99%+ overlap
/// which the trainer previously regenerated from scratch every step. Episodes are a pure
/// function of `(curriculum, seed, episode, split)`, so holding them is exact memoization:
/// batch contents are identical, only the cost changes.
#[derive(Default)]
struct EpisodeCache {
    key: Option<(String, u64, Split)>,
    /// Episode id of `episodes.front()`.
    first_episode: u64,
    episodes: std::collections::VecDeque<Vec<TransitionSample>>,
}

/// Safety cap: sliding window should stay near `physical_batch` episodes; this blocks
/// runaway growth if the cursor stalls or jumps backward within the same key.
const EPISODE_CACHE_MAX_EPISODES: usize = 512;

impl EpisodeCache {
    fn key_matches(&self, curriculum: &str, seed: u64, split: Split) -> bool {
        match &self.key {
            Some((c, s, sp)) => c == curriculum && *s == seed && *sp == split,
            None => false,
        }
    }

    fn trim_excess(&mut self) {
        while self.episodes.len() > EPISODE_CACHE_MAX_EPISODES {
            self.episodes.pop_front();
            self.first_episode += 1;
        }
    }

    /// Same contract as [`collect_batch`], served from the window where possible.
    fn collect(
        &mut self,
        curriculum: &str,
        seed: u64,
        start_episode: u64,
        batch: usize,
        split: Split,
    ) -> Result<Vec<TransitionSample>> {
        self.reset_if_stale(curriculum, seed, split);

        // Release episodes the trainer has advanced past.
        while self.first_episode < start_episode && !self.episodes.is_empty() {
            self.episodes.pop_front();
            self.first_episode += 1;
        }
        if self.episodes.is_empty() {
            self.first_episode = start_episode;
        }

        let limit = start_episode.saturating_add(50_000);
        let threads = rayon::current_num_threads().max(1) as u64;
        loop {
            let have: usize = self.episodes.iter().map(Vec::len).sum();
            if have >= batch {
                break;
            }
            let next = self.first_episode + self.episodes.len() as u64;
            if next > limit {
                bail!("failed to collect batch={batch} from curriculum {curriculum}");
            }
            // Episodes have variable yield, so size the wave from what the window
            // already produced, rounded up to a whole wave so no worker idles. On a
            // cold cache there is nothing to extrapolate from, so run one pool-sized
            // probe rather than assuming a yield of 1 and over-generating ~8x.
            let wave = if self.episodes.is_empty() {
                threads
            } else {
                let per_episode = (have as f64 / self.episodes.len() as f64).max(1.0);
                (((batch - have) as f64 / per_episode).ceil() as u64)
                    .max(1)
                    .div_ceil(threads)
                    * threads
            };
            let generated: Vec<Vec<TransitionSample>> = (0..wave)
                .into_par_iter()
                .map(|offset| {
                    generate_curriculum(curriculum, seed, next.wrapping_add(offset), split)
                })
                .collect::<Result<_>>()?;
            self.episodes.extend(generated);
            self.trim_excess();
        }

        let mut out = Vec::with_capacity(batch);
        'fill: for episode in &self.episodes {
            for sample in episode {
                out.push(sample.clone());
                if out.len() == batch {
                    break 'fill;
                }
            }
        }
        Ok(out)
    }

    /// First cached episode at or after `start_episode` with at least two samples,
    /// matching [`collect_rollout_trace`]'s search. Returns the next id to probe when
    /// the window holds no qualifying episode.
    fn rollout_trace(
        &self,
        curriculum: &str,
        seed: u64,
        start_episode: u64,
        split: Split,
    ) -> std::result::Result<Vec<TransitionSample>, u64> {
        if !self.key_matches(curriculum, seed, split) || start_episode < self.first_episode {
            return Err(start_episode);
        }
        let skip = (start_episode - self.first_episode) as usize;
        for episode in self.episodes.iter().skip(skip) {
            if episode.len() >= 2 {
                return Ok(episode.clone());
            }
        }
        Err(self.first_episode + self.episodes.len() as u64)
    }

    fn reset_if_stale(&mut self, curriculum: &str, seed: u64, split: Split) {
        if !self.key_matches(curriculum, seed, split) {
            self.key = Some((curriculum.to_string(), seed, split));
            self.episodes.clear();
            self.first_episode = 0;
        }
    }
}

/// Uncached reference collector: the batch step `start_episode` must receive.
///
/// The trainer serves batches from [`EpisodeCache`]; this exists so the tests can assert
/// the cache is exact memoization rather than resampling. Episodes are generated in
/// parallel waves and concatenated in episode order, so the result is byte-for-byte
/// `concat(gen(start), gen(start+1), …)[..batch]`.
#[cfg(test)]
fn collect_batch(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
    batch: usize,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    collect_batch_uncached(curriculum, seed, start_episode, batch, split, None)
}

/// Elementwise BCE-with-logits in the saturation-safe form
/// `max(x, 0) - x * t + log(1 + exp(-|x|))`.
///
/// The naive `t * log(sigmoid(x)) + (1 - t) * log(1 - sigmoid(x))` form (which
/// `candle_nn::loss::binary_cross_entropy_with_logit` uses) produces NaN once a
/// logit saturates sigmoid in f32: `sigmoid(x)` rounds to exactly `1.0` for
/// `x > ~16.6` and to `0.0` for `x < ~-104`, so one `log` returns `-inf` while
/// its coefficient is `0`, and `0 * -inf` is NaN. This form never evaluates a
/// log at 0 — `exp(-|x|)` is in `(0, 1]`, so the log argument is in `(1, 2]`.
fn bce_with_logits_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let hinge = logits.relu()?;
    let xt = logits.broadcast_mul(targets)?;
    let softplus = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    hinge.sub(&xt)?.add(&softplus).map_err(Into::into)
}

fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    bce_with_logits_elem(logits, targets)?
        .mean_all()
        .map_err(Into::into)
}

/// Copy-gate BCE over the same complete 63x64 gameplay frame used by composed
/// exactness and padding-false-edit metrics. `changed` is content-masked, so
/// every PAD pixel is an unchanged target with unit weight.
fn foundation_v2_copy_gate_loss(
    gate_logits: &Tensor,
    changed: &Tensor,
    changed_weight: f64,
) -> Result<Tensor> {
    let gate_weights = changed.affine(changed_weight - 1.0, 1.0)?;
    bce_with_logits_elem(gate_logits, changed)?
        .mul(&gate_weights)?
        .mean_all()
        .map_err(Into::into)
}

/// Whether this optimizer step should include PTRM ranking loss.
pub fn ptrm_rank_this_step(
    lesson: &str,
    global_step: u64,
    every: usize,
    baseline_d1: bool,
) -> bool {
    if baseline_d1 {
        return false;
    }
    matches!(
        lesson,
        "sequential" | "q_calibration" | "falsification" | "retarget"
    ) && every > 0
        && global_step.is_multiple_of(every as u64)
}

pub fn ptrm_rank_k_for_lesson(lesson: &str) -> usize {
    if lesson == "falsification" {
        PTRM_RANK_K_FALSIFICATION
    } else {
        PTRM_RANK_K_DEFAULT
    }
}

/// Per-step recursion depth. When enabled, samples outer in `1..=max` and inner in `1..=max`.
pub fn sample_recursion_depth(cfg: &TrainConfig, global_step: u64) -> RecursionDepth {
    if cfg.baseline_d1 {
        return RecursionDepth {
            inner_steps: 1,
            outer_steps: 1,
        };
    }
    if cfg.steady_gpu || !cfg.randomize_depth {
        return RecursionDepth {
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
        };
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        cfg.seed
            .wrapping_add(global_step)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    let max_outer = cfg.outer_steps.max(1);
    // Outer depth 1 combined with `residual_y_update` makes the recursion the
    // identity map, which supervises the copy-forward solution directly. Draw
    // from 2 whenever the configured range allows it.
    let min_outer = max_outer.min(2);
    RecursionDepth {
        inner_steps: rng.random_range(1..=cfg.inner_steps),
        outer_steps: rng.random_range(min_outer..=max_outer),
    }
}

/// Effective auxiliary loss weights for the current lesson and step.
#[derive(Debug, Clone, Copy)]
pub struct LessonLossWeights {
    /// Weight on all world/representation objectives. Full V4 observer lessons
    /// set this to zero and route only detached representations to their heads.
    pub world: f64,
    pub sigreg: f64,
    pub event: f64,
    pub q: f64,
    pub rollout: f64,
    pub prefix: f64,
    pub reliability: f64,
    pub ptrm_rank: bool,
    pub ptrm_rank_k: usize,
}

pub fn lesson_loss_weights(
    lesson: &str,
    cfg: &TrainConfig,
    step_in_lesson: usize,
    global_step: u64,
) -> LessonLossWeights {
    let lesson_steps = steps_for_lesson(cfg, lesson);
    if !cfg.phased_training {
        return LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight,
            q: cfg.q_weight,
            rollout: cfg.rollout_weight,
            prefix: cfg.prefix_weight,
            reliability: cfg.reliability_weight,
            ptrm_rank: ptrm_rank_this_step(
                lesson,
                global_step,
                cfg.ptrm_rank_every,
                cfg.baseline_d1,
            ),
            ptrm_rank_k: ptrm_rank_k_for_lesson(lesson),
        };
    }
    let rollout_scale = rollout_weight_ramp(step_in_lesson, lesson_steps);
    let aux_warm = lesson_weight_ramp(step_in_lesson, lesson_steps, 0.0);
    let rank = ptrm_rank_this_step(lesson, global_step, cfg.ptrm_rank_every, cfg.baseline_d1);
    let rank_k = ptrm_rank_k_for_lesson(lesson);
    match lesson {
        "dynamics" => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: 0.0,
            q: 0.0,
            rollout: 0.0,
            prefix: 0.0,
            reliability: 0.0,
            ptrm_rank: false,
            ptrm_rank_k: rank_k,
        },
        "factual_branches" => LessonLossWeights {
            world: 1.0,
            sigreg: 0.0,
            event: 0.0,
            q: 0.0,
            rollout: 0.0,
            prefix: 0.0,
            reliability: 0.0,
            ptrm_rank: false,
            ptrm_rank_k: rank_k,
        },
        "exploration" => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: 0.0,
            q: 0.0,
            rollout: 0.0,
            prefix: 0.0,
            reliability: 0.0,
            ptrm_rank: false,
            ptrm_rank_k: rank_k,
        },
        "sequential" => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: 0.0,
            q: 0.0,
            rollout: cfg.rollout_weight * rollout_scale,
            prefix: cfg.prefix_weight * rollout_scale,
            reliability: 0.0,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        "q_calibration" => LessonLossWeights {
            world: if cfg.recipe == TrainingRecipe::FullV4 {
                0.0
            } else {
                1.0
            },
            sigreg: if cfg.recipe == TrainingRecipe::FullV4 {
                0.0
            } else {
                cfg.sigreg_weight
            },
            event: cfg.event_weight * aux_warm,
            q: cfg.q_weight * aux_warm,
            rollout: 0.0,
            prefix: 0.0,
            reliability: cfg.reliability_weight * aux_warm,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        "events" => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight * aux_warm,
            q: 0.0,
            rollout: 0.0,
            prefix: 0.0,
            reliability: 0.0,
            ptrm_rank: false,
            ptrm_rank_k: rank_k,
        },
        "falsification" => LessonLossWeights {
            world: if cfg.recipe == TrainingRecipe::FullV4 {
                0.0
            } else {
                1.0
            },
            sigreg: if cfg.recipe == TrainingRecipe::FullV4 {
                0.0
            } else {
                cfg.sigreg_weight
            },
            event: cfg.event_weight * aux_warm,
            q: cfg.q_weight * aux_warm,
            rollout: 0.0,
            prefix: cfg.prefix_weight * aux_warm,
            reliability: cfg.reliability_weight * aux_warm,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        "retarget" => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight,
            q: cfg.q_weight,
            rollout: cfg.rollout_weight * rollout_scale * RETARGET_ROLLOUT_SCALE,
            prefix: cfg.prefix_weight,
            reliability: cfg.reliability_weight,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        _ => LessonLossWeights {
            world: 1.0,
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight,
            q: cfg.q_weight,
            rollout: cfg.rollout_weight,
            prefix: cfg.prefix_weight,
            reliability: cfg.reliability_weight,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
    }
}

fn rollout_weight_ramp(step_in_lesson: usize, steps_per_lesson: usize) -> f64 {
    lesson_weight_ramp(step_in_lesson, steps_per_lesson, 0.25)
}

/// Linear ramp from `start` to `1.0` across a lesson (used for rollout and aux heads).
fn lesson_weight_ramp(step_in_lesson: usize, steps_per_lesson: usize, start: f64) -> f64 {
    let denom = steps_per_lesson.max(1) as f64;
    let t = (step_in_lesson as f64 / denom).clamp(0.0, 1.0);
    start + (1.0 - start) * t
}

/// Ramp open-loop horizon 2 → 4 → 8 → 16 within a lesson when phased training is on.
pub fn rollout_horizon_for_lesson(
    lesson: &str,
    step_in_lesson: usize,
    steps_per_lesson: usize,
) -> usize {
    let max_horizon = if lesson == "retarget" {
        RETARGET_MAX_ROLLOUT_HORIZON
    } else {
        DEFAULT_MAX_ROLLOUT_HORIZON
    };
    rollout_horizon(step_in_lesson, steps_per_lesson, max_horizon)
}

/// Ramp open-loop horizon 2 → 4 → 8 → 16 within a lesson when phased training is on.
pub fn rollout_horizon(
    step_in_lesson: usize,
    steps_per_lesson: usize,
    max_horizon: usize,
) -> usize {
    let max_horizon = max_horizon.max(2);
    if steps_per_lesson == 0 {
        return max_horizon;
    }
    let t = step_in_lesson as f64 / steps_per_lesson as f64;
    if t < 0.25 {
        2
    } else if t < 0.5 {
        4.min(max_horizon)
    } else if t < 0.75 {
        8.min(max_horizon)
    } else {
        max_horizon
    }
}

/// Scheduled sampling mix for AR-forcing rollout training (0 = pure model, 1 = always reset).
pub fn rollout_teacher_mix(lesson: &str, step_in_lesson: usize, steps_per_lesson: usize) -> f64 {
    let t = step_in_lesson as f64 / steps_per_lesson.max(1) as f64;
    let start = if lesson == "retarget" { 0.75 } else { 0.5 };
    let end = if lesson == "retarget" { 0.5 } else { 0.0 };
    start + (end - start) * t
}

/// Frobenius norm ||Cov(h) - I|| on centered batch encoder outputs (training monitor).
pub fn batch_latent_covariance_frobenius(z: &Tensor) -> Result<f64> {
    let flat = flatten_latent(z)?;
    let z = flat.to_dtype(DType::F32)?;
    let (batch, dim) = z.dims2()?;
    if batch < 2 || dim == 0 {
        return Ok(f64::NAN);
    }
    let flat = z.flatten_all()?.to_vec1::<f32>()?;
    let n = batch as f64;
    let mut means = vec![0f64; dim];
    for row in 0..batch {
        for col in 0..dim {
            means[col] += flat[row * dim + col] as f64;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }
    let mut err = 0f64;
    let denom = (n - 1.0).max(1.0);
    for i in 0..dim {
        for j in 0..dim {
            let mut cov = 0f64;
            for row in 0..batch {
                let vi = flat[row * dim + i] as f64 - means[i];
                let vj = flat[row * dim + j] as f64 - means[j];
                cov += vi * vj;
            }
            cov /= denom;
            let target = if i == j { 1.0 } else { 0.0 };
            let delta = cov - target;
            err += delta * delta;
        }
    }
    Ok(err.sqrt())
}

fn event_slot_weight_tensor(device: &Device) -> Result<Tensor> {
    Tensor::from_slice(&EVENT_SLOT_WEIGHTS, (1, DEFAULT_NUM_EVENTS), device).map_err(Into::into)
}

fn masked_bce_with_slot_weights(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
    slot_weights: Option<&Tensor>,
) -> Result<Tensor> {
    let effective_mask = match slot_weights {
        Some(w) => mask.broadcast_mul(w)?,
        None => mask.clone(),
    };
    let elem = bce_with_logits_elem(logits, targets)?;
    let weighted = (elem * &effective_mask)?;
    let divisor = effective_mask
        .sum_all()?
        .to_dtype(DType::F32)?
        .clamp(1.0f32, f32::INFINITY)?;
    weighted
        .sum_all()?
        .broadcast_div(&divisor)
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub fn ptrm_ranking_loss(
    model: &WorldModel,
    cur_z: &Tensor,
    batch: &BatchTensors,
    next_z: &Tensor,
    depth: RecursionDepth,
    k: usize,
    sigma: f64,
    seed: u64,
) -> Result<Tensor> {
    if k < 2 {
        return Tensor::zeros((), DType::F32, next_z.device()).map_err(Into::into);
    }
    let ptrm = model.ptrm_ranking_trajectories_from_encoded(
        cur_z,
        &batch.actions,
        &batch.action_coords,
        depth,
        PtrmConfig {
            k,
            sigma,
            seed: Some(seed),
        },
    )?;
    let mut q_rows = Vec::with_capacity(k);
    let mut y_rows = Vec::with_capacity(k);
    for traj in &ptrm {
        q_rows.push(traj.q_logit.squeeze(1)?);
        y_rows.push(traj.y.clone());
    }
    let q_logits = Tensor::stack(&q_rows, 1)?;
    let y_stack = Tensor::stack(&y_rows, 1)?;
    let target = next_z.unsqueeze(1)?.broadcast_as(y_stack.dims())?;
    let mse = y_stack.sub(&target)?.sqr()?.flatten_from(2)?.mean(2)?;
    let labels = mse.argmin(1)?.to_dtype(DType::U32)?;
    candle_nn::loss::cross_entropy(&q_logits, &labels).map_err(Into::into)
}

/// Geometrically balanced weight for horizon `h` in `{1,2,4,8,16}`.
pub fn prefix_horizon_weight(horizon: usize) -> f64 {
    let log_h = (horizon.max(1) as f64).log2();
    1.0 / log_h.max(1.0)
}

pub fn prefix_one_step_loss(
    model: &WorldModel,
    batch: &BatchTensors,
    cur_z: &Tensor,
    next_z: &Tensor,
) -> Result<Tensor> {
    let pred = model.prefix_predict(cur_z, &batch.actions, &batch.action_coords)?;
    candle_nn::loss::mse(&pred, next_z).map_err(Into::into)
}

pub fn prefix_multi_horizon_loss(
    model: &WorldModel,
    trace: &OrderedTraceTensors,
) -> Result<Tensor> {
    let trace_len = trace.frames.dim(0)?;
    let valid_horizons = PREFIX_HORIZONS
        .iter()
        .copied()
        .filter(|&horizon| trace_len > horizon)
        .collect::<Vec<_>>();
    let max_horizon = valid_horizons
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("prefix trace too short"))?;
    let mut total: Option<Tensor> = None;
    let mut weight_sum = 0f64;
    let target_frames = valid_horizons
        .iter()
        .map(|&horizon| trace.frames.narrow(0, horizon, 1).map_err(Into::into))
        .collect::<Result<Vec<Tensor>>>()?;
    let targets = model.encode_state(&Tensor::cat(&target_frames, 0)?)?;
    let mut target_index = 0usize;
    let mut z = model.encode_state(&trace.frames.narrow(0, 0, 1)?)?;
    for step in 0..max_horizon {
        z = model.prefix_predict(
            &z,
            &trace.actions.narrow(0, step, 1)?,
            &trace.action_coords.narrow(0, step, 1)?,
        )?;
        let horizon = step + 1;
        if !valid_horizons.contains(&horizon) {
            continue;
        }
        let target = targets.narrow(0, target_index, 1)?;
        target_index += 1;
        let w = prefix_horizon_weight(horizon);
        let robust = candle_nn::loss::huber(&z, &target, 1.0)?;
        let term = smooth_cap_nonnegative(&robust, ROLLOUT_STEP_LOSS_CAP)?.affine(w, 0.0)?;
        total = Some(match total {
            None => term,
            Some(acc) => acc.add(&term)?,
        });
        weight_sum += w;
    }
    total
        .expect("valid_horizons is non-empty")
        .affine(1.0 / weight_sum.max(1e-8), 0.0)
        .map_err(Into::into)
}

#[cfg(test)]
fn prefix_multi_horizon_loss_reference(
    model: &WorldModel,
    trace: &OrderedTraceTensors,
) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    let mut weight_sum = 0f64;
    for &horizon in &PREFIX_HORIZONS {
        if trace.frames.dim(0)? <= horizon {
            continue;
        }
        let mut z = model.encode_state(&trace.frames.narrow(0, 0, 1)?)?;
        for step in 0..horizon {
            z = model.prefix_predict(
                &z,
                &trace.actions.narrow(0, step, 1)?,
                &trace.action_coords.narrow(0, step, 1)?,
            )?;
        }
        let target = model.encode_state(&trace.frames.narrow(0, horizon, 1)?)?;
        let weight = prefix_horizon_weight(horizon);
        let robust = candle_nn::loss::huber(&z, &target, 1.0)?;
        let term = smooth_cap_nonnegative(&robust, ROLLOUT_STEP_LOSS_CAP)?.affine(weight, 0.0)?;
        total = Some(match total {
            None => term,
            Some(acc) => acc.add(&term)?,
        });
        weight_sum += weight;
    }
    total
        .ok_or_else(|| anyhow::anyhow!("prefix trace too short"))?
        .affine(1.0 / weight_sum.max(1e-8), 0.0)
        .map_err(Into::into)
}

#[cfg(test)]
fn ensure_finite(name: &str, t: &Tensor) -> Result<f32> {
    Ok(ensure_all_finite(&[(name, t)])?[0])
}

/// Read several loss scalars back in a single device round trip and check them all.
///
/// Each `to_scalar` drains the CUDA stream, so checking N scalars separately costs N
/// stalls. Stacking them first makes the whole step cost one. Semantics are unchanged:
/// every value is still checked every step, and the failing name is still reported.
fn ensure_all_finite(named: &[(&str, &Tensor)]) -> Result<Vec<f32>> {
    if named.is_empty() {
        return Ok(Vec::new());
    }
    let scalars = named
        .iter()
        .map(|(_, t)| t.to_dtype(DType::F32)?.reshape(1).map_err(Into::into))
        .collect::<Result<Vec<Tensor>>>()?;
    let values = Tensor::cat(&scalars, 0)?.to_vec1::<f32>()?;
    for ((name, _), value) in named.iter().zip(&values) {
        if !value.is_finite() {
            bail!("{name} is not finite: {value}");
        }
    }
    Ok(values)
}

fn gradient_l2_for_parameter_prefix(
    grads: &GradStore,
    varmap: &VarMap,
    prefix: &str,
) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    let mut sum_sq: Option<Tensor> = None;
    for (_, var) in data.iter().filter(|(name, _)| name.starts_with(prefix)) {
        if let Some(gradient) = grads.get(var.as_tensor()) {
            let squared = gradient.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                None => squared,
                Some(acc) => acc.add(&squared)?,
            });
        }
    }
    let norm = sum_sq
        .ok_or_else(|| anyhow::anyhow!("no gradients found for parameter prefix {prefix}"))?
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    if !norm.is_finite() {
        bail!("gradient norm for {prefix} is not finite: {norm}");
    }
    Ok(norm)
}

fn gradient_l2_all_parameters(grads: &GradStore, varmap: &VarMap) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    let mut sum_sq: Option<Tensor> = None;
    for var in data.values() {
        if let Some(gradient) = grads.get(var.as_tensor()) {
            let squared = gradient.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                None => squared,
                Some(acc) => acc.add(&squared)?,
            });
        }
    }
    let norm = sum_sq
        .ok_or_else(|| anyhow::anyhow!("no gradients found for model parameters"))?
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    if !norm.is_finite() {
        bail!("global gradient norm is not finite: {norm}");
    }
    Ok(norm)
}

fn gradient_cosine_for_parameter_prefix(
    left: &GradStore,
    right: &GradStore,
    varmap: &VarMap,
    prefix: &str,
) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    let mut dot: Option<Tensor> = None;
    let mut left_sq: Option<Tensor> = None;
    let mut right_sq: Option<Tensor> = None;
    for (name, var) in data.iter().filter(|(name, _)| name.starts_with(prefix)) {
        let tensor = var.as_tensor();
        let (Some(left), Some(right)) = (left.get(tensor), right.get(tensor)) else {
            continue;
        };
        let left = left.to_dtype(DType::F32)?;
        let right = right.to_dtype(DType::F32)?;
        let product = left.mul(&right)?.sum_all()?;
        let l2 = left.sqr()?.sum_all()?;
        let r2 = right.sqr()?.sum_all()?;
        dot = Some(match dot {
            None => product,
            Some(value) => value.add(&product)?,
        });
        left_sq = Some(match left_sq {
            None => l2,
            Some(value) => value.add(&l2)?,
        });
        right_sq = Some(match right_sq {
            None => r2,
            Some(value) => value.add(&r2)?,
        });
        let _ = name;
    }
    let dot = dot
        .ok_or_else(|| anyhow::anyhow!("no paired gradients found for parameter prefix {prefix}"))?
        .to_scalar::<f32>()? as f64;
    let left_norm = left_sq
        .expect("paired gradients have left norm")
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    let right_norm = right_sq
        .expect("paired gradients have right norm")
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    if !(dot.is_finite() && left_norm.is_finite() && right_norm.is_finite()) {
        bail!("gradient cosine for {prefix} is not finite");
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }
    Ok((dot / (left_norm * right_norm)).clamp(-1.0, 1.0))
}

#[derive(Debug, Clone)]
pub struct LossBreakdown {
    pub total: Tensor,
    pub next_latent: Tensor,
    pub sigreg_raw: Tensor,
    pub sigreg_bounded: Tensor,
    pub patch_grounding: Tensor,
    pub grounding_changed_patches: usize,
    pub grounding_unchanged_patches: usize,
    pub event: Tensor,
    pub q: Tensor,
    pub q_surprise: Tensor,
    pub ptrm_rank: Tensor,
    pub prefix: Tensor,
    pub reliability: Tensor,
    pub branch_total: Tensor,
    pub outcome_pull: Tensor,
    pub outcome_push: Tensor,
    pub action_recovery: Tensor,
    pub coordinate_recovery: Tensor,
    pub changed_margin: Tensor,
    pub spatial_variance: Tensor,
    pub spatial_covariance: Tensor,
    pub pooled_variance: Tensor,
    pub pooled_covariance: Tensor,
    pub displacement_variance: Tensor,
    pub displacement_covariance: Tensor,
    pub branch_audit: BranchLearningAudit,
}

#[derive(Debug)]
struct CheckedTrainingLosses {
    total: f32,
    next_latent: f32,
    rollout: f32,
    sigreg_raw: f32,
    sigreg_bounded: f32,
    patch_grounding: f32,
    event: f32,
    q: f32,
    prefix: f32,
    reliability: f32,
    branch_total: f32,
    outcome_pull: f32,
    outcome_push: f32,
    action_recovery: f32,
    coordinate_recovery: f32,
    changed_margin: f32,
    spatial_variance: f32,
    spatial_covariance: f32,
    pooled_variance: f32,
    pooled_covariance: f32,
    displacement_variance: f32,
    displacement_covariance: f32,
}

fn training_loss_tensors(
    losses: &LossBreakdown,
    rollout: &Tensor,
    prefix_multi: &Tensor,
    total: &Tensor,
) -> [Tensor; 25] {
    [
        losses.next_latent.detach(),
        rollout.detach(),
        losses.sigreg_raw.detach(),
        losses.sigreg_bounded.detach(),
        losses.patch_grounding.detach(),
        losses.event.detach(),
        losses.q.detach(),
        losses.q_surprise.detach(),
        losses.ptrm_rank.detach(),
        losses.prefix.detach(),
        prefix_multi.detach(),
        losses.reliability.detach(),
        losses.branch_total.detach(),
        losses.outcome_pull.detach(),
        losses.outcome_push.detach(),
        losses.action_recovery.detach(),
        losses.coordinate_recovery.detach(),
        losses.changed_margin.detach(),
        losses.spatial_variance.detach(),
        losses.spatial_covariance.detach(),
        losses.pooled_variance.detach(),
        losses.pooled_covariance.detach(),
        losses.displacement_variance.detach(),
        losses.displacement_covariance.detach(),
        total.detach(),
    ]
}

fn checked_training_losses(tensors: &[[Tensor; 25]]) -> Result<Vec<CheckedTrainingLosses>> {
    const NAMES: [&str; 25] = [
        "next_latent",
        "rollout",
        "sigreg_raw",
        "sigreg_bounded",
        "patch_grounding",
        "event",
        "q",
        "q_surprise",
        "ptrm_rank",
        "prefix",
        "prefix_multi",
        "reliability",
        "branch_total",
        "outcome_pull",
        "outcome_push",
        "action_recovery",
        "coordinate_recovery",
        "changed_margin",
        "spatial_variance",
        "spatial_covariance",
        "pooled_variance",
        "pooled_covariance",
        "displacement_variance",
        "displacement_covariance",
        "total",
    ];
    let named = tensors
        .iter()
        .flat_map(|micro| NAMES.iter().copied().zip(micro))
        .collect::<Vec<_>>();
    let values = ensure_all_finite(&named)?;
    Ok(values
        .chunks_exact(25)
        .map(|values| CheckedTrainingLosses {
            total: values[24],
            next_latent: values[0],
            rollout: values[1],
            sigreg_raw: values[2],
            sigreg_bounded: values[3],
            patch_grounding: values[4],
            event: values[5],
            q: values[6],
            prefix: values[9],
            reliability: values[11],
            branch_total: values[12],
            outcome_pull: values[13],
            outcome_push: values[14],
            action_recovery: values[15],
            coordinate_recovery: values[16],
            changed_margin: values[17],
            spatial_variance: values[18],
            spatial_covariance: values[19],
            pooled_variance: values[20],
            pooled_covariance: values[21],
            displacement_variance: values[22],
            displacement_covariance: values[23],
        })
        .collect())
}

/// Randomly subsample the population axis to cap activation memory.
/// Rank-3 `T×B×D` populations retain every temporal position and sample the
/// same `B` indices at each position, preserving the estimator's semantics.
pub fn subsample_sigreg_rows(stack: &Tensor, max_rows: usize, seed: u64) -> Result<Tensor> {
    let axis = match stack.rank() {
        2 => 0,
        3 => 1,
        rank => bail!("SIGReg population must be rank 2 or 3, got rank {rank}"),
    };
    let n = stack.dim(axis)?;
    if max_rows == 0 || n <= max_rows {
        return Ok(stack.clone());
    }
    use rand::seq::SliceRandom;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.partial_shuffle(&mut rng, max_rows);
    indices.truncate(max_rows);
    let idx = Tensor::from_vec(indices, (max_rows,), stack.device())?;
    stack.index_select(&idx, axis).map_err(Into::into)
}

/// Stack current/next latents for SIGReg (flattened or per spatial cell).
pub fn stack_latents_for_sigreg(
    cur_z: &Tensor,
    next_z: &Tensor,
    spatial: bool,
    pool: bool,
) -> Result<Tensor> {
    if spatial {
        let (cur, next) = if pool {
            (cur_z.avg_pool2d(2)?, next_z.avg_pool2d(2)?)
        } else {
            (cur_z.clone(), next_z.clone())
        };
        let (b, c, h, w) = cur.dims4()?;
        let cur = cur.permute((0, 2, 3, 1))?.reshape((b * h * w, c))?;
        let next = next.permute((0, 2, 3, 1))?.reshape((b * h * w, c))?;
        Tensor::cat(&[cur, next], 0).map_err(Into::into)
    } else {
        Tensor::stack(&[flatten_latent(cur_z)?, flatten_latent(next_z)?], 0).map_err(Into::into)
    }
}

fn smooth_cap_nonnegative(raw: &Tensor, cap: f64) -> Result<Tensor> {
    let nonnegative = raw.clamp(0.0, f64::INFINITY)?;
    nonnegative
        .affine(cap, 0.0)?
        .div(&nonnegative.affine(1.0, cap)?)
        .map_err(Into::into)
}

pub(crate) fn bounded_sigreg_loss(raw: &Tensor) -> Result<Tensor> {
    smooth_cap_nonnegative(raw, SIGREG_LOSS_CAP)
}

pub(crate) fn sigreg_loss_for_stack(
    stack: &Tensor,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<Tensor> {
    match cfg.sigreg_statistic {
        SigregStatistic::EppsPulley => {
            sigreg_epps_pulley_seeded(stack, cfg.sigreg_projections, cfg.sigreg_knots, seed)
        }
        SigregStatistic::Quantile => sigreg_quantile_seeded(stack, cfg.sigreg_projections, seed),
    }
}

/// Select the preregistered SIGReg representation without changing dynamics latents.
pub fn sigreg_stack_for_encoded_pair(
    cur_z: &Tensor,
    next_z: &Tensor,
    cur_raw: &Tensor,
    next_raw: &Tensor,
    projected: Option<&Tensor>,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<Tensor> {
    if cfg.sigreg_pre_rms_spatial {
        return subsample_sigreg_rows(
            &stack_latents_for_sigreg(cur_raw, next_raw, true, false)?,
            effective_sigreg_max_rows(cfg),
            seed.wrapping_add(0x5196_0001),
        );
    }
    match projected {
        Some(stack) => Ok(stack.clone()),
        None => subsample_sigreg_rows(
            &stack_latents_for_sigreg(cur_z, next_z, cfg.sigreg_spatial, cfg.sigreg_spatial_pool)?,
            effective_sigreg_max_rows(cfg),
            seed.wrapping_add(0x5196_0001),
        ),
    }
}

/// Raw and smoothly bounded SIGReg for an already encoded current/next pair.
pub fn sigreg_losses_for_encoded_pair(
    cur_z: &Tensor,
    next_z: &Tensor,
    cur_raw: &Tensor,
    next_raw: &Tensor,
    projected: Option<&Tensor>,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<(Tensor, Tensor)> {
    let stack =
        sigreg_stack_for_encoded_pair(cur_z, next_z, cur_raw, next_raw, projected, cfg, seed)?;
    let raw = sigreg_loss_for_stack(&stack, cfg, seed)?;
    let bounded = bounded_sigreg_loss(&raw)?;
    Ok((raw, bounded))
}

/// Resolve the persisted training representation before scoring SIGReg. Full
/// V4 regularizes its consumed canonical state; historical recipes retain
/// their recorded latent/projector geometry.
#[allow(clippy::too_many_arguments)]
pub fn model_sigreg_losses_for_encoded_pair(
    model: &WorldModel,
    cur_z: &Tensor,
    next_z: &Tensor,
    cur_raw: &Tensor,
    next_raw: &Tensor,
    projected: Option<&Tensor>,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<(Tensor, Tensor)> {
    if cfg.recipe == TrainingRecipe::FullV4 {
        let current = model.canonical_representation(cur_z)?;
        let next = model.canonical_representation(next_z)?;
        let stack = Tensor::stack(&[current, next], 0)?;
        let raw = sigreg_loss_for_stack(&stack, cfg, seed)?;
        return Ok((raw.clone(), raw));
    }
    sigreg_losses_for_encoded_pair(cur_z, next_z, cur_raw, next_raw, projected, cfg, seed)
}

/// Apply the existing post-RMS spatial SIGReg geometry to an ordered population.
/// The encoder is deliberately called before target selection, so marginal and
/// temporal-residual arms have identical frame batches and encoder call shapes.
pub fn sigreg_stack_for_ordered_windows(
    latents: &Tensor,
    windows: &OrderedSigregWindows,
    target: SigregTarget,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<Tensor> {
    let (rows, channels, height, width) = latents.dims4()?;
    if windows.row_indices.len() != windows.window.saturating_mul(windows.windows) {
        bail!("ordered SIGReg window metadata has an invalid row count");
    }
    if windows.row_indices.iter().any(|&row| row >= rows) {
        bail!("ordered SIGReg window row is outside encoded batch");
    }
    let indices = Tensor::from_vec(
        windows
            .row_indices
            .iter()
            .map(|&row| row as u32)
            .collect::<Vec<_>>(),
        (windows.row_indices.len(),),
        latents.device(),
    )?;
    let ordered = latents.index_select(&indices, 0)?.reshape((
        windows.window,
        windows.windows,
        channels,
        height,
        width,
    ))?;
    // Pool after gathering. This is the same post-RMS 2x2 control geometry; the
    // leading time/window axes are temporarily folded only for the pool operator.
    let pooled = if cfg.sigreg_spatial && cfg.sigreg_spatial_pool {
        let flat = ordered.reshape((windows.window * windows.windows, channels, height, width))?;
        let pooled = flat.avg_pool2d(2)?;
        let (_, _, pooled_height, pooled_width) = pooled.dims4()?;
        pooled.reshape((
            windows.window,
            windows.windows,
            channels,
            pooled_height,
            pooled_width,
        ))?
    } else {
        ordered
    };
    let centered = if target == SigregTarget::TemporalResidual {
        let mean = pooled.sum(0)?.affine(1.0 / windows.window as f64, 0.0)?;
        pooled.broadcast_sub(&mean.broadcast_as(pooled.dims())?)?
    } else {
        pooled
    };
    let (_, _, _, pooled_height, pooled_width) = centered.dims5()?;
    let population = if cfg.sigreg_spatial {
        centered.permute((0, 1, 3, 4, 2))?.reshape((
            windows.window,
            windows.windows * pooled_height * pooled_width,
            channels,
        ))?
    } else {
        centered.reshape((
            windows.window,
            windows.windows,
            channels * pooled_height * pooled_width,
        ))?
    };
    subsample_sigreg_rows(
        &population,
        effective_sigreg_max_rows(cfg),
        seed.wrapping_add(0x5196_0001),
    )
}

/// Build the exact globally pooled population used by downstream `B x C`
/// consumers, while retaining the same ordered windows and temporal centering.
pub fn sigreg_global_stack_for_ordered_windows(
    latents: &Tensor,
    windows: &OrderedSigregWindows,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<Tensor> {
    let (rows, channels, height, width) = latents.dims4()?;
    if windows.row_indices.len() != windows.window.saturating_mul(windows.windows) {
        bail!("ordered SIGReg window metadata has an invalid row count");
    }
    if windows.row_indices.iter().any(|&row| row >= rows) {
        bail!("ordered SIGReg window row is outside encoded batch");
    }
    let indices = Tensor::from_vec(
        windows
            .row_indices
            .iter()
            .map(|&row| row as u32)
            .collect::<Vec<_>>(),
        (windows.row_indices.len(),),
        latents.device(),
    )?;
    let ordered = latents.index_select(&indices, 0)?.reshape((
        windows.window,
        windows.windows,
        channels,
        height,
        width,
    ))?;
    // Spatial pooling and temporal centering are both linear and commute. Pool
    // first so the regularized rows exactly match global `B x C` consumers.
    let pooled = ordered.mean(4)?.mean(3)?;
    let temporal_mean = pooled.sum(0)?.affine(1.0 / windows.window as f64, 0.0)?;
    let centered = pooled.broadcast_sub(&temporal_mean.broadcast_as(pooled.dims())?)?;
    let population = centered.reshape((windows.window, windows.windows, channels))?;
    subsample_sigreg_rows(
        &population,
        effective_sigreg_max_rows(cfg),
        seed.wrapping_add(0x5196_6001),
    )
}

fn sigreg_losses_for_ordered_windows(
    latents: &Tensor,
    windows: &OrderedSigregWindows,
    cfg: &TrainConfig,
    seed: u64,
) -> Result<(Tensor, Tensor)> {
    let mix = cfg.sigreg_global_mix;
    if mix == 0.0 {
        let stack =
            sigreg_stack_for_ordered_windows(latents, windows, cfg.sigreg_target, cfg, seed)?;
        let raw = sigreg_loss_for_stack(&stack, cfg, seed)?;
        let bounded = bounded_sigreg_loss(&raw)?;
        return Ok((raw, bounded));
    }

    let global_stack = sigreg_global_stack_for_ordered_windows(latents, windows, cfg, seed)?;
    let global_raw = sigreg_loss_for_stack(&global_stack, cfg, seed.wrapping_add(0x0061_0BA1))?;
    let global_bounded = bounded_sigreg_loss(&global_raw)?;
    if mix == 1.0 {
        return Ok((global_raw, global_bounded));
    }

    let cell_stack =
        sigreg_stack_for_ordered_windows(latents, windows, cfg.sigreg_target, cfg, seed)?;
    let cell_raw = sigreg_loss_for_stack(&cell_stack, cfg, seed)?;
    let cell_bounded = bounded_sigreg_loss(&cell_raw)?;
    let cell_weight = 1.0 - mix;
    Ok((
        cell_raw
            .affine(cell_weight, 0.0)?
            .add(&global_raw.affine(mix, 0.0)?)?,
        cell_bounded
            .affine(cell_weight, 0.0)?
            .add(&global_bounded.affine(mix, 0.0)?)?,
    ))
}

fn q_targets_from_mse(per: &Tensor, cfg: &TrainConfig) -> Result<Tensor> {
    if !cfg.q_quantile_targets {
        return per
            .lt(cfg.q_mse_threshold)?
            .to_dtype(DType::F32)
            .map_err(Into::into);
    }
    let flat = per.flatten_all()?;
    if flat.elem_count() == 0 {
        bail!("q_quantile_targets requires at least one sample");
    }
    let (sorted, _) = flat.sort_last_dim(true)?;
    let median = sorted.narrow(0, sorted.dim(0)? / 2, 1)?;
    per.lt(&median.broadcast_as(per.dims())?)?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

#[derive(Debug, Clone, Copy)]
pub struct FoundationV2ObjectiveConfig {
    pub ep_weight: f64,
    pub sigreg_projections: usize,
    pub sigreg_knots: usize,
    pub sigreg_seed: u64,
    pub q_mse_threshold: f64,
    pub rollout_enabled: bool,
    pub split_ce_weighting: SplitCeWeighting,
    pub split_ce_changed_budget: Option<f64>,
    /// Collect detached seam tensors for a profiled update's mechanism
    /// packet. Off on ordinary updates.
    pub capture_mechanism_seams: bool,
}

impl Default for FoundationV2ObjectiveConfig {
    fn default() -> Self {
        Self {
            ep_weight: 0.01,
            sigreg_projections: 8,
            sigreg_knots: 5,
            sigreg_seed: 1,
            q_mse_threshold: 0.05,
            rollout_enabled: false,
            split_ce_weighting: SplitCeWeighting::CurrentDouble,
            split_ce_changed_budget: None,
            capture_mechanism_seams: false,
        }
    }
}

#[derive(Debug, Clone)]
/// Detached seam tensors captured for a profiled update's mechanism packet.
pub struct FoundationV2MechanismSeams {
    pub out_y: Tensor,
    pub current_canonical: Tensor,
    pub predicted_canonical: Tensor,
    pub gate_logits: Tensor,
}

pub struct FoundationV2LossBreakdown {
    pub total: Tensor,
    pub non_ep_total: Tensor,
    pub pred_ce: Tensor,
    pub gate: Tensor,
    pub latent: Tensor,
    pub enc_ce: Tensor,
    pub separation: Tensor,
    pub pull: Tensor,
    pub inverse_action: Tensor,
    pub ep: Tensor,
    pub rollout: Tensor,
    pub event: Tensor,
    pub q: Tensor,
    pub reliability: Tensor,
    pub changed_weights: ChangedPixelWeights,
    pub factual_groups: usize,
    pub equivalent_pairs: usize,
    pub distinct_pairs: usize,
    pub inverse_action_rows: usize,
    /// Number of eligible fragments only when the rollout loss cleared its
    /// activation floor; zero means that the rollout objective was inert.
    pub rollout_fragments: usize,
    /// Populated only when the objective requested mechanism-seam capture.
    pub mechanism_seams: Option<FoundationV2MechanismSeams>,
}

pub const BF16_BENCHMARK_SCHEMA: &str = "p2.bf16_recurrent_core_benchmark.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bf16BenchmarkArm {
    pub bf16_recurrent_core: bool,
    pub median_step_ms: f64,
    pub min_step_ms: f64,
    pub max_step_ms: f64,
    pub step_ms: Vec<f64>,
    pub last_loss: f64,
    pub rollout_fragments: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bf16BenchmarkReport {
    pub schema: String,
    pub checkpoint: PathBuf,
    pub device: String,
    pub physical_batch: usize,
    pub grad_accum: usize,
    pub warmup_updates: usize,
    pub measured_updates: usize,
    /// Batch construction is excluded; each sample is one complete
    /// forward/loss/backward/clip/optimizer/EMA update bounded by device syncs.
    pub measured_region: String,
    pub f32: Bf16BenchmarkArm,
    pub bf16: Bf16BenchmarkArm,
    /// `f32 median / bf16 median`; values above one favor the treatment.
    pub speedup: f64,
}

fn median_f64(values: &[f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("BF16 benchmark requires non-empty finite timing samples");
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    })
}

fn benchmark_bf16_arm(
    baseline_cfg: &TrainConfig,
    checkpoint: &Path,
    mixed: &MixedStreamBatch,
    device: &Device,
    bf16_recurrent_core: bool,
    warmup_updates: usize,
    measured_updates: usize,
) -> Result<Bf16BenchmarkArm> {
    let mut cfg = baseline_cfg.clone();
    cfg.bf16_recurrent_core = bf16_recurrent_core;
    cfg.validate()?;
    let varmap = VarMap::new();
    let model = WorldModel::new(
        cfg.model_config(),
        VarBuilder::from_varmap(&varmap, DType::F32, device),
    )?;
    load_varmap_exact(&varmap, checkpoint)
        .with_context(|| format!("load BF16 benchmark checkpoint {}", checkpoint.display()))?;
    let mut optimizer = CheckpointHybridOptimizer::new(
        &varmap,
        adam_params(&cfg),
        cfg.muon_momentum,
        cfg.muon_rms_scale,
    )?;
    let mut ema = ModelEma::with_default_decay(&varmap)?;
    let event_slot_weights = event_slot_weight_tensor(device)?;
    let host = prepare_foundation_v2_batch_host(mixed)?;
    let total_updates = warmup_updates + measured_updates;
    let mut step_ms = Vec::with_capacity(measured_updates);
    let mut last_loss = f64::NAN;
    let mut rollout_fragments = 0usize;

    for update in 0..total_updates {
        sync_cuda_device(device)?;
        let started = std::time::Instant::now();
        let losses = foundation_v2_training_loss_with_event_weights(
            &model,
            mixed,
            &host,
            device,
            FoundationV2ObjectiveConfig {
                ep_weight: 0.01,
                sigreg_projections: cfg.sigreg_projections,
                sigreg_knots: cfg.sigreg_knots,
                sigreg_seed: cfg.seed.wrapping_add(update as u64),
                q_mse_threshold: cfg.q_mse_threshold,
                rollout_enabled: true,
                split_ce_weighting: cfg.split_ce_weighting,
                split_ce_changed_budget: cfg.split_ce_changed_budget,
                capture_mechanism_seams: false,
            },
            &event_slot_weights,
        )?;
        let total = losses.total.clone();
        let mut grads = total.backward()?;
        clip_gradients_gpu_with_stats(&mut grads, &varmap, MAX_GRAD_NORM)?;
        let learning_rate =
            foundation_v2_wsd_learning_rate(update + 1, cfg.steps_per_lesson.max(total_updates));
        optimizer.set_learning_rate(learning_rate)?;
        optimizer.step(&grads)?;
        ema.update(&varmap)?;
        sync_cuda_device(device)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

        last_loss = f64::from(total.to_dtype(DType::F32)?.to_scalar::<f32>()?);
        if !last_loss.is_finite() {
            bail!(
                "BF16 benchmark update {} produced non-finite loss",
                update + 1
            );
        }
        rollout_fragments = losses.rollout_fragments;
        if update >= warmup_updates {
            step_ms.push(elapsed_ms);
        }
        drop(grads);
        drop(total);
        drop(losses);
    }

    let median_step_ms = median_f64(&step_ms)?;
    let min_step_ms = step_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_step_ms = step_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Bf16BenchmarkArm {
        bf16_recurrent_core,
        median_step_ms,
        min_step_ms,
        max_step_ms,
        step_ms,
        last_loss,
        rollout_fragments,
    })
}

/// Run the preregistered warmed throughput comparison from identical F32
/// checkpoint weights and a single fixed foundation-v2 batch. This deliberately
/// excludes data generation while retaining the complete training update.
pub fn benchmark_bf16_recurrent_core(
    baseline_cfg: &TrainConfig,
    checkpoint: &Path,
    device_spec: &str,
    warmup_updates: usize,
    measured_updates: usize,
) -> Result<Bf16BenchmarkReport> {
    if warmup_updates < 20 || measured_updates < 100 {
        bail!("BF16 benchmark protocol requires at least 20 warmup and 100 measured updates");
    }
    if baseline_cfg.bf16_recurrent_core {
        bail!("BF16 benchmark requires an F32 baseline config");
    }
    baseline_cfg.validate()?;
    let device = resolve_device(device_spec)?;
    let mixed = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size: baseline_cfg.physical_batch,
            seed: baseline_cfg.seed,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        },
        0.0,
        0,
        V5DataSplit::Train,
    )?;
    let f32 = benchmark_bf16_arm(
        baseline_cfg,
        checkpoint,
        &mixed,
        &device,
        false,
        warmup_updates,
        measured_updates,
    )?;
    let bf16 = benchmark_bf16_arm(
        baseline_cfg,
        checkpoint,
        &mixed,
        &device,
        true,
        warmup_updates,
        measured_updates,
    )?;
    let speedup = f32.median_step_ms / bf16.median_step_ms;
    if !speedup.is_finite() {
        bail!("BF16 benchmark produced non-finite speedup");
    }
    Ok(Bf16BenchmarkReport {
        schema: BF16_BENCHMARK_SCHEMA.into(),
        checkpoint: checkpoint.to_path_buf(),
        device: device_spec.into(),
        physical_batch: baseline_cfg.physical_batch,
        grad_accum: baseline_cfg.grad_accum,
        warmup_updates,
        measured_updates,
        measured_region:
            "device-synchronized forward+loss+backward+clip+optimizer+ema; fixed batch".into(),
        f32,
        bf16,
        speedup,
    })
}

fn masked_mean_with_count(values: &Tensor, mask: &Tensor, count: usize) -> Result<Tensor> {
    if count == 0 {
        return values.zeros_like()?.sum_all().map_err(Into::into);
    }
    values
        .broadcast_mul(mask)?
        .sum_all()?
        .affine(1.0 / count as f64, 0.0)
        .map_err(Into::into)
}

fn mean_tensors_or_zero(values: Vec<Tensor>, zero: &Tensor) -> Result<Tensor> {
    if values.is_empty() {
        return Ok(zero.clone());
    }
    Tensor::stack(&values.iter().collect::<Vec<_>>(), 0)?
        .mean_all()
        .map_err(Into::into)
}

pub(crate) fn foundation_v2_unimix_ce(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let pixels = labels.elem_count();
    let selected = candle_nn::ops::softmax(logits, D::Minus1)?
        .reshape((pixels, PALETTE_SIZE))?
        .gather(&labels.flatten_all()?.unsqueeze(1)?, 1)?
        .reshape(labels.dims())?;
    selected
        .affine(0.99, 0.01 / PALETTE_SIZE as f64)?
        .log()?
        .neg()
        .map_err(Into::into)
}

#[cfg(test)]
fn foundation_v2_unimix_ce_reference(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let pixels = labels.elem_count();
    candle_nn::ops::softmax(logits, D::Minus1)?
        .affine(0.99, 0.01 / PALETTE_SIZE as f64)?
        .log()?
        .reshape((pixels, PALETTE_SIZE))?
        .gather(&labels.contiguous()?.flatten_all()?.unsqueeze(1)?, 1)?
        .reshape(labels.dims())?
        .neg()
        .map_err(Into::into)
}

pub fn split_weighted_ce(
    per_pixel: &Tensor,
    positive_mask: &Tensor,
    negative_mask: &Tensor,
    positive_count: usize,
    negative_count: usize,
    positive_weight: f64,
) -> Result<Tensor> {
    let positive = masked_mean_with_count(per_pixel, positive_mask, positive_count)?;
    let negative = masked_mean_with_count(per_pixel, negative_mask, negative_count)?;
    positive
        .affine(positive_weight, 0.0)?
        .add(&negative)
        .map_err(Into::into)
}

/// Weighting construction for the foundation-v2 split cross-entropy
/// (objective-isolation ablation). `CurrentDouble` is the ADR 0003 default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SplitCeWeighting {
    /// Existing behavior: the positive-stratum mean is scaled by
    /// `clamp((1-p)/p, 1, 50)`, so the per-pixel coefficient ratio is
    /// `((1-p)/p)^2` pre-cap ("double weighting").
    #[default]
    CurrentDouble,
    /// Equal aggregate coefficient shares for the two stratum means, rescaled
    /// to preserve the legacy nominal coefficient mass.
    EqualMeans,
    /// One pooled masked mean over all content pixels with per-pixel weight
    /// `w` on positive and 1.0 on negative pixels, so the per-pixel
    /// coefficient ratio is `(1-p)/p` once (copy-gate BCE construction).
    PooledPerPixel,
}

/// Mode dispatch for the split CE. `CurrentDouble` without a budget override
/// calls [`split_weighted_ce`] with unchanged arguments, keeping the default
/// path bit-for-bit identical. Every alternative redistributes the same total
/// coefficient mass as that legacy loss (`positive_weight + 1` when both
/// strata exist). This controls nominal loss coefficients; it does not
/// preserve gradient direction, norm, or clipping pressure. `changed_budget`
/// is an aggregate *coefficient* share, not a measured gradient share. If
/// either stratum is empty, all modes fall back to the only observed stratum
/// with the legacy coefficient mass.
#[allow(clippy::too_many_arguments)]
pub fn split_ce_with_weighting(
    per_pixel: &Tensor,
    positive_mask: &Tensor,
    negative_mask: &Tensor,
    positive_count: usize,
    negative_count: usize,
    positive_weight: f64,
    mode: SplitCeWeighting,
    changed_budget: Option<f64>,
) -> Result<Tensor> {
    if mode == SplitCeWeighting::CurrentDouble && changed_budget.is_none() {
        return split_weighted_ce(
            per_pixel,
            positive_mask,
            negative_mask,
            positive_count,
            negative_count,
            positive_weight,
        );
    }
    if positive_count == 0 && negative_count == 0 {
        return per_pixel.zeros_like()?.sum_all().map_err(Into::into);
    }

    let positive = masked_mean_with_count(per_pixel, positive_mask, positive_count)?;
    let negative = masked_mean_with_count(per_pixel, negative_mask, negative_count)?;
    if positive_count == 0 {
        return Ok(negative);
    }
    if negative_count == 0 {
        return positive.affine(positive_weight, 0.0).map_err(Into::into);
    }

    let coefficient_mass = positive_weight + 1.0;
    let positive_share = match changed_budget {
        Some(budget) => budget,
        None if mode == SplitCeWeighting::EqualMeans => 0.5,
        None => {
            let weighted_positive = positive_weight * positive_count as f64;
            weighted_positive / (weighted_positive + negative_count as f64)
        }
    };
    positive
        .affine(coefficient_mass * positive_share, 0.0)?
        .add(&negative.affine(coefficient_mass * (1.0 - positive_share), 0.0)?)
        .map_err(Into::into)
}

pub(crate) fn latent_content_mask<'a>(
    masks: impl ExactSizeIterator<Item = &'a ContentMask>,
    latent_height: usize,
    latent_width: usize,
    device: &Device,
) -> Result<Tensor> {
    let batch_size = masks.len();
    if batch_size == 0
        || latent_height == 0
        || latent_width == 0
        || !FRAME_SIDE.is_multiple_of(latent_height)
        || !FRAME_SIDE.is_multiple_of(latent_width)
    {
        bail!("invalid latent/content-mask geometry");
    }
    let patch_height = FRAME_SIDE / latent_height;
    let patch_width = FRAME_SIDE / latent_width;
    let mut values = Vec::with_capacity(batch_size * latent_height * latent_width);
    for mask in masks {
        if mask.values.len() != FRAME_SIDE * FRAME_SIDE {
            bail!("content mask is not fixed 64x64");
        }
        for latent_y in 0..latent_height {
            for latent_x in 0..latent_width {
                let occupied = (0..patch_height).any(|dy| {
                    let y = latent_y * patch_height + dy;
                    (0..patch_width).any(|dx| {
                        let x = latent_x * patch_width + dx;
                        mask.values[y * FRAME_SIDE + x] != 0
                    })
                });
                values.push(f32::from(occupied));
            }
        }
    }
    Tensor::from_vec(values, (batch_size, 1, latent_height, latent_width), device)
        .map_err(Into::into)
}

fn masked_spatial_huber(input: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if input.dims() != target.dims() {
        bail!("masked Huber input/target shapes differ");
    }
    let (_, channels, height, width) = input.dims4()?;
    if mask.dims() != [input.dim(0)?, 1, height, width] {
        bail!("masked Huber mask shape does not match latent geometry");
    }
    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;
    let quadratic = diff.sqr()?.affine(0.5, 0.0)?;
    let linear = abs_diff.affine(1.0, -0.5)?;
    let elementwise = abs_diff.le(1.0)?.where_cond(&quadratic, &linear)?;
    let numerator = elementwise.broadcast_mul(mask)?.sum_all()?;
    let denominator = mask
        .sum_all()?
        .clamp(1.0f32, f32::INFINITY)?
        .affine(channels as f64, 0.0)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn foundation_v2_rollout_loss(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    batch: &BatchTensors,
    encoded: &TrainingEncodedPair,
    device: &Device,
) -> Result<(Tensor, usize)> {
    foundation_v2_rollout_loss_inner(model, mixed, batch, encoded, device, false)
}

/// Read-only H2 rollout seam used by the frozen-checkpoint BF16 falsifier.
/// It deliberately evaluates only the rollout term, avoiding unrelated EP and
/// observer work while preserving the production fragment floor and graph.
pub fn foundation_v2_rollout_falsifier(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    device: &Device,
) -> Result<(f64, usize)> {
    let batch = batch_from_rows(mixed.samples(), device)?;
    let encoded = model
        .encode_state_pair_for_training_staged(&batch.model_frames, &batch.model_next_frames)?;
    let (loss, fragments) = foundation_v2_rollout_loss(model, mixed, &batch, &encoded, device)?;
    let value = f64::from(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?);
    if !value.is_finite() {
        bail!("BF16 falsifier H2 rollout loss is non-finite");
    }
    Ok((value, fragments))
}

/// `detach_open_loop_input` exists only for the gradient-attribution premise
/// test: detaching `first_out.y` before transition two isolates the
/// first-transition contribution to the core gradient. Production always
/// passes `false`.
fn foundation_v2_rollout_loss_inner(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    batch: &BatchTensors,
    encoded: &TrainingEncodedPair,
    device: &Device,
    detach_open_loop_input: bool,
) -> Result<(Tensor, usize)> {
    let mut fragments =
        BTreeMap::<(u64, u64, String), Vec<(usize, &TransitionSample, &ContentMask)>>::new();
    for (row, sample) in mixed
        .samples()
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.provenance.stream == MixedStreamKind::SequentialFragments)
    {
        let transition = sample.transition();
        fragments
            .entry((
                transition.seed,
                transition.episode_id,
                transition.family.clone(),
            ))
            .or_default()
            .push((row, transition, &sample.content_mask));
    }
    let mut first_rows = Vec::new();
    let mut second_rows = Vec::new();
    let mut second_masks = Vec::new();
    for fragment in fragments.values_mut() {
        fragment.sort_by_key(|(_, sample, _)| sample.transition_index);
        if let Some(pair) = fragment
            .windows(2)
            .find(|pair| pair[1].1.transition_index == pair[0].1.transition_index + 1)
        {
            first_rows.push(pair[0].0 as u32);
            second_rows.push(pair[1].0 as u32);
            second_masks.push(pair[1].2);
        }
    }
    if first_rows.len() < FOUNDATION_V2_MIN_ROLLOUT_FRAGMENTS {
        return Ok((Tensor::zeros((), DType::F32, device)?, 0));
    }
    // These are exact row selections from the batch encoded above: Foundation
    // V2 applies neither frame augmentation nor encoder noise, and recurrence
    // noise is independently fixed to zero. Reusing the shared tensors keeps
    // the encoder graph while removing the duplicate convolutional pass.
    let first_indices = Tensor::from_vec(first_rows.clone(), (first_rows.len(),), device)?;
    let second_indices = Tensor::from_vec(second_rows.clone(), (second_rows.len(),), device)?;
    let current = encoded.current.index_select(&first_indices, 0)?;
    let target_h2 = encoded.next.index_select(&second_indices, 0)?;
    let first_actions = batch.actions.index_select(&first_indices, 0)?;
    let first_action_coords = batch.action_coords.index_select(&first_indices, 0)?;
    let first_operator_conditioning = batch
        .operator_conditioning
        .index_select(&first_indices, 0)?;
    let second_actions = batch.actions.index_select(&second_indices, 0)?;
    let second_action_coords = batch.action_coords.index_select(&second_indices, 0)?;
    let second_operator_conditioning = batch
        .operator_conditioning
        .index_select(&second_indices, 0)?;
    // ADR 0005 §4: each rollout transition receives its own row's context.
    let context = batch
        .context
        .as_ref()
        .filter(|_| model.config().world_core_v6);
    let first_context = context
        .map(|context| context.select_rows(&first_rows))
        .transpose()?
        .flatten();
    let second_context = context
        .map(|context| context.select_rows(&second_rows))
        .transpose()?
        .flatten();
    let current_canonical = model.canonical_representation(&current)?;
    let first_out = model
        .full_v4_training_latents_from_encoded_state_with_operator_conditioning_with_context(
            &current,
            &current_canonical,
            &first_actions,
            &first_action_coords,
            &first_operator_conditioning,
            first_context.as_ref(),
            RecursionDepth::from_config(model.config()),
            0.0,
            None,
            RecursionOpts::training(true),
        )?;
    let open_loop_input = if detach_open_loop_input {
        first_out.y.detach()
    } else {
        first_out.y.clone()
    };
    let h1_canonical = model.canonical_representation(&open_loop_input)?;
    let second_out = model
        .full_v4_training_latents_from_encoded_state_with_operator_conditioning_with_context(
            &open_loop_input,
            &h1_canonical,
            &second_actions,
            &second_action_coords,
            &second_operator_conditioning,
            second_context.as_ref(),
            RecursionDepth::from_config(model.config()),
            0.0,
            None,
            RecursionOpts::training(true),
        )?;
    let (_, _, height, width) = second_out.y.dims4()?;
    let content_mask = latent_content_mask(second_masks.iter().copied(), height, width, device)?;
    let spatial = masked_spatial_huber(&second_out.y, &target_h2, &content_mask)?;
    let predicted_canonical =
        model.canonical_representation(&second_out.y.broadcast_mul(&content_mask)?)?;
    let target_canonical =
        model.canonical_representation(&target_h2.broadcast_mul(&content_mask)?)?;
    Ok((
        spatial.add(&candle_nn::loss::huber(
            &predicted_canonical,
            &target_canonical,
            1.0,
        )?)?,
        first_rows.len(),
    ))
}

/// Graded Foundation-v2 observer target shared by training and evaluation.
/// Changed rows score the composed copy-gate decode only on factually changed
/// pixels; no-change rows score it over the exact content mask. The target is
/// detached because Q/reliability are observer heads, not decoder objectives.
pub fn foundation_v2_graded_q_targets(
    model: &WorldModel,
    predicted_latent: &Tensor,
    current_frames: &Tensor,
    next_frames: &Tensor,
    content: &Tensor,
) -> Result<Tensor> {
    let predicted_latent = predicted_latent.detach();
    let predicted_logits = model.exact_gameplay_logits_detached(&predicted_latent)?;
    let copy_gate = model.exact_copy_gate(&predicted_latent)?.detach();
    Ok(foundation_v2_graded_q_targets_from_parts(
        model,
        &predicted_logits,
        &copy_gate,
        current_frames,
        next_frames,
        content,
    )?
    .0)
}

fn foundation_v2_graded_q_targets_from_parts(
    model: &WorldModel,
    predicted_logits: &Tensor,
    copy_gate: &Tensor,
    current_frames: &Tensor,
    next_frames: &Tensor,
    content: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch_size, rows, _, _) = predicted_logits.dims4()?;
    if content.dims() != [batch_size, rows, FRAME_SIDE] {
        bail!("foundation-v2 Q content mask has the wrong shape");
    }
    let current_labels = current_frames
        .narrow(2, 0, rows)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let target_labels = next_frames
        .narrow(2, 0, rows)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let composed =
        model.composed_gameplay_decode_from_parts(predicted_logits, copy_gate, &current_labels)?;
    foundation_v2_graded_q_targets_from_labels(&composed, &current_labels, &target_labels, content)
}

fn foundation_v2_graded_q_targets_from_labels(
    composed: &Tensor,
    current_labels: &Tensor,
    target_labels: &Tensor,
    content: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if composed.dims() != target_labels.dims()
        || current_labels.dims() != target_labels.dims()
        || content.dims() != target_labels.dims()
    {
        bail!("foundation-v2 Q label tensors have mismatched shapes");
    }
    let changed = current_labels
        .ne(target_labels)?
        .to_dtype(DType::F32)?
        .mul(content)?;
    let correct = composed.eq(target_labels)?.to_dtype(DType::F32)?;
    let changed_count_per_sample = changed.sum(2)?.sum(1)?;
    let content_count_per_sample = content.sum(2)?.sum(1)?;
    let changed_accuracy = correct
        .mul(&changed)?
        .sum(2)?
        .sum(1)?
        .div(&changed_count_per_sample.clamp(1.0, f64::INFINITY)?)?;
    let content_accuracy = correct
        .mul(content)?
        .sum(2)?
        .sum(1)?
        .div(&content_count_per_sample.clamp(1.0, f64::INFINITY)?)?;
    let targets = changed_count_per_sample
        .gt(0.0)?
        .where_cond(&changed_accuracy, &content_accuracy)?
        .unsqueeze(1)?
        .detach();
    let mask = content_count_per_sample
        .gt(0.0)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    Ok((targets, mask))
}

/// Foundation-v2 reliability is confidence in the factual next latent, using
/// the same full-spatial prediction/encoder-target MSE seam as evaluation.
fn foundation_v2_reliability_targets(
    predicted_latent: &Tensor,
    factual_target_latent: &Tensor,
    mse_threshold: f64,
) -> Result<Tensor> {
    if !(mse_threshold.is_finite() && mse_threshold >= 0.0) {
        bail!("foundation-v2 reliability MSE threshold must be finite and >= 0");
    }
    let mse = latent_mse_per_sample(predicted_latent, factual_target_latent)?.detach();
    // Keep the hard boundary used by evaluation. A sigmoid margin target would
    // require changing the binary calibration semantics as well.
    Ok(mse.le(mse_threshold)?.to_dtype(DType::F32)?.detach())
}

/// Foundation-v2's single mixed-stream objective. It is intentionally not an
/// adapter over `lesson_loss_weights`: all world and detached observer heads
/// are active together from update zero.
pub fn foundation_v2_training_loss(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    device: &Device,
    objective: FoundationV2ObjectiveConfig,
) -> Result<FoundationV2LossBreakdown> {
    let event_slot_weights = event_slot_weight_tensor(device)?;
    let host = prepare_foundation_v2_batch_host(mixed)?;
    foundation_v2_training_loss_with_event_weights(
        model,
        mixed,
        &host,
        device,
        objective,
        &event_slot_weights,
    )
}

fn foundation_v2_training_loss_with_event_weights(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    host: &PreparedFoundationV2BatchHost,
    device: &Device,
    objective: FoundationV2ObjectiveConfig,
    event_slot_weights: &Tensor,
) -> Result<FoundationV2LossBreakdown> {
    if !model.config().world_core_v4 || model.config().patch_size != PATCH_SIZE {
        bail!("foundation-v2 loss requires the patch-4 exact-decoder topology");
    }
    let batch = batch_from_foundation_v2_host(host, device)?;
    let v6 = model.config().world_core_v6;
    // ADR 0005 §1.3: v6 encodes the whole frame (row 63 is content), so the
    // host-staged EMPTY status rows are bypassed; legacy keeps the staged path.
    let encoded = if v6 {
        model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?
    } else {
        model
            .encode_state_pair_for_training_staged(&batch.model_frames, &batch.model_next_frames)?
    };
    // ADR 0005 §4: context rows use the same losses with the context supplied.
    let context = if v6 { batch.context.as_ref() } else { None };
    let current_canonical = model.canonical_representation(&encoded.current)?;
    let out = model
        .full_v4_training_latents_from_encoded_state_with_operator_conditioning_with_context(
            &encoded.current,
            &current_canonical,
            &batch.actions,
            &batch.action_coords,
            &batch.operator_conditioning,
            context,
            RecursionDepth::from_config(model.config()),
            0.0,
            None,
            RecursionOpts::training(true),
        )?;
    let predicted_canonical = model.canonical_representation(&out.y)?;
    let (_, _, latent_height, latent_width) = out.y.dims4()?;
    let expected_latent_side = FRAME_SIDE / PATCH_SIZE;
    if latent_height != expected_latent_side || latent_width != expected_latent_side {
        bail!("foundation-v2 latent geometry differs from patch-4 host preparation");
    }
    let latent_mask = Tensor::from_vec(
        host.latent_content_mask.clone(),
        (host.batch_size, 1, latent_height, latent_width),
        device,
    )?;
    let content_current_canonical =
        model.canonical_representation(&encoded.current.broadcast_mul(&latent_mask)?)?;
    let content_target_canonical =
        model.canonical_representation(&encoded.next.broadcast_mul(&latent_mask)?)?;
    let content_predicted_canonical =
        model.canonical_representation(&out.y.broadcast_mul(&latent_mask)?)?;
    let batch_size = host.batch_size;
    let rows = host.gameplay_rows;
    // ADR 0005 §1.1: a v6 batch supervises all 64 rows and therefore needs a
    // decoder that emits them; a legacy batch needs the 63-row decoder.
    if gameplay_rows(model.config().world_core_v6) != rows {
        bail!(
            "data contract supervises {rows} rows but the model decodes {}; set data_contract_v6 and world_core_v6 together",
            gameplay_rows(model.config().world_core_v6)
        );
    }
    let changed_weights = host.changed_weights;
    let content = Tensor::from_vec(
        host.content_values.clone(),
        (batch_size, rows, FRAME_SIDE),
        device,
    )?;
    let changed = Tensor::from_vec(
        host.changed_values.clone(),
        (batch_size, rows, FRAME_SIDE),
        device,
    )?;
    let unchanged = Tensor::from_vec(
        host.unchanged_values.clone(),
        (batch_size, rows, FRAME_SIDE),
        device,
    )?;
    let current_labels = batch
        .frames
        .narrow(2, 0, rows)?
        .squeeze(1)?
        .to_dtype(DType::U32)?
        .contiguous()?;
    let target_labels = batch
        .next_frames
        .narrow(2, 0, rows)?
        .squeeze(1)?
        .to_dtype(DType::U32)?
        .contiguous()?;

    let predicted_logits = model.exact_gameplay_logits_trainable(&out.y)?;
    let pred_per_pixel = foundation_v2_unimix_ce(&predicted_logits, &target_labels)?;
    let pred_ce = split_ce_with_weighting(
        &pred_per_pixel,
        &changed,
        &unchanged,
        changed_weights.changed_pixels,
        changed_weights.unchanged_pixels,
        changed_weights.changed_weight,
        objective.split_ce_weighting,
        objective.split_ce_changed_budget,
    )?;

    let gate_logits = model.exact_copy_gate_logits_trainable(&out.y)?;
    let gate =
        foundation_v2_copy_gate_loss(&gate_logits, &changed, changed_weights.changed_weight)?;

    let latent = masked_spatial_huber(&out.y, &encoded.next, &latent_mask)?.add(
        &candle_nn::loss::huber(&content_predicted_canonical, &content_target_canonical, 1.0)?,
    )?;

    let foreground = Tensor::from_vec(
        host.foreground_values.clone(),
        (batch_size, rows, FRAME_SIDE),
        device,
    )?;
    let background = content.sub(&foreground)?;
    // Current-frame foreground/background reconstruction is a distinct
    // objective and deliberately retains its fixed historical weighting. The
    // ablation knob changes only changed-vs-unchanged transition terms.
    let encoded_current_ce = split_weighted_ce(
        &foundation_v2_unimix_ce(
            &model.exact_gameplay_logits_trainable(&encoded.current)?,
            &current_labels,
        )?,
        &foreground,
        &background,
        host.foreground_count,
        host.background_count,
        host.foreground_weight,
    )?;
    let encoded_next_ce = split_ce_with_weighting(
        &foundation_v2_unimix_ce(
            &model.exact_gameplay_logits_trainable(&encoded.next)?,
            &target_labels,
        )?,
        &changed,
        &unchanged,
        changed_weights.changed_pixels,
        changed_weights.unchanged_pixels,
        changed_weights.changed_weight,
        objective.split_ce_weighting,
        objective.split_ce_changed_budget,
    )?;
    let enc_ce = encoded_current_ce.add(&encoded_next_ce)?.affine(0.5, 0.0)?;

    let zero = Tensor::zeros((), DType::F32, device)?;
    // Branch-effect and inverse-action auxiliaries must not classify the
    // translated PAD field. Derive displacement from the same content-masked
    // canonical seam used by the latent/EP corrections.
    let displacement = content_predicted_canonical.sub(&content_current_canonical)?;
    // Smooth the norm at the copy-bypass zero init. Besides keeping sqrt(0)
    // backward finite, eps=1e-3 bounds this normalization's Jacobian scale by
    // 1/eps=1_000 instead of the previous effective scale of 1_000_000.
    let displacement_norm = displacement
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0, FOUNDATION_V2_DISPLACEMENT_NORM_EPS.powi(2))?
        .sqrt()?;
    let normalized_displacement = displacement.broadcast_div(&displacement_norm)?;
    let mut pull_terms = Vec::new();
    let mut separation_terms = Vec::new();
    let mut equivalent_pairs = 0usize;
    let mut distinct_pairs = 0usize;
    if let Some(factual) = mixed.factual() {
        for label in factual.pairwise_board_effect_labels() {
            let factual_range = &factual.group_ranges()[label.group_index];
            let mixed_range = &mixed.factual_group_ranges()[label.group_index];
            let left = mixed_range.start + (label.left_row - factual_range.start);
            let right = mixed_range.start + (label.right_row - factual_range.start);
            let left = normalized_displacement.narrow(0, left, 1)?;
            let right = normalized_displacement.narrow(0, right, 1)?;
            if label.equivalent {
                pull_terms.push(left.sub(&right)?.sqr()?.mean_all()?);
                equivalent_pairs += 1;
            } else {
                separation_terms.push(separation_hinge_term(
                    &left,
                    &right,
                    FOUNDATION_V2_SEPARATION_MARGIN,
                )?);
                distinct_pairs += 1;
            }
        }
    }
    let pull = mean_tensors_or_zero(pull_terms, &zero)?;
    let separation = mean_tensors_or_zero(separation_terms, &zero)?;

    let mut inverse_rows = Vec::<u32>::new();
    if let Some(factual) = mixed.factual() {
        for (group, range) in factual.groups().iter().zip(mixed.factual_group_ranges()) {
            for (local, branch) in group.branches().iter().enumerate() {
                if branch.board_effect.changed {
                    inverse_rows.push((range.start + local) as u32);
                }
            }
        }
    }
    let inverse_action = if inverse_rows.is_empty() {
        zero.clone()
    } else {
        let indices = Tensor::from_vec(inverse_rows.clone(), (inverse_rows.len(),), device)?;
        let selected = displacement.index_select(&indices, 0)?;
        let (action_logits, coordinate_prediction) = model.decode_action_displacement(&selected)?;
        let action_targets = Tensor::from_vec(
            inverse_rows
                .iter()
                .map(|&row| u32::from(mixed.samples()[row as usize].transition().action.id))
                .collect::<Vec<_>>(),
            (inverse_rows.len(),),
            device,
        )?;
        let action_ce = candle_nn::loss::cross_entropy(&action_logits, &action_targets)?;
        let action6 = inverse_rows
            .iter()
            .enumerate()
            .filter_map(|(selected_row, &mixed_row)| {
                (mixed.samples()[mixed_row as usize].transition().action.id == 6)
                    .then_some((selected_row, mixed_row))
            })
            .collect::<Vec<_>>();
        if action6.is_empty() {
            action_ce
        } else {
            let action6_indices = Tensor::from_vec(
                action6
                    .iter()
                    .map(|(selected_row, _)| *selected_row as u32)
                    .collect::<Vec<_>>(),
                (action6.len(),),
                device,
            )?;
            let predicted_coords = coordinate_prediction.index_select(&action6_indices, 0)?;
            let expected_coords = Tensor::from_vec(
                action6
                    .iter()
                    .flat_map(|(_, mixed_row)| {
                        let action = &mixed.samples()[*mixed_row as usize].transition().action;
                        [
                            f32::from(action.x.expect("ACTION6 x")) / 63.0,
                            f32::from(action.y.expect("ACTION6 y")) / 63.0,
                        ]
                    })
                    .collect::<Vec<_>>(),
                (action6.len(), 2),
                device,
            )?;
            action_ce.add(&predicted_coords.sub(&expected_coords)?.sqr()?.mean_all()?)?
        }
    };

    let ep_population = Tensor::stack(
        &[
            content_current_canonical.clone(),
            content_target_canonical.clone(),
        ],
        0,
    )?;
    let ep = sigreg_epps_pulley_seeded(
        &ep_population,
        objective.sigreg_projections,
        objective.sigreg_knots,
        objective.sigreg_seed,
    )?;

    let detached_canonical = predicted_canonical.detach();
    let event_logits = model.event_logits_from_canonical(&detached_canonical, &batch.goals)?;
    let event = masked_bce_with_slot_weights(
        &event_logits,
        &batch.event_targets,
        &batch.event_mask,
        Some(event_slot_weights),
    )?;
    // Both projections above consumed this exact post-outer-step `out.y`.
    // Reuse their detached values for the observer targets instead of
    // repeating the full exact decoder and copy-gate convolution.
    let predicted_copy_gate = candle_nn::ops::sigmoid(&gate_logits.detach())?;
    let (graded_targets, graded_mask) = foundation_v2_graded_q_targets_from_parts(
        model,
        &predicted_logits.detach(),
        &predicted_copy_gate,
        &batch.frames,
        &batch.next_frames,
        &content,
    )?;
    let q = masked_bce_with_slot_weights(
        &model.q_logit_from_canonical(&detached_canonical)?,
        &graded_targets,
        &graded_mask,
        None,
    )?;
    let reliability_targets =
        foundation_v2_reliability_targets(&out.y, &encoded.next, objective.q_mse_threshold)?;
    // Q masks rows without gameplay content because pixel accuracy is then
    // undefined. Full-spatial latent MSE remains defined for every encoded
    // row, so reliability uses the same masked-BCE policy with an all-row mask.
    let reliability_mask = Tensor::ones_like(&reliability_targets)?;
    let reliability = masked_bce_with_slot_weights(
        &model.reliability_logit_from_canonical(&detached_canonical)?,
        &reliability_targets,
        &reliability_mask,
        None,
    )?;
    let (rollout, rollout_fragments) = if objective.rollout_enabled {
        foundation_v2_rollout_loss(model, mixed, &batch, &encoded, device)?
    } else {
        (zero.clone(), 0)
    };

    let mut non_ep_total = pred_ce.clone();
    for (weight, loss) in [
        (0.5, &gate),
        (0.25, &latent),
        (0.1, &enc_ce),
        (0.2, &separation),
        (0.1, &pull),
        (0.1, &inverse_action),
        (0.02, &rollout),
        (0.1, &event),
        (0.1, &q),
        (0.1, &reliability),
    ] {
        non_ep_total = non_ep_total.add(&loss.affine(weight, 0.0)?)?;
    }
    let total = non_ep_total.add(&ep.affine(objective.ep_weight, 0.0)?)?;
    let mechanism_seams = objective
        .capture_mechanism_seams
        .then(|| FoundationV2MechanismSeams {
            out_y: out.y.detach(),
            current_canonical: current_canonical.detach(),
            predicted_canonical: predicted_canonical.detach(),
            gate_logits: gate_logits.detach(),
        });
    Ok(FoundationV2LossBreakdown {
        total,
        non_ep_total,
        pred_ce,
        gate,
        latent,
        enc_ce,
        separation,
        pull,
        inverse_action,
        ep,
        rollout,
        event,
        q,
        reliability,
        changed_weights,
        factual_groups: mixed.factual_group_ranges().len(),
        equivalent_pairs,
        distinct_pairs,
        inverse_action_rows: inverse_rows.len(),
        rollout_fragments,
        mechanism_seams,
    })
}

/// Fixed Full V4 objective behind one recipe-specific interface.
///
/// The legacy objective remains an adapter for historical recipes. This module
/// owns Full V4's lesson split, canonical-state reuse, frozen observer semantics,
/// and fine-grained device attribution so those facts do not leak through the
/// generic loss implementation.
struct FullV4Objective<'a> {
    model: &'a WorldModel,
    batch: &'a BatchTensors,
    cfg: &'a TrainConfig,
    depth: RecursionDepth,
    sigreg_seed: u64,
    weights: LessonLossWeights,
    profile: Option<&'a RepresentativeUpdateCapture>,
}

impl FullV4Objective<'_> {
    fn phase<T>(
        &self,
        name: &str,
        step: Option<ExecutionStep>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match self.profile {
            Some(profile) => profile.synchronized_phase(
                self.batch.frames.device(),
                name,
                SpanKind::Module,
                step,
                f,
            ),
            None => f(),
        }
    }

    fn compute(&self) -> Result<LossBreakdown> {
        if self.cfg.recipe != TrainingRecipe::FullV4 {
            bail!("FullV4Objective requires the Full V4 recipe");
        }
        if self.weights.rollout > 0.0 || self.weights.prefix > 0.0 || self.weights.ptrm_rank {
            // Rollout is owned by the update loop and Full V4 excludes prefix
            // and PTRM objectives. Fail closed if the resolved recipe drifts.
            if self.weights.prefix > 0.0 || self.weights.ptrm_rank {
                bail!("Full V4 objective received an excluded prefix or PTRM loss");
            }
        }
        if self.weights.world > 0.0 || self.weights.sigreg > 0.0 {
            self.world_lesson()
        } else {
            self.observer_lesson()
        }
    }

    fn world_lesson(&self) -> Result<LossBreakdown> {
        let encoded = self.phase(
            "objective.encode_pair",
            Some(ExecutionStep::Forward),
            || {
                self.model.encode_state_pair_for_training_staged(
                    &self.batch.model_frames,
                    &self.batch.model_next_frames,
                )
            },
        )?;
        let current_canonical = self.phase(
            "objective.current_canonical",
            Some(ExecutionStep::Forward),
            || self.model.canonical_representation(&encoded.current),
        )?;
        let target_canonical = self.phase(
            "objective.target_canonical",
            Some(ExecutionStep::Forward),
            || self.model.canonical_representation(&encoded.next),
        )?;
        let out = self.phase("objective.recurrence", Some(ExecutionStep::Forward), || {
            self.model.full_v4_training_latents_from_encoded_state(
                &encoded.current,
                &current_canonical,
                &self.batch.actions,
                &self.batch.action_coords,
                self.depth,
                0.0,
                Some(self.sigreg_seed.wrapping_add(0x7E57)),
                RecursionOpts::training(true),
            )
        })?;
        let predicted_canonical = self.phase(
            "objective.predicted_canonical",
            Some(ExecutionStep::Forward),
            || self.model.canonical_representation(&out.y),
        )?;
        let next_latent = self.phase(
            "objective.world_prediction",
            Some(ExecutionStep::Forward),
            || {
                let spatial = candle_nn::loss::huber(&out.y, &encoded.next, 1.0)?;
                let canonical =
                    candle_nn::loss::huber(&predicted_canonical, &target_canonical, 1.0)?;
                spatial.add(&canonical).map_err(Into::into)
            },
        )?;
        let grounding = self.phase(
            "objective.exact_grounding",
            Some(ExecutionStep::Forward),
            || {
                let current = self
                    .model
                    .exact_grounding_loss(&encoded.current, &self.batch.frames)?;
                let target = self
                    .model
                    .exact_grounding_loss(&encoded.next, &self.batch.next_frames)?;
                current.add(&target)?.affine(0.5, 0.0).map_err(Into::into)
            },
        )?;
        let (sigreg_raw, sigreg_bounded) = if self.weights.sigreg > 0.0 {
            let raw = self.phase("objective.sigreg", Some(ExecutionStep::Forward), || {
                let population =
                    Tensor::stack(&[current_canonical.clone(), target_canonical.clone()], 0)?;
                sigreg_epps_pulley_seeded(
                    &population,
                    self.cfg.sigreg_projections,
                    self.cfg.sigreg_knots,
                    self.sigreg_seed,
                )
            })?;
            (raw.clone(), raw)
        } else {
            let zero = Tensor::zeros((), DType::F32, self.batch.frames.device())?;
            (zero.clone(), zero)
        };
        let mut total = next_latent.affine(self.weights.world, 0.0)?;
        if self.weights.sigreg > 0.0 {
            total = total.add(&sigreg_bounded.affine(self.weights.sigreg, 0.0)?)?;
        }
        total = total
            .add(&grounding.affine(self.cfg.exact_grounding_weight * self.weights.world, 0.0)?)?;
        self.breakdown(
            total,
            next_latent,
            sigreg_raw,
            sigreg_bounded,
            grounding,
            None,
            None,
            None,
        )
    }

    fn observer_lesson(&self) -> Result<LossBreakdown> {
        let device = self.batch.frames.device();
        if self.weights.event == 0.0 && self.weights.q == 0.0 && self.weights.reliability == 0.0 {
            let zero = Tensor::zeros((), DType::F32, device)?;
            return self.breakdown(
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
                None,
                None,
                None,
            );
        }
        let current = self.phase(
            "objective.encode_current",
            Some(ExecutionStep::Forward),
            || self.model.encode_state(&self.batch.frames),
        )?;
        let current_canonical = self.phase(
            "objective.current_canonical",
            Some(ExecutionStep::Forward),
            || self.model.canonical_representation(&current),
        )?;
        let out = self.phase("objective.recurrence", Some(ExecutionStep::Forward), || {
            self.model.full_v4_training_latents_from_encoded_state(
                &current,
                &current_canonical,
                &self.batch.actions,
                &self.batch.action_coords,
                self.depth,
                0.0,
                Some(self.sigreg_seed.wrapping_add(0x7E57)),
                RecursionOpts::training(true),
            )
        })?;
        let predicted_canonical = self.phase(
            "objective.predicted_canonical",
            Some(ExecutionStep::Forward),
            || self.model.canonical_representation(&out.y),
        )?;
        let detached_canonical = predicted_canonical.detach();
        let exact_targets = if self.weights.q > 0.0 || self.weights.reliability > 0.0 {
            Some(self.phase(
                "objective.exact_observer_targets",
                Some(ExecutionStep::Forward),
                || {
                    self.model.exact_transition_correctness(
                        &out.y.detach(),
                        &self.batch.frames,
                        &self.batch.next_frames,
                    )
                },
            )?)
        } else {
            None
        };
        let event = if self.weights.event > 0.0 {
            Some(
                self.phase("objective.event_head", Some(ExecutionStep::Forward), || {
                    let logits = self
                        .model
                        .event_logits_from_canonical(&detached_canonical, &self.batch.goals)?;
                    let slot_weights = event_slot_weight_tensor(device)?;
                    masked_bce_with_slot_weights(
                        &logits,
                        &self.batch.event_targets,
                        &self.batch.event_mask,
                        Some(&slot_weights),
                    )
                })?,
            )
        } else {
            None
        };
        let q = if self.weights.q > 0.0 {
            Some(
                self.phase("objective.q_head", Some(ExecutionStep::Forward), || {
                    let logits = self.model.q_logit_from_canonical(&detached_canonical)?;
                    bce_with_logits(&logits, exact_targets.as_ref().expect("Full V4 Q labels"))
                })?,
            )
        } else {
            None
        };
        let reliability = if self.weights.reliability > 0.0 {
            Some(self.phase(
                "objective.reliability_head",
                Some(ExecutionStep::Forward),
                || {
                    let logits = self
                        .model
                        .reliability_logit_from_canonical(&detached_canonical)?;
                    bce_with_logits(
                        &logits,
                        exact_targets.as_ref().expect("Full V4 reliability labels"),
                    )
                },
            )?)
        } else {
            None
        };
        let mut total = Tensor::zeros((), DType::F32, device)?;
        for (weight, loss) in [
            (self.weights.event, event.as_ref()),
            (self.weights.q, q.as_ref()),
            (self.weights.reliability, reliability.as_ref()),
        ] {
            if let Some(loss) = loss {
                total = total.add(&loss.affine(weight, 0.0)?)?;
            }
        }
        let zero = Tensor::zeros((), DType::F32, device)?;
        self.breakdown(
            total,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            event,
            q,
            reliability,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn breakdown(
        &self,
        total: Tensor,
        next_latent: Tensor,
        sigreg_raw: Tensor,
        sigreg_bounded: Tensor,
        grounding: Tensor,
        event: Option<Tensor>,
        q: Option<Tensor>,
        reliability: Option<Tensor>,
    ) -> Result<LossBreakdown> {
        let zero = Tensor::zeros((), DType::F32, self.batch.frames.device())?;
        Ok(LossBreakdown {
            total,
            next_latent,
            sigreg_raw,
            sigreg_bounded,
            patch_grounding: grounding,
            grounding_changed_patches: 0,
            grounding_unchanged_patches: 0,
            event: event.unwrap_or_else(|| zero.clone()),
            q: q.unwrap_or_else(|| zero.clone()),
            q_surprise: zero.clone(),
            ptrm_rank: zero.clone(),
            prefix: zero.clone(),
            reliability: reliability.unwrap_or_else(|| zero.clone()),
            branch_total: zero.clone(),
            outcome_pull: zero.clone(),
            outcome_push: zero.clone(),
            action_recovery: zero.clone(),
            coordinate_recovery: zero.clone(),
            changed_margin: zero.clone(),
            spatial_variance: zero.clone(),
            spatial_covariance: zero.clone(),
            pooled_variance: zero.clone(),
            pooled_covariance: zero.clone(),
            displacement_variance: zero.clone(),
            displacement_covariance: zero,
            branch_audit: BranchLearningAudit::default(),
        })
    }
}

/// LeWorld loss: mean next-latent MSE over outer steps + SIGReg + masked aux heads.
pub fn leworld_loss(
    model: &WorldModel,
    batch: &BatchTensors,
    cfg: &TrainConfig,
    depth: RecursionDepth,
    sigreg_seed: u64,
    weights: LessonLossWeights,
) -> Result<LossBreakdown> {
    leworld_loss_with_sigreg_windows(
        model,
        batch,
        None,
        None,
        cfg,
        depth,
        sigreg_seed,
        weights,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn leworld_loss_with_sigreg_windows(
    model: &WorldModel,
    batch: &BatchTensors,
    sigreg_windows: Option<&OrderedSigregWindows>,
    samples: Option<&[TransitionSample]>,
    cfg: &TrainConfig,
    depth: RecursionDepth,
    sigreg_seed: u64,
    weights: LessonLossWeights,
    profile: Option<&RepresentativeUpdateCapture>,
) -> Result<LossBreakdown> {
    if cfg.recipe == TrainingRecipe::FullV4 {
        return FullV4Objective {
            model,
            batch,
            cfg,
            depth,
            sigreg_seed,
            weights,
            profile,
        }
        .compute();
    }
    let z_noise = if cfg.train_z_noise > 0.0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(sigreg_seed.wrapping_add(0x5A5A_5A5A));
        if rng.random::<f64>() < 0.5 {
            cfg.train_z_noise
        } else {
            0.0
        }
    } else {
        0.0
    };
    let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
    let cur_z = encoded.current;
    let next_z = encoded.next;
    let out = model.training_latents_from_encoded_state(
        &cur_z,
        &batch.actions,
        &batch.action_coords,
        depth,
        z_noise,
        Some(sigreg_seed.wrapping_add(0x7E57)),
        RecursionOpts::training(cfg.supervise_last_outer_only),
    )?;

    let device = batch.frames.device();
    let zero = Tensor::zeros((), DType::F32, device)?;
    let next_latent = if weights.world == 0.0 {
        zero.clone()
    } else if cfg.recipe == TrainingRecipe::FullV4 {
        let spatial = candle_nn::loss::huber(&out.y, &next_z, 1.0)?;
        let predicted_canonical = model.canonical_representation(&out.y)?;
        let target_canonical = model.canonical_representation(&next_z)?;
        let canonical = candle_nn::loss::huber(&predicted_canonical, &target_canonical, 1.0)?;
        spatial.add(&canonical)?
    } else if cfg.supervise_last_outer_only {
        out.y.sub(&next_z)?.sqr()?.mean_all()?
    } else {
        let mut pred_acc: Option<Tensor> = None;
        for step in &out.steps {
            let mse = step.sub(&next_z)?.sqr()?.mean_all()?;
            pred_acc = Some(match pred_acc {
                None => mse,
                Some(acc) => acc.add(&mse)?,
            });
        }
        let n_steps = out.steps.len().max(1) as f64;
        pred_acc
            .ok_or_else(|| anyhow::anyhow!("no outer steps"))?
            .affine(1.0 / n_steps, 0.0)?
    };

    let grounding = if cfg.recipe == TrainingRecipe::FullV4 && weights.world > 0.0 {
        let current = model.exact_grounding_loss(&cur_z, &batch.frames)?;
        let target = model.exact_grounding_loss(&next_z, &batch.next_frames)?;
        crate::p2::grounding::PatchGroundingLoss {
            total: current.add(&target)?.affine(0.5, 0.0)?,
            changed_patches: 0,
            unchanged_patches: 0,
        }
    } else if cfg.patch_grounding_weight > 0.0 {
        let samples = samples.ok_or_else(|| {
            anyhow::anyhow!("patch grounding requires transition sample provenance")
        })?;
        model.patch_histogram_grounding_loss(&out.y, &next_z, samples, cfg.patch_grounding_mode)?
    } else {
        crate::p2::grounding::PatchGroundingLoss {
            total: zero.clone(),
            changed_patches: 0,
            unchanged_patches: 0,
        }
    };
    let (sigreg_raw, sigreg_bounded) = if weights.sigreg == 0.0 {
        (zero.clone(), zero.clone())
    } else {
        if cfg.recipe == TrainingRecipe::FullV4 {
            model_sigreg_losses_for_encoded_pair(
                model,
                &cur_z,
                &next_z,
                &encoded.current_raw,
                &encoded.next_raw,
                encoded.projected_sigreg.as_ref(),
                cfg,
                sigreg_seed,
            )?
        } else {
            match sigreg_windows {
                Some(windows) if !cfg.sigreg_pre_rms_spatial && !cfg.sigreg_projector => {
                    sigreg_losses_for_ordered_windows(&cur_z, windows, cfg, sigreg_seed)?
                }
                None if cfg.sigreg_target == SigregTarget::TemporalResidual => bail!(
                "temporal-residual SIGReg requires at least one complete ordered transition window"
            ),
                _ => sigreg_losses_for_encoded_pair(
                    &cur_z,
                    &next_z,
                    &encoded.current_raw,
                    &encoded.next_raw,
                    encoded.projected_sigreg.as_ref(),
                    cfg,
                    sigreg_seed,
                )?,
            }
        }
    };

    let (event_raw, event) = if weights.event > 0.0 {
        let slot_weights = event_slot_weight_tensor(device)?;
        let event_logits = if cfg.recipe == TrainingRecipe::FullV4 {
            let canonical = model.canonical_representation(&out.y)?.detach();
            model.event_logits_from_canonical(&canonical, &batch.goals)?
        } else {
            let event_y = if cfg.stop_grad_event_y {
                out.y.detach()
            } else {
                out.y.clone()
            };
            model.event_logits_from(&event_y, &batch.goals)?
        };
        let raw = masked_bce_with_slot_weights(
            &event_logits,
            &batch.event_targets,
            &batch.event_mask,
            Some(&slot_weights),
        )?;
        (raw.clone(), raw)
    } else {
        (zero.clone(), zero.clone())
    };

    let exact_observer_targets =
        if cfg.recipe == TrainingRecipe::FullV4 && (weights.q > 0.0 || weights.reliability > 0.0) {
            Some(model.exact_transition_correctness(
                &out.y.detach(),
                &batch.frames,
                &batch.next_frames,
            )?)
        } else {
            None
        };

    let (q_raw, q, q_logit, q_mse_per_sample) = if weights.q > 0.0 {
        let q_y = if cfg.recipe == TrainingRecipe::FullV4 || cfg.stop_grad_q_y {
            out.y.detach()
        } else {
            out.y.clone()
        };
        let q_logit = if cfg.recipe == TrainingRecipe::FullV4 {
            let canonical = model.canonical_representation(&q_y)?.detach();
            model.q_logit_from_canonical(&canonical)?
        } else {
            model.q_logit_from_y(&q_y)?
        };
        let per = latent_mse_per_sample(&q_y, &next_z)?;
        let q_targets = if cfg.recipe == TrainingRecipe::FullV4 {
            exact_observer_targets
                .as_ref()
                .expect("Full V4 observer labels were prepared")
                .clone()
        } else {
            q_targets_from_mse(&per.detach(), cfg)?
        };
        let raw = bce_with_logits(&q_logit, &q_targets)?;
        (raw.clone(), raw, Some(q_logit), Some(per))
    } else {
        (zero.clone(), zero.clone(), None, None)
    };

    let (rel_raw, reliability) = if weights.reliability > 0.0 {
        let detached_y = out.y.detach();
        let q_targets = if cfg.recipe == TrainingRecipe::FullV4 {
            exact_observer_targets
                .as_ref()
                .expect("Full V4 observer labels were prepared")
                .clone()
        } else {
            let per = latent_mse_per_sample(&detached_y, &next_z)?.detach();
            q_targets_from_mse(&per, cfg)?
        };
        let reliability_logit = if cfg.recipe == TrainingRecipe::FullV4 {
            let canonical = model.canonical_representation(&detached_y)?.detach();
            model.reliability_logit_from_canonical(&canonical)?
        } else {
            model.reliability_logit_from_y(&detached_y)?
        };
        let raw = bce_with_logits(&reliability_logit, &q_targets)?;
        (raw.clone(), raw)
    } else {
        (zero.clone(), zero.clone())
    };

    let branch = if cfg.world_core_v2 {
        let _ = samples.ok_or_else(|| {
            anyhow::anyhow!("world-core-v2 loss requires factual sample provenance")
        })?;
        let transition = ConsumerTransition::try_new(cur_z.clone(), out.y.clone(), next_z.clone())?;
        branch_learning_loss(
            model,
            batch.factual.as_ref(),
            &transition,
            &cfg.branch_learning,
            batch.factual.is_some(),
        )?
    } else {
        let transition = ConsumerTransition::try_new(cur_z.clone(), out.y.clone(), next_z.clone())?;
        branch_learning_loss(
            model,
            None,
            &transition,
            &BranchLearningConfig::default(),
            false,
        )?
    };

    let (prefix_raw, prefix) = if weights.prefix > 0.0 {
        let raw = prefix_one_step_loss(model, batch, &cur_z, &next_z)?;
        (raw.clone(), raw)
    } else {
        (zero.clone(), zero.clone())
    };

    let mut total = if weights.world > 0.0 {
        next_latent.affine(weights.world, 0.0)?
    } else {
        zero.clone()
    };
    for (weight, loss) in [
        (weights.sigreg, &sigreg_bounded),
        (
            if cfg.recipe == TrainingRecipe::FullV4 {
                cfg.exact_grounding_weight * weights.world
            } else {
                cfg.patch_grounding_weight
            },
            &grounding.total,
        ),
        (weights.event, &event),
        (weights.q, &q),
        (weights.reliability, &reliability),
        (weights.prefix, &prefix),
    ] {
        if weight > 0.0 {
            total = total.add(&loss.affine(weight, 0.0)?)?;
        }
    }
    if cfg.world_core_v2 {
        total = total.add(&branch.total)?;
    }
    let q_surprise = if weights.q > 0.0 && !cfg.stop_grad_q_y {
        let q_prob = candle_nn::ops::sigmoid(q_logit.as_ref().expect("active Q head"))?;
        q_prob
            .mul(q_mse_per_sample.as_ref().expect("active Q error"))?
            .mean_all()?
    } else {
        zero.clone()
    };
    if weights.q > 0.0 && !cfg.stop_grad_q_y {
        total = total.add(&q_surprise.affine(Q_SURPRISE_WEIGHT, 0.0)?)?;
    }
    let ptrm_rank = if weights.ptrm_rank {
        ptrm_ranking_loss(
            model,
            &cur_z,
            batch,
            &next_z,
            depth,
            weights.ptrm_rank_k,
            0.1,
            sigreg_seed.wrapping_add(1),
        )?
    } else {
        zero
    };
    if weights.ptrm_rank {
        total = total.add(&ptrm_rank.affine(PTRM_RANK_WEIGHT, 0.0)?)?;
    }

    Ok(LossBreakdown {
        total,
        next_latent,
        sigreg_raw,
        sigreg_bounded,
        patch_grounding: grounding.total,
        grounding_changed_patches: grounding.changed_patches,
        grounding_unchanged_patches: grounding.unchanged_patches,
        event: event_raw,
        q: q_raw,
        q_surprise,
        ptrm_rank,
        prefix: prefix_raw,
        reliability: rel_raw,
        branch_total: branch.total,
        outcome_pull: branch.outcome_pull,
        outcome_push: branch.outcome_push,
        action_recovery: branch.action_recovery,
        coordinate_recovery: branch.coordinate_recovery,
        changed_margin: branch.changed_margin,
        spatial_variance: branch.spatial_variance,
        spatial_covariance: branch.spatial_covariance,
        pooled_variance: branch.pooled_variance,
        pooled_covariance: branch.pooled_covariance,
        displacement_variance: branch.displacement_variance,
        displacement_covariance: branch.displacement_covariance,
        branch_audit: branch.audit,
    })
}

fn collect_rollout_trace(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    for offset in 0..1024u64 {
        let samples =
            generate_curriculum(curriculum, seed, start_episode.wrapping_add(offset), split)?;
        if samples.len() >= 2 {
            return Ok(samples);
        }
    }
    bail!("failed to find a multi-step trace for curriculum {curriculum}")
}

fn batch_prefetch_requests(
    curriculum: &str,
    cfg: &TrainConfig,
    global_step: u64,
    accum: usize,
) -> Vec<PrefetchRequest> {
    let mut reqs = Vec::with_capacity(accum);
    for micro in 0..accum {
        reqs.push(PrefetchRequest {
            curriculum: curriculum.to_string(),
            seed: cfg.seed,
            episode_start: scheduled_episode_start(
                cfg.seed,
                global_step,
                micro,
                accum,
                cfg.shuffled_episodes,
            ),
            physical_batch: cfg.physical_batch,
            split: Split::Train,
        });
    }
    reqs
}

fn enqueue_batch_prefetch(
    prefetcher: &mut BatchPrefetcher,
    curriculum: &str,
    cfg: &TrainConfig,
    global_step: u64,
    accum: usize,
) -> Result<()> {
    prefetcher.submit_many(&batch_prefetch_requests(
        curriculum,
        cfg,
        global_step,
        accum,
    ))
}

fn prefetch_lookahead_steps() -> u64 {
    std::env::var("TOFY_P2_PREFETCH_LOOKAHEAD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(1)
}

fn ensure_prefetch_scope(
    prefetcher: &mut Option<BatchPrefetcher>,
    prefetched_through_step: &mut u64,
    curriculum: &str,
    cfg: &TrainConfig,
    global_step: u64,
) {
    let expected = PrefetchScope {
        curriculum: curriculum.to_string(),
        seed: cfg.seed,
        physical_batch: cfg.physical_batch,
        split: Split::Train,
    };
    let stale = prefetcher
        .as_ref()
        .and_then(BatchPrefetcher::scope)
        .is_some_and(|active| active != &expected);
    if stale {
        if let Some(active) = prefetcher.as_mut() {
            active.shutdown();
        }
        *prefetcher = Some(BatchPrefetcher::new());
        *prefetched_through_step = global_step;
    }
}

fn restart_prefetch_pipeline(
    prefetcher: &mut Option<BatchPrefetcher>,
    prefetched_through_step: &mut u64,
    global_step: u64,
) {
    *prefetcher = Some(BatchPrefetcher::new());
    *prefetched_through_step = global_step;
}

/// Keep `lookahead` optimizer steps of microbatches queued so CPU generation runs ahead of GPU.
fn top_up_prefetch(
    prefetched_through_step: &mut u64,
    prefetcher: &mut BatchPrefetcher,
    curriculum: &str,
    cfg: &TrainConfig,
    global_step: u64,
    accum: usize,
) -> Result<()> {
    prefetcher.poll();
    let want_through = global_step.saturating_add(prefetch_lookahead_steps());
    while *prefetched_through_step < want_through {
        enqueue_batch_prefetch(prefetcher, curriculum, cfg, *prefetched_through_step, accum)?;
        *prefetched_through_step = prefetched_through_step.saturating_add(1);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_one_micro_sample_batch(
    micro: usize,
    accum: usize,
    use_prefetch: bool,
    prefetcher: Option<&mut BatchPrefetcher>,
    cfg: &TrainConfig,
    curriculum: &str,
    global_step: u64,
    episode_cache: &mut EpisodeCache,
) -> Result<Vec<TransitionSample>> {
    if use_prefetch {
        return prefetcher
            .expect("prefetch enabled without prefetcher")
            .recv();
    }
    let episode_start =
        scheduled_episode_start(cfg.seed, global_step, micro, accum, cfg.shuffled_episodes);
    if cfg.shuffled_episodes {
        collect_batch_uncached(
            curriculum,
            cfg.seed,
            episode_start,
            cfg.physical_batch,
            Split::Train,
            None,
        )
    } else {
        episode_cache.collect(
            curriculum,
            cfg.seed,
            episode_start,
            cfg.physical_batch,
            Split::Train,
        )
    }
}

/// Open-loop latent rollout with optional AR-forcing resets to real encodings.
pub fn open_loop_latent_loss(
    model: &WorldModel,
    trace: &OrderedTraceTensors,
    horizon: usize,
    depth: RecursionDepth,
    teacher_mix: f64,
    seed: u64,
) -> Result<Tensor> {
    let trace_len = trace.frames.dim(0)?;
    if trace_len < 2 || horizon < 2 {
        bail!("open-loop loss requires at least two ordered transitions");
    }
    let steps = horizon.min(trace_len);
    let mut latent = model.encode_state(&trace.frames.narrow(0, 0, 1)?)?;
    let targets = model.encode_state(&trace.next_frames.narrow(0, 0, steps)?)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut total: Option<Tensor> = None;
    let mut n = 0usize;
    for step in 0..steps {
        let predicted = model.predict_latent_with_depth(
            &latent,
            &trace.actions.narrow(0, step, 1)?,
            &trace.action_coords.narrow(0, step, 1)?,
            depth,
        )?;
        let target = targets.narrow(0, step, 1)?;
        let mse = candle_nn::loss::huber(&predicted, &target, 1.0)?;
        let capped = smooth_cap_nonnegative(&mse, ROLLOUT_STEP_LOSS_CAP)?;
        total = Some(match total {
            None => capped,
            Some(acc) => acc.add(&capped)?,
        });
        n += 1;
        let teacher = teacher_mix > 0.0 && rng.random::<f64>() < teacher_mix;
        let reset = mse.mean_all()?.gt(ROLLOUT_ERROR_RESET as f64)?;
        if teacher {
            latent = target.detach();
        } else {
            let reset_mask = reset
                .reshape((1, 1, 1, 1))?
                .broadcast_as(predicted.dims())?;
            latent = reset_mask.where_cond(&target.detach(), &predicted)?;
        }
    }
    total
        .ok_or_else(|| anyhow::anyhow!("open-loop trace was empty"))?
        .affine(1.0 / n.max(1) as f64, 0.0)
        .map_err(Into::into)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(value).context("serialize json")?;
    let tmp = {
        let mut os = path.as_os_str().to_owned();
        os.push(".tmp");
        PathBuf::from(os)
    };
    fs::write(&tmp, &json).with_context(|| format!("write {}", tmp.display()))?;
    File::open(&tmp)
        .with_context(|| format!("open {} for sync", tmp.display()))?
        .sync_all()
        .with_context(|| format!("sync {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "rename {} -> {} (atomic replace)",
            tmp.display(),
            path.display()
        )
    })?;
    if let Some(parent) = path.parent() {
        File::open(parent)
            .with_context(|| format!("open {} for sync", parent.display()))?
            .sync_all()
            .with_context(|| format!("sync {}", parent.display()))?;
    }
    Ok(())
}

fn ensure_profile_campaign_manifest(cfg: &TrainConfig, capture_steps: &[u64]) -> Result<()> {
    if capture_steps.is_empty() {
        return Ok(());
    }
    let mut steps = capture_steps.to_vec();
    steps.sort_unstable();
    let manifest = CampaignManifest {
        schema: CAMPAIGN_SCHEMA.into(),
        campaign_id: format!("tofy.p2.{:?}.seed-{}", cfg.recipe, cfg.seed).to_lowercase(),
        entrypoint: PROFILE_ENTRYPOINT.into(),
        planned: steps
            .into_iter()
            .map(|capture_step| PlannedCapture {
                capture_step,
                bundle: format!("update-{capture_step:012}"),
            })
            .collect(),
    };
    manifest.validate()?;
    let path = cfg.output_dir.join("profile/campaign.json");
    if path.exists() {
        let existing = CampaignManifest::load(&path)?;
        if existing != manifest {
            bail!(
                "profile campaign manifest {} conflicts with the requested capture plan",
                path.display()
            );
        }
        return Ok(());
    }
    write_json_atomic(&path, &manifest)
}

pub fn save_checkpoint(varmap: &VarMap, cfg: &TrainConfig, report: &TrainReport) -> Result<()> {
    fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("create {}", cfg.output_dir.display()))?;
    let weights = cfg.output_dir.join("model.safetensors");
    let weights_tmp = cfg.output_dir.join("model.safetensors.tmp");
    // Foundation-v2's exported evaluation checkpoint is EMA by contract; the
    // resumable bundle still retains both live and EMA weights.
    let bundle_weights =
        report
            .latest_checkpoint
            .join(if cfg.recipe == TrainingRecipe::FoundationV2 {
                "ema.safetensors"
            } else {
                "model.safetensors"
            });
    // Fail-closed verification, run unconditionally for foundation-v2: a
    // `checkpoints/best` that exists but does not match this run's own
    // gate-history selection (a foreign bundle, another resume branch, a
    // stale step) must abort publication loudly. Running it only when the
    // report already claims an export would make the loud path unreachable,
    // because the claim itself is produced by the same verification.
    if cfg.recipe == TrainingRecipe::FoundationV2 {
        if let Some(foundation) = report.foundation_v2.as_ref() {
            let verified = foundation_v2_verified_best_export(
                cfg,
                foundation.promotion_metric,
                &foundation.gate_history,
            )?;
            if report.export_checkpoint.as_deref() != verified.as_deref() {
                bail!(
                    "export checkpoint claim {:?} disagrees with best-selection \
                     verification {:?}",
                    report.export_checkpoint,
                    verified
                );
            }
        }
    }
    if let Some(export) = &report.export_checkpoint {
        fs::copy(export, &weights_tmp).with_context(|| {
            format!(
                "copy export checkpoint {} -> {}",
                export.display(),
                weights_tmp.display()
            )
        })?;
        if cfg.recipe == TrainingRecipe::FoundationV2 {
            let bundle = export
                .parent()
                .ok_or_else(|| anyhow::anyhow!("best EMA export has no bundle directory"))?;
            verify_checkpoint_artifact_hash(bundle, "ema.safetensors", export)?;
            verify_checkpoint_artifact_hash(bundle, "ema.safetensors", &weights_tmp)?;
        }
    } else if bundle_weights.is_file() {
        fs::copy(&bundle_weights, &weights_tmp).with_context(|| {
            format!(
                "copy checkpoint weights {} -> {}",
                bundle_weights.display(),
                weights_tmp.display()
            )
        })?;
    } else {
        varmap
            .save(&weights_tmp)
            .with_context(|| format!("save {}", weights_tmp.display()))?;
    }
    File::open(&weights_tmp)?.sync_all()?;
    fs::rename(&weights_tmp, &weights)
        .with_context(|| format!("rename {} -> {}", weights_tmp.display(), weights.display()))?;
    File::open(&cfg.output_dir)?.sync_all()?;
    write_json_atomic(
        &cfg.output_dir.join("config.json"),
        &persist_train_config(cfg),
    )?;
    write_json_atomic(&cfg.output_dir.join("train_report.json"), report)?;
    Ok(())
}

fn save_export_snapshot(varmap: &VarMap, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir).with_context(|| format!("create {}", output_dir.display()))?;
    let path = output_dir.join("model.best.safetensors");
    let tmp = output_dir.join("model.best.safetensors.tmp");
    varmap
        .save(&tmp)
        .with_context(|| format!("save {}", tmp.display()))?;
    File::open(&tmp)?.sync_all()?;
    fs::rename(&tmp, &path).with_context(|| {
        format!(
            "publish export snapshot {} -> {}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

#[derive(Deserialize)]
struct BundleGlobalStep {
    global_step: u64,
}

/// Foundation-v2 export selector bound to the run's own selection state
/// (ADR 0003: the exported evaluation model is the selected checkpoint's
/// EMA). Fails when `checkpoints/best` exists but does not hold the step this
/// run's gate history actually promoted — an explicitly resumed foreign
/// bundle must not be published as this run's best.
fn foundation_v2_verified_best_export(
    cfg: &TrainConfig,
    metric: PromotionMetric,
    gate_history: &[FoundationV2GateEvaluation],
) -> Result<Option<PathBuf>> {
    let best_dir = cfg.output_dir.join("checkpoints/best");
    let ema = best_dir.join("ema.safetensors");
    let expected_step = foundation_v2_selected_best_step(metric, gate_history);
    if !ema.is_file() {
        if let Some(expected_step) = expected_step {
            bail!(
                "gate history selected checkpoint step {expected_step}, but {} is missing; refusing to fall back to latest EMA weights",
                best_dir.display()
            );
        }
        if best_dir.exists() {
            bail!(
                "{} exists without ema.safetensors; refusing an incomplete best checkpoint",
                best_dir.display()
            );
        }
        return Ok(None);
    }
    let Some(expected_step) = expected_step else {
        bail!(
            "{} exists but this run's gate history has promoted no checkpoint; \
             refusing to export an unattributed best",
            best_dir.display()
        );
    };
    verify_checkpoint_bundle(&best_dir)
        .with_context(|| format!("verify selected best checkpoint {}", best_dir.display()))?;
    let bundle: BundleGlobalStep = read_json(&best_dir.join("trainer_state.json"))?;
    if bundle.global_step != expected_step {
        bail!(
            "best checkpoint at {} holds step {} but this run's gate history selected \
             step {}; refusing a misattributed export",
            best_dir.display(),
            bundle.global_step,
            expected_step
        );
    }
    Ok(Some(ema))
}

fn verify_checkpoint_artifact_hash(bundle: &Path, artifact: &str, path: &Path) -> Result<()> {
    let manifest: CheckpointBundleManifest = read_json(&bundle.join("bundle-manifest.json"))?;
    let expected = manifest
        .artifacts
        .iter()
        .find(|entry| entry.path == artifact)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint manifest {} does not bind {artifact}",
                bundle.display()
            )
        })?;
    let actual = sha256_file(path)?;
    if actual.bytes != expected.bytes || actual.sha256 != expected.sha256 {
        bail!(
            "checkpoint artifact integrity mismatch for export {} against {}",
            path.display(),
            bundle.display()
        );
    }
    Ok(())
}

fn export_checkpoint_path(cfg: &TrainConfig, state: &TrainerState) -> Option<PathBuf> {
    if cfg.recipe == TrainingRecipe::FoundationV2 {
        let foundation = state.foundation_v2.as_ref()?;
        // A verification failure yields no export claim here; `save_checkpoint`
        // re-runs the same check fallibly and fails publication closed.
        foundation_v2_verified_best_export(cfg, cfg.promotion_metric, &foundation.gate_history)
            .ok()
            .flatten()
    } else {
        let best = cfg.output_dir.join("model.best.safetensors");
        best.exists().then_some(best)
    }
}

pub fn load_weights(varmap: &mut VarMap, path: &Path) -> Result<()> {
    varmap
        .load(path)
        .with_context(|| format!("load weights {}", path.display()))
}

pub fn load_train_config(path: &Path) -> Result<TrainConfig> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).context("parse TrainConfig")
}

/// Per-phase step timing, enabled by setting `TOFY_P2_STEP_PROFILE` to a report interval.
///
/// Candle's CUDA ops are asynchronous, so every phase boundary is forced to a device
/// sync before the clock is read. That costs a little throughput, which is why this is
/// opt-in: without the syncs the timings would all pile onto whichever call happens to
/// block first.
#[derive(Default)]
struct StepProfile {
    interval: usize,
    steps: usize,
    generate: f64,
    stage: f64,
    forward: f64,
    backward: f64,
    optimizer: f64,
    metrics: f64,
    checkpoint: f64,
}

impl StepProfile {
    fn from_env() -> Self {
        let interval = std::env::var("TOFY_P2_STEP_PROFILE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        Self {
            interval,
            ..Self::default()
        }
    }

    fn enabled(&self) -> bool {
        self.interval > 0
    }

    fn report(&mut self, step: u64) {
        if !self.enabled() || self.steps < self.interval {
            return;
        }
        let n = self.steps as f64;
        let total = self.generate
            + self.stage
            + self.forward
            + self.backward
            + self.optimizer
            + self.metrics
            + self.checkpoint;
        println!(
            "[profile step {step}] {:.1}ms/step = generate {:.1} | stage+h2d {:.1} | \
             forward {:.1} | backward {:.1} | optimizer {:.1} | metrics(d2h) {:.1} | \
             checkpoint {:.1}",
            total / n,
            self.generate / n,
            self.stage / n,
            self.forward / n,
            self.backward / n,
            self.optimizer / n,
            self.metrics / n,
            self.checkpoint / n,
        );
        let interval = self.interval;
        *self = Self {
            interval,
            ..Self::default()
        };
    }
}

/// Time `f` into `sink`, draining the device queue first so the measurement covers
/// only `f` and not whatever earlier async work happened to still be in flight.
fn timed<T>(
    enabled: bool,
    device: &Device,
    sink: &mut f64,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !enabled {
        return f();
    }
    device.synchronize()?;
    let start = std::time::Instant::now();
    let value = f()?;
    device.synchronize()?;
    *sink += start.elapsed().as_secs_f64() * 1e3;
    Ok(value)
}

fn install_pause_handler() -> Result<()> {
    let result = PAUSE_HANDLER.get_or_init(|| {
        ctrlc::set_handler(|| {
            if PAUSE_REQUESTED.swap(true, Ordering::SeqCst) {
                eprintln!("second interrupt — forcing exit");
                std::process::exit(130);
            }
            eprintln!("pause requested — finishing current step and saving checkpoint (Ctrl+C again to force quit)");
        })
        .map_err(|err| err.to_string())
    });
    result
        .as_ref()
        .map_err(|err| anyhow::anyhow!("install SIGINT/SIGTERM pause handler: {err}"))
        .copied()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn implicit_resume_source(cfg: &TrainConfig) -> Option<PathBuf> {
    if cfg.resume.is_some() {
        return None;
    }
    let checkpoints = cfg.output_dir.join("checkpoints");
    if checkpoints.join("latest.json").is_file() {
        Some(checkpoints)
    } else {
        None
    }
}

fn batch_schedule_migration(
    saved: &TrainingContract,
    requested: &TrainingContract,
) -> Option<BatchScheduleMigration> {
    let mut migrated = saved.clone();
    migrated.physical_batch = requested.physical_batch;
    migrated.grad_accum = requested.grad_accum;
    (migrated == *requested
        && (saved.physical_batch, saved.grad_accum)
            != (requested.physical_batch, requested.grad_accum)
        && effective_batch_contract(saved) == effective_batch_contract(requested))
    .then(|| BatchScheduleMigration {
        from_physical_batch: saved.physical_batch,
        from_grad_accum: saved.grad_accum,
        to_physical_batch: requested.physical_batch,
        to_grad_accum: requested.grad_accum,
        effective_batch: effective_batch_contract(saved),
        label: "trajectory_migration_equal_effective_batch".into(),
    })
}

fn resolve_resume_checkpoint(path: &Path) -> Result<PathBuf> {
    let (bundle, latest) = if path.join("trainer_state.json").is_file() {
        (path.to_path_buf(), None)
    } else {
        let latest_path = if path.is_file() {
            path.to_path_buf()
        } else if path.join("latest.json").is_file() {
            path.join("latest.json")
        } else {
            path.join("checkpoints/latest.json")
        };
        let latest: LatestCheckpoint = read_json(&latest_path).with_context(|| {
            format!(
                "resume expects a checkpoint bundle or a directory containing latest.json: {}",
                path.display()
            )
        })?;
        if latest.schema != TRAINER_STATE_SCHEMA {
            bail!("unsupported latest checkpoint schema {}", latest.schema);
        }
        let parent = latest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("latest checkpoint path has no parent"))?;
        if latest.directory.contains('/') || latest.directory.contains('\\') {
            bail!(
                "latest checkpoint directory must be a plain child name: {}",
                latest.directory
            );
        }
        let mut components = Path::new(&latest.directory).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            bail!(
                "latest checkpoint directory must be a plain child name: {}",
                latest.directory
            );
        }
        (parent.join(&latest.directory), Some(latest))
    };
    if bundle
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
    {
        bail!(
            "refusing to resume from staging directory {}",
            bundle.display()
        );
    }
    for required in checkpoint_required_artifacts(&bundle)?
        .iter()
        .copied()
        .chain(["bundle-manifest.json"])
    {
        if !bundle.join(required).is_file() {
            bail!(
                "checkpoint bundle is incomplete (missing {required}): {}",
                bundle.display()
            );
        }
    }
    verify_checkpoint_bundle(&bundle)?;
    let state: BundleGlobalStep = read_json(&bundle.join("trainer_state.json"))?;
    if let Some(latest) = latest {
        if latest.global_step != state.global_step {
            bail!(
                "latest checkpoint step {} disagrees with restored trainer state step {} in {}",
                latest.global_step,
                state.global_step,
                bundle.display()
            );
        }
    }
    Ok(bundle)
}

fn checkpoint_required_artifacts(bundle: &Path) -> Result<&'static [&'static str]> {
    #[derive(Deserialize)]
    struct RecipeOnly {
        #[serde(default)]
        recipe: TrainingRecipe,
    }

    let config: RecipeOnly = read_json(&bundle.join("config.json"))?;
    Ok(if config.recipe == TrainingRecipe::FoundationV2 {
        FOUNDATION_V2_CHECKPOINT_ARTIFACTS
    } else {
        LEGACY_CHECKPOINT_ARTIFACTS
    })
}

fn sha256_file(path: &Path) -> Result<CheckpointArtifactDigest> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok(CheckpointArtifactDigest {
        path: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .into(),
        bytes,
        sha256: format!("sha256:{:x}", digest.finalize()),
    })
}

fn checkpoint_parameter_group(name: &str) -> &'static str {
    if name.starts_with("exact_grounding_head.") {
        "exact_decoder"
    } else if name.starts_with("event_head.")
        || name.starts_with("q_head.")
        || name.starts_with("reliability_head.")
        || name.starts_with("goal_proj.")
        || name.starts_with("action_decoder.")
        || name.starts_with("coordinate_decoder.")
    {
        "observers"
    } else if name.starts_with("grounding_head.") || name.starts_with("sigreg_projector.") {
        "auxiliary_decoders"
    } else {
        "world"
    }
}

fn model_parameter_group_hashes(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .with_context(|| format!("read safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len > 64 * 1024 * 1024 {
        bail!("implausible safetensors header length {header_len}");
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("read safetensors header from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;
    let tensors = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header is not an object"))?;
    let mut entries = tensors
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .map(|(name, tensor)| {
            let offsets = tensor
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("tensor {name} has no data_offsets"))?;
            if offsets.len() != 2 {
                bail!("tensor {name} has invalid data_offsets");
            }
            Ok((
                name.clone(),
                offsets[0]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("tensor {name} start is not u64"))?,
                offsets[1]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("tensor {name} end is not u64"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let data_start = 8 + header_len;
    let mut digests = BTreeMap::<String, Sha256>::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for (name, start, end) in entries {
        if end < start {
            bail!("tensor {name} has reversed offsets");
        }
        let group = checkpoint_parameter_group(&name).to_string();
        let digest = digests.entry(group).or_default();
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((end - start).to_le_bytes());
        file.seek(SeekFrom::Start(data_start + start))?;
        let mut remaining = end - start;
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..take])?;
            digest.update(&buffer[..take]);
            remaining -= take as u64;
        }
    }
    Ok(digests
        .into_iter()
        .map(|(group, digest)| (group, format!("sha256:{:x}", digest.finalize())))
        .collect())
}

pub(crate) fn verify_checkpoint_bundle(bundle: &Path) -> Result<()> {
    let manifest: CheckpointBundleManifest = read_json(&bundle.join("bundle-manifest.json"))?;
    if manifest.schema != "p2.checkpoint_bundle.v1" {
        bail!("unsupported checkpoint bundle schema {}", manifest.schema);
    }
    let required = checkpoint_required_artifacts(bundle)?
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    let mut declared = BTreeSet::new();
    for expected in &manifest.artifacts {
        let mut components = Path::new(&expected.path).components();
        if expected.path.contains('/')
            || expected.path.contains('\\')
            || !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
        {
            bail!("unsafe checkpoint artifact path {}", expected.path);
        }
        if !declared.insert(expected.path.clone()) {
            bail!("duplicate checkpoint artifact path {}", expected.path);
        }
    }
    if declared != required {
        bail!(
            "checkpoint manifest artifact set mismatch in {}: declared {:?}, required {:?}",
            bundle.display(),
            declared,
            required
        );
    }
    for expected in &manifest.artifacts {
        let actual = sha256_file(&bundle.join(&expected.path))?;
        if actual.bytes != expected.bytes || actual.sha256 != expected.sha256 {
            bail!(
                "checkpoint artifact integrity mismatch for {}",
                bundle.join(&expected.path).display()
            );
        }
    }
    let state: BundleGlobalStep = read_json(&bundle.join("trainer_state.json"))?;
    if manifest.global_step != state.global_step {
        bail!(
            "checkpoint manifest step {} disagrees with trainer state step {} in {}",
            manifest.global_step,
            state.global_step,
            bundle.display()
        );
    }
    let actual_groups = model_parameter_group_hashes(&bundle.join("model.safetensors"))?;
    if actual_groups != manifest.parameter_groups {
        bail!(
            "checkpoint parameter-group integrity mismatch in {}",
            bundle.display()
        );
    }
    Ok(())
}

fn unused_checkpoint_sibling(parent: &Path, stem: &str) -> PathBuf {
    for sequence in 0u64.. {
        let candidate = parent.join(format!(".{stem}-{}-{sequence}", std::process::id()));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("u64 checkpoint sibling namespace exhausted")
}

fn write_latest_checkpoint(checkpoints: &Path, directory: String, global_step: u64) -> Result<()> {
    write_json_atomic(
        &checkpoints.join("latest.json"),
        &LatestCheckpoint {
            schema: TRAINER_STATE_SCHEMA.into(),
            directory,
            global_step,
        },
    )
}

fn save_training_checkpoint(
    varmap: &VarMap,
    optimizer: &CheckpointHybridOptimizer,
    ema: Option<&ModelEma>,
    state: &TrainerState,
    cfg: &TrainConfig,
) -> Result<PathBuf> {
    let output_dir = &cfg.output_dir;
    let checkpoints = output_dir.join("checkpoints");
    fs::create_dir_all(&checkpoints)
        .with_context(|| format!("create {}", checkpoints.display()))?;
    let directory = format!("step-{:012}", state.global_step);
    let final_dir = checkpoints.join(&directory);
    if final_dir.exists() {
        match verify_checkpoint_bundle(&final_dir) {
            Ok(()) => {
                let existing: BundleGlobalStep = read_json(&final_dir.join("trainer_state.json"))?;
                if existing.global_step != state.global_step {
                    bail!(
                        "existing checkpoint {} holds step {}, expected {}",
                        final_dir.display(),
                        existing.global_step,
                        state.global_step
                    );
                }
                tracing::warn!(
                    "adopting verified deterministic checkpoint {} and repairing latest.json",
                    final_dir.display()
                );
                write_latest_checkpoint(&checkpoints, directory, state.global_step)?;
                return Ok(final_dir);
            }
            Err(error) => {
                let backup =
                    unused_checkpoint_sibling(&checkpoints, &format!("{directory}.corrupt"));
                fs::rename(&final_dir, &backup).with_context(|| {
                    format!(
                        "preserve corrupt checkpoint {} -> {}",
                        final_dir.display(),
                        backup.display()
                    )
                })?;
                File::open(&checkpoints)?.sync_all()?;
                tracing::warn!(
                    "preserved unverifiable checkpoint {} at {} before replacement: {error:#}",
                    final_dir.display(),
                    backup.display()
                );
            }
        }
    }
    let staging = unused_checkpoint_sibling(&checkpoints, &format!("{directory}.tmp"));
    fs::create_dir(&staging).with_context(|| format!("create {}", staging.display()))?;

    let model_path = staging.join("model.safetensors");
    let optimizer_path = staging.join("optimizer.safetensors");
    varmap
        .save(&model_path)
        .with_context(|| format!("save {}", model_path.display()))?;
    optimizer
        .save(&optimizer_path)
        .with_context(|| format!("save {}", optimizer_path.display()))?;
    let ema_path = ema
        .map(|ema| -> Result<PathBuf> {
            let path = staging.join("ema.safetensors");
            ema.weights()
                .save(&path)
                .with_context(|| format!("save {}", path.display()))?;
            Ok(path)
        })
        .transpose()?;
    let trainer_state_path = staging.join("trainer_state.json");
    let bundle_config_path = staging.join("config.json");
    let persisted_config = persist_train_config(cfg);
    write_json_atomic(&trainer_state_path, state)?;
    let gate_history_path = state
        .foundation_v2
        .as_ref()
        .map(|foundation| {
            let path = staging.join("gate_history.json");
            write_json_atomic(&path, &foundation.gate_history).map(|_| path)
        })
        .transpose()?;
    write_json_atomic(&bundle_config_path, &persisted_config)?;
    write_json_atomic(&output_dir.join("config.json"), &persisted_config)?;
    let mut artifact_paths = vec![
        model_path.clone(),
        optimizer_path.clone(),
        trainer_state_path.clone(),
        bundle_config_path.clone(),
    ];
    artifact_paths.extend(ema_path.iter().cloned());
    artifact_paths.extend(gate_history_path.iter().cloned());
    let artifacts = artifact_paths
        .iter()
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;
    let bundle_manifest_path = staging.join("bundle-manifest.json");
    write_json_atomic(
        &bundle_manifest_path,
        &CheckpointBundleManifest {
            schema: "p2.checkpoint_bundle.v1".into(),
            global_step: state.global_step,
            artifacts,
            parameter_groups: model_parameter_group_hashes(&model_path)?,
        },
    )?;
    File::open(&model_path)?.sync_all()?;
    File::open(&optimizer_path)?.sync_all()?;
    if let Some(path) = &ema_path {
        File::open(path)?.sync_all()?;
    }
    if let Some(path) = &gate_history_path {
        File::open(path)?.sync_all()?;
    }
    File::open(&trainer_state_path)?.sync_all()?;
    File::open(&bundle_config_path)?.sync_all()?;
    File::open(&bundle_manifest_path)?.sync_all()?;
    File::open(&staging)?.sync_all()?;
    fs::rename(&staging, &final_dir).with_context(|| {
        format!(
            "publish checkpoint {} -> {}",
            staging.display(),
            final_dir.display()
        )
    })?;
    File::open(&checkpoints)?.sync_all()?;
    write_latest_checkpoint(&checkpoints, directory, state.global_step)?;
    Ok(final_dir)
}

fn load_training_checkpoint(
    bundle: &Path,
    cfg: &TrainConfig,
    varmap: &mut VarMap,
    optimizer: &mut CheckpointHybridOptimizer,
    ema: Option<&mut ModelEma>,
) -> Result<TrainerState> {
    let mut state: TrainerState = read_json(&bundle.join("trainer_state.json"))?;
    if state.schema != TRAINER_STATE_SCHEMA {
        bail!("unsupported trainer state schema {}", state.schema);
    }
    let requested = TrainingContract::from(cfg);
    if state.contract.foundation_objective_revision != requested.foundation_objective_revision {
        bail!(
            "foundation-v2 objective revision mismatch: checkpoint={} requested={}",
            state.contract.foundation_objective_revision,
            requested.foundation_objective_revision
        );
    }
    // Older V5 bundles predate the derived experiment field. Their legacy
    // contract already carries every input used to resolve it, and those fields
    // are compared below; hydrate only the absent derived value for exact resume.
    if state.contract.experiment.is_none() {
        state.contract.experiment = requested.experiment.clone();
    }
    if state.contract.profile_updates.is_none() {
        state.contract.profile_updates = requested.profile_updates.clone();
    }
    if state.contract != requested {
        let migration = cfg
            .allow_batch_schedule_migration
            .then(|| batch_schedule_migration(&state.contract, &requested))
            .flatten()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resume training contract mismatch; checkpoint={} requested={}",
                    serde_json::to_string(&state.contract).unwrap_or_default(),
                    serde_json::to_string(&requested).unwrap_or_default()
                )
            })?;
        tracing::warn!(
            "{}: physical_batch {}→{} grad_accum {}→{} (effective_batch={})",
            migration.label,
            migration.from_physical_batch,
            migration.to_physical_batch,
            migration.from_grad_accum,
            migration.to_grad_accum,
            migration.effective_batch,
        );
        state.contract = requested;
        state.batch_schedule_migrations.push(migration);
    }
    if state.parameter_names != optimizer.parameter_names() {
        bail!("resume parameter names do not exactly match the current model");
    }
    if state.global_step
        != state.optimizer_step as u64 + state.nonfinite_skipped_updates.len() as u64
    {
        bail!(
            "checkpoint cursor mismatch: global_step={} optimizer_step={} nonfinite_skips={}",
            state.global_step,
            state.optimizer_step,
            state.nonfinite_skipped_updates.len()
        );
    }
    if cfg.recipe == TrainingRecipe::FoundationV2 {
        let foundation = state
            .foundation_v2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("foundation-v2 cannot resume a v4 trainer state"))?;
        if state.lesson_index != 0
            || state.step_in_lesson != 0
            || !state.completed_lessons.is_empty()
            || foundation.total_steps != cfg.steps_per_lesson
            || state.global_step > foundation.total_steps as u64
        {
            bail!("foundation-v2 checkpoint contains an invalid non-lesson cursor");
        }
    } else {
        if state.foundation_v2.is_some() {
            bail!("legacy recipes cannot resume a foundation-v2 trainer state");
        }
        let lesson_steps = resolved_lesson_steps(cfg);
        if state.lesson_index > cfg.lessons.len()
            || state.lesson_index == cfg.lessons.len() && state.step_in_lesson != 0
        {
            bail!("checkpoint lesson cursor is out of range");
        }
        if state.lesson_index < cfg.lessons.len()
            && state.step_in_lesson >= lesson_steps[state.lesson_index]
        {
            bail!("checkpoint step_in_lesson exceeds lesson budget");
        }
        let expected_step =
            global_step_from_cursor(&lesson_steps, state.lesson_index, state.step_in_lesson);
        if state.global_step != expected_step as u64 {
            bail!(
                "checkpoint global step {} disagrees with lesson cursor {}",
                state.global_step,
                expected_step
            );
        }
        if state.completed_lessons.len() != state.lesson_index {
            bail!(
                "checkpoint has {} completed lesson reports at lesson index {}",
                state.completed_lessons.len(),
                state.lesson_index
            );
        }
        for (index, report) in state.completed_lessons.iter().enumerate() {
            let lesson = &cfg.lessons[index];
            if report.lesson != *lesson
                || report.curriculum != lesson_to_curriculum(lesson)?
                || report.steps != lesson_steps[index]
            {
                bail!("checkpoint completed lesson report {index} is inconsistent");
            }
        }
    }
    let model_path = bundle.join("model.safetensors");
    let optimizer_path = bundle.join("optimizer.safetensors");
    load_varmap_exact(varmap, &model_path)?;
    optimizer.load(&optimizer_path, state.optimizer_step)?;
    match (state.foundation_v2.is_some(), ema) {
        (true, Some(ema)) => load_varmap_exact(ema.weights(), &bundle.join("ema.safetensors"))?,
        (true, None) => bail!("foundation-v2 resume requires EMA state"),
        (false, Some(_)) => bail!("legacy checkpoint unexpectedly requested EMA state"),
        (false, None) => {}
    }
    Ok(state)
}

fn copy_checkpoint_bundle(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).with_context(|| format!("create {}", destination.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            bail!("checkpoint bundle contains a non-file artifact");
        }
        let copied = destination.join(entry.file_name());
        fs::copy(entry.path(), &copied).with_context(|| {
            format!(
                "copy checkpoint artifact {} -> {}",
                entry.path().display(),
                destination.display()
            )
        })?;
        File::open(&copied)
            .with_context(|| format!("open copied artifact {} for sync", copied.display()))?
            .sync_all()
            .with_context(|| format!("sync copied artifact {}", copied.display()))?;
    }
    File::open(destination)
        .with_context(|| format!("open copied bundle {} for sync", destination.display()))?
        .sync_all()
        .with_context(|| format!("sync copied bundle {}", destination.display()))?;
    verify_checkpoint_bundle(destination)
}

fn publish_permanent_checkpoint(cfg: &TrainConfig, checkpoint: &Path) -> Result<PathBuf> {
    verify_checkpoint_bundle(checkpoint)?;
    let source_step = checkpoint_step(checkpoint)?;
    let parent = cfg.output_dir.join("checkpoints/permanent");
    fs::create_dir_all(&parent).with_context(|| format!("create {}", parent.display()))?;
    let name = checkpoint
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("checkpoint path has no directory name"))?;
    let destination = parent.join(name);
    if destination.exists() {
        let existing = verify_checkpoint_bundle(&destination).and_then(|()| {
            let destination_step = checkpoint_step(&destination)?;
            if destination_step != source_step {
                bail!(
                    "permanent checkpoint {} holds step {destination_step}, expected {source_step}",
                    destination.display()
                );
            }
            Ok(())
        });
        match existing {
            Ok(()) => return Ok(destination),
            Err(error) => {
                let backup = unused_checkpoint_sibling(
                    &parent,
                    &format!("{}.corrupt", name.to_string_lossy()),
                );
                fs::rename(&destination, &backup).with_context(|| {
                    format!(
                        "preserve corrupt permanent checkpoint {} -> {}",
                        destination.display(),
                        backup.display()
                    )
                })?;
                File::open(&parent)?.sync_all()?;
                tracing::warn!(
                    "preserved unverifiable permanent checkpoint {} at {} before replacement: {error:#}",
                    destination.display(),
                    backup.display()
                );
            }
        }
    }
    let staging = unused_checkpoint_sibling(&parent, &format!("{}.tmp", name.to_string_lossy()));
    copy_checkpoint_bundle(checkpoint, &staging)?;
    fs::rename(&staging, &destination).with_context(|| {
        format!(
            "publish permanent checkpoint {} -> {}",
            staging.display(),
            destination.display()
        )
    })?;
    File::open(&parent)?.sync_all()?;
    verify_checkpoint_bundle(&destination)?;
    Ok(destination)
}

fn checkpoint_step(bundle: &Path) -> Result<u64> {
    Ok(read_json::<BundleGlobalStep>(&bundle.join("trainer_state.json"))?.global_step)
}

fn checkpoint_step_directory(cfg: &TrainConfig, step: u64) -> PathBuf {
    cfg.output_dir
        .join("checkpoints")
        .join(format!("step-{step:012}"))
}

fn recorded_permanent_destination(cfg: &TrainConfig, recorded: &Path) -> Result<PathBuf> {
    let name = recorded
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!("recorded permanent checkpoint has no UTF-8 directory name")
        })?;
    let step = name
        .strip_prefix("step-")
        .and_then(|step| (step.len() == 12).then_some(step))
        .and_then(|step| step.parse::<u64>().ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "recorded permanent checkpoint has invalid step directory {}",
                recorded.display()
            )
        })?;
    Ok(cfg
        .output_dir
        .join("checkpoints/permanent")
        .join(format!("step-{step:012}")))
}

fn publish_best_checkpoint(cfg: &TrainConfig, checkpoint: &Path) -> Result<PathBuf> {
    let checkpoints = cfg.output_dir.join("checkpoints");
    let best = checkpoints.join("best");
    let staging = unused_checkpoint_sibling(&checkpoints, "best.tmp");
    copy_checkpoint_bundle(checkpoint, &staging)?;
    if best.exists() {
        let old = unused_checkpoint_sibling(&checkpoints, "best.old");
        fs::rename(&best, &old)
            .with_context(|| format!("rotate {} -> {}", best.display(), old.display()))?;
        File::open(&checkpoints)?.sync_all()?;
        fs::rename(&staging, &best)
            .with_context(|| format!("publish {} -> {}", staging.display(), best.display()))?;
        File::open(&checkpoints)?.sync_all()?;
        fs::remove_dir_all(&old).with_context(|| format!("remove replaced {}", old.display()))?;
        File::open(&checkpoints)?.sync_all()?;
    } else {
        fs::rename(&staging, &best)
            .with_context(|| format!("publish {} -> {}", staging.display(), best.display()))?;
        File::open(&checkpoints)?.sync_all()?;
    }
    verify_checkpoint_bundle(&best)?;
    Ok(best)
}

fn reconcile_foundation_v2_best_checkpoint(
    cfg: &TrainConfig,
    foundation: &FoundationV2TrainerState,
) -> Result<()> {
    let Some(expected_step) =
        foundation_v2_selected_best_step(cfg.promotion_metric, &foundation.gate_history)
    else {
        return Ok(());
    };
    let best = cfg.output_dir.join("checkpoints/best");
    let matches = verify_checkpoint_bundle(&best)
        .and_then(|()| checkpoint_step(&best))
        .is_ok_and(|step| step == expected_step);
    if matches {
        return Ok(());
    }
    let source = checkpoint_step_directory(cfg, expected_step);
    // Rolling step checkpoints are pruned; the permanent mirror keeps every
    // FOUNDATION_V2_PERMANENT_EVERY bundle, so a late re-selection (e.g.
    // after a promotion-rule correction) must be able to source from it.
    let source = if verify_checkpoint_bundle(&source).is_ok() {
        source
    } else {
        cfg.output_dir
            .join("checkpoints/permanent")
            .join(format!("step-{expected_step:012}"))
    };
    verify_checkpoint_bundle(&source).with_context(|| {
        format!(
            "cannot reconcile selected best step {expected_step}: source bundle {} is unavailable or corrupt",
            source.display()
        )
    })?;
    if checkpoint_step(&source)? != expected_step {
        bail!(
            "selected best source {} does not hold step {expected_step}",
            source.display()
        );
    }
    tracing::warn!(
        "reconciling checkpoints/best to gate-history-selected step {expected_step} from {}",
        source.display()
    );
    publish_best_checkpoint(cfg, &source)?;
    Ok(())
}

fn reconcile_foundation_v2_permanent_checkpoints(
    cfg: &TrainConfig,
    permanent_checkpoints: &mut [PathBuf],
) -> Result<()> {
    for recorded in permanent_checkpoints {
        let destination = recorded_permanent_destination(cfg, recorded)?;
        let name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .expect("validated permanent checkpoint name");
        let step = name
            .strip_prefix("step-")
            .expect("validated permanent checkpoint prefix")
            .parse::<u64>()?;
        let source = checkpoint_step_directory(cfg, step);
        let needs_copy = match verify_checkpoint_bundle(&destination)
            .and_then(|()| checkpoint_step(&destination))
        {
            Ok(published_step) => published_step != step,
            Err(_) => true,
        };
        if needs_copy {
            tracing::warn!(
                "reconciling recorded permanent checkpoint step {step} from {}",
                source.display()
            );
            publish_permanent_checkpoint(cfg, &source)?;
        }
        *recorded = destination;
    }
    Ok(())
}

fn ensure_foundation_v2_resume_not_aborted(
    cfg: &TrainConfig,
    foundation: Option<&FoundationV2TrainerState>,
) -> Result<()> {
    if cfg.resume_after_abort {
        return Ok(());
    }
    let marker = cfg.output_dir.join(FOUNDATION_V2_ABORT_MARKER);
    if marker.is_file() {
        bail!(
            "foundation-v2 run is durably aborted at {}; pass --resume-after-abort to continue explicitly",
            marker.display()
        );
    }
    if foundation.is_some_and(|state| foundation_v2_gate_history_aborts(&state.gate_history)) {
        bail!(
            "restored foundation-v2 gate history records an abort; pass --resume-after-abort to continue explicitly"
        );
    }
    Ok(())
}

fn persist_foundation_v2_abort_marker(
    cfg: &TrainConfig,
    global_step: u64,
    checkpoint: &Path,
) -> Result<()> {
    write_json_atomic(
        &cfg.output_dir.join(FOUNDATION_V2_ABORT_MARKER),
        &FoundationV2AbortMarker {
            schema: "p2.foundation_v2_abort.v1".into(),
            global_step,
            checkpoint: checkpoint.to_path_buf(),
        },
    )
}

fn seal_foundation_v2_abort(
    cfg: &TrainConfig,
    checkpoint: &Path,
    report: &TrainReport,
) -> Result<PathBuf> {
    let directory = cfg
        .output_dir
        .join("diagnostics")
        .join(format!("abort-step-{:012}", report.global_step));
    if directory.exists() {
        bail!(
            "refusing to overwrite diagnostic bundle {}",
            directory.display()
        );
    }
    fs::create_dir_all(&directory)?;
    let checkpoint_copy = directory.join("checkpoint");
    copy_checkpoint_bundle(checkpoint, &checkpoint_copy)?;
    let report_path = directory.join("train_report.json");
    let gate_history_path = directory.join("gate_history.json");
    write_json_atomic(&report_path, report)?;
    write_json_atomic(
        &gate_history_path,
        &report
            .foundation_v2
            .as_ref()
            .expect("foundation-v2 abort report")
            .gate_history,
    )?;
    let mut sealed = BTreeMap::new();
    for path in [
        &report_path,
        &gate_history_path,
        &checkpoint_copy.join("bundle-manifest.json"),
    ] {
        let digest = sha256_file(path)?;
        sealed.insert(
            path.strip_prefix(&directory)?.display().to_string(),
            digest.sha256,
        );
    }
    write_json_atomic(&directory.join("diagnostic-manifest.json"), &sealed)?;
    Ok(directory)
}

fn loss_means(sums: &LessonLossMeans, count: usize) -> LessonLossMeans {
    let n = count as f64;
    LessonLossMeans {
        total: sums.total / n,
        next_latent: sums.next_latent / n,
        rollout: sums.rollout / n,
        sigreg_raw: sums.sigreg_raw / n,
        sigreg_bounded: sums.sigreg_bounded / n,
        patch_grounding: sums.patch_grounding / n,
        grounding_changed_patches: sums.grounding_changed_patches / n,
        grounding_unchanged_patches: sums.grounding_unchanged_patches / n,
        pre_clip_gradient_norm: sums.pre_clip_gradient_norm / n,
        gradient_clip_scale: sums.gradient_clip_scale / n,
        clipped_updates: sums.clipped_updates / n,
        event: sums.event / n,
        q: sums.q / n,
        prefix: sums.prefix / n,
        reliability: sums.reliability / n,
        branch_total: sums.branch_total / n,
        outcome_pull: sums.outcome_pull / n,
        outcome_push: sums.outcome_push / n,
        action_recovery: sums.action_recovery / n,
        coordinate_recovery: sums.coordinate_recovery / n,
        changed_margin: sums.changed_margin / n,
        spatial_variance: sums.spatial_variance / n,
        spatial_covariance: sums.spatial_covariance / n,
        pooled_variance: sums.pooled_variance / n,
        pooled_covariance: sums.pooled_covariance / n,
        displacement_variance: sums.displacement_variance / n,
        displacement_covariance: sums.displacement_covariance / n,
        branch_groups: sums.branch_groups / n,
        changed_branches: sums.changed_branches / n,
        equivalent_pairs: sums.equivalent_pairs / n,
        distinct_pairs: sums.distinct_pairs / n,
        action6_branches: sums.action6_branches / n,
        action_recovery_branches: sums.action_recovery_branches / n,
        spatial_population_rows: sums.spatial_population_rows / n,
        pooled_population_rows: sums.pooled_population_rows / n,
        displacement_population_rows: sums.displacement_population_rows / n,
        unique_changed_outcomes: sums.unique_changed_outcomes / n,
    }
}

fn foundation_v2_loss_means(sums: &FoundationV2LossMeans, count: u64) -> FoundationV2LossMeans {
    if count == 0 {
        return FoundationV2LossMeans::default();
    }
    let n = count as f64;
    FoundationV2LossMeans {
        total: sums.total / n,
        pred_ce: sums.pred_ce / n,
        gate: sums.gate / n,
        latent: sums.latent / n,
        enc_ce: sums.enc_ce / n,
        separation: sums.separation / n,
        pull: sums.pull / n,
        inverse_action: sums.inverse_action / n,
        ep: sums.ep / n,
        rollout: sums.rollout / n,
        event: sums.event / n,
        q: sums.q / n,
        reliability: sums.reliability / n,
        pre_clip_gradient_norm: sums.pre_clip_gradient_norm / n,
        gradient_clip_scale: sums.gradient_clip_scale / n,
        clipped_fraction: sums.clipped_fraction / n,
    }
}

fn build_report(
    cfg: &TrainConfig,
    state: &TrainerState,
    status: TrainStatus,
    parameter_count: usize,
    latest_checkpoint: PathBuf,
    resumed_from: Option<PathBuf>,
    run_attempts: &[RunAttempt],
) -> TrainReport {
    let foundation_v2 = state.foundation_v2.as_ref().map(|foundation| {
        let best_promotion_value =
            foundation_v2_best_evaluation(cfg.promotion_metric, &foundation.gate_history).and_then(
                |evaluation| {
                    foundation_v2_promotion_value(cfg.promotion_metric, &evaluation.metrics)
                },
            );
        // Advertise the best-bundle path only when it passes the same
        // selection verification as the export claim; a foreign or stale
        // `checkpoints/best` must not be published as this run's best.
        let best_checkpoint =
            foundation_v2_verified_best_export(cfg, cfg.promotion_metric, &foundation.gate_history)
                .ok()
                .flatten()
                .map(|_| cfg.output_dir.join("checkpoints/best"));
        FoundationV2TrainingReport {
            total_steps: foundation.total_steps,
            mean_losses: foundation_v2_loss_means(&foundation.loss_sums, foundation.loss_steps),
            ep_weight: foundation.ep_weight,
            ep_gradient_budget: foundation.ep_gradient_budget.clone(),
            gate_history: foundation.gate_history.clone(),
            best_changed_exact: foundation.best_changed_exact,
            promotion_metric: cfg.promotion_metric,
            best_promotion_value,
            best_checkpoint,
            rollout_enabled: foundation.rollout_enabled,
            permanent_checkpoints: foundation.permanent_checkpoints.clone(),
            event_label_census: foundation.event_label_census,
            event_label_census_complete: foundation.event_label_census_complete,
            mechanism_history: foundation.mechanism_history.clone(),
            profile_bundles: foundation
                .profiles_published
                .iter()
                .map(|update| {
                    cfg.output_dir
                        .join("profile")
                        .join(format!("update-{update:012}"))
                })
                .collect(),
            // Documented ADR 0003 approximation: EP is first constrained by
            // its encoder-gradient controller, then the combined gradient is
            // clipped at 1.0. We do not claim a separately clipped EP store.
            clip_strategy: "adaptive EP budget, then combined global L2 clip at 1.0".into(),
        }
    });
    TrainReport {
        schema: TRAIN_REPORT_SCHEMA.into(),
        world_core_schema: cfg
            .resolved_experiment()
            .expect("validated training config resolves an experiment")
            .report_schema
            .clone(),
        experiment: cfg
            .resolved_experiment()
            .expect("validated training config resolves an experiment"),
        seed: cfg.seed,
        physical_batch: cfg.physical_batch,
        grad_accum: cfg.grad_accum,
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        parameter_count,
        training_population_fingerprint: format!("fnv1a64:{:016x}", state.training_population_hash),
        training_content_fingerprint: format!("sha256:{}", hex_bytes(&state.training_content_hash)),
        training_population_rows: state.training_population_rows,
        device: cfg.device.clone(),
        lessons: state.completed_lessons.clone(),
        status,
        global_step: state.global_step,
        latest_checkpoint,
        resumed_from,
        batch_schedule_migrations: state.batch_schedule_migrations.clone(),
        checkpoint: cfg.output_dir.join("model.safetensors"),
        export_checkpoint: export_checkpoint_path(cfg, state),
        config_path: cfg.output_dir.join("config.json"),
        profile: state.profile.clone(),
        gradient_pressure: state.gradient_pressure.clone(),
        gradient_pressure_samples: state.gradient_pressure_samples.clone(),
        foundation_v2,
        research_claim: false,
        resume_count: resume_count(run_attempts),
        run_attempts: run_attempts.to_vec(),
    }
}

fn publish_run_artifacts(varmap: &VarMap, cfg: &TrainConfig, report: &TrainReport) -> Result<()> {
    save_checkpoint(varmap, cfg, report)?;
    crate::p2::evidence::publish_training_evidence(&cfg.output_dir, report)?;
    Ok(())
}

/// Bounded skip budget for non-finite-gradient updates. Returns the abort
/// reason when the budget is exhausted: isolated data/state-specific NaNs are
/// skippable (standard loss-scaler practice), but a dense cluster of skips —
/// or NaNs on consecutive batches — indicates genuine divergence and must
/// fail closed. The short density window catches local numerical bursts while
/// preventing sparse skips from exhausting a run-lineage lifetime cap.
fn foundation_v2_nonfinite_skip_exhausted(skipped_steps: &[u64]) -> Option<String> {
    const NONFINITE_SKIP_WINDOW_STEPS: u64 = 64;
    const MAX_SKIPS_PER_WINDOW: usize = 16;
    const MAX_CONSECUTIVE_SKIPS: usize = 3;
    let tail = skipped_steps
        .iter()
        .rev()
        .take(MAX_CONSECUTIVE_SKIPS)
        .copied()
        .collect::<Vec<_>>();
    if tail.len() == MAX_CONSECUTIVE_SKIPS && tail.windows(2).all(|pair| pair[0] == pair[1] + 1) {
        return Some(format!(
            "{MAX_CONSECUTIVE_SKIPS} consecutive updates ending at step {} produced \
             non-finite gradients; this is divergence, not an isolated bad batch",
            tail[0]
        ));
    }
    if let Some(&latest_step) = skipped_steps.last() {
        let window_start = latest_step.saturating_sub(NONFINITE_SKIP_WINDOW_STEPS - 1);
        let skips_in_window = skipped_steps
            .iter()
            .rev()
            .take_while(|&&step| step >= window_start)
            .count();
        if skips_in_window >= MAX_SKIPS_PER_WINDOW {
            return Some(format!(
                "{skips_in_window} updates skipped for non-finite gradients in steps \
                 {window_start}..={latest_step} (rolling {NONFINITE_SKIP_WINDOW_STEPS}-step \
                 budget {MAX_SKIPS_PER_WINDOW})"
            ));
        }
    }
    None
}

fn foundation_v2_loss_values(
    losses: &FoundationV2LossBreakdown,
    total: &Tensor,
    pre_clip_gradient_norm: f64,
    gradient_clip_scale: f64,
) -> Result<FoundationV2LossMeans> {
    let values = ensure_all_finite(&[
        ("foundation_v2.total", &total.detach()),
        ("foundation_v2.pred_ce", &losses.pred_ce.detach()),
        ("foundation_v2.gate", &losses.gate.detach()),
        ("foundation_v2.latent", &losses.latent.detach()),
        ("foundation_v2.enc_ce", &losses.enc_ce.detach()),
        ("foundation_v2.separation", &losses.separation.detach()),
        ("foundation_v2.pull", &losses.pull.detach()),
        (
            "foundation_v2.inverse_action",
            &losses.inverse_action.detach(),
        ),
        ("foundation_v2.ep", &losses.ep.detach()),
        ("foundation_v2.rollout", &losses.rollout.detach()),
        ("foundation_v2.event", &losses.event.detach()),
        ("foundation_v2.q", &losses.q.detach()),
        ("foundation_v2.reliability", &losses.reliability.detach()),
    ])?;
    Ok(FoundationV2LossMeans {
        total: values[0] as f64,
        pred_ce: values[1] as f64,
        gate: values[2] as f64,
        latent: values[3] as f64,
        enc_ce: values[4] as f64,
        separation: values[5] as f64,
        pull: values[6] as f64,
        inverse_action: values[7] as f64,
        ep: values[8] as f64,
        rollout: values[9] as f64,
        event: values[10] as f64,
        q: values[11] as f64,
        reliability: values[12] as f64,
        pre_clip_gradient_norm,
        gradient_clip_scale,
        clipped_fraction: f64::from(gradient_clip_scale < 1.0),
    })
}

enum FoundationV2LossLogCommand {
    Append(FoundationV2LossLogEntryOwned),
    Flush(mpsc::Sender<Result<()>>),
    Shutdown(mpsc::Sender<Result<()>>),
}

#[derive(Serialize)]
struct FoundationV2LossLogEntryOwned {
    global_step: u64,
    #[serde(flatten)]
    values: FoundationV2LossMeans,
    learning_rate: f64,
}

struct FoundationV2LossLog {
    path: PathBuf,
    sender: SyncSender<FoundationV2LossLogCommand>,
    worker: Option<thread::JoinHandle<()>>,
}

/// Destination of the loss-log worker. Seam so tests can inject transient and
/// persistent I/O failures; production uses an append-mode [`File`].
trait FoundationV2LossLogSink: Write + Send + 'static {
    fn sync(&mut self) -> std::io::Result<()>;
}

impl FoundationV2LossLogSink for File {
    fn sync(&mut self) -> std::io::Result<()> {
        self.sync_data()
    }
}

/// Number of write attempts per record before the worker gives up and the
/// training thread is allowed to fail loudly.
const LOSS_LOG_WRITE_ATTEMPTS: u32 = 5;

impl FoundationV2LossLog {
    fn open(output_dir: &Path) -> Result<Self> {
        let path = output_dir.join("loss_log.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open append-only loss log {}", path.display()))?;
        Self::with_sink(file, path, true)
    }

    /// Disk writes and JSON serialization are intentionally isolated from the
    /// training thread. `flush` remains a synchronous durability gate before
    /// every checkpoint and at shutdown. `backoff` is disabled in tests.
    fn with_sink<S: FoundationV2LossLogSink>(
        mut sink: S,
        path: PathBuf,
        backoff: bool,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(256);
        let worker_path = path.clone();
        let worker = thread::Builder::new()
            .name("foundation-v2-loss-log".into())
            .spawn(move || {
                while let Ok(command) = receiver.recv() {
                    match command {
                        FoundationV2LossLogCommand::Append(entry) => {
                            // Network mounts (/workspace on RunPod) can return
                            // transient write errors; a single blip must not
                            // silently kill the writer, because the training
                            // thread then dies one step later on a closed
                            // channel. Track the byte offset so a retry
                            // resumes mid-line and never duplicates output;
                            // give up only after persistent failure, and say
                            // so on stderr because tracing has no subscriber
                            // in normal builds.
                            let mut line = match serde_json::to_vec(&entry) {
                                Ok(line) => line,
                                Err(error) => {
                                    eprintln!(
                                        "loss-log record for {} does not serialize: {error:#}",
                                        worker_path.display()
                                    );
                                    break;
                                }
                            };
                            line.push(b'\n');
                            let mut written = 0usize;
                            let mut attempt = 0u32;
                            let gave_up = loop {
                                if written == line.len() {
                                    break false;
                                }
                                match sink.write(&line[written..]) {
                                    Ok(0) => break true,
                                    Ok(count) => written += count,
                                    Err(error)
                                        if error.kind() == std::io::ErrorKind::Interrupted => {}
                                    Err(error) => {
                                        attempt += 1;
                                        if attempt >= LOSS_LOG_WRITE_ATTEMPTS {
                                            eprintln!(
                                                "loss-log writer giving up on {} after \
                                                 {attempt} attempts: {error:#}",
                                                worker_path.display()
                                            );
                                            break true;
                                        }
                                        if backoff {
                                            thread::sleep(std::time::Duration::from_millis(
                                                200 << attempt.min(4),
                                            ));
                                        }
                                    }
                                }
                            };
                            if gave_up {
                                break;
                            }
                        }
                        FoundationV2LossLogCommand::Flush(reply) => {
                            let result =
                                sink.flush().and_then(|_| sink.sync()).with_context(|| {
                                    format!("flush loss log {}", worker_path.display())
                                });
                            let stop = result.is_err();
                            let _ = reply.send(result);
                            if stop {
                                break;
                            }
                        }
                        FoundationV2LossLogCommand::Shutdown(reply) => {
                            let result =
                                sink.flush().and_then(|_| sink.sync()).with_context(|| {
                                    format!("flush loss log {}", worker_path.display())
                                });
                            let _ = reply.send(result);
                            break;
                        }
                    }
                }
            })
            .context("spawn foundation-v2 loss-log worker")?;
        Ok(Self {
            path,
            sender,
            worker: Some(worker),
        })
    }

    fn append(
        &mut self,
        global_step: u64,
        values: &FoundationV2LossMeans,
        learning_rate: f64,
    ) -> Result<()> {
        let entry = FoundationV2LossLogEntryOwned {
            global_step,
            values: values.clone(),
            learning_rate,
        };
        self.sender
            .send(FoundationV2LossLogCommand::Append(entry))
            .with_context(|| format!("queue loss record for {}", self.path.display()))
    }

    fn flush(&mut self) -> Result<()> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender
            .send(FoundationV2LossLogCommand::Flush(reply_tx))
            .with_context(|| format!("request loss-log flush for {}", self.path.display()))?;
        reply_rx
            .recv()
            .with_context(|| format!("await loss-log flush for {}", self.path.display()))?
    }
}

impl Drop for FoundationV2LossLog {
    fn drop(&mut self) {
        let result = if let Some(worker) = self.worker.take() {
            let (reply_tx, reply_rx) = mpsc::channel();
            let result = self
                .sender
                .send(FoundationV2LossLogCommand::Shutdown(reply_tx))
                .with_context(|| format!("request loss-log shutdown for {}", self.path.display()))
                .and_then(|_| {
                    reply_rx.recv().with_context(|| {
                        format!("await loss-log shutdown for {}", self.path.display())
                    })?
                });
            let _ = worker.join();
            result
        } else {
            Ok(())
        };
        if let Err(error) = result {
            tracing::error!(
                "failed to flush foundation-v2 loss log {} on drop: {error:#}",
                self.path.display()
            );
        }
    }
}

fn add_foundation_v2_loss_sums(sums: &mut FoundationV2LossMeans, values: &FoundationV2LossMeans) {
    sums.total += values.total;
    sums.pred_ce += values.pred_ce;
    sums.gate += values.gate;
    sums.latent += values.latent;
    sums.enc_ce += values.enc_ce;
    sums.separation += values.separation;
    sums.pull += values.pull;
    sums.inverse_action += values.inverse_action;
    sums.ep += values.ep;
    sums.rollout += values.rollout;
    sums.event += values.event;
    sums.q += values.q;
    sums.reliability += values.reliability;
    sums.pre_clip_gradient_norm += values.pre_clip_gradient_norm;
    sums.gradient_clip_scale += values.gradient_clip_scale;
    sums.clipped_fraction += values.clipped_fraction;
}

fn foundation_v2_active_loss_means(sums: &FoundationV2LossMeans, count: u64) -> LessonLossMeans {
    let means = foundation_v2_loss_means(sums, count);
    LessonLossMeans {
        total: means.total,
        rollout: means.rollout,
        pre_clip_gradient_norm: means.pre_clip_gradient_norm,
        gradient_clip_scale: means.gradient_clip_scale,
        event: means.event,
        q: means.q,
        reliability: means.reliability,
        ..LessonLossMeans::default()
    }
}

fn update_foundation_v2_rollout_zero_streak(
    consecutive_steps: &mut u64,
    rollout_enabled: bool,
    rollout_loss: f64,
    global_step: u64,
) {
    if rollout_enabled && rollout_loss == 0.0 {
        *consecutive_steps = consecutive_steps.saturating_add(1);
        if *consecutive_steps == FOUNDATION_V2_GATE_EVERY {
            tracing::warn!(
                "foundation-v2 realized rollout loss has remained exactly zero for {} \
                 consecutive rollout-enabled updates through step {global_step}",
                FOUNDATION_V2_GATE_EVERY
            );
        }
    } else {
        *consecutive_steps = 0;
    }
}

fn validate_foundation_v2_profile_resume(
    profile_updates: &[u64],
    profiles_published: &mut Vec<u64>,
    global_step: u64,
    output_dir: &Path,
    device: &str,
) -> Result<()> {
    let recorded = profiles_published.iter().copied().collect::<BTreeSet<_>>();
    profiles_published.clear();
    for &update in profile_updates {
        if let Some(artifacts) = reconcile_profile_bundle(output_dir, update, device)? {
            profiles_published.push(update);
            if !recorded.contains(&update) {
                tracing::warn!(
                    "reconciled foundation-v2 profile update {update} from verified bundle {}",
                    artifacts.directory.display()
                );
            }
        }
    }
    profiles_published.sort_unstable();
    profiles_published.dedup();
    let missing = profile_updates
        .iter()
        .copied()
        .filter(|update| *update <= global_step && !profiles_published.contains(update))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "foundation-v2 resume is missing preregistered profile evidence for completed updates {:?}",
            missing
        );
    }
    Ok(())
}

fn record_foundation_v2_profile_tensors(
    profile: &RepresentativeUpdateCapture,
    range: &ProfileRange<'_>,
    losses: &FoundationV2LossBreakdown,
    effective_total: &Tensor,
) -> Result<()> {
    for (name, tensor) in [
        ("total", effective_total),
        ("non_ep_total", &losses.non_ep_total),
        ("pred_ce", &losses.pred_ce),
        ("gate", &losses.gate),
        ("latent", &losses.latent),
        ("enc_ce", &losses.enc_ce),
        ("separation", &losses.separation),
        ("pull", &losses.pull),
        ("inverse_action", &losses.inverse_action),
        ("ep", &losses.ep),
        ("rollout", &losses.rollout),
        ("event", &losses.event),
        ("q", &losses.q),
        ("reliability", &losses.reliability),
    ] {
        let label = format!("loss/{name}");
        profile.record_tensor(range, &label, tensor, Some(ExecutionStep::Forward))?;
    }
    let seams = losses
        .mechanism_seams
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("profiled foundation-v2 loss omitted mechanism seams"))?;
    for (label, tensor) in [
        ("seam/out_y", &seams.out_y),
        ("seam/current_canonical", &seams.current_canonical),
        ("seam/predicted_canonical", &seams.predicted_canonical),
        ("seam/gate_logits", &seams.gate_logits),
    ] {
        profile.record_tensor(range, label, tensor, Some(ExecutionStep::Forward))?;
        profile.record_tensor_stats(range, label, tensor)?;
    }
    Ok(())
}

fn record_foundation_v2_profile_scalars(
    profile: &RepresentativeUpdateCapture,
    span_id: Option<SpanId>,
    values: &FoundationV2LossMeans,
    learning_rate: f64,
    ep_weight: f64,
) -> Result<()> {
    for (label, value) in [
        ("loss/total", values.total),
        ("loss/pred_ce", values.pred_ce),
        ("loss/gate", values.gate),
        ("loss/latent", values.latent),
        ("loss/enc_ce", values.enc_ce),
        ("loss/separation", values.separation),
        ("loss/pull", values.pull),
        ("loss/inverse_action", values.inverse_action),
        ("loss/ep", values.ep),
        ("loss/rollout", values.rollout),
        ("loss/event", values.event),
        ("loss/q", values.q),
        ("loss/reliability", values.reliability),
        (
            "optimizer/pre_clip_gradient_norm",
            values.pre_clip_gradient_norm,
        ),
        ("optimizer/clip_scale", values.gradient_clip_scale),
        ("optimizer/clipped", values.clipped_fraction),
        ("optimizer/learning_rate", learning_rate),
        ("objective/ep_weight", ep_weight),
    ] {
        profile.record_scalar(span_id, label, value)?;
    }
    Ok(())
}

fn record_training_profile_scalars(
    profile: &RepresentativeUpdateCapture,
    span_id: Option<SpanId>,
    values: &LessonLossMeans,
    learning_rate: f64,
) -> Result<()> {
    for (label, value) in [
        ("loss/total", values.total),
        ("loss/next_latent", values.next_latent),
        ("loss/rollout", values.rollout),
        ("loss/sigreg_raw", values.sigreg_raw),
        ("loss/sigreg_bounded", values.sigreg_bounded),
        ("loss/patch_grounding", values.patch_grounding),
        ("loss/event", values.event),
        ("loss/q", values.q),
        ("loss/prefix", values.prefix),
        ("loss/reliability", values.reliability),
        ("loss/branch_total", values.branch_total),
        ("loss/outcome_pull", values.outcome_pull),
        ("loss/outcome_push", values.outcome_push),
        ("loss/action_recovery", values.action_recovery),
        ("loss/coordinate_recovery", values.coordinate_recovery),
        ("loss/changed_margin", values.changed_margin),
        ("loss/spatial_variance", values.spatial_variance),
        ("loss/spatial_covariance", values.spatial_covariance),
        ("loss/pooled_variance", values.pooled_variance),
        ("loss/pooled_covariance", values.pooled_covariance),
        ("loss/displacement_variance", values.displacement_variance),
        (
            "loss/displacement_covariance",
            values.displacement_covariance,
        ),
        (
            "optimizer/pre_clip_gradient_norm",
            values.pre_clip_gradient_norm,
        ),
        ("optimizer/clip_scale", values.gradient_clip_scale),
        ("optimizer/clipped", values.clipped_updates),
        ("optimizer/learning_rate", learning_rate),
    ] {
        profile.record_scalar(span_id, label, value)?;
    }
    Ok(())
}

fn foundation_v2_mechanism_sample(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    step: u64,
) -> Result<FoundationV2MechanismSample> {
    if samples.is_empty() {
        bail!("foundation-v2 mechanism sample requires at least one gate row");
    }
    let rows = &samples[..samples.len().min(128)];
    let batch = batch_from_samples(rows, device)?;
    let current = model.encode_state(&batch.frames)?;
    let out = model.forward_from_encoded_state_with_operator_conditioning(
        &current,
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        &batch.operator_conditioning,
        RecursionDepth::from_config(model.config()),
        0.0,
        None,
        RecursionOpts {
            record_probes: true,
            store_intermediate_steps: false,
        },
    )?;
    let gate = model
        .exact_copy_gate(&out.y)?
        .detach()
        .to_dtype(DType::F32)?;
    let gate_mean_probability = gate.mean_all()?.to_scalar::<f32>()? as f64;
    let gate_open_rate = gate
        .ge(0.5)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()? as f64;
    if !gate_mean_probability.is_finite() || !gate_open_rate.is_finite() {
        bail!("foundation-v2 mechanism sample produced non-finite gate statistics");
    }
    Ok(FoundationV2MechanismSample {
        step,
        copy_bypass_alpha: model.copy_bypass_alpha()?,
        outer_step_cosines: out
            .recursion_probes
            .iter()
            .map(|probe| probe.mean_step_cosine)
            .collect(),
        gate_open_rate,
        gate_mean_probability,
    })
}

fn train_foundation_v2(requested_cfg: &TrainConfig) -> Result<TrainReport> {
    let mut cfg = requested_cfg.clone();
    if cfg.resume.is_none() && implicit_resume_source(&cfg).is_some() {
        cfg.resume = Some(cfg.output_dir.join("checkpoints"));
        tracing::info!(
            "auto-resuming foundation-v2 from {}",
            cfg.output_dir.join("checkpoints").display()
        );
    }
    cfg.validate()?;
    fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("create {}", cfg.output_dir.display()))?;
    ensure_foundation_v2_resume_not_aborted(&cfg, None)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&cfg.output_dir)?)
    } else {
        None
    };
    let _train_pid = TrainPidGuard::install(&cfg.output_dir)?;
    PAUSE_REQUESTED.store(false, Ordering::SeqCst);
    install_pause_handler()?;
    let device = resolve_device(&cfg.device)?;
    let model_cfg = cfg.model_config();
    model_cfg.validate()?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = WorldModel::new(model_cfg, vb)?;
    let mut optimizer = CheckpointHybridOptimizer::new(
        &varmap,
        adam_params(&cfg),
        cfg.muon_momentum,
        cfg.muon_rms_scale,
    )?;
    let parameter_names = optimizer.parameter_names();
    let parameter_count = parameter_count(&varmap);
    let resume_source = cfg.resume.clone().or_else(|| implicit_resume_source(&cfg));
    let resumed_from = resume_source
        .as_deref()
        .map(resolve_resume_checkpoint)
        .transpose()?;
    if resumed_from.is_none() {
        reinit_varmap_deterministic(&varmap, cfg.init_seed.unwrap_or(cfg.seed))?;
        // FiLM identity, copy-bypass, and gate-bias-prior initialization
        // are restored exactly once after fresh deterministic init (the
        // reinitializer zeroes every bias, which would silently turn the
        // configured prior into a 50/50 gate). Checkpoint loads must retain
        // the learned values.
        zero_action_film_projections(&varmap)?;
        zero_operator_conditioning_projection(&varmap)?;
        zero_context_film_projections(&varmap)?;
        init_copy_bypass_gate(&varmap)?;
        restore_copy_gate_bias_prior(&varmap, cfg.copy_gate_bias_prior)?;
        if let Some(source) = &cfg.init_context_from_v5 {
            // ADR 0005 §3.4 warm start: every v5 tensor by name; context
            // parameters keep the fresh init above (FiLM zero => exact v5).
            load_varmap_warm_start_context(&varmap, source)?;
            tracing::info!(
                "initialized v6 non-context parameters from {}",
                source.display()
            );
        }
    }
    let mut ema = ModelEma::with_default_decay(&varmap)?;
    let mut state = if let Some(bundle) = &resumed_from {
        load_training_checkpoint(bundle, &cfg, &mut varmap, &mut optimizer, Some(&mut ema))?
    } else {
        TrainerState {
            schema: TRAINER_STATE_SCHEMA.into(),
            contract: TrainingContract::from(&cfg),
            global_step: 0,
            lesson_index: 0,
            step_in_lesson: 0,
            optimizer_step: 0,
            completed_lessons: Vec::new(),
            active_sums: LessonLossMeans::default(),
            parameter_names,
            training_population_hash: default_training_population_hash(),
            training_content_hash: [0; 32],
            training_population_rows: 0,
            batch_schedule_migrations: Vec::new(),
            nonfinite_skipped_updates: Vec::new(),
            profile: ProfileState::Pending,
            gradient_pressure: None,
            gradient_pressure_samples: Vec::new(),
            foundation_v2: Some(FoundationV2TrainerState {
                total_steps: cfg.steps_per_lesson,
                ep_weight: 0.01,
                ep_gradient_budget: Vec::new(),
                gate_history: Vec::new(),
                best_changed_exact: None,
                rollout_enabled: true,
                loss_sums: FoundationV2LossMeans::default(),
                loss_steps: 0,
                permanent_checkpoints: Vec::new(),
                event_label_census: EventLabelCensus::default(),
                event_label_census_complete: true,
                mechanism_history: Vec::new(),
                profiles_published: Vec::new(),
                rollout_zero_loss_consecutive_steps: 0,
                gate_population_identity: None,
            }),
        }
    };
    ensure_profile_campaign_manifest(&cfg, &cfg.profile_updates)?;
    if resumed_from.is_some() {
        ensure_foundation_v2_resume_not_aborted(&cfg, state.foundation_v2.as_ref())?;
    }
    // The durable start event precedes every possible loss-log mutation. The
    // TrainPidGuard above owns this output root while start/repair/result are
    // appended, so two writers cannot select the same attempt number.
    let attempt = begin_run_attempt(
        &cfg.output_dir,
        resumed_from.as_deref(),
        resumed_from.as_ref().map(|_| state.global_step),
    )?;
    let repair_result = if resumed_from.is_some() {
        repair_loss_log_for_resume(&cfg.output_dir, state.global_step, attempt)
    } else {
        Ok(None)
    };
    let loss_log_repair = match repair_result {
        Ok(repair) => repair,
        Err(error) => {
            fail_run_attempt_repair(&cfg.output_dir, attempt, &error)
                .context("journal failed loss-log repair")?;
            return Err(error);
        }
    };
    let run_attempts = complete_run_attempt_repair(&cfg.output_dir, attempt, loss_log_repair)?;
    let mut loss_log = FoundationV2LossLog::open(&cfg.output_dir)?;
    let mut latest_checkpoint = if resumed_from.is_none() {
        loss_log.flush()?;
        Some(save_training_checkpoint(
            &varmap,
            &optimizer,
            Some(&ema),
            &state,
            &cfg,
        )?)
    } else {
        resumed_from.clone()
    };
    let stream_schedule = if cfg.data_contract_v6 {
        adaptation_v6_stream_schedule
    } else {
        foundation_v2_stream_schedule
    };
    let gate_batch = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size: FOUNDATION_V2_GATE_ROWS,
            seed: FOUNDATION_V2_GATE_SEED,
            schedule: stream_schedule,
            data_contract_v6: cfg.data_contract_v6,
            ..MixedStreamConfig::default()
        },
        1.0,
        0,
        V5DataSplit::UnseenSeed7x7,
    )?;
    let gate_samples = gate_batch.transitions().cloned().collect::<Vec<_>>();
    let gate_content_masks = gate_batch
        .samples()
        .iter()
        .map(|sample| sample.content_mask.clone())
        .collect::<Vec<_>>();
    let gate_provenance = gate_batch
        .samples()
        .iter()
        .map(|sample| sample.provenance.clone())
        .collect::<Vec<V5SampleProvenance>>();
    {
        let identity = foundation_v2_gate_population_identity(
            &gate_samples,
            &gate_content_masks,
            &gate_provenance,
        )?;
        let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
        reconcile_foundation_v2_gate_population_identity(foundation, identity)?;
    }
    if resumed_from.is_some() {
        let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
        reconcile_foundation_v2_best_checkpoint(&cfg, foundation)?;
        reconcile_foundation_v2_permanent_checkpoints(&cfg, &mut foundation.permanent_checkpoints)?;
    }
    validate_foundation_v2_profile_resume(
        &cfg.profile_updates,
        &mut state
            .foundation_v2
            .as_mut()
            .expect("foundation-v2 state")
            .profiles_published,
        state.global_step,
        &cfg.output_dir,
        &cfg.device,
    )?;
    let stream_config = MixedStreamConfig {
        batch_size: cfg.physical_batch,
        seed: cfg.seed,
        schedule: stream_schedule,
        data_contract_v6: cfg.data_contract_v6,
        ..MixedStreamConfig::default()
    };
    let mut updates_this_run = 0usize;
    let total_steps = state
        .foundation_v2
        .as_ref()
        .expect("foundation-v2 state")
        .total_steps;
    let event_slot_weights = event_slot_weight_tensor(&device)?;
    let mut data_prefetcher = if cfg.prefetch_batches {
        Some(MixedStreamBatchPrefetcher::new(
            stream_config.clone(),
            total_steps,
            state.global_step,
            cfg.data_workers,
        )?)
    } else {
        stream_config.validate()?;
        None
    };

    loop {
        let complete = state.global_step >= total_steps as u64;
        if complete || PAUSE_REQUESTED.load(Ordering::SeqCst) {
            if let Some(prefetcher) = data_prefetcher.as_mut() {
                prefetcher.shutdown();
            }
            if latest_checkpoint
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                != Some(&format!("step-{:012}", state.global_step))
            {
                loss_log.flush()?;
                latest_checkpoint = Some(save_training_checkpoint(
                    &varmap,
                    &optimizer,
                    Some(&ema),
                    &state,
                    &cfg,
                )?);
            }
            let report = build_report(
                &cfg,
                &state,
                if complete {
                    TrainStatus::Completed
                } else {
                    TrainStatus::Paused
                },
                parameter_count,
                latest_checkpoint.expect("foundation-v2 writes a final checkpoint"),
                resumed_from,
                &run_attempts,
            );
            publish_run_artifacts(&varmap, &cfg, &report)?;
            sync_cuda_device(&device)?;
            return Ok(report);
        }

        let next_step = state.global_step + 1;
        let pending_profile = ProfileState::Pending;
        let cg_profile = if cfg.profile_updates.contains(&next_step)
            && !state
                .foundation_v2
                .as_ref()
                .expect("foundation-v2 state")
                .profiles_published
                .contains(&next_step)
        {
            Some(RepresentativeUpdateCapture::begin(CaptureSpec {
                completed_updates: state.global_step,
                selected_update: next_step,
                state: &pending_profile,
                output_dir: &cfg.output_dir,
                device: &cfg.device,
                measured_region_device_synchronized: device.is_cuda(),
                lesson: "foundation_v2",
                physical_batch: cfg.physical_batch,
                grad_accum: cfg.grad_accum,
                hidden_dim: cfg.hidden_dim,
                inner_steps: cfg.inner_steps,
                outer_steps: cfg.outer_steps,
                precision: if cfg.bf16_recurrent_core {
                    "bf16-recurrent-core/f32-rest"
                } else if cfg.bf16_conv {
                    "bf16-conv/f32-rest"
                } else {
                    "f32"
                },
                varmap: &varmap,
                gradient_clip_state: GradientClipState::PreClip,
            })?)
        } else {
            None
        };
        if cg_profile.as_ref().is_some_and(|profile| profile.active()) {
            sync_cuda_device(&device)?;
        }
        let profile_measurement = cg_profile
            .as_ref()
            .and_then(RepresentativeUpdateCapture::measurement);
        let profile_measurement_span = profile_measurement.as_ref().and_then(ProfileRange::span_id);
        let (mixed, host, content_batch_digest) = if let Some(prefetcher) = data_prefetcher.as_mut()
        {
            let (batch_index, prepared) = if let Some(profile) = &cg_profile {
                profile.synchronized_phase(
                    &device,
                    "prefetched_receive",
                    SpanKind::Module,
                    None,
                    || prefetcher.recv_next(),
                )?
            } else {
                prefetcher.recv_next()?
            };
            if batch_index != state.global_step {
                bail!(
                    "foundation-v2 prefetch returned batch {batch_index} while step {} was required",
                    state.global_step
                );
            }
            prepared.into_parts()
        } else {
            let progress = state.global_step as f32 / total_steps.max(1) as f32;
            if let Some(profile) = &cg_profile {
                profile.synchronized_phase(&device, "generation", SpanKind::Module, None, || {
                    let batch = compose_mixed_stream_batch(
                        &stream_config,
                        progress,
                        state.global_step,
                        V5DataSplit::Train,
                    )?;
                    let digest =
                        training_content_batch_digest(batch.transitions(), batch.content_masks())?;
                    let host = prepare_foundation_v2_batch_host(&batch)?;
                    Ok((batch, host, digest))
                })?
            } else {
                let batch = compose_mixed_stream_batch(
                    &stream_config,
                    progress,
                    state.global_step,
                    V5DataSplit::Train,
                )?;
                let digest =
                    training_content_batch_digest(batch.transitions(), batch.content_masks())?;
                let host = prepare_foundation_v2_batch_host(&batch)?;
                (batch, host, digest)
            }
        };
        let batch_event_census = mixed.event_label_census();
        {
            let total = &mut state
                .foundation_v2
                .as_mut()
                .expect("foundation-v2 state")
                .event_label_census;
            total.rows += batch_event_census.rows;
            for slot in 0..4 {
                total.labeled[slot] += batch_event_census.labeled[slot];
                total.positive[slot] += batch_event_census.positive[slot];
            }
        }
        update_training_population(
            &mut state,
            mixed.transitions(),
            mixed.content_masks(),
            Some(content_batch_digest),
        )?;
        let (ep_weight, rollout_enabled) = {
            let foundation = state.foundation_v2.as_ref().expect("foundation-v2 state");
            (foundation.ep_weight, foundation.rollout_enabled)
        };
        let objective = FoundationV2ObjectiveConfig {
            ep_weight,
            sigreg_projections: cfg.sigreg_projections,
            sigreg_knots: cfg.sigreg_knots,
            sigreg_seed: cfg.seed.wrapping_add(state.global_step),
            q_mse_threshold: cfg.q_mse_threshold,
            rollout_enabled,
            split_ce_weighting: cfg.split_ce_weighting,
            split_ce_changed_budget: cfg.split_ce_changed_budget,
            capture_mechanism_seams: cg_profile
                .as_ref()
                .is_some_and(RepresentativeUpdateCapture::active),
        };
        let losses = if let Some(profile) = &cg_profile {
            profile.synchronized_phase(
                &device,
                "forward_loss",
                SpanKind::Function,
                Some(ExecutionStep::Forward),
                || {
                    foundation_v2_training_loss_with_event_weights(
                        &model,
                        &mixed,
                        &host,
                        &device,
                        objective,
                        &event_slot_weights,
                    )
                },
            )?
        } else {
            foundation_v2_training_loss_with_event_weights(
                &model,
                &mixed,
                &host,
                &device,
                objective,
                &event_slot_weights,
            )?
        };
        if next_step.is_multiple_of(128) {
            // These two attribution stores are read-only. Their graphs are
            // reused below for the actual combined backward pass.
            let ep_grads = losses.ep.backward()?;
            let pred_grads = losses.pred_ce.backward()?;
            let ep_norm = gradient_l2_for_parameter_prefix(&ep_grads, &varmap, "encoder.")?;
            let pred_norm = gradient_l2_for_parameter_prefix(&pred_grads, &varmap, "encoder.")?;
            let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
            foundation.ep_weight =
                foundation_v2_ep_weight_update(foundation.ep_weight, ep_norm, pred_norm);
            let (weighted_budget_ratio, budget_met, rail) =
                foundation_v2_ep_budget_status(foundation.ep_weight, ep_norm, pred_norm);
            foundation
                .ep_gradient_budget
                .push(FoundationV2EpGradientSample {
                    step: next_step,
                    encoder_ep_gradient_l2: ep_norm,
                    encoder_prediction_gradient_l2: pred_norm,
                    ep_weight: foundation.ep_weight,
                    weighted_budget_ratio,
                    budget_met,
                    rail: Some(rail),
                });
        }
        let effective_ep_weight = state
            .foundation_v2
            .as_ref()
            .expect("foundation-v2 state")
            .ep_weight;
        let total = losses
            .non_ep_total
            .add(&losses.ep.affine(effective_ep_weight, 0.0)?)?;
        if let Some(profile) = &cg_profile {
            profile.synchronized_phase_with_range(
                &device,
                "loss_tensors",
                SpanKind::Module,
                Some(ExecutionStep::Forward),
                |range| {
                    if let Some(range) = range {
                        record_foundation_v2_profile_tensors(profile, range, &losses, &total)?;
                    }
                    Ok(())
                },
            )?;
        }
        let mut grads = if let Some(profile) = &cg_profile {
            profile.synchronized_phase(
                &device,
                "backward",
                SpanKind::Function,
                Some(ExecutionStep::Backward),
                || total.backward().map_err(Into::into),
            )?
        } else {
            total.backward()?
        };
        if let Some(profile) = &cg_profile {
            profile.synchronized_phase(
                &device,
                "gradients",
                SpanKind::Module,
                Some(ExecutionStep::Backward),
                || profile.record_gradients(&grads),
            )?;
        }
        // ADR 0003's documented approximation: the adaptive controller bounds
        // EP's encoder contribution first, then one combined gradient store is
        // clipped at 1.0. There is no separately clipped EP accumulation.
        let clip = if let Some(profile) = &cg_profile {
            profile.synchronized_phase(
                &device,
                "gradient_clip",
                SpanKind::Module,
                Some(ExecutionStep::Backward),
                || try_clip_gradients_gpu_with_stats(&mut grads, &varmap, MAX_GRAD_NORM),
            )?
        } else {
            try_clip_gradients_gpu_with_stats(&mut grads, &varmap, MAX_GRAD_NORM)?
        };
        let learning_rate = foundation_v2_wsd_learning_rate(next_step as usize, total_steps);
        if let Some(clip) = clip {
            optimizer.set_learning_rate(learning_rate)?;
            if let Some(profile) = &cg_profile {
                profile.synchronized_phase(
                    &device,
                    "optimizer",
                    SpanKind::Function,
                    Some(ExecutionStep::Optimizer),
                    || optimizer.step(&grads),
                )?;
            } else {
                optimizer.step(&grads)?;
            }
            let values =
                foundation_v2_loss_values(&losses, &total, clip.pre_clip_norm, clip.scale)?;
            if let Some(profile) = &cg_profile {
                record_foundation_v2_profile_scalars(
                    profile,
                    profile_measurement_span,
                    &values,
                    learning_rate,
                    effective_ep_weight,
                )?;
            }
            drop(profile_measurement);
            ema.update(&varmap)?;
            loss_log.append(next_step, &values, learning_rate)?;
            state.active_sums = {
                let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
                add_foundation_v2_loss_sums(&mut foundation.loss_sums, &values);
                foundation.loss_steps += 1;
                update_foundation_v2_rollout_zero_streak(
                    &mut foundation.rollout_zero_loss_consecutive_steps,
                    rollout_enabled,
                    values.rollout,
                    next_step,
                );
                foundation_v2_active_loss_means(&foundation.loss_sums, foundation.loss_steps)
            };
        } else {
            // Data/state-specific non-finite gradient: skip this update
            // (standard loss-scaler practice) instead of killing the run.
            // Weights, EMA, optimizer moments, and loss statistics are
            // untouched; the step is recorded in trainer state and the skip
            // budget fails closed on divergence-shaped patterns.
            if cg_profile.is_some() {
                bail!(
                    "non-finite gradient on preregistered profile update {next_step}; \
                     rerun with a different profile step"
                );
            }
            eprintln!(
                "skipping update {next_step}: non-finite gradient norm \
                 ({} skipped so far this run lineage)",
                state.nonfinite_skipped_updates.len() + 1
            );
            state.nonfinite_skipped_updates.push(next_step);
            if let Some(reason) =
                foundation_v2_nonfinite_skip_exhausted(&state.nonfinite_skipped_updates)
            {
                bail!("non-finite gradient skip budget exhausted: {reason}");
            }
            drop(profile_measurement);
        }
        // Release the training step's entire autograd graph, batch tensors,
        // and gradient store before any same-step evaluation work. The first
        // bundle run OOMed exactly at step 1024, where the gate evaluation,
        // the mechanism sample, and (then) a profiled capture stacked on top
        // of these still-live allocations at 39.6/48 GB steady state.
        drop(total);
        drop(losses);
        drop(grads);
        drop(mixed);
        state.global_step = next_step;
        state.optimizer_step = optimizer.step_t();
        updates_this_run += 1;

        let mut improved_best = false;
        let mut abort = false;
        if state.global_step.is_multiple_of(FOUNDATION_V2_GATE_EVERY) {
            let (metrics, mechanism_sample) = ema.with_eval_weights(&varmap, || {
                let metrics = evaluate_gate_support_with_v5_provenance(
                    &model,
                    &gate_samples,
                    &gate_content_masks,
                    &gate_provenance,
                    &device,
                )?;
                let mechanism_sample = foundation_v2_mechanism_sample(
                    &model,
                    &gate_samples,
                    &device,
                    state.global_step,
                )?;
                Ok((metrics, mechanism_sample))
            })?;
            if let (Some(profile), Some(copy_bypass_alpha)) =
                (&cg_profile, mechanism_sample.copy_bypass_alpha)
            {
                profile.record_scalar(
                    profile_measurement_span,
                    "mechanism/copy_bypass_alpha",
                    copy_bypass_alpha,
                )?;
            }
            let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
            let prior_best = foundation.best_changed_exact;
            // Only armed evaluations may seed the composed collapse floor.
            // Warmup values PASS by fiat and can peak on transient copy-decode
            // behavior; anchoring the floor to such a peak aborted arm s6 at
            // the first armed evaluation (gate policy v5).
            let prior_composed_best = foundation
                .gate_history
                .iter()
                .filter(|evaluation| evaluation.step >= FOUNDATION_V2_GATE_WARMUP_STEPS)
                .filter_map(|evaluation| evaluation.metrics.one_step_composed_changed_exact)
                .filter(|value| value.is_finite())
                .reduce(f64::max);
            let evaluation = foundation_v2_gate_evaluation(
                state.global_step,
                metrics,
                prior_best,
                prior_composed_best,
                &foundation.gate_history,
            )?;
            improved_best = foundation_v2_evaluation_improves(
                cfg.promotion_metric,
                prior_best,
                &foundation.gate_history,
                &evaluation,
            );
            foundation.best_changed_exact = evaluation.running_best_after;
            foundation.rollout_enabled =
                foundation_v2_named_gate_passed(&evaluation, "one_step_collapse");
            foundation.gate_history.push(evaluation);
            foundation.mechanism_history.push(mechanism_sample);
            abort = foundation_v2_gate_history_aborts(&foundation.gate_history);
        }
        let published_profile = if let Some(profile) = cg_profile {
            let artifacts = profile
                .finish()?
                .ok_or_else(|| anyhow::anyhow!("selected foundation-v2 profile was inactive"))?;
            if artifacts.update != next_step {
                bail!(
                    "foundation-v2 profile published update {} while update {next_step} was selected",
                    artifacts.update
                );
            }
            state
                .foundation_v2
                .as_mut()
                .expect("foundation-v2 state")
                .profiles_published
                .push(next_step);
            true
        } else {
            false
        };

        let complete = state.global_step >= total_steps as u64;
        let requested_pause = PAUSE_REQUESTED.load(Ordering::SeqCst)
            || cfg.max_steps_this_run == Some(updates_this_run);
        let periodic = cfg.checkpoint_every_steps > 0
            && state
                .global_step
                .is_multiple_of(cfg.checkpoint_every_steps as u64);
        let permanent = state
            .global_step
            .is_multiple_of(FOUNDATION_V2_PERMANENT_EVERY);
        if abort || complete || requested_pause {
            if let Some(prefetcher) = data_prefetcher.as_mut() {
                prefetcher.shutdown();
            }
        }
        if published_profile
            || periodic
            || permanent
            || improved_best
            || abort
            || complete
            || requested_pause
        {
            sync_cuda_device(&device)?;
            loss_log.flush()?;
            if permanent {
                state
                    .foundation_v2
                    .as_mut()
                    .expect("foundation-v2 state")
                    .permanent_checkpoints
                    .push(
                        cfg.output_dir
                            .join("checkpoints/permanent")
                            .join(format!("step-{:012}", state.global_step)),
                    );
            }
            let checkpoint =
                save_training_checkpoint(&varmap, &optimizer, Some(&ema), &state, &cfg)?;
            if permanent {
                publish_permanent_checkpoint(&cfg, &checkpoint)?;
                latest_checkpoint = Some(checkpoint);
            } else {
                latest_checkpoint = Some(checkpoint);
            }
            if improved_best {
                publish_best_checkpoint(
                    &cfg,
                    latest_checkpoint.as_ref().expect("saved checkpoint"),
                )?;
            }
        }
        if abort {
            let checkpoint = latest_checkpoint.clone().expect("abort saves checkpoint");
            persist_foundation_v2_abort_marker(&cfg, state.global_step, &checkpoint)?;
            let report = build_report(
                &cfg,
                &state,
                TrainStatus::Aborted,
                parameter_count,
                checkpoint.clone(),
                resumed_from.clone(),
                &run_attempts,
            );
            publish_run_artifacts(&varmap, &cfg, &report)?;
            let diagnostic = seal_foundation_v2_abort(&cfg, &checkpoint, &report)?;
            bail!(
                "foundation-v2 aborted after two consecutive gate failures; diagnostic bundle {}",
                diagnostic.display()
            );
        }
        if requested_pause && !complete {
            let report = build_report(
                &cfg,
                &state,
                TrainStatus::Paused,
                parameter_count,
                latest_checkpoint.clone().expect("pause saves checkpoint"),
                resumed_from.clone(),
                &run_attempts,
            );
            publish_run_artifacts(&varmap, &cfg, &report)?;
            sync_cuda_device(&device)?;
            return Ok(report);
        }
    }
}

/// Train lessons in order. SIGINT/SIGTERM pauses after the current optimizer update.
pub fn train(cfg: &TrainConfig) -> Result<TrainReport> {
    if cfg.recipe == TrainingRecipe::FoundationV2 {
        return train_foundation_v2(cfg);
    }
    let mut cfg = cfg.clone();
    let explicit_resume = cfg.resume.is_some();
    if !explicit_resume && implicit_resume_source(&cfg).is_some() {
        cfg.resume = Some(cfg.output_dir.join("checkpoints"));
        tracing::info!(
            "auto-resuming from {}",
            cfg.output_dir.join("checkpoints").display()
        );
    }
    let cfg = &cfg;
    cfg.validate()?;
    fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("create {}", cfg.output_dir.display()))?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&cfg.output_dir)?)
    } else {
        None
    };
    let _train_pid = TrainPidGuard::install(&cfg.output_dir)?;
    PAUSE_REQUESTED.store(false, Ordering::SeqCst);
    install_pause_handler()?;
    let device = resolve_device(&cfg.device)?;
    let model_cfg = cfg.model_config();
    model_cfg.validate()?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = WorldModel::new(model_cfg, vb)?;
    let adam = adam_params(cfg);
    let mut optimizer =
        CheckpointHybridOptimizer::new(&varmap, adam, cfg.muon_momentum, cfg.muon_rms_scale)?;
    let parameter_names = optimizer.parameter_names();
    let parameter_count = parameter_count(&varmap);

    let resume_source = cfg.resume.clone().or_else(|| implicit_resume_source(cfg));
    let resumed_from = resume_source
        .as_deref()
        .map(resolve_resume_checkpoint)
        .transpose()?;
    let mut state = if let Some(bundle) = &resumed_from {
        load_training_checkpoint(bundle, cfg, &mut varmap, &mut optimizer, None)?
    } else {
        reinit_varmap_deterministic(&varmap, cfg.init_seed.unwrap_or(cfg.seed))?;
        if cfg.world_core_v3 {
            zero_v3_spatial_residual(&varmap)?;
        }
        // Treatment initializations survive the generic reinit on every
        // fresh-init path, not only foundation-v2.
        zero_operator_conditioning_projection(&varmap)?;
        zero_context_film_projections(&varmap)?;
        init_copy_bypass_gate(&varmap)?;
        restore_copy_gate_bias_prior(&varmap, cfg.copy_gate_bias_prior)?;
        TrainerState {
            schema: TRAINER_STATE_SCHEMA.into(),
            contract: TrainingContract::from(cfg),
            global_step: 0,
            lesson_index: 0,
            step_in_lesson: 0,
            optimizer_step: 0,
            completed_lessons: Vec::with_capacity(cfg.lessons.len()),
            active_sums: LessonLossMeans::default(),
            parameter_names,
            training_population_hash: default_training_population_hash(),
            training_content_hash: [0; 32],
            training_population_rows: 0,
            batch_schedule_migrations: Vec::new(),
            nonfinite_skipped_updates: Vec::new(),
            profile: ProfileState::Pending,
            gradient_pressure: None,
            gradient_pressure_samples: Vec::new(),
            foundation_v2: None,
        }
    };
    ensure_profile_campaign_manifest(cfg, &[cfg.profile_update])?;
    let attempt = begin_run_attempt(
        &cfg.output_dir,
        resumed_from.as_deref(),
        resumed_from.as_ref().map(|_| state.global_step),
    )?;
    let run_attempts = complete_run_attempt_repair(&cfg.output_dir, attempt, None)?;
    let mut latest_checkpoint = resumed_from.clone();
    let mut latest_checkpoint_step = resumed_from.as_ref().map(|_| state.global_step);
    if resumed_from.is_none() {
        let initial = save_training_checkpoint(&varmap, &optimizer, None, &state, cfg)?;
        latest_checkpoint = Some(initial);
        latest_checkpoint_step = Some(0);
    }
    let mut updates_this_run = 0usize;
    if resumed_from.is_some() {
        device.synchronize()?;
    }
    if state.global_step >= cfg.profile_update && matches!(state.profile, ProfileState::Pending) {
        if let Some(artifacts) =
            reconcile_profile_bundle(&cfg.output_dir, cfg.profile_update, &cfg.device)?
        {
            tracing::warn!(
                "reconciled profile update {} from verified bundle {}",
                cfg.profile_update,
                artifacts.directory.display()
            );
            state.profile = ProfileState::Published(artifacts);
        } else {
            bail!(
                "resume state has passed profile update {} without a publishable evidence bundle",
                cfg.profile_update
            );
        }
    }
    // Derived state only: never checkpointed, since a cold cache regenerates the
    // identical episodes on resume.
    let mut episode_cache = EpisodeCache::default();
    let mut profile = StepProfile::from_env();
    let malloc_trim_every = crate::alloc::trim_interval_from_env();
    let use_prefetch = cfg.prefetch_batches;
    let mut prefetcher = if use_prefetch {
        Some(BatchPrefetcher::new())
    } else {
        None
    };
    let mut prefetched_through_step = state.global_step;

    loop {
        let complete = state.lesson_index == cfg.lessons.len();
        if complete {
            if latest_checkpoint.is_none() {
                latest_checkpoint = Some(save_training_checkpoint(
                    &varmap, &optimizer, None, &state, cfg,
                )?);
            }
            let report = build_report(
                cfg,
                &state,
                TrainStatus::Completed,
                parameter_count,
                latest_checkpoint.expect("completed training has a checkpoint"),
                resumed_from,
                &run_attempts,
            );
            publish_run_artifacts(&varmap, cfg, &report)?;
            if let Some(p) = prefetcher.as_mut() {
                p.shutdown();
            }
            sync_cuda_device(&device)?;
            return Ok(report);
        }

        // A signal received between updates can reuse the last durable bundle.
        if PAUSE_REQUESTED.load(Ordering::SeqCst) {
            let checkpoint = match (latest_checkpoint, latest_checkpoint_step) {
                (Some(path), Some(step)) if step == state.global_step => path,
                _ => save_training_checkpoint(&varmap, &optimizer, None, &state, cfg)?,
            };
            let report = build_report(
                cfg,
                &state,
                TrainStatus::Paused,
                parameter_count,
                checkpoint,
                resumed_from,
                &run_attempts,
            );
            publish_run_artifacts(&varmap, cfg, &report)?;
            if let Some(p) = prefetcher.as_mut() {
                p.shutdown();
            }
            sync_cuda_device(&device)?;
            return Ok(report);
        }

        let lesson = &cfg.lessons[state.lesson_index];
        let active_lesson_steps = steps_for_lesson(cfg, lesson);
        let curriculum = lesson_to_curriculum(lesson)?;
        let cg_profile = RepresentativeUpdateCapture::begin(CaptureSpec {
            completed_updates: state.global_step,
            selected_update: cfg.profile_update,
            state: &state.profile,
            output_dir: &cfg.output_dir,
            device: &cfg.device,
            measured_region_device_synchronized: device.is_cuda(),
            lesson,
            physical_batch: cfg.physical_batch,
            grad_accum: cfg.grad_accum,
            hidden_dim: cfg.hidden_dim,
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
            precision: if cfg.bf16_recurrent_core {
                "bf16-recurrent-core/f32-rest"
            } else if cfg.bf16_conv {
                "bf16-conv/f32-rest"
            } else {
                "f32"
            },
            varmap: &varmap,
            gradient_clip_state: GradientClipState::PostClip,
        })?;
        let prof = profile.enabled();
        if cg_profile.active() {
            sync_cuda_device(&device)?;
        }
        let profile_measurement = cg_profile.measurement();
        let profile_measurement_span = profile_measurement.as_ref().and_then(ProfileRange::span_id);
        let accum = cfg.grad_accum.max(1);
        let loss_weights =
            lesson_loss_weights(lesson, cfg, state.step_in_lesson, state.global_step);
        let depth = sample_recursion_depth(cfg, state.global_step);
        let run_rollout_this_step =
            loss_weights.rollout > 0.0 && matches!(lesson.as_str(), "sequential" | "retarget");
        let rollout_episode_start =
            scheduled_episode_start(cfg.seed, state.global_step, 0, accum, cfg.shuffled_episodes);
        let mut rollout_trace_handle = if run_rollout_this_step {
            let curriculum_owned = curriculum.to_string();
            let seed = cfg.seed;
            Some(thread::spawn(move || {
                collect_rollout_trace(&curriculum_owned, seed, rollout_episode_start, Split::Train)
            }))
        } else {
            None
        };
        if use_prefetch {
            ensure_prefetch_scope(
                &mut prefetcher,
                &mut prefetched_through_step,
                curriculum,
                cfg,
                state.global_step,
            );
            top_up_prefetch(
                &mut prefetched_through_step,
                prefetcher.as_mut().unwrap(),
                curriculum,
                cfg,
                state.global_step,
                accum,
            )?;
        }
        let accum_f = accum as f64;
        let inv = 1.0 / accum_f;
        let mut accumulated_grads: Option<GradStore> = None;
        let mut metric_tensors = Vec::with_capacity(accum);
        let mut step_metrics = LessonLossMeans::default();
        let mut rollout_trace_cache: Option<Vec<TransitionSample>> = None;
        for micro in 0..accum {
            let samples = cg_profile.synchronized_phase(
                &device,
                "generate",
                SpanKind::Module,
                None,
                || {
                    timed(prof, &device, &mut profile.generate, || {
                        let _span = tracing::info_span!("generate").entered();
                        collect_one_micro_sample_batch(
                            micro,
                            accum,
                            use_prefetch,
                            if use_prefetch {
                                prefetcher.as_mut()
                            } else {
                                None
                            },
                            cfg,
                            curriculum,
                            state.global_step,
                            &mut episode_cache,
                        )
                    })
                },
            )?;
            let content_masks = samples
                .iter()
                .map(training_content_mask_from_provenance)
                .collect::<Result<Vec<_>>>()?;
            update_training_population(&mut state, samples.iter(), content_masks.iter(), None)?;
            if micro == 0 && use_prefetch {
                top_up_prefetch(
                    &mut prefetched_through_step,
                    prefetcher.as_mut().unwrap(),
                    curriculum,
                    cfg,
                    state.global_step,
                    accum,
                )?;
            }
            if micro == 0 && run_rollout_this_step && rollout_trace_cache.is_none() {
                rollout_trace_cache = Some(if let Some(handle) = rollout_trace_handle.take() {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("rollout trace thread panicked"))??
                } else {
                    let episode_start = scheduled_episode_start(
                        cfg.seed,
                        state.global_step,
                        0,
                        accum,
                        cfg.shuffled_episodes,
                    );
                    match episode_cache.rollout_trace(
                        curriculum,
                        cfg.seed,
                        episode_start,
                        Split::Train,
                    ) {
                        Ok(trace) => trace,
                        Err(resume_from) => {
                            collect_rollout_trace(curriculum, cfg.seed, resume_from, Split::Train)?
                        }
                    }
                });
            }
            let (batch, ordered_trace, sigreg_windows) = cg_profile.synchronized_phase_with_range(
                &device,
                "stage",
                SpanKind::Module,
                None,
                |range| {
                    let staged = timed(prof, &device, &mut profile.stage, || {
                        let _span = tracing::info_span!("stage").entered();
                        let batch = batch_from_samples(&samples, &device)?;
                        // Both arms derive their population from these exact ordered rows.
                        // Target selection happens only after the shared encoder pass.
                        let sigreg_windows = if cfg.sigreg_target == SigregTarget::Marginal
                            && cfg.sigreg_temporal_window < 2
                        {
                            // The window is deliberately ignored for legacy marginal configs.
                            None
                        } else {
                            ordered_sigreg_windows(&samples, cfg.sigreg_temporal_window)?
                        };
                        let ordered_trace = if micro == 0 && run_rollout_this_step {
                            rollout_trace_cache
                                .as_deref()
                                .map(|trace| ordered_trace_from_samples(trace, &device))
                                .transpose()?
                        } else {
                            None
                        };
                        Ok((batch, ordered_trace, sigreg_windows))
                    })?;
                    if let Some(range) = range {
                        cg_profile.record_tensor(range, "batch.frames", &staged.0.frames, None)?;
                    }
                    Ok(staged)
                },
            )?;
            let micro_sigreg_seed =
                legacy_sigreg_projection_seed(cfg.seed, state.global_step, accum, micro);
            let run_rollout = micro == 0
                && loss_weights.rollout > 0.0
                && matches!(lesson.as_str(), "sequential" | "retarget");
            let (micro_losses, micro_rollout, micro_prefix_multi, micro_total) = cg_profile
                .synchronized_phase_with_range(
                    &device,
                    "forward",
                    SpanKind::Function,
                    Some(ExecutionStep::Forward),
                    |range| {
                        let result = timed(prof, &device, &mut profile.forward, || {
                            let _span = tracing::info_span!("forward").entered();
                            let losses = leworld_loss_with_sigreg_windows(
                                &model,
                                &batch,
                                sigreg_windows.as_ref(),
                                Some(&samples),
                                cfg,
                                depth,
                                micro_sigreg_seed,
                                loss_weights,
                                Some(&cg_profile),
                            )?;
                            let rollout_trace = if run_rollout {
                                ordered_trace.as_ref()
                            } else {
                                None
                            };
                            let zero = Tensor::zeros((), DType::F32, &device)?;
                            let rollout = if let Some(trace) = rollout_trace {
                                let horizon = if cfg.phased_training {
                                    rollout_horizon_for_lesson(
                                        lesson,
                                        state.step_in_lesson,
                                        active_lesson_steps,
                                    )
                                } else if lesson == "retarget" {
                                    RETARGET_MAX_ROLLOUT_HORIZON
                                } else {
                                    8
                                };
                                open_loop_latent_loss(
                                    &model,
                                    trace,
                                    horizon,
                                    depth,
                                    rollout_teacher_mix(
                                        lesson,
                                        state.step_in_lesson,
                                        active_lesson_steps,
                                    ),
                                    cfg.seed.wrapping_add(state.global_step),
                                )?
                            } else {
                                zero.clone()
                            };
                            let prefix_multi = if let Some(trace) = rollout_trace {
                                if loss_weights.prefix > 0.0 {
                                    prefix_multi_horizon_loss(&model, trace)?
                                } else {
                                    zero.clone()
                                }
                            } else {
                                zero
                            };
                            let mut total = losses.total.clone();
                            if loss_weights.rollout > 0.0 {
                                total = total.add(&rollout.affine(loss_weights.rollout, 0.0)?)?;
                            }
                            if loss_weights.prefix > 0.0 && rollout_trace.is_some() {
                                total =
                                    total.add(&prefix_multi.affine(loss_weights.prefix, 0.0)?)?;
                            }
                            Ok((losses, rollout, prefix_multi, total))
                        })?;
                        if let Some(range) = range {
                            cg_profile.record_tensor(
                                range,
                                "loss.total",
                                &result.3,
                                Some(ExecutionStep::Forward),
                            )?;
                        }
                        Ok(result)
                    },
                )?;
            step_metrics.branch_groups += micro_losses.branch_audit.groups as f64 * inv;
            step_metrics.changed_branches +=
                micro_losses.branch_audit.changed_branches as f64 * inv;
            step_metrics.equivalent_pairs +=
                micro_losses.branch_audit.equivalent_pairs as f64 * inv;
            step_metrics.distinct_pairs += micro_losses.branch_audit.distinct_pairs as f64 * inv;
            step_metrics.action6_branches +=
                micro_losses.branch_audit.action6_branches as f64 * inv;
            step_metrics.action_recovery_branches +=
                micro_losses.branch_audit.action_recovery_branches as f64 * inv;
            step_metrics.spatial_population_rows +=
                micro_losses.branch_audit.spatial_population_rows as f64 * inv;
            step_metrics.pooled_population_rows +=
                micro_losses.branch_audit.pooled_population_rows as f64 * inv;
            step_metrics.displacement_population_rows +=
                micro_losses.branch_audit.displacement_population_rows as f64 * inv;
            step_metrics.unique_changed_outcomes +=
                micro_losses.branch_audit.unique_changed_outcomes as f64 * inv;
            step_metrics.grounding_changed_patches +=
                micro_losses.grounding_changed_patches as f64 * inv;
            step_metrics.grounding_unchanged_patches +=
                micro_losses.grounding_unchanged_patches as f64 * inv;
            let pressure_update = state.global_step.saturating_add(1);
            let pressure_scheduled = if cfg.pressure_updates.is_empty() {
                pressure_update == cfg.profile_update.saturating_sub(1).max(1)
                    && state.gradient_pressure_samples.is_empty()
            } else {
                cfg.pressure_updates.contains(&pressure_update)
                    && !state
                        .gradient_pressure_samples
                        .iter()
                        .any(|sample| sample.update == pressure_update)
            };
            if micro == 0 && pressure_scheduled {
                // Read-only attribution: these stores are discarded before the
                // normal total-loss backward and never reach the optimizer.
                if loss_weights.sigreg > 0.0
                    || cfg.patch_grounding_weight > 0.0
                    || (loss_weights.q > 0.0 && !cfg.stop_grad_q_y)
                    || cfg.branch_learning.displacement_health.is_none()
                {
                    let next_grads = micro_losses.next_latent.backward()?;
                    let next_norm =
                        gradient_l2_for_parameter_prefix(&next_grads, &varmap, "encoder.")?;
                    let (sigreg_norm, sigreg_grads) = if loss_weights.sigreg > 0.0 {
                        let grads = micro_losses
                            .sigreg_bounded
                            .affine(loss_weights.sigreg, 0.0)?
                            .backward()?;
                        (
                            Some(gradient_l2_for_parameter_prefix(
                                &grads, &varmap, "encoder.",
                            )?),
                            Some(grads),
                        )
                    } else {
                        (None, None)
                    };
                    let readout_norm = if loss_weights.q > 0.0 && !cfg.stop_grad_q_y {
                        let combined_readout = micro_losses
                            .q
                            .affine(loss_weights.q, 0.0)?
                            .add(&micro_losses.q_surprise.affine(Q_SURPRISE_WEIGHT, 0.0)?)?;
                        let grads = combined_readout.backward()?;
                        Some(gradient_l2_for_parameter_prefix(
                            &grads, &varmap, "encoder.",
                        )?)
                    } else {
                        None
                    };
                    let (grounding_norm, grounding_head_norm, grounding_grads) =
                        if cfg.patch_grounding_weight > 0.0 {
                            let grads = micro_losses
                                .patch_grounding
                                .affine(cfg.patch_grounding_weight, 0.0)?
                                .backward()?;
                            (
                                Some(gradient_l2_for_parameter_prefix(
                                    &grads, &varmap, "encoder.",
                                )?),
                                Some(gradient_l2_for_parameter_prefix(
                                    &grads,
                                    &varmap,
                                    "grounding_head.",
                                )?),
                                Some(grads),
                            )
                        } else {
                            (None, None, None)
                        };
                    let sigreg_next_cosine = sigreg_grads
                        .as_ref()
                        .map(|grads| {
                            gradient_cosine_for_parameter_prefix(
                                grads,
                                &next_grads,
                                &varmap,
                                "encoder.",
                            )
                        })
                        .transpose()?;
                    let grounding_next_cosine = grounding_grads
                        .as_ref()
                        .map(|grads| {
                            gradient_cosine_for_parameter_prefix(
                                grads,
                                &next_grads,
                                &varmap,
                                "encoder.",
                            )
                        })
                        .transpose()?;
                    let grounding_sigreg_cosine = grounding_grads
                        .as_ref()
                        .zip(sigreg_grads.as_ref())
                        .map(|(grounding, sigreg)| {
                            gradient_cosine_for_parameter_prefix(
                                grounding, sigreg, &varmap, "encoder.",
                            )
                        })
                        .transpose()?;
                    let diagnostic = GradientPressureDiagnostics {
                        update: pressure_update,
                        encoder_next_latent_l2: next_norm,
                        encoder_sigreg_weighted_l2: sigreg_norm.unwrap_or(0.0),
                        sigreg_to_next_ratio: sigreg_norm
                            .filter(|_| next_norm > 0.0)
                            .map(|norm| norm / next_norm),
                        encoder_grounding_weighted_l2: grounding_norm,
                        grounding_to_next_ratio: grounding_norm
                            .filter(|_| next_norm > 0.0)
                            .map(|norm| norm / next_norm),
                        grounding_head_weighted_l2: grounding_head_norm,
                        sigreg_next_cosine,
                        grounding_next_cosine,
                        grounding_sigreg_cosine,
                        encoder_readout_weighted_l2: readout_norm,
                        readout_to_next_ratio: readout_norm
                            .filter(|_| next_norm > 0.0)
                            .map(|norm| norm / next_norm),
                        model_next_latent_l2: None,
                        displacement_health_weighted_l2: None,
                        displacement_health_to_next_ratio: None,
                    };
                    state.gradient_pressure = Some(diagnostic.clone());
                    state.gradient_pressure_samples.push(diagnostic);
                } else if let Some(health) = cfg.branch_learning.displacement_health {
                    let next_grads = micro_losses.next_latent.backward()?;
                    let next_norm = gradient_l2_all_parameters(&next_grads, &varmap)?;
                    drop(next_grads);
                    let weighted_health = micro_losses
                        .displacement_variance
                        .affine(f64::from(health.variance_weight), 0.0)?
                        .add(
                            &micro_losses
                                .displacement_covariance
                                .affine(f64::from(health.covariance_weight), 0.0)?,
                        )?;
                    let health_grads = weighted_health.backward()?;
                    let health_norm = gradient_l2_all_parameters(&health_grads, &varmap)?;
                    let diagnostic = GradientPressureDiagnostics {
                        update: pressure_update,
                        encoder_next_latent_l2: 0.0,
                        encoder_sigreg_weighted_l2: 0.0,
                        sigreg_to_next_ratio: None,
                        encoder_grounding_weighted_l2: None,
                        grounding_to_next_ratio: None,
                        grounding_head_weighted_l2: None,
                        sigreg_next_cosine: None,
                        grounding_next_cosine: None,
                        grounding_sigreg_cosine: None,
                        encoder_readout_weighted_l2: None,
                        readout_to_next_ratio: None,
                        model_next_latent_l2: Some(next_norm),
                        displacement_health_weighted_l2: Some(health_norm),
                        displacement_health_to_next_ratio: (next_norm > 0.0)
                            .then_some(health_norm / next_norm),
                    };
                    state.gradient_pressure = Some(diagnostic.clone());
                    state.gradient_pressure_samples.push(diagnostic);
                }
            }
            let scaled_micro = micro_total.affine(inv, 0.0)?;
            let micro_grads = cg_profile.synchronized_phase(
                &device,
                "backward",
                SpanKind::Function,
                Some(ExecutionStep::Backward),
                || {
                    timed(prof, &device, &mut profile.backward, || {
                        let _span = tracing::info_span!("backward").entered();
                        scaled_micro.backward().map_err(Into::into)
                    })
                },
            )?;
            accumulate_parameter_gradients(&mut accumulated_grads, micro_grads, &varmap)?;
            metric_tensors.push(training_loss_tensors(
                &micro_losses,
                &micro_rollout,
                &micro_prefix_multi,
                &micro_total,
            ));
        }
        let checked_losses = cg_profile.synchronized_phase(
            &device,
            "loss_readback",
            SpanKind::Module,
            None,
            || checked_training_losses(&metric_tensors),
        )?;
        for micro_vals in checked_losses {
            step_metrics.total += micro_vals.total as f64 * inv;
            step_metrics.next_latent += micro_vals.next_latent as f64 * inv;
            step_metrics.rollout += micro_vals.rollout as f64 * inv;
            step_metrics.sigreg_raw += micro_vals.sigreg_raw as f64 * inv;
            step_metrics.sigreg_bounded += micro_vals.sigreg_bounded as f64 * inv;
            step_metrics.patch_grounding += micro_vals.patch_grounding as f64 * inv;
            step_metrics.event += micro_vals.event as f64 * inv;
            step_metrics.q += micro_vals.q as f64 * inv;
            step_metrics.prefix += micro_vals.prefix as f64 * inv;
            step_metrics.reliability += micro_vals.reliability as f64 * inv;
            step_metrics.branch_total += micro_vals.branch_total as f64 * inv;
            step_metrics.outcome_pull += micro_vals.outcome_pull as f64 * inv;
            step_metrics.outcome_push += micro_vals.outcome_push as f64 * inv;
            step_metrics.action_recovery += micro_vals.action_recovery as f64 * inv;
            step_metrics.coordinate_recovery += micro_vals.coordinate_recovery as f64 * inv;
            step_metrics.changed_margin += micro_vals.changed_margin as f64 * inv;
            step_metrics.spatial_variance += micro_vals.spatial_variance as f64 * inv;
            step_metrics.spatial_covariance += micro_vals.spatial_covariance as f64 * inv;
            step_metrics.pooled_variance += micro_vals.pooled_variance as f64 * inv;
            step_metrics.pooled_covariance += micro_vals.pooled_covariance as f64 * inv;
            step_metrics.displacement_variance += micro_vals.displacement_variance as f64 * inv;
            step_metrics.displacement_covariance += micro_vals.displacement_covariance as f64 * inv;
        }
        let mut grads = accumulated_grads
            .ok_or_else(|| anyhow::anyhow!("grad_accum produced no microbatches"))?;
        let clip_stats = cg_profile.synchronized_phase(
            &device,
            "gradient_clip",
            SpanKind::Module,
            Some(ExecutionStep::Backward),
            || clip_gradients_gpu_with_stats(&mut grads, &varmap, MAX_GRAD_NORM),
        )?;
        step_metrics.pre_clip_gradient_norm = clip_stats.pre_clip_norm;
        step_metrics.gradient_clip_scale = clip_stats.scale;
        step_metrics.clipped_updates = f64::from(clip_stats.scale < 1.0);
        if cg_profile.active() {
            cg_profile.synchronized_phase(
                &device,
                "gradients",
                SpanKind::Module,
                Some(ExecutionStep::Backward),
                || cg_profile.record_gradients(&grads),
            )?;
        }
        cg_profile.synchronized_phase(
            &device,
            "optimizer",
            SpanKind::Function,
            Some(ExecutionStep::Optimizer),
            || {
                timed(prof, &device, &mut profile.optimizer, || {
                    let _span = tracing::info_span!("optimizer").entered();
                    optimizer.step(&grads)
                })
            },
        )?;
        drop(grads);

        cg_profile.synchronized_phase(&device, "metrics", SpanKind::Module, None, || {
            timed(prof, &device, &mut profile.metrics, || {
                let _span = tracing::info_span!("metrics").entered();
                state.active_sums.total += step_metrics.total;
                state.active_sums.next_latent += step_metrics.next_latent;
                state.active_sums.rollout += step_metrics.rollout;
                state.active_sums.sigreg_raw += step_metrics.sigreg_raw;
                state.active_sums.sigreg_bounded += step_metrics.sigreg_bounded;
                state.active_sums.patch_grounding += step_metrics.patch_grounding;
                state.active_sums.grounding_changed_patches +=
                    step_metrics.grounding_changed_patches;
                state.active_sums.grounding_unchanged_patches +=
                    step_metrics.grounding_unchanged_patches;
                state.active_sums.pre_clip_gradient_norm += step_metrics.pre_clip_gradient_norm;
                state.active_sums.gradient_clip_scale += step_metrics.gradient_clip_scale;
                state.active_sums.clipped_updates += step_metrics.clipped_updates;
                state.active_sums.event += step_metrics.event;
                state.active_sums.q += step_metrics.q;
                state.active_sums.prefix += step_metrics.prefix;
                state.active_sums.reliability += step_metrics.reliability;
                state.active_sums.branch_total += step_metrics.branch_total;
                state.active_sums.outcome_pull += step_metrics.outcome_pull;
                state.active_sums.outcome_push += step_metrics.outcome_push;
                state.active_sums.action_recovery += step_metrics.action_recovery;
                state.active_sums.coordinate_recovery += step_metrics.coordinate_recovery;
                state.active_sums.changed_margin += step_metrics.changed_margin;
                state.active_sums.spatial_variance += step_metrics.spatial_variance;
                state.active_sums.spatial_covariance += step_metrics.spatial_covariance;
                state.active_sums.pooled_variance += step_metrics.pooled_variance;
                state.active_sums.pooled_covariance += step_metrics.pooled_covariance;
                state.active_sums.displacement_variance += step_metrics.displacement_variance;
                state.active_sums.displacement_covariance += step_metrics.displacement_covariance;
                state.active_sums.branch_groups += step_metrics.branch_groups;
                state.active_sums.changed_branches += step_metrics.changed_branches;
                state.active_sums.equivalent_pairs += step_metrics.equivalent_pairs;
                state.active_sums.distinct_pairs += step_metrics.distinct_pairs;
                state.active_sums.action6_branches += step_metrics.action6_branches;
                state.active_sums.action_recovery_branches += step_metrics.action_recovery_branches;
                state.active_sums.spatial_population_rows += step_metrics.spatial_population_rows;
                state.active_sums.pooled_population_rows += step_metrics.pooled_population_rows;
                state.active_sums.displacement_population_rows +=
                    step_metrics.displacement_population_rows;
                state.active_sums.unique_changed_outcomes += step_metrics.unique_changed_outcomes;
                Ok(())
            })
        })?;
        record_training_profile_scalars(
            &cg_profile,
            profile_measurement_span,
            &step_metrics,
            cfg.lr,
        )?;
        if cg_profile.active() {
            sync_cuda_device(&device)?;
        }
        drop(profile_measurement);
        let published_profile = if let Some(artifacts) = cg_profile.finish()? {
            state.profile = ProfileState::Published(artifacts);
            true
        } else {
            false
        };
        state.global_step = state
            .global_step
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("global step overflow"))?;
        state.optimizer_step = optimizer.step_t();
        state.step_in_lesson += 1;
        updates_this_run += 1;

        let lesson_boundary = state.step_in_lesson == active_lesson_steps;
        if lesson_boundary {
            state.completed_lessons.push(LessonReport {
                lesson: lesson.clone(),
                curriculum: curriculum.to_string(),
                steps: active_lesson_steps,
                mean_losses: loss_means(&state.active_sums, active_lesson_steps),
            });
            if lesson == "falsification"
                && matches!(
                    cfg.lessons.get(state.lesson_index + 1).map(String::as_str),
                    Some("retarget") | None
                )
            {
                save_export_snapshot(&varmap, &cfg.output_dir)?;
            }
            state.lesson_index += 1;
            state.step_in_lesson = 0;
            state.active_sums = LessonLossMeans::default();
        }

        let complete = state.lesson_index == cfg.lessons.len();
        let requested_pause = PAUSE_REQUESTED.load(Ordering::SeqCst)
            || cfg.max_steps_this_run == Some(updates_this_run);
        let periodic = published_profile
            || lesson_boundary
            || (cfg.checkpoint_every_steps > 0
                && state.global_step % cfg.checkpoint_every_steps as u64 == 0);
        if complete || requested_pause || periodic {
            if let Some(p) = prefetcher.as_mut() {
                p.shutdown();
            }
            sync_cuda_device(&device)?;
            crate::alloc::trim_host_heap();
            latest_checkpoint = Some(timed(prof, &device, &mut profile.checkpoint, || {
                let _span = tracing::info_span!("checkpoint").entered();
                save_training_checkpoint(&varmap, &optimizer, None, &state, cfg)
            })?);
            latest_checkpoint_step = Some(state.global_step);
            if use_prefetch {
                // The queued batches died with the old workers, so rewind the
                // submission cursor and refill the whole lookahead window.
                restart_prefetch_pipeline(
                    &mut prefetcher,
                    &mut prefetched_through_step,
                    state.global_step,
                );
            }
        }
        profile.steps += 1;
        profile.report(state.global_step);
        if malloc_trim_every > 0 && (state.global_step as usize).is_multiple_of(malloc_trim_every) {
            crate::alloc::trim_host_heap();
        }
        if requested_pause && !complete {
            let report = build_report(
                cfg,
                &state,
                TrainStatus::Paused,
                parameter_count,
                latest_checkpoint.expect("pause writes a checkpoint"),
                resumed_from,
                &run_attempts,
            );
            publish_run_artifacts(&varmap, cfg, &report)?;
            if let Some(p) = prefetcher.as_mut() {
                p.shutdown();
            }
            sync_cuda_device(&device)?;
            return Ok(report);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Split;
    use crate::p2::data::{palette, ArcAction, EpisodeOperator, GoalFeatures, OperatorFamily};

    #[test]
    fn foundation_v2_unimix_ce_matches_materialized_reference() -> Result<()> {
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCE_F005);
        let logits = Tensor::from_vec(
            (0..3 * 5 * 7 * PALETTE_SIZE)
                .map(|_| rng.random_range(-8.0f32..8.0f32))
                .collect::<Vec<_>>(),
            (3, 5, 7, PALETTE_SIZE),
            &device,
        )?;
        let labels = Tensor::from_vec(
            (0..3 * 5 * 7)
                .map(|_| rng.random_range(0..PALETTE_SIZE as u32))
                .collect::<Vec<_>>(),
            (3, 5, 7),
            &device,
        )?;
        let fused = foundation_v2_unimix_ce(&logits, &labels)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = foundation_v2_unimix_ce_reference(&logits, &labels)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (got, want)) in fused.iter().zip(&reference).enumerate() {
            let relative_error = (got - want).abs() / want.abs().max(f32::MIN_POSITIVE);
            assert!(
                relative_error <= 1e-6,
                "unimix CE differs at {index}: got {got}, want {want}, relative error {relative_error}"
            );
        }
        Ok(())
    }

    #[test]
    fn foundation_v2_copy_gate_supervises_pad_as_unchanged() -> Result<()> {
        let device = Device::Cpu;
        // Pixel 0 stands for content and pixel 1 for PAD. Neither changed, so
        // opening the gate only on PAD must increase the objective.
        let changed = Tensor::zeros((1, 1, 2), DType::F32, &device)?;
        let pad_closed = Tensor::new(&[[[0.0f32, -20.0]]], &device)?;
        let pad_open = Tensor::new(&[[[0.0f32, 20.0]]], &device)?;
        let closed =
            foundation_v2_copy_gate_loss(&pad_closed, &changed, 7.0)?.to_scalar::<f32>()?;
        let open = foundation_v2_copy_gate_loss(&pad_open, &changed, 7.0)?.to_scalar::<f32>()?;
        assert!(
            open > closed + 9.0,
            "PAD gate loss did not activate: {closed} vs {open}"
        );
        Ok(())
    }

    #[test]
    fn foundation_v2_q_targets_ignore_pad_changes_and_mask_empty_rows() -> Result<()> {
        let device = Device::Cpu;
        let current = Tensor::new(&[[[0u32, 0]], [[0, 0]]], &device)?;
        let target = Tensor::new(&[[[1u32, 0]], [[1, 0]]], &device)?;
        let composed = Tensor::new(&[[[0u32, 0]], [[0, 0]]], &device)?;
        let content = Tensor::new(&[[[0.0f32, 1.0]], [[0.0, 0.0]]], &device)?;
        let (targets, mask) =
            foundation_v2_graded_q_targets_from_labels(&composed, &current, &target, &content)?;
        assert_eq!(targets.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 0.0]);
        assert_eq!(mask.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 0.0]);

        let logits = Tensor::new(&[[0.0f32], [100.0]], &device)?;
        let loss =
            masked_bce_with_slot_weights(&logits, &targets, &mask, None)?.to_scalar::<f32>()?;
        assert!(
            (loss - std::f32::consts::LN_2).abs() < 1e-6,
            "masked Q BCE was {loss}"
        );
        Ok(())
    }

    #[test]
    fn foundation_v2_q_and_reliability_receive_distinct_targets() -> Result<()> {
        let device = Device::Cpu;
        let labels = Tensor::new(&[[[3u32]]], &device)?;
        let content = Tensor::new(&[[[1.0f32]]], &device)?;
        let (q_targets, q_mask) =
            foundation_v2_graded_q_targets_from_labels(&labels, &labels, &labels, &content)?;
        let predicted_latent = Tensor::new(&[[[[1.0f32]]]], &device)?;
        let factual_target_latent = Tensor::zeros_like(&predicted_latent)?;
        let reliability_targets =
            foundation_v2_reliability_targets(&predicted_latent, &factual_target_latent, 0.05)?;

        assert_eq!(q_targets.flatten_all()?.to_vec1::<f32>()?, vec![1.0]);
        assert_eq!(q_mask.flatten_all()?.to_vec1::<f32>()?, vec![1.0]);
        assert_eq!(
            reliability_targets.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0]
        );
        Ok(())
    }

    #[test]
    fn foundation_v2_reliability_target_includes_threshold_boundary() -> Result<()> {
        let device = Device::Cpu;
        let predicted = Tensor::new(&[[0.5f32], [0.5001]], &device)?;
        let factual = Tensor::zeros_like(&predicted)?;
        let targets = foundation_v2_reliability_targets(&predicted, &factual, 0.25)?;
        assert_eq!(targets.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn foundation_v2_reliability_targets_detach_latent_path() -> Result<()> {
        let device = Device::Cpu;
        let predicted = Var::from_tensor(&Tensor::new(&[[1.0f32]], &device)?)?;
        let factual = Tensor::zeros((1, 1), DType::F32, &device)?;
        let targets = foundation_v2_reliability_targets(&predicted, &factual, 0.05)?;
        let logits = Var::from_tensor(&Tensor::new(&[[0.0f32]], &device)?)?;
        let mask = Tensor::ones_like(&targets)?;
        let loss = masked_bce_with_slot_weights(&logits, &targets, &mask, None)?;
        let grads = loss.backward()?;

        assert!(
            grads.get(&predicted).is_none(),
            "reliability target leaked a gradient into the latent prediction"
        );
        assert!(
            grads.get(&logits).is_some(),
            "detached targets must still train the reliability logit"
        );
        Ok(())
    }

    #[test]
    fn masked_spatial_huber_is_zero_for_an_empty_mask() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::ones((1, 2, 1, 1), DType::F32, &device)?;
        let target = Tensor::zeros_like(&input)?;
        let mask = Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?;
        let loss = masked_spatial_huber(&input, &target, &mask)?.to_scalar::<f32>()?;
        assert_eq!(loss, 0.0);
        Ok(())
    }

    #[test]
    fn training_content_chain_binds_content_mask_bits() -> Result<()> {
        let mut rows = generate_curriculum("random_one_step", 17, 0, Split::Train)?;
        let sample = rows.remove(0);
        let mask = training_content_mask_from_provenance(&sample)?;
        let mut changed_mask = mask.clone();
        changed_mask.values[0] ^= 1;
        let first_digest =
            training_content_batch_digest(std::iter::once(&sample), std::iter::once(&mask))?;
        let second_digest = training_content_batch_digest(
            std::iter::once(&sample),
            std::iter::once(&changed_mask),
        )?;
        let first = training_content_hash_append([0; 32], first_digest);
        let second = training_content_hash_append([0; 32], second_digest);
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn foundation_v2_training_content_chain_resumes_at_batch_boundary() -> Result<()> {
        let config = MixedStreamConfig {
            batch_size: 20,
            seed: 0xDA7A_0007,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        };
        let total_steps = 5usize;
        let digests = (0..total_steps as u64)
            .map(|batch_index| {
                let batch = compose_mixed_stream_batch(
                    &config,
                    batch_index as f32 / total_steps as f32,
                    batch_index,
                    V5DataSplit::Train,
                )?;
                training_content_batch_digest(batch.transitions(), batch.content_masks())
            })
            .collect::<Result<Vec<_>>>()?;
        let uninterrupted = digests
            .iter()
            .copied()
            .fold([0; 32], training_content_hash_append);

        let paused = digests[..2]
            .iter()
            .copied()
            .fold([0; 32], training_content_hash_append);
        // TrainerState persists this field as the same fixed byte array. Round
        // trip it before continuing so the test covers the checkpoint seam.
        let persisted: [u8; 32] = serde_json::from_slice(&serde_json::to_vec(&paused)?)?;
        let resumed = digests[2..]
            .iter()
            .copied()
            .fold(persisted, training_content_hash_append);
        assert_eq!(resumed, uninterrupted);
        Ok(())
    }

    fn resume_test_config(output_dir: PathBuf) -> TrainConfig {
        TrainConfig {
            lessons: vec!["sequential".into()],
            steps_per_lesson: 2,
            physical_batch: 2,
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projections: 2,
            sigreg_knots: 3,
            checkpoint_every_steps: 0,
            profile_update: 1,
            output_dir,
            ..TrainConfig::default()
        }
    }

    fn checkpoint_test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tofy-{label}-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    fn write_fake_checkpoint_bundle(bundle: &Path, step: u64, foundation_v2: bool) -> Result<()> {
        fs::create_dir_all(bundle)?;
        let mut safetensors = Vec::from((2u64).to_le_bytes());
        safetensors.extend_from_slice(b"{}");
        fs::write(bundle.join("model.safetensors"), &safetensors)?;
        fs::write(bundle.join("optimizer.safetensors"), b"optimizer")?;
        write_json_atomic(
            &bundle.join("trainer_state.json"),
            &serde_json::json!({"global_step": step}),
        )?;
        write_json_atomic(
            &bundle.join("config.json"),
            &serde_json::json!({
                "recipe": if foundation_v2 { "foundation_v2" } else { "legacy_experimental" }
            }),
        )?;
        if foundation_v2 {
            fs::write(bundle.join("ema.safetensors"), &safetensors)?;
            write_json_atomic(
                &bundle.join("gate_history.json"),
                &Vec::<serde_json::Value>::new(),
            )?;
        }
        let artifacts = checkpoint_required_artifacts(bundle)?
            .iter()
            .map(|name| sha256_file(&bundle.join(name)))
            .collect::<Result<Vec<_>>>()?;
        write_json_atomic(
            &bundle.join("bundle-manifest.json"),
            &CheckpointBundleManifest {
                schema: "p2.checkpoint_bundle.v1".into(),
                global_step: step,
                artifacts,
                parameter_groups: BTreeMap::new(),
            },
        )?;
        verify_checkpoint_bundle(bundle)
    }

    fn checkpoint_gate_metrics(value: f64) -> GateSupportMetrics {
        GateSupportMetrics {
            samples: 1,
            population_fingerprint: "sha256:test".into(),
            content_mask_fingerprint: Some("sha256:masks".into()),
            evidence_class: "selection_only".into(),
            changed_transitions: 1,
            changed_pixels: 1,
            foreground_pixels: 1,
            improvement_fraction: Some(1.0),
            shuffled_action_changed_pixel_ratio: Some(0.0),
            shuffled_action_rows: 1,
            shuffled_action_eligible_rows: 1,
            shuffled_action_changed_tuples: 1,
            shuffled_action_outcome_changing_tuples: None,
            foreground_reconstruction_accuracy: Some(1.0),
            one_step_changed_exact: Some(value),
            one_step_full_exact: Some(value),
            one_step_raw_full_exact: Some(value),
            one_step_composed_changed_exact: Some(value),
            one_step_all_rows_exact: Some(value),
            false_edit_rate: Some(0.0),
            padding_false_edit_rate: Some(0.0),
            raw_false_edit_rate: Some(0.0),
            raw_padding_false_edit_rate: Some(0.0),
            population_contract: "test".into(),
        }
    }

    fn floor_gate_metrics(foreground: f64, shuffled: f64) -> GateSupportMetrics {
        // Denominators large enough that floor violations in these tests are
        // statistically significant under the v6 noise margins.
        GateSupportMetrics {
            shuffled_action_changed_pixel_ratio: Some(shuffled),
            shuffled_action_outcome_changing_tuples: Some(2_000),
            foreground_reconstruction_accuracy: Some(foreground),
            foreground_pixels: 20_000,
            changed_transitions: 400,
            ..checkpoint_gate_metrics(1.0)
        }
    }

    fn shuffled_gate_metrics_with_rows(shuffled: f64, rows: usize) -> GateSupportMetrics {
        GateSupportMetrics {
            shuffled_action_outcome_changing_tuples: Some(rows),
            ..floor_gate_metrics(1.0, shuffled)
        }
    }

    #[test]
    fn foundation_v2_improving_foreground_floor_replays_s5_without_abort() -> Result<()> {
        let mut history = Vec::new();
        for (step, foreground) in [0.5077, 0.5500, 0.5672, 0.5742, 0.5887, 0.5985]
            .into_iter()
            .enumerate()
        {
            let evaluation = foundation_v2_gate_evaluation(
                4_096 + step as u64 * 1_024,
                floor_gate_metrics(foreground, 0.0),
                Some(1.0),
                Some(1.0),
                &history,
            )?;
            history.push(evaluation);
            assert!(!foundation_v2_gate_history_aborts(&history));
        }
        let foreground = history
            .last()
            .and_then(|evaluation| {
                evaluation
                    .gates
                    .iter()
                    .find(|gate| gate.name == "foreground_reconstruction")
            })
            .expect("foreground-reconstruction gate");
        assert!(!foreground.passed);
        assert!(foreground.abort_exempt);
        assert!(foreground
            .abort_exemption_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("higher is better")));
        Ok(())
    }

    #[test]
    fn foundation_v2_foreground_floor_plateau_or_decline_aborts() -> Result<()> {
        for trajectory in [[0.589, 0.589, 0.589], [0.589, 0.575, 0.560]] {
            let mut history = Vec::new();
            for (step, foreground) in trajectory.into_iter().enumerate() {
                let evaluation = foundation_v2_gate_evaluation(
                    8_192 + step as u64 * 1_024,
                    floor_gate_metrics(foreground, 0.0),
                    Some(1.0),
                    Some(1.0),
                    &history,
                )?;
                history.push(evaluation);
            }
            assert!(foundation_v2_gate_history_aborts(&history));
        }
        Ok(())
    }

    #[test]
    fn foundation_v2_shuffled_floor_improves_only_when_decreasing() -> Result<()> {
        for (trajectory, aborts) in [
            ([0.99, 0.985, 0.98, 0.975, 0.97], false),
            ([0.955, 0.96, 0.97, 0.98, 0.99], true),
        ] {
            let mut history = Vec::new();
            for (step, shuffled) in trajectory.into_iter().enumerate() {
                let evaluation = foundation_v2_gate_evaluation(
                    4_096 + step as u64 * 1_024,
                    floor_gate_metrics(1.0, shuffled),
                    Some(1.0),
                    Some(1.0),
                    &history,
                )?;
                history.push(evaluation);
            }
            assert_eq!(foundation_v2_gate_history_aborts(&history), aborts);
            let shuffled = history
                .last()
                .and_then(|evaluation| {
                    evaluation
                        .gates
                        .iter()
                        .find(|gate| gate.name == "shuffled_action_ratio")
                })
                .expect("shuffled-action gate");
            assert!(!shuffled.passed);
            assert_eq!(shuffled.abort_exempt, !aborts);
            if !aborts {
                assert!(shuffled
                    .abort_exemption_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("lower is better")));
            }
        }
        Ok(())
    }

    /// Replays run s8's full recorded gate history: the untrained step-1024
    /// checkpoint (composed 0.0, near-zero false edits because it never
    /// edits) must not latch promotion; the peak real candidate at step
    /// 20480 must be selected.
    #[test]
    fn foundation_v2_degenerate_copy_model_cannot_latch_best_selection() {
        let history: Vec<FoundationV2GateEvaluation> = [
            (1024, true, 0.0, 0.296875, 0.001153402537, 0.0),
            (
                2048,
                true,
                0.1242937853,
                0.041015625,
                0.2967128028,
                0.0001878099109,
            ),
            (
                3072,
                true,
                0.1666666667,
                0.048828125,
                0.226314055,
                0.0008257751695,
            ),
            (
                4096,
                false,
                0.1525423729,
                0.072265625,
                0.1644834404,
                0.0007272117437,
            ),
            (
                5120,
                false,
                0.1581920904,
                0.044921875,
                0.1536496952,
                0.0008748116997,
            ),
            (
                6144,
                true,
                0.2033898305,
                0.052734375,
                0.1384906904,
                0.0008890322935,
            ),
            (
                7168,
                true,
                0.2966101695,
                0.0625,
                0.1362250783,
                0.0007806615616,
            ),
            (
                8192,
                false,
                0.3276836158,
                0.064453125,
                0.1351952546,
                0.0006404170851,
            ),
            (
                9216,
                false,
                0.3870056497,
                0.052734375,
                0.1328884495,
                0.000646791834,
            ),
            (
                10240,
                true,
                0.3926553672,
                0.05859375,
                0.1393969352,
                0.0007532011047,
            ),
            (
                11264,
                true,
                0.4491525424,
                0.056640625,
                0.1356071841,
                0.0007522203741,
            ),
            (
                12288,
                true,
                0.4548022599,
                0.05859375,
                0.1279452958,
                0.00075369147,
            ),
            (
                13312,
                true,
                0.4689265537,
                0.0625,
                0.1231257209,
                0.0007728157168,
            ),
            (
                14336,
                true,
                0.4604519774,
                0.06640625,
                0.1211896523,
                0.0007904688677,
            ),
            (
                15360,
                true,
                0.4576271186,
                0.072265625,
                0.1150107102,
                0.0007791904657,
            ),
            (
                16384,
                true,
                0.4576271186,
                0.05078125,
                0.1139808865,
                0.0007448648946,
            ),
            (
                17408,
                true,
                0.4661016949,
                0.04296875,
                0.1108914154,
                0.0007953725207,
            ),
            (
                18432,
                true,
                0.4802259887,
                0.052734375,
                0.1073076289,
                0.0007688927944,
            ),
            (
                19456,
                true,
                0.488700565,
                0.04296875,
                0.1037650354,
                0.0008061605574,
            ),
            (
                20480,
                true,
                0.5197740113,
                0.048828125,
                0.1058658758,
                0.000823323343,
            ),
            (
                21504,
                true,
                0.5141242938,
                0.044921875,
                0.1051655957,
                0.0008247944389,
            ),
            (22528, true, 0.5, 0.05078125, 0.09976931949, 0.0008110642104),
            (
                23552,
                true,
                0.5084745763,
                0.05078125,
                0.09729774263,
                0.0008502934346,
            ),
            (
                24576,
                true,
                0.5141242938,
                0.044921875,
                0.09853353106,
                0.0008826575446,
            ),
        ]
        .into_iter()
        .map(|(step, armed_ok, composed, all_rows, fe, pad_fe)| {
            let metrics = GateSupportMetrics {
                one_step_composed_changed_exact: Some(composed),
                one_step_all_rows_exact: Some(all_rows),
                false_edit_rate: Some(fe),
                padding_false_edit_rate: Some(pad_fe),
                ..checkpoint_gate_metrics(composed)
            };
            FoundationV2GateEvaluation {
                step,
                metrics,
                running_best_before: None,
                running_best_after: Some(composed),
                gates: vec![FoundationV2GateResult {
                    name: "one_step_collapse".into(),
                    passed: armed_ok,
                    measured: Some(composed),
                    threshold: "replay".into(),
                    abort_exempt: false,
                    abort_exemption_reason: None,
                    floor: None,
                    noise_margin: None,
                }],
                diagnostics: vec![],
            }
        })
        .collect();
        assert_eq!(
            foundation_v2_selected_best_step(PromotionMetric::ComposedExactGuarded, &history),
            Some(20_480)
        );
        // The false-edit guard still binds between real candidates: a higher
        // composed value with a doubled false-edit rate is not promoted.
        let mut guarded = history.clone();
        let mut regressed = guarded.last().expect("history").clone();
        regressed.step = 25_600;
        regressed.metrics.one_step_composed_changed_exact = Some(0.60);
        regressed.metrics.false_edit_rate = Some(0.25);
        guarded.push(regressed);
        assert_eq!(
            foundation_v2_selected_best_step(PromotionMetric::ComposedExactGuarded, &guarded),
            Some(20_480)
        );
    }

    /// Isolated non-finite-gradient skips stay within budget; sixteen within
    /// one short rolling window or three consecutive skips must abort.
    #[test]
    fn foundation_v2_nonfinite_skip_budget() {
        assert!(foundation_v2_nonfinite_skip_exhausted(&[]).is_none());
        assert!(foundation_v2_nonfinite_skip_exhausted(&[10_613]).is_none());
        assert!(foundation_v2_nonfinite_skip_exhausted(&[100, 102, 103]).is_none());
        let scattered: Vec<u64> = (0..16).map(|i| i * 100).collect();
        assert!(foundation_v2_nonfinite_skip_exhausted(&scattered).is_none());
        let under_cap: Vec<u64> = (0..15).map(|i| 1_000 + i * 4).collect();
        assert!(foundation_v2_nonfinite_skip_exhausted(&under_cap).is_none());
        let capped: Vec<u64> = (0..16).map(|i| 1_000 + i * 4).collect();
        assert!(foundation_v2_nonfinite_skip_exhausted(&capped)
            .is_some_and(|reason| reason.contains("budget")));
        let mut boundary: Vec<u64> = (0..15).map(|i| 1_000 + i * 2).collect();
        boundary.push(1_063);
        assert!(foundation_v2_nonfinite_skip_exhausted(&boundary).is_some());
        let mut outside_boundary = boundary;
        *outside_boundary.last_mut().unwrap() = 1_064;
        assert!(foundation_v2_nonfinite_skip_exhausted(&outside_boundary).is_none());
        assert!(foundation_v2_nonfinite_skip_exhausted(&[50, 100, 101, 102])
            .is_some_and(|reason| reason.contains("consecutive")));

        // Regression for bundle-s8 through the failed broad-window repairs:
        // 30 lineage skips remain below the truly local burst budget.
        let bundle_s8 = [
            10_613, 11_015, 11_599, 11_934, 12_040, 12_120, 12_501, 12_553, 12_581, 12_585, 12_838,
            12_848, 12_895, 12_911, 13_055, 13_062, 13_073, 13_075, 13_077, 13_116, 13_133, 13_137,
            13_155, 13_158, 13_166, 13_171, 13_181, 13_183, 13_190, 13_191,
        ];
        assert!(foundation_v2_nonfinite_skip_exhausted(&bundle_s8).is_none());
    }

    fn composed_gate_metrics(composed: f64, shuffled: f64) -> GateSupportMetrics {
        GateSupportMetrics {
            one_step_composed_changed_exact: Some(composed),
            ..floor_gate_metrics(1.0, shuffled)
        }
    }

    /// Replays arm s7: the shuffled floor fails at arming, then fails again
    /// while strictly improving. Policy v5 exempts the improving failure; a
    /// later worsening failure still aborts.
    #[test]
    fn foundation_v2_improving_shuffled_floor_replays_s7_without_abort() -> Result<()> {
        let trajectory = [1.0, 0.9732, 0.9754, 0.9904, 0.9567];
        let mut history = Vec::new();
        for (index, shuffled) in trajectory.into_iter().enumerate() {
            let evaluation = foundation_v2_gate_evaluation(
                1_024 + index as u64 * 1_024,
                floor_gate_metrics(1.0, shuffled),
                Some(1.0),
                Some(1.0),
                &history,
            )?;
            history.push(evaluation);
            assert!(!foundation_v2_gate_history_aborts(&history));
        }
        let shuffled = history
            .last()
            .and_then(|evaluation| {
                evaluation
                    .gates
                    .iter()
                    .find(|gate| gate.name == "shuffled_action_ratio")
            })
            .expect("shuffled-action gate");
        assert!(!shuffled.passed);
        assert!(shuffled.abort_exempt);
        // Under v6, sustained significant regression still aborts: three
        // consecutive worsening, significant failures.
        for (step, shuffled) in [(6_144u64, 0.97), (7_168, 0.98), (8_192, 0.99)] {
            let evaluation = foundation_v2_gate_evaluation(
                step,
                floor_gate_metrics(1.0, shuffled),
                Some(1.0),
                Some(1.0),
                &history,
            )?;
            history.push(evaluation);
        }
        assert!(foundation_v2_gate_history_aborts(&history));
        Ok(())
    }

    /// Replays the s8 step-5120 abort under v6: on 115 outcome-changing
    /// tuples the one-sided 95% noise margin at the 0.95 floor is ~0.033, so
    /// violations of 0.0021 and 0.0256 are within measurement noise — they
    /// block promotion but must not abort, even three in a row. A truly
    /// action-blind arm (ratio 1.0, violation 0.05 > margin) still aborts.
    #[test]
    fn foundation_v2_shuffled_floor_noise_replays_s8_without_abort() -> Result<()> {
        let mut history = Vec::new();
        for (step, shuffled) in [
            (1_024u64, 1.0),
            (2_048, 0.927632),
            (3_072, 0.979592),
            (4_096, 0.952153),
            (5_120, 0.975610),
            (6_144, 0.970200),
        ] {
            let evaluation = foundation_v2_gate_evaluation(
                step,
                shuffled_gate_metrics_with_rows(shuffled, 115),
                Some(1.0),
                Some(1.0),
                &history,
            )?;
            history.push(evaluation);
            assert!(
                !foundation_v2_gate_history_aborts(&history),
                "aborted at step {step} on a noise-level violation"
            );
        }
        let latest = history.last().expect("history");
        let shuffled = latest
            .gates
            .iter()
            .find(|gate| gate.name == "shuffled_action_ratio")
            .expect("shuffled gate");
        assert!(
            !shuffled.passed,
            "noise-level failure must still block promotion"
        );

        let mut blind_history = Vec::new();
        for step in [4_096u64, 5_120, 6_144] {
            let evaluation = foundation_v2_gate_evaluation(
                step,
                shuffled_gate_metrics_with_rows(1.0, 115),
                Some(1.0),
                Some(1.0),
                &blind_history,
            )?;
            blind_history.push(evaluation);
        }
        assert!(foundation_v2_gate_history_aborts(&blind_history));
        Ok(())
    }

    /// Replays arm s6: the composed metric peaks during warmup (0.1638 at
    /// step 2048) and re-balances below 0.8x that peak by arming. Policy v5
    /// anchors the floor to armed evaluations only, so the arm survives; a
    /// genuine armed-regime collapse still fails.
    #[test]
    fn foundation_v2_composed_floor_ignores_warmup_peak_replays_s6() -> Result<()> {
        let trajectory = [0.0, 0.1638, 0.1469, 0.096, 0.1045];
        let mut history = Vec::new();
        for (index, composed) in trajectory.into_iter().enumerate() {
            let step = 1_024 + index as u64 * 1_024;
            let prior_composed_best = history
                .iter()
                .filter(|evaluation: &&FoundationV2GateEvaluation| {
                    evaluation.step >= FOUNDATION_V2_GATE_WARMUP_STEPS
                })
                .filter_map(|evaluation| evaluation.metrics.one_step_composed_changed_exact)
                .filter(|value: &f64| value.is_finite())
                .reduce(f64::max);
            let evaluation = foundation_v2_gate_evaluation(
                step,
                composed_gate_metrics(composed, 0.9),
                Some(1.0),
                prior_composed_best,
                &history,
            )?;
            let composed_gate = evaluation
                .gates
                .iter()
                .find(|gate| gate.name == "composed_changed_exact_collapse")
                .expect("composed collapse gate");
            assert!(
                composed_gate.passed,
                "step {step}: composed {composed} unexpectedly failed ({})",
                composed_gate.threshold
            );
            history.push(evaluation);
            assert!(!foundation_v2_gate_history_aborts(&history));
        }
        // An armed-regime collapse (>20% below the armed best) still fails.
        let evaluation = foundation_v2_gate_evaluation(
            6_144,
            composed_gate_metrics(0.05, 0.9),
            Some(1.0),
            Some(0.1045),
            &history,
        )?;
        let composed_gate = evaluation
            .gates
            .iter()
            .find(|gate| gate.name == "composed_changed_exact_collapse")
            .expect("composed collapse gate");
        assert!(!composed_gate.passed);
        Ok(())
    }

    fn checkpoint_gate_evaluation(
        step: u64,
        value: f64,
        passed: bool,
    ) -> FoundationV2GateEvaluation {
        FoundationV2GateEvaluation {
            step,
            metrics: checkpoint_gate_metrics(value),
            running_best_before: None,
            running_best_after: Some(value),
            gates: vec![FoundationV2GateResult {
                name: "one_step_collapse".into(),
                passed,
                measured: Some(value),
                threshold: "test".into(),
                abort_exempt: false,
                abort_exemption_reason: None,
                floor: None,
                noise_margin: None,
            }],
            diagnostics: vec![],
        }
    }

    #[test]
    fn foundation_v2_old_gate_history_defaults_to_non_exempt_replay() -> Result<()> {
        let original = vec![
            checkpoint_gate_evaluation(1_024, 0.5, true),
            checkpoint_gate_evaluation(2_048, 0.4, false),
            checkpoint_gate_evaluation(3_072, 0.35, false),
            checkpoint_gate_evaluation(4_096, 0.3, false),
        ];
        let expected_abort = foundation_v2_gate_history_aborts(&original);
        let expected_best =
            foundation_v2_selected_best_step(PromotionMetric::ChangedExact, &original);
        let mut old_format = serde_json::to_value(&original)?;
        for evaluation in old_format
            .as_array_mut()
            .expect("serialized gate-history array")
        {
            for gate in evaluation["gates"]
                .as_array_mut()
                .expect("serialized gates")
            {
                let gate = gate.as_object_mut().expect("serialized gate");
                gate.remove("abort_exempt");
                gate.remove("abort_exemption_reason");
            }
        }

        let replayed: Vec<FoundationV2GateEvaluation> = serde_json::from_value(old_format)?;
        assert_eq!(replayed, original);
        assert_eq!(foundation_v2_gate_history_aborts(&replayed), expected_abort);
        assert!(expected_abort);
        assert_eq!(
            foundation_v2_selected_best_step(PromotionMetric::ChangedExact, &replayed),
            expected_best
        );
        assert_eq!(expected_best, Some(1_024));
        Ok(())
    }

    #[test]
    fn foundation_v2_exempt_floor_failure_still_blocks_promotion() -> Result<()> {
        let mut history = Vec::new();
        for (step, foreground) in [0.570, 0.580, 0.590].into_iter().enumerate() {
            let evaluation = foundation_v2_gate_evaluation(
                6_144 + step as u64 * 1_024,
                floor_gate_metrics(foreground, 0.0),
                Some(1.0),
                Some(1.0),
                &history,
            )?;
            history.push(evaluation);
        }
        let evaluation = history.last().expect("latest gate evaluation");
        let foreground = evaluation
            .gates
            .iter()
            .find(|gate| gate.name == "foreground_reconstruction")
            .expect("foreground-reconstruction gate");
        assert!(foreground.abort_exempt);
        assert!(foundation_v2_promotion_improved(
            PromotionMetric::ChangedExact,
            None,
            &[],
            &evaluation.metrics,
        ));
        assert!(!foundation_v2_evaluation_improves(
            PromotionMetric::ChangedExact,
            None,
            &[],
            evaluation,
        ));
        Ok(())
    }

    #[test]
    fn foundation_v2_collapse_gate_cannot_be_abort_exempt() {
        let mut history = vec![
            checkpoint_gate_evaluation(1_024, 0.5, false),
            checkpoint_gate_evaluation(2_048, 0.45, false),
            checkpoint_gate_evaluation(3_072, 0.4, false),
        ];
        let latest = history
            .last_mut()
            .and_then(|evaluation| evaluation.gates.last_mut())
            .expect("latest collapse gate");
        latest.abort_exempt = true;
        latest.abort_exemption_reason = Some("forged trend exemption".into());
        assert!(foundation_v2_gate_history_aborts(&history));
    }

    fn checkpoint_foundation_state(
        gate_history: Vec<FoundationV2GateEvaluation>,
    ) -> FoundationV2TrainerState {
        FoundationV2TrainerState {
            total_steps: 24_576,
            ep_weight: 0.01,
            ep_gradient_budget: vec![],
            gate_history,
            best_changed_exact: None,
            rollout_enabled: true,
            loss_sums: FoundationV2LossMeans::default(),
            loss_steps: 0,
            permanent_checkpoints: vec![],
            event_label_census: EventLabelCensus::default(),
            event_label_census_complete: true,
            mechanism_history: vec![],
            profiles_published: vec![],
            rollout_zero_loss_consecutive_steps: 0,
            gate_population_identity: None,
        }
    }

    #[test]
    fn checkpoint_save_adopts_verified_target_and_preserves_corrupt_target_on_replace() -> Result<()>
    {
        let root = checkpoint_test_root("checkpoint-collision");
        let mut cfg = resume_test_config(root.clone());
        cfg.steps_per_lesson = 1;
        let varmap = VarMap::new();
        varmap.data().lock().unwrap().insert(
            "test.weight".into(),
            Var::from_tensor(&Tensor::ones((1,), DType::F32, &Device::Cpu)?)?,
        );
        let optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            adam_params(&cfg),
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?;
        let state = TrainerState {
            schema: TRAINER_STATE_SCHEMA.into(),
            contract: TrainingContract::from(&cfg),
            global_step: 0,
            lesson_index: 0,
            step_in_lesson: 0,
            optimizer_step: 0,
            completed_lessons: vec![],
            active_sums: LessonLossMeans::default(),
            parameter_names: optimizer.parameter_names(),
            training_population_hash: default_training_population_hash(),
            training_content_hash: [0; 32],
            training_population_rows: 0,
            batch_schedule_migrations: vec![],
            nonfinite_skipped_updates: vec![],
            profile: ProfileState::Pending,
            gradient_pressure: None,
            gradient_pressure_samples: vec![],
            foundation_v2: None,
        };

        let checkpoint = save_training_checkpoint(&varmap, &optimizer, None, &state, &cfg)?;
        fs::write(checkpoint.join("adoption-sentinel"), b"preserve")?;
        fs::remove_file(root.join("checkpoints/latest.json"))?;
        assert_eq!(
            save_training_checkpoint(&varmap, &optimizer, None, &state, &cfg)?,
            checkpoint
        );
        assert_eq!(fs::read(checkpoint.join("adoption-sentinel"))?, b"preserve");
        let latest: LatestCheckpoint = read_json(&root.join("checkpoints/latest.json"))?;
        assert_eq!(latest.global_step, 0);

        fs::write(checkpoint.join("model.safetensors"), b"corrupt")?;
        save_training_checkpoint(&varmap, &optimizer, None, &state, &cfg)?;
        verify_checkpoint_bundle(&checkpoint)?;
        let backup = fs::read_dir(root.join("checkpoints"))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".step-000000000000.corrupt-"))
            })
            .expect("corrupt checkpoint was preserved");
        assert_eq!(fs::read(backup.join("model.safetensors"))?, b"corrupt");
        assert!(backup.join("adoption-sentinel").is_file());
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn best_resume_reconciliation_repairs_missing_and_mismatched_rotation() -> Result<()> {
        let root = checkpoint_test_root("best-reconcile");
        let cfg = TrainConfig {
            recipe: TrainingRecipe::FoundationV2,
            output_dir: root.clone(),
            ..TrainConfig::default()
        };
        let selected = checkpoint_step_directory(&cfg, 1_024);
        write_fake_checkpoint_bundle(&selected, 1_024, true)?;
        let foundation =
            checkpoint_foundation_state(vec![checkpoint_gate_evaluation(1_024, 0.5, true)]);
        assert!(foundation_v2_verified_best_export(
            &cfg,
            PromotionMetric::ChangedExact,
            &foundation.gate_history
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to fall back"));

        reconcile_foundation_v2_best_checkpoint(&cfg, &foundation)?;
        assert_eq!(checkpoint_step(&root.join("checkpoints/best"))?, 1_024);
        assert!(foundation_v2_verified_best_export(
            &cfg,
            PromotionMetric::ChangedExact,
            &foundation.gate_history
        )?
        .is_some());

        let wrong = checkpoint_step_directory(&cfg, 2_048);
        write_fake_checkpoint_bundle(&wrong, 2_048, true)?;
        publish_best_checkpoint(&cfg, &wrong)?;
        assert_eq!(checkpoint_step(&root.join("checkpoints/best"))?, 2_048);
        reconcile_foundation_v2_best_checkpoint(&cfg, &foundation)?;
        assert_eq!(checkpoint_step(&root.join("checkpoints/best"))?, 1_024);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn permanent_resume_reconciliation_recopies_missing_and_corrupt_bundle() -> Result<()> {
        let root = checkpoint_test_root("permanent-reconcile");
        let cfg = TrainConfig {
            recipe: TrainingRecipe::FoundationV2,
            output_dir: root.clone(),
            ..TrainConfig::default()
        };
        let source = checkpoint_step_directory(&cfg, 2_048);
        write_fake_checkpoint_bundle(&source, 2_048, true)?;
        let permanent = root.join("checkpoints/permanent/step-000000002048");
        let mut recorded = vec![permanent.clone()];
        reconcile_foundation_v2_permanent_checkpoints(&cfg, &mut recorded)?;
        verify_checkpoint_bundle(&permanent)?;

        fs::write(permanent.join("ema.safetensors"), b"corrupt")?;
        reconcile_foundation_v2_permanent_checkpoints(&cfg, &mut recorded)?;
        verify_checkpoint_bundle(&permanent)?;
        assert!(
            fs::read_dir(root.join("checkpoints/permanent"))?.any(|entry| {
                entry
                    .ok()
                    .and_then(|entry| entry.file_name().into_string().ok())
                    .is_some_and(|name| name.starts_with(".step-000000002048.corrupt-"))
            })
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn aborted_resume_requires_explicit_override_for_history_or_marker() -> Result<()> {
        let root = checkpoint_test_root("abort-marker");
        fs::create_dir_all(&root)?;
        let mut cfg = TrainConfig {
            output_dir: root.clone(),
            ..TrainConfig::default()
        };
        // Gate policy v6 aborts only after three consecutive counting
        // failures of the same gate.
        let foundation = checkpoint_foundation_state(vec![
            checkpoint_gate_evaluation(1_024, 0.1, false),
            checkpoint_gate_evaluation(2_048, 0.1, false),
            checkpoint_gate_evaluation(3_072, 0.1, false),
        ]);
        assert!(ensure_foundation_v2_resume_not_aborted(&cfg, Some(&foundation)).is_err());
        cfg.resume_after_abort = true;
        ensure_foundation_v2_resume_not_aborted(&cfg, Some(&foundation))?;

        cfg.resume_after_abort = false;
        persist_foundation_v2_abort_marker(&cfg, 2_048, Path::new("checkpoint"))?;
        assert!(ensure_foundation_v2_resume_not_aborted(&cfg, None).is_err());
        cfg.resume_after_abort = true;
        ensure_foundation_v2_resume_not_aborted(&cfg, None)?;
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn gate_identity_is_backfilled_only_before_gate_history_exists() -> Result<()> {
        let identity = GatePopulationIdentity {
            rows_sha256: "sha256:rows".into(),
            masks_sha256: "sha256:masks".into(),
            provenance_sha256: "sha256:provenance".into(),
            policy_schema: FOUNDATION_V2_GATE_POLICY_SCHEMA.into(),
        };
        let mut fresh = checkpoint_foundation_state(vec![]);
        reconcile_foundation_v2_gate_population_identity(&mut fresh, identity.clone())?;
        assert_eq!(fresh.gate_population_identity, Some(identity.clone()));

        let mut historical =
            checkpoint_foundation_state(vec![checkpoint_gate_evaluation(1_024, 0.5, true)]);
        let error = reconcile_foundation_v2_gate_population_identity(&mut historical, identity)
            .expect_err("history without identity must fail closed");
        assert!(error.to_string().contains("gate history"));
        Ok(())
    }

    #[test]
    fn checkpoint_manifest_and_latest_pointer_are_complete_and_step_bound() -> Result<()> {
        let root = checkpoint_test_root("checkpoint-manifest");
        let checkpoints = root.join("checkpoints");
        let bundle = checkpoints.join("step-000000000007");
        write_fake_checkpoint_bundle(&bundle, 7, true)?;

        let mut manifest: CheckpointBundleManifest =
            read_json(&bundle.join("bundle-manifest.json"))?;
        manifest
            .artifacts
            .retain(|artifact| artifact.path != "ema.safetensors");
        write_json_atomic(&bundle.join("bundle-manifest.json"), &manifest)?;
        assert!(verify_checkpoint_bundle(&bundle)
            .unwrap_err()
            .to_string()
            .contains("artifact set mismatch"));
        write_fake_checkpoint_bundle(&bundle, 7, true)?;

        write_json_atomic(
            &checkpoints.join("latest.json"),
            &LatestCheckpoint {
                schema: TRAINER_STATE_SCHEMA.into(),
                directory: "../step-000000000007".into(),
                global_step: 7,
            },
        )?;
        assert!(resolve_resume_checkpoint(&checkpoints)
            .unwrap_err()
            .to_string()
            .contains("plain child name"));
        write_json_atomic(
            &checkpoints.join("latest.json"),
            &LatestCheckpoint {
                schema: TRAINER_STATE_SCHEMA.into(),
                directory: "step-000000000007".into(),
                global_step: 8,
            },
        )?;
        assert!(resolve_resume_checkpoint(&checkpoints)
            .unwrap_err()
            .to_string()
            .contains("disagrees with restored trainer state"));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn foundation_v2_loss_means_reports_clipped_fraction() {
        let sums = FoundationV2LossMeans {
            clipped_fraction: 3.0,
            ..FoundationV2LossMeans::default()
        };
        assert_eq!(foundation_v2_loss_means(&sums, 4).clipped_fraction, 0.75);
    }

    /// Compare state produced by two independently executed runs.
    ///
    /// candle's CPU backend reduces over rayon, so float accumulation order —
    /// and therefore the low bits — depends on how the work happens to be
    /// split at runtime. Model weights and lesson losses are already compared
    /// with a tolerance for exactly this reason; asserting *bitwise* equality
    /// on the optimizer moments and `active_sums` made this test flaky (the
    /// same binary passes or fails run to run on identical input) while
    /// claiming to check resume fidelity. The property that actually matters is
    /// that resumed state matches to within accumulated float error — a
    /// genuinely dropped or mis-restored moment is orders of magnitude larger
    /// than this tolerance and still fails.
    fn assert_close_f32(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "length mismatch at {what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            // 5e-5: revision-3 objective arithmetic (fused unimix gather,
            // PAD-inclusive gate BCE, halved EP quadrature) lands rayon
            // accumulation noise at ~1.3e-5 on conv moments, just past the
            // old 1e-5 floor; a dropped moment still exceeds this by orders.
            let tol = 5e-5 * x.abs().max(y.abs()).max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "optimizer mismatch at {what}[{i}]: {x} vs {y}"
            );
        }
    }

    fn assert_loss_means_close(a: &LessonLossMeans, b: &LessonLossMeans, eps: f64) {
        for (name, x, y) in [
            ("total", a.total, b.total),
            ("next_latent", a.next_latent, b.next_latent),
            ("rollout", a.rollout, b.rollout),
            ("sigreg_raw", a.sigreg_raw, b.sigreg_raw),
            ("sigreg_bounded", a.sigreg_bounded, b.sigreg_bounded),
            ("event", a.event, b.event),
            ("q", a.q, b.q),
            ("prefix", a.prefix, b.prefix),
            ("reliability", a.reliability, b.reliability),
        ] {
            assert!(
                (x - y).abs() <= eps * x.abs().max(y.abs()).max(1.0),
                "active_sums {name} diverged: {x} vs {y}"
            );
        }
    }

    fn lessons_match_within_eps(a: &[LessonReport], b: &[LessonReport], eps: f64) {
        assert_eq!(a.len(), b.len(), "lesson report count");
        for (left, right) in a.iter().zip(b) {
            assert_eq!(left.lesson, right.lesson);
            assert_eq!(left.curriculum, right.curriculum);
            assert_eq!(left.steps, right.steps);
            let dl = &left.mean_losses;
            let dr = &right.mean_losses;
            assert!((dl.total - dr.total).abs() < eps, "total loss");
            assert!((dl.next_latent - dr.next_latent).abs() < eps, "next_latent");
            assert!((dl.rollout - dr.rollout).abs() < eps, "rollout");
            assert!((dl.sigreg_raw - dr.sigreg_raw).abs() < eps, "sigreg_raw");
            assert!(
                (dl.sigreg_bounded - dr.sigreg_bounded).abs() < eps,
                "sigreg_bounded"
            );
            assert!((dl.event - dr.event).abs() < eps, "event");
            assert!((dl.q - dr.q).abs() < eps, "q");
            assert!((dl.prefix - dr.prefix).abs() < eps, "prefix");
            assert!((dl.reliability - dr.reliability).abs() < eps, "reliability");
        }
    }

    fn loaded_model_values(cfg: &TrainConfig, path: &Path) -> Result<Vec<(String, Vec<f32>)>> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = WorldModel::new(cfg.model_config(), vb)?;
        varmap.load(path)?;
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<_> = data.keys().cloned().collect();
        names.sort();
        names
            .into_iter()
            .map(|name| {
                let values = data[&name].as_tensor().flatten_all()?.to_vec1::<f32>()?;
                Ok((name, values))
            })
            .collect()
    }

    fn toy_frame(fill: u8) -> ArcFrame {
        ArcFrame::new(
            FRAME_SIDE as u16,
            FRAME_SIDE as u16,
            vec![fill; FRAME_SIDE * FRAME_SIDE],
        )
        .unwrap()
    }

    fn toy_sample(pix: u8) -> TransitionSample {
        TransitionSample {
            current: toy_frame(pix),
            next: toy_frame((pix + 1) % 16),
            action: ArcAction::new(1, None, None).unwrap(),
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: Some(false),
            goal_failed: None,
            exhausted: Some(false),
            split: Split::Train,
            family: "test".into(),
            seed: 0,
            episode_id: 0,
            transition_index: 0,
            provenance: crate::p2::data::TransitionProvenance::full_frame(
                0,
                0,
                Split::Train,
                "test",
            ),
            oracle_latent: None,
            context: Vec::new(),
        }
    }

    #[test]
    fn index_staging_and_event_mask() -> Result<()> {
        let device = Device::Cpu;
        let mut coordinate_sample = toy_sample(7);
        coordinate_sample.action = ArcAction::new(6, Some(63), Some(21))?;
        let samples = vec![toy_sample(3), coordinate_sample];
        let batch = batch_from_samples(&samples, &device)?;
        assert_eq!(batch.frames.dims(), &[2, 1, 64, 64]);
        let f0 = batch.frames.get(0)?;
        let pix = f0.flatten_all()?.to_vec1::<u8>()?;
        let status_start = (FRAME_SIDE - 1) * FRAME_SIDE;
        assert!(pix.iter().all(|&v| v == 3));
        let model_pix = batch.model_frames.get(0)?.flatten_all()?.to_vec1::<u8>()?;
        assert_eq!(&model_pix[..status_start], &pix[..status_start]);
        assert!(model_pix[status_start..].iter().all(|&v| v == 0));

        let targets = batch.event_targets.to_vec2::<f32>()?;
        let mask = batch.event_mask.to_vec2::<f32>()?;
        // goal_failed is None → mask 0
        assert_eq!(mask[0][2], 0.0);
        assert_eq!(mask[0][0], 1.0);
        assert_eq!(targets[0][0], 0.0);
        assert_eq!(
            batch.action_coords.to_vec2::<f32>()?,
            vec![vec![0.0, 0.0], vec![1.0, 21.0 / 63.0],]
        );
        Ok(())
    }

    #[test]
    fn operator_tensorization_maps_train_held_out_and_missing_provenance() -> Result<()> {
        let device = Device::Cpu;
        let mut samples = vec![toy_sample(1), toy_sample(2), toy_sample(3)];
        samples[0].provenance.operator = Some(EpisodeOperator {
            family: OperatorFamily::Toggle,
            agent_color: 7,
            primary_color: 11,
            secondary_color: 4,
            empty_color: 0,
        });
        samples[1].provenance.operator = Some(EpisodeOperator {
            family: OperatorFamily::SwapRegion,
            agent_color: 7,
            primary_color: 11,
            secondary_color: 4,
            empty_color: 0,
        });
        let rows = operator_conditioning_from_samples(&samples, &device)?.to_vec2::<f32>()?;

        assert_eq!(rows[0][OperatorFamily::Toggle.conditioning_token()], 1.0);
        assert_eq!(rows[0][OPERATOR_FAMILY_VOCAB + 7], 1.0);
        assert_eq!(rows[0][OPERATOR_FAMILY_VOCAB + PALETTE_SIZE + 11], 1.0);
        assert_eq!(rows[0][OPERATOR_FAMILY_VOCAB + 2 * PALETTE_SIZE + 4], 1.0);
        for row in [&rows[1], &rows[2]] {
            assert_eq!(row[OPERATOR_FAMILY_UNKNOWN], 1.0);
            assert!(row[OPERATOR_FAMILY_VOCAB..]
                .iter()
                .all(|value| *value == 0.0));
        }
        Ok(())
    }

    #[test]
    fn v6_batch_trains_a_whole_frame_decoder_and_rejects_a_legacy_one() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = TrainConfig::default();
        cfg.apply_foundation_v2_recipe();
        let mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 64,
                seed: 0x7607,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            0.0,
            0,
            V5DataSplit::Train,
        )?;
        let objective = FoundationV2ObjectiveConfig {
            ep_weight: 0.01,
            sigreg_projections: 8,
            sigreg_knots: 5,
            sigreg_seed: 1,
            q_mse_threshold: cfg.q_mse_threshold,
            rollout_enabled: false,
            split_ce_weighting: Default::default(),
            split_ce_changed_budget: None,
            capture_mechanism_seams: false,
        };
        let host = prepare_foundation_v2_batch_host(&mixed)?;
        assert_eq!(host.gameplay_rows, FRAME_SIDE);
        assert_eq!(host.content_values.len(), 64 * FRAME_SIDE * FRAME_SIDE);

        let mut model_cfg = cfg.model_config();
        model_cfg.world_core_v6 = true;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            model_cfg,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let losses = foundation_v2_training_loss(&model, &mixed, &device, objective)?;
        assert!(losses
            .total
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?
            .is_finite());

        let legacy_varmap = VarMap::new();
        let legacy = WorldModel::new(
            cfg.model_config(),
            VarBuilder::from_varmap(&legacy_varmap, DType::F32, &device),
        )?;
        let err = match foundation_v2_training_loss(&legacy, &mixed, &device, objective) {
            Ok(_) => panic!("a 63-row decoder cannot supervise whole-frame rows"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("supervises 64 rows"));
        Ok(())
    }

    #[test]
    fn v6_batch_conditions_unknown_and_keeps_row_63_in_model_frames() -> Result<()> {
        let mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 128,
                seed: 0x7606,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            0.0,
            0,
            V5DataSplit::Train,
        )?;
        let host = prepare_foundation_v2_batch_host(&mixed)?;
        for row in host
            .operator_conditioning
            .chunks_exact(OPERATOR_CONDITION_DIM)
        {
            assert_eq!(row[OPERATOR_FAMILY_UNKNOWN], 1.0);
            assert_eq!(
                row.iter().sum::<f32>(),
                1.0,
                "v6 rows carry no colour triple"
            );
        }
        assert_eq!(host.model_frames, host.frames);
        assert_eq!(host.model_next_frames, host.next_frames);
        assert!(host.latent_content_mask.iter().all(|&value| value == 1.0));
        Ok(())
    }

    #[test]
    fn operator_tensorization_uses_conjugated_toggle_colors() -> Result<()> {
        let device = Device::Cpu;
        let config = MixedStreamConfig {
            batch_size: 128,
            seed: 73,
            ..MixedStreamConfig::default()
        };
        let sample = (0..64)
            .find_map(|batch_index| {
                let batch =
                    compose_mixed_stream_batch(&config, 1.0, batch_index, V5DataSplit::Train)
                        .ok()?;
                batch.into_samples().into_iter().find(|sample| {
                    sample.provenance.operator.family == OperatorFamily::Toggle
                        && sample.provenance.augmentation.color_permutation[palette::AGENT as usize]
                            != palette::AGENT
                })
            })
            .expect("deterministic search finds a color-permuted Toggle row");
        let operator = sample.provenance.operator;
        let permutation = sample.provenance.augmentation.color_permutation;
        assert_eq!(operator.agent_color, permutation[palette::AGENT as usize]);
        assert_eq!(
            operator.primary_color,
            permutation[palette::SWITCH_BASE as usize]
        );
        assert_eq!(
            operator.secondary_color,
            permutation[(palette::SWITCH_BASE + 1) as usize]
        );
        assert_eq!(sample.transition.provenance.operator, Some(operator));
        let row =
            operator_conditioning_from_samples(std::slice::from_ref(&sample.transition), &device)?
                .to_vec2::<f32>()?
                .remove(0);
        for (slot, color) in [
            operator.agent_color,
            operator.primary_color,
            operator.secondary_color,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                row[OPERATOR_FAMILY_VOCAB + slot * PALETTE_SIZE + color as usize],
                1.0
            );
            assert!(
                sample.transition.current.pixels.contains(&color)
                    || sample.transition.next.pixels.contains(&color),
                "conjugated color {color} is absent from the recolored row"
            );
        }
        Ok(())
    }

    #[test]
    fn ordered_trace_staging_matches_full_batch_fields_exactly() -> Result<()> {
        let device = Device::Cpu;
        let samples = vec![toy_sample(1), toy_sample(2), toy_sample(3)];
        let full = batch_from_samples(&samples, &device)?;
        let trace = ordered_trace_from_samples(&samples, &device)?;
        assert_eq!(
            full.frames.flatten_all()?.to_vec1::<u8>()?,
            trace.frames.flatten_all()?.to_vec1::<u8>()?
        );
        assert_eq!(
            full.next_frames.flatten_all()?.to_vec1::<u8>()?,
            trace.next_frames.flatten_all()?.to_vec1::<u8>()?
        );
        assert_eq!(
            full.actions.flatten_all()?.to_vec1::<u32>()?,
            trace.actions.flatten_all()?.to_vec1::<u32>()?
        );
        assert_eq!(
            full.action_coords.flatten_all()?.to_vec1::<f32>()?,
            trace.action_coords.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn ordered_sigreg_windows_never_cross_trace_boundaries() -> Result<()> {
        let mut samples = (0..4).map(toy_sample).collect::<Vec<_>>();
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.transition_index = index as u64;
        }
        let mut other = (0..8).map(toy_sample).collect::<Vec<_>>();
        for (index, sample) in other.iter_mut().enumerate() {
            sample.episode_id = 1;
            sample.transition_index = index as u64;
        }
        // Each identity component independently breaks an ordered run.
        other[2].seed += 1;
        other[4].family = "different-family".into();
        other[6].transition_index = 9;
        samples.extend(other);

        let windows = ordered_sigreg_windows(&samples, 3)?.expect("one complete window");
        assert_eq!(windows.windows, 1);
        assert_eq!(windows.row_indices, vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    fn temporal_sigreg_is_invariant_to_window_local_offsets() -> Result<()> {
        let device = Device::Cpu;
        let windows = OrderedSigregWindows {
            window: 2,
            windows: 2,
            row_indices: vec![0, 2, 1, 3],
        };
        let cfg = TrainConfig {
            physical_batch: 4,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 0,
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            ..TrainConfig::default()
        };
        let base = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0,
            ],
            (4, 1, 2, 2),
            &device,
        )?;
        let offsets = Tensor::from_vec(
            vec![100.0f32; 8]
                .into_iter()
                .chain(vec![200.0f32; 8])
                .collect::<Vec<_>>(),
            (4, 1, 2, 2),
            &device,
        )?;
        let shifted = base.add(&offsets)?;
        let actual = sigreg_stack_for_ordered_windows(
            &base,
            &windows,
            SigregTarget::TemporalResidual,
            &cfg,
            7,
        )?;
        let expected = sigreg_stack_for_ordered_windows(
            &shifted,
            &windows,
            SigregTarget::TemporalResidual,
            &cfg,
            7,
        )?;
        assert_eq!(actual.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn tc_sigreg_arms_share_ordered_rows_and_encoder_shape() -> Result<()> {
        let device = Device::Cpu;
        let mut samples = (0..8).map(toy_sample).collect::<Vec<_>>();
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.transition_index = index as u64;
        }
        let selected = ordered_sigreg_windows(&samples, 4)?.expect("two ordered windows");
        assert_eq!(selected.row_indices, vec![0, 4, 1, 5, 2, 6, 3, 7]);
        let latents = Tensor::zeros((8, 3, 4, 4), DType::F32, &device)?;
        let cfg = TrainConfig {
            physical_batch: 8,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 0,
            sigreg_temporal_window: 4,
            ..TrainConfig::default()
        };
        let control = sigreg_stack_for_ordered_windows(
            &latents,
            &selected,
            SigregTarget::Marginal,
            &cfg,
            11,
        )?;
        let treatment = sigreg_stack_for_ordered_windows(
            &latents,
            &selected,
            SigregTarget::TemporalResidual,
            &cfg,
            11,
        )?;
        // Target selection is after the shared `B×C×H×W` encoder result.
        assert_eq!(control.dims(), treatment.dims());
        assert_eq!(control.dims(), &[4, 8, 3]);
        Ok(())
    }

    #[test]
    fn global_tc_rows_match_window_population_and_center_each_trace() -> Result<()> {
        let device = Device::Cpu;
        let windows = OrderedSigregWindows {
            window: 2,
            windows: 2,
            row_indices: vec![0, 2, 1, 3],
        };
        let cfg = TrainConfig {
            physical_batch: 4,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 0,
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            sigreg_global_mix: 1.0,
            ..TrainConfig::default()
        };
        let latents = Tensor::from_vec(
            (0..32).map(|value| value as f32).collect::<Vec<_>>(),
            (4, 2, 2, 2),
            &device,
        )?;
        let rows = sigreg_global_stack_for_ordered_windows(&latents, &windows, &cfg, 19)?;
        assert_eq!(rows.dims(), &[2, 2, 2]);
        let centered = rows.sum(0)?.to_vec2::<f32>()?;
        assert!(centered.iter().flatten().all(|value| value.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn zero_global_mix_is_exactly_the_original_cell_objective() -> Result<()> {
        let device = Device::Cpu;
        let windows = OrderedSigregWindows {
            window: 2,
            windows: 2,
            row_indices: vec![0, 2, 1, 3],
        };
        let cfg = TrainConfig {
            physical_batch: 4,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 0,
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            sigreg_global_mix: 0.0,
            sigreg_projections: 3,
            sigreg_knots: 5,
            ..TrainConfig::default()
        };
        let latents = Tensor::from_vec(
            (0..64)
                .map(|value| (value as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            (4, 4, 2, 2),
            &device,
        )?;
        let seed = 29;
        let stack = sigreg_stack_for_ordered_windows(
            &latents,
            &windows,
            SigregTarget::TemporalResidual,
            &cfg,
            seed,
        )?;
        let expected_raw =
            sigreg_epps_pulley_seeded(&stack, cfg.sigreg_projections, cfg.sigreg_knots, seed)?;
        let expected_bounded = bounded_sigreg_loss(&expected_raw)?;
        let (actual_raw, actual_bounded) =
            sigreg_losses_for_ordered_windows(&latents, &windows, &cfg, seed)?;
        assert_eq!(
            actual_raw.to_scalar::<f32>()?.to_bits(),
            expected_raw.to_scalar::<f32>()?.to_bits()
        );
        assert_eq!(
            actual_bounded.to_scalar::<f32>()?.to_bits(),
            expected_bounded.to_scalar::<f32>()?.to_bits()
        );
        Ok(())
    }

    #[test]
    fn global_tc_is_invariant_to_window_local_spatial_offsets() -> Result<()> {
        let device = Device::Cpu;
        let windows = OrderedSigregWindows {
            window: 2,
            windows: 2,
            row_indices: vec![0, 2, 1, 3],
        };
        let cfg = TrainConfig {
            physical_batch: 4,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 0,
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            sigreg_global_mix: 0.5,
            ..TrainConfig::default()
        };
        let base = Tensor::from_vec(
            (0..32).map(|value| value as f32).collect::<Vec<_>>(),
            (4, 2, 2, 2),
            &device,
        )?;
        // Original rows 0/1 belong to trace 0 and 2/3 to trace 1.
        let offsets = Tensor::from_vec(
            [100.0f32, 100.0, 200.0, 200.0]
                .into_iter()
                .flat_map(|offset| std::iter::repeat_n(offset, 8))
                .collect::<Vec<_>>(),
            (4, 2, 2, 2),
            &device,
        )?;
        let shifted = base.add(&offsets)?;
        let actual = sigreg_global_stack_for_ordered_windows(&base, &windows, &cfg, 23)?;
        let expected = sigreg_global_stack_for_ordered_windows(&shifted, &windows, &cfg, 23)?;
        let actual = actual.to_vec3::<f32>()?;
        let expected = expected.to_vec3::<f32>()?;
        for (actual, expected) in actual
            .iter()
            .flatten()
            .flatten()
            .zip(expected.iter().flatten().flatten())
        {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
        Ok(())
    }

    #[test]
    fn temporal_sigreg_config_requires_window_and_control_geometry() {
        let invalid_window = TrainConfig {
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 1,
            sigreg_spatial: true,
            ..TrainConfig::default()
        };
        assert!(invalid_window.validate().is_err());
        let valid_post_rms_unpooled = TrainConfig {
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            sigreg_spatial: true,
            sigreg_spatial_pool: false,
            ..TrainConfig::default()
        };
        assert!(valid_post_rms_unpooled.validate().is_ok());
        assert!(TrainConfig {
            sigreg_target: SigregTarget::Marginal,
            sigreg_temporal_window: 1,
            ..TrainConfig::default()
        }
        .validate()
        .is_ok());
        assert!(TrainConfig {
            sigreg_target: SigregTarget::TemporalResidual,
            sigreg_temporal_window: 2,
            sigreg_spatial: true,
            sigreg_global_mix: 1.01,
            ..TrainConfig::default()
        }
        .validate()
        .is_err());
        assert!(TrainConfig {
            sigreg_target: SigregTarget::Marginal,
            sigreg_global_mix: 0.5,
            ..TrainConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn older_serialized_config_loads_marginal_tc_defaults() -> Result<()> {
        let mut value = serde_json::to_value(TrainConfig::default())?;
        let object = value.as_object_mut().expect("config object");
        object.remove("sigreg_target");
        object.remove("sigreg_temporal_window");
        object.remove("sigreg_global_mix");
        let loaded: TrainConfig = serde_json::from_value(value)?;
        assert_eq!(loaded.sigreg_target, SigregTarget::Marginal);
        assert_eq!(loaded.sigreg_temporal_window, 8);
        assert_eq!(loaded.sigreg_global_mix, 0.0);
        Ok(())
    }

    #[test]
    fn legacy_training_contract_without_tc_fields_is_rejected() -> Result<()> {
        let contract = TrainingContract::from(&TrainConfig::default());
        let mut value = serde_json::to_value(contract)?;
        let object = value.as_object_mut().expect("training contract object");
        object.remove("sigreg_target");
        object.remove("sigreg_temporal_window");
        object.remove("sigreg_global_mix");
        assert!(serde_json::from_value::<TrainingContract>(value).is_err());
        Ok(())
    }

    #[test]
    fn deterministic_init_repeats() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            frame_side: FRAME_SIDE,
            hidden_dim: 16,
            action_dim: 4,
            goal_dim: GOAL_FEATURES_DIM,
            inner_steps: 1,
            outer_steps: 1,
            num_events: DEFAULT_NUM_EVENTS,
            ..Default::default()
        };
        let mut maps = Vec::new();
        for _ in 0..2 {
            let map = VarMap::new();
            let vb = VarBuilder::from_varmap(&map, DType::F32, &device);
            let _model = WorldModel::new(cfg.clone(), vb)?;
            reinit_varmap_deterministic(&map, 42)?;
            maps.push(map);
        }
        let a = maps[0].data().lock().unwrap();
        let b = maps[1].data().lock().unwrap();
        let mut names: Vec<_> = a.keys().cloned().collect();
        names.sort();
        for name in names {
            let va = a
                .get(&name)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?;
            let vb = b
                .get(&name)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(va, vb, "mismatch at {name}");
            if name.ends_with("bias") {
                assert!(va.iter().all(|v| *v == 0.0), "bias not zero: {name}");
            }
        }
        Ok(())
    }

    #[test]
    fn v3_residual_projection_starts_exactly_zero() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TrainConfig {
            world_core_v2: true,
            world_core_v3: true,
            spatial_action_field: true,
            spatial_action_residual: true,
            sigreg_weight: 0.0,
            lessons: vec!["factual_branches".into()],
            branch_learning: BranchLearningConfig {
                enabled: true,
                ..BranchLearningConfig::default()
            },
            ..TrainConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        zero_v3_spatial_residual(&varmap)?;
        let data = varmap.data().lock().unwrap();
        let residuals = data
            .iter()
            .filter(|(name, _)| name.starts_with("spatial_action_proj."))
            .collect::<Vec<_>>();
        assert!(!residuals.is_empty());
        for (name, var) in residuals {
            let max = var.as_tensor().abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(max, 0.0, "{name}");
        }
        Ok(())
    }

    #[test]
    fn exact_model_load_rejects_missing_tensor() -> Result<()> {
        let device = Device::Cpu;
        let target = VarMap::new();
        target.data().lock().unwrap().insert(
            "first".into(),
            Var::from_tensor(&Tensor::zeros((2,), DType::F32, &device)?)?,
        );
        target.data().lock().unwrap().insert(
            "second".into(),
            Var::from_tensor(&Tensor::zeros((3,), DType::F32, &device)?)?,
        );
        let checkpoint = VarMap::new();
        checkpoint.data().lock().unwrap().insert(
            "first".into(),
            Var::from_tensor(&Tensor::ones((2,), DType::F32, &device)?)?,
        );
        let path = std::env::temp_dir().join(format!(
            "tofy-p2-model-missing-tensor-{}.safetensors",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        checkpoint.save(&path)?;
        let err = load_varmap_exact(&target, &path)
            .expect_err("exact model load must reject a missing tensor");
        assert!(err.to_string().contains("missing"), "{err:#}");
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn exact_model_load_rejects_extra_shape_and_dtype_mismatches() -> Result<()> {
        let device = Device::Cpu;
        let target = VarMap::new();
        target.data().lock().unwrap().insert(
            "weight".into(),
            Var::from_tensor(&Tensor::zeros((2,), DType::F32, &device)?)?,
        );
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-model-exact-mismatch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;

        let extra = VarMap::new();
        extra.data().lock().unwrap().insert(
            "weight".into(),
            Var::from_tensor(&Tensor::ones((2,), DType::F32, &device)?)?,
        );
        extra.data().lock().unwrap().insert(
            "unexpected".into(),
            Var::from_tensor(&Tensor::ones((1,), DType::F32, &device)?)?,
        );
        let path = root.join("extra.safetensors");
        extra.save(&path)?;
        let err = load_varmap_exact(&target, &path).expect_err("extra tensor must reject");
        assert!(err.to_string().contains("extra"), "{err:#}");

        let wrong_shape = VarMap::new();
        wrong_shape.data().lock().unwrap().insert(
            "weight".into(),
            Var::from_tensor(&Tensor::ones((3,), DType::F32, &device)?)?,
        );
        let path = root.join("shape.safetensors");
        wrong_shape.save(&path)?;
        let err = load_varmap_exact(&target, &path).expect_err("shape mismatch must reject");
        assert!(err.to_string().contains("shape mismatch"), "{err:#}");

        let wrong_dtype = VarMap::new();
        wrong_dtype.data().lock().unwrap().insert(
            "weight".into(),
            Var::from_tensor(&Tensor::ones((2,), DType::F64, &device)?)?,
        );
        let path = root.join("dtype.safetensors");
        wrong_dtype.save(&path)?;
        let err = load_varmap_exact(&target, &path).expect_err("dtype mismatch must reject");
        assert!(err.to_string().contains("dtype mismatch"), "{err:#}");

        fs::remove_dir_all(root)?;
        Ok(())
    }

    /// The parallel wave scheduler must not perturb batch contents: training
    /// determinism and the resume contract both depend on step N seeing exactly
    /// `concat(gen(N), gen(N+1), …)[..batch]`.
    #[test]
    fn parallel_collect_batch_matches_sequential() -> Result<()> {
        fn sequential(
            curriculum: &str,
            seed: u64,
            start_episode: u64,
            batch: usize,
            split: Split,
        ) -> Result<Vec<TransitionSample>> {
            let mut out = Vec::with_capacity(batch);
            let mut ep = start_episode;
            while out.len() < batch {
                for s in generate_curriculum(curriculum, seed, ep, split)? {
                    out.push(s);
                    if out.len() == batch {
                        break;
                    }
                }
                ep = ep.wrapping_add(1);
            }
            Ok(out)
        }

        // Fixed and variable yield per episode, and a batch that is not a wave multiple.
        for curriculum in ["random_one_step", "sequential"] {
            for (start, batch) in [(0u64, 64usize), (7, 100), (13, 257)] {
                let want = sequential(curriculum, 1, start, batch, Split::Train)?;
                let got = collect_batch(curriculum, 1, start, batch, Split::Train)?;
                assert_eq!(got.len(), batch);
                assert_eq!(got, want, "{curriculum} start={start} batch={batch}");
            }
        }
        Ok(())
    }

    /// Profiling probe, not an assertion. Run with
    /// `cargo test --release --lib episode_cache_steady_state -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn episode_cache_steady_state_cost() -> Result<()> {
        use std::time::Instant;
        for curriculum in ["random_one_step", "sequential", "p1c_falsification"] {
            let mut cache = EpisodeCache::default();
            let t0 = Instant::now();
            cache.collect(curriculum, 1, 0, 1024, Split::Train)?;
            let cold = t0.elapsed().as_secs_f64() * 1e3;

            let steps = 20u64;
            let t1 = Instant::now();
            for step in 1..=steps {
                cache.collect(curriculum, 1, step, 1024, Split::Train)?;
            }
            let warm = t1.elapsed().as_secs_f64() * 1e3 / steps as f64;
            println!("{curriculum:<20} cold={cold:>8.1}ms  warm={warm:>8.1}ms/step");
        }
        Ok(())
    }

    /// The sliding window is memoization, not resampling: every step must see the
    /// same batch it would have seen with a cold cache, including across a lesson
    /// switch (which changes curriculum) and a resume (which starts cache-cold).
    #[test]
    fn episode_cache_matches_uncached_batches() -> Result<()> {
        let batch = 100;
        let mut cache = EpisodeCache::default();

        for step in 0..12u64 {
            let want = collect_batch("random_one_step", 1, step, batch, Split::Train)?;
            let got = cache.collect("random_one_step", 1, step, batch, Split::Train)?;
            assert_eq!(got, want, "step {step}");
        }
        // Lesson switch: different curriculum through the same cache.
        for step in 12..16u64 {
            let want = collect_batch("sequential", 1, step, batch, Split::Train)?;
            let got = cache.collect("sequential", 1, step, batch, Split::Train)?;
            assert_eq!(got, want, "post-switch step {step}");
        }
        // Resume: a fresh cache mid-stream must reproduce the same batch.
        let mut cold = EpisodeCache::default();
        assert_eq!(
            cold.collect("sequential", 1, 15, batch, Split::Train)?,
            collect_batch("sequential", 1, 15, batch, Split::Train)?
        );

        // Rollout traces served from the window must match the scanning search.
        let mut cache = EpisodeCache::default();
        for step in 0..6u64 {
            let _ = cache.collect("sequential", 1, step, batch, Split::Train)?;
            let want = collect_rollout_trace("sequential", 1, step, Split::Train)?;
            let got = match cache.rollout_trace("sequential", 1, step, Split::Train) {
                Ok(trace) => trace,
                Err(resume) => collect_rollout_trace("sequential", 1, resume, Split::Train)?,
            };
            assert_eq!(got, want, "rollout trace step {step}");
        }
        Ok(())
    }

    /// Saturated logits are the failure mode that killed the overnight run at
    /// step ~2200: with every `q` target at 1.0 the head drifts past the f32
    /// point where `sigmoid` rounds to exactly 1.0.
    #[test]
    fn bce_with_logits_survives_saturated_logits() -> Result<()> {
        let device = Device::Cpu;
        let logit_var = Var::from_tensor(&Tensor::new(
            &[-200.0f32, -20.0, 0.0, 20.0, 200.0],
            &device,
        )?)?;
        let logits = logit_var.as_tensor();
        let ones = Tensor::ones_like(logits)?;
        let zeros = Tensor::zeros_like(logits)?;

        // The naive formulation candle ships is NaN here; ours must not be.
        let naive =
            candle_nn::loss::binary_cross_entropy_with_logit(logits, &ones)?.to_scalar::<f32>()?;
        assert!(
            naive.is_nan(),
            "expected the naive form to be NaN, got {naive}"
        );

        for targets in [&ones, &zeros] {
            let loss = bce_with_logits(logits, targets)?.to_scalar::<f32>()?;
            assert!(loss.is_finite(), "loss not finite: {loss}");
            let elem = bce_with_logits_elem(logits, targets)?;
            let grads = elem.sum_all()?.backward()?;
            let g = grads
                .get(&logit_var)
                .expect("logit gradient")
                .to_vec1::<f32>()?;
            assert!(
                g.iter().all(|v| v.is_finite()),
                "gradient not finite: {g:?}"
            );
        }

        // Matches the closed form on values where the naive version is safe.
        let mid = Tensor::new(&[-2.0f32, -0.5, 0.5, 3.0], &device)?;
        let t = Tensor::new(&[0.0f32, 1.0, 0.0, 1.0], &device)?;
        let got = bce_with_logits(&mid, &t)?.to_scalar::<f32>()?;
        let want =
            candle_nn::loss::binary_cross_entropy_with_logit(&mid, &t)?.to_scalar::<f32>()?;
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        Ok(())
    }

    #[test]
    fn per_micro_backward_matches_summed_loss_gradients() -> Result<()> {
        use candle_core::Module;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lin = candle_nn::linear(3, 2, vb.pp("lin"))?;
        let x = Tensor::new(&[0.5f32, -1.0, 0.25], &device)?.reshape((1, 3))?;
        let terms = [
            lin.forward(&x)?.sum_all()?,
            lin.forward(&x.affine(2.0, 0.0)?)?.sum_all()?,
        ];
        let accum_f = 2.0;
        let summed = terms[0].add(&terms[1])?.affine(1.0 / accum_f, 0.0)?;
        let summed_grads = summed.backward()?;

        let mut accumulated: Option<GradStore> = None;
        for term in terms {
            let micro_grads = term.affine(1.0 / accum_f, 0.0)?.backward()?;
            accumulate_parameter_gradients(&mut accumulated, micro_grads, &varmap)?;
        }
        let micro_grads = accumulated.expect("micro gradients");

        let w = lin.weight();
        let g_sum = summed_grads.get(w).expect("summed grad").to_vec2::<f32>()?;
        let g_micro = micro_grads.get(w).expect("micro grad").to_vec2::<f32>()?;
        assert_eq!(g_sum, g_micro);
        Ok(())
    }

    #[test]
    fn one_optimizer_step_changes_finite_loss() -> Result<()> {
        let cfg = TrainConfig {
            steps_per_lesson: 1,
            lessons: vec!["dynamics".into()],
            physical_batch: 2,
            grad_accum: 1,
            output_dir: std::env::temp_dir().join(format!("tofy-p2-train-{}", std::process::id())),
            ..TrainConfig::default()
        };
        let _ = fs::remove_dir_all(&cfg.output_dir);

        let device = resolve_device(&cfg.device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let mut opt = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: cfg.lr,
                ..ParamsAdamW::default()
            },
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?;

        let samples = collect_batch(
            "random_one_step",
            cfg.seed,
            0,
            cfg.physical_batch,
            Split::Train,
        )?;
        let batch = batch_from_samples(&samples, &device)?;
        let depth = RecursionDepth {
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
        };
        let weights = lesson_loss_weights("dynamics", &cfg, 0, 0);
        let before = leworld_loss(&model, &batch, &cfg, depth, cfg.seed, weights)?;
        let v0 = ensure_finite("before", &before.total)?;
        let mut grads = before.total.backward()?;
        crate::p2::optimizer::clip_gradients_gpu(&mut grads, &varmap, MAX_GRAD_NORM)?;
        opt.step(&grads)?;
        let after = leworld_loss(
            &model,
            &batch,
            &cfg,
            depth,
            cfg.seed.wrapping_add(1),
            weights,
        )?;
        let v1 = ensure_finite("after", &after.total)?;
        assert!(v0.is_finite() && v1.is_finite());
        assert!(
            v1 < v0 || (v1 - v0).abs() > 1e-8,
            "expected loss to decrease or change, got {v0} -> {v1}"
        );
        let _ = fs::remove_dir_all(&cfg.output_dir);
        Ok(())
    }

    #[test]
    fn event_stop_gradient_updates_observer_head_and_goal_projection() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TrainConfig {
            physical_batch: 2,
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projections: 2,
            sigreg_knots: 3,
            stop_grad_event_y: true,
            ..TrainConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let mut first = toy_sample(3);
        first.goal_features.values[0] = 1.0;
        let mut second = toy_sample(7);
        second.goal_features.values[1] = 1.0;
        let batch = batch_from_samples(&[first, second], &device)?;
        let losses = leworld_loss(
            &model,
            &batch,
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            cfg.seed,
            LessonLossWeights {
                world: 1.0,
                rollout: 0.0,
                sigreg: 0.0,
                event: 1.0,
                q: 0.0,
                ptrm_rank: false,
                ptrm_rank_k: 1,
                prefix: 0.0,
                reliability: 0.0,
            },
        )?;
        let grads = losses.event.backward()?;
        let (event_weight, goal_weight) = {
            let data = varmap.data().lock().unwrap();
            (
                data["event_head.weight"].clone(),
                data["goal_proj.weight"].clone(),
            )
        };
        for (name, var) in [
            ("event_head.weight", &event_weight),
            ("goal_proj.weight", &goal_weight),
        ] {
            let grad = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("missing gradient for {name}"));
            let norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            assert!(
                norm > 0.0,
                "expected nonzero gradient for {name}, got {norm}"
            );
        }

        let before = event_weight.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let mut optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: cfg.lr,
                ..ParamsAdamW::default()
            },
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?;
        optimizer.step(&grads)?;
        let after = event_weight.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        assert_ne!(before, after, "event head parameters did not update");
        Ok(())
    }

    #[test]
    fn q_stop_gradient_updates_observer_head() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TrainConfig {
            physical_batch: 2,
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projections: 2,
            sigreg_knots: 3,
            stop_grad_q_y: true,
            ..TrainConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let batch = batch_from_samples(&[toy_sample(3), toy_sample(7)], &device)?;
        let losses = leworld_loss(
            &model,
            &batch,
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            cfg.seed,
            LessonLossWeights {
                world: 1.0,
                rollout: 0.0,
                sigreg: 0.0,
                event: 0.0,
                q: 1.0,
                ptrm_rank: false,
                ptrm_rank_k: 1,
                prefix: 0.0,
                reliability: 0.0,
            },
        )?;
        let grads = losses.q.backward()?;
        let q_weight = {
            let data = varmap.data().lock().unwrap();
            data["q_head.weight"].clone()
        };
        let grad = grads
            .get(q_weight.as_tensor())
            .expect("missing gradient for q_head.weight");
        let norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(norm > 0.0, "expected nonzero Q-head gradient, got {norm}");

        let before = q_weight.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let mut optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: cfg.lr,
                ..ParamsAdamW::default()
            },
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?;
        optimizer.step(&grads)?;
        let after = q_weight.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        assert_ne!(before, after, "Q head parameters did not update");
        Ok(())
    }

    #[test]
    fn projector_sigreg_updates_learned_projector() -> Result<()> {
        let device = Device::Cpu;
        let cfg = TrainConfig {
            physical_batch: 2,
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projections: 2,
            sigreg_knots: 3,
            sigreg_projector: true,
            sigreg_projector_dim: 6,
            ..TrainConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let batch = batch_from_samples(&[toy_sample(3), toy_sample(7)], &device)?;
        let losses = leworld_loss(
            &model,
            &batch,
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            cfg.seed,
            LessonLossWeights {
                world: 1.0,
                rollout: 0.0,
                sigreg: 1.0,
                event: 0.0,
                q: 0.0,
                ptrm_rank: false,
                ptrm_rank_k: 1,
                prefix: 0.0,
                reliability: 0.0,
            },
        )?;
        let grads = losses.sigreg_raw.backward()?;
        let projector = {
            let data = varmap.data().lock().unwrap();
            data["sigreg_projector.weight"].clone()
        };
        let grad = grads
            .get(projector.as_tensor())
            .expect("SIGReg must reach the learned projector");
        let norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(
            norm > 0.0,
            "expected nonzero projector gradient, got {norm}"
        );
        Ok(())
    }

    #[test]
    fn phased_lesson_weights_gate_auxiliary_losses() {
        let cfg = TrainConfig {
            event_weight: 0.1,
            q_weight: 0.1,
            rollout_weight: 0.1,
            steps_per_lesson: 100,
            phased_training: true,
            ..TrainConfig::default()
        };
        let dyn_w = lesson_loss_weights("dynamics", &cfg, 0, 0);
        assert_eq!(dyn_w.event, 0.0);
        assert_eq!(dyn_w.q, 0.0);
        assert_eq!(dyn_w.rollout, 0.0);
        assert_eq!(dyn_w.prefix, 0.0);
        assert_eq!(dyn_w.reliability, 0.0);
        assert!(!dyn_w.ptrm_rank);

        let seq_w = lesson_loss_weights("sequential", &cfg, 50, 1);
        assert_eq!(seq_w.event, 0.0);
        assert_eq!(seq_w.q, 0.0);
        assert!(seq_w.rollout > 0.0);
        assert!(!seq_w.ptrm_rank);
        assert!(lesson_loss_weights("sequential", &cfg, 50, 4).ptrm_rank);

        let q_w = lesson_loss_weights("q_calibration", &cfg, 0, 0);
        assert_eq!(q_w.event, 0.0);
        assert_eq!(q_w.q, 0.0);
        assert_eq!(q_w.rollout, 0.0);
        let q_w_late = lesson_loss_weights("q_calibration", &cfg, 99, 4);
        assert!((q_w_late.q - cfg.q_weight * 0.99).abs() < 1e-9);
        assert!((q_w_late.event - cfg.event_weight * 0.99).abs() < 1e-9);
        assert!(q_w_late.ptrm_rank);

        let fals_w = lesson_loss_weights("falsification", &cfg, 0, 0);
        assert_eq!(fals_w.event, 0.0);
        assert_eq!(fals_w.q, 0.0);
        assert_eq!(fals_w.rollout, 0.0);
        let fals_w_late = lesson_loss_weights("falsification", &cfg, 99, 4);
        assert!((fals_w_late.event - cfg.event_weight * 0.99).abs() < 1e-9);
        assert!(fals_w_late.ptrm_rank);
        assert_eq!(fals_w_late.ptrm_rank_k, 4);

        let exp_w = lesson_loss_weights("exploration", &cfg, 50, 0);
        assert_eq!(exp_w.event, 0.0);
        assert_eq!(exp_w.q, 0.0);
        assert_eq!(exp_w.prefix, 0.0);
        assert_eq!(exp_w.reliability, 0.0);
        assert!(!exp_w.ptrm_rank);

        let seq_w_late = lesson_loss_weights("sequential", &cfg, 99, 8);
        let ret_w = lesson_loss_weights("retarget", &cfg, 99, 8);
        assert!(ret_w.rollout > 0.0);
        assert!(ret_w.rollout < seq_w_late.rollout);
        assert!(ret_w.ptrm_rank);
    }

    #[test]
    fn full_v4_recipe_is_resolved_and_rejects_drift() -> Result<()> {
        let mut cfg = TrainConfig::default();
        cfg.apply_full_v4_recipe();
        cfg.validate()?;
        let resolved = cfg.resolved_experiment()?;
        assert_eq!(resolved.family, crate::p2::experiment::WorldCoreFamily::V4);
        assert_eq!(cfg.sigreg_projections, 1024);
        assert_eq!(cfg.sigreg_knots, 17);
        assert_eq!(cfg.sigreg_max_rows, 0);
        assert_eq!(cfg.consumer_readout, ConsumerReadoutTopology::SpatialQuery);

        cfg.sigreg_target = SigregTarget::TemporalResidual;
        assert!(cfg.validate().is_err());
        cfg.apply_full_v4_recipe();
        cfg.hidden_dim = 64;
        assert!(cfg.validate().is_err());
        Ok(())
    }

    #[test]
    fn full_v4_observer_lessons_disable_world_objectives() {
        let mut cfg = TrainConfig::default();
        cfg.apply_full_v4_recipe();
        for lesson in ["q_calibration", "falsification"] {
            let weights = lesson_loss_weights(lesson, &cfg, 1, 1);
            assert_eq!(weights.world, 0.0);
            assert_eq!(weights.sigreg, 0.0);
            assert_eq!(weights.rollout, 0.0);
            assert_eq!(weights.prefix, 0.0);
            assert!(!weights.ptrm_rank);
        }
    }

    #[test]
    fn full_v4_q_observer_has_no_world_core_gradients() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = TrainConfig::default();
        cfg.apply_full_v4_recipe();
        cfg.physical_batch = 2;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let batch = batch_from_samples(&[toy_sample(3), toy_sample(7)], &device)?;
        let losses = leworld_loss(
            &model,
            &batch,
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            cfg.seed,
            LessonLossWeights {
                world: 0.0,
                sigreg: 0.0,
                event: 0.0,
                q: 1.0,
                rollout: 0.0,
                prefix: 0.0,
                reliability: 0.0,
                ptrm_rank: false,
                ptrm_rank_k: 1,
            },
        )?;
        let grads = losses.total.backward()?;
        let data = varmap.data().lock().unwrap();
        assert!(grads.get(data["q_head.weight"].as_tensor()).is_some());
        let world_names = [
            "encoder.patch.weight",
            "block.c1.weight",
            "consumer_readout.query_score.weight",
            "exact_grounding_head.decoder.weight",
        ];
        for name in world_names {
            assert!(
                grads.get(data[name].as_tensor()).is_none(),
                "observer loss leaked a gradient into {name}"
            );
        }
        let world_before = world_names
            .iter()
            .map(|name| {
                data[*name]
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .map(|values| ((*name).to_string(), values))
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let q_before = data["q_head.weight"]
            .as_tensor()
            .flatten_all()?
            .to_vec1::<f32>()?;
        drop(data);
        let mut optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: cfg.lr,
                weight_decay: cfg.weight_decay,
                ..ParamsAdamW::default()
            },
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?;
        optimizer.step(&grads)?;
        let data = varmap.data().lock().unwrap();
        for (name, before) in world_before {
            let after = data[&name].as_tensor().flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(before, after, "observer optimizer step changed {name}");
        }
        let q_after = data["q_head.weight"]
            .as_tensor()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_ne!(
            q_before, q_after,
            "observer optimizer did not update Q head"
        );
        Ok(())
    }

    #[test]
    fn full_v4_world_objective_reaches_canonical_and_exact_grounding() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = TrainConfig::default();
        cfg.apply_full_v4_recipe();
        cfg.physical_batch = 2;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let mut coordinate = toy_sample(7);
        coordinate.action = ArcAction::new(6, Some(31), Some(47))?;
        let batch = batch_from_samples(&[toy_sample(3), coordinate], &device)?;
        let losses = leworld_loss(
            &model,
            &batch,
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            cfg.seed,
            LessonLossWeights {
                world: 1.0,
                sigreg: cfg.sigreg_weight,
                event: 0.0,
                q: 0.0,
                rollout: 0.0,
                prefix: 0.0,
                reliability: 0.0,
                ptrm_rank: false,
                ptrm_rank_k: 1,
            },
        )?;
        assert!(losses.total.to_scalar::<f32>()?.is_finite());
        assert_eq!(
            losses.sigreg_raw.to_scalar::<f32>()?,
            losses.sigreg_bounded.to_scalar::<f32>()?,
            "Full V4 must not silently cap EP"
        );
        let grads = losses.total.backward()?;
        let data = varmap.data().lock().unwrap();
        for name in [
            "encoder.patch.weight",
            "spatial_action_proj.weight",
            "consumer_readout.query_score.weight",
            "exact_grounding_head.decoder.weight",
        ] {
            let grad = grads
                .get(data[name].as_tensor())
                .unwrap_or_else(|| panic!("missing Full V4 gradient for {name}"));
            assert!(grad.sqr()?.sum_all()?.to_scalar::<f32>()? > 0.0);
        }
        Ok(())
    }

    #[test]
    fn effective_sigreg_max_rows_honours_configured_cap() {
        // `sigreg_max_rows` is authoritative, bounded only by the rows the
        // spatial stack actually has. A previous batch-keyed clamp pinned this
        // to 1024 for any physical_batch >= 128, so raising the batch gave
        // SIGReg no extra samples to estimate its statistic from.
        let cfg = TrainConfig {
            physical_batch: 128,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 4096,
            ..TrainConfig::default()
        };
        // pooled: 128 * 16 * 2 = 4096 available, cap 4096.
        assert_eq!(effective_sigreg_max_rows(&cfg), 4096);
        let unpooled = TrainConfig {
            sigreg_spatial_pool: false,
            ..cfg.clone()
        };
        // unpooled: 128 * 64 * 2 = 16384 available, so the cap binds.
        assert_eq!(effective_sigreg_max_rows(&unpooled), 4096);
        // Larger batches now actually reach more rows.
        let big = TrainConfig {
            physical_batch: 1024,
            sigreg_max_rows: 32768,
            ..cfg.clone()
        };
        assert_eq!(effective_sigreg_max_rows(&big), 32768);
        // A small explicit cap is still respected (tight-VRAM profile).
        let laptop = TrainConfig {
            physical_batch: 1024,
            sigreg_max_rows: 1024,
            ..cfg.clone()
        };
        assert_eq!(effective_sigreg_max_rows(&laptop), 1024);
        // Availability binds below the cap for small batches.
        let mid = TrainConfig {
            physical_batch: 64,
            sigreg_spatial: true,
            sigreg_max_rows: 4096,
            ..TrainConfig::default()
        };
        assert_eq!(effective_sigreg_max_rows(&mid), 2048);
        let loose = TrainConfig {
            physical_batch: 32,
            sigreg_spatial: true,
            sigreg_max_rows: 4096,
            ..TrainConfig::default()
        };
        assert_eq!(effective_sigreg_max_rows(&loose), 1024);
    }

    #[test]
    fn pre_rms_spatial_sigreg_uses_unpooled_raw_cells() -> Result<()> {
        let device = Device::Cpu;
        let normalized = Tensor::zeros((2, 3, 2, 2), DType::F32, &device)?;
        let raw_current = Tensor::ones((2, 3, 2, 2), DType::F32, &device)?;
        let raw_next = raw_current.affine(2.0, 0.0)?;
        let cfg = TrainConfig {
            physical_batch: 2,
            sigreg_spatial: true,
            sigreg_spatial_pool: false,
            sigreg_pre_rms_spatial: true,
            sigreg_max_rows: 0,
            ..TrainConfig::default()
        };
        cfg.validate()?;
        let stack = sigreg_stack_for_encoded_pair(
            &normalized,
            &normalized,
            &raw_current,
            &raw_next,
            None,
            &cfg,
            7,
        )?;
        assert_eq!(stack.dims(), &[16, 3]);
        let rows = stack.to_vec2::<f32>()?;
        assert!(rows[..8].iter().flatten().all(|value| *value == 1.0));
        assert!(rows[8..].iter().flatten().all(|value| *value == 2.0));
        Ok(())
    }

    #[test]
    fn pre_rms_spatial_sigreg_rejects_conflicting_geometry() {
        let invalid = [
            TrainConfig {
                sigreg_pre_rms_spatial: true,
                sigreg_spatial: false,
                sigreg_spatial_pool: false,
                ..TrainConfig::default()
            },
            TrainConfig {
                sigreg_pre_rms_spatial: true,
                sigreg_spatial: true,
                sigreg_spatial_pool: true,
                ..TrainConfig::default()
            },
            TrainConfig {
                sigreg_pre_rms_spatial: true,
                sigreg_spatial: true,
                sigreg_spatial_pool: false,
                sigreg_projector: true,
                ..TrainConfig::default()
            },
        ];
        for cfg in invalid {
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn sigreg_cap_retains_gradient_above_reported_limit() -> Result<()> {
        let device = Device::Cpu;
        let raw = Var::new(&[20_000f32], &device)?;
        let bounded = bounded_sigreg_loss(raw.as_tensor())?.sum_all()?;
        let grads = bounded.backward()?;
        let gradient = grads
            .get(raw.as_tensor())
            .expect("SIGReg cap must retain a gradient")
            .to_vec1::<f32>()?[0];
        assert!(
            gradient.is_finite() && gradient > 0.0,
            "expected positive finite gradient above the cap, got {gradient}"
        );
        Ok(())
    }

    #[test]
    fn loss_check_reports_constituent_before_non_finite_total() -> Result<()> {
        let device = Device::Cpu;
        let zero = Tensor::new(0f32, &device)?;
        let nan = Tensor::new(f32::NAN, &device)?;
        let losses = LossBreakdown {
            total: nan.clone(),
            next_latent: zero.clone(),
            sigreg_raw: zero.clone(),
            sigreg_bounded: zero.clone(),
            patch_grounding: zero.clone(),
            grounding_changed_patches: 0,
            grounding_unchanged_patches: 0,
            event: zero.clone(),
            q: zero.clone(),
            q_surprise: zero.clone(),
            ptrm_rank: zero.clone(),
            prefix: zero.clone(),
            reliability: zero.clone(),
            branch_total: zero.clone(),
            outcome_pull: zero.clone(),
            outcome_push: zero.clone(),
            action_recovery: zero.clone(),
            coordinate_recovery: zero.clone(),
            changed_margin: zero.clone(),
            spatial_variance: zero.clone(),
            spatial_covariance: zero.clone(),
            pooled_variance: zero.clone(),
            pooled_covariance: zero.clone(),
            displacement_variance: zero.clone(),
            displacement_covariance: zero.clone(),
            branch_audit: BranchLearningAudit::default(),
        };

        let tensors = training_loss_tensors(&losses, &zero, &nan, &nan);
        let error = checked_training_losses(&[tensors]).unwrap_err();
        assert!(
            error.to_string().contains("prefix_multi is not finite"),
            "expected the originating component, got {error:#}"
        );
        Ok(())
    }

    #[test]
    fn world_core_v3_loss_reports_factual_and_health_populations() -> Result<()> {
        let cfg = TrainConfig {
            lessons: vec!["factual_branches".into()],
            physical_batch: crate::p2::data::FACTUAL_BRANCHES_PER_GROUP,
            grad_accum: 1,
            sigreg_weight: 0.0,
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            world_core_v2: true,
            world_core_v3: true,
            spatial_action_field: true,
            spatial_action_residual: true,
            branch_learning: BranchLearningConfig {
                enabled: true,
                spatial_health: Some(crate::p2::representation::VicRegConfig {
                    variance_weight: 0.05,
                    covariance_weight: 0.005,
                    minimum_std: 1.0,
                    epsilon: 1e-4,
                    maximum_rows: 128,
                }),
                pooled_health: Some(crate::p2::representation::VicRegConfig {
                    variance_weight: 0.05,
                    covariance_weight: 0.005,
                    minimum_std: 1.0,
                    epsilon: 1e-4,
                    maximum_rows: 12,
                }),
                displacement_health: Some(crate::p2::representation::VicRegConfig {
                    variance_weight: 0.02,
                    covariance_weight: 0.01,
                    minimum_std: 0.1,
                    epsilon: 1e-4,
                    maximum_rows: 4,
                }),
                ..BranchLearningConfig::default()
            },
            ..TrainConfig::default()
        };
        cfg.validate()?;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let samples = collect_batch(
            "factual_branches",
            cfg.seed,
            0,
            crate::p2::data::FACTUAL_BRANCHES_PER_GROUP,
            Split::Train,
        )?;
        let batch = batch_from_samples(&samples, &device)?;
        let losses = leworld_loss_with_sigreg_windows(
            &model,
            &batch,
            None,
            Some(&samples),
            &cfg,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            7,
            lesson_loss_weights("factual_branches", &cfg, 0, 0),
            None,
        )?;
        assert_eq!(losses.branch_audit.groups, 1);
        assert_eq!(
            losses.branch_audit.branches,
            crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
        );
        assert_eq!(losses.branch_audit.spatial_population_rows, 128);
        assert_eq!(losses.branch_audit.pooled_population_rows, 12);
        assert!(losses.branch_audit.unique_changed_outcomes >= 2);
        assert!(losses.branch_audit.displacement_population_rows >= 2);
        assert!(losses.branch_total.to_scalar::<f32>()?.is_finite());
        let gradients = losses.total.backward()?;
        let data = varmap.data().lock().unwrap();
        let decoder = data
            .get("action_decoder.weight")
            .expect("world-core-v2 action decoder");
        assert!(gradients.get(decoder.as_tensor()).is_some());
        Ok(())
    }

    #[test]
    fn rollout_horizon_caps_retarget() {
        assert_eq!(rollout_horizon_for_lesson("retarget", 80, 100), 4);
        assert_eq!(rollout_horizon_for_lesson("sequential", 80, 100), 8);
    }

    #[test]
    fn rollout_teacher_mix_is_higher_on_retarget() {
        assert!(
            rollout_teacher_mix("retarget", 0, 100) > rollout_teacher_mix("sequential", 0, 100)
        );
        assert_eq!(rollout_teacher_mix("sequential", 100, 100), 0.0);
    }

    #[test]
    fn lesson_to_curriculum_maps_auxiliary_warmup_lessons() {
        assert_eq!(lesson_to_curriculum("q_calibration").unwrap(), "sequential");
        assert_eq!(lesson_to_curriculum("events").unwrap(), "sequential");
        assert_eq!(lesson_to_curriculum("exploration").unwrap(), "exploration");
        assert_eq!(
            lesson_to_curriculum("falsification").unwrap(),
            "p1c_falsification"
        );
    }

    #[test]
    fn rollout_horizon_ramps_within_lesson() {
        assert_eq!(rollout_horizon(0, 100, 8), 2);
        assert_eq!(rollout_horizon(30, 100, 8), 4);
        assert_eq!(rollout_horizon(60, 100, 8), 8);
        assert_eq!(rollout_horizon(80, 100, 8), 8);
    }

    #[test]
    fn sample_recursion_depth_respects_bounds() {
        let cfg = TrainConfig {
            inner_steps: 4,
            outer_steps: 3,
            randomize_depth: true,
            ..TrainConfig::default()
        };
        for step in 0..32 {
            let depth = sample_recursion_depth(&cfg, step);
            assert!((1..=4).contains(&depth.inner_steps));
            assert!((1..=3).contains(&depth.outer_steps));
        }
        let fixed = TrainConfig {
            inner_steps: 2,
            outer_steps: 2,
            randomize_depth: false,
            ..TrainConfig::default()
        };
        let depth = sample_recursion_depth(&fixed, 0);
        assert_eq!(depth.inner_steps, 2);
        assert_eq!(depth.outer_steps, 2);
        let steady = TrainConfig {
            inner_steps: 2,
            outer_steps: 8,
            randomize_depth: true,
            steady_gpu: true,
            ..TrainConfig::default()
        };
        let depth = sample_recursion_depth(&steady, 99);
        assert_eq!(depth.inner_steps, 2);
        assert_eq!(depth.outer_steps, 8);
    }

    #[test]
    fn ptrm_rank_cadence_gates_sequential_and_calibration() {
        assert!(ptrm_rank_this_step("sequential", 0, 4, false));
        assert!(!ptrm_rank_this_step("sequential", 1, 4, false));
        assert!(ptrm_rank_this_step("sequential", 4, 4, false));
        assert!(ptrm_rank_this_step("q_calibration", 4, 4, false));
        assert!(ptrm_rank_this_step("falsification", 4, 4, false));
        assert!(!ptrm_rank_this_step("dynamics", 4, 4, false));
        assert!(!ptrm_rank_this_step("exploration", 4, 4, false));
        assert!(ptrm_rank_this_step("retarget", 8, 4, false));
        assert!(ptrm_rank_this_step("sequential", 3, 1, false));
        assert!(!ptrm_rank_this_step("sequential", 4, 4, true));
    }

    #[test]
    fn effective_batch_multiplies_accum() {
        let cfg = TrainConfig {
            physical_batch: 256,
            grad_accum: 2,
            ..Default::default()
        };
        assert_eq!(effective_batch(&cfg), 512);
    }

    #[test]
    fn legacy_sigreg_projection_seeds_stride_by_accumulation() {
        let seed = 41;
        let accum = 4;
        let first = (0..accum)
            .map(|micro| legacy_sigreg_projection_seed(seed, 7, accum, micro))
            .collect::<Vec<_>>();
        let second = (0..accum)
            .map(|micro| legacy_sigreg_projection_seed(seed, 8, accum, micro))
            .collect::<Vec<_>>();
        assert_eq!(first, vec![69, 70, 71, 72]);
        assert_eq!(second, vec![73, 74, 75, 76]);
        assert!(first.iter().all(|value| !second.contains(value)));
        assert_eq!(
            legacy_sigreg_projection_seed(seed, 7, 1, 0),
            seed.wrapping_add(7)
        );
    }

    #[test]
    fn train_config_rejects_invalid_muon_hyperparameters() {
        for (momentum, rms_scale, expected) in [
            (f64::NAN, MUON_RMS_SCALE, "muon_momentum"),
            (-0.1, MUON_RMS_SCALE, "muon_momentum"),
            (1.0, MUON_RMS_SCALE, "muon_momentum"),
            (0.95, f64::NAN, "muon_rms_scale"),
            (0.95, 0.0, "muon_rms_scale"),
            (0.95, -1.0, "muon_rms_scale"),
        ] {
            let cfg = TrainConfig {
                muon_momentum: momentum,
                muon_rms_scale: rms_scale,
                ..TrainConfig::default()
            };
            let error = cfg.validate().expect_err("invalid Muon config must reject");
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[test]
    fn scheduled_episode_ids_are_disjoint_per_microbatch() {
        let a = scheduled_episode_start(1, 5, 0, 2, false);
        let b = scheduled_episode_start(1, 5, 1, 2, false);
        assert_ne!(a, b);
        assert_eq!(a, 10);
        assert_eq!(b, 11);
    }

    #[test]
    fn open_loop_loss_is_finite_and_backpropagates() -> Result<()> {
        let cfg = TrainConfig {
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..TrainConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let trace = collect_rollout_trace("sequential", cfg.seed, 0, Split::Train)?;
        let depth = RecursionDepth {
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
        };
        let trace = ordered_trace_from_samples(&trace, &device)?;
        let loss = open_loop_latent_loss(&model, &trace, 4, depth, 0.25, cfg.seed)?;
        assert!(ensure_finite("open_loop", &loss)?.is_finite());
        let grads = loss.backward()?;
        assert!(varmap
            .all_vars()
            .iter()
            .any(|var| grads.get(var.as_tensor()).is_some()));
        Ok(())
    }

    #[test]
    fn shared_prefix_rollout_matches_recomputed_reference() -> Result<()> {
        let cfg = TrainConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..TrainConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let samples = collect_rollout_trace("sequential", cfg.seed, 0, Split::Train)?;
        let trace = ordered_trace_from_samples(&samples, &device)?;

        let actual = prefix_multi_horizon_loss(&model, &trace)?;
        let expected = prefix_multi_horizon_loss_reference(&model, &trace)?;
        let actual_value = actual.to_scalar::<f32>()?;
        let expected_value = expected.to_scalar::<f32>()?;
        assert!((actual_value - expected_value).abs() <= 1e-6);

        let actual_grads = actual.backward()?;
        let expected_grads = expected.backward()?;
        for (name, var) in varmap.data().lock().unwrap().iter() {
            match (
                actual_grads.get(var.as_tensor()),
                expected_grads.get(var.as_tensor()),
            ) {
                (Some(actual), Some(expected)) => {
                    let diff = actual.sub(expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(diff <= 2e-5, "prefix gradient mismatch for {name}: {diff}");
                }
                (None, None) => {}
                _ => panic!("prefix gradient presence mismatch for {name}"),
            }
        }
        Ok(())
    }

    #[test]
    fn report_serialization_roundtrip() -> Result<()> {
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
            world_core_schema: "legacy_p2_eval_compatible".into(),
            experiment: ResolvedExperiment::default(),
            seed: 1,
            physical_batch: 2,
            grad_accum: 1,
            lr: 1e-3,
            weight_decay: 0.01,
            parameter_count: 10,
            training_population_fingerprint: "fnv1a64:0000000000000000".into(),
            training_content_fingerprint: "sha256:00".into(),
            training_population_rows: 0,
            device: "cpu".into(),
            lessons: vec![],
            status: TrainStatus::Completed,
            global_step: 0,
            latest_checkpoint: PathBuf::from("checkpoint"),
            resumed_from: None,
            batch_schedule_migrations: vec![],
            checkpoint: PathBuf::from("m.safetensors"),
            export_checkpoint: None,
            config_path: PathBuf::from("c.json"),
            profile: ProfileState::Pending,
            gradient_pressure: None,
            gradient_pressure_samples: vec![],
            foundation_v2: None,
            research_claim: false,
            resume_count: 0,
            run_attempts: vec![],
        };
        let s = serde_json::to_string(&report)?;
        let back: TrainReport = serde_json::from_str(&s)?;
        assert_eq!(back.schema, TRAIN_REPORT_SCHEMA);
        assert_eq!(back.grad_accum, 1);
        assert!(!back.research_claim);
        Ok(())
    }

    #[test]
    fn auto_resumes_from_output_checkpoints_without_explicit_resume() -> Result<()> {
        let root = std::env::temp_dir().join(format!("tofy-p2-auto-resume-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.join("run"));
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        assert_eq!(paused.status, TrainStatus::Paused);
        assert_eq!(paused.global_step, 1);
        let ProfileState::Published(artifacts) = &paused.profile else {
            panic!("first optimizer update must publish profile evidence");
        };
        assert!(artifacts.trace.is_file());
        assert!(artifacts.evidence_json.is_file());
        assert!(artifacts.evidence_markdown.is_file());
        assert!(artifacts.viewer_html.is_file());
        let evidence: candle_graph::EvidencePacket =
            serde_json::from_slice(&fs::read(&artifacts.evidence_json)?)?;
        assert!(evidence.health.structurally_valid);
        assert!(evidence.health.capture_complete);
        assert!(evidence.health.coverage.forward_spans > 0);
        assert!(evidence.health.coverage.backward_spans > 0);
        assert!(evidence.health.coverage.optimizer_spans > 0);
        let trace_events = fs::read_to_string(&artifacts.trace)?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert!(trace_events.iter().any(|event| {
            event.get("kind").and_then(serde_json::Value::as_str) == Some("gradient")
                && event.get("root").and_then(serde_json::Value::as_str) == Some("vb/post_clip")
        }));
        assert!(trace_events.iter().any(|event| {
            event.get("kind").and_then(serde_json::Value::as_str) == Some("tensor")
                && event.get("label").and_then(serde_json::Value::as_str) == Some("loss.total")
                && event.get("tensor_id").and_then(serde_json::Value::as_str) != Some("loss.total")
        }));

        cfg.max_steps_this_run = None;
        cfg.resume = None;
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        assert_eq!(resumed.global_step, paused.global_step + 1);
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn pause_resume_matches_uninterrupted_training_within_cpu_reduction_tolerance() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("tofy-p2-exact-resume-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let full_cfg = resume_test_config(root.join("full"));
        let full = train(&full_cfg)?;
        assert_eq!(full.status, TrainStatus::Completed);

        let mut split_cfg = resume_test_config(root.join("split"));
        split_cfg.max_steps_this_run = Some(1);
        let paused = train(&split_cfg)?;
        assert_eq!(paused.status, TrainStatus::Paused);
        assert_eq!(paused.global_step, 1);
        assert!(paused
            .latest_checkpoint
            .join("optimizer.safetensors")
            .is_file());

        split_cfg.max_steps_this_run = None;
        split_cfg.resume = Some(paused.latest_checkpoint.clone());
        let resumed = train(&split_cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        assert_eq!(resumed.global_step, full.global_step);
        lessons_match_within_eps(&resumed.lessons, &full.lessons, 1e-5);

        let full_values = loaded_model_values(&full_cfg, &full.checkpoint)?;
        let resumed_values = loaded_model_values(&split_cfg, &resumed.checkpoint)?;
        for ((name_a, a), (name_b, b)) in full_values.iter().zip(&resumed_values) {
            assert_eq!(name_a, name_b);
            assert_eq!(a.len(), b.len(), "length mismatch at {name_a}");
            for (va, vb) in a.iter().zip(b) {
                assert!(
                    (va - vb).abs() < 1e-5,
                    "weight mismatch at {name_a}: {va} vs {vb}"
                );
            }
        }

        let full_state: TrainerState =
            read_json(&full.latest_checkpoint.join("trainer_state.json"))?;
        let resumed_state: TrainerState =
            read_json(&resumed.latest_checkpoint.join("trainer_state.json"))?;
        assert_eq!(resumed_state.global_step, full_state.global_step);
        assert_eq!(resumed_state.optimizer_step, full_state.optimizer_step);
        assert_eq!(
            resumed_state.training_content_hash,
            full_state.training_content_hash
        );
        lessons_match_within_eps(
            &resumed_state.completed_lessons,
            &full_state.completed_lessons,
            1e-5,
        );
        assert_loss_means_close(&resumed_state.active_sums, &full_state.active_sums, 1e-5);

        let full_moments = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(
                full.latest_checkpoint.join("optimizer.safetensors"),
            )?
        };
        let resumed_moments = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(
                resumed.latest_checkpoint.join("optimizer.safetensors"),
            )?
        };
        for name in full_state.parameter_names {
            let muon_key = format!("muon.momentum.{name}");
            if full_moments.load(&muon_key, &Device::Cpu).is_ok() {
                let a = full_moments
                    .load(&muon_key, &Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let b = resumed_moments
                    .load(&muon_key, &Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_close_f32(&a, &b, &muon_key);
                continue;
            }
            for prefix in ["first_moment", "second_moment"] {
                let key = format!("{prefix}.{name}");
                let a = full_moments
                    .load(&key, &Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let b = resumed_moments
                    .load(&key, &Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_close_f32(&a, &b, &key);
            }
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn curriculum_transition_matches_with_and_without_prefetch() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-prefetch-curriculum-scope-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let base = TrainConfig {
            lessons: vec!["dynamics".into(), "exploration".into()],
            steps_per_lesson: 1,
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projections: 2,
            sigreg_knots: 3,
            profile_update: 99,
            checkpoint_every_steps: 0,
            ..TrainConfig::default()
        };
        let without_cfg = TrainConfig {
            prefetch_batches: false,
            output_dir: root.join("without"),
            ..base.clone()
        };
        let with_cfg = TrainConfig {
            prefetch_batches: true,
            output_dir: root.join("with"),
            ..base
        };
        let without = train(&without_cfg)?;
        let with = train(&with_cfg)?;
        lessons_match_within_eps(&without.lessons, &with.lessons, 1e-5);
        let without_values = loaded_model_values(&without_cfg, &without.checkpoint)?;
        let with_values = loaded_model_values(&with_cfg, &with.checkpoint)?;
        for ((without_name, without), (with_name, with)) in without_values.iter().zip(&with_values)
        {
            assert_eq!(without_name, with_name);
            assert_close_f32(without, with, without_name);
        }
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn exact_resume_rejects_requested_trajectory_change() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-resume-merge-contract-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.lr *= 2.0;
        cfg.sigreg_weight = 999.0;
        cfg.inner_steps = 99;
        cfg.outer_steps = 99;
        cfg.sigreg_target = SigregTarget::TemporalResidual;
        cfg.sigreg_temporal_window = 2;
        cfg.sigreg_spatial = true;
        cfg.sigreg_spatial_pool = true;
        let err = train(&cfg).expect_err("changed trajectory config must reject exact resume");
        assert!(
            err.to_string().contains("training contract mismatch"),
            "{err:#}"
        );
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_contract_covers_all_trajectory_config_fields() {
        let base = resume_test_config(PathBuf::from("unused"));
        let base_contract = TrainingContract::from(&base);
        let mut changed = Vec::new();
        let make_v2 = |cfg: &mut TrainConfig| {
            cfg.world_core_v2 = true;
            cfg.branch_learning.enabled = true;
            cfg.sigreg_weight = 0.0;
            cfg.lessons.push("factual_branches".into());
        };

        let mut cfg = base.clone();
        cfg.ptrm_rank_every += 1;
        changed.push(("ptrm_rank_every", cfg));
        let mut cfg = base.clone();
        cfg.baseline_d1 = !cfg.baseline_d1;
        changed.push(("baseline_d1", cfg));
        let mut cfg = base.clone();
        cfg.prefix_weight += 0.25;
        changed.push(("prefix_weight", cfg));
        let mut cfg = base.clone();
        cfg.reliability_weight += 0.25;
        changed.push(("reliability_weight", cfg));
        let mut cfg = base.clone();
        cfg.bf16_conv = !cfg.bf16_conv;
        changed.push(("bf16_conv", cfg));
        let mut cfg = base.clone();
        cfg.bf16_recurrent_core = !cfg.bf16_recurrent_core;
        changed.push(("bf16_recurrent_core", cfg));
        let mut cfg = base.clone();
        cfg.sigreg_max_rows += 1;
        changed.push(("sigreg_max_rows", cfg));
        let mut cfg = base.clone();
        cfg.sigreg_spatial = true;
        cfg.sigreg_spatial_pool = false;
        cfg.sigreg_pre_rms_spatial = true;
        changed.push(("sigreg_pre_rms_spatial", cfg));
        let mut cfg = base.clone();
        cfg.sigreg_target = SigregTarget::TemporalResidual;
        cfg.sigreg_spatial = true;
        changed.push(("sigreg_target", cfg));
        let mut cfg = base.clone();
        cfg.sigreg_temporal_window += 1;
        changed.push(("sigreg_temporal_window", cfg));
        let mut cfg = base.clone();
        cfg.sigreg_target = SigregTarget::TemporalResidual;
        cfg.sigreg_spatial = true;
        cfg.sigreg_global_mix = 0.5;
        changed.push(("sigreg_global_mix", cfg));
        let mut cfg = base.clone();
        make_v2(&mut cfg);
        changed.push(("world_core_v2", cfg));
        let mut cfg = base.clone();
        make_v2(&mut cfg);
        cfg.world_core_v3 = true;
        changed.push(("world_core_v3", cfg));
        let mut cfg = base.clone();
        make_v2(&mut cfg);
        cfg.spatial_action_field = true;
        changed.push(("spatial_action_field", cfg));
        let mut cfg = base.clone();
        make_v2(&mut cfg);
        cfg.world_core_v3 = true;
        cfg.spatial_action_field = true;
        cfg.spatial_action_residual = true;
        changed.push(("spatial_action_residual", cfg));
        let mut cfg = base.clone();
        cfg.spatial_action_residual_scale += 0.1;
        changed.push(("spatial_action_residual_scale", cfg));
        let mut cfg = base.clone();
        cfg.split_ce_weighting = SplitCeWeighting::PooledPerPixel;
        changed.push(("split_ce_weighting", cfg));
        let mut cfg = base.clone();
        cfg.split_ce_changed_budget = Some(0.5);
        changed.push(("split_ce_changed_budget", cfg));
        let mut cfg = base.clone();
        cfg.promotion_metric = PromotionMetric::FullExact;
        changed.push(("promotion_metric", cfg));
        let mut cfg = base.clone();
        cfg.branch_learning.outcome_pull_weight += 0.01;
        changed.push(("branch_learning", cfg));
        let mut cfg = base.clone();
        cfg.data_contract_v6 = true;
        changed.push(("data_contract_v6", cfg));

        for (name, cfg) in changed {
            assert_ne!(
                base_contract,
                TrainingContract::from(&cfg),
                "trajectory field {name} is absent from the resume contract"
            );
        }
    }

    #[test]
    fn exact_resume_rejects_equal_effective_batch_schedule_change() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-resume-batch-schedule-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.physical_batch = 2;
        cfg.grad_accum = 2;
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.physical_batch = 4;
        cfg.grad_accum = 1;
        let err = train(&cfg).expect_err("exact resume must reject a changed batch schedule");
        assert!(
            err.to_string().contains("training contract mismatch"),
            "{err:#}"
        );
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn explicit_batch_schedule_migration_is_labeled_durably() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-explicit-batch-migration-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.physical_batch = 2;
        cfg.grad_accum = 2;
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.physical_batch = 4;
        cfg.grad_accum = 1;
        cfg.allow_batch_schedule_migration = true;
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        let state: TrainerState = read_json(&resumed.latest_checkpoint.join("trainer_state.json"))?;
        assert_eq!(state.contract.physical_batch, 4);
        assert_eq!(state.contract.grad_accum, 1);
        assert_eq!(state.batch_schedule_migrations.len(), 1);
        let migration = &state.batch_schedule_migrations[0];
        assert_eq!(
            (migration.from_physical_batch, migration.from_grad_accum),
            (2, 2)
        );
        assert_eq!(
            (migration.to_physical_batch, migration.to_grad_accum),
            (4, 1)
        );
        assert_eq!(migration.effective_batch, 4);
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_rejects_effective_batch_change() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-resume-effective-batch-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.physical_batch = 2;
        cfg.grad_accum = 2;
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.physical_batch = 8;
        cfg.grad_accum = 1;
        let err = train(&cfg).expect_err("effective batch change must reject resume");
        assert!(
            err.to_string().contains("training contract mismatch"),
            "{err:#}"
        );
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn exact_resume_rejects_steady_gpu_toggle() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("tofy-p2-resume-steady-gpu-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.steady_gpu = true;
        let err = train(&cfg).expect_err("steady_gpu changes the sampled depth trajectory");
        assert!(
            err.to_string().contains("training contract mismatch"),
            "{err:#}"
        );
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_rejects_training_contract_changes() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("tofy-p2-resume-contract-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint.clone());
        let saved_lr = cfg.lr;
        cfg.lr *= 2.0;
        let err = train(&cfg).expect_err("changed requested contract must reject resume");
        assert!(err.to_string().contains("training contract mismatch"));

        cfg.lr = saved_lr;
        fs::remove_file(paused.latest_checkpoint.join("optimizer.safetensors"))?;
        let err = train(&cfg).expect_err("missing optimizer state must reject resume");
        assert!(err.to_string().contains("checkpoint bundle is incomplete"));
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn foundation_v2_rollout_floor_and_activation_premise() -> Result<()> {
        // Papers-audit mandatory control: the imported run trained with
        // rollout "enabled" but a mean rollout loss of exactly zero, so no
        // architecture arm can be interpreted until this path demonstrably
        // fires. Below the 16-fragment floor the loss must be exactly zero;
        // at a full batch it must be finite, nonzero, and reach the
        // recurrent core's parameters.
        let device = Device::Cpu;
        let mut cfg = TrainConfig::default();
        cfg.apply_foundation_v2_recipe();
        let varmap = VarMap::new();
        let model = WorldModel::new(
            cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        // Launch-faithful initialization: production restores these
        // immediately after the generic reinit, so the premise must hold for
        // the exact weights a run starts from.
        reinit_varmap_deterministic(&varmap, 23)?;
        zero_action_film_projections(&varmap)?;
        zero_operator_conditioning_projection(&varmap)?;
        init_copy_bypass_gate(&varmap)?;
        restore_copy_gate_bias_prior(&varmap, cfg.copy_gate_bias_prior)?;

        let small = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 128,
                seed: 51,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            0.5,
            0,
            V5DataSplit::Train,
        )?;
        let small_samples = small.transitions().cloned().collect::<Vec<_>>();
        let small_batch = batch_from_samples(&small_samples, &device)?;
        let small_encoded =
            model.encode_state_pair_for_training(&small_batch.frames, &small_batch.next_frames)?;
        let (small_loss, small_fragments) =
            foundation_v2_rollout_loss(&model, &small, &small_batch, &small_encoded, &device)?;
        assert_eq!(small_fragments, 0, "inert rollout must report inactive");
        assert_eq!(small_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?, 0.0);

        let full = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 512,
                seed: 52,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            0.5,
            0,
            V5DataSplit::Train,
        )?;
        let full_samples = full.transitions().cloned().collect::<Vec<_>>();
        let full_batch = batch_from_samples(&full_samples, &device)?;
        let full_encoded =
            model.encode_state_pair_for_training(&full_batch.frames, &full_batch.next_frames)?;
        let (loss, fragments) =
            foundation_v2_rollout_loss(&model, &full, &full_batch, &full_encoded, &device)?;
        assert!(fragments >= FOUNDATION_V2_MIN_ROLLOUT_FRAGMENTS);
        let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        assert!(value.is_finite() && value > 0.0, "rollout loss {value}");
        let core_grad_vec = |loss: &Tensor| -> Result<Vec<f32>> {
            let grads = loss.backward()?;
            let data = varmap.data().lock().unwrap();
            let weight = data.get("block.c1.weight").expect("core weight exists");
            Ok(grads
                .get(weight.as_tensor())
                .expect("rollout loss must reach the recurrent core")
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        let attached = core_grad_vec(&loss)?;
        assert!(attached
            .iter()
            .all(|g| g.is_finite() && attached.iter().any(|g| *g != 0.0)));
        // Attribution control: detaching the open-loop input isolates the
        // first transition's contribution. A shared core means a nonzero
        // final gradient alone cannot prove the graph traverses transition
        // one; the attached/detached difference can.
        let (detached_loss, _) = foundation_v2_rollout_loss_inner(
            &model,
            &full,
            &full_batch,
            &full_encoded,
            &device,
            true,
        )?;
        let detached = core_grad_vec(&detached_loss)?;
        assert!(
            attached
                .iter()
                .zip(&detached)
                .any(|(a, d)| (a - d).abs() > 0.0),
            "first open-loop transition contributes no core gradient"
        );
        Ok(())
    }

    #[test]
    fn foundation_v2_mechanism_sample_reads_probes_and_copy_gate() -> Result<()> {
        let device = Device::Cpu;
        let model_cfg = ModelConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 2,
            world_core_v4: true,
            spatial_action_field: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            residual_y_update: true,
            warm_start_y: true,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(
            model_cfg,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, 71)?;
        zero_action_film_projections(&varmap)?;
        zero_operator_conditioning_projection(&varmap)?;
        restore_copy_gate_bias_prior(&varmap, None)?;
        let mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 16,
                seed: 72,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::UnseenSeed7x7,
        )?;
        let samples = mixed.transitions().cloned().collect::<Vec<_>>();
        let sample = foundation_v2_mechanism_sample(&model, &samples, &device, 1_024)?;
        assert_eq!(sample.step, 1_024);
        assert_eq!(sample.copy_bypass_alpha, None);
        assert_eq!(sample.outer_step_cosines.len(), 2);
        assert!(sample
            .outer_step_cosines
            .iter()
            .all(|value| value.is_finite()));
        assert!((0.0..=1.0).contains(&sample.gate_open_rate));
        assert!((0.0..=1.0).contains(&sample.gate_mean_probability));
        Ok(())
    }

    #[test]
    fn foundation_v2_profile_resume_rejects_passed_unpublished_target() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-foundation-v2-missing-profile-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut published = vec![4];
        let error = validate_foundation_v2_profile_resume(&[2, 4], &mut published, 4, &root, "cpu")
            .expect_err("checkpoint bookkeeping without bundles cannot satisfy the guard");
        assert!(
            error.to_string().contains("completed updates [2, 4]"),
            "{error:#}"
        );
        for update in [2, 4] {
            let bundle = root.join("profile").join(format!("update-{update:012}"));
            let run = candle_graph::ProfileRun::training(PROFILE_ENTRYPOINT, update, "cpu")
                .correlation_id(format!("tofy.p2/update-{update:012}"));
            let capture = match candle_graph::CaptureRun::begin(&bundle, run)? {
                candle_graph::CaptureBegin::Active(capture) => capture,
                candle_graph::CaptureBegin::AlreadyPublished(_) => unreachable!(),
            };
            let measured = capture.session().begin_measurement("resume/guard");
            drop(measured);
            capture.publish()?;
        }
        let mut repaired = vec![2, 4];
        validate_foundation_v2_profile_resume(&[2, 4], &mut repaired, 4, &root, "cpu")?;
        assert_eq!(repaired, [2, 4]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn foundation_v2_profile_resume_reconciles_complete_bundle() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-foundation-v2-reconcile-profile-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let bundle = root.join("profile/update-000000000002");
        let run = candle_graph::ProfileRun::training(PROFILE_ENTRYPOINT, 2, "cpu")
            .correlation_id("tofy.p2/update-000000000002");
        let capture = match candle_graph::CaptureRun::begin(&bundle, run)? {
            candle_graph::CaptureBegin::Active(capture) => capture,
            candle_graph::CaptureBegin::AlreadyPublished(_) => {
                unreachable!("test destination starts absent")
            }
        };
        let measured = capture.session().begin_measurement("resume/reconcile");
        drop(measured);
        capture.publish()?;
        let mut published = Vec::new();
        validate_foundation_v2_profile_resume(&[2], &mut published, 1, &root, "cpu")?;
        assert_eq!(published, [2]);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// Sink that fails a configured number of write calls before succeeding,
    /// recording all bytes that were accepted. Models the transient I/O blips
    /// of a network-mounted /workspace.
    struct FlakyLossLogSink {
        data: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        remaining_failures: usize,
        fail_forever: bool,
    }

    impl Write for FlakyLossLogSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.fail_forever {
                return Err(std::io::Error::other("persistent write failure"));
            }
            if self.remaining_failures > 0 {
                self.remaining_failures -= 1;
                return Err(std::io::Error::other("transient write failure"));
            }
            // Partial writes are legal for Write; exercise offset tracking.
            let take = buf.len().min(3);
            self.data.lock().unwrap().extend_from_slice(&buf[..take]);
            Ok(take)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl FoundationV2LossLogSink for FlakyLossLogSink {
        fn sync(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Regression: a transient write failure must not silently kill the
    /// writer thread — the s8 trainer died one step after exactly that, with
    /// "sending on a closed channel". Every appended row must come through
    /// exactly once, uncorrupted, despite failures and partial writes.
    #[test]
    fn foundation_v2_loss_log_survives_transient_write_failures() -> Result<()> {
        let data = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = FlakyLossLogSink {
            data: data.clone(),
            remaining_failures: 3,
            fail_forever: false,
        };
        let values = FoundationV2LossMeans::default();
        let mut log =
            FoundationV2LossLog::with_sink(sink, PathBuf::from("test-loss-log.jsonl"), false)?;
        for step in 0..4u64 {
            log.append(step, &values, 1e-3)?;
        }
        log.flush()?;
        drop(log);
        let contents = String::from_utf8(data.lock().unwrap().clone())?;
        let steps: Vec<u64> = contents
            .lines()
            .map(|line| {
                let row: serde_json::Value = serde_json::from_str(line).expect("valid JSONL row");
                row["global_step"].as_u64().expect("global_step")
            })
            .collect();
        assert_eq!(steps, vec![0, 1, 2, 3]);
        Ok(())
    }

    /// Regression: a persistent write failure must surface as a training-side
    /// error (fail loudly) instead of hanging or panicking.
    #[test]
    fn foundation_v2_loss_log_persistent_failure_fails_loudly() -> Result<()> {
        let data = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = FlakyLossLogSink {
            data,
            remaining_failures: 0,
            fail_forever: true,
        };
        let values = FoundationV2LossMeans::default();
        let mut log =
            FoundationV2LossLog::with_sink(sink, PathBuf::from("test-loss-log.jsonl"), false)?;
        // The worker gives up while draining the bounded queue; subsequent
        // appends or the flush must report the failure to the caller.
        let mut failed = false;
        for step in 0..8u64 {
            if log.append(step, &values, 1e-3).is_err() {
                failed = true;
                break;
            }
        }
        failed = failed || log.flush().is_err();
        assert!(
            failed,
            "persistent sink failure never surfaced to the caller"
        );
        Ok(())
    }

    #[test]
    fn foundation_v2_loss_log_flushes_complete_jsonl_rows() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-foundation-v2-loss-log-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        let values = FoundationV2LossMeans {
            total: 1.0,
            rollout: 0.25,
            pre_clip_gradient_norm: 2.0,
            gradient_clip_scale: 0.5,
            ..FoundationV2LossMeans::default()
        };
        {
            let mut log = FoundationV2LossLog::open(&root)?;
            log.append(7, &values, 1e-3)?;
        }
        let contents = fs::read_to_string(root.join("loss_log.jsonl"))?;
        let rows = contents.lines().collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        let row: serde_json::Value = serde_json::from_str(rows[0])?;
        assert_eq!(row["global_step"], 7);
        assert_eq!(row["rollout"], 0.25);
        assert_eq!(row["learning_rate"], 1e-3);
        let active = foundation_v2_active_loss_means(&values, 1);
        assert_eq!(active.total, 1.0);
        assert_eq!(active.rollout, 0.25);
        assert_eq!(active.pre_clip_gradient_norm, 2.0);
        assert_eq!(active.gradient_clip_scale, 0.5);
        assert_eq!(active.next_latent, 0.0);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn loss_log_repair_moves_only_a_strict_suffix() -> Result<()> {
        let root = checkpoint_test_root("loss-log-repair");
        fs::create_dir_all(&root)?;
        assert!(repair_loss_log_for_resume(&root, 7, 1)?.is_none());
        let row = |attempt: &str, step: u64| {
            format!("{{\"global_step\":{step},\"total\":1.0,\"attempt\":\"{attempt}\"}}\n")
        };
        let mut log = String::new();
        for step in 1..=9 {
            log.push_str(&row("a", step));
        }
        let path = root.join("loss_log.jsonl");
        fs::write(&path, &log)?;

        let repair = repair_loss_log_for_resume(&root, 7, 2)?.expect("log exists");
        assert_eq!(repair.resumed_step, 7);
        assert_eq!(repair.rows_before, 9);
        assert_eq!(repair.rows_kept, 7);
        assert_eq!(repair.rows_removed, 2);
        let sidecar = root.join("loss_log.jsonl.attempt-2");
        assert_eq!(repair.removed_rows_path.as_deref(), Some(sidecar.as_path()));
        let kept_steps = fs::read_to_string(&path)?
            .lines()
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).expect("kept row");
                value["global_step"].as_u64().unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(kept_steps, (1..=7).collect::<Vec<_>>());
        let removed = fs::read_to_string(&sidecar)?;
        let removed_lines = removed.lines().collect::<Vec<_>>();
        assert_eq!(removed_lines.len(), 2);
        assert!(removed_lines[0].contains("\"global_step\":8"));
        assert!(removed_lines[1].contains("\"global_step\":9"));

        // A consistent log is left byte-identical and gets no sidecar.
        let before = fs::read(&path)?;
        let again = repair_loss_log_for_resume(&root, 7, 3)?.expect("log exists");
        assert_eq!(again.rows_removed, 0);
        assert_eq!(again.removed_rows_path, None);
        assert_eq!(fs::read(&path)?, before);
        assert!(!root.join("loss_log.jsonl.attempt-3").exists());
        assert!(fs::read_dir(&root)?.all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("repair")
        }));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// Audit counterexample: checkpoint step 7 is the main lineage, then a
    /// failed explicit resume from step 3 appends 4..=6. Step ordering alone
    /// cannot say which duplicate branch owns the checkpoint, so no rewrite
    /// is allowed.
    #[test]
    fn loss_log_repair_rejects_ambiguous_mixed_lineage_without_mutation() -> Result<()> {
        let root = checkpoint_test_root("loss-log-ambiguous-lineage");
        fs::create_dir_all(&root)?;
        let row = |lineage: &str, step: u64| {
            format!("{{\"global_step\":{step},\"lineage\":\"{lineage}\"}}\n")
        };
        let mut log = String::new();
        for step in 1..=7 {
            log.push_str(&row("checkpoint-7", step));
        }
        for step in 4..=6 {
            log.push_str(&row("failed-resume-3", step));
        }
        let path = root.join("loss_log.jsonl");
        fs::write(&path, &log)?;
        let before = fs::read(&path)?;

        let error = repair_loss_log_for_resume(&root, 7, 4)
            .expect_err("duplicate/nonmonotonic log must fail closed");
        assert!(format!("{error:#}").contains("does not establish checkpoint lineage"));
        assert_eq!(fs::read(&path)?, before);
        assert!(!root.join("loss_log.jsonl.attempt-4").exists());
        assert!(fs::read_dir(&root)?.all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("repair")
        }));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn repair_journal_records_start_then_failure_and_retry_is_non_destructive() -> Result<()> {
        let root = checkpoint_test_root("loss-log-repair-journal");
        fs::create_dir_all(&root)?;
        let path = root.join("loss_log.jsonl");
        fs::write(
            &path,
            b"{\"global_step\":1}\n{\"global_step\":2}\n{\"global_step\":2}\n",
        )?;
        let before = fs::read(&path)?;

        for expected_attempt in 1..=2 {
            let attempt = begin_run_attempt(&root, Some(Path::new("checkpoint")), Some(2))?;
            assert_eq!(attempt, expected_attempt);
            let pending = read_run_attempts(&root)?;
            assert_eq!(
                pending.last().map(|record| record.repair_state),
                Some(RunAttemptRepairState::Pending)
            );
            let error = repair_loss_log_for_resume(&root, 2, attempt)
                .expect_err("ambiguous history must not be repaired");
            fail_run_attempt_repair(&root, attempt, &error)?;
            let recorded = read_run_attempts(&root)?;
            assert_eq!(
                recorded.last().map(|record| record.repair_state),
                Some(RunAttemptRepairState::Failed)
            );
            assert!(recorded
                .last()
                .and_then(|record| record.repair_failure.as_deref())
                .is_some_and(|failure| failure.contains("checkpoint lineage")));
            assert_eq!(fs::read(&path)?, before);
            assert!(!root
                .join(format!("loss_log.jsonl.attempt-{attempt}"))
                .exists());
        }
        assert_eq!(
            fs::read_to_string(root.join(RUN_ATTEMPTS_FILE))?
                .lines()
                .count(),
            4,
            "each attempt has one durable start and one failure event"
        );
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn legacy_flat_attempt_journal_defaults_to_completed_unknown_build_fields() -> Result<()> {
        let root = checkpoint_test_root("legacy-attempt-journal");
        fs::create_dir_all(&root)?;
        let legacy = serde_json::json!({
            "attempt": 1,
            "kind": "fresh",
            "started_unix_secs": 10,
            "pid": 1,
            "source_revision": "deadbeef",
            "source_revision_origin": "git:legacy-runtime-tree",
            "source_dirty": false,
            "binary_path": "tofy",
            "binary_sha256": "sha256:00",
            "candle_graph_revision": "cafef00d",
            "candle_graph_dirty": false
        });
        fs::write(
            root.join(RUN_ATTEMPTS_FILE),
            format!("{}\n", serde_json::to_string(&legacy)?),
        )?;
        let attempts = read_run_attempts(&root)?;
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].repair_state, RunAttemptRepairState::Completed);
        assert_eq!(
            attempts[0].provenance.build_command,
            crate::p2::evidence::UNKNOWN_PROVENANCE
        );
        assert_eq!(attempts[0].provenance.source_pushed, None);
        assert!(!attempts[0].provenance.source_revision_known());
        assert_eq!(
            attempts[0].provenance.runtime_checkout,
            crate::p2::evidence::RuntimeCheckoutProvenance::default()
        );
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// CPU end-to-end: a v6 foundation-v2 run persists as v6 in the report,
    /// the checkpoint contract, and the evidence manifest; the manifest carries
    /// explicit provenance and treatment flags; and a resume reconciles a
    /// duplicated loss log into one trajectory while recording each attempt.
    #[test]
    fn v6_smoke_persists_identity_provenance_and_repairs_loss_log_on_resume() -> Result<()> {
        use crate::p2::experiment::WorldCoreFamily;
        let root = checkpoint_test_root("v6-identity-smoke");
        let _ = fs::remove_dir_all(&root);
        let mut cfg = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            v6_recursion_steps: 2,
            ..TrainConfig::default()
        };
        cfg.apply_foundation_v2_recipe();
        cfg.seed = 73;
        cfg.steps_per_lesson = 2;
        // Smallest batch the mixed stream accepts: one factual group plus a row.
        cfg.physical_batch = crate::p2::data::FACTUAL_BRANCHES_PER_GROUP + 1;
        cfg.output_dir = root.join("run");
        cfg.checkpoint_every_steps = 0;
        cfg.prefetch_batches = false;
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        assert_eq!(paused.status, TrainStatus::Paused);
        assert_eq!(paused.global_step, 1);
        assert_eq!(paused.experiment.family, WorldCoreFamily::V6);
        assert_eq!(paused.world_core_schema, "world_core_v6");
        assert_eq!(paused.resume_count, 0);
        assert_eq!(paused.run_attempts.len(), 1);
        assert_eq!(paused.run_attempts[0].kind, RunAttemptKind::Fresh);
        assert_eq!(
            paused.run_attempts[0].repair_state,
            RunAttemptRepairState::Completed
        );
        let state: TrainerState = read_json(&paused.latest_checkpoint.join("trainer_state.json"))?;
        assert_eq!(
            state.contract.experiment.as_ref().map(|e| e.family),
            Some(WorldCoreFamily::V6)
        );
        let persisted: serde_json::Value = read_json(&cfg.output_dir.join("train_report.json"))?;
        assert_eq!(persisted["experiment"]["family"], "v6");
        assert_eq!(persisted["world_core_schema"], "world_core_v6");

        let manifest: serde_json::Value = read_json(
            &cfg.output_dir
                .join(crate::p2::evidence::EVIDENCE_MANIFEST_FILE),
        )?;
        let identity = &manifest["identity"];
        assert_eq!(identity["recipe"], "foundation_v2");
        assert_eq!(identity["family"], "v6");
        assert_eq!(identity["world_core_schema"], "world_core_v6");
        assert_eq!(identity["world_core_v6"], true);
        assert_eq!(identity["data_contract_v6"], true);
        assert_eq!(identity["v6_recursion_steps"], 2);
        assert_eq!(identity["physical_batch"], cfg.physical_batch as u64);
        assert_eq!(identity["grad_accum"], 1);
        assert_eq!(identity["effective_batch"], cfg.physical_batch as u64);
        assert_eq!(manifest["comparison"]["treatment"]["family"], "v6");
        let provenance = &manifest["provenance"];
        for key in [
            "source_revision",
            "source_revision_origin",
            "binary_sha256",
            "candle_graph_revision",
        ] {
            let value = provenance[key].as_str().unwrap_or_default();
            assert!(!value.is_empty(), "provenance.{key} must be explicit");
        }
        assert!(provenance["binary_sha256"]
            .as_str()
            .is_some_and(|sha| sha.starts_with("sha256:")));
        assert_eq!(provenance["resume_count"], 0);
        assert_eq!(provenance["attempts"].as_array().map(Vec::len), Some(1));
        assert_eq!(provenance["attempts"][0]["kind"], "fresh");
        assert!(provenance["attempts"][0]["binary_sha256"].is_string());

        // A later launch that resumed from step 1, logged steps 2..=3, and
        // died before checkpointing leaves stale rows past the checkpoint.
        let log = cfg.output_dir.join("loss_log.jsonl");
        let mut stale = fs::read_to_string(&log)?;
        assert_eq!(stale.lines().count(), 1);
        for step in 2..=3u64 {
            stale.push_str(&format!(
                "{{\"global_step\":{step},\"total\":0.0,\"stale\":true}}\n"
            ));
        }
        fs::write(&log, stale)?;

        cfg.max_steps_this_run = None;
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        assert_eq!(resumed.global_step, 2);
        assert_eq!(resumed.experiment.family, WorldCoreFamily::V6);
        assert_eq!(resumed.resume_count, 1);
        assert_eq!(resumed.run_attempts.len(), 2);
        let attempt = &resumed.run_attempts[1];
        assert_eq!(attempt.attempt, 2);
        assert_eq!(attempt.kind, RunAttemptKind::Resume);
        assert_eq!(attempt.resumed_step, Some(1));
        assert_eq!(attempt.repair_state, RunAttemptRepairState::Completed);
        let repair = attempt
            .loss_log_repair
            .as_ref()
            .expect("resume repairs the log");
        assert_eq!(
            (repair.rows_before, repair.rows_kept, repair.rows_removed),
            (3, 1, 2)
        );
        let sidecar = cfg.output_dir.join("loss_log.jsonl.attempt-2");
        assert_eq!(repair.removed_rows_path.as_deref(), Some(sidecar.as_path()));
        assert_eq!(fs::read_to_string(&sidecar)?.lines().count(), 2);
        let rows = fs::read_to_string(&log)?
            .lines()
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).expect("row");
                assert!(value.get("stale").is_none(), "stale row survived: {line}");
                value["global_step"].as_u64().unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![1, 2]);
        let manifest: serde_json::Value = read_json(
            &cfg.output_dir
                .join(crate::p2::evidence::EVIDENCE_MANIFEST_FILE),
        )?;
        assert_eq!(manifest["provenance"]["resume_count"], 1);
        assert_eq!(
            manifest["provenance"]["attempts"].as_array().map(Vec::len),
            Some(2)
        );
        assert_eq!(manifest["provenance"]["attempts"][1]["kind"], "resume");
        assert_eq!(
            manifest["provenance"]["attempts"][1]["repair_state"],
            "completed"
        );
        let artifact_roles = manifest["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|artifact| artifact["role"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(artifact_roles.contains(&"run_attempt_journal"));
        assert!(artifact_roles.contains(&"active_loss_log"));
        assert!(artifact_roles.contains(&"loss_log_repair_sidecar"));
        assert!(manifest["gaps"]
            .as_array()
            .unwrap()
            .iter()
            .any(|gap| gap.as_str().unwrap().contains("loss log was repaired")));
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn foundation_v2_rollout_zero_streak_tracks_enabled_realized_loss() {
        let mut streak = FOUNDATION_V2_GATE_EVERY - 1;
        update_foundation_v2_rollout_zero_streak(&mut streak, true, 0.0, 1_024);
        assert_eq!(streak, FOUNDATION_V2_GATE_EVERY);
        update_foundation_v2_rollout_zero_streak(&mut streak, true, 0.1, 1_025);
        assert_eq!(streak, 0);
        update_foundation_v2_rollout_zero_streak(&mut streak, false, 0.0, 1_026);
        assert_eq!(streak, 0);
    }

    #[test]
    fn foundation_v2_profile_update_publishes_tensor_stats_bundle() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("tofy-foundation-v2-profile-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = TrainConfig::default();
        cfg.apply_foundation_v2_recipe();
        cfg.seed = 73;
        cfg.steps_per_lesson = 2;
        cfg.physical_batch = crate::p2::data::FACTUAL_BRANCHES_PER_GROUP + 1;
        cfg.output_dir = root.join("run");
        cfg.checkpoint_every_steps = 0;
        cfg.prefetch_batches = false;
        cfg.profile_updates = vec![2];
        let report = train(&cfg)?;
        let foundation = report
            .foundation_v2
            .as_ref()
            .expect("foundation-v2 report exists");
        let loss_rows = fs::read_to_string(cfg.output_dir.join("loss_log.jsonl"))?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(loss_rows.len(), 2);
        for (index, row) in loss_rows.iter().enumerate() {
            assert_eq!(row["global_step"], index as u64 + 1);
            for field in [
                "total",
                "pred_ce",
                "gate",
                "latent",
                "enc_ce",
                "separation",
                "pull",
                "inverse_action",
                "ep",
                "rollout",
                "event",
                "q",
                "reliability",
                "pre_clip_gradient_norm",
                "gradient_clip_scale",
                "learning_rate",
            ] {
                assert!(row[field].is_number(), "loss row missing numeric {field}");
            }
        }
        let checkpoint_state: TrainerState =
            read_json(&report.latest_checkpoint.join("trainer_state.json"))?;
        // Compare field-wise with a 1-ulp-scale tolerance: the persisted
        // means were divided from in-memory sums while this recomputation
        // divides the JSON-round-tripped sums.
        let expected_active = foundation_v2_active_loss_means(
            &checkpoint_state
                .foundation_v2
                .as_ref()
                .expect("foundation-v2 checkpoint state")
                .loss_sums,
            checkpoint_state
                .foundation_v2
                .as_ref()
                .expect("foundation-v2 checkpoint state")
                .loss_steps,
        );
        let persisted = serde_json::to_value(&checkpoint_state.active_sums)?;
        let expected = serde_json::to_value(&expected_active)?;
        for (field, expected_value) in expected.as_object().expect("loss-means object") {
            let a = persisted[field].as_f64().expect("numeric loss-mean field");
            let b = expected_value.as_f64().expect("numeric loss-mean field");
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "active_sums.{field} diverged: {a} vs {b}"
            );
        }
        assert_eq!(foundation.profile_bundles.len(), 1);
        let bundle = &foundation.profile_bundles[0];
        for name in [
            "bundle.json",
            "trace.jsonl",
            "evidence.json",
            "report.md",
            "viewer.html",
        ] {
            assert!(bundle.join(name).is_file(), "missing {name}");
        }
        candle_graph::verify_bundle(bundle)?;
        let trace_document = candle_graph::parse_trace(bundle.join("trace.jsonl"))?;
        let contract = &trace_document.run.capture_contract;
        assert_eq!(
            contract.measurement_scope,
            candle_graph::MeasurementScope::ProfiledWork
        );
        assert_eq!(contract.operations, candle_graph::CoverageLevel::None);
        assert_eq!(contract.tensors, candle_graph::CoverageLevel::Partial);
        assert_eq!(contract.gradients, candle_graph::CoverageLevel::Complete);
        let gradient_contract = contract
            .gradient_contract
            .as_ref()
            .expect("complete gradient coverage has an exact contract");
        assert!(gradient_contract
            .expected
            .iter()
            .all(|gradient| gradient.root == "vb/pre_clip"));
        assert_eq!(
            gradient_contract
                .families
                .iter()
                .map(|family| family.family.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["auxiliary_decoders", "exact_decoder", "observers", "world",])
        );
        assert!(fs::read_dir(bundle.parent().expect("profile directory"))?
            .filter_map(std::result::Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().starts_with('.')));
        let campaign = CampaignManifest::load(&cfg.output_dir.join("profile/campaign.json"))?;
        assert_eq!(campaign.entrypoint, PROFILE_ENTRYPOINT);
        assert_eq!(campaign.planned.len(), 1);
        assert_eq!(campaign.planned[0].capture_step, 2);
        assert_eq!(campaign.planned[0].bundle, "update-000000000002");
        let campaign_status =
            candle_graph::campaign_status(&cfg.output_dir.join("profile/campaign.json"))?;
        assert_eq!(campaign_status.published, 1);
        assert_eq!(campaign_status.missing, 0);
        let required = [
            "seam/out_y",
            "seam/current_canonical",
            "seam/predicted_canonical",
            "seam/gate_logits",
        ];
        let mut seen = BTreeMap::new();
        let mut saw_pre_clip_gradients = false;
        for line in fs::read_to_string(bundle.join("trace.jsonl"))?.lines() {
            let event: serde_json::Value = serde_json::from_str(line)?;
            if event.get("kind").and_then(serde_json::Value::as_str) == Some("tensor_stats") {
                if let Some(label) = event.get("label").and_then(serde_json::Value::as_str) {
                    seen.insert(label.to_string(), ());
                }
            }
            if event.get("kind").and_then(serde_json::Value::as_str) == Some("gradient")
                && event.get("root").and_then(serde_json::Value::as_str) == Some("vb/pre_clip")
            {
                saw_pre_clip_gradients = true;
            }
        }
        for label in required {
            assert!(seen.contains_key(label), "missing tensor stats for {label}");
        }
        for label in [
            "loss/total",
            "optimizer/pre_clip_gradient_norm",
            "optimizer/clip_scale",
            "optimizer/clipped",
            "optimizer/learning_rate",
            "objective/ep_weight",
        ] {
            assert!(
                seen.contains_key(label),
                "missing scalar evidence for {label}"
            );
        }
        assert!(saw_pre_clip_gradients);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn model_treatment_flags_are_contract_recorded_and_foundation_legal() -> Result<()> {
        let mut cfg = TrainConfig::default();
        cfg.apply_foundation_v2_recipe();
        cfg.seed = 3;
        cfg.physical_batch = 16;
        cfg.bf16_recurrent_core = true;
        cfg.copy_bypass_gate = true;
        cfg.grid_scaled_action_impulse = true;
        cfg.copy_gate_bias_prior = Some(0.02);
        cfg.decode_composition = DecodeComposition::JointCopyMixture;
        // Multi-treatment arms need the explicit attribution waiver.
        assert!(cfg.validate().is_err());
        cfg.allow_multi_treatment_arm = true;
        cfg.validate()?;
        let model_cfg = cfg.model_config();
        assert!(model_cfg.bf16_recurrent_core);
        assert!(model_cfg.copy_bypass_gate);
        assert!(model_cfg.grid_scaled_action_impulse);
        assert_eq!(model_cfg.copy_gate_bias_prior, Some(0.02));
        assert_eq!(
            model_cfg.decode_composition,
            DecodeComposition::JointCopyMixture
        );
        // A resume across arms must fail closed: every treatment field and
        // the waiver must individually break contract equality.
        let contract = TrainingContract::from(&cfg);
        let variants: [fn(&mut TrainConfig); 7] = [
            |c| c.bf16_recurrent_core = false,
            |c| c.copy_bypass_gate = false,
            |c| c.copy_gate_bias_prior = None,
            |c| c.grid_scaled_action_impulse = false,
            |c| c.decode_composition = DecodeComposition::LegacyHardGate,
            |c| c.positional_value_readout = !c.positional_value_readout,
            |c| c.allow_multi_treatment_arm = false,
        ];
        for mutate in variants {
            let mut control = cfg.clone();
            mutate(&mut control);
            assert_ne!(
                contract,
                TrainingContract::from(&control),
                "a treatment field failed to participate in the resume contract"
            );
        }
        let mut changed_profiles = cfg.clone();
        changed_profiles.profile_updates = vec![2, 4];
        assert_ne!(contract, TrainingContract::from(&changed_profiles));
        // Applicability is enforced for the silently-inert combinations.
        let mut inert = ModelConfig {
            spatial_action_field: false,
            world_core_v2: false,
            grid_scaled_action_impulse: true,
            ..ModelConfig::default()
        };
        assert!(inert.validate().is_err());
        inert.grid_scaled_action_impulse = false;
        inert.decode_composition = DecodeComposition::JointCopyMixture;
        assert!(!inert.world_core_v4);
        assert!(inert.validate().is_err());
        Ok(())
    }

    #[test]
    fn legacy_contract_adopts_configured_profile_updates() -> Result<()> {
        let mut cfg = TrainConfig::default();
        cfg.apply_foundation_v2_recipe();
        cfg.profile_updates = vec![2, 1_024];
        let requested = TrainingContract::from(&cfg);
        let mut json = serde_json::to_value(&requested)?;
        json.as_object_mut()
            .expect("contract serializes as an object")
            .remove("profile_updates");
        let mut legacy: TrainingContract = serde_json::from_value(json)?;
        assert_eq!(legacy.profile_updates, None);
        legacy.profile_updates = requested.profile_updates.clone();
        assert_eq!(legacy, requested);
        Ok(())
    }

    // ---- ADR 0005 world-core v6 -------------------------------------------

    fn tiny_v5_model_cfg() -> ModelConfig {
        ModelConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 2,
            world_core_v4: true,
            world_core_v5: true,
            spatial_action_field: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            residual_y_update: true,
            warm_start_y: true,
            ..ModelConfig::default()
        }
    }

    fn fresh_model(cfg: ModelConfig, seed: u64, device: &Device) -> Result<(WorldModel, VarMap)> {
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, device))?;
        reinit_varmap_deterministic(&varmap, seed)?;
        zero_action_film_projections(&varmap)?;
        zero_operator_conditioning_projection(&varmap)?;
        zero_context_film_projections(&varmap)?;
        init_copy_bypass_gate(&varmap)?;
        Ok((model, varmap))
    }

    /// Give `every`-th row a context window built from other rows' frames.
    fn inject_synthetic_context(mixed: &mut MixedStreamBatch, every: usize) -> usize {
        let donors = mixed
            .transitions()
            .map(|row| (row.current.clone(), row.action.clone(), row.next.clone()))
            .collect::<Vec<_>>();
        let mut injected = 0;
        for (index, row) in mixed.transitions_mut().enumerate() {
            if index % every != 0 {
                continue;
            }
            let len = 1 + (index / every) % 3;
            row.context = (0..len)
                .map(|offset| {
                    let (current, action, next) = &donors[(index + offset + 1) % donors.len()];
                    crate::p2::data::ContextTransition {
                        current: current.clone(),
                        action: action.clone(),
                        next: next.clone(),
                    }
                })
                .collect();
            injected += 1;
        }
        injected
    }

    #[test]
    fn v6_recipe_defaults_to_two_by_two_and_v5_stays_two_by_two() {
        let mut v6 = TrainConfig {
            world_core_v6: true,
            ..TrainConfig::default()
        };
        v6.apply_foundation_v2_recipe();
        assert_eq!(v6.inner_steps, V6_RECURSION_STEPS);
        assert_eq!(v6.outer_steps, V6_RECURSION_STEPS);
        assert_eq!(v6.model_config().inner_steps, 2);
        assert_eq!(v6.model_config().outer_steps, 2);
        let mut v5 = TrainConfig::default();
        v5.apply_foundation_v2_recipe();
        assert_eq!((v5.inner_steps, v5.outer_steps), (2, 2));
    }

    /// 2026-09-04 audit: `--grad-accum` must survive the recipe. E2 was
    /// launched with `--grad-accum 2` and silently ran at 1.
    #[test]
    fn recipes_preserve_caller_owned_grad_accum() {
        for recipe in [TrainingRecipe::FoundationV2, TrainingRecipe::FullV4] {
            let mut cfg = TrainConfig {
                world_core_v6: recipe == TrainingRecipe::FoundationV2,
                data_contract_v6: recipe == TrainingRecipe::FoundationV2,
                grad_accum: 8,
                ..TrainConfig::default()
            };
            match recipe {
                TrainingRecipe::FoundationV2 => cfg.apply_foundation_v2_recipe(),
                _ => cfg.apply_full_v4_recipe(),
            }
            cfg.physical_batch = 128;
            assert_eq!(cfg.grad_accum, 8, "{recipe:?} recipe overwrote grad_accum");
            if recipe == TrainingRecipe::FoundationV2 {
                cfg.validate()
                    .expect("accumulation is caller-owned under foundation-v2");
                assert_eq!(TrainingContract::from(&cfg).grad_accum, 8);
            }
        }
    }

    /// The deferred depth treatment runs a v6 arm at 3x3 through the recipe,
    /// so it validates like the 2x2 baseline and is recorded in the contract
    /// via `inner_steps`/`outer_steps`.
    #[test]
    fn v6_recursion_override_reaches_the_recipe_and_contract() {
        let mut shallow = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            v6_recursion_steps: 2,
            ..TrainConfig::default()
        };
        shallow.apply_foundation_v2_recipe();
        shallow.physical_batch = 64;
        shallow.validate().expect("2x2 v6 arm validates");
        assert_eq!((shallow.inner_steps, shallow.outer_steps), (2, 2));
        let deep = {
            let mut c = shallow.clone();
            c.v6_recursion_steps = 3;
            c.apply_foundation_v2_recipe();
            c
        };
        deep.validate().expect("3x3 v6 treatment validates");
        assert_ne!(
            TrainingContract::from(&shallow),
            TrainingContract::from(&deep),
            "depth is trajectory-changing and must split the contract"
        );
        let mut legacy = TrainConfig {
            v6_recursion_steps: 3,
            ..TrainConfig::default()
        };
        legacy.apply_foundation_v2_recipe();
        legacy.physical_batch = 64;
        assert!(
            legacy.validate().is_err(),
            "override without world_core_v6 is rejected"
        );
    }

    #[test]
    fn v6_config_requires_foundation_v2_and_flags_compose() {
        let plain = TrainConfig {
            world_core_v6: true,
            ..TrainConfig::default()
        };
        assert!(plain.validate().is_err());
        // Flags precede the recipe, as in `P2TrainArgs::to_config`: the recipe
        // resolves the v6 recursion depth from `world_core_v6` (ADR 0005 §3.5).
        let mut v6 = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            ..TrainConfig::default()
        };
        v6.apply_foundation_v2_recipe();
        v6.physical_batch = 64;
        v6.validate().expect("foundation-v2 + v6 is a valid arm");
        assert!(v6.model_config().world_core_v6);
        let mut orphan_init = v6.clone();
        orphan_init.world_core_v6 = false;
        orphan_init.init_context_from_v5 = Some(PathBuf::from("x"));
        assert!(orphan_init.validate().is_err());
        let contract = TrainingContract::from(&v6);
        assert!(contract.world_core_v6);
        assert!(contract.data_contract_v6);
    }

    /// A v6 config must persist as v6 everywhere the resolved identity is
    /// consumed (report schema, training contract, resume comparison), and a
    /// v5 config must keep its historical identity byte-for-byte.
    #[test]
    fn v6_config_resolves_v6_identity_and_v5_keeps_v5() -> Result<()> {
        use crate::p2::experiment::{WorldCoreFamily, WORLD_CORE_V5_SCHEMA, WORLD_CORE_V6_SCHEMA};
        let mut v6 = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            ..TrainConfig::default()
        };
        v6.apply_foundation_v2_recipe();
        v6.physical_batch = 64;
        v6.validate()?;
        let resolved_v6 = v6.resolved_experiment()?;
        assert_eq!(resolved_v6.family, WorldCoreFamily::V6);
        assert_eq!(resolved_v6.report_schema, WORLD_CORE_V6_SCHEMA);

        let mut v5 = TrainConfig::default();
        v5.apply_foundation_v2_recipe();
        v5.physical_batch = 64;
        v5.validate()?;
        let resolved_v5 = v5.resolved_experiment()?;
        assert_eq!(resolved_v5.family, WorldCoreFamily::V5);
        assert_eq!(resolved_v5.report_schema, WORLD_CORE_V5_SCHEMA);
        let v5_json = serde_json::to_value(&resolved_v5)?;
        assert_eq!(v5_json["family"], "v5");
        assert_eq!(v5_json["report_schema"], "world_core_v5");

        // The persisted contracts differ, so a v5 bundle never resumes as v6.
        let v6_contract = TrainingContract::from(&v6);
        let v5_contract = TrainingContract::from(&v5);
        assert_ne!(v6_contract.experiment, v5_contract.experiment);
        assert_eq!(
            v6_contract.experiment.as_ref().map(|e| e.family),
            Some(WorldCoreFamily::V6)
        );
        Ok(())
    }

    /// ADR 0005 §1.1: the data and model contracts are paired. The only
    /// legal v6-model / legacy-data combination is the warm-start smoke.
    #[test]
    fn v6_data_and_model_contracts_must_be_paired() {
        // The recipe resolves recursion depth from `world_core_v6`, so each arm
        // sets its flags first and applies the recipe afterwards (CLI order).
        let arm = |world_core_v6: bool, data_contract_v6: bool| {
            let mut cfg = TrainConfig {
                world_core_v6,
                data_contract_v6,
                ..TrainConfig::default()
            };
            cfg.apply_foundation_v2_recipe();
            cfg.physical_batch = 64;
            cfg
        };

        // data_contract_v6 without world_core_v6: rejected at config time.
        let data_only = arm(false, true);
        let err = data_only
            .validate()
            .expect_err("v6 data on a legacy model must fail closed");
        assert!(
            err.to_string()
                .contains("data_contract_v6 requires world_core_v6"),
            "{err:#}"
        );

        // world_core_v6 without data_contract_v6 and without warm start: rejected.
        let model_only = arm(true, false);
        let err = model_only
            .validate()
            .expect_err("v6 model on legacy data without warm start must fail closed");
        assert!(err.to_string().contains("ADR 0005 §1.1"), "{err:#}");

        // world_core_v6 without data_contract_v6 but with init_context_from_v5:
        // the warm-start smoke is allowed.
        let mut warm_start = model_only.clone();
        warm_start.init_context_from_v5 = Some(PathBuf::from("runs/v5/checkpoints/best"));
        warm_start
            .validate()
            .expect("v5 warm-start smoke on legacy data is allowed");

        // Both flags on: the §5.1 arm.
        let paired = arm(true, true);
        paired.validate().expect("paired v6 contracts validate");
        let contract = TrainingContract::from(&paired);
        assert!(contract.world_core_v6 && contract.data_contract_v6);
    }

    #[test]
    fn v5_checkpoint_into_v6_fails_closed_unless_warm_started() -> Result<()> {
        let device = Device::Cpu;
        let root = checkpoint_test_root("v6-warm-start");
        fs::create_dir_all(&root)?;
        let (v5, v5_vars) = fresh_model(tiny_v5_model_cfg(), 41, &device)?;
        let checkpoint = root.join("model.safetensors");
        v5_vars.save(&checkpoint)?;

        let v6_cfg = ModelConfig {
            world_core_v6: true,
            ..tiny_v5_model_cfg()
        };
        let (v6, v6_vars) = fresh_model(v6_cfg, 99, &device)?;
        let error =
            load_varmap_exact(&v6_vars, &checkpoint).expect_err("v5 into v6 must fail closed");
        assert!(error.to_string().contains("missing"), "{error}");
        // Warm start into a v5 target is meaningless (nothing missing) and must fail too.
        assert!(load_varmap_warm_start_context(&v5_vars, &checkpoint).is_err());
        load_varmap_warm_start_context(&v6_vars, &root)?;

        let samples = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 16,
                seed: 7,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::Train,
        )?
        .transitions()
        .cloned()
        .collect::<Vec<_>>();
        let batch = batch_from_samples(&samples, &device)?;
        let context = {
            let mut with_context = samples.clone();
            for (index, row) in with_context.iter_mut().enumerate() {
                if index % 3 == 0 {
                    row.context = vec![crate::p2::data::ContextTransition {
                        current: samples[(index + 1) % samples.len()].current.clone(),
                        action: samples[(index + 1) % samples.len()].action.clone(),
                        next: samples[(index + 1) % samples.len()].next.clone(),
                    }];
                }
            }
            ContextBatch::from_samples(&with_context, &device)?.expect("context present")
        };
        // The loaded v6 model computes the v5 function, with or without
        // context. Row 63 handling differs by design (§1.3), so compare on
        // frames whose status row is already EMPTY: only weights are under test.
        let empty_status = {
            let mut pixels = batch.frames.flatten_all()?.to_vec1::<u8>()?;
            for frame in pixels.chunks_exact_mut(FRAME_SIDE * FRAME_SIDE) {
                frame[(FRAME_SIDE - 1) * FRAME_SIDE..].fill(0);
            }
            Tensor::from_vec(pixels, batch.frames.dims(), &device)?
        };
        let reference = v5.forward_with_operator_conditioning(
            &empty_status,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            &batch.operator_conditioning,
        )?;
        for context in [None, Some(&context)] {
            let out = v6.forward_with_depth_and_operator_conditioning_with_context(
                &empty_status,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
                context,
                RecursionDepth::from_config(v6.config()),
                0.0,
                None,
            )?;
            assert_eq!(
                reference.y.flatten_all()?.to_vec1::<f32>()?,
                out.y.flatten_all()?.to_vec1::<f32>()?,
                "warm-started v6 must reproduce v5 (context = {})",
                context.is_some()
            );
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn foundation_v2_loss_is_unchanged_by_context_plumbing_at_init() -> Result<()> {
        let device = Device::Cpu;
        let mut mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 32,
                seed: 61,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            0.5,
            0,
            V5DataSplit::Train,
        )?;
        // Legacy rows: no context is ever materialized.
        assert!(prepare_foundation_v2_batch_host(&mixed)?
            .context()
            .is_none());
        let objective = FoundationV2ObjectiveConfig::default();
        let scalar = |breakdown: FoundationV2LossBreakdown| -> Result<f32> {
            Ok(breakdown.total.to_dtype(DType::F32)?.to_scalar::<f32>()?)
        };
        let (v5, _) = fresh_model(tiny_v5_model_cfg(), 23, &device)?;
        let v5_plain = scalar(foundation_v2_training_loss(
            &v5, &mixed, &device, objective,
        )?)?;

        let injected = inject_synthetic_context(&mut mixed, 4);
        assert!(injected > 0);
        let host = prepare_foundation_v2_batch_host(&mixed)?;
        let context = host.context().expect("context rows are materialized");
        assert_eq!(context.batch, 32);
        assert_eq!(context.k, 3);
        // rows 0,4,...,28 carry windows of length 1,2,3,1,2,3,1,2.
        assert_eq!(context.valid.iter().filter(|v| **v != 0.0).count(), 15);

        // v5 ignores context rows entirely.
        let v5_ctx = scalar(foundation_v2_training_loss(
            &v5, &mixed, &device, objective,
        )?)?;
        assert!(v5_plain.is_finite());
        assert_eq!(v5_plain, v5_ctx, "legacy model must ignore context rows");

        // A v6 model decodes all 64 rows, so it is paired with the v6 data
        // contract (ADR 0005 §1.1); the legacy 63-row batch fails closed.
        let (v6, _) = fresh_model(
            ModelConfig {
                world_core_v6: true,
                ..tiny_v5_model_cfg()
            },
            23,
            &device,
        )?;
        assert!(foundation_v2_training_loss(&v6, &mixed, &device, objective).is_err());
        let mut mixed_v6 = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 32,
                seed: 61,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            0.5,
            0,
            V5DataSplit::Train,
        )?;
        let v6_plain = scalar(foundation_v2_training_loss(
            &v6, &mixed_v6, &device, objective,
        )?)?;
        assert!(inject_synthetic_context(&mut mixed_v6, 4) > 0);
        // v6 at zero context FiLM is bit-identical with and without context.
        let v6_ctx = scalar(foundation_v2_training_loss(
            &v6, &mixed_v6, &device, objective,
        )?)?;
        assert!(v6_plain.is_finite());
        assert_eq!(
            v6_plain, v6_ctx,
            "zero context FiLM must not change the v6 loss"
        );
        Ok(())
    }
}
