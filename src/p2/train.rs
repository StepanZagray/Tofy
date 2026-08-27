//! P2 LeWorld / TRM training on synthetic curriculum only.

use crate::domain::Split;
use crate::gpu_lock::{GpuSessionGuard, TrainPidGuard};
use crate::p2::branch_learning::{branch_learning_loss, BranchLearningAudit, BranchLearningConfig};
use crate::p2::cg_profile::{CaptureSpec, ProfileState, RepresentativeUpdateCapture};
use crate::p2::consumer_transition::ConsumerTransition;
use crate::p2::data::{
    compose_mixed_stream_batch, foundation_v2_stream_schedule, generate_curriculum, ArcFrame,
    ContentMask, EventLabelCensus, FactualBatch, MixedStreamBatch, MixedStreamConfig,
    MixedStreamKind, TransitionSample, V5DataSplit, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::eval::{evaluate_gate_support_with_content_masks, GateSupportMetrics};
use crate::p2::experiment::{
    ConsumerReadoutTopology, ExperimentRequest, ResolvedExperiment, SigregPopulation,
    SigregStatistic, TrainingRecipe,
};
use crate::p2::grounding::PatchGroundingMode;
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, zero_action_film_projections, ModelConfig, PtrmConfig,
    RecursionDepth, RecursionOpts, WorldModel, ACTION_VOCAB, DEFAULT_NUM_EVENTS, LEGACY_PATCH_SIZE,
    PALETTE_SIZE, PATCH_SIZE, PREFIX_HORIZONS,
};
use crate::p2::muon::MUON_RMS_SCALE;
use crate::p2::optimizer::{
    accumulate_parameter_gradients, clip_gradients_gpu_with_stats, CheckpointHybridOptimizer,
    ModelEma,
};
use crate::p2::prefetch::{
    BatchPrefetcher, MixedStreamBatchPrefetcher, PrefetchRequest, PrefetchScope,
};
use crate::p2::sigreg::{sigreg_epps_pulley_seeded, sigreg_quantile_seeded};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var, D};
use candle_graph::{ExecutionStep, SpanKind};
use candle_nn::init::FanInOut;
use candle_nn::optim::ParamsAdamW;
use candle_nn::{VarBuilder, VarMap};
use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
/// Global gradient L2 clip for recursive training stability.
const MAX_GRAD_NORM: f64 = 1.0;
const FOUNDATION_V2_EP_MIN_WEIGHT: f64 = 1e-4;
const FOUNDATION_V2_EP_MAX_WEIGHT: f64 = 0.1;
const FOUNDATION_V2_EP_GRADIENT_BUDGET: f64 = 0.3;
const FOUNDATION_V2_GATE_EVERY: u64 = 1_024;
const FOUNDATION_V2_PERMANENT_EVERY: u64 = 2_048;
const FOUNDATION_V2_GATE_ROWS: usize = 512;
const FOUNDATION_V2_GATE_SEED: u64 = 0xF0A2_DA7A_0000_0005;
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
/// with the reachable separation hinge and budget-exact EP controller.
pub const FOUNDATION_OBJECTIVE_REVISION: u32 = 2;
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
        changed_weight: ratio.clamp(1.0, 64.0),
        unchanged_weight: 1.0,
    })
}

/// Pure multiplicative EP controller. The returned weight targets
/// `weight * ||g_ep|| = 0.3 * ||g_pred||`, then applies the ADR bounds. When
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
    let target =
        FOUNDATION_V2_EP_GRADIENT_BUDGET * prediction_gradient_l2 / ep_gradient_l2;
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
pub fn separation_hinge_term(
    left: &Tensor,
    right: &Tensor,
    margin: f64,
) -> Result<Tensor> {
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationV2GateEvaluation {
    pub step: u64,
    pub metrics: GateSupportMetrics,
    pub running_best_before: Option<f64>,
    pub running_best_after: Option<f64>,
    pub gates: Vec<FoundationV2GateResult>,
}

/// Checkpoint-promotion selection metric. `ChangedExact` is the historical
/// default; `FullExact` selects on full-transition exactness (unchanged
/// pixels included) without touching the gate or collapse-floor semantics.
/// `ComposedExactGuarded` selects on composed all-row exactness (the deployed
/// copy-gate output, no-op rows included) and additionally refuses a
/// candidate whose content or padding false-edit rate regresses versus the
/// incumbent best — the deployed-behavior selection the canonical library
/// prescribes; preregister it for new evidence runs.
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
        PromotionMetric::ComposedExactGuarded => metrics.one_step_all_rows_exact,
    }
}

/// Guard for `ComposedExactGuarded`: a candidate may not regress either
/// false-edit rate versus the incumbent best. A measured incumbent rate with
/// a missing candidate rate fails conservatively.
fn foundation_v2_false_edit_guard(
    incumbent: &GateSupportMetrics,
    candidate: &GateSupportMetrics,
) -> bool {
    fn not_regressed(incumbent: Option<f64>, candidate: Option<f64>) -> bool {
        match (incumbent, candidate) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(prior), Some(current)) => current <= prior + 1e-12,
        }
    }
    not_regressed(incumbent.false_edit_rate, candidate.false_edit_rate)
        && not_regressed(
            incumbent.padding_false_edit_rate,
            candidate.padding_false_edit_rate,
        )
}

/// Whether `candidate` replaces `incumbent` under the configured promotion
/// metric: a strictly larger metric value, plus the false-edit guard for
/// `ComposedExactGuarded`.
pub fn foundation_v2_candidate_improves(
    metric: PromotionMetric,
    incumbent: Option<&GateSupportMetrics>,
    candidate: &GateSupportMetrics,
) -> bool {
    let Some(value) = foundation_v2_promotion_value(metric, candidate) else {
        return false;
    };
    let better = incumbent
        .and_then(|metrics| foundation_v2_promotion_value(metric, metrics))
        .is_none_or(|best| value > best);
    match metric {
        PromotionMetric::ComposedExactGuarded => {
            better
                && incumbent
                    .is_none_or(|metrics| foundation_v2_false_edit_guard(metrics, candidate))
        }
        _ => better,
    }
}

/// The gate evaluation currently holding the promotion under `metric`,
/// obtained by replaying the strict-improvement scan over the history.
pub fn foundation_v2_best_evaluation<'a>(
    metric: PromotionMetric,
    gate_history: &'a [FoundationV2GateEvaluation],
) -> Option<&'a FoundationV2GateEvaluation> {
    let mut best: Option<&FoundationV2GateEvaluation> = None;
    for evaluation in gate_history {
        if foundation_v2_candidate_improves(
            metric,
            best.map(|evaluation| &evaluation.metrics),
            &evaluation.metrics,
        ) {
            best = Some(evaluation);
        }
    }
    best
}

/// Whether the new measurement beats the running best of the configured
/// promotion metric. `ChangedExact` compares against the persisted
/// `best_changed_exact` exactly as before; the other metrics replay the gate
/// history through the same strict-improvement rule used for selection.
pub fn foundation_v2_promotion_improved(
    metric: PromotionMetric,
    best_changed_exact: Option<f64>,
    gate_history: &[FoundationV2GateEvaluation],
    metrics: &GateSupportMetrics,
) -> bool {
    if metric == PromotionMetric::ChangedExact {
        return foundation_v2_promotion_value(metric, metrics)
            .is_some_and(|current| best_changed_exact.is_none_or(|best| current > best));
    }
    foundation_v2_candidate_improves(
        metric,
        foundation_v2_best_evaluation(metric, gate_history).map(|evaluation| &evaluation.metrics),
        metrics,
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
) -> FoundationV2GateEvaluation {
    let current_exact = metrics.one_step_changed_exact;
    let running_best_after = match (running_best, current_exact) {
        (Some(best), Some(current)) => Some(best.max(current)),
        (None, Some(current)) => Some(current),
        (best, None) => best,
    };
    let collapse_floor = running_best_after.map(|best| best * 0.8);
    // Absolute-quality gates get the same warmup grace as foreground
    // reconstruction: before step 4096 the model cannot yet be expected to
    // beat latent copy or show action sensitivity, so those gates are
    // measured and logged but PASS by fiat. The collapse detector stays
    // active from the first evaluation because it is relative to the run's
    // own best.
    let warmup_done = step >= 4_096;
    // Foreground reconstruction ramps slowly under the changed-pixel-weighted
    // CE (0.086 -> 0.639 over the first 4096 steps of the first launch);
    // enforce it once the decoder has had half the pre-decay run to mature.
    let foreground_active = step >= 8_192;
    let gates = vec![
        FoundationV2GateResult {
            // Latent-MSE improvement over copy measures proximity in a space
            // the v5 objective does not optimize: pixel CE trains exact
            // decodability, and the pixel-space copy baseline scores zero on
            // changed pixels by definition while changed-exact is tracked by
            // the collapse gate. Kept as a logged diagnostic; never enforced.
            name: "positive_improvement".into(),
            passed: true,
            measured: metrics.improvement_fraction,
            threshold: "diagnostic-only (latent-MSE; superseded by pixel-space gates)".into(),
        },
        FoundationV2GateResult {
            name: "shuffled_action_ratio".into(),
            passed: !warmup_done
                || metrics
                    .shuffled_action_changed_pixel_ratio
                    .is_some_and(|value| value <= 0.95),
            measured: metrics.shuffled_action_changed_pixel_ratio,
            threshold: if warmup_done {
                "<= 0.95".into()
            } else {
                "warmup PASS until step 4096".into()
            },
        },
        FoundationV2GateResult {
            // Under changed-pixel-weighted CE the encoded-state foreground
            // reconstruction asymptotes near 0.67 (first launch measured
            // 0.639 -> 0.675 over steps 4096..9216 while changed-exact kept
            // climbing). The gate exists to catch decoder collapse, so it is
            // a regression floor below the observed asymptote, not an
            // aspirational target.
            name: "foreground_reconstruction".into(),
            passed: !foreground_active
                || metrics
                    .foreground_reconstruction_accuracy
                    .is_some_and(|value| value >= 0.60),
            measured: metrics.foreground_reconstruction_accuracy,
            threshold: if foreground_active {
                ">= 0.60 (collapse floor)".into()
            } else {
                "warmup PASS until step 8192".into()
            },
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
        },
    ];
    FoundationV2GateEvaluation {
        step,
        metrics,
        running_best_before: running_best,
        running_best_after,
        gates,
    }
}

/// Fail closed after the same named gate fails in the two latest evaluations.
pub fn foundation_v2_gate_history_aborts(history: &[FoundationV2GateEvaluation]) -> bool {
    if history.len() < 2 {
        return false;
    }
    let Some((latest, prior)) = history.last().zip(history.get(history.len() - 2)) else {
        return false;
    };
    latest.gates.iter().any(|gate| {
        !gate.passed
            && prior
                .gates
                .iter()
                .find(|candidate| candidate.name == gate.name)
                .is_some_and(|candidate| !candidate.passed)
    })
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
    /// Intentionally checkpoint-incompatible action-faithful world core.
    #[serde(default)]
    pub world_core_v2: bool,
    /// V3 experiment schema: V2 topology plus residual spatial conditioning
    /// and scale-normalized factual displacement health.
    #[serde(default)]
    pub world_core_v3: bool,
    #[serde(default)]
    pub world_core_v4: bool,
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

fn sync_cuda_device(device: &Device) -> Result<()> {
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(())
}

/// Config fields safe to persist for resume/eval (omit per-run hooks).
fn persist_train_config(cfg: &TrainConfig) -> TrainConfig {
    let mut persisted = cfg.clone();
    persisted.resume = None;
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
            world_core_v2: false,
            world_core_v3: false,
            world_core_v4: false,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: default_spatial_action_residual_scale(),
            split_ce_weighting: SplitCeWeighting::CurrentDouble,
            split_ce_changed_budget: None,
            promotion_metric: PromotionMetric::ChangedExact,
            branch_learning: BranchLearningConfig::default(),
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
            sigreg_projector: self.sigreg_projector,
            sigreg_projector_dim: self.sigreg_projector_dim,
            spatial_action_field: self.spatial_action_field,
            spatial_action_residual: self.spatial_action_residual,
            spatial_action_residual_scale: self.spatial_action_residual_scale,
            world_core_v2: self.world_core_v2,
            world_core_v3: self.world_core_v3,
            world_core_v4: self.world_core_v4,
            world_core_v5: self.recipe == TrainingRecipe::FoundationV2,
            consumer_readout: self.consumer_readout,
        }
    }

    /// Resolve the one supported V4 training contract. Runtime/provenance
    /// values (seed, steps, batch, device, output, checkpoints) remain caller
    /// controlled; model and objective choices do not.
    pub fn apply_full_v4_recipe(&mut self) {
        self.recipe = TrainingRecipe::FullV4;
        self.lessons = DEFAULT_LESSONS.iter().map(|s| (*s).to_string()).collect();
        self.grad_accum = 1;
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
        self.grad_accum = 1;
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
        self.inner_steps = 2;
        self.outer_steps = 2;
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
        canonical.device = self.device.clone();
        canonical.output_dir = self.output_dir.clone();
        canonical.resume = self.resume.clone();
        canonical.allow_batch_schedule_migration = self.allow_batch_schedule_migration;
        canonical.checkpoint_every_steps = self.checkpoint_every_steps;
        canonical.max_steps_this_run = self.max_steps_this_run;
        canonical.init_seed = self.init_seed;
        canonical.profile_update = self.profile_update;
        canonical.pressure_updates = self.pressure_updates.clone();
        canonical.prefetch_batches = self.prefetch_batches;
        canonical.data_workers = self.data_workers;
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
    /// Ordered provenance fingerprint of every generated training row consumed
    /// by completed optimizer updates.
    pub training_population_fingerprint: String,
    /// Cryptographic chain over provenance plus complete current/next frames.
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
            sigreg_max_rows: cfg.sigreg_max_rows,
            sigreg_target: cfg.sigreg_target,
            sigreg_temporal_window: cfg.sigreg_temporal_window,
            sigreg_global_mix: cfg.sigreg_global_mix,
            world_core_v2: cfg.world_core_v2,
            world_core_v3: cfg.world_core_v3,
            world_core_v4: cfg.world_core_v4,
            spatial_action_field: cfg.spatial_action_field,
            spatial_action_residual: cfg.spatial_action_residual,
            spatial_action_residual_scale: cfg.spatial_action_residual_scale,
            split_ce_weighting: cfg.split_ce_weighting,
            split_ce_changed_budget: cfg.split_ce_changed_budget,
            promotion_metric: cfg.promotion_metric,
            branch_learning: cfg.branch_learning.clone(),
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
    pub policy_schema: String,
}

/// Version of the in-trainer gate policy (thresholds, warmups, abort rule,
/// shuffle construction). Bump on any change so a resumed run cannot compare
/// new measurements against bests recorded under a different policy.
const FOUNDATION_V2_GATE_POLICY_SCHEMA: &str = "p2.gate_policy.v2";

fn foundation_v2_gate_population_identity(
    samples: &[TransitionSample],
    masks: &[ContentMask],
) -> Result<GatePopulationIdentity> {
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
    Ok(GatePopulationIdentity {
        rows_sha256: format!("sha256:{:x}", rows.finalize()),
        masks_sha256: format!("sha256:{:x}", mask_digest.finalize()),
        policy_schema: FOUNDATION_V2_GATE_POLICY_SCHEMA.into(),
    })
}

fn default_training_population_hash() -> u64 {
    FNV1A64_OFFSET
}

fn update_training_population(state: &mut TrainerState, samples: &[TransitionSample]) {
    fn bytes(hash: &mut u64, value: &[u8]) {
        for byte in value {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV1A64_PRIME);
        }
    }

    for sample in samples {
        let mut digest = Sha256::new();
        digest.update(state.training_content_hash);
        digest.update(sample.seed.to_le_bytes());
        digest.update(sample.episode_id.to_le_bytes());
        digest.update(sample.transition_index.to_le_bytes());
        digest.update((sample.family.len() as u64).to_le_bytes());
        digest.update(sample.family.as_bytes());
        digest.update(sample.provenance.content_width.to_le_bytes());
        digest.update(sample.provenance.content_height.to_le_bytes());
        digest.update((sample.provenance.source_kind.len() as u64).to_le_bytes());
        digest.update(sample.provenance.source_kind.as_bytes());
        digest.update((sample.provenance.trajectory_id.len() as u64).to_le_bytes());
        digest.update(sample.provenance.trajectory_id.as_bytes());
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
        state.training_content_hash = digest.finalize().into();
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

fn sample_frames_to_indices(
    samples: &[TransitionSample],
    next: bool,
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
            let frame = if next { &sample.next } else { &sample.current };
            ensure_fixed_frame(frame)?;
            if let Some(&pix) = frame.pixels.iter().find(|&&p| p as usize >= PALETTE_SIZE) {
                bail!("palette value {pix} out of 0..{PALETTE_SIZE}");
            }
            slot.copy_from_slice(&frame.pixels);
            Ok(())
        })?;
    Tensor::from_vec(indices, (samples.len(), 1, FRAME_SIDE, FRAME_SIDE), device)
        .map_err(Into::into)
}

fn sample_frame_pair_to_indices(
    samples: &[TransitionSample],
    device: &Device,
) -> (Result<Tensor>, Result<Tensor>) {
    // CUDA tensor construction shares one device stream. Building both frame
    // tensors from separate Rayon workers can race their host-to-device copies,
    // leaving otherwise validated palette indices corrupted by the time the
    // embedding kernel reads them. CPU construction is independent and keeps
    // the parallel path.
    if device.is_cuda() {
        (
            sample_frames_to_indices(samples, false, device),
            sample_frames_to_indices(samples, true, device),
        )
    } else {
        rayon::join(
            || sample_frames_to_indices(samples, false, device),
            || sample_frames_to_indices(samples, true, device),
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
    let b = samples.len();
    let mut targets = vec![0f32; b * DEFAULT_NUM_EVENTS];
    let mut mask = vec![0f32; b * DEFAULT_NUM_EVENTS];
    for (i, s) in samples.iter().enumerate() {
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
    pub actions: Tensor,
    /// Normalized `(x,y)` for ACTION6, zeros for simple actions.
    pub action_coords: Tensor,
    pub goals: Tensor,
    pub event_targets: Tensor,
    pub event_mask: Tensor,
    pub factual: Option<FactualBatch>,
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
    let actions: Vec<u32> = samples
        .par_iter()
        .map(|sample| {
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
        .map(|sample| match (sample.action.x, sample.action.y) {
            (Some(x), Some(y)) => Ok([f32::from(x) / 63.0, f32::from(y) / 63.0]),
            (None, None) => Ok([0.0, 0.0]),
            _ => Err(anyhow::anyhow!("action coordinate pair is incomplete")),
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
    let rows = factual.as_ref().map_or(samples, FactualBatch::rows);
    let (frames, next_frames) = sample_frame_pair_to_indices(rows, device);
    let frames = frames?;
    let next_frames = next_frames?;
    let (actions, action_coords) = action_tensors_from_samples(rows, device)?;
    let goals: Vec<f32> = rows
        .iter()
        .flat_map(|s| s.goal_features.values.iter().copied())
        .collect();
    let goals = Tensor::from_vec(goals, (rows.len(), GOAL_FEATURES_DIM), device)?;
    let (event_targets, event_mask) = event_targets_and_mask(rows, device)?;
    Ok(BatchTensors {
        frames,
        next_frames,
        actions,
        action_coords,
        goals,
        event_targets,
        event_mask,
        factual,
    })
}

pub fn ordered_trace_from_samples(
    samples: &[TransitionSample],
    device: &Device,
) -> Result<OrderedTraceTensors> {
    if samples.len() < 2 {
        bail!("ordered trace requires at least two transitions");
    }
    let (frames, next_frames) = sample_frame_pair_to_indices(samples, device);
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

fn bounded_sigreg_loss(raw: &Tensor) -> Result<Tensor> {
    smooth_cap_nonnegative(raw, SIGREG_LOSS_CAP)
}

pub(crate) fn sigreg_loss_for_stack(stack: &Tensor, cfg: &TrainConfig, seed: u64) -> Result<Tensor> {
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
    pub rollout_enabled: bool,
    pub split_ce_weighting: SplitCeWeighting,
    pub split_ce_changed_budget: Option<f64>,
}

impl Default for FoundationV2ObjectiveConfig {
    fn default() -> Self {
        Self {
            ep_weight: 0.01,
            sigreg_projections: 8,
            sigreg_knots: 5,
            sigreg_seed: 1,
            rollout_enabled: false,
            split_ce_weighting: SplitCeWeighting::CurrentDouble,
            split_ce_changed_budget: None,
        }
    }
}

#[derive(Debug, Clone)]
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
    pub rollout_fragments: usize,
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

fn foundation_v2_unimix_ce(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let pixels = labels.elem_count();
    let probs =
        candle_nn::ops::softmax(logits, D::Minus1)?.affine(0.99, 0.01 / PALETTE_SIZE as f64)?;
    probs
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
    /// `clamp((1-p)/p, 1, 64)`, so the per-pixel coefficient ratio is
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

pub(crate) fn latent_content_mask(
    masks: &[&ContentMask],
    latent_height: usize,
    latent_width: usize,
    device: &Device,
) -> Result<Tensor> {
    if masks.is_empty()
        || latent_height == 0
        || latent_width == 0
        || !FRAME_SIDE.is_multiple_of(latent_height)
        || !FRAME_SIDE.is_multiple_of(latent_width)
    {
        bail!("invalid latent/content-mask geometry");
    }
    let patch_height = FRAME_SIDE / latent_height;
    let patch_width = FRAME_SIDE / latent_width;
    let mut values = Vec::with_capacity(masks.len() * latent_height * latent_width);
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
    Tensor::from_vec(
        values,
        (masks.len(), 1, latent_height, latent_width),
        device,
    )
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
    let denominator = mask.sum_all()?.affine(channels as f64, 0.0)?;
    numerator.broadcast_div(&denominator).map_err(Into::into)
}

fn foundation_v2_rollout_loss(
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    device: &Device,
) -> Result<(Tensor, usize)> {
    let mut fragments =
        BTreeMap::<(u64, u64, String), Vec<(&TransitionSample, &ContentMask)>>::new();
    for sample in mixed
        .samples()
        .iter()
        .filter(|sample| sample.provenance.stream == MixedStreamKind::SequentialFragments)
    {
        let transition = sample.transition();
        fragments
            .entry((
                transition.seed,
                transition.episode_id,
                transition.family.clone(),
            ))
            .or_default()
            .push((transition, &sample.content_mask));
    }
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut second_masks = Vec::new();
    for fragment in fragments.values_mut() {
        fragment.sort_by_key(|(sample, _)| sample.transition_index);
        if let Some(pair) = fragment
            .windows(2)
            .find(|pair| pair[1].0.transition_index == pair[0].0.transition_index + 1)
        {
            first.push(pair[0].0.clone());
            second.push(pair[1].0.clone());
            second_masks.push(pair[1].1);
        }
    }
    if first.len() < FOUNDATION_V2_MIN_ROLLOUT_FRAGMENTS {
        return Ok((Tensor::zeros((), DType::F32, device)?, first.len()));
    }
    let first_batch = batch_from_samples(&first, device)?;
    let second_batch = batch_from_samples(&second, device)?;
    let (current, target_h2) = model
        .encode_state_pair_for_training(&first_batch.frames, &second_batch.next_frames)
        .map(|encoded| (encoded.current, encoded.next))?;
    let current_canonical = model.canonical_representation(&current)?;
    let first_out = model.full_v4_training_latents_from_encoded_state(
        &current,
        &current_canonical,
        &first_batch.actions,
        &first_batch.action_coords,
        RecursionDepth::from_config(model.config()),
        0.0,
        None,
        RecursionOpts::training(true),
    )?;
    let h1_canonical = model.canonical_representation(&first_out.y)?;
    let second_out = model.full_v4_training_latents_from_encoded_state(
        &first_out.y,
        &h1_canonical,
        &second_batch.actions,
        &second_batch.action_coords,
        RecursionDepth::from_config(model.config()),
        0.0,
        None,
        RecursionOpts::training(true),
    )?;
    let (_, _, height, width) = second_out.y.dims4()?;
    let content_mask = latent_content_mask(&second_masks, height, width, device)?;
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
        first.len(),
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
    let batch_size = predicted_latent.dim(0)?;
    if content.dims() != [batch_size, FRAME_SIDE - 1, FRAME_SIDE] {
        bail!("foundation-v2 Q content mask has the wrong shape");
    }
    let current_labels = current_frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let target_labels = next_frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let changed = current_labels.ne(&target_labels)?.to_dtype(DType::F32)?;
    let composed = model.composed_gameplay_decode(&predicted_latent.detach(), current_frames)?;
    let correct = composed.eq(&target_labels)?.to_dtype(DType::F32)?;
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
    Ok(changed_count_per_sample
        .gt(0.0)?
        .where_cond(&changed_accuracy, &content_accuracy)?
        .unsqueeze(1)?
        .detach())
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
    if !model.config().world_core_v4 || model.config().patch_size != PATCH_SIZE {
        bail!("foundation-v2 loss requires the patch-4 exact-decoder topology");
    }
    let samples = mixed.transitions().cloned().collect::<Vec<_>>();
    let batch = batch_from_samples(&samples, device)?;
    let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
    let current_canonical = model.canonical_representation(&encoded.current)?;
    let out = model.full_v4_training_latents_from_encoded_state(
        &encoded.current,
        &current_canonical,
        &batch.actions,
        &batch.action_coords,
        RecursionDepth::from_config(model.config()),
        0.0,
        None,
        RecursionOpts::training(true),
    )?;
    let predicted_canonical = model.canonical_representation(&out.y)?;
    let (_, _, latent_height, latent_width) = out.y.dims4()?;
    let content_masks = mixed.content_masks().collect::<Vec<_>>();
    let latent_mask = latent_content_mask(&content_masks, latent_height, latent_width, device)?;
    let content_current_canonical =
        model.canonical_representation(&encoded.current.broadcast_mul(&latent_mask)?)?;
    let content_target_canonical =
        model.canonical_representation(&encoded.next.broadcast_mul(&latent_mask)?)?;
    let content_predicted_canonical =
        model.canonical_representation(&out.y.broadcast_mul(&latent_mask)?)?;
    let batch_size = samples.len();
    let gameplay_pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
    let current_pixels = samples
        .iter()
        .flat_map(|sample| sample.current.pixels[..gameplay_pixels].iter().copied())
        .collect::<Vec<_>>();
    let target_pixels = samples
        .iter()
        .flat_map(|sample| sample.next.pixels[..gameplay_pixels].iter().copied())
        .collect::<Vec<_>>();
    let content_values = mixed
        .content_masks()
        .flat_map(|mask| mask.values[..gameplay_pixels].iter().copied())
        .collect::<Vec<_>>();
    let changed_weights =
        foundation_v2_loss_weights_from_masks(&current_pixels, &target_pixels, &content_values)?;
    let changed_values = current_pixels
        .iter()
        .zip(&target_pixels)
        .zip(&content_values)
        .map(|((before, after), content)| f32::from(*content != 0 && before != after))
        .collect::<Vec<_>>();
    let unchanged_values = changed_values
        .iter()
        .zip(&content_values)
        .map(|(changed, content)| f32::from(*content != 0) - changed)
        .collect::<Vec<_>>();
    let content = Tensor::from_vec(
        content_values
            .iter()
            .map(|&value| f32::from(value))
            .collect(),
        (batch_size, FRAME_SIDE - 1, FRAME_SIDE),
        device,
    )?;
    let changed = Tensor::from_vec(
        changed_values,
        (batch_size, FRAME_SIDE - 1, FRAME_SIDE),
        device,
    )?;
    let unchanged = Tensor::from_vec(
        unchanged_values,
        (batch_size, FRAME_SIDE - 1, FRAME_SIDE),
        device,
    )?;
    let current_labels = batch
        .frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let target_labels = batch
        .next_frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;

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
    let gate_weights = changed.affine(changed_weights.changed_weight - 1.0, 1.0)?;
    let gate = bce_with_logits_elem(&gate_logits, &changed)?
        .mul(&gate_weights)?
        .mul(&content)?
        .sum_all()?
        .affine(1.0 / changed_weights.content_pixels as f64, 0.0)?;

    let latent = masked_spatial_huber(&out.y, &encoded.next, &latent_mask)?.add(
        &candle_nn::loss::huber(&content_predicted_canonical, &content_target_canonical, 1.0)?,
    )?;

    let foreground_values = current_pixels
        .iter()
        .zip(&content_values)
        .map(|(pixel, content)| f32::from(*content != 0 && *pixel != 0))
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
    let foreground = Tensor::from_vec(
        foreground_values,
        (batch_size, FRAME_SIDE - 1, FRAME_SIDE),
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
        foreground_count,
        background_count,
        foreground_weight,
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
    let displacement_norm = displacement
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
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
                .map(|&row| u32::from(samples[row as usize].action.id))
                .collect::<Vec<_>>(),
            (inverse_rows.len(),),
            device,
        )?;
        let action_ce = candle_nn::loss::cross_entropy(&action_logits, &action_targets)?;
        let action6 = inverse_rows
            .iter()
            .enumerate()
            .filter_map(|(selected_row, &mixed_row)| {
                (samples[mixed_row as usize].action.id == 6).then_some((selected_row, mixed_row))
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
                        let action = &samples[*mixed_row as usize].action;
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
        Some(&event_slot_weight_tensor(device)?),
    )?;
    let graded_targets =
        foundation_v2_graded_q_targets(model, &out.y, &batch.frames, &batch.next_frames, &content)?;
    let q = bce_with_logits(
        &model.q_logit_from_canonical(&detached_canonical)?,
        &graded_targets,
    )?;
    let reliability = bce_with_logits(
        &model.reliability_logit_from_canonical(&detached_canonical)?,
        &graded_targets,
    )?;
    let (rollout, rollout_fragments) = if objective.rollout_enabled {
        foundation_v2_rollout_loss(model, mixed, device)?
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
                self.model
                    .encode_state_pair_for_training(&self.batch.frames, &self.batch.next_frames)
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
    if let Some(export) = &report.export_checkpoint {
        // Fail-closed re-verification: the exported evaluation model must be
        // the checkpoint this run's own gate history selected, never a
        // best-directory leftover from another trajectory or resume branch.
        if cfg.recipe == TrainingRecipe::FoundationV2 {
            let foundation = report
                .foundation_v2
                .as_ref()
                .context("foundation-v2 export requires a foundation report")?;
            let verified = foundation_v2_verified_best_export(
                cfg,
                foundation.promotion_metric,
                &foundation.gate_history,
            )?;
            if verified.as_deref() != Some(export.as_path()) {
                bail!(
                    "export checkpoint {} failed best-selection verification",
                    export.display()
                );
            }
        }
        fs::copy(export, &weights_tmp).with_context(|| {
            format!(
                "copy export checkpoint {} -> {}",
                export.display(),
                weights_tmp.display()
            )
        })?;
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
    if !ema.is_file() {
        return Ok(None);
    }
    let Some(expected_step) = foundation_v2_selected_best_step(metric, gate_history) else {
        bail!(
            "{} exists but this run's gate history has promoted no checkpoint; \
             refusing to export an unattributed best",
            best_dir.display()
        );
    };
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
    let bundle = if path.join("trainer_state.json").is_file() {
        path.to_path_buf()
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
        parent.join(latest.directory)
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
    for required in [
        "trainer_state.json",
        "model.safetensors",
        "optimizer.safetensors",
        "config.json",
        "bundle-manifest.json",
    ] {
        if !bundle.join(required).is_file() {
            bail!(
                "checkpoint bundle is incomplete (missing {required}): {}",
                bundle.display()
            );
        }
    }
    verify_checkpoint_bundle(&bundle)?;
    Ok(bundle)
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
    for expected in &manifest.artifacts {
        if expected.path.contains('/') || expected.path.contains('\\') || expected.path == "." {
            bail!("unsafe checkpoint artifact path {}", expected.path);
        }
        let actual = sha256_file(&bundle.join(&expected.path))?;
        if actual.bytes != expected.bytes || actual.sha256 != expected.sha256 {
            bail!(
                "checkpoint artifact integrity mismatch for {}",
                bundle.join(&expected.path).display()
            );
        }
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
        let complete = final_dir.join("model.safetensors").is_file()
            && final_dir.join("optimizer.safetensors").is_file()
            && final_dir.join("trainer_state.json").is_file()
            && (state.foundation_v2.is_none() || final_dir.join("ema.safetensors").is_file());
        if complete {
            bail!(
                "refusing to overwrite existing checkpoint {}",
                final_dir.display()
            );
        }
        fs::remove_dir_all(&final_dir)
            .with_context(|| format!("remove incomplete checkpoint {}", final_dir.display()))?;
    }
    let staging = checkpoints.join(format!(".{directory}.tmp-{}", std::process::id()));
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
    write_json_atomic(
        &checkpoints.join("latest.json"),
        &LatestCheckpoint {
            schema: TRAINER_STATE_SCHEMA.into(),
            directory,
            global_step: state.global_step,
        },
    )?;
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
    // Older V5 bundles predate the derived experiment field. Their legacy
    // contract already carries every input used to resolve it, and those fields
    // are compared below; hydrate only the absent derived value for exact resume.
    if state.contract.experiment.is_none() {
        state.contract.experiment = requested.experiment.clone();
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
    if state.global_step != state.optimizer_step as u64 {
        bail!(
            "checkpoint cursor mismatch: global_step={} optimizer_step={}",
            state.global_step,
            state.optimizer_step
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
        fs::copy(entry.path(), destination.join(entry.file_name())).with_context(|| {
            format!(
                "copy checkpoint artifact {} -> {}",
                entry.path().display(),
                destination.display()
            )
        })?;
    }
    verify_checkpoint_bundle(destination)
}

fn publish_permanent_checkpoint(cfg: &TrainConfig, checkpoint: &Path) -> Result<PathBuf> {
    let destination = cfg.output_dir.join("checkpoints/permanent").join(
        checkpoint
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("checkpoint path has no directory name"))?,
    );
    if destination.exists() {
        verify_checkpoint_bundle(&destination)?;
        return Ok(destination);
    }
    copy_checkpoint_bundle(checkpoint, &destination)?;
    Ok(destination)
}

fn publish_best_checkpoint(cfg: &TrainConfig, checkpoint: &Path) -> Result<PathBuf> {
    let checkpoints = cfg.output_dir.join("checkpoints");
    let best = checkpoints.join("best");
    let staging = checkpoints.join(format!(".best.tmp-{}", std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .with_context(|| format!("remove incomplete {}", staging.display()))?;
    }
    copy_checkpoint_bundle(checkpoint, &staging)?;
    if best.exists() {
        let old = checkpoints.join(format!(".best.old-{}", std::process::id()));
        if old.exists() {
            fs::remove_dir_all(&old).with_context(|| format!("remove stale {}", old.display()))?;
        }
        fs::rename(&best, &old)
            .with_context(|| format!("rotate {} -> {}", best.display(), old.display()))?;
        fs::rename(&staging, &best)
            .with_context(|| format!("publish {} -> {}", staging.display(), best.display()))?;
        fs::remove_dir_all(&old).with_context(|| format!("remove replaced {}", old.display()))?;
    } else {
        fs::rename(&staging, &best)
            .with_context(|| format!("publish {} -> {}", staging.display(), best.display()))?;
    }
    verify_checkpoint_bundle(&best)?;
    Ok(best)
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
    }
}

fn build_report(
    cfg: &TrainConfig,
    state: &TrainerState,
    status: TrainStatus,
    parameter_count: usize,
    latest_checkpoint: PathBuf,
    resumed_from: Option<PathBuf>,
) -> TrainReport {
    let foundation_v2 = state.foundation_v2.as_ref().map(|foundation| {
        let best_promotion_value = match cfg.promotion_metric {
            PromotionMetric::ChangedExact => foundation.best_changed_exact,
            metric => foundation_v2_best_evaluation(metric, &foundation.gate_history)
                .and_then(|evaluation| foundation_v2_promotion_value(metric, &evaluation.metrics)),
        };
        let best_checkpoint = cfg.output_dir.join("checkpoints/best");
        FoundationV2TrainingReport {
            total_steps: foundation.total_steps,
            mean_losses: foundation_v2_loss_means(&foundation.loss_sums, foundation.loss_steps),
            ep_weight: foundation.ep_weight,
            ep_gradient_budget: foundation.ep_gradient_budget.clone(),
            gate_history: foundation.gate_history.clone(),
            best_changed_exact: foundation.best_changed_exact,
            promotion_metric: cfg.promotion_metric,
            best_promotion_value,
            best_checkpoint: best_checkpoint.exists().then_some(best_checkpoint),
            rollout_enabled: foundation.rollout_enabled,
            permanent_checkpoints: foundation.permanent_checkpoints.clone(),
            event_label_census: foundation.event_label_census,
            event_label_census_complete: foundation.event_label_census_complete,
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
    }
}

fn publish_run_artifacts(varmap: &VarMap, cfg: &TrainConfig, report: &TrainReport) -> Result<()> {
    save_checkpoint(varmap, cfg, report)?;
    crate::p2::evidence::publish_training_evidence(&cfg.output_dir, report)?;
    Ok(())
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
    })
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
    let _train_pid = TrainPidGuard::install(&cfg.output_dir)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&cfg.output_dir)?)
    } else {
        None
    };
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
        // FiLM identity initialization is restored exactly once after fresh
        // deterministic init. Checkpoint loads must retain learned projections.
        zero_action_film_projections(&varmap)?;
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
                gate_population_identity: None,
            }),
        }
    };
    let mut latest_checkpoint = if resumed_from.is_none() {
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
    let gate_batch = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size: FOUNDATION_V2_GATE_ROWS,
            seed: FOUNDATION_V2_GATE_SEED,
            schedule: foundation_v2_stream_schedule,
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
    {
        let identity =
            foundation_v2_gate_population_identity(&gate_samples, &gate_content_masks)?;
        let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
        match &foundation.gate_population_identity {
            Some(stored) if *stored != identity => bail!(
                "resumed gate population/policy identity mismatch: stored {:?} vs \
                 regenerated {:?}; best/collapse comparisons would span incomparable \
                 populations",
                stored,
                identity
            ),
            Some(_) => {}
            None => foundation.gate_population_identity = Some(identity),
        }
    }
    let stream_config = MixedStreamConfig {
        batch_size: cfg.physical_batch,
        seed: cfg.seed,
        schedule: foundation_v2_stream_schedule,
        ..MixedStreamConfig::default()
    };
    let mut updates_this_run = 0usize;
    let total_steps = state
        .foundation_v2
        .as_ref()
        .expect("foundation-v2 state")
        .total_steps;
    let mut data_prefetcher = if cfg.prefetch_batches {
        Some(MixedStreamBatchPrefetcher::new(
            stream_config.clone(),
            total_steps,
            state.global_step,
            cfg.data_workers,
        )?)
    } else {
        None
    };

    loop {
        let total_steps = state
            .foundation_v2
            .as_ref()
            .expect("foundation-v2 state")
            .total_steps;
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
            );
            publish_run_artifacts(&varmap, &cfg, &report)?;
            sync_cuda_device(&device)?;
            return Ok(report);
        }

        let mixed = if let Some(prefetcher) = data_prefetcher.as_mut() {
            let (batch_index, batch) = prefetcher.recv_next()?;
            if batch_index != state.global_step {
                bail!(
                    "foundation-v2 prefetch returned batch {batch_index} while step {} was required",
                    state.global_step
                );
            }
            batch
        } else {
            let progress = state.global_step as f32 / total_steps.max(1) as f32;
            compose_mixed_stream_batch(
                &stream_config,
                progress,
                state.global_step,
                V5DataSplit::Train,
            )?
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
        let consumed = mixed.transitions().cloned().collect::<Vec<_>>();
        update_training_population(&mut state, &consumed);
        let (ep_weight, rollout_enabled) = {
            let foundation = state.foundation_v2.as_ref().expect("foundation-v2 state");
            (foundation.ep_weight, foundation.rollout_enabled)
        };
        let losses = foundation_v2_training_loss(
            &model,
            &mixed,
            &device,
            FoundationV2ObjectiveConfig {
                ep_weight,
                sigreg_projections: cfg.sigreg_projections,
                sigreg_knots: cfg.sigreg_knots,
                sigreg_seed: cfg.seed.wrapping_add(state.global_step),
                rollout_enabled,
                split_ce_weighting: cfg.split_ce_weighting,
                split_ce_changed_budget: cfg.split_ce_changed_budget,
            },
        )?;
        let next_step = state.global_step + 1;
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
        let mut grads = total.backward()?;
        // ADR 0003's documented approximation: the adaptive controller bounds
        // EP's encoder contribution first, then one combined gradient store is
        // clipped at 1.0. There is no separately clipped EP accumulation.
        let clip = clip_gradients_gpu_with_stats(&mut grads, &varmap, MAX_GRAD_NORM)?;
        optimizer.set_learning_rate(foundation_v2_wsd_learning_rate(
            next_step as usize,
            total_steps,
        ))?;
        optimizer.step(&grads)?;
        ema.update(&varmap)?;
        let values = foundation_v2_loss_values(&losses, &total, clip.pre_clip_norm, clip.scale)?;
        {
            let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
            add_foundation_v2_loss_sums(&mut foundation.loss_sums, &values);
            foundation.loss_steps += 1;
        }
        state.global_step = next_step;
        state.optimizer_step = optimizer.step_t();
        updates_this_run += 1;

        let mut improved_best = false;
        let mut abort = false;
        if state.global_step.is_multiple_of(FOUNDATION_V2_GATE_EVERY) {
            let metrics = ema.with_eval_weights(&varmap, || {
                evaluate_gate_support_with_content_masks(
                    &model,
                    &gate_samples,
                    Some(&gate_content_masks),
                    &device,
                )
            })?;
            let foundation = state.foundation_v2.as_mut().expect("foundation-v2 state");
            let prior_best = foundation.best_changed_exact;
            let evaluation = foundation_v2_gate_evaluation(state.global_step, metrics, prior_best);
            improved_best = foundation_v2_promotion_improved(
                cfg.promotion_metric,
                prior_best,
                &foundation.gate_history,
                &evaluation.metrics,
            );
            foundation.best_changed_exact = evaluation.running_best_after;
            foundation.rollout_enabled = evaluation.gates[3].passed;
            foundation.gate_history.push(evaluation);
            abort = foundation_v2_gate_history_aborts(&foundation.gate_history);
        }

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
        if periodic || permanent || improved_best || abort || complete || requested_pause {
            sync_cuda_device(&device)?;
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
            let report = build_report(
                &cfg,
                &state,
                TrainStatus::Aborted,
                parameter_count,
                checkpoint.clone(),
                resumed_from.clone(),
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
    let _train_pid = TrainPidGuard::install(&cfg.output_dir)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&cfg.output_dir)?)
    } else {
        None
    };
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
            profile: ProfileState::Pending,
            gradient_pressure: None,
            gradient_pressure_samples: Vec::new(),
            foundation_v2: None,
        }
    };
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
        bail!(
            "resume state has passed profile update {} without a published evidence bundle",
            cfg.profile_update
        );
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
            precision: if cfg.bf16_conv {
                "bf16-conv/f32-rest"
            } else {
                "f32"
            },
        })?;
        if cg_profile.active() {
            sync_cuda_device(&device)?;
        }
        let profile_measurement = cg_profile.measurement();
        let prof = profile.enabled();
        let accum = cfg.grad_accum.max(1);
        let sigreg_seed = cfg.seed.wrapping_add(state.global_step);
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
            update_training_population(&mut state, &samples);
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
            let micro_sigreg_seed = sigreg_seed.wrapping_add(micro as u64);
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
                || cg_profile.record_gradients(&varmap, &grads),
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
    use crate::p2::data::{ArcAction, GoalFeatures};

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
            let tol = 1e-5 * x.abs().max(y.abs()).max(1.0);
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
        assert!(pix.iter().all(|&v| v == 3));

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
}
