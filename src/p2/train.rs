//! P2 LeWorld / TRM training on synthetic curriculum only.

use crate::domain::Split;
use crate::gpu_lock::{GpuSessionGuard, TrainPidGuard};
use crate::p2::branch_learning::{branch_learning_loss, BranchLearningAudit, BranchLearningConfig};
use crate::p2::cg_profile::{CaptureSpec, ProfileState, RepresentativeUpdateCapture};
use crate::p2::consumer_transition::ConsumerTransition;
use crate::p2::data::{
    generate_curriculum, ArcFrame, FactualBatch, TransitionSample, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::experiment::{
    ExperimentRequest, ResolvedExperiment, SigregPopulation, SigregStatistic,
};
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, ModelConfig, PtrmConfig, RecursionDepth, RecursionOpts,
    WorldModel, ACTION_VOCAB, DEFAULT_NUM_EVENTS, PALETTE_SIZE, PREFIX_HORIZONS,
};
use crate::p2::muon::MUON_RMS_SCALE;
use crate::p2::optimizer::{
    accumulate_parameter_gradients, clip_gradients_gpu, CheckpointHybridOptimizer,
};
use crate::p2::prefetch::{BatchPrefetcher, PrefetchRequest, PrefetchScope};
use crate::p2::sigreg::{sigreg_epps_pulley_seeded, sigreg_quantile_seeded};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_graph::{ExecutionStep, SpanKind};
use candle_nn::init::FanInOut;
use candle_nn::optim::ParamsAdamW;
use candle_nn::{VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
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
/// Per-event-slot multipliers: noop, satisfied, failed, exhausted.
const EVENT_SLOT_WEIGHTS: [f32; 4] = [1.0, 1.0, 4.0, 2.0];
pub const TRAIN_REPORT_SCHEMA: &str = "p2.train_report.v8";
pub const TRAINER_STATE_SCHEMA: &str = "p2.trainer_state.v5";

pub type SigregTarget = SigregPopulation;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
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
    /// Run PTRM ranking loss every N optimizer steps on sequential/retarget (`1` = every step).
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
    /// Intentionally checkpoint-incompatible action-faithful world core.
    #[serde(default)]
    pub world_core_v2: bool,
    /// V3 experiment schema: V2 topology plus residual spatial conditioning
    /// and scale-normalized factual displacement health.
    #[serde(default)]
    pub world_core_v3: bool,
    /// Localized ACTION6 conditioning, independently switchable inside V2.
    #[serde(default)]
    pub spatial_action_field: bool,
    #[serde(default)]
    pub spatial_action_residual: bool,
    #[serde(default = "default_spatial_action_residual_scale")]
    pub spatial_action_residual_scale: f64,
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
            world_core_v2: false,
            world_core_v3: false,
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: default_spatial_action_residual_scale(),
            branch_learning: BranchLearningConfig::default(),
        }
    }
}

impl TrainConfig {
    pub fn resolved_experiment(&self) -> Result<ResolvedExperiment> {
        ResolvedExperiment::resolve(ExperimentRequest {
            world_core_v2: self.world_core_v2,
            world_core_v3: self.world_core_v3,
            spatial_action_field: self.spatial_action_field,
            spatial_action_residual: self.spatial_action_residual,
            spatial_action_residual_scale: self.spatial_action_residual_scale,
            branch_learning_enabled: self.branch_learning.enabled,
            displacement_health_enabled: self.branch_learning.displacement_health.is_some(),
            sigreg_weight: self.sigreg_weight,
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
        if self.physical_batch < 2 {
            bail!("physical_batch must be >= 2 (SIGReg needs batch >= 2)");
        }
        if self.grad_accum == 0 {
            bail!("grad_accum must be >= 1");
        }
        if self.lessons.is_empty() {
            bail!("at least one lesson is required");
        }
        if self.max_steps_this_run == Some(0) {
            bail!("max_steps_this_run must be > 0 when provided");
        }
        if self.profile_update == 0 {
            bail!("profile_update is one-based and must be > 0");
        }
        for lesson in &self.lessons {
            lesson_to_curriculum(lesson)?;
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            bail!("lr must be finite and > 0");
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            bail!("weight_decay must be finite and >= 0");
        }
        for (name, weight) in [
            ("sigreg_weight", self.sigreg_weight),
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
        if self.ptrm_rank_every == 0 {
            bail!("ptrm_rank_every must be >= 1 (use 1 for every step)");
        }
        if !self.train_z_noise.is_finite() || self.train_z_noise < 0.0 {
            bail!("train_z_noise must be finite and >= 0");
        }
        self.branch_learning.validate(self.grad_accum)?;
        let resolved = self.resolved_experiment()?;
        if resolved.factual_learning
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
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LessonLossMeans {
    pub total: f64,
    pub next_latent: f64,
    pub rollout: f64,
    pub sigreg_raw: f64,
    pub sigreg_bounded: f64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStatus {
    Completed,
    Paused,
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
    pub model_next_latent_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displacement_health_weighted_l2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displacement_health_to_next_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrainingContract {
    seed: u64,
    lessons: Vec<String>,
    steps_per_lesson: usize,
    #[serde(default)]
    lesson_steps: Vec<usize>,
    physical_batch: usize,
    grad_accum: usize,
    profile_update: u64,
    lr: f64,
    weight_decay: f64,
    sigreg_projections: usize,
    sigreg_knots: usize,
    sigreg_weight: f64,
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
    spatial_action_residual: bool,
    #[serde(default = "default_spatial_action_residual_scale")]
    spatial_action_residual_scale: f64,
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
            seed: cfg.seed,
            lessons: cfg.lessons.clone(),
            steps_per_lesson: cfg.steps_per_lesson,
            lesson_steps: resolved_lesson_steps(cfg),
            physical_batch: cfg.physical_batch,
            grad_accum: cfg.grad_accum,
            profile_update: cfg.profile_update,
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            sigreg_projections: cfg.sigreg_projections,
            sigreg_knots: cfg.sigreg_knots,
            sigreg_weight: cfg.sigreg_weight,
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
            spatial_action_field: cfg.spatial_action_field,
            spatial_action_residual: cfg.spatial_action_residual,
            spatial_action_residual_scale: cfg.spatial_action_residual_scale,
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
    #[serde(default)]
    batch_schedule_migrations: Vec<BatchScheduleMigration>,
    profile: ProfileState,
    #[serde(default)]
    gradient_pressure: Option<GradientPressureDiagnostics>,
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
            if !(1..ACTION_VOCAB as u32).contains(&id) {
                bail!("action id {id} out of official range 1..{ACTION_VOCAB}");
            }
            match (id, sample.action.x, sample.action.y) {
                (6, Some(_), Some(_)) | (1..=5, None, None) | (7, None, None) => Ok(id),
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
    let factual = samples
        .iter()
        .all(|sample| sample.family.starts_with("factual_"))
        .then(|| FactualBatch::from_rows(samples))
        .transpose()?;
    let rows = factual.as_ref().map_or(samples, FactualBatch::rows);
    let (frames, next_frames) = rayon::join(
        || sample_frames_to_indices(rows, false, device),
        || sample_frames_to_indices(rows, true, device),
    );
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
    let (frames, next_frames) = rayon::join(
        || sample_frames_to_indices(samples, false, device),
        || sample_frames_to_indices(samples, true, device),
    );
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
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight * aux_warm,
            q: cfg.q_weight * aux_warm,
            rollout: 0.0,
            prefix: 0.0,
            reliability: cfg.reliability_weight * aux_warm,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        "events" => LessonLossWeights {
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
            sigreg: cfg.sigreg_weight,
            event: cfg.event_weight * aux_warm,
            q: cfg.q_weight * aux_warm,
            rollout: 0.0,
            prefix: cfg.prefix_weight * aux_warm,
            reliability: cfg.reliability_weight * aux_warm,
            ptrm_rank: rank,
            ptrm_rank_k: rank_k,
        },
        "retarget" => LessonLossWeights {
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

#[derive(Debug, Clone)]
pub struct LossBreakdown {
    pub total: Tensor,
    pub next_latent: Tensor,
    pub sigreg_raw: Tensor,
    pub sigreg_bounded: Tensor,
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
) -> [Tensor; 24] {
    [
        losses.next_latent.detach(),
        rollout.detach(),
        losses.sigreg_raw.detach(),
        losses.sigreg_bounded.detach(),
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

fn checked_training_losses(tensors: &[[Tensor; 24]]) -> Result<Vec<CheckedTrainingLosses>> {
    const NAMES: [&str; 24] = [
        "next_latent",
        "rollout",
        "sigreg_raw",
        "sigreg_bounded",
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
        .chunks_exact(24)
        .map(|values| CheckedTrainingLosses {
            total: values[23],
            next_latent: values[0],
            rollout: values[1],
            sigreg_raw: values[2],
            sigreg_bounded: values[3],
            event: values[4],
            q: values[5],
            prefix: values[8],
            reliability: values[10],
            branch_total: values[11],
            outcome_pull: values[12],
            outcome_push: values[13],
            action_recovery: values[14],
            coordinate_recovery: values[15],
            changed_margin: values[16],
            spatial_variance: values[17],
            spatial_covariance: values[18],
            pooled_variance: values[19],
            pooled_covariance: values[20],
            displacement_variance: values[21],
            displacement_covariance: values[22],
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

fn sigreg_loss_for_stack(stack: &Tensor, cfg: &TrainConfig, seed: u64) -> Result<Tensor> {
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

/// LeWorld loss: mean next-latent MSE over outer steps + SIGReg + masked aux heads.
pub fn leworld_loss(
    model: &WorldModel,
    batch: &BatchTensors,
    cfg: &TrainConfig,
    depth: RecursionDepth,
    sigreg_seed: u64,
    weights: LessonLossWeights,
) -> Result<LossBreakdown> {
    leworld_loss_with_sigreg_windows(model, batch, None, None, cfg, depth, sigreg_seed, weights)
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
) -> Result<LossBreakdown> {
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

    let next_latent = if cfg.supervise_last_outer_only {
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

    let device = batch.frames.device();
    let zero = Tensor::zeros((), DType::F32, device)?;
    let (sigreg_raw, sigreg_bounded) = if weights.sigreg == 0.0 {
        (zero.clone(), zero.clone())
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
    };

    let (event_raw, event) = if weights.event > 0.0 {
        let slot_weights = event_slot_weight_tensor(device)?;
        let event_y = if cfg.stop_grad_event_y {
            out.y.detach()
        } else {
            out.y.clone()
        };
        let event_logits = model.event_logits_from(&event_y, &batch.goals)?;
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

    let (q_raw, q, q_logit, q_mse_per_sample) = if weights.q > 0.0 {
        let q_y = if cfg.stop_grad_q_y {
            out.y.detach()
        } else {
            out.y.clone()
        };
        let q_logit = model.q_logit_from_y(&q_y)?;
        let per = latent_mse_per_sample(&q_y, &next_z)?;
        let q_targets = q_targets_from_mse(&per.detach(), cfg)?;
        let raw = bce_with_logits(&q_logit, &q_targets)?;
        (raw.clone(), raw, Some(q_logit), Some(per))
    } else {
        (zero.clone(), zero.clone(), None, None)
    };

    let (rel_raw, reliability) = if weights.reliability > 0.0 {
        let per = latent_mse_per_sample(&out.y.detach(), &next_z)?.detach();
        let q_targets = q_targets_from_mse(&per, cfg)?;
        let reliability_logit = model.reliability_logit_from_y(&out.y.detach())?;
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

    let mut total = next_latent.clone();
    for (weight, loss) in [
        (weights.sigreg, &sigreg_bounded),
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
    let bundle_weights = report.latest_checkpoint.join("model.safetensors");
    if let Some(export) = &report.export_checkpoint {
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

fn export_checkpoint_path(output_dir: &Path) -> Option<PathBuf> {
    let best = output_dir.join("model.best.safetensors");
    best.exists().then_some(best)
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
    ] {
        if !bundle.join(required).is_file() {
            bail!(
                "checkpoint bundle is incomplete (missing {required}): {}",
                bundle.display()
            );
        }
    }
    Ok(bundle)
}

fn save_training_checkpoint(
    varmap: &VarMap,
    optimizer: &CheckpointHybridOptimizer,
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
            && final_dir.join("trainer_state.json").is_file();
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
    write_json_atomic(&staging.join("trainer_state.json"), state)?;
    write_json_atomic(&output_dir.join("config.json"), &persist_train_config(cfg))?;
    File::open(&model_path)?.sync_all()?;
    File::open(&optimizer_path)?.sync_all()?;
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
    let model_path = bundle.join("model.safetensors");
    let optimizer_path = bundle.join("optimizer.safetensors");
    load_varmap_exact(varmap, &model_path)?;
    optimizer.load(&optimizer_path, state.optimizer_step)?;
    Ok(state)
}

fn loss_means(sums: &LessonLossMeans, count: usize) -> LessonLossMeans {
    let n = count as f64;
    LessonLossMeans {
        total: sums.total / n,
        next_latent: sums.next_latent / n,
        rollout: sums.rollout / n,
        sigreg_raw: sums.sigreg_raw / n,
        sigreg_bounded: sums.sigreg_bounded / n,
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

fn build_report(
    cfg: &TrainConfig,
    state: &TrainerState,
    status: TrainStatus,
    parameter_count: usize,
    latest_checkpoint: PathBuf,
    resumed_from: Option<PathBuf>,
) -> TrainReport {
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
        device: cfg.device.clone(),
        lessons: state.completed_lessons.clone(),
        status,
        global_step: state.global_step,
        latest_checkpoint,
        resumed_from,
        batch_schedule_migrations: state.batch_schedule_migrations.clone(),
        checkpoint: cfg.output_dir.join("model.safetensors"),
        export_checkpoint: export_checkpoint_path(&cfg.output_dir),
        config_path: cfg.output_dir.join("config.json"),
        profile: state.profile.clone(),
        gradient_pressure: state.gradient_pressure.clone(),
        research_claim: false,
    }
}

fn publish_run_artifacts(varmap: &VarMap, cfg: &TrainConfig, report: &TrainReport) -> Result<()> {
    save_checkpoint(varmap, cfg, report)?;
    let _ = report;
    Ok(())
}

/// Train lessons in order. SIGINT/SIGTERM pauses after the current optimizer update.
pub fn train(cfg: &TrainConfig) -> Result<TrainReport> {
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
        load_training_checkpoint(bundle, cfg, &mut varmap, &mut optimizer)?
    } else {
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
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
            batch_schedule_migrations: Vec::new(),
            profile: ProfileState::Pending,
            gradient_pressure: None,
        }
    };
    let mut latest_checkpoint = resumed_from.clone();
    let mut latest_checkpoint_step = resumed_from.as_ref().map(|_| state.global_step);
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
                latest_checkpoint =
                    Some(save_training_checkpoint(&varmap, &optimizer, &state, cfg)?);
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
                _ => save_training_checkpoint(&varmap, &optimizer, &state, cfg)?,
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
            let samples = {
                let _cg = cg_profile.phase("generate", SpanKind::Module, None);
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
                })?
            };
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
            let (batch, ordered_trace, sigreg_windows) = {
                let cg = cg_profile.phase("stage", SpanKind::Module, None);
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
                if let Some(range) = cg.as_ref() {
                    cg_profile.record_tensor(range, "batch.frames", &staged.0.frames, None)?;
                }
                staged
            };
            let micro_sigreg_seed = sigreg_seed.wrapping_add(micro as u64);
            let run_rollout = micro == 0
                && loss_weights.rollout > 0.0
                && matches!(lesson.as_str(), "sequential" | "retarget");
            let (micro_losses, micro_rollout, micro_prefix_multi, micro_total) = {
                let cg =
                    cg_profile.phase("forward", SpanKind::Function, Some(ExecutionStep::Forward));
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
                            rollout_teacher_mix(lesson, state.step_in_lesson, active_lesson_steps),
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
                        total = total.add(&prefix_multi.affine(loss_weights.prefix, 0.0)?)?;
                    }
                    Ok((losses, rollout, prefix_multi, total))
                })?;
                if let Some(range) = cg.as_ref() {
                    cg_profile.record_tensor(
                        range,
                        "loss.total",
                        &result.3,
                        Some(ExecutionStep::Forward),
                    )?;
                }
                result
            };
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
            let pressure_update = cfg.profile_update.saturating_sub(1).max(1);
            if micro == 0
                && state.gradient_pressure.is_none()
                && state.global_step.saturating_add(1) == pressure_update
                && (loss_weights.sigreg > 0.0 || cfg.branch_learning.displacement_health.is_some())
            {
                // Read-only attribution: these stores are discarded before the
                // normal total-loss backward and never reach the optimizer.
                if loss_weights.sigreg > 0.0 {
                    let next_grads = micro_losses.next_latent.backward()?;
                    let next_norm =
                        gradient_l2_for_parameter_prefix(&next_grads, &varmap, "encoder.")?;
                    drop(next_grads);
                    let sigreg_grads = micro_losses
                        .sigreg_bounded
                        .affine(loss_weights.sigreg, 0.0)?
                        .backward()?;
                    let sigreg_norm =
                        gradient_l2_for_parameter_prefix(&sigreg_grads, &varmap, "encoder.")?;
                    state.gradient_pressure = Some(GradientPressureDiagnostics {
                        update: pressure_update,
                        encoder_next_latent_l2: next_norm,
                        encoder_sigreg_weighted_l2: sigreg_norm,
                        sigreg_to_next_ratio: (next_norm > 0.0).then_some(sigreg_norm / next_norm),
                        model_next_latent_l2: None,
                        displacement_health_weighted_l2: None,
                        displacement_health_to_next_ratio: None,
                    });
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
                    state.gradient_pressure = Some(GradientPressureDiagnostics {
                        update: pressure_update,
                        encoder_next_latent_l2: 0.0,
                        encoder_sigreg_weighted_l2: 0.0,
                        sigreg_to_next_ratio: None,
                        model_next_latent_l2: Some(next_norm),
                        displacement_health_weighted_l2: Some(health_norm),
                        displacement_health_to_next_ratio: (next_norm > 0.0)
                            .then_some(health_norm / next_norm),
                    });
                }
            }
            let scaled_micro = micro_total.affine(inv, 0.0)?;
            let micro_grads = {
                let _cg = cg_profile.phase(
                    "backward",
                    SpanKind::Function,
                    Some(ExecutionStep::Backward),
                );
                timed(prof, &device, &mut profile.backward, || {
                    let _span = tracing::info_span!("backward").entered();
                    scaled_micro.backward().map_err(Into::into)
                })?
            };
            accumulate_parameter_gradients(&mut accumulated_grads, micro_grads, &varmap)?;
            metric_tensors.push(training_loss_tensors(
                &micro_losses,
                &micro_rollout,
                &micro_prefix_multi,
                &micro_total,
            ));
        }
        for micro_vals in checked_training_losses(&metric_tensors)? {
            step_metrics.total += micro_vals.total as f64 * inv;
            step_metrics.next_latent += micro_vals.next_latent as f64 * inv;
            step_metrics.rollout += micro_vals.rollout as f64 * inv;
            step_metrics.sigreg_raw += micro_vals.sigreg_raw as f64 * inv;
            step_metrics.sigreg_bounded += micro_vals.sigreg_bounded as f64 * inv;
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
        clip_gradients_gpu(&mut grads, &varmap, MAX_GRAD_NORM)?;
        if cg_profile.active() {
            let _cg =
                cg_profile.phase("gradients", SpanKind::Module, Some(ExecutionStep::Backward));
            cg_profile.record_gradients(&varmap, &grads)?;
        }
        {
            let _cg = cg_profile.phase(
                "optimizer",
                SpanKind::Function,
                Some(ExecutionStep::Optimizer),
            );
            timed(prof, &device, &mut profile.optimizer, || {
                let _span = tracing::info_span!("optimizer").entered();
                optimizer.step(&grads)
            })?;
        }
        drop(grads);

        {
            let _cg = cg_profile.phase("metrics", SpanKind::Module, None);
            timed(prof, &device, &mut profile.metrics, || {
                let _span = tracing::info_span!("metrics").entered();
                state.active_sums.total += step_metrics.total;
                state.active_sums.next_latent += step_metrics.next_latent;
                state.active_sums.rollout += step_metrics.rollout;
                state.active_sums.sigreg_raw += step_metrics.sigreg_raw;
                state.active_sums.sigreg_bounded += step_metrics.sigreg_bounded;
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
            })?;
        }
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

        if state.step_in_lesson == active_lesson_steps {
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
                save_training_checkpoint(&varmap, &optimizer, &state, cfg)
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
        clip_gradients_gpu(&mut grads, &varmap, MAX_GRAD_NORM)?;
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
            physical_batch: 4,
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
        let samples = collect_batch("factual_branches", cfg.seed, 0, 4, Split::Train)?;
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
        )?;
        assert_eq!(losses.branch_audit.groups, 1);
        assert_eq!(losses.branch_audit.branches, 4);
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
        assert!(evidence.health.trusted);
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
