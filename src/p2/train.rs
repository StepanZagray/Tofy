//! P2 LeWorld / TRM training on synthetic curriculum only.

use crate::domain::Split;
use crate::gpu_lock::{GpuSessionGuard, TrainPidGuard};
use crate::p2::data::{
    generate_curriculum, ArcFrame, TransitionSample, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::cg_profile::StepProfileCapture;
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, ModelConfig, PtrmConfig, RecursionDepth,
    RecursionOpts, WorldModel,
    ACTION_VOCAB, DEFAULT_NUM_EVENTS, PIXEL_CHANNELS, PREFIX_HORIZONS,
};
use crate::p2::muon::MUON_RMS_SCALE;
use crate::p2::optimizer::{
    accumulate_grad_store, clip_gradients_gpu, CheckpointHybridOptimizer,
};
use crate::p2::prefetch::{BatchPrefetcher, PrefetchRequest};
use crate::p2::sigreg::sigreg_epps_pulley_seeded;
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor};
use candle_nn::init::FanInOut;
use candle_nn::optim::ParamsAdamW;
use candle_graph::SpanKind;
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
/// Global gradient L2 clip for recursive training stability.
const MAX_GRAD_NORM: f64 = 1.0;
/// Per-event-slot multipliers: noop, satisfied, failed, exhausted.
const EVENT_SLOT_WEIGHTS: [f32; 4] = [1.0, 1.0, 4.0, 2.0];
pub const TRAIN_REPORT_SCHEMA: &str = "p2.train_report.v3";
pub const TRAINER_STATE_SCHEMA: &str = "p2.trainer_state.v1";

/// Optimizer steps for a lesson (`dynamics` / `exploration` get 2× base steps).
pub fn steps_for_lesson(cfg: &TrainConfig, lesson: &str) -> usize {
    match lesson {
        "dynamics" | "exploration" => cfg.steps_per_lesson.saturating_mul(2),
        _ => cfg.steps_per_lesson,
    }
}

pub fn resolved_lesson_steps(cfg: &TrainConfig) -> Vec<usize> {
    cfg.lessons
        .iter()
        .map(|lesson| steps_for_lesson(cfg, lesson))
        .collect()
}

pub fn global_step_from_cursor(lesson_steps: &[usize], lesson_index: usize, step_in_lesson: usize) -> u64 {
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
    /// Save a complete resumable checkpoint every N optimizer updates. Zero disables it.
    pub checkpoint_every_steps: usize,
    /// Stop cleanly after this many updates in this invocation (scheduler/testing hook).
    pub max_steps_this_run: Option<usize>,
    /// Run PTRM ranking loss every N optimizer steps on sequential/retarget (`1` = every step).
    /// Not part of the resume contract; safe to change across pause/resume.
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
    #[serde(
        default = "default_stop_grad_event_y",
        alias = "stop_grad_auxiliary_y"
    )]
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
    /// Muon optimizer for 2D weight matrices (DeepSeek-V4 hybrid with AdamW on emb/bias).
    #[serde(default = "default_use_muon")]
    pub use_muon: bool,
    #[serde(default = "default_muon_momentum")]
    pub muon_momentum: f64,
    #[serde(default = "default_muon_rms_scale")]
    pub muon_rms_scale: f64,
    /// Cap SIGReg row count (0 = no cap). Reduces VRAM for spatial SIGReg.
    #[serde(default = "default_sigreg_max_rows")]
    pub sigreg_max_rows: usize,
    /// Overlap CPU batch generation with GPU work.
    #[serde(default = "default_prefetch_batches")]
    pub prefetch_batches: bool,
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

fn default_use_muon() -> bool {
    true
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
    let mut effective = cap.min(spatial_rows);
  if cfg.physical_batch >= 128 {
        // 8GB: batch 128 + full-depth recursion peaks ~7 GiB; leave headroom for checkpoint IO.
        effective = effective.min(1024);
    } else if cfg.physical_batch >= 64 {
        effective = effective.min(2048);
    }
    effective
}

fn default_prefetch_batches() -> bool {
    true
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
            checkpoint_every_steps: 100,
            max_steps_this_run: None,
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
            stop_grad_q_y: false,
            q_quantile_targets: false,
            train_z_noise: 0.0,
            shuffled_episodes: false,
            baseline_d1: false,
            prefix_weight: 0.0,
            reliability_weight: 0.0,
            bf16_conv: false,
            ensemble_members: 8,
            use_muon: true,
            muon_momentum: 0.95,
            muon_rms_scale: MUON_RMS_SCALE,
            sigreg_max_rows: 4096,
            prefetch_batches: true,
        }
    }
}

impl TrainConfig {
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
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LessonLossMeans {
    pub total: f64,
    pub next_latent: f64,
    pub rollout: f64,
    pub sigreg: f64,
    pub event: f64,
    pub q: f64,
    #[serde(default)]
    pub prefix: f64,
    #[serde(default)]
    pub reliability: f64,
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
    pub checkpoint: PathBuf,
    /// Weights exported for eval when a pre-retarget snapshot exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_checkpoint: Option<PathBuf>,
    pub config_path: PathBuf,
    /// candle-graph trace from the first update (`profile.jsonl`, trace/4).
    pub profile_trace: Option<PathBuf>,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
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
    lr: f64,
    weight_decay: f64,
    sigreg_projections: usize,
    sigreg_knots: usize,
    sigreg_weight: f64,
    event_weight: f64,
    q_weight: f64,
    rollout_weight: f64,
    q_mse_threshold: f64,
    hidden_dim: usize,
    action_dim: usize,
    inner_steps: usize,
    outer_steps: usize,
    randomize_depth: bool,
    #[serde(default)]
    steady_gpu: bool,
    #[serde(default)]
    supervise_last_outer_only: bool,
    phased_training: bool,
    #[serde(
        default = "default_stop_grad_event_y",
        alias = "stop_grad_auxiliary_y"
    )]
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
    stop_grad_q_y: bool,
    #[serde(default)]
    q_quantile_targets: bool,
    #[serde(default)]
    train_z_noise: f64,
    #[serde(default)]
    shuffled_episodes: bool,
    device: String,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_eps: f64,
    #[serde(default = "default_use_muon")]
    use_muon: bool,
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
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            sigreg_projections: cfg.sigreg_projections,
            sigreg_knots: cfg.sigreg_knots,
            sigreg_weight: cfg.sigreg_weight,
            event_weight: cfg.event_weight,
            q_weight: cfg.q_weight,
            rollout_weight: cfg.rollout_weight,
            q_mse_threshold: cfg.q_mse_threshold,
            hidden_dim: cfg.hidden_dim,
            action_dim: cfg.action_dim,
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
            randomize_depth: cfg.randomize_depth,
            steady_gpu: cfg.steady_gpu,
            supervise_last_outer_only: cfg.supervise_last_outer_only,
            phased_training: cfg.phased_training,
            stop_grad_event_y: cfg.stop_grad_event_y,
            residual_y_update: cfg.residual_y_update,
            warm_start_y: cfg.warm_start_y,
            sigreg_spatial: cfg.sigreg_spatial,
            sigreg_spatial_pool: cfg.sigreg_spatial_pool,
            stop_grad_q_y: cfg.stop_grad_q_y,
            q_quantile_targets: cfg.q_quantile_targets,
            train_z_noise: cfg.train_z_noise,
            shuffled_episodes: cfg.shuffled_episodes,
            device: cfg.device.clone(),
            adam_beta1: adam.beta1,
            adam_beta2: adam.beta2,
            adam_eps: adam.eps,
            use_muon: cfg.use_muon,
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
    profile_emitted: bool,
    #[serde(default, skip_serializing, alias = "runtime_trace")]
    legacy_runtime_trace: Option<serde_json::Value>,
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
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }
    if spec.eq_ignore_ascii_case("cuda") {
        return Device::new_cuda(0).context("open cuda:0");
    }
    if let Some(rest) = spec
        .strip_prefix("cuda:")
        .or_else(|| spec.strip_prefix("CUDA:"))
    {
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

/// Load matching tensors from a checkpoint; deterministically init any missing keys.
pub fn load_varmap_flexible(varmap: &VarMap, path: &Path, master_seed: u64) -> Result<()> {
    let device = varmap
        .all_vars()
        .first()
        .map(|v| v.device().clone())
        .ok_or_else(|| anyhow::anyhow!("empty varmap"))?;
    let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    let names: Vec<String> = {
        let data = varmap.data().lock().unwrap();
        let mut n: Vec<_> = data.keys().cloned().collect();
        n.sort();
        n
    };
    for name in names {
        let var = {
            let data = varmap.data().lock().unwrap();
            data.get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?
                .clone()
        };
        if let Ok(t) = mmap.load(&name, &device) {
            if t.dims() == var.shape().dims() {
                var.set(&t)?;
                continue;
            }
            tracing::info!(
                "checkpoint shape mismatch for {name}: {:?} vs {:?}; reinitializing",
                t.dims(),
                var.shape().dims()
            );
        }
        let shape = var.shape().dims().to_vec();
        let n = var.elem_count();
        let seed = stable_name_seed(master_seed, &name);
        let is_bias = name.rsplit('.').next() == Some("bias") || name.ends_with("bias");
        let values = if is_bias {
            vec![0f32; n]
        } else {
            xavier_uniform_vec(&shape, seed)
        };
        let t = Tensor::from_vec(values, shape.as_slice(), &device)?.to_dtype(var.dtype())?;
        var.set(&t)?;
        tracing::info!("checkpoint missing {name}; initialized deterministically");
    }
    Ok(())
}

pub fn parameter_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.elem_count()).sum()
}

/// Palette indices `B×1×64×64` on device (compact vs one-hot).
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
            if let Some(&pix) = frame.pixels.iter().find(|&&p| p as usize >= PIXEL_CHANNELS) {
                bail!("palette value {pix} out of 0..{PIXEL_CHANNELS}");
            }
            slot.copy_from_slice(&frame.pixels);
            Ok(())
        })?;
    Tensor::from_vec(indices, (b, 1, FRAME_SIDE, FRAME_SIDE), device).map_err(Into::into)
}

/// Convert palette frames to categorical `B×16×64×64` one-hot tensors (legacy/tests).
///
/// Only the palette indices cross the bus: `B×64×64` `u8` is staged to the device and
/// expanded to one-hot there. Expanding on the host first would push 64x more data
/// through both host memory and PCIe — 268 MB per call at batch 1024, against 4 MB —
/// and the trainer builds two of these every step.
pub fn frames_to_one_hot(frames: &[ArcFrame], device: &Device) -> Result<Tensor> {
    let b = frames.len();
    if b == 0 {
        bail!("frames_to_one_hot requires at least one frame");
    }
    let pixels = FRAME_SIDE * FRAME_SIDE;
    let mut indices = vec![0u8; b * pixels];
    // Frames are already row-major `u8` palette indices in exactly this layout, so
    // each slot is a validated memcpy. Disjoint slices, so no synchronization.
    indices
        .par_chunks_mut(pixels)
        .zip(frames.par_iter())
        .try_for_each(|(slot, frame)| -> Result<()> {
            ensure_fixed_frame(frame)?;
            if let Some(&pix) = frame.pixels.iter().find(|&&p| p as usize >= PIXEL_CHANNELS) {
                bail!("palette value {pix} out of 0..{PIXEL_CHANNELS}");
            }
            slot.copy_from_slice(&frame.pixels);
            Ok(())
        })?;

    let indices = Tensor::from_vec(indices, (b, 1, FRAME_SIDE, FRAME_SIDE), device)?;
    let channels = Tensor::arange(0u8, PIXEL_CHANNELS as u8, device)?
        .reshape((1, PIXEL_CHANNELS, 1, 1))?;
    // Broadcast compare lands directly in NCHW, so there is no transpose/contiguous
    // pass afterwards.
    indices
        .broadcast_eq(&channels)?
        .to_dtype(DType::F32)
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
}

pub fn batch_from_samples(samples: &[TransitionSample], device: &Device) -> Result<BatchTensors> {
    if samples.is_empty() {
        bail!("empty batch");
    }
    let currents: Vec<ArcFrame> = samples.iter().map(|s| s.current.clone()).collect();
    let nexts: Vec<ArcFrame> = samples.iter().map(|s| s.next.clone()).collect();
    let (frames, next_frames) = rayon::join(
        || frames_to_indices(&currents, device),
        || frames_to_indices(&nexts, device),
    );
    let frames = frames?;
    let next_frames = next_frames?;
    let actions: Vec<u32> = samples
        .par_iter()
        .map(|s| {
            let id = s.action.id as u32;
            if id as usize >= ACTION_VOCAB {
                bail!("action id {id} out of ACTION_VOCAB={ACTION_VOCAB}");
            }
            Ok(id)
        })
        .collect::<Result<Vec<_>>>()?;
    let actions = Tensor::from_vec(actions, (samples.len(),), device)?;
    let coords: Vec<f32> = samples
        .par_iter()
        .map(|sample| {
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
    let goals: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.goal_features.values.iter().copied())
        .collect();
    let goals = Tensor::from_vec(goals, (samples.len(), GOAL_FEATURES_DIM), device)?;
    let (event_targets, event_mask) = event_targets_and_mask(samples, device)?;
    Ok(BatchTensors {
        frames,
        next_frames,
        actions,
        action_coords,
        goals,
        event_targets,
        event_mask,
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
                .map(|offset| generate_curriculum(curriculum, seed, next.wrapping_add(offset), split))
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
        if !self.key_matches(curriculum, seed, split)
            || start_episode < self.first_episode
        {
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
pub fn ptrm_rank_this_step(lesson: &str, global_step: u64, every: usize, baseline_d1: bool) -> bool {
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
    RecursionDepth {
        inner_steps: rng.random_range(1..=cfg.inner_steps),
        outer_steps: rng.random_range(1..=max_outer),
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
            ptrm_rank: ptrm_rank_this_step(lesson, global_step, cfg.ptrm_rank_every, cfg.baseline_d1),
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
pub fn rollout_horizon(step_in_lesson: usize, steps_per_lesson: usize, max_horizon: usize) -> usize {
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

fn event_slot_weight_tensor(batch: usize, device: &Device) -> Result<Tensor> {
    let mut weights = Vec::with_capacity(batch * DEFAULT_NUM_EVENTS);
    for _ in 0..batch {
        weights.extend_from_slice(&EVENT_SLOT_WEIGHTS);
    }
    Tensor::from_vec(weights, (batch, DEFAULT_NUM_EVENTS), device).map_err(Into::into)
}

fn masked_bce_with_slot_weights(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
    slot_weights: Option<&Tensor>,
) -> Result<Tensor> {
    let effective_mask = match slot_weights {
        Some(w) => (mask * w)?,
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
    let (x, goal_h, y_init) = model.prepare_transition_from_encoded(
        cur_z,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
    )?;
    let ptrm = model.forward_ptrm_prepared(
        &x,
        &goal_h,
        y_init,
        depth,
        PtrmConfig {
            k,
            sigma,
            seed: Some(seed),
        },
    )?;
    let mut q_rows = Vec::with_capacity(k);
    let mut y_rows = Vec::with_capacity(k);
    for traj in &ptrm.trajectories {
        q_rows.push(traj.q_logit.squeeze(1)?);
        y_rows.push(traj.y.clone());
    }
    let q_logits = Tensor::stack(&q_rows, 1)?;
    let y_stack = Tensor::stack(&y_rows, 1)?;
    let target = next_z.unsqueeze(1)?.broadcast_as(y_stack.dims())?;
    let mse = y_stack
        .sub(&target)?
        .sqr()?
        .flatten_from(2)?
        .mean(2)?;
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
    samples: &[TransitionSample],
    device: &Device,
) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    let mut weight_sum = 0f64;
    for &horizon in &PREFIX_HORIZONS {
        if samples.len() <= horizon {
            continue;
        }
        let start = batch_from_samples(&[samples[0].clone()], device)?;
        let mut z = model.encode_state(&start.frames)?;
        for step in 0..horizon {
            let batch = batch_from_samples(&[samples[step].clone()], device)?;
            z = model.prefix_predict(&z, &batch.actions, &batch.action_coords)?;
        }
        let target = model.encode_state(&batch_from_samples(&[samples[horizon].clone()], device)?.frames)?;
        let w = prefix_horizon_weight(horizon);
        let mse = candle_nn::loss::mse(&z, &target)?;
        let term = mse.affine(w, 0.0)?;
        total = Some(match total {
            None => term,
            Some(acc) => acc.add(&term)?,
        });
        weight_sum += w;
    }
    total
        .ok_or_else(|| anyhow::anyhow!("prefix trace too short"))
        .and_then(|t| t.affine(1.0 / weight_sum.max(1e-8), 0.0).map_err(Into::into))
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

#[derive(Debug, Clone)]
pub struct LossBreakdown {
    pub total: Tensor,
    pub next_latent: Tensor,
    pub sigreg: Tensor,
    pub event: Tensor,
    pub q: Tensor,
    pub prefix: Tensor,
    pub reliability: Tensor,
}

/// Randomly subsample SIGReg rows to cap activation memory.
pub fn subsample_sigreg_rows(stack: &Tensor, max_rows: usize, seed: u64) -> Result<Tensor> {
    let n = stack.dim(0)?;
    if max_rows == 0 || n <= max_rows {
        return Ok(stack.clone());
    }
    use rand::seq::SliceRandom;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.partial_shuffle(&mut rng, max_rows);
    indices.truncate(max_rows);
    let idx = Tensor::from_vec(indices, (max_rows,), stack.device())?;
    stack.index_select(&idx, 0).map_err(Into::into)
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
        let cur = cur
            .permute((0, 2, 3, 1))?
            .reshape((b * h * w, c))?;
        let next = next
            .permute((0, 2, 3, 1))?
            .reshape((b * h * w, c))?;
        Tensor::cat(&[cur, next], 0).map_err(Into::into)
    } else {
        Tensor::stack(
            &[flatten_latent(cur_z)?, flatten_latent(next_z)?],
            0,
        )
        .map_err(Into::into)
    }
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
    let (cur_z, next_z) = model.encode_state_pair(&batch.frames, &batch.next_frames)?;
    let out = model.forward_from_encoded_state(
        &cur_z,
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
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
            let mse = step.y.sub(&next_z)?.sqr()?.mean_all()?;
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

    let stack = subsample_sigreg_rows(
        &stack_latents_for_sigreg(
            &cur_z,
            &next_z,
            cfg.sigreg_spatial,
            cfg.sigreg_spatial_pool,
        )?,
        effective_sigreg_max_rows(cfg),
        sigreg_seed.wrapping_add(0x5196_0001),
    )?;
    let sigreg = sigreg_epps_pulley_seeded(
        &stack,
        cfg.sigreg_projections,
        cfg.sigreg_knots,
        sigreg_seed,
    )?
    .clamp(0.0, 10_000.0)?;

    let device = batch.frames.device();
    let zero_scalar = || -> Result<Tensor> { Tensor::zeros((), DType::F32, device).map_err(Into::into) };

    let (event_raw, event) = if weights.event > 0.0 {
        let slot_weights = event_slot_weight_tensor(batch.frames.dim(0)?, device)?;
        let event_logits = if cfg.stop_grad_event_y {
            out.event_logits.detach()
        } else {
            out.event_logits.clone()
        };
        let raw = masked_bce_with_slot_weights(
            &event_logits,
            &batch.event_targets,
            &batch.event_mask,
            Some(&slot_weights),
        )?;
        (raw.clone(), raw)
    } else {
        (zero_scalar()?, zero_scalar()?)
    };

    let (q_raw, q) = if weights.q > 0.0 {
        let q_logit = if cfg.stop_grad_q_y {
            out.q_logit.detach()
        } else {
            out.q_logit.clone()
        };
        let q_pred = if cfg.stop_grad_q_y {
            out.y.detach()
        } else {
            out.y.clone()
        };
        let per = latent_mse_per_sample(&q_pred, &next_z)?.detach();
        let q_targets = q_targets_from_mse(&per, cfg)?;
        let raw = bce_with_logits(&q_logit, &q_targets)?;
        (raw.clone(), raw)
    } else {
        (zero_scalar()?, zero_scalar()?)
    };

    let (rel_raw, reliability) = if weights.reliability > 0.0 {
        let per = latent_mse_per_sample(&out.y.detach(), &next_z)?.detach();
        let q_targets = q_targets_from_mse(&per, cfg)?;
        let reliability_logit = model.reliability_logit_from_y(&out.y.detach())?;
        let raw = bce_with_logits(&reliability_logit, &q_targets)?;
        (raw.clone(), raw)
    } else {
        (zero_scalar()?, zero_scalar()?)
    };

    let (prefix_raw, prefix) = if weights.prefix > 0.0 {
        let raw = prefix_one_step_loss(model, batch, &cur_z, &next_z)?;
        (raw.clone(), raw)
    } else {
        (zero_scalar()?, zero_scalar()?)
    };

    let mut total = next_latent
        .add(&sigreg.affine(weights.sigreg, 0.0)?)?
        .add(&event.affine(weights.event, 0.0)?)?
        .add(&q.affine(weights.q, 0.0)?)?
        .add(&reliability.affine(weights.reliability, 0.0)?)?
        .add(&prefix.affine(weights.prefix, 0.0)?)?;
    if weights.q > 0.0 && !cfg.stop_grad_q_y {
        let q_logit = out.q_logit.clone();
        let q_prob = candle_nn::ops::sigmoid(&q_logit)?;
        let mse_per = latent_mse_per_sample(&out.y, &next_z)?;
        let surprise = q_prob.mul(&mse_per)?.mean_all()?;
        total = total.add(&surprise.affine(Q_SURPRISE_WEIGHT, 0.0)?)?;
    }
    if weights.ptrm_rank {
        let rank = ptrm_ranking_loss(
            model,
            &cur_z,
            batch,
            &next_z,
            depth,
            weights.ptrm_rank_k,
            0.1,
            sigreg_seed.wrapping_add(1),
        )?;
        total = total.add(&rank.affine(PTRM_RANK_WEIGHT, 0.0)?)?;
    }

    Ok(LossBreakdown {
        total,
        next_latent,
        sigreg,
        event: event_raw,
        q: q_raw,
        prefix: prefix_raw,
        reliability: rel_raw,
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
    prefetcher.submit_many(&batch_prefetch_requests(curriculum, cfg, global_step, accum))
}

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
    let episode_start = scheduled_episode_start(
        cfg.seed,
        global_step,
        micro,
        accum,
        cfg.shuffled_episodes,
    );
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
    samples: &[TransitionSample],
    device: &Device,
    horizon: usize,
    depth: RecursionDepth,
    teacher_mix: f64,
    seed: u64,
) -> Result<Tensor> {
    if samples.len() < 2 || horizon < 2 {
        bail!("open-loop loss requires at least two ordered transitions");
    }
    let steps = horizon.min(samples.len());
    let first = batch_from_samples(&[samples[0].clone()], device)?;
    let mut latent = model.encode_state(&first.frames)?;
    let nexts: Vec<ArcFrame> = samples.iter().take(steps).map(|s| s.next.clone()).collect();
    let targets = model.encode_state(&frames_to_indices(&nexts, device)?)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut total: Option<Tensor> = None;
    let mut n = 0usize;
    for (step, sample) in samples.iter().take(steps).enumerate() {
        let batch = batch_from_samples(std::slice::from_ref(sample), device)?;
        let predicted = model.forward_from_latent_with_depth(
            &latent,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            depth,
        )?;
        let target = targets.narrow(0, step, 1)?;
        let mse = candle_nn::loss::huber(&predicted.y, &target, 1.0)?;
        let capped = mse.clamp(0.0, ROLLOUT_STEP_LOSS_CAP)?;
        total = Some(match total {
            None => capped,
            Some(acc) => acc.add(&capped)?,
        });
        n += 1;
        let teacher = teacher_mix > 0.0 && rng.random::<f64>() < teacher_mix;
        let reset = mse
            .mean_all()?
            .gt(ROLLOUT_ERROR_RESET as f64)?
            .to_dtype(DType::F32)?;
        if teacher {
            latent = target.detach();
        } else {
            let reset_mask = reset
                .to_dtype(predicted.y.dtype())?
                .reshape((1, 1, 1, 1))?
                .broadcast_as(predicted.y.dims())?;
            let keep_pred = (Tensor::ones_like(&reset_mask)? - &reset_mask)?;
            latent = predicted
                .y
                .mul(&keep_pred)?
                .add(&target.detach().mul(&reset_mask)?)?;
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
    fs::create_dir_all(output_dir)
        .with_context(|| format!("create {}", output_dir.display()))?;
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

fn merge_saved_training_contract(cfg: &mut TrainConfig) -> Result<()> {
    let path = cfg.output_dir.join("config.json");
    if !path.is_file() {
        return Ok(());
    }
    let cli_physical_batch = cfg.physical_batch;
    let cli_grad_accum = cfg.grad_accum;
    let cli_effective = effective_batch(cfg);
    let saved: TrainConfig = read_json(&path)?;
    let saved_effective = effective_batch(&saved);
    if cli_effective != saved_effective
        && (cli_physical_batch != saved.physical_batch || cli_grad_accum != saved.grad_accum)
    {
        bail!(
            "resume would change effective batch from {saved_effective} (physical_batch={} grad_accum={}) \
             to {cli_effective} (physical_batch={cli_physical_batch} grad_accum={cli_grad_accum}); \
             effective batch must match when resuming",
            saved.physical_batch,
            saved.grad_accum,
        );
    }
    cfg.seed = saved.seed;
    cfg.lessons = saved.lessons;
    cfg.steps_per_lesson = saved.steps_per_lesson;
    cfg.physical_batch = saved.physical_batch;
    cfg.grad_accum = saved.grad_accum;
    cfg.lr = saved.lr;
    cfg.weight_decay = saved.weight_decay;
    cfg.sigreg_projections = saved.sigreg_projections;
    cfg.sigreg_knots = saved.sigreg_knots;
    cfg.sigreg_weight = saved.sigreg_weight;
    cfg.event_weight = saved.event_weight;
    cfg.q_weight = saved.q_weight;
    cfg.rollout_weight = saved.rollout_weight;
    cfg.q_mse_threshold = saved.q_mse_threshold;
    cfg.hidden_dim = saved.hidden_dim;
    cfg.action_dim = saved.action_dim;
    cfg.inner_steps = saved.inner_steps;
    cfg.outer_steps = saved.outer_steps;
    cfg.randomize_depth = saved.randomize_depth;
    cfg.phased_training = saved.phased_training;
    cfg.stop_grad_event_y = saved.stop_grad_event_y;
    cfg.residual_y_update = saved.residual_y_update;
    cfg.warm_start_y = saved.warm_start_y;
    cfg.sigreg_spatial = saved.sigreg_spatial;
    cfg.stop_grad_q_y = saved.stop_grad_q_y;
    cfg.q_quantile_targets = saved.q_quantile_targets;
    cfg.train_z_noise = saved.train_z_noise;
    cfg.shuffled_episodes = saved.shuffled_episodes;
    cfg.device = saved.device;
    cfg.use_muon = saved.use_muon;
    cfg.muon_momentum = saved.muon_momentum;
    cfg.muon_rms_scale = saved.muon_rms_scale;
    cfg.sigreg_max_rows = saved.sigreg_max_rows;
    cfg.prefetch_batches = saved.prefetch_batches;
    if cli_effective == saved_effective {
        cfg.physical_batch = cli_physical_batch;
        cfg.grad_accum = cli_grad_accum;
    }
    Ok(())
}

fn contract_resume_migration_ok(saved: &TrainingContract, requested: &TrainingContract) -> bool {
    let mut saved = saved.clone();
    saved.adam_beta2 = requested.adam_beta2;
    saved.use_muon = requested.use_muon;
    saved.muon_momentum = requested.muon_momentum;
    saved.muon_rms_scale = requested.muon_rms_scale;
    // Runtime VRAM knobs; safe to toggle when resuming.
    saved.steady_gpu = requested.steady_gpu;
    saved.supervise_last_outer_only = requested.supervise_last_outer_only;
    saved.sigreg_spatial_pool = requested.sigreg_spatial_pool;
    // Microbatch schedule (physical_batch × grad_accum) when effective batch is unchanged.
    if effective_batch_contract(&saved) == effective_batch_contract(requested) {
        saved.physical_batch = requested.physical_batch;
        saved.grad_accum = requested.grad_accum;
    }
    saved == *requested
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
        fs::remove_dir_all(&final_dir).with_context(|| {
            format!("remove incomplete checkpoint {}", final_dir.display())
        })?;
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
    if state.contract != requested && !contract_resume_migration_ok(&state.contract, &requested) {
        bail!(
            "resume training contract mismatch; checkpoint={} requested={}",
            serde_json::to_string(&state.contract)?,
            serde_json::to_string(&requested)?
        );
    }
    if state.contract != requested {
        let saved_batch = (
            state.contract.physical_batch,
            state.contract.grad_accum,
            effective_batch_contract(&state.contract),
        );
        state.contract = requested;
        if (state.contract.physical_batch, state.contract.grad_accum) != (saved_batch.0, saved_batch.1)
            && saved_batch.2 == effective_batch_contract(&state.contract)
        {
            tracing::info!(
                "resume batch schedule migrated: physical_batch {}→{} grad_accum {}→{} (effective_batch={})",
                saved_batch.0,
                state.contract.physical_batch,
                saved_batch.1,
                state.contract.grad_accum,
                saved_batch.2,
            );
        }
    }
    if state.parameter_names != optimizer.parameter_names() {
        let current = optimizer.parameter_names();
        if !state
            .parameter_names
            .iter()
            .all(|n| current.iter().any(|c| c == n))
        {
            bail!("resume parameter names do not match the current model");
        }
        state.parameter_names = current;
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
    let expected_step = global_step_from_cursor(&lesson_steps, state.lesson_index, state.step_in_lesson);
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
    load_varmap_flexible(varmap, &model_path, cfg.seed)?;
    optimizer.load(&optimizer_path, state.optimizer_step)?;
    if state.legacy_runtime_trace.is_some() {
        state.profile_emitted = true;
    }
    state.legacy_runtime_trace = None;
    Ok(state)
}

fn loss_means(sums: &LessonLossMeans, count: usize) -> LessonLossMeans {
    let n = count as f64;
    LessonLossMeans {
        total: sums.total / n,
        next_latent: sums.next_latent / n,
        rollout: sums.rollout / n,
        sigreg: sums.sigreg / n,
        event: sums.event / n,
        q: sums.q / n,
        prefix: sums.prefix / n,
        reliability: sums.reliability / n,
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
        checkpoint: cfg.output_dir.join("model.safetensors"),
        export_checkpoint: export_checkpoint_path(&cfg.output_dir),
        config_path: cfg.output_dir.join("config.json"),
        profile_trace: state
            .profile_emitted
            .then(|| cfg.output_dir.join("profile.jsonl")),
        research_claim: false,
    }
}

fn publish_run_artifacts(
    varmap: &VarMap,
    cfg: &TrainConfig,
    report: &TrainReport,
) -> Result<()> {
    save_checkpoint(varmap, cfg, report)?;
    let _ = report;
    Ok(())
}

/// Train lessons in order. SIGINT/SIGTERM pauses after the current optimizer update.
pub fn train(cfg: &TrainConfig) -> Result<TrainReport> {
    let mut cfg = cfg.clone();
    let explicit_resume = cfg.resume.is_some();
    if explicit_resume || implicit_resume_source(&cfg).is_some() {
        merge_saved_training_contract(&mut cfg)?;
        if !explicit_resume {
            cfg.resume = Some(cfg.output_dir.join("checkpoints"));
            tracing::info!(
                "auto-resuming from {}",
                cfg.output_dir.join("checkpoints").display()
            );
        }
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
    let mut optimizer = if cfg.use_muon {
        CheckpointHybridOptimizer::new(
            &varmap,
            adam,
            cfg.muon_momentum,
            cfg.muon_rms_scale,
        )?
    } else {
        bail!("use_muon=false is no longer supported; Muon+AdamW hybrid is required");
    };
    let parameter_names = optimizer.parameter_names();
    let parameter_count = parameter_count(&varmap);

    let resume_source = cfg
        .resume
        .clone()
        .or_else(|| implicit_resume_source(cfg));
    let resumed_from = resume_source
        .as_deref()
        .map(resolve_resume_checkpoint)
        .transpose()?;
    let mut state = if let Some(bundle) = &resumed_from {
        load_training_checkpoint(bundle, cfg, &mut varmap, &mut optimizer)?
    } else {
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
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
            profile_emitted: false,
            legacy_runtime_trace: None,
        }
    };
    let mut latest_checkpoint = resumed_from.clone();
    let mut latest_checkpoint_step = resumed_from.as_ref().map(|_| state.global_step);
    let mut updates_this_run = 0usize;
    if resumed_from.is_some() {
        device.synchronize()?;
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

    loop {
        let complete = state.lesson_index == cfg.lessons.len();
        if complete {
            if latest_checkpoint.is_none() {
                latest_checkpoint = Some(save_training_checkpoint(
                    &varmap,
                    &optimizer,
                    &state,
                    cfg,
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
        let trace_step = state.global_step;
        let cg_profile =
            StepProfileCapture::begin(trace_step, state.profile_emitted, &cfg.output_dir)?;
        let prof = profile.enabled();
        let accum = cfg.grad_accum.max(1);
        let sigreg_seed = cfg.seed.wrapping_add(state.global_step);
        let loss_weights = lesson_loss_weights(
            lesson,
            cfg,
            state.step_in_lesson,
            state.global_step,
        );
        let depth = sample_recursion_depth(cfg, state.global_step);
        let run_rollout_this_step = loss_weights.rollout > 0.0
            && matches!(lesson.as_str(), "sequential" | "retarget");
        let rollout_episode_start = scheduled_episode_start(
            cfg.seed,
            state.global_step,
            0,
            accum,
            cfg.shuffled_episodes,
        );
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
            let pf = prefetcher.as_mut().unwrap();
            if pf.ready_len() < accum {
                enqueue_batch_prefetch(pf, curriculum, cfg, state.global_step, accum)?;
            }
        }
        let accum_f = accum as f64;
        let mut accumulated_grads: Option<GradStore> = None;
        let mut step_metrics = LessonLossMeans::default();
        let mut rollout_trace_cache: Option<Vec<TransitionSample>> = None;
        for micro in 0..accum {
            let samples = {
                let _cg = cg_profile.span("generate", SpanKind::Module);
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
                enqueue_batch_prefetch(
                    prefetcher.as_mut().unwrap(),
                    curriculum,
                    cfg,
                    state.global_step.wrapping_add(1),
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
                        Err(resume_from) => collect_rollout_trace(
                            curriculum,
                            cfg.seed,
                            resume_from,
                            Split::Train,
                        )?,
                    }
                });
            }
            let batch = {
                let _cg = cg_profile.span("stage", SpanKind::Module);
                timed(prof, &device, &mut profile.stage, || {
                    let _span = tracing::info_span!("stage").entered();
                    batch_from_samples(&samples, &device)
                })?
            };
            let micro_sigreg_seed = sigreg_seed.wrapping_add(micro as u64);
            let run_rollout = micro == 0
                && loss_weights.rollout > 0.0
                && matches!(lesson.as_str(), "sequential" | "retarget");
            let (micro_losses, micro_rollout, micro_total) = {
                let _cg = cg_profile.span("forward", SpanKind::Function);
                timed(prof, &device, &mut profile.forward, || {
                    let _span = tracing::info_span!("forward").entered();
                    let losses = leworld_loss(
                        &model,
                        &batch,
                        cfg,
                        depth,
                        micro_sigreg_seed,
                        loss_weights,
                    )?;
                    let rollout_trace = if run_rollout {
                        rollout_trace_cache.as_ref()
                    } else {
                        None
                    };
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
                            &device,
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
                        Tensor::zeros((), DType::F32, &device)?
                    };
                    let prefix_multi = if rollout_trace.is_some() && loss_weights.prefix > 0.0 {
                        prefix_multi_horizon_loss(
                            &model,
                            rollout_trace.unwrap(),
                            &device,
                        )?
                    } else {
                        Tensor::zeros((), DType::F32, &device)?
                    };
                    let total = losses
                        .total
                        .add(&rollout.affine(loss_weights.rollout, 0.0)?)?
                        .add(&prefix_multi.affine(loss_weights.prefix, 0.0)?)?;
                    Ok((losses, rollout, total))
                })?
            };
            let micro_vals = ensure_all_finite(&[
                ("total", &micro_total),
                ("next_latent", &micro_losses.next_latent),
                ("rollout", &micro_rollout),
                ("sigreg", &micro_losses.sigreg),
                ("event", &micro_losses.event),
                ("q", &micro_losses.q),
            ])?;
            let inv = 1.0 / accum_f;
            step_metrics.total += micro_vals[0] as f64 * inv;
            step_metrics.next_latent += micro_vals[1] as f64 * inv;
            step_metrics.rollout += micro_vals[2] as f64 * inv;
            step_metrics.sigreg += micro_vals[3] as f64 * inv;
            step_metrics.event += micro_vals[4] as f64 * inv;
            step_metrics.q += micro_vals[5] as f64 * inv;

            let scaled_micro = micro_total.affine(inv, 0.0)?;
            let micro_grads = {
                let _cg = cg_profile.span("backward", SpanKind::Function);
                timed(prof, &device, &mut profile.backward, || {
                    let _span = tracing::info_span!("backward").entered();
                    scaled_micro.backward().map_err(Into::into)
                })?
            };
            match accumulated_grads {
                None => accumulated_grads = Some(micro_grads),
                Some(ref mut acc) => accumulate_grad_store(acc, micro_grads)?,
            }
        }
        let mut grads = accumulated_grads
            .ok_or_else(|| anyhow::anyhow!("grad_accum produced no microbatches"))?;
        clip_gradients_gpu(&mut grads, &varmap, MAX_GRAD_NORM)?;
        if cg_profile.active() {
            cg_profile.record_gradients(&varmap, &grads)?;
            cg_profile.finish()?;
            state.profile_emitted = true;
        }
        timed(prof, &device, &mut profile.optimizer, || {
            let _span = tracing::info_span!("optimizer").entered();
            optimizer.step(&grads)
        })?;
        drop(grads);
        if device.is_cuda() {
            let _ = device.synchronize();
        }

        timed(prof, &device, &mut profile.metrics, || {
            let _span = tracing::info_span!("metrics").entered();
            state.active_sums.total += step_metrics.total;
            state.active_sums.next_latent += step_metrics.next_latent;
            state.active_sums.rollout += step_metrics.rollout;
            state.active_sums.sigreg += step_metrics.sigreg;
            state.active_sums.event += step_metrics.event;
            state.active_sums.q += step_metrics.q;
            Ok(())
        })?;
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
        let periodic = cfg.checkpoint_every_steps > 0
            && state.global_step % cfg.checkpoint_every_steps as u64 == 0;
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
                prefetcher = Some(BatchPrefetcher::new());
            }
        }
        profile.steps += 1;
        profile.report(state.global_step);
        if malloc_trim_every > 0 && state.global_step as usize % malloc_trim_every == 0 {
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
    use candle_core::Var;
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
            output_dir,
            ..TrainConfig::default()
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
            assert!((dl.sigreg - dr.sigreg).abs() < eps, "sigreg");
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
    fn one_hot_conversion_and_event_mask() -> Result<()> {
        let device = Device::Cpu;
        let mut coordinate_sample = toy_sample(7);
        coordinate_sample.action = ArcAction::new(6, Some(63), Some(21))?;
        let samples = vec![toy_sample(3), coordinate_sample];
        let batch = batch_from_samples(&samples, &device)?;
        assert_eq!(batch.frames.dims(), &[2, 1, 64, 64]);
        let f0 = batch.frames.get(0)?;
        let pix = f0.flatten_all()?.to_vec1::<u8>()?;
        assert!(pix.iter().all(|&v| v == 3));

        let one_hot = frames_to_one_hot(
            &samples.iter().map(|s| s.current.clone()).collect::<Vec<_>>(),
            &device,
        )?;
        assert_eq!(one_hot.dims(), &[2, 16, 64, 64]);
        let oh0 = one_hot.get(0)?;
        let ch3 = oh0.get(3)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(ch3.iter().all(|v| *v == 1.0));

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
            assert!(g.iter().all(|v| v.is_finite()), "gradient not finite: {g:?}");
        }

        // Matches the closed form on values where the naive version is safe.
        let mid = Tensor::new(&[-2.0f32, -0.5, 0.5, 3.0], &device)?;
        let t = Tensor::new(&[0.0f32, 1.0, 0.0, 1.0], &device)?;
        let got = bce_with_logits(&mid, &t)?.to_scalar::<f32>()?;
        let want = candle_nn::loss::binary_cross_entropy_with_logit(&mid, &t)?.to_scalar::<f32>()?;
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
            match accumulated {
                None => accumulated = Some(micro_grads),
                Some(ref mut acc) => accumulate_grad_store(acc, micro_grads)?,
            }
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
    fn effective_sigreg_max_rows_caps_spatial_batch_128() {
        let cfg = TrainConfig {
            physical_batch: 128,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_max_rows: 4096,
            ..TrainConfig::default()
        };
        assert_eq!(effective_sigreg_max_rows(&cfg), 1024);
        let unpooled = TrainConfig {
            sigreg_spatial_pool: false,
            ..cfg.clone()
        };
        assert_eq!(effective_sigreg_max_rows(&unpooled), 1024);
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
    fn rollout_horizon_caps_retarget() {
        assert_eq!(rollout_horizon_for_lesson("retarget", 80, 100), 4);
        assert_eq!(rollout_horizon_for_lesson("sequential", 80, 100), 8);
    }

    #[test]
    fn rollout_teacher_mix_is_higher_on_retarget() {
        assert!(rollout_teacher_mix("retarget", 0, 100) > rollout_teacher_mix("sequential", 0, 100));
        assert_eq!(rollout_teacher_mix("sequential", 100, 100), 0.0);
    }

    #[test]
    fn lesson_to_curriculum_maps_auxiliary_warmup_lessons() {
        assert_eq!(lesson_to_curriculum("q_calibration").unwrap(), "sequential");
        assert_eq!(lesson_to_curriculum("events").unwrap(), "sequential");
        assert_eq!(lesson_to_curriculum("exploration").unwrap(), "exploration");
        assert_eq!(lesson_to_curriculum("falsification").unwrap(), "p1c_falsification");
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
        let loss = open_loop_latent_loss(&model, &trace, &device, 4, depth, 0.25, cfg.seed)?;
        assert!(ensure_finite("open_loop", &loss)?.is_finite());
        let grads = loss.backward()?;
        assert!(varmap
            .all_vars()
            .iter()
            .any(|var| grads.get(var.as_tensor()).is_some()));
        Ok(())
    }

    #[test]
    fn report_serialization_roundtrip() -> Result<()> {
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
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
            checkpoint: PathBuf::from("m.safetensors"),
            export_checkpoint: None,
            config_path: PathBuf::from("c.json"),
            profile_trace: Some(PathBuf::from("profile.jsonl")),
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
        let root =
            std::env::temp_dir().join(format!("tofy-p2-auto-resume-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.join("run"));
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        assert_eq!(paused.status, TrainStatus::Paused);
        assert_eq!(paused.global_step, 1);

        cfg.max_steps_this_run = None;
        cfg.resume = None;
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        assert_eq!(resumed.global_step, paused.global_step + 1);
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn pause_resume_matches_uninterrupted_training_exactly() -> Result<()> {
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
            resumed_state.completed_lessons,
            full_state.completed_lessons
        );
        assert_eq!(resumed_state.active_sums, full_state.active_sums);

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
                assert_eq!(a, b, "optimizer mismatch at {muon_key}");
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
                assert_eq!(a, b, "optimizer mismatch at {key}");
            }
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_merges_saved_contract_on_explicit_resume() -> Result<()> {
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
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_allows_equal_effective_batch_schedule_migration() -> Result<()> {
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
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        let state: TrainerState =
            read_json(&resumed.latest_checkpoint.join("trainer_state.json"))?;
        assert_eq!(state.contract.physical_batch, 4);
        assert_eq!(state.contract.grad_accum, 1);
        assert_eq!(effective_batch_contract(&state.contract), 4);
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
        assert!(err.to_string().contains("effective batch"));
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn resume_allows_steady_gpu_toggle_on_explicit_resume() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-resume-steady-gpu-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut cfg = resume_test_config(root.clone());
        cfg.max_steps_this_run = Some(1);
        let paused = train(&cfg)?;
        cfg.max_steps_this_run = None;
        cfg.resume = Some(paused.latest_checkpoint);
        cfg.steady_gpu = true;
        let resumed = train(&cfg)?;
        assert_eq!(resumed.status, TrainStatus::Completed);
        let state: TrainerState =
            read_json(&resumed.latest_checkpoint.join("trainer_state.json"))?;
        assert!(state.contract.steady_gpu);
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
        let config_path = cfg.output_dir.join("config.json");
        let saved: TrainConfig = read_json(&config_path)?;
        let mut tampered = saved.clone();
        tampered.lr *= 2.0;
        write_json_atomic(&config_path, &tampered)?;
        let err = train(&cfg).expect_err("tampered config.json must reject resume");
        assert!(err.to_string().contains("training contract mismatch"));

        write_json_atomic(&config_path, &saved)?;
        fs::remove_file(paused.latest_checkpoint.join("optimizer.safetensors"))?;
        let err = train(&cfg).expect_err("missing optimizer state must reject resume");
        assert!(err.to_string().contains("checkpoint bundle is incomplete"));
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }
}
