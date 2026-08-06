//! Public clap argument structs and entrypoints for P2 train / eval.
//!
//! Defaults are tiny smoke settings and must not be treated as a research result.
//! Wiring into the top-level CLI is owned by the primary agent.

use crate::p2::eval::{evaluate, evaluate_arc3, EvalConfig};
use crate::p2::train::{train, TrainConfig, DEFAULT_LESSONS};
use crate::p2::muon::MUON_RMS_SCALE;
use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

/// `p2-train` — synthetic curriculum only (no ARC public recordings).
#[derive(Debug, Clone, Args)]
pub struct P2TrainArgs {
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Comma-separated lessons in order.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "dynamics,exploration,sequential,q_calibration,falsification"
    )]
    pub lessons: Vec<String>,

    #[arg(long, default_value_t = 2)]
    pub steps_per_lesson: usize,

    #[arg(long, default_value_t = 2)]
    pub physical_batch: usize,

    /// Recorded; initial trainer requires 1.
    #[arg(long, default_value_t = 1)]
    pub grad_accum: usize,

    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,

    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,

    #[arg(long, default_value_t = 8)]
    pub sigreg_projections: usize,

    #[arg(long, default_value_t = 5)]
    pub sigreg_knots: usize,

    #[arg(long, default_value_t = 0.003)]
    pub sigreg_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub event_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub q_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub rollout_weight: f64,

    #[arg(long, default_value_t = 0.05)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value_t = 128)]
    pub hidden_dim: usize,

    #[arg(long, default_value_t = 8)]
    pub action_dim: usize,

    #[arg(long, default_value_t = 2)]
    pub inner_steps: usize,

    #[arg(long, default_value_t = 2)]
    pub outer_steps: usize,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long, default_value = "runs/p2/smoke")]
    pub output_dir: PathBuf,

    /// Complete checkpoint bundle, checkpoints directory, or run directory to resume.
    #[arg(long)]
    pub resume: Option<PathBuf>,

    /// Save a complete resumable checkpoint every N updates; zero disables periodic saves.
    #[arg(long, default_value_t = 100)]
    pub checkpoint_every_steps: usize,

    /// Cleanly pause after N updates in this invocation (useful for batch schedulers).
    #[arg(long)]
    pub max_steps_this_run: Option<usize>,

    /// PTRM ranking loss cadence on sequential/retarget (`1` = every step). Not in resume contract.
    #[arg(long, default_value_t = 4)]
    pub ptrm_rank_every: usize,

    /// Sample inner/outer recursion depth uniformly in `1..=configured` each optimizer step.
    #[arg(long, default_value_t = false)]
    pub randomize_depth: bool,

    /// Fixed recursion depth every step for steadier GPU utilization (ignores randomize_depth).
    #[arg(long, default_value_t = false)]
    pub steady_gpu: bool,

    /// Supervise only the final outer recursion step (saves VRAM vs TRM deep supervision).
    #[arg(long, default_value_t = false)]
    pub supervise_last_outer_only: bool,

    /// Lesson-scoped loss schedule (dynamics → rollout → events/Q → PTRM).
    #[arg(long, default_value_t = true)]
    pub phased_training: bool,

    /// Stop-gradient on predicted `y` for event loss only (Q keeps full gradients).
    #[arg(long, default_value_t = true)]
    pub stop_grad_event_y: bool,

    #[arg(long, default_value_t = false)]
    pub residual_y_update: bool,

    #[arg(long, default_value_t = false)]
    pub warm_start_y: bool,

    #[arg(long, default_value_t = false)]
    pub sigreg_spatial: bool,

    /// 2×2 avg-pool latents before spatial SIGReg (4× fewer rows).
    #[arg(long, default_value_t = true)]
    pub sigreg_spatial_pool: bool,

    #[arg(long, default_value_t = false)]
    pub stop_grad_q_y: bool,

    #[arg(long, default_value_t = false)]
    pub q_quantile_targets: bool,

    #[arg(long, default_value_t = 0.0)]
    pub train_z_noise: f64,

    #[arg(long, default_value_t = false)]
    pub shuffled_episodes: bool,

    /// Force D=1 residual baseline (disables PTRM ranking and randomized depth).
    #[arg(long, default_value_t = false)]
    pub baseline_d1: bool,

    #[arg(long, default_value_t = 0.1)]
    pub prefix_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub reliability_weight: f64,

    #[arg(long, default_value_t = false)]
    pub bf16_conv: bool,

    #[arg(long, default_value_t = 8)]
    pub ensemble_members: usize,

    #[arg(long, default_value_t = true)]
    pub use_muon: bool,

    #[arg(long, default_value_t = 0.95)]
    pub muon_momentum: f64,

    #[arg(long, default_value_t = MUON_RMS_SCALE)]
    pub muon_rms_scale: f64,

    #[arg(long, default_value_t = 4096)]
    pub sigreg_max_rows: usize,

    #[arg(long, default_value_t = true)]
    pub prefetch_batches: bool,
}

impl P2TrainArgs {
    pub fn to_config(&self) -> TrainConfig {
        let lessons = if self.lessons.is_empty() {
            DEFAULT_LESSONS.iter().map(|s| (*s).to_string()).collect()
        } else {
            self.lessons.clone()
        };
        TrainConfig {
            seed: self.seed,
            lessons,
            steps_per_lesson: self.steps_per_lesson,
            physical_batch: self.physical_batch,
            grad_accum: self.grad_accum,
            lr: self.lr,
            weight_decay: self.weight_decay,
            sigreg_projections: self.sigreg_projections,
            sigreg_knots: self.sigreg_knots,
            sigreg_weight: self.sigreg_weight,
            event_weight: self.event_weight,
            q_weight: self.q_weight,
            rollout_weight: self.rollout_weight,
            q_mse_threshold: self.q_mse_threshold,
            hidden_dim: self.hidden_dim,
            action_dim: self.action_dim,
            inner_steps: self.inner_steps,
            outer_steps: self.outer_steps,
            device: self.device.clone(),
            output_dir: self.output_dir.clone(),
            resume: self.resume.clone(),
            checkpoint_every_steps: self.checkpoint_every_steps,
            max_steps_this_run: self.max_steps_this_run,
            ptrm_rank_every: self.ptrm_rank_every,
            randomize_depth: self.randomize_depth,
            steady_gpu: self.steady_gpu,
            supervise_last_outer_only: self.supervise_last_outer_only,
            phased_training: self.phased_training,
            stop_grad_event_y: self.stop_grad_event_y,
            residual_y_update: self.residual_y_update,
            warm_start_y: self.warm_start_y,
            sigreg_spatial: self.sigreg_spatial,
            sigreg_spatial_pool: self.sigreg_spatial_pool,
            stop_grad_q_y: self.stop_grad_q_y,
            q_quantile_targets: self.q_quantile_targets,
            train_z_noise: self.train_z_noise,
            shuffled_episodes: self.shuffled_episodes,
            baseline_d1: self.baseline_d1,
            prefix_weight: self.prefix_weight,
            reliability_weight: self.reliability_weight,
            bf16_conv: self.bf16_conv,
            ensemble_members: self.ensemble_members,
            use_muon: self.use_muon,
            muon_momentum: self.muon_momentum,
            muon_rms_scale: self.muon_rms_scale,
            sigreg_max_rows: self.sigreg_max_rows,
            prefetch_batches: self.prefetch_batches,
        }
    }
}

pub fn run_p2_train(args: P2TrainArgs) -> Result<()> {
    let cfg = args.to_config();
    let report = train(&cfg)?;
    println!(
        "p2-train status={:?} research_claim={} params={} step={} checkpoint={}",
        report.status,
        report.research_claim,
        report.parameter_count,
        report.global_step,
        report.latest_checkpoint.display()
    );
    for lesson in &report.lessons {
        println!(
            "  lesson={} curriculum={} mean_total={:.6} mean_rollout={:.6}",
            lesson.lesson, lesson.curriculum, lesson.mean_losses.total, lesson.mean_losses.rollout
        );
    }
    Ok(())
}

/// `p2-eval` — synthetic held-out (+ optional ARC recordings transfer).
#[derive(Debug, Clone, Args)]
pub struct P2EvalArgs {
    #[arg(long, default_value = "runs/p2/smoke/model.safetensors")]
    pub checkpoint: PathBuf,

    #[arg(long, default_value = "runs/p2/smoke/config.json")]
    pub train_config: PathBuf,

    #[arg(long, default_value_t = 2)]
    pub seed: u64,

    #[arg(long, default_value_t = 4)]
    pub synthetic_episodes: usize,

    #[arg(long, default_value_t = 2)]
    pub physical_batch: usize,

    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    pub ptrm_k: Vec<usize>,

    #[arg(long, default_value_t = 0.1)]
    pub ptrm_noise: f64,

    #[arg(long, default_value_t = 0.05)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long)]
    pub arc_recordings_dir: Option<PathBuf>,

    /// Official scorecard JSON (RHAE per https://docs.arcprize.org/methodology).
    #[arg(long)]
    pub scorecard_json: Option<PathBuf>,

    #[arg(long, default_value = "runs/p2/smoke/eval_report.json")]
    pub output: PathBuf,

    /// Raw per-episode rollout metrics as JSONL (Phase A).
    #[arg(long)]
    pub episode_jsonl: Option<PathBuf>,

    /// Bootstrap ensemble members for uncertainty (Phase D).
    #[arg(long, default_value_t = 8)]
    pub ensemble_members: usize,
}

impl P2EvalArgs {
    pub fn to_config(&self) -> EvalConfig {
        EvalConfig {
            checkpoint: self.checkpoint.clone(),
            train_config: self.train_config.clone(),
            seed: self.seed,
            synthetic_episodes: self.synthetic_episodes,
            physical_batch: self.physical_batch,
            ptrm_k: self.ptrm_k.clone(),
            ptrm_noise: self.ptrm_noise,
            q_mse_threshold: self.q_mse_threshold,
            device: self.device.clone(),
            arc_recordings_dir: self.arc_recordings_dir.clone(),
            scorecard_json: self.scorecard_json.clone(),
            output: self.output.clone(),
            episode_jsonl: self.episode_jsonl.clone(),
            ensemble_members: self.ensemble_members,
        }
    }
}

pub fn run_p2_eval(args: P2EvalArgs) -> Result<()> {
    let report = evaluate(&args.to_config())?;
    println!(
        "p2-eval smoke complete research_claim={} official_rhae={:?} public_fit={} \
         dyn_mse={:?} plan_mse={:?} dyn_closed@16={:?} dyn_open@16={:?} \
         q_unreliable={:?} confident_err={:?} r2={:?} cov_fro={:?} arc3_runs={:?}",
        report.research_claim,
        report.official_rhae,
        report.public_data_used_for_fitting,
        report.synthetic_dynamics.one_step_latent_mse,
        report.synthetic_planner.one_step_latent_mse,
        report
            .synthetic_dynamics
            .closed_loop
            .as_ref()
            .and_then(|r| r.mse_16),
        report
            .synthetic_dynamics
            .rollout
            .as_ref()
            .and_then(|r| r.mse_16),
        report
            .synthetic_dynamics
            .q_surprise
            .as_ref()
            .and_then(|q| q.mean_q_when_unreliable),
        report
            .synthetic_dynamics
            .q_surprise
            .as_ref()
            .and_then(|q| q.confident_error_rate),
        report
            .synthetic_dynamics
            .identifiability
            .as_ref()
            .and_then(|m| m.r2_h_to_z),
        report
            .synthetic_dynamics
            .identifiability
            .as_ref()
            .and_then(|m| m.latent_covariance_frobenius),
        report
            .arc3_recording_runs
            .as_ref()
            .map(|b| b.n_runs),
    );
    Ok(())
}

/// `p2-arc3-eval` — ARC recording-dir transfer eval (requires recordings path).
#[derive(Debug, Clone, Args)]
pub struct P2Arc3EvalArgs {
    #[arg(long, default_value = "runs/p2/smoke/model.safetensors")]
    pub checkpoint: PathBuf,

    #[arg(long, default_value = "runs/p2/smoke/config.json")]
    pub train_config: PathBuf,

    #[arg(long, default_value_t = 2)]
    pub seed: u64,

    #[arg(long, default_value_t = 2)]
    pub physical_batch: usize,

    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    pub ptrm_k: Vec<usize>,

    #[arg(long, default_value_t = 0.1)]
    pub ptrm_noise: f64,

    #[arg(long, default_value_t = 0.05)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long)]
    pub arc_recordings_dir: PathBuf,

    #[arg(long)]
    pub scorecard_json: Option<PathBuf>,

    #[arg(long, default_value = "runs/p2/smoke/arc3_eval_report.json")]
    pub output: PathBuf,
}

impl P2Arc3EvalArgs {
    pub fn to_config(&self) -> EvalConfig {
        EvalConfig {
            checkpoint: self.checkpoint.clone(),
            train_config: self.train_config.clone(),
            seed: self.seed,
            synthetic_episodes: 0,
            physical_batch: self.physical_batch,
            ptrm_k: self.ptrm_k.clone(),
            ptrm_noise: self.ptrm_noise,
            q_mse_threshold: self.q_mse_threshold,
            device: self.device.clone(),
            arc_recordings_dir: Some(self.arc_recordings_dir.clone()),
            scorecard_json: self.scorecard_json.clone(),
            output: self.output.clone(),
            episode_jsonl: None,
            ensemble_members: 8,
        }
    }
}

pub fn run_p2_arc3_eval(args: P2Arc3EvalArgs) -> Result<()> {
    let report = evaluate_arc3(&args.to_config())?;
    let n = report
        .arc3_transfer
        .as_ref()
        .map(|s| s.n_samples)
        .unwrap_or(0);
    let runs = report
        .arc3_recording_runs
        .as_ref()
        .map(|b| b.n_runs)
        .unwrap_or(0);
    println!(
        "p2-arc3-eval smoke complete research_claim={} official_rhae={:?} public_fit={} samples={} recording_runs={}",
        report.research_claim, report.official_rhae, report.public_data_used_for_fitting, n, runs
    );
    Ok(())
}
