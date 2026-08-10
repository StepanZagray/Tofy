//! Public clap argument structs and entrypoints for P2 train / eval.
//!
//! Defaults are tiny smoke settings and must not be treated as a research result.
//! Wiring into the top-level CLI is owned by the primary agent.

use crate::p2::arc3_live::{evaluate_live, list_public_games, LiveEvalConfig};
use crate::p2::eval::{evaluate, evaluate_arc3, EvalConfig, EvalMode};
use crate::p2::muon::MUON_RMS_SCALE;
use crate::p2::train::{train, SigregTarget, TrainConfig, DEFAULT_LESSONS};
use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

/// `p2-train` — synthetic curriculum only (no ARC public recordings).
#[derive(Debug, Clone, Args)]
pub struct P2TrainArgs {
    #[arg(long, default_value_t = 2)]
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

    /// Number of physical microbatches per optimizer update.
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

    /// Explicitly migrate physical batch / accumulation at equal effective batch.
    /// This changes the trajectory and is durably labeled as a migration.
    #[arg(long, default_value_t = false)]
    pub allow_batch_schedule_migration: bool,

    /// Save a complete resumable checkpoint every N updates; zero disables periodic saves.
    #[arg(long, default_value_t = 100)]
    pub checkpoint_every_steps: usize,

    /// Cleanly pause after N updates in this invocation (useful for batch schedulers).
    #[arg(long)]
    pub max_steps_this_run: Option<usize>,

    /// One-based optimizer update captured as a candle-graph evidence bundle.
    #[arg(long, default_value_t = 2)]
    pub profile_update: u64,

    /// PTRM ranking loss cadence on sequential/retarget (`1` = every step).
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

    /// Feed unpooled pre-RMS spatial cells directly to SIGReg without a projector.
    #[arg(long, default_value_t = false, conflicts_with = "sigreg_projector")]
    pub sigreg_pre_rms_spatial: bool,

    /// Experimental pre-RMS pooled encoder projector with T×B×D SIGReg geometry.
    #[arg(long, default_value_t = false)]
    pub sigreg_projector: bool,

    #[arg(long, default_value_t = 128)]
    pub sigreg_projector_dim: usize,

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

    #[arg(long, default_value_t = 0.95)]
    pub muon_momentum: f64,

    #[arg(long, default_value_t = MUON_RMS_SCALE)]
    pub muon_rms_scale: f64,

    #[arg(long, default_value_t = 4096)]
    pub sigreg_max_rows: usize,

    /// SIGReg population: marginal control or temporally centered residuals.
    #[arg(long, value_enum, default_value_t = SigregTarget::Marginal)]
    pub sigreg_target: SigregTarget,

    /// Consecutive transitions per ordered SIGReg window (temporal-residual needs >= 2).
    #[arg(long, default_value_t = 8)]
    pub sigreg_temporal_window: usize,

    /// Convex weight on global-pooled TC rows; 0 keeps the original cell-row objective.
    #[arg(long, default_value_t = 0.0)]
    pub sigreg_global_mix: f64,

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
            allow_batch_schedule_migration: self.allow_batch_schedule_migration,
            checkpoint_every_steps: self.checkpoint_every_steps,
            max_steps_this_run: self.max_steps_this_run,
            profile_update: self.profile_update,
            ptrm_rank_every: self.ptrm_rank_every,
            randomize_depth: self.randomize_depth,
            steady_gpu: self.steady_gpu,
            supervise_last_outer_only: self.supervise_last_outer_only,
            phased_training: self.phased_training,
            stop_grad_event_y: self.stop_grad_event_y,
            residual_y_update: self.residual_y_update,
            warm_start_y: self.warm_start_y,
            sigreg_spatial: self.sigreg_spatial || self.sigreg_pre_rms_spatial,
            sigreg_spatial_pool: self.sigreg_spatial_pool && !self.sigreg_pre_rms_spatial,
            sigreg_pre_rms_spatial: self.sigreg_pre_rms_spatial,
            sigreg_projector: self.sigreg_projector,
            sigreg_projector_dim: self.sigreg_projector_dim,
            stop_grad_q_y: self.stop_grad_q_y,
            q_quantile_targets: self.q_quantile_targets,
            train_z_noise: self.train_z_noise,
            shuffled_episodes: self.shuffled_episodes,
            baseline_d1: self.baseline_d1,
            prefix_weight: self.prefix_weight,
            reliability_weight: self.reliability_weight,
            bf16_conv: self.bf16_conv,
            ensemble_members: self.ensemble_members,
            muon_momentum: self.muon_momentum,
            muon_rms_scale: self.muon_rms_scale,
            sigreg_max_rows: self.sigreg_max_rows,
            sigreg_target: self.sigreg_target,
            sigreg_temporal_window: self.sigreg_temporal_window,
            sigreg_global_mix: self.sigreg_global_mix,
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

    /// Evaluation graph: complete, representation-only, or rollout-only.
    #[arg(long, value_enum, default_value_t = EvalMode::Full)]
    pub eval_mode: EvalMode,
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
            mode: self.eval_mode,
            representation_row_cap: crate::p2::representation::DEFAULT_REPRESENTATION_ROW_CAP,
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
        report.arc3_recording_runs.as_ref().map(|b| b.n_runs),
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
            mode: EvalMode::Full,
            representation_row_cap: crate::p2::representation::DEFAULT_REPRESENTATION_ROW_CAP,
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

/// `p2-arc3-live-eval` — run a frozen checkpoint on public ARC-AGI-3 games.
#[derive(Debug, Clone, Args)]
pub struct P2Arc3LiveEvalArgs {
    #[arg(long, default_value = "runs/p2/smoke/model.safetensors")]
    pub checkpoint: PathBuf,

    #[arg(long, default_value = "runs/p2/smoke/config.json")]
    pub train_config: PathBuf,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long, default_value = "https://three.arcprize.org")]
    pub base_url: String,

    /// Environment variable containing the API key. `.env` is loaded automatically.
    #[arg(long, default_value = "ARC_API_KEY")]
    pub api_key_env: String,

    /// Optional comma-separated subset. The default evaluates every discovered public game.
    #[arg(long, value_delimiter = ',')]
    pub games: Vec<String>,

    #[arg(long, default_value_t = 512)]
    pub max_actions_per_game: usize,

    /// Maximum candidate actions scored together by Candle.
    #[arg(long, default_value_t = 128)]
    pub physical_batch: usize,

    /// Maximum candidate coordinates considered whenever ACTION6 is available.
    #[arg(long, default_value_t = 128)]
    pub action6_max_candidates: usize,

    /// Uniform ACTION6 grid spacing, augmented with visible-object coordinates.
    #[arg(long, default_value_t = 8)]
    pub action6_grid_stride: usize,

    #[arg(long, default_value_t = 30)]
    pub request_timeout_secs: u64,

    #[arg(long, default_value = "runs/p2/arc3_live_report.json")]
    pub output: PathBuf,

    /// Authenticate and list public games without opening a scorecard or loading a checkpoint.
    #[arg(long, default_value_t = false)]
    pub list_only: bool,
}

impl P2Arc3LiveEvalArgs {
    pub fn to_config(&self) -> LiveEvalConfig {
        LiveEvalConfig {
            checkpoint: self.checkpoint.clone(),
            train_config: self.train_config.clone(),
            device: self.device.clone(),
            base_url: self.base_url.clone(),
            api_key_env: self.api_key_env.clone(),
            games: self.games.clone(),
            max_actions_per_game: self.max_actions_per_game,
            physical_batch: self.physical_batch,
            action6_max_candidates: self.action6_max_candidates,
            action6_grid_stride: self.action6_grid_stride,
            request_timeout_secs: self.request_timeout_secs,
            output: self.output.clone(),
        }
    }
}

pub fn run_p2_arc3_live_eval(args: P2Arc3LiveEvalArgs) -> Result<()> {
    let config = args.to_config();
    if args.list_only {
        let games = list_public_games(&config)?;
        println!("public ARC-AGI-3 games: {}", games.len());
        for game in games {
            println!("{}\t{}", game.game_id, game.title);
        }
        return Ok(());
    }

    let report = evaluate_live(&config)?;
    let completed = report
        .games
        .iter()
        .filter(|game| game.stop_reason == "completed")
        .count();
    let actions: usize = report.games.iter().map(|game| game.actions).sum();
    println!(
        "p2-arc3-live-eval complete games={}/{} completed={} actions={} official_rhae={:?} public_fit={} report={}",
        report.selected_game_count,
        report.discovered_games.len(),
        completed,
        actions,
        report.official_rhae,
        report.public_data_used_for_fitting,
        config.output.display(),
    );
    Ok(())
}
