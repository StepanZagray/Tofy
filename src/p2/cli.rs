//! Public clap argument structs and entrypoints for P2 train / eval.
//!
//! Defaults are tiny smoke settings and must not be treated as a research result.
//! Wiring into the top-level CLI is owned by the primary agent.

use crate::p2::eval::{evaluate, evaluate_arc3, EvalConfig};
use crate::p2::train::{train, TrainConfig, DEFAULT_LESSONS};
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
        default_value = "dynamics,sequential,falsification,retarget"
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

    #[arg(long, default_value_t = 0.01)]
    pub sigreg_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub event_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub q_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub rollout_weight: f64,

    #[arg(long, default_value_t = 0.5)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value_t = 32)]
    pub hidden_dim: usize,

    #[arg(long, default_value_t = 8)]
    pub action_dim: usize,

    #[arg(long, default_value_t = 1)]
    pub inner_steps: usize,

    #[arg(long, default_value_t = 1)]
    pub outer_steps: usize,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long, default_value = "runs/p2/smoke")]
    pub output_dir: PathBuf,

    /// Optional safetensors resume path (weights only).
    #[arg(long)]
    pub resume: Option<PathBuf>,
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
        }
    }
}

pub fn run_p2_train(args: P2TrainArgs) -> Result<()> {
    let cfg = args.to_config();
    let report = train(&cfg)?;
    println!(
        "p2-train smoke complete research_claim={} params={} checkpoint={}",
        report.research_claim,
        report.parameter_count,
        report.checkpoint.display()
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

    #[arg(long, default_value_t = 0.5)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long)]
    pub arc_recordings_dir: Option<PathBuf>,

    #[arg(long, default_value = "runs/p2/smoke/eval_report.json")]
    pub output: PathBuf,
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
            output: self.output.clone(),
        }
    }
}

pub fn run_p2_eval(args: P2EvalArgs) -> Result<()> {
    let report = evaluate(&args.to_config())?;
    println!(
        "p2-eval smoke complete research_claim={} official_rhae={:?} public_fit={} one_step_mse={:?}",
        report.research_claim,
        report.official_rhae,
        report.public_data_used_for_fitting,
        report.synthetic.one_step_latent_mse
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

    #[arg(long, default_value_t = 0.5)]
    pub q_mse_threshold: f64,

    #[arg(long, default_value = "cpu")]
    pub device: String,

    #[arg(long)]
    pub arc_recordings_dir: PathBuf,

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
            output: self.output.clone(),
        }
    }
}

pub fn run_p2_arc3_eval(args: P2Arc3EvalArgs) -> Result<()> {
    let report = evaluate_arc3(&args.to_config())?;
    let n = report.arc3.as_ref().map(|s| s.n_samples).unwrap_or(0);
    println!(
        "p2-arc3-eval smoke complete research_claim={} official_rhae={:?} public_fit={} samples={}",
        report.research_claim, report.official_rhae, report.public_data_used_for_fitting, n
    );
    Ok(())
}
