//! P2 world-model evaluation (synthetic held-out + ARC recording transfer).

use crate::domain::Split;
use crate::gpu_lock::GpuSessionGuard;
use crate::p2::arc3::{import_recordings_dir, summarize_recordings_dir, RecordingRunSummary};
use crate::p2::calibration::{binary_auroc, expected_calibration_error, risk_coverage_buckets};
use crate::p2::data::{generate_curriculum, generate_hazard_one_step, TransitionSample, ORACLE_LATENT_DIM};
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, RecursionStepProbe, PtrmConfig, WorldModel,
    EVENT_GOAL_FAILED,
};
use crate::p2::rhae::{benchmark_from_scorecard_json, official_rhae_from_benchmark, ScorecardBenchmark};
use crate::p2::train::{
    batch_from_samples, load_train_config, resolve_device, BatchTensors, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, VarBuilder, VarMap};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use rand::Rng;
use rand::SeedableRng;

pub const EVAL_REPORT_SCHEMA: &str = "p2.eval_report.v7";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonRolloutStats {
    pub n: usize,
    pub finite_n: usize,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub trimmed_mean: Option<f64>,
    pub p90: Option<f64>,
    pub p95: Option<f64>,
    pub ci95_low: Option<f64>,
    pub ci95_high: Option<f64>,
    pub max: Option<f64>,
    /// Model MSE divided by copy-forward MSE at the same horizon (when available).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_mean: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub seed: u64,
    pub synthetic_episodes: usize,
    pub physical_batch: usize,
    pub ptrm_k: Vec<usize>,
    pub ptrm_noise: f64,
    pub q_mse_threshold: f64,
    pub device: String,
    pub arc_recordings_dir: Option<PathBuf>,
    /// Official scorecard JSON from https://docs.arcprize.org/scorecards .
    pub scorecard_json: Option<PathBuf>,
    pub output: PathBuf,
    /// Optional JSONL sink for raw per-episode rollout rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episode_jsonl: Option<PathBuf>,
    #[serde(default = "default_ensemble_members")]
    pub ensemble_members: usize,
}

fn default_ensemble_members() -> usize {
    8
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::from("runs/p2/smoke/model.safetensors"),
            train_config: PathBuf::from("runs/p2/smoke/config.json"),
            seed: 2,
            synthetic_episodes: 4,
            physical_batch: 2,
            ptrm_k: vec![1, 2, 4],
            ptrm_noise: 0.1,
            q_mse_threshold: 0.5,
            device: "cpu".into(),
            arc_recordings_dir: None,
            scorecard_json: None,
            output: PathBuf::from("runs/p2/smoke/eval_report.json"),
            episode_jsonl: None,
            ensemble_members: 8,
        }
    }
}

impl EvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.physical_batch == 0 {
            bail!("physical_batch must be > 0");
        }
        if self.ptrm_k.is_empty() || self.ptrm_k.contains(&0) {
            bail!("ptrm_k must contain only positive values");
        }
        let unique: BTreeSet<_> = self.ptrm_k.iter().copied().collect();
        if unique.len() != self.ptrm_k.len() {
            bail!("ptrm_k values must be unique");
        }
        if !(self.ptrm_noise.is_finite() && self.ptrm_noise >= 0.0) {
            bail!("ptrm_noise must be finite and >= 0");
        }
        if !(self.q_mse_threshold.is_finite() && self.q_mse_threshold >= 0.0) {
            bail!("q_mse_threshold must be finite and >= 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetrics {
    pub labeled: usize,
    pub accuracy: Option<f64>,
    pub bce: Option<f64>,
    pub hazard_failure_labeled: usize,
    pub hazard_false_negatives: usize,
    pub hazard_false_negative_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMetrics {
    pub n: usize,
    pub brier: Option<f64>,
    pub accuracy: Option<f64>,
    /// Fraction of transitions labeled positive (MSE below threshold).
    pub positive_label_rate: Option<f64>,
    /// Mean of per-class recall when both classes appear.
    pub balanced_accuracy: Option<f64>,
    /// True when >90% of labels are positive (threshold accuracy is saturated).
    pub saturated: bool,
}

#[derive(Debug, Clone, Default)]
struct QEvalAccum {
    n: usize,
    brier_sum: f64,
    correct: usize,
    positive_labels: usize,
    tp: usize,
    tn: usize,
    fp: usize,
    fn_: usize,
}

impl QEvalAccum {
    fn merge(&mut self, other: QEvalAccum) {
        self.n += other.n;
        self.brier_sum += other.brier_sum;
        self.correct += other.correct;
        self.positive_labels += other.positive_labels;
        self.tp += other.tp;
        self.tn += other.tn;
        self.fp += other.fp;
        self.fn_ += other.fn_;
    }

    fn finalize(self) -> QMetrics {
        if self.n == 0 {
            return QMetrics {
                n: 0,
                brier: None,
                accuracy: None,
                positive_label_rate: None,
                balanced_accuracy: None,
                saturated: false,
            };
        }
        let positive_label_rate = self.positive_labels as f64 / self.n as f64;
        let tpr = if self.tp + self.fn_ > 0 {
            self.tp as f64 / (self.tp + self.fn_) as f64
        } else {
            f64::NAN
        };
        let tnr = if self.tn + self.fp > 0 {
            self.tn as f64 / (self.tn + self.fp) as f64
        } else {
            f64::NAN
        };
        let balanced_accuracy = if tpr.is_finite() && tnr.is_finite() {
            Some((tpr + tnr) / 2.0)
        } else if tpr.is_finite() {
            Some(tpr)
        } else if tnr.is_finite() {
            Some(tnr)
        } else {
            None
        };
        QMetrics {
            n: self.n,
            brier: Some(self.brier_sum / self.n as f64),
            accuracy: Some(self.correct as f64 / self.n as f64),
            positive_label_rate: Some(positive_label_rate),
            balanced_accuracy,
            saturated: positive_label_rate > 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtrmKMetrics {
    pub k: usize,
    pub noise: f64,
    pub n: usize,
    pub pass_at_k: f64,
    pub best_q_at_k: f64,
    /// `pass_at_k - best_q_at_k` (ranking utility gap; lower is better).
    pub ranking_gap: f64,
    /// Fraction of transitions where argmax Q picks the lowest-MSE trajectory.
    pub q_oracle_rank_accuracy: Option<f64>,
    pub disagreement: f64,
    /// Deterministic model-trajectory evaluations per transition.
    pub trajectory_evaluations_per_transition: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutMetrics {
    pub n4: usize,
    pub mse_4: Option<f64>,
    pub n8: usize,
    pub mse_8: Option<f64>,
    pub n16: usize,
    pub mse_16: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h4: Option<HorizonRolloutStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h8: Option<HorizonRolloutStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h16: Option<HorizonRolloutStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_closed_ratio_8: Option<f64>,
}

/// Q reliability vs latent error (anti-hallucination calibration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QSurpriseMetrics {
    pub n: usize,
    /// Mean sigmoid(Q) on transitions where MSE > threshold (should be low).
    pub mean_q_when_unreliable: Option<f64>,
    /// Mean sigmoid(Q) on transitions where MSE <= threshold (should be high).
    pub mean_q_when_reliable: Option<f64>,
    /// Fraction of high-MSE steps where Q > 0.5 (confident hallucination rate).
    pub confident_error_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub n: usize,
    pub ece: Option<f64>,
    pub reliability_auroc: Option<f64>,
    pub risk_coverage: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationGates {
    pub ece_pass: Option<bool>,
    pub auroc_pass: Option<bool>,
    pub risk_monotone_pass: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveProbeMetrics {
    pub noop_identity_mse: Option<f64>,
    pub action_effect_mse: Option<f64>,
    pub inverse_action_cosine: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursionProbeSummary {
    pub n_steps: usize,
    pub mean_residual_norm: Option<f64>,
    pub mean_latent_norm: Option<f64>,
    pub mean_amplification: Option<f64>,
    pub steps: Vec<RecursionStepProbe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetrics {
    pub members: usize,
    pub mean_disagreement: Option<f64>,
    pub uncertainty_auroc: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRolloutRow {
    pub seed: u64,
    pub episode_id: u64,
    pub family: String,
    pub horizon: usize,
    pub open_mse: Option<f64>,
    pub closed_mse: Option<f64>,
    pub copy_forward_mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedComputeMetrics {
    pub ptrm_k_equivalent: usize,
    pub outer_steps: usize,
    pub n: usize,
    pub one_step_latent_mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiabilityMetrics {
    pub n_labeled: usize,
    pub oracle_dim: usize,
    pub latent_dim: usize,
    /// Held-out validation R² (ridge bridge fit on train split only).
    pub r2_h_to_z: Option<f64>,
    /// In-sample train R² (diagnostic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r2_h_to_z_train: Option<f64>,
    /// Frobenius norm ||Cov(h) - I|| on centered encoder outputs.
    pub latent_covariance_frobenius: Option<f64>,
    /// Mean ||h(s') - h(s)||² on consecutive oracle-labeled transitions.
    pub mean_encoder_pair_mse: Option<f64>,
    /// Mean ||z' - z||² on the same consecutive pairs.
    pub mean_oracle_pair_mse: Option<f64>,
    /// Cosine between ridge-projected Δh and Δz on held-out pairs.
    pub pair_increment_cosine: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitEval {
    pub source: String,
    pub n_samples: usize,
    pub one_step_latent_mse: Option<f64>,
    pub identifiability: Option<IdentifiabilityMetrics>,
    pub events: EventMetrics,
    pub q: QMetrics,
    pub ptrm: Vec<PtrmKMetrics>,
    pub deterministic_matched_compute: Vec<MatchedComputeMetrics>,
    pub rollout: Option<RolloutMetrics>,
    /// Re-encode real frame every step (matches ARC play).
    pub closed_loop: Option<RolloutMetrics>,
    /// Predict `z0` every step (no dynamics); rollout error baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_forward: Option<RolloutMetrics>,
    pub q_surprise: Option<QSurpriseMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_gates: Option<CalibrationGates>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrastive: Option<ContrastiveProbeMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursion_probes: Option<RecursionProbeSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble: Option<EnsembleMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arc3RecordingBenchmark {
    pub n_runs: usize,
    pub total_actions: usize,
    pub total_levels_completed: i64,
    pub runs: Vec<RecordingRunSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub schema: String,
    pub seed: u64,
    pub checkpoint: PathBuf,
    pub device: String,
    pub q_mse_threshold: f64,
    pub ptrm_k: Vec<usize>,
    pub ptrm_noise: f64,
    /// Official ARC-AGI-3 RHAE (0–100%) when `--scorecard-json` is supplied.
    pub official_rhae: Option<f64>,
    /// Parsed official scorecard; see https://docs.arcprize.org/methodology .
    pub official_scorecard: Option<ScorecardBenchmark>,
    /// Recording-derived run counters (no human baselines; not RHAE).
    pub arc3_recording_runs: Option<Arc3RecordingBenchmark>,
    /// Public ARC recordings/games were not used for fitting.
    pub public_data_used_for_fitting: bool,
    /// Goal-free dynamics: one-step, exploration, coordinate, interact.
    pub synthetic_dynamics: SplitEval,
    /// Planning / calibration / falsification / retarget held-out probes.
    pub synthetic_planner: SplitEval,
    /// Optional transfer on imported ARC recordings (never used for training).
    pub arc3_transfer: Option<SplitEval>,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(value).context("serialize eval report")?;
    let tmp = {
        let mut os = path.as_os_str().to_owned();
        os.push(".tmp");
        PathBuf::from(os)
    };
    fs::write(&tmp, &json).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn load_model(
    train_cfg: &TrainConfig,
    weights: &Path,
    device: &Device,
) -> Result<(WorldModel, VarMap)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = WorldModel::new(train_cfg.model_config(), vb)?;
    varmap
        .load(weights)
        .with_context(|| format!("load {}", weights.display()))?;
    Ok((model, varmap))
}

fn with_thread_local_model<R>(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    device: &Device,
    f: impl FnOnce(&WorldModel) -> Result<R>,
) -> Result<R> {
    thread_local! {
        static TLS: RefCell<Option<(PathBuf, Arc<WorldModel>)>> = const { RefCell::new(None) };
    }
    let model = TLS.with(|cell| -> Result<Arc<WorldModel>> {
        let mut guard = cell.borrow_mut();
        let reload = match guard.as_ref() {
            None => true,
            Some((path, _)) => path.as_path() != checkpoint,
        };
        if reload {
            let (model, _varmap) = load_model(train_cfg, checkpoint, device)?;
            *guard = Some((checkpoint.to_path_buf(), Arc::new(model)));
        }
        Ok(guard
            .as_ref()
            .expect("model loaded")
            .1
            .clone())
    })?;
    f(model.as_ref())
}

fn collect_synthetic(seed: u64, episodes: usize, kinds: &[&str]) -> Result<Vec<TransitionSample>> {
    let jobs: Vec<(usize, usize, &str)> = kinds
        .iter()
        .enumerate()
        .flat_map(|(kind_index, kind)| (0..episodes).map(move |ep| (kind_index, ep, *kind)))
        .collect();
    let mut parts: Vec<(usize, Vec<TransitionSample>)> = jobs
        .par_iter()
        .enumerate()
        .map(|(job_idx, &(kind_index, ep, kind))| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            let samples = generate_curriculum(kind, seed, episode_id, Split::HeldOutComposition)?;
            Ok((job_idx, samples))
        })
        .collect::<Result<Vec<_>>>()?;
    parts.sort_by_key(|(job_idx, _)| *job_idx);
    Ok(parts.into_iter().flat_map(|(_, samples)| samples).collect())
}

fn collect_hazard_samples(seed: u64, episodes: usize) -> Result<Vec<TransitionSample>> {
    if episodes == 0 {
        return Ok(Vec::new());
    }
    let mut parts: Vec<(usize, Vec<TransitionSample>)> = (0..episodes)
        .into_par_iter()
        .map(|ep| {
            generate_hazard_one_step(
                seed.wrapping_add(0xFA17),
                ep as u64,
                Split::HeldOutComposition,
                4,
            )
            .map(|samples| (ep, samples))
        })
        .collect::<Result<_>>()?;
    parts.sort_by_key(|(ep, _)| *ep);
    Ok(parts.into_iter().flat_map(|(_, samples)| samples).collect())
}

fn collect_dynamics_rollout_samples(seed: u64, episodes: usize) -> Result<Vec<TransitionSample>> {
    let jobs: Vec<(usize, usize, &str)> = ["random_one_step", "exploration"]
        .iter()
        .enumerate()
        .flat_map(|(kind_index, kind)| (0..episodes).map(move |ep| (kind_index, ep, *kind)))
        .collect();
    let mut parts: Vec<(usize, Vec<TransitionSample>)> = jobs
        .par_iter()
        .enumerate()
        .map(|(job_idx, &(kind_index, ep, kind))| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            let samples = generate_curriculum(kind, seed, episode_id, Split::HeldOutComposition)?;
            Ok((job_idx, samples))
        })
        .collect::<Result<Vec<_>>>()?;
    parts.sort_by_key(|(job_idx, _)| *job_idx);
    Ok(parts.into_iter().flat_map(|(_, samples)| samples).collect())
}

fn collect_planner_rollout_samples(seed: u64, episodes: usize) -> Result<Vec<TransitionSample>> {
    let jobs: Vec<(usize, usize, &str)> = ["sequential", "p1c_hard_retarget"]
        .iter()
        .enumerate()
        .flat_map(|(kind_index, kind)| (0..episodes).map(move |ep| (kind_index, ep, *kind)))
        .collect();
    let mut parts: Vec<(usize, Vec<TransitionSample>)> = jobs
        .par_iter()
        .enumerate()
        .map(|(job_idx, &(kind_index, ep, kind))| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            let samples = generate_curriculum(kind, seed, episode_id, Split::HeldOutComposition)?;
            Ok((job_idx, samples))
        })
        .collect::<Result<Vec<_>>>()?;
    parts.sort_by_key(|(job_idx, _)| *job_idx);
    Ok(parts.into_iter().flat_map(|(_, samples)| samples).collect())
}

fn batch_ranges(len: usize, batch: usize) -> Vec<(usize, usize)> {
    if len == 0 || batch == 0 {
        return Vec::new();
    }
    (0..len)
        .step_by(batch)
        .map(|start| (start, (start + batch).min(len)))
        .collect()
}

fn per_sample_mse(pred: &Tensor, target: &Tensor) -> Result<Vec<f32>> {
    let mse = latent_mse_per_sample(pred, target)?;
    mse.squeeze(1)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Into::into)
}

fn mean(xs: &[f32]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().map(|v| *v as f64).sum::<f64>() / xs.len() as f64)
}

fn percentile(sorted: &[f32], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)] as f64)
}

fn median(xs: &[f32]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&sorted, 0.5)
}

fn trimmed_mean(xs: &[f32], trim_frac: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let trim = ((sorted.len() as f64 * trim_frac).floor() as usize).min(sorted.len() / 2);
    let slice = &sorted[trim..sorted.len().saturating_sub(trim)];
    if slice.is_empty() {
        mean(xs)
    } else {
        mean(slice)
    }
}

fn bootstrap_ci95(values: &[f32], seed: u64) -> (Option<f64>, Option<f64>) {
    if values.len() < 2 {
        let m = mean(values);
        return (m, m);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_boot = 400usize;
    let mut means = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut sum = 0f64;
        for _ in 0..values.len() {
            let idx = rng.random_range(0..values.len());
            sum += values[idx] as f64;
        }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[(0.025 * means.len() as f64) as usize];
    let hi = means[((0.975 * means.len() as f64) as usize).min(means.len() - 1)];
    (Some(lo), Some(hi))
}

fn summarize_horizon(values: &[f32], normalized: &[f32], seed: u64) -> HorizonRolloutStats {
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mut sorted = finite.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (ci95_low, ci95_high) = bootstrap_ci95(&finite, seed);
    HorizonRolloutStats {
        n: values.len(),
        finite_n: finite.len(),
        mean: mean(&finite),
        median: median(&finite),
        trimmed_mean: trimmed_mean(&finite, 0.1),
        p90: percentile(&sorted, 0.9),
        p95: percentile(&sorted, 0.95),
        ci95_low,
        ci95_high,
        max: sorted.last().map(|v| *v as f64),
        normalized_mean: mean(normalized),
    }
}

fn center_columns(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let cols = matrix[0].len();
    let mut means = vec![0f64; cols];
    for row in matrix {
        for (j, value) in row.iter().enumerate() {
            means[j] += *value as f64;
        }
    }
    for mean in &mut means {
        *mean /= matrix.len() as f64;
    }
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, value)| (*value as f64 - means[j]) as f32)
                .collect()
        })
        .collect()
}

fn flatten_matrix(matrix: &[Vec<f32>]) -> (Vec<f32>, usize, usize) {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(rows * cols);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    (flat, rows, cols)
}

fn mat_transpose_mul(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f64; cols * cols];
    for i in 0..rows {
        for c in 0..cols {
            let av = a[i * cols + c] as f64;
            for d in 0..cols {
                out[c * cols + d] += av * a[i * cols + d] as f64;
            }
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

fn mat_transpose_mul_rhs(a: &[f32], rows: usize, cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
    let mut out = vec![0f64; cols * b_cols];
    for i in 0..rows {
        for c in 0..cols {
            let av = a[i * cols + c] as f64;
            for d in 0..b_cols {
                out[c * b_cols + d] += av * b[i * b_cols + d] as f64;
            }
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

fn mat_mul(a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let av = a[i * a_cols + k];
            for j in 0..b_cols {
                out[i * b_cols + j] += av * b[k * b_cols + j];
            }
        }
    }
    out
}

fn solve_linear(mut a: Vec<f32>, n: usize, b: &mut [f32], nrhs: usize) -> bool {
    const EPS: f64 = 1e-12;
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let value = a[row * n + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < EPS as f32 {
            return false;
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            for rhs in 0..nrhs {
                b.swap(col * nrhs + rhs, pivot * nrhs + rhs);
            }
        }
        let pivot_val = a[col * n + col];
        for j in col..n {
            a[col * n + j] /= pivot_val;
        }
        for rhs in 0..nrhs {
            b[col * nrhs + rhs] /= pivot_val;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            for rhs in 0..nrhs {
                b[row * nrhs + rhs] -= factor * b[col * nrhs + rhs];
            }
        }
    }
    true
}

fn linear_r2_with_weights(
    encoder: &[Vec<f32>],
    oracle: &[Vec<f32>],
    weights: &[f32],
    h_dim: usize,
    z_dim: usize,
) -> Option<f64> {
    if encoder.is_empty() || encoder.len() != oracle.len() {
        return None;
    }
    let h = center_columns(encoder);
    let z = center_columns(oracle);
    let (h_flat, n, _) = flatten_matrix(&h);
    let (z_flat, _, _) = flatten_matrix(&z);
    let pred = mat_mul(&h_flat, n, h_dim, weights, z_dim);
    let mut ss_res = 0f64;
    let mut ss_tot = 0f64;
    for i in 0..n {
        for j in 0..z_dim {
            let target = z_flat[i * z_dim + j] as f64;
            let estimate = pred[i * z_dim + j] as f64;
            let err = target - estimate;
            ss_res += err * err;
            ss_tot += target * target;
        }
    }
    if ss_tot <= f64::EPSILON {
        return Some(1.0);
    }
    Some((1.0 - ss_res / ss_tot).clamp(-f64::INFINITY, 1.0))
}

fn fit_ridge_h_to_z(encoder: &[Vec<f32>], oracle: &[Vec<f32>]) -> Option<(Vec<f32>, usize, usize)> {
    if encoder.len() < 2 || encoder.len() != oracle.len() {
        return None;
    }
    let h_dim = encoder[0].len();
    let z_dim = oracle[0].len();
    if h_dim == 0 || z_dim == 0 {
        return None;
    }
    let h = center_columns(encoder);
    let z = center_columns(oracle);
    let (h_flat, n, _) = flatten_matrix(&h);
    let (z_flat, _, _) = flatten_matrix(&z);
    let mut gram = mat_transpose_mul(&h_flat, n, h_dim);
    const RIDGE: f32 = 1e-2;
    for i in 0..h_dim {
        gram[i * h_dim + i] += RIDGE;
    }
    let rhs = mat_transpose_mul_rhs(&h_flat, n, h_dim, &z_flat, z_dim);
    let mut weights = rhs;
    if !solve_linear(gram, h_dim, &mut weights, z_dim) {
        return None;
    }
    Some((weights, h_dim, z_dim))
}

fn project_encoder_delta(weights: &[f32], h_dim: usize, z_dim: usize, dh: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; z_dim];
    for j in 0..z_dim {
        let mut sum = 0f32;
        for i in 0..h_dim {
            sum += dh[i] * weights[i * z_dim + j];
        }
        out[j] = sum;
    }
    out
}

fn latent_covariance_frobenius(encoder: &[Vec<f32>]) -> Option<f64> {
    if encoder.len() < 2 {
        return None;
    }
    let dim = encoder[0].len();
    if dim == 0 {
        return None;
    }
    let centered = center_columns(encoder);
    let n = centered.len() as f64;
    let mut cov = vec![0f64; dim * dim];
    for row in &centered {
        for i in 0..dim {
            let vi = row[i] as f64;
            for j in 0..dim {
                cov[i * dim + j] += vi * row[j] as f64;
            }
        }
    }
    let denom = (n - 1.0).max(1.0);
    let mut err = 0f64;
    for i in 0..dim {
        for j in 0..dim {
            let value = cov[i * dim + j] / denom;
            let target = if i == j { 1.0 } else { 0.0 };
            let delta = value - target;
            err += delta * delta;
        }
    }
    Some(err.sqrt())
}

fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f64::EPSILON || nb <= f64::EPSILON {
        return None;
    }
    Some(dot / na.sqrt() / nb.sqrt())
}

struct IdentifiabilityPairState {
    key: (u64, u64),
    encoder: Vec<f32>,
    oracle: Vec<f32>,
}

fn eval_identifiability(
    samples: &[TransitionSample],
    encoders: &[Option<Vec<f32>>],
) -> Option<IdentifiabilityMetrics> {
    if samples.len() != encoders.len() {
        return None;
    }
    let latent_dim = encoders.iter().find_map(|row| row.as_ref().map(|h| h.len()))?;
    let mut labeled_h = Vec::new();
    let mut labeled_z = Vec::new();
    for (sample, encoder) in samples.iter().zip(encoders.iter()) {
        let Some(encoder) = encoder.as_ref() else {
            continue;
        };
        let Some(oracle) = sample.oracle_latent.as_ref() else {
            continue;
        };
        if oracle.len() != ORACLE_LATENT_DIM || encoder.len() != latent_dim {
            return None;
        }
        labeled_h.push(encoder.clone());
        labeled_z.push(oracle.clone());
    }
    if labeled_h.is_empty() {
        return None;
    }

    let n = labeled_h.len();
    let split = ((n as f64) * 0.8).floor() as usize;
    let split = split.clamp(1, n.saturating_sub(1));
    let (train_h, val_h) = labeled_h.split_at(split);
    let (train_z, val_z) = labeled_z.split_at(split);
    let bridge = fit_ridge_h_to_z(train_h, train_z);
    let (r2_h_to_z, r2_h_to_z_train, bridge_w, h_dim, z_dim) = match bridge {
        Some((w, hd, zd)) => (
            linear_r2_with_weights(val_h, val_z, &w, hd, zd),
            linear_r2_with_weights(train_h, train_z, &w, hd, zd),
            Some(w),
            hd,
            zd,
        ),
        None => (None, None, None, latent_dim, ORACLE_LATENT_DIM),
    };

    let mut encoder_pair_mse = Vec::new();
    let mut oracle_pair_mse = Vec::new();
    let mut pair_cosine = Vec::new();
    let mut prev: Option<IdentifiabilityPairState> = None;
    for (sample, encoder) in samples.iter().zip(encoders.iter()) {
        let encoder = match encoder {
            Some(value) => value,
            None => {
                prev = None;
                continue;
            }
        };
        let oracle = match sample.oracle_latent.as_ref() {
            Some(value) => value,
            None => {
                prev = None;
                continue;
            }
        };
        let key = (sample.seed, sample.episode_id);
        if let Some(prev_state) = prev.as_ref() {
            if prev_state.key == key {
                let dh: Vec<f32> = encoder
                    .iter()
                    .zip(prev_state.encoder.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let dz: Vec<f32> = oracle
                    .iter()
                    .zip(prev_state.oracle.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                encoder_pair_mse.push(l2_sq(encoder, &prev_state.encoder) as f32);
                oracle_pair_mse.push(l2_sq(oracle, &prev_state.oracle) as f32);
                if let Some(w) = bridge_w.as_ref() {
                    let projected = project_encoder_delta(w, h_dim, z_dim, &dh);
                    if let Some(cos) = cosine_similarity(&projected, &dz) {
                        pair_cosine.push(cos as f32);
                    }
                }
            }
        }
        prev = Some(IdentifiabilityPairState {
            key,
            encoder: encoder.clone(),
            oracle: oracle.clone(),
        });
    }

    Some(IdentifiabilityMetrics {
        n_labeled: labeled_h.len(),
        oracle_dim: ORACLE_LATENT_DIM,
        latent_dim,
        r2_h_to_z,
        r2_h_to_z_train,
        latent_covariance_frobenius: latent_covariance_frobenius(&labeled_h),
        mean_encoder_pair_mse: mean(&encoder_pair_mse),
        mean_oracle_pair_mse: mean(&oracle_pair_mse),
        pair_increment_cosine: mean(&pair_cosine),
    })
}

fn eval_events(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<(usize, Option<f64>, Option<f64>)> {
    let logits = logits.to_dtype(DType::F32)?;
    let targets = targets.to_dtype(DType::F32)?;
    let mask = mask.to_dtype(DType::F32)?;
    let (b, e) = logits.dims2()?;
    let logit_v = logits.flatten_all()?.to_vec1::<f32>()?;
    let tgt_v = targets.flatten_all()?.to_vec1::<f32>()?;
    let mask_v = mask.flatten_all()?.to_vec1::<f32>()?;
    let mut labeled = 0usize;
    let mut correct = 0usize;
    let mut bce_sum = 0f64;
    for i in 0..b * e {
        if mask_v[i] <= 0.0 {
            continue;
        }
        labeled += 1;
        let p = 1.0 / (1.0 + (-logit_v[i]).exp());
        let y = tgt_v[i];
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
        let p = p.clamp(1e-7, 1.0 - 1e-7);
        bce_sum += -(y as f64 * (p as f64).ln() + (1.0 - y as f64) * (1.0 - p as f64).ln());
    }
    if labeled == 0 {
        Ok((0, None, None))
    } else {
        Ok((
            labeled,
            Some(correct as f64 / labeled as f64),
            Some(bce_sum / labeled as f64),
        ))
    }
}

fn eval_q(q_logit: &Tensor, mse: &[f32], threshold: f64) -> Result<QEvalAccum> {
    let probs = ops::sigmoid(q_logit)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if probs.len() != mse.len() {
        bail!("Q batch {} != mse {}", probs.len(), mse.len());
    }
    let mut out = QEvalAccum {
        n: probs.len(),
        ..Default::default()
    };
    for (p, m) in probs.iter().zip(mse.iter()) {
        let y = if (*m as f64) < threshold { 1.0 } else { 0.0 };
        let p = *p as f64;
        out.brier_sum += (p - y).powi(2);
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            out.correct += 1;
        }
        if y >= 0.5 {
            out.positive_labels += 1;
        }
        match (pred, y) {
            (1.0, 1.0) => out.tp += 1,
            (0.0, 0.0) => out.tn += 1,
            (1.0, 0.0) => out.fp += 1,
            (0.0, 1.0) => out.fn_ += 1,
            _ => {}
        }
    }
    Ok(out)
}

fn ptrm_metrics_for_k(
    trajectories: &[crate::p2::model::PtrmTrajectory],
    best_indices: &[usize],
    next_z: &Tensor,
    k: usize,
    noise: f64,
    threshold: f64,
) -> Result<PtrmKMetrics> {
    let b = next_z.dim(0)?;
    let mut pass = 0usize;
    let mut best_q_pass = 0usize;
    let mut disagree_acc = 0f64;
    let mut oracle_rank = 0usize;
    for sample in 0..b {
        let mut any_pass = false;
        let mut ys = Vec::with_capacity(k);
        let mut mses = Vec::with_capacity(k);
        for traj in trajectories.iter().take(k) {
            let y = traj.y.get(sample)?;
            let t = next_z.get(sample)?;
            let mse = y.sub(&t)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
            mses.push(mse);
            if mse < threshold {
                any_pass = true;
            }
            ys.push(y.flatten_all()?.to_vec1::<f32>()?);
        }
        if any_pass {
            pass += 1;
        }
        let best = best_indices[sample];
        if mses.get(best).copied().unwrap_or(f64::INFINITY) < threshold {
            best_q_pass += 1;
        }
        if k >= 2 {
            let oracle = mses
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if best == oracle {
                oracle_rank += 1;
            }
        }
        let mut dsum = 0f64;
        let mut dpairs = 0usize;
        for i in 0..k {
            for j in (i + 1)..k {
                let mut s = 0f64;
                for (a, b) in ys[i].iter().zip(ys[j].iter()) {
                    let d = (*a - *b) as f64;
                    s += d * d;
                }
                dsum += s.sqrt();
                dpairs += 1;
            }
        }
        if dpairs > 0 {
            disagree_acc += dsum / dpairs as f64;
        }
    }
    let pass_at_k = pass as f64 / b as f64;
    let best_q_at_k = best_q_pass as f64 / b as f64;
    Ok(PtrmKMetrics {
        k,
        noise,
        n: b,
        pass_at_k,
        best_q_at_k,
        ranking_gap: pass_at_k - best_q_at_k,
        q_oracle_rank_accuracy: (k >= 2).then_some(oracle_rank as f64 / b as f64),
        disagreement: disagree_acc / b as f64,
        trajectory_evaluations_per_transition: k,
    })
}

fn ptrm_metrics(
    model: &WorldModel,
    batch: &BatchTensors,
    next_z: &Tensor,
    ks: &[usize],
    noise: f64,
    threshold: f64,
    seed: u64,
) -> Result<Vec<PtrmKMetrics>> {
    if ks.is_empty() {
        return Ok(Vec::new());
    }
    let max_k = *ks.iter().max().unwrap_or(&1);
    let effective_noise = if max_k == 1 { 0.0 } else { noise };
    let ptrm = model.forward_ptrm(
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        PtrmConfig {
            k: max_k,
            sigma: effective_noise,
            seed: Some(seed),
        },
    )?;
    ks.iter()
        .filter(|&&k| k > 0)
        .map(|&k| {
            ptrm_metrics_for_k(
                &ptrm.trajectories,
                &ptrm.best_indices,
                next_z,
                k,
                if k == 1 { 0.0 } else { effective_noise },
                threshold,
            )
        })
        .collect()
}

#[derive(Default)]
struct BatchEvalPartial {
    mse_all: Vec<f32>,
    encoder_embeddings: Vec<Option<Vec<f32>>>,
    event_labeled: usize,
    event_correct_weighted: f64,
    event_bce_weighted: f64,
    hazard_failure_labeled: usize,
    hazard_false_negatives: usize,
    q_acc: QEvalAccum,
    q_probs: Vec<f32>,
    reliability_probs: Vec<f32>,
    recursion_probes: Vec<RecursionStepProbe>,
    ptrm_acc: BTreeMap<usize, (f64, f64, f64, f64, usize)>,
    matched_acc: BTreeMap<usize, (f64, usize, usize)>,
    ensemble_disagreement: f64,
    ensemble_n: usize,
}

fn eval_one_batch(
    model: &WorldModel,
    chunk: &[TransitionSample],
    bi: usize,
    cfg: &EvalConfig,
    device: &Device,
) -> Result<BatchEvalPartial> {
    if chunk.is_empty() {
        return Ok(BatchEvalPartial::default());
    }
    let batch = batch_from_samples(chunk, device)?;
    let out = model.forward(
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
    )?;
    let current_z = model.encode_state(&batch.frames)?;
    let current_vecs = flatten_latent(&current_z)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut partial = BatchEvalPartial {
        encoder_embeddings: chunk
            .iter()
            .zip(current_vecs.iter())
            .map(|(sample, vec)| {
                if sample.oracle_latent.is_some() {
                    Some(vec.clone())
                } else {
                    None
                }
            })
            .collect(),
        ..Default::default()
    };
    let next_z = model.encode_state(&batch.next_frames)?;
    let mses = per_sample_mse(&out.y, &next_z)?;
    partial.mse_all.extend(mses.iter().copied());

    let (lab, acc, bce) = eval_events(&out.event_logits, &batch.event_targets, &batch.event_mask)?;
    if lab > 0 {
        partial.event_labeled = lab;
        partial.event_correct_weighted = acc.unwrap_or(0.0) * lab as f64;
        partial.event_bce_weighted = bce.unwrap_or(0.0) * lab as f64;
    }
    let event_logits = out.event_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    for (sample, logits) in chunk.iter().zip(event_logits.iter()) {
        if sample.family == "avoid_hazard_reach_marker" && sample.goal_failed == Some(true) {
            partial.hazard_failure_labeled += 1;
            if logits[EVENT_GOAL_FAILED] < 0.0 {
                partial.hazard_false_negatives += 1;
            }
        }
    }

    partial.q_acc = eval_q(&out.q_logit, &mses, cfg.q_mse_threshold)?;
    partial.q_probs = candle_nn::ops::sigmoid(&out.q_logit.to_dtype(DType::F32)?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    partial.reliability_probs =
        candle_nn::ops::sigmoid(&out.reliability_logit.to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
    partial.recursion_probes.extend(out.recursion_probes);

    if cfg.ensemble_members >= 2 {
        let ptrm = model.forward_ptrm(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            PtrmConfig {
                k: cfg.ensemble_members,
                sigma: cfg.ptrm_noise.max(0.01),
                seed: Some(cfg.seed.wrapping_add(bi as u64).wrapping_add(0xE11A)),
            },
        )?;
        let b = next_z.dim(0)?;
        for sample in 0..b {
            let mut ys = Vec::with_capacity(cfg.ensemble_members);
            for traj in &ptrm.trajectories {
                ys.push(
                    traj.y
                        .get(sample)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
            }
            let mut dsum = 0f64;
            let mut dpairs = 0usize;
            for i in 0..cfg.ensemble_members {
                for j in (i + 1)..cfg.ensemble_members {
                    let mut s = 0f64;
                    for (a, b) in ys[i].iter().zip(ys[j].iter()) {
                        let d = (*a - *b) as f64;
                        s += d * d;
                    }
                    dsum += s.sqrt();
                    dpairs += 1;
                }
            }
            if dpairs > 0 {
                partial.ensemble_disagreement += dsum / dpairs as f64;
                partial.ensemble_n += 1;
            }
        }
    }

    for &k in &cfg.ptrm_k {
        let outer_steps = model
            .config()
            .outer_steps
            .checked_mul(k)
            .ok_or_else(|| anyhow::anyhow!("matched-compute outer_steps overflow"))?;
        let deterministic = model.forward_with_outer_steps(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            outer_steps,
        )?;
        let values = per_sample_mse(&deterministic.y, &next_z)?;
        let entry = partial.matched_acc.entry(k).or_insert((0.0, 0, outer_steps));
        entry.0 += values.iter().map(|value| f64::from(*value)).sum::<f64>();
        entry.1 += values.len();
    }

    let ptrm = ptrm_metrics(
        model,
        &batch,
        &next_z,
        &cfg.ptrm_k,
        cfg.ptrm_noise,
        cfg.q_mse_threshold,
        cfg.seed.wrapping_add(bi as u64),
    )?;
    for m in ptrm {
        let e = partial
            .ptrm_acc
            .entry(m.k)
            .or_insert((0.0, 0.0, 0.0, 0.0, 0));
        e.0 += m.pass_at_k * m.n as f64;
        e.1 += m.best_q_at_k * m.n as f64;
        e.2 += m.disagreement * m.n as f64;
        e.3 += m.q_oracle_rank_accuracy.unwrap_or(0.0) * m.n as f64;
        e.4 += m.n;
    }
    Ok(partial)
}

fn merge_batch_partials(mut partials: Vec<(usize, BatchEvalPartial)>) -> BatchEvalPartial {
    partials.sort_by_key(|(bi, _)| *bi);
    let mut merged = BatchEvalPartial::default();
    for (_, partial) in partials {
        merged.mse_all.extend(partial.mse_all);
        merged.encoder_embeddings.extend(partial.encoder_embeddings);
        merged.event_labeled += partial.event_labeled;
        merged.event_correct_weighted += partial.event_correct_weighted;
        merged.event_bce_weighted += partial.event_bce_weighted;
        merged.hazard_failure_labeled += partial.hazard_failure_labeled;
        merged.hazard_false_negatives += partial.hazard_false_negatives;
        merged.q_acc.merge(partial.q_acc);
        merged.q_probs.extend(partial.q_probs);
        merged.reliability_probs.extend(partial.reliability_probs);
        merged.recursion_probes.extend(partial.recursion_probes);
        merged.ensemble_disagreement += partial.ensemble_disagreement;
        merged.ensemble_n += partial.ensemble_n;
        for (k, (p, bq, d, oracle, n)) in partial.ptrm_acc {
            let e = merged.ptrm_acc.entry(k).or_insert((0.0, 0.0, 0.0, 0.0, 0));
            e.0 += p;
            e.1 += bq;
            e.2 += d;
            e.3 += oracle;
            e.4 += n;
        }
        for (k, (sum, n, outer_steps)) in partial.matched_acc {
            let e = merged.matched_acc.entry(k).or_insert((0.0, 0, outer_steps));
            e.0 += sum;
            e.1 += n;
        }
    }
    merged
}

fn eval_rollout_group(
    model: &WorldModel,
    steps: &[TransitionSample],
    device: &Device,
    closed_loop: bool,
) -> Result<(Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)> {
    if steps.len() < 4 {
        return Ok((None, None, None, None, None, None));
    }
    let mut mse4 = None;
    let mut mse8 = None;
    let mut mse16 = None;
    let mut cf4 = None;
    let mut cf8 = None;
    let mut cf16 = None;
    let z0 = {
        let first = batch_from_samples(std::slice::from_ref(&steps[0]), device)?;
        model.encode_state(&first.frames)?
    };
    let mut latent = z0.clone();
    for (idx, sample) in steps.iter().enumerate() {
        let batch = batch_from_samples(std::slice::from_ref(sample), device)?;
        if closed_loop {
            latent = model.encode_state(&batch.frames)?;
        }
        let out = model.forward_from_latent(
            &latent,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
        )?;
        let pred = out.y;
        if !closed_loop {
            latent = pred.clone();
        }
        let target = model.encode_state(&batch.next_frames)?;
        let mse = pred
            .sub(&target)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f32;
        let cf = z0
            .sub(&target)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()? as f32;
        let step_no = idx + 1;
        if step_no == 4 {
            mse4 = Some(mse);
            cf4 = Some(cf);
        }
        if step_no == 8 {
            mse8 = Some(mse);
            cf8 = Some(cf);
        }
        if step_no == 16 {
            mse16 = Some(mse);
            cf16 = Some(cf);
        }
    }
    Ok((mse4, mse8, mse16, cf4, cf8, cf16))
}

fn normalized_ratio(model: &[f32], baseline: &[f32]) -> Vec<f32> {
    model
        .iter()
        .zip(baseline)
        .map(|(m, b)| {
            if b.is_finite() && *b > 1e-8 {
                m / b
            } else {
                f32::NAN
            }
        })
        .collect()
}

fn merge_rollout_sextuples(
    sextuples: &[(Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)],
    seed: u64,
) -> RolloutMetrics {
    let mut mse4 = Vec::new();
    let mut mse8 = Vec::new();
    let mut mse16 = Vec::new();
    let mut cf4 = Vec::new();
    let mut cf8 = Vec::new();
    let mut cf16 = Vec::new();
    for (m4, m8, m16, c4, c8, c16) in sextuples {
        if let Some(v) = m4 {
            mse4.push(*v);
        }
        if let Some(v) = m8 {
            mse8.push(*v);
        }
        if let Some(v) = m16 {
            mse16.push(*v);
        }
        if let Some(v) = c4 {
            cf4.push(*v);
        }
        if let Some(v) = c8 {
            cf8.push(*v);
        }
        if let Some(v) = c16 {
            cf16.push(*v);
        }
    }
    RolloutMetrics {
        n4: mse4.len(),
        mse_4: mean(&mse4),
        n8: mse8.len(),
        mse_8: mean(&mse8),
        n16: mse16.len(),
        mse_16: mean(&mse16),
        h4: Some(summarize_horizon(
            &mse4,
            &normalized_ratio(&mse4, &cf4),
            seed ^ 0x04,
        )),
        h8: Some(summarize_horizon(
            &mse8,
            &normalized_ratio(&mse8, &cf8),
            seed ^ 0x08,
        )),
        h16: Some(summarize_horizon(
            &mse16,
            &normalized_ratio(&mse16, &cf16),
            seed ^ 0x10,
        )),
        open_closed_ratio_8: None,
    }
}

fn merge_copy_forward_sextuples(
    sextuples: &[(Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)],
    seed: u64,
) -> RolloutMetrics {
    let mut cf4 = Vec::new();
    let mut cf8 = Vec::new();
    let mut cf16 = Vec::new();
    for (_, _, _, c4, c8, c16) in sextuples {
        if let Some(v) = c4 {
            cf4.push(*v);
        }
        if let Some(v) = c8 {
            cf8.push(*v);
        }
        if let Some(v) = c16 {
            cf16.push(*v);
        }
    }
    RolloutMetrics {
        n4: cf4.len(),
        mse_4: mean(&cf4),
        n8: cf8.len(),
        mse_8: mean(&cf8),
        n16: cf16.len(),
        mse_16: mean(&cf16),
        h4: Some(summarize_horizon(&cf4, &[], seed ^ 0x14)),
        h8: Some(summarize_horizon(&cf8, &[], seed ^ 0x18)),
        h16: Some(summarize_horizon(&cf16, &[], seed ^ 0x1C)),
        open_closed_ratio_8: None,
    }
}

fn eval_q_surprise(q_probs: &[f32], mses: &[f32], threshold: f64) -> QSurpriseMetrics {
    let n = q_probs.len().min(mses.len());
    if n == 0 {
        return QSurpriseMetrics {
            n: 0,
            mean_q_when_unreliable: None,
            mean_q_when_reliable: None,
            confident_error_rate: None,
        };
    }
    let thr = threshold as f32;
    let mut unreliable_q = Vec::new();
    let mut reliable_q = Vec::new();
    let mut confident_errors = 0usize;
    let mut unreliable_count = 0usize;
    for i in 0..n {
        let q = q_probs[i];
        let mse = mses[i];
        if mse > thr {
            unreliable_q.push(q);
            unreliable_count += 1;
            if q > 0.5 {
                confident_errors += 1;
            }
        } else {
            reliable_q.push(q);
        }
    }
    QSurpriseMetrics {
        n,
        mean_q_when_unreliable: mean(&unreliable_q),
        mean_q_when_reliable: mean(&reliable_q),
        confident_error_rate: (unreliable_count > 0)
            .then_some(confident_errors as f64 / unreliable_count as f64),
    }
}

/// Group by stable episode identity. `family` can change during retarget traces.
fn group_rollouts(samples: &[TransitionSample]) -> BTreeMap<(u64, u64), Vec<&TransitionSample>> {
    let mut map: BTreeMap<(u64, u64), Vec<&TransitionSample>> = BTreeMap::new();
    for s in samples {
        map.entry((s.seed, s.episode_id)).or_default().push(s);
    }
    for steps in map.values_mut() {
        steps.sort_by_key(|s| s.transition_index);
    }
    map
}

fn eval_rollouts(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    closed_loop: bool,
    seed: u64,
) -> Result<RolloutMetrics> {
    let groups: Vec<Vec<TransitionSample>> = group_rollouts(samples)
        .into_values()
        .filter(|steps| steps.len() >= 4)
        .map(|steps| steps.iter().map(|s| (*s).clone()).collect())
        .collect();
    let sextuples: Vec<(Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)> = if device.is_cpu() {
        groups
            .into_par_iter()
            .map(|steps| {
                with_thread_local_model(train_cfg, checkpoint, device, |m| {
                    eval_rollout_group(m, &steps, device, closed_loop)
                })
            })
            .collect::<Result<_>>()?
    } else {
        groups
            .iter()
            .map(|steps| eval_rollout_group(model, steps, device, closed_loop))
            .collect::<Result<_>>()?
    };
    Ok(merge_rollout_sextuples(
        &sextuples,
        seed ^ if closed_loop { 0xC1 } else { 0x01 },
    ))
}

pub fn eval_copy_forward_rollouts(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    seed: u64,
) -> Result<RolloutMetrics> {
    let groups: Vec<Vec<TransitionSample>> = group_rollouts(samples)
        .into_values()
        .filter(|steps| steps.len() >= 4)
        .map(|steps| steps.iter().map(|s| (*s).clone()).collect())
        .collect();
    let sextuples: Vec<(Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)> = if device.is_cpu() {
        groups
            .into_par_iter()
            .map(|steps| {
                with_thread_local_model(train_cfg, checkpoint, device, |m| {
                    eval_rollout_group(m, &steps, device, false)
                })
            })
            .collect::<Result<_>>()?
    } else {
        groups
            .iter()
            .map(|steps| eval_rollout_group(model, steps, device, false))
            .collect::<Result<_>>()?
    };
    Ok(merge_copy_forward_sextuples(&sextuples, seed ^ 0xCF))
}

#[allow(clippy::too_many_arguments)]
fn eval_sample_set(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    source: &str,
    cfg: &EvalConfig,
    device: &Device,
    with_rollout: bool,
) -> Result<SplitEval> {
    if samples.is_empty() {
        return Ok(SplitEval {
            source: source.into(),
            n_samples: 0,
            one_step_latent_mse: None,
            identifiability: None,
            events: EventMetrics {
                labeled: 0,
                accuracy: None,
                bce: None,
                hazard_failure_labeled: 0,
                hazard_false_negatives: 0,
                hazard_false_negative_rate: None,
            },
            q: QMetrics {
                n: 0,
                brier: None,
                accuracy: None,
                positive_label_rate: None,
                balanced_accuracy: None,
                saturated: false,
            },
            ptrm: Vec::new(),
            deterministic_matched_compute: Vec::new(),
            rollout: None,
            closed_loop: None,
            copy_forward: None,
            q_surprise: None,
            calibration: None,
            calibration_gates: None,
            contrastive: None,
            recursion_probes: None,
            ensemble: None,
        });
    }

    let batch_size = cfg.physical_batch.max(1);
    let ranges = batch_ranges(samples.len(), batch_size);
    let merged = if device.is_cpu() {
        let partials: Vec<(usize, BatchEvalPartial)> = ranges
            .par_iter()
            .enumerate()
            .map(|(bi, &(start, end))| {
                let chunk = &samples[start..end];
                with_thread_local_model(train_cfg, checkpoint, device, |m| {
                    eval_one_batch(m, chunk, bi, cfg, device).map(|partial| (bi, partial))
                })
            })
            .collect::<Result<_>>()?;
        merge_batch_partials(partials)
    } else {
        let partials: Vec<(usize, BatchEvalPartial)> = ranges
            .iter()
            .enumerate()
            .map(|(bi, &(start, end))| {
                let partial = eval_one_batch(model, &samples[start..end], bi, cfg, device)
                    .map(|partial| (bi, partial))?;
                if device.is_cuda() {
                    device.synchronize()?;
                }
                Ok(partial)
            })
            .collect::<Result<_>>()?;
        merge_batch_partials(partials)
    };

    let mse_all = merged.mse_all;
    let encoder_embeddings = merged.encoder_embeddings;
    let event_labeled = merged.event_labeled;
    let event_correct = merged.event_correct_weighted;
    let event_bce = merged.event_bce_weighted;
    let hazard_failure_labeled = merged.hazard_failure_labeled;
    let hazard_false_negatives = merged.hazard_false_negatives;
    let q_acc = merged.q_acc;
    let ptrm_acc = merged.ptrm_acc;
    let matched_acc = merged.matched_acc;

    let events = if event_labeled == 0 {
        EventMetrics {
            labeled: 0,
            accuracy: None,
            bce: None,
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: None,
        }
    } else {
        EventMetrics {
            labeled: event_labeled,
            accuracy: Some(event_correct / event_labeled as f64),
            bce: Some(event_bce / event_labeled as f64),
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: (hazard_failure_labeled > 0)
                .then_some(hazard_false_negatives as f64 / hazard_failure_labeled as f64),
        }
    };
    let q = q_acc.finalize();
    let ptrm = ptrm_acc
        .into_iter()
        .map(|(k, (p, bq, d, oracle, n))| {
            let pass_at_k = if n == 0 { 0.0 } else { p / n as f64 };
            let best_q_at_k = if n == 0 { 0.0 } else { bq / n as f64 };
            PtrmKMetrics {
                k,
                noise: if k == 1 { 0.0 } else { cfg.ptrm_noise },
                n,
                pass_at_k,
                best_q_at_k,
                ranking_gap: pass_at_k - best_q_at_k,
                q_oracle_rank_accuracy: (k >= 2 && n > 0).then_some(oracle / n as f64),
                disagreement: if n == 0 { 0.0 } else { d / n as f64 },
                trajectory_evaluations_per_transition: k,
            }
        })
        .collect();
    let deterministic_matched_compute = matched_acc
        .into_iter()
        .map(|(k, (sum, n, outer_steps))| MatchedComputeMetrics {
            ptrm_k_equivalent: k,
            outer_steps,
            n,
            one_step_latent_mse: (n > 0).then_some(sum / n as f64),
        })
        .collect();

    let q_surprise = eval_q_surprise(&merged.q_probs, &mse_all, cfg.q_mse_threshold);
    let labels: Vec<bool> = mse_all
        .iter()
        .map(|m| f64::from(*m) <= cfg.q_mse_threshold)
        .collect();
    let (calibration, calibration_gates) =
        if merged.reliability_probs.len() == labels.len() && !labels.is_empty() {
            let ece = expected_calibration_error(&merged.reliability_probs, &labels, 10);
            let auroc = binary_auroc(&merged.reliability_probs, &labels);
            let risk_coverage = risk_coverage_buckets(&merged.reliability_probs, &labels, 5);
            let risk_monotone = risk_coverage.windows(2).all(|w| w[1].1 >= w[0].1 - 1e-6);
            (
                Some(CalibrationMetrics {
                    n: labels.len(),
                    ece,
                    reliability_auroc: auroc,
                    risk_coverage: risk_coverage.clone(),
                }),
                Some(CalibrationGates {
                    ece_pass: ece.map(|v| v <= 0.05),
                    auroc_pass: auroc.map(|v| v >= 0.85),
                    risk_monotone_pass: Some(risk_monotone),
                }),
            )
        } else {
            (None, None)
        };
    let recursion_probes = summarize_recursion_probes(&merged.recursion_probes);
    let ensemble = (merged.ensemble_n > 0).then(|| {
        let mean_disagreement = Some(merged.ensemble_disagreement / merged.ensemble_n as f64);
        let high_error: Vec<bool> = mse_all
            .iter()
            .take(merged.ensemble_n.min(mse_all.len()))
            .map(|m| f64::from(*m) > cfg.q_mse_threshold)
            .collect();
        let uncertainty: Vec<f32> = (0..high_error.len())
            .map(|i| merged.reliability_probs.get(i).copied().unwrap_or(0.5))
            .collect();
        EnsembleMetrics {
            members: cfg.ensemble_members,
            mean_disagreement,
            uncertainty_auroc: if uncertainty.len() == high_error.len() && !high_error.is_empty() {
                binary_auroc(
                    &uncertainty.iter().map(|u| 1.0 - u).collect::<Vec<_>>(),
                    &high_error,
                )
            } else {
                None
            },
        }
    });
    let contrastive = eval_contrastive_probes(model, samples, device).ok();

    let rollout = if with_rollout {
        Some(eval_rollouts(
            train_cfg,
            checkpoint,
            model,
            samples,
            device,
            false,
            cfg.seed,
        )?)
    } else {
        None
    };
    let closed_loop = if with_rollout {
        Some(eval_rollouts(
            train_cfg,
            checkpoint,
            model,
            samples,
            device,
            true,
            cfg.seed,
        )?)
    } else {
        None
    };
    let identifiability = eval_identifiability(samples, &encoder_embeddings);

    Ok(SplitEval {
        source: source.into(),
        n_samples: samples.len(),
        one_step_latent_mse: mean(&mse_all),
        identifiability,
        events,
        q,
        ptrm,
        deterministic_matched_compute,
        rollout,
        closed_loop,
        copy_forward: None,
        q_surprise: Some(q_surprise),
        calibration,
        calibration_gates,
        contrastive,
        recursion_probes,
        ensemble,
    })
}

fn summarize_recursion_probes(probes: &[RecursionStepProbe]) -> Option<RecursionProbeSummary> {
    if probes.is_empty() {
        return None;
    }
    let n = probes.len();
    Some(RecursionProbeSummary {
        n_steps: n,
        mean_residual_norm: Some(
            probes.iter().map(|p| p.mean_residual_norm).sum::<f64>() / n as f64,
        ),
        mean_latent_norm: Some(
            probes.iter().map(|p| p.mean_latent_norm).sum::<f64>() / n as f64,
        ),
        mean_amplification: Some(
            probes.iter().map(|p| p.mean_amplification).sum::<f64>() / n as f64,
        ),
        steps: probes.to_vec(),
    })
}

fn eval_contrastive_probes(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
) -> Result<ContrastiveProbeMetrics> {
    if samples.is_empty() {
        bail!("contrastive probes need samples");
    }
    let sample = &samples[0];
    let batch = batch_from_samples(&[sample.clone()], device)?;
    let z = model.encode_state(&batch.frames)?;
    let noop = Tensor::zeros((1,), DType::U32, device)?;
    let noop_coords = Tensor::zeros((1, 2), DType::F32, device)?;
    let pred_noop = model.forward_from_latent(&z, &noop, &noop_coords, &batch.goals)?;
    let noop_mse = pred_noop
        .y
        .sub(&z)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()? as f64;
    let pred_action = model.forward_from_latent(
        &z,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
    )?;
    let next_z = model.encode_state(&batch.next_frames)?;
    let action_mse = pred_action
        .y
        .sub(&next_z)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()? as f64;
    let delta_fwd = pred_action.y.sub(&z)?;
    let delta_inv = z.sub(&pred_action.y)?;
    let num = delta_fwd
        .mul(&delta_inv)?
        .sum_all()?
        .to_scalar::<f32>()? as f64;
    let den = (delta_fwd.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt()
        * (delta_inv.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt()
        + 1e-8;
    Ok(ContrastiveProbeMetrics {
        noop_identity_mse: Some(noop_mse),
        action_effect_mse: Some(action_mse),
        inverse_action_cosine: Some(num / den),
    })
}

fn write_episode_jsonl(path: &Path, rows: &[EpisodeRolloutRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut file = fs::File::create(path)?;
    use std::io::Write;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    Ok(())
}

/// Full evaluation: synthetic held-out (+ optional ARC recordings dir).
pub fn evaluate(cfg: &EvalConfig) -> Result<EvalReport> {
    cfg.validate()?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    if (cfg.q_mse_threshold - train_cfg.q_mse_threshold).abs() > f64::EPSILON {
        bail!(
            "evaluation q_mse_threshold={} differs from frozen training threshold={}",
            cfg.q_mse_threshold,
            train_cfg.q_mse_threshold
        );
    }
    let device = resolve_device(&cfg.device)?;
    let (model, _varmap) = load_model(&train_cfg, &cfg.checkpoint, &device)?;

    let mut dynamics_samples = collect_synthetic(
        cfg.seed,
        cfg.synthetic_episodes,
        &["random_one_step", "exploration"],
    )?;
    dynamics_samples.extend(collect_hazard_samples(
        cfg.seed,
        cfg.synthetic_episodes,
    )?);
    let dynamics_rollout_samples =
        collect_dynamics_rollout_samples(cfg.seed, cfg.synthetic_episodes)?;

    let planner_samples = collect_synthetic(
        cfg.seed,
        cfg.synthetic_episodes,
        &[
            "sequential",
            "hypothesis_probe",
            "p1c_falsification",
            "p1c_hard_retarget",
        ],
    )?;
    let planner_rollout_samples =
        collect_planner_rollout_samples(cfg.seed, cfg.synthetic_episodes)?;

    let mut synthetic_dynamics = eval_sample_set(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_samples,
        "synthetic_dynamics",
        cfg,
        &device,
        false,
    )?;
    synthetic_dynamics.rollout = Some(eval_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_rollout_samples,
        &device,
        false,
        cfg.seed,
    )?);
    synthetic_dynamics.closed_loop = Some(eval_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_rollout_samples,
        &device,
        true,
        cfg.seed,
    )?);
    synthetic_dynamics.copy_forward = Some(eval_copy_forward_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_rollout_samples,
        &device,
        cfg.seed,
    )?);
    if let (Some(open), Some(closed)) = (
        synthetic_dynamics.rollout.as_mut(),
        synthetic_dynamics.closed_loop.as_ref(),
    ) {
        if let (Some(o8), Some(c8)) = (open.mse_8, closed.mse_8) {
            if c8 > 0.0 {
                open.open_closed_ratio_8 = Some(o8 / c8);
            }
        }
    }

    let mut synthetic_planner = eval_sample_set(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &planner_samples,
        "synthetic_planner",
        cfg,
        &device,
        false,
    )?;
    synthetic_planner.rollout = Some(eval_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &planner_rollout_samples,
        &device,
        false,
        cfg.seed,
    )?);
    synthetic_planner.closed_loop = Some(eval_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &planner_rollout_samples,
        &device,
        true,
        cfg.seed,
    )?);
    synthetic_planner.copy_forward = Some(eval_copy_forward_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &planner_rollout_samples,
        &device,
        cfg.seed,
    )?);
    if let (Some(open), Some(closed)) = (
        synthetic_planner.rollout.as_mut(),
        synthetic_planner.closed_loop.as_ref(),
    ) {
        if let (Some(o8), Some(c8)) = (open.mse_8, closed.mse_8) {
            if c8 > 0.0 {
                open.open_closed_ratio_8 = Some(o8 / c8);
            }
        }
    }

    let arc3_transfer = if let Some(dir) = &cfg.arc_recordings_dir {
        let samples = import_recordings_dir(dir)?;
        Some(eval_sample_set(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &samples,
            "arc3_transfer",
            cfg,
            &device,
            false,
        )?)
    } else {
        None
    };

    let arc3_recording_runs = if let Some(dir) = &cfg.arc_recordings_dir {
        let runs = summarize_recordings_dir(dir)?;
        Some(Arc3RecordingBenchmark {
            n_runs: runs.len(),
            total_actions: runs.iter().map(|r| r.actions).sum(),
            total_levels_completed: runs.iter().map(|r| r.levels_completed).sum(),
            runs,
        })
    } else {
        None
    };

    let official_scorecard = if let Some(path) = &cfg.scorecard_json {
        Some(benchmark_from_scorecard_json(path)?)
    } else {
        None
    };
    let official_rhae = official_scorecard
        .as_ref()
        .and_then(official_rhae_from_benchmark);

    let report = EvalReport {
        schema: EVAL_REPORT_SCHEMA.into(),
        seed: cfg.seed,
        checkpoint: cfg.checkpoint.clone(),
        device: cfg.device.clone(),
        q_mse_threshold: cfg.q_mse_threshold,
        ptrm_k: cfg.ptrm_k.clone(),
        ptrm_noise: cfg.ptrm_noise,
        official_rhae,
        official_scorecard,
        arc3_recording_runs,
        public_data_used_for_fitting: false,
        synthetic_dynamics,
        synthetic_planner,
        arc3_transfer,
        research_claim: false,
    };
    if let Some(jsonl) = &cfg.episode_jsonl {
        let rows = vec![
            EpisodeRolloutRow {
                seed: cfg.seed,
                episode_id: 0,
                family: "synthetic_dynamics".into(),
                horizon: 8,
                open_mse: report.synthetic_dynamics.rollout.as_ref().and_then(|r| r.mse_8),
                closed_mse: report.synthetic_dynamics.closed_loop.as_ref().and_then(|r| r.mse_8),
                copy_forward_mse: report
                    .synthetic_dynamics
                    .copy_forward
                    .as_ref()
                    .and_then(|r| r.mse_8),
            },
            EpisodeRolloutRow {
                seed: cfg.seed,
                episode_id: 1,
                family: "synthetic_planner".into(),
                horizon: 8,
                open_mse: report.synthetic_planner.rollout.as_ref().and_then(|r| r.mse_8),
                closed_mse: report.synthetic_planner.closed_loop.as_ref().and_then(|r| r.mse_8),
                copy_forward_mse: report
                    .synthetic_planner
                    .copy_forward
                    .as_ref()
                    .and_then(|r| r.mse_8),
            },
        ];
        write_episode_jsonl(jsonl, &rows)?;
    }
    write_json_atomic(&cfg.output, &report)?;
    Ok(report)
}

/// ARC-only transfer eval helper.
pub fn evaluate_arc3(cfg: &EvalConfig) -> Result<EvalReport> {
    if cfg.arc_recordings_dir.is_none() {
        bail!("p2-arc3-eval requires --arc-recordings-dir");
    }
    evaluate(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::{
        reinit_varmap_deterministic, save_checkpoint, TrainReport, TRAIN_REPORT_SCHEMA,
    };

    #[test]
    fn tiny_eval_and_report_schema() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("tofy-p2-eval-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;

        let train_cfg = TrainConfig {
            output_dir: dir.clone(),
            steps_per_lesson: 1,
            lessons: vec!["dynamics".into()],
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..TrainConfig::default()
        };

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = WorldModel::new(train_cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, train_cfg.seed)?;
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
            seed: train_cfg.seed,
            physical_batch: train_cfg.physical_batch,
            grad_accum: 1,
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            parameter_count: 1,
            device: "cpu".into(),
            lessons: vec![],
            status: crate::p2::train::TrainStatus::Completed,
            global_step: 0,
            latest_checkpoint: dir.join("checkpoints/step-000000000000"),
            resumed_from: None,
            checkpoint: dir.join("model.safetensors"),
            export_checkpoint: None,
            config_path: dir.join("config.json"),
            profile_trace: Some(dir.join("profile.jsonl")),
            research_claim: false,
        };
        save_checkpoint(&varmap, &train_cfg, &report)?;

        let eval_cfg = EvalConfig {
            checkpoint: dir.join("model.safetensors"),
            train_config: dir.join("config.json"),
            seed: 3,
            synthetic_episodes: 1,
            physical_batch: 2,
            ptrm_k: vec![1, 2],
            ptrm_noise: 0.0,
            q_mse_threshold: train_cfg.q_mse_threshold,
            device: "cpu".into(),
            arc_recordings_dir: None,
            scorecard_json: None,
            output: dir.join("eval.json"),
            episode_jsonl: None,
            ensemble_members: 4,
        };
        let eval = evaluate(&eval_cfg)?;
        assert_eq!(eval.schema, EVAL_REPORT_SCHEMA);
        assert!(eval.official_rhae.is_none());
        assert!(!eval.public_data_used_for_fitting);
        assert!(!eval.research_claim);
        assert!(eval.synthetic_dynamics.n_samples > 0);
        assert!(eval.synthetic_planner.n_samples > 0);
        assert!(eval
            .synthetic_dynamics
            .one_step_latent_mse
            .is_some_and(f64::is_finite));
        assert!(eval
            .synthetic_dynamics
            .identifiability
            .as_ref()
            .is_some_and(|m| m.n_labeled > 0));
        assert_eq!(
            eval.synthetic_dynamics
                .ptrm
                .iter()
                .find(|row| row.k == 1)
                .map(|row| row.noise),
            Some(0.0)
        );
        assert_eq!(eval.synthetic_dynamics.deterministic_matched_compute.len(), 2);
        let text = fs::read_to_string(&eval_cfg.output)?;
        let back: EvalReport = serde_json::from_str(&text)?;
        assert_eq!(back.schema, EVAL_REPORT_SCHEMA);

        let recordings = dir.join("empty-recordings");
        fs::create_dir_all(&recordings)?;
        let mut arc_cfg = eval_cfg.clone();
        arc_cfg.synthetic_episodes = 0;
        arc_cfg.arc_recordings_dir = Some(recordings);
        arc_cfg.output = dir.join("arc-eval.json");
        let arc_eval = evaluate_arc3(&arc_cfg)?;
        assert_eq!(arc_eval.synthetic_dynamics.one_step_latent_mse, None);
        assert_eq!(arc_eval.synthetic_planner.one_step_latent_mse, None);
        assert_eq!(
            arc_eval.arc3_transfer.as_ref().map(|split| split.n_samples),
            Some(0)
        );
        let _: EvalReport = serde_json::from_str(&fs::read_to_string(&arc_cfg.output)?)?;

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn identifiability_linear_r2_recovers_known_map() {
        let encoder = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![4.0, 0.0],
        ];
        let oracle = vec![
            vec![2.0, 1.0],
            vec![4.0, 1.0],
            vec![6.0, 1.0],
            vec![8.0, 1.0],
        ];
        let r2 = fit_ridge_h_to_z(&encoder, &oracle)
            .and_then(|(w, hd, zd)| linear_r2_with_weights(&encoder, &oracle, &w, hd, zd))
            .expect("r2");
        assert!(r2 > 0.99, "expected near-perfect linear recovery, got {r2}");
    }
}
