//! P2 world-model evaluation (synthetic held-out + ARC recording transfer).

use crate::domain::Split;
use crate::gpu_lock::GpuSessionGuard;
use crate::p2::arc3::{import_recordings_dir, summarize_recordings_dir, RecordingRunSummary};
use crate::p2::calibration::{binary_auroc, expected_calibration_error, risk_coverage_buckets};
use crate::p2::data::{
    generate_curriculum, generate_hazard_one_step, TransitionSample, ORACLE_LATENT_DIM,
};
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, pool_latent, PtrmConfig, RecursionStepProbe, WorldModel,
    EVENT_GOAL_FAILED,
};
use crate::p2::rhae::{
    benchmark_from_scorecard_json, official_rhae_from_benchmark, ScorecardBenchmark,
};
use crate::p2::train::{
    action_tensors_from_samples, batch_from_samples, load_train_config, load_varmap_exact,
    resolve_device, sigreg_losses_for_encoded_pair, BatchTensors, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, VarBuilder, VarMap};
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const EVAL_REPORT_SCHEMA: &str = "p2.eval_report.v9";

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
    /// `action_effect_mse / noop_identity_mse`: how much more a real action
    /// moves the latent than a no-op does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_effect_ratio: Option<f64>,
    /// Copy-forward tripwire. False when actions barely change the prediction,
    /// i.e. the model has settled on `next ~= current`. A world model that
    /// fails this cannot support planning no matter how low its one-step MSE
    /// is, so it is reported as an explicit gate rather than left to be
    /// inferred from two separate numbers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_forward_pass: Option<bool>,
}

/// Minimum `action_effect_mse / noop_identity_mse` for the copy-forward gate.
pub const COPY_FORWARD_MIN_RATIO: f64 = 2.0;

/// Shuffled actions should increase one-step prediction error by more than 10%.
/// Ratios at or below this threshold indicate action-marginalized dynamics.
pub const ACTION_SHUFFLE_MIN_RATIO: f64 = 1.1;
pub const SIGREG_BOUND: f64 = 10_000.0;
pub const SIGREG_NEAR_BOUND_FRACTION: f64 = 0.99;
/// Preregistered collapse floors for normalized pooled encoder features.
pub const ENCODER_MIN_MEAN_VARIANCE: f64 = 1e-4;
pub const ENCODER_MIN_EFFECTIVE_RANK_FRACTION: f64 = 0.10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionShuffleMetrics {
    pub n: usize,
    /// Rows whose shuffled `(id,x,y)` differs from the true conditioning.
    pub changed_conditionings: usize,
    pub changed_fraction: Option<f64>,
    /// One-step latent MSE under the transition's true action.
    pub true_action_mse: Option<f64>,
    /// One-step latent MSE after deranging action IDs and coordinates while
    /// keeping current frames, targets, and goals fixed.
    pub shuffled_action_mse: Option<f64>,
    /// `shuffled_action_mse / true_action_mse`; larger means actions matter.
    pub ratio: Option<f64>,
    pub ratio_ci95_low: Option<f64>,
    pub ratio_ci95_high: Option<f64>,
    /// False at the published action-marginalization threshold (`ratio <= 1.1`).
    pub action_conditioning_pass: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangedTransitionMetrics {
    pub n: usize,
    pub learned_mse: Option<f64>,
    pub copy_forward_mse: Option<f64>,
    /// `(copy_forward_mse - learned_mse) / copy_forward_mse`.
    pub improvement_fraction: Option<f64>,
    pub improvement_ci95_low: Option<f64>,
    pub improvement_ci95_high: Option<f64>,
    pub ten_percent_improvement_pass: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationDiagnostics {
    pub sigreg_raw: Option<f64>,
    pub sigreg_bounded: Option<f64>,
    pub sigreg_bound: f64,
    pub sigreg_near_bound: Option<bool>,
    pub encoder_rows: usize,
    pub encoder_dim: usize,
    pub mean_encoder_variance: Option<f64>,
    /// Covariance participation ratio `(tr C)^2 / tr(C^2)`.
    pub effective_rank: Option<f64>,
    pub effective_rank_fraction: Option<f64>,
    pub min_mean_variance: f64,
    pub min_effective_rank_fraction: f64,
    pub noncollapse_pass: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCoverageMetrics {
    pub n: usize,
    pub action_counts: BTreeMap<u8, usize>,
    pub action_fractions: BTreeMap<u8, f64>,
    pub noop_labeled: usize,
    pub noop_rate: Option<f64>,
    pub coordinate_actions: usize,
    pub distinct_coordinate_actions: usize,
    pub distinct_action_conditionings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSourceDiagnostics {
    pub shuffle: ActionShuffleMetrics,
    pub coverage: ActionCoverageMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDiagnostics {
    pub aggregate: ActionSourceDiagnostics,
    pub by_source: BTreeMap<String, ActionSourceDiagnostics>,
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
    /// Stable episode-rollout row schema. The evaluation report remains v9 in P0-A.
    pub schema: String,
    /// Evaluation population that supplied this factual episode.
    pub source: String,
    pub seed: u64,
    pub episode_id: u64,
    /// Sorted, deduplicated family labels from the anchor through `horizon`.
    pub families_through_horizon: Vec<String>,
    pub horizon: usize,
    pub open_mse: Option<f64>,
    pub closed_mse: Option<f64>,
    pub copy_forward_mse: Option<f64>,
    /// `open_mse / copy_forward_mse` only for finite denominators above `1e-8`.
    pub normalized_open_mse: Option<f64>,
}

#[derive(Debug, Clone)]
struct EpisodeRolloutResult {
    seed: u64,
    episode_id: u64,
    families_through_horizon: Vec<String>,
    horizon: usize,
    open_mse: Option<f64>,
    closed_mse: Option<f64>,
    copy_forward_mse: Option<f64>,
}

impl EpisodeRolloutResult {
    fn into_row(self, source: &str) -> EpisodeRolloutRow {
        let normalized_open_mse =
            self.open_mse
                .zip(self.copy_forward_mse)
                .and_then(|(numerator, denominator)| {
                    (numerator.is_finite() && denominator.is_finite() && denominator > 1e-8)
                        .then_some(numerator / denominator)
                });
        EpisodeRolloutRow {
            schema: "p2.episode_rollout.v2".into(),
            source: source.into(),
            seed: self.seed,
            episode_id: self.episode_id,
            families_through_horizon: self.families_through_horizon,
            horizon: self.horizon,
            open_mse: self.open_mse,
            closed_mse: self.closed_mse,
            copy_forward_mse: self.copy_forward_mse,
            normalized_open_mse,
        }
    }
}

fn episode_rollout_result<T: Borrow<TransitionSample>>(
    steps: &[T],
    horizon: usize,
    open_mse: f64,
    closed_mse: f64,
    copy_forward_mse: f64,
) -> EpisodeRolloutResult {
    EpisodeRolloutResult {
        seed: steps[0].borrow().seed,
        episode_id: steps[0].borrow().episode_id,
        families_through_horizon: steps[..horizon]
            .iter()
            .map(|step| step.borrow().family.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        horizon,
        open_mse: Some(open_mse),
        closed_mse: Some(closed_mse),
        copy_forward_mse: Some(copy_forward_mse),
    }
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
    pub representation: Option<RepresentationDiagnostics>,
    pub changed_transitions: Option<ChangedTransitionMetrics>,
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
    pub action_diagnostics: Option<ActionDiagnostics>,
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
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = WorldModel::new(train_cfg.model_config(), vb)?;
    load_varmap_exact(&varmap, weights)
        .with_context(|| format!("load exact model checkpoint {}", weights.display()))?;
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
        Ok(guard.as_ref().expect("model loaded").1.clone())
    })?;
    f(model.as_ref())
}

fn collect_synthetic_sources(
    seed: u64,
    episodes: usize,
    kinds: &[&str],
) -> Result<Vec<(String, Vec<TransitionSample>)>> {
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
    let mut sources = kinds
        .iter()
        .map(|kind| ((*kind).to_string(), Vec::new()))
        .collect::<Vec<_>>();
    for (job_index, samples) in parts {
        let kind_index = job_index / episodes.max(1);
        sources[kind_index].1.extend(samples);
    }
    Ok(sources)
}

fn flatten_sources(sources: &[(String, Vec<TransitionSample>)]) -> Vec<TransitionSample> {
    sources
        .iter()
        .flat_map(|(_, samples)| samples.iter().cloned())
        .collect()
}

fn source_lengths(sources: &[(String, Vec<TransitionSample>)]) -> Vec<(String, usize)> {
    sources
        .iter()
        .map(|(name, samples)| (name.clone(), samples.len()))
        .collect()
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

/// Sattolo's algorithm: a seeded single-cycle permutation with no fixed points
/// for `len > 1`. This makes the action ablation deterministic and guarantees
/// that a transition never receives its own action-conditioning tuple.
fn action_shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    if len < 2 {
        return indices;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for i in (1..len).rev() {
        let j = rng.random_range(0..i);
        indices.swap(i, j);
    }
    indices
}

/// Copy only action conditioning from the permuted source transition. ACTION6
/// coordinates live inside `ArcAction`, so ID and coordinates cannot separate.
fn shuffled_action_samples(
    samples: &[TransitionSample],
    permutation: &[usize],
) -> Result<Vec<TransitionSample>> {
    if samples.len() != permutation.len() {
        bail!(
            "action permutation length {} != sample length {}",
            permutation.len(),
            samples.len()
        );
    }
    let unique: BTreeSet<_> = permutation.iter().copied().collect();
    if unique.len() != samples.len() || permutation.iter().any(|&idx| idx >= samples.len()) {
        bail!("action permutation must contain every sample index exactly once");
    }
    Ok(samples
        .iter()
        .zip(permutation.iter().copied())
        .map(|(target, source)| {
            let mut shuffled = target.clone();
            shuffled.action = samples[source].action.clone();
            shuffled
        })
        .collect())
}

fn summarize_action_shuffle(
    true_errors: &[f32],
    shuffled_errors: &[f32],
    changed_conditionings: usize,
    seed: u64,
) -> Result<ActionShuffleMetrics> {
    if true_errors.len() != shuffled_errors.len() {
        bail!(
            "true-action error count {} != shuffled-action error count {}",
            true_errors.len(),
            shuffled_errors.len()
        );
    }
    let true_action_mse = mean(true_errors);
    let shuffled_action_mse = mean(shuffled_errors);
    let ratio = match (true_action_mse, shuffled_action_mse) {
        (Some(true_mse), Some(shuffled_mse)) if true_mse > 0.0 => Some(shuffled_mse / true_mse),
        _ => None,
    };
    let (ratio_ci95_low, ratio_ci95_high) =
        bootstrap_paired_ratio_ci95(shuffled_errors, true_errors, seed);
    Ok(ActionShuffleMetrics {
        n: true_errors.len(),
        changed_conditionings,
        changed_fraction: (!true_errors.is_empty())
            .then_some(changed_conditionings as f64 / true_errors.len() as f64),
        true_action_mse,
        shuffled_action_mse,
        ratio,
        ratio_ci95_low,
        ratio_ci95_high,
        action_conditioning_pass: (changed_conditionings > 0).then_some(
            ratio.is_some_and(|value| value >= ACTION_SHUFFLE_MIN_RATIO)
                && ratio_ci95_low.is_some_and(|value| value > 1.0),
        ),
    })
}

fn action_coverage(samples: &[TransitionSample]) -> ActionCoverageMetrics {
    let mut action_counts = BTreeMap::new();
    let mut noop_labeled = 0usize;
    let mut noop_count = 0usize;
    let mut coordinate_actions = 0usize;
    let mut coordinates = BTreeSet::new();
    let mut action_conditionings = BTreeSet::new();
    for sample in samples {
        *action_counts.entry(sample.action.id).or_insert(0) += 1;
        action_conditionings.insert((sample.action.id, sample.action.x, sample.action.y));
        if let Some(noop) = sample.noop {
            noop_labeled += 1;
            noop_count += usize::from(noop);
        }
        if let (Some(x), Some(y)) = (sample.action.x, sample.action.y) {
            coordinate_actions += 1;
            coordinates.insert((x, y));
        }
    }
    let action_fractions = action_counts
        .iter()
        .map(|(&action, &count)| {
            (
                action,
                if samples.is_empty() {
                    0.0
                } else {
                    count as f64 / samples.len() as f64
                },
            )
        })
        .collect();
    ActionCoverageMetrics {
        n: samples.len(),
        action_counts,
        action_fractions,
        noop_labeled,
        noop_rate: (noop_labeled > 0).then_some(noop_count as f64 / noop_labeled as f64),
        coordinate_actions,
        distinct_coordinate_actions: coordinates.len(),
        distinct_action_conditionings: action_conditionings.len(),
    }
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

fn bootstrap_paired_ratio_ci95(
    numerators: &[f32],
    denominators: &[f32],
    seed: u64,
) -> (Option<f64>, Option<f64>) {
    if numerators.is_empty() || numerators.len() != denominators.len() {
        return (None, None);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut ratios = Vec::with_capacity(400);
    for _ in 0..400 {
        let mut numerator = 0f64;
        let mut denominator = 0f64;
        for _ in 0..numerators.len() {
            let idx = rng.random_range(0..numerators.len());
            numerator += numerators[idx] as f64;
            denominator += denominators[idx] as f64;
        }
        if denominator > f64::EPSILON {
            ratios.push(numerator / denominator);
        }
    }
    if ratios.is_empty() {
        return (None, None);
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = ratios[(0.025 * ratios.len() as f64) as usize];
    let hi = ratios[((0.975 * ratios.len() as f64) as usize).min(ratios.len() - 1)];
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

fn mat_transpose_mul_rhs(
    a: &[f32],
    rows: usize,
    cols: usize,
    b: &[f32],
    b_cols: usize,
) -> Vec<f32> {
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

fn encoder_variance_and_effective_rank(rows: &[Vec<f32>]) -> (Option<f64>, Option<f64>) {
    if rows.len() < 2 || rows.first().is_none_or(Vec::is_empty) {
        return (None, None);
    }
    let dim = rows[0].len();
    if rows.iter().any(|row| row.len() != dim) {
        return (None, None);
    }
    let centered = center_columns(rows);
    let denom = (centered.len() as f64 - 1.0).max(1.0);
    let mut covariance = vec![0f64; dim * dim];
    for row in &centered {
        for i in 0..dim {
            let vi = row[i] as f64;
            for j in 0..dim {
                covariance[i * dim + j] += vi * row[j] as f64 / denom;
            }
        }
    }
    let trace: f64 = (0..dim).map(|i| covariance[i * dim + i]).sum();
    let trace_sq: f64 = covariance.iter().map(|value| value * value).sum();
    let mean_variance = trace / dim as f64;
    let effective_rank = (trace_sq > f64::EPSILON).then_some(trace * trace / trace_sq);
    (Some(mean_variance), effective_rank)
}

fn summarize_representation(
    rows: &[Vec<f32>],
    sigreg_raw_sum: f64,
    sigreg_bounded_sum: f64,
    sigreg_n: usize,
) -> RepresentationDiagnostics {
    let encoder_dim = rows.first().map(Vec::len).unwrap_or(0);
    let (mean_encoder_variance, effective_rank) = encoder_variance_and_effective_rank(rows);
    let effective_rank_fraction = effective_rank
        .filter(|_| encoder_dim > 0)
        .map(|rank| rank / encoder_dim as f64);
    let sigreg_raw = (sigreg_n > 0).then_some(sigreg_raw_sum / sigreg_n as f64);
    let sigreg_bounded = (sigreg_n > 0).then_some(sigreg_bounded_sum / sigreg_n as f64);
    RepresentationDiagnostics {
        sigreg_raw,
        sigreg_bounded,
        sigreg_bound: SIGREG_BOUND,
        sigreg_near_bound: sigreg_bounded
            .map(|value| value >= SIGREG_BOUND * SIGREG_NEAR_BOUND_FRACTION),
        encoder_rows: rows.len(),
        encoder_dim,
        mean_encoder_variance,
        effective_rank,
        effective_rank_fraction,
        min_mean_variance: ENCODER_MIN_MEAN_VARIANCE,
        min_effective_rank_fraction: ENCODER_MIN_EFFECTIVE_RANK_FRACTION,
        noncollapse_pass: mean_encoder_variance.map(|variance| {
            variance >= ENCODER_MIN_MEAN_VARIANCE
                && effective_rank_fraction.unwrap_or(0.0) >= ENCODER_MIN_EFFECTIVE_RANK_FRACTION
        }),
    }
}

fn summarize_changed_transitions(
    learned: &[f32],
    copy_forward: &[f32],
    seed: u64,
) -> Result<ChangedTransitionMetrics> {
    if learned.len() != copy_forward.len() {
        bail!("changed-transition learned/copy-forward counts differ");
    }
    let learned_mse = mean(learned);
    let copy_forward_mse = mean(copy_forward);
    let improvement_fraction = learned_mse
        .zip(copy_forward_mse)
        .and_then(|(learned, copy)| (copy > 0.0).then_some(1.0 - learned / copy));
    let (ratio_low, ratio_high) = bootstrap_paired_ratio_ci95(learned, copy_forward, seed);
    let improvement_ci95_low = ratio_high.map(|ratio| 1.0 - ratio);
    let improvement_ci95_high = ratio_low.map(|ratio| 1.0 - ratio);
    Ok(ChangedTransitionMetrics {
        n: learned.len(),
        learned_mse,
        copy_forward_mse,
        improvement_fraction,
        improvement_ci95_low,
        improvement_ci95_high,
        ten_percent_improvement_pass: improvement_fraction.map(|value| value >= 0.10),
    })
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
    let latent_dim = encoders
        .iter()
        .find_map(|row| row.as_ref().map(|h| h.len()))?;
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
    next_z: &Tensor,
    k: usize,
    noise: f64,
    threshold: f64,
) -> Result<PtrmKMetrics> {
    let b = next_z.dim(0)?;
    // Q logits of the k trajectories actually under consideration, read once.
    let q_vals: Vec<Vec<f32>> = trajectories
        .iter()
        .take(k)
        .map(|t| Ok(t.q_logit.flatten_all()?.to_vec1::<f32>()?))
        .collect::<Result<_>>()?;
    let best_q_index_within = |sample: usize| -> usize {
        (0..q_vals.len())
            .max_by(|&a, &b| {
                q_vals[a][sample]
                    .partial_cmp(&q_vals[b][sample])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    };
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
        // `best_indices` ranks all `max_k` trajectories, but `mses` only covers
        // the first `k`. Indexing the former into the latter scored every
        // `best >= k` sample as a failure, so `best_q_at_k` and `ranking_gap`
        // for k < max_k measured the truncation rate (1 - k/max_k), not Q's
        // ranking. Re-rank over the truncated slice instead.
        let best = best_q_index_within(sample);
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
                next_z,
                k,
                // Trajectory 0 is forced deterministic (see `forward_ptrm`), so
                // the k=1 row really is the noise-free baseline this reports.
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
    representation_embeddings: Vec<Vec<f32>>,
    sigreg_raw_weighted: f64,
    sigreg_bounded_weighted: f64,
    sigreg_n: usize,
    changed_learned_errors: Vec<f32>,
    changed_copy_forward_errors: Vec<f32>,
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
    /// Per-sample disagreement, in sample order, so uncertainty can be scored
    /// against per-sample error instead of the reliability head's output.
    ensemble_disagreements: Vec<f32>,
    ensemble_n: usize,
}

fn eval_one_batch(
    model: &WorldModel,
    chunk: &[TransitionSample],
    bi: usize,
    train_cfg: &TrainConfig,
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
    let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
    let current_z = encoded.current;
    let next_z = encoded.next;
    let sigreg = (chunk.len() >= 2)
        .then(|| {
            sigreg_losses_for_encoded_pair(
                &current_z,
                &next_z,
                &encoded.current_raw,
                &encoded.next_raw,
                encoded.projected_sigreg.as_ref(),
                train_cfg,
                cfg.seed.wrapping_add(bi as u64),
            )
        })
        .transpose()?;
    let current_vecs = flatten_latent(&current_z)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let mut representation_embeddings = pool_latent(&current_z)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    representation_embeddings.extend(
        pool_latent(&next_z)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
    );
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
        representation_embeddings,
        sigreg_raw_weighted: match &sigreg {
            Some((raw, _)) => raw.to_scalar::<f32>()? as f64 * chunk.len() as f64,
            None => 0.0,
        },
        sigreg_bounded_weighted: match &sigreg {
            Some((_, bounded)) => bounded.to_scalar::<f32>()? as f64 * chunk.len() as f64,
            None => 0.0,
        },
        sigreg_n: sigreg.as_ref().map_or(0, |_| chunk.len()),
        ..Default::default()
    };
    let mses = per_sample_mse(&out.y, &next_z)?;
    let copy_forward_mses = per_sample_mse(&current_z, &next_z)?;
    partial.mse_all.extend(mses.iter().copied());
    for ((sample, learned), copy_forward) in chunk
        .iter()
        .zip(mses.iter().copied())
        .zip(copy_forward_mses.iter().copied())
    {
        if sample.current != sample.next {
            partial.changed_learned_errors.push(learned);
            partial.changed_copy_forward_errors.push(copy_forward);
        }
    }

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
                ys.push(traj.y.get(sample)?.flatten_all()?.to_vec1::<f32>()?);
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
                let per_sample = dsum / dpairs as f64;
                partial.ensemble_disagreement += per_sample;
                partial.ensemble_disagreements.push(per_sample as f32);
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
        let entry = partial
            .matched_acc
            .entry(k)
            .or_insert((0.0, 0, outer_steps));
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
        merged
            .representation_embeddings
            .extend(partial.representation_embeddings);
        merged.sigreg_raw_weighted += partial.sigreg_raw_weighted;
        merged.sigreg_bounded_weighted += partial.sigreg_bounded_weighted;
        merged.sigreg_n += partial.sigreg_n;
        merged
            .changed_learned_errors
            .extend(partial.changed_learned_errors);
        merged
            .changed_copy_forward_errors
            .extend(partial.changed_copy_forward_errors);
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
        merged
            .ensemble_disagreements
            .extend(partial.ensemble_disagreements);
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

fn eval_shuffled_action_batch(
    model: &WorldModel,
    samples: &[TransitionSample],
    shuffled: &[TransitionSample],
    device: &Device,
) -> Result<Vec<f32>> {
    if samples.len() != shuffled.len() {
        bail!("shuffled action batch must preserve sample count");
    }
    let batch = batch_from_samples(samples, device)?;
    let (shuffled_actions, shuffled_action_coords) = action_tensors_from_samples(shuffled, device)?;
    let out = model.forward(
        &batch.frames,
        &shuffled_actions,
        &shuffled_action_coords,
        &batch.goals,
    )?;
    let next_z = model.encode_state(&batch.next_frames)?;
    per_sample_mse(&out.y, &next_z)
}

#[allow(clippy::too_many_arguments)]
fn eval_action_diagnostics(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    true_errors: &[f32],
    source_lengths: &[(String, usize)],
    batch_size: usize,
    device: &Device,
    seed: u64,
) -> Result<ActionDiagnostics> {
    if samples.len() != true_errors.len() {
        bail!(
            "action diagnostics sample count {} != true-error count {}",
            samples.len(),
            true_errors.len()
        );
    }
    let declared: usize = source_lengths.iter().map(|(_, len)| *len).sum();
    if declared != samples.len() {
        bail!(
            "action diagnostic source lengths sum to {declared}, expected {}",
            samples.len()
        );
    }

    let mut permutation = Vec::with_capacity(samples.len());
    let mut source_ranges = Vec::with_capacity(source_lengths.len());
    let mut start = 0usize;
    for (source_index, (name, len)) in source_lengths.iter().enumerate() {
        let end = start + len;
        let local = action_shuffle_indices(
            *len,
            seed.wrapping_add(source_index as u64).wrapping_add(0xA571),
        );
        permutation.extend(local.into_iter().map(|index| start + index));
        source_ranges.push((name.clone(), start, end));
        start = end;
    }
    let shuffled = shuffled_action_samples(samples, &permutation)?;
    let ranges = batch_ranges(samples.len(), batch_size.max(1));
    let mut partials: Vec<(usize, Vec<f32>)> = if device.is_cpu() {
        ranges
            .par_iter()
            .enumerate()
            .map(|(batch_index, &(start, end))| {
                with_thread_local_model(train_cfg, checkpoint, device, |thread_model| {
                    eval_shuffled_action_batch(
                        thread_model,
                        &samples[start..end],
                        &shuffled[start..end],
                        device,
                    )
                    .map(|errors| (batch_index, errors))
                })
            })
            .collect::<Result<_>>()?
    } else {
        ranges
            .iter()
            .enumerate()
            .map(|(batch_index, &(start, end))| {
                let errors = eval_shuffled_action_batch(
                    model,
                    &samples[start..end],
                    &shuffled[start..end],
                    device,
                )?;
                if device.is_cuda() {
                    device.synchronize()?;
                }
                Ok((batch_index, errors))
            })
            .collect::<Result<_>>()?
    };
    partials.sort_by_key(|(batch_index, _)| *batch_index);
    let shuffled_errors: Vec<f32> = partials
        .into_iter()
        .flat_map(|(_, errors)| errors)
        .collect();
    if shuffled_errors.len() != true_errors.len() {
        bail!("shuffled-action evaluation changed sample count");
    }

    let mut aggregate_true = Vec::new();
    let mut aggregate_shuffled = Vec::new();
    let mut aggregate_changed = 0usize;
    let mut by_source = BTreeMap::new();
    for (source_index, (name, start, end)) in source_ranges.into_iter().enumerate() {
        let valid_shuffle = end.saturating_sub(start) >= 2;
        let true_slice = if valid_shuffle {
            &true_errors[start..end]
        } else {
            &[]
        };
        let shuffled_slice = if valid_shuffle {
            &shuffled_errors[start..end]
        } else {
            &[]
        };
        let changed_conditionings = samples[start..end]
            .iter()
            .zip(shuffled[start..end].iter())
            .filter(|(truth, ablated)| truth.action != ablated.action)
            .count();
        aggregate_true.extend_from_slice(true_slice);
        aggregate_shuffled.extend_from_slice(shuffled_slice);
        aggregate_changed += changed_conditionings;
        by_source.insert(
            name,
            ActionSourceDiagnostics {
                shuffle: summarize_action_shuffle(
                    true_slice,
                    shuffled_slice,
                    changed_conditionings,
                    seed.wrapping_add(source_index as u64).wrapping_add(0xB005),
                )?,
                coverage: action_coverage(&samples[start..end]),
            },
        );
    }
    Ok(ActionDiagnostics {
        aggregate: ActionSourceDiagnostics {
            shuffle: summarize_action_shuffle(
                &aggregate_true,
                &aggregate_shuffled,
                aggregate_changed,
                seed.wrapping_add(0xA661),
            )?,
            coverage: action_coverage(samples),
        },
        by_source,
    })
}

fn eval_rollout_group(
    model: &WorldModel,
    steps: &[TransitionSample],
    device: &Device,
) -> Result<Vec<EpisodeRolloutResult>> {
    if steps.len() < 4 {
        return Ok(Vec::new());
    }
    let z0 = {
        let sample = &steps[0];
        let first =
            batch_from_samples(std::slice::from_ref(sample), device).with_context(|| {
                rollout_operation_context(sample, "open-loop", "initial batch construction")
            })?;
        model.encode_state(&first.frames).with_context(|| {
            rollout_operation_context(sample, "open-loop", "initial state encoding")
        })?
    };
    let mut open_latent = z0.clone();
    let mut rows = Vec::new();
    for (idx, sample) in steps.iter().enumerate() {
        let batch =
            batch_from_samples(std::slice::from_ref(sample), device).with_context(|| {
                rollout_operation_context(sample, "open-loop", "step batch construction")
            })?;
        let open_pred = model
            .forward_from_latent(
                &open_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
            )
            .with_context(|| rollout_operation_context(sample, "open-loop", "latent forward"))?
            .y;
        open_latent = open_pred.clone();
        let closed_latent = model
            .encode_state(&batch.frames)
            .with_context(|| rollout_operation_context(sample, "closed-loop", "state encoding"))?;
        let closed_pred = model
            .forward_from_latent(
                &closed_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
            )
            .with_context(|| rollout_operation_context(sample, "closed-loop", "latent forward"))?
            .y;
        let target = model.encode_state(&batch.next_frames).with_context(|| {
            rollout_operation_context(sample, "open-loop", "target state encoding")
        })?;
        let open_mse = open_pred
            .sub(&target)
            .and_then(|value| value.sqr())
            .and_then(|value| value.mean_all())
            .and_then(|value| value.to_scalar::<f32>())
            .with_context(|| {
                rollout_operation_context(sample, "open-loop", "model MSE reduction")
            })? as f64;
        let closed_mse = closed_pred
            .sub(&target)
            .and_then(|value| value.sqr())
            .and_then(|value| value.mean_all())
            .and_then(|value| value.to_scalar::<f32>())
            .with_context(|| {
                rollout_operation_context(sample, "closed-loop", "model MSE reduction")
            })? as f64;
        let copy_forward_mse = z0
            .sub(&target)
            .and_then(|value| value.sqr())
            .and_then(|value| value.mean_all())
            .and_then(|value| value.to_scalar::<f32>())
            .with_context(|| {
                rollout_operation_context(sample, "open-loop", "copy-forward MSE reduction")
            })? as f64;
        let horizon = idx + 1;
        if matches!(horizon, 4 | 8 | 16) {
            rows.push(episode_rollout_result(
                steps,
                horizon,
                open_mse,
                closed_mse,
                copy_forward_mse,
            ));
        }
    }
    Ok(rows)
}

fn rollout_operation_context(sample: &TransitionSample, mode: &str, operation: &str) -> String {
    format!(
        "{operation} failed during {} rollout (seed={}, episode={}, transition={})",
        mode, sample.seed, sample.episode_id, sample.transition_index,
    )
}

#[derive(Clone, Copy)]
enum RolloutMetric {
    Open,
    Closed,
    CopyForward,
}

fn rollout_metrics_from_rows(
    rows: &[EpisodeRolloutRow],
    metric: RolloutMetric,
    seed: u64,
) -> RolloutMetrics {
    let values = |horizon| -> Vec<f32> {
        rows.iter()
            .filter(|row| row.horizon == horizon)
            .filter_map(|row| match metric {
                RolloutMetric::Open => row.open_mse,
                RolloutMetric::Closed => row.closed_mse,
                RolloutMetric::CopyForward => row.copy_forward_mse,
            })
            .map(|value| value as f32)
            .collect()
    };
    let normalized = |horizon| -> Vec<f32> {
        (matches!(metric, RolloutMetric::Open))
            .then(|| {
                rows.iter()
                    .filter(|row| row.horizon == horizon)
                    .filter_map(|row| row.normalized_open_mse)
                    .map(|value| value as f32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let h4_values = values(4);
    let h8_values = values(8);
    let h16_values = values(16);
    let (h4_seed_offset, h8_seed_offset, h16_seed_offset) = match metric {
        RolloutMetric::CopyForward => (0x14, 0x18, 0x1C),
        RolloutMetric::Open | RolloutMetric::Closed => (0x04, 0x08, 0x10),
    };
    RolloutMetrics {
        n4: h4_values.len(),
        mse_4: mean(&h4_values),
        n8: h8_values.len(),
        mse_8: mean(&h8_values),
        n16: h16_values.len(),
        mse_16: mean(&h16_values),
        h4: Some(summarize_horizon(
            &h4_values,
            &normalized(4),
            seed ^ h4_seed_offset,
        )),
        h8: Some(summarize_horizon(
            &h8_values,
            &normalized(8),
            seed ^ h8_seed_offset,
        )),
        h16: Some(summarize_horizon(
            &h16_values,
            &normalized(16),
            seed ^ h16_seed_offset,
        )),
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

fn eval_episode_rollouts(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    source: &str,
) -> Result<Vec<EpisodeRolloutRow>> {
    let groups: Vec<Vec<TransitionSample>> = group_rollouts(samples)
        .into_values()
        .filter(|steps| steps.len() >= 4)
        .map(|steps| steps.iter().map(|s| (*s).clone()).collect())
        .collect();
    let results: Vec<Vec<EpisodeRolloutResult>> = if device.is_cpu() {
        groups
            .into_par_iter()
            .map(|steps| {
                with_thread_local_model(train_cfg, checkpoint, device, |m| {
                    eval_rollout_group(m, &steps, device)
                        .with_context(|| format!("{source} rollout group failed"))
                })
            })
            .collect::<Result<_>>()?
    } else {
        groups
            .iter()
            .map(|steps| {
                eval_rollout_group(model, steps, device)
                    .with_context(|| format!("{source} rollout group failed"))
            })
            .collect::<Result<_>>()?
    };
    let mut rows: Vec<_> = results
        .into_iter()
        .flatten()
        .map(|result| result.into_row(source))
        .collect();
    sort_episode_rows(&mut rows);
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn eval_sample_set(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    source: &str,
    action_sources: Option<&[(String, usize)]>,
    cfg: &EvalConfig,
    device: &Device,
    with_rollout: bool,
) -> Result<SplitEval> {
    if samples.is_empty() {
        return Ok(SplitEval {
            source: source.into(),
            n_samples: 0,
            one_step_latent_mse: None,
            representation: None,
            changed_transitions: None,
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
            action_diagnostics: None,
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
                    eval_one_batch(m, chunk, bi, train_cfg, cfg, device)
                        .map(|partial| (bi, partial))
                })
            })
            .collect::<Result<_>>()?;
        merge_batch_partials(partials)
    } else {
        let partials: Vec<(usize, BatchEvalPartial)> = ranges
            .iter()
            .enumerate()
            .map(|(bi, &(start, end))| {
                let partial =
                    eval_one_batch(model, &samples[start..end], bi, train_cfg, cfg, device)
                        .map(|partial| (bi, partial))?;
                if device.is_cuda() {
                    device.synchronize()?;
                }
                Ok(partial)
            })
            .collect::<Result<_>>()?;
        merge_batch_partials(partials)
    };

    let representation = Some(summarize_representation(
        &merged.representation_embeddings,
        merged.sigreg_raw_weighted,
        merged.sigreg_bounded_weighted,
        merged.sigreg_n,
    ));
    let changed_transitions = Some(summarize_changed_transitions(
        &merged.changed_learned_errors,
        &merged.changed_copy_forward_errors,
        cfg.seed.wrapping_add(0xC0F1),
    )?);
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
        // Score ensemble *disagreement* against error. This previously read
        // `reliability_probs`, i.e. the reliability head, which made
        // `uncertainty_auroc` a bit-identical copy of `reliability_auroc` and
        // meant the pre-registered "disagreement is uncorrelated with error"
        // stop rule could never be evaluated. Higher disagreement should
        // predict higher error, so the score is used directly.
        let uncertainty: Vec<f32> = merged
            .ensemble_disagreements
            .iter()
            .take(high_error.len())
            .copied()
            .collect();
        EnsembleMetrics {
            members: cfg.ensemble_members,
            mean_disagreement,
            uncertainty_auroc: if uncertainty.len() == high_error.len() && !high_error.is_empty() {
                binary_auroc(&uncertainty, &high_error)
            } else {
                None
            },
        }
    });
    let contrastive = eval_contrastive_probes(model, samples, device).ok();
    let fallback_sources = [(source.to_string(), samples.len())];
    let action_diagnostics = Some(eval_action_diagnostics(
        train_cfg,
        checkpoint,
        model,
        samples,
        &mse_all,
        action_sources.unwrap_or(&fallback_sources),
        cfg.physical_batch,
        device,
        cfg.seed,
    )?);

    let rollout_rows = if with_rollout {
        Some(eval_episode_rollouts(
            train_cfg, checkpoint, model, samples, device, source,
        )?)
    } else {
        None
    };
    let rollout = rollout_rows
        .as_deref()
        .map(|rows| rollout_metrics_from_rows(rows, RolloutMetric::Open, cfg.seed ^ 0x01));
    let closed_loop = rollout_rows
        .as_deref()
        .map(|rows| rollout_metrics_from_rows(rows, RolloutMetric::Closed, cfg.seed ^ 0xC1));
    let identifiability = eval_identifiability(samples, &encoder_embeddings);

    Ok(SplitEval {
        source: source.into(),
        n_samples: samples.len(),
        one_step_latent_mse: mean(&mse_all),
        representation,
        changed_transitions,
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
        action_diagnostics,
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
        mean_latent_norm: Some(probes.iter().map(|p| p.mean_latent_norm).sum::<f64>() / n as f64),
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
    let batch = batch_from_samples(std::slice::from_ref(sample), device)?;
    let z = model.encode_state(&batch.frames)?;
    let noop = Tensor::zeros((1,), DType::U32, device)?;
    let noop_coords = Tensor::zeros((1, 2), DType::F32, device)?;
    let pred_noop = model.forward_from_latent(&z, &noop, &noop_coords, &batch.goals)?;
    let noop_mse = pred_noop.y.sub(&z)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let pred_action =
        model.forward_from_latent(&z, &batch.actions, &batch.action_coords, &batch.goals)?;
    // This probe measures whether the dynamics react to the action at all.
    // Prediction error against `next_z` is reported separately and is not an
    // action-effect magnitude; using it here made the ratio compare unrelated
    // quantities (target error versus no-op drift).
    let action_mse = pred_action
        .y
        .sub(&z)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()? as f64;
    let ratio = (noop_mse > 0.0).then(|| action_mse / noop_mse);
    Ok(ContrastiveProbeMetrics {
        noop_identity_mse: Some(noop_mse),
        action_effect_mse: Some(action_mse),
        // A real inverse-action probe needs a paired inverse transition. The
        // previous implementation compared a delta with its own negation and
        // therefore reported -1 for every model.
        inverse_action_cosine: None,
        action_effect_ratio: ratio,
        copy_forward_pass: ratio.map(|r| r >= COPY_FORWARD_MIN_RATIO),
    })
}

static EPISODE_JSONL_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn episode_jsonl_bytes(rows: &[EpisodeRolloutRow]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for row in rows {
        writeln!(
            bytes,
            "{}",
            serde_json::to_string(row).context("serialize episode row")?
        )?;
    }
    Ok(bytes)
}

fn staging_path(destination: &Path, parent: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("episodes.jsonl");
    let sequence = EPISODE_JSONL_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    staging_path_with_sequence(name, parent, sequence)
}

fn staging_path_with_sequence(name: &str, parent: &Path, sequence: u64) -> PathBuf {
    parent.join(format!(
        ".{name}.{}.{}.staging",
        std::process::id(),
        sequence
    ))
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EpisodeJsonlWritePhase {
    BeforeWrite,
    BeforeFileSync,
    BeforeRename,
    BeforeParentSync,
}

fn write_episode_jsonl_bytes_with<F>(path: &Path, bytes: &[u8], before: F) -> Result<()>
where
    F: Fn(EpisodeJsonlWritePhase) -> Result<()>,
{
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    loop {
        let staging = staging_path(path, parent);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("create staging {}", staging.display()))
            }
        };
        let result = (|| {
            before(EpisodeJsonlWritePhase::BeforeWrite)?;
            file.write_all(bytes)
                .with_context(|| format!("write staging {}", staging.display()))?;
            file.flush()
                .with_context(|| format!("flush staging {}", staging.display()))?;
            before(EpisodeJsonlWritePhase::BeforeFileSync)?;
            file.sync_all()
                .with_context(|| format!("sync staging {}", staging.display()))?;
            drop(file);
            before(EpisodeJsonlWritePhase::BeforeRename)?;
            fs::rename(&staging, path).with_context(|| {
                format!("atomic rename {} -> {}", staging.display(), path.display())
            })?;
            before(EpisodeJsonlWritePhase::BeforeParentSync)?;
            fs::File::open(parent)
                .with_context(|| format!("open parent directory {}", parent.display()))?
                .sync_all()
                .with_context(|| format!("sync parent directory {}", parent.display()))?;
            Ok(())
        })();
        if result.is_err() && staging.exists() {
            fs::remove_file(&staging)
                .with_context(|| format!("remove failed staging {}", staging.display()))?;
        }
        return result;
    }
}

fn sort_episode_rows(rows: &mut [EpisodeRolloutRow]) {
    rows.sort_by(|left, right| {
        (&left.source, left.seed, left.episode_id, left.horizon).cmp(&(
            &right.source,
            right.seed,
            right.episode_id,
            right.horizon,
        ))
    });
}

fn write_episode_jsonl(path: &Path, rows: &[EpisodeRolloutRow]) -> Result<()> {
    let mut rows = rows.to_vec();
    sort_episode_rows(&mut rows);
    write_episode_jsonl_bytes_with(path, &episode_jsonl_bytes(&rows)?, |_| Ok(()))
}

fn maybe_write_episode_jsonl(path: Option<&Path>, rows: &[EpisodeRolloutRow]) -> Result<()> {
    if let Some(path) = path {
        write_episode_jsonl(path, rows)?;
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

    let mut dynamics_sources = collect_synthetic_sources(
        cfg.seed,
        cfg.synthetic_episodes,
        &["random_one_step", "exploration"],
    )?;
    dynamics_sources.push((
        "hazard_one_step".into(),
        collect_hazard_samples(cfg.seed, cfg.synthetic_episodes)?,
    ));
    let dynamics_samples = flatten_sources(&dynamics_sources);
    let dynamics_source_lengths = source_lengths(&dynamics_sources);
    let dynamics_rollout_samples =
        collect_dynamics_rollout_samples(cfg.seed, cfg.synthetic_episodes)?;

    let planner_sources = collect_synthetic_sources(
        cfg.seed,
        cfg.synthetic_episodes,
        &[
            "sequential",
            "hypothesis_probe",
            "p1c_falsification",
            "p1c_hard_retarget",
        ],
    )?;
    let planner_samples = flatten_sources(&planner_sources);
    let planner_source_lengths = source_lengths(&planner_sources);
    let planner_rollout_samples =
        collect_planner_rollout_samples(cfg.seed, cfg.synthetic_episodes)?;

    let mut synthetic_dynamics = eval_sample_set(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_samples,
        "synthetic_dynamics",
        Some(&dynamics_source_lengths),
        cfg,
        &device,
        false,
    )?;
    let dynamics_rollout_rows = eval_episode_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &dynamics_rollout_samples,
        &device,
        "synthetic_dynamics",
    )?;
    synthetic_dynamics.rollout = Some(rollout_metrics_from_rows(
        &dynamics_rollout_rows,
        RolloutMetric::Open,
        cfg.seed ^ 0x01,
    ));
    synthetic_dynamics.closed_loop = Some(rollout_metrics_from_rows(
        &dynamics_rollout_rows,
        RolloutMetric::Closed,
        cfg.seed ^ 0xC1,
    ));
    synthetic_dynamics.copy_forward = Some(rollout_metrics_from_rows(
        &dynamics_rollout_rows,
        RolloutMetric::CopyForward,
        cfg.seed ^ 0xCF,
    ));
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
        Some(&planner_source_lengths),
        cfg,
        &device,
        false,
    )?;
    let planner_rollout_rows = eval_episode_rollouts(
        &train_cfg,
        &cfg.checkpoint,
        &model,
        &planner_rollout_samples,
        &device,
        "synthetic_planner",
    )?;
    synthetic_planner.rollout = Some(rollout_metrics_from_rows(
        &planner_rollout_rows,
        RolloutMetric::Open,
        cfg.seed ^ 0x01,
    ));
    synthetic_planner.closed_loop = Some(rollout_metrics_from_rows(
        &planner_rollout_rows,
        RolloutMetric::Closed,
        cfg.seed ^ 0xC1,
    ));
    synthetic_planner.copy_forward = Some(rollout_metrics_from_rows(
        &planner_rollout_rows,
        RolloutMetric::CopyForward,
        cfg.seed ^ 0xCF,
    ));
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
            None,
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
    if let Some(jsonl) = cfg.episode_jsonl.as_deref() {
        let mut rows = dynamics_rollout_rows;
        rows.extend(planner_rollout_rows);
        sort_episode_rows(&mut rows);
        maybe_write_episode_jsonl(Some(jsonl), &rows)?;
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
    use crate::p2::data::{ArcAction, ArcFrame, GoalFeatures};
    use crate::p2::train::{
        reinit_varmap_deterministic, save_checkpoint, TrainReport, TRAIN_REPORT_SCHEMA,
    };

    fn action_diagnostic_sample(action: ArcAction, episode_id: u64) -> Result<TransitionSample> {
        let current = ArcFrame::new(1, 1, vec![episode_id as u8])?;
        let next = ArcFrame::new(1, 1, vec![episode_id as u8 + 1])?;
        Ok(TransitionSample {
            current,
            next,
            action,
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: None,
            goal_failed: None,
            exhausted: None,
            split: Split::HeldOutComposition,
            family: "action_diagnostic".into(),
            seed: 7,
            episode_id,
            transition_index: 0,
            oracle_latent: None,
        })
    }

    fn rollout_fixture_sample(
        seed: u64,
        episode_id: u64,
        transition_index: u64,
        family: &str,
    ) -> Result<TransitionSample> {
        let mut sample = action_diagnostic_sample(ArcAction::new(1, None, None)?, 1)?;
        sample.seed = seed;
        sample.episode_id = episode_id;
        sample.transition_index = transition_index;
        sample.family = family.into();
        Ok(sample)
    }

    #[allow(clippy::too_many_arguments)]
    fn rollout_row(
        source: &str,
        seed: u64,
        episode_id: u64,
        horizon: usize,
        open_mse: Option<f64>,
        closed_mse: Option<f64>,
        copy_forward_mse: Option<f64>,
        families: &[&str],
    ) -> EpisodeRolloutRow {
        let mut families_through_horizon: Vec<_> =
            families.iter().map(|family| (*family).into()).collect();
        families_through_horizon.sort();
        families_through_horizon.dedup();
        EpisodeRolloutResult {
            seed,
            episode_id,
            families_through_horizon,
            horizon,
            open_mse,
            closed_mse,
            copy_forward_mse,
        }
        .into_row(source)
    }

    #[test]
    fn rollout_rows_preserve_episode_identity_horizons_and_retarget_provenance() -> Result<()> {
        let mut samples = Vec::new();
        for index in 0..16 {
            samples.push(rollout_fixture_sample(
                9,
                42,
                index,
                if index < 3 { "alpha" } else { "retarget" },
            )?);
        }
        for index in 0..7 {
            samples.push(rollout_fixture_sample(9, 77, index, "short")?);
        }
        samples.reverse();

        let groups = group_rollouts(&samples);
        let rows: Vec<_> = groups
            .values()
            .flat_map(|steps| {
                [4, 8, 16]
                    .into_iter()
                    .filter(move |horizon| steps.len() >= *horizon)
                    .map(|horizon| {
                        episode_rollout_result(steps, horizon, 1.0, 2.0, 4.0)
                            .into_row("synthetic_planner")
                    })
            })
            .collect();

        assert_eq!(
            rows.iter()
                .map(|row| (row.seed, row.episode_id, row.horizon))
                .collect::<Vec<_>>(),
            vec![(9, 42, 4), (9, 42, 8), (9, 42, 16), (9, 77, 4)]
        );
        assert_eq!(rows[0].families_through_horizon, vec!["alpha", "retarget"]);
        assert_eq!(rows[1].families_through_horizon, vec!["alpha", "retarget"]);
        assert_eq!(rows[3].families_through_horizon, vec!["short"]);
        Ok(())
    }

    #[test]
    fn rollout_rows_reconcile_aggregates_and_weight_each_episode_once() {
        let rows = vec![
            rollout_row(
                "synthetic_dynamics",
                1,
                3,
                4,
                Some(2.0),
                Some(4.0),
                Some(1.0),
                &["a"],
            ),
            rollout_row(
                "synthetic_dynamics",
                1,
                4,
                4,
                Some(6.0),
                Some(8.0),
                Some(2.0),
                &["a", "b", "c"],
            ),
            rollout_row(
                "synthetic_dynamics",
                1,
                3,
                8,
                Some(10.0),
                Some(20.0),
                Some(5.0),
                &["a"],
            ),
        ];
        let open = rollout_metrics_from_rows(&rows, RolloutMetric::Open, 7);
        let closed = rollout_metrics_from_rows(&rows, RolloutMetric::Closed, 7);
        let copy_forward = rollout_metrics_from_rows(&rows, RolloutMetric::CopyForward, 7);

        assert_eq!(
            (open.n4, open.mse_4, open.n8, open.mse_8, open.n16),
            (2, Some(4.0), 1, Some(10.0), 0)
        );
        assert_eq!((closed.mse_4, closed.mse_8), (Some(6.0), Some(20.0)));
        assert_eq!(
            (copy_forward.mse_4, copy_forward.mse_8),
            (Some(1.5), Some(5.0))
        );
        assert_eq!(open.h4.and_then(|stats| stats.normalized_mean), Some(2.5));
    }

    #[test]
    fn rollout_normalization_rejects_invalid_copy_forward_denominators() {
        for denominator in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                rollout_row(
                    "synthetic_dynamics",
                    1,
                    1,
                    4,
                    Some(3.0),
                    Some(2.0),
                    Some(denominator),
                    &["a"]
                )
                .normalized_open_mse,
                None
            );
        }
        assert_eq!(
            rollout_row(
                "synthetic_dynamics",
                1,
                1,
                4,
                Some(3.0),
                Some(2.0),
                Some(2.0),
                &["a"]
            )
            .normalized_open_mse,
            Some(1.5)
        );
        for numerator in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                rollout_row(
                    "synthetic_dynamics",
                    1,
                    1,
                    4,
                    Some(numerator),
                    Some(2.0),
                    Some(2.0),
                    &["a"]
                )
                .normalized_open_mse,
                None
            );
        }
    }

    #[test]
    fn copy_forward_bootstrap_seeds_preserve_horizon_offsets() {
        let h4 = [1.0_f32, 3.0, 7.0, 13.0, 21.0];
        let h8 = [2.0_f32, 5.0, 11.0, 17.0, 23.0];
        let h16 = [4.0_f32, 6.0, 10.0, 16.0, 26.0];
        let rows = h4
            .iter()
            .chain(h8.iter())
            .chain(h16.iter())
            .enumerate()
            .map(|(index, value)| {
                rollout_row(
                    "synthetic_dynamics",
                    1,
                    index as u64,
                    if index < h4.len() {
                        4
                    } else if index < h4.len() + h8.len() {
                        8
                    } else {
                        16
                    },
                    Some(f64::from(*value)),
                    Some(f64::from(*value)),
                    Some(f64::from(*value)),
                    &["a"],
                )
            })
            .collect::<Vec<_>>();
        let caller_seed = 0x1234_u64 ^ 0xCF;
        let metrics = rollout_metrics_from_rows(&rows, RolloutMetric::CopyForward, caller_seed);

        assert_eq!(
            (
                metrics.h4.as_ref().unwrap().ci95_low,
                metrics.h4.as_ref().unwrap().ci95_high
            ),
            bootstrap_ci95(&h4, caller_seed ^ 0x14)
        );
        assert_eq!(
            (
                metrics.h8.as_ref().unwrap().ci95_low,
                metrics.h8.as_ref().unwrap().ci95_high
            ),
            bootstrap_ci95(&h8, caller_seed ^ 0x18)
        );
        assert_eq!(
            (
                metrics.h16.as_ref().unwrap().ci95_low,
                metrics.h16.as_ref().unwrap().ci95_high
            ),
            bootstrap_ci95(&h16, caller_seed ^ 0x1C)
        );
    }

    #[test]
    fn episode_jsonl_is_deterministic_sorted_and_durable() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("tofy-episode-jsonl-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        let path = dir.join("episodes.jsonl");
        let mut rows = vec![
            rollout_row(
                "synthetic_planner",
                4,
                2,
                8,
                Some(2.0),
                Some(3.0),
                Some(1.0),
                &["x"],
            ),
            rollout_row(
                "synthetic_dynamics",
                3,
                1,
                4,
                Some(1.0),
                Some(2.0),
                Some(1.0),
                &["y"],
            ),
        ];
        write_episode_jsonl(&path, &rows)?;
        let first = fs::read(&path)?;
        rows.reverse();
        write_episode_jsonl(&path, &rows)?;
        assert_eq!(fs::read(&path)?, first);
        let sorted_rows: Vec<EpisodeRolloutRow> = String::from_utf8(first.clone())?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(
            sorted_rows
                .iter()
                .map(|row| (row.source.as_str(), row.seed, row.episode_id, row.horizon))
                .collect::<Vec<_>>(),
            vec![
                ("synthetic_dynamics", 3, 1, 4),
                ("synthetic_planner", 4, 2, 8),
            ]
        );

        let phases = RefCell::new(Vec::new());
        write_episode_jsonl_bytes_with(&path, &first, |phase| {
            phases.borrow_mut().push(phase);
            Ok(())
        })?;
        assert_eq!(
            *phases.borrow(),
            vec![
                EpisodeJsonlWritePhase::BeforeWrite,
                EpisodeJsonlWritePhase::BeforeFileSync,
                EpisodeJsonlWritePhase::BeforeRename,
                EpisodeJsonlWritePhase::BeforeParentSync,
            ]
        );

        for failure_phase in [
            EpisodeJsonlWritePhase::BeforeWrite,
            EpisodeJsonlWritePhase::BeforeFileSync,
            EpisodeJsonlWritePhase::BeforeRename,
        ] {
            let before_failure = fs::read(&path)?;
            assert!(
                write_episode_jsonl_bytes_with(&path, b"replacement\n", |phase| {
                    if phase == failure_phase {
                        bail!("forced pre-rename failure");
                    }
                    Ok(())
                })
                .is_err()
            );
            assert_eq!(fs::read(&path)?, before_failure);
            let staging_entries: Vec<_> = fs::read_dir(&dir)?
                .map(|entry| entry.map(|entry| entry.file_name()))
                .collect::<std::io::Result<_>>()?;
            assert!(staging_entries
                .iter()
                .all(|name| !name.to_string_lossy().ends_with(".staging")));
        }
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn episode_jsonl_skips_existing_staging_candidate() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "tofy-episode-jsonl-collision-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        let path = dir.join("episodes.jsonl");
        let sequence = EPISODE_JSONL_STAGING_SEQUENCE.load(Ordering::Relaxed);
        let collision = staging_path_with_sequence("episodes.jsonl", &dir, sequence);
        fs::write(&collision, b"crash-leftover\n")?;

        write_episode_jsonl_bytes_with(&path, b"replacement\n", |_| Ok(()))?;

        assert_eq!(fs::read(&collision)?, b"crash-leftover\n");
        assert_eq!(fs::read(&path)?, b"replacement\n");
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn disabled_episode_jsonl_does_not_change_row_aggregates() {
        let rows = vec![rollout_row(
            "synthetic_dynamics",
            2,
            8,
            4,
            Some(7.0),
            Some(9.0),
            Some(2.0),
            &["a"],
        )];
        let before = rollout_metrics_from_rows(&rows, RolloutMetric::Open, 1);
        maybe_write_episode_jsonl(None, &rows).expect("disabled JSONL output");
        let after = rollout_metrics_from_rows(&rows, RolloutMetric::Open, 1);
        assert_eq!(before.mse_4, after.mse_4);
        assert_eq!(
            before.h4.and_then(|stats| stats.normalized_mean),
            after.h4.and_then(|stats| stats.normalized_mean)
        );
    }

    #[test]
    fn action_shuffle_is_a_deterministic_derangement_and_keeps_coordinates_attached() -> Result<()>
    {
        let samples = vec![
            action_diagnostic_sample(ArcAction::new(1, None, None)?, 0)?,
            action_diagnostic_sample(ArcAction::new(6, Some(11), Some(23))?, 1)?,
            action_diagnostic_sample(ArcAction::new(7, None, None)?, 2)?,
        ];
        let permutation = action_shuffle_indices(samples.len(), 19);
        assert_eq!(permutation, action_shuffle_indices(samples.len(), 19));
        assert!(permutation
            .iter()
            .enumerate()
            .all(|(target, source)| target != *source));

        let shuffled = shuffled_action_samples(&samples, &permutation)?;
        for (target, source) in permutation.into_iter().enumerate() {
            assert_eq!(shuffled[target].action, samples[source].action);
            assert_eq!(shuffled[target].current, samples[target].current);
            assert_eq!(shuffled[target].next, samples[target].next);
            assert_eq!(
                shuffled[target].goal_features,
                samples[target].goal_features
            );
        }
        Ok(())
    }

    #[test]
    fn action_shuffle_ratio_compares_prediction_error_on_the_same_targets() -> Result<()> {
        let metrics = summarize_action_shuffle(&[1.0, 3.0], &[2.0, 4.0], 2, 7)?;
        assert_eq!(metrics.n, 2);
        assert_eq!(metrics.changed_conditionings, 2);
        assert_eq!(metrics.changed_fraction, Some(1.0));
        assert_eq!(metrics.true_action_mse, Some(2.0));
        assert_eq!(metrics.shuffled_action_mse, Some(3.0));
        assert_eq!(metrics.ratio, Some(1.5));
        assert_eq!(metrics.action_conditioning_pass, Some(true));

        let failed = summarize_action_shuffle(&[2.0, 2.0], &[2.1, 2.1], 2, 7)?;
        assert_eq!(failed.action_conditioning_pass, Some(false));
        let unavailable = summarize_action_shuffle(&[2.0, 2.0], &[2.0, 2.0], 0, 7)?;
        assert_eq!(unavailable.action_conditioning_pass, None);
        Ok(())
    }

    #[test]
    fn representation_diagnostics_apply_preregistered_collapse_thresholds() {
        let collapsed = summarize_representation(
            &[vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
            30_000.0,
            29_850.0,
            3,
        );
        assert_eq!(collapsed.sigreg_raw, Some(10_000.0));
        assert_eq!(collapsed.sigreg_bounded, Some(9_950.0));
        assert_eq!(collapsed.sigreg_near_bound, Some(true));
        assert_eq!(collapsed.noncollapse_pass, Some(false));

        let diverse = summarize_representation(
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
            40.0,
            39.0,
            4,
        );
        assert!(diverse.mean_encoder_variance.unwrap() > ENCODER_MIN_MEAN_VARIANCE);
        assert!(diverse.effective_rank_fraction.unwrap() > ENCODER_MIN_EFFECTIVE_RANK_FRACTION);
        assert_eq!(diverse.noncollapse_pass, Some(true));
    }

    #[test]
    fn changed_transition_metrics_compare_only_paired_changed_rows() -> Result<()> {
        let metrics = summarize_changed_transitions(&[0.8, 0.6, 0.4], &[1.0, 0.8, 0.6], 17)?;
        assert_eq!(metrics.n, 3);
        assert!(metrics.improvement_fraction.unwrap() >= 0.10);
        assert_eq!(metrics.ten_percent_improvement_pass, Some(true));
        assert!(metrics.improvement_ci95_low.is_some());
        assert!(metrics.improvement_ci95_high.is_some());
        Ok(())
    }

    #[test]
    fn action_tensor_conversion_rejects_coordinates_on_non_coordinate_actions() -> Result<()> {
        let invalid = action_diagnostic_sample(
            ArcAction {
                id: 1,
                x: Some(11),
                y: Some(23),
            },
            0,
        )?;
        assert!(action_tensors_from_samples(&[invalid], &Device::Cpu).is_err());
        Ok(())
    }

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
            batch_schedule_migrations: vec![],
            checkpoint: dir.join("model.safetensors"),
            export_checkpoint: None,
            config_path: dir.join("config.json"),
            profile: crate::p2::cg_profile::ProfileState::Pending,
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
            episode_jsonl: Some(dir.join("episodes.jsonl")),
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
        assert_eq!(
            eval.synthetic_dynamics.deterministic_matched_compute.len(),
            2
        );
        let action_diagnostics = eval
            .synthetic_dynamics
            .action_diagnostics
            .as_ref()
            .expect("synthetic action diagnostics");
        assert!(action_diagnostics.aggregate.shuffle.n > 0);
        assert!(action_diagnostics.aggregate.shuffle.ratio.is_some());
        let random_one_step = action_diagnostics
            .by_source
            .get("random_one_step")
            .expect("random-one-step action diagnostics");
        assert!(random_one_step
            .coverage
            .action_counts
            .get(&6)
            .is_some_and(|count| *count > 0));
        assert!(random_one_step.coverage.distinct_coordinate_actions > 0);
        assert!(eval
            .synthetic_planner
            .action_diagnostics
            .as_ref()
            .is_some_and(|metrics| metrics.by_source.contains_key("sequential")));
        let text = fs::read_to_string(&eval_cfg.output)?;
        let back: EvalReport = serde_json::from_str(&text)?;
        assert_eq!(back.schema, EVAL_REPORT_SCHEMA);
        let episode_rows: Vec<EpisodeRolloutRow> =
            fs::read_to_string(eval_cfg.episode_jsonl.as_ref().expect("episode JSONL path"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        assert!(!episode_rows.is_empty());
        assert!(episode_rows.iter().all(|row| {
            row.schema == "p2.episode_rollout.v2"
                && matches!(row.horizon, 4 | 8 | 16)
                && !row.families_through_horizon.is_empty()
        }));
        assert!(episode_rows
            .iter()
            .any(|row| row.source == "synthetic_dynamics"));
        assert!(episode_rows
            .iter()
            .any(|row| row.source == "synthetic_planner"));

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
