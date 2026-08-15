//! P2 world-model evaluation (synthetic held-out + ARC recording transfer).

use crate::domain::Split;
use crate::gpu_lock::GpuSessionGuard;
use crate::p2::arc3::{import_recordings_dir, summarize_recordings_dir, RecordingRunSummary};
use crate::p2::board_probe::{
    BoardProbeRows, BoardProbeTransitions, BoardTransitionMetrics, FixedBoardProbe, PATCH_COUNT,
};
use crate::p2::calibration::{binary_auroc, expected_calibration_error, risk_coverage_buckets};
use crate::p2::data::{
    generate_curriculum, generate_factual_branch_group, generate_hazard_one_step, BranchGroup,
    FactualBatch, TransitionSample, ORACLE_LATENT_DIM,
};
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, pool_latent, PtrmConfig, RecursionDepth, RecursionOpts,
    RecursionStepProbe, WorldModel, EVENT_GOAL_FAILED,
};
use crate::p2::representation::{
    RepresentationRowCollector, RepresentationSeam, RepresentationSeamCollector,
    RepresentationSeamMap, DEFAULT_REPRESENTATION_ROW_CAP,
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
use clap::ValueEnum;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const EVAL_REPORT_SCHEMA: &str = "p2.eval_report.v13";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EvalMode {
    Full,
    Representation,
    Rollout,
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_median: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_p95: Option<f64>,
    /// Mean of the worst 5% normalized errors (at least one row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_cvar95: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fraction_beating_copy: Option<f64>,
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
    #[serde(default = "default_eval_mode")]
    pub mode: EvalMode,
    #[serde(default = "default_representation_row_cap")]
    pub representation_row_cap: usize,
}

fn default_ensemble_members() -> usize {
    8
}

fn default_eval_mode() -> EvalMode {
    EvalMode::Full
}

fn default_representation_row_cap() -> usize {
    DEFAULT_REPRESENTATION_ROW_CAP
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
            mode: default_eval_mode(),
            representation_row_cap: default_representation_row_cap(),
        }
    }
}

impl EvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.physical_batch == 0 {
            bail!("physical_batch must be > 0");
        }
        if self.representation_row_cap == 0 {
            bail!("representation_row_cap must be > 0");
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
        } else {
            None
        };
        QMetrics {
            n: self.n,
            brier: Some(self.brier_sum / self.n as f64),
            accuracy: Some(self.correct as f64 / self.n as f64),
            positive_label_rate: Some(positive_label_rate),
            balanced_accuracy,
            saturated: !(0.1..=0.9).contains(&positive_label_rate),
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
pub struct ActionKindDiagnostics {
    /// ACTION1..ACTION5 and ACTION7 target rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simple: Option<ActionShuffleMetrics>,
    /// ACTION6 target rows, including their target coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate: Option<ActionShuffleMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionKindDiagnostics {
    /// Rows explicitly labeled as state-changing (`sample.noop == Some(false)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_transition: Option<ActionShuffleMetrics>,
    /// Rows explicitly labeled as no-ops (`sample.noop == Some(true)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noop: Option<ActionShuffleMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDiagnostics {
    pub aggregate: ActionSourceDiagnostics,
    pub by_source: BTreeMap<String, ActionSourceDiagnostics>,
    /// Primary deconfounded action intervention: only paired rows whose shuffled
    /// full `(id,x,y)` tuple differs from the true conditioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_conditioning_only: Option<ActionShuffleMetrics>,
    /// Paired action-shuffle errors grouped by the target transition's action ID.
    /// Missing from older reports; only source-local permutations with at least
    /// two rows contribute, matching `aggregate.shuffle`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_target_action_id: Option<BTreeMap<u8, ActionShuffleMetrics>>,
    /// Paired action-shuffle errors split by the target action form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_target_action_kind: Option<ActionKindDiagnostics>,
    /// Paired action-shuffle errors split by the transition label, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_transition_kind: Option<TransitionKindDiagnostics>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_seams: Option<RepresentationSeamMap>,
    pub changed_transitions: Option<ChangedTransitionMetrics>,
    pub identifiability: Option<IdentifiabilityMetrics>,
    pub events: Option<EventMetrics>,
    pub q: Option<QMetrics>,
    pub ptrm: Option<Vec<PtrmKMetrics>>,
    pub deterministic_matched_compute: Option<Vec<MatchedComputeMetrics>>,
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

/// Counts for a factual-branch population stratum. The strata use the exact
/// board-only effect relation carried by `FactualActionBranch`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactualBranchStratumCounts {
    pub branches: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub recoverable: usize,
    pub action6: usize,
}

/// Persisted branch-level evidence. Aggregate metrics must reconcile exactly
/// with these rows, making action/outcome confounds visible after a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualBranchRowMetric {
    pub row_index: usize,
    pub group_index: usize,
    pub group_key: String,
    pub family: String,
    pub action_id: u8,
    pub action_x: Option<u8>,
    pub action_y: Option<u8>,
    pub changed: bool,
    pub changed_cells: Vec<u16>,
    pub status_changed_cells: Vec<u16>,
    pub outcome_class: usize,
    pub recoverable: bool,
    pub predicted_displacement_norm: f64,
    pub predicted_action_id: Option<u8>,
    pub predicted_action_x_normalized: f32,
    pub predicted_action_y_normalized: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualBranchGroupMetric {
    pub group_index: usize,
    pub group_key: String,
    pub row_start: usize,
    pub row_end: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub outcome_classes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupBootstrapInterval {
    pub estimate: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub resamples: usize,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualBranchBootstrap {
    pub changed_norm_gap: Option<GroupBootstrapInterval>,
    pub action_recovery_top1: Option<GroupBootstrapInterval>,
}

/// Held-out same-state factual-action evaluation for world-core-v2 only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualBranchMetrics {
    /// Stable FNV-1a fingerprint of the generated, ordered branch population.
    pub population_fingerprint: String,
    pub groups: usize,
    pub branches: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub recoverable: usize,
    pub action6: usize,
    /// Anchors with at least one equivalent and one distinct candidate.
    pub outcome_equivalence_anchors: usize,
    /// Nearest-displacement outcome-equivalence accuracy on eligible anchors.
    pub outcome_equivalence_retrieval_accuracy: Option<f64>,
    /// Changed, board-effect-unique branches eligible for action recovery.
    pub unique_changed_effect_action_n: usize,
    pub unique_changed_effect_action_top1: Option<f64>,
    /// Recoverable ACTION6 branches with coordinate supervision.
    pub action6_coordinate_n: usize,
    pub action6_coordinate_rmse_normalized: Option<f64>,
    pub action6_coordinate_rmse_pixels: Option<f64>,
    pub changed_displacement_norm_mean: Option<f64>,
    pub unchanged_displacement_norm_mean: Option<f64>,
    pub changed_vs_unchanged_displacement_norm_auroc: Option<f64>,
    /// `changed_mean / unchanged_mean`, only when both means are supported.
    pub changed_to_unchanged_displacement_norm_ratio: Option<f64>,
    /// Eval-only standardized ridge probe. The decoder is fitted on a
    /// deterministic disjoint prefix and scored on the remaining factual rows.
    pub board_probe: Option<BoardTransitionMetrics>,
    pub by_family: BTreeMap<String, FactualBranchStratumCounts>,
    pub by_action_id: BTreeMap<u8, FactualBranchStratumCounts>,
    #[serde(default)]
    pub rows: Vec<FactualBranchRowMetric>,
    #[serde(default)]
    pub group_summaries: Vec<FactualBranchGroupMetric>,
    /// Exact reconciliation of persisted rows with all top-level counts.
    #[serde(default)]
    pub rows_reconciled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_bootstrap: Option<FactualBranchBootstrap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub majority_action_baseline_top1: Option<f64>,
}

/// Model-family-independent semantic probe on a deterministic held-out
/// synthetic transition population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardProbeEvaluation {
    pub population_fingerprint: String,
    pub fit_frames: usize,
    pub held_out_frames: usize,
    pub metrics: BoardTransitionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub schema: String,
    pub mode: EvalMode,
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
    /// Eval-only patch/palette grounding probe on the same model-independent
    /// population for every consumer-readout arm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board_probe: Option<BoardProbeEvaluation>,
    /// Held-out same-state factual branch metrics, available for world-core-v2
    /// checkpoints only.
    #[serde(default)]
    pub factual_branches: Option<FactualBranchMetrics>,
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

/// Immutable checkpoint-derived population used by bounded semantic-access audits.
/// Evaluation owns model loading and synthetic population construction so audit
/// decoders never gain access to trainer state or checkpoint gradients.
pub(crate) struct FrozenBoardProbePopulation {
    pub samples: Vec<TransitionSample>,
    pub source_by_sample: Vec<String>,
    pub target_rows: BoardProbeRows,
    pub predicted_rows: Option<BoardProbeRows>,
    /// Fingerprint before an audit-specific partition filter is applied.
    pub source_population_fingerprint: String,
    pub population_fingerprint: String,
}

/// Which rows a semantic-access process is permitted to encode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SemanticAccessPopulation {
    /// Preserve the historical fit population while excluding the old held-out rows.
    SelectionFit,
    /// Encode every row from a fresh, domain-separated population seed.
    FreshFinal,
}

pub(crate) fn collect_frozen_board_probe_population(
    checkpoint: &Path,
    train_config: &Path,
    seed: u64,
    synthetic_episodes: usize,
    physical_batch: usize,
    device_spec: &str,
) -> Result<FrozenBoardProbePopulation> {
    collect_frozen_board_probe_population_inner(
        checkpoint,
        train_config,
        seed,
        synthetic_episodes,
        physical_batch,
        device_spec,
        false,
        SemanticAccessPopulation::FreshFinal,
    )
}

pub(crate) fn collect_frozen_board_probe_population_with_predictions(
    checkpoint: &Path,
    train_config: &Path,
    seed: u64,
    synthetic_episodes: usize,
    physical_batch: usize,
    device_spec: &str,
) -> Result<FrozenBoardProbePopulation> {
    collect_frozen_board_probe_population_inner(
        checkpoint,
        train_config,
        seed,
        synthetic_episodes,
        physical_batch,
        device_spec,
        true,
        SemanticAccessPopulation::FreshFinal,
    )
}

pub(crate) fn collect_frozen_board_probe_population_partition_with_predictions(
    checkpoint: &Path,
    train_config: &Path,
    seed: u64,
    synthetic_episodes: usize,
    physical_batch: usize,
    device_spec: &str,
    population: SemanticAccessPopulation,
) -> Result<FrozenBoardProbePopulation> {
    collect_frozen_board_probe_population_inner(
        checkpoint,
        train_config,
        seed,
        synthetic_episodes,
        physical_batch,
        device_spec,
        true,
        population,
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_frozen_board_probe_population_inner(
    checkpoint: &Path,
    train_config: &Path,
    seed: u64,
    synthetic_episodes: usize,
    physical_batch: usize,
    device_spec: &str,
    include_predictions: bool,
    partition: SemanticAccessPopulation,
) -> Result<FrozenBoardProbePopulation> {
    let train_cfg = load_train_config(train_config)?;
    let device = resolve_device(device_spec)?;
    let (model, _varmap) = load_model(&train_cfg, checkpoint, &device)?;
    let mut sources = collect_synthetic_sources(
        seed,
        synthetic_episodes,
        &["random_one_step", "exploration"],
    )?;
    sources.push((
        "hazard_one_step".into(),
        collect_hazard_samples(seed, synthetic_episodes)?,
    ));
    let source_population_fingerprint = semantic_population_fingerprint(&flatten_sources(&sources));
    if partition == SemanticAccessPopulation::SelectionFit {
        for (_, samples) in &mut sources {
            samples.retain(|sample| sample.episode_id.is_multiple_of(3));
        }
    }
    let samples = flatten_sources(&sources);
    let source_by_sample = sources
        .iter()
        .flat_map(|(source, rows)| std::iter::repeat_n(source.clone(), rows.len()))
        .collect::<Vec<_>>();
    let (target_rows, predicted_rows) = if include_predictions {
        let (target, predicted) =
            board_probe_rows_for_samples(&model, &samples, physical_batch, &device)?;
        (target, Some(predicted))
    } else {
        let mut target_rows: Option<BoardProbeRows> = None;
        for (start, end) in batch_ranges(samples.len(), physical_batch) {
            let batch = batch_from_samples(&samples[start..end], &device)?;
            let rows =
                BoardProbeRows::from_spatial_latent(&model.encode_state(&batch.next_frames)?)?;
            if let Some(all) = &mut target_rows {
                all.append(rows);
            } else {
                target_rows = Some(rows);
            }
        }
        (
            target_rows.ok_or_else(|| anyhow::anyhow!("semantic-access population is empty"))?,
            None,
        )
    };
    Ok(FrozenBoardProbePopulation {
        source_population_fingerprint,
        population_fingerprint: semantic_population_fingerprint(&samples),
        samples,
        source_by_sample,
        target_rows,
        predicted_rows,
    })
}

fn semantic_population_fingerprint(samples: &[TransitionSample]) -> String {
    let mut digest = Sha256::new();
    for sample in samples {
        digest.update(sample.seed.to_le_bytes());
        digest.update(sample.episode_id.to_le_bytes());
        digest.update(sample.transition_index.to_le_bytes());
        digest.update((sample.family.len() as u64).to_le_bytes());
        digest.update(sample.family.as_bytes());
        digest.update([
            sample.action.id,
            sample.action.x.unwrap_or(u8::MAX),
            sample.action.y.unwrap_or(u8::MAX),
        ]);
        for frame in [&sample.current, &sample.next] {
            digest.update(frame.width.to_le_bytes());
            digest.update(frame.height.to_le_bytes());
            digest.update((frame.pixels.len() as u64).to_le_bytes());
            digest.update(&frame.pixels);
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

fn sample_population_fingerprint(samples: &[TransitionSample]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for sample in samples {
        for byte in sample
            .seed
            .to_le_bytes()
            .into_iter()
            .chain(sample.episode_id.to_le_bytes())
            .chain(sample.transition_index.to_le_bytes())
            .chain(sample.family.bytes())
            .chain([
                0xff,
                sample.action.id,
                sample.action.x.unwrap_or(u8::MAX),
                sample.action.y.unwrap_or(u8::MAX),
            ])
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn board_probe_rows_for_samples(
    model: &WorldModel,
    samples: &[TransitionSample],
    physical_batch: usize,
    device: &Device,
) -> Result<(BoardProbeRows, BoardProbeRows)> {
    let mut target_rows: Option<BoardProbeRows> = None;
    let mut predicted_rows: Option<BoardProbeRows> = None;
    for (start, end) in batch_ranges(samples.len(), physical_batch) {
        let batch = batch_from_samples(&samples[start..end], device)?;
        let target = BoardProbeRows::from_spatial_latent(&model.encode_state(&batch.next_frames)?)?;
        let current = model.encode_state(&batch.frames)?;
        let predicted = BoardProbeRows::from_spatial_latent(
            &model
                .training_latents_from_encoded_state(
                    &current,
                    &batch.actions,
                    &batch.action_coords,
                    RecursionDepth {
                        inner_steps: model.config().inner_steps,
                        outer_steps: model.config().outer_steps,
                    },
                    0.0,
                    None,
                    RecursionOpts::EVAL,
                )?
                .y,
        )?;
        if let Some(rows) = &mut target_rows {
            rows.append(target);
        } else {
            target_rows = Some(target);
        }
        if let Some(rows) = &mut predicted_rows {
            rows.append(predicted);
        } else {
            predicted_rows = Some(predicted);
        }
    }
    Ok((
        target_rows.ok_or_else(|| anyhow::anyhow!("board probe fit population is empty"))?,
        predicted_rows
            .ok_or_else(|| anyhow::anyhow!("board probe prediction population is empty"))?,
    ))
}

fn evaluate_board_probe(
    model: &WorldModel,
    samples: &[TransitionSample],
    physical_batch: usize,
    device: &Device,
) -> Result<Option<BoardProbeEvaluation>> {
    let (fit, held_out): (Vec<_>, Vec<_>) = samples
        .iter()
        .cloned()
        .partition(|sample| sample.episode_id % 3 == 0);
    if fit.is_empty() || held_out.is_empty() {
        return Ok(None);
    }
    let (fit_target_rows, _) = board_probe_rows_for_samples(model, &fit, physical_batch, device)?;
    let (held_out_target_rows, held_out_predicted_rows) =
        board_probe_rows_for_samples(model, &held_out, physical_batch, device)?;
    let probe = FixedBoardProbe::fit_spatial(
        &fit_target_rows,
        &fit.iter()
            .map(|sample| sample.next.clone())
            .collect::<Vec<_>>(),
    )?;
    let transitions = BoardProbeTransitions::try_new(
        held_out
            .iter()
            .map(|sample| sample.current.clone())
            .collect(),
        held_out.iter().map(|sample| sample.next.clone()).collect(),
    )?;
    Ok(Some(BoardProbeEvaluation {
        population_fingerprint: sample_population_fingerprint(samples),
        fit_frames: fit.len(),
        held_out_frames: held_out.len(),
        metrics: probe.summarize_transitions(
            &held_out_target_rows,
            &held_out_predicted_rows,
            &transitions,
        )?,
    }))
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

fn summarize_action_stratum(
    samples: &[TransitionSample],
    shuffled: &[TransitionSample],
    true_errors: &[f32],
    shuffled_errors: &[f32],
    indices: &[usize],
    seed: u64,
) -> Result<Option<ActionShuffleMetrics>> {
    if indices.is_empty() {
        return Ok(None);
    }
    let true_errors: Vec<_> = indices.iter().map(|&index| true_errors[index]).collect();
    let shuffled_errors: Vec<_> = indices
        .iter()
        .map(|&index| shuffled_errors[index])
        .collect();
    let changed_conditionings = indices
        .iter()
        .filter(|&&index| samples[index].action != shuffled[index].action)
        .count();
    Ok(Some(summarize_action_shuffle(
        &true_errors,
        &shuffled_errors,
        changed_conditionings,
        seed,
    )?))
}

fn summarize_action_strata(
    samples: &[TransitionSample],
    shuffled: &[TransitionSample],
    true_errors: &[f32],
    shuffled_errors: &[f32],
    paired: &[bool],
    seed: u64,
) -> Result<(
    BTreeMap<u8, ActionShuffleMetrics>,
    ActionKindDiagnostics,
    TransitionKindDiagnostics,
)> {
    if samples.len() != shuffled.len()
        || samples.len() != true_errors.len()
        || samples.len() != shuffled_errors.len()
        || samples.len() != paired.len()
    {
        bail!("action strata inputs must have matching lengths");
    }

    let mut by_action = BTreeMap::<u8, Vec<usize>>::new();
    let mut simple = Vec::new();
    let mut coordinate = Vec::new();
    let mut changed_transition = Vec::new();
    let mut noop = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        if !paired[index] {
            continue;
        }
        by_action.entry(sample.action.id).or_default().push(index);
        if sample.action.id == 6 {
            coordinate.push(index);
        } else {
            simple.push(index);
        }
        match sample.noop {
            Some(false) => changed_transition.push(index),
            Some(true) => noop.push(index),
            None => {}
        }
    }

    let by_target_action_id = by_action
        .into_iter()
        .map(|(action_id, indices)| {
            Ok((
                action_id,
                summarize_action_stratum(
                    samples,
                    shuffled,
                    true_errors,
                    shuffled_errors,
                    &indices,
                    seed.wrapping_add(action_id as u64).wrapping_add(0xA710),
                )?
                .expect("non-empty action stratum"),
            ))
        })
        .collect::<Result<_>>()?;
    Ok((
        by_target_action_id,
        ActionKindDiagnostics {
            simple: summarize_action_stratum(
                samples,
                shuffled,
                true_errors,
                shuffled_errors,
                &simple,
                seed.wrapping_add(0x0051_A1E3),
            )?,
            coordinate: summarize_action_stratum(
                samples,
                shuffled,
                true_errors,
                shuffled_errors,
                &coordinate,
                seed.wrapping_add(0xC00D),
            )?,
        },
        TransitionKindDiagnostics {
            changed_transition: summarize_action_stratum(
                samples,
                shuffled,
                true_errors,
                shuffled_errors,
                &changed_transition,
                seed.wrapping_add(0xC4A6ED),
            )?,
            noop: summarize_action_stratum(
                samples,
                shuffled,
                true_errors,
                shuffled_errors,
                &noop,
                seed.wrapping_add(0x0A00),
            )?,
        },
    ))
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
    let mut normalized_sorted = normalized
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    normalized_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail_start = ((normalized_sorted.len() as f64 * 0.95).floor() as usize)
        .min(normalized_sorted.len().saturating_sub(1));
    let normalized_cvar95 = (!normalized_sorted.is_empty())
        .then(|| mean(&normalized_sorted[tail_start..]))
        .flatten();
    let fraction_beating_copy = (!normalized_sorted.is_empty()).then(|| {
        normalized_sorted
            .iter()
            .filter(|value| **value <= 1.0)
            .count() as f64
            / normalized_sorted.len() as f64
    });
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
        normalized_median: median(&normalized_sorted),
        normalized_p95: percentile(&normalized_sorted, 0.95),
        normalized_cvar95,
        fraction_beating_copy,
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

#[cfg(test)]
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

#[cfg(test)]
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

fn summarize_representation_from_seam(
    seam: &crate::p2::representation::RepresentationSeamMetrics,
    sigreg_raw_sum: f64,
    sigreg_bounded_sum: f64,
    sigreg_n: usize,
) -> RepresentationDiagnostics {
    let sigreg_raw = (sigreg_n > 0).then_some(sigreg_raw_sum / sigreg_n as f64);
    let sigreg_bounded = (sigreg_n > 0).then_some(sigreg_bounded_sum / sigreg_n as f64);
    RepresentationDiagnostics {
        sigreg_raw,
        sigreg_bounded,
        sigreg_bound: SIGREG_BOUND,
        sigreg_near_bound: sigreg_bounded
            .map(|value| value >= SIGREG_BOUND * SIGREG_NEAR_BOUND_FRACTION),
        encoder_rows: seam.rows_used,
        encoder_dim: seam.dimension,
        mean_encoder_variance: seam.mean_variance,
        effective_rank: seam.effective_rank,
        effective_rank_fraction: seam.effective_rank_fraction,
        min_mean_variance: ENCODER_MIN_MEAN_VARIANCE,
        min_effective_rank_fraction: ENCODER_MIN_EFFECTIVE_RANK_FRACTION,
        noncollapse_pass: seam.mean_variance.map(|variance| {
            variance >= ENCODER_MIN_MEAN_VARIANCE
                && seam.effective_rank_fraction.unwrap_or(0.0)
                    >= ENCODER_MIN_EFFECTIVE_RANK_FRACTION
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
    representation_tensors: BTreeMap<RepresentationSeam, Tensor>,
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
    let out = (cfg.mode == EvalMode::Full)
        .then(|| {
            model.forward(
                &batch.frames,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
            )
        })
        .transpose()?;
    let diagnostic = model.representation_diagnostic(
        &batch.frames,
        &batch.next_frames,
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
    let mses = match &out {
        Some(out) => per_sample_mse(&out.y, &next_z)?,
        None => per_sample_mse(
            diagnostic
                .seams
                .get(&RepresentationSeam::PredictionFinalSpatial)
                .expect("diagnostic forward always captures the final prediction"),
            &next_z,
        )?,
    };
    let copy_forward_mses = per_sample_mse(&current_z, &next_z)?;
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
        representation_tensors: diagnostic.seams,
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

    if cfg.mode == EvalMode::Full {
        let out = out.as_ref().expect("full mode runs the observer forward");
        let (lab, acc, bce) =
            eval_events(&out.event_logits, &batch.event_targets, &batch.event_mask)?;
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
        partial
            .recursion_probes
            .extend(out.recursion_probes.clone());

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
    }
    Ok(partial)
}

fn merge_batch_partial(merged: &mut BatchEvalPartial, partial: BatchEvalPartial) {
    merged.mse_all.extend(partial.mse_all);
    merged.encoder_embeddings.extend(partial.encoder_embeddings);
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

fn action_diagnostics_from_pairs(
    samples: &[TransitionSample],
    shuffled: &[TransitionSample],
    true_errors: &[f32],
    shuffled_errors: &[f32],
    source_ranges: &[(String, usize, usize)],
    seed: u64,
) -> Result<ActionDiagnostics> {
    if samples.len() != shuffled.len()
        || samples.len() != true_errors.len()
        || samples.len() != shuffled_errors.len()
    {
        bail!("action diagnostics pair inputs must have matching lengths");
    }

    let mut aggregate_true = Vec::new();
    let mut aggregate_shuffled = Vec::new();
    let mut aggregate_changed = 0usize;
    let mut by_source = BTreeMap::new();
    let mut paired = vec![false; samples.len()];
    for (source_index, (name, start, end)) in source_ranges.iter().enumerate() {
        if start > end || *end > samples.len() {
            bail!("action diagnostic source range is outside the sample population");
        }
        let valid_shuffle = end.saturating_sub(*start) >= 2;
        let true_slice = if valid_shuffle {
            &true_errors[*start..*end]
        } else {
            &[]
        };
        let shuffled_slice = if valid_shuffle {
            &shuffled_errors[*start..*end]
        } else {
            &[]
        };
        if valid_shuffle {
            for paired in &mut paired[*start..*end] {
                *paired = true;
            }
        }
        let changed_conditionings = if valid_shuffle {
            samples[*start..*end]
                .iter()
                .zip(shuffled[*start..*end].iter())
                .filter(|(truth, ablated)| truth.action != ablated.action)
                .count()
        } else {
            0
        };
        aggregate_true.extend_from_slice(true_slice);
        aggregate_shuffled.extend_from_slice(shuffled_slice);
        aggregate_changed += changed_conditionings;
        by_source.insert(
            name.clone(),
            ActionSourceDiagnostics {
                shuffle: summarize_action_shuffle(
                    true_slice,
                    shuffled_slice,
                    changed_conditionings,
                    seed.wrapping_add(source_index as u64).wrapping_add(0xB005),
                )?,
                coverage: action_coverage(&samples[*start..*end]),
            },
        );
    }
    let (by_target_action_id, by_target_action_kind, by_transition_kind) = summarize_action_strata(
        samples,
        shuffled,
        true_errors,
        shuffled_errors,
        &paired,
        seed.wrapping_add(0x0005_7A7A),
    )?;
    let changed_conditioning_indices = paired
        .iter()
        .enumerate()
        .filter_map(|(index, &is_paired)| {
            (is_paired && samples[index].action != shuffled[index].action).then_some(index)
        })
        .collect::<Vec<_>>();
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
        changed_conditioning_only: summarize_action_stratum(
            samples,
            shuffled,
            true_errors,
            shuffled_errors,
            &changed_conditioning_indices,
            seed.wrapping_add(0xC4A6_710A),
        )?,
        by_target_action_id: Some(by_target_action_id),
        by_target_action_kind: Some(by_target_action_kind),
        by_transition_kind: Some(by_transition_kind),
    })
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

    action_diagnostics_from_pairs(
        samples,
        &shuffled,
        true_errors,
        &shuffled_errors,
        &source_ranges,
        seed,
    )
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

fn empty_split(source: &str, n_samples: usize) -> SplitEval {
    SplitEval {
        source: source.into(),
        n_samples,
        one_step_latent_mse: None,
        representation: None,
        representation_seams: None,
        changed_transitions: None,
        identifiability: None,
        events: None,
        q: None,
        ptrm: None,
        deterministic_matched_compute: None,
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
    }
}

fn collect_top_level_representation(
    collector: &mut RepresentationRowCollector,
    tensors: &BTreeMap<RepresentationSeam, Tensor>,
    sample_start: usize,
    sample_count: usize,
) -> Result<()> {
    let current = tensors
        .get(&RepresentationSeam::EncoderPostRmsPooled)
        .expect("diagnostic forward always captures the current post-RMS pooled seam");
    let target = tensors
        .get(&RepresentationSeam::TargetPostRmsPooled)
        .expect("diagnostic forward always captures the target post-RMS pooled seam");
    let current_rows = current.dim(0)?;
    let target_rows = target.dim(0)?;
    if current_rows != sample_count || target_rows != sample_count {
        bail!("pooled representation seams must have one row per sample");
    }
    let pooled = Tensor::cat(&[current, target], 0)?;
    let row_ids = (sample_start..sample_start + sample_count)
        .map(|sample| sample as u64 * 2)
        .chain((sample_start..sample_start + sample_count).map(|sample| sample as u64 * 2 + 1))
        .collect();
    collector.collect_rows(&pooled, row_ids)
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
        return Ok(empty_split(source, 0));
    }

    let batch_size = cfg.physical_batch.max(1);
    let ranges = batch_ranges(samples.len(), batch_size);
    let mut merged = BatchEvalPartial::default();
    let mut seam_collector = RepresentationSeamCollector::new(cfg.seed, cfg.representation_row_cap);
    // Historical `SplitEval.representation` pooled current and target post-RMS latents.
    // This separate internal collector preserves that population while named seams remain split.
    let mut top_level_collector = RepresentationRowCollector::new(
        cfg.seed,
        0x504F_4F4C_4544_5F5A,
        cfg.representation_row_cap,
    );
    for (bi, &(start, end)) in ranges.iter().enumerate() {
        let partial = if device.is_cpu() {
            with_thread_local_model(train_cfg, checkpoint, device, |m| {
                eval_one_batch(m, &samples[start..end], bi, train_cfg, cfg, device)
            })?
        } else {
            eval_one_batch(model, &samples[start..end], bi, train_cfg, cfg, device)?
        };
        seam_collector.collect_batch(&partial.representation_tensors, start, end - start)?;
        collect_top_level_representation(
            &mut top_level_collector,
            &partial.representation_tensors,
            start,
            end - start,
        )?;
        merge_batch_partial(&mut merged, partial);
        if device.is_cuda() {
            device.synchronize()?;
        }
    }

    let representation_seams = seam_collector.summarize()?;
    let post_rms_pooled = top_level_collector.summarize()?;
    let representation = Some(summarize_representation_from_seam(
        &post_rms_pooled,
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

    let events = if cfg.mode != EvalMode::Full {
        None
    } else if event_labeled == 0 {
        Some(EventMetrics {
            labeled: 0,
            accuracy: None,
            bce: None,
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: None,
        })
    } else {
        Some(EventMetrics {
            labeled: event_labeled,
            accuracy: Some(event_correct / event_labeled as f64),
            bce: Some(event_bce / event_labeled as f64),
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: (hazard_failure_labeled > 0)
                .then_some(hazard_false_negatives as f64 / hazard_failure_labeled as f64),
        })
    };
    let q = (cfg.mode == EvalMode::Full).then(|| q_acc.finalize());
    let ptrm: Vec<_> = ptrm_acc
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
    let ptrm = (cfg.mode == EvalMode::Full).then_some(ptrm);
    let deterministic_matched_compute: Vec<_> = matched_acc
        .into_iter()
        .map(|(k, (sum, n, outer_steps))| MatchedComputeMetrics {
            ptrm_k_equivalent: k,
            outer_steps,
            n,
            one_step_latent_mse: (n > 0).then_some(sum / n as f64),
        })
        .collect();
    let deterministic_matched_compute =
        (cfg.mode == EvalMode::Full).then_some(deterministic_matched_compute);

    let q_surprise = (cfg.mode == EvalMode::Full)
        .then(|| eval_q_surprise(&merged.q_probs, &mse_all, cfg.q_mse_threshold));
    let labels: Vec<bool> = mse_all
        .iter()
        .map(|m| f64::from(*m) <= cfg.q_mse_threshold)
        .collect();
    let (calibration, calibration_gates) = if cfg.mode == EvalMode::Full
        && merged.reliability_probs.len() == labels.len()
        && !labels.is_empty()
    {
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
    let recursion_probes = (cfg.mode == EvalMode::Full)
        .then(|| summarize_recursion_probes(&merged.recursion_probes))
        .flatten();
    let ensemble = (cfg.mode == EvalMode::Full && merged.ensemble_n > 0).then(|| {
        let mean_disagreement = Some(merged.ensemble_disagreement / merged.ensemble_n as f64);
        let high_error: Vec<bool> = mse_all
            .iter()
            .take(merged.ensemble_n.min(mse_all.len()))
            .map(|m| f64::from(*m) > cfg.q_mse_threshold)
            .collect();
        let uncertainty: Vec<f32> = (0..high_error.len())
            .map(|index| merged.reliability_probs.get(index).copied().unwrap_or(0.5))
            .collect();
        EnsembleMetrics {
            members: cfg.ensemble_members,
            mean_disagreement,
            uncertainty_auroc: if uncertainty.len() == high_error.len() && !high_error.is_empty() {
                binary_auroc(
                    &uncertainty
                        .iter()
                        .map(|value| 1.0 - value)
                        .collect::<Vec<_>>(),
                    &high_error,
                )
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
        representation_seams: Some(representation_seams),
        changed_transitions,
        identifiability,
        events,
        q,
        ptrm,
        deterministic_matched_compute,
        rollout,
        closed_loop,
        copy_forward: None,
        q_surprise,
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

const FACTUAL_BRANCH_EVAL_DOMAIN: &[u8] = b"p2.factual_branch_eval.v1";

fn fnv1a_append(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn factual_branch_eval_seed(seed: u64) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325;
    fnv1a_append(&mut hash, FACTUAL_BRANCH_EVAL_DOMAIN);
    fnv1a_append(&mut hash, &seed.to_le_bytes());
    hash
}

fn factual_population_fingerprint(groups: &[BranchGroup], synthetic_episodes: usize) -> String {
    let mut hash = 0xCBF2_9CE4_8422_2325;
    fnv1a_append(&mut hash, FACTUAL_BRANCH_EVAL_DOMAIN);
    fnv1a_append(&mut hash, &(synthetic_episodes as u64).to_le_bytes());
    for group in groups {
        fnv1a_append(&mut hash, &(group.branches().len() as u64).to_le_bytes());
        for branch in group.branches() {
            let transition = &branch.transition;
            fnv1a_append(&mut hash, &transition.seed.to_le_bytes());
            fnv1a_append(&mut hash, &transition.episode_id.to_le_bytes());
            fnv1a_append(&mut hash, transition.family.as_bytes());
            fnv1a_append(&mut hash, &[transition.action.id]);
            fnv1a_append(&mut hash, &[transition.action.x.unwrap_or(u8::MAX)]);
            fnv1a_append(&mut hash, &[transition.action.y.unwrap_or(u8::MAX)]);
            fnv1a_append(&mut hash, &transition.current.pixels);
            fnv1a_append(&mut hash, &transition.next.pixels);
            fnv1a_append(&mut hash, &[u8::from(branch.board_effect.changed)]);
            for cell in &branch.board_effect.changed_cells {
                fnv1a_append(&mut hash, &cell.to_le_bytes());
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn increment_factual_stratum(
    counts: &mut FactualBranchStratumCounts,
    changed: bool,
    recoverable: bool,
    action_id: u8,
) {
    counts.branches += 1;
    if changed {
        counts.changed += 1;
    } else {
        counts.unchanged += 1;
    }
    if recoverable {
        counts.recoverable += 1;
    }
    if action_id == 6 {
        counts.action6 += 1;
    }
}

fn mean_or_none(values: &[f32]) -> Option<f64> {
    (!values.is_empty())
        .then(|| values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64)
}

fn squared_distance(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = f64::from(*left) - f64::from(*right);
            delta * delta
        })
        .sum()
}

fn factual_changed_norm_gap(rows: &[&FactualBranchRowMetric]) -> Option<f64> {
    let changed = rows
        .iter()
        .filter(|row| row.changed)
        .map(|row| row.predicted_displacement_norm)
        .collect::<Vec<_>>();
    let unchanged = rows
        .iter()
        .filter(|row| !row.changed)
        .map(|row| row.predicted_displacement_norm)
        .collect::<Vec<_>>();
    (!changed.is_empty() && !unchanged.is_empty()).then(|| {
        changed.iter().sum::<f64>() / changed.len() as f64
            - unchanged.iter().sum::<f64>() / unchanged.len() as f64
    })
}

fn factual_action_top1(rows: &[&FactualBranchRowMetric]) -> Option<f64> {
    let eligible = rows
        .iter()
        .filter(|row| row.recoverable)
        .collect::<Vec<_>>();
    (!eligible.is_empty()).then(|| {
        eligible
            .iter()
            .filter(|row| row.predicted_action_id == Some(row.action_id))
            .count() as f64
            / eligible.len() as f64
    })
}

fn group_bootstrap_interval(
    rows: &[FactualBranchRowMetric],
    groups: &[FactualBranchGroupMetric],
    seed: u64,
    metric: fn(&[&FactualBranchRowMetric]) -> Option<f64>,
) -> Option<GroupBootstrapInterval> {
    const RESAMPLES: usize = 1000;
    let observed_rows = rows.iter().collect::<Vec<_>>();
    let estimate = metric(&observed_rows)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut sampled = Vec::new();
        for _ in 0..groups.len() {
            let group = &groups[rng.random_range(0..groups.len())];
            sampled.extend(rows[group.row_start..group.row_end].iter());
        }
        if let Some(value) = metric(&sampled) {
            values.push(value);
        }
    }
    if values.len() < RESAMPLES * 9 / 10 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let lower = values[((values.len() as f64 * 0.025).floor() as usize).min(values.len() - 1)];
    let upper = values[((values.len() as f64 * 0.975).floor() as usize).min(values.len() - 1)];
    Some(GroupBootstrapInterval {
        estimate,
        lower_95: lower,
        upper_95: upper,
        resamples: values.len(),
        unit: "branch_group".into(),
    })
}

fn evaluate_factual_branches(
    model: &WorldModel,
    cfg: &EvalConfig,
    device: &Device,
) -> Result<FactualBranchMetrics> {
    let seed = factual_branch_eval_seed(cfg.seed);
    // The generic evaluation count controls expensive rollouts. Factual
    // branches are cheap one-step groups, so use four times as many to keep the
    // frozen overnight population at 256 groups when synthetic_episodes=64.
    let factual_group_count = cfg.synthetic_episodes.saturating_mul(4);
    if factual_group_count == 0 {
        return Ok(FactualBranchMetrics {
            population_fingerprint: factual_population_fingerprint(&[], 0),
            groups: 0,
            branches: 0,
            changed: 0,
            unchanged: 0,
            recoverable: 0,
            action6: 0,
            outcome_equivalence_anchors: 0,
            outcome_equivalence_retrieval_accuracy: None,
            unique_changed_effect_action_n: 0,
            unique_changed_effect_action_top1: None,
            action6_coordinate_n: 0,
            action6_coordinate_rmse_normalized: None,
            action6_coordinate_rmse_pixels: None,
            changed_displacement_norm_mean: None,
            unchanged_displacement_norm_mean: None,
            changed_vs_unchanged_displacement_norm_auroc: None,
            changed_to_unchanged_displacement_norm_ratio: None,
            board_probe: None,
            by_family: BTreeMap::new(),
            by_action_id: BTreeMap::new(),
            rows: Vec::new(),
            group_summaries: Vec::new(),
            rows_reconciled: true,
            group_bootstrap: None,
            majority_action_baseline_top1: None,
        });
    }
    let generated_groups = (0..factual_group_count)
        .map(|episode| {
            generate_factual_branch_group(seed, episode as u64, Split::HeldOutComposition)
        })
        .collect::<Result<Vec<_>>>()?;
    // Canonicalize once, before inference and metric construction. Rebuilding
    // batches independently would reorder branch actions for model input while
    // leaving labels in generator order, silently misaligning evidence rows.
    let factual_batch = FactualBatch::from_groups(generated_groups)?;
    let groups = factual_batch.groups().to_vec();
    let population_fingerprint = factual_population_fingerprint(&groups, factual_group_count);
    let samples = factual_batch.rows().to_vec();

    let mut predicted_displacements = Vec::with_capacity(samples.len());
    let mut action_logits = Vec::with_capacity(samples.len());
    let mut coordinate_predictions = Vec::with_capacity(samples.len());
    let mut target_patch_latents: Option<BoardProbeRows> = None;
    let mut predicted_patch_latents: Option<BoardProbeRows> = None;
    let factual_eval_batch = cfg
        .physical_batch
        .max(crate::p2::data::FACTUAL_BRANCHES_PER_GROUP)
        / crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
        * crate::p2::data::FACTUAL_BRANCHES_PER_GROUP;
    for (start, end) in batch_ranges(samples.len(), factual_eval_batch) {
        let batch = batch_from_samples(&samples[start..end], device)?;
        let current = model.encode_state(&batch.frames)?;
        let output = model.forward(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
        )?;
        let target = model.encode_state(&batch.next_frames)?;
        let displacement = pool_latent(&output.y.sub(&current)?)?;
        let (decoded_actions, decoded_coordinates) =
            model.decode_action_displacement(&displacement)?;
        predicted_displacements.extend(displacement.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        action_logits.extend(decoded_actions.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        coordinate_predictions.extend(decoded_coordinates.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        let target_rows = BoardProbeRows::from_spatial_latent(&target)?;
        let predicted_rows = BoardProbeRows::from_spatial_latent(&output.y)?;
        if let Some(rows) = &mut target_patch_latents {
            rows.append(target_rows);
        } else {
            target_patch_latents = Some(target_rows);
        }
        if let Some(rows) = &mut predicted_patch_latents {
            rows.append(predicted_rows);
        } else {
            predicted_patch_latents = Some(predicted_rows);
        }
    }

    let board_probe = if groups.len() >= 2 {
        // Split only between same-state groups. Cutting a four-branch group
        // would leak the shared current state (and sometimes an equivalent
        // outcome) from decoder fit into the held-out score.
        let fit_group_count = (groups.len() / 3).max(1).min(groups.len() - 1);
        let fit_samples = groups[..fit_group_count]
            .iter()
            .map(|group| group.branches().len())
            .sum::<usize>();
        let fit_rows = fit_samples * PATCH_COUNT;
        let target_patch_latents = target_patch_latents
            .as_ref()
            .expect("non-empty factual rows");
        let predicted_patch_latents = predicted_patch_latents
            .as_ref()
            .expect("non-empty factual rows");
        let probe = FixedBoardProbe::fit_spatial(
            &target_patch_latents.slice(0..fit_rows)?,
            &samples[..fit_samples]
                .iter()
                .map(|sample| sample.next.clone())
                .collect::<Vec<_>>(),
        )?;
        let held_out = &samples[fit_samples..];
        let transitions = BoardProbeTransitions::try_new(
            held_out
                .iter()
                .map(|sample| sample.current.clone())
                .collect(),
            held_out.iter().map(|sample| sample.next.clone()).collect(),
        )?;
        Some(probe.summarize_transitions(
            &target_patch_latents.slice(fit_rows..target_patch_latents.as_rows().len())?,
            &predicted_patch_latents.slice(fit_rows..predicted_patch_latents.as_rows().len())?,
            &transitions,
        )?)
    } else {
        None
    };

    let mut changed_norms = Vec::new();
    let mut unchanged_norms = Vec::new();
    let mut norm_scores = Vec::new();
    let mut norm_labels = Vec::new();
    let mut outcome_equivalence_anchors = 0usize;
    let mut outcome_equivalence_correct = 0usize;
    let mut unique_changed_effect_action_n = 0usize;
    let mut unique_changed_effect_action_correct = 0usize;
    let mut action6_coordinate_n = 0usize;
    let mut action6_coordinate_sum_squared = 0f64;
    let mut counts = FactualBranchStratumCounts::default();
    let mut by_family = BTreeMap::<String, FactualBranchStratumCounts>::new();
    let mut by_action_id = BTreeMap::<u8, FactualBranchStratumCounts>::new();
    let mut rows = Vec::with_capacity(samples.len());
    let mut group_summaries = Vec::with_capacity(groups.len());
    let mut offset = 0usize;
    for (group_index, group) in groups.iter().enumerate() {
        let branches = group.branches();
        let source = &branches[0].transition;
        let group_key = format!(
            "{}:{}:{}:{}",
            source.seed,
            source.episode_id,
            source.family,
            crate::p2::data::BranchGroupId::from_transition_for_eval(source).current_fingerprint
        );
        let recoverable = group
            .unique_changed_effect_indices()
            .into_iter()
            .collect::<BTreeSet<_>>();
        for (local, branch) in branches.iter().enumerate() {
            let global = offset + local;
            let changed = branch.board_effect.changed;
            let is_recoverable = recoverable.contains(&local);
            increment_factual_stratum(
                &mut counts,
                changed,
                is_recoverable,
                branch.transition.action.id,
            );
            increment_factual_stratum(
                by_family
                    .entry(branch.transition.family.clone())
                    .or_default(),
                changed,
                is_recoverable,
                branch.transition.action.id,
            );
            increment_factual_stratum(
                by_action_id.entry(branch.transition.action.id).or_default(),
                changed,
                is_recoverable,
                branch.transition.action.id,
            );

            let norm = (predicted_displacements[global]
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                / predicted_displacements[global].len().max(1) as f64)
                .sqrt() as f32;
            let predicted_action = action_logits[global]
                .iter()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.partial_cmp(right)
                        .unwrap_or_else(|| left_index.cmp(right_index))
                })
                .map(|(index, _)| index as u8);
            let outcome_class = branches
                .iter()
                .position(|candidate| branch.outcome_equivalent(candidate))
                .expect("branch is equivalent to itself");
            rows.push(FactualBranchRowMetric {
                row_index: global,
                group_index,
                group_key: group_key.clone(),
                family: branch.transition.family.clone(),
                action_id: branch.transition.action.id,
                action_x: branch.transition.action.x,
                action_y: branch.transition.action.y,
                changed,
                changed_cells: branch.board_effect.changed_cells.clone(),
                status_changed_cells: branch.status_changed_cells.clone(),
                outcome_class,
                recoverable: is_recoverable,
                predicted_displacement_norm: f64::from(norm),
                predicted_action_id: predicted_action,
                predicted_action_x_normalized: coordinate_predictions[global][0],
                predicted_action_y_normalized: coordinate_predictions[global][1],
            });
            norm_scores.push(norm);
            norm_labels.push(changed);
            if changed {
                changed_norms.push(norm);
            } else {
                unchanged_norms.push(norm);
            }

            let equivalent = branches
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != local)
                .filter(|(_, other)| branch.outcome_equivalent(other))
                .count();
            let distinct = branches.len().saturating_sub(1 + equivalent);
            if equivalent > 0 && distinct > 0 {
                outcome_equivalence_anchors += 1;
                let nearest = branches
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != local)
                    .min_by(|(left_index, _), (right_index, _)| {
                        squared_distance(
                            &predicted_displacements[global],
                            &predicted_displacements[offset + *left_index],
                        )
                        .partial_cmp(&squared_distance(
                            &predicted_displacements[global],
                            &predicted_displacements[offset + *right_index],
                        ))
                        .unwrap_or_else(|| left_index.cmp(right_index))
                    })
                    .expect("eligible factual anchor has a candidate");
                if branch.outcome_equivalent(nearest.1) {
                    outcome_equivalence_correct += 1;
                }
            }

            if is_recoverable {
                unique_changed_effect_action_n += 1;
                if predicted_action == Some(branch.transition.action.id) {
                    unique_changed_effect_action_correct += 1;
                }
                if branch.transition.action.id == 6 {
                    let action = &branch.transition.action;
                    let expected = [
                        f64::from(action.x.expect("recoverable ACTION6 x")) / 63.0,
                        f64::from(action.y.expect("recoverable ACTION6 y")) / 63.0,
                    ];
                    action6_coordinate_sum_squared += coordinate_predictions[global]
                        .iter()
                        .zip(expected)
                        .map(|(predicted, expected)| {
                            let delta = f64::from(*predicted) - expected;
                            delta * delta
                        })
                        .sum::<f64>();
                    action6_coordinate_n += 1;
                }
            }
        }
        let changed = branches
            .iter()
            .filter(|branch| branch.board_effect.changed)
            .count();
        let outcome_classes = branches
            .iter()
            .enumerate()
            .filter(|(index, branch)| {
                !branches[..*index]
                    .iter()
                    .any(|previous| branch.outcome_equivalent(previous))
            })
            .count();
        group_summaries.push(FactualBranchGroupMetric {
            group_index,
            group_key,
            row_start: offset,
            row_end: offset + branches.len(),
            changed,
            unchanged: branches.len() - changed,
            outcome_classes,
        });
        offset += branches.len();
    }
    let changed_displacement_norm_mean = mean_or_none(&changed_norms);
    let unchanged_displacement_norm_mean = mean_or_none(&unchanged_norms);
    let action6_coordinate_rmse_normalized = (action6_coordinate_n > 0)
        .then(|| (action6_coordinate_sum_squared / (action6_coordinate_n * 2) as f64).sqrt());
    let group_bootstrap = Some(FactualBranchBootstrap {
        changed_norm_gap: group_bootstrap_interval(
            &rows,
            &group_summaries,
            seed ^ 0xB007_0001,
            factual_changed_norm_gap,
        ),
        action_recovery_top1: group_bootstrap_interval(
            &rows,
            &group_summaries,
            seed ^ 0xB007_0002,
            factual_action_top1,
        ),
    });
    let recoverable_action_counts = rows.iter().filter(|row| row.recoverable).fold(
        BTreeMap::<u8, usize>::new(),
        |mut counts, row| {
            *counts.entry(row.action_id).or_default() += 1;
            counts
        },
    );
    let majority_action_baseline_top1 = (!recoverable_action_counts.is_empty()).then(|| {
        recoverable_action_counts
            .values()
            .copied()
            .max()
            .unwrap_or(0) as f64
            / recoverable_action_counts.values().sum::<usize>() as f64
    });
    Ok(FactualBranchMetrics {
        population_fingerprint,
        groups: groups.len(),
        branches: counts.branches,
        changed: counts.changed,
        unchanged: counts.unchanged,
        recoverable: counts.recoverable,
        action6: counts.action6,
        outcome_equivalence_anchors,
        outcome_equivalence_retrieval_accuracy: (outcome_equivalence_anchors > 0)
            .then_some(outcome_equivalence_correct as f64 / outcome_equivalence_anchors as f64),
        unique_changed_effect_action_n,
        unique_changed_effect_action_top1: (unique_changed_effect_action_n > 0).then_some(
            unique_changed_effect_action_correct as f64 / unique_changed_effect_action_n as f64,
        ),
        action6_coordinate_n,
        action6_coordinate_rmse_normalized,
        action6_coordinate_rmse_pixels: action6_coordinate_rmse_normalized
            .map(|value| value * 63.0),
        changed_displacement_norm_mean,
        unchanged_displacement_norm_mean,
        changed_vs_unchanged_displacement_norm_auroc: binary_auroc(&norm_scores, &norm_labels),
        changed_to_unchanged_displacement_norm_ratio: changed_displacement_norm_mean
            .zip(unchanged_displacement_norm_mean)
            .and_then(|(changed, unchanged)| {
                (unchanged > f64::EPSILON).then_some(changed / unchanged)
            }),
        board_probe,
        by_family,
        by_action_id,
        rows_reconciled: rows.len() == counts.branches
            && rows.iter().filter(|row| row.changed).count() == counts.changed
            && rows.iter().filter(|row| !row.changed).count() == counts.unchanged
            && rows.iter().filter(|row| row.recoverable).count() == counts.recoverable,
        rows,
        group_summaries,
        group_bootstrap,
        majority_action_baseline_top1,
    })
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
    let factual_branches = train_cfg
        .world_core_v2
        .then(|| evaluate_factual_branches(&model, cfg, &device))
        .transpose()?;

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
    let board_probe = evaluate_board_probe(&model, &dynamics_samples, cfg.physical_batch, &device)?;
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

    let mut synthetic_dynamics = if cfg.mode == EvalMode::Rollout {
        empty_split("synthetic_dynamics", dynamics_samples.len())
    } else {
        eval_sample_set(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &dynamics_samples,
            "synthetic_dynamics",
            Some(&dynamics_source_lengths),
            cfg,
            &device,
            false,
        )?
    };
    let dynamics_rollout_rows = if cfg.mode != EvalMode::Representation {
        eval_episode_rollouts(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &dynamics_rollout_samples,
            &device,
            "synthetic_dynamics",
        )?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
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
    }
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

    let mut synthetic_planner = if cfg.mode == EvalMode::Rollout {
        empty_split("synthetic_planner", planner_samples.len())
    } else {
        eval_sample_set(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &planner_samples,
            "synthetic_planner",
            Some(&planner_source_lengths),
            cfg,
            &device,
            false,
        )?
    };
    let planner_rollout_rows = if cfg.mode != EvalMode::Representation {
        eval_episode_rollouts(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &planner_rollout_samples,
            &device,
            "synthetic_planner",
        )?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
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
    }
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

    let arc3_transfer = if cfg.mode != EvalMode::Rollout {
        if let Some(dir) = &cfg.arc_recordings_dir {
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
        }
    } else {
        None
    };

    let arc3_recording_runs = if cfg.mode == EvalMode::Full {
        if let Some(dir) = &cfg.arc_recordings_dir {
            let runs = summarize_recordings_dir(dir)?;
            Some(Arc3RecordingBenchmark {
                n_runs: runs.len(),
                total_actions: runs.iter().map(|r| r.actions).sum(),
                total_levels_completed: runs.iter().map(|r| r.levels_completed).sum(),
                runs,
            })
        } else {
            None
        }
    } else {
        None
    };

    let official_scorecard = if cfg.mode == EvalMode::Full {
        if let Some(path) = &cfg.scorecard_json {
            Some(benchmark_from_scorecard_json(path)?)
        } else {
            None
        }
    } else {
        None
    };
    let official_rhae = official_scorecard
        .as_ref()
        .and_then(official_rhae_from_benchmark);

    let report = EvalReport {
        schema: EVAL_REPORT_SCHEMA.into(),
        mode: cfg.mode,
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
        board_probe,
        factual_branches,
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

    #[test]
    fn q_metrics_require_both_classes_and_reject_extreme_label_rates() {
        let all_negative = QEvalAccum {
            n: 10,
            correct: 10,
            tn: 10,
            ..Default::default()
        }
        .finalize();
        assert!(all_negative.saturated);
        assert_eq!(all_negative.balanced_accuracy, None);

        let both_classes = QEvalAccum {
            n: 10,
            correct: 8,
            positive_labels: 4,
            tp: 3,
            tn: 5,
            fp: 1,
            fn_: 1,
            ..Default::default()
        }
        .finalize();
        assert!(!both_classes.saturated);
        assert!(both_classes.balanced_accuracy.is_some());
    }

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
    fn action_diagnostic_strata_partition_paired_target_rows_without_changing_aggregates(
    ) -> Result<()> {
        let mut samples = vec![
            action_diagnostic_sample(ArcAction::new(1, None, None)?, 0)?,
            action_diagnostic_sample(ArcAction::new(6, Some(11), Some(23))?, 1)?,
            action_diagnostic_sample(ArcAction::new(6, Some(12), Some(24))?, 2)?,
            action_diagnostic_sample(ArcAction::new(7, None, None)?, 3)?,
            action_diagnostic_sample(ArcAction::new(4, None, None)?, 4)?,
        ];
        samples[1].noop = Some(true);
        samples[3].noop = None;
        let shuffled = shuffled_action_samples(&samples, &[2, 3, 1, 0, 4])?;
        let true_errors = [1.0, 2.0, 3.0, 4.0, 99.0];
        let shuffled_errors = [1.5, 2.5, 3.5, 4.5, 999.0];
        let seed = 17;
        let source_ranges = vec![
            ("empty".into(), 0, 0),
            ("paired".into(), 0, 4),
            ("singleton".into(), 4, 5),
        ];

        let diagnostics = action_diagnostics_from_pairs(
            &samples,
            &shuffled,
            &true_errors,
            &shuffled_errors,
            &source_ranges,
            seed,
        )?;

        let changed_only = diagnostics
            .changed_conditioning_only
            .as_ref()
            .expect("changed conditioning rows");
        assert_eq!(changed_only.n, 4);
        assert_eq!(changed_only.changed_conditionings, changed_only.n);
        assert_eq!(changed_only.changed_fraction, Some(1.0));

        // The pre-stratification aggregate and source summaries use exactly the
        // same eligible rows and seeds as before.
        let expected_aggregate = ActionSourceDiagnostics {
            shuffle: summarize_action_shuffle(
                &true_errors[..4],
                &shuffled_errors[..4],
                4,
                seed + 0xA661,
            )?,
            coverage: action_coverage(&samples),
        };
        let expected_paired = ActionSourceDiagnostics {
            shuffle: summarize_action_shuffle(
                &true_errors[..4],
                &shuffled_errors[..4],
                4,
                seed + 1 + 0xB005,
            )?,
            coverage: action_coverage(&samples[..4]),
        };
        let expected_empty = ActionSourceDiagnostics {
            shuffle: summarize_action_shuffle(&[], &[], 0, seed + 0xB005)?,
            coverage: action_coverage(&[]),
        };
        let expected_singleton = ActionSourceDiagnostics {
            shuffle: summarize_action_shuffle(&[], &[], 0, seed + 2 + 0xB005)?,
            coverage: action_coverage(&samples[4..]),
        };
        assert_eq!(
            serde_json::to_value(&diagnostics.aggregate)?,
            serde_json::to_value(expected_aggregate)?
        );
        assert_eq!(
            serde_json::to_value(diagnostics.by_source.get("paired").expect("paired source"))?,
            serde_json::to_value(expected_paired)?
        );
        for (source, expected) in [("empty", expected_empty), ("singleton", expected_singleton)] {
            assert_eq!(
                serde_json::to_value(diagnostics.by_source.get(source).expect("source"))?,
                serde_json::to_value(expected)?
            );
            let shuffle = &diagnostics.by_source.get(source).expect("source").shuffle;
            assert_eq!(shuffle.n, 0);
            assert_eq!(shuffle.true_action_mse, None);
            assert_eq!(shuffle.shuffled_action_mse, None);
            assert_eq!(shuffle.ratio, None);
        }

        // ACTION6 remains an atomic `(id, x, y)` conditioning in the source-local shuffle.
        assert_eq!(shuffled[0].action, samples[2].action);
        assert_eq!(shuffled[2].action, samples[1].action);

        let by_action = diagnostics
            .by_target_action_id
            .as_ref()
            .expect("target-action strata");
        assert_eq!(
            by_action.values().map(|metrics| metrics.n).sum::<usize>(),
            4
        );
        assert_eq!(by_action.get(&1).map(|metrics| metrics.n), Some(1));
        assert_eq!(by_action.get(&6).map(|metrics| metrics.n), Some(2));
        assert_eq!(by_action.get(&7).map(|metrics| metrics.n), Some(1));
        assert!(!by_action.contains_key(&4));
        assert_eq!(
            by_action
                .values()
                .map(|metrics| metrics.true_action_mse.expect("non-empty") * metrics.n as f64)
                .sum::<f64>()
                / 4.0,
            diagnostics
                .aggregate
                .shuffle
                .true_action_mse
                .expect("paired aggregate")
        );
        assert_eq!(
            by_action
                .values()
                .map(|metrics| metrics.shuffled_action_mse.expect("non-empty") * metrics.n as f64)
                .sum::<f64>()
                / 4.0,
            diagnostics
                .aggregate
                .shuffle
                .shuffled_action_mse
                .expect("paired aggregate")
        );

        let action_kind = diagnostics
            .by_target_action_kind
            .as_ref()
            .expect("target-action-kind strata");
        let simple = action_kind.simple.as_ref().expect("simple rows");
        let coordinate = action_kind.coordinate.as_ref().expect("coordinate rows");
        assert_eq!(simple.n + coordinate.n, 4);
        assert_eq!(coordinate.n, 2);
        assert_eq!(coordinate.true_action_mse, Some(2.5));
        assert_eq!(
            (simple.true_action_mse.expect("simple") * simple.n as f64
                + coordinate.true_action_mse.expect("coordinate") * coordinate.n as f64)
                / (simple.n + coordinate.n) as f64,
            diagnostics
                .aggregate
                .shuffle
                .true_action_mse
                .expect("paired aggregate")
        );

        let transition_kind = diagnostics
            .by_transition_kind
            .as_ref()
            .expect("transition-kind strata");
        let changed = transition_kind
            .changed_transition
            .as_ref()
            .expect("changed rows");
        let noop = transition_kind.noop.as_ref().expect("no-op rows");
        assert_eq!(changed.n + noop.n, 3);
        assert_eq!(changed.n, 2);
        assert_eq!(noop.n, 1);
        assert_eq!(changed.true_action_mse, Some(2.0));
        assert_eq!(noop.true_action_mse, Some(2.0));
        assert_eq!(
            (changed.true_action_mse.expect("changed") * changed.n as f64
                + noop.true_action_mse.expect("no-op") * noop.n as f64)
                / (changed.n + noop.n) as f64,
            2.0
        );

        let legacy = serde_json::json!({
            "aggregate": diagnostics.aggregate,
            "by_source": diagnostics.by_source,
        });
        let decoded: ActionDiagnostics = serde_json::from_value(legacy)?;
        assert!(decoded.changed_conditioning_only.is_none());
        assert!(decoded.by_target_action_id.is_none());
        assert!(decoded.by_target_action_kind.is_none());
        assert!(decoded.by_transition_kind.is_none());
        Ok(())
    }

    #[test]
    fn changed_conditioning_action_metric_excludes_unchanged_action_tuples() -> Result<()> {
        let samples = vec![
            action_diagnostic_sample(ArcAction::new(1, None, None)?, 0)?,
            action_diagnostic_sample(ArcAction::new(1, None, None)?, 1)?,
            action_diagnostic_sample(ArcAction::new(6, Some(12), Some(24))?, 2)?,
        ];
        let shuffled = shuffled_action_samples(&samples, &[1, 2, 0])?;
        let diagnostics = action_diagnostics_from_pairs(
            &samples,
            &shuffled,
            &[1.0, 2.0, 3.0],
            &[100.0, 4.0, 6.0],
            &[("paired".into(), 0, 3)],
            29,
        )?;

        assert_eq!(diagnostics.aggregate.shuffle.n, 3);
        assert_eq!(diagnostics.aggregate.shuffle.changed_conditionings, 2);
        let changed_only = diagnostics
            .changed_conditioning_only
            .expect("changed conditioning rows");
        assert_eq!(changed_only.n, 2);
        assert_eq!(changed_only.changed_conditionings, 2);
        assert_eq!(changed_only.changed_fraction, Some(1.0));
        assert_eq!(changed_only.true_action_mse, Some(2.5));
        assert_eq!(changed_only.shuffled_action_mse, Some(5.0));
        assert_eq!(changed_only.ratio, Some(2.0));
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
    fn current_seam_keys_are_snake_case() -> Result<()> {
        let metrics = crate::p2::representation::RepresentationSeamMetrics {
            rows_seen: 9,
            rows_used: 4,
            non_finite_rows: 0,
            dimension: 3,
            mean_rms: Some(1.0),
            mean_variance: Some(0.25),
            effective_rank: Some(2.0),
            effective_rank_fraction: Some(2.0 / 3.0),
        };
        let seams = BTreeMap::from([(RepresentationSeam::EncoderPostRmsPooled, metrics.clone())]);
        let json = serde_json::to_string(&seams)?;
        assert!(json.contains("\"encoder_post_rms_pooled\""));

        let top = summarize_representation_from_seam(&metrics, 0.0, 0.0, 0);
        assert_eq!(top.encoder_rows, metrics.rows_used);
        assert_eq!(top.encoder_dim, metrics.dimension);
        assert_eq!(top.mean_encoder_variance, metrics.mean_variance);
        assert_eq!(top.effective_rank, metrics.effective_rank);
        assert_eq!(top.effective_rank_fraction, metrics.effective_rank_fraction);
        Ok(())
    }

    #[test]
    fn top_level_representation_pools_current_and_target_post_rms_rows() -> Result<()> {
        let current = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &Device::Cpu)?;
        let target = Tensor::from_vec(vec![3f32, 0., 0., 3.], (2, 2), &Device::Cpu)?;
        let seams = BTreeMap::from([
            (RepresentationSeam::EncoderPostRmsPooled, current.clone()),
            (RepresentationSeam::TargetPostRmsPooled, target.clone()),
        ]);
        let mut collector = RepresentationRowCollector::new(7, 0x504F_4F4C_4544_5F5A, 8);
        collect_top_level_representation(&mut collector, &seams, 0, 2)?;
        let pooled = collector.summarize()?;
        let expected = crate::p2::representation::summarize_seam(
            &Tensor::cat(&[&current, &target], 0)?,
            RepresentationSeam::EncoderPostRmsPooled,
            7,
            8,
        )?;
        assert_eq!(pooled.rows_seen, 4);
        assert_eq!(pooled.rows_used, 4);
        assert_eq!(pooled.mean_variance, expected.mean_variance);
        assert_eq!(pooled.effective_rank, expected.effective_rank);
        Ok(())
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
    fn factual_population_is_stable_and_domain_separated() -> Result<()> {
        let first_seed = factual_branch_eval_seed(3);
        assert_ne!(first_seed, 3);
        assert_eq!(first_seed, factual_branch_eval_seed(3));
        let groups = (0..2)
            .map(|episode| {
                generate_factual_branch_group(first_seed, episode, Split::HeldOutComposition)
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            factual_population_fingerprint(&groups, 2),
            factual_population_fingerprint(&groups, 2)
        );
        assert_ne!(
            factual_population_fingerprint(&groups, 2),
            factual_population_fingerprint(&groups, 1)
        );
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
            lessons: vec!["dynamics".into(), "factual_branches".into()],
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_weight: 0.0,
            world_core_v2: true,
            branch_learning: crate::p2::branch_learning::BranchLearningConfig {
                enabled: true,
                ..Default::default()
            },
            ..TrainConfig::default()
        };

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = WorldModel::new(train_cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, train_cfg.seed)?;
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
            world_core_schema: crate::p2::branch_learning::WORLD_CORE_V2_SCHEMA.into(),
            experiment: train_cfg.resolved_experiment()?,
            seed: train_cfg.seed,
            physical_batch: train_cfg.physical_batch,
            grad_accum: 1,
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            parameter_count: 1,
            training_population_fingerprint: "fnv1a64:0000000000000000".into(),
            training_content_fingerprint: "sha256:00".into(),
            training_population_rows: 0,
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
            gradient_pressure: None,
            gradient_pressure_samples: vec![],
            research_claim: false,
        };
        save_checkpoint(&varmap, &train_cfg, &report)?;

        let eval_cfg = EvalConfig {
            checkpoint: dir.join("model.safetensors"),
            train_config: dir.join("config.json"),
            seed: 3,
            synthetic_episodes: 2,
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
            mode: EvalMode::Full,
            representation_row_cap: 7,
        };
        let eval = evaluate(&eval_cfg)?;
        assert_eq!(eval.schema, EVAL_REPORT_SCHEMA);
        assert!(eval.official_rhae.is_none());
        assert!(!eval.public_data_used_for_fitting);
        assert!(!eval.research_claim);
        assert!(eval.synthetic_dynamics.n_samples > 0);
        assert!(eval.synthetic_planner.n_samples > 0);
        let factual = eval
            .factual_branches
            .as_ref()
            .expect("world-core-v2 factual evaluation");
        assert_eq!(factual.groups, eval_cfg.synthetic_episodes * 4);
        assert_eq!(factual.branches, factual.changed + factual.unchanged);
        assert_eq!(factual.action6_coordinate_n, factual.action6);
        assert!(factual.action6_coordinate_rmse_normalized.is_some());
        assert!(factual.action6_coordinate_rmse_pixels.is_some());
        assert!(!factual.population_fingerprint.is_empty());
        assert!(factual.board_probe.is_some());
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
                .as_ref()
                .expect("full eval PTRM")
                .iter()
                .find(|row| row.k == 1)
                .map(|row| row.noise),
            Some(0.0)
        );
        assert_eq!(
            eval.synthetic_dynamics
                .deterministic_matched_compute
                .as_ref()
                .expect("full eval matched compute")
                .len(),
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
        assert_eq!(
            back.factual_branches
                .as_ref()
                .map(|metrics| &metrics.population_fingerprint),
            Some(&factual.population_fingerprint)
        );
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

        let post_rms = eval
            .synthetic_dynamics
            .representation_seams
            .as_ref()
            .and_then(|seams| seams.get(&RepresentationSeam::EncoderPostRmsPooled))
            .expect("post-RMS pooled seam");
        let target_post_rms = eval
            .synthetic_dynamics
            .representation_seams
            .as_ref()
            .and_then(|seams| seams.get(&RepresentationSeam::TargetPostRmsPooled))
            .expect("target post-RMS pooled seam");
        let top = eval
            .synthetic_dynamics
            .representation
            .as_ref()
            .expect("top-level representation");
        assert_eq!(top.encoder_rows, eval_cfg.representation_row_cap);
        assert_eq!(top.encoder_dim, post_rms.dimension);
        assert_eq!(top.encoder_dim, target_post_rms.dimension);

        let mut representation_cfg = eval_cfg.clone();
        representation_cfg.mode = EvalMode::Representation;
        representation_cfg.output = dir.join("representation.json");
        let representation_eval = evaluate(&representation_cfg)?;
        assert_eq!(representation_eval.mode, EvalMode::Representation);
        assert!(representation_eval
            .synthetic_dynamics
            .representation_seams
            .is_some());
        assert!(representation_eval.synthetic_dynamics.events.is_none());
        assert!(representation_eval.synthetic_dynamics.q.is_none());
        assert!(representation_eval.synthetic_dynamics.ptrm.is_none());
        assert!(representation_eval.synthetic_dynamics.rollout.is_none());

        let mut physical_batch_one = representation_cfg.clone();
        physical_batch_one.physical_batch = 1;
        physical_batch_one.output = dir.join("representation-batch-one.json");
        let batch_one_eval = evaluate(&physical_batch_one)?;
        let representation_json = serde_json::to_value(&representation_eval.synthetic_dynamics)?;
        let batch_one_json = serde_json::to_value(&batch_one_eval.synthetic_dynamics)?;
        // SIGReg is a batch statistic and is unavailable for a one-row physical batch.
        // The bounded seam-derived geometry must still be partition invariant.
        for field in [
            "encoder_rows",
            "encoder_dim",
            "mean_encoder_variance",
            "effective_rank",
            "effective_rank_fraction",
            "noncollapse_pass",
        ] {
            assert_eq!(
                representation_json["representation"][field],
                batch_one_json["representation"][field]
            );
        }
        assert_eq!(
            representation_json["representation_seams"],
            batch_one_json["representation_seams"]
        );

        let mut rollout_cfg = eval_cfg.clone();
        rollout_cfg.mode = EvalMode::Rollout;
        rollout_cfg.output = dir.join("rollout.json");
        let rollout_eval = evaluate(&rollout_cfg)?;
        assert_eq!(rollout_eval.mode, EvalMode::Rollout);
        assert!(rollout_eval
            .synthetic_dynamics
            .one_step_latent_mse
            .is_none());
        assert!(rollout_eval.synthetic_dynamics.representation.is_none());
        assert!(rollout_eval
            .synthetic_dynamics
            .representation_seams
            .is_none());
        assert!(rollout_eval.synthetic_dynamics.action_diagnostics.is_none());
        assert!(rollout_eval.synthetic_dynamics.rollout.is_some());
        assert!(rollout_eval.synthetic_dynamics.closed_loop.is_some());
        assert!(rollout_eval.synthetic_dynamics.copy_forward.is_some());

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
        let empty_factual = arc_eval
            .factual_branches
            .as_ref()
            .expect("world-core-v2 report keeps factual branch metric shape");
        assert_eq!(empty_factual.branches, 0);
        assert_eq!(empty_factual.outcome_equivalence_retrieval_accuracy, None);
        assert_eq!(empty_factual.unique_changed_effect_action_top1, None);
        assert_eq!(empty_factual.action6_coordinate_rmse_normalized, None);
        assert_eq!(
            empty_factual.changed_vs_unchanged_displacement_norm_auroc,
            None
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
