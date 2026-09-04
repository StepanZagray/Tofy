//! P2 world-model evaluation (synthetic held-out + ARC recording transfer).

use crate::domain::Split;
use crate::gpu_lock::GpuSessionGuard;
use crate::p2::adaptation::{
    AdaptationMode, ContextScopeKind, FastWeightAdapter, ADAPT_MIN_LEVEL_TRANSITIONS,
};
use crate::p2::arc3::{import_and_summarize_recordings_dir, RecordingRunSummary};
use crate::p2::board_probe::{
    BoardProbeRows, BoardProbeTransitions, BoardTransitionMetrics, FixedBoardProbe, PATCH_COUNT,
};
use crate::p2::calibration::{
    binary_auroc, clopper_pearson_upper, expected_calibration_error, risk_coverage_buckets,
};
use crate::p2::cg_profile::{
    ensure_eval_profile_campaign, EvalCaptureSpec, RepresentativeUpdateCapture,
    EVAL_PROFILE_ENTRYPOINT,
};
use crate::p2::data::{
    adaptation_v6_stream_schedule, augmented_learning_history, compose_mixed_stream_batch,
    foundation_v2_stream_schedule, gameplay_rows, generate_curriculum,
    generate_factual_branch_group, generate_hazard_one_step, ArcAction, AugmentedLearningHistory,
    BranchGroup, ContentMask, ContentRect, FactualBatch, MixedStreamConfig, OperatorFamilySplit,
    TransitionSample, V5DataSplit, V5SampleProvenance, CONTEXT_WINDOW_MAX,
    FACTUAL_BRANCHES_PER_GROUP, FRAME_SIDE, GOAL_FEATURES_DIM, ORACLE_LATENT_DIM,
};
use crate::p2::grounding::DecodeComposition;
use crate::p2::latent_planning::trust::{
    CalibrationBin, PhaseACalibration, PhaseACalibrationFit, SYNTHETIC_HOLDOUT_SOURCE,
};
use crate::p2::model::{
    flatten_latent, latent_mse_per_sample, pool_latent, ContextBatch, PtrmConfig, RecursionDepth,
    RecursionOpts, RecursionStepProbe, WorldModel, EVENT_EXHAUSTED, EVENT_GOAL_FAILED,
    EVENT_GOAL_SATISFIED, EVENT_NOOP,
};
use crate::p2::representation::{
    RepresentationRowCollector, RepresentationSeam, RepresentationSeamCollector,
    RepresentationSeamMap, DEFAULT_REPRESENTATION_ROW_CAP,
};
use crate::p2::rhae::{
    benchmark_from_scorecard_json, official_rhae_from_benchmark, ScorecardBenchmark,
};
use crate::p2::semantic_eval::{
    action_controllability_probe, aggregate_decoder_metrics, ambiguity_ceiling, collision_census,
    evaluate_semantics_with_control, latent_semantic_metrics, shuffled_action_control_population,
    shuffled_action_control_samples, ActionControllabilityMetrics, AmbiguityCeiling,
    CollisionCensus, SemanticControlConfig, SemanticDecoderMetrics, SemanticEvaluation,
    ShuffledActionControlPopulation,
};
use crate::p2::train::{
    action_tensors_from_samples, batch_from_samples, foundation_v2_graded_q_targets,
    latent_content_mask, load_train_config, load_varmap_exact,
    model_sigreg_losses_for_encoded_pair, resolve_device, sigreg_loss_for_stack,
    verify_checkpoint_bundle, BatchTensors, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_graph::{ExecutionStep, PlannedCapture, SpanKind};
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
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// v18 (2026-08-27): overall semantic reducer omits content-derived
/// false-edit scalars; foundation-v2 SIGReg scored on the content-masked
/// canonical population (whole-population, batch-invariant); identifiability
/// split is group-disjoint; the correctness seam is composed copy-gate
/// semantics; content masks honor the provenance origin; v5_holdout_gates
/// added. v17 and v18 reports are not field-for-field comparable.
pub const EVAL_REPORT_SCHEMA: &str = "p2.eval_report.v18";
pub const ACTION_CONTROLLABILITY_LATENT_DISTANCE_THRESHOLD: f64 = 1e-3;

fn log_eval_phase(phase: &str, detail: &str, elapsed: Duration) {
    eprintln!(
        "[p2-eval timing] phase={phase} detail={detail} elapsed_s={:.3}",
        elapsed.as_secs_f64()
    );
}

fn timed_eval_phase<T>(phase: &str, detail: &str, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = f();
    log_eval_phase(phase, detail, started.elapsed());
    result
}

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
    /// Unseen seed for the legacy origin-aligned curriculum generator used by
    /// the `synthetic_iid_*` sections. This is NOT the foundation-v2 V5
    /// training composition (no geometry randomization, symmetry
    /// augmentation, or goal dropout); the `v5_holdout_gates` populations
    /// cover the actual ADR 0003 held-out distributions. Also seeds the
    /// domain-tagged V5 holdout populations.
    pub iid_seed: u64,
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
    /// Publish representative candle-graph inference evidence. CLI defaults
    /// this on; the serde/default path stays off for legacy embedded configs.
    #[serde(default)]
    pub profile_eval: bool,
    /// ADR 0005 §5.1 memorization diagnostic: score the v6 held-out rows a
    /// second time with every context window masked to `K = 0` and report the
    /// changed-exact delta per context-length stratum. Only meaningful for
    /// `world_core_v6` checkpoints; off keeps the report byte-identical.
    #[serde(default)]
    pub context_ablation: bool,
    /// ADR 0005 §5.2 adaptation falsifier (E3): prequential Channel A vs
    /// Channel A+B (reset, carry) on held-out synthetic Learning Histories.
    /// `world_core_v6` checkpoints only; off keeps the report byte-identical.
    #[serde(default)]
    pub adaptation_falsifier: bool,
    /// Falsifier-only warm-up override for Channel B (§6.2 default 8 unique
    /// transitions per level). Recorded in the report; the live loop is
    /// unaffected.
    #[serde(default = "default_adaptation_falsifier_min_level_transitions")]
    pub adaptation_falsifier_min_level_transitions: usize,
}

fn default_ensemble_members() -> usize {
    8
}

fn default_adaptation_falsifier_min_level_transitions() -> usize {
    ADAPT_MIN_LEVEL_TRANSITIONS
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
            iid_seed: 3,
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
            profile_eval: false,
            context_ablation: false,
            adaptation_falsifier: false,
            adaptation_falsifier_min_level_transitions: ADAPT_MIN_LEVEL_TRANSITIONS,
        }
    }
}

impl EvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.physical_batch == 0 {
            bail!("physical_batch must be > 0");
        }
        if self.adaptation_falsifier_min_level_transitions == 0 {
            bail!("adaptation_falsifier_min_level_transitions must be > 0");
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
    /// Non-serialized Rust compatibility for the legacy CLI summary.
    /// Eval-report v16 and episode-row v3 deliberately emit no h16 metric.
    #[serde(skip)]
    pub mse_16: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h4: Option<HorizonRolloutStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h8: Option<HorizonRolloutStats>,
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
    pub source_kind: String,
    pub true_action_effect_mse: Option<f64>,
    pub action_shuffled_effect_mse: Option<f64>,
    pub true_vs_action_shuffled_prediction_mse: Option<f64>,
    pub action_control_contract: String,
}

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
    /// Foundation-v2 gate: predictor improves over latent copy on average.
    pub positive_improvement_pass: Option<bool>,
}

/// Cheap fixed-population measurements consumed by the foundation-v2 trainer
/// gates. `evaluate_gate_support` performs one current/target encoding batch
/// and deterministic true/shuffled action forwards over the supplied rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSupportMetrics {
    pub samples: usize,
    /// SHA-256 of the exact ordered serialized transition rows, plus aligned
    /// V5 provenance when counterfactual evaluation uses it.
    #[serde(default)]
    pub population_fingerprint: String,
    /// SHA-256 of the exact ordered V5 content masks when the caller supplies
    /// them. `None` identifies the provenance-origin rectangle reconstruction.
    #[serde(default)]
    pub content_mask_fingerprint: Option<String>,
    /// This population participates in stopping/objective/checkpoint choices;
    /// it is selection-only evidence, never an untouched confirmation set.
    #[serde(default)]
    pub evidence_class: String,
    pub changed_transitions: usize,
    pub changed_pixels: usize,
    pub foreground_pixels: usize,
    /// `1 - mean(prediction_mse) / mean(copy_forward_mse)` on Changed Transitions.
    pub improvement_fraction: Option<f64>,
    /// Shuffled-action changed-pixel accuracy divided by true-action
    /// changed-pixel accuracy on known outcome-changing interventions, divided
    /// by true-action accuracy; lower means stronger action conditioning.
    pub shuffled_action_changed_pixel_ratio: Option<f64>,
    /// Rows in the shuffled-action control population.
    #[serde(default)]
    pub shuffled_action_rows: usize,
    /// Rows belonging to a source-local group with at least two donors.
    #[serde(default)]
    pub shuffled_action_eligible_rows: usize,
    /// Rows whose complete `(id,x,y)` tuple genuinely changed after the
    /// source-local marginal-preserving shuffle.
    #[serde(default)]
    pub shuffled_action_changed_tuples: usize,
    /// Rows where replaying the alternative tuple under the recorded V5
    /// operator changes the status-excluded simulator outcome. `None` means
    /// the population supplied no counterfactual operator sidecar.
    #[serde(default)]
    pub shuffled_action_outcome_changing_tuples: Option<usize>,
    /// Exact-decoder pixel accuracy on non-empty pixels of the encoded next state.
    pub foreground_reconstruction_accuracy: Option<f64>,
    /// Fraction of Changed Transitions whose every factually changed gameplay
    /// pixel is decoded exactly from the one-step prediction.
    pub one_step_changed_exact: Option<f64>,
    /// Diagnostic only: fraction of the same Changed Transitions whose entire
    /// status-excluded gameplay frame (changed and unchanged pixels alike) is
    /// decoded exactly by the composed copy-gate output used at inference.
    #[serde(default)]
    pub one_step_full_exact: Option<f64>,
    /// Raw exact-decoder counterpart to `one_step_full_exact`, retained to
    /// distinguish palette-head errors from copy-gate composition errors.
    #[serde(default)]
    pub one_step_raw_full_exact: Option<f64>,
    /// Composed-decode counterpart to `one_step_changed_exact`: the deployed
    /// copy-gate output, scored on the same changed-transition rows.
    #[serde(default)]
    pub one_step_composed_changed_exact: Option<f64>,
    /// Composed-decode full-frame exactness over every supplied row, no-op
    /// transitions included, so hallucinations on unchanged rows depress it.
    #[serde(default)]
    pub one_step_all_rows_exact: Option<f64>,
    /// Diagnostic only: fraction of factually unchanged pixels inside each
    /// sample's exact content rectangle that the composed decode edits.
    #[serde(default)]
    pub false_edit_rate: Option<f64>,
    /// Diagnostic only: fraction of unchanged padding pixels hallucinated by
    /// the composed decode, reported separately from content false edits.
    #[serde(default)]
    pub padding_false_edit_rate: Option<f64>,
    /// Raw exact-decoder false-edit counterparts, retained as diagnostics.
    #[serde(default)]
    pub raw_false_edit_rate: Option<f64>,
    #[serde(default)]
    pub raw_padding_false_edit_rate: Option<f64>,
    pub population_contract: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationMetrics {
    pub one_step_changed_exact: Option<f64>,
    pub one_step_all_rows_exact: Option<f64>,
    pub false_edit_rate: Option<f64>,
    pub padding_false_edit_rate: Option<f64>,
    pub improvement_fraction: Option<f64>,
    pub shuffled_action_changed_pixel_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationPair {
    pub baseline_description: String,
    pub baseline: AblationMetrics,
    pub variant_description: String,
    pub variant: AblationMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlphaSweepPoint {
    pub alpha: f64,
    pub metrics: AblationMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MechanismAblationReport {
    pub rows: usize,
    pub population_fingerprint: String,
    pub decode_composition: Option<AblationPair>,
    pub action_impulse: Option<AblationPair>,
    pub copy_bypass_alpha_sweep: Option<Vec<AlphaSweepPoint>>,
    pub evidence_class: String,
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
    /// Which intervention produced these paired errors, with a fingerprint of
    /// the exact donor permutation. This is a *different* intervention from
    /// the semantic evaluator's maximum-change cyclic rotation: the two
    /// sections must not be read as one causal control.
    #[serde(default)]
    pub intervention_contract: String,
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
    /// Stable episode-rollout row schema. V3 intentionally reports only the
    /// populated h4/h8 horizons; the never-populated h16 row was removed.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_semantic: Option<SemanticDecoderMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_semantic: Option<SemanticDecoderMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned_copy_semantic: Option<SemanticDecoderMetrics>,
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
    open_semantic: Option<SemanticDecoderMetrics>,
    closed_semantic: Option<SemanticDecoderMetrics>,
    learned_copy_semantic: Option<SemanticDecoderMetrics>,
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
            schema: "p2.episode_rollout.v3".into(),
            source: source.into(),
            seed: self.seed,
            episode_id: self.episode_id,
            families_through_horizon: self.families_through_horizon,
            horizon: self.horizon,
            open_mse: self.open_mse,
            closed_mse: self.closed_mse,
            copy_forward_mse: self.copy_forward_mse,
            normalized_open_mse,
            open_semantic: self.open_semantic,
            closed_semantic: self.closed_semantic,
            learned_copy_semantic: self.learned_copy_semantic,
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
        open_semantic: None,
        closed_semantic: None,
        learned_copy_semantic: None,
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
    /// Full V4 exact-decoder metrics and controls, split by semantic mask/source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticEvaluation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_rollout: Option<SemanticRolloutMetrics>,
    pub collision_census: CollisionCensus,
    /// Visible-state/action factual-successor ambiguity at history 1 and 2.
    pub ambiguity_ceiling: AmbiguityCeiling,
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
pub struct SemanticRolloutMetrics {
    pub population_contract: String,
    pub comparable_to_one_step: bool,
    pub open: BTreeMap<usize, SemanticDecoderMetrics>,
    pub closed: BTreeMap<usize, SemanticDecoderMetrics>,
    pub learned_copy: BTreeMap<usize, SemanticDecoderMetrics>,
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
    pub predicted_action_x_normalized: Option<f32>,
    pub predicted_action_y_normalized: Option<f32>,
    /// NLL of this action-conditioned prediction under its factual board outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factual_outcome_nll: Option<f64>,
    /// Distinct same-state outcomes tied for minimum NLL (class indices).
    #[serde(default)]
    pub best_outcome_classes: Vec<usize>,
    /// `1/ties` when the factual class is among the minima, otherwise zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factual_outcome_retrieval_credit: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factual_outcome_chance: Option<f64>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualSemanticOutcomeStratum {
    pub n: usize,
    pub retrieval_accuracy: Option<f64>,
    pub chance: Option<f64>,
    pub factual_nll_mean: Option<f64>,
}

/// Held-out same-state factual-action evaluation for spatial-action world cores.
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
    /// All factual legal actions are forwarded from each fixed held-out branch
    /// state; distances are between predicted Consumer Latents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_controllability: Option<ActionControllabilityMetrics>,
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
    /// Primary Full V4 test: each supplied action must select its factual
    /// semantic outcome among all distinct outcomes of the same current state.
    pub semantic_outcome_retrieval_n: usize,
    pub semantic_outcome_retrieval_accuracy: Option<f64>,
    pub semantic_outcome_chance: Option<f64>,
    pub semantic_factual_nll_mean: Option<f64>,
    pub semantic_outcome_by_family: BTreeMap<String, FactualSemanticOutcomeStratum>,
}

/// Model-family-independent semantic probe on a deterministic held-out
/// synthetic transition population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardProbeEvaluation {
    /// Versioned, domain-separated SHA-256 over the exact ordered full-row
    /// population the probe consumed (see [`BOARD_PROBE_POPULATION_DOMAIN`]).
    /// Reports written before Wave 22 carried a `fnv1a64:` projection over
    /// `(seed, episode_id, transition_index, family, action)` only, which
    /// could not distinguish populations that differed in frames, operator
    /// conditioning, content origin, or labels.
    pub population_fingerprint: String,
    pub fit_frames: usize,
    pub held_out_frames: usize,
    pub metrics: BoardTransitionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualInterval {
    pub estimate: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub lower_98_75: f64,
    pub upper_98_75: f64,
    pub groups: usize,
    pub pairs: usize,
    pub resamples: usize,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualGroupIdentity {
    pub group_index: usize,
    pub family: String,
    /// `movement` for exact-simulator groups and `coordinate` for synthetic
    /// ACTION6 groups. The two populations are never pooled into simulator
    /// claims.
    pub population: String,
    /// Canonical SHA-256 over dimensions, pixels, actions, public goals, and
    /// board-outcome classes. Provenance seed and episode identifiers are
    /// deliberately excluded.
    pub content_fingerprint: String,
    pub current_sha256: String,
    pub next_sha256: Vec<String>,
    pub actions: Vec<ArcAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualPairLedgerRow {
    pub group: OutcomeCounterfactualGroupIdentity,
    pub left_branch_index: usize,
    pub right_branch_index: usize,
    pub left_action: ArcAction,
    pub right_action: ArcAction,
    pub left_outcome_class: usize,
    pub right_outcome_class: usize,
    pub left_changed: bool,
    pub right_changed: bool,
    pub left_changed_cells: Vec<u16>,
    pub right_changed_cells: Vec<u16>,
    pub target_pair_mse: f64,
    pub concordant_loss: f64,
    pub crossed_loss: f64,
    pub margin: f64,
    pub eligible: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeCounterfactualActionAnchors {
    pub changed: usize,
    pub unchanged: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualPopulationGates {
    pub eligible_simulator_groups: usize,
    pub eligible_simulator_groups_at_least_100: bool,
    pub movement_action_anchors: BTreeMap<u8, OutcomeCounterfactualActionAnchors>,
    pub each_movement_action_at_least_16_changed_and_16_unchanged: bool,
    pub simulator_changed_changed_pairs: usize,
    pub simulator_changed_changed_pairs_at_least_100: bool,
    pub target_collapse_failure: bool,
    pub population_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualStateScrambledControl {
    pub available: bool,
    pub estimate: Option<f64>,
    pub groups: usize,
    pub pairs: usize,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualControls {
    pub pixel_oracle_estimate: Option<f64>,
    pub pixel_oracle_exactly_one: bool,
    pub latent_oracle_estimate: Option<f64>,
    pub latent_oracle_at_least_0_99: bool,
    pub target_collapse_failure: bool,
    pub target_collapsed_pairs: usize,
    pub swapped_oracle_estimate: Option<f64>,
    pub swapped_oracle_at_most_negative_0_99: bool,
    pub action_masked_max_abs_margin: Option<f64>,
    pub action_masked_max_abs_at_most_1e_6: bool,
    pub identity_max_abs_margin: Option<f64>,
    pub identity_max_abs_at_most_1e_6: bool,
    pub outcome_equivalent_pairs: usize,
    pub outcome_equivalent_max_abs_margin: Option<f64>,
    pub outcome_equivalent_max_abs_at_most_1e_6: bool,
    pub state_scrambled_same_action_template: OutcomeCounterfactualStateScrambledControl,
    pub required_controls_pass: bool,
}

/// Model-family-independent same-state outcome counterfactual evaluation.
/// Predictions and targets are full spatial displacements from the one shared
/// encoded current state in each factual branch group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCounterfactualMetrics {
    pub population_fingerprint: String,
    pub groups: usize,
    pub movement_groups: usize,
    pub coordinate_groups: usize,
    pub unordered_pairs: usize,
    pub eligible_pairs: usize,
    pub outcome_equivalent_pairs: usize,
    pub changed_changed_pairs: usize,
    pub changed_unchanged_pairs: usize,
    pub epsilon: f64,
    pub material_threshold: f64,
    pub overall: Option<OutcomeCounterfactualInterval>,
    pub movement: Option<OutcomeCounterfactualInterval>,
    pub coordinate: Option<OutcomeCounterfactualInterval>,
    pub changed_changed: Option<OutcomeCounterfactualInterval>,
    pub changed_unchanged: Option<OutcomeCounterfactualInterval>,
    /// True iff the exact-simulator movement estimate and its lower 95% bound
    /// both strictly exceed `material_threshold`. Synthetic coordinate groups
    /// are reported separately and never contribute to this claim.
    pub action_separation_pass: bool,
    pub controls: OutcomeCounterfactualControls,
    pub population_gates: OutcomeCounterfactualPopulationGates,
    pub pair_ledger: Vec<OutcomeCounterfactualPairLedgerRow>,
    pub ledger_reconciled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub schema: String,
    pub mode: EvalMode,
    pub seed: u64,
    pub iid_seed: u64,
    pub identity: EvaluationIdentity,
    pub checkpoint: PathBuf,
    pub device: String,
    pub q_mse_threshold: f64,
    /// Full V4 uses exact decoder-derived gameplay-pixel correctness; legacy
    /// checkpoints retain their frozen latent-MSE threshold labels.
    pub q_label_definition: String,
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
    /// Unseen-seed training-composition control. This distinguishes ordinary
    /// generalization from the existing held-out-composition OOD population.
    pub synthetic_iid_dynamics: SplitEval,
    pub synthetic_iid_planner: SplitEval,
    /// Eval-only patch/palette grounding probe on the same model-independent
    /// population for every consumer-readout arm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board_probe: Option<BoardProbeEvaluation>,
    /// Held-out same-state factual branch metrics for V2 and Full V4.
    #[serde(default)]
    pub factual_branches: Option<FactualBranchMetrics>,
    /// Full-mode, model-family-independent same-state counterfactual metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_counterfactuals: Option<OutcomeCounterfactualMetrics>,
    /// Optional transfer on imported ARC recordings (never used for training).
    pub arc3_transfer: Option<SplitEval>,
    /// Foundation-v2 gate metrics on each named ADR 0003 V5 held-out split
    /// (unseen seed, composition, translation, size, held-out operator
    /// families), evaluated with exact content-mask sidecars. The legacy
    /// `synthetic_iid_*` sections use the origin-aligned curriculum
    /// generator, not the V5 training composition; these populations are the
    /// contract's actual held-out distributions. Selection-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v5_holdout_gates: Option<BTreeMap<String, GateSupportMetrics>>,
    /// ADR 0005 §7: the same gate metrics per held-out split, stratified by
    /// the row's context window length (`"0"`, `"1-4"`, `"5-16"`). Present
    /// only for `world_core_v6` checkpoints; selection-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v6_context_strata: Option<BTreeMap<String, BTreeMap<String, GateSupportMetrics>>>,
    /// ADR 0005 §5.1 memorization diagnostic (`--context-ablation`): the
    /// unseen-seed held-out population scored with its full context windows
    /// and again with every window masked to `K = 0`. Present only when the
    /// flag is set on a `world_core_v6` checkpoint; selection-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v6_context_ablation: Option<ContextAblationReport>,
    /// ADR 0005 §5.2 adaptation falsifier (`--adaptation-falsifier`):
    /// prequential Channel A vs A+B on held-out Learning Histories. Present
    /// only when the flag is set on a `world_core_v6` checkpoint;
    /// selection-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v6_adaptation_falsifier: Option<AdaptationFalsifierReport>,
    /// Same-checkpoint, same-row mechanism ablations for bundled treatments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mechanism_ablations: Option<MechanismAblationReport>,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationIdentity {
    pub command: Vec<String>,
    pub command_sha256: String,
    pub checkpoint_sha256: String,
    pub train_config_sha256: String,
    pub eval_config_sha256: String,
    pub evaluator_binary: PathBuf,
    pub evaluator_binary_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_bundle_manifest_sha256: Option<String>,
    pub population_sha256: BTreeMap<String, String>,
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn evaluation_identity(
    cfg: &EvalConfig,
    populations: BTreeMap<String, String>,
) -> Result<EvaluationIdentity> {
    let evaluator_binary = std::env::current_exe().context("resolve evaluator binary")?;
    let bundle_manifest = cfg
        .checkpoint
        .parent()
        .map(|parent| parent.join("bundle-manifest.json"))
        .filter(|path| path.is_file());
    let command = std::env::args().collect::<Vec<_>>();
    let command_sha256 = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&command).context("serialize evaluator argv")?)
    );
    Ok(EvaluationIdentity {
        command,
        command_sha256,
        checkpoint_sha256: file_sha256(&cfg.checkpoint)?,
        train_config_sha256: file_sha256(&cfg.train_config)?,
        eval_config_sha256: format!(
            "sha256:{:x}",
            Sha256::digest(serde_json::to_vec(cfg).context("serialize eval config for identity")?)
        ),
        evaluator_binary_sha256: file_sha256(&evaluator_binary)?,
        evaluator_binary,
        checkpoint_bundle_manifest_sha256: bundle_manifest
            .as_deref()
            .map(file_sha256)
            .transpose()?,
        population_sha256: populations,
    })
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

fn write_eval_digest(path: &Path) -> Result<()> {
    let digest = file_sha256(path)?;
    let digest = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| anyhow::anyhow!("unexpected digest format"))?;
    let sidecar = PathBuf::from(format!("{}.sha256", path.display()));
    let tmp = PathBuf::from(format!("{}.tmp", sidecar.display()));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("evaluation output has no file name"))?;
    fs::write(&tmp, format!("{digest}  {file_name}\n"))
        .with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, &sidecar)
        .with_context(|| format!("rename {} -> {}", tmp.display(), sidecar.display()))?;
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
    split: Split,
) -> Result<Vec<(String, Vec<TransitionSample>)>> {
    let jobs: Vec<(usize, usize, &str)> = kinds
        .iter()
        .enumerate()
        .flat_map(|(kind_index, kind)| (0..episodes).map(move |ep| (kind_index, ep, *kind)))
        .collect();
    let parts = jobs
        .par_iter()
        .enumerate()
        .map(|(job_idx, &(kind_index, ep, kind))| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            let samples = generate_curriculum(kind, seed, episode_id, split)?;
            Ok((job_idx, samples))
        })
        .collect::<Vec<Result<_>>>();
    let mut sources = kinds
        .iter()
        .map(|kind| ((*kind).to_string(), Vec::new()))
        .collect::<Vec<_>>();
    for part in parts {
        let (job_index, samples) = part?;
        let kind_index = job_index / episodes.max(1);
        sources[kind_index].1.extend(samples);
    }
    Ok(sources)
}

fn semantic_population_fingerprint(samples: &[TransitionSample]) -> String {
    let mut digest = Sha256::new();
    for sample in samples {
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
        match sample.provenance.operator {
            Some(operator) => digest.update([
                operator.family.conditioning_token() as u8,
                operator.agent_color,
                operator.primary_color,
                operator.secondary_color,
            ]),
            None => digest.update([0, 0, 0, 0]),
        }
        digest.update([
            sample.action.id,
            sample.action.x.unwrap_or(u8::MAX),
            sample.action.y.unwrap_or(u8::MAX),
            match sample.noop {
                Some(false) => 0,
                Some(true) => 1,
                None => u8::MAX,
            },
        ]);
        for goal in sample.goal_features.values {
            digest.update(goal.to_bits().to_le_bytes());
        }
        for frame in [&sample.current, &sample.next] {
            digest.update(frame.width.to_le_bytes());
            digest.update(frame.height.to_le_bytes());
            digest.update((frame.pixels.len() as u64).to_le_bytes());
            digest.update(&frame.pixels);
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

/// Domain tag for the board-probe population identity. Bump the version
/// whenever [`update_canonical_transition_row`] changes.
pub(crate) const BOARD_PROBE_POPULATION_DOMAIN: &str = "tofy.p2.board_probe_population.v2";

fn digest_framed_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn digest_option_bool(digest: &mut Sha256, value: Option<bool>) {
    digest.update([match value {
        None => 0u8,
        Some(false) => 1,
        Some(true) => 2,
    }]);
}

fn digest_frame(digest: &mut Sha256, frame: &crate::p2::data::ArcFrame) {
    digest.update(frame.width.to_le_bytes());
    digest.update(frame.height.to_le_bytes());
    digest_framed_bytes(digest, &frame.pixels);
}

fn digest_action(digest: &mut Sha256, action: &ArcAction) {
    digest.update([action.id]);
    for coordinate in [action.x, action.y] {
        match coordinate {
            None => digest.update([0u8, 0]),
            Some(value) => digest.update([1u8, value]),
        }
    }
}

/// Canonical full-row codec for one [`TransitionSample`]: fixed field order,
/// little-endian fixed-width scalars, length-prefixed variable fields, and
/// explicit option/enum tags. Every field of the row is folded, including the
/// ones a given consumer does not read, so two populations that differ in any
/// serialized field get different identities.
pub(crate) fn update_canonical_transition_row(digest: &mut Sha256, sample: &TransitionSample) {
    digest.update(sample.seed.to_le_bytes());
    digest.update(sample.episode_id.to_le_bytes());
    digest.update(sample.transition_index.to_le_bytes());
    digest_framed_bytes(digest, sample.family.as_bytes());
    digest.update([match sample.split {
        Split::Train => 0u8,
        Split::HeldOutComposition => 1,
    }]);
    digest_frame(digest, &sample.current);
    digest_frame(digest, &sample.next);
    digest_action(digest, &sample.action);
    digest.update((sample.goal_features.values.len() as u64).to_le_bytes());
    for goal in sample.goal_features.values {
        digest.update(goal.to_bits().to_le_bytes());
    }
    digest_option_bool(digest, sample.noop);
    digest_option_bool(digest, sample.goal_satisfied);
    digest_option_bool(digest, sample.goal_failed);
    digest_option_bool(digest, sample.exhausted);

    let provenance = &sample.provenance;
    digest.update(provenance.content_width.to_le_bytes());
    digest.update(provenance.content_height.to_le_bytes());
    digest.update(provenance.content_x.to_le_bytes());
    digest.update(provenance.content_y.to_le_bytes());
    digest_framed_bytes(digest, provenance.source_kind.as_bytes());
    digest_framed_bytes(digest, provenance.trajectory_id.as_bytes());
    match provenance.operator {
        None => digest.update([0u8; 6]),
        Some(operator) => digest.update([
            1u8,
            operator.family.conditioning_token() as u8,
            operator.agent_color,
            operator.primary_color,
            operator.secondary_color,
            operator.empty_color,
        ]),
    }
    digest.update(provenance.rule_id.to_le_bytes());
    digest.update(provenance.level_index.to_le_bytes());
    digest.update([
        provenance.available_actions,
        provenance.context_len,
        provenance.background_color,
    ]);

    match &sample.oracle_latent {
        None => digest.update([0u8]),
        Some(latent) => {
            digest.update([1u8]);
            digest.update((latent.len() as u64).to_le_bytes());
            for value in latent {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
    }
    digest.update((sample.context.len() as u64).to_le_bytes());
    for context in &sample.context {
        digest_frame(digest, &context.current);
        digest_action(digest, &context.action);
        digest_frame(digest, &context.next);
    }
}

/// Versioned, domain-separated SHA-256 identity of an ordered transition
/// population: `domain || row_count || (ordinal || canonical_row)*`.
pub(crate) fn canonical_transition_population_sha256(
    domain: &str,
    samples: &[TransitionSample],
) -> String {
    let mut digest = Sha256::new();
    digest_framed_bytes(&mut digest, domain.as_bytes());
    digest.update((samples.len() as u64).to_le_bytes());
    for (ordinal, sample) in samples.iter().enumerate() {
        digest.update((ordinal as u64).to_le_bytes());
        update_canonical_transition_row(&mut digest, sample);
    }
    format!("sha256:{:x}", digest.finalize())
}

/// Board-probe population identity over the full ordered population handed to
/// [`evaluate_board_probe`] (fit and held-out partitions included).
fn sample_population_fingerprint(samples: &[TransitionSample]) -> String {
    canonical_transition_population_sha256(BOARD_PROBE_POPULATION_DOMAIN, samples)
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
                .training_latents_from_encoded_state_with_operator_conditioning(
                    &current,
                    &batch.actions,
                    &batch.action_coords,
                    &batch.operator_conditioning,
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

fn collect_hazard_samples(
    seed: u64,
    episodes: usize,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    if episodes == 0 {
        return Ok(Vec::new());
    }
    let parts = (0..episodes)
        .into_par_iter()
        .map(|ep| {
            generate_hazard_one_step(seed.wrapping_add(0xFA17), ep as u64, split, 4)
                .map(|samples| (ep, samples))
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::new();
    for part in parts {
        samples.extend(part?.1);
    }
    Ok(samples)
}

#[cfg(test)]
fn collect_dynamics_rollout_samples(
    seed: u64,
    episodes: usize,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    let jobs: Vec<(usize, usize, &str)> = ["random_one_step", "exploration"]
        .iter()
        .enumerate()
        .flat_map(|(kind_index, kind)| (0..episodes).map(move |ep| (kind_index, ep, *kind)))
        .collect();
    let parts = jobs
        .par_iter()
        .enumerate()
        .map(|(job_idx, &(kind_index, ep, kind))| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            let samples = generate_curriculum(kind, seed, episode_id, split)?;
            Ok((job_idx, samples))
        })
        .collect::<Vec<Result<_>>>();
    let mut samples = Vec::new();
    for part in parts {
        samples.extend(part?.1);
    }
    Ok(samples)
}

fn collect_planner_rollout_samples(
    seed: u64,
    episodes: usize,
    split: Split,
    cached_sequential: Option<&[TransitionSample]>,
) -> Result<Vec<TransitionSample>> {
    let mut samples = match cached_sequential {
        Some(cached) => cached.to_vec(),
        None => collect_curriculum_source(seed, episodes, 0, "sequential", split)?,
    };
    samples.extend(collect_curriculum_source(
        seed,
        episodes,
        1,
        "p1c_hard_retarget",
        split,
    )?);
    Ok(samples)
}

fn collect_curriculum_source(
    seed: u64,
    episodes: usize,
    kind_index: usize,
    kind: &str,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    let parts = (0..episodes)
        .into_par_iter()
        .map(|episode| {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(episode as u64);
            generate_curriculum(kind, seed, episode_id, split)
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::new();
    for part in parts {
        samples.extend(part?);
    }
    Ok(samples)
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
    let improvement_fraction = improvement_fraction(learned, copy_forward);
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
        positive_improvement_pass: improvement_fraction.map(|value| value > 0.0),
    })
}

/// The authoritative Changed Transition predicate. `noop` is derived from the
/// status-row-excluded board effect by synthetic/import adapters.
pub fn is_board_changed_transition(sample: &TransitionSample) -> bool {
    sample.noop == Some(false)
}

pub fn board_changed_transition_count(samples: &[TransitionSample]) -> usize {
    samples
        .iter()
        .filter(|sample| is_board_changed_transition(sample))
        .count()
}

pub fn improvement_fraction(learned: &[f32], copy_forward: &[f32]) -> Option<f64> {
    if learned.len() != copy_forward.len()
        || learned.is_empty()
        || learned
            .iter()
            .chain(copy_forward)
            .any(|value| !value.is_finite())
    {
        return None;
    }
    mean(learned)
        .zip(mean(copy_forward))
        .and_then(|(learned, copy)| (copy > 0.0).then_some(1.0 - learned / copy))
}

/// Board pixels of a row. `gameplay_len` is the decoder's output width:
/// `63*64` for legacy heads, `64*64` under ADR 0005 §1.1.
fn gameplay_pixels(sample: &TransitionSample, gameplay_len: usize) -> (&[u8], &[u8]) {
    let current_end = gameplay_len.min(sample.current.pixels.len());
    let next_end = gameplay_len.min(sample.next.pixels.len());
    (
        &sample.current.pixels[..current_end],
        &sample.next.pixels[..next_end],
    )
}

pub const MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS: usize = 32;

pub fn shuffled_action_changed_pixel_ratio(
    samples: &[TransitionSample],
    shuffled_samples: &[TransitionSample],
    outcome_changing: &[Option<bool>],
    true_action_predictions: &[Vec<u8>],
    shuffled_action_predictions: &[Vec<u8>],
) -> Result<Option<f64>> {
    if samples.len() != true_action_predictions.len()
        || samples.len() != shuffled_samples.len()
        || samples.len() != outcome_changing.len()
        || samples.len() != shuffled_action_predictions.len()
    {
        bail!("gate changed-pixel rows do not match the sample count");
    }
    let known_outcome_changing = outcome_changing
        .iter()
        .copied()
        .try_fold(0usize, |count, changed| {
            changed.map(|changed| count + usize::from(changed))
        });
    if known_outcome_changing.is_some_and(|rows| rows < MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS) {
        return Ok(None);
    }
    let mut changed_pixels = 0usize;
    let mut true_correct = 0usize;
    let mut shuffled_correct = 0usize;
    for ((((sample, shuffled_sample), outcome_changing), true_prediction), shuffled_prediction) in
        samples
            .iter()
            .zip(shuffled_samples)
            .zip(outcome_changing)
            .zip(true_action_predictions)
            .zip(shuffled_action_predictions)
    {
        // Unknown counterfactuals retain the historical per-row behavior;
        // known outcome-equivalent tuples are not causal interventions.
        if sample.action == shuffled_sample.action
            || outcome_changing == &Some(false)
            || !is_board_changed_transition(sample)
        {
            continue;
        }
        let (current, target) = gameplay_pixels(sample, true_prediction.len());
        if current.len() != target.len()
            || true_prediction.len() != target.len()
            || shuffled_prediction.len() != target.len()
        {
            bail!("gate changed-pixel prediction width does not match gameplay target");
        }
        for (((before, after), true_pixel), shuffled_pixel) in current
            .iter()
            .zip(target)
            .zip(true_prediction)
            .zip(shuffled_prediction)
        {
            if before == after {
                continue;
            }
            changed_pixels += 1;
            true_correct += usize::from(true_pixel == after);
            shuffled_correct += usize::from(shuffled_pixel == after);
        }
    }
    if changed_pixels == 0 || true_correct == 0 {
        return Ok(None);
    }
    let true_accuracy = true_correct as f64 / changed_pixels as f64;
    let shuffled_accuracy = shuffled_correct as f64 / changed_pixels as f64;
    Ok(Some(shuffled_accuracy / true_accuracy))
}

pub fn foreground_reconstruction_accuracy(
    samples: &[TransitionSample],
    target_reconstructions: &[Vec<u8>],
) -> Result<Option<f64>> {
    if samples.len() != target_reconstructions.len() {
        bail!("gate foreground reconstruction rows do not match the sample count");
    }
    let mut foreground = 0usize;
    let mut correct = 0usize;
    for (sample, prediction) in samples.iter().zip(target_reconstructions) {
        let (_, target) = gameplay_pixels(sample, prediction.len());
        if prediction.len() != target.len() {
            bail!("gate foreground prediction width does not match gameplay target");
        }
        // Background is the row's rendered EMPTY colour (ADR 0005 §1.2), index 0
        // for legacy rows.
        let background = sample.provenance.background_color;
        for (predicted, target) in prediction.iter().zip(target) {
            if *target == background {
                continue;
            }
            foreground += 1;
            correct += usize::from(predicted == target);
        }
    }
    Ok((foreground > 0).then_some(correct as f64 / foreground as f64))
}

pub fn one_step_changed_exact(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
) -> Result<Option<f64>> {
    let (transitions, exact) = one_step_changed_exact_counts(samples, one_step_predictions)?;
    Ok((transitions > 0).then_some(exact as f64 / transitions as f64))
}

/// `(changed transitions, exactly decoded changed transitions)` behind
/// [`one_step_changed_exact`], so callers can pool rows across batches.
pub fn one_step_changed_exact_counts(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
) -> Result<(usize, usize)> {
    if samples.len() != one_step_predictions.len() {
        bail!("gate one-step rows do not match the sample count");
    }
    let mut transitions = 0usize;
    let mut exact = 0usize;
    for (sample, prediction) in samples.iter().zip(one_step_predictions) {
        if !is_board_changed_transition(sample) {
            continue;
        }
        let (current, target) = gameplay_pixels(sample, prediction.len());
        if prediction.len() != target.len() || current.len() != target.len() {
            bail!("gate one-step prediction width does not match gameplay target");
        }
        let mut changed_pixels = 0usize;
        let mut transition_exact = true;
        for ((before, after), predicted) in current.iter().zip(target).zip(prediction) {
            if before == after {
                continue;
            }
            changed_pixels += 1;
            transition_exact &= predicted == after;
        }
        if changed_pixels > 0 {
            transitions += 1;
            exact += usize::from(transition_exact);
        }
    }
    Ok((transitions, exact))
}

/// ADR 0003 §6: evaluate the foundation-v2 gate metrics on each named V5
/// held-out split with exact content-mask sidecars, so offline reports cover
/// the actual held-out populations (unseen seed, composition, translation,
/// size, held-out operator families) instead of only the legacy
/// origin-aligned curriculum rows. Selection-only; each population uses an
/// eval-domain seed distinct from the reserved in-trainer gate seed.
fn foundation_v2_v5_holdout_gates(
    model: &WorldModel,
    cfg: &EvalConfig,
    train_cfg: &TrainConfig,
    device: &Device,
    profile: Option<&RepresentativeUpdateCapture>,
) -> Result<(
    BTreeMap<String, GateSupportMetrics>,
    Option<BTreeMap<String, BTreeMap<String, GateSupportMetrics>>>,
    Vec<TransitionSample>,
    Vec<ContentMask>,
    Vec<V5SampleProvenance>,
)> {
    const V5_HOLDOUT_ROWS: usize = 512;
    const V5_HOLDOUT_SEED_DOMAIN: u64 = 0x5645_4C35_1D00_0000;
    let mut splits: Vec<(String, V5DataSplit)> = vec![
        ("unseen_seed_7x7".into(), V5DataSplit::UnseenSeed7x7),
        ("composition_8x8".into(), V5DataSplit::Composition8x8),
        ("translated_7x7".into(), V5DataSplit::Translated7x7),
        ("size_16x16".into(), V5DataSplit::Size16x16),
    ];
    for family in OperatorFamilySplit::default().held_out {
        splits.push((
            format!("held_out_operator_{family:?}").to_lowercase(),
            V5DataSplit::HeldOutOperator(family),
        ));
    }
    let mut gates = BTreeMap::new();
    let mut context_strata = model.config().world_core_v6.then(BTreeMap::new);
    let mut ablation_population = None;
    // ADR 0005 §2: a v6 data contract draws the held-out rows under the same
    // whole-frame rendering and stream schedule (LearningHistories with real
    // context windows) as training; legacy stays byte-identical.
    let train_seed = train_cfg.seed;
    let schedule = if train_cfg.data_contract_v6 {
        adaptation_v6_stream_schedule
    } else {
        foundation_v2_stream_schedule
    };
    for (lane, (name, split)) in splits.into_iter().enumerate() {
        let seed = cfg
            .iid_seed
            .wrapping_add(V5_HOLDOUT_SEED_DOMAIN)
            .wrapping_add(lane as u64);
        if seed == crate::p2::train::FOUNDATION_V2_GATE_SEED {
            bail!("v5 holdout eval seed collides with the reserved in-trainer gate seed");
        }
        if seed == train_seed {
            bail!(
                "v5 holdout eval seed for {name} collides with the checkpoint's \
                 training seed; the unseen-seed claim would be false"
            );
        }
        let batch = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: V5_HOLDOUT_ROWS,
                seed,
                schedule,
                data_contract_v6: train_cfg.data_contract_v6,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            split,
        )?;
        let samples = batch.transitions().cloned().collect::<Vec<_>>();
        let masks = batch
            .samples()
            .iter()
            .map(|sample| sample.content_mask.clone())
            .collect::<Vec<_>>();
        let provenance = batch
            .samples()
            .iter()
            .map(|sample| sample.provenance.clone())
            .collect::<Vec<_>>();
        let metrics = evaluate_gate_support_impl(
            model,
            &samples,
            Some(&masks),
            Some(&provenance),
            device,
            None,
            if lane == 0 { profile } else { None },
        )?;
        if lane == 0 {
            ablation_population = Some((samples.clone(), masks.clone(), provenance.clone()));
        }
        if let Some(strata) = context_strata.as_mut() {
            strata.insert(
                name.clone(),
                foundation_v2_context_strata(model, &samples, &masks, &provenance, device)?,
            );
        }
        gates.insert(name, metrics);
    }
    let (samples, masks, provenance) =
        ablation_population.expect("unseen-seed V5 holdout population is always present");
    Ok((gates, context_strata, samples, masks, provenance))
}

/// Full-transition exactness on the same changed-transition rows counted by
/// `one_step_changed_exact`: every status-excluded gameplay pixel, unchanged
/// pixels included, must be decoded exactly.
pub fn one_step_full_exact(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
) -> Result<Option<f64>> {
    if samples.len() != one_step_predictions.len() {
        bail!("gate one-step rows do not match the sample count");
    }
    let mut transitions = 0usize;
    let mut exact = 0usize;
    for (sample, prediction) in samples.iter().zip(one_step_predictions) {
        if !is_board_changed_transition(sample) {
            continue;
        }
        let (current, target) = gameplay_pixels(sample, prediction.len());
        if prediction.len() != target.len() || current.len() != target.len() {
            bail!("gate one-step prediction width does not match gameplay target");
        }
        let changed_pixels = current.iter().zip(target).filter(|(a, b)| a != b).count();
        if changed_pixels == 0 {
            continue;
        }
        transitions += 1;
        exact += usize::from(prediction.iter().zip(target).all(|(a, b)| a == b));
    }
    Ok((transitions > 0).then_some(exact as f64 / transitions as f64))
}

/// Full-frame exactness over every supplied row, no-op transitions included:
/// the fraction of rows whose entire status-excluded gameplay frame is
/// predicted exactly. A copy policy scores every genuinely unchanged row
/// correct here, so hallucinations on no-op rows depress this metric even
/// when changed-row exactness improves.
pub fn one_step_all_rows_exact(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
) -> Result<Option<f64>> {
    if samples.len() != one_step_predictions.len() {
        bail!("gate one-step rows do not match the sample count");
    }
    let mut exact = 0usize;
    for (sample, prediction) in samples.iter().zip(one_step_predictions) {
        let (current, target) = gameplay_pixels(sample, prediction.len());
        if prediction.len() != target.len() || current.len() != target.len() {
            bail!("gate one-step prediction width does not match gameplay target");
        }
        exact += usize::from(prediction.iter().zip(target).all(|(a, b)| a == b));
    }
    Ok((!samples.is_empty()).then_some(exact as f64 / samples.len() as f64))
}

/// Fraction of factually unchanged gameplay pixels edited by the one-step
/// decode, measured over every supplied row (`1 -` unchanged-pixel accuracy).
pub fn one_step_false_edit_rate(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
    padding: bool,
) -> Result<Option<f64>> {
    one_step_false_edit_rate_with_content_masks(samples, one_step_predictions, None, padding)
}

pub fn one_step_false_edit_rate_with_content_masks(
    samples: &[TransitionSample],
    one_step_predictions: &[Vec<u8>],
    content_masks: Option<&[ContentMask]>,
    padding: bool,
) -> Result<Option<f64>> {
    if samples.len() != one_step_predictions.len() {
        bail!("gate one-step rows do not match the sample count");
    }
    if content_masks.is_some_and(|masks| masks.len() != samples.len()) {
        bail!("gate content-mask rows do not match the sample count");
    }
    if content_masks.is_some_and(|masks| {
        masks
            .iter()
            .any(|mask| mask.values.len() != FRAME_SIDE * FRAME_SIDE)
    }) {
        bail!("gate content mask is not fixed 64x64");
    }
    let mut unchanged = 0usize;
    let mut edited = 0usize;
    for (row, (sample, prediction)) in samples.iter().zip(one_step_predictions).enumerate() {
        let (current, target) = gameplay_pixels(sample, prediction.len());
        if prediction.len() != target.len() || current.len() != target.len() {
            bail!("gate one-step prediction width does not match gameplay target");
        }
        let content_x = usize::from(sample.provenance.content_x);
        let content_y = usize::from(sample.provenance.content_y);
        let content_width = usize::from(sample.provenance.content_width).min(FRAME_SIDE);
        let content_height =
            usize::from(sample.provenance.content_height).min(target.len() / FRAME_SIDE);
        for (index, ((before, after), predicted)) in
            current.iter().zip(target).zip(prediction).enumerate()
        {
            if before != after {
                continue;
            }
            let in_content = match content_masks {
                Some(masks) => masks[row].values[index] != 0,
                None => {
                    let x = index % FRAME_SIDE;
                    let y = index / FRAME_SIDE;
                    (content_x..content_x + content_width).contains(&x)
                        && (content_y..content_y + content_height).contains(&y)
                }
            };
            if in_content == padding {
                continue;
            }
            unchanged += 1;
            edited += usize::from(predicted != after);
        }
    }
    Ok((unchanged > 0).then_some(edited as f64 / unchanged as f64))
}

fn exact_palette_predictions(model: &WorldModel, latent: &Tensor) -> Result<Vec<Vec<u8>>> {
    let batch = latent.dim(0)?;
    model
        .exact_gameplay_logits(latent)?
        .argmax(D::Minus1)?
        .reshape((batch, ()))?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()
        .map_err(Into::into)
}

/// Evaluate all four automated run-gate measurements on one fixed batch.
/// Callers own and persist the held-out population; this function never
/// samples or mutates it.
pub fn evaluate_gate_support(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
) -> Result<GateSupportMetrics> {
    evaluate_gate_support_with_content_masks(model, samples, None, device)
}

/// Peak-memory bound for held-out gate forwards (ADR 0005). A v6 row carries
/// up to `CONTEXT_WINDOW_MAX` context frame pairs, so one 512-row forward runs
/// the frame encoder over thousands of frames and exceeds the envelope of a
/// training step. Chunking keeps the gate evaluation inside that envelope;
/// every op on the path is per-row, so results are unchanged.
pub const FOUNDATION_V2_GATE_PHYSICAL_BATCH: usize = 32;

#[allow(clippy::too_many_arguments)]
fn forward_gate_rows_chunked(
    model: &WorldModel,
    current: &Tensor,
    actions: &Tensor,
    action_coords: &Tensor,
    goals: &Tensor,
    operator_conditioning: &Tensor,
    context: Option<&ContextBatch>,
    depth: RecursionDepth,
    chunk: usize,
) -> Result<Tensor> {
    let rows = current.dim(0)?;
    let chunk = chunk.max(1);
    // Evaluation never back-propagates, but every candle op output keeps its
    // inputs alive as autograd history. Without `detach`, concatenating the
    // chunk outputs would pin every chunk's activations (context encoder,
    // all recursion steps) at once, and chunking would bound nothing.
    if rows <= chunk {
        return Ok(model
            .forward_from_latent_with_depth_and_operator_conditioning_with_context(
                current,
                actions,
                action_coords,
                goals,
                operator_conditioning,
                context,
                depth,
            )?
            .y
            .detach());
    }
    let mut parts = Vec::with_capacity(rows.div_ceil(chunk));
    let mut start = 0;
    while start < rows {
        let len = chunk.min(rows - start);
        let indices = (start..start + len)
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        let chunk_context = match context {
            Some(context) => context.select_rows(&indices)?,
            None => None,
        };
        parts.push(
            model
                .forward_from_latent_with_depth_and_operator_conditioning_with_context(
                    &current.narrow(0, start, len)?.contiguous()?,
                    &actions.narrow(0, start, len)?.contiguous()?,
                    &action_coords.narrow(0, start, len)?.contiguous()?,
                    &goals.narrow(0, start, len)?.contiguous()?,
                    &operator_conditioning.narrow(0, start, len)?.contiguous()?,
                    chunk_context.as_ref(),
                    depth,
                )?
                .y
                .detach(),
        );
        start += len;
    }
    Tensor::cat(&parts, 0).map_err(Into::into)
}

struct EncodedGateSupportPopulation {
    batch: BatchTensors,
    current: Tensor,
    target: Tensor,
    shuffled: ShuffledActionControlPopulation,
    shuffled_actions: Tensor,
    shuffled_coords: Tensor,
    /// ADR 0005 context windows, consumed only by `world_core_v6` models.
    context: Option<ContextBatch>,
}

fn encode_gate_support_population(
    model: &WorldModel,
    samples: &[TransitionSample],
    provenance: Option<&[V5SampleProvenance]>,
    device: &Device,
) -> Result<EncodedGateSupportPopulation> {
    let batch = batch_from_samples(samples, device)?;
    // Detached: the encoder graph over the whole population is never needed.
    let (current, target) = model.encode_state_pair(&batch.frames, &batch.next_frames)?;
    let (current, target) = (current.detach(), target.detach());
    let shuffled = shuffled_action_control_population(samples, provenance)?;
    let (shuffled_actions, shuffled_coords) =
        action_tensors_from_samples(&shuffled.samples, device)?;
    let context = model
        .config()
        .world_core_v6
        .then(|| ContextBatch::from_samples(samples, device))
        .transpose()?
        .flatten();
    Ok(EncodedGateSupportPopulation {
        batch,
        current,
        target,
        shuffled,
        shuffled_actions,
        shuffled_coords,
        context,
    })
}

/// ADR 0005 §7 evaluation stratum of a row's context window length.
pub fn context_len_stratum(context_len: usize) -> &'static str {
    match context_len {
        0 => "0",
        1..=4 => "1-4",
        _ => "5-16",
    }
}

/// Gate metrics of one population split by `context_len` stratum, keyed by
/// [`context_len_stratum`]. Reported only for `world_core_v6` checkpoints.
fn foundation_v2_context_strata(
    model: &WorldModel,
    samples: &[TransitionSample],
    masks: &[ContentMask],
    provenance: &[V5SampleProvenance],
    device: &Device,
) -> Result<BTreeMap<String, GateSupportMetrics>> {
    let mut strata = BTreeMap::<String, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        strata
            .entry(context_len_stratum(sample.context.len()).to_string())
            .or_default()
            .push(index);
    }
    strata
        .into_iter()
        .map(|(name, rows)| {
            let pick = |index: &usize| rows.contains(index);
            let samples = samples
                .iter()
                .enumerate()
                .filter(|(index, _)| pick(index))
                .map(|(_, sample)| sample.clone())
                .collect::<Vec<_>>();
            let masks = rows
                .iter()
                .map(|row| masks[*row].clone())
                .collect::<Vec<_>>();
            let provenance = rows
                .iter()
                .map(|row| provenance[*row].clone())
                .collect::<Vec<_>>();
            let metrics = evaluate_gate_support_impl(
                model,
                &samples,
                Some(&masks),
                Some(&provenance),
                device,
                None,
                None,
            )?;
            Ok((name, metrics))
        })
        .collect()
}

/// ADR 0005 §5.1 changed-exact comparison of one row set scored with its
/// context windows and with the windows masked to `K = 0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAblationMetrics {
    pub rows: usize,
    /// Rows whose window is non-empty; the masked pass only differs on these.
    pub rows_with_context: usize,
    pub changed_transitions: usize,
    pub changed_exact_with_context: Option<f64>,
    pub changed_exact_without_context: Option<f64>,
    /// `with - without`; `None` when either side has no changed transition.
    pub delta: Option<f64>,
    pub composed_changed_exact_with_context: Option<f64>,
    pub composed_changed_exact_without_context: Option<f64>,
    pub composed_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAblationStratum {
    /// [`context_len_stratum`] of the rows' original context length.
    pub stratum: String,
    #[serde(flatten)]
    pub metrics: ContextAblationMetrics,
}

/// ADR 0005 §5.1 memorization diagnostic report. The masked pass drops every
/// row's `context` before the `ContextBatch` is built, so those rows are
/// scored exactly like an all-`K = 0` population (no context FiLM).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAblationReport {
    /// Held-out population the rows come from (`v5_holdout_gates` key).
    pub population: String,
    /// Fingerprint of the with-context rows (matches the gate population).
    pub population_fingerprint: String,
    pub evidence_class: String,
    #[serde(flatten)]
    pub overall: ContextAblationMetrics,
    /// Ordered by stratum key (`"0"`, `"1-4"`, `"5-16"`).
    pub per_stratum: Vec<ContextAblationStratum>,
}

/// The same rows with every context window dropped (`K = 0`); the V5
/// sidecar's `context_len` is masked alongside so row and provenance agree.
fn context_masked_rows(
    samples: &[TransitionSample],
    provenance: &[V5SampleProvenance],
) -> (Vec<TransitionSample>, Vec<V5SampleProvenance>) {
    let rows = samples
        .iter()
        .cloned()
        .map(|mut sample| {
            sample.context.clear();
            sample.provenance.context_len = 0;
            sample
        })
        .collect();
    let provenance = provenance
        .iter()
        .cloned()
        .map(|mut provenance| {
            provenance.source.context_len = 0;
            provenance
        })
        .collect();
    (rows, provenance)
}

fn context_ablation_metrics(
    samples: &[TransitionSample],
    with_context: &GateSupportMetrics,
    without_context: &GateSupportMetrics,
) -> ContextAblationMetrics {
    let delta = |with: Option<f64>, without: Option<f64>| Some(with? - without?);
    ContextAblationMetrics {
        rows: samples.len(),
        rows_with_context: samples
            .iter()
            .filter(|sample| !sample.context.is_empty())
            .count(),
        changed_transitions: with_context.changed_transitions,
        changed_exact_with_context: with_context.one_step_changed_exact,
        changed_exact_without_context: without_context.one_step_changed_exact,
        delta: delta(
            with_context.one_step_changed_exact,
            without_context.one_step_changed_exact,
        ),
        composed_changed_exact_with_context: with_context.one_step_composed_changed_exact,
        composed_changed_exact_without_context: without_context.one_step_composed_changed_exact,
        composed_delta: delta(
            with_context.one_step_composed_changed_exact,
            without_context.one_step_composed_changed_exact,
        ),
    }
}

/// ADR 0005 §5.1: score `samples` with their context windows and again with
/// every window masked to `K = 0`, overall and per [`context_len_stratum`] of
/// the original window length. `world_core_v6` checkpoints only.
fn evaluate_context_ablation(
    model: &WorldModel,
    population: &str,
    samples: &[TransitionSample],
    masks: &[ContentMask],
    provenance: &[V5SampleProvenance],
    device: &Device,
) -> Result<ContextAblationReport> {
    if !model.config().world_core_v6 {
        bail!("the context ablation requires a world_core_v6 checkpoint");
    }
    let score = |rows: &[TransitionSample], masks: &[ContentMask], prov: &[V5SampleProvenance]| {
        let with_context =
            evaluate_gate_support_impl(model, rows, Some(masks), Some(prov), device, None, None)?;
        let (masked, masked_prov) = context_masked_rows(rows, prov);
        let without_context = evaluate_gate_support_impl(
            model,
            &masked,
            Some(masks),
            Some(&masked_prov),
            device,
            None,
            None,
        )?;
        Ok::<_, anyhow::Error>((
            context_ablation_metrics(rows, &with_context, &without_context),
            with_context.population_fingerprint,
        ))
    };
    let (overall, population_fingerprint) = score(samples, masks, provenance)?;
    let mut strata = BTreeMap::<String, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        strata
            .entry(context_len_stratum(sample.context.len()).to_string())
            .or_default()
            .push(index);
    }
    let per_stratum = strata
        .into_iter()
        .map(|(stratum, rows)| {
            let pick = |source: &[TransitionSample]| {
                rows.iter()
                    .map(|row| source[*row].clone())
                    .collect::<Vec<_>>()
            };
            let masks = rows
                .iter()
                .map(|row| masks[*row].clone())
                .collect::<Vec<_>>();
            let prov = rows
                .iter()
                .map(|row| provenance[*row].clone())
                .collect::<Vec<_>>();
            let (metrics, _) = score(&pick(samples), &masks, &prov)?;
            Ok(ContextAblationStratum { stratum, metrics })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ContextAblationReport {
        population: population.into(),
        population_fingerprint,
        evidence_class: "selection_only".into(),
        overall,
        per_stratum,
    })
}

// ---- ADR 0005 §5.2 adaptation falsifier (E3) -------------------------------

/// Preregistered held-out population seed of the §5.2 falsifier.
pub const ADAPTATION_FALSIFIER_POPULATION_SEED: u64 = 1_000_002;
/// Preregistered adapter seed: the reservoir sampler is reseeded with it at
/// every episode start, so each episode's adaptation batches are a function
/// of that episode's own observation sequence only.
pub const ADAPTATION_FALSIFIER_ADAPTER_SEED: u64 = 7;
/// Prefix lengths `t`: adapt on chronological transitions `1..t`, score
/// `t+1..t+ADAPTATION_FALSIFIER_SCORE_WINDOW` (1-based, preregistered).
pub const ADAPTATION_FALSIFIER_PREFIX_LENGTHS: [usize; 3] = [8, 16, 32];
pub const ADAPTATION_FALSIFIER_SCORE_WINDOW: usize = 4;
/// Promotion threshold on prequential changed-exact (absolute, preregistered).
pub const ADAPTATION_FALSIFIER_PROMOTION_DELTA: f64 = 0.02;
/// Absolute tolerance on the threshold comparison so a delta that is `0.02`
/// up to floating-point rounding of the two fractions is not rejected.
const ADAPTATION_FALSIFIER_DELTA_TOLERANCE: f64 = 1e-9;

pub const ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY: &str = "context_only";
pub const ADAPTATION_FALSIFIER_ARM_RESET: &str = "reset";
pub const ADAPTATION_FALSIFIER_ARM_CARRY: &str = "carry";

/// Knobs of one falsifier run. Production uses [`Self::preregistered`]; tests
/// shrink the population and the prefix lengths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierSpec {
    pub episodes: usize,
    pub population_seed: u64,
    pub adapter_seed: u64,
    pub prefix_lengths: Vec<usize>,
    pub score_window: usize,
    /// Channel B warm-up (§6.2 default [`ADAPT_MIN_LEVEL_TRANSITIONS`]).
    pub min_level_transitions: usize,
}

impl AdaptationFalsifierSpec {
    pub fn preregistered(episodes: usize, min_level_transitions: usize) -> Self {
        Self {
            episodes,
            population_seed: ADAPTATION_FALSIFIER_POPULATION_SEED,
            adapter_seed: ADAPTATION_FALSIFIER_ADAPTER_SEED,
            prefix_lengths: ADAPTATION_FALSIFIER_PREFIX_LENGTHS.to_vec(),
            score_window: ADAPTATION_FALSIFIER_SCORE_WINDOW,
            min_level_transitions,
        }
    }
}

/// Channel B telemetry pooled over episodes, cumulative through the prefix
/// at which the rows were scored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierTelemetry {
    pub updates: usize,
    pub skipped: usize,
    pub reverted: usize,
    /// `maybe_update` calls that ended in the §6.2 warm-up (`warmup` note).
    pub warmup_calls: usize,
    pub level_step_cap_calls: usize,
    /// Largest `||theta - theta_0||^2` over the fast subset at scoring time.
    pub max_drift_from_prior: f64,
}

/// One arm scored at one prefix length over every episode long enough.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierArmMetrics {
    pub arm: String,
    pub prefix_len: usize,
    /// Episodes whose chronological length covers `prefix_len + score_window`.
    pub episodes: usize,
    pub rows: usize,
    /// Rows entering the changed-exact denominator (board-changed rows with at
    /// least one changed gameplay pixel), as in the gate metrics.
    pub changed_transitions: usize,
    pub changed_exact: Option<f64>,
    pub composed_changed_exact: Option<f64>,
    pub adaptation: AdaptationFalsifierTelemetry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierDelta {
    pub prefix_len: usize,
    /// `changed_exact(arm) - changed_exact(context_only)` on the same rows.
    pub changed_exact_delta: Option<f64>,
    pub composed_changed_exact_delta: Option<f64>,
    /// `changed_exact_delta >= promotion_delta`.
    pub improvement: bool,
    /// Frozen-after-adaptation check: neither changed-exact nor composed
    /// changed-exact of the adapted-then-frozen weights is below the prior.
    pub not_worse: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierArmVerdict {
    pub arm: String,
    pub deltas: Vec<AdaptationFalsifierDelta>,
    pub improvement_at_every_t: bool,
    pub not_worse_at_every_t: bool,
    pub promote: bool,
}

/// The preregistered §5.2 rule applied to the pooled metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierVerdict {
    pub promotion_delta: f64,
    /// Prefix lengths with at least one changed transition in the
    /// context-only arm; the rule is applied over exactly these.
    pub evaluated_prefix_lengths: Vec<usize>,
    /// Preregistered prefix lengths no episode was long enough for (or with
    /// no changed transition). The rule cannot be evaluated there.
    pub skipped_prefix_lengths: Vec<usize>,
    pub arms: Vec<AdaptationFalsifierArmVerdict>,
    pub promote_channel_b: bool,
    pub satisfied_by: Vec<String>,
    pub note: String,
}

/// ADR 0005 §5.2 adaptation falsifier report (`--adaptation-falsifier`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationFalsifierReport {
    pub population: String,
    pub population_seed: u64,
    pub adapter_seed: u64,
    pub episodes: usize,
    /// SHA-256 over every episode's chronological transitions as scored
    /// (augmented frames, actions, level indices, changed flags).
    pub population_fingerprint: String,
    pub evidence_class: String,
    pub prefix_lengths: Vec<usize>,
    pub score_window: usize,
    pub context_window_max: usize,
    /// Channel B warm-up in force (§6.2 default 8; anything else is a
    /// recorded deviation).
    pub min_level_transitions: usize,
    /// Histogram of chronological episode lengths (`len` -> episodes).
    pub chronological_lengths: BTreeMap<String, usize>,
    /// Ordered arm-major, then by prefix length.
    pub arms: Vec<AdaptationFalsifierArmMetrics>,
    pub verdict: AdaptationFalsifierVerdict,
}

/// Stream configuration under which the falsifier population is rendered: the
/// v6 unit augmentation of the `LearningHistories` stream with the default
/// operator-family split, seeded by the preregistered population seed.
fn adaptation_falsifier_stream_config(population_seed: u64) -> MixedStreamConfig {
    MixedStreamConfig {
        seed: population_seed,
        schedule: adaptation_v6_stream_schedule,
        data_contract_v6: true,
        ..MixedStreamConfig::default()
    }
}

fn adaptation_falsifier_population(
    spec: &AdaptationFalsifierSpec,
) -> Result<Vec<AugmentedLearningHistory>> {
    let stream_cfg = adaptation_falsifier_stream_config(spec.population_seed);
    stream_cfg.validate()?;
    (0..spec.episodes)
        .into_par_iter()
        .map(|meta_episode_id| {
            augmented_learning_history(
                &stream_cfg,
                V5DataSplit::UnseenSeed7x7,
                meta_episode_id as u64,
            )
        })
        .collect()
}

fn adaptation_falsifier_population_fingerprint(histories: &[AugmentedLearningHistory]) -> String {
    let mut hash = Sha256::new();
    for history in histories {
        hash.update(history.meta_episode_id.to_le_bytes());
        hash.update((history.levels as u64).to_le_bytes());
        hash.update((history.chronological.len() as u64).to_le_bytes());
        for position in 0..history.chronological.len() {
            let row = &history.chronological_row(position).transition;
            hash.update(row.provenance.level_index.to_le_bytes());
            hash.update(&row.current.pixels[..]);
            hash.update([
                row.action.id,
                row.action.x.unwrap_or(0),
                row.action.y.unwrap_or(0),
                u8::from(is_board_changed_transition(row)),
            ]);
            hash.update(&row.next.pixels[..]);
        }
    }
    format!("sha256:{:x}", hash.finalize())
}

/// Scoring rows for prefix `t`: chronological positions `t..t+window`
/// (0-based), each carrying Channel A's window, the last
/// `<= CONTEXT_WINDOW_MAX` chronological transitions before it (episode
/// scope, identical for every arm; the arms differ only in the weights).
fn adaptation_falsifier_scoring_rows(
    history: &AugmentedLearningHistory,
    t: usize,
    window: usize,
) -> Vec<TransitionSample> {
    (t..t + window)
        .map(|position| {
            let mut row = history.chronological_row(position).transition.clone();
            row.context = history.context_window_before(position);
            row.provenance.context_len =
                u8::try_from(row.context.len()).expect("context window fits u8");
            row
        })
        .collect()
}

struct AdaptationFalsifierDecodes {
    true_predictions: Vec<Vec<u8>>,
    composed: Vec<Vec<u8>>,
}

/// One-step exact decodes of `rows` under the model's current weights on the
/// gate path: `encode_state_pair` -> [`forward_gate_rows_chunked`] (detached,
/// `<= FOUNDATION_V2_GATE_PHYSICAL_BATCH` rows per forward) -> raw palette
/// argmax and [`WorldModel::composed_gameplay_decode`]. Nothing here retains a
/// tensor beyond the call.
fn adaptation_falsifier_decode_rows(
    model: &WorldModel,
    rows: &[TransitionSample],
    device: &Device,
) -> Result<AdaptationFalsifierDecodes> {
    let batch = batch_from_samples(rows, device)?;
    let (current, _target) = model.encode_state_pair(&batch.frames, &batch.next_frames)?;
    let current = current.detach();
    let prediction = forward_gate_rows_chunked(
        model,
        &current,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        &batch.operator_conditioning,
        batch.context.as_ref(),
        RecursionDepth::from_config(model.config()),
        FOUNDATION_V2_GATE_PHYSICAL_BATCH,
    )?;
    let true_predictions = exact_palette_predictions(model, &prediction)?;
    let composed = model
        .composed_gameplay_decode(&prediction, &batch.frames)?
        .reshape((rows.len(), ()))?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()?;
    Ok(AdaptationFalsifierDecodes {
        true_predictions,
        composed,
    })
}

#[derive(Debug, Default)]
struct AdaptationFalsifierAccum {
    episodes: usize,
    rows: usize,
    changed: usize,
    exact: usize,
    composed_exact: usize,
    telemetry: AdaptationFalsifierTelemetry,
}

impl AdaptationFalsifierAccum {
    fn add(
        &mut self,
        rows: &[TransitionSample],
        decodes: &AdaptationFalsifierDecodes,
        telemetry: &AdaptationFalsifierTelemetry,
    ) -> Result<()> {
        let (changed, exact) = one_step_changed_exact_counts(rows, &decodes.true_predictions)?;
        let (changed_composed, composed_exact) =
            one_step_changed_exact_counts(rows, &decodes.composed)?;
        if changed != changed_composed {
            bail!("raw and composed decodes disagree on the changed-transition count");
        }
        self.episodes += 1;
        self.rows += rows.len();
        self.changed += changed;
        self.exact += exact;
        self.composed_exact += composed_exact;
        self.telemetry.updates += telemetry.updates;
        self.telemetry.skipped += telemetry.skipped;
        self.telemetry.reverted += telemetry.reverted;
        self.telemetry.warmup_calls += telemetry.warmup_calls;
        self.telemetry.level_step_cap_calls += telemetry.level_step_cap_calls;
        self.telemetry.max_drift_from_prior = self
            .telemetry
            .max_drift_from_prior
            .max(telemetry.max_drift_from_prior);
        Ok(())
    }

    fn metrics(&self, arm: &str, prefix_len: usize) -> AdaptationFalsifierArmMetrics {
        let fraction =
            |exact: usize| (self.changed > 0).then(|| exact as f64 / self.changed as f64);
        AdaptationFalsifierArmMetrics {
            arm: arm.into(),
            prefix_len,
            episodes: self.episodes,
            rows: self.rows,
            changed_transitions: self.changed,
            changed_exact: fraction(self.exact),
            composed_changed_exact: fraction(self.composed_exact),
            adaptation: self.telemetry.clone(),
        }
    }
}

/// Prequential pass over one episode. With `adapter = None` this is Channel A
/// (frozen prior weights). With an adapter it is the live loop of §6.2:
/// observe each chronological transition in order, `maybe_update` after each,
/// `on_level_transition` at every level boundary; at each prefix length the
/// next `score_window` rows are scored with the weights as adapted so far and
/// no further update (frozen-after-adaptation). The caller owns
/// `begin_game` / `restore_prior` around this function.
fn adaptation_falsifier_episode(
    model: &WorldModel,
    device: &Device,
    history: &AugmentedLearningHistory,
    spec: &AdaptationFalsifierSpec,
    mut adapter: Option<&mut FastWeightAdapter<'_>>,
    accums: &mut BTreeMap<usize, AdaptationFalsifierAccum>,
) -> Result<()> {
    let len = history.chronological.len();
    let mut telemetry = AdaptationFalsifierTelemetry::default();
    let mut level: Option<u16> = None;
    for position in 0..len {
        if spec.prefix_lengths.contains(&position) && position + spec.score_window <= len {
            let rows = adaptation_falsifier_scoring_rows(history, position, spec.score_window);
            let decodes = adaptation_falsifier_decode_rows(model, &rows, device)?;
            telemetry.max_drift_from_prior = match adapter.as_deref() {
                Some(adapter) => adapter.drift_from_prior()?,
                None => 0.0,
            };
            accums
                .get_mut(&position)
                .ok_or_else(|| anyhow::anyhow!("no accumulator for prefix {position}"))?
                .add(&rows, &decodes, &telemetry)?;
        }
        let Some(adapter) = adapter.as_deref_mut() else {
            continue;
        };
        let row = &history.chronological_row(position).transition;
        let row_level = row.provenance.level_index;
        if level.is_some_and(|current| current != row_level) {
            adapter.on_level_transition(row_level)?;
        }
        level = Some(row_level);
        adapter.observe(&row.current, &row.action, &row.next, row_level);
        if let Some(trace) = adapter.maybe_update()? {
            telemetry.updates += trace.updates;
            telemetry.skipped += trace.skipped;
            telemetry.reverted += trace.reverted;
            match trace.note.as_deref() {
                Some("warmup") => telemetry.warmup_calls += 1,
                Some("level_step_cap") => telemetry.level_step_cap_calls += 1,
                _ => {}
            }
        }
    }
    Ok(())
}

fn adaptation_falsifier_arm(
    model: &WorldModel,
    device: &Device,
    histories: &[AugmentedLearningHistory],
    spec: &AdaptationFalsifierSpec,
    mut adapter: Option<&mut FastWeightAdapter<'_>>,
) -> Result<BTreeMap<usize, AdaptationFalsifierAccum>> {
    let mut accums = spec
        .prefix_lengths
        .iter()
        .map(|t| (*t, AdaptationFalsifierAccum::default()))
        .collect::<BTreeMap<_, _>>();
    for history in histories {
        if let Some(adapter) = adapter.as_deref_mut() {
            adapter.begin_game()?;
            adapter.reseed_reservoir(spec.adapter_seed);
        }
        adaptation_falsifier_episode(
            model,
            device,
            history,
            spec,
            adapter.as_deref_mut(),
            &mut accums,
        )?;
        if let Some(adapter) = adapter.as_deref_mut() {
            adapter.restore_prior()?;
        }
    }
    Ok(accums)
}

/// The preregistered §5.2 rule over pooled arm metrics: promote Channel B only
/// if some adapted arm improves changed-exact over `context_only` by at least
/// `promotion_delta` at every evaluated prefix length AND is not worse than
/// the prior (changed-exact and composed changed-exact) at every one. Prefix
/// lengths without a changed transition in the context-only arm are skipped
/// and listed; if none remains, nothing is promoted.
pub fn adaptation_falsifier_verdict(
    arms: &[AdaptationFalsifierArmMetrics],
    promotion_delta: f64,
) -> AdaptationFalsifierVerdict {
    let baseline = arms
        .iter()
        .filter(|metrics| metrics.arm == ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY)
        .map(|metrics| (metrics.prefix_len, metrics))
        .collect::<BTreeMap<_, _>>();
    let prefix_lengths = arms
        .iter()
        .map(|metrics| metrics.prefix_len)
        .collect::<BTreeSet<_>>();
    let evaluated = prefix_lengths
        .iter()
        .copied()
        .filter(|t| baseline.get(t).is_some_and(|m| m.changed_transitions > 0))
        .collect::<Vec<_>>();
    let skipped = prefix_lengths
        .iter()
        .copied()
        .filter(|t| !evaluated.contains(t))
        .collect::<Vec<_>>();
    let delta = |arm: Option<f64>, prior: Option<f64>| Some(arm? - prior?);
    let not_worse = |arm: Option<f64>, prior: Option<f64>| match (arm, prior) {
        (Some(arm), Some(prior)) => arm >= prior,
        (None, None) => true,
        _ => false,
    };
    let mut arm_names = Vec::new();
    for metrics in arms {
        if metrics.arm != ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY && !arm_names.contains(&metrics.arm)
        {
            arm_names.push(metrics.arm.clone());
        }
    }
    let arm_verdicts = arm_names
        .iter()
        .map(|arm| {
            let deltas = evaluated
                .iter()
                .map(|t| {
                    let prior = baseline[t];
                    let adapted = arms
                        .iter()
                        .find(|metrics| &metrics.arm == arm && metrics.prefix_len == *t);
                    match adapted {
                        Some(adapted) => {
                            let changed_exact_delta =
                                delta(adapted.changed_exact, prior.changed_exact);
                            AdaptationFalsifierDelta {
                                prefix_len: *t,
                                changed_exact_delta,
                                composed_changed_exact_delta: delta(
                                    adapted.composed_changed_exact,
                                    prior.composed_changed_exact,
                                ),
                                improvement: changed_exact_delta.is_some_and(|d| {
                                    d + ADAPTATION_FALSIFIER_DELTA_TOLERANCE >= promotion_delta
                                }),
                                not_worse: not_worse(adapted.changed_exact, prior.changed_exact)
                                    && not_worse(
                                        adapted.composed_changed_exact,
                                        prior.composed_changed_exact,
                                    ),
                            }
                        }
                        None => AdaptationFalsifierDelta {
                            prefix_len: *t,
                            changed_exact_delta: None,
                            composed_changed_exact_delta: None,
                            improvement: false,
                            not_worse: false,
                        },
                    }
                })
                .collect::<Vec<_>>();
            let improvement_at_every_t = !deltas.is_empty() && deltas.iter().all(|d| d.improvement);
            let not_worse_at_every_t = !deltas.is_empty() && deltas.iter().all(|d| d.not_worse);
            AdaptationFalsifierArmVerdict {
                arm: arm.clone(),
                deltas,
                improvement_at_every_t,
                not_worse_at_every_t,
                promote: improvement_at_every_t && not_worse_at_every_t,
            }
        })
        .collect::<Vec<_>>();
    let satisfied_by = arm_verdicts
        .iter()
        .filter(|verdict| verdict.promote)
        .map(|verdict| verdict.arm.clone())
        .collect::<Vec<_>>();
    let promote_channel_b = !satisfied_by.is_empty();
    let note = if evaluated.is_empty() {
        "no prefix length had a changed transition in the context-only arm; the rule \
         cannot be evaluated and Channel B stays disabled"
            .to_string()
    } else if skipped.is_empty() {
        format!(
            "rule applied at every preregistered prefix length {evaluated:?}; promote Channel B \
             = {promote_channel_b} (satisfied by {satisfied_by:?})"
        )
    } else {
        format!(
            "rule applied at prefix lengths {evaluated:?} only; {skipped:?} had no scorable \
             changed transition (no episode long enough or no changed row) and could not be \
             evaluated; promote Channel B = {promote_channel_b} (satisfied by {satisfied_by:?})"
        )
    };
    AdaptationFalsifierVerdict {
        promotion_delta,
        evaluated_prefix_lengths: evaluated,
        skipped_prefix_lengths: skipped,
        arms: arm_verdicts,
        promote_channel_b,
        satisfied_by,
        note,
    }
}

/// ADR 0005 §5.2 (E3): prequential Channel A vs Channel A+B on held-out
/// synthetic Learning Histories. Arms: `context_only` (frozen prior weights),
/// `reset` (Channel B, fast weights reset to theta_0 at each level boundary)
/// and `carry` (fast weights persist across levels). Every arm scores the
/// same rows with the same Context Window; only the weights differ. The
/// model's fast weights are back at theta_0 bitwise when this returns.
pub fn evaluate_adaptation_falsifier(
    model: &WorldModel,
    varmap: &VarMap,
    device: &Device,
    spec: &AdaptationFalsifierSpec,
) -> Result<AdaptationFalsifierReport> {
    if !model.config().world_core_v6 {
        bail!("the adaptation falsifier requires a world_core_v6 checkpoint");
    }
    if spec.episodes == 0 {
        bail!("the adaptation falsifier needs at least one episode");
    }
    if spec.score_window == 0 || spec.prefix_lengths.is_empty() {
        bail!("the adaptation falsifier needs a score window and prefix lengths");
    }
    let histories = timed_eval_phase("adaptation_falsifier", "population_generation", || {
        adaptation_falsifier_population(spec)
    })?;
    evaluate_adaptation_falsifier_on(model, varmap, device, spec, &histories)
}

/// [`evaluate_adaptation_falsifier`] over a caller-supplied population (the
/// production path generates it from `spec`; tests shorten episodes).
fn evaluate_adaptation_falsifier_on(
    model: &WorldModel,
    varmap: &VarMap,
    device: &Device,
    spec: &AdaptationFalsifierSpec,
    histories: &[AugmentedLearningHistory],
) -> Result<AdaptationFalsifierReport> {
    if histories.is_empty() {
        bail!("the adaptation falsifier needs at least one episode");
    }
    let population_fingerprint = adaptation_falsifier_population_fingerprint(histories);
    let mut chronological_lengths = BTreeMap::<String, usize>::new();
    for history in histories {
        *chronological_lengths
            .entry(history.chronological.len().to_string())
            .or_default() += 1;
    }
    let push_arm = |arms: &mut Vec<AdaptationFalsifierArmMetrics>,
                    name: &str,
                    accums: &BTreeMap<usize, AdaptationFalsifierAccum>| {
        for (t, accum) in accums {
            arms.push(accum.metrics(name, *t));
        }
    };
    let mut arms = Vec::new();
    let context_only = timed_eval_phase("adaptation_falsifier", "arm=context_only", || {
        adaptation_falsifier_arm(model, device, histories, spec, None)
    })?;
    push_arm(
        &mut arms,
        ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY,
        &context_only,
    );
    // Preregistered E3 arms: the row-context scope follows the arm (reset =
    // level, carry = game), as the falsifier was registered; the live
    // `--context-scope` knob is deliberately not exposed here.
    for (name, mode, scope) in [
        (
            ADAPTATION_FALSIFIER_ARM_RESET,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        ),
        (
            ADAPTATION_FALSIFIER_ARM_CARRY,
            AdaptationMode::Carry,
            ContextScopeKind::Game,
        ),
    ] {
        let mut adapter = FastWeightAdapter::new(model, varmap, device, mode, scope)?;
        adapter.set_min_level_transitions(spec.min_level_transitions);
        let accums = timed_eval_phase("adaptation_falsifier", &format!("arm={name}"), || {
            adaptation_falsifier_arm(model, device, histories, spec, Some(&mut adapter))
        })?;
        if !adapter.fast_weights_equal_prior()? {
            bail!("fast weights differ from theta_0 after the {name} arm");
        }
        drop(adapter);
        push_arm(&mut arms, name, &accums);
    }
    let verdict = adaptation_falsifier_verdict(&arms, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
    Ok(AdaptationFalsifierReport {
        population: "learning_histories/unseen_seed_7x7".into(),
        population_seed: spec.population_seed,
        adapter_seed: spec.adapter_seed,
        episodes: histories.len(),
        population_fingerprint,
        evidence_class: "selection_only".into(),
        prefix_lengths: spec.prefix_lengths.clone(),
        score_window: spec.score_window,
        context_window_max: CONTEXT_WINDOW_MAX,
        min_level_transitions: spec.min_level_transitions,
        chronological_lengths,
        arms,
        verdict,
    })
}

fn eval_profile_phase<T>(
    profile: Option<&RepresentativeUpdateCapture>,
    device: &Device,
    name: &str,
    kind: SpanKind,
    step: Option<ExecutionStep>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match profile {
        Some(profile) => profile.synchronized_phase(device, name, kind, step, f),
        None => f(),
    }
}

/// Foundation-v2 gate evaluation with the exact V5 content-mask sidecar.
/// Legacy callers may use [`evaluate_gate_support`], whose origin-aligned
/// fallback is valid for the non-translated curriculum rows it accepts.
pub fn evaluate_gate_support_with_content_masks(
    model: &WorldModel,
    samples: &[TransitionSample],
    content_masks: Option<&[ContentMask]>,
    device: &Device,
) -> Result<GateSupportMetrics> {
    evaluate_gate_support_impl(model, samples, content_masks, None, device, None, None)
}

/// Foundation-v2 gate evaluation with the exact V5 operator provenance needed
/// to replay shuffled ACTION5/ACTION6 tuples on each target current board.
pub fn evaluate_gate_support_with_v5_provenance(
    model: &WorldModel,
    samples: &[TransitionSample],
    content_masks: &[ContentMask],
    provenance: &[V5SampleProvenance],
    device: &Device,
) -> Result<GateSupportMetrics> {
    evaluate_gate_support_impl(
        model,
        samples,
        Some(content_masks),
        Some(provenance),
        device,
        None,
        None,
    )
}

fn evaluate_gate_support_impl(
    model: &WorldModel,
    samples: &[TransitionSample],
    content_masks: Option<&[ContentMask]>,
    provenance: Option<&[V5SampleProvenance]>,
    device: &Device,
    encoded: Option<&EncodedGateSupportPopulation>,
    profile: Option<&RepresentativeUpdateCapture>,
) -> Result<GateSupportMetrics> {
    if content_masks.is_some_and(|masks| masks.len() != samples.len()) {
        bail!("gate content-mask rows do not match the sample count");
    }
    if provenance.is_some_and(|provenance| provenance.len() != samples.len()) {
        bail!("gate V5 provenance rows do not match the sample count");
    }
    let population_bytes = match provenance {
        Some(provenance) => serde_json::to_vec(&(samples, provenance))?,
        None => serde_json::to_vec(samples)?,
    };
    let population_fingerprint = format!("sha256:{:x}", Sha256::digest(population_bytes));
    let content_mask_fingerprint = content_masks
        .map(|masks| {
            serde_json::to_vec(masks).map(|bytes| format!("sha256:{:x}", Sha256::digest(bytes)))
        })
        .transpose()?;
    if samples.is_empty() {
        return Ok(GateSupportMetrics {
            samples: 0,
            population_fingerprint,
            content_mask_fingerprint,
            evidence_class: "selection_only".into(),
            changed_transitions: 0,
            changed_pixels: 0,
            foreground_pixels: 0,
            improvement_fraction: None,
            shuffled_action_changed_pixel_ratio: None,
            shuffled_action_rows: 0,
            shuffled_action_eligible_rows: 0,
            shuffled_action_changed_tuples: 0,
            shuffled_action_outcome_changing_tuples: provenance.map(|_| 0),
            foreground_reconstruction_accuracy: None,
            one_step_changed_exact: None,
            one_step_full_exact: None,
            one_step_raw_full_exact: None,
            one_step_composed_changed_exact: None,
            one_step_all_rows_exact: None,
            false_edit_rate: None,
            padding_false_edit_rate: None,
            raw_false_edit_rate: None,
            raw_padding_false_edit_rate: None,
            population_contract: "caller-owned fixed held-out transition set; empty population"
                .into(),
        });
    }
    if !model.config().world_core_v4 {
        bail!("foundation-v2 gate support requires the exact gameplay decoder");
    }
    let measurement = profile.and_then(RepresentativeUpdateCapture::measurement);
    let owned_encoded;
    let encoded = if let Some(encoded) = encoded {
        encoded
    } else {
        owned_encoded =
            eval_profile_phase(profile, device, "encode", SpanKind::Module, None, || {
                encode_gate_support_population(model, samples, provenance, device)
            })?;
        &owned_encoded
    };
    let shuffled_action_eligible_rows = encoded.shuffled.eligible_rows;
    let shuffled_action_changed_tuples = encoded.shuffled.changed_tuples(samples);
    let shuffled_action_outcome_changing = encoded.shuffled.outcome_changing(samples);
    let shuffled_action_outcome_changing_tuples = encoded.shuffled.outcome_changing_tuples(samples);
    let (prediction, shuffled_prediction) = eval_profile_phase(
        profile,
        device,
        "forward",
        SpanKind::Module,
        Some(ExecutionStep::Forward),
        || {
            // ADR 0005 §6.1: v6 rows are scored with their context window
            // (None for legacy rows, so v5 scoring is unchanged).
            let depth = RecursionDepth::from_config(model.config());
            let chunk = FOUNDATION_V2_GATE_PHYSICAL_BATCH;
            let prediction = forward_gate_rows_chunked(
                model,
                &encoded.current,
                &encoded.batch.actions,
                &encoded.batch.action_coords,
                &encoded.batch.goals,
                &encoded.batch.operator_conditioning,
                encoded.context.as_ref(),
                depth,
                chunk,
            )?;
            let shuffled_prediction = forward_gate_rows_chunked(
                model,
                &encoded.current,
                &encoded.shuffled_actions,
                &encoded.shuffled_coords,
                &encoded.batch.goals,
                &encoded.batch.operator_conditioning,
                encoded.context.as_ref(),
                depth,
                chunk,
            )?;
            Ok((prediction, shuffled_prediction))
        },
    )?;
    let (
        learned_errors,
        copy_errors,
        true_predictions,
        composed_predictions,
        shuffled_predictions,
        target_reconstructions,
    ) = eval_profile_phase(profile, device, "decode", SpanKind::Module, None, || {
        let learned_errors = per_sample_mse(&prediction, &encoded.target)?;
        let copy_errors = per_sample_mse(&encoded.current, &encoded.target)?;
        let true_predictions = exact_palette_predictions(model, &prediction)?;
        let composed_predictions = model
            .composed_gameplay_decode(&prediction, &encoded.batch.frames)?
            .reshape((samples.len(), ()))?
            .to_dtype(DType::U8)?
            .to_vec2::<u8>()?;
        let shuffled_predictions = exact_palette_predictions(model, &shuffled_prediction)?;
        let target_reconstructions = exact_palette_predictions(model, &encoded.target)?;
        Ok((
            learned_errors,
            copy_errors,
            true_predictions,
            composed_predictions,
            shuffled_predictions,
            target_reconstructions,
        ))
    })?;
    let metrics_range =
        profile.and_then(|profile| profile.phase("metrics", SpanKind::Function, None));
    let changed_indices = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| is_board_changed_transition(sample).then_some(index))
        .collect::<Vec<_>>();
    let changed_learned = changed_indices
        .iter()
        .map(|index| learned_errors[*index])
        .collect::<Vec<_>>();
    let changed_copy = changed_indices
        .iter()
        .map(|index| copy_errors[*index])
        .collect::<Vec<_>>();
    let gameplay_len = gameplay_rows(model.config().world_core_v6) * FRAME_SIDE;
    let changed_pixels = samples
        .iter()
        .filter(|sample| is_board_changed_transition(sample))
        .map(|sample| {
            let (current, target) = gameplay_pixels(sample, gameplay_len);
            current.iter().zip(target).filter(|(a, b)| a != b).count()
        })
        .sum();
    let foreground_pixels = samples
        .iter()
        .map(|sample| {
            gameplay_pixels(sample, gameplay_len)
                .1
                .iter()
                .filter(|pixel| **pixel != sample.provenance.background_color)
                .count()
        })
        .sum();
    let metrics = GateSupportMetrics {
        samples: samples.len(),
        population_fingerprint,
        content_mask_fingerprint,
        evidence_class: "selection_only".into(),
        changed_transitions: changed_indices.len(),
        changed_pixels,
        foreground_pixels,
        improvement_fraction: improvement_fraction(&changed_learned, &changed_copy),
        shuffled_action_changed_pixel_ratio: shuffled_action_changed_pixel_ratio(
            samples,
            &encoded.shuffled.samples,
            &shuffled_action_outcome_changing,
            &true_predictions,
            &shuffled_predictions,
        )?,
        shuffled_action_rows: samples.len(),
        shuffled_action_eligible_rows,
        shuffled_action_changed_tuples,
        shuffled_action_outcome_changing_tuples,
        foreground_reconstruction_accuracy: foreground_reconstruction_accuracy(
            samples,
            &target_reconstructions,
        )?,
        one_step_changed_exact: one_step_changed_exact(samples, &true_predictions)?,
        one_step_full_exact: one_step_full_exact(samples, &composed_predictions)?,
        one_step_raw_full_exact: one_step_full_exact(samples, &true_predictions)?,
        one_step_composed_changed_exact: one_step_changed_exact(samples, &composed_predictions)?,
        one_step_all_rows_exact: one_step_all_rows_exact(samples, &composed_predictions)?,
        false_edit_rate: one_step_false_edit_rate_with_content_masks(
            samples,
            &composed_predictions,
            content_masks,
            false,
        )?,
        padding_false_edit_rate: one_step_false_edit_rate_with_content_masks(
            samples,
            &composed_predictions,
            content_masks,
            true,
        )?,
        raw_false_edit_rate: one_step_false_edit_rate_with_content_masks(
            samples,
            &true_predictions,
            content_masks,
            false,
        )?,
        raw_padding_false_edit_rate: one_step_false_edit_rate_with_content_masks(
            samples,
            &true_predictions,
            content_masks,
            true,
        )?,
        population_contract: if provenance.is_some() {
            "caller-owned fixed selection-only transition set; board-changed rows are exactly noop==Some(false); status row 63 excluded; full exactness and primary false-edit rates use the composed copy-gate decode; raw counterparts are diagnostic; content false edits use exact V5 masks; ACTION5/ACTION6 tuples use the maximum-change cyclic shuffle within provenance.source_kind, with ACTION6 rectangle-relative coordinates conjugated onto each target content rectangle; every shuffled tuple is replayed under the target row's recorded episode operator; total/eligible/genuinely changed/outcome-changing counts are explicit; the shuffled-action ratio includes only genuinely changed tuples whose counterfactual gameplay outcome differs from the factual next board; one encode batch plus true/shuffled forwards".into()
        } else {
            "caller-owned fixed selection-only transition set; board-changed rows are exactly noop==Some(false); status row 63 excluded; full exactness and primary false-edit rates use the composed copy-gate decode; raw counterparts are diagnostic; content false edits use provenance-origin rectangle reconstruction; action tuples use the maximum-change cyclic shuffle within provenance.source_kind, with ACTION6 rectangle-relative coordinates conjugated onto each target content rectangle; total/eligible/genuinely changed counts are explicit, while outcome-changing count is unavailable without V5 operator provenance and rows retain the historical tuple-difference ratio behavior; one encode batch plus true/shuffled forwards".into()
        },
    };
    if let Some(profile) = profile {
        let span_id = metrics_range.as_ref().and_then(|range| range.span_id());
        let scalar = |label: &str, value: Option<f64>| -> Result<()> {
            if let Some(value) = value {
                profile.record_scalar(span_id, label, value)?;
            }
            Ok(())
        };
        scalar("eval/changed_exact", metrics.one_step_changed_exact)?;
        scalar("eval/full_exact_composed", metrics.one_step_full_exact)?;
        scalar("eval/full_exact_raw", metrics.one_step_raw_full_exact)?;
        scalar(
            "eval/changed_exact_composed",
            metrics.one_step_composed_changed_exact,
        )?;
        scalar(
            "eval/all_rows_exact_composed",
            metrics.one_step_all_rows_exact,
        )?;
        scalar("eval/false_edit_rate", metrics.false_edit_rate)?;
        scalar(
            "eval/padding_false_edit_rate",
            metrics.padding_false_edit_rate,
        )?;
        scalar("eval/raw_false_edit_rate", metrics.raw_false_edit_rate)?;
        scalar(
            "eval/raw_padding_false_edit_rate",
            metrics.raw_padding_false_edit_rate,
        )?;
        scalar(
            "eval/shuffled_action_changed_pixel_ratio",
            metrics.shuffled_action_changed_pixel_ratio,
        )?;
        scalar(
            "eval/foreground_reconstruction_accuracy",
            metrics.foreground_reconstruction_accuracy,
        )?;
        profile.record_scalar(
            span_id,
            "eval/foreground_pixels",
            metrics.foreground_pixels as f64,
        )?;
    }
    drop(metrics_range);
    drop(measurement);
    Ok(metrics)
}

fn ablation_metrics(metrics: &GateSupportMetrics) -> AblationMetrics {
    AblationMetrics {
        one_step_changed_exact: metrics.one_step_changed_exact,
        one_step_all_rows_exact: metrics.one_step_all_rows_exact,
        false_edit_rate: metrics.false_edit_rate,
        padding_false_edit_rate: metrics.padding_false_edit_rate,
        improvement_fraction: metrics.improvement_fraction,
        shuffled_action_changed_pixel_ratio: metrics.shuffled_action_changed_pixel_ratio,
    }
}

fn decode_composition_description(composition: DecodeComposition) -> String {
    match composition {
        DecodeComposition::LegacyHardGate => "decode_composition=legacy_hard_gate".into(),
        DecodeComposition::JointCopyMixture => "decode_composition=joint_copy_mixture".into(),
    }
}

fn action_impulse_description(enabled: bool) -> String {
    format!(
        "action_impulse={}",
        if enabled {
            "grid_scaled"
        } else {
            "legacy_field"
        }
    )
}

fn deep_copy_varmap(source: &VarMap) -> Result<VarMap> {
    let copied = {
        let data = source.data().lock().unwrap();
        data.iter()
            .map(|(name, var)| {
                Ok((
                    name.clone(),
                    Var::from_tensor(&var.as_tensor().copy()?.detach())?,
                ))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let target = VarMap::new();
    target.data().lock().unwrap().extend(copied);
    Ok(target)
}

fn set_copy_bypass_alpha(varmap: &VarMap, alpha: f64) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut matched = 0usize;
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.ends_with("y_copy_bypass_alpha"))
    {
        var.set(&Tensor::full(
            alpha as f32,
            var.shape().dims(),
            var.device(),
        )?)
        .map_err(|error| anyhow::anyhow!("set {name} to {alpha}: {error}"))?;
        matched += 1;
    }
    if matched != 1 {
        bail!("expected one copy-bypass alpha tensor, found {matched}");
    }
    Ok(())
}

/// Evaluate parameter-free and scalar treatment ablations on identical rows
/// without mutating the checkpoint VarMap used by the main evaluation.
pub fn evaluate_mechanism_ablations(
    model: &WorldModel,
    varmap: &VarMap,
    samples: &[TransitionSample],
    content_masks: Option<&[ContentMask]>,
    provenance: Option<&[V5SampleProvenance]>,
    device: &Device,
) -> Result<MechanismAblationReport> {
    let encoded = (!samples.is_empty())
        .then(|| encode_gate_support_population(model, samples, provenance, device))
        .transpose()?;
    let evaluate_variant = |candidate: &WorldModel| {
        evaluate_gate_support_impl(
            candidate,
            samples,
            content_masks,
            provenance,
            device,
            encoded.as_ref(),
            None,
        )
    };
    let baseline_gate = evaluate_variant(model)?;
    let baseline = ablation_metrics(&baseline_gate);
    let baseline_cfg = model.config().clone();

    let mut decode_cfg = baseline_cfg.clone();
    decode_cfg.decode_composition = match baseline_cfg.decode_composition {
        DecodeComposition::LegacyHardGate => DecodeComposition::JointCopyMixture,
        DecodeComposition::JointCopyMixture => DecodeComposition::LegacyHardGate,
    };
    let decode_model = WorldModel::new(
        decode_cfg.clone(),
        VarBuilder::from_varmap(varmap, DType::F32, device),
    )?;
    let decode_variant = ablation_metrics(&evaluate_variant(&decode_model)?);

    let mut impulse_cfg = baseline_cfg.clone();
    impulse_cfg.grid_scaled_action_impulse = !baseline_cfg.grid_scaled_action_impulse;
    let impulse_model = WorldModel::new(
        impulse_cfg.clone(),
        VarBuilder::from_varmap(varmap, DType::F32, device),
    )?;
    let impulse_variant = ablation_metrics(&evaluate_variant(&impulse_model)?);

    let copy_bypass_alpha_sweep = model
        .copy_bypass_alpha()?
        .map(|trained_alpha| -> Result<Vec<AlphaSweepPoint>> {
            let sweep_varmap = deep_copy_varmap(varmap)?;
            let sweep_model = WorldModel::new(
                baseline_cfg.clone(),
                VarBuilder::from_varmap(&sweep_varmap, DType::F32, device),
            )?;
            let mut points = vec![AlphaSweepPoint {
                alpha: trained_alpha,
                metrics: baseline.clone(),
            }];
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
                set_copy_bypass_alpha(&sweep_varmap, alpha)?;
                points.push(AlphaSweepPoint {
                    alpha,
                    metrics: ablation_metrics(&evaluate_variant(&sweep_model)?),
                });
            }
            Ok(points)
        })
        .transpose()?;

    Ok(MechanismAblationReport {
        rows: samples.len(),
        population_fingerprint: baseline_gate.population_fingerprint,
        decode_composition: Some(AblationPair {
            baseline_description: decode_composition_description(baseline_cfg.decode_composition),
            baseline: baseline.clone(),
            variant_description: decode_composition_description(decode_cfg.decode_composition),
            variant: decode_variant,
        }),
        action_impulse: Some(AblationPair {
            baseline_description: action_impulse_description(
                baseline_cfg.grid_scaled_action_impulse,
            ),
            baseline,
            variant_description: action_impulse_description(impulse_cfg.grid_scaled_action_impulse),
            variant: impulse_variant,
        }),
        copy_bypass_alpha_sweep,
        evidence_class: "selection_only".into(),
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
    // Group-disjoint split: an episode/trajectory must never straddle the
    // bridge fit and its validation, and pair metrics must only score
    // validation groups — otherwise memorized training deltas inflate the
    // reported held-out alignment.
    fn identifiability_group_hash(sample: &TransitionSample) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut push = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        push(&sample.seed.to_le_bytes());
        push(&sample.episode_id.to_le_bytes());
        push(sample.provenance.trajectory_id.as_bytes());
        hash
    }
    let mut labeled_h = Vec::new();
    let mut labeled_z = Vec::new();
    let mut labeled_val = Vec::new();
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
        labeled_val.push(identifiability_group_hash(sample) % 5 == 0);
    }
    if labeled_h.is_empty() {
        return None;
    }
    // Degenerate assignments (every group on one side) leave no honest
    // held-out population; fit on everything and report no validation R².
    let split_usable = labeled_val.iter().any(|&v| v) && labeled_val.iter().any(|&v| !v);
    let mut train_h = Vec::new();
    let mut train_z = Vec::new();
    let mut val_h = Vec::new();
    let mut val_z = Vec::new();
    for ((h, z), &validation) in labeled_h.iter().zip(&labeled_z).zip(&labeled_val) {
        if split_usable && validation {
            val_h.push(h.clone());
            val_z.push(z.clone());
        } else {
            train_h.push(h.clone());
            train_z.push(z.clone());
        }
    }
    let (train_h, val_h) = (train_h.as_slice(), val_h.as_slice());
    let (train_z, val_z) = (train_z.as_slice(), val_z.as_slice());
    let bridge = fit_ridge_h_to_z(train_h, train_z);
    let (r2_h_to_z, r2_h_to_z_train, bridge_w, h_dim, z_dim) = match bridge {
        Some((w, hd, zd)) => (
            split_usable
                .then(|| linear_r2_with_weights(val_h, val_z, &w, hd, zd))
                .flatten(),
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
        // Pair metrics score only validation groups: training-pair alignment
        // must not raise the reported held-out increment cosine.
        if split_usable && identifiability_group_hash(sample) % 5 != 0 {
            prev = None;
            continue;
        }
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
    let labels = mse
        .iter()
        .map(|m| if (*m as f64) < threshold { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    eval_q_labels(q_logit, &labels)
}

fn eval_q_labels(q_logit: &Tensor, labels: &[f32]) -> Result<QEvalAccum> {
    let probs = ops::sigmoid(q_logit)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if probs.len() != labels.len() {
        bail!("Q batch {} != label batch {}", probs.len(), labels.len());
    }
    let mut out = QEvalAccum {
        n: probs.len(),
        ..Default::default()
    };
    for (p, y) in probs.iter().zip(labels.iter().copied()) {
        let p = *p as f64;
        let y = y as f64;
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

/// Exact content mask reconstructed from provenance: dimensions plus the
/// recorded placement origin (zero for legacy origin-aligned populations,
/// the sampled origin for translated V5 rows).
fn provenance_content_mask(sample: &TransitionSample) -> Result<ContentMask> {
    let rect = ContentRect {
        x: u8::try_from(sample.provenance.content_x).context("content x does not fit u8")?,
        y: u8::try_from(sample.provenance.content_y).context("content y does not fit u8")?,
        width: u8::try_from(sample.provenance.content_width)
            .context("content width does not fit u8")?,
        height: u8::try_from(sample.provenance.content_height)
            .context("content height does not fit u8")?,
    };
    ContentMask::from_rect(rect)
}

fn origin_content_tensor(
    samples: &[TransitionSample],
    whole_frame: bool,
    device: &Device,
) -> Result<Tensor> {
    let rows = gameplay_rows(whole_frame);
    let mut values = Vec::with_capacity(samples.len() * rows * FRAME_SIDE);
    for sample in samples {
        // ADR 0005 §1.1: every pixel of a v6 frame is content.
        let mask = if whole_frame {
            ContentMask::all_ones()
        } else {
            provenance_content_mask(sample)?
        };
        values.extend(
            mask.values[..rows * FRAME_SIDE]
                .iter()
                .map(|&value| f32::from(value)),
        );
    }
    Tensor::from_vec(values, (samples.len(), rows, FRAME_SIDE), device).map_err(Into::into)
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
    let ptrm = model.forward_ptrm_with_operator_conditioning(
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        &batch.operator_conditioning,
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
    /// Foundation-v2 only: content-masked canonical current/next rows in
    /// sample order, for one whole-population EP statistic computed after all
    /// chunks (per-chunk EP is nonlinear and physical-batch-dependent).
    foundation_ep_current: Vec<Vec<f32>>,
    foundation_ep_next: Vec<Vec<f32>>,
    changed_learned_errors: Vec<f32>,
    changed_copy_forward_errors: Vec<f32>,
    event_labeled: usize,
    event_correct_weighted: f64,
    event_bce_weighted: f64,
    hazard_failure_labeled: usize,
    hazard_false_negatives: usize,
    q_acc: QEvalAccum,
    q_probs: Vec<f32>,
    /// Labels for the recipe-specific Q objective (pixel accuracy for current
    /// Foundation-v2 and Full-v4 checkpoints).
    q_labels: Vec<bool>,
    /// Thresholded full-spatial prediction/encoded-target latent MSE used by
    /// Q-surprise and Foundation-v2 reliability calibration.
    latent_reliability_labels: Vec<bool>,
    reliability_probs: Vec<f32>,
    /// Labels matching the active recipe's reliability training target.
    reliability_labels: Vec<bool>,
    recursion_probes: Vec<RecursionStepProbe>,
    ptrm_acc: BTreeMap<usize, (f64, f64, f64, f64, usize)>,
    matched_acc: BTreeMap<usize, (f64, usize, usize)>,
    ensemble_disagreement: f64,
    ensemble_n: usize,
    ptrm_forward_elapsed: Duration,
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
            model.forward_with_operator_conditioning(
                &batch.frames,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
            )
        })
        .transpose()?;
    let diagnostic = model.representation_diagnostic_with_operator_conditioning(
        &batch.frames,
        &batch.next_frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        &batch.operator_conditioning,
    )?;
    let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
    let current_z = encoded.current;
    let next_z = encoded.next;
    let foundation_recipe = train_cfg.recipe == crate::p2::experiment::TrainingRecipe::FoundationV2;
    // Foundation-v2 SIGReg is not scored per chunk: training regularizes the
    // content-masked canonical population, and EP is nonlinear in the batch,
    // so chunk-wise statistics change with physical batching. Collect the
    // exact training-population rows instead and evaluate once at the end.
    let sigreg = (chunk.len() >= 2 && !foundation_recipe)
        .then(|| {
            model_sigreg_losses_for_encoded_pair(
                model,
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
    let foundation_ep = foundation_recipe
        .then(|| -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
            let (_, _, latent_height, latent_width) = current_z.dims4()?;
            let masks = chunk
                .iter()
                .map(provenance_content_mask)
                .collect::<Result<Vec<_>>>()?;
            let latent_mask =
                latent_content_mask(masks.iter(), latent_height, latent_width, device)?;
            let current = model
                .canonical_representation(&current_z.broadcast_mul(&latent_mask)?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let next = model
                .canonical_representation(&next_z.broadcast_mul(&latent_mask)?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            Ok((current, next))
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
        foundation_ep_current: foundation_ep
            .as_ref()
            .map(|(current, _)| current.clone())
            .unwrap_or_default(),
        foundation_ep_next: foundation_ep.map(|(_, next)| next).unwrap_or_default(),
        ..Default::default()
    };
    partial.mse_all.extend(mses.iter().copied());
    for ((sample, learned), copy_forward) in chunk
        .iter()
        .zip(mses.iter().copied())
        .zip(copy_forward_mses.iter().copied())
    {
        if is_board_changed_transition(sample) {
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

        let observer_q_labels = match train_cfg.recipe {
            crate::p2::experiment::TrainingRecipe::FullV4 => Some(
                model
                    .exact_transition_correctness(&out.y, &batch.frames, &batch.next_frames)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            ),
            crate::p2::experiment::TrainingRecipe::FoundationV2 => Some(
                foundation_v2_graded_q_targets(
                    model,
                    &out.y,
                    &batch.frames,
                    &batch.next_frames,
                    &origin_content_tensor(chunk, model.config().world_core_v6, device)?,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?,
            ),
            _ => None,
        };
        partial.q_acc = if let Some(labels) = &observer_q_labels {
            eval_q_labels(&out.q_logit, labels)?
        } else {
            eval_q(&out.q_logit, &mses, cfg.q_mse_threshold)?
        };
        partial.q_labels = observer_q_labels.map_or_else(
            || {
                mses.iter()
                    .map(|m| f64::from(*m) < cfg.q_mse_threshold)
                    .collect()
            },
            |labels| labels.into_iter().map(|label| label >= 0.5).collect(),
        );
        partial.q_probs = candle_nn::ops::sigmoid(&out.q_logit.to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        partial.latent_reliability_labels = mses
            .iter()
            .map(|mse| f64::from(*mse) <= cfg.q_mse_threshold)
            .collect();
        // Foundation-v2 now trains reliability on latent confidence. Preserve
        // the exact-pixel Full-v4 calibration contract outside this recipe.
        partial.reliability_labels = if foundation_recipe {
            partial.latent_reliability_labels.clone()
        } else {
            partial.q_labels.clone()
        };
        partial.reliability_probs =
            candle_nn::ops::sigmoid(&out.reliability_logit.to_dtype(DType::F32)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
        partial
            .recursion_probes
            .extend(out.recursion_probes.clone());

        let ptrm_forward_started = Instant::now();
        if cfg.ensemble_members >= 2 {
            let ptrm = model.forward_ptrm_with_operator_conditioning(
                &batch.frames,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
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

        // Full V4 deliberately excludes PTRM training. Its Q target is exact
        // gameplay-pixel correctness, whereas this historical diagnostic uses
        // latent-MSE oracles. Do not emit incomparable selection metrics.
        if train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FullV4 {
            for &k in &cfg.ptrm_k {
                let outer_steps = model
                    .config()
                    .outer_steps
                    .checked_mul(k)
                    .ok_or_else(|| anyhow::anyhow!("matched-compute outer_steps overflow"))?;
                let deterministic = model.forward_with_outer_steps_and_operator_conditioning(
                    &batch.frames,
                    &batch.actions,
                    &batch.action_coords,
                    &batch.goals,
                    &batch.operator_conditioning,
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
        partial.ptrm_forward_elapsed = ptrm_forward_started.elapsed();
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
        .foundation_ep_current
        .extend(partial.foundation_ep_current);
    merged.foundation_ep_next.extend(partial.foundation_ep_next);
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
    merged.q_labels.extend(partial.q_labels);
    merged
        .latent_reliability_labels
        .extend(partial.latent_reliability_labels);
    merged.reliability_probs.extend(partial.reliability_probs);
    merged.reliability_labels.extend(partial.reliability_labels);
    merged.recursion_probes.extend(partial.recursion_probes);
    merged.ensemble_disagreement += partial.ensemble_disagreement;
    merged.ensemble_n += partial.ensemble_n;
    merged.ptrm_forward_elapsed += partial.ptrm_forward_elapsed;
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
    let out = model.forward_with_operator_conditioning(
        &batch.frames,
        &shuffled_actions,
        &shuffled_action_coords,
        &batch.goals,
        &batch.operator_conditioning,
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
        intervention_contract: {
            let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
            let mut push = |bytes: &[u8]| {
                for byte in bytes {
                    fingerprint ^= u64::from(*byte);
                    fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
                }
            };
            for ablated in shuffled {
                push(&[
                    ablated.action.id,
                    ablated.action.x.unwrap_or(u8::MAX),
                    ablated.action.y.unwrap_or(u8::MAX),
                ]);
            }
            format!(
                "per-source seeded Sattolo single-cycle permutation of complete action \
                 tuples; changed_tuples={}/{}; donor_fingerprint=fnv1a64:{fingerprint:016x}; \
                 distinct from the semantic evaluator's maximum-change cyclic rotation \
                 (action_shuffled_prediction) - do not read the two sections as one control",
                changed_conditioning_indices.len(),
                samples.len(),
            )
        },
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
    for (idx, sample) in steps.iter().take(8).enumerate() {
        let batch =
            batch_from_samples(std::slice::from_ref(sample), device).with_context(|| {
                rollout_operation_context(sample, "open-loop", "step batch construction")
            })?;
        let open_pred = model
            .forward_from_latent_with_operator_conditioning(
                &open_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
            )
            .with_context(|| rollout_operation_context(sample, "open-loop", "latent forward"))?
            .y;
        open_latent = open_pred.clone();
        let closed_latent = model
            .encode_state(&batch.frames)
            .with_context(|| rollout_operation_context(sample, "closed-loop", "state encoding"))?;
        let closed_pred = model
            .forward_from_latent_with_operator_conditioning(
                &closed_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
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
        if matches!(horizon, 4 | 8) {
            let mut row =
                episode_rollout_result(steps, horizon, open_mse, closed_mse, copy_forward_mse);
            if model.config().world_core_v4 {
                row.open_semantic = Some(latent_semantic_metrics(model, &open_pred, sample)?);
                row.closed_semantic = Some(latent_semantic_metrics(model, &closed_pred, sample)?);
                row.learned_copy_semantic = Some(latent_semantic_metrics(model, &z0, sample)?);
            }
            rows.push(row);
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

fn aggregate_rollout_semantics(
    rows: &[EpisodeRolloutRow],
    read: fn(&EpisodeRolloutRow) -> Option<&SemanticDecoderMetrics>,
) -> BTreeMap<usize, SemanticDecoderMetrics> {
    [4usize, 8]
        .into_iter()
        .filter_map(|horizon| {
            let metrics = rows
                .iter()
                .filter(|row| row.horizon == horizon)
                .filter_map(read)
                .cloned()
                .collect::<Vec<_>>();
            (!metrics.is_empty()).then(|| (horizon, aggregate_decoder_metrics(metrics)))
        })
        .collect()
}

fn attach_rollout_metrics(split: &mut SplitEval, rows: &[EpisodeRolloutRow], seed: u64) {
    split.rollout = Some(rollout_metrics_from_rows(
        rows,
        RolloutMetric::Open,
        seed ^ 0x01,
    ));
    split.closed_loop = Some(rollout_metrics_from_rows(
        rows,
        RolloutMetric::Closed,
        seed ^ 0xC1,
    ));
    split.copy_forward = Some(rollout_metrics_from_rows(
        rows,
        RolloutMetric::CopyForward,
        seed ^ 0xCF,
    ));
    if let (Some(open), Some(closed)) = (split.rollout.as_mut(), split.closed_loop.as_ref()) {
        if let (Some(o8), Some(c8)) = (open.mse_8, closed.mse_8) {
            if c8 > 0.0 {
                open.open_closed_ratio_8 = Some(o8 / c8);
            }
        }
    }
    let semantic_rollout = SemanticRolloutMetrics {
        population_contract: "trajectory-filtered rollout population (contiguous provenance.trajectory_id with length>=4); separately fingerprinted and not comparable as a horizon curve to the one-step semantic population".into(),
        comparable_to_one_step: false,
        open: aggregate_rollout_semantics(rows, |row| row.open_semantic.as_ref()),
        closed: aggregate_rollout_semantics(rows, |row| row.closed_semantic.as_ref()),
        learned_copy: aggregate_rollout_semantics(rows, |row| row.learned_copy_semantic.as_ref()),
    };
    if !semantic_rollout.open.is_empty()
        || !semantic_rollout.closed.is_empty()
        || !semantic_rollout.learned_copy.is_empty()
    {
        split.semantic_rollout = Some(semantic_rollout);
    }
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
    let (h4_seed_offset, h8_seed_offset) = match metric {
        RolloutMetric::CopyForward => (0x14, 0x18),
        RolloutMetric::Open | RolloutMetric::Closed => (0x04, 0x08),
    };
    RolloutMetrics {
        n4: h4_values.len(),
        mse_4: mean(&h4_values),
        n8: h8_values.len(),
        mse_8: mean(&h8_values),
        mse_16: None,
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
        open_closed_ratio_8: None,
    }
}

fn eval_q_surprise_labels(q_probs: &[f32], reliable: &[bool]) -> QSurpriseMetrics {
    let n = q_probs.len().min(reliable.len());
    if n == 0 {
        return QSurpriseMetrics {
            n: 0,
            mean_q_when_unreliable: None,
            mean_q_when_reliable: None,
            confident_error_rate: None,
        };
    }
    let mut unreliable_q = Vec::new();
    let mut reliable_q = Vec::new();
    let mut confident_errors = 0usize;
    let mut unreliable_count = 0usize;
    for i in 0..n {
        let q = q_probs[i];
        if !reliable[i] {
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

/// Group by the explicit trajectory identity. `family` and source strata may
/// change during retarget traces and are not valid sequence keys.
fn group_rollouts(
    samples: &[TransitionSample],
) -> Result<BTreeMap<String, Vec<&TransitionSample>>> {
    let mut map: BTreeMap<String, Vec<&TransitionSample>> = BTreeMap::new();
    for s in samples {
        map.entry(s.provenance.trajectory_id.clone())
            .or_default()
            .push(s);
    }
    for (trajectory, steps) in &mut map {
        steps.sort_by_key(|s| s.transition_index);
        for pair in steps.windows(2) {
            if pair[1].transition_index != pair[0].transition_index + 1 {
                bail!("trajectory {trajectory} has a transition-index gap or duplicate");
            }
            if pair[0].next != pair[1].current {
                bail!("trajectory {trajectory} has discontinuous rendered frames");
            }
        }
    }
    Ok(map)
}

fn eval_episode_rollouts(
    train_cfg: &TrainConfig,
    checkpoint: &Path,
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    source: &str,
) -> Result<Vec<EpisodeRolloutRow>> {
    let groups: Vec<Vec<TransitionSample>> = group_rollouts(samples)?
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
    let no_samples = Vec::new();
    SplitEval {
        source: source.into(),
        n_samples,
        one_step_latent_mse: None,
        semantic: None,
        semantic_rollout: None,
        collision_census: collision_census(&no_samples, &[]),
        ambiguity_ceiling: ambiguity_ceiling(&no_samples),
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
    let sample_set_started = Instant::now();
    if samples.is_empty() {
        let split = empty_split(source, 0);
        log_eval_phase("sample_set", source, sample_set_started.elapsed());
        return Ok(split);
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

    log_eval_phase("ptrm_forwards", source, merged.ptrm_forward_elapsed);
    let metric_reduction_started = Instant::now();
    // Sub-phase timing inside the reduction window: `detail=<source>/<step>`.
    let mut sub_phase_started = Instant::now();
    let mut log_sub_phase = |step: &str| {
        log_eval_phase(
            "metric_reduction",
            &format!("{source}/{step}"),
            sub_phase_started.elapsed(),
        );
        sub_phase_started = Instant::now();
    };

    if merged.foundation_ep_current.len() != merged.foundation_ep_next.len() {
        bail!("foundation-v2 EP population current/next row counts diverged");
    }
    if merged.foundation_ep_current.len() >= 2 {
        // One EP statistic over the complete ordered content-masked canonical
        // population with one fixed projection seed: the reported value is
        // invariant to the evaluator's physical batch size and matches the
        // representation the training objective actually regularizes.
        let rows = merged.foundation_ep_current.len();
        let dim = merged.foundation_ep_current[0].len();
        let mut values = Vec::with_capacity(2 * rows * dim);
        for row in merged
            .foundation_ep_current
            .iter()
            .chain(&merged.foundation_ep_next)
        {
            if row.len() != dim {
                bail!("foundation-v2 EP population row dimensions diverged");
            }
            values.extend_from_slice(row);
        }
        let stack = Tensor::from_vec(values, (2, rows, dim), device)?;
        let raw_tensor = sigreg_loss_for_stack(&stack, train_cfg, cfg.seed)?;
        let raw = f64::from(raw_tensor.to_dtype(DType::F32)?.to_scalar::<f32>()?);
        // The bounded field keeps its smooth-cap semantics even though the
        // foundation objective consumes the raw statistic directly.
        let bounded = f64::from(
            crate::p2::train::bounded_sigreg_loss(&raw_tensor)?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
        );
        merged.sigreg_raw_weighted = raw * rows as f64;
        merged.sigreg_bounded_weighted = bounded * rows as f64;
        merged.sigreg_n = rows;
    }
    log_sub_phase("foundation_ep_sigreg");

    let representation_seams = seam_collector.summarize()?;
    let post_rms_pooled = top_level_collector.summarize()?;
    log_sub_phase("representation_summaries");
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
    log_sub_phase("changed_transitions");
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
    let ptrm = (cfg.mode == EvalMode::Full
        && train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FullV4)
        .then_some(ptrm);
    let deterministic_matched_compute: Vec<_> = matched_acc
        .into_iter()
        .map(|(k, (sum, n, outer_steps))| MatchedComputeMetrics {
            ptrm_k_equivalent: k,
            outer_steps,
            n,
            one_step_latent_mse: (n > 0).then_some(sum / n as f64),
        })
        .collect();
    let deterministic_matched_compute = (cfg.mode == EvalMode::Full
        && train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FullV4)
        .then_some(deterministic_matched_compute);

    let q_surprise = (cfg.mode == EvalMode::Full)
        .then(|| eval_q_surprise_labels(&merged.q_probs, &merged.latent_reliability_labels));
    let labels = merged.reliability_labels.clone();
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
        let high_error: Vec<bool> = merged
            .reliability_labels
            .iter()
            .take(merged.ensemble_n.min(merged.reliability_labels.len()))
            .map(|reliable| !reliable)
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
    log_sub_phase("scalar_metrics");
    let contrastive = eval_contrastive_probes(model, samples, device).ok();
    log_sub_phase("contrastive_probes[model]");
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
    log_sub_phase("action_diagnostics[model]");

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
    if with_rollout {
        log_sub_phase("rollouts[model]");
    }
    let identifiability = eval_identifiability(samples, &encoder_embeddings);
    log_sub_phase("identifiability");
    let semantic = if train_cfg.world_core_v4 {
        Some(timed_eval_phase("semantic_decode", source, || {
            evaluate_semantics_with_control(
                model,
                samples,
                action_sources.unwrap_or(&fallback_sources),
                cfg.physical_batch,
                device,
                SemanticControlConfig {
                    trained_null_action_id: (train_cfg.recipe
                        == crate::p2::experiment::TrainingRecipe::FoundationV2)
                        .then_some(0),
                },
            )
        })?)
    } else {
        None
    };
    if semantic.is_some() {
        log_sub_phase("semantic_decode[model]");
    }
    let collision_census = collision_census(samples, action_sources.unwrap_or(&fallback_sources));
    let ambiguity_ceiling = ambiguity_ceiling(samples);
    log_sub_phase("census");

    let split = SplitEval {
        source: source.into(),
        n_samples: samples.len(),
        one_step_latent_mse: mean(&mse_all),
        semantic,
        semantic_rollout: None,
        collision_census,
        ambiguity_ceiling,
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
    };
    log_eval_phase(
        "metric_reduction",
        source,
        metric_reduction_started.elapsed(),
    );
    log_eval_phase("sample_set", source, sample_set_started.elapsed());
    Ok(split)
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
    let shuffled = shuffled_action_control_samples(samples);
    let index = samples
        .iter()
        .zip(&shuffled)
        .position(|(true_sample, shuffled_sample)| true_sample.action != shuffled_sample.action)
        .context("contrastive probe needs a source-local changed action conditioning")?;
    let probe_rows = vec![samples[index].clone(), shuffled[index].clone()];
    let batch = batch_from_samples(&probe_rows, device)?;
    let z = model.encode_state(&batch.frames)?;
    let predictions = model
        .forward_from_latent_with_operator_conditioning(
            &z,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            &batch.operator_conditioning,
        )?
        .y;
    let effects = per_sample_mse(&predictions, &z)?;
    let true_prediction = predictions.narrow(0, 0, 1)?;
    let shuffled_prediction = predictions.narrow(0, 1, 1)?;
    let prediction_mse = true_prediction
        .sub(&shuffled_prediction)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()? as f64;
    Ok(ContrastiveProbeMetrics {
        source_kind: samples[index].provenance.source_kind.clone(),
        true_action_effect_mse: effects.first().copied().map(f64::from),
        action_shuffled_effect_mse: effects.get(1).copied().map(f64::from),
        true_vs_action_shuffled_prediction_mse: Some(prediction_mse),
        action_control_contract: "one complete action tuple replaced by a deterministic donor from the same provenance.source_kind; no untrained action id is used".into(),
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

fn generate_factual_eval_population(
    seed: u64,
    synthetic_episodes: usize,
) -> Result<Option<FactualBatch>> {
    let group_count = synthetic_episodes.saturating_mul(4);
    if group_count == 0 {
        return Ok(None);
    }
    let seed = factual_branch_eval_seed(seed);
    let generated = (0..group_count)
        .into_par_iter()
        .map(|episode| {
            generate_factual_branch_group(seed, episode as u64, Split::HeldOutComposition)
        })
        .collect::<Vec<_>>();
    let mut groups = Vec::with_capacity(group_count);
    for group in generated {
        groups.push(group?);
    }
    Ok(Some(FactualBatch::from_groups(groups)?))
}

fn factual_population_fingerprint(groups: &[BranchGroup], synthetic_episodes: usize) -> String {
    let mut hash = Sha256::new();
    hash.update(FACTUAL_BRANCH_EVAL_DOMAIN);
    hash.update((synthetic_episodes as u64).to_le_bytes());
    for group in groups {
        hash.update((group.branches().len() as u64).to_le_bytes());
        for branch in group.branches() {
            let transition = &branch.transition;
            hash.update(transition.seed.to_le_bytes());
            hash.update(transition.episode_id.to_le_bytes());
            hash.update(transition.family.as_bytes());
            hash.update([transition.action.id]);
            hash.update([transition.action.x.unwrap_or(u8::MAX)]);
            hash.update([transition.action.y.unwrap_or(u8::MAX)]);
            hash.update(&transition.current.pixels);
            hash.update(&transition.next.pixels);
            hash.update([u8::from(branch.board_effect.changed)]);
            for cell in &branch.board_effect.changed_cells {
                hash.update(cell.to_le_bytes());
            }
        }
    }
    format!("sha256:{:x}", hash.finalize())
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
        .filter(|row| row.recoverable && row.predicted_action_id.is_some())
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

const OUTCOME_COUNTERFACTUAL_EPSILON: f64 = f64::MIN_POSITIVE;
const OUTCOME_COUNTERFACTUAL_MATERIAL_THRESHOLD: f64 = 0.10;
const OUTCOME_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES: usize = 10_000;
const OUTCOME_COUNTERFACTUAL_TARGET_COLLAPSE_MSE: f64 = 1e-12;

fn vector_mse(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len(), "counterfactual vector dimensions");
    if left.is_empty() {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = f64::from(*left) - f64::from(*right);
            delta * delta
        })
        .sum::<f64>()
        / left.len() as f64
}

fn outcome_counterfactual_margin(
    left_prediction: &[f32],
    right_prediction: &[f32],
    left_target: &[f32],
    right_target: &[f32],
) -> (f64, f64, f64) {
    let concordant =
        vector_mse(left_prediction, left_target) + vector_mse(right_prediction, right_target);
    let crossed =
        vector_mse(left_prediction, right_target) + vector_mse(right_prediction, left_target);
    let margin = (crossed - concordant) / (crossed + concordant + OUTCOME_COUNTERFACTUAL_EPSILON);
    (concordant, crossed, margin)
}

fn append_sha256_field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn finish_sha256(hash: Sha256) -> String {
    format!("sha256:{:x}", hash.finalize())
}

fn frame_content_sha256(frame: &crate::p2::data::ArcFrame) -> String {
    let mut hash = Sha256::new();
    append_sha256_field(&mut hash, &frame.width.to_le_bytes());
    append_sha256_field(&mut hash, &frame.height.to_le_bytes());
    append_sha256_field(&mut hash, &frame.pixels);
    finish_sha256(hash)
}

fn canonical_outcome_classes(group: &BranchGroup) -> Vec<usize> {
    let branches = group.branches();
    branches
        .iter()
        .map(|branch| {
            branches
                .iter()
                .position(|candidate| branch.outcome_equivalent(candidate))
                .expect("branch is outcome-equivalent to itself")
        })
        .collect()
}

fn outcome_group_content_sha256(group: &BranchGroup, outcome_classes: &[usize]) -> String {
    let mut hash = Sha256::new();
    let first = &group.branches()[0].transition;
    append_sha256_field(&mut hash, &first.current.width.to_le_bytes());
    append_sha256_field(&mut hash, &first.current.height.to_le_bytes());
    append_sha256_field(&mut hash, &first.current.pixels);
    for (branch, outcome_class) in group.branches().iter().zip(outcome_classes) {
        let transition = &branch.transition;
        append_sha256_field(&mut hash, &transition.next.width.to_le_bytes());
        append_sha256_field(&mut hash, &transition.next.height.to_le_bytes());
        append_sha256_field(&mut hash, &transition.next.pixels);
        append_sha256_field(&mut hash, &[transition.action.id]);
        append_sha256_field(&mut hash, &[transition.action.x.unwrap_or(u8::MAX)]);
        append_sha256_field(&mut hash, &[transition.action.y.unwrap_or(u8::MAX)]);
        for goal in transition.goal_features.values {
            append_sha256_field(&mut hash, &goal.to_bits().to_le_bytes());
        }
        append_sha256_field(&mut hash, &(*outcome_class as u64).to_le_bytes());
        append_sha256_field(&mut hash, &[u8::from(branch.board_effect.changed)]);
        for cell in &branch.board_effect.changed_cells {
            append_sha256_field(&mut hash, &cell.to_le_bytes());
        }
    }
    finish_sha256(hash)
}

fn outcome_population_fingerprint(group_fingerprints: &[String]) -> String {
    let mut canonical = group_fingerprints.to_vec();
    canonical.sort();
    let mut hash = Sha256::new();
    append_sha256_field(&mut hash, b"p2.outcome_counterfactual.content.v1");
    for fingerprint in canonical {
        append_sha256_field(&mut hash, fingerprint.as_bytes());
    }
    finish_sha256(hash)
}

fn outcome_percentile(sorted: &[f64], probability: f64) -> f64 {
    let index = (probability * (sorted.len() - 1) as f64).floor() as usize;
    sorted[index]
}

fn outcome_interval_from_group_means(
    group_means: &[(usize, f64, usize)],
    seed: u64,
) -> Option<OutcomeCounterfactualInterval> {
    if group_means.is_empty() {
        return None;
    }
    let estimate =
        group_means.iter().map(|(_, value, _)| value).sum::<f64>() / group_means.len() as f64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bootstrap = Vec::with_capacity(OUTCOME_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES);
    for _ in 0..OUTCOME_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES {
        let sum = (0..group_means.len())
            .map(|_| group_means[rng.random_range(0..group_means.len())].1)
            .sum::<f64>();
        bootstrap.push(sum / group_means.len() as f64);
    }
    bootstrap.sort_by(f64::total_cmp);
    Some(OutcomeCounterfactualInterval {
        estimate,
        lower_95: outcome_percentile(&bootstrap, 0.025),
        upper_95: outcome_percentile(&bootstrap, 0.975),
        lower_98_75: outcome_percentile(&bootstrap, 0.00625),
        upper_98_75: outcome_percentile(&bootstrap, 0.99375),
        groups: group_means.len(),
        pairs: group_means.iter().map(|(_, _, pairs)| pairs).sum(),
        resamples: OUTCOME_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES,
        unit: "whole_branch_group".into(),
    })
}

fn ledger_group_means(
    ledger: &[OutcomeCounterfactualPairLedgerRow],
    include: impl Fn(&OutcomeCounterfactualPairLedgerRow) -> bool,
) -> Vec<(usize, f64, usize)> {
    let mut values = BTreeMap::<usize, Vec<f64>>::new();
    for row in ledger.iter().filter(|row| row.eligible && include(row)) {
        values
            .entry(row.group.group_index)
            .or_default()
            .push(row.margin);
    }
    values
        .into_iter()
        .map(|(group, values)| {
            let pairs = values.len();
            (group, values.iter().sum::<f64>() / pairs as f64, pairs)
        })
        .collect()
}

fn outcome_interval_reconciles(
    interval: &Option<OutcomeCounterfactualInterval>,
    group_means: &[(usize, f64, usize)],
) -> bool {
    match (interval, group_means.is_empty()) {
        (None, true) => true,
        (Some(interval), false) => {
            let estimate = group_means.iter().map(|(_, value, _)| value).sum::<f64>()
                / group_means.len() as f64;
            interval.groups == group_means.len()
                && interval.pairs == group_means.iter().map(|(_, _, pairs)| pairs).sum::<usize>()
                && (interval.estimate - estimate).abs() <= 1e-12
        }
        _ => false,
    }
}

fn equal_weight_control_mean(values: &[(usize, f64)]) -> Option<f64> {
    let mut groups = BTreeMap::<usize, Vec<f64>>::new();
    for (group, value) in values {
        groups.entry(*group).or_default().push(*value);
    }
    (!groups.is_empty()).then(|| {
        groups
            .values()
            .map(|values| values.iter().sum::<f64>() / values.len() as f64)
            .sum::<f64>()
            / groups.len() as f64
    })
}

fn pixel_displacement(branch: &crate::p2::data::FactualActionBranch) -> Vec<f32> {
    branch
        .transition
        .next
        .pixels
        .iter()
        .zip(&branch.transition.current.pixels)
        .map(|(next, current)| f32::from(*next) - f32::from(*current))
        .collect()
}

fn empty_outcome_counterfactual_metrics() -> OutcomeCounterfactualMetrics {
    let controls = OutcomeCounterfactualControls {
        pixel_oracle_estimate: None,
        pixel_oracle_exactly_one: false,
        latent_oracle_estimate: None,
        latent_oracle_at_least_0_99: false,
        target_collapse_failure: false,
        target_collapsed_pairs: 0,
        swapped_oracle_estimate: None,
        swapped_oracle_at_most_negative_0_99: false,
        action_masked_max_abs_margin: None,
        action_masked_max_abs_at_most_1e_6: false,
        identity_max_abs_margin: None,
        identity_max_abs_at_most_1e_6: false,
        outcome_equivalent_pairs: 0,
        outcome_equivalent_max_abs_margin: None,
        outcome_equivalent_max_abs_at_most_1e_6: false,
        state_scrambled_same_action_template: OutcomeCounterfactualStateScrambledControl {
            available: false,
            estimate: None,
            groups: 0,
            pairs: 0,
            reason: Some("unavailable: fewer than two eligible movement groups".into()),
        },
        required_controls_pass: false,
    };
    OutcomeCounterfactualMetrics {
        population_fingerprint: outcome_population_fingerprint(&[]),
        groups: 0,
        movement_groups: 0,
        coordinate_groups: 0,
        unordered_pairs: 0,
        eligible_pairs: 0,
        outcome_equivalent_pairs: 0,
        changed_changed_pairs: 0,
        changed_unchanged_pairs: 0,
        epsilon: OUTCOME_COUNTERFACTUAL_EPSILON,
        material_threshold: OUTCOME_COUNTERFACTUAL_MATERIAL_THRESHOLD,
        overall: None,
        movement: None,
        coordinate: None,
        changed_changed: None,
        changed_unchanged: None,
        action_separation_pass: false,
        controls,
        population_gates: OutcomeCounterfactualPopulationGates {
            eligible_simulator_groups: 0,
            eligible_simulator_groups_at_least_100: false,
            movement_action_anchors: BTreeMap::new(),
            each_movement_action_at_least_16_changed_and_16_unchanged: false,
            simulator_changed_changed_pairs: 0,
            simulator_changed_changed_pairs_at_least_100: false,
            target_collapse_failure: false,
            population_pass: false,
        },
        pair_ledger: Vec::new(),
        ledger_reconciled: true,
    }
}

fn evaluate_outcome_counterfactuals(
    model: &WorldModel,
    cfg: &EvalConfig,
    device: &Device,
    factual_population: Option<&FactualBatch>,
) -> Result<OutcomeCounterfactualMetrics> {
    let group_count = cfg.synthetic_episodes.saturating_mul(4);
    if group_count == 0 {
        return Ok(empty_outcome_counterfactual_metrics());
    }
    let factual_batch = factual_population
        .context("outcome counterfactual evaluation is missing its factual population")?;
    let seed = factual_branch_eval_seed(cfg.seed);
    let groups = factual_batch.groups();
    let samples = factual_batch.rows();
    let eval_batch = cfg.physical_batch.max(FACTUAL_BRANCHES_PER_GROUP)
        / FACTUAL_BRANCHES_PER_GROUP
        * FACTUAL_BRANCHES_PER_GROUP;
    let forward_started = Instant::now();
    let mut predicted_displacements = Vec::<Vec<f32>>::with_capacity(samples.len());
    let mut target_displacements = Vec::<Vec<f32>>::with_capacity(samples.len());
    for (start, end) in batch_ranges(samples.len(), eval_batch) {
        let batch = batch_from_samples(&samples[start..end], device)?;
        let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
        let output = model.forward_with_operator_conditioning(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            &batch.operator_conditioning,
        )?;
        let (_, channels, height, width) = encoded.current.dims4()?;
        let mut shared_currents = Vec::new();
        for local_group in (0..end - start).step_by(FACTUAL_BRANCHES_PER_GROUP) {
            shared_currents.push(encoded.current.narrow(0, local_group, 1)?.broadcast_as((
                FACTUAL_BRANCHES_PER_GROUP,
                channels,
                height,
                width,
            ))?);
        }
        let shared_refs = shared_currents.iter().collect::<Vec<_>>();
        let shared_current = Tensor::cat(&shared_refs, 0)?;
        let predicted = flatten_latent(&output.y.sub(&shared_current)?)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let target = flatten_latent(&encoded.next.sub(&shared_current)?)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        predicted_displacements.extend(predicted);
        target_displacements.extend(target);
    }
    log_eval_phase(
        "model_forwards",
        "factual_outcome_counterfactuals",
        forward_started.elapsed(),
    );
    let reduction_started = Instant::now();

    let pairs_per_group =
        FACTUAL_BRANCHES_PER_GROUP.saturating_mul(FACTUAL_BRANCHES_PER_GROUP.saturating_sub(1)) / 2;
    let mut ledger = Vec::with_capacity(groups.len() * pairs_per_group);
    let mut group_fingerprints = Vec::with_capacity(groups.len());
    let mut pixel_oracle_values = Vec::new();
    let mut latent_oracle_values = Vec::new();
    let mut swapped_oracle_values = Vec::new();
    let mut action_masked_values = Vec::new();
    let mut identity_values = Vec::new();
    let mut target_collapsed_pairs = 0usize;
    let mut movement_action_anchors = BTreeMap::<u8, OutcomeCounterfactualActionAnchors>::new();
    let mut offset = 0usize;
    for (group_index, group) in groups.iter().enumerate() {
        let branches = group.branches();
        let source = &branches[0].transition;
        let is_movement = source.episode_id.is_multiple_of(2);
        let population = if is_movement {
            "movement"
        } else {
            "coordinate"
        };
        let outcome_classes = canonical_outcome_classes(group);
        let content_fingerprint = outcome_group_content_sha256(group, &outcome_classes);
        group_fingerprints.push(content_fingerprint.clone());
        let identity = OutcomeCounterfactualGroupIdentity {
            group_index,
            family: source.family.clone(),
            population: population.into(),
            content_fingerprint,
            current_sha256: frame_content_sha256(&source.current),
            next_sha256: branches
                .iter()
                .map(|branch| frame_content_sha256(&branch.transition.next))
                .collect(),
            actions: branches
                .iter()
                .map(|branch| branch.transition.action.clone())
                .collect(),
        };
        if is_movement {
            for branch in branches {
                let anchors = movement_action_anchors
                    .entry(branch.transition.action.id)
                    .or_default();
                if branch.board_effect.changed {
                    anchors.changed += 1;
                } else {
                    anchors.unchanged += 1;
                }
            }
        }
        let pixel_displacements = branches.iter().map(pixel_displacement).collect::<Vec<_>>();
        let identity_prediction = vec![0.0f32; target_displacements[offset].len()];
        for left in 0..branches.len() {
            for right in left + 1..branches.len() {
                let global_left = offset + left;
                let global_right = offset + right;
                let (concordant_loss, crossed_loss, margin) = outcome_counterfactual_margin(
                    &predicted_displacements[global_left],
                    &predicted_displacements[global_right],
                    &target_displacements[global_left],
                    &target_displacements[global_right],
                );
                let target_pair_mse = vector_mse(
                    &target_displacements[global_left],
                    &target_displacements[global_right],
                );
                let eligible = outcome_classes[left] != outcome_classes[right];
                if eligible {
                    let (_, _, pixel_oracle) = outcome_counterfactual_margin(
                        &pixel_displacements[left],
                        &pixel_displacements[right],
                        &pixel_displacements[left],
                        &pixel_displacements[right],
                    );
                    let (_, _, latent_oracle) = outcome_counterfactual_margin(
                        &target_displacements[global_left],
                        &target_displacements[global_right],
                        &target_displacements[global_left],
                        &target_displacements[global_right],
                    );
                    let (_, _, swapped_oracle) = outcome_counterfactual_margin(
                        &target_displacements[global_right],
                        &target_displacements[global_left],
                        &target_displacements[global_left],
                        &target_displacements[global_right],
                    );
                    let (_, _, action_masked) = outcome_counterfactual_margin(
                        &predicted_displacements[offset],
                        &predicted_displacements[offset],
                        &target_displacements[global_left],
                        &target_displacements[global_right],
                    );
                    let (_, _, identity_margin) = outcome_counterfactual_margin(
                        &identity_prediction,
                        &identity_prediction,
                        &target_displacements[global_left],
                        &target_displacements[global_right],
                    );
                    pixel_oracle_values.push((group_index, pixel_oracle));
                    latent_oracle_values.push((group_index, latent_oracle));
                    swapped_oracle_values.push((group_index, swapped_oracle));
                    action_masked_values.push((group_index, action_masked));
                    identity_values.push((group_index, identity_margin));
                    if target_pair_mse <= OUTCOME_COUNTERFACTUAL_TARGET_COLLAPSE_MSE {
                        target_collapsed_pairs += 1;
                    }
                }
                ledger.push(OutcomeCounterfactualPairLedgerRow {
                    group: identity.clone(),
                    left_branch_index: left,
                    right_branch_index: right,
                    left_action: branches[left].transition.action.clone(),
                    right_action: branches[right].transition.action.clone(),
                    left_outcome_class: outcome_classes[left],
                    right_outcome_class: outcome_classes[right],
                    left_changed: branches[left].board_effect.changed,
                    right_changed: branches[right].board_effect.changed,
                    left_changed_cells: branches[left].board_effect.changed_cells.clone(),
                    right_changed_cells: branches[right].board_effect.changed_cells.clone(),
                    target_pair_mse,
                    concordant_loss,
                    crossed_loss,
                    margin,
                    eligible,
                    reason: if eligible {
                        "distinct_canonical_board_outcomes".into()
                    } else {
                        "outcome_equivalent".into()
                    },
                });
            }
        }
        offset += branches.len();
    }

    let overall_group_means = ledger_group_means(&ledger, |_| true);
    let movement_group_means =
        ledger_group_means(&ledger, |row| row.group.population == "movement");
    let coordinate_group_means =
        ledger_group_means(&ledger, |row| row.group.population == "coordinate");
    let changed_changed_group_means =
        ledger_group_means(&ledger, |row| row.left_changed && row.right_changed);
    let changed_unchanged_group_means =
        ledger_group_means(&ledger, |row| row.left_changed != row.right_changed);
    let overall = outcome_interval_from_group_means(&overall_group_means, seed ^ 0xCF00_0001);
    let movement = outcome_interval_from_group_means(&movement_group_means, seed ^ 0xCF00_0002);
    let coordinate = outcome_interval_from_group_means(&coordinate_group_means, seed ^ 0xCF00_0003);
    let changed_changed =
        outcome_interval_from_group_means(&changed_changed_group_means, seed ^ 0xCF00_0004);
    let changed_unchanged =
        outcome_interval_from_group_means(&changed_unchanged_group_means, seed ^ 0xCF00_0005);
    let eligible_pairs = ledger.iter().filter(|row| row.eligible).count();
    let outcome_equivalent_pairs = ledger.len() - eligible_pairs;
    let changed_changed_pairs = ledger
        .iter()
        .filter(|row| row.eligible && row.left_changed && row.right_changed)
        .count();
    let changed_unchanged_pairs = ledger
        .iter()
        .filter(|row| row.eligible && row.left_changed != row.right_changed)
        .count();
    let equivalent_max_abs = ledger
        .iter()
        .filter(|row| !row.eligible)
        .map(|row| row.margin.abs())
        .reduce(f64::max);
    let action_masked_max_abs = action_masked_values
        .iter()
        .map(|(_, value)| value.abs())
        .reduce(f64::max);
    let identity_max_abs = identity_values
        .iter()
        .map(|(_, value)| value.abs())
        .reduce(f64::max);
    let pixel_oracle_estimate = equal_weight_control_mean(&pixel_oracle_values);
    let latent_oracle_estimate = equal_weight_control_mean(&latent_oracle_values);
    let swapped_oracle_estimate = equal_weight_control_mean(&swapped_oracle_values);
    let movement_group_indices = groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| {
            group.branches()[0]
                .transition
                .episode_id
                .is_multiple_of(2)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let movement_template = movement_group_indices.first().map(|index| {
        groups[*index]
            .branches()
            .iter()
            .map(|branch| branch.transition.action.clone())
            .collect::<Vec<_>>()
    });
    let templates_match = movement_template.as_ref().is_some_and(|template| {
        movement_group_indices.iter().all(|index| {
            groups[*index]
                .branches()
                .iter()
                .map(|branch| &branch.transition.action)
                .eq(template.iter())
        })
    });
    let mut state_scrambled_values = Vec::new();
    if movement_group_indices.len() >= 2 && templates_match {
        for (position, destination_group) in movement_group_indices.iter().copied().enumerate() {
            let source_group =
                movement_group_indices[(position + 1) % movement_group_indices.len()];
            let destination_classes = canonical_outcome_classes(&groups[destination_group]);
            let destination_offset = destination_group * FACTUAL_BRANCHES_PER_GROUP;
            let source_offset = source_group * FACTUAL_BRANCHES_PER_GROUP;
            for left in 0..FACTUAL_BRANCHES_PER_GROUP {
                for right in left + 1..FACTUAL_BRANCHES_PER_GROUP {
                    if destination_classes[left] == destination_classes[right] {
                        continue;
                    }
                    let (_, _, margin) = outcome_counterfactual_margin(
                        &predicted_displacements[source_offset + left],
                        &predicted_displacements[source_offset + right],
                        &target_displacements[destination_offset + left],
                        &target_displacements[destination_offset + right],
                    );
                    state_scrambled_values.push((destination_group, margin));
                }
            }
        }
    }
    let state_scrambled_estimate = equal_weight_control_mean(&state_scrambled_values);
    let state_scrambled_groups = state_scrambled_values
        .iter()
        .map(|(group, _)| *group)
        .collect::<BTreeSet<_>>()
        .len();
    let state_scrambled_same_action_template = OutcomeCounterfactualStateScrambledControl {
        available: state_scrambled_estimate.is_some(),
        estimate: state_scrambled_estimate,
        groups: state_scrambled_groups,
        pairs: state_scrambled_values.len(),
        reason: if state_scrambled_estimate.is_some() {
            None
        } else if !templates_match {
            Some("unavailable: movement groups do not share an identical action template".into())
        } else {
            Some("unavailable: fewer than two eligible movement groups".into())
        },
    };
    let target_collapse_failure = target_collapsed_pairs > 0;
    let pixel_oracle_exactly_one = !pixel_oracle_values.is_empty()
        && pixel_oracle_values.iter().all(|(_, value)| *value == 1.0);
    let latent_oracle_at_least_0_99 =
        !target_collapse_failure && latent_oracle_estimate.is_some_and(|value| value >= 0.99);
    let swapped_oracle_at_most_negative_0_99 =
        swapped_oracle_estimate.is_some_and(|value| value <= -0.99);
    let action_masked_pass = action_masked_max_abs.is_some_and(|value| value <= 1e-6);
    let identity_pass = identity_max_abs.is_some_and(|value| value <= 1e-6);
    let equivalent_pass = equivalent_max_abs.is_some_and(|value| value <= 1e-6);
    let required_controls_pass = pixel_oracle_exactly_one
        && latent_oracle_at_least_0_99
        && swapped_oracle_at_most_negative_0_99
        && action_masked_pass
        && identity_pass
        && equivalent_pass;
    let controls = OutcomeCounterfactualControls {
        pixel_oracle_estimate,
        pixel_oracle_exactly_one,
        latent_oracle_estimate,
        latent_oracle_at_least_0_99,
        target_collapse_failure,
        target_collapsed_pairs,
        swapped_oracle_estimate,
        swapped_oracle_at_most_negative_0_99,
        action_masked_max_abs_margin: action_masked_max_abs,
        action_masked_max_abs_at_most_1e_6: action_masked_pass,
        identity_max_abs_margin: identity_max_abs,
        identity_max_abs_at_most_1e_6: identity_pass,
        outcome_equivalent_pairs,
        outcome_equivalent_max_abs_margin: equivalent_max_abs,
        outcome_equivalent_max_abs_at_most_1e_6: equivalent_pass,
        state_scrambled_same_action_template,
        required_controls_pass,
    };
    let eligible_simulator_groups = movement_group_means.len();
    let simulator_changed_changed_pairs = ledger
        .iter()
        .filter(|row| {
            row.eligible
                && row.group.population == "movement"
                && row.left_changed
                && row.right_changed
        })
        .count();
    let eligible_simulator_groups_at_least_100 = eligible_simulator_groups >= 100;
    let each_movement_action_at_least_16_changed_and_16_unchanged = (1..=4).all(|action| {
        movement_action_anchors
            .get(&action)
            .is_some_and(|anchors| anchors.changed >= 16 && anchors.unchanged >= 16)
    });
    let simulator_changed_changed_pairs_at_least_100 = simulator_changed_changed_pairs >= 100;
    let population_pass = eligible_simulator_groups_at_least_100
        && each_movement_action_at_least_16_changed_and_16_unchanged
        && simulator_changed_changed_pairs_at_least_100
        && !target_collapse_failure;
    let population_gates = OutcomeCounterfactualPopulationGates {
        eligible_simulator_groups,
        eligible_simulator_groups_at_least_100,
        movement_action_anchors,
        each_movement_action_at_least_16_changed_and_16_unchanged,
        simulator_changed_changed_pairs,
        simulator_changed_changed_pairs_at_least_100,
        target_collapse_failure,
        population_pass,
    };
    let movement_groups = groups
        .iter()
        .filter(|group| group.branches()[0].transition.episode_id.is_multiple_of(2))
        .count();
    let action_separation_pass = movement.as_ref().is_some_and(|interval| {
        interval.estimate > OUTCOME_COUNTERFACTUAL_MATERIAL_THRESHOLD
            && interval.lower_95 > OUTCOME_COUNTERFACTUAL_MATERIAL_THRESHOLD
    });
    let ledger_reconciled = ledger.len() == groups.len() * pairs_per_group
        && eligible_pairs + outcome_equivalent_pairs == ledger.len()
        && ledger.iter().all(|row| {
            row.group.actions.len() == FACTUAL_BRANCHES_PER_GROUP
                && row.group.next_sha256.len() == FACTUAL_BRANCHES_PER_GROUP
                && row.left_branch_index < row.right_branch_index
                && row.right_branch_index < FACTUAL_BRANCHES_PER_GROUP
                && row.group.actions[row.left_branch_index] == row.left_action
                && row.group.actions[row.right_branch_index] == row.right_action
                && row.eligible == (row.left_outcome_class != row.right_outcome_class)
                && (row.margin
                    - (row.crossed_loss - row.concordant_loss)
                        / (row.crossed_loss + row.concordant_loss + OUTCOME_COUNTERFACTUAL_EPSILON))
                    .abs()
                    <= 1e-12
        })
        && outcome_interval_reconciles(&overall, &overall_group_means)
        && outcome_interval_reconciles(&movement, &movement_group_means)
        && outcome_interval_reconciles(&coordinate, &coordinate_group_means)
        && outcome_interval_reconciles(&changed_changed, &changed_changed_group_means)
        && outcome_interval_reconciles(&changed_unchanged, &changed_unchanged_group_means);
    let metrics = OutcomeCounterfactualMetrics {
        population_fingerprint: outcome_population_fingerprint(&group_fingerprints),
        groups: groups.len(),
        movement_groups,
        coordinate_groups: groups.len() - movement_groups,
        unordered_pairs: ledger.len(),
        eligible_pairs,
        outcome_equivalent_pairs,
        changed_changed_pairs,
        changed_unchanged_pairs,
        epsilon: OUTCOME_COUNTERFACTUAL_EPSILON,
        material_threshold: OUTCOME_COUNTERFACTUAL_MATERIAL_THRESHOLD,
        overall,
        movement,
        coordinate,
        changed_changed,
        changed_unchanged,
        action_separation_pass,
        controls,
        population_gates,
        pair_ledger: ledger,
        ledger_reconciled,
    };
    log_eval_phase(
        "metric_reduction",
        "factual_outcome_counterfactuals",
        reduction_started.elapsed(),
    );
    Ok(metrics)
}

fn evaluate_factual_branches(
    model: &WorldModel,
    cfg: &EvalConfig,
    device: &Device,
    factual_population: Option<&FactualBatch>,
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
            action_controllability: None,
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
            semantic_outcome_retrieval_n: 0,
            semantic_outcome_retrieval_accuracy: None,
            semantic_outcome_chance: None,
            semantic_factual_nll_mean: None,
            semantic_outcome_by_family: BTreeMap::new(),
        });
    }
    // The shared population is canonicalized once before either factual metric
    // consumer runs, so inference rows and labels cannot diverge in ordering.
    let factual_batch = factual_population
        .context("factual branch evaluation is missing its factual population")?;
    let groups = factual_batch.groups().to_vec();
    let population_fingerprint = factual_population_fingerprint(&groups, factual_group_count);
    let samples = factual_batch.rows().to_vec();

    let forward_started = Instant::now();
    let mut predicted_displacements = Vec::with_capacity(samples.len());
    let mut predicted_consumer_latents = Vec::with_capacity(samples.len());
    let mut action_logits = Vec::with_capacity(samples.len());
    let mut coordinate_predictions = Vec::with_capacity(samples.len());
    let mut target_patch_latents: Option<BoardProbeRows> = None;
    let mut predicted_patch_latents: Option<BoardProbeRows> = None;
    let mut predicted_gameplay_log_probs = Vec::<Vec<f32>>::new();
    let factual_eval_batch = cfg
        .physical_batch
        .max(crate::p2::data::FACTUAL_BRANCHES_PER_GROUP)
        / crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
        * crate::p2::data::FACTUAL_BRANCHES_PER_GROUP;
    for (start, end) in batch_ranges(samples.len(), factual_eval_batch) {
        let batch = batch_from_samples(&samples[start..end], device)?;
        let current = model.encode_state(&batch.frames)?;
        let output = model.forward_with_operator_conditioning(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            &batch.operator_conditioning,
        )?;
        predicted_consumer_latents.extend(
            flatten_latent(&output.y)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?,
        );
        let target = model.encode_state(&batch.next_frames)?;
        let displacement = pool_latent(&output.y.sub(&current)?)?;
        predicted_displacements.extend(displacement.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        if model.config().world_core_v2 {
            let (decoded_actions, decoded_coordinates) =
                model.decode_action_displacement(&displacement)?;
            action_logits.extend(decoded_actions.to_dtype(DType::F32)?.to_vec2::<f32>()?);
            coordinate_predictions
                .extend(decoded_coordinates.to_dtype(DType::F32)?.to_vec2::<f32>()?);
        }
        if model.config().world_core_v4 {
            predicted_gameplay_log_probs.extend(
                ops::log_softmax(&model.exact_gameplay_logits(&output.y)?, D::Minus1)?
                    .reshape((end - start, ()))?
                    .to_vec2::<f32>()?,
            );
        }
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
    log_eval_phase(
        "model_forwards",
        "factual_branches",
        forward_started.elapsed(),
    );
    let reduction_started = Instant::now();

    let legal_actions = groups
        .iter()
        .map(|group| {
            group
                .branches()
                .iter()
                .map(|branch| branch.transition.action.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let action_controllability = {
        let mut group_offsets = Vec::with_capacity(groups.len());
        let mut offset = 0usize;
        for group in &groups {
            group_offsets.push(offset);
            offset += group.branches().len();
        }
        Some(action_controllability_probe(
            &legal_actions,
            ACTION_CONTROLLABILITY_LATENT_DISTANCE_THRESHOLD,
            |state_index, action| {
                let local = legal_actions[state_index]
                    .iter()
                    .position(|candidate| candidate == action)
                    .expect("probe action comes from the state's legal-action list");
                Ok(predicted_consumer_latents[group_offsets[state_index] + local].clone())
            },
        )?)
    };

    let board_probe_started = Instant::now();
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
    log_eval_phase(
        "board_probe",
        "factual_branches",
        board_probe_started.elapsed(),
    );

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
    let mut semantic_outcome_retrieval_credit = 0.0f64;
    let mut semantic_outcome_chance_sum = 0.0f64;
    let mut semantic_factual_nll_sum = 0.0f64;
    let mut semantic_outcome_retrieval_n = 0usize;
    let mut offset = 0usize;
    for (group_index, group) in groups.iter().enumerate() {
        let branches = group.branches();
        let mut outcome_representatives = Vec::<usize>::new();
        for (candidate, branch) in branches.iter().enumerate() {
            if !outcome_representatives
                .iter()
                .any(|representative| branch.outcome_equivalent(&branches[*representative]))
            {
                outcome_representatives.push(candidate);
            }
        }
        let source = &branches[0].transition;
        let group_key = format!(
            "{}:{}:{}:{}",
            source.seed,
            source.episode_id,
            source.provenance.trajectory_id,
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
            let predicted_action = action_logits.get(global).and_then(|logits| {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(left_index, left), (right_index, right)| {
                        left.partial_cmp(right)
                            .unwrap_or_else(|| left_index.cmp(right_index))
                    })
                    .map(|(index, _)| index as u8)
            });
            let outcome_class = outcome_representatives
                .iter()
                .position(|representative| branch.outcome_equivalent(&branches[*representative]))
                .expect("branch is equivalent to its representative");
            let (
                factual_outcome_nll,
                best_outcome_classes,
                factual_outcome_retrieval_credit,
                factual_outcome_chance,
            ) = if model.config().world_core_v4 {
                let log_probs = &predicted_gameplay_log_probs[global];
                let width = usize::from(branch.transition.provenance.content_width)
                    .min(crate::p2::data::FRAME_SIDE);
                let height = usize::from(branch.transition.provenance.content_height).min(
                    log_probs.len()
                        / (crate::p2::data::FRAME_SIDE * crate::p2::model::PALETTE_SIZE),
                );
                let mut class_nll = Vec::with_capacity(outcome_representatives.len());
                for representative in &outcome_representatives {
                    let target = &branches[*representative].transition.next.pixels;
                    let mut sum = 0.0f64;
                    let mut count = 0usize;
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = y * crate::p2::data::FRAME_SIDE + x;
                            sum -= f64::from(
                                log_probs[pixel * crate::p2::model::PALETTE_SIZE
                                    + target[pixel] as usize],
                            );
                            count += 1;
                        }
                    }
                    class_nll.push(if count > 0 {
                        sum / count as f64
                    } else {
                        f64::INFINITY
                    });
                }
                let best = class_nll.iter().copied().fold(f64::INFINITY, f64::min);
                let ties = class_nll
                    .iter()
                    .enumerate()
                    .filter_map(|(class, nll)| ((nll - best).abs() <= 1e-9).then_some(class))
                    .collect::<Vec<_>>();
                let credit = if ties.contains(&outcome_class) {
                    1.0 / ties.len().max(1) as f64
                } else {
                    0.0
                };
                let factual_nll = class_nll[outcome_class];
                let chance = 1.0 / outcome_representatives.len().max(1) as f64;
                semantic_outcome_retrieval_credit += credit;
                semantic_outcome_chance_sum += chance;
                semantic_factual_nll_sum += factual_nll;
                semantic_outcome_retrieval_n += 1;
                (Some(factual_nll), ties, Some(credit), Some(chance))
            } else {
                (None, Vec::new(), None, None)
            };
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
                predicted_action_x_normalized: coordinate_predictions
                    .get(global)
                    .map(|coordinates| coordinates[0]),
                predicted_action_y_normalized: coordinate_predictions
                    .get(global)
                    .map(|coordinates| coordinates[1]),
                factual_outcome_nll,
                best_outcome_classes,
                factual_outcome_retrieval_credit,
                factual_outcome_chance,
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

            if is_recoverable && predicted_action.is_some() {
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
    let recoverable_action_counts = rows
        .iter()
        .filter(|row| row.recoverable && row.predicted_action_id.is_some())
        .fold(BTreeMap::<u8, usize>::new(), |mut counts, row| {
            *counts.entry(row.action_id).or_default() += 1;
            counts
        });
    let majority_action_baseline_top1 = (!recoverable_action_counts.is_empty()).then(|| {
        recoverable_action_counts
            .values()
            .copied()
            .max()
            .unwrap_or(0) as f64
            / recoverable_action_counts.values().sum::<usize>() as f64
    });
    let mut semantic_family_rows = BTreeMap::<String, Vec<&FactualBranchRowMetric>>::new();
    for row in &rows {
        if row.factual_outcome_retrieval_credit.is_some() {
            semantic_family_rows
                .entry(row.family.clone())
                .or_default()
                .push(row);
        }
    }
    let semantic_outcome_by_family = semantic_family_rows
        .into_iter()
        .map(|(family, rows)| {
            let n = rows.len();
            let sum = |read: fn(&FactualBranchRowMetric) -> Option<f64>| {
                rows.iter().filter_map(|row| read(row)).sum::<f64>()
            };
            (
                family,
                FactualSemanticOutcomeStratum {
                    n,
                    retrieval_accuracy: (n > 0)
                        .then_some(sum(|row| row.factual_outcome_retrieval_credit) / n as f64),
                    chance: (n > 0).then_some(sum(|row| row.factual_outcome_chance) / n as f64),
                    factual_nll_mean: (n > 0)
                        .then_some(sum(|row| row.factual_outcome_nll) / n as f64),
                },
            )
        })
        .collect();
    let metrics = FactualBranchMetrics {
        population_fingerprint,
        groups: groups.len(),
        branches: counts.branches,
        changed: counts.changed,
        unchanged: counts.unchanged,
        recoverable: counts.recoverable,
        action6: counts.action6,
        action_controllability,
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
        semantic_outcome_retrieval_n,
        semantic_outcome_retrieval_accuracy: (semantic_outcome_retrieval_n > 0)
            .then_some(semantic_outcome_retrieval_credit / semantic_outcome_retrieval_n as f64),
        semantic_outcome_chance: (semantic_outcome_retrieval_n > 0)
            .then_some(semantic_outcome_chance_sum / semantic_outcome_retrieval_n as f64),
        semantic_factual_nll_mean: (semantic_outcome_retrieval_n > 0)
            .then_some(semantic_factual_nll_sum / semantic_outcome_retrieval_n as f64),
        semantic_outcome_by_family,
    };
    log_eval_phase(
        "metric_reduction",
        "factual_branches",
        reduction_started.elapsed(),
    );
    Ok(metrics)
}

fn begin_gate_eval_profile(
    cfg: &EvalConfig,
    train_cfg: &TrainConfig,
) -> Result<Option<RepresentativeUpdateCapture>> {
    if !cfg.profile_eval
        || cfg.mode != EvalMode::Full
        || train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FoundationV2
    {
        return Ok(None);
    }
    let output_dir = cfg
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let profile_dir = output_dir.join("profile");
    let bundle_name = "eval-000000000001";
    ensure_eval_profile_campaign(
        &profile_dir,
        "eval-campaign.json",
        format!("tofy.p2.eval.{}", file_sha256(&cfg.checkpoint)?),
        EVAL_PROFILE_ENTRYPOINT,
        vec![PlannedCapture {
            capture_step: 1,
            bundle: bundle_name.into(),
        }],
    )?;
    let destination = profile_dir.join(bundle_name);
    RepresentativeUpdateCapture::begin_eval(EvalCaptureSpec {
        destination: &destination,
        capture_step: 1,
        entrypoint: EVAL_PROFILE_ENTRYPOINT,
        correlation_id: "tofy.p2/eval-000000000001".into(),
        device: &cfg.device,
        required_phases: &["encode", "forward", "decode", "metrics"],
        tags: &[
            ("population", "v5_unseen_seed_7x7".into()),
            ("physical_batch", "512".into()),
        ],
    })
    .map(Some)
}

/// Full evaluation: synthetic held-out (+ optional ARC recordings dir).
pub fn evaluate(cfg: &EvalConfig) -> Result<EvalReport> {
    evaluate_impl(cfg, true)
}

fn evaluate_impl(cfg: &EvalConfig, allow_gate_profile: bool) -> Result<EvalReport> {
    let evaluation_started = Instant::now();
    cfg.validate()?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    if let Some(bundle) = cfg.checkpoint.parent() {
        let manifest = bundle.join("bundle-manifest.json");
        if manifest.is_file() {
            verify_checkpoint_bundle(bundle)?;
            let bundled_config = bundle.join("config.json");
            if file_sha256(&bundled_config)? != file_sha256(&cfg.train_config)? {
                bail!(
                    "evaluation train config {} does not match checkpoint bundle {}",
                    cfg.train_config.display(),
                    bundled_config.display()
                );
            }
        } else if train_cfg.world_core_v4 {
            bail!(
                "Full V4 evaluation requires a verified checkpoint bundle manifest beside {}",
                cfg.checkpoint.display()
            );
        }
    }
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    if train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FullV4
        && (cfg.q_mse_threshold - train_cfg.q_mse_threshold).abs() > f64::EPSILON
    {
        bail!(
            "evaluation q_mse_threshold={} differs from frozen training threshold={}",
            cfg.q_mse_threshold,
            train_cfg.q_mse_threshold
        );
    }
    let device = resolve_device(&cfg.device)?;
    let (model, varmap) = timed_eval_phase("model_load", "checkpoint", || {
        load_model(&train_cfg, &cfg.checkpoint, &device)
    })?;
    let needs_factual_population =
        cfg.mode == EvalMode::Full || train_cfg.world_core_v2 || train_cfg.world_core_v4;
    let factual_population = if needs_factual_population {
        timed_eval_phase("population_generation", "factual_shared", || {
            generate_factual_eval_population(cfg.seed, cfg.synthetic_episodes)
        })?
    } else {
        None
    };
    let outcome_counterfactuals = (cfg.mode == EvalMode::Full)
        .then(|| {
            evaluate_outcome_counterfactuals(&model, cfg, &device, factual_population.as_ref())
        })
        .transpose()?;
    let factual_branches = (train_cfg.world_core_v2 || train_cfg.world_core_v4)
        .then(|| evaluate_factual_branches(&model, cfg, &device, factual_population.as_ref()))
        .transpose()?;

    let mut dynamics_sources = timed_eval_phase(
        "population_generation",
        "synthetic_ood_dynamics[random_one_step,exploration]",
        || {
            collect_synthetic_sources(
                cfg.seed,
                cfg.synthetic_episodes,
                &["random_one_step", "exploration"],
                Split::HeldOutComposition,
            )
        },
    )?;
    let dynamics_rollout_samples = timed_eval_phase(
        "rollout_collection",
        "synthetic_ood_dynamics[cached_h1]",
        || flatten_sources(&dynamics_sources),
    );
    dynamics_sources.push((
        "hazard_one_step".into(),
        timed_eval_phase(
            "population_generation",
            "synthetic_ood_dynamics[hazard_one_step]",
            || collect_hazard_samples(cfg.seed, cfg.synthetic_episodes, Split::HeldOutComposition),
        )?,
    ));
    let dynamics_samples = flatten_sources(&dynamics_sources);
    let dynamics_source_lengths = source_lengths(&dynamics_sources);
    let board_probe = timed_eval_phase("board_probe", "synthetic_ood_dynamics", || {
        evaluate_board_probe(&model, &dynamics_samples, cfg.physical_batch, &device)
    })?;

    let planner_sources = timed_eval_phase(
        "population_generation",
        "synthetic_ood_planner[sequential,hypothesis_probe,p1c_falsification,p1c_hard_retarget]",
        || {
            collect_synthetic_sources(
                cfg.seed,
                cfg.synthetic_episodes,
                &[
                    "sequential",
                    "hypothesis_probe",
                    "p1c_falsification",
                    "p1c_hard_retarget",
                ],
                Split::HeldOutComposition,
            )
        },
    )?;
    let planner_samples = flatten_sources(&planner_sources);
    let planner_source_lengths = source_lengths(&planner_sources);
    let planner_rollout_samples =
        timed_eval_phase("rollout_collection", "synthetic_ood_planner", || {
            collect_planner_rollout_samples(
                cfg.seed,
                cfg.synthetic_episodes,
                Split::HeldOutComposition,
                Some(&planner_sources[0].1),
            )
        })?;

    let mut iid_dynamics_sources = timed_eval_phase(
        "population_generation",
        "synthetic_iid_dynamics[random_one_step,exploration]",
        || {
            collect_synthetic_sources(
                cfg.iid_seed,
                cfg.synthetic_episodes,
                &["random_one_step", "exploration"],
                Split::Train,
            )
        },
    )?;
    let iid_dynamics_rollout_samples = timed_eval_phase(
        "rollout_collection",
        "synthetic_iid_dynamics[cached_h1]",
        || flatten_sources(&iid_dynamics_sources),
    );
    iid_dynamics_sources.push((
        "hazard_one_step".into(),
        timed_eval_phase(
            "population_generation",
            "synthetic_iid_dynamics[hazard_one_step]",
            || collect_hazard_samples(cfg.iid_seed, cfg.synthetic_episodes, Split::Train),
        )?,
    ));
    let iid_dynamics_samples = flatten_sources(&iid_dynamics_sources);
    let iid_dynamics_source_lengths = source_lengths(&iid_dynamics_sources);
    let iid_planner_sources = timed_eval_phase(
        "population_generation",
        "synthetic_iid_planner[sequential,hypothesis_probe,p1c_falsification,p1c_hard_retarget]",
        || {
            collect_synthetic_sources(
                cfg.iid_seed,
                cfg.synthetic_episodes,
                &[
                    "sequential",
                    "hypothesis_probe",
                    "p1c_falsification",
                    "p1c_hard_retarget",
                ],
                Split::Train,
            )
        },
    )?;
    let iid_planner_samples = flatten_sources(&iid_planner_sources);
    let iid_planner_source_lengths = source_lengths(&iid_planner_sources);
    let iid_planner_rollout_samples =
        timed_eval_phase("rollout_collection", "synthetic_iid_planner", || {
            collect_planner_rollout_samples(
                cfg.iid_seed,
                cfg.synthetic_episodes,
                Split::Train,
                Some(&iid_planner_sources[0].1),
            )
        })?;

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
        timed_eval_phase("rollout_forwards", "synthetic_dynamics", || {
            eval_episode_rollouts(
                &train_cfg,
                &cfg.checkpoint,
                &model,
                &dynamics_rollout_samples,
                &device,
                "synthetic_dynamics",
            )
        })?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
        attach_rollout_metrics(&mut synthetic_dynamics, &dynamics_rollout_rows, cfg.seed);
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
        timed_eval_phase("rollout_forwards", "synthetic_planner", || {
            eval_episode_rollouts(
                &train_cfg,
                &cfg.checkpoint,
                &model,
                &planner_rollout_samples,
                &device,
                "synthetic_planner",
            )
        })?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
        attach_rollout_metrics(&mut synthetic_planner, &planner_rollout_rows, cfg.seed);
    }

    let mut synthetic_iid_dynamics = if cfg.mode == EvalMode::Rollout {
        empty_split("synthetic_iid_dynamics", iid_dynamics_samples.len())
    } else {
        eval_sample_set(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &iid_dynamics_samples,
            "synthetic_iid_dynamics",
            Some(&iid_dynamics_source_lengths),
            cfg,
            &device,
            false,
        )?
    };
    let iid_dynamics_rollout_rows = if cfg.mode != EvalMode::Representation {
        timed_eval_phase("rollout_forwards", "synthetic_iid_dynamics", || {
            eval_episode_rollouts(
                &train_cfg,
                &cfg.checkpoint,
                &model,
                &iid_dynamics_rollout_samples,
                &device,
                "synthetic_iid_dynamics",
            )
        })?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
        attach_rollout_metrics(
            &mut synthetic_iid_dynamics,
            &iid_dynamics_rollout_rows,
            cfg.iid_seed,
        );
    }

    let mut synthetic_iid_planner = if cfg.mode == EvalMode::Rollout {
        empty_split("synthetic_iid_planner", iid_planner_samples.len())
    } else {
        eval_sample_set(
            &train_cfg,
            &cfg.checkpoint,
            &model,
            &iid_planner_samples,
            "synthetic_iid_planner",
            Some(&iid_planner_source_lengths),
            cfg,
            &device,
            false,
        )?
    };
    let iid_planner_rollout_rows = if cfg.mode != EvalMode::Representation {
        timed_eval_phase("rollout_forwards", "synthetic_iid_planner", || {
            eval_episode_rollouts(
                &train_cfg,
                &cfg.checkpoint,
                &model,
                &iid_planner_rollout_samples,
                &device,
                "synthetic_iid_planner",
            )
        })?
    } else {
        Vec::new()
    };
    if cfg.mode != EvalMode::Representation {
        attach_rollout_metrics(
            &mut synthetic_iid_planner,
            &iid_planner_rollout_rows,
            cfg.iid_seed,
        );
    }

    let arc3_recordings = if cfg.mode != EvalMode::Rollout {
        cfg.arc_recordings_dir
            .as_ref()
            .map(|dir| {
                timed_eval_phase("population_generation", "arc3_recordings[shared]", || {
                    import_and_summarize_recordings_dir(dir)
                })
            })
            .transpose()?
    } else {
        None
    };
    let (arc3_transfer, arc3_population_fingerprint) = if let Some((samples, _)) = &arc3_recordings
    {
        let fingerprint = semantic_population_fingerprint(&samples);
        (
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
            )?),
            Some(fingerprint),
        )
    } else {
        (None, None)
    };

    let arc3_recording_runs = if cfg.mode == EvalMode::Full {
        if let Some((_, runs)) = arc3_recordings {
            let reduction_started = Instant::now();
            Some(Arc3RecordingBenchmark {
                n_runs: runs.len(),
                total_actions: runs.iter().map(|r| r.actions).sum(),
                total_levels_completed: runs.iter().map(|r| r.levels_completed).sum(),
                runs,
            })
            .inspect(|_| {
                log_eval_phase(
                    "metric_reduction",
                    "arc3_recording_runs",
                    reduction_started.elapsed(),
                )
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

    let mut eval_profile = if allow_gate_profile {
        begin_gate_eval_profile(cfg, &train_cfg)?
    } else {
        None
    };
    let foundation_holdout = (cfg.mode == EvalMode::Full
        && train_cfg.recipe == crate::p2::experiment::TrainingRecipe::FoundationV2)
        .then(|| {
            timed_eval_phase("population_generation", "v5_holdout_gates", || {
                foundation_v2_v5_holdout_gates(
                    &model,
                    cfg,
                    &train_cfg,
                    &device,
                    eval_profile.as_ref(),
                )
            })
        })
        .transpose()?;
    if let Some(profile) = eval_profile.take() {
        profile.finish()?;
    }
    let (v5_holdout_gates, v6_context_strata, v6_context_ablation, mechanism_ablations) =
        if let Some((gates, context_strata, samples, masks, provenance)) = foundation_holdout {
            let ablations = timed_eval_phase("mechanism_ablations", "v5_unseen_seed_7x7", || {
                evaluate_mechanism_ablations(
                    &model,
                    &varmap,
                    &samples,
                    Some(&masks),
                    Some(&provenance),
                    &device,
                )
            })?;
            let context_ablation = (cfg.context_ablation && model.config().world_core_v6)
                .then(|| {
                    timed_eval_phase("context_ablation", "v5_unseen_seed_7x7", || {
                        evaluate_context_ablation(
                            &model,
                            "unseen_seed_7x7",
                            &samples,
                            &masks,
                            &provenance,
                            &device,
                        )
                    })
                })
                .transpose()?;
            (
                Some(gates),
                context_strata,
                context_ablation,
                Some(ablations),
            )
        } else {
            (None, None, None, None)
        };
    if cfg.context_ablation && !model.config().world_core_v6 {
        eprintln!(
            "p2-eval: --context-ablation ignored; the checkpoint is not world_core_v6 \
             (ADR 0005 §5.1 applies to v6 checkpoints only)"
        );
    }
    let v6_adaptation_falsifier = if cfg.adaptation_falsifier && model.config().world_core_v6 {
        let spec = AdaptationFalsifierSpec::preregistered(
            cfg.synthetic_episodes,
            cfg.adaptation_falsifier_min_level_transitions,
        );
        if spec.population_seed == train_cfg.seed {
            bail!(
                "adaptation falsifier population seed collides with the checkpoint's \
                 training seed; the held-out claim would be false"
            );
        }
        Some(timed_eval_phase(
            "adaptation_falsifier",
            "learning_histories_unseen_seed_7x7",
            || evaluate_adaptation_falsifier(&model, &varmap, &device, &spec),
        )?)
    } else {
        if cfg.adaptation_falsifier {
            eprintln!(
                "p2-eval: --adaptation-falsifier ignored; the checkpoint is not world_core_v6 \
                 (ADR 0005 §5.2 applies to v6 checkpoints only)"
            );
        }
        None
    };

    let report_reduction_started = Instant::now();
    let mut population_sha256 = BTreeMap::from([
        (
            "synthetic_ood_dynamics_h1".into(),
            semantic_population_fingerprint(&dynamics_samples),
        ),
        (
            "synthetic_ood_dynamics_rollout".into(),
            semantic_population_fingerprint(&dynamics_rollout_samples),
        ),
        (
            "synthetic_ood_planner_h1".into(),
            semantic_population_fingerprint(&planner_samples),
        ),
        (
            "synthetic_ood_planner_rollout".into(),
            semantic_population_fingerprint(&planner_rollout_samples),
        ),
        (
            "synthetic_iid_dynamics_h1".into(),
            semantic_population_fingerprint(&iid_dynamics_samples),
        ),
        (
            "synthetic_iid_dynamics_rollout".into(),
            semantic_population_fingerprint(&iid_dynamics_rollout_samples),
        ),
        (
            "synthetic_iid_planner_h1".into(),
            semantic_population_fingerprint(&iid_planner_samples),
        ),
        (
            "synthetic_iid_planner_rollout".into(),
            semantic_population_fingerprint(&iid_planner_rollout_samples),
        ),
    ]);
    if let Some(factual) = &factual_branches {
        population_sha256.insert(
            "factual_same_state_branches".into(),
            factual.population_fingerprint.clone(),
        );
    }
    if let Some(fingerprint) = arc3_population_fingerprint {
        population_sha256.insert("arc3_transfer".into(), fingerprint);
    }
    if let Some(gates) = &v5_holdout_gates {
        for (name, metrics) in gates {
            population_sha256.insert(
                format!("v5_holdout_{name}"),
                metrics.population_fingerprint.clone(),
            );
        }
    }
    if let Some(strata) = &v6_context_strata {
        for (split, by_stratum) in strata {
            for (stratum, metrics) in by_stratum {
                population_sha256.insert(
                    format!("v6_context_{split}_{stratum}"),
                    metrics.population_fingerprint.clone(),
                );
            }
        }
    }
    let identity = evaluation_identity(cfg, population_sha256)?;

    let report = EvalReport {
        schema: EVAL_REPORT_SCHEMA.into(),
        mode: cfg.mode,
        seed: cfg.seed,
        iid_seed: cfg.iid_seed,
        identity,
        checkpoint: cfg.checkpoint.clone(),
        device: cfg.device.clone(),
        q_mse_threshold: cfg.q_mse_threshold,
        q_label_definition: if train_cfg.recipe == crate::p2::experiment::TrainingRecipe::FullV4 {
            "exact_gameplay_pixels:overall>=0.99,changed>=0.90,status_row_excluded".into()
        } else {
            format!("latent_mse<{}", cfg.q_mse_threshold)
        },
        ptrm_k: cfg.ptrm_k.clone(),
        ptrm_noise: cfg.ptrm_noise,
        official_rhae,
        official_scorecard,
        arc3_recording_runs,
        public_data_used_for_fitting: false,
        synthetic_dynamics,
        synthetic_planner,
        synthetic_iid_dynamics,
        synthetic_iid_planner,
        board_probe,
        factual_branches,
        outcome_counterfactuals,
        arc3_transfer,
        v5_holdout_gates,
        v6_context_strata,
        v6_context_ablation,
        v6_adaptation_falsifier,
        mechanism_ablations,
        research_claim: false,
    };
    if let Some(jsonl) = cfg.episode_jsonl.as_deref() {
        let mut rows = dynamics_rollout_rows;
        rows.extend(planner_rollout_rows);
        rows.extend(iid_dynamics_rollout_rows);
        rows.extend(iid_planner_rollout_rows);
        sort_episode_rows(&mut rows);
        maybe_write_episode_jsonl(Some(jsonl), &rows)?;
    }
    write_json_atomic(&cfg.output, &report)?;
    write_eval_digest(&cfg.output)?;
    log_eval_phase(
        "metric_reduction",
        "report_and_output",
        report_reduction_started.elapsed(),
    );
    log_eval_phase("total", "p2_eval", evaluation_started.elapsed());
    Ok(report)
}

/// ARC-only transfer eval helper.
pub fn evaluate_arc3(cfg: &EvalConfig) -> Result<EvalReport> {
    if cfg.arc_recordings_dir.is_none() {
        bail!("p2-arc3-eval requires --arc-recordings-dir");
    }
    let report = evaluate_impl(cfg, false)?;
    if cfg.profile_eval {
        crate::p2::arc3_live::profile_recorded_decisions(
            &cfg.checkpoint,
            &cfg.train_config,
            &cfg.device,
            cfg.arc_recordings_dir
                .as_deref()
                .expect("recordings path checked above"),
            &cfg.output,
            cfg.physical_batch,
        )?;
    }
    Ok(report)
}

// ---------------------------------------------------------------------------
// ADR 0004 A3 Phase A calibration, fitted from the synthetic held-out
// population only (`p2-eval --emit-phase-a-calibration <path>`).
// ---------------------------------------------------------------------------

/// ADR 0004 `epsilon = 0.02`: the likelihood clip of the belief update and the
/// quantile level (`1 - epsilon`) reported as `score_error_bound`.
pub const PHASE_A_CALIBRATION_EPSILON: f64 = 0.02;
/// The held-out population the artifact is fitted on (`v5_holdout_gates` key).
pub const PHASE_A_CALIBRATION_POPULATION: &str = "unseen_seed_7x7";

/// Exact terminal channel of a held-out row under its own (non-dropped) goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseAOutcome {
    Satisfied,
    Failed,
    Exhausted,
    Ordinary,
}

/// One held-out row's raw head readouts and exact outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseACalibrationRow {
    pub q_raw: f32,
    pub reliability_raw: f32,
    /// Exact composed-transition correctness of the decoded next frame.
    pub exact: bool,
    pub noop_raw: f32,
    pub satisfied_raw: f32,
    pub failed_raw: f32,
    pub exhausted_raw: f32,
    /// `None` when the row's goal was dropped or its labels are masked.
    pub outcome: Option<PhaseAOutcome>,
}

fn phase_a_outcome(sample: &TransitionSample) -> Option<PhaseAOutcome> {
    if sample.goal_features.values == [0.0; GOAL_FEATURES_DIM] {
        return None;
    }
    match (sample.goal_satisfied, sample.goal_failed, sample.exhausted) {
        (Some(true), _, _) => Some(PhaseAOutcome::Satisfied),
        (_, Some(true), _) => Some(PhaseAOutcome::Failed),
        (_, _, Some(true)) => Some(PhaseAOutcome::Exhausted),
        (Some(false), Some(false), _) => Some(PhaseAOutcome::Ordinary),
        _ => None,
    }
}

/// Raw q / reliability / event readouts and exact outcomes of `samples`,
/// scored exactly like the gate evaluator (frame-encoded current latent,
/// the row's own context window, `FOUNDATION_V2_GATE_PHYSICAL_BATCH` chunks,
/// outputs detached per chunk).
pub fn phase_a_calibration_rows(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
) -> Result<Vec<PhaseACalibrationRow>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if !model.config().world_core_v4 {
        bail!("Phase A calibration requires the exact gameplay decoder (world_core_v4)");
    }
    let batch = batch_from_samples(samples, device)?;
    let (current, _target) = model.encode_state_pair(&batch.frames, &batch.next_frames)?;
    let current = current.detach();
    let context = model
        .config()
        .world_core_v6
        .then(|| ContextBatch::from_samples(samples, device))
        .transpose()?
        .flatten();
    let depth = RecursionDepth::from_config(model.config());
    let chunk = FOUNDATION_V2_GATE_PHYSICAL_BATCH.max(1);
    let mut rows = Vec::with_capacity(samples.len());
    let mut start = 0;
    while start < samples.len() {
        let len = chunk.min(samples.len() - start);
        let indices = (start..start + len)
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        let chunk_context = match context.as_ref() {
            Some(context) => context.select_rows(&indices)?,
            None => None,
        };
        let out = model.forward_from_latent_with_depth_and_operator_conditioning_with_context(
            &current.narrow(0, start, len)?.contiguous()?,
            &batch.actions.narrow(0, start, len)?.contiguous()?,
            &batch.action_coords.narrow(0, start, len)?.contiguous()?,
            &batch.goals.narrow(0, start, len)?.contiguous()?,
            &batch
                .operator_conditioning
                .narrow(0, start, len)?
                .contiguous()?,
            chunk_context.as_ref(),
            depth,
        )?;
        let q = ops::sigmoid(&out.q_logit.detach().to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reliability = ops::sigmoid(&out.reliability_logit.detach().to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let events = ops::sigmoid(&out.event_logits.detach().to_dtype(DType::F32)?)?
            .reshape((len, ()))?
            .to_vec2::<f32>()?;
        let exact = model
            .exact_transition_correctness(
                &out.y.detach(),
                &batch.frames.narrow(0, start, len)?.contiguous()?,
                &batch.next_frames.narrow(0, start, len)?.contiguous()?,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if q.len() != len || reliability.len() != len || events.len() != len || exact.len() != len {
            bail!("Phase A calibration readout rows do not match the chunk");
        }
        for index in 0..len {
            let read = |event: usize| events[index].get(event).copied().unwrap_or(0.0);
            rows.push(PhaseACalibrationRow {
                q_raw: q[index],
                reliability_raw: reliability[index],
                exact: exact[index] >= 0.5,
                noop_raw: read(EVENT_NOOP),
                satisfied_raw: read(EVENT_GOAL_SATISFIED),
                failed_raw: read(EVENT_GOAL_FAILED),
                exhausted_raw: read(EVENT_EXHAUSTED),
                outcome: phase_a_outcome(&samples[start + index]),
            });
        }
        start += len;
    }
    Ok(rows)
}

/// 95% Clopper-Pearson upper endpoint (the existing risk-coverage bound) on
/// `failures / total`; `None` with no support.
fn phase_a_error_bin(failures: u64, total: u64) -> Option<CalibrationBin> {
    let upper_error_bound_95 = clopper_pearson_upper(failures, total)?;
    Some(CalibrationBin {
        upper_error_bound_95,
        support: total,
    })
}

/// Empirical upper `p`-quantile (smallest value with at least a `p` fraction
/// of the sample at or below it).
fn phase_a_upper_quantile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let index = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    Some(sorted[index])
}

/// Fit the ADR 0004 A3 record from held-out rows. Every bin is the observed
/// error rate of the region the corresponding gate trusts, with the 95%
/// Clopper-Pearson upper endpoint and the region's row count as support:
///
/// - `q_direction`: `-1` when AUROC(q, exact) < 0.5, else `1`;
/// - `ordinary`: rows with direction-adjusted `q >= 0.5` and
///   `reliability >= 0.5` (the `edge_trust` region); error = not exact;
/// - `event_false_safe`: labeled rows the event head calls safe
///   (`failed + exhausted < 0.5`); error = observed failed/exhausted;
/// - `satisfaction`: labeled rows with `satisfied >= 0.5`; error = not
///   satisfied;
/// - `tau_unknown`: 95th percentile of `-ln(clip(L, eps, 1 - eps))` of the
///   observed channel's raw likelihood, clamped into the `(0, 1)` domain the
///   record requires (the raw quantile is kept in `fit`);
/// - `score_error_bound`: the `1 - eps` quantile of
///   `|satisfied_raw - 1[satisfied]|` on labeled rows (ADR 0004 Amendment 1
///   charges `2 x` this bound);
/// - `ptrm`: not fitted (`None`; no gate consumes it yet).
///
/// The generator masks the `exhausted` label (see `data.rs`), so the
/// false-safe bin covers failure. No row is empty-population tolerant: with
/// no rows the record is `uncalibrated`.
pub fn fit_phase_a_calibration(rows: &[PhaseACalibrationRow]) -> PhaseACalibration {
    let eps = PHASE_A_CALIBRATION_EPSILON;
    let mut calibration = PhaseACalibration::fail_closed();
    let mut fit = PhaseACalibrationFit {
        rows: rows.len(),
        score_error_quantile: 1.0 - eps,
        bound_method: "clopper_pearson_upper (95%)".into(),
        unfitted: vec![
            "ptrm: PTRM disagreement calibration is not emitted; no gate consumes it".into(),
        ],
        ..PhaseACalibrationFit::default()
    };
    if rows.is_empty() {
        calibration.fit = Some(fit);
        return calibration;
    }
    let q_scores = rows.iter().map(|row| row.q_raw).collect::<Vec<_>>();
    let exact = rows.iter().map(|row| row.exact).collect::<Vec<_>>();
    fit.q_auroc = binary_auroc(&q_scores, &exact);
    fit.exact_rate = Some(exact.iter().filter(|e| **e).count() as f64 / rows.len() as f64);
    let q_direction: i8 = if fit.q_auroc.is_some_and(|auroc| auroc < 0.5) {
        -1
    } else {
        1
    };
    let adjusted_q = |row: &PhaseACalibrationRow| {
        if q_direction < 0 {
            1.0 - f64::from(row.q_raw)
        } else {
            f64::from(row.q_raw)
        }
    };
    let trusted = rows
        .iter()
        .filter(|row| adjusted_q(row) >= 0.5 && f64::from(row.reliability_raw) >= 0.5)
        .collect::<Vec<_>>();
    let ordinary = phase_a_error_bin(
        trusted.iter().filter(|row| !row.exact).count() as u64,
        trusted.len() as u64,
    );

    let labeled = rows
        .iter()
        .filter_map(|row| row.outcome.map(|outcome| (row, outcome)))
        .collect::<Vec<_>>();
    fit.goal_labeled_rows = labeled.len();
    let called_safe = labeled
        .iter()
        .filter(|(row, _)| f64::from(row.failed_raw) + f64::from(row.exhausted_raw) < 0.5)
        .collect::<Vec<_>>();
    let event_false_safe = phase_a_error_bin(
        called_safe
            .iter()
            .filter(|(_, outcome)| {
                matches!(outcome, PhaseAOutcome::Failed | PhaseAOutcome::Exhausted)
            })
            .count() as u64,
        called_safe.len() as u64,
    );
    let claimed = labeled
        .iter()
        .filter(|(row, _)| f64::from(row.satisfied_raw) >= 0.5)
        .collect::<Vec<_>>();
    let satisfaction = phase_a_error_bin(
        claimed
            .iter()
            .filter(|(_, outcome)| *outcome != PhaseAOutcome::Satisfied)
            .count() as u64,
        claimed.len() as u64,
    );

    let surprises = labeled
        .iter()
        .map(|(row, outcome)| {
            let likelihood = match outcome {
                PhaseAOutcome::Satisfied => f64::from(row.satisfied_raw),
                PhaseAOutcome::Failed => f64::from(row.failed_raw),
                PhaseAOutcome::Exhausted => f64::from(row.exhausted_raw),
                PhaseAOutcome::Ordinary => 1.0 - f64::from(row.satisfied_raw.max(row.failed_raw)),
            };
            -likelihood.clamp(eps, 1.0 - eps).ln()
        })
        .collect::<Vec<_>>();
    fit.tau_unknown_raw_p95 = phase_a_upper_quantile(&surprises, 0.95);
    let tau_unknown = fit
        .tau_unknown_raw_p95
        .map_or(0.5, |raw| raw.clamp(1e-3, 1.0 - 1e-3));
    let score_errors = labeled
        .iter()
        .map(|(row, outcome)| {
            let target = f64::from(*outcome == PhaseAOutcome::Satisfied);
            (f64::from(row.satisfied_raw) - target).abs()
        })
        .collect::<Vec<_>>();
    // No labeled row: the selection charge 2.0 rejects every finalist.
    let score_error_bound = phase_a_upper_quantile(&score_errors, 1.0 - eps).unwrap_or(1.0);

    calibration.q_direction = q_direction;
    calibration.tau_unknown = tau_unknown;
    calibration.score_error_bound = score_error_bound;
    calibration.ordinary = ordinary;
    calibration.event_false_safe = event_false_safe;
    calibration.satisfaction = satisfaction;
    calibration.ptrm = None;
    calibration.uncalibrated = false;
    calibration.fit = Some(fit);
    calibration
}

/// The bundle check `evaluate` applies before loading a checkpoint.
fn verify_phase_a_calibration_checkpoint(cfg: &EvalConfig, train_cfg: &TrainConfig) -> Result<()> {
    let Some(bundle) = cfg.checkpoint.parent() else {
        return Ok(());
    };
    let manifest = bundle.join("bundle-manifest.json");
    if manifest.is_file() {
        verify_checkpoint_bundle(bundle)?;
        let bundled_config = bundle.join("config.json");
        if file_sha256(&bundled_config)? != file_sha256(&cfg.train_config)? {
            bail!(
                "evaluation train config {} does not match checkpoint bundle {}",
                cfg.train_config.display(),
                bundled_config.display()
            );
        }
    } else if train_cfg.world_core_v4 {
        bail!(
            "Phase A calibration requires a verified checkpoint bundle manifest beside {}",
            cfg.checkpoint.display()
        );
    }
    Ok(())
}

/// `p2-eval --emit-phase-a-calibration <path>`: fit the ADR 0004 A3 record on
/// the synthetic `unseen_seed_7x7` held-out gate population (the same rows,
/// seed derivation and fingerprint as `v5_holdout_gates`), stamp it
/// `source = synthetic_holdout` with the emitter revision, the population
/// fingerprint and the checkpoint hash, validate it under
/// `PhaseACalibration::from_json`, and write it to `output`. No public or
/// recorded game is read.
pub fn emit_phase_a_calibration(cfg: &EvalConfig, output: &Path) -> Result<PhaseACalibration> {
    cfg.validate()?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    verify_phase_a_calibration_checkpoint(cfg, &train_cfg)?;
    if train_cfg.recipe != crate::p2::experiment::TrainingRecipe::FoundationV2 {
        bail!(
            "Phase A calibration is defined for foundation-v2 checkpoints (frozen heads, exact \
             decoder); got {:?}",
            train_cfg.recipe
        );
    }
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    let device = resolve_device(&cfg.device)?;
    let (model, _varmap) = timed_eval_phase("model_load", "checkpoint", || {
        load_model(&train_cfg, &cfg.checkpoint, &device)
    })?;
    let (gates, _strata, samples, _masks, _provenance) =
        timed_eval_phase("population_generation", "v5_holdout_gates", || {
            foundation_v2_v5_holdout_gates(&model, cfg, &train_cfg, &device, None)
        })?;
    let gate = gates
        .get(PHASE_A_CALIBRATION_POPULATION)
        .with_context(|| format!("held-out gate population {PHASE_A_CALIBRATION_POPULATION}"))?;
    let rows = timed_eval_phase("phase_a_calibration", "readouts", || {
        phase_a_calibration_rows(&model, &samples, &device)
    })?;
    let mut calibration = fit_phase_a_calibration(&rows);
    calibration.source = Some(SYNTHETIC_HOLDOUT_SOURCE.into());
    calibration.revision = crate::p2::arc3_live::live_run_provenance()
        .ok()
        .map(|provenance| {
            if provenance.git_dirty {
                format!("{}-dirty", provenance.git_revision)
            } else {
                provenance.git_revision
            }
        });
    calibration.population = Some(format!("v5_holdout_gates/{PHASE_A_CALIBRATION_POPULATION}"));
    calibration.population_fingerprint = Some(gate.population_fingerprint.clone());
    calibration.checkpoint_sha256 = Some(file_sha256(&cfg.checkpoint)?);
    let json = calibration.to_json()?;
    // Fail closed before anything reaches disk: the artifact must load and
    // validate under the same parser the live policies use. (Bitwise equality
    // is not asserted: serde_json's default float parsing may differ by one
    // ulp from the emitted value.)
    let reloaded = PhaseACalibration::from_json(&json)?;
    if reloaded.uncalibrated != calibration.uncalibrated
        || reloaded.source != calibration.source
        || reloaded.q_direction != calibration.q_direction
    {
        bail!("Phase A calibration artifact does not round-trip");
    }
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, json).with_context(|| format!("write {}", output.display()))?;
    Ok(calibration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::data::{ArcAction, ArcFrame, GoalFeatures};
    use crate::p2::train::{
        reinit_varmap_deterministic, save_checkpoint, TrainReport, TRAIN_REPORT_SCHEMA,
    };
    use rayon::ThreadPoolBuilder;

    fn serialized_eval_population_fixture() -> Result<Vec<u8>> {
        let episodes = 1;
        let seed = 0xE7A1_0005u64;
        let iid_seed = seed.wrapping_add(1);
        let factual = generate_factual_eval_population(seed, episodes)?;

        let mut dynamics = collect_synthetic_sources(
            seed,
            episodes,
            &["random_one_step", "exploration"],
            Split::HeldOutComposition,
        )?;
        let dynamics_rollout = flatten_sources(&dynamics);
        assert_eq!(
            dynamics_rollout,
            collect_dynamics_rollout_samples(seed, episodes, Split::HeldOutComposition)?
        );
        dynamics.push((
            "hazard_one_step".into(),
            collect_hazard_samples(seed, episodes, Split::HeldOutComposition)?,
        ));
        let planner = collect_synthetic_sources(
            seed,
            episodes,
            &[
                "sequential",
                "hypothesis_probe",
                "p1c_falsification",
                "p1c_hard_retarget",
            ],
            Split::HeldOutComposition,
        )?;
        let planner_rollout = collect_planner_rollout_samples(
            seed,
            episodes,
            Split::HeldOutComposition,
            Some(&planner[0].1),
        )?;
        assert_eq!(
            planner_rollout,
            collect_planner_rollout_samples(seed, episodes, Split::HeldOutComposition, None,)?
        );

        let mut iid_dynamics = collect_synthetic_sources(
            iid_seed,
            episodes,
            &["random_one_step", "exploration"],
            Split::Train,
        )?;
        let iid_dynamics_rollout = flatten_sources(&iid_dynamics);
        assert_eq!(
            iid_dynamics_rollout,
            collect_dynamics_rollout_samples(iid_seed, episodes, Split::Train)?
        );
        iid_dynamics.push((
            "hazard_one_step".into(),
            collect_hazard_samples(iid_seed, episodes, Split::Train)?,
        ));
        let iid_planner = collect_synthetic_sources(
            iid_seed,
            episodes,
            &[
                "sequential",
                "hypothesis_probe",
                "p1c_falsification",
                "p1c_hard_retarget",
            ],
            Split::Train,
        )?;
        let iid_planner_rollout = collect_planner_rollout_samples(
            iid_seed,
            episodes,
            Split::Train,
            Some(&iid_planner[0].1),
        )?;
        assert_eq!(
            iid_planner_rollout,
            collect_planner_rollout_samples(iid_seed, episodes, Split::Train, None)?
        );

        Ok(serde_json::to_vec(&(
            factual,
            dynamics,
            dynamics_rollout,
            planner,
            planner_rollout,
            iid_dynamics,
            iid_dynamics_rollout,
            iid_planner,
            iid_planner_rollout,
        ))?)
    }

    #[test]
    fn eval_population_bytes_are_identical_across_rayon_thread_counts() -> Result<()> {
        let serial = ThreadPoolBuilder::new().num_threads(1).build()?;
        let parallel = ThreadPoolBuilder::new().num_threads(4).build()?;
        let serial_started = Instant::now();
        let serial_bytes = serial.install(serialized_eval_population_fixture)?;
        let serial_elapsed = serial_started.elapsed();
        let parallel_started = Instant::now();
        let parallel_bytes = parallel.install(serialized_eval_population_fixture)?;
        let parallel_elapsed = parallel_started.elapsed();
        eprintln!(
            "eval population determinism fixture: one_thread_s={:.3} four_threads_s={:.3}",
            serial_elapsed.as_secs_f64(),
            parallel_elapsed.as_secs_f64()
        );
        assert_eq!(
            Sha256::digest(&serial_bytes),
            Sha256::digest(&parallel_bytes)
        );
        assert_eq!(serial_bytes, parallel_bytes);
        Ok(())
    }

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

    #[test]
    fn full_v4_eval_uses_exact_q_labels_and_omits_legacy_ptrm_oracles() -> Result<()> {
        let device = Device::Cpu;
        let mut train_cfg = TrainConfig::default();
        train_cfg.apply_full_v4_recipe();
        train_cfg.physical_batch = 2;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, train_cfg.seed)?;
        let samples = (0..2)
            .map(|episode| {
                generate_curriculum(
                    "random_one_step",
                    train_cfg.seed,
                    episode,
                    Split::HeldOutComposition,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .take(2)
            .collect::<Vec<_>>();
        let eval_cfg = EvalConfig {
            physical_batch: 2,
            synthetic_episodes: 2,
            ptrm_k: vec![1, 2, 4],
            q_mse_threshold: 999.0,
            ensemble_members: 1,
            ..EvalConfig::default()
        };
        let partial = eval_one_batch(&model, &samples, 0, &train_cfg, &eval_cfg, &device)?;
        assert_eq!(partial.q_acc.n, samples.len());
        assert_eq!(partial.q_labels.len(), samples.len());
        assert!(partial.ptrm_acc.is_empty());
        assert!(partial.matched_acc.is_empty());
        Ok(())
    }

    #[test]
    fn gate_support_profile_publishes_eval_bundle_and_scalars() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("tofy-p2-gate-eval-profile-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let device = Device::Cpu;
        let mut train_cfg = TrainConfig::default();
        train_cfg.apply_foundation_v2_recipe();
        train_cfg.hidden_dim = 8;
        train_cfg.action_dim = 4;
        train_cfg.inner_steps = 1;
        train_cfg.outer_steps = 1;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 32,
                seed: 0xE7A1_0006,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::UnseenSeed7x7,
        )?;
        let samples = mixed.transitions().cloned().collect::<Vec<_>>();
        let masks = mixed
            .samples()
            .iter()
            .map(|sample| sample.content_mask.clone())
            .collect::<Vec<_>>();
        let provenance = mixed
            .samples()
            .iter()
            .map(|sample| sample.provenance.clone())
            .collect::<Vec<_>>();
        let profile_dir = root.join("profile");
        ensure_eval_profile_campaign(
            &profile_dir,
            "eval-campaign.json",
            "tofy.p2.eval.test".into(),
            EVAL_PROFILE_ENTRYPOINT,
            vec![PlannedCapture {
                capture_step: 1,
                bundle: "eval-000000000001".into(),
            }],
        )?;
        let destination = profile_dir.join("eval-000000000001");
        let capture = RepresentativeUpdateCapture::begin_eval(EvalCaptureSpec {
            destination: &destination,
            capture_step: 1,
            entrypoint: EVAL_PROFILE_ENTRYPOINT,
            correlation_id: "tofy.p2/eval-000000000001".into(),
            device: "cpu",
            required_phases: &["encode", "forward", "decode", "metrics"],
            tags: &[("population", "test".into())],
        })?;
        evaluate_gate_support_impl(
            &model,
            &samples,
            Some(&masks),
            Some(&provenance),
            &device,
            None,
            Some(&capture),
        )?;
        capture.finish()?;

        candle_graph::verify_bundle(&destination)?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        assert_eq!(trace.run.phase, candle_graph::ExecutionPhase::Infer);
        assert_eq!(
            trace.run.tags.get("phase").map(String::as_str),
            Some("eval")
        );
        for phase in ["encode", "forward", "decode", "metrics"] {
            let label = format!("tofy.p2/eval-000000000001/{phase}");
            assert!(trace.spans.iter().any(|span| span.name == label));
        }
        assert!(trace
            .tensor_stats
            .iter()
            .any(|event| event.label == "eval/foreground_pixels"));
        let status = candle_graph::campaign_status(&profile_dir.join("eval-campaign.json"))?;
        assert_eq!(status.published, 1);
        assert_eq!(status.missing, 0);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn full_v4_factual_eval_scores_semantic_outcomes_not_only_action_ids() -> Result<()> {
        let device = Device::Cpu;
        let mut train_cfg = TrainConfig::default();
        train_cfg.apply_full_v4_recipe();
        train_cfg.hidden_dim = 16;
        train_cfg.action_dim = 4;
        train_cfg.inner_steps = 1;
        train_cfg.outer_steps = 1;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, train_cfg.seed)?;
        let eval_cfg = EvalConfig {
            seed: 17,
            synthetic_episodes: 1,
            physical_batch: 4,
            ..EvalConfig::default()
        };
        let factual_population =
            generate_factual_eval_population(eval_cfg.seed, eval_cfg.synthetic_episodes)?;
        let metrics =
            evaluate_factual_branches(&model, &eval_cfg, &device, factual_population.as_ref())?;
        assert_eq!(metrics.semantic_outcome_retrieval_n, metrics.branches);
        assert!(metrics.semantic_outcome_retrieval_accuracy.is_some());
        assert!(metrics.semantic_outcome_chance.is_some());
        assert!(metrics
            .semantic_factual_nll_mean
            .is_some_and(f64::is_finite));
        assert!(metrics
            .rows
            .iter()
            .all(|row| row.factual_outcome_nll.is_some()));
        Ok(())
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
            provenance: crate::p2::data::TransitionProvenance {
                content_width: 1,
                content_height: 1,
                content_x: 0,
                content_y: 0,
                source_kind: "action_diagnostic".into(),
                trajectory_id: format!("test/action_diagnostic/{episode_id}"),
                operator: None,
                rule_id: 0,
                level_index: 0,
                available_actions: 0,
                context_len: 0,
                background_color: 0,
            },
            oracle_latent: None,
            context: Vec::new(),
        })
    }

    /// Pre-Wave-22 board-probe projection, kept here only to prove the
    /// witness: it is blind to every field outside `(seed, episode_id,
    /// transition_index, family, action)`.
    fn legacy_board_probe_fnv_projection(samples: &[TransitionSample]) -> String {
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

    #[test]
    fn board_probe_population_identity_binds_every_row_field() -> Result<()> {
        use crate::p2::data::{ContextTransition, EpisodeOperator, OperatorFamily};

        let base = vec![
            action_diagnostic_sample(ArcAction::new(5, None, None)?, 0)?,
            action_diagnostic_sample(ArcAction::new(6, Some(0), Some(0))?, 1)?,
        ];
        let baseline = sample_population_fingerprint(&base);
        assert!(baseline.starts_with("sha256:"), "{baseline}");
        assert_eq!(sample_population_fingerprint(&base.clone()), baseline);
        assert_ne!(
            sample_population_fingerprint(&base),
            canonical_transition_population_sha256("other.domain", &base),
            "identity must be domain separated"
        );

        // Fields the legacy projection ignored but the probe (or the report's
        // meaning) depends on. Each mutation must change the new identity and
        // is shown to have left the legacy FNV projection unchanged.
        type SampleMutation = (
            &'static str,
            Box<dyn Fn(&mut TransitionSample) -> Result<()>>,
        );
        let omitted_mutations: Vec<SampleMutation> = vec![
            (
                "current frame",
                Box::new(|s| {
                    s.current = ArcFrame::new(1, 1, vec![9])?;
                    Ok(())
                }),
            ),
            (
                "next frame",
                Box::new(|s| {
                    s.next = ArcFrame::new(1, 1, vec![9])?;
                    Ok(())
                }),
            ),
            (
                "noop",
                Box::new(|s| {
                    s.noop = Some(true);
                    Ok(())
                }),
            ),
            (
                "noop unknown",
                Box::new(|s| {
                    s.noop = None;
                    Ok(())
                }),
            ),
            (
                "goal_satisfied",
                Box::new(|s| {
                    s.goal_satisfied = Some(true);
                    Ok(())
                }),
            ),
            (
                "goal_failed",
                Box::new(|s| {
                    s.goal_failed = Some(false);
                    Ok(())
                }),
            ),
            (
                "exhausted",
                Box::new(|s| {
                    s.exhausted = Some(true);
                    Ok(())
                }),
            ),
            (
                "split",
                Box::new(|s| {
                    s.split = Split::Train;
                    Ok(())
                }),
            ),
            (
                "goal_features",
                Box::new(|s| {
                    s.goal_features.values[0] = 1.0;
                    Ok(())
                }),
            ),
            (
                "content_width",
                Box::new(|s| {
                    s.provenance.content_width = 2;
                    Ok(())
                }),
            ),
            (
                "content_height",
                Box::new(|s| {
                    s.provenance.content_height = 2;
                    Ok(())
                }),
            ),
            (
                "content_x",
                Box::new(|s| {
                    s.provenance.content_x = 3;
                    Ok(())
                }),
            ),
            (
                "content_y",
                Box::new(|s| {
                    s.provenance.content_y = 3;
                    Ok(())
                }),
            ),
            (
                "source_kind",
                Box::new(|s| {
                    s.provenance.source_kind = "other".into();
                    Ok(())
                }),
            ),
            (
                "trajectory_id",
                Box::new(|s| {
                    s.provenance.trajectory_id = "other".into();
                    Ok(())
                }),
            ),
            (
                "operator",
                Box::new(|s| {
                    s.provenance.operator = Some(EpisodeOperator {
                        family: OperatorFamily::Paint,
                        agent_color: 1,
                        primary_color: 2,
                        secondary_color: 3,
                        empty_color: 0,
                    });
                    Ok(())
                }),
            ),
            (
                "rule_id",
                Box::new(|s| {
                    s.provenance.rule_id = 42;
                    Ok(())
                }),
            ),
            (
                "level_index",
                Box::new(|s| {
                    s.provenance.level_index = 1;
                    Ok(())
                }),
            ),
            (
                "available_actions",
                Box::new(|s| {
                    s.provenance.available_actions = 0b1111_1111;
                    Ok(())
                }),
            ),
            (
                "context_len",
                Box::new(|s| {
                    s.provenance.context_len = 1;
                    Ok(())
                }),
            ),
            (
                "background_color",
                Box::new(|s| {
                    s.provenance.background_color = 4;
                    Ok(())
                }),
            ),
            (
                "oracle_latent",
                Box::new(|s| {
                    s.oracle_latent = Some(vec![0.5]);
                    Ok(())
                }),
            ),
            (
                "context window",
                Box::new(|s| {
                    s.context.push(ContextTransition {
                        current: ArcFrame::new(1, 1, vec![1])?,
                        action: ArcAction::new(1, None, None)?,
                        next: ArcFrame::new(1, 1, vec![2])?,
                    });
                    Ok(())
                }),
            ),
        ];
        let legacy_baseline = legacy_board_probe_fnv_projection(&base);
        for (name, mutate) in &omitted_mutations {
            let mut mutated = base.clone();
            mutate(&mut mutated[1])?;
            assert_ne!(mutated, base, "{name}: mutation must change the row");
            assert_eq!(
                legacy_board_probe_fnv_projection(&mutated),
                legacy_baseline,
                "{name}: witness expects the legacy projection to be blind"
            );
            assert_ne!(
                sample_population_fingerprint(&mutated),
                baseline,
                "{name}: new identity must distinguish the populations"
            );
        }

        // Operator sub-fields must each matter, including `empty_color`.
        let mut with_operator = base.clone();
        omitted_mutations[15].1(&mut with_operator[1])?;
        let operator_baseline = sample_population_fingerprint(&with_operator);
        let mut empty_color = with_operator.clone();
        empty_color[1]
            .provenance
            .operator
            .as_mut()
            .unwrap()
            .empty_color = 7;
        assert_ne!(
            sample_population_fingerprint(&empty_color),
            operator_baseline
        );
        let mut family = with_operator.clone();
        family[1].provenance.operator.as_mut().unwrap().family = OperatorFamily::Toggle;
        assert_ne!(sample_population_fingerprint(&family), operator_baseline);

        // Keyed-field controls the legacy projection also caught.
        let keyed_mutations: Vec<SampleMutation> = vec![
            (
                "seed",
                Box::new(|s| {
                    s.seed += 1;
                    Ok(())
                }),
            ),
            (
                "episode_id",
                Box::new(|s| {
                    s.episode_id += 100;
                    Ok(())
                }),
            ),
            (
                "transition_index",
                Box::new(|s| {
                    s.transition_index += 1;
                    Ok(())
                }),
            ),
            (
                "family",
                Box::new(|s| {
                    s.family = "other".into();
                    Ok(())
                }),
            ),
            (
                "action id",
                Box::new(|s| {
                    s.action = ArcAction::new(2, None, None)?;
                    Ok(())
                }),
            ),
            (
                "action coords",
                Box::new(|s| {
                    s.action = ArcAction::new(6, Some(1), Some(0))?;
                    Ok(())
                }),
            ),
        ];
        for (name, mutate) in &keyed_mutations {
            let mut mutated = base.clone();
            mutate(&mut mutated[1])?;
            assert_ne!(sample_population_fingerprint(&mutated), baseline, "{name}");
        }

        // Row order and row count are part of the identity.
        let reversed = vec![base[1].clone(), base[0].clone()];
        assert_ne!(sample_population_fingerprint(&reversed), baseline);
        assert_ne!(sample_population_fingerprint(&base[..1]), baseline);
        assert_ne!(sample_population_fingerprint(&[]), baseline);
        Ok(())
    }

    #[test]
    fn shuffled_action_ratio_excludes_known_outcome_equivalent_rows() -> Result<()> {
        let samples = (0..32)
            .map(|episode_id| {
                action_diagnostic_sample(ArcAction::new(5, None, None)?, episode_id % 15)
            })
            .collect::<Result<Vec<_>>>()?;
        let shuffled = samples
            .iter()
            .map(|sample| {
                let mut sample = sample.clone();
                sample.action = ArcAction::new(6, Some(0), Some(0))?;
                Ok(sample)
            })
            .collect::<Result<Vec<_>>>()?;
        let predictions = samples
            .iter()
            .map(|sample| sample.next.pixels.to_vec())
            .collect::<Vec<_>>();
        let outcome_changing = vec![Some(false); samples.len()];

        assert_eq!(
            shuffled_action_changed_pixel_ratio(
                &samples,
                &shuffled,
                &outcome_changing,
                &predictions,
                &predictions,
            )?,
            None
        );
        Ok(())
    }

    #[test]
    fn perfect_counterfactual_oracle_has_a_low_shuffled_action_ratio() -> Result<()> {
        let samples = (0..MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS as u64)
            .map(|episode_id| {
                action_diagnostic_sample(ArcAction::new(5, None, None)?, episode_id % 15)
            })
            .collect::<Result<Vec<_>>>()?;
        let shuffled = samples
            .iter()
            .map(|sample| {
                let mut sample = sample.clone();
                sample.action = ArcAction::new(6, Some(0), Some(0))?;
                Ok(sample)
            })
            .collect::<Result<Vec<_>>>()?;
        let factual_predictions = samples
            .iter()
            .map(|sample| sample.next.pixels.to_vec())
            .collect::<Vec<_>>();
        let counterfactual_predictions = samples
            .iter()
            .map(|sample| sample.current.pixels.to_vec())
            .collect::<Vec<_>>();
        let outcome_changing = vec![Some(true); samples.len()];

        assert_eq!(
            shuffled_action_changed_pixel_ratio(
                &samples,
                &shuffled,
                &outcome_changing,
                &factual_predictions,
                &counterfactual_predictions,
            )?,
            Some(0.0)
        );
        Ok(())
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
        sample.provenance.trajectory_id = format!("test/{seed}/{episode_id}");
        sample.current = ArcFrame::new(1, 1, vec![(transition_index % 16) as u8])?;
        sample.next = ArcFrame::new(1, 1, vec![((transition_index + 1) % 16) as u8])?;
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
            open_semantic: None,
            closed_semantic: None,
            learned_copy_semantic: None,
        }
        .into_row(source)
    }

    fn semantic_decoder(pixel_accuracy: f64, pixels: usize) -> SemanticDecoderMetrics {
        SemanticDecoderMetrics {
            masks: BTreeMap::from([
                (
                    "changed".into(),
                    crate::p2::semantic_eval::SemanticMaskMetrics {
                        pixels,
                        transitions: 1,
                        mean_nll: Some(1.0 - pixel_accuracy),
                        pixel_accuracy: Some(pixel_accuracy),
                        exact_transition_accuracy: Some(if pixel_accuracy == 1.0 {
                            1.0
                        } else {
                            0.0
                        }),
                        mean_transition_accuracy: Some(pixel_accuracy),
                    },
                ),
                (
                    "unchanged_content".into(),
                    crate::p2::semantic_eval::SemanticMaskMetrics {
                        pixels: 20,
                        transitions: 1,
                        mean_nll: Some(0.1),
                        pixel_accuracy: Some(0.9),
                        exact_transition_accuracy: Some(0.0),
                        mean_transition_accuracy: Some(0.9),
                    },
                ),
                (
                    "unchanged_padding".into(),
                    crate::p2::semantic_eval::SemanticMaskMetrics {
                        pixels: 30,
                        transitions: 1,
                        mean_nll: Some(0.0),
                        pixel_accuracy: Some(1.0),
                        exact_transition_accuracy: Some(1.0),
                        mean_transition_accuracy: Some(1.0),
                    },
                ),
            ]),
            false_edit_rate: None,
            false_edit_transition_rate: None,
            padding_false_edit_rate: None,
            padding_false_edit_transition_rate: None,
        }
    }

    #[test]
    fn semantic_rollout_h4_h8_are_aggregated_into_split_report() {
        let mut h4_a = rollout_row("source", 1, 1, 4, Some(1.0), Some(1.0), Some(1.0), &[]);
        h4_a.open_semantic = Some(semantic_decoder(0.5, 10));
        let mut h4_b = rollout_row("source", 1, 2, 4, Some(1.0), Some(1.0), Some(1.0), &[]);
        h4_b.open_semantic = Some(semantic_decoder(1.0, 30));
        let mut h8 = rollout_row("source", 1, 1, 8, Some(1.0), Some(1.0), Some(1.0), &[]);
        h8.open_semantic = Some(semantic_decoder(0.25, 20));
        let mut split = empty_split("source", 3);
        attach_rollout_metrics(&mut split, &[h4_a, h4_b, h8], 9);
        let semantic = split.semantic_rollout.expect("semantic rollout summary");
        assert!(!semantic.comparable_to_one_step);
        assert_eq!(
            semantic.open[&4].masks["changed"].pixel_accuracy,
            Some(0.875)
        );
        assert_eq!(
            semantic.open[&4].masks["changed"].mean_transition_accuracy,
            Some(0.75)
        );
        assert_eq!(
            semantic.open[&8].masks["changed"].pixel_accuracy,
            Some(0.25)
        );
        assert!((semantic.open[&4].false_edit_rate.unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(semantic.open[&4].false_edit_transition_rate, Some(1.0));
        assert_eq!(semantic.open[&8].padding_false_edit_rate, Some(0.0));
        assert!(!semantic.open[&4].masks.contains_key("unchanged_content"));
        assert!(!semantic.open[&4].masks.contains_key("unchanged_padding"));
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

        let groups = group_rollouts(&samples)?;
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
            (open.n4, open.mse_4, open.n8, open.mse_8),
            (2, Some(4.0), 1, Some(10.0))
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
    fn copy_forward_bootstrap_seeds_preserve_reported_horizon_offsets() {
        let h4 = [1.0_f32, 3.0, 7.0, 13.0, 21.0];
        let h8 = [2.0_f32, 5.0, 11.0, 17.0, 23.0];
        let rows = h4
            .iter()
            .chain(h8.iter())
            .enumerate()
            .map(|(index, value)| {
                rollout_row(
                    "synthetic_dynamics",
                    1,
                    index as u64,
                    if index < h4.len() { 4 } else { 8 },
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
    }

    #[test]
    fn rollout_v16_schema_omits_h16_fields() {
        let metrics = rollout_metrics_from_rows(&[], RolloutMetric::Open, 7);
        let value = serde_json::to_value(metrics).expect("serialize rollout metrics");
        let object = value.as_object().expect("rollout metrics object");
        assert!(!object.contains_key("n16"));
        assert!(!object.contains_key("mse_16"));
        assert!(!object.contains_key("h16"));
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
        let metrics = summarize_changed_transitions(&[0.99, 0.79, 0.59], &[1.0, 0.8, 0.6], 17)?;
        assert_eq!(metrics.n, 3);
        assert!(metrics
            .improvement_fraction
            .is_some_and(|value| value > 0.0 && value < 0.10));
        assert_eq!(metrics.positive_improvement_pass, Some(true));
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
    fn outcome_counterfactual_margin_algebra_distinguishes_concordant_and_swapped() {
        let left_prediction = [1.0f32, 0.0];
        let right_prediction = [0.0f32, 1.0];
        let left_target = [1.0f32, 0.0];
        let right_target = [0.0f32, 1.0];
        let (_, _, concordant) = outcome_counterfactual_margin(
            &left_prediction,
            &right_prediction,
            &left_target,
            &right_target,
        );
        let (_, _, swapped) = outcome_counterfactual_margin(
            &right_prediction,
            &left_prediction,
            &left_target,
            &right_target,
        );
        assert_eq!(concordant, 1.0);
        assert_eq!(swapped, -1.0);
    }

    #[test]
    fn outcome_counterfactual_enumeration_keeps_populations_and_equivalence_distinct() -> Result<()>
    {
        let seed = factual_branch_eval_seed(19);
        let movement = generate_factual_branch_group(seed, 0, Split::HeldOutComposition)?;
        let coordinate = generate_factual_branch_group(seed, 1, Split::HeldOutComposition)?;
        assert_eq!(movement.branches().len(), FACTUAL_BRANCHES_PER_GROUP);
        assert_eq!(coordinate.branches().len(), FACTUAL_BRANCHES_PER_GROUP);
        assert!(movement.branches()[0]
            .transition
            .family
            .starts_with("factual_branch"));
        assert_eq!(
            coordinate.branches()[0].transition.family,
            "factual_coordinate_branch"
        );
        let coordinate_classes = canonical_outcome_classes(&coordinate);
        let coordinate_outcomes = coordinate_classes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len();
        assert!(coordinate_outcomes > 1);
        assert!(coordinate_outcomes <= FACTUAL_BRANCHES_PER_GROUP);
        assert_eq!(
            (0..FACTUAL_BRANCHES_PER_GROUP)
                .flat_map(|left| left + 1..FACTUAL_BRANCHES_PER_GROUP)
                .count(),
            FACTUAL_BRANCHES_PER_GROUP * (FACTUAL_BRANCHES_PER_GROUP - 1) / 2
        );
        Ok(())
    }

    #[test]
    fn outcome_counterfactual_negative_controls_are_exactly_neutral() {
        let prediction = [0.25f32, -0.5];
        let left_target = [1.0f32, 0.0];
        let right_target = [0.0f32, 1.0];
        let (_, _, masked) =
            outcome_counterfactual_margin(&prediction, &prediction, &left_target, &right_target);
        let identity = [0.0f32, 0.0];
        let (_, _, identity_margin) =
            outcome_counterfactual_margin(&identity, &identity, &left_target, &right_target);
        let (_, _, equivalent) =
            outcome_counterfactual_margin(&left_target, &right_target, &left_target, &left_target);
        assert_eq!(masked, 0.0);
        assert_eq!(identity_margin, 0.0);
        assert_eq!(equivalent, 0.0);
    }

    #[test]
    fn outcome_counterfactual_bootstrap_is_deterministic_and_whole_group() {
        let groups = vec![(0, -0.5, 2), (1, 0.5, 4), (2, 1.0, 1)];
        let first = outcome_interval_from_group_means(&groups, 7).expect("interval");
        let second = outcome_interval_from_group_means(&groups, 7).expect("interval");
        assert_eq!(first.estimate, 1.0 / 3.0);
        assert_eq!(first.lower_95, second.lower_95);
        assert_eq!(first.upper_98_75, second.upper_98_75);
        assert_eq!(first.groups, 3);
        assert_eq!(first.pairs, 7);
        assert_eq!(first.resamples, 10_000);
        assert_eq!(first.unit, "whole_branch_group");
    }

    #[test]
    fn outcome_content_fingerprint_excludes_seed_and_episode_provenance() -> Result<()> {
        let group = generate_factual_branch_group(
            factual_branch_eval_seed(23),
            1,
            Split::HeldOutComposition,
        )?;
        let mut changed_json = serde_json::to_value(&group)?;
        for branch in changed_json["branches"]
            .as_array_mut()
            .expect("serialized branches")
        {
            branch["transition"]["seed"] = serde_json::json!(u64::MAX);
            branch["transition"]["episode_id"] = serde_json::json!(999_999u64);
        }
        let changed: BranchGroup = serde_json::from_value(changed_json)?;
        let original_classes = canonical_outcome_classes(&group);
        let changed_classes = canonical_outcome_classes(&changed);
        assert_eq!(
            outcome_group_content_sha256(&group, &original_classes),
            outcome_group_content_sha256(&changed, &changed_classes)
        );
        assert_eq!(
            frame_content_sha256(&group.branches()[0].transition.current),
            frame_content_sha256(&changed.branches()[0].transition.current)
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
            foundation_v2: None,
            research_claim: false,
        };
        save_checkpoint(&varmap, &train_cfg, &report)?;

        let eval_cfg = EvalConfig {
            checkpoint: dir.join("model.safetensors"),
            train_config: dir.join("config.json"),
            seed: 3,
            iid_seed: 4,
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
            profile_eval: false,
            context_ablation: false,
            adaptation_falsifier: false,
            adaptation_falsifier_min_level_transitions: ADAPT_MIN_LEVEL_TRANSITIONS,
        };
        let eval = evaluate(&eval_cfg)?;
        assert_eq!(eval.schema, EVAL_REPORT_SCHEMA);
        assert!(eval.v6_context_ablation.is_none());
        assert!(eval.v6_adaptation_falsifier.is_none());
        let eval_json = fs::read_to_string(dir.join("eval.json"))?;
        assert!(
            !eval_json.contains("v6_context_ablation"),
            "legacy report must not gain the ablation key"
        );
        assert!(
            !eval_json.contains("v6_adaptation_falsifier"),
            "legacy report must not gain the falsifier key"
        );
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
        assert!(factual.action6_coordinate_n > 0);
        assert!(factual.action6_coordinate_n <= factual.action6);
        assert!(factual.action6_coordinate_rmse_normalized.is_some());
        assert!(factual.action6_coordinate_rmse_pixels.is_some());
        assert!(factual
            .action_controllability
            .as_ref()
            .is_some_and(|metrics| metrics.states_with_action_pairs == factual.groups
                && metrics.action_pairs > 0));
        assert!(!factual.population_fingerprint.is_empty());
        assert!(factual.board_probe.is_some());
        let counterfactuals = eval
            .outcome_counterfactuals
            .as_ref()
            .expect("full evaluation outcome counterfactuals");
        assert_eq!(counterfactuals.groups, eval_cfg.synthetic_episodes * 4);
        assert_eq!(
            counterfactuals.movement_groups,
            counterfactuals.coordinate_groups
        );
        assert_eq!(
            counterfactuals.unordered_pairs,
            counterfactuals.groups * FACTUAL_BRANCHES_PER_GROUP * (FACTUAL_BRANCHES_PER_GROUP - 1)
                / 2
        );
        assert_eq!(
            counterfactuals.eligible_pairs + counterfactuals.outcome_equivalent_pairs,
            counterfactuals.unordered_pairs
        );
        assert!(counterfactuals.ledger_reconciled);
        assert_eq!(
            counterfactuals.overall.as_ref().map(|row| row.resamples),
            Some(10_000)
        );
        assert!(counterfactuals.controls.pixel_oracle_exactly_one);
        assert!(counterfactuals.controls.latent_oracle_at_least_0_99);
        assert!(
            counterfactuals
                .controls
                .swapped_oracle_at_most_negative_0_99
        );
        assert!(counterfactuals.controls.action_masked_max_abs_at_most_1e_6);
        assert!(counterfactuals.controls.identity_max_abs_at_most_1e_6);
        let state_scrambled = &counterfactuals
            .controls
            .state_scrambled_same_action_template;
        if state_scrambled.available {
            assert!(state_scrambled.estimate.is_some());
            assert_eq!(state_scrambled.groups, counterfactuals.movement_groups);
            assert!(state_scrambled.pairs > 0);
        } else {
            assert!(state_scrambled
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("identical action template")));
        }
        assert!(counterfactuals.pair_ledger.iter().all(|row| {
            row.group.actions.len() == FACTUAL_BRANCHES_PER_GROUP
                && row.group.next_sha256.len() == FACTUAL_BRANCHES_PER_GROUP
                && row.group.content_fingerprint.starts_with("sha256:")
        }));
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
            back.outcome_counterfactuals
                .as_ref()
                .map(|metrics| &metrics.population_fingerprint),
            Some(&counterfactuals.population_fingerprint)
        );
        assert_eq!(
            back.factual_branches
                .as_ref()
                .map(|metrics| &metrics.population_fingerprint),
            Some(&factual.population_fingerprint)
        );
        let mut compatible_value = serde_json::to_value(&eval)?;
        compatible_value
            .as_object_mut()
            .expect("serialized report object")
            .remove("outcome_counterfactuals");
        let compatible: EvalReport = serde_json::from_value(compatible_value)?;
        assert!(compatible.outcome_counterfactuals.is_none());
        let episode_rows: Vec<EpisodeRolloutRow> =
            fs::read_to_string(eval_cfg.episode_jsonl.as_ref().expect("episode JSONL path"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        assert!(!episode_rows.is_empty());
        assert!(episode_rows.iter().all(|row| {
            row.schema == "p2.episode_rollout.v3"
                && matches!(row.horizon, 4 | 8)
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
        assert!(representation_eval.outcome_counterfactuals.is_none());
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
        assert!(rollout_eval.outcome_counterfactuals.is_none());

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
        let empty_counterfactuals = arc_eval
            .outcome_counterfactuals
            .as_ref()
            .expect("full mode keeps the outcome counterfactual block shape");
        assert_eq!(empty_counterfactuals.groups, 0);
        assert!(empty_counterfactuals.ledger_reconciled);
        let _: EvalReport = serde_json::from_str(&fs::read_to_string(&arc_cfg.output)?)?;

        let legacy_dir = dir.join("legacy");
        let legacy_cfg = TrainConfig {
            output_dir: legacy_dir.clone(),
            lessons: vec!["dynamics".into()],
            world_core_v2: false,
            branch_learning: crate::p2::branch_learning::BranchLearningConfig::default(),
            ..train_cfg.clone()
        };
        let legacy_varmap = VarMap::new();
        let legacy_vb = VarBuilder::from_varmap(&legacy_varmap, DType::F32, &device);
        let _legacy_model = WorldModel::new(legacy_cfg.model_config(), legacy_vb)?;
        reinit_varmap_deterministic(&legacy_varmap, legacy_cfg.seed)?;
        let mut legacy_report = report.clone();
        legacy_report.world_core_schema = "legacy_p2_eval_compatible".into();
        legacy_report.experiment = legacy_cfg.resolved_experiment()?;
        legacy_report.latest_checkpoint = legacy_dir.join("checkpoints/step-000000000000");
        legacy_report.checkpoint = legacy_dir.join("model.safetensors");
        legacy_report.config_path = legacy_dir.join("config.json");
        save_checkpoint(&legacy_varmap, &legacy_cfg, &legacy_report)?;
        let legacy_eval = evaluate(&EvalConfig {
            checkpoint: legacy_dir.join("model.safetensors"),
            train_config: legacy_dir.join("config.json"),
            synthetic_episodes: 1,
            ensemble_members: 0,
            episode_jsonl: None,
            output: legacy_dir.join("eval.json"),
            ..eval_cfg.clone()
        })?;
        assert!(legacy_eval.factual_branches.is_none());
        let legacy_counterfactuals = legacy_eval
            .outcome_counterfactuals
            .expect("legacy checkpoint full counterfactual evaluation");
        assert_eq!(legacy_counterfactuals.groups, 4);
        assert_eq!(
            legacy_counterfactuals.pair_ledger.len(),
            legacy_counterfactuals.groups
                * FACTUAL_BRANCHES_PER_GROUP
                * (FACTUAL_BRANCHES_PER_GROUP - 1)
                / 2
        );
        assert!(legacy_counterfactuals.ledger_reconciled);

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

    #[test]
    fn context_len_stratum_boundaries_follow_adr_0005() {
        assert_eq!(context_len_stratum(0), "0");
        assert_eq!(context_len_stratum(1), "1-4");
        assert_eq!(context_len_stratum(4), "1-4");
        assert_eq!(context_len_stratum(5), "5-16");
        assert_eq!(context_len_stratum(16), "5-16");
    }

    /// Chunking the held-out gate forward bounds peak memory (a v6 row carries
    /// up to 16 context frame pairs, so a 512-row forward exceeds a training
    /// step's envelope). Every op on the path is per-row, so chunking is
    /// mathematically transparent; it is not bit-identical, because conv
    /// kernels sum in a different order at a different batch size. The
    /// observed drift is ~1e-6 on latents of order 1, the same class of
    /// nondeterminism the repository already tolerates across rayon thread
    /// counts, and it can in principle flip an exact-decode argmax tie.
    #[test]
    fn chunked_gate_forward_matches_the_unchunked_forward() -> Result<()> {
        let device = Device::Cpu;
        for v6 in [false, true] {
            let mut train_cfg = TrainConfig::default();
            train_cfg.world_core_v6 = v6;
            train_cfg.apply_foundation_v2_recipe();
            train_cfg.hidden_dim = 8;
            train_cfg.action_dim = 4;
            train_cfg.inner_steps = 1;
            train_cfg.outer_steps = 1;
            train_cfg.data_contract_v6 = v6;
            let varmap = VarMap::new();
            let model = WorldModel::new(
                train_cfg.model_config(),
                VarBuilder::from_varmap(&varmap, DType::F32, &device),
            )?;
            let mixed = compose_mixed_stream_batch(
                &MixedStreamConfig {
                    batch_size: 32,
                    seed: 0xC4_0000_5A1E,
                    schedule: if v6 {
                        adaptation_v6_stream_schedule
                    } else {
                        foundation_v2_stream_schedule
                    },
                    data_contract_v6: v6,
                    ..MixedStreamConfig::default()
                },
                1.0,
                0,
                V5DataSplit::UnseenSeed7x7,
            )?;
            let samples = mixed.transitions().cloned().collect::<Vec<_>>();
            let provenance = mixed
                .samples()
                .iter()
                .map(|sample| sample.provenance.clone())
                .collect::<Vec<_>>();
            let encoded =
                encode_gate_support_population(&model, &samples, Some(&provenance), &device)?;
            assert_eq!(encoded.context.is_some(), v6, "v6 rows carry context");
            let depth = RecursionDepth::from_config(model.config());
            let forward = |chunk: usize| -> Result<Vec<f32>> {
                Ok(forward_gate_rows_chunked(
                    &model,
                    &encoded.current,
                    &encoded.batch.actions,
                    &encoded.batch.action_coords,
                    &encoded.batch.goals,
                    &encoded.batch.operator_conditioning,
                    encoded.context.as_ref(),
                    depth,
                    chunk,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?)
            };
            let whole = forward(samples.len())?;
            let chunked = forward(5)?;
            assert_eq!(whole.len(), chunked.len());
            let worst = whole
                .iter()
                .zip(&chunked)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("v6={v6} worst chunk delta {worst:e}");
            assert!(
                worst <= 1e-5,
                "chunked gate forward changed the prediction by {worst:e}"
            );
        }
        Ok(())
    }

    /// ADR 0005 §5.1: the ablation scores the same rows twice (full context,
    /// `K = 0`), reports per-stratum deltas, and is exactly zero for a model
    /// whose context FiLM is at its identity initialization.
    #[test]
    fn context_ablation_scores_the_same_rows_with_and_without_context() -> Result<()> {
        let device = Device::Cpu;
        let mut train_cfg = TrainConfig::default();
        train_cfg.apply_foundation_v2_recipe();
        train_cfg.hidden_dim = 8;
        train_cfg.action_dim = 4;
        train_cfg.inner_steps = 1;
        train_cfg.outer_steps = 1;
        train_cfg.world_core_v6 = true;
        train_cfg.data_contract_v6 = true;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let mixed = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 32,
                seed: 0xE7A1_0051,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::UnseenSeed7x7,
        )?;
        let samples = mixed.transitions().cloned().collect::<Vec<_>>();
        let masks = mixed
            .samples()
            .iter()
            .map(|sample| sample.content_mask.clone())
            .collect::<Vec<_>>();
        let provenance = mixed
            .samples()
            .iter()
            .map(|sample| sample.provenance.clone())
            .collect::<Vec<_>>();
        let with_context = samples
            .iter()
            .filter(|sample| !sample.context.is_empty())
            .count();
        assert!(
            with_context > 0,
            "the v6 data contract must yield LearningHistories rows with context"
        );

        let report = evaluate_context_ablation(
            &model,
            "unseen_seed_7x7",
            &samples,
            &masks,
            &provenance,
            &device,
        )?;
        assert_eq!(report.overall.rows, samples.len());
        assert_eq!(report.overall.rows_with_context, with_context);
        assert_eq!(report.evidence_class, "selection_only");
        let gates = evaluate_gate_support_impl(
            &model,
            &samples,
            Some(&masks),
            Some(&provenance),
            &device,
            None,
            None,
        )?;
        assert_eq!(report.population_fingerprint, gates.population_fingerprint);
        assert_eq!(
            report.overall.changed_exact_with_context,
            gates.one_step_changed_exact
        );
        // Identity FiLM: context cannot change the prediction, so both passes agree.
        assert_eq!(report.overall.delta, Some(0.0));
        assert_eq!(report.overall.composed_delta, Some(0.0));
        let strata = report
            .per_stratum
            .iter()
            .map(|stratum| stratum.stratum.as_str())
            .collect::<Vec<_>>();
        assert!(strata.contains(&"0"), "{strata:?}");
        assert!(
            strata.iter().any(|name| *name != "0"),
            "rows with context must populate a non-zero stratum: {strata:?}"
        );
        assert_eq!(
            report
                .per_stratum
                .iter()
                .map(|stratum| stratum.metrics.rows)
                .sum::<usize>(),
            samples.len()
        );
        for stratum in &report.per_stratum {
            assert_eq!(
                stratum.metrics.rows_with_context,
                if stratum.stratum == "0" {
                    0
                } else {
                    stratum.metrics.rows
                }
            );
        }
        // The masked rows are the same rows with an empty window.
        let (masked, masked_provenance) = context_masked_rows(&samples, &provenance);
        assert!(masked.iter().all(|row| row.context.is_empty()));
        assert!(masked
            .iter()
            .zip(&masked_provenance)
            .all(|(row, provenance)| provenance.source == row.provenance));
        assert!(masked
            .iter()
            .zip(&samples)
            .all(|(masked, original)| masked.current == original.current
                && masked.next == original.next
                && masked.action == original.action));
        // The report key is serde-flattened and round-trips.
        let json = serde_json::to_value(&report)?;
        assert!(json.get("rows").is_some() && json.get("per_stratum").is_some());
        let back: ContextAblationReport = serde_json::from_value(json)?;
        assert_eq!(back, report);
        Ok(())
    }

    // ---- ADR 0005 §5.2 adaptation falsifier ---------------------------------

    fn tiny_v6_model(device: &Device) -> Result<(WorldModel, VarMap)> {
        let mut train_cfg = TrainConfig::default();
        train_cfg.apply_foundation_v2_recipe();
        train_cfg.hidden_dim = 8;
        train_cfg.action_dim = 4;
        train_cfg.inner_steps = 1;
        train_cfg.outer_steps = 1;
        train_cfg.world_core_v6 = true;
        train_cfg.data_contract_v6 = true;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, device),
        )?;
        reinit_varmap_deterministic(&varmap, 0xE3)?;
        Ok((model, varmap))
    }

    fn var_snapshots(varmap: &VarMap) -> Result<Vec<(String, Vec<u32>)>> {
        let data = varmap.data().lock().unwrap();
        let mut out = Vec::new();
        for (name, var) in data.iter() {
            let bits = var
                .as_tensor()
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            out.push((name.clone(), bits));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn falsifier_metrics(
        arm: &str,
        prefix_len: usize,
        changed: usize,
        changed_exact: Option<f64>,
        composed_changed_exact: Option<f64>,
    ) -> AdaptationFalsifierArmMetrics {
        AdaptationFalsifierArmMetrics {
            arm: arm.into(),
            prefix_len,
            episodes: 1,
            rows: changed,
            changed_transitions: changed,
            changed_exact,
            composed_changed_exact,
            adaptation: AdaptationFalsifierTelemetry::default(),
        }
    }

    /// Test (1): at the identity context FiLM, Channel A's scoring path (the
    /// gate forward with the episode window) equals the plain context-free
    /// forward bit-for-bit, and its pooled metric equals the gate metric.
    #[test]
    fn adaptation_falsifier_context_only_arm_matches_plain_forward_at_identity_film() -> Result<()>
    {
        let device = Device::Cpu;
        let (model, varmap) = tiny_v6_model(&device)?;
        crate::p2::model::zero_context_film_projections(&varmap)?;
        let stream_cfg = adaptation_falsifier_stream_config(ADAPTATION_FALSIFIER_POPULATION_SEED);
        let history = augmented_learning_history(&stream_cfg, V5DataSplit::UnseenSeed7x7, 0)?;
        let len = history.chronological.len();
        assert!(len >= 14 && len % 7 == 0, "{len}");
        let t = 8;
        let window = ADAPTATION_FALSIFIER_SCORE_WINDOW;
        let rows = adaptation_falsifier_scoring_rows(&history, t, window);
        assert_eq!(rows.len(), window);
        for (offset, row) in rows.iter().enumerate() {
            let position = t + offset;
            assert_eq!(row.context.len(), position.min(CONTEXT_WINDOW_MAX));
            assert_eq!(usize::from(row.provenance.context_len), row.context.len());
            let previous = &history.chronological_row(position - 1).transition;
            let last = row.context.last().expect("window");
            assert_eq!(last.current, previous.current);
            assert_eq!(last.next, previous.next);
            assert_eq!(
                row.provenance.available_actions,
                crate::p2::data::ALL_ACTIONS_AVAILABLE,
                "v6 contract row"
            );
            assert!(
                row.provenance.operator.is_none(),
                "v6 rows condition as UNKNOWN"
            );
        }

        let with_context = adaptation_falsifier_decode_rows(&model, &rows, &device)?;
        // Plain forward: the same rows through the model without any context.
        let batch = batch_from_samples(&rows, &device)?;
        let (current, _) = model.encode_state_pair(&batch.frames, &batch.next_frames)?;
        let plain = model
            .forward_from_latent_with_depth_and_operator_conditioning_with_context(
                &current,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
                None,
                RecursionDepth::from_config(model.config()),
            )?
            .y;
        let plain_true = exact_palette_predictions(&model, &plain)?;
        let plain_composed = model
            .composed_gameplay_decode(&plain, &batch.frames)?
            .reshape((rows.len(), ()))?
            .to_dtype(DType::U8)?
            .to_vec2::<u8>()?;
        assert!(batch.context.is_some(), "the scoring rows carry a window");
        assert_eq!(with_context.true_predictions, plain_true);
        assert_eq!(with_context.composed, plain_composed);
        assert!(with_context
            .true_predictions
            .iter()
            .all(|row| row.len() == FRAME_SIDE * FRAME_SIDE));

        // Pooled metric == the gate metric on the same rows.
        let gates = evaluate_gate_support_impl(&model, &rows, None, None, &device, None, None)?;
        let spec = AdaptationFalsifierSpec {
            episodes: 1,
            prefix_lengths: vec![t],
            ..AdaptationFalsifierSpec::preregistered(1, ADAPT_MIN_LEVEL_TRANSITIONS)
        };
        let accums = adaptation_falsifier_arm(&model, &device, &[history], &spec, None)?;
        let metrics = accums[&t].metrics(ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY, t);
        assert_eq!(metrics.rows, window);
        assert_eq!(metrics.episodes, 1);
        assert_eq!(metrics.changed_exact, gates.one_step_changed_exact);
        assert_eq!(
            metrics.composed_changed_exact,
            gates.one_step_composed_changed_exact
        );
        assert_eq!(metrics.adaptation, AdaptationFalsifierTelemetry::default());
        Ok(())
    }

    /// Test (2): Channel B moves only the fast subset during an episode,
    /// `restore_prior` returns every parameter to theta_0 bitwise, and the
    /// full falsifier leaves the VarMap untouched.
    #[test]
    fn adaptation_falsifier_adapts_fast_weights_and_restores_the_prior_bitwise() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_v6_model(&device)?;
        let before = var_snapshots(&varmap)?;
        let stream_cfg = adaptation_falsifier_stream_config(ADAPTATION_FALSIFIER_POPULATION_SEED);
        let mut history = augmented_learning_history(&stream_cfg, V5DataSplit::UnseenSeed7x7, 0)?;
        // Debug-mode adaptation is slow; eight chronological transitions
        // (level 0 complete plus the first of level 1) cover t = 4 and its
        // window, with the warm-up lowered to two transitions.
        history.chronological.truncate(8);
        let t = 4;
        let spec = AdaptationFalsifierSpec {
            episodes: 1,
            prefix_lengths: vec![t],
            min_level_transitions: 2,
            ..AdaptationFalsifierSpec::preregistered(1, 2)
        };

        // One carry episode by hand, inspected before the restore.
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Carry,
            ContextScopeKind::Game,
        )?;
        adapter.set_min_level_transitions(spec.min_level_transitions);
        adapter.begin_game()?;
        adapter.reseed_reservoir(spec.adapter_seed);
        let mut accums = BTreeMap::from([(t, AdaptationFalsifierAccum::default())]);
        adaptation_falsifier_episode(
            &model,
            &device,
            &history,
            &spec,
            Some(&mut adapter),
            &mut accums,
        )?;
        assert!(accums[&t].telemetry.updates > 0, "{:?}", accums[&t]);
        assert!(accums[&t].telemetry.max_drift_from_prior > 0.0);
        assert!(!adapter.fast_weights_equal_prior()?);
        let during = var_snapshots(&varmap)?;
        let mut fast_changed = 0usize;
        for ((name, was), (_, now)) in before.iter().zip(&during) {
            if crate::p2::model::is_fast_weight(name) {
                fast_changed += usize::from(was != now);
            } else {
                assert_eq!(was, now, "frozen parameter {name} changed under adaptation");
            }
        }
        assert!(fast_changed > 0, "adaptation moved no fast weight");
        adapter.restore_prior()?;
        assert!(adapter.fast_weights_equal_prior()?);
        drop(adapter);
        assert_eq!(var_snapshots(&varmap)?, before, "restore_prior is bitwise");

        // The whole falsifier: three arms, prior restored, report round-trips.
        let histories = [history];
        let report = evaluate_adaptation_falsifier_on(&model, &varmap, &device, &spec, &histories)?;
        assert_eq!(
            var_snapshots(&varmap)?,
            before,
            "falsifier left the VarMap changed"
        );
        assert_eq!(report.episodes, 1);
        assert_eq!(report.min_level_transitions, 2);
        assert_eq!(report.population_seed, ADAPTATION_FALSIFIER_POPULATION_SEED);
        assert_eq!(report.adapter_seed, ADAPTATION_FALSIFIER_ADAPTER_SEED);
        assert_eq!(report.evidence_class, "selection_only");
        assert!(report.population_fingerprint.starts_with("sha256:"));
        assert_eq!(report.chronological_lengths.values().sum::<usize>(), 1);
        let arm = |name: &str| {
            report
                .arms
                .iter()
                .find(|metrics| metrics.arm == name && metrics.prefix_len == t)
                .unwrap_or_else(|| panic!("{name} arm at t={t}"))
        };
        let context_only = arm(ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY);
        let reset = arm(ADAPTATION_FALSIFIER_ARM_RESET);
        let carry = arm(ADAPTATION_FALSIFIER_ARM_CARRY);
        assert_eq!(context_only.rows, ADAPTATION_FALSIFIER_SCORE_WINDOW);
        assert_eq!(reset.rows, context_only.rows);
        assert_eq!(carry.rows, context_only.rows);
        assert_eq!(context_only.adaptation.updates, 0);
        assert!(carry.adaptation.updates > 0, "{carry:?}");
        assert!(carry.adaptation.max_drift_from_prior > 0.0);
        // Position 4 is inside level 0: no level boundary has passed, so the
        // reset and carry arms saw the same observations, the same reservoir
        // seed and the same updates, and must coincide exactly.
        assert_eq!(reset.adaptation, carry.adaptation, "{reset:?}");
        assert_eq!(reset.changed_exact, carry.changed_exact);
        assert_eq!(reset.composed_changed_exact, carry.composed_changed_exact);
        assert_eq!(
            report.verdict.arms.len(),
            2,
            "one verdict per adapted arm: {:?}",
            report.verdict
        );
        let json = serde_json::to_string(&report)?;
        let back: AdaptationFalsifierReport = serde_json::from_str(&json)?;
        assert_eq!(back, report);

        // Determinism: the same spec reproduces the same report bit for bit.
        let again = evaluate_adaptation_falsifier_on(&model, &varmap, &device, &spec, &histories)?;
        assert_eq!(again, report);
        assert_eq!(var_snapshots(&varmap)?, before);
        Ok(())
    }

    /// The default §6.2 warm-up (8 unique transitions per level) can never be
    /// met on 7-transition synthetic levels: both adapted arms stay in warm-up
    /// and reproduce the context-only arm exactly. The report makes that
    /// visible through the telemetry rather than hiding it.
    #[test]
    fn adaptation_falsifier_default_warmup_never_updates_on_synthetic_levels() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_v6_model(&device)?;
        let spec = AdaptationFalsifierSpec {
            episodes: 1,
            prefix_lengths: vec![8],
            ..AdaptationFalsifierSpec::preregistered(1, ADAPT_MIN_LEVEL_TRANSITIONS)
        };
        let stream_cfg = adaptation_falsifier_stream_config(spec.population_seed);
        let mut history = augmented_learning_history(&stream_cfg, V5DataSplit::UnseenSeed7x7, 1)?;
        history.chronological.truncate(12);
        let report = evaluate_adaptation_falsifier_on(&model, &varmap, &device, &spec, &[history])?;
        let context_only = &report.arms[0];
        assert_eq!(context_only.arm, ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY);
        for metrics in &report.arms[1..] {
            assert_eq!(metrics.adaptation.updates, 0, "{metrics:?}");
            assert!(metrics.adaptation.warmup_calls > 0, "{metrics:?}");
            assert_eq!(metrics.adaptation.max_drift_from_prior, 0.0);
            assert_eq!(metrics.changed_exact, context_only.changed_exact);
            assert_eq!(
                metrics.composed_changed_exact,
                context_only.composed_changed_exact
            );
        }
        assert!(!report.verdict.promote_channel_b);
        Ok(())
    }

    /// Test (3): the preregistered rule on hand-built metric tables.
    #[test]
    fn adaptation_falsifier_verdict_applies_the_preregistered_rule() {
        let a = ADAPTATION_FALSIFIER_ARM_CONTEXT_ONLY;
        let reset = ADAPTATION_FALSIFIER_ARM_RESET;
        let carry = ADAPTATION_FALSIFIER_ARM_CARRY;
        let prior = |t: usize, ce: f64, cce: f64| falsifier_metrics(a, t, 100, Some(ce), Some(cce));
        let baseline = vec![
            prior(8, 0.50, 0.55),
            prior(16, 0.40, 0.45),
            prior(32, 0.70, 0.75),
        ];

        // Pass: carry improves by >= 0.02 at every t (0.02 exactly at t=32
        // up to rounding) and is never worse; reset equals the prior.
        let mut pass = baseline.clone();
        pass.extend([
            falsifier_metrics(reset, 8, 100, Some(0.50), Some(0.55)),
            falsifier_metrics(reset, 16, 100, Some(0.40), Some(0.45)),
            falsifier_metrics(reset, 32, 100, Some(0.70), Some(0.75)),
            falsifier_metrics(carry, 8, 100, Some(0.53), Some(0.56)),
            falsifier_metrics(carry, 16, 100, Some(0.43), Some(0.46)),
            falsifier_metrics(carry, 32, 100, Some(0.72), Some(0.75)),
        ]);
        let verdict = adaptation_falsifier_verdict(&pass, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
        assert!(verdict.promote_channel_b, "{verdict:?}");
        assert_eq!(verdict.satisfied_by, vec![carry.to_string()]);
        assert_eq!(verdict.evaluated_prefix_lengths, vec![8, 16, 32]);
        assert!(verdict.skipped_prefix_lengths.is_empty());
        let reset_verdict = &verdict.arms[0];
        assert_eq!(reset_verdict.arm, reset);
        assert!(!reset_verdict.promote && reset_verdict.not_worse_at_every_t);
        assert!(!reset_verdict.improvement_at_every_t);
        let carry_verdict = &verdict.arms[1];
        assert!(carry_verdict.promote);
        assert_eq!(carry_verdict.deltas.len(), 3);
        assert!(carry_verdict
            .deltas
            .iter()
            .all(|d| d.improvement && d.not_worse));

        // Fail on one t: carry improves by only 0.01 at t=16.
        let mut one_t = pass.clone();
        one_t
            .iter_mut()
            .find(|m| m.arm == carry && m.prefix_len == 16)
            .unwrap()
            .changed_exact = Some(0.41);
        let verdict = adaptation_falsifier_verdict(&one_t, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
        assert!(!verdict.promote_channel_b, "{verdict:?}");
        assert!(verdict.satisfied_by.is_empty());
        let carry_verdict = &verdict.arms[1];
        assert!(!carry_verdict.improvement_at_every_t && carry_verdict.not_worse_at_every_t);
        assert!(!carry_verdict.deltas[1].improvement);
        assert!(carry_verdict.deltas[0].improvement && carry_verdict.deltas[2].improvement);

        // Fail on frozen-worse: changed-exact improves everywhere, but the
        // adapted-then-frozen composed decode is worse than the prior at t=8.
        let mut frozen_worse = pass.clone();
        frozen_worse
            .iter_mut()
            .find(|m| m.arm == carry && m.prefix_len == 8)
            .unwrap()
            .composed_changed_exact = Some(0.54);
        let verdict =
            adaptation_falsifier_verdict(&frozen_worse, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
        assert!(!verdict.promote_channel_b, "{verdict:?}");
        let carry_verdict = &verdict.arms[1];
        assert!(carry_verdict.improvement_at_every_t);
        assert!(!carry_verdict.not_worse_at_every_t);
        assert!(!carry_verdict.deltas[0].not_worse);
        assert_eq!(
            carry_verdict.deltas[0].composed_changed_exact_delta,
            Some(0.54 - 0.55)
        );

        // A prefix length with no changed transition in the prior arm is
        // skipped (t=32 is structurally empty on 2-4 level histories), and the
        // rule is applied over the remaining ones only.
        let mut skipped = pass.clone();
        for metrics in skipped.iter_mut().filter(|m| m.prefix_len == 32) {
            metrics.changed_transitions = 0;
            metrics.rows = 0;
            metrics.changed_exact = None;
            metrics.composed_changed_exact = None;
        }
        let verdict = adaptation_falsifier_verdict(&skipped, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
        assert_eq!(verdict.evaluated_prefix_lengths, vec![8, 16]);
        assert_eq!(verdict.skipped_prefix_lengths, vec![32]);
        assert!(verdict.promote_channel_b);
        assert!(verdict.note.contains("[32]"));

        // Nothing evaluable: no promotion.
        let verdict = adaptation_falsifier_verdict(
            &[
                falsifier_metrics(a, 8, 0, None, None),
                falsifier_metrics(carry, 8, 0, None, None),
            ],
            ADAPTATION_FALSIFIER_PROMOTION_DELTA,
        );
        assert!(!verdict.promote_channel_b);
        assert!(verdict.evaluated_prefix_lengths.is_empty());
        assert!(!verdict.arms[0].promote);

        // A missing adapted-arm entry at an evaluated t fails that arm.
        let mut missing = pass.clone();
        missing.retain(|m| !(m.arm == carry && m.prefix_len == 16));
        let verdict = adaptation_falsifier_verdict(&missing, ADAPTATION_FALSIFIER_PROMOTION_DELTA);
        assert!(!verdict.promote_channel_b);
        assert!(!verdict.arms[1].deltas[1].improvement);
        assert!(verdict.arms[1].deltas[1].changed_exact_delta.is_none());
    }

    fn calibration_row(
        q: f32,
        reliability: f32,
        exact: bool,
        satisfied: f32,
        failed: f32,
        outcome: Option<PhaseAOutcome>,
    ) -> PhaseACalibrationRow {
        PhaseACalibrationRow {
            q_raw: q,
            reliability_raw: reliability,
            exact,
            noop_raw: 0.1,
            satisfied_raw: satisfied,
            failed_raw: failed,
            exhausted_raw: 0.0,
            outcome,
        }
    }

    /// The fitted record's bins are the observed error rates of the regions
    /// the gates trust, with Clopper-Pearson upper endpoints and honest
    /// support; direction, tau and the score bound follow the documented rules.
    #[test]
    fn fit_phase_a_calibration_bins_direction_tau_and_score_bound() -> Result<()> {
        let empty = fit_phase_a_calibration(&[]);
        assert!(empty.uncalibrated);
        assert!(empty.ordinary.is_none());
        assert!(PhaseACalibration::from_json(&empty.to_json()?).is_ok());

        let mut rows = Vec::new();
        // 100 trusted rows (q, reliability >= 0.5), 10 of them not exact.
        for i in 0..100 {
            rows.push(calibration_row(
                0.9,
                0.8,
                i >= 10,
                if i % 2 == 0 { 0.9 } else { 0.1 },
                0.05,
                Some(if i % 2 == 0 {
                    PhaseAOutcome::Satisfied
                } else {
                    PhaseAOutcome::Ordinary
                }),
            ));
        }
        // 50 untrusted rows (low reliability) that are exact: excluded from
        // the ordinary bin, unlabeled (dropped goal).
        for _ in 0..50 {
            rows.push(calibration_row(0.9, 0.2, true, 0.5, 0.5, None));
        }
        // 20 rows the event head calls safe but which failed.
        for _ in 0..20 {
            rows.push(calibration_row(
                0.1,
                0.9,
                false,
                0.1,
                0.2,
                Some(PhaseAOutcome::Failed),
            ));
        }
        let record = fit_phase_a_calibration(&rows);
        assert_eq!(record.q_direction, 1);
        let ordinary = record.ordinary.as_ref().expect("ordinary bin");
        assert_eq!(ordinary.support, 100);
        assert_eq!(
            ordinary.upper_error_bound_95,
            clopper_pearson_upper(10, 100).unwrap()
        );
        let false_safe = record.event_false_safe.as_ref().expect("false-safe bin");
        assert_eq!(false_safe.support, 120, "labeled rows called safe");
        assert_eq!(
            false_safe.upper_error_bound_95,
            clopper_pearson_upper(20, 120).unwrap()
        );
        let satisfaction = record.satisfaction.as_ref().expect("satisfaction bin");
        assert_eq!(satisfaction.support, 50);
        assert_eq!(
            satisfaction.upper_error_bound_95,
            clopper_pearson_upper(0, 50).unwrap()
        );
        assert!(record.ptrm.is_none());
        assert!(!record.uncalibrated);
        assert!(record.tau_unknown > 0.0 && record.tau_unknown < 1.0);
        let fit = record.fit.as_ref().expect("fit metadata");
        assert_eq!(fit.rows, 170);
        assert_eq!(fit.goal_labeled_rows, 120);
        // The failed rows are the most surprising (likelihood 0.2 -> 1.61).
        assert!((fit.tau_unknown_raw_p95.unwrap() - (-f64::from(0.2f32).ln())).abs() < 1e-9);
        assert_eq!(record.tau_unknown, 1.0 - 1e-3, "clamped into (0, 1)");
        // Score errors: 0.1 (120 rows) and the 20 failed rows at 0.1; the 98%
        // quantile is 0.1.
        assert!((record.score_error_bound - 0.1).abs() < 1e-6);
        assert!(!fit.unfitted.is_empty());

        // Anti-correlated q flips the direction and the trusted region.
        let flipped = rows
            .iter()
            .map(|row| PhaseACalibrationRow {
                q_raw: 1.0 - row.q_raw,
                ..row.clone()
            })
            .collect::<Vec<_>>();
        let record = fit_phase_a_calibration(&flipped);
        assert_eq!(record.q_direction, -1);
        assert_eq!(record.ordinary.as_ref().unwrap().support, 100);
        // The record validates under the live parser after stamping.
        let mut stamped = record.clone();
        stamped.source = Some(SYNTHETIC_HOLDOUT_SOURCE.into());
        let parsed = PhaseACalibration::from_json(&stamped.to_json()?)?;
        assert_eq!(parsed.q_direction, -1);
        assert_eq!(parsed.source.as_deref(), Some(SYNTHETIC_HOLDOUT_SOURCE));
        assert!((parsed.score_error_bound - stamped.score_error_bound).abs() < 1e-9);
        Ok(())
    }

    /// The readout pass scores every held-out row of a v6 population with
    /// probabilities in range, exact outcomes and goal labels, and the fitted
    /// record validates under the live parser.
    #[test]
    fn phase_a_calibration_rows_score_the_held_out_population() -> Result<()> {
        let device = Device::Cpu;
        let (model, _varmap) = tiny_v6_model(&device)?;
        let batch = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 12,
                seed: 0x5645_4C35_1D00_0007,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::UnseenSeed7x7,
        )?;
        let samples = batch.transitions().cloned().collect::<Vec<_>>();
        let rows = phase_a_calibration_rows(&model, &samples, &device)?;
        assert_eq!(rows.len(), samples.len());
        for (row, sample) in rows.iter().zip(&samples) {
            for value in [
                row.q_raw,
                row.reliability_raw,
                row.noop_raw,
                row.satisfied_raw,
                row.failed_raw,
                row.exhausted_raw,
            ] {
                assert!((0.0..=1.0).contains(&value), "{value}");
            }
            assert_eq!(row.outcome, phase_a_outcome(sample));
        }
        assert!(phase_a_calibration_rows(&model, &[], &device)?.is_empty());
        let mut record = fit_phase_a_calibration(&rows);
        record.source = Some(SYNTHETIC_HOLDOUT_SOURCE.into());
        record.population_fingerprint = Some("sha256:test".into());
        let json = record.to_json()?;
        // serde_json's default float parsing may differ by one ulp, so the
        // round-trip is compared field-wise with a tolerance.
        let parsed = PhaseACalibration::from_json(&json)?;
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9;
        assert_eq!(parsed.q_direction, record.q_direction);
        assert!(close(parsed.tau_unknown, record.tau_unknown));
        assert!(close(parsed.score_error_bound, record.score_error_bound));
        for (a, b) in [
            (&parsed.ordinary, &record.ordinary),
            (&parsed.event_false_safe, &record.event_false_safe),
            (&parsed.satisfaction, &record.satisfaction),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.support, b.support);
                    assert!(close(a.upper_error_bound_95, b.upper_error_bound_95));
                }
                (None, None) => {}
                _ => panic!("bin presence differs"),
            }
        }
        assert_eq!(parsed.source, record.source);
        assert_eq!(parsed.population_fingerprint, record.population_fingerprint);
        assert_eq!(parsed.fit.as_ref().unwrap().rows, samples.len());
        assert_eq!(record.fit.as_ref().unwrap().rows, samples.len());

        // Outcome labeling: dropped goals and masked labels yield `None`.
        let mut sample = samples[0].clone();
        sample.goal_features.values = [0.0; GOAL_FEATURES_DIM];
        sample.goal_features.values[0] = 1.0;
        sample.goal_satisfied = Some(false);
        sample.goal_failed = Some(false);
        assert_eq!(phase_a_outcome(&sample), Some(PhaseAOutcome::Ordinary));
        sample.goal_failed = Some(true);
        assert_eq!(phase_a_outcome(&sample), Some(PhaseAOutcome::Failed));
        sample.goal_satisfied = Some(true);
        assert_eq!(phase_a_outcome(&sample), Some(PhaseAOutcome::Satisfied));
        sample.goal_satisfied = None;
        sample.goal_failed = None;
        sample.exhausted = Some(true);
        assert_eq!(phase_a_outcome(&sample), Some(PhaseAOutcome::Exhausted));
        sample.exhausted = None;
        assert_eq!(phase_a_outcome(&sample), None);
        sample.goal_satisfied = Some(true);
        sample.goal_features.values = [0.0; GOAL_FEATURES_DIM];
        assert_eq!(phase_a_outcome(&sample), None, "dropped goal");
        assert_eq!(phase_a_upper_quantile(&[], 0.95), None);
        assert_eq!(phase_a_upper_quantile(&[3.0, 1.0, 2.0], 0.5), Some(2.0));
        assert_eq!(phase_a_upper_quantile(&[3.0, 1.0, 2.0], 0.95), Some(3.0));
        Ok(())
    }
}
