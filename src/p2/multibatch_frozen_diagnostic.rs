//! Read-only frozen-checkpoint diagnosis selected by registered multi-batch G.
//!
//! The complete contract is frozen in
//! `docs/research/2026-09-06-v6-multibatch-frozen-diagnostic-prereg.md`.
//! This module never takes an optimizer or EMA step and never writes a model.

use crate::gpu_lock::TrainPidGuard;
use crate::p2::bf16_falsifier::write_json_report;
use crate::p2::context_wiring::{
    external_manifest_paths, file_sha256_hex, identity_frame_sha256, open_diagnostic_device,
    open_run_root, query_gpu_identity, registered_provenance_guard, same_build_identity,
    seal_run_root, unix_seconds, verify_manifest, verify_manifest_sidecar, GpuIdentity,
    LifecycleRecord, FAILED_EVIDENCE_CLASS, LIFECYCLE_COMPLETE, LIFECYCLE_FAILED,
    LIFECYCLE_RUNNING, REPORT_FILE, RUN_CLASS_PREFLIGHT, RUN_CLASS_REGISTERED,
};
use crate::p2::data::{MixedStreamBatch, TransitionSample, V5Sample, FRAME_SIDE};
use crate::p2::eval::{raw_one_step_logits, raw_one_step_logits_with_chunk};
use crate::p2::evidence::{launch_provenance, LaunchProvenance};
use crate::p2::model::{WorldModel, PALETTE_SIZE};
use crate::p2::multibatch_screen::{
    action6_coordinate_metrics, bind_screen_report, exact_metrics, group_routing_metrics,
    BatchUnion, ExactMetrics, MultibatchPopulation, MultibatchScreenReport, PopulationCensus,
    ScreenSnapshot, ScreenUpdateRecord, OUTCOME_DOES_NOT_SCALE, REGISTERED_BATCH_SIZE,
    REGISTERED_SEED, SNAPSHOT_STEPS, TRAIN_BATCHES,
};
use crate::p2::muon::{hybrid_newton_schulz, matrix_view, muon_shape_rescale, uses_muon};
use crate::p2::optimizer::accumulate_parameter_gradients;
use crate::p2::positive_control::{
    ensure_operator_projection_zero, ensure_registered_config, route_norms, EvidenceBinding,
    RouteNorms, REGISTERED_ROLLOUT_FRAGMENTS, REGISTERED_SIGREG_KNOTS,
    REGISTERED_SIGREG_PROJECTIONS,
};
use crate::p2::train::{
    batch_from_foundation_v2_host, batch_from_samples, event_slot_weight_tensor,
    foundation_v2_dedicated_rollout_loss, foundation_v2_gradient_route_stats,
    foundation_v2_loss_values, foundation_v2_training_loss_with_event_weights,
    gradient_cosine_for_optimizer_route, gradient_l2_for_optimizer_route, load_train_config,
    load_varmap_exact, prepare_foundation_v2_batch_host, retain_parameter_gradients,
    sync_cuda_device, BatchTensors, FoundationV2GradientRouteStats, FoundationV2LossMeans,
    FoundationV2ObjectiveConfig, OptimizerRoute, PreparedFoundationV2BatchHost, TrainConfig,
};
use anyhow::{ensure, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub const FROZEN_DIAGNOSTIC_SCHEMA: &str = "p2.v6_multibatch_frozen_diagnostic.v1";
pub const DIAGNOSTIC_EVIDENCE_CLASS: &str =
    "selection_only_single_seed_frozen_checkpoint_diagnostic";
const COMMAND_TAG: &str = "p2-v6-multibatch-frozen-diagnostic";
const PREFLIGHT_MAX_WALL: Duration = Duration::from_secs(5 * 60);
const REGISTERED_MAX_WALL: Duration = Duration::from_secs(20 * 60);
const ADMISSION_DEVICE_SECONDS: f64 = 900.0;
const PRIMARY_POPULATION: &str = "primary_same_forward";
const LEGACY_POPULATION: &str = "legacy_g_chunk32";
const NORM_EPSILON: f64 = 1e-6;
const RECONSTRUCTION_REL_TOLERANCE: f64 = 1e-5;
const NEGLIGIBLE_COSINE: f64 = 0.95;
const CONFLICT_COSINE: f64 = 0.80;
const NEGLIGIBLE_AUX_SHARE: f64 = 0.10;
const CONFLICT_AUX_SHARE: f64 = 0.30;
const NEGLIGIBLE_KAPPA: f64 = -0.10;
const CONFLICT_KAPPA: f64 = -0.25;
const PRED_BLIND_SHARE: f64 = 0.05;
const INTERNAL_CONFLICT_COSINE: f64 = -0.25;
const FALSE_EDIT_AUX_CONFLICT_COSINE: f64 = -0.25;
const ATTRACTOR_PERSISTENCE: f64 = 0.60;
const ATTRACTOR_MARGIN: f64 = 1.0;
const SEAM_CHARACTERIZATION_SCHEMA: &str = "p2.v6_frozen_seam_characterization.v1";
const SEAM_CHARACTERIZATION_EVIDENCE_CLASS: &str =
    "characterization_only_implementation_integrity_diagnostic";
const SEAM_CHARACTERIZATION_RUN_CLASS: &str = "registered_characterization";
const SEAM_CHARACTERIZATION_TAG: &str = "p2-v6-frozen-seam-characterization";
const SEAM_CHARACTERIZATION_MAX_WALL: Duration = Duration::from_secs(120);
const FAILED_PREFLIGHT_REPORT_SHA256: &str =
    "54cc1ddf07aca7e6f8ffd72cb3f216d4d1b079899f995610f807c00fbeb3cb74";
const FAILED_PREFLIGHT_MANIFEST_SHA256: &str =
    "41705d4a9103702600523647475205301fea9383df0011b98a854e7a53b33dbb";
const FAILED_PREFLIGHT_IDENTITY: &str =
    "sha256:ce0d4b5a75488459e01790aee59bc2b7f5994bd17e6c1162c92b6a14c082228e";
const FAILED_PREFLIGHT_SOURCE: &str = "7907731b6fa69043089c55ddb92c573a79b2f29d";
const FAILED_PREFLIGHT_BINARY_SHA256: &str =
    "sha256:22781d436df998f98213390e8770cffdbcea3f8ae53ca9080278469d8f4d3691";
const REGISTERED_G_REPORT_SHA256: &str =
    "03f645a5cccfbd4dcf72bf9927ac15589dc8fa579c6330f254101ed70c18789f";
const REGISTERED_G_MANIFEST_SHA256: &str =
    "900488b44e0f1623513234839f0f777d2c1d052ec3c46458fb714383868748d4";
const REGISTERED_G_IDENTITY: &str =
    "sha256:1d470fc1b5680e33efa07e32c51bf74fa0a73bea6e839828da09fd7922eee265";
const REGISTERED_G_SOURCE: &str = "dba110de8ed467b58dbaa2a936565f0dc8a7b794";
const REGISTERED_G_BINARY_SHA256: &str =
    "sha256:faf6fb74820582de8f1a62675392de19a729a9961a2573fc8bf8c96358631f44";
const REGISTERED_G_CARGO_LOCK_SHA256: &str =
    "b3d7c2e65ee49f07e5fb8c0ba5d3e183bb839c9f0117ef5e7ff820d80bc367cc";
const REGISTERED_TRAIN_CONFIG_SHA256: &str =
    "874d53e53e68cfb5dbaada83bf25b5558f2874ae23f3af62997e13ec1263f3c1";
const REGISTERED_RAW_SHA256: [&str; 7] = [
    "0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79",
    "7c62689c8938a9e351468a8b56e53bb9314cbe84d351a9d948f0cb457d57a3da",
    "72d4b26e00205469b1bf1f9c31eaaf2dbec4f3e61a01b929c947df3b0d473e40",
    "8ad2790b3d85db4c6d3d4485ded3ddb34645dc242e8f00d64fa92538f0f8dd66",
    "d27eb12ad58f9b9788780f713c7cc009e816e09f92ddabf7ba3428182a9beef5",
    "0738ba97ae70d74d7e425b5c18c460d5a032bab740aa39f033d6a0335218c787",
    "07ef8d1b3b79d99fad61c3fb9ae4de193ff2a2fc77ba8a4a1e5b33bcbdb70b1a",
];
const GRADIENT_STEPS: [usize; 3] = [1024, 1536, 2048];
const ANATOMY_STEPS: [usize; 3] = GRADIENT_STEPS;
const POSITIVE_PREFIXES: [&str; 9] = [
    "block.",
    "exact_grounding_head.decoder.",
    "action_emb.",
    "action_proj.",
    "action_film_gamma.weight",
    "action_film_beta.weight",
    "spatial_action_proj.weight",
    "operator_conditioning_proj.weight",
    "encoder.",
];
const ZERO_PREFIXES: [&str; 3] = ["coord_proj.", "grounding_head.decoder.", "prefix_head."];

#[derive(Debug, Clone, Args)]
pub struct P2MultibatchFrozenDiagnosticArgs {
    /// Sealed registered G report.
    #[arg(long)]
    pub g_report: PathBuf,

    #[arg(long, default_value = "cuda")]
    pub device: String,

    #[arg(long)]
    pub output_root: PathBuf,

    #[arg(long, default_value_t = false)]
    pub registered: bool,

    /// Sealed same-binary diagnostic preflight; required only when registered.
    #[arg(long)]
    pub preflight_report: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct P2FrozenSeamCharacterizationArgs {
    /// Sealed registered G report.
    #[arg(long)]
    pub g_report: PathBuf,

    /// Exact sealed failed preflight that selected this characterization.
    #[arg(long)]
    pub failed_preflight_report: PathBuf,

    #[arg(long, default_value = "cuda")]
    pub device: String,

    #[arg(long)]
    pub output_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorFingerprint {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInputComparison {
    pub training: TensorFingerprint,
    pub evaluator: TensorFingerprint,
    pub exact_equal: bool,
    pub active_for_predicted_latent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputComparisonReport {
    pub fields: BTreeMap<String, TensorInputComparison>,
    pub active_inputs_equal: bool,
    pub operator_conditioning_equal: bool,
    pub accepted_inert_operator_non_identity: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericSummary {
    pub count: usize,
    pub zero_count: usize,
    pub min: Option<f64>,
    pub median: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitVariantReport {
    pub name: String,
    pub logits: TensorFingerprint,
    pub seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitComparisonReport {
    pub left: String,
    pub right: String,
    pub bit_identical: bool,
    pub argmax_disagreement_pixels: usize,
    pub argmax_disagreement_rows: usize,
    pub absolute_logit_delta: NumericSummary,
    pub disagreement_reference_margin: NumericSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeamCharacterizationTiming {
    pub population_seconds: f64,
    pub input_comparison_seconds: f64,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeamCharacterizationReport {
    pub schema: String,
    pub evidence_class: String,
    pub run_class: String,
    pub registered: bool,
    pub research_claim: bool,
    pub public_data_read: bool,
    pub no_backward_calls: bool,
    pub no_optimizer_step: bool,
    pub no_ema_update: bool,
    pub no_checkpoint_write: bool,
    pub lifecycle: LifecycleRecord,
    pub provenance: LaunchProvenance,
    pub command: Vec<String>,
    pub device: String,
    pub device_is_cuda: bool,
    pub gpu_identity: Option<GpuIdentity>,
    pub output_root: PathBuf,
    pub g: Option<EvidenceBinding>,
    pub failed_preflight: Option<EvidenceBinding>,
    pub train_config_sha256: String,
    pub cargo_lock_sha256: String,
    pub population_census_sha256: String,
    pub population: Option<PopulationCensus>,
    pub checkpoint_sha256: String,
    pub inputs: Option<InputComparisonReport>,
    pub variants: BTreeMap<String, LogitVariantReport>,
    pub comparisons: BTreeMap<String, LogitComparisonReport>,
    pub self_repeat_unstable: bool,
    pub active_input_preparation_differs: bool,
    pub execution_shape_argmax_flip: bool,
    pub same_shape_train_eval_argmax_flip: bool,
    pub numeric_shape_drift_without_argmax: bool,
    pub numeric_same_shape_drift_without_argmax: bool,
    pub branch: Option<String>,
    pub timing: SeamCharacterizationTiming,
    pub identity_root: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenDiagnosticSpec {
    pub gradient_steps: Vec<usize>,
    pub anatomy_steps: Vec<usize>,
    pub rescore_steps: Vec<usize>,
    pub train_batch_positions: Vec<usize>,
    pub heldout_batch_positions: Vec<usize>,
    pub no_optimizer_step: bool,
    pub no_ema_update: bool,
    pub max_wall_seconds: u64,
    pub admission_device_seconds: u64,
}

impl FrozenDiagnosticSpec {
    fn new(registered: bool) -> Self {
        Self {
            gradient_steps: if registered {
                GRADIENT_STEPS.to_vec()
            } else {
                vec![0, 1024]
            },
            anatomy_steps: if registered {
                ANATOMY_STEPS.to_vec()
            } else {
                vec![2048]
            },
            rescore_steps: if registered {
                SNAPSHOT_STEPS.to_vec()
            } else {
                vec![0, 1024, 2048]
            },
            train_batch_positions: if registered {
                (0..TRAIN_BATCHES).collect()
            } else {
                vec![0]
            },
            heldout_batch_positions: if registered {
                (0..TRAIN_BATCHES).collect()
            } else {
                vec![0]
            },
            no_optimizer_step: true,
            no_ema_update: true,
            max_wall_seconds: if registered {
                REGISTERED_MAX_WALL.as_secs()
            } else {
                PREFLIGHT_MAX_WALL.as_secs()
            },
            admission_device_seconds: ADMISSION_DEVICE_SECONDS as u64,
        }
    }

    fn validate(&self, registered: bool) -> Result<()> {
        ensure!(*self == Self::new(registered), "diagnostic spec drift");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefixGradientNorm {
    pub state: String,
    pub l2: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteGeometry {
    pub l2: f64,
    pub share_of_full: Option<f64>,
    pub cosine_to_prediction: Option<f64>,
    pub kappa_to_prediction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedGeometry {
    pub global: RouteGeometry,
    pub adamw: RouteGeometry,
    pub muon: RouteGeometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientComponentReport {
    pub objective: String,
    pub weight: f64,
    pub routes: RoutedGeometry,
    pub prefixes: BTreeMap<String, PrefixGradientNorm>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionCheck {
    pub global_relative_or_absolute_residual: f64,
    pub adamw_relative_or_absolute_residual: f64,
    pub muon_relative_or_absolute_residual: f64,
    #[serde(default)]
    pub global_reference_l2: f64,
    #[serde(default)]
    pub global_absolute_residual_l2: f64,
    #[serde(default)]
    pub adamw_reference_l2: f64,
    #[serde(default)]
    pub adamw_absolute_residual_l2: f64,
    #[serde(default)]
    pub muon_reference_l2: f64,
    #[serde(default)]
    pub muon_absolute_residual_l2: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossBinding {
    pub expected_step: u64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientCellReport {
    pub snapshot_step: usize,
    pub batch_position: usize,
    pub sigreg_seed: u64,
    pub ep_weight: f64,
    pub false_edit_pixels: usize,
    #[serde(default)]
    pub mask_population: String,
    #[serde(default)]
    pub prediction_logits_fingerprint: Option<TensorFingerprint>,
    #[serde(default)]
    pub false_edit_mask_sha256: String,
    #[serde(default)]
    pub mask_binding: Option<SameForwardMaskBinding>,
    pub training_raw_argmax_disagreement_pixels: usize,
    pub losses: FoundationV2LossMeans,
    pub components: Vec<GradientComponentReport>,
    pub prediction: RoutedGeometry,
    pub auxiliary: RoutedGeometry,
    pub combined: RoutedGeometry,
    pub full_reconstruction: ReconstructionCheck,
    pub prediction_reconstruction: ReconstructionCheck,
    pub ns_muon_cosine_full_to_prediction: Option<f64>,
    pub false_edit_to_auxiliary_cosine: Option<f64>,
    pub false_edit_to_prediction_cosine: Option<f64>,
    pub changed_to_unchanged_cosine: Option<f64>,
    pub false_edit_share_of_prediction: Option<f64>,
    pub loss_binding: Option<LossBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SameForwardMaskBinding {
    pub logits: TensorFingerprint,
    pub mask_sha256: String,
    pub false_edit_pixels: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticLogitComparison {
    pub snapshot_step: usize,
    pub batch_position: usize,
    pub split: String,
    pub comparison: String,
    pub left_logits: TensorFingerprint,
    pub right_logits: TensorFingerprint,
    pub result: LogitComparisonReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerBatchScore {
    #[serde(default)]
    pub population: String,
    pub batch_position: usize,
    pub batch_index: u64,
    pub raw: ExactMetrics,
    pub action_routed: bool,
    pub raw_full_exact_branches: usize,
    pub reproduced_distinct_changed_classes: usize,
    pub action6_changed_full_exact: usize,
    pub action6_coordinate_routed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginSummary {
    pub count: usize,
    pub bins: BTreeMap<String, usize>,
    pub min: Option<f64>,
    pub median: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalseEditAnatomy {
    #[serde(default)]
    pub population: String,
    pub false_edit_pixels: usize,
    pub margins: MarginSummary,
    pub locations: BTreeMap<String, usize>,
    pub class_pairs: BTreeMap<String, usize>,
    pub predicted_class_from_changed_region: usize,
    pub distance_to_changed: BTreeMap<String, usize>,
    pub row_histogram_changed_rows: BTreeMap<String, usize>,
    pub row_histogram_no_change_rows: BTreeMap<String, usize>,
    pub changed_exact_near_miss: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotRescore {
    pub step: usize,
    #[serde(default)]
    pub legacy_population: String,
    #[serde(default)]
    pub primary_population: String,
    #[serde(default)]
    pub anatomy_population: String,
    pub train: Vec<PerBatchScore>,
    pub heldout: Vec<PerBatchScore>,
    #[serde(default)]
    pub primary_train: Vec<PerBatchScore>,
    #[serde(default)]
    pub primary_heldout: Vec<PerBatchScore>,
    pub train_anatomy: Option<FalseEditAnatomy>,
    pub heldout_anatomy: Option<FalseEditAnatomy>,
    pub train_anatomy_by_batch: Vec<FalseEditAnatomy>,
    pub heldout_anatomy_by_batch: Vec<FalseEditAnatomy>,
    pub complete_union_binding: bool,
    pub seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistenceReport {
    #[serde(default)]
    pub population: String,
    pub split: String,
    pub false_edits_1024: usize,
    pub false_edits_1536: usize,
    pub false_edits_2048: usize,
    pub intersection_all_three: usize,
    pub union_all_three: usize,
    pub persistent_1536_to_2048: usize,
    pub persistence_1536_to_2048: Option<f64>,
    pub final_present_all_three_fraction: Option<f64>,
    pub new_1024_to_1536: usize,
    pub resolved_1024_to_1536: usize,
    pub new_1536_to_2048: usize,
    pub resolved_1536_to_2048: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEstimate {
    pub d1_cell_seconds: f64,
    pub d2d3_pair_seconds: f64,
    pub estimated_device_seconds: f64,
    pub admitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticVerdict {
    #[serde(default)]
    pub population: String,
    pub auxiliary_class: String,
    pub conflict_cells: usize,
    pub conflict_cells_at_2048: usize,
    pub ns_only_conflict_cells: usize,
    pub pred_blind: bool,
    pub internal_conflict: bool,
    pub attractor: bool,
    pub interpretation_caveat: String,
    pub next_action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenDiagnosticTiming {
    pub population_seconds: f64,
    pub gradient_cell_seconds: Vec<f64>,
    pub rescore_seconds: Vec<f64>,
    #[serde(default)]
    pub comparison_seconds: Vec<f64>,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenDiagnosticReport {
    pub schema: String,
    pub evidence_class: String,
    pub run_class: String,
    pub registered: bool,
    pub research_claim: bool,
    pub public_data_read: bool,
    pub lifecycle: LifecycleRecord,
    pub provenance: LaunchProvenance,
    pub command: Vec<String>,
    pub device: String,
    pub device_is_cuda: bool,
    pub gpu_identity: Option<GpuIdentity>,
    pub output_root: PathBuf,
    pub spec: FrozenDiagnosticSpec,
    pub g: Option<EvidenceBinding>,
    pub g_identity: String,
    pub preflight: Option<EvidenceBinding>,
    pub train_config_sha256: String,
    pub cargo_lock_sha256: String,
    pub population_census_sha256: String,
    pub population: Option<PopulationCensus>,
    pub rescoring: Vec<SnapshotRescore>,
    pub gradients: Vec<GradientCellReport>,
    #[serde(default)]
    pub comparisons: Vec<DiagnosticLogitComparison>,
    pub persistence: Vec<PersistenceReport>,
    pub runtime_estimate: Option<RuntimeEstimate>,
    pub verdict: Option<DiagnosticVerdict>,
    pub timing: FrozenDiagnosticTiming,
    pub identity_root: String,
    pub error: Option<String>,
}

#[derive(Debug)]
struct AnatomyWithKeys {
    report: FalseEditAnatomy,
    keys: BTreeSet<u64>,
}

fn safe_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator >= NORM_EPSILON).then_some(numerator / denominator)
}

fn route_geometry(
    stats: &FoundationV2GradientRouteStats,
    full: &FoundationV2GradientRouteStats,
    prediction: &FoundationV2GradientRouteStats,
) -> RoutedGeometry {
    let make = |l2: f64, full_l2: f64, pred_l2: f64, cosine: Option<f64>| RouteGeometry {
        l2,
        share_of_full: safe_ratio(l2, full_l2),
        cosine_to_prediction: cosine.filter(|_| l2 >= NORM_EPSILON && pred_l2 >= NORM_EPSILON),
        kappa_to_prediction: if pred_l2 < NORM_EPSILON {
            None
        } else if l2 == 0.0 {
            Some(0.0)
        } else {
            cosine.map(|value| value * l2 / pred_l2)
        },
    };
    RoutedGeometry {
        global: make(
            stats.global_l2,
            full.global_l2,
            prediction.global_l2,
            stats.global_cosine_to_prediction,
        ),
        adamw: make(
            stats.adamw_l2,
            full.adamw_l2,
            prediction.adamw_l2,
            stats.adamw_cosine_to_prediction,
        ),
        muon: make(
            stats.muon_l2,
            full.muon_l2,
            prediction.muon_l2,
            stats.muon_cosine_to_prediction,
        ),
    }
}

fn prefix_norms(
    varmap: &VarMap,
    grads: &GradStore,
) -> Result<BTreeMap<String, PrefixGradientNorm>> {
    let data = varmap.data().lock().unwrap();
    let mut out = BTreeMap::new();
    for prefix in POSITIVE_PREFIXES.into_iter().chain(ZERO_PREFIXES) {
        let matching = data
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .collect::<Vec<_>>();
        ensure!(!matching.is_empty(), "missing topology prefix {prefix}");
        let mut sum_sq: Option<Tensor> = None;
        let mut present = false;
        for (_, var) in matching {
            if let Some(gradient) = grads.get(var.as_tensor()) {
                present = true;
                let squared = gradient.to_dtype(DType::F32)?.sqr()?.sum_all()?;
                sum_sq = Some(match sum_sq {
                    None => squared,
                    Some(acc) => acc.add(&squared)?,
                });
            }
        }
        let (state, l2) = if present {
            let value = sum_sq
                .expect("present gradient has norm")
                .sqrt()?
                .to_scalar::<f32>()? as f64;
            ensure!(value.is_finite(), "non-finite prefix norm {prefix}");
            ("present".into(), Some(value))
        } else {
            ("structurally_absent".into(), None)
        };
        out.insert(prefix.into(), PrefixGradientNorm { state, l2 });
    }
    Ok(out)
}

fn route_applies(name: &str, tensor: &Tensor, route: OptimizerRoute) -> bool {
    match route {
        OptimizerRoute::All => true,
        OptimizerRoute::AdamW => !uses_muon(name, tensor.dims()),
        OptimizerRoute::Muon => uses_muon(name, tensor.dims()),
    }
}

fn residual_norm(
    left: &GradStore,
    right: &GradStore,
    varmap: &VarMap,
    route: OptimizerRoute,
) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    let mut sum_sq: Option<Tensor> = None;
    for (name, var) in data.iter() {
        let parameter = var.as_tensor();
        if !route_applies(name, parameter, route) {
            continue;
        }
        let difference = match (left.get(parameter), right.get(parameter)) {
            (Some(a), Some(b)) => a.to_dtype(DType::F32)?.sub(&b.to_dtype(DType::F32)?)?,
            (Some(a), None) => a.to_dtype(DType::F32)?,
            (None, Some(b)) => b.to_dtype(DType::F32)?.neg()?,
            (None, None) => continue,
        };
        let squared = difference.sqr()?.sum_all()?;
        sum_sq = Some(match sum_sq {
            None => squared,
            Some(acc) => acc.add(&squared)?,
        });
    }
    let Some(sum_sq) = sum_sq else {
        return Ok(0.0);
    };
    let norm = sum_sq.sqrt()?.to_scalar::<f32>()? as f64;
    ensure!(norm.is_finite(), "non-finite residual norm");
    Ok(norm)
}

fn reconstruction_check(
    direct: &GradStore,
    reconstructed: &GradStore,
    varmap: &VarMap,
) -> Result<ReconstructionCheck> {
    let check = |route| -> Result<(f64, bool, f64, f64)> {
        let reference = gradient_l2_for_optimizer_route(direct, varmap, route)?;
        let residual = residual_norm(direct, reconstructed, varmap, route)?;
        if reference < NORM_EPSILON {
            Ok((residual, residual <= NORM_EPSILON, reference, residual))
        } else {
            let relative = residual / reference;
            Ok((
                relative,
                relative <= RECONSTRUCTION_REL_TOLERANCE,
                reference,
                residual,
            ))
        }
    };
    let (global, global_pass, global_reference_l2, global_absolute_residual_l2) =
        check(OptimizerRoute::All)?;
    let (adamw, adamw_pass, adamw_reference_l2, adamw_absolute_residual_l2) =
        check(OptimizerRoute::AdamW)?;
    let (muon, muon_pass, muon_reference_l2, muon_absolute_residual_l2) =
        check(OptimizerRoute::Muon)?;
    Ok(ReconstructionCheck {
        global_relative_or_absolute_residual: global,
        adamw_relative_or_absolute_residual: adamw,
        muon_relative_or_absolute_residual: muon,
        global_reference_l2,
        global_absolute_residual_l2,
        adamw_reference_l2,
        adamw_absolute_residual_l2,
        muon_reference_l2,
        muon_absolute_residual_l2,
        passed: global_pass && adamw_pass && muon_pass,
    })
}

fn ensure_reconstructions(
    full: &ReconstructionCheck,
    prediction: &ReconstructionCheck,
) -> Result<()> {
    ensure!(
        full.passed && prediction.passed,
        "gradient component reconstruction failed: {}",
        serde_json::json!({"full": full, "prediction": prediction})
    );
    Ok(())
}

fn bounded_cosine(
    left: &GradStore,
    right: &GradStore,
    varmap: &VarMap,
    route: OptimizerRoute,
) -> Result<Option<f64>> {
    let left_norm = gradient_l2_for_optimizer_route(left, varmap, route)?;
    let right_norm = gradient_l2_for_optimizer_route(right, varmap, route)?;
    if left_norm < NORM_EPSILON || right_norm < NORM_EPSILON {
        return Ok(None);
    }
    gradient_cosine_for_optimizer_route(left, right, varmap, route)
}

fn ns_muon_cosine(
    full: &GradStore,
    prediction: &GradStore,
    varmap: &VarMap,
) -> Result<Option<f64>> {
    let data = varmap.data().lock().unwrap();
    let mut dot: Option<Tensor> = None;
    let mut full_sq: Option<Tensor> = None;
    let mut prediction_sq: Option<Tensor> = None;
    for (name, var) in data.iter() {
        let parameter = var.as_tensor();
        if !uses_muon(name, parameter.dims()) {
            continue;
        }
        let full_gradient = full.get(parameter);
        let prediction_gradient = prediction.get(parameter);
        if full_gradient.is_none() && prediction_gradient.is_none() {
            continue;
        }
        let matrix_shape = matrix_view(parameter)?.dims().to_vec();
        let transform = |gradient: Option<&Tensor>| -> Result<Tensor> {
            match gradient {
                Some(gradient) => Ok(muon_shape_rescale(
                    &hybrid_newton_schulz(&matrix_view(gradient)?)?,
                    0.2,
                )?),
                None => Tensor::zeros(matrix_shape.as_slice(), DType::F32, parameter.device())
                    .map_err(Into::into),
            }
        };
        let full_update = transform(full_gradient)?;
        let prediction_update = transform(prediction_gradient)?;
        let product = full_update.mul(&prediction_update)?.sum_all()?;
        let left = full_update.sqr()?.sum_all()?;
        let right = prediction_update.sqr()?.sum_all()?;
        dot = Some(match dot {
            None => product,
            Some(acc) => acc.add(&product)?,
        });
        full_sq = Some(match full_sq {
            None => left,
            Some(acc) => acc.add(&left)?,
        });
        prediction_sq = Some(match prediction_sq {
            None => right,
            Some(acc) => acc.add(&right)?,
        });
    }
    let (Some(dot), Some(full_sq), Some(prediction_sq)) = (dot, full_sq, prediction_sq) else {
        return Ok(None);
    };
    let values = Tensor::stack(&[&dot, &full_sq, &prediction_sq], 0)?.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "memoryless NS-Muon cosine contains a non-finite reduction"
    );
    let full_norm = f64::from(values[1]).sqrt();
    let prediction_norm = f64::from(values[2]).sqrt();
    if full_norm < NORM_EPSILON || prediction_norm < NORM_EPSILON {
        return Ok(None);
    }
    let cosine = f64::from(values[0]) / (full_norm * prediction_norm);
    ensure!(
        cosine.is_finite(),
        "memoryless NS-Muon cosine is non-finite"
    );
    Ok(Some(cosine.clamp(-1.0, 1.0)))
}

fn predictions_from_logits(logits: &Tensor) -> Result<Vec<Vec<u8>>> {
    let batch = logits.dim(0)?;
    logits
        .argmax(D::Minus1)?
        .reshape((batch, ()))?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()
        .map_err(Into::into)
}

fn histogram_bin(count: usize) -> &'static str {
    match count {
        0 => "0",
        1 => "1",
        2..=3 => "2-3",
        4..=7 => "4-7",
        8..=15 => "8-15",
        16..=31 => "16-31",
        32..=63 => "32-63",
        _ => ">=64",
    }
}

fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(index).copied()
}

fn anatomy(
    samples: &[V5Sample],
    predictions: &[Vec<u8>],
    logits: &[f32],
    batch_positions: &[usize],
) -> Result<AnatomyWithKeys> {
    ensure!(
        samples.len() == predictions.len()
            && logits.len() == samples.len() * FRAME_SIDE * FRAME_SIDE * PALETTE_SIZE
            && samples.len() == batch_positions.len() * REGISTERED_BATCH_SIZE,
        "anatomy row mismatch"
    );
    let mut margins = Vec::new();
    let mut margin_bins = BTreeMap::from([
        ("[0,0.5]".into(), 0usize),
        ("(0.5,1]".into(), 0),
        ("(1,2]".into(), 0),
        ("(2,4]".into(), 0),
        (">4".into(), 0),
    ]);
    let mut locations = BTreeMap::from([
        ("row63".into(), 0usize),
        ("other_border".into(), 0),
        ("interior".into(), 0),
    ]);
    let mut class_pairs = BTreeMap::new();
    let mut distance = BTreeMap::from([
        ("none".into(), 0usize),
        ("1".into(), 0),
        ("2-3".into(), 0),
        ("4-7".into(), 0),
        (">=8".into(), 0),
    ]);
    let mut changed_rows_hist = BTreeMap::new();
    let mut no_change_rows_hist = BTreeMap::new();
    let mut near_miss = BTreeMap::from([
        ("<=1".into(), 0usize),
        ("<=2".into(), 0),
        ("<=4".into(), 0),
        ("<=8".into(), 0),
    ]);
    let mut spill = 0usize;
    let mut keys = BTreeSet::new();

    for row in 0..samples.len() {
        let sample = &samples[row];
        let current = &sample.transition.current.pixels;
        let target = &sample.transition.next.pixels;
        ensure!(
            current.len() == FRAME_SIDE * FRAME_SIDE
                && target.len() == FRAME_SIDE * FRAME_SIDE
                && predictions[row].len() == FRAME_SIDE * FRAME_SIDE,
            "anatomy geometry mismatch"
        );
        let changed_pixels = current
            .iter()
            .zip(target)
            .enumerate()
            .filter_map(|(pixel, (before, after))| (before != after).then_some(pixel))
            .collect::<Vec<_>>();
        let changed_classes = changed_pixels
            .iter()
            .map(|pixel| target[*pixel])
            .collect::<BTreeSet<_>>();
        let mut changed_exact = !changed_pixels.is_empty();
        let mut row_false_edits = 0usize;
        for pixel in 0..target.len() {
            if current[pixel] != target[pixel] {
                changed_exact &= predictions[row][pixel] == target[pixel];
                continue;
            }
            let predicted = predictions[row][pixel];
            if predicted == target[pixel] {
                continue;
            }
            row_false_edits += 1;
            let y = pixel / FRAME_SIDE;
            let x = pixel % FRAME_SIDE;
            let palette_offset = (row * FRAME_SIDE * FRAME_SIDE + pixel) * PALETTE_SIZE;
            let palette = &logits[palette_offset..palette_offset + PALETTE_SIZE];
            let margin =
                f64::from(palette[usize::from(predicted)] - palette[usize::from(target[pixel])]);
            ensure!(
                margin.is_finite() && margin >= -1e-6,
                "invalid false-edit margin"
            );
            let margin = margin.max(0.0);
            margins.push(margin);
            let bin = if margin <= 0.5 {
                "[0,0.5]"
            } else if margin <= 1.0 {
                "(0.5,1]"
            } else if margin <= 2.0 {
                "(1,2]"
            } else if margin <= 4.0 {
                "(2,4]"
            } else {
                ">4"
            };
            *margin_bins.get_mut(bin).expect("fixed margin bin") += 1;
            let location = if y == FRAME_SIDE - 1 {
                "row63"
            } else if x == 0 || x == FRAME_SIDE - 1 || y == 0 {
                "other_border"
            } else {
                "interior"
            };
            *locations.get_mut(location).expect("fixed location") += 1;
            *class_pairs
                .entry(format!("{}->{}", target[pixel], predicted))
                .or_insert(0) += 1;
            spill += usize::from(changed_classes.contains(&predicted));
            let distance_bin = changed_pixels
                .iter()
                .map(|changed| {
                    let cy = changed / FRAME_SIDE;
                    let cx = changed % FRAME_SIDE;
                    x.abs_diff(cx).max(y.abs_diff(cy))
                })
                .min()
                .map(|value| match value {
                    0 | 1 => "1",
                    2..=3 => "2-3",
                    4..=7 => "4-7",
                    _ => ">=8",
                })
                .unwrap_or("none");
            *distance.get_mut(distance_bin).expect("fixed distance bin") += 1;
            let batch_slot = row / REGISTERED_BATCH_SIZE;
            let local_row = row % REGISTERED_BATCH_SIZE;
            let stable_row = batch_positions[batch_slot] * REGISTERED_BATCH_SIZE + local_row;
            keys.insert((stable_row * FRAME_SIDE * FRAME_SIDE + pixel) as u64);
        }
        let histogram = if changed_pixels.is_empty() {
            &mut no_change_rows_hist
        } else {
            &mut changed_rows_hist
        };
        *histogram
            .entry(histogram_bin(row_false_edits).into())
            .or_insert(0) += 1;
        if changed_exact {
            for limit in [1usize, 2, 4, 8] {
                if row_false_edits <= limit {
                    *near_miss
                        .get_mut(&format!("<={limit}"))
                        .expect("fixed limit") += 1;
                }
            }
        }
    }
    margins.sort_by(f64::total_cmp);
    let report = FalseEditAnatomy {
        population: PRIMARY_POPULATION.into(),
        false_edit_pixels: margins.len(),
        margins: MarginSummary {
            count: margins.len(),
            bins: margin_bins,
            min: margins.first().copied(),
            median: quantile(&margins, 0.5),
            p90: quantile(&margins, 0.9),
            p99: quantile(&margins, 0.99),
            max: margins.last().copied(),
        },
        locations,
        class_pairs,
        predicted_class_from_changed_region: spill,
        distance_to_changed: distance,
        row_histogram_changed_rows: changed_rows_hist,
        row_histogram_no_change_rows: no_change_rows_hist,
        changed_exact_near_miss: near_miss,
    };
    ensure!(
        report.false_edit_pixels == keys.len(),
        "false-edit key collision"
    );
    Ok(AnatomyWithKeys { report, keys })
}

fn persistence(split: &str, sets: &BTreeMap<usize, BTreeSet<u64>>) -> Result<PersistenceReport> {
    let at_1024 = sets
        .get(&1024)
        .context("missing step-1024 false-edit set")?;
    let at_1536 = sets
        .get(&1536)
        .context("missing step-1536 false-edit set")?;
    let at_2048 = sets
        .get(&2048)
        .context("missing step-2048 false-edit set")?;
    let all_three = at_1024
        .intersection(at_1536)
        .filter(|key| at_2048.contains(*key))
        .count();
    let union = at_1024
        .union(at_1536)
        .copied()
        .collect::<BTreeSet<_>>()
        .union(at_2048)
        .count();
    let persistent = at_2048.intersection(at_1536).count();
    Ok(PersistenceReport {
        population: PRIMARY_POPULATION.into(),
        split: split.into(),
        false_edits_1024: at_1024.len(),
        false_edits_1536: at_1536.len(),
        false_edits_2048: at_2048.len(),
        intersection_all_three: all_three,
        union_all_three: union,
        persistent_1536_to_2048: persistent,
        persistence_1536_to_2048: (!at_2048.is_empty())
            .then_some(persistent as f64 / at_2048.len() as f64),
        final_present_all_three_fraction: (!at_2048.is_empty())
            .then_some(all_three as f64 / at_2048.len() as f64),
        new_1024_to_1536: at_1536.difference(at_1024).count(),
        resolved_1024_to_1536: at_1024.difference(at_1536).count(),
        new_1536_to_2048: at_2048.difference(at_1536).count(),
        resolved_1536_to_2048: at_1536.difference(at_2048).count(),
    })
}

fn snapshot_by_step(report: &MultibatchScreenReport, step: usize) -> Result<&ScreenSnapshot> {
    report
        .snapshots
        .iter()
        .find(|snapshot| snapshot.step == step)
        .with_context(|| format!("G report lacks raw snapshot {step}"))
}

struct BatchSlice<'a> {
    position: usize,
    union: BatchUnion,
    predictions: &'a [Vec<u8>],
}

fn sliced_batches<'a>(
    samples: &'a [V5Sample],
    predictions: &'a [Vec<u8>],
    positions: &[usize],
) -> Result<Vec<BatchSlice<'a>>> {
    ensure!(
        samples.len() == predictions.len()
            && samples.len() == positions.len() * REGISTERED_BATCH_SIZE,
        "batch slicing mismatch"
    );
    Ok(positions
        .iter()
        .enumerate()
        .map(|(slot, position)| {
            let start = slot * REGISTERED_BATCH_SIZE;
            let end = start + REGISTERED_BATCH_SIZE;
            BatchSlice {
                position: *position,
                union: BatchUnion {
                    samples: samples[start..end].to_vec(),
                    factual_group_ranges: std::iter::once(43..53).collect(),
                },
                predictions: &predictions[start..end],
            }
        })
        .collect())
}

fn per_batch_scores(
    samples: &[V5Sample],
    predictions: &[Vec<u8>],
    positions: &[usize],
    index_offset: u64,
    population: &str,
) -> Result<Vec<PerBatchScore>> {
    sliced_batches(samples, predictions, positions)?
        .into_iter()
        .map(|slice| {
            let raw = exact_metrics(&slice.union.samples, slice.predictions)?;
            let groups = group_routing_metrics(&slice.union, slice.predictions)?;
            let action6 = action6_coordinate_metrics(&slice.union, slice.predictions)?;
            ensure!(
                groups.groups.len() == 1 && action6.groups.len() == 1,
                "batch group drift"
            );
            Ok(PerBatchScore {
                population: population.into(),
                batch_position: slice.position,
                batch_index: index_offset + slice.position as u64,
                raw,
                action_routed: groups.groups[0].action_routed,
                raw_full_exact_branches: groups.groups[0].raw_full_exact_branches,
                reproduced_distinct_changed_classes: groups.groups[0]
                    .reproduced_distinct_changed_classes,
                action6_changed_full_exact: action6.changed_action6_full_exact,
                action6_coordinate_routed: action6.groups[0].coordinate_routed,
            })
        })
        .collect()
}

fn additive_matches(scores: &[PerBatchScore], expected: &ExactMetrics) -> bool {
    let sum = |select: fn(&ExactMetrics) -> usize| {
        scores.iter().map(|row| select(&row.raw)).sum::<usize>()
    };
    sum(|m| m.rows) == expected.rows
        && sum(|m| m.changed_rows) == expected.changed_rows
        && sum(|m| m.changed_exact) == expected.changed_exact
        && sum(|m| m.full_exact) == expected.full_exact
        && sum(|m| m.all_row_exact) == expected.all_row_exact
        && sum(|m| m.unchanged_pixels) == expected.unchanged_pixels
        && sum(|m| m.false_edit_pixels) == expected.false_edit_pixels
        && sum(|m| m.false_edit_rows) == expected.false_edit_rows
}

fn group_matches(
    scores: &[PerBatchScore],
    expected: &crate::p2::multibatch_screen::UnionSnapshotMetrics,
) -> bool {
    scores.len() == expected.group_routing.groups.len()
        && scores.len() == expected.action6.groups.len()
        && scores.iter().enumerate().all(|(index, score)| {
            let group = &expected.group_routing.groups[index];
            let action6 = &expected.action6.groups[index];
            score.action_routed == group.action_routed
                && score.raw_full_exact_branches == group.raw_full_exact_branches
                && score.reproduced_distinct_changed_classes
                    == group.reproduced_distinct_changed_classes
                && score.action6_changed_full_exact == action6.raw_full_exact_changed_action6_rows
                && score.action6_coordinate_routed == action6.coordinate_routed
        })
}

fn diagnostic_identity(report: &FrozenDiagnosticReport) -> Result<String> {
    identity_frame_sha256(&[
        ("domain", FROZEN_DIAGNOSTIC_SCHEMA.as_bytes().to_vec()),
        (
            "source",
            report.provenance.source_revision.as_bytes().to_vec(),
        ),
        (
            "binary",
            report.provenance.binary_sha256.as_bytes().to_vec(),
        ),
        ("cargo", report.cargo_lock_sha256.as_bytes().to_vec()),
        ("g", report.g_identity.as_bytes().to_vec()),
        ("config", report.train_config_sha256.as_bytes().to_vec()),
        (
            "census",
            report.population_census_sha256.as_bytes().to_vec(),
        ),
        ("spec", serde_json::to_vec(&report.spec)?),
        ("preflight", serde_json::to_vec(&report.preflight)?),
    ])
}

fn clone_grad_store(source: &GradStore, varmap: &VarMap) -> GradStore {
    let mut cloned = GradStore::default();
    for var in varmap.all_vars() {
        let tensor = var.as_tensor();
        if let Some(gradient) = source.get(tensor) {
            cloned.insert(tensor, gradient.clone());
        }
    }
    cloned
}

fn accumulate_clone(
    target: &mut Option<GradStore>,
    source: &GradStore,
    varmap: &VarMap,
) -> Result<()> {
    accumulate_parameter_gradients(target, clone_grad_store(source, varmap), varmap)
}

fn component_report(
    objective: &str,
    weight: f64,
    grads: &GradStore,
    prediction: &GradStore,
    varmap: &VarMap,
    full_stats: &FoundationV2GradientRouteStats,
    prediction_stats: &FoundationV2GradientRouteStats,
) -> Result<GradientComponentReport> {
    let stats = foundation_v2_gradient_route_stats(grads, Some(prediction), varmap)?;
    Ok(GradientComponentReport {
        objective: objective.into(),
        weight,
        routes: route_geometry(&stats, full_stats, prediction_stats),
        prefixes: prefix_norms(varmap, grads)?,
    })
}

struct PredictionMasks {
    changed: Vec<f32>,
    unchanged: Vec<f32>,
    false_edit: Vec<f32>,
    changed_count: usize,
    unchanged_count: usize,
    false_edit_count: usize,
}

fn masks_for_prediction_sub_losses(
    samples: &[V5Sample],
    raw_predictions: &[Vec<u8>],
) -> Result<PredictionMasks> {
    ensure!(
        samples.len() == raw_predictions.len(),
        "prediction-mask row mismatch"
    );
    let pixels = samples.len() * FRAME_SIDE * FRAME_SIDE;
    let mut changed = Vec::with_capacity(pixels);
    let mut unchanged = Vec::with_capacity(pixels);
    let mut false_edit = Vec::with_capacity(pixels);
    let mut changed_count = 0usize;
    let mut unchanged_count = 0usize;
    let mut false_edit_count = 0usize;
    for (sample, predicted) in samples.iter().zip(raw_predictions) {
        ensure!(
            predicted.len() == FRAME_SIDE * FRAME_SIDE
                && sample.transition.current.pixels.len() == FRAME_SIDE * FRAME_SIDE
                && sample.transition.next.pixels.len() == FRAME_SIDE * FRAME_SIDE
                && sample.content_mask.values.len() == FRAME_SIDE * FRAME_SIDE,
            "prediction-mask geometry mismatch"
        );
        for (pixel, predicted_pixel) in predicted.iter().enumerate() {
            let content = sample.content_mask.values[pixel] != 0;
            let is_changed = content
                && sample.transition.current.pixels[pixel] != sample.transition.next.pixels[pixel];
            let is_unchanged = content && !is_changed;
            let is_false_edit =
                is_unchanged && *predicted_pixel != sample.transition.next.pixels[pixel];
            changed.push(f32::from(is_changed));
            unchanged.push(f32::from(is_unchanged));
            false_edit.push(f32::from(is_false_edit));
            changed_count += usize::from(is_changed);
            unchanged_count += usize::from(is_unchanged);
            false_edit_count += usize::from(is_false_edit);
        }
    }
    Ok(PredictionMasks {
        changed,
        unchanged,
        false_edit,
        changed_count,
        unchanged_count,
        false_edit_count,
    })
}

fn prediction_mask_sha256(mask: &[f32]) -> Result<String> {
    identity_frame_sha256(&[
        ("dtype", b"F32".to_vec()),
        ("shape", serde_json::to_vec(&[mask.len()])?),
        (
            "values",
            mask.iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        ),
    ])
}

fn bind_prediction_masks(
    samples: &[V5Sample],
    capture: &LogitCapture,
) -> Result<(PredictionMasks, SameForwardMaskBinding)> {
    let masks = masks_for_prediction_sub_losses(samples, &capture.predictions)?;
    // Rebuild the binding from the captured argmax, independently of the mask
    // vectors consumed by the gradient losses. No second forward or argmax.
    let recomputed = masks_for_prediction_sub_losses(samples, &capture.predictions)?;
    let binding = SameForwardMaskBinding {
        logits: capture.fingerprint.clone(),
        mask_sha256: prediction_mask_sha256(&recomputed.false_edit)?,
        false_edit_pixels: recomputed.false_edit_count,
    };
    ensure!(
        masks.false_edit_count == binding.false_edit_pixels
            && prediction_mask_sha256(&masks.false_edit)? == binding.mask_sha256,
        "same-forward mask reconstruction failed"
    );
    Ok((masks, binding))
}

fn ensure_same_forward_binding(cell: &GradientCellReport) -> Result<()> {
    let binding = cell
        .mask_binding
        .as_ref()
        .context("missing same-forward binding")?;
    ensure!(
        cell.mask_population == PRIMARY_POPULATION
            && cell.prediction_logits_fingerprint.as_ref() == Some(&binding.logits)
            && cell.false_edit_mask_sha256 == binding.mask_sha256
            && cell.false_edit_pixels == binding.false_edit_pixels
            && !binding.mask_sha256.is_empty()
            && !binding.logits.sha256.is_empty(),
        "same-forward count or fingerprint binding drift"
    );
    Ok(())
}

fn diagnostic_objective(
    cfg: &TrainConfig,
    ep_weight: f64,
    snapshot_step: usize,
    batch_position: usize,
) -> FoundationV2ObjectiveConfig {
    FoundationV2ObjectiveConfig {
        ep_weight,
        sigreg_projections: REGISTERED_SIGREG_PROJECTIONS,
        sigreg_knots: REGISTERED_SIGREG_KNOTS,
        sigreg_seed: REGISTERED_SEED
            .wrapping_add(snapshot_step as u64)
            .wrapping_add(batch_position as u64),
        q_mse_threshold: cfg.q_mse_threshold,
        rollout_enabled: false,
        split_ce_weighting: cfg.split_ce_weighting,
        split_ce_changed_budget: cfg.split_ce_changed_budget,
        capture_mechanism_seams: false,
        capture_pred_per_pixel: true,
    }
}

fn masked_prediction_loss(
    per_pixel: &Tensor,
    mask: Vec<f32>,
    denominator: usize,
    coefficient: f64,
) -> Result<Tensor> {
    ensure!(
        denominator > 0,
        "diagnostic sub-loss has a zero denominator"
    );
    ensure!(
        mask.len() == per_pixel.elem_count(),
        "diagnostic sub-loss mask geometry mismatch"
    );
    let mask = Tensor::from_vec(mask, per_pixel.dims(), per_pixel.device())?;
    per_pixel
        .mul(&mask)?
        .sum_all()?
        .affine(coefficient / denominator as f64, 0.0)
        .map_err(Into::into)
}

fn all_zero_prefixes(component: &GradientComponentReport) -> bool {
    ZERO_PREFIXES.iter().all(|prefix| {
        component
            .prefixes
            .get(*prefix)
            .is_some_and(|norm| norm.state == "structurally_absent" || norm.l2 == Some(0.0))
    })
}

fn cell_conflicts(cell: &GradientCellReport) -> bool {
    [
        cell.combined.global.cosine_to_prediction,
        cell.combined.adamw.cosine_to_prediction,
        cell.combined.muon.cosine_to_prediction,
        cell.ns_muon_cosine_full_to_prediction,
    ]
    .into_iter()
    .flatten()
    .any(|value| value < CONFLICT_COSINE)
        || cell
            .auxiliary
            .global
            .share_of_full
            .is_some_and(|value| value > CONFLICT_AUX_SHARE)
        || cell
            .auxiliary
            .global
            .kappa_to_prediction
            .is_some_and(|value| value <= CONFLICT_KAPPA)
        || cell
            .false_edit_to_auxiliary_cosine
            .is_some_and(|value| value <= FALSE_EDIT_AUX_CONFLICT_COSINE)
}

fn cell_conflicts_without_ns(cell: &GradientCellReport) -> bool {
    [
        cell.combined.global.cosine_to_prediction,
        cell.combined.adamw.cosine_to_prediction,
        cell.combined.muon.cosine_to_prediction,
    ]
    .into_iter()
    .flatten()
    .any(|value| value < CONFLICT_COSINE)
        || cell
            .auxiliary
            .global
            .share_of_full
            .is_some_and(|value| value > CONFLICT_AUX_SHARE)
        || cell
            .auxiliary
            .global
            .kappa_to_prediction
            .is_some_and(|value| value <= CONFLICT_KAPPA)
        || cell
            .false_edit_to_auxiliary_cosine
            .is_some_and(|value| value <= FALSE_EDIT_AUX_CONFLICT_COSINE)
}

fn cell_negligible(cell: &GradientCellReport) -> bool {
    [
        cell.combined.global.cosine_to_prediction,
        cell.combined.adamw.cosine_to_prediction,
        cell.combined.muon.cosine_to_prediction,
        cell.ns_muon_cosine_full_to_prediction,
    ]
    .into_iter()
    .all(|value| value.is_some_and(|value| value >= NEGLIGIBLE_COSINE))
        && cell
            .auxiliary
            .global
            .share_of_full
            .is_some_and(|value| value <= NEGLIGIBLE_AUX_SHARE)
        && cell
            .auxiliary
            .global
            .kappa_to_prediction
            .is_some_and(|value| value >= NEGLIGIBLE_KAPPA)
        && cell
            .false_edit_to_auxiliary_cosine
            .is_some_and(|value| value >= NEGLIGIBLE_KAPPA)
}

fn classify(
    gradients: &[GradientCellReport],
    rescoring: &[SnapshotRescore],
    persistence_rows: &[PersistenceReport],
) -> Result<DiagnosticVerdict> {
    ensure!(
        gradients
            .iter()
            .all(|cell| cell.mask_population == PRIMARY_POPULATION)
            && rescoring.iter().all(|snapshot| {
                snapshot.primary_population == PRIMARY_POPULATION
                    && snapshot.anatomy_population == PRIMARY_POPULATION
                    && snapshot
                        .train_anatomy
                        .iter()
                        .chain(&snapshot.heldout_anatomy)
                        .chain(&snapshot.train_anatomy_by_batch)
                        .chain(&snapshot.heldout_anatomy_by_batch)
                        .all(|anatomy| anatomy.population == PRIMARY_POPULATION)
            })
            && persistence_rows
                .iter()
                .all(|row| row.population == PRIMARY_POPULATION),
        "classifier requires primary_same_forward populations"
    );
    let cells = gradients
        .iter()
        .filter(|cell| GRADIENT_STEPS.contains(&cell.snapshot_step))
        .collect::<Vec<_>>();
    ensure!(
        cells.len() == GRADIENT_STEPS.len() * TRAIN_BATCHES,
        "classification requires 24 cells"
    );
    let conflicts = cells.iter().filter(|cell| cell_conflicts(cell)).count();
    let conflicts_at_2048 = cells
        .iter()
        .filter(|cell| cell.snapshot_step == 2048 && cell_conflicts(cell))
        .count();
    let ns_only_conflicts = cells
        .iter()
        .filter(|cell| {
            cell.ns_muon_cosine_full_to_prediction
                .is_some_and(|value| value < CONFLICT_COSINE)
                && !cell_conflicts_without_ns(cell)
        })
        .count();
    let auxiliary_class = if conflicts >= 4 || conflicts_at_2048 >= 2 {
        "AUX_CONFLICTS"
    } else if cells.iter().all(|cell| cell_negligible(cell)) {
        "AUX_NEGLIGIBLE"
    } else {
        "AUX_MIXED"
    };
    let pred_blind = cells
        .iter()
        .filter(|cell| matches!(cell.snapshot_step, 1536 | 2048))
        .all(|cell| {
            cell.false_edit_share_of_prediction
                .is_some_and(|share| share < PRED_BLIND_SHARE)
        });
    let internal_conflict = cells
        .iter()
        .filter(|cell| {
            cell.changed_to_unchanged_cosine
                .is_some_and(|cosine| cosine <= INTERNAL_CONFLICT_COSINE)
        })
        .count()
        >= 8;
    let train_persistence = persistence_rows
        .iter()
        .find(|row| row.split == "train")
        .context("missing train persistence")?;
    let final_train = rescoring
        .iter()
        .find(|snapshot| snapshot.step == 2048)
        .and_then(|snapshot| snapshot.train_anatomy.as_ref())
        .context("missing step-2048 train anatomy")?;
    let attractor = train_persistence
        .persistence_1536_to_2048
        .is_some_and(|value| value >= ATTRACTOR_PERSISTENCE)
        && final_train
            .margins
            .median
            .is_some_and(|value| value >= ATTRACTOR_MARGIN);
    let next_action = if pred_blind || internal_conflict {
        "preregister the G-mandated prediction-only arm as causally non-decisive and do not launch it; preregister a matched changed-versus-unchanged prediction-weight discriminator"
    } else if auxiliary_class == "AUX_NEGLIGIBLE" {
        "preregister the G-mandated prediction-only arm, disclose its first-order near-null forecast, and launch nothing without new authority"
    } else {
        "preregister the G-mandated matched prediction-only arm as the weakest direct discriminator and launch nothing without new authority"
    };
    Ok(DiagnosticVerdict {
        population: PRIMARY_POPULATION.into(),
        auxiliary_class: auxiliary_class.into(),
        conflict_cells: conflicts,
        conflict_cells_at_2048: conflicts_at_2048,
        ns_only_conflict_cells: ns_only_conflicts,
        pred_blind,
        internal_conflict,
        attractor,
        interpretation_caveat: "classifier thresholds were registered on chunk-32 and now apply to the batch-128 primary population; primary D2 is backward-free and does not reproduce D1 rollout-backward allocator history; step-2048 EP weight is 0.0002540643898500332 (nearly EP-off); NS-only conflicts may reflect memoryless Muon whitening rather than objective competition".into(),
        next_action: next_action.into(),
    })
}

fn runtime_estimate(report: &FrozenDiagnosticReport) -> Result<RuntimeEstimate> {
    ensure!(!report.registered, "runtime is measured only by preflight");
    let d1_cell_seconds = report
        .timing
        .gradient_cell_seconds
        .iter()
        .copied()
        .reduce(f64::max)
        .context("preflight lacks a D1 timing")?;
    let d2d3_pair_seconds = report
        .timing
        .rescore_seconds
        .iter()
        .copied()
        .reduce(f64::max)
        .context("preflight lacks a D2/D3 timing")?;
    let estimated_device_seconds = 25.0 * d1_cell_seconds + 56.0 * d2d3_pair_seconds;
    Ok(RuntimeEstimate {
        d1_cell_seconds,
        d2d3_pair_seconds,
        estimated_device_seconds,
        admitted: estimated_device_seconds <= ADMISSION_DEVICE_SECONDS,
    })
}

fn ensure_source_descends_from_g(provenance: &LaunchProvenance) -> Result<()> {
    let status = Command::new("git")
        .args([
            "merge-base",
            "--is-ancestor",
            REGISTERED_G_SOURCE,
            provenance.source_revision.as_str(),
        ])
        .status()
        .context("run git ancestry check")?;
    ensure!(
        status.success(),
        "diagnostic source does not descend from registered G"
    );
    Ok(())
}

fn bind_g(path: &Path) -> Result<(MultibatchScreenReport, EvidenceBinding)> {
    let (g, binding) = bind_screen_report(path)?;
    ensure!(
        file_sha256_hex(&binding.report)? == REGISTERED_G_REPORT_SHA256
            && binding.manifest_sha256 == REGISTERED_G_MANIFEST_SHA256
            && binding.identity_root == REGISTERED_G_IDENTITY,
        "G evidence identity differs from the frozen registration"
    );
    ensure!(
        g.registered
            && g.provenance.source_revision == REGISTERED_G_SOURCE
            && g.provenance.binary_sha256 == REGISTERED_G_BINARY_SHA256
            && g.cargo_lock_sha256 == REGISTERED_G_CARGO_LOCK_SHA256
            && g.train_config_sha256 == REGISTERED_TRAIN_CONFIG_SHA256
            && g.verdict.as_ref().is_some_and(|verdict| {
                verdict.outcome == OUTCOME_DOES_NOT_SCALE && verdict.extension_signal == Some(false)
            }),
        "G is not the frozen DOES_NOT_SCALE / extension-false parent"
    );
    ensure!(
        g.snapshots.len() == SNAPSHOT_STEPS.len(),
        "G snapshot count drift"
    );
    for ((snapshot, step), sha256) in g
        .snapshots
        .iter()
        .zip(SNAPSHOT_STEPS)
        .zip(REGISTERED_RAW_SHA256)
    {
        ensure!(
            snapshot.step == step && snapshot.raw_sha256 == sha256,
            "G raw snapshot {step} hash drift"
        );
    }
    Ok((g, binding))
}

fn bind_diagnostic_report(path: &Path) -> Result<(FrozenDiagnosticReport, EvidenceBinding)> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize diagnostic report {}", path.display()))?;
    let report: FrozenDiagnosticReport =
        serde_json::from_slice(&fs::read(&report_path)?).context("parse diagnostic report")?;
    let root = fs::canonicalize(&report.output_root)?;
    ensure!(
        report_path == fs::canonicalize(root.join(REPORT_FILE))?,
        "diagnostic report is outside its claimed root"
    );
    let (manifest, _) = external_manifest_paths(&root)?;
    let manifest = fs::canonicalize(manifest)?;
    let manifest_sha256 = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &manifest_sha256)?;
    ensure_completed_cleanly(&report)?;
    validate_report_files(&root, &report)?;
    ensure!(
        report.identity_root == diagnostic_identity(&report)?,
        "diagnostic identity drift"
    );
    let binding = EvidenceBinding {
        report: report_path,
        root,
        manifest,
        manifest_sha256,
        identity_root: report.identity_root.clone(),
    };
    Ok((report, binding))
}

fn ensure_preflight_binds(
    preflight: &FrozenDiagnosticReport,
    current: &FrozenDiagnosticReport,
) -> Result<RuntimeEstimate> {
    ensure!(
        !preflight.registered
            && preflight.run_class == RUN_CLASS_PREFLIGHT
            && preflight.verdict.is_none()
            && current.registered,
        "registered diagnostic did not bind the required preflight class"
    );
    ensure!(
        same_build_identity(&preflight.provenance, &current.provenance)
            && preflight.g_identity == current.g_identity
            && preflight.g == current.g
            && preflight.train_config_sha256 == current.train_config_sha256
            && preflight.cargo_lock_sha256 == current.cargo_lock_sha256
            && preflight.population == current.population
            && preflight.population_census_sha256 == current.population_census_sha256
            && preflight.gpu_identity == current.gpu_identity,
        "preflight build, G, config, population, or GPU identity differs"
    );
    let estimate = runtime_estimate(preflight)?;
    ensure!(
        preflight.runtime_estimate.as_ref() == Some(&estimate),
        "preflight runtime estimate drift"
    );
    ensure!(
        estimate.admitted,
        "preflight estimate exceeds the 900-second device admission cap"
    );
    Ok(estimate)
}

fn ensure_preflight_controls_bind(
    preflight: &FrozenDiagnosticReport,
    current: &FrozenDiagnosticReport,
) -> Result<()> {
    let expected_cells = [(0usize, 0usize), (1024, 0)];
    for (step, position) in expected_cells {
        let expected = preflight
            .gradients
            .iter()
            .find(|cell| cell.snapshot_step == step && cell.batch_position == position)
            .context("preflight lacks a required control cell")?;
        let observed = current
            .gradients
            .iter()
            .find(|cell| cell.snapshot_step == step && cell.batch_position == position)
            .context("registered run lacks a required preflight control cell")?;
        ensure_gradient_control_matches(expected, observed)?;
    }
    for expected_rescore in &preflight.rescoring {
        let observed_rescore = current
            .rescoring
            .iter()
            .find(|snapshot| snapshot.step == expected_rescore.step)
            .with_context(|| {
                format!(
                    "registered run lacks preflight D2 control at step {}",
                    expected_rescore.step
                )
            })?;
        ensure_rescore_control_matches(expected_rescore, observed_rescore)?;
    }
    Ok(())
}

fn json_values_approximately_equal(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    json_values_approximately_equal_at_key(left, right, None)
}

fn json_values_approximately_equal_at_key(
    left: &serde_json::Value,
    right: &serde_json::Value,
    key: Option<&str>,
) -> bool {
    match (left, right) {
        (serde_json::Value::Number(left), serde_json::Value::Number(right)) => {
            if left.is_i64() || left.is_u64() || right.is_i64() || right.is_u64() {
                left == right
            } else {
                left.as_f64()
                    .zip(right.as_f64())
                    .is_some_and(|(left, right)| match key {
                        Some("pre_clip_gradient_norm") => norm_approximately(left, right),
                        _ => approximately(left, right),
                    })
            }
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| json_values_approximately_equal_at_key(left, right, key))
        }
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, left)| {
                    right.get(key).is_some_and(|right| {
                        json_values_approximately_equal_at_key(left, right, Some(key))
                    })
                })
        }
        _ => left == right,
    }
}

fn ensure_gradient_control_matches(
    expected: &GradientCellReport,
    observed: &GradientCellReport,
) -> Result<()> {
    let projected = |cell: &GradientCellReport| -> Result<serde_json::Value> {
        let mut value = serde_json::to_value(cell)?;
        let object = value
            .as_object_mut()
            .context("gradient control is not an object")?;
        object.remove("prediction_logits_fingerprint");
        object.remove("false_edit_mask_sha256");
        object.remove("training_raw_argmax_disagreement_pixels");
        if let Some(binding) = object
            .get_mut("mask_binding")
            .and_then(serde_json::Value::as_object_mut)
        {
            binding.remove("logits");
            binding.remove("mask_sha256");
        }
        Ok(value)
    };
    ensure!(
        json_values_approximately_equal(&projected(expected)?, &projected(observed)?),
        "registered gradient control differs from preflight beyond the frozen tolerance"
    );
    Ok(())
}

fn ensure_rescore_control_matches(
    expected: &SnapshotRescore,
    observed: &SnapshotRescore,
) -> Result<()> {
    let expected_control = serde_json::json!({
        "step": expected.step,
        "train": expected.train.first(),
        "heldout": expected.heldout.first(),
        "primary_train": expected.primary_train.first(),
        "primary_heldout": expected.primary_heldout.first(),
        "primary_population": expected.primary_population,
        "anatomy_population": expected.anatomy_population,
        "legacy_population": expected.legacy_population,
    });
    let observed_control = serde_json::json!({
        "step": observed.step,
        "train": observed.train.first(),
        "heldout": observed.heldout.first(),
        "primary_train": observed.primary_train.first(),
        "primary_heldout": observed.primary_heldout.first(),
        "primary_population": observed.primary_population,
        "anatomy_population": observed.anatomy_population,
        "legacy_population": observed.legacy_population,
    });
    ensure!(
        json_values_approximately_equal(&expected_control, &observed_control),
        "registered batch-0 D2/D3 control differs from preflight beyond the frozen tolerance"
    );
    let step = expected.step;
    for (split, expected_anatomy, observed_anatomy) in [
        (
            "train",
            expected.train_anatomy_by_batch.first(),
            observed.train_anatomy_by_batch.first(),
        ),
        (
            "heldout",
            expected.heldout_anatomy_by_batch.first(),
            observed.heldout_anatomy_by_batch.first(),
        ),
    ] {
        if let Some(expected_anatomy) = expected_anatomy {
            let observed_anatomy = observed_anatomy.with_context(|| {
                format!("registered {split} anatomy control is missing at step {step}")
            })?;
            ensure!(
                anatomy_integer_control(expected_anatomy)
                    == anatomy_integer_control(observed_anatomy),
                "registered {split} anatomy integer control differs at step {step}"
            );
            ensure!(
                expected_anatomy.population == observed_anatomy.population,
                "registered {split} primary anatomy population differs at step {step}"
            );
        }
    }
    Ok(())
}

fn preflight_rescore_control(
    preflight: &FrozenDiagnosticReport,
    step: usize,
) -> Option<&SnapshotRescore> {
    preflight
        .rescoring
        .iter()
        .find(|snapshot| snapshot.step == step)
}

fn anatomy_integer_control(anatomy: &FalseEditAnatomy) -> serde_json::Value {
    serde_json::json!({
        "false_edit_pixels": anatomy.false_edit_pixels,
        "margin_count": anatomy.margins.count,
        "margin_bins": anatomy.margins.bins,
        "locations": anatomy.locations,
        "class_pairs": anatomy.class_pairs,
        "predicted_class_from_changed_region": anatomy.predicted_class_from_changed_region,
        "distance_to_changed": anatomy.distance_to_changed,
        "row_histogram_changed_rows": anatomy.row_histogram_changed_rows,
        "row_histogram_no_change_rows": anatomy.row_histogram_no_change_rows,
        "changed_exact_near_miss": anatomy.changed_exact_near_miss,
    })
}

fn approximately(observed: f64, expected: f64) -> bool {
    (observed - expected).abs() <= 1e-6f64.max(expected.abs() * 1e-6)
}

fn norm_approximately(observed: f64, expected: f64) -> bool {
    (observed - expected).abs() <= 1e-6 + expected.abs() * 1e-5
}

fn loss_means_match(observed: &FoundationV2LossMeans, expected: &FoundationV2LossMeans) -> bool {
    let ordinary = [
        (observed.total, expected.total),
        (observed.pred_ce, expected.pred_ce),
        (observed.gate, expected.gate),
        (observed.latent, expected.latent),
        (observed.enc_ce, expected.enc_ce),
        (observed.separation, expected.separation),
        (observed.pull, expected.pull),
        (observed.inverse_action, expected.inverse_action),
        (observed.ep, expected.ep),
        (observed.rollout, expected.rollout),
        (observed.event, expected.event),
        (observed.q, expected.q),
        (observed.reliability, expected.reliability),
        (observed.gradient_clip_scale, expected.gradient_clip_scale),
        (observed.clipped_fraction, expected.clipped_fraction),
    ];
    ordinary
        .into_iter()
        .all(|(observed, expected)| approximately(observed, expected))
        && norm_approximately(
            observed.pre_clip_gradient_norm,
            expected.pre_clip_gradient_norm,
        )
}

fn route_norms_match(observed: &RouteNorms, expected: &RouteNorms) -> bool {
    observed.positive_pass == expected.positive_pass
        && observed.zero_pass == expected.zero_pass
        && observed.norms.len() == expected.norms.len()
        && observed.norms.iter().all(|(name, value)| {
            expected
                .norms
                .get(name)
                .is_some_and(|expected| norm_approximately(*value, *expected))
        })
}

fn load_loss_controls(g: &MultibatchScreenReport) -> Result<BTreeMap<usize, ScreenUpdateRecord>> {
    let path = Path::new(&g.output_root).join("loss_log.jsonl");
    let wanted = BTreeSet::from([1usize, 1025, 1537]);
    let mut controls = BTreeMap::new();
    for line in BufReader::new(fs::File::open(&path)?).lines() {
        let record: ScreenUpdateRecord = serde_json::from_str(&line?)?;
        if wanted.contains(&record.update.step) {
            controls.insert(record.update.step, record);
        }
    }
    ensure!(
        controls.len() == wanted.len(),
        "G loss log lacks frozen control rows"
    );
    Ok(controls)
}

#[allow(clippy::too_many_arguments)]
fn gradient_cell(
    snapshot_step: usize,
    batch_position: usize,
    ep_weight: f64,
    cfg: &TrainConfig,
    model: &WorldModel,
    varmap: &VarMap,
    main: &MixedStreamBatch,
    rollout: &MixedStreamBatch,
    host: &PreparedFoundationV2BatchHost,
    device: &Device,
    d2_predictions: &[Vec<u8>],
    controls: &BTreeMap<usize, ScreenUpdateRecord>,
    g: &MultibatchScreenReport,
) -> Result<(GradientCellReport, LogitCapture)> {
    let samples = main.samples();
    ensure!(
        d2_predictions.len() == REGISTERED_BATCH_SIZE,
        "D1 raw prediction row drift"
    );
    let objective = diagnostic_objective(cfg, ep_weight, snapshot_step, batch_position);
    let sigreg_seed = objective.sigreg_seed;
    let event_slot_weights = event_slot_weight_tensor(device)?;
    let (attached_rollout, rollout_fragments) =
        foundation_v2_dedicated_rollout_loss(model, rollout, device)?;
    ensure!(
        rollout_fragments == REGISTERED_ROLLOUT_FRAGMENTS,
        "diagnostic rollout fragment count drift"
    );
    let rollout_value = attached_rollout.detach();
    let weighted_rollout = attached_rollout.affine(cfg.rollout_weight, 0.0)?;
    let rollout_grads = retain_parameter_gradients(weighted_rollout.backward()?, varmap)?;

    let mut losses = foundation_v2_training_loss_with_event_weights(
        model,
        main,
        host,
        device,
        objective,
        &event_slot_weights,
    )?;
    losses.rollout = rollout_value.clone();
    losses.rollout_fragments = rollout_fragments;
    let per_pixel = losses
        .pred_per_pixel
        .as_ref()
        .context("diagnostic per-pixel CE was not captured")?;
    let training_logits = losses
        .diagnostic_predicted_logits
        .as_ref()
        .context("diagnostic training logits were not captured")?;
    let capture = capture_logits(training_logits)?;
    let (masks, mask_binding) = bind_prediction_masks(samples, &capture)?;
    ensure!(
        losses.changed_weights.changed_pixels == masks.changed_count
            && losses.changed_weights.unchanged_pixels == masks.unchanged_count
            && losses.changed_weights.changed_weight == 50.0,
        "CurrentDouble changed/unchanged geometry drift"
    );
    let prediction_logits_fingerprint = tensor_fingerprint(training_logits)?;
    let false_edit_mask_sha256 = prediction_mask_sha256(&masks.false_edit)?;
    let training_raw_argmax_disagreement_pixels = capture
        .predictions
        .iter()
        .zip(d2_predictions)
        .map(|(training, raw)| {
            training
                .iter()
                .zip(raw)
                .filter(|(left, right)| left != right)
                .count()
        })
        .sum::<usize>();

    let changed_loss = masked_prediction_loss(
        per_pixel,
        masks.changed,
        masks.changed_count,
        losses.changed_weights.changed_weight,
    )?;
    let unchanged_loss =
        masked_prediction_loss(per_pixel, masks.unchanged, masks.unchanged_count, 1.0)?;
    let false_edit_loss =
        masked_prediction_loss(per_pixel, masks.false_edit, masks.unchanged_count, 1.0)?;

    let prediction_grads = retain_parameter_gradients(losses.pred_ce.backward()?, varmap)?;
    let changed_grads = retain_parameter_gradients(changed_loss.backward()?, varmap)?;
    let unchanged_grads = retain_parameter_gradients(unchanged_loss.backward()?, varmap)?;
    let false_edit_grads = retain_parameter_gradients(false_edit_loss.backward()?, varmap)?;

    let weighted_components = vec![
        ("gate", 0.5, losses.gate.affine(0.5, 0.0)?),
        ("latent", 0.25, losses.latent.affine(0.25, 0.0)?),
        ("enc_ce", 0.1, losses.enc_ce.affine(0.1, 0.0)?),
        ("separation", 0.2, losses.separation.affine(0.2, 0.0)?),
        ("pull", 0.1, losses.pull.affine(0.1, 0.0)?),
        (
            "inverse_action",
            0.1,
            losses.inverse_action.affine(0.1, 0.0)?,
        ),
        ("ep", ep_weight, losses.ep.affine(ep_weight, 0.0)?),
        ("event", 0.1, losses.event.affine(0.1, 0.0)?),
        ("q", 0.1, losses.q.affine(0.1, 0.0)?),
        ("reliability", 0.1, losses.reliability.affine(0.1, 0.0)?),
    ];
    let mut component_stores = Vec::with_capacity(weighted_components.len() + 1);
    for (name, weight, tensor) in weighted_components {
        component_stores.push((
            name,
            weight,
            retain_parameter_gradients(tensor.backward()?, varmap)?,
        ));
    }
    component_stores.push(("rollout", cfg.rollout_weight, rollout_grads));

    let mut auxiliary_sum = None;
    for (_, _, grads) in &component_stores {
        accumulate_clone(&mut auxiliary_sum, grads, varmap)?;
    }
    let auxiliary_grads = auxiliary_sum.context("auxiliary gradient sum is empty")?;
    let mut reconstructed_full = None;
    accumulate_clone(&mut reconstructed_full, &prediction_grads, varmap)?;
    accumulate_clone(&mut reconstructed_full, &auxiliary_grads, varmap)?;
    let reconstructed_full = reconstructed_full.context("full reconstruction is empty")?;

    let mut direct_full = None;
    accumulate_parameter_gradients(
        &mut direct_full,
        retain_parameter_gradients(losses.total.backward()?, varmap)?,
        varmap,
    )?;
    let rollout_for_direct = component_stores
        .last()
        .map(|(_, _, grads)| grads)
        .context("missing rollout gradient")?;
    accumulate_clone(&mut direct_full, rollout_for_direct, varmap)?;
    let direct_full = direct_full.context("direct full gradient is empty")?;

    let mut reconstructed_prediction = None;
    accumulate_clone(&mut reconstructed_prediction, &changed_grads, varmap)?;
    accumulate_clone(&mut reconstructed_prediction, &unchanged_grads, varmap)?;
    let reconstructed_prediction =
        reconstructed_prediction.context("prediction reconstruction is empty")?;
    let full_reconstruction = reconstruction_check(&direct_full, &reconstructed_full, varmap)?;
    let prediction_reconstruction =
        reconstruction_check(&prediction_grads, &reconstructed_prediction, varmap)?;
    // The ignored characterization stops here in a test executable so it can
    // observe both pass and fail outcomes without changing production admission.
    #[cfg(test)]
    if std::env::var("TOFY_FROZEN_RECONSTRUCTION_PROBE").as_deref() == Ok("1") {
        anyhow::bail!(
            "TOFY_FROZEN_RECONSTRUCTION {}",
            serde_json::json!({
                "full": full_reconstruction,
                "prediction": prediction_reconstruction,
                "logits": capture.fingerprint,
                "mask_binding": mask_binding,
                "false_edit_pixels": masks.false_edit_count,
            })
        );
    }
    ensure_reconstructions(&full_reconstruction, &prediction_reconstruction)?;

    let full_stats =
        foundation_v2_gradient_route_stats(&direct_full, Some(&prediction_grads), varmap)?;
    let prediction_stats =
        foundation_v2_gradient_route_stats(&prediction_grads, Some(&prediction_grads), varmap)?;
    let auxiliary_stats =
        foundation_v2_gradient_route_stats(&auxiliary_grads, Some(&prediction_grads), varmap)?;
    let mut components = Vec::with_capacity(component_stores.len() + 3);
    components.push(component_report(
        "pred_ce",
        1.0,
        &prediction_grads,
        &prediction_grads,
        varmap,
        &full_stats,
        &prediction_stats,
    )?);
    for (name, weight, grads) in &component_stores {
        components.push(component_report(
            name,
            *weight,
            grads,
            &prediction_grads,
            varmap,
            &full_stats,
            &prediction_stats,
        )?);
    }
    components.push(component_report(
        "g_aux",
        1.0,
        &auxiliary_grads,
        &prediction_grads,
        varmap,
        &full_stats,
        &prediction_stats,
    )?);
    components.push(component_report(
        "g_full",
        1.0,
        &direct_full,
        &prediction_grads,
        varmap,
        &full_stats,
        &prediction_stats,
    )?);
    ensure!(
        components.iter().all(all_zero_prefixes),
        "a registered zero prefix received gradient pressure"
    );

    let pre_clip_norm = full_stats.global_l2;
    let clip_scale = if pre_clip_norm > 1.0 {
        1.0 / pre_clip_norm
    } else {
        1.0
    };
    let logged_total = losses
        .total
        .add(&rollout_value.affine(cfg.rollout_weight, 0.0)?)?;
    let observed_losses =
        foundation_v2_loss_values(&losses, &logged_total, pre_clip_norm, clip_scale)?;
    let loss_binding = if batch_position == 0 && matches!(snapshot_step, 0 | 1024 | 1536) {
        let expected_step = snapshot_step + 1;
        let expected = controls
            .get(&expected_step)
            .with_context(|| format!("missing G loss control {expected_step}"))?;
        let route_passed = if snapshot_step == 0 {
            let expected_route = &g
                .route_premise
                .as_ref()
                .context("G lacks step-0 route premise")?
                .prediction_unclipped;
            route_norms_match(&route_norms(&prediction_grads, varmap)?, expected_route)
        } else {
            true
        };
        let passed = expected.train_batch_position == batch_position
            && expected.update.sigreg_seed == sigreg_seed
            && expected.update.rollout_fragments == rollout_fragments
            && approximately(expected.update.ep_weight, ep_weight)
            && loss_means_match(&observed_losses, &expected.update.losses)
            && route_passed;
        ensure!(
            passed,
            "loss/route binding failed at checkpoint {snapshot_step}"
        );
        Some(LossBinding {
            expected_step: expected_step as u64,
            passed,
        })
    } else {
        None
    };

    let false_edit_share_of_prediction = safe_ratio(
        gradient_l2_for_optimizer_route(&false_edit_grads, varmap, OptimizerRoute::All)?,
        prediction_stats.global_l2,
    );
    let report = GradientCellReport {
        snapshot_step,
        batch_position,
        sigreg_seed,
        ep_weight,
        false_edit_pixels: masks.false_edit_count,
        mask_population: PRIMARY_POPULATION.into(),
        prediction_logits_fingerprint: Some(prediction_logits_fingerprint),
        false_edit_mask_sha256,
        mask_binding: Some(mask_binding),
        training_raw_argmax_disagreement_pixels,
        losses: observed_losses,
        components,
        prediction: route_geometry(&prediction_stats, &full_stats, &prediction_stats),
        auxiliary: route_geometry(&auxiliary_stats, &full_stats, &prediction_stats),
        combined: route_geometry(&full_stats, &full_stats, &prediction_stats),
        full_reconstruction,
        prediction_reconstruction,
        ns_muon_cosine_full_to_prediction: ns_muon_cosine(&direct_full, &prediction_grads, varmap)?,
        false_edit_to_auxiliary_cosine: bounded_cosine(
            &false_edit_grads,
            &auxiliary_grads,
            varmap,
            OptimizerRoute::All,
        )?,
        false_edit_to_prediction_cosine: bounded_cosine(
            &false_edit_grads,
            &prediction_grads,
            varmap,
            OptimizerRoute::All,
        )?,
        changed_to_unchanged_cosine: bounded_cosine(
            &changed_grads,
            &unchanged_grads,
            varmap,
            OptimizerRoute::All,
        )?,
        false_edit_share_of_prediction,
        loss_binding,
    };
    ensure_same_forward_binding(&report)?;
    Ok((report, capture))
}

fn selected_samples(batches: &[MixedStreamBatch], positions: &[usize]) -> Result<Vec<V5Sample>> {
    let mut samples = Vec::with_capacity(positions.len() * REGISTERED_BATCH_SIZE);
    for position in positions {
        let batch = batches
            .get(*position)
            .with_context(|| format!("missing selected batch position {position}"))?;
        ensure!(
            batch.samples().len() == REGISTERED_BATCH_SIZE
                && batch.factual_group_ranges().len() == 1
                && batch.factual_group_ranges()[0] == (43..53),
            "selected batch geometry drift"
        );
        samples.extend_from_slice(batch.samples());
    }
    Ok(samples)
}

fn ensure_full_frame_metric_domain(
    population: &MultibatchPopulation,
    spec: &FrozenDiagnosticSpec,
) -> Result<()> {
    for (split, batches, positions) in [
        (
            "train",
            population.train_main.as_slice(),
            spec.train_batch_positions.as_slice(),
        ),
        (
            "heldout",
            population.heldout_main.as_slice(),
            spec.heldout_batch_positions.as_slice(),
        ),
    ] {
        for position in positions {
            let batch = batches
                .get(*position)
                .with_context(|| format!("missing {split} metric batch position {position}"))?;
            ensure!(
                batch.samples().iter().all(|sample| {
                    sample.content_mask.values.len() == FRAME_SIDE * FRAME_SIDE
                        && sample.content_mask.values.iter().all(|value| *value != 0)
                }),
                "{split} batch {position} content mask does not match the frozen full-frame D2/D3 metric domain"
            );
        }
    }
    Ok(())
}

fn partial_group_matches(
    scores: &[PerBatchScore],
    positions: &[usize],
    expected: &crate::p2::multibatch_screen::UnionSnapshotMetrics,
) -> bool {
    scores.len() == positions.len()
        && scores.iter().zip(positions).all(|(score, position)| {
            let Some(group) = expected.group_routing.groups.get(*position) else {
                return false;
            };
            let Some(action6) = expected.action6.groups.get(*position) else {
                return false;
            };
            score.action_routed == group.action_routed
                && score.raw_full_exact_branches == group.raw_full_exact_branches
                && score.reproduced_distinct_changed_classes
                    == group.reproduced_distinct_changed_classes
                && score.action6_changed_full_exact == action6.raw_full_exact_changed_action6_rows
                && score.action6_coordinate_routed == action6.coordinate_routed
        })
}

struct ScoredSnapshot {
    report: SnapshotRescore,
    primary_train_captures: Vec<LogitCapture>,
    legacy_train_captures: Vec<LogitCapture>,
    comparisons: Vec<DiagnosticLogitComparison>,
    comparison_seconds: f64,
    train_keys: Option<BTreeSet<u64>>,
    heldout_keys: Option<BTreeSet<u64>>,
}

struct SplitScore {
    scores: Vec<PerBatchScore>,
    captures: Vec<LogitCapture>,
    anatomy: Option<AnatomyWithKeys>,
    anatomy_by_batch: Vec<FalseEditAnatomy>,
}

fn finish_split_score(
    samples: &[V5Sample],
    positions: &[usize],
    index_offset: u64,
    population: &str,
    captures: Vec<LogitCapture>,
    anatomy_enabled: bool,
) -> Result<SplitScore> {
    let predictions = captures
        .iter()
        .flat_map(|capture| capture.predictions.iter().cloned())
        .collect::<Vec<_>>();
    let scores = per_batch_scores(samples, &predictions, positions, index_offset, population)?;
    let (anatomy, anatomy_by_batch) = if anatomy_enabled {
        ensure!(
            population == PRIMARY_POPULATION,
            "legacy scorer cannot produce anatomy"
        );
        let host_logits = captures
            .iter()
            .flat_map(|capture| capture.values.iter().copied())
            .collect::<Vec<_>>();
        let row_stride = FRAME_SIDE * FRAME_SIDE * PALETTE_SIZE;
        let mut by_batch = Vec::with_capacity(positions.len());
        for (slot, position) in positions.iter().enumerate() {
            let start = slot * REGISTERED_BATCH_SIZE;
            let end = start + REGISTERED_BATCH_SIZE;
            by_batch.push(
                anatomy(
                    &samples[start..end],
                    &predictions[start..end],
                    &host_logits[start * row_stride..end * row_stride],
                    &[*position],
                )?
                .report,
            );
        }
        (
            Some(anatomy(samples, &predictions, &host_logits, positions)?),
            by_batch,
        )
    } else {
        (None, Vec::new())
    };
    Ok(SplitScore {
        scores,
        captures,
        anatomy,
        anatomy_by_batch,
    })
}

fn diagnostic_comparison(
    snapshot_step: usize,
    batch_position: usize,
    split: &str,
    comparison: &str,
    left: &LogitCapture,
    right: &LogitCapture,
) -> Result<DiagnosticLogitComparison> {
    let right_population = if comparison == "d1_primary_vs_d2_primary" {
        PRIMARY_POPULATION
    } else {
        LEGACY_POPULATION
    };
    Ok(DiagnosticLogitComparison {
        snapshot_step,
        batch_position,
        split: split.into(),
        comparison: comparison.into(),
        left_logits: left.fingerprint.clone(),
        right_logits: right.fingerprint.clone(),
        result: compare_logit_captures(PRIMARY_POPULATION, left, right_population, right)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_frozen_snapshot(
    step: usize,
    spec: &FrozenDiagnosticSpec,
    anatomy_enabled: bool,
    population: &MultibatchPopulation,
    expected: &ScreenSnapshot,
    cfg: &TrainConfig,
    model: &WorldModel,
    device: &Device,
    train_hosts: &[PreparedFoundationV2BatchHost],
    heldout_hosts: &[PreparedFoundationV2BatchHost],
) -> Result<ScoredSnapshot> {
    let started = Instant::now();
    let train_positions = spec.train_batch_positions.as_slice();
    let heldout_positions = spec.heldout_batch_positions.as_slice();
    let train_samples = selected_samples(&population.train_main, train_positions)?;
    let heldout_samples = selected_samples(&population.heldout_main, heldout_positions)?;
    let score_split =
        |samples: &[V5Sample], positions: &[usize], index_offset: u64| -> Result<SplitScore> {
            let transitions = samples
                .iter()
                .map(|sample| sample.transition.clone())
                .collect::<Vec<_>>();
            let logits = raw_one_step_logits(model, &transitions, device)?;
            let captures = (0..positions.len())
                .map(|slot| {
                    capture_logits(&logits.narrow(
                        0,
                        slot * REGISTERED_BATCH_SIZE,
                        REGISTERED_BATCH_SIZE,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            finish_split_score(
                samples,
                positions,
                index_offset,
                LEGACY_POPULATION,
                captures,
                false,
            )
        };
    let train_split = score_split(&train_samples, train_positions, 0)?;
    let heldout_split = score_split(&heldout_samples, heldout_positions, TRAIN_BATCHES as u64)?;
    let event_slot_weights = event_slot_weight_tensor(device)?;
    let score_primary_split = |samples: &[V5Sample],
                               batches: &[MixedStreamBatch],
                               hosts: &[PreparedFoundationV2BatchHost],
                               positions: &[usize],
                               index_offset: u64|
     -> Result<SplitScore> {
        let mut captures = Vec::with_capacity(positions.len());
        for position in positions {
            let batch = batches
                .get(*position)
                .with_context(|| format!("missing primary batch position {position}"))?;
            let host = hosts
                .get(*position)
                .with_context(|| format!("missing primary host position {position}"))?;
            let logits = training_seam_logits(
                diagnostic_objective(cfg, expected.ep_weight, step, *position),
                model,
                batch,
                host,
                device,
                &event_slot_weights,
            )?;
            captures.push(capture_logits(&logits)?);
        }
        finish_split_score(
            samples,
            positions,
            index_offset,
            PRIMARY_POPULATION,
            captures,
            anatomy_enabled,
        )
    };
    let primary_train_split = score_primary_split(
        &train_samples,
        &population.train_main,
        train_hosts,
        train_positions,
        0,
    )?;
    let primary_heldout_split = score_primary_split(
        &heldout_samples,
        &population.heldout_main,
        heldout_hosts,
        heldout_positions,
        TRAIN_BATCHES as u64,
    )?;
    let train = train_split.scores;
    let primary_train = primary_train_split.scores;
    let train_anatomy = primary_train_split.anatomy;
    let train_anatomy_by_batch = primary_train_split.anatomy_by_batch;
    let heldout = heldout_split.scores;
    let primary_heldout = primary_heldout_split.scores;
    let heldout_anatomy = primary_heldout_split.anatomy;
    let heldout_anatomy_by_batch = primary_heldout_split.anatomy_by_batch;
    let complete_union_binding = train_positions.len() == TRAIN_BATCHES
        && heldout_positions.len() == TRAIN_BATCHES
        && additive_matches(&train, &expected.train.raw)
        && additive_matches(&heldout, &expected.heldout.raw)
        && group_matches(&train, &expected.train)
        && group_matches(&heldout, &expected.heldout);
    if train_positions.len() == TRAIN_BATCHES && heldout_positions.len() == TRAIN_BATCHES {
        ensure!(
            complete_union_binding,
            "D2 union binding failed at snapshot {step}"
        );
    } else {
        ensure!(
            partial_group_matches(&train, train_positions, &expected.train)
                && partial_group_matches(&heldout, heldout_positions, &expected.heldout),
            "preflight group binding failed at snapshot {step}"
        );
    }
    if let Some(anatomy) = &train_anatomy {
        ensure!(
            anatomy.report.false_edit_pixels
                == primary_train
                    .iter()
                    .map(|score| score.raw.false_edit_pixels)
                    .sum::<usize>(),
            "train anatomy false-edit count differs from D2"
        );
    }
    if let Some(anatomy) = &heldout_anatomy {
        ensure!(
            anatomy.report.false_edit_pixels
                == primary_heldout
                    .iter()
                    .map(|score| score.raw.false_edit_pixels)
                    .sum::<usize>(),
            "held-out anatomy false-edit count differs from D2"
        );
    }
    ensure!(
        train_anatomy_by_batch.len()
            == if anatomy_enabled {
                train_positions.len()
            } else {
                0
            }
            && heldout_anatomy_by_batch.len()
                == if anatomy_enabled {
                    heldout_positions.len()
                } else {
                    0
                },
        "per-batch anatomy grid drift"
    );
    for (score, anatomy) in primary_train.iter().zip(&train_anatomy_by_batch) {
        ensure!(
            score.raw.false_edit_pixels == anatomy.false_edit_pixels,
            "train per-batch anatomy differs from D2"
        );
    }
    for (score, anatomy) in primary_heldout.iter().zip(&heldout_anatomy_by_batch) {
        ensure!(
            score.raw.false_edit_pixels == anatomy.false_edit_pixels,
            "held-out per-batch anatomy differs from D2"
        );
    }
    sync_cuda_device(device)?;
    let seconds = started.elapsed().as_secs_f64();
    let comparison_started = Instant::now();
    let mut comparisons = Vec::new();
    for (split, positions, primary, legacy) in [
        (
            "train",
            train_positions,
            &primary_train_split.captures,
            &train_split.captures,
        ),
        (
            "heldout",
            heldout_positions,
            &primary_heldout_split.captures,
            &heldout_split.captures,
        ),
    ] {
        for ((position, primary), legacy) in positions.iter().zip(primary).zip(legacy) {
            comparisons.push(diagnostic_comparison(
                step,
                *position,
                split,
                "d2_primary_vs_legacy",
                primary,
                legacy,
            )?);
        }
    }
    let comparison_seconds = comparison_started.elapsed().as_secs_f64();
    Ok(ScoredSnapshot {
        report: SnapshotRescore {
            step,
            legacy_population: "legacy_g_chunk32".into(),
            primary_population: "primary_same_forward".into(),
            anatomy_population: "primary_same_forward".into(),
            train,
            heldout,
            primary_train,
            primary_heldout,
            train_anatomy: train_anatomy.as_ref().map(|value| value.report.clone()),
            heldout_anatomy: heldout_anatomy.as_ref().map(|value| value.report.clone()),
            train_anatomy_by_batch,
            heldout_anatomy_by_batch,
            complete_union_binding,
            seconds,
        },
        primary_train_captures: primary_train_split.captures,
        legacy_train_captures: train_split.captures,
        comparisons,
        comparison_seconds,
        train_keys: train_anatomy.map(|value| value.keys),
        heldout_keys: heldout_anatomy.map(|value| value.keys),
    })
}

fn validate_report_files(root: &Path, report: &FrozenDiagnosticReport) -> Result<()> {
    ensure!(
        file_sha256_hex(&root.join("train_config.json"))? == report.train_config_sha256,
        "copied train config hash drift"
    );
    ensure!(
        file_sha256_hex(&root.join("Cargo.lock"))? == report.cargo_lock_sha256,
        "copied Cargo.lock hash drift"
    );
    let census_path = root.join("population/census.json");
    ensure!(
        file_sha256_hex(&census_path)? == report.population_census_sha256,
        "diagnostic population census hash drift"
    );
    let stored: PopulationCensus = serde_json::from_slice(&fs::read(&census_path)?)?;
    ensure!(
        report.population.as_ref() == Some(&stored),
        "stored census differs from report"
    );
    stored.ensure_registered()?;
    for record in &stored.batches {
        ensure!(
            file_sha256_hex(&root.join(&record.file))? == record.population_sha256,
            "serialized diagnostic batch hash drift"
        );
    }
    Ok(())
}

fn ensure_comparison_grid(report: &FrozenDiagnosticReport) -> Result<()> {
    let mut expected = BTreeSet::new();
    for snapshot in &report.rescoring {
        for (split, positions) in [
            ("train", &report.spec.train_batch_positions),
            ("heldout", &report.spec.heldout_batch_positions),
        ] {
            for position in positions {
                expected.insert((snapshot.step, *position, split, "d2_primary_vs_legacy"));
            }
        }
    }
    for cell in &report.gradients {
        for name in ["d1_primary_vs_d2_primary", "d1_primary_vs_legacy"] {
            expected.insert((cell.snapshot_step, cell.batch_position, "train", name));
        }
    }
    let observed = report
        .comparisons
        .iter()
        .map(|row| {
            (
                row.snapshot_step,
                row.batch_position,
                row.split.as_str(),
                row.comparison.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    ensure!(
        observed == expected && observed.len() == report.comparisons.len(),
        "diagnostic descriptive comparison grid drift"
    );
    for row in &report.comparisons {
        let right_population = if row.comparison == "d1_primary_vs_d2_primary" {
            PRIMARY_POPULATION
        } else {
            LEGACY_POPULATION
        };
        ensure!(
            row.result.left == PRIMARY_POPULATION && row.result.right == right_population,
            "diagnostic descriptive comparison population drift"
        );
        if row.comparison.starts_with("d1_") {
            let cell = report
                .gradients
                .iter()
                .find(|cell| {
                    cell.snapshot_step == row.snapshot_step
                        && cell.batch_position == row.batch_position
                })
                .context("comparison lacks D1 cell")?;
            ensure!(
                cell.prediction_logits_fingerprint.as_ref() == Some(&row.left_logits),
                "descriptive D1 comparison fingerprint is not bound to its cell"
            );
        }
        // Numerical disagreement is descriptive, never an equality premise.
    }
    Ok(())
}

fn ensure_completed_cleanly(report: &FrozenDiagnosticReport) -> Result<()> {
    ensure!(
        report.schema == FROZEN_DIAGNOSTIC_SCHEMA
            && report.evidence_class == DIAGNOSTIC_EVIDENCE_CLASS
            && report.lifecycle.evidence_class == DIAGNOSTIC_EVIDENCE_CLASS
            && report.lifecycle.state == LIFECYCLE_COMPLETE
            && report.lifecycle.run_class == report.run_class
            && report.error.is_none(),
        "diagnostic did not complete cleanly"
    );
    ensure!(
        report.run_class
            == if report.registered {
                RUN_CLASS_REGISTERED
            } else {
                RUN_CLASS_PREFLIGHT
            }
            && report.device_is_cuda
            && report.gpu_identity.is_some()
            && !report.research_claim
            && !report.public_data_read,
        "diagnostic run class or data boundary drift"
    );
    report.spec.validate(report.registered)?;
    ensure!(
        report.g.is_some()
            && report.g_identity == REGISTERED_G_IDENTITY
            && report.population.is_some()
            && !report.identity_root.is_empty(),
        "diagnostic lacks frozen evidence bindings"
    );
    ensure!(
        report.gradients.iter().all(|cell| {
            cell.full_reconstruction.passed
                && cell.prediction_reconstruction.passed
                && cell.components.len() == 14
                && cell.components.iter().all(all_zero_prefixes)
        }),
        "diagnostic gradient integrity failed"
    );
    for cell in &report.gradients {
        ensure_same_forward_binding(cell)?;
    }
    ensure_comparison_grid(report)?;
    ensure!(
        report.rescoring.iter().all(|snapshot| {
            snapshot.legacy_population == "legacy_g_chunk32"
                && snapshot.primary_population == "primary_same_forward"
                && snapshot.anatomy_population == "primary_same_forward"
                && snapshot.primary_train.len() == report.spec.train_batch_positions.len()
                && snapshot.primary_heldout.len() == report.spec.heldout_batch_positions.len()
                && snapshot
                    .primary_train
                    .iter()
                    .chain(&snapshot.primary_heldout)
                    .all(|score| score.population == PRIMARY_POPULATION)
                && snapshot
                    .train
                    .iter()
                    .chain(&snapshot.heldout)
                    .all(|score| score.population == LEGACY_POPULATION)
                && snapshot
                    .train_anatomy
                    .iter()
                    .chain(&snapshot.heldout_anatomy)
                    .chain(&snapshot.train_anatomy_by_batch)
                    .chain(&snapshot.heldout_anatomy_by_batch)
                    .all(|anatomy| anatomy.population == PRIMARY_POPULATION)
        }),
        "diagnostic score population contract failed"
    );
    let expected_cells = if report.registered { 25 } else { 2 };
    ensure!(
        report.gradients.len() == expected_cells,
        "diagnostic gradient grid is incomplete"
    );
    let expected_grid = if report.registered {
        std::iter::once((0usize, 0usize))
            .chain(
                GRADIENT_STEPS
                    .into_iter()
                    .flat_map(|step| (0..TRAIN_BATCHES).map(move |position| (step, position))),
            )
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::from([(0usize, 0usize), (1024, 0)])
    };
    let observed_grid = report
        .gradients
        .iter()
        .map(|cell| (cell.snapshot_step, cell.batch_position))
        .collect::<BTreeSet<_>>();
    ensure!(
        observed_grid == expected_grid,
        "diagnostic gradient grid drift"
    );
    let step_zero_prediction = report
        .gradients
        .iter()
        .find(|cell| cell.snapshot_step == 0 && cell.batch_position == 0)
        .and_then(|cell| {
            cell.components
                .iter()
                .find(|row| row.objective == "pred_ce")
        })
        .context("diagnostic lacks step-0 prediction component")?;
    ensure!(
        POSITIVE_PREFIXES.iter().all(|prefix| {
            step_zero_prediction
                .prefixes
                .get(*prefix)
                .is_some_and(|norm| norm.state == "present" && norm.l2.is_some_and(|v| v > 0.0))
        }),
        "step-0 prediction-positive route premise failed"
    );
    let bound_losses = report
        .gradients
        .iter()
        .filter_map(|cell| cell.loss_binding.as_ref())
        .collect::<Vec<_>>();
    let expected_loss_bindings = if report.registered { 3 } else { 2 };
    ensure!(
        bound_losses.len() == expected_loss_bindings
            && bound_losses.iter().all(|binding| binding.passed),
        "diagnostic loss controls are incomplete"
    );
    if report.registered {
        let rescore_steps = report
            .rescoring
            .iter()
            .map(|snapshot| snapshot.step)
            .collect::<Vec<_>>();
        ensure!(
            report.preflight.is_some()
                && report
                    .runtime_estimate
                    .as_ref()
                    .is_some_and(|estimate| estimate.admitted)
                && report.rescoring.len() == SNAPSHOT_STEPS.len()
                && rescore_steps == SNAPSHOT_STEPS
                && report
                    .rescoring
                    .iter()
                    .all(|snapshot| snapshot.complete_union_binding)
                && report.persistence.len() == 2
                && report.verdict.is_some(),
            "registered diagnostic is incomplete"
        );
        ensure!(
            report.rescoring.iter().all(|snapshot| {
                if ANATOMY_STEPS.contains(&snapshot.step) {
                    snapshot.train_anatomy.is_some()
                        && snapshot.heldout_anatomy.is_some()
                        && snapshot.train_anatomy_by_batch.len() == TRAIN_BATCHES
                        && snapshot.heldout_anatomy_by_batch.len() == TRAIN_BATCHES
                } else {
                    snapshot.train_anatomy.is_none()
                        && snapshot.heldout_anatomy.is_none()
                        && snapshot.train_anatomy_by_batch.is_empty()
                        && snapshot.heldout_anatomy_by_batch.is_empty()
                }
            }),
            "registered D3 anatomy grid is incomplete"
        );
        ensure!(
            report.verdict.as_ref()
                == Some(&classify(
                    &report.gradients,
                    &report.rescoring,
                    &report.persistence
                )?),
            "registered diagnostic verdict does not reproduce"
        );
    } else {
        let rescore_steps = report
            .rescoring
            .iter()
            .map(|snapshot| snapshot.step)
            .collect::<Vec<_>>();
        let final_rescore = report
            .rescoring
            .iter()
            .find(|snapshot| snapshot.step == 2048)
            .context("preflight lacks step-2048 D2/D3 control")?;
        ensure!(
            report.preflight.is_none()
                && rescore_steps == [0, 1024, 2048]
                && report
                    .rescoring
                    .iter()
                    .all(|snapshot| !snapshot.complete_union_binding)
                && report
                    .rescoring
                    .iter()
                    .filter(|snapshot| snapshot.step != 2048)
                    .all(|snapshot| {
                        snapshot.train_anatomy.is_none()
                            && snapshot.heldout_anatomy.is_none()
                            && snapshot.train_anatomy_by_batch.is_empty()
                            && snapshot.heldout_anatomy_by_batch.is_empty()
                    })
                && final_rescore.train_anatomy.is_some()
                && final_rescore.heldout_anatomy.is_some()
                && final_rescore.train_anatomy_by_batch.len() == 1
                && final_rescore.heldout_anatomy_by_batch.len() == 1
                && report.persistence.is_empty()
                && report.verdict.is_none()
                && report.runtime_estimate.as_ref() == Some(&runtime_estimate(report)?),
            "diagnostic preflight is incomplete"
        );
    }
    ensure!(
        report.timing.population_seconds.is_finite()
            && report.timing.population_seconds >= 0.0
            && report.timing.gradient_cell_seconds.len() == expected_cells
            && report.timing.rescore_seconds.len() == report.rescoring.len()
            && report.timing.wall_seconds > 0.0
            && report.timing.wall_seconds <= report.spec.max_wall_seconds as f64,
        "diagnostic timing contract failed"
    );
    Ok(())
}

fn write_progress(root: &Path, report: &FrozenDiagnosticReport) -> Result<()> {
    write_json_report(&root.join("progress.json"), report)
}

fn checked_snapshot_path(root: &Path, snapshot: &ScreenSnapshot) -> Result<PathBuf> {
    ensure!(
        !snapshot.raw_checkpoint.is_absolute()
            && !snapshot
                .raw_checkpoint
                .components()
                .any(|component| { matches!(component, std::path::Component::ParentDir) }),
        "G snapshot path escapes its root"
    );
    let path = fs::canonicalize(root.join(&snapshot.raw_checkpoint))?;
    ensure!(
        path.starts_with(root),
        "G snapshot resolves outside its root"
    );
    ensure!(
        file_sha256_hex(&path)? == snapshot.raw_sha256,
        "G snapshot hash drift"
    );
    Ok(path)
}

fn run_inner(
    args: &P2MultibatchFrozenDiagnosticArgs,
    report: &mut FrozenDiagnosticReport,
    started: Instant,
) -> Result<()> {
    let root = args.output_root.as_path();
    report.spec.validate(args.registered)?;
    ensure!(
        args.device.trim().starts_with("cuda"),
        "frozen diagnostic requires CUDA"
    );
    registered_provenance_guard(&report.provenance)?;
    ensure_source_descends_from_g(&report.provenance)?;
    if args.registered {
        ensure!(
            args.preflight_report.is_some(),
            "registered diagnostic requires --preflight-report"
        );
    } else {
        ensure!(
            args.preflight_report.is_none(),
            "preflight cannot bind another preflight"
        );
    }

    let (g, g_binding) = bind_g(&args.g_report)?;
    report.g_identity = g_binding.identity_root.clone();
    report.g = Some(g_binding.clone());
    let config_path = g_binding.root.join("train_config.json");
    let config_bytes = fs::read(&config_path)?;
    report.train_config_sha256 = file_sha256_hex(&config_path)?;
    ensure!(
        report.train_config_sha256 == REGISTERED_TRAIN_CONFIG_SHA256
            && report.train_config_sha256 == g.train_config_sha256,
        "diagnostic train config drift"
    );
    fs::write(root.join("train_config.json"), &config_bytes)?;
    let cfg = load_train_config(&config_path)?;
    cfg.validate()?;
    ensure_registered_config(&cfg)?;
    ensure!(
        cfg.seed == REGISTERED_SEED
            && cfg.physical_batch == REGISTERED_BATCH_SIZE
            && cfg.data_contract_v6
            && args.device == cfg.device,
        "diagnostic config population or device drift"
    );

    let cargo_lock = fs::canonicalize(std::env::current_dir()?.join("Cargo.lock"))?;
    report.cargo_lock_sha256 = file_sha256_hex(&cargo_lock)?;
    ensure!(
        report.cargo_lock_sha256 == REGISTERED_G_CARGO_LOCK_SHA256,
        "diagnostic Cargo.lock differs from registered G"
    );
    fs::copy(&cargo_lock, root.join("Cargo.lock"))?;

    let population_started = Instant::now();
    let population =
        MultibatchPopulation::compose(cfg.seed, cfg.physical_batch, cfg.data_contract_v6)?;
    ensure_full_frame_metric_domain(&population, &report.spec)?;
    let census = population.write(root)?;
    let census_path = root.join("population/census.json");
    report.population_census_sha256 = file_sha256_hex(&census_path)?;
    ensure!(
        g.population.as_ref() == Some(&census)
            && g.population_census_sha256.as_deref()
                == Some(report.population_census_sha256.as_str()),
        "recomposed diagnostic population differs from G"
    );
    report.population = Some(census);
    let hosts = population
        .train_main
        .iter()
        .map(prepare_foundation_v2_batch_host)
        .collect::<Result<Vec<_>>>()?;
    let heldout_hosts = population
        .heldout_main
        .iter()
        .map(prepare_foundation_v2_batch_host)
        .collect::<Result<Vec<_>>>()?;
    report.timing.population_seconds = population_started.elapsed().as_secs_f64();

    report.gpu_identity = Some(query_gpu_identity(&args.device)?);
    ensure!(
        report.gpu_identity == g.gpu_identity,
        "diagnostic GPU identity differs from G"
    );
    let bound_preflight = if let Some(path) = args.preflight_report.as_deref() {
        let (preflight, binding) = bind_diagnostic_report(path)?;
        report.runtime_estimate = Some(ensure_preflight_binds(&preflight, report)?);
        report.preflight = Some(binding);
        Some(preflight)
    } else {
        None
    };
    report.identity_root = diagnostic_identity(report)?;
    write_progress(root, report)?;

    let _pid_guard = TrainPidGuard::install(root)?;
    let diagnostic_device = open_diagnostic_device(&args.device, root)?;
    ensure!(
        diagnostic_device.device.is_cuda() && diagnostic_device.gpu_identity == report.gpu_identity,
        "diagnostic CUDA device identity changed during open"
    );
    report.device_is_cuda = true;
    let device = &diagnostic_device.device;
    let varmap = VarMap::new();
    let model = WorldModel::new(
        cfg.model_config(),
        VarBuilder::from_varmap(&varmap, DType::F32, device),
    )?;
    let controls = load_loss_controls(&g)?;
    let mut train_key_sets = BTreeMap::new();
    let mut heldout_key_sets = BTreeMap::new();
    let mut steps = report
        .spec
        .rescore_steps
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    steps.extend(report.spec.gradient_steps.iter().copied());
    if args.registered {
        steps.insert(0);
    }

    for step in steps {
        ensure!(
            started.elapsed() <= Duration::from_secs(report.spec.max_wall_seconds),
            "frozen diagnostic wall-time cap exceeded"
        );
        let expected = snapshot_by_step(&g, step)?;
        let checkpoint = checked_snapshot_path(&g_binding.root, expected)?;
        load_varmap_exact(&varmap, &checkpoint)?;
        ensure_operator_projection_zero(&varmap)?;

        let mut d2_train_captures = None;
        if report.spec.rescore_steps.contains(&step) {
            let anatomy_enabled = report.spec.anatomy_steps.contains(&step);
            let scored = score_frozen_snapshot(
                step,
                &report.spec,
                anatomy_enabled,
                &population,
                expected,
                &cfg,
                &model,
                device,
                &hosts,
                &heldout_hosts,
            )?;
            if let Some(preflight) = bound_preflight.as_ref() {
                if let Some(expected_control) = preflight_rescore_control(preflight, step) {
                    ensure_rescore_control_matches(expected_control, &scored.report)?;
                }
            }
            if let Some(keys) = scored.train_keys {
                train_key_sets.insert(step, keys);
            }
            if let Some(keys) = scored.heldout_keys {
                heldout_key_sets.insert(step, keys);
            }
            report.timing.rescore_seconds.push(scored.report.seconds);
            report
                .timing
                .comparison_seconds
                .push(scored.comparison_seconds);
            report.comparisons.extend(scored.comparisons);
            d2_train_captures = Some((scored.primary_train_captures, scored.legacy_train_captures));
            report.rescoring.push(scored.report);
            write_progress(root, report)?;
        }

        let gradient_positions = if step == 0 {
            vec![0]
        } else if report.spec.gradient_steps.contains(&step) {
            report.spec.train_batch_positions.clone()
        } else {
            Vec::new()
        };
        for position in gradient_positions {
            sync_cuda_device(device)?;
            let cell_started = Instant::now();
            let (d2_primary, d2_legacy) = d2_train_captures
                .as_ref()
                .context("D1 requires the frozen D2 raw-logit prediction seam")?;
            let slot = report
                .spec
                .train_batch_positions
                .iter()
                .position(|candidate| *candidate == position)
                .expect("gradient position belongs to D2 positions");
            let (cell, capture) = gradient_cell(
                step,
                position,
                expected.ep_weight,
                &cfg,
                &model,
                &varmap,
                &population.train_main[position],
                &population.train_rollout[position],
                &hosts[position],
                device,
                &d2_legacy[slot].predictions,
                &controls,
                &g,
            )?;
            if let Some(preflight) = bound_preflight.as_ref() {
                if position == 0 && matches!(step, 0 | 1024) {
                    let expected_control = preflight
                        .gradients
                        .iter()
                        .find(|cell| cell.snapshot_step == step && cell.batch_position == position)
                        .with_context(|| {
                            format!("preflight lacks D1 control at step {step} position {position}")
                        })?;
                    ensure_gradient_control_matches(expected_control, &cell)?;
                }
            }
            sync_cuda_device(device)?;
            report
                .timing
                .gradient_cell_seconds
                .push(cell_started.elapsed().as_secs_f64());
            let comparison_started = Instant::now();
            for (name, right) in [
                ("d1_primary_vs_d2_primary", &d2_primary[slot]),
                ("d1_primary_vs_legacy", &d2_legacy[slot]),
            ] {
                report.comparisons.push(diagnostic_comparison(
                    step, position, "train", name, &capture, right,
                )?);
            }
            report
                .timing
                .comparison_seconds
                .push(comparison_started.elapsed().as_secs_f64());
            report.gradients.push(cell);
            write_progress(root, report)?;
        }
    }

    if args.registered {
        report.persistence = vec![
            persistence("train", &train_key_sets)?,
            persistence("heldout", &heldout_key_sets)?,
        ];
        report.verdict = Some(classify(
            &report.gradients,
            &report.rescoring,
            &report.persistence,
        )?);
        ensure_preflight_controls_bind(
            bound_preflight
                .as_ref()
                .context("registered preflight disappeared")?,
            report,
        )?;
    } else {
        report.runtime_estimate = Some(runtime_estimate(report)?);
    }

    sync_cuda_device(device)?;
    drop(diagnostic_device);
    let (_, rebound_g) = bind_g(&args.g_report)?;
    ensure!(
        rebound_g == g_binding,
        "G evidence changed during diagnostic"
    );
    ensure!(
        file_sha256_hex(&config_path)? == report.train_config_sha256
            && file_sha256_hex(&cargo_lock)? == report.cargo_lock_sha256
            && file_sha256_hex(&census_path)? == report.population_census_sha256,
        "diagnostic input changed during device work"
    );
    let binary_after = format!(
        "sha256:{}",
        file_sha256_hex(&report.provenance.binary_path)?
    );
    ensure!(
        binary_after == report.provenance.binary_sha256,
        "diagnostic binary changed during device work"
    );
    ensure!(
        diagnostic_identity(report)? == report.identity_root,
        "diagnostic identity changed"
    );
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    validate_report_files(root, report)?;
    ensure!(
        started.elapsed() <= Duration::from_secs(report.spec.max_wall_seconds),
        "frozen diagnostic wall-time cap exceeded during validation"
    );
    Ok(())
}

pub fn run_p2_multibatch_frozen_diagnostic(args: P2MultibatchFrozenDiagnosticArgs) -> Result<()> {
    let started = Instant::now();
    let run_class = if args.registered {
        RUN_CLASS_REGISTERED
    } else {
        RUN_CLASS_PREFLIGHT
    };
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: DIAGNOSTIC_EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        note: "read-only V6 frozen-checkpoint diagnostic in progress".into(),
    };
    let command = open_run_root(&args.output_root, &lifecycle, COMMAND_TAG)?;
    let mut report = FrozenDiagnosticReport {
        schema: FROZEN_DIAGNOSTIC_SCHEMA.into(),
        evidence_class: DIAGNOSTIC_EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        registered: args.registered,
        research_claim: false,
        public_data_read: false,
        lifecycle,
        provenance: launch_provenance().clone(),
        command,
        device: args.device.clone(),
        device_is_cuda: false,
        gpu_identity: None,
        output_root: args.output_root.clone(),
        spec: FrozenDiagnosticSpec::new(args.registered),
        g: None,
        g_identity: String::new(),
        preflight: None,
        train_config_sha256: String::new(),
        cargo_lock_sha256: String::new(),
        population_census_sha256: String::new(),
        population: None,
        rescoring: Vec::new(),
        gradients: Vec::new(),
        comparisons: Vec::new(),
        persistence: Vec::new(),
        runtime_estimate: None,
        verdict: None,
        timing: FrozenDiagnosticTiming {
            population_seconds: 0.0,
            gradient_cell_seconds: Vec::new(),
            rescore_seconds: Vec::new(),
            comparison_seconds: Vec::new(),
            wall_seconds: 0.0,
        },
        identity_root: String::new(),
        error: None,
    };
    let mut outcome = run_inner(&args, &mut report, started);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    if report.identity_root.is_empty() && report.population.is_some() {
        match diagnostic_identity(&report) {
            Ok(identity) => report.identity_root = identity,
            Err(error) => outcome = Err(error.context("compute frozen diagnostic identity")),
        }
    }
    report.lifecycle = match &outcome {
        Ok(()) => LifecycleRecord {
            state: LIFECYCLE_COMPLETE.into(),
            unix_seconds: unix_seconds(),
            evidence_class: DIAGNOSTIC_EVIDENCE_CLASS.into(),
            run_class: run_class.into(),
            note: report.verdict.as_ref().map_or_else(
                || "same-binary diagnostic preflight complete; no model update or training authority"
                    .into(),
                |verdict| format!(
                    "{}; selection-only diagnosis; no model update or treatment launch authority",
                    verdict.auxiliary_class
                ),
            ),
        },
        Err(error) => {
            report.error = Some(format!("{error:#}"));
            report.evidence_class = FAILED_EVIDENCE_CLASS.into();
            LifecycleRecord {
                state: LIFECYCLE_FAILED.into(),
                unix_seconds: unix_seconds(),
                evidence_class: FAILED_EVIDENCE_CLASS.into(),
                run_class: run_class.into(),
                note: format!("{error:#}"),
            }
        }
    };
    if outcome.is_ok() {
        if let Err(error) = ensure_completed_cleanly(&report) {
            report.error = Some(format!("{error:#}"));
            report.evidence_class = FAILED_EVIDENCE_CLASS.into();
            report.lifecycle = LifecycleRecord {
                state: LIFECYCLE_FAILED.into(),
                unix_seconds: unix_seconds(),
                evidence_class: FAILED_EVIDENCE_CLASS.into(),
                run_class: run_class.into(),
                note: format!("{error:#}"),
            };
            outcome = Err(error.context("post-completion frozen diagnostic validation"));
        }
    }
    let lifecycle = report.lifecycle.clone();
    seal_run_root(&args.output_root, COMMAND_TAG, &report, &lifecycle)?;
    outcome
}

fn bind_failed_frozen_preflight(path: &Path) -> Result<EvidenceBinding> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize failed preflight {}", path.display()))?;
    let report: FrozenDiagnosticReport =
        serde_json::from_slice(&fs::read(&report_path)?).context("parse failed preflight")?;
    let root = fs::canonicalize(&report.output_root)?;
    ensure!(
        report_path == fs::canonicalize(root.join(REPORT_FILE))?,
        "failed preflight report is outside its claimed root"
    );
    let (manifest, _) = external_manifest_paths(&root)?;
    let manifest = fs::canonicalize(manifest)?;
    let manifest_sha256 = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &manifest_sha256)?;
    validate_report_files(&root, &report)?;
    ensure!(
        file_sha256_hex(&report_path)? == FAILED_PREFLIGHT_REPORT_SHA256
            && manifest_sha256 == FAILED_PREFLIGHT_MANIFEST_SHA256
            && report.identity_root == FAILED_PREFLIGHT_IDENTITY
            && report.identity_root == diagnostic_identity(&report)?,
        "failed preflight identity differs from the frozen characterization parent"
    );
    ensure!(
        report.lifecycle.state == LIFECYCLE_FAILED
            && report.evidence_class == FAILED_EVIDENCE_CLASS
            && report.device_is_cuda
            && report.provenance.source_revision == FAILED_PREFLIGHT_SOURCE
            && report.provenance.binary_sha256 == FAILED_PREFLIGHT_BINARY_SHA256
            && report.error.as_deref() == Some("training and raw prediction seams disagree")
            && report.gradients.is_empty()
            && report.rescoring.len() == 1,
        "failed preflight does not reproduce the frozen seam falsifier"
    );
    Ok(EvidenceBinding {
        report: report_path,
        root,
        manifest,
        manifest_sha256,
        identity_root: report.identity_root,
    })
}

fn seam_characterization_identity(report: &SeamCharacterizationReport) -> Result<String> {
    identity_frame_sha256(&[
        ("domain", report.schema.as_bytes().to_vec()),
        (
            "source",
            report.provenance.source_revision.as_bytes().to_vec(),
        ),
        (
            "binary",
            report.provenance.binary_sha256.as_bytes().to_vec(),
        ),
        ("cargo", report.cargo_lock_sha256.as_bytes().to_vec()),
        (
            "g",
            report
                .g
                .as_ref()
                .map(|binding| binding.identity_root.as_bytes().to_vec())
                .unwrap_or_default(),
        ),
        (
            "failed_preflight",
            report
                .failed_preflight
                .as_ref()
                .map(|binding| binding.identity_root.as_bytes().to_vec())
                .unwrap_or_default(),
        ),
        ("config", report.train_config_sha256.as_bytes().to_vec()),
        (
            "population",
            report.population_census_sha256.as_bytes().to_vec(),
        ),
        ("checkpoint", report.checkpoint_sha256.as_bytes().to_vec()),
        (
            "frozen_order",
            b"V5a,V5b,V4,V1a,V1b,V5c;step=0;train_batch=0;eval_chunk=32;full_chunk=128;wall=120"
                .to_vec(),
        ),
    ])
}

fn tensor_fingerprint(tensor: &Tensor) -> Result<TensorFingerprint> {
    let shape = tensor.dims().to_vec();
    let dtype = format!("{:?}", tensor.dtype());
    let flat = tensor.flatten_all()?.contiguous()?;
    let bytes = match tensor.dtype() {
        DType::U8 => flat.to_vec1::<u8>()?,
        DType::U32 => flat
            .to_vec1::<u32>()?
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        DType::F32 => flat
            .to_vec1::<f32>()?
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
        other => anyhow::bail!("unsupported characterization tensor dtype {other:?}"),
    };
    let sha256 = identity_frame_sha256(&[
        ("dtype", dtype.as_bytes().to_vec()),
        ("shape", serde_json::to_vec(&shape)?),
        ("values", bytes),
    ])?;
    Ok(TensorFingerprint {
        shape,
        dtype,
        sha256,
    })
}

fn compare_input_tensor(
    fields: &mut BTreeMap<String, TensorInputComparison>,
    name: &str,
    training: &Tensor,
    evaluator: &Tensor,
    active_for_predicted_latent: bool,
) -> Result<()> {
    let training = tensor_fingerprint(training)?;
    let evaluator = tensor_fingerprint(evaluator)?;
    let exact_equal = training == evaluator;
    fields.insert(
        name.into(),
        TensorInputComparison {
            training,
            evaluator,
            exact_equal,
            active_for_predicted_latent,
        },
    );
    Ok(())
}

fn compare_batch_inputs(
    training: &BatchTensors,
    evaluator: &BatchTensors,
) -> Result<InputComparisonReport> {
    let mut fields = BTreeMap::new();
    for (name, training, evaluator, active) in [
        ("frames", &training.frames, &evaluator.frames, true),
        (
            "next_frames",
            &training.next_frames,
            &evaluator.next_frames,
            true,
        ),
        (
            "model_frames",
            &training.model_frames,
            &evaluator.model_frames,
            false,
        ),
        (
            "model_next_frames",
            &training.model_next_frames,
            &evaluator.model_next_frames,
            false,
        ),
        ("actions", &training.actions, &evaluator.actions, true),
        (
            "action_coords",
            &training.action_coords,
            &evaluator.action_coords,
            true,
        ),
        ("goals", &training.goals, &evaluator.goals, false),
        (
            "event_targets",
            &training.event_targets,
            &evaluator.event_targets,
            false,
        ),
        (
            "event_mask",
            &training.event_mask,
            &evaluator.event_mask,
            false,
        ),
        (
            "operator_conditioning",
            &training.operator_conditioning,
            &evaluator.operator_conditioning,
            false,
        ),
    ] {
        compare_input_tensor(&mut fields, name, training, evaluator, active)?;
    }
    let training_context = training
        .context
        .as_ref()
        .context("training seam lacks frozen V6 context")?;
    let evaluator_context = evaluator
        .context
        .as_ref()
        .context("evaluator seam lacks frozen V6 context")?;
    ensure!(
        training_context.k() == evaluator_context.k()
            && training_context.current.dims() == evaluator_context.current.dims()
            && training_context.next.dims() == evaluator_context.next.dims()
            && training_context.actions.dims() == evaluator_context.actions.dims()
            && training_context.coords.dims() == evaluator_context.coords.dims()
            && training_context.valid.dims() == evaluator_context.valid.dims(),
        "training and evaluator context geometry differs"
    );
    for (name, training, evaluator) in [
        (
            "context.current",
            &training_context.current,
            &evaluator_context.current,
        ),
        (
            "context.next",
            &training_context.next,
            &evaluator_context.next,
        ),
        (
            "context.actions",
            &training_context.actions,
            &evaluator_context.actions,
        ),
        (
            "context.coords",
            &training_context.coords,
            &evaluator_context.coords,
        ),
        (
            "context.valid",
            &training_context.valid,
            &evaluator_context.valid,
        ),
    ] {
        compare_input_tensor(&mut fields, name, training, evaluator, true)?;
    }
    let active_inputs_equal = fields
        .values()
        .filter(|field| field.active_for_predicted_latent)
        .all(|field| field.exact_equal);
    let operator_conditioning_equal = fields
        .get("operator_conditioning")
        .is_some_and(|field| field.exact_equal);
    Ok(InputComparisonReport {
        fields,
        active_inputs_equal,
        operator_conditioning_equal,
        accepted_inert_operator_non_identity: !operator_conditioning_equal,
    })
}

#[derive(Debug)]
struct LogitCapture {
    fingerprint: TensorFingerprint,
    values: Vec<f32>,
    predictions: Vec<Vec<u8>>,
}

fn capture_logits(logits: &Tensor) -> Result<LogitCapture> {
    ensure!(
        logits.dims4()? == (REGISTERED_BATCH_SIZE, FRAME_SIDE, FRAME_SIDE, PALETTE_SIZE),
        "seam characterization logits have the wrong shape"
    );
    ensure!(
        logits.dtype() == DType::F32,
        "characterization logits are not F32"
    );
    let predictions = predictions_from_logits(logits)?;
    let fingerprint = tensor_fingerprint(logits)?;
    let values = logits.flatten_all()?.contiguous()?.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "characterization logits contain a non-finite value"
    );
    Ok(LogitCapture {
        fingerprint,
        values,
        predictions,
    })
}

fn run_logit_variant<F>(
    name: &str,
    device: &Device,
    forward: F,
) -> Result<(LogitCapture, LogitVariantReport)>
where
    F: FnOnce() -> Result<Tensor>,
{
    sync_cuda_device(device)?;
    let started = Instant::now();
    let logits = forward()?;
    sync_cuda_device(device)?;
    let seconds = started.elapsed().as_secs_f64();
    ensure!(seconds.is_finite(), "non-finite variant duration");
    let capture = capture_logits(&logits)?;
    let report = LogitVariantReport {
        name: name.into(),
        logits: capture.fingerprint.clone(),
        seconds,
    };
    Ok((capture, report))
}

fn selected_numeric_summary(mut values: Vec<f64>) -> Result<NumericSummary> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "numeric summary contains a non-finite value"
    );
    if values.is_empty() {
        return Ok(NumericSummary {
            count: 0,
            zero_count: 0,
            min: None,
            median: None,
            p90: None,
            p99: None,
            max: None,
        });
    }
    let count = values.len();
    let zero_count = values.iter().filter(|value| **value == 0.0).count();
    let min = values.iter().copied().reduce(f64::min);
    let max = values.iter().copied().reduce(f64::max);
    let select = |values: &mut [f64], q: f64| {
        let index = ((values.len() - 1) as f64 * q).round() as usize;
        *values.select_nth_unstable_by(index, f64::total_cmp).1
    };
    let median = Some(select(&mut values, 0.5));
    let p90 = Some(select(&mut values, 0.9));
    let p99 = Some(select(&mut values, 0.99));
    Ok(NumericSummary {
        count,
        zero_count,
        min,
        median,
        p90,
        p99,
        max,
    })
}

fn reference_margin(logits: &[f32]) -> Result<f64> {
    ensure!(logits.len() == PALETTE_SIZE, "margin palette width drift");
    let mut first = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in logits {
        ensure!(value.is_finite(), "margin input is non-finite");
        if *value > first {
            second = first;
            first = *value;
        } else if *value > second {
            second = *value;
        }
    }
    let margin = f64::from(first) - f64::from(second);
    ensure!(
        margin.is_finite() && margin >= 0.0,
        "invalid top-two margin"
    );
    Ok(margin)
}

fn compare_logit_captures(
    left_name: &str,
    left: &LogitCapture,
    right_name: &str,
    right: &LogitCapture,
) -> Result<LogitComparisonReport> {
    ensure!(
        left.values.len() == right.values.len()
            && left.predictions.len() == right.predictions.len()
            && left
                .predictions
                .iter()
                .chain(&right.predictions)
                .all(|row| row.len() == FRAME_SIDE * FRAME_SIDE)
            && left.values.len() == left.predictions.len() * FRAME_SIDE * FRAME_SIDE * PALETTE_SIZE,
        "logit comparison geometry drift"
    );
    let deltas = left
        .values
        .iter()
        .zip(&right.values)
        .map(|(left, right)| f64::from((*left - *right).abs()))
        .collect::<Vec<_>>();
    let mut disagreement_rows = 0usize;
    let mut disagreement_pixels = 0usize;
    let mut margins = Vec::new();
    for (row, (left_predictions, right_predictions)) in
        left.predictions.iter().zip(&right.predictions).enumerate()
    {
        let mut row_disagrees = false;
        for (pixel, (left_prediction, right_prediction)) in
            left_predictions.iter().zip(right_predictions).enumerate()
        {
            if left_prediction == right_prediction {
                continue;
            }
            row_disagrees = true;
            disagreement_pixels += 1;
            let offset = (row * FRAME_SIDE * FRAME_SIDE + pixel) * PALETTE_SIZE;
            margins.push(reference_margin(
                &left.values[offset..offset + PALETTE_SIZE],
            )?);
        }
        disagreement_rows += usize::from(row_disagrees);
    }
    Ok(LogitComparisonReport {
        left: left_name.into(),
        right: right_name.into(),
        bit_identical: left.fingerprint == right.fingerprint,
        argmax_disagreement_pixels: disagreement_pixels,
        argmax_disagreement_rows: disagreement_rows,
        absolute_logit_delta: selected_numeric_summary(deltas)?,
        disagreement_reference_margin: selected_numeric_summary(margins)?,
    })
}

fn seam_branch(
    inputs: &InputComparisonReport,
    comparisons: &BTreeMap<String, LogitComparisonReport>,
) -> Result<String> {
    let comparison = |name: &str| {
        comparisons
            .get(name)
            .with_context(|| format!("missing seam comparison {name}"))
    };
    let self_repeat_unstable = ["v5a_v5b", "v5a_v5c", "v1a_v1b"]
        .into_iter()
        .try_fold(false, |unstable, name| {
            Ok::<_, anyhow::Error>(unstable || !comparison(name)?.bit_identical)
        })?;
    if self_repeat_unstable {
        return Ok("SELF_REPEAT_UNSTABLE".into());
    }
    if !inputs.active_inputs_equal {
        return Ok("ACTIVE_INPUT_PREPARATION_DIFFERS".into());
    }
    let shape = comparison("v4_v5a")?.argmax_disagreement_pixels > 0;
    let same_shape = comparison("v1a_v4")?.argmax_disagreement_pixels > 0;
    Ok(match (shape, same_shape) {
        (true, true) => "COMPOUND_SHAPE_AND_SAME_SHAPE",
        (true, false) => "EXECUTION_SHAPE",
        (false, true) => "SAME_SHAPE_TRAIN_EVAL",
        (false, false) => "NOT_REPRODUCED",
    }
    .into())
}

fn training_seam_logits(
    objective: FoundationV2ObjectiveConfig,
    model: &WorldModel,
    mixed: &MixedStreamBatch,
    host: &PreparedFoundationV2BatchHost,
    device: &Device,
    event_slot_weights: &Tensor,
) -> Result<Tensor> {
    let losses = foundation_v2_training_loss_with_event_weights(
        model,
        mixed,
        host,
        device,
        objective,
        event_slot_weights,
    )?;
    losses
        .diagnostic_predicted_logits
        .context("training seam did not capture prediction logits")
}

fn record_logit_variant(
    root: &Path,
    report: &mut SeamCharacterizationReport,
    captures: &mut BTreeMap<String, LogitCapture>,
    name: &str,
    result: (LogitCapture, LogitVariantReport),
) -> Result<()> {
    let (capture, variant) = result;
    ensure!(
        captures.insert(name.into(), capture).is_none()
            && report.variants.insert(name.into(), variant).is_none(),
        "duplicate characterization variant {name}"
    );
    write_json_report(&root.join("progress.json"), report)
}

fn ensure_seam_characterization_complete(report: &SeamCharacterizationReport) -> Result<()> {
    ensure!(
        report.schema == SEAM_CHARACTERIZATION_SCHEMA
            && report.evidence_class == SEAM_CHARACTERIZATION_EVIDENCE_CLASS
            && report.run_class == SEAM_CHARACTERIZATION_RUN_CLASS
            && report.registered
            && !report.research_claim
            && !report.public_data_read
            && report.no_backward_calls
            && report.no_optimizer_step
            && report.no_ema_update
            && report.no_checkpoint_write
            && report.lifecycle.state == LIFECYCLE_COMPLETE
            && report.device_is_cuda
            && report.g.is_some()
            && report.failed_preflight.is_some()
            && report.population.is_some()
            && report.inputs.is_some()
            && report.variants.len() == 6
            && report.comparisons.len() == 6
            && report.branch.is_some()
            && report.error.is_none(),
        "seam characterization completion contract failed"
    );
    let expected_logits = REGISTERED_BATCH_SIZE * FRAME_SIDE * FRAME_SIDE * PALETTE_SIZE;
    for comparison in report.comparisons.values() {
        ensure!(
            comparison.absolute_logit_delta.count == expected_logits
                && comparison.disagreement_reference_margin.count
                    == comparison.argmax_disagreement_pixels
                && comparison.argmax_disagreement_pixels
                    <= REGISTERED_BATCH_SIZE * FRAME_SIDE * FRAME_SIDE
                && comparison.argmax_disagreement_rows <= REGISTERED_BATCH_SIZE,
            "seam comparison count contract failed"
        );
    }
    let inputs = report.inputs.as_ref().context("missing completed inputs")?;
    ensure!(
        report.active_input_preparation_differs != inputs.active_inputs_equal
            && report.self_repeat_unstable
                == ["v5a_v5b", "v5a_v5c", "v1a_v1b"].into_iter().any(|name| {
                    report
                        .comparisons
                        .get(name)
                        .is_some_and(|comparison| !comparison.bit_identical)
                })
            && report.execution_shape_argmax_flip
                == report
                    .comparisons
                    .get("v4_v5a")
                    .is_some_and(|value| value.argmax_disagreement_pixels > 0)
            && report.same_shape_train_eval_argmax_flip
                == report
                    .comparisons
                    .get("v1a_v4")
                    .is_some_and(|value| value.argmax_disagreement_pixels > 0)
            && report.branch.as_ref() == Some(&seam_branch(inputs, &report.comparisons)?),
        "seam branch does not reproduce from its frozen inputs"
    );
    if !matches!(
        report.branch.as_deref(),
        Some("SELF_REPEAT_UNSTABLE" | "NOT_REPRODUCED")
    ) {
        ensure!(
            report
                .comparisons
                .get("v1a_v5a")
                .is_some_and(|value| value.argmax_disagreement_pixels > 0),
            "characterization branch did not reproduce the original argmax disagreement"
        );
    }
    ensure!(
        report.timing.population_seconds.is_finite()
            && report.timing.population_seconds >= 0.0
            && report.timing.input_comparison_seconds.is_finite()
            && report.timing.input_comparison_seconds >= 0.0
            && report.timing.wall_seconds > 0.0
            && report.timing.wall_seconds <= SEAM_CHARACTERIZATION_MAX_WALL.as_secs_f64()
            && report
                .variants
                .values()
                .all(|variant| variant.seconds.is_finite() && variant.seconds >= 0.0),
        "seam characterization timing contract failed"
    );
    ensure!(
        seam_characterization_identity(report)? == report.identity_root,
        "seam characterization identity drift"
    );
    Ok(())
}

fn run_seam_characterization_inner(
    args: &P2FrozenSeamCharacterizationArgs,
    report: &mut SeamCharacterizationReport,
    started: Instant,
) -> Result<()> {
    let root = args.output_root.as_path();
    ensure!(
        args.device.trim().starts_with("cuda"),
        "seam characterization requires CUDA"
    );
    registered_provenance_guard(&report.provenance)?;
    ensure_source_descends_from_g(&report.provenance)?;

    let (g, g_binding) = bind_g(&args.g_report)?;
    report.g = Some(g_binding.clone());
    let failed_preflight = bind_failed_frozen_preflight(&args.failed_preflight_report)?;
    report.failed_preflight = Some(failed_preflight.clone());

    let config_path = g_binding.root.join("train_config.json");
    report.train_config_sha256 = file_sha256_hex(&config_path)?;
    ensure!(
        report.train_config_sha256 == REGISTERED_TRAIN_CONFIG_SHA256
            && report.train_config_sha256 == g.train_config_sha256,
        "seam characterization train config drift"
    );
    fs::copy(&config_path, root.join("train_config.json"))?;
    let cfg = load_train_config(&config_path)?;
    cfg.validate()?;
    ensure_registered_config(&cfg)?;
    ensure!(
        cfg.seed == REGISTERED_SEED
            && cfg.physical_batch == REGISTERED_BATCH_SIZE
            && cfg.data_contract_v6
            && args.device == cfg.device,
        "seam characterization config drift"
    );

    let cargo_lock = fs::canonicalize(std::env::current_dir()?.join("Cargo.lock"))?;
    report.cargo_lock_sha256 = file_sha256_hex(&cargo_lock)?;
    ensure!(
        report.cargo_lock_sha256 == REGISTERED_G_CARGO_LOCK_SHA256,
        "seam characterization Cargo.lock differs from G"
    );
    fs::copy(&cargo_lock, root.join("Cargo.lock"))?;

    let population_started = Instant::now();
    let population =
        MultibatchPopulation::compose(cfg.seed, cfg.physical_batch, cfg.data_contract_v6)?;
    let census = population.write(root)?;
    let census_path = root.join("population/census.json");
    report.population_census_sha256 = file_sha256_hex(&census_path)?;
    ensure!(
        g.population.as_ref() == Some(&census)
            && g.population_census_sha256.as_deref()
                == Some(report.population_census_sha256.as_str()),
        "seam characterization population differs from G"
    );
    report.population = Some(census);
    let mixed = population
        .train_main
        .first()
        .context("G population lacks train batch position 0")?;
    ensure!(
        mixed.samples().len() == REGISTERED_BATCH_SIZE
            && mixed.factual_group_ranges().len() == 1
            && mixed.factual_group_ranges()[0] == (43..53),
        "seam characterization batch-0 geometry drift"
    );
    let host = prepare_foundation_v2_batch_host(mixed)?;
    let transitions = mixed
        .transitions()
        .cloned()
        .collect::<Vec<TransitionSample>>();
    report.timing.population_seconds = population_started.elapsed().as_secs_f64();

    report.gpu_identity = Some(query_gpu_identity(&args.device)?);
    ensure!(
        report.gpu_identity == g.gpu_identity,
        "seam characterization GPU identity differs from G"
    );
    let expected = snapshot_by_step(&g, 0)?;
    let checkpoint = checked_snapshot_path(&g_binding.root, expected)?;
    report.checkpoint_sha256 = file_sha256_hex(&checkpoint)?;
    ensure!(
        report.checkpoint_sha256 == expected.raw_sha256,
        "step-0 checkpoint hash drift"
    );
    report.identity_root = seam_characterization_identity(report)?;
    write_json_report(&root.join("progress.json"), report)?;

    let _pid_guard = TrainPidGuard::install(root)?;
    let diagnostic_device = open_diagnostic_device(&args.device, root)?;
    ensure!(
        diagnostic_device.device.is_cuda() && diagnostic_device.gpu_identity == report.gpu_identity,
        "seam characterization CUDA identity changed during open"
    );
    report.device_is_cuda = true;
    let device = &diagnostic_device.device;
    let varmap = VarMap::new();
    let model = WorldModel::new(
        cfg.model_config(),
        VarBuilder::from_varmap(&varmap, DType::F32, device),
    )?;
    load_varmap_exact(&varmap, &checkpoint)?;
    ensure_operator_projection_zero(&varmap)?;

    sync_cuda_device(device)?;
    let inputs_started = Instant::now();
    let training_batch = batch_from_foundation_v2_host(&host, device)?;
    let evaluator_batch = batch_from_samples(&transitions, device)?;
    report.inputs = Some(compare_batch_inputs(&training_batch, &evaluator_batch)?);
    sync_cuda_device(device)?;
    report.timing.input_comparison_seconds = inputs_started.elapsed().as_secs_f64();
    write_json_report(&root.join("progress.json"), report)?;

    let event_slot_weights = event_slot_weight_tensor(device)?;
    let mut captures = BTreeMap::<String, LogitCapture>::new();

    let result = run_logit_variant("V5a", device, || {
        raw_one_step_logits(&model, &transitions, device)
    })?;
    record_logit_variant(root, report, &mut captures, "V5a", result)?;
    let result = run_logit_variant("V5b", device, || {
        raw_one_step_logits(&model, &transitions, device)
    })?;
    record_logit_variant(root, report, &mut captures, "V5b", result)?;
    let result = run_logit_variant("V4", device, || {
        raw_one_step_logits_with_chunk(&model, &transitions, device, REGISTERED_BATCH_SIZE)
    })?;
    record_logit_variant(root, report, &mut captures, "V4", result)?;
    let result = run_logit_variant("V1a", device, || {
        training_seam_logits(
            diagnostic_objective(&cfg, expected.ep_weight, 0, 0),
            &model,
            mixed,
            &host,
            device,
            &event_slot_weights,
        )
    })?;
    record_logit_variant(root, report, &mut captures, "V1a", result)?;
    let result = run_logit_variant("V1b", device, || {
        training_seam_logits(
            diagnostic_objective(&cfg, expected.ep_weight, 0, 0),
            &model,
            mixed,
            &host,
            device,
            &event_slot_weights,
        )
    })?;
    record_logit_variant(root, report, &mut captures, "V1b", result)?;
    let result = run_logit_variant("V5c", device, || {
        raw_one_step_logits(&model, &transitions, device)
    })?;
    record_logit_variant(root, report, &mut captures, "V5c", result)?;

    for (name, left, right) in [
        ("v5a_v5b", "V5a", "V5b"),
        ("v5a_v5c", "V5a", "V5c"),
        ("v1a_v1b", "V1a", "V1b"),
        ("v4_v5a", "V4", "V5a"),
        ("v1a_v4", "V1a", "V4"),
        ("v1a_v5a", "V1a", "V5a"),
    ] {
        let comparison = compare_logit_captures(
            left,
            captures
                .get(left)
                .with_context(|| format!("missing variant {left}"))?,
            right,
            captures
                .get(right)
                .with_context(|| format!("missing variant {right}"))?,
        )?;
        ensure!(
            report.comparisons.insert(name.into(), comparison).is_none(),
            "duplicate seam comparison {name}"
        );
        ensure!(
            started.elapsed() <= SEAM_CHARACTERIZATION_MAX_WALL,
            "seam characterization wall-time cap exceeded"
        );
        write_json_report(&root.join("progress.json"), report)?;
    }
    let inputs = report
        .inputs
        .as_ref()
        .context("input comparison disappeared")?;
    report.self_repeat_unstable = ["v5a_v5b", "v5a_v5c", "v1a_v1b"]
        .into_iter()
        .any(|name| !report.comparisons[name].bit_identical);
    report.active_input_preparation_differs = !inputs.active_inputs_equal;
    report.execution_shape_argmax_flip =
        report.comparisons["v4_v5a"].argmax_disagreement_pixels > 0;
    report.same_shape_train_eval_argmax_flip =
        report.comparisons["v1a_v4"].argmax_disagreement_pixels > 0;
    report.numeric_shape_drift_without_argmax =
        !report.comparisons["v4_v5a"].bit_identical && !report.execution_shape_argmax_flip;
    report.numeric_same_shape_drift_without_argmax =
        !report.comparisons["v1a_v4"].bit_identical && !report.same_shape_train_eval_argmax_flip;
    report.branch = Some(seam_branch(inputs, &report.comparisons)?);

    sync_cuda_device(device)?;
    drop(captures);
    drop(training_batch);
    drop(evaluator_batch);
    drop(diagnostic_device);

    let (_, rebound_g) = bind_g(&args.g_report)?;
    ensure!(rebound_g == g_binding, "G changed during characterization");
    ensure!(
        bind_failed_frozen_preflight(&args.failed_preflight_report)? == failed_preflight,
        "failed preflight changed during characterization"
    );
    ensure!(
        file_sha256_hex(&config_path)? == report.train_config_sha256
            && file_sha256_hex(&cargo_lock)? == report.cargo_lock_sha256
            && file_sha256_hex(&census_path)? == report.population_census_sha256
            && file_sha256_hex(&checkpoint)? == report.checkpoint_sha256
            && file_sha256_hex(&root.join("train_config.json"))? == report.train_config_sha256
            && file_sha256_hex(&root.join("Cargo.lock"))? == report.cargo_lock_sha256,
        "characterization input changed during device work"
    );
    ensure!(
        format!(
            "sha256:{}",
            file_sha256_hex(&report.provenance.binary_path)?
        ) == report.provenance.binary_sha256,
        "characterization binary changed during device work"
    );
    ensure!(
        Some(query_gpu_identity(&args.device)?) == report.gpu_identity,
        "characterization GPU identity changed"
    );
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    ensure!(
        report.timing.wall_seconds <= SEAM_CHARACTERIZATION_MAX_WALL.as_secs_f64(),
        "seam characterization wall-time cap exceeded during validation"
    );
    Ok(())
}

pub fn run_p2_frozen_seam_characterization(args: P2FrozenSeamCharacterizationArgs) -> Result<()> {
    let started = Instant::now();
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: SEAM_CHARACTERIZATION_EVIDENCE_CLASS.into(),
        run_class: SEAM_CHARACTERIZATION_RUN_CLASS.into(),
        note: "no-gradient V6 train/raw seam characterization in progress".into(),
    };
    let command = open_run_root(&args.output_root, &lifecycle, SEAM_CHARACTERIZATION_TAG)?;
    let mut report = SeamCharacterizationReport {
        schema: SEAM_CHARACTERIZATION_SCHEMA.into(),
        evidence_class: SEAM_CHARACTERIZATION_EVIDENCE_CLASS.into(),
        run_class: SEAM_CHARACTERIZATION_RUN_CLASS.into(),
        registered: true,
        research_claim: false,
        public_data_read: false,
        no_backward_calls: true,
        no_optimizer_step: true,
        no_ema_update: true,
        no_checkpoint_write: true,
        lifecycle,
        provenance: launch_provenance().clone(),
        command,
        device: args.device.clone(),
        device_is_cuda: false,
        gpu_identity: None,
        output_root: args.output_root.clone(),
        g: None,
        failed_preflight: None,
        train_config_sha256: String::new(),
        cargo_lock_sha256: String::new(),
        population_census_sha256: String::new(),
        population: None,
        checkpoint_sha256: String::new(),
        inputs: None,
        variants: BTreeMap::new(),
        comparisons: BTreeMap::new(),
        self_repeat_unstable: false,
        active_input_preparation_differs: false,
        execution_shape_argmax_flip: false,
        same_shape_train_eval_argmax_flip: false,
        numeric_shape_drift_without_argmax: false,
        numeric_same_shape_drift_without_argmax: false,
        branch: None,
        timing: SeamCharacterizationTiming {
            population_seconds: 0.0,
            input_comparison_seconds: 0.0,
            wall_seconds: 0.0,
        },
        identity_root: String::new(),
        error: None,
    };
    let mut outcome = run_seam_characterization_inner(&args, &mut report, started);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    if report.identity_root.is_empty() && report.population.is_some() {
        match seam_characterization_identity(&report) {
            Ok(identity) => report.identity_root = identity,
            Err(error) => outcome = Err(error.context("compute seam characterization identity")),
        }
    }
    report.lifecycle = match &outcome {
        Ok(()) => LifecycleRecord {
            state: LIFECYCLE_COMPLETE.into(),
            unix_seconds: unix_seconds(),
            evidence_class: SEAM_CHARACTERIZATION_EVIDENCE_CLASS.into(),
            run_class: SEAM_CHARACTERIZATION_RUN_CLASS.into(),
            note: format!(
                "{}; characterization only; no gradient, model update, or training authority",
                report.branch.as_deref().unwrap_or("missing_branch")
            ),
        },
        Err(error) => {
            report.error = Some(format!("{error:#}"));
            report.evidence_class = FAILED_EVIDENCE_CLASS.into();
            LifecycleRecord {
                state: LIFECYCLE_FAILED.into(),
                unix_seconds: unix_seconds(),
                evidence_class: FAILED_EVIDENCE_CLASS.into(),
                run_class: SEAM_CHARACTERIZATION_RUN_CLASS.into(),
                note: format!("{error:#}"),
            }
        }
    };
    if outcome.is_ok() {
        if let Err(error) = ensure_seam_characterization_complete(&report) {
            report.error = Some(format!("{error:#}"));
            report.evidence_class = FAILED_EVIDENCE_CLASS.into();
            report.lifecycle = LifecycleRecord {
                state: LIFECYCLE_FAILED.into(),
                unix_seconds: unix_seconds(),
                evidence_class: FAILED_EVIDENCE_CLASS.into(),
                run_class: SEAM_CHARACTERIZATION_RUN_CLASS.into(),
                note: format!("{error:#}"),
            };
            outcome = Err(error.context("post-completion seam characterization validation"));
        }
    }
    let lifecycle = report.lifecycle.clone();
    seal_run_root(
        &args.output_root,
        SEAM_CHARACTERIZATION_TAG,
        &report,
        &lifecycle,
    )?;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    #[cfg(feature = "cudnn")]
    fn parameter_tensor_fingerprints(
        varmap: &VarMap,
    ) -> Result<BTreeMap<String, TensorFingerprint>> {
        varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| Ok((name.clone(), tensor_fingerprint(var.as_tensor())?)))
            .collect()
    }

    #[test]
    #[ignore = "manual frozen-checkpoint precision characterization"]
    #[cfg(feature = "cudnn")]
    fn frozen_reconstruction_precision_characterization() -> Result<()> {
        ensure!(
            std::env::var("TOFY_FROZEN_RECONSTRUCTION_PROBE").as_deref() == Ok("1"),
            "frozen precision probe requires its explicit test-only stop"
        );
        let g_path = PathBuf::from(std::env::var("TOFY_PRECISION_G_REPORT")?);
        let root = PathBuf::from(std::env::var("TOFY_PRECISION_PROBE_ROOT")?);
        let (g, binding) = bind_g(&g_path)?;
        let cfg_path = binding.root.join("train_config.json");
        ensure!(
            file_sha256_hex(&cfg_path)? == REGISTERED_TRAIN_CONFIG_SHA256,
            "precision probe config drift"
        );
        let cfg = load_train_config(&cfg_path)?;
        cfg.validate()?;
        ensure_registered_config(&cfg)?;
        let population =
            MultibatchPopulation::compose(cfg.seed, cfg.physical_batch, cfg.data_contract_v6)?;
        let census = population.write(&root)?;
        ensure!(
            g.population.as_ref() == Some(&census)
                && g.population_census_sha256.as_deref()
                    == Some(file_sha256_hex(&root.join("population/census.json"))?.as_str()),
            "precision probe population drift"
        );
        let expected = snapshot_by_step(&g, 0)?;
        let checkpoint = checked_snapshot_path(&binding.root, expected)?;
        ensure!(
            Some(query_gpu_identity("cuda")?) == g.gpu_identity,
            "precision probe GPU drift"
        );
        let device = Device::new_cuda(0)?;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        load_varmap_exact(&varmap, &checkpoint)?;
        ensure_operator_projection_zero(&varmap)?;
        let before = parameter_tensor_fingerprints(&varmap)?;
        let host = prepare_foundation_v2_batch_host(&population.train_main[0])?;
        let transitions = population.train_main[0]
            .transitions()
            .cloned()
            .collect::<Vec<_>>();
        // Identical bounded warmup in both arms; no claim to reproduce the
        // original preflight's full D2/held-out allocator history.
        let raw = capture_logits(&raw_one_step_logits(&model, &transitions, &device)?)?;
        let error = gradient_cell(
            0,
            0,
            expected.ep_weight,
            &cfg,
            &model,
            &varmap,
            &population.train_main[0],
            &population.train_rollout[0],
            &host,
            &device,
            &raw.predictions,
            &load_loss_controls(&g)?,
            &g,
        )
        .err()
        .context("precision probe did not stop at reconstruction capture")?
        .to_string();
        let checks: serde_json::Value = serde_json::from_str(
            error
                .strip_prefix("TOFY_FROZEN_RECONSTRUCTION ")
                .context(error.clone())?,
        )?;
        sync_cuda_device(&device)?;
        ensure!(
            before == parameter_tensor_fingerprints(&varmap)?,
            "probe changed weights"
        );
        let (_, rebound) = bind_g(&g_path)?;
        ensure!(binding == rebound, "precision probe G changed");
        println!(
            "TOFY_FROZEN_PRECISION {}",
            serde_json::json!({
                "schema": "p2.frozen_reconstruction_precision.v1",
                "evidence_class": "frozen_checkpoint_precision_characterization_only",
                "tf32_override": std::env::var("NVIDIA_TF32_OVERRIDE").ok(),
                "snapshot_step": 0,
                "batch_position": 0,
                "physical_batch": cfg.physical_batch,
                "checkpoint_sha256": expected.raw_sha256,
                "population_census_sha256": file_sha256_hex(&root.join("population/census.json"))?,
                "parameters_unchanged": true,
                "checks": checks,
            })
        );
        Ok(())
    }

    /// A bounded backend control, not a model or frozen-diagnostic admission test.
    #[test]
    #[ignore = "manual precision characterization; requires a preregistered device run"]
    #[cfg(feature = "cudnn")]
    fn convolution_backward_additivity_characterization() -> Result<()> {
        let requested = std::env::var("TOFY_PRECISION_PROBE_DEVICE")?;
        let device = match requested.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0)?,
            _ => anyhow::bail!("precision probe device must be cpu or cuda"),
        };
        // Host-generated dyadic values give all processes exactly the same fixture.
        let fixture = |count: usize, seed: u32| {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as i32 % 16_384 - 8_192) as f32 / 8_192.0
                })
                .collect::<Vec<_>>()
        };
        let input_values = fixture(8 * 32 * 16 * 16, 17);
        let weight_values = fixture(32 * 32 * 3 * 3, 29);
        let input = Var::from_vec(input_values.clone(), (8, 32, 16, 16), &device)?;
        let weight = Var::from_vec(weight_values.clone(), (32, 32, 3, 3), &device)?;
        let varmap = VarMap::new();
        {
            let mut vars = varmap.data().lock().unwrap();
            vars.insert("probe.input".into(), input.clone());
            vars.insert("probe.weight".into(), weight.clone());
        }
        let output = input.conv2d(weight.as_tensor(), 1, 1, 1, 1)?;
        let left = Tensor::from_vec(fixture(output.elem_count(), 41), output.dims(), &device)?;
        let right = Tensor::from_vec(fixture(output.elem_count(), 53), output.dims(), &device)?;
        let left_loss = output.mul(&left)?.sum_all()?;
        let right_loss = output.mul(&right)?.sum_all()?;
        let combined_loss = left_loss.add(&right_loss)?;
        let direct = retain_parameter_gradients(combined_loss.backward()?, &varmap)?;
        let repeated = retain_parameter_gradients(combined_loss.backward()?, &varmap)?;
        let mut summed = None;
        accumulate_parameter_gradients(&mut summed, left_loss.backward()?, &varmap)?;
        accumulate_parameter_gradients(&mut summed, right_loss.backward()?, &varmap)?;
        let summed = summed.context("missing probe gradients")?;
        let mut comparisons = BTreeMap::new();
        for (name, parameter) in [("input", &input), ("weight", &weight)] {
            let one = VarMap::new();
            one.data()
                .lock()
                .unwrap()
                .insert(format!("probe.{name}"), parameter.clone());
            comparisons.insert(
                name,
                serde_json::json!({
                    "additivity": reconstruction_check(&direct, &summed, &one)?,
                    "repeat": reconstruction_check(&direct, &repeated, &one)?,
                }),
            );
        }
        sync_cuda_device(&device)?;
        ensure!(
            input.flatten_all()?.to_vec1::<f32>()? == input_values
                && weight.flatten_all()?.to_vec1::<f32>()? == weight_values,
            "precision probe mutated its fixture"
        );
        println!(
            "TOFY_CONV_ADDITIVITY {}",
            serde_json::json!({
                "schema": "p2.convolution_backward_additivity.v1",
                "evidence_class": "synthetic_backend_characterization_only",
                "device": requested,
                "tf32_override": std::env::var("NVIDIA_TF32_OVERRIDE").ok(),
                "input_shape": input.dims(),
                "weight_shape": weight.dims(),
                "fixture_unchanged": true,
                "comparisons": comparisons,
            })
        );
        // A passed test means telemetry was captured, not that additivity passed.
        Ok(())
    }

    fn geometry(cosine: Option<f64>, share: Option<f64>, kappa: Option<f64>) -> RoutedGeometry {
        let route = RouteGeometry {
            l2: 1.0,
            share_of_full: share,
            cosine_to_prediction: cosine,
            kappa_to_prediction: kappa,
        };
        RoutedGeometry {
            global: route.clone(),
            adamw: route.clone(),
            muon: route,
        }
    }

    fn cell(step: usize, position: usize) -> GradientCellReport {
        GradientCellReport {
            snapshot_step: step,
            batch_position: position,
            sigreg_seed: 0,
            ep_weight: 0.1,
            false_edit_pixels: 1,
            mask_population: "primary_same_forward".into(),
            prediction_logits_fingerprint: Some(TensorFingerprint {
                shape: vec![REGISTERED_BATCH_SIZE, FRAME_SIDE, FRAME_SIDE, PALETTE_SIZE],
                dtype: "F32".into(),
                sha256: "sha256:test".into(),
            }),
            false_edit_mask_sha256: "sha256:test-mask".into(),
            mask_binding: Some(SameForwardMaskBinding {
                logits: TensorFingerprint {
                    shape: vec![REGISTERED_BATCH_SIZE, FRAME_SIDE, FRAME_SIDE, PALETTE_SIZE],
                    dtype: "F32".into(),
                    sha256: "sha256:test".into(),
                },
                mask_sha256: "sha256:test-mask".into(),
                false_edit_pixels: 1,
            }),
            training_raw_argmax_disagreement_pixels: 0,
            losses: FoundationV2LossMeans::default(),
            components: Vec::new(),
            prediction: geometry(Some(1.0), Some(1.0), Some(1.0)),
            auxiliary: geometry(Some(0.0), Some(0.05), Some(0.0)),
            combined: geometry(Some(0.99), Some(1.0), Some(1.0)),
            full_reconstruction: ReconstructionCheck {
                global_relative_or_absolute_residual: 0.0,
                adamw_relative_or_absolute_residual: 0.0,
                muon_relative_or_absolute_residual: 0.0,
                passed: true,
                ..Default::default()
            },
            prediction_reconstruction: ReconstructionCheck {
                global_relative_or_absolute_residual: 0.0,
                adamw_relative_or_absolute_residual: 0.0,
                muon_relative_or_absolute_residual: 0.0,
                passed: true,
                ..Default::default()
            },
            ns_muon_cosine_full_to_prediction: Some(0.99),
            false_edit_to_auxiliary_cosine: Some(0.0),
            false_edit_to_prediction_cosine: Some(1.0),
            changed_to_unchanged_cosine: Some(0.0),
            false_edit_share_of_prediction: Some(0.10),
            loss_binding: None,
        }
    }

    fn classification_context() -> (Vec<SnapshotRescore>, Vec<PersistenceReport>) {
        let anatomy = FalseEditAnatomy {
            population: PRIMARY_POPULATION.into(),
            false_edit_pixels: 1,
            margins: MarginSummary {
                count: 1,
                bins: BTreeMap::new(),
                min: Some(0.5),
                median: Some(0.5),
                p90: Some(0.5),
                p99: Some(0.5),
                max: Some(0.5),
            },
            locations: BTreeMap::new(),
            class_pairs: BTreeMap::new(),
            predicted_class_from_changed_region: 0,
            distance_to_changed: BTreeMap::new(),
            row_histogram_changed_rows: BTreeMap::new(),
            row_histogram_no_change_rows: BTreeMap::new(),
            changed_exact_near_miss: BTreeMap::new(),
        };
        let rescoring = vec![SnapshotRescore {
            step: 2048,
            legacy_population: "legacy_g_chunk32".into(),
            primary_population: "primary_same_forward".into(),
            anatomy_population: "primary_same_forward".into(),
            train: Vec::new(),
            heldout: Vec::new(),
            primary_train: Vec::new(),
            primary_heldout: Vec::new(),
            train_anatomy: Some(anatomy),
            heldout_anatomy: None,
            train_anatomy_by_batch: Vec::new(),
            heldout_anatomy_by_batch: Vec::new(),
            complete_union_binding: true,
            seconds: 1.0,
        }];
        let persistence = vec![PersistenceReport {
            population: PRIMARY_POPULATION.into(),
            split: "train".into(),
            false_edits_1024: 1,
            false_edits_1536: 1,
            false_edits_2048: 1,
            intersection_all_three: 0,
            union_all_three: 1,
            persistent_1536_to_2048: 0,
            persistence_1536_to_2048: Some(0.0),
            final_present_all_three_fraction: Some(0.0),
            new_1024_to_1536: 0,
            resolved_1024_to_1536: 0,
            new_1536_to_2048: 0,
            resolved_1536_to_2048: 0,
        }];
        (rescoring, persistence)
    }

    fn minimal_report(registered: bool) -> FrozenDiagnosticReport {
        FrozenDiagnosticReport {
            schema: FROZEN_DIAGNOSTIC_SCHEMA.into(),
            evidence_class: DIAGNOSTIC_EVIDENCE_CLASS.into(),
            run_class: if registered {
                RUN_CLASS_REGISTERED.into()
            } else {
                RUN_CLASS_PREFLIGHT.into()
            },
            registered,
            research_claim: false,
            public_data_read: false,
            lifecycle: LifecycleRecord {
                state: LIFECYCLE_RUNNING.into(),
                unix_seconds: 1,
                evidence_class: DIAGNOSTIC_EVIDENCE_CLASS.into(),
                run_class: if registered {
                    RUN_CLASS_REGISTERED.into()
                } else {
                    RUN_CLASS_PREFLIGHT.into()
                },
                note: String::new(),
            },
            provenance: LaunchProvenance::unknown(Path::new("tofy")),
            command: Vec::new(),
            device: "cuda".into(),
            device_is_cuda: true,
            gpu_identity: None,
            output_root: "out".into(),
            spec: FrozenDiagnosticSpec::new(registered),
            g: None,
            g_identity: "g".into(),
            preflight: None,
            train_config_sha256: "config".into(),
            cargo_lock_sha256: "cargo".into(),
            population_census_sha256: "census".into(),
            population: None,
            rescoring: Vec::new(),
            gradients: Vec::new(),
            comparisons: Vec::new(),
            persistence: Vec::new(),
            runtime_estimate: None,
            verdict: None,
            timing: FrozenDiagnosticTiming {
                population_seconds: 1.0,
                gradient_cell_seconds: vec![1.0, 1.0],
                rescore_seconds: vec![1.0, 1.0, 1.0],
                comparison_seconds: Vec::new(),
                wall_seconds: 1.0,
            },
            identity_root: String::new(),
            error: None,
        }
    }

    fn empty_rescore(step: usize) -> SnapshotRescore {
        SnapshotRescore {
            step,
            legacy_population: "legacy_g_chunk32".into(),
            primary_population: "primary_same_forward".into(),
            anatomy_population: "primary_same_forward".into(),
            train: Vec::new(),
            heldout: Vec::new(),
            primary_train: Vec::new(),
            primary_heldout: Vec::new(),
            train_anatomy: None,
            heldout_anatomy: None,
            train_anatomy_by_batch: Vec::new(),
            heldout_anatomy_by_batch: Vec::new(),
            complete_union_binding: false,
            seconds: 1.0,
        }
    }

    #[test]
    fn frozen_threshold_priority_and_independent_flags_are_exact() -> Result<()> {
        let mut cells = GRADIENT_STEPS
            .into_iter()
            .flat_map(|step| (0..TRAIN_BATCHES).map(move |position| cell(step, position)))
            .collect::<Vec<_>>();
        let (rescoring, persistence_rows) = classification_context();
        assert_eq!(
            classify(&cells, &rescoring, &persistence_rows)?.auxiliary_class,
            "AUX_NEGLIGIBLE"
        );
        for conflict in cells.iter_mut().take(4) {
            conflict.combined.global.cosine_to_prediction = Some(0.79);
        }
        let verdict = classify(&cells, &rescoring, &persistence_rows)?;
        assert_eq!(verdict.auxiliary_class, "AUX_CONFLICTS");
        assert_eq!(verdict.conflict_cells, 4);
        for cell in cells.iter_mut().filter(|cell| cell.snapshot_step >= 1536) {
            cell.false_edit_share_of_prediction = Some(0.01);
        }
        for cell in cells.iter_mut().take(8) {
            cell.changed_to_unchanged_cosine = Some(-0.25);
        }
        let verdict = classify(&cells, &rescoring, &persistence_rows)?;
        assert!(verdict.pred_blind && verdict.internal_conflict);
        assert!(verdict.next_action.contains("changed-versus-unchanged"));
        Ok(())
    }

    #[test]
    fn preflight_rescores_the_two_d1_controls_and_final_anatomy() {
        let spec = FrozenDiagnosticSpec::new(false);
        assert_eq!(spec.gradient_steps, [0, 1024]);
        assert_eq!(spec.rescore_steps, [0, 1024, 2048]);
        assert_eq!(spec.anatomy_steps, [2048]);
    }

    #[test]
    fn frozen_parent_binary_digest_matches_launch_provenance_representation() {
        assert_eq!(REGISTERED_G_BINARY_SHA256.len(), "sha256:".len() + 64);
        assert!(REGISTERED_G_BINARY_SHA256.starts_with("sha256:"));
    }

    #[test]
    fn tensor_fingerprint_binds_f32_bits_shape_and_dtype() -> Result<()> {
        let left = Tensor::from_vec(vec![0.0f32, -0.0, 1.0], (3,), &Device::Cpu)?;
        let same = Tensor::from_vec(vec![0.0f32, -0.0, 1.0], (3,), &Device::Cpu)?;
        let signed_zero_drift = Tensor::from_vec(vec![0.0f32, 0.0, 1.0], (3,), &Device::Cpu)?;
        let reshaped = Tensor::from_vec(vec![0.0f32, -0.0, 1.0], (1, 3), &Device::Cpu)?;
        assert_eq!(tensor_fingerprint(&left)?, tensor_fingerprint(&same)?);
        assert_ne!(
            tensor_fingerprint(&left)?,
            tensor_fingerprint(&signed_zero_drift)?
        );
        assert_ne!(tensor_fingerprint(&left)?, tensor_fingerprint(&reshaped)?);
        Ok(())
    }

    #[test]
    fn numeric_summary_uses_frozen_nearest_rank_and_rejects_nonfinite() -> Result<()> {
        let summary = selected_numeric_summary(vec![4.0, 0.0, 3.0, 1.0, 2.0])?;
        assert_eq!(summary.count, 5);
        assert_eq!(summary.zero_count, 1);
        assert_eq!(summary.min, Some(0.0));
        assert_eq!(summary.median, Some(2.0));
        assert_eq!(summary.p90, Some(4.0));
        assert_eq!(summary.p99, Some(4.0));
        assert_eq!(summary.max, Some(4.0));
        assert!(selected_numeric_summary(vec![f64::NAN]).is_err());
        Ok(())
    }

    #[test]
    fn input_comparison_accepts_only_the_inert_operator_exception() -> Result<()> {
        let population =
            MultibatchPopulation::compose(REGISTERED_SEED, REGISTERED_BATCH_SIZE, true)?;
        let mixed = population
            .train_main
            .first()
            .context("missing train batch")?;
        let host = prepare_foundation_v2_batch_host(mixed)?;
        let training = batch_from_foundation_v2_host(&host, &Device::Cpu)?;
        let transitions = mixed.transitions().cloned().collect::<Vec<_>>();
        let mut evaluator = batch_from_samples(&transitions, &Device::Cpu)?;
        evaluator.operator_conditioning = (&training.operator_conditioning + 1.0)?;

        let comparison = compare_batch_inputs(&training, &evaluator)?;
        assert!(comparison.active_inputs_equal);
        assert!(!comparison.operator_conditioning_equal);
        assert!(comparison.accepted_inert_operator_non_identity);
        assert!(!comparison.fields["operator_conditioning"].active_for_predicted_latent);
        assert!(comparison
            .fields
            .values()
            .filter(|field| field.active_for_predicted_latent)
            .all(|field| field.exact_equal));
        Ok(())
    }

    #[test]
    fn logit_comparison_counts_exact_argmax_location_and_reference_margin() -> Result<()> {
        let rows = 2;
        let elements = rows * FRAME_SIDE * FRAME_SIDE * PALETTE_SIZE;
        let mut left_values = vec![0.0f32; elements];
        let mut right_values = left_values.clone();
        let row = 1;
        let pixel = 2;
        let offset = (row * FRAME_SIDE * FRAME_SIDE + pixel) * PALETTE_SIZE;
        left_values[offset] = 5.0;
        left_values[offset + 2] = 3.0;
        right_values[offset] = 1.0;
        right_values[offset + 1] = 4.0;
        let mut left_predictions = vec![vec![0u8; FRAME_SIDE * FRAME_SIDE]; rows];
        let mut right_predictions = left_predictions.clone();
        left_predictions[row][pixel] = 0;
        right_predictions[row][pixel] = 1;
        let fingerprint = |sha256: &str| TensorFingerprint {
            shape: vec![rows, FRAME_SIDE, FRAME_SIDE, PALETTE_SIZE],
            dtype: "F32".into(),
            sha256: sha256.into(),
        };
        let left = LogitCapture {
            fingerprint: fingerprint("left"),
            values: left_values,
            predictions: left_predictions,
        };
        let right = LogitCapture {
            fingerprint: fingerprint("right"),
            values: right_values,
            predictions: right_predictions,
        };

        let comparison = compare_logit_captures("left", &left, "right", &right)?;
        assert!(!comparison.bit_identical);
        assert_eq!(comparison.argmax_disagreement_pixels, 1);
        assert_eq!(comparison.argmax_disagreement_rows, 1);
        assert_eq!(comparison.absolute_logit_delta.count, elements);
        assert_eq!(comparison.absolute_logit_delta.zero_count, elements - 3);
        assert_eq!(comparison.disagreement_reference_margin.count, 1);
        assert_eq!(comparison.disagreement_reference_margin.min, Some(2.0));
        assert_eq!(comparison.disagreement_reference_margin.max, Some(2.0));
        Ok(())
    }

    fn seam_comparison(bit_identical: bool, disagreements: usize) -> LogitComparisonReport {
        LogitComparisonReport {
            left: String::new(),
            right: String::new(),
            bit_identical,
            argmax_disagreement_pixels: disagreements,
            argmax_disagreement_rows: usize::from(disagreements > 0),
            absolute_logit_delta: NumericSummary {
                count: 1,
                zero_count: usize::from(bit_identical),
                min: Some(if bit_identical { 0.0 } else { 1.0 }),
                median: Some(if bit_identical { 0.0 } else { 1.0 }),
                p90: Some(if bit_identical { 0.0 } else { 1.0 }),
                p99: Some(if bit_identical { 0.0 } else { 1.0 }),
                max: Some(if bit_identical { 0.0 } else { 1.0 }),
            },
            disagreement_reference_margin: NumericSummary {
                count: disagreements,
                zero_count: 0,
                min: (disagreements > 0).then_some(1.0),
                median: (disagreements > 0).then_some(1.0),
                p90: (disagreements > 0).then_some(1.0),
                p99: (disagreements > 0).then_some(1.0),
                max: (disagreements > 0).then_some(1.0),
            },
        }
    }

    fn seam_comparisons(
        shape: usize,
        same_shape: usize,
    ) -> BTreeMap<String, LogitComparisonReport> {
        BTreeMap::from([
            ("v5a_v5b".into(), seam_comparison(true, 0)),
            ("v5a_v5c".into(), seam_comparison(true, 0)),
            ("v1a_v1b".into(), seam_comparison(true, 0)),
            ("v4_v5a".into(), seam_comparison(shape == 0, shape)),
            (
                "v1a_v4".into(),
                seam_comparison(same_shape == 0, same_shape),
            ),
            (
                "v1a_v5a".into(),
                seam_comparison(shape == 0 && same_shape == 0, shape + same_shape),
            ),
        ])
    }

    fn seam_inputs(active_inputs_equal: bool) -> InputComparisonReport {
        InputComparisonReport {
            fields: BTreeMap::new(),
            active_inputs_equal,
            operator_conditioning_equal: false,
            accepted_inert_operator_non_identity: true,
        }
    }

    #[test]
    fn seam_branch_priority_and_matrix_are_exact() -> Result<()> {
        assert_eq!(
            seam_branch(&seam_inputs(true), &seam_comparisons(0, 0))?,
            "NOT_REPRODUCED"
        );
        assert_eq!(
            seam_branch(&seam_inputs(true), &seam_comparisons(1, 0))?,
            "EXECUTION_SHAPE"
        );
        assert_eq!(
            seam_branch(&seam_inputs(true), &seam_comparisons(0, 1))?,
            "SAME_SHAPE_TRAIN_EVAL"
        );
        assert_eq!(
            seam_branch(&seam_inputs(true), &seam_comparisons(1, 1))?,
            "COMPOUND_SHAPE_AND_SAME_SHAPE"
        );
        assert_eq!(
            seam_branch(&seam_inputs(false), &seam_comparisons(1, 1))?,
            "ACTIVE_INPUT_PREPARATION_DIFFERS"
        );
        let mut unstable = seam_comparisons(1, 1);
        unstable.insert("v5a_v5b".into(), seam_comparison(false, 0));
        assert_eq!(
            seam_branch(&seam_inputs(false), &unstable)?,
            "SELF_REPEAT_UNSTABLE"
        );
        Ok(())
    }

    #[test]
    fn cross_process_control_comparison_tolerates_only_float_roundoff() {
        let left = serde_json::json!({"count": 3, "norm": 1.0, "state": "present"});
        let close = serde_json::json!({"count": 3, "norm": 1.0000005, "state": "present"});
        let wrong_float = serde_json::json!({"count": 3, "norm": 1.000002, "state": "present"});
        let wrong_count = serde_json::json!({"count": 4, "norm": 1.0, "state": "present"});
        assert!(json_values_approximately_equal(&left, &close));
        assert!(!json_values_approximately_equal(&left, &wrong_float));
        assert!(!json_values_approximately_equal(&left, &wrong_count));

        let norm_close = serde_json::json!({"pre_clip_gradient_norm": 1.000010});
        let norm_wrong = serde_json::json!({"pre_clip_gradient_norm": 1.000012});
        let norm_expected = serde_json::json!({"pre_clip_gradient_norm": 1.0});
        assert!(json_values_approximately_equal(&norm_expected, &norm_close));
        assert!(!json_values_approximately_equal(
            &norm_expected,
            &norm_wrong
        ));
    }

    #[test]
    fn frozen_population_matches_the_full_frame_metric_domain() -> Result<()> {
        let population =
            MultibatchPopulation::compose(REGISTERED_SEED, REGISTERED_BATCH_SIZE, true)?;
        ensure_full_frame_metric_domain(&population, &FrozenDiagnosticSpec::new(true))
    }

    #[test]
    fn preflight_binding_admits_registered_superset_and_checks_only_frozen_controls() -> Result<()>
    {
        let mut preflight = minimal_report(false);
        preflight.runtime_estimate = Some(runtime_estimate(&preflight)?);
        preflight.gradients = vec![cell(0, 0), cell(1024, 0)];
        preflight.rescoring = vec![empty_rescore(0), empty_rescore(1024), empty_rescore(2048)];
        let mut final_anatomy = classification_context().0[0]
            .train_anatomy
            .clone()
            .context("anatomy fixture")?;
        preflight.rescoring[2].train_anatomy_by_batch = vec![final_anatomy.clone()];

        let mut current = minimal_report(true);
        current.gradients = preflight.gradients.clone();
        current.rescoring = SNAPSHOT_STEPS.map(empty_rescore).to_vec();
        current.rescoring[4].train_anatomy_by_batch = vec![final_anatomy.clone()];
        final_anatomy.margins.median = Some(0.5000005);
        current.rescoring[6].train_anatomy_by_batch = vec![final_anatomy];
        assert!(ensure_preflight_binds(&preflight, &current)?.admitted);
        ensure_preflight_controls_bind(&preflight, &current)?;
        assert!(preflight_rescore_control(&preflight, 1).is_none());

        current.rescoring[6].train_anatomy_by_batch[0].false_edit_pixels += 1;
        assert!(ensure_preflight_controls_bind(&preflight, &current).is_err());
        current.rescoring[6].train_anatomy_by_batch[0].false_edit_pixels -= 1;
        current.g_identity = "drift".into();
        assert!(ensure_preflight_binds(&preflight, &current).is_err());
        Ok(())
    }

    #[test]
    fn report_identity_binds_spec_and_census() -> Result<()> {
        let report = minimal_report(false);
        let identity = diagnostic_identity(&report)?;
        let mut drifted = report.clone();
        drifted.population_census_sha256 = "other".into();
        assert_ne!(identity, diagnostic_identity(&drifted)?);
        drifted = report.clone();
        drifted.spec.max_wall_seconds += 1;
        assert_ne!(identity, diagnostic_identity(&drifted)?);
        Ok(())
    }

    #[test]
    fn histogram_and_quantile_boundaries_are_frozen() {
        assert_eq!(histogram_bin(0), "0");
        assert_eq!(histogram_bin(1), "1");
        assert_eq!(histogram_bin(2), "2-3");
        assert_eq!(histogram_bin(8), "8-15");
        assert_eq!(histogram_bin(64), ">=64");
        assert_eq!(quantile(&[0.0, 1.0, 2.0, 3.0, 4.0], 0.5), Some(2.0));
        assert_eq!(quantile(&[], 0.5), None);
    }

    #[test]
    fn kappa_is_zero_for_a_zero_numerator_and_defined_prediction_denominator() {
        let zero = FoundationV2GradientRouteStats::default();
        let full = FoundationV2GradientRouteStats {
            global_l2: 1.0,
            adamw_l2: 1.0,
            muon_l2: 1.0,
            ..FoundationV2GradientRouteStats::default()
        };
        let prediction = full.clone();
        let geometry = route_geometry(&zero, &full, &prediction);
        assert_eq!(geometry.global.cosine_to_prediction, None);
        assert_eq!(geometry.global.kappa_to_prediction, Some(0.0));
    }

    #[test]
    fn null_negligible_quantity_yields_mixed_not_conflict() -> Result<()> {
        let mut cells = GRADIENT_STEPS
            .into_iter()
            .flat_map(|step| (0..TRAIN_BATCHES).map(move |position| cell(step, position)))
            .collect::<Vec<_>>();
        cells[0].combined.adamw.cosine_to_prediction = None;
        let (rescoring, persistence_rows) = classification_context();
        assert_eq!(
            classify(&cells, &rescoring, &persistence_rows)?.auxiliary_class,
            "AUX_MIXED"
        );
        Ok(())
    }

    #[test]
    fn prediction_masks_keep_false_edits_on_original_unchanged_denominator() -> Result<()> {
        let population =
            MultibatchPopulation::compose(REGISTERED_SEED, REGISTERED_BATCH_SIZE, true)?;
        let sample = population.train_main[0].samples()[0].clone();
        let mut predicted = sample.transition.next.pixels.to_vec();
        let pixel = (0..predicted.len())
            .find(|pixel| {
                sample.content_mask.values[*pixel] != 0
                    && sample.transition.current.pixels[*pixel]
                        == sample.transition.next.pixels[*pixel]
            })
            .context("sample lacks an unchanged pixel")?;
        predicted[pixel] = (predicted[pixel] + 1) % PALETTE_SIZE as u8;
        let masks = masks_for_prediction_sub_losses(&[sample], &[predicted])?;
        assert_eq!(masks.false_edit_count, 1);
        assert_eq!(
            masks
                .false_edit
                .iter()
                .filter(|value| **value == 1.0)
                .count(),
            1
        );
        assert!(masks.unchanged_count > 1);
        assert_eq!(
            masks.changed_count + masks.unchanged_count,
            FRAME_SIDE * FRAME_SIDE
        );
        Ok(())
    }

    #[test]
    fn same_forward_mask_fingerprint_is_bound_to_its_argmax() -> Result<()> {
        let population =
            MultibatchPopulation::compose(REGISTERED_SEED, REGISTERED_BATCH_SIZE, true)?;
        let sample = population.train_main[0].samples()[0].clone();
        let pixel = (0..FRAME_SIDE * FRAME_SIDE)
            .find(|pixel| {
                sample.content_mask.values[*pixel] != 0
                    && sample.transition.current.pixels[*pixel]
                        == sample.transition.next.pixels[*pixel]
            })
            .context("sample lacks unchanged content")?;
        let samples = vec![sample; REGISTERED_BATCH_SIZE];
        let targets = samples
            .iter()
            .flat_map(|sample| sample.transition.next.pixels.iter().map(|v| u32::from(*v)))
            .collect::<Vec<_>>();
        let mut values = vec![0.0f32; targets.len() * PALETTE_SIZE];
        for (offset, target) in targets.iter().enumerate() {
            values[offset * PALETTE_SIZE + *target as usize] = 6.0;
        }
        let offset = pixel * PALETTE_SIZE;
        values[offset + targets[pixel] as usize] = 0.0;
        values[offset + (targets[pixel] as usize + 1) % PALETTE_SIZE] = 6.0;
        let logits = candle_core::Var::from_tensor(&Tensor::from_vec(
            values,
            (REGISTERED_BATCH_SIZE, FRAME_SIDE, FRAME_SIDE, PALETTE_SIZE),
            &Device::Cpu,
        )?)?;
        let labels = Tensor::from_vec(
            targets,
            (REGISTERED_BATCH_SIZE, FRAME_SIDE, FRAME_SIDE),
            &Device::Cpu,
        )?;
        // Exercise the same CE primitive and detached logit capture used by the
        // production training loss, including a real backward with no update.
        let per_pixel = crate::p2::train::foundation_v2_unimix_ce(logits.as_tensor(), &labels)?;
        let capture = capture_logits(&logits.detach())?;
        let (masks, binding) = bind_prediction_masks(&samples, &capture)?;
        assert_eq!(masks.false_edit_count, 1);
        let loss = masked_prediction_loss(
            &per_pixel,
            masks.false_edit.clone(),
            masks.unchanged_count,
            1.0,
        )?;
        assert!(loss.to_scalar::<f32>()? > 0.0);
        let grads = loss.backward()?;
        let gradient = grads
            .get(logits.as_tensor())
            .context("missing prediction gradient")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(gradient[offset..offset + PALETTE_SIZE]
            .iter()
            .any(|v| *v != 0.0));
        assert!(gradient[..offset]
            .iter()
            .chain(&gradient[offset + PALETTE_SIZE..])
            .all(|v| *v == 0.0));
        assert_eq!(tensor_fingerprint(logits.as_tensor())?, binding.logits);

        let unrelated = capture_logits(&logits.affine(-1.0, 0.0)?)?;
        let (_, unrelated_binding) = bind_prediction_masks(&samples, &unrelated)?;
        assert_ne!(binding.mask_sha256, unrelated_binding.mask_sha256);
        let (_, rebound) = bind_prediction_masks(&samples, &capture)?;
        assert_eq!(binding, rebound);
        assert_eq!(
            prediction_mask_sha256(&masks.false_edit)?,
            binding.mask_sha256
        );
        Ok(())
    }

    #[test]
    fn cross_process_gradient_control_excludes_hashes_but_not_integer_counts() -> Result<()> {
        let expected = cell(0, 0);
        let mut observed = expected.clone();
        observed
            .prediction_logits_fingerprint
            .as_mut()
            .unwrap()
            .sha256 = "different".into();
        observed.false_edit_mask_sha256 = "different-mask".into();
        observed.mask_binding.as_mut().unwrap().logits.sha256 = "different-binding".into();
        observed.mask_binding.as_mut().unwrap().mask_sha256 = "different-binding-mask".into();
        observed.training_raw_argmax_disagreement_pixels = 17;
        ensure_gradient_control_matches(&expected, &observed)?;
        observed.false_edit_pixels += 1;
        assert!(ensure_gradient_control_matches(&expected, &observed).is_err());
        Ok(())
    }

    #[test]
    fn completion_binding_rejects_logit_bit_mask_bit_and_pixel_corruption() -> Result<()> {
        let valid = cell(0, 0);
        ensure_same_forward_binding(&valid)?;
        let mut corrupted = valid.clone();
        corrupted
            .prediction_logits_fingerprint
            .as_mut()
            .unwrap()
            .sha256
            .push('1');
        assert!(ensure_same_forward_binding(&corrupted).is_err());
        corrupted = valid.clone();
        corrupted.false_edit_mask_sha256.push('1');
        assert!(ensure_same_forward_binding(&corrupted).is_err());
        corrupted = valid.clone();
        corrupted.false_edit_pixels += 1;
        assert!(ensure_same_forward_binding(&corrupted).is_err());
        corrupted = valid.clone();
        corrupted.mask_binding = None;
        assert!(ensure_same_forward_binding(&corrupted).is_err());
        Ok(())
    }

    #[test]
    fn classifier_rejects_legacy_labels_and_ignores_descriptive_disagreement() -> Result<()> {
        let mut cells = GRADIENT_STEPS
            .into_iter()
            .flat_map(|step| (0..TRAIN_BATCHES).map(move |position| cell(step, position)))
            .collect::<Vec<_>>();
        let (mut scores, mut persistent) = classification_context();
        let baseline = classify(&cells, &scores, &persistent)?;
        for cell in &mut cells {
            cell.training_raw_argmax_disagreement_pixels = 100;
        }
        assert_eq!(baseline, classify(&cells, &scores, &persistent)?);
        scores[0].train_anatomy.as_mut().unwrap().population = LEGACY_POPULATION.into();
        assert!(classify(&cells, &scores, &persistent).is_err());
        scores[0].train_anatomy.as_mut().unwrap().population = PRIMARY_POPULATION.into();
        persistent[0].population = LEGACY_POPULATION.into();
        assert!(classify(&cells, &scores, &persistent).is_err());
        persistent[0].population = PRIMARY_POPULATION.into();
        cells[0].mask_population = LEGACY_POPULATION.into();
        assert!(classify(&cells, &scores, &persistent).is_err());
        Ok(())
    }

    #[test]
    fn descriptive_primary_mismatch_does_not_fail_completion_binding() -> Result<()> {
        let mut report = minimal_report(false);
        let cell = cell(0, 0);
        let fingerprint = cell.prediction_logits_fingerprint.clone().unwrap();
        for (name, right) in [
            ("d1_primary_vs_d2_primary", PRIMARY_POPULATION),
            ("d1_primary_vs_legacy", LEGACY_POPULATION),
        ] {
            let mut result = seam_comparison(false, 13);
            result.left = PRIMARY_POPULATION.into();
            result.right = right.into();
            report.comparisons.push(DiagnosticLogitComparison {
                snapshot_step: 0,
                batch_position: 0,
                split: "train".into(),
                comparison: name.into(),
                left_logits: fingerprint.clone(),
                right_logits: TensorFingerprint {
                    sha256: "different-forward".into(),
                    ..fingerprint.clone()
                },
                result,
            });
        }
        report.gradients.push(cell);
        ensure_same_forward_binding(&report.gradients[0])?;
        ensure_comparison_grid(&report)?;
        report.comparisons.pop();
        assert!(ensure_comparison_grid(&report).is_err());
        Ok(())
    }

    #[test]
    fn primary_objective_uses_registered_step_and_position_seed() {
        let cfg = TrainConfig::default();
        let objective = diagnostic_objective(&cfg, 0.2, 1024, 7);
        assert_eq!(objective.sigreg_seed, REGISTERED_SEED + 1024 + 7);
        assert_eq!(objective.ep_weight, 0.2);
        assert_eq!(objective.sigreg_projections, REGISTERED_SIGREG_PROJECTIONS);
        assert_eq!(objective.sigreg_knots, REGISTERED_SIGREG_KNOTS);
        assert!(!objective.rollout_enabled);
        assert!(objective.capture_pred_per_pixel);
    }

    #[test]
    fn score_population_labels_round_trip_and_default_for_sealed_parent() -> Result<()> {
        let current = empty_rescore(0);
        let round_trip: SnapshotRescore = serde_json::from_value(serde_json::to_value(&current)?)?;
        assert_eq!(round_trip.primary_population, "primary_same_forward");
        assert_eq!(round_trip.legacy_population, "legacy_g_chunk32");

        let mut legacy = serde_json::to_value(current)?;
        let object = legacy.as_object_mut().context("rescore fixture")?;
        for field in [
            "legacy_population",
            "primary_population",
            "anatomy_population",
            "primary_train",
            "primary_heldout",
        ] {
            object.remove(field);
        }
        let legacy: SnapshotRescore = serde_json::from_value(legacy)?;
        assert!(legacy.legacy_population.is_empty());
        assert!(legacy.primary_train.is_empty());
        Ok(())
    }

    #[test]
    fn sealed_report_defaults_preserve_identity_and_nested_population_labels() -> Result<()> {
        let mut report = minimal_report(false);
        report.gradients.push(cell(0, 0));
        (report.rescoring, report.persistence) = classification_context();
        let score = PerBatchScore {
            population: PRIMARY_POPULATION.into(),
            batch_position: 0,
            batch_index: 0,
            raw: ExactMetrics {
                rows: 1,
                changed_rows: 1,
                changed_exact: 0,
                full_exact: 0,
                all_row_exact: 0,
                unchanged_pixels: 1,
                false_edit_pixels: 1,
                false_edit_rows: 1,
                changed_exact_fraction: 0.0,
                full_exact_fraction: 0.0,
                all_row_exact_fraction: 0.0,
                false_edit_rate: 1.0,
            },
            action_routed: false,
            raw_full_exact_branches: 0,
            reproduced_distinct_changed_classes: 0,
            action6_changed_full_exact: 0,
            action6_coordinate_routed: false,
        };
        report.rescoring[0].primary_train.push(score);
        let current = serde_json::to_value(&report)?;
        let round_trip: FrozenDiagnosticReport = serde_json::from_value(current.clone())?;
        assert_eq!(round_trip, report);
        assert_eq!(
            round_trip.rescoring[0].primary_train[0].population,
            PRIMARY_POPULATION
        );
        assert_eq!(
            round_trip.rescoring[0]
                .train_anatomy
                .as_ref()
                .unwrap()
                .population,
            PRIMARY_POPULATION
        );
        assert_eq!(round_trip.persistence[0].population, PRIMARY_POPULATION);

        fn strip_additions(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Object(object) => {
                    for key in [
                        "population",
                        "mask_population",
                        "prediction_logits_fingerprint",
                        "false_edit_mask_sha256",
                        "mask_binding",
                        "comparisons",
                        "comparison_seconds",
                        "legacy_population",
                        "primary_population",
                        "anatomy_population",
                        "primary_train",
                        "primary_heldout",
                    ] {
                        object.remove(key);
                    }
                    for child in object.values_mut() {
                        strip_additions(child);
                    }
                }
                serde_json::Value::Array(array) => {
                    for child in array {
                        strip_additions(child);
                    }
                }
                _ => {}
            }
        }
        let mut legacy = current;
        strip_additions(&mut legacy);
        // The report-level population census is an original field, unrelated
        // to the new string-valued labels stripped from nested objects.
        legacy
            .as_object_mut()
            .unwrap()
            .insert("population".into(), serde_json::Value::Null);
        let parsed: FrozenDiagnosticReport = serde_json::from_value(legacy)?;
        assert_eq!(diagnostic_identity(&report)?, diagnostic_identity(&parsed)?);
        assert!(parsed.gradients[0].mask_binding.is_none());
        assert!(parsed.persistence[0].population.is_empty());
        assert!(parsed.comparisons.is_empty());
        Ok(())
    }

    #[test]
    fn registered_runtime_forecast_counts_all_twenty_five_d1_cells() -> Result<()> {
        let report = minimal_report(false);
        let estimate = runtime_estimate(&report)?;
        assert_eq!(estimate.estimated_device_seconds, 81.0);
        Ok(())
    }

    #[test]
    fn persistence_uses_final_set_as_its_denominator() -> Result<()> {
        let sets = BTreeMap::from([
            (1024, BTreeSet::from([1, 2, 3])),
            (1536, BTreeSet::from([2, 3, 4])),
            (2048, BTreeSet::from([3, 4, 5, 6])),
        ]);
        let report = persistence("train", &sets)?;
        assert_eq!(report.intersection_all_three, 1);
        assert_eq!(report.persistent_1536_to_2048, 2);
        assert_eq!(report.persistence_1536_to_2048, Some(0.5));
        assert_eq!(report.new_1536_to_2048, 2);
        assert_eq!(report.resolved_1536_to_2048, 1);
        assert!(persistence("train", &BTreeMap::new()).is_err());
        Ok(())
    }

    #[test]
    fn reconstruction_checks_tensor_residual_not_only_norm() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let var = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &device)?)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("test.bias".into(), var.clone());
        let mut direct = GradStore::default();
        direct.insert(var.as_tensor(), Tensor::new(&[3f32, 4.0], &device)?);
        let matching = clone_grad_store(&direct, &varmap);
        assert!(reconstruction_check(&direct, &matching, &varmap)?.passed);
        let mut rotated = GradStore::default();
        rotated.insert(var.as_tensor(), Tensor::new(&[4f32, 3.0], &device)?);
        assert!(!reconstruction_check(&direct, &rotated, &varmap)?.passed);
        assert_eq!(
            bounded_cosine(&GradStore::default(), &direct, &varmap, OptimizerRoute::All)?,
            None
        );
        Ok(())
    }

    #[test]
    fn reconstruction_serialization_retains_reference_and_absolute_residual() -> Result<()> {
        let varmap = VarMap::new();
        let var = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &Device::Cpu)?)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("test.bias".into(), var.clone());
        let mut direct = GradStore::default();
        direct.insert(var.as_tensor(), Tensor::new(&[3f32, 4.0], &Device::Cpu)?);
        let mut reconstructed = GradStore::default();
        reconstructed.insert(var.as_tensor(), Tensor::new(&[4f32, 3.0], &Device::Cpu)?);
        let check = reconstruction_check(&direct, &reconstructed, &varmap)?;
        assert!(!check.passed);
        let matching = reconstruction_check(&direct, &direct, &varmap)?;
        ensure_reconstructions(&matching, &matching)?;
        let error = ensure_reconstructions(&check, &matching)
            .unwrap_err()
            .to_string();
        let serialized = error
            .strip_prefix("gradient component reconstruction failed: ")
            .unwrap();
        let failure: serde_json::Value = serde_json::from_str(serialized)?;
        assert_eq!(failure["full"], serde_json::to_value(&check)?);
        assert_eq!(failure["prediction"], serde_json::to_value(&matching)?);
        let value = serde_json::to_value(check)?;
        assert_eq!(value["global_reference_l2"], 5.0);
        assert!(
            (value["global_absolute_residual_l2"].as_f64().unwrap() - 2f64.sqrt()).abs() < 1e-6
        );
        assert_eq!(value["muon_reference_l2"], 0.0);
        assert_eq!(value["muon_absolute_residual_l2"], 0.0);
        Ok(())
    }

    #[test]
    fn reconstruction_capture_preserves_relative_and_near_zero_boundaries() -> Result<()> {
        let varmap = VarMap::new();
        let var = Var::from_tensor(&Tensor::zeros((1,), DType::F32, &Device::Cpu)?)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("test.bias".into(), var.clone());
        let gradient = |value: f32| -> Result<GradStore> {
            let mut store = GradStore::default();
            store.insert(var.as_tensor(), Tensor::new(&[value], &Device::Cpu)?);
            Ok(store)
        };
        for (reference, delta, expected) in [
            (1f32, 5e-6, true),
            (1.0, 2e-5, false),
            (0.0, 5e-7, true),
            (0.0, 2e-6, false),
        ] {
            let check = reconstruction_check(
                &gradient(reference)?,
                &gradient(reference + delta)?,
                &varmap,
            )?;
            assert_eq!(check.passed, expected);
            assert_eq!(check.global_reference_l2, f64::from(reference));
            assert!(check.global_absolute_residual_l2 > 0.0);
        }
        Ok(())
    }

    #[test]
    fn ns_muon_cosine_counts_full_only_parameters_in_full_norm() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let first = Var::from_tensor(&Tensor::zeros((8, 8), DType::F32, &device)?)?;
        let second = Var::from_tensor(&Tensor::zeros((8, 8), DType::F32, &device)?)?;
        {
            let mut data = varmap.data().lock().unwrap();
            data.insert("block.first.weight".into(), first.clone());
            data.insert("block.second.weight".into(), second.clone());
        }
        let gradient = Tensor::arange(0f32, 64f32, &device)?.reshape((8, 8))?;
        let mut full = GradStore::default();
        full.insert(first.as_tensor(), gradient.clone());
        full.insert(second.as_tensor(), gradient.clone());
        let mut prediction = GradStore::default();
        prediction.insert(first.as_tensor(), gradient);
        let cosine = ns_muon_cosine(&full, &prediction, &varmap)?.context("defined cosine")?;
        assert!(cosine > 0.6 && cosine < 0.8, "unexpected cosine {cosine}");
        Ok(())
    }
}
