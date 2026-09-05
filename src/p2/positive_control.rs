//! Raw-weight, fixed-batch Foundation-v2 positive control.
//!
//! This diagnostic is registered in
//! `docs/research/2026-09-05-v6-fixed-batch-positive-control.md`. It never
//! reads public ARC data and has no checkpoint-promotion authority.

use crate::gpu_lock::TrainPidGuard;
use crate::p2::bf16_falsifier::write_json_report;
use crate::p2::context_wiring::{
    external_manifest_paths, file_sha256_hex, identity_frame_sha256, open_diagnostic_device,
    open_run_root, registered_provenance_guard, same_build_identity, seal_run_root, unix_seconds,
    verify_manifest, verify_manifest_sidecar, verify_no_input_drift, GpuIdentity, LifecycleRecord,
    EVIDENCE_CLASS, FAILED_EVIDENCE_CLASS, LIFECYCLE_COMPLETE, LIFECYCLE_FAILED, LIFECYCLE_RUNNING,
    REPORT_FILE, RUN_CLASS_PREFLIGHT, RUN_CLASS_REGISTERED,
};
use crate::p2::data::{
    adaptation_v6_stream_schedule, compose_mixed_stream_batch, compose_rollout_fragment_batch,
    gameplay_rows, MixedStreamBatch, MixedStreamConfig, V5DataSplit, V5Sample, FRAME_SIDE,
};
use crate::p2::eval::{
    evaluate_gate_support_with_v5_provenance, raw_one_step_predictions, GateSupportMetrics,
};
use crate::p2::evidence::{launch_provenance, LaunchProvenance};
use crate::p2::experiment::TrainingRecipe;
use crate::p2::model::WorldModel;
use crate::p2::optimizer::{
    accumulate_parameter_gradients, clip_gradients_gpu_with_stats, CheckpointHybridOptimizer,
    ModelEma,
};
use crate::p2::semantic_eval::shuffled_action_control_population;
use crate::p2::train::{
    adam_params, event_slot_weight_tensor, foundation_v2_dedicated_rollout_loss,
    foundation_v2_ep_weight_update, foundation_v2_loss_values,
    foundation_v2_training_loss_with_event_weights, foundation_v2_wsd_learning_rate,
    gradient_l2_for_parameter_prefix, load_train_config, load_varmap_exact,
    prepare_foundation_v2_batch_host, retain_parameter_gradients, sync_cuda_device,
    training_content_batch_digest, FoundationV2LossMeans, FoundationV2ObjectiveConfig,
    PreparedFoundationV2BatchHost, SplitCeWeighting, TrainConfig,
};
use anyhow::{ensure, Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Args;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const POSITIVE_CONTROL_SCHEMA: &str = "p2.v6_fixed_batch_positive_control.v1";
pub const REGISTERED_CHECKPOINT_SHA256: &str =
    "0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79";
pub const REGISTERED_TRAIN_CONFIG_SHA256: &str =
    "874d53e53e68cfb5dbaada83bf25b5558f2874ae23f3af62997e13ec1263f3c1";
pub const REGISTERED_UPDATES: usize = 1024;
pub const PREFLIGHT_UPDATES: usize = 2;
pub const REGISTERED_MAIN_ROWS: usize = 128;
pub const REGISTERED_CHANGED_ROWS: usize = 95;
pub const REGISTERED_FACTUAL_GROUPS: usize = 1;
pub const REGISTERED_FACTUAL_GROUP_ROWS: usize = 10;
pub const REGISTERED_DISTINCT_CHANGED_OUTCOMES: usize = 8;
pub const REGISTERED_ROLLOUT_FRAGMENTS: usize = 16;
pub const REGISTERED_ROLLOUT_ROWS: usize = 32;
pub const REGISTERED_SIGREG_PROJECTIONS: usize = 1024;
pub const REGISTERED_SIGREG_KNOTS: usize = 17;
pub const MAX_WALL_TIME: Duration = Duration::from_secs(90 * 60);
const MAX_GRAD_NORM: f64 = 1.0;
const ROLLOUT_SEED_DOMAIN: u64 = 0xA011_0A77_0000_0002;
const COMMAND_TAG: &str = "p2-v6-fixed-batch-positive-control";
const FULL_OBJECTIVE: &str = "full_objective";
const PREDICTION_ONLY: &str = "prediction_only";
const OUTCOME_PASS: &str = "same_row_action_conditioned_fit";
const OUTCOME_PARTIAL: &str = "partial_fit";
const OUTCOME_WITHOUT_ACTION: &str = "fit_without_action";
const OUTCOME_FAIL: &str = "fail";
const OUTCOME_PREFLIGHT: &str = "unregistered_preflight_complete";
const SNAPSHOT_STEPS: [usize; 7] = [0, 1, 128, 256, 512, 768, 1024];
const REQUIRED_POSITIVE_ROUTES: [&str; 9] = [
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
const REQUIRED_ZERO_ROUTES: [&str; 3] = ["coord_proj.", "grounding_head.decoder.", "prefix_head."];

#[derive(Debug, Clone, Args)]
pub struct P2FixedBatchPositiveControlArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,

    #[arg(long)]
    pub train_config: PathBuf,

    #[arg(long, default_value = "cuda")]
    pub device: String,

    #[arg(long)]
    pub output_root: PathBuf,

    /// Registered evidence requires 1024; an unregistered preflight requires 2.
    #[arg(long, default_value_t = PREFLIGHT_UPDATES)]
    pub max_updates: usize,

    #[arg(long, default_value_t = false)]
    pub registered: bool,

    /// Conditional Q arm. A registered full-objective FAIL report is required.
    #[arg(long, default_value_t = false)]
    pub prediction_only: bool,

    /// Sealed two-update report from this exact binary; required for registered P.
    #[arg(long)]
    pub preflight_report: Option<PathBuf>,

    /// Sealed registered P report with outcome FAIL; required for registered Q.
    #[arg(long)]
    pub parent_full_report: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositiveControlSpec {
    pub arm: String,
    pub max_updates: usize,
    pub total_schedule_steps: usize,
    pub snapshot_steps: Vec<usize>,
    pub sigreg_projections: usize,
    pub sigreg_knots: usize,
    pub sigreg_seed_contract: String,
    pub fail_on_nonfinite: bool,
    pub max_wall_seconds: u64,
}

impl PositiveControlSpec {
    fn from_args(args: &P2FixedBatchPositiveControlArgs) -> Self {
        let snapshots = if args.max_updates == PREFLIGHT_UPDATES {
            vec![0, 1, 2]
        } else {
            SNAPSHOT_STEPS.to_vec()
        };
        Self {
            arm: if args.prediction_only {
                PREDICTION_ONLY.into()
            } else {
                FULL_OBJECTIVE.into()
            },
            max_updates: args.max_updates,
            total_schedule_steps: 2048,
            snapshot_steps: snapshots,
            sigreg_projections: REGISTERED_SIGREG_PROJECTIONS,
            sigreg_knots: REGISTERED_SIGREG_KNOTS,
            sigreg_seed_contract: "seed.wrapping_add(zero_based_update)".into(),
            fail_on_nonfinite: true,
            max_wall_seconds: MAX_WALL_TIME.as_secs(),
        }
    }

    fn validate(&self, registered: bool) -> Result<()> {
        let expected_updates = if registered {
            REGISTERED_UPDATES
        } else {
            PREFLIGHT_UPDATES
        };
        ensure!(
            self.max_updates == expected_updates,
            "{} run requires exactly {expected_updates} updates",
            if registered {
                "registered"
            } else {
                "preflight"
            }
        );
        ensure!(
            self.arm == FULL_OBJECTIVE || self.arm == PREDICTION_ONLY,
            "unknown positive-control arm {}",
            self.arm
        );
        ensure!(
            registered || self.arm == FULL_OBJECTIVE,
            "prediction-only work is forbidden before a registered P-arm FAIL"
        );
        ensure!(
            self.sigreg_projections == REGISTERED_SIGREG_PROJECTIONS
                && self.sigreg_knots == REGISTERED_SIGREG_KNOTS,
            "SIGReg geometry drifted from the registered contract"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationRecord {
    pub main_rows: usize,
    pub changed_rows: usize,
    pub factual_group_ranges: Vec<[usize; 2]>,
    pub factual_group_rows: usize,
    pub distinct_changed_outcomes: usize,
    pub identical_current_frames: bool,
    pub identical_contexts: bool,
    pub identical_goal_features: bool,
    pub identical_operator_conditioning: bool,
    pub operator_conditioning_width: usize,
    pub operator_conditioning_sha256: String,
    pub main_content_digest: String,
    pub rollout_content_digest: String,
    pub main_population_sha256: String,
    pub rollout_population_sha256: String,
    pub rollout_rows: usize,
    pub rollout_fragments: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteNorms {
    pub norms: BTreeMap<String, f64>,
    pub positive_pass: bool,
    pub zero_pass: bool,
}

impl RouteNorms {
    fn passed(&self) -> bool {
        self.positive_pass && self.zero_pass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutePremise {
    pub prediction_unclipped: RouteNorms,
    pub combined_clipped: RouteNorms,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionExactMetrics {
    pub rows: usize,
    pub changed_rows: usize,
    pub changed_exact: usize,
    pub full_exact: usize,
    pub changed_exact_fraction: f64,
    pub full_exact_fraction: f64,
    pub non_background_changed_exact: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionClassMetrics {
    pub group_rows: usize,
    pub changed_group_rows: usize,
    pub distinct_changed_classes: usize,
    pub maximum_changed_class_multiplicity: usize,
    pub raw_full_exact_branches: usize,
    pub reproduced_distinct_changed_classes: usize,
    pub reproduced_class_sha256: Vec<String>,
    pub action_routed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualActionMetrics {
    pub rows: usize,
    pub eligible_rows: usize,
    pub changed_tuples: usize,
    pub outcome_changing_tuples: usize,
    pub disagreement_pixels: usize,
    pub counterfactual_target_accuracy: Option<f64>,
    pub factual_target_accuracy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotMetrics {
    pub step: usize,
    pub raw_checkpoint: PathBuf,
    pub raw_sha256: String,
    pub ema_checkpoint: PathBuf,
    pub ema_sha256: String,
    pub ep_weight: f64,
    pub raw: PredictionExactMetrics,
    pub copy_control: PredictionExactMetrics,
    pub background_control: PredictionExactMetrics,
    pub direct_target_control: PredictionExactMetrics,
    pub factual_action: ActionClassMetrics,
    pub counterfactual_action: CounterfactualActionMetrics,
    pub gate_evaluator: GateSupportMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub step: usize,
    pub sigreg_seed: u64,
    pub learning_rate: f64,
    pub ep_weight: f64,
    pub rollout_fragments: usize,
    pub losses: FoundationV2LossMeans,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateOneBinding {
    pub passed: bool,
    pub checks: BTreeMap<String, bool>,
    pub observed: UpdateRecord,
    pub changed_pixels: usize,
    pub unchanged_pixels: usize,
    pub changed_coefficient: f64,
    pub coefficient_mass: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositiveControlVerdict {
    pub outcome: String,
    pub broad_fit_768: Option<bool>,
    pub broad_fit_1024: Option<bool>,
    pub monotone_changed_exact: Option<bool>,
    pub action_routed_768: Option<bool>,
    pub action_routed_1024: Option<bool>,
    pub next_action: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceBinding {
    pub report: PathBuf,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    pub identity_root: String,
}

#[derive(Debug, Clone)]
struct ParentArmReference {
    prediction_route: RouteNorms,
    prediction_ce: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositiveControlTiming {
    pub population_seconds: f64,
    pub training_seconds: f64,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositiveControlReport {
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
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub spec: PositiveControlSpec,
    pub population: Option<PopulationRecord>,
    pub preflight: Option<EvidenceBinding>,
    pub parent_full_arm: Option<EvidenceBinding>,
    pub route_premise: Option<RoutePremise>,
    pub update_one_binding: Option<UpdateOneBinding>,
    pub updates_completed: usize,
    pub snapshots: Vec<SnapshotMetrics>,
    pub loss_log_sha256: Option<String>,
    pub verdict: Option<PositiveControlVerdict>,
    pub timing: PositiveControlTiming,
    pub identity_root: String,
    pub error: Option<String>,
}

fn bytes_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("write to string");
    }
    out
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn registered_sigreg_seed(seed: u64, zero_based_update: usize) -> u64 {
    seed.wrapping_add(zero_based_update as u64)
}

fn board_changed(sample: &V5Sample) -> bool {
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    sample.transition.current.pixels[..pixels] != sample.transition.next.pixels[..pixels]
}

fn population_record(
    root: &Path,
    main: &MixedStreamBatch,
    rollout: &MixedStreamBatch,
    host: &PreparedFoundationV2BatchHost,
) -> Result<PopulationRecord> {
    let population_dir = root.join("population");
    fs::create_dir(&population_dir)?;
    let main_path = population_dir.join("main.json");
    let rollout_path = population_dir.join("rollout.json");
    write_json_report(&main_path, main)?;
    write_json_report(&rollout_path, rollout)?;

    ensure!(
        main.samples().len() == REGISTERED_MAIN_ROWS,
        "main population has {} rows, expected {REGISTERED_MAIN_ROWS}",
        main.samples().len()
    );
    let changed_rows = main
        .samples()
        .iter()
        .filter(|row| board_changed(row))
        .count();
    ensure!(
        changed_rows >= 32,
        "main population has only {changed_rows} changed rows; registered floor is 32"
    );
    ensure!(
        changed_rows == REGISTERED_CHANGED_ROWS,
        "main population has {changed_rows} changed rows, expected {REGISTERED_CHANGED_ROWS}"
    );
    ensure!(
        main.factual_group_ranges().len() == REGISTERED_FACTUAL_GROUPS,
        "main population has {} factual groups, expected {REGISTERED_FACTUAL_GROUPS}",
        main.factual_group_ranges().len()
    );
    let group_range = &main.factual_group_ranges()[0];
    ensure!(
        group_range.len() == REGISTERED_FACTUAL_GROUP_ROWS,
        "factual group has {} rows, expected {REGISTERED_FACTUAL_GROUP_ROWS}",
        group_range.len()
    );
    let group = &main.samples()[group_range.clone()];
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    let first = &group[0];
    let identical_current_frames = group.iter().all(|row| {
        row.transition.current.pixels[..pixels] == first.transition.current.pixels[..pixels]
    });
    let identical_contexts = group
        .iter()
        .all(|row| row.transition.context == first.transition.context);
    let identical_goal_features = group
        .iter()
        .all(|row| row.transition.goal_features == first.transition.goal_features);
    let distinct_changed_outcomes = group
        .iter()
        .filter(|row| board_changed(row))
        .map(|row| row.transition.next.pixels[..pixels].to_vec())
        .collect::<BTreeSet<_>>()
        .len();
    ensure!(
        distinct_changed_outcomes == REGISTERED_DISTINCT_CHANGED_OUTCOMES,
        "factual group has {distinct_changed_outcomes} distinct changed outcomes, expected {REGISTERED_DISTINCT_CHANGED_OUTCOMES}"
    );

    let operator = host.operator_conditioning();
    ensure!(
        host.batch_size() == REGISTERED_MAIN_ROWS,
        "host batch size drift"
    );
    ensure!(
        operator.len().is_multiple_of(host.batch_size()),
        "operator-conditioning vector shape is not row-major"
    );
    let operator_width = operator.len() / host.batch_size();
    let first_operator =
        &operator[group_range.start * operator_width..(group_range.start + 1) * operator_width];
    let identical_operator_conditioning = group_range
        .clone()
        .all(|row| operator[row * operator_width..(row + 1) * operator_width] == *first_operator);
    ensure!(
        identical_current_frames
            && identical_contexts
            && identical_goal_features
            && identical_operator_conditioning,
        "factual group differs in a registered non-action input"
    );
    ensure!(
        first.transition.context.is_empty(),
        "registered factual group context is not empty"
    );
    ensure!(
        operator_width > 0
            && first_operator.first() == Some(&1.0)
            && first_operator[1..].iter().all(|value| *value == 0.0),
        "registered operator conditioning is not UNKNOWN one-hot"
    );
    let operator_bytes = operator
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();

    ensure!(
        rollout.samples().len() == REGISTERED_ROLLOUT_ROWS,
        "rollout population has {} rows, expected {REGISTERED_ROLLOUT_ROWS}",
        rollout.samples().len()
    );
    let main_digest = training_content_batch_digest(main.transitions(), main.content_masks())?;
    let rollout_digest =
        training_content_batch_digest(rollout.transitions(), rollout.content_masks())?;
    Ok(PopulationRecord {
        main_rows: main.samples().len(),
        changed_rows,
        factual_group_ranges: main
            .factual_group_ranges()
            .iter()
            .map(|range| [range.start, range.end])
            .collect(),
        factual_group_rows: group.len(),
        distinct_changed_outcomes,
        identical_current_frames,
        identical_contexts,
        identical_goal_features,
        identical_operator_conditioning,
        operator_conditioning_width: operator_width,
        operator_conditioning_sha256: digest_bytes(&operator_bytes),
        main_content_digest: bytes_hex(&main_digest),
        rollout_content_digest: bytes_hex(&rollout_digest),
        main_population_sha256: file_sha256_hex(&main_path)?,
        rollout_population_sha256: file_sha256_hex(&rollout_path)?,
        rollout_rows: rollout.samples().len(),
        rollout_fragments: REGISTERED_ROLLOUT_FRAGMENTS,
    })
}

fn ensure_registered_config(cfg: &TrainConfig) -> Result<()> {
    ensure!(cfg.recipe == TrainingRecipe::FoundationV2, "recipe drift");
    ensure!(cfg.seed == 5 && cfg.init_seed == Some(5), "seed drift");
    ensure!(
        cfg.physical_batch == REGISTERED_MAIN_ROWS && cfg.grad_accum == 1,
        "batch schedule drift"
    );
    ensure!(
        cfg.hidden_dim == 128 && cfg.action_dim == 32,
        "model dimension drift"
    );
    ensure!(
        cfg.world_core_v4 && cfg.world_core_v6 && cfg.data_contract_v6,
        "V6 contract drift"
    );
    ensure!(
        cfg.spatial_action_field && !cfg.spatial_action_residual,
        "spatial-action contract drift"
    );
    ensure!(
        !cfg.bf16_conv && !cfg.bf16_recurrent_core,
        "registered arm must remain F32"
    );
    ensure!(
        cfg.split_ce_weighting == SplitCeWeighting::CurrentDouble
            && cfg.split_ce_changed_budget.is_none(),
        "CurrentDouble contract drift"
    );
    ensure!(
        cfg.lr == 1e-3
            && cfg.weight_decay == 0.01
            && cfg.muon_momentum == 0.95
            && cfg.muon_rms_scale == 0.2
            && cfg.rollout_weight == 0.02,
        "optimizer/objective scalar drift"
    );
    ensure!(
        cfg.steps_per_lesson == 2048
            && cfg.sigreg_projections == REGISTERED_SIGREG_PROJECTIONS
            && cfg.sigreg_knots == REGISTERED_SIGREG_KNOTS,
        "schedule or SIGReg drift"
    );
    Ok(())
}

fn route_norms(grads: &GradStore, varmap: &VarMap) -> Result<RouteNorms> {
    let mut norms = BTreeMap::new();
    for route in REQUIRED_POSITIVE_ROUTES {
        norms.insert(
            route.to_string(),
            gradient_l2_for_parameter_prefix(grads, varmap, route)?,
        );
    }
    for route in REQUIRED_ZERO_ROUTES {
        norms.insert(
            route.to_string(),
            gradient_l2_or_absent_zero(grads, varmap, route)?,
        );
    }
    let positive_pass = REQUIRED_POSITIVE_ROUTES.iter().all(|route| {
        norms
            .get(*route)
            .is_some_and(|value| value.is_finite() && *value > 0.0)
    });
    let zero_pass = REQUIRED_ZERO_ROUTES
        .iter()
        .all(|route| norms.get(*route) == Some(&0.0));
    Ok(RouteNorms {
        norms,
        positive_pass,
        zero_pass,
    })
}

/// A deliberately bypassed topology has no `GradStore` entry. Require the
/// named parameters to exist, then treat that absence as the registered zero
/// rather than weakening the topology assertion.
fn gradient_l2_or_absent_zero(grads: &GradStore, varmap: &VarMap, prefix: &str) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    let matching = data
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .collect::<Vec<_>>();
    ensure!(
        !matching.is_empty(),
        "no model parameters found for zero-route prefix {prefix}"
    );
    let has_gradient_entry = matching
        .iter()
        .any(|(_, var)| grads.get(var.as_tensor()).is_some());
    drop(data);
    if has_gradient_entry {
        gradient_l2_for_parameter_prefix(grads, varmap, prefix)
    } else {
        Ok(0.0)
    }
}

fn exact_metrics(samples: &[V5Sample], predictions: &[Vec<u8>]) -> Result<PredictionExactMetrics> {
    ensure!(
        samples.len() == predictions.len(),
        "prediction row mismatch"
    );
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    let mut changed_rows = 0usize;
    let mut changed_exact = 0usize;
    let mut full_exact = 0usize;
    let mut non_background_changed_exact = 0usize;
    for (sample, prediction) in samples.iter().zip(predictions) {
        ensure!(
            prediction.len() == pixels,
            "prediction pixel width mismatch"
        );
        let current = &sample.transition.current.pixels[..pixels];
        let target = &sample.transition.next.pixels[..pixels];
        if current == target {
            continue;
        }
        changed_rows += 1;
        let changed_ok = current
            .iter()
            .zip(target)
            .zip(prediction)
            .all(|((before, after), predicted)| before == after || after == predicted);
        changed_exact += usize::from(changed_ok);
        full_exact += usize::from(prediction.as_slice() == target);
        let empty = sample.provenance.operator.empty_color;
        let non_background_target = current
            .iter()
            .zip(target)
            .any(|(before, after)| before != after && *after != empty);
        non_background_changed_exact += usize::from(changed_ok && non_background_target);
    }
    ensure!(changed_rows > 0, "exact scorer has no changed rows");
    Ok(PredictionExactMetrics {
        rows: samples.len(),
        changed_rows,
        changed_exact,
        full_exact,
        changed_exact_fraction: changed_exact as f64 / changed_rows as f64,
        full_exact_fraction: full_exact as f64 / changed_rows as f64,
        non_background_changed_exact,
    })
}

fn control_predictions(samples: &[V5Sample], kind: &str) -> Vec<Vec<u8>> {
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    samples
        .iter()
        .map(|sample| match kind {
            "copy" => sample.transition.current.pixels[..pixels].to_vec(),
            "background" => vec![sample.provenance.operator.empty_color; pixels],
            "target" => sample.transition.next.pixels[..pixels].to_vec(),
            _ => unreachable!("fixed control kind"),
        })
        .collect()
}

fn action_class_metrics(
    samples: &[V5Sample],
    group: Range<usize>,
    predictions: &[Vec<u8>],
) -> Result<ActionClassMetrics> {
    ensure!(
        group.end <= samples.len(),
        "factual group range out of bounds"
    );
    ensure!(
        predictions.len() == samples.len(),
        "prediction row mismatch"
    );
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    let mut multiplicities = BTreeMap::<Vec<u8>, usize>::new();
    let mut reproduced = BTreeSet::<Vec<u8>>::new();
    let mut changed_group_rows = 0usize;
    let mut raw_full_exact_branches = 0usize;
    for row in group.clone() {
        let sample = &samples[row];
        let current = &sample.transition.current.pixels[..pixels];
        let target = sample.transition.next.pixels[..pixels].to_vec();
        if current == target.as_slice() {
            continue;
        }
        changed_group_rows += 1;
        *multiplicities.entry(target.clone()).or_default() += 1;
        if predictions[row] == target {
            raw_full_exact_branches += 1;
            reproduced.insert(target);
        }
    }
    let reproduced_class_sha256 = reproduced
        .iter()
        .map(|target| digest_bytes(target))
        .collect::<Vec<_>>();
    Ok(ActionClassMetrics {
        group_rows: group.len(),
        changed_group_rows,
        distinct_changed_classes: multiplicities.len(),
        maximum_changed_class_multiplicity: multiplicities.values().copied().max().unwrap_or(0),
        raw_full_exact_branches,
        reproduced_distinct_changed_classes: reproduced.len(),
        reproduced_class_sha256,
        action_routed: reproduced.len() >= 2,
    })
}

fn counterfactual_action_metrics(
    model: &WorldModel,
    samples: &[V5Sample],
    device: &Device,
) -> Result<CounterfactualActionMetrics> {
    let transitions = samples
        .iter()
        .map(|sample| sample.transition.clone())
        .collect::<Vec<_>>();
    let provenance = samples
        .iter()
        .map(|sample| sample.provenance.clone())
        .collect::<Vec<_>>();
    let shuffled = shuffled_action_control_population(&transitions, Some(&provenance))?;
    let predictions = raw_one_step_predictions(model, &shuffled.samples, device)?;
    let pixels = gameplay_rows(true) * FRAME_SIDE;
    let mut outcome_changing_tuples = 0usize;
    let mut disagreement_pixels = 0usize;
    let mut counterfactual_correct = 0usize;
    let mut factual_correct = 0usize;
    for (row, ((factual, counterfactual), prediction)) in transitions
        .iter()
        .zip(&shuffled.counterfactual_next)
        .zip(&predictions)
        .enumerate()
    {
        if factual.action == shuffled.samples[row].action {
            continue;
        }
        let Some(counterfactual) = counterfactual else {
            continue;
        };
        let factual_target = &factual.next.pixels[..pixels];
        let counterfactual_target = &counterfactual.pixels[..pixels];
        if factual_target == counterfactual_target {
            continue;
        }
        outcome_changing_tuples += 1;
        for ((factual_pixel, counterfactual_pixel), predicted) in factual_target
            .iter()
            .zip(counterfactual_target)
            .zip(prediction)
        {
            if factual_pixel == counterfactual_pixel {
                continue;
            }
            disagreement_pixels += 1;
            counterfactual_correct += usize::from(predicted == counterfactual_pixel);
            factual_correct += usize::from(predicted == factual_pixel);
        }
    }
    Ok(CounterfactualActionMetrics {
        rows: samples.len(),
        eligible_rows: shuffled.eligible_rows,
        changed_tuples: shuffled.changed_tuples(&transitions),
        outcome_changing_tuples,
        disagreement_pixels,
        counterfactual_target_accuracy: (disagreement_pixels > 0)
            .then_some(counterfactual_correct as f64 / disagreement_pixels as f64),
        factual_target_accuracy: (disagreement_pixels > 0)
            .then_some(factual_correct as f64 / disagreement_pixels as f64),
    })
}

#[allow(clippy::too_many_arguments)]
fn save_snapshot(
    root: &Path,
    step: usize,
    ep_weight: f64,
    varmap: &VarMap,
    ema: &ModelEma,
    model: &WorldModel,
    main: &MixedStreamBatch,
    device: &Device,
) -> Result<SnapshotMetrics> {
    sync_cuda_device(device)?;
    let directory = root.join("snapshots").join(format!("step-{step:012}"));
    fs::create_dir_all(&directory)?;
    let raw_checkpoint = directory.join("model.safetensors");
    let ema_checkpoint = directory.join("ema.safetensors");
    varmap.save(&raw_checkpoint)?;
    ema.weights().save(&ema_checkpoint)?;
    let samples = main.samples();
    let transitions = samples
        .iter()
        .map(|sample| sample.transition.clone())
        .collect::<Vec<_>>();
    let masks = samples
        .iter()
        .map(|sample| sample.content_mask.clone())
        .collect::<Vec<_>>();
    let provenance = samples
        .iter()
        .map(|sample| sample.provenance.clone())
        .collect::<Vec<_>>();
    let predictions = raw_one_step_predictions(model, &transitions, device)?;
    let raw = exact_metrics(samples, &predictions)?;
    let copy_control = exact_metrics(samples, &control_predictions(samples, "copy"))?;
    let background_control = exact_metrics(samples, &control_predictions(samples, "background"))?;
    let direct_target_control = exact_metrics(samples, &control_predictions(samples, "target"))?;
    let factual_action = action_class_metrics(
        samples,
        main.factual_group_ranges()[0].clone(),
        &predictions,
    )?;
    let counterfactual_action = counterfactual_action_metrics(model, samples, device)?;
    let gate_evaluator =
        evaluate_gate_support_with_v5_provenance(model, &transitions, &masks, &provenance, device)?;
    sync_cuda_device(device)?;
    Ok(SnapshotMetrics {
        step,
        raw_checkpoint: raw_checkpoint
            .strip_prefix(root)
            .unwrap_or(&raw_checkpoint)
            .to_path_buf(),
        raw_sha256: file_sha256_hex(&raw_checkpoint)?,
        ema_checkpoint: ema_checkpoint
            .strip_prefix(root)
            .unwrap_or(&ema_checkpoint)
            .to_path_buf(),
        ema_sha256: file_sha256_hex(&ema_checkpoint)?,
        ep_weight,
        raw,
        copy_control,
        background_control,
        direct_target_control,
        factual_action,
        counterfactual_action,
        gate_evaluator,
    })
}

fn approximately(observed: f64, expected: f64) -> bool {
    (observed - expected).abs() <= 1e-6f64.max(expected.abs() * 1e-6)
}

fn update_one_binding(
    observed: UpdateRecord,
    changed_pixels: usize,
    unchanged_pixels: usize,
    changed_coefficient: f64,
    coefficient_mass: f64,
) -> UpdateOneBinding {
    let mut checks = BTreeMap::new();
    checks.insert(
        "total".into(),
        approximately(observed.losses.total, 175.38963317871094),
    );
    checks.insert(
        "pred_ce".into(),
        approximately(observed.losses.pred_ce, 155.89109802246094),
    );
    checks.insert(
        "rollout".into(),
        approximately(observed.losses.rollout, 1.3734819889068604),
    );
    checks.insert(
        "pre_clip_gradient_norm".into(),
        approximately(observed.losses.pre_clip_gradient_norm, 450.4061279296875),
    );
    checks.insert(
        "learning_rate".into(),
        approximately(observed.learning_rate, 0.000002),
    );
    checks.insert("changed_pixels".into(), changed_pixels == 176);
    checks.insert("unchanged_pixels".into(), unchanged_pixels == 524_112);
    checks.insert(
        "changed_coefficient".into(),
        approximately(changed_coefficient, 50.0),
    );
    checks.insert(
        "coefficient_mass".into(),
        approximately(coefficient_mass, 51.0),
    );
    checks.insert(
        "rollout_fragments".into(),
        observed.rollout_fragments == REGISTERED_ROLLOUT_FRAGMENTS,
    );
    UpdateOneBinding {
        passed: checks.values().all(|passed| *passed),
        checks,
        observed,
        changed_pixels,
        unchanged_pixels,
        changed_coefficient,
        coefficient_mass,
    }
}

fn classify_decision(
    broad_fit_768: bool,
    broad_fit_1024: bool,
    monotone_changed_exact: bool,
    action_routed_768: bool,
    action_routed_1024: bool,
) -> (&'static str, &'static str) {
    if broad_fit_768
        && broad_fit_1024
        && monotone_changed_exact
        && action_routed_768
        && action_routed_1024
    {
        (
            OUTCOME_PASS,
            "preregister a multi-batch generalization screen; A/C/D and public ARC stay blocked",
        )
    } else if action_routed_1024 {
        (
            OUTCOME_PARTIAL,
            "preregister the otherwise identical 2048-update extension",
        )
    } else if broad_fit_1024 {
        (
            OUTCOME_WITHOUT_ACTION,
            "diagnose action projection, FiLM, and spatial action conditioning",
        )
    } else {
        (
            OUTCOME_FAIL,
            "launch the registered matched prediction-only arm",
        )
    }
}

fn final_verdict(
    spec: &PositiveControlSpec,
    snapshots: &[SnapshotMetrics],
) -> Result<PositiveControlVerdict> {
    if spec.max_updates == PREFLIGHT_UPDATES {
        return Ok(PositiveControlVerdict {
            outcome: OUTCOME_PREFLIGHT.into(),
            broad_fit_768: None,
            broad_fit_1024: None,
            monotone_changed_exact: None,
            action_routed_768: None,
            action_routed_1024: None,
            next_action: "bind this sealed same-binary preflight to registered P".into(),
        });
    }
    let snapshot = |step| {
        snapshots
            .iter()
            .find(|snapshot| snapshot.step == step)
            .ok_or_else(|| anyhow::anyhow!("missing registered snapshot {step}"))
    };
    let s512 = snapshot(512)?;
    let s768 = snapshot(768)?;
    let s1024 = snapshot(1024)?;
    let broad_fit_768 = s768.raw.full_exact_fraction >= 0.5;
    let broad_fit_1024 = s1024.raw.full_exact_fraction >= 0.5;
    let monotone_changed_exact = s512.raw.changed_exact <= s768.raw.changed_exact
        && s768.raw.changed_exact <= s1024.raw.changed_exact;
    let action_routed_768 = s768.factual_action.action_routed;
    let action_routed_1024 = s1024.factual_action.action_routed;
    let (outcome, next_action) = classify_decision(
        broad_fit_768,
        broad_fit_1024,
        monotone_changed_exact,
        action_routed_768,
        action_routed_1024,
    );
    Ok(PositiveControlVerdict {
        outcome: outcome.into(),
        broad_fit_768: Some(broad_fit_768),
        broad_fit_1024: Some(broad_fit_1024),
        monotone_changed_exact: Some(monotone_changed_exact),
        action_routed_768: Some(action_routed_768),
        action_routed_1024: Some(action_routed_1024),
        next_action: next_action.into(),
    })
}

fn append_loss(root: &Path, update: &UpdateRecord) -> Result<()> {
    let path = root.join("loss_log.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    serde_json::to_writer(&mut file, update)?;
    writeln!(file)?;
    file.sync_data()?;
    Ok(())
}

fn report_identity(report: &PositiveControlReport) -> Result<String> {
    let population = report
        .population
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("identity requires population"))?;
    identity_frame_sha256(&[
        ("domain", POSITIVE_CONTROL_SCHEMA.as_bytes().to_vec()),
        ("checkpoint", report.checkpoint_sha256.as_bytes().to_vec()),
        ("config", report.train_config_sha256.as_bytes().to_vec()),
        (
            "source_revision",
            report.provenance.source_revision.as_bytes().to_vec(),
        ),
        (
            "binary",
            report.provenance.binary_sha256.as_bytes().to_vec(),
        ),
        (
            "main_population",
            population.main_population_sha256.as_bytes().to_vec(),
        ),
        (
            "rollout_population",
            population.rollout_population_sha256.as_bytes().to_vec(),
        ),
        ("spec", serde_json::to_vec(&report.spec)?),
        ("preflight", serde_json::to_vec(&report.preflight)?),
        ("parent_full", serde_json::to_vec(&report.parent_full_arm)?),
    ])
}

fn bind_report(path: &Path) -> Result<(PositiveControlReport, EvidenceBinding)> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize report {}", path.display()))?;
    let bytes = fs::read(&report_path)?;
    let report: PositiveControlReport =
        serde_json::from_slice(&bytes).context("parse positive-control report")?;
    let root = fs::canonicalize(&report.output_root)?;
    ensure!(
        report_path == fs::canonicalize(root.join(REPORT_FILE))?,
        "bound report is not inside its claimed root"
    );
    let (manifest, _) = external_manifest_paths(&root)?;
    let manifest = fs::canonicalize(manifest)?;
    let manifest_sha256 = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &manifest_sha256)?;
    ensure!(
        report.lifecycle.state == LIFECYCLE_COMPLETE
            && report.evidence_class == EVIDENCE_CLASS
            && report.lifecycle.evidence_class == EVIDENCE_CLASS
            && report.error.is_none(),
        "bound evidence did not complete cleanly"
    );
    let recomputed_identity = report_identity(&report)?;
    ensure!(
        !report.identity_root.is_empty() && report.identity_root == recomputed_identity,
        "bound report identity root is missing or invalid"
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

fn same_population(left: &PopulationRecord, right: &PopulationRecord) -> bool {
    left == right
}

fn bind_preflight(path: &Path, current: &PositiveControlReport) -> Result<EvidenceBinding> {
    let (preflight, binding) = bind_report(path)?;
    ensure!(
        preflight.schema == POSITIVE_CONTROL_SCHEMA,
        "preflight schema drift"
    );
    ensure!(
        !preflight.registered
            && preflight.run_class == RUN_CLASS_PREFLIGHT
            && preflight.spec.arm == FULL_OBJECTIVE
            && preflight.spec.max_updates == PREFLIGHT_UPDATES
            && preflight.device_is_cuda,
        "preflight is not the required two-update full-objective run"
    );
    ensure!(
        preflight.lifecycle.state == LIFECYCLE_COMPLETE
            && preflight.error.is_none()
            && preflight.verdict.as_ref().map(|v| v.outcome.as_str()) == Some(OUTCOME_PREFLIGHT),
        "preflight did not complete cleanly"
    );
    ensure!(
        preflight
            .route_premise
            .as_ref()
            .is_some_and(|route| route.passed)
            && preflight
                .update_one_binding
                .as_ref()
                .is_some_and(|binding| binding.passed),
        "preflight did not pass route and update-1 binding"
    );
    ensure!(
        preflight.checkpoint_sha256 == current.checkpoint_sha256
            && preflight.train_config_sha256 == current.train_config_sha256
            && same_population(
                preflight
                    .population
                    .as_ref()
                    .context("preflight population")?,
                current.population.as_ref().context("current population")?,
            )
            && same_build_identity(&preflight.provenance, &current.provenance),
        "preflight identity differs from the registered launch"
    );
    Ok(binding)
}

fn bind_failed_full_arm(
    path: &Path,
    current: &PositiveControlReport,
) -> Result<(PositiveControlReport, EvidenceBinding)> {
    let (parent, binding) = bind_report(path)?;
    ensure!(
        parent.schema == POSITIVE_CONTROL_SCHEMA
            && parent.registered
            && parent.run_class == RUN_CLASS_REGISTERED
            && parent.spec.arm == FULL_OBJECTIVE
            && parent.spec.max_updates == REGISTERED_UPDATES,
        "Q parent is not registered P"
    );
    ensure!(
        parent.lifecycle.state == LIFECYCLE_COMPLETE
            && parent.error.is_none()
            && parent.verdict.as_ref().map(|v| v.outcome.as_str()) == Some(OUTCOME_FAIL)
            && parent
                .route_premise
                .as_ref()
                .is_some_and(|route| route.passed)
            && parent
                .update_one_binding
                .as_ref()
                .is_some_and(|binding| binding.passed),
        "prediction-only arm requires a completed P outcome FAIL"
    );
    ensure!(
        parent.checkpoint_sha256 == current.checkpoint_sha256
            && parent.train_config_sha256 == current.train_config_sha256
            && same_population(
                parent.population.as_ref().context("P population")?,
                current.population.as_ref().context("Q population")?,
            )
            && same_build_identity(&parent.provenance, &current.provenance),
        "P and Q identities differ"
    );
    Ok((parent, binding))
}

fn compare_route_norms(left: &RouteNorms, right: &RouteNorms) -> bool {
    left.norms.len() == right.norms.len()
        && left.norms.iter().all(|(name, value)| {
            right
                .norms
                .get(name)
                .is_some_and(|other| approximately(*value, *other))
        })
}

fn ensure_operator_projection_zero(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    for name in [
        "operator_conditioning_proj.weight",
        "operator_conditioning_proj.bias",
    ] {
        let var = data
            .get(name)
            .with_context(|| format!("registered checkpoint lacks {name}"))?;
        let values = var
            .as_tensor()
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        ensure!(
            values.iter().all(|value| *value == 0.0),
            "registered step-0 {name} is not exactly zero"
        );
    }
    Ok(())
}

fn write_progress(root: &Path, report: &PositiveControlReport) -> Result<()> {
    write_json_report(&root.join("progress.json"), report)
}

#[allow(clippy::too_many_arguments)]
fn run_training(
    args: &P2FixedBatchPositiveControlArgs,
    report: &mut PositiveControlReport,
    cfg: &TrainConfig,
    main: &MixedStreamBatch,
    rollout: &MixedStreamBatch,
    host: &PreparedFoundationV2BatchHost,
    device: &Device,
    model: &WorldModel,
    varmap: &VarMap,
    mut optimizer: CheckpointHybridOptimizer,
    mut ema: ModelEma,
    parent_reference: Option<ParentArmReference>,
    started: Instant,
) -> Result<()> {
    let training_started = Instant::now();
    let event_slot_weights = event_slot_weight_tensor(device)?;
    report.snapshots.push(save_snapshot(
        &args.output_root,
        0,
        0.01,
        varmap,
        &ema,
        model,
        main,
        device,
    )?);
    write_progress(&args.output_root, report)?;
    let mut ep_weight = 0.01;
    let prediction_only = report.spec.arm == PREDICTION_ONLY;
    ensure!(
        prediction_only == parent_reference.is_some(),
        "conditional parent-route binding disagrees with arm"
    );

    for zero_based_update in 0..report.spec.max_updates {
        ensure!(
            started.elapsed() <= MAX_WALL_TIME,
            "positive-control wall-time cap exceeded"
        );
        let step = zero_based_update + 1;
        let (attached_rollout, rollout_fragments) =
            foundation_v2_dedicated_rollout_loss(model, rollout, device)?;
        ensure!(
            rollout_fragments == REGISTERED_ROLLOUT_FRAGMENTS,
            "rollout fragment count drifted at update {step}"
        );
        let rollout_loss = attached_rollout.detach();
        let mut rollout_grads = if prediction_only {
            None
        } else {
            let weighted = attached_rollout.affine(cfg.rollout_weight, 0.0)?;
            let grads = retain_parameter_gradients(weighted.backward()?, varmap)?;
            drop(weighted);
            Some(grads)
        };
        drop(attached_rollout);

        let sigreg_seed = registered_sigreg_seed(cfg.seed, zero_based_update);
        let mut losses = foundation_v2_training_loss_with_event_weights(
            model,
            main,
            host,
            device,
            FoundationV2ObjectiveConfig {
                ep_weight,
                sigreg_projections: REGISTERED_SIGREG_PROJECTIONS,
                sigreg_knots: REGISTERED_SIGREG_KNOTS,
                sigreg_seed,
                q_mse_threshold: cfg.q_mse_threshold,
                rollout_enabled: false,
                split_ce_weighting: cfg.split_ce_weighting,
                split_ce_changed_budget: cfg.split_ce_changed_budget,
                capture_mechanism_seams: false,
            },
            &event_slot_weights,
        )?;
        losses.rollout = rollout_loss.clone();
        losses.rollout_fragments = rollout_fragments;

        let prediction_route = if step == 1 {
            Some(route_norms(
                &retain_parameter_gradients(losses.pred_ce.backward()?, varmap)?,
                varmap,
            )?)
        } else {
            None
        };

        if !prediction_only && step.is_multiple_of(128) {
            let ep_grads = retain_parameter_gradients(losses.ep.backward()?, varmap)?;
            let pred_grads = retain_parameter_gradients(losses.pred_ce.backward()?, varmap)?;
            let ep_norm = gradient_l2_for_parameter_prefix(&ep_grads, varmap, "encoder.")?;
            let pred_norm = gradient_l2_for_parameter_prefix(&pred_grads, varmap, "encoder.")?;
            ep_weight = foundation_v2_ep_weight_update(ep_weight, ep_norm, pred_norm);
        }

        let main_total = if prediction_only {
            losses.pred_ce.clone()
        } else {
            losses
                .non_ep_total
                .add(&losses.ep.affine(ep_weight, 0.0)?)?
        };
        let logged_total = if prediction_only {
            main_total.clone()
        } else {
            main_total.add(&rollout_loss.affine(cfg.rollout_weight, 0.0)?)?
        };
        let raw_main_grads = main_total.backward()?;
        let mut accumulated = None;
        accumulate_parameter_gradients(&mut accumulated, raw_main_grads, varmap)?;
        if let Some(grads) = rollout_grads.take() {
            accumulate_parameter_gradients(&mut accumulated, grads, varmap)?;
        }
        let mut grads = accumulated.context("positive-control update has no gradients")?;
        let clip = clip_gradients_gpu_with_stats(&mut grads, varmap, MAX_GRAD_NORM)?;
        let learning_rate = foundation_v2_wsd_learning_rate(step, report.spec.total_schedule_steps);
        let values =
            foundation_v2_loss_values(&losses, &logged_total, clip.pre_clip_norm, clip.scale)?;
        let update = UpdateRecord {
            step,
            sigreg_seed,
            learning_rate,
            ep_weight,
            rollout_fragments,
            losses: values,
        };

        if step == 1 {
            let prediction_unclipped = prediction_route.context("missing update-1 route")?;
            let combined_clipped = route_norms(&grads, varmap)?;
            let route = RoutePremise {
                passed: prediction_unclipped.passed() && combined_clipped.passed(),
                prediction_unclipped,
                combined_clipped,
            };
            report.route_premise = Some(route.clone());
            write_progress(&args.output_root, report)?;
            ensure!(
                route.passed,
                "registered prediction-gradient route premise failed"
            );
            if let Some(parent) = &parent_reference {
                ensure!(
                    compare_route_norms(&route.prediction_unclipped, &parent.prediction_route),
                    "Q update-1 prediction route differs from P"
                );
                ensure!(
                    approximately(update.losses.pred_ce, parent.prediction_ce),
                    "Q update-1 prediction CE differs from P initialization"
                );
            } else {
                let changed_pixels = losses.changed_weights.changed_pixels;
                let unchanged_pixels = losses.changed_weights.unchanged_pixels;
                let changed_coefficient = losses.changed_weights.changed_weight;
                let coefficient_mass = changed_coefficient + 1.0;
                let binding = update_one_binding(
                    update.clone(),
                    changed_pixels,
                    unchanged_pixels,
                    changed_coefficient,
                    coefficient_mass,
                );
                report.update_one_binding = Some(binding.clone());
                write_progress(&args.output_root, report)?;
                ensure!(binding.passed, "update-1 production binding failed");
            }
        }

        optimizer.set_learning_rate(learning_rate)?;
        optimizer.step(&grads)?;
        ema.update(varmap)?;
        append_loss(&args.output_root, &update)?;
        report.updates_completed = step;
        drop(grads);
        drop(logged_total);
        drop(main_total);
        drop(losses);

        if report.spec.snapshot_steps.contains(&step) {
            report.snapshots.push(save_snapshot(
                &args.output_root,
                step,
                ep_weight,
                varmap,
                &ema,
                model,
                main,
                device,
            )?);
            write_progress(&args.output_root, report)?;
        }
        ensure!(
            started.elapsed() <= MAX_WALL_TIME,
            "positive-control wall-time cap exceeded"
        );
    }
    sync_cuda_device(device)?;
    report.timing.training_seconds = training_started.elapsed().as_secs_f64();
    let loss_log = args.output_root.join("loss_log.jsonl");
    report.loss_log_sha256 = Some(file_sha256_hex(&loss_log)?);
    report.verdict = Some(final_verdict(&report.spec, &report.snapshots)?);
    Ok(())
}

fn run_inner(
    args: &P2FixedBatchPositiveControlArgs,
    report: &mut PositiveControlReport,
    started: Instant,
) -> Result<()> {
    report.spec.validate(args.registered)?;
    ensure!(
        args.device.trim().starts_with("cuda"),
        "positive-control preflight and registered runs require CUDA"
    );
    if args.registered {
        registered_provenance_guard(&report.provenance)?;
    }
    if report.spec.arm == FULL_OBJECTIVE {
        ensure!(
            args.parent_full_report.is_none(),
            "P arm does not accept --parent-full-report"
        );
        ensure!(
            !args.registered || args.preflight_report.is_some(),
            "registered P requires --preflight-report"
        );
    } else {
        ensure!(args.registered, "Q must be registered");
        ensure!(
            args.parent_full_report.is_some(),
            "registered Q requires --parent-full-report"
        );
        ensure!(
            args.preflight_report.is_none(),
            "registered Q binds P directly and does not accept a preflight"
        );
    }

    let config_bytes = fs::read(&args.train_config)
        .with_context(|| format!("read {}", args.train_config.display()))?;
    report.train_config_sha256 = digest_bytes(&config_bytes);
    ensure!(
        report.train_config_sha256 == REGISTERED_TRAIN_CONFIG_SHA256,
        "train config hash is not registered"
    );
    fs::write(args.output_root.join("train_config.json"), &config_bytes)?;
    let cfg = load_train_config(&args.train_config)?;
    cfg.validate()?;
    ensure_registered_config(&cfg)?;
    ensure!(
        args.device == cfg.device,
        "device differs from frozen config"
    );
    report.checkpoint_sha256 = file_sha256_hex(&args.checkpoint)?;
    ensure!(
        report.checkpoint_sha256 == REGISTERED_CHECKPOINT_SHA256,
        "step-0 checkpoint hash is not registered"
    );

    let population_started = Instant::now();
    let stream_config = MixedStreamConfig {
        batch_size: cfg.physical_batch,
        seed: cfg.seed,
        schedule: adaptation_v6_stream_schedule,
        data_contract_v6: cfg.data_contract_v6,
        ..MixedStreamConfig::default()
    };
    let rollout_config = MixedStreamConfig {
        batch_size: cfg.physical_batch,
        seed: cfg.seed ^ ROLLOUT_SEED_DOMAIN,
        schedule: adaptation_v6_stream_schedule,
        data_contract_v6: cfg.data_contract_v6,
        ..MixedStreamConfig::default()
    };
    let main = compose_mixed_stream_batch(&stream_config, 0.0, 0, V5DataSplit::Train)?;
    let rollout = compose_rollout_fragment_batch(
        &rollout_config,
        REGISTERED_ROLLOUT_FRAGMENTS,
        0,
        V5DataSplit::Train,
    )?;
    let host = prepare_foundation_v2_batch_host(&main)?;
    report.population = Some(population_record(
        &args.output_root,
        &main,
        &rollout,
        &host,
    )?);
    report.timing.population_seconds = population_started.elapsed().as_secs_f64();

    let parent_reference = if args.registered && report.spec.arm == FULL_OBJECTIVE {
        report.preflight = Some(bind_preflight(
            args.preflight_report.as_deref().expect("checked preflight"),
            report,
        )?);
        None
    } else if report.spec.arm == PREDICTION_ONLY {
        let (parent, binding) = bind_failed_full_arm(
            args.parent_full_report
                .as_deref()
                .expect("checked P parent"),
            report,
        )?;
        let prediction_route = parent
            .route_premise
            .as_ref()
            .context("P report lacks route premise")?
            .prediction_unclipped
            .clone();
        let prediction_ce = parent
            .update_one_binding
            .as_ref()
            .context("P report lacks update-1 binding")?
            .observed
            .losses
            .pred_ce;
        report.parent_full_arm = Some(binding);
        Some(ParentArmReference {
            prediction_route,
            prediction_ce,
        })
    } else {
        None
    };

    let _pid_guard = TrainPidGuard::install(&args.output_root)?;
    let diagnostic_device = open_diagnostic_device(&args.device, &args.output_root)?;
    report.device_is_cuda = diagnostic_device.device.is_cuda();
    report.gpu_identity = diagnostic_device.gpu_identity.clone();
    let device = &diagnostic_device.device;
    let varmap = VarMap::new();
    let model = WorldModel::new(
        cfg.model_config(),
        VarBuilder::from_varmap(&varmap, DType::F32, device),
    )?;
    load_varmap_exact(&varmap, &args.checkpoint)?;
    ensure_operator_projection_zero(&varmap)?;
    let optimizer = CheckpointHybridOptimizer::new(
        &varmap,
        adam_params(&cfg),
        cfg.muon_momentum,
        cfg.muon_rms_scale,
    )?;
    let ema = ModelEma::with_default_decay(&varmap)?;
    run_training(
        args,
        report,
        &cfg,
        &main,
        &rollout,
        &host,
        device,
        &model,
        &varmap,
        optimizer,
        ema,
        parent_reference,
        started,
    )?;
    verify_no_input_drift(
        &args.checkpoint,
        &report.checkpoint_sha256,
        &report.provenance,
    )?;
    ensure!(
        digest_bytes(&fs::read(&args.train_config)?) == report.train_config_sha256,
        "train config changed during the run"
    );
    drop(diagnostic_device);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    report.identity_root = report_identity(report)?;
    Ok(())
}

pub fn run_p2_fixed_batch_positive_control(args: P2FixedBatchPositiveControlArgs) -> Result<()> {
    let started = Instant::now();
    let spec = PositiveControlSpec::from_args(&args);
    let run_class = if args.registered {
        RUN_CLASS_REGISTERED
    } else {
        RUN_CLASS_PREFLIGHT
    };
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        note: "V6 fixed-batch raw-weight positive control in progress".into(),
    };
    let command = open_run_root(&args.output_root, &lifecycle, COMMAND_TAG)?;
    let mut report = PositiveControlReport {
        schema: POSITIVE_CONTROL_SCHEMA.into(),
        evidence_class: EVIDENCE_CLASS.into(),
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
        checkpoint: args.checkpoint.clone(),
        checkpoint_sha256: String::new(),
        train_config: args.train_config.clone(),
        train_config_sha256: String::new(),
        spec,
        population: None,
        preflight: None,
        parent_full_arm: None,
        route_premise: None,
        update_one_binding: None,
        updates_completed: 0,
        snapshots: Vec::new(),
        loss_log_sha256: None,
        verdict: None,
        timing: PositiveControlTiming {
            population_seconds: 0.0,
            training_seconds: 0.0,
            wall_seconds: 0.0,
        },
        identity_root: String::new(),
        error: None,
    };
    let mut outcome = run_inner(&args, &mut report, started);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    if report.identity_root.is_empty() && report.population.is_some() {
        match report_identity(&report) {
            Ok(identity) => report.identity_root = identity,
            Err(error) => outcome = Err(error.context("compute positive-control identity")),
        }
    }
    report.lifecycle = match &outcome {
        Ok(()) => LifecycleRecord {
            state: LIFECYCLE_COMPLETE.into(),
            unix_seconds: unix_seconds(),
            evidence_class: EVIDENCE_CLASS.into(),
            run_class: run_class.into(),
            note: format!(
                "{}; implementation diagnostic only; A/C/D and public ARC remain blocked",
                report
                    .verdict
                    .as_ref()
                    .map_or("completed without verdict", |verdict| verdict
                        .outcome
                        .as_str())
            ),
        },
        Err(error) => {
            report.error = Some(format!("{error:#}"));
            LifecycleRecord {
                state: LIFECYCLE_FAILED.into(),
                unix_seconds: unix_seconds(),
                evidence_class: FAILED_EVIDENCE_CLASS.into(),
                run_class: run_class.into(),
                note: format!("{error:#}"),
            }
        }
    };
    let lifecycle = report.lifecycle.clone();
    seal_run_root(&args.output_root, COMMAND_TAG, &report, &lifecycle)?;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_rejects_unregistered_prediction_only_and_wrong_budgets() {
        let args = P2FixedBatchPositiveControlArgs {
            checkpoint: "checkpoint".into(),
            train_config: "config".into(),
            device: "cuda".into(),
            output_root: "out".into(),
            max_updates: PREFLIGHT_UPDATES,
            registered: false,
            prediction_only: true,
            preflight_report: None,
            parent_full_report: None,
        };
        assert!(PositiveControlSpec::from_args(&args)
            .validate(false)
            .is_err());
        let mut full = PositiveControlSpec::from_args(&P2FixedBatchPositiveControlArgs {
            prediction_only: false,
            ..args
        });
        assert!(full.validate(false).is_ok());
        full.max_updates = REGISTERED_UPDATES;
        assert!(full.validate(false).is_err());
        assert!(full.validate(true).is_ok());
    }

    #[test]
    fn sigreg_seed_progression_is_zero_based_and_wrapping() {
        assert_eq!(registered_sigreg_seed(5, 0), 5);
        assert_eq!(registered_sigreg_seed(5, 1023), 1028);
        assert_eq!(registered_sigreg_seed(u64::MAX, 1), 0);
    }

    #[test]
    fn decision_priority_is_exhaustive() {
        assert_eq!(
            classify_decision(true, true, true, true, true).0,
            OUTCOME_PASS
        );
        assert_eq!(
            classify_decision(true, true, false, false, true).0,
            OUTCOME_PARTIAL
        );
        assert_eq!(
            classify_decision(false, true, false, false, false).0,
            OUTCOME_WITHOUT_ACTION
        );
        assert_eq!(
            classify_decision(true, false, true, true, false).0,
            OUTCOME_FAIL
        );
    }

    #[test]
    fn action_blind_ceiling_cannot_reproduce_two_classes() -> Result<()> {
        let targets = [vec![1u8], vec![1], vec![2], vec![3]];
        let predictions = [vec![1u8], vec![1], vec![1], vec![1]];
        let mut multiplicities = BTreeMap::<Vec<u8>, usize>::new();
        let mut reproduced = BTreeSet::<Vec<u8>>::new();
        for (target, prediction) in targets.iter().zip(&predictions) {
            *multiplicities.entry(target.clone()).or_default() += 1;
            if target == prediction {
                reproduced.insert(target.clone());
            }
        }
        assert_eq!(multiplicities.values().copied().max(), Some(2));
        assert_eq!(reproduced.len(), 1);
        Ok(())
    }

    #[test]
    fn registered_population_census_is_stable() -> Result<()> {
        let main = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: REGISTERED_MAIN_ROWS,
                seed: 5,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            0.0,
            0,
            V5DataSplit::Train,
        )?;
        assert_eq!(main.samples().len(), REGISTERED_MAIN_ROWS);
        assert_eq!(
            main.samples()
                .iter()
                .filter(|row| board_changed(row))
                .count(),
            REGISTERED_CHANGED_ROWS
        );
        assert_eq!(main.factual_group_ranges().len(), 1);
        let range = main.factual_group_ranges()[0].clone();
        assert_eq!(range.len(), REGISTERED_FACTUAL_GROUP_ROWS);
        let pixels = gameplay_rows(true) * FRAME_SIDE;
        let distinct = main.samples()[range]
            .iter()
            .filter(|row| board_changed(row))
            .map(|row| row.transition.next.pixels[..pixels].to_vec())
            .collect::<BTreeSet<_>>()
            .len();
        assert_eq!(distinct, REGISTERED_DISTINCT_CHANGED_OUTCOMES);
        Ok(())
    }

    #[test]
    fn registered_targets_pass_exact_and_action_routes_but_action_blind_does_not() -> Result<()> {
        let main = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: REGISTERED_MAIN_ROWS,
                seed: 5,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            0.0,
            0,
            V5DataSplit::Train,
        )?;
        let target = control_predictions(main.samples(), "target");
        let exact = exact_metrics(main.samples(), &target)?;
        assert_eq!(exact.changed_rows, REGISTERED_CHANGED_ROWS);
        assert_eq!(exact.changed_exact, REGISTERED_CHANGED_ROWS);
        assert_eq!(exact.full_exact, REGISTERED_CHANGED_ROWS);
        let group = main.factual_group_ranges()[0].clone();
        let routed = action_class_metrics(main.samples(), group.clone(), &target)?;
        assert_eq!(
            routed.distinct_changed_classes,
            REGISTERED_DISTINCT_CHANGED_OUTCOMES
        );
        assert!(routed.action_routed);

        let first_changed_target = group
            .clone()
            .find(|row| board_changed(&main.samples()[*row]))
            .map(|row| target[row].clone())
            .context("registered group has no changed target")?;
        let action_blind = vec![first_changed_target; main.samples().len()];
        let blind = action_class_metrics(main.samples(), group, &action_blind)?;
        assert_eq!(blind.reproduced_distinct_changed_classes, 1);
        assert!(!blind.action_routed);
        assert!(blind.raw_full_exact_branches <= blind.maximum_changed_class_multiplicity);
        Ok(())
    }

    #[test]
    fn route_policy_requires_each_named_weight() {
        let valid = REQUIRED_POSITIVE_ROUTES
            .iter()
            .map(|name| ((*name).to_string(), 1.0))
            .chain(
                REQUIRED_ZERO_ROUTES
                    .iter()
                    .map(|name| ((*name).to_string(), 0.0)),
            )
            .collect::<BTreeMap<_, _>>();
        let route = RouteNorms {
            norms: valid.clone(),
            positive_pass: REQUIRED_POSITIVE_ROUTES
                .iter()
                .all(|name| valid[*name] > 0.0),
            zero_pass: REQUIRED_ZERO_ROUTES.iter().all(|name| valid[*name] == 0.0),
        };
        assert!(route.passed());
        let mut missing_spatial_weight = valid;
        missing_spatial_weight.insert("spatial_action_proj.weight".into(), 0.0);
        assert!(!REQUIRED_POSITIVE_ROUTES
            .iter()
            .all(|name| missing_spatial_weight[*name] > 0.0));
    }

    #[test]
    fn absent_gradient_is_zero_only_for_an_existing_topology() -> Result<()> {
        use candle_core::{Device, Tensor, Var};

        let varmap = VarMap::new();
        varmap.data().lock().unwrap().insert(
            "coord_proj.weight".into(),
            Var::from_tensor(&Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?)?,
        );
        let grads = GradStore::default();
        assert_eq!(
            gradient_l2_or_absent_zero(&grads, &varmap, "coord_proj.")?,
            0.0
        );
        assert!(gradient_l2_or_absent_zero(&grads, &varmap, "missing.").is_err());
        Ok(())
    }
}
