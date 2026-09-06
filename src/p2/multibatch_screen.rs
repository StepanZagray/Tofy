//! Host-only foundation for the V6 multi-batch function-learning screen.
//!
//! Registered in
//! `docs/research/2026-09-05-v6-multibatch-generalization-screen.md`. This
//! module freezes the contract, builds and censuses the populations, and
//! scores predictions; it exposes no command, opens no device, and cannot
//! launch training. Parent `positive_control.rs` behavior is unchanged.

use crate::p2::context_wiring::{
    file_sha256_hex, identity_frame_sha256, GpuIdentity, LifecycleRecord, RUN_CLASS_REGISTERED,
};
use crate::p2::data::{
    adaptation_v6_stream_schedule, compose_mixed_stream_batch, compose_rollout_fragment_batch,
    ArcFrame, MixedStreamBatch, MixedStreamConfig, TransitionSample, V5DataSplit, V5Sample,
    FRAME_SIDE,
};
use crate::p2::evidence::LaunchProvenance;
use crate::p2::positive_control::{
    action_class_metrics, bind_report, board_changed, bytes_hex, control_predictions, digest_bytes,
    registered_sigreg_seed, ActionClassMetrics, EvidenceBinding, RoutePremise, UpdateOneBinding,
    FULL_OBJECTIVE, MAX_WALL_TIME, OUTCOME_PASS, POSITIVE_CONTROL_SCHEMA,
    REGISTERED_CHECKPOINT_SHA256, REGISTERED_ROLLOUT_FRAGMENTS, REGISTERED_SIGREG_KNOTS,
    REGISTERED_SIGREG_PROJECTIONS, REGISTERED_TRAIN_CONFIG_SHA256,
    REGISTERED_UPDATES as PARENT_P_UPDATES, ROLLOUT_SEED_DOMAIN,
};
use crate::p2::semantic_eval::shuffled_action_control_population;
use crate::p2::train::training_content_batch_digest;
use anyhow::{ensure, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

pub const MULTIBATCH_SCREEN_SCHEMA: &str = "p2.v6_multibatch_generalization_screen.v1";
pub const POPULATION_CENSUS_SCHEMA: &str = "p2.v6_multibatch_population_census.v1";
pub const REGISTERED_UPDATES: usize = 2048;
pub const PREFLIGHT_UPDATES: usize = 8;
pub const TRAIN_BATCHES: usize = 8;
pub const TOTAL_SCHEDULE_STEPS: usize = 2048;
pub const EP_CONTROLLER_PERIOD: usize = 128;
pub const REGISTERED_SEED: u64 = 5;
pub const REGISTERED_BATCH_SIZE: usize = 128;
/// Every V6 row is playfield, including row 63.
pub const V6_PIXELS: usize = FRAME_SIDE * FRAME_SIDE;
pub const SNAPSHOT_STEPS: [usize; 7] = [0, 1, 256, 512, 1024, 1536, 2048];
pub const PREFLIGHT_SNAPSHOT_STEPS: [usize; 3] = [0, 1, 8];
pub const GATE_STEP: usize = 2048;
pub const EXTENSION_SIGNAL_FROM_STEP: usize = 1024;
pub const MAX_ADMISSION_ESTIMATE_SECONDS: f64 = 4_500.0;
pub const TRAIN_MAIN_INDICES: Range<u64> = 0..8;
pub const TRAIN_ROLLOUT_INDICES: Range<u64> = 0..8;
pub const HELDOUT_MAIN_INDICES: Range<u64> = 8..16;

pub const PARENT_SOURCE_REVISION: &str = "55d3e691cbb03e8aa6a7ccea6190944fb8c75bad";
pub const PARENT_P_REPORT_SHA256: &str =
    "a837bb95be922376fee9d7ec9b0f9a5ec03aa002ef55188e4b591bca51e64e53";
pub const PARENT_P_MANIFEST_SHA256: &str =
    "9867affe63584b5736c81d41833ab8259509a13fc056e88ae3036c2c65d7908f";
pub const PARENT_P_IDENTITY: &str =
    "sha256:521c077e4e7030697cabcfc1a4036d32b6e57e709d31625e05a413031fa89d7d";
pub const HOST_CENSUS_ARTIFACT_SHA256: &str =
    "060937144393121c30ca469f2175e2e297801e2a8728b3afa7f6bfcba98130a1";
pub const REGISTERED_STEP1_RAW_SHA256: &str =
    "7c62689c8938a9e351468a8b56e53bb9314cbe84d351a9d948f0cb457d57a3da";
pub const REGISTERED_STEP1_EMA_SHA256: &str =
    "834b63be6767a9d99538e37034d35dcb213361afec993a54e071ea004aa1d4c7";

pub const REGISTERED_CHANGED_ROWS_BY_BATCH: [usize; 16] = [
    95, 86, 87, 84, 91, 94, 91, 97, 75, 92, 99, 87, 85, 101, 98, 89,
];
pub const REGISTERED_CHANGED_FACTUAL_GROUP_ROWS_BY_BATCH: [usize; 16] =
    [8, 8, 4, 9, 9, 9, 8, 5, 8, 9, 9, 8, 5, 5, 9, 8];
pub const REGISTERED_DISTINCT_CHANGED_FACTUAL_CLASSES_BY_BATCH: [usize; 16] =
    [8, 6, 3, 8, 5, 5, 6, 4, 6, 8, 5, 6, 4, 4, 8, 7];
pub const REGISTERED_WITHIN_BATCH_OUTCOME_CHANGING_BY_BATCH: [usize; 16] = [
    27, 27, 21, 24, 24, 27, 23, 24, 23, 25, 25, 26, 26, 23, 27, 26,
];
pub const REGISTERED_HELDOUT_ACTION6_CLASSES_BY_GROUP: [usize; 8] = [4, 4, 4, 4, 1, 0, 4, 4];

pub const OUTCOME_GENERALIZES: &str = "GENERALIZES";
pub const OUTCOME_FRAME_GENERALIZES_ACTION_FAIL: &str = "FRAME_GENERALIZES_ACTION_FAIL";
pub const OUTCOME_FITS_NO_GENERALIZATION: &str = "FITS_NO_GENERALIZATION";
pub const OUTCOME_DOES_NOT_SCALE: &str = "DOES_NOT_SCALE";
pub const OUTCOME_PREFLIGHT: &str = "unregistered_preflight_complete";

/// Gate thresholds in whole percent of the preregistered denominators.
pub const TRAIN_FIT_PERCENT: usize = 50;
pub const GEN_CHANGED_PERCENT: usize = 20;
pub const GEN_FULL_PERCENT: usize = 10;
pub const CF_ACTION_PERCENT: usize = 50;
pub const EXTENSION_SIGNAL_PERCENT: usize = 10;
pub const GROUP_AR_MIN_GROUPS: usize = 2;
pub const COORD_MIN_GROUPS: usize = 2;
const BATCH_SELECTION_CONTRACT: &str = "zero_based_update % 8 for main and rollout";
const SIGREG_SEED_CONTRACT: &str = "seed.wrapping_add(zero_based_update)";

/// Union-level counts that must reproduce exactly before CUDA opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnionCensus {
    pub rows: usize,
    pub changed_rows: usize,
    pub factual_groups: usize,
    pub action6_rows: usize,
    pub changed_action6_rows: usize,
    pub changed_factual_action6_rows: usize,
    pub copy_changed_exact: usize,
    pub copy_full_exact: usize,
    pub background_changed_exact: usize,
    pub background_full_exact: usize,
    pub direct_target_changed_exact: usize,
    pub direct_target_full_exact: usize,
    pub shuffle_eligible_rows: usize,
    pub shuffle_changed_tuples: usize,
    pub shuffle_outcome_changing_tuples: usize,
}

pub const REGISTERED_TRAIN_UNION: UnionCensus = UnionCensus {
    rows: 1024,
    changed_rows: 725,
    factual_groups: 8,
    action6_rows: 145,
    changed_action6_rows: 112,
    changed_factual_action6_rows: 24,
    copy_changed_exact: 0,
    copy_full_exact: 0,
    background_changed_exact: 2,
    background_full_exact: 0,
    direct_target_changed_exact: 725,
    direct_target_full_exact: 725,
    shuffle_eligible_rows: 218,
    shuffle_changed_tuples: 217,
    shuffle_outcome_changing_tuples: 193,
};

pub const REGISTERED_HELDOUT_UNION: UnionCensus = UnionCensus {
    rows: 1024,
    changed_rows: 726,
    factual_groups: 8,
    action6_rows: 145,
    changed_action6_rows: 108,
    changed_factual_action6_rows: 25,
    copy_changed_exact: 0,
    copy_full_exact: 0,
    background_changed_exact: 3,
    background_full_exact: 0,
    direct_target_changed_exact: 726,
    direct_target_full_exact: 726,
    shuffle_eligible_rows: 219,
    shuffle_changed_tuples: 217,
    shuffle_outcome_changing_tuples: 200,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlapCensus {
    pub current_frames: usize,
    pub frame_action_keys: usize,
    pub sidecar_operator_input_tuples: usize,
    pub model_visible_input_tuples: usize,
    pub train_unique_current_frames: usize,
    pub heldout_unique_current_frames: usize,
    pub train_unique_frame_action_keys: usize,
    pub heldout_unique_frame_action_keys: usize,
    pub train_unique_sidecar_operator_input_tuples: usize,
    pub heldout_unique_sidecar_operator_input_tuples: usize,
    pub train_unique_model_visible_input_tuples: usize,
    pub heldout_unique_model_visible_input_tuples: usize,
}

pub const REGISTERED_OVERLAP: OverlapCensus = OverlapCensus {
    current_frames: 0,
    frame_action_keys: 0,
    sidecar_operator_input_tuples: 0,
    model_visible_input_tuples: 0,
    train_unique_current_frames: 701,
    heldout_unique_current_frames: 706,
    train_unique_frame_action_keys: 871,
    heldout_unique_frame_action_keys: 876,
    train_unique_sidecar_operator_input_tuples: 1006,
    heldout_unique_sidecar_operator_input_tuples: 1005,
    train_unique_model_visible_input_tuples: 955,
    heldout_unique_model_visible_input_tuples: 945,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutUnionCensus {
    pub rows: usize,
    pub unique_current_frames: usize,
    pub unique_frame_action_keys: usize,
    pub heldout_current_frame_overlap: usize,
    pub heldout_frame_action_overlap: usize,
}

pub const REGISTERED_ROLLOUT_UNION: RolloutUnionCensus = RolloutUnionCensus {
    rows: 256,
    unique_current_frames: 256,
    unique_frame_action_keys: 256,
    heldout_current_frame_overlap: 0,
    heldout_frame_action_overlap: 0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisagreementScore {
    pub counterfactual_correct: usize,
    pub factual_correct: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShuffleControls {
    pub disagreement_pixels: usize,
    pub copy_current: DisagreementScore,
    pub background: DisagreementScore,
    pub direct_factual_target: DisagreementScore,
    pub counterfactual_oracle: DisagreementScore,
}

pub const REGISTERED_HELDOUT_SHUFFLE_CONTROLS: ShuffleControls = ShuffleControls {
    disagreement_pixels: 1456,
    copy_current: DisagreementScore {
        counterfactual_correct: 643,
        factual_correct: 812,
    },
    background: DisagreementScore {
        counterfactual_correct: 571,
        factual_correct: 568,
    },
    direct_factual_target: DisagreementScore {
        counterfactual_correct: 0,
        factual_correct: 1456,
    },
    counterfactual_oracle: DisagreementScore {
        counterfactual_correct: 1456,
        factual_correct: 0,
    },
};

/// Command arguments. Not wired to any command in this slice.
#[derive(Debug, Clone, Args)]
pub struct P2MultibatchScreenArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,

    #[arg(long)]
    pub train_config: PathBuf,

    #[arg(long, default_value = "cuda")]
    pub device: String,

    #[arg(long)]
    pub output_root: PathBuf,

    /// Registered evidence requires 2048; an unregistered preflight requires 8.
    #[arg(long, default_value_t = PREFLIGHT_UPDATES)]
    pub max_updates: usize,

    #[arg(long, default_value_t = false)]
    pub registered: bool,

    /// Sealed registered parent P report with outcome
    /// `same_row_action_conditioned_fit`; required by every run.
    #[arg(long)]
    pub parent_p_report: PathBuf,

    /// Sealed eight-update report from this exact binary; required for registered G.
    #[arg(long)]
    pub preflight_report: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultibatchScreenSpec {
    pub arm: String,
    pub max_updates: usize,
    pub total_schedule_steps: usize,
    pub snapshot_steps: Vec<usize>,
    pub gate_step: usize,
    pub train_main_indices: Vec<u64>,
    pub train_rollout_indices: Vec<u64>,
    pub heldout_main_indices: Vec<u64>,
    pub batch_selection_contract: String,
    pub ep_controller_period: usize,
    pub rollout_fragments: usize,
    pub sigreg_projections: usize,
    pub sigreg_knots: usize,
    pub sigreg_seed_contract: String,
    pub fail_on_nonfinite: bool,
    pub max_wall_seconds: u64,
    pub max_admission_estimate_seconds: f64,
}

impl MultibatchScreenSpec {
    pub fn from_args(args: &P2MultibatchScreenArgs) -> Self {
        let snapshot_steps = if args.max_updates == PREFLIGHT_UPDATES {
            PREFLIGHT_SNAPSHOT_STEPS.to_vec()
        } else {
            SNAPSHOT_STEPS.to_vec()
        };
        Self {
            arm: FULL_OBJECTIVE.into(),
            max_updates: args.max_updates,
            total_schedule_steps: TOTAL_SCHEDULE_STEPS,
            snapshot_steps,
            gate_step: GATE_STEP,
            train_main_indices: TRAIN_MAIN_INDICES.collect(),
            train_rollout_indices: TRAIN_ROLLOUT_INDICES.collect(),
            heldout_main_indices: HELDOUT_MAIN_INDICES.collect(),
            batch_selection_contract: BATCH_SELECTION_CONTRACT.into(),
            ep_controller_period: EP_CONTROLLER_PERIOD,
            rollout_fragments: REGISTERED_ROLLOUT_FRAGMENTS,
            sigreg_projections: REGISTERED_SIGREG_PROJECTIONS,
            sigreg_knots: REGISTERED_SIGREG_KNOTS,
            sigreg_seed_contract: SIGREG_SEED_CONTRACT.into(),
            fail_on_nonfinite: true,
            max_wall_seconds: MAX_WALL_TIME.as_secs(),
            max_admission_estimate_seconds: MAX_ADMISSION_ESTIMATE_SECONDS,
        }
    }

    pub fn validate(&self, registered: bool) -> Result<()> {
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
            self.max_updates.is_multiple_of(TRAIN_BATCHES),
            "update budget must visit every train batch equally"
        );
        ensure!(
            self.arm == FULL_OBJECTIVE,
            "G is a single full-objective arm"
        );
        ensure!(
            self.total_schedule_steps == TOTAL_SCHEDULE_STEPS
                && self.gate_step == GATE_STEP
                && self.ep_controller_period == EP_CONTROLLER_PERIOD
                && self.rollout_fragments == REGISTERED_ROLLOUT_FRAGMENTS
                && self.batch_selection_contract == BATCH_SELECTION_CONTRACT,
            "schedule drifted from the registered contract"
        );
        ensure!(
            self.train_main_indices == TRAIN_MAIN_INDICES.collect::<Vec<_>>()
                && self.train_rollout_indices == TRAIN_ROLLOUT_INDICES.collect::<Vec<_>>()
                && self.heldout_main_indices == HELDOUT_MAIN_INDICES.collect::<Vec<_>>(),
            "batch indices drifted from the registered contract"
        );
        ensure!(
            self.sigreg_projections == REGISTERED_SIGREG_PROJECTIONS
                && self.sigreg_knots == REGISTERED_SIGREG_KNOTS,
            "SIGReg geometry drifted from the registered contract"
        );
        ensure!(
            self.sigreg_seed_contract == SIGREG_SEED_CONTRACT
                && self.fail_on_nonfinite
                && self.max_wall_seconds == MAX_WALL_TIME.as_secs()
                && self.max_admission_estimate_seconds == MAX_ADMISSION_ESTIMATE_SECONDS,
            "integrity or runtime contract drifted"
        );
        let expected_snapshots = if registered {
            SNAPSHOT_STEPS.as_slice()
        } else {
            PREFLIGHT_SNAPSHOT_STEPS.as_slice()
        };
        ensure!(
            self.snapshot_steps == expected_snapshots,
            "snapshot schedule drifted from the registered contract"
        );
        Ok(())
    }
}

/// Zero-based update `u` trains main and rollout index `u mod 8`.
pub fn train_batch_position(zero_based_update: usize) -> usize {
    zero_based_update % TRAIN_BATCHES
}

/// The EP controller measures on exact one-based step multiples of 128.
pub fn ep_controller_step(one_based_step: usize) -> bool {
    one_based_step > 0 && one_based_step.is_multiple_of(EP_CONTROLLER_PERIOD)
}

pub fn sigreg_seed(zero_based_update: usize) -> u64 {
    registered_sigreg_seed(REGISTERED_SEED, zero_based_update)
}

pub fn step_one_checkpoint_binding(raw_sha256: &str, ema_sha256: &str) -> BTreeMap<String, bool> {
    BTreeMap::from([
        ("raw".into(), raw_sha256 == REGISTERED_STEP1_RAW_SHA256),
        ("ema".into(), ema_sha256 == REGISTERED_STEP1_EMA_SHA256),
    ])
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEstimate {
    pub preflight_training_seconds: f64,
    pub preflight_snapshot_seconds: Vec<f64>,
    pub median_snapshot_seconds: f64,
    pub estimated_registered_seconds: f64,
    pub admitted: bool,
}

/// `(training / 8) * 2048 + median(snapshot) * 7` for seven union snapshots, admitted at or under 4,500 s.
pub fn runtime_estimate(
    training_seconds: f64,
    snapshot_seconds: &[f64],
) -> Result<RuntimeEstimate> {
    ensure!(
        snapshot_seconds.len() == PREFLIGHT_SNAPSHOT_STEPS.len(),
        "preflight must time exactly {} union snapshots",
        PREFLIGHT_SNAPSHOT_STEPS.len()
    );
    ensure!(
        training_seconds.is_finite()
            && training_seconds > 0.0
            && snapshot_seconds.iter().all(|s| s.is_finite() && *s >= 0.0),
        "preflight timings are not finite positive"
    );
    let mut sorted = snapshot_seconds.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median_snapshot_seconds = sorted[sorted.len() / 2];
    let estimated_registered_seconds = training_seconds / PREFLIGHT_UPDATES as f64
        * REGISTERED_UPDATES as f64
        + median_snapshot_seconds * SNAPSHOT_STEPS.len() as f64;
    Ok(RuntimeEstimate {
        preflight_training_seconds: training_seconds,
        preflight_snapshot_seconds: snapshot_seconds.to_vec(),
        median_snapshot_seconds,
        estimated_registered_seconds,
        admitted: estimated_registered_seconds <= MAX_ADMISSION_ESTIMATE_SECONDS,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchRole {
    TrainMain,
    TrainRollout,
    HeldoutMain,
}

impl BatchRole {
    fn directory(self) -> &'static str {
        match self {
            Self::TrainMain => "train_main",
            Self::TrainRollout => "train_rollout",
            Self::HeldoutMain => "heldout_main",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchRecord {
    pub role: BatchRole,
    pub index: u64,
    pub file: PathBuf,
    pub rows: usize,
    pub changed_rows: usize,
    pub factual_group_ranges: Vec<[usize; 2]>,
    pub changed_factual_group_rows: usize,
    pub distinct_changed_factual_classes: usize,
    pub within_batch_outcome_changing_tuples: Option<usize>,
    pub content_digest: String,
    pub population_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationCensus {
    pub schema: String,
    pub preregistered_host_census_sha256: String,
    pub v6_pixels: usize,
    pub batches: Vec<BatchRecord>,
    pub train_union: UnionCensus,
    pub heldout_union: UnionCensus,
    pub train_action6_classes_by_group: Vec<usize>,
    pub heldout_action6_classes_by_group: Vec<usize>,
    pub overlap: OverlapCensus,
    pub rollout_union: RolloutUnionCensus,
    pub train_shuffle_disagreement_pixels: usize,
    pub heldout_shuffle_controls: ShuffleControls,
    pub train_union_sha256: String,
    pub train_rollout_union_sha256: String,
    pub heldout_union_sha256: String,
}

/// Rows of several batches in index order with rebased factual group ranges.
#[derive(Debug, Clone)]
pub struct BatchUnion {
    pub samples: Vec<V5Sample>,
    pub factual_group_ranges: Vec<Range<usize>>,
}

impl BatchUnion {
    pub fn concat(batches: &[MixedStreamBatch]) -> Self {
        let mut samples = Vec::new();
        let mut factual_group_ranges = Vec::new();
        for batch in batches {
            let offset = samples.len();
            factual_group_ranges.extend(
                batch
                    .factual_group_ranges()
                    .iter()
                    .map(|range| range.start + offset..range.end + offset),
            );
            samples.extend_from_slice(batch.samples());
        }
        Self {
            samples,
            factual_group_ranges,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultibatchPopulation {
    pub train_main: Vec<MixedStreamBatch>,
    pub train_rollout: Vec<MixedStreamBatch>,
    pub heldout_main: Vec<MixedStreamBatch>,
}

impl MultibatchPopulation {
    pub fn compose(seed: u64, batch_size: usize, data_contract_v6: bool) -> Result<Self> {
        let main_config = MixedStreamConfig {
            batch_size,
            seed,
            schedule: adaptation_v6_stream_schedule,
            data_contract_v6,
            ..MixedStreamConfig::default()
        };
        let rollout_config = MixedStreamConfig {
            seed: seed ^ ROLLOUT_SEED_DOMAIN,
            ..main_config.clone()
        };
        let main = |index| compose_mixed_stream_batch(&main_config, 0.0, index, V5DataSplit::Train);
        Ok(Self {
            train_main: TRAIN_MAIN_INDICES.map(main).collect::<Result<_>>()?,
            train_rollout: TRAIN_ROLLOUT_INDICES
                .map(|index| {
                    compose_rollout_fragment_batch(
                        &rollout_config,
                        REGISTERED_ROLLOUT_FRAGMENTS,
                        index,
                        V5DataSplit::Train,
                    )
                })
                .collect::<Result<_>>()?,
            heldout_main: HELDOUT_MAIN_INDICES.map(main).collect::<Result<_>>()?,
        })
    }

    pub fn train_union(&self) -> BatchUnion {
        BatchUnion::concat(&self.train_main)
    }

    pub fn heldout_union(&self) -> BatchUnion {
        BatchUnion::concat(&self.heldout_main)
    }

    fn records(&self) -> Vec<(BatchRole, u64, &MixedStreamBatch)> {
        TRAIN_MAIN_INDICES
            .zip(&self.train_main)
            .map(|(index, batch)| (BatchRole::TrainMain, index, batch))
            .chain(
                TRAIN_ROLLOUT_INDICES
                    .zip(&self.train_rollout)
                    .map(|(index, batch)| (BatchRole::TrainRollout, index, batch)),
            )
            .chain(
                HELDOUT_MAIN_INDICES
                    .zip(&self.heldout_main)
                    .map(|(index, batch)| (BatchRole::HeldoutMain, index, batch)),
            )
            .collect()
    }

    /// Compute every frozen count without touching the filesystem.
    pub fn census(&self) -> Result<PopulationCensus> {
        ensure!(
            self.train_main.len() == TRAIN_BATCHES
                && self.train_rollout.len() == TRAIN_BATCHES
                && self.heldout_main.len() == TRAIN_BATCHES,
            "population must hold eight batches per role"
        );
        let batches = self
            .records()
            .into_iter()
            .map(|(role, index, batch)| batch_record(role, index, batch))
            .collect::<Result<Vec<_>>>()?;
        let train = self.train_union();
        let heldout = self.heldout_union();
        let train_shuffle = ShuffleSet::build(&train.samples)?;
        let heldout_shuffle = ShuffleSet::build(&heldout.samples)?;
        let rollout = BatchUnion::concat(&self.train_rollout);
        let union_sha = |role| ordered_union_sha256(&batches, role);
        Ok(PopulationCensus {
            schema: POPULATION_CENSUS_SCHEMA.into(),
            preregistered_host_census_sha256: HOST_CENSUS_ARTIFACT_SHA256.into(),
            v6_pixels: V6_PIXELS,
            train_union: union_census(&train, &train_shuffle)?,
            heldout_union: union_census(&heldout, &heldout_shuffle)?,
            train_action6_classes_by_group: action6_classes_by_group(&train),
            heldout_action6_classes_by_group: action6_classes_by_group(&heldout),
            overlap: overlap_census(&train.samples, &heldout.samples)?,
            rollout_union: rollout_union_census(&rollout.samples, &heldout.samples),
            train_shuffle_disagreement_pixels: train_shuffle.disagreement_pixels,
            heldout_shuffle_controls: heldout_shuffle.controls(&heldout.samples)?,
            train_union_sha256: union_sha(BatchRole::TrainMain)?,
            train_rollout_union_sha256: union_sha(BatchRole::TrainRollout)?,
            heldout_union_sha256: union_sha(BatchRole::HeldoutMain)?,
            batches,
        })
    }

    /// Serialize every batch and the verified census under `root/population`.
    /// Fails closed on any census drift before writing the census file.
    pub fn write(&self, root: &Path) -> Result<PopulationCensus> {
        let census = self.census()?;
        census.ensure_registered()?;
        for (record, (_, _, batch)) in census.batches.iter().zip(self.records()) {
            let path = root.join(&record.file);
            fs::create_dir_all(path.parent().context("batch file parent")?)?;
            fs::write(&path, batch_json_bytes(batch)?)?;
            ensure!(
                file_sha256_hex(&path)? == record.population_sha256,
                "serialized batch hash drifted for {}",
                record.file.display()
            );
        }
        let census_path = root.join("population").join("census.json");
        fs::write(&census_path, census_json_bytes(&census)?)?;
        Ok(census)
    }
}

impl PopulationCensus {
    /// Every frozen count from the preregistration, compared exactly.
    pub fn ensure_registered(&self) -> Result<()> {
        ensure!(
            self.schema == POPULATION_CENSUS_SCHEMA
                && self.preregistered_host_census_sha256 == HOST_CENSUS_ARTIFACT_SHA256
                && self.v6_pixels == V6_PIXELS,
            "census schema drift"
        );
        ensure!(
            self.batches.len() == 3 * TRAIN_BATCHES,
            "census holds {} batches, expected {}",
            self.batches.len(),
            3 * TRAIN_BATCHES
        );
        let expected_roles_and_indices = TRAIN_MAIN_INDICES
            .map(|index| (BatchRole::TrainMain, index))
            .chain(TRAIN_ROLLOUT_INDICES.map(|index| (BatchRole::TrainRollout, index)))
            .chain(HELDOUT_MAIN_INDICES.map(|index| (BatchRole::HeldoutMain, index)))
            .collect::<Vec<_>>();
        let observed_roles_and_indices = self
            .batches
            .iter()
            .map(|batch| (batch.role, batch.index))
            .collect::<Vec<_>>();
        ensure!(
            observed_roles_and_indices == expected_roles_and_indices,
            "batch roles or index ordering drifted from the frozen population"
        );
        for record in &self.batches {
            let expected_rows = if record.role == BatchRole::TrainRollout {
                2 * REGISTERED_ROLLOUT_FRAGMENTS
            } else {
                REGISTERED_BATCH_SIZE
            };
            ensure!(
                record.rows == expected_rows,
                "{:?} batch {} has {} rows, expected {expected_rows}",
                record.role,
                record.index,
                record.rows
            );
            if record.role == BatchRole::TrainRollout {
                continue;
            }
            let slot =
                usize::try_from(record.index).context("main batch index does not fit usize")?;
            ensure!(
                slot < REGISTERED_CHANGED_ROWS_BY_BATCH.len(),
                "main batch index {slot} is outside the frozen 0..16 census"
            );
            let expected = [
                (
                    "changed rows",
                    record.changed_rows,
                    REGISTERED_CHANGED_ROWS_BY_BATCH[slot],
                ),
                (
                    "changed factual group rows",
                    record.changed_factual_group_rows,
                    REGISTERED_CHANGED_FACTUAL_GROUP_ROWS_BY_BATCH[slot],
                ),
                (
                    "distinct changed factual classes",
                    record.distinct_changed_factual_classes,
                    REGISTERED_DISTINCT_CHANGED_FACTUAL_CLASSES_BY_BATCH[slot],
                ),
                (
                    "within-batch outcome-changing tuples",
                    record.within_batch_outcome_changing_tuples.unwrap_or(0),
                    REGISTERED_WITHIN_BATCH_OUTCOME_CHANGING_BY_BATCH[slot],
                ),
                ("factual groups", record.factual_group_ranges.len(), 1),
            ];
            for (name, observed, registered) in expected {
                ensure!(
                    observed == registered,
                    "main batch {slot} {name} drifted: observed {observed}, registered {registered}"
                );
            }
        }
        ensure!(
            self.train_union == REGISTERED_TRAIN_UNION,
            "train union census drifted:\n observed {:?}\n registered {:?}",
            self.train_union,
            REGISTERED_TRAIN_UNION
        );
        ensure!(
            self.heldout_union == REGISTERED_HELDOUT_UNION,
            "held-out union census drifted:\n observed {:?}\n registered {:?}",
            self.heldout_union,
            REGISTERED_HELDOUT_UNION
        );
        ensure!(
            self.heldout_action6_classes_by_group == REGISTERED_HELDOUT_ACTION6_CLASSES_BY_GROUP,
            "held-out ACTION6 class census drifted: {:?}",
            self.heldout_action6_classes_by_group
        );
        ensure!(
            self.overlap == REGISTERED_OVERLAP,
            "train/held-out overlap drifted:\n observed {:?}\n registered {:?}",
            self.overlap,
            REGISTERED_OVERLAP
        );
        ensure!(
            self.rollout_union == REGISTERED_ROLLOUT_UNION,
            "rollout union census drifted:\n observed {:?}\n registered {:?}",
            self.rollout_union,
            REGISTERED_ROLLOUT_UNION
        );
        ensure!(
            self.heldout_shuffle_controls == REGISTERED_HELDOUT_SHUFFLE_CONTROLS,
            "held-out shuffle controls drifted:\n observed {:?}\n registered {:?}",
            self.heldout_shuffle_controls,
            REGISTERED_HELDOUT_SHUFFLE_CONTROLS
        );
        Ok(())
    }
}

fn batch_json_bytes(batch: &MixedStreamBatch) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(batch).context("serialize batch")?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn census_json_bytes(census: &PopulationCensus) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(census).context("serialize census")?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn frame_pixels(frame: &ArcFrame) -> Result<&[u8]> {
    ensure!(
        frame.pixels.len() == V6_PIXELS,
        "frame holds {} pixels, expected {V6_PIXELS}",
        frame.pixels.len()
    );
    Ok(&frame.pixels[..])
}

fn is_action6(sample: &V5Sample) -> bool {
    sample.transition.action.id == 6
}

fn batch_record(role: BatchRole, index: u64, batch: &MixedStreamBatch) -> Result<BatchRecord> {
    let samples = batch.samples();
    let groups = batch.factual_group_ranges();
    let changed_in = |range: &Range<usize>| {
        samples[range.clone()]
            .iter()
            .filter(|row| board_changed(row))
    };
    let distinct_changed_factual_classes = groups
        .iter()
        .flat_map(changed_in)
        .map(|row| row.transition.next.pixels[..].to_vec())
        .collect::<BTreeSet<_>>()
        .len();
    let within_batch_outcome_changing_tuples = if role == BatchRole::TrainRollout {
        None
    } else {
        Some(ShuffleSet::build(samples)?.outcome_changing_tuples)
    };
    let digest = training_content_batch_digest(batch.transitions(), batch.content_masks())?;
    Ok(BatchRecord {
        role,
        index,
        file: PathBuf::from("population")
            .join(role.directory())
            .join(format!("batch-{index:02}.json")),
        rows: samples.len(),
        changed_rows: samples.iter().filter(|row| board_changed(row)).count(),
        factual_group_ranges: groups.iter().map(|r| [r.start, r.end]).collect(),
        changed_factual_group_rows: groups.iter().map(|r| changed_in(r).count()).sum(),
        distinct_changed_factual_classes,
        within_batch_outcome_changing_tuples,
        content_digest: bytes_hex(&digest),
        population_sha256: digest_bytes(&batch_json_bytes(batch)?),
    })
}

fn ordered_union_sha256(records: &[BatchRecord], role: BatchRole) -> Result<String> {
    let frames = records
        .iter()
        .filter(|record| record.role == role)
        .map(|record| {
            (
                record.file.to_str().context("batch file is not UTF-8"),
                record.population_sha256.as_bytes().to_vec(),
            )
        })
        .map(|(name, sha)| name.map(|name| (name, sha)))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        frames.len() == TRAIN_BATCHES,
        "ordered union needs eight batches"
    );
    identity_frame_sha256(&frames)
}

fn union_census(union: &BatchUnion, shuffle: &ShuffleSet) -> Result<UnionCensus> {
    let samples = &union.samples;
    let score = |kind| exact_metrics(samples, &control_predictions(samples, kind));
    let copy = score("copy")?;
    let background = score("background")?;
    let target = score("target")?;
    let changed_factual_action6_rows = union
        .factual_group_ranges
        .iter()
        .flat_map(|range| samples[range.clone()].iter())
        .filter(|row| is_action6(row) && board_changed(row))
        .count();
    Ok(UnionCensus {
        rows: samples.len(),
        changed_rows: target.changed_rows,
        factual_groups: union.factual_group_ranges.len(),
        action6_rows: samples.iter().filter(|row| is_action6(row)).count(),
        changed_action6_rows: samples
            .iter()
            .filter(|row| is_action6(row) && board_changed(row))
            .count(),
        changed_factual_action6_rows,
        copy_changed_exact: copy.changed_exact,
        copy_full_exact: copy.full_exact,
        background_changed_exact: background.changed_exact,
        background_full_exact: background.full_exact,
        direct_target_changed_exact: target.changed_exact,
        direct_target_full_exact: target.full_exact,
        shuffle_eligible_rows: shuffle.eligible_rows,
        shuffle_changed_tuples: shuffle.changed_tuples,
        shuffle_outcome_changing_tuples: shuffle.outcome_changing_tuples,
    })
}

fn action6_classes_by_group(union: &BatchUnion) -> Vec<usize> {
    union
        .factual_group_ranges
        .iter()
        .map(|range| {
            union.samples[range.clone()]
                .iter()
                .filter(|row| is_action6(row) && board_changed(row))
                .map(|row| row.transition.next.pixels[..].to_vec())
                .collect::<BTreeSet<_>>()
                .len()
        })
        .collect()
}

fn frame_key(sample: &V5Sample) -> Vec<u8> {
    sample.transition.current.pixels[..].to_vec()
}

fn action_bytes(action: &crate::p2::data::ArcAction) -> [u8; 3] {
    [
        action.id,
        action.x.unwrap_or(u8::MAX),
        action.y.unwrap_or(u8::MAX),
    ]
}

fn frame_action_key(sample: &V5Sample) -> Vec<u8> {
    let mut key = frame_key(sample);
    key.extend_from_slice(&action_bytes(&sample.transition.action));
    key
}

fn input_key_prefix(sample: &V5Sample) -> Vec<u8> {
    let transition = &sample.transition;
    let mut key = frame_action_key(sample);
    for value in transition.goal_features.values {
        key.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    key.extend_from_slice(&(transition.context.len() as u64).to_le_bytes());
    for context in &transition.context {
        key.extend_from_slice(&context.current.pixels[..]);
        key.extend_from_slice(&action_bytes(&context.action));
        key.extend_from_slice(&context.next.pixels[..]);
    }
    key
}

/// The preregistration census key, which includes the exact sidecar episode
/// operator used to replay counterfactual outcomes.
fn sidecar_operator_input_key(sample: &V5Sample) -> Result<Vec<u8>> {
    let mut key = input_key_prefix(sample);
    key.extend_from_slice(&serde_json::to_vec(&sample.provenance.operator)?);
    Ok(key)
}

/// The conditioning actually visible to the V6 model. Every V6 sidecar maps
/// to UNKNOWN (`None`), so this key is coarser than the replay-sidecar key.
fn model_visible_input_key(sample: &V5Sample) -> Result<Vec<u8>> {
    let mut key = input_key_prefix(sample);
    key.extend_from_slice(&serde_json::to_vec(
        &sample.provenance.conditioning_operator(),
    )?);
    Ok(key)
}

fn key_sets(samples: &[V5Sample]) -> Result<[BTreeSet<Vec<u8>>; 4]> {
    Ok([
        samples.iter().map(frame_key).collect(),
        samples.iter().map(frame_action_key).collect(),
        samples
            .iter()
            .map(sidecar_operator_input_key)
            .collect::<Result<_>>()?,
        samples
            .iter()
            .map(model_visible_input_key)
            .collect::<Result<_>>()?,
    ])
}

fn overlap_census(train: &[V5Sample], heldout: &[V5Sample]) -> Result<OverlapCensus> {
    let train = key_sets(train)?;
    let heldout = key_sets(heldout)?;
    let overlap = |slot: usize| train[slot].intersection(&heldout[slot]).count();
    Ok(OverlapCensus {
        current_frames: overlap(0),
        frame_action_keys: overlap(1),
        sidecar_operator_input_tuples: overlap(2),
        model_visible_input_tuples: overlap(3),
        train_unique_current_frames: train[0].len(),
        heldout_unique_current_frames: heldout[0].len(),
        train_unique_frame_action_keys: train[1].len(),
        heldout_unique_frame_action_keys: heldout[1].len(),
        train_unique_sidecar_operator_input_tuples: train[2].len(),
        heldout_unique_sidecar_operator_input_tuples: heldout[2].len(),
        train_unique_model_visible_input_tuples: train[3].len(),
        heldout_unique_model_visible_input_tuples: heldout[3].len(),
    })
}

fn rollout_union_census(rollout: &[V5Sample], heldout: &[V5Sample]) -> RolloutUnionCensus {
    let frames = rollout.iter().map(frame_key).collect::<BTreeSet<_>>();
    let keys = rollout
        .iter()
        .map(frame_action_key)
        .collect::<BTreeSet<_>>();
    let heldout_frames = heldout.iter().map(frame_key).collect::<BTreeSet<_>>();
    let heldout_keys = heldout
        .iter()
        .map(frame_action_key)
        .collect::<BTreeSet<_>>();
    RolloutUnionCensus {
        rows: rollout.len(),
        unique_current_frames: frames.len(),
        unique_frame_action_keys: keys.len(),
        heldout_current_frame_overlap: frames.intersection(&heldout_frames).count(),
        heldout_frame_action_overlap: keys.intersection(&heldout_keys).count(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactMetrics {
    pub rows: usize,
    pub changed_rows: usize,
    pub changed_exact: usize,
    pub full_exact: usize,
    pub all_row_exact: usize,
    pub unchanged_pixels: usize,
    pub false_edit_pixels: usize,
    pub false_edit_rows: usize,
    pub changed_exact_fraction: f64,
    pub full_exact_fraction: f64,
    pub all_row_exact_fraction: f64,
    pub false_edit_rate: f64,
}

/// Exact scorer over all 4,096 V6 pixels. Changed/full exact are counted over
/// board-changing rows; all-row exact and false edits over every row.
pub fn exact_metrics(samples: &[V5Sample], predictions: &[Vec<u8>]) -> Result<ExactMetrics> {
    ensure!(
        samples.len() == predictions.len(),
        "prediction row mismatch"
    );
    let mut metrics = ExactMetrics {
        rows: samples.len(),
        changed_rows: 0,
        changed_exact: 0,
        full_exact: 0,
        all_row_exact: 0,
        unchanged_pixels: 0,
        false_edit_pixels: 0,
        false_edit_rows: 0,
        changed_exact_fraction: 0.0,
        full_exact_fraction: 0.0,
        all_row_exact_fraction: 0.0,
        false_edit_rate: 0.0,
    };
    for (sample, prediction) in samples.iter().zip(predictions) {
        let current = frame_pixels(&sample.transition.current)?;
        let target = frame_pixels(&sample.transition.next)?;
        ensure!(
            prediction.len() == V6_PIXELS,
            "prediction holds {} pixels, expected {V6_PIXELS}",
            prediction.len()
        );
        let mut changed_ok = true;
        let mut false_edits = 0usize;
        for ((before, after), predicted) in current.iter().zip(target).zip(prediction) {
            if before == after {
                metrics.unchanged_pixels += 1;
                false_edits += usize::from(predicted != after);
            } else {
                changed_ok &= predicted == after;
            }
        }
        let exact = prediction.as_slice() == target;
        metrics.all_row_exact += usize::from(exact);
        metrics.false_edit_pixels += false_edits;
        metrics.false_edit_rows += usize::from(false_edits > 0);
        if current != target {
            metrics.changed_rows += 1;
            metrics.changed_exact += usize::from(changed_ok);
            metrics.full_exact += usize::from(exact);
        }
    }
    ensure!(metrics.changed_rows > 0, "exact scorer has no changed rows");
    metrics.changed_exact_fraction = metrics.changed_exact as f64 / metrics.changed_rows as f64;
    metrics.full_exact_fraction = metrics.full_exact as f64 / metrics.changed_rows as f64;
    metrics.all_row_exact_fraction = metrics.all_row_exact as f64 / metrics.rows as f64;
    metrics.false_edit_rate = metrics.false_edit_pixels as f64 / metrics.unchanged_pixels as f64;
    Ok(metrics)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupRoutingMetrics {
    pub groups: Vec<ActionClassMetrics>,
    pub passing_groups: usize,
}

/// A factual group passes AR when at least two distinct changed target
/// classes are raw full-board exact.
pub fn group_routing_metrics(
    union: &BatchUnion,
    predictions: &[Vec<u8>],
) -> Result<GroupRoutingMetrics> {
    let groups = union
        .factual_group_ranges
        .iter()
        .map(|range| action_class_metrics(&union.samples, range.clone(), predictions))
        .collect::<Result<Vec<_>>>()?;
    Ok(GroupRoutingMetrics {
        passing_groups: groups.iter().filter(|group| group.action_routed).count(),
        groups,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action6GroupMetrics {
    pub group_rows: usize,
    pub changed_action6_rows: usize,
    pub distinct_changed_action6_classes: usize,
    pub raw_full_exact_changed_action6_rows: usize,
    pub reproduced_distinct_action6_classes: usize,
    pub reproduced_class_sha256: Vec<String>,
    pub coordinate_routed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action6CoordinateMetrics {
    pub action6_rows: usize,
    pub changed_action6_rows: usize,
    pub changed_action6_full_exact: usize,
    pub changed_factual_action6_rows: usize,
    pub changed_factual_action6_full_exact: usize,
    pub groups: Vec<Action6GroupMetrics>,
    pub coordinate_groups_passing: usize,
}

/// Strengthened coordinate fit: a group passes only when at least two
/// distinct changed ACTION6 target classes are raw full-board exact.
pub fn action6_coordinate_metrics(
    union: &BatchUnion,
    predictions: &[Vec<u8>],
) -> Result<Action6CoordinateMetrics> {
    let samples = &union.samples;
    ensure!(
        samples.len() == predictions.len(),
        "prediction row mismatch"
    );
    let changed_action6 = |row: usize| is_action6(&samples[row]) && board_changed(&samples[row]);
    let exact =
        |row: usize| predictions[row].as_slice() == &samples[row].transition.next.pixels[..];
    let groups = union
        .factual_group_ranges
        .iter()
        .map(|range| {
            let rows = range
                .clone()
                .filter(|row| changed_action6(*row))
                .collect::<Vec<_>>();
            let classes = rows
                .iter()
                .map(|row| samples[*row].transition.next.pixels[..].to_vec())
                .collect::<BTreeSet<_>>();
            let reproduced = rows
                .iter()
                .filter(|row| exact(**row))
                .map(|row| samples[*row].transition.next.pixels[..].to_vec())
                .collect::<BTreeSet<_>>();
            Action6GroupMetrics {
                group_rows: range.len(),
                changed_action6_rows: rows.len(),
                distinct_changed_action6_classes: classes.len(),
                raw_full_exact_changed_action6_rows: rows.iter().filter(|row| exact(**row)).count(),
                reproduced_distinct_action6_classes: reproduced.len(),
                reproduced_class_sha256: reproduced.iter().map(|c| digest_bytes(c)).collect(),
                coordinate_routed: reproduced.len() >= 2,
            }
        })
        .collect::<Vec<_>>();
    let all_changed = (0..samples.len()).filter(|row| changed_action6(*row));
    let factual_changed = union
        .factual_group_ranges
        .iter()
        .flat_map(|range| range.clone())
        .filter(|row| changed_action6(*row));
    Ok(Action6CoordinateMetrics {
        action6_rows: samples.iter().filter(|row| is_action6(row)).count(),
        changed_action6_rows: all_changed.clone().count(),
        changed_action6_full_exact: all_changed.filter(|row| exact(*row)).count(),
        changed_factual_action6_rows: factual_changed.clone().count(),
        changed_factual_action6_full_exact: factual_changed.filter(|row| exact(*row)).count(),
        coordinate_groups_passing: groups.iter().filter(|g| g.coordinate_routed).count(),
        groups,
    })
}

/// One deterministic sidecar-aware global cyclic ACTION5/ACTION6 shuffle with
/// its replayed counterfactual targets and fixed disagreement mask.
#[derive(Debug, Clone)]
pub struct ShuffleSet {
    pub shuffled: Vec<TransitionSample>,
    pub counterfactual_next: Vec<Option<ArcFrame>>,
    pub eligible_rows: usize,
    pub changed_tuples: usize,
    pub outcome_changing_tuples: usize,
    /// Pixel indices where factual and counterfactual targets differ; empty
    /// unless the row is an outcome-changing tuple.
    pub disagreement: Vec<Vec<usize>>,
    pub disagreement_pixels: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionStratum {
    pub outcome_changing_tuples: usize,
    pub disagreement_pixels: usize,
    pub counterfactual_correct: usize,
    pub factual_correct: usize,
    pub counterfactual_target_accuracy: Option<f64>,
    pub factual_target_accuracy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualMetrics {
    pub rows: usize,
    pub eligible_rows: usize,
    pub changed_tuples: usize,
    pub outcome_changing_tuples: usize,
    pub disagreement_pixels: usize,
    pub counterfactual_correct: usize,
    pub factual_correct: usize,
    pub counterfactual_target_accuracy: Option<f64>,
    pub factual_target_accuracy: Option<f64>,
    /// Keyed by the resulting shuffled action id (5 or 6).
    pub by_shuffled_action: BTreeMap<u8, ActionStratum>,
}

fn accuracy(correct: usize, pixels: usize) -> Option<f64> {
    (pixels > 0).then_some(correct as f64 / pixels as f64)
}

impl ShuffleSet {
    pub fn build(samples: &[V5Sample]) -> Result<Self> {
        let transitions = samples
            .iter()
            .map(|sample| sample.transition.clone())
            .collect::<Vec<_>>();
        let provenance = samples
            .iter()
            .map(|sample| sample.provenance.clone())
            .collect::<Vec<_>>();
        let shuffled = shuffled_action_control_population(&transitions, Some(&provenance))?;
        let changed_tuples = shuffled.changed_tuples(&transitions);
        let mut disagreement = vec![Vec::new(); samples.len()];
        let mut outcome_changing_tuples = 0usize;
        for (row, (factual, shuffled_row)) in transitions.iter().zip(&shuffled.samples).enumerate()
        {
            if factual.action == shuffled_row.action {
                continue;
            }
            let counterfactual = shuffled.counterfactual_next[row]
                .as_ref()
                .context("changed tuple lacks a replayed counterfactual target")?;
            let factual_target = frame_pixels(&factual.next)?;
            let pixels = frame_pixels(counterfactual)?
                .iter()
                .zip(factual_target)
                .enumerate()
                .filter(|(_, (cf, f))| cf != f)
                .map(|(pixel, _)| pixel)
                .collect::<Vec<_>>();
            if pixels.is_empty() {
                continue;
            }
            outcome_changing_tuples += 1;
            disagreement[row] = pixels;
        }
        Ok(Self {
            disagreement_pixels: disagreement.iter().map(Vec::len).sum(),
            disagreement,
            outcome_changing_tuples,
            changed_tuples,
            eligible_rows: shuffled.eligible_rows,
            shuffled: shuffled.samples,
            counterfactual_next: shuffled.counterfactual_next,
        })
    }

    /// Score predictions made on `self.shuffled` under the fixed mask.
    pub fn score(
        &self,
        samples: &[V5Sample],
        predictions: &[Vec<u8>],
    ) -> Result<CounterfactualMetrics> {
        ensure!(
            samples.len() == self.shuffled.len() && predictions.len() == samples.len(),
            "shuffle scorer row mismatch"
        );
        let mut by_shuffled_action = BTreeMap::<u8, ActionStratum>::new();
        let mut counterfactual_correct = 0usize;
        let mut factual_correct = 0usize;
        for (row, pixels) in self.disagreement.iter().enumerate() {
            if pixels.is_empty() {
                continue;
            }
            let prediction = &predictions[row];
            ensure!(
                prediction.len() == V6_PIXELS,
                "prediction holds {} pixels, expected {V6_PIXELS}",
                prediction.len()
            );
            let factual = &samples[row].transition.next.pixels;
            let counterfactual = self.counterfactual_next[row]
                .as_ref()
                .context("outcome-changing row lacks a counterfactual")?;
            let stratum = by_shuffled_action
                .entry(self.shuffled[row].action.id)
                .or_insert(ActionStratum {
                    outcome_changing_tuples: 0,
                    disagreement_pixels: 0,
                    counterfactual_correct: 0,
                    factual_correct: 0,
                    counterfactual_target_accuracy: None,
                    factual_target_accuracy: None,
                });
            stratum.outcome_changing_tuples += 1;
            stratum.disagreement_pixels += pixels.len();
            for &pixel in pixels {
                let cf = usize::from(prediction[pixel] == counterfactual.pixels[pixel]);
                let f = usize::from(prediction[pixel] == factual[pixel]);
                stratum.counterfactual_correct += cf;
                stratum.factual_correct += f;
                counterfactual_correct += cf;
                factual_correct += f;
            }
        }
        for stratum in by_shuffled_action.values_mut() {
            stratum.counterfactual_target_accuracy =
                accuracy(stratum.counterfactual_correct, stratum.disagreement_pixels);
            stratum.factual_target_accuracy =
                accuracy(stratum.factual_correct, stratum.disagreement_pixels);
        }
        Ok(CounterfactualMetrics {
            rows: samples.len(),
            eligible_rows: self.eligible_rows,
            changed_tuples: self.changed_tuples,
            outcome_changing_tuples: self.outcome_changing_tuples,
            disagreement_pixels: self.disagreement_pixels,
            counterfactual_correct,
            factual_correct,
            counterfactual_target_accuracy: accuracy(
                counterfactual_correct,
                self.disagreement_pixels,
            ),
            factual_target_accuracy: accuracy(factual_correct, self.disagreement_pixels),
            by_shuffled_action,
        })
    }

    /// Copy, background, direct factual target, and counterfactual oracle
    /// under the identical disagreement mask.
    pub fn controls(&self, samples: &[V5Sample]) -> Result<ShuffleControls> {
        let score = |predictions: &[Vec<u8>]| -> Result<DisagreementScore> {
            let metrics = self.score(samples, predictions)?;
            Ok(DisagreementScore {
                counterfactual_correct: metrics.counterfactual_correct,
                factual_correct: metrics.factual_correct,
            })
        };
        let oracle = samples
            .iter()
            .zip(&self.counterfactual_next)
            .map(|(sample, counterfactual)| {
                counterfactual
                    .as_ref()
                    .map_or(
                        &sample.transition.next.pixels[..],
                        |frame| &frame.pixels[..],
                    )
                    .to_vec()
            })
            .collect::<Vec<_>>();
        Ok(ShuffleControls {
            disagreement_pixels: self.disagreement_pixels,
            copy_current: score(&control_predictions(samples, "copy"))?,
            background: score(&control_predictions(samples, "background"))?,
            direct_factual_target: score(&control_predictions(samples, "target"))?,
            counterfactual_oracle: score(&oracle)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionSnapshotMetrics {
    pub raw: ExactMetrics,
    pub copy_control: ExactMetrics,
    pub background_control: ExactMetrics,
    pub direct_target_control: ExactMetrics,
    pub group_routing: GroupRoutingMetrics,
    pub action6: Action6CoordinateMetrics,
    pub counterfactual: CounterfactualMetrics,
}

/// Score one union from factual predictions and predictions on the shuffled
/// conditioning. Pure: no model or device is touched here.
pub fn score_union(
    union: &BatchUnion,
    shuffle: &ShuffleSet,
    predictions: &[Vec<u8>],
    shuffled_predictions: &[Vec<u8>],
) -> Result<UnionSnapshotMetrics> {
    let samples = &union.samples;
    Ok(UnionSnapshotMetrics {
        raw: exact_metrics(samples, predictions)?,
        copy_control: exact_metrics(samples, &control_predictions(samples, "copy"))?,
        background_control: exact_metrics(samples, &control_predictions(samples, "background"))?,
        direct_target_control: exact_metrics(samples, &control_predictions(samples, "target"))?,
        group_routing: group_routing_metrics(union, predictions)?,
        action6: action6_coordinate_metrics(union, predictions)?,
        counterfactual: shuffle.score(samples, shuffled_predictions)?,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenSnapshot {
    pub step: usize,
    pub raw_checkpoint: PathBuf,
    pub raw_sha256: String,
    pub ema_checkpoint: PathBuf,
    pub ema_sha256: String,
    pub ep_weight: f64,
    pub seconds: f64,
    pub train: UnionSnapshotMetrics,
    pub heldout: UnionSnapshotMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateInputs {
    pub train_full_exact: usize,
    pub heldout_changed_exact: usize,
    pub heldout_full_exact: usize,
    pub heldout_background_changed_exact: usize,
    pub heldout_background_full_exact: usize,
    pub heldout_ar_groups: usize,
    pub heldout_disagreement_pixels: usize,
    pub heldout_counterfactual: DisagreementScore,
    pub heldout_coordinate_groups: usize,
}

impl GateInputs {
    pub fn from_snapshot(snapshot: &ScreenSnapshot) -> Self {
        let heldout = &snapshot.heldout;
        Self {
            train_full_exact: snapshot.train.raw.full_exact,
            heldout_changed_exact: heldout.raw.changed_exact,
            heldout_full_exact: heldout.raw.full_exact,
            heldout_background_changed_exact: heldout.background_control.changed_exact,
            heldout_background_full_exact: heldout.background_control.full_exact,
            heldout_ar_groups: heldout.group_routing.passing_groups,
            heldout_disagreement_pixels: heldout.counterfactual.disagreement_pixels,
            heldout_counterfactual: DisagreementScore {
                counterfactual_correct: heldout.counterfactual.counterfactual_correct,
                factual_correct: heldout.counterfactual.factual_correct,
            },
            heldout_coordinate_groups: heldout.action6.coordinate_groups_passing,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreenGates {
    pub train_fit: bool,
    pub gen_changed: bool,
    pub gen_full: bool,
    pub gen_fit: bool,
    pub group_ar: bool,
    pub cf_action: bool,
    pub coord: bool,
    pub gen_action: bool,
}

/// `numerator / denominator >= percent / 100` in exact integer arithmetic.
fn at_least_percent(numerator: usize, denominator: usize, percent: usize) -> bool {
    numerator * 100 >= percent * denominator
}

/// Frozen gates against the preregistered denominators and control counts.
pub fn evaluate_gates(inputs: &GateInputs) -> ScreenGates {
    let train_fit = at_least_percent(
        inputs.train_full_exact,
        REGISTERED_TRAIN_UNION.changed_rows,
        TRAIN_FIT_PERCENT,
    );
    let heldout_rows = REGISTERED_HELDOUT_UNION.changed_rows;
    let gen_changed = at_least_percent(
        inputs.heldout_changed_exact,
        heldout_rows,
        GEN_CHANGED_PERCENT,
    ) && inputs.heldout_changed_exact > inputs.heldout_background_changed_exact;
    let gen_full = at_least_percent(inputs.heldout_full_exact, heldout_rows, GEN_FULL_PERCENT)
        && inputs.heldout_full_exact > inputs.heldout_background_full_exact;
    let group_ar = inputs.heldout_ar_groups >= GROUP_AR_MIN_GROUPS;
    let controls = REGISTERED_HELDOUT_SHUFFLE_CONTROLS;
    let cf = inputs.heldout_counterfactual;
    let cf_action = inputs.heldout_disagreement_pixels == controls.disagreement_pixels
        && at_least_percent(
            cf.counterfactual_correct,
            controls.disagreement_pixels,
            CF_ACTION_PERCENT,
        )
        && cf.counterfactual_correct > cf.factual_correct
        && cf.counterfactual_correct > controls.copy_current.counterfactual_correct
        && cf.counterfactual_correct > controls.background.counterfactual_correct;
    let coord = inputs.heldout_coordinate_groups >= COORD_MIN_GROUPS;
    ScreenGates {
        train_fit,
        gen_changed,
        gen_full,
        gen_fit: gen_changed && gen_full,
        group_ar,
        cf_action,
        coord,
        gen_action: group_ar && cf_action && coord,
    }
}

/// Fixed-priority decision rule; exactly one class applies.
pub fn classify(gates: &ScreenGates) -> (&'static str, &'static str) {
    if gates.train_fit && gates.gen_fit && gates.gen_action {
        (
            OUTCOME_GENERALIZES,
            "preregister a matched confirmation/streaming contrast; no checkpoint or ARC promotion",
        )
    } else if gates.train_fit && gates.gen_fit {
        (
            OUTCOME_FRAME_GENERALIZES_ACTION_FAIL,
            "frozen-checkpoint action-FiLM/spatial-coordinate diagnosis; do not retrain first",
        )
    } else if gates.train_fit {
        (
            OUTCOME_FITS_NO_GENERALIZATION,
            "frozen representation/readout comparison of train versus held-out; do not change loss or extend budget first",
        )
    } else {
        (
            OUTCOME_DOES_NOT_SCALE,
            "compute EXTENSION_SIGNAL; preregister the 4096-update extension if true, else the matched prediction-only discriminator; launch neither automatically",
        )
    }
}

/// Train raw full-exact fraction improves by at least 0.10 absolute from
/// update 1,024 to update 2,048, over the preregistered 725-row denominator.
pub fn extension_signal(train_full_exact_1024: usize, train_full_exact_2048: usize) -> bool {
    at_least_percent(
        train_full_exact_2048.saturating_sub(train_full_exact_1024),
        REGISTERED_TRAIN_UNION.changed_rows,
        EXTENSION_SIGNAL_PERCENT,
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenVerdict {
    pub outcome: String,
    pub gate_step: usize,
    pub gates: Option<ScreenGates>,
    pub gate_inputs: Option<GateInputs>,
    pub extension_signal: Option<bool>,
    pub next_action: String,
}

pub fn final_verdict(
    spec: &MultibatchScreenSpec,
    snapshots: &[ScreenSnapshot],
) -> Result<ScreenVerdict> {
    if spec.max_updates == PREFLIGHT_UPDATES {
        return Ok(ScreenVerdict {
            outcome: OUTCOME_PREFLIGHT.into(),
            gate_step: spec.gate_step,
            gates: None,
            gate_inputs: None,
            extension_signal: None,
            next_action: "bind this sealed same-binary preflight to registered G".into(),
        });
    }
    let snapshot = |step: usize| {
        snapshots
            .iter()
            .find(|snapshot| snapshot.step == step)
            .with_context(|| format!("missing registered snapshot {step}"))
    };
    let gate = snapshot(spec.gate_step)?;
    let inputs = GateInputs::from_snapshot(gate);
    let gates = evaluate_gates(&inputs);
    let (outcome, next_action) = classify(&gates);
    let extension_signal = (outcome == OUTCOME_DOES_NOT_SCALE)
        .then(|| {
            snapshot(EXTENSION_SIGNAL_FROM_STEP).map(|earlier| {
                extension_signal(earlier.train.raw.full_exact, gate.train.raw.full_exact)
            })
        })
        .transpose()?;
    Ok(ScreenVerdict {
        outcome: outcome.into(),
        gate_step: spec.gate_step,
        gates: Some(gates),
        gate_inputs: Some(inputs),
        extension_signal,
        next_action: next_action.into(),
    })
}

/// Reverify parent P from its sealed root: registered, complete, outcome
/// `same_row_action_conditioned_fit`, and the frozen identities.
pub fn bind_parent_p(path: &Path) -> Result<EvidenceBinding> {
    let (parent, binding) = bind_report(path)?;
    ensure!(
        parent.schema == POSITIVE_CONTROL_SCHEMA
            && parent.registered
            && parent.run_class == RUN_CLASS_REGISTERED
            && parent.spec.arm == FULL_OBJECTIVE
            && parent.spec.max_updates == PARENT_P_UPDATES,
        "parent is not registered P"
    );
    ensure!(
        parent.verdict.as_ref().map(|v| v.outcome.as_str()) == Some(OUTCOME_PASS)
            && parent.route_premise.as_ref().is_some_and(|r| r.passed)
            && parent.update_one_binding.as_ref().is_some_and(|b| b.passed),
        "parent P outcome, route, or update-1 binding is not the registered pass"
    );
    ensure!(
        parent.checkpoint_sha256 == REGISTERED_CHECKPOINT_SHA256
            && parent.train_config_sha256 == REGISTERED_TRAIN_CONFIG_SHA256
            && parent.provenance.source_revision == PARENT_SOURCE_REVISION,
        "parent P checkpoint, config, or source drifted"
    );
    ensure!(
        binding.identity_root == PARENT_P_IDENTITY
            && binding.manifest_sha256 == PARENT_P_MANIFEST_SHA256
            && file_sha256_hex(&binding.report)? == PARENT_P_REPORT_SHA256,
        "parent P report, manifest, or identity hash is not the frozen parent"
    );
    Ok(binding)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenTiming {
    pub population_seconds: f64,
    pub training_seconds: f64,
    pub snapshot_seconds: Vec<f64>,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultibatchScreenReport {
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
    pub spec: MultibatchScreenSpec,
    pub population: Option<PopulationCensus>,
    pub population_census_sha256: Option<String>,
    pub parent_p: Option<EvidenceBinding>,
    pub preflight: Option<EvidenceBinding>,
    pub runtime_estimate: Option<RuntimeEstimate>,
    pub route_premise: Option<RoutePremise>,
    pub update_one_binding: Option<UpdateOneBinding>,
    pub step_one_checkpoint_binding: Option<BTreeMap<String, bool>>,
    pub updates_completed: usize,
    pub visits_per_train_batch: Vec<usize>,
    pub snapshots: Vec<ScreenSnapshot>,
    pub loss_log_sha256: Option<String>,
    pub verdict: Option<ScreenVerdict>,
    pub timing: ScreenTiming,
    pub identity_root: String,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::positive_control::exact_metrics as parent_exact_metrics;
    use crate::p2::train::foundation_v2_wsd_learning_rate;
    use std::sync::OnceLock;

    fn population() -> &'static MultibatchPopulation {
        static POPULATION: OnceLock<MultibatchPopulation> = OnceLock::new();
        POPULATION.get_or_init(|| {
            MultibatchPopulation::compose(REGISTERED_SEED, REGISTERED_BATCH_SIZE, true)
                .expect("compose frozen population")
        })
    }

    fn census() -> &'static PopulationCensus {
        static CENSUS: OnceLock<PopulationCensus> = OnceLock::new();
        CENSUS.get_or_init(|| population().census().expect("census"))
    }

    fn args(max_updates: usize) -> P2MultibatchScreenArgs {
        P2MultibatchScreenArgs {
            checkpoint: "checkpoint".into(),
            train_config: "config".into(),
            device: "cuda".into(),
            output_root: "out".into(),
            max_updates,
            registered: false,
            parent_p_report: "parent".into(),
            preflight_report: None,
        }
    }

    #[test]
    fn spec_requires_registered_budgets_and_frozen_indices() {
        let preflight = MultibatchScreenSpec::from_args(&args(PREFLIGHT_UPDATES));
        assert!(preflight.validate(false).is_ok());
        assert!(preflight.validate(true).is_err());
        assert_eq!(preflight.snapshot_steps, PREFLIGHT_SNAPSHOT_STEPS);
        let mut registered = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        assert!(registered.validate(true).is_ok());
        assert!(registered.validate(false).is_err());
        assert_eq!(registered.snapshot_steps, SNAPSHOT_STEPS);
        registered.heldout_main_indices = (7..15).collect();
        assert!(registered.validate(true).is_err());
        let mut drifted = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        drifted.max_wall_seconds -= 1;
        assert!(drifted.validate(true).is_err());
        let mut drifted = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        drifted.max_admission_estimate_seconds += 1.0;
        assert!(drifted.validate(true).is_err());
        let mut drifted = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        drifted.batch_selection_contract.push_str(" drift");
        assert!(drifted.validate(true).is_err());
        let mut drifted = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        drifted.sigreg_seed_contract.push_str(" drift");
        assert!(drifted.validate(true).is_err());
        let mut drifted = MultibatchScreenSpec::from_args(&args(REGISTERED_UPDATES));
        drifted.snapshot_steps.push(2047);
        assert!(drifted.validate(true).is_err());
    }

    #[test]
    fn batch_selection_sigreg_and_schedule_follow_the_contract() {
        let visits = (0..REGISTERED_UPDATES).fold([0usize; TRAIN_BATCHES], |mut visits, u| {
            visits[train_batch_position(u)] += 1;
            visits
        });
        assert_eq!(visits, [REGISTERED_UPDATES / TRAIN_BATCHES; TRAIN_BATCHES]);
        assert_eq!(train_batch_position(0), 0);
        assert_eq!(train_batch_position(7), 7);
        assert_eq!(train_batch_position(8), 0);
        let controller_steps = (1..=REGISTERED_UPDATES)
            .filter(|step| ep_controller_step(*step))
            .collect::<Vec<_>>();
        assert_eq!(controller_steps.len(), 16);
        assert!(controller_steps
            .iter()
            .all(|step| step % 128 == 0 && train_batch_position(step - 1) == 7));
        assert!(!ep_controller_step(0));
        let seeds = (0..REGISTERED_UPDATES).map(sigreg_seed).collect::<Vec<_>>();
        assert_eq!(seeds.first(), Some(&5));
        assert_eq!(seeds.last(), Some(&2052));
        assert!(seeds.windows(2).all(|w| w[1] == w[0] + 1));
        let lr = |step| foundation_v2_wsd_learning_rate(step, TOTAL_SCHEDULE_STEPS);
        assert!((lr(1) - 2e-6).abs() < 1e-12);
        assert_eq!(lr(500), 1e-3);
        assert_eq!(lr(1740), 1e-3);
        assert!(lr(1741) < 1e-3);
        assert!((lr(2048) - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn runtime_estimate_admits_only_under_seventy_five_minutes() -> Result<()> {
        let admitted = runtime_estimate(8.0, &[30.0, 10.0, 20.0])?;
        assert_eq!(admitted.median_snapshot_seconds, 20.0);
        assert_eq!(admitted.estimated_registered_seconds, 2048.0 + 140.0);
        assert!(admitted.admitted);
        let rejected = runtime_estimate(17.0, &[30.0, 30.0, 30.0])?;
        assert!(rejected.estimated_registered_seconds > MAX_ADMISSION_ESTIMATE_SECONDS);
        assert!(!rejected.admitted);
        assert!(runtime_estimate(8.0, &[1.0]).is_err());
        Ok(())
    }

    #[test]
    fn gate_thresholds_sit_exactly_on_the_preregistered_rows() {
        let base = GateInputs {
            train_full_exact: 363,
            heldout_changed_exact: 146,
            heldout_full_exact: 73,
            heldout_background_changed_exact: 3,
            heldout_background_full_exact: 0,
            heldout_ar_groups: 2,
            heldout_disagreement_pixels: 1456,
            heldout_counterfactual: DisagreementScore {
                counterfactual_correct: 728,
                factual_correct: 727,
            },
            heldout_coordinate_groups: 2,
        };
        let gates = evaluate_gates(&base);
        assert!(gates.train_fit && gates.gen_fit && gates.gen_action);
        assert_eq!(classify(&gates).0, OUTCOME_GENERALIZES);
        let below = |edit: fn(&mut GateInputs)| {
            let mut inputs = base;
            edit(&mut inputs);
            evaluate_gates(&inputs)
        };
        assert!(!below(|i| i.train_full_exact = 362).train_fit);
        assert!(!below(|i| i.heldout_changed_exact = 145).gen_changed);
        assert!(!below(|i| i.heldout_full_exact = 72).gen_full);
        assert!(!below(|i| i.heldout_background_full_exact = 73).gen_full);
        assert!(!below(|i| i.heldout_ar_groups = 1).group_ar);
        assert!(!below(|i| i.heldout_coordinate_groups = 1).coord);
        assert!(!below(|i| i.heldout_counterfactual.counterfactual_correct = 727).cf_action);
        assert!(!below(|i| i.heldout_counterfactual.factual_correct = 728).cf_action);
        assert!(!below(|i| i.heldout_disagreement_pixels = 1455).cf_action);
        let weak_but_above_half = below(|i| {
            i.heldout_counterfactual = DisagreementScore {
                counterfactual_correct: 643,
                factual_correct: 0,
            }
        });
        assert!(!weak_but_above_half.cf_action);
        assert!(extension_signal(100, 173));
        assert!(!extension_signal(100, 172));
        assert!(!extension_signal(200, 100));
    }

    #[test]
    fn decision_priority_is_exhaustive() {
        let gates = |train_fit, gen_fit, gen_action| ScreenGates {
            train_fit,
            gen_changed: gen_fit,
            gen_full: gen_fit,
            gen_fit,
            group_ar: gen_action,
            cf_action: gen_action,
            coord: gen_action,
            gen_action,
        };
        assert_eq!(classify(&gates(true, true, true)).0, OUTCOME_GENERALIZES);
        assert_eq!(
            classify(&gates(true, true, false)).0,
            OUTCOME_FRAME_GENERALIZES_ACTION_FAIL
        );
        assert_eq!(
            classify(&gates(true, false, true)).0,
            OUTCOME_FITS_NO_GENERALIZATION
        );
        assert_eq!(
            classify(&gates(true, false, false)).0,
            OUTCOME_FITS_NO_GENERALIZATION
        );
        for gen_fit in [false, true] {
            for gen_action in [false, true] {
                assert_eq!(
                    classify(&gates(false, gen_fit, gen_action)).0,
                    OUTCOME_DOES_NOT_SCALE
                );
            }
        }
        assert_eq!(
            step_one_checkpoint_binding(REGISTERED_STEP1_RAW_SHA256, REGISTERED_STEP1_EMA_SHA256)
                .values()
                .filter(|ok| **ok)
                .count(),
            2
        );
    }

    #[test]
    fn frozen_population_census_reproduces_every_registered_count() -> Result<()> {
        let census = census();
        census.ensure_registered()?;
        let changed = census
            .batches
            .iter()
            .filter(|b| b.role != BatchRole::TrainRollout)
            .map(|b| b.changed_rows)
            .collect::<Vec<_>>();
        assert_eq!(changed, REGISTERED_CHANGED_ROWS_BY_BATCH);
        assert_eq!(census.batches.len(), 24);
        assert!(census
            .batches
            .iter()
            .all(|b| b.population_sha256.len() == 64 && b.content_digest.len() == 64));
        assert_ne!(census.train_union_sha256, census.heldout_union_sha256);
        assert_eq!(census.overlap.sidecar_operator_input_tuples, 0);
        assert_eq!(census.overlap.model_visible_input_tuples, 0);
        assert_eq!(
            (
                census.overlap.train_unique_sidecar_operator_input_tuples,
                census.overlap.heldout_unique_sidecar_operator_input_tuples,
                census.overlap.train_unique_model_visible_input_tuples,
                census.overlap.heldout_unique_model_visible_input_tuples,
            ),
            (1006, 1005, 955, 945)
        );
        let mut drifted = census.clone();
        drifted.heldout_union.shuffle_outcome_changing_tuples -= 1;
        assert!(drifted.ensure_registered().is_err());
        let mut invalid_index = census.clone();
        invalid_index
            .batches
            .iter_mut()
            .find(|batch| batch.role == BatchRole::HeldoutMain)
            .expect("held-out batch")
            .index = 16;
        assert!(invalid_index.ensure_registered().is_err());
        let round_trip: PopulationCensus = serde_json::from_slice(&census_json_bytes(census)?)?;
        assert_eq!(&round_trip, census);
        Ok(())
    }

    #[test]
    fn row_63_is_scored_and_scorer_matches_parent_on_changed_and_full_exact() -> Result<()> {
        let union = population().heldout_union();
        let samples = &union.samples;
        let target = control_predictions(samples, "target");
        let ours = exact_metrics(samples, &target)?;
        let parent = parent_exact_metrics(samples, &target)?;
        assert_eq!(
            (ours.changed_rows, ours.changed_exact, ours.full_exact),
            (parent.changed_rows, parent.changed_exact, parent.full_exact)
        );
        assert_eq!(ours.all_row_exact, samples.len());
        assert_eq!(ours.false_edit_pixels, 0);
        let row_63 = (FRAME_SIDE - 1) * FRAME_SIDE..V6_PIXELS;
        let rows_changing_row_63 = samples
            .iter()
            .filter(|s| {
                s.transition.current.pixels[row_63.clone()]
                    != s.transition.next.pixels[row_63.clone()]
            })
            .count();
        assert!(
            rows_changing_row_63 > 0,
            "V6 playfield must exercise row 63"
        );
        let legacy_scored = samples
            .iter()
            .map(|s| {
                let mut prediction = s.transition.next.pixels[..].to_vec();
                prediction[row_63.clone()]
                    .copy_from_slice(&s.transition.current.pixels[row_63.clone()]);
                prediction
            })
            .collect::<Vec<_>>();
        let legacy = exact_metrics(samples, &legacy_scored)?;
        let parent_legacy = parent_exact_metrics(samples, &legacy_scored)?;
        assert_eq!(
            (legacy.changed_rows, legacy.changed_exact, legacy.full_exact),
            (
                parent_legacy.changed_rows,
                parent_legacy.changed_exact,
                parent_legacy.full_exact,
            )
        );
        assert_eq!(legacy.full_exact, ours.full_exact - rows_changing_row_63);
        assert_eq!(
            legacy.changed_exact,
            ours.changed_exact - rows_changing_row_63
        );
        assert_eq!(legacy.false_edit_pixels, 0);
        let copy = exact_metrics(samples, &control_predictions(samples, "copy"))?;
        assert_eq!(
            (copy.changed_exact, copy.full_exact, copy.false_edit_pixels),
            (0, 0, 0)
        );
        assert_eq!(copy.all_row_exact, samples.len() - copy.changed_rows);
        let background = exact_metrics(samples, &control_predictions(samples, "background"))?;
        assert!(background.false_edit_rate > 0.0);
        Ok(())
    }

    #[test]
    fn heldout_shuffle_controls_and_strata_are_sensitive() -> Result<()> {
        let union = population().heldout_union();
        let shuffle = ShuffleSet::build(&union.samples)?;
        assert_eq!(
            shuffle.controls(&union.samples)?,
            REGISTERED_HELDOUT_SHUFFLE_CONTROLS
        );
        assert_eq!(shuffle.outcome_changing_tuples, 200);
        let oracle = union
            .samples
            .iter()
            .zip(&shuffle.counterfactual_next)
            .map(|(s, cf)| cf.as_ref().unwrap_or(&s.transition.next).pixels[..].to_vec())
            .collect::<Vec<_>>();
        let scored = shuffle.score(&union.samples, &oracle)?;
        assert_eq!(scored.counterfactual_target_accuracy, Some(1.0));
        assert_eq!(scored.factual_target_accuracy, Some(0.0));
        assert_eq!(
            scored
                .by_shuffled_action
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![5, 6]
        );
        let strata_tuples: usize = scored
            .by_shuffled_action
            .values()
            .map(|s| s.outcome_changing_tuples)
            .sum();
        let strata_pixels: usize = scored
            .by_shuffled_action
            .values()
            .map(|s| s.disagreement_pixels)
            .sum();
        assert_eq!((strata_tuples, strata_pixels), (200, 1456));
        let gates = evaluate_gates(&GateInputs {
            train_full_exact: 725,
            heldout_changed_exact: 726,
            heldout_full_exact: 726,
            heldout_background_changed_exact: 3,
            heldout_background_full_exact: 0,
            heldout_ar_groups: 8,
            heldout_disagreement_pixels: scored.disagreement_pixels,
            heldout_counterfactual: REGISTERED_HELDOUT_SHUFFLE_CONTROLS.direct_factual_target,
            heldout_coordinate_groups: 6,
        });
        assert!(
            !gates.cf_action,
            "the factual target must not pass CF_ACTION"
        );
        assert_eq!(classify(&gates).0, OUTCOME_FRAME_GENERALIZES_ACTION_FAIL);
        Ok(())
    }

    #[test]
    fn coordinate_blind_predictor_cannot_pass_the_strengthened_gate() -> Result<()> {
        let union = population().heldout_union();
        let samples = &union.samples;
        let target = control_predictions(samples, "target");
        let routed = action6_coordinate_metrics(&union, &target)?;
        assert_eq!(
            routed
                .groups
                .iter()
                .map(|g| g.distinct_changed_action6_classes)
                .collect::<Vec<_>>(),
            REGISTERED_HELDOUT_ACTION6_CLASSES_BY_GROUP
        );
        assert_eq!(routed.coordinate_groups_passing, 6);
        assert_eq!(routed.changed_action6_full_exact, 108);
        assert_eq!(routed.changed_factual_action6_full_exact, 25);
        assert_eq!(group_routing_metrics(&union, &target)?.passing_groups, 8);
        let mut blind = target.clone();
        for range in &union.factual_group_ranges {
            let Some(first) = range
                .clone()
                .find(|row| is_action6(&samples[*row]) && board_changed(&samples[*row]))
            else {
                continue;
            };
            for row in range.clone().filter(|row| is_action6(&samples[*row])) {
                blind[row] = target[first].clone();
            }
        }
        let blind_metrics = action6_coordinate_metrics(&union, &blind)?;
        assert_eq!(blind_metrics.coordinate_groups_passing, 0);
        assert!(blind_metrics
            .groups
            .iter()
            .all(|g| g.reproduced_distinct_action6_classes <= 1));
        assert!(blind_metrics.changed_factual_action6_full_exact >= 6);
        let scored = score_union(&union, &ShuffleSet::build(samples)?, &blind, &target)?;
        assert!(
            !evaluate_gates(&GateInputs {
                train_full_exact: 725,
                ..GateInputs::from_snapshot(&ScreenSnapshot {
                    step: GATE_STEP,
                    raw_checkpoint: PathBuf::new(),
                    raw_sha256: String::new(),
                    ema_checkpoint: PathBuf::new(),
                    ema_sha256: String::new(),
                    ep_weight: 0.0,
                    seconds: 0.0,
                    train: scored.clone(),
                    heldout: scored,
                })
            })
            .coord
        );
        Ok(())
    }

    #[test]
    fn write_serializes_batches_and_verified_census() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-multibatch-screen-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir_all(&root)?;
        let written = population().write(&root)?;
        assert_eq!(&written, census());
        for record in &written.batches {
            assert_eq!(
                file_sha256_hex(&root.join(&record.file))?,
                record.population_sha256
            );
        }
        let stored: PopulationCensus =
            serde_json::from_slice(&fs::read(root.join("population/census.json"))?)?;
        assert_eq!(stored, written);
        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
