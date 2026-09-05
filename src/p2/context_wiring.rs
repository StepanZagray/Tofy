//! `p2-context-wiring` — E2W two-row context-wiring overfit diagnostic
//! (`docs/research/2026-09-03-v6-local-falsifiers-prereg.md`, "Post-E2
//! diagnostic E2W").
//!
//! Starting from the sealed E2 step-4096 EMA, two otherwise-identical queries
//! (one selected synthetic twin pair, one chronological position) are trained
//! with direct exact-decoder Unimix cross-entropy on their target-disagreement
//! pixels, once with their own K=16 histories (`correct` arm) and once with the
//! histories exchanged (`swapped` arm). Every model parameter is trained by a
//! fresh AdamW under the production 2x2 F32 recurrence. Frozen evaluations at
//! the fixed checkpoint family compare own, paired and K0 context per query
//! direction. The result is `implementation_smoke` evidence only: it cannot
//! promote a model, unblock E3, justify 3x3, or say anything about ARC-AGI-3.
//!
//! Data boundary: only the registered synthetic twin population is generated.
//! Nothing in this module reads public ARC-AGI-3 data.

use crate::p2::bf16_falsifier::write_json_report;
use crate::p2::data::{
    gameplay_rows, AugmentedTwinPair, ContextTransition, LearningHistoryConfig, TransitionSample,
    FRAME_SIDE,
};
use crate::p2::eval::{
    learning_history_population_fingerprint, load_model, twin_continuous_decode_rows,
    twin_memorization_census, twin_memorization_population, twin_memorization_scoring_row,
    twin_window_has_evidence, update_canonical_transition_row, validate_twin_memorization_census,
    TwinContinuousDecodes, TwinMemorizationCensus, TwinMemorizationSpec,
    TWIN_MEMORIZATION_CONTEXT_LEN, TWIN_MEMORIZATION_DEFAULT_PAIRS,
    TWIN_MEMORIZATION_POPULATION_SEED,
};
use crate::p2::evidence::{launch_provenance, LaunchProvenance};
use crate::p2::experiment::TrainingRecipe;
use crate::p2::model::{
    RecursionDepth, RecursionOpts, WorldModel, CONTEXT_PARAMETER_PREFIX, PALETTE_SIZE, PATCH_SIZE,
};
use crate::p2::optimizer::{try_clip_gradients_gpu_with_stats, GradientClipStats};
use crate::p2::train::{
    batch_from_samples, foundation_v2_unimix_ce, resolve_device, sync_cuda_device, BatchTensors,
    TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarMap};
use clap::Args;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const CONTEXT_WIRING_SCHEMA: &str = "p2.context_wiring_diagnostic.v1";
/// The complete, fixed evaluation family (updates after which both arms are
/// scored). `0` is the pre-training evaluation.
pub const CONTEXT_WIRING_CHECKPOINT_FAMILY: [usize; 7] = [0, 8, 16, 32, 64, 128, 256];
/// Hard cap on optimizer updates per arm.
pub const CONTEXT_WIRING_MAX_UPDATES: usize = 256;
pub const CONTEXT_WIRING_LEARNING_RATE: f64 = 1e-3;
pub const CONTEXT_WIRING_BETA1: f64 = 0.9;
pub const CONTEXT_WIRING_BETA2: f64 = 0.999;
pub const CONTEXT_WIRING_EPSILON: f64 = 1e-8;
pub const CONTEXT_WIRING_WEIGHT_DECAY: f64 = 0.0;
pub const CONTEXT_WIRING_GRADIENT_CLIP: f64 = 1.0;
pub const CONTEXT_WIRING_PHYSICAL_BATCH: usize = 2;
pub const CONTEXT_WIRING_GRAD_ACCUM: usize = 1;
/// `D_correct > 1e-4`, `D_swapped < -1e-4`.
pub const CONTEXT_WIRING_D_THRESHOLD: f64 = 1e-4;
/// `D_correct - D_swapped > 2e-4`.
pub const CONTEXT_WIRING_INTERACTION_THRESHOLD: f64 = 2e-4;
/// Pooled own-versus-paired raw-softmax probability L1 `> 1e-6` (or at least
/// one raw argmax disagreement) in each arm.
pub const CONTEXT_WIRING_PROBABILITY_L1_THRESHOLD: f64 = 1e-6;

/// Registered fixed inputs (`--registered` fails closed on any drift).
pub const REGISTERED_CHECKPOINT_SHA256: &str =
    "c53bf0c42dc6c8f7945ff4d17bd6bd63a6db23e8b6e377b6b0b92903e66d694a";
pub const REGISTERED_TRAIN_CONFIG_SHA256: &str =
    "f479cc4eb1dd6d687fcbdb3ef7bdbe71d29a219ddd44e9aeee3334ad6507160f";
pub const REGISTERED_POPULATION_FINGERPRINT: &str =
    "sha256:484e8615e41102895997ddb9bec19665604fb7f62d21db9cc5ecea1470e58f42";
pub const REGISTERED_PAIRS: usize = TWIN_MEMORIZATION_DEFAULT_PAIRS;
pub const REGISTERED_CANDLE_GRAPH_REVISION: &str = "8e012f25e38f0c597c14268f0c705e504a5b5c28";
pub const REGISTERED_PARENT_RUN_ID: &str = "v6-e2-2x2-registered-20260904T2217-CDT";
pub const REGISTERED_PARENT_MANIFEST_SHA256: &str =
    "59ce9db70fd71bc0395a89eefca378c0849a83d4dfa23f090f2481cb8e0a1c97";
pub const REGISTERED_BUILD_COMMAND: &str = "cargo build --release --locked --features cudnn";
const REGISTERED_PARENT_CHECKPOINT_RELATIVE: &str = "checkpoints/step-000000004096/ema.safetensors";
const REGISTERED_PARENT_CONFIG_RELATIVE: &str = "config.json";

pub(crate) const EVIDENCE_CLASS: &str = "implementation_smoke";
pub(crate) const FAILED_EVIDENCE_CLASS: &str = "failed_infrastructure_or_integrity";
pub(crate) const LIFECYCLE_RUNNING: &str = "running";
pub(crate) const LIFECYCLE_COMPLETE: &str = "complete_pending_analysis";
pub(crate) const LIFECYCLE_FAILED: &str = "failed_integrity_or_evaluation";
pub(crate) const RUN_CLASS_REGISTERED: &str = "registered_diagnostic";
pub(crate) const RUN_CLASS_PREFLIGHT: &str = "unregistered_preflight";
pub(crate) const REPORT_FILE: &str = "report.json";
const LIFECYCLE_FILE: &str = "lifecycle.json";
const COMMAND_LOG_FILE: &str = "command.log";
const TRAIN_CONFIG_COPY_FILE: &str = "train_config.json";
const IDENTITY_DOMAIN: &str = "tofy.p2.context_wiring_diagnostic.identity.v1";
const COMMAND_TAG: &str = "p2-context-wiring";

/// Registered census scalars of the fixed 256-pair E2 population (E2R
/// registration, 2026-09-05). The complete census and fingerprint are pinned.
pub(crate) fn registered_census_matches(census: &TwinMemorizationCensus) -> Result<()> {
    let expected: [(&str, usize, usize); 10] = [
        ("pairs", census.pairs, 256),
        ("episodes", census.episodes, 512),
        (
            "single_frame_rule_identifiable",
            census.single_frame_rule_identifiable,
            0,
        ),
        ("divergent_pairs", census.divergent_pairs, 256),
        (
            "pairs_diverging_before_context_len",
            census.pairs_diverging_before_context_len,
            254,
        ),
        ("outcome_changing_rows", census.outcome_changing_rows, 724),
        ("state_differing_rows", census.state_differing_rows, 0),
        ("scorable_rows", census.scorable_rows, 2912),
        (
            "scorable_rows_after_first_divergence",
            census.scorable_rows_after_first_divergence,
            2878,
        ),
        (
            "scorable_rows_with_evidence_in_window",
            census.scorable_rows_with_evidence_in_window,
            2858,
        ),
    ];
    for (name, got, want) in expected {
        if got != want {
            bail!("registered census drift: {name}={got}, registered {want}");
        }
    }
    if census.chronological_transitions != 10_780 {
        bail!(
            "registered census drift: chronological_transitions={}, registered 10780",
            census.chronological_transitions
        );
    }
    let expected_histogram = BTreeMap::from([
        ("6".to_owned(), 245usize),
        ("13".to_owned(), 9usize),
        ("20".to_owned(), 1usize),
        ("27".to_owned(), 1usize),
    ]);
    if census.first_divergence_histogram != expected_histogram {
        bail!(
            "registered census drift: first_divergence_histogram={:?}, registered {:?}",
            census.first_divergence_histogram,
            expected_histogram
        );
    }
    Ok(())
}

// ---- specification -------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWiringSpec {
    pub pairs: usize,
    pub population_seed: u64,
    pub context_len: usize,
    pub history: LearningHistoryConfig,
    /// Hard cap on updates per arm; must belong to the checkpoint family.
    pub max_updates: usize,
    /// Members of [`CONTEXT_WIRING_CHECKPOINT_FAMILY`] `<= max_updates`.
    pub checkpoint_family: Vec<usize>,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub gradient_clip: f64,
    pub physical_batch: usize,
    pub grad_accum: usize,
    pub d_threshold: f64,
    pub interaction_threshold: f64,
    pub probability_l1_threshold: f64,
}

impl ContextWiringSpec {
    /// The exact registered E2W configuration.
    pub fn registered() -> Self {
        Self::with_budget(REGISTERED_PAIRS, CONTEXT_WIRING_MAX_UPDATES)
    }

    /// The registered contract with a smaller population scan and/or update
    /// budget (unregistered preflights, tests). `max_updates` must belong to
    /// the fixed family.
    pub fn with_budget(pairs: usize, max_updates: usize) -> Self {
        Self {
            pairs,
            population_seed: TWIN_MEMORIZATION_POPULATION_SEED,
            context_len: TWIN_MEMORIZATION_CONTEXT_LEN,
            history: LearningHistoryConfig::training(),
            max_updates,
            checkpoint_family: CONTEXT_WIRING_CHECKPOINT_FAMILY
                .iter()
                .copied()
                .filter(|update| *update <= max_updates)
                .collect(),
            learning_rate: CONTEXT_WIRING_LEARNING_RATE,
            beta1: CONTEXT_WIRING_BETA1,
            beta2: CONTEXT_WIRING_BETA2,
            epsilon: CONTEXT_WIRING_EPSILON,
            weight_decay: CONTEXT_WIRING_WEIGHT_DECAY,
            gradient_clip: CONTEXT_WIRING_GRADIENT_CLIP,
            physical_batch: CONTEXT_WIRING_PHYSICAL_BATCH,
            grad_accum: CONTEXT_WIRING_GRAD_ACCUM,
            d_threshold: CONTEXT_WIRING_D_THRESHOLD,
            interaction_threshold: CONTEXT_WIRING_INTERACTION_THRESHOLD,
            probability_l1_threshold: CONTEXT_WIRING_PROBABILITY_L1_THRESHOLD,
        }
    }

    pub fn twin_spec(&self) -> TwinMemorizationSpec {
        TwinMemorizationSpec {
            pairs: self.pairs,
            population_seed: self.population_seed,
            context_len: self.context_len,
            history: self.history.clone(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.twin_spec().validate()?;
        if self.max_updates == 0 || self.max_updates > CONTEXT_WIRING_MAX_UPDATES {
            bail!("max_updates must be in 1..={CONTEXT_WIRING_MAX_UPDATES}");
        }
        if !CONTEXT_WIRING_CHECKPOINT_FAMILY.contains(&self.max_updates) {
            bail!(
                "max_updates {} is not a member of the fixed checkpoint family {:?}",
                self.max_updates,
                CONTEXT_WIRING_CHECKPOINT_FAMILY
            );
        }
        let expected = CONTEXT_WIRING_CHECKPOINT_FAMILY
            .iter()
            .copied()
            .filter(|update| *update <= self.max_updates)
            .collect::<Vec<_>>();
        if self.checkpoint_family != expected {
            bail!(
                "checkpoint family {:?} is not the fixed family prefix {expected:?}",
                self.checkpoint_family
            );
        }
        if self.physical_batch != CONTEXT_WIRING_PHYSICAL_BATCH
            || self.grad_accum != CONTEXT_WIRING_GRAD_ACCUM
        {
            bail!("E2W trains exactly two rows with physical batch 2 and accumulation 1");
        }
        Ok(())
    }

    /// Whether this spec is the exact registered contract.
    pub fn is_registered_contract(&self) -> bool {
        *self == Self::registered()
    }
}

/// Fail closed unless `cfg` is the v6 2x2 F32 foundation-v2 contract.
pub fn ensure_v6_2x2_f32_config(cfg: &TrainConfig) -> Result<()> {
    if cfg.recipe != TrainingRecipe::FoundationV2 {
        bail!(
            "E2W requires the foundation-v2 recipe, got {:?}",
            cfg.recipe
        );
    }
    if !cfg.world_core_v6 || !cfg.data_contract_v6 {
        bail!("E2W requires world_core_v6 and data_contract_v6");
    }
    if cfg.inner_steps != 2 || cfg.outer_steps != 2 || cfg.v6_recursion_steps != 2 {
        bail!(
            "E2W requires the 2x2 recursion, got inner={} outer={} v6_recursion_steps={}",
            cfg.inner_steps,
            cfg.outer_steps,
            cfg.v6_recursion_steps
        );
    }
    if cfg.bf16_conv || cfg.bf16_recurrent_core {
        bail!("E2W requires F32 computation (bf16_conv and bf16_recurrent_core off)");
    }
    if cfg.model_config().patch_size != PATCH_SIZE {
        bail!("E2W requires the patch-{PATCH_SIZE} exact-decoder topology");
    }
    Ok(())
}

// ---- model-free row selection ---------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionInvariants {
    pub state_differing_positions: usize,
    /// Current frames and actions are bit-identical at every window position.
    pub window_states_identical: bool,
    /// Current frame, action (with coordinates), goal, content mask and
    /// content rectangle of the two score rows are bit-identical.
    pub query_inputs_identical: bool,
    /// Both score rows carry no operator: v6 UNKNOWN conditioning.
    pub unknown_operator_conditioning: bool,
    pub targets_differ: bool,
    pub disagreement_mask_nonempty: bool,
}

/// Frozen identity of the selected pair and row (model-free).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWiringSelection {
    pub pair_ordinal: usize,
    pub meta_episode_id: u64,
    pub position: usize,
    pub context_len: usize,
    pub window_start: usize,
    pub first_divergence: Option<usize>,
    pub outcome_changing_positions_in_window: Vec<usize>,
    pub pairs_scanned: usize,
    pub gameplay_pixels: usize,
    pub target_disagreement_pixels: usize,
    /// Canonical full-row hashes of the two score rows carrying their own
    /// K=16 windows (the exact training rows of the `correct` arm).
    pub primary_row_sha256: String,
    pub twin_row_sha256: String,
    pub primary_window_sha256: String,
    pub twin_window_sha256: String,
    pub primary_target_sha256: String,
    pub twin_target_sha256: String,
    pub invariants: SelectionInvariants,
}

/// The two score rows with their own windows plus the disagreement mask.
#[derive(Debug, Clone)]
pub struct ContextWiringRows {
    pub primary: TransitionSample,
    pub twin: TransitionSample,
    pub primary_window: Vec<ContextTransition>,
    pub twin_window: Vec<ContextTransition>,
    /// Gameplay pixel indices where the two targets differ.
    pub disagreement: Vec<usize>,
}

fn sha256_hex(digest: Sha256) -> String {
    format!("sha256:{:x}", digest.finalize())
}

fn row_sha256(row: &TransitionSample) -> String {
    let mut digest = Sha256::new();
    update_canonical_transition_row(&mut digest, row);
    sha256_hex(digest)
}

fn window_sha256(window: &[ContextTransition]) -> String {
    let mut digest = Sha256::new();
    digest.update((window.len() as u64).to_le_bytes());
    for transition in window {
        digest.update(&transition.current.pixels[..]);
        digest.update([transition.action.id]);
        for coordinate in [transition.action.x, transition.action.y] {
            digest.update(coordinate.map_or([0u8, 0], |value| [1u8, value]));
        }
        digest.update(&transition.next.pixels[..]);
    }
    sha256_hex(digest)
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    sha256_hex(digest)
}

/// Whether position `p` of `pair` satisfies (a) differing targets, (b) an
/// outcome-changing transition at `p`, and (c) at least one earlier
/// outcome-changing transition inside `[p - k, p)`.
fn position_qualifies(pair: &AugmentedTwinPair, position: usize, k: usize) -> bool {
    let targets_differ = pair.primary.chronological_row(position).transition.next
        != pair.twin.chronological_row(position).transition.next;
    let outcome_changing = pair
        .divergence
        .outcome_changing_positions
        .contains(&position);
    targets_differ && outcome_changing && twin_window_has_evidence(&pair.divergence, position, k)
}

/// Build the score rows at `position` with their own exact `k` windows.
pub fn context_wiring_rows(
    pair: &AugmentedTwinPair,
    position: usize,
    k: usize,
    gameplay_pixels: usize,
) -> Result<ContextWiringRows> {
    let primary = twin_memorization_scoring_row(&pair.primary, position, k)?;
    let twin = twin_memorization_scoring_row(&pair.twin, position, k)?;
    let disagreement = primary
        .next
        .pixels
        .iter()
        .zip(twin.next.pixels.iter())
        .take(gameplay_pixels)
        .enumerate()
        .filter_map(|(pixel, (left, right))| (left != right).then_some(pixel))
        .collect::<Vec<_>>();
    Ok(ContextWiringRows {
        primary_window: primary.context.clone(),
        twin_window: twin.context.clone(),
        primary,
        twin,
        disagreement,
    })
}

/// Check every preregistered model-free invariant of the selected rows.
/// Any failure is an integrity failure, not a negative model result.
fn verify_selection_invariants(
    pair: &AugmentedTwinPair,
    rows: &ContextWiringRows,
    position: usize,
    k: usize,
) -> Result<SelectionInvariants> {
    let state_differing_positions = pair.divergence.state_differing_positions;
    let window_states_identical = (position - k..position).all(|index| {
        let left = &pair.primary.chronological_row(index).transition;
        let right = &pair.twin.chronological_row(index).transition;
        left.current == right.current && left.action == right.action
    });
    let (left, right) = (
        pair.primary.chronological_row(position),
        pair.twin.chronological_row(position),
    );
    let query_inputs_identical = left.transition.current == right.transition.current
        && left.transition.action == right.transition.action
        && left.transition.goal_features == right.transition.goal_features
        && left.content_mask == right.content_mask
        && left.transition.provenance.content_width == right.transition.provenance.content_width
        && left.transition.provenance.content_height == right.transition.provenance.content_height
        && left.transition.provenance.content_x == right.transition.provenance.content_x
        && left.transition.provenance.content_y == right.transition.provenance.content_y
        && left.transition.provenance.available_actions
            == right.transition.provenance.available_actions
        && left.transition.provenance.background_color
            == right.transition.provenance.background_color;
    let unknown_operator_conditioning = left.transition.provenance.operator.is_none()
        && right.transition.provenance.operator.is_none();
    let targets_differ = rows.primary.next != rows.twin.next;
    let disagreement_mask_nonempty = !rows.disagreement.is_empty();
    let invariants = SelectionInvariants {
        state_differing_positions,
        window_states_identical,
        query_inputs_identical,
        unknown_operator_conditioning,
        targets_differ,
        disagreement_mask_nonempty,
    };
    if state_differing_positions != 0
        || !window_states_identical
        || !query_inputs_identical
        || !unknown_operator_conditioning
        || !targets_differ
        || !disagreement_mask_nonempty
    {
        bail!("E2W selection invariants failed (integrity failure): {invariants:?}");
    }
    Ok(invariants)
}

/// Scan `pairs` in ascending meta-episode order and select the first pair
/// that is not single-frame rule identifiable together with the earliest
/// position `p >= k` satisfying the preregistered (a)/(b)/(c) conditions.
/// Fails closed when no pair qualifies or an invariant fails.
pub fn select_context_wiring_rows(
    pairs: &[AugmentedTwinPair],
    k: usize,
    gameplay_pixels: usize,
) -> Result<(ContextWiringSelection, ContextWiringRows)> {
    select_context_wiring_rows_from(pairs, k, gameplay_pixels, 0)
}

/// [`select_context_wiring_rows`] restricted to meta-episode IDs
/// `>= first_meta_episode_id` (E2C scans `2..`, excluding E2W's rejected
/// meta-episode 0 and trained meta-episode 1). `pair_ordinal` remains the
/// index into the complete population; `pairs_scanned` counts scanned pairs.
pub fn select_context_wiring_rows_from(
    pairs: &[AugmentedTwinPair],
    k: usize,
    gameplay_pixels: usize,
    first_meta_episode_id: u64,
) -> Result<(ContextWiringSelection, ContextWiringRows)> {
    if pairs.is_empty() {
        bail!("E2W selection needs at least one twin pair");
    }
    if pairs
        .windows(2)
        .any(|window| window[0].primary.meta_episode_id >= window[1].primary.meta_episode_id)
    {
        bail!("E2W selection requires pairs in strictly ascending meta-episode order");
    }
    let skipped = pairs
        .iter()
        .take_while(|pair| pair.primary.meta_episode_id < first_meta_episode_id)
        .count();
    if skipped == pairs.len() {
        bail!(
            "no twin pair among {} has meta-episode ID >= {first_meta_episode_id}",
            pairs.len()
        );
    }
    for (ordinal, pair) in pairs.iter().enumerate().skip(skipped) {
        if pair.divergence.single_frame_rule_identifiable {
            continue;
        }
        if pair.primary.chronological.len() != pair.twin.chronological.len() {
            bail!("twin pair {ordinal} has different chronological lengths");
        }
        let len = pair.primary.chronological.len();
        let Some(position) = (k..len).find(|&position| position_qualifies(pair, position, k))
        else {
            continue;
        };
        let rows = context_wiring_rows(pair, position, k, gameplay_pixels)?;
        let invariants = verify_selection_invariants(pair, &rows, position, k)?;
        let selection = ContextWiringSelection {
            pair_ordinal: ordinal,
            meta_episode_id: pair.primary.meta_episode_id,
            position,
            context_len: k,
            window_start: position - k,
            first_divergence: pair.divergence.first_divergence,
            outcome_changing_positions_in_window: pair
                .divergence
                .outcome_changing_positions
                .iter()
                .copied()
                .filter(|&changed| changed >= position - k && changed < position)
                .collect(),
            pairs_scanned: ordinal + 1 - skipped,
            gameplay_pixels,
            target_disagreement_pixels: rows.disagreement.len(),
            primary_row_sha256: row_sha256(&rows.primary),
            twin_row_sha256: row_sha256(&rows.twin),
            primary_window_sha256: window_sha256(&rows.primary_window),
            twin_window_sha256: window_sha256(&rows.twin_window),
            primary_target_sha256: bytes_sha256(&rows.primary.next.pixels),
            twin_target_sha256: bytes_sha256(&rows.twin.next.pixels),
            invariants,
        };
        return Ok((selection, rows));
    }
    bail!(
        "no twin pair among {} (meta-episode IDs >= {first_meta_episode_id}) qualifies (need a \
         non-identifiable pair with a position p >= {k} whose targets differ, is \
         outcome-changing and has an earlier outcome-changing row in its window)",
        pairs.len()
    );
}

pub(crate) fn with_context(
    row: &TransitionSample,
    window: &[ContextTransition],
) -> TransitionSample {
    let mut row = row.clone();
    row.context = window.to_vec();
    row.provenance.context_len = u8::try_from(window.len()).expect("context window fits u8");
    row
}

// ---- training ------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub update: usize,
    /// Direct Unimix CE over target-disagreement pixels (pre-update).
    pub loss: f64,
    pub pre_clip_gradient_norm: f64,
    pub gradient_clip_scale: f64,
    /// L2 norm of the gradient over every `context_*` parameter.
    pub context_gradient_norm: f64,
}

/// How an arm orders its floating parameters when constructing AdamW,
/// reducing the global gradient norm, applying the clip and measuring the
/// context-parameter gradient norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterOrdering {
    /// E2W as registered: `VarMap` hash-map iteration order (process-random
    /// floating reduction order; kept verbatim so E2W stays reproducible).
    VarMapIteration,
    /// E2C: canonical ascending parameter-name order in every arm and process.
    CanonicalNameSorted,
}

pub(crate) struct Arm {
    pub(crate) name: &'static str,
    pub(crate) model: WorldModel,
    pub(crate) varmap: VarMap,
    pub(crate) ordering: ParameterOrdering,
    /// Floating parameters in canonical name order (used by
    /// [`ParameterOrdering::CanonicalNameSorted`]).
    pub(crate) parameters: Vec<(String, Var)>,
    pub(crate) optimizer: AdamW,
    pub(crate) batch: BatchTensors,
    pub(crate) updates: Vec<UpdateRecord>,
    pub(crate) checkpoints: Vec<CheckpointEvaluation>,
}

/// Every floating parameter of `varmap` sorted by canonical name.
pub(crate) fn ordered_float_parameters(varmap: &VarMap) -> Vec<(String, Var)> {
    let data = varmap.data().lock().unwrap();
    let mut parameters = data
        .iter()
        .filter(|(_, var)| var.dtype().is_float())
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    parameters
}

/// SHA-256 over the ordered parameter names, dtypes and shapes.
pub(crate) fn ordered_parameter_names_sha256(parameters: &[(String, Var)]) -> String {
    let mut digest = Sha256::new();
    digest.update((parameters.len() as u64).to_le_bytes());
    for (name, var) in parameters {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(format!("{:?}", var.dtype()).as_bytes());
        digest.update((var.shape().rank() as u64).to_le_bytes());
        for dim in var.shape().dims() {
            digest.update((*dim as u64).to_le_bytes());
        }
    }
    sha256_hex(digest)
}

/// SHA-256 over the F32 bits of the already canonically ordered floating
/// parameter list. E2C uses the same list for AdamW, clipping and state hashes.
pub(crate) fn ordered_parameter_sha256(parameters: &[(String, Var)]) -> Result<String> {
    let mut digest = Sha256::new();
    for (name, var) in parameters {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(format!("{:?}", var.dtype()).as_bytes());
        digest.update((var.shape().rank() as u64).to_le_bytes());
        for dim in var.shape().dims() {
            digest.update((*dim as u64).to_le_bytes());
        }
        let values = var
            .as_tensor()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        digest.update((values.len() as u64).to_le_bytes());
        for value in values {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(sha256_hex(digest))
}

/// Global-norm clip whose floating reduction runs in the given parameter
/// order (E2C: canonical name order). Same contract as
/// [`try_clip_gradients_gpu_with_stats`]: `Ok(None)` on a non-finite norm.
pub(crate) fn clip_gradients_ordered(
    grads: &mut GradStore,
    parameters: &[(String, Var)],
    max_norm: f64,
) -> Result<Option<GradientClipStats>> {
    let mut sum_sq: Option<Tensor> = None;
    for (_, var) in parameters {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let sq = grad.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                None => sq,
                Some(acc) => acc.add(&sq)?,
            });
        }
    }
    let Some(sum_sq) = sum_sq else {
        return Ok(Some(GradientClipStats {
            pre_clip_norm: 0.0,
            scale: 1.0,
        }));
    };
    let norm = f64::from(sum_sq.sqrt()?.to_scalar::<f32>()?);
    if !norm.is_finite() {
        return Ok(None);
    }
    if norm <= max_norm {
        return Ok(Some(GradientClipStats {
            pre_clip_norm: norm,
            scale: 1.0,
        }));
    }
    let scale = max_norm / norm;
    for (_, var) in parameters {
        let tensor = var.as_tensor();
        if let Some(grad) = grads.get(tensor) {
            grads.insert(tensor, grad.affine(scale, 0.0)?);
        }
    }
    Ok(Some(GradientClipStats {
        pre_clip_norm: norm,
        scale,
    }))
}

/// Load a fresh model, build the fixed-row physical batch of one arm and its
/// zero-state AdamW in the requested parameter order.
pub(crate) fn build_arm(
    name: &'static str,
    rows: &ContextWiringRows,
    windows: (&[ContextTransition], &[ContextTransition]),
    spec: &ContextWiringSpec,
    ordering: ParameterOrdering,
    device: &Device,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
) -> Result<Arm> {
    let (model, varmap) = load_arm()?;
    if !model.config().world_core_v6 {
        bail!("E2W requires a world_core_v6 model");
    }
    let params = ParamsAdamW {
        lr: spec.learning_rate,
        beta1: spec.beta1,
        beta2: spec.beta2,
        eps: spec.epsilon,
        weight_decay: spec.weight_decay,
    };
    let parameters = ordered_float_parameters(&varmap);
    let vars = match ordering {
        ParameterOrdering::VarMapIteration => varmap
            .all_vars()
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect::<Vec<_>>(),
        ParameterOrdering::CanonicalNameSorted => {
            parameters.iter().map(|(_, var)| var.clone()).collect()
        }
    };
    let optimizer = AdamW::new(vars, params)?;
    let batch = batch_from_samples(
        &[
            with_context(&rows.primary, windows.0),
            with_context(&rows.twin, windows.1),
        ],
        device,
    )?;
    if batch.context.is_none() {
        bail!("E2W training rows carry no context");
    }
    Ok(Arm {
        name,
        model,
        varmap,
        ordering,
        parameters,
        optimizer,
        batch,
        updates: Vec::new(),
        checkpoints: Vec::new(),
    })
}

/// Direct raw exact-decoder Unimix cross-entropy under the production 2x2
/// training recurrence (zero noise), reduced only over `mask` pixels.
pub(crate) fn context_wiring_loss(
    model: &WorldModel,
    batch: &BatchTensors,
    mask: &Tensor,
    mask_pixels: usize,
) -> Result<Tensor> {
    let rows = gameplay_rows(model.config().world_core_v6);
    let encoded = model.encode_state_pair_for_training(&batch.frames, &batch.next_frames)?;
    let current_canonical = model.canonical_representation(&encoded.current)?;
    let out = model
        .full_v4_training_latents_from_encoded_state_with_operator_conditioning_with_context(
            &encoded.current,
            &current_canonical,
            &batch.actions,
            &batch.action_coords,
            &batch.operator_conditioning,
            batch.context.as_ref(),
            RecursionDepth::from_config(model.config()),
            0.0,
            None,
            RecursionOpts::training(true),
        )?;
    let target_labels = batch
        .next_frames
        .narrow(2, 0, rows)?
        .squeeze(1)?
        .to_dtype(DType::U32)?
        .contiguous()?;
    let logits = model.exact_gameplay_logits_trainable(&out.y)?;
    let per_pixel = foundation_v2_unimix_ce(&logits, &target_labels)?;
    reduce_disagreement_loss(&per_pixel, mask, mask_pixels)
}

fn reduce_disagreement_loss(
    per_pixel: &Tensor,
    mask: &Tensor,
    mask_pixels: usize,
) -> Result<Tensor> {
    if mask_pixels == 0 {
        bail!("E2W disagreement loss denominator is zero");
    }
    per_pixel
        .mul(mask)?
        .sum_all()?
        .affine(1.0 / mask_pixels as f64, 0.0)
        .map_err(Into::into)
}

fn context_gradient_norm(varmap: &VarMap, grads: &GradStore) -> Result<f64> {
    let data = varmap.data().lock().unwrap();
    context_gradient_norm_of(data.iter(), grads)
}

/// L2 norm of the gradient over every `context_*` parameter, reduced in the
/// iteration order of `parameters`.
pub(crate) fn context_gradient_norm_of<'a>(
    parameters: impl Iterator<Item = (&'a String, &'a Var)>,
    grads: &GradStore,
) -> Result<f64> {
    let mut matched = 0usize;
    let mut sum = 0f64;
    for (_, var) in parameters.filter(|(name, _)| name.starts_with(CONTEXT_PARAMETER_PREFIX)) {
        matched += 1;
        if let Some(grad) = grads.get(var.as_tensor()) {
            sum += f64::from(
                grad.to_dtype(DType::F32)?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
        }
    }
    if matched == 0 {
        bail!("the model has no `{CONTEXT_PARAMETER_PREFIX}*` parameters");
    }
    Ok(sum.sqrt())
}

pub(crate) fn disagreement_mask(
    disagreement: &[usize],
    rows: usize,
    gameplay_rows: usize,
    device: &Device,
) -> Result<Tensor> {
    let pixels = gameplay_rows * FRAME_SIDE;
    let mut values = vec![0f32; rows * pixels];
    for row in 0..rows {
        for &pixel in disagreement {
            values[row * pixels + pixel] = 1.0;
        }
    }
    Tensor::from_vec(values, (rows, gameplay_rows, FRAME_SIDE), device).map_err(Into::into)
}

pub(crate) fn train_update(
    arm: &mut Arm,
    mask: &Tensor,
    mask_pixels: usize,
    spec: &ContextWiringSpec,
    device: &Device,
) -> Result<UpdateRecord> {
    let update = arm.updates.len() + 1;
    let loss = context_wiring_loss(&arm.model, &arm.batch, mask, mask_pixels)?;
    let loss_value = f64::from(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?);
    if !loss_value.is_finite() {
        bail!(
            "{} arm update {update}: non-finite loss {loss_value}",
            arm.name
        );
    }
    let mut grads = loss.backward()?;
    let context_gradient_norm = match arm.ordering {
        ParameterOrdering::VarMapIteration => context_gradient_norm(&arm.varmap, &grads)?,
        ParameterOrdering::CanonicalNameSorted => {
            context_gradient_norm_of(arm.parameters.iter().map(|(name, var)| (name, var)), &grads)?
        }
    };
    if !context_gradient_norm.is_finite() || (update == 1 && context_gradient_norm == 0.0) {
        bail!(
            "{} arm update {update}: context-parameter gradient norm {context_gradient_norm} \
             (zero or non-finite fails closed)",
            arm.name
        );
    }
    let clip = match arm.ordering {
        ParameterOrdering::VarMapIteration => {
            try_clip_gradients_gpu_with_stats(&mut grads, &arm.varmap, spec.gradient_clip)?
        }
        ParameterOrdering::CanonicalNameSorted => {
            clip_gradients_ordered(&mut grads, &arm.parameters, spec.gradient_clip)?
        }
    };
    let Some(clip) = clip else {
        bail!(
            "{} arm update {update}: non-finite global gradient norm",
            arm.name
        );
    };
    arm.optimizer.step(&grads)?;
    sync_cuda_device(device)?;
    let record = UpdateRecord {
        update,
        loss: loss_value,
        pre_clip_gradient_norm: clip.pre_clip_norm,
        gradient_clip_scale: clip.scale,
        context_gradient_norm,
    };
    arm.updates.push(record.clone());
    Ok(record)
}

// ---- frozen evaluation ------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextArmScores {
    /// Mean `-log softmax(logits)[target]` over disagreement pixels.
    pub raw_softmax_nll: f64,
    /// Mean `-log(0.99 p + 0.01/16)` over the same pixels (the objective).
    pub unimix_nll: f64,
    pub raw_argmax_correct: usize,
    pub composed_argmax_correct: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextComparison {
    /// Mean L1 between the two 16-colour distributions over disagreement pixels.
    pub probability_l1: f64,
    pub latent_rms_difference: f64,
    pub context_summary_rms_difference: f64,
    /// Mean absolute copy-gate difference over disagreement pixels.
    pub copy_gate_mean_absolute_difference: f64,
    pub raw_argmax_disagreement_pixels: usize,
    pub composed_argmax_disagreement_pixels: usize,
}

/// Bit-identity of a K0 row inside `[K0, own-context carrier]` versus the
/// same row inside a shape-matched all-K0 batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MixedK0Invariant {
    pub pass: bool,
    pub latent_elements_differing: usize,
    pub raw_probability_elements_differing: usize,
    pub log_probability_elements_differing: usize,
    pub copy_gate_elements_differing: usize,
    pub raw_argmax_pixels_differing: usize,
    pub composed_argmax_pixels_differing: usize,
    pub raw_nll_bit_identical: bool,
    pub unimix_nll_bit_identical: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionEvaluation {
    pub direction: String,
    pub disagreement_pixels: usize,
    pub own: ContextArmScores,
    pub paired: ContextArmScores,
    pub k0: ContextArmScores,
    pub own_vs_paired: ContextComparison,
    pub own_vs_k0: ContextComparison,
    /// `raw_softmax_nll(paired) - raw_softmax_nll(own)`.
    pub d: f64,
    pub mixed_k0_invariant: MixedK0Invariant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointEvaluation {
    pub update: usize,
    pub directions: Vec<DirectionEvaluation>,
    /// `D` averaged equally across the two query directions.
    pub d: f64,
    /// Own-versus-paired probability L1 pooled across both directions.
    pub probability_l1: f64,
    pub raw_argmax_disagreement_pixels: usize,
    pub mixed_k0_invariant_pass: bool,
    /// Own context has strictly more correct raw argmax pixels than paired
    /// context and than K0 in each direction.
    pub promotion_gate: bool,
}

pub(crate) struct DirectionRows {
    pub(crate) direction: &'static str,
    pub(crate) own: TransitionSample,
    pub(crate) paired: TransitionSample,
    pub(crate) k0: TransitionSample,
    pub(crate) target: Vec<u8>,
    pub(crate) disagreement: Vec<usize>,
}

/// The two data-true query directions: each score row with its own window,
/// the paired (exchanged) window and no context.
pub(crate) fn direction_rows(rows: &ContextWiringRows) -> [DirectionRows; 2] {
    let no_context = |row: &TransitionSample| with_context(row, &[]);
    [
        DirectionRows {
            direction: "primary",
            own: with_context(&rows.primary, &rows.primary_window),
            paired: with_context(&rows.primary, &rows.twin_window),
            k0: no_context(&rows.primary),
            target: rows.primary.next.pixels.to_vec(),
            disagreement: rows.disagreement.clone(),
        },
        DirectionRows {
            direction: "twin",
            own: with_context(&rows.twin, &rows.twin_window),
            paired: with_context(&rows.twin, &rows.primary_window),
            k0: no_context(&rows.twin),
            target: rows.twin.next.pixels.to_vec(),
            disagreement: rows.disagreement.clone(),
        },
    ]
}

fn decode_single(
    model: &WorldModel,
    row: &TransitionSample,
    device: &Device,
) -> Result<TwinContinuousDecodes> {
    twin_continuous_decode_rows(model, std::slice::from_ref(row), device)
}

pub(crate) fn ensure_finite(decodes: &TwinContinuousDecodes, label: &str) -> Result<()> {
    let finite = decodes
        .latent
        .iter()
        .chain(&decodes.probabilities)
        .chain(&decodes.log_probs)
        .chain(&decodes.copy_gate)
        .chain(&decodes.context_summary)
        .flatten()
        .all(|value| value.is_finite());
    if !finite {
        bail!("E2W evaluation produced a non-finite value ({label})");
    }
    Ok(())
}

fn arm_scores(
    decodes: &TwinContinuousDecodes,
    target: &[u8],
    disagreement: &[usize],
) -> Result<ContextArmScores> {
    arm_scores_row(decodes, 0, target, disagreement)
}

/// [`arm_scores`] for row `row` of a multi-row decode.
pub(crate) fn arm_scores_row(
    decodes: &TwinContinuousDecodes,
    row: usize,
    target: &[u8],
    disagreement: &[usize],
) -> Result<ContextArmScores> {
    let (Some(log_probs), Some(probabilities), Some(raw), Some(composed)) = (
        decodes.log_probs.get(row),
        decodes.probabilities.get(row),
        decodes.true_predictions.get(row),
        decodes.composed.get(row),
    ) else {
        bail!("E2W decode has no row {row}");
    };
    if log_probs.len() != raw.len() * PALETTE_SIZE
        || probabilities.len() != log_probs.len()
        || raw.len() != composed.len()
    {
        bail!("E2W decode shapes are inconsistent");
    }
    let mut raw_nll = 0f64;
    let mut unimix_nll = 0f64;
    let mut raw_correct = 0usize;
    let mut composed_correct = 0usize;
    for &pixel in disagreement {
        let color = usize::from(target[pixel]);
        if color >= PALETTE_SIZE || pixel >= raw.len() {
            bail!("E2W target pixel outside the palette or frame");
        }
        let log_p = f64::from(log_probs[pixel * PALETTE_SIZE + color]);
        let probability = f64::from(probabilities[pixel * PALETTE_SIZE + color]);
        raw_nll -= log_p;
        unimix_nll -= (0.99 * probability + 0.01 / PALETTE_SIZE as f64).ln();
        raw_correct += usize::from(raw[pixel] == target[pixel]);
        composed_correct += usize::from(composed[pixel] == target[pixel]);
    }
    let count = disagreement.len() as f64;
    Ok(ContextArmScores {
        raw_softmax_nll: raw_nll / count,
        unimix_nll: unimix_nll / count,
        raw_argmax_correct: raw_correct,
        composed_argmax_correct: composed_correct,
    })
}

fn rms_difference(left: &[f32], right: &[f32]) -> Result<f64> {
    if left.len() != right.len() || left.is_empty() {
        bail!("E2W compares vectors of different or zero length");
    }
    let sum = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let difference = f64::from(*a) - f64::from(*b);
            difference * difference
        })
        .sum::<f64>();
    Ok((sum / left.len() as f64).sqrt())
}

fn compare(
    left: &TwinContinuousDecodes,
    right: &TwinContinuousDecodes,
    disagreement: &[usize],
) -> Result<ContextComparison> {
    let mut probability_l1 = 0f64;
    let mut gate = 0f64;
    let mut raw_disagreement = 0usize;
    let mut composed_disagreement = 0usize;
    for &pixel in disagreement {
        probability_l1 += (0..PALETTE_SIZE)
            .map(|color| {
                let offset = pixel * PALETTE_SIZE + color;
                (f64::from(left.probabilities[0][offset])
                    - f64::from(right.probabilities[0][offset]))
                .abs()
            })
            .sum::<f64>();
        gate += (f64::from(left.copy_gate[0][pixel]) - f64::from(right.copy_gate[0][pixel])).abs();
        raw_disagreement +=
            usize::from(left.true_predictions[0][pixel] != right.true_predictions[0][pixel]);
        composed_disagreement += usize::from(left.composed[0][pixel] != right.composed[0][pixel]);
    }
    let count = disagreement.len() as f64;
    Ok(ContextComparison {
        probability_l1: probability_l1 / count,
        latent_rms_difference: rms_difference(&left.latent[0], &right.latent[0])?,
        context_summary_rms_difference: rms_difference(
            &left.context_summary[0],
            &right.context_summary[0],
        )?,
        copy_gate_mean_absolute_difference: gate / count,
        raw_argmax_disagreement_pixels: raw_disagreement,
        composed_argmax_disagreement_pixels: composed_disagreement,
    })
}

pub(crate) fn bits_differing(left: &[f32], right: &[f32]) -> usize {
    if left.len() != right.len() {
        return left.len().max(right.len());
    }
    left.iter()
        .zip(right)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// Row 0 of `[K0, own]` versus row 0 of `[K0, K0]`, bit for bit.
fn mixed_k0_invariant(
    model: &WorldModel,
    k0: &TransitionSample,
    own: &TransitionSample,
    target: &[u8],
    disagreement: &[usize],
    device: &Device,
) -> Result<MixedK0Invariant> {
    let mixed = twin_continuous_decode_rows(model, &[k0.clone(), own.clone()], device)?;
    let control = twin_continuous_decode_rows(model, &[k0.clone(), k0.clone()], device)?;
    ensure_finite(&mixed, "mixed K0 batch")?;
    ensure_finite(&control, "all-K0 control batch")?;
    let raw = mixed.true_predictions[0]
        .iter()
        .zip(&control.true_predictions[0])
        .filter(|(a, b)| a != b)
        .count();
    let composed = mixed.composed[0]
        .iter()
        .zip(&control.composed[0])
        .filter(|(a, b)| a != b)
        .count();
    let mixed_scores = arm_scores(&mixed, target, disagreement)?;
    let control_scores = arm_scores(&control, target, disagreement)?;
    let invariant = MixedK0Invariant {
        pass: false,
        latent_elements_differing: bits_differing(&mixed.latent[0], &control.latent[0]),
        raw_probability_elements_differing: bits_differing(
            &mixed.probabilities[0],
            &control.probabilities[0],
        ),
        log_probability_elements_differing: bits_differing(
            &mixed.log_probs[0],
            &control.log_probs[0],
        ),
        copy_gate_elements_differing: bits_differing(&mixed.copy_gate[0], &control.copy_gate[0]),
        raw_argmax_pixels_differing: raw,
        composed_argmax_pixels_differing: composed,
        raw_nll_bit_identical: mixed_scores.raw_softmax_nll.to_bits()
            == control_scores.raw_softmax_nll.to_bits(),
        unimix_nll_bit_identical: mixed_scores.unimix_nll.to_bits()
            == control_scores.unimix_nll.to_bits(),
    };
    Ok(MixedK0Invariant {
        pass: invariant.latent_elements_differing == 0
            && invariant.raw_probability_elements_differing == 0
            && invariant.log_probability_elements_differing == 0
            && invariant.copy_gate_elements_differing == 0
            && invariant.raw_argmax_pixels_differing == 0
            && invariant.composed_argmax_pixels_differing == 0
            && invariant.raw_nll_bit_identical
            && invariant.unimix_nll_bit_identical,
        ..invariant
    })
}

/// Evaluate one query direction and return the exact batch-1 singleton K0
/// decode that produced its reported `k0` score (E2D retains it for the
/// semantic batch-invariance comparison; E2W and E2C discard it).
fn evaluate_direction(
    model: &WorldModel,
    rows: &DirectionRows,
    device: &Device,
) -> Result<(DirectionEvaluation, TwinContinuousDecodes)> {
    let own = decode_single(model, &rows.own, device)?;
    let paired = decode_single(model, &rows.paired, device)?;
    let k0 = decode_single(model, &rows.k0, device)?;
    ensure_finite(&own, "own context")?;
    ensure_finite(&paired, "paired context")?;
    ensure_finite(&k0, "K0")?;
    let own_scores = arm_scores(&own, &rows.target, &rows.disagreement)?;
    let paired_scores = arm_scores(&paired, &rows.target, &rows.disagreement)?;
    let k0_scores = arm_scores(&k0, &rows.target, &rows.disagreement)?;
    let mixed_k0_invariant = mixed_k0_invariant(
        model,
        &rows.k0,
        &rows.own,
        &rows.target,
        &rows.disagreement,
        device,
    )?;
    let evaluation = DirectionEvaluation {
        direction: rows.direction.into(),
        disagreement_pixels: rows.disagreement.len(),
        d: paired_scores.raw_softmax_nll - own_scores.raw_softmax_nll,
        own: own_scores,
        paired: paired_scores,
        k0: k0_scores,
        own_vs_paired: compare(&own, &paired, &rows.disagreement)?,
        own_vs_k0: compare(&own, &k0, &rows.disagreement)?,
        mixed_k0_invariant,
    };
    Ok((evaluation, k0))
}

pub(crate) fn evaluate_checkpoint(
    model: &WorldModel,
    directions: &[DirectionRows],
    update: usize,
    device: &Device,
) -> Result<CheckpointEvaluation> {
    evaluate_checkpoint_retaining_k0(model, directions, update, device)
        .map(|(evaluation, _)| evaluation)
}

/// [`evaluate_checkpoint`] plus, per direction, the retained canonical
/// batch-1 singleton K0 decode behind the reported `k0` score. The operation
/// order is identical to the plain evaluator.
pub(crate) fn evaluate_checkpoint_retaining_k0(
    model: &WorldModel,
    directions: &[DirectionRows],
    update: usize,
    device: &Device,
) -> Result<(CheckpointEvaluation, Vec<TwinContinuousDecodes>)> {
    let (evaluated, retained_k0): (Vec<_>, Vec<_>) = directions
        .iter()
        .map(|rows| evaluate_direction(model, rows, device))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let count = evaluated.len() as f64;
    let d = evaluated.iter().map(|direction| direction.d).sum::<f64>() / count;
    let probability_l1 = evaluated
        .iter()
        .map(|direction| direction.own_vs_paired.probability_l1)
        .sum::<f64>()
        / count;
    if !d.is_finite() || !probability_l1.is_finite() {
        bail!("E2W checkpoint {update}: non-finite verdict statistic");
    }
    let mixed_k0_invariant_pass = evaluated
        .iter()
        .all(|direction| direction.mixed_k0_invariant.pass);
    if !mixed_k0_invariant_pass {
        bail!(
            "E2W checkpoint {update}: mixed-batch K0 leak (fail closed): {:?}",
            evaluated
                .iter()
                .map(|direction| &direction.mixed_k0_invariant)
                .collect::<Vec<_>>()
        );
    }
    let evaluation = CheckpointEvaluation {
        update,
        d,
        probability_l1,
        raw_argmax_disagreement_pixels: evaluated
            .iter()
            .map(|direction| direction.own_vs_paired.raw_argmax_disagreement_pixels)
            .sum(),
        mixed_k0_invariant_pass,
        promotion_gate: evaluated.iter().all(|direction| {
            direction.own.raw_argmax_correct > direction.paired.raw_argmax_correct
                && direction.own.raw_argmax_correct > direction.k0.raw_argmax_correct
        }),
        directions: evaluated,
    };
    Ok((evaluation, retained_k0))
}

// ---- verdict ---------------------------------------------------------------

/// Gate facts of one checkpoint across both arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointGate {
    pub update: usize,
    pub d_correct: f64,
    pub d_swapped: f64,
    pub interaction: f64,
    pub probability_l1_correct: f64,
    pub probability_l1_swapped: f64,
    pub raw_argmax_disagreement_correct: usize,
    pub raw_argmax_disagreement_swapped: usize,
    pub mixed_k0_invariant_pass: bool,
    pub wiring_gate: bool,
    pub promotion_gate: bool,
}

pub fn checkpoint_gate(
    correct: &CheckpointEvaluation,
    swapped: &CheckpointEvaluation,
    spec: &ContextWiringSpec,
) -> Result<CheckpointGate> {
    if correct.update != swapped.update {
        bail!(
            "arms evaluated different updates: correct={} swapped={}",
            correct.update,
            swapped.update
        );
    }
    let sensitive = |evaluation: &CheckpointEvaluation| {
        evaluation.probability_l1 > spec.probability_l1_threshold
            || evaluation.raw_argmax_disagreement_pixels >= 1
    };
    let interaction = correct.d - swapped.d;
    let mixed_k0_invariant_pass =
        correct.mixed_k0_invariant_pass && swapped.mixed_k0_invariant_pass;
    let wiring_gate = correct.d > spec.d_threshold
        && swapped.d < -spec.d_threshold
        && interaction > spec.interaction_threshold
        && sensitive(correct)
        && sensitive(swapped)
        && mixed_k0_invariant_pass;
    Ok(CheckpointGate {
        update: correct.update,
        d_correct: correct.d,
        d_swapped: swapped.d,
        interaction,
        probability_l1_correct: correct.probability_l1,
        probability_l1_swapped: swapped.probability_l1,
        raw_argmax_disagreement_correct: correct.raw_argmax_disagreement_pixels,
        raw_argmax_disagreement_swapped: swapped.raw_argmax_disagreement_pixels,
        mixed_k0_invariant_pass,
        wiring_gate,
        promotion_gate: correct.promotion_gate,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWiringDecision {
    /// Earliest two consecutive family checkpoints with the wiring gate.
    pub wiring_checkpoints: Option<(usize, usize)>,
    pub wiring_pass: bool,
    /// Earliest two consecutive checkpoints where both wiring and promotion
    /// gates hold. This may be later than `wiring_checkpoints`.
    pub promotion_checkpoints: Option<(usize, usize)>,
    pub promotion_pass: bool,
    /// Early success: stop both arms after the second of those checkpoints.
    pub early_stop: bool,
}

/// Apply the preregistered early-decision rule to the gates evaluated so far
/// (in family order). Wiring and joint wiring+promotion pairs are tracked
/// independently so a wiring-only pair does not suppress a later promotion.
pub fn context_wiring_decision(
    gates: &[CheckpointGate],
    family: &[usize],
) -> Result<ContextWiringDecision> {
    if gates.len() > family.len()
        || gates
            .iter()
            .zip(family)
            .any(|(gate, update)| gate.update != *update)
    {
        bail!("E2W gates are not in fixed family order");
    }
    let wiring_pair = gates
        .windows(2)
        .position(|pair| pair[0].wiring_gate && pair[1].wiring_gate);
    let promotion_pair = gates.windows(2).position(|pair| {
        pair[0].wiring_gate
            && pair[1].wiring_gate
            && pair[0].promotion_gate
            && pair[1].promotion_gate
    });
    let checkpoints = |index: usize| (gates[index].update, gates[index + 1].update);
    Ok(ContextWiringDecision {
        wiring_checkpoints: wiring_pair.map(checkpoints),
        wiring_pass: wiring_pair.is_some(),
        promotion_checkpoints: promotion_pair.map(checkpoints),
        promotion_pass: promotion_pair.is_some(),
        early_stop: promotion_pair.is_some(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWiringVerdict {
    pub statistic: String,
    pub rule: String,
    pub d_threshold: f64,
    pub interaction_threshold: f64,
    pub probability_l1_threshold: f64,
    pub checkpoint_family: Vec<usize>,
    pub evaluated_checkpoints: Vec<usize>,
    pub gates: Vec<CheckpointGate>,
    pub wiring_pass: bool,
    pub wiring_checkpoints: Option<(usize, usize)>,
    pub promotion_pass: bool,
    pub promotion_checkpoints: Option<(usize, usize)>,
    pub early_stop_update: Option<usize>,
    pub updates_run: usize,
    pub outcome: String,
    pub note: String,
}

pub fn context_wiring_verdict(
    gates: Vec<CheckpointGate>,
    spec: &ContextWiringSpec,
    updates_run: usize,
) -> Result<ContextWiringVerdict> {
    let decision = context_wiring_decision(&gates, &spec.checkpoint_family)?;
    let early_stop_update = decision
        .early_stop
        .then(|| decision.promotion_checkpoints.map(|(_, second)| second))
        .flatten()
        .filter(|update| *update < spec.max_updates);
    let outcome = if decision.promotion_pass {
        "wiring_and_promotion_pass"
    } else if decision.wiring_pass {
        "wiring_only_no_promotion"
    } else if updates_run >= CONTEXT_WIRING_MAX_UPDATES {
        "reject_local_trainability_by_update_256"
    } else {
        "no_wiring_pass_within_preflight_budget"
    };
    let note = match (decision.wiring_checkpoints, decision.promotion_checkpoints) {
        (Some((first, second)), Some((promotion_first, promotion_second))) => format!(
            "earliest wiring pair {first}/{second}; earliest joint wiring+promotion pair \
             {promotion_first}/{promotion_second}; outcome {outcome} after {updates_run} updates per arm"
        ),
        (Some((first, second)), None) => format!(
            "earliest wiring pair {first}/{second}; no consecutive joint wiring+promotion pair; \
             outcome {outcome} after {updates_run} updates per arm"
        ),
        (None, _) => format!(
            "no two consecutive family checkpoints satisfied the wiring gate; outcome {outcome} \
             after {updates_run} updates per arm (budget {})",
            spec.max_updates
        ),
    };
    Ok(ContextWiringVerdict {
        statistic: "D = raw_softmax_NLL(paired K=16) - raw_softmax_NLL(own K=16) over \
                    target-disagreement pixels, averaged within each query direction then \
                    equally across both directions"
            .into(),
        rule: "wiring_pass: at the same two consecutive family checkpoints D_correct > 1e-4, \
               D_swapped < -1e-4, D_correct - D_swapped > 2e-4, each arm pooled own-vs-paired \
               probability L1 > 1e-6 or >= 1 raw argmax disagreement, and the mixed-K0 invariant \
               holds. promotion_pass additionally requires \
               the correct arm's own context to have strictly more correct raw argmax pixels \
               than paired and K0 in each direction at the same two consecutive checkpoints; \
               wiring-only does not block a later joint pair, and early stop occurs at the first \
               joint wiring+promotion pair."
            .into(),
        d_threshold: spec.d_threshold,
        interaction_threshold: spec.interaction_threshold,
        probability_l1_threshold: spec.probability_l1_threshold,
        checkpoint_family: spec.checkpoint_family.clone(),
        evaluated_checkpoints: gates.iter().map(|gate| gate.update).collect(),
        gates,
        wiring_pass: decision.wiring_pass,
        wiring_checkpoints: decision.wiring_checkpoints,
        promotion_pass: decision.promotion_pass,
        promotion_checkpoints: decision.promotion_checkpoints,
        early_stop_update,
        updates_run,
        outcome: outcome.into(),
        note,
    })
}

// ---- two-arm execution -----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmReport {
    pub name: String,
    /// Row order of the physical batch: `[primary, twin]` with the context
    /// assignment of this arm.
    pub row_contexts: Vec<String>,
    pub updates: Vec<UpdateRecord>,
    pub checkpoints: Vec<CheckpointEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWiringArms {
    pub correct: ArmReport,
    pub swapped: ArmReport,
    /// SHA-256 over the sorted parameter bits of each arm at initialization
    /// (both must equal the loaded checkpoint state).
    pub initial_parameter_sha256: String,
    pub arms_initialized_identically: bool,
    pub updates_run: usize,
}

/// SHA-256 over every parameter's F32 bits in canonical name order.
pub(crate) fn parameter_sha256(varmap: &VarMap) -> Result<String> {
    let data = varmap.data().lock().unwrap();
    let mut names = data.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut digest = Sha256::new();
    for name in names {
        digest.update(name.as_bytes());
        let values = data[&name]
            .as_tensor()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for value in values {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(sha256_hex(digest))
}

/// The `[physical_batch, gameplay_rows, FRAME_SIDE]` disagreement mask after
/// checking the mask lies inside the decoded gameplay region.
pub(crate) fn training_disagreement_mask(
    rows: &ContextWiringRows,
    spec: &ContextWiringSpec,
    device: &Device,
) -> Result<Tensor> {
    let gameplay = gameplay_rows(true);
    if rows
        .disagreement
        .iter()
        .any(|&pixel| pixel >= gameplay * FRAME_SIDE)
    {
        bail!("disagreement mask lies outside the decoded gameplay region");
    }
    disagreement_mask(&rows.disagreement, spec.physical_batch, gameplay, device)
}

/// Run both arms in lockstep on already generated rows. `load_arm` must
/// return a freshly loaded model each time it is called (bit-identical
/// initialization for both arms).
pub(crate) fn run_context_wiring_arms(
    spec: &ContextWiringSpec,
    rows: &ContextWiringRows,
    device: &Device,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
    mut on_progress: impl FnMut(&str) -> Result<()>,
) -> Result<(ContextWiringArms, ContextWiringVerdict)> {
    spec.validate()?;
    let build_arm = |name: &'static str,
                     primary_window: &[ContextTransition],
                     twin_window: &[ContextTransition]|
     -> Result<Arm> {
        build_arm(
            name,
            rows,
            (primary_window, twin_window),
            spec,
            ParameterOrdering::VarMapIteration,
            device,
            load_arm,
        )
    };
    let mut correct = build_arm("correct", &rows.primary_window, &rows.twin_window)?;
    let mut swapped = build_arm("swapped", &rows.twin_window, &rows.primary_window)?;
    let initial_parameter_sha256 = parameter_sha256(&correct.varmap)?;
    let arms_initialized_identically =
        initial_parameter_sha256 == parameter_sha256(&swapped.varmap)?;
    if !arms_initialized_identically {
        bail!("the two arms did not initialize bit-identically");
    }
    let mask = training_disagreement_mask(rows, spec, device)?;
    let mask_pixels = rows.disagreement.len() * spec.physical_batch;
    let directions = direction_rows(rows);

    let mut gates = Vec::new();
    let mut updates_run = 0usize;
    let mut early_stopped = false;
    let evaluate_both = |correct: &mut Arm,
                         swapped: &mut Arm,
                         update: usize,
                         gates: &mut Vec<CheckpointGate>,
                         on_progress: &mut dyn FnMut(&str) -> Result<()>|
     -> Result<ContextWiringDecision> {
        let correct_eval = evaluate_checkpoint(&correct.model, &directions, update, device)?;
        let swapped_eval = evaluate_checkpoint(&swapped.model, &directions, update, device)?;
        let gate = checkpoint_gate(&correct_eval, &swapped_eval, spec)?;
        on_progress(&format!(
            "checkpoint {update}: D_correct={:.6e} D_swapped={:.6e} interaction={:.6e} \
             l1_correct={:.3e} l1_swapped={:.3e} wiring={} promotion={}",
            gate.d_correct,
            gate.d_swapped,
            gate.interaction,
            gate.probability_l1_correct,
            gate.probability_l1_swapped,
            gate.wiring_gate,
            gate.promotion_gate
        ))?;
        correct.checkpoints.push(correct_eval);
        swapped.checkpoints.push(swapped_eval);
        gates.push(gate);
        context_wiring_decision(gates, &spec.checkpoint_family)
    };
    evaluate_both(&mut correct, &mut swapped, 0, &mut gates, &mut on_progress)?;
    for update in 1..=spec.max_updates {
        let correct_record = train_update(&mut correct, &mask, mask_pixels, spec, device)?;
        let swapped_record = train_update(&mut swapped, &mask, mask_pixels, spec, device)?;
        updates_run = update;
        if update == 1 || spec.checkpoint_family.contains(&update) {
            on_progress(&format!(
                "update {update}: loss correct={:.6} swapped={:.6}; pre-clip norm correct={:.4} \
                 swapped={:.4}; context grad norm correct={:.4e} swapped={:.4e}",
                correct_record.loss,
                swapped_record.loss,
                correct_record.pre_clip_gradient_norm,
                swapped_record.pre_clip_gradient_norm,
                correct_record.context_gradient_norm,
                swapped_record.context_gradient_norm
            ))?;
        }
        if spec.checkpoint_family.contains(&update) {
            let decision = evaluate_both(
                &mut correct,
                &mut swapped,
                update,
                &mut gates,
                &mut on_progress,
            )?;
            if decision.early_stop && update < spec.max_updates {
                early_stopped = true;
                on_progress(&format!(
                    "early success at checkpoint {update}; stopping both arms"
                ))?;
                break;
            }
        }
    }
    let verdict = context_wiring_verdict(gates, spec, updates_run)?;
    if early_stopped != verdict.early_stop_update.is_some() {
        bail!("early-stop bookkeeping disagrees with the verdict");
    }
    let report = |arm: Arm, contexts: [&str; 2]| ArmReport {
        name: arm.name.into(),
        row_contexts: contexts
            .iter()
            .map(|context| (*context).to_string())
            .collect(),
        updates: arm.updates,
        checkpoints: arm.checkpoints,
    };
    Ok((
        ContextWiringArms {
            correct: report(correct, ["primary<-primary_window", "twin<-twin_window"]),
            swapped: report(swapped, ["primary<-twin_window", "twin<-primary_window"]),
            initial_parameter_sha256,
            arms_initialized_identically,
            updates_run,
        },
        verdict,
    ))
}

// ---- report, lifecycle, provenance --------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleRecord {
    pub state: String,
    pub unix_seconds: u64,
    pub evidence_class: String,
    pub run_class: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfigSummary {
    pub recipe: String,
    pub world_core_v6: bool,
    pub data_contract_v6: bool,
    pub inner_steps: usize,
    pub outer_steps: usize,
    pub hidden_dim: usize,
    pub patch_size: usize,
    pub bf16_conv: bool,
    pub bf16_recurrent_core: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationRecord {
    pub population: String,
    pub population_seed: u64,
    pub pairs: usize,
    pub context_len: usize,
    pub history: LearningHistoryConfig,
    pub fingerprint: String,
    pub census: TwinMemorizationCensus,
    pub registered_fingerprint_match: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWiringTiming {
    pub population_seconds: f64,
    pub arms_seconds: f64,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuIdentity {
    pub ordinal: usize,
    pub name: String,
    pub uuid: String,
    pub memory_total_mib: String,
    pub driver_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParentEvidenceBinding {
    pub source_run_identity: String,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    pub checkpoint_relative: PathBuf,
    pub config_relative: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreflightBinding {
    pub report: PathBuf,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    pub identity_root: String,
    pub max_updates: usize,
    pub selection: ContextWiringSelection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextWiringReport {
    pub schema: String,
    pub evidence_class: String,
    /// `registered_diagnostic` or `unregistered_preflight`. Only the former
    /// can address the E2W registration; a preflight cannot satisfy E2W.
    pub run_class: String,
    pub registered: bool,
    pub public_data_read: bool,
    pub lifecycle: LifecycleRecord,
    pub provenance: LaunchProvenance,
    pub package_version: String,
    pub command: Vec<String>,
    pub device: String,
    pub device_is_cuda: bool,
    pub gpu_identity: Option<GpuIdentity>,
    pub output_root: PathBuf,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub parent_evidence: Option<ParentEvidenceBinding>,
    pub preflight: Option<PreflightBinding>,
    pub model_config: Option<ModelConfigSummary>,
    pub spec: ContextWiringSpec,
    pub population: Option<PopulationRecord>,
    pub selection: Option<ContextWiringSelection>,
    pub arms: Option<ContextWiringArms>,
    pub verdict: Option<ContextWiringVerdict>,
    pub timing: ContextWiringTiming,
    /// Domain-separated SHA-256 over checkpoint, config, binary, population
    /// fingerprint, selection hashes and the spec.
    pub identity_root: String,
    pub error: Option<String>,
}

pub(crate) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn file_sha256_hex(path: &Path) -> Result<String> {
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
    Ok(format!("{:x}", digest.finalize()))
}

/// Domain-separated SHA-256 over length-framed `(role, value)` pairs.
pub(crate) fn identity_frame_sha256(frames: &[(&str, Vec<u8>)]) -> Result<String> {
    let mut digest = Sha256::new();
    for (role, value) in frames {
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    }
    Ok(sha256_hex(digest))
}

fn identity_root(report: &ContextWiringReport) -> Result<String> {
    let mut digest = Sha256::new();
    let mut frame = |role: &str, value: &[u8]| {
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    };
    frame("domain", IDENTITY_DOMAIN.as_bytes());
    frame("checkpoint_sha256", report.checkpoint_sha256.as_bytes());
    frame("train_config_sha256", report.train_config_sha256.as_bytes());
    frame("binary_sha256", report.provenance.binary_sha256.as_bytes());
    frame(
        "source_revision",
        report.provenance.source_revision.as_bytes(),
    );
    frame("gpu_identity", &serde_json::to_vec(&report.gpu_identity)?);
    frame(
        "parent_evidence",
        &serde_json::to_vec(&report.parent_evidence)?,
    );
    frame("preflight", &serde_json::to_vec(&report.preflight)?);
    frame(
        "population_fingerprint",
        report
            .population
            .as_ref()
            .map_or("", |population| population.fingerprint.as_str())
            .as_bytes(),
    );
    if let Some(selection) = &report.selection {
        frame(
            "primary_row_sha256",
            selection.primary_row_sha256.as_bytes(),
        );
        frame("twin_row_sha256", selection.twin_row_sha256.as_bytes());
    }
    frame("spec", &serde_json::to_vec(&report.spec)?);
    Ok(sha256_hex(digest))
}

pub(crate) fn write_lifecycle(root: &Path, lifecycle: &LifecycleRecord) -> Result<()> {
    write_json_report(&root.join(LIFECYCLE_FILE), lifecycle)
}

pub(crate) fn append_command_log(root: &Path, tag: &str, line: &str) -> Result<()> {
    let path = root.join(COMMAND_LOG_FILE);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    writeln!(file, "{} {line}", unix_seconds())
        .with_context(|| format!("write {}", path.display()))?;
    eprintln!("[{tag}] {line}");
    Ok(())
}

fn collect_regular_files(root: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    fn visit(root: &Path, directory: &Path, files: &mut Vec<(PathBuf, PathBuf)>) -> Result<()> {
        for entry in
            fs::read_dir(directory).with_context(|| format!("list {}", directory.display()))?
        {
            let path = entry?.path();
            let metadata =
                fs::symlink_metadata(&path).with_context(|| format!("stat {}", path.display()))?;
            if metadata.file_type().is_symlink() {
                bail!(
                    "evidence roots may not contain symlinks: {}",
                    path.display()
                );
            }
            if metadata.is_dir() {
                visit(root, &path, files)?;
            } else if metadata.is_file() {
                files.push((
                    path.strip_prefix(root)
                        .with_context(|| format!("relativize {}", path.display()))?
                        .to_path_buf(),
                    path,
                ));
            } else {
                bail!("unsupported evidence file type: {}", path.display());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(files)
}

pub(crate) fn parse_manifest(manifest: &str) -> Result<BTreeMap<PathBuf, String>> {
    let mut entries = BTreeMap::new();
    for (index, line) in manifest.lines().enumerate() {
        let (digest, relative) = line
            .split_once("  ")
            .ok_or_else(|| anyhow::anyhow!("malformed manifest line {}", index + 1))?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("invalid sha256 on manifest line {}", index + 1);
        }
        let relative = relative.strip_prefix("./").unwrap_or(relative);
        let path = PathBuf::from(relative);
        if path.as_os_str().is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            bail!("unsafe manifest path on line {}: {relative}", index + 1);
        }
        if entries.insert(path, digest.to_ascii_lowercase()).is_some() {
            bail!("duplicate manifest path on line {}", index + 1);
        }
    }
    if entries.is_empty() {
        bail!("evidence manifest is empty");
    }
    Ok(entries)
}

pub(crate) fn verify_manifest(root: &Path, manifest_path: &Path) -> Result<String> {
    let manifest =
        fs::read(manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&manifest));
    let text = std::str::from_utf8(&manifest)
        .with_context(|| format!("decode {} as UTF-8", manifest_path.display()))?;
    let registered = parse_manifest(text)?;
    let actual = collect_regular_files(root)?;
    let actual_paths = actual
        .iter()
        .map(|(relative, _)| relative.clone())
        .collect::<BTreeSet<_>>();
    let registered_paths = registered.keys().cloned().collect::<BTreeSet<_>>();
    if actual_paths != registered_paths {
        bail!(
            "manifest file set differs from {} (actual {}, registered {})",
            manifest_path.display(),
            actual_paths.len(),
            registered_paths.len()
        );
    }
    for (relative, path) in actual {
        let actual_digest = file_sha256_hex(&path)?;
        if registered.get(&relative) != Some(&actual_digest) {
            bail!(
                "manifest digest mismatch for {} (actual {actual_digest}, registered {:?})",
                path.display(),
                registered.get(&relative)
            );
        }
    }
    Ok(digest)
}

pub(crate) fn verify_manifest_sidecar(manifest_path: &Path, expected_digest: &str) -> Result<()> {
    let sidecar = PathBuf::from(format!("{}.sha256", manifest_path.display()));
    let text =
        fs::read_to_string(&sidecar).with_context(|| format!("read {}", sidecar.display()))?;
    let (digest, named_path) = text
        .trim()
        .split_once("  ")
        .ok_or_else(|| anyhow::anyhow!("malformed manifest sidecar {}", sidecar.display()))?;
    if digest != expected_digest {
        bail!("manifest sidecar digest {digest} does not match verified {expected_digest}");
    }
    let named_file = Path::new(named_path)
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("manifest sidecar names no file"))?;
    if Some(named_file) != manifest_path.file_name() {
        bail!("manifest sidecar names a different manifest: {named_path}");
    }
    Ok(())
}

pub(crate) fn external_manifest_paths(root: &Path) -> Result<(PathBuf, PathBuf)> {
    let root_name = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("output root has no name"))?;
    let manifest = root.with_file_name(format!("{root_name}.files.sha256"));
    let sidecar = root.with_file_name(format!("{root_name}.files.sha256.sha256"));
    Ok((manifest, sidecar))
}

/// Write `<root>.files.sha256` next to (outside) the run root over every
/// finalized regular file, fsync it and its sidecar, then re-read and verify
/// the complete root before returning the manifest digest.
pub(crate) fn finalize_root_manifest(root: &Path) -> Result<String> {
    let entries = collect_regular_files(root)?;
    let mut manifest = String::new();
    for (relative, path) in entries {
        manifest.push_str(&format!(
            "{}  ./{}\n",
            file_sha256_hex(&path)?,
            relative.display()
        ));
    }
    let (manifest_path, sidecar) = external_manifest_paths(root)?;
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)
        .with_context(|| format!("create fresh {}", manifest_path.display()))?;
    manifest_file
        .write_all(manifest.as_bytes())
        .with_context(|| format!("write {}", manifest_path.display()))?;
    manifest_file
        .sync_all()
        .with_context(|| format!("sync {}", manifest_path.display()))?;
    let digest = format!("{:x}", Sha256::digest(manifest.as_bytes()));
    let manifest_name = manifest_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("manifest path has no name"))?
        .to_string_lossy();
    let mut sidecar_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&sidecar)
        .with_context(|| format!("create fresh {}", sidecar.display()))?;
    sidecar_file
        .write_all(format!("{digest}  {manifest_name}\n").as_bytes())
        .with_context(|| format!("write {}", sidecar.display()))?;
    sidecar_file
        .sync_all()
        .with_context(|| format!("sync {}", sidecar.display()))?;
    File::open(root)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync root directory {}", root.display()))?;
    if let Some(parent) = manifest_path.parent() {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("sync manifest directory {}", parent.display()))?;
    }
    let verified_digest = verify_manifest(root, &manifest_path)?;
    if verified_digest != digest {
        bail!("manifest changed while finalizing: {digest} -> {verified_digest}");
    }
    verify_manifest_sidecar(&manifest_path, &digest)?;
    Ok(digest)
}

pub(crate) fn bind_parent_evidence(args: &P2ContextWiringArgs) -> Result<ParentEvidenceBinding> {
    let root = fs::canonicalize(&args.parent_root)
        .with_context(|| format!("canonicalize {}", args.parent_root.display()))?;
    if root.file_name().and_then(|name| name.to_str()) != Some(REGISTERED_PARENT_RUN_ID) {
        bail!("parent root is not the registered E2 run {REGISTERED_PARENT_RUN_ID}");
    }
    let manifest = fs::canonicalize(&args.parent_manifest)
        .with_context(|| format!("canonicalize {}", args.parent_manifest.display()))?;
    let digest = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &digest)?;
    if digest != REGISTERED_PARENT_MANIFEST_SHA256 {
        bail!(
            "parent manifest sha256 {digest} is not the registered {REGISTERED_PARENT_MANIFEST_SHA256}"
        );
    }
    let checkpoint_relative = PathBuf::from(REGISTERED_PARENT_CHECKPOINT_RELATIVE);
    let config_relative = PathBuf::from(REGISTERED_PARENT_CONFIG_RELATIVE);
    let checkpoint = fs::canonicalize(&args.checkpoint)
        .with_context(|| format!("canonicalize {}", args.checkpoint.display()))?;
    let config = fs::canonicalize(&args.train_config)
        .with_context(|| format!("canonicalize {}", args.train_config.display()))?;
    if checkpoint != fs::canonicalize(root.join(&checkpoint_relative))? {
        bail!("checkpoint is not the registered file inside the sealed parent run");
    }
    if config != fs::canonicalize(root.join(&config_relative))? {
        bail!("train config is not the registered file inside the sealed parent run");
    }
    let entries = parse_manifest(&fs::read_to_string(&manifest)?)?;
    if entries.get(&checkpoint_relative).map(String::as_str) != Some(REGISTERED_CHECKPOINT_SHA256)
        || entries.get(&config_relative).map(String::as_str) != Some(REGISTERED_TRAIN_CONFIG_SHA256)
    {
        bail!("parent manifest does not bind the registered checkpoint and config hashes");
    }
    Ok(ParentEvidenceBinding {
        source_run_identity: REGISTERED_PARENT_RUN_ID.into(),
        root,
        manifest,
        manifest_sha256: digest,
        checkpoint_relative,
        config_relative,
    })
}

fn bind_preflight(
    path: &Path,
    current: &ContextWiringReport,
    selection: &ContextWiringSelection,
) -> Result<PreflightBinding> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize preflight report {}", path.display()))?;
    let bytes = fs::read(&report_path)
        .with_context(|| format!("read preflight report {}", report_path.display()))?;
    let preflight: ContextWiringReport =
        serde_json::from_slice(&bytes).context("parse E2W preflight report")?;
    let root = fs::canonicalize(&preflight.output_root).with_context(|| {
        format!(
            "canonicalize preflight root {}",
            preflight.output_root.display()
        )
    })?;
    if report_path != fs::canonicalize(root.join(REPORT_FILE))? {
        bail!("--preflight-report is not the report inside its claimed output root");
    }
    let (manifest, _) = external_manifest_paths(&root)?;
    let manifest = fs::canonicalize(&manifest)
        .with_context(|| format!("canonicalize preflight manifest {}", manifest.display()))?;
    let manifest_sha256 = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &manifest_sha256)?;
    if preflight.schema != CONTEXT_WIRING_SCHEMA
        || preflight.evidence_class != EVIDENCE_CLASS
        || preflight.run_class != "unregistered_preflight"
        || preflight.registered
        || preflight.public_data_read
        || preflight.lifecycle.state != LIFECYCLE_COMPLETE
        || preflight.lifecycle.evidence_class != EVIDENCE_CLASS
        || preflight.lifecycle.run_class != "unregistered_preflight"
        || preflight.error.is_some()
        || !preflight.device_is_cuda
        || preflight.preflight.is_some()
        || preflight.package_version != current.package_version
    {
        bail!("preflight report is not a completed, clean, CUDA E2W preflight");
    }
    let expected_spec = ContextWiringSpec::with_budget(REGISTERED_PAIRS, 8);
    if preflight.spec != expected_spec {
        bail!("registered E2W requires the exact 256-pair, 8-update preflight");
    }
    registered_provenance_guard(&preflight.provenance)?;
    if !same_build_identity(&preflight.provenance, &current.provenance) {
        bail!("preflight and registered run do not use the same build identity");
    }
    if preflight.checkpoint_sha256 != REGISTERED_CHECKPOINT_SHA256
        || preflight.train_config_sha256 != REGISTERED_TRAIN_CONFIG_SHA256
        || preflight.checkpoint_sha256 != current.checkpoint_sha256
        || preflight.train_config_sha256 != current.train_config_sha256
    {
        bail!("preflight checkpoint/config identity differs from the registered inputs");
    }
    let population = preflight
        .population
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("preflight has no population record"))?;
    if population.fingerprint != REGISTERED_POPULATION_FINGERPRINT
        || population.pairs != 256
        || population.registered_fingerprint_match != Some(true)
    {
        bail!("preflight did not scan the complete registered population");
    }
    registered_census_matches(&population.census)?;
    if preflight.selection.as_ref() != Some(selection) {
        bail!("preflight selected a different registered row");
    }
    if preflight.parent_evidence != current.parent_evidence {
        bail!("preflight and registered run do not bind the same sealed parent evidence");
    }
    if preflight.gpu_identity.is_none() || preflight.gpu_identity != current.gpu_identity {
        bail!("preflight and registered run do not bind the same GPU identity");
    }
    let arms = preflight
        .arms
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("preflight has no arm record"))?;
    let arm_complete = |arm: &ArmReport| {
        arm.updates.iter().map(|record| record.update).eq(1..=8)
            && arm
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.update)
                .eq([0, 8])
    };
    if preflight.verdict.as_ref().map(|verdict| {
        (
            verdict.updates_run,
            verdict.evaluated_checkpoints.as_slice(),
        ) == (8, [0, 8].as_slice())
    }) != Some(true)
        || arms.updates_run != 8
        || !arms.arms_initialized_identically
        || !arm_complete(&arms.correct)
        || !arm_complete(&arms.swapped)
    {
        bail!("preflight did not complete 8 updates in both arms");
    }
    if identity_root(&preflight)? != preflight.identity_root {
        bail!("preflight identity root does not match its report fields");
    }
    Ok(PreflightBinding {
        report: report_path,
        root,
        manifest,
        manifest_sha256,
        identity_root: preflight.identity_root,
        max_updates: preflight.spec.max_updates,
        selection: selection.clone(),
    })
}

// ---- CLI ---------------------------------------------------------------------

/// `p2-context-wiring` — E2W two-row context-wiring overfit diagnostic.
#[derive(Debug, Clone, Args)]
pub struct P2ContextWiringArgs {
    /// Initial checkpoint (`ema.safetensors` of the sealed E2 step-4096 bundle).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// The checkpoint bundle's own `config.json`.
    #[arg(long)]
    pub train_config: PathBuf,
    /// Sealed E2 source-run root that owns the checkpoint and config.
    #[arg(long)]
    pub parent_root: PathBuf,
    /// External point-in-time manifest sealing `parent_root`.
    #[arg(long)]
    pub parent_manifest: PathBuf,
    /// Completed exact-binary CUDA preflight report. Required only for a
    /// registered run; the preflight itself omits this argument.
    #[arg(long)]
    pub preflight_report: Option<PathBuf>,
    /// Never-reused run root; must not exist.
    #[arg(long)]
    pub output_root: PathBuf,
    #[arg(long, default_value = "cuda")]
    pub device: String,
    /// Fail closed unless every registered fixed input holds (checkpoint and
    /// config hashes, 256-pair fingerprint and census, 256 updates, CUDA,
    /// clean and pushed known build provenance). Without it the run is an
    /// unregistered preflight that cannot satisfy the registration.
    #[arg(long)]
    pub registered: bool,
    /// Updates per arm; must belong to the fixed family 8/16/32/64/128/256.
    #[arg(long, default_value_t = CONTEXT_WIRING_MAX_UPDATES)]
    pub max_updates: usize,
    /// Twin pairs generated for the model-free scan (registered: 256).
    #[arg(long, default_value_t = REGISTERED_PAIRS)]
    pub pairs: usize,
}

pub(crate) fn registered_provenance_guard(provenance: &LaunchProvenance) -> Result<()> {
    if !provenance.source_revision_known() {
        bail!("registered run requires an embedded build source revision");
    }
    if provenance.source_dirty != Some(false) {
        bail!(
            "registered run requires a clean build source (source_dirty={:?})",
            provenance.source_dirty
        );
    }
    if provenance.source_pushed != Some(true) {
        bail!(
            "registered run requires a pushed build source (source_pushed={:?})",
            provenance.source_pushed
        );
    }
    if provenance.binary_sha256 == crate::p2::evidence::UNKNOWN_PROVENANCE {
        bail!("registered run requires a hashable binary");
    }
    if provenance.candle_graph_revision != REGISTERED_CANDLE_GRAPH_REVISION
        || provenance.candle_graph_dirty != Some(false)
        || provenance.candle_graph_pushed != Some(true)
    {
        bail!(
            "registered run requires candle_graph revision {REGISTERED_CANDLE_GRAPH_REVISION}, clean and pushed"
        );
    }
    if provenance.runtime_checkout.revision != provenance.source_revision
        || provenance.runtime_checkout.dirty != Some(false)
    {
        bail!("registered run requires a clean runtime checkout at the embedded source revision");
    }
    if provenance.build_command != REGISTERED_BUILD_COMMAND {
        bail!(
            "registered run requires build command {REGISTERED_BUILD_COMMAND:?}, got {:?}",
            provenance.build_command
        );
    }
    if provenance.cargo_profile != "release" {
        bail!("registered run requires Cargo release profile");
    }
    if !provenance
        .cargo_features
        .iter()
        .any(|feature| feature == "cudnn")
    {
        bail!("registered run requires the cudnn Cargo feature");
    }
    if provenance.cargo_target == crate::p2::evidence::UNKNOWN_PROVENANCE {
        bail!("registered run requires a known Cargo target");
    }
    Ok(())
}

pub(crate) fn same_build_identity(left: &LaunchProvenance, right: &LaunchProvenance) -> bool {
    left.source_revision == right.source_revision
        && left.source_revision_origin == right.source_revision_origin
        && left.source_dirty == right.source_dirty
        && left.source_pushed == right.source_pushed
        && left.build_command == right.build_command
        && left.cargo_features == right.cargo_features
        && left.cargo_profile == right.cargo_profile
        && left.cargo_target == right.cargo_target
        && left.binary_sha256 == right.binary_sha256
        && left.candle_graph_revision == right.candle_graph_revision
        && left.candle_graph_dirty == right.candle_graph_dirty
        && left.candle_graph_pushed == right.candle_graph_pushed
}

fn cuda_ordinal(device: &str) -> Result<usize> {
    match device.trim() {
        "cuda" => Ok(0),
        value if value.starts_with("cuda:") => value[5..]
            .parse::<usize>()
            .with_context(|| format!("parse CUDA ordinal from {value:?}")),
        value => bail!("GPU identity requested for non-CUDA device {value:?}"),
    }
}

pub(crate) fn query_gpu_identity(device: &str) -> Result<GpuIdentity> {
    let ordinal = cuda_ordinal(device)?;
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            &ordinal.to_string(),
        ])
        .output()
        .context("run nvidia-smi for registered GPU identity")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi GPU identity query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8(output.stdout).context("decode nvidia-smi output")?;
    let lines = stdout.lines().collect::<Vec<_>>();
    if lines.len() != 1 {
        bail!("nvidia-smi returned {} GPU identity rows", lines.len());
    }
    let fields = lines[0]
        .split(',')
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if fields.len() != 4 || fields.iter().any(String::is_empty) {
        bail!("malformed nvidia-smi GPU identity row: {:?}", lines[0]);
    }
    Ok(GpuIdentity {
        ordinal,
        name: fields[0].clone(),
        uuid: fields[1].clone(),
        memory_total_mib: fields[2].clone(),
        driver_version: fields[3].clone(),
    })
}

/// Verified fixed inputs shared by the E2W and E2C diagnostics.
pub(crate) struct DiagnosticInputs {
    pub(crate) train_cfg: TrainConfig,
    pub(crate) model_config: ModelConfigSummary,
    pub(crate) checkpoint_sha256: String,
    pub(crate) train_config_sha256: String,
    pub(crate) parent_evidence: ParentEvidenceBinding,
}

/// Read, validate and copy the v6 2x2 F32 config, hash the checkpoint,
/// optionally require the registered hashes, and bind the sealed E2 parent.
pub(crate) fn verify_diagnostic_inputs(
    args: &P2ContextWiringArgs,
    root: &Path,
    require_registered_hashes: bool,
) -> Result<DiagnosticInputs> {
    let config_bytes = fs::read(&args.train_config)
        .with_context(|| format!("read {}", args.train_config.display()))?;
    let train_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    fs::write(root.join(TRAIN_CONFIG_COPY_FILE), &config_bytes)?;
    let train_cfg: TrainConfig =
        serde_json::from_slice(&config_bytes).context("parse TrainConfig")?;
    train_cfg.validate()?;
    ensure_v6_2x2_f32_config(&train_cfg)?;
    let model_config = train_cfg.model_config();
    let model_config = ModelConfigSummary {
        recipe: format!("{:?}", train_cfg.recipe),
        world_core_v6: train_cfg.world_core_v6,
        data_contract_v6: train_cfg.data_contract_v6,
        inner_steps: model_config.inner_steps,
        outer_steps: model_config.outer_steps,
        hidden_dim: model_config.hidden_dim,
        patch_size: model_config.patch_size,
        bf16_conv: model_config.bf16_conv,
        bf16_recurrent_core: model_config.bf16_recurrent_core,
    };
    let checkpoint_sha256 = file_sha256_hex(&args.checkpoint)?;
    if require_registered_hashes {
        if checkpoint_sha256 != REGISTERED_CHECKPOINT_SHA256 {
            bail!(
                "checkpoint sha256 {checkpoint_sha256} is not the registered {REGISTERED_CHECKPOINT_SHA256}"
            );
        }
        if train_config_sha256 != REGISTERED_TRAIN_CONFIG_SHA256 {
            bail!(
                "train config sha256 {train_config_sha256} is not the registered {REGISTERED_TRAIN_CONFIG_SHA256}"
            );
        }
    }
    let parent_evidence = bind_parent_evidence(args)?;
    Ok(DiagnosticInputs {
        train_cfg,
        model_config,
        checkpoint_sha256,
        train_config_sha256,
        parent_evidence,
    })
}

/// Resolved device plus, for CUDA, the GPU identity and the exclusive
/// session lock held for the lifetime of this value.
pub(crate) struct DiagnosticDevice {
    pub(crate) device: Device,
    pub(crate) gpu_identity: Option<GpuIdentity>,
    _gpu: Option<crate::gpu_lock::GpuSessionGuard>,
}

pub(crate) fn open_diagnostic_device(spec: &str, root: &Path) -> Result<DiagnosticDevice> {
    let cuda = spec.trim().starts_with("cuda");
    let gpu_identity = cuda.then(|| query_gpu_identity(spec)).transpose()?;
    let _gpu = cuda
        .then(|| crate::gpu_lock::GpuSessionGuard::acquire(root))
        .transpose()?;
    let device = resolve_device(spec)?;
    if gpu_identity.is_some() != device.is_cuda() {
        bail!("device resolution and GPU identity disagree");
    }
    Ok(DiagnosticDevice {
        device,
        gpu_identity,
        _gpu,
    })
}

/// Generate the registered twin population, census and fingerprint;
/// `require_registered` fails closed on fingerprint or census drift.
pub(crate) fn generate_population(
    spec: &ContextWiringSpec,
    require_registered: bool,
) -> Result<(Vec<AugmentedTwinPair>, PopulationRecord)> {
    let twin_spec = spec.twin_spec();
    let pairs = twin_memorization_population(&twin_spec)?;
    let census = twin_memorization_census(&pairs, twin_spec.context_len);
    validate_twin_memorization_census(&twin_spec, &census)?;
    let fingerprint = learning_history_population_fingerprint(
        pairs.iter().flat_map(|pair| [&pair.primary, &pair.twin]),
    );
    let registered_fingerprint_match =
        (spec.pairs == REGISTERED_PAIRS).then(|| fingerprint == REGISTERED_POPULATION_FINGERPRINT);
    if require_registered {
        if fingerprint != REGISTERED_POPULATION_FINGERPRINT {
            bail!("population fingerprint {fingerprint} is not the registered {REGISTERED_POPULATION_FINGERPRINT}");
        }
        registered_census_matches(&census)?;
    }
    let record = PopulationRecord {
        population: "twin_learning_histories/unseen_seed_7x7".into(),
        population_seed: twin_spec.population_seed,
        pairs: pairs.len(),
        context_len: twin_spec.context_len,
        history: twin_spec.history,
        fingerprint,
        census,
        registered_fingerprint_match,
    };
    Ok((pairs, record))
}

/// Fail closed if the checkpoint or the running binary changed during a run.
pub(crate) fn verify_no_input_drift(
    checkpoint: &Path,
    checkpoint_sha256: &str,
    provenance: &LaunchProvenance,
) -> Result<()> {
    let checkpoint_after = file_sha256_hex(checkpoint)?;
    if checkpoint_after != checkpoint_sha256 {
        bail!("checkpoint changed during the run: {checkpoint_sha256} -> {checkpoint_after}");
    }
    let binary_after = file_sha256_hex(&provenance.binary_path)
        .unwrap_or_else(|_| crate::p2::evidence::UNKNOWN_PROVENANCE.into());
    let binary_after = format!("sha256:{binary_after}");
    if binary_after != provenance.binary_sha256 {
        bail!(
            "binary changed during the run: {} -> {binary_after}",
            provenance.binary_sha256
        );
    }
    Ok(())
}

/// Create the never-reused run root (no root, external manifest or sidecar
/// may exist), write the running lifecycle and log the command line.
pub(crate) fn open_run_root(
    root: &Path,
    lifecycle: &LifecycleRecord,
    tag: &str,
) -> Result<Vec<String>> {
    if let Some(parent) = root
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let (external_manifest, external_sidecar) = external_manifest_paths(root)?;
    if external_manifest.exists() || external_sidecar.exists() {
        bail!(
            "fresh output identity already has an external manifest or sidecar: {} / {}",
            external_manifest.display(),
            external_sidecar.display()
        );
    }
    fs::create_dir(root).with_context(|| {
        format!(
            "create never-reused output root {} (it must not already exist)",
            root.display()
        )
    })?;
    write_lifecycle(root, lifecycle)?;
    let command = std::env::args().collect::<Vec<_>>();
    append_command_log(
        root,
        tag,
        &format!("start {}", serde_json::to_string(&command)?),
    )?;
    Ok(command)
}

/// Write the final report and lifecycle, log the end state, then seal the
/// root with a freshly written and re-verified external manifest.
pub(crate) fn seal_run_root(
    root: &Path,
    tag: &str,
    report: &impl Serialize,
    lifecycle: &LifecycleRecord,
) -> Result<String> {
    write_json_report(&root.join(REPORT_FILE), report)?;
    write_lifecycle(root, lifecycle)?;
    append_command_log(
        root,
        tag,
        &format!("end state={} note={}", lifecycle.state, lifecycle.note),
    )?;
    let manifest_digest = finalize_root_manifest(root)?;
    eprintln!(
        "[{tag}] finalized {} (external manifest sha256 {manifest_digest})",
        root.display()
    );
    Ok(manifest_digest)
}

fn run_inner(
    args: &P2ContextWiringArgs,
    report: &mut ContextWiringReport,
    root: &Path,
    started: Instant,
) -> Result<()> {
    let exact_preflight =
        !args.registered && report.spec == ContextWiringSpec::with_budget(REGISTERED_PAIRS, 8);
    if args.registered {
        registered_provenance_guard(&report.provenance)?;
        if !report.spec.is_registered_contract() {
            bail!(
                "registered run requires --max-updates {CONTEXT_WIRING_MAX_UPDATES} and --pairs {REGISTERED_PAIRS}"
            );
        }
        if !args.device.trim().starts_with("cuda") {
            bail!("registered run requires a CUDA device");
        }
        if args.preflight_report.is_none() {
            bail!("registered run requires --preflight-report");
        }
    } else if args.preflight_report.is_some() {
        bail!("--preflight-report is valid only with --registered");
    }
    if exact_preflight {
        registered_provenance_guard(&report.provenance)?;
        if !args.device.trim().starts_with("cuda") {
            bail!("the bindable 256-pair, 8-update preflight requires CUDA");
        }
    }
    let inputs = verify_diagnostic_inputs(args, root, args.registered || exact_preflight)?;
    report.train_config_sha256 = inputs.train_config_sha256;
    report.checkpoint_sha256 = inputs.checkpoint_sha256;
    report.model_config = Some(inputs.model_config);
    report.parent_evidence = Some(inputs.parent_evidence);
    let train_cfg = inputs.train_cfg;
    append_command_log(
        root,
        COMMAND_TAG,
        &format!(
            "inputs verified: checkpoint={} config={} class={}",
            report.checkpoint_sha256, report.train_config_sha256, report.run_class
        ),
    )?;

    let diagnostic_device = open_diagnostic_device(&args.device, root)?;
    report.gpu_identity = diagnostic_device.gpu_identity.clone();
    let device = diagnostic_device.device.clone();
    report.device_is_cuda = device.is_cuda();

    let population_started = Instant::now();
    let (pairs, population) =
        generate_population(&report.spec, args.registered || exact_preflight)?;
    report.timing.population_seconds = population_started.elapsed().as_secs_f64();
    append_command_log(
        root,
        COMMAND_TAG,
        &format!(
            "population: pairs={} fingerprint={} outcome_changing_rows={} evidence_rows={}",
            pairs.len(),
            population.fingerprint,
            population.census.outcome_changing_rows,
            population.census.scorable_rows_with_evidence_in_window
        ),
    )?;
    let context_len = population.context_len;
    report.population = Some(population);

    let gameplay_pixels = gameplay_rows(train_cfg.world_core_v6) * FRAME_SIDE;
    let (selection, rows) = select_context_wiring_rows(&pairs, context_len, gameplay_pixels)?;
    append_command_log(
        root,
        COMMAND_TAG,
        &format!(
            "selection: meta_episode_id={} position={} disagreement_pixels={} primary_row={} twin_row={}",
            selection.meta_episode_id,
            selection.position,
            selection.target_disagreement_pixels,
            selection.primary_row_sha256,
            selection.twin_row_sha256
        ),
    )?;
    report.selection = Some(selection.clone());
    if let Some(preflight_report) = args.preflight_report.as_deref() {
        report.preflight = Some(bind_preflight(preflight_report, report, &selection)?);
    }
    report.identity_root = identity_root(report)?;
    drop(pairs);

    let arms_started = Instant::now();
    let checkpoint = args.checkpoint.clone();
    let arm_device = device.clone();
    let load_arm = move || load_model(&train_cfg, &checkpoint, &arm_device);
    let (arms, verdict) =
        run_context_wiring_arms(&report.spec, &rows, &device, &load_arm, |line| {
            append_command_log(root, COMMAND_TAG, line)
        })?;
    report.timing.arms_seconds = arms_started.elapsed().as_secs_f64();
    report.arms = Some(arms);
    report.verdict = Some(verdict);

    verify_no_input_drift(
        &args.checkpoint,
        &report.checkpoint_sha256,
        &report.provenance,
    )?;
    drop(diagnostic_device);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    Ok(())
}

pub fn run_p2_context_wiring(args: P2ContextWiringArgs) -> Result<()> {
    let started = Instant::now();
    let spec = ContextWiringSpec::with_budget(args.pairs, args.max_updates);
    spec.validate()?;
    let run_class = if args.registered {
        RUN_CLASS_REGISTERED
    } else {
        RUN_CLASS_PREFLIGHT
    };
    let root = &args.output_root;
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        note: "E2W two-row context-wiring diagnostic in progress".into(),
    };
    let command = open_run_root(root, &lifecycle, COMMAND_TAG)?;
    let mut report = ContextWiringReport {
        schema: CONTEXT_WIRING_SCHEMA.into(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        registered: args.registered,
        public_data_read: false,
        lifecycle,
        provenance: launch_provenance().clone(),
        package_version: env!("CARGO_PKG_VERSION").into(),
        command,
        device: args.device.clone(),
        device_is_cuda: false,
        gpu_identity: None,
        output_root: root.clone(),
        checkpoint: args.checkpoint.clone(),
        checkpoint_sha256: String::new(),
        train_config: args.train_config.clone(),
        train_config_sha256: String::new(),
        parent_evidence: None,
        preflight: None,
        model_config: None,
        spec,
        population: None,
        selection: None,
        arms: None,
        verdict: None,
        timing: ContextWiringTiming {
            population_seconds: 0.0,
            arms_seconds: 0.0,
            wall_seconds: 0.0,
        },
        identity_root: String::new(),
        error: None,
    };
    let outcome = run_inner(&args, &mut report, root, started);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    report.lifecycle = match &outcome {
        Ok(()) => LifecycleRecord {
            state: LIFECYCLE_COMPLETE.into(),
            unix_seconds: unix_seconds(),
            evidence_class: EVIDENCE_CLASS.into(),
            run_class: run_class.into(),
            note: match &report.verdict {
                Some(verdict) => format!(
                    "{}; {}",
                    verdict.outcome,
                    if args.registered {
                        "registered E2W diagnostic (implementation_smoke; cannot promote a model or unblock E3)"
                    } else {
                        "unregistered preflight; cannot satisfy E2W"
                    }
                ),
                None => "completed without a verdict".into(),
            },
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
    if report.identity_root.is_empty() {
        report.identity_root = identity_root(&report)?;
    }
    let lifecycle = report.lifecycle.clone();
    seal_run_root(root, COMMAND_TAG, &report, &lifecycle)?;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::reinit_varmap_deterministic;
    use candle_nn::VarBuilder;

    fn tiny_v6_model(device: &Device, seed: u64) -> Result<(WorldModel, VarMap)> {
        let mut train_cfg = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            ..TrainConfig::default()
        };
        train_cfg.apply_foundation_v2_recipe();
        train_cfg.hidden_dim = 8;
        train_cfg.action_dim = 4;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_cfg.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, device),
        )?;
        reinit_varmap_deterministic(&varmap, seed)?;
        Ok((model, varmap))
    }

    fn population(pairs: usize) -> Result<Vec<AugmentedTwinPair>> {
        twin_memorization_population(&ContextWiringSpec::with_budget(pairs, 8).twin_spec())
    }

    fn evaluation(
        update: usize,
        d: f64,
        l1: f64,
        disagreement: usize,
        promotion: bool,
    ) -> CheckpointEvaluation {
        CheckpointEvaluation {
            update,
            directions: Vec::new(),
            d,
            probability_l1: l1,
            raw_argmax_disagreement_pixels: disagreement,
            mixed_k0_invariant_pass: true,
            promotion_gate: promotion,
        }
    }

    fn gate(update: usize, wiring: bool, promotion: bool) -> CheckpointGate {
        CheckpointGate {
            update,
            d_correct: 0.0,
            d_swapped: 0.0,
            interaction: 0.0,
            probability_l1_correct: 0.0,
            probability_l1_swapped: 0.0,
            raw_argmax_disagreement_correct: 0,
            raw_argmax_disagreement_swapped: 0,
            mixed_k0_invariant_pass: true,
            wiring_gate: wiring,
            promotion_gate: promotion,
        }
    }

    fn valid_registered_provenance() -> LaunchProvenance {
        let mut provenance = LaunchProvenance::unknown(Path::new("/test/tofy"));
        provenance.source_revision = "8fd1cea5".repeat(5);
        provenance.source_revision_origin = "embedded-build:git".into();
        provenance.source_dirty = Some(false);
        provenance.source_pushed = Some(true);
        provenance.build_command = REGISTERED_BUILD_COMMAND.into();
        provenance.cargo_features = vec!["cudnn".into()];
        provenance.cargo_profile = "release".into();
        provenance.cargo_target = "x86_64-unknown-linux-gnu".into();
        provenance.binary_sha256 = "sha256:test-binary".into();
        provenance.candle_graph_revision = REGISTERED_CANDLE_GRAPH_REVISION.into();
        provenance.candle_graph_dirty = Some(false);
        provenance.candle_graph_pushed = Some(true);
        provenance.runtime_checkout.revision = provenance.source_revision.clone();
        provenance.runtime_checkout.dirty = Some(false);
        provenance
    }

    #[test]
    fn registered_spec_is_the_preregistered_contract() -> Result<()> {
        let spec = ContextWiringSpec::registered();
        spec.validate()?;
        assert!(spec.is_registered_contract());
        assert_eq!(spec.pairs, 256);
        assert_eq!(spec.population_seed, 1_000_002);
        assert_eq!(spec.context_len, 16);
        assert_eq!(spec.history, LearningHistoryConfig::training());
        assert_eq!(spec.max_updates, 256);
        assert_eq!(spec.checkpoint_family, vec![0, 8, 16, 32, 64, 128, 256]);
        assert_eq!(
            (
                spec.learning_rate,
                spec.beta1,
                spec.beta2,
                spec.epsilon,
                spec.weight_decay
            ),
            (1e-3, 0.9, 0.999, 1e-8, 0.0)
        );
        assert_eq!(spec.gradient_clip, 1.0);
        assert_eq!((spec.physical_batch, spec.grad_accum), (2, 1));
        let preflight = ContextWiringSpec::with_budget(256, 32);
        preflight.validate()?;
        assert!(!preflight.is_registered_contract());
        assert_eq!(preflight.checkpoint_family, vec![0, 8, 16, 32]);
        assert!(ContextWiringSpec::with_budget(256, 12).validate().is_err());
        assert!(ContextWiringSpec::with_budget(256, 512).validate().is_err());
        let mut tampered = ContextWiringSpec::registered();
        tampered.physical_batch = 4;
        assert!(tampered.validate().is_err());
        Ok(())
    }

    #[test]
    fn config_guard_accepts_only_v6_2x2_f32() {
        let mut cfg = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            ..TrainConfig::default()
        };
        cfg.apply_foundation_v2_recipe();
        assert!(ensure_v6_2x2_f32_config(&cfg).is_ok());
        let mut depth3 = cfg.clone();
        depth3.v6_recursion_steps = 3;
        depth3.apply_foundation_v2_recipe();
        assert!(ensure_v6_2x2_f32_config(&depth3).is_err());
        let mut bf16 = cfg.clone();
        bf16.bf16_recurrent_core = true;
        assert!(ensure_v6_2x2_f32_config(&bf16).is_err());
        let mut v5 = TrainConfig::default();
        v5.apply_foundation_v2_recipe();
        assert!(ensure_v6_2x2_f32_config(&v5).is_err());
        let mut legacy = cfg.clone();
        legacy.recipe = TrainingRecipe::LegacyExperimental;
        assert!(ensure_v6_2x2_f32_config(&legacy).is_err());
    }

    #[test]
    fn selection_is_model_free_deterministic_and_satisfies_invariants() -> Result<()> {
        let pairs = population(8)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows(&pairs, 16, gameplay)?;
        let pair = &pairs[selection.pair_ordinal];
        assert_eq!(selection.meta_episode_id, pair.primary.meta_episode_id);
        assert!(selection.position >= 16);
        assert_eq!(selection.window_start, selection.position - 16);
        assert!(!pair.divergence.single_frame_rule_identifiable);
        // (a) targets differ, (b) outcome-changing at p, (c) earlier evidence.
        assert_ne!(rows.primary.next, rows.twin.next);
        assert!(pair
            .divergence
            .outcome_changing_positions
            .contains(&selection.position));
        assert!(!selection.outcome_changing_positions_in_window.is_empty());
        assert!(selection
            .outcome_changing_positions_in_window
            .iter()
            .all(|&changed| changed >= selection.window_start && changed < selection.position));
        // Earliest qualifying position in the earliest qualifying pair.
        for earlier in pairs.iter().take(selection.pair_ordinal) {
            let len = earlier.primary.chronological.len();
            assert!(
                earlier.divergence.single_frame_rule_identifiable
                    || (16..len).all(|position| !position_qualifies(earlier, position, 16))
            );
        }
        assert!((16..selection.position).all(|position| !position_qualifies(pair, position, 16)));
        // Invariants and the model-visible query identity.
        assert_eq!(selection.invariants.state_differing_positions, 0);
        assert!(selection.invariants.window_states_identical);
        assert!(selection.invariants.query_inputs_identical);
        assert!(selection.invariants.unknown_operator_conditioning);
        assert!(selection.invariants.targets_differ);
        assert!(selection.invariants.disagreement_mask_nonempty);
        assert_eq!(rows.primary.current, rows.twin.current);
        assert_eq!(rows.primary.action, rows.twin.action);
        assert_eq!(rows.primary.goal_features, rows.twin.goal_features);
        assert_eq!(rows.primary.context.len(), 16);
        assert_eq!(rows.twin.context.len(), 16);
        assert_ne!(rows.primary_window, rows.twin_window);
        for (left, right) in rows.primary_window.iter().zip(&rows.twin_window) {
            assert_eq!(left.current, right.current);
            assert_eq!(left.action, right.action);
        }
        assert_eq!(
            selection.target_disagreement_pixels,
            rows.disagreement.len()
        );
        assert!(rows
            .disagreement
            .iter()
            .all(|&pixel| rows.primary.next.pixels[pixel] != rows.twin.next.pixels[pixel]));
        assert_ne!(selection.primary_row_sha256, selection.twin_row_sha256);
        assert_ne!(
            selection.primary_window_sha256,
            selection.twin_window_sha256
        );
        assert_ne!(
            selection.primary_target_sha256,
            selection.twin_target_sha256
        );
        // Deterministic, and independent of the pairs scanned after the pick.
        let again = select_context_wiring_rows(&pairs, 16, gameplay)?.0;
        assert_eq!(again, selection);
        let prefix = select_context_wiring_rows(&pairs[..=selection.pair_ordinal], 16, gameplay)?.0;
        assert_eq!(prefix, selection);
        let json = serde_json::to_string(&selection)?;
        let back: ContextWiringSelection = serde_json::from_str(&json)?;
        assert_eq!(back, selection);
        Ok(())
    }

    #[test]
    fn selection_fails_closed_without_a_qualifying_pair_or_ordered_scan() -> Result<()> {
        let pairs = population(3)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        assert!(select_context_wiring_rows(&[], 16, gameplay).is_err());
        // Mark every pair single-frame identifiable: nothing may be selected.
        let mut identifiable = pairs.clone();
        for pair in &mut identifiable {
            pair.divergence.single_frame_rule_identifiable = true;
        }
        assert!(select_context_wiring_rows(&identifiable, 16, gameplay).is_err());
        // No outcome-changing evidence at all: fail closed.
        let mut silent = pairs.clone();
        for pair in &mut silent {
            pair.divergence.outcome_changing_positions.clear();
        }
        assert!(select_context_wiring_rows(&silent, 16, gameplay).is_err());
        // A state-differing census is an integrity failure, not a selection.
        let selected = select_context_wiring_rows(&pairs, 16, gameplay)?
            .0
            .pair_ordinal;
        let mut differing = pairs.clone();
        differing[selected].divergence.state_differing_positions = 1;
        let error = select_context_wiring_rows(&differing, 16, gameplay)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains("integrity"), "{error}");
        // Out-of-order pairs are rejected.
        let mut reversed = pairs;
        reversed.reverse();
        assert!(select_context_wiring_rows(&reversed, 16, gameplay).is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_gate_applies_the_registered_thresholds_exactly() -> Result<()> {
        let spec = ContextWiringSpec::registered();
        let pass = checkpoint_gate(
            &evaluation(8, 2e-4, 1e-5, 0, true),
            &evaluation(8, -2e-4, 0.0, 1, false),
            &spec,
        )?;
        assert!(pass.wiring_gate && pass.promotion_gate);
        assert!((pass.interaction - 4e-4).abs() < 1e-12);
        // Boundary: D exactly 1e-4 is not strictly greater.
        let boundary = checkpoint_gate(
            &evaluation(8, 1e-4, 1e-5, 0, true),
            &evaluation(8, -2e-4, 1e-5, 0, false),
            &spec,
        )?;
        assert!(!boundary.wiring_gate);
        // Boundary: D_swapped exactly -1e-4 is not strictly smaller. (The
        // interaction threshold 2e-4 is implied whenever both D gates hold
        // and is still reported.)
        let swapped_boundary = checkpoint_gate(
            &evaluation(8, 2e-4, 1e-5, 0, true),
            &evaluation(8, -1e-4, 1e-5, 0, false),
            &spec,
        )?;
        assert!(!swapped_boundary.wiring_gate);
        assert!((swapped_boundary.interaction - 3e-4).abs() < 1e-12);
        // Each arm must be distribution-sensitive.
        let inert = checkpoint_gate(
            &evaluation(8, 2e-4, 1e-5, 0, true),
            &evaluation(8, -2e-4, 1e-7, 0, false),
            &spec,
        )?;
        assert!(!inert.wiring_gate);
        // The mixed-K0 invariant is part of the wiring gate.
        let mut leaked = evaluation(8, -2e-4, 1e-5, 0, false);
        leaked.mixed_k0_invariant_pass = false;
        assert!(!checkpoint_gate(&evaluation(8, 2e-4, 1e-5, 0, true), &leaked, &spec)?.wiring_gate);
        // Arms at different updates are an error, not a silent comparison.
        assert!(checkpoint_gate(
            &evaluation(8, 2e-4, 1e-5, 0, true),
            &evaluation(16, -2e-4, 1e-5, 0, false),
            &spec
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn decision_allows_a_later_joint_promotion_pair() -> Result<()> {
        let family = CONTEXT_WIRING_CHECKPOINT_FAMILY;
        // Wiring at 8 and 16 without promotion does not suppress a later
        // consecutive joint wiring+promotion pair.
        let gates = vec![
            gate(0, false, false),
            gate(8, true, false),
            gate(16, true, true),
            gate(32, false, false),
            gate(64, false, false),
            gate(128, true, true),
            gate(256, true, true),
        ];
        let decision = context_wiring_decision(&gates, &family)?;
        assert_eq!(decision.wiring_checkpoints, Some((8, 16)));
        assert_eq!(decision.promotion_checkpoints, Some((128, 256)));
        assert!(decision.wiring_pass && decision.promotion_pass && decision.early_stop);
        let spec = ContextWiringSpec::registered();
        let verdict = context_wiring_verdict(gates, &spec, 256)?;
        assert_eq!(verdict.outcome, "wiring_and_promotion_pass");
        assert_eq!(verdict.early_stop_update, None);
        // Non-consecutive wiring passes never count.
        let sparse = vec![
            gate(0, true, true),
            gate(8, false, false),
            gate(16, true, true),
        ];
        let decision = context_wiring_decision(&sparse, &family)?;
        assert!(!decision.wiring_pass);
        // Early success on the earliest pair with both gates.
        let early = vec![
            gate(0, false, false),
            gate(8, true, true),
            gate(16, true, true),
        ];
        let decision = context_wiring_decision(&early, &family)?;
        assert_eq!(decision.wiring_checkpoints, Some((8, 16)));
        assert_eq!(decision.promotion_checkpoints, Some((8, 16)));
        assert!(decision.promotion_pass && decision.early_stop);
        let verdict = context_wiring_verdict(early, &spec, 16)?;
        assert_eq!(verdict.outcome, "wiring_and_promotion_pass");
        assert_eq!(verdict.early_stop_update, Some(16));
        // Full budget without wiring rejects; a shorter preflight does not.
        let none = family
            .iter()
            .map(|&update| gate(update, false, false))
            .collect::<Vec<_>>();
        let verdict = context_wiring_verdict(none.clone(), &spec, 256)?;
        assert_eq!(verdict.outcome, "reject_local_trainability_by_update_256");
        let preflight = ContextWiringSpec::with_budget(256, 16);
        let verdict = context_wiring_verdict(none[..3].to_vec(), &preflight, 16)?;
        assert_eq!(verdict.outcome, "no_wiring_pass_within_preflight_budget");
        // Gates out of family order fail closed.
        assert!(context_wiring_decision(&[gate(8, true, true)], &family).is_err());
        Ok(())
    }

    #[test]
    fn registered_guards_fail_closed_on_drift() {
        let census = TwinMemorizationCensus {
            pairs: 256,
            episodes: 512,
            single_frame_rule_identifiable: 0,
            divergent_pairs: 256,
            pairs_diverging_before_context_len: 254,
            first_divergence_histogram: BTreeMap::from([
                ("6".to_owned(), 245),
                ("13".to_owned(), 9),
                ("20".to_owned(), 1),
                ("27".to_owned(), 1),
            ]),
            chronological_transitions: 10_780,
            outcome_changing_rows: 724,
            state_differing_rows: 0,
            scorable_rows: 2912,
            scorable_rows_after_first_divergence: 2878,
            scorable_rows_with_evidence_in_window: 2858,
        };
        assert!(registered_census_matches(&census).is_ok());
        let mut drifted = census;
        drifted.outcome_changing_rows = 723;
        assert!(registered_census_matches(&drifted).is_err());

        let mut provenance = LaunchProvenance::unknown(Path::new("/nonexistent/tofy"));
        assert!(registered_provenance_guard(&provenance).is_err());
        provenance = valid_registered_provenance();
        assert!(registered_provenance_guard(&provenance).is_ok());
        provenance.source_pushed = Some(false);
        assert!(registered_provenance_guard(&provenance).is_err());
        provenance.source_pushed = Some(true);
        provenance.source_dirty = None;
        assert!(registered_provenance_guard(&provenance).is_err());
    }

    #[test]
    fn disagreement_loss_uses_the_explicit_two_row_pixel_denominator() -> Result<()> {
        let device = Device::Cpu;
        let per_pixel = Tensor::from_vec(vec![1f32, 100.0, 3.0, 200.0], (2, 1, 2), &device)?;
        let mask = Tensor::from_vec(vec![1f32, 0.0, 1.0, 0.0], (2, 1, 2), &device)?;
        let reduced = reduce_disagreement_loss(&per_pixel, &mask, 2)?.to_scalar::<f32>()?;
        assert_eq!(reduced.to_bits(), 2f32.to_bits());
        assert!(reduce_disagreement_loss(&per_pixel, &mask, 0).is_err());
        Ok(())
    }

    #[test]
    fn registered_scores_use_raw_probabilities_not_exp_of_host_log_probs() -> Result<()> {
        let decode = |target_probability: f32, target_log_probability: f32| {
            let mut probabilities = vec![0f32; PALETTE_SIZE];
            probabilities[3] = target_probability;
            let mut log_probs = vec![f32::NEG_INFINITY; PALETTE_SIZE];
            log_probs[3] = target_log_probability;
            TwinContinuousDecodes {
                true_predictions: vec![vec![3]],
                composed: vec![vec![3]],
                latent: vec![vec![0.0]],
                probabilities: vec![probabilities],
                log_probs: vec![log_probs],
                copy_gate: vec![vec![0.0]],
                context_summary: vec![vec![0.0]],
            }
        };
        let own = decode(0.8, 0.5f32.ln());
        let paired = decode(0.7, 0.5f32.ln());
        let score = arm_scores(&own, &[3], &[0])?;
        let expected_unimix = -(0.99 * 0.8f64 + 0.01 / PALETTE_SIZE as f64).ln();
        assert!((score.raw_softmax_nll + 0.5f64.ln()).abs() < 1e-7);
        assert!((score.unimix_nll - expected_unimix).abs() < 1e-7);
        assert!((compare(&own, &paired, &[0])?.probability_l1 - 0.1).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn finalized_manifest_round_trips_and_detects_tampering() -> Result<()> {
        let parent = std::env::temp_dir().join(format!(
            "tofy-e2w-manifest-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let root = parent.join("run");
        fs::create_dir_all(root.join("nested"))?;
        fs::write(root.join("report.json"), b"report")?;
        fs::write(root.join("nested/artifact.bin"), b"artifact")?;
        let digest = finalize_root_manifest(&root)?;
        let (manifest, _) = external_manifest_paths(&root)?;
        assert_eq!(verify_manifest(&root, &manifest)?, digest);
        verify_manifest_sidecar(&manifest, &digest)?;
        fs::write(root.join("nested/artifact.bin"), b"tampered")?;
        assert!(verify_manifest(&root, &manifest).is_err());
        fs::remove_dir_all(&parent)?;
        Ok(())
    }

    #[test]
    fn registered_preflight_binding_requires_a_complete_sealed_run() -> Result<()> {
        let parent = std::env::temp_dir().join(format!(
            "tofy-e2w-preflight-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let root = parent.join("preflight");
        fs::create_dir_all(&root)?;
        let spec = ContextWiringSpec::with_budget(REGISTERED_PAIRS, 8);
        let pairs = twin_memorization_population(&spec.twin_spec())?;
        let census = twin_memorization_census(&pairs, spec.context_len);
        registered_census_matches(&census)?;
        let fingerprint = learning_history_population_fingerprint(
            pairs.iter().flat_map(|pair| [&pair.primary, &pair.twin]),
        );
        let (selection, _) =
            select_context_wiring_rows(&pairs, spec.context_len, gameplay_rows(true) * FRAME_SIDE)?;
        let gpu = GpuIdentity {
            ordinal: 0,
            name: "test-gpu".into(),
            uuid: "GPU-test".into(),
            memory_total_mib: "1".into(),
            driver_version: "test".into(),
        };
        let parent_evidence = ParentEvidenceBinding {
            source_run_identity: REGISTERED_PARENT_RUN_ID.into(),
            root: PathBuf::from("/test/parent"),
            manifest: PathBuf::from("/test/parent.files.sha256"),
            manifest_sha256: REGISTERED_PARENT_MANIFEST_SHA256.into(),
            checkpoint_relative: REGISTERED_PARENT_CHECKPOINT_RELATIVE.into(),
            config_relative: REGISTERED_PARENT_CONFIG_RELATIVE.into(),
        };
        let updates = (1..=8)
            .map(|update| UpdateRecord {
                update,
                loss: 1.0,
                pre_clip_gradient_norm: 1.0,
                gradient_clip_scale: 1.0,
                context_gradient_norm: 1.0,
            })
            .collect::<Vec<_>>();
        let arm = |name: &str| ArmReport {
            name: name.into(),
            row_contexts: vec!["a".into(), "b".into()],
            updates: updates.clone(),
            checkpoints: vec![
                evaluation(0, 0.0, 0.0, 0, false),
                evaluation(8, 0.0, 0.0, 0, false),
            ],
        };
        let gates = vec![gate(0, false, false), gate(8, false, false)];
        let mut preflight = ContextWiringReport {
            schema: CONTEXT_WIRING_SCHEMA.into(),
            evidence_class: EVIDENCE_CLASS.into(),
            run_class: "unregistered_preflight".into(),
            registered: false,
            public_data_read: false,
            lifecycle: LifecycleRecord {
                state: LIFECYCLE_COMPLETE.into(),
                unix_seconds: unix_seconds(),
                evidence_class: EVIDENCE_CLASS.into(),
                run_class: "unregistered_preflight".into(),
                note: "test".into(),
            },
            provenance: valid_registered_provenance(),
            package_version: env!("CARGO_PKG_VERSION").into(),
            command: vec!["tofy".into()],
            device: "cuda".into(),
            device_is_cuda: true,
            gpu_identity: Some(gpu),
            output_root: root.clone(),
            checkpoint: PathBuf::from("/test/checkpoint"),
            checkpoint_sha256: REGISTERED_CHECKPOINT_SHA256.into(),
            train_config: PathBuf::from("/test/config"),
            train_config_sha256: REGISTERED_TRAIN_CONFIG_SHA256.into(),
            parent_evidence: Some(parent_evidence),
            preflight: None,
            model_config: None,
            spec: spec.clone(),
            population: Some(PopulationRecord {
                population: "twin_learning_histories/unseen_seed_7x7".into(),
                population_seed: spec.population_seed,
                pairs: REGISTERED_PAIRS,
                context_len: spec.context_len,
                history: spec.history.clone(),
                fingerprint,
                census,
                registered_fingerprint_match: Some(true),
            }),
            selection: Some(selection.clone()),
            arms: Some(ContextWiringArms {
                correct: arm("correct"),
                swapped: arm("swapped"),
                initial_parameter_sha256: "test".into(),
                arms_initialized_identically: true,
                updates_run: 8,
            }),
            verdict: Some(context_wiring_verdict(gates, &spec, 8)?),
            timing: ContextWiringTiming {
                population_seconds: 1.0,
                arms_seconds: 1.0,
                wall_seconds: 2.0,
            },
            identity_root: String::new(),
            error: None,
        };
        preflight.identity_root = identity_root(&preflight)?;
        write_json_report(&root.join(REPORT_FILE), &preflight)?;
        finalize_root_manifest(&root)?;
        let current = preflight.clone();
        let binding = bind_preflight(&root.join(REPORT_FILE), &current, &selection)?;
        assert_eq!(binding.max_updates, 8);
        fs::write(root.join(REPORT_FILE), b"tampered")?;
        assert!(bind_preflight(&root.join(REPORT_FILE), &current, &selection).is_err());
        fs::remove_dir_all(&parent)?;
        Ok(())
    }

    /// End-to-end CPU smoke on a tiny v6 model with the preflight budget 8:
    /// both arms initialize bit-identically, the update-1 context gradient is
    /// nonzero, the loss is finite, the mixed-K0 invariant holds on CPU, the
    /// family checkpoints `0` and `8` are evaluated, and the report
    /// round-trips. No verdict value is asserted (a tiny random model is not
    /// the registered checkpoint).
    #[test]
    fn two_arm_smoke_on_cpu_records_gradients_checkpoints_and_invariants() -> Result<()> {
        let device = Device::Cpu;
        let spec = ContextWiringSpec::with_budget(4, 8);
        let pairs = population(spec.pairs)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (_selection, rows) = select_context_wiring_rows(&pairs, spec.context_len, gameplay)?;
        let load_arm = || tiny_v6_model(&device, 0xE2);
        let (arms, verdict) =
            run_context_wiring_arms(&spec, &rows, &device, &load_arm, |_| Ok(()))?;
        assert!(arms.arms_initialized_identically);
        assert_eq!(arms.updates_run, 8);
        for arm in [&arms.correct, &arms.swapped] {
            assert_eq!(arm.updates.len(), 8);
            assert!(arm.updates[0].context_gradient_norm > 0.0);
            assert!(arm.updates.iter().all(|update| update.loss.is_finite()
                && update.pre_clip_gradient_norm.is_finite()
                && (0.0..=1.0).contains(&update.gradient_clip_scale)));
            assert_eq!(
                arm.checkpoints
                    .iter()
                    .map(|checkpoint| checkpoint.update)
                    .collect::<Vec<_>>(),
                vec![0, 8]
            );
            for checkpoint in &arm.checkpoints {
                assert!(checkpoint.mixed_k0_invariant_pass);
                assert_eq!(checkpoint.directions.len(), 2);
                for direction in &checkpoint.directions {
                    assert_eq!(direction.disagreement_pixels, rows.disagreement.len());
                    assert!(direction.own.raw_softmax_nll.is_finite());
                    assert!(direction.own.unimix_nll <= -(0.01f64 / 16.0).ln() + 1e-9);
                    assert!(direction.own.raw_argmax_correct <= direction.disagreement_pixels);
                    assert!(
                        (direction.d
                            - (direction.paired.raw_softmax_nll - direction.own.raw_softmax_nll))
                            .abs()
                            < 1e-12
                    );
                }
            }
        }
        // Both arms share the pre-training checkpoint bit for bit.
        assert_eq!(arms.correct.checkpoints[0], arms.swapped.checkpoints[0]);
        assert_eq!(verdict.evaluated_checkpoints, vec![0, 8]);
        assert_eq!(verdict.updates_run, 8);
        let json = serde_json::to_string(&(arms.clone(), verdict.clone()))?;
        let back: (ContextWiringArms, ContextWiringVerdict) = serde_json::from_str(&json)?;
        assert_eq!(back, (arms, verdict));
        Ok(())
    }
}
