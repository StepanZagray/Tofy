//! `p2-context-confirmation` — E2C second-pair exact wiring and launch
//! stability confirmation (`docs/research/2026-09-03-v6-local-falsifiers-prereg.md`,
//! "Post-E2 confirmation E2C").
//!
//! E2C repeats the E2W two-row context-wiring diagnostic on the next
//! model-free qualifying twin pair (meta-episode IDs `2..`) with three
//! bit-identically initialized arms (`correct_a`, `correct_b`, `swapped`), a
//! deterministically ordered diagnostic optimizer (every floating parameter
//! sorted by canonical name before AdamW construction, the global-norm
//! reduction, the clip and state hashing), an exact argmax confirmation gate
//! on top of E2W's continuous gate, cross-direction K0 identities, a sealed
//! E2W checkpoint-0 legacy parity gate, and a bit-exact
//! preflight-versus-registered launch parity gate at update 8. Its result is
//! single-seed `implementation_smoke` evidence only.
//!
//! Data boundary: only the registered synthetic twin population is generated.
//! Nothing in this module reads public ARC-AGI-3 data or saves checkpoints.

use crate::p2::cli::P2ContextWiringArgs;
use crate::p2::context_wiring::{
    append_command_log, arm_scores_row, bits_differing, build_arm, direction_rows, ensure_finite,
    evaluate_checkpoint, external_manifest_paths, file_sha256_hex, generate_population,
    identity_frame_sha256, open_diagnostic_device, open_run_root, ordered_parameter_names_sha256,
    ordered_parameter_sha256, registered_census_matches, registered_provenance_guard,
    same_build_identity, seal_run_root, select_context_wiring_rows_from, train_update,
    training_disagreement_mask, unix_seconds, verify_diagnostic_inputs, verify_manifest,
    verify_manifest_sidecar, verify_no_input_drift, Arm, CheckpointEvaluation, ContextArmScores,
    ContextWiringReport, ContextWiringRows, ContextWiringSelection, ContextWiringSpec,
    ContextWiringTiming, DirectionRows, GpuIdentity, LifecycleRecord, ModelConfigSummary,
    ParameterOrdering, ParentEvidenceBinding, PopulationRecord, UpdateRecord,
    CONTEXT_WIRING_MAX_UPDATES, CONTEXT_WIRING_SCHEMA, EVIDENCE_CLASS, FAILED_EVIDENCE_CLASS,
    LIFECYCLE_COMPLETE, LIFECYCLE_FAILED, LIFECYCLE_RUNNING, REGISTERED_CHECKPOINT_SHA256,
    REGISTERED_PAIRS, REGISTERED_POPULATION_FINGERPRINT, REGISTERED_TRAIN_CONFIG_SHA256,
    REPORT_FILE, RUN_CLASS_PREFLIGHT, RUN_CLASS_REGISTERED,
};
use crate::p2::data::{gameplay_rows, AugmentedTwinPair, FRAME_SIDE};
use crate::p2::eval::{load_model, twin_continuous_decode_rows, TwinContinuousDecodes};
use crate::p2::evidence::{launch_provenance, LaunchProvenance};
use crate::p2::model::WorldModel;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const CONTEXT_CONFIRMATION_SCHEMA: &str = "p2.context_wiring_confirmation.v1";
/// Meta-episode 0 was rejected by E2W's scan and 1 was E2W's trained pair.
pub const CONFIRMATION_SCAN_START_META_EPISODE_ID: u64 = 2;
pub const CONFIRMATION_EXCLUDED_META_EPISODE_IDS: [u64; 2] = [0, 1];
pub const CONFIRMATION_ARMS: [&str; 3] = ["correct_a", "correct_b", "swapped"];
/// Sealed registered E2W report and root manifest (legacy parity source).
pub const REGISTERED_E2W_REPORT_SHA256: &str =
    "0d3e54f9a8f8fa17be553cb48db44b4184259242b7ea2e782a565b3166a04eca";
pub const REGISTERED_E2W_MANIFEST_SHA256: &str =
    "af133734e0e5d886e29427b4f8e06a8e94337e2d0ae8daeec861fe77321a403d";
pub const REGISTERED_E2W_META_EPISODE_ID: u64 = 1;
pub const REGISTERED_E2W_OUTCOME: &str = "wiring_only_no_promotion";
/// Internal wall-clock caps enforced by the executable between updates (an
/// outer process timeout is applied by the launcher as well).
pub const CONFIRMATION_PREFLIGHT_DEADLINE_SECONDS: u64 = 120;
pub const CONFIRMATION_REGISTERED_DEADLINE_SECONDS: u64 = 600;
/// Update at which the preflight and registered launches must agree bit for bit.
pub const LAUNCH_PARITY_UPDATE: usize = 8;

pub const OUTCOME_CONFIRMATION_PASS: &str = "confirmation_pass";
pub const OUTCOME_REJECT: &str = "reject_second_pair_exact_wiring_by_update_256";
/// Unregistered budgets (< 256 updates) cannot reach either registered label.
pub const OUTCOME_PREFLIGHT: &str = "no_confirmation_pass_within_preflight_budget";

pub(crate) const RUN_CLASS_REGISTERED_CONFIRMATION: &str = "registered_confirmation";
const COMMAND_TAG: &str = "p2-context-confirmation";
const IDENTITY_DOMAIN: &str = "tofy.p2.context_wiring_confirmation.identity.v1";
const DISAGREEMENT_MASK_DOMAIN: &str = "tofy.p2.context_wiring_confirmation.mask.v1";

// ---- specification -------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextConfirmationSpec {
    /// The E2W optimizer/budget/threshold contract this run inherits.
    pub wiring: ContextWiringSpec,
    pub scan_start_meta_episode_id: u64,
    pub excluded_meta_episode_ids: Vec<u64>,
    pub parameter_ordering: ParameterOrdering,
    pub arms: Vec<String>,
    pub legacy_parity_meta_episode_id: u64,
    pub launch_parity_update: usize,
    /// Internal wall-clock cap (preflight 120 s, registered 600 s).
    pub deadline_seconds: u64,
}

impl ContextConfirmationSpec {
    /// The exact registered E2C configuration.
    pub fn registered() -> Self {
        Self::with_budget(REGISTERED_PAIRS, CONTEXT_WIRING_MAX_UPDATES, true)
    }

    /// The bindable 256-pair, 8-update CUDA preflight configuration.
    pub fn exact_preflight() -> Self {
        Self::with_budget(REGISTERED_PAIRS, LAUNCH_PARITY_UPDATE, false)
    }

    pub fn with_budget(pairs: usize, max_updates: usize, registered: bool) -> Self {
        Self {
            wiring: ContextWiringSpec::with_budget(pairs, max_updates),
            scan_start_meta_episode_id: CONFIRMATION_SCAN_START_META_EPISODE_ID,
            excluded_meta_episode_ids: CONFIRMATION_EXCLUDED_META_EPISODE_IDS.to_vec(),
            parameter_ordering: ParameterOrdering::CanonicalNameSorted,
            arms: CONFIRMATION_ARMS
                .iter()
                .map(|arm| (*arm).to_owned())
                .collect(),
            legacy_parity_meta_episode_id: REGISTERED_E2W_META_EPISODE_ID,
            launch_parity_update: LAUNCH_PARITY_UPDATE,
            deadline_seconds: if registered {
                CONFIRMATION_REGISTERED_DEADLINE_SECONDS
            } else {
                CONFIRMATION_PREFLIGHT_DEADLINE_SECONDS
            },
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.wiring.validate()?;
        if self.scan_start_meta_episode_id != CONFIRMATION_SCAN_START_META_EPISODE_ID
            || self.excluded_meta_episode_ids != CONFIRMATION_EXCLUDED_META_EPISODE_IDS
        {
            bail!("E2C scans meta-episode IDs 2.. and excludes 0 and 1");
        }
        if self.parameter_ordering != ParameterOrdering::CanonicalNameSorted {
            bail!("E2C requires canonically name-sorted floating parameters");
        }
        if self.arms != CONFIRMATION_ARMS {
            bail!("E2C arms must be exactly {CONFIRMATION_ARMS:?}");
        }
        if self.legacy_parity_meta_episode_id != REGISTERED_E2W_META_EPISODE_ID
            || self.launch_parity_update != LAUNCH_PARITY_UPDATE
        {
            bail!("E2C legacy/launch parity anchors are fixed");
        }
        if self.deadline_seconds != CONFIRMATION_PREFLIGHT_DEADLINE_SECONDS
            && self.deadline_seconds != CONFIRMATION_REGISTERED_DEADLINE_SECONDS
        {
            bail!("E2C deadline must be the 2-minute preflight or 10-minute registered cap");
        }
        Ok(())
    }

    pub fn is_registered_contract(&self) -> bool {
        *self == Self::registered()
    }
}

/// Hash the semantic one-bit disagreement mask over the decoded gameplay
/// region. The digest is invariant to the input index order and binds both the
/// region size and every selected pixel. E2D reuses this exact digest so its
/// fixed mask hash is comparable with the failed E2C preflights.
pub(crate) fn disagreement_mask_sha256(
    disagreement: &[usize],
    gameplay_pixels: usize,
) -> Result<String> {
    if disagreement.is_empty() {
        bail!("E2C disagreement mask is empty");
    }
    let mut mask = vec![0u8; gameplay_pixels];
    for &pixel in disagreement {
        let value = mask
            .get_mut(pixel)
            .ok_or_else(|| anyhow::anyhow!("E2C disagreement pixel {pixel} is out of bounds"))?;
        *value = 1;
    }
    identity_frame_sha256(&[
        ("domain", DISAGREEMENT_MASK_DOMAIN.as_bytes().to_vec()),
        (
            "gameplay_pixels",
            (gameplay_pixels as u64).to_le_bytes().to_vec(),
        ),
        ("mask", mask),
    ])
}

// ---- frozen evaluation additions ---------------------------------------------

/// The two data-true K0 directions decoded together must be bit-identical
/// (identical query inputs), each direction's shared-decode NLLs must equal
/// its singleton decode bit for bit, and aggregate K0 raw correctness is
/// bounded by `m` of `2m` because the targets differ on every disagreement
/// pixel. Any violation is an integrity failure, not a negative result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingletonSharedNllComparison {
    pub direction: String,
    pub singleton_raw_nll: f64,
    pub shared_raw_nll: f64,
    pub raw_nll_abs_difference: f64,
    pub raw_nll_ulp_distance: u64,
    pub singleton_unimix_nll: f64,
    pub shared_unimix_nll: f64,
    pub unimix_nll_abs_difference: f64,
    pub unimix_nll_ulp_distance: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedK0Invariant {
    pub pass: bool,
    pub latent_elements_differing: usize,
    pub raw_probability_elements_differing: usize,
    pub log_probability_elements_differing: usize,
    pub copy_gate_elements_differing: usize,
    pub raw_argmax_pixels_differing: usize,
    pub composed_argmax_pixels_differing: usize,
    /// Per direction (`primary`, `twin`).
    pub singleton_raw_nll_bit_identical: Vec<bool>,
    pub singleton_unimix_nll_bit_identical: Vec<bool>,
    /// Exact values and bit distances retained when batch-size parity fails.
    pub singleton_shared_nll: Vec<SingletonSharedNllComparison>,
    pub disagreement_pixels_per_direction: usize,
    pub k0_raw_argmax_correct_total: usize,
    /// `k0_raw_argmax_correct_total <= m`.
    pub finite_bound_holds: bool,
}

pub(crate) fn shared_k0_invariant(
    shared: &TwinContinuousDecodes,
    singleton: &[ContextArmScores; 2],
    targets: &[&[u8]; 2],
    disagreement: &[usize],
) -> Result<SharedK0Invariant> {
    if shared.latent.len() != 2
        || shared.probabilities.len() != 2
        || shared.log_probs.len() != 2
        || shared.copy_gate.len() != 2
        || shared.true_predictions.len() != 2
        || shared.composed.len() != 2
    {
        bail!("shared K0 decode must contain exactly the two directions");
    }
    if disagreement.is_empty() {
        bail!("shared K0 invariant needs a nonempty disagreement mask");
    }
    let pixels_differing = |left: &[u8], right: &[u8]| {
        if left.len() != right.len() {
            return left.len().max(right.len());
        }
        left.iter().zip(right).filter(|(a, b)| a != b).count()
    };
    let mut singleton_raw_nll_bit_identical = Vec::with_capacity(2);
    let mut singleton_unimix_nll_bit_identical = Vec::with_capacity(2);
    let mut singleton_shared_nll = Vec::with_capacity(2);
    for (row, (target, single)) in targets.iter().zip(singleton).enumerate() {
        let scores = arm_scores_row(shared, row, target, disagreement)?;
        singleton_raw_nll_bit_identical
            .push(scores.raw_softmax_nll.to_bits() == single.raw_softmax_nll.to_bits());
        singleton_unimix_nll_bit_identical
            .push(scores.unimix_nll.to_bits() == single.unimix_nll.to_bits());
        singleton_shared_nll.push(SingletonSharedNllComparison {
            direction: ["primary", "twin"][row].into(),
            singleton_raw_nll: single.raw_softmax_nll,
            shared_raw_nll: scores.raw_softmax_nll,
            raw_nll_abs_difference: (single.raw_softmax_nll - scores.raw_softmax_nll).abs(),
            raw_nll_ulp_distance: single
                .raw_softmax_nll
                .to_bits()
                .abs_diff(scores.raw_softmax_nll.to_bits()),
            singleton_unimix_nll: single.unimix_nll,
            shared_unimix_nll: scores.unimix_nll,
            unimix_nll_abs_difference: (single.unimix_nll - scores.unimix_nll).abs(),
            unimix_nll_ulp_distance: single
                .unimix_nll
                .to_bits()
                .abs_diff(scores.unimix_nll.to_bits()),
        });
    }
    let m = disagreement.len();
    let k0_raw_argmax_correct_total =
        singleton[0].raw_argmax_correct + singleton[1].raw_argmax_correct;
    let invariant = SharedK0Invariant {
        pass: false,
        latent_elements_differing: bits_differing(&shared.latent[0], &shared.latent[1]),
        raw_probability_elements_differing: bits_differing(
            &shared.probabilities[0],
            &shared.probabilities[1],
        ),
        log_probability_elements_differing: bits_differing(
            &shared.log_probs[0],
            &shared.log_probs[1],
        ),
        copy_gate_elements_differing: bits_differing(&shared.copy_gate[0], &shared.copy_gate[1]),
        raw_argmax_pixels_differing: pixels_differing(
            &shared.true_predictions[0],
            &shared.true_predictions[1],
        ),
        composed_argmax_pixels_differing: pixels_differing(
            &shared.composed[0],
            &shared.composed[1],
        ),
        singleton_raw_nll_bit_identical,
        singleton_unimix_nll_bit_identical,
        singleton_shared_nll,
        disagreement_pixels_per_direction: m,
        k0_raw_argmax_correct_total,
        finite_bound_holds: k0_raw_argmax_correct_total <= m,
    };
    Ok(SharedK0Invariant {
        pass: invariant.latent_elements_differing == 0
            && invariant.raw_probability_elements_differing == 0
            && invariant.log_probability_elements_differing == 0
            && invariant.copy_gate_elements_differing == 0
            && invariant.raw_argmax_pixels_differing == 0
            && invariant.composed_argmax_pixels_differing == 0
            && invariant
                .singleton_raw_nll_bit_identical
                .iter()
                .all(|ok| *ok)
            && invariant
                .singleton_unimix_nll_bit_identical
                .iter()
                .all(|ok| *ok)
            && invariant.finite_bound_holds,
        ..invariant
    })
}

/// Raw-argmax totals over both directions (`2m` pixels) for the exact gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactArgmaxTotals {
    pub disagreement_pixels_per_direction: usize,
    pub total_pixels: usize,
    pub own_raw_argmax_correct: usize,
    pub paired_raw_argmax_correct: usize,
    pub k0_raw_argmax_correct: usize,
}

pub fn exact_totals(evaluation: &CheckpointEvaluation) -> Result<ExactArgmaxTotals> {
    let [primary, twin] = evaluation.directions.as_slice() else {
        bail!("E2C evaluation must contain exactly two query directions");
    };
    let m = primary.disagreement_pixels;
    if m == 0 || twin.disagreement_pixels != m {
        bail!("E2C directions must share a nonempty disagreement mask");
    }
    Ok(ExactArgmaxTotals {
        disagreement_pixels_per_direction: m,
        total_pixels: 2 * m,
        own_raw_argmax_correct: primary.own.raw_argmax_correct + twin.own.raw_argmax_correct,
        paired_raw_argmax_correct: primary.paired.raw_argmax_correct
            + twin.paired.raw_argmax_correct,
        k0_raw_argmax_correct: primary.k0.raw_argmax_correct + twin.k0.raw_argmax_correct,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmationCheckpoint {
    pub update: usize,
    /// SHA-256 over every parameter's F32 bits in canonical name order.
    pub parameter_sha256: String,
    /// Every legacy E2W checkpoint field.
    pub evaluation: CheckpointEvaluation,
    pub shared_k0_invariant: SharedK0Invariant,
    pub exact: ExactArgmaxTotals,
}

/// Per-arm checkpoint record of a three-arm confirmation protocol (E2C's
/// [`ConfirmationCheckpoint`], E2D's semantic checkpoint). The shared gate,
/// verdict, replica and launch-parity logic only needs these accessors.
pub trait ArmCheckpoint:
    Clone + PartialEq + std::fmt::Debug + Serialize + serde::de::DeserializeOwned
{
    fn update(&self) -> usize;
    fn parameter_sha256(&self) -> &str;
    fn evaluation(&self) -> &CheckpointEvaluation;
    fn exact(&self) -> &ExactArgmaxTotals;
    /// Mixed-batch and protocol-specific cross-direction K0 invariants.
    fn k0_invariants_pass(&self) -> bool;
}

impl ArmCheckpoint for ConfirmationCheckpoint {
    fn update(&self) -> usize {
        self.update
    }
    fn parameter_sha256(&self) -> &str {
        &self.parameter_sha256
    }
    fn evaluation(&self) -> &CheckpointEvaluation {
        &self.evaluation
    }
    fn exact(&self) -> &ExactArgmaxTotals {
        &self.exact
    }
    fn k0_invariants_pass(&self) -> bool {
        self.evaluation.mixed_k0_invariant_pass && self.shared_k0_invariant.pass
    }
}

fn evaluate_arm_checkpoint(
    arm: &Arm,
    directions: &[DirectionRows; 2],
    update: usize,
    device: &Device,
) -> Result<ConfirmationCheckpoint> {
    let evaluation = evaluate_checkpoint(&arm.model, directions, update, device)?;
    let shared = twin_continuous_decode_rows(
        &arm.model,
        &[directions[0].k0.clone(), directions[1].k0.clone()],
        device,
    )?;
    ensure_finite(&shared, "shared K0 decode")?;
    let [primary, twin] = evaluation.directions.as_slice() else {
        bail!("E2C evaluation must contain exactly two query directions");
    };
    let shared_k0_invariant = shared_k0_invariant(
        &shared,
        &[primary.k0.clone(), twin.k0.clone()],
        &[&directions[0].target, &directions[1].target],
        &directions[0].disagreement,
    )?;
    if !shared_k0_invariant.pass {
        bail!(
            "{} arm checkpoint {update}: cross-direction K0 identity, singleton parity or finite \
             bound failed (integrity failure): {shared_k0_invariant:?}",
            arm.name
        );
    }
    let exact = exact_totals(&evaluation)?;
    Ok(ConfirmationCheckpoint {
        update,
        parameter_sha256: ordered_parameter_sha256(&arm.parameters)?,
        evaluation,
        shared_k0_invariant,
        exact,
    })
}

// ---- gates and verdict -----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmationGate {
    pub update: usize,
    pub d_correct_a: f64,
    pub d_correct_b: f64,
    pub d_swapped: f64,
    pub interaction_a: f64,
    pub interaction_b: f64,
    pub probability_l1_correct_a: f64,
    pub probability_l1_correct_b: f64,
    pub probability_l1_swapped: f64,
    pub raw_argmax_disagreement_correct_a: usize,
    pub raw_argmax_disagreement_correct_b: usize,
    pub raw_argmax_disagreement_swapped: usize,
    /// Mixed-batch and cross-direction K0 invariants in every arm.
    pub k0_invariants_pass: bool,
    pub continuous_gate: bool,
    pub exact_correct_a: bool,
    pub exact_correct_b: bool,
    pub exact_swapped: bool,
    pub exact_gate: bool,
    pub confirmation_gate: bool,
    /// Derived evaluator invariant (`2m > K0 <= m`), not independent evidence.
    pub correct_own_exceeds_k0_derived: bool,
}

/// E2C's continuous and exact gates at one checkpoint. E2D reuses the same
/// definition and thresholds unchanged; only the per-checkpoint K0 record
/// behind [`ArmCheckpoint::k0_invariants_pass`] differs between protocols.
pub fn confirmation_gate<C: ArmCheckpoint>(
    correct_a: &C,
    correct_b: &C,
    swapped: &C,
    spec: &ContextWiringSpec,
) -> Result<ConfirmationGate> {
    if correct_a.update() != correct_b.update() || correct_a.update() != swapped.update() {
        bail!(
            "arms evaluated different updates: correct_a={} correct_b={} swapped={}",
            correct_a.update(),
            correct_b.update(),
            swapped.update()
        );
    }
    let sensitive = |checkpoint: &C| {
        checkpoint.evaluation().probability_l1 > spec.probability_l1_threshold
            || checkpoint.evaluation().raw_argmax_disagreement_pixels >= 1
    };
    let k0_invariants_pass = [correct_a, correct_b, swapped]
        .iter()
        .all(|checkpoint| checkpoint.k0_invariants_pass());
    let (eval_a, eval_b, eval_s) = (
        correct_a.evaluation(),
        correct_b.evaluation(),
        swapped.evaluation(),
    );
    let interaction_a = eval_a.d - eval_s.d;
    let interaction_b = eval_b.d - eval_s.d;
    let continuous_gate = eval_a.d > spec.d_threshold
        && eval_b.d > spec.d_threshold
        && eval_s.d < -spec.d_threshold
        && interaction_a > spec.interaction_threshold
        && interaction_b > spec.interaction_threshold
        && sensitive(correct_a)
        && sensitive(correct_b)
        && sensitive(swapped)
        && k0_invariants_pass;
    let exact_correct = |checkpoint: &C| {
        checkpoint.exact().own_raw_argmax_correct == checkpoint.exact().total_pixels
            && checkpoint.exact().paired_raw_argmax_correct == 0
    };
    let exact_correct_a = exact_correct(correct_a);
    let exact_correct_b = exact_correct(correct_b);
    let exact_swapped = swapped.exact().own_raw_argmax_correct == 0
        && swapped.exact().paired_raw_argmax_correct == swapped.exact().total_pixels;
    let exact_gate = exact_correct_a && exact_correct_b && exact_swapped;
    Ok(ConfirmationGate {
        update: correct_a.update(),
        d_correct_a: eval_a.d,
        d_correct_b: eval_b.d,
        d_swapped: eval_s.d,
        interaction_a,
        interaction_b,
        probability_l1_correct_a: eval_a.probability_l1,
        probability_l1_correct_b: eval_b.probability_l1,
        probability_l1_swapped: eval_s.probability_l1,
        raw_argmax_disagreement_correct_a: eval_a.raw_argmax_disagreement_pixels,
        raw_argmax_disagreement_correct_b: eval_b.raw_argmax_disagreement_pixels,
        raw_argmax_disagreement_swapped: eval_s.raw_argmax_disagreement_pixels,
        k0_invariants_pass,
        continuous_gate,
        exact_correct_a,
        exact_correct_b,
        exact_swapped,
        exact_gate,
        confirmation_gate: continuous_gate && exact_gate,
        correct_own_exceeds_k0_derived: exact_correct_a
            && exact_correct_b
            && correct_a.exact().own_raw_argmax_correct > correct_a.exact().k0_raw_argmax_correct
            && correct_b.exact().own_raw_argmax_correct > correct_b.exact().k0_raw_argmax_correct,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmationDecision {
    /// Earliest two consecutive family checkpoints with the joint gate.
    pub confirmation_checkpoints: Option<(usize, usize)>,
    pub confirmation_pass: bool,
    /// Stop at the second checkpoint of the first such pair.
    pub early_stop: bool,
}

pub fn confirmation_decision(
    gates: &[ConfirmationGate],
    family: &[usize],
) -> Result<ConfirmationDecision> {
    if gates.len() > family.len()
        || gates
            .iter()
            .zip(family)
            .any(|(gate, update)| gate.update != *update)
    {
        bail!("E2C gates are not in fixed family order");
    }
    let pair = gates
        .windows(2)
        .position(|pair| pair[0].confirmation_gate && pair[1].confirmation_gate)
        .map(|index| (gates[index].update, gates[index + 1].update));
    Ok(ConfirmationDecision {
        confirmation_checkpoints: pair,
        confirmation_pass: pair.is_some(),
        early_stop: pair.is_some(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmationVerdict {
    pub statistic: String,
    pub rule: String,
    pub d_threshold: f64,
    pub interaction_threshold: f64,
    pub probability_l1_threshold: f64,
    pub checkpoint_family: Vec<usize>,
    pub evaluated_checkpoints: Vec<usize>,
    pub gates: Vec<ConfirmationGate>,
    pub confirmation_pass: bool,
    pub confirmation_checkpoints: Option<(usize, usize)>,
    pub early_stop_update: Option<usize>,
    pub updates_run: usize,
    pub outcome: String,
    pub note: String,
}

/// Outcome labels and rule text of one confirmation protocol. The decision
/// logic (two consecutive family checkpoints, early stop, preflight can never
/// pass) is shared and fixed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VerdictLabels {
    /// Protocol name used in notes (`E2C`, `E2D`).
    pub(crate) protocol: &'static str,
    pub(crate) pass: &'static str,
    pub(crate) reject: &'static str,
    pub(crate) preflight: &'static str,
    pub(crate) statistic: &'static str,
    pub(crate) rule: &'static str,
}

pub(crate) const E2C_VERDICT_LABELS: VerdictLabels = VerdictLabels {
    protocol: "E2C",
    pass: OUTCOME_CONFIRMATION_PASS,
    reject: OUTCOME_REJECT,
    preflight: OUTCOME_PREFLIGHT,
    statistic: "D = raw_softmax_NLL(paired K=16) - raw_softmax_NLL(own K=16) over \
                target-disagreement pixels, averaged within each query direction then \
                equally across both directions; exact gate on raw-argmax totals over 2m pixels",
    rule: "confirmation_pass: at the same two consecutive family checkpoints D_correct_a > 1e-4, \
           D_correct_b > 1e-4, D_swapped < -1e-4, each correct-minus-swapped interaction > 2e-4, \
           every arm pooled own-vs-paired probability L1 > 1e-6 or >= 1 raw argmax \
           disagreement, all mixed-K0 and cross-direction K0 invariants hold, each correct \
           arm scores own 2m/2m and paired 0/2m raw-argmax pixels, and swapped scores own \
           0/2m and paired 2m/2m. Stop at the second checkpoint of the first such pair; \
           otherwise run through update 256 and reject the bounded single-pair claim.",
};

pub fn confirmation_verdict(
    gates: Vec<ConfirmationGate>,
    spec: &ContextConfirmationSpec,
    updates_run: usize,
) -> Result<ConfirmationVerdict> {
    confirmation_verdict_labeled(gates, spec, updates_run, &E2C_VERDICT_LABELS)
}

pub(crate) fn confirmation_verdict_labeled(
    gates: Vec<ConfirmationGate>,
    spec: &ContextConfirmationSpec,
    updates_run: usize,
    labels: &VerdictLabels,
) -> Result<ConfirmationVerdict> {
    let wiring = &spec.wiring;
    let decision = confirmation_decision(&gates, &wiring.checkpoint_family)?;
    let registered_confirmation_pass = spec.is_registered_contract() && decision.confirmation_pass;
    let early_stop_update = decision
        .confirmation_checkpoints
        .map(|(_, second)| second)
        .filter(|update| spec.is_registered_contract() && *update < wiring.max_updates);
    let outcome = if !spec.is_registered_contract() {
        labels.preflight
    } else if registered_confirmation_pass {
        labels.pass
    } else if updates_run >= CONTEXT_WIRING_MAX_UPDATES {
        labels.reject
    } else {
        labels.preflight
    };
    let note = match (
        spec.is_registered_contract(),
        decision.confirmation_checkpoints,
    ) {
        (false, observed) => format!(
            "unregistered run observed confirmation checkpoints {observed:?}, but cannot satisfy \
             {}; outcome {outcome} after {updates_run} updates per arm",
            labels.protocol
        ),
        (true, Some((first, second))) => format!(
            "continuous and exact gates held in correct_a, correct_b and swapped at consecutive \
             checkpoints {first}/{second}; outcome {outcome} after {updates_run} updates per arm \
             (single-seed implementation_smoke; authorizes only the preregistered multi-pair \
             screen, never a model promotion or E3)"
        ),
        (true, None) => format!(
            "no two consecutive family checkpoints jointly satisfied the continuous and exact \
             gates in all three arms; outcome {outcome} after {updates_run} updates per arm \
             (budget {})",
            wiring.max_updates
        ),
    };
    Ok(ConfirmationVerdict {
        statistic: labels.statistic.into(),
        rule: labels.rule.into(),
        d_threshold: wiring.d_threshold,
        interaction_threshold: wiring.interaction_threshold,
        probability_l1_threshold: wiring.probability_l1_threshold,
        checkpoint_family: wiring.checkpoint_family.clone(),
        evaluated_checkpoints: gates.iter().map(|gate| gate.update).collect(),
        gates,
        confirmation_pass: registered_confirmation_pass,
        confirmation_checkpoints: decision.confirmation_checkpoints,
        early_stop_update,
        updates_run,
        outcome: outcome.into(),
        note,
    })
}

// ---- bit parity helpers -----------------------------------------------------

/// Serialized-JSON identity: `serde_json` round-trips every finite float
/// exactly, so equal bytes mean equal bits (non-finite values are rejected
/// earlier by the finiteness guards).
pub(crate) fn json_bit_identical<T: Serialize>(left: &T, right: &T) -> Result<bool> {
    Ok(serde_json::to_vec(left)? == serde_json::to_vec(right)?)
}

fn json_leaf_mismatches(left: &Value, right: &Value, path: &str, out: &mut Vec<String>) {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            for key in left
                .keys()
                .chain(right.keys().filter(|key| !left.contains_key(*key)))
            {
                let child = format!("{path}.{key}");
                match (left.get(key), right.get(key)) {
                    (Some(a), Some(b)) => json_leaf_mismatches(a, b, &child, out),
                    _ => out.push(format!("{child}: present on one side only")),
                }
            }
        }
        (Value::Array(left), Value::Array(right)) if left.len() == right.len() => {
            for (index, (a, b)) in left.iter().zip(right).enumerate() {
                json_leaf_mismatches(a, b, &format!("{path}[{index}]"), out);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.to_string() == b.to_string() => {}
        (Value::Number(_), Value::Number(_)) => out.push(format!("{path}: {left} != {right}")),
        _ if left == right => {}
        _ => out.push(format!("{path}: {left} != {right}")),
    }
}

/// Human-readable leaf paths where two serializable values differ.
pub(crate) fn describe_mismatches<T: Serialize>(
    label: &str,
    left: &T,
    right: &T,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    json_leaf_mismatches(
        &serde_json::to_value(left)?,
        &serde_json::to_value(right)?,
        label,
        &mut out,
    );
    Ok(out)
}

// ---- cross-launch parity ----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchParity {
    pub compared_update: usize,
    pub pass: bool,
    pub mismatches: Vec<String>,
}

/// E2C's registered parity set split into the two components E2D reports
/// separately: `evaluator` (arm initialization hashes, ordered parameter
/// names, row contexts, every checkpoint-0 field) and `optimizer` (update
/// records `1..=8`, update-8 parameter hashes, every checkpoint-8 field).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct LaunchParityMismatches {
    pub(crate) evaluator: Vec<String>,
    pub(crate) optimizer: Vec<String>,
}

pub(crate) fn launch_parity_mismatches<C: ArmCheckpoint>(
    preflight: &ConfirmationArms<C>,
    current: &ConfirmationArms<C>,
) -> Result<LaunchParityMismatches> {
    let mut mismatches = LaunchParityMismatches::default();
    if preflight.parameter_count != current.parameter_count {
        mismatches.evaluator.push(format!(
            "parameter_count: preflight {} != current {}",
            preflight.parameter_count, current.parameter_count
        ));
    }
    if preflight.ordered_parameter_names_sha256 != current.ordered_parameter_names_sha256 {
        mismatches
            .evaluator
            .push("ordered_parameter_names_sha256 differs".into());
    }
    for (before, after) in [
        (&preflight.correct_a, &current.correct_a),
        (&preflight.correct_b, &current.correct_b),
        (&preflight.swapped, &current.swapped),
    ] {
        let name = &after.name;
        if before.name != after.name {
            mismatches
                .evaluator
                .push(format!("arm name {} != {name}", before.name));
        }
        if before.row_contexts != after.row_contexts {
            mismatches
                .evaluator
                .push(format!("{name}: row_contexts differs"));
        }
        if before.initial_parameter_sha256 != after.initial_parameter_sha256 {
            mismatches
                .evaluator
                .push(format!("{name}: initial_parameter_sha256 differs"));
        }
        if before.updates.len() < LAUNCH_PARITY_UPDATE || after.updates.len() < LAUNCH_PARITY_UPDATE
        {
            mismatches.optimizer.push(format!(
                "{name}: fewer than {LAUNCH_PARITY_UPDATE} update records (preflight {}, current {})",
                before.updates.len(),
                after.updates.len()
            ));
        }
        for (record_before, record_after) in before
            .updates
            .iter()
            .zip(&after.updates)
            .take(LAUNCH_PARITY_UPDATE)
        {
            if !json_bit_identical(record_before, record_after)? {
                mismatches.optimizer.extend(describe_mismatches(
                    &format!("{name}.update[{}]", record_after.update),
                    record_before,
                    record_after,
                )?);
            }
        }
        for update in [0, LAUNCH_PARITY_UPDATE] {
            let component = if update == 0 {
                &mut mismatches.evaluator
            } else {
                &mut mismatches.optimizer
            };
            let find = |arm: &ConfirmationArmReport<C>| {
                arm.checkpoints
                    .iter()
                    .find(|checkpoint| checkpoint.update() == update)
                    .cloned()
            };
            match (find(before), find(after)) {
                (Some(checkpoint_before), Some(checkpoint_after)) => {
                    if checkpoint_before.parameter_sha256() != checkpoint_after.parameter_sha256() {
                        component.push(format!(
                            "{name}: parameter_sha256 after update {update} differs"
                        ));
                    }
                    if !json_bit_identical(&checkpoint_before, &checkpoint_after)? {
                        component.extend(describe_mismatches(
                            &format!("{name}.checkpoint[{update}]"),
                            &checkpoint_before,
                            &checkpoint_after,
                        )?);
                    }
                }
                _ => component.push(format!("{name}: checkpoint {update} missing on one side")),
            }
        }
    }
    Ok(mismatches)
}

/// Bit parity between the exact preflight and the current run: arm
/// initialization hashes, ordered parameter names, parameter hashes after
/// update 8, update records `1..=8` and every checkpoint-0/8 field per arm.
pub fn launch_parity(
    preflight: &ConfirmationArms,
    current: &ConfirmationArms,
) -> Result<LaunchParity> {
    let LaunchParityMismatches {
        evaluator,
        optimizer,
    } = launch_parity_mismatches(preflight, current)?;
    let mismatches = [evaluator, optimizer].concat();
    Ok(LaunchParity {
        compared_update: LAUNCH_PARITY_UPDATE,
        pass: mismatches.is_empty(),
        mismatches,
    })
}

// ---- legacy E2W parity ------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2wEvidenceBinding {
    pub report: PathBuf,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub report_sha256: String,
    pub manifest_sha256: String,
    pub source_revision: String,
    pub binary_sha256: String,
    pub outcome: String,
    pub selection: ContextWiringSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyParity {
    pub meta_episode_id: u64,
    pub selection_identical: bool,
    pub correct_checkpoint0_bit_identical: bool,
    pub swapped_checkpoint0_bit_identical: bool,
    pub pass: bool,
    pub mismatches: Vec<String>,
}

/// Compare this binary's re-evaluation of E2W's meta-episode-1 checkpoint-0
/// inputs against both sealed E2W arm records, bit for bit.
pub fn legacy_checkpoint0_parity(
    legacy_selection: Option<&ContextWiringSelection>,
    legacy_correct: &CheckpointEvaluation,
    legacy_swapped: &CheckpointEvaluation,
    selection: &ContextWiringSelection,
    current: &CheckpointEvaluation,
) -> Result<LegacyParity> {
    let mut mismatches = Vec::new();
    let selection_identical = legacy_selection == Some(selection);
    if !selection_identical {
        mismatches
            .push("legacy selection differs from the re-derived meta-episode-1 selection".into());
    }
    if current.update != 0 || legacy_correct.update != 0 || legacy_swapped.update != 0 {
        bail!("legacy parity compares checkpoint 0 only");
    }
    let correct_checkpoint0_bit_identical = json_bit_identical(legacy_correct, current)?;
    if !correct_checkpoint0_bit_identical {
        mismatches.extend(describe_mismatches(
            "correct.checkpoint[0]",
            legacy_correct,
            current,
        )?);
    }
    let swapped_checkpoint0_bit_identical = json_bit_identical(legacy_swapped, current)?;
    if !swapped_checkpoint0_bit_identical {
        mismatches.extend(describe_mismatches(
            "swapped.checkpoint[0]",
            legacy_swapped,
            current,
        )?);
    }
    Ok(LegacyParity {
        meta_episode_id: selection.meta_episode_id,
        selection_identical,
        correct_checkpoint0_bit_identical,
        swapped_checkpoint0_bit_identical,
        pass: mismatches.is_empty(),
        mismatches,
    })
}

fn legacy_checkpoint0(report: &ContextWiringReport, arm: &str) -> Result<CheckpointEvaluation> {
    let arms = report
        .arms
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("sealed E2W report has no arm record"))?;
    let arm = match arm {
        "correct" => &arms.correct,
        "swapped" => &arms.swapped,
        _ => bail!("unknown E2W arm {arm}"),
    };
    arm.checkpoints
        .iter()
        .find(|checkpoint| checkpoint.update == 0)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("sealed E2W {} arm has no checkpoint 0", arm.name))
}

/// Verify the sealed registered E2W report (fixed report and root-manifest
/// digests) and extract its frozen selection.
pub(crate) fn bind_e2w_evidence(path: &Path) -> Result<(E2wEvidenceBinding, ContextWiringReport)> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize E2W report {}", path.display()))?;
    let bytes = fs::read(&report_path)
        .with_context(|| format!("read E2W report {}", report_path.display()))?;
    let report_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if report_sha256 != REGISTERED_E2W_REPORT_SHA256 {
        bail!(
            "E2W report sha256 {report_sha256} is not the sealed registered {REGISTERED_E2W_REPORT_SHA256}"
        );
    }
    let report: ContextWiringReport =
        serde_json::from_slice(&bytes).context("parse sealed E2W report")?;
    let root = report_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("E2W report has no parent directory"))?
        .to_path_buf();
    if report_path.file_name().and_then(|name| name.to_str()) != Some(REPORT_FILE)
        || root.file_name() != report.output_root.file_name()
    {
        bail!("--e2w-report must be the report.json inside its sealed E2W run root");
    }
    let (manifest, _) = external_manifest_paths(&root)?;
    let manifest = fs::canonicalize(&manifest)
        .with_context(|| format!("canonicalize E2W manifest {}", manifest.display()))?;
    let manifest_sha256 = verify_manifest(&root, &manifest)?;
    verify_manifest_sidecar(&manifest, &manifest_sha256)?;
    if manifest_sha256 != REGISTERED_E2W_MANIFEST_SHA256 {
        bail!(
            "E2W root manifest sha256 {manifest_sha256} is not the sealed registered {REGISTERED_E2W_MANIFEST_SHA256}"
        );
    }
    let selection = report
        .selection
        .clone()
        .ok_or_else(|| anyhow::anyhow!("sealed E2W report has no selection"))?;
    if report.schema != CONTEXT_WIRING_SCHEMA
        || !report.registered
        || report.run_class != RUN_CLASS_REGISTERED
        || report.lifecycle.state != LIFECYCLE_COMPLETE
        || report.error.is_some()
        || report.public_data_read
        || !report.device_is_cuda
        || report.checkpoint_sha256 != REGISTERED_CHECKPOINT_SHA256
        || report.train_config_sha256 != REGISTERED_TRAIN_CONFIG_SHA256
        || report
            .population
            .as_ref()
            .map(|population| population.fingerprint.as_str())
            != Some(REGISTERED_POPULATION_FINGERPRINT)
        || selection.meta_episode_id != REGISTERED_E2W_META_EPISODE_ID
        || report
            .verdict
            .as_ref()
            .map(|verdict| verdict.outcome.as_str())
            != Some(REGISTERED_E2W_OUTCOME)
    {
        bail!("sealed E2W report is not the completed registered meta-episode-1 diagnostic");
    }
    legacy_checkpoint0(&report, "correct")?;
    legacy_checkpoint0(&report, "swapped")?;
    let binding = E2wEvidenceBinding {
        report: report_path,
        root,
        manifest,
        report_sha256,
        manifest_sha256,
        source_revision: report.provenance.source_revision.clone(),
        binary_sha256: report.provenance.binary_sha256.clone(),
        outcome: REGISTERED_E2W_OUTCOME.into(),
        selection,
    };
    Ok((binding, report))
}

// ---- three-arm execution ----------------------------------------------------

/// One arm's evidence; `C` is the protocol's per-checkpoint record
/// (E2C: [`ConfirmationCheckpoint`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmationArmReport<C = ConfirmationCheckpoint> {
    pub name: String,
    /// Row order of the physical batch: `[primary, twin]` with this arm's
    /// context assignment.
    pub row_contexts: Vec<String>,
    pub initial_parameter_sha256: String,
    pub updates: Vec<UpdateRecord>,
    pub checkpoints: Vec<C>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaScoreDifference {
    pub raw_softmax_nll_abs: f64,
    pub unimix_nll_abs: f64,
    pub raw_argmax_correct: i64,
    pub composed_argmax_correct: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaComparisonDifference {
    pub probability_l1_abs: f64,
    pub latent_rms_abs: f64,
    pub context_summary_rms_abs: f64,
    pub copy_gate_mean_absolute_abs: f64,
    pub raw_argmax_disagreement_pixels: i64,
    pub composed_argmax_disagreement_pixels: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaDirectionComparison {
    pub direction: String,
    pub own: ReplicaScoreDifference,
    pub paired: ReplicaScoreDifference,
    pub k0: ReplicaScoreDifference,
    pub own_vs_paired: ReplicaComparisonDifference,
    pub own_vs_k0: ReplicaComparisonDifference,
}

/// `correct_a` versus `correct_b` at one checkpoint. Bit-identical replicas
/// show within-process stability only, never independent replication or
/// cross-launch evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaComparison {
    pub update: usize,
    pub parameter_sha256_identical: bool,
    pub evaluation_bit_identical: bool,
    pub directions: Vec<ReplicaDirectionComparison>,
}

pub fn replica_comparison<C: ArmCheckpoint>(
    correct_a: &C,
    correct_b: &C,
) -> Result<ReplicaComparison> {
    let (update, evaluation_a, evaluation_b) = (
        correct_a.update(),
        correct_a.evaluation(),
        correct_b.evaluation(),
    );
    if update != correct_b.update()
        || evaluation_a.directions.len() != evaluation_b.directions.len()
    {
        bail!("replica checkpoints are not comparable");
    }
    if evaluation_a
        .directions
        .iter()
        .zip(&evaluation_b.directions)
        .any(|(a, b)| a.direction != b.direction)
    {
        bail!("replica checkpoint directions differ");
    }
    let signed = |left: usize, right: usize| left as i64 - right as i64;
    let score_difference = |a: &ContextArmScores, b: &ContextArmScores| ReplicaScoreDifference {
        raw_softmax_nll_abs: (a.raw_softmax_nll - b.raw_softmax_nll).abs(),
        unimix_nll_abs: (a.unimix_nll - b.unimix_nll).abs(),
        raw_argmax_correct: signed(a.raw_argmax_correct, b.raw_argmax_correct),
        composed_argmax_correct: signed(a.composed_argmax_correct, b.composed_argmax_correct),
    };
    let comparison_difference =
        |a: &crate::p2::context_wiring::ContextComparison,
         b: &crate::p2::context_wiring::ContextComparison| {
            ReplicaComparisonDifference {
                probability_l1_abs: (a.probability_l1 - b.probability_l1).abs(),
                latent_rms_abs: (a.latent_rms_difference - b.latent_rms_difference).abs(),
                context_summary_rms_abs: (a.context_summary_rms_difference
                    - b.context_summary_rms_difference)
                    .abs(),
                copy_gate_mean_absolute_abs: (a.copy_gate_mean_absolute_difference
                    - b.copy_gate_mean_absolute_difference)
                    .abs(),
                raw_argmax_disagreement_pixels: signed(
                    a.raw_argmax_disagreement_pixels,
                    b.raw_argmax_disagreement_pixels,
                ),
                composed_argmax_disagreement_pixels: signed(
                    a.composed_argmax_disagreement_pixels,
                    b.composed_argmax_disagreement_pixels,
                ),
            }
        };
    let directions = evaluation_a
        .directions
        .iter()
        .zip(&evaluation_b.directions)
        .map(|(a, b)| ReplicaDirectionComparison {
            direction: a.direction.clone(),
            own: score_difference(&a.own, &b.own),
            paired: score_difference(&a.paired, &b.paired),
            k0: score_difference(&a.k0, &b.k0),
            own_vs_paired: comparison_difference(&a.own_vs_paired, &b.own_vs_paired),
            own_vs_k0: comparison_difference(&a.own_vs_k0, &b.own_vs_k0),
        })
        .collect();
    Ok(ReplicaComparison {
        update,
        parameter_sha256_identical: correct_a.parameter_sha256() == correct_b.parameter_sha256(),
        evaluation_bit_identical: json_bit_identical(evaluation_a, evaluation_b)?,
        directions,
    })
}

/// Three-arm evidence; `C` is the protocol's per-checkpoint record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmationArms<C = ConfirmationCheckpoint> {
    pub correct_a: ConfirmationArmReport<C>,
    pub correct_b: ConfirmationArmReport<C>,
    pub swapped: ConfirmationArmReport<C>,
    pub parameter_ordering: ParameterOrdering,
    pub parameter_count: usize,
    /// SHA-256 over the canonically ordered parameter names/dtypes/shapes.
    pub ordered_parameter_names_sha256: String,
    pub ordered_parameter_names_identical: bool,
    pub initial_parameter_sha256: String,
    pub arms_initialized_identically: bool,
    pub replica_comparisons: Vec<ReplicaComparison>,
    pub updates_run: usize,
}

/// Result of the three-arm loop. `failure` carries any mid-run integrity
/// failure (deadline, K0 leak, launch mismatch, non-finite value) while the
/// arm evidence gathered so far is still returned for the report.
pub(crate) struct ConfirmationRun {
    pub(crate) arms: ConfirmationArms,
    pub(crate) launch_parity: Option<LaunchParity>,
    pub(crate) verdict: Option<ConfirmationVerdict>,
    pub(crate) failure: Option<String>,
}

/// Protocol-specific steps of the shared three-arm loop
/// ([`run_protocol_arms`]): how one arm is evaluated at a checkpoint, what
/// extra integrity controls run once all three arms are evaluated, and how
/// preflight-versus-registered parity is judged at the parity update. The
/// arm construction, ordered optimizer, deadline, family checkpoints, gates,
/// decision and early stop are shared and fixed.
pub(crate) trait ConfirmationProtocol {
    type Checkpoint: ArmCheckpoint;

    fn labels(&self) -> &VerdictLabels;

    fn evaluate_arm(
        &self,
        arm: &Arm,
        directions: &[DirectionRows; 2],
        update: usize,
        device: &Device,
    ) -> Result<Self::Checkpoint>;

    /// Runs after all three arms were evaluated at `update` and before the
    /// records are stored. An error is an integrity failure that stops the
    /// run (at checkpoint 0 this is before update 1).
    fn after_checkpoint(
        &mut self,
        update: usize,
        evaluated: &[Self::Checkpoint],
        arms: &[Arm],
        directions: &[DirectionRows; 2],
        device: &Device,
        on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()>;

    /// Judge and record launch parity; returns every mismatch (empty = pass).
    fn launch_parity(
        &mut self,
        preflight: &ConfirmationArms<Self::Checkpoint>,
        current: &ConfirmationArms<Self::Checkpoint>,
        on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<Vec<String>>;
}

/// E2C as registered: the [`SharedK0Invariant`] checkpoint, no extra
/// in-process control, and one undivided launch-parity mismatch list.
pub(crate) struct E2cProtocol {
    pub(crate) launch_parity: Option<LaunchParity>,
}

impl ConfirmationProtocol for E2cProtocol {
    type Checkpoint = ConfirmationCheckpoint;

    fn labels(&self) -> &VerdictLabels {
        &E2C_VERDICT_LABELS
    }

    fn evaluate_arm(
        &self,
        arm: &Arm,
        directions: &[DirectionRows; 2],
        update: usize,
        device: &Device,
    ) -> Result<ConfirmationCheckpoint> {
        evaluate_arm_checkpoint(arm, directions, update, device)
    }

    fn after_checkpoint(
        &mut self,
        _update: usize,
        _evaluated: &[ConfirmationCheckpoint],
        _arms: &[Arm],
        _directions: &[DirectionRows; 2],
        _device: &Device,
        _on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }

    fn launch_parity(
        &mut self,
        preflight: &ConfirmationArms,
        current: &ConfirmationArms,
        _on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<Vec<String>> {
        let parity = launch_parity(preflight, current)?;
        let mismatches = parity.mismatches.clone();
        self.launch_parity = Some(parity);
        Ok(mismatches)
    }
}

/// Result of [`run_protocol_arms`]; see [`ConfirmationRun`].
pub(crate) struct ProtocolRun<C> {
    pub(crate) arms: ConfirmationArms<C>,
    pub(crate) verdict: Option<ConfirmationVerdict>,
    pub(crate) failure: Option<String>,
}

struct RunState<'a, P: ConfirmationProtocol> {
    spec: &'a ContextConfirmationSpec,
    device: &'a Device,
    protocol: &'a mut P,
    arms: Vec<Arm>,
    checkpoints: Vec<Vec<P::Checkpoint>>,
    initial_parameter_sha256: Vec<String>,
    parameter_count: usize,
    ordered_parameter_names_sha256: String,
    directions: [DirectionRows; 2],
    mask: Tensor,
    mask_pixels: usize,
    gates: Vec<ConfirmationGate>,
    updates_run: usize,
    early_stopped: bool,
    deadline: Instant,
    preflight_arms: Option<&'a ConfirmationArms<P::Checkpoint>>,
}

const ROW_CONTEXTS: [[&str; 2]; 3] = [
    ["primary<-primary_window", "twin<-twin_window"],
    ["primary<-primary_window", "twin<-twin_window"],
    ["primary<-twin_window", "twin<-primary_window"],
];

impl<P: ConfirmationProtocol> RunState<'_, P> {
    fn check_deadline(&self, stage: &str) -> Result<()> {
        if Instant::now() >= self.deadline {
            bail!(
                "internal wall-clock cap of {} s exceeded {stage} (integrity failure, run stopped)",
                self.spec.deadline_seconds
            );
        }
        Ok(())
    }

    fn arm_report(&self, index: usize) -> ConfirmationArmReport<P::Checkpoint> {
        ConfirmationArmReport {
            name: self.arms[index].name.into(),
            row_contexts: ROW_CONTEXTS[index]
                .iter()
                .map(|context| (*context).to_owned())
                .collect(),
            initial_parameter_sha256: self.initial_parameter_sha256[index].clone(),
            updates: self.arms[index].updates.clone(),
            checkpoints: self.checkpoints[index].clone(),
        }
    }

    fn arms_report(&self) -> Result<ConfirmationArms<P::Checkpoint>> {
        let replica_comparisons = self.checkpoints[0]
            .iter()
            .zip(&self.checkpoints[1])
            .map(|(a, b)| replica_comparison(a, b))
            .collect::<Result<Vec<_>>>()?;
        let ordered_parameter_names_identical = self.arms.iter().all(|arm| {
            arm.parameters.len() == self.parameter_count
                && ordered_parameter_names_sha256(&arm.parameters)
                    == self.ordered_parameter_names_sha256
        });
        let arms_initialized_identically = self
            .initial_parameter_sha256
            .iter()
            .all(|hash| *hash == self.initial_parameter_sha256[0]);
        Ok(ConfirmationArms {
            correct_a: self.arm_report(0),
            correct_b: self.arm_report(1),
            swapped: self.arm_report(2),
            parameter_ordering: self.spec.parameter_ordering,
            parameter_count: self.parameter_count,
            ordered_parameter_names_sha256: self.ordered_parameter_names_sha256.clone(),
            ordered_parameter_names_identical,
            initial_parameter_sha256: self.initial_parameter_sha256[0].clone(),
            arms_initialized_identically,
            replica_comparisons,
            updates_run: self.updates_run,
        })
    }

    fn evaluate_all(
        &mut self,
        update: usize,
        on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ConfirmationDecision> {
        let mut evaluated = Vec::with_capacity(3);
        for arm in &self.arms {
            evaluated.push(self.protocol.evaluate_arm(
                arm,
                &self.directions,
                update,
                self.device,
            )?);
        }
        let gate = confirmation_gate(
            &evaluated[0],
            &evaluated[1],
            &evaluated[2],
            &self.spec.wiring,
        )?;
        let exact = |index: usize| evaluated[index].exact();
        on_progress(&format!(
            "checkpoint {update}: D_a={:.6e} D_b={:.6e} D_swapped={:.6e} l1_a={:.3e} l1_b={:.3e} \
             l1_swapped={:.3e} own/paired/k0 a={}/{}/{} b={}/{}/{} swapped={}/{}/{} of {} \
             continuous={} exact={} confirmation={} replicas_identical={}",
            gate.d_correct_a,
            gate.d_correct_b,
            gate.d_swapped,
            gate.probability_l1_correct_a,
            gate.probability_l1_correct_b,
            gate.probability_l1_swapped,
            exact(0).own_raw_argmax_correct,
            exact(0).paired_raw_argmax_correct,
            exact(0).k0_raw_argmax_correct,
            exact(1).own_raw_argmax_correct,
            exact(1).paired_raw_argmax_correct,
            exact(1).k0_raw_argmax_correct,
            exact(2).own_raw_argmax_correct,
            exact(2).paired_raw_argmax_correct,
            exact(2).k0_raw_argmax_correct,
            exact(0).total_pixels,
            gate.continuous_gate,
            gate.exact_gate,
            gate.confirmation_gate,
            evaluated[0].parameter_sha256() == evaluated[1].parameter_sha256()
        ))?;
        self.protocol.after_checkpoint(
            update,
            &evaluated,
            &self.arms,
            &self.directions,
            self.device,
            on_progress,
        )?;
        for (index, checkpoint) in evaluated.into_iter().enumerate() {
            self.checkpoints[index].push(checkpoint);
        }
        self.gates.push(gate);
        confirmation_decision(&self.gates, &self.spec.wiring.checkpoint_family)
    }

    fn drive(&mut self, on_progress: &mut dyn FnMut(&str) -> Result<()>) -> Result<()> {
        let wiring = &self.spec.wiring;
        let family = wiring.checkpoint_family.clone();
        let max_updates = wiring.max_updates;
        self.check_deadline("before checkpoint 0")?;
        self.evaluate_all(0, on_progress)?;
        for update in 1..=max_updates {
            self.check_deadline(&format!("before update {update}"))?;
            let mut records = Vec::with_capacity(3);
            for arm in self.arms.iter_mut() {
                records.push(train_update(
                    arm,
                    &self.mask,
                    self.mask_pixels,
                    &self.spec.wiring,
                    self.device,
                )?);
            }
            self.updates_run = update;
            if update == 1 || family.contains(&update) {
                on_progress(&format!(
                    "update {update}: loss a={:.6} b={:.6} swapped={:.6}; pre-clip norm a={:.4} \
                     b={:.4} swapped={:.4}; clip scale a={:.4} b={:.4} swapped={:.4}; context grad \
                     norm a={:.4e} b={:.4e} swapped={:.4e}",
                    records[0].loss,
                    records[1].loss,
                    records[2].loss,
                    records[0].pre_clip_gradient_norm,
                    records[1].pre_clip_gradient_norm,
                    records[2].pre_clip_gradient_norm,
                    records[0].gradient_clip_scale,
                    records[1].gradient_clip_scale,
                    records[2].gradient_clip_scale,
                    records[0].context_gradient_norm,
                    records[1].context_gradient_norm,
                    records[2].context_gradient_norm
                ))?;
            }
            if !family.contains(&update) {
                continue;
            }
            self.check_deadline(&format!("before checkpoint {update}"))?;
            let decision = self.evaluate_all(update, on_progress)?;
            if update == self.spec.launch_parity_update {
                if let Some(preflight) = self.preflight_arms {
                    let current = self.arms_report()?;
                    let mismatches =
                        self.protocol
                            .launch_parity(preflight, &current, on_progress)?;
                    on_progress(&format!(
                        "launch parity at update {update}: pass={} mismatches={}",
                        mismatches.is_empty(),
                        mismatches.len()
                    ))?;
                    if !mismatches.is_empty() {
                        bail!(
                            "preflight-versus-registered bit parity failed at update {update} \
                             (integrity failure; registered run stopped, not a negative model \
                             result): {}",
                            mismatches.join("; ")
                        );
                    }
                }
            }
            if self.spec.is_registered_contract() && decision.early_stop && update < max_updates {
                self.early_stopped = true;
                on_progress(&format!(
                    "confirmation gates held at two consecutive checkpoints; stopping all arms after \
                     checkpoint {update}"
                ))?;
                break;
            }
        }
        Ok(())
    }
}

/// E2C's three-arm loop: [`run_protocol_arms`] under [`E2cProtocol`].
pub(crate) fn run_confirmation_arms(
    spec: &ContextConfirmationSpec,
    rows: &ContextWiringRows,
    device: &Device,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
    deadline: Instant,
    preflight_arms: Option<&ConfirmationArms>,
    on_progress: impl FnMut(&str) -> Result<()>,
) -> Result<ConfirmationRun> {
    let mut protocol = E2cProtocol {
        launch_parity: None,
    };
    let run = run_protocol_arms(
        spec,
        rows,
        device,
        load_arm,
        deadline,
        preflight_arms,
        &mut protocol,
        on_progress,
    )?;
    Ok(ConfirmationRun {
        arms: run.arms,
        launch_parity: protocol.launch_parity,
        verdict: run.verdict,
        failure: run.failure,
    })
}

/// Run the three arms in lockstep on already generated rows. `load_arm` must
/// return a freshly loaded model on every call (bit-identical initialization).
/// Setup failures (before any arm evidence exists) are returned as `Err`;
/// mid-run failures are recorded in [`ProtocolRun::failure`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_protocol_arms<P: ConfirmationProtocol>(
    spec: &ContextConfirmationSpec,
    rows: &ContextWiringRows,
    device: &Device,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
    deadline: Instant,
    preflight_arms: Option<&ConfirmationArms<P::Checkpoint>>,
    protocol: &mut P,
    mut on_progress: impl FnMut(&str) -> Result<()>,
) -> Result<ProtocolRun<P::Checkpoint>> {
    spec.validate()?;
    let wiring = &spec.wiring;
    let windows = [
        (&rows.primary_window, &rows.twin_window),
        (&rows.primary_window, &rows.twin_window),
        (&rows.twin_window, &rows.primary_window),
    ];
    let mut arms = Vec::with_capacity(3);
    for (name, (primary_window, twin_window)) in CONFIRMATION_ARMS.iter().zip(windows) {
        arms.push(build_arm(
            name,
            rows,
            (primary_window, twin_window),
            wiring,
            spec.parameter_ordering,
            device,
            load_arm,
        )?);
    }
    let parameter_count = arms[0].parameters.len();
    if parameter_count == 0 {
        bail!("the model has no floating parameters");
    }
    if arms[0]
        .parameters
        .windows(2)
        .any(|pair| pair[0].0 >= pair[1].0)
    {
        bail!("floating parameters are not strictly name-sorted (integrity failure)");
    }
    let names_sha256 = ordered_parameter_names_sha256(&arms[0].parameters);
    if arms.iter().any(|arm| {
        arm.parameters.len() != parameter_count
            || ordered_parameter_names_sha256(&arm.parameters) != names_sha256
    }) {
        bail!("arms do not share one ordered parameter-name list (integrity failure)");
    }
    let initial_parameter_sha256 = arms
        .iter()
        .map(|arm| ordered_parameter_sha256(&arm.parameters))
        .collect::<Result<Vec<_>>>()?;
    if initial_parameter_sha256
        .iter()
        .any(|hash| *hash != initial_parameter_sha256[0])
    {
        bail!("the three arms did not initialize bit-identically (integrity failure)");
    }
    let mask = training_disagreement_mask(rows, wiring, device)?;
    let mut state = RunState {
        spec,
        device,
        protocol,
        checkpoints: vec![Vec::new(), Vec::new(), Vec::new()],
        arms,
        initial_parameter_sha256,
        parameter_count,
        ordered_parameter_names_sha256: names_sha256,
        directions: direction_rows(rows),
        mask,
        mask_pixels: rows.disagreement.len() * wiring.physical_batch,
        gates: Vec::new(),
        updates_run: 0,
        early_stopped: false,
        deadline,
        preflight_arms,
    };
    let outcome = state.drive(&mut on_progress);
    let arms = state.arms_report()?;
    let (verdict, failure) = match outcome {
        Ok(()) => {
            let verdict = confirmation_verdict_labeled(
                state.gates.clone(),
                spec,
                state.updates_run,
                state.protocol.labels(),
            )?;
            if state.early_stopped != verdict.early_stop_update.is_some() {
                bail!("early-stop bookkeeping disagrees with the verdict");
            }
            (Some(verdict), None)
        }
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    Ok(ProtocolRun {
        arms,
        verdict,
        failure,
    })
}

// ---- report, provenance, CLI -------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmationPreflightBinding {
    pub report: PathBuf,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    pub identity_root: String,
    pub max_updates: usize,
    pub selection: ContextWiringSelection,
    pub disagreement_mask_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextConfirmationReport {
    pub schema: String,
    pub evidence_class: String,
    /// `registered_confirmation` or `unregistered_preflight`; a preflight
    /// cannot satisfy E2C.
    pub run_class: String,
    pub registered: bool,
    pub public_data_read: bool,
    pub checkpoints_saved: bool,
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
    pub e2w_evidence: Option<E2wEvidenceBinding>,
    pub legacy_parity: Option<LegacyParity>,
    pub preflight: Option<ConfirmationPreflightBinding>,
    pub launch_parity: Option<LaunchParity>,
    pub model_config: Option<ModelConfigSummary>,
    pub spec: ContextConfirmationSpec,
    pub population: Option<PopulationRecord>,
    pub selection: Option<ContextWiringSelection>,
    pub disagreement_mask_sha256: String,
    pub arms: Option<ConfirmationArms>,
    pub verdict: Option<ConfirmationVerdict>,
    pub timing: ContextWiringTiming,
    /// Domain-separated SHA-256 over checkpoint, config, binary, source, GPU,
    /// parent/E2W/preflight bindings, population fingerprint, selection rows
    /// and the spec.
    pub identity_root: String,
    pub error: Option<String>,
}

fn identity_root(report: &ContextConfirmationReport) -> Result<String> {
    let selection_rows = report
        .selection
        .as_ref()
        .map(|selection| {
            (
                selection.primary_row_sha256.as_str(),
                selection.twin_row_sha256.as_str(),
            )
        })
        .unwrap_or_default();
    identity_frame_sha256(&[
        ("domain", IDENTITY_DOMAIN.as_bytes().to_vec()),
        (
            "checkpoint_sha256",
            report.checkpoint_sha256.as_bytes().to_vec(),
        ),
        (
            "train_config_sha256",
            report.train_config_sha256.as_bytes().to_vec(),
        ),
        (
            "binary_sha256",
            report.provenance.binary_sha256.as_bytes().to_vec(),
        ),
        (
            "source_revision",
            report.provenance.source_revision.as_bytes().to_vec(),
        ),
        ("gpu_identity", serde_json::to_vec(&report.gpu_identity)?),
        (
            "parent_evidence",
            serde_json::to_vec(&report.parent_evidence)?,
        ),
        ("e2w_evidence", serde_json::to_vec(&report.e2w_evidence)?),
        ("preflight", serde_json::to_vec(&report.preflight)?),
        (
            "population_fingerprint",
            report
                .population
                .as_ref()
                .map_or("", |population| population.fingerprint.as_str())
                .as_bytes()
                .to_vec(),
        ),
        ("primary_row_sha256", selection_rows.0.as_bytes().to_vec()),
        ("twin_row_sha256", selection_rows.1.as_bytes().to_vec()),
        (
            "disagreement_mask_sha256",
            report.disagreement_mask_sha256.as_bytes().to_vec(),
        ),
        ("spec", serde_json::to_vec(&report.spec)?),
    ])
}

/// Whether an arm record holds exactly the 8-update preflight evidence.
pub(crate) fn arm_complete<C: ArmCheckpoint>(arm: &ConfirmationArmReport<C>, name: &str) -> bool {
    arm.name == name
        && arm
            .updates
            .iter()
            .map(|record| record.update)
            .eq(1..=LAUNCH_PARITY_UPDATE)
        && arm
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.update())
            .eq([0, LAUNCH_PARITY_UPDATE])
}

/// Verify the exact 256-pair, 8-update CUDA preflight of this binary and
/// return its binding plus the arm evidence for the update-8 parity gate.
fn bind_confirmation_preflight(
    path: &Path,
    current: &ContextConfirmationReport,
    selection: &ContextWiringSelection,
) -> Result<(ConfirmationPreflightBinding, ConfirmationArms)> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize preflight report {}", path.display()))?;
    let bytes = fs::read(&report_path)
        .with_context(|| format!("read preflight report {}", report_path.display()))?;
    let preflight: ContextConfirmationReport =
        serde_json::from_slice(&bytes).context("parse E2C preflight report")?;
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
    if preflight.schema != CONTEXT_CONFIRMATION_SCHEMA
        || preflight.evidence_class != EVIDENCE_CLASS
        || preflight.run_class != RUN_CLASS_PREFLIGHT
        || preflight.registered
        || preflight.public_data_read
        || preflight.checkpoints_saved
        || preflight.lifecycle.state != LIFECYCLE_COMPLETE
        || preflight.lifecycle.evidence_class != EVIDENCE_CLASS
        || preflight.lifecycle.run_class != RUN_CLASS_PREFLIGHT
        || preflight.error.is_some()
        || !preflight.device_is_cuda
        || preflight.preflight.is_some()
        || preflight.launch_parity.is_some()
        || preflight.package_version != current.package_version
    {
        bail!("preflight report is not a completed, clean, CUDA E2C preflight");
    }
    if preflight.spec != ContextConfirmationSpec::exact_preflight() {
        bail!("registered E2C requires the exact 256-pair, 8-update preflight");
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
        || population.pairs != REGISTERED_PAIRS
        || population.registered_fingerprint_match != Some(true)
    {
        bail!("preflight did not scan the complete registered population");
    }
    registered_census_matches(&population.census)?;
    if preflight.selection.as_ref() != Some(selection) {
        bail!("preflight selected a different registered row");
    }
    if preflight.disagreement_mask_sha256.is_empty()
        || preflight.disagreement_mask_sha256 != current.disagreement_mask_sha256
    {
        bail!("preflight selected a different disagreement mask");
    }
    if preflight.parent_evidence != current.parent_evidence {
        bail!("preflight and registered run do not bind the same sealed parent evidence");
    }
    if preflight.e2w_evidence.is_none() || preflight.e2w_evidence != current.e2w_evidence {
        bail!("preflight and registered run do not bind the same sealed E2W evidence");
    }
    if preflight.legacy_parity.as_ref().map(|parity| parity.pass) != Some(true) {
        bail!("preflight did not pass the E2W legacy checkpoint-0 parity gate");
    }
    if preflight.gpu_identity.is_none() || preflight.gpu_identity != current.gpu_identity {
        bail!("preflight and registered run do not bind the same GPU identity");
    }
    let arms = preflight
        .arms
        .clone()
        .ok_or_else(|| anyhow::anyhow!("preflight has no arm record"))?;
    if preflight.verdict.as_ref().map(|verdict| {
        (
            verdict.updates_run,
            verdict.evaluated_checkpoints.as_slice(),
        ) == (LAUNCH_PARITY_UPDATE, [0, LAUNCH_PARITY_UPDATE].as_slice())
    }) != Some(true)
        || arms.updates_run != LAUNCH_PARITY_UPDATE
        || !arms.arms_initialized_identically
        || !arms.ordered_parameter_names_identical
        || arms.parameter_ordering != ParameterOrdering::CanonicalNameSorted
        || !arm_complete(&arms.correct_a, CONFIRMATION_ARMS[0])
        || !arm_complete(&arms.correct_b, CONFIRMATION_ARMS[1])
        || !arm_complete(&arms.swapped, CONFIRMATION_ARMS[2])
    {
        bail!("preflight did not complete 8 updates in all three arms");
    }
    if identity_root(&preflight)? != preflight.identity_root {
        bail!("preflight identity root does not match its report fields");
    }
    Ok((
        ConfirmationPreflightBinding {
            report: report_path,
            root,
            manifest,
            manifest_sha256,
            identity_root: preflight.identity_root,
            max_updates: preflight.spec.wiring.max_updates,
            selection: selection.clone(),
            disagreement_mask_sha256: preflight.disagreement_mask_sha256,
        },
        arms,
    ))
}

/// `p2-context-confirmation` — E2C second-pair exact wiring and launch
/// stability confirmation (implementation smoke).
#[derive(Debug, Clone, Args)]
pub struct P2ContextConfirmationArgs {
    #[command(flatten)]
    pub wiring: P2ContextWiringArgs,
    /// Sealed registered E2W `report.json` (inside its sealed run root) for the
    /// meta-episode-1 checkpoint-0 legacy parity gate. Required for the
    /// registered run and the bindable 256-pair, 8-update preflight.
    #[arg(long)]
    pub e2w_report: Option<PathBuf>,
}

fn run_inner(
    args: &P2ContextConfirmationArgs,
    report: &mut ContextConfirmationReport,
    root: &Path,
    started: Instant,
) -> Result<()> {
    let wiring = &args.wiring;
    let registered = wiring.registered;
    let exact_preflight = !registered && report.spec == ContextConfirmationSpec::exact_preflight();
    let cuda = wiring.device.trim().starts_with("cuda");
    if registered {
        registered_provenance_guard(&report.provenance)?;
        if !report.spec.is_registered_contract() {
            bail!(
                "registered run requires --max-updates {CONTEXT_WIRING_MAX_UPDATES} and --pairs {REGISTERED_PAIRS}"
            );
        }
        if !cuda {
            bail!("registered run requires a CUDA device");
        }
        if wiring.preflight_report.is_none() {
            bail!("registered run requires --preflight-report");
        }
    } else if wiring.preflight_report.is_some() {
        bail!("--preflight-report is valid only with --registered");
    }
    if exact_preflight {
        registered_provenance_guard(&report.provenance)?;
        if !cuda {
            bail!("the bindable 256-pair, 8-update preflight requires CUDA");
        }
    }
    if (registered || exact_preflight) && args.e2w_report.is_none() {
        bail!("registered and bindable preflight runs require --e2w-report");
    }
    let inputs = verify_diagnostic_inputs(wiring, root, registered || exact_preflight)?;
    report.train_config_sha256 = inputs.train_config_sha256;
    report.checkpoint_sha256 = inputs.checkpoint_sha256;
    report.model_config = Some(inputs.model_config);
    report.parent_evidence = Some(inputs.parent_evidence);
    let train_cfg = inputs.train_cfg;
    let legacy = match args.e2w_report.as_deref() {
        Some(path) => {
            let (binding, legacy_report) = bind_e2w_evidence(path)?;
            if legacy_report.checkpoint_sha256 != report.checkpoint_sha256
                || legacy_report.train_config_sha256 != report.train_config_sha256
            {
                bail!("sealed E2W evidence does not bind this run's checkpoint and config");
            }
            report.e2w_evidence = Some(binding);
            Some(legacy_report)
        }
        None => None,
    };
    append_command_log(
        root,
        COMMAND_TAG,
        &format!(
            "inputs verified: checkpoint={} config={} e2w_report={} class={}",
            report.checkpoint_sha256,
            report.train_config_sha256,
            report
                .e2w_evidence
                .as_ref()
                .map_or("none", |binding| binding.report_sha256.as_str()),
            report.run_class
        ),
    )?;

    let diagnostic_device = open_diagnostic_device(&wiring.device, root)?;
    report.gpu_identity = diagnostic_device.gpu_identity.clone();
    let device = diagnostic_device.device.clone();
    report.device_is_cuda = device.is_cuda();

    let population_started = Instant::now();
    let (pairs, population) =
        generate_population(&report.spec.wiring, registered || exact_preflight)?;
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
    let (selection, rows) = select_context_wiring_rows_from(
        &pairs,
        context_len,
        gameplay_pixels,
        report.spec.scan_start_meta_episode_id,
    )?;
    if report
        .spec
        .excluded_meta_episode_ids
        .contains(&selection.meta_episode_id)
    {
        bail!(
            "selected excluded meta-episode {} (integrity failure)",
            selection.meta_episode_id
        );
    }
    let selected_disagreement_mask_sha256 =
        disagreement_mask_sha256(&rows.disagreement, gameplay_pixels)?;
    append_command_log(
        root,
        COMMAND_TAG,
        &format!(
            "selection: meta_episode_id={} position={} disagreement_pixels={} primary_row={} \
             twin_row={} primary_window={} twin_window={} primary_target={} twin_target={} \
             disagreement_mask={}",
            selection.meta_episode_id,
            selection.position,
            selection.target_disagreement_pixels,
            selection.primary_row_sha256,
            selection.twin_row_sha256,
            selection.primary_window_sha256,
            selection.twin_window_sha256,
            selection.primary_target_sha256,
            selection.twin_target_sha256,
            selected_disagreement_mask_sha256
        ),
    )?;
    report.selection = Some(selection.clone());
    report.disagreement_mask_sha256 = selected_disagreement_mask_sha256;
    let preflight_arms = match wiring.preflight_report.as_deref() {
        Some(path) => {
            let (binding, arms) = bind_confirmation_preflight(path, report, &selection)?;
            report.preflight = Some(binding);
            Some(arms)
        }
        None => None,
    };
    report.identity_root = identity_root(report)?;

    let checkpoint = wiring.checkpoint.clone();
    let arm_device = device.clone();
    let load_arm = move || load_model(&train_cfg, &checkpoint, &arm_device);
    if let Some(legacy_report) = &legacy {
        let parity = legacy_parity(
            &pairs,
            context_len,
            gameplay_pixels,
            legacy_report,
            &load_arm,
            &device,
        )?;
        append_command_log(
            root,
            COMMAND_TAG,
            &format!(
                "legacy parity: meta_episode_id={} selection_identical={} correct={} swapped={} pass={}",
                parity.meta_episode_id,
                parity.selection_identical,
                parity.correct_checkpoint0_bit_identical,
                parity.swapped_checkpoint0_bit_identical,
                parity.pass
            ),
        )?;
        let pass = parity.pass;
        let mismatches = parity.mismatches.clone();
        report.legacy_parity = Some(parity);
        if !pass {
            bail!(
                "E2W legacy checkpoint-0 parity failed (integrity failure): {}",
                mismatches.join("; ")
            );
        }
    }
    drop(pairs);

    let deadline = started + Duration::from_secs(report.spec.deadline_seconds);
    let arms_started = Instant::now();
    let run = run_confirmation_arms(
        &report.spec,
        &rows,
        &device,
        &load_arm,
        deadline,
        preflight_arms.as_ref(),
        |line| append_command_log(root, COMMAND_TAG, line),
    )?;
    report.timing.arms_seconds = arms_started.elapsed().as_secs_f64();
    report.arms = Some(run.arms);
    report.launch_parity = run.launch_parity;
    report.verdict = run.verdict;
    if let Some(failure) = run.failure {
        bail!("{failure}");
    }
    verify_no_input_drift(
        &wiring.checkpoint,
        &report.checkpoint_sha256,
        &report.provenance,
    )?;
    drop(diagnostic_device);
    report.timing.wall_seconds = started.elapsed().as_secs_f64();
    Ok(())
}

/// Re-derive E2W's meta-episode-1 selection from the same population, load a
/// fresh model and re-evaluate checkpoint 0 with this binary (the unchanged
/// batch-1 singleton evaluator shared by E2W, E2C and E2D).
pub(crate) fn legacy_parity(
    pairs: &[AugmentedTwinPair],
    context_len: usize,
    gameplay_pixels: usize,
    legacy_report: &ContextWiringReport,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
    device: &Device,
) -> Result<LegacyParity> {
    let (legacy_selection, legacy_rows) =
        select_context_wiring_rows_from(pairs, context_len, gameplay_pixels, 0)?;
    if legacy_selection.meta_episode_id != REGISTERED_E2W_META_EPISODE_ID {
        bail!(
            "the E2W scan re-derived meta-episode {} instead of {REGISTERED_E2W_META_EPISODE_ID}",
            legacy_selection.meta_episode_id
        );
    }
    let (model, _varmap) = load_arm()?;
    let directions = direction_rows(&legacy_rows);
    let current = evaluate_checkpoint(&model, &directions, 0, device)?;
    legacy_checkpoint0_parity(
        legacy_report.selection.as_ref(),
        &legacy_checkpoint0(legacy_report, "correct")?,
        &legacy_checkpoint0(legacy_report, "swapped")?,
        &legacy_selection,
        &current,
    )
}

pub fn run_p2_context_confirmation(args: P2ContextConfirmationArgs) -> Result<()> {
    let started = Instant::now();
    let wiring = &args.wiring;
    let spec =
        ContextConfirmationSpec::with_budget(wiring.pairs, wiring.max_updates, wiring.registered);
    spec.validate()?;
    let run_class = if wiring.registered {
        RUN_CLASS_REGISTERED_CONFIRMATION
    } else {
        RUN_CLASS_PREFLIGHT
    };
    let root = &wiring.output_root;
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        note: "E2C second-pair exact wiring confirmation in progress".into(),
    };
    let command = open_run_root(root, &lifecycle, COMMAND_TAG)?;
    let mut report = ContextConfirmationReport {
        schema: CONTEXT_CONFIRMATION_SCHEMA.into(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        registered: wiring.registered,
        public_data_read: false,
        checkpoints_saved: false,
        lifecycle,
        provenance: launch_provenance().clone(),
        package_version: env!("CARGO_PKG_VERSION").into(),
        command,
        device: wiring.device.clone(),
        device_is_cuda: false,
        gpu_identity: None,
        output_root: root.clone(),
        checkpoint: wiring.checkpoint.clone(),
        checkpoint_sha256: String::new(),
        train_config: wiring.train_config.clone(),
        train_config_sha256: String::new(),
        parent_evidence: None,
        e2w_evidence: None,
        legacy_parity: None,
        preflight: None,
        launch_parity: None,
        model_config: None,
        spec,
        population: None,
        selection: None,
        disagreement_mask_sha256: String::new(),
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
                    if wiring.registered {
                        "registered E2C confirmation (single-seed implementation_smoke; cannot \
                         promote a model, preserve weights, or unblock E3)"
                    } else {
                        "unregistered preflight; cannot satisfy E2C"
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
    let manifest_digest = seal_run_root(root, COMMAND_TAG, &report, &lifecycle)?;
    // Re-verify the sealed report and external manifest before returning.
    let (manifest, _) = external_manifest_paths(root)?;
    let reverified = verify_manifest(root, &manifest)?;
    verify_manifest_sidecar(&manifest, &reverified)?;
    if reverified != manifest_digest {
        bail!("sealed manifest changed during re-verification: {manifest_digest} -> {reverified}");
    }
    let sealed_report = file_sha256_hex(&root.join(REPORT_FILE))?;
    eprintln!("[{COMMAND_TAG}] sealed report sha256 {sealed_report}");
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::context_wiring::{
        clip_gradients_ordered, context_gradient_norm_of, ordered_float_parameters,
        select_context_wiring_rows, ContextComparison, DirectionEvaluation, MixedK0Invariant,
        CONTEXT_WIRING_CHECKPOINT_FAMILY,
    };
    use crate::p2::eval::twin_memorization_population;
    use crate::p2::experiment::TrainingRecipe;
    use crate::p2::model::PALETTE_SIZE;
    use crate::p2::train::{reinit_varmap_deterministic, TrainConfig};
    use candle_core::{DType, Var};
    use candle_nn::VarBuilder;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tiny_v6_model(device: &Device, seed: u64) -> Result<(WorldModel, VarMap)> {
        let mut train_cfg = TrainConfig {
            world_core_v6: true,
            data_contract_v6: true,
            ..TrainConfig::default()
        };
        train_cfg.apply_foundation_v2_recipe();
        assert_eq!(train_cfg.recipe, TrainingRecipe::FoundationV2);
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

    fn scores(raw_nll: f64, correct: usize) -> ContextArmScores {
        ContextArmScores {
            raw_softmax_nll: raw_nll,
            unimix_nll: raw_nll,
            raw_argmax_correct: correct,
            composed_argmax_correct: 0,
        }
    }

    fn comparison(l1: f64, raw_disagreement: usize) -> ContextComparison {
        ContextComparison {
            probability_l1: l1,
            latent_rms_difference: 0.0,
            context_summary_rms_difference: 0.0,
            copy_gate_mean_absolute_difference: 0.0,
            raw_argmax_disagreement_pixels: raw_disagreement,
            composed_argmax_disagreement_pixels: 0,
        }
    }

    fn mixed_k0(pass: bool) -> MixedK0Invariant {
        MixedK0Invariant {
            pass,
            latent_elements_differing: 0,
            raw_probability_elements_differing: 0,
            log_probability_elements_differing: 0,
            copy_gate_elements_differing: 0,
            raw_argmax_pixels_differing: 0,
            composed_argmax_pixels_differing: 0,
            raw_nll_bit_identical: true,
            unimix_nll_bit_identical: true,
        }
    }

    /// A checkpoint fixture with `m` disagreement pixels per direction and
    /// per-direction `(own, paired, k0)` correct counts.
    fn checkpoint(
        update: usize,
        d: f64,
        l1: f64,
        m: usize,
        counts: [(usize, usize, usize); 2],
        parameter_sha256: &str,
    ) -> ConfirmationCheckpoint {
        let directions = ["primary", "twin"]
            .iter()
            .zip(counts)
            .map(|(direction, (own, paired, k0))| DirectionEvaluation {
                direction: (*direction).into(),
                disagreement_pixels: m,
                own: scores(1.0, own),
                paired: scores(1.0 + d, paired),
                k0: scores(1.5, k0),
                own_vs_paired: comparison(l1, 0),
                own_vs_k0: comparison(0.1, 0),
                d,
                mixed_k0_invariant: mixed_k0(true),
            })
            .collect::<Vec<_>>();
        let evaluation = CheckpointEvaluation {
            update,
            directions,
            d,
            probability_l1: l1,
            raw_argmax_disagreement_pixels: 0,
            mixed_k0_invariant_pass: true,
            promotion_gate: false,
        };
        let k0_total = counts[0].2 + counts[1].2;
        ConfirmationCheckpoint {
            update,
            parameter_sha256: parameter_sha256.into(),
            shared_k0_invariant: SharedK0Invariant {
                pass: k0_total <= m,
                latent_elements_differing: 0,
                raw_probability_elements_differing: 0,
                log_probability_elements_differing: 0,
                copy_gate_elements_differing: 0,
                raw_argmax_pixels_differing: 0,
                composed_argmax_pixels_differing: 0,
                singleton_raw_nll_bit_identical: vec![true, true],
                singleton_unimix_nll_bit_identical: vec![true, true],
                singleton_shared_nll: ["primary", "twin"]
                    .iter()
                    .map(|direction| SingletonSharedNllComparison {
                        direction: (*direction).into(),
                        singleton_raw_nll: 1.0,
                        shared_raw_nll: 1.0,
                        raw_nll_abs_difference: 0.0,
                        raw_nll_ulp_distance: 0,
                        singleton_unimix_nll: 1.0,
                        shared_unimix_nll: 1.0,
                        unimix_nll_abs_difference: 0.0,
                        unimix_nll_ulp_distance: 0,
                    })
                    .collect(),
                disagreement_pixels_per_direction: m,
                k0_raw_argmax_correct_total: k0_total,
                finite_bound_holds: k0_total <= m,
            },
            exact: exact_totals(&evaluation).expect("fixture totals"),
            evaluation,
        }
    }

    fn passing_arms(update: usize, m: usize) -> [ConfirmationCheckpoint; 3] {
        [
            checkpoint(update, 2e-4, 1e-5, m, [(m, 0, 1), (m, 0, 0)], "a"),
            checkpoint(update, 3e-4, 1e-5, m, [(m, 0, 1), (m, 0, 0)], "b"),
            checkpoint(update, -2e-4, 1e-5, m, [(0, m, 1), (0, m, 0)], "s"),
        ]
    }

    fn gate(update: usize, confirmation: bool) -> ConfirmationGate {
        let spec = ContextWiringSpec::registered();
        let [a, b, s] = passing_arms(update, 2);
        let mut gate = confirmation_gate(&a, &b, &s, &spec).expect("fixture gate");
        gate.confirmation_gate = confirmation;
        gate
    }

    fn decode_rows(rows: &[(Vec<f32>, Vec<u8>)]) -> TwinContinuousDecodes {
        // `rows`: per row (target probability distribution over 2 pixels, argmax).
        TwinContinuousDecodes {
            true_predictions: rows.iter().map(|(_, argmax)| argmax.clone()).collect(),
            composed: rows.iter().map(|(_, argmax)| argmax.clone()).collect(),
            latent: rows.iter().map(|_| vec![0.25, -0.5]).collect(),
            probabilities: rows.iter().map(|(p, _)| p.clone()).collect(),
            log_probs: rows
                .iter()
                .map(|(p, _)| p.iter().map(|value| value.ln()).collect())
                .collect(),
            copy_gate: rows.iter().map(|_| vec![0.5, 0.5]).collect(),
            context_summary: rows.iter().map(|_| vec![0.0]).collect(),
        }
    }

    fn uniform_row(target_probability: f32, argmax: u8) -> (Vec<f32>, Vec<u8>) {
        let rest = (1.0 - target_probability) / (PALETTE_SIZE - 1) as f32;
        let mut pixel = vec![rest; PALETTE_SIZE];
        pixel[usize::from(argmax)] = target_probability;
        let mut probabilities = pixel.clone();
        probabilities.extend(pixel);
        (probabilities, vec![argmax, argmax])
    }

    #[test]
    fn spec_is_the_preregistered_contract() -> Result<()> {
        let spec = ContextConfirmationSpec::registered();
        spec.validate()?;
        assert!(spec.is_registered_contract());
        assert!(spec.wiring.is_registered_contract());
        assert_eq!(spec.scan_start_meta_episode_id, 2);
        assert_eq!(spec.excluded_meta_episode_ids, vec![0, 1]);
        assert_eq!(
            spec.parameter_ordering,
            ParameterOrdering::CanonicalNameSorted
        );
        assert_eq!(spec.arms, ["correct_a", "correct_b", "swapped"]);
        assert_eq!(spec.deadline_seconds, 600);
        assert_eq!(spec.launch_parity_update, 8);
        let preflight = ContextConfirmationSpec::exact_preflight();
        preflight.validate()?;
        assert!(!preflight.is_registered_contract());
        assert_eq!(preflight.deadline_seconds, 120);
        assert_eq!(preflight.wiring.checkpoint_family, vec![0, 8]);
        let mut tampered = ContextConfirmationSpec::registered();
        tampered.parameter_ordering = ParameterOrdering::VarMapIteration;
        assert!(tampered.validate().is_err());
        let mut tampered = ContextConfirmationSpec::registered();
        tampered.scan_start_meta_episode_id = 1;
        assert!(tampered.validate().is_err());
        let mut tampered = ContextConfirmationSpec::registered();
        tampered.deadline_seconds = 900;
        assert!(tampered.validate().is_err());
        assert!(ContextConfirmationSpec::with_budget(256, 12, true)
            .validate()
            .is_err());
        assert_eq!(
            REGISTERED_E2W_REPORT_SHA256,
            "0d3e54f9a8f8fa17be553cb48db44b4184259242b7ea2e782a565b3166a04eca"
        );
        assert_eq!(
            REGISTERED_E2W_MANIFEST_SHA256,
            "af133734e0e5d886e29427b4f8e06a8e94337e2d0ae8daeec861fe77321a403d"
        );
        Ok(())
    }

    #[test]
    fn selection_scans_from_meta_episode_two_and_excludes_e2w_pairs() -> Result<()> {
        let pairs = population(6)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows_from(&pairs, 16, gameplay, 2)?;
        assert!(selection.meta_episode_id >= 2);
        assert!(!CONFIRMATION_EXCLUDED_META_EPISODE_IDS.contains(&selection.meta_episode_id));
        assert_eq!(
            pairs[selection.pair_ordinal].primary.meta_episode_id,
            selection.meta_episode_id
        );
        assert_eq!(selection.pairs_scanned, selection.pair_ordinal + 1 - 2);
        assert!(selection.position >= 16);
        assert!(!rows.disagreement.is_empty());
        assert!(selection.invariants.query_inputs_identical);
        // The unrestricted E2W scan still picks the earlier pair.
        let (legacy, _) = select_context_wiring_rows(&pairs, 16, gameplay)?;
        assert!(legacy.meta_episode_id < selection.meta_episode_id);
        // Scanning the tail directly yields the same rows (ordinal differs).
        let (tail, tail_rows) = select_context_wiring_rows(&pairs[2..], 16, gameplay)?;
        assert_eq!(tail.meta_episode_id, selection.meta_episode_id);
        assert_eq!(tail.primary_row_sha256, selection.primary_row_sha256);
        assert_eq!(tail.twin_row_sha256, selection.twin_row_sha256);
        assert_eq!(tail_rows.disagreement, rows.disagreement);
        let mask_hash = disagreement_mask_sha256(&rows.disagreement, gameplay)?;
        let mut reordered = rows.disagreement.clone();
        reordered.reverse();
        assert_eq!(mask_hash, disagreement_mask_sha256(&reordered, gameplay)?);
        assert_ne!(
            mask_hash,
            disagreement_mask_sha256(&rows.disagreement, gameplay + 1)?
        );
        assert!(disagreement_mask_sha256(&[gameplay], gameplay).is_err());
        // No pair at or after the start ID fails closed.
        assert!(select_context_wiring_rows_from(&pairs, 16, gameplay, 99).is_err());
        Ok(())
    }

    #[test]
    fn ordered_parameters_and_clip_reduce_in_canonical_name_order() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let names = [
            "zeta.weight",
            "alpha.bias",
            "context_projector.weight",
            "mid.weight",
        ];
        let coefficients = [3f64, 1.0, 2.0, 0.5];
        {
            let mut data = varmap.data().lock().unwrap();
            for name in names {
                data.insert(
                    name.into(),
                    Var::from_tensor(&Tensor::ones((2, 2), DType::F32, &device)?)?,
                );
            }
            data.insert(
                "counter".into(),
                Var::from_tensor(&Tensor::zeros((2,), DType::U32, &device)?)?,
            );
        }
        let parameters = ordered_float_parameters(&varmap);
        assert_eq!(
            parameters
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "alpha.bias",
                "context_projector.weight",
                "mid.weight",
                "zeta.weight"
            ]
        );
        let names_hash = ordered_parameter_names_sha256(&parameters);
        let mut reversed = parameters.clone();
        reversed.reverse();
        assert_ne!(names_hash, ordered_parameter_names_sha256(&reversed));
        assert_eq!(names_hash, ordered_parameter_names_sha256(&parameters));
        // loss = sum_i c_i * sum(var_i): grad_i = c_i everywhere.
        let mut loss: Option<Tensor> = None;
        for (name, coefficient) in names.iter().zip(coefficients) {
            let var = parameters
                .iter()
                .find(|(candidate, _)| candidate == name)
                .map(|(_, var)| var.clone())
                .expect("named parameter");
            let term = var.as_tensor().affine(coefficient, 0.0)?.sum_all()?;
            loss = Some(match loss {
                None => term,
                Some(acc) => acc.add(&term)?,
            });
        }
        let mut grads = loss.expect("loss").backward()?;
        let context_norm =
            context_gradient_norm_of(parameters.iter().map(|(name, var)| (name, var)), &grads)?;
        assert!((context_norm - (4.0f64 * 4.0).sqrt()).abs() < 1e-6);
        let expected_norm = coefficients.iter().map(|c| c * c * 4.0).sum::<f64>().sqrt();
        let stats = clip_gradients_ordered(&mut grads, &parameters, 1.0)?.expect("finite norm");
        assert!((stats.pre_clip_norm - expected_norm).abs() < 1e-5);
        assert!((stats.scale - 1.0 / expected_norm).abs() < 1e-6);
        let zeta = parameters
            .iter()
            .find(|(name, _)| name == "zeta.weight")
            .map(|(_, var)| var.clone())
            .expect("zeta");
        let clipped = grads
            .get(zeta.as_tensor())
            .expect("zeta gradient")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(clipped
            .iter()
            .all(|value| (f64::from(*value) - 3.0 * stats.scale).abs() < 1e-6));
        // Below the threshold nothing is scaled.
        let small = Tensor::ones((2, 2), DType::F32, &device)?
            .affine(0.01, 0.0)?
            .sum_all()?;
        let mut small_grads = small.backward()?;
        let stats = clip_gradients_ordered(&mut small_grads, &parameters, 1.0)?.expect("finite");
        assert_eq!(stats.scale, 1.0);
        Ok(())
    }

    #[test]
    fn shared_k0_invariant_detects_identity_parity_and_bound_violations() -> Result<()> {
        let row = uniform_row(0.7, 3);
        let shared = decode_rows(&[row.clone(), row.clone()]);
        let disagreement = vec![0usize, 1];
        // Targets differ on both pixels: 3 (primary) and 5 (twin).
        let targets: [&[u8]; 2] = [&[3, 3], &[5, 5]];
        let primary = arm_scores_row(&shared, 0, targets[0], &disagreement)?;
        let twin = arm_scores_row(&shared, 1, targets[1], &disagreement)?;
        assert_eq!(primary.raw_argmax_correct, 2);
        assert_eq!(twin.raw_argmax_correct, 0);
        let invariant = shared_k0_invariant(
            &shared,
            &[primary.clone(), twin.clone()],
            &targets,
            &disagreement,
        )?;
        assert!(invariant.pass, "{invariant:?}");
        assert_eq!(invariant.k0_raw_argmax_correct_total, 2);
        assert!(invariant.finite_bound_holds);
        // A single differing probability bit breaks the identity.
        let mut perturbed = uniform_row(0.7, 3);
        perturbed.0[0] = f32::from_bits(perturbed.0[0].to_bits() ^ 1);
        let broken = decode_rows(&[row.clone(), perturbed]);
        let invariant = shared_k0_invariant(
            &broken,
            &[primary.clone(), twin.clone()],
            &targets,
            &disagreement,
        )?;
        assert!(!invariant.pass);
        assert!(invariant.raw_probability_elements_differing >= 1);
        // Singleton NLL parity: a different singleton NLL fails closed.
        let mut drifted = primary.clone();
        drifted.raw_softmax_nll += 1e-12;
        let invariant =
            shared_k0_invariant(&shared, &[drifted, twin.clone()], &targets, &disagreement)?;
        assert!(!invariant.pass);
        assert_eq!(invariant.singleton_raw_nll_bit_identical, vec![false, true]);
        assert!(invariant.singleton_shared_nll[0].raw_nll_abs_difference > 0.0);
        assert!(invariant.singleton_shared_nll[0].raw_nll_ulp_distance > 0);
        // Aggregate K0 correctness above m is an integrity failure.
        let mut inflated = twin.clone();
        inflated.raw_argmax_correct = 1;
        let invariant =
            shared_k0_invariant(&shared, &[primary, inflated], &targets, &disagreement)?;
        assert!(!invariant.finite_bound_holds && !invariant.pass);
        assert!(shared_k0_invariant(
            &decode_rows(&[row]),
            &[scores(1.0, 0), scores(1.0, 0)],
            &targets,
            &disagreement
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn confirmation_gate_requires_continuous_and_exact_conditions() -> Result<()> {
        let spec = ContextWiringSpec::registered();
        let [a, b, s] = passing_arms(8, 2);
        let gate = confirmation_gate(&a, &b, &s, &spec)?;
        assert!(gate.continuous_gate && gate.exact_gate && gate.confirmation_gate);
        assert!(gate.correct_own_exceeds_k0_derived);
        assert!((gate.interaction_a - 4e-4).abs() < 1e-12);
        // One correct replica failing D fails the continuous gate.
        let weak = checkpoint(8, 1e-4, 1e-5, 2, [(2, 0, 1), (2, 0, 0)], "b");
        let gate = confirmation_gate(&a, &weak, &s, &spec)?;
        assert!(!gate.continuous_gate && gate.exact_gate && !gate.confirmation_gate);
        // Own 3/4 in a correct arm fails the exact gate but not the continuous gate.
        let inexact = checkpoint(8, 3e-4, 1e-5, 2, [(2, 0, 1), (1, 0, 0)], "b");
        let gate = confirmation_gate(&a, &inexact, &s, &spec)?;
        assert!(gate.continuous_gate && !gate.exact_gate && !gate.confirmation_gate);
        // Swapped must be exactly reversed: paired 4/4 and own 0/4.
        let partial_swap = checkpoint(8, -2e-4, 1e-5, 2, [(0, 2, 1), (1, 1, 0)], "s");
        assert!(!confirmation_gate(&a, &b, &partial_swap, &spec)?.exact_gate);
        // Insensitive swapped arm (no L1 and no argmax disagreement) fails.
        let inert = checkpoint(8, -2e-4, 1e-7, 2, [(0, 2, 1), (0, 2, 0)], "s");
        assert!(!confirmation_gate(&a, &b, &inert, &spec)?.continuous_gate);
        // K0 invariants participate in the continuous gate.
        let mut leaked = s.clone();
        leaked.shared_k0_invariant.pass = false;
        let gate = confirmation_gate(&a, &b, &leaked, &spec)?;
        assert!(!gate.k0_invariants_pass && !gate.continuous_gate);
        // Different updates are an error.
        let later = checkpoint(16, -2e-4, 1e-5, 2, [(0, 2, 1), (0, 2, 0)], "s");
        assert!(confirmation_gate(&a, &b, &later, &spec).is_err());
        // Exact totals need two directions with a shared nonempty mask.
        let mut lopsided = a.evaluation.clone();
        lopsided.directions[1].disagreement_pixels = 3;
        assert!(exact_totals(&lopsided).is_err());
        Ok(())
    }

    #[test]
    fn verdict_requires_two_consecutive_confirmation_checkpoints() -> Result<()> {
        let family = CONTEXT_WIRING_CHECKPOINT_FAMILY;
        let spec = ContextConfirmationSpec::registered();
        // Non-consecutive passes never count.
        let sparse = vec![
            gate(0, true),
            gate(8, false),
            gate(16, true),
            gate(32, false),
        ];
        assert!(!confirmation_decision(&sparse, &family)?.confirmation_pass);
        // The first consecutive pair stops the run at its second checkpoint.
        let early = vec![gate(0, false), gate(8, true), gate(16, true)];
        let decision = confirmation_decision(&early, &family)?;
        assert_eq!(decision.confirmation_checkpoints, Some((8, 16)));
        assert!(decision.early_stop);
        let verdict = confirmation_verdict(early, &spec, 16)?;
        assert_eq!(verdict.outcome, OUTCOME_CONFIRMATION_PASS);
        assert_eq!(verdict.early_stop_update, Some(16));
        // A pass only at the final pair is not an early stop.
        let late = family
            .iter()
            .map(|&update| gate(update, update >= 128))
            .collect::<Vec<_>>();
        let verdict = confirmation_verdict(late, &spec, 256)?;
        assert_eq!(verdict.confirmation_checkpoints, Some((128, 256)));
        assert_eq!(verdict.early_stop_update, None);
        assert_eq!(verdict.outcome, OUTCOME_CONFIRMATION_PASS);
        // Full budget without a pair rejects; a preflight budget does not.
        let none = family
            .iter()
            .map(|&update| gate(update, false))
            .collect::<Vec<_>>();
        assert_eq!(
            confirmation_verdict(none.clone(), &spec, 256)?.outcome,
            OUTCOME_REJECT
        );
        let preflight = ContextConfirmationSpec::exact_preflight();
        assert_eq!(
            confirmation_verdict(none[..2].to_vec(), &preflight, 8)?.outcome,
            OUTCOME_PREFLIGHT
        );
        let observed = confirmation_verdict(
            vec![gate(0, true), gate(8, true)],
            &preflight,
            LAUNCH_PARITY_UPDATE,
        )?;
        assert!(!observed.confirmation_pass);
        assert_eq!(observed.outcome, OUTCOME_PREFLIGHT);
        assert_eq!(observed.confirmation_checkpoints, Some((0, 8)));
        assert_eq!(observed.early_stop_update, None);
        assert!(confirmation_decision(&[gate(8, true)], &family).is_err());
        Ok(())
    }

    fn arms_fixture() -> ConfirmationArms {
        let updates = (1..=8)
            .map(|update| UpdateRecord {
                update,
                loss: 1.0 / update as f64,
                pre_clip_gradient_norm: 2.0,
                gradient_clip_scale: 0.5,
                context_gradient_norm: 0.25,
            })
            .collect::<Vec<_>>();
        let arm =
            |name: &str, hash: &str, counts: [(usize, usize, usize); 2]| ConfirmationArmReport {
                name: name.into(),
                row_contexts: vec!["a".into(), "b".into()],
                initial_parameter_sha256: "init".into(),
                updates: updates.clone(),
                checkpoints: vec![
                    checkpoint(0, 0.0, 0.0, 2, [(1, 1, 1), (0, 0, 0)], "init"),
                    checkpoint(8, 2e-4, 1e-5, 2, counts, hash),
                ],
            };
        ConfirmationArms {
            correct_a: arm("correct_a", "h8", [(2, 0, 1), (2, 0, 0)]),
            correct_b: arm("correct_b", "h8", [(2, 0, 1), (2, 0, 0)]),
            swapped: arm("swapped", "s8", [(0, 2, 1), (0, 2, 0)]),
            parameter_ordering: ParameterOrdering::CanonicalNameSorted,
            parameter_count: 4,
            ordered_parameter_names_sha256: "names".into(),
            ordered_parameter_names_identical: true,
            initial_parameter_sha256: "init".into(),
            arms_initialized_identically: true,
            replica_comparisons: Vec::new(),
            updates_run: 8,
        }
    }

    #[test]
    fn launch_parity_passes_on_identity_and_names_every_tampered_field() -> Result<()> {
        let preflight = arms_fixture();
        let parity = launch_parity(&preflight, &preflight)?;
        assert!(parity.pass && parity.mismatches.is_empty());
        assert_eq!(parity.compared_update, 8);
        // A registered run that continued past update 8 still passes on the
        // compared prefix.
        let mut longer = preflight.clone();
        longer.updates_run = 16;
        longer.correct_a.updates.push(UpdateRecord {
            update: 9,
            loss: 0.1,
            pre_clip_gradient_norm: 1.0,
            gradient_clip_scale: 1.0,
            context_gradient_norm: 0.1,
        });
        assert!(launch_parity(&preflight, &longer)?.pass);
        // One update-record ULP flip fails.
        let mut tampered = preflight.clone();
        tampered.correct_b.updates[3].loss =
            f64::from_bits(tampered.correct_b.updates[3].loss.to_bits() ^ 1);
        let parity = launch_parity(&preflight, &tampered)?;
        assert!(!parity.pass);
        assert!(
            parity
                .mismatches
                .iter()
                .any(|line| line.contains("correct_b.update[4].loss")),
            "{:?}",
            parity.mismatches
        );
        // Parameter hash after update 8 and checkpoint fields fail.
        let mut tampered = preflight.clone();
        tampered.swapped.checkpoints[1].parameter_sha256 = "other".into();
        let parity = launch_parity(&preflight, &tampered)?;
        assert!(parity
            .mismatches
            .iter()
            .any(|line| line.contains("swapped: parameter_sha256 after update 8")));
        let mut tampered = preflight.clone();
        tampered.correct_a.checkpoints[0].evaluation.directions[1]
            .k0
            .raw_softmax_nll = 1.4999;
        let parity = launch_parity(&preflight, &tampered)?;
        assert!(
            parity.mismatches.iter().any(|line| {
                line.contains("correct_a.checkpoint[0].evaluation.directions[1].k0.raw_softmax_nll")
            }),
            "{:?}",
            parity.mismatches
        );
        // Initialization hash and ordered names fail.
        let mut tampered = preflight.clone();
        tampered.correct_a.initial_parameter_sha256 = "other".into();
        assert!(!launch_parity(&preflight, &tampered)?.pass);
        let mut tampered = preflight.clone();
        tampered.ordered_parameter_names_sha256 = "other".into();
        assert!(!launch_parity(&preflight, &tampered)?.pass);
        let mut tampered = preflight.clone();
        tampered.correct_a.row_contexts.swap(0, 1);
        let parity = launch_parity(&preflight, &tampered)?;
        assert!(parity
            .mismatches
            .iter()
            .any(|line| line.contains("correct_a: row_contexts differs")));
        // Missing checkpoint 8 fails.
        let mut tampered = preflight.clone();
        tampered.correct_b.checkpoints.pop();
        assert!(!launch_parity(&preflight, &tampered)?.pass);
        Ok(())
    }

    #[test]
    fn legacy_parity_binds_bit_identical_checkpoint0_and_registered_hashes() -> Result<()> {
        let pairs = population(2)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, _) = select_context_wiring_rows(&pairs, 16, gameplay)?;
        let legacy = checkpoint(0, 1e-6, 1e-6, 2, [(1, 1, 1), (0, 0, 0)], "x").evaluation;
        let parity =
            legacy_checkpoint0_parity(Some(&selection), &legacy, &legacy, &selection, &legacy)?;
        assert!(parity.pass);
        assert_eq!(parity.meta_episode_id, selection.meta_episode_id);
        // One float bit in the re-evaluation fails both arm comparisons.
        let mut current = legacy.clone();
        current.directions[0].own_vs_k0.latent_rms_difference = f64::from_bits(
            current.directions[0]
                .own_vs_k0
                .latent_rms_difference
                .to_bits()
                ^ 1,
        );
        let parity =
            legacy_checkpoint0_parity(Some(&selection), &legacy, &legacy, &selection, &current)?;
        assert!(!parity.pass);
        assert!(
            !parity.correct_checkpoint0_bit_identical && !parity.swapped_checkpoint0_bit_identical
        );
        assert!(
            parity.mismatches.iter().any(|line| {
                line.contains("correct.checkpoint[0].directions[0].own_vs_k0.latent_rms_difference")
            }),
            "{:?}",
            parity.mismatches
        );
        // A different legacy selection fails even with identical fields.
        let mut other = selection.clone();
        other.position += 1;
        let parity =
            legacy_checkpoint0_parity(Some(&other), &legacy, &legacy, &selection, &legacy)?;
        assert!(!parity.pass && !parity.selection_identical);
        let mut later = legacy.clone();
        later.update = 8;
        assert!(
            legacy_checkpoint0_parity(Some(&selection), &legacy, &legacy, &selection, &later)
                .is_err()
        );
        // The evidence binder rejects any report whose digest is not the sealed one.
        let parent = std::env::temp_dir().join(format!(
            "tofy-e2c-legacy-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let root = parent.join("v6-e2w-registered-fake");
        fs::create_dir_all(&root)?;
        fs::write(root.join(REPORT_FILE), b"{}")?;
        let error = bind_e2w_evidence(&root.join(REPORT_FILE))
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(REGISTERED_E2W_REPORT_SHA256), "{error}");
        fs::remove_dir_all(&parent)?;
        Ok(())
    }

    /// End-to-end CPU smoke on a tiny v6 model with the preflight budget 8:
    /// three arms initialize bit-identically with one ordered parameter list,
    /// every checkpoint passes the mixed and shared K0 invariants, the two
    /// correct replicas stay bit-identical on CPU, launch parity against the
    /// run's own record passes, and the report round-trips. No verdict value
    /// is asserted (a tiny random model is not the registered checkpoint).
    #[test]
    fn three_arm_smoke_on_cpu_records_ordered_state_invariants_and_parity() -> Result<()> {
        let device = Device::Cpu;
        let spec = ContextConfirmationSpec::with_budget(4, 8, false);
        let pairs = population(spec.wiring.pairs)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows_from(
            &pairs,
            spec.wiring.context_len,
            gameplay,
            spec.scan_start_meta_episode_id,
        )?;
        assert!(selection.meta_episode_id >= 2);
        let load_arm = || tiny_v6_model(&device, 0xE2C);
        // Generous test-only cap: the registered caps are exercised below with
        // an already-expired deadline, and this CPU smoke may share the
        // machine with the rest of the suite.
        let fresh_deadline = || Instant::now() + Duration::from_secs(3600);
        let run = run_confirmation_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            fresh_deadline(),
            None,
            |_| Ok(()),
        )?;
        assert_eq!(run.failure, None);
        assert!(run.launch_parity.is_none());
        let arms = run.arms;
        let verdict = run.verdict.expect("verdict");
        assert!(arms.arms_initialized_identically && arms.ordered_parameter_names_identical);
        assert!(arms.parameter_count > 0);
        assert_eq!(arms.updates_run, 8);
        let m = rows.disagreement.len();
        for arm in [&arms.correct_a, &arms.correct_b, &arms.swapped] {
            assert_eq!(arm.initial_parameter_sha256, arms.initial_parameter_sha256);
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
            assert_eq!(
                arm.checkpoints[0].parameter_sha256,
                arms.initial_parameter_sha256
            );
            assert_ne!(
                arm.checkpoints[1].parameter_sha256,
                arms.initial_parameter_sha256
            );
            for checkpoint in &arm.checkpoints {
                assert!(checkpoint.evaluation.mixed_k0_invariant_pass);
                assert!(checkpoint.shared_k0_invariant.pass);
                assert_eq!(checkpoint.exact.disagreement_pixels_per_direction, m);
                assert_eq!(checkpoint.exact.total_pixels, 2 * m);
                assert!(checkpoint.exact.k0_raw_argmax_correct <= m);
                assert!(checkpoint.exact.own_raw_argmax_correct <= 2 * m);
            }
        }
        // Pre-training checkpoints are shared by all arms bit for bit.
        assert_eq!(
            arms.correct_a.checkpoints[0].evaluation,
            arms.swapped.checkpoints[0].evaluation
        );
        // CPU replicas are bit-identical (within-process stability only).
        assert_eq!(arms.replica_comparisons.len(), 2);
        for comparison in &arms.replica_comparisons {
            assert!(comparison.parameter_sha256_identical && comparison.evaluation_bit_identical);
            assert!(comparison
                .directions
                .iter()
                .all(|direction| direction.own.raw_argmax_correct == 0
                    && direction.paired.raw_argmax_correct == 0
                    && direction.k0.raw_argmax_correct == 0
                    && direction.own.unimix_nll_abs == 0.0
                    && direction.own_vs_paired.latent_rms_abs == 0.0
                    && direction.own_vs_k0.composed_argmax_disagreement_pixels == 0));
        }
        assert_eq!(arms.correct_a.updates, arms.correct_b.updates);
        // Swapped diverges from the correct arms after training.
        assert_ne!(
            arms.correct_a.checkpoints[1].parameter_sha256,
            arms.swapped.checkpoints[1].parameter_sha256
        );
        assert_eq!(verdict.evaluated_checkpoints, vec![0, 8]);
        assert_eq!(verdict.updates_run, 8);
        assert_eq!(verdict.gates.len(), 2);
        assert!(verdict.gates.iter().all(|gate| gate.k0_invariants_pass));
        // The run's own record satisfies launch parity; a tampered copy does not.
        assert!(launch_parity(&arms, &arms)?.pass);
        let matched = run_confirmation_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            fresh_deadline(),
            Some(&arms),
            |_| Ok(()),
        )?;
        assert_eq!(matched.failure, None);
        assert_eq!(matched.launch_parity.map(|parity| parity.pass), Some(true));
        assert!(matched.verdict.is_some());
        let mut tampered = arms.clone();
        tampered.swapped.updates[7].pre_clip_gradient_norm += 1e-9;
        assert!(!launch_parity(&arms, &tampered)?.pass);
        // A registered-style run whose preflight disagrees stops at update 8
        // with an integrity failure while keeping its evidence.
        let stopped = run_confirmation_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            fresh_deadline(),
            Some(&tampered),
            |_| Ok(()),
        )?;
        let failure = stopped.failure.expect("parity failure");
        assert!(
            failure.contains("bit parity failed at update 8"),
            "{failure}"
        );
        assert!(stopped.verdict.is_none());
        assert_eq!(stopped.launch_parity.map(|parity| parity.pass), Some(false));
        assert_eq!(stopped.arms.updates_run, 8);
        // An already-expired internal deadline fails closed before training.
        let expired = run_confirmation_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            Instant::now(),
            None,
            |_| Ok(()),
        )?;
        let failure = expired.failure.expect("deadline failure");
        assert!(failure.contains("wall-clock cap"), "{failure}");
        assert_eq!(expired.arms.updates_run, 0);
        let json = serde_json::to_string(&(arms.clone(), verdict.clone()))?;
        let back: (ConfirmationArms, ConfirmationVerdict) = serde_json::from_str(&json)?;
        assert_eq!(back, (arms, verdict));
        Ok(())
    }
}
