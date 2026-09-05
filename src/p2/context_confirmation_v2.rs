//! `p2-context-confirmation-v2` — E2D canonical-singleton scoring with
//! semantic batch invariance
//! (`docs/research/2026-09-03-v6-local-falsifiers-prereg.md`,
//! "Post-E2 confirmation E2D").
//!
//! E2D reruns E2C's three-arm, canonically ordered diagnostic on the same
//! fixed meta-episode-2 pair with a repaired evaluator-integrity contract:
//! every reported `own`/`paired`/K0 score comes from the unchanged batch-1
//! singleton evaluator; the two-direction batch-2 K0 decode must be
//! internally bit-identical (including the context summary) and must match
//! each retained singleton decode exactly in its raw and composed argmax
//! labels over all gameplay pixels, while batch-1-versus-batch-2 NLL
//! differences are recorded descriptively and never gate; a checkpoint-0
//! in-process evaluator null control; `correct_a`/`correct_b` bit identity as
//! an integrity gate; and cross-launch parity split into `evaluator_parity`
//! and `optimizer_parity`. Both prior failed E2C preflight manifests and the
//! exact pair/row/window/target/mask digests are bound into the spec and
//! identity root. Arms, optimizer, loss, budget, checkpoint family,
//! thresholds, decision rule, deadlines and seals are E2C's, reused from
//! [`crate::p2::context_confirmation`]. The result is single-seed
//! `implementation_smoke` evidence only; a failed root carries
//! `failed_infrastructure_or_integrity` at the top level and in its lifecycle.
//!
//! Data boundary: only the registered synthetic twin population is generated.
//! Nothing in this module reads public ARC-AGI-3 data or saves checkpoints.

use crate::p2::context_confirmation::{
    arm_complete, bind_e2w_evidence, describe_mismatches, disagreement_mask_sha256, exact_totals,
    json_bit_identical, launch_parity_mismatches, legacy_parity, run_protocol_arms, ArmCheckpoint,
    ConfirmationArms, ConfirmationPreflightBinding, ConfirmationProtocol, ConfirmationVerdict,
    ContextConfirmationSpec, E2wEvidenceBinding, ExactArgmaxTotals, LegacyParity,
    P2ContextConfirmationArgs, VerdictLabels, CONFIRMATION_ARMS, LAUNCH_PARITY_UPDATE,
    OUTCOME_REJECT,
};
use crate::p2::context_wiring::{
    append_command_log, arm_scores_row, bits_differing, ensure_finite,
    evaluate_checkpoint_retaining_k0, external_manifest_paths, file_sha256_hex,
    generate_population, identity_frame_sha256, open_diagnostic_device, open_run_root,
    ordered_parameter_sha256, registered_census_matches, registered_provenance_guard,
    same_build_identity, seal_run_root, select_context_wiring_rows_from, unix_seconds,
    verify_diagnostic_inputs, verify_manifest, verify_manifest_sidecar, verify_no_input_drift, Arm,
    CheckpointEvaluation, ContextArmScores, ContextWiringRows, ContextWiringSelection,
    ContextWiringTiming, DirectionRows, GpuIdentity, LifecycleRecord, ModelConfigSummary,
    ParameterOrdering, ParentEvidenceBinding, PopulationRecord, UpdateRecord,
    CONTEXT_WIRING_MAX_UPDATES, EVIDENCE_CLASS, FAILED_EVIDENCE_CLASS, LIFECYCLE_COMPLETE,
    LIFECYCLE_FAILED, LIFECYCLE_RUNNING, REGISTERED_CHECKPOINT_SHA256, REGISTERED_PAIRS,
    REGISTERED_POPULATION_FINGERPRINT, REGISTERED_TRAIN_CONFIG_SHA256, REPORT_FILE,
    RUN_CLASS_PREFLIGHT,
};
use crate::p2::data::{gameplay_rows, FRAME_SIDE};
use crate::p2::eval::{load_model, twin_continuous_decode_rows, TwinContinuousDecodes};
use crate::p2::evidence::{launch_provenance, LaunchProvenance};
use crate::p2::model::WorldModel;
use anyhow::{bail, Context, Result};
use candle_core::Device;
use candle_nn::VarMap;
use clap::Args;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const CONTEXT_CONFIRMATION_V2_SCHEMA: &str = "p2.context_wiring_confirmation.v2";
/// The semantic label comparison covers every decoded gameplay pixel.
pub const SEMANTIC_GAMEPLAY_PIXELS: usize = 4096;
/// Arm whose checkpoint-0 evaluation is repeated as the in-process null control.
pub const NULL_CONTROL_ARM: &str = "correct_a";

/// Prior failed E2C evidence bound into the E2D spec and identity root
/// (identified, never promoted).
pub const PRIOR_E2C_PREFLIGHT_RUN_ID: &str = "v6-e2c-preflight-20260905T050507-CDT";
pub const PRIOR_E2C_PREFLIGHT_MANIFEST_SHA256: &str =
    "9338ef26e42d4b24d90f15ee54459383df7ec66fe9803d4fdeecacf992424b6f";
pub const PRIOR_E2C_DIAGNOSTIC_RUN_ID: &str = "v6-e2c-preflight-diagnostic-20260905T051337-CDT";
pub const PRIOR_E2C_DIAGNOSTIC_MANIFEST_SHA256: &str =
    "75022190ecdb765911ddaa49c48cc4a0fd76fa236089d7d8ec91b5f92320d575";

/// Fixed meta-episode-2 selection recorded by the failed E2C preflights.
pub const FIXED_META_EPISODE_ID: u64 = 2;
pub const FIXED_POSITION: usize = 20;
pub const FIXED_TARGET_DISAGREEMENT_PIXELS: usize = 28;
pub const FIXED_PRIMARY_ROW_SHA256: &str =
    "sha256:7b7de9f9ac4ce4372aefd3728626e0f73011c8b3923c520de41a3938a3352d9b";
pub const FIXED_TWIN_ROW_SHA256: &str =
    "sha256:bad5c1241a9f0364be50b7a573c6d585ae423656415bf450f00cc05273a5d1ca";
pub const FIXED_PRIMARY_WINDOW_SHA256: &str =
    "sha256:926163e3e4d0a2f7b4e67b08e19f62a75f050a4bbd6fe52fedc72b22e28f6f86";
pub const FIXED_TWIN_WINDOW_SHA256: &str =
    "sha256:7923eeba11a02c5c899249f10ab286199828a481c38549947fb56eeb1e19236b";
pub const FIXED_PRIMARY_TARGET_SHA256: &str =
    "sha256:95392091831edd5b3fac05b786c10c32eb5cff17d6ad4da19c1fa6e7b27c7792";
pub const FIXED_TWIN_TARGET_SHA256: &str =
    "sha256:6062508b01e502976238ae983190c63189760d09e77366b07737617da7c843b6";
pub const FIXED_DISAGREEMENT_MASK_SHA256: &str =
    "sha256:a725cdaaf2cb9101b2987fca0fcd328c6e40a6e2deef5e883c61e3feb52a818e";

pub const OUTCOME_SEMANTIC_CONFIRMATION_PASS: &str = "semantic_batch_confirmation_pass";
/// `second_pair` means the second selected twin pair (meta-episode 2).
pub const OUTCOME_SEMANTIC_REJECT: &str = OUTCOME_REJECT;
/// Unregistered budgets (< 256 updates) cannot reach either registered label.
pub const OUTCOME_SEMANTIC_PREFLIGHT: &str =
    "no_semantic_batch_confirmation_within_preflight_budget";

pub(crate) const RUN_CLASS_REGISTERED_SEMANTIC: &str = "registered_semantic_confirmation";
const COMMAND_TAG: &str = "p2-context-confirmation-v2";
const IDENTITY_DOMAIN: &str = "tofy.p2.context_wiring_confirmation.identity.v2";

// ---- specification -------------------------------------------------------

/// The exact pair/row/window/target/mask identity E2D must re-derive; any
/// drift is an integrity failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedSelectionDigests {
    pub meta_episode_id: u64,
    pub position: usize,
    pub target_disagreement_pixels: usize,
    pub primary_row_sha256: String,
    pub twin_row_sha256: String,
    pub primary_window_sha256: String,
    pub twin_window_sha256: String,
    pub primary_target_sha256: String,
    pub twin_target_sha256: String,
    pub disagreement_mask_sha256: String,
}

impl FixedSelectionDigests {
    pub fn registered() -> Self {
        Self {
            meta_episode_id: FIXED_META_EPISODE_ID,
            position: FIXED_POSITION,
            target_disagreement_pixels: FIXED_TARGET_DISAGREEMENT_PIXELS,
            primary_row_sha256: FIXED_PRIMARY_ROW_SHA256.into(),
            twin_row_sha256: FIXED_TWIN_ROW_SHA256.into(),
            primary_window_sha256: FIXED_PRIMARY_WINDOW_SHA256.into(),
            twin_window_sha256: FIXED_TWIN_WINDOW_SHA256.into(),
            primary_target_sha256: FIXED_PRIMARY_TARGET_SHA256.into(),
            twin_target_sha256: FIXED_TWIN_TARGET_SHA256.into(),
            disagreement_mask_sha256: FIXED_DISAGREEMENT_MASK_SHA256.into(),
        }
    }

    /// The fixed digests of an actual selection (tests, forensics).
    pub fn of(selection: &ContextWiringSelection, disagreement_mask_sha256: &str) -> Self {
        Self {
            meta_episode_id: selection.meta_episode_id,
            position: selection.position,
            target_disagreement_pixels: selection.target_disagreement_pixels,
            primary_row_sha256: selection.primary_row_sha256.clone(),
            twin_row_sha256: selection.twin_row_sha256.clone(),
            primary_window_sha256: selection.primary_window_sha256.clone(),
            twin_window_sha256: selection.twin_window_sha256.clone(),
            primary_target_sha256: selection.primary_target_sha256.clone(),
            twin_target_sha256: selection.twin_target_sha256.clone(),
            disagreement_mask_sha256: disagreement_mask_sha256.into(),
        }
    }

    /// Every field where the re-derived selection drifts from the fixed digests.
    pub fn mismatches(
        &self,
        selection: &ContextWiringSelection,
        disagreement_mask_sha256: &str,
    ) -> Vec<String> {
        let selected = Self::of(selection, disagreement_mask_sha256);
        let mut out = Vec::new();
        let mut check =
            |field: &str, fixed: &dyn std::fmt::Display, got: &dyn std::fmt::Display| {
                if fixed.to_string() != got.to_string() {
                    out.push(format!("{field}: fixed {fixed} != selected {got}"));
                }
            };
        check(
            "meta_episode_id",
            &self.meta_episode_id,
            &selected.meta_episode_id,
        );
        check("position", &self.position, &selected.position);
        check(
            "target_disagreement_pixels",
            &self.target_disagreement_pixels,
            &selected.target_disagreement_pixels,
        );
        check(
            "primary_row_sha256",
            &self.primary_row_sha256,
            &selected.primary_row_sha256,
        );
        check(
            "twin_row_sha256",
            &self.twin_row_sha256,
            &selected.twin_row_sha256,
        );
        check(
            "primary_window_sha256",
            &self.primary_window_sha256,
            &selected.primary_window_sha256,
        );
        check(
            "twin_window_sha256",
            &self.twin_window_sha256,
            &selected.twin_window_sha256,
        );
        check(
            "primary_target_sha256",
            &self.primary_target_sha256,
            &selected.primary_target_sha256,
        );
        check(
            "twin_target_sha256",
            &self.twin_target_sha256,
            &selected.twin_target_sha256,
        );
        check(
            "disagreement_mask_sha256",
            &self.disagreement_mask_sha256,
            &selected.disagreement_mask_sha256,
        );
        out
    }
}

/// A prior sealed root identified by its external manifest digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorEvidenceRecord {
    pub run_id: String,
    pub manifest_sha256: String,
    pub evidence_class: String,
    pub note: String,
}

/// Both failed E2C preflight roots (failed evaluator-integrity evidence only).
pub fn prior_e2c_evidence() -> Vec<PriorEvidenceRecord> {
    vec![
        PriorEvidenceRecord {
            run_id: PRIOR_E2C_PREFLIGHT_RUN_ID.into(),
            manifest_sha256: PRIOR_E2C_PREFLIGHT_MANIFEST_SHA256.into(),
            evidence_class: FAILED_EVIDENCE_CLASS.into(),
            note: "E2C exact preflight; stopped before update 1 on batch-1 versus batch-2 K0 NLL \
                   non-identity (not a negative model result)"
                .into(),
        },
        PriorEvidenceRecord {
            run_id: PRIOR_E2C_DIAGNOSTIC_RUN_ID.into(),
            manifest_sha256: PRIOR_E2C_DIAGNOSTIC_MANIFEST_SHA256.into(),
            evidence_class: FAILED_EVIDENCE_CLASS.into(),
            note:
                "E2C failure-forensics rerun; measured raw-NLL differences 2.1521534239177242e-6 \
                   and 4.621488707723387e-6 with otherwise bit-identical batch-2 K0 rows"
                    .into(),
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticConfirmationSpec {
    /// The complete E2C contract this run inherits (E2W optimizer/budget/
    /// thresholds, scan start, canonical ordering, arms, parity anchors,
    /// deadline).
    pub confirmation: ContextConfirmationSpec,
    pub fixed_selection: FixedSelectionDigests,
    pub prior_e2c_evidence: Vec<PriorEvidenceRecord>,
    pub semantic_gameplay_pixels: usize,
    pub null_control_arm: String,
    /// `correct_a`/`correct_b` divergence is an integrity failure.
    pub replica_identity_is_integrity_gate: bool,
    /// Batch-1 versus batch-2 NLL bit identity is descriptive only.
    pub nll_batch_identity_gates: bool,
}

impl SemanticConfirmationSpec {
    /// The exact registered E2D configuration.
    pub fn registered() -> Self {
        Self::with_budget(REGISTERED_PAIRS, CONTEXT_WIRING_MAX_UPDATES, true)
    }

    /// The bindable 256-pair, 8-update CUDA preflight configuration.
    pub fn exact_preflight() -> Self {
        Self::with_budget(REGISTERED_PAIRS, LAUNCH_PARITY_UPDATE, false)
    }

    pub fn with_budget(pairs: usize, max_updates: usize, registered: bool) -> Self {
        Self {
            confirmation: ContextConfirmationSpec::with_budget(pairs, max_updates, registered),
            fixed_selection: FixedSelectionDigests::registered(),
            prior_e2c_evidence: prior_e2c_evidence(),
            semantic_gameplay_pixels: SEMANTIC_GAMEPLAY_PIXELS,
            null_control_arm: NULL_CONTROL_ARM.into(),
            replica_identity_is_integrity_gate: true,
            nll_batch_identity_gates: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.confirmation.validate()?;
        if self.fixed_selection != FixedSelectionDigests::registered() {
            bail!(
                "E2D fixed selection digests are the meta-episode-2 identity of the E2C preflights"
            );
        }
        if self.prior_e2c_evidence != prior_e2c_evidence() {
            bail!("E2D binds exactly the two prior failed E2C preflight manifests");
        }
        if self.semantic_gameplay_pixels != SEMANTIC_GAMEPLAY_PIXELS {
            bail!("E2D compares argmax labels over all {SEMANTIC_GAMEPLAY_PIXELS} gameplay pixels");
        }
        if self.null_control_arm != NULL_CONTROL_ARM
            || !self.replica_identity_is_integrity_gate
            || self.nll_batch_identity_gates
        {
            bail!("E2D null control, replica gate and descriptive NLL contract are fixed");
        }
        Ok(())
    }

    pub fn is_registered_contract(&self) -> bool {
        *self == Self::registered()
    }
}

// ---- semantic batch invariance ------------------------------------------------

/// One retained batch-1 singleton K0 decode versus its batch-2 row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingletonBatchComparison {
    pub direction: String,
    pub gameplay_pixels: usize,
    /// Re-scoring the retained decode reproduces the reported K0 score bit for
    /// bit (the semantic comparison never re-runs or substitutes the estimator).
    pub retained_decode_reproduces_reported_score: bool,
    pub raw_argmax_pixels_differing: usize,
    pub composed_argmax_pixels_differing: usize,
    pub raw_argmax_identical: bool,
    pub composed_argmax_identical: bool,
    /// Descriptive only: never a gate nor evidence of leakage.
    pub singleton_raw_nll: f64,
    pub batch2_raw_nll: f64,
    pub raw_nll_abs_difference: f64,
    pub raw_nll_bit_identical: bool,
    pub singleton_unimix_nll: f64,
    pub batch2_unimix_nll: f64,
    pub unimix_nll_abs_difference: f64,
    pub unimix_nll_bit_identical: bool,
}

/// E2D's K0 integrity record: batch-2 internal identity (including the
/// context summary), exact raw/composed argmax label identity between each
/// retained singleton decode and its batch-2 row over all gameplay pixels,
/// descriptive NLL deltas, and the finite bound `K0 <= m` of `2m`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticK0Invariant {
    pub pass: bool,
    pub batch2_latent_elements_differing: usize,
    pub batch2_raw_probability_elements_differing: usize,
    pub batch2_log_probability_elements_differing: usize,
    pub batch2_copy_gate_elements_differing: usize,
    pub batch2_context_summary_elements_differing: usize,
    pub batch2_raw_argmax_pixels_differing: usize,
    pub batch2_composed_argmax_pixels_differing: usize,
    pub batch2_identity_holds: bool,
    pub directions: Vec<SingletonBatchComparison>,
    pub semantic_label_identity_holds: bool,
    pub retained_decodes_reproduce_reported_scores: bool,
    /// Always `false`: NLL bit identity is recorded, not gated.
    pub nll_identity_is_gate: bool,
    pub disagreement_pixels_per_direction: usize,
    pub k0_raw_argmax_correct_total: usize,
    pub finite_bound_holds: bool,
}

pub(crate) fn semantic_k0_invariant(
    shared: &TwinContinuousDecodes,
    retained: &[TwinContinuousDecodes],
    reported: &[ContextArmScores; 2],
    targets: &[&[u8]; 2],
    disagreement: &[usize],
    gameplay_pixels: usize,
) -> Result<SemanticK0Invariant> {
    if [
        shared.latent.len(),
        shared.probabilities.len(),
        shared.log_probs.len(),
        shared.copy_gate.len(),
        shared.context_summary.len(),
        shared.true_predictions.len(),
        shared.composed.len(),
    ]
    .iter()
    .any(|len| *len != 2)
    {
        bail!("batch-2 K0 decode must contain exactly the two directions");
    }
    if retained.len() != 2
        || retained.iter().any(|decode| {
            decode.true_predictions.len() != 1
                || decode.composed.len() != 1
                || decode.probabilities.len() != 1
                || decode.log_probs.len() != 1
        })
    {
        bail!("E2D needs one retained batch-1 singleton K0 decode per direction");
    }
    if disagreement.is_empty() {
        bail!("semantic K0 invariant needs a nonempty disagreement mask");
    }
    let pixels_differing = |left: &[u8], right: &[u8]| {
        if left.len() != right.len() {
            return left.len().max(right.len());
        }
        left.iter().zip(right).filter(|(a, b)| a != b).count()
    };
    let mut directions = Vec::with_capacity(2);
    for (row, (target, score)) in targets.iter().zip(reported).enumerate() {
        let single = &retained[row];
        if [
            single.true_predictions[0].len(),
            single.composed[0].len(),
            shared.true_predictions[row].len(),
            shared.composed[row].len(),
        ]
        .iter()
        .any(|len| *len != gameplay_pixels)
        {
            bail!("K0 decodes do not cover exactly {gameplay_pixels} gameplay pixels");
        }
        let recomputed = arm_scores_row(single, 0, target, disagreement)?;
        let batch2 = arm_scores_row(shared, row, target, disagreement)?;
        let raw_argmax_pixels_differing =
            pixels_differing(&single.true_predictions[0], &shared.true_predictions[row]);
        let composed_argmax_pixels_differing =
            pixels_differing(&single.composed[0], &shared.composed[row]);
        directions.push(SingletonBatchComparison {
            direction: ["primary", "twin"][row].into(),
            gameplay_pixels,
            retained_decode_reproduces_reported_score: recomputed.raw_softmax_nll.to_bits()
                == score.raw_softmax_nll.to_bits()
                && recomputed.unimix_nll.to_bits() == score.unimix_nll.to_bits()
                && recomputed.raw_argmax_correct == score.raw_argmax_correct
                && recomputed.composed_argmax_correct == score.composed_argmax_correct,
            raw_argmax_pixels_differing,
            composed_argmax_pixels_differing,
            raw_argmax_identical: raw_argmax_pixels_differing == 0,
            composed_argmax_identical: composed_argmax_pixels_differing == 0,
            singleton_raw_nll: score.raw_softmax_nll,
            batch2_raw_nll: batch2.raw_softmax_nll,
            raw_nll_abs_difference: (score.raw_softmax_nll - batch2.raw_softmax_nll).abs(),
            raw_nll_bit_identical: score.raw_softmax_nll.to_bits()
                == batch2.raw_softmax_nll.to_bits(),
            singleton_unimix_nll: score.unimix_nll,
            batch2_unimix_nll: batch2.unimix_nll,
            unimix_nll_abs_difference: (score.unimix_nll - batch2.unimix_nll).abs(),
            unimix_nll_bit_identical: score.unimix_nll.to_bits() == batch2.unimix_nll.to_bits(),
        });
    }
    let m = disagreement.len();
    let k0_raw_argmax_correct_total =
        reported[0].raw_argmax_correct + reported[1].raw_argmax_correct;
    let invariant = SemanticK0Invariant {
        pass: false,
        batch2_latent_elements_differing: bits_differing(&shared.latent[0], &shared.latent[1]),
        batch2_raw_probability_elements_differing: bits_differing(
            &shared.probabilities[0],
            &shared.probabilities[1],
        ),
        batch2_log_probability_elements_differing: bits_differing(
            &shared.log_probs[0],
            &shared.log_probs[1],
        ),
        batch2_copy_gate_elements_differing: bits_differing(
            &shared.copy_gate[0],
            &shared.copy_gate[1],
        ),
        batch2_context_summary_elements_differing: bits_differing(
            &shared.context_summary[0],
            &shared.context_summary[1],
        ),
        batch2_raw_argmax_pixels_differing: pixels_differing(
            &shared.true_predictions[0],
            &shared.true_predictions[1],
        ),
        batch2_composed_argmax_pixels_differing: pixels_differing(
            &shared.composed[0],
            &shared.composed[1],
        ),
        batch2_identity_holds: false,
        semantic_label_identity_holds: directions
            .iter()
            .all(|direction| direction.raw_argmax_identical && direction.composed_argmax_identical),
        retained_decodes_reproduce_reported_scores: directions
            .iter()
            .all(|direction| direction.retained_decode_reproduces_reported_score),
        nll_identity_is_gate: false,
        disagreement_pixels_per_direction: m,
        k0_raw_argmax_correct_total,
        finite_bound_holds: k0_raw_argmax_correct_total <= m,
        directions,
    };
    let batch2_identity_holds = invariant.batch2_latent_elements_differing == 0
        && invariant.batch2_raw_probability_elements_differing == 0
        && invariant.batch2_log_probability_elements_differing == 0
        && invariant.batch2_copy_gate_elements_differing == 0
        && invariant.batch2_context_summary_elements_differing == 0
        && invariant.batch2_raw_argmax_pixels_differing == 0
        && invariant.batch2_composed_argmax_pixels_differing == 0;
    Ok(SemanticK0Invariant {
        pass: batch2_identity_holds
            && invariant.semantic_label_identity_holds
            && invariant.retained_decodes_reproduce_reported_scores
            && invariant.finite_bound_holds,
        batch2_identity_holds,
        ..invariant
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticCheckpoint {
    pub update: usize,
    /// SHA-256 over every parameter's F32 bits in canonical name order.
    pub parameter_sha256: String,
    /// Every legacy E2W checkpoint field from the batch-1 singleton evaluator.
    pub evaluation: CheckpointEvaluation,
    pub semantic_k0_invariant: SemanticK0Invariant,
    pub exact: ExactArgmaxTotals,
}

impl ArmCheckpoint for SemanticCheckpoint {
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
        self.evaluation.mixed_k0_invariant_pass && self.semantic_k0_invariant.pass
    }
}

/// The canonical batch-1 evaluation (retaining each singleton K0 decode),
/// then the batch-2 K0 decode and the semantic invariant.
pub(crate) fn evaluate_semantic_checkpoint(
    arm: &Arm,
    directions: &[DirectionRows; 2],
    update: usize,
    device: &Device,
    gameplay_pixels: usize,
) -> Result<SemanticCheckpoint> {
    let (evaluation, retained) =
        evaluate_checkpoint_retaining_k0(&arm.model, directions, update, device)?;
    let shared = twin_continuous_decode_rows(
        &arm.model,
        &[directions[0].k0.clone(), directions[1].k0.clone()],
        device,
    )?;
    ensure_finite(&shared, "batch-2 K0 decode")?;
    let [primary, twin] = evaluation.directions.as_slice() else {
        bail!("E2D evaluation must contain exactly two query directions");
    };
    let semantic_k0_invariant = semantic_k0_invariant(
        &shared,
        &retained,
        &[primary.k0.clone(), twin.k0.clone()],
        &[&directions[0].target, &directions[1].target],
        &directions[0].disagreement,
        gameplay_pixels,
    )?;
    if !semantic_k0_invariant.pass {
        bail!(
            "{} arm checkpoint {update}: batch-2 K0 identity, semantic argmax label identity or \
             finite bound failed (integrity failure): {semantic_k0_invariant:?}",
            arm.name
        );
    }
    let exact = exact_totals(&evaluation)?;
    Ok(SemanticCheckpoint {
        update,
        parameter_sha256: ordered_parameter_sha256(&arm.parameters)?,
        evaluation,
        semantic_k0_invariant,
        exact,
    })
}

// ---- in-process controls ------------------------------------------------------

fn json_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

/// Checkpoint-0 determinism control: the same arm evaluated twice in the same
/// operation order must report bit-identical checkpoint fields. Not model
/// evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InProcessEvaluatorParity {
    pub arm: String,
    pub update: usize,
    pub first_sha256: String,
    pub second_sha256: String,
    pub pass: bool,
    pub mismatches: Vec<String>,
}

pub fn in_process_evaluator_parity(
    arm: &str,
    first: &SemanticCheckpoint,
    second: &SemanticCheckpoint,
) -> Result<InProcessEvaluatorParity> {
    if first.update != second.update {
        bail!("in-process evaluator parity compares one checkpoint");
    }
    let mismatches = if json_bit_identical(first, second)? {
        Vec::new()
    } else {
        describe_mismatches(
            &format!("{arm}.checkpoint[{}]", first.update),
            first,
            second,
        )?
    };
    Ok(InProcessEvaluatorParity {
        arm: arm.into(),
        update: first.update,
        first_sha256: json_sha256(first)?,
        second_sha256: json_sha256(second)?,
        pass: mismatches.is_empty(),
        mismatches,
    })
}

/// `correct_a` versus `correct_b` at one checkpoint as an integrity gate:
/// identical inputs and ordered update operations must give bit-identical
/// update records, state hashes and complete checkpoint records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaIntegrityGate {
    pub update: usize,
    pub update_records_bit_identical: bool,
    pub parameter_sha256_identical: bool,
    pub checkpoint_bit_identical: bool,
    pub pass: bool,
    pub mismatches: Vec<String>,
}

pub fn replica_integrity_gate(
    updates_a: &[UpdateRecord],
    updates_b: &[UpdateRecord],
    correct_a: &SemanticCheckpoint,
    correct_b: &SemanticCheckpoint,
) -> Result<ReplicaIntegrityGate> {
    if correct_a.update != correct_b.update {
        bail!("replica integrity compares one checkpoint");
    }
    let mut mismatches = Vec::new();
    let update_records_bit_identical = json_bit_identical(&updates_a, &updates_b)?;
    if !update_records_bit_identical {
        mismatches.extend(describe_mismatches(
            "correct_b.updates",
            &updates_a,
            &updates_b,
        )?);
    }
    let parameter_sha256_identical = correct_a.parameter_sha256 == correct_b.parameter_sha256;
    if !parameter_sha256_identical {
        mismatches.push(format!(
            "correct_b: parameter_sha256 at update {} differs",
            correct_a.update
        ));
    }
    let checkpoint_bit_identical = json_bit_identical(correct_a, correct_b)?;
    if !checkpoint_bit_identical {
        mismatches.extend(describe_mismatches(
            &format!("correct_b.checkpoint[{}]", correct_a.update),
            correct_a,
            correct_b,
        )?);
    }
    Ok(ReplicaIntegrityGate {
        update: correct_a.update,
        update_records_bit_identical,
        parameter_sha256_identical,
        checkpoint_bit_identical,
        pass: mismatches.is_empty(),
        mismatches,
    })
}

// ---- split cross-launch parity ------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchParityComponent {
    pub pass: bool,
    pub mismatches: Vec<String>,
}

/// E2C's registered parity set reported as two conjunctive components.
/// `evaluator_parity` binds arm initialization hashes, ordered names, the
/// in-process null control and every checkpoint-0 field; `optimizer_parity`
/// binds update records `1..=8`, update-8 parameter hashes and every
/// checkpoint-8 field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticLaunchParity {
    pub compared_update: usize,
    pub evaluator_parity: LaunchParityComponent,
    pub optimizer_parity: LaunchParityComponent,
    pub pass: bool,
}

impl SemanticLaunchParity {
    pub fn mismatches(&self) -> Vec<String> {
        [
            self.evaluator_parity.mismatches.clone(),
            self.optimizer_parity.mismatches.clone(),
        ]
        .concat()
    }
}

pub fn semantic_launch_parity(
    preflight: &ConfirmationArms<SemanticCheckpoint>,
    preflight_null_control: Option<&InProcessEvaluatorParity>,
    current: &ConfirmationArms<SemanticCheckpoint>,
    current_null_control: Option<&InProcessEvaluatorParity>,
) -> Result<SemanticLaunchParity> {
    let mut mismatches = launch_parity_mismatches(preflight, current)?;
    match (preflight_null_control, current_null_control) {
        (Some(before), Some(after)) => {
            if !before.pass || !after.pass {
                mismatches
                    .evaluator
                    .push("in_process_evaluator_parity failed on one side".into());
            }
            if !json_bit_identical(before, after)? {
                mismatches.evaluator.extend(describe_mismatches(
                    "in_process_evaluator_parity",
                    before,
                    after,
                )?);
            }
        }
        _ => mismatches
            .evaluator
            .push("in_process_evaluator_parity missing on one side".into()),
    }
    let evaluator_parity = LaunchParityComponent {
        pass: mismatches.evaluator.is_empty(),
        mismatches: mismatches.evaluator,
    };
    let optimizer_parity = LaunchParityComponent {
        pass: mismatches.optimizer.is_empty(),
        mismatches: mismatches.optimizer,
    };
    Ok(SemanticLaunchParity {
        compared_update: LAUNCH_PARITY_UPDATE,
        pass: evaluator_parity.pass && optimizer_parity.pass,
        evaluator_parity,
        optimizer_parity,
    })
}

// ---- protocol -----------------------------------------------------------------

pub(crate) const E2D_VERDICT_LABELS: VerdictLabels = VerdictLabels {
    protocol: "E2D",
    pass: OUTCOME_SEMANTIC_CONFIRMATION_PASS,
    reject: OUTCOME_SEMANTIC_REJECT,
    preflight: OUTCOME_SEMANTIC_PREFLIGHT,
    statistic: "D = raw_softmax_NLL(paired K=16) - raw_softmax_NLL(own K=16) over \
                target-disagreement pixels from the batch-1 singleton evaluator, averaged within \
                each query direction then equally across both directions; exact gate on \
                raw-argmax totals over 2m pixels",
    rule: "semantic_batch_confirmation_pass: at the same two consecutive family checkpoints \
           D_correct_a > 1e-4, D_correct_b > 1e-4, D_swapped < -1e-4, each correct-minus-swapped \
           interaction > 2e-4, every arm pooled own-vs-paired probability L1 > 1e-6 or >= 1 raw \
           argmax disagreement, all K0 integrity controls hold (same-shape mixed-K0 identity; \
           batch-2 identity including the context summary; exact raw and composed argmax label \
           identity between each retained batch-1 singleton K0 decode and its batch-2 row over \
           all 4096 gameplay pixels; finite K0 <= m of 2m; NLL bit identity is descriptive \
           only), each correct arm scores own 2m/2m and paired 0/2m raw-argmax pixels, and \
           swapped scores own 0/2m and paired 2m/2m. Stop at the second checkpoint of the first \
           such pair; otherwise run through update 256 with \
           reject_second_pair_exact_wiring_by_update_256 (second selected pair = meta-episode 2). \
           An unregistered preflight always reports \
           no_semantic_batch_confirmation_within_preflight_budget.",
};

/// Sealed exact-preflight evidence needed by the registered run's parity gate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SemanticPreflightEvidence {
    pub(crate) arms: ConfirmationArms<SemanticCheckpoint>,
    pub(crate) in_process_evaluator_parity: InProcessEvaluatorParity,
}

pub(crate) struct SemanticProtocol {
    gameplay_pixels: usize,
    null_control_arm: String,
    preflight_null_control: Option<InProcessEvaluatorParity>,
    pub(crate) in_process_evaluator_parity: Option<InProcessEvaluatorParity>,
    pub(crate) replica_integrity: Vec<ReplicaIntegrityGate>,
    pub(crate) launch_parity: Option<SemanticLaunchParity>,
}

impl ConfirmationProtocol for SemanticProtocol {
    type Checkpoint = SemanticCheckpoint;

    fn labels(&self) -> &VerdictLabels {
        &E2D_VERDICT_LABELS
    }

    fn evaluate_arm(
        &self,
        arm: &Arm,
        directions: &[DirectionRows; 2],
        update: usize,
        device: &Device,
    ) -> Result<SemanticCheckpoint> {
        evaluate_semantic_checkpoint(arm, directions, update, device, self.gameplay_pixels)
    }

    fn after_checkpoint(
        &mut self,
        update: usize,
        evaluated: &[SemanticCheckpoint],
        arms: &[Arm],
        directions: &[DirectionRows; 2],
        device: &Device,
        on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        if update == 0 {
            let index = arms
                .iter()
                .position(|arm| arm.name == self.null_control_arm)
                .ok_or_else(|| {
                    anyhow::anyhow!("null-control arm {} missing", self.null_control_arm)
                })?;
            let second = evaluate_semantic_checkpoint(
                &arms[index],
                directions,
                update,
                device,
                self.gameplay_pixels,
            )?;
            let parity =
                in_process_evaluator_parity(&self.null_control_arm, &evaluated[index], &second)?;
            on_progress(&format!(
                "in-process evaluator parity ({}, checkpoint 0): pass={} mismatches={}",
                parity.arm,
                parity.pass,
                parity.mismatches.len()
            ))?;
            let (pass, mismatches) = (parity.pass, parity.mismatches.clone());
            self.in_process_evaluator_parity = Some(parity);
            if !pass {
                bail!(
                    "in-process evaluator parity failed at checkpoint 0 (integrity failure; \
                     stopped before update 1, not a negative model result): {}",
                    mismatches.join("; ")
                );
            }
        }
        let gate = replica_integrity_gate(
            &arms[0].updates,
            &arms[1].updates,
            &evaluated[0],
            &evaluated[1],
        )?;
        on_progress(&format!(
            "replica integrity (correct_a vs correct_b, checkpoint {update}): updates={} state={} \
             checkpoint={} pass={}",
            gate.update_records_bit_identical,
            gate.parameter_sha256_identical,
            gate.checkpoint_bit_identical,
            gate.pass
        ))?;
        let (pass, mismatches) = (gate.pass, gate.mismatches.clone());
        self.replica_integrity.push(gate);
        if !pass {
            bail!(
                "correct_a and correct_b diverged at checkpoint {update} despite identical inputs \
                 and ordered updates (integrity failure, not a negative model result): {}",
                mismatches.join("; ")
            );
        }
        Ok(())
    }

    fn launch_parity(
        &mut self,
        preflight: &ConfirmationArms<SemanticCheckpoint>,
        current: &ConfirmationArms<SemanticCheckpoint>,
        on_progress: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<Vec<String>> {
        let parity = semantic_launch_parity(
            preflight,
            self.preflight_null_control.as_ref(),
            current,
            self.in_process_evaluator_parity.as_ref(),
        )?;
        on_progress(&format!(
            "launch parity components: evaluator_parity={} ({} mismatches) optimizer_parity={} \
             ({} mismatches)",
            parity.evaluator_parity.pass,
            parity.evaluator_parity.mismatches.len(),
            parity.optimizer_parity.pass,
            parity.optimizer_parity.mismatches.len()
        ))?;
        let mismatches = parity.mismatches();
        self.launch_parity = Some(parity);
        Ok(mismatches)
    }
}

/// Result of the E2D three-arm loop; see
/// [`crate::p2::context_confirmation::ConfirmationRun`].
pub(crate) struct SemanticRun {
    pub(crate) arms: ConfirmationArms<SemanticCheckpoint>,
    pub(crate) in_process_evaluator_parity: Option<InProcessEvaluatorParity>,
    pub(crate) replica_integrity: Vec<ReplicaIntegrityGate>,
    pub(crate) launch_parity: Option<SemanticLaunchParity>,
    pub(crate) verdict: Option<ConfirmationVerdict>,
    pub(crate) failure: Option<String>,
}

pub(crate) fn run_semantic_arms(
    spec: &SemanticConfirmationSpec,
    rows: &ContextWiringRows,
    device: &Device,
    load_arm: &dyn Fn() -> Result<(WorldModel, VarMap)>,
    deadline: Instant,
    preflight: Option<&SemanticPreflightEvidence>,
    on_progress: impl FnMut(&str) -> Result<()>,
) -> Result<SemanticRun> {
    spec.validate()?;
    let mut protocol = SemanticProtocol {
        gameplay_pixels: spec.semantic_gameplay_pixels,
        null_control_arm: spec.null_control_arm.clone(),
        preflight_null_control: preflight
            .map(|evidence| evidence.in_process_evaluator_parity.clone()),
        in_process_evaluator_parity: None,
        replica_integrity: Vec::new(),
        launch_parity: None,
    };
    let run = run_protocol_arms(
        &spec.confirmation,
        rows,
        device,
        load_arm,
        deadline,
        preflight.map(|evidence| &evidence.arms),
        &mut protocol,
        on_progress,
    )?;
    Ok(SemanticRun {
        arms: run.arms,
        in_process_evaluator_parity: protocol.in_process_evaluator_parity,
        replica_integrity: protocol.replica_integrity,
        launch_parity: protocol.launch_parity,
        verdict: run.verdict,
        failure: run.failure,
    })
}

// ---- report, provenance, CLI -------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticConfirmationReport {
    pub schema: String,
    /// `implementation_smoke` while healthy; `failed_infrastructure_or_integrity`
    /// at the top level of any failed root.
    pub evidence_class: String,
    /// `registered_semantic_confirmation` or `unregistered_preflight`; a
    /// preflight cannot satisfy E2D.
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
    pub launch_parity: Option<SemanticLaunchParity>,
    pub model_config: Option<ModelConfigSummary>,
    pub spec: SemanticConfirmationSpec,
    pub population: Option<PopulationRecord>,
    pub selection: Option<ContextWiringSelection>,
    pub disagreement_mask_sha256: String,
    /// `Some` whenever the registered 256-pair population was scanned.
    pub fixed_selection_match: Option<bool>,
    pub in_process_evaluator_parity: Option<InProcessEvaluatorParity>,
    pub replica_integrity: Vec<ReplicaIntegrityGate>,
    pub arms: Option<ConfirmationArms<SemanticCheckpoint>>,
    pub verdict: Option<ConfirmationVerdict>,
    pub timing: ContextWiringTiming,
    /// Domain-separated SHA-256 over checkpoint, config, binary, source, GPU,
    /// parent/E2W/preflight bindings, population fingerprint, the complete
    /// selection (pair/row/window/target hashes), the mask digest and the spec
    /// (fixed digests and prior E2C manifests included).
    pub identity_root: String,
    pub error: Option<String>,
}

fn identity_root(report: &SemanticConfirmationReport) -> Result<String> {
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
        ("selection", serde_json::to_vec(&report.selection)?),
        (
            "disagreement_mask_sha256",
            report.disagreement_mask_sha256.as_bytes().to_vec(),
        ),
        ("spec", serde_json::to_vec(&report.spec)?),
    ])
}

/// Verify the exact 256-pair, 8-update CUDA E2D preflight of this binary and
/// return its binding plus the arm and null-control evidence for the
/// update-8 split parity gate.
fn bind_semantic_preflight(
    path: &Path,
    current: &SemanticConfirmationReport,
    selection: &ContextWiringSelection,
) -> Result<(ConfirmationPreflightBinding, SemanticPreflightEvidence)> {
    let report_path = fs::canonicalize(path)
        .with_context(|| format!("canonicalize preflight report {}", path.display()))?;
    let bytes = fs::read(&report_path)
        .with_context(|| format!("read preflight report {}", report_path.display()))?;
    let preflight: SemanticConfirmationReport =
        serde_json::from_slice(&bytes).context("parse E2D preflight report")?;
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
    if preflight.schema != CONTEXT_CONFIRMATION_V2_SCHEMA
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
        bail!("preflight report is not a completed, clean, CUDA E2D preflight");
    }
    if preflight.spec != SemanticConfirmationSpec::exact_preflight() {
        bail!("registered E2D requires the exact 256-pair, 8-update preflight");
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
        || preflight.fixed_selection_match != Some(true)
    {
        bail!("preflight did not re-derive the fixed E2D selection and mask");
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
    let in_process_evaluator_parity = preflight
        .in_process_evaluator_parity
        .clone()
        .filter(|parity| parity.pass && parity.update == 0 && parity.arm == NULL_CONTROL_ARM)
        .ok_or_else(|| anyhow::anyhow!("preflight did not pass the in-process evaluator parity"))?;
    if preflight.replica_integrity.len() != 2
        || preflight.replica_integrity.iter().any(|gate| !gate.pass)
    {
        bail!("preflight did not pass the correct_a/correct_b replica integrity gate");
    }
    let arms = preflight
        .arms
        .clone()
        .ok_or_else(|| anyhow::anyhow!("preflight has no arm record"))?;
    if preflight.verdict.as_ref().map(|verdict| {
        (
            verdict.updates_run,
            verdict.evaluated_checkpoints.as_slice(),
            verdict.outcome.as_str(),
        ) == (
            LAUNCH_PARITY_UPDATE,
            [0, LAUNCH_PARITY_UPDATE].as_slice(),
            OUTCOME_SEMANTIC_PREFLIGHT,
        )
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
            max_updates: preflight.spec.confirmation.wiring.max_updates,
            selection: selection.clone(),
            disagreement_mask_sha256: preflight.disagreement_mask_sha256,
        },
        SemanticPreflightEvidence {
            arms,
            in_process_evaluator_parity,
        },
    ))
}

/// `p2-context-confirmation-v2` — E2D canonical-singleton scoring with
/// semantic batch invariance (implementation smoke). Same inputs as
/// `p2-context-confirmation`; the preflight report must be an E2D preflight.
#[derive(Debug, Clone, Args)]
pub struct P2ContextConfirmationV2Args {
    #[command(flatten)]
    pub confirmation: P2ContextConfirmationArgs,
}

fn run_inner(
    args: &P2ContextConfirmationV2Args,
    report: &mut SemanticConfirmationReport,
    root: &Path,
    started: Instant,
) -> Result<()> {
    let args = &args.confirmation;
    let wiring = &args.wiring;
    let registered = wiring.registered;
    let exact_preflight = !registered && report.spec == SemanticConfirmationSpec::exact_preflight();
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
            "inputs verified: checkpoint={} config={} e2w_report={} prior_e2c_manifests={:?} class={}",
            report.checkpoint_sha256,
            report.train_config_sha256,
            report
                .e2w_evidence
                .as_ref()
                .map_or("none", |binding| binding.report_sha256.as_str()),
            report
                .spec
                .prior_e2c_evidence
                .iter()
                .map(|prior| prior.manifest_sha256.as_str())
                .collect::<Vec<_>>(),
            report.run_class
        ),
    )?;

    let diagnostic_device = open_diagnostic_device(&wiring.device, root)?;
    report.gpu_identity = diagnostic_device.gpu_identity.clone();
    let device = diagnostic_device.device.clone();
    report.device_is_cuda = device.is_cuda();

    let population_started = Instant::now();
    let (pairs, population) = generate_population(
        &report.spec.confirmation.wiring,
        registered || exact_preflight,
    )?;
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
    let registered_population = population.pairs == REGISTERED_PAIRS;
    report.population = Some(population);

    let gameplay_pixels = gameplay_rows(train_cfg.world_core_v6) * FRAME_SIDE;
    if gameplay_pixels != report.spec.semantic_gameplay_pixels {
        bail!(
            "decoded gameplay region has {gameplay_pixels} pixels, not the fixed {}",
            report.spec.semantic_gameplay_pixels
        );
    }
    let (selection, rows) = select_context_wiring_rows_from(
        &pairs,
        context_len,
        gameplay_pixels,
        report.spec.confirmation.scan_start_meta_episode_id,
    )?;
    if report
        .spec
        .confirmation
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
    report.disagreement_mask_sha256 = selected_disagreement_mask_sha256.clone();
    if registered_population {
        let drift = report
            .spec
            .fixed_selection
            .mismatches(&selection, &selected_disagreement_mask_sha256);
        report.fixed_selection_match = Some(drift.is_empty());
        append_command_log(
            root,
            COMMAND_TAG,
            &format!(
                "fixed selection: match={} drift={:?}",
                drift.is_empty(),
                drift
            ),
        )?;
        if !drift.is_empty() {
            bail!(
                "selection drifted from the fixed E2D pair/row/window/target/mask digests \
                 (integrity failure): {}",
                drift.join("; ")
            );
        }
    }
    let preflight_evidence = match wiring.preflight_report.as_deref() {
        Some(path) => {
            let (binding, evidence) = bind_semantic_preflight(path, report, &selection)?;
            report.preflight = Some(binding);
            Some(evidence)
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

    let deadline = started + Duration::from_secs(report.spec.confirmation.deadline_seconds);
    let arms_started = Instant::now();
    let run = run_semantic_arms(
        &report.spec,
        &rows,
        &device,
        &load_arm,
        deadline,
        preflight_evidence.as_ref(),
        |line| append_command_log(root, COMMAND_TAG, line),
    )?;
    report.timing.arms_seconds = arms_started.elapsed().as_secs_f64();
    report.arms = Some(run.arms);
    report.in_process_evaluator_parity = run.in_process_evaluator_parity;
    report.replica_integrity = run.replica_integrity;
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

pub fn run_p2_context_confirmation_v2(args: P2ContextConfirmationV2Args) -> Result<()> {
    let started = Instant::now();
    let wiring = &args.confirmation.wiring;
    let spec =
        SemanticConfirmationSpec::with_budget(wiring.pairs, wiring.max_updates, wiring.registered);
    spec.validate()?;
    let run_class = if wiring.registered {
        RUN_CLASS_REGISTERED_SEMANTIC
    } else {
        RUN_CLASS_PREFLIGHT
    };
    let root = &wiring.output_root;
    let lifecycle = LifecycleRecord {
        state: LIFECYCLE_RUNNING.into(),
        unix_seconds: unix_seconds(),
        evidence_class: EVIDENCE_CLASS.into(),
        run_class: run_class.into(),
        note: "E2D canonical-singleton semantic batch-invariance confirmation in progress".into(),
    };
    let command = open_run_root(root, &lifecycle, COMMAND_TAG)?;
    let mut report = SemanticConfirmationReport {
        schema: CONTEXT_CONFIRMATION_V2_SCHEMA.into(),
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
        fixed_selection_match: None,
        in_process_evaluator_parity: None,
        replica_integrity: Vec::new(),
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
                        "registered E2D confirmation (single-seed implementation_smoke; cannot \
                         promote a model, preserve weights, or unblock E3)"
                    } else {
                        "unregistered preflight; cannot satisfy E2D"
                    }
                ),
                None => "completed without a verdict".into(),
            },
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
    use crate::p2::context_confirmation::{
        confirmation_gate, confirmation_verdict_labeled, shared_k0_invariant,
        ConfirmationArmReport, ConfirmationGate, OUTCOME_CONFIRMATION_PASS,
    };
    use crate::p2::context_wiring::{
        ContextComparison, ContextWiringSpec, DirectionEvaluation, MixedK0Invariant,
        CONTEXT_WIRING_CHECKPOINT_FAMILY,
    };
    use crate::p2::data::AugmentedTwinPair;
    use crate::p2::eval::twin_memorization_population;
    use crate::p2::experiment::TrainingRecipe;
    use crate::p2::model::PALETTE_SIZE;
    use crate::p2::train::{reinit_varmap_deterministic, TrainConfig};
    use candle_core::DType;
    use candle_nn::VarBuilder;

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

    fn decode_rows(rows: &[(Vec<f32>, Vec<u8>)]) -> TwinContinuousDecodes {
        // `rows`: per row (distribution over 2 pixels x palette, argmax labels).
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

    fn scores(raw_nll: f64, correct: usize) -> ContextArmScores {
        ContextArmScores {
            raw_softmax_nll: raw_nll,
            unimix_nll: raw_nll,
            raw_argmax_correct: correct,
            composed_argmax_correct: 0,
        }
    }

    fn comparison(l1: f64) -> ContextComparison {
        ContextComparison {
            probability_l1: l1,
            latent_rms_difference: 0.0,
            context_summary_rms_difference: 0.0,
            copy_gate_mean_absolute_difference: 0.0,
            raw_argmax_disagreement_pixels: 0,
            composed_argmax_disagreement_pixels: 0,
        }
    }

    fn semantic_fixture(m: usize, k0_total: usize) -> SemanticK0Invariant {
        let direction = |name: &str| SingletonBatchComparison {
            direction: name.into(),
            gameplay_pixels: SEMANTIC_GAMEPLAY_PIXELS,
            retained_decode_reproduces_reported_score: true,
            raw_argmax_pixels_differing: 0,
            composed_argmax_pixels_differing: 0,
            raw_argmax_identical: true,
            composed_argmax_identical: true,
            singleton_raw_nll: 1.5,
            batch2_raw_nll: 1.5000000000000002,
            raw_nll_abs_difference: 2.220446049250313e-16,
            raw_nll_bit_identical: false,
            singleton_unimix_nll: 1.5,
            batch2_unimix_nll: 1.5,
            unimix_nll_abs_difference: 0.0,
            unimix_nll_bit_identical: true,
        };
        SemanticK0Invariant {
            pass: k0_total <= m,
            batch2_latent_elements_differing: 0,
            batch2_raw_probability_elements_differing: 0,
            batch2_log_probability_elements_differing: 0,
            batch2_copy_gate_elements_differing: 0,
            batch2_context_summary_elements_differing: 0,
            batch2_raw_argmax_pixels_differing: 0,
            batch2_composed_argmax_pixels_differing: 0,
            batch2_identity_holds: true,
            directions: vec![direction("primary"), direction("twin")],
            semantic_label_identity_holds: true,
            retained_decodes_reproduce_reported_scores: true,
            nll_identity_is_gate: false,
            disagreement_pixels_per_direction: m,
            k0_raw_argmax_correct_total: k0_total,
            finite_bound_holds: k0_total <= m,
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
    ) -> SemanticCheckpoint {
        let directions = ["primary", "twin"]
            .iter()
            .zip(counts)
            .map(|(direction, (own, paired, k0))| DirectionEvaluation {
                direction: (*direction).into(),
                disagreement_pixels: m,
                own: scores(1.0, own),
                paired: scores(1.0 + d, paired),
                k0: scores(1.5, k0),
                own_vs_paired: comparison(l1),
                own_vs_k0: comparison(0.1),
                d,
                mixed_k0_invariant: MixedK0Invariant {
                    pass: true,
                    latent_elements_differing: 0,
                    raw_probability_elements_differing: 0,
                    log_probability_elements_differing: 0,
                    copy_gate_elements_differing: 0,
                    raw_argmax_pixels_differing: 0,
                    composed_argmax_pixels_differing: 0,
                    raw_nll_bit_identical: true,
                    unimix_nll_bit_identical: true,
                },
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
        SemanticCheckpoint {
            update,
            parameter_sha256: parameter_sha256.into(),
            semantic_k0_invariant: semantic_fixture(m, counts[0].2 + counts[1].2),
            exact: exact_totals(&evaluation).expect("fixture totals"),
            evaluation,
        }
    }

    fn passing_arms(update: usize, m: usize) -> [SemanticCheckpoint; 3] {
        [
            checkpoint(update, 2e-4, 1e-5, m, [(m, 0, 1), (m, 0, 0)], "a"),
            checkpoint(update, 3e-4, 1e-5, m, [(m, 0, 1), (m, 0, 0)], "b"),
            checkpoint(update, -2e-4, 1e-5, m, [(0, m, 1), (0, m, 0)], "s"),
        ]
    }

    fn gate(update: usize, confirmation: bool) -> ConfirmationGate {
        let [a, b, s] = passing_arms(update, 2);
        let mut gate =
            confirmation_gate(&a, &b, &s, &ContextWiringSpec::registered()).expect("gate");
        gate.confirmation_gate = confirmation;
        gate
    }

    fn updates(count: usize) -> Vec<UpdateRecord> {
        (1..=count)
            .map(|update| UpdateRecord {
                update,
                loss: 1.0 / update as f64,
                pre_clip_gradient_norm: 2.0,
                gradient_clip_scale: 0.5,
                context_gradient_norm: 0.25,
            })
            .collect()
    }

    fn arms_fixture() -> ConfirmationArms<SemanticCheckpoint> {
        let arm =
            |name: &str, hash: &str, counts: [(usize, usize, usize); 2]| ConfirmationArmReport {
                name: name.into(),
                row_contexts: vec!["a".into(), "b".into()],
                initial_parameter_sha256: "init".into(),
                updates: updates(8),
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

    fn null_control() -> InProcessEvaluatorParity {
        let first = checkpoint(0, 0.0, 0.0, 2, [(1, 1, 1), (0, 0, 0)], "init");
        in_process_evaluator_parity(NULL_CONTROL_ARM, &first, &first).expect("null control")
    }

    fn valid_registered_provenance() -> LaunchProvenance {
        let mut provenance = LaunchProvenance::unknown(Path::new("/test/tofy"));
        provenance.source_revision = "8fd1cea5".repeat(5);
        provenance.source_revision_origin = "embedded-build:git".into();
        provenance.source_dirty = Some(false);
        provenance.source_pushed = Some(true);
        provenance.build_command = crate::p2::context_wiring::REGISTERED_BUILD_COMMAND.into();
        provenance.cargo_features = vec!["cudnn".into()];
        provenance.cargo_profile = "release".into();
        provenance.cargo_target = "x86_64-unknown-linux-gnu".into();
        provenance.binary_sha256 = "sha256:test-binary".into();
        provenance.candle_graph_revision =
            crate::p2::context_wiring::REGISTERED_CANDLE_GRAPH_REVISION.into();
        provenance.candle_graph_dirty = Some(false);
        provenance.candle_graph_pushed = Some(true);
        provenance.runtime_checkout.revision = provenance.source_revision.clone();
        provenance.runtime_checkout.dirty = Some(false);
        provenance
    }

    fn exact_preflight_report(root: &Path) -> Result<SemanticConfirmationReport> {
        let spec = SemanticConfirmationSpec::exact_preflight();
        let (pairs, population) = generate_population(&spec.confirmation.wiring, true)?;
        let gameplay_pixels = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows_from(
            &pairs,
            population.context_len,
            gameplay_pixels,
            spec.confirmation.scan_start_meta_episode_id,
        )?;
        let disagreement_mask_sha256 =
            disagreement_mask_sha256(&rows.disagreement, gameplay_pixels)?;
        let selection_drift = spec
            .fixed_selection
            .mismatches(&selection, &disagreement_mask_sha256);
        assert!(selection_drift.is_empty(), "{selection_drift:?}");

        let arms = arms_fixture();
        let gates = (0..2)
            .map(|checkpoint| {
                confirmation_gate(
                    &arms.correct_a.checkpoints[checkpoint],
                    &arms.correct_b.checkpoints[checkpoint],
                    &arms.swapped.checkpoints[checkpoint],
                    &spec.confirmation.wiring,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let verdict = confirmation_verdict_labeled(
            gates,
            &spec.confirmation,
            LAUNCH_PARITY_UPDATE,
            &E2D_VERDICT_LABELS,
        )?;
        assert_eq!(verdict.outcome, OUTCOME_SEMANTIC_PREFLIGHT);
        let replica_integrity = vec![
            replica_integrity_gate(
                &arms.correct_a.updates[..0],
                &arms.correct_b.updates[..0],
                &arms.correct_a.checkpoints[0],
                &arms.correct_b.checkpoints[0],
            )?,
            replica_integrity_gate(
                &arms.correct_a.updates,
                &arms.correct_b.updates,
                &arms.correct_a.checkpoints[1],
                &arms.correct_b.checkpoints[1],
            )?,
        ];
        assert!(replica_integrity.iter().all(|gate| gate.pass));

        let provenance = valid_registered_provenance();
        let parent_evidence = ParentEvidenceBinding {
            source_run_identity: crate::p2::context_wiring::REGISTERED_PARENT_RUN_ID.into(),
            root: PathBuf::from("/test/parent"),
            manifest: PathBuf::from("/test/parent.files.sha256"),
            manifest_sha256: crate::p2::context_wiring::REGISTERED_PARENT_MANIFEST_SHA256.into(),
            checkpoint_relative: PathBuf::from("checkpoints/step-000000004096/ema.safetensors"),
            config_relative: PathBuf::from("config.json"),
        };
        let e2w_evidence = E2wEvidenceBinding {
            report: PathBuf::from("/test/e2w/report.json"),
            root: PathBuf::from("/test/e2w"),
            manifest: PathBuf::from("/test/e2w.files.sha256"),
            report_sha256: crate::p2::context_confirmation::REGISTERED_E2W_REPORT_SHA256.into(),
            manifest_sha256: crate::p2::context_confirmation::REGISTERED_E2W_MANIFEST_SHA256.into(),
            source_revision: provenance.source_revision.clone(),
            binary_sha256: provenance.binary_sha256.clone(),
            outcome: crate::p2::context_confirmation::REGISTERED_E2W_OUTCOME.into(),
            selection: selection.clone(),
        };
        let legacy_parity = LegacyParity {
            meta_episode_id: crate::p2::context_confirmation::REGISTERED_E2W_META_EPISODE_ID,
            selection_identical: true,
            correct_checkpoint0_bit_identical: true,
            swapped_checkpoint0_bit_identical: true,
            pass: true,
            mismatches: Vec::new(),
        };
        let lifecycle = LifecycleRecord {
            state: LIFECYCLE_COMPLETE.into(),
            unix_seconds: unix_seconds(),
            evidence_class: EVIDENCE_CLASS.into(),
            run_class: RUN_CLASS_PREFLIGHT.into(),
            note: "test E2D preflight".into(),
        };
        let mut report = SemanticConfirmationReport {
            schema: CONTEXT_CONFIRMATION_V2_SCHEMA.into(),
            evidence_class: EVIDENCE_CLASS.into(),
            run_class: RUN_CLASS_PREFLIGHT.into(),
            registered: false,
            public_data_read: false,
            checkpoints_saved: false,
            lifecycle,
            provenance,
            package_version: env!("CARGO_PKG_VERSION").into(),
            command: vec!["tofy".into(), COMMAND_TAG.into()],
            device: "cuda".into(),
            device_is_cuda: true,
            gpu_identity: Some(GpuIdentity {
                ordinal: 0,
                name: "test-gpu".into(),
                uuid: "GPU-test".into(),
                memory_total_mib: "1".into(),
                driver_version: "test".into(),
            }),
            output_root: root.to_path_buf(),
            checkpoint: PathBuf::from("/test/checkpoint"),
            checkpoint_sha256: REGISTERED_CHECKPOINT_SHA256.into(),
            train_config: PathBuf::from("/test/config"),
            train_config_sha256: REGISTERED_TRAIN_CONFIG_SHA256.into(),
            parent_evidence: Some(parent_evidence),
            e2w_evidence: Some(e2w_evidence),
            legacy_parity: Some(legacy_parity),
            preflight: None,
            launch_parity: None,
            model_config: None,
            spec,
            population: Some(population),
            selection: Some(selection),
            disagreement_mask_sha256,
            fixed_selection_match: Some(true),
            in_process_evaluator_parity: Some(null_control()),
            replica_integrity,
            arms: Some(arms),
            verdict: Some(verdict),
            timing: ContextWiringTiming {
                population_seconds: 1.0,
                arms_seconds: 1.0,
                wall_seconds: 2.0,
            },
            identity_root: String::new(),
            error: None,
        };
        report.identity_root = identity_root(&report)?;
        Ok(report)
    }

    fn seal_preflight_fixture(
        parent: &Path,
        name: &str,
        mut report: SemanticConfirmationReport,
    ) -> Result<PathBuf> {
        let root = parent.join(name);
        fs::create_dir(&root)?;
        report.output_root = root.clone();
        report.identity_root = identity_root(&report)?;
        let lifecycle = report.lifecycle.clone();
        seal_run_root(&root, "e2d-test", &report, &lifecycle)?;
        Ok(root.join(REPORT_FILE))
    }

    #[test]
    fn spec_binds_prior_e2c_manifests_and_exact_fixed_digests() -> Result<()> {
        let spec = SemanticConfirmationSpec::registered();
        spec.validate()?;
        assert!(spec.is_registered_contract());
        assert!(spec.confirmation.is_registered_contract());
        assert_eq!(spec.confirmation.deadline_seconds, 600);
        assert_eq!(spec.semantic_gameplay_pixels, 4096);
        assert_eq!(gameplay_rows(true) * FRAME_SIDE, SEMANTIC_GAMEPLAY_PIXELS);
        assert_eq!(spec.null_control_arm, "correct_a");
        assert!(spec.replica_identity_is_integrity_gate && !spec.nll_batch_identity_gates);
        let fixed = &spec.fixed_selection;
        assert_eq!(
            (
                fixed.meta_episode_id,
                fixed.position,
                fixed.target_disagreement_pixels
            ),
            (2, 20, 28)
        );
        assert_eq!(
            fixed.primary_row_sha256,
            "sha256:7b7de9f9ac4ce4372aefd3728626e0f73011c8b3923c520de41a3938a3352d9b"
        );
        assert_eq!(
            fixed.twin_row_sha256,
            "sha256:bad5c1241a9f0364be50b7a573c6d585ae423656415bf450f00cc05273a5d1ca"
        );
        assert_eq!(
            fixed.primary_window_sha256,
            "sha256:926163e3e4d0a2f7b4e67b08e19f62a75f050a4bbd6fe52fedc72b22e28f6f86"
        );
        assert_eq!(
            fixed.twin_window_sha256,
            "sha256:7923eeba11a02c5c899249f10ab286199828a481c38549947fb56eeb1e19236b"
        );
        assert_eq!(
            fixed.primary_target_sha256,
            "sha256:95392091831edd5b3fac05b786c10c32eb5cff17d6ad4da19c1fa6e7b27c7792"
        );
        assert_eq!(
            fixed.twin_target_sha256,
            "sha256:6062508b01e502976238ae983190c63189760d09e77366b07737617da7c843b6"
        );
        assert_eq!(
            fixed.disagreement_mask_sha256,
            "sha256:a725cdaaf2cb9101b2987fca0fcd328c6e40a6e2deef5e883c61e3feb52a818e"
        );
        assert_eq!(
            spec.prior_e2c_evidence
                .iter()
                .map(|prior| (prior.run_id.as_str(), prior.manifest_sha256.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    "v6-e2c-preflight-20260905T050507-CDT",
                    "9338ef26e42d4b24d90f15ee54459383df7ec66fe9803d4fdeecacf992424b6f"
                ),
                (
                    "v6-e2c-preflight-diagnostic-20260905T051337-CDT",
                    "75022190ecdb765911ddaa49c48cc4a0fd76fa236089d7d8ec91b5f92320d575"
                ),
            ]
        );
        assert!(spec
            .prior_e2c_evidence
            .iter()
            .all(|prior| prior.evidence_class == FAILED_EVIDENCE_CLASS));
        let preflight = SemanticConfirmationSpec::exact_preflight();
        preflight.validate()?;
        assert!(!preflight.is_registered_contract());
        assert_eq!(preflight.confirmation.deadline_seconds, 120);
        assert_eq!(preflight.confirmation.wiring.checkpoint_family, vec![0, 8]);
        let mut tampered = SemanticConfirmationSpec::registered();
        tampered
            .fixed_selection
            .twin_target_sha256
            .replace_range(..1, "0");
        assert!(tampered.validate().is_err());
        let mut tampered = SemanticConfirmationSpec::registered();
        tampered.prior_e2c_evidence.pop();
        assert!(tampered.validate().is_err());
        let mut tampered = SemanticConfirmationSpec::registered();
        tampered.semantic_gameplay_pixels = 28;
        assert!(tampered.validate().is_err());
        let mut tampered = SemanticConfirmationSpec::registered();
        tampered.nll_batch_identity_gates = true;
        assert!(tampered.validate().is_err());
        let mut tampered = SemanticConfirmationSpec::registered();
        tampered.confirmation.parameter_ordering = ParameterOrdering::VarMapIteration;
        assert!(tampered.validate().is_err());
        assert_eq!(
            OUTCOME_SEMANTIC_REJECT,
            "reject_second_pair_exact_wiring_by_update_256"
        );
        assert_eq!(
            CONTEXT_CONFIRMATION_V2_SCHEMA,
            "p2.context_wiring_confirmation.v2"
        );
        Ok(())
    }

    #[test]
    fn sealed_preflight_binding_rejects_each_new_e2d_admission_guard() -> Result<()> {
        let parent = std::env::temp_dir().join(format!(
            "tofy-e2d-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&parent)?;
        let template = exact_preflight_report(&parent.join("template"))?;
        let current = template.clone();
        let selection = template.selection.clone().expect("fixed selection");

        let valid = seal_preflight_fixture(&parent, "valid", template.clone())?;
        let (binding, evidence) = bind_semantic_preflight(&valid, &current, &selection)?;
        assert_eq!(binding.max_updates, LAUNCH_PARITY_UPDATE);
        assert_eq!(evidence.arms.updates_run, LAUNCH_PARITY_UPDATE);
        assert!(evidence.in_process_evaluator_parity.pass);

        let mut wrong_outcome = template.clone();
        wrong_outcome.verdict.as_mut().expect("verdict").outcome =
            OUTCOME_SEMANTIC_CONFIRMATION_PASS.into();
        let path = seal_preflight_fixture(&parent, "wrong-outcome", wrong_outcome)?;
        assert!(bind_semantic_preflight(&path, &current, &selection).is_err());

        let mut selection_not_fixed = template.clone();
        selection_not_fixed.fixed_selection_match = Some(false);
        let path = seal_preflight_fixture(&parent, "selection-not-fixed", selection_not_fixed)?;
        assert!(bind_semantic_preflight(&path, &current, &selection).is_err());

        let mut missing_null_control = template.clone();
        missing_null_control.in_process_evaluator_parity = None;
        let path = seal_preflight_fixture(&parent, "missing-null-control", missing_null_control)?;
        assert!(bind_semantic_preflight(&path, &current, &selection).is_err());

        let mut incomplete_replica_gates = template;
        incomplete_replica_gates.replica_integrity.pop();
        let path = seal_preflight_fixture(
            &parent,
            "incomplete-replica-gates",
            incomplete_replica_gates,
        )?;
        assert!(bind_semantic_preflight(&path, &current, &selection).is_err());

        fs::remove_dir_all(parent)?;
        Ok(())
    }

    #[test]
    fn failed_root_is_sealed_with_failed_classes_and_fixed_inputs_bind_identity() -> Result<()> {
        let parent = std::env::temp_dir().join(format!(
            "tofy-e2d-failed-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&parent)?;
        let root = parent.join("failed");
        let error = run_p2_context_confirmation_v2(P2ContextConfirmationV2Args {
            confirmation: P2ContextConfirmationArgs {
                wiring: crate::p2::context_wiring::P2ContextWiringArgs {
                    checkpoint: parent.join("missing-checkpoint.safetensors"),
                    train_config: parent.join("missing-config.json"),
                    parent_root: parent.join("missing-parent"),
                    parent_manifest: parent.join("missing-parent.files.sha256"),
                    preflight_report: None,
                    output_root: root.clone(),
                    device: "cpu".into(),
                    registered: false,
                    max_updates: LAUNCH_PARITY_UPDATE,
                    pairs: 4,
                },
                e2w_report: None,
            },
        })
        .expect_err("missing inputs must fail after opening and sealing the root");
        assert!(error.to_string().contains("missing-config"));

        let report: SemanticConfirmationReport =
            serde_json::from_slice(&fs::read(root.join(REPORT_FILE))?)?;
        assert_eq!(report.evidence_class, FAILED_EVIDENCE_CLASS);
        assert_eq!(report.lifecycle.evidence_class, FAILED_EVIDENCE_CLASS);
        assert_eq!(report.lifecycle.state, LIFECYCLE_FAILED);
        assert!(report.error.is_some());
        assert!(!report.public_data_read && !report.checkpoints_saved);
        assert_eq!(identity_root(&report)?, report.identity_root);
        let (manifest, _) = external_manifest_paths(&root)?;
        let manifest_sha256 = verify_manifest(&root, &manifest)?;
        verify_manifest_sidecar(&manifest, &manifest_sha256)?;

        let baseline = identity_root(&report)?;
        let mut changed_prior = report.clone();
        changed_prior.spec.prior_e2c_evidence[0]
            .manifest_sha256
            .push('0');
        assert_ne!(identity_root(&changed_prior)?, baseline);
        let mut changed_fixed_selection = report.clone();
        changed_fixed_selection
            .spec
            .fixed_selection
            .primary_row_sha256
            .push('0');
        assert_ne!(identity_root(&changed_fixed_selection)?, baseline);
        let mut changed_mask = report;
        changed_mask.disagreement_mask_sha256 = "different-mask".into();
        assert_ne!(identity_root(&changed_mask)?, baseline);

        fs::remove_dir_all(parent)?;
        Ok(())
    }

    #[test]
    fn fixed_selection_names_every_drifting_digest() -> Result<()> {
        let pairs = population(6)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows_from(&pairs, 16, gameplay, 2)?;
        let mask = disagreement_mask_sha256(&rows.disagreement, gameplay)?;
        let matched = FixedSelectionDigests::of(&selection, &mask);
        assert!(matched.mismatches(&selection, &mask).is_empty());
        assert!(FixedSelectionDigests::registered()
            .mismatches(&selection, &mask)
            .is_empty());

        type DigestField = for<'a> fn(&'a mut FixedSelectionDigests) -> &'a mut String;
        let fields: &[(&str, DigestField)] = &[
            ("primary_row_sha256", |fixed| &mut fixed.primary_row_sha256),
            ("twin_row_sha256", |fixed| &mut fixed.twin_row_sha256),
            ("primary_window_sha256", |fixed| {
                &mut fixed.primary_window_sha256
            }),
            ("twin_window_sha256", |fixed| &mut fixed.twin_window_sha256),
            ("primary_target_sha256", |fixed| {
                &mut fixed.primary_target_sha256
            }),
            ("twin_target_sha256", |fixed| &mut fixed.twin_target_sha256),
            ("disagreement_mask_sha256", |fixed| {
                &mut fixed.disagreement_mask_sha256
            }),
        ];
        for (field, select) in fields {
            let mut one_digest = matched.clone();
            select(&mut one_digest).replace_range(..1, "f");
            let drift = one_digest.mismatches(&selection, &mask);
            assert_eq!(drift.len(), 1, "{field}: {drift:?}");
            assert!(drift[0].starts_with(field), "{field}: {drift:?}");
        }
        Ok(())
    }

    #[test]
    fn semantic_invariant_gates_labels_and_identity_but_not_nll() -> Result<()> {
        let row = uniform_row(0.7, 3);
        let retained = [
            decode_rows(std::slice::from_ref(&row)),
            decode_rows(std::slice::from_ref(&row)),
        ];
        let shared = decode_rows(&[row.clone(), row.clone()]);
        let disagreement = vec![0usize, 1];
        let targets: [&[u8]; 2] = [&[3, 3], &[5, 5]];
        let reported = [
            arm_scores_row(&retained[0], 0, targets[0], &disagreement)?,
            arm_scores_row(&retained[1], 0, targets[1], &disagreement)?,
        ];
        assert_eq!(reported[0].raw_argmax_correct, 2);
        assert_eq!(reported[1].raw_argmax_correct, 0);
        let invariant =
            semantic_k0_invariant(&shared, &retained, &reported, &targets, &disagreement, 2)?;
        assert!(invariant.pass, "{invariant:?}");
        assert!(invariant.batch2_identity_holds && invariant.semantic_label_identity_holds);
        assert!(invariant.retained_decodes_reproduce_reported_scores);
        assert!(!invariant.nll_identity_is_gate);
        assert_eq!(invariant.k0_raw_argmax_correct_total, 2);
        assert!(invariant.finite_bound_holds);
        assert!(invariant
            .directions
            .iter()
            .all(|direction| direction.raw_nll_abs_difference == 0.0
                && direction.raw_nll_bit_identical
                && direction.unimix_nll_bit_identical
                && direction.gameplay_pixels == 2));
        // Batch-2 NLL drift with unchanged labels is descriptive: E2D passes,
        // E2C's strict singleton NLL identity fails on the same inputs.
        let drifted_row = uniform_row(0.7001, 3);
        let drifted = decode_rows(&[drifted_row.clone(), drifted_row]);
        let invariant =
            semantic_k0_invariant(&drifted, &retained, &reported, &targets, &disagreement, 2)?;
        assert!(invariant.pass, "{invariant:?}");
        assert!(invariant.directions[0].raw_nll_abs_difference > 0.0);
        assert!(invariant.directions[0].unimix_nll_abs_difference > 0.0);
        assert!(!invariant.directions[0].raw_nll_bit_identical);
        assert!(!invariant.directions[0].unimix_nll_bit_identical);
        let strict = shared_k0_invariant(&drifted, &reported, &targets, &disagreement)?;
        assert!(!strict.pass);
        assert_eq!(strict.singleton_raw_nll_bit_identical, vec![false, false]);
        assert!(strict.latent_elements_differing == 0 && strict.finite_bound_holds);
        // A context-summary bit differing between the two batch-2 rows fails.
        let mut summary = decode_rows(&[row.clone(), row.clone()]);
        summary.context_summary[1][0] = 1.0;
        let invariant =
            semantic_k0_invariant(&summary, &retained, &reported, &targets, &disagreement, 2)?;
        assert!(!invariant.pass && !invariant.batch2_identity_holds);
        assert_eq!(invariant.batch2_context_summary_elements_differing, 1);
        // A label change outside the disagreement mask (scores unchanged) fails
        // the semantic identity: the comparison covers every gameplay pixel.
        let mask_one = vec![0usize];
        let reported_one = [
            arm_scores_row(&retained[0], 0, targets[0], &mask_one)?,
            arm_scores_row(&retained[1], 0, targets[1], &mask_one)?,
        ];
        let mut relabeled = decode_rows(&[row.clone(), row.clone()]);
        relabeled.true_predictions[0][1] = 4;
        relabeled.true_predictions[1][1] = 4;
        let invariant =
            semantic_k0_invariant(&relabeled, &retained, &reported_one, &targets, &mask_one, 2)?;
        assert!(invariant.batch2_identity_holds);
        assert!(!invariant.semantic_label_identity_holds && !invariant.pass);
        assert_eq!(invariant.directions[0].raw_argmax_pixels_differing, 1);
        assert_eq!(invariant.directions[1].composed_argmax_pixels_differing, 0);
        let mut composed = decode_rows(&[row.clone(), row.clone()]);
        composed.composed[0][0] = 9;
        composed.composed[1][0] = 9;
        let invariant =
            semantic_k0_invariant(&composed, &retained, &reported, &targets, &disagreement, 2)?;
        assert!(!invariant.pass && !invariant.directions[0].composed_argmax_identical);
        // A reported score the retained decode does not reproduce fails.
        let mut substituted = reported.clone();
        substituted[1].raw_softmax_nll += 1e-12;
        let invariant =
            semantic_k0_invariant(&shared, &retained, &substituted, &targets, &disagreement, 2)?;
        assert!(!invariant.pass && !invariant.retained_decodes_reproduce_reported_scores);
        assert!(!invariant.directions[1].retained_decode_reproduces_reported_score);
        // Aggregate K0 correctness above m violates the finite bound.
        let mut inflated = reported.clone();
        inflated[1].raw_argmax_correct = 1;
        let invariant =
            semantic_k0_invariant(&shared, &retained, &inflated, &targets, &disagreement, 2)?;
        assert!(!invariant.finite_bound_holds && !invariant.pass);
        // Shape guards.
        assert!(semantic_k0_invariant(
            &shared,
            &retained,
            &reported,
            &targets,
            &disagreement,
            4096
        )
        .is_err());
        assert!(semantic_k0_invariant(
            &decode_rows(std::slice::from_ref(&row)),
            &retained,
            &reported,
            &targets,
            &disagreement,
            2
        )
        .is_err());
        assert!(semantic_k0_invariant(
            &shared,
            &retained[..1],
            &reported,
            &targets,
            &disagreement,
            2
        )
        .is_err());
        assert!(semantic_k0_invariant(&shared, &retained, &reported, &targets, &[], 2).is_err());
        Ok(())
    }

    #[test]
    fn in_process_parity_and_replica_gate_detect_single_bit_drift() -> Result<()> {
        let first = checkpoint(0, 0.0, 0.0, 2, [(1, 1, 1), (0, 0, 0)], "init");
        let parity = in_process_evaluator_parity("correct_a", &first, &first)?;
        assert!(parity.pass && parity.mismatches.is_empty());
        assert_eq!(parity.first_sha256, parity.second_sha256);
        assert_eq!((parity.arm.as_str(), parity.update), ("correct_a", 0));
        let mut second = first.clone();
        second.semantic_k0_invariant.directions[1].batch2_raw_nll = f64::from_bits(
            second.semantic_k0_invariant.directions[1]
                .batch2_raw_nll
                .to_bits()
                ^ 1,
        );
        let parity = in_process_evaluator_parity("correct_a", &first, &second)?;
        assert!(!parity.pass);
        assert_ne!(parity.first_sha256, parity.second_sha256);
        assert!(
            parity.mismatches.iter().any(|line| line.contains(
                "correct_a.checkpoint[0].semantic_k0_invariant.directions[1].batch2_raw_nll"
            )),
            "{:?}",
            parity.mismatches
        );
        let mut later = first.clone();
        later.update = 8;
        assert!(in_process_evaluator_parity("correct_a", &first, &later).is_err());
        // Replica gate: identical replicas pass; any divergence fails.
        let a = checkpoint(8, 2e-4, 1e-5, 2, [(2, 0, 1), (2, 0, 0)], "h8");
        let gate = replica_integrity_gate(&updates(8), &updates(8), &a, &a)?;
        assert!(gate.pass && gate.update_records_bit_identical);
        assert!(gate.parameter_sha256_identical && gate.checkpoint_bit_identical);
        let mut drifted_updates = updates(8);
        drifted_updates[4].loss = f64::from_bits(drifted_updates[4].loss.to_bits() ^ 1);
        let gate = replica_integrity_gate(&updates(8), &drifted_updates, &a, &a)?;
        assert!(!gate.pass && !gate.update_records_bit_identical);
        assert!(gate
            .mismatches
            .iter()
            .any(|line| line.contains("correct_b.updates[4].loss")));
        let mut b = a.clone();
        b.parameter_sha256 = "other".into();
        let gate = replica_integrity_gate(&updates(8), &updates(8), &a, &b)?;
        assert!(!gate.pass && !gate.parameter_sha256_identical);
        assert!(!gate.checkpoint_bit_identical);
        let mut b = a.clone();
        b.evaluation.directions[0].own.raw_softmax_nll += 1e-9;
        let gate = replica_integrity_gate(&updates(8), &updates(8), &a, &b)?;
        assert!(!gate.pass && gate.parameter_sha256_identical && !gate.checkpoint_bit_identical);
        assert!(gate.mismatches.iter().any(|line| {
            line.contains("correct_b.checkpoint[8].evaluation.directions[0].own.raw_softmax_nll")
        }));
        Ok(())
    }

    #[test]
    fn split_launch_parity_attributes_mismatches_to_evaluator_or_optimizer() -> Result<()> {
        let preflight = arms_fixture();
        let control = null_control();
        let parity =
            semantic_launch_parity(&preflight, Some(&control), &preflight, Some(&control))?;
        assert!(parity.pass && parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        assert_eq!(parity.compared_update, 8);
        assert!(parity.mismatches().is_empty());
        // Checkpoint-0 field drift is an evaluator mismatch only.
        let mut tampered = preflight.clone();
        tampered.correct_a.checkpoints[0].evaluation.directions[1]
            .k0
            .raw_softmax_nll = 1.4999;
        let parity = semantic_launch_parity(&preflight, Some(&control), &tampered, Some(&control))?;
        assert!(!parity.pass && !parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        assert!(parity.evaluator_parity.mismatches.iter().any(|line| {
            line.contains("correct_a.checkpoint[0].evaluation.directions[1].k0.raw_softmax_nll")
        }));
        // Update-record, update-8 hash and checkpoint-8 drift are optimizer only.
        let mut tampered = preflight.clone();
        tampered.correct_b.updates[3].loss =
            f64::from_bits(tampered.correct_b.updates[3].loss.to_bits() ^ 1);
        tampered.swapped.checkpoints[1].parameter_sha256 = "other".into();
        let parity = semantic_launch_parity(&preflight, Some(&control), &tampered, Some(&control))?;
        assert!(parity.evaluator_parity.pass && !parity.optimizer_parity.pass && !parity.pass);
        assert!(parity
            .optimizer_parity
            .mismatches
            .iter()
            .any(|line| line.contains("correct_b.update[4].loss")));
        assert!(parity
            .optimizer_parity
            .mismatches
            .iter()
            .any(|line| line.contains("swapped: parameter_sha256 after update 8")));
        // Initialization hash and ordered names are evaluator mismatches.
        let mut tampered = preflight.clone();
        tampered.correct_a.initial_parameter_sha256 = "other".into();
        tampered.ordered_parameter_names_sha256 = "other".into();
        let parity = semantic_launch_parity(&preflight, Some(&control), &tampered, Some(&control))?;
        assert!(!parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        // The in-process null control binds into evaluator parity.
        let mut other_control = control.clone();
        other_control.second_sha256 = "other".into();
        let parity =
            semantic_launch_parity(&preflight, Some(&control), &preflight, Some(&other_control))?;
        assert!(!parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        assert!(parity
            .evaluator_parity
            .mismatches
            .iter()
            .any(|line| line.contains("in_process_evaluator_parity.second_sha256")));
        let parity = semantic_launch_parity(&preflight, Some(&control), &preflight, None)?;
        assert!(!parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        let mut failed_control = control.clone();
        failed_control.pass = false;
        let parity = semantic_launch_parity(
            &preflight,
            Some(&failed_control),
            &preflight,
            Some(&failed_control),
        )?;
        assert!(!parity.evaluator_parity.pass);
        Ok(())
    }

    #[test]
    fn verdict_uses_e2d_labels_and_the_shared_decision_rule() -> Result<()> {
        let family = CONTEXT_WIRING_CHECKPOINT_FAMILY;
        let spec = SemanticConfirmationSpec::registered();
        let early = vec![gate(0, false), gate(8, true), gate(16, true)];
        let verdict =
            confirmation_verdict_labeled(early, &spec.confirmation, 16, &E2D_VERDICT_LABELS)?;
        assert_eq!(verdict.outcome, OUTCOME_SEMANTIC_CONFIRMATION_PASS);
        assert_eq!(verdict.outcome, "semantic_batch_confirmation_pass");
        assert_ne!(verdict.outcome, OUTCOME_CONFIRMATION_PASS);
        assert_eq!(verdict.confirmation_checkpoints, Some((8, 16)));
        assert_eq!(verdict.early_stop_update, Some(16));
        assert!(verdict.rule.contains("4096 gameplay pixels"));
        assert!(verdict
            .rule
            .contains("NLL bit identity is descriptive only"));
        let none = family
            .iter()
            .map(|&update| gate(update, false))
            .collect::<Vec<_>>();
        let verdict = confirmation_verdict_labeled(
            none.clone(),
            &spec.confirmation,
            256,
            &E2D_VERDICT_LABELS,
        )?;
        assert_eq!(verdict.outcome, OUTCOME_SEMANTIC_REJECT);
        let preflight = SemanticConfirmationSpec::exact_preflight();
        let observed = confirmation_verdict_labeled(
            vec![gate(0, true), gate(8, true)],
            &preflight.confirmation,
            8,
            &E2D_VERDICT_LABELS,
        )?;
        assert!(!observed.confirmation_pass);
        assert_eq!(observed.outcome, OUTCOME_SEMANTIC_PREFLIGHT);
        assert_eq!(
            observed.outcome,
            "no_semantic_batch_confirmation_within_preflight_budget"
        );
        assert_eq!(observed.confirmation_checkpoints, Some((0, 8)));
        assert_eq!(observed.early_stop_update, None);
        assert!(observed.note.contains("cannot satisfy E2D"));
        assert_eq!(
            confirmation_verdict_labeled(
                none[..2].to_vec(),
                &preflight.confirmation,
                8,
                &E2D_VERDICT_LABELS
            )?
            .outcome,
            OUTCOME_SEMANTIC_PREFLIGHT
        );
        Ok(())
    }

    /// End-to-end CPU smoke on a tiny v6 model with the preflight budget 8:
    /// every checkpoint retains its singleton K0 decodes and passes the
    /// semantic invariant over all 4,096 gameplay pixels, the checkpoint-0
    /// null control and both replica integrity gates pass, split launch
    /// parity against the run's own record passes, and tampered preflight
    /// evidence stops the run at update 8 in the attributed component. No
    /// verdict value is asserted (a tiny random model is not the registered
    /// checkpoint).
    #[test]
    fn three_arm_semantic_smoke_on_cpu_records_controls_and_split_parity() -> Result<()> {
        let device = Device::Cpu;
        let spec = SemanticConfirmationSpec::with_budget(4, 8, false);
        let pairs = population(spec.confirmation.wiring.pairs)?;
        let gameplay = gameplay_rows(true) * FRAME_SIDE;
        let (selection, rows) = select_context_wiring_rows_from(
            &pairs,
            spec.confirmation.wiring.context_len,
            gameplay,
            spec.confirmation.scan_start_meta_episode_id,
        )?;
        assert!(selection.meta_episode_id >= 2);
        let load_arm = || tiny_v6_model(&device, 0xE2D);
        let fresh_deadline = || Instant::now() + Duration::from_secs(3600);
        let run = run_semantic_arms(
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
        let control = run.in_process_evaluator_parity.expect("null control");
        assert!(control.pass, "{control:?}");
        assert_eq!((control.arm.as_str(), control.update), ("correct_a", 0));
        assert_eq!(run.replica_integrity.len(), 2);
        assert!(run.replica_integrity.iter().all(|gate| gate.pass));
        assert_eq!(
            run.replica_integrity
                .iter()
                .map(|gate| gate.update)
                .collect::<Vec<_>>(),
            vec![0, 8]
        );
        let arms = run.arms;
        let verdict = run.verdict.expect("verdict");
        assert!(arms.arms_initialized_identically && arms.ordered_parameter_names_identical);
        assert_eq!(arms.updates_run, 8);
        let m = rows.disagreement.len();
        for arm in [&arms.correct_a, &arms.correct_b, &arms.swapped] {
            assert_eq!(arm.updates.len(), 8);
            assert_eq!(
                arm.checkpoints
                    .iter()
                    .map(|checkpoint| checkpoint.update)
                    .collect::<Vec<_>>(),
                vec![0, 8]
            );
            for checkpoint in &arm.checkpoints {
                assert!(checkpoint.evaluation.mixed_k0_invariant_pass);
                let semantic = &checkpoint.semantic_k0_invariant;
                assert!(semantic.pass, "{semantic:?}");
                assert!(semantic.batch2_identity_holds && semantic.semantic_label_identity_holds);
                assert!(semantic.retained_decodes_reproduce_reported_scores);
                assert!(!semantic.nll_identity_is_gate);
                assert_eq!(semantic.disagreement_pixels_per_direction, m);
                assert!(semantic.finite_bound_holds && semantic.k0_raw_argmax_correct_total <= m);
                assert_eq!(semantic.directions.len(), 2);
                assert!(semantic
                    .directions
                    .iter()
                    .all(
                        |direction| direction.gameplay_pixels == SEMANTIC_GAMEPLAY_PIXELS
                            && direction.raw_argmax_identical
                            && direction.composed_argmax_identical
                            && direction.raw_nll_abs_difference.is_finite()
                            && direction.unimix_nll_abs_difference.is_finite()
                    ));
                assert_eq!(checkpoint.exact.total_pixels, 2 * m);
            }
        }
        assert_eq!(arms.correct_a.checkpoints, arms.correct_b.checkpoints);
        assert_eq!(arms.correct_a.updates, arms.correct_b.updates);
        assert_ne!(
            arms.correct_a.checkpoints[1].parameter_sha256,
            arms.swapped.checkpoints[1].parameter_sha256
        );
        assert_eq!(verdict.evaluated_checkpoints, vec![0, 8]);
        assert_eq!(verdict.outcome, OUTCOME_SEMANTIC_PREFLIGHT);
        assert!(verdict.gates.iter().all(|gate| gate.k0_invariants_pass));
        // The run's own record satisfies split launch parity.
        let evidence = SemanticPreflightEvidence {
            arms: arms.clone(),
            in_process_evaluator_parity: control.clone(),
        };
        let matched = run_semantic_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            fresh_deadline(),
            Some(&evidence),
            |_| Ok(()),
        )?;
        assert_eq!(matched.failure, None);
        let parity = matched.launch_parity.expect("launch parity");
        assert!(parity.pass && parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        assert!(matched.verdict.is_some());
        // Tampered optimizer evidence stops the run at update 8 with the
        // mismatch attributed to optimizer parity only.
        let mut tampered = evidence.clone();
        tampered.arms.swapped.updates[7].pre_clip_gradient_norm += 1e-9;
        let stopped = run_semantic_arms(
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
        let parity = stopped.launch_parity.expect("launch parity");
        assert!(!parity.pass && parity.evaluator_parity.pass && !parity.optimizer_parity.pass);
        assert_eq!(stopped.arms.updates_run, 8);
        // A tampered null control is an evaluator-parity failure.
        let mut tampered = evidence.clone();
        tampered.in_process_evaluator_parity.first_sha256 = "other".into();
        let stopped = run_semantic_arms(
            &spec,
            &rows,
            &device,
            &load_arm,
            fresh_deadline(),
            Some(&tampered),
            |_| Ok(()),
        )?;
        assert!(stopped.failure.is_some());
        let parity = stopped.launch_parity.expect("launch parity");
        assert!(!parity.evaluator_parity.pass && parity.optimizer_parity.pass);
        // Non-identical arm initialization fails closed before any evidence.
        let counter = std::cell::Cell::new(0u64);
        let drifting_load = || {
            counter.set(counter.get() + 1);
            tiny_v6_model(&device, 0xE2D + counter.get())
        };
        let error = run_semantic_arms(
            &spec,
            &rows,
            &device,
            &drifting_load,
            fresh_deadline(),
            None,
            |_| Ok(()),
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
        assert!(
            error.contains("did not initialize bit-identically"),
            "{error}"
        );
        let json = serde_json::to_string(&(arms.clone(), verdict.clone(), control.clone()))?;
        let back: (
            ConfirmationArms<SemanticCheckpoint>,
            ConfirmationVerdict,
            InProcessEvaluatorParity,
        ) = serde_json::from_str(&json)?;
        assert_eq!(back, (arms, verdict, control));
        Ok(())
    }
}
