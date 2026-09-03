//! Foundation-v2 semantic evaluation at the exact-decoder seam.

use crate::p2::data::{
    apply_episode_operator, gameplay_rows, ArcAction, ArcFrame, TransitionSample,
    V5SampleProvenance, FRAME_SIDE,
};
use crate::p2::model::{WorldModel, PALETTE_SIZE};
use crate::p2::train::{action_tensors_from_samples, batch_from_samples};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SEMANTIC_EVAL_SCHEMA: &str = "p2.semantic_eval.v3";

const NONCOMPARABLE_CONTENT_MASKS: [&str; 5] = [
    "content",
    "padding",
    "changed_content",
    "unchanged_content",
    "unchanged_padding",
];

const MASKS: [&str; 9] = [
    "content",
    "padding",
    "foreground",
    "changed",
    "unchanged",
    "changed_content",
    "unchanged_content",
    "unchanged_padding",
    "gameplay",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMaskMetrics {
    pub pixels: usize,
    pub transitions: usize,
    pub mean_nll: Option<f64>,
    pub pixel_accuracy: Option<f64>,
    /// Every pixel in the named mask was correct for the transition.
    pub exact_transition_accuracy: Option<f64>,
    /// Each transition has equal weight, independent of board size.
    pub mean_transition_accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDecoderMetrics {
    pub masks: BTreeMap<String, SemanticMaskMetrics>,
    /// Derived from `unchanged_content`: the fraction of unchanged board
    /// content pixels the decode edits. Padding cannot dilute this metric.
    #[serde(default)]
    pub false_edit_rate: Option<f64>,
    /// Derived from `unchanged_content`: fraction of transitions with at least
    /// one false edit in the content rectangle.
    #[serde(default)]
    pub false_edit_transition_rate: Option<f64>,
    /// Padding hallucination rate, kept separate from content false edits.
    #[serde(default)]
    pub padding_false_edit_rate: Option<f64>,
    /// Fraction of transitions with at least one padding hallucination.
    #[serde(default)]
    pub padding_false_edit_transition_rate: Option<f64>,
}

impl SemanticDecoderMetrics {
    fn from_masks(masks: BTreeMap<String, SemanticMaskMetrics>) -> Self {
        let unchanged = masks.get("unchanged_content");
        let padding = masks.get("unchanged_padding");
        Self {
            false_edit_rate: unchanged
                .and_then(|mask| mask.pixel_accuracy)
                .map(|accuracy| 1.0 - accuracy),
            false_edit_transition_rate: unchanged
                .and_then(|mask| mask.exact_transition_accuracy)
                .map(|accuracy| 1.0 - accuracy),
            padding_false_edit_rate: padding
                .and_then(|mask| mask.pixel_accuracy)
                .map(|accuracy| 1.0 - accuracy),
            padding_false_edit_transition_rate: padding
                .and_then(|mask| mask.exact_transition_accuracy)
                .map(|accuracy| 1.0 - accuracy),
            masks,
        }
    }
}

pub fn aggregate_decoder_metrics(
    rows: impl IntoIterator<Item = SemanticDecoderMetrics>,
) -> SemanticDecoderMetrics {
    let mut masks = BTreeMap::<String, Vec<SemanticMaskMetrics>>::new();
    for row in rows {
        for (name, metrics) in row.masks {
            masks.entry(name).or_default().push(metrics);
        }
    }
    let mut metrics = SemanticDecoderMetrics::from_masks(
        masks
            .into_iter()
            .map(|(name, rows)| {
                let pixels = rows.iter().map(|row| row.pixels).sum::<usize>();
                let transitions = rows.iter().map(|row| row.transitions).sum::<usize>();
                let weighted = |read: fn(&SemanticMaskMetrics) -> Option<f64>, weights_pixels| {
                    let (sum, weight) = rows.iter().fold((0.0, 0usize), |(sum, weight), row| {
                        let row_weight = if weights_pixels {
                            row.pixels
                        } else {
                            row.transitions
                        };
                        match read(row) {
                            Some(value) => (sum + value * row_weight as f64, weight + row_weight),
                            None => (sum, weight),
                        }
                    });
                    (weight > 0).then_some(sum / weight as f64)
                };
                (
                    name,
                    SemanticMaskMetrics {
                        pixels,
                        transitions,
                        mean_nll: weighted(|row| row.mean_nll, true),
                        pixel_accuracy: weighted(|row| row.pixel_accuracy, true),
                        exact_transition_accuracy: weighted(
                            |row| row.exact_transition_accuracy,
                            false,
                        ),
                        mean_transition_accuracy: weighted(
                            |row| row.mean_transition_accuracy,
                            false,
                        ),
                    },
                )
            })
            .collect(),
    );
    // Content rectangles may differ across rollout sources, so their raw mask
    // aggregates are not comparable. Derive the scalar false-edit diagnostics
    // first, then omit only the misleading per-mask rows.
    metrics
        .masks
        .retain(|name, _| !NONCOMPARABLE_CONTENT_MASKS.contains(&name.as_str()));
    metrics
}

/// Model-independent reducer used by the report path and by action-blindness
/// regression tests. `legal_actions[state]` is the complete candidate set for
/// that held-out state; the callback returns the flattened predicted latent.
pub fn action_controllability_probe<F>(
    legal_actions: &[Vec<ArcAction>],
    difference_threshold: f64,
    mut predict: F,
) -> Result<ActionControllabilityMetrics>
where
    F: FnMut(usize, &ArcAction) -> Result<Vec<f32>>,
{
    ensure!(
        difference_threshold.is_finite() && difference_threshold >= 0.0,
        "action-controllability threshold must be finite and non-negative"
    );
    let mut pair_distance_sum = 0.0f64;
    let mut pairs = 0usize;
    let mut eligible_states = 0usize;
    let mut states_above_threshold = 0usize;
    let mut prediction_count = 0usize;
    for (state_index, actions) in legal_actions.iter().enumerate() {
        if actions.len() < 2 {
            continue;
        }
        eligible_states += 1;
        let predictions = actions
            .iter()
            .map(|action| predict(state_index, action))
            .collect::<Result<Vec<_>>>()?;
        prediction_count += predictions.len();
        let dimension = predictions.first().map_or(0, Vec::len);
        ensure!(dimension > 0, "action-controllability latent is empty");
        ensure!(
            predictions.iter().all(|row| row.len() == dimension),
            "action-controllability predictions have inconsistent dimensions"
        );
        ensure!(
            predictions.iter().flatten().all(|value| value.is_finite()),
            "action-controllability predictions contain a non-finite latent"
        );
        let mut state_above_threshold = false;
        for left in 0..predictions.len() {
            for right in left + 1..predictions.len() {
                let squared = predictions[left]
                    .iter()
                    .zip(&predictions[right])
                    .map(|(left, right)| {
                        let delta = f64::from(*left) - f64::from(*right);
                        delta * delta
                    })
                    .sum::<f64>();
                let distance = (squared / dimension as f64).sqrt();
                pair_distance_sum += distance;
                pairs += 1;
                state_above_threshold |= distance > difference_threshold;
            }
        }
        states_above_threshold += usize::from(state_above_threshold);
    }
    Ok(ActionControllabilityMetrics {
        states: legal_actions.len(),
        states_with_action_pairs: eligible_states,
        action_predictions: prediction_count,
        action_pairs: pairs,
        mean_pairwise_latent_distance: (pairs > 0).then_some(pair_distance_sum / pairs as f64),
        difference_threshold,
        fraction_states_above_threshold: (eligible_states > 0)
            .then_some(states_above_threshold as f64 / eligible_states as f64),
        action_contract: "all caller-supplied legal actions per fixed held-out state; unordered pair RMS distance over flattened predicted consumer latents".into(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSourceMetrics {
    pub transitions: usize,
    pub status_pixels: usize,
    /// Fraction of gameplay pixels where the raw argmax one-step decode and
    /// the composed copy-gate decode disagree.
    #[serde(default)]
    pub raw_composed_pixel_disagreement: Option<f64>,
    /// Fraction of factually changed gameplay pixels whose copy gate opens
    /// (sigmoid >= 0.5 selects the predicted colour over the current pixel).
    #[serde(default)]
    pub copy_gate_open_rate_changed: Option<f64>,
    /// Fraction of factually unchanged gameplay pixels whose copy gate opens;
    /// the gate-side driver of false edits in the composed decode.
    #[serde(default)]
    pub copy_gate_open_rate_unchanged: Option<f64>,
    pub variants: BTreeMap<String, SemanticDecoderMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEvaluation {
    pub schema: String,
    pub mask_contract: String,
    pub reduction_contract: String,
    pub population_contract: String,
    pub action_control_contract: String,
    pub overall: SemanticSourceMetrics,
    pub by_source: BTreeMap<String, SemanticSourceMetrics>,
}

/// Action-ablation configuration. Generic callers default to no NULL action;
/// the Foundation-v2 evaluator supplies id 0 because its mixed stream trains
/// that explicit no-op embedding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticControlConfig {
    pub trained_null_action_id: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionControllabilityMetrics {
    pub states: usize,
    pub states_with_action_pairs: usize,
    pub action_predictions: usize,
    pub action_pairs: usize,
    /// RMS distance per latent element, averaged over unordered action pairs.
    pub mean_pairwise_latent_distance: Option<f64>,
    pub difference_threshold: f64,
    /// Fraction of pair-eligible states with at least one distance above the
    /// fixed threshold.
    pub fraction_states_above_threshold: Option<f64>,
    pub action_contract: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityHistoryMetrics {
    pub history_length: usize,
    pub rows: usize,
    pub groups: usize,
    pub repeated_groups: usize,
    pub ambiguous_groups: usize,
    pub rows_in_ambiguous_groups: usize,
    pub ambiguous_group_fraction: Option<f64>,
    /// Majority factual-successor accuracy within each visible-input group.
    pub deterministic_exact_ceiling: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityCeiling {
    pub key_contract: String,
    pub outcome_contract: String,
    pub history_1: AmbiguityHistoryMetrics,
    pub history_2: AmbiguityHistoryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionCensusMetrics {
    pub rows: usize,
    pub unique_visible_inputs: usize,
    pub repeated_visible_inputs: usize,
    pub conflicting_visible_inputs: usize,
    pub rows_in_conflicts: usize,
    /// Best possible exact-next-board accuracy for any deterministic predictor
    /// that sees only `(current gameplay pixels, action id, coordinates)`.
    pub deterministic_exact_ceiling: Option<f64>,
    pub deterministic_pixel_ceiling: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionCensus {
    pub key_contract: String,
    pub outcome_contract: String,
    pub overall: CollisionCensusMetrics,
    pub by_source: BTreeMap<String, CollisionCensusMetrics>,
}

#[derive(Default, Clone)]
struct MaskAccum {
    pixels: usize,
    transitions: usize,
    nll_sum: f64,
    nll_pixels: usize,
    correct: usize,
    exact: usize,
    transition_accuracy_sum: f64,
}

impl MaskAccum {
    fn add(&mut self, predictions: &[u8], log_probs: Option<&[f32]>, labels: &[u8], mask: &[bool]) {
        let mut pixels = 0usize;
        let mut correct = 0usize;
        let mut exact = true;
        let mut nll_sum = 0.0;
        for (index, selected) in mask.iter().copied().enumerate() {
            if !selected {
                continue;
            }
            pixels += 1;
            let is_correct = predictions[index] == labels[index];
            correct += usize::from(is_correct);
            exact &= is_correct;
            if let Some(log_probs) = log_probs {
                nll_sum -= f64::from(log_probs[index * PALETTE_SIZE + labels[index] as usize]);
            }
        }
        if pixels == 0 {
            return;
        }
        self.pixels += pixels;
        self.transitions += 1;
        self.correct += correct;
        self.exact += usize::from(exact);
        self.transition_accuracy_sum += correct as f64 / pixels as f64;
        if log_probs.is_some() {
            self.nll_sum += nll_sum;
            self.nll_pixels += pixels;
        }
    }

    fn finish(self) -> SemanticMaskMetrics {
        SemanticMaskMetrics {
            pixels: self.pixels,
            transitions: self.transitions,
            mean_nll: (self.nll_pixels > 0).then_some(self.nll_sum / self.nll_pixels as f64),
            pixel_accuracy: (self.pixels > 0).then_some(self.correct as f64 / self.pixels as f64),
            exact_transition_accuracy: (self.transitions > 0)
                .then_some(self.exact as f64 / self.transitions as f64),
            mean_transition_accuracy: (self.transitions > 0)
                .then_some(self.transition_accuracy_sum / self.transitions as f64),
        }
    }
}

#[derive(Default, Clone)]
struct SourceAccum {
    transitions: usize,
    status_pixels: usize,
    decode_pixels: usize,
    raw_composed_disagreements: usize,
    gate_changed_pixels: usize,
    gate_open_changed: usize,
    gate_unchanged_pixels: usize,
    gate_open_unchanged: usize,
    variants: BTreeMap<String, BTreeMap<String, MaskAccum>>,
}

struct SemanticRow<'a> {
    predictions: &'a [u8],
    log_probs: Option<&'a [f32]>,
    current: &'a [u8],
    transition_target: &'a [u8],
    decoder_labels: &'a [u8],
    sample: &'a TransitionSample,
}

impl SourceAccum {
    fn add_variant(&mut self, variant: &str, row: SemanticRow<'_>) {
        let masks = semantic_masks(row.current, row.transition_target, row.sample);
        let entry = self.variants.entry(variant.into()).or_default();
        for (name, mask) in MASKS.into_iter().zip(masks) {
            entry.entry(name.into()).or_default().add(
                row.predictions,
                row.log_probs,
                row.decoder_labels,
                &mask,
            );
        }
    }

    /// Cross-tabulate the raw one-step argmax decode against the composed
    /// copy-gate decode and the thresholded gate itself on one transition.
    fn add_composition_diagnostics(
        &mut self,
        raw: &[u8],
        composed: &[u8],
        gate_open: &[u8],
        current: &[u8],
        target: &[u8],
    ) {
        for index in 0..raw.len() {
            self.decode_pixels += 1;
            self.raw_composed_disagreements += usize::from(raw[index] != composed[index]);
            let open = usize::from(gate_open[index] != 0);
            if current[index] != target[index] {
                self.gate_changed_pixels += 1;
                self.gate_open_changed += open;
            } else {
                self.gate_unchanged_pixels += 1;
                self.gate_open_unchanged += open;
            }
        }
    }

    fn finish(self) -> SemanticSourceMetrics {
        SemanticSourceMetrics {
            transitions: self.transitions,
            status_pixels: self.status_pixels,
            raw_composed_pixel_disagreement: (self.decode_pixels > 0)
                .then_some(self.raw_composed_disagreements as f64 / self.decode_pixels as f64),
            copy_gate_open_rate_changed: (self.gate_changed_pixels > 0)
                .then_some(self.gate_open_changed as f64 / self.gate_changed_pixels as f64),
            copy_gate_open_rate_unchanged: (self.gate_unchanged_pixels > 0)
                .then_some(self.gate_open_unchanged as f64 / self.gate_unchanged_pixels as f64),
            variants: self
                .variants
                .into_iter()
                .map(|(variant, masks)| {
                    (
                        variant,
                        SemanticDecoderMetrics::from_masks(
                            masks
                                .into_iter()
                                .map(|(name, metrics)| (name, metrics.finish()))
                                .collect(),
                        ),
                    )
                })
                .collect(),
        }
    }
}

fn semantic_masks(current: &[u8], target: &[u8], sample: &TransitionSample) -> [Vec<bool>; 9] {
    // 63 rows for legacy rows, 64 under ADR 0005 §1.1 (whole-frame decoders).
    let gameplay_pixels = target.len();
    let rows = gameplay_pixels / FRAME_SIDE;
    let mut content = vec![false; gameplay_pixels];
    // Provenance carries the exact placement origin (zero for legacy rows),
    // so translated V5 content is classified as content, not padding.
    let origin_x = usize::from(sample.provenance.content_x);
    let origin_y = usize::from(sample.provenance.content_y);
    let width =
        usize::from(sample.provenance.content_width).min(FRAME_SIDE.saturating_sub(origin_x));
    let height = usize::from(sample.provenance.content_height).min(rows.saturating_sub(origin_y));
    for y in origin_y..origin_y + height {
        for x in origin_x..origin_x + width {
            content[y * FRAME_SIDE + x] = true;
        }
    }
    let padding: Vec<bool> = content.iter().map(|selected| !selected).collect();
    // Background is the row's rendered EMPTY colour (ADR 0005 §1.2); legacy
    // rows render it as index 0.
    let background = sample.provenance.background_color;
    let foreground: Vec<bool> = target.iter().map(|pixel| *pixel != background).collect();
    let changed: Vec<_> = current
        .iter()
        .zip(target)
        .map(|(before, after)| before != after)
        .collect();
    let unchanged: Vec<bool> = changed.iter().map(|selected| !selected).collect();
    let changed_content: Vec<bool> = changed
        .iter()
        .zip(&content)
        .map(|(changed, content)| *changed && *content)
        .collect();
    let unchanged_content: Vec<bool> = unchanged
        .iter()
        .zip(&content)
        .map(|(unchanged, content)| *unchanged && *content)
        .collect();
    let unchanged_padding: Vec<bool> = unchanged
        .iter()
        .zip(&padding)
        .map(|(unchanged, padding)| *unchanged && *padding)
        .collect();
    // Full-frame transition mask: every status-excluded gameplay pixel,
    // padding included. Padding pixels are part of the decode contract (a
    // decode that hallucinates content into padding is not exact), and the
    // fixed 4032-pixel extent makes the mask comparable across source kinds,
    // so it stays out of NONCOMPARABLE_CONTENT_MASKS.
    let gameplay = vec![true; gameplay_pixels];
    [
        content,
        padding,
        foreground,
        changed,
        unchanged,
        changed_content,
        unchanged_content,
        unchanged_padding,
        gameplay,
    ]
}

struct DecodedRows {
    predictions: Vec<Vec<u8>>,
    log_probs: Vec<Vec<f32>>,
}

fn decoded_rows(logits: &Tensor) -> Result<DecodedRows> {
    let (batch, rows, width, _) = logits.dims4()?;
    let pixels = rows * width;
    let predictions = logits
        .argmax(D::Minus1)?
        .reshape((batch, pixels))?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()?;
    let log_probs = ops::log_softmax(logits, D::Minus1)?
        .reshape((batch, pixels * PALETTE_SIZE))?
        .to_vec2::<f32>()?;
    Ok(DecodedRows {
        predictions,
        log_probs,
    })
}

pub fn latent_semantic_metrics(
    model: &WorldModel,
    latent: &Tensor,
    sample: &TransitionSample,
) -> Result<SemanticDecoderMetrics> {
    let decoded = decoded_rows(&model.exact_gameplay_logits(latent)?)?;
    ensure!(
        decoded.predictions.len() == 1,
        "rollout semantic latent must have batch size one"
    );
    let gameplay_len = gameplay_rows(model.config().world_core_v6) * FRAME_SIDE;
    let current = gameplay_prefix(&sample.current, gameplay_len);
    let target = gameplay_prefix(&sample.next, gameplay_len);
    let mut accum = SourceAccum::default();
    accum.add_variant(
        "rollout",
        SemanticRow {
            predictions: &decoded.predictions[0],
            log_probs: Some(&decoded.log_probs[0]),
            current,
            transition_target: target,
            decoder_labels: target,
            sample,
        },
    );
    let mut variants = accum.finish().variants;
    let metrics = variants
        .remove("rollout")
        .expect("rollout variant was inserted");
    Ok(metrics)
}

/// Legacy 63-row board slice used by model-free collision censuses.
fn gameplay(frame: &crate::p2::data::ArcFrame) -> &[u8] {
    gameplay_prefix(frame, (FRAME_SIDE - 1) * FRAME_SIDE)
}

/// Board slice matching the evaluated decoder's rows (63 legacy, 64 for
/// `world_core_v6`).
fn gameplay_prefix(frame: &crate::p2::data::ArcFrame, gameplay_len: usize) -> &[u8] {
    &frame.pixels[..gameplay_len.min(frame.pixels.len())]
}

fn append_action_key(key: &mut Vec<u8>, action: &ArcAction) {
    key.push(action.id);
    key.push(action.x.unwrap_or(u8::MAX));
    key.push(action.y.unwrap_or(u8::MAX));
}

fn ambiguity_history_metrics(
    history_length: usize,
    rows: impl IntoIterator<Item = (Vec<u8>, Vec<u8>)>,
) -> AmbiguityHistoryMetrics {
    let mut groups = BTreeMap::<Vec<u8>, Vec<Vec<u8>>>::new();
    for (key, successor) in rows {
        groups.entry(key).or_default().push(successor);
    }
    let rows = groups.values().map(Vec::len).sum::<usize>();
    let repeated_groups = groups
        .values()
        .filter(|successors| successors.len() > 1)
        .count();
    let mut ambiguous_groups = 0usize;
    let mut rows_in_ambiguous_groups = 0usize;
    let mut deterministic_exact_correct = 0usize;
    for successors in groups.values() {
        let mut outcomes = BTreeMap::<&[u8], usize>::new();
        for successor in successors {
            *outcomes.entry(successor).or_default() += 1;
        }
        if outcomes.len() > 1 {
            ambiguous_groups += 1;
            rows_in_ambiguous_groups += successors.len();
        }
        deterministic_exact_correct += outcomes.values().copied().max().unwrap_or(0);
    }
    AmbiguityHistoryMetrics {
        history_length,
        rows,
        groups: groups.len(),
        repeated_groups,
        ambiguous_groups,
        rows_in_ambiguous_groups,
        ambiguous_group_fraction: (!groups.is_empty())
            .then_some(ambiguous_groups as f64 / groups.len() as f64),
        deterministic_exact_ceiling: (rows > 0)
            .then_some(deterministic_exact_correct as f64 / rows as f64),
    }
}

/// Measure factual-successor ambiguity for a visible-state/action predictor.
/// History two is formed only from contiguous rows sharing the explicit
/// provenance trajectory identity; family labels are never used as sequence
/// identity.
pub fn ambiguity_ceiling(samples: &[TransitionSample]) -> AmbiguityCeiling {
    let history_1_rows = samples.iter().map(|sample| {
        let mut key = Vec::with_capacity(3 + gameplay(&sample.current).len());
        append_action_key(&mut key, &sample.action);
        key.extend_from_slice(gameplay(&sample.current));
        (key, gameplay(&sample.next).to_vec())
    });

    let mut trajectories = BTreeMap::<&str, Vec<&TransitionSample>>::new();
    for sample in samples {
        trajectories
            .entry(sample.provenance.trajectory_id.as_str())
            .or_default()
            .push(sample);
    }
    let mut history_2_rows = Vec::new();
    for steps in trajectories.values_mut() {
        steps.sort_by_key(|sample| sample.transition_index);
        for pair in steps.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if current.transition_index != previous.transition_index.saturating_add(1)
                || gameplay(&previous.next) != gameplay(&current.current)
            {
                continue;
            }
            let mut key = Vec::with_capacity(3 + 2 * gameplay(&current.current).len());
            append_action_key(&mut key, &current.action);
            key.extend_from_slice(gameplay(&previous.current));
            key.extend_from_slice(gameplay(&current.current));
            history_2_rows.push((key, gameplay(&current.next).to_vec()));
        }
    }

    AmbiguityCeiling {
        key_contract: "h1=(visible current gameplay pixels, action id, ACTION6 coordinates); h2=(previous visible gameplay pixels, visible current gameplay pixels, action), with previous rows joined only by provenance.trajectory_id and contiguous transition_index".into(),
        outcome_contract: "distinct factual successor = distinct status-row-excluded next gameplay pixels".into(),
        history_1: ambiguity_history_metrics(1, history_1_rows),
        history_2: ambiguity_history_metrics(2, history_2_rows),
    }
}

fn source_labels(samples: &[TransitionSample], _source_lengths: &[(String, usize)]) -> Vec<String> {
    samples
        .iter()
        .map(|sample| sample.provenance.source_kind.clone())
        .collect()
}

fn census(samples: &[&TransitionSample]) -> CollisionCensusMetrics {
    let mut groups = BTreeMap::<Vec<u8>, Vec<Vec<u8>>>::new();
    for sample in samples {
        let mut key = Vec::with_capacity(
            3 + sample.goal_features.values.len() * std::mem::size_of::<f32>()
                + (FRAME_SIDE - 1) * FRAME_SIDE,
        );
        key.push(sample.action.id);
        key.push(sample.action.x.unwrap_or(u8::MAX));
        key.push(sample.action.y.unwrap_or(u8::MAX));
        for goal in sample.goal_features.values {
            key.extend_from_slice(&goal.to_bits().to_le_bytes());
        }
        key.extend_from_slice(gameplay(&sample.current));
        groups
            .entry(key)
            .or_default()
            .push(gameplay(&sample.next).to_vec());
    }
    let mut repeated = 0usize;
    let mut conflicting = 0usize;
    let mut conflict_rows = 0usize;
    let mut exact_correct = 0usize;
    let mut pixel_correct = 0usize;
    let mut pixel_total = 0usize;
    for outcomes in groups.values() {
        repeated += usize::from(outcomes.len() > 1);
        let mut classes = BTreeMap::<&[u8], usize>::new();
        for outcome in outcomes {
            *classes.entry(outcome).or_default() += 1;
        }
        if classes.len() > 1 {
            conflicting += 1;
            conflict_rows += outcomes.len();
        }
        exact_correct += classes.values().copied().max().unwrap_or(0);
        if let Some(first) = outcomes.first() {
            for pixel in 0..first.len() {
                let mut palette_counts = [0usize; PALETTE_SIZE];
                for outcome in outcomes {
                    palette_counts[outcome[pixel] as usize] += 1;
                }
                pixel_correct += palette_counts.into_iter().max().unwrap_or(0);
                pixel_total += outcomes.len();
            }
        }
    }
    CollisionCensusMetrics {
        rows: samples.len(),
        unique_visible_inputs: groups.len(),
        repeated_visible_inputs: repeated,
        conflicting_visible_inputs: conflicting,
        rows_in_conflicts: conflict_rows,
        deterministic_exact_ceiling: (!samples.is_empty())
            .then_some(exact_correct as f64 / samples.len() as f64),
        deterministic_pixel_ceiling: (pixel_total > 0)
            .then_some(pixel_correct as f64 / pixel_total as f64),
    }
}

pub fn collision_census(
    samples: &[TransitionSample],
    source_lengths: &[(String, usize)],
) -> CollisionCensus {
    let labels = source_labels(samples, source_lengths);
    let mut grouped = BTreeMap::<String, Vec<&TransitionSample>>::new();
    for (sample, source) in samples.iter().zip(labels) {
        grouped.entry(source).or_default().push(sample);
    }
    let all = samples.iter().collect::<Vec<_>>();
    CollisionCensus {
        key_contract: "status-excluded current gameplay pixels + public goal features + action id + ACTION6 coordinates".into(),
        outcome_contract: "status-excluded exact next gameplay pixels".into(),
        overall: census(&all),
        by_source: grouped
            .into_iter()
            .map(|(source, rows)| (source, census(&rows)))
            .collect(),
    }
}

/// Deterministically rotate action tuples only among rows that share
/// `provenance.source_kind`. ACTION6 coordinates rotate as rectangle-relative
/// offsets and are re-based onto each target row's content rectangle. The
/// cyclic offset maximizing genuinely changed resulting tuples is selected per
/// source (lowest offset breaks ties), exposing unavoidable duplicates instead
/// of pretending every row was perturbed.
fn shuffled_action_source_rows(samples: &[TransitionSample]) -> BTreeMap<&str, Vec<usize>> {
    let mut source_rows = BTreeMap::<&str, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        source_rows
            .entry(sample.provenance.source_kind.as_str())
            .or_default()
            .push(index);
    }
    source_rows
}

/// Rows eligible for the shuffled-action control: members of a
/// `provenance.source_kind` group with at least two rows. Shares the exact
/// grouping rule with `shuffled_action_control_samples` so the reported
/// eligibility count cannot drift from the control's construction.
pub fn shuffled_action_eligible_rows(samples: &[TransitionSample]) -> usize {
    shuffled_action_source_rows(samples)
        .values()
        .filter(|indices| indices.len() >= 2)
        .map(Vec::len)
        .sum()
}

fn conjugated_action_for_target(donor: &TransitionSample, target: &TransitionSample) -> ArcAction {
    if donor.action.id != 6 {
        return donor.action.clone();
    }

    fn rebase(
        coordinate: u8,
        donor_origin: u16,
        donor_size: u16,
        target_origin: u16,
        target_size: u16,
    ) -> Option<u8> {
        let offset = u16::from(coordinate).checked_sub(donor_origin)?;
        if offset >= donor_size || target_size == 0 {
            return None;
        }
        let offset = if donor_size == target_size {
            offset
        } else {
            offset.min(target_size.saturating_sub(1))
        };
        u8::try_from(target_origin + offset).ok()
    }

    let conjugated = donor.action.x.zip(donor.action.y).and_then(|(x, y)| {
        Some((
            rebase(
                x,
                donor.provenance.content_x,
                donor.provenance.content_width,
                target.provenance.content_x,
                target.provenance.content_width,
            )?,
            rebase(
                y,
                donor.provenance.content_y,
                donor.provenance.content_height,
                target.provenance.content_y,
                target.provenance.content_height,
            )?,
        ))
    });
    match conjugated {
        Some((x, y)) => ArcAction {
            id: donor.action.id,
            x: Some(x),
            y: Some(y),
        },
        None => donor.action.clone(),
    }
}

fn shuffled_action_control_samples_where(
    samples: &[TransitionSample],
    eligible: impl Fn(&TransitionSample) -> bool,
) -> (Vec<TransitionSample>, usize) {
    let mut shuffled = samples.to_vec();
    let source_rows = shuffled_action_source_rows(samples)
        .into_iter()
        .map(|(source, indices)| {
            (
                source,
                indices
                    .into_iter()
                    .filter(|&index| eligible(&samples[index]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let eligible_rows = source_rows
        .values()
        .filter(|indices| indices.len() >= 2)
        .map(Vec::len)
        .sum();
    for indices in source_rows.values() {
        if indices.len() < 2 {
            continue;
        }
        let mut best_offset = 1usize;
        let mut best_changed = 0usize;
        for offset in 1..indices.len() {
            let changed = indices
                .iter()
                .enumerate()
                .filter(|(position, target)| {
                    let donor = indices[(*position + offset) % indices.len()];
                    samples[**target].action
                        != conjugated_action_for_target(&samples[donor], &samples[**target])
                })
                .count();
            if changed > best_changed {
                best_changed = changed;
                best_offset = offset;
            }
        }
        for (position, &target) in indices.iter().enumerate() {
            let donor = indices[(position + best_offset) % indices.len()];
            shuffled[target].action =
                conjugated_action_for_target(&samples[donor], &samples[target]);
        }
    }
    (shuffled, eligible_rows)
}

pub fn shuffled_action_control_samples(samples: &[TransitionSample]) -> Vec<TransitionSample> {
    shuffled_action_control_samples_where(samples, |_| true).0
}

/// Sidecar-aware action intervention used by the foundation-v2 gate. Only the
/// ACTION5/ACTION6 rows governed by the recorded episode operator are rotated;
/// every changed tuple can therefore be replayed exactly on the target board.
#[derive(Debug, Clone)]
pub struct ShuffledActionControlPopulation {
    pub samples: Vec<TransitionSample>,
    pub counterfactual_next: Vec<Option<ArcFrame>>,
    pub eligible_rows: usize,
}

impl ShuffledActionControlPopulation {
    pub fn changed_tuples(&self, factual: &[TransitionSample]) -> usize {
        factual
            .iter()
            .zip(&self.samples)
            .filter(|(factual, shuffled)| factual.action != shuffled.action)
            .count()
    }

    pub fn outcome_changing(&self, factual: &[TransitionSample]) -> Vec<Option<bool>> {
        factual
            .iter()
            .zip(&self.counterfactual_next)
            .map(|(factual, counterfactual)| {
                counterfactual.as_ref().map(|counterfactual| {
                    let gameplay_len = (FRAME_SIDE - 1) * FRAME_SIDE;
                    counterfactual.pixels[..gameplay_len] != factual.next.pixels[..gameplay_len]
                })
            })
            .collect()
    }

    pub fn outcome_changing_tuples(&self, factual: &[TransitionSample]) -> Option<usize> {
        self.outcome_changing(factual)
            .into_iter()
            .try_fold(0usize, |count, changed| {
                changed.map(|changed| count + usize::from(changed))
            })
    }
}

/// Build the exact shuffled conditioning shown to the model and, when V5
/// operator provenance is available, replay that conditioning on each target
/// row's current board. Populations without the sidecar retain the historical
/// shuffle and expose unknown counterfactuals so callers can fail closed to the
/// old per-row metric behavior without claiming causal coverage.
pub fn shuffled_action_control_population(
    samples: &[TransitionSample],
    provenance: Option<&[V5SampleProvenance]>,
) -> Result<ShuffledActionControlPopulation> {
    let (shuffled, eligible_rows) = if provenance.is_some() {
        shuffled_action_control_samples_where(samples, |sample| matches!(sample.action.id, 5 | 6))
    } else {
        shuffled_action_control_samples_where(samples, |_| true)
    };
    let Some(provenance) = provenance else {
        return Ok(ShuffledActionControlPopulation {
            counterfactual_next: vec![None; samples.len()],
            samples: shuffled,
            eligible_rows,
        });
    };
    ensure!(
        provenance.len() == samples.len(),
        "shuffled-action V5 provenance rows do not match the sample count"
    );
    let counterfactual_next = samples
        .iter()
        .zip(&shuffled)
        .zip(provenance)
        .map(|((factual, shuffled), provenance)| {
            ensure!(
                provenance.source == factual.provenance,
                "shuffled-action V5 provenance does not match its transition"
            );
            if factual.action == shuffled.action {
                return Ok(Some(factual.next.clone()));
            }
            ensure!(
                matches!(shuffled.action.id, 5 | 6),
                "sidecar-aware shuffle produced an action outside the V5 episode operator"
            );
            apply_episode_operator(
                &factual.current,
                &shuffled.action,
                provenance.content_rect,
                provenance.operator,
            )
            .map(Some)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ShuffledActionControlPopulation {
        samples: shuffled,
        counterfactual_next,
        eligible_rows,
    })
}

/// Evaluate the current encoding, target encoding, predicted next state, and
/// preregistered hard/action controls through one shared exact decoder.
pub fn evaluate_semantics(
    model: &WorldModel,
    samples: &[TransitionSample],
    source_lengths: &[(String, usize)],
    physical_batch: usize,
    device: &Device,
) -> Result<SemanticEvaluation> {
    evaluate_semantics_with_control(
        model,
        samples,
        source_lengths,
        physical_batch,
        device,
        SemanticControlConfig::default(),
    )
}

pub fn evaluate_semantics_with_control(
    model: &WorldModel,
    samples: &[TransitionSample],
    source_lengths: &[(String, usize)],
    physical_batch: usize,
    device: &Device,
    control: SemanticControlConfig,
) -> Result<SemanticEvaluation> {
    ensure!(
        model.config().world_core_v4,
        "semantic evaluation requires Full V4"
    );
    let labels = source_labels(samples, source_lengths);
    let shuffled = shuffled_action_control_samples(samples);
    let mut overall = SourceAccum::default();
    let mut by_source = BTreeMap::<String, SourceAccum>::new();
    for start in (0..samples.len()).step_by(physical_batch.max(1)) {
        let end = (start + physical_batch.max(1)).min(samples.len());
        let rows = &samples[start..end];
        let batch = batch_from_samples(rows, device)?;
        let (current_latent, target_latent) =
            model.encode_state_pair(&batch.frames, &batch.next_frames)?;
        let prediction = model
            .forward_from_latent_with_operator_conditioning(
                &current_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                &batch.operator_conditioning,
            )?
            .y;
        let (shuffled_actions, shuffled_coords) =
            action_tensors_from_samples(&shuffled[start..end], device)?;
        let shuffled_prediction = model
            .forward_from_latent_with_operator_conditioning(
                &current_latent,
                &shuffled_actions,
                &shuffled_coords,
                &batch.goals,
                &batch.operator_conditioning,
            )?
            .y;
        let gameplay_pixels = gameplay_rows(model.config().world_core_v6) * FRAME_SIDE;
        let composed_predictions = model
            .composed_gameplay_decode(&prediction, &batch.frames)?
            .reshape((rows.len(), gameplay_pixels))?
            .to_dtype(DType::U8)?
            .to_vec2::<u8>()?;
        let gate_open = model
            .exact_copy_gate(&prediction)?
            .ge(0.5)?
            .reshape((rows.len(), gameplay_pixels))?
            .to_dtype(DType::U8)?
            .to_vec2::<u8>()?;
        let mut decoded = vec![
            (
                "current_reconstruction",
                model.exact_gameplay_logits(&current_latent)?,
            ),
            (
                "target_reconstruction",
                model.exact_gameplay_logits(&target_latent)?,
            ),
            (
                "one_step_prediction",
                model.exact_gameplay_logits(&prediction)?,
            ),
            (
                "learned_copy_control",
                model.exact_gameplay_logits(&current_latent)?,
            ),
            (
                "action_shuffled_prediction",
                model.exact_gameplay_logits(&shuffled_prediction)?,
            ),
        ];
        if let Some(null_action_id) = control.trained_null_action_id {
            ensure!(
                null_action_id <= 7,
                "trained NULL action id must be in the model embedding range 0..=7"
            );
            let null_actions = Tensor::from_vec(
                vec![u32::from(null_action_id); rows.len()],
                rows.len(),
                device,
            )?;
            let null_coords = Tensor::zeros((rows.len(), 2), DType::F32, device)?;
            let null_prediction = model
                .forward_from_latent_with_operator_conditioning(
                    &current_latent,
                    &null_actions,
                    &null_coords,
                    &batch.goals,
                    &batch.operator_conditioning,
                )?
                .y;
            decoded.push((
                "trained_null_action_prediction",
                model.exact_gameplay_logits(&null_prediction)?,
            ));
        }
        let decoded = decoded
            .into_iter()
            .map(|(name, logits)| Ok((name, decoded_rows(&logits)?)))
            .collect::<Result<Vec<_>>>()?;
        for (local, sample) in rows.iter().enumerate() {
            let current = gameplay_prefix(&sample.current, gameplay_pixels);
            let target = gameplay_prefix(&sample.next, gameplay_pixels);
            ensure!(
                current.len() == gameplay_pixels && target.len() == gameplay_pixels,
                "semantic evaluation requires fixed 64x64 frames"
            );
            let status_pixels = FRAME_SIDE * FRAME_SIDE - gameplay_pixels;
            overall.transitions += 1;
            overall.status_pixels += status_pixels;
            let source = by_source.entry(labels[start + local].clone()).or_default();
            source.transitions += 1;
            source.status_pixels += status_pixels;
            for (name, decoded) in &decoded {
                let decoder_target = if *name == "current_reconstruction" {
                    current
                } else {
                    target
                };
                let row = || SemanticRow {
                    predictions: &decoded.predictions[local],
                    log_probs: Some(&decoded.log_probs[local]),
                    current,
                    transition_target: target,
                    decoder_labels: decoder_target,
                    sample,
                };
                overall.add_variant(name, row());
                source.add_variant(name, row());
            }
            let raw_one_step = decoded
                .iter()
                .find(|(name, _)| *name == "one_step_prediction")
                .map(|(_, rows)| rows.predictions[local].as_slice())
                .expect("one-step decode variant is always present");
            overall.add_composition_diagnostics(
                raw_one_step,
                &composed_predictions[local],
                &gate_open[local],
                current,
                target,
            );
            source.add_composition_diagnostics(
                raw_one_step,
                &composed_predictions[local],
                &gate_open[local],
                current,
                target,
            );
            let copy = current;
            // Background control: the row's rendered EMPTY colour everywhere.
            let zero = vec![sample.provenance.background_color; target.len()];
            for (name, prediction) in [
                ("hard_copy_control", copy),
                ("zero_control", zero.as_slice()),
                ("direct_target_positive_control", target),
                ("composed_copy_gate", composed_predictions[local].as_slice()),
            ] {
                let row = || SemanticRow {
                    predictions: prediction,
                    log_probs: None,
                    current,
                    transition_target: target,
                    decoder_labels: target,
                    sample,
                };
                overall.add_variant(name, row());
                source.add_variant(name, row());
            }
        }
    }
    let mut overall = overall.finish();
    for metrics in overall.variants.values_mut() {
        for mask in NONCOMPARABLE_CONTENT_MASKS {
            metrics.masks.remove(mask);
        }
        // The scalar false-edit fields derive from unchanged_content and
        // unchanged_padding, which pool non-comparable content geometries
        // across sources; `by_source` retains them, `overall` must not.
        metrics.false_edit_rate = None;
        metrics.false_edit_transition_rate = None;
        metrics.padding_false_edit_rate = None;
        metrics.padding_false_edit_transition_rate = None;
    }
    Ok(SemanticEvaluation {
        schema: SEMANTIC_EVAL_SCHEMA.into(),
        mask_contract: "gameplay=all rows[0,63) pixels (fixed 4032-pixel full-transition mask, padding included, source-comparable); content=[x,x+width)x[y,y+height) from the provenance origin (zero for legacy rows); padding=gameplay-content; foreground=target!=EMPTY; changed=current!=target; unchanged=gameplay-changed; unchanged_content=unchanged&content; unchanged_padding=unchanged&padding; status=row63 excluded. Content-rectangle masks are retained by source. false-edit metrics use unchanged_content, while padding hallucinations are reported separately from unchanged_padding. composed_copy_gate decodes via composed_gameplay_decode under the model's configured decode_composition (legacy hard gate: gate>=0.5 selects the predicted colour; joint_copy_mixture: two-candidate mixture MAP that can only convert edits into copies)".into(),
        reduction_contract: "overall aggregates only source-comparable masks and omits content-derived false-edit scalars; by_source reports pixel aggregate plus equal-transition mean within provenance.source_kind, including the false-edit scalars".into(),
        population_contract: "one_step_population; not comparable as a horizon curve with semantic_rollout, whose trajectory-filtered population is separately fingerprinted".into(),
        action_control_contract: match control.trained_null_action_id {
            Some(id) => format!("action_shuffled_prediction rotates action tuples only within provenance.source_kind and conjugates ACTION6 rectangle-relative coordinates onto each target content rectangle; trained_null_action_prediction uses configured trained NULL action id {id} with zero coordinates"),
            None => "action_shuffled_prediction rotates action tuples only within provenance.source_kind and conjugates ACTION6 rectangle-relative coordinates onto each target content rectangle; no action-masked/null variant was configured for this checkpoint".into(),
        },
        overall,
        by_source: by_source
            .into_iter()
            .map(|(source, metrics)| (source, metrics.finish()))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Split;
    use crate::p2::data::palette;
    use crate::p2::data::{
        compose_mixed_stream_batch, foundation_v2_stream_schedule, generate_curriculum, ArcAction,
        ArcFrame, ContentRect, D4Transform, EpisodeOperator, GoalFeatures, MixedStreamConfig,
        MixedStreamKind, OperatorFamily, SymmetryAugmentation, TransitionProvenance, V5DataSplit,
        V5SampleProvenance, FRAME_SIDE,
    };
    use crate::p2::train::{
        reinit_varmap_deterministic, TrainConfig, FOUNDATION_V2_GATE_SEED,
    };
    use candle_nn::{VarBuilder, VarMap};

    fn sample(next_pixel: u8) -> TransitionSample {
        let current = vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE];
        let mut next = current.clone();
        next[0] = next_pixel;
        TransitionSample {
            current: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current).unwrap(),
            next: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next).unwrap(),
            action: ArcAction::new(1, None, None).unwrap(),
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: None,
            goal_failed: None,
            exhausted: None,
            split: Split::Train,
            family: "movement".into(),
            seed: 9,
            episode_id: 4,
            transition_index: 0,
            provenance: TransitionProvenance {
                content_width: 7,
                content_height: 7,
                content_x: 0,
                content_y: 0,
                source_kind: "movement".into(),
                trajectory_id: "sim/Train/9/4".into(),
                operator: None,
                rule_id: 0,
                level_index: 0,
                available_actions: 0,
                context_len: 0,
                background_color: 0,
            },
            oracle_latent: None,
            context: Vec::new(),
        }
    }

    fn operator_row(
        current: ArcFrame,
        action: ArcAction,
        operator: EpisodeOperator,
        episode_id: u64,
    ) -> Result<(TransitionSample, V5SampleProvenance)> {
        let content_rect = ContentRect {
            x: 0,
            y: 0,
            width: 7,
            height: 7,
        };
        let next = apply_episode_operator(&current, &action, content_rect, operator)?;
        let source = TransitionProvenance {
            content_width: 7,
            content_height: 7,
            content_x: 0,
            content_y: 0,
            source_kind: "operator_control".into(),
            trajectory_id: format!("test/operator_control/{episode_id}"),
            operator: Some(operator),
            rule_id: 0,
            level_index: 0,
            available_actions: 0,
            context_len: 0,
            background_color: 0,
        };
        let noop = current.pixels[..(FRAME_SIDE - 1) * FRAME_SIDE]
            == next.pixels[..(FRAME_SIDE - 1) * FRAME_SIDE];
        let sample = TransitionSample {
            current,
            next,
            action,
            goal_features: GoalFeatures::zeros(),
            noop: Some(noop),
            goal_satisfied: None,
            goal_failed: None,
            exhausted: None,
            split: Split::Train,
            family: "operator_control".into(),
            seed: 11,
            episode_id,
            transition_index: 0,
            provenance: source.clone(),
            oracle_latent: None,
            context: Vec::new(),
        };
        let provenance = V5SampleProvenance {
            source,
            content_rect,
            data_split: V5DataSplit::Train,
            stream: MixedStreamKind::FactualBranches,
            operator,
            augmentation: SymmetryAugmentation {
                d4: D4Transform::Identity,
                color_permutation: std::array::from_fn(|index| index as u8),
            },
            goal_dropped: false,
            branch_group_id: None,
            contract_v6: false,
        };
        Ok((sample, provenance))
    }

    #[test]
    fn masks_keep_content_padding_change_and_status_semantically_distinct() {
        let mut sample = sample(1);
        let padding_pixel = 20 * FRAME_SIDE + 20;
        sample.next.pixels[padding_pixel] = 2;
        let masks = semantic_masks(gameplay(&sample.current), gameplay(&sample.next), &sample);
        assert_eq!(masks[0].iter().filter(|selected| **selected).count(), 49);
        assert_eq!(
            masks[1].iter().filter(|selected| **selected).count(),
            (FRAME_SIDE - 1) * FRAME_SIDE - 49
        );
        assert_eq!(masks[3].iter().filter(|selected| **selected).count(), 2);
        assert_eq!(masks[5].iter().filter(|selected| **selected).count(), 1);
        assert_eq!(masks[8].len(), (FRAME_SIDE - 1) * FRAME_SIDE);
        assert!(masks[8].iter().all(|selected| *selected));
    }

    #[test]
    fn gameplay_exactness_and_false_edit_fields_derive_from_the_unchanged_mask() {
        let sample = sample(1);
        let current = gameplay(&sample.current);
        let target = gameplay(&sample.next);
        let row = |predictions| SemanticRow {
            predictions,
            log_probs: None,
            current,
            transition_target: target,
            decoder_labels: target,
            sample: &sample,
        };
        // Perfect on the single changed pixel, but edits one unchanged pixel.
        let mut edited = target.to_vec();
        edited[5] = 3;
        let mut accum = SourceAccum::default();
        accum.add_variant("edited", row(&edited));
        accum.add_variant("exact", row(target));
        let source = accum.finish();

        let edited = &source.variants["edited"];
        assert_eq!(edited.masks["changed"].exact_transition_accuracy, Some(1.0));
        assert_eq!(
            edited.masks["gameplay"].exact_transition_accuracy,
            Some(0.0)
        );
        let unchanged_pixels = 48.0;
        assert!((edited.false_edit_rate.unwrap() - 1.0 / unchanged_pixels).abs() < 1e-12);
        assert_eq!(edited.false_edit_transition_rate, Some(1.0));
        assert_eq!(edited.padding_false_edit_rate, Some(0.0));

        // Exactness on the gameplay mask is 1.0 iff every frame pixel matches.
        let exact = &source.variants["exact"];
        assert_eq!(exact.masks["gameplay"].exact_transition_accuracy, Some(1.0));
        assert_eq!(exact.false_edit_rate, Some(0.0));
        assert_eq!(exact.false_edit_transition_rate, Some(0.0));
        assert_eq!(exact.padding_false_edit_rate, Some(0.0));
    }

    #[test]
    fn collision_census_exposes_visible_input_ambiguity_and_ceiling() {
        let rows = vec![sample(1), sample(2)];
        let census = collision_census(&rows, &[("ignored".into(), rows.len())]);
        assert_eq!(census.overall.unique_visible_inputs, 1);
        assert_eq!(census.overall.conflicting_visible_inputs, 1);
        assert_eq!(census.overall.rows_in_conflicts, 2);
        assert_eq!(census.overall.deterministic_exact_ceiling, Some(0.5));
        assert_eq!(census.by_source["movement"].conflicting_visible_inputs, 1);
    }

    #[test]
    fn shuffled_action6_coordinates_are_conjugated_into_target_rectangles() -> Result<()> {
        let mut first = sample(1);
        first.provenance.content_x = 10;
        first.provenance.content_y = 20;
        first.action = ArcAction::new(6, Some(12), Some(23))?;

        let mut second = sample(2);
        second.provenance.content_x = 30;
        second.provenance.content_y = 40;
        second.action = ArcAction::new(6, Some(34), Some(45))?;

        let shuffled = shuffled_action_control_samples(&[first.clone(), second.clone()]);
        assert_eq!(shuffled[0].action, ArcAction::new(6, Some(14), Some(25))?);
        assert_eq!(shuffled[1].action, ArcAction::new(6, Some(32), Some(43))?);
        for (target, shuffled) in [first, second].iter().zip(&shuffled) {
            let x = shuffled.action.x.expect("ACTION6 x");
            let y = shuffled.action.y.expect("ACTION6 y");
            assert!(x >= target.provenance.content_x as u8);
            assert!(x < (target.provenance.content_x + target.provenance.content_width) as u8);
            assert!(y >= target.provenance.content_y as u8);
            assert!(y < (target.provenance.content_y + target.provenance.content_height) as u8);
        }
        Ok(())
    }

    #[test]
    fn shuffled_toggle_inert_coordinates_are_not_outcome_changing() -> Result<()> {
        let current = ArcFrame::new(
            FRAME_SIDE as u16,
            FRAME_SIDE as u16,
            vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE],
        )?;
        // Equal EMPTY toggle colors make both selected cells explicitly inert
        // without changing the production operator semantics under test.
        let operator = EpisodeOperator {
            family: OperatorFamily::Toggle,
            agent_color: palette::AGENT,
            primary_color: palette::EMPTY,
            secondary_color: palette::EMPTY,
            empty_color: palette::EMPTY,
        };
        let (first, first_provenance) = operator_row(
            current.clone(),
            ArcAction::new(6, Some(1), Some(1))?,
            operator,
            1,
        )?;
        let (second, second_provenance) = operator_row(
            current,
            ArcAction::new(6, Some(5), Some(5))?,
            operator,
            2,
        )?;
        let factual = vec![first, second];
        let control = shuffled_action_control_population(
            &factual,
            Some(&[first_provenance, second_provenance]),
        )?;

        assert_eq!(control.samples.len(), 2);
        assert_eq!(control.eligible_rows, 2);
        assert_eq!(control.changed_tuples(&factual), 2);
        assert_eq!(control.outcome_changing(&factual), vec![Some(false); 2]);
        assert_eq!(control.outcome_changing_tuples(&factual), Some(0));
        Ok(())
    }

    #[test]
    fn shuffled_teleport_coordinate_that_moves_the_agent_is_outcome_changing() -> Result<()> {
        let mut pixels = vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE];
        pixels[FRAME_SIDE + 1] = palette::AGENT;
        let current = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels)?;
        let operator = EpisodeOperator {
            family: OperatorFamily::Teleport,
            agent_color: palette::AGENT,
            primary_color: palette::SWITCH_BASE,
            secondary_color: palette::SWITCH_BASE + 1,
            empty_color: palette::EMPTY,
        };
        let (first, first_provenance) = operator_row(
            current.clone(),
            ArcAction::new(6, Some(2), Some(2))?,
            operator,
            1,
        )?;
        let (second, second_provenance) = operator_row(
            current,
            ArcAction::new(6, Some(5), Some(5))?,
            operator,
            2,
        )?;
        let factual = vec![first, second];
        let provenance = vec![first_provenance, second_provenance];
        let control = shuffled_action_control_population(&factual, Some(&provenance))?;

        assert_eq!(control.changed_tuples(&factual), 2);
        assert_eq!(control.outcome_changing(&factual), vec![Some(true); 2]);
        assert_eq!(control.outcome_changing_tuples(&factual), Some(2));
        for ((sample, shuffled), counterfactual) in factual
            .iter()
            .zip(&control.samples)
            .zip(&control.counterfactual_next)
        {
            let expected = apply_episode_operator(
                &sample.current,
                &shuffled.action,
                provenance[0].content_rect,
                operator,
            )?;
            assert_eq!(counterfactual.as_ref(), Some(&expected));
        }
        Ok(())
    }

    #[test]
    fn fixed_foundation_gate_population_has_causal_shuffle_support() -> Result<()> {
        let batch = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 512,
                seed: FOUNDATION_V2_GATE_SEED,
                schedule: foundation_v2_stream_schedule,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            V5DataSplit::UnseenSeed7x7,
        )?;
        let samples = batch.transitions().cloned().collect::<Vec<_>>();
        let provenance = batch
            .samples()
            .iter()
            .map(|sample| sample.provenance.clone())
            .collect::<Vec<_>>();
        let control = shuffled_action_control_population(&samples, Some(&provenance))?;
        let outcome_changing = control
            .outcome_changing_tuples(&samples)
            .expect("V5 sidecars cover the entire gate population");

        assert!(control.changed_tuples(&samples) >= outcome_changing);
        assert!(
            outcome_changing >= crate::p2::eval::MIN_SHUFFLED_ACTION_OUTCOME_CHANGING_ROWS,
            "fixed gate produced only {outcome_changing} causal interventions"
        );
        Ok(())
    }

    #[test]
    fn full_v4_semantics_keep_movement_hazard_action5_and_action6_sources_separate() -> Result<()> {
        let device = Device::Cpu;
        let mut config = TrainConfig::default();
        config.apply_full_v4_recipe();
        config.hidden_dim = 16;
        config.action_dim = 4;
        config.inner_steps = 1;
        config.outer_steps = 1;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            config.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, config.seed)?;
        let samples = generate_curriculum("random_one_step", 91, 7, Split::Train)?;
        let metrics = evaluate_semantics(
            &model,
            &samples,
            &[("mixed_parent".into(), samples.len())],
            2,
            &device,
        )?;
        for source in [
            "dynamics",
            "hazard_failure",
            "action5_interact",
            "coordinate_action",
        ] {
            assert!(metrics.by_source.contains_key(source), "missing {source}");
        }
        for variant in [
            "current_reconstruction",
            "target_reconstruction",
            "one_step_prediction",
            "learned_copy_control",
            "hard_copy_control",
            "zero_control",
            "direct_target_positive_control",
            "action_shuffled_prediction",
            "composed_copy_gate",
        ] {
            assert!(metrics.overall.variants.contains_key(variant));
        }
        assert!(metrics.overall.raw_composed_pixel_disagreement.is_some());
        assert!(metrics.overall.copy_gate_open_rate_unchanged.is_some());
        for variant in metrics.overall.variants.values() {
            assert!(variant.masks.contains_key("gameplay"));
        }
        assert!(!metrics
            .overall
            .variants
            .contains_key("action_masked_prediction"));
        assert!(metrics.overall.variants.values().all(|variant| {
            NONCOMPARABLE_CONTENT_MASKS
                .iter()
                .all(|mask| !variant.masks.contains_key(*mask))
        }));
        assert!(metrics.by_source.values().all(|source| {
            source
                .variants
                .values()
                .all(|variant| variant.masks.contains_key("content"))
        }));
        Ok(())
    }
}
