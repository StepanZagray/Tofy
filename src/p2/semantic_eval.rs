//! Full V4 semantic evaluation at the exact-decoder seam.

use crate::p2::data::{palette, TransitionSample, FRAME_SIDE};
use crate::p2::model::{WorldModel, PALETTE_SIZE};
use crate::p2::train::{action_tensors_from_samples, batch_from_samples};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SEMANTIC_EVAL_SCHEMA: &str = "p2.semantic_eval.v1";

const MASKS: [&str; 6] = [
    "content",
    "padding",
    "foreground",
    "changed",
    "unchanged",
    "changed_content",
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
    SemanticDecoderMetrics {
        masks: masks
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
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSourceMetrics {
    pub transitions: usize,
    pub status_pixels: usize,
    pub variants: BTreeMap<String, SemanticDecoderMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEvaluation {
    pub schema: String,
    pub mask_contract: String,
    pub reduction_contract: String,
    pub action_mask_contract: String,
    pub overall: SemanticSourceMetrics,
    pub by_source: BTreeMap<String, SemanticSourceMetrics>,
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

    fn finish(self) -> SemanticSourceMetrics {
        SemanticSourceMetrics {
            transitions: self.transitions,
            status_pixels: self.status_pixels,
            variants: self
                .variants
                .into_iter()
                .map(|(variant, masks)| {
                    (
                        variant,
                        SemanticDecoderMetrics {
                            masks: masks
                                .into_iter()
                                .map(|(name, metrics)| (name, metrics.finish()))
                                .collect(),
                        },
                    )
                })
                .collect(),
        }
    }
}

fn semantic_masks(current: &[u8], target: &[u8], sample: &TransitionSample) -> [Vec<bool>; 6] {
    let gameplay_pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
    let mut content = vec![false; gameplay_pixels];
    let width = usize::from(sample.provenance.content_width).min(FRAME_SIDE);
    let height = usize::from(sample.provenance.content_height).min(FRAME_SIDE - 1);
    for y in 0..height {
        for x in 0..width {
            content[y * FRAME_SIDE + x] = true;
        }
    }
    let padding = content.iter().map(|selected| !selected).collect();
    let foreground = target
        .iter()
        .map(|pixel| *pixel != palette::EMPTY)
        .collect();
    let changed: Vec<_> = current
        .iter()
        .zip(target)
        .map(|(before, after)| before != after)
        .collect();
    let unchanged = changed.iter().map(|selected| !selected).collect();
    let changed_content = changed
        .iter()
        .zip(&content)
        .map(|(changed, content)| *changed && *content)
        .collect();
    [
        content,
        padding,
        foreground,
        changed,
        unchanged,
        changed_content,
    ]
}

struct DecodedRows {
    predictions: Vec<Vec<u8>>,
    log_probs: Vec<Vec<f32>>,
}

fn decoded_rows(logits: &Tensor) -> Result<DecodedRows> {
    let batch = logits.dim(0)?;
    let pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
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
    let current = gameplay(&sample.current);
    let target = gameplay(&sample.next);
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
    Ok(variants
        .remove("rollout")
        .expect("rollout variant was inserted"))
}

fn gameplay(frame: &crate::p2::data::ArcFrame) -> &[u8] {
    let end = ((FRAME_SIDE - 1) * FRAME_SIDE).min(frame.pixels.len());
    &frame.pixels[..end]
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

fn shuffled_samples(
    samples: &[TransitionSample],
    source_lengths: &[(String, usize)],
) -> Vec<TransitionSample> {
    let mut shuffled = samples.to_vec();
    let mut start = 0usize;
    for (_, len) in source_lengths {
        let end = (start + *len).min(samples.len());
        if end > start + 1 {
            for (index, row) in shuffled.iter_mut().enumerate().take(end).skip(start) {
                let donor = start + (index - start + 1) % (end - start);
                row.action = samples[donor].action.clone();
            }
        }
        start = end;
    }
    shuffled
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
    ensure!(
        model.config().world_core_v4,
        "semantic evaluation requires Full V4"
    );
    let labels = source_labels(samples, source_lengths);
    let shuffled = shuffled_samples(samples, source_lengths);
    let mut overall = SourceAccum::default();
    let mut by_source = BTreeMap::<String, SourceAccum>::new();
    for start in (0..samples.len()).step_by(physical_batch.max(1)) {
        let end = (start + physical_batch.max(1)).min(samples.len());
        let rows = &samples[start..end];
        let batch = batch_from_samples(rows, device)?;
        let (current_latent, target_latent) =
            model.encode_state_pair(&batch.frames, &batch.next_frames)?;
        let prediction = model
            .forward_from_latent(
                &current_latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
            )?
            .y;
        let masked_actions = Tensor::zeros((rows.len(),), DType::U32, device)?;
        let masked_coords = Tensor::zeros((rows.len(), 2), DType::F32, device)?;
        let masked_prediction = model
            .forward_from_latent(
                &current_latent,
                &masked_actions,
                &masked_coords,
                &batch.goals,
            )?
            .y;
        let (shuffled_actions, shuffled_coords) =
            action_tensors_from_samples(&shuffled[start..end], device)?;
        let shuffled_prediction = model
            .forward_from_latent(
                &current_latent,
                &shuffled_actions,
                &shuffled_coords,
                &batch.goals,
            )?
            .y;
        let decoded = [
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
                "action_masked_prediction",
                model.exact_gameplay_logits(&masked_prediction)?,
            ),
            (
                "action_shuffled_prediction",
                model.exact_gameplay_logits(&shuffled_prediction)?,
            ),
        ];
        let decoded = decoded
            .into_iter()
            .map(|(name, logits)| Ok((name, decoded_rows(&logits)?)))
            .collect::<Result<Vec<_>>>()?;
        for (local, sample) in rows.iter().enumerate() {
            let current = gameplay(&sample.current);
            let target = gameplay(&sample.next);
            ensure!(
                current.len() == (FRAME_SIDE - 1) * FRAME_SIDE
                    && target.len() == (FRAME_SIDE - 1) * FRAME_SIDE,
                "semantic evaluation requires fixed 64x64 frames"
            );
            overall.transitions += 1;
            overall.status_pixels += FRAME_SIDE;
            let source = by_source.entry(labels[start + local].clone()).or_default();
            source.transitions += 1;
            source.status_pixels += FRAME_SIDE;
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
            let copy = current;
            let zero = vec![palette::EMPTY; target.len()];
            for (name, prediction) in [
                ("hard_copy_control", copy),
                ("zero_control", zero.as_slice()),
                ("direct_target_positive_control", target),
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
    Ok(SemanticEvaluation {
        schema: SEMANTIC_EVAL_SCHEMA.into(),
        mask_contract: "gameplay=rows[0,63); content=[0,width)x[0,height); padding=gameplay-content; foreground=target!=EMPTY; changed=current!=target; unchanged=gameplay-changed; status=row63 excluded from decoder metrics".into(),
        reduction_contract: "pixel aggregate plus equal-transition mean; every source is also reported independently".into(),
        action_mask_contract: "masked uses action id 0 and zero coordinates; shuffled rotates actions only within each named source".into(),
        overall: overall.finish(),
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
    use crate::p2::data::{
        generate_curriculum, ArcAction, ArcFrame, GoalFeatures, TransitionProvenance, FRAME_SIDE,
    };
    use crate::p2::train::{reinit_varmap_deterministic, TrainConfig};
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
                source_kind: "movement".into(),
                trajectory_id: "sim/Train/9/4".into(),
            },
            oracle_latent: None,
        }
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
            "action_masked_prediction",
            "action_shuffled_prediction",
        ] {
            assert!(metrics.overall.variants.contains_key(variant));
        }
        Ok(())
    }
}
