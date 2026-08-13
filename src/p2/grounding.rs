//! Training-only semantic grounding for spatial world-model latents.
//!
//! The head is instantiated in every model so objective on/off comparisons keep
//! an identical parameter topology and name-seeded initialization.

use crate::p2::data::TransitionSample;
use crate::p2::model::{FRAME_SIDE, LATENT_GRID, PALETTE_SIZE};
use anyhow::{ensure, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

const PATCH_SIDE: usize = FRAME_SIDE / LATENT_GRID;

/// Which half of the patch-histogram grounding bundle contributes gradients.
/// The shared decoder and both raw losses exist in every mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PatchGroundingMode {
    Target,
    Predicted,
    #[default]
    Both,
}

#[derive(Debug)]
pub struct PatchGroundingLoss {
    pub total: Tensor,
    pub changed_patches: usize,
    pub unchanged_patches: usize,
}

/// Shared linear decoder from one spatial latent token to a normalized
/// 16-colour histogram for the corresponding observation patch.
pub struct PatchHistogramGrounding {
    decoder: Linear,
}

impl PatchHistogramGrounding {
    pub fn new(hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            decoder: linear(hidden_dim, PALETTE_SIZE, vb.pp("decoder"))?,
        })
    }

    /// Average target-latent and predicted-latent soft cross entropy. Changed
    /// and unchanged patches each receive half of the mass when both exist.
    pub fn loss(
        &self,
        predicted: &Tensor,
        target: &Tensor,
        samples: &[TransitionSample],
        mode: PatchGroundingMode,
    ) -> Result<PatchGroundingLoss> {
        let (batch, hidden, height, width) = predicted.dims4()?;
        ensure!(
            target.dims4()? == (batch, hidden, height, width),
            "grounding target and prediction shapes must match"
        );
        ensure!(
            height == LATENT_GRID && width == LATENT_GRID,
            "grounding expects {LATENT_GRID}x{LATENT_GRID} spatial latents"
        );
        ensure!(
            samples.len() == batch,
            "grounding sample count {} != latent batch {batch}",
            samples.len()
        );

        let (histograms, changed, changed_count) = patch_targets(samples)?;
        let patch_count = batch * LATENT_GRID * LATENT_GRID;
        let unchanged_count = patch_count - changed_count;
        let weights = balanced_patch_weights(&changed, changed_count, unchanged_count);
        let device = predicted.device();
        let histograms = Tensor::from_vec(histograms, (patch_count, PALETTE_SIZE), device)?;
        let weights = Tensor::from_vec(weights, (patch_count,), device)?;

        let target_loss = self.soft_cross_entropy(target, &histograms, &weights)?;
        let predicted_loss = self.soft_cross_entropy(predicted, &histograms, &weights)?;
        let total = combine_losses(target_loss, predicted_loss, mode)?;
        Ok(PatchGroundingLoss {
            total,
            changed_patches: changed_count,
            unchanged_patches: unchanged_count,
        })
    }

    fn soft_cross_entropy(
        &self,
        latents: &Tensor,
        histograms: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let (batch, hidden, height, width) = latents.dims4()?;
        let tokens = latents
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((batch * height * width, hidden))?;
        let logits = self.decoder.forward(&tokens)?.to_dtype(DType::F32)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let per_patch = log_probs.mul(histograms)?.sum(D::Minus1)?.neg()?;
        per_patch.mul(weights)?.sum_all().map_err(Into::into)
    }
}

fn combine_losses(target: Tensor, predicted: Tensor, mode: PatchGroundingMode) -> Result<Tensor> {
    match mode {
        PatchGroundingMode::Target => Ok(target),
        PatchGroundingMode::Predicted => Ok(predicted),
        PatchGroundingMode::Both => target.add(&predicted)?.affine(0.5, 0.0).map_err(Into::into),
    }
}

fn patch_targets(samples: &[TransitionSample]) -> Result<(Vec<f32>, Vec<bool>, usize)> {
    let patch_count = samples.len() * LATENT_GRID * LATENT_GRID;
    let mut histograms = vec![0.0f32; patch_count * PALETTE_SIZE];
    let mut changed = vec![false; patch_count];
    let mut changed_count = 0usize;

    for (sample_index, sample) in samples.iter().enumerate() {
        for frame in [&sample.current, &sample.next] {
            ensure!(
                frame.width as usize == FRAME_SIDE
                    && frame.height as usize == FRAME_SIDE
                    && frame.pixels.len() == FRAME_SIDE * FRAME_SIDE,
                "grounding requires fixed {FRAME_SIDE}x{FRAME_SIDE} frames"
            );
            ensure!(
                frame
                    .pixels
                    .iter()
                    .all(|pixel| (*pixel as usize) < PALETTE_SIZE),
                "grounding frame contains a palette value outside 0..{}",
                PALETTE_SIZE - 1
            );
        }
        for patch_y in 0..LATENT_GRID {
            for patch_x in 0..LATENT_GRID {
                let patch = (sample_index * LATENT_GRID + patch_y) * LATENT_GRID + patch_x;
                let mut is_changed = false;
                let mut gameplay_pixels = 0usize;
                for dy in 0..PATCH_SIDE {
                    let y = patch_y * PATCH_SIDE + dy;
                    // The final row is a synthetic budget/status display, not board state.
                    if y == FRAME_SIDE - 1 {
                        continue;
                    }
                    for dx in 0..PATCH_SIDE {
                        let x = patch_x * PATCH_SIDE + dx;
                        let pixel = y * FRAME_SIDE + x;
                        let colour = sample.next.pixels[pixel] as usize;
                        histograms[patch * PALETTE_SIZE + colour] += 1.0;
                        gameplay_pixels += 1;
                        is_changed |= sample.current.pixels[pixel] != sample.next.pixels[pixel];
                    }
                }
                ensure!(
                    gameplay_pixels > 0,
                    "grounding patch has no gameplay pixels"
                );
                for colour in 0..PALETTE_SIZE {
                    histograms[patch * PALETTE_SIZE + colour] /= gameplay_pixels as f32;
                }
                changed[patch] = is_changed;
                changed_count += usize::from(is_changed);
            }
        }
    }
    Ok((histograms, changed, changed_count))
}

fn balanced_patch_weights(
    changed: &[bool],
    changed_count: usize,
    unchanged_count: usize,
) -> Vec<f32> {
    if changed_count == 0 || unchanged_count == 0 {
        return vec![1.0 / changed.len().max(1) as f32; changed.len()];
    }
    let changed_weight = 0.5 / changed_count as f32;
    let unchanged_weight = 0.5 / unchanged_count as f32;
    changed
        .iter()
        .map(|is_changed| {
            if *is_changed {
                changed_weight
            } else {
                unchanged_weight
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Split;
    use crate::p2::data::{ArcAction, ArcFrame, GoalFeatures};

    fn sample(current: Vec<u8>, next: Vec<u8>) -> TransitionSample {
        TransitionSample {
            current: ArcFrame::new(64, 64, current).unwrap(),
            next: ArcFrame::new(64, 64, next).unwrap(),
            action: ArcAction::new(1, None, None).unwrap(),
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: Some(false),
            goal_failed: Some(false),
            exhausted: Some(false),
            split: Split::Train,
            family: "test".into(),
            seed: 1,
            episode_id: 1,
            transition_index: 0,
            oracle_latent: None,
        }
    }

    #[test]
    fn balanced_weights_split_mass_between_changed_and_unchanged() {
        let changed = [true, false, false, true, false];
        let weights = balanced_patch_weights(&changed, 2, 3);
        let changed_mass: f32 = weights
            .iter()
            .zip(changed)
            .filter_map(|(weight, changed)| changed.then_some(*weight))
            .sum();
        let unchanged_mass: f32 = weights
            .iter()
            .zip(changed)
            .filter_map(|(weight, changed)| (!changed).then_some(*weight))
            .sum();
        assert!((changed_mass - 0.5).abs() < 1e-6);
        assert!((unchanged_mass - 0.5).abs() < 1e-6);
    }

    #[test]
    fn grounding_decoder_is_routed_as_an_adamw_head() {
        assert!(!crate::p2::muon::uses_muon(
            "grounding_head.decoder.weight",
            &[PALETTE_SIZE, 128],
        ));
    }

    #[test]
    fn patch_targets_exclude_status_row_and_normalize_gameplay_pixels() {
        let current = vec![0; FRAME_SIDE * FRAME_SIDE];
        let mut next = current.clone();
        next[(FRAME_SIDE - 1) * FRAME_SIDE] = 2;
        let (_, _, changed_count) = patch_targets(&[sample(current.clone(), next)]).unwrap();
        assert_eq!(changed_count, 0, "status-only change must be ignored");

        let mut next = current.clone();
        next[(FRAME_SIDE - 2) * FRAME_SIDE] = 1;
        let (histograms, _, changed_count) = patch_targets(&[sample(current, next)]).unwrap();
        assert_eq!(changed_count, 1);
        let bottom_left = (7 * LATENT_GRID) * PALETTE_SIZE;
        assert!((histograms[bottom_left] - 55.0 / 56.0).abs() < 1e-6);
        assert!((histograms[bottom_left + 1] - 1.0 / 56.0).abs() < 1e-6);
        let sum: f32 = histograms[bottom_left..bottom_left + PALETTE_SIZE]
            .iter()
            .sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn grounding_modes_select_exact_bundle_components() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let target = Tensor::new(2f32, &device)?;
        let predicted = Tensor::new(6f32, &device)?;
        assert_eq!(
            combine_losses(
                target.clone(),
                predicted.clone(),
                PatchGroundingMode::Target
            )?
            .to_scalar::<f32>()?,
            2.0
        );
        assert_eq!(
            combine_losses(
                target.clone(),
                predicted.clone(),
                PatchGroundingMode::Predicted
            )?
            .to_scalar::<f32>()?,
            6.0
        );
        assert_eq!(
            combine_losses(target, predicted, PatchGroundingMode::Both)?.to_scalar::<f32>()?,
            4.0
        );
        Ok(())
    }
}
