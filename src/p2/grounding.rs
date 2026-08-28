//! Training-only semantic grounding for spatial world-model latents.
//!
//! The head is instantiated in every model so objective on/off comparisons keep
//! an identical parameter topology and name-seeded initialization.

use crate::p2::data::TransitionSample;
use crate::p2::model::{FRAME_SIDE, PALETTE_SIZE};
use anyhow::{ensure, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

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

fn ensure_composed_decode_finite(tensor: &Tensor, label: &str) -> Result<()> {
    let tensor = tensor.to_dtype(DType::F32)?;
    // Count a bounded 0/1 mask instead of summing the values themselves:
    // finite f32 values can overflow a sum even when every element is valid.
    let finite_count = tensor
        .eq(&tensor)?
        .to_dtype(DType::F32)?
        .mul(&tensor.abs()?.le(f32::MAX)?.to_dtype(DType::F32)?)?
        .sum_all()?
        .to_scalar::<f32>()?;
    ensure!(
        finite_count == tensor.elem_count() as f32,
        "composed decode received non-finite {label}"
    );
    Ok(())
}

/// Shared linear decoder from one spatial latent token to a normalized
/// 16-colour histogram for the corresponding observation patch.
pub struct PatchHistogramGrounding {
    decoder: Linear,
    patch_side: usize,
    latent_grid: usize,
}

/// Composition rule used by the deployed gameplay decode.
///
/// `JointCopyMixture` treats the gate as mixture weight over copy and the
/// color distribution: `P(c) = (1-g)*1[c=current] + g*softmax(logits)_c`,
/// decoded by MAP. Because the copy component always holds mass `1-g`, a
/// sub-0.5 gate can never be overridden; the mixture differs from the hard
/// gate one-directionally, converting above-0.5 gates with unconfident or
/// current-favoring color evidence back into copies (fewer false edits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DecodeComposition {
    #[default]
    LegacyHardGate,
    JointCopyMixture,
}

/// Exact, position-preserving decoder used by Full V4. Each spatial token
/// predicts the palette indices in its corresponding observation patch.
/// The synthetic status row is removed before both loss and correctness are
/// computed.
pub struct ExactPatchGrounding {
    decoder: Linear,
    copy_gate: candle_nn::Conv2d,
    patch_side: usize,
    latent_grid: usize,
    composition: DecodeComposition,
}

impl ExactPatchGrounding {
    pub fn new(
        hidden_dim: usize,
        patch_side: usize,
        copy_gate_bias_prior: Option<f64>,
        composition: DecodeComposition,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        ensure!(
            patch_side > 0 && FRAME_SIDE.is_multiple_of(patch_side),
            "exact grounding patch side must divide {FRAME_SIDE}"
        );
        let gate_out = patch_side * patch_side;
        let gate_vb = vb.pp("copy_gate");
        // Mirror candle's conv2d initialization exactly, overriding only the
        // bias when a changed-pixel prior is configured: composition then
        // starts as calibrated copy instead of a ~50/50 gate. Checkpoint
        // loading overrides init hints, so resumed runs are unaffected.
        let copy_gate = {
            let weight = gate_vb.get_with_hints(
                (gate_out, hidden_dim, 1, 1),
                "weight",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?;
            let bias_init = match copy_gate_bias_prior {
                Some(prior) => {
                    ensure!(
                        prior.is_finite() && prior > 0.0 && prior < 1.0,
                        "copy_gate_bias_prior must be a probability in (0, 1)"
                    );
                    candle_nn::Init::Const((prior / (1.0 - prior)).ln())
                }
                None => {
                    let bound = 1.0 / (hidden_dim as f64).sqrt();
                    candle_nn::Init::Uniform {
                        lo: -bound,
                        up: bound,
                    }
                }
            };
            let bias = gate_vb.get_with_hints(gate_out, "bias", bias_init)?;
            candle_nn::Conv2d::new(weight, Some(bias), Default::default())
        };
        Ok(Self {
            decoder: linear(
                hidden_dim,
                patch_side * patch_side * PALETTE_SIZE,
                vb.pp("decoder"),
            )?,
            copy_gate,
            patch_side,
            latent_grid: FRAME_SIDE / patch_side,
            composition,
        })
    }

    /// `B×63×64×16` logits in gameplay-pixel order. The full per-token
    /// projection is immediately rearranged and the status row is discarded.
    pub fn gameplay_logits(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, hidden, height, width) = latents.dims4()?;
        ensure!(
            height == self.latent_grid && width == self.latent_grid,
            "exact grounding expects {}x{} spatial latents",
            self.latent_grid,
            self.latent_grid
        );
        let tokens = latents
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((batch * height * width, hidden))?;
        let patch_logits = self.decoder.forward(&tokens)?.to_dtype(DType::F32)?;
        let patch_logits = patch_logits.reshape((
            batch,
            self.latent_grid,
            self.latent_grid,
            self.patch_side,
            self.patch_side,
            PALETTE_SIZE,
        ))?;
        patch_tokens_to_pixels(&patch_logits)?
            .narrow(1, 0, FRAME_SIDE - 1)
            .map_err(Into::into)
    }

    /// Per-gameplay-pixel copy/change logits decoded from a predicted latent.
    pub fn copy_gate_logits(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, _, height, width) = latents.dims4()?;
        ensure!(
            height == self.latent_grid && width == self.latent_grid,
            "copy gate expects {}x{} spatial latents",
            self.latent_grid,
            self.latent_grid
        );
        let patch_logits = self.copy_gate.forward(latents)?.to_dtype(DType::F32)?;
        let patch_logits = patch_logits.permute((0, 2, 3, 1))?.contiguous()?.reshape((
            batch,
            self.latent_grid,
            self.latent_grid,
            self.patch_side,
            self.patch_side,
            1,
        ))?;
        patch_tokens_to_pixels(&patch_logits)?
            .squeeze(3)?
            .narrow(1, 0, FRAME_SIDE - 1)
            .map_err(Into::into)
    }

    /// Per-gameplay-pixel copy/change probability decoded from a predicted latent.
    pub fn copy_gate(&self, latents: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(&self.copy_gate_logits(latents)?).map_err(Into::into)
    }

    pub fn loss(&self, latents: &Tensor, frames: &Tensor) -> Result<Tensor> {
        let batch = latents.dim(0)?;
        ensure!(
            frames.dims4()? == (batch, 1, FRAME_SIDE, FRAME_SIDE),
            "exact grounding frames must be Bx1x{FRAME_SIDE}x{FRAME_SIDE}"
        );
        let logits = self
            .gameplay_logits(latents)?
            .reshape((batch * (FRAME_SIDE - 1) * FRAME_SIDE, PALETTE_SIZE))?;
        let labels = frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .contiguous()?
            .flatten_all()?
            .to_dtype(DType::U32)?;
        candle_nn::loss::cross_entropy(&logits, &labels).map_err(Into::into)
    }

    /// Frozen, pixel-derived transition labels for observer heads. A positive
    /// requires >=99% gameplay accuracy and, when the transition changes any
    /// gameplay pixels, >=90% accuracy on those changed pixels.
    pub fn raw_decoder_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        let predicted = self
            .gameplay_logits(predicted)?
            .detach()
            .argmax(D::Minus1)?;
        let current = current_frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        let target = next_frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        transition_correctness_from_gameplay(&predicted, &current, &target)
    }

    /// Deployed discrete gameplay decode: detached logits and gate composed
    /// per the configured [`DecodeComposition`]. Returns `B×63×64` U32 pixels.
    pub fn compose_gameplay_pixels(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
    ) -> Result<Tensor> {
        let batch = predicted.dim(0)?;
        ensure!(
            current_frames.dims4()? == (batch, 1, FRAME_SIDE, FRAME_SIDE),
            "current frames must be Bx1x{FRAME_SIDE}x{FRAME_SIDE}"
        );
        let logits = self.gameplay_logits(predicted)?.detach();
        let current = current_frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        let gate = self.copy_gate(predicted)?.detach();
        // Fail closed on numerical corruption: comparison ops treat NaN as
        // false, so a NaN gate or logit row would otherwise decode as a
        // spuriously valid copy and inflate copy-heavy exactness metrics.
        for (tensor, label) in [(&logits, "gameplay logits"), (&gate, "copy gate")] {
            ensure_composed_decode_finite(tensor, label)?;
        }
        match self.composition {
            DecodeComposition::LegacyHardGate => {
                let predicted_pixels = logits.argmax(D::Minus1)?;
                gate.ge(0.5)?
                    .where_cond(&predicted_pixels, &current)
                    .map_err(Into::into)
            }
            DecodeComposition::JointCopyMixture => {
                // MAP of the mixture reduces exactly to two candidates: the
                // color argmax (mass g*p_max) and the current pixel (mass
                // (1-g) + g*p_cur); no other color can exceed both.
                let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
                let predicted_pixels = logits.argmax(D::Minus1)?;
                let p_max = probs.max(D::Minus1)?;
                let p_cur = probs
                    .gather(&current.unsqueeze(D::Minus1)?, D::Minus1)?
                    .squeeze(D::Minus1)?;
                let edit_score = gate.mul(&p_max)?;
                let copy_score = gate.affine(-1.0, 1.0)?.add(&gate.mul(&p_cur)?)?;
                edit_score
                    .gt(&copy_score)?
                    .where_cond(&predicted_pixels, &current)
                    .map_err(Into::into)
            }
        }
    }

    pub fn composed_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        let current = current_frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        let target = next_frames
            .narrow(2, 0, FRAME_SIDE - 1)?
            .squeeze(1)?
            .to_dtype(DType::U32)?;
        let composed = self.compose_gameplay_pixels(predicted, current_frames)?;
        transition_correctness_from_gameplay(&composed, &current, &target)
    }

    /// Compatibility seam for observer labels. This intentionally uses the
    /// deployed copy-gate composition rather than raw decoder colours.
    pub fn transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        self.composed_transition_correctness(predicted, current_frames, next_frames)
    }
}

fn transition_correctness_from_gameplay(
    predicted: &Tensor,
    current: &Tensor,
    target: &Tensor,
) -> Result<Tensor> {
    let batch = predicted.dim(0)?;
    ensure!(
        current.dims3()? == (batch, FRAME_SIDE - 1, FRAME_SIDE)
            && target.dims3()? == (batch, FRAME_SIDE - 1, FRAME_SIDE),
        "exact correctness inputs must share a Bx63x64 gameplay grid"
    );
    ensure!(
        predicted.dims3()? == (batch, FRAME_SIDE - 1, FRAME_SIDE),
        "exact decoder produced an invalid gameplay grid"
    );
    let correct = predicted.eq(target)?.to_dtype(DType::F32)?;
    let changed = current.ne(target)?.to_dtype(DType::F32)?;
    let correct_flat = correct.reshape((batch, (FRAME_SIDE - 1) * FRAME_SIDE))?;
    let changed_flat = changed.reshape((batch, (FRAME_SIDE - 1) * FRAME_SIDE))?;
    let overall_ok = correct_flat
        .mean_keepdim(D::Minus1)?
        .ge(0.99)?
        .to_dtype(DType::F32)?;
    let changed_count = changed_flat.sum_keepdim(D::Minus1)?;
    let changed_accuracy = correct_flat
        .mul(&changed_flat)?
        .sum_keepdim(D::Minus1)?
        .div(&changed_count.clamp(1.0, f64::INFINITY)?)?;
    let no_change = changed_count.eq(0f32)?.to_dtype(DType::F32)?;
    let changed_ok = changed_accuracy
        .ge(0.9)?
        .to_dtype(DType::F32)?
        .add(&no_change)?
        .clamp(0.0, 1.0)?;
    overall_ok.mul(&changed_ok).map_err(Into::into)
}

impl PatchHistogramGrounding {
    pub fn new(hidden_dim: usize, patch_side: usize, vb: VarBuilder<'_>) -> Result<Self> {
        ensure!(
            patch_side > 0 && FRAME_SIDE.is_multiple_of(patch_side),
            "patch grounding patch side must divide {FRAME_SIDE}"
        );
        Ok(Self {
            decoder: linear(hidden_dim, PALETTE_SIZE, vb.pp("decoder"))?,
            patch_side,
            latent_grid: FRAME_SIDE / patch_side,
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
            height == self.latent_grid && width == self.latent_grid,
            "grounding expects {}x{} spatial latents",
            self.latent_grid,
            self.latent_grid
        );
        ensure!(
            samples.len() == batch,
            "grounding sample count {} != latent batch {batch}",
            samples.len()
        );

        let (histograms, changed, changed_count) =
            patch_targets(samples, self.latent_grid, self.patch_side)?;
        let patch_count = batch * self.latent_grid * self.latent_grid;
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

fn patch_targets(
    samples: &[TransitionSample],
    latent_grid: usize,
    patch_side: usize,
) -> Result<(Vec<f32>, Vec<bool>, usize)> {
    let patch_count = samples.len() * latent_grid * latent_grid;
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
        for patch_y in 0..latent_grid {
            for patch_x in 0..latent_grid {
                let patch = (sample_index * latent_grid + patch_y) * latent_grid + patch_x;
                let mut is_changed = false;
                let mut gameplay_pixels = 0usize;
                for dy in 0..patch_side {
                    let y = patch_y * patch_side + dy;
                    // The final row is a synthetic budget/status display, not board state.
                    if y == FRAME_SIDE - 1 {
                        continue;
                    }
                    for dx in 0..patch_side {
                        let x = patch_x * patch_side + dx;
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

/// Rearrange `[B, patch-y, patch-x, dy, dx, channels]` into pixel order.
///
/// Keeping this operation in one geometry seam prevents the exact palette
/// decoder and copy gate from silently disagreeing about sub-pixel order.
pub fn patch_tokens_to_pixels(patches: &Tensor) -> Result<Tensor> {
    let dims = patches.dims();
    ensure!(
        dims.len() == 6,
        "patch token rearrangement expects rank 6, got {}",
        dims.len()
    );
    let (batch, grid_y, grid_x, patch_y, patch_x, channels) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    ensure!(patch_y == patch_x, "decoder patches must be square");
    patches
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((batch, grid_y * patch_y, grid_x * patch_x, channels))
        .map_err(Into::into)
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
    use candle_nn::{VarBuilder, VarMap};

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
            provenance: crate::p2::data::TransitionProvenance::full_frame(
                1,
                1,
                Split::Train,
                "test",
            ),
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
    fn composed_decode_fails_closed_on_non_finite_latents() -> Result<()> {
        // Comparison ops treat NaN as false, so without the guard a NaN gate
        // or logit row decodes as a spuriously valid copy.
        let device = candle_core::Device::Cpu;
        let vars = VarMap::new();
        for composition in [
            DecodeComposition::LegacyHardGate,
            DecodeComposition::JointCopyMixture,
        ] {
            let head = ExactPatchGrounding::new(
                4,
                4,
                None,
                composition,
                VarBuilder::from_varmap(&vars, DType::F32, &device),
            )?;
            let current = Tensor::zeros((1, 1, FRAME_SIDE, FRAME_SIDE), DType::U32, &device)?;
            let finite = Tensor::zeros((1, 4, 16, 16), DType::F32, &device)?;
            head.compose_gameplay_pixels(&finite, &current)?;
            let poisoned = Tensor::full(f32::NAN, (1, 4, 16, 16), &device)?;
            assert!(head.compose_gameplay_pixels(&poisoned, &current).is_err());
        }
        Ok(())
    }

    #[test]
    fn composed_decode_finiteness_check_does_not_overflow() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let large_finite = Tensor::full(1e38f32, (1024,), &device)?;
        ensure_composed_decode_finite(&large_finite, "test tensor")?;

        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let poisoned = Tensor::from_vec(vec![1.0f32, value], (2,), &device)?;
            assert!(ensure_composed_decode_finite(&poisoned, "test tensor").is_err());
        }
        Ok(())
    }

    #[test]
    fn exact_grounding_ignores_status_only_differences() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let vars = VarMap::new();
        let head = ExactPatchGrounding::new(
            4,
            4,
            None,
            DecodeComposition::default(),
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        let latents = Tensor::zeros((1, 4, 16, 16), DType::F32, &device)?;
        let clean = Tensor::zeros((1, 1, FRAME_SIDE, FRAME_SIDE), DType::U32, &device)?;
        let status = Tensor::ones((1, 1, 1, FRAME_SIDE), DType::U32, &device)?;
        let changed = Tensor::cat(&[&clean.narrow(2, 0, FRAME_SIDE - 1)?, &status], 2)?;
        let clean_loss = head.loss(&latents, &clean)?.to_scalar::<f32>()?;
        let changed_loss = head.loss(&latents, &changed)?.to_scalar::<f32>()?;
        assert!((clean_loss - changed_loss).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn exact_correctness_enforces_overall_and_changed_pixel_thresholds() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
        let current_values = vec![0u32; pixels];
        let mut target_values = current_values.clone();
        target_values[..10].fill(1);
        let current = Tensor::from_vec(
            current_values.clone(),
            (1, FRAME_SIDE - 1, FRAME_SIDE),
            &device,
        )?;
        let target = Tensor::from_vec(
            target_values.clone(),
            (1, FRAME_SIDE - 1, FRAME_SIDE),
            &device,
        )?;

        let mut eighty_percent_changed = target_values.clone();
        eighty_percent_changed[8..10].fill(0);
        let predicted = Tensor::from_vec(
            eighty_percent_changed,
            (1, FRAME_SIDE - 1, FRAME_SIDE),
            &device,
        )?;
        assert_eq!(
            transition_correctness_from_gameplay(&predicted, &current, &target)?
                .to_vec2::<f32>()?[0][0],
            0.0
        );

        let mut ninety_percent_changed = target_values;
        ninety_percent_changed[9] = 0;
        let predicted = Tensor::from_vec(
            ninety_percent_changed,
            (1, FRAME_SIDE - 1, FRAME_SIDE),
            &device,
        )?;
        assert_eq!(
            transition_correctness_from_gameplay(&predicted, &current, &target)?
                .to_vec2::<f32>()?[0][0],
            1.0
        );

        let no_change_target = current.clone();
        let mut too_many_errors = current_values;
        too_many_errors[..41].fill(1);
        let predicted =
            Tensor::from_vec(too_many_errors, (1, FRAME_SIDE - 1, FRAME_SIDE), &device)?;
        assert_eq!(
            transition_correctness_from_gameplay(&predicted, &current, &no_change_target)?
                .to_vec2::<f32>()?[0][0],
            0.0
        );
        Ok(())
    }

    #[test]
    fn patch_targets_exclude_status_row_and_normalize_gameplay_pixels() {
        let current = vec![0; FRAME_SIDE * FRAME_SIDE];
        let mut next = current.clone();
        next[(FRAME_SIDE - 1) * FRAME_SIDE] = 2;
        let (_, _, changed_count) = patch_targets(&[sample(current.clone(), next)], 8, 8).unwrap();
        assert_eq!(changed_count, 0, "status-only change must be ignored");

        let mut next = current.clone();
        next[(FRAME_SIDE - 2) * FRAME_SIDE] = 1;
        let (histograms, _, changed_count) = patch_targets(&[sample(current, next)], 8, 8).unwrap();
        assert_eq!(changed_count, 1);
        let bottom_left = (7 * 8) * PALETTE_SIZE;
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
