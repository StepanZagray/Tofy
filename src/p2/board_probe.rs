//! A small, CPU-only probe for whether patch latents retain board state.
//!
//! The probe intentionally has no dependency on evaluation code: it converts each fixed ARC board
//! into 64 palette-count targets and fits a ridge decoder from independently supplied patch
//! latents.  `FIXED_TARGET_DECODER_MSE_CEILING` is deliberately a fixed gate rather than a
//! data-dependent threshold: a transition score is trusted only when the held-out target-latent
//! decoder MSE is at most that ceiling.

use anyhow::{bail, ensure, Result};
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::p2::data::{ArcFrame, FRAME_SIDE};

/// Number of patches along either side of the fixed 64 by 64 board.
pub const PATCHES_PER_SIDE: usize = 8;
/// Number of board patches represented by this probe.
pub const PATCH_COUNT: usize = PATCHES_PER_SIDE * PATCHES_PER_SIDE;
/// Palette IDs are the ARC categorical values 0 through 15.
pub const PALETTE_SIZE: usize = 16;
/// L2 penalty used for every fitted decoder.  It is part of the persisted probe contract.
pub const RIDGE: f64 = 1e-2;
/// A held-out target decoder MSE at or below this fixed value is required for trusted metrics.
pub const FIXED_TARGET_DECODER_MSE_CEILING: f64 = 1e-3;
/// A patch is predicted changed when its predicted-vs-current histogram MSE exceeds this value.
pub const FIXED_PREDICTION_DELTA_THRESHOLD: f64 = 0.01;

/// Opaque row-major patch population. The board-probe module owns the
/// `B×C×8×8 -> (B*64)×C` mapping used by fit and held-out scoring.
#[derive(Clone, Debug, PartialEq)]
pub struct BoardProbeRows(Vec<Vec<f32>>);

impl BoardProbeRows {
    pub fn from_spatial_latent(latent: &Tensor) -> Result<Self> {
        let (batch, channels, height, width) = latent.dims4()?;
        ensure!(
            channels > 0 && height == PATCHES_PER_SIDE && width == PATCHES_PER_SIDE,
            "board probe requires BxCx8x8 spatial latents"
        );
        let values = latent
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut rows = Vec::with_capacity(batch * PATCH_COUNT);
        for sample in 0..batch {
            for y in 0..PATCHES_PER_SIDE {
                for x in 0..PATCHES_PER_SIDE {
                    rows.push(
                        (0..channels)
                            .map(|channel| {
                                values[((sample * channels + channel) * PATCHES_PER_SIDE + y)
                                    * PATCHES_PER_SIDE
                                    + x]
                            })
                            .collect(),
                    );
                }
            }
        }
        Ok(Self(rows))
    }

    pub fn as_rows(&self) -> &[Vec<f32>] {
        &self.0
    }

    pub fn append(&mut self, other: Self) {
        self.0.extend(other.0);
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self> {
        ensure!(
            range.end <= self.0.len(),
            "board probe row slice is out of bounds"
        );
        Ok(Self(self.0[range].to_vec()))
    }
}

/// Held-out board facts aligned with a [`BoardProbeRows`] population.
#[derive(Clone, Debug, PartialEq)]
pub struct BoardProbeTransitions {
    current: Vec<ArcFrame>,
    target: Vec<ArcFrame>,
    exact_changed_patches: Vec<bool>,
}

impl BoardProbeTransitions {
    pub fn try_new(current: Vec<ArcFrame>, target: Vec<ArcFrame>) -> Result<Self> {
        ensure!(!target.is_empty(), "held-out frames must not be empty");
        ensure!(
            current.len() == target.len(),
            "current and target frame counts differ"
        );
        let mut exact_changed_patches = Vec::with_capacity(target.len() * PATCH_COUNT);
        for (before, after) in current.iter().zip(&target) {
            ensure_fixed_frame(before)?;
            ensure_fixed_frame(after)?;
            for patch_y in 0..PATCHES_PER_SIDE {
                for patch_x in 0..PATCHES_PER_SIDE {
                    let y_start = patch_y * PATCHES_PER_SIDE;
                    let y_end = ((patch_y + 1) * PATCHES_PER_SIDE).min(FRAME_SIDE - 1);
                    let x_start = patch_x * PATCHES_PER_SIDE;
                    let x_end = (patch_x + 1) * PATCHES_PER_SIDE;
                    exact_changed_patches.push((y_start..y_end).any(|y| {
                        (x_start..x_end).any(|x| {
                            before.pixels[y * FRAME_SIDE + x] != after.pixels[y * FRAME_SIDE + x]
                        })
                    }));
                }
            }
        }
        Ok(Self {
            current,
            target,
            exact_changed_patches,
        })
    }
}

/// A fitted linear probe from standardized C-dimensional patch latents to palette histograms.
///
/// `weights` is `[input_dim][16]`.  Predictions are `output_mean + standardized_input * weights`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FixedBoardProbe {
    pub input_dim: usize,
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    pub output_mean: [f32; PALETTE_SIZE],
    pub weights: Vec<[f32; PALETTE_SIZE]>,
    pub ridge: f64,
}

/// Held-out transition diagnostics produced by [`FixedBoardProbe::summarize_held_out`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoardTransitionMetrics {
    /// MSE decoding target board histograms from target latents, on held-out rows.
    pub target_latent_decoder_mse: f64,
    /// MSE decoding next-board histograms from predicted-next latents.
    pub predicted_next_histogram_mse: f64,
    /// Baseline MSE from literally copying the current board histogram.
    pub literal_current_copy_mse: f64,
    /// `(literal_current_copy_mse - predicted_next_histogram_mse) / literal_current_copy_mse`.
    /// Absent when the exact copy baseline has zero error.
    pub improvement_fraction: Option<f64>,
    pub changed_patch_precision: f64,
    pub changed_patch_recall: f64,
    pub changed_patch_f1: f64,
    /// Predicted-next histogram MSE restricted to exact unchanged patches.
    pub unchanged_patch_mse: f64,
    pub frame_count: usize,
    pub patch_count: usize,
    pub changed_patch_count: usize,
    pub unchanged_patch_count: usize,
    pub prediction_delta_threshold: f64,
    pub target_decoder_mse_ceiling: f64,
    /// True only when the fixed target-latent decoder gate passes.
    pub trusted: bool,
}

impl FixedBoardProbe {
    pub fn fit_spatial(train_latents: &BoardProbeRows, train_targets: &[ArcFrame]) -> Result<Self> {
        Self::fit(train_latents.as_rows(), train_targets)
    }

    /// Fits only the supplied training rows.  Each frame contributes 64 patch targets in row order.
    pub fn fit(train_latents: &[Vec<f32>], train_targets: &[ArcFrame]) -> Result<Self> {
        let targets = histograms_for_frames(train_targets)?;
        validate_fit_latents(train_latents, targets.len())?;
        let input_dim = train_latents[0].len();
        let row_count = train_latents.len();

        let mut input_mean = vec![0.0_f64; input_dim];
        for row in train_latents {
            for (feature, &value) in row.iter().enumerate() {
                input_mean[feature] += value as f64;
            }
        }
        for value in &mut input_mean {
            *value /= row_count as f64;
        }
        let mut input_std = vec![0.0_f64; input_dim];
        for row in train_latents {
            for feature in 0..input_dim {
                let delta = row[feature] as f64 - input_mean[feature];
                input_std[feature] += delta * delta;
            }
        }
        for (feature, value) in input_std.iter_mut().enumerate() {
            *value = (*value / row_count as f64).sqrt();
            ensure!(
                value.is_finite(),
                "latent feature {feature} has non-finite variance"
            );
            // A dead channel is evidence rather than an evaluator failure. Its
            // standardized value remains zero and ridge assigns it no weight.
            *value = value.max(1e-6);
        }

        let mut output_mean = [0.0_f64; PALETTE_SIZE];
        for target in &targets {
            for palette_id in 0..PALETTE_SIZE {
                output_mean[palette_id] += target[palette_id] as f64;
            }
        }
        for value in &mut output_mean {
            *value /= row_count as f64;
        }

        let mut gram = vec![vec![0.0_f64; input_dim]; input_dim];
        let mut rhs = vec![[0.0_f64; PALETTE_SIZE]; input_dim];
        for (row, target) in train_latents.iter().zip(&targets) {
            let standardized: Vec<f64> = (0..input_dim)
                .map(|feature| (row[feature] as f64 - input_mean[feature]) / input_std[feature])
                .collect();
            for left in 0..input_dim {
                for right in 0..input_dim {
                    gram[left][right] += standardized[left] * standardized[right];
                }
                for palette_id in 0..PALETTE_SIZE {
                    rhs[left][palette_id] +=
                        standardized[left] * (target[palette_id] as f64 - output_mean[palette_id]);
                }
            }
        }
        for (feature, row) in gram.iter_mut().enumerate() {
            row[feature] += RIDGE;
        }

        let mut weights = vec![[0.0_f32; PALETTE_SIZE]; input_dim];
        for palette_id in 0..PALETTE_SIZE {
            let solution = solve_gaussian(
                gram.clone(),
                rhs.iter().map(|row| row[palette_id]).collect(),
            )?;
            for feature in 0..input_dim {
                ensure!(
                    solution[feature].is_finite(),
                    "ridge solution is non-finite"
                );
                let weight = solution[feature] as f32;
                ensure!(
                    weight.is_finite(),
                    "ridge weight cannot be represented as f32"
                );
                weights[feature][palette_id] = weight;
            }
        }

        Ok(Self {
            input_dim,
            input_mean: input_mean.into_iter().map(|value| value as f32).collect(),
            input_std: input_std.into_iter().map(|value| value as f32).collect(),
            output_mean: output_mean.map(|value| value as f32),
            weights,
            ridge: RIDGE,
        })
    }

    /// Decodes one palette histogram for every supplied patch latent row.
    pub fn predict_histograms(&self, latents: &[Vec<f32>]) -> Result<Vec<[f32; PALETTE_SIZE]>> {
        validate_probe(self)?;
        validate_score_latents(latents, self.input_dim)?;
        latents
            .iter()
            .map(|row| {
                let mut prediction = self.output_mean;
                for (feature, value) in row.iter().enumerate() {
                    let standardized = (*value as f64 - self.input_mean[feature] as f64)
                        / self.input_std[feature] as f64;
                    for (palette_id, output) in prediction.iter_mut().enumerate() {
                        *output += (standardized * self.weights[feature][palette_id] as f64) as f32;
                    }
                }
                ensure!(
                    prediction.iter().all(|value| value.is_finite()),
                    "non-finite probe prediction"
                );
                Ok(prediction)
            })
            .collect()
    }

    /// Summarizes held-out rows.  `exact_changed_labels` is intentionally supplied separately
    /// from the frames so that change detection can use the exact simulator label.
    pub fn summarize_held_out(
        &self,
        target_latents: &[Vec<f32>],
        predicted_next_latents: &[Vec<f32>],
        current_frames: &[ArcFrame],
        target_frames: &[ArcFrame],
        exact_changed_labels: &[bool],
    ) -> Result<BoardTransitionMetrics> {
        ensure!(
            !target_frames.is_empty(),
            "held-out frames must not be empty"
        );
        ensure!(
            current_frames.len() == target_frames.len(),
            "current and target frame counts differ"
        );
        let target_histograms = histograms_for_frames(target_frames)?;
        let current_histograms = histograms_for_frames(current_frames)?;
        let expected_rows = target_histograms.len();
        ensure!(
            target_latents.len() == expected_rows,
            "target latent row count does not match frames"
        );
        ensure!(
            predicted_next_latents.len() == expected_rows,
            "predicted-next latent row count does not match frames"
        );
        ensure!(
            exact_changed_labels.len() == expected_rows,
            "exact changed-label count does not match patches"
        );

        let decoded_targets = self.predict_histograms(target_latents)?;
        let decoded_next = self.predict_histograms(predicted_next_latents)?;
        let target_latent_decoder_mse = mean_histogram_mse(&decoded_targets, &target_histograms)?;
        let predicted_next_histogram_mse = mean_histogram_mse(&decoded_next, &target_histograms)?;
        let literal_current_copy_mse = mean_histogram_mse(&current_histograms, &target_histograms)?;
        let improvement_fraction = (literal_current_copy_mse > 0.0).then_some(
            (literal_current_copy_mse - predicted_next_histogram_mse) / literal_current_copy_mse,
        );

        let mut true_positive = 0usize;
        let mut false_positive = 0usize;
        let mut false_negative = 0usize;
        let mut unchanged_mse_sum = 0.0;
        let mut unchanged_patch_count = 0usize;
        for index in 0..expected_rows {
            let predicted_changed = histogram_mse(&decoded_next[index], &current_histograms[index])
                > FIXED_PREDICTION_DELTA_THRESHOLD;
            match (predicted_changed, exact_changed_labels[index]) {
                (true, true) => true_positive += 1,
                (true, false) => false_positive += 1,
                (false, true) => false_negative += 1,
                (false, false) => {}
            }
            if !exact_changed_labels[index] {
                unchanged_mse_sum += histogram_mse(&decoded_next[index], &target_histograms[index]);
                unchanged_patch_count += 1;
            }
        }
        let precision = ratio_or_zero(true_positive, true_positive + false_positive);
        let recall = ratio_or_zero(true_positive, true_positive + false_negative);
        let changed_patch_f1 = ratio_or_zero(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        );
        let unchanged_patch_mse = if unchanged_patch_count == 0 {
            0.0
        } else {
            unchanged_mse_sum / unchanged_patch_count as f64
        };

        Ok(BoardTransitionMetrics {
            target_latent_decoder_mse,
            predicted_next_histogram_mse,
            literal_current_copy_mse,
            improvement_fraction,
            changed_patch_precision: precision,
            changed_patch_recall: recall,
            changed_patch_f1,
            unchanged_patch_mse,
            frame_count: target_frames.len(),
            patch_count: expected_rows,
            changed_patch_count: exact_changed_labels
                .iter()
                .filter(|&&changed| changed)
                .count(),
            unchanged_patch_count,
            prediction_delta_threshold: FIXED_PREDICTION_DELTA_THRESHOLD,
            target_decoder_mse_ceiling: FIXED_TARGET_DECODER_MSE_CEILING,
            trusted: target_latent_decoder_mse <= FIXED_TARGET_DECODER_MSE_CEILING,
        })
    }

    pub fn summarize_transitions(
        &self,
        target_latents: &BoardProbeRows,
        predicted_next_latents: &BoardProbeRows,
        transitions: &BoardProbeTransitions,
    ) -> Result<BoardTransitionMetrics> {
        self.summarize_held_out(
            target_latents.as_rows(),
            predicted_next_latents.as_rows(),
            &transitions.current,
            &transitions.target,
            &transitions.exact_changed_patches,
        )
    }
}

/// Converts one fixed board to 64 deterministic row-major 8 by 8 patch histograms.
/// The global `y == 63` status row is excluded, including from the final patch row.
pub fn frame_patch_histograms(frame: &ArcFrame) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    ensure_fixed_frame(frame)?;
    let mut histograms = vec![[0.0_f32; PALETTE_SIZE]; PATCH_COUNT];
    for y in 0..(FRAME_SIDE - 1) {
        for x in 0..FRAME_SIDE {
            let patch = (y / PATCHES_PER_SIDE) * PATCHES_PER_SIDE + x / PATCHES_PER_SIDE;
            let palette_id = frame.pixels[y * FRAME_SIDE + x] as usize;
            ensure!(
                palette_id < PALETTE_SIZE,
                "palette ID {palette_id} is outside 0..15"
            );
            histograms[patch][palette_id] += 1.0;
        }
    }
    Ok(histograms)
}

fn histograms_for_frames(frames: &[ArcFrame]) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    let mut result = Vec::with_capacity(frames.len() * PATCH_COUNT);
    for frame in frames {
        result.extend(frame_patch_histograms(frame)?);
    }
    Ok(result)
}

fn ensure_fixed_frame(frame: &ArcFrame) -> Result<()> {
    ensure!(
        frame.width as usize == FRAME_SIDE && frame.height as usize == FRAME_SIDE,
        "board probe requires a 64x64 frame, got {}x{}",
        frame.width,
        frame.height
    );
    ensure!(
        frame.pixels.len() == FRAME_SIDE * FRAME_SIDE,
        "64x64 frame has {} pixels",
        frame.pixels.len()
    );
    ensure!(
        frame
            .pixels
            .iter()
            .all(|&value| (value as usize) < PALETTE_SIZE),
        "frame contains palette value outside 0..15"
    );
    Ok(())
}

fn validate_fit_latents(latents: &[Vec<f32>], expected_rows: usize) -> Result<()> {
    ensure!(!latents.is_empty(), "training latents must not be empty");
    ensure!(
        latents.len() == expected_rows,
        "training latent row count does not match training frames"
    );
    let input_dim = latents[0].len();
    ensure!(
        input_dim > 0,
        "training latent rows must have at least one feature"
    );
    ensure!(
        latents.len() > input_dim,
        "training has {} rows but needs more than its {} input features",
        latents.len(),
        input_dim
    );
    validate_score_latents(latents, input_dim)
}

fn validate_score_latents(latents: &[Vec<f32>], input_dim: usize) -> Result<()> {
    ensure!(!latents.is_empty(), "latent rows must not be empty");
    for (index, row) in latents.iter().enumerate() {
        ensure!(
            row.len() == input_dim,
            "latent row {index} has {} features, expected {input_dim}",
            row.len()
        );
        ensure!(
            row.iter().all(|value| value.is_finite()),
            "latent row {index} contains a non-finite value"
        );
    }
    Ok(())
}

fn validate_probe(probe: &FixedBoardProbe) -> Result<()> {
    ensure!(
        probe.ridge == RIDGE,
        "board probe ridge must be the fixed value {RIDGE}"
    );
    ensure!(
        probe.input_dim > 0
            && probe.input_mean.len() == probe.input_dim
            && probe.input_std.len() == probe.input_dim
            && probe.weights.len() == probe.input_dim,
        "invalid persisted board probe dimensions"
    );
    ensure!(
        probe.input_mean.iter().all(|value| value.is_finite())
            && probe
                .input_std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && probe.output_mean.iter().all(|value| value.is_finite())
            && probe
                .weights
                .iter()
                .flatten()
                .all(|value| value.is_finite()),
        "persisted board probe contains non-finite values"
    );
    Ok(())
}

fn solve_gaussian(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = matrix.len();
    ensure!(
        n > 0 && rhs.len() == n && matrix.iter().all(|row| row.len() == n),
        "invalid ridge linear system"
    );
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .partial_cmp(&matrix[right][pivot].abs())
                    .unwrap()
            })
            .expect("non-empty pivot range");
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let diagonal = matrix[pivot][pivot];
        if !diagonal.is_finite() || diagonal.abs() <= 1e-12 {
            bail!("ridge linear system is singular or non-finite at pivot {pivot}");
        }
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / diagonal;
            matrix[row][pivot] = 0.0;
            let pivot_tail = matrix[pivot][pivot + 1..].to_vec();
            for (value, pivot_value) in matrix[row][pivot + 1..].iter_mut().zip(pivot_tail) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let residual = rhs[row]
            - ((row + 1)..n)
                .map(|column| matrix[row][column] * solution[column])
                .sum::<f64>();
        solution[row] = residual / matrix[row][row];
    }
    ensure!(
        solution.iter().all(|value| value.is_finite()),
        "non-finite ridge solution"
    );
    Ok(solution)
}

fn histogram_mse(left: &[f32; PALETTE_SIZE], right: &[f32; PALETTE_SIZE]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let difference = a as f64 - b as f64;
            difference * difference
        })
        .sum::<f64>()
        / PALETTE_SIZE as f64
}

fn mean_histogram_mse(
    predictions: &[[f32; PALETTE_SIZE]],
    targets: &[[f32; PALETTE_SIZE]],
) -> Result<f64> {
    ensure!(
        !predictions.is_empty() && predictions.len() == targets.len(),
        "incompatible histogram rows"
    );
    Ok(predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| histogram_mse(prediction, target))
        .sum::<f64>()
        / predictions.len() as f64)
}

fn ratio_or_zero(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(mut pixel: impl FnMut(usize, usize) -> u8) -> ArcFrame {
        let mut pixels = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            for x in 0..64 {
                pixels.push(pixel(x, y));
            }
        }
        ArcFrame::new(64, 64, pixels).unwrap()
    }

    fn latent_rows(frames: &[ArcFrame], scales: &[f32]) -> Vec<Vec<f32>> {
        frames
            .iter()
            .flat_map(|frame| frame_patch_histograms(frame).unwrap())
            .map(|histogram| {
                let a = histogram[1] - histogram[0];
                let b = histogram[2] + histogram[3] * 0.5;
                vec![a * scales[0], b * scales[1]]
            })
            .collect()
    }

    #[test]
    fn patch_histograms_are_row_major_and_skip_global_status_row() {
        let board = frame(|x, y| {
            if y == 63 {
                9
            } else if x < 8 && y < 8 {
                3
            } else {
                1
            }
        });
        let histograms = frame_patch_histograms(&board).unwrap();
        assert_eq!(histograms.len(), 64);
        assert_eq!(histograms[0][3], 64.0);
        assert_eq!(histograms[0][1], 0.0);
        assert_eq!(histograms[63][1], 56.0);
        assert_eq!(histograms[63][9], 0.0);
    }

    #[test]
    fn fit_and_score_are_invariant_to_positive_per_feature_rescaling() {
        let train = vec![
            frame(|x, y| ((x / 8 + y / 8) % 4) as u8),
            frame(|x, y| ((2 * (x / 8) + y / 8 + 1) % 4) as u8),
        ];
        let held_out = vec![frame(|x, y| ((x / 8 + 3 * (y / 8) + 2) % 4) as u8)];
        let probe = FixedBoardProbe::fit(&latent_rows(&train, &[1.0, 1.0]), &train).unwrap();
        let scaled_probe =
            FixedBoardProbe::fit(&latent_rows(&train, &[7.0, 0.125]), &train).unwrap();
        let predictions = probe
            .predict_histograms(&latent_rows(&held_out, &[1.0, 1.0]))
            .unwrap();
        let scaled_predictions = scaled_probe
            .predict_histograms(&latent_rows(&held_out, &[7.0, 0.125]))
            .unwrap();
        for (left, right) in predictions.iter().zip(&scaled_predictions) {
            for palette_id in 0..PALETTE_SIZE {
                assert!((left[palette_id] - right[palette_id]).abs() < 2e-4);
            }
        }
        let current = frame(|_, _| 0);
        let exact_labels = vec![false; PATCH_COUNT];
        let metrics = probe
            .summarize_held_out(
                &latent_rows(&held_out, &[1.0, 1.0]),
                &latent_rows(&held_out, &[1.0, 1.0]),
                std::slice::from_ref(&current),
                &held_out,
                &exact_labels,
            )
            .unwrap();
        let scaled_metrics = scaled_probe
            .summarize_held_out(
                &latent_rows(&held_out, &[7.0, 0.125]),
                &latent_rows(&held_out, &[7.0, 0.125]),
                &[current],
                &held_out,
                &exact_labels,
            )
            .unwrap();
        assert!(
            (metrics.target_latent_decoder_mse - scaled_metrics.target_latent_decoder_mse).abs()
                < 2e-5
        );
        assert!(
            (metrics.predicted_next_histogram_mse - scaled_metrics.predicted_next_histogram_mse)
                .abs()
                < 2e-5
        );
        assert!((metrics.unchanged_patch_mse - scaled_metrics.unchanged_patch_mse).abs() < 2e-5);
        assert_eq!(metrics.trusted, scaled_metrics.trusted);
    }

    #[test]
    fn held_out_summary_uses_exact_labels_and_fixed_gate() {
        let train = vec![
            frame(|x, y| ((x / 8 + y / 8) % 4) as u8),
            frame(|x, y| ((2 * (x / 8) + y / 8 + 1) % 4) as u8),
        ];
        let current = frame(|_, _| 0);
        let target = frame(|x, y| if x < 8 && y < 8 { 1 } else { 0 });
        let probe = FixedBoardProbe::fit(&latent_rows(&train, &[1.0, 1.0]), &train).unwrap();
        let target_latents = latent_rows(std::slice::from_ref(&target), &[1.0, 1.0]);
        let labels = (0..PATCH_COUNT).map(|patch| patch == 0).collect::<Vec<_>>();
        let metrics = probe
            .summarize_held_out(
                &target_latents,
                &target_latents,
                &[current],
                &[target],
                &labels,
            )
            .unwrap();
        assert_eq!(metrics.patch_count, 64);
        assert_eq!(metrics.changed_patch_count, 1);
        assert_eq!(metrics.unchanged_patch_count, 63);
        assert!(metrics.literal_current_copy_mse > 0.0);
        assert!(metrics.target_decoder_mse_ceiling == FIXED_TARGET_DECODER_MSE_CEILING);
        assert_eq!(
            metrics.prediction_delta_threshold,
            FIXED_PREDICTION_DELTA_THRESHOLD
        );
    }

    #[test]
    fn accepts_dead_features_but_rejects_nonfinite_and_undersized_rows() {
        let board = frame(|_, _| 0);
        let rows = vec![vec![0.0, 1.0]; PATCH_COUNT];
        assert!(FixedBoardProbe::fit(&rows, std::slice::from_ref(&board)).is_ok());
        let rows = vec![vec![0.0; PATCH_COUNT]; PATCH_COUNT];
        assert!(FixedBoardProbe::fit(&rows, std::slice::from_ref(&board)).is_err());
        let mut rows = vec![vec![0.0]; PATCH_COUNT];
        rows[0][0] = f32::NAN;
        assert!(FixedBoardProbe::fit(&rows, &[board]).is_err());
    }
}
