//! Named, bounded representation-seam summaries for P2 evaluation.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default maximum host rows retained for each representation seam.
pub const DEFAULT_REPRESENTATION_ROW_CAP: usize = 8192;

/// Configuration for the VICReg-style latent health penalty.
///
/// The penalty is evaluated in F32. `maximum_rows` bounds the covariance
/// computation; selected row positions are deterministic in the canonical
/// logical order, rather than depending on random sampling or device state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VicRegConfig {
    pub variance_weight: f32,
    pub covariance_weight: f32,
    pub minimum_std: f32,
    pub epsilon: f32,
    pub maximum_rows: usize,
}

impl Default for VicRegConfig {
    fn default() -> Self {
        Self {
            variance_weight: 25.0,
            covariance_weight: 1.0,
            minimum_std: 1.0,
            epsilon: 1e-4,
            maximum_rows: DEFAULT_REPRESENTATION_ROW_CAP,
        }
    }
}

impl VicRegConfig {
    /// Reject settings that would make the regularizer undefined or misleading.
    pub fn validate(&self) -> Result<()> {
        if !self.variance_weight.is_finite() || self.variance_weight < 0.0 {
            bail!("VICReg variance_weight must be finite and >= 0");
        }
        if !self.covariance_weight.is_finite() || self.covariance_weight < 0.0 {
            bail!("VICReg covariance_weight must be finite and >= 0");
        }
        if !self.minimum_std.is_finite() || self.minimum_std <= 0.0 {
            bail!("VICReg minimum_std must be finite and > 0");
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            bail!("VICReg epsilon must be finite and > 0");
        }
        if self.maximum_rows < 2 {
            bail!("VICReg maximum_rows must be >= 2");
        }
        Ok(())
    }
}

/// Differentiable components of a [`vicreg_latent_health`] evaluation.
///
/// `variance` and `covariance` are unweighted scalar terms. `weighted_total`
/// is the scalar to add to the consumer's loss, and `rows` records the exact
/// deterministic sample size used for both terms.
#[derive(Debug, Clone)]
pub struct VicRegLoss {
    pub variance: Tensor,
    pub covariance: Tensor,
    pub weighted_total: Tensor,
    pub rows: usize,
}

/// Flatten rank-2 pooled or rank-4 spatial latents into canonical `rows × C`.
///
/// Spatial rows use `B×H×W×C` order. This establishes a stable logical row
/// population before any cap is applied, independent of physical layout.
fn vicreg_rows(latents: &Tensor) -> Result<Tensor> {
    match latents.rank() {
        2 => Ok(latents.clone()),
        4 => {
            let (batch, channels, height, width) = latents.dims4()?;
            latents
                .permute((0, 2, 3, 1))?
                .contiguous()?
                .reshape((batch * height * width, channels))
                .map_err(Into::into)
        }
        rank => bail!("VICReg latents must be rank 2 (B×C) or rank 4 (B×C×H×W), got {rank}"),
    }
}

/// Return evenly distributed canonical row positions, including both ends.
///
/// This has no RNG or host value dependency: identical logical latent rows
/// receive identical selection regardless of device execution details.
fn vicreg_row_indices(rows: usize, maximum_rows: usize) -> Result<Vec<u32>> {
    if rows <= maximum_rows {
        return (0..rows)
            .map(|row| u32::try_from(row).map_err(|_| anyhow::anyhow!("VICReg row index overflow")))
            .collect();
    }
    (0..maximum_rows)
        .map(|selected| {
            u32::try_from(selected * (rows - 1) / (maximum_rows - 1))
                .map_err(|_| anyhow::anyhow!("VICReg row index overflow"))
        })
        .collect()
}

/// Compute VICReg-style variance and covariance health penalties for latents.
///
/// The input remains on its current autograd path: F32 conversion, reshape,
/// row selection, centering, and both penalty terms are Candle operations with
/// live gradients. The covariance term is the mean squared off-diagonal entry
/// of the unbiased feature covariance matrix (zero for a one-channel latent).
pub fn vicreg_latent_health(latents: &Tensor, config: VicRegConfig) -> Result<VicRegLoss> {
    config.validate()?;
    let rows = vicreg_rows(latents)?.to_dtype(DType::F32)?;
    let row_count = rows.dim(0)?;
    let channels = rows.dim(1)?;
    if row_count < 2 {
        bail!("VICReg requires at least two latent rows, got {row_count}");
    }
    if channels == 0 {
        bail!("VICReg requires at least one latent channel");
    }

    let selected_indices = vicreg_row_indices(row_count, config.maximum_rows)?;
    let selected_rows = selected_indices.len();
    let indices = Tensor::from_vec(selected_indices, (selected_rows,), rows.device())?;
    let rows = rows.index_select(&indices, 0)?;

    let std = rows
        .var(0)?
        .affine(1.0, f64::from(config.epsilon))?
        .sqrt()?;
    let variance = std
        .affine(-1.0, f64::from(config.minimum_std))?
        .relu()?
        .mean_all()?;

    let centered = rows.broadcast_sub(&rows.mean_keepdim(0)?)?;
    let covariance = centered
        .transpose(0, 1)?
        .matmul(&centered)?
        .affine(1.0 / (selected_rows - 1) as f64, 0.0)?;
    let covariance = if channels == 1 {
        covariance.zeros_like()?.sum_all()?
    } else {
        let off_diagonal = Tensor::ones((channels, channels), DType::F32, rows.device())?
            .sub(&Tensor::eye(channels, DType::F32, rows.device())?)?;
        covariance
            .sqr()?
            .mul(&off_diagonal)?
            .sum_all()?
            .affine(1.0 / (channels * (channels - 1)) as f64, 0.0)?
    };
    let weighted_total = variance
        .affine(f64::from(config.variance_weight), 0.0)?
        .add(&covariance.affine(f64::from(config.covariance_weight), 0.0)?)?;

    Ok(VicRegLoss {
        variance,
        covariance,
        weighted_total,
        rows: selected_rows,
    })
}

/// VICReg variance with a scale-normalized off-diagonal correlation penalty.
///
/// This variant is intended for already semantically selected displacement
/// rows. Per-feature standard deviations stay on the autograd path: the
/// variance term observes feature spread after scalar population normalization,
/// while correlation has no radial scale gradient.
pub fn vicreg_displacement_health(latents: &Tensor, config: VicRegConfig) -> Result<VicRegLoss> {
    let base = vicreg_latent_health(latents, config)?;
    let rows = vicreg_rows(latents)?.to_dtype(DType::F32)?;
    let row_count = rows.dim(0)?;
    let channels = rows.dim(1)?;
    let selected_indices = vicreg_row_indices(row_count, config.maximum_rows)?;
    let selected_rows = selected_indices.len();
    let indices = Tensor::from_vec(selected_indices, (selected_rows,), rows.device())?;
    let rows = rows.index_select(&indices, 0)?;
    let centered = rows.broadcast_sub(&rows.mean_keepdim(0)?)?;
    let std = centered
        .var(0)?
        .affine(1.0, f64::from(config.epsilon))?
        .sqrt()?
        .clamp(f64::from(config.epsilon).sqrt(), f64::INFINITY)?;
    let standardized = centered.broadcast_div(&std)?;
    let correlation = standardized
        .transpose(0, 1)?
        .matmul(&standardized)?
        .affine(1.0 / (selected_rows - 1) as f64, 0.0)?;
    let covariance = if channels == 1 {
        correlation.zeros_like()?.sum_all()?
    } else {
        let off_diagonal = Tensor::ones((channels, channels), DType::F32, rows.device())?
            .sub(&Tensor::eye(channels, DType::F32, rows.device())?)?;
        correlation
            .sqr()?
            .mul(&off_diagonal)?
            .sum_all()?
            .affine(1.0 / (channels * (channels - 1)) as f64, 0.0)?
    };
    let weighted_total = base
        .variance
        .affine(f64::from(config.variance_weight), 0.0)?
        .add(&covariance.affine(f64::from(config.covariance_weight), 0.0)?)?;
    Ok(VicRegLoss {
        variance: base.variance,
        covariance,
        weighted_total,
        rows: selected_rows,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationSeam {
    EncoderPreRmsPooled,
    EncoderPostRmsPooled,
    EncoderPreRmsSpatial,
    EncoderPostRmsSpatial,
    ActionConditionedInputSpatial,
    RecursionOuterOneSpatial,
    PredictionFinalPooled,
    PredictionFinalSpatial,
    TargetPostRmsPooled,
    TargetPostRmsSpatial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationSeamMetrics {
    pub rows_seen: usize,
    pub rows_used: usize,
    /// Rows containing a NaN or infinity. Summary statistics are unavailable when nonzero.
    pub non_finite_rows: usize,
    pub dimension: usize,
    pub mean_rms: Option<f64>,
    pub mean_variance: Option<f64>,
    pub effective_rank: Option<f64>,
    pub effective_rank_fraction: Option<f64>,
}

pub type RepresentationSeamMap = BTreeMap<RepresentationSeam, RepresentationSeamMetrics>;

/// Flatten a named seam into `rows × feature_dim` without transferring it to the host.
pub fn seam_rows(tensor: &Tensor, seam: RepresentationSeam) -> Result<Tensor> {
    match seam {
        RepresentationSeam::EncoderPreRmsPooled
        | RepresentationSeam::EncoderPostRmsPooled
        | RepresentationSeam::PredictionFinalPooled
        | RepresentationSeam::TargetPostRmsPooled => {
            if tensor.rank() != 2 {
                bail!(
                    "pooled representation seam must be rank 2, got {}",
                    tensor.rank()
                );
            }
            Ok(tensor.clone())
        }
        RepresentationSeam::EncoderPreRmsSpatial
        | RepresentationSeam::EncoderPostRmsSpatial
        | RepresentationSeam::ActionConditionedInputSpatial
        | RepresentationSeam::RecursionOuterOneSpatial
        | RepresentationSeam::PredictionFinalSpatial
        | RepresentationSeam::TargetPostRmsSpatial => {
            let (batch, channels, height, width) = tensor.dims4()?;
            tensor
                .permute((0, 2, 3, 1))?
                .contiguous()?
                .reshape((batch * height * width, channels))
                .map_err(Into::into)
        }
    }
}

fn variance_and_rank(rows: &[Vec<f32>]) -> (Option<f64>, Option<f64>) {
    if rows.len() < 2 || rows.first().is_none_or(Vec::is_empty) {
        return (None, None);
    }
    let dimension = rows[0].len();
    if rows.iter().any(|row| row.len() != dimension) {
        return (None, None);
    }
    let mut means = vec![0.0f64; dimension];
    for row in rows {
        for (column, value) in row.iter().enumerate() {
            means[column] += f64::from(*value);
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    let denom = (rows.len() - 1) as f64;
    let mut covariance = vec![0.0f64; dimension * dimension];
    for row in rows {
        for i in 0..dimension {
            let left = f64::from(row[i]) - means[i];
            for j in 0..dimension {
                covariance[i * dimension + j] += left * (f64::from(row[j]) - means[j]) / denom;
            }
        }
    }
    let trace: f64 = (0..dimension).map(|i| covariance[i * dimension + i]).sum();
    let trace_squared: f64 = covariance.iter().map(|value| value * value).sum();
    (
        Some(trace / dimension as f64),
        (trace_squared > f64::EPSILON).then_some(trace * trace / trace_squared),
    )
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct RowRank {
    priority: u64,
    row_id: u64,
}

fn mix(value: u64) -> u64 {
    let value = value ^ (value >> 30);
    let value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let value = value ^ (value >> 27);
    value.wrapping_mul(0x94D0_49BB_1331_11EB) ^ (value >> 31)
}

fn row_rank(eval_seed: u64, sampling_key: u64, row_id: u64) -> RowRank {
    RowRank {
        priority: mix(eval_seed ^ sampling_key.rotate_left(17) ^ row_id),
        row_id,
    }
}

/// A bounded device-resident sample of one representation population.
///
/// Ranking rows by their stable identifiers makes selection independent of physical batches.
/// The sole host transfer happens in `summarize` after all batches have been collected.
pub struct RepresentationRowCollector {
    sampling_key: u64,
    eval_seed: u64,
    cap: usize,
    rows_seen: usize,
    dimension: Option<usize>,
    finite_rows: Option<Tensor>,
    selected: Option<Tensor>,
    selected_ranks: Vec<RowRank>,
}

impl RepresentationRowCollector {
    pub fn new(eval_seed: u64, sampling_key: u64, cap: usize) -> Self {
        Self {
            sampling_key,
            eval_seed,
            cap,
            rows_seen: 0,
            dimension: None,
            finite_rows: None,
            selected: None,
            selected_ranks: Vec::new(),
        }
    }

    /// Add a `rows × dimensions` tensor and its stable row identifiers.
    pub fn collect_rows(&mut self, rows: &Tensor, row_ids: Vec<u64>) -> Result<()> {
        if self.cap == 0 {
            bail!("representation row cap must be > 0");
        }
        if rows.rank() != 2 {
            bail!("representation rows must be rank 2, got {}", rows.rank());
        }
        let row_count = rows.dim(0)?;
        let dimension = rows.dim(1)?;
        if row_ids.len() != row_count {
            bail!(
                "representation row ids {} do not match row count {row_count}",
                row_ids.len()
            );
        }
        if let Some(previous) = self.dimension {
            if previous != dimension {
                bail!("representation seam dimension changed from {previous} to {dimension}");
            }
        } else {
            self.dimension = Some(dimension);
        }

        // Keep the finite-row count on device. It is appended to the final row transfer,
        // which lets non-finite input fail closed without batch-wise host copies.
        let finite = rows
            .eq(rows)?
            .to_dtype(DType::F32)?
            .mul(&rows.abs()?.le(f32::MAX)?.to_dtype(DType::F32)?)?
            .sum(D::Minus1)?
            .eq(dimension as f32)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        self.finite_rows = Some(match self.finite_rows.take() {
            Some(previous) => previous.add(&finite)?,
            None => finite,
        });

        let selected_len = self.selected_ranks.len();
        let mut ranked_positions: Vec<(RowRank, usize)> = self
            .selected_ranks
            .iter()
            .copied()
            .enumerate()
            .map(|(position, rank)| (rank, position))
            .chain(row_ids.into_iter().enumerate().map(|(position, row_id)| {
                (
                    row_rank(self.eval_seed, self.sampling_key, row_id),
                    selected_len + position,
                )
            }))
            .collect();
        ranked_positions.sort_unstable_by_key(|(rank, _)| *rank);
        ranked_positions.truncate(self.cap);

        let source = match self.selected.as_ref() {
            Some(selected) => Tensor::cat(&[selected, rows], 0)?,
            None => rows.clone(),
        };
        let positions: Vec<u32> = ranked_positions
            .iter()
            .map(|(_, position)| {
                u32::try_from(*position)
                    .map_err(|_| anyhow::anyhow!("representation row index overflow"))
            })
            .collect::<Result<_>>()?;
        let indices = Tensor::from_vec(positions, (ranked_positions.len(),), rows.device())?;
        self.selected = Some(source.index_select(&indices, 0)?);
        self.selected_ranks = ranked_positions.into_iter().map(|(rank, _)| rank).collect();
        self.rows_seen += row_count;
        Ok(())
    }

    /// Finish the sample with one device-to-host transfer for its retained rows and finite count.
    pub fn summarize(self) -> Result<RepresentationSeamMetrics> {
        let dimension = self.dimension.unwrap_or(0);
        let selected = self
            .selected
            .ok_or_else(|| anyhow::anyhow!("representation collector received no rows"))?;
        let finite_rows = self
            .finite_rows
            .ok_or_else(|| anyhow::anyhow!("representation collector has no finite-row count"))?;
        let payload = Tensor::cat(
            &[
                &selected.to_dtype(DType::F32)?.flatten_all()?,
                &finite_rows.to_dtype(DType::F32)?.reshape((1,))?,
            ],
            0,
        )?
        .to_vec1::<f32>()?;
        let finite_rows = payload
            .last()
            .copied()
            .unwrap_or_default()
            .round()
            .clamp(0.0, self.rows_seen as f32) as usize;
        let non_finite_rows = self.rows_seen.saturating_sub(finite_rows);
        let row_values = &payload[..payload.len().saturating_sub(1)];
        let host_rows: Vec<Vec<f32>> = row_values
            .chunks(dimension)
            .map(ToOwned::to_owned)
            .collect();
        if non_finite_rows > 0 {
            return Ok(RepresentationSeamMetrics {
                rows_seen: self.rows_seen,
                rows_used: host_rows.len(),
                non_finite_rows,
                dimension,
                mean_rms: None,
                mean_variance: None,
                effective_rank: None,
                effective_rank_fraction: None,
            });
        }
        let mean_rms = (!host_rows.is_empty() && dimension > 0).then(|| {
            host_rows
                .iter()
                .map(|row| {
                    (row.iter()
                        .map(|value| f64::from(*value).powi(2))
                        .sum::<f64>()
                        / dimension as f64)
                        .sqrt()
                })
                .sum::<f64>()
                / host_rows.len() as f64
        });
        let (mean_variance, effective_rank) = variance_and_rank(&host_rows);
        Ok(RepresentationSeamMetrics {
            rows_seen: self.rows_seen,
            rows_used: host_rows.len(),
            non_finite_rows,
            dimension,
            mean_rms,
            mean_variance,
            effective_rank,
            effective_rank_fraction: effective_rank.map(|rank| rank / dimension as f64),
        })
    }
}

/// Bounded collectors for every named seam in an evaluation split.
pub struct RepresentationSeamCollector {
    eval_seed: u64,
    cap: usize,
    seams: BTreeMap<RepresentationSeam, RepresentationRowCollector>,
}

impl RepresentationSeamCollector {
    pub fn new(eval_seed: u64, cap: usize) -> Self {
        Self {
            eval_seed,
            cap,
            seams: BTreeMap::new(),
        }
    }

    pub fn collect_batch(
        &mut self,
        tensors: &BTreeMap<RepresentationSeam, Tensor>,
        sample_start: usize,
        sample_count: usize,
    ) -> Result<()> {
        for (&seam, tensor) in tensors {
            let rows = seam_rows(tensor, seam)?;
            let row_count = rows.dim(0)?;
            if row_count % sample_count != 0 {
                bail!("representation seam rows are not divisible by the batch size");
            }
            let rows_per_sample = row_count / sample_count;
            let row_ids = (sample_start..sample_start + sample_count).flat_map(|sample| {
                (0..rows_per_sample)
                    .map(move |row| sample as u64 * rows_per_sample as u64 + row as u64)
            });
            self.seams
                .entry(seam)
                .or_insert_with(|| {
                    RepresentationRowCollector::new(self.eval_seed, seam as u64, self.cap)
                })
                .collect_rows(&rows, row_ids.collect())?;
        }
        Ok(())
    }

    pub fn summarize(self) -> Result<RepresentationSeamMap> {
        self.seams
            .into_iter()
            .map(|(seam, collector)| Ok((seam, collector.summarize()?)))
            .collect()
    }
}

/// Summarize one complete tensor. Used by focused unit tests; split evaluation uses the collector.
pub fn summarize_seam(
    tensor: &Tensor,
    seam: RepresentationSeam,
    eval_seed: u64,
    cap: usize,
) -> Result<RepresentationSeamMetrics> {
    let rows = seam_rows(tensor, seam)?;
    let row_count = rows.dim(0)?;
    let mut collector = RepresentationRowCollector::new(eval_seed, seam as u64, cap);
    collector.collect_rows(&rows, (0..row_count).map(|row| row as u64).collect())?;
    collector.summarize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn vicreg_rejects_invalid_config_and_too_few_rows() -> Result<()> {
        let device = Device::Cpu;
        let one_row = Tensor::zeros((1, 2), DType::F32, &device)?;
        assert!(vicreg_latent_health(&one_row, VicRegConfig::default()).is_err());
        assert!(VicRegConfig {
            maximum_rows: 1,
            ..VicRegConfig::default()
        }
        .validate()
        .is_err());
        assert!(VicRegConfig {
            epsilon: 0.0,
            ..VicRegConfig::default()
        }
        .validate()
        .is_err());
        Ok(())
    }

    #[test]
    fn vicreg_orders_healthy_latents_below_collapsed_latents() -> Result<()> {
        let device = Device::Cpu;
        let config = VicRegConfig {
            variance_weight: 1.0,
            covariance_weight: 1.0,
            minimum_std: 0.9,
            epsilon: 1e-4,
            maximum_rows: 16,
        };
        let healthy = Tensor::from_vec(vec![1f32, 0., -1., 0., 0., 1., 0., -1.], (4, 2), &device)?;
        let collapsed = Tensor::zeros((4, 2), DType::F32, &device)?;
        let healthy = vicreg_latent_health(&healthy, config)?;
        let collapsed = vicreg_latent_health(&collapsed, config)?;
        assert_eq!(healthy.rows, 4);
        assert!(
            healthy.weighted_total.to_vec0::<f32>()? < collapsed.weighted_total.to_vec0::<f32>()?
        );
        assert!(healthy.covariance.to_vec0::<f32>()? < 1e-6);
        Ok(())
    }

    #[test]
    fn vicreg_spatial_rows_are_canonical_and_capped_deterministically() -> Result<()> {
        let device = Device::Cpu;
        let latents = Tensor::from_vec((0..16).map(|v| v as f32).collect(), (2, 2, 2, 2), &device)?;
        let loss = vicreg_latent_health(
            &latents,
            VicRegConfig {
                maximum_rows: 3,
                ..VicRegConfig::default()
            },
        )?;
        assert_eq!(loss.rows, 3);
        assert!(loss.weighted_total.to_vec0::<f32>()?.is_finite());
        assert_eq!(vicreg_row_indices(8, 3)?, vec![0, 3, 7]);
        Ok(())
    }

    #[test]
    fn vicreg_one_channel_covariance_is_a_scalar_zero() -> Result<()> {
        let device = Device::Cpu;
        let latents = Tensor::from_vec(vec![1f32, -1., 0.5, -0.5], (4, 1), &device)?;
        let loss = vicreg_latent_health(&latents, VicRegConfig::default())?;
        assert_eq!(loss.covariance.to_vec0::<f32>()?, 0.0);
        assert!(loss.weighted_total.to_vec0::<f32>()?.is_finite());
        Ok(())
    }

    #[test]
    fn vicreg_backward_produces_finite_input_gradients() -> Result<()> {
        let device = Device::Cpu;
        let variable = Var::new(&[0f32, 0., 1., 0., -1., 0.5, 0.4, -0.9], &device)?;
        let latents = variable.as_tensor().reshape((4, 2))?;
        let loss = vicreg_latent_health(
            &latents,
            VicRegConfig {
                variance_weight: 1.0,
                covariance_weight: 1.0,
                minimum_std: 1.2,
                epsilon: 1e-4,
                maximum_rows: 4,
            },
        )?;
        let gradients = loss.weighted_total.backward()?;
        let gradient = gradients
            .get(&variable)
            .expect("VICReg should retain input gradients")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| *value != 0.0));
        Ok(())
    }

    #[test]
    fn displacement_correlation_is_invariant_to_feature_scale() -> Result<()> {
        let device = Device::Cpu;
        let values = Tensor::from_vec(
            vec![1f32, 2., -1., -2., 0.5, -0.25, -0.5, 0.25],
            (4, 2),
            &device,
        )?;
        let scaled = values.broadcast_mul(&Tensor::new(&[3f32, 0.2], &device)?)?;
        let config = VicRegConfig {
            variance_weight: 0.0,
            covariance_weight: 1.0,
            minimum_std: 0.1,
            epsilon: 1e-8,
            maximum_rows: 4,
        };
        let first = vicreg_displacement_health(&values, config)?
            .covariance
            .to_scalar::<f32>()?;
        let second = vicreg_displacement_health(&scaled, config)?
            .covariance
            .to_scalar::<f32>()?;
        assert!((first - second).abs() < 1e-4, "{first} != {second}");
        Ok(())
    }

    #[test]
    fn scalar_normalized_displacement_health_has_no_radial_scale_gradient() -> Result<()> {
        let device = Device::Cpu;
        let variable = Var::new(&[1f32, 2., -1., -2., 0.5, -0.25, -0.5, 0.25], &device)?;
        let values = variable.as_tensor().reshape((4, 2))?;
        let rms = values.sqr()?.mean_all()?.sqrt()?;
        let normalized = values.broadcast_div(&rms)?;
        let loss = vicreg_displacement_health(
            &normalized,
            VicRegConfig {
                variance_weight: 1.0,
                covariance_weight: 1.0,
                minimum_std: 1.0,
                epsilon: 1e-8,
                maximum_rows: 4,
            },
        )?;
        let gradients = loss.weighted_total.backward()?;
        let gradient = gradients
            .get(&variable)
            .expect("normalized health should retain directional gradients")
            .reshape((4, 2))?;
        let radial = gradient.mul(&values)?.sum_all()?.to_scalar::<f32>()?;
        assert!(radial.abs() < 1e-5, "radial gradient was {radial}");
        Ok(())
    }

    #[test]
    fn constant_rows_fail_variance_and_rank_gates() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1f32; 12], (3, 4), &Device::Cpu)?;
        let summary = summarize_seam(&tensor, RepresentationSeam::EncoderPostRmsPooled, 7, 32)?;
        assert_eq!(summary.mean_variance, Some(0.0));
        assert_eq!(summary.effective_rank, None);
        assert_eq!(summary.effective_rank_fraction, None);
        Ok(())
    }

    #[test]
    fn diverse_rows_have_high_rank_fraction() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![
                1f32, 0., 0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0.,
                0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., -1.,
            ],
            (8, 4),
            &Device::Cpu,
        )?;
        let summary = summarize_seam(&tensor, RepresentationSeam::EncoderPostRmsPooled, 7, 32)?;
        assert!(summary
            .effective_rank_fraction
            .is_some_and(|value| value > 0.9));
        Ok(())
    }

    #[test]
    fn pooled_and_spatial_accounting_is_exact() -> Result<()> {
        let pooled = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let spatial = Tensor::zeros((2, 3, 4, 5), DType::F32, &Device::Cpu)?;
        let pooled = summarize_seam(&pooled, RepresentationSeam::EncoderPostRmsPooled, 0, 100)?;
        let spatial = summarize_seam(&spatial, RepresentationSeam::EncoderPostRmsSpatial, 0, 100)?;
        assert_eq!((pooled.rows_seen, pooled.dimension), (2, 3));
        assert_eq!((spatial.rows_seen, spatial.dimension), (40, 3));
        Ok(())
    }

    #[test]
    fn row_selection_is_deterministic_and_partition_invariant() -> Result<()> {
        let values = (0..200).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (100, 2), &Device::Cpu)?;
        let mut direct = RepresentationRowCollector::new(11, 7, 13);
        direct.collect_rows(&tensor, (0..100).map(|row| row as u64).collect())?;
        let mut batched = RepresentationRowCollector::new(11, 7, 13);
        batched.collect_rows(
            &tensor.narrow(0, 0, 37)?,
            (0..37).map(|row| row as u64).collect(),
        )?;
        batched.collect_rows(
            &tensor.narrow(0, 37, 63)?,
            (37..100).map(|row| row as u64).collect(),
        )?;
        let direct = direct.summarize()?;
        let batched = batched.summarize()?;
        assert_eq!(direct.rows_seen, batched.rows_seen);
        assert_eq!(direct.rows_used, batched.rows_used);
        assert_eq!(direct.mean_rms, batched.mean_rms);
        assert_eq!(direct.mean_variance, batched.mean_variance);
        assert_eq!(direct.effective_rank, batched.effective_rank);
        Ok(())
    }

    #[test]
    fn non_finite_rows_report_unavailable_metrics() -> Result<()> {
        let tensor = Tensor::from_vec(
            vec![1f32, 2., f32::NAN, 4., 5., f32::INFINITY],
            (3, 2),
            &Device::Cpu,
        )?;
        let summary = summarize_seam(&tensor, RepresentationSeam::EncoderPostRmsPooled, 7, 32)?;
        assert_eq!(summary.non_finite_rows, 2);
        assert_eq!(summary.mean_rms, None);
        assert_eq!(summary.mean_variance, None);
        assert_eq!(summary.effective_rank, None);
        assert_eq!(summary.effective_rank_fraction, None);
        Ok(())
    }
}
