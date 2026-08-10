//! Named, bounded representation-seam summaries for P2 evaluation.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default maximum host rows retained for each representation seam.
pub const DEFAULT_REPRESENTATION_ROW_CAP: usize = 8192;

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
    use candle_core::Device;

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
