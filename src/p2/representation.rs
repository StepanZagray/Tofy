//! Named, bounded representation-seam summaries for P2 evaluation.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use rand::{seq::SliceRandom, SeedableRng};
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
    pub dimension: usize,
    pub mean_rms: Option<f64>,
    pub mean_variance: Option<f64>,
    pub effective_rank: Option<f64>,
    pub effective_rank_fraction: Option<f64>,
}

pub type RepresentationSeamMap = BTreeMap<RepresentationSeam, RepresentationSeamMetrics>;

/// Deterministically choose rows after the complete population has been constructed.
/// Sorting preserves source order while keeping the selected set independent of batching.
pub fn capped_row_indices(
    eval_seed: u64,
    seam: RepresentationSeam,
    rows_seen: usize,
    cap: usize,
) -> Vec<u32> {
    if cap == 0 || rows_seen <= cap {
        return (0..rows_seen as u32).collect();
    }
    let seam_tag = seam as u64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        eval_seed
            ^ seam_tag.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (rows_seen as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (cap as u64),
    );
    let mut indices: Vec<u32> = (0..rows_seen as u32).collect();
    indices.partial_shuffle(&mut rng, cap);
    indices.truncate(cap);
    indices.sort_unstable();
    indices
}

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

/// Summarize one complete device-resident seam with exactly one bounded host transfer.
pub fn summarize_seam(
    tensor: &Tensor,
    seam: RepresentationSeam,
    eval_seed: u64,
    cap: usize,
) -> Result<RepresentationSeamMetrics> {
    let rows = seam_rows(tensor, seam)?;
    let rows_seen = rows.dim(0)?;
    let dimension = rows.dim(1)?;
    let indices = capped_row_indices(eval_seed, seam, rows_seen, cap);
    let selected = if indices.len() == rows_seen {
        rows
    } else {
        let count = indices.len();
        let indices = Tensor::from_vec(indices, (count,), rows.device())?;
        rows.index_select(&indices, 0)?
    };
    // This is the only device-to-host transfer for the seam. All reduction math below is F64.
    let host_rows = selected.to_dtype(DType::F32)?.to_vec2::<f32>()?;
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
        rows_seen,
        rows_used: host_rows.len(),
        dimension,
        mean_rms,
        mean_variance,
        effective_rank,
        effective_rank_fraction: effective_rank.map(|rank| rank / dimension as f64),
    })
}

pub fn summarize_seams(
    tensors: &BTreeMap<RepresentationSeam, Tensor>,
    eval_seed: u64,
    cap: usize,
) -> Result<RepresentationSeamMap> {
    tensors
        .iter()
        .map(|(&seam, tensor)| Ok((seam, summarize_seam(tensor, seam, eval_seed, cap)?)))
        .collect()
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
        let full = capped_row_indices(11, RepresentationSeam::PredictionFinalSpatial, 100, 13);
        assert_eq!(
            full,
            capped_row_indices(11, RepresentationSeam::PredictionFinalSpatial, 100, 13)
        );
        let partitioned: Vec<u32> = (0..100)
            .filter(|index| full.binary_search(&(*index as u32)).is_ok())
            .map(|index| index as u32)
            .collect();
        assert_eq!(full, partitioned);
        let values = (0..200).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (100, 2), &Device::Cpu)?;
        let split = Tensor::cat(&[&tensor.narrow(0, 0, 37)?, &tensor.narrow(0, 37, 63)?], 0)?;
        let direct = summarize_seam(&tensor, RepresentationSeam::PredictionFinalPooled, 11, 13)?;
        let batched = summarize_seam(&split, RepresentationSeam::PredictionFinalPooled, 11, 13)?;
        assert_eq!(direct.rows_seen, batched.rows_seen);
        assert_eq!(direct.rows_used, batched.rows_used);
        assert_eq!(direct.mean_rms, batched.mean_rms);
        assert_eq!(direct.mean_variance, batched.mean_variance);
        assert_eq!(direct.effective_rank, batched.effective_rank);
        Ok(())
    }
}
