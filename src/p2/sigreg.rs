//! Exact differentiable SIGReg (Epps–Pulley) matching official LeWorldModel.

use anyhow::{bail, Result};
use candle_core::{Device, Tensor, D};
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};

const SIGREG_PROJECTION_SEED: u64 = 0x5147_5253_4947_4552;
static SIGREG_CALL: AtomicU64 = AtomicU64::new(0);

fn standard_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0f32);
    let u2 = rng.random_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn sigreg_projection(dim: usize, num_slices: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut projection = vec![0f32; dim * num_slices];
    for slice in 0..num_slices {
        let mut norm = 0f32;
        for d in 0..dim {
            let value = standard_normal(&mut rng);
            projection[d * num_slices + slice] = value;
            norm += value * value;
        }
        let norm = norm.sqrt().max(1e-6);
        for d in 0..dim {
            projection[d * num_slices + slice] /= norm;
        }
    }
    Tensor::from_vec(projection, (dim, num_slices), device).map_err(Into::into)
}

fn sigreg_knots(num_points: usize) -> Vec<(f32, f32, f32)> {
    let knots = num_points.max(3);
    let dt = 3.0f32 / (knots - 1) as f32;
    (0..knots)
        .map(|index| {
            let t = index as f32 * dt;
            let normal_cf = (-0.5 * t * t).exp();
            let trapezoid = if index == 0 || index + 1 == knots {
                dt
            } else {
                2.0 * dt
            };
            (t, normal_cf, trapezoid * normal_cf)
        })
        .collect()
}

fn validate_sigreg_args(
    num_slices: usize,
    num_points: usize,
    batch: usize,
    dim: usize,
) -> Result<()> {
    if num_slices == 0 {
        bail!("SIGReg requires num_slices >= 1");
    }
    if num_points < 3 {
        bail!("SIGReg requires num_points >= 3 (got {num_points})");
    }
    if batch < 2 {
        bail!("SIGReg requires at least two batch samples");
    }
    if dim == 0 {
        bail!("SIGReg requires embedding dim >= 1");
    }
    Ok(())
}

/// Exact LeWorldModel SIGReg discretization.
///
/// Accepts embeddings shaped `B×D` or `T×B×D`. Rank-3 inputs apply the
/// statistic independently at each timestep over the batch. Projection
/// directions are Gaussian, unit-normalized, and resampled on every call.
pub fn sigreg_epps_pulley(x: &Tensor, num_slices: usize, num_points: usize) -> Result<Tensor> {
    let call = SIGREG_CALL.fetch_add(1, Ordering::Relaxed);
    sigreg_epps_pulley_seeded(
        x,
        num_slices,
        num_points,
        SIGREG_PROJECTION_SEED.wrapping_add(call),
    )
}

pub fn sigreg_epps_pulley_seeded(
    x: &Tensor,
    num_slices: usize,
    num_points: usize,
    seed: u64,
) -> Result<Tensor> {
    let device = x.device();
    let work_dtype = candle_core::DType::F32;
    // Normalize to T×B×D then B×T×D for the batch-mean Epps–Pulley path.
    let x = match x.rank() {
        2 => x.unsqueeze(0)?,
        3 => x.clone(),
        rank => bail!("SIGReg expects rank 2 (B×D) or 3 (T×B×D), got rank {rank}"),
    }
    .to_dtype(work_dtype)?;
    let (time, batch, dim) = x.dims3()?;
    if time == 0 {
        bail!("SIGReg requires at least one timestep");
    }
    validate_sigreg_args(num_slices, num_points, batch, dim)?;

    let x = x.transpose(0, 1)?; // B×T×D
    let proj = sigreg_projection(dim, num_slices, seed, device)?.to_dtype(work_dtype)?;
    let projected = x
        .reshape((batch * time, dim))?
        .matmul(&proj)?
        .reshape((batch, time, num_slices))?
        .unsqueeze(3)?; // [B, T, M, 1]

    let knots = sigreg_knots(num_points);
    let knot_values = knots.iter().map(|&(t, _, _)| t).collect::<Vec<_>>();
    let normal_cf = knots
        .iter()
        .map(|&(_, normal_cf, _)| normal_cf)
        .collect::<Vec<_>>();
    let integration_weights = knots
        .iter()
        .map(|&(_, _, weight)| weight)
        .collect::<Vec<_>>();
    let knots_len = knots.len();
    let knots_tensor = Tensor::from_vec(knot_values, (1, 1, 1, knots_len), device)?;
    let normal_cf = Tensor::from_vec(normal_cf, (knots_len,), device)?;
    let weights = Tensor::from_vec(integration_weights, (knots_len,), device)?;
    let values = projected.broadcast_mul(&knots_tensor)?;
    let error = values
        .cos()?
        .mean(0)?
        .broadcast_sub(&normal_cf)?
        .sqr()?
        .broadcast_add(&values.sin()?.mean(0)?.sqr()?)?;
    error
        .broadcast_mul(&weights)?
        .sum(D::Minus1)?
        .affine(batch as f64, 0.0)?
        .mean_all()
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn sigreg_finite_for_bxd_and_txbxd() -> Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..32).map(|i| (i as f32 * 0.17).sin()).collect();
        let x2 = Tensor::from_vec(values.clone(), (8, 4), &device)?;
        let loss2 = sigreg_epps_pulley_seeded(&x2, 16, 9, 7)?;
        let v2 = loss2.to_vec0::<f32>()?;
        assert!(v2.is_finite(), "{v2}");

        let x3 = Tensor::from_vec(values, (2, 4, 4), &device)?;
        let loss3 = sigreg_epps_pulley_seeded(&x3, 16, 9, 7)?;
        let v3 = loss3.to_vec0::<f32>()?;
        assert!(v3.is_finite(), "{v3}");
        Ok(())
    }

    #[test]
    fn sigreg_rejects_invalid_args() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
        assert!(sigreg_epps_pulley_seeded(&x, 0, 9, 1).is_err());
        assert!(sigreg_epps_pulley_seeded(&x, 8, 2, 1).is_err());
        let tiny = Tensor::zeros((1, 4), candle_core::DType::F32, &device).unwrap();
        assert!(sigreg_epps_pulley_seeded(&tiny, 8, 9, 1).is_err());
    }

    #[test]
    fn sigreg_backward_finite_grads() -> Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..24).map(|i| (i as f32 * 0.31 - 1.0).tanh()).collect();
        let variable = Var::new(values.as_slice(), &device)?;
        let x = variable.as_tensor().reshape((6, 4))?;
        let loss = sigreg_epps_pulley_seeded(&x, 8, 5, 11)?;
        let grads = loss.backward()?;
        let g = grads
            .get(&variable)
            .expect("SIGReg should produce input gradients");
        let flat = g.flatten_all()?.to_vec1::<f32>()?;
        assert!(!flat.is_empty());
        assert!(flat.iter().all(|v| v.is_finite()));
        assert!(flat.iter().any(|v| *v != 0.0));
        Ok(())
    }
}
