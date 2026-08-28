//! Differentiable SIGReg (Epps–Pulley) with reference trapezoidal integration.

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
                0.5 * dt
            } else {
                dt
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

/// LeWorldModel SIGReg discretization using the true trapezoid rule.
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
    sigreg_epps_pulley_seeded_impl(x, num_slices, num_points, seed, false)
}

fn sigreg_epps_pulley_seeded_impl(
    x: &Tensor,
    num_slices: usize,
    num_points: usize,
    seed: u64,
    include_zero_knot: bool,
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

    // At t=0 the empirical characteristic function is exactly (1, 0), as is
    // the standard-normal target, so both the loss and its derivative vanish.
    // Keep the reference path for parity tests, but do not allocate or evaluate
    // that B×T×M slice during training.
    let knots = sigreg_knots(num_points)
        .into_iter()
        .skip(if include_zero_knot { 0 } else { 1 })
        .collect::<Vec<_>>();
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

/// QQWorld rank-matched Gaussian regularization (Eq. 6 in arXiv:2607.28415).
///
/// Accepts `B×D` or `T×B×D`. Rank-3 inputs form an independent ranking
/// population at every temporal position. The loss sums over batch ranks and
/// averages over time and random projection directions, matching the paper's
/// per-slice definition without a cross-batch queue.
pub fn sigreg_quantile_seeded(x: &Tensor, num_slices: usize, seed: u64) -> Result<Tensor> {
    let device = x.device();
    let x = match x.rank() {
        2 => x.unsqueeze(0)?,
        3 => x.clone(),
        rank => bail!("QQ regularization expects B×D or T×B×D, got rank {rank}"),
    }
    .to_dtype(candle_core::DType::F32)?;
    let (time, batch, dim) = x.dims3()?;
    if time == 0 {
        bail!("QQ regularization requires at least one timestep");
    }
    validate_sigreg_args(num_slices, 3, batch, dim)?;
    let projection = sigreg_projection(dim, num_slices, seed, device)?;
    let projected = x
        .reshape((time * batch, dim))?
        .matmul(&projection)?
        .reshape((time, batch, num_slices))?
        .permute((0, 2, 1))?
        .contiguous()?; // T×S×B: rank only across the physical population.
    let (ordered, _) = projected.sort_last_dim(true)?;
    let quantiles = (1..=batch)
        .map(|rank| inverse_standard_normal((rank as f64 - 0.5) / batch as f64) as f32)
        .collect::<Vec<_>>();
    let target = Tensor::from_vec(quantiles, (1, 1, batch), device)?;
    ordered
        .broadcast_sub(&target)?
        .sqr()?
        .sum(D::Minus1)?
        .mean_all()
        .map_err(Into::into)
}

// Peter J. Acklam's rational approximation. The target is constant and is
// computed on the host, so this adds no approximation to model gradients.
fn inverse_standard_normal(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.02425;
    const HIGH: f64 = 1.0 - LOW;
    if p < LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn sigreg_knots_use_true_trapezoid_weights() {
        let knots = sigreg_knots(4);
        let normalized_weights = knots
            .iter()
            .map(|(_, normal_cf, weighted)| weighted / normal_cf)
            .collect::<Vec<_>>();
        for (actual, expected) in normalized_weights.iter().zip([0.5, 1.0, 1.0, 0.5]) {
            assert!((actual - expected).abs() <= f32::EPSILON);
        }
    }

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

    #[test]
    fn omitting_zero_knot_preserves_loss_and_gradients() -> Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..48).map(|i| (i as f32 * 0.19 - 2.0).sin()).collect();
        let optimized_var = Var::new(values.as_slice(), &device)?;
        let reference_var = Var::new(values.as_slice(), &device)?;
        let optimized = sigreg_epps_pulley_seeded_impl(
            &optimized_var.as_tensor().reshape((8, 6))?,
            12,
            17,
            29,
            false,
        )?;
        let reference = sigreg_epps_pulley_seeded_impl(
            &reference_var.as_tensor().reshape((8, 6))?,
            12,
            17,
            29,
            true,
        )?;
        let optimized_value = optimized.to_scalar::<f32>()?;
        let reference_value = reference.to_scalar::<f32>()?;
        assert!((optimized_value - reference_value).abs() <= 1e-5);

        let optimized_grads = optimized.backward()?;
        let reference_grads = reference.backward()?;
        let optimized_grad = optimized_grads
            .get(&optimized_var)
            .expect("optimized SIGReg gradient")
            .to_vec1::<f32>()?;
        let reference_grad = reference_grads
            .get(&reference_var)
            .expect("reference SIGReg gradient")
            .to_vec1::<f32>()?;
        assert_eq!(optimized_grad.len(), reference_grad.len());
        assert!(optimized_grad
            .iter()
            .zip(reference_grad.iter())
            .all(|(left, right)| (left - right).abs() <= 1e-5));
        Ok(())
    }

    #[test]
    fn qq_matches_ranked_gaussian_quantiles_and_backpropagates() -> Result<()> {
        let device = Device::Cpu;
        let quantiles = (1..=8)
            .map(|rank| inverse_standard_normal((rank as f64 - 0.5) / 8.0) as f32)
            .collect::<Vec<_>>();
        let variable = Var::new(quantiles.as_slice(), &device)?;
        let loss = sigreg_quantile_seeded(&variable.reshape((8, 1))?, 1, 17)?;
        assert!(loss.to_scalar::<f32>()? < 1e-10);
        let displaced = variable.affine(2.0, 0.5)?.reshape((8, 1))?;
        let displaced_loss = sigreg_quantile_seeded(&displaced, 1, 17)?;
        assert!(displaced_loss.to_scalar::<f32>()? > 0.1);
        let gradients = displaced_loss.backward()?;
        let gradient = gradients.get(&variable).expect("QQ must backpropagate");
        assert!(gradient
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }
}
