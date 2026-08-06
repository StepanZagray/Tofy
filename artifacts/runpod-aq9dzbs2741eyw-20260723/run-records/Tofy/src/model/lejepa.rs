use anyhow::Result;
use candle_core::{Tensor, D};
use rand::{RngExt, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn prediction_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let pred = pred.to_dtype(candle_core::DType::F32)?;
    let target = target.to_dtype(candle_core::DType::F32)?;
    Ok(pred.broadcast_sub(&target)?.sqr()?.mean_all()?)
}

pub fn mean_cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    if a_dims != b_dims {
        anyhow::bail!("cosine tensors must have identical dims: {a_dims:?} vs {b_dims:?}");
    }
    let Some(&dim) = a_dims.last() else {
        anyhow::bail!("cosine tensors must have at least one dimension");
    };
    let rows = a_dims[..a_dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    let a = a.reshape((rows, dim))?.to_dtype(candle_core::DType::F32)?;
    let b = b.reshape((rows, dim))?.to_dtype(candle_core::DType::F32)?;
    let dot = a.broadcast_mul(&b)?.sum(1)?;
    let a_norm = a.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    let b_norm = b.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    dot.broadcast_div(&a_norm.broadcast_mul(&b_norm)?)?
        .clamp(-1.0, 1.0)?
        .mean_all()
        .map_err(Into::into)
}

pub fn tensor_rms(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.mean_all()?.sqrt().map_err(Into::into)
}

fn association_logits(task_slots: &Tensor, doc_slots: &Tensor) -> Result<Tensor> {
    let (batch, slots, dim) = task_slots.dims3()?;
    if doc_slots.dims3()? != (batch, slots, dim) {
        anyhow::bail!("association tensors must have identical [batch, slots, dim] shapes");
    }
    let task = task_slots.reshape((batch, slots * dim))?;
    let docs = doc_slots.reshape((batch, slots * dim))?;
    let task_norm = task
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let doc_norm = docs
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    task.broadcast_div(&task_norm)?
        .matmul(&docs.broadcast_div(&doc_norm)?.t()?)?
        .affine(1.0 / 0.07, 0.0)
        .map_err(Into::into)
}

/// Retrieval accuracy is a probe only; it is deliberately not a loss term.
pub fn association_top1_accuracy(task_slots: &Tensor, doc_slots: &Tensor) -> Result<f32> {
    let predictions = association_logits(task_slots, doc_slots)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?;
    let correct = predictions
        .iter()
        .enumerate()
        .filter(|(index, prediction)| **prediction as usize == *index)
        .count();
    Ok(correct as f32 / predictions.len().max(1) as f32)
}

pub fn flatten_latent_slots(latent_slots: &Tensor) -> Result<Tensor> {
    let (batch, slots, dim) = latent_slots.dims3()?;
    latent_slots
        .reshape((batch * slots, dim))
        .map_err(Into::into)
}

const SIGREG_PROJECTION_SEED: u64 = 0x5147_5253_4947_4552;
static SIGREG_CALL: AtomicU64 = AtomicU64::new(0);

fn standard_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0f32);
    let u2 = rng.random_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Exact LeWorldModel SIGReg discretization.
///
/// Rank-3 input is `[batch, positions, dim]`: every position is tested over
/// the batch, matching LeWM's per-timestep statistic. Rank-2 input is treated
/// as a single position. Projection directions are Gaussian, unit-normalized,
/// and resampled on every call as in the official implementation.
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
    let x = match x.rank() {
        2 => x.unsqueeze(1)?,
        3 => x.clone(),
        rank => anyhow::bail!("SIGReg expects rank 2 or 3 input, got rank {rank}"),
    }
    .to_dtype(work_dtype)?;
    let (batch, positions, dim) = x.dims3()?;
    if batch < 2 {
        anyhow::bail!("SIGReg requires at least two batch samples")
    }
    let num_slices = num_slices.max(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut proj = vec![0f32; dim * num_slices];
    for slice in 0..num_slices {
        let mut norm = 0f32;
        for d in 0..dim {
            let v = standard_normal(&mut rng);
            proj[d * num_slices + slice] = v;
            norm += v * v;
        }
        let norm = norm.sqrt().max(1e-6);
        for d in 0..dim {
            proj[d * num_slices + slice] /= norm;
        }
    }
    let proj = Tensor::from_vec(proj, (dim, num_slices), device)?.to_dtype(work_dtype)?;
    let projected = x
        .reshape((batch * positions, dim))?
        .matmul(&proj)?
        .reshape((batch, positions, num_slices))?
        .unsqueeze(3)?; // [B, P, M, 1]

    let knots = num_points.max(3);
    let dt = 3.0f32 / (knots - 1) as f32;
    let mut knot_values = Vec::with_capacity(knots);
    let mut normal_cf = Vec::with_capacity(knots);
    let mut integration_weights = Vec::with_capacity(knots);
    for i in 0..knots {
        let t = i as f32 * dt;
        let phi = (-0.5 * t * t).exp();
        let trapezoid = if i == 0 || i + 1 == knots {
            dt
        } else {
            2.0 * dt
        };
        knot_values.push(t);
        normal_cf.push(phi);
        integration_weights.push(trapezoid * phi);
    }
    let knots_tensor = Tensor::from_vec(knot_values, (1, 1, 1, knots), device)?;
    let normal_cf = Tensor::from_vec(normal_cf, (knots,), device)?;
    let weights = Tensor::from_vec(integration_weights, (knots,), device)?;
    let values = projected.broadcast_mul(&knots_tensor)?;
    let error = values
        .cos()?
        .mean(0)?
        .broadcast_sub(&normal_cf)?
        .sqr()?
        .broadcast_add(&values.sin()?.mean(0)?.sqr()?)?;
    error
        .broadcast_mul(&weights)?
        .sum(candle_core::D::Minus1)?
        .affine(batch as f64, 0.0)?
        .mean_all()
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn prediction_loss_penalizes_vector_scale_error() -> Result<()> {
        let device = Device::Cpu;
        let target = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device)?;
        let pred = Tensor::from_vec(vec![6.0f32, 8.0], (1, 2), &device)?;
        let loss = prediction_loss(&pred, &target)?.to_vec0::<f32>()?;
        assert!(
            (loss - 12.5).abs() < 1e-5,
            "raw embedding MSE should be 12.5, got {loss}"
        );
        Ok(())
    }

    #[test]
    fn mean_cosine_similarity_uses_last_dimension_for_slot_tensors() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            (2, 2, 2),
            &device,
        )?;
        let b = Tensor::from_vec(
            vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            (2, 2, 2),
            &device,
        )?;

        let value = mean_cosine_similarity(&a, &b)?.to_vec0::<f32>()?;

        assert!((value - 0.25).abs() < 1e-6, "cosine was {value}");
        Ok(())
    }

    #[test]
    fn sigreg_supports_slotwise_batches_and_resampled_directions() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (8, 3, 4), &device)?;
        let first = sigreg_epps_pulley_seeded(&x, 32, 17, 1)?.to_vec0::<f32>()?;
        let second = sigreg_epps_pulley_seeded(&x, 32, 17, 2)?.to_vec0::<f32>()?;
        assert!(first.is_finite() && first >= 0.0);
        assert!(second.is_finite() && second >= 0.0);
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn association_top1_is_perfect_for_distinct_identical_pairs() -> Result<()> {
        let slots = Tensor::from_vec(
            vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            (2, 2, 2),
            &Device::Cpu,
        )?;
        assert_eq!(association_top1_accuracy(&slots, &slots)?, 1.0);
        Ok(())
    }
}
