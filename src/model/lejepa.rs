use anyhow::Result;
use candle_core::Tensor;
use rand::{RngExt, SeedableRng};

fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let Some(&dim) = dims.last() else {
        anyhow::bail!("normalization tensor must have at least one dimension");
    };
    let rows = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    let x = x.reshape((rows, dim))?;
    let norm = x.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    x.broadcast_div(&norm.unsqueeze(1)?)
        .and_then(|normalized| normalized.reshape(dims))
        .map_err(Into::into)
}

pub fn prediction_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let pred_norm = l2_normalize_last_dim(pred)?;
    let target_norm = l2_normalize_last_dim(target)?;
    Ok(pred_norm
        .broadcast_sub(&target_norm)?
        .sqr()?
        .mean_all()?)
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
    let a = a.reshape((rows, dim))?;
    let b = b.reshape((rows, dim))?;
    let dot = a.broadcast_mul(&b)?.sum(1)?;
    let a_norm = a.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    let b_norm = b.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    dot.broadcast_div(&a_norm.broadcast_mul(&b_norm)?)?
        .mean_all()
        .map_err(Into::into)
}

pub fn tensor_rms(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.mean_all()?.sqrt().map_err(Into::into)
}

pub fn flatten_latent_slots(latent_slots: &Tensor) -> Result<Tensor> {
    let (batch, slots, dim) = latent_slots.dims3()?;
    latent_slots
        .reshape((batch * slots, dim))
        .map_err(Into::into)
}

/// Seed for the SIGReg projection directions. Fixed so the regularization
/// target is stable across forward passes; resampling directions every call
/// makes the loss a moving target with high-variance gradients.
const SIGREG_PROJECTION_SEED: u64 = 0x5147_5253_4947_4552;

/// Lightweight in-repo SIGReg approximation using fixed random 1D projections
/// and an Epps-Pulley-style characteristic-function match to N(0, 1).
pub fn sigreg_epps_pulley(x: &Tensor, num_slices: usize, num_points: usize) -> Result<Tensor> {
    let (_, dim) = x.dims2()?;
    let device = x.device();
    let work_dtype = x.dtype();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SIGREG_PROJECTION_SEED);

    let mut proj = vec![0f32; dim * num_slices];
    for slice in 0..num_slices {
        let mut norm = 0f32;
        for d in 0..dim {
            let v = rng.random_range(-1.0f32..1.0f32);
            proj[d * num_slices + slice] = v;
            norm += v * v;
        }
        let norm = norm.sqrt().max(1e-6);
        for d in 0..dim {
            proj[d * num_slices + slice] /= norm;
        }
    }
    let proj = Tensor::from_vec(proj, (dim, num_slices), device)?.to_dtype(work_dtype)?;
    let projected = x.matmul(&proj)?; // [N, M]

    let knots = num_points.max(3);
    let mut per_t = Vec::with_capacity(knots);
    for i in 0..knots {
        let t = -5.0f32 + 10.0f32 * (i as f32) / ((knots - 1) as f32);
        let expected_cf = (-0.5f32 * t * t).exp();
        let scaled = projected.affine(t as f64, 0.0)?;
        let cos_mean = scaled.cos()?.mean(0)?;
        let sin_mean = scaled.sin()?.mean(0)?;
        let expected = Tensor::from_vec(vec![expected_cf; num_slices], (num_slices,), device)?
            .to_dtype(work_dtype)?;
        let err = cos_mean
            .broadcast_sub(&expected)?
            .sqr()?
            .broadcast_add(&sin_mean.sqr()?)?
            .affine(expected_cf as f64, 0.0)?;
        per_t.push(err.unsqueeze(0)?);
    }

    let stacked = Tensor::cat(&per_t, 0)?;
    stacked.mean_all().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn prediction_loss_is_invariant_to_vector_scale() -> Result<()> {
        let device = Device::Cpu;
        let target = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device)?;
        let pred = Tensor::from_vec(vec![6.0f32, 8.0], (1, 2), &device)?;
        let loss = prediction_loss(&pred, &target)?.to_vec0::<f32>()?;
        assert!(loss < 1e-5, "normalized loss should be near zero, got {loss}");
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
}
