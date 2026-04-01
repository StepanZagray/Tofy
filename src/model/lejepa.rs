use anyhow::Result;
use candle_core::Tensor;
use rand::Rng;

pub fn prediction_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    Ok(pred.broadcast_sub(target)?.sqr()?.mean_all()?)
}

pub fn mean_cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dot = a.broadcast_mul(b)?.sum(1)?;
    let a_norm = a.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    let b_norm = b.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, 1e10)?;
    dot.broadcast_div(&a_norm.broadcast_mul(&b_norm)?)?.mean_all().map_err(Into::into)
}

pub fn tensor_rms(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.mean_all()?.sqrt().map_err(Into::into)
}

pub fn flatten_latent_slots(latent_slots: &Tensor) -> Result<Tensor> {
    let (batch, slots, dim) = latent_slots.dims3()?;
    latent_slots.reshape((batch * slots, dim)).map_err(Into::into)
}

/// Lightweight in-repo SIGReg approximation using random 1D projections and
/// an Epps-Pulley-style characteristic-function match to N(0, 1).
pub fn sigreg_epps_pulley(
    x: &Tensor,
    num_slices: usize,
    num_points: usize,
) -> Result<Tensor> {
    let (num_samples, dim) = x.dims2()?;
    let device = x.device();
    let mut rng = rand::thread_rng();

    let mut proj = vec![0f32; dim * num_slices];
    for slice in 0..num_slices {
        let mut norm = 0f32;
        for d in 0..dim {
            let v = rng.gen_range(-1.0f32..1.0f32);
            proj[d * num_slices + slice] = v;
            norm += v * v;
        }
        let norm = norm.sqrt().max(1e-6);
        for d in 0..dim {
            proj[d * num_slices + slice] /= norm;
        }
    }
    let proj = Tensor::from_vec(proj, (dim, num_slices), device)?;
    let projected = x.matmul(&proj)?; // [N, M]

    let knots = num_points.max(3);
    let mut per_t = Vec::with_capacity(knots);
    for i in 0..knots {
        let t = -5.0f32 + 10.0f32 * (i as f32) / ((knots - 1) as f32);
        let expected_cf = (-0.5f32 * t * t).exp();
        let scaled = projected.affine(t as f64, 0.0)?;
        let cos_mean = scaled.cos()?.mean(0)?;
        let sin_mean = scaled.sin()?.mean(0)?;
        let expected = Tensor::from_vec(vec![expected_cf; num_slices], (num_slices,), device)?;
        let err = cos_mean
            .broadcast_sub(&expected)?
            .sqr()?
            .broadcast_add(&sin_mean.sqr()?)?
            .affine(expected_cf as f64, 0.0)?;
        per_t.push(err.unsqueeze(0)?);
    }

    let stacked = Tensor::cat(&per_t, 0)?;
    let _ = num_samples;
    stacked.mean_all().map_err(Into::into)
}
