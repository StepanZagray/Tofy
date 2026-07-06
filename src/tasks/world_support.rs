use anyhow::Result;
use candle_core::Tensor;

pub(crate) fn batch_shuffled_conditioning_latent(world_latent: &Tensor) -> Result<Tensor> {
    let batch = world_latent.dim(0)?;
    if batch <= 1 {
        return Ok(world_latent.clone());
    }
    let mut perm: Vec<usize> = (0..batch).collect();
    use rand::seq::SliceRandom;
    perm.shuffle(&mut rand::rng());
    let ids = Tensor::from_vec(
        perm.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
        (batch,),
        world_latent.device(),
    )?;
    world_latent
        .contiguous()?
        .index_select(&ids, 0)
        .map_err(Into::into)
}

pub(crate) fn hard_mismatched_conditioning_latent(world_latent: &Tensor) -> Result<Tensor> {
    shifted_conditioning_latent(world_latent, 1)
}

fn shifted_conditioning_latent(world_latent: &Tensor, offset: usize) -> Result<Tensor> {
    let batch = world_latent.dim(0)?;
    if batch <= 1 {
        return Ok(world_latent.zeros_like()?);
    }
    let offset = offset % batch;
    let mut perm = Vec::with_capacity(batch);
    for idx in 0..batch {
        perm.push(((idx + offset) % batch) as u32);
    }
    let ids = Tensor::from_vec(perm, (batch,), world_latent.device())?;
    world_latent
        .contiguous()?
        .index_select(&ids, 0)
        .map_err(Into::into)
}

pub(crate) fn masked_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    masked_weighted_cross_entropy(logits, labels, mask, None)
}

fn masked_weighted_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
    weights: Option<&Tensor>,
) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?;
    let token_nll = log_probs
        .gather(
            &labels.unsqueeze(candle_core::D::Minus1)?,
            candle_core::D::Minus1,
        )?
        .squeeze(candle_core::D::Minus1)?
        .affine(-1.0, 0.0)?
        .reshape((labels.elem_count(),))?;
    let flat_mask = mask
        .reshape((mask.elem_count(),))?
        .to_dtype(token_nll.dtype())?;
    let weighted = if let Some(weights) = weights {
        let flat_weights = weights
            .reshape((weights.elem_count(),))?
            .to_dtype(token_nll.dtype())?;
        token_nll
            .broadcast_mul(&flat_mask)?
            .broadcast_mul(&flat_weights)?
            .sum_all()?
            .broadcast_div(&flat_mask.broadcast_mul(&flat_weights)?.sum_all()?)?
    } else {
        token_nll
            .broadcast_mul(&flat_mask)?
            .sum_all()?
            .broadcast_div(&flat_mask.sum_all()?)?
    };
    Ok(weighted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn masked_cross_entropy_supports_batch_one() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![2.0f32, 0.0, -1.0, 0.0, 2.0, -1.0],
            (1, 2, 3),
            &Device::Cpu,
        )?;
        let labels = Tensor::from_vec(vec![0u32, 1], (1, 2), &Device::Cpu)?;
        let mask = Tensor::ones((1, 2), candle_core::DType::F32, &Device::Cpu)?;
        let loss = masked_cross_entropy(&logits, &labels, &mask)?.to_scalar::<f32>()?;
        assert!(loss.is_finite() && loss > 0.0);
        Ok(())
    }
}
