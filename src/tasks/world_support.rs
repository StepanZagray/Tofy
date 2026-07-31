use anyhow::Result;
use candle_core::Tensor;

pub(crate) fn different_group_conditioning_latent(
    world_latent: &Tensor,
    groups: &[usize],
    offset: usize,
) -> Result<Tensor> {
    let batch = world_latent.dim(0)?;
    if groups.len() != batch {
        anyhow::bail!("conditioning group count must match batch size");
    }
    if batch <= 1 || groups.iter().all(|group| *group == groups[0]) {
        return Ok(world_latent.zeros_like()?);
    }
    let start_offset = offset.max(1);
    let mut perm = Vec::with_capacity(batch);
    for index in 0..batch {
        let replacement = (start_offset..start_offset + batch)
            .map(|delta| (index + delta) % batch)
            .find(|candidate| groups[*candidate] != groups[index])
            .expect("at least two groups checked above");
        perm.push(replacement as u32);
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

/// Cross-entropy against a `smoothing`-interpolated uniform target.
///
/// Callers must keep this off the reported validation CE: smoothing adds a
/// vocabulary-sized constant floor, so a smoothed number is not comparable
/// with an unsmoothed one across runs.
pub(crate) fn masked_cross_entropy_smoothed(
    logits: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
    smoothing: f64,
) -> Result<Tensor> {
    let nll = masked_weighted_cross_entropy(logits, labels, mask, None)?;
    if smoothing <= 0.0 {
        return Ok(nll);
    }
    let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?;
    let flat_mask = mask
        .reshape((mask.elem_count(),))?
        .to_dtype(log_probs.dtype())?;
    let uniform = log_probs
        .mean(candle_core::D::Minus1)?
        .affine(-1.0, 0.0)?
        .reshape((labels.elem_count(),))?
        .broadcast_mul(&flat_mask)?
        .sum_all()?
        .broadcast_div(&flat_mask.sum_all()?)?;
    // (1 - eps) * NLL + eps * mean_vocab(-log p); the mean already divides by
    // the vocabulary size, so no extra 1/V factor here.
    nll.affine(1.0 - smoothing, 0.0)?
        .broadcast_add(&uniform.affine(smoothing, 0.0)?)
        .map_err(Into::into)
}

/// Penalizes the target tokens under a deliberately wrong conditioning state.
/// Unlike `-cross_entropy`, this keeps a useful gradient when the decoder has
/// already assigned the target probability very close to one.
pub(crate) fn masked_unlikelihood(
    logits: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    // Qwen's shifted sequence view is non-contiguous on CUDA, while CUDA
    // gather requires contiguous storage.
    let logits = logits.contiguous()?;
    let target_indices = labels.unsqueeze(candle_core::D::Minus1)?;
    let target_logits = logits
        .gather(&target_indices, candle_core::D::Minus1)?
        .squeeze(candle_core::D::Minus1)?;
    // -log(1 - p_target) is exactly softplus(target_logit -
    // logsumexp(other_logits)). Computing it from the logit margin, rather
    // than `log(1 - exp(log_p))`, avoids cancellation and keeps a unit-scale
    // target gradient even when p_target is nearly one.
    let target_penalty =
        Tensor::full(-1e4f32, target_indices.dims(), logits.device())?.to_dtype(logits.dtype())?;
    let other_logits = logits.broadcast_add(&logits.zeros_like()?.scatter(
        &target_indices,
        &target_penalty,
        candle_core::D::Minus1,
    )?)?;
    let margin = target_logits.broadcast_sub(&other_logits.log_sum_exp(candle_core::D::Minus1)?)?;
    let one = Tensor::new(1f32, logits.device())?.to_dtype(logits.dtype())?;
    let softplus_correction = margin
        .abs()?
        .affine(-1.0, 0.0)?
        .exp()?
        .broadcast_add(&one)?
        .log()?;
    let penalties = margin
        .relu()?
        .broadcast_add(&softplus_correction)?
        .reshape((labels.elem_count(),))?;
    let flat_mask = mask
        .reshape((mask.elem_count(),))?
        .to_dtype(penalties.dtype())?;
    penalties
        .broadcast_mul(&flat_mask)?
        .sum_all()?
        .broadcast_div(&flat_mask.sum_all()?)
        .map_err(Into::into)
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
    use anyhow::Context;
    use candle_core::{Device, Var};

    #[test]
    fn masked_cross_entropy_supports_batch_one() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![2.0f32, 0.0, -1.0, 0.0, 2.0, -1.0],
            (1, 2, 3),
            &Device::Cpu,
        )?;
        let labels = Tensor::from_vec(vec![0u32, 1], (1, 2), &Device::Cpu)?;
        let mask = Tensor::ones((1, 2), candle_core::DType::F32, &Device::Cpu)?;
        let loss = crate::util::scalar_f32(&masked_cross_entropy(&logits, &labels, &mask)?)?;
        assert!(loss.is_finite() && loss > 0.0);
        Ok(())
    }

    #[test]
    fn unlikelihood_penalizes_an_overconfident_target() -> Result<()> {
        let device = Device::Cpu;
        let logits_var = Var::new(&[12.0f32, -12.0, -12.0], &device)?;
        let logits = logits_var.as_tensor().reshape((1, 1, 3))?;
        let labels = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
        let mask = Tensor::ones((1, 1), candle_core::DType::F32, &device)?;
        let loss = masked_unlikelihood(&logits, &labels, &mask)?;
        let loss_value = crate::util::scalar_f32(&loss)?;
        let grads = loss.backward()?;
        let target_gradient = grads
            .get(&logits_var)
            .context("missing unlikelihood target-logit gradient")?
            .to_vec1::<f32>()?[0];
        assert!(
            loss_value.is_finite() && loss_value > 1.0,
            "loss={loss_value}"
        );
        assert!(
            target_gradient.abs() > 0.1,
            "overconfident wrong target must retain a useful gradient, got {target_gradient}"
        );
        Ok(())
    }

    #[test]
    fn mismatched_conditioning_never_uses_the_same_group() -> Result<()> {
        let values = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (4, 1, 1), &Device::Cpu)?;
        let groups = [1usize, 1, 2, 2];
        let mismatched = different_group_conditioning_latent(&values, &groups, 1)?;
        assert_eq!(
            crate::util::vec1_f32(&mismatched.flatten_all()?)?,
            vec![2.0, 2.0, 0.0, 0.0]
        );
        Ok(())
    }
}
