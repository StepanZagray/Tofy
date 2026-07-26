use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{self as nn, ops, Module, VarBuilder};

use crate::model::latent_resampler::LatentResampler;

/// Decoder-specific adapter: turns private context slots into generation-facing conditioning slots.
pub struct DecoderConditioningAdapter {
    resampler: LatentResampler,
    index_proj: nn::Linear,
    gate_proj: nn::Linear,
    proj: nn::Linear,
    output_norm: nn::RmsNorm,
    num_output_slots: usize,
    compress_rate: usize,
}

impl DecoderConditioningAdapter {
    pub fn new(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
    ) -> Result<Self> {
        let compress_rate = std::env::var("TOFY_DECODER_ADAPTER_COMPRESS_RATE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        Self::new_with_compress_rate(vb, planner_dim, model_dim, num_output_slots, compress_rate)
    }

    pub fn new_with_compress_rate(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
        compress_rate: usize,
    ) -> Result<Self> {
        if compress_rate == 0 {
            anyhow::bail!("decoder adapter compress_rate must be non-zero");
        }
        let depth = std::env::var("TOFY_ADAPTER_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let resampler = LatentResampler::new(
            vb.pp("resampler"),
            planner_dim,
            adapter_heads(planner_dim),
            num_output_slots,
            depth,
        )?;
        let index_proj = nn::linear(planner_dim, 1, vb.pp("index_proj"))?;
        let gate_proj = nn::linear(planner_dim, planner_dim, vb.pp("gate_proj"))?;
        let proj = nn::linear(planner_dim, model_dim, vb.pp("proj"))?;
        let output_norm = nn::rms_norm(model_dim, 1e-6, vb.pp("output_norm"))?;
        Ok(Self {
            resampler,
            index_proj,
            gate_proj,
            proj,
            output_norm,
            num_output_slots,
            compress_rate,
        })
    }

    pub fn compress_rate(&self) -> usize {
        self.compress_rate
    }

    /// Input: context slots [B, S, planner_dim]. Output: decoder conditioning slots [B, A, model_dim].
    pub fn forward(&self, context_slots: &Tensor) -> Result<Tensor> {
        // A learned query bank is useful for reading a variable-length state, but
        // it must never become a static soft prompt.  Centering removes every
        // query/bias-only path: adapter(0) is exactly zero, and non-zero decoder
        // conditioning has to be explained by the supplied world slots.
        let output = self.forward_uncentered(context_slots)?;
        // Keep the baseline attached. Detaching it would produce a biased
        // straight-through gradient for query/bias-only paths that cancel in
        // the actual forward function.
        let baseline = self.forward_uncentered(&context_slots.zeros_like()?)?;
        output.broadcast_sub(&baseline).map_err(Into::into)
    }

    fn forward_uncentered(&self, context_slots: &Tensor) -> Result<Tensor> {
        let batch = context_slots.dim(0)?;
        let memory = compressed_context_compressor(
            context_slots,
            self.num_output_slots,
            self.compress_rate,
        )?
        .contiguous()?;
        let salience = ops::softmax(&self.index_proj.forward(&memory)?, D::Minus2)?;
        let salience_memory = memory.broadcast_mul(&salience)?;
        let global_memory = salience_memory.sum(1)?;
        let memory_gate = self
            .gate_proj
            .forward(&global_memory)
            .and_then(|gate| ops::sigmoid(&gate))?
            .unsqueeze(1)?;

        let queries = self
            .resampler
            .queries(batch, context_slots.device())?
            .broadcast_add(&global_memory.unsqueeze(1)?)?;
        let slots = self
            .resampler
            .forward_from_queries(&queries, &memory, None)?
            .broadcast_mul(&memory_gate)?;
        Ok(self.output_norm.forward_diff(&self.proj.forward(&slots)?)?)
    }
}

fn adapter_heads(dim: usize) -> usize {
    [16, 8, 4, 2]
        .into_iter()
        .find(|heads| dim.is_multiple_of(*heads))
        .unwrap_or(1)
}

fn compressed_context_compressor(
    context_slots: &Tensor,
    output_slots: usize,
    compress_rate: usize,
) -> Result<Tensor> {
    let (_, slots, _) = context_slots.dims3()?;
    let recent = output_slots.clamp(1, slots.max(1)).min(slots);
    let recent_start = slots.saturating_sub(recent);
    let recent_memory = context_slots.narrow(1, recent_start, recent)?;
    if recent_start == 0 {
        return Ok(recent_memory);
    }

    let compress_rate = compress_rate.max(1);
    let mut compressed = Vec::new();
    let mut start = 0usize;
    while start < recent_start {
        let len = (recent_start - start).min(compress_rate);
        let scale = 1.0 / len.max(1) as f64;
        compressed.push(
            context_slots
                .narrow(1, start, len)?
                .sum(1)?
                .affine(scale, 0.0)?
                .unsqueeze(1)?,
        );
        start += len;
    }

    if compressed.is_empty() {
        return Ok(recent_memory);
    }
    let mut refs = compressed.iter().collect::<Vec<_>>();
    refs.push(&recent_memory);
    Tensor::cat(&refs, 1).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn zero_world_slots_emit_zero_conditioning() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let adapter = DecoderConditioningAdapter::new_with_compress_rate(
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            8,
            8,
            4,
            1,
        )?;
        let zero = Tensor::zeros((2, 4, 8), DType::F32, &device)?;
        let output = adapter.forward(&zero)?;
        let max = crate::util::scalar_f32(&output.abs()?.max_all()?)?;
        assert!(max <= 1e-6, "adapter(0) must be zero, got {max}");
        Ok(())
    }
}
