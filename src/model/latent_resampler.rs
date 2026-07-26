use anyhow::{bail, Result};
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::attention::{CrossAttention, MultiHeadAttention};

/// Perceiver/Q-Former-style learned latent resampler.
///
/// Every layer first reads the source memory, then lets the latent queries
/// exchange information, and finally applies a parameter-matched SwiGLU MLP.
/// Cheap latent-space depth is where the fixed-size bottleneck is refined.
pub(crate) struct LatentResampler {
    query_embed: nn::Embedding,
    blocks: Vec<LatentResamplerBlock>,
    output_norm: nn::RmsNorm,
    num_queries: usize,
    dim: usize,
}

struct LatentResamplerBlock {
    cross_norm: nn::RmsNorm,
    cross_attn: CrossAttention,
    self_norm: nn::RmsNorm,
    self_attn: MultiHeadAttention,
    ff_norm: nn::RmsNorm,
    ff_gate: nn::Linear,
    ff_up: nn::Linear,
    ff_down: nn::Linear,
}

impl LatentResamplerBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, heads: usize) -> Result<Self> {
        let ff_dim = ((dim * 8) / 3).max(dim);
        Ok(Self {
            cross_norm: nn::rms_norm(dim, 1e-6, vb.pp("cross_norm"))?,
            cross_attn: CrossAttention::new(vb.pp("cross_attn"), dim, dim, heads)?,
            self_norm: nn::rms_norm(dim, 1e-6, vb.pp("self_norm"))?,
            self_attn: MultiHeadAttention::new(vb.pp("self_attn"), dim, heads)?,
            ff_norm: nn::rms_norm(dim, 1e-6, vb.pp("ff_norm"))?,
            ff_gate: nn::linear_no_bias(dim, ff_dim, vb.pp("ff_gate"))?,
            ff_up: nn::linear_no_bias(dim, ff_dim, vb.pp("ff_up"))?,
            ff_down: nn::linear_no_bias(ff_dim, dim, vb.pp("ff_down"))?,
        })
    }

    fn forward(
        &self,
        latents: &Tensor,
        memory: &Tensor,
        memory_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cross = self.cross_attn.forward_masked(
            &self.cross_norm.forward_diff(latents)?,
            memory,
            memory_mask,
        )?;
        let latents = (latents + cross)?;
        let mixed = self
            .self_attn
            .forward(&self.self_norm.forward_diff(&latents)?)?;
        let latents = (latents + mixed)?;
        let normed = self.ff_norm.forward_diff(&latents)?;
        let ff = self
            .ff_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.ff_up.forward(&normed)?)?;
        (latents + self.ff_down.forward(&ff)?).map_err(Into::into)
    }
}

impl LatentResampler {
    pub(crate) fn new(
        vb: VarBuilder<'_>,
        dim: usize,
        heads: usize,
        num_queries: usize,
        depth: usize,
    ) -> Result<Self> {
        if num_queries == 0 || depth == 0 {
            bail!("latent resampler requires non-zero query count and depth");
        }
        if !dim.is_multiple_of(heads) {
            bail!("latent resampler dim {dim} must be divisible by {heads} heads");
        }
        let query_embed = nn::embedding(num_queries, dim, vb.pp("query_embed"))?;
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            blocks.push(LatentResamplerBlock::new(
                vb.pp(format!("blocks.{index}")),
                dim,
                heads,
            )?);
        }
        Ok(Self {
            query_embed,
            blocks,
            output_norm: nn::rms_norm(dim, 1e-6, vb.pp("output_norm"))?,
            num_queries,
            dim,
        })
    }

    pub(crate) fn queries(&self, batch: usize, device: &candle_core::Device) -> Result<Tensor> {
        let ids = Tensor::arange(0u32, self.num_queries as u32, device)?.unsqueeze(0)?;
        self.query_embed
            .forward(&ids)?
            .broadcast_as((batch, self.num_queries, self.dim))?
            .contiguous()
            .map_err(Into::into)
    }

    pub(crate) fn forward(&self, memory: &Tensor, memory_mask: Option<&Tensor>) -> Result<Tensor> {
        let queries = self.queries(memory.dim(0)?, memory.device())?;
        self.forward_from_queries(&queries, memory, memory_mask)
    }

    pub(crate) fn forward_from_queries(
        &self,
        queries: &Tensor,
        memory: &Tensor,
        memory_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut latents = queries.clone();
        for block in &self.blocks {
            latents = block.forward(&latents, memory, memory_mask)?;
        }
        self.output_norm.forward_diff(&latents).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn resampler_preserves_fixed_latent_shape() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let resampler = LatentResampler::new(
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            16,
            4,
            5,
            2,
        )?;
        let memory = Tensor::randn(0f32, 1f32, (3, 11, 16), &device)?;
        let mask = Tensor::ones((3, 11), DType::F32, &device)?;
        assert_eq!(resampler.forward(&memory, Some(&mask))?.dims(), &[3, 5, 16]);
        Ok(())
    }
}
