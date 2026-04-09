use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use crate::model::attention::CrossAttention;

pub struct MultiQueryPool {
    query_embed: nn::Embedding,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    num_queries: usize,
    dim: usize,
}

impl MultiQueryPool {
    pub fn new(
        vb: VarBuilder<'_>,
        dim: usize,
        num_heads: usize,
        num_queries: usize,
    ) -> Result<Self> {
        let query_embed = nn::embedding(num_queries, dim, vb.pp("query_embed"))?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), dim, dim, num_heads)?;
        let ln1 = nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?;
        let ff1 = nn::linear(dim, dim * 4, vb.pp("ff1"))?;
        let ff2 = nn::linear(dim * 4, dim, vb.pp("ff2"))?;
        Ok(Self {
            query_embed,
            cross_attn,
            ln1,
            ln2,
            ff1,
            ff2,
            num_queries,
            dim,
        })
    }

    pub fn forward(&self, memory: &Tensor) -> Result<Tensor> {
        let (batch, _, _) = memory.dims3()?;
        let query_ids: Vec<u32> = (0..self.num_queries as u32).collect();
        let query_ids = Tensor::from_vec(query_ids, (1, self.num_queries), memory.device())?;
        let queries = self.query_embed.forward(&query_ids)?.broadcast_as((
            batch,
            self.num_queries,
            self.dim,
        ))?;
        let normed = self.ln1.forward(&queries)?;
        let attended = self.cross_attn.forward(&normed, memory)?;
        let pooled = (queries + attended)?;
        let normed = self.ln2.forward(&pooled)?;
        let ff = self.ff1.forward(&normed)?.gelu()?;
        Ok((pooled + self.ff2.forward(&ff)?)?)
    }
}
