use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::super::attention::{positional_encoding, TransformerBlock};

/// Shared transformer-based encoder implementation used by both online and teacher encoders.
pub(crate) struct EncoderBackbone {
    embed: nn::Embedding,
    blocks: Vec<TransformerBlock>,
    ln_final: nn::LayerNorm,
    dim: usize,
}

impl EncoderBackbone {
    pub(crate) fn new(
        vb: VarBuilder<'_>,
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let embed = nn::embedding(vocab_size, dim, vb.pp("embed"))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = TransformerBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                dim * 4,
            )?;
            blocks.push(block);
        }

        let ln_final = nn::layer_norm(dim, 1e-5, vb.pp("ln_final"))?;
        Ok(Self {
            embed,
            blocks,
            ln_final,
            dim,
        })
    }

    pub(crate) fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        let mut h = self.embed.forward(x)?;
        let pe = positional_encoding(seq_len, self.dim, x.device())?;
        h = h.broadcast_add(&pe)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        Ok(self.ln_final.forward(&h)?)
    }

    #[allow(dead_code)]
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.forward_sequence(x)?;
        Ok(h.mean(1)?)
    }

    pub(crate) fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed.forward(ids)?)
    }
}
