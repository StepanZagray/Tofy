use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::attention::{positional_encoding, TransformerBlock};

/// Transformer-based encoder with self-attention
pub struct Encoder {
    embed: nn::Embedding,
    blocks: Vec<TransformerBlock>,
    ln_final: nn::LayerNorm,
    dim: usize,
}

impl Encoder {
    pub fn new(vb: VarBuilder<'_>, vocab_size: usize, dim: usize, num_layers: usize, num_heads: usize) -> Result<Self> {
        let embed = nn::embedding(vocab_size, dim, vb.pp("embed"))?;
        
        // Stack of transformer blocks with self-attention
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = TransformerBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                dim * 4, // FF dimension is typically 4x embedding dim
            )?;
            blocks.push(block);
        }
        
        let ln_final = nn::layer_norm(dim, 1e-5, vb.pp("ln_final"))?;
        
        Ok(Self { embed, blocks, ln_final, dim })
    }

    /// Forward pass returning sequence of hidden states [B, T, D]
    pub fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        
        // Embed tokens
        let mut h = self.embed.forward(x)?; // [B, T, D]
        
        // Add positional encoding
        let pe = positional_encoding(seq_len, self.dim, x.device())?;
        h = h.broadcast_add(&pe)?;
        
        // Pass through transformer blocks
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        
        // Final layer norm
        Ok(self.ln_final.forward(&h)?)
    }

    /// Forward pass returning single latent vector [B, D] (mean pooling)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.forward_sequence(x)?;
        Ok(h.mean(1)?) // Mean pool over sequence dimension
    }
}
