use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::attention::{positional_encoding, DecoderBlock};

/// Transformer decoder with self-attention and cross-attention to encoder output
pub struct Decoder {
    embed: nn::Embedding,
    blocks: Vec<DecoderBlock>,
    ln_final: nn::LayerNorm,
    output_proj: nn::Linear, // Projects to vocab logits
    dim: usize,
}

impl Decoder {
    pub fn new(vb: VarBuilder<'_>, dim: usize, vocab_size: usize, num_layers: usize, num_heads: usize) -> Result<Self> {
        let embed = nn::embedding(vocab_size, dim, vb.pp("embed"))?;
        
        // Stack of decoder blocks with self-attention and cross-attention
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = DecoderBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                dim * 4,
            )?;
            blocks.push(block);
        }
        
        let ln_final = nn::layer_norm(dim, 1e-5, vb.pp("ln_final"))?;
        let output_proj = nn::linear(dim, vocab_size, vb.pp("output_proj"))?;
        
        Ok(Self { embed, blocks, ln_final, output_proj, dim })
    }

    /// Forward pass for training: given target tokens and encoder output, predict next tokens
    /// Input: target_ids [B, T], encoder_out [B, T_enc, D]
    /// Output: logits [B, T, vocab_size]
    pub fn forward(&self, target_ids: &Tensor, encoder_out: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = target_ids.dims2()?;
        
        // Embed target tokens
        let mut h = self.embed.forward(target_ids)?; // [B, T, D]
        
        // Add positional encoding
        let pe = positional_encoding(seq_len, self.dim, target_ids.device())?;
        h = h.broadcast_add(&pe)?;
        
        // Pass through decoder blocks with cross-attention to encoder
        for block in &self.blocks {
            h = block.forward(&h, encoder_out)?;
        }
        
        // Final layer norm and project to vocabulary
        let h = self.ln_final.forward(&h)?;
        Ok(self.output_proj.forward(&h)?)
    }

    /// Generate next token logits given encoder output and previous tokens
    /// Used during inference for autoregressive generation
    pub fn forward_step(&self, prev_tokens: &Tensor, encoder_out: &Tensor) -> Result<Tensor> {
        let logits = self.forward(prev_tokens, encoder_out)?;
        // Return only the last position's logits
        let last_pos = logits.dim(1)? - 1;
        Ok(logits.narrow(1, last_pos, 1)?.squeeze(1)?)
    }

    /// Generate initial token logits from encoder output (no previous tokens)
    /// Uses a learned start token embedding
    pub fn forward_initial(&self, encoder_out: &Tensor, start_token_id: u32) -> Result<Tensor> {
        let batch_size = encoder_out.dim(0)?;
        let device = encoder_out.device();
        
        // Create start token tensor [B, 1]
        let start_tokens = Tensor::full(start_token_id, (batch_size, 1), device)?;
        
        let logits = self.forward(&start_tokens, encoder_out)?;
        Ok(logits.squeeze(1)?) // [B, vocab_size]
    }
}

/// Causal mask for decoder self-attention (prevents attending to future tokens)
/// Returns a mask where mask[i][j] = 0 if j <= i, else -inf
#[allow(dead_code)]
pub fn create_causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    // Create lower triangular mask manually since tril isn't available
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                mask_data[i * seq_len + j] = 0.0; // Allow attending
            } else {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY; // Block attending
            }
        }
    }
    Ok(Tensor::from_vec(mask_data, (seq_len, seq_len), device)?)
}
