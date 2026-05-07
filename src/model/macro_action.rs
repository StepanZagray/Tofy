use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::attention::TransformerBlock;
use super::orchestrator_head::NUM_ACTIONS;

/// Encodes a variable-length sequence of primitive actions into one macro-action vector.
pub struct MacroActionEncoder {
    action_embed: nn::Embedding,
    len_embed: nn::Embedding,
    ln: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    max_len: usize,
}

impl MacroActionEncoder {
    pub fn new(vb: VarBuilder<'_>, dim: usize, max_len: usize) -> Result<Self> {
        let max_len = max_len.max(1);
        let action_embed = nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?;
        let len_embed = nn::embedding(max_len + 1, dim, vb.pp("len_embed"))?;
        let ln = nn::layer_norm(dim, 1e-5, vb.pp("ln"))?;
        let hidden = (dim * 2).max(256);
        let fc1 = nn::linear(dim, hidden, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden, dim, vb.pp("fc2"))?;
        Ok(Self {
            action_embed,
            len_embed,
            ln,
            fc1,
            fc2,
            max_len,
        })
    }

    /// action_ids: [batch, macro_len], lengths: [batch].
    pub fn forward(&self, action_ids: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        let (batch, macro_len) = action_ids.dims2()?;
        let action_emb = self.action_embed.forward(action_ids)?;
        let len_values = lengths.to_vec1::<u32>()?;
        let mut mask = Vec::with_capacity(batch * macro_len);
        let mut clamped_lens = Vec::with_capacity(batch);
        for &len in &len_values {
            let len = (len as usize).clamp(1, self.max_len).min(macro_len);
            clamped_lens.push(len as u32);
            for pos in 0..macro_len {
                mask.push(if pos < len { 1.0f32 } else { 0.0f32 });
            }
        }
        let mask = Tensor::from_vec(mask, (batch, macro_len, 1), action_ids.device())?
            .to_dtype(action_emb.dtype())?;
        let summed = action_emb.broadcast_mul(&mask)?.sum(1)?;
        let denom = Tensor::from_vec(
            clamped_lens
                .iter()
                .map(|&len| len.max(1) as f32)
                .collect::<Vec<_>>(),
            (batch, 1),
            action_ids.device(),
        )?
        .to_dtype(action_emb.dtype())?;
        let pooled = summed.broadcast_div(&denom)?;
        let len_ids = Tensor::from_vec(clamped_lens, (batch,), action_ids.device())?;
        let with_len = (pooled + self.len_embed.forward(&len_ids)?)?;
        let hidden = self.fc1.forward(&self.ln.forward(&with_len)?)?.gelu()?;
        self.fc2.forward(&hidden).map_err(Into::into)
    }

    pub fn forward_from_slices(
        &self,
        action_ids: &[Vec<u32>],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let batch = action_ids.len().max(1);
        let macro_len = action_ids
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(1)
            .clamp(1, self.max_len);
        let mut flat = Vec::with_capacity(batch * macro_len);
        let mut lens = Vec::with_capacity(batch);
        for seq in action_ids {
            let len = seq.len().clamp(1, macro_len);
            lens.push(len as u32);
            for pos in 0..macro_len {
                let id = seq.get(pos).copied().unwrap_or(0);
                flat.push(id.min((NUM_ACTIONS - 1) as u32));
            }
        }
        let action_tensor = Tensor::from_vec(flat, (batch, macro_len), device)?;
        let lengths = Tensor::from_vec(lens, (batch,), device)?;
        self.forward(&action_tensor, &lengths)
    }
}

/// Long-horizon latent transition conditioned on a learned macro-action vector.
pub struct HighLevelWorldTransition {
    macro_proj: nn::Linear,
    blocks: Vec<TransformerBlock>,
    delta_ln: nn::LayerNorm,
    dim: usize,
}

impl HighLevelWorldTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let macro_proj = nn::linear(dim, dim, vb.pp("macro_proj"))?;
        let num_blocks = 3;
        let ff_dim = (dim * 5).max(320);
        let heads = if dim.is_multiple_of(8) { 8 } else { 1 };
        let mut blocks = Vec::with_capacity(num_blocks);
        for idx in 0..num_blocks {
            blocks.push(TransformerBlock::new(
                vb.pp(format!("block_{idx}")),
                dim,
                heads,
                ff_dim,
            )?);
        }
        let delta_ln = nn::layer_norm(dim, 1e-5, vb.pp("delta_ln"))?;
        Ok(Self {
            macro_proj,
            blocks,
            delta_ln,
            dim,
        })
    }

    pub fn forward_delta(&self, state_slots: &Tensor, macro_action: &Tensor) -> Result<Tensor> {
        let (batch, slots, _) = state_slots.dims3()?;
        let macro_vec = self
            .macro_proj
            .forward(macro_action)?
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        let mut hidden = (state_slots + macro_vec)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.delta_ln.forward(&hidden).map_err(Into::into)
    }

    pub fn forward(&self, state_slots: &Tensor, macro_action: &Tensor) -> Result<Tensor> {
        let delta = self.forward_delta(state_slots, macro_action)?;
        (state_slots + delta).map_err(Into::into)
    }
}
