use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use super::action_classifier_head::NUM_ACTIONS;
use super::attention::TransformerBlock;

const MASK_VALUE: f32 = -1.0e4;

/// Encodes a variable-length sequence of primitive actions into one macro-action vector.
pub struct ActionSequenceEncoder {
    action_embed: nn::Embedding,
    position_embed: nn::Embedding,
    len_embed: nn::Embedding,
    blocks: Vec<TransformerBlock>,
    token_ln: nn::LayerNorm,
    out_ln: nn::LayerNorm,
    pool_proj: nn::Linear,
    fc1: nn::Linear,
    fc2: nn::Linear,
    max_len: usize,
    dim: usize,
}

impl ActionSequenceEncoder {
    pub fn new(vb: VarBuilder<'_>, dim: usize, max_len: usize) -> Result<Self> {
        let max_len = max_len.max(1);
        let action_embed = nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?;
        let position_embed = nn::embedding(max_len, dim, vb.pp("position_embed"))?;
        let len_embed = nn::embedding(max_len + 1, dim, vb.pp("len_embed"))?;
        let heads = transition_heads(dim);
        let ff_dim = (dim * 4).max(256);
        let mut blocks = Vec::with_capacity(2);
        for idx in 0..2 {
            blocks.push(TransformerBlock::new(
                vb.pp(format!("block_{idx}")),
                dim,
                heads,
                ff_dim,
            )?);
        }
        let token_ln = nn::layer_norm(dim, 1e-5, vb.pp("token_ln"))?;
        let out_ln = nn::layer_norm(dim, 1e-5, vb.pp("out_ln"))?;
        let pool_proj = nn::linear(dim, 1, vb.pp("pool_proj"))?;
        let hidden = (dim * 2).max(256);
        let fc1 = nn::linear(dim, hidden, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden, dim, vb.pp("fc2"))?;
        Ok(Self {
            action_embed,
            position_embed,
            len_embed,
            blocks,
            token_ln,
            out_ln,
            pool_proj,
            fc1,
            fc2,
            max_len,
            dim,
        })
    }

    /// action_ids: [batch, macro_len], lengths: [batch].
    pub fn forward(&self, action_ids: &Tensor, lengths: &Tensor) -> Result<Tensor> {
        let (batch, macro_len) = action_ids.dims2()?;
        let len_values = lengths.to_vec1::<u32>()?;
        let mut mask_values = Vec::with_capacity(batch * macro_len);
        let mut clamped_lens = Vec::with_capacity(batch);
        for batch_idx in 0..batch {
            let len = len_values.get(batch_idx).copied().unwrap_or(1);
            let len = (len as usize).clamp(1, self.max_len).min(macro_len);
            clamped_lens.push(len as u32);
            for pos in 0..macro_len {
                mask_values.push(if pos < len { 1.0f32 } else { 0.0f32 });
            }
        }
        let mask = Tensor::from_vec(mask_values, (batch, macro_len), action_ids.device())?;
        let mask_3d = mask.unsqueeze(2)?.to_dtype(DType::F32)?;
        let action_emb = self.action_embed.forward(action_ids)?;
        let pos_ids = Tensor::from_vec(
            (0..macro_len as u32).collect::<Vec<_>>(),
            (1, macro_len),
            action_ids.device(),
        )?;
        let pos_emb = self
            .position_embed
            .forward(&pos_ids)?
            .broadcast_as((batch, macro_len, self.dim))?;
        let mut hidden = action_emb
            .broadcast_add(&pos_emb)?
            .broadcast_mul(&mask_3d.to_dtype(action_emb.dtype())?)?;
        let mask_for_attention = mask.to_dtype(action_emb.dtype())?;
        for block in &self.blocks {
            hidden = block.forward_masked(&hidden, &mask_for_attention)?;
            hidden = hidden.broadcast_mul(&mask_3d.to_dtype(hidden.dtype())?)?;
        }

        let normed = self.token_ln.forward(&hidden)?;
        let scores = self.pool_proj.forward(&normed)?.squeeze(2)?;
        let bias = mask
            .affine(-MASK_VALUE as f64, MASK_VALUE as f64)?
            .to_dtype(scores.dtype())?;
        let weights = nn::ops::softmax(&scores.broadcast_add(&bias)?, 1)?
            .unsqueeze(2)?
            .to_dtype(hidden.dtype())?;
        let pooled = hidden.broadcast_mul(&weights)?.sum(1)?;
        let len_ids = Tensor::from_vec(clamped_lens, (batch,), action_ids.device())?;
        let with_len = (pooled + self.len_embed.forward(&len_ids)?)?;
        let hidden = self.fc1.forward(&self.out_ln.forward(&with_len)?)?.gelu()?;
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
pub struct MacroActionStateTransition {
    macro_proj: nn::Linear,
    blocks: Vec<TransformerBlock>,
    delta_ln: nn::LayerNorm,
    delta_proj: nn::Linear,
    delta_gate: nn::Linear,
    dim: usize,
}

impl MacroActionStateTransition {
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
        let delta_proj = nn::linear(dim, dim, vb.pp("delta_proj"))?;
        let delta_gate = nn::linear(dim, dim, vb.pp("delta_gate"))?;
        Ok(Self {
            macro_proj,
            blocks,
            delta_ln,
            delta_proj,
            delta_gate,
            dim,
        })
    }

    pub fn forward_delta(&self, state_slots: &Tensor, macro_action: &Tensor) -> Result<Tensor> {
        let (batch, slots, _) = state_slots.dims3()?;
        let macro_vec = self.macro_proj.forward(macro_action)?;
        let macro_bias = macro_vec
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        let mut hidden = state_slots.broadcast_add(&macro_bias)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let delta = self
            .delta_proj
            .forward(&self.delta_ln.forward(&hidden)?)?
            .tanh()?;
        let gate = nn::ops::sigmoid(&self.delta_gate.forward(&macro_vec)?)?
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        delta.broadcast_mul(&gate).map_err(Into::into)
    }

    pub fn forward(&self, state_slots: &Tensor, macro_action: &Tensor) -> Result<Tensor> {
        let delta = self.forward_delta(state_slots, macro_action)?;
        (state_slots + delta).map_err(Into::into)
    }
}

fn transition_heads(dim: usize) -> usize {
    [8, 4, 2]
        .into_iter()
        .find(|heads| dim.is_multiple_of(*heads))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn action_sequence_encoder_is_order_sensitive() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = ActionSequenceEncoder::new(vb, 16, 4)?;
        let out = encoder.forward_from_slices(&[vec![1, 2], vec![2, 1]], &device)?;
        let rows = out.to_vec2::<f32>()?;
        let diff = rows[0]
            .iter()
            .zip(rows[1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(diff > 1e-5, "order-sensitive encoder diff {diff}");
        Ok(())
    }
}
