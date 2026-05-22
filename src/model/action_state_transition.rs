use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::action_classifier_head::NUM_ACTIONS;
use super::attention::MultiHeadAttention;

struct ActionConditionedBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    action_mod: nn::Linear,
    dim: usize,
}

impl ActionConditionedBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?,
            ln1: nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: nn::linear(dim, ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(ff_dim, dim, vb.pp("ff2"))?,
            action_mod: nn::linear(dim, dim * 4, vb.pp("action_mod"))?,
            dim,
        })
    }

    fn modulate(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let scale = scale.affine(0.01, 1.0)?;
        let shift = shift.affine(0.01, 0.0)?;
        Ok(x.broadcast_mul(&scale)?.broadcast_add(&shift)?)
    }

    fn forward(&self, x: &Tensor, action_vec: &Tensor) -> Result<Tensor> {
        let modulation = self.action_mod.forward(action_vec)?.unsqueeze(1)?;
        let attn_scale = modulation.narrow(2, 0, self.dim)?;
        let attn_shift = modulation.narrow(2, self.dim, self.dim)?;
        let ff_scale = modulation.narrow(2, self.dim * 2, self.dim)?;
        let ff_shift = modulation.narrow(2, self.dim * 3, self.dim)?;

        let normed = self.modulate(&self.ln1.forward(x)?, &attn_scale, &attn_shift)?;
        let x = (x + self.attn.forward(&normed)?)?;
        let normed = self.modulate(&self.ln2.forward(&x)?, &ff_scale, &ff_shift)?;
        let ff = self.ff2.forward(&self.ff1.forward(&normed)?.gelu()?)?;
        Ok((x + ff)?)
    }
}

/// Predicts next latent-slot sequence conditioned on current slots and an action id.
pub struct ActionStateTransition {
    action_embed: nn::Embedding,
    blocks: Vec<ActionConditionedBlock>,
    delta_ln: nn::LayerNorm,
}

impl ActionStateTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let action_embed = nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?;
        let num_blocks = 6;
        let num_heads = 16;
        let ff_dim = (dim * 4).max(320);
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(ActionConditionedBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                ff_dim,
            )?);
        }
        let delta_ln = nn::layer_norm(dim, 1e-5, vb.pp("delta_ln"))?;
        Ok(Self {
            action_embed,
            blocks,
            delta_ln,
        })
    }

    pub fn forward_delta(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let (batch, _, _) = state_slots.dims3()?;
        let mut action_ids = action_labels.to_vec();
        if action_ids.len() < batch {
            action_ids.resize(batch, 0);
        } else {
            action_ids.truncate(batch);
        }
        let action_ids = Tensor::from_vec(action_ids, (batch,), state_slots.device())?;
        let action_vec = self.action_embed.forward(&action_ids)?;
        let mut hidden = state_slots.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &action_vec)?;
        }
        self.delta_ln.forward(&hidden).map_err(Into::into)
    }

    pub fn forward(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let delta = self.forward_delta(state_slots, action_labels)?;
        (state_slots + delta).map_err(Into::into)
    }

    pub fn forward_one(&self, state_slots: &Tensor, action_label: u32) -> Result<Tensor> {
        self.forward(state_slots, &[action_label])
    }
}
