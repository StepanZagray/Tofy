use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::attention::TransformerBlock;
use super::orchestrator_head::NUM_ACTIONS;

/// Predicts next latent-slot sequence conditioned on current slots and an action id.
pub struct WorldTransition {
    action_embed: nn::Embedding,
    blocks: Vec<TransformerBlock>,
    ln: nn::LayerNorm,
    dim: usize,
}

impl WorldTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let action_embed = nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?;
        let mut blocks = Vec::with_capacity(2);
        for i in 0..2 {
            blocks.push(TransformerBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                8,
                (dim * 4).max(256),
            )?);
        }
        let ln = nn::layer_norm(dim, 1e-5, vb.pp("ln"))?;
        Ok(Self {
            action_embed,
            blocks,
            ln,
            dim,
        })
    }

    pub fn forward(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let (batch, slots, _) = state_slots.dims3()?;
        let mut action_ids = action_labels.to_vec();
        if action_ids.len() < batch {
            action_ids.resize(batch, 0);
        }
        let action_ids = Tensor::from_vec(action_ids, (batch,), state_slots.device())?;
        let action_vec = self
            .action_embed
            .forward(&action_ids)?
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        let mut hidden = (state_slots + action_vec)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.ln.forward(&hidden).map_err(Into::into)
    }

    pub fn forward_one(&self, state_slots: &Tensor, action_label: u32) -> Result<Tensor> {
        self.forward(state_slots, &[action_label])
    }
}
