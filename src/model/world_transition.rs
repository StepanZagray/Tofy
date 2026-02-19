use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Predicts next latent conditioned on current latent and a fixed reply action embedding.
pub struct WorldTransition {
    action_embed: nn::Embedding,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl WorldTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let action_dim = (dim / 4).max(32);
        let hidden = (dim * 3 / 2).max(128);
        let action_embed = nn::embedding(1, action_dim, vb.pp("action_embed"))?;
        let fc1 = nn::linear(dim + action_dim, hidden, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden, dim, vb.pp("fc2"))?;
        Ok(Self {
            action_embed,
            fc1,
            fc2,
        })
    }

    pub fn forward_reply(&self, state_latent: &Tensor) -> Result<Tensor> {
        let batch = state_latent.dim(0)?;
        let reply_ids = Tensor::from_vec(vec![0u32; batch], (batch,), state_latent.device())?;
        let action_vec = self.action_embed.forward(&reply_ids)?;
        let fused = Tensor::cat(&[state_latent, &action_vec], 1)?;
        let h = self.fc1.forward(&fused)?.relu()?;
        Ok(self.fc2.forward(&h)?)
    }
}
