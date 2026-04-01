//! Orchestrator action head: JEPA-style world-model head that predicts the next action from the transition latent.
//!
//! Input: next latent from WorldTransition (or encoder → transition). Output: logits over actions
//! (TextReply, Code, WriteFile, RunCli, Done). Trained jointly with the world model; used at inference
//! to decide the next action from the current state (not from a fixed step policy).

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Number of actions the orchestrator can predict.
pub const NUM_ACTIONS: usize = 5;

/// MLP head on top of pooled latent slots → action logits.
pub struct OrchestratorActionHead {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl OrchestratorActionHead {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let hidden = (dim * 2).max(256);
        let fc1 = nn::linear(dim, hidden, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden, NUM_ACTIONS, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    /// latent_slots: [batch, slots, dim]. Returns logits [batch, NUM_ACTIONS].
    pub fn forward(&self, latent_slots: &Tensor) -> Result<Tensor> {
        let pooled = latent_slots.mean(1)?;
        let h = self.fc1.forward(&pooled)?.relu()?;
        Ok(self.fc2.forward(&h)?)
    }

    /// Argmax over actions. latent_slots: [1, slots, dim]. Returns action index 0..NUM_ACTIONS.
    pub fn predict(&self, latent_slots: &Tensor) -> Result<usize> {
        let logits = self.forward(latent_slots)?;
        let logits_v = logits.to_vec2::<f32>()?;
        let row = logits_v.first().ok_or_else(|| anyhow::anyhow!("empty logits"))?;
        let (idx, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("empty row"))?;
        Ok(idx)
    }
}
