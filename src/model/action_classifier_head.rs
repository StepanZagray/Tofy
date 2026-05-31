//! Orchestrator action head: predicts whether the current reply should use text, code, legacy docs, or stop.
//!
//! Input: planner-state slots. Output: logits over supported actions
//! (`TextReply`, `Code`, `Done`, legacy `FetchDocs`).
//! Trained jointly with the dialog-transition model and used at inference to choose which decoder to run.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Number of actions the action_classifier can predict.
pub const NUM_ACTIONS: usize = 4;

/// MLP head on top of pooled latent slots → action logits.
pub struct NextActionClassifier {
    slot_score: nn::Linear,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl NextActionClassifier {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let hidden = (dim * 2).max(256);
        let slot_score = nn::linear(dim, 1, vb.pp("slot_score"))?;
        let fc1 = nn::linear(dim, hidden, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden, NUM_ACTIONS, vb.pp("fc2"))?;
        Ok(Self {
            slot_score,
            fc1,
            fc2,
        })
    }

    /// latent_slots: [batch, slots, dim]. Returns logits [batch, NUM_ACTIONS].
    pub fn forward(&self, latent_slots: &Tensor) -> Result<Tensor> {
        let (batch, slots, dim) = latent_slots.dims3()?;
        let scores = self
            .slot_score
            .forward(latent_slots)?
            .reshape((batch, slots))?;
        let weights = nn::ops::softmax(&scores, 1)?
            .unsqueeze(2)?
            .broadcast_as((batch, slots, dim))?;
        let pooled = latent_slots.broadcast_mul(&weights)?.sum(1)?;
        let h = self.fc1.forward(&pooled)?.relu()?;
        Ok(self.fc2.forward(&h)?)
    }
}
