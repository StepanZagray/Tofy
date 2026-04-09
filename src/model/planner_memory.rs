use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::attention::CrossAttention;
use super::encoders::EncoderFeatures;

/// Planner memory: resamples encoder hidden states into a fixed set of private task-state slots.
/// These slots are used by the orchestrator/router and world transition, not directly by the decoders.
pub struct PlannerMemory {
    slot_embed: nn::Embedding,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    proj: nn::Linear,
    pool_proj: nn::Linear,
    num_slots: usize,
    in_dim: usize,
}

impl PlannerMemory {
    pub fn new(
        vb: VarBuilder<'_>,
        in_dim: usize,
        planner_dim: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let slot_embed = nn::embedding(num_slots, in_dim, vb.pp("slot_embed"))?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), in_dim, in_dim, 8)?;
        let ln1 = nn::layer_norm(in_dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(in_dim, 1e-5, vb.pp("ln2"))?;
        let ff_hidden = (in_dim * 4).max(256);
        let ff1 = nn::linear(in_dim, ff_hidden, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_hidden, in_dim, vb.pp("ff2"))?;
        let proj = nn::linear(in_dim, planner_dim, vb.pp("proj"))?;
        let pool_proj = nn::linear(planner_dim, 1, vb.pp("pool_proj"))?;
        Ok(Self {
            slot_embed,
            cross_attn,
            ln1,
            ln2,
            ff1,
            ff2,
            proj,
            pool_proj,
            num_slots,
            in_dim,
        })
    }

    /// Input: encoder hidden states [B, T, in_dim]. Output: planner slots [B, S, planner_dim].
    #[allow(dead_code)]
    pub fn forward(&self, encoder_hidden: &Tensor) -> Result<Tensor> {
        self.forward_masked(encoder_hidden, None)
    }

    pub fn forward_masked(
        &self,
        encoder_hidden: &Tensor,
        encoder_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let encoder_hidden = encoder_hidden.contiguous()?;
        let (batch, _, _) = encoder_hidden.dims3()?;
        let slot_ids: Vec<u32> = (0..self.num_slots as u32).collect();
        let slot_ids = Tensor::from_vec(slot_ids, (1, self.num_slots), encoder_hidden.device())?;
        let queries = self
            .slot_embed
            .forward(&slot_ids)?
            .broadcast_as((batch, self.num_slots, self.in_dim))?
            .contiguous()?;

        let normed = self.ln1.forward(&queries)?;
        let attended = self
            .cross_attn
            .forward_masked(&normed, &encoder_hidden, encoder_mask)?;
        let slots = (queries + attended)?;

        let normed = self.ln2.forward(&slots)?;
        let ff = self.ff1.forward(&normed)?.gelu()?;
        let slots = (slots + self.ff2.forward(&ff)?)?;

        Ok(self.proj.forward(&slots)?)
    }

    #[allow(dead_code)]
    pub fn forward_encoder(&self, features: &EncoderFeatures) -> Result<Tensor> {
        self.forward(&features.memory()?)
    }

    pub fn pool(&self, planner_slots: &Tensor) -> Result<Tensor> {
        let (batch, slots, dim) = planner_slots.dims3()?;
        let scores = self
            .pool_proj
            .forward(planner_slots)?
            .reshape((batch, slots))?;
        let weights = nn::ops::softmax(&scores, 1)?
            .unsqueeze(2)?
            .broadcast_as((batch, slots, dim))?;
        Ok(planner_slots.broadcast_mul(&weights)?.sum(1)?)
    }

    pub fn fold_slots(
        &self,
        prev_slots: &Tensor,
        next_slots: &Tensor,
        retain: f64,
    ) -> Result<Tensor> {
        let retain = retain.clamp(0.0, 1.0);
        prev_slots
            .affine(retain, 0.0)?
            .broadcast_add(&next_slots.affine(1.0 - retain, 0.0)?)
            .map_err(Into::into)
    }
}
