use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use crate::model::attention::CrossAttention;

use super::DecoderKind;

/// Decoder-specific adapter: turns private planner slots into generation-facing conditioning slots.
/// This is the explicit bridge between planner memory and a concrete decoder family.
pub struct DecoderAdapter {
    query_embed: nn::Embedding,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    proj: nn::Linear,
    num_output_slots: usize,
    planner_dim: usize,
}

impl DecoderAdapter {
    pub fn new(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
    ) -> Result<Self> {
        let query_embed = nn::embedding(num_output_slots, planner_dim, vb.pp("query_embed"))?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), planner_dim, planner_dim, 8)?;
        let ln1 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln2"))?;
        let ff_hidden = (planner_dim * 4).max(256);
        let ff1 = nn::linear(planner_dim, ff_hidden, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_hidden, planner_dim, vb.pp("ff2"))?;
        let proj = nn::linear(planner_dim, model_dim, vb.pp("proj"))?;
        Ok(Self {
            query_embed,
            cross_attn,
            ln1,
            ln2,
            ff1,
            ff2,
            proj,
            num_output_slots,
            planner_dim,
        })
    }

    pub fn output_slots_for(kind: DecoderKind, planner_slots: usize) -> usize {
        match kind {
            DecoderKind::TextGeneralist => planner_slots.clamp(4, 8),
            DecoderKind::CodeSpecialist => planner_slots.clamp(16, 64),
        }
    }

    /// Input: planner slots [B, S, planner_dim]. Output: decoder conditioning slots [B, A, model_dim].
    pub fn forward(&self, planner_slots: &Tensor) -> Result<Tensor> {
        let (batch, _, _) = planner_slots.dims3()?;
        let query_ids: Vec<u32> = (0..self.num_output_slots as u32).collect();
        let query_ids =
            Tensor::from_vec(query_ids, (1, self.num_output_slots), planner_slots.device())?;
        let queries = self
            .query_embed
            .forward(&query_ids)?
            .broadcast_as((batch, self.num_output_slots, self.planner_dim))?;

        let normed = self.ln1.forward(&queries)?;
        let attended = self.cross_attn.forward(&normed, planner_slots)?;
        let slots = (queries + attended)?;

        let normed = self.ln2.forward(&slots)?;
        let ff = self.ff1.forward(&normed)?.gelu()?;
        let slots = (slots + self.ff2.forward(&ff)?)?;

        Ok(self.proj.forward(&slots)?)
    }
}
