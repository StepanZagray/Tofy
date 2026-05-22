use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{self as nn, ops, Module, VarBuilder};

use crate::model::attention::CrossAttention;

use super::DecoderKind;

/// Decoder-specific adapter: turns private context slots into generation-facing conditioning slots.
/// This is the explicit bridge between context compressor and a concrete decoder family.
pub struct DecoderConditioningAdapter {
    query_embed: nn::Embedding,
    action_embed: nn::Embedding,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    index_proj: nn::Linear,
    gate_proj: nn::Linear,
    proj: nn::Linear,
    num_output_slots: usize,
    planner_dim: usize,
}

impl DecoderConditioningAdapter {
    pub fn new(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
    ) -> Result<Self> {
        let query_embed = nn::embedding(num_output_slots, planner_dim, vb.pp("query_embed"))?;
        let action_embed = nn::embedding(
            crate::model::action_classifier_head::NUM_ACTIONS,
            planner_dim,
            vb.pp("action_embed"),
        )?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), planner_dim, planner_dim, 8)?;
        let ln1 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln2"))?;
        let ff_hidden = (planner_dim * 4).max(256);
        let ff1 = nn::linear(planner_dim, ff_hidden, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_hidden, planner_dim, vb.pp("ff2"))?;
        let index_proj = nn::linear(planner_dim, 1, vb.pp("index_proj"))?;
        let gate_proj = nn::linear(planner_dim, planner_dim, vb.pp("gate_proj"))?;
        let proj = nn::linear(planner_dim, model_dim, vb.pp("proj"))?;
        Ok(Self {
            query_embed,
            action_embed,
            cross_attn,
            ln1,
            ln2,
            ff1,
            ff2,
            index_proj,
            gate_proj,
            proj,
            num_output_slots,
            planner_dim,
        })
    }

    pub fn output_slots_for(kind: DecoderKind, context_slots: usize) -> usize {
        match kind {
            DecoderKind::TextGeneralist => context_slots.clamp(4, 8),
            DecoderKind::CodeSpecialist => context_slots.clamp(16, 64),
        }
    }

    /// Input: context slots [B, S, planner_dim]. Output: decoder conditioning slots [B, A, model_dim].
    pub fn forward(&self, context_slots: &Tensor) -> Result<Tensor> {
        self.forward_with_action(context_slots, 0)
    }

    /// Builds action-aware local decoder-plan slots from global/planned context slots.
    ///
    /// The world/planner state is good at broad routing; this adapter specializes it into
    /// short-horizon memory that the next decoder call can cross-attend to.
    pub fn forward_with_action(&self, context_slots: &Tensor, action_id: u32) -> Result<Tensor> {
        let (batch, _, _) = context_slots.dims3()?;
        let memory = compressed_context_compressor(context_slots, self.num_output_slots)?;
        let action_ids =
            Tensor::from_vec(vec![action_id; batch], (batch,), context_slots.device())?;
        let action_state = self.action_embed.forward(&action_ids)?;
        let action_memory = memory.broadcast_add(&action_state.unsqueeze(1)?)?;
        let salience = ops::softmax(&self.index_proj.forward(&action_memory)?, D::Minus2)?;
        let memory = action_memory.broadcast_mul(&salience)?;
        let global_memory = memory.sum(1)?;
        let memory_gate = self
            .gate_proj
            .forward(&global_memory)?
            .relu()?
            .clamp(0.0, 1.0)?
            .unsqueeze(1)?;

        let query_ids: Vec<u32> = (0..self.num_output_slots as u32).collect();
        let query_ids = Tensor::from_vec(
            query_ids,
            (1, self.num_output_slots),
            context_slots.device(),
        )?;
        let queries = self.query_embed.forward(&query_ids)?.broadcast_as((
            batch,
            self.num_output_slots,
            self.planner_dim,
        ))?;

        let action_queries = queries.broadcast_add(&action_state.unsqueeze(1)?)?;
        let gated_queries = action_queries.broadcast_add(&global_memory.unsqueeze(1)?)?;
        let normed = self.ln1.forward(&gated_queries)?;
        let attended = self
            .cross_attn
            .forward(&normed, &memory)?
            .broadcast_mul(&memory_gate)?;
        let slots = (action_queries + attended)?;

        let normed = self.ln2.forward(&slots)?;
        let ff = self.ff1.forward(&normed)?.gelu()?;
        let slots = (slots + self.ff2.forward(&ff)?)?;

        Ok(self.proj.forward(&slots)?)
    }
}

fn compressed_context_compressor(context_slots: &Tensor, output_slots: usize) -> Result<Tensor> {
    let (_, slots, _) = context_slots.dims3()?;
    let recent = output_slots.clamp(1, slots.max(1)).min(slots);
    let recent_start = slots.saturating_sub(recent);
    let recent_memory = context_slots.narrow(1, recent_start, recent)?;
    if recent_start == 0 {
        return Ok(recent_memory);
    }

    let compress_rate = std::env::var("TOFY_DECODER_ADAPTER_COMPRESS_RATE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let mut compressed = Vec::new();
    let mut start = 0usize;
    while start < recent_start {
        let len = (recent_start - start).min(compress_rate);
        let scale = 1.0 / len.max(1) as f64;
        compressed.push(
            context_slots
                .narrow(1, start, len)?
                .sum(1)?
                .affine(scale, 0.0)?
                .unsqueeze(1)?,
        );
        start += len;
    }

    if compressed.is_empty() {
        return Ok(recent_memory);
    }
    let mut refs = compressed.iter().collect::<Vec<_>>();
    refs.push(&recent_memory);
    Tensor::cat(&refs, 1).map_err(Into::into)
}
