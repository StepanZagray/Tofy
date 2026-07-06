use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{self as nn, ops, Module, VarBuilder};

use crate::model::attention::CrossAttention;

/// Decoder-specific adapter: turns private context slots into generation-facing conditioning slots.
pub struct DecoderConditioningAdapter {
    query_embed: nn::Embedding,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    extra_blocks: Vec<AdapterBlock>,
    index_proj: nn::Linear,
    gate_proj: nn::Linear,
    proj: nn::Linear,
    output_norm: nn::RmsNorm,
    num_output_slots: usize,
    planner_dim: usize,
    compress_rate: usize,
}

struct AdapterBlock {
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl AdapterBlock {
    fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let hidden = (dim * 4).max(256);
        Ok(Self {
            cross_attn: CrossAttention::new(vb.pp("cross_attn"), dim, dim, 8)?,
            ln1: nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: nn::linear(dim, hidden, vb.pp("ff1"))?,
            ff2: nn::linear(hidden, dim, vb.pp("ff2"))?,
        })
    }

    fn forward(&self, slots: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let attended = self.cross_attn.forward(&self.ln1.forward(slots)?, memory)?;
        let slots = (slots + attended)?;
        let ff = self.ff1.forward(&self.ln2.forward(&slots)?)?.gelu()?;
        (slots + self.ff2.forward(&ff)?).map_err(Into::into)
    }
}

impl DecoderConditioningAdapter {
    pub fn new(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
    ) -> Result<Self> {
        let compress_rate = std::env::var("TOFY_DECODER_ADAPTER_COMPRESS_RATE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        Self::new_with_compress_rate(vb, planner_dim, model_dim, num_output_slots, compress_rate)
    }

    pub fn new_with_compress_rate(
        vb: VarBuilder<'_>,
        planner_dim: usize,
        model_dim: usize,
        num_output_slots: usize,
        compress_rate: usize,
    ) -> Result<Self> {
        if compress_rate == 0 {
            anyhow::bail!("decoder adapter compress_rate must be non-zero");
        }
        let query_embed = nn::embedding(num_output_slots, planner_dim, vb.pp("query_embed"))?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), planner_dim, planner_dim, 8)?;
        let ln1 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(planner_dim, 1e-5, vb.pp("ln2"))?;
        let ff_hidden = (planner_dim * 4).max(256);
        let ff1 = nn::linear(planner_dim, ff_hidden, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_hidden, planner_dim, vb.pp("ff2"))?;
        let depth = std::env::var("TOFY_ADAPTER_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let mut extra_blocks = Vec::with_capacity(depth.saturating_sub(1));
        for index in 1..depth {
            extra_blocks.push(AdapterBlock::new(
                vb.pp(format!("blocks.{index}")),
                planner_dim,
            )?);
        }
        let index_proj = nn::linear(planner_dim, 1, vb.pp("index_proj"))?;
        let gate_proj = nn::linear(planner_dim, planner_dim, vb.pp("gate_proj"))?;
        let proj = nn::linear(planner_dim, model_dim, vb.pp("proj"))?;
        let output_norm = nn::rms_norm(model_dim, 1e-6, vb.pp("output_norm"))?;
        Ok(Self {
            query_embed,
            cross_attn,
            ln1,
            ln2,
            ff1,
            ff2,
            extra_blocks,
            index_proj,
            gate_proj,
            proj,
            output_norm,
            num_output_slots,
            planner_dim,
            compress_rate,
        })
    }

    pub fn compress_rate(&self) -> usize {
        self.compress_rate
    }

    /// Input: context slots [B, S, planner_dim]. Output: decoder conditioning slots [B, A, model_dim].
    pub fn forward(&self, context_slots: &Tensor) -> Result<Tensor> {
        let batch = context_slots.dim(0)?;
        let memory = compressed_context_compressor(
            context_slots,
            self.num_output_slots,
            self.compress_rate,
        )?;
        let salience = ops::softmax(&self.index_proj.forward(&memory)?, D::Minus2)?;
        let salience_memory = memory.broadcast_mul(&salience)?;
        let global_memory = salience_memory.sum(1)?;
        let memory_gate = self
            .gate_proj
            .forward(&global_memory)
            .and_then(|gate| ops::sigmoid(&gate))?
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

        let gated_queries = queries.broadcast_add(&global_memory.unsqueeze(1)?)?;
        let normed = self.ln1.forward(&gated_queries)?;
        let attended = self
            .cross_attn
            .forward(&normed, &memory)?
            .broadcast_mul(&memory_gate)?;
        let slots = (queries + attended)?;

        let normed = self.ln2.forward(&slots)?;
        let ff = self.ff1.forward(&normed)?.gelu()?;
        let mut slots = (slots + self.ff2.forward(&ff)?)?;
        for block in &self.extra_blocks {
            slots = block.forward(&slots, &memory)?;
        }

        Ok(self.output_norm.forward(&self.proj.forward(&slots)?)?)
    }
}

fn compressed_context_compressor(
    context_slots: &Tensor,
    output_slots: usize,
    compress_rate: usize,
) -> Result<Tensor> {
    let (_, slots, _) = context_slots.dims3()?;
    let recent = output_slots.clamp(1, slots.max(1)).min(slots);
    let recent_start = slots.saturating_sub(recent);
    let recent_memory = context_slots.narrow(1, recent_start, recent)?;
    if recent_start == 0 {
        return Ok(recent_memory);
    }

    let compress_rate = compress_rate.max(1);
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
