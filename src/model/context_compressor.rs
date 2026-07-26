use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{self as nn, Module, VarBuilder};

use super::encoders::EncoderFeatures;
use super::latent_resampler::LatentResampler;

const ATTENTION_MASK_VALUE: f32 = -1.0e4;

/// Resamples encoder states into stable knowledge slots consumed by the Qwen adapter.
pub struct ContextCompressor {
    resampler: LatentResampler,
    proj: nn::Linear,
    output_norm: nn::RmsNorm,
    pool_proj: nn::Linear,
    memory_ln: nn::LayerNorm,
    memory_score: nn::Linear,
    memory_value: nn::Linear,
    memory_gate: nn::Linear,
    memory_importance: nn::Linear,
    num_slots: usize,
    in_dim: usize,
}

impl ContextCompressor {
    pub fn new(
        vb: VarBuilder<'_>,
        in_dim: usize,
        planner_dim: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let depth = std::env::var("TOFY_CONTEXT_COMPRESSOR_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let resampler = LatentResampler::new(
            vb.pp("resampler"),
            in_dim,
            resampler_heads(in_dim),
            num_slots,
            depth,
        )?;
        let proj = nn::linear(in_dim, planner_dim, vb.pp("proj"))?;
        let output_norm = nn::rms_norm(planner_dim, 1e-6, vb.pp("output_norm"))?;
        let pool_proj = nn::linear(planner_dim, 1, vb.pp("pool_proj"))?;
        let memory_ln = nn::layer_norm(in_dim, 1e-5, vb.pp("memory_ln"))?;
        let memory_score = nn::linear(in_dim, 1, vb.pp("memory_score"))?;
        let memory_value = nn::linear(in_dim, in_dim, vb.pp("memory_value"))?;
        let memory_gate = nn::linear(in_dim, in_dim, vb.pp("memory_gate"))?;
        let memory_importance = nn::linear(in_dim, 1, vb.pp("memory_importance"))?;
        Ok(Self {
            resampler,
            proj,
            output_norm,
            pool_proj,
            memory_ln,
            memory_score,
            memory_value,
            memory_gate,
            memory_importance,
            num_slots,
            in_dim,
        })
    }

    /// Input: encoder hidden states [B, T, in_dim]. Output: context slots [B, S, planner_dim].
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
        let slots = self.resampler.forward(&encoder_hidden, encoder_mask)?;
        Ok(self.output_norm.forward_diff(&self.proj.forward(&slots)?)?)
    }

    /// Hybrid long-context path for segmented memory.
    ///
    /// Recent memory stays exact, older memory is exposed through compressed
    /// bidirectional blocks, and the compressor's own slot queries retrieve a
    /// small set of query-adaptive old-memory summaries before the normal
    /// slot-to-memory cross-attention runs.
    pub fn forward_hybrid_masked(
        &self,
        encoder_hidden: &Tensor,
        encoder_mask: Option<&Tensor>,
        exact_tail: usize,
        block_size: usize,
        retrieval_slots: usize,
    ) -> Result<Tensor> {
        let encoder_hidden = encoder_hidden.contiguous()?;
        let (batch, memory_len, _) = encoder_hidden.dims3()?;
        if memory_len <= exact_tail.max(1) || block_size <= 1 {
            return self.forward_masked(&encoder_hidden, encoder_mask);
        }

        let exact_tail = exact_tail.min(memory_len).max(1);
        let old_len = memory_len.saturating_sub(exact_tail);
        if old_len == 0 {
            return self.forward_masked(&encoder_hidden, encoder_mask);
        }

        let mask = match encoder_mask {
            Some(mask) => mask.contiguous()?,
            None => Tensor::ones(
                (batch, memory_len),
                encoder_hidden.dtype(),
                encoder_hidden.device(),
            )?,
        };

        let learned_old = self.learned_old_memory(&encoder_hidden, &mask, old_len, block_size)?;
        let mut memory_parts = Vec::new();
        let mut mask_parts = Vec::new();

        let exact_old_tokens = old_len.min(
            std::env::var("TOFY_CONTEXT_EXACT_OLD_TOKENS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| retrieval_slots.saturating_mul(2).min(16)),
        );
        if exact_old_tokens > 0 {
            let exact_old = self.select_exact_old_memory(
                &encoder_hidden.narrow(1, 0, old_len)?,
                &mask.narrow(1, 0, old_len)?,
                exact_old_tokens,
            )?;
            memory_parts.push(exact_old.0);
            mask_parts.push(exact_old.1);
        }

        let blocks = learned_old.0;
        let block_mask = learned_old.1;
        memory_parts.push(blocks.clone());
        mask_parts.push(block_mask.clone());

        let retrieval_slots = retrieval_slots.min(self.num_slots).min(blocks.dim(1)?);
        if retrieval_slots > 0 {
            let queries = self
                .resampler
                .queries(batch, encoder_hidden.device())?
                .narrow(1, 0, retrieval_slots)?;
            let scores = queries.matmul(&blocks.transpose(D::Minus2, D::Minus1)?)?;
            let scores = (scores / (self.in_dim as f64).sqrt())?;
            let bias = block_mask
                .affine(-ATTENTION_MASK_VALUE as f64, ATTENTION_MASK_VALUE as f64)?
                .unsqueeze(1)?
                .to_dtype(scores.dtype())?;
            let weights = nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
            memory_parts.push(weights.matmul(&blocks)?);
            let retrieved_mask = block_mask
                .sum(1)?
                .clamp(0.0, 1.0)?
                .unsqueeze(1)?
                .broadcast_as((batch, retrieval_slots))?;
            mask_parts.push(retrieved_mask);
        }

        memory_parts.push(encoder_hidden.narrow(1, old_len, exact_tail)?);
        mask_parts.push(mask.narrow(1, old_len, exact_tail)?);

        let memory_refs = memory_parts.iter().collect::<Vec<_>>();
        let hybrid_memory = Tensor::cat(&memory_refs, 1)?;
        let mask_refs = mask_parts.iter().collect::<Vec<_>>();
        let hybrid_mask = Tensor::cat(&mask_refs, 1)?;
        self.forward_masked(&hybrid_memory, Some(&hybrid_mask))
    }

    fn learned_old_memory(
        &self,
        encoder_hidden: &Tensor,
        mask: &Tensor,
        old_len: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut block_tensors = Vec::new();
        let mut block_mask_tensors = Vec::new();
        let mut start = 0usize;
        while start < old_len {
            let len = (old_len - start).min(block_size);
            let block = encoder_hidden.narrow(1, start, len)?;
            let block_mask = mask.narrow(1, start, len)?;
            let weights = block_mask.unsqueeze(2)?.to_dtype(block.dtype())?;
            let denom = block_mask
                .sum(1)?
                .clamp(1.0, f64::INFINITY)?
                .unsqueeze(1)?
                .to_dtype(block.dtype())?;
            let summary = block
                .broadcast_mul(&weights)?
                .sum(1)?
                .broadcast_div(&denom)?;
            let normed = crate::util::layer_norm_diff(&self.memory_ln, &block)?;
            let score = self.memory_score.forward(&normed)?.squeeze(2)?;
            let score_bias = block_mask
                .affine(-ATTENTION_MASK_VALUE as f64, ATTENTION_MASK_VALUE as f64)?
                .to_dtype(score.dtype())?;
            let attn = nn::ops::softmax(&score.broadcast_add(&score_bias)?, 1)?;
            let value = self.memory_value.forward(&normed)?;
            let learned = value.broadcast_mul(&attn.unsqueeze(2)?)?.sum(1)?;
            let gate = self
                .memory_gate
                .forward(&summary)
                .and_then(|gate| nn::ops::sigmoid(&gate))?;
            let gated = learned
                .broadcast_mul(&gate)?
                .broadcast_add(&summary.broadcast_mul(&gate.affine(-1.0, 1.0)?)?)?;

            let valid = block_mask.sum(1)?.clamp(0.0, 1.0)?.unsqueeze(1)?;
            block_tensors.push(gated.broadcast_mul(&valid)?.unsqueeze(1)?);
            block_mask_tensors.push(valid);
            start += len;
        }

        if block_tensors.is_empty() {
            anyhow::bail!("learned old memory received no old blocks");
        }

        let block_refs = block_tensors.iter().collect::<Vec<_>>();
        let blocks = Tensor::cat(&block_refs, 1)?;
        let block_mask_refs = block_mask_tensors.iter().collect::<Vec<_>>();
        let block_mask = Tensor::cat(&block_mask_refs, 1)?;
        Ok((blocks, block_mask))
    }

    fn select_exact_old_memory(
        &self,
        old_memory: &Tensor,
        old_mask: &Tensor,
        keep: usize,
    ) -> Result<(Tensor, Tensor)> {
        let old_memory = old_memory.contiguous()?;
        let old_mask = old_mask.contiguous()?;
        let (batch, old_len, dim) = old_memory.dims3()?;
        let keep = keep.min(old_len).max(1);
        let normed = crate::util::layer_norm_diff(&self.memory_ln, &old_memory)?;
        let scores = self
            .memory_importance
            .forward(&normed)?
            .squeeze(2)?
            .to_dtype(candle_core::DType::F32)?;
        let mask_bias = old_mask
            .to_dtype(candle_core::DType::F32)?
            .affine(-ATTENTION_MASK_VALUE as f64, ATTENTION_MASK_VALUE as f64)?;
        let top_ids = scores
            .broadcast_add(&mask_bias)?
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, keep)?
            .contiguous()?;
        let chronological_order = top_ids.arg_sort_last_dim(true)?.contiguous()?;
        let top_ids = top_ids
            .gather(&chronological_order, D::Minus1)?
            .contiguous()?;
        let memory_idx = top_ids
            .unsqueeze(2)?
            .broadcast_as((batch, keep, dim))?
            .contiguous()?;
        let mask_idx = top_ids.contiguous()?;
        Ok((
            old_memory.gather(&memory_idx, 1)?,
            old_mask.gather(&mask_idx, 1)?,
        ))
    }

    #[allow(dead_code)]
    pub fn forward_encoder(&self, features: &EncoderFeatures) -> Result<Tensor> {
        self.forward(&features.memory()?)
    }

    pub fn pool(&self, context_slots: &Tensor) -> Result<Tensor> {
        let (batch, slots, dim) = context_slots.dims3()?;
        let scores = self
            .pool_proj
            .forward(context_slots)?
            .reshape((batch, slots))?;
        let weights = nn::ops::softmax(&scores, 1)?
            .unsqueeze(2)?
            .broadcast_as((batch, slots, dim))?;
        Ok(context_slots.broadcast_mul(&weights)?.sum(1)?)
    }
}

fn resampler_heads(dim: usize) -> usize {
    [16, 8, 4, 2]
        .into_iter()
        .find(|heads| dim.is_multiple_of(*heads))
        .unwrap_or(1)
}
