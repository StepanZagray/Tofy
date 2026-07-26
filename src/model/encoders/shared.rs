use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::super::attention::{CrossAttention, LocalTransformerBlock, TransformerBlock};
use super::pooling::MultiQueryPool;

const DEFAULT_CHUNK_SIZE: usize = 16;
const DEFAULT_LOCAL_WINDOW: usize = 16;
const TARGET_NUM_CHUNKS: usize = 16;
const MAX_LOCAL_WINDOW: usize = 64;
const NUM_POOL_QUERIES: usize = 2;
const NUM_GLOBAL_TOKENS: usize = 4;
const PAD_TOKEN_ID: u32 = 0;

pub struct EncoderFeatures {
    pub token_states: Tensor,
    pub chunk_states: Tensor,
    pub global_states: Tensor,
    pub pooled_queries: Tensor,
}

impl EncoderFeatures {
    pub fn detached(&self) -> Self {
        Self {
            token_states: self.token_states.detach(),
            chunk_states: self.chunk_states.detach(),
            global_states: self.global_states.detach(),
            pooled_queries: self.pooled_queries.detach(),
        }
    }

    pub fn planner_summary(&self) -> Result<Tensor> {
        Ok(self.pooled_queries.narrow(1, 0, 1)?)
    }

    pub fn routing_summary(&self) -> Result<Tensor> {
        Ok(self.pooled_queries.narrow(1, 1, 1)?)
    }

    pub fn memory(&self) -> Result<Tensor> {
        let planner = self.planner_summary()?;
        let routing = self.routing_summary()?;
        Tensor::cat(
            &[
                self.token_states.clone(),
                self.chunk_states.clone(),
                self.global_states.clone(),
                planner,
                routing,
            ],
            1,
        )
        .map_err(Into::into)
    }
}

pub(crate) struct EncoderBackbone {
    embed: nn::Embedding,
    pre_local_blocks: Vec<LocalTransformerBlock>,
    global_blocks: Vec<TransformerBlock>,
    refine_local_blocks: Vec<LocalTransformerBlock>,
    global_token_embed: nn::Embedding,
    chunk_score_norm: nn::RmsNorm,
    chunk_score: nn::Linear,
    feedback_query_norm: nn::RmsNorm,
    feedback_memory_norm: nn::RmsNorm,
    feedback: CrossAttention,
    chunk_merge: nn::Linear,
    token_ln_final: nn::RmsNorm,
    chunk_ln_final: nn::RmsNorm,
    global_ln_final: nn::RmsNorm,
    token_projection: nn::Linear,
    chunk_projection: nn::Linear,
    global_projection: nn::Linear,
    predictor_norm: nn::RmsNorm,
    predictor_gate: nn::Linear,
    predictor_up: nn::Linear,
    predictor_down: nn::Linear,
    pool: MultiQueryPool,
    chunk_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EncoderDepthPlan {
    pre_local: usize,
    global: usize,
    refine_local: usize,
}

impl EncoderDepthPlan {
    fn new(total: usize) -> Result<Self> {
        if total == 0 {
            anyhow::bail!("encoder must contain at least one transformer block");
        }
        let global = if total >= 2 { (total / 3).max(1) } else { 0 };
        let refine_local = if total >= 3 { (total / 3).max(1) } else { 0 };
        let pre_local = total - global - refine_local;
        debug_assert_eq!(pre_local + global + refine_local, total);
        Ok(Self {
            pre_local,
            global,
            refine_local,
        })
    }
}

impl EncoderBackbone {
    pub(crate) fn new(
        vb: VarBuilder<'_>,
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let embed = nn::embedding(vocab_size, dim, vb.pp("embed"))?;
        let plan = EncoderDepthPlan::new(num_layers)?;

        let mut pre_local_blocks = Vec::with_capacity(plan.pre_local);
        for i in 0..plan.pre_local {
            pre_local_blocks.push(LocalTransformerBlock::new(
                vb.pp(format!("pre_local_block_{i}")),
                dim,
                num_heads,
                dim * 4,
                DEFAULT_LOCAL_WINDOW,
            )?);
        }

        let mut global_blocks = Vec::with_capacity(plan.global);
        for i in 0..plan.global {
            global_blocks.push(TransformerBlock::new(
                vb.pp(format!("global_block_{i}")),
                dim,
                num_heads,
                dim * 4,
            )?);
        }
        let mut refine_local_blocks = Vec::with_capacity(plan.refine_local);
        for i in 0..plan.refine_local {
            refine_local_blocks.push(LocalTransformerBlock::new(
                vb.pp(format!("refine_local_block_{i}")),
                dim,
                num_heads,
                dim * 4,
                DEFAULT_LOCAL_WINDOW,
            )?);
        }

        let global_token_embed =
            nn::embedding(NUM_GLOBAL_TOKENS, dim, vb.pp("global_token_embed"))?;
        let chunk_score_norm = nn::rms_norm(dim, 1e-6, vb.pp("chunk_score_norm"))?;
        let chunk_score = nn::linear_no_bias(dim, 1, vb.pp("chunk_score"))?;
        let feedback_query_norm = nn::rms_norm(dim, 1e-6, vb.pp("feedback_query_norm"))?;
        let feedback_memory_norm = nn::rms_norm(dim, 1e-6, vb.pp("feedback_memory_norm"))?;
        let feedback = CrossAttention::new(vb.pp("feedback"), dim, dim, num_heads)?;
        let chunk_merge = nn::linear(dim * 2, dim, vb.pp("chunk_merge"))?;
        let token_ln_final = nn::rms_norm(dim, 1e-6, vb.pp("token_ln_final"))?;
        let chunk_ln_final = nn::rms_norm(dim, 1e-6, vb.pp("chunk_ln_final"))?;
        let global_ln_final = nn::rms_norm(dim, 1e-6, vb.pp("global_ln_final"))?;
        // LeWorldModel projects after the encoder's final normalization so
        // SIGReg can control latent scale and covariance.
        let token_projection = nn::linear(dim, dim, vb.pp("token_projection"))?;
        let chunk_projection = nn::linear(dim, dim, vb.pp("chunk_projection"))?;
        let global_projection = nn::linear(dim, dim, vb.pp("global_projection"))?;
        let predictor_dim = ((dim * 8) / 3).max(dim);
        let predictor_norm = nn::rms_norm(dim, 1e-6, vb.pp("predictor_norm"))?;
        let predictor_gate = nn::linear_no_bias(dim, predictor_dim, vb.pp("predictor_gate"))?;
        let predictor_up = nn::linear_no_bias(dim, predictor_dim, vb.pp("predictor_up"))?;
        let predictor_down = nn::linear_no_bias(predictor_dim, dim, vb.pp("predictor_down"))?;
        let pool = MultiQueryPool::new(vb.pp("pool"), dim, num_heads, NUM_POOL_QUERIES)?;
        Ok(Self {
            embed,
            pre_local_blocks,
            global_blocks,
            refine_local_blocks,
            global_token_embed,
            chunk_score_norm,
            chunk_score,
            feedback_query_norm,
            feedback_memory_norm,
            feedback,
            chunk_merge,
            token_ln_final,
            chunk_ln_final,
            global_ln_final,
            token_projection,
            chunk_projection,
            global_projection,
            predictor_norm,
            predictor_gate,
            predictor_up,
            predictor_down,
            pool,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }

    pub(crate) fn chunk_size_for_seq_len(&self, seq_len: usize) -> usize {
        self.effective_chunk_size(seq_len)
    }

    pub(crate) fn attention_work_summary(&self, seq_len: usize) -> String {
        let chunk_size = self.effective_chunk_size(seq_len);
        let local_window = chunk_size.clamp(DEFAULT_LOCAL_WINDOW, MAX_LOCAL_WINDOW);
        let local_keys = seq_len.min(local_window.saturating_mul(2).saturating_sub(1));
        let local_pairs_per_layer = seq_len.saturating_mul(local_keys);
        let chunks = seq_len.div_ceil(chunk_size);
        let global_positions = chunks + NUM_GLOBAL_TOKENS;
        let pre_local_pairs = self
            .pre_local_blocks
            .len()
            .saturating_mul(local_pairs_per_layer);
        let global_pairs = self
            .global_blocks
            .len()
            .saturating_mul(global_positions.saturating_mul(global_positions));
        let feedback_pairs = seq_len.saturating_mul(global_positions);
        let refine_pairs = self
            .refine_local_blocks
            .len()
            .saturating_mul(local_pairs_per_layer);
        format!(
            "Encoder depth parameters: pre-local/global/refine={}/{}/{} blocks; attention-score work upper bound at seq={seq_len}: pre-local={pre_local_pairs} global={global_pairs} feedback={feedback_pairs} refine={refine_pairs} pairs",
            self.pre_local_blocks.len(),
            self.global_blocks.len(),
            self.refine_local_blocks.len(),
        )
    }

    pub(crate) fn predict_states(&self, states: &Tensor) -> Result<Tensor> {
        let normed = self.predictor_norm.forward_diff(states)?;
        let hidden = self
            .predictor_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.predictor_up.forward(&normed)?)?;
        self.predictor_down.forward(&hidden).map_err(Into::into)
    }

    fn effective_chunk_size(&self, seq_len: usize) -> usize {
        let seq_len = seq_len.max(1);
        let min_chunk = self.chunk_size.min(seq_len);
        let target_chunk = seq_len.div_ceil(TARGET_NUM_CHUNKS).max(min_chunk);
        target_chunk.min(seq_len).max(1)
    }

    fn pool_chunks(
        &self,
        token_states: &Tensor,
        token_mask: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, dim) = token_states.dims3()?;
        let num_chunks = seq_len.div_ceil(chunk_size);
        let padded_len = num_chunks * chunk_size;
        let (source, padded_mask) = if padded_len > seq_len {
            let pad_len = padded_len - seq_len;
            let state_pad = Tensor::zeros(
                (batch, pad_len, dim),
                token_states.dtype(),
                token_states.device(),
            )?;
            let mask_pad =
                Tensor::zeros((batch, pad_len), token_mask.dtype(), token_states.device())?;
            (
                Tensor::cat(&[token_states.clone(), state_pad], 1)?,
                Tensor::cat(&[token_mask.clone(), mask_pad], 1)?,
            )
        } else {
            (token_states.clone(), token_mask.clone())
        };
        let source = source.reshape((batch, num_chunks, chunk_size, dim))?;
        let chunk_mask = padded_mask.reshape((batch, num_chunks, chunk_size))?;
        let score = self
            .chunk_score
            .forward(&self.chunk_score_norm.forward_diff(&source)?)?
            .squeeze(3)?;
        let bias = chunk_mask.to_dtype(score.dtype())?.affine(1.0e4, -1.0e4)?;
        let weights = nn::ops::softmax(&score.broadcast_add(&bias)?, 2)?.unsqueeze(3)?;
        let valid_chunks = chunk_mask.sum(2)?.clamp(0.0, 1.0)?;
        let chunks = source
            .broadcast_mul(&weights)?
            .sum(2)?
            .broadcast_mul(&valid_chunks.unsqueeze(2)?)?;
        Ok((chunks, valid_chunks))
    }

    pub(crate) fn forward_features(&self, x: &Tensor) -> Result<EncoderFeatures> {
        let (_, seq_len) = x.dims2()?;
        let mut token_states = self.embed.forward(x)?;
        let token_mask = x.ne(PAD_TOKEN_ID)?.to_dtype(token_states.dtype())?;
        let token_mask_3d = token_mask.unsqueeze(2)?;
        token_states = token_states.broadcast_mul(&token_mask_3d)?;
        let chunk_size = self.effective_chunk_size(seq_len);
        let local_window = chunk_size.clamp(DEFAULT_LOCAL_WINDOW, MAX_LOCAL_WINDOW);
        for block in &self.pre_local_blocks {
            token_states =
                block.forward_with_window_masked(&token_states, local_window, &token_mask)?;
            token_states = token_states.broadcast_mul(&token_mask_3d)?;
        }

        let (batch, _, dim) = token_states.dims3()?;
        let num_chunks = seq_len.div_ceil(chunk_size);
        let (chunk_states, valid_chunks) =
            self.pool_chunks(&token_states, &token_mask, chunk_size)?;
        let global_ids: Vec<u32> = (0..NUM_GLOBAL_TOKENS as u32).collect();
        let global_ids = Tensor::from_vec(global_ids, (1, NUM_GLOBAL_TOKENS), x.device())?;
        let global_states = self
            .global_token_embed
            .forward(&global_ids)?
            .broadcast_as((batch, NUM_GLOBAL_TOKENS, dim))?;
        let mut global_seq = Tensor::cat(&[chunk_states, global_states], 1)?;
        let global_token_mask =
            Tensor::ones((batch, NUM_GLOBAL_TOKENS), token_mask.dtype(), x.device())?;
        let global_seq_mask = Tensor::cat(&[&valid_chunks, &global_token_mask], 1)?;
        let global_seq_mask_3d = global_seq_mask.unsqueeze(2)?;
        for block in &self.global_blocks {
            global_seq = block.forward_masked(&global_seq, &global_seq_mask)?;
            global_seq = global_seq.broadcast_mul(&global_seq_mask_3d)?;
        }
        let chunk_states = global_seq
            .narrow(1, 0, num_chunks)?
            .broadcast_mul(&valid_chunks.unsqueeze(2)?)?;
        let global_states = global_seq.narrow(1, num_chunks, NUM_GLOBAL_TOKENS)?;

        // Every token can now read all updated chunks and global registers,
        // rather than receiving only a repeated copy of its own chunk.
        let feedback_memory = Tensor::cat(&[&chunk_states, &global_states], 1)?;
        let feedback_mask = Tensor::cat(&[&valid_chunks, &global_token_mask], 1)?;
        let token_context = self.feedback.forward_masked(
            &self.feedback_query_norm.forward_diff(&token_states)?,
            &self.feedback_memory_norm.forward_diff(&feedback_memory)?,
            Some(&feedback_mask),
        )?;
        token_states = token_states.broadcast_add(&token_context)?;
        token_states = token_states.broadcast_mul(&token_mask_3d)?;
        for block in &self.refine_local_blocks {
            token_states =
                block.forward_with_window_masked(&token_states, local_window, &token_mask)?;
            token_states = token_states.broadcast_mul(&token_mask_3d)?;
        }
        let (refined_chunks, _) = self.pool_chunks(&token_states, &token_mask, chunk_size)?;
        let chunk_states = self
            .chunk_merge
            .forward(&Tensor::cat(&[chunk_states, refined_chunks], 2)?)?
            .broadcast_mul(&valid_chunks.unsqueeze(2)?)?;

        let token_states = self
            .token_projection
            .forward(&self.token_ln_final.forward_diff(&token_states)?)?
            .broadcast_mul(&token_mask_3d)?;
        let chunk_states = self
            .chunk_projection
            .forward(&self.chunk_ln_final.forward_diff(&chunk_states)?)?
            .broadcast_mul(&valid_chunks.unsqueeze(2)?)?;
        let global_states = self
            .global_projection
            .forward(&self.global_ln_final.forward_diff(&global_states)?)?;
        let pool_memory = Tensor::cat(
            &[
                token_states.clone(),
                chunk_states.clone(),
                global_states.clone(),
            ],
            1,
        )?;
        let pool_mask = Tensor::cat(&[&token_mask, &valid_chunks, &global_token_mask], 1)?;
        let pooled_queries = self.pool.forward_masked(&pool_memory, Some(&pool_mask))?;

        Ok(EncoderFeatures {
            token_states,
            chunk_states,
            global_states,
            pooled_queries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn encoder_zeroes_padded_token_outputs() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let encoder = EncoderBackbone::new(
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
            16,
            8,
            2,
            2,
        )?;
        let input = Tensor::from_vec(vec![3u32, 4, PAD_TOKEN_ID, PAD_TOKEN_ID], (1, 4), &device)?;

        let features = encoder.forward_features(&input)?;
        let rows = features.token_states.to_vec3::<f32>()?;

        for value in rows[0][2].iter().chain(rows[0][3].iter()) {
            assert!(value.abs() < 1e-5, "padded token output was {value}");
        }
        Ok(())
    }

    #[test]
    fn encoder_keeps_chunk_resolution_for_prime_sequence_lengths() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let encoder = EncoderBackbone::new(
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
            128,
            8,
            2,
            2,
        )?;
        let ids = (1u32..=67).collect::<Vec<_>>();
        let input = Tensor::from_vec(ids, (1, 67), &device)?;

        let features = encoder.forward_features(&input)?;

        assert!(
            features.chunk_states.dim(1)? > 1,
            "prime-length inputs should not collapse to one chunk"
        );
        Ok(())
    }

    #[test]
    fn depth_plan_is_exact_and_reserves_post_global_refinement() -> Result<()> {
        assert_eq!(
            EncoderDepthPlan::new(7)?,
            EncoderDepthPlan {
                pre_local: 3,
                global: 2,
                refine_local: 2,
            }
        );
        for depth in 1..16 {
            let plan = EncoderDepthPlan::new(depth)?;
            assert_eq!(plan.pre_local + plan.global + plan.refine_local, depth);
            if depth >= 3 {
                assert!(plan.global > 0);
                assert!(plan.refine_local > 0);
            }
        }
        Ok(())
    }
}
