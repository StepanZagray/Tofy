use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::super::attention::{positional_encoding, LocalTransformerBlock, TransformerBlock};
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
    local_blocks: Vec<LocalTransformerBlock>,
    global_blocks: Vec<TransformerBlock>,
    global_token_embed: nn::Embedding,
    token_context_proj: nn::Linear,
    token_ln_final: nn::LayerNorm,
    chunk_ln_final: nn::LayerNorm,
    global_ln_final: nn::LayerNorm,
    pool: MultiQueryPool,
    dim: usize,
    chunk_size: usize,
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
        let num_local_layers = ((num_layers * 2) / 3).max(1);
        let num_global_layers = (num_layers - num_local_layers).max(1);

        let mut local_blocks = Vec::with_capacity(num_local_layers);
        for i in 0..num_local_layers {
            local_blocks.push(LocalTransformerBlock::new(
                vb.pp(format!("local_block_{}", i)),
                dim,
                num_heads,
                dim * 4,
                DEFAULT_LOCAL_WINDOW,
            )?);
        }

        let mut global_blocks = Vec::with_capacity(num_global_layers);
        for i in 0..num_global_layers {
            global_blocks.push(TransformerBlock::new(
                vb.pp(format!("global_block_{}", i)),
                dim,
                num_heads,
                dim * 4,
            )?);
        }

        let global_token_embed =
            nn::embedding(NUM_GLOBAL_TOKENS, dim, vb.pp("global_token_embed"))?;
        let token_context_proj = nn::linear(dim, dim, vb.pp("token_context_proj"))?;
        let token_ln_final = nn::layer_norm(dim, 1e-5, vb.pp("token_ln_final"))?;
        let chunk_ln_final = nn::layer_norm(dim, 1e-5, vb.pp("chunk_ln_final"))?;
        let global_ln_final = nn::layer_norm(dim, 1e-5, vb.pp("global_ln_final"))?;
        let pool = MultiQueryPool::new(vb.pp("pool"), dim, num_heads, NUM_POOL_QUERIES)?;
        Ok(Self {
            embed,
            local_blocks,
            global_blocks,
            global_token_embed,
            token_context_proj,
            token_ln_final,
            chunk_ln_final,
            global_ln_final,
            pool,
            dim,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }

    fn effective_chunk_size(&self, seq_len: usize) -> usize {
        let seq_len = seq_len.max(1);
        let min_chunk = self.chunk_size.min(seq_len);
        let target_chunk = seq_len.div_ceil(TARGET_NUM_CHUNKS).max(min_chunk);
        let mut chunk = target_chunk.min(seq_len);
        while !seq_len.is_multiple_of(chunk) && chunk < seq_len {
            chunk += 1;
        }
        chunk.max(1)
    }

    pub(crate) fn forward_features(&self, x: &Tensor) -> Result<EncoderFeatures> {
        let (_, seq_len) = x.dims2()?;
        let mut token_states = self.embed.forward(x)?;
        let token_mask = x.ne(PAD_TOKEN_ID)?.to_dtype(token_states.dtype())?;
        let token_mask_3d = token_mask.unsqueeze(2)?;
        let pe =
            positional_encoding(seq_len, self.dim, x.device())?.to_dtype(token_states.dtype())?;
        token_states = token_states.broadcast_add(&pe)?;
        token_states = token_states.broadcast_mul(&token_mask_3d)?;
        let chunk_size = self.effective_chunk_size(seq_len);
        let local_window = chunk_size.clamp(DEFAULT_LOCAL_WINDOW, MAX_LOCAL_WINDOW);
        for block in &self.local_blocks {
            token_states =
                block.forward_with_window_masked(&token_states, local_window, &token_mask)?;
            token_states = token_states.broadcast_mul(&token_mask_3d)?;
        }

        let (batch, _, dim) = token_states.dims3()?;
        let num_chunks = seq_len / chunk_size;
        let chunk_mask = token_mask.reshape((batch, num_chunks, chunk_size))?;
        let chunk_weights = chunk_mask.unsqueeze(3)?.to_dtype(token_states.dtype())?;
        let chunk_denom = chunk_mask
            .sum(2)?
            .clamp(1.0, f64::INFINITY)?
            .unsqueeze(2)?
            .to_dtype(token_states.dtype())?;
        let valid_chunks = chunk_mask.sum(2)?.clamp(0.0, 1.0)?;
        let mut chunk_states = token_states
            .reshape((batch, num_chunks, chunk_size, dim))?
            .broadcast_mul(&chunk_weights)?
            .sum(2)?
            .broadcast_div(&chunk_denom)?;
        let chunk_pe = positional_encoding(num_chunks, self.dim, x.device())?
            .to_dtype(chunk_states.dtype())?;
        chunk_states = chunk_states.broadcast_add(&chunk_pe)?;
        chunk_states = chunk_states.broadcast_mul(&valid_chunks.unsqueeze(2)?)?;
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

        let chunk_context = chunk_states
            .unsqueeze(2)?
            .broadcast_as((batch, num_chunks, chunk_size, dim))?
            .reshape((batch, seq_len, dim))?;
        let token_context = self.token_context_proj.forward(&chunk_context)?;
        let token_states = self
            .token_ln_final
            .forward(&token_states.broadcast_add(&token_context)?)?
            .broadcast_mul(&token_mask_3d)?;
        let chunk_states = self
            .chunk_ln_final
            .forward(&chunk_states)?
            .broadcast_mul(&valid_chunks.unsqueeze(2)?)?;
        let global_states = self.global_ln_final.forward(&global_states)?;
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
}
