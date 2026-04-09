use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

use super::super::attention::{positional_encoding, LocalTransformerBlock, TransformerBlock};
use super::pooling::MultiQueryPool;

const DEFAULT_CHUNK_SIZE: usize = 16;
const DEFAULT_LOCAL_WINDOW: usize = 16;
const TARGET_NUM_CHUNKS: usize = 16;
const MAX_LOCAL_WINDOW: usize = 64;
const NUM_POOL_QUERIES: usize = 3;
const NUM_GLOBAL_TOKENS: usize = 4;

pub struct EncoderFeatures {
    pub token_states: Tensor,
    pub chunk_states: Tensor,
    pub global_states: Tensor,
    pub pooled_queries: Tensor,
}

pub struct PredictedEncoderFeatures {
    pub token_states: Tensor,
    pub chunk_states: Tensor,
    pub global_states: Tensor,
}

struct PredictorHead {
    ln: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl PredictorHead {
    fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        Ok(Self {
            ln: nn::layer_norm(dim, 1e-5, vb.pp("ln"))?,
            fc1: nn::linear(dim, dim * 4, vb.pp("fc1"))?,
            fc2: nn::linear(dim * 4, dim, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.ln.forward(x)?;
        let hidden = self.fc1.forward(&normed)?.gelu()?;
        Ok((x + self.fc2.forward(&hidden)?)?)
    }
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

    pub fn contrastive_summary(&self) -> Result<Tensor> {
        Ok(self.pooled_queries.narrow(1, 2, 1)?)
    }

    #[allow(dead_code)]
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
    token_predictor: PredictorHead,
    chunk_predictor: PredictorHead,
    global_predictor: PredictorHead,
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
        let token_predictor = PredictorHead::new(vb.pp("token_predictor"), dim)?;
        let chunk_predictor = PredictorHead::new(vb.pp("chunk_predictor"), dim)?;
        let global_predictor = PredictorHead::new(vb.pp("global_predictor"), dim)?;
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
            token_predictor,
            chunk_predictor,
            global_predictor,
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
        let pe =
            positional_encoding(seq_len, self.dim, x.device())?.to_dtype(token_states.dtype())?;
        token_states = token_states.broadcast_add(&pe)?;
        let chunk_size = self.effective_chunk_size(seq_len);
        let local_window = chunk_size.clamp(DEFAULT_LOCAL_WINDOW, MAX_LOCAL_WINDOW);
        for block in &self.local_blocks {
            token_states = block.forward_with_window(&token_states, local_window)?;
        }

        let (batch, _, dim) = token_states.dims3()?;
        let num_chunks = seq_len / chunk_size;
        let mut chunk_states = token_states
            .reshape((batch, num_chunks, chunk_size, dim))?
            .mean(2)?;
        let chunk_pe = positional_encoding(num_chunks, self.dim, x.device())?
            .to_dtype(chunk_states.dtype())?;
        chunk_states = chunk_states.broadcast_add(&chunk_pe)?;
        let global_ids: Vec<u32> = (0..NUM_GLOBAL_TOKENS as u32).collect();
        let global_ids = Tensor::from_vec(global_ids, (1, NUM_GLOBAL_TOKENS), x.device())?;
        let global_states = self
            .global_token_embed
            .forward(&global_ids)?
            .broadcast_as((batch, NUM_GLOBAL_TOKENS, dim))?;
        let mut global_seq = Tensor::cat(&[chunk_states, global_states], 1)?;
        for block in &self.global_blocks {
            global_seq = block.forward(&global_seq)?;
        }
        let chunk_states = global_seq.narrow(1, 0, num_chunks)?;
        let global_states = global_seq.narrow(1, num_chunks, NUM_GLOBAL_TOKENS)?;

        let chunk_context = chunk_states
            .unsqueeze(2)?
            .broadcast_as((batch, num_chunks, chunk_size, dim))?
            .reshape((batch, seq_len, dim))?;
        let token_context = self.token_context_proj.forward(&chunk_context)?;
        let token_states = self
            .token_ln_final
            .forward(&token_states.broadcast_add(&token_context)?)?;
        let chunk_states = self.chunk_ln_final.forward(&chunk_states)?;
        let global_states = self.global_ln_final.forward(&global_states)?;
        let pool_memory = Tensor::cat(
            &[
                token_states.clone(),
                chunk_states.clone(),
                global_states.clone(),
            ],
            1,
        )?;
        let pooled_queries = self.pool.forward(&pool_memory)?;

        Ok(EncoderFeatures {
            token_states,
            chunk_states,
            global_states,
            pooled_queries,
        })
    }

    pub(crate) fn predict_features(
        &self,
        features: &EncoderFeatures,
    ) -> Result<PredictedEncoderFeatures> {
        Ok(PredictedEncoderFeatures {
            token_states: self.token_predictor.forward(&features.token_states)?,
            chunk_states: self.chunk_predictor.forward(&features.chunk_states)?,
            global_states: self.global_predictor.forward(&features.global_states)?,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_features(x)?.token_states)
    }

    #[allow(dead_code)]
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.forward_features(x)?;
        Ok(features.contrastive_summary()?.squeeze(1)?)
    }
}
