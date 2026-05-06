//! Decoder with cross-attention to code world latent. Used for training (and optional inference)
//! when the decoder is conditioned on the world model's latent sequence instead of text.

use anyhow::Context;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};
use rand::RngExt;

use crate::model::attention::{
    positional_encoding, positional_encoding_from, AttentionKvCache, CrossAttention,
    MultiHeadAttention,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderKind {
    TextGeneralist,
    CodeSpecialist,
}

impl DecoderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextGeneralist => "text_generalist",
            Self::CodeSpecialist => "code_specialist",
        }
    }

    pub fn from_flag(flag: &str) -> Option<Self> {
        match flag.trim().to_ascii_lowercase().as_str() {
            "text" | "text_generalist" | "generalist" => Some(Self::TextGeneralist),
            "code" | "code_specialist" | "specialist" => Some(Self::CodeSpecialist),
            _ => None,
        }
    }

    fn id(self) -> u32 {
        match self {
            Self::TextGeneralist => 0,
            Self::CodeSpecialist => 1,
        }
    }

    fn local_window(self) -> usize {
        match self {
            Self::TextGeneralist => 96,
            Self::CodeSpecialist => 192,
        }
    }

    fn anchor_period(self) -> usize {
        match self {
            Self::TextGeneralist => 4,
            Self::CodeSpecialist => 3,
        }
    }
}

/// One decoder layer: causal self-attention + cross-attention to world latent + FFN.
struct CodeDecoderBlock {
    self_attn: MultiHeadAttention,
    cross_attn: CrossAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ln3: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    cross_gate: nn::Linear,
    adapter_down: nn::Linear,
    adapter_up: nn::Linear,
}

#[derive(Clone)]
pub struct DecoderGenerationState {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) self_kv_caches: Vec<Option<AttentionKvCache>>,
    pub(crate) cross_kv_caches: Vec<AttentionKvCache>,
    pub(crate) domain_state: Tensor,
    pub(crate) structure_state: Tensor,
    pub(crate) last_logits: Option<Tensor>,
}

impl CodeDecoderBlock {
    fn new(
        vb: VarBuilder<'_>,
        dim: usize,
        world_dim: usize,
        num_heads: usize,
        ff_dim: usize,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(vb.pp("self_attn"), dim, num_heads)?;
        let cross_attn = CrossAttention::new(vb.pp("cross_attn"), dim, world_dim, num_heads)?;
        let ln1 = nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?;
        let ln3 = nn::layer_norm(dim, 1e-5, vb.pp("ln3"))?;
        let ff1 = nn::linear(dim, ff_dim, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_dim, dim, vb.pp("ff2"))?;
        let cross_gate = nn::linear(dim, dim, vb.pp("cross_gate"))?;
        let adapter_rank = (dim / 4).max(64);
        let adapter_down = nn::linear(dim, adapter_rank, vb.pp("adapter_down"))?;
        let adapter_up = nn::linear(adapter_rank, dim, vb.pp("adapter_up"))?;
        Ok(Self {
            self_attn,
            cross_attn,
            ln1,
            ln2,
            ln3,
            ff1,
            ff2,
            cross_gate,
            adapter_down,
            adapter_up,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        world_latent: &Tensor,
        domain_state: &Tensor,
        use_full_attention: bool,
        local_window: usize,
    ) -> Result<Tensor> {
        let normed = self.ln1.forward(x)?;
        let self_out = if use_full_attention {
            self.self_attn.forward_causal(&normed)?
        } else {
            self.self_attn.forward_causal_local(&normed, local_window)?
        };
        let x = (x + self_out)?;

        let normed = self.ln2.forward(&x)?;
        let cross_gate_in = normed.broadcast_add(domain_state)?;
        let cross_gate = self
            .cross_gate
            .forward(&cross_gate_in)?
            .relu()?
            .clamp(0.0, 1.0)?;
        let cross_out = self
            .cross_attn
            .forward(&normed, world_latent)?
            .broadcast_mul(&cross_gate)?;
        let x = (x + cross_out)?;

        let normed = self.ln3.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let adapter_in = normed.broadcast_add(domain_state)?;
        let adapter = self
            .adapter_up
            .forward(&self.adapter_down.forward(&adapter_in)?.gelu()?)?;
        let ff_out = ff_out.broadcast_add(&adapter.affine(0.5, 0.0)?)?;
        Ok((x + ff_out)?)
    }

    fn precompute_cross_kv(&self, world_latent: &Tensor) -> Result<AttentionKvCache> {
        self.cross_attn.project_kv(world_latent)
    }

    fn forward_prefill(
        &self,
        x: &Tensor,
        cross_kv_cache: &AttentionKvCache,
        domain_state: &Tensor,
        use_full_attention: bool,
        local_window: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let normed = self.ln1.forward(x)?;
        let self_kv_cache = self.self_attn.project_self_kv(&normed)?;
        let self_out = if use_full_attention {
            self.self_attn.forward_causal(&normed)?
        } else {
            self.self_attn.forward_causal_local(&normed, local_window)?
        };
        let x = (x + self_out)?;

        let normed = self.ln2.forward(&x)?;
        let cross_gate_in = normed.broadcast_add(domain_state)?;
        let cross_gate = self
            .cross_gate
            .forward(&cross_gate_in)?
            .relu()?
            .clamp(0.0, 1.0)?;
        let cross_out = self
            .cross_attn
            .forward_precomputed(&normed, cross_kv_cache, None)?
            .broadcast_mul(&cross_gate)?;
        let x = (x + cross_out)?;

        let normed = self.ln3.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let adapter_in = normed.broadcast_add(domain_state)?;
        let adapter = self
            .adapter_up
            .forward(&self.adapter_down.forward(&adapter_in)?.gelu()?)?;
        let ff_out = ff_out.broadcast_add(&adapter.affine(0.5, 0.0)?)?;
        Ok(((x + ff_out)?, self_kv_cache))
    }

    fn forward_incremental(
        &self,
        x: &Tensor,
        cross_kv_cache: &AttentionKvCache,
        domain_state: &Tensor,
        use_full_attention: bool,
        local_window: usize,
        self_kv_cache: Option<&AttentionKvCache>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let normed = self.ln1.forward(x)?;
        let (self_out, next_self_kv_cache) = if use_full_attention {
            self.self_attn
                .forward_causal_incremental(&normed, self_kv_cache)?
        } else {
            self.self_attn
                .forward_causal_local_incremental(&normed, self_kv_cache, local_window)?
        };
        let x = (x + self_out)?;

        let normed = self.ln2.forward(&x)?;
        let cross_gate_in = normed.broadcast_add(domain_state)?;
        let cross_gate = self
            .cross_gate
            .forward(&cross_gate_in)?
            .relu()?
            .clamp(0.0, 1.0)?;
        let cross_out = self
            .cross_attn
            .forward_precomputed(&normed, cross_kv_cache, None)?
            .broadcast_mul(&cross_gate)?;
        let x = (x + cross_out)?;

        let normed = self.ln3.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let adapter_in = normed.broadcast_add(domain_state)?;
        let adapter = self
            .adapter_up
            .forward(&self.adapter_down.forward(&adapter_in)?.gelu()?)?;
        let ff_out = ff_out.broadcast_add(&adapter.affine(0.5, 0.0)?)?;
        Ok(((x + ff_out)?, next_self_kv_cache))
    }
}

/// Decoder-only transformer conditioned on world latent via cross-attention.
/// Input: token ids [B, T]. World latent: [B, T_world, world_dim].
/// Output: logits [B, T, vocab_size].
pub struct CodeDecoder {
    embed: nn::Embedding,
    kind_embed: nn::Embedding,
    structure_proj: nn::Linear,
    blocks: Vec<CodeDecoderBlock>,
    ln_final: nn::LayerNorm,
    lm_head: nn::Linear,
    dim: usize,
    kind: DecoderKind,
}

impl CodeDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder<'_>,
        vocab_size: usize,
        dim: usize,
        world_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
        kind: DecoderKind,
    ) -> Result<Self> {
        let embed = nn::embedding(vocab_size, dim, vb.pp("embed"))?;
        let kind_embed = nn::embedding(2, dim, vb.pp("kind_embed"))?;
        let structure_proj = nn::linear(world_dim, dim, vb.pp("structure_proj"))?;
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = CodeDecoderBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                world_dim,
                num_heads,
                ff_dim,
            )?;
            blocks.push(block);
        }
        let ln_final = nn::layer_norm(dim, 1e-5, vb.pp("ln_final"))?;
        let lm_head = nn::linear(dim, vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed,
            kind_embed,
            structure_proj,
            blocks,
            ln_final,
            lm_head,
            dim,
            kind,
        })
    }

    /// input_ids: [B, T], world_latent: [B, T_world, world_dim]
    /// Returns logits [B, T, vocab_size].
    pub fn forward(&self, input_ids: &Tensor, world_latent: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let mut h = self.embed.forward(input_ids)?;
        let pe = positional_encoding(t, self.dim, input_ids.device())?.to_dtype(h.dtype())?;
        h = h.broadcast_add(&pe)?;
        let domain_state = self.domain_state(input_ids.device(), b, t)?;
        let structure_state = self.structure_state(world_latent, b, t)?;
        h = h.broadcast_add(&domain_state)?;
        h = h.broadcast_add(&structure_state)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let use_full_attention =
                layer_idx % self.kind.anchor_period() == 0 || layer_idx + 1 == self.blocks.len();
            h = block.forward(
                &h,
                world_latent,
                &domain_state.broadcast_add(&structure_state)?,
                use_full_attention,
                self.kind.local_window(),
            )?;
        }
        h = self.ln_final.forward(&h)?;
        self.lm_head
            .forward(&h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    fn domain_state(&self, device: &Device, batch: usize, seq_len: usize) -> Result<Tensor> {
        let kind_ids = Tensor::from_vec(vec![self.kind.id(); batch], (batch,), device)?;
        self.kind_embed
            .forward(&kind_ids)?
            .unsqueeze(1)?
            .broadcast_as((batch, seq_len, self.dim))
            .map_err(Into::into)
    }

    fn structure_state(
        &self,
        world_latent: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let slots = world_latent.dim(1)?.max(1);
        let pooled = world_latent.sum(1)?.affine(1.0 / slots as f64, 0.0)?;
        self.structure_proj
            .forward(&pooled)?
            .tanh()?
            .unsqueeze(1)?
            .broadcast_as((batch, seq_len, self.dim))
            .map_err(Into::into)
    }

    fn precompute_cross_kv_caches(&self, world_latent: &Tensor) -> Result<Vec<AttentionKvCache>> {
        self.blocks
            .iter()
            .map(|block| block.precompute_cross_kv(world_latent))
            .collect()
    }

    fn decode_token(
        &self,
        device: &Device,
        token_id: u32,
        position: usize,
        state: &mut DecoderGenerationState,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token_id], (1, 1), device)?;
        let mut h = self.embed.forward(&input)?;
        let pe = positional_encoding_from(position, 1, self.dim, device)?.to_dtype(h.dtype())?;
        h = h.broadcast_add(&pe)?;
        h = h.broadcast_add(&state.domain_state)?;
        h = h.broadcast_add(&state.structure_state)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let use_full_attention =
                layer_idx % self.kind.anchor_period() == 0 || layer_idx + 1 == self.blocks.len();
            let (next_h, next_cache) = block.forward_incremental(
                &h,
                &state.cross_kv_caches[layer_idx],
                &state.domain_state.broadcast_add(&state.structure_state)?,
                use_full_attention,
                self.kind.local_window(),
                state.self_kv_caches[layer_idx].as_ref(),
            )?;
            h = next_h;
            state.self_kv_caches[layer_idx] = Some(next_cache);
        }
        h = self.ln_final.forward(&h)?;
        self.lm_head
            .forward(&h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    pub fn begin_generation(
        &self,
        device: &Device,
        prompt_ids: &[u32],
        world_latent: &Tensor,
    ) -> Result<DecoderGenerationState> {
        if prompt_ids.is_empty() {
            return Ok(DecoderGenerationState {
                token_ids: Vec::new(),
                self_kv_caches: vec![None; self.blocks.len()],
                cross_kv_caches: self.precompute_cross_kv_caches(world_latent)?,
                domain_state: self.domain_state(device, 1, 1)?,
                structure_state: self.structure_state(world_latent, 1, 1)?,
                last_logits: None,
            });
        }
        let prompt_len = prompt_ids.len();
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_len), device)?;
        let mut h = self.embed.forward(&input)?;
        let pe = positional_encoding(prompt_len, self.dim, device)?.to_dtype(h.dtype())?;
        h = h.broadcast_add(&pe)?;
        let domain_state_full = self.domain_state(device, 1, prompt_len)?;
        let structure_state_full = self.structure_state(world_latent, 1, prompt_len)?;
        h = h.broadcast_add(&domain_state_full)?;
        h = h.broadcast_add(&structure_state_full)?;
        let cross_kv_caches = self.precompute_cross_kv_caches(world_latent)?;
        let mut self_kv_caches = Vec::with_capacity(self.blocks.len());

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let use_full_attention =
                layer_idx % self.kind.anchor_period() == 0 || layer_idx + 1 == self.blocks.len();
            let (next_h, self_kv_cache) = block.forward_prefill(
                &h,
                &cross_kv_caches[layer_idx],
                &domain_state_full.broadcast_add(&structure_state_full)?,
                use_full_attention,
                self.kind.local_window(),
            )?;
            h = next_h;
            self_kv_caches.push(Some(self_kv_cache));
        }
        h = self.ln_final.forward(&h)?;
        let logits = self
            .lm_head
            .forward(&h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(DecoderGenerationState {
            token_ids: prompt_ids.to_vec(),
            self_kv_caches,
            cross_kv_caches,
            domain_state: self.domain_state(device, 1, 1)?,
            structure_state: self.structure_state(world_latent, 1, 1)?,
            last_logits: Some(logits.narrow(1, prompt_len - 1, 1)?),
        })
    }

    pub fn sample_from_last_logits(
        &self,
        state: &DecoderGenerationState,
        temperature: f32,
    ) -> Result<u32> {
        let logits = self.last_token_logits(state)?;
        let next_id = if temperature <= 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            let scaled = logits
                .iter()
                .map(|&logit| logit / temperature.max(1e-5))
                .collect::<Vec<_>>();
            let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scaled
                .iter()
                .map(|&logit| (logit - max_logit).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>().max(1e-8);
            let mut rng = rand::rng();
            let r: f32 = rng.random();
            let mut cum = 0.0f32;
            let mut chosen = 0u32;
            for (i, &w) in weights.iter().enumerate() {
                cum += w / total;
                if r <= cum {
                    chosen = i as u32;
                    break;
                }
            }
            chosen
        };
        Ok(next_id)
    }

    pub fn last_token_logits(&self, state: &DecoderGenerationState) -> Result<Vec<f32>> {
        state
            .last_logits
            .as_ref()
            .context("generation state has no last logits")?
            .i((0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1()
            .map_err(Into::into)
    }

    pub fn step_generation(
        &self,
        device: &Device,
        state: &mut DecoderGenerationState,
        next_input_id: u32,
    ) -> Result<()> {
        let logits = self.decode_token(device, next_input_id, state.token_ids.len(), state)?;
        state.token_ids.push(next_input_id);
        state.last_logits = Some(logits);
        Ok(())
    }

    /// Autoregressive generation conditioned on world latent.
    /// - prompt_ids: initial token ids (context).
    /// - world_latent: [1, T_world, world_dim].
    /// - temperature: 0.0 = argmax, >0 = sampling.
    /// - stop_at: if Some(id), stop generation when this token is produced (e.g. pad/EOS).
    ///
    /// Returns only the newly generated token ids (length <= max_new_tokens).
    #[allow(dead_code)]
    pub fn generate(
        &self,
        device: &Device,
        prompt_ids: &[u32],
        world_latent: &Tensor,
        max_new_tokens: usize,
        temperature: f32,
        stop_at: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut state = self.begin_generation(device, prompt_ids, world_latent)?;
        let mut generated = Vec::new();

        for _ in 0..max_new_tokens {
            let next_id = self.sample_from_last_logits(&state, temperature)?;
            if stop_at == Some(next_id) {
                break;
            }
            generated.push(next_id);
            self.step_generation(device, &mut state, next_id)?;
        }

        Ok(generated)
    }

    /// Single step: given current token ids, return next token id (for streaming).
    #[allow(dead_code)]
    pub fn step(
        &self,
        device: &Device,
        ids: &[u32],
        world_latent: &Tensor,
        temperature: f32,
    ) -> Result<u32> {
        if ids.is_empty() {
            return Ok(0);
        }
        let state = self.begin_generation(device, ids, world_latent)?;
        self.sample_from_last_logits(&state, temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn test_world_latent(device: &Device, seq: usize, dim: usize) -> Result<Tensor> {
        let values = (0..seq * dim)
            .map(|i| ((i % 29) as f32 - 14.0) / 15.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, seq, dim), device).map_err(Into::into)
    }

    #[test]
    fn incremental_prefill_matches_full_forward_last_logits() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let decoder = CodeDecoder::new(
            vb.pp("decoder"),
            64,
            16,
            12,
            3,
            4,
            32,
            DecoderKind::CodeSpecialist,
        )?;
        let input = Tensor::from_vec(vec![3u32, 5, 8, 13, 21], (1, 5), &device)?;
        let world_latent = test_world_latent(&device, 6, 12)?;
        let full = decoder.forward(&input, &world_latent)?;
        let full_last = full.i((0, 4, ..))?;

        let state = decoder.begin_generation(&device, &[3, 5, 8, 13, 21], &world_latent)?;
        let cached_last = state
            .last_logits
            .as_ref()
            .context("missing cached logits")?
            .i((0, 0, ..))?;

        let max_diff =
            crate::util::scalar_f32(&full_last.broadcast_sub(&cached_last)?.abs()?.max_all()?)?;
        assert!(max_diff < 1e-4, "incremental logits mismatch: {max_diff}");
        Ok(())
    }
}
