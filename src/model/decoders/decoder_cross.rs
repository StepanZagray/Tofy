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
pub struct DecoderArchitecture {
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ff_dim: usize,
}

impl DecoderArchitecture {
    pub fn new(dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        if dim == 0 || num_layers == 0 || num_heads == 0 || ff_dim == 0 {
            anyhow::bail!("decoder architecture values must be non-zero");
        }
        if !dim.is_multiple_of(num_heads) {
            anyhow::bail!("decoder dim {dim} must be divisible by heads {num_heads}");
        }
        Ok(Self {
            dim,
            num_layers,
            num_heads,
            ff_dim,
        })
    }
}

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

    fn csa_compress_rate(self) -> usize {
        match self {
            Self::TextGeneralist => 4,
            Self::CodeSpecialist => 4,
        }
    }

    fn hca_compress_rate(self) -> usize {
        match self {
            Self::TextGeneralist => 64,
            Self::CodeSpecialist => 128,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoderSelfAttentionKind {
    Sliding,
    CompressedSparse,
    HeavilyCompressed,
}

fn decoder_self_attention_kind(
    kind: DecoderKind,
    layer_idx: usize,
    num_layers: usize,
) -> DecoderSelfAttentionKind {
    if layer_idx == 0 {
        return DecoderSelfAttentionKind::Sliding;
    }
    if layer_idx + 1 == num_layers || layer_idx.is_multiple_of(kind.anchor_period()) {
        DecoderSelfAttentionKind::CompressedSparse
    } else {
        DecoderSelfAttentionKind::HeavilyCompressed
    }
}

fn decoder_cross_attention_enabled(layer_idx: usize, num_layers: usize) -> bool {
    match std::env::var("TOFY_DECODER_CROSS_ATTN_SCHEDULE")
        .unwrap_or_else(|_| "all".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "last" | "last-only" | "last_only" => layer_idx + 1 == num_layers,
        "every-3rd" | "every_3rd" | "third" => {
            layer_idx + 1 == num_layers || layer_idx.is_multiple_of(3)
        }
        "every-2nd" | "every_2nd" | "half" => {
            layer_idx + 1 == num_layers || layer_idx.is_multiple_of(2)
        }
        _ => true,
    }
}

fn decoder_latent_prefix_enabled() -> bool {
    std::env::var("TOFY_DECODER_LATENT_PREFIX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
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
    pub(crate) prefix_len: usize,
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        world_latent: &Tensor,
        domain_state: &Tensor,
        attention_kind: DecoderSelfAttentionKind,
        local_window: usize,
        csa_compress_rate: usize,
        hca_compress_rate: usize,
        use_cross_attention: bool,
    ) -> Result<Tensor> {
        let normed = self.ln1.forward(x)?;
        let self_out = match attention_kind {
            DecoderSelfAttentionKind::Sliding => {
                self.self_attn.forward_causal_local(&normed, local_window)?
            }
            DecoderSelfAttentionKind::CompressedSparse => {
                self.self_attn.forward_causal_compressed_sparse(
                    &normed,
                    local_window,
                    csa_compress_rate,
                    self.kind_index_topk(csa_compress_rate),
                )?
            }
            DecoderSelfAttentionKind::HeavilyCompressed => self
                .self_attn
                .forward_causal_heavily_compressed(&normed, local_window, hca_compress_rate)?,
        };
        let x = (x + self_out)?;

        let x = if use_cross_attention {
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
            (x + cross_out)?
        } else {
            x
        };

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

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        x: &Tensor,
        cross_kv_cache: &AttentionKvCache,
        domain_state: &Tensor,
        attention_kind: DecoderSelfAttentionKind,
        local_window: usize,
        csa_compress_rate: usize,
        hca_compress_rate: usize,
        use_cross_attention: bool,
        prefix_len: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let normed = self.ln1.forward(x)?;
        let self_out = match attention_kind {
            DecoderSelfAttentionKind::Sliding => {
                self.self_attn.forward_causal_local(&normed, local_window)?
            }
            DecoderSelfAttentionKind::CompressedSparse => {
                self.self_attn.forward_causal_compressed_sparse(
                    &normed,
                    local_window,
                    csa_compress_rate,
                    self.kind_index_topk(csa_compress_rate),
                )?
            }
            DecoderSelfAttentionKind::HeavilyCompressed => self
                .self_attn
                .forward_causal_heavily_compressed(&normed, local_window, hca_compress_rate)?,
        };
        let self_kv_cache = match attention_kind {
            DecoderSelfAttentionKind::Sliding => self
                .self_attn
                .project_self_kv_with_prefix(&normed, prefix_len)?,
            DecoderSelfAttentionKind::CompressedSparse => self
                .self_attn
                .project_self_kv_compressed(&normed, local_window, csa_compress_rate, prefix_len)?,
            DecoderSelfAttentionKind::HeavilyCompressed => self
                .self_attn
                .project_self_kv_compressed(&normed, local_window, hca_compress_rate, prefix_len)?,
        };
        let x = (x + self_out)?;

        let x = if use_cross_attention {
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
            (x + cross_out)?
        } else {
            x
        };

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

    #[allow(clippy::too_many_arguments)]
    fn forward_incremental(
        &self,
        x: &Tensor,
        cross_kv_cache: &AttentionKvCache,
        domain_state: &Tensor,
        attention_kind: DecoderSelfAttentionKind,
        local_window: usize,
        csa_compress_rate: usize,
        hca_compress_rate: usize,
        use_cross_attention: bool,
        self_kv_cache: Option<&AttentionKvCache>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let normed = self.ln1.forward(x)?;
        let (self_out, next_self_kv_cache) =
            match attention_kind {
                DecoderSelfAttentionKind::Sliding => self
                    .self_attn
                    .forward_causal_local_incremental(&normed, self_kv_cache, local_window)?,
                DecoderSelfAttentionKind::CompressedSparse => self
                    .self_attn
                    .forward_causal_compressed_sparse_incremental(
                        &normed,
                        self_kv_cache,
                        local_window,
                        csa_compress_rate,
                        self.kind_index_topk(csa_compress_rate),
                    )?,
                DecoderSelfAttentionKind::HeavilyCompressed => self
                    .self_attn
                    .forward_causal_heavily_compressed_incremental(
                        &normed,
                        self_kv_cache,
                        local_window,
                        hca_compress_rate,
                    )?,
            };
        let x = (x + self_out)?;

        let x = if use_cross_attention {
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
            (x + cross_out)?
        } else {
            x
        };

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

    fn kind_index_topk(&self, fallback: usize) -> usize {
        std::env::var("TOFY_DECODER_CSA_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback.max(1) * 2)
            .max(1)
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
        let device = input_ids.device();
        let prefix = self.latent_prefix(world_latent)?;
        let prefix_len = prefix.dim(1)?;
        let mut token_h = self.embed.forward(input_ids)?;
        let token_pe =
            positional_encoding_from(prefix_len, t, self.dim, device)?.to_dtype(token_h.dtype())?;
        token_h = token_h.broadcast_add(&token_pe)?;
        let token_domain_state = self.domain_state(device, b, t)?;
        let token_structure_state = self.structure_state(world_latent, b, t)?;
        token_h = token_h.broadcast_add(&token_domain_state)?;
        token_h = token_h.broadcast_add(&token_structure_state)?;

        let mut h = if prefix_len > 0 {
            let prefix_domain_state = self.domain_state(device, b, prefix_len)?;
            let prefix_pe =
                positional_encoding(prefix_len, self.dim, device)?.to_dtype(prefix.dtype())?;
            let prefix_h = prefix
                .broadcast_add(&prefix_domain_state)?
                .broadcast_add(&prefix_pe)?;
            Tensor::cat(&[prefix_h, token_h], 1)?
        } else {
            token_h
        };
        let total_len = h.dim(1)?;
        let domain_state = self.domain_state(device, b, total_len)?;
        let structure_state = self.structure_state(world_latent, b, total_len)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let attention_kind =
                decoder_self_attention_kind(self.kind, layer_idx, self.blocks.len());
            h = block.forward(
                &h,
                world_latent,
                &domain_state.broadcast_add(&structure_state)?,
                attention_kind,
                self.kind.local_window(),
                self.kind.csa_compress_rate(),
                self.kind.hca_compress_rate(),
                decoder_cross_attention_enabled(layer_idx, self.blocks.len()),
            )?;
        }
        h = self.ln_final.forward(&h)?;
        let token_h = h.narrow(1, prefix_len, t)?;
        self.lm_head
            .forward(&token_h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    fn latent_prefix(&self, world_latent: &Tensor) -> Result<Tensor> {
        if !decoder_latent_prefix_enabled() {
            let (batch, _, _) = world_latent.dims3()?;
            return Tensor::zeros(
                (batch, 0, self.dim),
                world_latent.dtype(),
                world_latent.device(),
            )
            .map_err(Into::into);
        }
        self.structure_proj
            .forward(world_latent)?
            .tanh()
            .map_err(Into::into)
    }

    fn token_logits(&self, h: &Tensor) -> Result<Tensor> {
        self.lm_head
            .forward(h)
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
        let pe = positional_encoding_from(state.prefix_len + position, 1, self.dim, device)?
            .to_dtype(h.dtype())?;
        h = h.broadcast_add(&pe)?;
        h = h.broadcast_add(&state.domain_state)?;
        h = h.broadcast_add(&state.structure_state)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let attention_kind =
                decoder_self_attention_kind(self.kind, layer_idx, self.blocks.len());
            let (next_h, next_cache) = block.forward_incremental(
                &h,
                &state.cross_kv_caches[layer_idx],
                &state.domain_state.broadcast_add(&state.structure_state)?,
                attention_kind,
                self.kind.local_window(),
                self.kind.csa_compress_rate(),
                self.kind.hca_compress_rate(),
                decoder_cross_attention_enabled(layer_idx, self.blocks.len()),
                state.self_kv_caches[layer_idx].as_ref(),
            )?;
            h = next_h;
            state.self_kv_caches[layer_idx] = Some(next_cache);
        }
        h = self.ln_final.forward(&h)?;
        self.token_logits(&h)
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
                prefix_len: 0,
                last_logits: None,
            });
        }
        let prompt_len = prompt_ids.len();
        let prefix = self.latent_prefix(world_latent)?;
        let prefix_len = prefix.dim(1)?;
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_len), device)?;
        let mut token_h = self.embed.forward(&input)?;
        let token_pe = positional_encoding_from(prefix_len, prompt_len, self.dim, device)?
            .to_dtype(token_h.dtype())?;
        token_h = token_h.broadcast_add(&token_pe)?;
        let token_domain_state = self.domain_state(device, 1, prompt_len)?;
        let token_structure_state = self.structure_state(world_latent, 1, prompt_len)?;
        token_h = token_h.broadcast_add(&token_domain_state)?;
        token_h = token_h.broadcast_add(&token_structure_state)?;

        let mut h = if prefix_len > 0 {
            let prefix_domain_state = self.domain_state(device, 1, prefix_len)?;
            let prefix_pe =
                positional_encoding(prefix_len, self.dim, device)?.to_dtype(prefix.dtype())?;
            let prefix_h = prefix
                .broadcast_add(&prefix_domain_state)?
                .broadcast_add(&prefix_pe)?;
            Tensor::cat(&[prefix_h, token_h], 1)?
        } else {
            token_h
        };
        let total_len = h.dim(1)?;
        let domain_state_full = self.domain_state(device, 1, total_len)?;
        let structure_state_full = self.structure_state(world_latent, 1, total_len)?;
        let cross_kv_caches = self.precompute_cross_kv_caches(world_latent)?;
        let mut self_kv_caches = Vec::with_capacity(self.blocks.len());

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let attention_kind =
                decoder_self_attention_kind(self.kind, layer_idx, self.blocks.len());
            let (next_h, self_kv_cache) = block.forward_prefill(
                &h,
                &cross_kv_caches[layer_idx],
                &domain_state_full.broadcast_add(&structure_state_full)?,
                attention_kind,
                self.kind.local_window(),
                self.kind.csa_compress_rate(),
                self.kind.hca_compress_rate(),
                decoder_cross_attention_enabled(layer_idx, self.blocks.len()),
                prefix_len,
            )?;
            h = next_h;
            self_kv_caches.push(Some(self_kv_cache));
        }
        h = self.ln_final.forward(&h)?;
        let token_h = h.narrow(1, prefix_len, prompt_len)?;
        let logits = self.token_logits(&token_h)?;
        Ok(DecoderGenerationState {
            token_ids: prompt_ids.to_vec(),
            self_kv_caches,
            cross_kv_caches,
            domain_state: self.domain_state(device, 1, 1)?,
            structure_state: self.structure_state(world_latent, 1, 1)?,
            prefix_len,
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
