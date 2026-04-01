//! Decoder with cross-attention to code world latent. Used for training (and optional inference)
//! when the decoder is conditioned on the world model's latent sequence instead of text.

use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{self as nn, VarBuilder};
use rand::Rng;

use crate::model::attention::{positional_encoding, CrossAttention, MultiHeadAttention};

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
        let cross_gate = self.cross_gate.forward(&cross_gate_in)?.relu()?.clamp(0.0, 1.0)?;
        let cross_out = self
            .cross_attn
            .forward(&normed, world_latent)?
            .broadcast_mul(&cross_gate)?;
        let x = (x + cross_out)?;

        let normed = self.ln3.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let adapter_in = normed.broadcast_add(domain_state)?;
        let adapter = self.adapter_up.forward(&self.adapter_down.forward(&adapter_in)?.gelu()?)?;
        let ff_out = ff_out.broadcast_add(&adapter.affine(0.5, 0.0)?)?;
        Ok((x + ff_out)?)
    }
}

/// Decoder-only transformer conditioned on world latent via cross-attention.
/// Input: token ids [B, T]. World latent: [B, T_world, world_dim].
/// Output: logits [B, T, vocab_size].
pub struct CodeDecoder {
    embed: nn::Embedding,
    kind_embed: nn::Embedding,
    blocks: Vec<CodeDecoderBlock>,
    ln_final: nn::LayerNorm,
    lm_head: nn::Linear,
    dim: usize,
    kind: DecoderKind,
}

impl CodeDecoder {
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
        let pe = positional_encoding(t, self.dim, input_ids.device())?;
        h = h.broadcast_add(&pe)?;
        let kind_ids = Tensor::from_vec(vec![self.kind.id(); b], (b,), input_ids.device())?;
        let domain_state = self
            .kind_embed
            .forward(&kind_ids)?
            .unsqueeze(1)?
            .broadcast_as((b, t, self.dim))?;
        h = h.broadcast_add(&domain_state)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let use_full_attention =
                layer_idx % self.kind.anchor_period() == 0 || layer_idx + 1 == self.blocks.len();
            h = block.forward(
                &h,
                world_latent,
                &domain_state,
                use_full_attention,
                self.kind.local_window(),
            )?;
        }
        h = self.ln_final.forward(&h)?;
        self.lm_head
            .forward(&h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    /// Autoregressive generation conditioned on world latent.
    /// - prompt_ids: initial token ids (context).
    /// - world_latent: [1, T_world, world_dim].
    /// - temperature: 0.0 = argmax, >0 = sampling.
    /// - stop_at: if Some(id), stop generation when this token is produced (e.g. pad/EOS).
    /// Returns only the newly generated token ids (length <= max_new_tokens).
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
        let mut ids = prompt_ids.to_vec();
        let _vocab_size = self.lm_head.weight().dims2()?.1;
        let start_len = ids.len();

        for _ in 0..max_new_tokens {
            let seq_len = ids.len();
            let input = Tensor::from_vec(ids.clone(), (1, seq_len), device)?;
            let logits = self.forward(&input, world_latent)?;
            let last = logits.i((0, seq_len - 1, ..))?;
            let next_id = if temperature <= 0.0 {
                let logits_v: Vec<f32> = last.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                let scale = 1.0_f64 / (temperature as f64);
                let scaled = (scale * &last)?;
                let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;
                let probs_v: Vec<f32> = probs.to_vec1()?;
                let mut rng = rand::thread_rng();
                let r: f32 = rng.gen();
                let mut chosen = 0u32;
                let mut cum = 0.0f32;
                for (i, &p) in probs_v.iter().enumerate() {
                    cum += p;
                    if r <= cum {
                        chosen = i as u32;
                        break;
                    }
                }
                chosen
            };
            if stop_at == Some(next_id) {
                break;
            }
            ids.push(next_id);
        }

        Ok(ids[start_len..].to_vec())
    }

    /// Single step: given current token ids, return next token id (for streaming).
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
        let seq_len = ids.len();
        let input = Tensor::from_vec(ids.to_vec(), (1, seq_len), device)?;
        let logits = self.forward(&input, world_latent)?;
        let last = logits.i((0, seq_len - 1, ..))?;
        let next_id = if temperature <= 0.0 {
            let logits_v: Vec<f32> = last.to_vec1()?;
            logits_v
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            let scale = 1.0_f64 / (temperature as f64);
            let scaled = (scale * &last)?;
            let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;
            let probs_v: Vec<f32> = probs.to_vec1()?;
            let mut rng = rand::thread_rng();
            let r: f32 = rng.gen();
            let mut chosen = 0u32;
            let mut cum = 0.0f32;
            for (i, &p) in probs_v.iter().enumerate() {
                cum += p;
                if r <= cum {
                    chosen = i as u32;
                    break;
                }
            }
            chosen
        };
        Ok(next_id)
    }
}
