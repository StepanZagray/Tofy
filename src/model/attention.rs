use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};

use crate::util;

#[derive(Clone)]
pub struct AttentionKvCache {
    pub k: Tensor,
    pub v: Tensor,
}

/// Multi-Head Attention using Candle's built-in primitives
/// Uses nn::Linear for projections, Tensor::matmul for attention, nn::ops::softmax
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize) -> Result<Self> {
        assert!(
            dim.is_multiple_of(num_heads),
            "dim must be divisible by num_heads"
        );
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).sqrt();

        // Use Candle's linear layers for Q, K, V, and output projections
        let q_proj = nn::linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = nn::linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = nn::linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = nn::linear(dim, dim, vb.pp("out_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Self-attention: query, key, value all come from the same source
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_cross(x, x)
    }

    /// Causal self-attention (for decoder): mask out future positions
    pub fn forward_causal(&self, x: &Tensor) -> Result<Tensor> {
        let mask = causal_mask(x.dim(1)?, x.device())?;
        self.forward_causal_with_mask(x, &mask)
    }

    /// Sliding-window bidirectional attention for encoder layers.
    pub fn forward_local(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        self.forward_local_windowed(x, window.max(1), false)
    }

    /// Sliding-window causal attention for sparse decoder layers.
    pub fn forward_causal_local(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        self.forward_local_windowed(x, window.max(1), true)
    }

    fn forward_causal_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(x, mask)
    }

    fn forward_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        util::ensure_same_dtype(&q, &k, "self attention q/k")?;
        let mut scores = q.matmul(&k_t)?;
        scores = (scores / self.scale)?;
        let mask = mask.to_dtype(scores.dtype())?;
        scores = scores.broadcast_add(&mask)?;
        self.attention_scores_to_output(scores, &v, b, t)
    }

    /// Cross-attention: query from one source, key/value from another
    pub fn forward_cross(&self, query: &Tensor, key_value: &Tensor) -> Result<Tensor> {
        let query = query.contiguous()?;
        let key_value = key_value.contiguous()?;
        let (b, t_q, _) = query.dims3()?;
        let (_, t_kv, _) = key_value.dims3()?;

        // Project to Q, K, V using Candle's Linear
        let q = self.q_proj.forward(&query)?;
        let k = self.k_proj.forward(&key_value)?;
        let v = self.v_proj.forward(&key_value)?;

        // Reshape for multi-head: [B, T, D] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        let q = q
            .reshape((b, t_q, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention scores: Q @ K^T / sqrt(d_k)
        // [B, num_heads, T_q, head_dim] @ [B, num_heads, head_dim, T_kv] -> [B, num_heads, T_q, T_kv]
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        util::ensure_same_dtype(&q, &k, "cross attention q/k")?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores / self.scale)?;

        self.attention_scores_to_output(scores, &v, b, t_q)
    }

    fn project_self_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    fn project_q(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        self.q_proj
            .forward(&x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Into::into)
    }

    pub fn project_self_kv(&self, x: &Tensor) -> Result<AttentionKvCache> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let k = self
            .k_proj
            .forward(&x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(&x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(AttentionKvCache { k, v })
    }

    fn append_to_cache(
        &self,
        cache: Option<&AttentionKvCache>,
        new_kv: AttentionKvCache,
        max_tokens: Option<usize>,
    ) -> Result<AttentionKvCache> {
        let mut k = if let Some(cache) = cache {
            Tensor::cat(&[cache.k.clone(), new_kv.k], 2)?
        } else {
            new_kv.k
        };
        let mut v = if let Some(cache) = cache {
            Tensor::cat(&[cache.v.clone(), new_kv.v], 2)?
        } else {
            new_kv.v
        };
        if let Some(limit) = max_tokens.filter(|&limit| limit > 0) {
            let cur = k.dim(2)?;
            if cur > limit {
                let keep = limit.min(cur);
                k = k.narrow(2, cur - keep, keep)?;
                v = v.narrow(2, cur - keep, keep)?;
            }
        }
        Ok(AttentionKvCache { k, v })
    }

    pub fn forward_causal_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_incremental_with_limit(x, cache, None)
    }

    pub fn forward_causal_local_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        window: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_incremental_with_limit(x, cache, Some(window.max(1)))
    }

    fn forward_causal_incremental_with_limit(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        max_tokens: Option<usize>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let x = x.contiguous()?;
        let (b, t_q, _) = x.dims3()?;
        let q = self.project_q(&x)?;
        let new_kv = self.project_self_kv(&x)?;
        let full_kv = self.append_to_cache(cache, new_kv, max_tokens)?;
        let (_, _, t_kv, _) = full_kv.k.dims4()?;
        let q = q
            .reshape((b * self.num_heads, t_q, self.head_dim))?
            .contiguous()?;
        let k = full_kv
            .k
            .contiguous()?
            .reshape((b * self.num_heads, t_kv, self.head_dim))?;
        let v = full_kv
            .v
            .contiguous()?
            .reshape((b * self.num_heads, t_kv, self.head_dim))?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        util::ensure_same_dtype(&q, &k, "incremental self attention q/k")?;
        let scores = q
            .matmul(&k_t)?
            .reshape((b, self.num_heads, t_q, t_kv))?
            .affine(1.0 / self.scale, 0.0)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?.reshape((
            b * self.num_heads,
            t_q,
            t_kv,
        ))?;
        let attn_output =
            attn_weights
                .matmul(&v)?
                .reshape((b, self.num_heads, t_q, self.head_dim))?;
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_q,
            self.num_heads * self.head_dim,
        ))?;
        let out = self
            .out_proj
            .forward(&out)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok((out, full_kv))
    }

    fn forward_local_windowed(&self, x: &Tensor, window: usize, causal: bool) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        let query_block = ((window.max(1) * 4).max(32)).min(t.max(1));
        let radius = window.saturating_sub(1);
        let mut outputs = Vec::new();

        for q_start in (0..t).step_by(query_block) {
            let q_len = (t - q_start).min(query_block);
            let kv_start = q_start.saturating_sub(radius);
            let kv_end = if causal {
                q_start + q_len
            } else {
                (q_start + q_len + radius).min(t)
            };
            let kv_len = kv_end.saturating_sub(kv_start).max(1);

            let q_chunk = q.narrow(2, q_start, q_len)?;
            let k_chunk = k.narrow(2, kv_start, kv_len)?;
            let v_chunk = v.narrow(2, kv_start, kv_len)?;
            util::ensure_same_dtype(&q_chunk, &k_chunk, "local attention q/k")?;
            let scores = (q_chunk.matmul(&k_chunk.transpose(D::Minus2, D::Minus1)?)? / self.scale)?;
            let bias =
                local_chunk_bias(q_start, q_len, kv_start, kv_len, window, causal, x.device())?;
            let bias = bias.to_dtype(scores.dtype())?;
            let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
            outputs.push(attn_weights.matmul(&v_chunk)?);
        }

        let attn_output = Tensor::cat(&outputs, 2)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj
            .forward(&attn_output)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    fn attention_scores_to_output(
        &self,
        scores: Tensor,
        v: &Tensor,
        b: usize,
        t_q: usize,
    ) -> Result<Tensor> {
        // Softmax over last dimension (key positions)
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Apply attention to values
        // [B, num_heads, T_q, T_kv] @ [B, num_heads, T_kv, head_dim] -> [B, num_heads, T_q, head_dim]
        let attn_output = attn_weights.matmul(v)?;

        // Reshape back: [B, num_heads, T_q, head_dim] -> [B, T_q, D]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_q,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.out_proj
            .forward(&attn_output)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }
}

/// Transformer block with self-attention, layer norm, and feed-forward
pub struct TransformerBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?;
        let ln1 = nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?;
        let ff1 = nn::linear(dim, ff_dim, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_dim, dim, vb.pp("ff2"))?;

        Ok(Self {
            attn,
            ln1,
            ln2,
            ff1,
            ff2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture (more stable training)
        // Self-attention with residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let x = (x + attn_out)?;

        // Feed-forward with residual
        let normed = self.ln2.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        Ok((x + ff_out)?)
    }
}

pub struct LocalTransformerBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl LocalTransformerBlock {
    pub fn new(
        vb: VarBuilder<'_>,
        dim: usize,
        num_heads: usize,
        ff_dim: usize,
        _window: usize,
    ) -> Result<Self> {
        let attn = MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?;
        let ln1 = nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?;
        let ln2 = nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?;
        let ff1 = nn::linear(dim, ff_dim, vb.pp("ff1"))?;
        let ff2 = nn::linear(ff_dim, dim, vb.pp("ff2"))?;

        Ok(Self {
            attn,
            ln1,
            ln2,
            ff1,
            ff2,
        })
    }

    pub fn forward_with_window(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attn.forward_local(&normed, window.max(1))?;
        let x = (x + attn_out)?;

        let normed = self.ln2.forward(&x)?;
        let ff_out = self.ff1.forward(&normed)?.gelu()?;
        let ff_out = self.ff2.forward(&ff_out)?;
        Ok((x + ff_out)?)
    }
}

/// Cross-attention: Q from decoder (dec_dim), K/V from world latent (world_dim). Output dim = dec_dim.
pub struct CrossAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl CrossAttention {
    pub fn new(
        vb: VarBuilder<'_>,
        decoder_dim: usize,
        world_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        assert!(
            decoder_dim.is_multiple_of(num_heads),
            "decoder_dim must be divisible by num_heads"
        );
        let head_dim = decoder_dim / num_heads;
        let scale = (head_dim as f64).sqrt();
        let q_proj = nn::linear(decoder_dim, decoder_dim, vb.pp("q_proj"))?;
        let k_proj = nn::linear(world_dim, decoder_dim, vb.pp("k_proj"))?;
        let v_proj = nn::linear(world_dim, decoder_dim, vb.pp("v_proj"))?;
        let out_proj = nn::linear(decoder_dim, decoder_dim, vb.pp("out_proj"))?;
        Ok(Self {
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    fn add_key_padding_bias(
        &self,
        scores: Tensor,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let Some(mask) = key_padding_mask else {
            return Ok(scores);
        };
        let (b, h, t_q, t_kv) = scores.dims4()?;
        let bias = mask
            .contiguous()?
            .affine(1e9, -1e9)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as((b, h, t_q, t_kv))?;
        let bias = bias.to_dtype(scores.dtype())?;
        Ok(scores.broadcast_add(&bias)?)
    }

    /// query: [B, T_dec, decoder_dim], key_value: [B, T_world, world_dim]
    pub fn forward(&self, query: &Tensor, key_value: &Tensor) -> Result<Tensor> {
        self.forward_masked(query, key_value, None)
    }

    pub fn project_kv(&self, key_value: &Tensor) -> Result<AttentionKvCache> {
        let key_value = key_value.contiguous()?;
        let (b, t_kv, _) = key_value.dims3()?;
        let k = self
            .k_proj
            .forward(&key_value)?
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(&key_value)?
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(AttentionKvCache { k, v })
    }

    pub fn forward_precomputed(
        &self,
        query: &Tensor,
        kv_cache: &AttentionKvCache,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query = query.contiguous()?;
        let (b, t_q, _) = query.dims3()?;
        let (_, _, t_kv, _) = kv_cache.k.dims4()?;
        let q = self.q_proj.forward(&query)?;
        let q = q
            .reshape((b, t_q, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b * self.num_heads, t_q, self.head_dim))?
            .contiguous()?;
        let k = kv_cache
            .k
            .contiguous()?
            .reshape((b * self.num_heads, t_kv, self.head_dim))?;
        let v = kv_cache
            .v
            .contiguous()?
            .reshape((b * self.num_heads, t_kv, self.head_dim))?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        util::ensure_same_dtype(&q, &k, "precomputed cross attention q/k")?;
        let scores = q
            .matmul(&k_t)?
            .reshape((b, self.num_heads, t_q, t_kv))?
            .affine(1.0 / self.scale, 0.0)?;
        let scores = self.add_key_padding_bias(scores, key_padding_mask)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?.reshape((
            b * self.num_heads,
            t_q,
            t_kv,
        ))?;
        let attn_output =
            attn_weights
                .matmul(&v)?
                .reshape((b, self.num_heads, t_q, self.head_dim))?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_q,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj
            .forward(&attn_output)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    /// query: [B, T_dec, decoder_dim], key_value: [B, T_world, world_dim], key_padding_mask: [B, T_world]
    pub fn forward_masked(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query = query.contiguous()?;
        let key_value = key_value.contiguous()?;
        let (b, t_q, _) = query.dims3()?;
        let (_, t_kv, _) = key_value.dims3()?;
        let q = self.q_proj.forward(&query)?;
        let k = self.k_proj.forward(&key_value)?;
        let v = self.v_proj.forward(&key_value)?;
        let q = q
            .reshape((b, t_q, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t_kv, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        util::ensure_same_dtype(&q, &k, "masked cross attention q/k")?;
        let scores =
            self.add_key_padding_bias((q.matmul(&k_t)? / self.scale)?, key_padding_mask)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_q,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj
            .forward(&attn_output)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }
}

/// Causal mask for decoder self-attention: [1, 1, seq_len, seq_len], (i,j) = 0 if j <= i else -1e9
fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            v[i * seq_len + j] = -1e9;
        }
    }
    Tensor::from_vec(v, (1, 1, seq_len, seq_len), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

#[cfg(test)]
fn local_mask(seq_len: usize, window: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; seq_len * seq_len];
    let radius = window.saturating_sub(1);
    for i in 0..seq_len {
        let left = i.saturating_sub(radius);
        let right = (i + radius + 1).min(seq_len);
        for j in 0..seq_len {
            if j < left || j >= right {
                v[i * seq_len + j] = -1e9;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, seq_len, seq_len), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

#[cfg(test)]
fn causal_local_mask(
    seq_len: usize,
    window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut v = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let left = i.saturating_sub(window.saturating_sub(1));
        for j in 0..seq_len {
            if j > i || j < left {
                v[i * seq_len + j] = -1e9;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, seq_len, seq_len), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

fn local_chunk_bias(
    q_start: usize,
    q_len: usize,
    kv_start: usize,
    kv_len: usize,
    window: usize,
    causal: bool,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let radius = window.saturating_sub(1);
    let mut v = vec![0f32; q_len * kv_len];
    for qi in 0..q_len {
        let q_abs = q_start + qi;
        let left = q_abs.saturating_sub(radius);
        let right = if causal {
            q_abs + 1
        } else {
            q_abs + radius + 1
        };
        for kj in 0..kv_len {
            let k_abs = kv_start + kj;
            if k_abs < left || k_abs >= right {
                v[qi * kv_len + kj] = -1e9;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, q_len, kv_len), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

/// Sinusoidal positional encoding (fixed, not learned)
pub fn positional_encoding(
    seq_len: usize,
    dim: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    positional_encoding_from(0, seq_len, dim, device)
}

pub fn positional_encoding_from(
    start: usize,
    seq_len: usize,
    dim: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut pe = vec![0f32; seq_len * dim];

    for pos in 0..seq_len {
        for i in 0..dim / 2 {
            let absolute_pos = start + pos;
            let angle = (absolute_pos as f64) / (10000f64).powf((2 * i) as f64 / dim as f64);
            pe[pos * dim + 2 * i] = angle.sin() as f32;
            pe[pos * dim + 2 * i + 1] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(pe, (1, seq_len, dim), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn test_attention_input(
        batch: usize,
        seq: usize,
        dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let values = (0..batch * seq * dim)
            .map(|i| ((i % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (batch, seq, dim), device).map_err(Into::into)
    }

    #[test]
    fn local_attention_matches_dense_mask() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = MultiHeadAttention::new(vb.pp("attn"), 8, 2)?;
        let x = test_attention_input(2, 19, 8, &device)?;
        let dense = attn.forward_with_mask(&x, &local_mask(19, 5, &device)?)?;
        let sparse = attn.forward_local(&x, 5)?;
        let max_diff = crate::util::scalar_f32(&dense.broadcast_sub(&sparse)?.abs()?.max_all()?)?;
        assert!(max_diff < 1e-4, "local attention mismatch: {max_diff}");
        Ok(())
    }

    #[test]
    fn causal_local_attention_matches_dense_mask() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = MultiHeadAttention::new(vb.pp("attn"), 12, 3)?;
        let x = test_attention_input(2, 23, 12, &device)?;
        let dense = attn.forward_with_mask(&x, &causal_local_mask(23, 7, &device)?)?;
        let sparse = attn.forward_causal_local(&x, 7)?;
        let max_diff = crate::util::scalar_f32(&dense.broadcast_sub(&sparse)?.abs()?.max_all()?)?;
        assert!(
            max_diff < 1e-4,
            "causal local attention mismatch: {max_diff}"
        );
        Ok(())
    }

    #[test]
    fn incremental_causal_attention_matches_last_token_full_attention() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = MultiHeadAttention::new(vb.pp("attn"), 8, 2)?;
        let x = test_attention_input(1, 9, 8, &device)?;
        let full = attn.forward_causal(&x)?;
        let mut cache = None;
        let mut last = None;
        for idx in 0..9 {
            let token = x.narrow(1, idx, 1)?;
            let (out, next_cache) = attn.forward_causal_incremental(&token, cache.as_ref())?;
            cache = Some(next_cache);
            last = Some(out);
        }
        let full_last = full.narrow(1, 8, 1)?;
        let inc_last = last.expect("incremental output");
        let max_diff =
            crate::util::scalar_f32(&full_last.broadcast_sub(&inc_last)?.abs()?.max_all()?)?;
        assert!(
            max_diff < 1e-4,
            "incremental causal attention mismatch: {max_diff}"
        );
        Ok(())
    }
}
