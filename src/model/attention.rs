use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};

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
        assert!(dim % num_heads == 0, "dim must be divisible by num_heads");
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

    /// Sliding-window causal attention for sparse decoder layers.
    pub fn forward_causal_local(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        let mask = causal_local_mask(x.dim(1)?, window.max(1), x.device())?;
        self.forward_causal_with_mask(x, &mask)
    }

    fn forward_causal_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
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
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        let mut scores = q.matmul(&k_t)?;
        scores = (scores / self.scale)?;
        scores = scores.broadcast_add(&mask)?;
        self.attention_scores_to_output(scores, &v, b, t)
    }

    /// Cross-attention: query from one source, key/value from another
    pub fn forward_cross(&self, query: &Tensor, key_value: &Tensor) -> Result<Tensor> {
        let (b, t_q, _) = query.dims3()?;
        let (_, t_kv, _) = key_value.dims3()?;

        // Project to Q, K, V using Candle's Linear
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key_value)?;
        let v = self.v_proj.forward(key_value)?;

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
        let scores = q.matmul(&k_t)?;
        let scores = (scores / self.scale)?;

        Ok(self.attention_scores_to_output(scores, &v, b, t_q)?)
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
        let attn_output = attn_weights.matmul(&v)?;

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
        assert!(decoder_dim % num_heads == 0, "decoder_dim must be divisible by num_heads");
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

    /// query: [B, T_dec, decoder_dim], key_value: [B, T_world, world_dim]
    pub fn forward(&self, query: &Tensor, key_value: &Tensor) -> Result<Tensor> {
        let (b, t_q, _) = query.dims3()?;
        let (_, t_kv, _) = key_value.dims3()?;
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key_value)?;
        let v = self.v_proj.forward(key_value)?;
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
        let scores = (q.matmul(&k_t)? / self.scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_q, self.num_heads * self.head_dim))?;
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

/// Sinusoidal positional encoding (fixed, not learned)
pub fn positional_encoding(
    seq_len: usize,
    dim: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut pe = vec![0f32; seq_len * dim];

    for pos in 0..seq_len {
        for i in 0..dim / 2 {
            let angle = (pos as f64) / (10000f64).powf((2 * i) as f64 / dim as f64);
            pe[pos * dim + 2 * i] = angle.sin() as f32;
            pe[pos * dim + 2 * i + 1] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(pe, (1, seq_len, dim), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}
