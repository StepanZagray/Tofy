use anyhow::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};

use crate::util;

#[derive(Clone)]
pub struct AttentionKvCache {
    pub k: Tensor,
    pub v: Tensor,
    pub compressed_k: Option<Tensor>,
    pub compressed_v: Option<Tensor>,
    pub compressed_block_ends: Vec<usize>,
    pub token_count: usize,
    pub prefix_len: usize,
}

fn attention_kv_cache(k: Tensor, v: Tensor) -> Result<AttentionKvCache> {
    attention_kv_cache_with_prefix(k, v, 0)
}

fn attention_kv_cache_with_prefix(
    k: Tensor,
    v: Tensor,
    prefix_len: usize,
) -> Result<AttentionKvCache> {
    let token_count = k.dim(2)?;
    Ok(AttentionKvCache {
        k,
        v,
        compressed_k: None,
        compressed_v: None,
        compressed_block_ends: Vec::new(),
        token_count,
        prefix_len: prefix_len.min(token_count),
    })
}

fn compress_prefix_blocks(
    k: &Tensor,
    v: &Tensor,
    prefix_len: usize,
    compress_rate: usize,
    base_offset: usize,
) -> Result<Option<(Tensor, Tensor, Vec<usize>)>> {
    if prefix_len == 0 {
        return Ok(None);
    }
    let compress_rate = compress_rate.max(1);
    let mut k_blocks = Vec::new();
    let mut v_blocks = Vec::new();
    let mut block_ends = Vec::new();
    let mut start = 0usize;
    while start < prefix_len {
        let len = (prefix_len - start).min(compress_rate);
        k_blocks.push(
            k.narrow(2, start, len)?
                .sum(2)?
                .affine(1.0 / len as f64, 0.0)?
                .unsqueeze(2)?,
        );
        v_blocks.push(
            v.narrow(2, start, len)?
                .sum(2)?
                .affine(1.0 / len as f64, 0.0)?
                .unsqueeze(2)?,
        );
        block_ends.push(base_offset + start + len);
        start += len;
    }
    let k_refs = k_blocks.iter().collect::<Vec<_>>();
    let v_refs = v_blocks.iter().collect::<Vec<_>>();
    Ok(Some((
        Tensor::cat(&k_refs, 2)?,
        Tensor::cat(&v_refs, 2)?,
        block_ends,
    )))
}

fn compress_cache_from_full_kv(
    k: Tensor,
    v: Tensor,
    exact_tail: usize,
    compress_rate: usize,
    prefix_len: usize,
) -> Result<AttentionKvCache> {
    let token_count = k.dim(2)?;
    let prefix_len = prefix_len.min(token_count);
    let dynamic_len = token_count.saturating_sub(prefix_len);
    if dynamic_len <= exact_tail {
        return attention_kv_cache_with_prefix(k, v, prefix_len);
    }
    let old_len = dynamic_len - exact_tail;
    let old_k = k.narrow(2, prefix_len, old_len)?;
    let old_v = v.narrow(2, prefix_len, old_len)?;
    let compressed = compress_prefix_blocks(&old_k, &old_v, old_len, compress_rate, prefix_len)?;
    let prefix_k = k.narrow(2, 0, prefix_len)?;
    let prefix_v = v.narrow(2, 0, prefix_len)?;
    let k_tail = k.narrow(2, prefix_len + old_len, exact_tail)?;
    let v_tail = v.narrow(2, prefix_len + old_len, exact_tail)?;
    let k_cache = Tensor::cat(&[prefix_k, k_tail], 2)?;
    let v_cache = Tensor::cat(&[prefix_v, v_tail], 2)?;
    let (compressed_k, compressed_v, compressed_block_ends) = match compressed {
        Some((ck, cv, ends)) => (Some(ck), Some(cv), ends),
        None => (None, None, Vec::new()),
    };
    Ok(AttentionKvCache {
        k: k_cache,
        v: v_cache,
        compressed_k,
        compressed_v,
        compressed_block_ends,
        token_count,
        prefix_len,
    })
}

fn append_to_compressed_cache(
    cache: Option<&AttentionKvCache>,
    new_kv: AttentionKvCache,
    exact_tail: usize,
    compress_rate: usize,
) -> Result<AttentionKvCache> {
    let Some(cache) = cache else {
        return compress_cache_from_full_kv(new_kv.k, new_kv.v, exact_tail, compress_rate, 0);
    };
    let full_k = Tensor::cat(&[cache.k.clone(), new_kv.k], 2)?;
    let full_v = Tensor::cat(&[cache.v.clone(), new_kv.v], 2)?;
    let full_len = full_k.dim(2)?;
    let prefix_len = cache.prefix_len.min(full_len);
    let new_token_count = cache.token_count + full_len.saturating_sub(cache.k.dim(2)?);
    let dynamic_len = full_len.saturating_sub(prefix_len);
    if dynamic_len <= exact_tail {
        return Ok(AttentionKvCache {
            k: full_k,
            v: full_v,
            compressed_k: cache.compressed_k.clone(),
            compressed_v: cache.compressed_v.clone(),
            compressed_block_ends: cache.compressed_block_ends.clone(),
            token_count: new_token_count,
            prefix_len,
        });
    }
    let overflow = dynamic_len - exact_tail;
    let dynamic_start = cache
        .token_count
        .saturating_sub(cache.k.dim(2)?.saturating_sub(prefix_len));
    let old_k = full_k.narrow(2, prefix_len, overflow)?;
    let old_v = full_v.narrow(2, prefix_len, overflow)?;
    let compressed =
        compress_prefix_blocks(&old_k, &old_v, overflow, compress_rate, dynamic_start)?;
    let prefix_k = full_k.narrow(2, 0, prefix_len)?;
    let prefix_v = full_v.narrow(2, 0, prefix_len)?;
    let k_tail = full_k.narrow(2, prefix_len + overflow, exact_tail)?;
    let v_tail = full_v.narrow(2, prefix_len + overflow, exact_tail)?;
    let k_cache = Tensor::cat(&[prefix_k, k_tail], 2)?;
    let v_cache = Tensor::cat(&[prefix_v, v_tail], 2)?;
    let mut compressed_k_parts = Vec::new();
    let mut compressed_v_parts = Vec::new();
    let mut block_ends = cache.compressed_block_ends.clone();
    if let (Some(ck), Some(cv)) = (&cache.compressed_k, &cache.compressed_v) {
        compressed_k_parts.push(ck.clone());
        compressed_v_parts.push(cv.clone());
    }
    if let Some((ck, cv, mut ends)) = compressed {
        compressed_k_parts.push(ck);
        compressed_v_parts.push(cv);
        block_ends.append(&mut ends);
    }
    let compressed_k = if compressed_k_parts.is_empty() {
        None
    } else {
        Some(Tensor::cat(&compressed_k_parts, 2)?)
    };
    let compressed_v = if compressed_v_parts.is_empty() {
        None
    } else {
        Some(Tensor::cat(&compressed_v_parts, 2)?)
    };
    Ok(AttentionKvCache {
        k: k_cache,
        v: v_cache,
        compressed_k,
        compressed_v,
        compressed_block_ends: block_ends,
        token_count: new_token_count,
        prefix_len,
    })
}

fn topk_bias_from_scores(
    scores: &Tensor,
    topk: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let (batch, heads, queries, keys) = scores.dims4()?;
    if keys == 0 || topk >= keys {
        return Tensor::zeros((batch, heads, queries, keys), DType::F32, device)
            .map_err(Into::into);
    }
    let cpu_topk_enabled = std::env::var("TOFY_DECODER_CPU_TOPK_BIAS")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if !cpu_topk_enabled {
        return Tensor::zeros((batch, heads, queries, keys), DType::F32, device)
            .map_err(Into::into);
    }
    let values = scores
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut bias = vec![-1e9f32; batch * heads * queries * keys];
    for b_idx in 0..batch {
        for h_idx in 0..heads {
            for q_idx in 0..queries {
                let base = ((b_idx * heads + h_idx) * queries + q_idx) * keys;
                let mut ranked = (0..keys)
                    .map(|k_idx| (k_idx, values[base + k_idx]))
                    .filter(|(_, score)| score.is_finite())
                    .collect::<Vec<_>>();
                ranked.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (k_idx, _) in ranked.into_iter().take(topk) {
                    bias[base + k_idx] = 0.0;
                }
            }
        }
    }
    Tensor::from_vec(bias, (batch, heads, queries, keys), device)
        .map_err(|e| anyhow::anyhow!("{:?}", e))
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
    #[allow(dead_code)]
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

    /// DeepSeek-V4-inspired compressed sparse causal attention.
    ///
    /// The query attends to exact recent tokens plus compressed long-range blocks.
    /// This keeps local code/layout dependencies sharp while making old context
    /// enter through a much shorter summary sequence.
    pub fn forward_causal_compressed_sparse(
        &self,
        x: &Tensor,
        local_window: usize,
        compress_rate: usize,
        index_topk: usize,
    ) -> Result<Tensor> {
        self.forward_causal_compressed_sparse_windowed(
            x,
            local_window,
            compress_rate.max(1),
            index_topk.max(1),
        )
    }

    /// DeepSeek-V4-inspired heavily compressed causal attention.
    ///
    /// Uses a larger compression rate than CSA and is intended for non-anchor
    /// layers where a global signal is useful but exact retrieval is wasteful.
    pub fn forward_causal_heavily_compressed(
        &self,
        x: &Tensor,
        local_window: usize,
        compress_rate: usize,
    ) -> Result<Tensor> {
        self.forward_causal_heavily_compressed_windowed(x, local_window, compress_rate.max(1))
    }

    #[allow(dead_code)]
    fn forward_causal_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(x, mask)
    }

    #[allow(dead_code)]
    fn forward_with_mask(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
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
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
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
        self.project_self_kv_with_prefix(x, 0)
    }

    pub fn project_self_kv_with_prefix(
        &self,
        x: &Tensor,
        prefix_len: usize,
    ) -> Result<AttentionKvCache> {
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
        attention_kv_cache_with_prefix(k, v, prefix_len)
    }

    fn append_to_cache(
        &self,
        cache: Option<&AttentionKvCache>,
        new_kv: AttentionKvCache,
        max_tokens: Option<usize>,
    ) -> Result<AttentionKvCache> {
        let prefix_len = cache.map(|cache| cache.prefix_len).unwrap_or(0);
        let token_count = cache
            .map(|cache| cache.token_count)
            .unwrap_or(0)
            .saturating_add(new_kv.token_count);
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
            let prefix_len = prefix_len.min(cur);
            let dynamic_len = cur.saturating_sub(prefix_len);
            if dynamic_len > limit {
                let keep = limit.min(dynamic_len);
                let prefix_k = k.narrow(2, 0, prefix_len)?;
                let prefix_v = v.narrow(2, 0, prefix_len)?;
                let tail_k = k.narrow(2, cur - keep, keep)?;
                let tail_v = v.narrow(2, cur - keep, keep)?;
                k = Tensor::cat(&[prefix_k, tail_k], 2)?;
                v = Tensor::cat(&[prefix_v, tail_v], 2)?;
            }
        }
        let mut cache = attention_kv_cache_with_prefix(k, v, prefix_len)?;
        cache.token_count = token_count;
        Ok(cache)
    }

    #[allow(dead_code)]
    pub fn forward_causal_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_incremental_with_limit(x, cache, None)
    }

    #[allow(dead_code)]
    pub fn forward_causal_local_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        window: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_incremental_with_limit(x, cache, Some(window.max(1)))
    }

    pub fn project_self_kv_compressed(
        &self,
        x: &Tensor,
        exact_tail: usize,
        compress_rate: usize,
        prefix_len: usize,
    ) -> Result<AttentionKvCache> {
        let kv = self.project_self_kv_with_prefix(x, prefix_len)?;
        compress_cache_from_full_kv(
            kv.k,
            kv.v,
            exact_tail.max(1),
            compress_rate.max(1),
            prefix_len,
        )
    }

    pub fn forward_causal_compressed_sparse_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        local_window: usize,
        compress_rate: usize,
        index_topk: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_compressed_incremental(
            x,
            cache,
            local_window.max(1),
            compress_rate.max(1),
            Some(index_topk.max(1)),
        )
    }

    pub fn forward_causal_heavily_compressed_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        local_window: usize,
        compress_rate: usize,
    ) -> Result<(Tensor, AttentionKvCache)> {
        self.forward_causal_compressed_incremental(
            x,
            cache,
            local_window.max(1),
            compress_rate.max(1),
            None,
        )
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

    fn forward_causal_compressed_incremental(
        &self,
        x: &Tensor,
        cache: Option<&AttentionKvCache>,
        exact_tail: usize,
        compress_rate: usize,
        index_topk: Option<usize>,
    ) -> Result<(Tensor, AttentionKvCache)> {
        let x = x.contiguous()?;
        let (b, t_q, _) = x.dims3()?;
        let q = self.project_q(&x)?;
        let new_kv = self.project_self_kv(&x)?;
        let full_kv =
            append_to_compressed_cache(cache, new_kv, exact_tail.max(1), compress_rate.max(1))?;

        let mut k_parts = vec![full_kv.k.clone()];
        let mut v_parts = vec![full_kv.v.clone()];
        let mut bias_parts = vec![Tensor::zeros(
            (b, self.num_heads, t_q, full_kv.k.dim(2)?),
            q.dtype(),
            x.device(),
        )?];
        if let (Some(comp_k), Some(comp_v)) = (&full_kv.compressed_k, &full_kv.compressed_v) {
            let comp_bias = if let Some(topk) = index_topk {
                let scores = q
                    .contiguous()?
                    .matmul(&comp_k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
                topk_bias_from_scores(&scores, topk, x.device())?
            } else {
                Tensor::zeros(
                    (b, self.num_heads, t_q, comp_k.dim(2)?),
                    q.dtype(),
                    x.device(),
                )?
            };
            k_parts.push(comp_k.clone());
            v_parts.push(comp_v.clone());
            bias_parts.push(comp_bias);
        }

        let k_all = Tensor::cat(&k_parts, 2)?;
        let v_all = Tensor::cat(&v_parts, 2)?;
        let bias = Tensor::cat(&bias_parts, 3)?.to_dtype(q.dtype())?;
        util::ensure_same_dtype(&q, &k_all, "compressed incremental attention q/k")?;
        let scores =
            (q.matmul(&k_all.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / self.scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
        let attn_output = attn_weights.contiguous()?.matmul(&v_all.contiguous()?)?;
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

            let q_chunk = q.narrow(2, q_start, q_len)?.contiguous()?;
            let k_chunk = k.narrow(2, kv_start, kv_len)?;
            let v_chunk = v.narrow(2, kv_start, kv_len)?;
            util::ensure_same_dtype(&q_chunk, &k_chunk, "local attention q/k")?;
            let scores = (q_chunk
                .matmul(&k_chunk.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                / self.scale)?;
            let bias =
                local_chunk_bias(q_start, q_len, kv_start, kv_len, window, causal, x.device())?;
            let bias = bias.to_dtype(scores.dtype())?;
            let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
            outputs.push(attn_weights.contiguous()?.matmul(&v_chunk.contiguous()?)?);
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

    fn forward_causal_compressed_sparse_windowed(
        &self,
        x: &Tensor,
        local_window: usize,
        compress_rate: usize,
        index_topk: usize,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        if t <= local_window.max(1) {
            return self.forward_local_windowed(&x, local_window.max(1), true);
        }

        let query_block = ((local_window.max(1) * 2).max(32)).min(t.max(1));
        let local_radius = local_window.saturating_sub(1);
        let mut outputs = Vec::new();

        for q_start in (0..t).step_by(query_block) {
            let q_len = (t - q_start).min(query_block);
            let local_start = q_start.saturating_sub(local_radius);
            let local_end = q_start + q_len;
            let local_len = local_end.saturating_sub(local_start).max(1);

            let q_chunk = q.narrow(2, q_start, q_len)?;
            let local_k = k.narrow(2, local_start, local_len)?;
            let local_v = v.narrow(2, local_start, local_len)?;
            let local_bias = local_chunk_bias(
                q_start,
                q_len,
                local_start,
                local_len,
                local_window.max(1),
                true,
                x.device(),
            )?
            .broadcast_as((b, self.num_heads, q_len, local_len))?;

            let mut k_parts = vec![local_k];
            let mut v_parts = vec![local_v];
            let mut bias_parts = vec![local_bias];

            let compressed = compressed_causal_blocks(&k, &v, q_start + q_len, compress_rate)?;
            if let Some((comp_k, comp_v, block_ends)) = compressed {
                let index_scores =
                    q_chunk.matmul(&comp_k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
                let index_bias = topk_bias_from_scores(&index_scores, index_topk, x.device())?;
                k_parts.push(comp_k);
                v_parts.push(comp_v);
                let causal_bias = compressed_block_bias(q_start, q_len, &block_ends, x.device())?;
                bias_parts.push(causal_bias.broadcast_add(&index_bias)?);
            }

            let k_chunk = Tensor::cat(&k_parts, 2)?;
            let v_chunk = Tensor::cat(&v_parts, 2)?;
            let bias = Tensor::cat(&bias_parts, 3)?;
            util::ensure_same_dtype(&q_chunk, &k_chunk, "compressed causal attention q/k")?;
            let scores = (q_chunk
                .matmul(&k_chunk.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                / self.scale)?;
            let bias = bias.to_dtype(scores.dtype())?;
            let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
            outputs.push(attn_weights.contiguous()?.matmul(&v_chunk.contiguous()?)?);
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

    fn forward_causal_heavily_compressed_windowed(
        &self,
        x: &Tensor,
        local_window: usize,
        compress_rate: usize,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        if t <= local_window.max(1) {
            return self.forward_local_windowed(&x, local_window.max(1), true);
        }

        let query_block = ((local_window.max(1) * 2).max(32)).min(t.max(1));
        let local_radius = local_window.saturating_sub(1);
        let mut outputs = Vec::new();

        for q_start in (0..t).step_by(query_block) {
            let q_len = (t - q_start).min(query_block);
            let local_start = q_start.saturating_sub(local_radius);
            let local_end = q_start + q_len;
            let local_len = local_end.saturating_sub(local_start).max(1);

            let q_chunk = q.narrow(2, q_start, q_len)?;
            let local_k = k.narrow(2, local_start, local_len)?;
            let local_v = v.narrow(2, local_start, local_len)?;
            let local_bias = local_chunk_bias(
                q_start,
                q_len,
                local_start,
                local_len,
                local_window.max(1),
                true,
                x.device(),
            )?;

            let mut k_parts = vec![local_k];
            let mut v_parts = vec![local_v];
            let mut bias_parts = vec![local_bias];
            let compressed = compressed_causal_blocks(&k, &v, q_start + q_len, compress_rate)?;
            if let Some((comp_k, comp_v, block_ends)) = compressed {
                k_parts.push(comp_k);
                v_parts.push(comp_v);
                bias_parts.push(compressed_block_bias(
                    q_start,
                    q_len,
                    &block_ends,
                    x.device(),
                )?);
            }

            let k_chunk = Tensor::cat(&k_parts, 2)?;
            let v_chunk = Tensor::cat(&v_parts, 2)?;
            let bias = Tensor::cat(&bias_parts, 3)?;
            util::ensure_same_dtype(
                &q_chunk,
                &k_chunk,
                "heavily compressed causal attention q/k",
            )?;
            let scores = (q_chunk
                .matmul(&k_chunk.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                / self.scale)?;
            let bias = bias.to_dtype(scores.dtype())?;
            let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?;
            outputs.push(attn_weights.contiguous()?.matmul(&v_chunk.contiguous()?)?);
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
        attention_kv_cache(k, v)
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
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        util::ensure_same_dtype(&q, &k, "masked cross attention q/k")?;
        let scores =
            self.add_key_padding_bias((q.matmul(&k_t)? / self.scale)?, key_padding_mask)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.contiguous()?.matmul(&v)?;
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
#[allow(dead_code)]
fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            v[i * seq_len + j] = -1e9;
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

fn compressed_causal_blocks(
    k: &Tensor,
    v: &Tensor,
    upto: usize,
    compress_rate: usize,
) -> Result<Option<(Tensor, Tensor, Vec<usize>)>> {
    let (_, _, t, _) = k.dims4()?;
    let upto = upto.min(t);
    if upto == 0 || compress_rate <= 1 {
        return Ok(None);
    }

    let mut k_blocks = Vec::new();
    let mut v_blocks = Vec::new();
    let mut block_ends = Vec::new();
    let mut start = 0usize;
    while start < upto {
        let len = (upto - start).min(compress_rate);
        let scale = 1.0 / len.max(1) as f64;
        k_blocks.push(
            k.narrow(2, start, len)?
                .sum(2)?
                .affine(scale, 0.0)?
                .unsqueeze(2)?,
        );
        v_blocks.push(
            v.narrow(2, start, len)?
                .sum(2)?
                .affine(scale, 0.0)?
                .unsqueeze(2)?,
        );
        block_ends.push(start + len);
        start += len;
    }

    if k_blocks.is_empty() {
        return Ok(None);
    }
    let k_refs = k_blocks.iter().collect::<Vec<_>>();
    let v_refs = v_blocks.iter().collect::<Vec<_>>();
    Ok(Some((
        Tensor::cat(&k_refs, 2)?,
        Tensor::cat(&v_refs, 2)?,
        block_ends,
    )))
}

fn compressed_block_bias(
    q_start: usize,
    q_len: usize,
    block_ends: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut v = vec![0f32; q_len * block_ends.len()];
    for qi in 0..q_len {
        let visible_until = q_start + qi + 1;
        for (bi, &block_end) in block_ends.iter().enumerate() {
            if block_end > visible_until {
                v[qi * block_ends.len() + bi] = -1e9;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, q_len, block_ends.len()), device)
        .map_err(|e| anyhow::anyhow!("{:?}", e))
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
