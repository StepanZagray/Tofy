use anyhow::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};

use crate::util;

const ATTENTION_MASK_VALUE: f32 = -1.0e4;

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

impl AttentionKvCache {
    pub(crate) fn detached(self) -> Self {
        Self {
            k: self.k.detach(),
            v: self.v.detach(),
            compressed_k: self.compressed_k.map(|tensor| tensor.detach()),
            compressed_v: self.compressed_v.map(|tensor| tensor.detach()),
            compressed_block_ends: self.compressed_block_ends,
            token_count: self.token_count,
            prefix_len: self.prefix_len,
        }
    }
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
    let overflow_k = full_k.narrow(2, prefix_len, overflow)?;
    let overflow_v = full_v.narrow(2, prefix_len, overflow)?;
    let prefix_k = full_k.narrow(2, 0, prefix_len)?;
    let prefix_v = full_v.narrow(2, 0, prefix_len)?;
    let k_tail = full_k.narrow(2, prefix_len + overflow, exact_tail)?;
    let v_tail = full_v.narrow(2, prefix_len + overflow, exact_tail)?;
    let k_cache = Tensor::cat(&[prefix_k, k_tail], 2)?;
    let v_cache = Tensor::cat(&[prefix_v, v_tail], 2)?;
    let (compressed_k, compressed_v, compressed_block_ends) = append_compressed_blocks(
        cache.compressed_k.as_ref(),
        cache.compressed_v.as_ref(),
        &cache.compressed_block_ends,
        &overflow_k,
        &overflow_v,
        dynamic_start,
        prefix_len,
        compress_rate,
    )?;
    Ok(AttentionKvCache {
        k: k_cache,
        v: v_cache,
        compressed_k,
        compressed_v,
        compressed_block_ends,
        token_count: new_token_count,
        prefix_len,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_compressed_blocks(
    compressed_k: Option<&Tensor>,
    compressed_v: Option<&Tensor>,
    block_ends: &[usize],
    overflow_k: &Tensor,
    overflow_v: &Tensor,
    overflow_start: usize,
    prefix_len: usize,
    compress_rate: usize,
) -> Result<(Option<Tensor>, Option<Tensor>, Vec<usize>)> {
    let overflow_len = overflow_k.dim(2)?;
    if overflow_len == 0 {
        return Ok((
            compressed_k.cloned(),
            compressed_v.cloned(),
            block_ends.to_vec(),
        ));
    }
    let compress_rate = compress_rate.max(1);
    let mut k_parts = Vec::new();
    let mut v_parts = Vec::new();
    let mut ends = block_ends.to_vec();

    if let (Some(ck), Some(cv)) = (compressed_k, compressed_v) {
        let block_count = ck.dim(2)?;
        for idx in 0..block_count {
            k_parts.push(ck.narrow(2, idx, 1)?);
            v_parts.push(cv.narrow(2, idx, 1)?);
        }
    }

    let mut consumed = 0usize;
    if let Some(last_end_idx) = ends.len().checked_sub(1) {
        let last_idx = k_parts.len().saturating_sub(1);
        let last_start = if last_end_idx >= 1 {
            ends[last_end_idx - 1]
        } else {
            prefix_len
        };
        let last_len = ends[last_end_idx].saturating_sub(last_start);
        if last_len > 0
            && last_len < compress_rate
            && ends[last_end_idx] == overflow_start
            && consumed < overflow_len
        {
            let take = (compress_rate - last_len).min(overflow_len - consumed);
            let new_k_sum = overflow_k.narrow(2, consumed, take)?.sum(2)?.unsqueeze(2)?;
            let new_v_sum = overflow_v.narrow(2, consumed, take)?.sum(2)?.unsqueeze(2)?;
            let total_len = last_len + take;
            let merged_k = k_parts[last_idx]
                .affine(last_len as f64, 0.0)?
                .broadcast_add(&new_k_sum)?
                .affine(1.0 / total_len as f64, 0.0)?;
            let merged_v = v_parts[last_idx]
                .affine(last_len as f64, 0.0)?
                .broadcast_add(&new_v_sum)?
                .affine(1.0 / total_len as f64, 0.0)?;
            k_parts[last_idx] = merged_k;
            v_parts[last_idx] = merged_v;
            ends[last_end_idx] += take;
            consumed += take;
        }
    }

    while consumed < overflow_len {
        let take = (overflow_len - consumed).min(compress_rate);
        let scale = 1.0 / take.max(1) as f64;
        k_parts.push(
            overflow_k
                .narrow(2, consumed, take)?
                .sum(2)?
                .affine(scale, 0.0)?
                .unsqueeze(2)?,
        );
        v_parts.push(
            overflow_v
                .narrow(2, consumed, take)?
                .sum(2)?
                .affine(scale, 0.0)?
                .unsqueeze(2)?,
        );
        ends.push(overflow_start + consumed + take);
        consumed += take;
    }

    let compressed_k = if k_parts.is_empty() {
        None
    } else {
        let refs = k_parts.iter().collect::<Vec<_>>();
        Some(Tensor::cat(&refs, 2)?)
    };
    let compressed_v = if v_parts.is_empty() {
        None
    } else {
        let refs = v_parts.iter().collect::<Vec<_>>();
        Some(Tensor::cat(&refs, 2)?)
    };
    Ok((compressed_k, compressed_v, ends))
}

fn dsa_topk_indices_from_scores(
    scores: &Tensor,
    causal_bias: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    let (_, _, _, keys) = scores.dims4()?;
    let topk = topk.max(1).min(keys.max(1));
    let selector_scores = scores.to_dtype(DType::F32)?.relu()?.mean_keepdim(1)?;
    let selector_causal_bias = causal_bias.to_dtype(DType::F32)?.narrow(1, 0, 1)?;
    selector_scores
        .broadcast_add(&selector_causal_bias)?
        .contiguous()?
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, topk)
        .and_then(|ids| ids.contiguous())
        .map_err(Into::into)
}

fn dsa_topk_bias_from_scores(
    scores: &Tensor,
    causal_bias: &Tensor,
    topk: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let (batch, heads, queries, keys) = scores.dims4()?;
    if keys == 0 || topk >= keys {
        return Tensor::zeros((batch, heads, queries, keys), DType::F32, device)
            .map_err(Into::into);
    }
    let topk_idxs = dsa_topk_indices_from_scores(scores, causal_bias, topk)?;
    let init = Tensor::full(ATTENTION_MASK_VALUE, (batch, 1, queries, keys), device)?;
    let zeros = Tensor::zeros((batch, 1, queries, topk.min(keys)), DType::F32, device)?;
    init.scatter(&topk_idxs, &zeros, D::Minus1)?
        .broadcast_as((batch, heads, queries, keys))
        .map_err(Into::into)
}

fn decoder_attention_query_block(_local_window: usize, seq_len: usize) -> usize {
    std::env::var("TOFY_DECODER_ATTENTION_QUERY_BLOCK")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
        .min(seq_len.max(1))
}

/// Multi-Head Attention using Candle's built-in primitives
/// Uses nn::Linear for projections, Tensor::matmul for attention, nn::ops::softmax
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    use_rope: bool,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize) -> Result<Self> {
        Self::new_impl(vb, dim, num_heads, false)
    }

    pub fn new_with_rope(vb: VarBuilder<'_>, dim: usize, num_heads: usize) -> Result<Self> {
        Self::new_impl(vb, dim, num_heads, true)
    }

    fn new_impl(vb: VarBuilder<'_>, dim: usize, num_heads: usize, use_rope: bool) -> Result<Self> {
        assert!(
            dim.is_multiple_of(num_heads),
            "dim must be divisible by num_heads"
        );
        let head_dim = dim / num_heads;
        if use_rope && !head_dim.is_multiple_of(2) {
            anyhow::bail!("RoPE requires an even attention head dimension, got {head_dim}");
        }
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
            use_rope,
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
        self.forward_local_windowed(x, window.max(1), false, None, 0)
    }

    pub fn forward_local_masked(
        &self,
        x: &Tensor,
        window: usize,
        key_padding_mask: &Tensor,
    ) -> Result<Tensor> {
        self.forward_local_windowed(x, window.max(1), false, Some(key_padding_mask), 0)
    }

    /// Sliding-window causal attention for sparse decoder layers.
    pub fn forward_causal_local(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        self.forward_local_windowed(x, window.max(1), true, None, 0)
    }

    pub fn forward_causal_local_with_prefix(
        &self,
        x: &Tensor,
        window: usize,
        prefix_len: usize,
    ) -> Result<Tensor> {
        self.forward_local_windowed(x, window.max(1), true, None, prefix_len)
    }

    pub fn forward_self_masked(&self, x: &Tensor, key_padding_mask: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        util::ensure_same_dtype(&q, &k, "self masked attention q/k")?;
        let scores = (q.matmul(&k_t)? / self.scale)?;
        let bias = key_padding_bias(key_padding_mask, b, self.num_heads, t, t, scores.dtype())?;
        self.attention_scores_to_output(scores.broadcast_add(&bias)?, &v, b, t)
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
        prefix_len: usize,
    ) -> Result<Tensor> {
        self.forward_causal_compressed_sparse_windowed(
            x,
            local_window,
            compress_rate.max(1),
            index_topk.max(1),
            prefix_len,
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
        prefix_len: usize,
    ) -> Result<Tensor> {
        self.forward_causal_heavily_compressed_windowed(
            x,
            local_window,
            compress_rate.max(1),
            prefix_len,
        )
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
        let q = self.apply_rope_if_enabled(q, 0)?;
        let k = self.apply_rope_if_enabled(k, 0)?;
        Ok((q, k, v))
    }

    fn project_q_with_offset(&self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(&x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        self.apply_rope_if_enabled(q, position_offset)
    }

    pub fn project_self_kv(&self, x: &Tensor) -> Result<AttentionKvCache> {
        self.project_self_kv_with_prefix(x, 0)
    }

    pub fn project_self_kv_with_prefix(
        &self,
        x: &Tensor,
        prefix_len: usize,
    ) -> Result<AttentionKvCache> {
        self.project_self_kv_with_prefix_and_offset(x, prefix_len, 0)
    }

    fn project_self_kv_with_prefix_and_offset(
        &self,
        x: &Tensor,
        prefix_len: usize,
        position_offset: usize,
    ) -> Result<AttentionKvCache> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let k = self
            .k_proj
            .forward(&x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.apply_rope_if_enabled(k, position_offset)?;
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
        let position_offset = cache.map(|cache| cache.token_count).unwrap_or(0);
        let q = self.project_q_with_offset(&x, position_offset)?;
        let new_kv = self.project_self_kv_with_prefix_and_offset(&x, 0, position_offset)?;
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
        let bias = cache_causal_bias(&full_kv, position_offset, t_q, x.device())?
            .to_dtype(scores.dtype())?;
        let attn_weights = candle_nn::ops::softmax(&scores.broadcast_add(&bias)?, D::Minus1)?
            .reshape((b * self.num_heads, t_q, t_kv))?;
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
        Ok((out, full_kv.detached()))
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
        let position_offset = cache.map(|cache| cache.token_count).unwrap_or(0);
        let q = self.project_q_with_offset(&x, position_offset)?;
        let new_kv = self.project_self_kv_with_prefix_and_offset(&x, 0, position_offset)?;
        let full_kv =
            append_to_compressed_cache(cache, new_kv, exact_tail.max(1), compress_rate.max(1))?;

        let exact_bias = cache_causal_bias(&full_kv, position_offset, t_q, x.device())?;
        let attn_output =
            if let (Some(comp_k), Some(comp_v)) = (&full_kv.compressed_k, &full_kv.compressed_v) {
                let comp_len = comp_k.dim(2)?;
                let causal_comp_bias = compressed_cache_block_bias(
                    position_offset,
                    t_q,
                    &full_kv.compressed_block_ends,
                    x.device(),
                )?
                .broadcast_as((b, self.num_heads, t_q, comp_len))?;
                self.compressed_sparse_attention_output(
                    &q,
                    &full_kv.k,
                    &full_kv.v,
                    &exact_bias,
                    Some((comp_k, comp_v, &causal_comp_bias)),
                    index_topk.unwrap_or(comp_len).max(1),
                    "compressed incremental attention q/k",
                )?
            } else {
                self.compressed_sparse_attention_output(
                    &q,
                    &full_kv.k,
                    &full_kv.v,
                    &exact_bias,
                    None,
                    1,
                    "compressed incremental attention q/k",
                )?
            };
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t_q,
            self.num_heads * self.head_dim,
        ))?;
        let out = self
            .out_proj
            .forward(&out)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok((out, full_kv.detached()))
    }

    fn forward_local_windowed(
        &self,
        x: &Tensor,
        window: usize,
        causal: bool,
        key_padding_mask: Option<&Tensor>,
        prefix_len: usize,
    ) -> Result<Tensor> {
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
            let exact_prefix_len = prefix_len.min(t);
            let include_prefix =
                causal && chunk_contains_non_prefix_queries(q_start, q_len, exact_prefix_len);
            let local_kv_start = if include_prefix {
                kv_start.max(exact_prefix_len)
            } else {
                kv_start
            };
            let local_kv_len = kv_end.saturating_sub(local_kv_start).max(1);

            let q_chunk = q.narrow(2, q_start, q_len)?.contiguous()?;
            let local_k = k.narrow(2, local_kv_start, local_kv_len)?;
            let local_v = v.narrow(2, local_kv_start, local_kv_len)?;
            let mut k_parts = Vec::new();
            let mut v_parts = Vec::new();
            let mut bias_parts = Vec::new();
            if include_prefix {
                k_parts.push(k.narrow(2, 0, exact_prefix_len)?);
                v_parts.push(v.narrow(2, 0, exact_prefix_len)?);
                let mut prefix_bias =
                    Tensor::zeros((1, 1, q_len, exact_prefix_len), DType::F32, x.device())?;
                if let Some(mask) = key_padding_mask {
                    prefix_bias = prefix_bias.broadcast_add(&key_padding_bias(
                        &mask.narrow(1, 0, exact_prefix_len)?,
                        b,
                        self.num_heads,
                        q_len,
                        exact_prefix_len,
                        DType::F32,
                    )?)?;
                }
                bias_parts.push(prefix_bias);
            }
            k_parts.push(local_k);
            v_parts.push(local_v);
            let mut local_bias = local_chunk_bias(
                q_start,
                q_len,
                local_kv_start,
                local_kv_len,
                window,
                causal,
                x.device(),
            )?;
            if let Some(mask) = key_padding_mask {
                local_bias = local_bias.broadcast_add(&key_padding_bias(
                    &mask.narrow(1, local_kv_start, local_kv_len)?,
                    b,
                    self.num_heads,
                    q_len,
                    local_kv_len,
                    DType::F32,
                )?)?;
            }
            bias_parts.push(local_bias);
            let k_chunk = Tensor::cat(&k_parts, 2)?;
            let v_chunk = Tensor::cat(&v_parts, 2)?;
            let bias = Tensor::cat(&bias_parts, 3)?;
            util::ensure_same_dtype(&q_chunk, &k_chunk, "local attention q/k")?;
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

    fn forward_causal_compressed_sparse_windowed(
        &self,
        x: &Tensor,
        local_window: usize,
        compress_rate: usize,
        index_topk: usize,
        prefix_len: usize,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        if t <= local_window.max(1) {
            return self.forward_local_windowed(&x, local_window.max(1), true, None, prefix_len);
        }

        let query_block = decoder_attention_query_block(local_window.max(1), t);
        let local_radius = local_window.saturating_sub(1);
        let exact_prefix_len = prefix_len.min(t);
        let mut outputs = Vec::new();

        for q_start in (0..t).step_by(query_block) {
            let q_len = (t - q_start).min(query_block);
            let local_start_raw = q_start.saturating_sub(local_radius);
            let local_end = q_start + q_len;
            let include_prefix =
                chunk_contains_non_prefix_queries(q_start, q_len, exact_prefix_len);
            let local_start = if include_prefix {
                local_start_raw.max(exact_prefix_len)
            } else {
                local_start_raw
            };
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

            let mut exact_k_parts = Vec::new();
            let mut exact_v_parts = Vec::new();
            let mut exact_bias_parts = Vec::new();
            if include_prefix {
                exact_k_parts.push(k.narrow(2, 0, exact_prefix_len)?);
                exact_v_parts.push(v.narrow(2, 0, exact_prefix_len)?);
                exact_bias_parts.push(Tensor::zeros(
                    (b, self.num_heads, q_len, exact_prefix_len),
                    DType::F32,
                    x.device(),
                )?);
            }
            exact_k_parts.push(local_k);
            exact_v_parts.push(local_v);
            exact_bias_parts.push(local_bias);
            let exact_k = Tensor::cat(&exact_k_parts, 2)?;
            let exact_v = Tensor::cat(&exact_v_parts, 2)?;
            let exact_bias = Tensor::cat(&exact_bias_parts, 3)?;

            let compressed = compressed_causal_blocks_for_local_queries(
                &k,
                &v,
                exact_prefix_len,
                q_start,
                q_len,
                local_window.max(1),
                compress_rate,
            )?;
            let attn_output = if let Some((comp_k, comp_v, block_ends)) = compressed {
                let causal_bias = compressed_local_block_bias(
                    q_start,
                    q_len,
                    local_window.max(1),
                    &block_ends,
                    x.device(),
                )?
                .broadcast_as((b, self.num_heads, q_len, block_ends.len()))?;
                self.compressed_sparse_attention_output(
                    &q_chunk,
                    &exact_k,
                    &exact_v,
                    &exact_bias,
                    Some((&comp_k, &comp_v, &causal_bias)),
                    index_topk,
                    "compressed causal attention q/k",
                )?
            } else {
                self.compressed_sparse_attention_output(
                    &q_chunk,
                    &exact_k,
                    &exact_v,
                    &exact_bias,
                    None,
                    index_topk,
                    "compressed causal attention q/k",
                )?
            };
            outputs.push(attn_output);
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
        prefix_len: usize,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_self_qkv(&x)?;
        if t <= local_window.max(1) {
            return self.forward_local_windowed(&x, local_window.max(1), true, None, prefix_len);
        }

        let query_block = decoder_attention_query_block(local_window.max(1), t);
        let local_radius = local_window.saturating_sub(1);
        let exact_prefix_len = prefix_len.min(t);
        let mut outputs = Vec::new();

        for q_start in (0..t).step_by(query_block) {
            let q_len = (t - q_start).min(query_block);
            let local_start_raw = q_start.saturating_sub(local_radius);
            let local_end = q_start + q_len;
            let include_prefix =
                chunk_contains_non_prefix_queries(q_start, q_len, exact_prefix_len);
            let local_start = if include_prefix {
                local_start_raw.max(exact_prefix_len)
            } else {
                local_start_raw
            };
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

            let mut k_parts = Vec::new();
            let mut v_parts = Vec::new();
            let mut bias_parts = Vec::new();
            if include_prefix {
                k_parts.push(k.narrow(2, 0, exact_prefix_len)?);
                v_parts.push(v.narrow(2, 0, exact_prefix_len)?);
                bias_parts.push(Tensor::zeros(
                    (b, self.num_heads, q_len, exact_prefix_len),
                    DType::F32,
                    x.device(),
                )?);
            }
            k_parts.push(local_k);
            v_parts.push(local_v);
            bias_parts.push(local_bias.broadcast_as((b, self.num_heads, q_len, local_len))?);
            let compressed = compressed_causal_blocks_for_local_queries(
                &k,
                &v,
                exact_prefix_len,
                q_start,
                q_len,
                local_window.max(1),
                compress_rate,
            )?;
            if let Some((comp_k, comp_v, block_ends)) = compressed {
                k_parts.push(comp_k);
                v_parts.push(comp_v);
                bias_parts.push(
                    compressed_local_block_bias(
                        q_start,
                        q_len,
                        local_window.max(1),
                        &block_ends,
                        x.device(),
                    )?
                    .broadcast_as((
                        b,
                        self.num_heads,
                        q_len,
                        block_ends.len(),
                    ))?,
                );
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

    #[allow(clippy::too_many_arguments)]
    fn compressed_sparse_attention_output(
        &self,
        q: &Tensor,
        exact_k: &Tensor,
        exact_v: &Tensor,
        exact_bias: &Tensor,
        compressed: Option<(&Tensor, &Tensor, &Tensor)>,
        index_topk: usize,
        dtype_label: &str,
    ) -> Result<Tensor> {
        let (batch, heads, queries, _) = q.dims4()?;
        let exact_len = exact_k.dim(2)?;
        util::ensure_same_dtype(q, exact_k, dtype_label)?;
        let exact_scores =
            (q.matmul(&exact_k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / self.scale)?;
        let exact_scores =
            exact_scores.broadcast_add(&exact_bias.to_dtype(exact_scores.dtype())?)?;

        if let Some((comp_k, comp_v, comp_bias)) = compressed {
            util::ensure_same_dtype(q, comp_k, dtype_label)?;
            let comp_len = comp_k.dim(2)?;
            let comp_k_t = comp_k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            let comp_index_scores = q.matmul(&comp_k_t)?;
            let comp_scores = comp_index_scores.affine(1.0 / self.scale, 0.0)?;
            let comp_bias = if comp_len > 0 && index_topk < comp_len {
                let topk_bias = dsa_topk_bias_from_scores(
                    &comp_index_scores,
                    comp_bias,
                    index_topk,
                    q.device(),
                )?;
                comp_bias.broadcast_add(&topk_bias)?
            } else {
                comp_bias.clone()
            };
            let comp_scores =
                comp_scores.broadcast_add(&comp_bias.to_dtype(comp_scores.dtype())?)?;
            let scores = Tensor::cat(&[exact_scores, comp_scores], D::Minus1)?;
            let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let exact_weights = attn_weights.narrow(D::Minus1, 0, exact_len)?;
            let exact_out = exact_weights.contiguous()?.matmul(&exact_v.contiguous()?)?;
            let comp_weights = attn_weights.narrow(D::Minus1, exact_len, comp_len)?;
            let comp_out = comp_weights.contiguous()?.matmul(&comp_v.contiguous()?)?;
            let output = exact_out.broadcast_add(&comp_out)?;
            debug_assert_eq!(output.dims(), &[batch, heads, queries, self.head_dim]);
            return Ok(output);
        }

        let attn_weights = candle_nn::ops::softmax(&exact_scores, D::Minus1)?;
        let output = attn_weights.contiguous()?.matmul(&exact_v.contiguous()?)?;
        debug_assert_eq!(output.dims(), &[batch, heads, queries, self.head_dim]);
        Ok(output)
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

    fn apply_rope_if_enabled(&self, x: Tensor, position_offset: usize) -> Result<Tensor> {
        if self.use_rope {
            apply_rotary_pos_emb(x, position_offset, 10_000.0)
        } else {
            Ok(x)
        }
    }
}

fn apply_rotary_pos_emb(x: Tensor, position_offset: usize, base: f64) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = x.dims4()?;
    if seq_len == 0 {
        return Ok(x);
    }
    if !head_dim.is_multiple_of(2) {
        anyhow::bail!("RoPE requires an even attention head dimension, got {head_dim}");
    }
    let half = head_dim / 2;
    let mut cos_values = Vec::with_capacity(seq_len * half);
    let mut sin_values = Vec::with_capacity(seq_len * half);
    for pos in position_offset..position_offset + seq_len {
        for idx in 0..half {
            let freq = base.powf(-2.0 * idx as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos_values.push(angle.cos() as f32);
            sin_values.push(angle.sin() as f32);
        }
    }
    let cos =
        Tensor::from_vec(cos_values, (1, 1, seq_len, half), x.device())?.to_dtype(x.dtype())?;
    let sin =
        Tensor::from_vec(sin_values, (1, 1, seq_len, half), x.device())?.to_dtype(x.dtype())?;
    let pairs = x.reshape((batch, heads, seq_len, half, 2))?;
    let even = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let even_cos = even.broadcast_mul(&cos)?;
    let odd_sin = odd.broadcast_mul(&sin)?;
    let rot_even = (&even_cos - &odd_sin)?;
    let even_sin = even.broadcast_mul(&sin)?;
    let odd_cos = odd.broadcast_mul(&cos)?;
    let rot_odd = (&even_sin + &odd_cos)?;
    Tensor::stack(&[&rot_even, &rot_odd], 4)?
        .reshape((batch, heads, seq_len, head_dim))
        .map_err(Into::into)
}

/// Transformer block with self-attention, layer norm, and feed-forward
pub struct TransformerBlock {
    attn: MultiHeadAttention,
    norm1: nn::RmsNorm,
    norm2: nn::RmsNorm,
    ff_gate: nn::Linear,
    ff_up: nn::Linear,
    ff_down: nn::Linear,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        let attn = MultiHeadAttention::new_with_rope(vb.pp("attn"), dim, num_heads)?;
        let norm1 = nn::rms_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = nn::rms_norm(dim, 1e-6, vb.pp("norm2"))?;
        // A two-branch SwiGLU needs three matrices.  Using 2/3 of the old
        // GELU width preserves the block's parameter budget while improving
        // the activation and gating path.
        let swiglu_dim = ((ff_dim * 2) / 3).max(dim);
        let ff_gate = nn::linear_no_bias(dim, swiglu_dim, vb.pp("ff_gate"))?;
        let ff_up = nn::linear_no_bias(dim, swiglu_dim, vb.pp("ff_up"))?;
        let ff_down = nn::linear_no_bias(swiglu_dim, dim, vb.pp("ff_down"))?;

        Ok(Self {
            attn,
            norm1,
            norm2,
            ff_gate,
            ff_up,
            ff_down,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.norm1.forward_diff(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let x = (x + attn_out)?;

        let normed = self.norm2.forward_diff(&x)?;
        let ff_out = self
            .ff_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.ff_up.forward(&normed)?)?;
        let ff_out = self.ff_down.forward(&ff_out)?;
        Ok((x + ff_out)?)
    }

    pub fn forward_masked(&self, x: &Tensor, key_padding_mask: &Tensor) -> Result<Tensor> {
        let normed = self.norm1.forward_diff(x)?;
        let attn_out = self.attn.forward_self_masked(&normed, key_padding_mask)?;
        let x = (x + attn_out)?;

        let normed = self.norm2.forward_diff(&x)?;
        let ff_out = self
            .ff_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.ff_up.forward(&normed)?)?;
        let ff_out = self.ff_down.forward(&ff_out)?;
        Ok((x + ff_out)?)
    }
}

pub struct LocalTransformerBlock {
    attn: MultiHeadAttention,
    norm1: nn::RmsNorm,
    norm2: nn::RmsNorm,
    ff_gate: nn::Linear,
    ff_up: nn::Linear,
    ff_down: nn::Linear,
}

impl LocalTransformerBlock {
    pub fn new(
        vb: VarBuilder<'_>,
        dim: usize,
        num_heads: usize,
        ff_dim: usize,
        _window: usize,
    ) -> Result<Self> {
        let attn = MultiHeadAttention::new_with_rope(vb.pp("attn"), dim, num_heads)?;
        let norm1 = nn::rms_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = nn::rms_norm(dim, 1e-6, vb.pp("norm2"))?;
        let swiglu_dim = ((ff_dim * 2) / 3).max(dim);
        let ff_gate = nn::linear_no_bias(dim, swiglu_dim, vb.pp("ff_gate"))?;
        let ff_up = nn::linear_no_bias(dim, swiglu_dim, vb.pp("ff_up"))?;
        let ff_down = nn::linear_no_bias(swiglu_dim, dim, vb.pp("ff_down"))?;

        Ok(Self {
            attn,
            norm1,
            norm2,
            ff_gate,
            ff_up,
            ff_down,
        })
    }

    pub fn forward_with_window(&self, x: &Tensor, window: usize) -> Result<Tensor> {
        let normed = self.norm1.forward_diff(x)?;
        let attn_out = self.attn.forward_local(&normed, window.max(1))?;
        let x = (x + attn_out)?;

        let normed = self.norm2.forward_diff(&x)?;
        let ff_out = self
            .ff_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.ff_up.forward(&normed)?)?;
        let ff_out = self.ff_down.forward(&ff_out)?;
        Ok((x + ff_out)?)
    }

    pub fn forward_with_window_masked(
        &self,
        x: &Tensor,
        window: usize,
        key_padding_mask: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward_diff(x)?;
        let attn_out = self
            .attn
            .forward_local_masked(&normed, window.max(1), key_padding_mask)?;
        let x = (x + attn_out)?;

        let normed = self.norm2.forward_diff(&x)?;
        let ff_out = self
            .ff_gate
            .forward(&normed)?
            .silu()?
            .broadcast_mul(&self.ff_up.forward(&normed)?)?;
        let ff_out = self.ff_down.forward(&ff_out)?;
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
        if decoder_dim == 0 || world_dim == 0 || num_heads == 0 {
            anyhow::bail!("cross-attention dimensions and head count must be non-zero");
        }
        if !decoder_dim.is_multiple_of(num_heads) {
            anyhow::bail!(
                "cross-attention decoder_dim {decoder_dim} must be divisible by {num_heads} heads"
            );
        }
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

    pub fn new_no_bias(
        vb: VarBuilder<'_>,
        decoder_dim: usize,
        world_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        Self::new_bottleneck_no_bias(vb, decoder_dim, world_dim, decoder_dim, num_heads)
    }

    /// Parameter-efficient cross-attention with an internal attention width.
    ///
    /// Q maps `decoder_dim -> attention_dim`, K/V map
    /// `world_dim -> attention_dim`, and O maps back to `decoder_dim`.
    /// At frozen-LLM injection sites this avoids four full hidden-width
    /// projections per site without reducing the decoder residual width.
    pub fn new_bottleneck_no_bias(
        vb: VarBuilder<'_>,
        decoder_dim: usize,
        world_dim: usize,
        attention_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        if decoder_dim == 0 || world_dim == 0 || attention_dim == 0 || num_heads == 0 {
            anyhow::bail!("cross-attention dimensions and head count must be non-zero");
        }
        if !attention_dim.is_multiple_of(num_heads) {
            anyhow::bail!(
                "cross-attention attention_dim {attention_dim} must be divisible by {num_heads} heads"
            );
        }
        let head_dim = attention_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(),
            q_proj: nn::linear_no_bias(decoder_dim, attention_dim, vb.pp("q_proj"))?,
            k_proj: nn::linear_no_bias(world_dim, attention_dim, vb.pp("k_proj"))?,
            v_proj: nn::linear_no_bias(world_dim, attention_dim, vb.pp("v_proj"))?,
            out_proj: nn::linear_no_bias(attention_dim, decoder_dim, vb.pp("out_proj"))?,
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
            .affine(-ATTENTION_MASK_VALUE as f64, ATTENTION_MASK_VALUE as f64)?
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

/// Causal mask for decoder self-attention: [1, 1, seq_len, seq_len], (i,j) = 0 if j <= i else a large negative bias.
#[allow(dead_code)]
fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            v[i * seq_len + j] = ATTENTION_MASK_VALUE;
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
                v[qi * kv_len + kj] = ATTENTION_MASK_VALUE;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, q_len, kv_len), device).map_err(|e| anyhow::anyhow!("{:?}", e))
}

fn chunk_contains_non_prefix_queries(q_start: usize, q_len: usize, prefix_len: usize) -> bool {
    prefix_len > 0 && q_start.saturating_add(q_len) > prefix_len
}

fn key_padding_bias(
    mask: &Tensor,
    batch: usize,
    heads: usize,
    queries: usize,
    keys: usize,
    dtype: DType,
) -> Result<Tensor> {
    mask.contiguous()?
        .affine(-ATTENTION_MASK_VALUE as f64, ATTENTION_MASK_VALUE as f64)?
        .unsqueeze(1)?
        .unsqueeze(1)?
        .broadcast_as((batch, heads, queries, keys))?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn cache_causal_bias(
    cache: &AttentionKvCache,
    query_start: usize,
    query_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let (batch, heads, key_len, _) = cache.k.dims4()?;
    let prefix_len = cache.prefix_len.min(key_len);
    let dynamic_len = key_len.saturating_sub(prefix_len);
    let dynamic_start = cache.token_count.saturating_sub(dynamic_len);
    let mut v = vec![0f32; query_len * key_len];
    for qi in 0..query_len {
        let q_abs = query_start + qi;
        for kj in prefix_len..key_len {
            let k_abs = dynamic_start + (kj - prefix_len);
            if k_abs > q_abs {
                v[qi * key_len + kj] = ATTENTION_MASK_VALUE;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, query_len, key_len), device)?
        .broadcast_as((batch, heads, query_len, key_len))
        .map_err(Into::into)
}

fn compressed_cache_block_bias(
    query_start: usize,
    query_len: usize,
    block_ends: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    if block_ends.is_empty() {
        return Tensor::zeros((1, 1, query_len, 0), DType::F32, device).map_err(Into::into);
    }
    let mut v = vec![0f32; query_len * block_ends.len()];
    for qi in 0..query_len {
        let visible_until = query_start + qi + 1;
        for (bi, &block_end) in block_ends.iter().enumerate() {
            if block_end > visible_until {
                v[qi * block_ends.len() + bi] = ATTENTION_MASK_VALUE;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, query_len, block_ends.len()), device).map_err(Into::into)
}

#[cfg(test)]
fn compressed_causal_blocks_from(
    k: &Tensor,
    v: &Tensor,
    start_at: usize,
    upto: usize,
    compress_rate: usize,
) -> Result<Option<(Tensor, Tensor, Vec<usize>)>> {
    let (_, _, t, _) = k.dims4()?;
    let start_at = start_at.min(t);
    let upto = upto.min(t);
    if start_at >= upto || compress_rate <= 1 {
        return Ok(None);
    }

    let mut k_blocks = Vec::new();
    let mut v_blocks = Vec::new();
    let mut block_ends = Vec::new();
    let mut start = start_at;
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

fn compressed_causal_blocks_for_local_queries(
    k: &Tensor,
    v: &Tensor,
    start_at: usize,
    q_start: usize,
    q_len: usize,
    local_window: usize,
    compress_rate: usize,
) -> Result<Option<(Tensor, Tensor, Vec<usize>)>> {
    let (_, _, t, _) = k.dims4()?;
    let start_at = start_at.min(t);
    let radius = local_window.saturating_sub(1);
    let mut forced_boundaries = Vec::with_capacity(q_len);
    for qi in 0..q_len {
        let boundary = (q_start + qi).saturating_sub(radius).clamp(start_at, t);
        if boundary > start_at {
            forced_boundaries.push(boundary);
        }
    }
    forced_boundaries.sort_unstable();
    forced_boundaries.dedup();
    let upto = forced_boundaries.last().copied().unwrap_or(start_at);
    if start_at >= upto || compress_rate <= 1 {
        return Ok(None);
    }

    let mut k_blocks = Vec::new();
    let mut v_blocks = Vec::new();
    let mut block_ends = Vec::new();
    let mut start = start_at;
    while start < upto {
        let next_boundary = forced_boundaries
            .iter()
            .copied()
            .find(|boundary| *boundary > start)
            .unwrap_or(upto);
        let end = (start + compress_rate).min(upto).min(next_boundary);
        let len = end.saturating_sub(start).max(1);
        let scale = 1.0 / len as f64;
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

fn compressed_local_block_bias(
    q_start: usize,
    q_len: usize,
    local_window: usize,
    block_ends: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let radius = local_window.saturating_sub(1);
    let mut v = vec![0f32; q_len * block_ends.len()];
    for qi in 0..q_len {
        let exact_left = (q_start + qi).saturating_sub(radius);
        for (bi, &block_end) in block_ends.iter().enumerate() {
            if block_end > exact_left {
                v[qi * block_ends.len() + bi] = ATTENTION_MASK_VALUE;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn test_attention_input(
        batch: usize,
        seq: usize,
        dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let values = (0..batch * seq * dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (batch, seq, dim), device).map_err(Into::into)
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) -> Result<()> {
        let max_diff = util::scalar_f32(&a.broadcast_sub(b)?.abs()?.max_all()?)?;
        assert!(max_diff < tol, "{label} mismatch: {max_diff}");
        Ok(())
    }

    #[test]
    fn cache_causal_bias_keeps_prefix_visible_and_masks_future_chunk_keys() -> Result<()> {
        let device = Device::Cpu;
        let k = Tensor::zeros((1, 1, 4, 1), DType::F32, &device)?;
        let v = Tensor::zeros((1, 1, 4, 1), DType::F32, &device)?;
        let cache = AttentionKvCache {
            k,
            v,
            compressed_k: None,
            compressed_v: None,
            compressed_block_ends: Vec::new(),
            token_count: 6,
            prefix_len: 2,
        };

        let bias = cache_causal_bias(&cache, 4, 2, &device)?.reshape((2, 4))?;
        let bias = util::vec2_f32(&bias)?;

        assert_eq!(bias[0], vec![0.0, 0.0, 0.0, ATTENTION_MASK_VALUE]);
        assert_eq!(bias[1], vec![0.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn compressed_cache_block_bias_masks_blocks_ending_after_query() -> Result<()> {
        let device = Device::Cpu;
        let bias = compressed_cache_block_bias(4, 2, &[3, 5, 6], &device)?.reshape((2, 3))?;
        let bias = util::vec2_f32(&bias)?;

        assert_eq!(bias[0], vec![0.0, 0.0, ATTENTION_MASK_VALUE]);
        assert_eq!(bias[1], vec![0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn compressed_local_block_bias_excludes_exact_window_overlap() -> Result<()> {
        let device = Device::Cpu;
        let bias =
            compressed_local_block_bias(4, 2, 3, &[1, 2, 3, 4, 5], &device)?.reshape((2, 5))?;
        let bias = util::vec2_f32(&bias)?;

        assert_eq!(
            bias[0],
            vec![
                0.0,
                0.0,
                ATTENTION_MASK_VALUE,
                ATTENTION_MASK_VALUE,
                ATTENTION_MASK_VALUE
            ]
        );
        assert_eq!(
            bias[1],
            vec![0.0, 0.0, 0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE]
        );
        Ok(())
    }

    #[test]
    fn topk_bias_masks_non_selected_compressed_entries() -> Result<()> {
        let device = Device::Cpu;
        let scores = Tensor::from_vec(vec![0.1f32, 0.5, -1.0, 0.4, 0.3], (1, 1, 1, 5), &device)?;
        let causal_bias = Tensor::zeros((1, 1, 1, 5), DType::F32, &device)?;

        let bias = dsa_topk_bias_from_scores(&scores, &causal_bias, 2, &device)?.reshape((5,))?;
        let bias = util::vec1_f32(&bias)?;

        assert_eq!(
            bias,
            vec![
                ATTENTION_MASK_VALUE,
                0.0,
                ATTENTION_MASK_VALUE,
                0.0,
                ATTENTION_MASK_VALUE
            ]
        );
        Ok(())
    }

    #[test]
    fn topk_bias_masks_future_before_selecting() -> Result<()> {
        let device = Device::Cpu;
        let scores = Tensor::from_vec(vec![0.1f32, 100.0, 0.5], (1, 1, 1, 3), &device)?;
        let causal_bias = Tensor::from_vec(
            vec![0.0f32, ATTENTION_MASK_VALUE, 0.0],
            (1, 1, 1, 3),
            &device,
        )?;

        let bias = dsa_topk_bias_from_scores(&scores, &causal_bias, 1, &device)?.reshape((3,))?;
        let bias = util::vec1_f32(&bias)?;

        assert_eq!(bias, vec![ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE, 0.0]);
        Ok(())
    }

    #[test]
    fn topk_bias_uses_shared_deepseek_style_selector_across_heads() -> Result<()> {
        let device = Device::Cpu;
        let scores = Tensor::from_vec(
            vec![
                10.0f32, 0.0, 0.0, //
                0.0, 5.0, 0.0,
            ],
            (1, 2, 1, 3),
            &device,
        )?;
        let causal_bias = Tensor::zeros((1, 2, 1, 3), DType::F32, &device)?;

        let bias = dsa_topk_bias_from_scores(&scores, &causal_bias, 1, &device)?.reshape((2, 3))?;
        let bias = util::vec2_f32(&bias)?;

        assert_eq!(
            bias,
            vec![
                vec![0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE],
                vec![0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE],
            ]
        );
        Ok(())
    }

    #[test]
    fn prefix_is_visible_for_chunks_that_cross_prefix_boundary() {
        assert!(!chunk_contains_non_prefix_queries(0, 4, 4));
        assert!(chunk_contains_non_prefix_queries(0, 5, 4));
        assert!(chunk_contains_non_prefix_queries(4, 2, 4));
        assert!(!chunk_contains_non_prefix_queries(0, 5, 0));
    }

    #[test]
    fn compressed_causal_blocks_from_keeps_prefix_out_of_blocks() -> Result<()> {
        let device = Device::Cpu;
        let values = (0..6).map(|value| value as f32).collect::<Vec<_>>();
        let k = Tensor::from_vec(values.clone(), (1, 1, 6, 1), &device)?;
        let v = Tensor::from_vec(values, (1, 1, 6, 1), &device)?;

        let (compressed_k, _, block_ends) =
            compressed_causal_blocks_from(&k, &v, 2, 6, 2)?.expect("compressed blocks");

        assert_eq!(block_ends, vec![4, 6]);
        let compressed = util::vec2_f32(&compressed_k.reshape((2, 1))?)?;
        assert_eq!(compressed, vec![vec![2.5], vec![4.5]]);
        Ok(())
    }

    #[test]
    fn compressed_incremental_cache_merges_partial_blocks() -> Result<()> {
        let device = Device::Cpu;
        let values = (0..6).map(|value| value as f32).collect::<Vec<_>>();
        let k = Tensor::from_vec(values.clone(), (1, 1, 6, 1), &device)?;
        let v = Tensor::from_vec(values, (1, 1, 6, 1), &device)?;
        let cache = compress_cache_from_full_kv(k, v, 2, 3, 0)?;

        assert_eq!(cache.compressed_block_ends, vec![3, 4]);
        let compressed = cache
            .compressed_k
            .as_ref()
            .expect("compressed k")
            .reshape((2, 1))?;
        let compressed = util::vec2_f32(&compressed)?;
        assert_eq!(compressed, vec![vec![1.0], vec![3.0]]);

        let new_k = Tensor::from_vec(vec![6.0f32], (1, 1, 1, 1), &device)?;
        let new_v = Tensor::from_vec(vec![6.0f32], (1, 1, 1, 1), &device)?;
        let cache =
            append_to_compressed_cache(Some(&cache), attention_kv_cache(new_k, new_v)?, 2, 3)?;

        assert_eq!(cache.compressed_block_ends, vec![3, 5]);
        let compressed = cache
            .compressed_k
            .as_ref()
            .expect("compressed k")
            .reshape((2, 1))?;
        let compressed = util::vec2_f32(&compressed)?;
        assert_eq!(compressed, vec![vec![1.0], vec![3.5]]);

        let new_k = Tensor::from_vec(vec![7.0f32], (1, 1, 1, 1), &device)?;
        let new_v = Tensor::from_vec(vec![7.0f32], (1, 1, 1, 1), &device)?;
        let cache =
            append_to_compressed_cache(Some(&cache), attention_kv_cache(new_k, new_v)?, 2, 3)?;

        assert_eq!(cache.compressed_block_ends, vec![3, 6]);
        let compressed = cache
            .compressed_k
            .as_ref()
            .expect("compressed k")
            .reshape((2, 1))?;
        let compressed = util::vec2_f32(&compressed)?;
        assert_eq!(compressed, vec![vec![1.0], vec![4.0]]);

        let new_k = Tensor::from_vec(vec![8.0f32], (1, 1, 1, 1), &device)?;
        let new_v = Tensor::from_vec(vec![8.0f32], (1, 1, 1, 1), &device)?;
        let cache =
            append_to_compressed_cache(Some(&cache), attention_kv_cache(new_k, new_v)?, 2, 3)?;

        assert_eq!(cache.compressed_block_ends, vec![3, 6, 7]);
        let compressed = cache
            .compressed_k
            .as_ref()
            .expect("compressed k")
            .reshape((3, 1))?;
        let compressed = util::vec2_f32(&compressed)?;
        assert_eq!(compressed, vec![vec![1.0], vec![4.0], vec![6.0]]);
        Ok(())
    }

    #[test]
    fn rope_sliding_incremental_matches_full_last_token_with_prefix() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let attn = MultiHeadAttention::new_with_rope(
            VarBuilder::from_varmap(&varmap, DType::F32, &device).pp("attn"),
            8,
            2,
        )?;
        let prefix_len = 3;
        let window = 4;
        let x = test_attention_input(1, 11, 8, &device)?;

        let full = attn
            .forward_causal_local_with_prefix(&x, window, prefix_len)?
            .narrow(1, 10, 1)?;
        let prefill = x.narrow(1, 0, 10)?;
        let last = x.narrow(1, 10, 1)?;
        let cache = attn.project_self_kv_with_prefix(&prefill, prefix_len)?;
        let (incremental, _) =
            attn.forward_causal_local_incremental(&last, Some(&cache), window)?;

        assert_close(&full, &incremental, 1e-4, "rope sliding incremental")?;
        Ok(())
    }

    #[test]
    fn compressed_sparse_incremental_matches_full_last_token_with_prefix() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let attn = MultiHeadAttention::new_with_rope(
            VarBuilder::from_varmap(&varmap, DType::F32, &device).pp("attn"),
            8,
            2,
        )?;
        let prefix_len = 2;
        let local_window = 3;
        let compress_rate = 2;
        let topk = 4;
        let x = test_attention_input(1, 12, 8, &device)?;

        let full = attn
            .forward_causal_compressed_sparse(&x, local_window, compress_rate, topk, prefix_len)?
            .narrow(1, 11, 1)?;
        let prefill = x.narrow(1, 0, 11)?;
        let last = x.narrow(1, 11, 1)?;
        let cache =
            attn.project_self_kv_compressed(&prefill, local_window, compress_rate, prefix_len)?;
        let (incremental, _) = attn.forward_causal_compressed_sparse_incremental(
            &last,
            Some(&cache),
            local_window,
            compress_rate,
            topk,
        )?;

        assert_close(&full, &incremental, 1e-4, "compressed sparse incremental")?;
        Ok(())
    }
}
