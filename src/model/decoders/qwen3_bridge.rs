//! Frozen Qwen3 decoder with trainable gated latent cross-attention sites.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{self as nn, Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;
use candle_transformers::utils::repeat_kv;

use crate::model::attention::{AttentionKvCache, CrossAttention};
use crate::util;

struct SelfAttention {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    q_lora: Option<Lora>,
    v_lora: Option<Lora>,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_sin: Tensor,
    rope_cos: Tensor,
}

struct Lora {
    a: nn::Linear,
    b: nn::Linear,
    scale: f64,
}

impl Lora {
    fn new(input: usize, output: usize, rank: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            a: nn::linear_no_bias(input, rank, vb.pp("a"))?,
            b: nn::Linear::new(
                vb.get_with_hints((output, rank), "b.weight", nn::Init::Const(0.0))?,
                None,
            ),
            scale: 16.0 / rank as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.b
            .forward(&self.a.forward(x)?)?
            .affine(self.scale, 0.0)
            .map_err(Into::into)
    }
}

#[derive(Clone)]
struct LayerKvCache {
    k: Tensor,
    v: Tensor,
}

pub struct Qwen3BridgeCache {
    layers: Vec<Option<LayerKvCache>>,
    cross: Vec<Option<AttentionKvCache>>,
    position: usize,
}

impl SelfAttention {
    fn new(
        cfg: &Config,
        vb: VarBuilder<'_>,
        lora_vb: Option<VarBuilder<'_>>,
        lora_rank: usize,
    ) -> Result<Self> {
        let linear = |input, output, vb| {
            if cfg.attention_bias {
                nn::linear(input, output, vb)
            } else {
                nn::linear_no_bias(input, output, vb)
            }
        };
        let half = cfg.head_dim / 2;
        let inv = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64) as f32)
            .collect::<Vec<_>>();
        let inv = Tensor::from_vec(inv, (1, half), vb.device())?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freq = positions.matmul(&inv)?;
        Ok(Self {
            q: linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("q_proj"),
            )?,
            k: linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?,
            v: linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?,
            o: linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_lora: lora_vb
                .as_ref()
                .map(|vb| {
                    Lora::new(
                        cfg.hidden_size,
                        cfg.num_attention_heads * cfg.head_dim,
                        lora_rank,
                        vb.pp("q"),
                    )
                })
                .transpose()?,
            v_lora: lora_vb
                .as_ref()
                .map(|vb| {
                    Lora::new(
                        cfg.hidden_size,
                        cfg.num_key_value_heads * cfg.head_dim,
                        lora_rank,
                        vb.pp("v"),
                    )
                })
                .transpose()?,
            q_norm: nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_sin: freq.sin()?.to_dtype(vb.dtype())?,
            rope_cos: freq.cos()?.to_dtype(vb.dtype())?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let q_proj = self.q.forward(x)?;
        let q_proj = match &self.q_lora {
            Some(lora) => q_proj.broadcast_add(&lora.forward(x)?)?,
            None => q_proj,
        };
        let q = self
            .q_norm
            .forward_diff(
                &q_proj
                    .reshape((batch, seq, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .flatten(0, 2)?,
            )?
            .reshape((batch, self.heads, seq, self.head_dim))?;
        let k = self
            .k_norm
            .forward_diff(
                &self
                    .k
                    .forward(x)?
                    .reshape((batch, seq, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .flatten(0, 2)?,
            )?
            .reshape((batch, self.kv_heads, seq, self.head_dim))?;
        let v_proj = self.v.forward(x)?;
        let v_proj = match &self.v_lora {
            Some(lora) => v_proj.broadcast_add(&lora.forward(x)?)?,
            None => v_proj,
        };
        let v = v_proj
            .reshape((batch, seq, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let sin = self.rope_sin.narrow(0, 0, seq)?;
        let cos = self.rope_cos.narrow(0, 0, seq)?;
        // Candle's fused RoPE and last-dimension softmax are forward-only.
        // Keep this training path on differentiable tensor operations.
        let q = candle_nn::rotary_emb::rope_slow(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_slow(&k, &cos, &sin)?;
        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(k, groups)?.contiguous()?;
        let v = repeat_kv(v, groups)?.contiguous()?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?
            .broadcast_add(mask)?;
        let values = nn::ops::softmax(&scores, candle_core::D::Minus1)?.matmul(&v)?;
        self.o
            .forward(
                &values
                    .transpose(1, 2)?
                    .reshape((batch, seq, self.heads * self.head_dim))?,
            )
            .map_err(Into::into)
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        previous: Option<&LayerKvCache>,
        position: usize,
    ) -> Result<(Tensor, LayerKvCache)> {
        let (batch, seq, _) = x.dims3()?;
        let q_proj = self.q.forward(x)?;
        let q_proj = match &self.q_lora {
            Some(lora) => q_proj.broadcast_add(&lora.forward(x)?)?,
            None => q_proj,
        };
        let q = self
            .q_norm
            .forward(
                &q_proj
                    .reshape((batch, seq, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .flatten(0, 2)?,
            )?
            .reshape((batch, self.heads, seq, self.head_dim))?;
        let k = self
            .k_norm
            .forward(
                &self
                    .k
                    .forward(x)?
                    .reshape((batch, seq, self.kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .flatten(0, 2)?,
            )?
            .reshape((batch, self.kv_heads, seq, self.head_dim))?;
        let v_proj = self.v.forward(x)?;
        let v_proj = match &self.v_lora {
            Some(lora) => v_proj.broadcast_add(&lora.forward(x)?)?,
            None => v_proj,
        };
        let v = v_proj
            .reshape((batch, seq, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let sin = self.rope_sin.narrow(0, position, seq)?;
        let cos = self.rope_cos.narrow(0, position, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        let (k, v) = if let Some(cache) = previous {
            (
                Tensor::cat(&[cache.k.clone(), k], 2)?,
                Tensor::cat(&[cache.v.clone(), v], 2)?,
            )
        } else {
            (k, v)
        };
        let total = k.dim(2)?;
        let groups = self.heads / self.kv_heads;
        let scores = q
            .matmul(
                &repeat_kv(k.clone(), groups)?
                    .contiguous()?
                    .transpose(2, 3)?,
            )?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let scores = if seq > 1 {
            let mask = (0..seq)
                .flat_map(|i| {
                    (0..total).map(move |j| {
                        if j <= position + i {
                            0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect::<Vec<_>>();
            scores.broadcast_add(
                &Tensor::from_vec(mask, (1, 1, seq, total), x.device())?.to_dtype(x.dtype())?,
            )?
        } else {
            scores
        };
        let values = nn::ops::softmax_last_dim(&scores)?
            .matmul(&repeat_kv(v.clone(), groups)?.contiguous()?)?;
        let output = self.o.forward(&values.transpose(1, 2)?.reshape((
            batch,
            seq,
            self.heads * self.head_dim,
        ))?)?;
        Ok((
            output,
            LayerKvCache {
                k: k.detach(),
                v: v.detach(),
            },
        ))
    }
}

struct Mlp {
    gate: nn::Linear,
    up: nn::Linear,
    down: nn::Linear,
    activation: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            activation: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.apply(&self.activation)?;
        self.down
            .forward(&(gate * self.up.forward(x)?)?)
            .map_err(Into::into)
    }
}

struct CrossSite {
    norm: nn::RmsNorm,
    attention: CrossAttention,
    gate: nn::Linear,
}

impl CrossSite {
    fn new(dim: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let attention_dim = std::env::var("TOFY_QWEN_CROSS_DIM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| dim.min(512));
        let heads = [8, 4, 2, 1]
            .into_iter()
            .find(|heads| attention_dim.is_multiple_of(*heads))
            .unwrap_or(1);
        Ok(Self {
            norm: nn::rms_norm(dim, eps, vb.pp("norm"))?,
            attention: CrossAttention::new_bottleneck_no_bias(
                vb.pp("attention"),
                dim,
                dim,
                attention_dim,
                heads,
            )?,
            gate: nn::Linear::new(
                vb.get_with_hints((1, dim), "gate.weight", nn::Init::Const(0.0))?,
                Some(vb.get_with_hints(1, "gate.bias", nn::Init::Const(-4.0))?),
            ),
        })
    }

    fn forward(&self, x: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward_diff(x)?;
        let gate = nn::ops::sigmoid(&self.gate.forward(&normed)?)?;
        let update = self
            .attention
            .forward(&normed, conditioning)?
            .broadcast_mul(&gate)?;
        (x + update).map_err(Into::into)
    }

    fn forward_with_gate_stats(
        &self,
        x: &Tensor,
        conditioning: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = self.norm.forward_diff(x)?;
        let gate = nn::ops::sigmoid(&self.gate.forward(&normed)?)?;
        let update = self
            .attention
            .forward(&normed, conditioning)?
            .broadcast_mul(&gate)?;
        Ok(((x + update)?, gate.mean_all()?, gate.max_all()?))
    }

    fn project_conditioning(&self, conditioning: &Tensor) -> Result<AttentionKvCache> {
        Ok(self.attention.project_kv(conditioning)?.detached())
    }

    fn forward_precomputed(&self, x: &Tensor, conditioning: &AttentionKvCache) -> Result<Tensor> {
        let normed = self.norm.forward_diff(x)?;
        let gate = nn::ops::sigmoid(&self.gate.forward(&normed)?)?;
        let update = self
            .attention
            .forward_precomputed(&normed, conditioning, None)?
            .broadcast_mul(&gate)?;
        (x + update).map_err(Into::into)
    }
}

struct Layer {
    attention: SelfAttention,
    mlp: Mlp,
    input_norm: nn::RmsNorm,
    post_norm: nn::RmsNorm,
    cross: Option<CrossSite>,
}

/// Base tensors come from `base_vb` and remain outside the optimizer VarMap.
/// Only adapter/cross-attention tensors should be created from `train_vb`.
pub struct Qwen3Bridge {
    tokens: nn::Embedding,
    layers: Vec<Layer>,
    norm: nn::RmsNorm,
    lm_head: Option<nn::Linear>,
    device: Device,
    dtype: DType,
}

fn tied_lm_head(hidden: &Tensor, embeddings: &Tensor) -> Result<Tensor> {
    let (batch, seq, dim) = hidden.dims3()?;
    let vocab = embeddings.dim(0)?;
    hidden
        .reshape((batch * seq, dim))?
        .matmul(&embeddings.t()?)?
        .reshape((batch, seq, vocab))
        .map_err(Into::into)
}

impl Qwen3Bridge {
    pub fn new(cfg: &Config, base_vb: VarBuilder<'_>, train_vb: VarBuilder<'_>) -> Result<Self> {
        let every = std::env::var("TOFY_QWEN_CROSS_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);
        if every == 0 {
            bail!("TOFY_QWEN_CROSS_EVERY must be non-zero");
        }
        let lora_rank = std::env::var("TOFY_QWEN_LORA_RANK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let tokens = nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            base_vb.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let vb = base_vb.pp(format!("model.layers.{index}"));
            layers.push(Layer {
                attention: SelfAttention::new(
                    cfg,
                    vb.pp("self_attn"),
                    (lora_rank > 0).then(|| train_vb.pp(format!("lora_layers.{index}"))),
                    lora_rank,
                )?,
                mlp: Mlp::new(cfg, vb.pp("mlp"))?,
                input_norm: nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("input_layernorm"),
                )?,
                post_norm: nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_attention_layernorm"),
                )?,
                cross: if lora_rank == 0 && (index + 1) % every == 0 {
                    Some(CrossSite::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        train_vb.pp(format!("cross_sites.{index}")),
                    )?)
                } else {
                    None
                },
            });
        }
        let lm_head = (!cfg.tie_word_embeddings)
            .then(|| nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, base_vb.pp("lm_head")))
            .transpose()?;
        Ok(Self {
            tokens,
            layers,
            norm: nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, base_vb.pp("model.norm"))?,
            lm_head,
            device: base_vb.device().clone(),
            dtype: base_vb.dtype(),
        })
    }

    pub fn forward(&self, input: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let (batch, seq) = input.dims2()?;
        let mask = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if j <= i { 0f32 } else { f32::NEG_INFINITY }))
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, seq, seq), &self.device)?
            .to_dtype(self.dtype)?
            .broadcast_as((batch, 1, seq, seq))?;
        let mut x = self.tokens.forward(input)?;
        for layer in &self.layers {
            let update = layer
                .attention
                .forward(&layer.input_norm.forward_diff(&x)?, &mask)?;
            x = (x + update)?;
            if let Some(cross) = &layer.cross {
                x = cross.forward(&x, conditioning)?;
            }
            let update = layer.mlp.forward(&layer.post_norm.forward_diff(&x)?)?;
            x = (x + update)?;
        }
        let hidden = self.norm.forward_diff(&x)?;
        match &self.lm_head {
            Some(head) => head.forward(&hidden).map_err(Into::into),
            None => tied_lm_head(&hidden, self.tokens.embeddings()),
        }
    }

    /// Frozen token representations used to align world latents with the
    /// decoder's native embedding space before generative bridge training.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.tokens.forward(token_ids).map_err(Into::into)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn gate_statistics(
        &self,
        input: &Tensor,
        conditioning: &Tensor,
    ) -> Result<Vec<(usize, f32, f32)>> {
        let (batch, seq) = input.dims2()?;
        let mask = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if j <= i { 0f32 } else { f32::NEG_INFINITY }))
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, seq, seq), &self.device)?
            .to_dtype(self.dtype)?
            .broadcast_as((batch, 1, seq, seq))?;
        let mut x = self.tokens.forward(input)?;
        let mut stats = Vec::new();
        let mut site_index = 0usize;
        for layer in &self.layers {
            let update = layer
                .attention
                .forward(&layer.input_norm.forward_diff(&x)?, &mask)?;
            x = (x + update)?;
            if let Some(cross) = &layer.cross {
                let (next, mean, max) = cross.forward_with_gate_stats(&x, conditioning)?;
                x = next;
                stats.push((
                    site_index,
                    util::scalar_f32(&mean)?,
                    util::scalar_f32(&max)?,
                ));
                site_index += 1;
            }
            let update = layer.mlp.forward(&layer.post_norm.forward_diff(&x)?)?;
            x = (x + update)?;
        }
        Ok(stats)
    }

    pub fn new_cache(&self) -> Qwen3BridgeCache {
        Qwen3BridgeCache {
            layers: vec![None; self.layers.len()],
            cross: vec![None; self.layers.len()],
            position: 0,
        }
    }

    pub fn forward_cached(
        &self,
        input: &Tensor,
        conditioning: &Tensor,
        cache: &mut Qwen3BridgeCache,
    ) -> Result<Tensor> {
        let (_, seq) = input.dims2()?;
        let mut x = self.tokens.forward(input)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let (update, next_cache) = layer.attention.forward_cached(
                &layer.input_norm.forward(&x)?,
                cache.layers[index].as_ref(),
                cache.position,
            )?;
            cache.layers[index] = Some(next_cache);
            x = (x + update)?;
            if let Some(cross) = &layer.cross {
                if cache.cross[index].is_none() {
                    cache.cross[index] = Some(cross.project_conditioning(conditioning)?);
                }
                x = cross.forward_precomputed(
                    &x,
                    cache.cross[index]
                        .as_ref()
                        .expect("cross cache initialized"),
                )?;
            }
            let update = layer.mlp.forward(&layer.post_norm.forward(&x)?)?;
            x = (x + update)?;
        }
        cache.position += seq;
        let hidden = self.norm.forward(&x)?;
        match &self.lm_head {
            Some(head) => head.forward(&hidden).map_err(Into::into),
            None => tied_lm_head(&hidden, self.tokens.embeddings()),
        }
    }
}

pub use candle_transformers::models::qwen3::Config as Qwen3Config;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use candle_nn::VarMap;

    fn test_config() -> Config {
        Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            head_dim: 4,
            attention_bias: false,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    fn causal_mask(seq: usize, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            (0..seq)
                .flat_map(|i| (0..seq).map(move |j| if j <= i { 0f32 } else { f32::NEG_INFINITY }))
                .collect::<Vec<_>>(),
            (1, 1, seq, seq),
            device,
        )
        .map_err(Into::into)
    }

    #[test]
    fn zero_conditioning_is_exact_identity() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let site = CrossSite::new(8, 1e-6, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
        let input = Tensor::randn(0f32, 1f32, (2, 3, 8), &device)?;
        let zero = Tensor::zeros((2, 4, 8), DType::F32, &device)?;
        let output = site.forward(&input, &zero)?;
        assert_eq!(input.to_vec3::<f32>()?, output.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn cross_site_trainables_receive_gradients() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let site = CrossSite::new(8, 1e-6, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
        let input = Tensor::randn(0f32, 1f32, (2, 3, 8), &device)?;
        let conditioning = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        let output = site.forward(&input, &conditioning)?;
        let gradients = output.sqr()?.sum_all()?.backward()?;
        let data = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("varmap lock poisoned"))?;
        for (name, var) in data.iter() {
            if name == "norm.weight" {
                continue;
            }
            let grad = gradients
                .get(var)
                .with_context(|| format!("missing gradient for {name}"))?;
            let magnitude = util::scalar_f32(&grad.abs()?.sum_all()?)?;
            assert!(magnitude > 0.0, "zero gradient for {name}");
        }
        Ok(())
    }

    #[test]
    fn lora_projection_receives_gradients() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let lora = Lora::new(8, 8, 2, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
        let input = Tensor::randn(0f32, 1f32, (2, 3, 8), &device)?;
        let grads = lora.forward(&input)?.sqr()?.sum_all()?.backward()?;
        let data = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("varmap lock poisoned"))?;
        for (name, var) in data.iter() {
            assert!(grads.get(var).is_some(), "missing LoRA gradient for {name}");
        }
        Ok(())
    }

    #[test]
    fn self_attention_lora_receives_gradients_through_attention() -> Result<()> {
        let device = Device::Cpu;
        let base_vars = VarMap::new();
        let lora_vars = VarMap::new();
        let attention = SelfAttention::new(
            &test_config(),
            VarBuilder::from_varmap(&base_vars, DType::F32, &device),
            Some(VarBuilder::from_varmap(&lora_vars, DType::F32, &device)),
            2,
        )?;
        let input = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
        let mask = causal_mask(4, &device)?;
        let grads = attention
            .forward(&input, &mask)?
            .sqr()?
            .sum_all()?
            .backward()?;
        let data = lora_vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("varmap lock poisoned"))?;
        for name in ["q.b.weight", "v.b.weight"] {
            let grad = grads
                .get(data.get(name).with_context(|| format!("missing {name}"))?)
                .with_context(|| format!("missing gradient for {name}"))?;
            let magnitude = util::scalar_f32(&grad.abs()?.sum_all()?)?;
            assert!(magnitude > 0.0, "zero gradient for {name}");
        }
        Ok(())
    }

    /// Guards the fused-kernel severing class of bug. Candle's `rotary_emb::rope`
    /// and `ops::softmax_last_dim` return graph leaves, so with them the query
    /// path carries no gradient at all: `q_proj`/`q_norm` would be missing from
    /// the gradient store and the attention input would only be reached through
    /// the value projection. Both properties are asserted here.
    #[test]
    fn self_attention_forward_keeps_the_query_path_differentiable() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let attention = SelfAttention::new(
            &test_config(),
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            None,
            0,
        )?;
        let x = candle_core::Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?)?;
        let mask = causal_mask(4, &device)?;
        let grads = attention
            .forward(x.as_tensor(), &mask)?
            .sqr()?
            .sum_all()?
            .backward()?;
        let input_grad = grads
            .get(&x)
            .context("gradient did not reach the attention input")?;
        assert!(
            util::scalar_f32(&input_grad.abs()?.sum_all()?)? > 0.0,
            "zero gradient at the attention input"
        );
        let data = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("varmap lock poisoned"))?;
        // Reachable only through the attention scores, i.e. RoPE and the softmax.
        for name in ["q_proj.weight", "q_norm.weight", "k_norm.weight"] {
            let grad = grads
                .get(data.get(name).with_context(|| format!("missing {name}"))?)
                .with_context(|| format!("severed gradient path for {name}"))?;
            let magnitude = util::scalar_f32(&grad.abs()?.sum_all()?)?;
            assert!(magnitude > 0.0, "zero gradient for {name}");
        }
        Ok(())
    }

    /// End-to-end contract for bridge training: the loss taken through the full
    /// training forward must differentiate the conditioning tensor and every
    /// cross-site trainable. A severed forward silently reports zero here.
    #[test]
    fn training_forward_backpropagates_into_conditioning() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = test_config();
        // The default `TOFY_QWEN_CROSS_EVERY` of 4 puts one cross site on the
        // last layer; the test avoids setting the env var because it is
        // process-global and other decoder tests run concurrently.
        cfg.num_hidden_layers = 4;
        let base_vars = VarMap::new();
        let train_vars = VarMap::new();
        let model = Qwen3Bridge::new(
            &cfg,
            VarBuilder::from_varmap(&base_vars, DType::F32, &device),
            VarBuilder::from_varmap(&train_vars, DType::F32, &device),
        )?;
        assert!(
            model.layers.iter().any(|layer| layer.cross.is_some()),
            "test config produced no cross site"
        );
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device)?;
        let conditioning =
            candle_core::Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 3, 8), &device)?)?;
        let grads = model
            .forward(&input, conditioning.as_tensor())?
            .sqr()?
            .sum_all()?
            .backward()?;
        let conditioning_grad = grads
            .get(&conditioning)
            .context("gradient did not reach the conditioning input")?;
        assert!(
            util::scalar_f32(&conditioning_grad.abs()?.sum_all()?)? > 0.0,
            "zero gradient at the conditioning input"
        );
        let data = train_vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("varmap lock poisoned"))?;
        assert!(!data.is_empty(), "no trainable variables were created");
        for (name, var) in data.iter() {
            let grad = grads
                .get(var)
                .with_context(|| format!("missing gradient for {name}"))?;
            let magnitude = util::scalar_f32(&grad.abs()?.sum_all()?)?;
            assert!(magnitude > 0.0, "zero gradient for {name}");
        }
        Ok(())
    }

    #[test]
    fn cached_attention_detaches_kv_state() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let attention = SelfAttention::new(
            &test_config(),
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            None,
            0,
        )?;
        let input = candle_core::Var::from_tensor(&Tensor::randn(0f32, 1f32, (1, 3, 8), &device)?)?;
        let (_, cache) = attention.forward_cached(input.as_tensor(), None, 0)?;
        let cache_grads = cache.k.sqr()?.sum_all()?.backward()?;
        assert!(cache_grads.get(&input).is_none());
        Ok(())
    }

    #[test]
    fn cached_attention_matches_full_attention() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let cfg = test_config();
        let attention = SelfAttention::new(
            &cfg,
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            None,
            0,
        )?;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
        let mask = causal_mask(4, &device)?;
        let full = attention.forward(&x, &mask)?.narrow(1, 3, 1)?;
        let (_, cache) = attention.forward_cached(&x.narrow(1, 0, 3)?, None, 0)?;
        let (cached, _) = attention.forward_cached(&x.narrow(1, 3, 1)?, Some(&cache), 3)?;
        let error = util::scalar_f32(&(&full - &cached)?.abs()?.max_all()?)?;
        assert!(error < 1e-5, "cached attention max error {error}");
        Ok(())
    }
}
