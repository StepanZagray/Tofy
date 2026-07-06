//! Training entry point for the frozen-Qwen knowledge bridge.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::data::{encode_world_examples, RawWorldExample};
use crate::model::decoders::{Qwen3Bridge, Qwen3Config};
use crate::model::{
    load_vocab_from_file, ContextCompressor, DecoderConditioningAdapter, OnlineEncoder,
};
use crate::tasks::veclab::{
    attach_docs, load_docs_map, load_task_rows, VeclabTaskRow, SEEN_FUNCTION_MAX,
};
use crate::tasks::world_context::{
    context_slots_from_world_pair_sequences, env_bool, env_f64, env_usize,
};
use crate::tasks::world_support::{
    batch_shuffled_conditioning_latent, hard_mismatched_conditioning_latent, masked_cross_entropy,
};
use crate::util;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConditioningNegatives {
    pub zero: bool,
    pub shuffle: bool,
    pub hard: bool,
}

impl ConditioningNegatives {
    pub fn none() -> Self {
        Self {
            zero: false,
            shuffle: false,
            hard: false,
        }
    }
    pub fn from_env() -> Self {
        let value = std::env::var("TOFY_DECODER_CONDITIONING_NEGATIVES")
            .unwrap_or_else(|_| "hard".to_string());
        let mut out = Self::none();
        for part in value.split(',').map(|v| v.trim().to_ascii_lowercase()) {
            match part.as_str() {
                "all" => {
                    out = Self {
                        zero: true,
                        shuffle: true,
                        hard: true,
                    }
                }
                "zero" | "ablated" => out.zero = true,
                "shuffle" | "shuffled" => out.shuffle = true,
                "hard" | "hard_mismatch" | "mismatch" => out.hard = true,
                "" | "none" => {}
                other => println!("Ignoring unknown conditioning negative: {other}"),
            }
        }
        out
    }
}

pub(crate) fn add_conditioning_margin_loss(
    existing: Option<Tensor>,
    positive: &Tensor,
    negative: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    let value = positive
        .broadcast_sub(negative)?
        .affine(1.0, margin)?
        .relu()?;
    match existing {
        Some(loss) => loss.broadcast_add(&value).map_err(Into::into),
        None => Ok(value),
    }
}

struct BridgeArgs {
    qwen_dir: PathBuf,
    encoder: PathBuf,
    vocab: PathBuf,
    world: PathBuf,
    data: PathBuf,
    output: PathBuf,
    steps: usize,
    batch: usize,
    resume: bool,
    seed: u64,
}

impl BridgeArgs {
    fn parse(args: &[String]) -> Result<Self> {
        if args.len() < 7 {
            bail!("usage: --train-bridge <qwen_dir> <encoder.safetensors> <encoder_vocab.txt> <world.safetensors> <tasks.txt> [steps] [batch] [output]");
        }
        let resume = args.iter().any(|arg| arg == "--resume");
        let seed = args
            .iter()
            .position(|arg| arg == "--seed")
            .and_then(|index| args.get(index + 1))
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(42);
        let positional = args[2..]
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                value.as_str() != "--resume"
                    && value.as_str() != "--seed"
                    && (*index == 0 || args[index + 1] != "--seed")
            })
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        if positional.len() < 5 {
            bail!("usage: --train-bridge <qwen_dir> <encoder.safetensors> <encoder_vocab.txt> <world.safetensors> <tasks.txt> [steps] [batch] [output] [--resume] [--seed N]");
        }
        Ok(Self {
            qwen_dir: PathBuf::from(positional[0]),
            encoder: PathBuf::from(positional[1]),
            vocab: PathBuf::from(positional[2]),
            world: PathBuf::from(positional[3]),
            data: PathBuf::from(positional[4]),
            steps: positional
                .get(5)
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000),
            batch: positional.get(6).and_then(|v| v.parse().ok()).unwrap_or(2),
            output: positional
                .get(7)
                .map(PathBuf::from)
                .unwrap_or_else(|| "local_models/qwen_bridge.safetensors".into()),
            resume,
            seed,
        })
    }
}

struct BridgeSampler {
    row_count: usize,
    seed: u64,
    epoch: usize,
    cursor: usize,
    order: Vec<usize>,
}

impl BridgeSampler {
    fn at_sample(row_count: usize, seed: u64, sample: usize) -> Self {
        let epoch = sample / row_count;
        let cursor = sample % row_count;
        let mut sampler = Self {
            row_count,
            seed,
            epoch,
            cursor,
            order: Vec::new(),
        };
        sampler.shuffle_epoch();
        sampler
    }

    fn shuffle_epoch(&mut self) {
        self.order = (0..self.row_count).collect();
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.epoch as u64));
        self.order.shuffle(&mut rng);
    }

    fn next_batch(&mut self, batch_size: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(batch_size);
        while indices.len() < batch_size {
            indices.push(self.order[self.cursor]);
            self.cursor += 1;
            if self.cursor == self.row_count {
                self.epoch += 1;
                self.cursor = 0;
                self.shuffle_epoch();
            }
        }
        indices
    }
}

pub(crate) fn qwen_weight_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|v| v.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        bail!("no safetensors files found in {}", dir.display());
    }
    Ok(paths)
}

fn qwen_batch(
    tokenizer: &Tokenizer,
    rows: &[VeclabTaskRow],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let encoded = rows
        .iter()
        .map(|row| {
            let prompt = tokenizer
                .encode(row.task.clone(), false)
                .map_err(anyhow::Error::msg)?;
            let completion = tokenizer
                .encode(row.completion.clone(), false)
                .map_err(anyhow::Error::msg)?;
            let mut ids = prompt.get_ids().to_vec();
            let prompt_len = ids.len();
            ids.extend_from_slice(completion.get_ids());
            Ok((ids, prompt_len))
        })
        .collect::<Result<Vec<_>>>()?;
    let max_len = encoded
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let pad = tokenizer.get_padding().map(|v| v.pad_id).unwrap_or(0);
    let mut inputs = vec![pad; encoded.len() * max_len];
    let mut labels = vec![pad; encoded.len() * (max_len - 1)];
    let mut mask = vec![0f32; encoded.len() * (max_len - 1)];
    for (batch, (ids, prompt_len)) in encoded.iter().enumerate() {
        let offset = batch * max_len;
        inputs[offset..offset + ids.len()].copy_from_slice(ids);
        let label_offset = batch * (max_len - 1);
        labels[label_offset..label_offset + ids.len().saturating_sub(1)].copy_from_slice(&ids[1..]);
        for index in prompt_len.saturating_sub(1)..ids.len().saturating_sub(1) {
            mask[label_offset + index] = 1.0;
        }
    }
    Ok((
        Tensor::from_vec(inputs, (encoded.len(), max_len), device)?,
        Tensor::from_vec(labels, (encoded.len(), max_len - 1), device)?,
        Tensor::from_vec(mask, (encoded.len(), max_len - 1), device)?,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BridgeRegime {
    Context,
    Weights,
}

impl BridgeRegime {
    fn from_env() -> Result<Self> {
        match std::env::var("TOFY_BRIDGE_REGIME")
            .unwrap_or_else(|_| "weights".into())
            .as_str()
        {
            "context" => Ok(Self::Context),
            "weights" => Ok(Self::Weights),
            value => bail!("TOFY_BRIDGE_REGIME must be context or weights, got {value}"),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Context => "context",
            Self::Weights => "weights",
        }
    }
}

fn world_rows(rows: &[VeclabTaskRow], regime: BridgeRegime) -> Vec<RawWorldExample> {
    rows.iter()
        .map(|row| RawWorldExample {
            state_text: match regime {
                BridgeRegime::Context => format!(
                    "Relevant veclab documentation:\n{}\n\nTask:\n{}",
                    row.docs, row.task
                ),
                BridgeRegime::Weights => row.task.clone(),
            },
            next_text: row.completion.clone(),
            action_label: 0,
        })
        .collect()
}

pub(crate) fn state_conditioning(
    rows: &[VeclabTaskRow],
    regime: BridgeRegime,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    vocab: &crate::model::Vocab,
    max_seq: usize,
    device: &Device,
) -> Result<Tensor> {
    let raw = world_rows(rows, regime);
    let encoded = encode_world_examples(&raw, vocab);
    let (state_slots, _) = context_slots_from_world_pair_sequences(
        encoder,
        compressor,
        &encoded,
        vocab.pad_id,
        max_seq,
        4,
        1,
        device,
    )?;
    debug_assert!(
        raw.iter()
            .zip(rows)
            .all(|(input, row)| row.completion.is_empty()
                || !input.state_text.contains(&row.completion)),
        "conditioning state must exclude gold completions"
    );
    Ok(state_slots)
}

pub(crate) struct BridgeRuntime {
    pub tokenizer: Tokenizer,
    pub model: Qwen3Bridge,
    pub encoder: OnlineEncoder,
    pub compressor: ContextCompressor,
    pub adapter: DecoderConditioningAdapter,
    static_prefix: Option<candle_nn::Embedding>,
    pub vocab: crate::model::Vocab,
    pub regime: BridgeRegime,
    pub max_seq: usize,
    pub device: Device,
    output_slots: usize,
    hidden_size: usize,
}

impl BridgeRuntime {
    pub fn load(
        qwen_dir: &Path,
        bridge_path: &Path,
        encoder_path: &Path,
        vocab_path: &Path,
        world_path: &Path,
    ) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let cfg: Qwen3Config =
            serde_json::from_str(&fs::read_to_string(qwen_dir.join("config.json"))?)?;
        let tokenizer =
            Tokenizer::from_file(qwen_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let weights = qwen_weight_paths(qwen_dir)?;
        let base_vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
        let mut bridge_map = VarMap::new();
        let bridge_vb = VarBuilder::from_varmap(&bridge_map, dtype, &device);
        let model = Qwen3Bridge::new(&cfg, base_vb, bridge_vb.pp("qwen_bridge"))?;
        let planner_dim = env_usize("TOFY_BRIDGE_DIM", 640);
        let output_slots = env_usize("TOFY_ADAPTER_OUTPUT_SLOTS", 64);
        let adapter = DecoderConditioningAdapter::new(
            bridge_vb.pp("adapter"),
            planner_dim,
            cfg.hidden_size,
            output_slots,
        )?;
        let static_prefix = if env_bool("TOFY_STATIC_SOFT_PREFIX", false) {
            Some(candle_nn::embedding(
                output_slots,
                cfg.hidden_size,
                bridge_vb.pp("static_prefix"),
            )?)
        } else {
            None
        };
        util::load_varmap_checked(&mut bridge_map, bridge_path)?;
        let vocab = load_vocab_from_file(vocab_path)?;
        let dim = env_usize("TOFY_ENCODER_DIM", 768);
        let layers = env_usize("TOFY_ENCODER_LAYERS", 9);
        let heads = env_usize("TOFY_ENCODER_HEADS", 8);
        let slots = env_usize("TOFY_NUM_LATENT_TOKENS", 64);
        let mut encoder_map = VarMap::new();
        let encoder = OnlineEncoder::new(
            VarBuilder::from_varmap(&encoder_map, dtype, &device).pp("encoder"),
            vocab.id_to_token.len(),
            dim,
            layers,
            heads,
        )?;
        util::load_varmap_checked(&mut encoder_map, encoder_path)?;
        let mut world_map = VarMap::new();
        let compressor = ContextCompressor::new(
            VarBuilder::from_varmap(&world_map, dtype, &device).pp("context_compressor"),
            dim,
            planner_dim,
            slots,
        )?;
        let unfrozen = bridge_path.with_extension("world.safetensors");
        util::load_varmap_checked(
            &mut world_map,
            if unfrozen.exists() {
                &unfrozen
            } else {
                world_path
            },
        )?;
        Ok(Self {
            tokenizer,
            model,
            encoder,
            compressor,
            adapter,
            static_prefix,
            vocab,
            regime: BridgeRegime::from_env()?,
            max_seq: env_usize("TOFY_BRIDGE_MAX_SEQ", 512),
            device,
            output_slots,
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn conditioning(&self, rows: &[VeclabTaskRow]) -> Result<Tensor> {
        let slots = state_conditioning(
            rows,
            self.regime,
            &self.encoder,
            &self.compressor,
            &self.vocab,
            self.max_seq,
            &self.device,
        )?;
        if let Some(prefix) = &self.static_prefix {
            let ids = Tensor::arange(0u32, self.output_slots as u32, &self.device)?.unsqueeze(0)?;
            prefix
                .forward(&ids)?
                .broadcast_as((rows.len(), self.output_slots, self.hidden_size))
                .map_err(Into::into)
        } else {
            self.adapter.forward(&slots)
        }
    }

    pub fn generate(&self, prompt: &str, conditioning: &Tensor, max_new: usize) -> Result<String> {
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(anyhow::Error::msg)?;
        let mut ids = encoded.get_ids().to_vec();
        let prompt_len = ids.len();
        let eos = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"));
        let mut cache = self.model.new_cache();
        let mut next_input = ids.clone();
        for _ in 0..max_new {
            let input = Tensor::from_vec(next_input.clone(), (1, next_input.len()), &self.device)?;
            let logits = self
                .model
                .forward_cached(&input, conditioning, &mut cache)?;
            let next = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .to_dtype(DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            ids.push(next);
            if Some(next) == eos {
                break;
            }
            next_input.clear();
            next_input.push(next);
        }
        self.tokenizer
            .decode(&ids[prompt_len..], true)
            .map_err(anyhow::Error::msg)
    }
}

#[allow(clippy::too_many_arguments)]
fn val_losses(
    rows: &[VeclabTaskRow],
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    vocab: &crate::model::Vocab,
    adapter: &DecoderConditioningAdapter,
    static_prefix: Option<&candle_nn::Embedding>,
    model: &Qwen3Bridge,
    max_seq: usize,
    device: &Device,
    output_slots: usize,
    hidden_size: usize,
) -> Result<(f32, f32)> {
    let slots = state_conditioning(rows, regime, encoder, compressor, vocab, max_seq, device)?;
    let cond = if let Some(prefix) = static_prefix {
        let ids = Tensor::arange(0u32, output_slots as u32, device)?.unsqueeze(0)?;
        prefix
            .forward(&ids)?
            .broadcast_as((rows.len(), output_slots, hidden_size))?
    } else {
        adapter.forward(&slots)?
    };
    let (input, labels, mask) = qwen_batch(tokenizer, rows, device)?;
    let matched = token_loss(model, &input, &labels, &mask, &cond)?;
    let zero = token_loss(model, &input, &labels, &mask, &cond.zeros_like()?)?;
    Ok((util::scalar_f32(&matched)?, util::scalar_f32(&zero)?))
}

#[allow(clippy::too_many_arguments)]
fn full_val_losses(
    rows: &[VeclabTaskRow],
    batch_size: usize,
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    vocab: &crate::model::Vocab,
    adapter: &DecoderConditioningAdapter,
    static_prefix: Option<&candle_nn::Embedding>,
    model: &Qwen3Bridge,
    max_seq: usize,
    device: &Device,
    output_slots: usize,
    hidden_size: usize,
) -> Result<(f32, f32)> {
    let mut matched = 0.0;
    let mut zero = 0.0;
    for chunk in rows.chunks(batch_size.max(1)) {
        let (chunk_matched, chunk_zero) = val_losses(
            chunk,
            regime,
            tokenizer,
            encoder,
            compressor,
            vocab,
            adapter,
            static_prefix,
            model,
            max_seq,
            device,
            output_slots,
            hidden_size,
        )?;
        matched += chunk_matched * chunk.len() as f32;
        zero += chunk_zero * chunk.len() as f32;
    }
    Ok((matched / rows.len() as f32, zero / rows.len() as f32))
}

fn token_loss(
    model: &Qwen3Bridge,
    input: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
    cond: &Tensor,
) -> Result<Tensor> {
    let logits = model.forward(input, cond)?;
    masked_cross_entropy(&logits.narrow(1, 0, logits.dim(1)? - 1)?, labels, mask)
}

fn conditioning_health(conditioning: &Tensor) -> Result<(f32, f32)> {
    let (batch, slots, dim) = conditioning.dims3()?;
    let flat = conditioning.reshape((batch * slots, dim))?;
    let norm_mean = flat.sqr()?.sum(1)?.sqrt()?.mean_all()?;
    let mean = flat.mean(0)?;
    let std = flat
        .broadcast_sub(&mean.unsqueeze(0)?)?
        .sqr()?
        .mean(0)?
        .sqrt()?
        .mean_all()?;
    Ok((util::scalar_f32(&norm_mean)?, util::scalar_f32(&std)?))
}

pub fn try_run_train_bridge(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("--train-bridge")
        && args.get(1).map(String::as_str) != Some("train-bridge")
    {
        return Ok(false);
    }
    train(BridgeArgs::parse(args)?)?;
    Ok(true)
}

pub fn try_run_logit_parity(args: &[String]) -> Result<bool> {
    if !matches!(
        args.get(1).map(String::as_str),
        Some("--check-bridge-logit-parity" | "check-bridge-logit-parity")
    ) {
        return Ok(false);
    }
    let qwen_dir = args
        .get(2)
        .map(PathBuf::from)
        .context("usage: --check-bridge-logit-parity <qwen_dir> [prompt]")?;
    let prompt = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("Write a short Go function that returns the sum of two integers.");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let cfg: Qwen3Config =
        serde_json::from_str(&fs::read_to_string(qwen_dir.join("config.json"))?)?;
    let tokenizer =
        Tokenizer::from_file(qwen_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let weights = qwen_weight_paths(&qwen_dir)?;
    let base_vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let train_vars = VarMap::new();
    let model = Qwen3Bridge::new(
        &cfg,
        base_vb,
        VarBuilder::from_varmap(&train_vars, dtype, &device).pp("qwen_bridge"),
    )?;
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(anyhow::Error::msg)?;
    let ids = encoded.get_ids();
    if ids.len() < 2 {
        bail!("parity prompt must encode to at least two tokens");
    }
    let conditioning = Tensor::zeros((1, 4, cfg.hidden_size), dtype, &device)?;
    let full_input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &device)?;
    let full = model
        .forward(&full_input, &conditioning)?
        .narrow(1, ids.len() - 1, 1)?
        .squeeze(1)?
        .to_dtype(DType::F32)?;
    let mut cache = model.new_cache();
    let prefix = Tensor::from_vec(ids[..ids.len() - 1].to_vec(), (1, ids.len() - 1), &device)?;
    model.forward_cached(&prefix, &conditioning, &mut cache)?;
    let last = Tensor::from_vec(vec![ids[ids.len() - 1]], (1, 1), &device)?;
    let cached = model
        .forward_cached(&last, &conditioning, &mut cache)?
        .squeeze(1)?
        .to_dtype(DType::F32)?;
    let abs_error = full.broadcast_sub(&cached)?.abs()?;
    let max_abs_error = abs_error.max_all()?.to_scalar::<f32>()?;
    let mean_abs_error = abs_error.mean_all()?.to_scalar::<f32>()?;
    let full_argmax = full
        .argmax(candle_core::D::Minus1)?
        .squeeze(0)?
        .to_scalar::<u32>()?;
    let cached_argmax = cached
        .argmax(candle_core::D::Minus1)?
        .squeeze(0)?
        .to_scalar::<u32>()?;
    println!(
        "bridge logit parity: tokens={} dtype={dtype:?} mean_abs_error={mean_abs_error:.8} max_abs_error={max_abs_error:.8} full_argmax={full_argmax} cached_argmax={cached_argmax}",
        ids.len()
    );
    // BF16 GEMMs accumulate in a different order for full-sequence and
    // single-token shapes; the top token must match and sub-logit drift must
    // remain comfortably below one logit unit.
    if full_argmax != cached_argmax || max_abs_error > 0.5 {
        bail!("cached bridge logits failed parity check");
    }
    Ok(true)
}

fn train(args: BridgeArgs) -> Result<()> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let cfg: Qwen3Config =
        serde_json::from_str(&fs::read_to_string(args.qwen_dir.join("config.json"))?)?;
    let tokenizer =
        Tokenizer::from_file(args.qwen_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let weights = qwen_weight_paths(&args.qwen_dir)?;
    // SAFETY: safetensor files remain immutable and mapped for the lifetime of the model.
    let base_vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
    let mut train_vars = VarMap::new();
    let train_vb = VarBuilder::from_varmap(&train_vars, dtype, &device);
    let qwen = Qwen3Bridge::new(&cfg, base_vb, train_vb.pp("qwen_bridge"))?;

    let encoder_vocab = load_vocab_from_file(&args.vocab)?;
    crate::tasks::veclab::print_vocab_identifier_sanity(
        &encoder_vocab,
        Path::new("data/fictional/veclab_docs.txt"),
    )?;
    crate::tasks::prepare_veclab::print_split_stats(Path::new("data/fictional"))?;
    let dim = env_usize("TOFY_ENCODER_DIM", 768);
    let planner_dim = env_usize("TOFY_BRIDGE_DIM", 256);
    let slots = env_usize("TOFY_NUM_LATENT_TOKENS", 64);
    let encoder_layers = env_usize("TOFY_ENCODER_LAYERS", 9);
    let encoder_heads = env_usize("TOFY_ENCODER_HEADS", 8);
    let max_seq = env_usize("TOFY_BRIDGE_MAX_SEQ", 512);
    let mut encoder_map = VarMap::new();
    let encoder = OnlineEncoder::new(
        VarBuilder::from_varmap(&encoder_map, dtype, &device).pp("encoder"),
        encoder_vocab.id_to_token.len(),
        dim,
        encoder_layers,
        encoder_heads,
    )?;
    util::load_varmap_checked(&mut encoder_map, &args.encoder)?;
    let mut world_map = VarMap::new();
    let compressor = ContextCompressor::new(
        VarBuilder::from_varmap(&world_map, dtype, &device).pp("context_compressor"),
        dim,
        planner_dim,
        slots,
    )?;
    util::load_varmap_checked(&mut world_map, &args.world)?;
    let output_slots = env_usize("TOFY_ADAPTER_OUTPUT_SLOTS", 64);
    let adapter = DecoderConditioningAdapter::new(
        train_vb.pp("adapter"),
        planner_dim,
        cfg.hidden_size,
        output_slots,
    )?;
    let static_prefix = if env_bool("TOFY_STATIC_SOFT_PREFIX", false) {
        Some(candle_nn::embedding(
            output_slots,
            cfg.hidden_size,
            train_vb.pp("static_prefix"),
        )?)
    } else {
        None
    };

    let unfreeze_world = env_bool("TOFY_KNOWLEDGE_UNFREEZE_WORLD", false);
    let lr = env_f64("TOFY_BRIDGE_LR", 2e-4);
    let mut named_vars = util::named_train_vars(&train_vars)?;
    if unfreeze_world {
        named_vars.extend(util::named_train_vars(&world_map)?);
    }
    named_vars.sort_by(|a, b| a.name.cmp(&b.name));
    let trainable_params = named_vars
        .iter()
        .map(|entry| entry.var.elem_count())
        .sum::<usize>();
    let optimizer_vars = named_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    println!(
        "Bridge conditioning={} trainable_params={trainable_params}",
        if static_prefix.is_some() {
            "static_prefix"
        } else {
            "world"
        }
    );
    let mut optimizer = util::TrainOptimizer::new_lr_named(named_vars, lr)?;
    let negatives = ConditioningNegatives::from_env();
    let margin = env_f64("TOFY_DECODER_CONDITIONING_MARGIN", 0.2);
    let margin_weight = env_f64("TOFY_DECODER_CONDITIONING_MARGIN_WEIGHT", 0.25);
    let dropout = env_f64("TOFY_CONDITIONING_DROPOUT", 0.1).clamp(0.0, 1.0);
    let regime = BridgeRegime::from_env()?;
    let all_rows = load_task_rows(&args.data)?;
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt")).unwrap_or_default();
    let mut seen: Vec<_> = all_rows
        .into_iter()
        .filter(|row| row.function_id <= SEEN_FUNCTION_MAX)
        .collect();
    attach_docs(&mut seen, &docs);
    let (val_rows, train_rows): (Vec<_>, Vec<_>) = seen
        .into_iter()
        .enumerate()
        .partition(|(index, _)| index % 20 == 0);
    let train_rows = train_rows
        .into_iter()
        .map(|(_, row)| row)
        .collect::<Vec<_>>();
    let val_rows = val_rows.into_iter().map(|(_, row)| row).collect::<Vec<_>>();
    if train_rows.is_empty() || val_rows.is_empty() {
        bail!("bridge needs non-empty seen train and validation rows");
    }
    println!(
        "Bridge regime={} train_rows={} val_rows={} heldout_task_rows=0",
        regime.as_str(),
        train_rows.len(),
        val_rows.len()
    );
    if std::env::var("TOFY_PRINT_SPLIT_STATS")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        return Ok(());
    }
    let val_every = env_usize("TOFY_BRIDGE_VAL_EVERY", 100).max(1);
    let log_every = env_usize("TOFY_BRIDGE_LOG_EVERY", 10).max(1);
    let grad_accum = env_usize("TOFY_BRIDGE_GRAD_ACCUM", 1).max(1);
    let clip_norm = env_f64("TOFY_BRIDGE_CLIP_NORM", 1.0).max(0.0);
    let best_path = args.output.with_extension("best.safetensors");
    let latest_path = args.output.with_extension("latest.safetensors");
    let resume_stage = util::resume_stage_name("bridge");
    let optimizer_path =
        util::checkpoint_sidecar_path(&args.output, &resume_stage, "optimizer.safetensors");
    let resume_path = util::checkpoint_sidecar_path(&args.output, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if args.resume {
        if latest_path.exists() {
            util::load_varmap_checked(&mut train_vars, &latest_path)?;
        }
        let latest_world = latest_path.with_extension("world.safetensors");
        if unfreeze_world && latest_world.exists() {
            util::load_varmap_checked(&mut world_map, &latest_world)?;
        }
        if optimizer_path.exists() {
            optimizer.load_state(&optimizer_path)?;
        }
        if let Some(state) = util::load_resume_state(&resume_path, &resume_stage)? {
            resume_state = state;
        }
    }
    let start_step = if args.resume {
        resume_state.step.min(args.steps)
    } else {
        0
    };
    let mut best_ce = if args.resume {
        resume_state.best_metric
    } else {
        f32::INFINITY
    };
    let mut sampler = BridgeSampler::at_sample(
        train_rows.len(),
        args.seed,
        start_step * args.batch * grad_accum,
    );
    println!(
        "Bridge grad_accum={grad_accum} effective_batch={} seed={} start_step={start_step}",
        args.batch * grad_accum,
        args.seed
    );
    let run_dir = util::create_run_dir("bridge")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    for step in (start_step + 1)..=args.steps {
        let mut accumulated = None;
        let mut loss_sum = 0.0f32;
        let mut positive_sum = 0.0f32;
        let mut zero_margin_sum = 0.0f32;
        let mut shuffle_margin_sum = 0.0f32;
        let mut hard_margin_sum = 0.0f32;
        for _ in 0..grad_accum {
            let indices = sampler.next_batch(args.batch);
            let batch_rows = indices
                .into_iter()
                .map(|index| train_rows[index].clone())
                .collect::<Vec<_>>();
            let state_slots = state_conditioning(
                &batch_rows,
                regime,
                &encoder,
                &compressor,
                &encoder_vocab,
                max_seq,
                &device,
            )?;
            let mut cond = if let Some(prefix) = &static_prefix {
                let ids = Tensor::arange(0u32, output_slots as u32, &device)?.unsqueeze(0)?;
                prefix.forward(&ids)?.broadcast_as((
                    batch_rows.len(),
                    output_slots,
                    cfg.hidden_size,
                ))?
            } else {
                adapter.forward(&state_slots)?
            };
            if rand::random::<f64>() < dropout {
                cond = cond.zeros_like()?;
            }
            let (input, labels, mask) = qwen_batch(&tokenizer, &batch_rows, &device)?;
            let positive = token_loss(&qwen, &input, &labels, &mask, &cond)?;
            let mut margin_loss: Option<Tensor> = None;
            if negatives.zero {
                let value = add_conditioning_margin_loss(
                    None,
                    &positive,
                    &token_loss(&qwen, &input, &labels, &mask, &cond.zeros_like()?)?,
                    margin,
                )?;
                if step % log_every == 0 {
                    zero_margin_sum += util::scalar_f32(&value)?;
                }
                margin_loss = Some(match margin_loss {
                    Some(total) => total.broadcast_add(&value)?,
                    None => value,
                });
            }
            if negatives.shuffle {
                let value = add_conditioning_margin_loss(
                    None,
                    &positive,
                    &token_loss(
                        &qwen,
                        &input,
                        &labels,
                        &mask,
                        &batch_shuffled_conditioning_latent(&cond)?,
                    )?,
                    margin,
                )?;
                if step % log_every == 0 {
                    shuffle_margin_sum += util::scalar_f32(&value)?;
                }
                margin_loss = Some(match margin_loss {
                    Some(total) => total.broadcast_add(&value)?,
                    None => value,
                });
            }
            if negatives.hard {
                let value = add_conditioning_margin_loss(
                    None,
                    &positive,
                    &token_loss(
                        &qwen,
                        &input,
                        &labels,
                        &mask,
                        &hard_mismatched_conditioning_latent(&cond)?,
                    )?,
                    margin,
                )?;
                if step % log_every == 0 {
                    hard_margin_sum += util::scalar_f32(&value)?;
                }
                margin_loss = Some(match margin_loss {
                    Some(total) => total.broadcast_add(&value)?,
                    None => value,
                });
            }
            let loss = match margin_loss {
                Some(value) => positive.broadcast_add(&value.affine(margin_weight, 0.0)?)?,
                None => positive.clone(),
            };
            if step % log_every == 0 {
                loss_sum += util::scalar_f32(&loss)?;
                positive_sum += util::scalar_f32(&positive)?;
            }
            util::accumulate_scaled_gradients(
                &mut accumulated,
                &optimizer_vars,
                &loss,
                grad_accum,
            )?;
        }
        let grad_norm =
            util::clip_accumulated_gradients_device(&mut accumulated, &optimizer_vars, clip_norm)?;
        util::optimizer_step_from_accumulated(&mut optimizer, &mut accumulated)?;
        if step % log_every == 0 {
            let divisor = grad_accum as f32;
            tb.add_scalar("loss/total", loss_sum / divisor, step);
            tb.add_scalar("loss/positive_ce", positive_sum / divisor, step);
            tb.add_scalar("loss/margin_zero", zero_margin_sum / divisor, step);
            tb.add_scalar("loss/margin_shuffle", shuffle_margin_sum / divisor, step);
            tb.add_scalar("loss/margin_hard", hard_margin_sum / divisor, step);
            tb.add_scalar("schedule/lr", lr as f32, step);
            if let Some(norm) = grad_norm {
                tb.add_scalar("grad/global_norm", util::scalar_f32(&norm)?, step);
            }
            println!(
                "bridge step {step}/{} loss {:.4}",
                args.steps,
                loss_sum / divisor
            );
        }
        if step % val_every == 0 || step == args.steps {
            let (matched, zero) = full_val_losses(
                &val_rows,
                args.batch,
                regime,
                &tokenizer,
                &encoder,
                &compressor,
                &encoder_vocab,
                &adapter,
                static_prefix.as_ref(),
                &qwen,
                max_seq,
                &device,
                output_slots,
                cfg.hidden_size,
            )?;
            tb.add_scalar("val/ce_matched", matched, step);
            tb.add_scalar("val/ce_zeroed", zero, step);
            tb.add_scalar("val/gap", zero - matched, step);
            let telemetry_rows = &val_rows[..val_rows.len().min(args.batch.max(1))];
            let slots = state_conditioning(
                telemetry_rows,
                regime,
                &encoder,
                &compressor,
                &encoder_vocab,
                max_seq,
                &device,
            )?;
            let telemetry_cond = if let Some(prefix) = &static_prefix {
                let ids = Tensor::arange(0u32, output_slots as u32, &device)?.unsqueeze(0)?;
                prefix.forward(&ids)?.broadcast_as((
                    telemetry_rows.len(),
                    output_slots,
                    cfg.hidden_size,
                ))?
            } else {
                adapter.forward(&slots)?
            };
            let (norm_mean, cond_std) = conditioning_health(&telemetry_cond)?;
            tb.add_scalar("cond/norm_mean", norm_mean, step);
            tb.add_scalar("cond/std", cond_std, step);
            let (telemetry_input, _, _) = qwen_batch(&tokenizer, telemetry_rows, &device)?;
            for (site, mean, max) in qwen.gate_statistics(&telemetry_input, &telemetry_cond)? {
                tb.add_scalar(&format!("gate/site_{site}_mean"), mean, step);
                tb.add_scalar(&format!("gate/site_{site}_max"), max, step);
            }
            tb.flush();
            println!("bridge val step={step} val_ce_matched={matched:.4} val_ce_zeroed={zero:.4} gap={:.4}", zero - matched);
            util::save_varmap_atomic(&train_vars, &latest_path)?;
            if unfreeze_world {
                util::save_varmap_atomic(
                    &world_map,
                    &latest_path.with_extension("world.safetensors"),
                )?;
            }
            if matched < best_ce {
                best_ce = matched;
                util::save_varmap_atomic(&train_vars, &best_path)?;
                util::save_varmap_atomic(&train_vars, &args.output)?;
                if unfreeze_world {
                    util::save_varmap_atomic(
                        &world_map,
                        &args.output.with_extension("world.safetensors"),
                    )?;
                }
            }
            optimizer.save_state(&optimizer_path)?;
            util::save_resume_state(
                &resume_path,
                &util::TrainingResumeState {
                    stage: resume_stage.clone(),
                    step,
                    best_metric: best_ce,
                    best_aux_metric: zero,
                    saved_checkpoint: best_ce.is_finite(),
                },
            )?;
        }
    }
    println!(
        "Best bridge saved to {} (val_ce={best_ce:.4}); latest={}",
        args.output.display(),
        latest_path.display()
    );
    tb.finish()?;
    Ok(())
}

pub fn try_run_eval_bridge(args: &[String]) -> Result<bool> {
    if !matches!(
        args.get(1).map(String::as_str),
        Some("--eval-bridge" | "eval-bridge")
    ) {
        return Ok(false);
    }
    crate::tasks::eval::try_run_code_eval(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioning_text_never_contains_completion() {
        let row = VeclabTaskRow {
            task: "TASK_SENTINEL".into(),
            completion: "GOLD_COMPLETION_SENTINEL".into(),
            function_id: 1,
            docs: "DOC_SENTINEL".into(),
        };
        for regime in [BridgeRegime::Context, BridgeRegime::Weights] {
            let world = world_rows(std::slice::from_ref(&row), regime);
            assert!(world[0].state_text.contains("TASK_SENTINEL"));
            assert!(!world[0].state_text.contains("GOLD_COMPLETION_SENTINEL"));
        }
    }

    #[test]
    fn sampler_uses_distinct_microbatches_and_resumes_exactly() {
        let mut sampler = BridgeSampler::at_sample(16, 7, 0);
        let first = sampler.next_batch(2);
        let second = sampler.next_batch(2);
        assert!(first.iter().all(|index| !second.contains(index)));

        let mut resumed = BridgeSampler::at_sample(16, 7, 4);
        assert_eq!(sampler.next_batch(2), resumed.next_batch(2));
    }
}
