//! Training entry point for the frozen-Qwen knowledge bridge.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::data::{encode_world_examples, RawWorldExample};
use crate::model::decoders::{Qwen3Bridge, Qwen3Config};
use crate::model::{
    load_vocab_from_file, ActionStateTransition, ContextCompressor, DecoderConditioningAdapter,
    OnlineEncoder,
};
use crate::tasks::veclab::{
    attach_docs, load_docs_map, load_task_rows, model_visible_task, VeclabTaskRow,
    SEEN_FUNCTION_MAX,
};
use crate::tasks::world_context::{
    context_slots_from_world_pair_sequences, env_bool, env_f64, env_usize,
};
use crate::tasks::world_support::{masked_cross_entropy, masked_unlikelihood};
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

fn mismatched_rows(
    pool: &[VeclabTaskRow],
    positives: &[VeclabTaskRow],
    offset: usize,
) -> Result<Vec<VeclabTaskRow>> {
    if pool.len() < 2 {
        bail!("mismatched conditioning requires at least two rows");
    }
    positives
        .iter()
        .enumerate()
        .map(|(index, positive)| {
            let start = (offset.max(1) + index) % pool.len();
            // Match the public Go signature first.  With counterfactual bridge
            // prompts this gives the decoder exactly the same visible request
            // under the matched and wrong world states, so a generic code
            // prefix cannot satisfy the negative objective.
            (0..pool.len())
                .map(|delta| &pool[(start + delta) % pool.len()])
                .find(|candidate| {
                    candidate.function_id != positive.function_id
                        && same_bridge_signature(candidate, positive)
                })
                .or_else(|| {
                    (0..pool.len())
                        .map(|delta| &pool[(start + delta) % pool.len()])
                        .find(|candidate| candidate.function_id != positive.function_id)
                })
                .cloned()
                .context("mismatched conditioning requires at least two function groups")
        })
        .collect()
}

fn solve_signature(completion: &str) -> Option<&str> {
    completion
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("func Solve("))
        .map(|line| {
            line.split_once('{')
                .map_or(line, |(signature, _)| signature)
                .trim_end()
        })
}

fn same_bridge_signature(left: &VeclabTaskRow, right: &VeclabTaskRow) -> bool {
    matches!(
        (solve_signature(&left.completion), solve_signature(&right.completion)),
        (Some(left), Some(right)) if left == right
    )
}

fn bridge_prompt_task(row: &VeclabTaskRow) -> String {
    if !env_bool("TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS", true) {
        return model_visible_task(&row.task).to_string();
    }
    let signature = solve_signature(&row.completion).unwrap_or("func Solve()");
    format!(
        "Implement exactly this Go entry point. The required behavior is supplied only by the latent world state.\n\n{signature}"
    )
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
    max_seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let max_seq = max_seq.max(2);
    let encoded = rows
        .iter()
        .map(|row| {
            let prompt = tokenizer
                .encode(
                    qwen_prompt(&bridge_prompt_task(row), &row.completion),
                    false,
                )
                .map_err(anyhow::Error::msg)?;
            let completion = tokenizer
                .encode(row.completion.clone(), false)
                .map_err(anyhow::Error::msg)?;
            let mut ids = prompt.get_ids().to_vec();
            let prompt_len = ids.len();
            ids.extend_from_slice(completion.get_ids());
            ids.push(qwen_eos_id(tokenizer)?);
            if ids.len() > max_seq {
                ids.truncate(max_seq);
            }
            let prompt_len = prompt_len.min(ids.len().saturating_sub(1));
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

fn qwen_eos_id(tokenizer: &Tokenizer) -> Result<u32> {
    tokenizer
        .token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .context("Qwen tokenizer is missing an EOS token")
}

fn qwen_prompt(task: &str, completion: &str) -> String {
    if completion.trim_start().starts_with("package solution") || completion.is_empty() {
        format!(
            "{task}\n\nReturn only complete Go source code. Start with `package solution`; do not use Markdown fences or explanatory prose."
        )
    } else {
        format!("{task}\n\nReturn the relevant reference documentation only.")
    }
}

fn split_bridge_rows(
    rows: Vec<VeclabTaskRow>,
    lora_mode: bool,
) -> Result<(Vec<VeclabTaskRow>, Vec<VeclabTaskRow>)> {
    let train_function_max = env_usize(
        "TOFY_BRIDGE_TRAIN_FUNCTION_MAX",
        SEEN_FUNCTION_MAX.saturating_sub(20),
    );
    let validation_function_max =
        env_usize("TOFY_BRIDGE_VALIDATION_FUNCTION_MAX", SEEN_FUNCTION_MAX);
    if train_function_max == 0 || train_function_max >= validation_function_max {
        bail!(
            "bridge function split must satisfy 0 < TOFY_BRIDGE_TRAIN_FUNCTION_MAX < TOFY_BRIDGE_VALIDATION_FUNCTION_MAX"
        );
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for row in rows {
        if row.function_id <= train_function_max {
            train.push(row);
        } else if row.function_id <= validation_function_max {
            validation.push(row);
        } else if lora_mode && !row.completion.trim_start().starts_with("package solution") {
            // Documentation-only LoRA controls can retain their full corpus;
            // causal bridge validation never consumes these rows.
            train.push(row);
        }
    }
    if train.is_empty() || validation.is_empty() {
        bail!("bridge function-disjoint split produced an empty partition");
    }
    Ok((train, validation))
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
                    row.docs,
                    model_visible_task(&row.task)
                ),
                BridgeRegime::Weights => model_visible_task(&row.task).to_string(),
            },
            next_text: row.completion.clone(),
            action_label: 0,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn state_conditioning(
    rows: &[VeclabTaskRow],
    regime: BridgeRegime,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    transition: &ActionStateTransition,
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
    match regime {
        BridgeRegime::Context => Ok(state_slots),
        BridgeRegime::Weights => transition.forward(&state_slots),
    }
}

pub(crate) struct BridgeRuntime {
    pub tokenizer: Tokenizer,
    pub model: Qwen3Bridge,
    pub encoder: OnlineEncoder,
    pub compressor: ContextCompressor,
    pub transition: ActionStateTransition,
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
        let transition = ActionStateTransition::new(
            VarBuilder::from_varmap(&world_map, dtype, &device).pp("action_state_transition"),
            planner_dim,
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
            transition,
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
        if std::env::var("TOFY_QWEN_LORA_RANK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            > 0
        {
            return Tensor::zeros((rows.len(), 1, self.hidden_size), DType::F32, &self.device)
                .map_err(Into::into);
        }
        let slots = state_conditioning(
            rows,
            self.regime,
            &self.encoder,
            &self.compressor,
            &self.transition,
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
            .encode(qwen_prompt(model_visible_task(prompt), ""), false)
            .map_err(anyhow::Error::msg)?;
        let mut ids = encoded.get_ids().to_vec();
        let prompt_len = ids.len();
        let eos = Some(qwen_eos_id(&self.tokenizer)?);
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

#[derive(Clone, Copy, Debug)]
struct BridgeValLosses {
    matched: f32,
    zeroed: f32,
    wrong: f32,
}

#[allow(clippy::too_many_arguments)]
fn val_losses(
    rows: &[VeclabTaskRow],
    wrong_rows: [&[VeclabTaskRow]; 2],
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    vocab: &crate::model::Vocab,
    adapter: &DecoderConditioningAdapter,
    static_prefix: Option<&candle_nn::Embedding>,
    model: &Qwen3Bridge,
    max_seq: usize,
    device: &Device,
    output_slots: usize,
    hidden_size: usize,
) -> Result<BridgeValLosses> {
    let lora_mode = std::env::var("TOFY_QWEN_LORA_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        > 0;
    let cond = if lora_mode {
        Tensor::zeros((rows.len(), 1, hidden_size), DType::F32, device)?
    } else if let Some(prefix) = static_prefix {
        let ids = Tensor::arange(0u32, output_slots as u32, device)?.unsqueeze(0)?;
        prefix
            .forward(&ids)?
            .broadcast_as((rows.len(), output_slots, hidden_size))?
    } else {
        let slots = state_conditioning(
            rows, regime, encoder, compressor, transition, vocab, max_seq, device,
        )?;
        adapter.forward(&slots)?
    };
    let (input, labels, mask) = qwen_batch(tokenizer, rows, max_seq, device)?;
    let matched = util::scalar_f32(&token_loss(model, &input, &labels, &mask, &cond)?)?;
    let zeroed = util::scalar_f32(&token_loss(
        model,
        &input,
        &labels,
        &mask,
        &cond.zeros_like()?,
    )?)?;
    let mut wrong = f32::INFINITY;
    for wrong_rows in wrong_rows {
        let wrong_cond = if lora_mode || static_prefix.is_some() {
            cond.clone()
        } else {
            let slots = state_conditioning(
                wrong_rows, regime, encoder, compressor, transition, vocab, max_seq, device,
            )?;
            adapter.forward(&slots)?
        };
        wrong = wrong.min(util::scalar_f32(&token_loss(
            model,
            &input,
            &labels,
            &mask,
            &wrong_cond,
        )?)?);
    }
    Ok(BridgeValLosses {
        matched,
        zeroed,
        wrong,
    })
}

#[allow(clippy::too_many_arguments)]
fn full_val_losses(
    rows: &[VeclabTaskRow],
    batch_size: usize,
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    vocab: &crate::model::Vocab,
    adapter: &DecoderConditioningAdapter,
    static_prefix: Option<&candle_nn::Embedding>,
    model: &Qwen3Bridge,
    max_seq: usize,
    device: &Device,
    output_slots: usize,
    hidden_size: usize,
) -> Result<BridgeValLosses> {
    let mut matched = 0.0;
    let mut zeroed = 0.0;
    let mut wrong = 0.0;
    for chunk in rows.chunks(batch_size.max(1)) {
        let wrong_a = mismatched_rows(rows, chunk, 1)?;
        let wrong_b = mismatched_rows(rows, chunk, 7)?;
        let losses = val_losses(
            chunk,
            [&wrong_a, &wrong_b],
            regime,
            tokenizer,
            encoder,
            compressor,
            transition,
            vocab,
            adapter,
            static_prefix,
            model,
            max_seq,
            device,
            output_slots,
            hidden_size,
        )?;
        matched += losses.matched * chunk.len() as f32;
        zeroed += losses.zeroed * chunk.len() as f32;
        wrong += losses.wrong * chunk.len() as f32;
    }
    let count = rows.len() as f32;
    Ok(BridgeValLosses {
        matched: matched / count,
        zeroed: zeroed / count,
        wrong: wrong / count,
    })
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

fn token_objectives(
    model: &Qwen3Bridge,
    input: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
    cond: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let logits = model.forward(input, cond)?;
    let logits = logits.narrow(1, 0, logits.dim(1)? - 1)?;
    Ok((
        masked_cross_entropy(&logits, labels, mask)?,
        masked_unlikelihood(&logits, labels, mask)?,
    ))
}

fn conditioning_separation_loss(
    matched: &Tensor,
    wrong: &Tensor,
    min_distance: f64,
) -> Result<Tensor> {
    let distance = matched.broadcast_sub(wrong)?.sqr()?.mean_all()?;
    Tensor::new(min_distance as f32, matched.device())?
        .to_dtype(distance.dtype())?
        .broadcast_sub(&distance)?
        .relu()
        .map_err(Into::into)
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
    let max_abs_error = util::scalar_f32(&abs_error.max_all()?)?;
    let mean_abs_error = util::scalar_f32(&abs_error.mean_all()?)?;
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
    let transition = ActionStateTransition::new(
        VarBuilder::from_varmap(&world_map, dtype, &device).pp("action_state_transition"),
        planner_dim,
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
    let lora_mode = std::env::var("TOFY_QWEN_LORA_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        > 0;
    let lr = env_f64("TOFY_BRIDGE_LR", 2e-4);
    let mut named_vars = util::named_train_vars(&train_vars)?;
    if static_prefix.is_some() || lora_mode {
        named_vars.retain(|entry| !entry.name.starts_with("adapter."));
    }
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
        } else if lora_mode {
            "lora"
        } else {
            "world"
        }
    );
    let mut optimizer = util::TrainOptimizer::new_lr_named(named_vars, lr)?;
    let negatives = if lora_mode {
        ConditioningNegatives::none()
    } else {
        ConditioningNegatives::from_env()
    };
    let margin = env_f64("TOFY_DECODER_CONDITIONING_MARGIN", 0.2);
    let margin_weight = env_f64("TOFY_DECODER_CONDITIONING_MARGIN_WEIGHT", 0.25);
    let unlikelihood_weight =
        env_f64("TOFY_DECODER_CONDITIONING_UNLIKELIHOOD_WEIGHT", 0.25).max(0.0);
    let separation_weight = env_f64("TOFY_DECODER_CONDITIONING_SEPARATION_WEIGHT", 0.05).max(0.0);
    let separation_min_distance = env_f64("TOFY_DECODER_CONDITIONING_MIN_DISTANCE", 0.1).max(0.0);
    let dropout = env_f64("TOFY_CONDITIONING_DROPOUT", 0.1).clamp(0.0, 1.0);
    let regime = BridgeRegime::from_env()?;
    let all_rows = load_task_rows(&args.data)?;
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt")).unwrap_or_default();
    let mut seen: Vec<_> = all_rows
        .into_iter()
        .filter(|row| {
            row.function_id <= SEEN_FUNCTION_MAX
                || (lora_mode && !row.completion.trim_start().starts_with("package solution"))
        })
        .collect();
    attach_docs(&mut seen, &docs);
    let (train_rows, val_rows) = split_bridge_rows(seen, lora_mode)?;
    let train_function_max = env_usize(
        "TOFY_BRIDGE_TRAIN_FUNCTION_MAX",
        SEEN_FUNCTION_MAX.saturating_sub(20),
    );
    let validation_function_max =
        env_usize("TOFY_BRIDGE_VALIDATION_FUNCTION_MAX", SEEN_FUNCTION_MAX);
    println!(
        "Bridge regime={} train_rows={} val_rows={} function_split=train:1-{train_function_max} val:{}-{validation_function_max} heldout_task_rows=0",
        regime.as_str(),
        train_rows.len(),
        val_rows.len(),
        train_function_max + 1,
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
        if unfreeze_world && latest_path.exists() {
            if !latest_world.exists() {
                bail!(
                    "joint world/bridge resume requires matching world sidecar: {}",
                    latest_world.display()
                );
            }
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
    let min_semantic_gap = env_f64("TOFY_BRIDGE_MIN_SEMANTIC_GAP", 0.02).max(0.0) as f32;
    let requires_semantic_gap = !lora_mode && static_prefix.is_none();
    let mut best_score = if args.resume {
        resume_state.best_metric
    } else {
        f32::INFINITY
    };
    let mut best_semantic_gap = if args.resume {
        resume_state.best_aux_metric
    } else {
        f32::NEG_INFINITY
    };
    let semantic_patience = env_usize("TOFY_BRIDGE_SEMANTIC_PATIENCE", 1_200);
    let semantic_warmup = env_usize("TOFY_BRIDGE_SEMANTIC_WARMUP", 400);
    let semantic_progress = env_f64("TOFY_BRIDGE_MIN_SEMANTIC_PROGRESS", 0.002).max(0.0) as f32;
    let mut best_observed_semantic_gap = f32::NEG_INFINITY;
    let mut last_semantic_progress_step = start_step;
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
        let mut wrong_unlikelihood_sum = 0.0f32;
        let mut conditioning_separation_sum = 0.0f32;
        for micro_step in 0..grad_accum {
            let indices = sampler.next_batch(args.batch);
            let batch_rows = indices
                .into_iter()
                .map(|index| train_rows[index].clone())
                .collect::<Vec<_>>();
            let mut cond = if lora_mode {
                Tensor::zeros((batch_rows.len(), 1, cfg.hidden_size), dtype, &device)?
            } else if let Some(prefix) = &static_prefix {
                let ids = Tensor::arange(0u32, output_slots as u32, &device)?.unsqueeze(0)?;
                prefix.forward(&ids)?.broadcast_as((
                    batch_rows.len(),
                    output_slots,
                    cfg.hidden_size,
                ))?
            } else {
                let state_slots = state_conditioning(
                    &batch_rows,
                    regime,
                    &encoder,
                    &compressor,
                    &transition,
                    &encoder_vocab,
                    max_seq,
                    &device,
                )?;
                adapter.forward(&state_slots)?
            };
            let dropout_seed =
                args.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ micro_step as u64;
            let conditioning_dropped =
                micro_step > 0 && StdRng::seed_from_u64(dropout_seed).random::<f64>() < dropout;
            if conditioning_dropped {
                cond = cond.zeros_like()?;
            }
            let (input, labels, mask) = qwen_batch(&tokenizer, &batch_rows, max_seq, &device)?;
            let semantic_negatives = !conditioning_dropped && !lora_mode && static_prefix.is_none();
            let mut negative_values = Vec::new();
            if negatives.zero && !conditioning_dropped {
                let value = util::scalar_f32(&token_loss(
                    &qwen,
                    &input,
                    &labels,
                    &mask,
                    &cond.zeros_like()?,
                )?)?;
                negative_values.push((None, value, "zero"));
            }
            for (enabled, offset, name) in [
                (negatives.shuffle, 7usize, "shuffle"),
                (negatives.hard, 1usize, "hard"),
            ] {
                if !enabled || !semantic_negatives {
                    continue;
                }
                let wrong_rows = mismatched_rows(&train_rows, &batch_rows, offset)?;
                let wrong_slots = state_conditioning(
                    &wrong_rows,
                    regime,
                    &encoder,
                    &compressor,
                    &transition,
                    &encoder_vocab,
                    max_seq,
                    &device,
                )?;
                let wrong_cond = adapter.forward(&wrong_slots)?.detach();
                let value =
                    util::scalar_f32(&token_loss(&qwen, &input, &labels, &mask, &wrong_cond)?)?;
                negative_values.push((Some(offset), value, name));
            }

            let positive = token_loss(&qwen, &input, &labels, &mask, &cond)?;
            let positive_value = util::scalar_f32(&positive)?;
            let active = negative_values
                .iter()
                .map(|(offset, negative, name)| {
                    let margin_value = (positive_value - negative + margin as f32).max(0.0);
                    (*offset, *name, margin_value)
                })
                .collect::<Vec<_>>();
            let active_count = active.iter().filter(|(_, _, value)| *value > 0.0).count();
            let positive_weight = 1.0 + margin_weight * active_count as f64;
            let weighted_positive = positive.affine(positive_weight, 0.0)?;
            util::accumulate_scaled_gradients(
                &mut accumulated,
                &optimizer_vars,
                &weighted_positive,
                grad_accum,
            )?;

            let mut unlikelihood_value = 0.0f32;
            let mut separation_value = 0.0f32;
            for (offset, _, margin_value) in &active {
                if *margin_value <= 0.0 {
                    continue;
                }
                let wrong_cond = match offset {
                    None => cond.zeros_like()?,
                    Some(offset) => {
                        let wrong_rows = mismatched_rows(&train_rows, &batch_rows, *offset)?;
                        let wrong_slots = state_conditioning(
                            &wrong_rows,
                            regime,
                            &encoder,
                            &compressor,
                            &transition,
                            &encoder_vocab,
                            max_seq,
                            &device,
                        )?;
                        adapter.forward(&wrong_slots)?
                    }
                };
                let (negative, wrong_unlikelihood) =
                    token_objectives(&qwen, &input, &labels, &mask, &wrong_cond)?;
                let weighted_negative = negative.affine(-margin_weight, 0.0)?;
                util::accumulate_scaled_gradients(
                    &mut accumulated,
                    &optimizer_vars,
                    &weighted_negative,
                    grad_accum,
                )?;
                if unlikelihood_weight > 0.0 {
                    let weighted_unlikelihood =
                        wrong_unlikelihood.affine(unlikelihood_weight, 0.0)?;
                    util::accumulate_scaled_gradients(
                        &mut accumulated,
                        &optimizer_vars,
                        &weighted_unlikelihood,
                        grad_accum,
                    )?;
                    if step % log_every == 0 {
                        unlikelihood_value += util::scalar_f32(&wrong_unlikelihood)?;
                    }
                }
                if offset.is_some() && separation_weight > 0.0 {
                    let separation =
                        conditioning_separation_loss(&cond, &wrong_cond, separation_min_distance)?;
                    let weighted_separation = separation.affine(separation_weight, 0.0)?;
                    util::accumulate_scaled_gradients(
                        &mut accumulated,
                        &optimizer_vars,
                        &weighted_separation,
                        grad_accum,
                    )?;
                    if step % log_every == 0 {
                        separation_value += util::scalar_f32(&separation)?;
                    }
                }
            }
            if step % log_every == 0 {
                let margin_total = active.iter().map(|(_, _, value)| *value).sum::<f32>();
                loss_sum += positive_value
                    + margin_weight as f32 * margin_total
                    + unlikelihood_weight as f32 * unlikelihood_value
                    + separation_weight as f32 * separation_value;
                positive_sum += positive_value;
                wrong_unlikelihood_sum += unlikelihood_value;
                conditioning_separation_sum += separation_value;
                for (_, name, value) in &active {
                    match *name {
                        "zero" => zero_margin_sum += *value,
                        "shuffle" => shuffle_margin_sum += *value,
                        "hard" => hard_margin_sum += *value,
                        _ => unreachable!(),
                    }
                }
            }
        }
        let gradient_count = util::accumulated_gradient_count(&accumulated, &optimizer_vars);
        if gradient_count == 0 {
            bail!(
                "bridge backward produced no optimizer gradients at step {step}; refusing a no-op optimizer step"
            );
        }
        let grad_norm =
            util::clip_accumulated_gradients_device(&mut accumulated, &optimizer_vars, clip_norm)?;
        if step == start_step + 1 {
            if let Some(norm) = grad_norm.as_ref() {
                let norm = util::scalar_f32(norm)?;
                if !norm.is_finite() || norm <= 0.0 {
                    bail!("bridge gradient norm is invalid at first step: {norm}");
                }
                println!(
                    "Bridge gradient preflight passed: gradients={gradient_count}/{} global_norm={norm:.6}",
                    optimizer_vars.len()
                );
            } else {
                println!(
                    "Bridge gradient preflight passed: gradients={gradient_count}/{}",
                    optimizer_vars.len()
                );
            }
        }
        util::optimizer_step_from_accumulated(&mut optimizer, &mut accumulated)?;
        if step % log_every == 0 {
            let divisor = grad_accum as f32;
            tb.add_scalar("loss/total", loss_sum / divisor, step);
            tb.add_scalar("loss/positive_ce", positive_sum / divisor, step);
            tb.add_scalar("loss/margin_zero", zero_margin_sum / divisor, step);
            tb.add_scalar("loss/margin_shuffle", shuffle_margin_sum / divisor, step);
            tb.add_scalar("loss/margin_hard", hard_margin_sum / divisor, step);
            tb.add_scalar(
                "loss/wrong_unlikelihood",
                wrong_unlikelihood_sum / divisor,
                step,
            );
            tb.add_scalar(
                "loss/conditioning_separation",
                conditioning_separation_sum / divisor,
                step,
            );
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
            let losses = full_val_losses(
                &val_rows,
                args.batch,
                regime,
                &tokenizer,
                &encoder,
                &compressor,
                &transition,
                &encoder_vocab,
                &adapter,
                static_prefix.as_ref(),
                &qwen,
                max_seq,
                &device,
                output_slots,
                cfg.hidden_size,
            )?;
            let zero_gap = losses.zeroed - losses.matched;
            let semantic_gap = losses.wrong - losses.matched;
            if semantic_gap >= min_semantic_gap
                || semantic_gap >= best_observed_semantic_gap + semantic_progress
            {
                best_observed_semantic_gap = best_observed_semantic_gap.max(semantic_gap);
                last_semantic_progress_step = step;
            }
            tb.add_scalar("val/ce_matched", losses.matched, step);
            tb.add_scalar("val/ce_zeroed", losses.zeroed, step);
            tb.add_scalar("val/ce_wrong", losses.wrong, step);
            tb.add_scalar("val/zero_gap", zero_gap, step);
            tb.add_scalar("val/semantic_gap", semantic_gap, step);
            if !lora_mode {
                let telemetry_rows = &val_rows[..val_rows.len().min(args.batch.max(1))];
                let slots = state_conditioning(
                    telemetry_rows,
                    regime,
                    &encoder,
                    &compressor,
                    &transition,
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
                let (telemetry_input, _, _) =
                    qwen_batch(&tokenizer, telemetry_rows, max_seq, &device)?;
                for (site, mean, max) in qwen.gate_statistics(&telemetry_input, &telemetry_cond)? {
                    tb.add_scalar(&format!("gate/site_{site}_mean"), mean, step);
                    tb.add_scalar(&format!("gate/site_{site}_max"), max, step);
                }
            }
            tb.flush();
            println!(
                "bridge val step={step} val_ce_matched={:.4} val_ce_zeroed={:.4} val_ce_wrong={:.4} zero_gap={zero_gap:.4} semantic_gap={semantic_gap:.4}",
                losses.matched, losses.zeroed, losses.wrong
            );
            util::save_varmap_atomic(&train_vars, &latest_path)?;
            if unfreeze_world {
                util::save_varmap_atomic(
                    &world_map,
                    &latest_path.with_extension("world.safetensors"),
                )?;
            }
            let eligible = !requires_semantic_gap || semantic_gap >= min_semantic_gap;
            let selection_score = losses.matched;
            tb.add_scalar("val/selection_score", selection_score, step);
            if eligible && selection_score < best_score {
                best_score = selection_score;
                best_semantic_gap = semantic_gap;
                util::save_varmap_atomic(&train_vars, &best_path)?;
                util::save_varmap_atomic(&train_vars, &args.output)?;
                if unfreeze_world {
                    util::save_varmap_atomic(
                        &world_map,
                        &args.output.with_extension("world.safetensors"),
                    )?;
                    util::save_varmap_atomic(
                        &world_map,
                        &best_path.with_extension("world.safetensors"),
                    )?;
                }
            }
            optimizer.save_state(&optimizer_path)?;
            util::save_resume_state(
                &resume_path,
                &util::TrainingResumeState {
                    stage: resume_stage.clone(),
                    step,
                    best_metric: best_score,
                    best_aux_metric: best_semantic_gap,
                    saved_checkpoint: best_score.is_finite(),
                },
            )?;
            if requires_semantic_gap
                && semantic_patience > 0
                && step >= semantic_warmup
                && step.saturating_sub(last_semantic_progress_step) >= semantic_patience
                && best_observed_semantic_gap < min_semantic_gap
            {
                bail!(
                    "semantic conditioning plateau at step {step}: best_gap={best_observed_semantic_gap:.4}, required={min_semantic_gap:.4}, patience={semantic_patience}"
                );
            }
        }
    }
    println!(
        "Best bridge saved to {} (selection_score={best_score:.4}, semantic_gap={best_semantic_gap:.4}); latest={}",
        args.output.display(),
        latest_path.display()
    );
    if requires_semantic_gap && best_semantic_gap < min_semantic_gap {
        bail!(
            "no bridge checkpoint reached the required semantic conditioning gap: selected={best_semantic_gap:.4}, required={min_semantic_gap:.4}"
        );
    }
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
    use candle_nn::Optimizer;

    #[test]
    fn conditioning_text_never_contains_completion() {
        let row = VeclabTaskRow {
            task: "[fn:001] TASK_SENTINEL".into(),
            completion: "GOLD_COMPLETION_SENTINEL".into(),
            function_id: 1,
            docs: "DOC_SENTINEL".into(),
        };
        for regime in [BridgeRegime::Context, BridgeRegime::Weights] {
            let world = world_rows(std::slice::from_ref(&row), regime);
            assert!(world[0].state_text.contains("TASK_SENTINEL"));
            assert!(!world[0].state_text.contains("[fn:001]"));
            assert!(!world[0].state_text.contains("GOLD_COMPLETION_SENTINEL"));
        }
    }

    #[test]
    fn batch_one_mismatch_uses_another_function() -> Result<()> {
        let rows = (1..=3)
            .map(|function_id| VeclabTaskRow {
                task: format!("[fn:{function_id:03}] task"),
                completion: "completion".into(),
                function_id,
                docs: format!("docs {function_id}"),
            })
            .collect::<Vec<_>>();
        let wrong = mismatched_rows(&rows, &rows[..1], 1)?;
        assert_eq!(wrong.len(), 1);
        assert_ne!(wrong[0].function_id, rows[0].function_id);
        Ok(())
    }

    #[test]
    fn hard_mismatch_keeps_the_counterfactual_prompt_fixed() -> Result<()> {
        let rows = ["Alpha", "Beta", "Gamma"]
            .into_iter()
            .enumerate()
            .map(|(index, name)| VeclabTaskRow {
                task: format!("[fn:{:03}] visible task {name}", index + 1),
                completion: format!(
                    "package solution\n\nfunc Solve(xs []float64, k int) float64 {{ return veclab.{name}(xs, k) }}"
                ),
                function_id: index + 1,
                docs: format!("docs {name}"),
            })
            .collect::<Vec<_>>();
        let wrong = mismatched_rows(&rows, &rows[..1], 1)?;
        assert_ne!(wrong[0].function_id, rows[0].function_id);
        assert!(same_bridge_signature(&wrong[0], &rows[0]));
        assert_eq!(bridge_prompt_task(&wrong[0]), bridge_prompt_task(&rows[0]));
        Ok(())
    }

    #[test]
    fn bridge_validation_is_function_disjoint() -> Result<()> {
        let rows = (1..=100)
            .flat_map(|function_id| {
                (0..2).map(move |_| VeclabTaskRow {
                    task: format!("[fn:{function_id:03}] task"),
                    completion: "package solution\nfunc Solve() {}".into(),
                    function_id,
                    docs: String::new(),
                })
            })
            .collect::<Vec<_>>();
        let (train, validation) = split_bridge_rows(rows, false)?;
        assert!(train.iter().all(|row| row.function_id <= 80));
        assert!(validation
            .iter()
            .all(|row| (81..=100).contains(&row.function_id)));
        Ok(())
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

    #[test]
    fn full_bridge_cuda_bf16_updates_trainables() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let dtype = DType::BF16;
        let cfg = Qwen3Config {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
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
            hidden_act: candle_nn::Activation::Silu,
        };

        // Materialize a base checkpoint, then rebuild it as detached tensors
        // to match the production frozen-Qwen graph.
        let base_vars = VarMap::new();
        let throwaway_train_vars = VarMap::new();
        let _ = Qwen3Bridge::new(
            &cfg,
            VarBuilder::from_varmap(&base_vars, dtype, &device),
            VarBuilder::from_varmap(&throwaway_train_vars, dtype, &device),
        )?;
        let frozen = util::frozen_tensors_from_varmap(&base_vars)?;

        let train_vars = VarMap::new();
        let train_vb = VarBuilder::from_varmap(&train_vars, dtype, &device);
        let model = Qwen3Bridge::new(
            &cfg,
            VarBuilder::from_tensors(frozen, dtype, &device),
            train_vb.pp("qwen_bridge"),
        )?;
        let adapter = DecoderConditioningAdapter::new_with_compress_rate(
            train_vb.pp("adapter"),
            8,
            cfg.hidden_size,
            2,
            1,
        )?;
        let state_slots = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?.to_dtype(dtype)?;
        let cond = adapter.forward(&state_slots)?;
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], (2, 4), &device)?;
        let labels = Tensor::from_vec(vec![2u32, 3, 4, 6, 7, 8], (2, 3), &device)?;
        let mask = Tensor::ones((2, 3), DType::F32, &device)?;
        let (cross_entropy, unlikelihood) =
            token_objectives(&model, &input, &labels, &mask, &cond)?;
        let loss = cross_entropy.broadcast_add(&unlikelihood)?;
        assert!(cond.track_op() && loss.track_op());
        let grads = loss.backward()?;
        let named = util::named_train_vars(&train_vars)?;
        let grad_count = named
            .iter()
            .filter(|entry| grads.get(&entry.var).is_some())
            .count();
        assert_eq!(
            grad_count,
            named.len(),
            "some full-bridge gradients are missing"
        );
        let gate = named
            .iter()
            .find(|entry| entry.name.ends_with("gate.bias"))
            .context("missing bridge gate bias")?
            .var
            .clone();
        let gate_grad = grads
            .get(&gate)
            .context("missing bridge gate gradient")?
            .to_dtype(DType::F32)?
            .abs()?
            .sum_all()?;
        let gate_grad = util::scalar_f32(&gate_grad)?;
        assert!(gate_grad.is_finite() && gate_grad > 0.0);

        let before = util::vec1_f32(gate.as_tensor())?;
        let mut optimizer = util::ResumableAdamW::new_lr_named(named, 0.1)?;
        optimizer.step(&grads)?;
        let after = util::vec1_f32(gate.as_tensor())?;
        assert_ne!(before, after, "optimizer did not update the bridge gate");
        Ok(())
    }
}
