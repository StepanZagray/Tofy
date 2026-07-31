//! Training entry point for the frozen-Qwen knowledge bridge.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::data::{encode_world_examples, RawWorldExample, ACTION_FETCH_DOCS};
use crate::model::decoders::{Qwen3Bridge, Qwen3Config};
use crate::model::{
    load_vocab_from_file, ContextCompressor, DecoderConditioningAdapter, LeWorldModel,
    OnlineEncoder,
};
use crate::tasks::eval::{compile_and_test, load_suite, EvalTask};
use crate::tasks::veclab::{
    attach_docs, load_docs_map, load_task_rows, model_visible_task, VeclabTaskRow,
    SEEN_FUNCTION_MAX,
};
use crate::tasks::world_context::{context_slots_from_world_states, env_bool, env_f64, env_usize};
use crate::tasks::world_support::{
    masked_cross_entropy, masked_cross_entropy_smoothed, masked_unlikelihood,
};
use crate::util;

/// Recorded frozen-Qwen RAG ceiling on the full 300-task slices (2026-07-23).
pub(crate) const DEFAULT_RAG_CEILING_SEEN: f64 = 0.35;
pub(crate) const DEFAULT_RAG_CEILING_HELDOUT: f64 = 0.42;

pub(crate) fn rag_ceiling_for_function(function_id: usize) -> f32 {
    let seen = function_id <= SEEN_FUNCTION_MAX;
    let env_name = if seen {
        "TOFY_RAG_CEILING_SEEN"
    } else {
        "TOFY_RAG_CEILING_HELDOUT"
    };
    let default = if seen {
        DEFAULT_RAG_CEILING_SEEN
    } else {
        DEFAULT_RAG_CEILING_HELDOUT
    };
    env_f64(env_name, default).clamp(0.0, 1.0) as f32
}

pub(crate) fn rag_ceiling_for_split_label(label: &str) -> f32 {
    match label {
        "seen" | "train" | "matched" => rag_ceiling_for_function(1),
        "heldout" | "held-out" => rag_ceiling_for_function(SEEN_FUNCTION_MAX + 1),
        _ => rag_ceiling_for_function(1),
    }
}

pub(crate) fn pass_rate_rag_fraction(pass_rate: f32, rag_ceiling: f32) -> f32 {
    if rag_ceiling <= 0.0 {
        0.0
    } else {
        pass_rate / rag_ceiling
    }
}

pub(crate) fn bridge_min_ar_pass_rate() -> f32 {
    if std::env::var_os("TOFY_BRIDGE_MIN_AR_PASS_RATE").is_some() {
        return env_f64("TOFY_BRIDGE_MIN_AR_PASS_RATE", 0.25).clamp(0.0, 1.0) as f32;
    }
    let fraction = env_f64("TOFY_BRIDGE_MIN_AR_PASS_FRACTION", 0.5).clamp(0.0, 1.0) as f32;
    fraction * rag_ceiling_for_function(1)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BridgeDecodeConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub samples: usize,
    pub pass_at_k: usize,
    pub seed: u64,
}

impl BridgeDecodeConfig {
    pub fn from_env() -> Self {
        let temperature = env_f64("TOFY_BRIDGE_DECODE_TEMP", 0.0).max(0.0);
        let top_k = env_usize("TOFY_BRIDGE_DECODE_TOP_K", 0);
        // At temperature 0 decoding is deterministic, so extra draws are byte-identical
        // copies. Collapsing to one keeps pass@k honest instead of reporting pass@8 for
        // what is really pass@1 at eight times the cost.
        let samples = if temperature > 0.0 {
            env_usize("TOFY_BRIDGE_DECODE_SAMPLES", 1).max(1)
        } else {
            1
        };
        let pass_at_k = env_usize("TOFY_BRIDGE_PASS_AT_K", samples).clamp(1, samples);
        let seed = std::env::var("TOFY_BRIDGE_DECODE_SEED")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(42);
        Self {
            temperature,
            top_k,
            samples,
            pass_at_k,
            seed,
        }
    }

    pub fn uses_sampling(self) -> bool {
        self.temperature > 0.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BridgeNonqualificationReason {
    SemanticPlateau,
    BudgetExhausted,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct BridgeStageStatus {
    pub(crate) attempt_id: String,
    pub(crate) stage: String,
    pub(crate) outcome: String,
    pub(crate) reason: BridgeNonqualificationReason,
    pub(crate) step: usize,
}

fn write_bridge_nonqualification_status(
    reason: BridgeNonqualificationReason,
    step: usize,
) -> Result<()> {
    let Some(path) = std::env::var_os("TOFY_STAGE_STATUS_PATH").map(PathBuf::from) else {
        return Ok(());
    };
    let attempt_id = std::env::var("TOFY_STAGE_ATTEMPT_ID")
        .context("TOFY_STAGE_STATUS_PATH requires TOFY_STAGE_ATTEMPT_ID")?;
    let stage = std::env::var("TOFY_RUN_STAGE_NAME")
        .context("TOFY_STAGE_STATUS_PATH requires TOFY_RUN_STAGE_NAME")?;
    let status = BridgeStageStatus {
        attempt_id,
        stage,
        outcome: "non_qualified".to_string(),
        reason,
        step,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
    fs::write(&temporary, serde_json::to_vec_pretty(&status)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

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

fn row_bridge_signature(row: &VeclabTaskRow) -> &str {
    solve_signature(&row.completion).unwrap_or_else(|| visible_solve_signature(&row.task))
}

fn same_bridge_signature(left: &VeclabTaskRow, right: &VeclabTaskRow) -> bool {
    row_bridge_signature(left) == row_bridge_signature(right)
}

fn bridge_ar_harness<'a>(
    harness_by_fn: &'a HashMap<usize, Vec<EvalTask>>,
    row: &VeclabTaskRow,
) -> Result<&'a EvalTask> {
    let tasks = harness_by_fn
        .get(&row.function_id)
        .with_context(|| format!("no eval harness for function {}", row.function_id))?;
    let signature = row_bridge_signature(row);
    tasks
        .iter()
        .find(|task| visible_solve_signature(&task.task) == signature)
        .or_else(|| tasks.first())
        .with_context(|| format!("empty eval harness list for function {}", row.function_id))
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

fn counterfactual_bridge_prompt(task: &str) -> String {
    if !env_bool("TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS", true) {
        return model_visible_task(task).to_string();
    }
    let visible = model_visible_task(task);
    let signature = visible
        .find("func Solve(")
        .and_then(|start| {
            let tail = &visible[start..];
            let end = ['`', '{', '\n']
                .into_iter()
                .filter_map(|delimiter| tail.find(delimiter))
                .chain(
                    [" by ", " that ", " must ", " should "]
                        .into_iter()
                        .filter_map(|delimiter| tail.find(delimiter)),
                )
                .min()
                .unwrap_or(tail.len());
            let signature = tail[..end].trim();
            signature.contains(')').then_some(signature)
        })
        .unwrap_or("func Solve()");
    format!(
        "Implement exactly this Go entry point. The required behavior is supplied only by the latent world state.\n\n{signature}"
    )
}

fn semantic_completion_spans(completion: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = completion[cursor..].find("veclab.") {
        let start = cursor + offset + "veclab.".len();
        let end = completion[start..]
            .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
            .map(|offset| start + offset)
            .unwrap_or(completion.len());
        if end == start {
            break;
        }
        spans.push((start, end));
        cursor = end;
    }
    if spans.is_empty() {
        let Some(start) = completion
            .find("return ")
            .map(|start| start + "return ".len())
        else {
            return spans;
        };
        let end = completion[start..]
            .find('\n')
            .map(|offset| start + offset)
            .unwrap_or(completion.len());
        if end > start {
            spans.push((start, end));
        }
    }
    spans
}

struct QwenExampleText {
    prompt: String,
    target: String,
    source_prefix: Option<String>,
}

fn code_source_prefix_and_body(completion: &str) -> Result<(&str, &str)> {
    let function_start = completion
        .find("func Solve(")
        .context("Go bridge completion is missing func Solve")?;
    let opening = completion[function_start..]
        .find('{')
        .map(|offset| function_start + offset)
        .context("Go bridge completion is missing the Solve body")?;
    Ok(completion.split_at(opening + 1))
}

fn visible_solve_signature(task: &str) -> &str {
    let visible = model_visible_task(task);
    visible
        .find("func Solve(")
        .and_then(|start| {
            let tail = &visible[start..];
            let end = ['`', '{', '\n']
                .into_iter()
                .filter_map(|delimiter| tail.find(delimiter))
                .chain(
                    [" by ", " that ", " must ", " should "]
                        .into_iter()
                        .filter_map(|delimiter| tail.find(delimiter)),
                )
                .min()
                .unwrap_or(tail.len());
            let signature = tail[..end].trim();
            signature.contains(')').then_some(signature)
        })
        .unwrap_or("func Solve()")
}

fn code_scaffold(signature: &str) -> String {
    format!(
        "package solution\n\nimport \"veclab.dev/veclab\"\n\n{} {{",
        signature.trim_end_matches('{').trim()
    )
}

fn code_completion_prompt(task: &str, source_prefix: &str) -> String {
    // A base model is trained to continue source, not to obey a chat-style
    // negative instruction list.  Keeping task information in a block comment
    // presents an ordinary Go completion distribution.
    let task = task.replace("*/", "* /");
    format!("/* Task context:\n{task}\n*/\n\n{source_prefix}")
}

fn qwen_example_text(task: &str, completion: &str) -> Result<QwenExampleText> {
    if completion.trim_start().starts_with("package solution") {
        let (source_prefix, body) = code_source_prefix_and_body(completion)?;
        return Ok(QwenExampleText {
            prompt: code_completion_prompt(task, source_prefix),
            target: body.to_string(),
            source_prefix: Some(source_prefix.to_string()),
        });
    }
    if completion.is_empty() {
        let source_prefix = code_scaffold(visible_solve_signature(task));
        return Ok(QwenExampleText {
            prompt: code_completion_prompt(task, &source_prefix),
            target: String::new(),
            source_prefix: Some(source_prefix),
        });
    }
    Ok(QwenExampleText {
        prompt: format!("{task}\n\nReference documentation:\n"),
        target: completion.to_string(),
        source_prefix: None,
    })
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
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let max_seq = max_seq.max(4);
    let encoded = rows
        .iter()
        .map(|row| {
            let text = qwen_example_text(&bridge_prompt_task(row), &row.completion)?;
            let prompt = tokenizer
                .encode(text.prompt, false)
                .map_err(anyhow::Error::msg)?;
            let completion = tokenizer
                .encode(text.target.clone(), false)
                .map_err(anyhow::Error::msg)?;
            if completion.get_ids().is_empty() {
                bail!(
                    "bridge row for function {} has an empty Qwen completion",
                    row.function_id
                );
            }
            if completion.get_ids().len() + 2 > max_seq {
                bail!(
                    "bridge target for function {} needs {} tokens but max_seq is {}; targets are never truncated",
                    row.function_id,
                    completion.get_ids().len() + 2,
                    max_seq
                );
            }
            // Preserve the entire target and EOS. Only the oldest prompt
            // context may be trimmed, which cannot change the supervised code
            // body or turn a prompt token into a target token.
            let max_prompt = max_seq - completion.get_ids().len() - 1;
            let prompt_ids = prompt.get_ids();
            let prompt_start = prompt_ids.len().saturating_sub(max_prompt);
            let mut ids = prompt_ids[prompt_start..].to_vec();
            let prompt_len = ids.len();
            ids.extend(completion.get_ids());
            ids.push(qwen_eos_id(tokenizer)?);
            let prompt_len = prompt_len.min(ids.len().saturating_sub(2));
            let semantic_spans = semantic_completion_spans(&text.target);
            let semantic_tokens = completion
                .get_offsets()
                .iter()
                .map(|&(start, end)| {
                    semantic_spans
                        .iter()
                        .any(|&(span_start, span_end)| end > span_start && start < span_end)
                })
                .collect::<Vec<_>>();
            Ok((ids, prompt_len, semantic_tokens))
        })
        .collect::<Result<Vec<_>>>()?;
    let max_len = encoded
        .iter()
        .map(|(ids, _, _)| ids.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let pad = tokenizer.get_padding().map(|v| v.pad_id).unwrap_or(0);
    let mut inputs = vec![pad; encoded.len() * max_len];
    let mut labels = vec![pad; encoded.len() * (max_len - 1)];
    let mut mask = vec![0f32; encoded.len() * (max_len - 1)];
    let mut semantic_mask = vec![0f32; encoded.len() * (max_len - 1)];
    for (batch, (ids, prompt_len, semantic_tokens)) in encoded.iter().enumerate() {
        let offset = batch * max_len;
        inputs[offset..offset + ids.len()].copy_from_slice(ids);
        let label_offset = batch * (max_len - 1);
        labels[label_offset..label_offset + ids.len().saturating_sub(1)].copy_from_slice(&ids[1..]);
        for index in prompt_len.saturating_sub(1)..ids.len().saturating_sub(1) {
            mask[label_offset + index] = 1.0;
        }
        for (completion_index, is_semantic) in semantic_tokens.iter().copied().enumerate() {
            let index = prompt_len.saturating_sub(1) + completion_index;
            if is_semantic && index < ids.len().saturating_sub(1) {
                semantic_mask[label_offset + index] = 1.0;
            }
        }
        let semantic_range = label_offset..label_offset + max_len - 1;
        if semantic_mask[semantic_range.clone()]
            .iter()
            .all(|value| *value == 0.0)
        {
            semantic_mask[semantic_range.clone()].copy_from_slice(&mask[semantic_range]);
        }
    }
    Ok((
        Tensor::from_vec(inputs, (encoded.len(), max_len), device)?,
        Tensor::from_vec(labels, (encoded.len(), max_len - 1), device)?,
        Tensor::from_vec(mask, (encoded.len(), max_len - 1), device)?,
        Tensor::from_vec(semantic_mask, (encoded.len(), max_len - 1), device)?,
    ))
}

fn qwen_eos_id(tokenizer: &Tokenizer) -> Result<u32> {
    tokenizer
        .token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .context("Qwen tokenizer is missing an EOS token")
}

fn split_bridge_rows(rows: Vec<VeclabTaskRow>) -> Result<(Vec<VeclabTaskRow>, Vec<VeclabTaskRow>)> {
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
        let is_code = row.completion.trim_start().starts_with("package solution");
        if !is_code || row.function_id <= train_function_max {
            // All documentation rows ground the decoder in the complete world
            // vocabulary; only code rows participate in the causal split.
            train.push(row);
        } else if row.function_id <= validation_function_max {
            validation.push(row);
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
            action_label: ACTION_FETCH_DOCS,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn state_conditioning(
    rows: &[VeclabTaskRow],
    regime: BridgeRegime,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    world: &LeWorldModel,
    vocab: &crate::model::Vocab,
    max_seq: usize,
    device: &Device,
) -> Result<Tensor> {
    let raw = world_rows(rows, regime);
    let encoded = encode_world_examples(&raw, vocab);
    let raw_state_slots = context_slots_from_world_states(
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
    let state_slots = world.encode(&raw_state_slots, false)?;
    match regime {
        BridgeRegime::Context => Ok(state_slots),
        BridgeRegime::Weights => {
            let actions =
                Tensor::from_vec(vec![ACTION_FETCH_DOCS; rows.len()], rows.len(), device)?;
            world.predict(&state_slots, &actions, false)
        }
    }
}

pub(crate) struct BridgeRuntime {
    pub tokenizer: Tokenizer,
    pub model: Qwen3Bridge,
    pub encoder: Option<OnlineEncoder>,
    pub compressor: Option<ContextCompressor>,
    pub world: Option<LeWorldModel>,
    pub adapter: Option<DecoderConditioningAdapter>,
    static_prefix: Option<candle_nn::Embedding>,
    pub vocab: Option<crate::model::Vocab>,
    pub regime: BridgeRegime,
    pub max_seq: usize,
    pub device: Device,
    output_slots: usize,
    hidden_size: usize,
}

fn go_outer_function_closed(source: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Lex {
        Code,
        DoubleQuote,
        Rune,
        Raw,
        LineComment,
        BlockComment,
    }

    let bytes = source.as_bytes();
    let mut state = Lex::Code;
    let mut escaped = false;
    let mut started = false;
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        match state {
            Lex::Code if byte == b'/' && next == Some(b'/') => {
                state = Lex::LineComment;
                index += 1;
            }
            Lex::Code if byte == b'/' && next == Some(b'*') => {
                state = Lex::BlockComment;
                index += 1;
            }
            Lex::Code if byte == b'"' => state = Lex::DoubleQuote,
            Lex::Code if byte == b'\'' => state = Lex::Rune,
            Lex::Code if byte == b'`' => state = Lex::Raw,
            Lex::Code if byte == b'{' => {
                started = true;
                depth += 1;
            }
            Lex::Code if byte == b'}' && started => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return true;
                }
            }
            Lex::DoubleQuote | Lex::Rune => {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if matches!((state, byte), (Lex::DoubleQuote, b'"') | (Lex::Rune, b'\'')) {
                    state = Lex::Code;
                }
            }
            Lex::Raw if byte == b'`' => state = Lex::Code,
            Lex::LineComment if byte == b'\n' => state = Lex::Code,
            Lex::BlockComment if byte == b'*' && next == Some(b'/') => {
                state = Lex::Code;
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    false
}

fn sample_token_from_logits(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    rng: &mut StdRng,
) -> Result<u32> {
    let mut logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if temperature <= 0.0 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .unwrap_or(0));
    }
    if top_k > 0 && top_k < logits.len() {
        let mut indexed = logits
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        indexed.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        let cutoff = indexed[top_k - 1].1;
        for value in &mut logits {
            if *value < cutoff {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|value| ((f64::from(*value - max)) / temperature).exp() as f32)
        .collect::<Vec<_>>();
    let total = exp.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let draw = rng.random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (index, weight) in exp.iter().enumerate() {
        cumulative += weight;
        if draw <= cumulative {
            return Ok(index as u32);
        }
    }
    Ok((logits.len().saturating_sub(1)) as u32)
}

fn generate_code(
    tokenizer: &Tokenizer,
    model: &Qwen3Bridge,
    task: &str,
    conditioning: &Tensor,
    device: &Device,
    max_new: usize,
    decode: BridgeDecodeConfig,
    sample_index: usize,
) -> Result<String> {
    let text = qwen_example_text(task, "")?;
    let source_prefix = text
        .source_prefix
        .context("code generation requires a source scaffold")?;
    let encoded = tokenizer
        .encode(text.prompt, false)
        .map_err(anyhow::Error::msg)?;
    let mut ids = encoded.get_ids().to_vec();
    let prompt_len = ids.len();
    let eos = Some(qwen_eos_id(tokenizer)?);
    let mut cache = model.new_cache();
    let mut next_input = ids.clone();
    let mut rng = StdRng::seed_from_u64(decode.seed ^ sample_index as u64);
    for _ in 0..max_new {
        let input = Tensor::from_vec(next_input.clone(), (1, next_input.len()), device)?;
        let logits = model.forward_cached(&input, conditioning, &mut cache)?;
        let next = sample_token_from_logits(
            &logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?,
            decode.temperature,
            decode.top_k,
            &mut rng,
        )?;
        ids.push(next);
        if Some(next) == eos {
            break;
        }
        let generated = tokenizer
            .decode(&ids[prompt_len..], true)
            .map_err(anyhow::Error::msg)?;
        if go_outer_function_closed(&format!("{source_prefix}{generated}")) {
            break;
        }
        next_input.clear();
        next_input.push(next);
    }
    let generated = tokenizer
        .decode(&ids[prompt_len..], true)
        .map_err(anyhow::Error::msg)?;
    Ok(format!("{source_prefix}{generated}"))
}

impl BridgeRuntime {
    pub fn load(
        qwen_dir: &Path,
        bridge_path: &Path,
        encoder_path: &Path,
        vocab_path: &Path,
        world_path: &Path,
    ) -> Result<Self> {
        let device = Device::new_cuda(0)
            .context("bridge runtime and evaluation require an available CUDA device 0")?;
        let dtype = DType::BF16;
        let cfg: Qwen3Config =
            serde_json::from_str(&fs::read_to_string(qwen_dir.join("config.json"))?)?;
        let tokenizer =
            Tokenizer::from_file(qwen_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let weights = qwen_weight_paths(qwen_dir)?;
        let base_vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
        let mut bridge_map = VarMap::new();
        let bridge_vb = VarBuilder::from_varmap(&bridge_map, dtype, &device);
        let model = Qwen3Bridge::new(&cfg, base_vb, bridge_vb.pp("qwen_bridge"))?;
        let eval_mode = std::env::var("TOFY_EVAL_MODE").unwrap_or_else(|_| "bridge".into());
        let planner_dim = env_usize("TOFY_BRIDGE_DIM", 640);
        let output_slots = env_usize("TOFY_ADAPTER_OUTPUT_SLOTS", 64);
        if matches!(eval_mode.as_str(), "rag" | "floor") {
            return Ok(Self {
                tokenizer,
                model,
                encoder: None,
                compressor: None,
                world: None,
                adapter: None,
                static_prefix: None,
                vocab: None,
                regime: BridgeRegime::from_env()?,
                max_seq: env_usize("TOFY_BRIDGE_MAX_SEQ", 512),
                device,
                output_slots,
                hidden_size: cfg.hidden_size,
            });
        }
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
        if bridge_path.exists() {
            util::load_varmap_checked(&mut bridge_map, bridge_path)?;
        } else {
            bail!(
                "bridge checkpoint does not exist outside a frozen-decoder control: {}",
                bridge_path.display()
            );
        }
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
        let world = LeWorldModel::new(
            VarBuilder::from_varmap(&world_map, dtype, &device),
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
            encoder: Some(encoder),
            compressor: Some(compressor),
            world: Some(world),
            adapter: Some(adapter),
            static_prefix,
            vocab: Some(vocab),
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
            return Tensor::zeros(
                (rows.len(), 1, self.hidden_size),
                self.model.dtype(),
                &self.device,
            )
            .map_err(Into::into);
        }
        let encoder = self
            .encoder
            .as_ref()
            .context("encoder is unavailable in frozen-decoder control mode")?;
        let compressor = self
            .compressor
            .as_ref()
            .context("compressor is unavailable in frozen-decoder control mode")?;
        let world = self
            .world
            .as_ref()
            .context("world model is unavailable in frozen-decoder control mode")?;
        let vocab = self
            .vocab
            .as_ref()
            .context("encoder vocabulary is unavailable in frozen-decoder control mode")?;
        let slots = state_conditioning(
            rows,
            self.regime,
            encoder,
            compressor,
            world,
            vocab,
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
            self.adapter
                .as_ref()
                .context("adapter is unavailable in frozen-decoder control mode")?
                .forward(&slots)
        }
    }

    pub fn zero_conditioning(&self, batch: usize) -> Result<Tensor> {
        Tensor::zeros(
            (batch, 1, self.hidden_size),
            self.model.dtype(),
            &self.device,
        )
        .map_err(Into::into)
    }

    pub fn output_slots(&self) -> usize {
        self.output_slots
    }

    pub fn generate(&self, prompt: &str, conditioning: &Tensor, max_new: usize) -> Result<String> {
        let eval_mode = std::env::var("TOFY_EVAL_MODE").unwrap_or_else(|_| "bridge".into());
        let prompt = if eval_mode == "bridge" {
            counterfactual_bridge_prompt(prompt)
        } else {
            model_visible_task(prompt).to_string()
        };
        let decode = BridgeDecodeConfig::from_env();
        generate_code(
            &self.tokenizer,
            &self.model,
            &prompt,
            conditioning,
            &self.device,
            max_new,
            decode,
            0,
        )
    }

    pub fn generate_samples(
        &self,
        prompt: &str,
        conditioning: &Tensor,
        max_new: usize,
        decode: BridgeDecodeConfig,
    ) -> Result<Vec<String>> {
        let eval_mode = std::env::var("TOFY_EVAL_MODE").unwrap_or_else(|_| "bridge".into());
        let prompt = if eval_mode == "bridge" {
            counterfactual_bridge_prompt(prompt)
        } else {
            model_visible_task(prompt).to_string()
        };
        (0..decode.samples)
            .map(|sample_index| {
                generate_code(
                    &self.tokenizer,
                    &self.model,
                    &prompt,
                    conditioning,
                    &self.device,
                    max_new,
                    decode,
                    sample_index,
                )
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct BridgeValLosses {
    matched: f32,
    zeroed: f32,
    wrong: f32,
    matched_semantic: f32,
    wrong_semantic: f32,
}

#[allow(clippy::too_many_arguments)]
fn val_losses(
    rows: &[VeclabTaskRow],
    wrong_rows: [&[VeclabTaskRow]; 2],
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    world: &LeWorldModel,
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
            rows, regime, encoder, compressor, world, vocab, max_seq, device,
        )?;
        adapter.forward(&slots)?
    };
    let (input, labels, mask, semantic_mask) = qwen_batch(tokenizer, rows, max_seq, device)?;
    let (matched, matched_semantic) =
        token_losses(model, &input, &labels, &mask, &semantic_mask, &cond, 0.0)?;
    let matched = util::scalar_f32(&matched)?;
    let matched_semantic = util::scalar_f32(&matched_semantic)?;
    let zeroed = util::scalar_f32(&token_loss(
        model,
        &input,
        &labels,
        &mask,
        &cond.zeros_like()?,
    )?)?;
    let mut wrong = f32::INFINITY;
    let mut wrong_semantic = f32::INFINITY;
    for wrong_rows in wrong_rows {
        let wrong_cond = if lora_mode || static_prefix.is_some() {
            cond.clone()
        } else {
            let slots = state_conditioning(
                wrong_rows, regime, encoder, compressor, world, vocab, max_seq, device,
            )?;
            adapter.forward(&slots)?
        };
        let (wrong_full_loss, wrong_semantic_loss) =
            token_losses(model, &input, &labels, &mask, &semantic_mask, &wrong_cond, 0.0)?;
        wrong = wrong.min(util::scalar_f32(&wrong_full_loss)?);
        wrong_semantic = wrong_semantic.min(util::scalar_f32(&wrong_semantic_loss)?);
    }
    Ok(BridgeValLosses {
        matched,
        zeroed,
        wrong,
        matched_semantic,
        wrong_semantic,
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
    world: &LeWorldModel,
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
    let mut matched_semantic = 0.0;
    let mut wrong_semantic = 0.0;
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
            world,
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
        matched_semantic += losses.matched_semantic * chunk.len() as f32;
        wrong_semantic += losses.wrong_semantic * chunk.len() as f32;
    }
    let count = rows.len() as f32;
    Ok(BridgeValLosses {
        matched: matched / count,
        zeroed: zeroed / count,
        wrong: wrong / count,
        matched_semantic: matched_semantic / count,
        wrong_semantic: wrong_semantic / count,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct AutoregressiveBridgeMetrics {
    matched_pass_rate: f32,
    wrong_pass_rate: f32,
    matched_pass_at_k: f32,
    rag_ceiling: f32,
    matched_rag_fraction: f32,
    matched_pass_at_k_rag_fraction: f32,
}

/// Round-robin across functions so the sample spans the whole split.
///
/// Validation rows arrive grouped by function and the corpus repeats rows
/// verbatim, so taking a flat prefix drew 40 copies of function 81 plus 20 of
/// function 82 and reported that as a 20-function transfer rate.
fn stratified_ar_rows(rows: &[VeclabTaskRow], sample_count: usize) -> Vec<VeclabTaskRow> {
    let mut by_function: BTreeMap<usize, Vec<&VeclabTaskRow>> = BTreeMap::new();
    let mut seen = HashSet::new();
    for row in rows {
        if seen.insert((row.function_id, row.task.as_str(), row.completion.as_str())) {
            by_function.entry(row.function_id).or_default().push(row);
        }
    }
    let mut selected = Vec::new();
    let deepest = by_function.values().map(Vec::len).max().unwrap_or(0);
    for depth in 0..deepest {
        for group in by_function.values() {
            if selected.len() == sample_count {
                return selected;
            }
            if let Some(row) = group.get(depth) {
                selected.push((*row).clone());
            }
        }
    }
    selected
}

#[allow(clippy::too_many_arguments)]
fn autoregressive_bridge_metrics(
    rows: &[VeclabTaskRow],
    sample_count: usize,
    label: &str,
    regime: BridgeRegime,
    tokenizer: &Tokenizer,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    world: &LeWorldModel,
    vocab: &crate::model::Vocab,
    adapter: &DecoderConditioningAdapter,
    model: &Qwen3Bridge,
    max_seq: usize,
    device: &Device,
) -> Result<AutoregressiveBridgeMetrics> {
    if rows.is_empty() {
        bail!("autoregressive bridge validation requires at least one row");
    }
    // Reuse the same validation rows as teacher-forced CE. The old path rebuilt
    // rows from eval/veclab_eval.jsonl, whose task phrasing never appears in
    // bridge training and shifted the conditioning encoder off-manifold.
    let selected = stratified_ar_rows(rows, sample_count.max(1));
    let function_ids = selected
        .iter()
        .map(|row| row.function_id)
        .collect::<HashSet<_>>();
    let mut harness_by_fn: HashMap<usize, Vec<EvalTask>> = HashMap::new();
    for task in load_suite(Path::new("eval/veclab_eval.jsonl"))? {
        if let Some(&fn_id) = task.fn_ids.first() {
            if function_ids.contains(&fn_id) {
                harness_by_fn.entry(fn_id).or_default().push(task);
            }
        }
    }
    if harness_by_fn.len() < function_ids.len() {
        bail!("autoregressive bridge validation is missing eval harness tasks");
    }
    // Draw negatives from the full split, not the sample, so the mismatched
    // control still has a signature-matched partner to choose from.
    let wrong_rows = mismatched_rows(rows, &selected, 1)?;
    let matched_slots = state_conditioning(
        &selected, regime, encoder, compressor, world, vocab, max_seq, device,
    )?;
    let wrong_slots = state_conditioning(
        &wrong_rows,
        regime,
        encoder,
        compressor,
        world,
        vocab,
        max_seq,
        device,
    )?;
    let matched = adapter.forward(&matched_slots)?;
    let wrong = adapter.forward(&wrong_slots)?;
    let max_new = env_usize("TOFY_BRIDGE_AR_MAX_NEW", 192);
    let decode = BridgeDecodeConfig::from_env();
    let rag_ceiling = rag_ceiling_for_split_label(label);
    let mut matched_passes = 0usize;
    let mut wrong_passes = 0usize;
    let mut matched_pass_at_k = 0usize;
    let mut matched_categories: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut by_function: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (index, row) in selected.iter().enumerate() {
        let harness = bridge_ar_harness(&harness_by_fn, row)?;
        // Keep the full task only on the world-state side. Qwen receives the
        // same signature-only counterfactual contract used by deployment, so
        // validation cannot leak the requested fictional API in text.
        let qwen_task = counterfactual_bridge_prompt(&row.task);
        let conditioning = matched.narrow(0, index, 1)?;
        let mut sample_passed = false;
        for sample_index in 0..decode.pass_at_k {
            let matched_code = generate_code(
                tokenizer,
                model,
                &qwen_task,
                &conditioning,
                device,
                harness.max_new_tokens.min(max_new),
                decode,
                sample_index,
            )?;
            let matched_category =
                compile_and_test(harness, &matched_code, "bridge-ar-matched")?;
            sample_passed |= matched_category.is_pass();
            if sample_index == 0 {
                matched_passes += usize::from(matched_category.is_pass());
                *matched_categories
                    .entry(matched_category.as_str())
                    .or_insert(0) += 1;
                let tally = by_function.entry(row.function_id).or_insert((0, 0));
                tally.0 += usize::from(matched_category.is_pass());
                tally.1 += 1;
            }
        }
        matched_pass_at_k += usize::from(sample_passed);
        let wrong_code = generate_code(
            tokenizer,
            model,
            &qwen_task,
            &wrong.narrow(0, index, 1)?,
            device,
            harness.max_new_tokens.min(max_new),
            decode,
            0,
        )?;
        wrong_passes +=
            usize::from(compile_and_test(harness, &wrong_code, "bridge-ar-wrong")?.is_pass());
    }
    let matched_pass_rate = matched_passes as f32 / selected.len() as f32;
    let matched_pass_at_k_rate = matched_pass_at_k as f32 / selected.len() as f32;
    // A bare 0.0 pass rate cannot distinguish "named a function that does not
    // exist" from "compiled and returned the wrong value", and those point at
    // different stages. Keep the breakdown next to the rate that summarizes it.
    let breakdown = matched_categories
        .iter()
        .map(|(category, count)| format!("{category}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    // Report the function coverage next to the rate. A sample that collapses
    // onto a few functions reads as a transfer result otherwise.
    let per_function = by_function
        .iter()
        .map(|(function_id, (passed, total))| format!("{function_id}:{passed}/{total}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "bridge ar {label} tasks={} functions={} pass_rate={matched_pass_rate:.4} pass@{k}={matched_pass_at_k_rate:.4} rag_ceiling={rag_ceiling:.4} rag_fraction={:.4} pass@{k}_rag_fraction={:.4} decode={decode:?} breakdown: {breakdown} per_function: {per_function}",
        selected.len(),
        by_function.len(),
        pass_rate_rag_fraction(matched_pass_rate, rag_ceiling),
        pass_rate_rag_fraction(matched_pass_at_k_rate, rag_ceiling),
        k = decode.pass_at_k,
    );
    Ok(AutoregressiveBridgeMetrics {
        matched_pass_rate,
        wrong_pass_rate: wrong_passes as f32 / selected.len() as f32,
        matched_pass_at_k: matched_pass_at_k_rate,
        rag_ceiling,
        matched_rag_fraction: pass_rate_rag_fraction(matched_pass_rate, rag_ceiling),
        matched_pass_at_k_rag_fraction: pass_rate_rag_fraction(matched_pass_at_k_rate, rag_ceiling),
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

/// `smoothing` must stay 0.0 for validation so `val_ce_matched` remains
/// comparable across runs; only the training objective is smoothed.
fn token_losses(
    model: &Qwen3Bridge,
    input: &Tensor,
    labels: &Tensor,
    mask: &Tensor,
    semantic_mask: &Tensor,
    cond: &Tensor,
    smoothing: f64,
) -> Result<(Tensor, Tensor)> {
    let logits = model.forward(input, cond)?;
    let logits = logits.narrow(1, 0, logits.dim(1)? - 1)?;
    Ok((
        masked_cross_entropy_smoothed(&logits, labels, mask, smoothing)?,
        masked_cross_entropy(&logits, labels, semantic_mask)?,
    ))
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

fn conditioning_alignment_loss(
    model: &Qwen3Bridge,
    labels: &Tensor,
    semantic_mask: &Tensor,
    conditioning: &Tensor,
    temperature: f64,
) -> Result<(Tensor, f32)> {
    let (batch, positions) = labels.dims2()?;
    let target_tokens = model.embed_tokens(labels)?.detach();
    let mask = semantic_mask
        .to_dtype(target_tokens.dtype())?
        .reshape((batch, positions, 1))?;
    let positive = fine_grained_token_slot_alignment(&target_tokens, semantic_mask, conditioning)?;
    let denom = semantic_mask
        .sum(1)?
        .clamp(1.0, f64::INFINITY)?
        .unsqueeze(1)?
        .to_dtype(target_tokens.dtype())?;
    let target = target_tokens
        .broadcast_mul(&mask)?
        .sum(1)?
        .broadcast_div(&denom)?;
    let condition = conditioning.mean(1)?;
    let condition_norm = condition
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let target_norm = target
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let condition = condition.broadcast_div(&condition_norm)?;
    let target = target.broadcast_div(&target_norm)?;
    if batch == 1 {
        return Ok((combine_alignment_losses(&positive, None)?, 1.0));
    }
    let logits = condition
        .matmul(&target.t()?)?
        .affine(1.0 / temperature.max(1e-4), 0.0)?
        .to_dtype(DType::F32)?;
    let targets = Tensor::arange(0u32, batch as u32, labels.device())?;
    let forward = candle_nn::loss::cross_entropy(&logits, &targets)?;
    let backward = candle_nn::loss::cross_entropy(&logits.t()?, &targets)?;
    let contrastive = forward.broadcast_add(&backward)?.affine(0.5, 0.0)?;
    let predictions = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?;
    let correct = predictions
        .iter()
        .enumerate()
        .filter(|(index, prediction)| **prediction as usize == *index)
        .count();
    Ok((
        combine_alignment_losses(&positive, Some(&contrastive))?,
        correct as f32 / batch as f32,
    ))
}

fn combine_alignment_losses(positive: &Tensor, contrastive: Option<&Tensor>) -> Result<Tensor> {
    let positive = positive.to_dtype(DType::F32)?;
    match contrastive {
        Some(contrastive) => positive
            .broadcast_add(&contrastive.to_dtype(DType::F32)?)
            .map_err(Into::into),
        None => Ok(positive),
    }
}

/// BLIP-2/late-interaction-style token-to-query alignment.
///
/// Every supervised target token must be represented by at least one adapter
/// slot. A pooled-only cosine can hide missing facts when unrelated slot
/// directions cancel in the mean.
fn fine_grained_token_slot_alignment(
    target_tokens: &Tensor,
    semantic_mask: &Tensor,
    conditioning: &Tensor,
) -> Result<Tensor> {
    let target_norm = target_tokens
        .sqr()?
        .sum_keepdim(2)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let condition_norm = conditioning
        .sqr()?
        .sum_keepdim(2)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let target_tokens = target_tokens.broadcast_div(&target_norm)?;
    let conditioning = conditioning.broadcast_div(&condition_norm)?;
    let similarities = conditioning.matmul(&target_tokens.transpose(1, 2)?)?;
    let best_slot = similarities.max(1)?;
    let mask = semantic_mask.to_dtype(best_slot.dtype())?;
    let numerator = best_slot.broadcast_mul(&mask)?.sum_all()?;
    let denominator = mask.sum_all()?.clamp(1.0, f64::INFINITY)?;
    numerator
        .broadcast_div(&denominator)?
        .affine(-1.0, 1.0)
        .map_err(Into::into)
}

fn semantic_gap_improved(
    semantic_gap: f32,
    best_semantic_gap: f32,
    minimum_gap: f32,
    minimum_progress: f32,
) -> bool {
    semantic_gap >= minimum_gap
        && (!best_semantic_gap.is_finite() || semantic_gap >= best_semantic_gap + minimum_progress)
}

fn val_semantic_ce_improved(val_ce: f32, best_val_ce: f32, minimum_progress: f32) -> bool {
    val_ce.is_finite() && (!best_val_ce.is_finite() || val_ce <= best_val_ce - minimum_progress)
}

/// `best_score` is minimized and `best_semantic_gap` is maximized, so an empty
/// selection has to persist a sentinel at each metric's own worst end. Storing
/// one direction-agnostic sentinel makes a resumed run read "no checkpoint" as a
/// perfect semantic gap and clear the release gate with nothing selected.
fn bridge_selection_resume_state(
    stage: &str,
    step: usize,
    best_score: f32,
    best_semantic_gap: f32,
    terminal: Option<util::TrainingTerminal>,
) -> util::TrainingResumeState {
    let selected = best_score.is_finite();
    util::TrainingResumeState {
        stage: stage.to_string(),
        step,
        best_metric: if selected { best_score } else { f32::MAX },
        best_aux_metric: if selected { best_semantic_gap } else { f32::MIN },
        saved_checkpoint: selected,
        terminal,
    }
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
    let device =
        Device::new_cuda(0).context("bridge logit parity requires an available CUDA device 0")?;
    let dtype = DType::BF16;
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
    let device =
        Device::new_cuda(0).context("bridge training requires an available CUDA device 0")?;
    let dtype = DType::BF16;
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
    let planner_dim = env_usize("TOFY_BRIDGE_DIM", 640);
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
    let world = LeWorldModel::new(
        VarBuilder::from_varmap(&world_map, dtype, &device),
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
    // The bridge contains many Qwen cross-site trainables.  The former 2e-4
    // reached a causal gap quickly but then degraded matched CE; 1e-4 keeps
    // the same effective batch while avoiding that overshoot.
    let lr = env_f64("TOFY_BRIDGE_LR", 1e-4);
    let mut named_vars = util::named_train_vars(&train_vars)?;
    if static_prefix.is_some() || lora_mode {
        named_vars.retain(|entry| !entry.name.starts_with("adapter."));
    }
    if unfreeze_world {
        named_vars.extend(util::named_train_vars(&world_map)?);
        named_vars.retain(|entry| {
            !entry.name.ends_with("running_mean") && !entry.name.ends_with("running_var")
        });
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
    // Exact-zero conditioning is an identity path through the frozen decoder,
    // so dropout contributes no trainable gradient. Hard negatives provide the
    // useful causal signal without spending microbatches on no-op forwards.
    let dropout = env_f64("TOFY_CONDITIONING_DROPOUT", 0.0).clamp(0.0, 1.0);
    let regime = BridgeRegime::from_env()?;
    let all_rows = load_task_rows(&args.data)?;
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt")).unwrap_or_default();
    let mut seen: Vec<_> = all_rows
        .into_iter()
        .filter(|row| {
            row.function_id <= SEEN_FUNCTION_MAX
                || !row.completion.trim_start().starts_with("package solution")
        })
        .collect();
    attach_docs(&mut seen, &docs);
    let (train_rows, val_rows) = split_bridge_rows(seen)?;
    let alignment_rows = train_rows
        .iter()
        .filter(|row| !row.completion.trim_start().starts_with("package solution"))
        .cloned()
        .collect::<Vec<_>>();
    let train_code_rows = train_rows
        .iter()
        .filter(|row| row.completion.trim_start().starts_with("package solution"))
        .cloned()
        .collect::<Vec<_>>();
    if alignment_rows.is_empty() && !lora_mode && static_prefix.is_none() {
        bail!(
            "two-stage bridge training requires documentation rows for latent-language alignment"
        );
    }
    let default_alignment_steps = (args.steps / 5).min(1_000);
    let alignment_steps = if lora_mode || static_prefix.is_some() {
        0
    } else {
        std::env::var("TOFY_BRIDGE_ALIGNMENT_STEPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_alignment_steps)
            .min(args.steps)
    };
    let alignment_temperature = env_f64("TOFY_BRIDGE_ALIGNMENT_TEMPERATURE", 0.07).max(1e-4);
    let train_function_max = env_usize(
        "TOFY_BRIDGE_TRAIN_FUNCTION_MAX",
        SEEN_FUNCTION_MAX.saturating_sub(20),
    );
    let validation_function_max =
        env_usize("TOFY_BRIDGE_VALIDATION_FUNCTION_MAX", SEEN_FUNCTION_MAX);
    println!(
        "Bridge regime={} train_rows={} alignment_rows={} alignment_steps={} val_rows={} function_split=train:1-{train_function_max} val:{}-{validation_function_max} heldout_task_rows=0 full_world_grounding=true",
        regime.as_str(),
        train_rows.len(),
        alignment_rows.len(),
        alignment_steps,
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
    let ar_val_every = env_usize("TOFY_BRIDGE_AR_VAL_EVERY", 500).max(val_every);
    let ar_val_rows = env_usize("TOFY_BRIDGE_AR_VAL_ROWS", 60);
    let ar_train_rows = env_usize("TOFY_BRIDGE_AR_TRAIN_ROWS", 60).min(train_code_rows.len());
    // The pipeline has always recorded TOFY_LABEL_SMOOTHING in the run
    // manifest, but nothing read it, so every run so far reported a smoothing
    // it never applied.
    let label_smoothing = env_f64("TOFY_LABEL_SMOOTHING", 0.0).clamp(0.0, 0.5);
    let min_ar_pass_rate = bridge_min_ar_pass_rate();
    let min_ar_advantage = env_f64("TOFY_BRIDGE_MIN_AR_ADVANTAGE", 0.125).clamp(0.0, 1.0) as f32;
    println!(
        "Bridge AR gate uses min_pass_rate={min_ar_pass_rate:.4} ({:.1}% of seen RAG ceiling {:.4})",
        (min_ar_pass_rate / rag_ceiling_for_function(1) * 100.0),
        rag_ceiling_for_function(1),
    );
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
    let mut resume_metadata = util::ResumeCheckpointMetadata::default();
    if args.resume {
        let loaded_state = util::load_resume_state(&resume_path, &resume_stage)?;
        resume_metadata = util::load_resume_checkpoint_metadata(&resume_path)?;
        let latest_world = latest_path.with_extension("world.safetensors");
        let mut weight_paths = vec![latest_path.as_path()];
        if unfreeze_world {
            weight_paths.push(latest_world.as_path());
        }
        util::validate_resume_checkpoint_tuple(
            loaded_state.as_ref(),
            &resume_metadata,
            &weight_paths,
            &optimizer_path,
        )?;
        if let Some(state) = loaded_state {
            resume_state = state;
        }
        if resume_state.step > 0 {
            util::load_varmap_checked(&mut train_vars, &latest_path)?;
            if unfreeze_world {
                util::load_varmap_checked(&mut world_map, &latest_world)?;
            }
            optimizer.load_state(&optimizer_path)?;
            if optimizer.step_t() != resume_state.step {
                bail!(
                    "bridge optimizer loaded at step {}, expected {}",
                    optimizer.step_t(),
                    resume_state.step
                );
            }
        } else if args.output.exists() || best_path.exists() {
            bail!(
                "cannot --resume bridge training from best export without a complete latest/optimizer/resume tuple"
            );
        }
    }
    resume_metadata.validate_and_set_batch_schedule(args.batch, grad_accum)?;
    let start_step = if args.resume { resume_state.step } else { 0 };
    let min_semantic_gap = env_f64("TOFY_BRIDGE_MIN_SEMANTIC_GAP", 0.02).max(0.0) as f32;
    let requires_semantic_gap = !lora_mode && static_prefix.is_none();
    // Only inherit a selection that is still backed by an exported checkpoint.
    // An empty selection persists sentinels, and trusting those would carry a
    // fake best semantic gap into the release gate.
    let resumed_selection = args.resume
        && resume_state.saved_checkpoint
        && resume_state.best_metric < f32::MAX
        && best_path.exists();
    let mut best_score = if resumed_selection {
        resume_state.best_metric
    } else {
        f32::INFINITY
    };
    let mut best_semantic_gap = if resumed_selection {
        resume_state.best_aux_metric
    } else {
        f32::NEG_INFINITY
    };
    let mut best_ar_pass_rate = 0.0f32;
    let _semantic_patience = env_usize("TOFY_BRIDGE_SEMANTIC_PATIENCE", 1_200);
    let semantic_warmup = env_usize("TOFY_BRIDGE_SEMANTIC_WARMUP", 400);
    let semantic_progress = env_f64("TOFY_BRIDGE_MIN_SEMANTIC_PROGRESS", 0.002).max(0.0) as f32;
    let val_semantic_patience = env_usize("TOFY_BRIDGE_VAL_SEMANTIC_PATIENCE", 1_200);
    let val_semantic_progress =
        env_f64("TOFY_BRIDGE_MIN_VAL_SEMANTIC_PROGRESS", 0.002).max(0.0) as f32;
    let mut best_observed_semantic_gap = resume_metadata
        .best_observed_aux_metric
        .unwrap_or(f32::NEG_INFINITY);
    let mut best_observed_val_semantic_ce = f32::INFINITY;
    let patience_origin = start_step.max(alignment_steps);
    let mut last_semantic_progress_step = resume_metadata
        .last_aux_progress_step
        .unwrap_or(patience_origin)
        .min(start_step)
        .max(alignment_steps);
    let mut last_val_semantic_progress_step = resume_metadata
        .last_improvement_step
        .unwrap_or(patience_origin)
        .min(start_step)
        .max(alignment_steps);
    let mut sampler = BridgeSampler::at_sample(
        train_rows.len(),
        args.seed,
        start_step
            .saturating_sub(alignment_steps)
            .saturating_mul(args.batch)
            .saturating_mul(grad_accum),
    );
    let mut alignment_sampler = (!alignment_rows.is_empty()).then(|| {
        BridgeSampler::at_sample(
            alignment_rows.len(),
            args.seed ^ 0x414c_4947_4e4d_454e,
            start_step.min(alignment_steps) * args.batch * grad_accum,
        )
    });
    println!(
        "Bridge grad_accum={grad_accum} effective_batch={} seed={} start_step={start_step}",
        args.batch * grad_accum,
        args.seed
    );
    let run_dir = util::create_run_dir("bridge")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut completed_step = start_step;
    let mut stopped_early = false;
    for step in (start_step + 1)..=args.steps {
        completed_step = step;
        let alignment_only = step <= alignment_steps;
        let mut accumulated = None;
        let mut loss_sum = 0.0f32;
        let mut positive_sum = 0.0f32;
        let mut positive_semantic_sum = 0.0f32;
        let mut zero_margin_sum = 0.0f32;
        let mut shuffle_margin_sum = 0.0f32;
        let mut hard_margin_sum = 0.0f32;
        let mut wrong_unlikelihood_sum = 0.0f32;
        let mut conditioning_separation_sum = 0.0f32;
        let mut alignment_accuracy_sum = 0.0f32;
        for micro_step in 0..grad_accum {
            let indices = if alignment_only {
                alignment_sampler
                    .as_mut()
                    .context("alignment sampler is unavailable")?
                    .next_batch(args.batch)
            } else {
                sampler.next_batch(args.batch)
            };
            let sample_rows = if alignment_only {
                &alignment_rows
            } else {
                &train_rows
            };
            let batch_rows = indices
                .into_iter()
                .map(|index| sample_rows[index].clone())
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
                    &world,
                    &encoder_vocab,
                    max_seq,
                    &device,
                )?;
                let state_slots = if unfreeze_world && !alignment_only {
                    state_slots
                } else {
                    state_slots.detach()
                };
                adapter.forward(&state_slots)?
            };
            let dropout_seed =
                args.seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ micro_step as u64;
            let conditioning_dropped =
                micro_step > 0 && StdRng::seed_from_u64(dropout_seed).random::<f64>() < dropout;
            if conditioning_dropped {
                cond = cond.zeros_like()?;
            }
            let (input, labels, mask, semantic_mask) =
                qwen_batch(&tokenizer, &batch_rows, max_seq, &device)?;
            if alignment_only {
                let (alignment_loss, alignment_accuracy) = conditioning_alignment_loss(
                    &qwen,
                    &labels,
                    &semantic_mask,
                    &cond,
                    alignment_temperature,
                )?;
                util::accumulate_scaled_gradients(
                    &mut accumulated,
                    &optimizer_vars,
                    &alignment_loss,
                    1,
                )?;
                if step % log_every == 0 {
                    let value = util::scalar_f32(&alignment_loss)?;
                    loss_sum += value;
                    positive_sum += value;
                    alignment_accuracy_sum += alignment_accuracy;
                }
                continue;
            }
            let semantic_negatives = !conditioning_dropped && !lora_mode && static_prefix.is_none();
            let mut negative_values = Vec::new();
            if negatives.zero && !conditioning_dropped {
                let value = util::scalar_f32(&token_loss(
                    &qwen,
                    &input,
                    &labels,
                    &semantic_mask,
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
                    &world,
                    &encoder_vocab,
                    max_seq,
                    &device,
                )?;
                let wrong_cond = adapter.forward(&wrong_slots)?.detach();
                let value = util::scalar_f32(&token_loss(
                    &qwen,
                    &input,
                    &labels,
                    &semantic_mask,
                    &wrong_cond,
                )?)?;
                negative_values.push((Some(offset), value, name));
            }

            let (positive, positive_semantic) =
                token_losses(&qwen, &input, &labels, &mask, &semantic_mask, &cond, label_smoothing)?;
            let positive_value = util::scalar_f32(&positive)?;
            let positive_semantic_value = util::scalar_f32(&positive_semantic)?;
            let active = negative_values
                .iter()
                .map(|(offset, negative, name)| {
                    let margin_value =
                        (positive_semantic_value - negative + margin as f32).max(0.0);
                    (*offset, *name, margin_value)
                })
                .collect::<Vec<_>>();
            let active_count = active.iter().filter(|(_, _, value)| *value > 0.0).count();
            let semantic_weight = margin_weight * active_count as f64;
            let weighted_positive = if semantic_weight > 0.0 {
                positive.broadcast_add(&positive_semantic.affine(semantic_weight, 0.0)?)?
            } else {
                positive
            };
            util::accumulate_scaled_gradients(
                &mut accumulated,
                &optimizer_vars,
                &weighted_positive,
                1,
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
                            &world,
                            &encoder_vocab,
                            max_seq,
                            &device,
                        )?;
                        adapter.forward(&wrong_slots)?
                    }
                };
                let (negative, wrong_unlikelihood) =
                    token_objectives(&qwen, &input, &labels, &semantic_mask, &wrong_cond)?;
                let weighted_negative = negative.affine(-margin_weight, 0.0)?;
                util::accumulate_scaled_gradients(
                    &mut accumulated,
                    &optimizer_vars,
                    &weighted_negative,
                    1,
                )?;
                if unlikelihood_weight > 0.0 {
                    let weighted_unlikelihood =
                        wrong_unlikelihood.affine(unlikelihood_weight, 0.0)?;
                    util::accumulate_scaled_gradients(
                        &mut accumulated,
                        &optimizer_vars,
                        &weighted_unlikelihood,
                        1,
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
                        1,
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
                positive_semantic_sum += positive_semantic_value;
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
        // Scaling the loss itself inserts an autograd multiply at peak Qwen
        // activation memory. Scale detached accumulated gradients instead so
        // production batch sizes do not fail on that transient allocation.
        util::scale_accumulated_gradients(
            &mut accumulated,
            &optimizer_vars,
            1.0 / grad_accum as f64,
        )?;
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
        // Warmup + cosine decay, matching the latent and world stages. A flat
        // 1e-4 across the whole budget let the decoder keep memorizing the
        // training functions long after held-out CE had bottomed out.
        let step_lr = util::scheduled_lr(lr, step, args.steps);
        optimizer.set_learning_rate(step_lr);
        util::optimizer_step_from_accumulated(&mut optimizer, &mut accumulated)?;
        if step % log_every == 0 {
            let divisor = grad_accum as f32;
            tb.add_scalar("loss/total", loss_sum / divisor, step);
            tb.add_scalar("loss/positive_ce", positive_sum / divisor, step);
            tb.add_scalar(
                "loss/positive_semantic_ce",
                positive_semantic_sum / divisor,
                step,
            );
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
            // Only the alignment branch feeds this accumulator, so reporting it
            // during conditional_generation prints a constant 0.000 that reads
            // as total retrieval collapse.
            if alignment_only {
                tb.add_scalar(
                    "alignment/retrieval_top1",
                    alignment_accuracy_sum / divisor,
                    step,
                );
            }
            tb.add_scalar("schedule/lr", step_lr as f32, step);
            if let Some(norm) = grad_norm {
                tb.add_scalar("grad/global_norm", util::scalar_f32(&norm)?, step);
            }
            if alignment_only {
                println!(
                    "bridge step {step}/{} stage=latent_language_alignment loss {:.4} alignment_top1 {:.3}",
                    args.steps,
                    loss_sum / divisor,
                    alignment_accuracy_sum / divisor,
                );
            } else {
                println!(
                    "bridge step {step}/{} stage=conditional_generation loss {:.4}",
                    args.steps,
                    loss_sum / divisor,
                );
            }
        }
        if alignment_only {
            if step % val_every == 0 || step == alignment_steps {
                let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, step);
                resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
                resume_metadata.best_observed_aux_metric = best_observed_semantic_gap
                    .is_finite()
                    .then_some(best_observed_semantic_gap);
                resume_metadata.last_aux_progress_step = Some(last_semantic_progress_step);
                util::save_varmap_resume_checkpoint_atomic(
                    &train_vars,
                    &latest_path,
                    &checkpoint_id,
                )?;
                if unfreeze_world {
                    util::save_varmap_resume_checkpoint_atomic(
                        &world_map,
                        &latest_path.with_extension("world.safetensors"),
                        &checkpoint_id,
                    )?;
                }
                util::save_optimizer_resume_checkpoint_atomic(
                    &optimizer,
                    &optimizer_path,
                    &checkpoint_id,
                )?;
                util::save_resume_state_with_metadata(
                    &resume_path,
                    &bridge_selection_resume_state(
                        &resume_stage,
                        step,
                        best_score,
                        best_semantic_gap,
                        None,
                    ),
                    &resume_metadata,
                )?;
                tb.flush();
            }
            continue;
        }
        if step % val_every == 0 || step == args.steps {
            let losses = full_val_losses(
                &val_rows,
                args.batch,
                regime,
                &tokenizer,
                &encoder,
                &compressor,
                &world,
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
            let semantic_gap = losses.wrong_semantic - losses.matched_semantic;
            let (train_ar_metrics, ar_metrics) = if requires_semantic_gap {
                if step % ar_val_every == 0 || step == args.steps {
                    // The train-function rate is the control that separates "the
                    // conditioning channel cannot carry function identity at all"
                    // from "it carries it but does not generalize". Without it a
                    // held-out rate of zero is unattributable.
                    let train_ar = if ar_train_rows > 0 {
                        Some(autoregressive_bridge_metrics(
                            &train_code_rows,
                            ar_train_rows,
                            "train",
                            regime,
                            &tokenizer,
                            &encoder,
                            &compressor,
                            &world,
                            &encoder_vocab,
                            &adapter,
                            &qwen,
                            max_seq,
                            &device,
                        )?)
                    } else {
                        None
                    };
                    let val_ar = Some(autoregressive_bridge_metrics(
                        &val_rows,
                        ar_val_rows,
                        "val",
                        regime,
                        &tokenizer,
                        &encoder,
                        &compressor,
                        &world,
                        &encoder_vocab,
                        &adapter,
                        &qwen,
                        max_seq,
                        &device,
                    )?);
                    (train_ar, val_ar)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
            if val_semantic_ce_improved(
                losses.matched_semantic,
                best_observed_val_semantic_ce,
                val_semantic_progress,
            ) {
                best_observed_val_semantic_ce = losses.matched_semantic;
                last_val_semantic_progress_step = step;
            }
            if semantic_gap_improved(
                semantic_gap,
                best_observed_semantic_gap,
                min_semantic_gap,
                semantic_progress,
            ) {
                best_observed_semantic_gap = semantic_gap;
                last_semantic_progress_step = step;
            }
            tb.add_scalar("val/ce_matched", losses.matched, step);
            tb.add_scalar("val/ce_zeroed", losses.zeroed, step);
            tb.add_scalar("val/ce_wrong", losses.wrong, step);
            tb.add_scalar("val/semantic_ce_matched", losses.matched_semantic, step);
            tb.add_scalar("val/semantic_ce_wrong", losses.wrong_semantic, step);
            tb.add_scalar("val/zero_gap", zero_gap, step);
            tb.add_scalar("val/semantic_gap", semantic_gap, step);
            if let Some(ar) = ar_metrics {
                tb.add_scalar("val/ar_matched_pass_rate", ar.matched_pass_rate, step);
                tb.add_scalar("val/ar_wrong_pass_rate", ar.wrong_pass_rate, step);
                tb.add_scalar("val/ar_matched_pass_at_k", ar.matched_pass_at_k, step);
                tb.add_scalar("val/ar_matched_rag_fraction", ar.matched_rag_fraction, step);
                tb.add_scalar(
                    "val/ar_matched_pass_at_k_rag_fraction",
                    ar.matched_pass_at_k_rag_fraction,
                    step,
                );
                tb.add_scalar(
                    "val/ar_causal_advantage",
                    ar.matched_pass_rate - ar.wrong_pass_rate,
                    step,
                );
            }
            if let Some(train_ar) = train_ar_metrics {
                tb.add_scalar("train/ar_matched_pass_rate", train_ar.matched_pass_rate, step);
                tb.add_scalar("train/ar_matched_pass_at_k", train_ar.matched_pass_at_k, step);
                tb.add_scalar(
                    "train/ar_matched_rag_fraction",
                    train_ar.matched_rag_fraction,
                    step,
                );
                tb.add_scalar(
                    "train/ar_matched_pass_at_k_rag_fraction",
                    train_ar.matched_pass_at_k_rag_fraction,
                    step,
                );
            }
            if !lora_mode {
                let telemetry_rows = &val_rows[..val_rows.len().min(args.batch.max(1))];
                let slots = state_conditioning(
                    telemetry_rows,
                    regime,
                    &encoder,
                    &compressor,
                    &world,
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
                let (telemetry_input, _, _, _) =
                    qwen_batch(&tokenizer, telemetry_rows, max_seq, &device)?;
                for (site, mean, max) in qwen.gate_statistics(&telemetry_input, &telemetry_cond)? {
                    tb.add_scalar(&format!("gate/site_{site}_mean"), mean, step);
                    tb.add_scalar(&format!("gate/site_{site}_max"), max, step);
                }
            }
            tb.flush();
            println!(
                "bridge val step={step} val_ce_matched={:.4} val_ce_zeroed={:.4} val_ce_wrong={:.4} val_semantic_ce_matched={:.4} val_semantic_ce_wrong={:.4} zero_gap={zero_gap:.4} semantic_gap={semantic_gap:.4} train_ar={train_ar_metrics:?} val_ar={ar_metrics:?}",
                losses.matched,
                losses.zeroed,
                losses.wrong,
                losses.matched_semantic,
                losses.wrong_semantic,
            );
            let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, step);
            resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
            resume_metadata.best_observed_aux_metric = best_observed_semantic_gap
                .is_finite()
                .then_some(best_observed_semantic_gap);
            resume_metadata.last_aux_progress_step = Some(last_semantic_progress_step);
            resume_metadata.last_improvement_step = Some(last_val_semantic_progress_step);
            util::save_varmap_resume_checkpoint_atomic(&train_vars, &latest_path, &checkpoint_id)?;
            if unfreeze_world {
                util::save_varmap_resume_checkpoint_atomic(
                    &world_map,
                    &latest_path.with_extension("world.safetensors"),
                    &checkpoint_id,
                )?;
            }
            if let Some(ar) = ar_metrics {
                let causal_advantage = ar.matched_pass_rate - ar.wrong_pass_rate;
                let eligible = (!requires_semantic_gap || semantic_gap >= min_semantic_gap)
                    && ar.matched_pass_rate >= min_ar_pass_rate
                    && (!requires_semantic_gap || causal_advantage >= min_ar_advantage);
                // A one-point change in compile-and-harness pass rate dominates
                // any plausible teacher-forced CE movement. CE only breaks
                // ties between checkpoints with the same deployment behavior.
                // Tie-break on the identifier-span CE, not the whole-completion
                // CE. The latter is dominated by boilerplate the decoder always
                // gets right, so it barely moved between checkpoints that
                // differed sharply in whether they recovered the function name.
                let selection_score = (1.0 - ar.matched_pass_rate) * 100.0
                    + ar.wrong_pass_rate * 25.0
                    + losses.matched_semantic;
                tb.add_scalar("val/selection_score", selection_score, step);
                if eligible && selection_score < best_score {
                    best_score = selection_score;
                    best_semantic_gap = semantic_gap;
                    best_ar_pass_rate = ar.matched_pass_rate;
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
            }
            util::save_optimizer_resume_checkpoint_atomic(
                &optimizer,
                &optimizer_path,
                &checkpoint_id,
            )?;
            util::save_resume_state_with_metadata(
                &resume_path,
                &bridge_selection_resume_state(
                    &resume_stage,
                    step,
                    best_score,
                    best_semantic_gap,
                    None,
                ),
                &resume_metadata,
            )?;
            if requires_semantic_gap
                && val_semantic_patience > 0
                && step >= semantic_warmup
                && step.saturating_sub(last_val_semantic_progress_step) >= val_semantic_patience
            {
                if best_score.is_finite() {
                    println!(
                        "Bridge early stopping at step {step}: val_semantic_ce={best_observed_val_semantic_ce:.4}, improvement={val_semantic_progress:.4}, patience={val_semantic_patience}"
                    );
                    stopped_early = true;
                    break;
                }
                write_bridge_nonqualification_status(
                    BridgeNonqualificationReason::SemanticPlateau,
                    step,
                )?;
                bail!(
                    "val semantic CE plateau without a qualifying checkpoint at step {step}: best_val_semantic_ce={best_observed_val_semantic_ce:.4}, improvement={val_semantic_progress:.4}, patience={val_semantic_patience}"
                );
            }
        }
    }
    if !best_score.is_finite() || (requires_semantic_gap && best_semantic_gap < min_semantic_gap) {
        tb.finish()?;
        write_bridge_nonqualification_status(
            BridgeNonqualificationReason::BudgetExhausted,
            completed_step,
        )?;
        bail!(
            "no bridge checkpoint passed the joint autoregressive/causal gate: selected_semantic_gap={best_semantic_gap:.4}, best_observed_semantic_gap={best_observed_semantic_gap:.4}, required_semantic_gap={min_semantic_gap:.4}, required_ar_pass_rate={min_ar_pass_rate:.4}, required_ar_advantage={min_ar_advantage:.4}; latest={}",
            latest_path.display()
        );
    }
    println!(
        "Best bridge saved to {} (selection_score={best_score:.4}, ar_pass_rate={best_ar_pass_rate:.4}, semantic_gap={best_semantic_gap:.4}); latest={}",
        args.output.display(),
        latest_path.display()
    );
    resume_metadata.best_observed_aux_metric = best_observed_semantic_gap
        .is_finite()
        .then_some(best_observed_semantic_gap);
    resume_metadata.last_aux_progress_step = Some(last_semantic_progress_step);
    util::save_resume_state_with_metadata(
        &resume_path,
        &bridge_selection_resume_state(
            &resume_stage,
            completed_step,
            best_score,
            best_semantic_gap,
            Some(if stopped_early {
                util::TrainingTerminal::EarlyStopped
            } else {
                util::TrainingTerminal::TargetReached
            }),
        ),
        &resume_metadata,
    )?;
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
    fn empty_selection_never_persists_a_qualifying_semantic_gap() {
        let state = bridge_selection_resume_state(
            "bridge_context",
            3_500,
            f32::INFINITY,
            f32::NEG_INFINITY,
            Some(util::TrainingTerminal::EarlyStopped),
        );
        assert!(!state.saved_checkpoint);
        assert_eq!(state.best_metric, f32::MAX);
        // The gap is maximized, so an empty selection must persist the low end;
        // f32::MAX here would clear every `best_semantic_gap >= min` release gate.
        assert_eq!(state.best_aux_metric, f32::MIN);
        assert!(state.best_aux_metric < 0.02);
    }

    #[test]
    fn selected_checkpoint_persists_its_own_metrics() {
        let state = bridge_selection_resume_state("bridge_context", 900, 12.5, 0.31, None);
        assert!(state.saved_checkpoint);
        assert_eq!(state.best_metric, 12.5);
        assert_eq!(state.best_aux_metric, 0.31);
    }

    #[test]
    fn fine_grained_alignment_requires_each_target_token_in_some_slot() -> Result<()> {
        let device = Device::Cpu;
        let target = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0], (1, 2, 2), &device)?;
        let mask = Tensor::ones((1, 2), DType::F32, &device)?;
        let complete_slots = target.clone();
        let collapsed_slots = Tensor::from_vec(vec![1f32, 0.0, 1.0, 0.0], (1, 2, 2), &device)?;

        let complete = util::scalar_f32(&fine_grained_token_slot_alignment(
            &target,
            &mask,
            &complete_slots,
        )?)?;
        let collapsed = util::scalar_f32(&fine_grained_token_slot_alignment(
            &target,
            &mask,
            &collapsed_slots,
        )?)?;

        assert!(complete.abs() < 1e-6);
        assert!(collapsed > complete + 0.4);
        Ok(())
    }

    #[test]
    fn alignment_loss_combines_bf16_positive_and_f32_contrastive() -> Result<()> {
        let device = Device::Cpu;
        let positive = Tensor::new(0.25f32, &device)?.to_dtype(DType::BF16)?;
        let contrastive = Tensor::new(0.75f32, &device)?;

        let combined = combine_alignment_losses(&positive, Some(&contrastive))?;
        assert_eq!(combined.dtype(), DType::F32);
        assert!((util::scalar_f32(&combined)? - 1.0).abs() < 1e-6);

        let positive_only = combine_alignment_losses(&positive, None)?;
        assert_eq!(positive_only.dtype(), DType::F32);
        assert!((util::scalar_f32(&positive_only)? - 0.25).abs() < 1e-6);
        Ok(())
    }

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
    fn semantic_completion_spans_target_api_identifiers() {
        let completion = "package solution\n\nfunc Solve(xs []float64, k int) float64 {\n    return veclab.Alpha(xs, k) + veclab.Beta(xs, k)\n}\n";
        let spans = semantic_completion_spans(completion);
        assert_eq!(
            spans
                .into_iter()
                .map(|(start, end)| &completion[start..end])
                .collect::<Vec<_>>(),
            ["Alpha", "Beta"]
        );
    }

    #[test]
    fn eval_counterfactual_prompt_keeps_only_the_signature() {
        let prompt = counterfactual_bridge_prompt(
            "Evaluation harness: `func Solve(values []float64, n int) float64` must call a hidden API.",
        );
        assert!(prompt.ends_with("func Solve(values []float64, n int) float64"));
        assert!(!prompt.contains("hidden API"));
    }

    #[test]
    fn code_examples_train_only_the_body_after_a_source_scaffold() -> Result<()> {
        let completion = "package solution\n\nimport \"veclab.dev/veclab\"\n\nfunc Solve(xs []float64) float64 {\n    return veclab.Alpha(xs)\n}\n";
        let text = qwen_example_text("func Solve(xs []float64) float64", completion)?;
        assert!(text.prompt.contains("package solution"));
        assert!(text.prompt.ends_with("func Solve(xs []float64) float64 {"));
        assert_eq!(
            text.source_prefix.as_deref(),
            Some(&completion[..completion.find('{').unwrap() + 1])
        );
        assert_eq!(text.target, "\n    return veclab.Alpha(xs)\n}\n");
        Ok(())
    }

    #[test]
    fn generation_stops_only_after_the_outer_go_body_closes() {
        assert!(!go_outer_function_closed(
            "package solution\nfunc Solve() string {\nreturn \"}\""
        ));
        assert!(!go_outer_function_closed(
            "package solution\nfunc Solve() int {\nif true { return 1 }"
        ));
        assert!(go_outer_function_closed(
            "package solution\nfunc Solve() int {\nif true { return 1 }\nreturn 0\n}"
        ));
        assert!(!go_outer_function_closed(
            "package solution\nfunc Solve() string {\nreturn `}`"
        ));
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
        let (train, validation) = split_bridge_rows(rows)?;
        assert!(train.iter().all(|row| row.function_id <= 80));
        assert!(validation
            .iter()
            .all(|row| (81..=100).contains(&row.function_id)));
        Ok(())
    }

    #[test]
    fn stratified_ar_rows_spread_across_functions_and_drop_duplicates() {
        // Validation rows arrive grouped by function and repeat verbatim. A
        // flat prefix took 40 copies of one function and reported it as a
        // 20-function transfer rate.
        let mut rows = Vec::new();
        for function_id in 81..=100 {
            for variant in 0..40 {
                rows.push(VeclabTaskRow {
                    task: format!("[fn:{function_id}] variant {}", variant % 4),
                    completion: "package solution\nfunc Solve() {}".into(),
                    function_id,
                    docs: String::new(),
                });
            }
        }
        let selected = stratified_ar_rows(&rows, 60);
        assert_eq!(selected.len(), 60);
        let functions = selected
            .iter()
            .map(|row| row.function_id)
            .collect::<HashSet<_>>();
        assert_eq!(functions.len(), 20, "sample must span every val function");
        let distinct = selected
            .iter()
            .map(|row| (row.function_id, row.task.as_str()))
            .collect::<HashSet<_>>();
        assert_eq!(distinct.len(), 60, "duplicate rows must not be sampled");
    }

    #[test]
    fn bridge_grounding_keeps_heldout_docs_but_not_heldout_code() -> Result<()> {
        let rows = vec![
            VeclabTaskRow {
                task: "[fn:150] docs query".into(),
                completion: "func Heldout() documentation".into(),
                function_id: 150,
                docs: String::new(),
            },
            VeclabTaskRow {
                task: "[fn:150] code task".into(),
                completion: "package solution\nfunc Solve() {}".into(),
                function_id: 150,
                docs: String::new(),
            },
            VeclabTaskRow {
                task: "[fn:090] validation".into(),
                completion: "package solution\nfunc Solve() {}".into(),
                function_id: 90,
                docs: String::new(),
            },
        ];
        let (train, validation) = split_bridge_rows(rows)?;
        assert_eq!(train.len(), 1);
        assert_eq!(train[0].function_id, 150);
        assert!(!train[0].completion.starts_with("package solution"));
        assert_eq!(validation.len(), 1);
        Ok(())
    }

    #[test]
    fn semantic_plateau_requires_a_new_qualified_best() {
        assert!(semantic_gap_improved(0.03, f32::NEG_INFINITY, 0.02, 0.002));
        assert!(!semantic_gap_improved(0.031, 0.03, 0.02, 0.002));
        assert!(semantic_gap_improved(0.032, 0.03, 0.02, 0.002));
        assert!(!semantic_gap_improved(0.10, 0.11, 0.02, 0.002));
        assert!(!semantic_gap_improved(
            0.019,
            f32::NEG_INFINITY,
            0.02,
            0.002
        ));
    }

    #[test]
    fn empty_completion_signature_matches_from_task_text() {
        let rows = [
            VeclabTaskRow {
                task: "Write `func Solve(xs []float64, k int) float64` that returns veclab.Alpha(xs, k)."
                    .into(),
                completion: String::new(),
                function_id: 1,
                docs: String::new(),
            },
            VeclabTaskRow {
                task: "Write `func Solve(xs []float64, k int) float64` that returns veclab.Beta(xs, k)."
                    .into(),
                completion:
                    "package solution\n\nfunc Solve(xs []float64, k int) float64 { return veclab.Beta(xs, k) }"
                        .into(),
                function_id: 2,
                docs: String::new(),
            },
        ];
        assert!(same_bridge_signature(&rows[0], &rows[1]));
    }

    #[test]
    fn val_semantic_ce_improvement_tracks_a_decrease() {
        assert!(val_semantic_ce_improved(1.5, f32::INFINITY, 0.01));
        assert!(!val_semantic_ce_improved(1.495, 1.5, 0.01));
        assert!(val_semantic_ce_improved(1.48, 1.5, 0.01));
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

    #[test]
    fn pass_rate_rag_fraction_normalizes_against_ceiling() {
        assert!((pass_rate_rag_fraction(0.07, 0.35) - 0.2).abs() < 1e-6);
        assert!((pass_rate_rag_fraction(0.0, 0.35)).abs() < 1e-6);
    }
}
