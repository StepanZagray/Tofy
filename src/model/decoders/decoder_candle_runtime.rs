//! Decoder runtime: Candle cross-attention decoders for code and text.
//! - Code: JEPA_USE_CANDLE_DECODER=1, JEPA_CANDLE_DECODER=<path>
//! - Text: JEPA_USE_TEXT_DECODER=1, JEPA_TEXT_DECODER=<path>

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::RngExt;
use std::fs;
use std::path::{Path, PathBuf};

use super::decoder_cross::DecoderGenerationState;
use super::{
    CodeDecoder, DecoderArchitecture, DecoderAttentionConfig, DecoderConditioningAdapter,
    DecoderCrossAttentionSchedule, DecoderKind, LocalDecoderRuntime,
};
use crate::data::{
    encode_text_with_vocab_mode, TokenizationMode, CODE_CONTROL_TOKENS, CODE_EOS_TOKEN,
};
use crate::model::vocab::{load_vocab_from_file, vocab_signature, Vocab};

const DECODER_STOP_TOKENS: &[&str] = &[
    CODE_EOS_TOKEN,
    "<eos>",
    "</s>",
    "<|endoftext|>",
    "<|end_of_text|>",
];

#[derive(Clone, Debug)]
struct TreeCoderConfig {
    enabled: bool,
    width: usize,
    branch: usize,
    max_expansions: usize,
    min_complete_tokens: usize,
    length_penalty: f32,
    constraint_weight: f32,
    no_comments: bool,
}

#[derive(Clone)]
struct TreeCoderNode {
    state: DecoderGenerationState,
    generated: Vec<u32>,
    constraints: CodeTreeConstraintState,
    logprob: f32,
    score: f32,
}

#[derive(Clone, Debug, Default)]
struct CodeTreeConstraintState {
    stack: Vec<char>,
    quote: Option<char>,
    escaped: bool,
    emitted_non_ws: bool,
    saw_code_signal: bool,
    token_count: usize,
}

/// Decoder backend that runs the Candle CodeDecoder with cross-attention to the world latent sequence.
/// Expects conditioning to be the **flattened** latent sequence (length = num_latent_tokens * world_dim).
pub struct CandleCrossAttnDecoder {
    adapter: DecoderConditioningAdapter,
    decoder: CodeDecoder,
    vocab: Vocab,
    device: Device,
    runtime_dtype: DType,
    kind: DecoderKind,
    planner_dim: usize,
    temperature: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    top_k: usize,
    top_p: f32,
    tokenization_mode: TokenizationMode,
    max_prompt_tokens: Option<usize>,
}

impl TreeCoderConfig {
    fn from_env(kind: DecoderKind, action: &str, max_new_tokens: usize) -> Self {
        let action_is_code = action.trim().eq_ignore_ascii_case("code");
        let default_enabled = kind == DecoderKind::CodeSpecialist && action_is_code;
        let enabled = runtime_env_bool_any(
            &["TOFY_DECODER_TREECODER", "JEPA_DECODER_TREECODER"],
            default_enabled,
        );
        let width = runtime_env_usize("TOFY_DECODER_TREECODER_WIDTH", 3).clamp(1, 16);
        let branch = runtime_env_usize("TOFY_DECODER_TREECODER_BRANCH", width.max(3) * 2)
            .max(width)
            .min(64);
        let max_expansions = runtime_env_usize(
            "TOFY_DECODER_TREECODER_MAX_EXPANSIONS",
            max_new_tokens.saturating_mul(width).saturating_mul(branch),
        )
        .max(width)
        .min(100_000);
        Self {
            enabled,
            width,
            branch,
            max_expansions,
            min_complete_tokens: runtime_env_usize("TOFY_DECODER_TREECODER_MIN_TOKENS", 8).max(1),
            length_penalty: runtime_env_f32("TOFY_DECODER_TREECODER_LENGTH_PENALTY", 0.7)
                .clamp(0.0, 2.0),
            constraint_weight: runtime_env_f32("TOFY_DECODER_TREECODER_CONSTRAINT_WEIGHT", 0.25)
                .clamp(0.0, 4.0),
            no_comments: runtime_env_bool_any(&["TOFY_DECODER_TREECODER_NO_COMMENTS"], false),
        }
    }
}

impl CodeTreeConstraintState {
    fn advance(&self, raw_token: &str, decoded_token: &str, cfg: &TreeCoderConfig) -> Option<Self> {
        if treecoder_forbidden_token(raw_token, decoded_token, cfg) {
            return None;
        }
        let mut next = self.clone();
        next.token_count += 1;
        if decoded_token.chars().any(|ch| !ch.is_whitespace()) {
            next.emitted_non_ws = true;
        }
        if decoded_token.contains("func ")
            || decoded_token == "func"
            || decoded_token.contains("package ")
            || decoded_token.contains("return")
            || decoded_token.contains(":=")
        {
            next.saw_code_signal = true;
        }
        for ch in decoded_token.chars() {
            next.consume_char(ch)?;
        }
        Some(next)
    }

    fn consume_char(&mut self, ch: char) -> Option<()> {
        if let Some(quote) = self.quote {
            if quote != '`' && self.escaped {
                self.escaped = false;
            } else if quote != '`' && ch == '\\' {
                self.escaped = true;
            } else if ch == quote {
                self.quote = None;
            }
            return Some(());
        }
        match ch {
            '"' | '\'' | '`' => self.quote = Some(ch),
            '(' | '[' | '{' => self.stack.push(ch),
            ')' if self.stack.pop() != Some('(') => return None,
            ']' if self.stack.pop() != Some('[') => return None,
            '}' if self.stack.pop() != Some('{') => return None,
            ')' | ']' | '}' => {}
            _ => {}
        }
        Some(())
    }

    fn can_complete(&self) -> bool {
        self.emitted_non_ws && self.quote.is_none() && self.stack.is_empty()
    }

    fn partial_score(&self) -> f32 {
        let mut score = 0.0f32;
        if self.saw_code_signal {
            score += 0.25;
        }
        score -= 0.08 * self.stack.len() as f32;
        if self.quote.is_some() {
            score -= 0.35;
        }
        score
    }

    fn final_score(&self) -> f32 {
        if self.can_complete() {
            self.partial_score() + 0.5
        } else {
            self.partial_score() - 0.75
        }
    }
}

impl CandleCrossAttnDecoder {
    pub fn metadata_path(checkpoint_path: &Path) -> PathBuf {
        checkpoint_path.with_extension("meta.txt")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_metadata(
        checkpoint_path: &Path,
        vocab: &Vocab,
        kind: DecoderKind,
        planner_dim: usize,
        context_slots: usize,
        architecture: DecoderArchitecture,
        attention: DecoderAttentionConfig,
        adapter_compress_rate: usize,
    ) -> Result<()> {
        let metadata = format!(
            "kind={}\nvocab_signature={}\nvocab_size={}\nplanner_dim={}\ncontext_slots={}\nconditioner=action_aware_local_plan_v2\ndecoder_arch=rope_rmsnorm_swiglu_tied_v3\ndecoder_dim={}\ndecoder_layers={}\ndecoder_heads={}\ndecoder_ff_dim={}\ndecoder_adapter_compress_rate={}\ndecoder_local_window={}\ndecoder_anchor_period={}\ndecoder_csa_compress_rate={}\ndecoder_hca_compress_rate={}\ndecoder_csa_topk={}\ndecoder_cross_attention_schedule={}\ndecoder_latent_prefix={}\n",
            kind.as_str(),
            vocab_signature(vocab),
            vocab.id_to_token.len(),
            planner_dim,
            context_slots,
            architecture.dim,
            architecture.num_layers,
            architecture.num_heads,
            architecture.ff_dim,
            adapter_compress_rate,
            attention.local_window,
            attention.anchor_period,
            attention.csa_compress_rate,
            attention.hca_compress_rate,
            attention.csa_topk,
            attention.cross_attention_schedule.as_str(),
            if attention.latent_prefix { "true" } else { "false" }
        );
        fs::write(Self::metadata_path(checkpoint_path), metadata)?;
        Ok(())
    }

    fn load_metadata_config(
        checkpoint_path: &Path,
        vocab: &Vocab,
        kind: DecoderKind,
        planner_dim: usize,
        context_slots: usize,
    ) -> Result<(DecoderArchitecture, DecoderAttentionConfig, usize)> {
        let metadata_path = Self::metadata_path(checkpoint_path);
        if !metadata_path.exists() {
            anyhow::bail!(
                "decoder metadata not found for {:?}; runtime requires checkpoint metadata with decoder architecture and attention config",
                checkpoint_path
            );
        }
        let metadata = fs::read_to_string(&metadata_path)
            .with_context(|| format!("read decoder metadata from {:?}", metadata_path))?;
        let mut parsed = std::collections::HashMap::new();
        for line in metadata.lines() {
            if let Some((key, value)) = line.split_once('=') {
                parsed.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        if let Some(saved_kind) = parsed.get("kind") {
            if saved_kind != kind.as_str() {
                anyhow::bail!(
                    "decoder kind mismatch for {:?}: metadata says {}, runtime requested {}",
                    checkpoint_path,
                    saved_kind,
                    kind.as_str()
                );
            }
        }
        if let Some(saved_sig) = parsed.get("vocab_signature") {
            let current_sig = vocab_signature(vocab);
            if *saved_sig != current_sig {
                anyhow::bail!(
                    "decoder vocab mismatch for {:?}: metadata signature {} does not match current {}",
                    checkpoint_path,
                    saved_sig,
                    current_sig
                );
            }
        }
        if let Some(saved_dim) = parsed.get("planner_dim") {
            if saved_dim.parse::<usize>().ok() != Some(planner_dim) {
                anyhow::bail!(
                    "decoder planner_dim mismatch for {:?}: metadata says {}, runtime requested {}",
                    checkpoint_path,
                    saved_dim,
                    planner_dim
                );
            }
        }
        if let Some(saved_slots) = parsed.get("context_slots") {
            if saved_slots.parse::<usize>().ok() != Some(context_slots) {
                anyhow::bail!(
                    "decoder context_slots mismatch for {:?}: metadata says {}, runtime requested {}",
                    checkpoint_path,
                    saved_slots,
                    context_slots
                );
            }
        }
        let require_metadata_value = |key: &str, expected: &str| -> Result<()> {
            let value = parsed.get(key).ok_or_else(|| {
                anyhow::anyhow!("decoder metadata {:?} is missing {}", metadata_path, key)
            })?;
            if value != expected {
                anyhow::bail!(
                    "decoder metadata {:?} has {}={}, expected {}",
                    metadata_path,
                    key,
                    value,
                    expected
                );
            }
            Ok(())
        };
        require_metadata_value("conditioner", "action_aware_local_plan_v2")?;
        require_metadata_value("decoder_arch", "rope_rmsnorm_swiglu_tied_v3")?;
        let parse_required = |key: &str| -> Result<usize> {
            parsed
                .get(key)
                .ok_or_else(|| {
                    anyhow::anyhow!("decoder metadata {:?} is missing {}", metadata_path, key)
                })?
                .parse()
                .with_context(|| format!("parse {} from {:?}", key, metadata_path))
        };
        let parse_required_bool = |key: &str| -> Result<bool> {
            let value = parsed
                .get(key)
                .ok_or_else(|| {
                    anyhow::anyhow!("decoder metadata {:?} is missing {}", metadata_path, key)
                })?
                .trim()
                .to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "yes" => Ok(true),
                "0" | "false" | "no" => Ok(false),
                _ => anyhow::bail!("parse {} from {:?}: {}", key, metadata_path, value),
            }
        };
        let architecture = DecoderArchitecture::new(
            parse_required("decoder_dim")?,
            parse_required("decoder_layers")?,
            parse_required("decoder_heads")?,
            parse_required("decoder_ff_dim")?,
        )?;
        let schedule = parsed
            .get("decoder_cross_attention_schedule")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "decoder metadata {:?} is missing decoder_cross_attention_schedule",
                    metadata_path
                )
            })
            .and_then(|value| {
                DecoderCrossAttentionSchedule::from_flag(value).ok_or_else(|| {
                    anyhow::anyhow!(
                        "parse decoder_cross_attention_schedule from {:?}: {}",
                        metadata_path,
                        value
                    )
                })
            })?;
        let attention = DecoderAttentionConfig::new(
            parse_required("decoder_local_window")?,
            parse_required("decoder_anchor_period")?,
            parse_required("decoder_csa_compress_rate")?,
            parse_required("decoder_hca_compress_rate")?,
            parse_required("decoder_csa_topk")?,
            schedule,
            parse_required_bool("decoder_latent_prefix")?,
        )?;
        let adapter_compress_rate = parse_required("decoder_adapter_compress_rate")?;
        if adapter_compress_rate == 0 {
            anyhow::bail!(
                "decoder metadata {:?} has zero decoder_adapter_compress_rate",
                metadata_path
            );
        }
        Ok((architecture, attention, adapter_compress_rate))
    }

    fn infer_vocab_path(checkpoint_path: &Path) -> Result<PathBuf> {
        let inferred = checkpoint_path.with_extension("vocab.txt");
        if inferred.exists() {
            Ok(inferred)
        } else {
            Err(anyhow::anyhow!(
                "decoder vocab not found for {:?}; expected {:?}",
                checkpoint_path,
                inferred
            ))
        }
    }

    /// Load CodeDecoder from checkpoint using the architecture stored in checkpoint metadata.
    pub fn new(
        checkpoint_path: PathBuf,
        vocab_path: PathBuf,
        planner_dim: usize,
        world_dim: usize,
        context_slots: usize,
        temperature: f32,
        kind: DecoderKind,
    ) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let checkpoint_dtype =
            crate::util::checkpoint_float_dtype(&checkpoint_path)?.unwrap_or(DType::F32);
        let runtime_dtype = crate::util::resolve_runtime_dtype(&device);
        if runtime_dtype != checkpoint_dtype {
            anyhow::bail!(
                "decoder runtime dtype {:?} does not match checkpoint dtype {:?} for {:?}",
                runtime_dtype,
                checkpoint_dtype,
                checkpoint_path
            );
        }
        let mut varmap = VarMap::new();
        let vocab = load_vocab_from_file(&vocab_path)
            .with_context(|| format!("load decoder vocab from {:?}", vocab_path))?;
        let (architecture, attention, adapter_compress_rate) =
            Self::load_metadata_config(&checkpoint_path, &vocab, kind, planner_dim, context_slots)?;
        let vocab_size = vocab.id_to_token.len();
        let default_repeat_penalty = if kind == DecoderKind::CodeSpecialist {
            1.12
        } else {
            1.08
        };
        let default_repeat_last_n = if kind == DecoderKind::CodeSpecialist {
            160
        } else {
            96
        };
        let default_top_k = if kind == DecoderKind::CodeSpecialist {
            40
        } else {
            0
        };
        let default_top_p = if kind == DecoderKind::CodeSpecialist {
            0.92
        } else {
            1.0
        };
        let adapter = DecoderConditioningAdapter::new_with_compress_rate(
            VarBuilder::from_varmap(&varmap, checkpoint_dtype, &device)
                .pp("decoder_conditioning_adapter"),
            planner_dim,
            world_dim,
            DecoderConditioningAdapter::output_slots_for(kind, context_slots),
            adapter_compress_rate,
        )?;
        let decoder = CodeDecoder::new_with_attention_config(
            VarBuilder::from_varmap(&varmap, checkpoint_dtype, &device).pp("decoder"),
            vocab_size,
            architecture.dim,
            world_dim,
            architecture.num_layers,
            architecture.num_heads,
            architecture.ff_dim,
            kind,
            attention,
        )?;
        varmap
            .load(&checkpoint_path)
            .with_context(|| format!("load code decoder from {:?}", checkpoint_path))?;
        crate::util::cast_varmap_dtype(&mut varmap, runtime_dtype)?;
        Ok(Self {
            adapter,
            decoder,
            vocab,
            device,
            runtime_dtype,
            kind,
            planner_dim,
            temperature,
            repeat_penalty: std::env::var("JEPA_DECODER_REPEAT_PENALTY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_repeat_penalty),
            repeat_last_n: std::env::var("JEPA_DECODER_REPEAT_LAST_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_repeat_last_n),
            top_k: std::env::var("JEPA_DECODER_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_top_k),
            top_p: std::env::var("JEPA_DECODER_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_top_p),
            tokenization_mode: if kind == DecoderKind::CodeSpecialist {
                TokenizationMode::CodeAware
            } else {
                TokenizationMode::Default
            },
            max_prompt_tokens: std::env::var("JEPA_CANDLE_DECODER_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v: &usize| v > 0),
        })
    }

    /// Code decoder: JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER=<path>. Optional JEPA_DECODER_TEMP.
    pub fn try_new_from_env_code(planner_dim: usize, context_slots: usize) -> Result<Self> {
        let use_candle = std::env::var("JEPA_USE_CANDLE_DECODER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let path = std::env::var("JEPA_CANDLE_DECODER").ok();
        match (use_candle, path) {
            (true, Some(p)) => {
                let checkpoint_path = PathBuf::from(p);
                let vocab_path = std::env::var("JEPA_CANDLE_DECODER_VOCAB")
                    .map(PathBuf::from)
                    .ok()
                    .unwrap_or(Self::infer_vocab_path(&checkpoint_path)?);
                let temp = std::env::var("JEPA_DECODER_TEMP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.35);
                Self::new(
                    checkpoint_path,
                    vocab_path,
                    planner_dim,
                    planner_dim,
                    context_slots,
                    temp,
                    DecoderKind::CodeSpecialist,
                )
            }
            _ => Err(anyhow::anyhow!(
                "JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER=<path> required"
            )),
        }
    }

    /// Text decoder: JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER=<path>. Optional JEPA_TEXT_DECODER_TEMP (default 0.7).
    /// Same architecture as code decoder; trained on dialog data (e.g. ultrachat_pairs) for general text reply.
    pub fn try_new_from_env_text(planner_dim: usize, context_slots: usize) -> Result<Self> {
        let use_text = std::env::var("JEPA_USE_TEXT_DECODER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let path = std::env::var("JEPA_TEXT_DECODER").ok();
        match (use_text, path) {
            (true, Some(p)) => {
                let checkpoint_path = PathBuf::from(p);
                let vocab_path = std::env::var("JEPA_TEXT_DECODER_VOCAB")
                    .map(PathBuf::from)
                    .ok()
                    .unwrap_or(Self::infer_vocab_path(&checkpoint_path)?);
                let temp = std::env::var("JEPA_TEXT_DECODER_TEMP")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.7);
                Self::new(
                    checkpoint_path,
                    vocab_path,
                    planner_dim,
                    planner_dim,
                    context_slots,
                    temp,
                    DecoderKind::TextGeneralist,
                )
            }
            _ => Err(anyhow::anyhow!(
                "JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER=<path> required"
            )),
        }
    }

    fn generate_treecoder(
        &self,
        prompt_ids: &[u32],
        world_latent: &Tensor,
        max_new_tokens: usize,
        cfg: TreeCoderConfig,
    ) -> Result<String> {
        let root_state = self
            .decoder
            .begin_generation(&self.device, prompt_ids, world_latent)?;
        let mut active = vec![TreeCoderNode {
            state: root_state,
            generated: Vec::new(),
            constraints: CodeTreeConstraintState::default(),
            logprob: 0.0,
            score: 0.0,
        }];
        let mut complete = Vec::new();
        let mut expansions = 0usize;

        for _ in 0..max_new_tokens {
            if active.is_empty() || expansions >= cfg.max_expansions {
                break;
            }
            active.sort_by(treecoder_score_desc);
            active.truncate(cfg.width);
            let mut next_active = Vec::with_capacity(cfg.width * cfg.branch);

            let current_active = std::mem::take(&mut active);
            for node in current_active {
                if expansions >= cfg.max_expansions {
                    next_active.push(node);
                    break;
                }
                let candidates = self.treecoder_candidate_log_probs(&node.state, &cfg)?;
                for (next_id, logprob) in candidates {
                    if self.is_stop_id(next_id) {
                        if node.generated.len() >= cfg.min_complete_tokens
                            && node.constraints.can_complete()
                        {
                            let mut finished = node.clone();
                            finished.logprob += logprob;
                            finished.score = treecoder_node_score(
                                finished.logprob,
                                finished.generated.len(),
                                &finished.constraints,
                                &cfg,
                                true,
                            );
                            complete.push(finished);
                        }
                        continue;
                    }

                    let raw_token = self
                        .vocab
                        .id_to_token
                        .get(next_id as usize)
                        .map(|token| token.as_str())
                        .unwrap_or("<unk>");
                    let decoded_token = self.vocab.decode_ids_lossy(&[next_id]);
                    let Some(constraints) =
                        node.constraints.advance(raw_token, &decoded_token, &cfg)
                    else {
                        continue;
                    };

                    let mut child_state = node.state.clone();
                    self.decoder
                        .step_generation(&self.device, &mut child_state, next_id)?;
                    let mut generated = node.generated.clone();
                    generated.push(next_id);
                    let child_logprob = node.logprob + logprob;
                    let child_score = treecoder_node_score(
                        child_logprob,
                        generated.len(),
                        &constraints,
                        &cfg,
                        false,
                    );
                    next_active.push(TreeCoderNode {
                        state: child_state,
                        generated,
                        constraints,
                        logprob: child_logprob,
                        score: child_score,
                    });
                    expansions += 1;
                    if expansions >= cfg.max_expansions {
                        break;
                    }
                }
            }

            if next_active.is_empty() {
                break;
            }
            next_active.sort_by(treecoder_score_desc);
            next_active.truncate(cfg.width);
            active = next_active;
        }

        for mut node in active {
            node.score = treecoder_node_score(
                node.logprob,
                node.generated.len(),
                &node.constraints,
                &cfg,
                true,
            );
            complete.push(node);
        }
        complete.sort_by(treecoder_score_desc);
        let best = complete
            .first()
            .map(|node| node.generated.as_slice())
            .unwrap_or(&[]);
        Ok(clean_candle_decoder_output(
            &self.vocab.decode_ids_lossy(best),
        ))
    }

    fn treecoder_candidate_log_probs(
        &self,
        state: &DecoderGenerationState,
        cfg: &TreeCoderConfig,
    ) -> Result<Vec<(u32, f32)>> {
        let mut logits = self.decoder.last_token_logits(state)?;
        self.apply_repeat_penalty(&mut logits, state);
        self.apply_token_masks(&mut logits);
        let scale = if self.temperature > 0.0 {
            1.0 / self.temperature.max(1e-5)
        } else {
            1.0
        };
        let mut indexed = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .map(|(idx, logit)| (idx, logit * scale))
            .collect::<Vec<_>>();
        if indexed.is_empty() {
            return Ok(Vec::new());
        }
        let max_logit = indexed
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let logsumexp = max_logit
            + indexed
                .iter()
                .map(|(_, logit)| (*logit - max_logit).exp())
                .sum::<f32>()
                .max(1e-20)
                .ln();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(cfg.branch);
        Ok(indexed
            .into_iter()
            .map(|(idx, logit)| (idx as u32, logit - logsumexp))
            .collect())
    }

    fn sample_next_id(&self, state: &DecoderGenerationState) -> Result<u32> {
        let mut logits = self.decoder.last_token_logits(state)?;
        self.apply_repeat_penalty(&mut logits, state);
        self.apply_token_masks(&mut logits);
        if self.temperature <= 0.0 {
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0));
        }
        let distribution = self.sample_distribution(&logits);
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        let mut cum = 0.0f32;
        let mut chosen = distribution
            .first()
            .map(|(idx, _)| *idx as u32)
            .unwrap_or(0);
        for &(idx, prob) in &distribution {
            cum += prob;
            if r <= cum {
                chosen = idx as u32;
                break;
            }
        }
        Ok(chosen)
    }

    fn apply_repeat_penalty(&self, logits: &mut [f32], state: &DecoderGenerationState) {
        if self.repeat_penalty <= 1.0 {
            return;
        }
        let len = state.token_ids.len();
        let start = len.saturating_sub(self.repeat_last_n.max(1));
        let mut seen = std::collections::HashSet::new();
        for &token_id in &state.token_ids[start..] {
            let idx = token_id as usize;
            if idx >= logits.len() || !seen.insert(idx) {
                continue;
            }
            let logit = &mut logits[idx];
            if *logit >= 0.0 {
                *logit /= self.repeat_penalty;
            } else {
                *logit *= self.repeat_penalty;
            }
        }
    }

    fn apply_token_masks(&self, logits: &mut [f32]) {
        for &bad_id in &[self.vocab.pad_id, self.vocab.unk_id] {
            if let Some(logit) = logits.get_mut(bad_id as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn sample_candidate_indices(&self, logits: &[f32]) -> Vec<usize> {
        let mut indexed = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .collect::<Vec<_>>();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if self.top_k > 0 && indexed.len() > self.top_k {
            indexed.truncate(self.top_k);
        }
        if indexed.is_empty() {
            return vec![0];
        }
        indexed.into_iter().map(|(idx, _)| idx).collect()
    }

    fn sample_distribution(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let candidates = self.sample_candidate_indices(logits);
        self.softmax_over_candidates(logits, &candidates)
            .unwrap_or_else(|_| vec![(0, 1.0)])
    }

    fn softmax_over_candidates(
        &self,
        logits: &[f32],
        candidates: &[usize],
    ) -> Result<Vec<(usize, f32)>> {
        let scale = 1.0f32 / self.temperature.max(1e-5);
        let max_logit = candidates
            .iter()
            .map(|&idx| logits[idx] * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut scored = candidates
            .iter()
            .map(|&idx| (idx, ((logits[idx] * scale) - max_logit).exp()))
            .collect::<Vec<_>>();
        let mut total = scored.iter().map(|(_, prob)| *prob).sum::<f32>().max(1e-8);
        for (_, prob) in &mut scored {
            *prob /= total;
        }
        if self.top_p < 1.0 {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut kept = Vec::new();
            let mut cumulative = 0.0f32;
            for (idx, prob) in scored {
                cumulative += prob;
                kept.push((idx, prob));
                if cumulative >= self.top_p.max(1e-3) {
                    break;
                }
            }
            total = kept.iter().map(|(_, prob)| *prob).sum::<f32>().max(1e-8);
            let kept_map = kept
                .into_iter()
                .map(|(idx, prob)| (idx, prob / total))
                .collect::<std::collections::HashMap<_, _>>();
            let mut result = Vec::with_capacity(candidates.len().min(kept_map.len()));
            for &idx in candidates {
                if let Some(prob) = kept_map.get(&idx).copied() {
                    result.push((idx, prob));
                }
            }
            return Ok(result);
        }
        Ok(candidates
            .iter()
            .map(|idx| {
                scored
                    .iter()
                    .find(|(cand_idx, _)| cand_idx == idx)
                    .map(|(_, prob)| (*idx, *prob))
                    .unwrap_or((*idx, 0.0))
            })
            .collect())
    }

    fn is_stop_id(&self, token_id: u32) -> bool {
        if token_id == self.vocab.pad_id {
            return true;
        }
        self.vocab
            .id_to_token
            .get(token_id as usize)
            .is_some_and(|token| DECODER_STOP_TOKENS.contains(&token.as_str()))
    }
}

impl LocalDecoderRuntime for CandleCrossAttnDecoder {
    fn is_available(&self) -> bool {
        true
    }

    fn generate(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        let num_context_slots = conditioning.len() / self.planner_dim;
        if num_context_slots == 0 || conditioning.len() != num_context_slots * self.planner_dim {
            anyhow::bail!(
                "conditioning length {} must equal num_context_slots * planner_dim ({})",
                conditioning.len(),
                self.planner_dim
            );
        }
        let dtype_ref = Tensor::zeros((1,), self.runtime_dtype, &self.device)?;
        let context_slots = crate::util::from_vec_like(
            conditioning.to_vec(),
            (1, num_context_slots, self.planner_dim),
            &dtype_ref,
        )?;
        let context_slots = apply_conditioning_budget(context_slots)?;
        let world_latent = self
            .adapter
            .forward_with_action(&context_slots, decoder_action_id(action))?;
        let mut prompt_ids =
            encode_text_with_vocab_mode(prompt, &self.vocab, self.tokenization_mode);
        prepare_prompt_ids(&mut prompt_ids, self.max_prompt_tokens, self.vocab.pad_id);
        let treecoder_cfg = TreeCoderConfig::from_env(self.kind, action, max_new_tokens);
        if treecoder_cfg.enabled && max_new_tokens > 0 {
            return self.generate_treecoder(
                &prompt_ids,
                &world_latent,
                max_new_tokens,
                treecoder_cfg,
            );
        }
        let mut state = self
            .decoder
            .begin_generation(&self.device, &prompt_ids, &world_latent)?;
        let mut generated = Vec::new();
        for _ in 0..max_new_tokens {
            let next_id = self.sample_next_id(&state)?;
            if self.is_stop_id(next_id) {
                break;
            }
            generated.push(next_id);
            self.decoder
                .step_generation(&self.device, &mut state, next_id)?;
        }
        let text = self.vocab.decode_ids_lossy(&generated);
        Ok(clean_candle_decoder_output(&text))
    }

    /// Stream token-by-token so the client sees progress (no single buffered reply).
    fn generate_stream(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let num_context_slots = conditioning.len() / self.planner_dim;
        if num_context_slots == 0 || conditioning.len() != num_context_slots * self.planner_dim {
            anyhow::bail!(
                "conditioning length {} must equal num_context_slots * planner_dim ({})",
                conditioning.len(),
                self.planner_dim
            );
        }
        let dtype_ref = Tensor::zeros((1,), self.runtime_dtype, &self.device)?;
        let context_slots = crate::util::from_vec_like(
            conditioning.to_vec(),
            (1, num_context_slots, self.planner_dim),
            &dtype_ref,
        )?;
        let context_slots = apply_conditioning_budget(context_slots)?;
        let world_latent = self
            .adapter
            .forward_with_action(&context_slots, decoder_action_id(action))?;
        let mut prompt_ids =
            encode_text_with_vocab_mode(prompt, &self.vocab, self.tokenization_mode);
        prepare_prompt_ids(&mut prompt_ids, self.max_prompt_tokens, self.vocab.pad_id);
        let treecoder_cfg = TreeCoderConfig::from_env(self.kind, action, max_new_tokens);
        if treecoder_cfg.enabled && max_new_tokens > 0 {
            let text =
                self.generate_treecoder(&prompt_ids, &world_latent, max_new_tokens, treecoder_cfg)?;
            if !text.is_empty() {
                on_chunk(&text);
            }
            return Ok(());
        }
        let mut state = self
            .decoder
            .begin_generation(&self.device, &prompt_ids, &world_latent)?;
        let mut pending_bytes = Vec::new();
        for _ in 0..max_new_tokens {
            let next_id = self.sample_next_id(&state)?;
            if self.is_stop_id(next_id) {
                break;
            }
            stream_decoded_token(&self.vocab, next_id, &mut pending_bytes, on_chunk);
            self.decoder
                .step_generation(&self.device, &mut state, next_id)?;
        }
        flush_stream_bytes(&mut pending_bytes, on_chunk);
        Ok(())
    }
}

fn prepare_prompt_ids(prompt_ids: &mut Vec<u32>, max_prompt_tokens: Option<usize>, seed_id: u32) {
    if let Some(limit) = max_prompt_tokens {
        let limit = limit.max(1);
        if prompt_ids.len() > limit {
            *prompt_ids = prompt_ids[prompt_ids.len() - limit..].to_vec();
        }
    }
    if prompt_ids.is_empty() {
        prompt_ids.push(seed_id);
    }
}

fn parse_decoder_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<byte:")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then_some(hex)
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
}

fn flush_stream_bytes(pending_bytes: &mut Vec<u8>, on_chunk: &mut dyn FnMut(&str)) {
    if pending_bytes.is_empty() {
        return;
    }
    let text = String::from_utf8_lossy(pending_bytes).to_string();
    pending_bytes.clear();
    if !text.is_empty() {
        on_chunk(&text);
    }
}

fn stream_decoded_token(
    vocab: &Vocab,
    token_id: u32,
    pending_bytes: &mut Vec<u8>,
    on_chunk: &mut dyn FnMut(&str),
) {
    let raw_token = vocab
        .id_to_token
        .get(token_id as usize)
        .map(|token| token.as_str())
        .unwrap_or("<unk>");
    if let Some(byte) = parse_decoder_byte_token(raw_token) {
        pending_bytes.push(byte);
        match std::str::from_utf8(pending_bytes) {
            Ok(text) => {
                if !text.is_empty() {
                    on_chunk(text);
                }
                pending_bytes.clear();
            }
            Err(err) if err.error_len().is_some() => flush_stream_bytes(pending_bytes, on_chunk),
            Err(_) => {}
        }
        return;
    }

    flush_stream_bytes(pending_bytes, on_chunk);
    let text = vocab.decode_ids_lossy(&[token_id]);
    if !text.is_empty() {
        on_chunk(&text);
    }
}

fn treecoder_node_score(
    logprob: f32,
    generated_len: usize,
    constraints: &CodeTreeConstraintState,
    cfg: &TreeCoderConfig,
    final_node: bool,
) -> f32 {
    let len = generated_len.max(1) as f32;
    let model_score = logprob / len.powf(cfg.length_penalty);
    let constraint_score = if final_node {
        constraints.final_score()
    } else {
        constraints.partial_score()
    };
    model_score + cfg.constraint_weight * constraint_score
}

fn treecoder_score_desc(a: &TreeCoderNode, b: &TreeCoderNode) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn treecoder_forbidden_token(raw_token: &str, decoded_token: &str, cfg: &TreeCoderConfig) -> bool {
    if CODE_CONTROL_TOKENS
        .iter()
        .any(|token| raw_token == *token || decoded_token == *token)
    {
        return true;
    }
    if raw_token.contains("```") || decoded_token.contains("```") {
        return true;
    }
    let raw_trimmed = raw_token.trim();
    let decoded_trimmed = decoded_token.trim();
    if matches!(
        raw_trimmed,
        "Here"
            | "here"
            | "Explanation"
            | "explanation"
            | "assistant"
            | "Assistant"
            | "user"
            | "User"
            | "Compiler"
            | "compiler"
            | "feedback"
            | "Feedback"
    ) || matches!(
        decoded_trimmed,
        "Here"
            | "here"
            | "Explanation"
            | "explanation"
            | "assistant"
            | "Assistant"
            | "user"
            | "User"
            | "Compiler"
            | "compiler"
            | "feedback"
            | "Feedback"
    ) {
        return true;
    }
    cfg.no_comments
        && (decoded_token.contains("//")
            || decoded_token.contains("/*")
            || decoded_token.contains("*/"))
}

fn runtime_env_bool_any(names: &[&str], default: bool) -> bool {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok())
        .map(|value| {
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(default)
}

fn runtime_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn runtime_env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn decoder_action_id(action: &str) -> u32 {
    match action.trim().to_ascii_lowercase().as_str() {
        "code" => 1,
        "done" => 2,
        "fetch_docs" | "fetch-docs" | "docs" => 3,
        _ => 0,
    }
}

fn apply_conditioning_budget(context_slots: Tensor) -> Result<Tensor> {
    let budget = std::env::var("TOFY_DECODER_CONDITION_BUDGET")
        .or_else(|_| std::env::var("JEPA_DECODER_CONDITION_BUDGET"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let Some(budget) = budget else {
        return Ok(context_slots);
    };
    let (_, slots, _) = context_slots.dims3()?;
    if budget == 0 {
        return context_slots.affine(0.0, 0.0).map_err(Into::into);
    }
    if budget >= slots {
        return Ok(context_slots);
    }
    context_slots
        .narrow(1, slots - budget, budget)
        .map_err(Into::into)
}

/// Strip prompt echo and UI junk from Candle decoder output (e.g. "assistant", "/", ">", repeated prompt).
/// Public so world.rs can clean the accumulated segment when building assistant_content from streamed chunks.
pub fn clean_candle_decoder_output(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    for marker in DECODER_STOP_TOKENS
        .iter()
        .chain(CODE_CONTROL_TOKENS.iter())
        .copied()
    {
        if let Some(pos) = s.find(marker) {
            s = s[..pos].to_string();
        }
    }
    // Drop content after a new turn (model echoing User: or Assistant:).
    for sep in ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"] {
        if let Some(pos) = s.find(sep) {
            s = s[..pos].to_string();
        }
    }
    // Strip leading role/UI tokens (repeated).
    const PREFIXES: &[&str] = &[
        "Assistant:",
        "Assistant :",
        "assistant",
        "Assistant",
        "/",
        ">",
    ];
    loop {
        let t = s.trim_start();
        let mut changed = false;
        for p in PREFIXES {
            if let Some(stripped) = t.strip_prefix(p) {
                s = stripped.trim_start().to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        prepare_prompt_ids, stream_decoded_token, treecoder_node_score, CodeTreeConstraintState,
        TreeCoderConfig,
    };
    use crate::model::vocab::Vocab;

    fn cfg() -> TreeCoderConfig {
        TreeCoderConfig {
            enabled: true,
            width: 3,
            branch: 6,
            max_expansions: 128,
            min_complete_tokens: 1,
            length_penalty: 0.7,
            constraint_weight: 0.25,
            no_comments: false,
        }
    }

    #[test]
    fn treecoder_constraint_rejects_markdown_and_negative_delimiters() {
        let cfg = cfg();
        let state = CodeTreeConstraintState::default();

        assert!(state.advance("```", "```", &cfg).is_none());
        assert!(state.advance("}", "}", &cfg).is_none());
    }

    #[test]
    fn treecoder_constraint_tracks_balanced_code_completion() {
        let cfg = cfg();
        let state = CodeTreeConstraintState::default()
            .advance("func", "func", &cfg)
            .expect("func accepted")
            .advance(" ", " ", &cfg)
            .expect("space accepted")
            .advance("Add", "Add", &cfg)
            .expect("identifier accepted")
            .advance("(", "(", &cfg)
            .expect("open paren accepted")
            .advance(")", ")", &cfg)
            .expect("close paren accepted")
            .advance(" ", " ", &cfg)
            .expect("space accepted")
            .advance("{", "{", &cfg)
            .expect("open brace accepted")
            .advance("}", "}", &cfg)
            .expect("close brace accepted");

        assert!(state.can_complete());
        assert!(
            treecoder_node_score(-4.0, 8, &state, &cfg, true)
                > treecoder_node_score(-4.0, 8, &CodeTreeConstraintState::default(), &cfg, true)
        );
    }

    #[test]
    fn empty_prompt_is_seeded_after_prompt_truncation() {
        let mut prompt_ids = Vec::new();
        prepare_prompt_ids(&mut prompt_ids, None, 7);
        assert_eq!(prompt_ids, vec![7]);

        let mut prompt_ids = vec![1, 2, 3];
        prepare_prompt_ids(&mut prompt_ids, Some(2), 7);
        assert_eq!(prompt_ids, vec![2, 3]);

        let mut prompt_ids = vec![1, 2, 3];
        prepare_prompt_ids(&mut prompt_ids, Some(0), 7);
        assert_eq!(prompt_ids, vec![3]);
    }

    #[test]
    fn stream_decoding_buffers_byte_fallback_tokens() {
        let mut vocab = Vocab::new();
        vocab.ensure_byte_tokens();
        let first = *vocab.token_to_id.get("<byte:C3>").expect("byte token");
        let second = *vocab.token_to_id.get("<byte:A9>").expect("byte token");
        let expected = String::from_utf8(vec![0xC3, 0xA9]).expect("valid utf8");
        let mut pending = Vec::new();
        let mut chunks = Vec::new();

        stream_decoded_token(&vocab, first, &mut pending, &mut |chunk| {
            chunks.push(chunk.to_string());
        });
        assert!(chunks.is_empty());

        stream_decoded_token(&vocab, second, &mut pending, &mut |chunk| {
            chunks.push(chunk.to_string());
        });
        assert_eq!(chunks, vec![expected]);
        assert!(pending.is_empty());
    }
}
