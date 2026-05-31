//! Decoder runtime: Candle cross-attention decoders for code and text.
//! - Code: JEPA_USE_CANDLE_DECODER=1, JEPA_CANDLE_DECODER=<path>
//! - Text: JEPA_USE_TEXT_DECODER=1, JEPA_TEXT_DECODER=<path>

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::RngExt;
use std::fs;
use std::path::{Path, PathBuf};

use super::{
    CodeDecoder, DecoderArchitecture, DecoderConditioningAdapter, DecoderKind, LocalDecoderRuntime,
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

/// Decoder backend that runs the Candle CodeDecoder with cross-attention to the world latent sequence.
/// Expects conditioning to be the **flattened** latent sequence (length = num_latent_tokens * world_dim).
pub struct CandleCrossAttnDecoder {
    adapter: DecoderConditioningAdapter,
    decoder: CodeDecoder,
    vocab: Vocab,
    device: Device,
    runtime_dtype: DType,
    planner_dim: usize,
    temperature: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    top_k: usize,
    top_p: f32,
    tokenization_mode: TokenizationMode,
    max_prompt_tokens: Option<usize>,
}

impl CandleCrossAttnDecoder {
    pub fn metadata_path(checkpoint_path: &Path) -> PathBuf {
        checkpoint_path.with_extension("meta.txt")
    }

    pub fn write_metadata(
        checkpoint_path: &Path,
        vocab: &Vocab,
        kind: DecoderKind,
        planner_dim: usize,
        context_slots: usize,
        architecture: DecoderArchitecture,
    ) -> Result<()> {
        let metadata = format!(
            "kind={}\nvocab_signature={}\nvocab_size={}\nplanner_dim={}\ncontext_slots={}\nconditioner=action_aware_local_plan_v1\ndecoder_arch=rope_rmsnorm_swiglu_tied_v2\ndecoder_dim={}\ndecoder_layers={}\ndecoder_heads={}\ndecoder_ff_dim={}\n",
            kind.as_str(),
            vocab_signature(vocab),
            vocab.id_to_token.len(),
            planner_dim,
            context_slots,
            architecture.dim,
            architecture.num_layers,
            architecture.num_heads,
            architecture.ff_dim
        );
        fs::write(Self::metadata_path(checkpoint_path), metadata)?;
        Ok(())
    }

    fn load_metadata_architecture(
        checkpoint_path: &Path,
        vocab: &Vocab,
        kind: DecoderKind,
        planner_dim: usize,
        context_slots: usize,
    ) -> Result<DecoderArchitecture> {
        let metadata_path = Self::metadata_path(checkpoint_path);
        if !metadata_path.exists() {
            anyhow::bail!(
                "decoder metadata not found for {:?}; runtime requires checkpoint metadata with decoder_dim/layers/heads/ff_dim",
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
        let parse_required = |key: &str| -> Result<usize> {
            parsed
                .get(key)
                .ok_or_else(|| {
                    anyhow::anyhow!("decoder metadata {:?} is missing {}", metadata_path, key)
                })?
                .parse()
                .with_context(|| format!("parse {} from {:?}", key, metadata_path))
        };
        DecoderArchitecture::new(
            parse_required("decoder_dim")?,
            parse_required("decoder_layers")?,
            parse_required("decoder_heads")?,
            parse_required("decoder_ff_dim")?,
        )
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
        let architecture = Self::load_metadata_architecture(
            &checkpoint_path,
            &vocab,
            kind,
            planner_dim,
            context_slots,
        )?;
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
        let adapter = DecoderConditioningAdapter::new(
            VarBuilder::from_varmap(&varmap, checkpoint_dtype, &device)
                .pp("decoder_conditioning_adapter"),
            planner_dim,
            world_dim,
            DecoderConditioningAdapter::output_slots_for(kind, context_slots),
        )?;
        let decoder = CodeDecoder::new(
            VarBuilder::from_varmap(&varmap, checkpoint_dtype, &device).pp("decoder"),
            vocab_size,
            architecture.dim,
            world_dim,
            architecture.num_layers,
            architecture.num_heads,
            architecture.ff_dim,
            kind,
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

    fn sample_next_id(
        &self,
        state: &crate::model::decoders::decoder_cross::DecoderGenerationState,
    ) -> Result<u32> {
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

    fn apply_repeat_penalty(
        &self,
        logits: &mut [f32],
        state: &crate::model::decoders::decoder_cross::DecoderGenerationState,
    ) {
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
        if let Some(limit) = self.max_prompt_tokens {
            if prompt_ids.len() > limit {
                prompt_ids = prompt_ids[prompt_ids.len() - limit..].to_vec();
            }
        }
        if prompt_ids.is_empty() {
            return Ok(String::new());
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
        if let Some(limit) = self.max_prompt_tokens {
            if prompt_ids.len() > limit {
                prompt_ids = prompt_ids[prompt_ids.len() - limit..].to_vec();
            }
        }
        if prompt_ids.is_empty() {
            return Ok(());
        }
        let mut state = self
            .decoder
            .begin_generation(&self.device, &prompt_ids, &world_latent)?;
        for _ in 0..max_new_tokens {
            let next_id = self.sample_next_id(&state)?;
            if self.is_stop_id(next_id) {
                break;
            }
            let token_str = self
                .vocab
                .id_to_token
                .get(next_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            on_chunk(token_str);
            self.decoder
                .step_generation(&self.device, &mut state, next_id)?;
        }
        Ok(())
    }
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
