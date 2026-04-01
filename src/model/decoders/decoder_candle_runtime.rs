//! Decoder runtime: Candle cross-attention decoders for code and text.
//! - Code: JEPA_USE_CANDLE_DECODER=1, JEPA_CANDLE_DECODER=<path>
//! - Text: JEPA_USE_TEXT_DECODER=1, JEPA_TEXT_DECODER=<path>

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::{Path, PathBuf};

use crate::data::tokenize_for_inference;
use crate::model::vocab::{load_vocab_from_file, Vocab};
use super::{CodeDecoder, DecoderAdapter, DecoderKind, LocalDecoderRuntime};

/// Decoder architecture; must match training constants in world.rs. Sized for ~8GB VRAM (~90M params).
const DECODER_DIM: usize = 768;
const DECODER_LAYERS: usize = 8;
const DECODER_HEADS: usize = 8;
const DECODER_FF_DIM: usize = 3072;

/// Decoder backend that runs the Candle CodeDecoder with cross-attention to the world latent sequence.
/// Expects conditioning to be the **flattened** latent sequence (length = num_latent_tokens * world_dim).
pub struct CandleCrossAttnDecoder {
    adapter: DecoderAdapter,
    decoder: CodeDecoder,
    vocab: Vocab,
    device: Device,
    planner_dim: usize,
    temperature: f32,
}

impl CandleCrossAttnDecoder {
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

    /// Load CodeDecoder from checkpoint. Config: dim=768, 8 layers, 8 heads, ff=3072.
    pub fn new(
        checkpoint_path: PathBuf,
        vocab_path: PathBuf,
        planner_dim: usize,
        world_dim: usize,
        planner_slots: usize,
        temperature: f32,
        kind: DecoderKind,
    ) -> Result<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let mut varmap = VarMap::new();
        varmap
            .load(&checkpoint_path)
            .with_context(|| format!("load code decoder from {:?}", checkpoint_path))?;
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let vocab = load_vocab_from_file(&vocab_path)
            .with_context(|| format!("load decoder vocab from {:?}", vocab_path))?;
        let vocab_size = vocab.id_to_token.len();
        let adapter = DecoderAdapter::new(
            vb.pp("decoder_adapter"),
            planner_dim,
            world_dim,
            DecoderAdapter::output_slots_for(kind, planner_slots),
        )?;
        let decoder = CodeDecoder::new(
            vb.pp("decoder"),
            vocab_size,
            DECODER_DIM,
            world_dim,
            DECODER_LAYERS,
            DECODER_HEADS,
            DECODER_FF_DIM,
            kind,
        )?;
        Ok(Self {
            adapter,
            decoder,
            vocab,
            device,
            planner_dim,
            temperature,
        })
    }

    /// Code decoder: JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER=<path>. Optional JEPA_DECODER_TEMP.
    pub fn try_new_from_env_code(planner_dim: usize, planner_slots: usize) -> Result<Self> {
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
                    .unwrap_or(0.7);
                Self::new(
                    checkpoint_path,
                    vocab_path,
                    planner_dim,
                    planner_dim,
                    planner_slots,
                    temp,
                    DecoderKind::CodeSpecialist,
                )
            }
            _ => Err(anyhow::anyhow!("JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER=<path> required")),
        }
    }

    /// Text decoder: JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER=<path>. Optional JEPA_TEXT_DECODER_TEMP (default 0.7).
    /// Same architecture as code decoder; trained on dialog data (e.g. ultrachat_pairs) for general text reply.
    pub fn try_new_from_env_text(planner_dim: usize, planner_slots: usize) -> Result<Self> {
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
                    planner_slots,
                    temp,
                    DecoderKind::TextGeneralist,
                )
            }
            _ => Err(anyhow::anyhow!("JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER=<path> required")),
        }
    }
}

impl LocalDecoderRuntime for CandleCrossAttnDecoder {
    fn is_available(&self) -> bool {
        true
    }

    fn generate(
        &self,
        prompt: &str,
        _action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        let num_planner_slots = conditioning.len() / self.planner_dim;
        if num_planner_slots == 0 || conditioning.len() != num_planner_slots * self.planner_dim {
            anyhow::bail!(
                "conditioning length {} must equal num_planner_slots * planner_dim ({})",
                conditioning.len(),
                self.planner_dim
            );
        }
        let planner_slots = Tensor::from_vec(
            conditioning.to_vec(),
            (1, num_planner_slots, self.planner_dim),
            &self.device,
        )?;
        let world_latent = self.adapter.forward(&planner_slots)?;
        let tokens = tokenize_for_inference(prompt);
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let prompt_ids = self.vocab.encode(&tokens);
        let generated = self.decoder.generate(
            &self.device,
            &prompt_ids,
            &world_latent,
            max_new_tokens,
            self.temperature,
            Some(self.vocab.pad_id),
        )?;
        let text: String = generated
            .iter()
            .map(|&id| {
                self.vocab
                    .id_to_token
                    .get(id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("<unk>")
            })
            .collect();
        Ok(clean_candle_decoder_output(&text))
    }

    /// Stream token-by-token so the client sees progress (no single buffered reply).
    fn generate_stream(
        &self,
        prompt: &str,
        _action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let num_planner_slots = conditioning.len() / self.planner_dim;
        if num_planner_slots == 0 || conditioning.len() != num_planner_slots * self.planner_dim {
            anyhow::bail!(
                "conditioning length {} must equal num_planner_slots * planner_dim ({})",
                conditioning.len(),
                self.planner_dim
            );
        }
        let planner_slots = Tensor::from_vec(
            conditioning.to_vec(),
            (1, num_planner_slots, self.planner_dim),
            &self.device,
        )?;
        let world_latent = self.adapter.forward(&planner_slots)?;
        let tokens = tokenize_for_inference(prompt);
        if tokens.is_empty() {
            return Ok(());
        }
        let prompt_ids = self.vocab.encode(&tokens);
        let mut ids = prompt_ids.clone();
        for _ in 0..max_new_tokens {
            let next_id = self.decoder.step(&self.device, &ids, &world_latent, self.temperature)?;
            if next_id == self.vocab.pad_id {
                break;
            }
            ids.push(next_id);
            let token_str = self
                .vocab
                .id_to_token
                .get(next_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            on_chunk(token_str);
        }
        Ok(())
    }
}

/// Strip prompt echo and UI junk from Candle decoder output (e.g. "assistant", "/", ">", repeated prompt).
/// Public so world.rs can clean the accumulated segment when building assistant_content from streamed chunks.
pub fn clean_candle_decoder_output(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    // Drop content after a new turn (model echoing User: or Assistant:).
    for sep in ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"] {
        if let Some(pos) = s.find(sep) {
            s = s[..pos].to_string();
        }
    }
    // Strip leading role/UI tokens (repeated).
    const PREFIXES: &[&str] = &["Assistant:", "Assistant :", "assistant", "Assistant", "/", ">"];
    loop {
        let t = s.trim_start();
        let mut changed = false;
        for p in PREFIXES {
            if t.starts_with(p) {
                s = t[p.len()..].trim_start().to_string();
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
