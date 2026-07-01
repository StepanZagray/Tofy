use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, UNIX_EPOCH};

use crate::cli::resolve_data_path;
use crate::config::{
    DecoderTrainConfig, HighWorldTrainConfig, OrchestratorTrainConfig, ServeConfig,
    WorldEvalConfig, WorldTrainConfig,
};
use crate::data::{
    build_vocab_from_raw_world_file_with_mode_action_filter, count_raw_world_rows,
    count_raw_world_rows_split, count_raw_world_rows_split_with_mode_action_filter,
    encode_text_with_vocab_mode, encode_world_examples, encode_world_examples_with_mode,
    make_decoder_batch_from_slice_with_prompt_dropout, CachedDecoderStream, CachedWorldStream,
    RawWorldExample, RawWorldStream, TokenizationMode, WorldExample, ACTION_CODE, ACTION_DONE,
    ACTION_FETCH_DOCS, ACTION_TEXT_REPLY, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::encoders::EncoderFeatures;
use crate::model::vocab::vocab_signature;
use crate::model::{
    agentic_decoder_requested, clean_agentic_final, flatten_latent_slots, load_vocab_from_file,
    mean_cosine_similarity, parse_tool_call, prediction_loss, save_vocab_to_file,
    sigreg_epps_pulley, tensor_rms, ActionSequenceEncoder, ActionStateTransition, BashToolRegistry,
    CandleCrossAttnDecoder, CodeDecoder, ContextCompressor, DecoderArchitecture,
    DecoderConditioningAdapter, DecoderKind, LlamaCppDecoder, LocalDecoderRuntime,
    MacroActionStateTransition, NextActionClassifier, OnlineEncoder, RlmDecoderRuntime,
    StubLocalDecoder, Vocab,
};
use crate::tasks::prepare::{go_version_string, GoCompileFeedback};
use crate::tasks::world_support::{
    action_cross_entropy, compute_action_metrics, decoder_conditioning_gains,
    decoder_loss_masks_from_examples, decoder_prediction_metrics, decoder_reward_proxy,
    decoder_selection_score, encoded_examples_oov_rate, evaluate_decoder_batch,
    evaluate_decoder_cached_batch, evaluate_world_encoded_batch, forbidden_output_probability_loss,
    hard_mismatched_conditioning_latent, masked_cross_entropy, masked_label_smoothed_cross_entropy,
    masked_weighted_cross_entropy, perplexity_from_nll, raw_examples_oov_rate,
    shuffled_conditioning_latent, slot_delta_slots, world_selection_score, world_sigreg_loss,
    ActionMetrics, DecoderBatchMetrics, WorldBatchMetrics,
};
use crate::util;

const HELDOUT_SPLIT_MODULUS: usize = 20;
const HELDOUT_SPLIT_REMAINDER: usize = 0;
const TOKEN_CACHE_MANIFEST_VERSION: u32 = 8;
const DECODER_VOCAB_MANIFEST_VERSION: u32 = 1;
const CONDITIONED_DECODER_CACHE_VERSION: u32 = 2;
const CONDITIONED_DECODER_CACHE_MAGIC: &[u8] = b"TOFY_CONDITIONED_DECODER_CACHE_V2\n";
const DUAL_TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_DUAL_TOKEN_CACHE_V2\n";

type WorldConfig = WorldTrainConfig;

struct WorldLogSnapshot {
    loss_val: f32,
    trans_val: f32,
    sigreg_val: f32,
    act_val: f32,
    inv_val: f32,
    post_state_val: f32,
    rollout_val: f32,
    action_metrics: ActionMetrics,
    inverse_metrics: ActionMetrics,
    trans_cos: f32,
    state_slot_rms: f32,
    pred_slot_rms: f32,
}

struct HighWorldLogSnapshot {
    loss_val: f32,
    trans_val: f32,
    sigreg_val: f32,
    cosine: f32,
    pred_rms: f32,
    target_rms: f32,
}

struct OrchestratorLogSnapshot {
    action_loss_val: f32,
    metrics: ActionMetrics,
}

struct DecoderLogSnapshot {
    metrics: DecoderBatchMetrics,
    hard_mismatch_loss_val: f32,
    conditioning_loss_val: f32,
    format_loss_val: f32,
    mtp_loss_val: f32,
}

fn default_world_encoder_path(model_path: &Path) -> PathBuf {
    let raw = model_path.to_string_lossy();
    if let Some(prefix) = raw.strip_suffix(".safetensors") {
        PathBuf::from(format!("{prefix}.encoder.safetensors"))
    } else {
        PathBuf::from(format!("{raw}.encoder.safetensors"))
    }
}

fn default_high_world_path(world_model_path: &Path) -> PathBuf {
    if let Some(stage_dir) = world_model_path.parent() {
        if stage_dir.file_name().and_then(|name| name.to_str()) == Some("world") {
            if let Some(run_root) = stage_dir.parent() {
                return run_root.join("high_world").join("model.safetensors");
            }
        }
    }
    let raw = world_model_path.to_string_lossy();
    if let Some(prefix) = raw.strip_suffix(".safetensors") {
        PathBuf::from(format!("{prefix}.high_world.safetensors"))
    } else {
        PathBuf::from(format!("{raw}.high_world.safetensors"))
    }
}

#[derive(Debug, Deserialize)]
struct CacheManifestSource {
    path: String,
    len: u64,
    content_hash: String,
}

#[derive(Debug, Deserialize)]
struct CacheTokenManifest {
    #[serde(default)]
    version: u32,
    kind: String,
    source: CacheManifestSource,
    tokenizer: String,
    max_seq: usize,
    action_filter: Option<u32>,
    vocab_signature: String,
    token_cache_path: String,
    #[serde(default)]
    rows: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DecoderVocabManifest {
    version: u32,
    kind: String,
    tokenizer: String,
    max_vocab: usize,
    action_filter: u32,
    vocab_signature: String,
}

fn token_cache_path(kind: &str) -> Option<PathBuf> {
    if !env_bool("TOFY_USE_TOKEN_CACHE", true) {
        return None;
    }
    let cache_dir = std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string());
    let path = PathBuf::from(cache_dir).join(format!("{kind}.tokens.bin"));
    path.exists().then_some(path)
}

fn cache_dir() -> PathBuf {
    PathBuf::from(std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string()))
}

fn load_cache_token_manifest(path: &Path) -> Result<CacheTokenManifest> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn source_fingerprint_matches(path: &Path, source: &CacheManifestSource) -> Result<bool> {
    if source.path != path.to_string_lossy() {
        return Ok(false);
    }
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if source.len != metadata.len() {
        return Ok(false);
    }
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        for byte in &buf[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(source.content_hash == format!("{hash:016x}"))
}

fn compatible_world_cache_path(
    data_path: &Path,
    max_seq: usize,
    encoder_vocab_sig: &str,
) -> Result<Option<PathBuf>> {
    let cache_path = match token_cache_path("world") {
        Some(path) => path,
        None => return Ok(None),
    };
    let manifest_path = cache_dir().join("world_tokens.manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = load_cache_token_manifest(&manifest_path)
        .with_context(|| format!("load world cache manifest from {:?}", manifest_path))?;
    if manifest.version != TOKEN_CACHE_MANIFEST_VERSION
        || manifest.kind != "world"
        || manifest.tokenizer != TokenizationMode::Default.as_str()
        || manifest.max_seq < max_seq
        || manifest.action_filter.is_some()
        || manifest.vocab_signature != encoder_vocab_sig
        || manifest.token_cache_path != cache_path.to_string_lossy()
        || !source_fingerprint_matches(data_path, &manifest.source)?
    {
        return Ok(None);
    }
    Ok(Some(cache_path))
}

struct DecoderTokenCacheInfo {
    path: PathBuf,
    rows: usize,
}

fn split_match_count(rows: usize, modulus: usize, remainder: usize) -> usize {
    if rows == 0 || modulus == 0 || remainder >= modulus || rows <= remainder {
        0
    } else {
        1 + (rows - 1 - remainder) / modulus
    }
}

fn compatible_decoder_dual_cache_info(
    data_path: &Path,
    max_seq: usize,
    encoder_vocab_sig: &str,
    decoder_vocab_sig: &str,
) -> Result<Option<DecoderTokenCacheInfo>> {
    let cache_path = match token_cache_path("code_decoder_dual") {
        Some(path) => path,
        None => return Ok(None),
    };
    let manifest_path = cache_dir().join("code_decoder_dual_tokens.manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = load_cache_token_manifest(&manifest_path)
        .with_context(|| format!("load decoder cache manifest from {:?}", manifest_path))?;
    let expected_sig = format!("{encoder_vocab_sig}+{decoder_vocab_sig}");
    if manifest.version != TOKEN_CACHE_MANIFEST_VERSION
        || manifest.kind != "code_decoder_dual"
        || manifest.tokenizer != TokenizationMode::CodeAware.as_str()
        || manifest.max_seq < max_seq
        || manifest.action_filter != Some(ACTION_CODE)
        || !source_fingerprint_matches(data_path, &manifest.source)?
        || manifest.vocab_signature != expected_sig
        || manifest.token_cache_path != cache_path.to_string_lossy()
    {
        return Ok(None);
    }
    Ok(Some(DecoderTokenCacheInfo {
        path: cache_path,
        rows: manifest.rows,
    }))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct CacheFileFingerprint {
    path: String,
    len: u64,
    modified_unix_secs: u64,
    content_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ConditionedDecoderContextSignature {
    max_seq: usize,
    dim: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    rollout_steps: usize,
    decoder_action_label: u32,
    hybrid_context: bool,
    hybrid_exact_tail: usize,
    hybrid_block_size: usize,
    hybrid_retrieval_slots: usize,
    exact_old_tokens: usize,
    train_dtype: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ConditionedDecoderCacheManifest {
    version: u32,
    kind: String,
    data_path: String,
    cache_path: String,
    source_token_cache: CacheFileFingerprint,
    source_token_manifest: Option<CacheFileFingerprint>,
    encoder_model: CacheFileFingerprint,
    world_model: CacheFileFingerprint,
    encoder_vocab_signature: String,
    decoder_vocab_signature: String,
    context: ConditionedDecoderContextSignature,
    rows: usize,
}

#[derive(Clone)]
struct ConditionedDecoderCacheRecord {
    encoder_state_tokens: Vec<u32>,
    encoder_next_tokens: Vec<u32>,
    decoder_state_tokens: Vec<u32>,
    decoder_next_tokens: Vec<u32>,
    action_label: u32,
    next_context_slots: Vec<f32>,
}

#[derive(Clone)]
struct ConditionedDecoderExample {
    decoder: WorldExample,
    next_context_slots: Vec<f32>,
}

struct ConditionedDecoderStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<ConditionedDecoderExample>,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    row_index: usize,
}

impl ConditionedDecoderStream {
    fn with_split(
        path: &Path,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        let mut stream = Self {
            path: path.to_path_buf(),
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
        };
        stream.read_magic()?;
        Ok(stream)
    }

    fn read_magic(&mut self) -> Result<()> {
        let mut magic = vec![0u8; CONDITIONED_DECODER_CACHE_MAGIC.len()];
        self.reader.read_exact(&mut magic)?;
        if magic != CONDITIONED_DECODER_CACHE_MAGIC {
            bail!("invalid conditioned decoder cache magic in {:?}", self.path);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = BufReader::new(File::open(&self.path)?);
        self.row_index = 0;
        self.read_magic()
    }

    fn read_next_example(&mut self) -> Result<ConditionedDecoderExample> {
        loop {
            let Some(record) = read_conditioned_decoder_cache_record(&mut self.reader)? else {
                self.reset()?;
                continue;
            };
            let row_idx = self.row_index;
            self.row_index += 1;
            if let Some(modulus) = self.split_modulus {
                let is_match = row_idx % modulus == self.split_remainder;
                let keep = if self.exclude_split_matches {
                    !is_match
                } else {
                    is_match
                };
                if !keep {
                    continue;
                }
            }
            if record.encoder_state_tokens.is_empty()
                || record.encoder_next_tokens.is_empty()
                || record.decoder_state_tokens.is_empty()
                || record.decoder_next_tokens.is_empty()
                || record.next_context_slots.is_empty()
            {
                continue;
            }
            return Ok(ConditionedDecoderExample {
                decoder: WorldExample {
                    state_tokens: record.decoder_state_tokens,
                    next_tokens: record.decoder_next_tokens,
                    action_label: record.action_label,
                },
                next_context_slots: record.next_context_slots,
            });
        }
    }

    fn refill_shuffle_buffer(&mut self) -> Result<()> {
        while self.shuffle_buffer.len() < self.shuffle_buffer_size {
            let example = self.read_next_example()?;
            self.shuffle_buffer.push(example);
        }
        Ok(())
    }

    fn next_example(&mut self) -> Result<ConditionedDecoderExample> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_example();
        }
        self.refill_shuffle_buffer()?;
        let idx = rand::rng().random_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    fn next_batch(&mut self, batch_size: usize) -> Result<Vec<ConditionedDecoderExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
    }
}

struct DecoderMicroBatch {
    encoder_batch: Vec<WorldExample>,
    decoder_batch: Vec<WorldExample>,
    oov_rate: f32,
    next_context_slots: Option<Tensor>,
}

fn conditioned_decoder_cache_path() -> PathBuf {
    std::env::var("TOFY_DECODER_CONDITIONED_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cache_dir().join("code_decoder_go_conditioned.tokens.bin"))
}

fn conditioned_decoder_manifest_path(cache_path: &Path) -> PathBuf {
    let file_name = cache_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("code_decoder_go_conditioned.tokens.bin");
    cache_path.with_file_name(format!("{file_name}.manifest.json"))
}

fn cache_tmp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!("{file_name}.tmp.{}", std::process::id()))
}

fn cache_file_fingerprint(path: &Path, hash_contents: bool) -> Result<CacheFileFingerprint> {
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let modified_unix_secs = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let content_hash = if hash_contents {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut hash = 0xcbf29ce484222325u64;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buf)?;
            if read == 0 {
                break;
            }
            for byte in &buf[..read] {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        Some(format!("{hash:016x}"))
    } else {
        None
    };
    Ok(CacheFileFingerprint {
        path: path.to_string_lossy().into_owned(),
        len: metadata.len(),
        modified_unix_secs,
        content_hash,
    })
}

pub(crate) fn default_context_hybrid_exact_tail(
    max_seq: usize,
    recent_full_segments: usize,
) -> usize {
    let max_seq = max_seq.max(1);
    let min_chunk = 16usize.min(max_seq);
    let target_chunk = max_seq.div_ceil(16).max(min_chunk);
    let chunk = target_chunk.min(max_seq).max(1);
    let chunk_slots = max_seq.div_ceil(chunk).max(1);
    max_seq
        .saturating_add(chunk_slots)
        .saturating_add(6)
        .saturating_mul(recent_full_segments.max(1))
}

#[allow(clippy::too_many_arguments)]
fn conditioned_decoder_context_signature(
    max_seq: usize,
    dim: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    rollout_steps: usize,
    decoder_action_label: u32,
    train_dtype: DType,
) -> ConditionedDecoderContextSignature {
    let hybrid_retrieval_slots = env_usize("TOFY_CONTEXT_RETRIEVAL_SLOTS", 8);
    let exact_old_tokens = std::env::var("TOFY_CONTEXT_EXACT_OLD_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| hybrid_retrieval_slots.saturating_mul(2).min(16));
    ConditionedDecoderContextSignature {
        max_seq,
        dim,
        bridge_dim,
        num_latent_tokens,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        rollout_steps,
        decoder_action_label,
        hybrid_context: env_bool("TOFY_CONTEXT_HYBRID_MEMORY", true),
        hybrid_exact_tail: env_usize(
            "TOFY_CONTEXT_HYBRID_EXACT_TAIL",
            default_context_hybrid_exact_tail(max_seq, recent_full_segments),
        ),
        hybrid_block_size: env_usize("TOFY_CONTEXT_HYBRID_BLOCK_SIZE", 16),
        hybrid_retrieval_slots,
        exact_old_tokens,
        train_dtype: format!("{train_dtype:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn conditioned_decoder_expected_manifest(
    data_path: &Path,
    cache_path: &Path,
    source_cache_path: &Path,
    encoder_model_path: &Path,
    world_model_path: &Path,
    encoder_vocab_sig: &str,
    decoder_vocab_sig: &str,
    context: ConditionedDecoderContextSignature,
) -> Result<ConditionedDecoderCacheManifest> {
    let source_manifest_path =
        source_cache_path.with_file_name("code_decoder_dual_tokens.manifest.json");
    let source_token_manifest = source_manifest_path
        .exists()
        .then(|| cache_file_fingerprint(&source_manifest_path, true))
        .transpose()?;
    Ok(ConditionedDecoderCacheManifest {
        version: CONDITIONED_DECODER_CACHE_VERSION,
        kind: "code_decoder_go_conditioned".to_string(),
        data_path: data_path.to_string_lossy().into_owned(),
        cache_path: cache_path.to_string_lossy().into_owned(),
        source_token_cache: cache_file_fingerprint(source_cache_path, false)?,
        source_token_manifest,
        encoder_model: cache_file_fingerprint(encoder_model_path, true)?,
        world_model: cache_file_fingerprint(world_model_path, true)?,
        encoder_vocab_signature: encoder_vocab_sig.to_string(),
        decoder_vocab_signature: decoder_vocab_sig.to_string(),
        context,
        rows: 0,
    })
}

fn conditioned_decoder_manifest_matches(
    actual: &ConditionedDecoderCacheManifest,
    expected: &ConditionedDecoderCacheManifest,
) -> bool {
    actual.version == expected.version
        && actual.kind == expected.kind
        && actual.data_path == expected.data_path
        && actual.cache_path == expected.cache_path
        && actual.source_token_cache == expected.source_token_cache
        && actual.source_token_manifest == expected.source_token_manifest
        && actual.encoder_model == expected.encoder_model
        && actual.world_model == expected.world_model
        && actual.encoder_vocab_signature == expected.encoder_vocab_signature
        && actual.decoder_vocab_signature == expected.decoder_vocab_signature
        && actual.context == expected.context
}

fn read_cache_u32_le<R: Read>(reader: &mut R) -> Result<Option<u32>> {
    let mut buf = [0u8; 4];
    let mut read = 0usize;
    while read < buf.len() {
        let n = reader.read(&mut buf[read..])?;
        if n == 0 {
            if read == 0 {
                return Ok(None);
            }
            bail!("truncated decoder cache record");
        }
        read += n;
    }
    Ok(Some(u32::from_le_bytes(buf)))
}

fn read_cache_ids<R: Read>(reader: &mut R) -> Result<Option<Vec<u32>>> {
    let Some(len) = read_cache_u32_le(reader)? else {
        return Ok(None);
    };
    let len = len as usize;
    let byte_len = len
        .checked_mul(std::mem::size_of::<u32>())
        .context("decoder cache id sequence too large")?;
    let mut ids = vec![0u32; len];
    let bytes = unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr().cast::<u8>(), byte_len) };
    reader
        .read_exact(bytes)
        .context("truncated decoder cache id sequence")?;
    if cfg!(target_endian = "big") {
        for id in &mut ids {
            *id = u32::from_le(*id);
        }
    }
    Ok(Some(ids))
}

fn read_cache_f32s<R: Read>(reader: &mut R) -> Result<Vec<f32>> {
    let len = read_cache_u32_le(reader)?
        .context("truncated conditioned decoder cache slot length")? as usize;
    let byte_len = len
        .checked_mul(std::mem::size_of::<f32>())
        .context("conditioned decoder slot vector too large")?;
    let mut values = vec![0f32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_len) };
    reader
        .read_exact(bytes)
        .context("truncated conditioned decoder slot vector")?;
    if cfg!(target_endian = "big") {
        for value in &mut values {
            *value = f32::from_le_bytes(value.to_ne_bytes());
        }
    }
    Ok(values)
}

fn read_source_dual_cache_record<R: Read>(
    reader: &mut R,
) -> Result<Option<ConditionedDecoderCacheRecord>> {
    let Some(encoder_state_tokens) = read_cache_ids(reader)? else {
        return Ok(None);
    };
    let encoder_next_tokens =
        read_cache_ids(reader)?.context("truncated dual token cache encoder next sequence")?;
    let decoder_state_tokens =
        read_cache_ids(reader)?.context("truncated dual token cache decoder state sequence")?;
    let decoder_next_tokens =
        read_cache_ids(reader)?.context("truncated dual token cache decoder next sequence")?;
    let action_label = read_cache_u32_le(reader)?.context("truncated dual token cache action")?;
    Ok(Some(ConditionedDecoderCacheRecord {
        encoder_state_tokens,
        encoder_next_tokens,
        decoder_state_tokens,
        decoder_next_tokens,
        action_label,
        next_context_slots: Vec::new(),
    }))
}

fn read_conditioned_decoder_cache_record<R: Read>(
    reader: &mut R,
) -> Result<Option<ConditionedDecoderCacheRecord>> {
    let Some(mut record) = read_source_dual_cache_record(reader)? else {
        return Ok(None);
    };
    record.next_context_slots = read_cache_f32s(reader)?;
    Ok(Some(record))
}

fn write_cache_ids<W: Write>(writer: &mut W, ids: &[u32]) -> Result<()> {
    writer.write_all(&(ids.len() as u32).to_le_bytes())?;
    for id in ids {
        writer.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}

fn write_conditioned_decoder_cache_record<W: Write>(
    writer: &mut W,
    record: &ConditionedDecoderCacheRecord,
) -> Result<()> {
    write_cache_ids(writer, &record.encoder_state_tokens)?;
    write_cache_ids(writer, &record.encoder_next_tokens)?;
    write_cache_ids(writer, &record.decoder_state_tokens)?;
    write_cache_ids(writer, &record.decoder_next_tokens)?;
    writer.write_all(&record.action_label.to_le_bytes())?;
    writer.write_all(&(record.next_context_slots.len() as u32).to_le_bytes())?;
    for value in &record.next_context_slots {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn conditioned_slots_tensor(
    batch: &[ConditionedDecoderExample],
    num_latent_tokens: usize,
    bridge_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let slot_len = num_latent_tokens
        .checked_mul(bridge_dim)
        .context("conditioned decoder slot shape overflow")?;
    let mut flat = Vec::with_capacity(batch.len() * slot_len);
    for row in batch {
        if row.next_context_slots.len() != slot_len {
            bail!(
                "conditioned decoder slot length mismatch: got {}, expected {}",
                row.next_context_slots.len(),
                slot_len
            );
        }
        flat.extend_from_slice(&row.next_context_slots);
    }
    Tensor::from_vec(flat, (batch.len(), num_latent_tokens, bridge_dim), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn fill_conditioned_slots_for_records(
    records: &mut [ConditionedDecoderCacheRecord],
    context_cache: &mut DecoderContextCache,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    decoder_action_label: u32,
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    rollout_steps: usize,
    device: &Device,
) -> Result<()> {
    let mut valid_positions = Vec::new();
    let mut encoder_batch = Vec::new();
    for (idx, record) in records.iter().enumerate() {
        if record.encoder_state_tokens.is_empty()
            || record.encoder_next_tokens.is_empty()
            || record.decoder_state_tokens.is_empty()
            || record.decoder_next_tokens.is_empty()
        {
            continue;
        }
        valid_positions.push(idx);
        encoder_batch.push(WorldExample {
            state_tokens: record.encoder_state_tokens.clone(),
            next_tokens: record.encoder_next_tokens.clone(),
            action_label: record.action_label,
        });
    }
    if encoder_batch.is_empty() {
        return Ok(());
    }
    let next_slots = decoder_next_context_slots(
        context_cache,
        encoder,
        context_compressor,
        transition,
        &encoder_batch,
        decoder_action_label,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        rollout_steps,
        device,
    )?;
    for (slot_idx, record_idx) in valid_positions.iter().copied().enumerate() {
        let row_slots = next_slots.narrow(0, slot_idx, 1)?;
        records[record_idx].next_context_slots = util::vec1_f32(&row_slots.flatten_all()?)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_prepare_conditioned_decoder_cache(
    data_path: &Path,
    source_cache_path: Option<&PathBuf>,
    allow_build: bool,
    require_existing: bool,
    encoder_model_path: &Path,
    world_model_path: &Path,
    encoder_vocab_sig: &str,
    decoder_vocab_sig: &str,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    decoder_action_label: u32,
    pad_id: u32,
    max_seq: usize,
    dim: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    rollout_steps: usize,
    train_dtype: DType,
    device: &Device,
) -> Result<Option<PathBuf>> {
    if !allow_build && !require_existing {
        return Ok(None);
    }
    let Some(source_cache_path) = source_cache_path else {
        if allow_build || require_existing {
            bail!(
                "conditioned decoder cache requires a compatible decoder dual token cache for {:?}",
                data_path
            );
        }
        return Ok(None);
    };
    let cache_path = conditioned_decoder_cache_path();
    let manifest_path = conditioned_decoder_manifest_path(&cache_path);
    let context = conditioned_decoder_context_signature(
        max_seq,
        dim,
        bridge_dim,
        num_latent_tokens,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        rollout_steps,
        decoder_action_label,
        train_dtype,
    );
    let expected_manifest = conditioned_decoder_expected_manifest(
        data_path,
        &cache_path,
        source_cache_path,
        encoder_model_path,
        world_model_path,
        encoder_vocab_sig,
        decoder_vocab_sig,
        context,
    )?;
    if cache_path.exists() && manifest_path.exists() {
        if let Ok(text) = fs::read_to_string(&manifest_path) {
            if let Ok(actual) = serde_json::from_str::<ConditionedDecoderCacheManifest>(&text) {
                if conditioned_decoder_manifest_matches(&actual, &expected_manifest) {
                    println!(
                        "Token cache: using conditioned decoder slots {:?} (rows={})",
                        cache_path, actual.rows
                    );
                    return Ok(Some(cache_path));
                }
            }
        }
    }
    if require_existing {
        bail!(
            "conditioned decoder cache is missing or incompatible: {:?}",
            cache_path
        );
    }
    if !allow_build {
        return Ok(None);
    }

    println!(
        "Token cache: precomputing frozen decoder conditioning slots from {:?} to {:?}",
        source_cache_path, cache_path
    );
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = cache_tmp_path_for(&cache_path);
    let mut source_reader = BufReader::new(File::open(source_cache_path)?);
    let mut source_magic = vec![0u8; DUAL_TOKEN_CACHE_MAGIC.len()];
    source_reader.read_exact(&mut source_magic)?;
    if source_magic != DUAL_TOKEN_CACHE_MAGIC {
        bail!("invalid dual token cache magic in {:?}", source_cache_path);
    }
    let mut writer = BufWriter::new(File::create(&tmp_path)?);
    writer.write_all(CONDITIONED_DECODER_CACHE_MAGIC)?;
    let chunk_size = env_usize("TOFY_DECODER_SLOT_PRECOMPUTE_BATCH", 256);
    let mut context_cache = DecoderContextCache::from_env();
    let mut rows = 0usize;
    let mut next_progress = 10_000usize;
    loop {
        let mut records = Vec::with_capacity(chunk_size);
        let mut reached_eof = false;
        for _ in 0..chunk_size {
            match read_source_dual_cache_record(&mut source_reader)? {
                Some(record) => {
                    if record.action_label == decoder_action_label {
                        records.push(record);
                    }
                }
                None => {
                    reached_eof = true;
                    break;
                }
            }
        }
        if records.is_empty() {
            if reached_eof {
                break;
            }
            continue;
        }
        fill_conditioned_slots_for_records(
            &mut records,
            &mut context_cache,
            encoder,
            context_compressor,
            transition,
            decoder_action_label,
            pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            rollout_steps,
            device,
        )?;
        for record in &records {
            write_conditioned_decoder_cache_record(&mut writer, record)?;
        }
        rows += records.len();
        if rows >= next_progress {
            println!("Token cache: conditioned decoder slots progress rows={rows}");
            next_progress += 10_000;
        }
    }
    if rows == 0 {
        bail!(
            "conditioned decoder cache found no action={} rows in {:?}",
            decoder_action_label,
            source_cache_path
        );
    }
    writer.flush()?;
    fs::rename(&tmp_path, &cache_path)?;
    let mut manifest = expected_manifest;
    manifest.rows = rows;
    let manifest_tmp = cache_tmp_path_for(&manifest_path);
    fs::write(&manifest_tmp, serde_json::to_string_pretty(&manifest)?)?;
    fs::rename(manifest_tmp, &manifest_path)?;
    println!("Token cache: conditioned decoder slots saved rows={rows}");
    Ok(Some(cache_path))
}

fn world_batch_size_for_step(step: usize, config: &WorldConfig) -> usize {
    if config.batch_warmup_steps > 0
        && step <= config.batch_warmup_steps
        && config.batch_warmup_value != config.batch_size
    {
        config.batch_warmup_value.max(1)
    } else {
        config.batch_size.max(1)
    }
}

fn world_grad_accum_for_step(step: usize, config: &WorldConfig) -> usize {
    if config.grad_accum_warmup_steps > 0
        && step <= config.grad_accum_warmup_steps
        && config.grad_accum_warmup_value < config.grad_accum_steps
    {
        config.grad_accum_warmup_value.max(1)
    } else {
        config.grad_accum_steps.max(1)
    }
}

fn high_world_batch_size_for_step(step: usize, config: &HighWorldTrainConfig) -> usize {
    if config.batch_warmup_steps > 0
        && step <= config.batch_warmup_steps
        && config.batch_warmup_value != config.batch_size
    {
        config.batch_warmup_value.max(1)
    } else {
        config.batch_size.max(1)
    }
}

fn high_world_grad_accum_for_step(step: usize, config: &HighWorldTrainConfig) -> usize {
    if config.grad_accum_warmup_steps > 0
        && step <= config.grad_accum_warmup_steps
        && config.grad_accum_warmup_value < config.grad_accum_steps
    {
        config.grad_accum_warmup_value.max(1)
    } else {
        config.grad_accum_steps.max(1)
    }
}

fn decoder_batch_size_for_step(step: usize, config: &DecoderTrainConfig) -> usize {
    if config.batch_warmup_steps > 0
        && step <= config.batch_warmup_steps
        && config.batch_warmup_value != config.batch_size
    {
        config.batch_warmup_value.max(1)
    } else {
        config.batch_size.max(1)
    }
}

fn decoder_grad_accum_for_step(step: usize, config: &DecoderTrainConfig) -> usize {
    if config.grad_accum_warmup_steps > 0
        && step <= config.grad_accum_warmup_steps
        && config.grad_accum_warmup_value != config.grad_accum_steps
    {
        config.grad_accum_warmup_value.max(1)
    } else {
        config.grad_accum_steps.max(1)
    }
}

pub fn try_run_train(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--train-world" && args[1] != "train-world") {
        return Ok(false);
    }
    let data_path = resolve_data_path(&args[4])?.path;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[2] = data_path.to_string_lossy().to_string();
    let cfg = WorldConfig::from_args_after(&args_for_cfg)?;
    run_world_training(cfg)?;
    Ok(true)
}

pub fn try_run_train_high_world(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--train-high-world" && args[1] != "train-high-world") {
        return Ok(false);
    }
    let data_path = resolve_data_path(&args[5])?;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[3] = data_path.path.to_string_lossy().to_string();
    let cfg = HighWorldTrainConfig::from_args_after(&args_for_cfg)?;
    run_high_world_training(cfg)?;
    Ok(true)
}

pub fn try_run_train_orchestrator(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--train-orchestrator" && args[1] != "train-orchestrator") {
        return Ok(false);
    }
    let data_path = resolve_data_path(&args[5])?.path;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[3] = data_path.to_string_lossy().to_string();
    let cfg = OrchestratorTrainConfig::from_args_after(&args_for_cfg)?;
    run_orchestrator_training(cfg)?;
    Ok(true)
}

pub fn try_run_train_decoder(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--train-decoder" && args[1] != "train-decoder") {
        return Ok(false);
    }
    let encoder_model_path = PathBuf::from(&args[2]);
    let encoder_vocab_path = PathBuf::from(&args[3]);
    let world_path = PathBuf::from(&args[4]);
    let mut args_for_cfg = vec![
        encoder_model_path.to_string_lossy().to_string(),
        encoder_vocab_path.to_string_lossy().to_string(),
        world_path.to_string_lossy().to_string(),
        args[5].clone(),
    ];
    args_for_cfg.extend(args.iter().skip(6).cloned());
    let mut cfg = DecoderTrainConfig::from_args_after(&args_for_cfg)?;
    if cfg.mtp_loss_weight > 0.0 {
        anyhow::bail!(
            "--mtp-loss-weight is unsupported: this decoder has no dedicated future-token heads, \
             so reusing next-token logits for MTP trains conflicting targets"
        );
    }
    cfg.data_path = resolve_data_path(&args[5])?.path;
    run_decoder_training(cfg)?;
    Ok(true)
}

pub fn try_run_eval(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--eval-world" && args[1] != "eval-world") {
        return Ok(false);
    }
    let cfg = WorldEvalConfig::from_args_after(&args[2..])?;
    run_eval_world(cfg)?;
    Ok(true)
}

pub fn try_run_serve(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--serve" && args[1] != "serve") {
        return Ok(false);
    }
    let cfg = ServeConfig::from_args_after(&args[2..])?;
    if cfg.debug {
        std::env::set_var("JEPA_DEBUG", "1");
    }
    let rt = tokio::runtime::Runtime::new().context("create tokio runtime")?;
    rt.block_on(crate::tasks::serve::run(
        &cfg.bind,
        cfg.encoder_model_path,
        cfg.encoder_vocab_path,
        cfg.world_model_path,
        cfg.high_world_model_path,
        cfg.dim,
        cfg.max_seq,
        cfg.num_layers,
        cfg.num_heads,
        cfg.bridge_dim,
        cfg.num_latent_tokens,
        cfg.debug,
    ))?;
    Ok(true)
}

fn run_world_training(config: WorldConfig) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            tracing::warn!("CUDA not available: {}", e);
            Device::Cpu
        }
    };

    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let row_count = count_raw_world_rows(&config.data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &config.data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let mut world_stream = RawWorldStream::with_split(
        &config.data_path,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
        true,
    )?;
    let mut val_stream = if val_row_count > 0 {
        Some(RawWorldStream::with_split(
            &config.data_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            false,
        )?)
    } else {
        None
    };
    let world_cache_path =
        compatible_world_cache_path(&config.data_path, config.max_seq, &encoder_vocab_sig)?;
    let mut cached_world_stream = if let Some(cache_path) = world_cache_path.as_ref() {
        println!("Token cache: using world training cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        println!("Token cache: no world cache found; using raw tokenization stream");
        None
    };
    let mut cached_val_stream = if val_row_count > 0 {
        if let Some(cache_path) = world_cache_path.as_ref() {
            Some(CachedWorldStream::with_split(
                cache_path,
                DEFAULT_STREAM_SHUFFLE_BUFFER,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let vocab_size = encoder_vocab.id_to_token.len();

    let mut encoder_varmap = VarMap::new();
    let encoder = if config.train_encoder {
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
        let encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        encoder
    } else {
        let mut frozen_encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&frozen_encoder_varmap, train_dtype, &device);
        let loaded_encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut frozen_encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut frozen_encoder_varmap, train_dtype)?;
        let frozen_encoder_tensors = util::frozen_tensors_from_varmap(&frozen_encoder_varmap)?;
        drop(loaded_encoder);
        let encoder_vb = VarBuilder::from_tensors(frozen_encoder_tensors, train_dtype, &device);
        OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?
    };

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
    let context_compressor = ContextCompressor::new(
        world_vb.pp("context_compressor"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition =
        ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
    let action_classifier_head =
        NextActionClassifier::new(world_vb.pp("next_action_classifier"), config.bridge_dim)?;
    let inverse_action_classifier =
        NextActionClassifier::new(world_vb.pp("inverse_action_classifier"), config.bridge_dim)?;

    const ORCH_N: usize = crate::model::action_classifier_head::NUM_ACTIONS;
    let transition_ff = (config.bridge_dim * 4).max(320);
    let transition_params = ORCH_N * config.bridge_dim
        + 6 * (8 * config.bridge_dim * config.bridge_dim
            + 2 * config.bridge_dim * transition_ff
            + transition_ff
            + 13 * config.bridge_dim)
        + 2 * config.bridge_dim
        + 2 * (config.bridge_dim * config.bridge_dim + config.bridge_dim);
    let learned_memory_params =
        2 * config.dim + 2 * (config.dim + 1) + 2 * (config.dim * config.dim + config.dim);
    let planner_params = config.num_latent_tokens * config.dim
        + 2 * (config.dim * config.dim + config.dim * 4 * config.dim)
        + config.dim * config.bridge_dim
        + config.bridge_dim
        + learned_memory_params;
    let action_classifier_hidden = (config.bridge_dim * 2).max(256);
    let action_classifier_params = config.bridge_dim * action_classifier_hidden
        + action_classifier_hidden
        + action_classifier_hidden * ORCH_N
        + ORCH_N;
    let inverse_params = action_classifier_params;
    let total_params =
        transition_params + planner_params + action_classifier_params + inverse_params;
    let _ = fs::create_dir_all("local_models");
    let model_path = config.output_path.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "local_models/model_world_{}.safetensors",
            util::format_params(total_params)
        ))
    });
    let encoder_model_path = config
        .encoder_output_path
        .clone()
        .unwrap_or_else(|| default_world_encoder_path(&model_path));
    let resume_stage = util::resume_stage_name("world");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "train.safetensors");
    let encoder_train_checkpoint_path =
        util::checkpoint_sidecar_path(&encoder_model_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        util::load_varmap_checked(&mut world_varmap, &train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!("Resuming world weights from {:?}", train_checkpoint_path);
    } else if config.resume && model_path.exists() {
        util::load_varmap_checked(&mut world_varmap, &model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!(
            "Resuming world weights from best export {:?} without optimizer state",
            model_path
        );
    }
    if config.train_encoder {
        if config.resume && encoder_train_checkpoint_path.exists() {
            util::load_varmap_checked(&mut encoder_varmap, &encoder_train_checkpoint_path)?;
            util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
            println!(
                "Resuming LeWM encoder weights from {:?}",
                encoder_train_checkpoint_path
            );
        } else if config.resume && encoder_model_path.exists() {
            util::load_varmap_checked(&mut encoder_varmap, &encoder_model_path)?;
            util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
            println!(
                "Resuming LeWM encoder weights from best export {:?}",
                encoder_model_path
            );
        }
    }

    let mut named_train_vars = util::named_train_vars(&world_varmap)?;
    if config.train_encoder {
        named_train_vars.extend(util::named_train_vars(&encoder_varmap)?);
        named_train_vars.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume {
        if let Some(state) = util::load_resume_state(&resume_state_path, &resume_stage)? {
            resume_state = state;
        }
        if optimizer_checkpoint_path.exists() {
            opt.load_state(&optimizer_checkpoint_path)?;
            if resume_state.step == 0 {
                resume_state.step = opt.step_t();
            }
            println!(
                "Resuming world optimizer from {:?} at step {}",
                optimizer_checkpoint_path, resume_state.step
            );
        }
    }

    println!("Training (latent-only dialog transition model for text + code)");
    println!("Encoder checkpoint: {:?}", config.encoder_model_path);
    println!(
        "LeWM encoder export: {:?} ({})",
        encoder_model_path,
        if config.train_encoder {
            "trainable"
        } else {
            "frozen"
        }
    );
    println!("Encoder vocab: {:?}", config.encoder_vocab_path);
    println!(
        "Rows: train {} | val {} | encoder vocab {} | max_seq {} | context_slots {} | lambda {:.3}",
        train_row_count,
        val_row_count,
        vocab_size,
        config.max_seq,
        config.num_latent_tokens,
        config.lambda
    );
    println!(
        "World objective: LeWM post-state MSE + lambda * SIGReg | action auxiliary weight {:.2}",
        config.action_loss_weight
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "Estimated parameters: ~{} [context_compressor {} + transition {} + action_classifier {} + inverse {}]",
        util::format_params(total_params),
        util::format_params(planner_params),
        util::format_params(transition_params),
        util::format_params(action_classifier_params),
        util::format_params(inverse_params)
    );

    let mut best_loss = resume_state.best_aux_metric;
    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };

    let run_dir = util::create_run_dir("world")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let async_checkpoints = env_bool("TOFY_WORLD_ASYNC_CHECKPOINTS", true);
    let mut checkpoint_writer = if async_checkpoints {
        Some(util::AsyncCheckpointWriter::new())
    } else {
        None
    };
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("resume/start_step", start_step as f32, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar(
        "config/warmup_batch_size",
        config.batch_warmup_value as f32,
        0,
    );
    tb.add_scalar("config/dim", config.dim as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/context_slots", config.num_latent_tokens as f32, 0);
    tb.add_scalar("config/estimated_params", total_params as f32, 0);
    tb.add_scalar(
        "config/train_encoder",
        if config.train_encoder { 1.0 } else { 0.0 },
        0,
    );
    tb.add_scalar(
        "config/train_dtype",
        match train_dtype {
            DType::F16 => 16.0,
            DType::BF16 => 17.0,
            _ => 32.0,
        },
        0,
    );
    tb.add_scalar("config/grad_accum", config.grad_accum_steps as f32, 0);
    tb.add_scalar(
        "config/warmup_grad_accum",
        config.grad_accum_warmup_value as f32,
        0,
    );
    tb.add_scalar(
        "config/warmup_grad_accum_steps",
        config.grad_accum_warmup_steps as f32,
        0,
    );
    tb.add_scalar(
        "config/effective_batch_size",
        (config.batch_size * config.grad_accum_steps.max(1)) as f32,
        0,
    );
    tb.add_scalar(
        "config/warmup_effective_batch_size",
        (config.batch_warmup_value.max(1) * config.grad_accum_warmup_value.max(1)) as f32,
        0,
    );
    let inverse_loss_weight = std::env::var("TOFY_WORLD_INVERSE_LOSS_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0f64)
        .max(0.0);
    tb.add_scalar("config/inverse_loss_weight", inverse_loss_weight as f32, 0);
    if let Some(vram) = vram_tracker.sample() {
        tb.add_scalar("memory/used_mb", vram.used_mb, 0);
        tb.add_scalar("memory/free_mb", vram.free_mb, 0);
        tb.add_scalar("memory/total_mb", vram.total_mb, 0);
        tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, 0);
    }
    println!(
        "World sidecar checkpointing: async={} log_every={}",
        async_checkpoints, config.log_every
    );
    tb.flush();

    const TARGET_CODE_RATE: f32 = 0.35;
    const TARGET_DONE_RATE: f32 = 0.15;
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    tb.add_scalar("config/sigreg_slices", sigreg_slices as f32, 0);
    tb.add_scalar("config/sigreg_points", sigreg_points as f32, 0);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let post_state_loss_weight = world_post_state_loss_weight();
    let rollout_loss_weight = world_rollout_loss_weight();
    let rollout_steps = world_rollout_steps();
    let transition_cosine_weight = std::env::var("TOFY_WORLD_TRANS_COSINE_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.1)
        .max(0.0);
    let clip_norm = env_f64("TOFY_WORLD_CLIP_NORM", 1.0).max(0.0);
    tb.add_scalar(
        "config/post_state_loss_weight",
        post_state_loss_weight as f32,
        0,
    );
    tb.add_scalar("config/rollout_loss_weight", rollout_loss_weight as f32, 0);
    tb.add_scalar("config/rollout_steps", rollout_steps as f32, 0);
    tb.add_scalar("config/clip_norm", clip_norm as f32, 0);
    tb.add_scalar(
        "config/transition_cosine_weight",
        transition_cosine_weight as f32,
        0,
    );
    if start_step >= config.steps {
        println!(
            "World resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = world_batch_size_for_step(step, &config);
        let grad_accum_steps = world_grad_accum_for_step(step, &config);
        if config.batch_warmup_steps > 0
            && step == config.batch_warmup_steps + 1
            && (config.batch_warmup_value != config.batch_size
                || config.grad_accum_warmup_value < config.grad_accum_steps)
        {
            println!(
                "World warmup complete at step {}; switching to batch={} grad_accum={} (effective={})",
                config.batch_warmup_steps,
                batch_size,
                grad_accum_steps,
                batch_size * grad_accum_steps
            );
        }

        for micro_step in 0..grad_accum_steps {
            let batch = if let Some(ref mut cached_stream) = cached_world_stream {
                collect_action_training_batch_cached(
                    cached_stream,
                    batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?
            } else {
                let raw_batch = collect_action_training_batch(
                    &mut world_stream,
                    batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?;
                encode_world_examples(&raw_batch, &encoder_vocab)
            };
            let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
            let (state_slots, next_slots) = context_slots_from_world_pair_batch(
                &encoder,
                &context_compressor,
                &batch,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                recursive_context_compressor,
                !config.train_encoder,
                &device,
            )?;
            let pred_next_slots = transition.forward(&state_slots, &action_labels)?;
            let fixed_next_slots = next_slots.detach();

            let transition_loss = prediction_loss(&pred_next_slots, &fixed_next_slots)?;
            // Direction term: plain MSE permits the right norm with the wrong
            // direction, but downstream decoder conditioning is directional.
            let transition_direction_loss = if transition_cosine_weight > 0.0 {
                let cos = mean_cosine_similarity(
                    &flatten_latent_slots(&pred_next_slots)?,
                    &flatten_latent_slots(&fixed_next_slots)?,
                )?;
                cos.affine(-transition_cosine_weight, transition_cosine_weight)?
            } else {
                transition_loss.affine(0.0, 0.0)?
            };
            let post_state_loss = if post_state_loss_weight > 0.0 {
                transition_loss.clone()
            } else {
                transition_loss.affine(0.0, 0.0)?
            };
            let rollout_loss = rollout_loss_from_batch(
                &transition,
                &state_slots,
                &batch,
                rollout_steps,
                rollout_loss_weight,
            )?
            .unwrap_or(transition_loss.affine(0.0, 0.0)?);
            let state_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&state_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let next_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&next_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let pred_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&pred_next_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let sigreg_loss = world_sigreg_loss(&state_sigreg, &next_sigreg, &pred_sigreg)?;

            let action_logits = action_classifier_head.forward(&state_slots)?;
            let action_loss = action_cross_entropy(&action_logits, &action_labels, &device)?;
            let (inverse_loss, inverse_logits) = if inverse_loss_weight > 0.0 {
                let true_delta_slots = slot_delta_slots(&fixed_next_slots, &state_slots.detach())?;
                let pred_delta_slots = slot_delta_slots(&pred_next_slots, &state_slots)?;
                let inverse_logits_true = inverse_action_classifier.forward(&true_delta_slots)?;
                let inverse_logits_pred = inverse_action_classifier.forward(&pred_delta_slots)?;
                let inverse_true_loss =
                    action_cross_entropy(&inverse_logits_true, &action_labels, &device)?;
                let inverse_pred_loss =
                    action_cross_entropy(&inverse_logits_pred, &action_labels, &device)?;
                (
                    inverse_true_loss
                        .broadcast_add(&inverse_pred_loss)?
                        .affine(0.5, 0.0)?,
                    Some(inverse_logits_pred),
                )
            } else {
                (transition_loss.affine(0.0, 0.0)?, None)
            };
            let loss = transition_loss
                .broadcast_add(&transition_direction_loss)?
                .broadcast_add(&post_state_loss.affine(post_state_loss_weight, 0.0)?)?
                .broadcast_add(&rollout_loss.affine(rollout_loss_weight, 0.0)?)?
                .broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?
                .broadcast_add(&action_loss.affine(config.action_loss_weight, 0.0)?)?
                .broadcast_add(&inverse_loss.affine(inverse_loss_weight, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            let should_capture_log =
                step % config.log_every == 0 && micro_step + 1 == grad_accum_steps;
            if should_capture_log {
                let trans_val = util::scalar_f32(&transition_loss)?;
                let loss_val = util::scalar_f32(&loss)?;
                let sigreg_val = util::scalar_f32(&sigreg_loss)?;
                let act_val = util::scalar_f32(&action_loss)?;
                let inv_val = util::scalar_f32(&inverse_loss)?;
                let post_state_val = util::scalar_f32(&post_state_loss)?;
                let rollout_val = util::scalar_f32(&rollout_loss)?;
                let metric_action_logits = action_logits.detach();
                let action_metrics = compute_action_metrics(&metric_action_logits, &action_labels)?;
                let inverse_metrics = if let Some(inverse_logits) = inverse_logits.as_ref() {
                    let metric_inverse_logits = inverse_logits.detach();
                    compute_action_metrics(&metric_inverse_logits, &action_labels)?
                } else {
                    ActionMetrics::default()
                };
                let metric_pred_next_slots = pred_next_slots.detach();
                let metric_next_slots = next_slots.detach();
                let pred_slots_flat = flatten_latent_slots(&metric_pred_next_slots)?;
                let next_slots_flat = flatten_latent_slots(&metric_next_slots)?;
                let trans_cos =
                    util::scalar_f32(&mean_cosine_similarity(&pred_slots_flat, &next_slots_flat)?)?;
                let state_slot_rms = util::scalar_f32(&tensor_rms(&state_slots.detach())?)?;
                let pred_slot_rms = util::scalar_f32(&tensor_rms(&metric_pred_next_slots)?)?;
                log_snapshot = Some(WorldLogSnapshot {
                    loss_val,
                    trans_val,
                    sigreg_val,
                    act_val,
                    inv_val,
                    post_state_val,
                    rollout_val,
                    action_metrics,
                    inverse_metrics,
                    trans_cos,
                    state_slot_rms,
                    pred_slot_rms,
                });
            }
        }

        let scheduled_lr = util::scheduled_lr(config.lr, step, config.steps);
        opt.set_learning_rate(scheduled_lr);
        let grad_norm = util::clip_accumulated_gradients_device(
            &mut accumulated_grads,
            &train_vars,
            clip_norm,
        )?;
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            if let Some(grad_norm) = grad_norm.as_ref() {
                tb.add_scalar("grad/global_norm", util::scalar_f32(grad_norm)?, step);
            }
            let WorldLogSnapshot {
                loss_val,
                trans_val,
                sigreg_val,
                act_val,
                inv_val,
                post_state_val,
                rollout_val,
                action_metrics,
                inverse_metrics,
                trans_cos,
                state_slot_rms,
                pred_slot_rms,
            } = log_snapshot.context("world grad accumulation produced no log snapshot")?;

            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/trans", trans_val, step);
            tb.add_scalar("schedule/lr", scheduled_lr as f32, step);
            tb.add_scalar("loss/sigreg", sigreg_val, step);
            tb.add_scalar("loss/action", act_val, step);
            tb.add_scalar("loss/inverse_action", inv_val, step);
            tb.add_scalar("loss/post_state", post_state_val, step);
            tb.add_scalar("loss/rollout", rollout_val, step);
            tb.add_scalar("metrics/action_acc", action_metrics.accuracy, step);
            tb.add_scalar(
                "metrics/action_balanced_acc",
                action_metrics.balanced_accuracy,
                step,
            );
            tb.add_scalar("metrics/action_macro_f1", action_metrics.macro_f1, step);
            tb.add_scalar("metrics/inverse_action_acc", inverse_metrics.accuracy, step);
            tb.add_scalar(
                "metrics/inverse_action_balanced_acc",
                inverse_metrics.balanced_accuracy,
                step,
            );
            tb.add_scalar(
                "metrics/inverse_action_macro_f1",
                inverse_metrics.macro_f1,
                step,
            );
            tb.add_scalar("metrics/trans_cosine", trans_cos, step);
            tb.add_scalar("metrics/code_rate", action_metrics.code_rate, step);
            tb.add_scalar(
                "metrics/pred_code_rate",
                action_metrics.pred_code_rate,
                step,
            );
            tb.add_scalar(
                "metrics/code_precision",
                action_metrics.code_precision,
                step,
            );
            tb.add_scalar("metrics/code_recall", action_metrics.code_recall, step);
            tb.add_scalar("metrics/code_f1", action_metrics.code_f1, step);
            tb.add_scalar("metrics/done_rate", action_metrics.done_rate, step);
            tb.add_scalar(
                "metrics/pred_done_rate",
                action_metrics.pred_done_rate,
                step,
            );
            tb.add_scalar(
                "metrics/done_precision",
                action_metrics.done_precision,
                step,
            );
            tb.add_scalar("metrics/done_recall", action_metrics.done_recall, step);
            tb.add_scalar("metrics/done_f1", action_metrics.done_f1, step);
            tb.add_scalar(
                "metrics/fetch_docs_rate",
                action_metrics.fetch_docs_rate,
                step,
            );
            tb.add_scalar(
                "metrics/pred_fetch_docs_rate",
                action_metrics.pred_fetch_docs_rate,
                step,
            );
            tb.add_scalar(
                "metrics/fetch_docs_precision",
                action_metrics.fetch_docs_precision,
                step,
            );
            tb.add_scalar(
                "metrics/fetch_docs_recall",
                action_metrics.fetch_docs_recall,
                step,
            );
            tb.add_scalar("metrics/fetch_docs_f1", action_metrics.fetch_docs_f1, step);
            tb.add_scalar("metrics/state_slot_rms", state_slot_rms, step);
            tb.add_scalar("metrics/pred_slot_rms", pred_slot_rms, step);
            tb.add_scalar("schedule/batch_size", batch_size as f32, step);
            tb.add_scalar("schedule/grad_accum", grad_accum_steps as f32, step);
            tb.add_scalar(
                "schedule/effective_batch_size",
                (batch_size * grad_accum_steps) as f32,
                step,
            );
            let mut memory_note = String::new();
            if let Some(vram) = vram_tracker.sample() {
                tb.add_scalar("memory/used_mb", vram.used_mb, step);
                tb.add_scalar("memory/free_mb", vram.free_mb, step);
                tb.add_scalar("memory/total_mb", vram.total_mb, step);
                tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, step);
                memory_note = format!(
                    " vram {:.0}/{:.0}MB peak {:.0}MB",
                    vram.used_mb, vram.total_mb, vram.peak_used_mb
                );
            }
            let selection_metric;
            if val_stream.is_some() || cached_val_stream.is_some() {
                let val_batch = if let Some(ref mut cached_stream) = cached_val_stream {
                    collect_action_training_batch_cached(
                        cached_stream,
                        batch_size,
                        TARGET_CODE_RATE,
                        TARGET_DONE_RATE,
                    )?
                } else {
                    let val_stream = val_stream
                        .as_mut()
                        .context("world validation stream missing")?;
                    let val_raw_batch = collect_action_training_batch(
                        val_stream,
                        batch_size,
                        TARGET_CODE_RATE,
                        TARGET_DONE_RATE,
                    )?;
                    encode_world_examples(&val_raw_batch, &encoder_vocab)
                };
                let val_metrics = evaluate_world_encoded_batch(
                    &val_batch,
                    &encoder_vocab,
                    &encoder,
                    &context_compressor,
                    &transition,
                    &action_classifier_head,
                    &inverse_action_classifier,
                    config.max_seq,
                    config.lambda,
                    config.action_loss_weight,
                    inverse_loss_weight,
                    &device,
                )?;
                selection_metric = world_selection_score(&val_metrics);
                best_loss = best_loss.min(selection_metric);
                tb.add_scalar("val/total", val_metrics.total_loss, step);
                tb.add_scalar("val/trans", val_metrics.transition_loss, step);
                tb.add_scalar("val/sigreg", val_metrics.sigreg_loss, step);
                tb.add_scalar("val/action", val_metrics.action_loss, step);
                tb.add_scalar("val/inverse_action", val_metrics.inverse_loss, step);
                tb.add_scalar("val/action_acc", val_metrics.action_metrics.accuracy, step);
                tb.add_scalar(
                    "val/action_balanced_acc",
                    val_metrics.action_metrics.balanced_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/action_macro_f1",
                    val_metrics.action_metrics.macro_f1,
                    step,
                );
                tb.add_scalar(
                    "val/inverse_action_acc",
                    val_metrics.inverse_action_metrics.accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/inverse_action_balanced_acc",
                    val_metrics.inverse_action_metrics.balanced_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/inverse_action_macro_f1",
                    val_metrics.inverse_action_metrics.macro_f1,
                    step,
                );
                tb.add_scalar(
                    "val/code_precision",
                    val_metrics.action_metrics.code_precision,
                    step,
                );
                tb.add_scalar(
                    "val/code_recall",
                    val_metrics.action_metrics.code_recall,
                    step,
                );
                tb.add_scalar("val/code_f1", val_metrics.action_metrics.code_f1, step);
                tb.add_scalar("val/code_rate", val_metrics.action_metrics.code_rate, step);
                tb.add_scalar(
                    "val/pred_code_rate",
                    val_metrics.action_metrics.pred_code_rate,
                    step,
                );
                tb.add_scalar(
                    "val/done_precision",
                    val_metrics.action_metrics.done_precision,
                    step,
                );
                tb.add_scalar(
                    "val/done_recall",
                    val_metrics.action_metrics.done_recall,
                    step,
                );
                tb.add_scalar("val/done_f1", val_metrics.action_metrics.done_f1, step);
                tb.add_scalar("val/done_rate", val_metrics.action_metrics.done_rate, step);
                tb.add_scalar(
                    "val/pred_done_rate",
                    val_metrics.action_metrics.pred_done_rate,
                    step,
                );
                tb.add_scalar(
                    "val/fetch_docs_rate",
                    val_metrics.action_metrics.fetch_docs_rate,
                    step,
                );
                tb.add_scalar(
                    "val/pred_fetch_docs_rate",
                    val_metrics.action_metrics.pred_fetch_docs_rate,
                    step,
                );
                tb.add_scalar(
                    "val/fetch_docs_precision",
                    val_metrics.action_metrics.fetch_docs_precision,
                    step,
                );
                tb.add_scalar(
                    "val/fetch_docs_recall",
                    val_metrics.action_metrics.fetch_docs_recall,
                    step,
                );
                tb.add_scalar(
                    "val/fetch_docs_f1",
                    val_metrics.action_metrics.fetch_docs_f1,
                    step,
                );
                tb.add_scalar("val/trans_cosine", val_metrics.transition_cosine, step);
                tb.add_scalar("val/selection_score", selection_metric, step);
            } else {
                let train_metrics = WorldBatchMetrics {
                    total_loss: loss_val,
                    transition_loss: trans_val,
                    sigreg_loss: sigreg_val,
                    action_loss: act_val,
                    inverse_loss: inv_val,
                    action_metrics,
                    inverse_action_metrics: inverse_metrics,
                    transition_cosine: trans_cos,
                };
                selection_metric = world_selection_score(&train_metrics);
                best_loss = best_loss.min(selection_metric);
            }
            tb.flush();

            if selection_metric < best_metric {
                best_metric = selection_metric;
                util::save_varmap_atomic(&world_varmap, &model_path)?;
                if config.train_encoder {
                    util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
                }
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} post {post_state_val:.4} rollout {rollout_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} inverse {inv_val:.4} action_acc {:.3} bal_acc {:.3} macro_f1 {:.3} inv_acc {:.3} inv_bal {:.3} inv_f1 {:.3} code_p {:.3} code_r {:.3} code_f1 {:.3} done_f1 {:.3} trans_cos {trans_cos:.4} code_rate {:.3} pred_code {:.3} done_rate {:.3} pred_done {:.3} sel {selection_metric:.4}{} [saved best]",
                    config.steps,
                    action_metrics.accuracy,
                    action_metrics.balanced_accuracy,
                    action_metrics.macro_f1,
                    inverse_metrics.accuracy,
                    inverse_metrics.balanced_accuracy,
                    inverse_metrics.macro_f1,
                    action_metrics.code_precision,
                    action_metrics.code_recall,
                    action_metrics.code_f1,
                    action_metrics.done_f1,
                    action_metrics.code_rate,
                    action_metrics.pred_code_rate,
                    action_metrics.done_rate,
                    action_metrics.pred_done_rate,
                    memory_note
                );
            } else {
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} post {post_state_val:.4} rollout {rollout_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} inverse {inv_val:.4} action_acc {:.3} bal_acc {:.3} macro_f1 {:.3} inv_acc {:.3} inv_bal {:.3} inv_f1 {:.3} code_p {:.3} code_r {:.3} code_f1 {:.3} done_f1 {:.3} trans_cos {trans_cos:.4} code_rate {:.3} pred_code {:.3} done_rate {:.3} pred_done {:.3} sel {selection_metric:.4}{}",
                    config.steps,
                    action_metrics.accuracy,
                    action_metrics.balanced_accuracy,
                    action_metrics.macro_f1,
                    inverse_metrics.accuracy,
                    inverse_metrics.balanced_accuracy,
                    inverse_metrics.macro_f1,
                    action_metrics.code_precision,
                    action_metrics.code_recall,
                    action_metrics.code_f1,
                    action_metrics.done_f1,
                    action_metrics.code_rate,
                    action_metrics.pred_code_rate,
                    action_metrics.done_rate,
                    action_metrics.pred_done_rate,
                    memory_note
                );
            }
            let checkpoint_resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric,
                best_aux_metric: best_loss,
                saved_checkpoint,
            };
            let mut checkpoint_artifacts = vec![util::varmap_checkpoint_artifact(
                &world_varmap,
                &train_checkpoint_path,
            )?];
            if config.train_encoder {
                checkpoint_artifacts.push(util::varmap_checkpoint_artifact(
                    &encoder_varmap,
                    &encoder_train_checkpoint_path,
                )?);
            }
            checkpoint_artifacts.push(util::optimizer_checkpoint_artifact(
                &opt,
                &optimizer_checkpoint_path,
            )?);
            checkpoint_artifacts.push(util::resume_checkpoint_artifact(
                &checkpoint_resume_state,
                &resume_state_path,
            )?);
            util::save_checkpoint_job(
                checkpoint_writer.as_ref(),
                format!("world step {step}"),
                checkpoint_artifacts,
            )?;
        }
    }

    if let Some(writer) = checkpoint_writer.as_mut() {
        writer.finish()?;
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&world_varmap, &model_path)?;
        if config.train_encoder {
            util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
        }
        println!(
            "No checkpoint was saved during logging; saved final LeWM weights to {:?}",
            model_path
        );
    }
    util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
    if config.train_encoder {
        util::save_varmap_atomic(&encoder_varmap, &encoder_train_checkpoint_path)?;
    }
    opt.save_state(&optimizer_checkpoint_path)?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: config.steps,
        best_metric,
        best_aux_metric: best_loss,
        saved_checkpoint,
    };
    util::save_resume_state(&resume_state_path, &resume_state)?;
    tb.flush();
    tb.finish()?;
    let _ = vram_tracker.write_summary(&run_dir, "world");
    if saved_checkpoint {
        println!(
            "Best world model saved to {:?} (selection {:.4})",
            model_path, best_loss
        );
    } else {
        println!(
            "Final world model saved to {:?} (run finished before first logging checkpoint)",
            model_path
        );
    }
    if config.train_encoder {
        println!("LeWM encoder saved to {:?}", encoder_model_path);
    }
    Ok(())
}

struct HighWorldMacroBatch {
    examples: Vec<WorldExample>,
    action_sequences: Vec<Vec<u32>>,
}

#[cfg(test)]
fn high_world_macro_example_from_raw_span(
    span: &[RawWorldExample],
    vocab: &Vocab,
) -> Result<(WorldExample, Vec<u32>)> {
    let first = span
        .first()
        .context("high-world macro span unexpectedly empty")?;
    let mut next_tokens = Vec::new();
    for row in span {
        next_tokens.extend(encode_text_with_vocab_mode(
            &row.next_text,
            vocab,
            TokenizationMode::Default,
        ));
    }
    let action_sequence = span.iter().map(|row| row.action_label).collect();
    Ok((
        WorldExample {
            state_tokens: encode_text_with_vocab_mode(
                &first.state_text,
                vocab,
                TokenizationMode::Default,
            ),
            next_tokens,
            action_label: first.action_label,
        },
        action_sequence,
    ))
}

#[cfg(test)]
fn high_world_macro_example_from_cached_span(
    span: &[WorldExample],
) -> Result<(WorldExample, Vec<u32>)> {
    let first = span
        .first()
        .context("cached high-world macro span unexpectedly empty")?;
    let next_tokens = span
        .iter()
        .flat_map(|row| row.next_tokens.iter().copied())
        .collect();
    let action_sequence = span.iter().map(|row| row.action_label).collect();
    Ok((
        WorldExample {
            state_tokens: first.state_tokens.clone(),
            next_tokens,
            action_label: first.action_label,
        },
        action_sequence,
    ))
}

fn collect_macro_chains(
    examples: &[WorldExample],
    macro_min_len: usize,
    macro_max_len: usize,
) -> Vec<Vec<usize>> {
    if examples.is_empty() {
        return Vec::new();
    }
    let edges = continuation_edges(examples);
    let span_range = macro_max_len.saturating_sub(macro_min_len) + 1;
    let mut chains = Vec::new();
    for start in 0..examples.len() {
        for span_offset in 0..span_range {
            let target_len = macro_min_len + span_offset;
            let mut chain = vec![start];
            let mut current = start;
            while chain.len() < target_len {
                match edges[current] {
                    Some(next) if !chain.contains(&next) => {
                        chain.push(next);
                        current = next;
                    }
                    _ => break,
                }
            }
            if chain.len() >= macro_min_len {
                chains.push(chain);
            }
        }
    }
    chains
}

fn sample_macro_chains(chains: &[Vec<usize>], count: usize) -> Vec<Vec<usize>> {
    if chains.is_empty() {
        return Vec::new();
    }
    let mut rng = rand::rng();
    (0..count)
        .map(|_| chains[rng.random_range(0..chains.len())].clone())
        .collect()
}

fn macro_example_from_chain(
    examples: &[WorldExample],
    chain: &[usize],
) -> (WorldExample, Vec<u32>) {
    let first = &examples[chain[0]];
    let next_tokens = chain
        .iter()
        .flat_map(|&idx| examples[idx].next_tokens.iter().copied())
        .collect();
    let action_sequence = chain
        .iter()
        .map(|&idx| examples[idx].action_label)
        .collect();
    (
        WorldExample {
            state_tokens: first.state_tokens.clone(),
            next_tokens,
            action_label: first.action_label,
        },
        action_sequence,
    )
}

fn macro_batch_from_examples(
    examples: &[WorldExample],
    batch_size: usize,
    macro_min_len: usize,
    macro_max_len: usize,
) -> Result<HighWorldMacroBatch> {
    let macro_min_len = macro_min_len.max(1);
    let macro_max_len = macro_max_len.max(macro_min_len);
    let chains = collect_macro_chains(examples, macro_min_len, macro_max_len);
    let sampled = if chains.is_empty() {
        examples
            .iter()
            .take(batch_size)
            .map(|example| (example.clone(), vec![example.action_label]))
            .collect::<Vec<_>>()
    } else {
        sample_macro_chains(&chains, batch_size)
            .into_iter()
            .map(|chain| macro_example_from_chain(examples, &chain))
            .collect()
    };
    if sampled.is_empty() {
        anyhow::bail!(
            "high-world macro batch could not be built from {} examples",
            examples.len()
        );
    }
    let mut examples_out = Vec::with_capacity(sampled.len());
    let mut action_sequences = Vec::with_capacity(sampled.len());
    for (example, action_sequence) in sampled {
        examples_out.push(example);
        action_sequences.push(action_sequence);
    }
    Ok(HighWorldMacroBatch {
        examples: examples_out,
        action_sequences,
    })
}

fn collect_high_world_macro_batch(
    stream: &mut RawWorldStream,
    vocab: &Vocab,
    batch_size: usize,
    macro_min_len: usize,
    macro_max_len: usize,
) -> Result<HighWorldMacroBatch> {
    let macro_min_len = macro_min_len.max(1);
    let macro_max_len = macro_max_len.max(macro_min_len);
    let span_range = macro_max_len.saturating_sub(macro_min_len) + 1;
    let prefetch = batch_size
        .saturating_mul(span_range)
        .saturating_mul(16)
        .max(batch_size * macro_max_len);
    let rows = stream.next_batch(prefetch)?;
    let examples = rows
        .iter()
        .map(|row| {
            Ok(WorldExample {
                state_tokens: encode_text_with_vocab_mode(
                    &row.state_text,
                    vocab,
                    TokenizationMode::Default,
                ),
                next_tokens: encode_text_with_vocab_mode(
                    &row.next_text,
                    vocab,
                    TokenizationMode::Default,
                ),
                action_label: row.action_label,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    macro_batch_from_examples(&examples, batch_size, macro_min_len, macro_max_len)
}

fn collect_high_world_macro_batch_cached(
    stream: &mut CachedWorldStream,
    batch_size: usize,
    macro_min_len: usize,
    macro_max_len: usize,
) -> Result<HighWorldMacroBatch> {
    let macro_min_len = macro_min_len.max(1);
    let macro_max_len = macro_max_len.max(macro_min_len);
    let span_range = macro_max_len.saturating_sub(macro_min_len) + 1;
    let prefetch = batch_size
        .saturating_mul(span_range)
        .saturating_mul(16)
        .max(batch_size * macro_max_len);
    let examples = stream.next_batch(prefetch)?;
    macro_batch_from_examples(&examples, batch_size, macro_min_len, macro_max_len)
}

fn run_high_world_training(config: HighWorldTrainConfig) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            tracing::warn!("CUDA not available: {}", e);
            Device::Cpu
        }
    };

    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let row_count = count_raw_world_rows(&config.data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &config.data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let world_cache_path =
        compatible_world_cache_path(&config.data_path, config.max_seq, &encoder_vocab_sig)?;
    let mut cached_macro_stream = if let Some(cache_path) = world_cache_path.as_ref() {
        println!("Token cache: using high-world cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            cache_path,
            1,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        println!("Token cache: no high-world cache found; using raw tokenization stream");
        None
    };
    let mut macro_stream = if cached_macro_stream.is_none() {
        Some(RawWorldStream::with_split(
            &config.data_path,
            1,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        None
    };
    let mut cached_val_macro_stream = if val_row_count > 0 {
        if let Some(cache_path) = world_cache_path.as_ref() {
            Some(CachedWorldStream::with_split(
                cache_path,
                1,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let mut val_macro_stream = if val_row_count > 0 && cached_val_macro_stream.is_none() {
        Some(RawWorldStream::with_split(
            &config.data_path,
            1,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            false,
        )?)
    } else {
        None
    };
    let vocab_size = encoder_vocab.id_to_token.len();

    let encoder = {
        let mut encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
        let loaded_encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        let frozen_encoder_tensors = util::frozen_tensors_from_varmap(&encoder_varmap)?;
        drop(loaded_encoder);
        let encoder_vb = VarBuilder::from_tensors(frozen_encoder_tensors, train_dtype, &device);
        OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?
    };

    let context_compressor = {
        let mut world_varmap = VarMap::new();
        let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
        let loaded_context_compressor = ContextCompressor::new(
            world_vb.pp("context_compressor"),
            config.dim,
            config.bridge_dim,
            config.num_latent_tokens,
        )?;
        util::load_varmap_checked(&mut world_varmap, &config.world_model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        let frozen_world_tensors = util::frozen_tensors_from_varmap(&world_varmap)?;
        drop(loaded_context_compressor);
        let world_vb = VarBuilder::from_tensors(frozen_world_tensors, train_dtype, &device);
        ContextCompressor::new(
            world_vb.pp("context_compressor"),
            config.dim,
            config.bridge_dim,
            config.num_latent_tokens,
        )?
    };

    let mut high_varmap = VarMap::new();
    let high_vb = VarBuilder::from_varmap(&high_varmap, train_dtype, &device);
    let macro_encoder = ActionSequenceEncoder::new(
        high_vb.pp("action_sequence_encoder"),
        config.bridge_dim,
        config.macro_max_len,
    )?;
    let macro_transition = MacroActionStateTransition::new(
        high_vb.pp("macro_action_state_transition"),
        config.bridge_dim,
    )?;

    let macro_encoder_ff = (config.bridge_dim * 4).max(256);
    let macro_encoder_hidden = (config.bridge_dim * 2).max(256);
    let macro_encoder_params = crate::model::action_classifier_head::NUM_ACTIONS
        * config.bridge_dim
        + config.macro_max_len * config.bridge_dim
        + (config.macro_max_len + 1) * config.bridge_dim
        + 2 * (4 * config.bridge_dim * config.bridge_dim
            + 2 * config.bridge_dim * macro_encoder_ff
            + macro_encoder_ff
            + 9 * config.bridge_dim)
        + 2 * 2 * config.bridge_dim
        + config.bridge_dim
        + 1
        + 2 * config.bridge_dim * macro_encoder_hidden
        + macro_encoder_hidden
        + config.bridge_dim;
    let macro_transition_ff = (config.bridge_dim * 5).max(320);
    let macro_transition_params = config.bridge_dim * config.bridge_dim
        + config.bridge_dim
        + 3 * (4 * config.bridge_dim * config.bridge_dim
            + 2 * config.bridge_dim * macro_transition_ff
            + macro_transition_ff
            + 9 * config.bridge_dim)
        + 2 * config.bridge_dim
        + 2 * (config.bridge_dim * config.bridge_dim + config.bridge_dim);
    let total_params = macro_encoder_params
        + macro_transition_params
        + config.num_latent_tokens * config.bridge_dim;
    let _ = fs::create_dir_all("local_models");
    let model_path = config.output_path.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "local_models/model_high_world_{}.safetensors",
            util::format_params(total_params)
        ))
    });
    let resume_stage = util::resume_stage_name("high_world");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        util::load_varmap_checked(&mut high_varmap, &train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut high_varmap, train_dtype)?;
        println!(
            "Resuming high-world weights from {:?}",
            train_checkpoint_path
        );
    } else if config.resume && model_path.exists() {
        util::load_varmap_checked(&mut high_varmap, &model_path)?;
        util::cast_varmap_dtype(&mut high_varmap, train_dtype)?;
        println!(
            "Resuming high-world weights from best export {:?} without optimizer state",
            model_path
        );
    }

    let named_train_vars = util::named_train_vars(&high_varmap)?;
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume {
        if let Some(state) = util::load_resume_state(&resume_state_path, &resume_stage)? {
            resume_state = state;
        }
        if optimizer_checkpoint_path.exists() {
            opt.load_state(&optimizer_checkpoint_path)?;
            if resume_state.step == 0 {
                resume_state.step = opt.step_t();
            }
        }
    }

    let run_dir = util::create_run_dir("high_world")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let async_checkpoints = env_bool("TOFY_HIGH_WORLD_ASYNC_CHECKPOINTS", true);
    let mut checkpoint_writer = if async_checkpoints {
        Some(util::AsyncCheckpointWriter::new())
    } else {
        None
    };
    println!(
        "Training high-level latent world model: rows={} val={} batch={} macro_len={}..{} slots={} dim={} dtype={:?}",
        train_row_count,
        val_row_count,
        config.batch_size,
        config.macro_min_len,
        config.macro_max_len,
        config.num_latent_tokens,
        config.bridge_dim,
        train_dtype
    );
    println!("Low-level world checkpoint: {:?}", config.world_model_path);
    println!("High-level world output: {:?}", model_path);
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    println!(
        "High-world sidecar checkpointing: async={} log_every={}",
        async_checkpoints, config.log_every
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/macro_min_len", config.macro_min_len as f32, 0);
    tb.add_scalar("config/macro_max_len", config.macro_max_len as f32, 0);
    tb.flush();

    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    let clip_norm = env_f64("TOFY_HIGH_WORLD_CLIP_NORM", 1.0).max(0.0);
    tb.add_scalar("config/clip_norm", clip_norm as f32, 0);
    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };
    if start_step >= config.steps {
        println!(
            "High-world resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }

    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = high_world_batch_size_for_step(step, &config);
        let grad_accum_steps = high_world_grad_accum_for_step(step, &config);
        if config.batch_warmup_steps > 0
            && step == config.batch_warmup_steps + 1
            && (config.batch_warmup_value != config.batch_size
                || config.grad_accum_warmup_value < config.grad_accum_steps)
        {
            println!(
                "High-world warmup complete at step {}; switching to batch={} grad_accum={} (effective={})",
                config.batch_warmup_steps,
                batch_size,
                grad_accum_steps,
                batch_size * grad_accum_steps
            );
        }

        for micro_step in 0..grad_accum_steps {
            let batch = if let Some(stream) = cached_macro_stream.as_mut() {
                collect_high_world_macro_batch_cached(
                    stream,
                    batch_size,
                    config.macro_min_len,
                    config.macro_max_len,
                )?
            } else {
                collect_high_world_macro_batch(
                    macro_stream
                        .as_mut()
                        .context("high-world raw stream missing")?,
                    &encoder_vocab,
                    batch_size,
                    config.macro_min_len,
                    config.macro_max_len,
                )?
            };
            let (state_slots, target_slots) = context_slots_from_world_pair_batch(
                &encoder,
                &context_compressor,
                &batch.examples,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                recursive_context_compressor,
                true,
                &device,
            )?;
            let macro_action =
                macro_encoder.forward_from_slices(&batch.action_sequences, &device)?;
            let pred_slots = macro_transition.forward(&state_slots, &macro_action)?;
            let fixed_target_slots = target_slots.detach();
            let transition_loss = prediction_loss(&pred_slots, &fixed_target_slots)?;
            let target_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&target_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let pred_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&pred_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let sigreg_loss = world_sigreg_loss(&target_sigreg, &target_sigreg, &pred_sigreg)?;
            let loss = transition_loss.broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            let should_capture_log =
                step % config.log_every == 0 && micro_step + 1 == grad_accum_steps;
            if should_capture_log {
                let metric_pred_slots = pred_slots.detach();
                let metric_target_slots = target_slots.detach();
                log_snapshot = Some(HighWorldLogSnapshot {
                    loss_val: util::scalar_f32(&loss)?,
                    trans_val: util::scalar_f32(&transition_loss)?,
                    sigreg_val: util::scalar_f32(&sigreg_loss)?,
                    cosine: util::scalar_f32(&mean_cosine_similarity(
                        &metric_pred_slots,
                        &metric_target_slots,
                    )?)?,
                    pred_rms: util::scalar_f32(&tensor_rms(&metric_pred_slots)?)?,
                    target_rms: util::scalar_f32(&tensor_rms(&metric_target_slots)?)?,
                });
            }
        }

        opt.set_learning_rate(util::scheduled_lr(config.lr, step, config.steps));
        let grad_norm = util::clip_accumulated_gradients_device(
            &mut accumulated_grads,
            &train_vars,
            clip_norm,
        )?;
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            if let Some(grad_norm) = grad_norm.as_ref() {
                tb.add_scalar("grad/global_norm", util::scalar_f32(grad_norm)?, step);
            }
            let HighWorldLogSnapshot {
                loss_val,
                trans_val,
                sigreg_val,
                cosine,
                pred_rms,
                target_rms,
            } = log_snapshot.context("high-world accumulation produced no log snapshot")?;
            let mut selection_metric = trans_val + config.lambda as f32 * sigreg_val;
            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/transition", trans_val, step);
            tb.add_scalar("loss/sigreg", sigreg_val, step);
            tb.add_scalar("metrics/cosine", cosine, step);
            tb.add_scalar("metrics/pred_rms", pred_rms, step);
            tb.add_scalar("metrics/target_rms", target_rms, step);
            tb.add_scalar("metrics/selection", selection_metric, step);
            if cached_val_macro_stream.is_some() || val_macro_stream.is_some() {
                let val_batch = if let Some(stream) = cached_val_macro_stream.as_mut() {
                    collect_high_world_macro_batch_cached(
                        stream,
                        batch_size,
                        config.macro_min_len,
                        config.macro_max_len,
                    )?
                } else {
                    collect_high_world_macro_batch(
                        val_macro_stream
                            .as_mut()
                            .context("high-world validation stream missing")?,
                        &encoder_vocab,
                        batch_size,
                        config.macro_min_len,
                        config.macro_max_len,
                    )?
                };
                let (val_state_slots, val_target_slots) = context_slots_from_world_pair_batch(
                    &encoder,
                    &context_compressor,
                    &val_batch.examples,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_context_compressor,
                    true,
                    &device,
                )?;
                let val_macro_action =
                    macro_encoder.forward_from_slices(&val_batch.action_sequences, &device)?;
                let val_pred_slots =
                    macro_transition.forward(&val_state_slots, &val_macro_action)?;
                let val_trans = util::scalar_f32(&prediction_loss(
                    &val_pred_slots,
                    &val_target_slots.detach(),
                )?)?;
                let val_target_sigreg = sigreg_epps_pulley(
                    &flatten_latent_slots(&val_target_slots)?,
                    sigreg_slices,
                    sigreg_points,
                )?;
                let val_pred_sigreg = sigreg_epps_pulley(
                    &flatten_latent_slots(&val_pred_slots)?,
                    sigreg_slices,
                    sigreg_points,
                )?;
                let val_sigreg = util::scalar_f32(&world_sigreg_loss(
                    &val_target_sigreg,
                    &val_target_sigreg,
                    &val_pred_sigreg,
                )?)?;
                selection_metric = val_trans + config.lambda as f32 * val_sigreg;
                tb.add_scalar("val/transition", val_trans, step);
                tb.add_scalar("val/sigreg", val_sigreg, step);
                tb.add_scalar("val/selection", selection_metric, step);
            }
            if let Some(snapshot) = vram_tracker.sample() {
                tb.add_scalar("memory/used_mb", snapshot.used_mb, step);
                tb.add_scalar("memory/free_mb", snapshot.free_mb, step);
            }
            tb.flush();

            if selection_metric < best_metric {
                best_metric = selection_metric;
                util::save_varmap_atomic(&high_varmap, &model_path)?;
                saved_checkpoint = true;
                println!(
                    "step {}/{} high_world total {:.4} trans {:.4} sigreg {:.4} cosine {:.4} pred_rms {:.4} target_rms {:.4} sel {:.4} [saved best]",
                    step,
                    config.steps,
                    loss_val,
                    trans_val,
                    sigreg_val,
                    cosine,
                    pred_rms,
                    target_rms,
                    selection_metric
                );
            } else {
                println!(
                    "step {}/{} high_world total {:.4} trans {:.4} sigreg {:.4} cosine {:.4} pred_rms {:.4} target_rms {:.4} sel {:.4}",
                    step,
                    config.steps,
                    loss_val,
                    trans_val,
                    sigreg_val,
                    cosine,
                    pred_rms,
                    target_rms,
                    selection_metric
                );
            }
            let checkpoint_resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric,
                best_aux_metric: best_metric,
                saved_checkpoint,
            };
            util::save_checkpoint_job(
                checkpoint_writer.as_ref(),
                format!("high-world step {step}"),
                vec![
                    util::varmap_checkpoint_artifact(&high_varmap, &train_checkpoint_path)?,
                    util::optimizer_checkpoint_artifact(&opt, &optimizer_checkpoint_path)?,
                    util::resume_checkpoint_artifact(&checkpoint_resume_state, &resume_state_path)?,
                ],
            )?;
        }
    }

    if let Some(writer) = checkpoint_writer.as_mut() {
        writer.finish()?;
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&high_varmap, &model_path)?;
        saved_checkpoint = true;
    }
    util::save_varmap_atomic(&high_varmap, &train_checkpoint_path)?;
    opt.save_state(&optimizer_checkpoint_path)?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: config.steps,
        best_metric,
        best_aux_metric: best_metric,
        saved_checkpoint,
    };
    util::save_resume_state(&resume_state_path, &resume_state)?;
    tb.flush();
    tb.finish()?;
    let _ = vram_tracker.write_summary(&run_dir, "high_world");
    println!(
        "High-level world model saved to {:?} (selection {:.4})",
        model_path, best_metric
    );
    Ok(())
}

fn run_orchestrator_training(config: OrchestratorTrainConfig) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            tracing::warn!("CUDA not available: {}", e);
            Device::Cpu
        }
    };

    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let row_count = count_raw_world_rows(&config.data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &config.data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let mut train_stream = RawWorldStream::with_split(
        &config.data_path,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
        true,
    )?;
    let mut val_stream = if val_row_count > 0 {
        Some(RawWorldStream::with_split(
            &config.data_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            false,
        )?)
    } else {
        None
    };
    let world_cache_path =
        compatible_world_cache_path(&config.data_path, config.max_seq, &encoder_vocab_sig)?;
    let mut cached_train_stream = if let Some(cache_path) = world_cache_path.as_ref() {
        println!(
            "Token cache: using action_classifier world cache {:?}",
            cache_path
        );
        Some(CachedWorldStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        println!(
            "Token cache: no action_classifier world cache found; using raw tokenization stream"
        );
        None
    };
    let mut cached_val_stream = if val_row_count > 0 {
        if let Some(cache_path) = world_cache_path.as_ref() {
            Some(CachedWorldStream::with_split(
                cache_path,
                DEFAULT_STREAM_SHUFFLE_BUFFER,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    let encoder = {
        let mut encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
        let loaded_encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        let frozen_encoder_tensors = util::frozen_tensors_from_varmap(&encoder_varmap)?;
        drop(loaded_encoder);
        let encoder_vb = VarBuilder::from_tensors(frozen_encoder_tensors, train_dtype, &device);
        OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            config.dim,
            config.num_layers,
            config.num_heads,
        )?
    };

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
    let context_compressor = ContextCompressor::new(
        world_vb.pp("context_compressor"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let _transition =
        ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
    let action_classifier_head =
        NextActionClassifier::new(world_vb.pp("next_action_classifier"), config.bridge_dim)?;
    let _inverse_action_classifier =
        if checkpoint_has_prefix(&config.world_model_path, "inverse_action_classifier.") {
            Some(NextActionClassifier::new(
                world_vb.pp("inverse_action_classifier"),
                config.bridge_dim,
            )?)
        } else {
            None
        };
    util::load_varmap_checked(&mut world_varmap, &config.world_model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;

    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| config.world_model_path.clone());
    let resume_stage = util::resume_stage_name("action_classifier");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&output_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&output_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&output_path, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        util::load_varmap_checked(&mut world_varmap, &train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!(
            "Resuming action_classifier weights from {:?}",
            train_checkpoint_path
        );
    }

    // Only the classifier (and, when tuning the planner, the context
    // compressor) receive gradients here; keeping the frozen transition
    // weights out of the optimizer avoids decaying or mutating them.
    let named_train_vars = util::named_train_vars(&world_varmap)?
        .into_iter()
        .filter(|entry| {
            entry.name.starts_with("next_action_classifier.")
                || (config.tune_planner && entry.name.starts_with("context_compressor."))
        })
        .collect::<Vec<_>>();
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume {
        if let Some(state) = util::load_resume_state(&resume_state_path, &resume_stage)? {
            resume_state = state;
        }
        if optimizer_checkpoint_path.exists() {
            opt.load_state(&optimizer_checkpoint_path)?;
            if resume_state.step == 0 {
                resume_state.step = opt.step_t();
            }
            println!(
                "Resuming action_classifier optimizer from {:?} at step {}",
                optimizer_checkpoint_path, resume_state.step
            );
        }
    }

    println!("Training (context compressor/action_classifier action model)");
    println!("Encoder checkpoint: {:?}", config.encoder_model_path);
    println!("World checkpoint: {:?}", config.world_model_path);
    println!(
        "Rows: train {} | val {} | max_seq {} | context_slots {} | tune_planner {}",
        train_row_count,
        val_row_count,
        config.max_seq,
        config.num_latent_tokens,
        config.tune_planner
    );

    let run_dir = util::create_run_dir("action_classifier")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let async_checkpoints = env_bool("TOFY_ORCHESTRATOR_ASYNC_CHECKPOINTS", true);
    let mut checkpoint_writer = if async_checkpoints {
        Some(util::AsyncCheckpointWriter::new())
    } else {
        None
    };
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    println!(
        "Action classifier sidecar checkpointing: async={} log_every={}",
        async_checkpoints, config.log_every
    );
    tb.add_scalar("run/alive", 1.0, 0);
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };
    tb.add_scalar("resume/start_step", start_step as f32, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/context_slots", config.num_latent_tokens as f32, 0);
    tb.add_scalar("config/grad_accum", config.grad_accum_steps as f32, 0);
    tb.add_scalar(
        "config/train_dtype",
        match train_dtype {
            DType::F16 => 16.0,
            DType::BF16 => 17.0,
            _ => 32.0,
        },
        0,
    );
    tb.add_scalar(
        "config/tune_planner",
        if config.tune_planner { 1.0 } else { 0.0 },
        0,
    );
    if let Some(vram) = vram_tracker.sample() {
        tb.add_scalar("memory/used_mb", vram.used_mb, 0);
        tb.add_scalar("memory/free_mb", vram.free_mb, 0);
        tb.add_scalar("memory/total_mb", vram.total_mb, 0);
        tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, 0);
    }
    tb.flush();

    const TARGET_CODE_RATE: f32 = 0.35;
    const TARGET_DONE_RATE: f32 = 0.20;
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let clip_norm = env_f64("TOFY_ORCHESTRATOR_CLIP_NORM", 1.0).max(0.0);
    tb.add_scalar("config/clip_norm", clip_norm as f32, 0);
    let mut best_score = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;

    if start_step >= config.steps {
        println!(
            "Orchestrator resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;

        let grad_accum_steps = config.grad_accum_steps.max(1);
        for micro_step in 0..grad_accum_steps {
            let batch = if let Some(ref mut cached_stream) = cached_train_stream {
                collect_action_training_batch_cached(
                    cached_stream,
                    config.batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?
            } else {
                let raw_batch = collect_action_training_batch(
                    &mut train_stream,
                    config.batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?;
                encode_world_examples(&raw_batch, &encoder_vocab)
            };
            let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
            let state_tokens = batch
                .iter()
                .map(|row| row.state_tokens.as_slice())
                .collect::<Vec<_>>();
            let mut state_slots = context_slots_from_token_sequences(
                &encoder,
                &context_compressor,
                &state_tokens,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                recursive_context_compressor,
                &device,
            )?;
            if !config.tune_planner {
                state_slots = state_slots.detach();
            }
            let action_logits = action_classifier_head.forward(&state_slots)?;
            let action_loss = action_cross_entropy(&action_logits, &action_labels, &device)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &action_loss,
                grad_accum_steps,
            )?;

            let should_capture_log =
                step % config.log_every == 0 && micro_step + 1 == grad_accum_steps;
            if should_capture_log {
                let metric_action_logits = action_logits.detach();
                log_snapshot = Some(OrchestratorLogSnapshot {
                    action_loss_val: util::scalar_f32(&action_loss)?,
                    metrics: compute_action_metrics(&metric_action_logits, &action_labels)?,
                });
            }
        }

        opt.set_learning_rate(util::scheduled_lr(config.lr, step, config.steps));
        let grad_norm = util::clip_accumulated_gradients_device(
            &mut accumulated_grads,
            &train_vars,
            clip_norm,
        )?;
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            if let Some(grad_norm) = grad_norm.as_ref() {
                tb.add_scalar("grad/global_norm", util::scalar_f32(grad_norm)?, step);
            }
            let OrchestratorLogSnapshot {
                action_loss_val,
                metrics,
            } = log_snapshot
                .context("action_classifier grad accumulation produced no log snapshot")?;
            let mut selection_score = 1.0
                - (0.7 * metrics.macro_f1 + 0.3 * metrics.balanced_accuracy)
                + 0.05 * action_loss_val;

            tb.add_scalar("loss/action", action_loss_val, step);
            tb.add_scalar("metrics/action_acc", metrics.accuracy, step);
            tb.add_scalar(
                "metrics/action_balanced_acc",
                metrics.balanced_accuracy,
                step,
            );
            tb.add_scalar("metrics/action_macro_f1", metrics.macro_f1, step);
            tb.add_scalar("metrics/code_precision", metrics.code_precision, step);
            tb.add_scalar("metrics/code_recall", metrics.code_recall, step);
            tb.add_scalar("metrics/code_f1", metrics.code_f1, step);
            tb.add_scalar("metrics/code_rate", metrics.code_rate, step);
            tb.add_scalar("metrics/pred_code_rate", metrics.pred_code_rate, step);
            tb.add_scalar("metrics/done_precision", metrics.done_precision, step);
            tb.add_scalar("metrics/done_recall", metrics.done_recall, step);
            tb.add_scalar("metrics/done_f1", metrics.done_f1, step);
            tb.add_scalar("metrics/done_rate", metrics.done_rate, step);
            tb.add_scalar("metrics/pred_done_rate", metrics.pred_done_rate, step);
            tb.add_scalar("metrics/fetch_docs_rate", metrics.fetch_docs_rate, step);
            tb.add_scalar(
                "metrics/pred_fetch_docs_rate",
                metrics.pred_fetch_docs_rate,
                step,
            );
            tb.add_scalar(
                "metrics/fetch_docs_precision",
                metrics.fetch_docs_precision,
                step,
            );
            tb.add_scalar("metrics/fetch_docs_recall", metrics.fetch_docs_recall, step);
            tb.add_scalar("metrics/fetch_docs_f1", metrics.fetch_docs_f1, step);

            let mut memory_note = String::new();
            if let Some(vram) = vram_tracker.sample() {
                tb.add_scalar("memory/used_mb", vram.used_mb, step);
                tb.add_scalar("memory/free_mb", vram.free_mb, step);
                tb.add_scalar("memory/total_mb", vram.total_mb, step);
                tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, step);
                memory_note = format!(
                    " vram {:.0}/{:.0}MB peak {:.0}MB",
                    vram.used_mb, vram.total_mb, vram.peak_used_mb
                );
            }

            if val_stream.is_some() || cached_val_stream.is_some() {
                let batch = if let Some(ref mut cached_stream) = cached_val_stream {
                    collect_action_training_batch_cached(
                        cached_stream,
                        config.batch_size,
                        TARGET_CODE_RATE,
                        TARGET_DONE_RATE,
                    )?
                } else {
                    let stream = val_stream
                        .as_mut()
                        .context("action_classifier validation stream missing")?;
                    let raw_batch = collect_action_training_batch(
                        stream,
                        config.batch_size,
                        TARGET_CODE_RATE,
                        TARGET_DONE_RATE,
                    )?;
                    encode_world_examples(&raw_batch, &encoder_vocab)
                };
                let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
                let state_tokens = batch
                    .iter()
                    .map(|row| row.state_tokens.as_slice())
                    .collect::<Vec<_>>();
                let state_slots = context_slots_from_token_sequences(
                    &encoder,
                    &context_compressor,
                    &state_tokens,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_context_compressor,
                    &device,
                )?
                .detach();
                let action_logits = action_classifier_head.forward(&state_slots)?;
                let val_loss = action_cross_entropy(&action_logits, &action_labels, &device)?;
                let val_loss_val = util::scalar_f32(&val_loss)?;
                let val_metrics = compute_action_metrics(&action_logits, &action_labels)?;
                selection_score = 1.0
                    - (0.7 * val_metrics.macro_f1 + 0.3 * val_metrics.balanced_accuracy)
                    + 0.05 * val_loss_val;
                tb.add_scalar("val/action", val_loss_val, step);
                tb.add_scalar("val/action_acc", val_metrics.accuracy, step);
                tb.add_scalar(
                    "val/action_balanced_acc",
                    val_metrics.balanced_accuracy,
                    step,
                );
                tb.add_scalar("val/action_macro_f1", val_metrics.macro_f1, step);
                tb.add_scalar("val/code_precision", val_metrics.code_precision, step);
                tb.add_scalar("val/code_recall", val_metrics.code_recall, step);
                tb.add_scalar("val/code_f1", val_metrics.code_f1, step);
                tb.add_scalar("val/code_rate", val_metrics.code_rate, step);
                tb.add_scalar("val/pred_code_rate", val_metrics.pred_code_rate, step);
                tb.add_scalar("val/done_precision", val_metrics.done_precision, step);
                tb.add_scalar("val/done_recall", val_metrics.done_recall, step);
                tb.add_scalar("val/done_f1", val_metrics.done_f1, step);
                tb.add_scalar("val/done_rate", val_metrics.done_rate, step);
                tb.add_scalar("val/pred_done_rate", val_metrics.pred_done_rate, step);
                tb.add_scalar("val/fetch_docs_rate", val_metrics.fetch_docs_rate, step);
                tb.add_scalar(
                    "val/pred_fetch_docs_rate",
                    val_metrics.pred_fetch_docs_rate,
                    step,
                );
                tb.add_scalar(
                    "val/fetch_docs_precision",
                    val_metrics.fetch_docs_precision,
                    step,
                );
                tb.add_scalar("val/fetch_docs_recall", val_metrics.fetch_docs_recall, step);
                tb.add_scalar("val/fetch_docs_f1", val_metrics.fetch_docs_f1, step);
            }
            tb.add_scalar("val/selection_score", selection_score, step);
            tb.flush();

            if selection_score < best_score {
                best_score = selection_score;
                util::save_varmap_atomic(&world_varmap, &output_path)?;
                saved_checkpoint = true;
                println!(
                    "step {}/{} action {:.4} acc {:.3} bal {:.3} macro_f1 {:.3} code_f1 {:.3} done_f1 {:.3} sel {:.4}{} [saved best]",
                    step,
                    config.steps,
                    action_loss_val,
                    metrics.accuracy,
                    metrics.balanced_accuracy,
                    metrics.macro_f1,
                    metrics.code_f1,
                    metrics.done_f1,
                    selection_score,
                    memory_note
                );
            } else {
                println!(
                    "step {}/{} action {:.4} acc {:.3} bal {:.3} macro_f1 {:.3} code_f1 {:.3} done_f1 {:.3} sel {:.4}{}",
                    step,
                    config.steps,
                    action_loss_val,
                    metrics.accuracy,
                    metrics.balanced_accuracy,
                    metrics.macro_f1,
                    metrics.code_f1,
                    metrics.done_f1,
                    selection_score,
                    memory_note
                );
            }
            let checkpoint_resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric: best_score,
                best_aux_metric: best_score,
                saved_checkpoint,
            };
            util::save_checkpoint_job(
                checkpoint_writer.as_ref(),
                format!("action-classifier step {step}"),
                vec![
                    util::varmap_checkpoint_artifact(&world_varmap, &train_checkpoint_path)?,
                    util::optimizer_checkpoint_artifact(&opt, &optimizer_checkpoint_path)?,
                    util::resume_checkpoint_artifact(&checkpoint_resume_state, &resume_state_path)?,
                ],
            )?;
        }
    }

    if let Some(writer) = checkpoint_writer.as_mut() {
        writer.finish()?;
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&world_varmap, &output_path)?;
    }
    util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
    opt.save_state(&optimizer_checkpoint_path)?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: config.steps,
        best_metric: best_score,
        best_aux_metric: best_score,
        saved_checkpoint,
    };
    util::save_resume_state(&resume_state_path, &resume_state)?;
    tb.flush();
    tb.finish()?;
    let _ = vram_tracker.write_summary(&run_dir, "action_classifier");
    println!(
        "Best context compressor/action_classifier checkpoint saved to {:?} (selection {:.4})",
        output_path, best_score
    );
    Ok(())
}

/// Approximate parameter count for decoder checkpoint: decoder conditioning adapter + decoder.
fn decoder_param_count(
    vocab_size: usize,
    planner_dim: usize,
    world_dim: usize,
    kind: DecoderKind,
    context_slots: usize,
    arch: DecoderArchitecture,
) -> usize {
    let dim = arch.dim;
    let n_layers = arch.num_layers;
    let ff = arch.ff_dim;
    let embed = vocab_size * dim;
    let lm_head = dim * vocab_size;
    let ln_final = 2 * dim;
    let kind_embed = 2 * dim;
    let structure_proj = world_dim * dim + dim;
    let adapter_rank = (dim / 4).max(64);
    let per_block = 4 * dim * dim
        + 2 * dim * dim
        + 2 * world_dim * dim
        + 3 * 2 * dim
        + dim * dim
        + dim
        + dim * adapter_rank
        + adapter_rank
        + adapter_rank * dim
        + dim
        + dim * ff
        + ff
        + ff * dim
        + dim;
    let adapter_slots = DecoderConditioningAdapter::output_slots_for(kind, context_slots);
    let decoder_conditioning_adapter = adapter_slots * planner_dim
        + crate::model::action_classifier_head::NUM_ACTIONS * planner_dim
        + 2 * (planner_dim * planner_dim + planner_dim * 4 * planner_dim)
        + planner_dim * world_dim
        + world_dim;
    decoder_conditioning_adapter
        + embed
        + kind_embed
        + structure_proj
        + n_layers * per_block
        + ln_final
        + lm_head
}

fn load_decoder_varmap_checked(varmap: &mut VarMap, path: &Path, dim: usize) -> Result<()> {
    let missing = util::load_varmap_allow_missing(varmap, path, &["decoder.lm_head.weight"])?;
    if missing.iter().any(|name| name == "decoder.lm_head.weight") {
        util::init_linear_head_from_embedding(
            varmap,
            "decoder.embed.weight",
            "decoder.lm_head.weight",
            1.0 / (dim as f64).sqrt(),
        )?;
        println!(
            "Initialized decoder lm_head.weight from legacy tied embedding in {:?}",
            path
        );
    }
    Ok(())
}

fn default_decoder_vocab_path(decoder_path: &Path) -> PathBuf {
    decoder_path.with_extension("vocab.txt")
}

fn decoder_vocab_manifest_path(vocab_path: &Path) -> PathBuf {
    vocab_path.with_extension("manifest.json")
}

fn decoder_vocab_manifest_matches(
    manifest: &DecoderVocabManifest,
    vocab: &Vocab,
    kind: DecoderKind,
    token_mode: TokenizationMode,
    max_vocab: usize,
    action_filter: u32,
) -> bool {
    manifest.version == DECODER_VOCAB_MANIFEST_VERSION
        && manifest.kind == kind.as_str()
        && manifest.tokenizer == token_mode.as_str()
        && manifest.max_vocab == max_vocab
        && manifest.action_filter == action_filter
        && manifest.vocab_signature == vocab_signature(vocab)
}

fn save_decoder_vocab_manifest(
    vocab_path: &Path,
    vocab: &Vocab,
    kind: DecoderKind,
    token_mode: TokenizationMode,
    max_vocab: usize,
    action_filter: u32,
) -> Result<()> {
    let manifest = DecoderVocabManifest {
        version: DECODER_VOCAB_MANIFEST_VERSION,
        kind: kind.as_str().to_string(),
        tokenizer: token_mode.as_str().to_string(),
        max_vocab,
        action_filter,
        vocab_signature: vocab_signature(vocab),
    };
    fs::write(
        decoder_vocab_manifest_path(vocab_path),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

pub(crate) fn ensure_code_decoder_vocab_manifest(
    vocab_path: &Path,
    max_vocab: usize,
) -> Result<()> {
    let vocab = load_vocab_from_file(vocab_path)?;
    save_decoder_vocab_manifest(
        vocab_path,
        &vocab,
        DecoderKind::CodeSpecialist,
        TokenizationMode::CodeAware,
        max_vocab,
        ACTION_CODE,
    )
}

fn decoder_varmap_checkpoint_artifact(
    varmap: &VarMap,
    path: &Path,
) -> Result<util::CheckpointArtifact> {
    Ok(util::CheckpointArtifact::TensorMap {
        path: path.to_path_buf(),
        tensors: util::varmap_tensor_snapshot(varmap)?,
    })
}

fn decoder_optimizer_checkpoint_artifact(
    opt: &util::TrainOptimizer,
    path: &Path,
) -> Result<util::CheckpointArtifact> {
    Ok(util::CheckpointArtifact::TensorMap {
        path: path.to_path_buf(),
        tensors: opt.state_tensors_snapshot()?,
    })
}

fn decoder_resume_checkpoint_artifact(
    state: &util::TrainingResumeState,
    path: &Path,
) -> Result<util::CheckpointArtifact> {
    Ok(util::CheckpointArtifact::Json {
        path: path.to_path_buf(),
        text: serde_json::to_string_pretty(state)?,
    })
}

fn save_decoder_checkpoint_job(
    writer: Option<&util::AsyncCheckpointWriter>,
    label: String,
    artifacts: Vec<util::CheckpointArtifact>,
) -> Result<bool> {
    if artifacts.is_empty() {
        return Ok(true);
    }
    if let Some(writer) = writer {
        writer.try_submit(util::CheckpointJob { label, artifacts })
    } else {
        util::save_checkpoint_artifacts(artifacts)?;
        Ok(true)
    }
}

struct DecoderContextCache {
    capacity: usize,
    map: HashMap<Vec<u32>, Tensor>,
    order: VecDeque<Vec<u32>>,
    hits: usize,
    misses: usize,
}

impl DecoderContextCache {
    fn from_env() -> Self {
        let capacity = std::env::var("TOFY_DECODER_CONTEXT_CACHE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1024);
        Self {
            capacity,
            map: HashMap::with_capacity(capacity.min(1024)),
            order: VecDeque::with_capacity(capacity.min(1024)),
            hits: 0,
            misses: 0,
        }
    }

    fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    fn get(&mut self, tokens: &[u32]) -> Option<Tensor> {
        let value = self.map.get(tokens).cloned();
        if value.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        value
    }

    fn insert(&mut self, tokens: Vec<u32>, slots: Tensor) {
        if self.capacity == 0 || self.map.contains_key(&tokens) {
            return;
        }
        self.map.insert(tokens.clone(), slots.detach());
        self.order.push_back(tokens);
        while self.map.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

fn decoder_context_cache_key(action_label: u32, state_tokens: &[u32]) -> Vec<u32> {
    let mut key = Vec::with_capacity(state_tokens.len() + 2);
    key.push(action_label);
    key.push(u32::MAX);
    key.extend_from_slice(state_tokens);
    key
}

fn transition_slots_for_labels(
    transition: &ActionStateTransition,
    state_slots: &Tensor,
    action_labels: &[u32],
    rollout_steps: usize,
) -> Result<Tensor> {
    let steps = rollout_steps.max(1);
    let mut slots = state_slots.clone();
    for _ in 0..steps {
        slots = transition.forward(&slots, action_labels)?;
    }
    Ok(slots)
}

#[allow(clippy::too_many_arguments)]
fn decoder_next_context_slots(
    cache: &mut DecoderContextCache,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    encoder_batch: &[WorldExample],
    decoder_action_label: u32,
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    rollout_steps: usize,
    device: &Device,
) -> Result<Tensor> {
    if encoder_batch.is_empty() {
        bail!("decoder context batch is empty");
    }
    if !cache.is_enabled() {
        let state_tokens = encoder_batch
            .iter()
            .map(|row| row.state_tokens.as_slice())
            .collect::<Vec<_>>();
        let state_slots = context_slots_from_token_sequences(
            encoder,
            context_compressor,
            &state_tokens,
            pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            device,
        )?;
        let decoder_action_labels = vec![decoder_action_label; encoder_batch.len()];
        let next_slots = transition_slots_for_labels(
            transition,
            &state_slots,
            &decoder_action_labels,
            rollout_steps,
        )?;
        return Ok(next_slots.detach());
    }

    let mut slots_by_row: Vec<Option<Tensor>> = (0..encoder_batch.len()).map(|_| None).collect();
    let mut miss_positions = Vec::new();
    let mut miss_token_refs = Vec::new();

    for (idx, row) in encoder_batch.iter().enumerate() {
        let key = decoder_context_cache_key(decoder_action_label, &row.state_tokens);
        if let Some(slots) = cache.get(&key) {
            slots_by_row[idx] = Some(slots);
        } else {
            miss_positions.push(idx);
            miss_token_refs.push(row.state_tokens.as_slice());
        }
    }

    if !miss_positions.is_empty() {
        let state_slots = context_slots_from_token_sequences(
            encoder,
            context_compressor,
            &miss_token_refs,
            pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            device,
        )?;
        let decoder_action_labels = vec![decoder_action_label; miss_positions.len()];
        let next_slots = transition_slots_for_labels(
            transition,
            &state_slots,
            &decoder_action_labels,
            rollout_steps,
        )?;
        for (miss_idx, row_idx) in miss_positions.iter().copied().enumerate() {
            let row_slots = next_slots.narrow(0, miss_idx, 1)?.detach();
            cache.insert(
                decoder_context_cache_key(
                    decoder_action_label,
                    &encoder_batch[row_idx].state_tokens,
                ),
                row_slots.clone(),
            );
            slots_by_row[row_idx] = Some(row_slots);
        }
    }

    let slots = slots_by_row
        .iter()
        .map(|slots| slots.as_ref().context("decoder context cache missing row"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&slots, 0)?.detach())
}

pub(crate) fn decoder_tokenization_mode(kind: DecoderKind) -> TokenizationMode {
    if kind == DecoderKind::CodeSpecialist {
        TokenizationMode::CodeAware
    } else {
        TokenizationMode::Default
    }
}

pub(crate) fn decoder_action_label_for_kind(kind: DecoderKind) -> u32 {
    match kind {
        DecoderKind::TextGeneralist => ACTION_TEXT_REPLY,
        DecoderKind::CodeSpecialist => ACTION_CODE,
    }
}

fn decoder_action_matches_kind(kind: DecoderKind, action_label: u32) -> bool {
    action_label == decoder_action_label_for_kind(kind)
}

fn decoder_action_name(kind: DecoderKind) -> &'static str {
    match kind {
        DecoderKind::TextGeneralist => "text_reply",
        DecoderKind::CodeSpecialist => "code",
    }
}

fn decoder_batch_refill_rounds() -> usize {
    std::env::var("TOFY_DECODER_BATCH_REFILL_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(action_batch_refill_rounds)
        .max(1)
}

fn ensure_decoder_batch_not_empty(len: usize, kind: DecoderKind) -> Result<()> {
    if len == 0 {
        bail!(
            "{} decoder batch found no {} rows after {} refill rounds; use matching training data or increase TOFY_DECODER_BATCH_REFILL_ROUNDS",
            kind.as_str(),
            decoder_action_name(kind),
            decoder_batch_refill_rounds()
        );
    }
    Ok(())
}

fn collect_decoder_raw_batch(
    stream: &mut RawWorldStream,
    batch_size: usize,
    kind: DecoderKind,
) -> Result<Vec<RawWorldExample>> {
    let target = batch_size.max(1);
    let mut batch = Vec::with_capacity(target);
    for _ in 0..decoder_batch_refill_rounds() {
        for example in stream.next_batch(target)? {
            if decoder_action_matches_kind(kind, example.action_label) {
                batch.push(example);
                if batch.len() >= target {
                    return Ok(batch);
                }
            }
        }
    }
    ensure_decoder_batch_not_empty(batch.len(), kind)?;
    Ok(batch)
}

fn collect_decoder_cached_batch(
    stream: &mut CachedDecoderStream,
    batch_size: usize,
    kind: DecoderKind,
) -> Result<Vec<crate::data::CachedDecoderExample>> {
    let target = batch_size.max(1);
    let mut batch = Vec::with_capacity(target);
    for _ in 0..decoder_batch_refill_rounds() {
        for example in stream.next_batch(target)? {
            if decoder_action_matches_kind(kind, example.decoder.action_label) {
                batch.push(example);
                if batch.len() >= target {
                    return Ok(batch);
                }
            }
        }
    }
    ensure_decoder_batch_not_empty(batch.len(), kind)?;
    Ok(batch)
}

fn collect_conditioned_decoder_batch(
    stream: &mut ConditionedDecoderStream,
    batch_size: usize,
    kind: DecoderKind,
) -> Result<Vec<ConditionedDecoderExample>> {
    let target = batch_size.max(1);
    let mut batch = Vec::with_capacity(target);
    for _ in 0..decoder_batch_refill_rounds() {
        for example in stream.next_batch(target)? {
            if decoder_action_matches_kind(kind, example.decoder.action_label) {
                batch.push(example);
                if batch.len() >= target {
                    return Ok(batch);
                }
            }
        }
    }
    ensure_decoder_batch_not_empty(batch.len(), kind)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::{
        append_padded_segment, build_code_repair_prompt, collect_macro_chains,
        context_compressor_mask_from_lengths, decoder_action_label_for_kind,
        decoder_action_matches_kind, default_context_hybrid_exact_tail,
        extract_go_prompt_declarations, high_world_macro_example_from_cached_span,
        high_world_macro_example_from_raw_span, output_needs_code_repair,
    };
    use crate::data::{
        RawWorldExample, WorldExample, ACTION_CODE, ACTION_DONE, ACTION_FETCH_DOCS,
        ACTION_TEXT_REPLY,
    };
    use crate::model::{DecoderKind, Vocab};
    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn decoder_specialists_accept_only_their_declared_action() {
        assert_eq!(
            decoder_action_label_for_kind(DecoderKind::TextGeneralist),
            ACTION_TEXT_REPLY
        );
        assert_eq!(
            decoder_action_label_for_kind(DecoderKind::CodeSpecialist),
            ACTION_CODE
        );

        assert!(decoder_action_matches_kind(
            DecoderKind::TextGeneralist,
            ACTION_TEXT_REPLY
        ));
        assert!(!decoder_action_matches_kind(
            DecoderKind::TextGeneralist,
            ACTION_CODE
        ));
        assert!(decoder_action_matches_kind(
            DecoderKind::CodeSpecialist,
            ACTION_CODE
        ));
        assert!(!decoder_action_matches_kind(
            DecoderKind::CodeSpecialist,
            ACTION_TEXT_REPLY
        ));
        assert!(!decoder_action_matches_kind(
            DecoderKind::CodeSpecialist,
            ACTION_DONE
        ));
        assert!(!decoder_action_matches_kind(
            DecoderKind::CodeSpecialist,
            ACTION_FETCH_DOCS
        ));
    }

    #[test]
    fn extract_go_prompt_declarations_keeps_contract_lines() {
        let prompt = "Return only Go code. Implement exactly this function:\n\
type Interval struct { Start int; End int }\n\
func MergeIntervals(intervals []Interval) []Interval\n\n\
Rules:\n- Use package main.\n";
        let declarations = extract_go_prompt_declarations(prompt);
        assert_eq!(
            declarations,
            vec![
                "type Interval struct { Start int; End int }".to_string(),
                "func MergeIntervals(intervals []Interval) []Interval".to_string(),
            ]
        );
    }

    #[test]
    fn repair_prompt_includes_real_feedback_and_required_declarations() {
        let prompt = "Return only Go code. Implement exactly this function:\n\
type Interval struct { Start int; End int }\n\
func MergeIntervals(intervals []Interval) []Interval\n";
        let repair_prompt = build_code_repair_prompt(
            prompt,
            "func MergeIntervals(intervals []Interval) []Interval { return nil }",
            "candidate.go:3:1: undefined: Interval",
        );
        assert!(repair_prompt.contains("Compiler feedback:\ncandidate.go:3:1: undefined: Interval"));
        assert!(repair_prompt
            .contains("Required declarations:\n- type Interval struct { Start int; End int }"));
        assert!(repair_prompt.contains("- func MergeIntervals(intervals []Interval) []Interval"));
    }

    #[test]
    fn code_repair_gate_rejects_non_code_noise() {
        let prompt =
            "Return only Go code. Implement exactly this function:\nfunc Add(a int, b int) int";
        assert!(output_needs_code_repair(
            prompt,
            "Here is the code:\nfunc Add(a int, b int) int { return a + b }",
        ));
        assert!(!output_needs_code_repair(
            prompt,
            "func Add(a int, b int) int { return a + b }",
        ));
    }

    #[test]
    fn train_decoder_rejects_mtp_before_data_resolution() {
        let args = [
            "tofy",
            "--train-decoder",
            "missing_encoder.safetensors",
            "missing_vocab.txt",
            "missing_world.safetensors",
            "hub:missing/dataset",
            "--mtp-loss-weight",
            "0.1",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let err = super::try_run_train_decoder(&args)
            .expect_err("unsupported MTP should fail before resolving data");

        assert!(
            err.to_string().contains("--mtp-loss-weight is unsupported"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn high_world_raw_macro_example_concatenates_every_continuation() -> Result<()> {
        let mut vocab = Vocab::new();
        vocab.ensure_byte_tokens();
        let span = vec![
            RawWorldExample {
                state_text: "s0".to_string(),
                next_text: "a".to_string(),
                action_label: ACTION_TEXT_REPLY,
            },
            RawWorldExample {
                state_text: "s1".to_string(),
                next_text: "b".to_string(),
                action_label: ACTION_CODE,
            },
        ];

        let (example, actions) = high_world_macro_example_from_raw_span(&span, &vocab)?;

        assert_eq!(vocab.decode_ids_lossy(&example.state_tokens), "s0");
        assert_eq!(vocab.decode_ids_lossy(&example.next_tokens), "ab");
        assert_eq!(actions, vec![ACTION_TEXT_REPLY, ACTION_CODE]);
        assert_eq!(example.action_label, ACTION_TEXT_REPLY);
        Ok(())
    }

    #[test]
    fn macro_chains_prefer_token_continuations_over_adjacent_rows() {
        std::env::set_var("TOFY_WORLD_ROLLOUT_MIN_OVERLAP", "2");
        let examples = vec![
            WorldExample {
                state_tokens: vec![10, 11],
                next_tokens: vec![12, 13],
                action_label: ACTION_TEXT_REPLY,
            },
            WorldExample {
                state_tokens: vec![10, 11, 12, 13],
                next_tokens: vec![14],
                action_label: ACTION_CODE,
            },
            WorldExample {
                state_tokens: vec![99],
                next_tokens: vec![100],
                action_label: ACTION_DONE,
            },
        ];
        let chains = collect_macro_chains(&examples, 2, 2);
        assert!(chains.iter().any(|chain| chain == &[0, 1]));
        assert!(!chains.iter().any(|chain| chain == &[0, 2]));
    }

    #[test]
    fn high_world_cached_macro_example_concatenates_every_continuation() -> Result<()> {
        let span = vec![
            WorldExample {
                state_tokens: vec![1, 2],
                next_tokens: vec![3],
                action_label: ACTION_TEXT_REPLY,
            },
            WorldExample {
                state_tokens: vec![1, 2, 3],
                next_tokens: vec![4, 5],
                action_label: ACTION_DONE,
            },
        ];

        let (example, actions) = high_world_macro_example_from_cached_span(&span)?;

        assert_eq!(example.state_tokens, vec![1, 2]);
        assert_eq!(example.next_tokens, vec![3, 4, 5]);
        assert_eq!(actions, vec![ACTION_TEXT_REPLY, ACTION_DONE]);
        assert_eq!(example.action_label, ACTION_TEXT_REPLY);
        Ok(())
    }

    #[test]
    fn empty_context_segment_keeps_token_and_chunk_masks_zero() -> Result<()> {
        let mut buf = Vec::new();
        let token_len = append_padded_segment(&mut buf, &[], 0, 0, 4, 0);

        assert_eq!(token_len, 0);
        assert_eq!(buf, vec![0, 0, 0, 0]);

        let device = Device::Cpu;
        let features = crate::model::encoders::EncoderFeatures {
            token_states: Tensor::zeros((1, 4, 2), DType::F32, &device)?,
            chunk_states: Tensor::zeros((1, 2, 2), DType::F32, &device)?,
            global_states: Tensor::zeros((1, 1, 2), DType::F32, &device)?,
            pooled_queries: Tensor::zeros((1, 2, 2), DType::F32, &device)?,
        };

        let mask = context_compressor_mask_from_lengths(&features, &[token_len])?;
        let rows = mask.to_vec2::<f32>()?;

        assert_eq!(&rows[0][0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&rows[0][4..6], &[0.0, 0.0]);
        assert_eq!(&rows[0][6..], &[1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn hybrid_exact_tail_default_covers_recent_segment_summaries() {
        assert_eq!(default_context_hybrid_exact_tail(512, 1), 534);
        assert_eq!(default_context_hybrid_exact_tail(512, 2), 1068);
        assert!(default_context_hybrid_exact_tail(67, 1) > 67);
    }
}

// world/decoder metric helpers live in tasks/world_support.rs

static ACTION_BATCH_SHORTAGE_WARNED: AtomicBool = AtomicBool::new(false);

fn action_batch_refill_rounds() -> usize {
    std::env::var("TOFY_ACTION_BATCH_REFILL_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64usize)
        .max(1)
}

#[allow(clippy::too_many_arguments)]
fn warn_action_batch_shortage_once(
    target_code: usize,
    have_code: usize,
    target_text: usize,
    have_text: usize,
    target_done: usize,
    have_done: usize,
    target_docs: usize,
    have_docs: usize,
) {
    if ACTION_BATCH_SHORTAGE_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "warning: action-balanced batch could not meet target rates; targets code/text/done/docs={target_code}/{target_text}/{target_done}/{target_docs}, collected={have_code}/{have_text}/{have_done}/{have_docs}. Consider increasing rare action rows or TOFY_ACTION_BATCH_REFILL_ROUNDS."
    );
}

fn collect_action_training_batch(
    stream: &mut RawWorldStream,
    batch_size: usize,
    target_code_rate: f32,
    target_done_rate: f32,
) -> Result<Vec<RawWorldExample>> {
    let mut rng = rand::rng();
    let target_code = ((batch_size as f32) * target_code_rate.clamp(0.0, 0.5)).round() as usize;
    let target_done = ((batch_size as f32) * target_done_rate.clamp(0.0, 0.4)).round() as usize;
    let target_code = target_code.min(batch_size.saturating_sub(1));
    let target_done = target_done.min(batch_size.saturating_sub(target_code));
    let target_text = batch_size.saturating_sub(target_code + target_done);
    let mut code_examples = Vec::new();
    let mut text_examples = Vec::new();
    let mut done_examples = Vec::new();
    let mut docs_examples = Vec::new();
    let target_docs_rate = std::env::var("TOFY_WORLD_FETCH_DOCS_RATE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.12)
        .clamp(0.0, 0.35);
    let target_docs = ((batch_size as f32) * target_docs_rate).round() as usize;
    let target_docs = target_docs.min(batch_size.saturating_sub(target_code + target_done));
    let target_text = target_text.saturating_sub(target_docs);

    for _ in 0..action_batch_refill_rounds() {
        for example in stream.next_batch(batch_size.max(1))? {
            match example.action_label {
                ACTION_CODE => code_examples.push(example),
                ACTION_DONE => done_examples.push(example),
                ACTION_FETCH_DOCS => docs_examples.push(example),
                _ => text_examples.push(example),
            }
        }
        if code_examples.len() >= target_code
            && text_examples.len() >= target_text
            && done_examples.len() >= target_done
            && docs_examples.len() >= target_docs
        {
            break;
        }
    }
    if code_examples.len() < target_code
        || text_examples.len() < target_text
        || done_examples.len() < target_done
        || docs_examples.len() < target_docs
    {
        warn_action_batch_shortage_once(
            target_code,
            code_examples.len(),
            target_text,
            text_examples.len(),
            target_done,
            done_examples.len(),
            target_docs,
            docs_examples.len(),
        );
    }

    code_examples.shuffle(&mut rng);
    text_examples.shuffle(&mut rng);
    done_examples.shuffle(&mut rng);
    docs_examples.shuffle(&mut rng);

    let take_code = target_code.min(code_examples.len());
    let take_text = target_text.min(text_examples.len());
    let take_done = target_done.min(done_examples.len());
    let take_docs = target_docs.min(docs_examples.len());
    let mut batch = Vec::with_capacity(batch_size);
    batch.extend(code_examples.drain(..take_code));
    batch.extend(text_examples.drain(..take_text));
    batch.extend(done_examples.drain(..take_done));
    batch.extend(docs_examples.drain(..take_docs));

    let mut leftovers = Vec::new();
    leftovers.extend(code_examples);
    leftovers.extend(text_examples);
    leftovers.extend(done_examples);
    leftovers.extend(docs_examples);
    leftovers.shuffle(&mut rng);
    for example in leftovers
        .into_iter()
        .take(batch_size.saturating_sub(batch.len()))
    {
        batch.push(example);
    }

    while batch.len() < batch_size {
        let mut extra = stream.next_batch(1)?;
        if let Some(example) = extra.pop() {
            batch.push(example);
        }
    }

    batch.shuffle(&mut rng);
    Ok(batch)
}

fn collect_action_training_batch_cached(
    stream: &mut CachedWorldStream,
    batch_size: usize,
    target_code_rate: f32,
    target_done_rate: f32,
) -> Result<Vec<WorldExample>> {
    let mut rng = rand::rng();
    let target_code = ((batch_size as f32) * target_code_rate.clamp(0.0, 0.5)).round() as usize;
    let target_done = ((batch_size as f32) * target_done_rate.clamp(0.0, 0.4)).round() as usize;
    let target_code = target_code.min(batch_size.saturating_sub(1));
    let target_done = target_done.min(batch_size.saturating_sub(target_code));
    let target_text = batch_size.saturating_sub(target_code + target_done);
    let mut code_examples = Vec::new();
    let mut text_examples = Vec::new();
    let mut done_examples = Vec::new();
    let mut docs_examples = Vec::new();
    let target_docs_rate = std::env::var("TOFY_WORLD_FETCH_DOCS_RATE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.12)
        .clamp(0.0, 0.35);
    let target_docs = ((batch_size as f32) * target_docs_rate).round() as usize;
    let target_docs = target_docs.min(batch_size.saturating_sub(target_code + target_done));
    let target_text = target_text.saturating_sub(target_docs);

    for _ in 0..action_batch_refill_rounds() {
        for example in stream.next_batch(batch_size.max(1))? {
            match example.action_label {
                ACTION_CODE => code_examples.push(example),
                ACTION_DONE => done_examples.push(example),
                ACTION_FETCH_DOCS => docs_examples.push(example),
                _ => text_examples.push(example),
            }
        }
        if code_examples.len() >= target_code
            && text_examples.len() >= target_text
            && done_examples.len() >= target_done
            && docs_examples.len() >= target_docs
        {
            break;
        }
    }
    if code_examples.len() < target_code
        || text_examples.len() < target_text
        || done_examples.len() < target_done
        || docs_examples.len() < target_docs
    {
        warn_action_batch_shortage_once(
            target_code,
            code_examples.len(),
            target_text,
            text_examples.len(),
            target_done,
            done_examples.len(),
            target_docs,
            docs_examples.len(),
        );
    }

    code_examples.shuffle(&mut rng);
    text_examples.shuffle(&mut rng);
    done_examples.shuffle(&mut rng);
    docs_examples.shuffle(&mut rng);

    let take_code = target_code.min(code_examples.len());
    let take_text = target_text.min(text_examples.len());
    let take_done = target_done.min(done_examples.len());
    let take_docs = target_docs.min(docs_examples.len());
    let mut batch = Vec::with_capacity(batch_size);
    batch.extend(code_examples.drain(..take_code));
    batch.extend(text_examples.drain(..take_text));
    batch.extend(done_examples.drain(..take_done));
    batch.extend(docs_examples.drain(..take_docs));

    let mut leftovers = Vec::new();
    leftovers.extend(code_examples);
    leftovers.extend(text_examples);
    leftovers.extend(done_examples);
    leftovers.extend(docs_examples);
    leftovers.shuffle(&mut rng);
    for example in leftovers
        .into_iter()
        .take(batch_size.saturating_sub(batch.len()))
    {
        batch.push(example);
    }

    while batch.len() < batch_size {
        let mut extra = stream.next_batch(1)?;
        if let Some(example) = extra.pop() {
            batch.push(example);
        }
    }

    batch.shuffle(&mut rng);
    Ok(batch)
}

fn run_decoder_training(config: DecoderTrainConfig) -> Result<()> {
    if config.mtp_loss_weight > 0.0 {
        anyhow::bail!(
            "--mtp-loss-weight is unsupported: this decoder has no dedicated future-token heads, \
             so reusing next-token logits for MTP trains conflicting targets"
        );
    }
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            tracing::warn!("CUDA not available: {}", e);
            Device::Cpu
        }
    };

    let data_path = config.data_path.clone();
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let decoder_token_mode = decoder_tokenization_mode(config.decoder_kind);
    let decoder_action_label = decoder_action_label_for_kind(config.decoder_kind);
    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);

    let encoder = {
        let mut encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
        let loaded_encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        let frozen_encoder_tensors = util::frozen_tensors_from_varmap(&encoder_varmap)?;
        drop(loaded_encoder);
        let encoder_vb = VarBuilder::from_tensors(frozen_encoder_tensors, train_dtype, &device);
        OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            config.dim,
            config.num_layers,
            config.num_heads,
        )?
    };

    let (context_compressor, transition) = {
        let mut world_varmap = VarMap::new();
        let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
        let loaded_context_compressor = ContextCompressor::new(
            world_vb.pp("context_compressor"),
            config.dim,
            config.bridge_dim,
            config.num_latent_tokens,
        )?;
        let loaded_transition =
            ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
        let _inverse_action_classifier =
            if checkpoint_has_prefix(&config.world_model_path, "inverse_action_classifier.") {
                Some(NextActionClassifier::new(
                    world_vb.pp("inverse_action_classifier"),
                    config.bridge_dim,
                )?)
            } else {
                None
            };
        util::load_varmap_checked(&mut world_varmap, &config.world_model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        let frozen_world_tensors = util::frozen_tensors_from_varmap(&world_varmap)?;
        drop(loaded_context_compressor);
        drop(loaded_transition);
        let world_vb = VarBuilder::from_tensors(frozen_world_tensors, train_dtype, &device);
        let context_compressor = ContextCompressor::new(
            world_vb.pp("context_compressor"),
            config.dim,
            config.bridge_dim,
            config.num_latent_tokens,
        )?;
        let transition =
            ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
        (context_compressor, transition)
    };

    let mut decoder_varmap = VarMap::new();
    let decoder_vb = VarBuilder::from_varmap(&decoder_varmap, train_dtype, &device);
    let decoder_conditioning_adapter = DecoderConditioningAdapter::new(
        decoder_vb.pp("decoder_conditioning_adapter"),
        config.bridge_dim,
        config.bridge_dim,
        DecoderConditioningAdapter::output_slots_for(config.decoder_kind, config.num_latent_tokens),
    )?;
    let decoder_adapter_compress_rate = decoder_conditioning_adapter.compress_rate();
    let decoder_path = config.decoder_output_path.clone().unwrap_or_else(|| {
        config
            .world_model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!(
                "{}_decoder.safetensors",
                if config.decoder_kind == DecoderKind::TextGeneralist {
                    "text"
                } else {
                    "code"
                }
            ))
    });
    let resume_stage = util::resume_stage_name("decoder");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&decoder_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&decoder_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&decoder_path, &resume_stage, "resume.json");
    let decoder_vocab_path = config
        .decoder_vocab_path
        .clone()
        .unwrap_or_else(|| default_decoder_vocab_path(&decoder_path));
    let (decoder_vocab, built_decoder_vocab) = if decoder_vocab_path.exists() {
        let vocab = load_vocab_from_file(&decoder_vocab_path)?;
        let manifest_path = decoder_vocab_manifest_path(&decoder_vocab_path);
        let manifest_text = fs::read_to_string(&manifest_path).with_context(|| {
            format!(
                "decoder vocab {:?} exists but matching manifest {:?} is missing; rebuild the vocab under the current decoder action filter",
                decoder_vocab_path,
                manifest_path
            )
        })?;
        let manifest: DecoderVocabManifest = serde_json::from_str(&manifest_text)
            .with_context(|| format!("parse decoder vocab manifest {:?}", manifest_path))?;
        if !decoder_vocab_manifest_matches(
            &manifest,
            &vocab,
            config.decoder_kind,
            decoder_token_mode,
            config.decoder_max_vocab,
            decoder_action_label,
        ) {
            anyhow::bail!(
                "decoder vocab {:?} manifest {:?} does not match kind={} tokenizer={} max_vocab={} action_filter={}; rebuild the decoder vocab",
                decoder_vocab_path,
                manifest_path,
                config.decoder_kind.as_str(),
                decoder_token_mode.as_str(),
                config.decoder_max_vocab,
                decoder_action_label
            );
        }
        (vocab, false)
    } else {
        let (vocab, _, _) = build_vocab_from_raw_world_file_with_mode_action_filter(
            &data_path,
            config.decoder_max_vocab,
            decoder_token_mode,
            Some(decoder_action_label),
        )?;
        (vocab, true)
    };
    let decoder_vocab_sig = vocab_signature(&decoder_vocab);
    let decoder_cache_info = if config.decoder_kind == DecoderKind::CodeSpecialist {
        compatible_decoder_dual_cache_info(
            &data_path,
            config.max_seq,
            &encoder_vocab_sig,
            &decoder_vocab_sig,
        )?
    } else {
        None
    };
    let decoder_cache_path = decoder_cache_info.as_ref().map(|info| info.path.clone());
    let (train_row_count, val_row_count, mut raw_stream, mut val_stream) =
        if let Some(cache_info) = decoder_cache_info.as_ref().filter(|info| info.rows > 0) {
            let val_rows = split_match_count(
                cache_info.rows,
                HELDOUT_SPLIT_MODULUS,
                HELDOUT_SPLIT_REMAINDER,
            );
            println!(
                "Token cache: using decoder manifest row counts rows={} train={} val={}",
                cache_info.rows,
                cache_info.rows.saturating_sub(val_rows),
                val_rows
            );
            (
                cache_info.rows.saturating_sub(val_rows),
                val_rows,
                None,
                None,
            )
        } else {
            let row_count = count_raw_world_rows_split_with_mode_action_filter(
                &data_path,
                decoder_token_mode,
                None,
                0,
                Some(decoder_action_label),
            )?;
            let val_row_count = count_raw_world_rows_split_with_mode_action_filter(
                &data_path,
                decoder_token_mode,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                Some(decoder_action_label),
            )
            .unwrap_or(0);
            let train_row_count = row_count.saturating_sub(val_row_count);
            let raw_stream = Some(RawWorldStream::with_split_mode(
                &data_path,
                DEFAULT_STREAM_SHUFFLE_BUFFER,
                decoder_token_mode,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                true,
            )?);
            let val_stream = if val_row_count > 0 {
                Some(RawWorldStream::with_split_mode(
                    &data_path,
                    DEFAULT_STREAM_SHUFFLE_BUFFER,
                    decoder_token_mode,
                    Some(HELDOUT_SPLIT_MODULUS),
                    HELDOUT_SPLIT_REMAINDER,
                    false,
                )?)
            } else {
                None
            };
            (train_row_count, val_row_count, raw_stream, val_stream)
        };
    let mut cached_decoder_stream = if let Some(cache_path) = decoder_cache_path.as_ref() {
        println!("Token cache: using decoder dual cache {:?}", cache_path);
        Some(CachedDecoderStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        if config.decoder_kind == DecoderKind::CodeSpecialist {
            println!(
                "Token cache: no compatible decoder dual cache found; using raw tokenization stream"
            );
        }
        None
    };
    let mut cached_decoder_val_stream = if val_row_count > 0 {
        if let Some(cache_path) = decoder_cache_path.as_ref() {
            Some(CachedDecoderStream::with_split(
                cache_path,
                DEFAULT_STREAM_SHUFFLE_BUFFER,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let vocab_size = decoder_vocab.id_to_token.len();
    let decoder_arch = DecoderArchitecture::new(
        config.decoder_dim,
        config.decoder_layers,
        config.decoder_heads,
        config.decoder_ff_dim,
    )?;
    let decoder = CodeDecoder::new(
        decoder_vb.pp("decoder"),
        vocab_size,
        decoder_arch.dim,
        config.bridge_dim,
        decoder_arch.num_layers,
        decoder_arch.num_heads,
        decoder_arch.ff_dim,
        config.decoder_kind,
    )?;
    let decoder_attention = decoder.attention_config();
    util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        load_decoder_varmap_checked(
            &mut decoder_varmap,
            &train_checkpoint_path,
            decoder_arch.dim,
        )?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!("Resuming decoder weights from {:?}", train_checkpoint_path);
    } else if config.resume && decoder_path.exists() {
        load_decoder_varmap_checked(&mut decoder_varmap, &decoder_path, decoder_arch.dim)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!(
            "Resuming decoder weights from best export {:?} without optimizer state",
            decoder_path
        );
    } else if let Some(ref p) = config.init_decoder_path {
        load_decoder_varmap_checked(&mut decoder_varmap, p, decoder_arch.dim)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!("Initialized decoder weights from {:?}", p);
    }

    let named_train_vars = util::named_train_vars(&decoder_varmap)?;
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume {
        if let Some(state) = util::load_resume_state(&resume_state_path, &resume_stage)? {
            resume_state = state;
        }
        if optimizer_checkpoint_path.exists() {
            opt.load_state(&optimizer_checkpoint_path)?;
            if resume_state.step == 0 {
                resume_state.step = opt.step_t();
            }
            println!(
                "Resuming decoder optimizer from {:?} at step {}",
                optimizer_checkpoint_path, resume_state.step
            );
        }
    }

    let decoder_params = decoder_param_count(
        vocab_size,
        config.bridge_dim,
        config.bridge_dim,
        config.decoder_kind,
        config.num_latent_tokens,
        decoder_arch,
    );
    let requested_conditioning_loss_weight = config.conditioning_loss_weight;
    let conditioning_loss_weight = if config.conditioning_negative_forwards {
        requested_conditioning_loss_weight
    } else {
        if requested_conditioning_loss_weight > 0.0 {
            println!(
                "WARNING: --conditioning-loss-weight {} ignored because negative-conditioning \
                 forwards are disabled; pass --conditioning-negative-forwards or set \
                 TOFY_DECODER_NEGATIVE_FORWARDS=1",
                requested_conditioning_loss_weight
            );
        }
        0.0
    };
    let format_loss_weight = std::env::var("TOFY_DECODER_FORMAT_LOSS_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or_else(|| {
            if config.decoder_kind == DecoderKind::CodeSpecialist {
                0.03
            } else {
                0.0
            }
        })
        .max(0.0);
    let compute_conditioning_metrics =
        config.conditioning_negative_forwards && env_bool("TOFY_DECODER_ABLATION_METRICS", false);
    let conditioning_margin = config.conditioning_margin;
    let conditioning_negatives = if config.conditioning_negative_forwards {
        DecoderConditioningNegatives::from_env()
    } else {
        DecoderConditioningNegatives::none()
    };
    let decoder_attention_query_block = env_usize("TOFY_DECODER_ATTENTION_QUERY_BLOCK", 1).max(1);
    let clip_norm = env_f64("TOFY_DECODER_CLIP_NORM", 1.0).max(0.0);

    println!(
        "Training ({} decoder with cross-attention to world latent)",
        config.decoder_kind.as_str()
    );
    println!("Encoder model: {:?}", config.encoder_model_path);
    println!("Encoder vocab: {:?}", config.encoder_vocab_path);
    println!("World model: {:?}", config.world_model_path);
    println!(
        "Data: train {} rows | val {} rows | decoder vocab {} | max_seq {} | tokenizer {}",
        train_row_count,
        val_row_count,
        vocab_size,
        config.max_seq,
        decoder_token_mode.as_str()
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "Decoder: kind={} dim={} layers={} heads={} ff={} (~{} params)",
        config.decoder_kind.as_str(),
        decoder_arch.dim,
        decoder_arch.num_layers,
        decoder_arch.num_heads,
        decoder_arch.ff_dim,
        util::format_params(decoder_params)
    );
    println!(
        "Decoder attention: local_window={} anchor_period={} csa_rate={} hca_rate={} csa_topk={} cross_schedule={} latent_prefix={} adapter_compress_rate={}",
        decoder_attention.local_window,
        decoder_attention.anchor_period,
        decoder_attention.csa_compress_rate,
        decoder_attention.hca_compress_rate,
        decoder_attention.csa_topk,
        decoder_attention.cross_attention_schedule.as_str(),
        decoder_attention.latent_prefix,
        decoder_adapter_compress_rate
    );
    println!(
        "Decoder attention runtime: query_block={} csa_topk_mask=on-device",
        decoder_attention_query_block
    );
    println!(
        "Decoder negative conditioning forwards: {} (weight={} negatives={})",
        if config.conditioning_negative_forwards {
            "on"
        } else {
            "off"
        },
        conditioning_loss_weight,
        conditioning_negatives.count()
    );
    println!(
        "Decoder action filter: {} rows only",
        decoder_action_name(config.decoder_kind)
    );
    println!("Save path: {:?}", decoder_path);
    println!("Decoder vocab path: {:?}", decoder_vocab_path);
    if let Some(parent) = decoder_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Some(parent) = decoder_vocab_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if built_decoder_vocab {
        save_vocab_to_file(&decoder_vocab, &decoder_vocab_path)?;
        save_decoder_vocab_manifest(
            &decoder_vocab_path,
            &decoder_vocab,
            config.decoder_kind,
            decoder_token_mode,
            config.decoder_max_vocab,
            decoder_action_label,
        )?;
    }

    let run_dir = util::create_run_dir("decoder")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let prompt_dropout = std::env::var("TOFY_DECODER_PROMPT_DROPOUT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    tb.add_scalar("run/alive", 1.0, 0);
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };
    tb.add_scalar("resume/start_step", start_step as f32, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/context_slots", config.num_latent_tokens as f32, 0);
    tb.add_scalar("config/estimated_params", decoder_params as f32, 0);
    tb.add_scalar(
        "config/syntax_loss_weight",
        config.syntax_loss_weight as f32,
        0,
    );
    tb.add_scalar(
        "config/signature_loss_weight",
        config.signature_loss_weight as f32,
        0,
    );
    tb.add_scalar(
        "config/structure_loss_weight",
        config.structure_loss_weight as f32,
        0,
    );
    tb.add_scalar(
        "config/conditioning_loss_weight",
        conditioning_loss_weight as f32,
        0,
    );
    tb.add_scalar("config/format_loss_weight", format_loss_weight as f32, 0);
    tb.add_scalar(
        "config/requested_conditioning_loss_weight",
        requested_conditioning_loss_weight as f32,
        0,
    );
    tb.add_scalar(
        "config/conditioning_negative_forwards",
        if config.conditioning_negative_forwards {
            1.0
        } else {
            0.0
        },
        0,
    );
    tb.add_scalar("config/prompt_dropout", prompt_dropout as f32, 0);
    tb.add_scalar(
        "config/conditioning_metrics",
        if compute_conditioning_metrics {
            1.0
        } else {
            0.0
        },
        0,
    );
    tb.add_scalar("config/conditioning_margin", conditioning_margin as f32, 0);
    tb.add_scalar("config/mtp_loss_weight", config.mtp_loss_weight as f32, 0);
    tb.add_scalar("config/mtp_max_ahead", config.mtp_max_ahead as f32, 0);
    tb.add_scalar(
        "config/conditioning_negative_count",
        conditioning_negatives.count() as f32,
        0,
    );
    tb.add_scalar(
        "config/train_dtype",
        match train_dtype {
            DType::F16 => 16.0,
            DType::BF16 => 17.0,
            _ => 32.0,
        },
        0,
    );
    tb.add_scalar("config/grad_accum", config.grad_accum_steps as f32, 0);
    tb.add_scalar("config/clip_norm", clip_norm as f32, 0);
    tb.add_scalar(
        "config/decoder_attention_query_block",
        decoder_attention_query_block as f32,
        0,
    );
    tb.add_scalar("config/attention_csa_gpu_topk", 1.0, 0);
    tb.add_scalar(
        "config/effective_batch_size",
        (config.batch_size * config.grad_accum_steps.max(1)) as f32,
        0,
    );
    if let Some(vram) = vram_tracker.sample() {
        tb.add_scalar("memory/used_mb", vram.used_mb, 0);
        tb.add_scalar("memory/free_mb", vram.free_mb, 0);
        tb.add_scalar("memory/total_mb", vram.total_mb, 0);
        tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, 0);
    }
    tb.flush();
    let async_checkpoints = env_bool("TOFY_DECODER_ASYNC_CHECKPOINTS", true);
    let decoder_checkpoint_every = env_usize("TOFY_DECODER_CHECKPOINT_EVERY", config.log_every);
    let decoder_heartbeat_every = env_usize("TOFY_DECODER_HEARTBEAT_EVERY", 10);
    let decoder_micro_heartbeat_every = env_usize("TOFY_DECODER_MICRO_HEARTBEAT_EVERY", 8);
    let mut checkpoint_writer = if async_checkpoints {
        Some(util::AsyncCheckpointWriter::new())
    } else {
        None
    };
    println!(
        "Decoder checkpointing: async={} checkpoint_every={} log_every={}",
        async_checkpoints, decoder_checkpoint_every, config.log_every
    );

    let mut best_loss = resume_state.best_aux_metric;
    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let mut conditioning_diversity_logged = false;
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let rollout_steps = env_usize("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", 1);
    let mut decoder_context_cache = DecoderContextCache::from_env();
    let build_only_conditioned_cache = config.build_conditioned_cache && config.steps == 0;
    let should_prepare_conditioned_cache =
        config.build_conditioned_cache || config.from_conditioned_cache;
    let conditioned_decoder_cache_path = if should_prepare_conditioned_cache {
        maybe_prepare_conditioned_decoder_cache(
            &data_path,
            decoder_cache_path.as_ref(),
            config.build_conditioned_cache,
            config.from_conditioned_cache && !config.build_conditioned_cache,
            &config.encoder_model_path,
            &config.world_model_path,
            &encoder_vocab_sig,
            &decoder_vocab_sig,
            &encoder,
            &context_compressor,
            &transition,
            decoder_action_label,
            encoder_vocab.pad_id,
            config.max_seq,
            config.dim,
            config.bridge_dim,
            config.num_latent_tokens,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            rollout_steps,
            train_dtype,
            &device,
        )?
    } else {
        None
    };
    if build_only_conditioned_cache {
        println!("Conditioned decoder cache build complete; skipping decoder training.");
        tb.flush();
        tb.finish()?;
        let _ = vram_tracker.write_summary(&run_dir, "decoder");
        return Ok(());
    }
    let mut conditioned_decoder_stream =
        if let Some(cache_path) = conditioned_decoder_cache_path.as_ref() {
            cached_decoder_stream = None;
            println!("Token cache: using conditioned decoder training cache {cache_path:?}");
            Some(ConditionedDecoderStream::with_split(
                cache_path,
                DEFAULT_STREAM_SHUFFLE_BUFFER,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                true,
            )?)
        } else {
            None
        };
    tb.add_scalar(
        "config/conditioned_decoder_cache",
        if conditioned_decoder_stream.is_some() {
            1.0
        } else {
            0.0
        },
        0,
    );
    tb.add_scalar(
        "config/decoder_context_cache_rows",
        decoder_context_cache.capacity as f32,
        0,
    );
    tb.add_scalar(
        "config/decoder_prefill_batch_rows",
        config.batch_size as f32,
        0,
    );
    println!(
        "Decoder context prefill: per-micro-batch rows={} (effective batch still {})",
        config.batch_size,
        config.batch_size * config.grad_accum_steps.max(1)
    );

    if start_step >= config.steps {
        println!(
            "Decoder resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let step_start = Instant::now();
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = decoder_batch_size_for_step(step, &config);
        let grad_accum_steps = decoder_grad_accum_for_step(step, &config);
        if config.batch_warmup_steps > 0
            && step == config.batch_warmup_steps + 1
            && (config.batch_warmup_value != config.batch_size
                || config.grad_accum_warmup_value < config.grad_accum_steps)
        {
            println!(
                "Decoder warmup complete at step {}; switching to batch={} grad_accum={} (effective={})",
                config.batch_warmup_steps,
                batch_size,
                grad_accum_steps,
                batch_size * grad_accum_steps
            );
        }
        let logging_step = step % config.log_every == 0;

        for micro_step in 0..grad_accum_steps {
            let capture_micro_metrics = logging_step && micro_step + 1 == grad_accum_steps;
            let DecoderMicroBatch {
                encoder_batch,
                decoder_batch,
                oov_rate,
                next_context_slots: cached_next_context_slots,
            } = if let Some(ref mut conditioned_stream) = conditioned_decoder_stream {
                let conditioned_batch = collect_conditioned_decoder_batch(
                    conditioned_stream,
                    batch_size.max(1),
                    config.decoder_kind,
                )?;
                let decoder_batch = conditioned_batch
                    .iter()
                    .map(|row| row.decoder.clone())
                    .collect::<Vec<_>>();
                let next_context_slots = conditioned_slots_tensor(
                    &conditioned_batch,
                    config.num_latent_tokens,
                    config.bridge_dim,
                    train_dtype,
                    &device,
                )?;
                let oov_rate = if capture_micro_metrics {
                    encoded_examples_oov_rate(&decoder_batch, decoder_vocab.unk_id)
                } else {
                    0.0
                };
                DecoderMicroBatch {
                    encoder_batch: Vec::new(),
                    decoder_batch,
                    oov_rate,
                    next_context_slots: Some(next_context_slots),
                }
            } else if let Some(ref mut cached_stream) = cached_decoder_stream {
                let cached_batch = collect_decoder_cached_batch(
                    cached_stream,
                    batch_size.max(1),
                    config.decoder_kind,
                )?;
                let encoder_batch = cached_batch
                    .iter()
                    .map(|row| row.encoder.clone())
                    .collect::<Vec<_>>();
                let decoder_batch = cached_batch
                    .iter()
                    .map(|row| row.decoder.clone())
                    .collect::<Vec<_>>();
                let oov_rate = if capture_micro_metrics {
                    encoded_examples_oov_rate(&decoder_batch, decoder_vocab.unk_id)
                } else {
                    0.0
                };
                DecoderMicroBatch {
                    encoder_batch,
                    decoder_batch,
                    oov_rate,
                    next_context_slots: None,
                }
            } else {
                let raw_stream = raw_stream
                    .as_mut()
                    .context("decoder raw stream missing without token cache")?;
                let raw_batch =
                    collect_decoder_raw_batch(raw_stream, batch_size.max(1), config.decoder_kind)?;
                let encoder_batch = encode_world_examples(&raw_batch, &encoder_vocab);
                let decoder_batch =
                    encode_world_examples_with_mode(&raw_batch, &decoder_vocab, decoder_token_mode);
                let oov_rate = if capture_micro_metrics {
                    raw_examples_oov_rate(&raw_batch, &decoder_vocab, decoder_token_mode)
                } else {
                    0.0
                };
                DecoderMicroBatch {
                    encoder_batch,
                    decoder_batch,
                    oov_rate,
                    next_context_slots: None,
                }
            };
            let micro_next_context_slots = if let Some(slots) = cached_next_context_slots {
                slots
            } else {
                decoder_next_context_slots(
                    &mut decoder_context_cache,
                    &encoder,
                    &context_compressor,
                    &transition,
                    &encoder_batch,
                    decoder_action_label,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_context_compressor,
                    rollout_steps,
                    &device,
                )?
                .detach()
            };
            let decoder_action_labels = vec![decoder_action_label; decoder_batch.len()];
            let world_latent = decoder_conditioning_adapter
                .forward_with_actions(&micro_next_context_slots, &decoder_action_labels)?;
            if !conditioning_diversity_logged && decoder_batch.len() > 1 {
                let mismatched = hard_mismatched_conditioning_latent(&world_latent)?;
                let delta_rms =
                    util::scalar_f32(&tensor_rms(&world_latent.broadcast_sub(&mismatched)?)?)?;
                let latent_rms = util::scalar_f32(&tensor_rms(&world_latent)?)?;
                let relative_delta = delta_rms / latent_rms.max(1e-8);
                println!(
                    "Decoder conditioning diversity: mismatch_delta_rms={delta_rms:.6} latent_rms={latent_rms:.6} relative={relative_delta:.6}"
                );
                tb.add_scalar("metrics/conditioning_mismatch_delta_rms", delta_rms, step);
                tb.add_scalar(
                    "metrics/conditioning_mismatch_relative",
                    relative_delta,
                    step,
                );
                conditioning_diversity_logged = true;
            }

            let (dec_input, dec_target, loss_mask) =
                make_decoder_batch_from_slice_with_prompt_dropout(
                    &decoder_batch,
                    config.max_seq,
                    decoder_vocab.pad_id,
                    decoder_vocab.unk_id,
                    prompt_dropout,
                    &device,
                )?;
            let decoder_masks = decoder_loss_masks_from_examples(
                &decoder_batch,
                config.max_seq,
                decoder_vocab.pad_id,
                &decoder_vocab,
                &device,
            )?;

            let logits = decoder.forward(&dec_input, &world_latent)?;
            let token_loss =
                masked_weighted_cross_entropy(&logits, &dec_target, &decoder_masks.importance)?;
            let mut loss = token_loss.clone();
            if config.syntax_loss_weight > 0.0 {
                let syntax_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &decoder_masks.syntax)?;
                loss = loss.broadcast_add(&syntax_loss.affine(config.syntax_loss_weight, 0.0)?)?;
            }
            if config.signature_loss_weight > 0.0 {
                let signature_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &decoder_masks.signature)?;
                loss =
                    loss.broadcast_add(&signature_loss.affine(config.signature_loss_weight, 0.0)?)?;
            }
            if config.structure_loss_weight > 0.0 {
                let structure_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &decoder_masks.structure)?;
                loss =
                    loss.broadcast_add(&structure_loss.affine(config.structure_loss_weight, 0.0)?)?;
            }
            let format_loss = if format_loss_weight > 0.0 {
                forbidden_output_probability_loss(&logits, &loss_mask, &decoder_vocab)?
            } else {
                token_loss.affine(0.0, 0.0)?
            };
            if format_loss_weight > 0.0 {
                loss = loss.broadcast_add(&format_loss.affine(format_loss_weight, 0.0)?)?;
            }
            let conditioning_loss = if conditioning_loss_weight > 0.0 {
                let conditioned_loss =
                    masked_label_smoothed_cross_entropy(&logits, &dec_target, &loss_mask)?;
                let mut margin_sum = None;
                let mut negative_count = 0usize;
                if conditioning_negatives.zero {
                    let zero_world_latent = world_latent.affine(0.0, 0.0)?;
                    let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
                    let ablated_loss = masked_label_smoothed_cross_entropy(
                        &ablated_logits,
                        &dec_target,
                        &loss_mask,
                    )?;
                    margin_sum = Some(add_conditioning_margin_loss(
                        margin_sum,
                        &conditioned_loss,
                        &ablated_loss,
                        conditioning_margin,
                    )?);
                    negative_count += 1;
                }
                if conditioning_negatives.shuffle {
                    let shuffled_world_latent = shuffled_conditioning_latent(&world_latent)?;
                    let shuffled_logits = decoder.forward(&dec_input, &shuffled_world_latent)?;
                    let shuffled_loss = masked_label_smoothed_cross_entropy(
                        &shuffled_logits,
                        &dec_target,
                        &loss_mask,
                    )?;
                    margin_sum = Some(add_conditioning_margin_loss(
                        margin_sum,
                        &conditioned_loss,
                        &shuffled_loss,
                        conditioning_margin,
                    )?);
                    negative_count += 1;
                }
                if conditioning_negatives.hard {
                    let hard_mismatch_world_latent =
                        hard_mismatched_conditioning_latent(&world_latent)?;
                    let hard_mismatch_logits =
                        decoder.forward(&dec_input, &hard_mismatch_world_latent)?;
                    let hard_mismatch_loss = masked_label_smoothed_cross_entropy(
                        &hard_mismatch_logits,
                        &dec_target,
                        &loss_mask,
                    )?;
                    margin_sum = Some(add_conditioning_margin_loss(
                        margin_sum,
                        &conditioned_loss,
                        &hard_mismatch_loss,
                        conditioning_margin,
                    )?);
                    negative_count += 1;
                }
                margin_sum
                    .unwrap_or(conditioned_loss.affine(0.0, 0.0)?)
                    .affine(1.0 / negative_count.max(1) as f64, 0.0)?
            } else {
                token_loss.affine(0.0, 0.0)?
            };
            if conditioning_loss_weight > 0.0 {
                loss =
                    loss.broadcast_add(&conditioning_loss.affine(conditioning_loss_weight, 0.0)?)?;
            }

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;
            if decoder_micro_heartbeat_every > 0
                && (step <= start_step + 2 || logging_step)
                && (micro_step + 1 == grad_accum_steps
                    || (micro_step + 1) % decoder_micro_heartbeat_every == 0)
            {
                println!(
                    "decoder progress step {}/{} micro {}/{} elapsed {:.1}s",
                    step,
                    config.steps,
                    micro_step + 1,
                    grad_accum_steps,
                    step_start.elapsed().as_secs_f32()
                );
            }

            if capture_micro_metrics {
                let metric_logits = logits.detach();
                let metric_world_latent = world_latent.detach();
                let loss_val = util::scalar_f32(&loss)?;
                let raw_loss = masked_cross_entropy(&metric_logits, &dec_target, &loss_mask)?;
                let raw_loss_val = util::scalar_f32(&raw_loss)?;
                let (ablated_loss_val, shuffled_loss_val, hard_mismatch_loss_val) =
                    if compute_conditioning_metrics {
                        let zero_world_latent = metric_world_latent.affine(0.0, 0.0)?;
                        let ablated_logits =
                            decoder.forward(&dec_input, &zero_world_latent)?.detach();
                        let ablated_loss =
                            masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
                        let shuffled_world_latent =
                            shuffled_conditioning_latent(&metric_world_latent)?;
                        let shuffled_logits = decoder
                            .forward(&dec_input, &shuffled_world_latent)?
                            .detach();
                        let shuffled_loss =
                            masked_cross_entropy(&shuffled_logits, &dec_target, &loss_mask)?;
                        let hard_mismatch_world_latent =
                            hard_mismatched_conditioning_latent(&metric_world_latent)?;
                        let hard_mismatch_logits = decoder
                            .forward(&dec_input, &hard_mismatch_world_latent)?
                            .detach();
                        let hard_mismatch_loss =
                            masked_cross_entropy(&hard_mismatch_logits, &dec_target, &loss_mask)?;
                        (
                            util::scalar_f32(&ablated_loss)?,
                            util::scalar_f32(&shuffled_loss)?,
                            util::scalar_f32(&hard_mismatch_loss)?,
                        )
                    } else {
                        (raw_loss_val, raw_loss_val, raw_loss_val)
                    };
                let syntax_loss = masked_weighted_cross_entropy(
                    &metric_logits,
                    &dec_target,
                    &decoder_masks.syntax,
                )?;
                let signature_loss = masked_weighted_cross_entropy(
                    &metric_logits,
                    &dec_target,
                    &decoder_masks.signature,
                )?;
                let structure_loss = masked_weighted_cross_entropy(
                    &metric_logits,
                    &dec_target,
                    &decoder_masks.structure,
                )?;
                let syntax_loss_val = util::scalar_f32(&syntax_loss)?;
                let signature_loss_val = util::scalar_f32(&signature_loss)?;
                let structure_loss_val = util::scalar_f32(&structure_loss)?;
                let active_tokens = util::scalar_f32(&loss_mask.sum_all()?)?;
                let (loss_rows, _) = loss_mask.dims2()?;
                let total_tokens = (loss_rows.max(1) * config.max_seq * 2) as f32;
                let active_frac = active_tokens / total_tokens.max(1.0);
                let perplexity = perplexity_from_nll(raw_loss_val);
                let gains = decoder_conditioning_gains(
                    raw_loss_val,
                    ablated_loss_val,
                    shuffled_loss_val,
                    hard_mismatch_loss_val,
                    compute_conditioning_metrics,
                );
                let world_rms = util::scalar_f32(&tensor_rms(&metric_world_latent)?)?;
                let prediction_metrics = decoder_prediction_metrics(
                    &metric_logits,
                    &dec_target,
                    &loss_mask,
                    &decoder_vocab,
                )?;
                log_snapshot = Some(DecoderLogSnapshot {
                    metrics: DecoderBatchMetrics {
                        loss: loss_val,
                        raw_loss: raw_loss_val,
                        ablated_loss: ablated_loss_val,
                        conditioning_gain: gains.conditioning_gain,
                        zero_gain: gains.zero_gain,
                        shuffled_loss: shuffled_loss_val,
                        shuffle_gain: gains.shuffle_gain,
                        hard_negative_gain: gains.hard_negative_gain,
                        syntax_loss: syntax_loss_val,
                        signature_loss: signature_loss_val,
                        structure_loss: structure_loss_val,
                        perplexity,
                        active_tokens,
                        active_frac,
                        world_rms,
                        oov_rate,
                        token_accuracy: prediction_metrics.token_accuracy,
                        identifier_accuracy: prediction_metrics.identifier_accuracy,
                        delimiter_balance_rate: prediction_metrics.delimiter_balance_rate,
                        syntax_token_accuracy: prediction_metrics.syntax_token_accuracy,
                        function_skeleton_rate: prediction_metrics.function_skeleton_rate,
                        signature_token_accuracy: prediction_metrics.signature_token_accuracy,
                        signature_exact_rate: prediction_metrics.signature_exact_rate,
                        function_name_token_accuracy: prediction_metrics
                            .function_name_token_accuracy,
                        function_name_exact_rate: prediction_metrics.function_name_exact_rate,
                    },
                    hard_mismatch_loss_val,
                    conditioning_loss_val: if conditioning_loss_weight > 0.0 {
                        util::scalar_f32(&conditioning_loss)?
                    } else {
                        0.0
                    },
                    format_loss_val: if format_loss_weight > 0.0 {
                        util::scalar_f32(&format_loss)?
                    } else {
                        0.0
                    },
                    mtp_loss_val: 0.0,
                });
            }
        }

        opt.set_learning_rate(util::scheduled_lr(config.lr, step, config.steps));
        let grad_norm = util::clip_accumulated_gradients_device(
            &mut accumulated_grads,
            &train_vars,
            clip_norm,
        )?;
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;
        if decoder_heartbeat_every > 0
            && step % config.log_every != 0
            && (step <= start_step + 5 || step % decoder_heartbeat_every == 0)
        {
            println!(
                "decoder step {}/{} complete batch={} grad_accum={} effective={} elapsed {:.1}s cache_hit {:.1}%",
                step,
                config.steps,
                batch_size,
                grad_accum_steps,
                batch_size * grad_accum_steps,
                step_start.elapsed().as_secs_f32(),
                decoder_context_cache.hit_rate() * 100.0
            );
        }

        if step % config.log_every == 0 {
            if let Some(grad_norm) = grad_norm.as_ref() {
                tb.add_scalar("grad/global_norm", util::scalar_f32(grad_norm)?, step);
            }
            let DecoderLogSnapshot {
                metrics: train_metrics,
                hard_mismatch_loss_val,
                conditioning_loss_val,
                format_loss_val,
                mtp_loss_val,
            } = log_snapshot.context("decoder grad accumulation produced no log snapshot")?;
            let loss_val = train_metrics.loss;
            let raw_loss_val = train_metrics.raw_loss;
            let ablated_loss_val = train_metrics.ablated_loss;
            let conditioning_gain = train_metrics.conditioning_gain;
            let shuffled_loss_val = train_metrics.shuffled_loss;
            let zero_gain = train_metrics.zero_gain;
            let shuffle_gain = train_metrics.shuffle_gain;
            let hard_negative_gain = train_metrics.hard_negative_gain;
            let syntax_loss_val = train_metrics.syntax_loss;
            let signature_loss_val = train_metrics.signature_loss;
            let structure_loss_val = train_metrics.structure_loss;
            let perplexity = train_metrics.perplexity;
            let active_tokens = train_metrics.active_tokens;
            let active_frac = train_metrics.active_frac;
            let world_rms = train_metrics.world_rms;
            let oov_rate = train_metrics.oov_rate;
            let token_accuracy = train_metrics.token_accuracy;
            let identifier_accuracy = train_metrics.identifier_accuracy;
            let delimiter_balance_rate = train_metrics.delimiter_balance_rate;
            let syntax_token_accuracy = train_metrics.syntax_token_accuracy;
            let function_skeleton_rate = train_metrics.function_skeleton_rate;
            let signature_token_accuracy = train_metrics.signature_token_accuracy;
            let signature_exact_rate = train_metrics.signature_exact_rate;
            let function_name_token_accuracy = train_metrics.function_name_token_accuracy;
            let function_name_exact_rate = train_metrics.function_name_exact_rate;

            tb.add_scalar("loss/objective", loss_val, step);
            tb.add_scalar("loss/token_nll", raw_loss_val, step);
            tb.add_scalar("loss/ablated_token_nll", ablated_loss_val, step);
            tb.add_scalar("loss/shuffled_token_nll", shuffled_loss_val, step);
            tb.add_scalar("loss/hard_mismatch_token_nll", hard_mismatch_loss_val, step);
            tb.add_scalar("loss/conditioning_margin", conditioning_loss_val, step);
            tb.add_scalar("loss/mtp", mtp_loss_val, step);
            tb.add_scalar("loss/format_forbidden_prob", format_loss_val, step);
            tb.add_scalar("loss/syntax_ce", syntax_loss_val, step);
            tb.add_scalar("loss/signature_ce", signature_loss_val, step);
            tb.add_scalar("loss/structure_ce", structure_loss_val, step);
            tb.add_scalar("metrics/perplexity", perplexity, step);
            tb.add_scalar("metrics/active_tokens", active_tokens, step);
            tb.add_scalar("metrics/active_frac", active_frac, step);
            tb.add_scalar("metrics/world_latent_rms", world_rms, step);
            tb.add_scalar("metrics/conditioning_gain", conditioning_gain, step);
            tb.add_scalar("metrics/zero_gain", zero_gain, step);
            tb.add_scalar("metrics/shuffle_gain", shuffle_gain, step);
            tb.add_scalar("metrics/hard_negative_gain", hard_negative_gain, step);
            tb.add_scalar("metrics/oov_rate", oov_rate, step);
            tb.add_scalar(
                "metrics/decoder_context_cache_hit_rate",
                decoder_context_cache.hit_rate(),
                step,
            );
            tb.add_scalar("metrics/token_accuracy", token_accuracy, step);
            tb.add_scalar("metrics/identifier_accuracy", identifier_accuracy, step);
            tb.add_scalar("metrics/syntax_token_accuracy", syntax_token_accuracy, step);
            tb.add_scalar(
                "metrics/signature_token_accuracy",
                signature_token_accuracy,
                step,
            );
            tb.add_scalar("metrics/signature_exact_rate", signature_exact_rate, step);
            tb.add_scalar(
                "metrics/function_name_token_accuracy",
                function_name_token_accuracy,
                step,
            );
            tb.add_scalar(
                "metrics/function_name_exact_rate",
                function_name_exact_rate,
                step,
            );
            tb.add_scalar(
                "metrics/function_skeleton_rate",
                function_skeleton_rate,
                step,
            );
            tb.add_scalar(
                "metrics/delimiter_balance_rate",
                delimiter_balance_rate,
                step,
            );
            let mut memory_note = String::new();
            if let Some(vram) = vram_tracker.sample() {
                tb.add_scalar("memory/used_mb", vram.used_mb, step);
                tb.add_scalar("memory/free_mb", vram.free_mb, step);
                tb.add_scalar("memory/total_mb", vram.total_mb, step);
                tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, step);
                memory_note = format!(
                    " vram {:.0}/{:.0}MB peak {:.0}MB",
                    vram.used_mb, vram.total_mb, vram.peak_used_mb
                );
            }
            let train_reward_proxy = decoder_reward_proxy(&train_metrics);
            tb.add_scalar("reward/proxy", train_reward_proxy, step);
            let mut selection_metric = decoder_selection_score(
                &train_metrics,
                config.syntax_loss_weight,
                config.signature_loss_weight,
                config.structure_loss_weight,
            );
            let mut checkpoint_nll = raw_loss_val;
            if val_stream.is_some() || cached_decoder_val_stream.is_some() {
                let eval_batch_size = std::env::var("TOFY_DECODER_EVAL_BATCH")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| config.batch_size.min(8))
                    .max(1);
                let val_metrics = if let Some(ref mut cached_stream) = cached_decoder_val_stream {
                    let cached_batch = collect_decoder_cached_batch(
                        cached_stream,
                        eval_batch_size,
                        config.decoder_kind,
                    )?;
                    evaluate_decoder_cached_batch(
                        &cached_batch,
                        &encoder_vocab,
                        &decoder_vocab,
                        &encoder,
                        &context_compressor,
                        &transition,
                        &decoder_conditioning_adapter,
                        &decoder,
                        config.decoder_kind,
                        decoder_action_label,
                        compute_conditioning_metrics,
                        config.max_seq,
                        &device,
                    )?
                } else {
                    let val_stream = val_stream
                        .as_mut()
                        .context("decoder validation stream missing")?;
                    let val_raw_batch = collect_decoder_raw_batch(
                        val_stream,
                        eval_batch_size,
                        config.decoder_kind,
                    )?;
                    evaluate_decoder_batch(
                        &val_raw_batch,
                        &encoder_vocab,
                        &decoder_vocab,
                        &encoder,
                        &context_compressor,
                        &transition,
                        &decoder_conditioning_adapter,
                        &decoder,
                        config.decoder_kind,
                        decoder_action_label,
                        compute_conditioning_metrics,
                        config.max_seq,
                        &device,
                    )?
                };
                selection_metric = decoder_selection_score(
                    &val_metrics,
                    config.syntax_loss_weight,
                    config.signature_loss_weight,
                    config.structure_loss_weight,
                );
                checkpoint_nll = val_metrics.raw_loss;
                let val_reward_proxy = decoder_reward_proxy(&val_metrics);
                best_loss = best_loss.min(val_metrics.loss);
                tb.add_scalar("val/objective", val_metrics.loss, step);
                tb.add_scalar("val/token_nll", val_metrics.raw_loss, step);
                tb.add_scalar("val/ablated_token_nll", val_metrics.ablated_loss, step);
                tb.add_scalar("val/shuffled_token_nll", val_metrics.shuffled_loss, step);
                tb.add_scalar("val/syntax_ce", val_metrics.syntax_loss, step);
                tb.add_scalar("val/signature_ce", val_metrics.signature_loss, step);
                tb.add_scalar("val/structure_ce", val_metrics.structure_loss, step);
                tb.add_scalar("val/perplexity", val_metrics.perplexity, step);
                tb.add_scalar("val/active_tokens", val_metrics.active_tokens, step);
                tb.add_scalar("val/active_frac", val_metrics.active_frac, step);
                tb.add_scalar("val/world_latent_rms", val_metrics.world_rms, step);
                tb.add_scalar("val/conditioning_gain", val_metrics.conditioning_gain, step);
                tb.add_scalar("val/zero_gain", val_metrics.zero_gain, step);
                tb.add_scalar("val/shuffle_gain", val_metrics.shuffle_gain, step);
                tb.add_scalar(
                    "val/hard_negative_gain",
                    val_metrics.hard_negative_gain,
                    step,
                );
                tb.add_scalar("val/oov_rate", val_metrics.oov_rate, step);
                tb.add_scalar("val/token_accuracy", val_metrics.token_accuracy, step);
                tb.add_scalar(
                    "val/identifier_accuracy",
                    val_metrics.identifier_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/syntax_token_accuracy",
                    val_metrics.syntax_token_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/signature_token_accuracy",
                    val_metrics.signature_token_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/signature_exact_rate",
                    val_metrics.signature_exact_rate,
                    step,
                );
                tb.add_scalar(
                    "val/function_name_token_accuracy",
                    val_metrics.function_name_token_accuracy,
                    step,
                );
                tb.add_scalar(
                    "val/function_name_exact_rate",
                    val_metrics.function_name_exact_rate,
                    step,
                );
                tb.add_scalar(
                    "val/function_skeleton_rate",
                    val_metrics.function_skeleton_rate,
                    step,
                );
                tb.add_scalar(
                    "val/delimiter_balance_rate",
                    val_metrics.delimiter_balance_rate,
                    step,
                );
                tb.add_scalar("val/reward_proxy", val_reward_proxy, step);
            } else {
                best_loss = best_loss.min(loss_val);
            }
            tb.flush();
            let metric_improved = selection_metric < best_metric && checkpoint_nll.is_finite();
            let candidate_best_metric = if metric_improved {
                selection_metric
            } else {
                best_metric
            };
            let checkpoint_due = step % decoder_checkpoint_every == 0;
            let mut checkpoint_artifacts = Vec::new();
            if metric_improved {
                checkpoint_artifacts.push(decoder_varmap_checkpoint_artifact(
                    &decoder_varmap,
                    &decoder_path,
                )?);
            }
            let checkpoint_resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric: candidate_best_metric,
                best_aux_metric: best_loss,
                saved_checkpoint: saved_checkpoint || metric_improved,
            };
            if checkpoint_due {
                checkpoint_artifacts.push(decoder_varmap_checkpoint_artifact(
                    &decoder_varmap,
                    &train_checkpoint_path,
                )?);
                checkpoint_artifacts.push(decoder_optimizer_checkpoint_artifact(
                    &opt,
                    &optimizer_checkpoint_path,
                )?);
                checkpoint_artifacts.push(decoder_resume_checkpoint_artifact(
                    &checkpoint_resume_state,
                    &resume_state_path,
                )?);
            }
            let checkpoint_written_or_queued = save_decoder_checkpoint_job(
                checkpoint_writer.as_ref(),
                format!("decoder step {step}"),
                checkpoint_artifacts,
            )?;
            if metric_improved && checkpoint_written_or_queued {
                best_metric = selection_metric;
                saved_checkpoint = true;
            }
            if metric_improved {
                let best_note = if async_checkpoints {
                    "queued latest best"
                } else {
                    "saved best"
                };
                println!(
                    "step {}/{} objective {:.4} token_nll {:.4} ablate_nll {:.4} shuffle_nll {:.4} zero_gain {:.4} shuffle_gain {:.4} hard_gain {:.4} cond_loss {:.4} mtp {:.4} fmt {:.4} syntax_ce {:.4} sig_ce {:.4} struct_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% fn_name {:.2}% fn_name_exact {:.2}% delim {:.2}% fn_skel {:.2}% reward {:.3} sel {:.4}{} [{}]",
                    step,
                    config.steps,
                    loss_val,
                    raw_loss_val,
                    ablated_loss_val,
                    shuffled_loss_val,
                    zero_gain,
                    shuffle_gain,
                    hard_negative_gain,
                    conditioning_loss_val,
                    mtp_loss_val,
                    format_loss_val,
                    syntax_loss_val,
                    signature_loss_val,
                    structure_loss_val,
                    perplexity,
                    active_frac * 100.0,
                    oov_rate * 100.0,
                    token_accuracy * 100.0,
                    identifier_accuracy * 100.0,
                    syntax_token_accuracy * 100.0,
                    signature_token_accuracy * 100.0,
                    signature_exact_rate * 100.0,
                    function_name_token_accuracy * 100.0,
                    function_name_exact_rate * 100.0,
                    delimiter_balance_rate * 100.0,
                    function_skeleton_rate * 100.0,
                    train_reward_proxy,
                    selection_metric,
                    memory_note,
                    best_note
                );
            } else {
                println!(
                    "step {}/{} objective {:.4} token_nll {:.4} ablate_nll {:.4} shuffle_nll {:.4} zero_gain {:.4} shuffle_gain {:.4} hard_gain {:.4} cond_loss {:.4} mtp {:.4} fmt {:.4} syntax_ce {:.4} sig_ce {:.4} struct_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% fn_name {:.2}% fn_name_exact {:.2}% delim {:.2}% fn_skel {:.2}% reward {:.3} sel {:.4}{}",
                    step,
                    config.steps,
                    loss_val,
                    raw_loss_val,
                    ablated_loss_val,
                    shuffled_loss_val,
                    zero_gain,
                    shuffle_gain,
                    hard_negative_gain,
                    conditioning_loss_val,
                    mtp_loss_val,
                    format_loss_val,
                    syntax_loss_val,
                    signature_loss_val,
                    structure_loss_val,
                    perplexity,
                    active_frac * 100.0,
                    oov_rate * 100.0,
                    token_accuracy * 100.0,
                    identifier_accuracy * 100.0,
                    syntax_token_accuracy * 100.0,
                    signature_token_accuracy * 100.0,
                    signature_exact_rate * 100.0,
                    function_name_token_accuracy * 100.0,
                    function_name_exact_rate * 100.0,
                    delimiter_balance_rate * 100.0,
                    function_skeleton_rate * 100.0,
                    train_reward_proxy,
                    selection_metric,
                    memory_note
                );
            }
        }
    }

    if let Some(writer) = checkpoint_writer.as_mut() {
        writer.finish()?;
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&decoder_varmap, &decoder_path)?;
        println!(
            "No checkpoint was saved during logging; saved final decoder weights to {:?}",
            decoder_path
        );
    }
    util::save_varmap_atomic(&decoder_varmap, &train_checkpoint_path)?;
    opt.save_state(&optimizer_checkpoint_path)?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: config.steps,
        best_metric,
        best_aux_metric: best_loss,
        saved_checkpoint,
    };
    util::save_resume_state(&resume_state_path, &resume_state)?;
    tb.flush();
    tb.finish()?;
    let _ = vram_tracker.write_summary(&run_dir, "decoder");
    if saved_checkpoint {
        println!(
            "Best decoder saved to {:?} (loss {:.4})",
            decoder_path, best_loss
        );
    } else {
        println!(
            "Final decoder saved to {:?} (run finished before first logging checkpoint)",
            decoder_path
        );
    }
    CandleCrossAttnDecoder::write_metadata(
        &decoder_path,
        &decoder_vocab,
        config.decoder_kind,
        config.bridge_dim,
        config.num_latent_tokens,
        decoder_arch,
        decoder_attention,
        decoder_adapter_compress_rate,
    )?;
    println!("Decoder vocab saved to {:?}", decoder_vocab_path);
    if config.decoder_kind == DecoderKind::TextGeneralist {
        println!(
            "To use as text decoder: set JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER={:?}",
            decoder_path
        );
    } else {
        println!(
            "To use as code decoder: set JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER={:?}",
            decoder_path
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval_world(config: WorldEvalConfig) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let runtime_dtype = util::resolve_runtime_dtype(&device);
    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let data_path = resolve_data_path(&config.data_arg)?.path;
    let row_count = count_raw_world_rows(&data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let mut raw_stream = if val_row_count > 0 {
        RawWorldStream::with_split(
            &data_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            false,
        )?
    } else {
        RawWorldStream::new(&data_path)?
    };

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, runtime_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        encoder_vocab.id_to_token.len(),
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, runtime_dtype)?;

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, runtime_dtype, &device);
    let context_compressor = ContextCompressor::new(
        world_vb.pp("context_compressor"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition =
        ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
    let action_classifier_head =
        NextActionClassifier::new(world_vb.pp("next_action_classifier"), config.bridge_dim)?;
    util::load_varmap_checked(&mut world_varmap, &config.model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, runtime_dtype)?;

    println!("World-model evaluation");
    println!("model: {:?}", config.model_path);
    println!("encoder: {:?}", config.encoder_model_path);
    println!("rows: {}", row_count);
    println!("held-out eval rows: {}", val_row_count);
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={} planner_dim={} context_slots={}",
        config.eval_steps,
        config.batch_size,
        config.dim,
        config.max_seq,
        config.num_layers,
        config.num_heads,
        config.bridge_dim,
        config.num_latent_tokens
    );

    let mut n_total = 0usize;
    let mut sum_pred = 0.0f64;
    let mut sum_sigreg = 0.0f64;
    let mut sum_action_acc = 0.0f64;
    let mut sum_action_balanced_acc = 0.0f64;
    let mut sum_action_macro_f1 = 0.0f64;
    let mut sum_code_precision = 0.0f64;
    let mut sum_code_recall = 0.0f64;
    let mut sum_code_f1 = 0.0f64;
    let mut sum_code_rate = 0.0f64;
    let mut sum_pred_code_rate = 0.0f64;
    let mut sum_done_precision = 0.0f64;
    let mut sum_done_recall = 0.0f64;
    let mut sum_done_f1 = 0.0f64;
    let mut sum_done_rate = 0.0f64;
    let mut sum_pred_done_rate = 0.0f64;
    let mut batches = 0usize;
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    for _ in 0..config.eval_steps.max(1) {
        let raw_batch = raw_stream.next_batch(config.batch_size.max(1))?;
        let chunk = encode_world_examples(&raw_batch, &encoder_vocab);
        let action_labels = chunk.iter().map(|row| row.action_label).collect::<Vec<_>>();
        let (state_slots, next_slots) = context_slots_from_world_pair_sequences(
            &encoder,
            &context_compressor,
            &chunk,
            encoder_vocab.pad_id,
            config.max_seq,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            &device,
        )?;
        let pred_slots = transition.forward(&state_slots, &action_labels)?;
        let pred_loss = util::scalar_f32(&prediction_loss(&pred_slots, &next_slots)?)?;
        let sigreg_loss = util::scalar_f32(&sigreg_epps_pulley(
            &flatten_latent_slots(&pred_slots)?,
            128,
            17,
        )?)?;
        let action_logits = action_classifier_head.forward(&state_slots)?;
        let action_metrics = compute_action_metrics(&action_logits, &action_labels)?;
        n_total += chunk.len();
        sum_pred += pred_loss as f64;
        sum_sigreg += sigreg_loss as f64;
        sum_action_acc += action_metrics.accuracy as f64;
        sum_action_balanced_acc += action_metrics.balanced_accuracy as f64;
        sum_action_macro_f1 += action_metrics.macro_f1 as f64;
        sum_code_precision += action_metrics.code_precision as f64;
        sum_code_recall += action_metrics.code_recall as f64;
        sum_code_f1 += action_metrics.code_f1 as f64;
        sum_code_rate += action_metrics.code_rate as f64;
        sum_pred_code_rate += action_metrics.pred_code_rate as f64;
        sum_done_precision += action_metrics.done_precision as f64;
        sum_done_recall += action_metrics.done_recall as f64;
        sum_done_f1 += action_metrics.done_f1 as f64;
        sum_done_rate += action_metrics.done_rate as f64;
        sum_pred_done_rate += action_metrics.pred_done_rate as f64;
        batches += 1;
    }

    if n_total == 0 {
        bail!("world evaluation produced zero samples");
    }
    println!(
        "\nWorld metrics over {} samples ({} batches):",
        n_total, batches
    );
    println!(
        "  pred_mse:          {:.4}",
        sum_pred / batches.max(1) as f64
    );
    println!(
        "  pred_sigreg:       {:.4}",
        sum_sigreg / batches.max(1) as f64
    );
    println!(
        "  action_acc:        {:.4}",
        sum_action_acc / batches.max(1) as f64
    );
    println!(
        "  action_bal_acc:    {:.4}",
        sum_action_balanced_acc / batches.max(1) as f64
    );
    println!(
        "  action_macro_f1:   {:.4}",
        sum_action_macro_f1 / batches.max(1) as f64
    );
    println!(
        "  code_precision:    {:.4}",
        sum_code_precision / batches.max(1) as f64
    );
    println!(
        "  code_recall:       {:.4}",
        sum_code_recall / batches.max(1) as f64
    );
    println!(
        "  code_f1:           {:.4}",
        sum_code_f1 / batches.max(1) as f64
    );
    println!(
        "  code_rate:         {:.4}",
        sum_code_rate / batches.max(1) as f64
    );
    println!(
        "  pred_code_rate:    {:.4}",
        sum_pred_code_rate / batches.max(1) as f64
    );
    println!(
        "  done_precision:    {:.4}",
        sum_done_precision / batches.max(1) as f64
    );
    println!(
        "  done_recall:       {:.4}",
        sum_done_recall / batches.max(1) as f64
    );
    println!(
        "  done_f1:           {:.4}",
        sum_done_f1 / batches.max(1) as f64
    );
    println!(
        "  done_rate:         {:.4}",
        sum_done_rate / batches.max(1) as f64
    );
    println!(
        "  pred_done_rate:    {:.4}",
        sum_pred_done_rate / batches.max(1) as f64
    );
    Ok(())
}

/// Loaded world model + vocab for reuse (serve). Single-thread use or behind a Mutex.
pub struct AgentEngine {
    device: Device,
    _encoder_varmap: VarMap,
    _world_varmap: VarMap,
    _high_world_varmap: Option<VarMap>,
    encoder_vocab: Vocab,
    encoder: OnlineEncoder,
    context_compressor: ContextCompressor,
    transition: ActionStateTransition,
    high_world_model_path: Option<PathBuf>,
    action_sequence_encoder: Option<ActionSequenceEncoder>,
    macro_transition: Option<MacroActionStateTransition>,
    /// JEPA-style action_classifier: predicts next action from transition latent. None if checkpoint has no head.
    action_classifier_head: Option<NextActionClassifier>,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    world_rollout_steps: usize,
    high_macro_max_len: usize,
    code_decoder: Option<Arc<CandleCrossAttnDecoder>>,
    text_decoder: Option<Arc<CandleCrossAttnDecoder>>,
}

/// Returns true if the safetensors file contains any tensor whose name starts with `prefix`.
fn checkpoint_has_prefix(model_path: &Path, prefix: &str) -> bool {
    use candle_core::safetensors::MmapedSafetensors;
    let Ok(mapped) = (unsafe { MmapedSafetensors::new(model_path) }) else {
        return false;
    };
    mapped.tensors().iter().any(|(n, _)| n.starts_with(prefix))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodeRequestLanguage {
    Go,
}

fn code_request_language(prompt: &str) -> Option<CodeRequestLanguage> {
    let lower = prompt.to_ascii_lowercase();
    let explicit_go = lower.contains("return only go code")
        || lower.contains("go code")
        || lower.contains("golang")
        || lower.contains("package main");
    let go_like = explicit_go || lower.contains("func ");

    if explicit_go || go_like {
        Some(CodeRequestLanguage::Go)
    } else {
        None
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
struct LeWmPlanningConfig {
    enabled: bool,
    horizon: usize,
    max_candidates: usize,
    goal_weight: f32,
    route_weight: f32,
    prior_weight: f32,
    smoothness_weight: f32,
    done_penalty: f32,
}

#[derive(Debug, Clone)]
struct HwmPlanningConfig {
    high_horizon: usize,
    low_horizon: usize,
    macro_candidates: usize,
    subgoal_weight: f32,
}

#[derive(Debug, Clone)]
struct LatentReasoningConfig {
    enabled: bool,
    min_steps: usize,
    max_steps: usize,
    patience: usize,
    alpha: f64,
    goal_weight: f32,
    route_weight: f32,
    stability_weight: f32,
    improvement_eps: f32,
}

impl LatentReasoningConfig {
    fn from_env(prompt: &str, action: crate::tasks::orchestrator::Action) -> Self {
        let code_like = action == crate::tasks::orchestrator::Action::Code
            || action == crate::tasks::orchestrator::Action::FetchDocs
            || code_request_language(prompt).is_some();
        let default_max = if code_like { 8usize } else { 3usize };
        let default_min = if code_like { 2usize } else { 1usize };
        let max_steps = std::env::var("TOFY_LATENT_REASONING_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_max)
            .clamp(1usize, 64usize);
        Self {
            enabled: env_flag("TOFY_LATENT_REASONING", true),
            min_steps: std::env::var("TOFY_LATENT_REASONING_MIN_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default_min)
                .clamp(1usize, max_steps),
            max_steps,
            patience: std::env::var("TOFY_LATENT_REASONING_PATIENCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2usize)
                .max(1),
            alpha: std::env::var("TOFY_LATENT_REASONING_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35f64)
                .clamp(0.01, 1.0),
            goal_weight: std::env::var("TOFY_LATENT_REASONING_GOAL_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0f32),
            route_weight: std::env::var("TOFY_LATENT_REASONING_ROUTE_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25f32),
            stability_weight: std::env::var("TOFY_LATENT_REASONING_STABILITY_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05f32),
            improvement_eps: std::env::var("TOFY_LATENT_REASONING_EPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-4f32)
                .max(0.0),
        }
    }
}

impl HwmPlanningConfig {
    fn from_env() -> Self {
        Self {
            high_horizon: std::env::var("TOFY_HWM_HIGH_HORIZON")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3usize)
                .clamp(1, 8),
            low_horizon: std::env::var("TOFY_HWM_LOW_HORIZON")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2usize)
                .clamp(1, 6),
            macro_candidates: std::env::var("TOFY_HWM_MACRO_CANDIDATES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64usize)
                .max(1),
            subgoal_weight: std::env::var("TOFY_HWM_SUBGOAL_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0f32),
        }
    }
}

impl LeWmPlanningConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_flag("TOFY_LEWM_PLANNING", true),
            horizon: std::env::var("TOFY_LEWM_PLANNING_HORIZON")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3usize)
                .clamp(1, 6),
            max_candidates: std::env::var("TOFY_LEWM_PLANNING_CANDIDATES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128usize)
                .max(1),
            goal_weight: std::env::var("TOFY_LEWM_GOAL_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0f32),
            route_weight: std::env::var("TOFY_LEWM_ROUTE_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35f32),
            prior_weight: std::env::var("TOFY_LEWM_PRIOR_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25f32),
            smoothness_weight: std::env::var("TOFY_LEWM_SMOOTHNESS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.03f32),
            done_penalty: std::env::var("TOFY_LEWM_DONE_PENALTY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.0f32),
        }
    }
}

#[derive(Debug, Clone)]
struct LeWmPlan {
    actions: Vec<crate::tasks::orchestrator::Action>,
    first_action: crate::tasks::orchestrator::Action,
    planned_slots: Tensor,
    score: f32,
}

fn lewm_action_from_id(id: usize) -> crate::tasks::orchestrator::Action {
    crate::tasks::orchestrator::action_from_index(id)
}

fn lewm_prompt_goal(prompt: &str, action: crate::tasks::orchestrator::Action) -> String {
    format!(
        "<lewm_goal>\nUser request:\n{prompt}\n\nDesired next assistant state: {}\n</lewm_goal>",
        action.as_str()
    )
}

fn softmax_probability(row: &[f32], idx: usize) -> f32 {
    if row.is_empty() {
        return 1.0;
    }
    let max = row
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    let denom = row.iter().map(|v| (*v - max).exp()).sum::<f32>().max(1e-8);
    row.get(idx)
        .map(|v| (*v - max).exp() / denom)
        .unwrap_or(1e-8)
}

fn enumerate_action_sequences(
    horizon: usize,
    max_candidates: usize,
) -> Vec<Vec<crate::tasks::orchestrator::Action>> {
    let horizon = horizon.max(1);
    let base = crate::model::action_classifier_head::NUM_ACTIONS.max(1);
    let total = (0..horizon).fold(1usize, |acc, _| acc.saturating_mul(base));
    let count = total.min(max_candidates.max(1));
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut encoded = if count >= total {
            i
        } else if count == 1 {
            0
        } else {
            i.saturating_mul(total.saturating_sub(1)) / count.saturating_sub(1)
        };
        let mut sequence = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            sequence.push(lewm_action_from_id(encoded % base));
            encoded /= base;
        }
        out.push(sequence);
    }
    out
}

fn balanced_braces(text: &str) -> bool {
    let mut round = 0i32;
    let mut square = 0i32;
    let mut curly = 0i32;
    for ch in text.chars() {
        match ch {
            '(' => round += 1,
            ')' => round -= 1,
            '[' => square += 1,
            ']' => square -= 1,
            '{' => curly += 1,
            '}' => curly -= 1,
            _ => {}
        }
        if round < 0 || square < 0 || curly < 0 {
            return false;
        }
    }
    round == 0 && square == 0 && curly == 0
}

fn output_needs_code_repair(prompt: &str, output: &str) -> bool {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    let prompt_lower = prompt.to_ascii_lowercase();
    for marker in [
        "here is",
        "here's",
        "the code",
        "explanation",
        "compiler feedback",
    ] {
        if lower.contains(marker) {
            return true;
        }
    }
    let angle_noise = trimmed.matches('<').count() + trimmed.matches('>').count();
    let expected_keyword = match code_request_language(prompt) {
        Some(CodeRequestLanguage::Go) if prompt_lower.contains("func") => Some("func "),
        _ => None,
    };
    if let Some(keyword) = expected_keyword {
        if !lower.contains(keyword) {
            return true;
        }
    }
    if !balanced_braces(trimmed) {
        return true;
    }
    let has_code_keyword = lower.contains("func ");
    angle_noise >= 6 && !has_code_keyword
}

fn maybe_repair_code_output(
    decoder: &dyn LocalDecoderRuntime,
    prompt: &str,
    action: &str,
    cond_vec: &[f32],
    chunk_tokens: usize,
    temperature: Option<f32>,
    initial: String,
) -> String {
    let repair_passes = std::env::var("TOFY_CODE_REPAIR_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);
    if matches!(code_request_language(prompt), Some(CodeRequestLanguage::Go)) {
        if let Some(go_feedback) = go_compile_feedback_from_env() {
            return maybe_repair_go_output_with_compile(
                decoder,
                prompt,
                action,
                cond_vec,
                chunk_tokens,
                temperature,
                repair_passes,
                initial,
                go_feedback,
            );
        }
    }
    let mut assistant_content = initial;
    for _ in 0..repair_passes {
        if !output_needs_code_repair(prompt, &assistant_content) {
            break;
        }
        let repair_prompt = build_code_repair_prompt(
            prompt,
            &assistant_content,
            "No compiler feedback was captured; repair malformed or incomplete output.",
        );
        let repaired = match decoder.generate_with_temperature(
            &repair_prompt,
            action,
            cond_vec,
            chunk_tokens,
            temperature,
        ) {
            Ok(text) => text,
            Err(_) => break,
        };
        if repaired.trim().is_empty() || repaired.trim() == assistant_content.trim() {
            break;
        }
        assistant_content = repaired;
    }
    assistant_content
}

#[derive(Clone, Debug)]
struct GoCompileCandidate {
    text: String,
    compile_feedback: String,
    compile_ok: bool,
    heuristic_ok: bool,
    repair_depth: usize,
}

fn go_compile_feedback_from_env() -> Option<&'static GoCompileFeedback> {
    static FEEDBACK: OnceLock<Option<GoCompileFeedback>> = OnceLock::new();
    FEEDBACK
        .get_or_init(|| {
            let go_bin =
                std::env::var("TOFY_GO_CODE_REPAIR_BIN").unwrap_or_else(|_| "go".to_string());
            let timeout_sec = std::env::var("TOFY_GO_CODE_REPAIR_TIMEOUT_SEC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6.0f64)
                .max(1.0);
            let version = go_version_string(&go_bin).ok()?;
            GoCompileFeedback::new(&go_bin, &version, timeout_sec).ok()
        })
        .as_ref()
}

#[allow(clippy::too_many_arguments)]
fn maybe_repair_go_output_with_compile(
    decoder: &dyn LocalDecoderRuntime,
    prompt: &str,
    action: &str,
    cond_vec: &[f32],
    chunk_tokens: usize,
    temperature: Option<f32>,
    repair_passes: usize,
    initial: String,
    go_feedback: &GoCompileFeedback,
) -> String {
    let candidate_count = std::env::var("TOFY_GO_COMPILE_CANDIDATES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4usize)
        .max(1);
    let search_temperature = go_compile_search_temperature(temperature, candidate_count);
    let mut best = evaluate_go_compile_candidate(go_feedback, prompt, initial, 0);
    if best.compile_ok {
        return best.text;
    }

    for _ in 1..candidate_count {
        let extra = match decoder.generate_with_temperature(
            prompt,
            action,
            cond_vec,
            chunk_tokens,
            search_temperature,
        ) {
            Ok(text) => text,
            Err(_) => break,
        };
        let candidate = evaluate_go_compile_candidate(go_feedback, prompt, extra, 0);
        best = select_better_go_compile_candidate(best, candidate);
        if best.compile_ok {
            return best.text;
        }
    }

    for repair_depth in 1..=repair_passes {
        let repair_prompt = build_code_repair_prompt(prompt, &best.text, &best.compile_feedback);
        let repaired = match decoder.generate_with_temperature(
            &repair_prompt,
            action,
            cond_vec,
            chunk_tokens,
            search_temperature,
        ) {
            Ok(text) => text,
            Err(_) => break,
        };
        if repaired.trim().is_empty() || repaired.trim() == best.text.trim() {
            break;
        }
        let candidate = evaluate_go_compile_candidate(go_feedback, prompt, repaired, repair_depth);
        best = select_better_go_compile_candidate(best, candidate);
        if best.compile_ok {
            return best.text;
        }
    }

    best.text
}

fn go_compile_search_temperature(temperature: Option<f32>, candidate_count: usize) -> Option<f32> {
    if candidate_count <= 1 {
        return temperature;
    }
    if let Some(value) = std::env::var("TOFY_GO_COMPILE_SEARCH_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return Some(value);
    }
    match temperature {
        Some(value) if value > 0.0 => Some(value),
        _ => Some(0.35),
    }
}

fn evaluate_go_compile_candidate(
    go_feedback: &GoCompileFeedback,
    prompt: &str,
    text: String,
    repair_depth: usize,
) -> GoCompileCandidate {
    let heuristic_ok = !output_needs_code_repair(prompt, &text);
    let compile_feedback = if text.trim().is_empty() {
        "empty model output".to_string()
    } else {
        go_feedback
            .compile(&text)
            .unwrap_or_else(|err| format!("compile harness error: {err}"))
    };
    GoCompileCandidate {
        text,
        compile_ok: compile_feedback.trim().is_empty(),
        heuristic_ok,
        compile_feedback,
        repair_depth,
    }
}

fn select_better_go_compile_candidate(
    current: GoCompileCandidate,
    challenger: GoCompileCandidate,
) -> GoCompileCandidate {
    if go_compile_candidate_rank(&challenger) > go_compile_candidate_rank(&current) {
        challenger
    } else {
        current
    }
}

fn go_compile_candidate_rank(candidate: &GoCompileCandidate) -> i32 {
    let compile_lines = candidate
        .compile_feedback
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count() as i32;
    let text = candidate.text.trim();
    let balanced = i32::from(balanced_braces(text));
    let has_func = i32::from(text.to_ascii_lowercase().contains("func "));
    1024 * i32::from(candidate.compile_ok)
        + 64 * i32::from(candidate.heuristic_ok)
        + 16 * balanced
        + 8 * has_func
        - 2 * compile_lines
        - candidate.repair_depth as i32
}

fn extract_go_prompt_declarations(prompt: &str) -> Vec<String> {
    let mut declarations = Vec::new();
    for line in prompt.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("func ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("var ")
        {
            declarations.push(trimmed.to_string());
        }
    }
    declarations
}

fn build_code_repair_prompt(prompt: &str, attempt: &str, compiler_feedback: &str) -> String {
    let language = code_request_language(prompt).unwrap_or(CodeRequestLanguage::Go);
    let (language_name, fence) = match language {
        CodeRequestLanguage::Go => ("Go", "go"),
    };
    let declarations = extract_go_prompt_declarations(prompt);
    let declarations_block = if declarations.is_empty() {
        String::new()
    } else {
        format!(
            "Required declarations:\n{}\n\n",
            declarations
                .iter()
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    format!(
        "Return only corrected {language_name} code.\nFix the previous attempt using the compiler feedback.\n\nOriginal request:\n{prompt}\n\n{declarations_block}Previous attempt:\n```{fence}\n{attempt}\n```\n\nCompiler feedback:\n{compiler_feedback}\n\nRules:\n- Keep the exact requested function name and signature.\n- Keep any required type declarations.\n- Return only compilable {language_name} code.\n- Do not add explanation.\n"
    )
}

impl AgentEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        encoder_model_path: &Path,
        encoder_vocab_path: &Path,
        world_model_path: &Path,
        high_world_model_path: Option<&Path>,
        dim: usize,
        max_seq: usize,
        num_layers: usize,
        num_heads: usize,
        bridge_dim: usize,
        num_latent_tokens: usize,
    ) -> Result<Self> {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        };
        let runtime_dtype = util::resolve_runtime_dtype(&device);
        let encoder_vocab = load_vocab_from_file(encoder_vocab_path)?;

        let mut encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, runtime_dtype, &device);
        let encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            dim,
            num_layers,
            num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, runtime_dtype)?;

        let mut world_varmap = VarMap::new();
        let world_vb = VarBuilder::from_varmap(&world_varmap, runtime_dtype, &device);
        let context_compressor = ContextCompressor::new(
            world_vb.pp("context_compressor"),
            dim,
            bridge_dim,
            num_latent_tokens,
        )?;
        let transition =
            ActionStateTransition::new(world_vb.pp("action_state_transition"), bridge_dim)?;
        let action_classifier_head =
            if checkpoint_has_prefix(world_model_path, "next_action_classifier.") {
                Some(NextActionClassifier::new(
                    world_vb.pp("next_action_classifier"),
                    bridge_dim,
                )?)
            } else {
                None
            };
        util::load_varmap_checked(&mut world_varmap, world_model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, runtime_dtype)?;
        let explicit_high_world_model_path = high_world_model_path.map(Path::to_path_buf);
        let env_high_world_model_path = std::env::var("TOFY_HIGH_WORLD_MODEL")
            .ok()
            .map(PathBuf::from);
        let high_world_model_path = explicit_high_world_model_path
            .clone()
            .or_else(|| env_high_world_model_path.clone())
            .unwrap_or_else(|| default_high_world_path(world_model_path));
        if !high_world_model_path.exists()
            && (explicit_high_world_model_path.is_some() || env_high_world_model_path.is_some())
        {
            bail!(
                "high-world checkpoint not found at {:?}",
                high_world_model_path
            );
        }
        let high_world_model_path = high_world_model_path
            .exists()
            .then_some(high_world_model_path);
        let high_macro_max_len = env_usize("TOFY_HWM_MACRO_MAX_LEN", 4);
        let (high_world_varmap, action_sequence_encoder, macro_transition) =
            if let Some(path) = high_world_model_path.as_ref().filter(|path| path.exists()) {
                let mut high_varmap = VarMap::new();
                let high_vb = VarBuilder::from_varmap(&high_varmap, runtime_dtype, &device);
                let macro_encoder = ActionSequenceEncoder::new(
                    high_vb.pp("action_sequence_encoder"),
                    bridge_dim,
                    high_macro_max_len,
                )?;
                let macro_transition = MacroActionStateTransition::new(
                    high_vb.pp("macro_action_state_transition"),
                    bridge_dim,
                )?;
                util::load_varmap_checked(&mut high_varmap, path)?;
                util::cast_varmap_dtype(&mut high_varmap, runtime_dtype)?;
                (
                    Some(high_varmap),
                    Some(macro_encoder),
                    Some(macro_transition),
                )
            } else {
                (None, None, None)
            };
        let explicit_code_decoder = std::env::var("JEPA_USE_CANDLE_DECODER")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let code_decoder =
            match CandleCrossAttnDecoder::try_new_from_env_code(bridge_dim, num_latent_tokens) {
                Ok(decoder) => Some(Arc::new(decoder)),
                Err(err) if explicit_code_decoder => {
                    anyhow::bail!("explicit Candle code decoder failed to load: {err:#}")
                }
                Err(_) => None,
            };
        let text_decoder =
            CandleCrossAttnDecoder::try_new_from_env_text(bridge_dim, num_latent_tokens)
                .ok()
                .map(Arc::new);
        Ok(Self {
            device,
            _encoder_varmap: encoder_varmap,
            _world_varmap: world_varmap,
            _high_world_varmap: high_world_varmap,
            encoder_vocab,
            encoder,
            context_compressor,
            transition,
            high_world_model_path,
            action_sequence_encoder,
            macro_transition,
            action_classifier_head,
            max_seq,
            context_segments: std::env::var("TOFY_ENCODER_CONTEXT_SEGMENTS")
                .ok()
                .or_else(|| std::env::var("TOFY_WORLD_CONTEXT_SEGMENTS").ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(4usize)
                .max(1),
            recent_full_segments: std::env::var("TOFY_ENCODER_RECENT_FULL_SEGMENTS")
                .ok()
                .or_else(|| std::env::var("TOFY_WORLD_RECENT_FULL_SEGMENTS").ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            recursive_context_compressor: env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false),
            world_rollout_steps: std::env::var("TOFY_WORLD_ROLLOUT_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            high_macro_max_len,
            code_decoder,
            text_decoder,
        })
    }

    /// Encode current prompt into private context compressor slots.
    fn encode_prompt_context_compressor(&self, current_prompt: &str) -> Result<Tensor> {
        let ids = encode_text_with_vocab_mode(
            current_prompt,
            &self.encoder_vocab,
            TokenizationMode::Default,
        );
        if ids.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let token_sequences = [ids.as_slice()];
        context_slots_from_token_sequences(
            &self.encoder,
            &self.context_compressor,
            &token_sequences,
            self.encoder_vocab.pad_id,
            self.max_seq,
            self.context_segments,
            self.recent_full_segments,
            self.recursive_context_compressor,
            &self.device,
        )
    }

    fn route_action_from_state(
        &self,
        prompt: &str,
        state_slots: &Tensor,
    ) -> Result<crate::tasks::orchestrator::Action> {
        use crate::tasks::orchestrator::{
            action_from_index, decide_next_action, guard_inference_action,
        };
        if let Some(ref h) = self.action_classifier_head {
            let logits = h.forward(state_slots)?;
            let rows = crate::util::vec2_f32(&logits)?;
            let row = rows
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty action_classifier logits"))?;
            let predicted = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| action_from_index(idx))
                .unwrap_or_else(|| decide_next_action(prompt, ""));
            Ok(guard_inference_action(prompt, predicted, Some(row)))
        } else {
            Ok(decide_next_action(prompt, ""))
        }
    }

    fn lewm_goal_slots(
        &self,
        prompt: &str,
        action: crate::tasks::orchestrator::Action,
    ) -> Result<Tensor> {
        self.encode_prompt_context_compressor(&lewm_prompt_goal(prompt, action))
    }

    fn lewm_action_prior(&self, prompt: &str, action: crate::tasks::orchestrator::Action) -> f32 {
        use crate::tasks::orchestrator::{code_request_score, terminal_request_score, Action};
        let code = code_request_score(prompt);
        let terminal = terminal_request_score(prompt);
        match action {
            Action::Code => code,
            Action::FetchDocs => code - 0.2,
            Action::Done => terminal - 0.6,
            Action::TextReply => 0.8 - 0.25 * code.max(terminal),
        }
    }

    fn lewm_route_reward(
        &self,
        slots: &Tensor,
        action: crate::tasks::orchestrator::Action,
    ) -> Result<f32> {
        let Some(head) = self.action_classifier_head.as_ref() else {
            return Ok(0.0);
        };
        let logits = head.forward(slots)?;
        let rows = util::vec2_f32(&logits)?;
        let row = rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty planner route logits"))?;
        Ok(softmax_probability(row, action as usize).ln())
    }

    fn latent_reasoning_score(
        &self,
        slots: &Tensor,
        anchor_slots: &Tensor,
        goal_slots: &Tensor,
        action: crate::tasks::orchestrator::Action,
        cfg: &LatentReasoningConfig,
    ) -> Result<f32> {
        let goal_loss = util::scalar_f32(&prediction_loss(slots, goal_slots)?)?;
        let stability_loss = util::scalar_f32(&prediction_loss(slots, anchor_slots)?)?;
        let route_reward = self.lewm_route_reward(slots, action)?;
        Ok(
            cfg.goal_weight * goal_loss + cfg.stability_weight * stability_loss
                - cfg.route_weight * route_reward,
        )
    }

    fn refine_latent_for_decoder(
        &self,
        prompt: &str,
        action: crate::tasks::orchestrator::Action,
        anchor_slots: &Tensor,
    ) -> Result<Tensor> {
        let cfg = LatentReasoningConfig::from_env(prompt, action);
        if !cfg.enabled {
            return Ok(anchor_slots.clone());
        }
        let goal_slots = self.lewm_goal_slots(prompt, action)?;
        let mut current = anchor_slots.clone();
        let mut best = anchor_slots.clone();
        let mut best_score =
            self.latent_reasoning_score(&best, anchor_slots, &goal_slots, action, &cfg)?;
        let mut stale_steps = 0usize;

        for depth in 1..=cfg.max_steps {
            let proposed = self.transition.forward_one(&current, action as u32)?;
            let refined = proposed
                .affine(cfg.alpha, 0.0)?
                .broadcast_add(&anchor_slots.affine(1.0 - cfg.alpha, 0.0)?)?;
            let score =
                self.latent_reasoning_score(&refined, anchor_slots, &goal_slots, action, &cfg)?;
            if score + cfg.improvement_eps < best_score {
                best_score = score;
                best = refined.clone();
                stale_steps = 0;
            } else {
                stale_steps += 1;
            }
            current = refined;
            if depth >= cfg.min_steps && stale_steps >= cfg.patience {
                break;
            }
        }

        if std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] latent reasoning action={} score={:.4} max_steps={}",
                action.as_str(),
                best_score,
                cfg.max_steps
            );
            let _ = std::io::stderr().flush();
        }
        Ok(best)
    }

    fn lewm_score_sequence(
        &self,
        prompt: &str,
        state_slots: &Tensor,
        actions: &[crate::tasks::orchestrator::Action],
        cfg: &LeWmPlanningConfig,
    ) -> Result<LeWmPlan> {
        use crate::tasks::orchestrator::Action;
        let first_action = actions.first().copied().unwrap_or(Action::TextReply);
        let goal_slots = self.lewm_goal_slots(prompt, first_action)?;
        let mut slots = state_slots.clone();
        let mut first_step_slots: Option<Tensor> = None;
        let mut route_reward = 0.0f32;
        let mut prior_reward = 0.0f32;
        let mut smoothness = 0.0f32;
        let mut executed_actions = Vec::new();
        for (idx, action) in actions.iter().enumerate() {
            executed_actions.push(*action);
            route_reward += self.lewm_route_reward(&slots, *action)?;
            prior_reward += self.lewm_action_prior(prompt, *action);
            let next_slots = self.transition.forward_one(&slots, *action as u32)?;
            if idx == 0 {
                first_step_slots = Some(next_slots.clone());
            }
            smoothness += util::scalar_f32(&prediction_loss(&next_slots, &slots)?)?;
            slots = next_slots;
            if *action == Action::Done {
                break;
            }
        }
        let goal_loss = util::scalar_f32(&prediction_loss(&slots, &goal_slots)?)?;
        let premature_done_penalty = if first_action == Action::Done
            && crate::tasks::orchestrator::terminal_request_score(prompt) < 0.8
        {
            cfg.done_penalty
        } else {
            0.0
        };
        let denom = executed_actions.len().max(1) as f32;
        let score = cfg.goal_weight * goal_loss
            + cfg.smoothness_weight * (smoothness / denom)
            + premature_done_penalty
            - cfg.route_weight * (route_reward / denom)
            - cfg.prior_weight * (prior_reward / denom);
        Ok(LeWmPlan {
            actions: executed_actions,
            first_action,
            planned_slots: first_step_slots.unwrap_or(slots),
            score,
        })
    }

    fn hwm_score_low_sequence_to_subgoal(
        &self,
        prompt: &str,
        state_slots: &Tensor,
        subgoal_slots: &Tensor,
        actions: &[crate::tasks::orchestrator::Action],
        lewm_cfg: &LeWmPlanningConfig,
        hwm_cfg: &HwmPlanningConfig,
    ) -> Result<LeWmPlan> {
        use crate::tasks::orchestrator::Action;
        let first_action = actions.first().copied().unwrap_or(Action::TextReply);
        let mut slots = state_slots.clone();
        let mut first_step_slots: Option<Tensor> = None;
        let mut route_reward = 0.0f32;
        let mut prior_reward = 0.0f32;
        let mut smoothness = 0.0f32;
        let mut executed_actions = Vec::new();
        for (idx, action) in actions.iter().enumerate() {
            executed_actions.push(*action);
            route_reward += self.lewm_route_reward(&slots, *action)?;
            prior_reward += self.lewm_action_prior(prompt, *action);
            let next_slots = self.transition.forward_one(&slots, *action as u32)?;
            if idx == 0 {
                first_step_slots = Some(next_slots.clone());
            }
            smoothness += util::scalar_f32(&prediction_loss(&next_slots, &slots)?)?;
            slots = next_slots;
            if *action == Action::Done {
                break;
            }
        }
        let subgoal_loss = util::scalar_f32(&prediction_loss(&slots, subgoal_slots)?)?;
        let denom = executed_actions.len().max(1) as f32;
        let score = hwm_cfg.subgoal_weight * subgoal_loss
            + lewm_cfg.smoothness_weight * (smoothness / denom)
            - lewm_cfg.route_weight * (route_reward / denom)
            - lewm_cfg.prior_weight * (prior_reward / denom);
        Ok(LeWmPlan {
            actions: executed_actions,
            first_action,
            planned_slots: first_step_slots.unwrap_or(slots),
            score,
        })
    }

    fn hwm_plan_from_state(
        &self,
        prompt: &str,
        state_slots: &Tensor,
        lewm_cfg: &LeWmPlanningConfig,
    ) -> Result<Option<LeWmPlan>> {
        let hwm_cfg = HwmPlanningConfig::from_env();
        let (Some(macro_encoder), Some(macro_transition)) = (
            self.action_sequence_encoder.as_ref(),
            self.macro_transition.as_ref(),
        ) else {
            return Ok(None);
        };
        let high_candidates = enumerate_action_sequences(
            hwm_cfg.high_horizon.min(self.high_macro_max_len),
            hwm_cfg.macro_candidates,
        );
        let mut best_subgoal: Option<(Tensor, f32, crate::tasks::orchestrator::Action)> = None;
        for candidate in high_candidates {
            let first_action = candidate
                .first()
                .copied()
                .unwrap_or(crate::tasks::orchestrator::Action::TextReply);
            let goal_slots = self.lewm_goal_slots(prompt, first_action)?;
            let action_ids = vec![candidate
                .iter()
                .map(|action| *action as u32)
                .collect::<Vec<_>>()];
            let macro_action = macro_encoder.forward_from_slices(&action_ids, &self.device)?;
            let subgoal_slots = macro_transition.forward(state_slots, &macro_action)?;
            let goal_loss = util::scalar_f32(&prediction_loss(&subgoal_slots, &goal_slots)?)?;
            let prior = self.lewm_action_prior(prompt, first_action);
            let score = lewm_cfg.goal_weight * goal_loss - lewm_cfg.prior_weight * prior;
            if best_subgoal
                .as_ref()
                .map(|(_, best_score, _)| score < *best_score)
                .unwrap_or(true)
            {
                best_subgoal = Some((subgoal_slots, score, first_action));
            }
        }
        let Some((subgoal_slots, high_score, _)) = best_subgoal else {
            return Ok(None);
        };
        let low_candidates =
            enumerate_action_sequences(hwm_cfg.low_horizon, lewm_cfg.max_candidates);
        let mut best_plan: Option<LeWmPlan> = None;
        for sequence in low_candidates {
            let mut plan = self.hwm_score_low_sequence_to_subgoal(
                prompt,
                state_slots,
                &subgoal_slots,
                &sequence,
                lewm_cfg,
                &hwm_cfg,
            )?;
            plan.score += high_score;
            plan.first_action =
                crate::tasks::orchestrator::guard_inference_action(prompt, plan.first_action, None);
            if plan.first_action != sequence[0] {
                plan.actions[0] = plan.first_action;
                plan.planned_slots = self
                    .transition
                    .forward_one(state_slots, plan.first_action as u32)?;
                plan.score += 0.05;
            }
            if best_plan
                .as_ref()
                .map(|best| plan.score < best.score)
                .unwrap_or(true)
            {
                best_plan = Some(plan);
            }
        }
        Ok(best_plan)
    }

    fn lewm_plan_from_state(&self, prompt: &str, state_slots: &Tensor) -> Result<LeWmPlan> {
        let cfg = LeWmPlanningConfig::from_env();
        if !cfg.enabled {
            let action = self.route_action_from_state(prompt, state_slots)?;
            let planned_slots = rollout_transition_slots(
                &self.transition,
                state_slots,
                action as u32,
                self.world_rollout_steps,
            )?;
            return Ok(LeWmPlan {
                actions: vec![action],
                first_action: action,
                planned_slots,
                score: 0.0,
            });
        }
        if let Some(plan) = self.hwm_plan_from_state(prompt, state_slots, &cfg)? {
            if std::env::var("JEPA_DEBUG").is_ok() {
                let actions = plan
                    .actions
                    .iter()
                    .map(|action| action.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                let _ = writeln!(
                    std::io::stderr(),
                    "[tofy] hwm plan score={:.4} actions=[{}]",
                    plan.score,
                    actions
                );
                let _ = std::io::stderr().flush();
            }
            return Ok(plan);
        }
        let sequences = enumerate_action_sequences(cfg.horizon, cfg.max_candidates);
        let mut best_plan: Option<LeWmPlan> = None;
        for sequence in sequences {
            let mut plan = self.lewm_score_sequence(prompt, state_slots, &sequence, &cfg)?;
            plan.first_action =
                crate::tasks::orchestrator::guard_inference_action(prompt, plan.first_action, None);
            if plan.first_action != sequence[0] {
                plan.actions[0] = plan.first_action;
                plan.planned_slots = self
                    .transition
                    .forward_one(state_slots, plan.first_action as u32)?;
                plan.score += 0.05;
            }
            if best_plan
                .as_ref()
                .map(|best| plan.score < best.score)
                .unwrap_or(true)
            {
                best_plan = Some(plan);
            }
        }
        let plan = best_plan.context("LeWM planner produced no action candidates")?;
        if std::env::var("JEPA_DEBUG").is_ok() {
            let actions = plan
                .actions
                .iter()
                .map(|action| action.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] lewm plan score={:.4} actions=[{}]",
                plan.score,
                actions
            );
            let _ = std::io::stderr().flush();
        }
        Ok(plan)
    }

    pub fn high_world_model_path(&self) -> Option<&Path> {
        self.high_world_model_path.as_deref()
    }

    /// Build decoder + conditioning from predicted next context compressor.
    fn get_decoder_and_cond_from_context_compressor(
        &self,
        next_context_slots: &Tensor,
        action: crate::tasks::orchestrator::Action,
        ablate_conditioning: bool,
    ) -> Result<(Box<dyn LocalDecoderRuntime>, Vec<f32>)> {
        let planner_vec = util::vec1_f32(&next_context_slots.flatten_all()?)?;
        let pooled_planner = self
            .context_compressor
            .pool(next_context_slots)?
            .squeeze(0)?;
        let pooled_planner = util::vec1_f32(&pooled_planner)?;
        let (decoder, mut cond_vec): (Box<dyn LocalDecoderRuntime>, Vec<f32>) =
            if action == crate::tasks::orchestrator::Action::Code {
                if let Some(d) = self.code_decoder.as_ref() {
                    (Box::new(Arc::clone(d)), planner_vec.clone())
                } else {
                    (
                        match LlamaCppDecoder::try_new() {
                            Ok(l) => Box::new(l),
                            Err(_) => Box::new(StubLocalDecoder::new()),
                        },
                        pooled_planner.clone(),
                    )
                }
            } else {
                if let Some(d) = self.text_decoder.as_ref() {
                    (Box::new(Arc::clone(d)), planner_vec.clone())
                } else {
                    (
                        match LlamaCppDecoder::try_new() {
                            Ok(l) => Box::new(l),
                            Err(_) => Box::new(StubLocalDecoder::new()),
                        },
                        pooled_planner.clone(),
                    )
                }
            };
        if ablate_conditioning {
            cond_vec.fill(0.0);
        }
        let decoder = if RlmDecoderRuntime::should_wrap_action(action.as_str()) {
            Box::new(RlmDecoderRuntime::new(decoder)) as Box<dyn LocalDecoderRuntime>
        } else {
            decoder
        };
        Ok((decoder, cond_vec))
    }

    /// Max tokens per decoder chunk for text (brief reply) and code (block).
    const TEXT_CHUNK_TOKENS: usize = 256;
    const CODE_CHUNK_TOKENS: usize = 512;

    fn generate_agentic_with_tools(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        temperature: Option<f32>,
        registry: BashToolRegistry,
    ) -> Result<String> {
        let max_steps = env_usize("TOFY_AGENTIC_MAX_STEPS", 4).max(1);
        let step_tokens = env_usize("TOFY_AGENTIC_STEP_TOKENS", 384)
            .max(32)
            .min(max_new_tokens.max(1));
        let result_chars = env_usize("TOFY_TOOL_RESULT_CHARS", 12_000).max(512);
        let mut transcript = prompt.trim().to_string();

        for step in 1..=max_steps {
            let tool_prompt = registry.build_prompt(&transcript, step, max_steps);
            let output = self.generate_direct_with_temperature(
                &tool_prompt,
                step_tokens,
                ablate_conditioning,
                temperature,
            )?;
            let Some(call) = parse_tool_call(&output) else {
                return Ok(clean_agentic_final(&output));
            };
            let call_json = serde_json::json!({
                "tool": call.name.clone(),
                "args": call.args.clone(),
            })
            .to_string();
            let result = registry.execute(&call);
            transcript.push_str("\n\n<assistant_tool_call>");
            transcript.push_str(&call_json);
            transcript.push_str("</assistant_tool_call>\n");
            transcript.push_str(&result.to_prompt_block(result_chars));
        }

        let final_prompt = registry.build_final_prompt(&transcript, max_steps);
        let output = self.generate_direct_with_temperature(
            &final_prompt,
            max_new_tokens,
            ablate_conditioning,
            temperature,
        )?;
        Ok(clean_agentic_final(&output))
    }

    fn generate_direct_with_temperature(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        temperature: Option<f32>,
    ) -> Result<String> {
        let start = Instant::now();
        use crate::tasks::orchestrator::Action;
        let state_slots = self.encode_prompt_context_compressor(prompt)?;
        let plan = self.lewm_plan_from_state(prompt, &state_slots)?;
        let mut action = plan.first_action;
        let next_slots = plan.planned_slots;
        if action == Action::Done {
            return Ok(String::new());
        }
        let fetched_docs_action = action == Action::FetchDocs;
        let generation_prompt = prompt.to_string();
        let mut effective_slots = next_slots;
        if fetched_docs_action {
            effective_slots = self
                .transition
                .forward_one(&effective_slots, Action::Code as u32)?;
            action = Action::Code;
        }
        let chunk_tokens = match action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
            Action::FetchDocs => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
        };
        effective_slots =
            self.refine_latent_for_decoder(&generation_prompt, action, &effective_slots)?;
        let (decoder, cond_vec) = self.get_decoder_and_cond_from_context_compressor(
            &effective_slots,
            action,
            ablate_conditioning,
        )?;
        let decoder_tokens = if RlmDecoderRuntime::should_wrap_action(action.as_str()) {
            max_new_tokens
        } else {
            chunk_tokens
        };
        let mut assistant_content = decoder.generate_with_temperature(
            &generation_prompt,
            action.as_str(),
            &cond_vec,
            decoder_tokens,
            temperature,
        )?;
        if action == Action::Code && code_request_language(&generation_prompt).is_some() {
            assistant_content = maybe_repair_code_output(
                decoder.as_ref(),
                &generation_prompt,
                action.as_str(),
                &cond_vec,
                decoder_tokens,
                temperature,
                assistant_content,
            );
        }
        if std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] response in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
        Ok(assistant_content)
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
    ) -> Result<String> {
        self.generate_with_temperature(prompt, max_new_tokens, ablate_conditioning, None)
    }

    pub fn generate_with_temperature(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        temperature: Option<f32>,
    ) -> Result<String> {
        if agentic_decoder_requested() {
            if let Some(registry) = BashToolRegistry::load_from_env()? {
                return self.generate_agentic_with_tools(
                    prompt,
                    max_new_tokens,
                    ablate_conditioning,
                    temperature,
                    registry,
                );
            }
        }
        self.generate_direct_with_temperature(
            prompt,
            max_new_tokens,
            ablate_conditioning,
            temperature,
        )
    }

    pub fn generate_for_action(
        &self,
        prompt: &str,
        action: crate::tasks::orchestrator::Action,
        max_new_tokens: usize,
        ablate_conditioning: bool,
    ) -> Result<String> {
        use crate::tasks::orchestrator::Action;
        if action == Action::Done {
            return Ok(String::new());
        }
        let state_slots = self.encode_prompt_context_compressor(prompt)?;
        let generation_prompt = prompt.to_string();
        let mut effective_action = action;
        let mut next_slots = self.transition.forward_one(&state_slots, action as u32)?;
        if action == Action::FetchDocs {
            next_slots = self
                .transition
                .forward_one(&next_slots, Action::Code as u32)?;
            effective_action = Action::Code;
        }
        next_slots =
            self.refine_latent_for_decoder(&generation_prompt, effective_action, &next_slots)?;
        let (decoder, cond_vec) = self.get_decoder_and_cond_from_context_compressor(
            &next_slots,
            effective_action,
            ablate_conditioning,
        )?;
        let chunk_tokens = match effective_action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
            Action::FetchDocs => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
        };
        let decoder_tokens = if RlmDecoderRuntime::should_wrap_action(effective_action.as_str()) {
            max_new_tokens
        } else {
            chunk_tokens
        };
        let mut assistant_content = decoder.generate_with_temperature(
            &generation_prompt,
            effective_action.as_str(),
            &cond_vec,
            decoder_tokens,
            None,
        )?;
        if effective_action == Action::Code && code_request_language(&generation_prompt).is_some() {
            assistant_content = maybe_repair_code_output(
                decoder.as_ref(),
                &generation_prompt,
                effective_action.as_str(),
                &cond_vec,
                decoder_tokens,
                None,
                assistant_content,
            );
        }
        Ok(assistant_content)
    }

    pub fn predict_action(&self, prompt: &str) -> Result<crate::tasks::orchestrator::Action> {
        let state_slots = self.encode_prompt_context_compressor(prompt)?;
        Ok(self
            .lewm_plan_from_state(prompt, &state_slots)?
            .first_action)
    }

    pub fn uses_recursive_code_generation(
        &self,
        prompt: &str,
        action: crate::tasks::orchestrator::Action,
    ) -> bool {
        let cfg_min_chars = std::env::var("TOFY_DECODER_RLM_MIN_CHARS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3600usize);
        RlmDecoderRuntime::should_wrap_action(action.as_str())
            && (action == crate::tasks::orchestrator::Action::Code
                || prompt.chars().count() >= cfg_min_chars)
    }

    pub fn uses_fetch_docs(
        &self,
        prompt: &str,
        action: crate::tasks::orchestrator::Action,
    ) -> bool {
        let _ = (prompt, action);
        false
    }

    /// Stream generated text in chunks (for SSE). The action_classifier chooses a single decoder mode for the reply.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        self.generate_stream_with_temperature(
            prompt,
            max_new_tokens,
            ablate_conditioning,
            None,
            on_chunk,
        )
    }

    pub fn generate_stream_with_temperature(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        temperature: Option<f32>,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let start = Instant::now();
        if agentic_decoder_requested() {
            on_chunk("Thinking... ");
            let text = self.generate_with_temperature(
                prompt,
                max_new_tokens,
                ablate_conditioning,
                temperature,
            )?;
            if !text.trim().is_empty() {
                on_chunk(&text);
            }
            if std::env::var("JEPA_DEBUG").is_ok() {
                let _ = writeln!(
                    std::io::stderr(),
                    "[tofy] response in {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }
            return Ok(());
        }
        use crate::tasks::orchestrator::Action;
        on_chunk("Thinking... ");
        let state_slots = self.encode_prompt_context_compressor(prompt)?;
        let plan = self.lewm_plan_from_state(prompt, &state_slots)?;
        let mut action = plan.first_action;
        if action == Action::Done {
            on_chunk("Done. ");
            return Ok(());
        }
        let fetched_docs_action = action == Action::FetchDocs;
        let generation_prompt = prompt.to_string();
        let mut next_slots = plan.planned_slots;
        if fetched_docs_action {
            on_chunk("Preparing code. ");
            next_slots = self
                .transition
                .forward_one(&next_slots, Action::Code as u32)?;
            action = Action::Code;
        }
        match action {
            Action::TextReply => on_chunk("Writing text. "),
            Action::Code => on_chunk("Generating code. "),
            Action::Done => on_chunk("Done. "),
            Action::FetchDocs => on_chunk("Preparing code. "),
        }
        let chunk_tokens = match action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
            Action::FetchDocs => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
        };
        next_slots = self.refine_latent_for_decoder(&generation_prompt, action, &next_slots)?;
        let (decoder, cond_vec) = self.get_decoder_and_cond_from_context_compressor(
            &next_slots,
            action,
            ablate_conditioning,
        )?;
        let decoder_tokens = if RlmDecoderRuntime::should_wrap_action(action.as_str()) {
            max_new_tokens
        } else {
            chunk_tokens
        };
        if action == Action::Code && code_request_language(&generation_prompt).is_some() {
            let mut assistant_content = decoder.generate_with_temperature(
                &generation_prompt,
                action.as_str(),
                &cond_vec,
                decoder_tokens,
                temperature,
            )?;
            assistant_content = maybe_repair_code_output(
                decoder.as_ref(),
                &generation_prompt,
                action.as_str(),
                &cond_vec,
                decoder_tokens,
                temperature,
                assistant_content,
            );
            on_chunk(&assistant_content);
            if std::env::var("JEPA_DEBUG").is_ok() {
                let _ = writeln!(
                    std::io::stderr(),
                    "[tofy] response in {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }
            return Ok(());
        }
        decoder.generate_stream_with_temperature(
            &generation_prompt,
            action.as_str(),
            &cond_vec,
            decoder_tokens,
            temperature,
            &mut |chunk: &str| on_chunk(chunk),
        )?;
        if std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] response in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
        Ok(())
    }
}

fn context_compressor_mask_from_lengths(
    features: &EncoderFeatures,
    token_lengths: &[usize],
) -> Result<Tensor> {
    let (batch, token_slots, _) = features.token_states.dims3()?;
    let chunk_slots = features.chunk_states.dim(1)?;
    let global_slots = features.global_states.dim(1)?;
    let chunk_size = token_slots.div_ceil(chunk_slots.max(1)).max(1);
    let total_slots = token_slots + chunk_slots + global_slots + 2;
    let mut mask_buf: Vec<f32> = Vec::with_capacity(batch * total_slots);
    for b in 0..batch {
        let token_len = token_lengths
            .get(b)
            .copied()
            .unwrap_or(token_slots)
            .min(token_slots);
        let chunk_len = if token_len == 0 {
            0
        } else {
            token_len.div_ceil(chunk_size).min(chunk_slots)
        };
        mask_buf.extend((0..token_slots).map(|idx| if idx < token_len { 1.0f32 } else { 0.0f32 }));
        mask_buf.extend((0..chunk_slots).map(|idx| if idx < chunk_len { 1.0f32 } else { 0.0f32 }));
        mask_buf.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
    }
    Tensor::from_vec(
        mask_buf,
        (batch, total_slots),
        features.token_states.device(),
    )
    .map_err(Into::into)
}

fn planner_forward_encoder_masked(
    context_compressor: &ContextCompressor,
    features: &EncoderFeatures,
    token_lengths: &[usize],
) -> Result<Tensor> {
    let planner = features.planner_summary()?;
    let routing = features.routing_summary()?;
    let memory = Tensor::cat(
        &[
            features.token_states.clone(),
            features.chunk_states.clone(),
            features.global_states.clone(),
            planner,
            routing,
        ],
        1,
    )?;
    let mask = context_compressor_mask_from_lengths(features, token_lengths)?;
    context_compressor.forward_masked(&memory, Some(&mask))
}

fn maybe_detach_features(features: EncoderFeatures, detach: bool) -> EncoderFeatures {
    if detach {
        features.detached()
    } else {
        features
    }
}

fn context_segment_ranges(
    total_tokens: usize,
    max_seq: usize,
    max_segments: usize,
) -> Vec<(usize, usize)> {
    let max_seq = max_seq.max(1);
    let max_segments = max_segments.max(1);
    let keep_tokens = max_seq.saturating_mul(max_segments);
    let start = total_tokens.saturating_sub(keep_tokens);
    let mut ranges = Vec::new();
    let mut cursor = start;
    while cursor < total_tokens {
        let end = (cursor + max_seq).min(total_tokens);
        ranges.push((cursor, end));
        cursor = end;
    }
    if ranges.is_empty() {
        ranges.push((0, 0));
    }
    ranges
}

fn context_compressor_segment_batch_from_features(
    features: &EncoderFeatures,
    token_lengths: &[usize],
    include_tokens: bool,
) -> Result<(Tensor, Tensor)> {
    let planner = features.planner_summary()?;
    let routing = features.routing_summary()?;
    let batch = features.token_states.dim(0)?;
    let token_slots = features.token_states.dim(1)?;
    let chunk_slots = features.chunk_states.dim(1)?;
    let global_slots = features.global_states.dim(1)?;
    let chunk_size = token_slots.div_ceil(chunk_slots.max(1));
    let mask_slots = if include_tokens {
        token_slots + chunk_slots + global_slots + 2
    } else {
        chunk_slots + global_slots + 2
    };
    let mut mask_buf = Vec::with_capacity(batch * mask_slots);
    for b in 0..batch {
        let token_len = token_lengths
            .get(b)
            .copied()
            .unwrap_or(token_slots)
            .min(token_slots);
        let valid_chunks = if token_len == 0 {
            0
        } else {
            token_len.div_ceil(chunk_size).min(chunk_slots)
        };
        if include_tokens {
            mask_buf.extend((0..token_slots).map(|idx| if idx < token_len { 1.0 } else { 0.0 }));
        }
        mask_buf.extend((0..chunk_slots).map(|idx| if idx < valid_chunks { 1.0 } else { 0.0 }));
        mask_buf.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
    }

    let memory = if include_tokens {
        Tensor::cat(
            &[
                features.token_states.clone(),
                features.chunk_states.clone(),
                features.global_states.clone(),
                planner,
                routing,
            ],
            1,
        )?
    } else {
        Tensor::cat(
            &[
                features.chunk_states.clone(),
                features.global_states.clone(),
                planner,
                routing,
            ],
            1,
        )?
    };
    let mask = util::from_vec_like(mask_buf, (batch, mask_slots), &memory)?;
    Ok((memory, mask))
}

fn recursive_memory_retain(
    segment_idx: usize,
    total_segments: usize,
    recent_full_segments: usize,
) -> f64 {
    let remaining = total_segments.saturating_sub(segment_idx + 1);
    if remaining < recent_full_segments.max(1) {
        0.42
    } else {
        0.72
    }
}

struct PlannerSegmentRecord {
    sample_idx: usize,
    segment_idx: usize,
    total_segments: usize,
    recent_full_segments: usize,
    token_len: usize,
    include_tokens: bool,
}

fn append_padded_segment(
    out: &mut Vec<u32>,
    tokens: &[u32],
    start: usize,
    end: usize,
    max_seq: usize,
    pad_id: u32,
) -> usize {
    let row_start = out.len();
    let token_len = if start < end {
        let len = (end - start).min(max_seq);
        out.extend(tokens[start..end].iter().take(len).copied());
        len
    } else {
        0
    };
    while out.len() - row_start < max_seq {
        out.push(pad_id);
    }
    token_len
}

fn make_tail_token_batch(
    token_sequences: &[&[u32]],
    max_seq: usize,
    pad_id: u32,
) -> (Vec<u32>, Vec<usize>) {
    let mut input_buf = Vec::with_capacity(token_sequences.len() * max_seq);
    let mut token_lengths = Vec::with_capacity(token_sequences.len());
    for tokens in token_sequences {
        let ranges = context_segment_ranges(tokens.len(), max_seq, 1);
        let (start, end) = ranges[0];
        let token_len = append_padded_segment(&mut input_buf, tokens, start, end, max_seq, pad_id);
        token_lengths.push(token_len);
    }
    (input_buf, token_lengths)
}

fn make_segment_token_batch(
    token_sequences: &[&[u32]],
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    pad_id: u32,
) -> (Vec<u32>, Vec<PlannerSegmentRecord>, Vec<Vec<usize>>) {
    let mut input_buf = Vec::with_capacity(token_sequences.len() * context_segments * max_seq);
    let mut records = Vec::with_capacity(token_sequences.len() * context_segments);
    let mut records_by_sample = Vec::with_capacity(token_sequences.len());

    for (sample_idx, tokens) in token_sequences.iter().enumerate() {
        let segments = context_segment_ranges(tokens.len(), max_seq, context_segments);
        let sample_recent_full_segments = recent_full_segments.min(segments.len()).max(1);
        let mut sample_records = Vec::with_capacity(segments.len());
        for (segment_idx, (start, end)) in segments.iter().copied().enumerate() {
            let token_len =
                append_padded_segment(&mut input_buf, tokens, start, end, max_seq, pad_id);
            let include_tokens = segment_idx + sample_recent_full_segments >= segments.len();
            let record_idx = records.len();
            records.push(PlannerSegmentRecord {
                sample_idx,
                segment_idx,
                total_segments: segments.len(),
                recent_full_segments: sample_recent_full_segments,
                token_len,
                include_tokens,
            });
            sample_records.push(record_idx);
        }
        records_by_sample.push(sample_records);
    }

    (input_buf, records, records_by_sample)
}

fn select_encoder_features(
    features: &EncoderFeatures,
    record_indices: &[usize],
) -> Result<EncoderFeatures> {
    let index_values = record_indices
        .iter()
        .map(|idx| *idx as u32)
        .collect::<Vec<_>>();
    let indexes = Tensor::from_vec(
        index_values,
        (record_indices.len(),),
        features.token_states.device(),
    )?;
    Ok(EncoderFeatures {
        token_states: features
            .token_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        chunk_states: features
            .chunk_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        global_states: features
            .global_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        pooled_queries: features
            .pooled_queries
            .contiguous()?
            .index_select(&indexes, 0)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_slots_from_token_sequences(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    token_sequences: &[&[u32]],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    device: &Device,
) -> Result<Tensor> {
    context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        true,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn context_slots_from_token_sequences_with_detach(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    token_sequences: &[&[u32]],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    detach_encoder: bool,
    device: &Device,
) -> Result<Tensor> {
    if token_sequences.is_empty() {
        bail!("context slot batch is empty");
    }
    let max_seq = max_seq.max(1);
    let context_segments = context_segments.max(1);
    let segment_batch_limit = env_usize("TOFY_CONTEXT_SEGMENT_BATCH", 64);
    let hybrid_context = env_bool("TOFY_CONTEXT_HYBRID_MEMORY", true);
    let hybrid_exact_tail = env_usize(
        "TOFY_CONTEXT_HYBRID_EXACT_TAIL",
        default_context_hybrid_exact_tail(max_seq, recent_full_segments),
    );
    let hybrid_block_size = env_usize("TOFY_CONTEXT_HYBRID_BLOCK_SIZE", 16);
    let hybrid_retrieval_slots = env_usize("TOFY_CONTEXT_RETRIEVAL_SLOTS", 8);

    if context_segments == 1 && !recursive_context_compressor {
        let mut chunk_slots =
            Vec::with_capacity(token_sequences.len().div_ceil(segment_batch_limit));
        for chunk in token_sequences.chunks(segment_batch_limit) {
            let (input_buf, token_lengths) = make_tail_token_batch(chunk, max_seq, pad_id);
            let input_ids = Tensor::from_vec(input_buf, (chunk.len(), max_seq), device)?;
            let features =
                maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
            chunk_slots.push(planner_forward_encoder_masked(
                context_compressor,
                &features,
                &token_lengths,
            )?);
        }
        let refs = chunk_slots.iter().collect::<Vec<_>>();
        return Tensor::cat(&refs, 0).map_err(Into::into);
    }

    let (input_buf, records, records_by_sample) = make_segment_token_batch(
        token_sequences,
        max_seq,
        context_segments,
        recent_full_segments,
        pad_id,
    );
    let mut sample_slots = Vec::with_capacity(token_sequences.len());
    if recursive_context_compressor {
        let mut slots_by_record: Vec<Option<Tensor>> = (0..records.len()).map(|_| None).collect();
        for chunk_start in (0..records.len()).step_by(segment_batch_limit) {
            let chunk_end = (chunk_start + segment_batch_limit).min(records.len());
            let chunk_len = chunk_end - chunk_start;
            let offset = chunk_start * max_seq;
            let end = chunk_end * max_seq;
            let input_ids = Tensor::from_vec(
                input_buf[offset..end].to_vec(),
                (chunk_len, max_seq),
                device,
            )?;
            let features =
                maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
            let mut summary_indices = Vec::new();
            let mut full_indices = Vec::new();
            for local_idx in 0..chunk_len {
                if records[chunk_start + local_idx].include_tokens {
                    full_indices.push(local_idx);
                } else {
                    summary_indices.push(local_idx);
                }
            }
            for (include_tokens, local_indices) in [(false, summary_indices), (true, full_indices)]
            {
                if local_indices.is_empty() {
                    continue;
                }
                let selected = select_encoder_features(&features, &local_indices)?;
                let token_lengths = local_indices
                    .iter()
                    .map(|idx| records[chunk_start + *idx].token_len)
                    .collect::<Vec<_>>();
                let (memory, mask) = context_compressor_segment_batch_from_features(
                    &selected,
                    &token_lengths,
                    include_tokens,
                )?;
                let slots = context_compressor.forward_masked(&memory, Some(&mask))?;
                for (group_pos, local_idx) in local_indices.iter().copied().enumerate() {
                    let record_idx = chunk_start + local_idx;
                    slots_by_record[record_idx] = Some(slots.narrow(0, group_pos, 1)?);
                }
            }
        }

        for sample_records in &records_by_sample {
            let mut folded_slots: Option<Tensor> = None;
            for record_idx in sample_records {
                let record = &records[*record_idx];
                debug_assert_eq!(record.sample_idx, sample_slots.len());
                let segment_slots = slots_by_record[*record_idx]
                    .as_ref()
                    .context("missing context slots for segment record")?;
                folded_slots = Some(match folded_slots {
                    Some(prev_slots) => context_compressor.fold_slots(
                        &prev_slots,
                        segment_slots,
                        recursive_memory_retain(
                            record.segment_idx,
                            record.total_segments,
                            record.recent_full_segments,
                        ),
                    )?,
                    None => segment_slots.clone(),
                });
            }
            sample_slots.push(folded_slots.context("recursive planner fold produced no slots")?);
        }

        let refs = sample_slots.iter().collect::<Vec<_>>();
        return Tensor::cat(&refs, 0).map_err(Into::into);
    }

    let mut memory_by_record: Vec<Option<(Tensor, Tensor)>> =
        (0..records.len()).map(|_| None).collect();
    for chunk_start in (0..records.len()).step_by(segment_batch_limit) {
        let chunk_end = (chunk_start + segment_batch_limit).min(records.len());
        let chunk_len = chunk_end - chunk_start;
        let offset = chunk_start * max_seq;
        let end = chunk_end * max_seq;
        let input_ids = Tensor::from_vec(
            input_buf[offset..end].to_vec(),
            (chunk_len, max_seq),
            device,
        )?;
        let features = maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
        let mut summary_indices = Vec::new();
        let mut full_indices = Vec::new();
        for local_idx in 0..chunk_len {
            if records[chunk_start + local_idx].include_tokens {
                full_indices.push(local_idx);
            } else {
                summary_indices.push(local_idx);
            }
        }
        for (include_tokens, local_indices) in [(false, summary_indices), (true, full_indices)] {
            if local_indices.is_empty() {
                continue;
            }
            let selected = select_encoder_features(&features, &local_indices)?;
            let token_lengths = local_indices
                .iter()
                .map(|idx| records[chunk_start + *idx].token_len)
                .collect::<Vec<_>>();
            let (memory, mask) = context_compressor_segment_batch_from_features(
                &selected,
                &token_lengths,
                include_tokens,
            )?;
            for (group_pos, local_idx) in local_indices.iter().copied().enumerate() {
                let record_idx = chunk_start + local_idx;
                memory_by_record[record_idx] = Some((
                    memory.narrow(0, group_pos, 1)?,
                    mask.narrow(0, group_pos, 1)?,
                ));
            }
        }
    }

    for sample_records in &records_by_sample {
        let mut memory_refs = Vec::with_capacity(sample_records.len());
        let mut mask_refs = Vec::with_capacity(sample_records.len());
        for record_idx in sample_records {
            let (memory, mask) = memory_by_record[*record_idx]
                .as_ref()
                .context("missing context compressor for segment record")?;
            memory_refs.push(memory);
            mask_refs.push(mask);
        }
        let memory = Tensor::cat(&memory_refs, 1)?;
        let mask = Tensor::cat(&mask_refs, 1)?;
        if hybrid_context && sample_records.len() > 1 {
            sample_slots.push(context_compressor.forward_hybrid_masked(
                &memory,
                Some(&mask),
                hybrid_exact_tail,
                hybrid_block_size,
                hybrid_retrieval_slots,
            )?);
        } else {
            sample_slots.push(context_compressor.forward_masked(&memory, Some(&mask))?);
        }
    }

    let refs = sample_slots.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn context_slots_from_world_pair_batch(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    detach_encoder: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if batch.is_empty() {
        bail!("world pair batch is empty");
    }
    let post_state_sequences = world_post_state_token_sequences(batch);
    let mut token_sequences = Vec::with_capacity(batch.len() * 2);
    token_sequences.extend(batch.iter().map(|row| row.state_tokens.as_slice()));
    token_sequences.extend(post_state_sequences.iter().map(Vec::as_slice));
    let slots = context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        &token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        detach_encoder,
        device,
    )?;
    let batch_size = batch.len();
    let state_slots = slots.narrow(0, 0, batch_size)?;
    let next_slots = slots.narrow(0, batch_size, batch_size)?;
    Ok((state_slots, next_slots))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_slots_from_world_pair_sequences(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if batch.is_empty() {
        bail!("world pair batch is empty");
    }
    let post_state_sequences = world_post_state_token_sequences(batch);
    let mut token_sequences = Vec::with_capacity(batch.len() * 2);
    token_sequences.extend(batch.iter().map(|row| row.state_tokens.as_slice()));
    token_sequences.extend(post_state_sequences.iter().map(Vec::as_slice));
    let slots = context_slots_from_token_sequences(
        encoder,
        context_compressor,
        &token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        device,
    )?;
    let batch_size = batch.len();
    let state_slots = slots.narrow(0, 0, batch_size)?;
    let next_slots = slots.narrow(0, batch_size, batch_size)?;
    Ok((state_slots, next_slots))
}

fn world_post_state_token_sequences(batch: &[WorldExample]) -> Vec<Vec<u32>> {
    batch
        .iter()
        .map(|row| {
            let mut tokens = Vec::with_capacity(row.state_tokens.len() + row.next_tokens.len());
            tokens.extend(row.state_tokens.iter().copied());
            tokens.extend(row.next_tokens.iter().copied());
            tokens
        })
        .collect()
}

fn world_post_state_loss_weight() -> f64 {
    let requested = std::env::var("TOFY_WORLD_POST_STATE_LOSS_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    if requested > 0.0 {
        eprintln!(
            "WARNING: TOFY_WORLD_POST_STATE_LOSS_WEIGHT={} ignored; the transition target is already encoded as post-state (state + next), so this auxiliary would duplicate transition loss.",
            requested
        );
    }
    0.0
}

fn world_rollout_loss_weight() -> f64 {
    std::env::var("TOFY_WORLD_ROLLOUT_LOSS_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.25)
        .max(0.0)
}

fn world_rollout_steps() -> usize {
    std::env::var("TOFY_WORLD_TRAIN_ROLLOUT_STEPS")
        .or_else(|_| std::env::var("TOFY_WORLD_ROLLOUT_STEPS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1)
}

fn continuation_overlap(left: &[u32], right: &[u32]) -> usize {
    let max = left.len().min(right.len());
    for len in (1..=max).rev() {
        if left[left.len() - len..] == right[..len] {
            return len;
        }
    }
    0
}

fn continuation_edge_score_with_combined(
    from_next_tokens: &[u32],
    combined: &[u32],
    to: &WorldExample,
) -> usize {
    if from_next_tokens.is_empty() || combined.is_empty() || to.state_tokens.is_empty() {
        return 0;
    }
    if to.state_tokens.starts_with(combined) {
        return combined.len();
    }
    if combined.ends_with(&to.state_tokens) {
        return to.state_tokens.len();
    }
    let next_overlap = continuation_overlap(from_next_tokens, &to.state_tokens);
    let combined_overlap = continuation_overlap(combined, &to.state_tokens);
    next_overlap.max(combined_overlap)
}

fn continuation_edges(batch: &[WorldExample]) -> Vec<Option<usize>> {
    let min_overlap = env_usize("TOFY_WORLD_ROLLOUT_MIN_OVERLAP", 24);
    let mut by_first_token: HashMap<u32, Vec<usize>> = HashMap::new();
    for (idx, row) in batch.iter().enumerate() {
        if let Some(&first) = row.state_tokens.first() {
            by_first_token.entry(first).or_default().push(idx);
        }
    }

    let mut edges = vec![None; batch.len()];
    for (from_idx, from) in batch.iter().enumerate() {
        if from.state_tokens.is_empty() || from.next_tokens.is_empty() {
            continue;
        }
        let mut best = None;
        let mut best_score = min_overlap.saturating_sub(1);
        let mut combined = Vec::with_capacity(from.state_tokens.len() + from.next_tokens.len());
        combined.extend(from.state_tokens.iter().copied());
        combined.extend(from.next_tokens.iter().copied());

        let mut candidate_tokens = combined.clone();
        candidate_tokens.sort_unstable();
        candidate_tokens.dedup();
        let mut candidate_indices = Vec::new();
        for token in candidate_tokens {
            if let Some(indices) = by_first_token.get(&token) {
                candidate_indices.extend(indices.iter().copied());
            }
        }
        candidate_indices.sort_unstable();
        candidate_indices.dedup();

        for to_idx in candidate_indices {
            if from_idx == to_idx {
                continue;
            }
            let score =
                continuation_edge_score_with_combined(&from.next_tokens, &combined, &batch[to_idx]);
            if score > best_score {
                best = Some(to_idx);
                best_score = score;
            }
        }
        edges[from_idx] = best;
    }
    edges
}

fn index_slot_rows(slots: &Tensor, indices: &[usize]) -> Result<Tensor> {
    let ids = Tensor::from_vec(
        indices.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
        (indices.len(),),
        slots.device(),
    )?;
    slots
        .contiguous()?
        .index_select(&ids, 0)
        .map_err(Into::into)
}

fn rollout_loss_from_batch(
    transition: &ActionStateTransition,
    state_slots: &Tensor,
    batch: &[WorldExample],
    rollout_steps: usize,
    rollout_loss_weight: f64,
) -> Result<Option<Tensor>> {
    if batch.len() < 2 || rollout_loss_weight == 0.0 {
        return Ok(None);
    }
    let edges = continuation_edges(batch);
    let mut starts = Vec::new();
    let mut current_indices = Vec::new();
    for (idx, edge) in edges.iter().enumerate() {
        if edge.is_some() {
            starts.push(idx);
            current_indices.push(idx);
        }
    }
    if starts.is_empty() {
        return Ok(None);
    }

    let mut pred = index_slot_rows(state_slots, &starts)?;
    let mut losses = Vec::new();
    for depth in 0..rollout_steps.max(1) {
        let mut labels = Vec::with_capacity(current_indices.len());
        let mut target_indices = Vec::with_capacity(current_indices.len());
        let mut kept_positions = Vec::with_capacity(current_indices.len());
        for (pos, &current_idx) in current_indices.iter().enumerate() {
            if let Some(target_idx) = edges[current_idx] {
                labels.push(batch[current_idx].action_label);
                target_indices.push(target_idx);
                kept_positions.push(pos);
            }
        }
        if target_indices.is_empty() {
            break;
        }
        if kept_positions.len() != current_indices.len() {
            pred = index_slot_rows(&pred, &kept_positions)?;
        }
        let next_pred = transition.forward(&pred, &labels)?;
        let target = index_slot_rows(state_slots, &target_indices)?;
        let weight = 0.5f64.powi(depth as i32);
        losses.push(prediction_loss(&next_pred, &target.detach())?.affine(weight, 0.0)?);
        pred = next_pred;
        current_indices = target_indices;
    }

    if losses.is_empty() {
        return Ok(None);
    }
    let refs = losses.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 0)?
        .mean_all()
        .map(Some)
        .map_err(Into::into)
}

pub(crate) fn rollout_transition_slots(
    transition: &ActionStateTransition,
    state_slots: &Tensor,
    action_label: u32,
    rollout_steps: usize,
) -> Result<Tensor> {
    let mut slots = state_slots.clone();
    for _ in 0..rollout_steps.max(1) {
        slots = transition.forward_one(&slots, action_label)?;
    }
    Ok(slots)
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}

pub(crate) fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug)]
struct DecoderConditioningNegatives {
    zero: bool,
    shuffle: bool,
    hard: bool,
}

impl DecoderConditioningNegatives {
    fn none() -> Self {
        Self {
            zero: false,
            shuffle: false,
            hard: false,
        }
    }

    fn from_env() -> Self {
        let value = std::env::var("TOFY_DECODER_CONDITIONING_NEGATIVES")
            .unwrap_or_else(|_| "hard".to_string());
        let mut out = Self::none();
        let mut allow_empty = false;
        for part in value.split(',') {
            match part.trim().to_ascii_lowercase().as_str() {
                "all" => {
                    out.zero = true;
                    out.shuffle = true;
                    out.hard = true;
                }
                "zero" | "ablated" => out.zero = true,
                "none" => allow_empty = true,
                "shuffle" | "shuffled" => out.shuffle = true,
                "hard" | "hard_mismatch" | "mismatch" => out.hard = true,
                "" => {}
                other => {
                    println!("Ignoring unknown TOFY_DECODER_CONDITIONING_NEGATIVES entry: {other}")
                }
            }
        }
        if out.count() == 0 && !allow_empty {
            out.zero = true;
        }
        out
    }

    fn count(self) -> usize {
        usize::from(self.zero) + usize::from(self.shuffle) + usize::from(self.hard)
    }
}

fn add_conditioning_margin_loss(
    existing: Option<Tensor>,
    token_loss: &Tensor,
    negative_loss: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    let margin_loss = token_loss
        .broadcast_sub(negative_loss)?
        .affine(1.0, margin)?
        .relu()?;
    if let Some(existing) = existing {
        existing.broadcast_add(&margin_loss).map_err(Into::into)
    } else {
        Ok(margin_loss)
    }
}
