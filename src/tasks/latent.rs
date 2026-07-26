use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::Deserialize;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::cli::resolve_data_path;
use crate::config::{LatentEvalConfig, LatentTrainConfig};
use crate::data::{
    build_vocab_from_pair_file, count_pairs_with_vocab, make_augmented_jepa_batch,
    make_augmented_jepa_batch_from_pairs, prepare_ultrachat_pairs, tokenizer_spec_signature,
    AugmentedJepaBatch, CachedPairStream, CurriculumDenoisingConfig, PairStream, TokenizationMode,
    DEFAULT_MIN_TOKENS_PER_LINE, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::encoders::EncoderFeatures;
use crate::model::vocab::{vocab_signature, Pair, Vocab};
use crate::model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, sigreg_epps_pulley_linearization_chunked_seeded,
    sigreg_epps_pulley_variable_length,
    sigreg_epps_pulley_variable_length_linearization_chunked_seeded, sigreg_linear_surrogate,
    tensor_rms, OnlineEncoder,
};
use crate::util;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn latent_grad_accum_for_step(step: usize, config: &LatentTrainConfig) -> usize {
    if config.grad_accum_warmup_steps > 0
        && step <= config.grad_accum_warmup_steps
        && config.grad_accum_warmup_value < config.grad_accum_steps
    {
        config.grad_accum_warmup_value.max(1)
    } else {
        config.grad_accum_steps.max(1)
    }
}

fn latent_batch_size_for_step(step: usize, config: &LatentTrainConfig) -> usize {
    if config.batch_warmup_steps > 0
        && step <= config.batch_warmup_steps
        && config.batch_warmup_value != config.batch_size
    {
        config.batch_warmup_value.max(1)
    } else {
        config.batch_size.max(1)
    }
}

fn split_encoder_features(
    features: &EncoderFeatures,
    batch_size: usize,
) -> Result<[EncoderFeatures; 3]> {
    Ok([
        EncoderFeatures {
            token_states: features.token_states.narrow(0, 0, batch_size)?,
            chunk_states: features.chunk_states.narrow(0, 0, batch_size)?,
            global_states: features.global_states.narrow(0, 0, batch_size)?,
            pooled_queries: features.pooled_queries.narrow(0, 0, batch_size)?,
        },
        EncoderFeatures {
            token_states: features.token_states.narrow(0, batch_size, batch_size)?,
            chunk_states: features.chunk_states.narrow(0, batch_size, batch_size)?,
            global_states: features.global_states.narrow(0, batch_size, batch_size)?,
            pooled_queries: features.pooled_queries.narrow(0, batch_size, batch_size)?,
        },
        EncoderFeatures {
            token_states: features
                .token_states
                .narrow(0, batch_size * 2, batch_size)?,
            chunk_states: features
                .chunk_states
                .narrow(0, batch_size * 2, batch_size)?,
            global_states: features
                .global_states
                .narrow(0, batch_size * 2, batch_size)?,
            pooled_queries: features
                .pooled_queries
                .narrow(0, batch_size * 2, batch_size)?,
        },
    ])
}

const LATENT_HELDOUT_SPLIT_MODULUS: usize = 20;
const LATENT_HELDOUT_SPLIT_REMAINDER: usize = 0;
/// Below this many scanned pairs, train on everything and select on the train
/// metric instead of starving the run with a tiny heldout split.
const LATENT_MIN_PAIRS_FOR_VAL_SPLIT: usize = 200;

struct LatentForward {
    pred_loss: Tensor,
    token_pred_loss: Tensor,
    chunk_pred_loss: Tensor,
    global_pred_loss: Tensor,
    pred_token_flat: Tensor,
    target_flat: Tensor,
    context_targets: Tensor,
    target_targets: Tensor,
    pred_chunks_masked: Tensor,
    target_chunks_masked: Tensor,
    pred_global_flat: Tensor,
    target_global_flat: Tensor,
    valid_token_states: Tensor,
    regularized_chunk_states: Tensor,
    regularized_global_states: Tensor,
    token_valid_lens: Vec<usize>,
    chunk_valid_lens: Vec<usize>,
}

struct LatentSigRegBatch {
    token_states: Tensor,
    chunk_states: Tensor,
    global_states: Tensor,
    token_valid_lens: Vec<usize>,
    chunk_valid_lens: Vec<usize>,
}

impl LatentSigRegBatch {
    fn detached_from(forward: &LatentForward) -> Self {
        Self {
            token_states: forward.valid_token_states.detach(),
            chunk_states: forward.regularized_chunk_states.detach(),
            global_states: forward.regularized_global_states.detach(),
            token_valid_lens: forward.token_valid_lens.clone(),
            chunk_valid_lens: forward.chunk_valid_lens.clone(),
        }
    }
}

fn latent_sigreg_forward(
    encoder: &OnlineEncoder,
    batch: &AugmentedJepaBatch,
) -> Result<LatentSigRegBatch> {
    let all_view_ids = Tensor::cat(
        &[&batch.view_a_ids, &batch.target_ids, &batch.view_b_ids],
        0,
    )?;
    let features = encoder.forward_features(&all_view_ids)?;
    let (_, token_positions, _) = features.token_states.dims3()?;
    let (_, chunk_positions, _) = features.chunk_states.dims3()?;
    let chunk_size = encoder.chunk_size_for_seq_len(token_positions).max(1);
    let base_token_valid_lens = batch
        .valid_lens
        .iter()
        .copied()
        .map(|len| len.min(token_positions))
        .collect::<Vec<_>>();
    let base_chunk_valid_lens = base_token_valid_lens
        .iter()
        .map(|&len| len.div_ceil(chunk_size).min(chunk_positions))
        .collect::<Vec<_>>();
    Ok(LatentSigRegBatch {
        token_states: features.token_states,
        chunk_states: features.chunk_states,
        global_states: features.global_states,
        token_valid_lens: base_token_valid_lens.repeat(3),
        chunk_valid_lens: base_chunk_valid_lens.repeat(3),
    })
}

/// Shared three-view forward pass + multiscale prediction losses used by both
/// the training loop and held-out validation.
fn latent_forward(
    encoder: &OnlineEncoder,
    batch: &AugmentedJepaBatch,
    batch_size: usize,
    device: &Device,
) -> Result<LatentForward> {
    let all_view_ids = Tensor::cat(
        &[&batch.view_a_ids, &batch.target_ids, &batch.view_b_ids],
        0,
    )?;
    let all_view_features = encoder.forward_features(&all_view_ids)?;
    let [view_a_features, target_features, paired_view_targets] =
        split_encoder_features(&all_view_features, batch_size)?;
    let context_hidden = encoder.predict_states(&view_a_features.token_states)?;
    let target_hidden = target_features.token_states.clone();
    let (b, t, d) = context_hidden.dims3()?;
    let pred_token_flat = context_hidden.reshape((b * t, d))?;
    let target_flat = target_hidden.reshape((b * t, d))?;
    let context_targets = pred_token_flat.index_select(&batch.target_linear_indices, 0)?;
    let target_targets = target_flat.index_select(&batch.target_linear_indices, 0)?;
    let token_pred_loss = prediction_loss(&context_targets, &target_targets)?;

    // Restrict the chunk loss to chunks overlapping view A's masked spans:
    // unmasked chunks are nearly identical across views and only flatter the
    // chunk metric without contributing a learning signal.
    let (_, num_chunks, chunk_dim) = view_a_features.chunk_states.dims3()?;
    let chunk_size = encoder.chunk_size_for_seq_len(t).max(1);
    let mut masked_chunks: Vec<u32> = batch
        .target_linear_host
        .iter()
        .map(|&linear| {
            let row = linear as usize / t;
            let pos = linear as usize % t;
            (row * num_chunks + (pos / chunk_size).min(num_chunks - 1)) as u32
        })
        .collect();
    masked_chunks.sort_unstable();
    masked_chunks.dedup();
    let masked_chunk_count = masked_chunks.len();
    let masked_chunk_indices = Tensor::from_vec(masked_chunks, (masked_chunk_count,), device)?;
    let predicted_chunks = encoder.predict_states(&view_a_features.chunk_states)?;
    let pred_chunks_masked = predicted_chunks
        .reshape((b * num_chunks, chunk_dim))?
        .index_select(&masked_chunk_indices, 0)?;
    let target_chunks_masked = target_features
        .chunk_states
        .reshape((b * num_chunks, chunk_dim))?
        .index_select(&masked_chunk_indices, 0)?;
    let chunk_pred_loss = prediction_loss(&pred_chunks_masked, &target_chunks_masked)?;

    // Compare global latent tokens per slot instead of collapsing them into
    // one batch-mean vector.
    let predicted_global = encoder.predict_states(&view_a_features.global_states)?;
    let pred_global_flat = flatten_latent_slots(&predicted_global)?;
    let target_global_flat = flatten_latent_slots(&target_features.global_states)?;
    let global_pred_loss = prediction_loss(&pred_global_flat, &target_global_flat)?;

    let pred_loss = token_pred_loss
        .affine(0.70, 0.0)?
        .broadcast_add(&chunk_pred_loss.affine(0.20, 0.0)?)?
        .broadcast_add(&global_pred_loss.affine(0.10, 0.0)?)?;

    // Preserve latent position: SIGReg tests each position over independent
    // examples. Flattening slots into the sample axis lets different slot
    // distributions cancel one another and no longer implements that test.
    // Keep every real position and provide explicit row lengths to SIGReg.
    // Truncating to the shortest sequence would systematically discard the
    // long-context tail; treating padded chunks as samples would bias the
    // empirical characteristic function toward a point mass.
    let base_token_valid_lens = batch
        .valid_lens
        .iter()
        .copied()
        .map(|len| len.min(t))
        .collect::<Vec<_>>();
    let base_chunk_valid_lens = base_token_valid_lens
        .iter()
        .map(|&len| len.div_ceil(chunk_size).min(num_chunks))
        .collect::<Vec<_>>();
    let token_valid_lens = base_token_valid_lens.repeat(3);
    let chunk_valid_lens = base_chunk_valid_lens.repeat(3);
    let valid_token_states = Tensor::cat(
        &[
            &view_a_features.token_states,
            &target_features.token_states,
            &paired_view_targets.token_states,
        ],
        0,
    )?;
    let regularized_chunk_states = Tensor::cat(
        &[
            &view_a_features.chunk_states,
            &target_features.chunk_states,
            &paired_view_targets.chunk_states,
        ],
        0,
    )?;
    let regularized_global_states = Tensor::cat(
        &[
            &view_a_features.global_states,
            &target_features.global_states,
            &paired_view_targets.global_states,
        ],
        0,
    )?;

    Ok(LatentForward {
        pred_loss,
        token_pred_loss,
        chunk_pred_loss,
        global_pred_loss,
        pred_token_flat,
        target_flat,
        context_targets,
        target_targets,
        pred_chunks_masked,
        target_chunks_masked,
        pred_global_flat,
        target_global_flat,
        valid_token_states,
        regularized_chunk_states,
        regularized_global_states,
        token_valid_lens,
        chunk_valid_lens,
    })
}

/// Mean multiscale prediction loss over a few held-out batches, used for
/// checkpoint selection. Returns `None` when no validation stream exists.
#[allow(clippy::too_many_arguments)]
fn latent_validation_pred_loss(
    encoder: &OnlineEncoder,
    vocab: &Vocab,
    curriculum: &CurriculumDenoisingConfig,
    device: &Device,
    val_pair_stream: &mut Option<PairStream>,
    cached_val_stream: &mut Option<CachedPairStream>,
    batch_size: usize,
    batches: usize,
) -> Result<Option<f32>> {
    let mut total = 0f32;
    let mut count = 0usize;
    for _ in 0..batches {
        let batch = if let Some(stream) = cached_val_stream.as_mut() {
            let pairs = stream.next_batch(batch_size)?;
            make_augmented_jepa_batch_from_pairs(&pairs, vocab, curriculum, device)?
        } else if let Some(stream) = val_pair_stream.as_mut() {
            let tokens = stream.next_batch(batch_size)?;
            make_augmented_jepa_batch(&tokens, vocab, curriculum, device)?
        } else {
            return Ok(None);
        };
        let forward = latent_forward(encoder, &batch, batch_size, device)?;
        total += util::scalar_f32(&forward.pred_loss.detach())?;
        count += 1;
    }
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(total / count as f32))
}

const TOKEN_CACHE_MANIFEST_VERSION: u32 = 8;

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
    #[serde(default)]
    tokenizer_spec_signature: String,
    max_seq: usize,
    action_filter: Option<u32>,
    vocab_signature: String,
    token_cache_path: String,
}

fn token_cache_path(kind: &str) -> Option<PathBuf> {
    let enabled = std::env::var("TOFY_USE_TOKEN_CACHE")
        .ok()
        .is_none_or(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if !enabled {
        return None;
    }
    let cache_dir = std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string());
    let path = PathBuf::from(cache_dir).join(format!("{kind}.tokens.bin"));
    path.exists().then_some(path)
}

fn token_cache_manifest(kind: &str) -> Option<CacheTokenManifest> {
    let cache_dir = std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string());
    let path = PathBuf::from(cache_dir).join(format!("{kind}_tokens.manifest.json"));
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn source_fingerprint_matches(path: &Path, source: &CacheManifestSource) -> Result<bool> {
    if source.path != path.to_string_lossy() {
        return Ok(false);
    }
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if source.len != metadata.len() {
        return Ok(false);
    }
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
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

fn latent_token_cache_path(
    data_path: &Path,
    vocab_signature_value: &str,
    source_budget: usize,
) -> Result<Option<PathBuf>> {
    let Some(cache_path) = token_cache_path("encoder") else {
        return Ok(None);
    };
    let Some(manifest) = token_cache_manifest("encoder") else {
        return Ok(None);
    };
    if manifest.version != TOKEN_CACHE_MANIFEST_VERSION
        || manifest.kind != "encoder"
        || manifest.tokenizer != TokenizationMode::Default.as_str()
        || manifest.tokenizer_spec_signature != tokenizer_spec_signature(TokenizationMode::Default)
        || manifest.max_seq < source_budget
        || manifest.action_filter.is_some()
        || manifest.vocab_signature != vocab_signature_value
        || manifest.token_cache_path != cache_path.to_string_lossy().as_ref()
        || !source_fingerprint_matches(data_path, &manifest.source)?
    {
        return Ok(None);
    }
    Ok(Some(cache_path))
}

fn latent_curriculum(
    step: usize,
    total_steps: usize,
    config: &LatentTrainConfig,
) -> CurriculumDenoisingConfig {
    let progress = step as f64 / total_steps.max(1) as f64;
    let early_seq = (config.max_seq / 2).max(32);
    let mid_seq = ((config.max_seq * 3) / 4).max(early_seq);
    let (
        active_seq,
        min_masked_ratio,
        ratio_scale,
        span_budget_scale,
        span_scale,
        code_span_multiplier,
        identifier_focus_prob,
        block_focus_prob,
        comment_focus_prob,
        text_boundary_focus_prob,
    ) = if progress < 0.33 {
        (
            early_seq, 0.18, 1.15, 1.2, 0.85, 1.30, 0.72, 0.24, 0.18, 0.28,
        )
    } else if progress < 0.67 {
        (
            mid_seq, 0.26, 1.55, 1.55, 1.05, 1.55, 0.80, 0.34, 0.22, 0.36,
        )
    } else {
        (
            config.max_seq,
            0.34,
            1.9,
            1.8,
            1.25,
            1.70,
            0.86,
            0.42,
            0.28,
            0.44,
        )
    };
    CurriculumDenoisingConfig {
        max_seq: config.max_seq,
        active_seq,
        max_spans_per_sample: ((config.max_spans_per_sample.max(1) as f64) * span_budget_scale)
            .round()
            .max(1.0) as usize,
        max_span_len: ((config.max_span_len as f64) * span_scale).round() as usize,
        min_masked_ratio,
        max_masked_ratio: (config.max_masked_ratio * ratio_scale).clamp(min_masked_ratio, 0.55),
        code_span_multiplier,
        identifier_focus_prob,
        block_focus_prob,
        comment_focus_prob,
        text_boundary_focus_prob,
        code_masked_ratio_multiplier: 1.30,
        context_segments: config.latent_context_segments,
        recent_full_segments: config.latent_recent_full_segments,
        history_ratio: config.latent_history_ratio,
    }
}

#[derive(Clone, Copy)]
struct LatentLogSnapshot {
    loss_val: f32,
    pred_val: f32,
    token_pred_val: f32,
    chunk_pred_val: f32,
    global_pred_val: f32,
    sigreg_val: f32,
    pred_cos: f32,
    chunk_cos: f32,
    global_cos: f32,
    context_rms: f32,
    target_rms: f32,
    target_count: usize,
    target_frac: f32,
    code_fraction: f32,
    active_seq: usize,
    max_spans_per_sample: usize,
    min_masked_ratio: f32,
    max_masked_ratio: f32,
    reg_weight: f32,
}

fn mean_latent_log_snapshot(snapshots: &[LatentLogSnapshot]) -> Result<LatentLogSnapshot> {
    let last = *snapshots
        .last()
        .context("latent grad accumulation produced no log snapshot")?;
    let count = snapshots.len() as f32;
    let mean =
        |field: fn(&LatentLogSnapshot) -> f32| snapshots.iter().map(field).sum::<f32>() / count;
    Ok(LatentLogSnapshot {
        loss_val: mean(|row| row.loss_val),
        pred_val: mean(|row| row.pred_val),
        token_pred_val: mean(|row| row.token_pred_val),
        chunk_pred_val: mean(|row| row.chunk_pred_val),
        global_pred_val: mean(|row| row.global_pred_val),
        sigreg_val: mean(|row| row.sigreg_val),
        pred_cos: mean(|row| row.pred_cos),
        chunk_cos: mean(|row| row.chunk_cos),
        global_cos: mean(|row| row.global_cos),
        context_rms: mean(|row| row.context_rms),
        target_rms: mean(|row| row.target_rms),
        target_count: (snapshots
            .iter()
            .map(|row| row.target_count as f32)
            .sum::<f32>()
            / count)
            .round() as usize,
        target_frac: mean(|row| row.target_frac),
        code_fraction: mean(|row| row.code_fraction),
        active_seq: last.active_seq,
        max_spans_per_sample: last.max_spans_per_sample,
        min_masked_ratio: last.min_masked_ratio,
        max_masked_ratio: last.max_masked_ratio,
        reg_weight: last.reg_weight,
    })
}

pub fn try_run_prepare_ultrachat(args: &[String]) -> Result<bool> {
    if args.len() < 2 || (args[1] != "--prepare-ultrachat" && args[1] != "prepare-ultrachat") {
        return Ok(false);
    }

    let output = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "data/ultrachat_pairs.txt".to_string()),
    );
    let context_window = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(6usize);
    let min_tokens = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(2usize);
    let max_rows = args.get(5).and_then(|v| v.parse().ok());
    let written = prepare_ultrachat_pairs(&output, context_window, min_tokens, max_rows)?;
    println!(
        "Prepared UltraChat pairs: {} rows -> {} (context_window={}, min_tokens={})",
        written,
        output.display(),
        context_window,
        min_tokens
    );
    Ok(true)
}

pub fn try_run_train(args: &[String]) -> Result<bool> {
    if args.len() < 2 {
        return Ok(false);
    }

    match args[1].as_str() {
        "--latent" | "latent" => {
            let data_arg = args.get(2).map(String::as_str).unwrap_or("");
            let resolved = resolve_data_path(data_arg)?;
            let mut args_for_config = args[2..].to_vec();
            if let Some(first) = args_for_config.first_mut() {
                *first = resolved.path.to_string_lossy().into_owned();
            }
            let mut config = LatentTrainConfig::from_args_after(&args_for_config)?;
            config.is_paragraph_data = resolved.is_wikipedia;
            run_latent_training(config)?;
            Ok(true)
        }
        "--latent-from-checkpoint" | "latent-from-checkpoint" => {
            if args.len() < 4 {
                bail!(
                    "usage: --latent-from-checkpoint <encoder_checkpoint.safetensors> <data_path> [steps] ..."
                );
            }
            let init_path = PathBuf::from(&args[2]);
            let resolved = resolve_data_path(&args[3])?;
            let mut args_for_config = args[3..].to_vec();
            if let Some(first) = args_for_config.first_mut() {
                *first = resolved.path.to_string_lossy().into_owned();
            }
            let mut config = LatentTrainConfig::from_args_after(&args_for_config)?;
            config.init_encoder_path = Some(init_path);
            config.is_paragraph_data = resolved.is_wikipedia;
            run_latent_training(config)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub fn try_run_eval(args: &[String]) -> Result<bool> {
    if args.len() < 2 || (args[1] != "--eval-jepa" && args[1] != "eval-jepa") {
        return Ok(false);
    }
    let config = LatentEvalConfig::from_args_after(&args[2..])?;
    run_eval_jepa(config)?;
    Ok(true)
}

fn run_latent_training(config: LatentTrainConfig) -> Result<()> {
    let device =
        Device::new_cuda(0).context("latent training requires an available CUDA device 0")?;
    tracing::info!("using device: CUDA(0)");
    let min_tokens = if config.is_paragraph_data {
        Some(1)
    } else {
        None
    };
    let cached_encoder_vocab = std::env::var("TOFY_ENCODER_VOCAB")
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.exists() && !config.is_paragraph_data);
    let (vocab, vocab_stats, pair_count) = if let Some(vocab_path) = cached_encoder_vocab {
        println!(
            "Preparing latent training input from {:?}: loading cached encoder vocab from {:?}...",
            config.data_path, vocab_path
        );
        let vocab = load_vocab_from_file(&vocab_path)?;
        let pair_count = count_pairs_with_vocab(&config.data_path)?;
        (vocab, None, pair_count)
    } else {
        println!(
            "Preparing latent training input from {:?}: scanning dataset and building encoder vocab...",
            config.data_path
        );
        let (vocab, stats, pair_count) =
            build_vocab_from_pair_file(&config.data_path, config.max_vocab, min_tokens)?;
        (vocab, Some(stats), pair_count)
    };
    println!("Vocab scan complete. Initializing streaming reader...");
    let min_tokens_per_line = min_tokens.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE);
    let use_val_split = pair_count >= LATENT_MIN_PAIRS_FOR_VAL_SPLIT;
    let (split_modulus, exclude_heldout) = if use_val_split {
        (Some(LATENT_HELDOUT_SPLIT_MODULUS), true)
    } else {
        (None, false)
    };
    let mut pair_stream = PairStream::with_split(
        &config.data_path,
        min_tokens_per_line,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        split_modulus,
        LATENT_HELDOUT_SPLIT_REMAINDER,
        exclude_heldout,
    )?;
    let mut val_pair_stream = if use_val_split {
        Some(PairStream::with_split(
            &config.data_path,
            min_tokens_per_line,
            1,
            Some(LATENT_HELDOUT_SPLIT_MODULUS),
            LATENT_HELDOUT_SPLIT_REMAINDER,
            false,
        )?)
    } else {
        None
    };
    let latent_source_budget = config.max_seq.max(1) * config.latent_context_segments.max(1);
    let latent_cache_path = latent_token_cache_path(
        &config.data_path,
        &vocab_signature(&vocab),
        latent_source_budget,
    )?;
    let mut cached_pair_stream = if let Some(cache_path) = latent_cache_path.as_ref() {
        println!("Token cache: using latent encoder cache {:?}", cache_path);
        Some(CachedPairStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            split_modulus,
            LATENT_HELDOUT_SPLIT_REMAINDER,
            exclude_heldout,
        )?)
    } else {
        println!("Token cache: no compatible latent cache found; using raw tokenization stream");
        None
    };
    let mut cached_val_stream = if use_val_split {
        if let Some(cache_path) = latent_cache_path.as_ref() {
            Some(CachedPairStream::with_split(
                cache_path,
                1,
                Some(LATENT_HELDOUT_SPLIT_MODULUS),
                LATENT_HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    if use_val_split {
        println!(
            "Heldout split: 1/{} rows reserved for validation-based checkpoint selection",
            LATENT_HELDOUT_SPLIT_MODULUS
        );
    } else {
        println!(
            "Heldout split: disabled ({} pairs < {}); selecting on train metric",
            pair_count, LATENT_MIN_PAIRS_FOR_VAL_SPLIT
        );
    }
    println!("Streaming reader ready. Building training graph...");
    let vocab_size = vocab.id_to_token.len();
    let seq_len = config.max_seq;

    println!("Training (LeJEPA latent pretraining for code + text)");
    if let Some(ref p) = config.init_encoder_path {
        println!("Encoder init: {:?}", p);
    }
    println!(
        "Vocab size: {} (includes <mask>) | sampled pairs {} | seq_len {} | lambda {:.3}",
        vocab_size, pair_count, seq_len, config.lambda
    );
    if let Some(vocab_stats) = vocab_stats {
        let coverage =
            (vocab_stats.covered_tokens as f64 / vocab_stats.total_tokens as f64) * 100.0;
        println!(
            "Vocab coverage: {:.2}% (covered {} / total {}, OOV {}, unique {}, vocab {})",
            coverage,
            vocab_stats.covered_tokens,
            vocab_stats.total_tokens,
            vocab_stats.oov_tokens,
            vocab_stats.unique_tokens,
            vocab_stats.vocab_size
        );
    } else {
        println!("Vocab coverage: cached vocab loaded; coverage scan skipped.");
    }
    let mut varmap = VarMap::new();
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let vb = VarBuilder::from_varmap(&varmap, train_dtype, &device);

    let encoder = OnlineEncoder::new(
        vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    util::cast_varmap_dtype(&mut varmap, train_dtype)?;
    let latent_params = varmap
        .all_vars()
        .iter()
        .map(|var| var.elem_count())
        .sum::<usize>();
    println!(
        "Exact trainable parameters: {}",
        util::format_params(latent_params)
    );
    println!("{}", encoder.attention_work_summary(seq_len));

    let _ =
        fs::create_dir_all("local_models").and_then(|_| fs::create_dir_all("local_models/vocabs"));
    let model_path = config.output_path.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "local_models/model_latent_{}.safetensors",
            util::format_params(latent_params)
        ))
    });
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let vocab_path = PathBuf::from("local_models/vocabs/vocab_encoder.txt");
    save_vocab_to_file(&vocab, &vocab_path)?;
    println!("Encoder vocab saved before training to {:?}", vocab_path);
    let matched_vocab_path = model_path.with_extension("vocab.txt");
    save_vocab_to_file(&vocab, &matched_vocab_path)?;
    println!(
        "Matched encoder vocab saved before training to {:?}",
        matched_vocab_path
    );
    let resume_stage = util::resume_stage_name("latent");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    let mut resume_metadata = util::ResumeCheckpointMetadata::default();
    if config.resume {
        let loaded_state = util::load_resume_state(&resume_state_path, &resume_stage)?;
        resume_metadata = util::load_resume_checkpoint_metadata(&resume_state_path)?;
        util::validate_resume_checkpoint_tuple(
            loaded_state.as_ref(),
            &resume_metadata,
            &[&train_checkpoint_path],
            &optimizer_checkpoint_path,
        )?;
        if let Some(state) = loaded_state {
            resume_state = state;
        }
        if resume_state.step > 0 {
            util::load_varmap_checked(&mut varmap, &train_checkpoint_path)?;
            util::cast_varmap_dtype(&mut varmap, train_dtype)?;
            println!("Resuming latent weights from {:?}", train_checkpoint_path);
        } else if model_path.exists() {
            bail!(
                "cannot --resume latent training from best export {} without a complete train/optimizer/resume tuple",
                model_path.display()
            );
        }
    }
    if resume_state.step == 0 {
        if let Some(ref init_path) = config.init_encoder_path {
            util::load_varmap_checked(&mut varmap, init_path)?;
            util::cast_varmap_dtype(&mut varmap, train_dtype)?;
            println!("Initialized latent weights from {:?}", init_path);
        }
    }

    let named_train_vars = util::named_train_vars(&varmap)?;
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume && resume_state.step > 0 {
        opt.load_state(&optimizer_checkpoint_path)?;
        if opt.step_t() != resume_state.step {
            bail!(
                "latent optimizer loaded at step {}, expected {}",
                opt.step_t(),
                resume_state.step
            );
        }
        println!(
            "Resuming latent optimizer from {:?} at step {}",
            optimizer_checkpoint_path, resume_state.step
        );
    }
    resume_metadata.validate_and_set_batch_schedule(config.batch_size, config.grad_accum_steps)?;
    let mut best_pred = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume { resume_state.step } else { 0 };
    let mut completed_step = start_step;

    let run_dir = util::create_run_dir("latent")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let async_checkpoints = env_bool("TOFY_LATENT_ASYNC_CHECKPOINTS", true);
    let mut checkpoint_writer = if async_checkpoints {
        Some(util::AsyncCheckpointWriter::new())
    } else {
        None
    };
    println!("LeJEPA: online masked-view prediction + SIGReg");
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "Latent sidecar checkpointing: async={} log_every={}",
        async_checkpoints, config.log_every
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
    tb.add_scalar("config/num_layers", config.num_layers as f32, 0);
    tb.add_scalar("config/num_heads", config.num_heads as f32, 0);
    tb.add_scalar("config/vocab_size", vocab_size as f32, 0);
    tb.add_scalar("config/estimated_params", latent_params as f32, 0);
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
    tb.add_scalar(
        "config/context_segments",
        config.latent_context_segments as f32,
        0,
    );
    tb.add_scalar(
        "config/recent_full_segments",
        config.latent_recent_full_segments as f32,
        0,
    );
    if let Some(vram) = vram_tracker.sample() {
        tb.add_scalar("memory/used_mb", vram.used_mb, 0);
        tb.add_scalar("memory/free_mb", vram.free_mb, 0);
        tb.add_scalar("memory/total_mb", vram.total_mb, 0);
        tb.add_scalar("memory/peak_used_mb", vram.peak_used_mb, 0);
    }
    tb.flush();

    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    tb.add_scalar("config/sigreg_slices", sigreg_slices as f32, 0);
    tb.add_scalar("config/sigreg_points", sigreg_points as f32, 0);
    // Global-norm gradient clipping (0 disables).
    let clip_norm = std::env::var("TOFY_LATENT_CLIP_NORM")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0);
    let val_batches = env_usize("TOFY_LATENT_VAL_BATCHES", 4);
    tb.add_scalar("config/clip_norm", clip_norm as f32, 0);
    println!(
        "Latent SIGReg pools all gradient-accumulation microbatches: {} independent examples, {} three-view rows per optimizer step",
        config.batch_size * config.grad_accum_steps,
        config.batch_size * config.grad_accum_steps * 3
    );
    tb.add_scalar(
        "config/sigreg_independent_samples",
        (config.batch_size * config.grad_accum_steps) as f32,
        0,
    );
    tb.add_scalar(
        "config/sigreg_view_rows",
        (config.batch_size * config.grad_accum_steps * 3) as f32,
        0,
    );
    if start_step >= config.steps {
        println!(
            "Latent resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let batch_size = latent_batch_size_for_step(step, &config);
        let grad_accum_steps = latent_grad_accum_for_step(step, &config);
        let mut log_snapshots = Vec::with_capacity(grad_accum_steps);
        if config.batch_warmup_steps > 0
            && step == config.batch_warmup_steps + 1
            && (config.batch_warmup_value != config.batch_size
                || config.grad_accum_warmup_value < config.grad_accum_steps)
        {
            println!(
                "Latent warmup complete at step {}; switching to batch={} grad_accum={} (effective={})",
                config.batch_warmup_steps,
                batch_size,
                grad_accum_steps,
                batch_size * grad_accum_steps
            );
        }

        let curriculum = latent_curriculum(step, config.steps, &config);
        let mut microbatches = Vec::with_capacity(grad_accum_steps);
        let mut cached_sigreg = Vec::with_capacity(grad_accum_steps);
        for _micro_step in 0..grad_accum_steps {
            let batch = if let Some(ref mut cached_stream) = cached_pair_stream {
                let batch_pairs = cached_stream.next_batch(batch_size)?;
                make_augmented_jepa_batch_from_pairs(&batch_pairs, &vocab, &curriculum, &device)?
            } else {
                let batch_tokens = pair_stream.next_batch(batch_size)?;
                make_augmented_jepa_batch(&batch_tokens, &vocab, &curriculum, &device)?
            };
            let forward = latent_forward(&encoder, &batch, batch_size, &device)?;
            cached_sigreg.push(LatentSigRegBatch::detached_from(&forward));

            if step % config.log_every == 0 {
                let pred_cos = util::scalar_f32(&mean_cosine_similarity(
                    &forward.context_targets,
                    &forward.target_targets,
                )?)?;
                let chunk_cos = util::scalar_f32(&mean_cosine_similarity(
                    &forward.pred_chunks_masked,
                    &forward.target_chunks_masked,
                )?)?;
                let global_cos = util::scalar_f32(&mean_cosine_similarity(
                    &forward.pred_global_flat,
                    &forward.target_global_flat,
                )?)?;
                let context_rms = util::scalar_f32(&tensor_rms(&forward.pred_token_flat)?)?;
                let target_rms = util::scalar_f32(&tensor_rms(&forward.target_flat)?)?;
                let target_count = batch.target_count;
                let valid_tokens = batch.valid_lens.iter().sum::<usize>().max(1);
                let target_frac = target_count as f32 / valid_tokens as f32;
                let pred_val = util::scalar_f32(&forward.pred_loss)?;
                log_snapshots.push(LatentLogSnapshot {
                    loss_val: pred_val,
                    pred_val,
                    token_pred_val: util::scalar_f32(&forward.token_pred_loss)?,
                    chunk_pred_val: util::scalar_f32(&forward.chunk_pred_loss)?,
                    global_pred_val: util::scalar_f32(&forward.global_pred_loss)?,
                    sigreg_val: 0.0,
                    pred_cos,
                    chunk_cos,
                    global_cos,
                    context_rms,
                    target_rms,
                    target_count,
                    target_frac,
                    code_fraction: batch.code_fraction,
                    active_seq: curriculum.active_seq,
                    max_spans_per_sample: curriculum.max_spans_per_sample,
                    min_masked_ratio: curriculum.min_masked_ratio as f32,
                    max_masked_ratio: curriculum.max_masked_ratio as f32,
                    reg_weight: config.lambda as f32,
                });
            }

            let prediction_grads = util::scaled_gradients(&forward.pred_loss, grad_accum_steps)?;
            util::accumulate_gradients(&mut accumulated_grads, &train_vars, prediction_grads)?;
            microbatches.push(batch);
        }

        let cached_tokens = cached_sigreg
            .iter()
            .map(|batch| batch.token_states.clone())
            .collect::<Vec<_>>();
        let cached_chunks = cached_sigreg
            .iter()
            .map(|batch| batch.chunk_states.clone())
            .collect::<Vec<_>>();
        let cached_globals = cached_sigreg
            .iter()
            .map(|batch| batch.global_states.clone())
            .collect::<Vec<_>>();
        let pooled_token_valid_lens = cached_sigreg
            .iter()
            .flat_map(|batch| batch.token_valid_lens.iter().copied())
            .collect::<Vec<_>>();
        let pooled_chunk_valid_lens = cached_sigreg
            .iter()
            .flat_map(|batch| batch.chunk_valid_lens.iter().copied())
            .collect::<Vec<_>>();
        let sigreg_position_chunk = env_usize("TOFY_SIGREG_POSITION_CHUNK", 8);
        let step_seed = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let token_linearization = {
            let refs = cached_tokens.iter().collect::<Vec<_>>();
            let pooled = Tensor::cat(&refs, 0)?;
            sigreg_epps_pulley_variable_length_linearization_chunked_seeded(
                &pooled,
                &pooled_token_valid_lens,
                sigreg_slices,
                sigreg_points,
                sigreg_position_chunk,
                6,
                step_seed ^ 0x544f_4b45_4e00_0001,
            )?
        };
        let chunk_linearization = {
            let refs = cached_chunks.iter().collect::<Vec<_>>();
            let pooled = Tensor::cat(&refs, 0)?;
            sigreg_epps_pulley_variable_length_linearization_chunked_seeded(
                &pooled,
                &pooled_chunk_valid_lens,
                sigreg_slices,
                sigreg_points,
                sigreg_position_chunk,
                6,
                step_seed ^ 0x4348_554e_4b00_0002,
            )?
        };
        let global_linearization = {
            let refs = cached_globals.iter().collect::<Vec<_>>();
            let pooled = Tensor::cat(&refs, 0)?;
            sigreg_epps_pulley_linearization_chunked_seeded(
                &pooled,
                sigreg_slices,
                sigreg_points,
                sigreg_position_chunk,
                step_seed ^ 0x474c_4f42_414c_0003,
            )?
        };
        let pooled_sigreg_value = 0.70 * token_linearization.value
            + 0.20 * chunk_linearization.value
            + 0.10 * global_linearization.value;
        let token_rows = cached_tokens[0].dim(0)?;
        let chunk_rows = cached_chunks[0].dim(0)?;
        let global_rows = cached_globals[0].dim(0)?;
        for (micro_step, batch) in microbatches.iter().enumerate() {
            let live = latent_sigreg_forward(&encoder, batch)?;
            let token_surrogate = sigreg_linear_surrogate(
                &live.token_states,
                &token_linearization.input_gradient,
                micro_step * token_rows,
            )?;
            let chunk_surrogate = sigreg_linear_surrogate(
                &live.chunk_states,
                &chunk_linearization.input_gradient,
                micro_step * chunk_rows,
            )?;
            let global_surrogate = sigreg_linear_surrogate(
                &live.global_states,
                &global_linearization.input_gradient,
                micro_step * global_rows,
            )?;
            let sigreg_surrogate = token_surrogate
                .affine(0.70, 0.0)?
                .broadcast_add(&chunk_surrogate.affine(0.20, 0.0)?)?
                .broadcast_add(&global_surrogate.affine(0.10, 0.0)?)?
                .affine(config.lambda, 0.0)?;
            let sigreg_grads = util::scaled_gradients(&sigreg_surrogate, 1)?;
            util::accumulate_gradients(&mut accumulated_grads, &train_vars, sigreg_grads)?;
        }
        if step % config.log_every == 0 {
            for snapshot in &mut log_snapshots {
                snapshot.sigreg_val = pooled_sigreg_value;
                snapshot.loss_val = snapshot.pred_val + config.lambda as f32 * pooled_sigreg_value;
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
        completed_step = step;

        if step % config.log_every == 0 {
            let snapshot = mean_latent_log_snapshot(&log_snapshots)?;

            tb.add_scalar("loss/total", snapshot.loss_val, step);
            tb.add_scalar("loss/pred", snapshot.pred_val, step);
            tb.add_scalar("loss/pred_token", snapshot.token_pred_val, step);
            tb.add_scalar("loss/pred_chunk", snapshot.chunk_pred_val, step);
            tb.add_scalar("loss/pred_global", snapshot.global_pred_val, step);
            tb.add_scalar("loss/sigreg", snapshot.sigreg_val, step);
            tb.add_scalar("metrics/pred_cosine", snapshot.pred_cos, step);
            tb.add_scalar("metrics/chunk_cosine", snapshot.chunk_cos, step);
            tb.add_scalar("metrics/global_cosine", snapshot.global_cos, step);
            tb.add_scalar("metrics/context_rms", snapshot.context_rms, step);
            tb.add_scalar("metrics/target_rms", snapshot.target_rms, step);
            tb.add_scalar("metrics/target_count", snapshot.target_count as f32, step);
            tb.add_scalar("metrics/target_frac", snapshot.target_frac, step);
            tb.add_scalar("metrics/code_fraction", snapshot.code_fraction, step);
            tb.add_scalar("schedule/active_seq", snapshot.active_seq as f32, step);
            tb.add_scalar(
                "schedule/max_spans",
                snapshot.max_spans_per_sample as f32,
                step,
            );
            tb.add_scalar("schedule/min_masked_ratio", snapshot.min_masked_ratio, step);
            tb.add_scalar("schedule/max_masked_ratio", snapshot.max_masked_ratio, step);
            tb.add_scalar("schedule/reg_weight", snapshot.reg_weight, step);
            tb.add_scalar("schedule/batch_size", batch_size as f32, step);
            tb.add_scalar("schedule/grad_accum", grad_accum_steps as f32, step);
            tb.add_scalar(
                "schedule/effective_batch_size",
                (batch_size * grad_accum_steps) as f32,
                step,
            );
            tb.add_scalar("schedule/lr", scheduled_lr as f32, step);
            if let Some(grad_norm) = grad_norm {
                tb.add_scalar("metrics/grad_norm", util::scalar_f32(&grad_norm)?, step);
            }
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

            // Select checkpoints on held-out validation loss when available,
            // train pred loss otherwise.
            let val_pred = latent_validation_pred_loss(
                &encoder,
                &vocab,
                &latent_curriculum(step, config.steps, &config),
                &device,
                &mut val_pair_stream,
                &mut cached_val_stream,
                batch_size,
                val_batches,
            )?;
            let mut val_note = String::new();
            if let Some(val_pred) = val_pred {
                tb.add_scalar("val/pred", val_pred, step);
                val_note = format!(" val_pred {:.4}", val_pred);
            }
            tb.flush();
            let selection_metric = val_pred.unwrap_or(snapshot.pred_val);

            if selection_metric < best_pred {
                best_pred = selection_metric;
                util::save_varmap_atomic(&varmap, &model_path)?;
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {:.4} pred {:.4}{} tok {:.4} chk {:.4} glb {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4} lr {:.2e}{} [saved best]",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
                    val_note,
                    snapshot.token_pred_val,
                    snapshot.chunk_pred_val,
                    snapshot.global_pred_val,
                    snapshot.sigreg_val,
                    snapshot.pred_cos,
                    snapshot.chunk_cos,
                    snapshot.global_cos,
                    snapshot.target_count,
                    snapshot.code_fraction,
                    snapshot.active_seq,
                    snapshot.reg_weight,
                    scheduled_lr,
                    memory_note,
                );
            } else {
                println!(
                    "step {step}/{} total {:.4} pred {:.4}{} tok {:.4} chk {:.4} glb {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4} lr {:.2e}{}",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
                    val_note,
                    snapshot.token_pred_val,
                    snapshot.chunk_pred_val,
                    snapshot.global_pred_val,
                    snapshot.sigreg_val,
                    snapshot.pred_cos,
                    snapshot.chunk_cos,
                    snapshot.global_cos,
                    snapshot.target_count,
                    snapshot.code_fraction,
                    snapshot.active_seq,
                    snapshot.reg_weight,
                    scheduled_lr,
                    memory_note,
                );
            }
            let checkpoint_resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric: best_pred,
                best_aux_metric: best_pred,
                saved_checkpoint,
                terminal: None,
            };
            let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, step);
            resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
            util::save_checkpoint_job(
                checkpoint_writer.as_ref(),
                format!("latent step {step}"),
                vec![
                    util::varmap_checkpoint_artifact(
                        &varmap,
                        &train_checkpoint_path,
                        &checkpoint_id,
                    )?,
                    util::optimizer_checkpoint_artifact(
                        &opt,
                        &optimizer_checkpoint_path,
                        &checkpoint_id,
                    )?,
                    util::resume_checkpoint_artifact(
                        &checkpoint_resume_state,
                        &resume_metadata,
                        &resume_state_path,
                    )?,
                ],
            )?;
        }
    }

    if let Some(writer) = checkpoint_writer.as_mut() {
        writer.finish()?;
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&varmap, &model_path)?;
        saved_checkpoint = true;
        println!(
            "No checkpoint was saved during logging; saved final encoder weights to {:?}",
            model_path
        );
    }
    let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, completed_step);
    resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
    util::save_varmap_resume_checkpoint_atomic(&varmap, &train_checkpoint_path, &checkpoint_id)?;
    util::save_optimizer_resume_checkpoint_atomic(
        &opt,
        &optimizer_checkpoint_path,
        &checkpoint_id,
    )?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: completed_step,
        best_metric: best_pred,
        best_aux_metric: best_pred,
        saved_checkpoint,
        terminal: Some(util::TrainingTerminal::TargetReached),
    };
    util::save_resume_state_with_metadata(&resume_state_path, &resume_state, &resume_metadata)?;
    tb.flush();
    tb.finish()?;
    let _ = vram_tracker.write_summary(&run_dir, "latent");
    if saved_checkpoint {
        println!(
            "Best model saved to {:?} (pred {:.4})",
            model_path, best_pred
        );
    } else {
        println!(
            "Final model saved to {:?} (run finished before first logging checkpoint)",
            model_path
        );
    }

    save_vocab_to_file(&vocab, &vocab_path)?;
    println!("Encoder vocab saved to {:?}", vocab_path);
    save_vocab_to_file(&vocab, &matched_vocab_path)?;
    println!("Matched encoder vocab saved to {:?}", matched_vocab_path);
    println!("\nTo run JEPA-native evaluation:");
    println!(
        "  cargo run --release -- --eval-jepa {} local_models/vocabs/vocab_encoder.txt <data_path|hub:dataset_id> 200 32 {} {} {} {}",
        model_path.display(),
        config.dim, seq_len, config.num_layers, config.num_heads
    );
    Ok(())
}

fn run_eval_jepa(config: LatentEvalConfig) -> Result<()> {
    let device =
        Device::new_cuda(0).context("JEPA evaluation requires an available CUDA device 0")?;
    let runtime_dtype = util::resolve_runtime_dtype(&device);

    let vocab = load_vocab_from_file(&config.vocab_path)?;
    let data_path = resolve_data_path(&config.data_arg)?.path;
    let pair_count = count_pairs_with_vocab(&data_path)?;
    let mut pair_stream = PairStream::new(&data_path, DEFAULT_MIN_TOKENS_PER_LINE)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, runtime_dtype, &device);

    let vocab_size = vocab.id_to_token.len();
    let encoder = OnlineEncoder::new(
        vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    util::load_varmap_checked(&mut varmap, &config.model_path)?;
    util::cast_varmap_dtype(&mut varmap, runtime_dtype)?;

    let total_params = varmap
        .all_vars()
        .iter()
        .map(|var| var.elem_count())
        .sum::<usize>();
    let embed_params = vocab_size * config.dim;
    let non_embedding_params = total_params.saturating_sub(embed_params);

    println!("LeJEPA evaluation");
    println!("model: {:?}", config.model_path);
    println!("data: {:?}", data_path);
    println!("pairs: {}", pair_count);
    println!(
        "model size: {} [embed {} + hierarchy/predictor {}]",
        util::format_params(total_params),
        util::format_params(embed_params),
        util::format_params(non_embedding_params),
    );
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={}",
        config.eval_steps,
        config.batch_size,
        config.dim,
        config.max_seq,
        config.num_layers,
        config.num_heads
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );

    let mut n_total: usize = 0;
    let mut sum_pred: f64 = 0.0;
    let mut sum_sigreg: f64 = 0.0;
    let mut sum_chunk_cos: f64 = 0.0;
    let mut sum_global_cos: f64 = 0.0;
    let mut sum_rank: f64 = 0.0;
    let mut sum_rr: f64 = 0.0;
    let mut top1: usize = 0;
    let mut top5: usize = 0;
    let mut eval_batches: usize = 0;

    const EVAL_MAX_SPANS: usize = 3;
    const EVAL_MAX_SPAN_LEN: usize = 32;
    const EVAL_MAX_MASKED_RATIO: f64 = 0.25;
    let eval_curriculum = CurriculumDenoisingConfig {
        max_seq: config.max_seq,
        active_seq: config.max_seq,
        max_spans_per_sample: EVAL_MAX_SPANS,
        max_span_len: EVAL_MAX_SPAN_LEN,
        min_masked_ratio: 0.10,
        max_masked_ratio: EVAL_MAX_MASKED_RATIO,
        code_span_multiplier: 1.70,
        identifier_focus_prob: 0.86,
        block_focus_prob: 0.42,
        comment_focus_prob: 0.28,
        text_boundary_focus_prob: 0.44,
        code_masked_ratio_multiplier: 1.30,
        context_segments: 1,
        recent_full_segments: 1,
        history_ratio: 0.0,
    };
    for _ in 0..config.eval_steps {
        let batch_tokens = pair_stream.next_batch(config.batch_size)?;
        let batch_pairs = batch_tokens
            .iter()
            .map(|tokens| Pair {
                tokens: vocab.encode(tokens),
            })
            .collect::<Vec<_>>();
        let batch =
            make_augmented_jepa_batch_from_pairs(&batch_pairs, &vocab, &eval_curriculum, &device)?;
        let forward = latent_forward(&encoder, &batch, config.batch_size, &device)?;
        let online_at_targets = forward.context_targets.clone();
        let target_latents = forward.target_targets.clone();
        let pred_loss = util::scalar_f32(&forward.pred_loss)?;
        // Use the same SIGReg budget as training so eval values are comparable.
        let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
        let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
        let sigreg_position_chunk = env_usize("TOFY_SIGREG_POSITION_CHUNK", 8);
        let token_sigreg = sigreg_epps_pulley_variable_length(
            &forward.valid_token_states,
            &forward.token_valid_lens,
            sigreg_slices,
            sigreg_points,
            sigreg_position_chunk,
            6,
        )?;
        let chunk_sigreg = sigreg_epps_pulley_variable_length(
            &forward.regularized_chunk_states,
            &forward.chunk_valid_lens,
            sigreg_slices,
            sigreg_points,
            sigreg_position_chunk,
            6,
        )?;
        let global_sigreg = sigreg_epps_pulley(
            &forward.regularized_global_states,
            sigreg_slices,
            sigreg_points,
        )?;
        let sigreg_loss = token_sigreg
            .affine(0.70, 0.0)?
            .broadcast_add(&chunk_sigreg.affine(0.20, 0.0)?)?
            .broadcast_add(&global_sigreg.affine(0.10, 0.0)?)?;
        let sigreg_loss = util::scalar_f32(&sigreg_loss)?;
        let chunk_cos = util::scalar_f32(&mean_cosine_similarity(
            &forward.pred_chunks_masked,
            &forward.target_chunks_masked,
        )?)?;
        let global_cos = util::scalar_f32(&mean_cosine_similarity(
            &forward.pred_global_flat,
            &forward.target_global_flat,
        )?)?;
        let tgt_norm = target_latents
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit = (target_latents.clone() / tgt_norm.broadcast_as(target_latents.shape())?)?;
        let pred_norm = online_at_targets
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit =
            (online_at_targets.clone() / pred_norm.broadcast_as(online_at_targets.shape())?)?;
        let n = pred_unit.dim(0)?;
        n_total += n;
        sum_pred += pred_loss as f64;
        sum_sigreg += sigreg_loss as f64;
        sum_chunk_cos += chunk_cos as f64;
        sum_global_cos += global_cos as f64;
        eval_batches += 1;

        let scores = pred_unit
            .clone()
            .matmul(&tgt_unit.clone().transpose(0, 1)?)?;
        let scores_vec = util::vec2_f32(&scores)?;
        for (i, row) in scores_vec.iter().enumerate() {
            if i >= row.len() {
                continue;
            }
            let target_score = row[i];
            let mut gt_count = 0usize;
            for &score in row {
                if score > target_score {
                    gt_count += 1;
                }
            }
            let rank = gt_count + 1;
            sum_rank += rank as f64;
            sum_rr += 1.0 / rank as f64;
            if rank == 1 {
                top1 += 1;
            }
            if rank <= 5 {
                top5 += 1;
            }
        }
    }

    if n_total == 0 {
        bail!("evaluation produced zero targets");
    }

    let denom = n_total as f64;
    println!("\nJEPA metrics over {} targets:", n_total);
    println!(
        "  pred_mse:       {:.4}",
        sum_pred / config.eval_steps.max(1) as f64
    );
    println!(
        "  sigreg:         {:.4}",
        sum_sigreg / config.eval_steps.max(1) as f64
    );
    println!(
        "  chunk_cosine:   {:.4}",
        sum_chunk_cos / eval_batches.max(1) as f64
    );
    println!(
        "  global_cosine:  {:.4}",
        sum_global_cos / eval_batches.max(1) as f64
    );
    println!("  retrieval_top1: {:.4}", top1 as f64 / denom);
    println!("  retrieval_top5: {:.4}", top5 as f64 / denom);
    println!("  retrieval_mrr:  {:.4}", sum_rr / denom);
    println!("  mean_rank:      {:.2}", sum_rank / denom);
    Ok(())
}
