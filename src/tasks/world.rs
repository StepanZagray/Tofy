use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::cli::resolve_data_path;
use crate::config::{
    DecoderTrainConfig, HighWorldTrainConfig, OrchestratorTrainConfig, ServeConfig,
    WorldEvalConfig, WorldTrainConfig,
};
use crate::data::{
    build_vocab_from_raw_world_file_with_mode, count_raw_world_rows, count_raw_world_rows_split,
    count_raw_world_rows_split_with_mode, encode_world_examples, encode_world_examples_with_mode,
    make_decoder_batch, make_world_batch_from_slice, tokenize_for_inference, CachedDecoderStream,
    CachedWorldStream, RawWorldExample, RawWorldStream, TokenizationMode, WorldExample,
    ACTION_CODE, ACTION_DONE, ACTION_FETCH_DOCS, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::encoders::EncoderFeatures;
use crate::model::vocab::vocab_signature;
use crate::model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, tensor_rms, ActionSequenceEncoder,
    ActionStateTransition, CandleCrossAttnDecoder, CodeDecoder, ContextCompressor,
    DecoderArchitecture, DecoderConditioningAdapter, DecoderKind, LlamaCppDecoder,
    LocalDecoderRuntime, MacroActionStateTransition, NextActionClassifier, OnlineEncoder,
    RlmDecoderRuntime, StubLocalDecoder, Vocab,
};
use crate::tasks::world_support::{
    action_cross_entropy, compute_action_metrics, decoder_prediction_metrics,
    decoder_selection_score, encoded_examples_oov_rate, evaluate_decoder_batch,
    evaluate_decoder_cached_batch, evaluate_world_encoded_batch,
    hard_mismatched_conditioning_latent, importance_weight_mask, masked_cross_entropy,
    masked_weighted_cross_entropy, raw_examples_oov_rate, shuffled_conditioning_latent,
    signature_weight_mask, slot_delta_slots, structure_weight_mask, syntax_weight_mask,
    world_selection_score, DecoderBatchMetrics, WorldBatchMetrics,
};
use crate::util;

const HELDOUT_SPLIT_MODULUS: usize = 20;
const HELDOUT_SPLIT_REMAINDER: usize = 0;

type WorldConfig = WorldTrainConfig;

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
}

#[derive(Debug, Deserialize)]
struct CacheTokenManifest {
    kind: String,
    source: CacheManifestSource,
    tokenizer: String,
    max_seq: usize,
    vocab_signature: String,
    token_cache_path: String,
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

fn compatible_decoder_dual_cache_path(
    data_path: &Path,
    max_seq: usize,
    encoder_vocab_sig: &str,
    decoder_vocab_sig: &str,
) -> Result<Option<PathBuf>> {
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
    if manifest.kind != "code_decoder_dual"
        || manifest.tokenizer != TokenizationMode::CodeAware.as_str()
        || manifest.max_seq < max_seq
        || manifest.source.path != data_path.to_string_lossy()
        || manifest.vocab_signature != expected_sig
        || manifest.token_cache_path != cache_path.to_string_lossy()
    {
        return Ok(None);
    }
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
        && config.grad_accum_warmup_value < config.grad_accum_steps
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
    let data_path = resolve_data_path(&args[5])?.path;
    let mut args_for_cfg = vec![
        encoder_model_path.to_string_lossy().to_string(),
        encoder_vocab_path.to_string_lossy().to_string(),
        world_path.to_string_lossy().to_string(),
        data_path.to_string_lossy().to_string(),
    ];
    args_for_cfg.extend(args.iter().skip(6).cloned());
    let cfg = DecoderTrainConfig::from_args_after(&args_for_cfg)?;
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
    let mut cached_world_stream = if let Some(cache_path) = token_cache_path("world") {
        println!("Token cache: using world training cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            &cache_path,
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
        if let Some(cache_path) = token_cache_path("world") {
            Some(CachedWorldStream::with_split(
                &cache_path,
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
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;

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

    let transition_params = 3
        * (config.bridge_dim * config.bridge_dim * 4
            + config.bridge_dim * config.bridge_dim * 2
            + 4 * config.bridge_dim);
    let learned_memory_params =
        2 * config.dim + 2 * (config.dim + 1) + 2 * (config.dim * config.dim + config.dim);
    let planner_params = config.num_latent_tokens * config.dim
        + 2 * (config.dim * config.dim + config.dim * 4 * config.dim)
        + config.dim * config.bridge_dim
        + config.bridge_dim
        + learned_memory_params;
    let action_classifier_hidden = (config.bridge_dim * 2).max(256);
    const ORCH_N: usize = crate::model::action_classifier_head::NUM_ACTIONS;
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
        world_varmap.load(&train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!("Resuming world weights from {:?}", train_checkpoint_path);
    } else if config.resume && model_path.exists() {
        world_varmap.load(&model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!(
            "Resuming world weights from best export {:?} without optimizer state",
            model_path
        );
    }
    if config.train_encoder {
        if config.resume && encoder_train_checkpoint_path.exists() {
            encoder_varmap.load(&encoder_train_checkpoint_path)?;
            util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
            println!(
                "Resuming LeWM encoder weights from {:?}",
                encoder_train_checkpoint_path
            );
        } else if config.resume && encoder_model_path.exists() {
            encoder_varmap.load(&encoder_model_path)?;
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
    let mut opt = util::ResumableAdamW::new_lr_named(named_train_vars, config.lr)?;
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
        "World objective: LeWM next-embedding MSE + lambda * SIGReg | action auxiliary weight {:.2}",
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
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
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
    tb.flush();

    const TARGET_CODE_RATE: f32 = 0.35;
    const TARGET_DONE_RATE: f32 = 0.15;
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    tb.add_scalar("config/sigreg_slices", sigreg_slices as f32, 0);
    tb.add_scalar("config/sigreg_points", sigreg_points as f32, 0);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor =
        env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", context_segments > 1);
    tb.add_scalar(
        "config/post_state_loss_weight",
        world_post_state_loss_weight() as f32,
        0,
    );
    tb.add_scalar(
        "config/rollout_loss_weight",
        world_rollout_loss_weight() as f32,
        0,
    );
    tb.add_scalar("config/rollout_steps", world_rollout_steps() as f32, 0);
    if start_step >= config.steps {
        println!(
            "World resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut last_transition_loss = None;
        let mut last_sigreg_loss = None;
        let mut last_action_loss = None;
        let mut last_inverse_loss = None;
        let mut last_post_state_loss = None;
        let mut last_rollout_loss = None;
        let mut last_loss = None;
        let mut last_action_logits = None;
        let mut last_inverse_logits = None;
        let mut last_action_labels = Vec::new();
        let mut last_pred_next_slots = None;
        let mut last_next_slots = None;
        let mut last_state_slots = None;
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

        for _micro_step in 0..grad_accum_steps {
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

            let transition_loss = prediction_loss(&pred_next_slots, &next_slots)?;
            let post_state_loss = if world_post_state_loss_weight() > 0.0 {
                let post_state_slots = context_slots_from_world_post_state_batch(
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
                prediction_loss(&pred_next_slots, &post_state_slots)?
            } else {
                transition_loss.affine(0.0, 0.0)?
            };
            let rollout_loss =
                rollout_loss_from_batch(&transition, &state_slots, &batch, world_rollout_steps())?
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
            let sigreg_loss = state_sigreg
                .broadcast_add(&next_sigreg)?
                .broadcast_add(&pred_sigreg)?
                .affine(1.0 / 3.0, 0.0)?;

            let action_logits = action_classifier_head.forward(&state_slots)?;
            let action_loss = action_cross_entropy(&action_logits, &action_labels, &device)?;
            let true_delta_slots = slot_delta_slots(&next_slots, &state_slots)?;
            let pred_delta_slots = slot_delta_slots(&pred_next_slots, &state_slots)?;
            let inverse_logits_true = inverse_action_classifier.forward(&true_delta_slots)?;
            let inverse_logits_pred = inverse_action_classifier.forward(&pred_delta_slots)?;
            let inverse_true_loss =
                action_cross_entropy(&inverse_logits_true, &action_labels, &device)?;
            let inverse_pred_loss =
                action_cross_entropy(&inverse_logits_pred, &action_labels, &device)?;
            let inverse_loss = inverse_true_loss
                .broadcast_add(&inverse_pred_loss)?
                .affine(0.5, 0.0)?;
            let loss = transition_loss
                .broadcast_add(&post_state_loss.affine(world_post_state_loss_weight(), 0.0)?)?
                .broadcast_add(&rollout_loss.affine(world_rollout_loss_weight(), 0.0)?)?
                .broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?
                .broadcast_add(&action_loss.affine(config.action_loss_weight, 0.0)?)?
                .broadcast_add(&inverse_loss.affine(inverse_loss_weight, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            last_transition_loss = Some(transition_loss);
            last_sigreg_loss = Some(sigreg_loss);
            last_action_loss = Some(action_loss);
            last_inverse_loss = Some(inverse_loss);
            last_post_state_loss = Some(post_state_loss);
            last_rollout_loss = Some(rollout_loss);
            last_loss = Some(loss);
            last_action_logits = Some(action_logits);
            last_inverse_logits = Some(inverse_logits_pred);
            last_action_labels = action_labels;
            last_pred_next_slots = Some(pred_next_slots);
            last_next_slots = Some(next_slots);
            last_state_slots = Some(state_slots);
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let transition_loss = last_transition_loss
                .context("world grad accumulation produced no transition loss")?;
            let sigreg_loss =
                last_sigreg_loss.context("world grad accumulation produced no sigreg loss")?;
            let action_loss =
                last_action_loss.context("world grad accumulation produced no action loss")?;
            let inverse_loss =
                last_inverse_loss.context("world grad accumulation produced no inverse loss")?;
            let post_state_loss = last_post_state_loss
                .context("world grad accumulation produced no post-state loss")?;
            let rollout_loss =
                last_rollout_loss.context("world grad accumulation produced no rollout loss")?;
            let loss = last_loss.context("world grad accumulation produced no total loss")?;
            let action_logits =
                last_action_logits.context("world grad accumulation produced no action logits")?;
            let inverse_logits = last_inverse_logits
                .context("world grad accumulation produced no inverse action logits")?;
            let pred_next_slots = last_pred_next_slots
                .context("world grad accumulation produced no predicted slots")?;
            let next_slots =
                last_next_slots.context("world grad accumulation produced no next slots")?;
            let state_slots =
                last_state_slots.context("world grad accumulation produced no state slots")?;
            let trans_val = util::scalar_f32(&transition_loss)?;
            let loss_val = util::scalar_f32(&loss)?;
            let sigreg_val = util::scalar_f32(&sigreg_loss)?;
            let act_val = util::scalar_f32(&action_loss)?;
            let inv_val = util::scalar_f32(&inverse_loss)?;
            let post_state_val = util::scalar_f32(&post_state_loss)?;
            let rollout_val = util::scalar_f32(&rollout_loss)?;
            let action_metrics = compute_action_metrics(&action_logits, &last_action_labels)?;
            let inverse_metrics = compute_action_metrics(&inverse_logits, &last_action_labels)?;
            let pred_slots_flat = flatten_latent_slots(&pred_next_slots)?;
            let next_slots_flat = flatten_latent_slots(&next_slots)?;
            let trans_cos =
                util::scalar_f32(&mean_cosine_similarity(&pred_slots_flat, &next_slots_flat)?)?;
            let state_slot_rms = util::scalar_f32(&tensor_rms(&state_slots)?)?;
            let pred_slot_rms = util::scalar_f32(&tensor_rms(&pred_next_slots)?)?;

            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/trans", trans_val, step);
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
            util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
            if config.train_encoder {
                util::save_varmap_atomic(&encoder_varmap, &encoder_train_checkpoint_path)?;
            }
            opt.save_state(&optimizer_checkpoint_path)?;
            resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric,
                best_aux_metric: best_loss,
                saved_checkpoint,
            };
            util::save_resume_state(&resume_state_path, &resume_state)?;
        }
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

fn collect_high_world_macro_batch(
    stream: &mut RawWorldStream,
    vocab: &Vocab,
    batch_size: usize,
    macro_min_len: usize,
    macro_max_len: usize,
) -> Result<HighWorldMacroBatch> {
    let macro_min_len = macro_min_len.max(1);
    let macro_max_len = macro_max_len.max(macro_min_len);
    let span_range = macro_max_len - macro_min_len + 1;
    let mut raw_examples = Vec::with_capacity(batch_size);
    let mut action_sequences = Vec::with_capacity(batch_size);
    for sample_idx in 0..batch_size {
        let span_len = macro_min_len + (sample_idx % span_range);
        let rows = stream.next_batch(span_len)?;
        let first = rows
            .first()
            .context("high-world macro span unexpectedly empty")?;
        let last = rows
            .last()
            .context("high-world macro span unexpectedly empty")?;
        action_sequences.push(rows.iter().map(|row| row.action_label).collect());
        raw_examples.push(RawWorldExample {
            state_text: first.state_text.clone(),
            next_text: last.next_text.clone(),
            action_label: first.action_label,
        });
    }
    Ok(HighWorldMacroBatch {
        examples: encode_world_examples(&raw_examples, vocab),
        action_sequences,
    })
}

fn collect_high_world_macro_batch_cached(
    stream: &mut CachedWorldStream,
    batch_size: usize,
    macro_min_len: usize,
    macro_max_len: usize,
) -> Result<HighWorldMacroBatch> {
    let macro_min_len = macro_min_len.max(1);
    let macro_max_len = macro_max_len.max(macro_min_len);
    let span_range = macro_max_len - macro_min_len + 1;
    let span_lens = (0..batch_size)
        .map(|sample_idx| macro_min_len + (sample_idx % span_range))
        .collect::<Vec<_>>();
    let total_rows = span_lens.iter().sum();
    let rows = stream.next_batch(total_rows)?;
    let mut examples = Vec::with_capacity(batch_size);
    let mut action_sequences = Vec::with_capacity(batch_size);
    let mut offset = 0usize;
    for span_len in span_lens {
        let span = &rows[offset..offset + span_len];
        offset += span_len;
        let first = span
            .first()
            .context("cached high-world macro span unexpectedly empty")?;
        let last = span
            .last()
            .context("cached high-world macro span unexpectedly empty")?;
        action_sequences.push(span.iter().map(|row| row.action_label).collect());
        examples.push(WorldExample {
            state_tokens: first.state_tokens.clone(),
            next_tokens: last.next_tokens.clone(),
            action_label: first.action_label,
        });
    }
    Ok(HighWorldMacroBatch {
        examples,
        action_sequences,
    })
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
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let row_count = count_raw_world_rows(&config.data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &config.data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let mut cached_macro_stream = if let Some(cache_path) = token_cache_path("world") {
        println!("Token cache: using high-world cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            &cache_path,
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
    let vocab_size = encoder_vocab.id_to_token.len();

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
    let context_compressor = ContextCompressor::new(
        world_vb.pp("context_compressor"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    world_varmap.load(&config.world_model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;

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

    let total_params = config.bridge_dim * config.bridge_dim * 8
        + config.bridge_dim * config.macro_max_len
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
        high_varmap.load(&train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut high_varmap, train_dtype)?;
        println!(
            "Resuming high-world weights from {:?}",
            train_checkpoint_path
        );
    } else if config.resume && model_path.exists() {
        high_varmap.load(&model_path)?;
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
    let mut opt = util::ResumableAdamW::new_lr_named(named_train_vars, config.lr)?;
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
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
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
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/macro_min_len", config.macro_min_len as f32, 0);
    tb.add_scalar("config/macro_max_len", config.macro_max_len as f32, 0);
    tb.flush();

    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor =
        env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", context_segments > 1);
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
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
        let mut last_loss = None;
        let mut last_transition_loss = None;
        let mut last_sigreg_loss = None;
        let mut last_pred_slots = None;
        let mut last_target_slots = None;
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

        for _micro_step in 0..grad_accum_steps {
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
            let transition_loss = prediction_loss(&pred_slots, &target_slots)?;
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
            let sigreg_loss = target_sigreg
                .broadcast_add(&pred_sigreg)?
                .affine(0.5, 0.0)?;
            let loss = transition_loss.broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            last_loss = Some(loss);
            last_transition_loss = Some(transition_loss);
            last_sigreg_loss = Some(sigreg_loss);
            last_pred_slots = Some(pred_slots);
            last_target_slots = Some(target_slots);
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let loss = last_loss.context("high-world accumulation produced no loss")?;
            let transition_loss = last_transition_loss
                .context("high-world accumulation produced no transition loss")?;
            let sigreg_loss =
                last_sigreg_loss.context("high-world accumulation produced no sigreg loss")?;
            let pred_slots =
                last_pred_slots.context("high-world accumulation produced no predicted slots")?;
            let target_slots =
                last_target_slots.context("high-world accumulation produced no target slots")?;
            let loss_val = util::scalar_f32(&loss)?;
            let trans_val = util::scalar_f32(&transition_loss)?;
            let sigreg_val = util::scalar_f32(&sigreg_loss)?;
            let cosine = util::scalar_f32(&mean_cosine_similarity(&pred_slots, &target_slots)?)?;
            let pred_rms = util::scalar_f32(&tensor_rms(&pred_slots)?)?;
            let target_rms = util::scalar_f32(&tensor_rms(&target_slots)?)?;
            let selection_metric = trans_val + config.lambda as f32 * sigreg_val;
            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/transition", trans_val, step);
            tb.add_scalar("loss/sigreg", sigreg_val, step);
            tb.add_scalar("metrics/cosine", cosine, step);
            tb.add_scalar("metrics/pred_rms", pred_rms, step);
            tb.add_scalar("metrics/target_rms", target_rms, step);
            tb.add_scalar("metrics/selection", selection_metric, step);
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
            util::save_varmap_atomic(&high_varmap, &train_checkpoint_path)?;
            opt.save_state(&optimizer_checkpoint_path)?;
            resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric,
                best_aux_metric: best_metric,
                saved_checkpoint,
            };
            util::save_resume_state(&resume_state_path, &resume_state)?;
        }
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
    let mut cached_train_stream = if let Some(cache_path) = token_cache_path("world") {
        println!(
            "Token cache: using action_classifier world cache {:?}",
            cache_path
        );
        Some(CachedWorldStream::with_split(
            &cache_path,
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
        if let Some(cache_path) = token_cache_path("world") {
            Some(CachedWorldStream::with_split(
                &cache_path,
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

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        encoder_vocab.id_to_token.len(),
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;

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
    world_varmap.load(&config.world_model_path)?;
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
        world_varmap.load(&train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!(
            "Resuming action_classifier weights from {:?}",
            train_checkpoint_path
        );
    }

    let named_train_vars = util::named_train_vars(&world_varmap)?;
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::ResumableAdamW::new_lr_named(named_train_vars, config.lr)?;
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
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
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
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor =
        env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", context_segments > 1);
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
        let mut last_action_loss = None;
        let mut last_action_logits = None;
        let mut last_action_labels = Vec::new();

        for _micro_step in 0..config.grad_accum_steps.max(1) {
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
                config.grad_accum_steps,
            )?;

            last_action_loss = Some(action_loss);
            last_action_logits = Some(action_logits);
            last_action_labels = action_labels;
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let action_loss = last_action_loss
                .context("action_classifier grad accumulation produced no action loss")?;
            let action_logits = last_action_logits
                .context("action_classifier grad accumulation produced no action logits")?;
            let metrics = compute_action_metrics(&action_logits, &last_action_labels)?;
            let action_loss_val = util::scalar_f32(&action_loss)?;
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
            util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
            opt.save_state(&optimizer_checkpoint_path)?;
            resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric: best_score,
                best_aux_metric: best_score,
                saved_checkpoint,
            };
            util::save_resume_state(&resume_state_path, &resume_state)?;
        }
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

fn default_decoder_vocab_path(decoder_path: &Path) -> PathBuf {
    decoder_path.with_extension("vocab.txt")
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
        return if rollout_steps <= 1 {
            transition.forward(&state_slots, &decoder_action_labels)
        } else {
            rollout_transition_slots(
                transition,
                &state_slots,
                decoder_action_label,
                rollout_steps,
            )
        };
    }

    let mut slots_by_row: Vec<Option<Tensor>> = (0..encoder_batch.len()).map(|_| None).collect();
    let mut miss_positions = Vec::new();
    let mut miss_token_refs = Vec::new();

    for (idx, row) in encoder_batch.iter().enumerate() {
        if let Some(slots) = cache.get(&row.state_tokens) {
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
        let next_slots = if rollout_steps <= 1 {
            transition.forward(&state_slots, &decoder_action_labels)?
        } else {
            rollout_transition_slots(
                transition,
                &state_slots,
                decoder_action_label,
                rollout_steps,
            )?
        };
        for (miss_idx, row_idx) in miss_positions.iter().copied().enumerate() {
            let row_slots = next_slots.narrow(0, miss_idx, 1)?.detach();
            cache.insert(
                encoder_batch[row_idx].state_tokens.clone(),
                row_slots.clone(),
            );
            slots_by_row[row_idx] = Some(row_slots);
        }
    }

    let slots = slots_by_row
        .iter()
        .map(|slots| slots.as_ref().context("decoder context cache missing row"))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&slots, 0).map_err(Into::into)
}

pub(crate) fn decoder_tokenization_mode(kind: DecoderKind) -> TokenizationMode {
    if kind == DecoderKind::CodeSpecialist {
        TokenizationMode::CodeAware
    } else {
        TokenizationMode::Default
    }
}

// world/decoder metric helpers live in tasks/world_support.rs

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

    for _ in 0..8 {
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

    for _ in 0..8 {
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
    let row_count = count_raw_world_rows_split_with_mode(&data_path, decoder_token_mode, None, 0)?;
    let val_row_count = count_raw_world_rows_split_with_mode(
        &data_path,
        decoder_token_mode,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let mut raw_stream = RawWorldStream::with_split_mode(
        &data_path,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        decoder_token_mode,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
        true,
    )?;
    let mut val_stream = if val_row_count > 0 {
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
    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        encoder_vocab.id_to_token.len(),
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;

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
    let _inverse_action_classifier =
        if checkpoint_has_prefix(&config.world_model_path, "inverse_action_classifier.") {
            Some(NextActionClassifier::new(
                world_vb.pp("inverse_action_classifier"),
                config.bridge_dim,
            )?)
        } else {
            None
        };
    world_varmap.load(&config.world_model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;

    let mut decoder_varmap = VarMap::new();
    let decoder_vb = VarBuilder::from_varmap(&decoder_varmap, train_dtype, &device);
    let decoder_conditioning_adapter = DecoderConditioningAdapter::new(
        decoder_vb.pp("decoder_conditioning_adapter"),
        config.bridge_dim,
        config.bridge_dim,
        DecoderConditioningAdapter::output_slots_for(config.decoder_kind, config.num_latent_tokens),
    )?;
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
        (load_vocab_from_file(&decoder_vocab_path)?, false)
    } else {
        let (vocab, _, _) = build_vocab_from_raw_world_file_with_mode(
            &data_path,
            config.decoder_max_vocab,
            decoder_token_mode,
        )?;
        (vocab, true)
    };
    let decoder_vocab_sig = vocab_signature(&decoder_vocab);
    let decoder_cache_path = if config.decoder_kind == DecoderKind::CodeSpecialist {
        compatible_decoder_dual_cache_path(
            &data_path,
            config.max_seq,
            &encoder_vocab_sig,
            &decoder_vocab_sig,
        )?
    } else {
        None
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
    util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        decoder_varmap.load(&train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!("Resuming decoder weights from {:?}", train_checkpoint_path);
    } else if config.resume && decoder_path.exists() {
        decoder_varmap.load(&decoder_path)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!(
            "Resuming decoder weights from best export {:?} without optimizer state",
            decoder_path
        );
    } else if let Some(ref p) = config.init_decoder_path {
        decoder_varmap.load(p)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
        println!("Initialized decoder weights from {:?}", p);
    }

    let named_train_vars = util::named_train_vars(&decoder_varmap)?;
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::ResumableAdamW::new_lr_named(named_train_vars, config.lr)?;
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
    println!("Save path: {:?}", decoder_path);
    println!("Decoder vocab path: {:?}", decoder_vocab_path);
    if let Some(parent) = decoder_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if built_decoder_vocab {
        save_vocab_to_file(&decoder_vocab, &decoder_vocab_path)?;
    }

    let run_dir = util::create_run_dir("decoder")?;
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    let conditioning_loss_weight = config.conditioning_loss_weight;
    let compute_conditioning_metrics =
        conditioning_loss_weight > 0.0 || env_bool("TOFY_DECODER_ABLATION_METRICS", false);
    let conditioning_margin = config.conditioning_margin;
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

    let mut best_loss = resume_state.best_aux_metric;
    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let decoder_action_label = if config.decoder_kind == DecoderKind::TextGeneralist {
        0
    } else {
        1
    };
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor =
        env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", context_segments > 1);
    let rollout_steps = env_usize("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", 1);
    let mut decoder_context_cache = DecoderContextCache::from_env();
    tb.add_scalar(
        "config/decoder_context_cache_rows",
        decoder_context_cache.capacity as f32,
        0,
    );
    tb.add_scalar(
        "config/decoder_prefill_batch_rows",
        (config.batch_size * config.grad_accum_steps.max(1)) as f32,
        0,
    );

    if start_step >= config.steps {
        println!(
            "Decoder resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut last_oov_rate = None;
        let mut last_world_latent = None;
        let mut last_logits = None;
        let mut last_dec_input = None;
        let mut last_dec_target = None;
        let mut last_loss_mask = None;
        let mut last_loss = None;
        let mut last_conditioning_loss_val = 0.0f32;
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
        let mut micro_batches = Vec::with_capacity(grad_accum_steps);
        let mut prefill_encoder_batch = Vec::with_capacity(batch_size * grad_accum_steps);

        for _micro_step in 0..grad_accum_steps {
            let (encoder_batch, decoder_batch, oov_rate) = if let Some(ref mut cached_stream) =
                cached_decoder_stream
            {
                let cached_batch = cached_stream.next_batch(batch_size.max(1))?;
                let encoder_batch = cached_batch
                    .iter()
                    .map(|row| row.encoder.clone())
                    .collect::<Vec<_>>();
                let decoder_batch = cached_batch
                    .iter()
                    .map(|row| row.decoder.clone())
                    .collect::<Vec<_>>();
                let oov_rate = encoded_examples_oov_rate(&decoder_batch, decoder_vocab.unk_id);
                (encoder_batch, decoder_batch, oov_rate)
            } else {
                let raw_batch = raw_stream.next_batch(batch_size.max(1))?;
                let encoder_batch = encode_world_examples(&raw_batch, &encoder_vocab);
                let decoder_batch =
                    encode_world_examples_with_mode(&raw_batch, &decoder_vocab, decoder_token_mode);
                let oov_rate =
                    raw_examples_oov_rate(&raw_batch, &decoder_vocab, decoder_token_mode);
                (encoder_batch, decoder_batch, oov_rate)
            };
            prefill_encoder_batch.extend(encoder_batch.iter().cloned());
            micro_batches.push((encoder_batch, decoder_batch, oov_rate));
        }

        let next_context_slots = decoder_next_context_slots(
            &mut decoder_context_cache,
            &encoder,
            &context_compressor,
            &transition,
            &prefill_encoder_batch,
            decoder_action_label,
            encoder_vocab.pad_id,
            config.max_seq,
            context_segments,
            recent_full_segments,
            recursive_context_compressor,
            rollout_steps,
            &device,
        )?
        .detach();

        let mut row_offset = 0usize;
        for (encoder_batch, decoder_batch, oov_rate) in micro_batches {
            let micro_next_context_slots =
                next_context_slots.narrow(0, row_offset, encoder_batch.len())?;
            row_offset += encoder_batch.len();
            let world_latent = decoder_conditioning_adapter
                .forward_with_action(&micro_next_context_slots, decoder_action_label)?;

            let (dec_state_ids, dec_next_ids, state_lens, next_lens, _) =
                make_world_batch_from_slice(
                    &decoder_batch,
                    config.max_seq,
                    decoder_vocab.pad_id,
                    &device,
                )?;
            let (dec_input, dec_target, loss_mask) = make_decoder_batch(
                &dec_state_ids,
                &dec_next_ids,
                &state_lens,
                &next_lens,
                config.max_seq,
                decoder_vocab.pad_id,
                &device,
            )?;

            let logits = decoder.forward(&dec_input, &world_latent)?;
            let importance_mask =
                importance_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let token_loss = masked_weighted_cross_entropy(&logits, &dec_target, &importance_mask)?;
            let mut loss = token_loss.clone();
            if config.syntax_loss_weight > 0.0 {
                let syntax_mask =
                    syntax_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
                let syntax_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &syntax_mask)?;
                loss = loss.broadcast_add(&syntax_loss.affine(config.syntax_loss_weight, 0.0)?)?;
            }
            if config.signature_loss_weight > 0.0 {
                let signature_mask =
                    signature_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
                let signature_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &signature_mask)?;
                loss =
                    loss.broadcast_add(&signature_loss.affine(config.signature_loss_weight, 0.0)?)?;
            }
            if config.structure_loss_weight > 0.0 {
                let structure_mask =
                    structure_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
                let structure_loss =
                    masked_weighted_cross_entropy(&logits, &dec_target, &structure_mask)?;
                loss =
                    loss.broadcast_add(&structure_loss.affine(config.structure_loss_weight, 0.0)?)?;
            }
            let conditioning_loss = if conditioning_loss_weight > 0.0 {
                let zero_world_latent = world_latent.affine(0.0, 0.0)?;
                let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
                let ablated_loss = masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
                let shuffled_world_latent = shuffled_conditioning_latent(&world_latent)?;
                let shuffled_logits = decoder.forward(&dec_input, &shuffled_world_latent)?;
                let shuffled_loss =
                    masked_cross_entropy(&shuffled_logits, &dec_target, &loss_mask)?;
                let hard_mismatch_world_latent =
                    hard_mismatched_conditioning_latent(&world_latent)?;
                let hard_mismatch_logits =
                    decoder.forward(&dec_input, &hard_mismatch_world_latent)?;
                let hard_mismatch_loss =
                    masked_cross_entropy(&hard_mismatch_logits, &dec_target, &loss_mask)?;
                let zero_margin_loss = token_loss
                    .broadcast_sub(&ablated_loss.detach())?
                    .affine(1.0, conditioning_margin)?
                    .relu()?;
                let shuffle_margin_loss = token_loss
                    .broadcast_sub(&shuffled_loss.detach())?
                    .affine(1.0, conditioning_margin)?
                    .relu()?;
                let hard_margin_loss = token_loss
                    .broadcast_sub(&hard_mismatch_loss.detach())?
                    .affine(1.0, conditioning_margin)?
                    .relu()?;
                zero_margin_loss
                    .broadcast_add(&shuffle_margin_loss)?
                    .broadcast_add(&hard_margin_loss)?
                    .affine(1.0 / 3.0, 0.0)?
            } else {
                token_loss.affine(0.0, 0.0)?
            };
            if conditioning_loss_weight > 0.0 {
                last_conditioning_loss_val = util::scalar_f32(&conditioning_loss)?;
                loss =
                    loss.broadcast_add(&conditioning_loss.affine(conditioning_loss_weight, 0.0)?)?;
            } else {
                last_conditioning_loss_val = 0.0;
            }

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            last_oov_rate = Some(oov_rate);
            last_world_latent = Some(world_latent);
            last_logits = Some(logits);
            last_dec_input = Some(dec_input);
            last_dec_target = Some(dec_target);
            last_loss_mask = Some(loss_mask);
            last_loss = Some(loss);
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let oov_rate =
                last_oov_rate.context("decoder grad accumulation produced no OOV rate")?;
            let world_latent =
                last_world_latent.context("decoder grad accumulation produced no latent")?;
            let logits = last_logits.context("decoder grad accumulation produced no logits")?;
            let dec_input =
                last_dec_input.context("decoder grad accumulation produced no inputs")?;
            let dec_target =
                last_dec_target.context("decoder grad accumulation produced no targets")?;
            let loss_mask =
                last_loss_mask.context("decoder grad accumulation produced no loss mask")?;
            let loss = last_loss.context("decoder grad accumulation produced no loss")?;
            let loss_val = util::scalar_f32(&loss)?;
            let raw_loss = masked_cross_entropy(&logits, &dec_target, &loss_mask)?;
            let raw_loss_val = util::scalar_f32(&raw_loss)?;
            let (ablated_loss_val, shuffled_loss_val, hard_mismatch_loss_val) =
                if compute_conditioning_metrics {
                    let zero_world_latent = world_latent.affine(0.0, 0.0)?;
                    let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
                    let ablated_loss =
                        masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
                    let shuffled_world_latent = shuffled_conditioning_latent(&world_latent)?;
                    let shuffled_logits = decoder.forward(&dec_input, &shuffled_world_latent)?;
                    let shuffled_loss =
                        masked_cross_entropy(&shuffled_logits, &dec_target, &loss_mask)?;
                    let hard_mismatch_world_latent =
                        hard_mismatched_conditioning_latent(&world_latent)?;
                    let hard_mismatch_logits =
                        decoder.forward(&dec_input, &hard_mismatch_world_latent)?;
                    let hard_mismatch_loss =
                        masked_cross_entropy(&hard_mismatch_logits, &dec_target, &loss_mask)?;
                    (
                        util::scalar_f32(&ablated_loss)?,
                        util::scalar_f32(&shuffled_loss)?,
                        util::scalar_f32(&hard_mismatch_loss)?,
                    )
                } else {
                    (loss_val, loss_val, loss_val)
                };
            let conditioning_loss_val = last_conditioning_loss_val;
            let syntax_mask = syntax_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let syntax_loss = masked_weighted_cross_entropy(&logits, &dec_target, &syntax_mask)?;
            let signature_mask =
                signature_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let signature_loss =
                masked_weighted_cross_entropy(&logits, &dec_target, &signature_mask)?;
            let structure_mask =
                structure_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let structure_loss =
                masked_weighted_cross_entropy(&logits, &dec_target, &structure_mask)?;
            let syntax_loss_val = util::scalar_f32(&syntax_loss)?;
            let signature_loss_val = util::scalar_f32(&signature_loss)?;
            let structure_loss_val = util::scalar_f32(&structure_loss)?;
            let active_tokens = util::scalar_f32(&loss_mask.sum_all()?)?;
            let total_tokens = (config.batch_size.max(1) * config.max_seq * 2) as f32;
            let active_frac = active_tokens / total_tokens.max(1.0);
            let perplexity = loss_val.exp();
            let conditioning_gain = if compute_conditioning_metrics {
                ablated_loss_val - loss_val
            } else {
                0.0
            };
            let zero_gain = conditioning_gain;
            let shuffle_gain = if compute_conditioning_metrics {
                shuffled_loss_val - loss_val
            } else {
                0.0
            };
            let hard_negative_gain = if compute_conditioning_metrics {
                ablated_loss_val
                    .min(shuffled_loss_val)
                    .min(hard_mismatch_loss_val)
                    - loss_val
            } else {
                0.0
            };
            let world_rms = util::scalar_f32(&tensor_rms(&world_latent)?)?;
            let prediction_metrics =
                decoder_prediction_metrics(&logits, &dec_target, &loss_mask, &decoder_vocab)?;
            let token_accuracy = prediction_metrics.token_accuracy;
            let identifier_accuracy = prediction_metrics.identifier_accuracy;
            let delimiter_balance_rate = prediction_metrics.delimiter_balance_rate;
            let syntax_token_accuracy = prediction_metrics.syntax_token_accuracy;
            let function_skeleton_rate = prediction_metrics.function_skeleton_rate;
            let signature_token_accuracy = prediction_metrics.signature_token_accuracy;
            let signature_exact_rate = prediction_metrics.signature_exact_rate;
            let function_name_token_accuracy = prediction_metrics.function_name_token_accuracy;
            let function_name_exact_rate = prediction_metrics.function_name_exact_rate;

            tb.add_scalar("loss/token_ce", loss_val, step);
            tb.add_scalar("loss/raw_token_ce", raw_loss_val, step);
            tb.add_scalar("loss/ablated_token_ce", ablated_loss_val, step);
            tb.add_scalar("loss/shuffled_token_ce", shuffled_loss_val, step);
            tb.add_scalar("loss/hard_mismatch_token_ce", hard_mismatch_loss_val, step);
            tb.add_scalar("loss/conditioning_margin", conditioning_loss_val, step);
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
            let train_metrics = DecoderBatchMetrics {
                loss: loss_val,
                raw_loss: raw_loss_val,
                ablated_loss: ablated_loss_val,
                conditioning_gain,
                zero_gain,
                shuffled_loss: shuffled_loss_val,
                shuffle_gain,
                hard_negative_gain,
                syntax_loss: syntax_loss_val,
                signature_loss: signature_loss_val,
                structure_loss: structure_loss_val,
                perplexity,
                active_tokens,
                active_frac,
                world_rms,
                oov_rate,
                token_accuracy,
                identifier_accuracy,
                delimiter_balance_rate,
                syntax_token_accuracy,
                function_skeleton_rate,
                signature_token_accuracy,
                signature_exact_rate,
                function_name_token_accuracy,
                function_name_exact_rate,
            };
            let mut selection_metric = decoder_selection_score(
                &train_metrics,
                config.syntax_loss_weight,
                config.signature_loss_weight,
                config.structure_loss_weight,
            );
            if val_stream.is_some() || cached_decoder_val_stream.is_some() {
                let val_metrics = if let Some(ref mut cached_stream) = cached_decoder_val_stream {
                    let cached_batch = cached_stream.next_batch(config.batch_size.max(1))?;
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
                        config.max_seq,
                        &device,
                    )?
                } else {
                    let val_stream = val_stream
                        .as_mut()
                        .context("decoder validation stream missing")?;
                    let val_raw_batch = val_stream.next_batch(config.batch_size.max(1))?;
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
                best_loss = best_loss.min(val_metrics.loss);
                tb.add_scalar("val/token_ce", val_metrics.loss, step);
                tb.add_scalar("val/raw_token_ce", val_metrics.raw_loss, step);
                tb.add_scalar("val/ablated_token_ce", val_metrics.ablated_loss, step);
                tb.add_scalar("val/shuffled_token_ce", val_metrics.shuffled_loss, step);
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
            } else {
                best_loss = best_loss.min(loss_val);
            }
            tb.flush();
            if selection_metric < best_metric {
                best_metric = selection_metric;
                util::save_varmap_atomic(&decoder_varmap, &decoder_path)?;
                saved_checkpoint = true;
                println!(
                    "step {}/{} token_ce {:.4} ablate_ce {:.4} shuffle_ce {:.4} zero_gain {:.4} shuffle_gain {:.4} hard_gain {:.4} cond_loss {:.4} syntax_ce {:.4} sig_ce {:.4} struct_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% fn_name {:.2}% fn_name_exact {:.2}% delim {:.2}% fn_skel {:.2}% sel {:.4}{} [saved best]",
                    step,
                    config.steps,
                    loss_val,
                    ablated_loss_val,
                    shuffled_loss_val,
                    zero_gain,
                    shuffle_gain,
                    hard_negative_gain,
                    conditioning_loss_val,
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
                    selection_metric,
                    memory_note
                );
            } else {
                println!(
                    "step {}/{} token_ce {:.4} ablate_ce {:.4} shuffle_ce {:.4} zero_gain {:.4} shuffle_gain {:.4} hard_gain {:.4} cond_loss {:.4} syntax_ce {:.4} sig_ce {:.4} struct_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% fn_name {:.2}% fn_name_exact {:.2}% delim {:.2}% fn_skel {:.2}% sel {:.4}{}",
                    step,
                    config.steps,
                    loss_val,
                    ablated_loss_val,
                    shuffled_loss_val,
                    zero_gain,
                    shuffle_gain,
                    hard_negative_gain,
                    conditioning_loss_val,
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
                    selection_metric,
                    memory_note
                );
            }
            util::save_varmap_atomic(&decoder_varmap, &train_checkpoint_path)?;
            opt.save_state(&optimizer_checkpoint_path)?;
            resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric,
                best_aux_metric: best_loss,
                saved_checkpoint,
            };
            util::save_resume_state(&resume_state_path, &resume_state)?;
        }
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
    encoder_varmap.load(&config.encoder_model_path)?;
    util::cast_varmap_dtype(&mut encoder_varmap, runtime_dtype)?;
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, runtime_dtype, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        encoder_vocab.id_to_token.len(),
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;

    let mut world_varmap = VarMap::new();
    world_varmap.load(&config.model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, runtime_dtype)?;
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
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor =
        env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", context_segments > 1);
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
    bridge_dim: usize,
    num_latent_tokens: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_context_compressor: bool,
    world_rollout_steps: usize,
    high_macro_max_len: usize,
}

/// Returns true if the safetensors file contains any tensor whose name starts with `prefix`.
fn checkpoint_has_prefix(model_path: &PathBuf, prefix: &str) -> bool {
    use candle_core::safetensors::MmapedSafetensors;
    let Ok(mapped) = (unsafe { MmapedSafetensors::new(model_path) }) else {
        return false;
    };
    mapped.tensors().iter().any(|(n, _)| n.starts_with(prefix))
}

fn likely_rust_request(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    lower.contains("rust")
        || lower.contains("pub fn")
        || lower.contains("fn ")
        || lower.contains("implement exactly this function")
        || lower.contains("return only rust code")
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
            || likely_rust_request(prompt);
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

fn augment_prompt_with_rust_docs(prompt: &str) -> String {
    if prompt.contains("<ctx:rust_docs>") {
        return prompt.to_string();
    }
    let docs = crate::tasks::rust_docs::retrieve_rust_docs(prompt, 5, 2600);
    if docs.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt}\n\n{docs}")
    }
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
    let angle_noise = trimmed.matches('<').count() + trimmed.matches('>').count();
    if !lower.contains("fn ") && prompt.to_ascii_lowercase().contains("fn") {
        return true;
    }
    if !balanced_braces(trimmed) {
        return true;
    }
    angle_noise >= 6 && !lower.contains("fn ")
}

fn maybe_repair_code_output(
    decoder: &dyn LocalDecoderRuntime,
    prompt: &str,
    action: &str,
    cond_vec: &[f32],
    chunk_tokens: usize,
    initial: String,
) -> String {
    let repair_passes = std::env::var("TOFY_CODE_REPAIR_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);
    let mut assistant_content = initial;
    for _ in 0..repair_passes {
        if !output_needs_code_repair(prompt, &assistant_content) {
            break;
        }
        let repair_prompt = build_code_repair_prompt(prompt, &assistant_content);
        let repaired = match decoder.generate(&repair_prompt, action, cond_vec, chunk_tokens) {
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

fn build_code_repair_prompt(prompt: &str, attempt: &str) -> String {
    let mut out = String::from(
        "<action:repair_patch>\n<tool:read_error>\n<tool:repair_patch>\nReturn only corrected Rust code.\nFix the previous attempt while keeping the exact requested function name and signature.\n\n<ctx:original_request>\nOriginal request:\n",
    );
    out.push_str(prompt);
    out.push_str("\n\n<ctx:previous_attempt>\nPrevious attempt:\n```rust\n");
    out.push_str(attempt);
    out.push_str("\n```\n");
    out.push_str(
        "\n<ctx:constraints>\nRules:\n- Return only compilable Rust code.\n- Do not add explanation.\n",
    );
    out
}

impl AgentEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        encoder_model_path: &PathBuf,
        encoder_vocab_path: &PathBuf,
        world_model_path: &PathBuf,
        high_world_model_path: Option<&PathBuf>,
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
        encoder_varmap.load(encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, runtime_dtype)?;
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, runtime_dtype, &device);
        let encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            dim,
            num_layers,
            num_heads,
        )?;

        let mut world_varmap = VarMap::new();
        world_varmap.load(world_model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, runtime_dtype)?;
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
        let explicit_high_world_model_path = high_world_model_path.cloned();
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
                high_varmap.load(path)?;
                util::cast_varmap_dtype(&mut high_varmap, runtime_dtype)?;
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
                (
                    Some(high_varmap),
                    Some(macro_encoder),
                    Some(macro_transition),
                )
            } else {
                (None, None, None)
            };
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
            bridge_dim,
            num_latent_tokens,
            context_segments: std::env::var("TOFY_ENCODER_CONTEXT_SEGMENTS")
                .ok()
                .or_else(|| std::env::var("TOFY_WORLD_CONTEXT_SEGMENTS").ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            recent_full_segments: std::env::var("TOFY_ENCODER_RECENT_FULL_SEGMENTS")
                .ok()
                .or_else(|| std::env::var("TOFY_WORLD_RECENT_FULL_SEGMENTS").ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            recursive_context_compressor: env_bool(
                "TOFY_RECURSIVE_CONTEXT_COMPRESSION",
                std::env::var("TOFY_ENCODER_CONTEXT_SEGMENTS")
                    .ok()
                    .or_else(|| std::env::var("TOFY_WORLD_CONTEXT_SEGMENTS").ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1usize)
                    > 1,
            ),
            world_rollout_steps: std::env::var("TOFY_WORLD_ROLLOUT_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            high_macro_max_len,
        })
    }

    /// Encode current prompt into private context compressor slots.
    fn encode_prompt_context_compressor(&self, current_prompt: &str) -> Result<Tensor> {
        let tokens = tokenize_for_inference(current_prompt);
        if tokens.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let ids = self.encoder_vocab.encode(&tokens);
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
            Action::FetchDocs => crate::tasks::orchestrator::rust_docs_request_score(prompt),
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
        for (idx, action) in actions.iter().enumerate() {
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
        let denom = actions.len().max(1) as f32;
        let score = cfg.goal_weight * goal_loss
            + cfg.smoothness_weight * (smoothness / denom)
            + premature_done_penalty
            - cfg.route_weight * (route_reward / denom)
            - cfg.prior_weight * (prior_reward / denom);
        Ok(LeWmPlan {
            actions: actions.to_vec(),
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
        for (idx, action) in actions.iter().enumerate() {
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
        let denom = actions.len().max(1) as f32;
        let score = hwm_cfg.subgoal_weight * subgoal_loss
            + lewm_cfg.smoothness_weight * (smoothness / denom)
            - lewm_cfg.route_weight * (route_reward / denom)
            - lewm_cfg.prior_weight * (prior_reward / denom);
        Ok(LeWmPlan {
            actions: actions.to_vec(),
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
        let explicit_code_decoder = std::env::var("JEPA_USE_CANDLE_DECODER")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let code_decoder = match CandleCrossAttnDecoder::try_new_from_env_code(
            self.bridge_dim,
            self.num_latent_tokens,
        ) {
            Ok(decoder) => Some(decoder),
            Err(err) if explicit_code_decoder => {
                anyhow::bail!("explicit Candle code decoder failed to load: {err:#}")
            }
            Err(_) => None,
        };
        let text_decoder =
            CandleCrossAttnDecoder::try_new_from_env_text(self.bridge_dim, self.num_latent_tokens)
                .ok();
        let (decoder, mut cond_vec): (Box<dyn LocalDecoderRuntime>, Vec<f32>) =
            if action == crate::tasks::orchestrator::Action::Code {
                if let Some(d) = code_decoder {
                    (Box::new(d), planner_vec.clone())
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
                if let Some(d) = text_decoder {
                    (Box::new(d), planner_vec.clone())
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

    /// Single-step reply: choose decoder mode once, then generate with that decoder.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
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
        let mut generation_prompt = prompt.to_string();
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
        if fetched_docs_action {
            generation_prompt = augment_prompt_with_rust_docs(prompt);
        }
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
        let mut assistant_content = decoder.generate(
            &generation_prompt,
            action.as_str(),
            &cond_vec,
            decoder_tokens,
        )?;
        if action == Action::Code && likely_rust_request(&generation_prompt) {
            assistant_content = maybe_repair_code_output(
                decoder.as_ref(),
                &generation_prompt,
                action.as_str(),
                &cond_vec,
                decoder_tokens,
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

    pub fn uses_rust_docs(&self, prompt: &str, action: crate::tasks::orchestrator::Action) -> bool {
        action == crate::tasks::orchestrator::Action::FetchDocs
            || (action == crate::tasks::orchestrator::Action::Code
                && crate::tasks::orchestrator::rust_docs_request_score(prompt) >= 2.0
                && crate::tasks::rust_docs::rust_docs_enabled())
    }

    /// Stream generated text in chunks (for SSE). The action_classifier chooses a single decoder mode for the reply.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let start = Instant::now();
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
        let mut generation_prompt = prompt.to_string();
        let mut next_slots = plan.planned_slots;
        if fetched_docs_action {
            on_chunk("Fetching Rust docs. ");
            next_slots = self
                .transition
                .forward_one(&next_slots, Action::Code as u32)?;
            action = Action::Code;
        }
        match action {
            Action::TextReply => on_chunk("Writing text. "),
            Action::Code => on_chunk("Generating code. "),
            Action::Done => on_chunk("Done. "),
            Action::FetchDocs => on_chunk("Fetching Rust docs. "),
        }
        let chunk_tokens = match action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
            Action::FetchDocs => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
        };
        if fetched_docs_action {
            generation_prompt = augment_prompt_with_rust_docs(prompt);
        }
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
        decoder.generate_stream(
            &generation_prompt,
            action.as_str(),
            &cond_vec,
            decoder_tokens,
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
    let chunk_size = (token_slots / chunk_slots.max(1)).max(1);
    let total_slots = token_slots + chunk_slots + global_slots + 2;
    let mut mask_buf: Vec<f32> = Vec::with_capacity(batch * total_slots);
    for b in 0..batch {
        let token_len = token_lengths
            .get(b)
            .copied()
            .unwrap_or(token_slots)
            .clamp(1, token_slots);
        let chunk_len = token_len.div_ceil(chunk_size).clamp(1, chunk_slots);
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
            .clamp(1, token_slots);
        let valid_chunks = token_len.div_ceil(chunk_size).clamp(1, chunk_slots);
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
    if start < end {
        out.extend(tokens[start..end].iter().take(max_seq).copied());
    }
    if out.len() == row_start {
        out.push(pad_id);
    }
    let token_len = (out.len() - row_start).min(max_seq).max(1);
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
        max_seq.saturating_mul(recent_full_segments.max(1)),
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
    } else {
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
    if context_segments > 1 || recursive_context_compressor {
        let mut token_sequences = Vec::with_capacity(batch.len() * 2);
        token_sequences.extend(batch.iter().map(|row| row.state_tokens.as_slice()));
        token_sequences.extend(batch.iter().map(|row| row.next_tokens.as_slice()));
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
    } else {
        let (state_ids, next_ids, mut token_lengths, next_lens, _) =
            make_world_batch_from_slice(batch, max_seq, pad_id, device)?;
        token_lengths.extend(next_lens);
        let input_ids = Tensor::cat(&[&state_ids, &next_ids], 0)?;
        let features = maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
        let slots = planner_forward_encoder_masked(context_compressor, &features, &token_lengths)?;
        let batch_size = batch.len();
        let state_slots = slots.narrow(0, 0, batch_size)?;
        let next_slots = slots.narrow(0, batch_size, batch_size)?;
        Ok((state_slots, next_slots))
    }
}

#[allow(clippy::too_many_arguments)]
fn context_slots_from_world_post_state_batch(
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
) -> Result<Tensor> {
    if batch.is_empty() {
        bail!("world post-state batch is empty");
    }
    let mut owned_sequences = Vec::with_capacity(batch.len());
    for row in batch {
        let mut tokens = Vec::with_capacity(row.state_tokens.len() + row.next_tokens.len());
        tokens.extend(row.state_tokens.iter().copied());
        tokens.extend(row.next_tokens.iter().copied());
        owned_sequences.push(tokens);
    }
    let refs = owned_sequences
        .iter()
        .map(|tokens| tokens.as_slice())
        .collect::<Vec<_>>();
    context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        &refs,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        detach_encoder,
        device,
    )
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
    let mut token_sequences = Vec::with_capacity(batch.len() * 2);
    token_sequences.extend(batch.iter().map(|row| row.state_tokens.as_slice()));
    token_sequences.extend(batch.iter().map(|row| row.next_tokens.as_slice()));
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

fn world_post_state_loss_weight() -> f64 {
    std::env::var("TOFY_WORLD_POST_STATE_LOSS_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.35)
        .max(0.0)
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

fn continuation_edge_score(from: &WorldExample, to: &WorldExample) -> usize {
    if from.state_tokens.is_empty() || from.next_tokens.is_empty() || to.state_tokens.is_empty() {
        return 0;
    }
    let mut combined = Vec::with_capacity(from.state_tokens.len() + from.next_tokens.len());
    combined.extend(from.state_tokens.iter().copied());
    combined.extend(from.next_tokens.iter().copied());
    if to.state_tokens.starts_with(&combined) {
        return combined.len();
    }
    if combined.ends_with(&to.state_tokens) {
        return to.state_tokens.len();
    }
    let next_overlap = continuation_overlap(&from.next_tokens, &to.state_tokens);
    let combined_overlap = continuation_overlap(&combined, &to.state_tokens);
    next_overlap.max(combined_overlap)
}

fn continuation_edges(batch: &[WorldExample]) -> Vec<Option<usize>> {
    let min_overlap = env_usize("TOFY_WORLD_ROLLOUT_MIN_OVERLAP", 24);
    let mut edges = vec![None; batch.len()];
    for (from_idx, from) in batch.iter().enumerate() {
        let mut best = None;
        let mut best_score = min_overlap.saturating_sub(1);
        for (to_idx, to) in batch.iter().enumerate() {
            if from_idx == to_idx {
                continue;
            }
            let score = continuation_edge_score(from, to);
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
) -> Result<Option<Tensor>> {
    if batch.len() < 2 || world_rollout_loss_weight() == 0.0 {
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

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}
