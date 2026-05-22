use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::fs;
use std::path::PathBuf;
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::cli::resolve_data_path;
use crate::config::{LatentEvalConfig, LatentTrainConfig};
use crate::data::{
    build_vocab_from_pair_file, count_pairs_with_vocab, make_augmented_jepa_batch,
    make_augmented_jepa_batch_from_pairs, make_jepa_batch_from_pairs, prepare_ultrachat_pairs,
    CachedPairStream, CurriculumDenoisingConfig, PairStream, DEFAULT_MIN_TOKENS_PER_LINE,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::encoders::EncoderFeatures;
use crate::model::vocab::{vocab_signature, Pair};
use crate::model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, tensor_rms, OnlineEncoder,
};
use crate::util;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
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

fn token_cache_manifest(kind: &str) -> Option<serde_json::Value> {
    let cache_dir = std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string());
    let path = PathBuf::from(cache_dir).join(format!("{kind}_tokens.manifest.json"));
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn latent_token_cache_path(vocab_signature_value: &str, source_budget: usize) -> Option<PathBuf> {
    let manifest = token_cache_manifest("encoder")?;
    let max_seq = manifest.get("max_seq")?.as_u64()? as usize;
    let signature = manifest.get("vocab_signature")?.as_str()?;
    if max_seq < source_budget || signature != vocab_signature_value {
        return None;
    }
    token_cache_path("encoder")
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
        (vocab, None, 0)
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
    let mut pair_stream = PairStream::new(
        &config.data_path,
        min_tokens.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE),
    )?;
    let latent_source_budget = config.max_seq.max(1) * config.latent_context_segments.max(1);
    let mut cached_pair_stream = if let Some(cache_path) =
        latent_token_cache_path(&vocab_signature(&vocab), latent_source_budget)
    {
        println!("Token cache: using latent encoder cache {:?}", cache_path);
        Some(CachedPairStream::new(&cache_path)?)
    } else {
        println!("Token cache: no compatible latent cache found; using raw tokenization stream");
        None
    };
    println!("Streaming reader ready. Building training graph...");
    let vocab_size = vocab.id_to_token.len();
    let seq_len = config.max_seq;

    let embed_params = vocab_size * config.dim;
    let block_params =
        config.num_layers * (4 * config.dim * config.dim + 8 * config.dim * config.dim);
    let latent_params = embed_params + block_params;
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
    println!(
        "Estimated parameters: ~{}",
        util::format_params(latent_params)
    );

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
    if config.resume && train_checkpoint_path.exists() {
        varmap.load(&train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut varmap, train_dtype)?;
        println!("Resuming latent weights from {:?}", train_checkpoint_path);
    } else if config.resume && model_path.exists() {
        varmap.load(&model_path)?;
        util::cast_varmap_dtype(&mut varmap, train_dtype)?;
        println!(
            "Resuming latent weights from best export {:?} without optimizer state",
            model_path
        );
    } else if let Some(ref init_path) = config.init_encoder_path {
        varmap.load(init_path)?;
        util::cast_varmap_dtype(&mut varmap, train_dtype)?;
        println!("Initialized latent weights from {:?}", init_path);
    }

    let named_train_vars = util::named_train_vars(&varmap)?;
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
                "Resuming latent optimizer from {:?} at step {}",
                optimizer_checkpoint_path, resume_state.step
            );
        }
    }
    let mut best_pred = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };

    let run_dir = util::create_run_dir("latent")?;
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    println!("LeJEPA: online masked-view prediction + SIGReg");
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
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
    if start_step >= config.steps {
        println!(
            "Latent resume checkpoint already reached step {}/{}; skipping training.",
            start_step, config.steps
        );
    }
    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = latent_batch_size_for_step(step, &config);
        let grad_accum_steps = latent_grad_accum_for_step(step, &config);

        for micro_step in 0..grad_accum_steps {
            let curriculum = latent_curriculum(step, config.steps, &config);
            let batch = if let Some(ref mut cached_stream) = cached_pair_stream {
                let batch_pairs = cached_stream.next_batch(batch_size)?;
                make_augmented_jepa_batch_from_pairs(&batch_pairs, &vocab, &curriculum, &device)?
            } else {
                let batch_tokens = pair_stream.next_batch(batch_size)?;
                make_augmented_jepa_batch(&batch_tokens, &vocab, &curriculum, &device)?
            };

            let all_view_ids = Tensor::cat(
                &[&batch.view_a_ids, &batch.target_ids, &batch.view_b_ids],
                0,
            )?;
            let all_view_features = encoder.forward_features(&all_view_ids)?;
            let [view_a_features, target_features, paired_view_targets] =
                split_encoder_features(&all_view_features, batch_size)?;
            let context_hidden = view_a_features.token_states.clone();
            let target_hidden = target_features.token_states.clone();
            let (b, t, d) = context_hidden.dims3()?;
            let pred_token_flat = context_hidden.reshape((b * t, d))?;
            let target_flat = target_hidden.reshape((b * t, d))?;
            let context_targets = pred_token_flat.index_select(&batch.target_linear_indices, 0)?;
            let target_targets = target_flat.index_select(&batch.target_linear_indices, 0)?;
            let token_pred_loss = prediction_loss(&context_targets, &target_targets)?;
            let chunk_pred_loss = prediction_loss(
                &view_a_features.chunk_states,
                &paired_view_targets.chunk_states,
            )?;
            let target_global_mean = paired_view_targets.global_states.mean(1)?;
            let pred_global_mean = view_a_features.global_states.mean(1)?;
            let global_pred_loss = prediction_loss(&pred_global_mean, &target_global_mean)?;
            let sigreg_loss = sigreg_epps_pulley(&pred_token_flat, sigreg_slices, sigreg_points)?;
            let pred_loss = token_pred_loss
                .affine(0.82, 0.0)?
                .broadcast_add(&chunk_pred_loss.affine(0.12, 0.0)?)?
                .broadcast_add(&global_pred_loss.affine(0.06, 0.0)?)?;
            let loss = pred_loss.broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?;

            let should_capture_log =
                step % config.log_every == 0 && micro_step + 1 == grad_accum_steps;
            if should_capture_log {
                let pred_chunk_flat = flatten_latent_slots(&view_a_features.chunk_states)?;
                let target_chunk_flat = flatten_latent_slots(&paired_view_targets.chunk_states)?;
                let pred_cos =
                    util::scalar_f32(&mean_cosine_similarity(&context_targets, &target_targets)?)?;
                let chunk_cos = util::scalar_f32(&mean_cosine_similarity(
                    &pred_chunk_flat,
                    &target_chunk_flat,
                )?)?;
                let global_cos = util::scalar_f32(&mean_cosine_similarity(
                    &pred_global_mean,
                    &target_global_mean,
                )?)?;
                let context_rms = util::scalar_f32(&tensor_rms(&pred_token_flat)?)?;
                let target_rms = util::scalar_f32(&tensor_rms(&target_flat)?)?;
                let target_count = batch.target_count;
                let target_frac = target_count as f32 / (batch_size * config.max_seq).max(1) as f32;
                log_snapshot = Some(LatentLogSnapshot {
                    loss_val: util::scalar_f32(&loss)?,
                    pred_val: util::scalar_f32(&pred_loss)?,
                    token_pred_val: util::scalar_f32(&token_pred_loss)?,
                    chunk_pred_val: util::scalar_f32(&chunk_pred_loss)?,
                    global_pred_val: util::scalar_f32(&global_pred_loss)?,
                    sigreg_val: util::scalar_f32(&sigreg_loss)?,
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

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let snapshot =
                log_snapshot.context("latent grad accumulation produced no log snapshot")?;

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
            tb.flush();

            if snapshot.pred_val < best_pred {
                best_pred = snapshot.pred_val;
                util::save_varmap_atomic(&varmap, &model_path)?;
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {:.4} pred {:.4} tok {:.4} chk {:.4} glb {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4}{} [saved best_pred]",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
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
                    memory_note,
                );
            } else {
                println!(
                    "step {step}/{} total {:.4} pred {:.4} tok {:.4} chk {:.4} glb {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4}{}",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
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
                    memory_note,
                );
            }
            util::save_varmap_atomic(&varmap, &train_checkpoint_path)?;
            opt.save_state(&optimizer_checkpoint_path)?;
            resume_state = util::TrainingResumeState {
                stage: resume_stage.clone(),
                step,
                best_metric: best_pred,
                best_aux_metric: best_pred,
                saved_checkpoint,
            };
            util::save_resume_state(&resume_state_path, &resume_state)?;
        }
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&varmap, &model_path)?;
        println!(
            "No checkpoint was saved during logging; saved final encoder weights to {:?}",
            model_path
        );
    }
    util::save_varmap_atomic(&varmap, &train_checkpoint_path)?;
    opt.save_state(&optimizer_checkpoint_path)?;
    resume_state = util::TrainingResumeState {
        stage: resume_stage.clone(),
        step: config.steps,
        best_metric: best_pred,
        best_aux_metric: best_pred,
        saved_checkpoint,
    };
    util::save_resume_state(&resume_state_path, &resume_state)?;
    tb.flush();
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
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let runtime_dtype = util::resolve_runtime_dtype(&device);

    let vocab = load_vocab_from_file(&config.vocab_path)?;
    let data_path = resolve_data_path(&config.data_arg)?.path;
    let pair_count = count_pairs_with_vocab(&data_path)?;
    let mut pair_stream = PairStream::new(&data_path, DEFAULT_MIN_TOKENS_PER_LINE)?;

    let mut varmap = VarMap::new();
    varmap.load(&config.model_path)?;
    util::cast_varmap_dtype(&mut varmap, runtime_dtype)?;
    let vb = VarBuilder::from_varmap(&varmap, runtime_dtype, &device);

    let vocab_size = vocab.id_to_token.len();
    let encoder = OnlineEncoder::new(
        vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;

    let embed_params = vocab_size * config.dim;
    let block_params =
        config.num_layers * (4 * config.dim * config.dim + 8 * config.dim * config.dim);
    let total_params = embed_params + block_params;

    println!("LeJEPA evaluation");
    println!("model: {:?}", config.model_path);
    println!("data: {:?}", data_path);
    println!("pairs: {}", pair_count);
    println!(
        "model size: ~{} [embed {} + blocks {}]",
        util::format_params(total_params),
        util::format_params(embed_params),
        util::format_params(block_params),
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
    for _ in 0..config.eval_steps {
        let batch_tokens = pair_stream.next_batch(config.batch_size)?;
        let batch_pairs = batch_tokens
            .iter()
            .map(|tokens| Pair {
                tokens: vocab.encode(tokens),
            })
            .collect::<Vec<_>>();
        let (context_ids, target_ids, target_linear_indices) = make_jepa_batch_from_pairs(
            &batch_pairs,
            config.max_seq,
            vocab.pad_id,
            vocab.mask_id,
            EVAL_MAX_SPANS,
            EVAL_MAX_SPAN_LEN,
            EVAL_MAX_MASKED_RATIO,
            &device,
        )?;

        let context_features = encoder.forward_features(&context_ids)?;
        let target_features = encoder.forward_features(&target_ids)?.detached();
        let online_hidden = context_features.token_states;
        let target_hidden = target_features.token_states;
        let (b, t, d) = online_hidden.dims3()?;
        let online_flat = online_hidden.reshape((b * t, d))?;
        let target_flat = target_hidden.reshape((b * t, d))?;

        let online_at_targets = online_flat.index_select(&target_linear_indices, 0)?;
        let target_latents = target_flat.index_select(&target_linear_indices, 0)?;
        let pred_loss = util::scalar_f32(&prediction_loss(&online_at_targets, &target_latents)?)?;
        let sigreg_loss = util::scalar_f32(&sigreg_epps_pulley(&online_flat, 128, 17)?)?;
        let pred_chunk_flat = flatten_latent_slots(&context_features.chunk_states)?;
        let target_chunk_flat = flatten_latent_slots(&target_features.chunk_states)?;
        let target_global_mean = target_features.global_states.mean(1)?;
        let pred_global_mean = context_features.global_states.mean(1)?;
        let chunk_cos = util::scalar_f32(&mean_cosine_similarity(
            &pred_chunk_flat,
            &target_chunk_flat,
        )?)?;
        let global_cos = util::scalar_f32(&mean_cosine_similarity(
            &pred_global_mean,
            &target_global_mean,
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
