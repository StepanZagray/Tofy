mod config;
mod data;
mod model;
mod tasks;
mod util;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::fs;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

use config::Config;
use data::{
    build_vocab_from_pair_file, count_pairs_with_vocab, ensure_hub_dataset_cached,
    ensure_hub_wikipedia_cached, make_augmented_jepa_batch, make_jepa_batch_from_pairs,
    prepare_ultrachat_pairs, CurriculumDenoisingConfig, PairStream, DEFAULT_MIN_TOKENS_PER_LINE,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use model::vocab::Pair;
use model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, symmetric_contrastive_loss, tensor_rms, OnlineEncoder,
};
use tensorboard_rs::summary_writer::SummaryWriter;

fn latent_sigreg_weight(step: usize, total_steps: usize, lambda: f64) -> f64 {
    let warmup_steps = (total_steps / 10).max(1);
    let scale = (step as f64 / warmup_steps as f64).clamp(0.0, 1.0);
    lambda * scale
}

fn latent_contrastive_weight(step: usize, total_steps: usize) -> f64 {
    let warmup_steps = (total_steps / 8).max(1);
    let scale = (step as f64 / warmup_steps as f64).clamp(0.0, 1.0);
    0.10 * scale
}

fn latent_target_ema_decay(step: usize, total_steps: usize) -> f64 {
    let progress = step as f64 / total_steps.max(1) as f64;
    (0.992 + 0.007 * progress).clamp(0.992, 0.999)
}

fn latent_curriculum(
    step: usize,
    total_steps: usize,
    config: &Config,
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
    contrastive_val: f32,
    sigreg_val: f32,
    pred_cos: f32,
    chunk_cos: f32,
    global_cos: f32,
    contrastive_cos: f32,
    context_rms: f32,
    target_rms: f32,
    target_count: usize,
    target_frac: f32,
    code_fraction: f32,
    active_seq: usize,
    max_spans_per_sample: usize,
    min_masked_ratio: f32,
    max_masked_ratio: f32,
    contrastive_weight: f32,
    reg_weight: f32,
    target_ema_decay: f32,
}

fn copy_varmap_weights(dst: &mut VarMap, src: &VarMap) -> Result<()> {
    let src_data = src
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock source varmap for copy"))?;
    let dst_data = dst
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock destination varmap for copy"))?;
    for (name, dst_var) in dst_data.iter() {
        let Some(src_var) = src_data.get(name) else {
            continue;
        };
        let src_tensor = if src_var.as_tensor().dtype() == dst_var.as_tensor().dtype() {
            src_var.as_tensor().clone()
        } else {
            src_var.as_tensor().to_dtype(dst_var.as_tensor().dtype())?
        };
        dst_var.set(&src_tensor)?;
    }
    Ok(())
}

fn ema_update_varmap(dst: &mut VarMap, src: &VarMap, decay: f64) -> Result<()> {
    let src_data = src
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock source varmap for EMA update"))?;
    let dst_data = dst
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock destination varmap for EMA update"))?;
    for (name, dst_var) in dst_data.iter() {
        let Some(src_var) = src_data.get(name) else {
            continue;
        };
        let src_tensor = if src_var.as_tensor().dtype() == dst_var.as_tensor().dtype() {
            src_var.as_tensor().clone()
        } else {
            src_var.as_tensor().to_dtype(dst_var.as_tensor().dtype())?
        };
        let updated = dst_var
            .as_tensor()
            .affine(decay, 0.0)?
            .broadcast_add(&src_tensor.affine(1.0 - decay, 0.0)?)?;
        dst_var.set(&updated)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 && (args[1] == "--prepare-ultrachat" || args[1] == "prepare-ultrachat") {
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
        return Ok(());
    }

    if tasks::world::try_run_train(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_orchestrator(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_decoder(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_eval(&args)? {
        return Ok(());
    }
    if tasks::eval::try_run_code_eval(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_serve(&args)? {
        return Ok(());
    }

    // JEPA-native evaluation (latent alignment + retrieval), no token-id ranking.
    if args.len() >= 5 && (args[1] == "--eval-jepa" || args[1] == "eval-jepa") {
        return run_eval_jepa(
            &PathBuf::from(&args[2]),
            &PathBuf::from(&args[3]),
            &args[4],
            args.get(5).and_then(|v| v.parse().ok()).unwrap_or(200),
            args.get(6).and_then(|v| v.parse().ok()).unwrap_or(32),
            args.get(7).and_then(|v| v.parse().ok()).unwrap_or(768),
            args.get(8).and_then(|v| v.parse().ok()).unwrap_or(256),
            args.get(9).and_then(|v| v.parse().ok()).unwrap_or(9),
            args.get(10).and_then(|v| v.parse().ok()).unwrap_or(8),
        );
    }

    // LeJEPA-style training over masked and target views with shared encoder weights.
    if args.len() >= 2 && (args[1] == "--latent" || args[1] == "latent") {
        let data_arg = if args.len() > 2 { &args[2] } else { "" };
        let (args_for_config, is_wikipedia) = if data_arg.starts_with("hub:") {
            let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
            let is_wik = dataset_id.to_lowercase().contains("wikipedia");
            let cache_path = if is_wik {
                ensure_hub_wikipedia_cached(dataset_id, Path::new("data"))?
            } else {
                ensure_hub_dataset_cached(dataset_id, Path::new("data"))?
            };
            let mut a = args[2..].to_vec();
            a[0] = cache_path.to_string_lossy().to_string();
            (a, is_wik)
        } else {
            (args[2..].to_vec(), false)
        };
        let mut config = Config::from_args_after(&args_for_config)?;
        config.is_paragraph_data = is_wikipedia;
        return run_latent_training(config);
    }

    // Latent training with encoder initialized from checkpoint (e.g. previous latent run)
    if args.len() >= 4
        && (args[1] == "--latent-from-checkpoint" || args[1] == "latent-from-checkpoint")
    {
        let init_path = PathBuf::from(&args[2]);
        let data_arg = if args.len() > 3 { &args[3] } else { "" };
        let (args_after_data, is_wikipedia) = if data_arg.starts_with("hub:") {
            let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
            let is_wik = dataset_id.to_lowercase().contains("wikipedia");
            let cache_path = if is_wik {
                ensure_hub_wikipedia_cached(dataset_id, Path::new("data"))?
            } else {
                ensure_hub_dataset_cached(dataset_id, Path::new("data"))?
            };
            let mut a = args[3..].to_vec();
            a[0] = cache_path.to_string_lossy().to_string();
            (a, is_wik)
        } else {
            (args[3..].to_vec(), false)
        };
        let mut config = Config::from_args_after(&args_after_data)?;
        config.init_encoder_path = Some(init_path);
        config.is_paragraph_data = is_wikipedia;
        return run_latent_training(config);
    }

    // No mode: print usage (Training vs Inference explicit)
    eprintln!("usage (choose one):");
    eprintln!("  Training (learn from data):");
    eprintln!(
        "    {} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio] [lambda] [--grad-accum <int>]",
        args[0]
    );
    eprintln!(
        "    {} --latent-from-checkpoint <encoder_checkpoint.safetensors> <data_path> [steps] ...",
        args[0]
    );
    eprintln!("  Evaluation (JEPA-native):");
    eprintln!(
        "    {} --eval-jepa <model_path> <vocab_path> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads]",
        args[0]
    );
    eprintln!("  World model agent:");
    eprintln!(
        "    {} --prepare-ultrachat [output_path] [context_window] [min_tokens] [max_rows]",
        args[0]
    );
    eprintln!(
        "    {} --train-world <encoder_model.safetensors> <encoder_vocab.txt> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lambda <float>] [--lr <float>]",
        args[0]
    );
    eprintln!(
        "    {} --train-orchestrator <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lr <float>] [--grad-accum <int>] [--freeze-planner] [--output <path>]",
        args[0]
    );
    eprintln!(
        "    {} --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--init-decoder <path>] [--decoder-output <path>]",
        args[0]
    );
    eprintln!(
        "    {} --eval-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots]",
        args[0]
    );
    eprintln!(
        "    {} --eval-code-assistant <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <suite.jsonl> [max_new_tokens] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--code-decoder <path>] [--code-decoder-vocab <path>] [--ablate-conditioning]",
        args[0]
    );
    eprintln!(
        "    {} --serve <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> [bind] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--debug]",
        args[0]
    );
    bail!(
        "specify a mode: --prepare-ultrachat / --latent / --latent-from-checkpoint / --eval-jepa / --train-world / --train-orchestrator / --train-decoder / --eval-world / --eval-code-assistant / --serve"
    );
}

// --- JEPA-style latent training ---
fn run_latent_training(config: Config) -> Result<()> {
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
    println!(
        "Preparing latent training input from {:?}: scanning dataset and building encoder vocab...",
        config.data_path
    );
    let (vocab, vocab_stats, pair_count) =
        build_vocab_from_pair_file(&config.data_path, config.max_vocab, min_tokens)?;
    println!("Vocab scan complete. Initializing streaming reader...");
    let mut pair_stream = PairStream::new(
        &config.data_path,
        min_tokens.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE),
    )?;
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
        "Vocab size: {} (includes <mask>) | pairs {} | seq_len {} | lambda {:.3}",
        vocab_size, pair_count, seq_len, config.lambda
    );
    if vocab_stats.total_tokens > 0 {
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
    }
    println!(
        "Estimated parameters: ~{}",
        util::format_params(latent_params)
    );

    let mut varmap = VarMap::new();
    if let Some(ref init_path) = config.init_encoder_path {
        varmap.load(init_path)?;
    }
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

    let mut target_varmap = VarMap::new();
    let target_vb = VarBuilder::from_varmap(&target_varmap, train_dtype, &device);
    let target_encoder = OnlineEncoder::new(
        target_vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    util::cast_varmap_dtype(&mut target_varmap, train_dtype)?;
    copy_varmap_weights(&mut target_varmap, &varmap)?;

    let train_vars = varmap.all_vars();
    let mut opt = candle_nn::AdamW::new_lr(train_vars.clone(), config.lr)?;

    let _ =
        fs::create_dir_all("local_models").and_then(|_| fs::create_dir_all("local_models/vocabs"));
    let model_path = PathBuf::from(format!(
        "local_models/model_latent_{}.safetensors",
        util::format_params(latent_params)
    ));
    let mut best_pred = f32::MAX;
    let mut saved_checkpoint = false;

    let run_dir = util::create_run_dir("latent")?;
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    println!("LeJEPA: masked-view prediction + SIGReg warmup, EMA target encoder");
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
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
        "config/effective_batch_size",
        (config.batch_size * config.grad_accum_steps.max(1)) as f32,
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

    const SIGREG_SLICES: usize = 128;
    const SIGREG_POINTS: usize = 17;
    for step in 1..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let grad_accum_steps = config.grad_accum_steps.max(1);
        let ema_decay = latent_target_ema_decay(step, config.steps);

        for micro_step in 0..grad_accum_steps {
            let batch_tokens = pair_stream.next_batch(config.batch_size)?;
            let curriculum = latent_curriculum(step, config.steps, &config);
            let batch = make_augmented_jepa_batch(&batch_tokens, &vocab, &curriculum, &device)?;

            let view_a_features = encoder.forward_features(&batch.view_a_ids)?;
            let target_features = target_encoder
                .forward_features(&batch.target_ids)?
                .detached();
            let paired_view_targets = target_encoder
                .forward_features(&batch.view_b_ids)?
                .detached();
            let predicted_features = encoder.predict_features(&view_a_features)?;
            let view_a_summary = view_a_features.contrastive_summary()?.squeeze(1)?;
            let view_b_summary = paired_view_targets.contrastive_summary()?.squeeze(1)?;
            let context_hidden = predicted_features.token_states;
            let target_hidden = target_features.token_states.clone();
            let (b, t, d) = context_hidden.dims3()?;
            let pred_token_flat = context_hidden.reshape((b * t, d))?;
            let target_flat = target_hidden.reshape((b * t, d))?;
            let context_targets = pred_token_flat.index_select(&batch.target_linear_indices, 0)?;
            let target_targets = target_flat.index_select(&batch.target_linear_indices, 0)?;
            let token_pred_loss = prediction_loss(&context_targets, &target_targets)?;
            let chunk_pred_loss = prediction_loss(
                &predicted_features.chunk_states,
                &paired_view_targets.chunk_states,
            )?;
            let target_global_mean = paired_view_targets.global_states.mean(1)?;
            let pred_global_mean = predicted_features.global_states.mean(1)?;
            let global_pred_loss = prediction_loss(&pred_global_mean, &target_global_mean)?;
            let contrastive_loss =
                symmetric_contrastive_loss(&view_a_summary, &view_b_summary, 0.1)?;
            let sigreg_loss = sigreg_epps_pulley(&pred_token_flat, SIGREG_SLICES, SIGREG_POINTS)?;
            let reg_weight = latent_sigreg_weight(step, config.steps, config.lambda);
            let contrastive_weight = latent_contrastive_weight(step, config.steps);
            let pred_loss = token_pred_loss
                .affine(0.82, 0.0)?
                .broadcast_add(&chunk_pred_loss.affine(0.12, 0.0)?)?
                .broadcast_add(&global_pred_loss.affine(0.06, 0.0)?)?;
            let loss = pred_loss
                .broadcast_add(&contrastive_loss.affine(contrastive_weight, 0.0)?)?
                .broadcast_add(&sigreg_loss.affine(reg_weight, 0.0)?)?;

            let should_capture_log =
                step % config.log_every == 0 && micro_step + 1 == grad_accum_steps;
            if should_capture_log {
                let pred_chunk_flat = flatten_latent_slots(&predicted_features.chunk_states)?;
                let target_chunk_flat = flatten_latent_slots(&paired_view_targets.chunk_states)?;
                let pred_cos = mean_cosine_similarity(&context_targets, &target_targets)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let chunk_cos = mean_cosine_similarity(&pred_chunk_flat, &target_chunk_flat)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let global_cos = mean_cosine_similarity(&pred_global_mean, &target_global_mean)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let context_rms = tensor_rms(&pred_token_flat)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let target_rms = tensor_rms(&target_flat)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let contrastive_cos = mean_cosine_similarity(&view_a_summary, &view_b_summary)?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                let target_count = batch.target_count;
                let target_frac =
                    target_count as f32 / (config.batch_size * config.max_seq).max(1) as f32;
                log_snapshot = Some(LatentLogSnapshot {
                    loss_val: loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    pred_val: pred_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    token_pred_val: token_pred_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    chunk_pred_val: chunk_pred_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    global_pred_val: global_pred_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    contrastive_val: contrastive_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    sigreg_val: sigreg_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                    pred_cos,
                    chunk_cos,
                    global_cos,
                    contrastive_cos,
                    context_rms,
                    target_rms,
                    target_count,
                    target_frac,
                    code_fraction: batch.code_fraction,
                    active_seq: curriculum.active_seq,
                    max_spans_per_sample: curriculum.max_spans_per_sample,
                    min_masked_ratio: curriculum.min_masked_ratio as f32,
                    max_masked_ratio: curriculum.max_masked_ratio as f32,
                    contrastive_weight: contrastive_weight as f32,
                    reg_weight: reg_weight as f32,
                    target_ema_decay: ema_decay as f32,
                });
            }

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                config.grad_accum_steps,
            )?;
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;
        ema_update_varmap(&mut target_varmap, &varmap, ema_decay)?;

        if step % config.log_every == 0 {
            let snapshot =
                log_snapshot.context("latent grad accumulation produced no log snapshot")?;

            tb.add_scalar("loss/total", snapshot.loss_val, step);
            tb.add_scalar("loss/pred", snapshot.pred_val, step);
            tb.add_scalar("loss/pred_token", snapshot.token_pred_val, step);
            tb.add_scalar("loss/pred_chunk", snapshot.chunk_pred_val, step);
            tb.add_scalar("loss/pred_global", snapshot.global_pred_val, step);
            tb.add_scalar("loss/contrastive", snapshot.contrastive_val, step);
            tb.add_scalar("loss/sigreg", snapshot.sigreg_val, step);
            tb.add_scalar("metrics/pred_cosine", snapshot.pred_cos, step);
            tb.add_scalar("metrics/chunk_cosine", snapshot.chunk_cos, step);
            tb.add_scalar("metrics/global_cosine", snapshot.global_cos, step);
            tb.add_scalar("metrics/contrastive_cosine", snapshot.contrastive_cos, step);
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
            tb.add_scalar("schedule/target_ema_decay", snapshot.target_ema_decay, step);
            tb.add_scalar(
                "schedule/contrastive_weight",
                snapshot.contrastive_weight,
                step,
            );
            tb.add_scalar("schedule/reg_weight", snapshot.reg_weight, step);
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
                varmap.save(&model_path)?;
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {:.4} pred {:.4} tok {:.4} chk {:.4} glb {:.4} ctr {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} ctr_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4} ema {:.4}{} [saved best_pred]",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
                    snapshot.token_pred_val,
                    snapshot.chunk_pred_val,
                    snapshot.global_pred_val,
                    snapshot.contrastive_val,
                    snapshot.sigreg_val,
                    snapshot.pred_cos,
                    snapshot.chunk_cos,
                    snapshot.global_cos,
                    snapshot.contrastive_cos,
                    snapshot.target_count,
                    snapshot.code_fraction,
                    snapshot.active_seq,
                    snapshot.reg_weight,
                    snapshot.target_ema_decay,
                    memory_note,
                );
            } else {
                println!(
                    "step {step}/{} total {:.4} pred {:.4} tok {:.4} chk {:.4} glb {:.4} ctr {:.4} sigreg {:.4} pred_cos {:.4} chk_cos {:.4} glb_cos {:.4} ctr_cos {:.4} targets {} code_frac {:.2} seq {} reg_w {:.4} ema {:.4}{}",
                    config.steps,
                    snapshot.loss_val,
                    snapshot.pred_val,
                    snapshot.token_pred_val,
                    snapshot.chunk_pred_val,
                    snapshot.global_pred_val,
                    snapshot.contrastive_val,
                    snapshot.sigreg_val,
                    snapshot.pred_cos,
                    snapshot.chunk_cos,
                    snapshot.global_cos,
                    snapshot.contrastive_cos,
                    snapshot.target_count,
                    snapshot.code_fraction,
                    snapshot.active_seq,
                    snapshot.reg_weight,
                    snapshot.target_ema_decay,
                    memory_note,
                );
            }
        }
    }

    if !saved_checkpoint {
        varmap.save(&model_path)?;
        println!(
            "No checkpoint was saved during logging; saved final encoder weights to {:?}",
            model_path
        );
    }
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

    let vocab_path = PathBuf::from("local_models/vocabs/vocab_encoder.txt");
    save_vocab_to_file(&vocab, &vocab_path)?;
    println!("Encoder vocab saved to {:?}", vocab_path);
    println!("\nTo run JEPA-native evaluation:");
    println!(
        "  cargo run --release -- --eval-jepa {} local_models/vocabs/vocab_encoder.txt <data_path|hub:dataset_id> 200 32 {} {} {} {}",
        model_path.display(),
        config.dim, seq_len, config.num_layers, config.num_heads
    );
    Ok(())
}

fn resolve_data_path(data_arg: &str) -> Result<PathBuf> {
    if data_arg.starts_with("hub:") {
        let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
        if dataset_id.to_lowercase().contains("wikipedia") {
            ensure_hub_wikipedia_cached(dataset_id, Path::new("data"))
        } else {
            ensure_hub_dataset_cached(dataset_id, Path::new("data"))
        }
    } else {
        Ok(PathBuf::from(data_arg))
    }
}

// JEPA-native evaluation:
// - latent alignment (cosine/L2) on held-out target regions
// - in-batch latent retrieval (Top-1 / Top-5 / MRR)
#[allow(clippy::too_many_arguments)]
fn run_eval_jepa(
    model_path: &PathBuf,
    vocab_path: &PathBuf,
    data_arg: &str,
    eval_steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let runtime_dtype = util::resolve_runtime_dtype(&device);

    let vocab = load_vocab_from_file(vocab_path)?;
    let data_path = resolve_data_path(data_arg)?;
    let pair_count = count_pairs_with_vocab(&data_path)?;
    let mut pair_stream = PairStream::new(&data_path, DEFAULT_MIN_TOKENS_PER_LINE)?;

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    util::cast_varmap_dtype(&mut varmap, runtime_dtype)?;
    let vb = VarBuilder::from_varmap(&varmap, runtime_dtype, &device);

    let vocab_size = vocab.id_to_token.len();
    let encoder = OnlineEncoder::new(vb.pp("encoder"), vocab_size, dim, num_layers, num_heads)?;

    let embed_params = vocab_size * dim;
    let block_params = num_layers * (4 * dim * dim + 8 * dim * dim);
    let total_params = embed_params + block_params;

    println!("LeJEPA evaluation");
    println!("model: {:?}", model_path);
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
        eval_steps, batch_size, dim, max_seq, num_layers, num_heads
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
    for _ in 0..eval_steps {
        let batch_tokens = pair_stream.next_batch(batch_size)?;
        let batch_pairs = batch_tokens
            .iter()
            .map(|tokens| Pair {
                tokens: vocab.encode(tokens),
            })
            .collect::<Vec<_>>();
        let (context_ids, target_ids, target_linear_indices) = make_jepa_batch_from_pairs(
            &batch_pairs,
            max_seq,
            vocab.pad_id,
            vocab.mask_id,
            EVAL_MAX_SPANS,
            EVAL_MAX_SPAN_LEN,
            EVAL_MAX_MASKED_RATIO,
            &device,
        )?;

        let context_features = encoder.forward_features(&context_ids)?;
        let predicted_features = encoder.predict_features(&context_features)?;
        let target_features = encoder.forward_features(&target_ids)?.detached();
        let online_hidden = predicted_features.token_states; // [B, T, D]
        let target_hidden = target_features.token_states;
        let (b, t, d) = online_hidden.dims3()?;
        let online_flat = online_hidden.reshape((b * t, d))?;
        let target_flat = target_hidden.reshape((b * t, d))?;

        let online_at_targets = online_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let target_latents = target_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let pred_loss = prediction_loss(&online_at_targets, &target_latents)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let sigreg_loss = sigreg_epps_pulley(&online_flat, 128, 17)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let pred_chunk_flat = flatten_latent_slots(&predicted_features.chunk_states)?;
        let target_chunk_flat = flatten_latent_slots(&target_features.chunk_states)?;
        let target_global_mean = target_features.global_states.mean(1)?;
        let pred_global_mean = predicted_features.global_states.mean(1)?;
        let chunk_cos = mean_cosine_similarity(&pred_chunk_flat, &target_chunk_flat)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let global_cos = mean_cosine_similarity(&pred_global_mean, &target_global_mean)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
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

        // In-batch latent retrieval:
        // For each context latent i, rank all target latents j by cosine(pred_i, target_j).
        // Correct match is diagonal j=i.
        let scores = pred_unit
            .clone()
            .matmul(&tgt_unit.clone().transpose(0, 1)?)?; // [N, N]
        let scores_vec = scores.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (i, row) in scores_vec.iter().enumerate() {
            if i >= row.len() {
                continue;
            }
            let target_score = row[i];
            let mut gt_count = 0usize;
            for &s in row {
                if s > target_score {
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
        sum_pred / eval_steps.max(1) as f64
    );
    println!(
        "  sigreg:         {:.4}",
        sum_sigreg / eval_steps.max(1) as f64
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
