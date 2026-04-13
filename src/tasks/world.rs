use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::{
    build_vocab_from_raw_world_file_with_mode, count_raw_world_rows, count_raw_world_rows_split,
    count_raw_world_rows_split_with_mode, encode_text_with_vocab_mode, encode_world_examples,
    encode_world_examples_with_mode, ensure_hub_dataset_cached, make_decoder_batch,
    make_world_batch_from_slice, tokenize_for_inference, RawWorldExample, RawWorldStream,
    TokenizationMode, ACTION_CODE, ACTION_DONE, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::encoders::EncoderFeatures;
use crate::model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, tensor_rms, CandleCrossAttnDecoder, CodeDecoder,
    DecoderAdapter, DecoderKind, LlamaCppDecoder, LocalDecoderRuntime, OnlineEncoder,
    OrchestratorActionHead, PlannerMemory, StubLocalDecoder, Vocab, WorldTransition,
};
use crate::util;
use candle_nn::ops;

const HELDOUT_SPLIT_MODULUS: usize = 20;
const HELDOUT_SPLIT_REMAINDER: usize = 0;

#[derive(Clone)]
struct WorldConfig {
    encoder_model_path: PathBuf,
    encoder_vocab_path: PathBuf,
    data_path: PathBuf,
    steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    /// Number of planner-memory slots and default decoder-conditioning budget.
    num_latent_tokens: usize,
    lambda: f64,
    lr: f64,
    log_every: usize,
    grad_accum_steps: usize,
    action_loss_weight: f64,
    router_warmup_steps: usize,
    train_dtype: DType,
}

impl WorldConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                    "usage: --train-world <encoder_model.safetensors> <encoder_vocab.txt> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lambda <float>] [--lr <float>] [--grad-accum <int>]"
            );
        }
        let mut lr_override = None;
        let mut lambda_override = None;
        let mut grad_accum_steps = 1usize;
        let mut action_loss_weight = None;
        let mut router_warmup_steps = 0usize;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--lr" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                let lr: f64 = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--lr must be float, got {:?}", value))?;
                lr_override = Some(lr);
                i += 2;
                continue;
            }
            if args[i] == "--lambda" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lambda requires float"))?;
                let lambda: f64 = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--lambda must be float, got {:?}", value))?;
                lambda_override = Some(lambda.clamp(0.0, 1.0));
                i += 2;
                continue;
            }
            if args[i] == "--grad-accum" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--grad-accum requires integer"))?;
                grad_accum_steps = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--grad-accum must be integer"))?;
                i += 2;
                continue;
            }
            if args[i] == "--action-loss-weight" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--action-loss-weight requires float"))?;
                let parsed: f64 = value.parse().map_err(|_| {
                    anyhow::anyhow!("--action-loss-weight must be float, got {:?}", value)
                })?;
                action_loss_weight = Some(parsed.max(0.0));
                i += 2;
                continue;
            }
            if args[i] == "--router-warmup" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--router-warmup requires integer"))?;
                router_warmup_steps = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--router-warmup must be integer"))?;
                i += 2;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            data_path: PathBuf::from(&filtered[2]),
            steps: filtered
                .get(3)
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
            batch_size: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(24),
            dim: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(64),
            lambda: lambda_override.unwrap_or(0.2),
            lr: lr_override.unwrap_or(2e-4),
            log_every: 100,
            grad_accum_steps: grad_accum_steps.max(1),
            action_loss_weight: action_loss_weight.unwrap_or(1.0),
            router_warmup_steps,
            train_dtype: std::env::var("TOFY_TRAIN_DTYPE")
                .ok()
                .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                    "f16" | "float16" | "fp16" => Some(DType::F16),
                    "bf16" => Some(DType::BF16),
                    "f32" | "float32" | "fp32" => Some(DType::F32),
                    _ => None,
                })
                .unwrap_or(DType::F32),
        })
    }
}

#[derive(Clone)]
struct OrchestratorTrainConfig {
    encoder_model_path: PathBuf,
    encoder_vocab_path: PathBuf,
    world_model_path: PathBuf,
    data_path: PathBuf,
    steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    lr: f64,
    log_every: usize,
    grad_accum_steps: usize,
    tune_planner: bool,
    output_path: Option<PathBuf>,
    train_dtype: DType,
}

impl OrchestratorTrainConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-orchestrator <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lr <float>] [--grad-accum <int>] [--freeze-planner] [--output <path>]"
            );
        }
        let mut lr_override = None;
        let mut grad_accum_steps = 1usize;
        let mut tune_planner = true;
        let mut output_path = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--lr" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                    lr_override = Some(
                        value
                            .parse()
                            .map_err(|_| anyhow::anyhow!("--lr must be float"))?,
                    );
                    i += 2;
                }
                "--grad-accum" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--grad-accum requires integer"))?;
                    grad_accum_steps = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--grad-accum must be integer"))?;
                    i += 2;
                }
                "--freeze-planner" => {
                    tune_planner = false;
                    i += 1;
                }
                "--output" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--output requires path"))?;
                    output_path = Some(PathBuf::from(value));
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps: filtered
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(24),
            dim: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            lr: lr_override.unwrap_or(2e-4),
            log_every: 100,
            grad_accum_steps: grad_accum_steps.max(1),
            tune_planner,
            output_path,
            train_dtype: std::env::var("TOFY_TRAIN_DTYPE")
                .ok()
                .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                    "f16" | "float16" | "fp16" => Some(DType::F16),
                    "bf16" => Some(DType::BF16),
                    "f32" | "float32" | "fp32" => Some(DType::F32),
                    _ => None,
                })
                .unwrap_or(DType::F32),
        })
    }
}

pub fn try_run_train(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--train-world" && args[1] != "train-world") {
        return Ok(false);
    }
    let data_arg = &args[4];
    let data_path = resolve_world_data_path(data_arg)?;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[2] = data_path.to_string_lossy().to_string();
    let cfg = WorldConfig::from_args_after(&args_for_cfg)?;
    run_world_training(cfg)?;
    Ok(true)
}

pub fn try_run_train_orchestrator(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--train-orchestrator" && args[1] != "train-orchestrator") {
        return Ok(false);
    }
    let data_arg = &args[5];
    let data_path = resolve_world_data_path(data_arg)?;
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
    let data_arg = &args[5];
    let data_path = resolve_world_data_path(data_arg)?;
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
    if args.len() < 7 || (args[1] != "--eval-world" && args[1] != "eval-world") {
        return Ok(false);
    }
    run_eval_world(
        &PathBuf::from(&args[2]),
        &PathBuf::from(&args[3]),
        &PathBuf::from(&args[4]),
        &args[5],
        args.get(6).and_then(|v| v.parse().ok()).unwrap_or(200),
        args.get(7).and_then(|v| v.parse().ok()).unwrap_or(32),
        args.get(8).and_then(|v| v.parse().ok()).unwrap_or(768),
        args.get(9).and_then(|v| v.parse().ok()).unwrap_or(256),
        args.get(10).and_then(|v| v.parse().ok()).unwrap_or(9),
        args.get(11).and_then(|v| v.parse().ok()).unwrap_or(8),
        args.get(12).and_then(|v| v.parse().ok()).unwrap_or(256),
        args.get(13).and_then(|v| v.parse().ok()).unwrap_or(64),
    )?;
    Ok(true)
}

pub fn try_run_serve(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--serve" && args[1] != "serve") {
        return Ok(false);
    }
    let debug = args.iter().any(|a| a == "--debug");
    let positional: Vec<&str> = args
        .iter()
        .skip(2)
        .filter(|a| *a != "--debug")
        .map(String::as_str)
        .collect();
    if positional.len() < 3 {
        return Ok(false);
    }
    let encoder_model_path = PathBuf::from(positional[0]);
    let encoder_vocab_path = PathBuf::from(positional[1]);
    let world_model_path = PathBuf::from(positional[2]);
    let bind = positional.get(3).copied().unwrap_or("0.0.0.0:8080");
    let dim = positional
        .get(4)
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);
    let max_seq = positional
        .get(5)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let num_layers = positional.get(6).and_then(|v| v.parse().ok()).unwrap_or(9);
    let num_heads = positional.get(7).and_then(|v| v.parse().ok()).unwrap_or(8);
    let bridge_dim = positional
        .get(8)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let num_latent_tokens = positional.get(9).and_then(|v| v.parse().ok()).unwrap_or(64);
    if debug {
        std::env::set_var("JEPA_DEBUG", "1");
    }
    let rt = tokio::runtime::Runtime::new().context("create tokio runtime")?;
    rt.block_on(crate::tasks::serve::run(
        bind,
        encoder_model_path,
        encoder_vocab_path,
        world_model_path,
        dim,
        max_seq,
        num_layers,
        num_heads,
        bridge_dim,
        num_latent_tokens,
        debug,
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

    let world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), config.bridge_dim)?;
    let orchestrator_head =
        OrchestratorActionHead::new(world_vb.pp("orchestrator_action_head"), config.bridge_dim)?;
    let inverse_action_head =
        OrchestratorActionHead::new(world_vb.pp("inverse_action_head"), config.bridge_dim)?;

    let train_vars = world_varmap.all_vars();
    let mut opt = candle_nn::AdamW::new_lr(train_vars.clone(), config.lr)?;

    let transition_params = 3
        * (config.bridge_dim * config.bridge_dim * 4
            + config.bridge_dim * config.bridge_dim * 2
            + 4 * config.bridge_dim);
    let planner_params = config.num_latent_tokens * config.dim
        + 2 * (config.dim * config.dim + config.dim * 4 * config.dim)
        + config.dim * config.bridge_dim
        + config.bridge_dim;
    let orchestrator_hidden = (config.bridge_dim * 2).max(256);
    const ORCH_N: usize = crate::model::orchestrator_head::NUM_ACTIONS;
    let orchestrator_params = config.bridge_dim * orchestrator_hidden
        + orchestrator_hidden
        + orchestrator_hidden * ORCH_N
        + ORCH_N;
    let inverse_params = orchestrator_params;
    let total_params = transition_params + planner_params + orchestrator_params + inverse_params;
    let _ = fs::create_dir_all("local_models");
    let model_path = PathBuf::from(format!(
        "local_models/model_world_{}.safetensors",
        util::format_params(total_params)
    ));

    println!("Training (latent-only dialog transition model for text + code)");
    println!("Encoder checkpoint: {:?}", config.encoder_model_path);
    println!("Encoder vocab: {:?}", config.encoder_vocab_path);
    println!(
        "Rows: train {} | val {} | encoder vocab {} | max_seq {} | planner_slots {} | lambda {:.3}",
        train_row_count,
        val_row_count,
        vocab_size,
        config.max_seq,
        config.num_latent_tokens,
        config.lambda
    );
    println!(
        "Router training: action_loss_weight {:.2} | router_warmup {} steps",
        config.action_loss_weight, config.router_warmup_steps
    );
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "Estimated parameters: ~{} [planner_memory {} + transition {} + orchestrator {} + inverse {}]",
        util::format_params(total_params),
        util::format_params(planner_params),
        util::format_params(transition_params),
        util::format_params(orchestrator_params),
        util::format_params(inverse_params)
    );

    let mut best_loss = f32::MAX;
    let mut best_metric = f32::MAX;
    let mut saved_checkpoint = false;

    let run_dir = util::create_run_dir("world")?;
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar("config/dim", config.dim as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/planner_slots", config.num_latent_tokens as f32, 0);
    tb.add_scalar("config/estimated_params", total_params as f32, 0);
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
    let inverse_loss_weight = std::env::var("TOFY_WORLD_INVERSE_LOSS_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35f64)
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
    const SIGREG_SLICES: usize = 128;
    const SIGREG_POINTS: usize = 17;
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    for step in 1..=config.steps {
        let mut accumulated_grads = None;
        let mut last_transition_loss = None;
        let mut last_sigreg_loss = None;
        let mut last_action_loss = None;
        let mut last_inverse_loss = None;
        let mut last_loss = None;
        let mut last_action_logits = None;
        let mut last_inverse_logits = None;
        let mut last_action_labels = Vec::new();
        let mut last_pred_next_slots = None;
        let mut last_next_slots = None;
        let mut last_state_slots = None;
        let router_warmup_active = step <= config.router_warmup_steps;

        for _micro_step in 0..config.grad_accum_steps.max(1) {
            let raw_batch = collect_action_training_batch(
                &mut world_stream,
                config.batch_size,
                TARGET_CODE_RATE,
                TARGET_DONE_RATE,
            )?;
            let batch = encode_world_examples(&raw_batch, &encoder_vocab);
            let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
            let state_slots = if context_segments > 1 || recursive_planner_memory {
                let state_tokens = batch
                    .iter()
                    .map(|row| row.state_tokens.clone())
                    .collect::<Vec<_>>();
                planner_slots_from_token_sequences(
                    &encoder,
                    &planner_memory,
                    &state_tokens,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_planner_memory,
                    &device,
                )?
            } else {
                let (state_ids, _next_ids, state_lens, _next_lens, _) =
                    make_world_batch_from_slice(
                        &batch,
                        config.max_seq,
                        encoder_vocab.pad_id,
                        &device,
                    )?;
                let state_features = encoder.forward_features(&state_ids)?.detached();
                planner_forward_encoder_masked(&planner_memory, &state_features, &state_lens)?
            };
            let next_slots = if context_segments > 1 || recursive_planner_memory {
                let next_tokens = batch
                    .iter()
                    .map(|row| row.next_tokens.clone())
                    .collect::<Vec<_>>();
                planner_slots_from_token_sequences(
                    &encoder,
                    &planner_memory,
                    &next_tokens,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_planner_memory,
                    &device,
                )?
            } else {
                let (_state_ids, next_ids, _state_lens, next_lens, _) =
                    make_world_batch_from_slice(
                        &batch,
                        config.max_seq,
                        encoder_vocab.pad_id,
                        &device,
                    )?;
                let next_features = encoder.forward_features(&next_ids)?.detached();
                planner_forward_encoder_masked(&planner_memory, &next_features, &next_lens)?
            };
            let pred_next_slots = transition.forward(&state_slots, &action_labels)?;

            let transition_loss = prediction_loss(&pred_next_slots, &next_slots)?;
            let state_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&state_slots)?,
                SIGREG_SLICES,
                SIGREG_POINTS,
            )?;
            let next_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&next_slots)?,
                SIGREG_SLICES,
                SIGREG_POINTS,
            )?;
            let pred_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&pred_next_slots)?,
                SIGREG_SLICES,
                SIGREG_POINTS,
            )?;
            let sigreg_loss = state_sigreg
                .broadcast_add(&next_sigreg)?
                .broadcast_add(&pred_sigreg)?
                .affine(1.0 / 3.0, 0.0)?;

            let action_logits = orchestrator_head.forward(&state_slots)?;
            let action_loss = action_cross_entropy(&action_logits, &action_labels, &device)?;
            let true_delta_slots = slot_delta_slots(&next_slots, &state_slots)?;
            let pred_delta_slots = slot_delta_slots(&pred_next_slots, &state_slots)?;
            let inverse_logits_true = inverse_action_head.forward(&true_delta_slots)?;
            let inverse_logits_pred = inverse_action_head.forward(&pred_delta_slots)?;
            let inverse_true_loss =
                action_cross_entropy(&inverse_logits_true, &action_labels, &device)?;
            let inverse_pred_loss =
                action_cross_entropy(&inverse_logits_pred, &action_labels, &device)?;
            let inverse_loss = inverse_true_loss
                .broadcast_add(&inverse_pred_loss)?
                .affine(0.5, 0.0)?;
            let loss = if router_warmup_active {
                action_loss.broadcast_add(&inverse_loss.affine(inverse_loss_weight, 0.0)?)?
            } else {
                let pred_term = transition_loss.affine(1.0 - config.lambda, 0.0)?;
                let reg_term = sigreg_loss.affine(config.lambda, 0.0)?;
                let action_term = action_loss.affine(config.action_loss_weight, 0.0)?;
                let inverse_term = inverse_loss.affine(inverse_loss_weight, 0.0)?;
                pred_term
                    .broadcast_add(&reg_term)?
                    .broadcast_add(&action_term)?
                    .broadcast_add(&inverse_term)?
            };

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                config.grad_accum_steps,
            )?;

            last_transition_loss = Some(transition_loss);
            last_sigreg_loss = Some(sigreg_loss);
            last_action_loss = Some(action_loss);
            last_inverse_loss = Some(inverse_loss);
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
            tb.add_scalar("metrics/state_slot_rms", state_slot_rms, step);
            tb.add_scalar("metrics/pred_slot_rms", pred_slot_rms, step);
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
            if let Some(ref mut val_stream) = val_stream {
                let val_raw_batch = collect_action_training_batch(
                    val_stream,
                    config.batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?;
                let val_metrics = evaluate_world_batch(
                    &val_raw_batch,
                    &encoder_vocab,
                    &encoder,
                    &planner_memory,
                    &transition,
                    &orchestrator_head,
                    &inverse_action_head,
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
                world_varmap.save(&model_path)?;
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} inverse {inv_val:.4} action_acc {:.3} bal_acc {:.3} macro_f1 {:.3} inv_acc {:.3} inv_bal {:.3} inv_f1 {:.3} code_p {:.3} code_r {:.3} code_f1 {:.3} done_f1 {:.3} trans_cos {trans_cos:.4} code_rate {:.3} pred_code {:.3} done_rate {:.3} pred_done {:.3} sel {selection_metric:.4}{}{} [saved best]",
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
                    if router_warmup_active { " [router_warmup]" } else { "" },
                    memory_note
                );
            } else {
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} inverse {inv_val:.4} action_acc {:.3} bal_acc {:.3} macro_f1 {:.3} inv_acc {:.3} inv_bal {:.3} inv_f1 {:.3} code_p {:.3} code_r {:.3} code_f1 {:.3} done_f1 {:.3} trans_cos {trans_cos:.4} code_rate {:.3} pred_code {:.3} done_rate {:.3} pred_done {:.3} sel {selection_metric:.4}{}{}",
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
                    if router_warmup_active { " [router_warmup]" } else { "" },
                    memory_note
                );
            }
        }
    }

    if !saved_checkpoint {
        world_varmap.save(&model_path)?;
        println!(
            "No checkpoint was saved during logging; saved final world weights to {:?}",
            model_path
        );
    }
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
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let _transition = WorldTransition::new(world_vb.pp("world_transition"), config.bridge_dim)?;
    let orchestrator_head =
        OrchestratorActionHead::new(world_vb.pp("orchestrator_action_head"), config.bridge_dim)?;
    let _inverse_action_head =
        if checkpoint_has_prefix(&config.world_model_path, "inverse_action_head.") {
            Some(OrchestratorActionHead::new(
                world_vb.pp("inverse_action_head"),
                config.bridge_dim,
            )?)
        } else {
            None
        };
    world_varmap.load(&config.world_model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;

    let train_vars = world_varmap.all_vars();
    let mut opt = candle_nn::AdamW::new_lr(train_vars.clone(), config.lr)?;
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| config.world_model_path.clone());

    println!("Training (planner/orchestrator action model)");
    println!("Encoder checkpoint: {:?}", config.encoder_model_path);
    println!("World checkpoint: {:?}", config.world_model_path);
    println!(
        "Rows: train {} | val {} | max_seq {} | planner_slots {} | tune_planner {}",
        train_row_count,
        val_row_count,
        config.max_seq,
        config.num_latent_tokens,
        config.tune_planner
    );

    let run_dir = util::create_run_dir("orchestrator")?;
    let mut tb = SummaryWriter::new(&run_dir);
    let mut vram_tracker = util::VramTracker::default();
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/planner_slots", config.num_latent_tokens as f32, 0);
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
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    let mut best_score = f32::MAX;
    let mut saved_checkpoint = false;

    for step in 1..=config.steps {
        let mut accumulated_grads = None;
        let mut last_action_loss = None;
        let mut last_action_logits = None;
        let mut last_action_labels = Vec::new();

        for _micro_step in 0..config.grad_accum_steps.max(1) {
            let raw_batch = collect_action_training_batch(
                &mut train_stream,
                config.batch_size,
                TARGET_CODE_RATE,
                TARGET_DONE_RATE,
            )?;
            let batch = encode_world_examples(&raw_batch, &encoder_vocab);
            let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
            let state_tokens = batch
                .iter()
                .map(|row| row.state_tokens.clone())
                .collect::<Vec<_>>();
            let mut state_slots = planner_slots_from_token_sequences(
                &encoder,
                &planner_memory,
                &state_tokens,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                recursive_planner_memory,
                &device,
            )?;
            if !config.tune_planner {
                state_slots = state_slots.detach();
            }
            let action_logits = orchestrator_head.forward(&state_slots)?;
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
                .context("orchestrator grad accumulation produced no action loss")?;
            let action_logits = last_action_logits
                .context("orchestrator grad accumulation produced no action logits")?;
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

            if let Some(ref mut stream) = val_stream {
                let raw_batch = collect_action_training_batch(
                    stream,
                    config.batch_size,
                    TARGET_CODE_RATE,
                    TARGET_DONE_RATE,
                )?;
                let batch = encode_world_examples(&raw_batch, &encoder_vocab);
                let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
                let state_tokens = batch
                    .iter()
                    .map(|row| row.state_tokens.clone())
                    .collect::<Vec<_>>();
                let state_slots = planner_slots_from_token_sequences(
                    &encoder,
                    &planner_memory,
                    &state_tokens,
                    encoder_vocab.pad_id,
                    config.max_seq,
                    context_segments,
                    recent_full_segments,
                    recursive_planner_memory,
                    &device,
                )?
                .detach();
                let action_logits = orchestrator_head.forward(&state_slots)?;
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
            }
            tb.add_scalar("val/selection_score", selection_score, step);
            tb.flush();

            if selection_score < best_score {
                best_score = selection_score;
                world_varmap.save(&output_path)?;
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
        }
    }

    if !saved_checkpoint {
        world_varmap.save(&output_path)?;
    }
    tb.flush();
    let _ = vram_tracker.write_summary(&run_dir, "orchestrator");
    println!(
        "Best planner/orchestrator checkpoint saved to {:?} (selection {:.4})",
        output_path, best_score
    );
    Ok(())
}

/// Decoder training: load frozen planner stack (encoder + planner_memory + transition),
/// train decoder adapter + decoder jointly on top of planner memory.
/// Defaults sized for ~8GB VRAM (e.g. RTX 5060): ~90M decoder, batch 8.
const DECODER_DIM: usize = 768;
const DECODER_LAYERS: usize = 8;
const DECODER_HEADS: usize = 8;
const DECODER_FF_DIM: usize = 3072;

/// Approximate parameter count for decoder checkpoint: decoder adapter + decoder.
fn decoder_param_count(
    vocab_size: usize,
    planner_dim: usize,
    world_dim: usize,
    kind: DecoderKind,
    planner_slots: usize,
) -> usize {
    let dim = DECODER_DIM;
    let n_layers = DECODER_LAYERS;
    let ff = DECODER_FF_DIM;
    let embed = vocab_size * dim;
    let lm_head = dim * vocab_size;
    let ln_final = 2 * dim;
    let kind_embed = 2 * dim;
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
    let adapter_slots = DecoderAdapter::output_slots_for(kind, planner_slots);
    let decoder_adapter = adapter_slots * planner_dim
        + 2 * (planner_dim * planner_dim + planner_dim * 4 * planner_dim)
        + planner_dim * world_dim
        + world_dim;
    decoder_adapter + embed + kind_embed + n_layers * per_block + ln_final + lm_head
}

fn default_decoder_vocab_path(decoder_path: &Path) -> PathBuf {
    decoder_path.with_extension("vocab.txt")
}

fn decoder_tokenization_mode(kind: DecoderKind) -> TokenizationMode {
    if kind == DecoderKind::CodeSpecialist {
        TokenizationMode::CodeAware
    } else {
        TokenizationMode::Default
    }
}

#[derive(Clone)]
struct DecoderTrainConfig {
    encoder_model_path: PathBuf,
    encoder_vocab_path: PathBuf,
    world_model_path: PathBuf,
    data_path: PathBuf,
    steps: usize,
    batch_size: usize,
    max_seq: usize,
    dim: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    lr: f64,
    log_every: usize,
    grad_accum_steps: usize,
    train_dtype: DType,
    syntax_loss_weight: f64,
    signature_loss_weight: f64,
    init_decoder_path: Option<PathBuf>,
    decoder_kind: DecoderKind,
    decoder_vocab_path: Option<PathBuf>,
    decoder_max_vocab: usize,
    /// If set, save decoder here (e.g. text_decoder_26M.safetensors). Default: code_decoder_<size>.safetensors next to world model.
    decoder_output_path: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct ActionMetrics {
    accuracy: f32,
    balanced_accuracy: f32,
    macro_f1: f32,
    code_precision: f32,
    code_recall: f32,
    code_f1: f32,
    code_rate: f32,
    pred_code_rate: f32,
    done_precision: f32,
    done_recall: f32,
    done_f1: f32,
    done_rate: f32,
    pred_done_rate: f32,
}

struct WorldBatchMetrics {
    total_loss: f32,
    transition_loss: f32,
    sigreg_loss: f32,
    action_loss: f32,
    inverse_loss: f32,
    action_metrics: ActionMetrics,
    inverse_action_metrics: ActionMetrics,
    transition_cosine: f32,
}

struct DecoderBatchMetrics {
    loss: f32,
    ablated_loss: f32,
    conditioning_gain: f32,
    syntax_loss: f32,
    signature_loss: f32,
    perplexity: f32,
    active_tokens: f32,
    active_frac: f32,
    world_rms: f32,
    oov_rate: f32,
    token_accuracy: f32,
    identifier_accuracy: f32,
    delimiter_balance_rate: f32,
    syntax_token_accuracy: f32,
    function_skeleton_rate: f32,
    signature_token_accuracy: f32,
    signature_exact_rate: f32,
}

impl DecoderTrainConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--grad-accum <int>] [--init-decoder <path>] [--decoder-output <path>]"
            );
        }
        let mut init_decoder_path = None;
        let mut decoder_output_path = None;
        let mut decoder_vocab_path = None;
        let mut decoder_kind = None;
        let mut lr_override = None;
        let mut grad_accum_steps = 1usize;
        let mut decoder_max_vocab = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--init-decoder" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--init-decoder requires path"))?;
                init_decoder_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-output" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-output requires path"))?;
                decoder_output_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-vocab" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-vocab requires path"))?;
                decoder_vocab_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-max-vocab" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-max-vocab requires integer"))?;
                decoder_max_vocab = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-max-vocab must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-kind" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-kind requires text|code"))?;
                decoder_kind = DecoderKind::from_flag(value);
                if decoder_kind.is_none() {
                    bail!("--decoder-kind must be one of: text, code");
                }
                i += 2;
                continue;
            }
            if args[i] == "--lr" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                lr_override = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--lr must be float"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--grad-accum" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--grad-accum requires integer"))?;
                grad_accum_steps = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--grad-accum must be integer"))?;
                i += 2;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
        }
        Ok(Self {
            decoder_kind: decoder_kind.unwrap_or(DecoderKind::CodeSpecialist),
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps: filtered
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(40_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(8),
            max_seq: filtered
                .get(6)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    if decoder_kind.unwrap_or(DecoderKind::CodeSpecialist)
                        == DecoderKind::CodeSpecialist
                    {
                        192
                    } else {
                        128
                    }
                }),
            dim: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(768),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            lr: lr_override.unwrap_or(3e-4),
            log_every: 100,
            grad_accum_steps: grad_accum_steps.max(1),
            train_dtype: std::env::var("TOFY_TRAIN_DTYPE")
                .ok()
                .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                    "f16" | "float16" | "fp16" => Some(DType::F16),
                    "bf16" => Some(DType::BF16),
                    "f32" | "float32" | "fp32" => Some(DType::F32),
                    _ => None,
                })
                .unwrap_or(DType::F32),
            syntax_loss_weight: std::env::var("TOFY_DECODER_SYNTAX_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35f64)
                .max(0.0),
            signature_loss_weight: std::env::var("TOFY_DECODER_SIGNATURE_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.45f64)
                .max(0.0),
            init_decoder_path,
            decoder_vocab_path,
            decoder_max_vocab: decoder_max_vocab.unwrap_or(16_000),
            decoder_output_path,
        })
    }
}

/// Cross-entropy for orchestrator action head: logits [B, C], labels len B (class indices). Returns scalar.
fn action_cross_entropy(logits: &Tensor, labels: &[u32], device: &Device) -> Result<Tensor> {
    let log_probs = ops::log_softmax(logits, 1)?;
    let b = logits.dim(0)?;
    let n_classes = logits.dim(1)?;
    let class_weights = balanced_class_weights(labels, n_classes);
    let sample_labels = labels.iter().take(b).copied().collect::<Vec<_>>();
    let indices = Tensor::from_vec(
        sample_labels.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        (b,),
        device,
    )?
    .unsqueeze(1)?;
    let nll = log_probs
        .gather(&indices, 1)?
        .squeeze(1)?
        .affine(-1.0, 0.0)?;
    let sample_weights = util::from_vec_like(
        sample_labels
            .iter()
            .map(|&label| class_weights.get(label as usize).copied().unwrap_or(1.0))
            .collect::<Vec<_>>(),
        (b,),
        &nll,
    )?;
    let weighted_nll = nll.broadcast_mul(&sample_weights)?;
    let normalizer = sample_weights.sum_all()?.clamp(1e-8, 1e10)?;
    Ok(weighted_nll.sum_all()?.broadcast_div(&normalizer)?)
}

fn balanced_class_weights(labels: &[u32], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &label in labels {
        if let Some(count) = counts.get_mut(label as usize) {
            *count += 1;
        }
    }
    let present = counts.iter().filter(|&&count| count > 0).count().max(1) as f32;
    let total = labels.len().max(1) as f32;
    counts
        .into_iter()
        .map(|count| {
            if count == 0 {
                1.0
            } else {
                (total / (present * count as f32)).clamp(0.5, 4.0)
            }
        })
        .collect()
}

fn predicted_positive_rate(confusion: &[Vec<usize>], positive_label: usize) -> f32 {
    let pred_total = confusion
        .iter()
        .map(|row| row.get(positive_label).copied().unwrap_or(0))
        .sum::<usize>() as f32;
    let total = confusion
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .sum::<usize>()
        .max(1) as f32;
    pred_total / total
}

fn class_prf(confusion: &[Vec<usize>], label: usize) -> (f32, f32, f32, f32) {
    let tp = confusion
        .get(label)
        .and_then(|row| row.get(label))
        .copied()
        .unwrap_or(0) as f32;
    let true_total = confusion
        .get(label)
        .map(|row| row.iter().sum::<usize>())
        .unwrap_or(0) as f32;
    let pred_total = confusion
        .iter()
        .map(|row| row.get(label).copied().unwrap_or(0))
        .sum::<usize>() as f32;
    let precision = if pred_total > 0.0 {
        tp / pred_total
    } else {
        0.0
    };
    let recall = if true_total > 0.0 {
        tp / true_total
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let total = confusion
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .sum::<usize>()
        .max(1) as f32;
    (precision, recall, f1, true_total / total)
}

fn world_selection_score(metrics: &WorldBatchMetrics) -> f32 {
    let routing_gap = 1.0
        - (0.65 * metrics.action_metrics.macro_f1
            + 0.35 * metrics.action_metrics.balanced_accuracy);
    let inverse_gap = 1.0
        - (0.65 * metrics.inverse_action_metrics.macro_f1
            + 0.35 * metrics.inverse_action_metrics.balanced_accuracy);
    let collapse_penalty = if metrics.action_metrics.code_rate > 0.10
        && metrics.action_metrics.pred_code_rate < 0.05
    {
        0.25
    } else {
        0.0
    } + if metrics.action_metrics.done_rate > 0.05
        && metrics.action_metrics.pred_done_rate < 0.02
    {
        0.10
    } else {
        0.0
    };
    routing_gap
        + 0.6 * inverse_gap
        + 0.05 * metrics.action_loss
        + 0.03 * metrics.inverse_loss
        + 0.01 * metrics.transition_loss
        + collapse_penalty
}

fn slot_delta_slots(next_slots: &Tensor, state_slots: &Tensor) -> Result<Tensor> {
    next_slots.broadcast_sub(state_slots).map_err(Into::into)
}

fn compute_action_metrics(logits: &Tensor, labels: &[u32]) -> Result<ActionMetrics> {
    let rows = util::vec2_f32(logits)?;
    let n_classes = rows.first().map(|row| row.len()).unwrap_or(0).max(1);
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0usize;
    let mut total = 0usize;

    for (row, &label) in rows.iter().zip(labels.iter()) {
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        if pred == label {
            correct += 1;
        }
        if let Some(row_counts) = confusion.get_mut(label as usize) {
            if let Some(cell) = row_counts.get_mut(pred as usize) {
                *cell += 1;
            }
        }
        total += 1;
    }

    let mut recall_sum = 0.0f32;
    let mut f1_sum = 0.0f32;
    let mut present = 0usize;
    for label in 0..n_classes {
        let (_precision, recall, f1, rate) = class_prf(&confusion, label);
        if rate > 0.0 {
            recall_sum += recall;
            f1_sum += f1;
            present += 1;
        }
    }
    let balanced_accuracy = if present == 0 {
        0.0
    } else {
        recall_sum / present as f32
    };
    let macro_f1 = if present == 0 {
        0.0
    } else {
        f1_sum / present as f32
    };
    let (code_precision, code_recall, code_f1, code_rate) =
        class_prf(&confusion, ACTION_CODE as usize);
    let (done_precision, done_recall, done_f1, done_rate) =
        class_prf(&confusion, ACTION_DONE as usize);

    Ok(ActionMetrics {
        accuracy: correct as f32 / total.max(1) as f32,
        balanced_accuracy,
        macro_f1,
        code_precision,
        code_recall,
        code_f1,
        code_rate,
        pred_code_rate: predicted_positive_rate(&confusion, ACTION_CODE as usize),
        done_precision,
        done_recall,
        done_f1,
        done_rate,
        pred_done_rate: predicted_positive_rate(&confusion, ACTION_DONE as usize),
    })
}

fn collect_action_training_batch(
    stream: &mut RawWorldStream,
    batch_size: usize,
    target_code_rate: f32,
    target_done_rate: f32,
) -> Result<Vec<RawWorldExample>> {
    let mut rng = rand::thread_rng();
    let target_code = ((batch_size as f32) * target_code_rate.clamp(0.0, 0.5)).round() as usize;
    let target_done = ((batch_size as f32) * target_done_rate.clamp(0.0, 0.4)).round() as usize;
    let target_code = target_code.min(batch_size.saturating_sub(1));
    let target_done = target_done.min(batch_size.saturating_sub(target_code));
    let target_text = batch_size.saturating_sub(target_code + target_done);
    let mut code_examples = Vec::new();
    let mut text_examples = Vec::new();
    let mut done_examples = Vec::new();

    for _ in 0..8 {
        for example in stream.next_batch(batch_size.max(1))? {
            match example.action_label {
                ACTION_CODE => code_examples.push(example),
                ACTION_DONE => done_examples.push(example),
                _ => text_examples.push(example),
            }
        }
        if code_examples.len() >= target_code
            && text_examples.len() >= target_text
            && done_examples.len() >= target_done
        {
            break;
        }
    }

    code_examples.shuffle(&mut rng);
    text_examples.shuffle(&mut rng);
    done_examples.shuffle(&mut rng);

    let take_code = target_code.min(code_examples.len());
    let take_text = target_text.min(text_examples.len());
    let take_done = target_done.min(done_examples.len());
    let mut batch = Vec::with_capacity(batch_size);
    batch.extend(code_examples.drain(..take_code));
    batch.extend(text_examples.drain(..take_text));
    batch.extend(done_examples.drain(..take_done));

    let mut leftovers = Vec::new();
    leftovers.extend(code_examples);
    leftovers.extend(text_examples);
    leftovers.extend(done_examples);
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

fn raw_examples_oov_rate(rows: &[RawWorldExample], vocab: &Vocab, mode: TokenizationMode) -> f32 {
    let mut total = 0usize;
    let mut oov = 0usize;
    for row in rows {
        let state_ids = encode_text_with_vocab_mode(&row.state_text, vocab, mode);
        let next_ids = encode_text_with_vocab_mode(&row.next_text, vocab, mode);
        total += state_ids.len() + next_ids.len();
        oov += state_ids
            .iter()
            .chain(next_ids.iter())
            .filter(|&&id| id == vocab.unk_id)
            .count();
    }
    if total == 0 {
        0.0
    } else {
        oov as f32 / total as f32
    }
}

fn is_identifier_token(token: &str) -> bool {
    !token.is_empty()
        && token != "<num_lit>"
        && token != "<str_lit>"
        && token.chars().all(|ch| ch.is_ascii_alphanumeric())
        && token
            .chars()
            .next()
            .map(|ch| ch.is_ascii_alphabetic())
            .unwrap_or(false)
}

fn delimiter_balance_for_tokens(tokens: &[String]) -> bool {
    let mut round = 0i32;
    let mut square = 0i32;
    let mut curly = 0i32;
    for token in tokens {
        match token.as_str() {
            "(" => round += 1,
            ")" => round -= 1,
            "[" => square += 1,
            "]" => square -= 1,
            "{" => curly += 1,
            "}" => curly -= 1,
            _ => {}
        }
        if round < 0 || square < 0 || curly < 0 {
            return false;
        }
    }
    round == 0 && square == 0 && curly == 0
}

fn decode_active_tokens(ids: &[u32], mask: &[f32], vocab: &Vocab) -> Vec<String> {
    ids.iter()
        .zip(mask.iter())
        .filter_map(|(&id, &m)| {
            if m <= 0.0 {
                None
            } else {
                Some(
                    vocab
                        .id_to_token
                        .get(id as usize)
                        .cloned()
                        .unwrap_or_else(|| "<unk>".to_string()),
                )
            }
        })
        .collect()
}

fn is_syntax_token(token: &str) -> bool {
    matches!(
        token,
        "{" | "}"
            | "("
            | ")"
            | "["
            | "]"
            | ";"
            | ","
            | ":"
            | "::"
            | "=>"
            | "->"
            | "="
            | "=="
            | "<nl>"
            | "<indent_tab>"
            | "fn"
            | "pub"
            | "impl"
            | "struct"
            | "enum"
            | "match"
            | "return"
            | "let"
            | "use"
    )
}

fn syntax_weight_for_token(token: &str) -> f32 {
    if matches!(
        token,
        "{" | "}" | "(" | ")" | "[" | "]" | ";" | "fn" | "pub" | "impl" | "struct" | "enum"
    ) {
        2.0
    } else if is_syntax_token(token) {
        1.5
    } else {
        1.0
    }
}

fn syntax_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_ids = target.reshape((target.elem_count(),))?.to_vec1::<u32>()?;
    let mask_values = util::vec1_f32(&mask.reshape((mask.elem_count(),))?)?;
    let weights = target_ids
        .iter()
        .zip(mask_values.iter())
        .map(|(&id, &m)| {
            if m <= 0.0 {
                0.0
            } else {
                vocab
                    .id_to_token
                    .get(id as usize)
                    .map(|token| syntax_weight_for_token(token))
                    .unwrap_or(1.0)
            }
        })
        .collect::<Vec<_>>();
    let mask_like = mask.reshape((mask.elem_count(),))?;
    util::from_vec_like(weights, (target.elem_count(),), &mask_like)
}

fn signature_span_indices(ids: &[u32], mask: &[f32], vocab: &Vocab) -> Vec<usize> {
    let mut start = None;
    let mut end = None;
    for (idx, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
        if m <= 0.0 {
            continue;
        }
        let token = vocab
            .id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>");
        if start.is_none() && (token == "pub" || token == "fn") {
            start = Some(idx);
        }
        if start.is_some() && token == "{" {
            end = Some(idx);
            break;
        }
    }
    match (start, end) {
        (Some(s), Some(e)) if e >= s => (s..=e).collect(),
        _ => Vec::new(),
    }
}

fn signature_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let seq_len = target.dim(1)?;
    let mut weights = Vec::with_capacity(target.elem_count());
    for (target_row, mask_row) in target_rows.iter().zip(mask_rows.iter()) {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        for (idx, &m) in mask_row.iter().enumerate().take(seq_len) {
            if m <= 0.0 {
                weights.push(0.0);
            } else if signature_positions.contains(&idx) {
                weights.push(2.5);
            } else {
                weights.push(1.0);
            }
        }
    }
    util::from_vec_like(weights, (target.elem_count(),), mask)
}

fn rust_function_skeleton_for_tokens(tokens: &[String]) -> bool {
    let has_fn = tokens.iter().any(|token| token == "fn");
    let has_parens =
        tokens.iter().any(|token| token == "(") && tokens.iter().any(|token| token == ")");
    let has_body =
        tokens.iter().any(|token| token == "{") && tokens.iter().any(|token| token == "}");
    has_fn && has_parens && has_body && delimiter_balance_for_tokens(tokens)
}

fn decoder_prediction_metrics(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
) -> Result<(f32, f32, f32, f32, f32, f32, f32)> {
    let pred = logits.argmax(candle_core::D::Minus1)?;
    let pred_rows = pred.to_vec2::<u32>()?;
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut ident_total = 0usize;
    let mut ident_correct = 0usize;
    let mut balanced = 0usize;
    let mut syntax_total = 0usize;
    let mut syntax_correct = 0usize;
    let mut function_skeletons = 0usize;
    let mut signature_total = 0usize;
    let mut signature_correct = 0usize;
    let mut signature_exact = 0usize;
    for ((pred_row, target_row), mask_row) in pred_rows
        .iter()
        .zip(target_rows.iter())
        .zip(mask_rows.iter())
    {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab);
        let mut row_signature_ok = !signature_positions.is_empty();
        for ((&pred_id, &target_id), &m) in
            pred_row.iter().zip(target_row.iter()).zip(mask_row.iter())
        {
            if m <= 0.0 {
                continue;
            }
            total += 1;
            if pred_id == target_id {
                correct += 1;
            }
            let token = vocab
                .id_to_token
                .get(target_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            if is_identifier_token(token) {
                ident_total += 1;
                if pred_id == target_id {
                    ident_correct += 1;
                }
            }
            if is_syntax_token(token) {
                syntax_total += 1;
                if pred_id == target_id {
                    syntax_correct += 1;
                }
            }
        }
        for &idx in &signature_positions {
            signature_total += 1;
            if pred_row.get(idx).copied() == target_row.get(idx).copied() {
                signature_correct += 1;
            } else {
                row_signature_ok = false;
            }
        }
        if row_signature_ok {
            signature_exact += 1;
        }
        let pred_tokens = decode_active_tokens(pred_row, mask_row, vocab);
        if delimiter_balance_for_tokens(&pred_tokens) {
            balanced += 1;
        }
        if rust_function_skeleton_for_tokens(&pred_tokens) {
            function_skeletons += 1;
        }
    }
    let token_accuracy = correct as f32 / total.max(1) as f32;
    let identifier_accuracy = ident_correct as f32 / ident_total.max(1) as f32;
    let delimiter_balance_rate = balanced as f32 / pred_rows.len().max(1) as f32;
    let syntax_token_accuracy = syntax_correct as f32 / syntax_total.max(1) as f32;
    let function_skeleton_rate = function_skeletons as f32 / pred_rows.len().max(1) as f32;
    let signature_token_accuracy = signature_correct as f32 / signature_total.max(1) as f32;
    let signature_exact_rate = signature_exact as f32 / pred_rows.len().max(1) as f32;
    Ok((
        token_accuracy,
        identifier_accuracy,
        delimiter_balance_rate,
        syntax_token_accuracy,
        function_skeleton_rate,
        signature_token_accuracy,
        signature_exact_rate,
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_world_batch(
    raw_batch: &[RawWorldExample],
    encoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    planner_memory: &PlannerMemory,
    transition: &WorldTransition,
    orchestrator_head: &OrchestratorActionHead,
    inverse_action_head: &OrchestratorActionHead,
    max_seq: usize,
    lambda: f64,
    action_loss_weight: f64,
    inverse_loss_weight: f64,
    device: &Device,
) -> Result<WorldBatchMetrics> {
    const SIGREG_SLICES: usize = 128;
    const SIGREG_POINTS: usize = 17;
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    let batch = encode_world_examples(raw_batch, encoder_vocab);
    let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
    let state_slots = if context_segments > 1 || recursive_planner_memory {
        let state_tokens = batch
            .iter()
            .map(|row| row.state_tokens.clone())
            .collect::<Vec<_>>();
        planner_slots_from_token_sequences(
            encoder,
            planner_memory,
            &state_tokens,
            encoder_vocab.pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_planner_memory,
            device,
        )?
    } else {
        let (state_ids, _next_ids, state_lens, _next_lens, _) =
            make_world_batch_from_slice(&batch, max_seq, encoder_vocab.pad_id, device)?;
        let state_features = encoder.forward_features(&state_ids)?.detached();
        planner_forward_encoder_masked(planner_memory, &state_features, &state_lens)?
    };
    let next_slots = if context_segments > 1 || recursive_planner_memory {
        let next_tokens = batch
            .iter()
            .map(|row| row.next_tokens.clone())
            .collect::<Vec<_>>();
        planner_slots_from_token_sequences(
            encoder,
            planner_memory,
            &next_tokens,
            encoder_vocab.pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_planner_memory,
            device,
        )?
    } else {
        let (_state_ids, next_ids, _state_lens, next_lens, _) =
            make_world_batch_from_slice(&batch, max_seq, encoder_vocab.pad_id, device)?;
        let next_features = encoder.forward_features(&next_ids)?.detached();
        planner_forward_encoder_masked(planner_memory, &next_features, &next_lens)?
    };
    let pred_next_slots = transition.forward(&state_slots, &action_labels)?;
    let transition_loss = prediction_loss(&pred_next_slots, &next_slots)?;
    let state_sigreg = sigreg_epps_pulley(
        &flatten_latent_slots(&state_slots)?,
        SIGREG_SLICES,
        SIGREG_POINTS,
    )?;
    let next_sigreg = sigreg_epps_pulley(
        &flatten_latent_slots(&next_slots)?,
        SIGREG_SLICES,
        SIGREG_POINTS,
    )?;
    let pred_sigreg = sigreg_epps_pulley(
        &flatten_latent_slots(&pred_next_slots)?,
        SIGREG_SLICES,
        SIGREG_POINTS,
    )?;
    let sigreg_loss = state_sigreg
        .broadcast_add(&next_sigreg)?
        .broadcast_add(&pred_sigreg)?
        .affine(1.0 / 3.0, 0.0)?;
    let action_logits = orchestrator_head.forward(&state_slots)?;
    let action_loss = action_cross_entropy(&action_logits, &action_labels, device)?;
    let true_delta_slots = slot_delta_slots(&next_slots, &state_slots)?;
    let pred_delta_slots = slot_delta_slots(&pred_next_slots, &state_slots)?;
    let inverse_logits_true = inverse_action_head.forward(&true_delta_slots)?;
    let inverse_logits_pred = inverse_action_head.forward(&pred_delta_slots)?;
    let inverse_true_loss = action_cross_entropy(&inverse_logits_true, &action_labels, device)?;
    let inverse_pred_loss = action_cross_entropy(&inverse_logits_pred, &action_labels, device)?;
    let inverse_loss = inverse_true_loss
        .broadcast_add(&inverse_pred_loss)?
        .affine(0.5, 0.0)?;
    let total_loss = transition_loss
        .affine(1.0 - lambda, 0.0)?
        .broadcast_add(&sigreg_loss.affine(lambda, 0.0)?)?
        .broadcast_add(&action_loss.affine(action_loss_weight, 0.0)?)?
        .broadcast_add(&inverse_loss.affine(inverse_loss_weight, 0.0)?)?;
    let pred_slots_flat = flatten_latent_slots(&pred_next_slots)?;
    let next_slots_flat = flatten_latent_slots(&next_slots)?;
    Ok(WorldBatchMetrics {
        total_loss: util::scalar_f32(&total_loss)?,
        transition_loss: util::scalar_f32(&transition_loss)?,
        sigreg_loss: util::scalar_f32(&sigreg_loss)?,
        action_loss: util::scalar_f32(&action_loss)?,
        inverse_loss: util::scalar_f32(&inverse_loss)?,
        action_metrics: compute_action_metrics(&action_logits, &action_labels)?,
        inverse_action_metrics: compute_action_metrics(&inverse_logits_pred, &action_labels)?,
        transition_cosine: util::scalar_f32(&mean_cosine_similarity(
            &pred_slots_flat,
            &next_slots_flat,
        )?)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_decoder_batch(
    raw_batch: &[RawWorldExample],
    encoder_vocab: &Vocab,
    decoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    planner_memory: &PlannerMemory,
    transition: &WorldTransition,
    decoder_adapter: &DecoderAdapter,
    decoder: &CodeDecoder,
    decoder_kind: DecoderKind,
    decoder_action_label: u32,
    max_seq: usize,
    device: &Device,
) -> Result<DecoderBatchMetrics> {
    let decoder_token_mode = decoder_tokenization_mode(decoder_kind);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    let rollout_steps = env_usize("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", 1);
    let encoder_batch = encode_world_examples(raw_batch, encoder_vocab);
    let decoder_batch =
        encode_world_examples_with_mode(raw_batch, decoder_vocab, decoder_token_mode);
    let state_tokens = encoder_batch
        .iter()
        .map(|row| row.state_tokens.clone())
        .collect::<Vec<_>>();
    let state_slots = planner_slots_from_token_sequences(
        encoder,
        planner_memory,
        &state_tokens,
        encoder_vocab.pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_planner_memory,
        device,
    )?;
    let decoder_action_labels = vec![decoder_action_label; encoder_batch.len()];
    let next_planner_slots = if rollout_steps <= 1 {
        transition.forward(&state_slots, &decoder_action_labels)?
    } else {
        rollout_transition_slots(
            transition,
            &state_slots,
            decoder_action_label,
            rollout_steps,
        )?
    };
    let world_latent = decoder_adapter.forward(&next_planner_slots.detach())?;
    let zero_world_latent = world_latent.affine(0.0, 0.0)?;
    let (dec_state_ids, dec_next_ids, state_lens, next_lens, _) =
        make_world_batch_from_slice(&decoder_batch, max_seq, decoder_vocab.pad_id, device)?;
    let (dec_input, dec_target, loss_mask) = make_decoder_batch(
        &dec_state_ids,
        &dec_next_ids,
        &state_lens,
        &next_lens,
        max_seq,
        decoder_vocab.pad_id,
        device,
    )?;
    let logits = decoder.forward(&dec_input, &world_latent)?;
    let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
    let syntax_mask = syntax_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let signature_mask = signature_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let loss = masked_cross_entropy(&logits, &dec_target, &loss_mask)?;
    let ablated_loss = masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
    let syntax_loss = masked_weighted_cross_entropy(&logits, &dec_target, &syntax_mask)?;
    let signature_loss = masked_weighted_cross_entropy(&logits, &dec_target, &signature_mask)?;
    let loss_val = util::scalar_f32(&loss)?;
    let ablated_loss_val = util::scalar_f32(&ablated_loss)?;
    let syntax_loss_val = util::scalar_f32(&syntax_loss)?;
    let signature_loss_val = util::scalar_f32(&signature_loss)?;
    let active_tokens = util::scalar_f32(&loss_mask.sum_all()?)?;
    let total_tokens = (state_lens.len().max(1) * max_seq * 2) as f32;
    let (
        token_accuracy,
        identifier_accuracy,
        delimiter_balance_rate,
        syntax_token_accuracy,
        function_skeleton_rate,
        signature_token_accuracy,
        signature_exact_rate,
    ) = decoder_prediction_metrics(&logits, &dec_target, &loss_mask, decoder_vocab)?;
    Ok(DecoderBatchMetrics {
        loss: loss_val,
        ablated_loss: ablated_loss_val,
        conditioning_gain: ablated_loss_val - loss_val,
        syntax_loss: syntax_loss_val,
        signature_loss: signature_loss_val,
        perplexity: loss_val.exp(),
        active_tokens,
        active_frac: active_tokens / total_tokens.max(1.0),
        world_rms: util::scalar_f32(&tensor_rms(&world_latent)?)?,
        oov_rate: raw_examples_oov_rate(raw_batch, decoder_vocab, decoder_token_mode),
        token_accuracy,
        identifier_accuracy,
        delimiter_balance_rate,
        syntax_token_accuracy,
        function_skeleton_rate,
        signature_token_accuracy,
        signature_exact_rate,
    })
}

/// Masked cross-entropy: mean over positions where mask > 0. logits [B,T,V], target [B,T] u32, mask [B,T].
fn masked_weighted_cross_entropy(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?;
    let target_flat = target
        .reshape((b * t,))?
        .to_dtype(candle_core::DType::U32)?;
    let log_probs = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
    let nll_per = log_probs
        .gather(&target_flat.unsqueeze(1)?, 1)?
        .squeeze(1)?
        .affine(-1.0, 0.0)?;
    let mask_flat = mask.reshape((b * t,))?.to_dtype(nll_per.dtype())?;
    let sum_nll = (nll_per.broadcast_mul(&mask_flat)?).sum_all()?;
    let sum_mask = mask_flat.sum_all()?.clamp(1e-8, 1e10)?;
    Ok(sum_nll.broadcast_div(&sum_mask)?)
}

fn masked_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    masked_weighted_cross_entropy(logits, target, mask)
}

fn decoder_selection_score(
    metrics: &DecoderBatchMetrics,
    syntax_loss_weight: f64,
    signature_loss_weight: f64,
) -> f32 {
    metrics.loss
        + 0.20 * (0.05 - metrics.conditioning_gain).max(0.0)
        + (syntax_loss_weight as f32 * 0.5 * metrics.syntax_loss)
        + (signature_loss_weight as f32 * 0.5 * metrics.signature_loss)
        - 0.08 * metrics.syntax_token_accuracy
        - 0.08 * metrics.signature_token_accuracy
        - 0.06 * metrics.delimiter_balance_rate
        - 0.06 * metrics.function_skeleton_rate
        - 0.08 * metrics.signature_exact_rate
        - 0.04 * metrics.conditioning_gain.max(0.0)
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

    let data_path = resolve_world_data_path(&config.data_path.to_string_lossy())?;
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
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), config.bridge_dim)?;
    let _inverse_action_head =
        if checkpoint_has_prefix(&config.world_model_path, "inverse_action_head.") {
            Some(OrchestratorActionHead::new(
                world_vb.pp("inverse_action_head"),
                config.bridge_dim,
            )?)
        } else {
            None
        };
    world_varmap.load(&config.world_model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;

    let mut decoder_varmap = VarMap::new();
    if let Some(ref p) = config.init_decoder_path {
        decoder_varmap.load(p)?;
        util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;
    }
    let decoder_vb = VarBuilder::from_varmap(&decoder_varmap, train_dtype, &device);
    let decoder_adapter = DecoderAdapter::new(
        decoder_vb.pp("decoder_adapter"),
        config.bridge_dim,
        config.bridge_dim,
        DecoderAdapter::output_slots_for(config.decoder_kind, config.num_latent_tokens),
    )?;
    let decoder_path = config
        .decoder_output_path
        .clone()
        .map(|p| {
            // Put under local_models only if path is relative and not already under local_models
            if p.is_relative() && !p.starts_with("local_models") {
                PathBuf::from("local_models").join(p)
            } else {
                p
            }
        })
        .unwrap_or_else(|| {
            config
                .world_model_path
                .parent()
                .unwrap_or_else(|| Path::new("local_models"))
                .join(format!(
                    "{}_decoder.safetensors",
                    if config.decoder_kind == DecoderKind::TextGeneralist {
                        "text"
                    } else {
                        "code"
                    }
                ))
        });
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
    let vocab_size = decoder_vocab.id_to_token.len();
    let decoder = CodeDecoder::new(
        decoder_vb.pp("decoder"),
        vocab_size,
        DECODER_DIM,
        config.bridge_dim,
        DECODER_LAYERS,
        DECODER_HEADS,
        DECODER_FF_DIM,
        config.decoder_kind,
    )?;
    util::cast_varmap_dtype(&mut decoder_varmap, train_dtype)?;

    let train_vars = decoder_varmap.all_vars();
    let mut opt = candle_nn::AdamW::new_lr(train_vars.clone(), config.lr)?;

    let decoder_params = decoder_param_count(
        vocab_size,
        config.bridge_dim,
        config.bridge_dim,
        config.decoder_kind,
        config.num_latent_tokens,
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
        DECODER_DIM,
        DECODER_LAYERS,
        DECODER_HEADS,
        DECODER_FF_DIM,
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
    let conditioning_loss_weight = std::env::var("TOFY_DECODER_CONDITIONING_LOSS_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.30f64)
        .max(0.0);
    let conditioning_margin = std::env::var("TOFY_DECODER_CONDITIONING_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.10f64)
        .max(0.0);
    println!(
        "TensorBoard run dir: {} (view with: tensorboard --logdir runs)",
        run_dir
    );
    tb.add_scalar("run/alive", 1.0, 0);
    tb.add_scalar("config/batch_size", config.batch_size as f32, 0);
    tb.add_scalar("config/max_seq", config.max_seq as f32, 0);
    tb.add_scalar("config/planner_slots", config.num_latent_tokens as f32, 0);
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
        "config/conditioning_loss_weight",
        conditioning_loss_weight as f32,
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

    let mut best_loss = f32::MAX;
    let mut best_metric = f32::MAX;
    let mut saved_checkpoint = false;
    let decoder_action_label = if config.decoder_kind == DecoderKind::TextGeneralist {
        0
    } else {
        1
    };
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 1);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    let rollout_steps = env_usize("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", 1);

    for step in 1..=config.steps {
        let mut accumulated_grads = None;
        let mut last_raw_batch = None;
        let mut last_world_latent = None;
        let mut last_logits = None;
        let mut last_dec_target = None;
        let mut last_loss_mask = None;
        let mut last_loss = None;
        let mut last_ablated_loss = None;
        let mut last_conditioning_loss = None;
        let mut last_syntax_loss = None;
        let mut last_signature_loss = None;

        for _micro_step in 0..config.grad_accum_steps.max(1) {
            let raw_batch = raw_stream.next_batch(config.batch_size.max(1))?;
            let encoder_batch = encode_world_examples(&raw_batch, &encoder_vocab);
            let decoder_batch =
                encode_world_examples_with_mode(&raw_batch, &decoder_vocab, decoder_token_mode);
            let state_tokens = encoder_batch
                .iter()
                .map(|row| row.state_tokens.clone())
                .collect::<Vec<_>>();
            let state_slots = planner_slots_from_token_sequences(
                &encoder,
                &planner_memory,
                &state_tokens,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                recursive_planner_memory,
                &device,
            )?;
            let decoder_action_labels = vec![decoder_action_label; encoder_batch.len()];
            let next_planner_slots = if rollout_steps <= 1 {
                transition.forward(&state_slots, &decoder_action_labels)?
            } else {
                rollout_transition_slots(
                    &transition,
                    &state_slots,
                    decoder_action_label,
                    rollout_steps,
                )?
            };
            let world_latent = decoder_adapter.forward(&next_planner_slots.detach())?;
            let zero_world_latent = world_latent.affine(0.0, 0.0)?;

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
            let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
            let syntax_mask = syntax_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let signature_mask =
                signature_weight_mask(&dec_target, &loss_mask, &decoder_vocab, &device)?;
            let token_loss = masked_cross_entropy(&logits, &dec_target, &loss_mask)?;
            let ablated_loss = masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
            let syntax_loss = masked_weighted_cross_entropy(&logits, &dec_target, &syntax_mask)?;
            let signature_loss =
                masked_weighted_cross_entropy(&logits, &dec_target, &signature_mask)?;
            let conditioning_loss = token_loss
                .broadcast_sub(&ablated_loss.detach())?
                .affine(1.0, conditioning_margin)?
                .relu()?;
            let loss = token_loss
                .broadcast_add(&syntax_loss.affine(config.syntax_loss_weight, 0.0)?)?
                .broadcast_add(&signature_loss.affine(config.signature_loss_weight, 0.0)?)?
                .broadcast_add(&conditioning_loss.affine(conditioning_loss_weight, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                config.grad_accum_steps,
            )?;

            last_raw_batch = Some(raw_batch);
            last_world_latent = Some(world_latent);
            last_logits = Some(logits);
            last_dec_target = Some(dec_target);
            last_loss_mask = Some(loss_mask);
            last_loss = Some(loss);
            last_ablated_loss = Some(ablated_loss);
            last_conditioning_loss = Some(conditioning_loss);
            last_syntax_loss = Some(syntax_loss);
            last_signature_loss = Some(signature_loss);
        }

        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let raw_batch =
                last_raw_batch.context("decoder grad accumulation produced no raw batch")?;
            let world_latent =
                last_world_latent.context("decoder grad accumulation produced no latent")?;
            let logits = last_logits.context("decoder grad accumulation produced no logits")?;
            let dec_target =
                last_dec_target.context("decoder grad accumulation produced no targets")?;
            let loss_mask =
                last_loss_mask.context("decoder grad accumulation produced no loss mask")?;
            let loss = last_loss.context("decoder grad accumulation produced no loss")?;
            let ablated_loss =
                last_ablated_loss.context("decoder grad accumulation produced no ablated loss")?;
            let conditioning_loss = last_conditioning_loss
                .context("decoder grad accumulation produced no conditioning loss")?;
            let syntax_loss =
                last_syntax_loss.context("decoder grad accumulation produced no syntax loss")?;
            let signature_loss = last_signature_loss
                .context("decoder grad accumulation produced no signature loss")?;
            let loss_val = util::scalar_f32(&loss)?;
            let ablated_loss_val = util::scalar_f32(&ablated_loss)?;
            let conditioning_loss_val = util::scalar_f32(&conditioning_loss)?;
            let syntax_loss_val = util::scalar_f32(&syntax_loss)?;
            let signature_loss_val = util::scalar_f32(&signature_loss)?;
            let active_tokens = util::scalar_f32(&loss_mask.sum_all()?)?;
            let total_tokens = (config.batch_size.max(1) * config.max_seq * 2) as f32;
            let active_frac = active_tokens / total_tokens.max(1.0);
            let perplexity = loss_val.exp();
            let conditioning_gain = ablated_loss_val - loss_val;
            let world_rms = util::scalar_f32(&tensor_rms(&world_latent)?)?;
            let oov_rate = raw_examples_oov_rate(&raw_batch, &decoder_vocab, decoder_token_mode);
            let (
                token_accuracy,
                identifier_accuracy,
                delimiter_balance_rate,
                syntax_token_accuracy,
                function_skeleton_rate,
                signature_token_accuracy,
                signature_exact_rate,
            ) = decoder_prediction_metrics(&logits, &dec_target, &loss_mask, &decoder_vocab)?;

            tb.add_scalar("loss/token_ce", loss_val, step);
            tb.add_scalar("loss/ablated_token_ce", ablated_loss_val, step);
            tb.add_scalar("loss/conditioning_margin", conditioning_loss_val, step);
            tb.add_scalar("loss/syntax_ce", syntax_loss_val, step);
            tb.add_scalar("loss/signature_ce", signature_loss_val, step);
            tb.add_scalar("metrics/perplexity", perplexity, step);
            tb.add_scalar("metrics/active_tokens", active_tokens, step);
            tb.add_scalar("metrics/active_frac", active_frac, step);
            tb.add_scalar("metrics/world_latent_rms", world_rms, step);
            tb.add_scalar("metrics/conditioning_gain", conditioning_gain, step);
            tb.add_scalar("metrics/oov_rate", oov_rate, step);
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
                ablated_loss: ablated_loss_val,
                conditioning_gain,
                syntax_loss: syntax_loss_val,
                signature_loss: signature_loss_val,
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
            };
            let mut selection_metric = decoder_selection_score(
                &train_metrics,
                config.syntax_loss_weight,
                config.signature_loss_weight,
            );
            if let Some(ref mut val_stream) = val_stream {
                let val_raw_batch = val_stream.next_batch(config.batch_size.max(1))?;
                let val_metrics = evaluate_decoder_batch(
                    &val_raw_batch,
                    &encoder_vocab,
                    &decoder_vocab,
                    &encoder,
                    &planner_memory,
                    &transition,
                    &decoder_adapter,
                    &decoder,
                    config.decoder_kind,
                    decoder_action_label,
                    config.max_seq,
                    &device,
                )?;
                selection_metric = decoder_selection_score(
                    &val_metrics,
                    config.syntax_loss_weight,
                    config.signature_loss_weight,
                );
                best_loss = best_loss.min(val_metrics.loss);
                tb.add_scalar("val/token_ce", val_metrics.loss, step);
                tb.add_scalar("val/ablated_token_ce", val_metrics.ablated_loss, step);
                tb.add_scalar("val/syntax_ce", val_metrics.syntax_loss, step);
                tb.add_scalar("val/signature_ce", val_metrics.signature_loss, step);
                tb.add_scalar("val/perplexity", val_metrics.perplexity, step);
                tb.add_scalar("val/active_tokens", val_metrics.active_tokens, step);
                tb.add_scalar("val/active_frac", val_metrics.active_frac, step);
                tb.add_scalar("val/world_latent_rms", val_metrics.world_rms, step);
                tb.add_scalar("val/conditioning_gain", val_metrics.conditioning_gain, step);
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
                decoder_varmap.save(&decoder_path)?;
                saved_checkpoint = true;
                println!(
                    "step {}/{} token_ce {:.4} ablate_ce {:.4} cond_gain {:.4} cond_loss {:.4} syntax_ce {:.4} sig_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% delim {:.2}% fn_skel {:.2}% sel {:.4}{} [saved best]",
                    step,
                    config.steps,
                    loss_val,
                    ablated_loss_val,
                    conditioning_gain,
                    conditioning_loss_val,
                    syntax_loss_val,
                    signature_loss_val,
                    perplexity,
                    active_frac * 100.0,
                    oov_rate * 100.0,
                    token_accuracy * 100.0,
                    identifier_accuracy * 100.0,
                    syntax_token_accuracy * 100.0,
                    signature_token_accuracy * 100.0,
                    signature_exact_rate * 100.0,
                    delimiter_balance_rate * 100.0,
                    function_skeleton_rate * 100.0,
                    selection_metric,
                    memory_note
                );
            } else {
                println!(
                    "step {}/{} token_ce {:.4} ablate_ce {:.4} cond_gain {:.4} cond_loss {:.4} syntax_ce {:.4} sig_ce {:.4} ppl {:.2} active {:.1}% oov {:.2}% tok_acc {:.2}% ident_acc {:.2}% syntax_acc {:.2}% sig_acc {:.2}% sig_exact {:.2}% delim {:.2}% fn_skel {:.2}% sel {:.4}{}",
                    step,
                    config.steps,
                    loss_val,
                    ablated_loss_val,
                    conditioning_gain,
                    conditioning_loss_val,
                    syntax_loss_val,
                    signature_loss_val,
                    perplexity,
                    active_frac * 100.0,
                    oov_rate * 100.0,
                    token_accuracy * 100.0,
                    identifier_accuracy * 100.0,
                    syntax_token_accuracy * 100.0,
                    signature_token_accuracy * 100.0,
                    signature_exact_rate * 100.0,
                    delimiter_balance_rate * 100.0,
                    function_skeleton_rate * 100.0,
                    selection_metric,
                    memory_note
                );
            }
        }
    }

    if !saved_checkpoint {
        decoder_varmap.save(&decoder_path)?;
        println!(
            "No checkpoint was saved during logging; saved final decoder weights to {:?}",
            decoder_path
        );
    }
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
fn run_eval_world(
    encoder_model_path: &PathBuf,
    encoder_vocab_path: &PathBuf,
    model_path: &PathBuf,
    data_arg: &str,
    eval_steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let runtime_dtype = util::resolve_runtime_dtype(&device);
    let encoder_vocab = load_vocab_from_file(encoder_vocab_path)?;
    let data_path = resolve_world_data_path(data_arg)?;
    let row_count = count_raw_world_rows(&data_path)?;
    let mut raw_stream = RawWorldStream::new(&data_path)?;

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
    world_varmap.load(model_path)?;
    util::cast_varmap_dtype(&mut world_varmap, runtime_dtype)?;
    let world_vb = VarBuilder::from_varmap(&world_varmap, runtime_dtype, &device);
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        dim,
        bridge_dim,
        num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), bridge_dim)?;
    let orchestrator_head =
        OrchestratorActionHead::new(world_vb.pp("orchestrator_action_head"), bridge_dim)?;

    println!("World-model evaluation");
    println!("model: {:?}", model_path);
    println!("encoder: {:?}", encoder_model_path);
    println!("rows: {}", row_count);
    println!(
        "Streaming shuffle buffer: {}",
        DEFAULT_STREAM_SHUFFLE_BUFFER
    );
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={} planner_dim={} planner_slots={}",
        eval_steps, batch_size, dim, max_seq, num_layers, num_heads, bridge_dim, num_latent_tokens
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
    let recursive_planner_memory = env_bool("TOFY_RECURSIVE_PLANNER_MEMORY", context_segments > 1);
    for _ in 0..eval_steps.max(1) {
        let raw_batch = raw_stream.next_batch(batch_size.max(1))?;
        let chunk = encode_world_examples(&raw_batch, &encoder_vocab);
        let action_labels = chunk.iter().map(|row| row.action_label).collect::<Vec<_>>();
        let state_tokens = chunk
            .iter()
            .map(|row| row.state_tokens.clone())
            .collect::<Vec<_>>();
        let next_tokens = chunk
            .iter()
            .map(|row| row.next_tokens.clone())
            .collect::<Vec<_>>();
        let state_slots = planner_slots_from_token_sequences(
            &encoder,
            &planner_memory,
            &state_tokens,
            encoder_vocab.pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_planner_memory,
            &device,
        )?;
        let next_slots = planner_slots_from_token_sequences(
            &encoder,
            &planner_memory,
            &next_tokens,
            encoder_vocab.pad_id,
            max_seq,
            context_segments,
            recent_full_segments,
            recursive_planner_memory,
            &device,
        )?;
        let pred_slots = transition.forward(&state_slots, &action_labels)?;
        let pred_loss = util::scalar_f32(&prediction_loss(&pred_slots, &next_slots)?)?;
        let sigreg_loss = util::scalar_f32(&sigreg_epps_pulley(
            &flatten_latent_slots(&pred_slots)?,
            128,
            17,
        )?)?;
        let action_logits = orchestrator_head.forward(&state_slots)?;
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
    encoder_vocab: Vocab,
    encoder: OnlineEncoder,
    planner_memory: PlannerMemory,
    transition: WorldTransition,
    /// JEPA-style orchestrator head: predicts next action from transition latent. None if checkpoint has no head.
    orchestrator_head: Option<OrchestratorActionHead>,
    max_seq: usize,
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    num_layers: usize,
    #[allow(dead_code)]
    num_heads: usize,
    #[allow(dead_code)]
    bridge_dim: usize,
    #[allow(dead_code)]
    num_latent_tokens: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_planner_memory: bool,
    world_rollout_steps: usize,
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

fn rust_compile_feedback(code: &str) -> Option<String> {
    let rustc_bin = std::env::var("TOFY_CODE_REPAIR_RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    path.push(format!("tofy_repair_{stamp}.rs"));
    if fs::write(&path, format!("{code}\n")).is_err() {
        return None;
    }
    let output = Command::new(rustc_bin)
        .arg("--crate-type")
        .arg("lib")
        .arg(&path)
        .output()
        .ok();
    let _ = fs::remove_file(&path);
    let output = output?;
    if output.status.success() {
        return None;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lines = stderr
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(12)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn build_code_repair_prompt(prompt: &str, attempt: &str, feedback: Option<&str>) -> String {
    let mut out = String::from(
        "Return only corrected Rust code.\nFix the previous attempt while keeping the exact requested function name and signature.\n\nOriginal request:\n",
    );
    out.push_str(prompt);
    out.push_str("\n\nPrevious attempt:\n```rust\n");
    out.push_str(attempt);
    out.push_str("\n```\n");
    if let Some(feedback) = feedback {
        out.push_str("\nCompiler feedback:\n");
        out.push_str(feedback);
        out.push('\n');
    }
    out.push_str("\nRules:\n- Return only compilable Rust code.\n- Do not add explanation.\n");
    out
}

impl AgentEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        encoder_model_path: &PathBuf,
        encoder_vocab_path: &PathBuf,
        world_model_path: &PathBuf,
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
        let planner_memory = PlannerMemory::new(
            world_vb.pp("planner_memory"),
            dim,
            bridge_dim,
            num_latent_tokens,
        )?;
        let transition = WorldTransition::new(world_vb.pp("world_transition"), bridge_dim)?;
        let orchestrator_head =
            if checkpoint_has_prefix(world_model_path, "orchestrator_action_head.") {
                Some(OrchestratorActionHead::new(
                    world_vb.pp("orchestrator_action_head"),
                    bridge_dim,
                )?)
            } else {
                None
            };
        Ok(Self {
            device,
            _encoder_varmap: encoder_varmap,
            _world_varmap: world_varmap,
            encoder_vocab,
            encoder,
            planner_memory,
            transition,
            orchestrator_head,
            max_seq,
            dim,
            num_layers,
            num_heads,
            bridge_dim,
            num_latent_tokens,
            context_segments: std::env::var("TOFY_ENCODER_CONTEXT_SEGMENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4usize)
                .max(1),
            recent_full_segments: std::env::var("TOFY_ENCODER_RECENT_FULL_SEGMENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            recursive_planner_memory: std::env::var("TOFY_RECURSIVE_PLANNER_MEMORY")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
            world_rollout_steps: std::env::var("TOFY_WORLD_ROLLOUT_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
        })
    }

    /// Encode current prompt into private planner memory slots.
    fn encode_prompt_planner_memory(&self, current_prompt: &str) -> Result<Tensor> {
        let tokens = tokenize_for_inference(current_prompt);
        if tokens.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let ids = self.encoder_vocab.encode(&tokens);
        planner_slots_from_token_sequences(
            &self.encoder,
            &self.planner_memory,
            &[ids],
            self.encoder_vocab.pad_id,
            self.max_seq,
            self.context_segments,
            self.recent_full_segments,
            self.recursive_planner_memory,
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
        if let Some(ref h) = self.orchestrator_head {
            let logits = h.forward(state_slots)?;
            let rows = crate::util::vec2_f32(&logits)?;
            let row = rows
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty orchestrator logits"))?;
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

    /// Build decoder + conditioning from predicted next planner memory.
    fn get_decoder_and_cond_from_planner_memory(
        &self,
        next_planner_slots: &Tensor,
        action: crate::tasks::orchestrator::Action,
        ablate_conditioning: bool,
    ) -> Result<(Box<dyn LocalDecoderRuntime>, Vec<f32>)> {
        let planner_vec = util::vec1_f32(&next_planner_slots.flatten_all()?)?;
        let pooled_planner = self.planner_memory.pool(next_planner_slots)?.squeeze(0)?;
        let pooled_planner = util::vec1_f32(&pooled_planner)?;
        let code_decoder =
            CandleCrossAttnDecoder::try_new_from_env_code(self.bridge_dim, self.num_latent_tokens)
                .ok();
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
        let state_slots = self.encode_prompt_planner_memory(prompt)?;
        let action = self.route_action_from_state(prompt, &state_slots)?;
        let next_slots = rollout_transition_slots(
            &self.transition,
            &state_slots,
            action as u32,
            self.world_rollout_steps,
        )
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        if action == Action::Done {
            return Ok(String::new());
        }
        let chunk_tokens = match action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
        };
        let (decoder, cond_vec) = self.get_decoder_and_cond_from_planner_memory(
            &next_slots,
            action,
            ablate_conditioning,
        )?;
        let mut assistant_content =
            decoder.generate(prompt, action.as_str(), &cond_vec, chunk_tokens)?;
        if action == Action::Code && likely_rust_request(prompt) {
            let repair_passes = std::env::var("TOFY_CODE_REPAIR_PASSES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize);
            for _ in 0..repair_passes {
                if !output_needs_code_repair(prompt, &assistant_content) {
                    break;
                }
                let feedback = rust_compile_feedback(&assistant_content);
                let repair_prompt =
                    build_code_repair_prompt(prompt, &assistant_content, feedback.as_deref());
                let repaired =
                    decoder.generate(&repair_prompt, action.as_str(), &cond_vec, chunk_tokens)?;
                if repaired.trim().is_empty() || repaired.trim() == assistant_content.trim() {
                    break;
                }
                assistant_content = repaired;
            }
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
        let state_slots = self.encode_prompt_planner_memory(prompt)?;
        self.route_action_from_state(prompt, &state_slots)
    }

    /// Stream generated text in chunks (for SSE). The orchestrator chooses a single decoder mode for the reply.
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
        let state_slots = self.encode_prompt_planner_memory(prompt)?;
        let action = self.route_action_from_state(prompt, &state_slots)?;
        if action == Action::Done {
            on_chunk("Done. ");
            return Ok(());
        }
        match action {
            Action::TextReply => on_chunk("Writing text. "),
            Action::Code => on_chunk("Generating code. "),
            Action::Done => on_chunk("Done. "),
        }
        let next_slots = rollout_transition_slots(
            &self.transition,
            &state_slots,
            action as u32,
            self.world_rollout_steps,
        )
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let chunk_tokens = match action {
            Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
            Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
            Action::Done => 0,
        };
        let (decoder, cond_vec) = self.get_decoder_and_cond_from_planner_memory(
            &next_slots,
            action,
            ablate_conditioning,
        )?;
        decoder.generate_stream(
            prompt,
            action.as_str(),
            &cond_vec,
            chunk_tokens,
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

fn resolve_world_data_path(data_arg: &str) -> Result<PathBuf> {
    if data_arg.starts_with("hub:") {
        let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
        ensure_hub_dataset_cached(dataset_id, Path::new("data"))
    } else {
        Ok(PathBuf::from(data_arg))
    }
}

fn planner_memory_mask_from_lengths(
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
    planner_memory: &PlannerMemory,
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
    let mask = planner_memory_mask_from_lengths(features, token_lengths)?;
    planner_memory.forward_masked(&memory, Some(&mask))
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

fn planner_memory_segment_from_features(
    features: &EncoderFeatures,
    token_len: usize,
    include_tokens: bool,
) -> Result<(Tensor, Vec<f32>)> {
    let planner = features.planner_summary()?;
    let routing = features.routing_summary()?;
    let token_slots = features.token_states.dim(1)?;
    let chunk_slots = features.chunk_states.dim(1)?;
    let global_slots = features.global_states.dim(1)?;
    let chunk_size = token_slots.div_ceil(chunk_slots.max(1));
    let valid_tokens = token_len.clamp(1, token_slots);
    let valid_chunks = valid_tokens.div_ceil(chunk_size).clamp(1, chunk_slots);

    let mut mask = Vec::new();
    let memory = if include_tokens {
        mask.extend((0..token_slots).map(|idx| if idx < valid_tokens { 1.0f32 } else { 0.0f32 }));
        mask.extend((0..chunk_slots).map(|idx| if idx < valid_chunks { 1.0f32 } else { 0.0f32 }));
        mask.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
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
        mask.extend((0..chunk_slots).map(|idx| if idx < valid_chunks { 1.0f32 } else { 0.0f32 }));
        mask.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
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

#[allow(clippy::too_many_arguments)]
fn planner_slots_from_token_sequences(
    encoder: &OnlineEncoder,
    planner_memory: &PlannerMemory,
    token_sequences: &[Vec<u32>],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    recursive_planner_memory: bool,
    device: &Device,
) -> Result<Tensor> {
    let mut sample_slots = Vec::with_capacity(token_sequences.len().max(1));
    for tokens in token_sequences {
        let segments = context_segment_ranges(tokens.len(), max_seq, context_segments);
        let recent_full_segments = recent_full_segments.min(segments.len()).max(1);
        let mut folded_slots: Option<Tensor> = None;
        let mut memory_parts = Vec::with_capacity(segments.len());
        let mut mask_parts = Vec::new();

        for (segment_idx, (start, end)) in segments.iter().copied().enumerate() {
            let mut segment_ids = if start < end {
                tokens[start..end].to_vec()
            } else {
                vec![pad_id]
            };
            let token_len = segment_ids.len().min(max_seq).max(1);
            while segment_ids.len() < max_seq {
                segment_ids.push(pad_id);
            }
            let input_ids = Tensor::from_vec(segment_ids, (1, max_seq), device)?;
            let features = encoder.forward_features(&input_ids)?.detached();
            let include_tokens = segment_idx + recent_full_segments >= segments.len();
            let (segment_memory, segment_mask) =
                planner_memory_segment_from_features(&features, token_len, include_tokens)?;

            if recursive_planner_memory && segments.len() > 1 {
                let segment_mask_tensor = util::from_vec_like(
                    segment_mask,
                    (1, segment_memory.dim(1)?),
                    &segment_memory,
                )?;
                let segment_slots =
                    planner_memory.forward_masked(&segment_memory, Some(&segment_mask_tensor))?;
                folded_slots = Some(match folded_slots {
                    Some(prev_slots) => planner_memory.fold_slots(
                        &prev_slots,
                        &segment_slots,
                        recursive_memory_retain(segment_idx, segments.len(), recent_full_segments),
                    )?,
                    None => segment_slots,
                });
            } else {
                memory_parts.push(segment_memory);
                mask_parts.extend(segment_mask);
            }
        }

        let sample = if recursive_planner_memory && segments.len() > 1 {
            folded_slots.context("recursive planner fold produced no slots")?
        } else {
            let memory_refs = memory_parts.iter().collect::<Vec<_>>();
            let memory = Tensor::cat(&memory_refs, 1)?;
            let mask = util::from_vec_like(mask_parts, (1, memory.dim(1)?), &memory)?;
            planner_memory.forward_masked(&memory, Some(&mask))?
        };
        sample_slots.push(sample);
    }
    let refs = sample_slots.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

fn rollout_transition_slots(
    transition: &WorldTransition,
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

#[cfg(test)]
mod tests {
    use super::{context_segment_ranges, planner_memory_mask_from_lengths};
    use crate::model::encoders::EncoderFeatures;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn planner_memory_mask_tracks_valid_tokens_and_chunks() {
        let device = Device::Cpu;
        let features = EncoderFeatures {
            token_states: Tensor::zeros((1, 8, 4), DType::F32, &device).unwrap(),
            chunk_states: Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap(),
            global_states: Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap(),
            pooled_queries: Tensor::zeros((1, 3, 4), DType::F32, &device).unwrap(),
        };
        let mask = planner_memory_mask_from_lengths(&features, &[3]).unwrap();
        let values = crate::util::vec2_f32(&mask).unwrap();
        assert_eq!(
            values[0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn context_segment_ranges_keeps_recent_segments() {
        assert_eq!(
            context_segment_ranges(40, 16, 4),
            vec![(0, 16), (16, 32), (32, 40)]
        );
        assert_eq!(
            context_segment_ranges(80, 16, 3),
            vec![(32, 48), (48, 64), (64, 80)]
        );
        assert_eq!(context_segment_ranges(8, 16, 4), vec![(0, 8)]);
    }
}
