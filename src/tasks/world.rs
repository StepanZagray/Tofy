use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::fs;
use tensorboard_rs::summary_writer::SummaryWriter;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::data::{
    build_vocab_from_raw_world_file, count_raw_world_rows, encode_world_examples,
    ensure_hub_dataset_cached, make_decoder_batch, make_world_batch_from_slice, tokenize_for_inference,
    RawWorldStream, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::{
    flatten_latent_slots, load_vocab_from_file, mean_cosine_similarity, prediction_loss,
    save_vocab_to_file, sigreg_epps_pulley, tensor_rms,
    CandleCrossAttnDecoder, CodeDecoder, DecoderAdapter, DecoderKind, LlamaCppDecoder,
    LocalDecoderRuntime, OnlineEncoder, OrchestratorActionHead, PlannerMemory, StubLocalDecoder,
    Vocab, WorldTransition,
};
use crate::util;
use candle_nn::ops;

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
}

impl WorldConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                    "usage: --train-world <encoder_model.safetensors> <encoder_vocab.txt> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lambda <float>] [--lr <float>]"
            );
        }
        let mut lr_override = None;
        let mut lambda_override = None;
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
            filtered.push(args[i].clone());
            i += 1;
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            data_path: PathBuf::from(&filtered[2]),
            steps: filtered.get(3).and_then(|v| v.parse().ok()).unwrap_or(60_000),
            batch_size: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(24),
            dim: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(128),
            num_layers: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(6),
            num_heads: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(64),
            lambda: lambda_override.unwrap_or(0.2),
            lr: lr_override.unwrap_or(2e-4),
            log_every: 100,
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
        args.get(9).and_then(|v| v.parse().ok()).unwrap_or(128),
        args.get(10).and_then(|v| v.parse().ok()).unwrap_or(6),
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
    let positional: Vec<&str> = args.iter().skip(2).filter(|a| *a != "--debug").map(String::as_str).collect();
    if positional.len() < 3 {
        return Ok(false);
    }
    let encoder_model_path = PathBuf::from(positional[0]);
    let encoder_vocab_path = PathBuf::from(positional[1]);
    let world_model_path = PathBuf::from(positional[2]);
    let bind = positional.get(3).copied().unwrap_or("0.0.0.0:8080");
    let dim = positional.get(4).and_then(|v| v.parse().ok()).unwrap_or(768);
    let max_seq = positional.get(5).and_then(|v| v.parse().ok()).unwrap_or(128);
    let num_layers = positional.get(6).and_then(|v| v.parse().ok()).unwrap_or(6);
    let num_heads = positional.get(7).and_then(|v| v.parse().ok()).unwrap_or(8);
    let bridge_dim = positional.get(8).and_then(|v| v.parse().ok()).unwrap_or(256);
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
    let row_count = count_raw_world_rows(&config.data_path)?;
    let mut world_stream = RawWorldStream::new(&config.data_path)?;
    let vocab_size = encoder_vocab.id_to_token.len();

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;

    let world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, DType::F32, &device);
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), config.bridge_dim)?;
    let orchestrator_head =
        OrchestratorActionHead::new(world_vb.pp("orchestrator_action_head"), config.bridge_dim)?;

    let mut opt = candle_nn::AdamW::new_lr(world_varmap.all_vars(), config.lr)?;

    let transition_params = 2
        * (config.bridge_dim * config.bridge_dim * 4
            + config.bridge_dim * config.bridge_dim * 2
            + 4 * config.bridge_dim);
    let planner_params = config.num_latent_tokens * config.dim
        + 2 * (config.dim * config.dim + config.dim * 4 * config.dim)
        + config.dim * config.bridge_dim
        + config.bridge_dim;
    let orchestrator_hidden = (config.bridge_dim * 2).max(256);
    const ORCH_N: usize = 5;
    let orchestrator_params = config.bridge_dim * orchestrator_hidden + orchestrator_hidden
        + orchestrator_hidden * ORCH_N + ORCH_N;
    let total_params = transition_params + planner_params + orchestrator_params;
    let _ = fs::create_dir_all("local_models");
    let model_path = PathBuf::from(format!(
        "local_models/model_world_{}.safetensors",
        util::format_params(total_params)
    ));

    println!("Training (latent-only world model for code + text)");
    println!("Encoder checkpoint: {:?}", config.encoder_model_path);
    println!("Encoder vocab: {:?}", config.encoder_vocab_path);
    println!(
        "Rows: streaming {} | encoder vocab {} | max_seq {} | planner_slots {} | lambda {:.3}",
        row_count,
        vocab_size,
        config.max_seq,
        config.num_latent_tokens,
        config.lambda
    );
    println!("Streaming shuffle buffer: {}", DEFAULT_STREAM_SHUFFLE_BUFFER);
    println!(
        "Estimated parameters: ~{} [planner_memory {} + transition {} + orchestrator {}]",
        util::format_params(total_params),
        util::format_params(planner_params),
        util::format_params(transition_params),
        util::format_params(orchestrator_params)
    );

    let mut best_loss = f32::MAX;

    let run_dir = util::create_run_dir("world")?;
    let mut tb = SummaryWriter::new(&run_dir);
    println!("TensorBoard run dir: {} (view with: tensorboard --logdir runs)", run_dir);

    const ACTION_LOSS_WEIGHT: f64 = 0.2;
    const SIGREG_SLICES: usize = 128;
    const SIGREG_POINTS: usize = 17;
    for step in 1..=config.steps {
        let raw_batch = world_stream.next_batch(config.batch_size)?;
        let batch = encode_world_examples(&raw_batch, &encoder_vocab);
        let (state_ids, next_ids, _state_lens, _next_lens, action_labels) = make_world_batch_from_slice(
            &batch,
            config.max_seq,
            encoder_vocab.pad_id,
            &device,
        )?;
        let state_hidden = encoder.forward_sequence(&state_ids)?.detach();
        let next_hidden = encoder.forward_sequence(&next_ids)?.detach();
        let state_slots = planner_memory.forward(&state_hidden)?;
        let next_slots = planner_memory.forward(&next_hidden)?;
        let pred_next_slots = transition.forward(&state_slots, &action_labels)?;

        let transition_loss = prediction_loss(&pred_next_slots, &next_slots)?;
        let state_sigreg =
            sigreg_epps_pulley(&flatten_latent_slots(&state_slots)?, SIGREG_SLICES, SIGREG_POINTS)?;
        let next_sigreg =
            sigreg_epps_pulley(&flatten_latent_slots(&next_slots)?, SIGREG_SLICES, SIGREG_POINTS)?;
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
        let pred_term = transition_loss.affine(1.0 - config.lambda, 0.0)?;
        let reg_term = sigreg_loss.affine(config.lambda, 0.0)?;
        let action_term = action_loss.affine(ACTION_LOSS_WEIGHT, 0.0)?;
        let loss = pred_term
            .broadcast_add(&reg_term)?
            .broadcast_add(&action_term)?;

        opt.backward_step(&loss)?;

        if step % config.log_every == 0 {
            let trans_val = transition_loss.to_scalar::<f32>()?;
            let loss_val = loss.to_scalar::<f32>()?;
            let sigreg_val = sigreg_loss.to_scalar::<f32>()?;
            let act_val = action_loss.to_scalar::<f32>()?;
            let act_acc = batch_action_accuracy(&action_logits, &action_labels)?;
            let pred_slots_flat = flatten_latent_slots(&pred_next_slots)?;
            let next_slots_flat = flatten_latent_slots(&next_slots)?;
            let trans_cos = mean_cosine_similarity(&pred_slots_flat, &next_slots_flat)?.to_scalar::<f32>()?;
            let state_slot_rms = tensor_rms(&state_slots)?.to_scalar::<f32>()?;
            let pred_slot_rms = tensor_rms(&pred_next_slots)?.to_scalar::<f32>()?;
            let code_rate = action_positive_rate(&action_labels, 1);
            let pred_code_rate = predicted_positive_rate(&action_logits, 1)?;

            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/trans", trans_val, step);
            tb.add_scalar("loss/sigreg", sigreg_val, step);
            tb.add_scalar("loss/action", act_val, step);
            tb.add_scalar("metrics/action_acc", act_acc, step);
            tb.add_scalar("metrics/trans_cosine", trans_cos, step);
            tb.add_scalar("metrics/code_rate", code_rate, step);
            tb.add_scalar("metrics/pred_code_rate", pred_code_rate, step);
            tb.add_scalar("metrics/state_slot_rms", state_slot_rms, step);
            tb.add_scalar("metrics/pred_slot_rms", pred_slot_rms, step);
            tb.flush();

            if loss_val < best_loss {
                best_loss = loss_val;
                world_varmap.save(&model_path)?;
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} action_acc {act_acc:.3} trans_cos {trans_cos:.4} code_rate {code_rate:.3} [saved best]",
                    config.steps
                );
            } else {
                println!(
                    "step {step}/{} total {loss_val:.4} trans {trans_val:.4} sigreg {sigreg_val:.4} action {act_val:.4} action_acc {act_acc:.3} trans_cos {trans_cos:.4} code_rate {code_rate:.3}",
                    config.steps
                );
            }
        }
    }

    tb.flush();
    println!("Best world model saved to {:?} (loss {:.4})", model_path, best_loss);
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
        + dim * dim + dim
        + dim * adapter_rank + adapter_rank
        + adapter_rank * dim + dim
        + dim * ff + ff
        + ff * dim + dim;
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
    init_decoder_path: Option<PathBuf>,
    decoder_kind: DecoderKind,
    decoder_vocab_path: Option<PathBuf>,
    decoder_max_vocab: usize,
    /// If set, save decoder here (e.g. text_decoder_26M.safetensors). Default: code_decoder_<size>.safetensors next to world model.
    decoder_output_path: Option<PathBuf>,
}

impl DecoderTrainConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--init-decoder <path>] [--decoder-output <path>]"
            );
        }
        let mut init_decoder_path = None;
        let mut decoder_output_path = None;
        let mut decoder_vocab_path = None;
        let mut decoder_kind = None;
        let mut lr_override = None;
        let mut decoder_max_vocab = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--init-decoder" {
                let value = args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--init-decoder requires path"))?;
                init_decoder_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-output" {
                let value = args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--decoder-output requires path"))?;
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
                let value = args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--decoder-kind requires text|code"))?;
                decoder_kind = DecoderKind::from_flag(value);
                if decoder_kind.is_none() {
                    bail!("--decoder-kind must be one of: text, code");
                }
                i += 2;
                continue;
            }
            if args[i] == "--lr" {
                let value = args.get(i + 1).ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                lr_override = Some(value.parse().map_err(|_| anyhow::anyhow!("--lr must be float"))?);
                i += 2;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(40_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(8),
            max_seq: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(128),
            dim: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(768),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(6),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            lr: lr_override.unwrap_or(3e-4),
            log_every: 100,
            init_decoder_path,
            decoder_kind: decoder_kind.unwrap_or(DecoderKind::CodeSpecialist),
            decoder_vocab_path,
            decoder_max_vocab: decoder_max_vocab.unwrap_or(8_000),
            decoder_output_path,
        })
    }
}

/// Cross-entropy for orchestrator action head: logits [B, C], labels len B (class indices). Returns scalar.
fn action_cross_entropy(logits: &Tensor, labels: &[u32], device: &Device) -> Result<Tensor> {
    let log_probs = ops::log_softmax(logits, 1)?;
    let b = logits.dim(0)?;
    let indices = Tensor::from_vec(
        labels.iter().take(b).map(|&x| x as i64).collect::<Vec<_>>(),
        (b,),
        device,
    )?
    .unsqueeze(1)?;
    let nll = log_probs.gather(&indices, 1)?.squeeze(1)?.affine(-1.0, 0.0)?;
    Ok(nll.mean_all()?)
}

fn batch_action_accuracy(logits: &Tensor, labels: &[u32]) -> Result<f32> {
    let rows = logits.to_vec2::<f32>()?;
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
        total += 1;
    }
    Ok(correct as f32 / total.max(1) as f32)
}

fn action_positive_rate(labels: &[u32], positive_label: u32) -> f32 {
    let total = labels.len().max(1) as f32;
    labels.iter().filter(|&&label| label == positive_label).count() as f32 / total
}

fn predicted_positive_rate(logits: &Tensor, positive_label: u32) -> Result<f32> {
    let rows = logits.to_vec2::<f32>()?;
    let total = rows.len().max(1) as f32;
    let positives = rows
        .iter()
        .filter(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
                == positive_label
        })
        .count() as f32;
    Ok(positives / total)
}

/// Masked cross-entropy: mean over positions where mask > 0. logits [B,T,V], target [B,T] u32, mask [B,T].
fn masked_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?;
    let target_flat = target.reshape((b * t,))?.to_dtype(candle_core::DType::U32)?;
    let log_probs = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
    let nll_per = log_probs
        .gather(&target_flat.unsqueeze(1)?, 1)?
        .squeeze(1)?
        .affine(-1.0, 0.0)?;
    let mask_flat = mask.reshape((b * t,))?;
    let sum_nll = (nll_per.broadcast_mul(&mask_flat)?).sum_all()?;
    let sum_mask = mask_flat.sum_all()?.clamp(1e-8, 1e10)?;
    Ok(sum_nll.broadcast_div(&sum_mask)?)
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
    let row_count = count_raw_world_rows(&data_path)?;
    let mut raw_stream = RawWorldStream::new(&data_path)?;
    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
    let encoder = OnlineEncoder::new(
        encoder_vb.pp("encoder"),
        encoder_vocab.id_to_token.len(),
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    encoder_varmap.load(&config.encoder_model_path)?;

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, DType::F32, &device);
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), config.bridge_dim)?;
    world_varmap.load(&config.world_model_path)?;

    let mut decoder_varmap = VarMap::new();
    if let Some(ref p) = config.init_decoder_path {
        decoder_varmap.load(p)?;
    }
    let decoder_vb = VarBuilder::from_varmap(&decoder_varmap, DType::F32, &device);
    let decoder_adapter = DecoderAdapter::new(
        decoder_vb.pp("decoder_adapter"),
        config.bridge_dim,
        config.bridge_dim,
        DecoderAdapter::output_slots_for(config.decoder_kind, config.num_latent_tokens),
    )?;
    let decoder_path = config.decoder_output_path.clone().map(|p| {
        // Put under local_models only if path is relative and not already under local_models
        if p.is_relative() && !p.starts_with("local_models") {
            PathBuf::from("local_models").join(p)
        } else {
            p
        }
    }).unwrap_or_else(|| {
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
        let (vocab, _, _) = build_vocab_from_raw_world_file(&data_path, config.decoder_max_vocab)?;
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

    let mut opt = candle_nn::AdamW::new_lr(decoder_varmap.all_vars(), config.lr)?;

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
        "Data: {} rows | decoder vocab {} | max_seq {}",
        row_count,
        vocab_size,
        config.max_seq
    );
    println!("Streaming shuffle buffer: {}", DEFAULT_STREAM_SHUFFLE_BUFFER);
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
    println!("TensorBoard run dir: {} (view with: tensorboard --logdir runs)", run_dir);

    let mut best_loss = f32::MAX;

    for step in 1..=config.steps {
        let raw_batch = raw_stream.next_batch(config.batch_size.max(1))?;
        let encoder_batch = encode_world_examples(&raw_batch, &encoder_vocab);
        let decoder_batch = encode_world_examples(&raw_batch, &decoder_vocab);
        let (state_ids, _next_ids, _state_lens, _next_lens, action_labels) = make_world_batch_from_slice(
            &encoder_batch,
            config.max_seq,
            encoder_vocab.pad_id,
            &device,
        )?;

        let state_hidden = encoder.forward_sequence(&state_ids)?.detach();
        let state_slots = planner_memory.forward(&state_hidden)?;
        let next_planner_slots = transition.forward(&state_slots, &action_labels)?;
        let world_latent = decoder_adapter.forward(&next_planner_slots.detach())?;

        let (dec_state_ids, dec_next_ids, state_lens, next_lens, _) = make_world_batch_from_slice(
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
        let loss = masked_cross_entropy(&logits, &dec_target, &loss_mask)?;

        opt.backward_step(&loss)?;

        if step % config.log_every == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            let active_tokens = loss_mask.sum_all()?.to_scalar::<f32>()?;
            let total_tokens = (config.batch_size.max(1) * config.max_seq * 2) as f32;
            let active_frac = active_tokens / total_tokens.max(1.0);
            let perplexity = loss_val.exp();
            let world_rms = tensor_rms(&world_latent)?.to_scalar::<f32>()?;

            tb.add_scalar("loss/token_ce", loss_val, step);
            tb.add_scalar("metrics/perplexity", perplexity, step);
            tb.add_scalar("metrics/active_tokens", active_tokens, step);
            tb.add_scalar("metrics/active_frac", active_frac, step);
            tb.add_scalar("metrics/world_latent_rms", world_rms, step);
            tb.flush();
            if loss_val < best_loss {
                best_loss = loss_val;
                decoder_varmap.save(&decoder_path)?;
                println!(
                    "step {}/{} token_ce {:.4} ppl {:.2} active {:.1}% [saved best]",
                    step, config.steps, loss_val, perplexity, active_frac * 100.0
                );
            } else {
                println!(
                    "step {}/{} token_ce {:.4} ppl {:.2} active {:.1}%",
                    step, config.steps, loss_val, perplexity, active_frac * 100.0
                );
            }
        }
    }

    tb.flush();
    println!("Best decoder saved to {:?} (loss {:.4})", decoder_path, best_loss);
    println!("Decoder vocab saved to {:?}", decoder_vocab_path);
    if config.decoder_kind == DecoderKind::TextGeneralist {
        println!("To use as text decoder: set JEPA_USE_TEXT_DECODER=1 and JEPA_TEXT_DECODER={:?}", decoder_path);
    } else {
        println!("To use as code decoder: set JEPA_USE_CANDLE_DECODER=1 and JEPA_CANDLE_DECODER={:?}", decoder_path);
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
    let encoder_vocab = load_vocab_from_file(encoder_vocab_path)?;
    let data_path = resolve_world_data_path(data_arg)?;
    let row_count = count_raw_world_rows(&data_path)?;
    let mut raw_stream = RawWorldStream::new(&data_path)?;

    let mut encoder_varmap = VarMap::new();
    let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
    let encoder =
        OnlineEncoder::new(encoder_vb.pp("encoder"), encoder_vocab.id_to_token.len(), dim, num_layers, num_heads)?;
    encoder_varmap.load(encoder_model_path)?;

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, DType::F32, &device);
    let planner_memory = PlannerMemory::new(
        world_vb.pp("planner_memory"),
        dim,
        bridge_dim,
        num_latent_tokens,
    )?;
    let transition = WorldTransition::new(world_vb.pp("world_transition"), bridge_dim)?;
    let orchestrator_head =
        OrchestratorActionHead::new(world_vb.pp("orchestrator_action_head"), bridge_dim)?;
    world_varmap.load(model_path)?;

    println!("World-model evaluation");
    println!("model: {:?}", model_path);
    println!("encoder: {:?}", encoder_model_path);
    println!("rows: {}", row_count);
    println!("Streaming shuffle buffer: {}", DEFAULT_STREAM_SHUFFLE_BUFFER);
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={} planner_dim={} planner_slots={}",
        eval_steps, batch_size, dim, max_seq, num_layers, num_heads, bridge_dim, num_latent_tokens
    );

    let mut n_total = 0usize;
    let mut sum_pred = 0.0f64;
    let mut sum_sigreg = 0.0f64;
    let mut sum_action_acc = 0.0f64;
    let mut batches = 0usize;
    for _ in 0..eval_steps.max(1) {
        let raw_batch = raw_stream.next_batch(batch_size.max(1))?;
        let chunk = encode_world_examples(&raw_batch, &encoder_vocab);
        let (state_ids, next_ids, _state_lens, _next_lens, action_labels) =
            make_world_batch_from_slice(&chunk, max_seq, encoder_vocab.pad_id, &device)?;
        let state_hidden = encoder.forward_sequence(&state_ids)?.detach();
        let next_hidden = encoder.forward_sequence(&next_ids)?.detach();
        let state_slots = planner_memory.forward(&state_hidden)?;
        let next_slots = planner_memory.forward(&next_hidden)?;
        let pred_slots = transition.forward(&state_slots, &action_labels)?;
        let pred_loss = prediction_loss(&pred_slots, &next_slots)?.to_scalar::<f32>()?;
        let sigreg_loss =
            sigreg_epps_pulley(&flatten_latent_slots(&pred_slots)?, 128, 17)?.to_scalar::<f32>()?;
        let action_logits = orchestrator_head.forward(&state_slots)?;
        let action_acc = batch_action_accuracy(&action_logits, &action_labels)?;
        n_total += chunk.len();
        sum_pred += pred_loss as f64;
        sum_sigreg += sigreg_loss as f64;
        sum_action_acc += action_acc as f64;
        batches += 1;
    }

    if n_total == 0 {
        bail!("world evaluation produced zero samples");
    }
    println!("\nWorld metrics over {} samples ({} batches):", n_total, batches);
    println!("  pred_mse:          {:.4}", sum_pred / batches.max(1) as f64);
    println!("  pred_sigreg:       {:.4}", sum_sigreg / batches.max(1) as f64);
    println!("  action_acc:        {:.4}", sum_action_acc / batches.max(1) as f64);
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
}

/// Returns true if the safetensors file contains any tensor whose name starts with `prefix`.
fn checkpoint_has_prefix(model_path: &PathBuf, prefix: &str) -> bool {
    use candle_core::safetensors::MmapedSafetensors;
    let Ok(mapped) = (unsafe { MmapedSafetensors::new(model_path) }) else {
        return false;
    };
    mapped.tensors().iter().any(|(n, _)| n.starts_with(prefix))
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
        let encoder_vocab = load_vocab_from_file(encoder_vocab_path)?;

        let mut encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
        let encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            encoder_vocab.id_to_token.len(),
            dim,
            num_layers,
            num_heads,
        )?;
        encoder_varmap.load(encoder_model_path)?;

        let mut world_varmap = VarMap::new();
        let world_vb = VarBuilder::from_varmap(&world_varmap, DType::F32, &device);
        let planner_memory = PlannerMemory::new(
            world_vb.pp("planner_memory"),
            dim,
            bridge_dim,
            num_latent_tokens,
        )?;
        let transition = WorldTransition::new(world_vb.pp("world_transition"), bridge_dim)?;
        world_varmap.load(world_model_path)?;
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
        })
    }

    /// Encode current prompt into private planner memory slots.
    fn encode_prompt_planner_memory(&self, current_prompt: &str) -> Result<Tensor> {
        let device = &self.device;
        let tokens = tokenize_for_inference(current_prompt);
        if tokens.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let mut ids = self.encoder_vocab.encode(&tokens);
        if ids.len() > self.max_seq {
            ids = ids[ids.len() - self.max_seq..].to_vec();
        }
        while ids.len() < self.max_seq {
            ids.push(self.encoder_vocab.pad_id);
        }
        let input_ids = Tensor::from_vec(ids, (1, self.max_seq), device)?;
        let hidden = self.encoder.forward_sequence(&input_ids)?;
        self.planner_memory.forward(&hidden).map_err(Into::into)
    }

    /// Build decoder + conditioning from predicted next planner memory.
    fn get_decoder_and_cond_from_planner_memory(
        &self,
        next_planner_slots: &Tensor,
        action: crate::tasks::orchestrator::Action,
        ablate_conditioning: bool,
    ) -> Result<(Box<dyn LocalDecoderRuntime>, Vec<f32>)> {
        let planner_vec = next_planner_slots.flatten_all()?.to_vec1::<f32>()?;
        let pooled_planner = self
            .planner_memory
            .pool(next_planner_slots)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let code_decoder =
            CandleCrossAttnDecoder::try_new_from_env_code(self.bridge_dim, self.num_latent_tokens)
                .ok();
        let text_decoder =
            CandleCrossAttnDecoder::try_new_from_env_text(self.bridge_dim, self.num_latent_tokens)
                .ok();
        let (decoder, mut cond_vec): (Box<dyn LocalDecoderRuntime>, Vec<f32>) = if action == crate::tasks::orchestrator::Action::Code {
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

    /// Multi-step reply: one action at a time. Orchestrator (JEPA head) predicts next action from latent when loaded; else fixed policy.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
    ) -> Result<String> {
        let start = Instant::now();
        use crate::tasks::orchestrator::{action_from_index, decide_next_action, Action, MAX_ACTIONS_PER_REPLY};
        let mut assistant_content = String::new();
        for step in 0..MAX_ACTIONS_PER_REPLY {
            let current_prompt = if assistant_content.is_empty() {
                prompt.to_string()
            } else {
                format!("{}\nAssistant: {}", prompt, assistant_content)
            };
            let state_slots = self.encode_prompt_planner_memory(&current_prompt)?;
            let mut action = if let Some(ref h) = self.orchestrator_head {
                action_from_index(h.predict(&state_slots)?)
            } else {
                decide_next_action(step, prompt, &assistant_content)
            };
            if action == Action::Done && assistant_content.is_empty() {
                action = Action::TextReply;
            }
            if action == Action::Done {
                break;
            }
            let next_slots = self
                .transition
                .forward_one(&state_slots, action as u32)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let segment = if action.is_decoder() {
                let chunk_tokens = match action {
                    Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
                    Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
                    _ => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
                };
                let (decoder, cond_vec) =
                    self.get_decoder_and_cond_from_planner_memory(&next_slots, action, ablate_conditioning)?;
                let out = decoder.generate(&current_prompt, action.as_str(), &cond_vec, chunk_tokens)?;
                if out.is_empty() {
                    continue;
                }
                if assistant_content.is_empty() {
                    out
                } else {
                    format!("\n{}", out)
                }
            } else {
                // Tool stubs: one action at a time; no parallel tool calls.
                match action {
                    Action::WriteFile => "\n[Wrote file (stub)]".to_string(),
                    Action::RunCli => "\n[Ran command (stub)]".to_string(),
                    _ => continue,
                }
            };
            assistant_content.push_str(&segment);
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

    /// Stream generated text in chunks (for SSE). Multi-step reply: one action at a time; orchestrator head used when loaded.
    /// Sends status chunks ("Thinking... ", "Writing text. ", "Generating code. ") so the client can show progress.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let start = Instant::now();
        use crate::tasks::orchestrator::{action_from_index, decide_next_action, Action, MAX_ACTIONS_PER_REPLY};
        let mut assistant_content = String::new();
        for step in 0..MAX_ACTIONS_PER_REPLY {
            let current_prompt = if assistant_content.is_empty() {
                prompt.to_string()
            } else {
                format!("{}\nAssistant: {}", prompt, assistant_content)
            };
            on_chunk("Thinking... ");
            let state_slots = self.encode_prompt_planner_memory(&current_prompt)?;
            let mut action = if let Some(ref h) = self.orchestrator_head {
                action_from_index(h.predict(&state_slots)?)
            } else {
                decide_next_action(step, prompt, &assistant_content)
            };
            if action == Action::Done && assistant_content.is_empty() {
                action = Action::TextReply;
            }
            if action == Action::Done {
                break;
            }
            let next_slots = self
                .transition
                .forward_one(&state_slots, action as u32)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            if action.is_decoder() {
                match action {
                    Action::TextReply => on_chunk("Writing text. "),
                    Action::Code => on_chunk("Generating code. "),
                    _ => on_chunk("Writing text. "),
                }
                let chunk_tokens = match action {
                    Action::TextReply => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
                    Action::Code => Self::CODE_CHUNK_TOKENS.min(max_new_tokens),
                    _ => Self::TEXT_CHUNK_TOKENS.min(max_new_tokens),
                };
                let (decoder, cond_vec) =
                    self.get_decoder_and_cond_from_planner_memory(&next_slots, action, ablate_conditioning)?;
                let mut segment = String::new();
                decoder.generate_stream(&current_prompt, action.as_str(), &cond_vec, chunk_tokens, &mut |chunk: &str| {
                    segment.push_str(chunk);
                    on_chunk(chunk);
                })?;
                if !segment.is_empty() {
                    if !assistant_content.is_empty() {
                        assistant_content.push('\n');
                    }
                    let segment_cleaned = crate::model::clean_candle_decoder_output(&segment);
                    assistant_content.push_str(&segment_cleaned);
                }
            } else {
                let (status, stub) = match action {
                    Action::WriteFile => ("Writing file. ", "\n[Wrote file (stub)]"),
                    Action::RunCli => ("Running command. ", "\n[Ran command (stub)]"),
                    _ => continue,
                };
                on_chunk(status);
                on_chunk(stub);
                assistant_content.push_str(stub);
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

