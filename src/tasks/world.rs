use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::data::{
    build_vocab_and_world_examples, build_world_examples_with_vocab, ensure_hub_dataset_cached,
    make_world_batch, make_world_batch_from_slice, tokenize_for_inference,
};
use crate::model::{
    copy_matching_vars, ema_update_matching_vars, DecoderBridge, LlamaCppDecoder,
    LocalDecoderRuntime, OnlineEncoder, StubLocalDecoder, TeacherEncoder, Vocab, WorldTransition,
};

#[derive(Clone)]
struct WorldConfig {
    data_path: PathBuf,
    steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    max_vocab: usize,
    bridge_dim: usize,
    lr: f64,
    log_every: usize,
    init_encoder_path: Option<PathBuf>,
}

impl WorldConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            bail!(
                "usage: --train-world <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [bridge_dim] [--lr <float>] [--init-encoder <path>]"
            );
        }
        let mut init_encoder_path = None;
        let mut lr_override = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--init-encoder" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--init-encoder requires path"))?;
                init_encoder_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
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
            filtered.push(args[i].clone());
            i += 1;
        }
        Ok(Self {
            data_path: PathBuf::from(&filtered[0]),
            steps: filtered.get(1).and_then(|v| v.parse().ok()).unwrap_or(40_000),
            batch_size: filtered.get(2).and_then(|v| v.parse().ok()).unwrap_or(32),
            dim: filtered.get(3).and_then(|v| v.parse().ok()).unwrap_or(368),
            max_seq: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(128),
            num_layers: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(4),
            num_heads: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(8),
            max_vocab: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(8_000),
            bridge_dim: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(256),
            lr: lr_override.unwrap_or(2e-4),
            log_every: 100,
            init_encoder_path,
        })
    }
}

pub fn try_run_train(args: &[String]) -> Result<bool> {
    if args.len() < 3 || (args[1] != "--train-world" && args[1] != "train-world") {
        return Ok(false);
    }
    let data_arg = &args[2];
    let data_path = resolve_world_data_path(data_arg)?;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[0] = data_path.to_string_lossy().to_string();
    let cfg = WorldConfig::from_args_after(&args_for_cfg)?;
    run_world_training(cfg)?;
    Ok(true)
}

pub fn try_run_eval(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--eval-world" && args[1] != "eval-world") {
        return Ok(false);
    }
    run_eval_world(
        &PathBuf::from(&args[2]),
        &PathBuf::from(&args[3]),
        &args[4],
        args.get(5).and_then(|v| v.parse().ok()).unwrap_or(200),
        args.get(6).and_then(|v| v.parse().ok()).unwrap_or(32),
        args.get(7).and_then(|v| v.parse().ok()).unwrap_or(368),
        args.get(8).and_then(|v| v.parse().ok()).unwrap_or(128),
        args.get(9).and_then(|v| v.parse().ok()).unwrap_or(4),
        args.get(10).and_then(|v| v.parse().ok()).unwrap_or(8),
        args.get(11).and_then(|v| v.parse().ok()).unwrap_or(256),
    )?;
    Ok(true)
}

pub fn try_run_serve(args: &[String]) -> Result<bool> {
    if args.len() < 4 || (args[1] != "--serve" && args[1] != "serve") {
        return Ok(false);
    }
    let debug = args.iter().any(|a| a == "--debug");
    let positional: Vec<&str> = args.iter().skip(2).filter(|a| *a != "--debug").map(String::as_str).collect();
    if positional.len() < 2 {
        return Ok(false);
    }
    let model_path = PathBuf::from(positional[0]);
    let vocab_path = PathBuf::from(positional[1]);
    let bind = positional.get(2).copied().unwrap_or("0.0.0.0:8080");
    let dim = positional.get(3).and_then(|v| v.parse().ok()).unwrap_or(368);
    let max_seq = positional.get(4).and_then(|v| v.parse().ok()).unwrap_or(128);
    let num_layers = positional.get(5).and_then(|v| v.parse().ok()).unwrap_or(4);
    let num_heads = positional.get(6).and_then(|v| v.parse().ok()).unwrap_or(8);
    let bridge_dim = positional.get(7).and_then(|v| v.parse().ok()).unwrap_or(256);
    if debug {
        std::env::set_var("JEPA_DEBUG", "1");
    }
    let rt = tokio::runtime::Runtime::new().context("create tokio runtime")?;
    rt.block_on(crate::tasks::serve::run(
        bind,
        model_path,
        vocab_path,
        dim,
        max_seq,
        num_layers,
        num_heads,
        bridge_dim,
        debug,
    ))?;
    Ok(true)
}

pub fn try_run_infer(args: &[String]) -> Result<bool> {
    if args.len() < 5 || (args[1] != "--infer-agent" && args[1] != "infer-agent") {
        return Ok(false);
    }
    let mut ablate_conditioning = false;
    let mut filtered = Vec::with_capacity(args.len());
    for arg in args.iter().skip(2) {
        if arg == "--ablate-conditioning" {
            ablate_conditioning = true;
            continue;
        }
        filtered.push(arg.clone());
    }
    if filtered.len() < 3 {
        bail!("usage: --infer-agent <model_path> <vocab_path> <prompt> [dim] [max_seq] [num_layers] [num_heads] [bridge_dim] [max_new_tokens] [--ablate-conditioning]");
    }
    let model_path = PathBuf::from(&filtered[0]);
    let vocab_path = PathBuf::from(&filtered[1]);
    let prompt = filtered[2].as_str();
    let dim = filtered.get(3).and_then(|v| v.parse().ok()).unwrap_or(368);
    let max_seq = filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(128);
    let num_layers = filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(4);
    let num_heads = filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(8);
    let bridge_dim = filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(256);
    let max_new_tokens = filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let engine = AgentEngine::load(
        &model_path,
        &vocab_path,
        dim,
        max_seq,
        num_layers,
        num_heads,
        bridge_dim,
    )?;
    let text = engine.generate(prompt, max_new_tokens, ablate_conditioning)?;
    println!("Agent inference");
    println!("model: {:?}", model_path);
    println!("action: reply");
    println!("conditioning_ablation: {}", ablate_conditioning);
    println!("output: {}", text);
    Ok(true)
}

fn pool_mean(hidden: &Tensor, lengths: &[usize]) -> Result<Tensor> {
    let (b, t, _) = hidden.dims3()?;
    let mut pooled_rows = Vec::with_capacity(b);
    for i in 0..b {
        let sample = hidden.narrow(0, i, 1)?.squeeze(0)?;
        let valid = lengths.get(i).copied().unwrap_or(1).clamp(1, t);
        let mean = sample.narrow(0, 0, valid)?.mean(0)?;
        pooled_rows.push(mean.unsqueeze(0)?);
    }
    Ok(Tensor::cat(&pooled_rows, 0)?)
}

fn scheduled_lr(step: usize, total_steps: usize, base_lr: f64, min_lr: f64) -> f64 {
    let warmup = ((total_steps as f64) * 0.08).round().max(1.0) as usize;
    let warmup = warmup.min(2000).max(1);
    let flat_steps = ((total_steps.saturating_sub(warmup) as f64) * 0.30).round() as usize;
    let decay_start = warmup + flat_steps;
    if step <= warmup {
        base_lr * (step as f64 / warmup as f64)
    } else if step <= decay_start {
        base_lr
    } else {
        let progress = (step - decay_start) as f64 / (total_steps - decay_start).max(1) as f64;
        let cos = (std::f64::consts::PI * progress).cos();
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos)
    }
}

fn scheduled_ema_decay(step: usize, total_steps: usize, tau_min: f64, tau_max: f64) -> f64 {
    if total_steps <= 1 {
        return tau_max;
    }
    let progress = (step as f64 - 1.0) / (total_steps - 1).max(1) as f64;
    let cos = (std::f64::consts::PI * progress.clamp(0.0, 1.0)).cos();
    tau_min + 0.5 * (tau_max - tau_min) * (1.0 - cos)
}

fn run_world_training(config: WorldConfig) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            eprintln!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            eprintln!("CUDA not available: {e}");
            Device::Cpu
        }
    };

    let (vocab, mut rows, vocab_stats) =
        build_vocab_and_world_examples(&config.data_path, config.max_vocab)?;
    let split = ((rows.len() as f64) * 0.9) as usize;
    let split = split.clamp(1, rows.len().saturating_sub(1));
    let val_rows = rows.split_off(split);
    let train_rows = rows;
    let vocab_size = vocab.id_to_token.len();

    let mut varmap = VarMap::new();
    if let Some(ref init_path) = config.init_encoder_path {
        varmap.load(init_path)?;
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = OnlineEncoder::new(
        vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    let transition = WorldTransition::new(vb.pp("world_transition"), config.dim)?;
    let bridge = DecoderBridge::new(vb.pp("decoder_bridge"), config.dim, config.bridge_dim)?;

    let mut target_varmap = VarMap::new();
    let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, &device);
    let target_encoder = TeacherEncoder::new(
        target_vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    copy_matching_vars(&varmap, &mut target_varmap)?;

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), config.lr)?;

    let embed_params = vocab_size * config.dim;
    let block_params =
        config.num_layers * (4 * config.dim * config.dim + 8 * config.dim * config.dim);
    let transition_params = (config.dim / 4).max(32)
        + ((config.dim + (config.dim / 4).max(32)) * (config.dim * 3 / 2).max(128)
            + (config.dim * 3 / 2).max(128))
        + ((config.dim * 3 / 2).max(128) * config.dim + config.dim);
    let bridge_params = config.dim * config.bridge_dim + config.bridge_dim;
    let total_params = embed_params + block_params + transition_params + bridge_params;
    let model_path = PathBuf::from(format!("model_world_{}.safetensors", format_params(total_params)));
    let teacher_path = model_path.with_file_name(format!(
        "{}_teacher.safetensors",
        model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model_world")
    ));

    println!("Training (World model: fixed action=reply)");
    if let Some(ref p) = config.init_encoder_path {
        println!("Encoder init: {:?}", p);
    }
    println!(
        "Rows: train {} / val {} | vocab {} | max_seq {}",
        train_rows.len(),
        val_rows.len(),
        vocab_size,
        config.max_seq
    );
    if vocab_stats.total_tokens > 0 {
        let coverage = (vocab_stats.covered_tokens as f64 / vocab_stats.total_tokens as f64) * 100.0;
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
        "Estimated parameters: ~{} [encoder {} + transition {} + bridge {}]",
        format_params(total_params),
        format_params(embed_params + block_params),
        format_params(transition_params),
        format_params(bridge_params)
    );

    let mut best_score = -1.0f32;
    let mut score_ema = 0.0f32;
    const EMA_TAU_MIN: f64 = 0.996;
    const EMA_TAU_MAX: f64 = 0.9999;

    for step in 1..=config.steps {
        let (state_ids, next_ids, state_lens, next_lens) = make_world_batch(
            &train_rows,
            config.batch_size,
            config.max_seq,
            vocab.pad_id,
            &device,
        )?;
        let state_hidden = encoder.forward_sequence(&state_ids)?;
        let state_latent = pool_mean(&state_hidden, &state_lens)?;

        let pred_next_latent = transition.forward_reply(&state_latent)?;
        let next_hidden_teacher = target_encoder.forward_sequence(&next_ids)?;
        let target_next_latent = pool_mean(&next_hidden_teacher, &next_lens)?;

        let pred_norm = pred_next_latent
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit =
            (pred_next_latent.clone() / pred_norm.broadcast_as(pred_next_latent.shape())?)?;
        let tgt_norm = target_next_latent
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit =
            (target_next_latent.clone() / tgt_norm.broadcast_as(target_next_latent.shape())?)?;
        let diff = pred_unit.broadcast_sub(&tgt_unit)?;
        let transition_loss = diff.sqr()?.mean_all()?;
        let loss = transition_loss.clone();

        let _conditioning = bridge.forward(&state_latent)?;

        let lr = scheduled_lr(step, config.steps, config.lr, 1e-5);
        let ema_tau = scheduled_ema_decay(step, config.steps, EMA_TAU_MIN, EMA_TAU_MAX);
        opt.set_learning_rate(lr);
        opt.backward_step(&loss)?;
        ema_update_matching_vars(&varmap, &mut target_varmap, ema_tau)?;

        if step % config.log_every == 0 {
            let trans_val = transition_loss.to_scalar::<f32>()?;
            let loss_val = loss.to_scalar::<f32>()?;
            let cos_mean = 1.0 - 0.5 * trans_val;
            score_ema = 0.95 * score_ema + 0.05 * cos_mean;

            if score_ema > best_score {
                best_score = score_ema;
                varmap.save(&model_path)?;
                target_varmap.save(&teacher_path)?;
                println!(
                    "step {step}/{} loss {loss_val:.4} trans {trans_val:.4} trans_cos {cos_mean:.4} score_ema {score_ema:.4} action reply [saved best] lr {lr:.2e}",
                    config.steps
                );
            } else {
                println!(
                    "step {step}/{} loss {loss_val:.4} trans {trans_val:.4} trans_cos {cos_mean:.4} score_ema {score_ema:.4} action reply lr {lr:.2e}",
                    config.steps
                );
            }
        }
    }

    let vocab_path = PathBuf::from("vocab_world.txt");
    let mut vocab_text = String::new();
    for token in &vocab.id_to_token {
        vocab_text.push_str(token);
        vocab_text.push('\n');
    }
    fs::write(&vocab_path, vocab_text)?;
    println!(
        "Best world model saved to {:?} and teacher to {:?} (score_ema {:.4}, action=reply), vocab {:?}",
        model_path, teacher_path, best_score, vocab_path
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval_world(
    model_path: &PathBuf,
    vocab_path: &PathBuf,
    data_arg: &str,
    eval_steps: usize,
    batch_size: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
) -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let vocab = load_vocab_from_file(vocab_path)?;
    let data_path = resolve_world_data_path(data_arg)?;
    let rows = build_world_examples_with_vocab(&data_path, &vocab)?;
    let teacher_path = model_path.with_file_name(format!(
        "{}_teacher.safetensors",
        model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model_world")
    ));
    let use_teacher = teacher_path.exists();

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vocab_size = vocab.id_to_token.len();
    let encoder = OnlineEncoder::new(vb.pp("encoder"), vocab_size, dim, num_layers, num_heads)?;
    let transition = WorldTransition::new(vb.pp("world_transition"), dim)?;
    let _bridge = DecoderBridge::new(vb.pp("decoder_bridge"), dim, bridge_dim)?;
    let teacher_encoder = if use_teacher {
        let mut teacher_varmap = VarMap::new();
        teacher_varmap.load(&teacher_path)?;
        let teacher_vb = VarBuilder::from_varmap(&teacher_varmap, DType::F32, &device);
        Some(TeacherEncoder::new(
            teacher_vb.pp("encoder"),
            vocab_size,
            dim,
            num_layers,
            num_heads,
        )?)
    } else {
        None
    };

    println!("World-model evaluation");
    println!("model: {:?}", model_path);
    if use_teacher {
        println!("teacher: {:?} (target next-latent uses teacher)", teacher_path);
    } else {
        println!("teacher: not found (target next-latent uses same encoder)");
    }
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={} bridge_dim={}",
        eval_steps, batch_size, dim, max_seq, num_layers, num_heads, bridge_dim
    );

    let mut n_total = 0usize;
    let mut sum_cos = 0.0f64;
    let mut sum_l2 = 0.0f64;
    let mut batches = 0usize;
    for chunk in rows.chunks(batch_size.max(1)).take(eval_steps.max(1)) {
        let (state_ids, next_ids, state_lens, next_lens) =
            make_world_batch_from_slice(chunk, max_seq, vocab.pad_id, &device)?;
        let state_hidden = encoder.forward_sequence(&state_ids)?;
        let state_latent = pool_mean(&state_hidden, &state_lens)?;
        let pred = transition.forward_reply(&state_latent)?;
        let target_hidden = match &teacher_encoder {
            Some(te) => te.forward_sequence(&next_ids)?,
            None => encoder.forward_sequence(&next_ids)?,
        };
        let target = pool_mean(&target_hidden, &next_lens)?;

        let pred_norm = pred
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit = (pred.clone() / pred_norm.broadcast_as(pred.shape())?)?;
        let tgt_norm = target
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit = (target.clone() / tgt_norm.broadcast_as(target.shape())?)?;
        let cos_vec = (pred_unit.clone() * tgt_unit.clone())?.sum(1)?;
        let l2_vec = pred_unit
            .broadcast_sub(&tgt_unit)?
            .sqr()?
            .sum(1)?
            .sqrt()?;
        let cos_vals = cos_vec.to_vec1::<f32>()?;
        let l2_vals = l2_vec.to_vec1::<f32>()?;
        n_total += cos_vals.len();
        sum_cos += cos_vals.iter().map(|&v| v as f64).sum::<f64>();
        sum_l2 += l2_vals.iter().map(|&v| v as f64).sum::<f64>();
        batches += 1;
    }

    if n_total == 0 {
        bail!("world evaluation produced zero samples");
    }
    let denom = n_total as f64;
    println!("\nWorld metrics over {} samples ({} batches):", n_total, batches);
    println!("  action:            reply (fixed)");
    println!("  transition_cos:    {:.4}", sum_cos / denom);
    println!("  transition_l2:     {:.4}", sum_l2 / denom);
    Ok(())
}


fn load_vocab_from_file(vocab_path: &PathBuf) -> Result<Vocab> {
    let vocab_text = fs::read_to_string(vocab_path)?;
    let mut vocab = Vocab::new();
    for line in vocab_text.lines() {
        if line.is_empty() {
            continue;
        }
        vocab.add_token(line);
    }
    Ok(vocab)
}

/// Loaded world model + vocab for reuse (CLI or server). Single-thread use or behind a Mutex.
#[allow(dead_code)]
pub struct AgentEngine {
    device: Device,
    _varmap: VarMap,
    vocab: Vocab,
    encoder: OnlineEncoder,
    transition: WorldTransition,
    bridge: DecoderBridge,
    max_seq: usize,
    dim: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
}

impl AgentEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_path: &PathBuf,
        vocab_path: &PathBuf,
        dim: usize,
        max_seq: usize,
        num_layers: usize,
        num_heads: usize,
        bridge_dim: usize,
    ) -> Result<Self> {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        };
        let vocab = load_vocab_from_file(vocab_path)?;
        let mut varmap = VarMap::new();
        varmap.load(model_path)?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vocab_size = vocab.id_to_token.len();
        let encoder =
            OnlineEncoder::new(vb.pp("encoder"), vocab_size, dim, num_layers, num_heads)?;
        let transition = WorldTransition::new(vb.pp("world_transition"), dim)?;
        let bridge = DecoderBridge::new(vb.pp("decoder_bridge"), dim, bridge_dim)?;
        Ok(Self {
            device,
            _varmap: varmap,
            vocab,
            encoder,
            transition,
            bridge,
            max_seq,
            dim,
            num_layers,
            num_heads,
            bridge_dim,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
    ) -> Result<String> {
        let start = Instant::now();
        let device = &self.device;
        let tokens = tokenize_for_inference(prompt);
        if tokens.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let mut ids = self.vocab.encode(&tokens);
        if ids.len() > self.max_seq {
            ids.truncate(self.max_seq);
        }
        let length = ids.len().max(1);
        while ids.len() < self.max_seq {
            ids.push(self.vocab.pad_id);
        }

        let input_ids = Tensor::from_vec(ids, (1, self.max_seq), device)?;
        let hidden = self.encoder.forward_sequence(&input_ids)?;
        let latent = pool_mean(&hidden, &[length])?;
        let next_latent = self.transition.forward_reply(&latent)?;
        let conditioning = self.bridge.forward(&next_latent)?;
        let mut cond_vec = conditioning.squeeze(0)?.to_vec1::<f32>()?;
        if ablate_conditioning {
            cond_vec.fill(0.0);
        }

        let decoder: Box<dyn LocalDecoderRuntime> = match LlamaCppDecoder::try_new() {
            Ok(d) => Box::new(d),
            Err(_) => Box::new(StubLocalDecoder::new()),
        };
        let out = decoder.generate(prompt, "reply", &cond_vec, max_new_tokens);
        if out.is_ok() && std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] response in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
        out
    }

    /// Stream generated text in chunks (for SSE). Same as generate() but yields chunks via on_chunk.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_conditioning: bool,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let start = Instant::now();
        let device = &self.device;
        let tokens = tokenize_for_inference(prompt);
        if tokens.is_empty() {
            bail!("prompt tokenized to empty sequence");
        }
        let mut ids = self.vocab.encode(&tokens);
        if ids.len() > self.max_seq {
            ids.truncate(self.max_seq);
        }
        let length = ids.len().max(1);
        while ids.len() < self.max_seq {
            ids.push(self.vocab.pad_id);
        }

        let input_ids = Tensor::from_vec(ids, (1, self.max_seq), device)?;
        let hidden = self.encoder.forward_sequence(&input_ids)?;
        let latent = pool_mean(&hidden, &[length])?;
        let next_latent = self.transition.forward_reply(&latent)?;
        let conditioning = self.bridge.forward(&next_latent)?;
        let mut cond_vec = conditioning.squeeze(0)?.to_vec1::<f32>()?;
        if ablate_conditioning {
            cond_vec.fill(0.0);
        }

        let decoder: Box<dyn LocalDecoderRuntime> = match LlamaCppDecoder::try_new() {
            Ok(d) => Box::new(d),
            Err(_) => Box::new(StubLocalDecoder::new()),
        };
        let out = decoder.generate_stream(prompt, "reply", &cond_vec, max_new_tokens, on_chunk);
        if out.is_ok() && std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] response in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
        out
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

fn format_params(n: usize) -> String {
    const K: usize = 1_000;
    const M: usize = 1_000_000;
    const B: usize = 1_000_000_000;
    if n < M {
        format!("{:.1}k", n as f64 / K as f64)
    } else if n < B {
        format!("{:.2}M", n as f64 / M as f64)
    } else {
        format!("{:.2}B", n as f64 / B as f64)
    }
}
