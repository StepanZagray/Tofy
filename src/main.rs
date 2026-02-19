mod config;
mod data;
mod model;
mod tasks;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::{Path, PathBuf};

use config::Config;
use data::{
    build_pairs_with_vocab, build_vocab_and_pairs, ensure_hub_dataset_cached,
    ensure_hub_wikipedia_cached, make_jepa_batch, prepare_ultrachat_pairs,
};
use model::{
    copy_matching_vars, ema_update_matching_vars, OnlineEncoder, Predictor, TeacherEncoder,
};

/// Learning rate: linear warmup → constant base_lr → cosine decay to min_lr.
/// Keeping LR flat for ~25% of training after warmup helps avoid early plateau (e.g. acc_ema stuck by 40k).
fn scheduled_lr(step: usize, total_steps: usize, base_lr: f64, min_lr: f64) -> f64 {
    scheduled_lr_profile(step, total_steps, base_lr, min_lr, 0.10, 0.25, 2000)
}

/// EMA decay for teacher: cosine schedule from tau_min to tau_max (modern JEPA practice).
/// Start with faster teacher updates (lower tau), end with slower (higher tau) for stability.
fn scheduled_ema_decay(step: usize, total_steps: usize, tau_min: f64, tau_max: f64) -> f64 {
    if total_steps <= 1 {
        return tau_max;
    }
    let progress = (step as f64 - 1.0) / (total_steps - 1).max(1) as f64;
    let cos = (std::f64::consts::PI * progress.clamp(0.0, 1.0)).cos();
    tau_min + 0.5 * (tau_max - tau_min) * (1.0 - cos)
}

fn scheduled_lr_profile(
    step: usize,
    total_steps: usize,
    base_lr: f64,
    min_lr: f64,
    warmup_ratio: f64,
    flat_ratio_after_warmup: f64,
    warmup_cap: usize,
) -> f64 {
    let warmup = ((total_steps as f64) * warmup_ratio).round().max(1.0) as usize;
    let warmup = warmup.min(warmup_cap).max(1);
    let post_warmup = total_steps.saturating_sub(warmup);
    let flat_steps = ((post_warmup as f64) * flat_ratio_after_warmup)
        .round()
        .max(0.0) as usize;
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

fn main() -> Result<()> {
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
    if tasks::world::try_run_eval(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_infer(&args)? {
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
            args.get(7).and_then(|v| v.parse().ok()).unwrap_or(256),
            args.get(8).and_then(|v| v.parse().ok()).unwrap_or(72),
            args.get(9).and_then(|v| v.parse().ok()).unwrap_or(4),
            args.get(10).and_then(|v| v.parse().ok()).unwrap_or(4),
        );
    }

    // JEPA-style training (predict teacher target-view contextual latents from masked context view)
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
        "    {} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio]",
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
        "    {} --train-world <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [bridge_dim] [--lr <float>] [--init-encoder <path>]",
        args[0]
    );
    eprintln!(
        "    {} --eval-world <model_path> <vocab_path> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [bridge_dim]",
        args[0]
    );
    eprintln!(
        "    {} --infer-agent <model_path> <vocab_path> <prompt> [dim] [max_seq] [num_layers] [num_heads] [bridge_dim] [max_new_tokens] [--ablate-conditioning]",
        args[0]
    );
    eprintln!(
        "    {} --serve <model_path> <vocab_path> [bind] [dim] [max_seq] [num_layers] [num_heads] [bridge_dim] [--debug]",
        args[0]
    );
    bail!(
        "specify a mode: --prepare-ultrachat / --latent / --latent-from-checkpoint / --eval-jepa / --train-world / --eval-world / --infer-agent / --serve"
    );
}

// --- JEPA-style latent training ---
fn run_latent_training(config: Config) -> Result<()> {
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
    let min_tokens = if config.is_paragraph_data {
        Some(1)
    } else {
        None
    };
    let (vocab, pairs, vocab_stats) =
        build_vocab_and_pairs(&config.data_path, config.max_vocab, min_tokens)?;
    let vocab_size = vocab.id_to_token.len();
    let seq_len = config.max_seq;

    let embed_params = vocab_size * config.dim;
    let block_params =
        config.num_layers * (4 * config.dim * config.dim + 8 * config.dim * config.dim);
    let predictor_hidden = (config.dim / 4).max(32);
    let predictor_params = 2 * config.dim // ln
        + (config.dim * predictor_hidden + predictor_hidden) // fc1
        + (predictor_hidden * config.dim + config.dim); // fc2
    let latent_params = embed_params + block_params + predictor_params;
    println!("Training (JEPA-style latent alignment)");
    if let Some(ref p) = config.init_encoder_path {
        println!("Encoder init: {:?}", p);
    }
    println!("Vocab size: {} (includes <mask>)", vocab_size);
    println!("Pairs: {}, seq_len: {}", pairs.len(), seq_len);
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
    println!("Estimated parameters: ~{}", format_params(latent_params));

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
    let predictor = Predictor::new(vb.pp("predictor"), config.dim, predictor_hidden)?;

    // EMA teacher encoder (target network): no optimizer updates, only EMA from online encoder.
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

    let model_path = PathBuf::from(format!("model_latent_{}.safetensors", format_params(latent_params)));
    let teacher_path = model_path.with_file_name(format!(
        "{}_teacher.safetensors",
        model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
    ));
    let mut best_cos_ema = -1.0f32;
    let mut cos_ema = 0.0f32;
    const EMA_TAU_MIN: f64 = 0.996;
    const EMA_TAU_MAX: f64 = 0.9999;

    println!("JEPA: predictor bottleneck dim→{}→dim, EMA tau {:.3}→{:.4}, VICReg variance + covariance", predictor_hidden, EMA_TAU_MIN, EMA_TAU_MAX);

    for step in 1..=config.steps {
        let (context_ids, target_ids, target_linear_indices) = make_jepa_batch(
            &pairs,
            config.batch_size,
            config.max_seq,
            vocab.pad_id,
            vocab.mask_id,
            config.max_spans_per_sample,
            config.max_span_len,
            config.max_masked_ratio,
            &device,
        )?;

        // Context view: masked target regions.
        let online_hidden = encoder.forward_sequence(&context_ids)?; // [B, T, D]
                                                                     // Target view: original tokens through EMA teacher.
        let teacher_hidden = target_encoder.forward_sequence(&target_ids)?; // [B, T, D]
        let (b, t, dim) = online_hidden.dims3()?;
        let online_flat = online_hidden.reshape((b * t, dim))?;
        let teacher_flat = teacher_hidden.reshape((b * t, dim))?;

        // Gather all target positions across the batch (multi-position JEPA targets).
        let online_at_targets = online_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let target_latents = teacher_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let pred_latents = predictor.forward(&online_at_targets)?; // [N, D]

        // Normalize for cosine-style alignment.
        let pred_norm = pred_latents
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit = (pred_latents.clone() / pred_norm.broadcast_as(pred_latents.shape())?)?;
        let tgt_norm = target_latents
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit = (target_latents.clone() / tgt_norm.broadcast_as(target_latents.shape())?)?;
        let diff = pred_unit.broadcast_sub(&tgt_unit)?;
        let loss_align = diff.sqr()?.mean_all()?; // MSE on unit vectors = 2 - 2*cos

        // Variance regularization (VICReg-style) to reduce representation collapse risk.
        let pred_mean = pred_unit.mean(0)?;
        let pred_centered = (pred_unit.clone() - pred_mean.broadcast_as(pred_unit.shape())?)?;
        let pred_std = pred_centered.sqr()?.mean(0)?.sqrt()?.clamp(1e-6, 1e10)?;
        let tgt_mean = tgt_unit.mean(0)?;
        let tgt_centered = (tgt_unit.clone() - tgt_mean.broadcast_as(tgt_unit.shape())?)?;
        let tgt_std = tgt_centered.sqr()?.mean(0)?.sqrt()?.clamp(1e-6, 1e10)?;
        let ones = Tensor::from_vec(vec![1.0f32; dim], (dim,), &device)?;
        let pred_gap = ones.broadcast_sub(&pred_std)?;
        let tgt_gap = ones.broadcast_sub(&tgt_std)?;
        let loss_var_pred = pred_gap.relu()?.mean_all()?;
        let loss_var_tgt = tgt_gap.relu()?.mean_all()?;
        let loss_var = loss_var_pred.broadcast_add(&loss_var_tgt)?;

        // Covariance regularization (VICReg): penalize off-diagonal of cov matrix so dimensions decorrelate.
        let n_latents = pred_centered.dim(0)?;
        let n_inv = 1.0 / (n_latents as f64).max(1.0);
        let pred_cov = pred_centered.transpose(0, 1)?.matmul(&pred_centered)?;
        let pred_cov = pred_cov.affine(n_inv, 0.0)?;
        let tgt_cov = tgt_centered.transpose(0, 1)?.matmul(&tgt_centered)?;
        let tgt_cov = tgt_cov.affine(n_inv, 0.0)?;
        let eye = Tensor::eye(dim, DType::F32, &device)?;
        let pred_cov_sq = pred_cov.sqr()?;
        let tgt_cov_sq = tgt_cov.sqr()?;
        let pred_diag_sq = pred_cov_sq.broadcast_mul(&eye)?.sum_all()?;
        let tgt_diag_sq = tgt_cov_sq.broadcast_mul(&eye)?.sum_all()?;
        let loss_cov_pred = pred_cov_sq.sum_all()?.broadcast_sub(&pred_diag_sq)?;
        let loss_cov_tgt = tgt_cov_sq.sum_all()?.broadcast_sub(&tgt_diag_sq)?;
        let loss_cov = loss_cov_pred.broadcast_add(&loss_cov_tgt)?;

        // Weighted JEPA objective: alignment + variance + covariance (full VICReg-style).
        const ALIGN_WEIGHT: f64 = 1.0;
        const VAR_WEIGHT: f64 = 0.1;
        const COV_WEIGHT: f64 = 0.1;
        let loss_align_w = (loss_align.clone() / (1.0 / ALIGN_WEIGHT))?;
        let loss_var_w = (loss_var.clone() / (1.0 / VAR_WEIGHT))?;
        let loss_cov_w = (loss_cov.clone() / (1.0 / COV_WEIGHT))?;
        let loss = loss_align_w
            .broadcast_add(&loss_var_w)?
            .broadcast_add(&loss_cov_w)?;

        let lr = scheduled_lr(step, config.steps, config.lr, 1e-5);
        let ema_tau = scheduled_ema_decay(step, config.steps, EMA_TAU_MIN, EMA_TAU_MAX);
        opt.set_learning_rate(lr);
        opt.backward_step(&loss)?;
        ema_update_matching_vars(&varmap, &mut target_varmap, ema_tau)?;

        if step % config.log_every == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            let align_val = loss_align.to_scalar::<f32>()?;
            let var_val = loss_var.to_scalar::<f32>()?;
            let cov_val = loss_cov.to_scalar::<f32>()?;
            let cos_mean = 1.0 - 0.5 * align_val; // align = 2 - 2*cos
            cos_ema = 0.95 * cos_ema + 0.05 * cos_mean;
            let n_targets = target_linear_indices.dims1()?;

            if cos_ema > best_cos_ema {
                best_cos_ema = cos_ema;
                varmap.save(&model_path)?;
                target_varmap.save(&teacher_path)?;
                println!(
                    "step {step}/{} loss {loss_val:.4} align {align_val:.4} var {var_val:.4} cov {cov_val:.4} cos_mean {cos_mean:.4} cos_ema {cos_ema:.4} targets {n_targets} [saved best] lr {lr:.2e}",
                    config.steps
                );
            } else {
                println!(
                    "step {step}/{} loss {loss_val:.4} align {align_val:.4} var {var_val:.4} cov {cov_val:.4} cos_mean {cos_mean:.4} cos_ema {cos_ema:.4} targets {n_targets} lr {lr:.2e}",
                    config.steps
                );
            }
        }
    }

    println!(
        "Best model saved to {:?} and teacher to {:?} (cos_ema {:.4})",
        model_path, teacher_path, best_cos_ema
    );

    let vocab_path = PathBuf::from("vocab_latent.txt");
    let mut vocab_text = String::new();
    for token in &vocab.id_to_token {
        vocab_text.push_str(token);
        vocab_text.push('\n');
    }
    fs::write(&vocab_path, vocab_text)?;
    println!("Vocab saved to {:?}", vocab_path);
    println!("\nTo run JEPA-native evaluation:");
    println!(
        "  cargo run --release -- --eval-jepa {} vocab_latent.txt <data_path|hub:dataset_id> 200 32 {} {} {} {}",
        model_path.display(),
        config.dim, seq_len, config.num_layers, config.num_heads
    );
    Ok(())
}

/// Format parameter count: <1M → k, <1B → M, ≥1B → B.
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

fn load_vocab_from_file(vocab_path: &PathBuf) -> Result<model::Vocab> {
    let vocab_text = fs::read_to_string(vocab_path)?;
    let mut vocab = model::Vocab::new();
    for line in vocab_text.lines() {
        if line.is_empty() {
            continue;
        }
        vocab.add_token(line);
    }
    Ok(vocab)
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

    let vocab = load_vocab_from_file(vocab_path)?;
    let data_path = resolve_data_path(data_arg)?;
    let pairs = build_pairs_with_vocab(&data_path, &vocab)?;

    let teacher_path = model_path.with_file_name(format!(
        "{}_teacher.safetensors",
        model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
    ));
    let use_teacher = teacher_path.exists();

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let vocab_size = vocab.id_to_token.len();
    let predictor_hidden = (dim / 4).max(32);
    let encoder = OnlineEncoder::new(vb.pp("encoder"), vocab_size, dim, num_layers, num_heads)?;
    let predictor = Predictor::new(vb.pp("predictor"), dim, predictor_hidden)?;

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

    let embed_params = vocab_size * dim;
    let block_params = num_layers * (4 * dim * dim + 8 * dim * dim);
    let predictor_params = 2 * dim
        + (dim * predictor_hidden + predictor_hidden)
        + (predictor_hidden * dim + dim);
    let total_params = embed_params + block_params + predictor_params;

    println!("JEPA evaluation");
    println!("model: {:?}", model_path);
    if use_teacher {
        println!("teacher: {:?} (target view uses teacher, matches training)", teacher_path);
    } else {
        println!("teacher: not found (target view uses same encoder; metrics may be low)");
    }
    println!("data: {:?}", data_path);
    println!("pairs: {}", pairs.len());
    println!(
        "model size: ~{} [embed {} + blocks {} + predictor {}]",
        format_params(total_params),
        format_params(embed_params),
        format_params(block_params),
        format_params(predictor_params),
    );
    println!(
        "eval config: steps={} batch={} dim={} max_seq={} layers={} heads={}",
        eval_steps, batch_size, dim, max_seq, num_layers, num_heads
    );

    let mut n_total: usize = 0;
    let mut sum_cos: f64 = 0.0;
    let mut sum_l2: f64 = 0.0;
    let mut sum_rank: f64 = 0.0;
    let mut sum_rr: f64 = 0.0;
    let mut top1: usize = 0;
    let mut top5: usize = 0;

    const EVAL_MAX_SPANS: usize = 3;
    const EVAL_MAX_SPAN_LEN: usize = 32;
    const EVAL_MAX_MASKED_RATIO: f64 = 0.25;
    for _ in 0..eval_steps {
        let (context_ids, target_ids, target_linear_indices) = make_jepa_batch(
            &pairs,
            batch_size,
            max_seq,
            vocab.pad_id,
            vocab.mask_id,
            EVAL_MAX_SPANS,
            EVAL_MAX_SPAN_LEN,
            EVAL_MAX_MASKED_RATIO,
            &device,
        )?;

        let online_hidden = encoder.forward_sequence(&context_ids)?; // [B, T, D]
        let target_hidden = match &teacher_encoder {
            Some(te) => te.forward_sequence(&target_ids)?, // target view: teacher (matches training)
            None => encoder.forward_sequence(&target_ids)?,
        };
        let (b, t, d) = online_hidden.dims3()?;
        let online_flat = online_hidden.reshape((b * t, d))?;
        let target_flat = target_hidden.reshape((b * t, d))?;

        let online_at_targets = online_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let target_latents = target_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let pred_latents = predictor.forward(&online_at_targets)?; // [N, D]

        let pred_norm = pred_latents
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit = (pred_latents.clone() / pred_norm.broadcast_as(pred_latents.shape())?)?;
        let tgt_norm = target_latents
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit = (target_latents.clone() / tgt_norm.broadcast_as(target_latents.shape())?)?;

        let cos_vec = (pred_unit.clone() * tgt_unit.clone())?.sum(1)?; // [N]
        let l2_vec = pred_unit
            .clone()
            .broadcast_sub(&tgt_unit)?
            .sqr()?
            .sum(1)?
            .sqrt()?; // [N]
        let cos_vals = cos_vec.to_vec1::<f32>()?;
        let l2_vals = l2_vec.to_vec1::<f32>()?;
        let n = cos_vals.len();
        n_total += n;
        sum_cos += cos_vals.iter().map(|&v| v as f64).sum::<f64>();
        sum_l2 += l2_vals.iter().map(|&v| v as f64).sum::<f64>();

        // In-batch latent retrieval:
        // For each predicted latent i, rank all target latents j by cosine(pred_i, target_j).
        // Correct match is diagonal j=i.
        let scores = pred_unit
            .clone()
            .matmul(&tgt_unit.clone().transpose(0, 1)?)?; // [N, N]
        let scores_vec = scores.to_vec2::<f32>()?;
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
    println!("  cosine_mean: {:.4}", sum_cos / denom);
    println!("  l2_mean:     {:.4}", sum_l2 / denom);
    println!("  retrieval_top1: {:.4}", top1 as f64 / denom);
    println!("  retrieval_top5: {:.4}", top5 as f64 / denom);
    println!("  retrieval_mrr:  {:.4}", sum_rr / denom);
    println!("  mean_rank:      {:.2}", sum_rank / denom);
    Ok(())
}

