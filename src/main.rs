mod config;
mod data;
mod format;
mod model;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use format::{format_user_input, FormatInputError};
use rand::Rng;
use std::fs;
use std::path::{Path, PathBuf};

use config::Config;
use data::{build_vocab_and_pairs, ensure_hub_dataset_cached, make_latent_batch};
use model::{
    copy_matching_vars, ema_update_matching_vars, OnlineEncoder, Predictor, TeacherEncoder,
};

/// Learning rate: linear warmup → constant base_lr → cosine decay to min_lr.
/// Keeping LR flat for ~25% of training after warmup helps avoid early plateau (e.g. acc_ema stuck by 40k).
fn scheduled_lr(step: usize, total_steps: usize, base_lr: f64, min_lr: f64) -> f64 {
    let warmup = (total_steps / 10).min(2000).max(1);
    let post_warmup = total_steps.saturating_sub(warmup);
    let flat_steps = post_warmup / 4; // constant LR for first 25% of post-warmup steps
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

    // Latent-space diff: show L2 and cosine between predicted embedding and user-provided answer
    if args.len() >= 2 && (args[1] == "--diff" || args[1] == "diff") {
        if args.len() < 4 {
            bail!("usage: {} --diff <model_path> <vocab_path> [dim] [max_seq] [num_layers] [num_heads]", args[0]);
        }
        return run_diff(
            &PathBuf::from(&args[2]),
            &PathBuf::from(&args[3]),
            args.get(4).and_then(|v| v.parse().ok()).unwrap_or(256),
            args.get(5).and_then(|v| v.parse().ok()).unwrap_or(96),
            args.get(6).and_then(|v| v.parse().ok()).unwrap_or(4),
            args.get(7).and_then(|v| v.parse().ok()).unwrap_or(4),
        );
    }

    // Latent-only training (no decoder: predict in embedding space, JEPA-style)
    if args.len() >= 2 && (args[1] == "--latent" || args[1] == "latent") {
        let data_arg = if args.len() > 2 { &args[2] } else { "" };
        let args_for_config: Vec<String> = if data_arg.starts_with("hub:") {
            let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
            let cache_path = ensure_hub_dataset_cached(dataset_id, Path::new("data"))?;
            let mut a = args[2..].to_vec();
            a[0] = cache_path.to_string_lossy().to_string();
            a
        } else {
            args[2..].to_vec()
        };
        let config = Config::from_args_after(&args_for_config)?;
        return run_latent_training(config);
    }

    // Latent training with encoder initialized from checkpoint (e.g. previous latent run)
    if args.len() >= 4
        && (args[1] == "--latent-from-checkpoint" || args[1] == "latent-from-checkpoint")
    {
        let init_path = PathBuf::from(&args[2]);
        let mut config = Config::from_args_after(&args[3..])?;
        config.init_encoder_path = Some(init_path);
        return run_latent_training(config);
    }

    // No mode: print usage (Training vs Inference explicit)
    eprintln!("usage (choose one):");
    eprintln!("  Training (learn from data):");
    eprintln!(
        "    {} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab]",
        args[0]
    );
    eprintln!(
        "    {} --latent-from-checkpoint <encoder_checkpoint.safetensors> <data_path> [steps] ...",
        args[0]
    );
    eprintln!("  Inference (run trained model):");
    eprintln!(
        "    {} --diff  <model_path> <vocab_path> [dim] [max_seq] [num_layers] [num_heads]",
        args[0]
    );
    bail!("specify a mode: --latent or --latent-from-checkpoint (training) or --diff (inference)");
}

// --- Latent-only training (no decoder: predict in embedding space, JEPA-style) ---
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
    let (vocab, pairs, vocab_stats) = build_vocab_and_pairs(&config.data_path, config.max_vocab)?;
    let vocab_size = vocab.id_to_token.len();
    let seq_len = config.max_seq;

    let embed_params = vocab_size * config.dim;
    let block_params =
        config.num_layers * (4 * config.dim * config.dim + 8 * config.dim * config.dim);
    let predictor_params = 2 * (config.dim * config.dim + config.dim); // 2-layer MLP dim->dim->dim
    let latent_params = embed_params + block_params + predictor_params;
    println!("Training (Latent, embedding-space)");
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
    let predictor = Predictor::new(vb.pp("predictor"), config.dim)?;

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

    let model_path = PathBuf::from("model_latent.safetensors");
    let mut best_acc_ema = -1.0f32;
    let mut acc_ema = 0.0f32;
    const EMA_DECAY: f64 = 0.996;
    const FULL_VOCAB_EVAL_EVERY: usize = 1000;
    let all_vocab_ids = Tensor::from_vec(
        (0..vocab_size as u32).collect::<Vec<_>>(),
        (vocab_size,),
        &device,
    )?;

    for step in 1..=config.steps {
        let (input_ids, mask_positions, target_ids) = make_latent_batch(
            &pairs,
            config.batch_size,
            config.max_seq,
            vocab.pad_id,
            vocab.mask_id,
            vocab_size,
            &device,
        )?;

        let hidden = encoder.forward_sequence(&input_ids)?; // [B, T, D]
        let (b, t, dim) = hidden.dims3()?;
        let hidden_flat = hidden.reshape((b * t, dim))?;
        let pos_vec: Vec<u32> = mask_positions.to_vec1()?;
        let indices: Vec<u32> = (0..b)
            .map(|i| (i as u32) * (t as u32) + pos_vec[i])
            .collect();
        let indices_t = Tensor::from_vec(indices, (b,), &device)?;
        let hidden_at_mask = hidden_flat.index_select(&indices_t, 0)?; // [B, dim]
        let pred_embed = predictor.forward(&hidden_at_mask)?; // [B, dim]

        // Stop-grad targets from EMA teacher encoder.
        let target_embed = target_encoder.embed_tokens(&target_ids)?; // [B, dim]

        // L2-normalize so we optimize direction (cosine), not magnitude. Improves alignment at inference.
        let pred_norm = pred_embed
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let pred_unit = (pred_embed.clone() / pred_norm.broadcast_as(pred_embed.shape())?)?;
        let tgt_norm = target_embed
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let tgt_unit = (target_embed.clone() / tgt_norm.broadcast_as(target_embed.shape())?)?;
        let diff = (pred_unit.clone() - tgt_unit.clone())?;
        let loss_mse = diff.sqr()?.mean_all()?; // MSE on unit vectors = 2 - 2*cos, so we maximize cosine

        // Auxiliary: encourage encoder hidden at mask to point toward target embedding (same space as predictor target).
        // This helps the encoder put the "answer" direction in the context representation so the predictor can refine it.
        let hidden_norm = hidden_at_mask
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let hidden_unit =
            (hidden_at_mask.clone() / hidden_norm.broadcast_as(hidden_at_mask.shape())?)?;
        let cos_hidden_tgt = (hidden_unit * tgt_unit.clone())?.sum(1)?; // [B]
        let loss_aux = (Tensor::from_vec(vec![1.0f32; b], (b,), hidden_at_mask.device())?
            - cos_hidden_tgt)?
            .mean_all()?;

        // Sampled softmax loss: rank correct token above random negatives.
        const NUM_NEG: usize = 32;
        let target_vec: Vec<u32> = target_ids.to_vec1()?;
        let mut rng = rand::thread_rng();
        let mut neg_ids = Vec::with_capacity(b * NUM_NEG);
        for &t in &target_vec {
            for _ in 0..NUM_NEG {
                let mut id = rng.gen_range(0..vocab_size as u32);
                while id == t {
                    id = rng.gen_range(0..vocab_size as u32);
                }
                neg_ids.push(id);
            }
        }
        let neg_ids_t = Tensor::from_vec(neg_ids, (b * NUM_NEG,), &device)?;
        let neg_embed = encoder.embed_tokens(&neg_ids_t)?; // [B*N, dim]
        let neg_norm = neg_embed
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let neg_unit = (neg_embed.clone() / neg_norm.broadcast_as(neg_embed.shape())?)?
            .reshape((b, NUM_NEG, config.dim))?; // [B, N, dim]
        let pred_broadcast = pred_unit.unsqueeze(1)?.broadcast_as(neg_unit.shape())?; // [B, N, dim]
        let cos_neg = (pred_broadcast * neg_unit)?.sum(2)?; // [B, N]
        let cos_target = (pred_unit * tgt_unit)?.sum(1)?.unsqueeze(1)?; // [B, 1]
                                                                        // Lower RANK_TEMP = sharper softmax = stronger gradients to separate correct from negatives (helps full-vocab rank).
        const RANK_TEMP: f64 = 0.05;
        let to_cat = [cos_target.clone(), cos_neg.clone()];
        let logits = Tensor::cat(&to_cat[..], 1)?;
        let logits = (logits / RANK_TEMP)?;
        let target_idx = Tensor::from_vec(vec![0u32; b], (b,), &device)?;
        let loss_rank = candle_nn::loss::cross_entropy(&logits, &target_idx)?;
        // Loss weights: higher MSE_WEIGHT pushes pred exactly toward target (better full-vocab rank at inference).
        const MSE_WEIGHT: f64 = 20.0;
        const RANK_WEIGHT: f64 = 1.0;
        const AUX_WEIGHT: f64 = 0.1;
        let loss_mse_w = (loss_mse.clone() / (1.0 / MSE_WEIGHT))?;
        let loss_rank_w = (loss_rank / (1.0 / RANK_WEIGHT))?;
        let loss = loss_mse_w.broadcast_add(&loss_rank_w)?;
        let loss_aux_weighted = (loss_aux.clone() / (1.0 / AUX_WEIGHT))?;
        let loss = loss.broadcast_add(&loss_aux_weighted)?;

        let lr = scheduled_lr(step, config.steps, config.lr, 1e-5);
        opt.set_learning_rate(lr);
        opt.backward_step(&loss)?;
        ema_update_matching_vars(&varmap, &mut target_varmap, EMA_DECAY)?;

        if step % config.log_every == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            let cos_mean = 1.0 - 0.5 * loss_mse.to_scalar::<f32>()?; // MSE = 2-2*cos => cos = 1 - MSE/2
            let aux_val = loss_aux.to_scalar::<f32>()?;

            let cos_neg_vec = cos_neg.to_vec2::<f32>()?;
            let cos_target_vec = cos_target.squeeze(1)?.to_vec1::<f32>()?;
            let mut correct = 0usize;
            for (row, &ct) in cos_neg_vec.iter().zip(cos_target_vec.iter()) {
                let max_neg = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if ct > max_neg {
                    correct += 1;
                }
            }
            let acc_neg = if b > 0 {
                correct as f32 / b as f32
            } else {
                0.0
            };
            acc_ema = 0.92 * acc_ema + 0.08 * acc_neg;
            let mut full_eval_suffix = String::new();
            if step % FULL_VOCAB_EVAL_EVERY == 0 {
                let all_embeds = encoder.embed_tokens(&all_vocab_ids)?; // [V, D]
                let all_norm = all_embeds
                    .sqr()?
                    .sum(1)?
                    .unsqueeze(1)?
                    .sqrt()?
                    .clamp(1e-8, 1e10)?;
                let all_unit = (all_embeds.clone() / all_norm.broadcast_as(all_embeds.shape())?)?;
                let scores = pred_unit.matmul(&all_unit.transpose(0, 1)?)?; // [B, V]
                let scores_vec = scores.to_vec2::<f32>()?;
                let target_vec = target_ids.to_vec1::<u32>()?;
                let mut top1 = 0usize;
                let mut top10 = 0usize;
                let mut rank_sum = 0usize;
                for (row, &target_id) in scores_vec.iter().zip(target_vec.iter()) {
                    let target_idx = target_id as usize;
                    if target_idx >= row.len() {
                        continue;
                    }
                    let target_score = row[target_idx];
                    let mut gt_count = 0usize;
                    let mut best_idx = 0usize;
                    let mut best_score = f32::NEG_INFINITY;
                    for (j, &s) in row.iter().enumerate() {
                        if s > best_score {
                            best_score = s;
                            best_idx = j;
                        }
                        if s > target_score {
                            gt_count += 1;
                        }
                    }
                    let rank_1based = gt_count + 1;
                    rank_sum += rank_1based;
                    if best_idx == target_idx {
                        top1 += 1;
                    }
                    if rank_1based <= 10 {
                        top10 += 1;
                    }
                }
                let denom = if b > 0 { b } else { 1 };
                let full_top1 = top1 as f32 / denom as f32;
                let full_top10 = top10 as f32 / denom as f32;
                let mean_rank = rank_sum as f32 / denom as f32;
                full_eval_suffix = format!(
                    " full_top1 {full_top1:.3} full_top10 {full_top10:.3} mean_rank {mean_rank:.1}"
                );
            }

            if acc_ema > best_acc_ema {
                best_acc_ema = acc_ema;
                varmap.save(&model_path)?;
                println!(
                    "step {step}/{} loss {loss_val:.4} cos_mean {cos_mean:.4} aux {aux_val:.4} acc_neg {acc_neg:.3} acc_ema {acc_ema:.3} [saved best] lr {lr:.2e}{}",
                    config.steps,
                    full_eval_suffix
                );
            } else {
                println!(
                    "step {step}/{} loss {loss_val:.4} cos_mean {cos_mean:.4} aux {aux_val:.4} acc_neg {acc_neg:.3} acc_ema {acc_ema:.3} lr {lr:.2e}{}",
                    config.steps,
                    full_eval_suffix
                );
            }
        }
    }

    println!(
        "Best model saved to {:?} (acc_ema {:.3})",
        model_path, best_acc_ema
    );

    let vocab_path = PathBuf::from("vocab_latent.txt");
    let mut vocab_text = String::new();
    for token in &vocab.id_to_token {
        vocab_text.push_str(token);
        vocab_text.push('\n');
    }
    fs::write(&vocab_path, vocab_text)?;
    println!("Vocab saved to {:?}", vocab_path);
    println!("\nTo compare prediction vs answer in latent space:");
    println!(
        "  cargo run --release -- --diff model_latent.safetensors vocab_latent.txt {} {} {} {}",
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

// --- Latent diff: show L2 and cosine between predicted embedding and user-provided answer ---
#[allow(non_snake_case)] // Option::None in match arms triggers false-positive
fn run_diff(
    model_path: &PathBuf,
    vocab_path: &PathBuf,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
) -> Result<()> {
    use std::io::{self, BufRead, Write};

    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };

    let vocab_text = fs::read_to_string(vocab_path)?;
    let mut vocab = model::Vocab::new();
    for line in vocab_text.lines() {
        if line.is_empty() {
            continue;
        }
        vocab.add_token(line);
    }

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let vocab_size = vocab.id_to_token.len();
    let encoder = OnlineEncoder::new(vb.pp("encoder"), vocab_size, dim, num_layers, num_heads)?;
    let predictor = Predictor::new(vb.pp("predictor"), dim)?;

    let embed_params = vocab_size * dim;
    let block_params = num_layers * (4 * dim * dim + 8 * dim * dim);
    let predictor_params = 2 * (dim * dim + dim); // 2-layer MLP
    let total_params = embed_params + block_params + predictor_params;
    println!(
        "Model: ~{} [embed {} + blocks {} + predictor {}]",
        format_params(total_params),
        format_params(embed_params),
        format_params(block_params),
        format_params(predictor_params),
    );
    println!("Inference (latent diff): phrase _ more => answer (empty line to quit)\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = line?.trim().to_string();
        if line.is_empty() {
            break;
        }
        let input = match format_user_input(&line) {
            Ok(i) => i,
            Err(FormatInputError::NoSeparator) => {
                println!(
                    "(use => to separate phrase from answer, e.g. hello _ world => beautiful)\n"
                );
                continue;
            }
            Err(FormatInputError::NoMask) => {
                println!("(no _ or [MASK] found in phrase)\n");
                continue;
            }
        };
        let answer_token = input.answer_token.clone();
        let answer_id = match vocab.token_to_id.get(&answer_token) {
            Some(&id) => id,
            None => {
                println!("(answer token '{}' not in vocab)\n", answer_token);
                continue;
            }
        };

        // Train/inference parity: when the mask is at position P, the input at P is [MASK], so
        // the true token is not in the sequence there. The model was trained to predict from
        // context like [good, luck, MASK, school., ...] (right context = "school."). If we feed
        // [good, luck, MASK, with, school.], right context differs. Strip the answer from the
        // phrase so encoder input matches training.
        let tokens: Vec<String> = input
            .tokens
            .into_iter()
            .filter(|t| t != &answer_token)
            .collect();
        let mask_pos = match tokens.iter().position(|t| t == "<mask>") {
            Some(p) => p,
            None => {
                println!("(no mask in phrase after removing answer)\n");
                continue;
            }
        };

        let mut ids = vocab.encode(&tokens);
        // Must match training: one phrase sequence padded/truncated to max_seq.
        if ids.len() > max_seq {
            ids.truncate(max_seq);
        }
        while ids.len() < max_seq {
            ids.push(vocab.pad_id);
        }
        let seq_len = max_seq;

        let input_t = Tensor::from_vec(ids.clone(), (1, seq_len), &device)?;
        let hidden = encoder.forward_sequence(&input_t)?;
        let hidden_at_mask = hidden.narrow(1, mask_pos, 1)?; // [1, 1, dim]
        let pred_embed = predictor.forward(&hidden_at_mask)?; // [1, 1, dim] -> squeeze to [dim]
        let pred_embed = pred_embed.squeeze(0)?.squeeze(0)?; // [dim]

        let answer_ids = Tensor::from_vec(vec![answer_id], (1,), &device)?;
        let answer_embed = encoder.embed_tokens(&answer_ids)?.squeeze(0)?; // [dim]

        // Normalize to unit vectors (same as training objective) so L2 and cosine are comparable
        let norm_p = pred_embed
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-8);
        let norm_p_t =
            Tensor::from_vec(vec![norm_p], (1,), &device)?.broadcast_as(pred_embed.shape())?;
        let pred_unit = (pred_embed.clone() / norm_p_t)?;
        let norm_a = answer_embed
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-8);
        let norm_a_t =
            Tensor::from_vec(vec![norm_a], (1,), &device)?.broadcast_as(answer_embed.shape())?;
        let answer_unit = (answer_embed.clone() / norm_a_t)?;

        let cos = (pred_unit.clone() * answer_unit.clone())?
            .sum_all()?
            .to_scalar::<f32>()?;
        let l2_unit = (pred_unit.clone() - answer_unit)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt(); // sqrt(2-2*cos) for unit vectors

        // Top-5 nearest tokens in latent space (cosine); Candle has no topk, so we sort in Rust
        let vocab_size = vocab.id_to_token.len();
        let all_ids = Tensor::from_vec(
            (0..vocab_size as u32).collect::<Vec<_>>(),
            (vocab_size,),
            &device,
        )?;
        let all_embeds = encoder.embed_tokens(&all_ids)?; // [V, dim]
        let row_norm = all_embeds
            .sqr()?
            .sum(1)?
            .unsqueeze(1)?
            .sqrt()?
            .clamp(1e-8, 1e10)?;
        let all_unit = (all_embeds.clone() / row_norm.broadcast_as(all_embeds.shape())?)?;
        let cos_scores = all_unit.matmul(&pred_unit.unsqueeze(1)?)?.squeeze(1)?; // [V]
        let scores_vec: Vec<f32> = cos_scores.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = (0..vocab_size).zip(scores_vec).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let answer_idx: usize = answer_id.try_into().unwrap_or(0);
        let rank_1based = match indexed.iter().position(|&(i, _)| i == answer_idx) {
            Some(p) => p + 1,
            None => 0usize,
        };
        let top_tokens: Vec<String> = indexed
            .iter()
            .take(5)
            .map(|&(i, _)| {
                vocab
                    .id_to_token
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "?".to_string())
            })
            .collect();

        println!(
            "  L2 (unit): {:.4}  cosine: {:.4}  (1.0 = perfect alignment)",
            l2_unit, cos
        );
        println!(
            "  Top-5 nearest: {}  (your answer \"{}\" rank: {}/{} cos: {:.4})\n",
            top_tokens.join(", "),
            &answer_token,
            rank_1based,
            vocab_size,
            cos
        );
        stdout.flush()?;
    }
    Ok(())
}
