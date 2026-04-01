mod config;
mod data;
mod model;
mod tasks;
mod util;

use anyhow::{bail, Result};
use tracing_subscriber::EnvFilter;
use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::{Path, PathBuf};

use config::Config;
use data::{
    build_vocab_from_pair_file, count_pairs_with_vocab, ensure_hub_dataset_cached,
    ensure_hub_wikipedia_cached, make_jepa_batch_from_pairs, prepare_ultrachat_pairs,
    PairStream, DEFAULT_MIN_TOKENS_PER_LINE, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use model::{
    load_vocab_from_file, mean_cosine_similarity, prediction_loss, save_vocab_to_file,
    sigreg_epps_pulley, tensor_rms, OnlineEncoder,
};
use model::vocab::Pair;
use tensorboard_rs::summary_writer::SummaryWriter;

fn latent_sigreg_weight(step: usize, total_steps: usize, lambda: f64) -> f64 {
    let warmup_steps = (total_steps / 10).max(1);
    let scale = (step as f64 / warmup_steps as f64).clamp(0.0, 1.0);
    lambda * scale
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
    if tasks::world::try_run_train_decoder(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_eval(&args)? {
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
            args.get(8).and_then(|v| v.parse().ok()).unwrap_or(128),
            args.get(9).and_then(|v| v.parse().ok()).unwrap_or(6),
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
        "    {} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio] [lambda]",
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
        "    {} --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--init-decoder <path>] [--decoder-output <path>]",
        args[0]
    );
    eprintln!(
        "    {} --eval-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots]",
        args[0]
    );
    eprintln!(
        "    {} --serve <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> [bind] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--debug]",
        args[0]
    );
    bail!(
        "specify a mode: --prepare-ultrachat / --latent / --latent-from-checkpoint / --eval-jepa / --train-world / --train-decoder / --eval-world / --serve"
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
    let (vocab, vocab_stats, pair_count) =
        build_vocab_from_pair_file(&config.data_path, config.max_vocab, min_tokens)?;
    let mut pair_stream = PairStream::new(
        &config.data_path,
        min_tokens.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE),
    )?;
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
    println!("Estimated parameters: ~{}", util::format_params(latent_params));

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

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), config.lr)?;

    let _ = fs::create_dir_all("local_models").and_then(|_| fs::create_dir_all("local_models/vocabs"));
    let model_path = PathBuf::from(format!(
        "local_models/model_latent_{}.safetensors",
        util::format_params(latent_params)
    ));
    let mut best_pred = f32::MAX;

    let run_dir = util::create_run_dir("latent")?;
    let mut tb = SummaryWriter::new(&run_dir);
    println!("LeJEPA: masked-view prediction + SIGReg warmup, stop-grad target branch");
    println!("TensorBoard run dir: {} (view with: tensorboard --logdir runs)", run_dir);
    println!("Streaming shuffle buffer: {}", DEFAULT_STREAM_SHUFFLE_BUFFER);

    const SIGREG_SLICES: usize = 128;
    const SIGREG_POINTS: usize = 17;
    for step in 1..=config.steps {
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
            config.max_spans_per_sample,
            config.max_span_len,
            config.max_masked_ratio,
            &device,
        )?;

        let context_hidden = encoder.forward_sequence(&context_ids)?;
        let target_hidden = encoder.forward_sequence(&target_ids)?.detach();
        let (b, t, d) = context_hidden.dims3()?;
        let context_flat = context_hidden.reshape((b * t, d))?;
        let target_flat = target_hidden.reshape((b * t, d))?;
        let context_targets = context_flat.index_select(&target_linear_indices, 0)?;
        let target_targets = target_flat.index_select(&target_linear_indices, 0)?;

        let pred_loss = prediction_loss(&context_targets, &target_targets)?;
        let sigreg_loss = sigreg_epps_pulley(&context_flat, SIGREG_SLICES, SIGREG_POINTS)?;
        let reg_weight = latent_sigreg_weight(step, config.steps, config.lambda);
        let loss = pred_loss.broadcast_add(&sigreg_loss.affine(reg_weight, 0.0)?)?;

        opt.backward_step(&loss)?;

        if step % config.log_every == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            let pred_val = pred_loss.to_scalar::<f32>()?;
            let sigreg_val = sigreg_loss.to_scalar::<f32>()?;
            let n_targets = target_linear_indices.dims1()?;
            let pred_cos = mean_cosine_similarity(&context_targets, &target_targets)?.to_scalar::<f32>()?;
            let context_rms = tensor_rms(&context_flat)?.to_scalar::<f32>()?;
            let target_rms = tensor_rms(&target_flat)?.to_scalar::<f32>()?;
            let target_frac = n_targets as f32 / (config.batch_size * config.max_seq).max(1) as f32;

            tb.add_scalar("loss/total", loss_val, step);
            tb.add_scalar("loss/pred", pred_val, step);
            tb.add_scalar("loss/sigreg", sigreg_val, step);
            tb.add_scalar("metrics/pred_cosine", pred_cos, step);
            tb.add_scalar("metrics/context_rms", context_rms, step);
            tb.add_scalar("metrics/target_rms", target_rms, step);
            tb.add_scalar("metrics/target_count", n_targets as f32, step);
            tb.add_scalar("metrics/target_frac", target_frac, step);
            tb.add_scalar("schedule/reg_weight", reg_weight as f32, step);
            tb.flush();

            if pred_val < best_pred {
                best_pred = pred_val;
                varmap.save(&model_path)?;
                println!(
                    "step {step}/{} total {loss_val:.4} pred {pred_val:.4} sigreg {sigreg_val:.4} pred_cos {pred_cos:.4} targets {n_targets} reg_w {reg_weight:.4} [saved best_pred]",
                    config.steps
                );
            } else {
                println!(
                    "step {step}/{} total {loss_val:.4} pred {pred_val:.4} sigreg {sigreg_val:.4} pred_cos {pred_cos:.4} targets {n_targets} reg_w {reg_weight:.4}",
                    config.steps
                );
            }
        }
    }

    tb.flush();
    println!("Best model saved to {:?} (pred {:.4})", model_path, best_pred);

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
    let pair_count = count_pairs_with_vocab(&data_path)?;
    let mut pair_stream = PairStream::new(&data_path, DEFAULT_MIN_TOKENS_PER_LINE)?;

    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

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
    println!("Streaming shuffle buffer: {}", DEFAULT_STREAM_SHUFFLE_BUFFER);

    let mut n_total: usize = 0;
    let mut sum_pred: f64 = 0.0;
    let mut sum_sigreg: f64 = 0.0;
    let mut sum_rank: f64 = 0.0;
    let mut sum_rr: f64 = 0.0;
    let mut top1: usize = 0;
    let mut top5: usize = 0;

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

        let online_hidden = encoder.forward_sequence(&context_ids)?; // [B, T, D]
        let target_hidden = encoder.forward_sequence(&target_ids)?;
        let (b, t, d) = online_hidden.dims3()?;
        let online_flat = online_hidden.reshape((b * t, d))?;
        let target_flat = target_hidden.reshape((b * t, d))?;

        let online_at_targets = online_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let target_latents = target_flat.index_select(&target_linear_indices, 0)?; // [N, D]
        let pred_loss = prediction_loss(&online_at_targets, &target_latents)?.to_scalar::<f32>()?;
        let sigreg_loss = sigreg_epps_pulley(&online_flat, 128, 17)?.to_scalar::<f32>()?;
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
        let pred_unit = (online_at_targets.clone() / pred_norm.broadcast_as(online_at_targets.shape())?)?;
        let n = pred_unit.dim(0)?;
        n_total += n;
        sum_pred += pred_loss as f64;
        sum_sigreg += sigreg_loss as f64;

        // In-batch latent retrieval:
        // For each context latent i, rank all target latents j by cosine(pred_i, target_j).
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
    println!("  pred_mse:       {:.4}", sum_pred / eval_steps.max(1) as f64);
    println!("  sigreg:         {:.4}", sum_sigreg / eval_steps.max(1) as f64);
    println!("  retrieval_top1: {:.4}", top1 as f64 / denom);
    println!("  retrieval_top5: {:.4}", top5 as f64 / denom);
    println!("  retrieval_mrr:  {:.4}", sum_rr / denom);
    println!("  mean_rank:      {:.2}", sum_rank / denom);
    Ok(())
}

