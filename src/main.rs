mod chatbot;
mod config;
mod data;
mod model;

use anyhow::{bail, Result};
use candle_core::{Device, DType, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::fs;
use std::path::PathBuf;

use config::Config;
use data::{build_vocab_and_pairs, make_batch};
use model::{Decoder, Encoder};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    // Check for chatbot mode
    if args.len() >= 2 && (args[1] == "--chat" || args[1] == "chat") {
        if args.len() < 5 {
            bail!("usage: {} --chat <model_path> <vocab_path> <data_path> [dim] [max_ctx] [num_layers] [num_heads]", args[0]);
        }
        let model_path = PathBuf::from(&args[2]);
        let vocab_path = PathBuf::from(&args[3]);
        let data_path = PathBuf::from(&args[4]);
        let dim = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(256);
        let max_ctx = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(64);
        let num_layers = args.get(7).and_then(|v| v.parse().ok()).unwrap_or(4);
        let num_heads = args.get(8).and_then(|v| v.parse().ok()).unwrap_or(4);
        return chatbot::run_chatbot(&model_path, &vocab_path, &data_path, dim, max_ctx, num_layers, num_heads);
    }
    
    let config = Config::from_args()?;
    let device = match Device::new_cuda(0) {
        Ok(device) => {
            eprintln!("using device: CUDA(0)");
            device
        }
        Err(err) => {
            eprintln!("CUDA not available, falling back to CPU: {err}");
            Device::Cpu
        }
    };
    let (vocab, pairs) = build_vocab_and_pairs(&config.data_path)?;
    let vocab_size = vocab.id_to_token.len();
    
    println!("Vocab size: {}", vocab_size);
    println!("Training pairs: {}", pairs.len());
    println!("Model config: dim={}, layers={}, heads={}", config.dim, config.num_layers, config.num_heads);
    
    // Estimate parameter count
    let embed_params = vocab_size * config.dim * 2; // encoder + decoder embeddings
    let attn_params = config.num_layers * (4 * config.dim * config.dim); // Q,K,V,O per layer
    let ff_params = config.num_layers * (2 * config.dim * config.dim * 4); // 2 FF layers, 4x dim
    let total_params = embed_params + (attn_params + ff_params) * 2; // encoder + decoder
    println!("Estimated parameters: ~{}M", total_params / 1_000_000);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Transformer encoder with self-attention
    let encoder = Encoder::new(
        vb.pp("encoder"),
        vocab_size,
        config.dim,
        config.num_layers,
        config.num_heads,
    )?;
    
    // Transformer decoder with cross-attention
    let decoder = Decoder::new(
        vb.pp("decoder"),
        config.dim,
        vocab_size,
        config.num_layers,
        config.num_heads,
    )?;

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), config.lr)?;

    println!("\nStarting training...\n");
    
    for step in 1..=config.steps {
        let (ctx_ids, tgt_ids) = make_batch(
            &pairs,
            config.batch_size,
            config.max_ctx,
            config.max_tgt,
            vocab.pad_id,
            &device,
        )?;

        // Encode context sequence
        let encoder_out = encoder.forward_sequence(&ctx_ids)?; // [B, T_ctx, D]
        
        // Decoder input: target shifted right (teacher forcing)
        // Input: [<pad>, tok1, tok2, ...] -> Predict: [tok1, tok2, tok3, ...]
        let tgt_input = shift_right(&tgt_ids, vocab.pad_id)?;
        
        // Get decoder predictions for all positions
        let logits = decoder.forward(&tgt_input, &encoder_out)?; // [B, T_tgt, vocab]
        
        // Cross-entropy loss over all tokens (excluding padding)
        let loss = cross_entropy_loss(&logits, &tgt_ids, vocab.pad_id)?;
        
        opt.backward_step(&loss)?;

        if step % config.log_every == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            println!("step {step}/{} loss {loss_val:.4}", config.steps);
        }
    }

    println!("\nTraining completed! ({} steps)", config.steps);
    
    // Save model and vocab
    let model_path = PathBuf::from("model.safetensors");
    varmap.save(&model_path)?;
    println!("Model saved to {:?}", model_path);
    
    let vocab_path = PathBuf::from("vocab.txt");
    let mut vocab_text = String::new();
    for token in &vocab.id_to_token {
        vocab_text.push_str(token);
        vocab_text.push('\n');
    }
    fs::write(&vocab_path, vocab_text)?;
    println!("Vocab saved to {:?}", vocab_path);
    
    println!("\nTo chat with the model:");
    println!("  cargo run --release -- --chat model.safetensors vocab.txt {} {} {} {} {}", 
             config.data_path.display(), config.dim, config.max_ctx, config.num_layers, config.num_heads);
    
    Ok(())
}

/// Shift target sequence right for teacher forcing
/// [tok1, tok2, tok3, pad] -> [pad, tok1, tok2, tok3]
fn shift_right(tgt: &Tensor, pad_id: u32) -> Result<Tensor> {
    let (batch, seq_len) = tgt.dims2()?;
    let device = tgt.device();
    
    // Create pad column
    let pad_col = Tensor::full(pad_id, (batch, 1), device)?;
    
    // Take all but last token from target
    let tgt_prefix = tgt.narrow(1, 0, seq_len - 1)?;
    
    // Concatenate: [pad, tok1, tok2, ...]
    Ok(Tensor::cat(&[&pad_col, &tgt_prefix], 1)?)
}

/// Cross-entropy loss with padding mask
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor, pad_id: u32) -> Result<Tensor> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;
    
    // Reshape for cross-entropy: [B*T, vocab] and [B*T]
    let logits_flat = logits.reshape((batch * seq_len, vocab_size))?;
    let targets_flat = targets.reshape(batch * seq_len)?;
    
    // Compute log softmax using Candle's built-in ops
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
    
    // Gather the log probs for the target tokens
    let targets_u32 = targets_flat.to_vec1::<u32>()?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    
    let mut loss_sum = 0.0f32;
    let mut count = 0usize;
    
    for (i, &target_id) in targets_u32.iter().enumerate() {
        if target_id != pad_id {
            loss_sum -= log_probs_vec[i][target_id as usize];
            count += 1;
        }
    }
    
    let avg_loss = if count > 0 { loss_sum / count as f32 } else { 0.0 };
    Ok(Tensor::new(avg_loss, logits.device())?)
}
