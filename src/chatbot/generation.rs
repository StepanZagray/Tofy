use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::{thread_rng, Rng};

use crate::model::{Decoder, Encoder, Vocab};
use crate::data::encode_text;

/// Sample a token from logits using temperature and optional top-k filtering
pub fn sample_token(logits: &Tensor, temperature: f32, top_k: Option<usize>) -> Result<u32> {
    // Get logits as 1D vector (handle both [vocab] and [1, vocab] shapes)
    let logits_1d: Vec<f32> = if logits.dims().len() == 1 {
        logits.to_vec1::<f32>()?
    } else {
        logits.to_vec2::<f32>()?[0].clone()
    };
    
    // Apply temperature scaling
    let scaled: Vec<f32> = logits_1d.iter().map(|&x| x / temperature).collect();
    
    // Create index-value pairs for top-k
    let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    
    // Top-k filtering
    if let Some(k) = top_k {
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
    }
    
    // Find max for numerical stability
    let max_logit = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, |a, b| a.max(b));
    
    // Compute softmax
    let exps: Vec<(usize, f32)> = indexed.iter().map(|(i, x)| (*i, (*x - max_logit).exp())).collect();
    let sum_exp: f32 = exps.iter().map(|(_, e)| e).sum();
    let probs: Vec<(usize, f32)> = exps.iter().map(|(i, e)| (*i, *e / sum_exp)).collect();
    
    // Sample from distribution
    let mut rng = thread_rng();
    let sample: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    
    for (idx, prob) in probs.iter() {
        cumsum += prob;
        if sample <= cumsum {
            return Ok(*idx as u32);
        }
    }
    
    Ok(probs.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

/// Generate text using transformer encoder-decoder with cross-attention
pub fn generate_text(
    encoder: &Encoder,
    decoder: &Decoder,
    vocab: &Vocab,
    input_text: &str,
    max_ctx: usize,
    max_gen: usize,
    device: &Device,
) -> Result<String> {
    // Encode input sequence
    let input_ids = encode_text(vocab, input_text, max_ctx, vocab.pad_id);
    let input_tensor = Tensor::from_vec(input_ids, (1, max_ctx), device)?;
    
    // Get encoder output sequence (for cross-attention)
    let encoder_out = encoder.forward_sequence(&input_tensor)?; // [1, T, D]
    
    // Start with pad token (or could use a special <bos> token)
    let mut generated_ids: Vec<u32> = vec![vocab.pad_id];
    
    for _ in 0..max_gen {
        // Create tensor from generated tokens so far
        let gen_tensor = Tensor::from_vec(
            generated_ids.clone(),
            (1, generated_ids.len()),
            device,
        )?;
        
        // Get logits for next token
        let logits = decoder.forward_step(&gen_tensor, &encoder_out)?; // [1, vocab]
        
        // Sample next token
        let token_id = sample_token(&logits, 0.8, Some(40))?;
        
        // Check for stop conditions
        if token_id == vocab.pad_id {
            break;
        }
        
        if let Some(token) = vocab.id_to_token.get(token_id as usize) {
            if token == "<pad>" || token == "<unk>" {
                break;
            }
            generated_ids.push(token_id);
        } else {
            break;
        }
    }
    
    // Convert token IDs to text (skip initial pad)
    let tokens: Vec<String> = generated_ids[1..]
        .iter()
        .filter_map(|&id| vocab.id_to_token.get(id as usize).cloned())
        .collect();
    
    Ok(tokens.join(" "))
}
