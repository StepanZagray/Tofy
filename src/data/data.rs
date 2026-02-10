use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::fs;
use std::path::PathBuf;

use crate::model::vocab::{Pair, Vocab};

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || (ch as u32) == 39 {
            buf.push(ch);
        } else if ch.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(buf.clone());
                buf.clear();
            }
        } else {
            if !buf.is_empty() {
                tokens.push(buf.clone());
                buf.clear();
            }
            tokens.push(ch.to_string());
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

pub fn split_line(line: &str) -> Option<Vec<String>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some((left, _right)) = line.split_once("\t") {
        let tokens = tokenize_text(left);
        return if tokens.is_empty() { None } else { Some(tokens) };
    }
    if let Some((left, _right)) = line.split_once("|||") {
        let tokens = tokenize_text(left);
        return if tokens.is_empty() { None } else { Some(tokens) };
    }
    let tokens = tokenize_text(line);
    if tokens.len() < 2 {
        return None;
    }
    Some(tokens)
}

pub struct VocabStats {
    pub total_tokens: usize,
    pub covered_tokens: usize,
    pub oov_tokens: usize,
    pub unique_tokens: usize,
    pub vocab_size: usize,
}

pub fn build_vocab_and_pairs(
    path: &PathBuf,
    max_vocab: usize,
) -> Result<(Vocab, Vec<Pair>, VocabStats)> {
    use std::collections::HashMap;

    let text = fs::read_to_string(path)?;
    let mut phrases_raw: Vec<Vec<String>> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();

    for line in text.lines() {
        let Some(tokens) = split_line(line) else {
            continue;
        };
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        phrases_raw.push(tokens);
    }

    if phrases_raw.is_empty() {
        bail!("no usable lines found in {:?}", path);
    }

    // Build vocab from most frequent tokens (reserve 3 for <pad>, <unk>, <mask>)
    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let vocab_size = (max_vocab - 3).min(sorted.len());
    let top_tokens: Vec<String> = sorted
        .iter()
        .take(vocab_size)
        .map(|(t, _)| t.clone())
        .collect();

    let mut vocab = Vocab::new();
    for token in &top_tokens {
        vocab.add_token(token);
    }

    let mut pairs = Vec::with_capacity(phrases_raw.len());
    for tokens in phrases_raw {
        pairs.push(Pair {
            tokens: vocab.encode(&tokens),
        });
    }

    let total_tokens: usize = sorted.iter().map(|(_, c)| *c).sum();
    let covered_tokens: usize = sorted.iter().take(vocab_size).map(|(_, c)| *c).sum();
    let oov_tokens = total_tokens.saturating_sub(covered_tokens);
    let unique_tokens = sorted.len();
    let stats = VocabStats {
        total_tokens,
        covered_tokens,
        oov_tokens,
        unique_tokens,
        vocab_size: vocab.id_to_token.len(),
    };

    Ok((vocab, pairs, stats))
}

pub fn pad_or_truncate(ids: &mut Vec<u32>, max_len: usize, pad_id: u32) {
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
}

/// Build a latent training batch from one phrase per example.
/// Returns:
/// - input_ids: [B, max_seq]
/// - mask_positions: [B]
/// - target_ids: [B]
pub fn make_latent_batch(
    pairs: &[Pair],
    batch_size: usize,
    max_seq: usize,
    pad_id: u32,
    mask_id: u32,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = thread_rng();
    let mut input_buf = Vec::with_capacity(batch_size * max_seq);
    let mut mask_pos_buf = Vec::with_capacity(batch_size);
    let mut target_buf = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let pair = pairs.choose(&mut rng).expect("dataset non-empty");
        let mut seq = pair.tokens.clone();
        pad_or_truncate(&mut seq, max_seq, pad_id);

        let valid_positions: Vec<usize> = (0..max_seq)
            .filter(|&i| seq[i] != pad_id && seq[i] != mask_id)
            .collect();
        let &mask_pos = valid_positions.choose(&mut rng).unwrap_or(&0);
        let target_token = seq[mask_pos];

        // BERT-style masking noise:
        // 80% replace with <mask>, 10% random token, 10% keep original.
        match rng.gen_range(0..10) {
            0..=7 => seq[mask_pos] = mask_id,
            8 => seq[mask_pos] = rng.gen_range(0..vocab_size as u32),
            _ => {}
        }

        input_buf.extend(seq);
        mask_pos_buf.push(mask_pos as u32);
        target_buf.push(target_token);
    }

    let input_ids = Tensor::from_vec(input_buf, (batch_size, max_seq), device)?;
    let mask_positions = Tensor::from_vec(mask_pos_buf, (batch_size,), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size,), device)?;
    Ok((input_ids, mask_positions, target_ids))
}
