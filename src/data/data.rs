use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::HashSet;
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
    split_line_with_min_tokens(line, 2)
}

/// Like split_line but accepts lines with at least `min_tokens` tokens (e.g. 1 for paragraph mode).
pub fn split_line_with_min_tokens(line: &str, min_tokens: usize) -> Option<Vec<String>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let tokens = if let Some((left, _right)) = line.split_once("\t") {
        tokenize_text(left)
    } else if let Some((left, _right)) = line.split_once("|||") {
        tokenize_text(left)
    } else {
        tokenize_text(line)
    };
    if tokens.len() < min_tokens {
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

#[derive(Clone)]
pub struct WorldExample {
    pub state_tokens: Vec<u32>,
    pub next_tokens: Vec<u32>,
}

/// Minimum number of tokens per line to include (2 = skip single-token lines; 1 = paragraph mode).
pub const DEFAULT_MIN_TOKENS_PER_LINE: usize = 2;

pub fn build_vocab_and_pairs(
    path: &PathBuf,
    max_vocab: usize,
    min_tokens_per_line: Option<usize>,
) -> Result<(Vocab, Vec<Pair>, VocabStats)> {
    use std::collections::HashMap;

    let min_tok = min_tokens_per_line.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE);
    let text = fs::read_to_string(path)?;
    let mut phrases_raw: Vec<Vec<String>> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();

    for line in text.lines() {
        let Some(tokens) = split_line_with_min_tokens(line, min_tok) else {
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

/// Build encoded pairs from a data file using an existing vocab (used for evaluation).
pub fn build_pairs_with_vocab(path: &PathBuf, vocab: &Vocab) -> Result<Vec<Pair>> {
    let text = fs::read_to_string(path)?;
    let mut pairs = Vec::new();
    for line in text.lines() {
        let Some(tokens) = split_line(line) else {
            continue;
        };
        pairs.push(Pair {
            tokens: vocab.encode(&tokens),
        });
    }
    if pairs.is_empty() {
        bail!("no usable lines found in {:?}", path);
    }
    Ok(pairs)
}

pub fn pad_or_truncate(ids: &mut Vec<u32>, max_len: usize, pad_id: u32) {
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
}

/// Build a JEPA-style batch with:
/// - context view (target regions masked out),
/// - target view (original sequence),
/// - flattened indices of all target positions to align.
///
/// Returns:
/// - context_ids: [B, max_seq]
/// - target_ids: [B, max_seq]
/// - target_linear_indices: [N_targets] where each index is b * max_seq + pos
///
/// Each sample is exactly one paragraph/line (one topic); masking is within that sequence only.
/// Masked positions are capped at `max_masked_ratio` of valid (non-pad) context so the model
/// always sees most of the sequence (common practice: BERT ~15%, span masking often cap at 25–30%).
pub fn make_jepa_batch(
    pairs: &[Pair],
    batch_size: usize,
    max_seq: usize,
    pad_id: u32,
    mask_id: u32,
    max_spans_per_sample: usize,
    max_span_len: usize,
    max_masked_ratio: f64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = thread_rng();
    let mut context_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_linear = Vec::new();

    let span_count_cap = max_spans_per_sample.max(1);
    let span_len_cap = max_span_len.max(1);
    let ratio = max_masked_ratio.clamp(0.01, 1.0);

    for b in 0..batch_size {
        let pair = pairs.choose(&mut rng).expect("dataset non-empty");
        let mut seq = pair.tokens.clone();
        pad_or_truncate(&mut seq, max_seq, pad_id);
        let target_seq = seq.clone();
        let mut context_seq = seq;

        let valid_positions: Vec<usize> = (0..max_seq)
            .filter(|&i| target_seq[i] != pad_id && target_seq[i] != mask_id)
            .collect();
        let mut selected: HashSet<usize> = HashSet::new();

        if !valid_positions.is_empty() {
            let max_masked = (valid_positions.len() as f64 * ratio).ceil() as usize;
            let max_masked = max_masked.max(1).min(valid_positions.len());

            let span_count = rng.gen_range(1..=span_count_cap);
            for _ in 0..span_count {
                if selected.len() >= max_masked {
                    break;
                }
                let Some(&start) = valid_positions.choose(&mut rng) else {
                    break;
                };
                let span_len = rng.gen_range(1..=span_len_cap);
                for p in start..(start + span_len).min(max_seq) {
                    if selected.len() >= max_masked {
                        break;
                    }
                    if target_seq[p] != pad_id && target_seq[p] != mask_id {
                        selected.insert(p);
                    } else {
                        break;
                    }
                }
            }
            if selected.is_empty() {
                if let Some(&fallback) = valid_positions.choose(&mut rng) {
                    selected.insert(fallback);
                }
            }
        } else {
            selected.insert(0);
        }

        let mut selected_positions: Vec<usize> = selected.into_iter().collect();
        selected_positions.sort_unstable();
        for &p in &selected_positions {
            context_seq[p] = mask_id;
            target_linear.push((b * max_seq + p) as u32);
        }

        context_buf.extend(context_seq);
        target_buf.extend(target_seq);
    }

    let context_ids = Tensor::from_vec(context_buf, (batch_size, max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, max_seq), device)?;
    let n_targets = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear, (n_targets,), device)?;
    Ok((context_ids, target_ids, target_linear_indices))
}

pub fn tokenize_for_inference(text: &str) -> Vec<String> {
    tokenize_text(text)
}

pub fn build_vocab_and_world_examples(
    path: &PathBuf,
    max_vocab: usize,
) -> Result<(Vocab, Vec<WorldExample>, VocabStats)> {
    use std::collections::HashMap;

    let text = fs::read_to_string(path)?;
    let mut raw_rows: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((left, right)) = line
            .split_once('\t')
            .or_else(|| line.split_once("|||"))
        else {
            continue;
        };
        let state_tokens = tokenize_text(left.trim());
        let next_tokens = tokenize_text(right.trim());
        if state_tokens.is_empty() || next_tokens.is_empty() {
            continue;
        }
        for tok in state_tokens.iter().chain(next_tokens.iter()) {
            *counts.entry(tok.clone()).or_insert(0) += 1;
        }
        raw_rows.push((state_tokens, next_tokens));
    }

    if raw_rows.is_empty() {
        bail!("no usable world-model rows found in {:?}", path);
    }

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

    let mut rows = Vec::with_capacity(raw_rows.len());
    for (state_tokens, next_tokens) in raw_rows {
        rows.push(WorldExample {
            state_tokens: vocab.encode(&state_tokens),
            next_tokens: vocab.encode(&next_tokens),
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

    Ok((vocab, rows, stats))
}

pub fn build_world_examples_with_vocab(path: &PathBuf, vocab: &Vocab) -> Result<Vec<WorldExample>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((left, right)) = line
            .split_once('\t')
            .or_else(|| line.split_once("|||"))
        else {
            continue;
        };
        let state_tokens = tokenize_text(left.trim());
        let next_tokens = tokenize_text(right.trim());
        if state_tokens.is_empty() || next_tokens.is_empty() {
            continue;
        }
        rows.push(WorldExample {
            state_tokens: vocab.encode(&state_tokens),
            next_tokens: vocab.encode(&next_tokens),
        });
    }
    if rows.is_empty() {
        bail!("no usable world-model rows found in {:?}", path);
    }
    Ok(rows)
}

fn encode_sequence(
    tokens: &[u32],
    max_seq: usize,
    pad_id: u32,
) -> (Vec<u32>, usize) {
    let mut seq = Vec::with_capacity(max_seq);
    for &id in tokens.iter().take(max_seq) {
        seq.push(id);
    }
    let length = seq.len().max(1);
    while seq.len() < max_seq {
        seq.push(pad_id);
    }
    (seq, length)
}

pub fn make_world_batch(
    rows: &[WorldExample],
    batch_size: usize,
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>, Vec<usize>)> {
    let mut rng = thread_rng();
    let mut state_buf = Vec::with_capacity(batch_size * max_seq);
    let mut next_buf = Vec::with_capacity(batch_size * max_seq);
    let mut state_lens = Vec::with_capacity(batch_size);
    let mut next_lens = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let row = rows.choose(&mut rng).expect("dataset non-empty");
        let (state_seq, state_len) = encode_sequence(&row.state_tokens, max_seq, pad_id);
        let (next_seq, next_len) = encode_sequence(&row.next_tokens, max_seq, pad_id);
        state_buf.extend(state_seq);
        next_buf.extend(next_seq);
        state_lens.push(state_len);
        next_lens.push(next_len);
    }

    let state_ids = Tensor::from_vec(state_buf, (batch_size, max_seq), device)?;
    let next_ids = Tensor::from_vec(next_buf, (batch_size, max_seq), device)?;
    Ok((state_ids, next_ids, state_lens, next_lens))
}

pub fn make_world_batch_from_slice(
    rows: &[WorldExample],
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>, Vec<usize>)> {
    let batch_size = rows.len();
    let mut state_buf = Vec::with_capacity(batch_size * max_seq);
    let mut next_buf = Vec::with_capacity(batch_size * max_seq);
    let mut state_lens = Vec::with_capacity(batch_size);
    let mut next_lens = Vec::with_capacity(batch_size);

    for row in rows {
        let (state_seq, state_len) = encode_sequence(&row.state_tokens, max_seq, pad_id);
        let (next_seq, next_len) = encode_sequence(&row.next_tokens, max_seq, pad_id);
        state_buf.extend(state_seq);
        next_buf.extend(next_seq);
        state_lens.push(state_len);
        next_lens.push(next_len);
    }

    let state_ids = Tensor::from_vec(state_buf, (batch_size, max_seq), device)?;
    let next_ids = Tensor::from_vec(next_buf, (batch_size, max_seq), device)?;
    Ok((state_ids, next_ids, state_lens, next_lens))
}
