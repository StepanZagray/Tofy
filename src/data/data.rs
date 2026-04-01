use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
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

/// Heuristic action label for orchestrator training: 0 = TextReply, 1 = Code.
/// Used when building world examples from raw text (next turn string).
pub fn action_label_heuristic(next_turn: &str) -> u32 {
    let s = next_turn.trim();
    if s.contains("```") {
        return 1; // code block
    }
    let code_chars = s.chars().filter(|c| *c == '{' || *c == '}' || *c == ';' || *c == '(' || *c == ')').count();
    let len = s.chars().count().max(1);
    if code_chars * 10 > len {
        return 1; // code-like
    }
    0 // text
}

#[derive(Clone)]
pub struct WorldExample {
    pub state_tokens: Vec<u32>,
    pub next_tokens: Vec<u32>,
    /// 0 = TextReply, 1 = Code (for orchestrator action head training).
    pub action_label: u32,
}

#[derive(Clone)]
pub struct RawWorldExample {
    pub state_tokens: Vec<String>,
    pub next_tokens: Vec<String>,
    pub action_label: u32,
}

/// Minimum number of tokens per line to include (2 = skip single-token lines; 1 = paragraph mode).
pub const DEFAULT_MIN_TOKENS_PER_LINE: usize = 2;
pub const DEFAULT_STREAM_SHUFFLE_BUFFER: usize = 1024;

pub struct PairStream {
    path: PathBuf,
    min_tokens: usize,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<Vec<String>>,
}

impl PairStream {
    pub fn new(path: &PathBuf, min_tokens: usize) -> Result<Self> {
        Self::with_shuffle(path, min_tokens, DEFAULT_STREAM_SHUFFLE_BUFFER)
    }

    pub fn with_shuffle(path: &PathBuf, min_tokens: usize, shuffle_buffer_size: usize) -> Result<Self> {
        Ok(Self {
            path: path.clone(),
            min_tokens,
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = BufReader::new(File::open(&self.path)?);
        Ok(())
    }

    fn read_next_tokens(&mut self) -> Result<Vec<String>> {
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                self.reset()?;
                continue;
            }
            if let Some(tokens) = split_line_with_min_tokens(&line, self.min_tokens) {
                return Ok(tokens);
            }
        }
    }

    fn refill_shuffle_buffer(&mut self) -> Result<()> {
        while self.shuffle_buffer.len() < self.shuffle_buffer_size {
            let tokens = self.read_next_tokens()?;
            self.shuffle_buffer.push(tokens);
        }
        Ok(())
    }

    fn next_tokens(&mut self) -> Result<Vec<String>> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_tokens();
        }
        self.refill_shuffle_buffer()?;
        let idx = thread_rng().gen_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<Vec<String>>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_tokens()?);
        }
        Ok(batch)
    }
}

pub struct RawWorldStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<RawWorldExample>,
}

impl RawWorldStream {
    pub fn new(path: &PathBuf) -> Result<Self> {
        Self::with_shuffle(path, DEFAULT_STREAM_SHUFFLE_BUFFER)
    }

    pub fn with_shuffle(path: &PathBuf, shuffle_buffer_size: usize) -> Result<Self> {
        Ok(Self {
            path: path.clone(),
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = BufReader::new(File::open(&self.path)?);
        Ok(())
    }

    fn read_next_example(&mut self) -> Result<RawWorldExample> {
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                self.reset()?;
                continue;
            }
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
            let next_str = next_tokens.join(" ");
            return Ok(RawWorldExample {
                state_tokens,
                next_tokens,
                action_label: action_label_heuristic(&next_str),
            });
        }
    }

    fn refill_shuffle_buffer(&mut self) -> Result<()> {
        while self.shuffle_buffer.len() < self.shuffle_buffer_size {
            let example = self.read_next_example()?;
            self.shuffle_buffer.push(example);
        }
        Ok(())
    }

    fn next_example(&mut self) -> Result<RawWorldExample> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_example();
        }
        self.refill_shuffle_buffer()?;
        let idx = thread_rng().gen_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<RawWorldExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
    }
}

pub fn build_vocab_from_pair_file(
    path: &PathBuf,
    max_vocab: usize,
    min_tokens_per_line: Option<usize>,
) -> Result<(Vocab, VocabStats, usize)> {
    use std::collections::HashMap;

    let min_tok = min_tokens_per_line.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut pair_count = 0usize;
    let reader = BufReader::new(File::open(path)?);

    for line in reader.lines() {
        let line = line?;
        let Some(tokens) = split_line_with_min_tokens(&line, min_tok) else {
            continue;
        };
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        pair_count += 1;
    }

    if pair_count == 0 {
        bail!("no usable lines found in {:?}", path);
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

    Ok((vocab, stats, pair_count))
}

pub fn count_pairs_with_vocab(path: &PathBuf) -> Result<usize> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if split_line(&line).is_some() {
            count += 1;
        }
    }
    if count == 0 {
        bail!("no usable lines found in {:?}", path);
    }
    Ok(count)
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
pub fn make_jepa_batch_from_pairs(
    pairs: &[Pair],
    max_seq: usize,
    pad_id: u32,
    mask_id: u32,
    max_spans_per_sample: usize,
    max_span_len: usize,
    max_masked_ratio: f64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = thread_rng();
    let batch_size = pairs.len();
    let mut context_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_linear = Vec::new();

    let span_count_cap = max_spans_per_sample.max(1);
    let span_len_cap = max_span_len.max(1);
    let ratio = max_masked_ratio.clamp(0.01, 1.0);

    for (b, pair) in pairs.iter().enumerate() {
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

pub fn encode_world_examples(rows: &[RawWorldExample], vocab: &Vocab) -> Vec<WorldExample> {
    rows.iter()
        .map(|row| WorldExample {
            state_tokens: vocab.encode(&row.state_tokens),
            next_tokens: vocab.encode(&row.next_tokens),
            action_label: row.action_label,
        })
        .collect()
}

pub fn build_vocab_from_raw_world_file(
    path: &PathBuf,
    max_vocab: usize,
) -> Result<(Vocab, VocabStats, usize)> {
    use std::collections::HashMap;

    let reader = BufReader::new(File::open(path)?);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut row_count = 0usize;
    for line in reader.lines() {
        let line = line?;
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
        row_count += 1;
    }
    if row_count == 0 {
        bail!("cannot build vocab from empty raw world file {:?}", path);
    }

    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let vocab_size = (max_vocab - 3).min(sorted.len());
    let mut vocab = Vocab::new();
    for (token, _) in sorted.iter().take(vocab_size) {
        vocab.add_token(token);
    }
    let total_tokens: usize = sorted.iter().map(|(_, c)| *c).sum();
    let covered_tokens: usize = sorted.iter().take(vocab_size).map(|(_, c)| *c).sum();
    let oov_tokens = total_tokens.saturating_sub(covered_tokens);
    let unique_tokens = sorted.len();
    let vocab_len = vocab.id_to_token.len();
    Ok((
        vocab,
        VocabStats {
            total_tokens,
            covered_tokens,
            oov_tokens,
            unique_tokens,
            vocab_size: vocab_len,
        },
        row_count,
    ))
}

pub fn count_raw_world_rows(path: &PathBuf) -> Result<usize> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
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
        if tokenize_text(left.trim()).is_empty() || tokenize_text(right.trim()).is_empty() {
            continue;
        }
        count += 1;
    }
    if count == 0 {
        bail!("no usable world-model rows found in {:?}", path);
    }
    Ok(count)
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

/// Build decoder teacher-forcing batch from world batch.
/// input[b] = state[b, 0..state_len] ++ next[b, 0..next_len-1], padded to decoder_len.
/// target[b] = state[b, 1..state_len] ++ next[b, 0..next_len], padded to decoder_len.
/// loss_mask[b] = 1.0 for (state_len-1)+next_len positions, 0.0 elsewhere.
/// decoder_len = 2 * max_seq.
pub fn make_decoder_batch(
    state_ids: &Tensor,
    next_ids: &Tensor,
    state_lens: &[usize],
    next_lens: &[usize],
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch_size, _) = state_ids.dims2()?;
    let decoder_len = 2 * max_seq;
    let state_v = state_ids.to_vec2::<u32>()?;
    let next_v = next_ids.to_vec2::<u32>()?;

    let mut input_buf = Vec::with_capacity(batch_size * decoder_len);
    let mut target_buf = Vec::with_capacity(batch_size * decoder_len);
    let mut mask_buf = Vec::with_capacity(batch_size * decoder_len);

    for b in 0..batch_size {
        let sl = state_lens.get(b).copied().unwrap_or(1).min(max_seq);
        let nl = next_lens.get(b).copied().unwrap_or(1).min(max_seq);

        for i in 0..sl {
            input_buf.push(state_v[b][i]);
        }
        for i in 0..nl.saturating_sub(1) {
            input_buf.push(next_v[b][i]);
        }
        let input_len = sl + nl.saturating_sub(1);
        for _ in input_len..decoder_len {
            input_buf.push(pad_id);
        }

        for i in 1..sl {
            target_buf.push(state_v[b][i]);
        }
        for i in 0..nl {
            target_buf.push(next_v[b][i]);
        }
        let target_len = sl.saturating_sub(1) + nl;
        for _ in target_len..decoder_len {
            target_buf.push(pad_id);
        }

        let n_loss = sl.saturating_sub(1) + nl;
        for _ in 0..n_loss {
            mask_buf.push(1.0f32);
        }
        for _ in n_loss..decoder_len {
            mask_buf.push(0.0f32);
        }
    }

    let input_ids = Tensor::from_vec(input_buf, (batch_size, decoder_len), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, decoder_len), device)?;
    let loss_mask = Tensor::from_vec(mask_buf, (batch_size, decoder_len), device)?;
    Ok((input_ids, target_ids, loss_mask))
}

pub fn make_world_batch_from_slice(
    rows: &[WorldExample],
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>, Vec<usize>, Vec<u32>)> {
    let batch_size = rows.len();
    let mut state_buf = Vec::with_capacity(batch_size * max_seq);
    let mut next_buf = Vec::with_capacity(batch_size * max_seq);
    let mut state_lens = Vec::with_capacity(batch_size);
    let mut next_lens = Vec::with_capacity(batch_size);
    let mut action_labels = Vec::with_capacity(batch_size);

    for row in rows {
        let (state_seq, state_len) = encode_sequence(&row.state_tokens, max_seq, pad_id);
        let (next_seq, next_len) = encode_sequence(&row.next_tokens, max_seq, pad_id);
        state_buf.extend(state_seq);
        next_buf.extend(next_seq);
        state_lens.push(state_len);
        next_lens.push(next_len);
        action_labels.push(row.action_label);
    }

    let state_ids = Tensor::from_vec(state_buf, (batch_size, max_seq), device)?;
    let next_ids = Tensor::from_vec(next_buf, (batch_size, max_seq), device)?;
    Ok((state_ids, next_ids, state_lens, next_lens, action_labels))
}
