use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use std::path::PathBuf;

use crate::model::vocab::{Pair, Vocab};

pub fn split_line(line: &str) -> Option<(Vec<String>, Vec<String>)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some((left, right)) = line.split_once('\t') {
        let ctx: Vec<String> = left.split_whitespace().map(|t| t.to_string()).collect();
        let tgt: Vec<String> = right.split_whitespace().map(|t| t.to_string()).collect();
        if ctx.is_empty() || tgt.is_empty() {
            return None;
        }
        return Some((ctx, tgt));
    }
    if let Some((left, right)) = line.split_once("|||") {
        let ctx: Vec<String> = left.split_whitespace().map(|t| t.to_string()).collect();
        let tgt: Vec<String> = right.split_whitespace().map(|t| t.to_string()).collect();
        if ctx.is_empty() || tgt.is_empty() {
            return None;
        }
        return Some((ctx, tgt));
    }
    let tokens: Vec<String> = line.split_whitespace().map(|t| t.to_string()).collect();
    if tokens.len() < 2 {
        return None;
    }
    let mid = tokens.len() / 2;
    Some((tokens[..mid].to_vec(), tokens[mid..].to_vec()))
}

pub fn build_vocab_and_pairs(path: &PathBuf) -> Result<(Vocab, Vec<Pair>)> {
    let text = fs::read_to_string(path)?;
    let mut vocab = Vocab::new();
    let mut pairs = Vec::new();
    for line in text.lines() {
        let Some((ctx_tokens, tgt_tokens)) = split_line(line) else {
            continue;
        };
        for token in ctx_tokens.iter().chain(tgt_tokens.iter()) {
            vocab.add_token(token);
        }
        pairs.push(Pair {
            context: vocab.encode(&ctx_tokens),
            target: vocab.encode(&tgt_tokens),
        });
    }
    if pairs.is_empty() {
        bail!("no usable lines found in {:?}", path);
    }
    Ok((vocab, pairs))
}

pub fn pad_or_truncate(ids: &mut Vec<u32>, max_len: usize, pad_id: u32) {
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
}

pub fn make_batch(
    pairs: &[Pair],
    batch_size: usize,
    max_ctx: usize,
    max_tgt: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = thread_rng();
    let mut ctx_buf = Vec::with_capacity(batch_size * max_ctx);
    let mut tgt_buf = Vec::with_capacity(batch_size * max_tgt);
    for _ in 0..batch_size {
        let pair = pairs.choose(&mut rng).expect("dataset is non-empty");
        let mut ctx = pair.context.clone();
        let mut tgt = pair.target.clone();
        pad_or_truncate(&mut ctx, max_ctx, pad_id);
        pad_or_truncate(&mut tgt, max_tgt, pad_id);
        ctx_buf.extend(ctx);
        tgt_buf.extend(tgt);
    }
    let ctx = Tensor::from_vec(ctx_buf, (batch_size, max_ctx), device)?;
    let tgt = Tensor::from_vec(tgt_buf, (batch_size, max_tgt), device)?;
    Ok((ctx, tgt))
}

pub fn encode_text(vocab: &Vocab, text: &str, max_len: usize, pad_id: u32) -> Vec<u32> {
    let tokens: Vec<String> = text.split_whitespace().map(|t| t.to_string()).collect();
    let mut ids = vocab.encode(&tokens);
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
    ids
}
