use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
use rand::rng;
use rand::seq::IndexedRandom;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use std::time::Instant;

use crate::model::vocab::{Pair, Vocab};

pub const PAIR_SOURCE_MANIFEST_HEADER: &str = "# tofy-pair-sources-v1";

fn stable_split_hash_parts<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> usize {
    fn update(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    let mut hash = 0xcbf29ce484222325u64;
    for part in parts {
        hash = update(hash, part);
        hash = update(hash, &[0xff]);
    }
    hash as usize
}

fn stable_token_pair_split_hash(left: &[u32], right: &[u32], action: u32) -> usize {
    let mut hash = 0xcbf29ce484222325u64;
    for ids in [left, right] {
        for id in ids {
            for byte in id.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let action = action.to_le_bytes();
    for byte in action {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash as usize
}

fn split_keeps(
    hash: usize,
    modulus: Option<usize>,
    remainder: usize,
    exclude_matches: bool,
) -> bool {
    let Some(modulus) = modulus else {
        return true;
    };
    let is_match = hash % modulus == remainder;
    if exclude_matches {
        !is_match
    } else {
        is_match
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizationMode {
    Default,
    CodeAware,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerSpec {
    pub version: u32,
    pub mode: String,
    pub normalization: String,
    pub pretokenizer: String,
    #[serde(default)]
    pub code_control_tokens: Vec<String>,
    pub byte_fallback: bool,
    pub reserved_byte_tokens: bool,
    pub subword_identifier_fallback: bool,
    pub byte_token_format: String,
    pub byte_native: bool,
    pub adaptive_boundaries: bool,
    pub boundaryless_bpe: bool,
}

impl TokenizationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::CodeAware => "code-aware",
        }
    }
}

pub const CODE_EOS_TOKEN: &str = "<eos_code>";
pub const CODE_CONTROL_TOKENS: &[&str] = &[
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
    CODE_EOS_TOKEN,
];

const TOKENIZER_SPEC_VERSION: u32 = 9;

pub fn tokenizer_spec(mode: TokenizationMode) -> TokenizerSpec {
    let code_aware = mode == TokenizationMode::CodeAware;
    TokenizerSpec {
        version: TOKENIZER_SPEC_VERSION,
        mode: mode.as_str().to_string(),
        normalization: "identity_utf8".to_string(),
        pretokenizer: if code_aware {
            "code_lexical_v1".to_string()
        } else {
            "lexical_v1".to_string()
        },
        code_control_tokens: if code_aware {
            CODE_CONTROL_TOKENS
                .iter()
                .map(|token| token.to_string())
                .collect()
        } else {
            Vec::new()
        },
        byte_fallback: true,
        reserved_byte_tokens: true,
        subword_identifier_fallback: false,
        byte_token_format: "<byte:XX>".to_string(),
        byte_native: true,
        adaptive_boundaries: true,
        boundaryless_bpe: false,
    }
}

pub fn tokenizer_spec_signature(mode: TokenizationMode) -> String {
    let text =
        serde_json::to_string(&tokenizer_spec(mode)).unwrap_or_else(|_| mode.as_str().into());
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn unescape_pair_field(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn tokenize_text(text: &str) -> Vec<String> {
    text.chars().map(|ch| ch.to_string()).collect()
}

fn code_aware_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let mut chunk = String::new();
        chunk.push(ch);
        if ch.is_ascii_whitespace() {
            while let Some(next) = chars.peek().copied() {
                if !next.is_ascii_whitespace() {
                    break;
                }
                chunk.push(next);
                chars.next();
            }
        } else if ch == '_' || ch.is_ascii_alphabetic() {
            while let Some(next) = chars.peek().copied() {
                if !(next == '_' || next.is_ascii_alphanumeric()) {
                    break;
                }
                chunk.push(next);
                chars.next();
            }
        } else if ch.is_ascii_digit() {
            while let Some(next) = chars.peek().copied() {
                if !(next == '_' || next == '.' || next.is_ascii_alphanumeric()) {
                    break;
                }
                chunk.push(next);
                chars.next();
            }
        } else if ch == '"' || ch == '\'' || ch == '`' {
            let quote = ch;
            let mut escaped = false;
            for next in chars.by_ref() {
                chunk.push(next);
                if quote != '`' && escaped {
                    escaped = false;
                    continue;
                }
                if quote != '`' && next == '\\' {
                    escaped = true;
                    continue;
                }
                if next == quote {
                    break;
                }
            }
        } else if ch.is_ascii_punctuation() {
            while let Some(next) = chars.peek().copied() {
                if !next.is_ascii_punctuation()
                    || next == '_'
                    || next == '"'
                    || next == '\''
                    || next == '`'
                {
                    break;
                }
                chunk.push(next);
                chars.next();
            }
        }
        chunks.push(chunk);
    }
    chunks
}

/// Lexical boundaries keep frequency-based BPE from memorizing a repetitive
/// template (or an entire training example) as one token. Identifiers remain
/// intact chunks so fictional API names can still become single vocabulary
/// entries, while merges cannot cross words, whitespace, or punctuation.
fn default_lexical_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let mut chunk = String::from(ch);
        if ch.is_whitespace() {
            while chars.peek().is_some_and(|next| next.is_whitespace()) {
                chunk.push(chars.next().expect("peeked whitespace"));
            }
        } else if ch == '_' || ch.is_alphabetic() {
            while chars
                .peek()
                .is_some_and(|next| *next == '_' || next.is_alphanumeric())
            {
                chunk.push(chars.next().expect("peeked identifier character"));
            }
        } else if ch.is_numeric() {
            while chars
                .peek()
                .is_some_and(|next| *next == '_' || *next == '.' || next.is_alphanumeric())
            {
                chunk.push(chars.next().expect("peeked numeric character"));
            }
        }
        chunks.push(chunk);
    }
    chunks
}

pub fn tokenize_with_mode(text: &str, mode: TokenizationMode) -> Vec<String> {
    match mode {
        TokenizationMode::Default => tokenize_text(text),
        TokenizationMode::CodeAware => code_aware_chunks(text),
    }
}

pub fn split_line(line: &str) -> Option<Vec<String>> {
    split_line_with_min_tokens(line, 2)
}

/// Like split_line but accepts lines with at least `min_tokens` tokens (e.g. 1 for paragraph mode).
pub fn split_line_with_min_tokens(line: &str, min_tokens: usize) -> Option<Vec<String>> {
    split_line_with_min_tokens_mode(line, min_tokens, TokenizationMode::Default)
}

pub fn split_line_with_min_tokens_mode(
    line: &str,
    min_tokens: usize,
    mode: TokenizationMode,
) -> Option<Vec<String>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let tokens = if let Some((left, right)) = line.split_once("\t") {
        tokenize_with_mode(&format!("{left}\n{right}"), mode)
    } else if let Some((left, right)) = line.split_once("|||") {
        tokenize_with_mode(&format!("{left}\n{right}"), mode)
    } else {
        tokenize_with_mode(line, mode)
    };
    if tokens.len() < min_tokens {
        return None;
    }
    Some(tokens)
}

pub fn encode_line_with_vocab_mode(
    line: &str,
    vocab: &Vocab,
    mode: TokenizationMode,
    min_tokens: usize,
) -> Option<Vec<u32>> {
    let text = extract_text_side_for_vocab(line)?;
    let token_count = tokenize_with_mode(&text, mode).len();
    if token_count < min_tokens {
        return None;
    }
    Some(encode_text_with_vocab_mode(&text, vocab, mode))
}

fn extract_text_side_for_vocab(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some((left, right)) = line.split_once('\t') {
        return Some(format!(
            "{}\n{}",
            unescape_pair_field(left.trim()),
            unescape_pair_field(right.trim())
        ));
    }
    if let Some((left, right)) = line.split_once("|||") {
        return Some(format!(
            "{}\n{}",
            unescape_pair_field(left.trim()),
            unescape_pair_field(right.trim())
        ));
    }
    Some(unescape_pair_field(line))
}

fn flatten_pair_side(text: &str) -> Option<String> {
    let text = unescape_pair_field(text.trim());
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

pub(crate) fn encoder_texts_from_line(line: &str) -> Vec<String> {
    let line = line.trim();
    if line.is_empty() {
        return Vec::new();
    }
    if let Some((left, right)) = line.split_once('\t') {
        return [left, right]
            .into_iter()
            .filter_map(flatten_pair_side)
            .collect();
    }
    if let Some((left, right)) = line.split_once("|||") {
        return [left, right]
            .into_iter()
            .filter_map(flatten_pair_side)
            .collect();
    }
    flatten_pair_side(line).into_iter().collect()
}

fn pair_source_manifest_paths(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut first_line = String::new();
    if reader.read_line(&mut first_line)? == 0 || first_line.trim() != PAIR_SOURCE_MANIFEST_HEADER {
        return Ok(None);
    }

    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut paths = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let direct = PathBuf::from(trimmed);
        if direct.exists() || direct.is_absolute() {
            paths.push(direct);
        } else {
            paths.push(base_dir.join(direct));
        }
    }
    if paths.is_empty() {
        bail!("pair source manifest {:?} contains no input paths", path);
    }
    Ok(Some(paths))
}

pub(crate) fn pair_input_paths(path: &Path) -> Result<(Vec<PathBuf>, bool)> {
    if let Some(paths) = pair_source_manifest_paths(path)? {
        return Ok((paths, true));
    }
    Ok((vec![path.to_path_buf()], false))
}

pub struct VocabStats {
    pub total_tokens: usize,
    pub covered_tokens: usize,
    pub oov_tokens: usize,
    pub unique_tokens: usize,
    pub vocab_size: usize,
}

pub const ACTION_TEXT_REPLY: u32 = 0;
pub const ACTION_CODE: u32 = 1;
pub const ACTION_DONE: u32 = 2;
pub const ACTION_FETCH_DOCS: u32 = 3;
pub const DONE_SENTINEL: &str = "<done>";

fn parse_action_label_str(value: &str) -> Option<u32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "text" | "text_reply" | "explain" | "summarize" => Some(ACTION_TEXT_REPLY),
        "1" | "code" | "inspect_file" | "read_file" | "edit_file" | "apply_patch" | "run_tests"
        | "read_error" | "repair_patch" | "compiler_feedback" => Some(ACTION_CODE),
        "2" | "done" | "final" | "finalize" | "stop" => Some(ACTION_DONE),
        "3" | "fetch_docs" | "docs" | "retrieve_docs" | "doc_lookup" => Some(ACTION_FETCH_DOCS),
        _ => None,
    }
}

fn explicit_action_from_next_text(next_turn: &str) -> Option<(u32, String)> {
    let trimmed = next_turn.trim();
    let rest = trimmed.strip_prefix("<action:")?;
    let (label, tail) = rest.split_once('>')?;
    let action = parse_action_label_str(label)?;
    Some((action, tail.trim().to_string()))
}

fn parse_world_line_fields(line: &str) -> Option<(String, String, Option<u32>)> {
    for delimiter in ['\t', '|'] {
        if delimiter == '|' && !line.contains("|||") {
            continue;
        }
        let parts = if delimiter == '\t' {
            line.split('\t').collect::<Vec<_>>()
        } else {
            line.split("|||").collect::<Vec<_>>()
        };
        if parts.len() < 2 {
            continue;
        }
        let left = unescape_pair_field(parts[0].trim());
        let right = unescape_pair_field(parts[1].trim());
        let action = parts.get(2).and_then(|value| parse_action_label_str(value));
        return Some((left, right, action));
    }
    None
}

/// Heuristic action label for orchestrator training:
/// 0 = TextReply, 1 = Code, 2 = Done, 3 = FetchDocs.
/// Used when building world examples from raw text (next turn string).
pub fn action_label_heuristic(next_turn: &str) -> u32 {
    let s = next_turn.trim();
    if s.is_empty() {
        return ACTION_DONE;
    }
    if s.eq_ignore_ascii_case(DONE_SENTINEL) {
        return ACTION_DONE;
    }
    if let Some((explicit, _stripped)) = explicit_action_from_next_text(s) {
        return explicit;
    }
    if s.contains("```") {
        return ACTION_CODE; // code block
    }
    let lower = s.to_ascii_lowercase();
    if lower.contains("<tool:fetch_docs>") || lower.contains("<action:fetch_docs>") {
        return ACTION_FETCH_DOCS;
    }
    let code_keywords = [
        "const ",
        "var ",
        "func ",
        "package ",
        "package main",
        "type ",
        "struct ",
        "interface ",
        "import ",
        "return ",
        "defer ",
        "go ",
        "range ",
        "select ",
        "chan ",
        "map[",
    ];
    let prose_markers = ["i think", "you can", "here is", "this means", "for example"];
    let mut code_score = 0usize;
    let mut prose_score = 0usize;

    for marker in code_keywords {
        if lower.contains(marker) {
            code_score += 2;
        }
    }
    for marker in prose_markers {
        if lower.contains(marker) {
            prose_score += 1;
        }
    }

    let punctuation = s
        .chars()
        .filter(|c| {
            matches!(
                c,
                '{' | '}' | ';' | '(' | ')' | '[' | ']' | '<' | '>' | '=' | ':' | '/'
            )
        })
        .count();
    if punctuation * 5 > s.chars().count().max(1) {
        code_score += 2;
    }

    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if line.starts_with("    ") || line.starts_with('\t') {
            code_score += 2;
        }
        if trimmed.ends_with('{') || trimmed.contains(":=") || trimmed.contains("func ") {
            code_score += 2;
        }
        if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
            prose_score += 1;
        }
        let word_count = trimmed.split_whitespace().count();
        if word_count >= 8 && !trimmed.contains(":=") && !trimmed.contains("func ") {
            prose_score += 1;
        }
    }

    if code_score >= prose_score + 2 && code_score >= 2 {
        return ACTION_CODE;
    }
    ACTION_TEXT_REPLY
}

#[derive(Clone)]
pub struct WorldExample {
    pub state_tokens: Vec<u32>,
    pub next_tokens: Vec<u32>,
    /// 0 = TextReply, 1 = Code, 2 = Done, 3 = FetchDocs.
    pub action_label: u32,
}

#[derive(Clone)]
pub struct RawWorldExample {
    pub state_text: String,
    pub next_text: String,
    pub action_label: u32,
}

pub fn raw_world_example_from_line_with_mode(
    line: &str,
    mode: TokenizationMode,
) -> Option<RawWorldExample> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let (left, right, explicit_action) = parse_world_line_fields(line)?;
    let state_text = left;
    let mut next_text = right;
    let embedded_action = explicit_action_from_next_text(&next_text);
    let action_label = if let Some(action) = explicit_action {
        if let Some((_embedded_action, stripped)) = embedded_action {
            next_text = stripped;
        }
        action
    } else if let Some((action, stripped)) = embedded_action {
        next_text = stripped;
        action
    } else {
        action_label_heuristic(&next_text)
    };
    let state_tokens = tokenize_with_mode(&state_text, mode);
    let next_tokens = tokenize_with_mode(&next_text, mode);
    if state_tokens.is_empty() || next_tokens.is_empty() {
        return None;
    }
    Some(RawWorldExample {
        state_text,
        next_text,
        action_label,
    })
}

pub fn encode_raw_world_line_with_vocab_mode(
    line: &str,
    vocab: &Vocab,
    mode: TokenizationMode,
) -> Option<WorldExample> {
    let row = raw_world_example_from_line_with_mode(line, mode)?;
    Some(WorldExample {
        state_tokens: encode_text_with_vocab_mode(&row.state_text, vocab, mode),
        next_tokens: encode_text_with_vocab_mode(&row.next_text, vocab, mode),
        action_label: row.action_label,
    })
}

/// Minimum number of tokens per line to include (2 = skip single-token lines; 1 = paragraph mode).
pub const DEFAULT_MIN_TOKENS_PER_LINE: usize = 2;
pub const DEFAULT_STREAM_SHUFFLE_BUFFER: usize = 1024;
const TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_TOKEN_CACHE_V2\n";
const DUAL_TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_DUAL_TOKEN_CACHE_V2\n";
const DEFAULT_TOKEN_CACHE_READER_MB: usize = 32;
const DEFAULT_TOKEN_CACHE_PREFETCH_CHUNKS: usize = 8;
const MAX_TOKEN_CACHE_PREFETCH_CHUNKS: usize = 32;
const DEFAULT_VOCAB_SCAN_PROGRESS_EVERY_ROWS: usize = 500_000;
const DEFAULT_VOCAB_SCAN_CHUNK_LINES: usize = 16_384;
const DEFAULT_BPE_PROGRESS_EVERY_MERGES: usize = 250;
type TokenCacheRecord = (Vec<u32>, Vec<u32>, u32);
type PrefetchRx<T> = Receiver<Result<Vec<T>>>;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

#[derive(Clone, Copy, Debug, Default)]
struct VocabSampleBudget {
    max_rows: Option<usize>,
    max_text_bytes: Option<usize>,
}

impl VocabSampleBudget {
    fn describe(self) -> String {
        let rows = self
            .max_rows
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unlimited".to_string());
        let bytes = self
            .max_text_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unlimited".to_string());
        format!("rows={rows}, text_bytes={bytes}")
    }
}

fn code_vocab_sample_budget() -> VocabSampleBudget {
    VocabSampleBudget {
        max_rows: env_usize("TOFY_CODE_VOCAB_SAMPLE_ROWS").filter(|&value| value > 0),
        max_text_bytes: env_usize("TOFY_CODE_VOCAB_SAMPLE_BYTES").filter(|&value| value > 0),
    }
}

fn encoder_vocab_sample_budget() -> VocabSampleBudget {
    VocabSampleBudget {
        max_rows: env_usize("TOFY_ENCODER_VOCAB_SAMPLE_ROWS").filter(|&value| value > 0),
        max_text_bytes: env_usize("TOFY_ENCODER_VOCAB_SAMPLE_BYTES").filter(|&value| value > 0),
    }
}

fn bpe_progress_every_merges() -> usize {
    env_usize("TOFY_BPE_PROGRESS_EVERY_MERGES")
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_BPE_PROGRESS_EVERY_MERGES)
}

fn bpe_max_merges() -> Option<usize> {
    env_usize("TOFY_BPE_MAX_MERGES").filter(|&value| value > 0)
}

fn token_cache_reader_capacity() -> usize {
    let mb = env_usize("TOFY_TOKEN_CACHE_READER_MB")
        .unwrap_or(DEFAULT_TOKEN_CACHE_READER_MB)
        .clamp(1, 256);
    mb * 1024 * 1024
}

fn token_cache_prefetch_chunks() -> usize {
    env_usize("TOFY_CACHE_PREFETCH_BATCHES")
        .unwrap_or(DEFAULT_TOKEN_CACHE_PREFETCH_CHUNKS)
        .min(MAX_TOKEN_CACHE_PREFETCH_CHUNKS)
}

fn token_cache_prefetch_chunk_size(batch_size: usize) -> usize {
    env_usize("TOFY_CACHE_PREFETCH_CHUNK")
        .filter(|&value| value > 0)
        .unwrap_or(batch_size.max(1))
}

fn token_cache_reader(path: &PathBuf) -> Result<BufReader<File>> {
    Ok(BufReader::with_capacity(
        token_cache_reader_capacity(),
        File::open(path)?,
    ))
}

fn vocab_scan_chunk_lines() -> usize {
    env_usize("TOFY_VOCAB_SCAN_CHUNK_LINES")
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_VOCAB_SCAN_CHUNK_LINES)
}

fn read_line_chunk<I>(lines: &mut I, max_lines: usize) -> Result<Vec<String>>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let mut chunk = Vec::with_capacity(max_lines);
    for _ in 0..max_lines {
        let Some(line) = lines.next() else {
            break;
        };
        chunk.push(line?);
    }
    Ok(chunk)
}

fn recv_prefetched_batch<T>(
    rx: &PrefetchRx<T>,
    stash: &mut VecDeque<T>,
    batch_size: usize,
    label: &str,
) -> Result<Vec<T>> {
    let mut batch = Vec::with_capacity(batch_size);
    while batch.len() < batch_size {
        if let Some(item) = stash.pop_front() {
            batch.push(item);
            continue;
        }
        let chunk = rx
            .recv()
            .map_err(|_| anyhow!("{label} prefetch worker stopped"))??;
        stash.extend(chunk);
    }
    Ok(batch)
}

pub struct PairStream {
    paths: Vec<PathBuf>,
    path_index: usize,
    source_manifest: bool,
    min_tokens: usize,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<Vec<String>>,
    pending_texts: VecDeque<String>,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    row_index: usize,
}

impl PairStream {
    pub fn new(path: &Path, min_tokens: usize) -> Result<Self> {
        Self::with_shuffle(path, min_tokens, DEFAULT_STREAM_SHUFFLE_BUFFER)
    }

    pub fn with_shuffle(
        path: &Path,
        min_tokens: usize,
        shuffle_buffer_size: usize,
    ) -> Result<Self> {
        Self::with_split(path, min_tokens, shuffle_buffer_size, None, 0, false)
    }

    pub fn with_split(
        path: &Path,
        min_tokens: usize,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        let (paths, source_manifest) = pair_input_paths(path)?;
        let reader = BufReader::new(File::open(&paths[0])?);
        Ok(Self {
            paths,
            path_index: 0,
            source_manifest,
            min_tokens,
            reader,
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            pending_texts: VecDeque::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.path_index = (self.path_index + 1) % self.paths.len();
        if self.path_index == 0 {
            self.row_index = 0;
        }
        self.reader = BufReader::new(File::open(&self.paths[self.path_index])?);
        Ok(())
    }

    fn keep_row(&mut self) -> bool {
        let row_idx = self.row_index;
        self.row_index += 1;
        let Some(modulus) = self.split_modulus else {
            return true;
        };
        let is_match = row_idx % modulus == self.split_remainder;
        if self.exclude_split_matches {
            !is_match
        } else {
            is_match
        }
    }

    fn read_next_tokens(&mut self) -> Result<Vec<String>> {
        let mut full_passes_without_row = 0usize;
        loop {
            if let Some(text) = self.pending_texts.pop_front() {
                if let Some(tokens) = split_line_with_min_tokens(&text, self.min_tokens) {
                    if !self.keep_row() {
                        continue;
                    }
                    return Ok(tokens);
                }
            }
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                self.reset()?;
                if self.path_index == 0 {
                    full_passes_without_row += 1;
                    if full_passes_without_row >= 2 {
                        bail!(
                            "pair stream produced no usable rows after two passes (paths={:?}, split_modulus={:?}, split_remainder={}, exclude_matches={})",
                            self.paths,
                            self.split_modulus,
                            self.split_remainder,
                            self.exclude_split_matches
                        );
                    }
                }
                continue;
            }
            if self.source_manifest {
                self.pending_texts.extend(encoder_texts_from_line(&line));
                continue;
            }
            if let Some(tokens) = split_line_with_min_tokens(&line, self.min_tokens) {
                if !self.keep_row() {
                    continue;
                }
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
        let idx = rng().random_range(0..self.shuffle_buffer.len());
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

pub struct CachedPairStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<Pair>,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    row_index: usize,
    prefetch_rx: Option<PrefetchRx<Pair>>,
    prefetch_stash: VecDeque<Pair>,
}

impl CachedPairStream {
    pub fn new(path: &PathBuf) -> Result<Self> {
        Self::with_shuffle(path, DEFAULT_STREAM_SHUFFLE_BUFFER)
    }

    pub fn with_shuffle(path: &PathBuf, shuffle_buffer_size: usize) -> Result<Self> {
        Self::with_split(path, shuffle_buffer_size, None, 0, false)
    }

    pub fn with_split(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        let mut stream = Self {
            path: path.clone(),
            reader: token_cache_reader(path)?,
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
            prefetch_rx: None,
            prefetch_stash: VecDeque::new(),
        };
        stream.read_magic()?;
        Ok(stream)
    }

    fn read_magic(&mut self) -> Result<()> {
        let mut magic = vec![0u8; TOKEN_CACHE_MAGIC.len()];
        self.reader.read_exact(&mut magic)?;
        if magic != TOKEN_CACHE_MAGIC {
            bail!("invalid token cache magic in {:?}", self.path);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = token_cache_reader(&self.path)?;
        self.row_index = 0;
        self.read_magic()
    }

    fn read_next_pair(&mut self) -> Result<Pair> {
        let mut full_passes_without_row = 0usize;
        loop {
            match read_token_cache_record(&mut self.reader)? {
                Some((tokens, _right, _action)) if tokens.len() >= 2 => {
                    let row_idx = self.row_index;
                    self.row_index += 1;
                    if let Some(modulus) = self.split_modulus {
                        let is_match = row_idx % modulus == self.split_remainder;
                        let keep = if self.exclude_split_matches {
                            !is_match
                        } else {
                            is_match
                        };
                        if !keep {
                            continue;
                        }
                    }
                    return Ok(Pair { tokens });
                }
                Some(_) => continue,
                None => {
                    self.reset()?;
                    full_passes_without_row += 1;
                    if full_passes_without_row >= 2 {
                        bail!(
                            "cached pair stream {:?} produced no usable rows after two passes (split_modulus={:?}, split_remainder={}, exclude_matches={})",
                            self.path,
                            self.split_modulus,
                            self.split_remainder,
                            self.exclude_split_matches
                        );
                    }
                }
            }
        }
    }

    fn refill_shuffle_buffer(&mut self) -> Result<()> {
        while self.shuffle_buffer.len() < self.shuffle_buffer_size {
            let pair = self.read_next_pair()?;
            self.shuffle_buffer.push(pair);
        }
        Ok(())
    }

    fn next_pair(&mut self) -> Result<Pair> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_pair();
        }
        self.refill_shuffle_buffer()?;
        let idx = rng().random_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    fn next_batch_direct(&mut self, batch_size: usize) -> Result<Vec<Pair>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_pair()?);
        }
        Ok(batch)
    }

    fn start_prefetch(&mut self, batch_size: usize) -> Result<()> {
        if self.prefetch_rx.is_some() {
            return Ok(());
        }
        let prefetch_chunks = token_cache_prefetch_chunks();
        if prefetch_chunks == 0 {
            return Ok(());
        }
        let path = self.path.clone();
        let shuffle_buffer_size = self.shuffle_buffer_size;
        let split_modulus = self.split_modulus;
        let split_remainder = self.split_remainder;
        let exclude_split_matches = self.exclude_split_matches;
        let chunk_size = token_cache_prefetch_chunk_size(batch_size);
        let (tx, rx) = sync_channel(prefetch_chunks);
        thread::Builder::new()
            .name("tofy-cache-prefetch-pair".to_string())
            .spawn(move || {
                let mut stream = match CachedPairStream::with_split(
                    &path,
                    shuffle_buffer_size,
                    split_modulus,
                    split_remainder,
                    exclude_split_matches,
                ) {
                    Ok(stream) => stream,
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        return;
                    }
                };
                loop {
                    let result = stream.next_batch_direct(chunk_size);
                    let should_continue = result.is_ok();
                    if tx.send(result).is_err() || !should_continue {
                        break;
                    }
                }
            })?;
        println!(
            "Token cache prefetch: encoder chunks={} chunk_size={} reader_mb={}",
            prefetch_chunks,
            chunk_size,
            token_cache_reader_capacity() / (1024 * 1024)
        );
        self.prefetch_rx = Some(rx);
        Ok(())
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<Pair>> {
        self.start_prefetch(batch_size)?;
        if let Some(rx) = &self.prefetch_rx {
            return recv_prefetched_batch(rx, &mut self.prefetch_stash, batch_size, "encoder");
        }
        self.next_batch_direct(batch_size)
    }
}

pub struct RawWorldStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<RawWorldExample>,
    tokenization_mode: TokenizationMode,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    prefetch_rx: Option<PrefetchRx<RawWorldExample>>,
    prefetch_stash: VecDeque<RawWorldExample>,
}

pub struct CachedWorldStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<WorldExample>,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    row_index: usize,
    prefetch_rx: Option<PrefetchRx<WorldExample>>,
    prefetch_stash: VecDeque<WorldExample>,
}

#[derive(Clone)]
pub struct CachedDecoderExample {
    pub encoder: WorldExample,
    pub decoder: WorldExample,
}

pub struct CachedDecoderStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<CachedDecoderExample>,
    split_modulus: Option<usize>,
    split_remainder: usize,
    exclude_split_matches: bool,
    prefetch_rx: Option<PrefetchRx<CachedDecoderExample>>,
    prefetch_stash: VecDeque<CachedDecoderExample>,
}

impl CachedWorldStream {
    pub fn with_split(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        let mut stream = Self {
            path: path.clone(),
            reader: token_cache_reader(path)?,
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
            prefetch_rx: None,
            prefetch_stash: VecDeque::new(),
        };
        stream.read_magic()?;
        Ok(stream)
    }

    fn read_magic(&mut self) -> Result<()> {
        let mut magic = vec![0u8; TOKEN_CACHE_MAGIC.len()];
        self.reader.read_exact(&mut magic)?;
        if magic != TOKEN_CACHE_MAGIC {
            bail!("invalid token cache magic in {:?}", self.path);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = token_cache_reader(&self.path)?;
        self.row_index = 0;
        self.read_magic()
    }

    fn read_next_example(&mut self) -> Result<WorldExample> {
        let mut full_passes_without_row = 0usize;
        loop {
            let Some((state_tokens, next_tokens, action_label)) =
                read_token_cache_record(&mut self.reader)?
            else {
                self.reset()?;
                full_passes_without_row += 1;
                if full_passes_without_row >= 2 {
                    bail!(
                        "cached world stream {:?} produced no usable rows after two passes (split_modulus={:?}, split_remainder={}, exclude_matches={})",
                        self.path,
                        self.split_modulus,
                        self.split_remainder,
                        self.exclude_split_matches
                    );
                }
                continue;
            };
            let row_idx = self.row_index;
            self.row_index += 1;
            if let Some(modulus) = self.split_modulus {
                let is_match = row_idx % modulus == self.split_remainder;
                let keep = if self.exclude_split_matches {
                    !is_match
                } else {
                    is_match
                };
                if !keep {
                    continue;
                }
            }
            if state_tokens.is_empty() || next_tokens.is_empty() {
                continue;
            }
            return Ok(WorldExample {
                state_tokens,
                next_tokens,
                action_label,
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

    fn next_example(&mut self) -> Result<WorldExample> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_example();
        }
        self.refill_shuffle_buffer()?;
        let idx = rng().random_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    fn next_batch_direct(&mut self, batch_size: usize) -> Result<Vec<WorldExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
    }

    fn start_prefetch(&mut self, batch_size: usize) -> Result<()> {
        if self.prefetch_rx.is_some() {
            return Ok(());
        }
        let prefetch_chunks = token_cache_prefetch_chunks();
        if prefetch_chunks == 0 {
            return Ok(());
        }
        let path = self.path.clone();
        let shuffle_buffer_size = self.shuffle_buffer_size;
        let split_modulus = self.split_modulus;
        let split_remainder = self.split_remainder;
        let exclude_split_matches = self.exclude_split_matches;
        let chunk_size = token_cache_prefetch_chunk_size(batch_size);
        let (tx, rx) = sync_channel(prefetch_chunks);
        thread::Builder::new()
            .name("tofy-cache-prefetch-world".to_string())
            .spawn(move || {
                let mut stream = match CachedWorldStream::with_split(
                    &path,
                    shuffle_buffer_size,
                    split_modulus,
                    split_remainder,
                    exclude_split_matches,
                ) {
                    Ok(stream) => stream,
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        return;
                    }
                };
                loop {
                    let result = stream.next_batch_direct(chunk_size);
                    let should_continue = result.is_ok();
                    if tx.send(result).is_err() || !should_continue {
                        break;
                    }
                }
            })?;
        println!(
            "Token cache prefetch: world chunks={} chunk_size={} reader_mb={}",
            prefetch_chunks,
            chunk_size,
            token_cache_reader_capacity() / (1024 * 1024)
        );
        self.prefetch_rx = Some(rx);
        Ok(())
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<WorldExample>> {
        self.start_prefetch(batch_size)?;
        if let Some(rx) = &self.prefetch_rx {
            return recv_prefetched_batch(rx, &mut self.prefetch_stash, batch_size, "world");
        }
        self.next_batch_direct(batch_size)
    }
}

impl CachedDecoderStream {
    pub fn with_split(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        let mut stream = Self {
            path: path.clone(),
            reader: token_cache_reader(path)?,
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            prefetch_rx: None,
            prefetch_stash: VecDeque::new(),
        };
        stream.read_magic()?;
        Ok(stream)
    }

    fn read_magic(&mut self) -> Result<()> {
        let mut magic = vec![0u8; DUAL_TOKEN_CACHE_MAGIC.len()];
        self.reader.read_exact(&mut magic)?;
        if magic != DUAL_TOKEN_CACHE_MAGIC {
            bail!("invalid dual token cache magic in {:?}", self.path);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = token_cache_reader(&self.path)?;
        self.read_magic()
    }

    fn read_next_example(&mut self) -> Result<CachedDecoderExample> {
        let mut full_passes_without_row = 0usize;
        loop {
            let Some((enc_state, enc_next, dec_state, dec_next, action_label)) =
                read_dual_token_cache_record(&mut self.reader)?
            else {
                self.reset()?;
                full_passes_without_row += 1;
                if full_passes_without_row >= 2 {
                    bail!(
                        "cached decoder stream {:?} produced no usable rows after two passes (split_modulus={:?}, split_remainder={}, exclude_matches={})",
                        self.path,
                        self.split_modulus,
                        self.split_remainder,
                        self.exclude_split_matches
                    );
                }
                continue;
            };
            if enc_state.is_empty()
                || enc_next.is_empty()
                || dec_state.is_empty()
                || dec_next.is_empty()
            {
                continue;
            }
            let split_hash = stable_token_pair_split_hash(&dec_state, &dec_next, action_label);
            if !split_keeps(
                split_hash,
                self.split_modulus,
                self.split_remainder,
                self.exclude_split_matches,
            ) {
                continue;
            }
            return Ok(CachedDecoderExample {
                encoder: WorldExample {
                    state_tokens: enc_state,
                    next_tokens: enc_next,
                    action_label,
                },
                decoder: WorldExample {
                    state_tokens: dec_state,
                    next_tokens: dec_next,
                    action_label,
                },
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

    fn next_example(&mut self) -> Result<CachedDecoderExample> {
        if self.shuffle_buffer_size <= 1 {
            return self.read_next_example();
        }
        self.refill_shuffle_buffer()?;
        let idx = rng().random_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    fn next_batch_direct(&mut self, batch_size: usize) -> Result<Vec<CachedDecoderExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
    }

    fn start_prefetch(&mut self, batch_size: usize) -> Result<()> {
        if self.prefetch_rx.is_some() {
            return Ok(());
        }
        let prefetch_chunks = token_cache_prefetch_chunks();
        if prefetch_chunks == 0 {
            return Ok(());
        }
        let path = self.path.clone();
        let shuffle_buffer_size = self.shuffle_buffer_size;
        let split_modulus = self.split_modulus;
        let split_remainder = self.split_remainder;
        let exclude_split_matches = self.exclude_split_matches;
        let chunk_size = token_cache_prefetch_chunk_size(batch_size);
        let (tx, rx) = sync_channel(prefetch_chunks);
        thread::Builder::new()
            .name("tofy-cache-prefetch-decoder".to_string())
            .spawn(move || {
                let mut stream = match CachedDecoderStream::with_split(
                    &path,
                    shuffle_buffer_size,
                    split_modulus,
                    split_remainder,
                    exclude_split_matches,
                ) {
                    Ok(stream) => stream,
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        return;
                    }
                };
                loop {
                    let result = stream.next_batch_direct(chunk_size);
                    let should_continue = result.is_ok();
                    if tx.send(result).is_err() || !should_continue {
                        break;
                    }
                }
            })?;
        println!(
            "Token cache prefetch: decoder chunks={} chunk_size={} reader_mb={}",
            prefetch_chunks,
            chunk_size,
            token_cache_reader_capacity() / (1024 * 1024)
        );
        self.prefetch_rx = Some(rx);
        Ok(())
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<CachedDecoderExample>> {
        self.start_prefetch(batch_size)?;
        if let Some(rx) = &self.prefetch_rx {
            return recv_prefetched_batch(rx, &mut self.prefetch_stash, batch_size, "decoder");
        }
        self.next_batch_direct(batch_size)
    }
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<Option<u32>> {
    let mut buf = [0u8; 4];
    let mut read = 0usize;
    while read < buf.len() {
        let n = reader.read(&mut buf[read..])?;
        if n == 0 {
            if read == 0 {
                return Ok(None);
            }
            bail!("truncated token cache record");
        }
        read += n;
    }
    Ok(Some(u32::from_le_bytes(buf)))
}

fn read_ids<R: Read>(reader: &mut R) -> Result<Option<Vec<u32>>> {
    let Some(len) = read_u32_le(reader)? else {
        return Ok(None);
    };
    let len = len as usize;
    if len == 0 {
        return Ok(Some(Vec::new()));
    }
    let byte_len = len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow!("token cache id sequence too large: {len} ids"))?;
    let mut ids = vec![0u32; len];
    // The cache stores raw little-endian u32 ids; read directly into the output
    // allocation to avoid a temporary Vec<u8> for every sequence.
    let bytes = unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr().cast::<u8>(), byte_len) };
    if let Err(err) = reader.read_exact(bytes) {
        bail!("truncated token cache id sequence: {err}");
    }
    if cfg!(target_endian = "big") {
        for id in &mut ids {
            *id = u32::from_le(*id);
        }
    }
    Ok(Some(ids))
}

fn read_token_cache_record<R: Read>(reader: &mut R) -> Result<Option<TokenCacheRecord>> {
    let Some(left) = read_ids(reader)? else {
        return Ok(None);
    };
    let Some(right) = read_ids(reader)? else {
        bail!("truncated token cache right sequence");
    };
    let Some(action) = read_u32_le(reader)? else {
        bail!("truncated token cache action");
    };
    Ok(Some((left, right, action)))
}

type DualTokenCacheRecord = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32);

fn read_dual_token_cache_record<R: Read>(reader: &mut R) -> Result<Option<DualTokenCacheRecord>> {
    let Some(enc_left) = read_ids(reader)? else {
        return Ok(None);
    };
    let Some(enc_right) = read_ids(reader)? else {
        bail!("truncated dual token cache encoder right sequence");
    };
    let Some(dec_left) = read_ids(reader)? else {
        bail!("truncated dual token cache decoder left sequence");
    };
    let Some(dec_right) = read_ids(reader)? else {
        bail!("truncated dual token cache decoder right sequence");
    };
    let Some(action) = read_u32_le(reader)? else {
        bail!("truncated dual token cache action");
    };
    Ok(Some((enc_left, enc_right, dec_left, dec_right, action)))
}

impl RawWorldStream {
    pub fn new(path: &PathBuf) -> Result<Self> {
        Self::with_shuffle_mode(
            path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            TokenizationMode::Default,
        )
    }

    pub fn with_shuffle_mode(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        tokenization_mode: TokenizationMode,
    ) -> Result<Self> {
        Self::with_split_mode(path, shuffle_buffer_size, tokenization_mode, None, 0, false)
    }

    pub fn with_split(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        Self::with_split_mode(
            path,
            shuffle_buffer_size,
            TokenizationMode::Default,
            split_modulus,
            split_remainder,
            exclude_split_matches,
        )
    }

    pub fn with_split_mode(
        path: &PathBuf,
        shuffle_buffer_size: usize,
        tokenization_mode: TokenizationMode,
        split_modulus: Option<usize>,
        split_remainder: usize,
        exclude_split_matches: bool,
    ) -> Result<Self> {
        Ok(Self {
            path: path.clone(),
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            tokenization_mode,
            split_modulus,
            split_remainder,
            exclude_split_matches,
            prefetch_rx: None,
            prefetch_stash: VecDeque::new(),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = BufReader::new(File::open(&self.path)?);
        Ok(())
    }

    fn read_next_example(&mut self) -> Result<RawWorldExample> {
        let mut full_passes_without_row = 0usize;
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                self.reset()?;
                full_passes_without_row += 1;
                if full_passes_without_row >= 2 {
                    bail!(
                        "raw world stream {:?} produced no usable rows after two passes (split_modulus={:?}, split_remainder={}, exclude_matches={})",
                        self.path,
                        self.split_modulus,
                        self.split_remainder,
                        self.exclude_split_matches
                    );
                }
                continue;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((left, right, explicit_action)) = parse_world_line_fields(line) else {
                continue;
            };
            let state_text = left;
            let mut next_text = right;
            let embedded_action = explicit_action_from_next_text(&next_text);
            let action_label = if let Some(action) = explicit_action {
                if let Some((_embedded_action, stripped)) = embedded_action {
                    next_text = stripped;
                }
                action
            } else if let Some((action, stripped)) = embedded_action {
                next_text = stripped;
                action
            } else {
                action_label_heuristic(&next_text)
            };
            let action_bytes = action_label.to_le_bytes();
            let split_hash = stable_split_hash_parts([
                state_text.as_bytes(),
                next_text.as_bytes(),
                action_bytes.as_slice(),
            ]);
            if !split_keeps(
                split_hash,
                self.split_modulus,
                self.split_remainder,
                self.exclude_split_matches,
            ) {
                continue;
            }
            let state_tokens = tokenize_with_mode(&state_text, self.tokenization_mode);
            let next_tokens = tokenize_with_mode(&next_text, self.tokenization_mode);
            if state_tokens.is_empty() || next_tokens.is_empty() {
                continue;
            }
            return Ok(RawWorldExample {
                state_text,
                next_text,
                action_label,
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
        let idx = rng().random_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    fn next_batch_direct(&mut self, batch_size: usize) -> Result<Vec<RawWorldExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
    }

    fn start_prefetch(&mut self, batch_size: usize) -> Result<()> {
        if self.prefetch_rx.is_some() {
            return Ok(());
        }
        let prefetch_chunks = token_cache_prefetch_chunks();
        if prefetch_chunks == 0 {
            return Ok(());
        }
        let path = self.path.clone();
        let shuffle_buffer_size = self.shuffle_buffer_size;
        let tokenization_mode = self.tokenization_mode;
        let split_modulus = self.split_modulus;
        let split_remainder = self.split_remainder;
        let exclude_split_matches = self.exclude_split_matches;
        let chunk_size = token_cache_prefetch_chunk_size(batch_size);
        let (tx, rx) = sync_channel(prefetch_chunks);
        thread::Builder::new()
            .name("tofy-raw-prefetch-world".to_string())
            .spawn(move || {
                let mut stream = match RawWorldStream::with_split_mode(
                    &path,
                    shuffle_buffer_size,
                    tokenization_mode,
                    split_modulus,
                    split_remainder,
                    exclude_split_matches,
                ) {
                    Ok(stream) => stream,
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        return;
                    }
                };
                loop {
                    let result = stream.next_batch_direct(chunk_size);
                    let should_continue = result.is_ok();
                    if tx.send(result).is_err() || !should_continue {
                        break;
                    }
                }
            })?;
        println!(
            "Raw world prefetch: chunks={} chunk_size={}",
            prefetch_chunks, chunk_size
        );
        self.prefetch_rx = Some(rx);
        Ok(())
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<RawWorldExample>> {
        self.start_prefetch(batch_size)?;
        if let Some(rx) = &self.prefetch_rx {
            return recv_prefetched_batch(rx, &mut self.prefetch_stash, batch_size, "raw-world");
        }
        self.next_batch_direct(batch_size)
    }
}

pub fn build_vocab_from_pair_file(
    path: &PathBuf,
    max_vocab: usize,
    min_tokens_per_line: Option<usize>,
) -> Result<(Vocab, VocabStats, usize)> {
    const PROGRESS_EVERY_LINES: usize = 500_000;

    let min_tok = min_tokens_per_line.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE);
    let mut pair_count = 0usize;
    let mut raw_line_count = 0usize;
    let mut sampled_text_bytes = 0usize;
    let mut texts = Vec::new();
    let budget = encoder_vocab_sample_budget();
    println!(
        "encoder vocab sampling budget for {}: {}",
        path.display(),
        budget.describe()
    );
    let chunk_lines = vocab_scan_chunk_lines();
    println!(
        "encoder vocab parallel scan for {}: chunk_lines={} rayon_threads={}",
        path.display(),
        chunk_lines,
        rayon::current_num_threads()
    );

    let (paths, source_manifest) = pair_input_paths(path)?;
    'sources: for input_path in &paths {
        let mut lines = BufReader::new(File::open(input_path)?).lines();
        loop {
            let chunk = read_line_chunk(&mut lines, chunk_lines)?;
            if chunk.is_empty() {
                break;
            }
            raw_line_count += chunk.len();
            let chunk_texts = chunk
                .par_iter()
                .flat_map_iter(|line| {
                    let line_texts = if source_manifest {
                        encoder_texts_from_line(line)
                    } else {
                        extract_text_side_for_vocab(line).into_iter().collect()
                    };
                    line_texts
                        .into_iter()
                        .filter(move |text| split_line_with_min_tokens(text, min_tok).is_some())
                })
                .collect::<Vec<_>>();
            for text in chunk_texts {
                let text_len = text.len();
                if let Some(limit) = budget.max_rows {
                    if pair_count >= limit {
                        println!(
                            "encoder vocab row budget reached for {}: kept {pair_count} usable sequences",
                            path.display()
                        );
                        break 'sources;
                    }
                }
                if let Some(limit) = budget.max_text_bytes {
                    if pair_count > 0 && sampled_text_bytes.saturating_add(text_len) > limit {
                        println!(
                            "encoder vocab byte budget reached for {}: kept {pair_count} usable sequences and {} text bytes",
                            path.display(),
                            sampled_text_bytes
                        );
                        break 'sources;
                    }
                }
                sampled_text_bytes = sampled_text_bytes.saturating_add(text_len);
                texts.push(text);
                pair_count += 1;
            }
            if raw_line_count / PROGRESS_EVERY_LINES
                != raw_line_count.saturating_sub(chunk_lines) / PROGRESS_EVERY_LINES
            {
                println!(
                    "Vocab scan progress: {} raw lines read, {} usable sequences kept...",
                    raw_line_count, pair_count
                );
            }
        }
    }

    if pair_count == 0 {
        bail!("no usable lines found in {:?}", path);
    }

    let (vocab, total_tokens) = train_default_bpe_from_texts(&texts, max_vocab)?;
    let stats = VocabStats {
        total_tokens,
        covered_tokens: total_tokens,
        oov_tokens: 0,
        unique_tokens: vocab.id_to_token.len(),
        vocab_size: vocab.id_to_token.len(),
    };

    Ok((vocab, stats, pair_count))
}

pub fn count_pairs_with_vocab(path: &PathBuf) -> Result<usize> {
    let (paths, source_manifest) = pair_input_paths(path)?;
    let mut count = 0usize;
    for input_path in &paths {
        let reader = BufReader::new(File::open(input_path)?);
        for line in reader.lines() {
            let line = line?;
            if source_manifest {
                count += encoder_texts_from_line(&line)
                    .into_iter()
                    .filter(|text| split_line(text).is_some())
                    .count();
            } else if split_line(&line).is_some() {
                count += 1;
            }
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

pub struct CurriculumDenoisingConfig {
    pub max_seq: usize,
    pub active_seq: usize,
    pub max_spans_per_sample: usize,
    pub max_span_len: usize,
    pub min_masked_ratio: f64,
    pub max_masked_ratio: f64,
    pub code_span_multiplier: f64,
    pub identifier_focus_prob: f64,
    pub block_focus_prob: f64,
    pub comment_focus_prob: f64,
    pub text_boundary_focus_prob: f64,
    pub code_masked_ratio_multiplier: f64,
    pub context_segments: usize,
    pub recent_full_segments: usize,
    pub history_ratio: f64,
}

pub struct AugmentedJepaBatch {
    pub view_a_ids: Tensor,
    pub view_b_ids: Tensor,
    pub target_ids: Tensor,
    pub target_linear_indices: Tensor,
    /// Host copy of `target_linear_indices` (flattened `row * max_seq + pos`).
    pub target_linear_host: Vec<u32>,
    /// Valid (non-pad) token count per row, before padding to `max_seq`.
    pub valid_lens: Vec<usize>,
    pub target_count: usize,
    pub code_fraction: f32,
}

fn is_identifier_like(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_code_like_tokens(tokens: &[String]) -> bool {
    let code_chars = tokens
        .iter()
        .filter(|tok| {
            matches!(
                tok.as_str(),
                "{" | "}"
                    | "("
                    | ")"
                    | "["
                    | "]"
                    | ";"
                    | "::"
                    | "."
                    | "=>"
                    | "->"
                    | "="
                    | "=="
                    | "+"
                    | "-"
                    | "*"
                    | "/"
            )
        })
        .count();
    let identifiers = tokens.iter().filter(|tok| is_identifier_like(tok)).count();
    let len = tokens.len().max(1);
    code_chars * 5 > len || (code_chars > 4 && identifiers > len / 4)
}

fn token_str(vocab: &Vocab, id: u32) -> &str {
    vocab
        .id_to_token
        .get(id as usize)
        .map(String::as_str)
        .unwrap_or("<unk>")
}

fn is_code_like_ids(ids: &[u32], vocab: &Vocab) -> bool {
    let code_chars = ids
        .iter()
        .filter(|&&id| {
            matches!(
                token_str(vocab, id),
                "{" | "}"
                    | "("
                    | ")"
                    | "["
                    | "]"
                    | ";"
                    | "::"
                    | "."
                    | "=>"
                    | "->"
                    | "="
                    | "=="
                    | "+"
                    | "-"
                    | "*"
                    | "/"
            )
        })
        .count();
    let identifiers = ids
        .iter()
        .filter(|&&id| is_identifier_like(token_str(vocab, id)))
        .count();
    let len = ids.len().max(1);
    code_chars * 5 > len || (code_chars > 4 && identifiers > len / 4)
}

fn is_comment_token(token: &str) -> bool {
    matches!(token, "//" | "/*" | "*/" | "#" | "\"\"\"" | "'''")
}

fn is_block_token(token: &str) -> bool {
    matches!(
        token,
        "{" | "}" | "(" | ")" | "[" | "]" | ":" | "::" | "=>" | "->"
    )
}

fn is_text_boundary_token(token: &str) -> bool {
    matches!(token, "." | "!" | "?" | ":" | ";" | "," | "\n")
}

fn crop_tokens_for_curriculum(
    tokens: &[String],
    active_seq: usize,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<String> {
    if tokens.len() <= active_seq {
        return tokens.to_vec();
    }
    let start = rng.random_range(0..=tokens.len() - active_seq);
    tokens[start..start + active_seq].to_vec()
}

fn sample_even_history_tokens(
    tokens: &[String],
    budget: usize,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<String> {
    if budget == 0 || tokens.is_empty() {
        return Vec::new();
    }
    if tokens.len() <= budget {
        return tokens.to_vec();
    }
    let stride = tokens.len() as f64 / budget as f64;
    let jitter = stride.min(1.0);
    let mut out = Vec::with_capacity(budget);
    for idx in 0..budget {
        let base = (idx as f64 * stride).floor() as usize;
        let jitter_offset = if jitter > 0.0 {
            rng.random_range(0.0..=jitter).floor() as usize
        } else {
            0
        };
        let chosen = (base + jitter_offset).min(tokens.len().saturating_sub(1));
        out.push(tokens[chosen].clone());
    }
    out
}

fn prepare_segmented_context_tokens(
    tokens: &[String],
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<String> {
    let context_segments = cfg.context_segments.max(1);
    let recent_full_segments = cfg.recent_full_segments.min(context_segments).max(1);
    let source_budget = cfg.active_seq.max(1) * context_segments;
    let mut cropped = crop_tokens_for_curriculum(tokens, source_budget, rng);
    if context_segments == 1 || cropped.len() <= cfg.max_seq {
        return crop_tokens_for_curriculum(&cropped, cfg.active_seq.max(1).min(cfg.max_seq), rng);
    }

    let segment_len = cfg.active_seq.max(1);
    if cropped.len() > source_budget {
        cropped = cropped[cropped.len() - source_budget..].to_vec();
    }
    let recent_span = segment_len * recent_full_segments;
    let history_len = cropped.len().saturating_sub(recent_span);
    let history_budget = ((cfg.max_seq as f64) * cfg.history_ratio).round().max(0.0) as usize;
    let history_budget = history_budget.min(cfg.max_seq / 2).min(history_len);
    let recent_budget = cfg.max_seq.saturating_sub(history_budget).max(1);

    let history_tokens = if history_len > 0 {
        sample_even_history_tokens(&cropped[..history_len], history_budget, rng)
    } else {
        Vec::new()
    };
    let recent_tokens_all = &cropped[history_len..];
    let recent_tokens = if recent_tokens_all.len() > recent_budget {
        recent_tokens_all[recent_tokens_all.len() - recent_budget..].to_vec()
    } else {
        recent_tokens_all.to_vec()
    };

    let mut combined = Vec::with_capacity(history_tokens.len() + recent_tokens.len());
    combined.extend(history_tokens);
    combined.extend(recent_tokens);
    if combined.len() > cfg.max_seq {
        combined = combined[combined.len() - cfg.max_seq..].to_vec();
    }
    combined
}

fn crop_ids_for_curriculum(
    ids: &[u32],
    active_seq: usize,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<u32> {
    if ids.len() <= active_seq {
        return ids.to_vec();
    }
    let start = rng.random_range(0..=ids.len() - active_seq);
    ids[start..start + active_seq].to_vec()
}

fn sample_even_history_ids(
    ids: &[u32],
    budget: usize,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<u32> {
    if budget == 0 || ids.is_empty() {
        return Vec::new();
    }
    if ids.len() <= budget {
        return ids.to_vec();
    }
    let stride = ids.len() as f64 / budget as f64;
    let jitter = stride.min(1.0);
    let mut out = Vec::with_capacity(budget);
    for idx in 0..budget {
        let base = (idx as f64 * stride).floor() as usize;
        let jitter_offset = if jitter > 0.0 {
            rng.random_range(0.0..=jitter).floor() as usize
        } else {
            0
        };
        let chosen = (base + jitter_offset).min(ids.len().saturating_sub(1));
        out.push(ids[chosen]);
    }
    out
}

fn prepare_segmented_context_ids(
    ids: &[u32],
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<u32> {
    let context_segments = cfg.context_segments.max(1);
    let recent_full_segments = cfg.recent_full_segments.min(context_segments).max(1);
    let source_budget = cfg.active_seq.max(1) * context_segments;
    let mut cropped = crop_ids_for_curriculum(ids, source_budget, rng);
    if context_segments == 1 || cropped.len() <= cfg.max_seq {
        return crop_ids_for_curriculum(&cropped, cfg.active_seq.max(1).min(cfg.max_seq), rng);
    }

    let segment_len = cfg.active_seq.max(1);
    if cropped.len() > source_budget {
        cropped = cropped[cropped.len() - source_budget..].to_vec();
    }
    let recent_span = segment_len * recent_full_segments;
    let history_len = cropped.len().saturating_sub(recent_span);
    let history_budget = ((cfg.max_seq as f64) * cfg.history_ratio).round().max(0.0) as usize;
    let history_budget = history_budget.min(cfg.max_seq / 2).min(history_len);
    let recent_budget = cfg.max_seq.saturating_sub(history_budget).max(1);

    let history_ids = if history_len > 0 {
        sample_even_history_ids(&cropped[..history_len], history_budget, rng)
    } else {
        Vec::new()
    };
    let recent_ids_all = &cropped[history_len..];
    let recent_ids = if recent_ids_all.len() > recent_budget {
        recent_ids_all[recent_ids_all.len() - recent_budget..].to_vec()
    } else {
        recent_ids_all.to_vec()
    };

    let mut combined = Vec::with_capacity(history_ids.len() + recent_ids.len());
    combined.extend(history_ids);
    combined.extend(recent_ids);
    if combined.len() > cfg.max_seq {
        combined = combined[combined.len() - cfg.max_seq..].to_vec();
    }
    combined
}

struct MaskingPools {
    valid_len: usize,
    code_like: bool,
    valid_positions: Vec<usize>,
    identifier_positions: Vec<usize>,
    comment_positions: Vec<usize>,
    block_positions: Vec<usize>,
    text_boundary_positions: Vec<usize>,
}

fn masking_pools(token_strs: &[&str], code_like: bool) -> MaskingPools {
    let valid_len = token_strs.len().max(1);
    MaskingPools {
        valid_len,
        code_like,
        valid_positions: (0..valid_len).collect(),
        identifier_positions: token_strs
            .iter()
            .enumerate()
            .filter_map(|(idx, tok)| is_identifier_like(tok).then_some(idx))
            .collect(),
        comment_positions: token_strs
            .iter()
            .enumerate()
            .filter_map(|(idx, tok)| is_comment_token(tok).then_some(idx))
            .collect(),
        block_positions: token_strs
            .iter()
            .enumerate()
            .filter_map(|(idx, tok)| is_block_token(tok).then_some(idx))
            .collect(),
        text_boundary_positions: token_strs
            .iter()
            .enumerate()
            .filter_map(|(idx, tok)| is_text_boundary_token(tok).then_some(idx))
            .collect(),
    }
}

fn select_masked_positions(
    pools: &MaskingPools,
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> Vec<usize> {
    let MaskingPools {
        valid_len,
        code_like,
        valid_positions,
        identifier_positions,
        comment_positions,
        block_positions,
        text_boundary_positions,
    } = pools;
    let valid_len = *valid_len;
    let code_like = *code_like;
    let mut selected: HashSet<usize> = HashSet::new();

    let ratio_multiplier = if code_like {
        cfg.code_masked_ratio_multiplier.max(1.0)
    } else {
        1.0
    };
    let min_ratio = (cfg.min_masked_ratio * ratio_multiplier).clamp(0.08, 0.9);
    let max_ratio = (cfg.max_masked_ratio * ratio_multiplier).clamp(min_ratio, 0.9);
    // Prediction requires at least one visible token. Capping at len - 1 is
    // essential for short records; otherwise ceil(ratio * len) can mask the
    // complete sequence even when the input passed the two-token cache gate.
    let max_predictable = valid_positions.len().saturating_sub(1);
    if max_predictable == 0 {
        return Vec::new();
    }
    let min_masked = (valid_positions.len() as f64 * min_ratio).ceil() as usize;
    let min_masked = min_masked.max(1).min(max_predictable);
    let max_masked = (valid_positions.len() as f64 * max_ratio).ceil() as usize;
    let max_masked = max_masked.max(min_masked).min(max_predictable);
    let target_masked = if min_masked >= max_masked {
        max_masked
    } else {
        rng.random_range(min_masked..=max_masked)
    };
    let span_count = rng.random_range(1..=cfg.max_spans_per_sample.max(1));
    let mut span_len_cap = cfg.max_span_len.max(1);
    if code_like {
        span_len_cap = ((span_len_cap as f64) * cfg.code_span_multiplier).round() as usize;
        span_len_cap = span_len_cap.max(1);
    }

    for span_idx in 0..span_count {
        if selected.len() >= target_masked {
            break;
        }
        let use_identifier_focus = code_like
            && !identifier_positions.is_empty()
            && (span_idx == 0 || rng.random_bool(cfg.identifier_focus_prob.clamp(0.0, 1.0)));
        let use_comment_focus = !comment_positions.is_empty()
            && rng.random_bool(cfg.comment_focus_prob.clamp(0.0, 1.0));
        let use_block_focus = code_like
            && !block_positions.is_empty()
            && rng.random_bool(cfg.block_focus_prob.clamp(0.0, 1.0));
        let use_text_boundary_focus = !code_like
            && !text_boundary_positions.is_empty()
            && rng.random_bool(cfg.text_boundary_focus_prob.clamp(0.0, 1.0));
        let start_pool = if use_comment_focus {
            comment_positions
        } else if use_block_focus {
            block_positions
        } else if use_text_boundary_focus {
            text_boundary_positions
        } else if use_identifier_focus {
            identifier_positions
        } else {
            valid_positions
        };
        let Some(&start) = start_pool.choose(rng) else {
            continue;
        };
        let span_len = if use_comment_focus {
            rng.random_range(2..=span_len_cap.clamp(2, 12))
        } else if use_block_focus {
            rng.random_range(2..=span_len_cap.clamp(2, 10))
        } else if use_text_boundary_focus {
            rng.random_range(3..=span_len_cap.clamp(3, 12))
        } else if use_identifier_focus {
            rng.random_range(1..=span_len_cap.min(4))
        } else {
            rng.random_range(1..=span_len_cap)
        };
        for p in start..(start + span_len).min(valid_len) {
            if selected.len() >= target_masked {
                break;
            }
            selected.insert(p);
        }
    }

    while selected.len() < target_masked {
        let Some(&start) = valid_positions.choose(rng) else {
            break;
        };
        let span_len = rng.random_range(
            1..=span_len_cap.min(target_masked.saturating_sub(selected.len()).max(1)),
        );
        for p in start..(start + span_len).min(valid_len) {
            if selected.len() >= target_masked {
                break;
            }
            selected.insert(p);
        }
    }

    if selected.is_empty() {
        if let Some(&fallback) = valid_positions.choose(rng) {
            selected.insert(fallback);
        }
    }

    let mut selected_positions: Vec<usize> = selected.into_iter().collect();
    selected_positions.sort_unstable();
    selected_positions
}

fn apply_mask(target_ids: &[u32], positions: &[usize], mask_id: u32) -> Vec<u32> {
    let mut context_ids = target_ids.to_vec();
    for &p in positions {
        context_ids[p] = mask_id;
    }
    context_ids
}

struct MaskedViewPair {
    view_a: Vec<u32>,
    view_b: Vec<u32>,
    target: Vec<u32>,
    positions_a: Vec<usize>,
    valid_len: usize,
    code_like: bool,
}

/// Build two masked views over ONE shared crop/segment sample so that the
/// chunk/global multi-view losses compare representations of the same text
/// span.
fn build_masked_view_pair(
    tokens: &[String],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> MaskedViewPair {
    let prepared_tokens = prepare_segmented_context_tokens(tokens, cfg, rng);
    let code_like = is_code_like_tokens(&prepared_tokens);
    let mut target_ids = vocab.encode(&prepared_tokens);
    let valid_len = target_ids.len().min(cfg.max_seq);
    pad_or_truncate(&mut target_ids, cfg.max_seq, vocab.pad_id);
    let token_strs: Vec<&str> = prepared_tokens.iter().map(String::as_str).collect();
    let pools = masking_pools(&token_strs, code_like);
    let positions_a = select_masked_positions(&pools, cfg, rng);
    let positions_b = select_masked_positions(&pools, cfg, rng);
    let view_a = apply_mask(&target_ids, &positions_a, vocab.mask_id);
    let view_b = apply_mask(&target_ids, &positions_b, vocab.mask_id);
    MaskedViewPair {
        view_a,
        view_b,
        target: target_ids,
        positions_a,
        valid_len,
        code_like,
    }
}

fn build_masked_view_pair_from_ids(
    ids: &[u32],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> MaskedViewPair {
    let prepared_ids = prepare_segmented_context_ids(ids, cfg, rng);
    let code_like = is_code_like_ids(&prepared_ids, vocab);
    let mut target_ids = prepared_ids.clone();
    let valid_len = target_ids.len().min(cfg.max_seq);
    pad_or_truncate(&mut target_ids, cfg.max_seq, vocab.pad_id);
    let token_strs: Vec<&str> = prepared_ids
        .iter()
        .map(|&id| token_str(vocab, id))
        .collect();
    let pools = masking_pools(&token_strs, code_like);
    let positions_a = select_masked_positions(&pools, cfg, rng);
    let positions_b = select_masked_positions(&pools, cfg, rng);
    let view_a = apply_mask(&target_ids, &positions_a, vocab.mask_id);
    let view_b = apply_mask(&target_ids, &positions_b, vocab.mask_id);
    MaskedViewPair {
        view_a,
        view_b,
        target: target_ids,
        positions_a,
        valid_len,
        code_like,
    }
}

pub fn make_augmented_jepa_batch(
    token_batches: &[Vec<String>],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    device: &Device,
) -> Result<AugmentedJepaBatch> {
    let batch_size = token_batches.len();
    let rows = token_batches
        .par_iter()
        .enumerate()
        .map(|(b, tokens)| {
            let mut rng = rng();
            let pair = build_masked_view_pair(tokens, vocab, cfg, &mut rng);
            let target_linear = pair
                .positions_a
                .iter()
                .map(|&p| (b * cfg.max_seq + p) as u32)
                .collect::<Vec<_>>();
            (pair, target_linear)
        })
        .collect::<Vec<_>>();
    if rows.iter().any(|(pair, _)| pair.valid_len < 2) {
        bail!("JEPA batch contains a sequence shorter than two encoded tokens");
    }

    let mut view_a_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut view_b_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_linear = Vec::new();
    let mut valid_lens = Vec::with_capacity(batch_size);
    let mut code_like_count = 0usize;
    for (pair, row_target_linear) in rows {
        if pair.code_like {
            code_like_count += 1;
        }
        target_linear.extend(row_target_linear);
        view_a_buf.extend(pair.view_a);
        view_b_buf.extend(pair.view_b);
        target_buf.extend(pair.target);
        valid_lens.push(pair.valid_len);
    }

    let view_a_ids = Tensor::from_vec(view_a_buf, (batch_size, cfg.max_seq), device)?;
    let view_b_ids = Tensor::from_vec(view_b_buf, (batch_size, cfg.max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, cfg.max_seq), device)?;
    let target_count = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear.clone(), (target_count,), device)?;

    Ok(AugmentedJepaBatch {
        view_a_ids,
        view_b_ids,
        target_ids,
        target_linear_indices,
        target_linear_host: target_linear,
        valid_lens,
        target_count,
        code_fraction: code_like_count as f32 / batch_size.max(1) as f32,
    })
}

pub fn make_augmented_jepa_batch_from_pairs(
    pairs: &[Pair],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    device: &Device,
) -> Result<AugmentedJepaBatch> {
    let batch_size = pairs.len();
    let rows = pairs
        .par_iter()
        .enumerate()
        .map(|(b, pair)| {
            let mut rng = rng();
            let view_pair = build_masked_view_pair_from_ids(&pair.tokens, vocab, cfg, &mut rng);
            let target_linear = view_pair
                .positions_a
                .iter()
                .map(|&p| (b * cfg.max_seq + p) as u32)
                .collect::<Vec<_>>();
            (view_pair, target_linear)
        })
        .collect::<Vec<_>>();
    if rows.iter().any(|(pair, _)| pair.valid_len < 2) {
        bail!("JEPA batch contains a sequence shorter than two encoded tokens");
    }

    let mut view_a_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut view_b_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_linear = Vec::new();
    let mut valid_lens = Vec::with_capacity(batch_size);
    let mut code_like_count = 0usize;
    for (view_pair, row_target_linear) in rows {
        if view_pair.code_like {
            code_like_count += 1;
        }
        target_linear.extend(row_target_linear);
        view_a_buf.extend(view_pair.view_a);
        view_b_buf.extend(view_pair.view_b);
        target_buf.extend(view_pair.target);
        valid_lens.push(view_pair.valid_len);
    }

    let view_a_ids = Tensor::from_vec(view_a_buf, (batch_size, cfg.max_seq), device)?;
    let view_b_ids = Tensor::from_vec(view_b_buf, (batch_size, cfg.max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, cfg.max_seq), device)?;
    let target_count = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear.clone(), (target_count,), device)?;

    Ok(AugmentedJepaBatch {
        view_a_ids,
        view_b_ids,
        target_ids,
        target_linear_indices,
        target_linear_host: target_linear,
        valid_lens,
        target_count,
        code_fraction: code_like_count as f32 / batch_size.max(1) as f32,
    })
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
#[allow(clippy::too_many_arguments)]
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
    let batch_size = pairs.len();
    let span_count_cap = max_spans_per_sample.max(1);
    let span_len_cap = max_span_len.max(1);
    let ratio = max_masked_ratio.clamp(0.01, 1.0);
    let rows = pairs
        .par_iter()
        .enumerate()
        .map(|(b, pair)| {
            let mut rng = rng();
            let mut seq = pair.tokens.clone();
            if seq.len() > max_seq {
                let start = seq.len() - max_seq;
                seq.drain(..start);
            }
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

                let span_count = rng.random_range(1..=span_count_cap);
                for _ in 0..span_count {
                    if selected.len() >= max_masked {
                        break;
                    }
                    let Some(&start) = valid_positions.choose(&mut rng) else {
                        break;
                    };
                    let span_len = rng.random_range(1..=span_len_cap);
                    for (p, token) in target_seq
                        .iter()
                        .enumerate()
                        .take((start + span_len).min(max_seq))
                        .skip(start)
                    {
                        if selected.len() >= max_masked {
                            break;
                        }
                        if *token != pad_id && *token != mask_id {
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
            let target_linear = selected_positions
                .iter()
                .map(|&p| {
                    context_seq[p] = mask_id;
                    (b * max_seq + p) as u32
                })
                .collect::<Vec<_>>();
            (context_seq, target_seq, target_linear)
        })
        .collect::<Vec<_>>();

    let mut context_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * max_seq);
    let mut target_linear = Vec::new();
    for (context_seq, target_seq, row_target_linear) in rows {
        context_buf.extend(context_seq);
        target_buf.extend(target_seq);
        target_linear.extend(row_target_linear);
    }

    let context_ids = Tensor::from_vec(context_buf, (batch_size, max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, max_seq), device)?;
    let n_targets = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear, (n_targets,), device)?;
    Ok((context_ids, target_ids, target_linear_indices))
}

pub fn tokenize_for_inference(text: &str) -> Vec<String> {
    tokenize_for_inference_mode(text, TokenizationMode::Default)
}

pub fn tokenize_for_inference_mode(text: &str, mode: TokenizationMode) -> Vec<String> {
    tokenize_with_mode(text, mode)
}

pub fn encode_text_with_vocab_mode(text: &str, vocab: &Vocab, mode: TokenizationMode) -> Vec<u32> {
    match mode {
        TokenizationMode::Default => default_lexical_chunks(text)
            .into_iter()
            .flat_map(|chunk| vocab.encode_boundless(&chunk))
            .collect(),
        TokenizationMode::CodeAware => encode_code_aware_text(text, vocab),
    }
}

fn encode_code_aware_text(text: &str, vocab: &Vocab) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        if let Some(token) = CODE_CONTROL_TOKENS
            .iter()
            .copied()
            .find(|token| text[start..].starts_with(token))
        {
            if let Some(&id) = vocab.token_to_id.get(token) {
                ids.push(id);
                start += token.len();
                continue;
            }
        }
        let next_special = CODE_CONTROL_TOKENS
            .iter()
            .filter_map(|token| text[start..].find(token).map(|idx| start + idx))
            .filter(|&idx| idx > start)
            .min()
            .unwrap_or(text.len());
        let end = if next_special > start {
            next_special
        } else {
            start
                + text[start..]
                    .chars()
                    .next()
                    .map(char::len_utf8)
                    .unwrap_or(1)
        };
        for chunk in code_aware_chunks(&text[start..end]) {
            if let Some(&id) = vocab.token_to_id.get(&chunk) {
                ids.push(id);
            } else {
                ids.extend(vocab.encode_boundless(&chunk));
            }
        }
        start = end;
    }
    ids
}

pub fn encode_world_examples(rows: &[RawWorldExample], vocab: &Vocab) -> Vec<WorldExample> {
    encode_world_examples_with_mode(rows, vocab, TokenizationMode::Default)
}

pub fn encode_world_examples_with_mode(
    rows: &[RawWorldExample],
    vocab: &Vocab,
    mode: TokenizationMode,
) -> Vec<WorldExample> {
    rows.par_iter()
        .map(|row| WorldExample {
            state_tokens: encode_text_with_vocab_mode(&row.state_text, vocab, mode),
            next_tokens: encode_text_with_vocab_mode(&row.next_text, vocab, mode),
            action_label: row.action_label,
        })
        .collect()
}

fn merge_pair_counts(sequences: &[Vec<u32>]) -> HashMap<(u32, u32), usize> {
    sequences
        .par_iter()
        .fold(HashMap::new, |mut counts, seq| {
            for pair in seq.windows(2) {
                if let [left, right] = pair {
                    *counts.entry((*left, *right)).or_insert(0) += 1;
                }
            }
            counts
        })
        .reduce(HashMap::new, |mut left, right| {
            for (pair, count) in right {
                *left.entry(pair).or_insert(0) += count;
            }
            left
        })
}

fn apply_merge_to_sequence(seq: &mut Vec<u32>, left: u32, right: u32, merged: u32) {
    if seq.len() < 2 {
        return;
    }
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0usize;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            out.push(merged);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    *seq = out;
}

fn train_bpe_from_texts_by_chunker<F>(
    texts: &[String],
    max_vocab: usize,
    label: &str,
    chunker: F,
) -> Result<(Vocab, usize)>
where
    F: Fn(&str) -> Vec<String> + Sync,
{
    if texts.is_empty() {
        bail!("cannot train tokenizer on empty text set");
    }
    let start = Instant::now();
    let char_counts = texts
        .par_iter()
        .fold(HashMap::new, |mut counts, text| {
            for chunk in chunker(text) {
                for ch in chunk.chars() {
                    *counts.entry(ch).or_insert(0) += 1;
                }
            }
            counts
        })
        .reduce(HashMap::new, |mut left, right| {
            for (ch, count) in right {
                *left.entry(ch).or_insert(0) += count;
            }
            left
        });
    let mut chars = char_counts.into_iter().collect::<Vec<_>>();
    chars.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut vocab = Vocab::new();
    vocab.ensure_byte_tokens();
    if label == "code-aware" {
        for token in CODE_CONTROL_TOKENS {
            vocab.add_token(token);
        }
    }
    for (token, _) in chars {
        if vocab.id_to_token.len() >= max_vocab {
            break;
        }
        vocab.add_token(&token.to_string());
    }

    let char_ids = vocab
        .token_to_id
        .iter()
        .filter_map(|(token, id)| {
            let mut chars = token.chars();
            let ch = chars.next()?;
            if chars.next().is_none() {
                Some((ch, *id))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let mut sequences = texts
        .par_iter()
        .flat_map_iter(|text| {
            chunker(text).into_iter().map(|chunk| {
                let mut ids = Vec::new();
                for ch in chunk.chars() {
                    if let Some(&id) = char_ids.get(&ch) {
                        ids.push(id);
                    } else {
                        let mut buf = [0u8; 4];
                        ids.extend(ch.encode_utf8(&mut buf).as_bytes().iter().map(|byte| {
                            vocab
                                .token_to_id
                                .get(&format!("<byte:{byte:02X}>"))
                                .copied()
                                .unwrap_or(vocab.unk_id)
                        }));
                    }
                }
                ids
            })
        })
        .filter(|seq| !seq.is_empty())
        .collect::<Vec<_>>();

    let initial_vocab_len = vocab.id_to_token.len();
    let possible_merges = max_vocab.saturating_sub(initial_vocab_len);
    let target_merges = bpe_max_merges()
        .map(|limit| limit.min(possible_merges))
        .unwrap_or(possible_merges);
    if target_merges < possible_merges {
        println!(
            "{label} BPE merge budget capped: target_merges={target_merges}, requested_vocab={max_vocab}, initial_vocab={initial_vocab_len}"
        );
    }
    let progress_every = bpe_progress_every_merges();
    let mut merges_applied = 0usize;
    while vocab.id_to_token.len() < max_vocab && merges_applied < target_merges {
        let pair_counts = merge_pair_counts(&sequences);
        let best = pair_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .filter(|((left, right), _)| {
                !vocab.id_to_token[*left as usize].starts_with("<byte:")
                    && !vocab.id_to_token[*right as usize].starts_with("<byte:")
            })
            .max_by(|((l1, r1), c1), ((l2, r2), c2)| {
                c1.cmp(c2)
                    .then_with(|| {
                        vocab.id_to_token[*l1 as usize].cmp(&vocab.id_to_token[*l2 as usize])
                    })
                    .then_with(|| {
                        vocab.id_to_token[*r1 as usize].cmp(&vocab.id_to_token[*r2 as usize])
                    })
            });
        let Some(((left, right), _count)) = best else {
            break;
        };
        let merged_token = format!(
            "{}{}",
            vocab.id_to_token[left as usize], vocab.id_to_token[right as usize]
        );
        let merged = vocab.add_merge(left, right, &merged_token);
        sequences
            .par_iter_mut()
            .for_each(|seq| apply_merge_to_sequence(seq, left, right, merged));
        merges_applied += 1;
        if merges_applied.is_multiple_of(progress_every) {
            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "{label} BPE merge progress: {merges_applied}/{target_merges} merges, vocab={}, elapsed={elapsed:.1}s",
                vocab.id_to_token.len()
            );
        }
    }

    let total_tokens = sequences.iter().map(|seq| seq.len()).sum();
    let elapsed = start.elapsed().as_secs_f32();
    println!(
        "{label} BPE training complete: merges_applied={merges_applied}, vocab={}, elapsed={elapsed:.1}s",
        vocab.id_to_token.len()
    );
    Ok((vocab, total_tokens))
}

fn train_default_bpe_from_texts(texts: &[String], max_vocab: usize) -> Result<(Vocab, usize)> {
    train_bpe_from_texts_by_chunker(texts, max_vocab, "lexical", default_lexical_chunks)
}

fn train_code_aware_bpe_from_texts(texts: &[String], max_vocab: usize) -> Result<(Vocab, usize)> {
    train_bpe_from_texts_by_chunker(texts, max_vocab, "code-aware", code_aware_chunks)
}

pub fn build_vocab_from_raw_world_file_with_mode(
    path: &PathBuf,
    max_vocab: usize,
    mode: TokenizationMode,
) -> Result<(Vocab, VocabStats, usize)> {
    build_vocab_from_raw_world_file_with_mode_action_filter(path, max_vocab, mode, None)
}

pub fn build_vocab_from_raw_world_file_with_mode_action_filter(
    path: &PathBuf,
    max_vocab: usize,
    mode: TokenizationMode,
    action_filter: Option<u32>,
) -> Result<(Vocab, VocabStats, usize)> {
    let reader = BufReader::new(File::open(path)?);
    let mut row_count = 0usize;
    let mut texts = Vec::new();
    let budget = if mode == TokenizationMode::CodeAware {
        let configured = code_vocab_sample_budget();
        println!(
            "code-aware vocab sampling budget for {}: {}",
            path.display(),
            configured.describe()
        );
        configured
    } else {
        VocabSampleBudget::default()
    };
    let mut sampled_text_bytes = 0usize;
    let chunk_lines = vocab_scan_chunk_lines();
    println!(
        "{} vocab parallel scan for {}: chunk_lines={} rayon_threads={}",
        mode.as_str(),
        path.display(),
        chunk_lines,
        rayon::current_num_threads()
    );
    let mut lines = reader.lines();
    let mut next_progress = DEFAULT_VOCAB_SCAN_PROGRESS_EVERY_ROWS;
    'scan: loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        let rows = chunk
            .par_iter()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                let row = raw_world_example_from_line_with_mode(line, mode)?;
                if action_filter.is_some_and(|wanted| row.action_label != wanted) {
                    return None;
                }
                Some((row.state_text, row.next_text))
            })
            .collect::<Vec<_>>();
        for (left, right) in rows {
            if let Some(limit) = budget.max_rows {
                if row_count >= limit {
                    println!(
                        "vocab sampling row budget reached for {}: kept {row_count} rows",
                        path.display()
                    );
                    break 'scan;
                }
            }
            let pair_text_bytes = left.len() + right.len();
            if let Some(limit) = budget.max_text_bytes {
                if row_count > 0 && sampled_text_bytes.saturating_add(pair_text_bytes) > limit {
                    println!(
                        "vocab sampling byte budget reached for {}: kept {row_count} rows and {} text bytes",
                        path.display(),
                        sampled_text_bytes
                    );
                    break 'scan;
                }
            }
            texts.push(left);
            texts.push(right);
            sampled_text_bytes = sampled_text_bytes.saturating_add(pair_text_bytes);
            row_count += 1;
            while row_count >= next_progress {
                println!(
                    "Vocab scan progress: {row_count} usable rows kept for {} (sampled_text_bytes={sampled_text_bytes})",
                    path.display()
                );
                next_progress += DEFAULT_VOCAB_SCAN_PROGRESS_EVERY_ROWS;
            }
        }
    }
    if row_count == 0 {
        bail!("cannot build vocab from empty raw world file {:?}", path);
    }
    let (vocab, total_tokens) = if mode == TokenizationMode::CodeAware {
        train_code_aware_bpe_from_texts(&texts, max_vocab)?
    } else {
        train_default_bpe_from_texts(&texts, max_vocab)?
    };
    let vocab_len = vocab.id_to_token.len();
    Ok((
        vocab,
        VocabStats {
            total_tokens,
            covered_tokens: total_tokens,
            oov_tokens: 0,
            unique_tokens: vocab_len,
            vocab_size: vocab_len,
        },
        row_count,
    ))
}

pub fn count_raw_world_rows(path: &PathBuf) -> Result<usize> {
    count_raw_world_rows_split(path, None, 0)
}

pub fn count_raw_world_rows_split(
    path: &PathBuf,
    split_modulus: Option<usize>,
    split_remainder: usize,
) -> Result<usize> {
    count_raw_world_rows_split_with_mode(
        path,
        TokenizationMode::Default,
        split_modulus,
        split_remainder,
    )
}

pub fn count_raw_world_rows_split_with_mode(
    path: &PathBuf,
    mode: TokenizationMode,
    split_modulus: Option<usize>,
    split_remainder: usize,
) -> Result<usize> {
    count_raw_world_rows_split_with_mode_action_filter(
        path,
        mode,
        split_modulus,
        split_remainder,
        None,
    )
}

pub fn count_raw_world_rows_split_with_mode_action_filter(
    path: &PathBuf,
    mode: TokenizationMode,
    split_modulus: Option<usize>,
    split_remainder: usize,
    action_filter: Option<u32>,
) -> Result<usize> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if let Some(modulus) = split_modulus {
            if line_idx % modulus != split_remainder {
                continue;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(row) = raw_world_example_from_line_with_mode(line, mode) else {
            continue;
        };
        if action_filter.is_some_and(|wanted| row.action_label != wanted) {
            continue;
        }
        count += 1;
    }
    if count == 0 {
        bail!("no usable world-model rows found in {:?}", path);
    }
    Ok(count)
}

fn encode_sequence(tokens: &[u32], max_seq: usize, pad_id: u32) -> (Vec<u32>, usize) {
    let mut seq = Vec::with_capacity(max_seq);
    for &id in tokens.iter().take(max_seq) {
        seq.push(id);
    }
    let length = seq.len();
    while seq.len() < max_seq {
        seq.push(pad_id);
    }
    (seq, length)
}

fn encode_sequence_tail(tokens: &[u32], max_seq: usize, pad_id: u32) -> (Vec<u32>, usize) {
    let start = tokens.len().saturating_sub(max_seq);
    encode_sequence(&tokens[start..], max_seq, pad_id)
}

fn encode_sequence_head(tokens: &[u32], max_seq: usize, pad_id: u32) -> (Vec<u32>, usize) {
    let len = tokens.len().min(max_seq);
    let mut out = Vec::with_capacity(max_seq);
    out.extend(tokens.iter().take(len).copied());
    out.extend(std::iter::repeat_n(pad_id, max_seq.saturating_sub(len)));
    (out, len)
}

/// Build decoder teacher-forcing batch from world batch.
/// input[b] = state[b, 0..state_len] ++ next[b, 0..next_len-1], padded to decoder_len.
/// target[b] = <pad> for prompt-only positions, then next[b, 0..next_len], padded to decoder_len.
/// loss_mask[b] = 1.0 only for target continuation positions (the `next` side), 0.0 elsewhere.
///
/// This intentionally avoids training the decoder to reproduce the prompt/state tokens.
/// The prompt is context only; the supervised target is the continuation.
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
        let raw_sl = state_lens.get(b).copied().unwrap_or(1).min(max_seq);
        let nl = next_lens.get(b).copied().unwrap_or(1).min(max_seq);
        let prompt_len = if raw_sl == 0 && max_seq > 0 {
            1
        } else {
            raw_sl
        };

        if raw_sl == 0 && prompt_len > 0 {
            input_buf.push(pad_id);
        } else {
            input_buf.extend(state_v[b].iter().take(raw_sl).copied());
        }
        input_buf.extend(next_v[b].iter().take(nl.saturating_sub(1)).copied());
        let input_len = prompt_len + nl.saturating_sub(1);
        for _ in input_len..decoder_len {
            input_buf.push(pad_id);
        }

        target_buf.extend(std::iter::repeat_n(pad_id, prompt_len.saturating_sub(1)));
        target_buf.extend(next_v[b].iter().take(nl).copied());
        let target_len = prompt_len.saturating_sub(1) + nl;
        for _ in target_len..decoder_len {
            target_buf.push(pad_id);
        }

        mask_buf.extend(std::iter::repeat_n(0.0f32, prompt_len.saturating_sub(1)));
        mask_buf.extend(std::iter::repeat_n(1.0f32, nl));
        mask_buf.extend(std::iter::repeat_n(
            0.0f32,
            decoder_len.saturating_sub(target_len),
        ));
    }

    let input_ids = Tensor::from_vec(input_buf, (batch_size, decoder_len), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, decoder_len), device)?;
    let loss_mask = Tensor::from_vec(mask_buf, (batch_size, decoder_len), device)?;
    Ok((input_ids, target_ids, loss_mask))
}

/// Build decoder teacher-forcing tensors directly from CPU token rows.
///
/// Long prompts keep their tail so late constraints remain visible. Long completions keep their
/// head because autoregressive generation must learn imports, declarations, and exact signatures
/// before it can produce a valid body.
pub fn make_decoder_batch_from_slice(
    rows: &[WorldExample],
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    make_decoder_batch_from_slice_with_prompt_dropout(rows, max_seq, pad_id, pad_id, 0.0, device)
}

pub fn make_decoder_batch_from_slice_with_prompt_dropout(
    rows: &[WorldExample],
    max_seq: usize,
    pad_id: u32,
    drop_id: u32,
    prompt_dropout: f64,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = rows.len();
    let decoder_len = 2 * max_seq;
    let prompt_dropout = prompt_dropout.clamp(0.0, 1.0);
    let row_buffers = rows
        .par_iter()
        .map(|row| {
            let mut rng = rng();
            let (state_seq, state_len) = encode_sequence_tail(&row.state_tokens, max_seq, pad_id);
            let (next_seq, next_len) = encode_sequence_head(&row.next_tokens, max_seq, pad_id);
            let raw_sl = state_len.min(max_seq);
            let nl = next_len.min(max_seq);
            let prompt_len = if raw_sl == 0 && max_seq > 0 {
                1
            } else {
                raw_sl
            };

            let mut input = Vec::with_capacity(decoder_len);
            if raw_sl == 0 && prompt_len > 0 {
                input.push(pad_id);
            } else {
                input.extend(state_seq.iter().take(raw_sl).map(|token| {
                    if prompt_dropout > 0.0 && *token != pad_id && rng.random_bool(prompt_dropout) {
                        drop_id
                    } else {
                        *token
                    }
                }));
            }
            input.extend(next_seq.iter().take(nl.saturating_sub(1)).copied());
            let input_len = prompt_len + nl.saturating_sub(1);
            input.extend(std::iter::repeat_n(
                pad_id,
                decoder_len.saturating_sub(input_len),
            ));

            let mut target = Vec::with_capacity(decoder_len);
            target.extend(std::iter::repeat_n(pad_id, prompt_len.saturating_sub(1)));
            target.extend(next_seq.iter().take(nl).copied());
            let target_len = prompt_len.saturating_sub(1) + nl;
            target.extend(std::iter::repeat_n(
                pad_id,
                decoder_len.saturating_sub(target_len),
            ));

            let mut mask = Vec::with_capacity(decoder_len);
            mask.extend(std::iter::repeat_n(0.0f32, prompt_len.saturating_sub(1)));
            mask.extend(std::iter::repeat_n(1.0f32, nl));
            mask.extend(std::iter::repeat_n(
                0.0f32,
                decoder_len.saturating_sub(target_len),
            ));
            (input, target, mask)
        })
        .collect::<Vec<_>>();

    let mut input_buf = Vec::with_capacity(batch_size * decoder_len);
    let mut target_buf = Vec::with_capacity(batch_size * decoder_len);
    let mut mask_buf = Vec::with_capacity(batch_size * decoder_len);
    for (input, target, mask) in row_buffers {
        input_buf.extend(input);
        target_buf.extend(target);
        mask_buf.extend(mask);
    }

    let input_ids = Tensor::from_vec(input_buf, (batch_size, decoder_len), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, decoder_len), device)?;
    let loss_mask = Tensor::from_vec(mask_buf, (batch_size, decoder_len), device)?;
    Ok((input_ids, target_ids, loss_mask))
}

#[allow(clippy::type_complexity)]
pub fn make_world_batch_from_slice(
    rows: &[WorldExample],
    max_seq: usize,
    pad_id: u32,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<usize>, Vec<usize>, Vec<u32>)> {
    let batch_size = rows.len();
    let row_buffers = rows
        .par_iter()
        .map(|row| {
            let (state_seq, state_len) = encode_sequence_tail(&row.state_tokens, max_seq, pad_id);
            let (next_seq, next_len) = encode_sequence_tail(&row.next_tokens, max_seq, pad_id);
            (state_seq, next_seq, state_len, next_len, row.action_label)
        })
        .collect::<Vec<_>>();

    let mut state_buf = Vec::with_capacity(batch_size * max_seq);
    let mut next_buf = Vec::with_capacity(batch_size * max_seq);
    let mut state_lens = Vec::with_capacity(batch_size);
    let mut next_lens = Vec::with_capacity(batch_size);
    let mut action_labels = Vec::with_capacity(batch_size);
    for (state_seq, next_seq, state_len, next_len, action_label) in row_buffers {
        state_buf.extend(state_seq);
        next_buf.extend(next_seq);
        state_lens.push(state_len);
        next_lens.push(next_len);
        action_labels.push(action_label);
    }

    let state_ids = Tensor::from_vec(state_buf, (batch_size, max_seq), device)?;
    let next_ids = Tensor::from_vec(next_buf, (batch_size, max_seq), device)?;
    Ok((state_ids, next_ids, state_lens, next_lens, action_labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nanos}"))
    }

    #[test]
    fn repeated_decoder_rows_always_share_the_same_split() {
        let left = [10, 11, 12];
        let right = [20, 21, 22];
        let first = stable_token_pair_split_hash(&left, &right, ACTION_CODE);
        let repeated = stable_token_pair_split_hash(&left, &right, ACTION_CODE);
        let changed = stable_token_pair_split_hash(&left, &[20, 21, 23], ACTION_CODE);

        assert_eq!(first, repeated);
        assert_ne!(first, changed);
        assert_ne!(
            split_keeps(first, Some(20), 0, true),
            split_keeps(first, Some(20), 0, false)
        );
    }

    #[test]
    fn pair_source_manifest_counts_and_streams_both_pair_sides() -> Result<()> {
        let dir = unique_temp_dir("tofy-pair-source-manifest");
        fs::create_dir_all(&dir)?;
        let pairs = dir.join("pairs.txt");
        let plain = dir.join("plain.txt");
        let manifest = dir.join("sources.txt");
        fs::write(&pairs, "hello\tworld\n")?;
        fs::write(&plain, "plain text\n")?;
        fs::write(
            &manifest,
            format!(
                "{PAIR_SOURCE_MANIFEST_HEADER}\n{}\n{}\n",
                pairs.display(),
                plain.display()
            ),
        )?;

        assert_eq!(count_pairs_with_vocab(&manifest)?, 3);
        let mut stream = PairStream::with_shuffle(&manifest, 1, 1)?;
        let got: Vec<String> = stream
            .next_batch(3)?
            .into_iter()
            .map(|tokens| tokens.concat())
            .collect();
        assert_eq!(got, ["hello", "world", "plain text"]);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn decoder_batch_keeps_tail_of_long_prompt() -> Result<()> {
        let device = Device::Cpu;
        let row = WorldExample {
            state_tokens: vec![10, 11, 12, 13, 14],
            next_tokens: vec![20, 21, 22],
            action_label: ACTION_CODE,
        };

        let (input, target, mask) = make_decoder_batch_from_slice(&[row], 3, 0, &device)?;

        assert_eq!(input.to_vec2::<u32>()?[0], vec![12, 13, 14, 20, 21, 0]);
        assert_eq!(target.to_vec2::<u32>()?[0], vec![0, 0, 20, 21, 22, 0]);
        assert_eq!(
            crate::util::vec2_f32(&mask)?[0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn decoder_batch_does_not_train_on_empty_target_pad() -> Result<()> {
        let device = Device::Cpu;
        let row = WorldExample {
            state_tokens: vec![10, 11],
            next_tokens: Vec::new(),
            action_label: ACTION_CODE,
        };

        let (_input, target, mask) = make_decoder_batch_from_slice(&[row], 3, 0, &device)?;

        assert_eq!(target.to_vec2::<u32>()?[0], vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(crate::util::vec2_f32(&mask)?[0], vec![0.0; 6]);
        Ok(())
    }

    #[test]
    fn decoder_batch_seeds_empty_prompt_without_self_copy() -> Result<()> {
        let device = Device::Cpu;
        let row = WorldExample {
            state_tokens: Vec::new(),
            next_tokens: vec![20, 21],
            action_label: ACTION_CODE,
        };

        let (input, target, mask) = make_decoder_batch_from_slice(&[row], 3, 0, &device)?;

        assert_eq!(input.to_vec2::<u32>()?[0], vec![0, 20, 0, 0, 0, 0]);
        assert_eq!(target.to_vec2::<u32>()?[0], vec![20, 21, 0, 0, 0, 0]);
        assert_eq!(
            crate::util::vec2_f32(&mask)?[0],
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn tensor_decoder_batch_seeds_empty_prompt_without_self_copy() -> Result<()> {
        let device = Device::Cpu;
        let state = Tensor::from_vec(vec![0u32, 0, 0], (1, 3), &device)?;
        let next = Tensor::from_vec(vec![20u32, 21, 0], (1, 3), &device)?;

        let (input, target, mask) = make_decoder_batch(&state, &next, &[0], &[2], 3, 0, &device)?;

        assert_eq!(input.to_vec2::<u32>()?[0], vec![0, 20, 0, 0, 0, 0]);
        assert_eq!(target.to_vec2::<u32>()?[0], vec![20, 21, 0, 0, 0, 0]);
        assert_eq!(
            crate::util::vec2_f32(&mask)?[0],
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn decoder_batch_keeps_head_of_long_target() -> Result<()> {
        let device = Device::Cpu;
        let row = WorldExample {
            state_tokens: vec![10, 11],
            next_tokens: vec![20, 21, 22, 23, 24],
            action_label: ACTION_CODE,
        };

        let (input, target, mask) = make_decoder_batch_from_slice(&[row], 3, 0, &device)?;

        assert_eq!(input.to_vec2::<u32>()?[0], vec![10, 11, 20, 21, 0, 0]);
        assert_eq!(target.to_vec2::<u32>()?[0], vec![0, 20, 21, 22, 0, 0]);
        assert_eq!(
            crate::util::vec2_f32(&mask)?[0],
            vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn world_batch_keeps_state_and_next_tail() -> Result<()> {
        let device = Device::Cpu;
        let row = WorldExample {
            state_tokens: vec![10, 11, 12, 13],
            next_tokens: vec![20, 21, 22, 23],
            action_label: ACTION_CODE,
        };

        let (state, next, state_lens, next_lens, labels) =
            make_world_batch_from_slice(&[row], 3, 0, &device)?;

        assert_eq!(state.to_vec2::<u32>()?[0], vec![11, 12, 13]);
        assert_eq!(next.to_vec2::<u32>()?[0], vec![21, 22, 23]);
        assert_eq!(state_lens, vec![3]);
        assert_eq!(next_lens, vec![3]);
        assert_eq!(labels, vec![ACTION_CODE]);
        Ok(())
    }

    #[test]
    fn pair_line_tokenization_includes_completion_side() {
        let tokens = split_line_with_min_tokens_mode(
            "prompt words\tcompletion_tail",
            1,
            TokenizationMode::Default,
        )
        .expect("pair should tokenize");

        assert!(tokens.concat().contains("completion_tail"));
    }

    #[test]
    fn default_bpe_cannot_merge_a_repetitive_template_across_lexical_boundaries() -> Result<()> {
        let template = "Write func Solve using veclab.Mextrenstel for the input.".to_string();
        let texts = vec![template.clone(); 32];
        let (vocab, _) = train_default_bpe_from_texts(&texts, 800)?;
        let encoded = encode_text_with_vocab_mode(&template, &vocab, TokenizationMode::Default);

        assert!(encoded.len() >= 8, "template collapsed to {encoded:?}");
        assert!(vocab.id_to_token.iter().all(|token| {
            let has_whitespace = token.chars().any(char::is_whitespace);
            let has_non_whitespace = token.chars().any(|ch| !ch.is_whitespace());
            !(has_whitespace && has_non_whitespace)
        }));
        assert_eq!(vocab.decode_ids_lossy(&encoded), template);
        Ok(())
    }

    #[test]
    fn cached_pair_stream_skips_single_token_rows() -> Result<()> {
        let path = unique_temp_dir("tofy-short-cache").with_extension("bin");
        let mut bytes = TOKEN_CACHE_MAGIC.to_vec();
        for ids in [&[7u32][..], &[8u32, 9][..]] {
            bytes.extend_from_slice(&(ids.len() as u32).to_le_bytes());
            for id in ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        }
        fs::write(&path, bytes)?;

        let mut stream = CachedPairStream::with_shuffle(&path, 1)?;
        assert_eq!(stream.next_batch(1)?[0].tokens, vec![8, 9]);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn prose_with_common_words_does_not_default_to_code() {
        let text = "You can use ideas from the type of example above in your explanation.";

        assert_eq!(action_label_heuristic(text), ACTION_TEXT_REPLY);
    }

    #[test]
    fn explicit_column_action_strips_embedded_action_prefix() {
        let row = raw_world_example_from_line_with_mode(
            "prompt\t<action:text_reply> This is prose.\ttext_reply",
            TokenizationMode::Default,
        )
        .expect("row should parse");

        assert_eq!(row.action_label, ACTION_TEXT_REPLY);
        assert_eq!(row.next_text, "This is prose.");
    }

    #[test]
    fn empty_cached_pair_stream_returns_error() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-empty-cache-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::write(&path, TOKEN_CACHE_MAGIC)?;
        let mut stream = CachedPairStream::with_shuffle(&path, 1)?;
        let error = match stream.next_batch(1) {
            Ok(_) => panic!("empty cache must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("two passes"));
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn impossible_cached_pair_split_returns_error() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-impossible-split-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let mut bytes = TOKEN_CACHE_MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, bytes)?;
        let mut stream = CachedPairStream::with_split(&path, 1, Some(2), 1, false)?;
        let error = match stream.next_batch(1) {
            Ok(_) => panic!("impossible split must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("two passes"));
        std::fs::remove_file(path)?;
        Ok(())
    }
}
