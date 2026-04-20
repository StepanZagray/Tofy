use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

use crate::model::vocab::{Pair, Vocab};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizationMode {
    Default,
    CodeAware,
}

impl TokenizationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::CodeAware => "code-aware",
        }
    }
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

fn is_control_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '-' | '+' | '.')
}

fn try_read_control_token(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars.get(start).copied() != Some('<') {
        return None;
    }
    let mut end = start + 1;
    while end < chars.len() && chars[end] != '>' && is_control_token_char(chars[end]) {
        end += 1;
    }
    if end < chars.len() && chars[end] == '>' && end > start + 1 {
        let token: String = chars[start..=end].iter().collect();
        return Some((token, end + 1));
    }
    None
}

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if let Some((token, next_i)) = try_read_control_token(&chars, i) {
            if !buf.is_empty() {
                tokens.push(buf.clone());
                buf.clear();
            }
            tokens.push(token);
            i = next_i;
            continue;
        }
        let ch = chars[i];
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
        i += 1;
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn push_code_identifier_tokens(tokens: &mut Vec<String>, ident: &mut String) {
    if ident.is_empty() {
        return;
    }
    let chars: Vec<char> = ident.chars().collect();
    let mut piece = String::new();
    for (idx, ch) in chars.iter().copied().enumerate() {
        let prev = idx.checked_sub(1).and_then(|i| chars.get(i)).copied();
        let next = chars.get(idx + 1).copied();
        let boundary = if piece.is_empty() {
            false
        } else if ch.is_ascii_uppercase() {
            matches!(prev, Some(p) if p.is_ascii_lowercase() || p.is_ascii_digit())
                || matches!(
                    (prev, next),
                    (Some(p), Some(n)) if p.is_ascii_uppercase() && n.is_ascii_lowercase()
                )
        } else if ch.is_ascii_digit() {
            matches!(prev, Some(p) if p.is_ascii_alphabetic())
        } else if ch.is_ascii_lowercase() {
            matches!(prev, Some(p) if p.is_ascii_digit())
        } else {
            false
        };
        if boundary {
            tokens.push(piece.clone());
            piece.clear();
        }
        piece.push(ch);
    }
    if !piece.is_empty() {
        tokens.push(piece);
    }
    ident.clear();
}

fn push_indent_token(tokens: &mut Vec<String>, indent_cols: usize, saw_tab: bool) {
    if indent_cols == 0 && !saw_tab {
        return;
    }
    if saw_tab {
        tokens.push("<indent_tab>".to_string());
    }
    if indent_cols > 0 {
        let bucket = indent_cols.min(16);
        tokens.push(format!("<indent_{bucket}>"));
    }
}

fn is_lifetime_marker(chars: &[char], idx: usize) -> bool {
    chars.get(idx) == Some(&'\'')
        && chars
            .get(idx + 1)
            .copied()
            .map(|ch| ch.is_ascii_alphabetic() || ch == '_')
            .unwrap_or(false)
        && chars.get(idx + 2) != Some(&'\'')
}

fn tokenize_code_text(text: &str) -> Vec<String> {
    const THREE_CHAR_TOKENS: [&str; 8] = ["<<<", ">>>", "<<=", ">>=", "...", "===", "!==", "**="];
    const TWO_CHAR_TOKENS: [&str; 23] = [
        "->", "=>", "::", "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
        "%=", "&=", "|=", "^=", "<<", ">>", "**", "//",
    ];

    let mut tokens = Vec::new();
    let mut ident = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if let Some((token, next_i)) = try_read_control_token(&chars, i) {
            push_code_identifier_tokens(&mut tokens, &mut ident);
            tokens.push(token);
            i = next_i;
            continue;
        }
        if is_lifetime_marker(&chars, i) {
            push_code_identifier_tokens(&mut tokens, &mut ident);
            tokens.push("'".to_string());
            i += 1;
            continue;
        }
        let ch = chars[i];
        if ch == '"' || ch == '\'' || ch == '`' {
            push_code_identifier_tokens(&mut tokens, &mut ident);
            let quote = ch;
            i += 1;
            let mut escaped = false;
            let mut inner_len = 0usize;
            while i < chars.len() {
                let cur = chars[i];
                i += 1;
                if escaped {
                    escaped = false;
                    inner_len += 1;
                    continue;
                }
                if cur == '\\' {
                    escaped = true;
                    continue;
                }
                if cur == quote {
                    break;
                }
                if cur == '\n' {
                    break;
                }
                inner_len += 1;
            }
            if quote == '\'' && inner_len <= 4 {
                tokens.push("<char_lit>".to_string());
            } else {
                tokens.push("<str_lit>".to_string());
            }
            continue;
        }
        if ch.is_ascii_digit() && ident.is_empty() {
            push_code_identifier_tokens(&mut tokens, &mut ident);
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_hexdigit()
                    || matches!(chars[i], '_' | '.' | 'x' | 'X' | 'o' | 'O' | 'b' | 'B'))
            {
                i += 1;
            }
            tokens.push("<num_lit>".to_string());
            continue;
        }
        if ch.is_ascii_alphanumeric() {
            ident.push(ch);
            i += 1;
            continue;
        }
        push_code_identifier_tokens(&mut tokens, &mut ident);
        if ch == '\n' {
            tokens.push("<nl>".to_string());
            i += 1;
            let mut indent_cols = 0usize;
            let mut saw_tab = false;
            while i < chars.len() {
                match chars[i] {
                    ' ' => {
                        indent_cols += 1;
                        i += 1;
                    }
                    '\t' => {
                        indent_cols += 4;
                        saw_tab = true;
                        i += 1;
                    }
                    '\n' => {
                        tokens.push("<nl>".to_string());
                        i += 1;
                        indent_cols = 0;
                        saw_tab = false;
                    }
                    _ => break,
                }
            }
            push_indent_token(&mut tokens, indent_cols, saw_tab);
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if i + 2 < chars.len() {
            let three = format!("{}{}{}", chars[i], chars[i + 1], chars[i + 2]);
            if THREE_CHAR_TOKENS.contains(&three.as_str()) {
                tokens.push(three);
                i += 3;
                continue;
            }
        }
        if i + 1 < chars.len() {
            let two = format!("{}{}", chars[i], chars[i + 1]);
            if TWO_CHAR_TOKENS.contains(&two.as_str()) {
                tokens.push(two);
                i += 2;
                continue;
            }
        }
        tokens.push(ch.to_string());
        i += 1;
    }
    push_code_identifier_tokens(&mut tokens, &mut ident);
    tokens
}

fn tokenize_with_mode(text: &str, mode: TokenizationMode) -> Vec<String> {
    match mode {
        TokenizationMode::Default => tokenize_text(text),
        TokenizationMode::CodeAware => tokenize_code_text(text),
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
    let tokens = if let Some((left, _right)) = line.split_once("\t") {
        tokenize_with_mode(left, mode)
    } else if let Some((left, _right)) = line.split_once("|||") {
        tokenize_with_mode(left, mode)
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
    let tokens = split_line_with_min_tokens_mode(line, min_tokens, mode)?;
    Some(encode_tokens_with_vocab(&tokens, vocab, mode))
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
pub const DONE_SENTINEL: &str = "<done>";

fn parse_action_label_str(value: &str) -> Option<u32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "text" | "text_reply" => Some(ACTION_TEXT_REPLY),
        "1" | "code" => Some(ACTION_CODE),
        "2" | "done" => Some(ACTION_DONE),
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

/// Heuristic action label for orchestrator training: 0 = TextReply, 1 = Code, 2 = Done.
/// Used when building world examples from raw text (next turn string).
pub fn action_label_heuristic(next_turn: &str) -> u32 {
    let s = next_turn.trim();
    if s.is_empty() {
        return ACTION_DONE;
    }
    if s.eq_ignore_ascii_case(DONE_SENTINEL) {
        return ACTION_DONE;
    }
    if let Some((explicit, stripped)) = explicit_action_from_next_text(s) {
        if explicit == ACTION_DONE || stripped.is_empty() {
            return explicit;
        }
    }
    if s.contains("```") {
        return ACTION_CODE; // code block
    }
    let lower = s.to_ascii_lowercase();
    let code_keywords = [
        "fn ",
        "let ",
        "const ",
        "var ",
        "function ",
        "def ",
        "class ",
        "struct ",
        "enum ",
        "impl ",
        "import ",
        "from ",
        "return ",
        "pub ",
        "use ",
        "async ",
        "await ",
        "#include",
        "package ",
        "interface ",
        "type ",
        "select ",
        "insert ",
        "update ",
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
        if trimmed.ends_with('{')
            || trimmed.ends_with("};")
            || trimmed.contains("::")
            || trimmed.contains("->")
            || trimmed.contains("=>")
        {
            code_score += 2;
        }
        if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
            prose_score += 1;
        }
        let word_count = trimmed.split_whitespace().count();
        if word_count >= 8 && !trimmed.contains("::") && !trimmed.contains("->") {
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
    /// 0 = TextReply, 1 = Code, 2 = Done.
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
    let action_label = if let Some(action) = explicit_action {
        action
    } else if let Some((action, stripped)) = explicit_action_from_next_text(&next_text) {
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
        state_tokens: encode_tokens_with_vocab(
            &tokenize_with_mode(&row.state_text, mode),
            vocab,
            mode,
        ),
        next_tokens: encode_tokens_with_vocab(
            &tokenize_with_mode(&row.next_text, mode),
            vocab,
            mode,
        ),
        action_label: row.action_label,
    })
}

/// Minimum number of tokens per line to include (2 = skip single-token lines; 1 = paragraph mode).
pub const DEFAULT_MIN_TOKENS_PER_LINE: usize = 2;
pub const DEFAULT_STREAM_SHUFFLE_BUFFER: usize = 1024;
const TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_TOKEN_CACHE_V1\n";
const DUAL_TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_DUAL_TOKEN_CACHE_V1\n";
type TokenCacheRecord = (Vec<u32>, Vec<u32>, u32);

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

    pub fn with_shuffle(
        path: &PathBuf,
        min_tokens: usize,
        shuffle_buffer_size: usize,
    ) -> Result<Self> {
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

pub struct CachedPairStream {
    path: PathBuf,
    reader: BufReader<File>,
    shuffle_buffer_size: usize,
    shuffle_buffer: Vec<Pair>,
}

impl CachedPairStream {
    pub fn new(path: &PathBuf) -> Result<Self> {
        Self::with_shuffle(path, DEFAULT_STREAM_SHUFFLE_BUFFER)
    }

    pub fn with_shuffle(path: &PathBuf, shuffle_buffer_size: usize) -> Result<Self> {
        let mut stream = Self {
            path: path.clone(),
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
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
        self.reader = BufReader::new(File::open(&self.path)?);
        self.read_magic()
    }

    fn read_next_pair(&mut self) -> Result<Pair> {
        loop {
            match read_token_cache_record(&mut self.reader)? {
                Some((tokens, _right, _action)) if !tokens.is_empty() => {
                    return Ok(Pair { tokens });
                }
                Some(_) => continue,
                None => self.reset()?,
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
        let idx = thread_rng().gen_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<Pair>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_pair()?);
        }
        Ok(batch)
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
    line_index: usize,
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
    row_index: usize,
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
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
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
        self.reader = BufReader::new(File::open(&self.path)?);
        self.row_index = 0;
        self.read_magic()
    }

    fn read_next_example(&mut self) -> Result<WorldExample> {
        loop {
            let Some((state_tokens, next_tokens, action_label)) =
                read_token_cache_record(&mut self.reader)?
            else {
                self.reset()?;
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
        let idx = thread_rng().gen_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<WorldExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
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
            reader: BufReader::new(File::open(path)?),
            shuffle_buffer_size: shuffle_buffer_size.max(1),
            shuffle_buffer: Vec::new(),
            split_modulus,
            split_remainder,
            exclude_split_matches,
            row_index: 0,
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
        self.reader = BufReader::new(File::open(&self.path)?);
        self.row_index = 0;
        self.read_magic()
    }

    fn read_next_example(&mut self) -> Result<CachedDecoderExample> {
        loop {
            let Some((enc_state, enc_next, dec_state, dec_next, action_label)) =
                read_dual_token_cache_record(&mut self.reader)?
            else {
                self.reset()?;
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
            if enc_state.is_empty()
                || enc_next.is_empty()
                || dec_state.is_empty()
                || dec_next.is_empty()
            {
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
        let idx = thread_rng().gen_range(0..self.shuffle_buffer.len());
        Ok(self.shuffle_buffer.swap_remove(idx))
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<CachedDecoderExample>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(self.next_example()?);
        }
        Ok(batch)
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
    let mut ids = Vec::with_capacity(len);
    for _ in 0..len {
        let Some(id) = read_u32_le(reader)? else {
            bail!("truncated token cache id sequence");
        };
        ids.push(id);
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
            line_index: 0,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = BufReader::new(File::open(&self.path)?);
        self.line_index = 0;
        Ok(())
    }

    fn read_next_example(&mut self) -> Result<RawWorldExample> {
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                self.reset()?;
                continue;
            }
            let line_idx = self.line_index;
            self.line_index += 1;
            if let Some(modulus) = self.split_modulus {
                let is_match = line_idx % modulus == self.split_remainder;
                let keep = if self.exclude_split_matches {
                    !is_match
                } else {
                    is_match
                };
                if !keep {
                    continue;
                }
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
            let action_label = if let Some(action) = explicit_action {
                action
            } else if let Some((action, stripped)) = explicit_action_from_next_text(&next_text) {
                next_text = stripped;
                action
            } else {
                action_label_heuristic(&next_text)
            };
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
    const PROGRESS_EVERY_LINES: usize = 500_000;

    let min_tok = min_tokens_per_line.unwrap_or(DEFAULT_MIN_TOKENS_PER_LINE);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut pair_count = 0usize;
    let mut raw_line_count = 0usize;
    let reader = BufReader::new(File::open(path)?);

    for line in reader.lines() {
        let line = line?;
        raw_line_count += 1;
        let Some(tokens) = split_line_with_min_tokens(&line, min_tok) else {
            if raw_line_count.is_multiple_of(PROGRESS_EVERY_LINES) {
                println!(
                    "Vocab scan progress: {} raw lines read, {} usable sequences kept...",
                    raw_line_count, pair_count
                );
            }
            continue;
        };
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        pair_count += 1;
        if raw_line_count.is_multiple_of(PROGRESS_EVERY_LINES) {
            println!(
                "Vocab scan progress: {} raw lines read, {} usable sequences kept...",
                raw_line_count, pair_count
            );
        }
    }

    if pair_count == 0 {
        bail!("no usable lines found in {:?}", path);
    }

    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let vocab_size = max_vocab.saturating_sub(3).min(sorted.len());
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
    let start = rng.gen_range(0..=tokens.len() - active_seq);
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
            rng.gen_range(0.0..=jitter).floor() as usize
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
    let start = rng.gen_range(0..=ids.len() - active_seq);
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
            rng.gen_range(0.0..=jitter).floor() as usize
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

fn build_masked_view(
    tokens: &[String],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> (Vec<u32>, Vec<u32>, Vec<usize>, bool) {
    let prepared_tokens = prepare_segmented_context_tokens(tokens, cfg, rng);
    let code_like = is_code_like_tokens(&prepared_tokens);
    let mut target_ids = vocab.encode(&prepared_tokens);
    let valid_len = target_ids.len().max(1);
    pad_or_truncate(&mut target_ids, cfg.max_seq, vocab.pad_id);
    let mut context_ids = target_ids.clone();

    let valid_positions: Vec<usize> = (0..valid_len).collect();
    let identifier_positions: Vec<usize> = prepared_tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, tok)| is_identifier_like(tok).then_some(idx))
        .collect();
    let comment_positions: Vec<usize> = prepared_tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, tok)| is_comment_token(tok).then_some(idx))
        .collect();
    let block_positions: Vec<usize> = prepared_tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, tok)| is_block_token(tok).then_some(idx))
        .collect();
    let text_boundary_positions: Vec<usize> = prepared_tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, tok)| is_text_boundary_token(tok).then_some(idx))
        .collect();
    let mut selected: HashSet<usize> = HashSet::new();

    let ratio_multiplier = if code_like {
        cfg.code_masked_ratio_multiplier.max(1.0)
    } else {
        1.0
    };
    let min_ratio = (cfg.min_masked_ratio * ratio_multiplier).clamp(0.08, 0.9);
    let max_ratio = (cfg.max_masked_ratio * ratio_multiplier).clamp(min_ratio, 0.9);
    let min_masked = (valid_positions.len() as f64 * min_ratio).ceil() as usize;
    let min_masked = min_masked.max(1).min(valid_positions.len());
    let max_masked = (valid_positions.len() as f64 * max_ratio).ceil() as usize;
    let max_masked = max_masked.max(min_masked).min(valid_positions.len());
    let target_masked = if min_masked >= max_masked {
        max_masked
    } else {
        rng.gen_range(min_masked..=max_masked)
    };
    let span_count = rng.gen_range(1..=cfg.max_spans_per_sample.max(1));
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
            && (span_idx == 0 || rng.gen_bool(cfg.identifier_focus_prob.clamp(0.0, 1.0)));
        let use_comment_focus =
            !comment_positions.is_empty() && rng.gen_bool(cfg.comment_focus_prob.clamp(0.0, 1.0));
        let use_block_focus = code_like
            && !block_positions.is_empty()
            && rng.gen_bool(cfg.block_focus_prob.clamp(0.0, 1.0));
        let use_text_boundary_focus = !code_like
            && !text_boundary_positions.is_empty()
            && rng.gen_bool(cfg.text_boundary_focus_prob.clamp(0.0, 1.0));
        let start_pool = if use_comment_focus {
            &comment_positions
        } else if use_block_focus {
            &block_positions
        } else if use_text_boundary_focus {
            &text_boundary_positions
        } else if use_identifier_focus {
            &identifier_positions
        } else {
            &valid_positions
        };
        let Some(&start) = start_pool.choose(rng) else {
            continue;
        };
        let span_len = if use_comment_focus {
            rng.gen_range(2..=span_len_cap.clamp(2, 12))
        } else if use_block_focus {
            rng.gen_range(2..=span_len_cap.clamp(2, 10))
        } else if use_text_boundary_focus {
            rng.gen_range(3..=span_len_cap.clamp(3, 12))
        } else if use_identifier_focus {
            rng.gen_range(1..=span_len_cap.min(4))
        } else {
            rng.gen_range(1..=span_len_cap)
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
        let span_len = rng
            .gen_range(1..=span_len_cap.min(target_masked.saturating_sub(selected.len()).max(1)));
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
    for &p in &selected_positions {
        context_ids[p] = vocab.mask_id;
    }

    (context_ids, target_ids, selected_positions, code_like)
}

fn build_masked_view_from_ids(
    ids: &[u32],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    rng: &mut rand::rngs::ThreadRng,
) -> (Vec<u32>, Vec<u32>, Vec<usize>, bool) {
    let prepared_ids = prepare_segmented_context_ids(ids, cfg, rng);
    let code_like = is_code_like_ids(&prepared_ids, vocab);
    let mut target_ids = prepared_ids.clone();
    let valid_len = target_ids.len().max(1);
    pad_or_truncate(&mut target_ids, cfg.max_seq, vocab.pad_id);
    let mut context_ids = target_ids.clone();

    let valid_positions: Vec<usize> = (0..valid_len).collect();
    let identifier_positions: Vec<usize> = prepared_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| is_identifier_like(token_str(vocab, id)).then_some(idx))
        .collect();
    let comment_positions: Vec<usize> = prepared_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| is_comment_token(token_str(vocab, id)).then_some(idx))
        .collect();
    let block_positions: Vec<usize> = prepared_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| is_block_token(token_str(vocab, id)).then_some(idx))
        .collect();
    let text_boundary_positions: Vec<usize> = prepared_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| is_text_boundary_token(token_str(vocab, id)).then_some(idx))
        .collect();
    let ratio_scale = if code_like {
        cfg.code_masked_ratio_multiplier
    } else {
        1.0
    };
    let min_ratio = (cfg.min_masked_ratio * ratio_scale).clamp(0.01, 0.60);
    let max_ratio = (cfg.max_masked_ratio * ratio_scale).clamp(min_ratio, 0.70);
    let sampled_ratio = if max_ratio > min_ratio {
        rng.gen_range(min_ratio..=max_ratio)
    } else {
        min_ratio
    };
    let target_masked = ((valid_len as f64) * sampled_ratio).ceil() as usize;
    let target_masked = target_masked.max(1).min(valid_len);
    let span_len_cap = cfg.max_span_len.max(1).min(valid_len.max(1));
    let span_count_cap = cfg.max_spans_per_sample.max(1);
    let mut selected: HashSet<usize> = HashSet::new();

    for _ in 0..span_count_cap {
        if selected.len() >= target_masked {
            break;
        }
        let use_identifier_focus = code_like
            && !identifier_positions.is_empty()
            && rng.gen_bool(cfg.identifier_focus_prob.clamp(0.0, 1.0));
        let use_comment_focus =
            code_like && !comment_positions.is_empty() && rng.gen_bool(cfg.comment_focus_prob);
        let use_block_focus = code_like
            && !block_positions.is_empty()
            && rng.gen_bool(cfg.block_focus_prob.clamp(0.0, 1.0));
        let use_text_boundary_focus = !code_like
            && !text_boundary_positions.is_empty()
            && rng.gen_bool(cfg.text_boundary_focus_prob.clamp(0.0, 1.0));
        let start_pool = if use_comment_focus {
            &comment_positions
        } else if use_block_focus {
            &block_positions
        } else if use_text_boundary_focus {
            &text_boundary_positions
        } else if use_identifier_focus {
            &identifier_positions
        } else {
            &valid_positions
        };
        let Some(&start) = start_pool.choose(rng) else {
            continue;
        };
        let span_len = if use_comment_focus {
            rng.gen_range(2..=span_len_cap.clamp(2, 12))
        } else if use_block_focus {
            rng.gen_range(2..=span_len_cap.clamp(2, 10))
        } else if use_text_boundary_focus {
            rng.gen_range(3..=span_len_cap.clamp(3, 12))
        } else if use_identifier_focus {
            rng.gen_range(1..=span_len_cap.min(4))
        } else {
            rng.gen_range(1..=span_len_cap)
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
        let span_len = rng
            .gen_range(1..=span_len_cap.min(target_masked.saturating_sub(selected.len()).max(1)));
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
    for &p in &selected_positions {
        context_ids[p] = vocab.mask_id;
    }

    (context_ids, target_ids, selected_positions, code_like)
}

pub fn make_augmented_jepa_batch(
    token_batches: &[Vec<String>],
    vocab: &Vocab,
    cfg: &CurriculumDenoisingConfig,
    device: &Device,
) -> Result<AugmentedJepaBatch> {
    let mut rng = thread_rng();
    let batch_size = token_batches.len();
    let mut view_a_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut view_b_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_linear = Vec::new();
    let mut code_like_count = 0usize;

    for (b, tokens) in token_batches.iter().enumerate() {
        let (view_a_ids, target_ids, selected_positions, code_like) =
            build_masked_view(tokens, vocab, cfg, &mut rng);
        let (view_b_ids, _, _, code_like_b) = build_masked_view(tokens, vocab, cfg, &mut rng);
        if code_like || code_like_b {
            code_like_count += 1;
        }
        for &p in &selected_positions {
            target_linear.push((b * cfg.max_seq + p) as u32);
        }
        view_a_buf.extend(view_a_ids);
        view_b_buf.extend(view_b_ids);
        target_buf.extend(target_ids);
    }

    let view_a_ids = Tensor::from_vec(view_a_buf, (batch_size, cfg.max_seq), device)?;
    let view_b_ids = Tensor::from_vec(view_b_buf, (batch_size, cfg.max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, cfg.max_seq), device)?;
    let target_count = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear, (target_count,), device)?;

    Ok(AugmentedJepaBatch {
        view_a_ids,
        view_b_ids,
        target_ids,
        target_linear_indices,
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
    let mut rng = thread_rng();
    let batch_size = pairs.len();
    let mut view_a_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut view_b_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_buf = Vec::with_capacity(batch_size * cfg.max_seq);
    let mut target_linear = Vec::new();
    let mut code_like_count = 0usize;

    for (b, pair) in pairs.iter().enumerate() {
        let (view_a_ids, target_ids, selected_positions, code_like) =
            build_masked_view_from_ids(&pair.tokens, vocab, cfg, &mut rng);
        let (view_b_ids, _, _, code_like_b) =
            build_masked_view_from_ids(&pair.tokens, vocab, cfg, &mut rng);
        if code_like || code_like_b {
            code_like_count += 1;
        }
        for &p in &selected_positions {
            target_linear.push((b * cfg.max_seq + p) as u32);
        }
        view_a_buf.extend(view_a_ids);
        view_b_buf.extend(view_b_ids);
        target_buf.extend(target_ids);
    }

    let view_a_ids = Tensor::from_vec(view_a_buf, (batch_size, cfg.max_seq), device)?;
    let view_b_ids = Tensor::from_vec(view_b_buf, (batch_size, cfg.max_seq), device)?;
    let target_ids = Tensor::from_vec(target_buf, (batch_size, cfg.max_seq), device)?;
    let target_count = target_linear.len();
    let target_linear_indices = Tensor::from_vec(target_linear, (target_count,), device)?;

    Ok(AugmentedJepaBatch {
        view_a_ids,
        view_b_ids,
        target_ids,
        target_linear_indices,
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
    tokenize_for_inference_mode(text, TokenizationMode::Default)
}

pub fn tokenize_for_inference_mode(text: &str, mode: TokenizationMode) -> Vec<String> {
    tokenize_with_mode(text, mode)
}

fn is_subword_candidate(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn add_codeaware_reserved_tokens(vocab: &mut Vocab) {
    const CONTROL_TOKENS: &[&str] = &[
        "<nl>",
        "<indent_tab>",
        "<str_lit>",
        "<char_lit>",
        "<num_lit>",
    ];
    const RUST_KEYWORDS: &[&str] = &[
        "fn", "pub", "let", "mut", "impl", "struct", "enum", "trait", "use", "mod", "self", "Self",
        "crate", "super", "where", "match", "if", "else", "for", "while", "loop", "return",
        "async", "await", "move", "const", "static", "type", "dyn", "in", "as", "unsafe", "extern",
        "ref", "Result", "Option", "Some", "None", "Ok", "Err", "Vec", "String", "bool", "str",
        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "f32", "f64",
    ];
    const DIRECT_TOKENS: &[&str] = &[
        "'", "_", "&", "*", "(", ")", "{", "}", "[", "]", "<", ">", ",", ".", ":", ";", "!", "?",
        "+", "-", "/", "%", "=", "|", "^", "#", "@", "$", "~", "\\", "::", "->", "=>", "==", "!=",
        "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "<<=",
        ">>=", "...", "//",
    ];
    for token in CONTROL_TOKENS {
        vocab.add_token(token);
    }
    for indent in 1..=16 {
        vocab.add_token(&format!("<indent_{indent}>"));
    }
    for token in DIRECT_TOKENS {
        vocab.add_token(token);
    }
    for token in RUST_KEYWORDS {
        vocab.add_token(token);
    }
    for ch in 'a'..='z' {
        vocab.add_token(&ch.to_string());
    }
    for ch in 'A'..='Z' {
        vocab.add_token(&ch.to_string());
    }
    for ch in '0'..='9' {
        vocab.add_token(&ch.to_string());
    }
}

fn encode_tokens_with_vocab(tokens: &[String], vocab: &Vocab, mode: TokenizationMode) -> Vec<u32> {
    let mut encoded = Vec::new();
    for token in tokens {
        if let Some(id) = vocab.token_to_id.get(token) {
            encoded.push(*id);
            continue;
        }
        if mode != TokenizationMode::CodeAware {
            encoded.push(vocab.unk_id);
            continue;
        }
        let chars: Vec<char> = token.chars().collect();
        let mut idx = 0usize;
        while idx < chars.len() {
            let mut matched: Option<u32> = None;
            let mut matched_end = idx + 1;
            for end in ((idx + 1)..=chars.len()).rev() {
                let piece: String = chars[idx..end].iter().collect();
                if let Some(id) = vocab.token_to_id.get(&piece) {
                    matched = Some(*id);
                    matched_end = end;
                    break;
                }
            }
            if let Some(id) = matched {
                encoded.push(id);
                idx = matched_end;
            } else {
                encoded.push(vocab.unk_id);
                idx += 1;
            }
        }
    }
    encoded
}

pub fn encode_text_with_vocab_mode(text: &str, vocab: &Vocab, mode: TokenizationMode) -> Vec<u32> {
    let tokens = tokenize_with_mode(text, mode);
    encode_tokens_with_vocab(&tokens, vocab, mode)
}

pub fn encode_world_examples(rows: &[RawWorldExample], vocab: &Vocab) -> Vec<WorldExample> {
    encode_world_examples_with_mode(rows, vocab, TokenizationMode::Default)
}

pub fn encode_world_examples_with_mode(
    rows: &[RawWorldExample],
    vocab: &Vocab,
    mode: TokenizationMode,
) -> Vec<WorldExample> {
    rows.iter()
        .map(|row| WorldExample {
            state_tokens: encode_tokens_with_vocab(
                &tokenize_with_mode(&row.state_text, mode),
                vocab,
                mode,
            ),
            next_tokens: encode_tokens_with_vocab(
                &tokenize_with_mode(&row.next_text, mode),
                vocab,
                mode,
            ),
            action_label: row.action_label,
        })
        .collect()
}

pub fn build_vocab_from_raw_world_file_with_mode(
    path: &PathBuf,
    max_vocab: usize,
    mode: TokenizationMode,
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
        let Some((left, right, _)) = parse_world_line_fields(line) else {
            continue;
        };
        let state_tokens = tokenize_with_mode(&left, mode);
        let next_tokens = tokenize_with_mode(&right, mode);
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
    let vocab_size = max_vocab.saturating_sub(3).min(sorted.len());
    let mut vocab = Vocab::new();
    if mode == TokenizationMode::CodeAware {
        add_codeaware_reserved_tokens(&mut vocab);
        let mut direct_tokens = Vec::new();
        let mut char_counts: HashMap<String, usize> = HashMap::new();
        let mut word_tokens = Vec::new();
        for (token, count) in &sorted {
            if is_subword_candidate(token) {
                word_tokens.push((token.clone(), *count));
            } else {
                direct_tokens.push((token.clone(), *count));
            }
            for ch in token.chars() {
                *char_counts.entry(ch.to_string()).or_insert(0) += *count;
            }
        }
        direct_tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        word_tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut chars: Vec<(String, usize)> = char_counts.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (token, _) in direct_tokens
            .iter()
            .chain(chars.iter())
            .chain(word_tokens.iter())
        {
            if vocab.id_to_token.len() >= max_vocab {
                break;
            }
            vocab.add_token(token);
        }
    } else {
        for (token, _) in sorted.iter().take(vocab_size) {
            vocab.add_token(token);
        }
    }
    let total_tokens: usize = sorted.iter().map(|(_, c)| *c).sum();
    let covered_tokens: usize = sorted
        .iter()
        .filter(|(token, _)| vocab.token_to_id.contains_key(token))
        .map(|(_, c)| *c)
        .sum();
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
        let Some((left, right, _)) = parse_world_line_fields(line) else {
            continue;
        };
        if tokenize_with_mode(&left, mode).is_empty() || tokenize_with_mode(&right, mode).is_empty()
        {
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
    let length = seq.len().max(1);
    while seq.len() < max_seq {
        seq.push(pad_id);
    }
    (seq, length)
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
        let sl = state_lens.get(b).copied().unwrap_or(1).min(max_seq);
        let nl = next_lens.get(b).copied().unwrap_or(1).min(max_seq);

        input_buf.extend(state_v[b].iter().take(sl).copied());
        input_buf.extend(next_v[b].iter().take(nl.saturating_sub(1)).copied());
        let input_len = sl + nl.saturating_sub(1);
        for _ in input_len..decoder_len {
            input_buf.push(pad_id);
        }

        target_buf.extend(std::iter::repeat_n(pad_id, sl.saturating_sub(1)));
        target_buf.extend(next_v[b].iter().take(nl).copied());
        let target_len = sl.saturating_sub(1) + nl;
        for _ in target_len..decoder_len {
            target_buf.push(pad_id);
        }

        mask_buf.extend(std::iter::repeat_n(0.0f32, sl.saturating_sub(1)));
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

#[allow(clippy::type_complexity)]
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

#[cfg(test)]
mod tests {
    use super::{make_decoder_batch, tokenize_for_inference_mode, TokenizationMode};
    use candle_core::{Device, Tensor};

    #[test]
    fn code_tokenizer_splits_snake_case_and_operators() {
        let tokens = tokenize_for_inference_mode(
            "my_special_function(value_one, valueTwo) -> result_value",
            TokenizationMode::CodeAware,
        );
        let expected = vec![
            "my", "_", "special", "_", "function", "(", "value", "_", "one", ",", "value", "Two",
            ")", "->", "result", "_", "value",
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn code_tokenizer_splits_camel_case_and_digits() {
        let tokens = tokenize_for_inference_mode("parseHTTP2Response", TokenizationMode::CodeAware);
        let expected = vec!["parse", "HTTP", "2", "Response"];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn code_tokenizer_keeps_rust_lifetimes_and_char_literals_distinct() {
        let tokens = tokenize_for_inference_mode(
            "fn map<'a>(x: &'a str, c: 'x')",
            TokenizationMode::CodeAware,
        );
        let expected = vec![
            "fn",
            "map",
            "<",
            "'",
            "a",
            ">",
            "(",
            "x",
            ":",
            "&",
            "'",
            "a",
            "str",
            ",",
            "c",
            ":",
            "<char_lit>",
            ")",
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn decoder_batch_only_supervises_continuation_tokens() {
        let device = Device::Cpu;
        let state = Tensor::from_vec(vec![10u32, 11, 12, 0], (1, 4), &device).unwrap();
        let next = Tensor::from_vec(vec![21u32, 22, 0, 0], (1, 4), &device).unwrap();
        let (input, target, mask) =
            make_decoder_batch(&state, &next, &[3], &[2], 4, 0, &device).unwrap();
        assert_eq!(
            input.to_vec2::<u32>().unwrap()[0],
            vec![10, 11, 12, 21, 0, 0, 0, 0]
        );
        assert_eq!(
            target.to_vec2::<u32>().unwrap()[0],
            vec![0, 0, 21, 22, 0, 0, 0, 0]
        );
        assert_eq!(
            mask.to_vec2::<f32>().unwrap()[0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
