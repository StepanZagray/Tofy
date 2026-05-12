use anyhow::{Context, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
struct RustDocChunk {
    title: String,
    path: PathBuf,
    symbols: Vec<String>,
    text: String,
    tokens: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct RustDocHit {
    pub(crate) title: String,
    pub(crate) path: PathBuf,
    pub(crate) score: f32,
    pub(crate) text: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RustDocIndex {
    chunks: Vec<RustDocChunk>,
    doc_freq: HashMap<String, usize>,
    inverted: HashMap<String, Vec<usize>>,
}

const DEFAULT_MAX_DOC_FILES: usize = 6000;
const DEFAULT_CHUNK_CHARS: usize = 1800;
const DEFAULT_CHUNK_OVERLAP: usize = 240;

static RUST_DOC_INDEX: OnceLock<Option<RustDocIndex>> = OnceLock::new();

pub(crate) fn rust_docs_enabled() -> bool {
    std::env::var("TOFY_RUST_DOCS")
        .ok()
        .map(|value| {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(true)
}

pub(crate) fn retrieve_rust_docs(query: &str, top_k: usize, char_budget: usize) -> String {
    if !rust_docs_enabled() {
        return String::new();
    }
    let Some(index) = RUST_DOC_INDEX.get_or_init(|| RustDocIndex::load_default().ok()) else {
        return String::new();
    };
    index.format_hits(query, top_k.max(1), char_budget.max(256))
}

pub(crate) fn default_rust_docs_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("TOFY_RUST_DOCS_ROOT") {
        let path = PathBuf::from(root);
        if path.exists() {
            return Some(path);
        }
    }
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })?;
    let sysroot = PathBuf::from(sysroot);
    for candidate in [
        sysroot.join("share/doc/rust/html/std"),
        sysroot.join("share/doc/rust/html/core"),
        sysroot.join("lib/rustlib/src/rust/library"),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

impl RustDocIndex {
    pub(crate) fn load_default() -> Result<Self> {
        let root = default_rust_docs_root().context(
            "Rust docs not found; install with `rustup component add rust-docs rust-src` or set TOFY_RUST_DOCS_ROOT",
        )?;
        Self::load_from_root(&root)
    }

    pub(crate) fn load_from_root(root: &Path) -> Result<Self> {
        let max_files = std::env::var("TOFY_RUST_DOCS_MAX_FILES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_MAX_DOC_FILES)
            .max(1);
        let chunk_chars = std::env::var("TOFY_RUST_DOCS_CHUNK_CHARS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_CHUNK_CHARS)
            .clamp(512, 8000);
        let mut files = Vec::new();
        collect_doc_files(root, &mut files, max_files)?;
        let mut chunks = Vec::new();
        let mut doc_freq = HashMap::<String, usize>::new();
        let mut inverted = HashMap::<String, Vec<usize>>::new();
        for path in files {
            let Ok(raw) = fs::read_to_string(&path) else {
                continue;
            };
            let text = normalize_doc_text(&raw);
            if text.split_whitespace().count() < 12 {
                continue;
            }
            let title = infer_title(&path, &text);
            let symbols = infer_symbols(&path, &title, &text);
            for chunk in chunk_text(&text, chunk_chars) {
                let tokens = token_counts(&chunk);
                if tokens.len() < 6 {
                    continue;
                }
                let chunk_idx = chunks.len();
                for token in tokens.keys() {
                    *doc_freq.entry(token.clone()).or_insert(0) += 1;
                    inverted.entry(token.clone()).or_default().push(chunk_idx);
                }
                for symbol in symbols.iter().map(|value| value.to_ascii_lowercase()) {
                    if symbol.len() >= 3 {
                        inverted.entry(symbol).or_default().push(chunk_idx);
                    }
                }
                chunks.push(RustDocChunk {
                    title: title.clone(),
                    path: path.clone(),
                    symbols: symbols.clone(),
                    text: chunk,
                    tokens,
                });
            }
        }
        Ok(Self {
            chunks,
            doc_freq,
            inverted,
        })
    }

    pub(crate) fn search(&self, query: &str, top_k: usize) -> Vec<RustDocHit> {
        let query_tokens = token_counts(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }
        let query_symbols = extract_symbol_like_terms(query);
        let total_docs = self.chunks.len().max(1) as f32;
        let mut candidate_ids = HashSet::new();
        for token in query_tokens.keys() {
            if let Some(ids) = self.inverted.get(token) {
                candidate_ids.extend(ids.iter().copied());
            }
        }
        for symbol in &query_symbols {
            if let Some(ids) = self.inverted.get(symbol) {
                candidate_ids.extend(ids.iter().copied());
            }
        }
        if candidate_ids.is_empty() {
            return Vec::new();
        }
        let mut scored = candidate_ids
            .iter()
            .filter_map(|chunk_idx| self.chunks.get(*chunk_idx))
            .filter_map(|chunk| {
                let score = score_chunk(
                    chunk,
                    &query_tokens,
                    &query_symbols,
                    &self.doc_freq,
                    total_docs,
                );
                (score > 0.0).then(|| RustDocHit {
                    title: chunk.title.clone(),
                    path: chunk.path.clone(),
                    score,
                    text: chunk.text.clone(),
                })
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored.truncate(top_k.max(1));
        scored
    }

    pub(crate) fn format_hits(&self, query: &str, top_k: usize, char_budget: usize) -> String {
        let hits = self.search(query, top_k);
        if hits.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str("<ctx:rust_docs>\n");
        out.push_str("Retrieved from installed Rust docs. Use only if relevant.\n");
        let mut remaining = char_budget.saturating_sub(out.len());
        for (idx, hit) in hits.iter().enumerate() {
            if remaining < 160 {
                break;
            }
            let header = format!(
                "\n[{}] {} score={:.2} path={}\n",
                idx + 1,
                hit.title,
                hit.score,
                hit.path.display()
            );
            out.push_str(&header);
            remaining = char_budget.saturating_sub(out.len());
            let excerpt = excerpt_for_budget(&hit.text, remaining.saturating_sub(32));
            out.push_str(&excerpt);
            out.push('\n');
            remaining = char_budget.saturating_sub(out.len());
        }
        out.push_str("</ctx:rust_docs>");
        out
    }

    pub(crate) fn jepa_rows(&self, max_rows: usize) -> Vec<String> {
        self.chunks
            .iter()
            .take(max_rows.max(1))
            .map(|chunk| chunk.text.replace(['\t', '\n'], " "))
            .collect()
    }

    pub(crate) fn tool_pairs(&self, max_rows: usize) -> Vec<String> {
        let mut rows = Vec::new();
        for chunk in self.chunks.iter().take(max_rows.max(1)) {
            let symbol = chunk
                .symbols
                .iter()
                .find(|value| value.len() >= 3)
                .cloned()
                .unwrap_or_else(|| chunk.title.clone());
            let query = format!(
                "<action:fetch_docs>\n<tool:fetch_docs>\nNeed installed Rust docs for `{}` before writing code.\nQuery: {}",
                symbol, symbol
            )
            .replace(['\t', '\n'], " ");
            let result = format!(
                "<ctx:rust_docs>\n[1] {} path={}\n{}\n</ctx:rust_docs>",
                chunk.title,
                chunk.path.display(),
                excerpt_for_budget(&chunk.text, 1000)
            )
            .replace(['\t', '\n'], " ");
            rows.push(format!("{query}\t{result}\tfetch_docs"));
        }
        rows
    }
}

fn collect_doc_files(root: &Path, out: &mut Vec<PathBuf>, max_files: usize) -> Result<()> {
    if out.len() >= max_files {
        return Ok(());
    }
    for entry in fs::read_dir(root).with_context(|| format!("read docs dir {}", root.display()))? {
        if out.len() >= max_files {
            break;
        }
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
            if matches!(name, "implementors" | "src" | "static.files") {
                continue;
            }
            collect_doc_files(&path, out, max_files)?;
        } else if is_doc_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_doc_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("html" | "md" | "rs")
    )
}

fn normalize_doc_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len().min(16_384));
    let mut in_tag = false;
    let mut in_entity = false;
    let mut entity = String::new();
    for ch in raw.chars() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                out.push(' ');
            }
            continue;
        }
        if in_entity {
            if ch == ';' {
                out.push(match entity.as_str() {
                    "lt" => '<',
                    "gt" => '>',
                    "amp" => '&',
                    "quot" => '"',
                    "apos" => '\'',
                    "nbsp" => ' ',
                    _ => ' ',
                });
                entity.clear();
                in_entity = false;
            } else if entity.len() < 12 {
                entity.push(ch);
            } else {
                entity.clear();
                in_entity = false;
            }
            continue;
        }
        match ch {
            '<' => in_tag = true,
            '&' => in_entity = true,
            '\n' | '\r' | '\t' => out.push(' '),
            _ => out.push(ch),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn infer_title(path: &Path, text: &str) -> String {
    let file = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("rust-doc");
    let prefix = text
        .split(['.', '!', '?'])
        .next()
        .unwrap_or(file)
        .split_whitespace()
        .take(10)
        .collect::<Vec<_>>()
        .join(" ");
    if prefix.len() >= 8 {
        prefix
    } else {
        file.replace('-', "::")
    }
}

fn infer_symbols(path: &Path, title: &str, text: &str) -> Vec<String> {
    let mut set = HashSet::new();
    for value in path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
    {
        for token in extract_symbol_like_terms(value) {
            set.insert(token);
        }
    }
    for token in extract_symbol_like_terms(title) {
        set.insert(token);
    }
    for token in extract_symbol_like_terms(&text.chars().take(800).collect::<String>()) {
        set.insert(token);
    }
    set.into_iter().collect()
}

fn chunk_text(text: &str, chunk_chars: usize) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= chunk_chars {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let overlap = DEFAULT_CHUNK_OVERLAP.min(chunk_chars / 3);
    while start < chars.len() {
        let mut end = (start + chunk_chars).min(chars.len());
        while end < chars.len() && end > start + chunk_chars / 2 && !chars[end - 1].is_whitespace()
        {
            end -= 1;
        }
        chunks.push(chars[start..end].iter().collect::<String>());
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    chunks
}

fn token_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in text
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == ':'))
        .filter(|token| token.len() >= 2)
    {
        let token = token.trim_matches(':').to_ascii_lowercase();
        if token.len() >= 2 && !is_stopword(&token) {
            *counts.entry(token).or_insert(0) += 1;
        }
    }
    counts
}

fn extract_symbol_like_terms(text: &str) -> HashSet<String> {
    text.split(|ch: char| {
        !(ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' || ch == '<' || ch == '>')
    })
    .filter(|token| token.len() >= 3)
    .flat_map(|token| {
        let clean = token
            .trim_matches(|ch: char| ch == '<' || ch == '>' || ch == ':' || ch == ',')
            .to_string();
        let mut terms = Vec::new();
        if clean.contains("::") {
            terms.push(clean.to_ascii_lowercase());
            if let Some(last) = clean.rsplit("::").next() {
                terms.push(last.to_ascii_lowercase());
            }
        } else if clean.chars().any(|ch| ch.is_ascii_uppercase()) || clean.contains('_') {
            terms.push(clean.to_ascii_lowercase());
        }
        terms
    })
    .collect()
}

fn score_chunk(
    chunk: &RustDocChunk,
    query_tokens: &HashMap<String, usize>,
    query_symbols: &HashSet<String>,
    doc_freq: &HashMap<String, usize>,
    total_docs: f32,
) -> f32 {
    let mut score = 0.0f32;
    let doc_len = chunk.tokens.values().sum::<usize>().max(1) as f32;
    for (token, qtf) in query_tokens {
        let Some(tf) = chunk.tokens.get(token).copied() else {
            continue;
        };
        let df = doc_freq.get(token).copied().unwrap_or(1) as f32;
        let idf = ((total_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.1);
        let tf = tf as f32;
        let bm25 = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_len / 180.0));
        score += *qtf as f32 * idf * bm25;
    }
    let chunk_symbols = chunk
        .symbols
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    for symbol in query_symbols {
        if chunk_symbols.contains(symbol) {
            score += 3.5;
        }
    }
    if query_tokens
        .keys()
        .any(|token| chunk.title.to_ascii_lowercase().contains(token))
    {
        score += 1.0;
    }
    score
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "you"
            | "with"
            | "that"
            | "this"
            | "from"
            | "into"
            | "rust"
            | "code"
            | "function"
            | "implement"
            | "return"
            | "only"
            | "use"
            | "pub"
            | "fn"
    )
}

fn excerpt_for_budget(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_string();
    }
    let mut out = text
        .chars()
        .take(budget.saturating_sub(4))
        .collect::<String>();
    out.push_str(" ...");
    out
}
