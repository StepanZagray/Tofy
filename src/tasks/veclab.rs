use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tasks::prepare::unescape_pair_field;
use crate::tasks::prepare_veclab::{self, PrepareOptions};

pub use crate::tasks::prepare_veclab::{DEFAULT_SEED, FUNCTION_COUNT, SEEN_FUNCTION_MAX};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VeclabTaskRow {
    pub task: String,
    pub completion: String,
    pub function_id: usize,
    pub docs: String,
}

pub fn parse_fn_tag(text: &str) -> Option<usize> {
    let start = text.find("[fn:")? + 4;
    let rest = &text[start..];
    let end = rest.find(']')?;
    rest[..end].parse().ok()
}

/// Removes corpus bookkeeping that must never become a model-visible shortcut.
pub fn model_visible_task(text: &str) -> &str {
    let text = text.trim();
    let Some(rest) = text.strip_prefix("[fn:") else {
        return text;
    };
    rest.find(']')
        .map(|end| rest[end + 1..].trim_start())
        .unwrap_or(text)
}

impl VeclabTaskRow {
    pub fn parse(line: &str) -> Result<Option<Self>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 {
            bail!("veclab row must have two TSV fields (state<TAB>next)");
        }
        let state = unescape_pair_field(fields[0]);
        let completion = unescape_pair_field(fields[1]);
        let function_id = parse_fn_tag(&state).context("missing [fn:NNN] tag in state field")?;
        Ok(Some(Self {
            task: state,
            completion,
            function_id,
            docs: String::new(),
        }))
    }
}

pub fn load_task_rows(path: &Path) -> Result<Vec<VeclabTaskRow>> {
    load_task_rows_with_docs(path, None)
}

pub fn load_docs_map(path: &Path) -> Result<BTreeMap<usize, String>> {
    let mut map = BTreeMap::new();
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 {
            continue;
        }
        let state = unescape_pair_field(fields[0]);
        let doc = unescape_pair_field(fields[1]);
        if let Some(id) = parse_fn_tag(&state) {
            map.insert(id, doc);
        }
    }
    Ok(map)
}

pub fn load_task_rows_with_docs(
    path: &Path,
    docs_path: Option<&Path>,
) -> Result<Vec<VeclabTaskRow>> {
    let docs = docs_path
        .map(load_docs_map)
        .transpose()?
        .unwrap_or_default();
    fs::read_to_string(path)?
        .lines()
        .map(VeclabTaskRow::parse)
        .filter_map(|row| row.transpose())
        .map(|row| {
            row.map(|mut row| {
                if row.docs.is_empty() {
                    row.docs = docs.get(&row.function_id).cloned().unwrap_or_default();
                }
                row
            })
        })
        .collect()
}

pub fn attach_docs(rows: &mut [VeclabTaskRow], docs: &BTreeMap<usize, String>) {
    for row in rows {
        if row.docs.is_empty() {
            row.docs = docs.get(&row.function_id).cloned().unwrap_or_default();
        }
    }
}

pub fn print_vocab_identifier_sanity(vocab: &crate::model::Vocab, docs_path: &Path) -> Result<()> {
    let text = fs::read_to_string(docs_path)?;
    let mut identifiers = Vec::new();
    let mut total_tokens = 0usize;
    let mut unknown_tokens = 0usize;
    let mut long = Vec::new();
    for line in text.lines() {
        let Some(signature) = line.split('\t').next() else {
            continue;
        };
        let Some(name) = signature
            .split_once("func ")
            .and_then(|(_, rest)| rest.split('(').next())
        else {
            continue;
        };
        let ids = vocab.encode_boundless(name);
        total_tokens += ids.len();
        unknown_tokens += ids.iter().filter(|&&id| id == vocab.unk_id).count();
        if ids.len() > 4 {
            long.push(format!("{name}:{}", ids.len()));
        }
        identifiers.push(name.to_string());
    }
    let oov_rate = unknown_tokens as f64 / total_tokens.max(1) as f64;
    println!(
        "veclab tokenizer sanity: identifiers={} tokens={} oov_rate={:.6} over_4_tokens={}",
        identifiers.len(),
        total_tokens,
        oov_rate,
        long.len()
    );
    if !long.is_empty() {
        eprintln!(
            "warning: fictional identifiers exceeding 4 encoder tokens: {}",
            long.join(", ")
        );
    }
    Ok(())
}

pub fn prepare(root: &Path, seed: u64, out: Option<&Path>) -> Result<()> {
    let data_dir = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("data/fictional"));
    prepare_veclab::prepare(PrepareOptions {
        seed,
        out: data_dir,
        root: root.to_path_buf(),
    })
}

pub fn try_run(args: &[String]) -> Result<bool> {
    if matches!(
        args.get(1).map(String::as_str),
        Some("--print-split-stats" | "print-split-stats")
    ) {
        let data_dir = args
            .iter()
            .position(|a| a == "--out")
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/fictional"));
        return prepare_veclab::print_split_stats(&data_dir).map(|_| true);
    }
    if !matches!(
        args.get(1).map(String::as_str),
        Some("--prepare-veclab" | "prepare-veclab")
    ) {
        return Ok(false);
    }
    let seed = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SEED);
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    prepare(&PathBuf::from("."), seed, out.as_deref())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::prepare::escape_pair_field;

    #[test]
    fn tagged_rows_round_trip() -> Result<()> {
        let row = format!(
            "{}\t{}\n",
            escape_pair_field("[fn:042] do something"),
            escape_pair_field("package solution\n\nfunc Solve() {}")
        );
        let parsed = VeclabTaskRow::parse(row.trim())?.expect("row");
        assert_eq!(parsed.function_id, 42);
        assert!(parsed.task.contains("[fn:042]"));
        Ok(())
    }

    #[test]
    fn parse_fn_tag_handles_padding() {
        assert_eq!(parse_fn_tag("[fn:007] query"), Some(7));
        assert_eq!(parse_fn_tag("no tag"), None);
    }

    #[test]
    fn model_visible_task_strips_only_leading_metadata() {
        assert_eq!(
            model_visible_task("[fn:007] Write Solve and mention [fn:008]"),
            "Write Solve and mention [fn:008]"
        );
        assert_eq!(model_visible_task("Write Solve"), "Write Solve");
    }
}
