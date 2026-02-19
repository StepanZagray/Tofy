//! Download a Hugging Face dataset once via candle-datasets/hf-hub, save as local
//! line-based text so training never re-downloads.

use anyhow::{Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Sanitize dataset id for use in a filename (e.g. "org/name" -> "org_name").
fn sanitize_dataset_id(id: &str) -> String {
    id.replace('/', "_").replace('\\', "_")
}

/// Minimum character length for a paragraph to be written (avoids junk fragments).
const MIN_PARAGRAPH_CHARS: usize = 15;
/// Maximum character length per paragraph line (avoids huge single lines).
const MAX_PARAGRAPH_CHARS: usize = 50_000;

/// English Wikipedia subset on Hugging Face (wikimedia/wikipedia): only parquet files under this path are used.
const WIKIPEDIA_EN_SUBSET: &str = "20231101.en/";
/// Same revision candle-datasets uses for parquet datasets.
const PARQUET_REVISION: &str = "refs/convert/parquet";
const ULTRACHAT_DATASET_ID: &str = "HuggingFaceH4/ultrachat_200k";

/// List English Wikipedia parquet filenames (Hub API only, no parquet download). Returns (sorted filenames, count).
fn wikipedia_english_parquet_list(
    api: &hf_hub::api::sync::Api,
) -> Result<(Vec<String>, usize)> {
    use hf_hub::{Repo, RepoType};

    let repo = Repo::with_revision(
        "wikimedia/wikipedia".to_string(),
        RepoType::Dataset,
        PARQUET_REVISION.to_string(),
    );
    let api_repo = api.repo(repo);
    let info = api_repo
        .info()
        .map_err(|e| anyhow::anyhow!("hub info: {}", e))?;

    let mut list: Vec<String> = info
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.contains(WIKIPEDIA_EN_SUBSET) && s.rfilename.ends_with(".parquet"))
        .map(|s| s.rfilename)
        .collect();
    list.sort();
    let n = list.len();
    Ok((list, n))
}

/// Open parquet readers for the given rfilenames (downloads files via Hub).
fn wikipedia_english_parquet_readers(
    api: &hf_hub::api::sync::Api,
    parquet_rfilenames: &[String],
) -> Result<Vec<SerializedFileReader<File>>> {
    use hf_hub::{Repo, RepoType};

    let repo = Repo::with_revision(
        "wikimedia/wikipedia".to_string(),
        RepoType::Dataset,
        PARQUET_REVISION.to_string(),
    );
    let api_repo = api.repo(repo);

    let mut readers = Vec::with_capacity(parquet_rfilenames.len());
    for rfilename in parquet_rfilenames {
        let local_path = api_repo
            .get(rfilename)
            .map_err(|e| anyhow::anyhow!("hub get: {}", e))?;
        let file = File::open(local_path).context("open parquet file")?;
        readers.push(SerializedFileReader::new(file).context("parquet reader")?);
    }
    Ok(readers)
}

/// Ensure the Hugging Face dataset is available as a local text file.
/// If `cache_dir/cached_<id>.txt` already exists, return that path (no download).
/// Otherwise download via hf-hub (cached by hf-hub), convert parquet rows to
/// one phrase per line, write to that file, then return the path.
pub fn ensure_hub_dataset_cached(dataset_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    let name = sanitize_dataset_id(dataset_id);
    let cache_file = cache_dir.join(format!("cached_{}.txt", name));

    if cache_file.exists() {
        eprintln!(
            "Using cached dataset at {} (delete to re-download)",
            cache_file.display()
        );
        return Ok(cache_file);
    }

    eprintln!(
        "Downloading dataset '{}' once (saving to {})...",
        dataset_id,
        cache_file.display()
    );

    let api = hf_hub::api::sync::Api::new().context("hf-hub API")?;
    let readers = candle_datasets::hub::from_hub(&api, dataset_id.to_string())
        .map_err(|e| anyhow::anyhow!("hub: {}", e))?;

    if readers.is_empty() {
        anyhow::bail!("dataset '{}' has no parquet files", dataset_id);
    }

    if let Some(parent) = cache_file.parent() {
        std::fs::create_dir_all(parent).context("create cache dir")?;
    }
    let mut f = BufWriter::new(File::create(&cache_file).context("create cache file")?);

    let mut line_count: usize = 0;
    for reader in &readers {
        let iter = reader
            .get_row_iter(None)
            .map_err(|e| anyhow::anyhow!("parquet row iter: {}", e))?;
        for row_result in iter {
            let row = row_result.map_err(|e| anyhow::anyhow!("parquet row: {}", e))?;
            let n = extract_lines_from_row(&row, &mut f)?;
            line_count += n;
        }
    }

    f.flush().context("flush cache file")?;
    eprintln!("Wrote {} lines to {}", line_count, cache_file.display());

    Ok(cache_file)
}

/// Ensure English Wikipedia is available as local text with **one paragraph per line**.
/// Uses `wikimedia/wikipedia` and downloads only the **English** subset (20231101.en).
/// Cache file is named `cached_<id>_<N>.txt` where N = number of parquet files used (e.g. cached_wikimedia_wikipedia_64.txt).
/// Set env `JEPA_WIKI_MAX_FILES` to limit how many parquet files to download (e.g. 1 for ~400MB).
pub fn ensure_hub_wikipedia_cached(dataset_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    let name = sanitize_dataset_id(dataset_id);

    let api = hf_hub::api::sync::Api::new().context("hf-hub API")?;
    let (mut rfilenames, total_parquets) = wikipedia_english_parquet_list(&api)?;

    if rfilenames.is_empty() {
        anyhow::bail!(
            "no English Wikipedia parquet files found (expected path containing '{}')",
            WIKIPEDIA_EN_SUBSET.trim_end_matches('/')
        );
    }

    let max_files: Option<usize> = std::env::var("JEPA_WIKI_MAX_FILES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0);
    if let Some(cap) = max_files {
        rfilenames.truncate(cap.min(rfilenames.len()));
        eprintln!(
            "JEPA_WIKI_MAX_FILES={}: using first {} of {} parquet file(s).",
            cap,
            rfilenames.len(),
            total_parquets
        );
    }

    let num_parquets_used = rfilenames.len();
    let cache_file = cache_dir.join(format!("cached_{}_{}.txt", name, num_parquets_used));

    if cache_file.exists() {
        eprintln!(
            "Using cached Wikipedia at {} (delete to re-download)",
            cache_file.display()
        );
        return Ok(cache_file);
    }

    eprintln!(
        "English Wikipedia ({}): downloading {} parquet file(s) → {}...",
        WIKIPEDIA_EN_SUBSET.trim_end_matches('/'),
        num_parquets_used,
        cache_file.display()
    );

    let readers = wikipedia_english_parquet_readers(&api, &rfilenames)?;

    if let Some(parent) = cache_file.parent() {
        std::fs::create_dir_all(parent).context("create cache dir")?;
    }
    let mut f = BufWriter::new(File::create(&cache_file).context("create cache file")?);

    let mut line_count: usize = 0;
    for reader in &readers {
        let iter = reader
            .get_row_iter(None)
            .map_err(|e| anyhow::anyhow!("parquet row iter: {}", e))?;
        for row_result in iter {
            let row = row_result.map_err(|e| anyhow::anyhow!("parquet row: {}", e))?;
            let n = write_paragraphs_from_row(&row, &mut f)?;
            line_count += n;
        }
    }

    f.flush().context("flush cache file")?;
    if line_count == 0 {
        anyhow::bail!(
            "dataset '{}' did not yield any paragraphs (expected a text column with article body)",
            dataset_id
        );
    }
    eprintln!(
        "Wrote {} paragraphs to {}",
        line_count,
        cache_file.display()
    );
    Ok(cache_file)
}

/// Extract article text from row (longest string field), split by `\n\n`, write each
/// non-empty paragraph as one line.
fn write_paragraphs_from_row<W: Write>(
    row: &parquet::record::Row,
    w: &mut W,
) -> Result<usize> {
    let mut best: Option<&str> = None;
    let n = row.len();
    for i in 0..n {
        if let Ok(s) = row.get_string(i) {
            let s = s.trim();
            if s.len() >= MIN_PARAGRAPH_CHARS && s.len() <= MAX_PARAGRAPH_CHARS {
                if best.map(|b| b.len() < s.len()).unwrap_or(true) {
                    best = Some(s);
                }
            }
        }
    }
    let Some(article) = best else {
        return Ok(0);
    };
    let mut count = 0;
    for block in article.split("\n\n") {
        let para = block.trim().replace('\n', " ");
        if para.len() >= MIN_PARAGRAPH_CHARS && para.len() <= MAX_PARAGRAPH_CHARS {
            writeln!(w, "{}", para).context("write paragraph line")?;
            count += 1;
        }
    }
    Ok(count)
}

/// Try to extract one or more text lines from a parquet row.
/// Supports: a single "text" (or first string) column, or a list column (e.g. "dialog").
fn extract_lines_from_row<W: Write>(row: &parquet::record::Row, w: &mut W) -> Result<usize> {
    use parquet::record::ListAccessor;
    let mut count = 0;
    let n = row.len();
    for i in 0..n {
        if let Ok(s) = row.get_string(i) {
            let s = s.trim();
            if !s.is_empty() && s.len() < 50_000 {
                writeln!(w, "{}", s).context("write line")?;
                count += 1;
            }
            return Ok(count);
        }
        if let Ok(list) = row.get_list(i) {
            let list_len = list.len();
            for j in 0..list_len {
                if let Ok(s) = list.get_string(j) {
                    let s = s.trim();
                    if !s.is_empty() && s.len() < 50_000 {
                        writeln!(w, "{}", s).context("write line")?;
                        count += 1;
                    }
                }
            }
            return Ok(count);
        }
    }
    Ok(count)
}

/// Prepare UltraChat context->next_turn pairs directly in Rust from Hub parquet.
/// Writes `context<TAB>next_turn` rows to `output_path`.
pub fn prepare_ultrachat_pairs(
    output_path: &Path,
    context_window: usize,
    min_tokens: usize,
    max_rows: Option<usize>,
) -> Result<usize> {
    let api = hf_hub::api::sync::Api::new().context("hf-hub API")?;
    let readers = candle_datasets::hub::from_hub(&api, ULTRACHAT_DATASET_ID.to_string())
        .map_err(|e| anyhow::anyhow!("hub: {}", e))?;
    if readers.is_empty() {
        anyhow::bail!("dataset '{}' has no parquet files", ULTRACHAT_DATASET_ID);
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).context("create output dir")?;
    }
    let mut out = BufWriter::new(File::create(output_path).context("create output file")?);

    let mut written = 0usize;
    for reader in &readers {
        let iter = reader
            .get_row_iter(None)
            .map_err(|e| anyhow::anyhow!("parquet row iter: {}", e))?;
        for row_result in iter {
            let row = row_result.map_err(|e| anyhow::anyhow!("parquet row: {}", e))?;
            let Some(turns) = extract_turns_from_row(&row) else {
                continue;
            };
            if turns.len() < 2 {
                continue;
            }

            for i in 0..turns.len() {
                if turns[i].0 != "assistant" {
                    continue;
                }
                let target = sanitize_field(&turns[i].1);
                if target.split_whitespace().count() < min_tokens {
                    continue;
                }
                let start = i.saturating_sub(context_window);
                let mut ctx_lines = Vec::new();
                for (role, text) in turns[start..i].iter() {
                    let role_name = match role.as_str() {
                        "assistant" => "Assistant",
                        "user" => "User",
                        _ => "Other",
                    };
                    let clean = sanitize_field(text);
                    if !clean.is_empty() {
                        ctx_lines.push(format!("{role_name}: {clean}"));
                    }
                }
                if ctx_lines.is_empty() {
                    continue;
                }
                let context = ctx_lines.join("\n");
                if context.split_whitespace().count() < min_tokens {
                    continue;
                }
                writeln!(out, "{}\t{}", context, target).context("write ultrachat pair")?;
                written += 1;
                if let Some(max_rows) = max_rows {
                    if written >= max_rows {
                        out.flush().context("flush output")?;
                        return Ok(written);
                    }
                }
            }

        }
    }
    out.flush().context("flush output")?;
    Ok(written)
}

fn sanitize_field(s: &str) -> String {
    s.trim().replace('\t', " ")
}

fn extract_turns_from_row(row: &parquet::record::Row) -> Option<Vec<(String, String)>> {
    use parquet::record::ListAccessor;

    for i in 0..row.len() {
        let Ok(list) = row.get_list(i) else {
            continue;
        };
        let mut turns = Vec::new();
        for j in 0..list.len() {
            let Ok(group) = list.get_group(j) else {
                continue;
            };
            if let Some(turn) = extract_turn_from_group(group) {
                turns.push(turn);
            }
        }
        if !turns.is_empty() {
            return Some(turns);
        }
    }
    None
}

fn extract_turn_from_group(group: &parquet::record::Row) -> Option<(String, String)> {
    let mut role: Option<String> = None;
    let mut content: Option<String> = None;
    for i in 0..group.len() {
        let Ok(s) = group.get_string(i) else {
            continue;
        };
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        let low = s.to_lowercase();
        if role.is_none() && matches!(low.as_str(), "assistant" | "gpt" | "bot" | "model") {
            role = Some("assistant".to_string());
            continue;
        }
        if role.is_none() && matches!(low.as_str(), "user" | "human" | "prompt" | "instruction")
        {
            role = Some("user".to_string());
            continue;
        }
        if content.is_none() {
            content = Some(s.to_string());
        }
    }

    match (role, content) {
        (Some(r), Some(c)) => Some((r, c)),
        _ => None,
    }
}
