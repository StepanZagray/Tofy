//! Download a Hugging Face dataset once via candle-datasets/hf-hub, save as local
//! line-based text so training never re-downloads.

use anyhow::{Context, Result};
use parquet::file::reader::FileReader;
use parquet::record::RowAccessor;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Sanitize dataset id for use in a filename (e.g. "org/name" -> "org_name").
fn sanitize_dataset_id(id: &str) -> String {
    id.replace('/', "_").replace('\\', "_")
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
