use anyhow::{bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use rand::{seq::SliceRandom, RngExt, SeedableRng};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ARTIFACT_CACHE_VERSION: u32 = 1;
const PARQUET_REVISION: &str = "refs/convert/parquet";
const GITHUB_TOP_CODE_DATASET_ID: &str = "ronantakizawa/github-top-code";
const CASUAL_CONVERSATION_DATASET_ID: &str = "SohamGhadge/casual-conversation";
const SCIQ_DATASET_ID: &str = "sciq";
const SQUAD_V2_DATASET_ID: &str = "rajpurkar/squad_v2";
const DEFAULT_GITHUB_LANGUAGES: &[&str] = &[
    "Rust",
    "TypeScript",
    "Go",
    "JavaScript",
    "C/C++ Header",
    "C",
    "C++",
    "TSX",
    "CSS",
    "HTML",
];
const CODE_EVAL_SUITE_JSONL: &str = include_str!("../../eval/code_assistant_rust_hard.jsonl");
const FORBIDDEN_DTYPE_PATTERNS: &[&str] = &[
    ".to_scalar::<f32>()",
    ".to_vec1::<f32>()",
    ".to_vec2::<f32>()",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SourceFingerprint {
    path: String,
    len: u64,
    modified_unix_nanos: u128,
    content_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ArtifactManifest {
    version: u32,
    kind: String,
    output_path: String,
    params: Value,
    inputs: Vec<SourceFingerprint>,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeStage {
    All,
    Latent,
    World,
    Decoder,
}

#[derive(Clone, Debug)]
struct OomProbeArgs {
    stage: ProbeStage,
    quick: bool,
    probe_dir: Option<PathBuf>,
    keep_local_models: bool,
    build: bool,
    binary: PathBuf,
    dtype: String,
    sample_interval_sec: f64,
    min_headroom_mb: i64,
    max_late_growth_mb: i64,
    data_rows: usize,
    run_group: String,
    dim: usize,
    max_seq: usize,
    layers: usize,
    heads: usize,
    bridge_dim: usize,
    planner_slots: usize,
    vocab: usize,
    latent_steps: usize,
    latent_batch: usize,
    latent_accum: usize,
    world_steps: usize,
    world_batch: usize,
    world_accum: usize,
    world_warmup_batch: usize,
    world_warmup_accum: usize,
    world_warmup_steps: Option<usize>,
    world_lambda: f64,
    world_lr: f64,
    world_action_loss_weight: f64,
    decoder_steps: usize,
    decoder_batch: usize,
    decoder_accum: usize,
    decoder_max_seq: usize,
    decoder_max_vocab: usize,
    setup_latent_steps: usize,
    setup_world_steps: usize,
    latent_model: Option<PathBuf>,
    encoder_vocab: Option<PathBuf>,
    world_model: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct VramSample {
    used_mb: i64,
    free_mb: i64,
    total_mb: i64,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ProbeResult {
    name: String,
    cmd: Vec<String>,
    return_code: i32,
    seconds: f64,
    log: String,
    oom: bool,
    headroom_ok: bool,
    growth_ok: bool,
    passed: bool,
    min_headroom_mb: i64,
    max_late_growth_mb: i64,
    sample_count: usize,
    peak_used_mb: Option<i64>,
    min_free_mb: Option<i64>,
    total_mb: Option<i64>,
    peak_fraction: Option<f64>,
    late_growth_mb: Option<i64>,
    first_used_mb: Option<i64>,
    last_used_mb: Option<i64>,
    samples_path: String,
    tail: String,
}

pub fn try_run_prepare(args: &[String]) -> Result<bool> {
    if args.len() < 2 {
        return Ok(false);
    }
    match args[1].as_str() {
        "--prepare-encoder-corpus" | "prepare-encoder-corpus" => {
            run_prepare_encoder_corpus(&args[2..])?;
            Ok(true)
        }
        "--prepare-rust-by-practice" | "prepare-rust-by-practice" => {
            run_prepare_rust_by_practice(&args[2..])?;
            Ok(true)
        }
        "--prepare-rust-docs" | "prepare-rust-docs" => {
            run_prepare_rust_docs(&args[2..])?;
            Ok(true)
        }
        "--prepare-rust-doc-trajectories" | "prepare-rust-doc-trajectories" => {
            run_prepare_rust_doc_trajectories(&args[2..])?;
            Ok(true)
        }
        "--prepare-github-top-code" | "prepare-github-top-code" => {
            run_prepare_github_top_code(&args[2..])?;
            Ok(true)
        }
        "--prepare-rust-function-tasks" | "prepare-rust-function-tasks" => {
            run_prepare_rust_function_tasks(&args[2..])?;
            Ok(true)
        }
        "--prepare-rust-repair-tasks" | "prepare-rust-repair-tasks" => {
            run_prepare_rust_repair_tasks(&args[2..])?;
            Ok(true)
        }
        "--prepare-world-mix" | "prepare-world-mix" => {
            run_prepare_world_mix(&args[2..])?;
            Ok(true)
        }
        "--prepare-code-poc-mix" | "prepare-code-poc-mix" => {
            run_prepare_code_poc_mix(&args[2..])?;
            Ok(true)
        }
        "--prepare-expert-pairs" | "prepare-expert-pairs" => {
            run_prepare_expert_pairs(&args[2..])?;
            Ok(true)
        }
        "--prepare-casual-conversation" | "prepare-casual-conversation" => {
            run_prepare_casual_conversation(&args[2..])?;
            Ok(true)
        }
        "--generate-code-eval-suite" | "generate-code-eval-suite" => {
            run_generate_code_eval_suite(&args[2..])?;
            Ok(true)
        }
        "--check-dtype-discipline" | "check-dtype-discipline" => {
            run_check_dtype_discipline(&args[2..])?;
            Ok(true)
        }
        "--convert-jsonl-context-response-to-tsv" | "convert-jsonl-context-response-to-tsv" => {
            run_convert_jsonl_context_response_to_tsv(&args[2..])?;
            Ok(true)
        }
        "--sustained-oom-probe" | "sustained-oom-probe" => {
            run_sustained_oom_probe(&args[2..])?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_flag_value(args: &[String], index: usize, flag: &str) -> Result<String> {
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn parse_path_value(args: &[String], index: usize, flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(parse_flag_value(args, index, flag)?))
}

fn parse_usize_value(args: &[String], index: usize, flag: &str) -> Result<usize> {
    parse_flag_value(args, index, flag)?
        .parse()
        .with_context(|| format!("{flag} must be an integer"))
}

fn parse_f64_value(args: &[String], index: usize, flag: &str) -> Result<f64> {
    parse_flag_value(args, index, flag)?
        .parse()
        .with_context(|| format!("{flag} must be a number"))
}

fn parse_i64_value(args: &[String], index: usize, flag: &str) -> Result<i64> {
    parse_flag_value(args, index, flag)?
        .parse()
        .with_context(|| format!("{flag} must be an integer"))
}

fn output_manifest_path(output_path: &Path) -> PathBuf {
    let suffix = output_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|ext| format!(".{ext}.manifest.json"))
        .unwrap_or_else(|| ".manifest.json".to_string());
    output_path.with_extension(suffix.trim_start_matches('.'))
}

fn load_manifest(path: &Path) -> Option<ArtifactManifest> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let tmp = path.with_extension(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| format!("{ext}.tmp"))
            .unwrap_or_else(|| "tmp".to_string()),
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&tmp)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn write_text_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = path.with_extension(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| format!("{ext}.tmp"))
            .unwrap_or_else(|| "tmp".to_string()),
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&tmp)?);
    writer.write_all(content.as_bytes())?;
    writer.flush()?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn source_fingerprint(path: &Path) -> Result<SourceFingerprint> {
    let metadata = fs::metadata(path).with_context(|| format!("stat input {}", path.display()))?;
    let modified = metadata
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let content_hash = sha256_file(path)?;
    Ok(SourceFingerprint {
        path: path.to_string_lossy().to_string(),
        len: metadata.len(),
        modified_unix_nanos: modified,
        content_hash,
    })
}

fn stat_probe(path: &Path) -> Result<(u64, u128)> {
    let metadata = fs::metadata(path).with_context(|| format!("stat input {}", path.display()))?;
    let modified = metadata
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok((metadata.len(), modified))
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn artifact_cache_hit(
    kind: &str,
    output_path: &Path,
    input_paths: &[PathBuf],
    params: &Value,
) -> Result<Option<ArtifactManifest>> {
    let manifest_path = output_manifest_path(output_path);
    let Some(manifest) = load_manifest(&manifest_path) else {
        return Ok(None);
    };
    if !output_path.exists()
        || manifest.version != ARTIFACT_CACHE_VERSION
        || manifest.kind != kind
        || manifest.output_path != output_path.to_string_lossy()
        || manifest.params != *params
        || manifest.inputs.len() != input_paths.len()
    {
        return Ok(None);
    }
    for (path, stored) in input_paths.iter().zip(manifest.inputs.iter()) {
        if !path.exists() || stored.path != path.to_string_lossy() {
            return Ok(None);
        }
        let (len, modified) = stat_probe(path)?;
        if len == stored.len && modified == stored.modified_unix_nanos {
            continue;
        }
        if sha256_file(path)? != stored.content_hash {
            return Ok(None);
        }
    }
    Ok(Some(manifest))
}

fn write_artifact_manifest(
    kind: &str,
    output_path: &Path,
    input_paths: &[PathBuf],
    params: Value,
    rows: usize,
) -> Result<()> {
    let mut inputs = Vec::with_capacity(input_paths.len());
    for path in input_paths {
        inputs.push(source_fingerprint(path)?);
    }
    let manifest = ArtifactManifest {
        version: ARTIFACT_CACHE_VERSION,
        kind: kind.to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        params,
        inputs,
        rows,
    };
    write_json_atomic(&output_manifest_path(output_path), &manifest)
}

fn unescape_pair_field(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek().copied() {
                Some('n') => {
                    out.push('\n');
                    chars.next();
                    continue;
                }
                Some('t') => {
                    out.push('\t');
                    chars.next();
                    continue;
                }
                Some('r') => {
                    out.push('\r');
                    chars.next();
                    continue;
                }
                Some('\\') => {
                    out.push('\\');
                    chars.next();
                    continue;
                }
                _ => {}
            }
        }
        out.push(ch);
    }
    out
}

fn escape_pair_field(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            '\t' => out.push_str("    "),
            _ => out.push(ch),
        }
    }
    out.trim().to_string()
}

fn flatten_for_encoder(text: &str) -> String {
    unescape_pair_field(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn run_prepare_encoder_corpus(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut force = false;
    let mut inputs = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value if value.starts_with("--") => bail!("unknown flag: {value}"),
            value => {
                inputs.push(PathBuf::from(value));
                i += 1;
            }
        }
    }
    let output = output.context("--output is required")?;
    if inputs.is_empty() {
        bail!("prepare-encoder-corpus requires one or more input files");
    }
    let params = json!({});
    if !force {
        if let Some(manifest) = artifact_cache_hit("encoder_corpus", &output, &inputs, &params)? {
            println!(
                "Encoder corpus cache hit: {} (lines={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }

    let mut written = 0usize;
    let tmp = output.with_extension("txt.tmp");
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&tmp)?);
    for path in &inputs {
        let reader = BufReader::new(File::open(path)?);
        for raw in reader.lines() {
            let line = raw?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((left, right)) = split_pair_line(trimmed) {
                for part in [left, right] {
                    if !part.trim().is_empty() {
                        writeln!(out, "{}", flatten_for_encoder(part.trim()))?;
                        written += 1;
                    }
                }
            } else {
                writeln!(out, "{}", flatten_for_encoder(trimmed))?;
                written += 1;
            }
        }
    }
    out.flush()?;
    fs::rename(tmp, &output)?;
    write_artifact_manifest("encoder_corpus", &output, &inputs, params, written)?;
    println!("Wrote {written} encoder lines to {}", output.display());
    Ok(())
}

fn split_pair_line(line: &str) -> Option<(&str, &str)> {
    if let Some((left, right)) = line.split_once('\t') {
        return Some((left, right));
    }
    if let Some((left, right)) = line.split_once("|||") {
        return Some((left, right));
    }
    None
}

fn run_prepare_rust_by_practice(args: &[String]) -> Result<()> {
    let mut input = PathBuf::from("data/sunface_rust-by-practice_en");
    let mut output = None;
    let mut mode = None;
    let mut no_split_headings = false;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = parse_path_value(args, i, "--input")?;
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--mode" => {
                mode = Some(parse_flag_value(args, i, "--mode")?);
                i += 2;
            }
            "--no-split-headings" => {
                no_split_headings = true;
                i += 1;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    let mode = mode.context("--mode is required (jepa|pairs)")?;
    let files = iter_md_files(&input)?;
    let input_paths = files.clone();
    let params = json!({
        "mode": mode,
        "split_headings": !no_split_headings,
    });
    if !force {
        if let Some(manifest) =
            artifact_cache_hit("rust_by_practice", &output, &input_paths, &params)?
        {
            println!(
                "Rust-by-Practice cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let mut rows = Vec::new();
    match mode.as_str() {
        "jepa" => {
            for path in &files {
                let text = fs::read_to_string(path)?.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                if no_split_headings {
                    let line = text.replace(['\t', '\n'], " ");
                    if line.split_whitespace().count() >= 5 {
                        rows.push(line);
                    }
                } else {
                    for chunk in split_markdown_by_heading(&text) {
                        if chunk.split_whitespace().count() >= 5 {
                            rows.push(chunk.replace(['\n', '\t'], " "));
                        }
                    }
                }
            }
        }
        "pairs" => {
            for path in &files {
                let text = fs::read_to_string(path)?.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                let chunks = split_markdown_by_heading(&text);
                for pair in chunks.windows(2) {
                    let prev = pair[0].replace('\t', " ");
                    let next = pair[1].replace('\t', " ");
                    if prev.split_whitespace().count() >= 3 && next.split_whitespace().count() >= 3
                    {
                        rows.push(format!("{prev}\t{next}"));
                    }
                }
            }
        }
        _ => bail!("--mode must be jepa or pairs"),
    }
    let mut content = String::new();
    for row in &rows {
        content.push_str(row.trim());
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest(
        "rust_by_practice",
        &output,
        &input_paths,
        params,
        rows.len(),
    )?;
    println!("Wrote {} lines to {}", rows.len(), output.display());
    Ok(())
}

fn run_prepare_rust_docs(args: &[String]) -> Result<()> {
    let mut input = crate::tasks::rust_docs::default_rust_docs_root();
    let mut output = None;
    let mut mode = None;
    let mut max_rows = 20_000usize;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(parse_path_value(args, i, "--input")?);
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--mode" => {
                mode = Some(parse_flag_value(args, i, "--mode")?);
                i += 2;
            }
            "--max-rows" => {
                max_rows = parse_usize_value(args, i, "--max-rows")?.max(1);
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let input =
        input.context("Rust docs root not found; install rust-docs/rust-src or pass --input")?;
    let output = output.context("--output is required")?;
    let mode = mode.context("--mode is required (jepa|tool-pairs)")?;
    let params = json!({
        "mode": mode,
        "max_rows": max_rows,
        "source": "installed-rust-docs",
        "input": input.to_string_lossy(),
    });
    let inputs = Vec::new();
    if !force {
        if let Some(manifest) = artifact_cache_hit("rust_docs", &output, &inputs, &params)? {
            println!(
                "Rust docs cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let index = crate::tasks::rust_docs::RustDocIndex::load_from_root(&input)?;
    let rows = match mode.as_str() {
        "jepa" => index.jepa_rows(max_rows),
        "tool-pairs" => index.tool_pairs(max_rows),
        _ => bail!("--mode must be jepa or tool-pairs"),
    };
    let mut content = String::new();
    for row in &rows {
        content.push_str(row.trim());
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest("rust_docs", &output, &inputs, params, rows.len())?;
    println!(
        "Wrote {} Rust docs rows to {} from {}",
        rows.len(),
        output.display(),
        input.display()
    );
    Ok(())
}

fn run_prepare_rust_doc_trajectories(args: &[String]) -> Result<()> {
    let mut input = None;
    let mut output = None;
    let mut code_output = None;
    let mut docs_root = crate::tasks::rust_docs::default_rust_docs_root();
    let mut max_rows = 12_000usize;
    let mut docs_top_k = 4usize;
    let mut docs_chars = 2200usize;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(parse_path_value(args, i, "--input")?);
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--code-output" => {
                code_output = Some(parse_path_value(args, i, "--code-output")?);
                i += 2;
            }
            "--docs-root" => {
                docs_root = Some(parse_path_value(args, i, "--docs-root")?);
                i += 2;
            }
            "--max-rows" => {
                max_rows = parse_usize_value(args, i, "--max-rows")?.max(1);
                i += 2;
            }
            "--docs-top-k" => {
                docs_top_k = parse_usize_value(args, i, "--docs-top-k")?.max(1);
                i += 2;
            }
            "--docs-chars" => {
                docs_chars = parse_usize_value(args, i, "--docs-chars")?.max(256);
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let input = input.context("--input is required")?;
    let output = output.context("--output is required")?;
    let docs_root = docs_root
        .context("Rust docs root not found; install rust-docs/rust-src or pass --docs-root")?;
    let params = json!({
        "max_rows": max_rows,
        "docs_top_k": docs_top_k,
        "docs_chars": docs_chars,
        "docs_root": docs_root.to_string_lossy(),
        "code_output": code_output.as_ref().map(|path: &PathBuf| path.to_string_lossy().to_string()),
        "trajectory": "prompt->fetch_docs->docs_conditioned_code",
    });
    let inputs = vec![input.clone()];
    if !force {
        if let Some(manifest) =
            artifact_cache_hit("rust_doc_trajectories", &output, &inputs, &params)?
        {
            if code_output
                .as_ref()
                .map(|path| path.exists())
                .unwrap_or(true)
            {
                println!(
                    "Rust doc trajectory cache hit: {} (rows={})",
                    output.display(),
                    manifest.rows
                );
                return Ok(());
            }
        }
    }
    let index = crate::tasks::rust_docs::RustDocIndex::load_from_root(&docs_root)?;
    let task_pairs = load_raw_pairs(&input)?;
    let mut rows = Vec::new();
    let mut code_rows = Vec::new();
    for (prompt, code) in task_pairs {
        if rows.len() >= max_rows {
            break;
        }
        let docs = index.format_hits(&prompt, docs_top_k, docs_chars);
        if docs.trim().is_empty() {
            continue;
        }
        let fetch_state = format!(
            "<action:fetch_docs>\n<tool:fetch_docs>\nUser request:\n{}\n\nQuery:\n{}",
            prompt,
            rust_doc_query_hint(&prompt)
        );
        rows.push(format!(
            "{}\t{}\tfetch_docs",
            escape_pair_field(&fetch_state),
            escape_pair_field(&docs)
        ));
        if rows.len() >= max_rows {
            break;
        }
        let code_state = format!(
            "<action:code>\n<ctx:user_request>\n{}\n</ctx:user_request>\n\n{}",
            prompt, docs
        );
        rows.push(format!(
            "{}\t{}\tcode",
            escape_pair_field(&code_state),
            escape_pair_field(&code)
        ));
        code_rows.push(format!(
            "{}\t{}",
            escape_pair_field(&code_state),
            escape_pair_field(&code)
        ));
    }
    let mut content = String::new();
    for row in &rows {
        content.push_str(row);
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    if let Some(code_output) = code_output.as_ref() {
        let mut code_content = String::new();
        for row in &code_rows {
            code_content.push_str(row);
            code_content.push('\n');
        }
        write_text_atomic(code_output, &code_content)?;
    }
    write_artifact_manifest(
        "rust_doc_trajectories",
        &output,
        &inputs,
        params,
        rows.len(),
    )?;
    println!(
        "Wrote {} Rust doc trajectory rows to {} from {}",
        rows.len(),
        output.display(),
        input.display()
    );
    Ok(())
}

fn rust_doc_query_hint(prompt: &str) -> String {
    let mut terms = Vec::new();
    for token in prompt.split(|ch: char| {
        !(ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' || ch == '<' || ch == '>')
    }) {
        let clean = token.trim_matches(|ch: char| ch == '<' || ch == '>' || ch == ',');
        if clean.len() < 3 {
            continue;
        }
        if clean.contains("::")
            || clean.chars().any(|ch| ch.is_ascii_uppercase())
            || matches!(
                clean.to_ascii_lowercase().as_str(),
                "iterator"
                    | "hashmap"
                    | "btree"
                    | "binaryheap"
                    | "vecdeque"
                    | "fromstr"
                    | "result"
                    | "option"
                    | "trait"
                    | "lifetime"
            )
        {
            terms.push(clean.to_string());
        }
        if terms.len() >= 12 {
            break;
        }
    }
    if terms.is_empty() {
        prompt
            .split_whitespace()
            .take(24)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        terms.join(" ")
    }
}

fn iter_md_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            if path.is_dir() {
                walk(&path, out)?;
            } else if path.extension().and_then(OsStr::to_str) == Some("md") {
                out.push(path);
            }
        }
        Ok(())
    }
    walk(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn split_markdown_by_heading(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        let is_heading = line.starts_with("# ") || line.starts_with("## ");
        if is_heading && !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

fn run_prepare_github_top_code(args: &[String]) -> Result<()> {
    let mut output = PathBuf::from("data/github_top_code_pairs.txt");
    let mut split = "train".to_string();
    let mut languages: Option<Vec<String>> = None;
    let mut default_languages = false;
    let mut max_files: Option<usize> = None;
    let mut min_lines = 5usize;
    let mut min_lines_prefix = 2usize;
    let mut min_lines_completion = 2usize;
    let mut split_ratio = 0.5f64;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = parse_path_value(args, i, "--output")?;
                i += 2;
            }
            "--split" => {
                split = parse_flag_value(args, i, "--split")?;
                i += 2;
            }
            "--languages" => {
                i += 1;
                let mut langs = Vec::new();
                while i < args.len() && !args[i].starts_with("--") {
                    langs.push(args[i].clone());
                    i += 1;
                }
                languages = Some(langs);
            }
            "--default-languages" => {
                default_languages = true;
                i += 1;
            }
            "--max-files" => {
                max_files = Some(parse_usize_value(args, i, "--max-files")?);
                i += 2;
            }
            "--min-lines" => {
                min_lines = parse_usize_value(args, i, "--min-lines")?;
                i += 2;
            }
            "--min-lines-prefix" => {
                min_lines_prefix = parse_usize_value(args, i, "--min-lines-prefix")?;
                i += 2;
            }
            "--min-lines-completion" => {
                min_lines_completion = parse_usize_value(args, i, "--min-lines-completion")?;
                i += 2;
            }
            "--split-ratio" => {
                split_ratio = parse_f64_value(args, i, "--split-ratio")?;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let lang_values = if default_languages {
        Some(
            DEFAULT_GITHUB_LANGUAGES
                .iter()
                .map(|v| (*v).to_string())
                .collect::<Vec<_>>(),
        )
    } else {
        languages
    };
    let params = json!({
        "split": split,
        "languages": lang_values,
        "max_files": max_files,
        "min_lines": min_lines,
        "min_lines_prefix": min_lines_prefix,
        "min_lines_completion": min_lines_completion,
        "split_ratio": split_ratio,
    });
    let mut input_paths = vec![PathBuf::from(format!(
        "hub:{GITHUB_TOP_CODE_DATASET_ID}:{split}"
    ))];
    if !force && output.exists() {
        // Keep a stable cache surface; hub inputs are represented only by params here.
        input_paths.clear();
        if let Some(manifest) =
            artifact_cache_hit("github_top_code", &output, &input_paths, &params)?
        {
            println!(
                "GitHub-top-code cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }

    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let mut language_counts: HashMap<String, usize> = HashMap::new();
    let per_language_limit = lang_values
        .as_ref()
        .zip(max_files)
        .map(|(langs, max)| max.max(1) / langs.len().max(1))
        .filter(|v| *v > 0);
    for row in iter_dataset_rows(GITHUB_TOP_CODE_DATASET_ID, &split)? {
        let language = row_string(&row, "file_language").unwrap_or_else(|| "Code".to_string());
        if let Some(ref langs) = lang_values {
            if !langs.iter().any(|v| v == &language) {
                continue;
            }
        }
        if let Some(limit) = per_language_limit {
            if language_counts.get(&language).copied().unwrap_or(0) >= limit {
                continue;
            }
        }
        let Some(content) = row_string(&row, "content") else {
            continue;
        };
        if looks_generated(&content) || looks_minified(&content) {
            continue;
        }
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < min_lines {
            continue;
        }
        let Some((prefix, completion)) = split_file_into_code_pair(
            &content,
            &language,
            min_lines_prefix,
            min_lines_completion,
            split_ratio,
        ) else {
            continue;
        };
        let digest = format!("{:x}", md5::compute(format!("{prefix}\t{completion}")));
        if !seen.insert(digest) {
            continue;
        }
        rows.push(format!("{prefix}\t{completion}"));
        *language_counts.entry(language).or_insert(0) += 1;
        if let Some(limit) = max_files {
            if rows.len() >= limit {
                break;
            }
        }
    }
    let mut content = String::new();
    for row in &rows {
        content.push_str(row);
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest("github_top_code", &output, &input_paths, params, rows.len())?;
    println!("Wrote {} pairs to {}", rows.len(), output.display());
    Ok(())
}

fn slugify_language(language: &str) -> String {
    let mut out = String::new();
    let mut prev_sep = false;
    for ch in language.trim().chars() {
        let ch = ch.to_ascii_lowercase();
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_sep = false;
        } else if !prev_sep {
            out.push('_');
            prev_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn normalize_code_field(text: &str, language: &str, tag: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    format!(
        "<lang:{}> <{}>\\n{}",
        slugify_language(language),
        tag,
        escape_pair_field(normalized.trim())
    )
}

fn looks_generated(content: &str) -> bool {
    let sample = content
        .lines()
        .take(12)
        .collect::<Vec<_>>()
        .join("\n")
        .to_ascii_lowercase();
    [
        "generated by",
        "auto-generated",
        "automatically generated",
        "do not edit",
        "@generated",
    ]
    .iter()
    .any(|marker| sample.contains(marker))
}

fn looks_minified(content: &str) -> bool {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return true;
    }
    let max_line_len = lines.iter().map(|v| v.len()).max().unwrap_or(0);
    let avg_line_len = lines.iter().map(|v| v.len()).sum::<usize>() as f64 / lines.len() as f64;
    max_line_len > 600 || avg_line_len > 160.0
}

fn split_file_into_code_pair(
    content: &str,
    language: &str,
    min_lines_prefix: usize,
    min_lines_completion: usize,
    split_ratio: f64,
) -> Option<(String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < min_lines_prefix + min_lines_completion {
        return None;
    }
    let n = lines.len();
    let cut = ((n as f64) * split_ratio).round() as usize;
    let cut = cut
        .max(min_lines_prefix)
        .min(n.saturating_sub(min_lines_completion));
    if cut < min_lines_prefix || n - cut < min_lines_completion {
        return None;
    }
    let prefix = lines[..cut].join("\n");
    let completion = lines[cut..].join("\n");
    if prefix.trim().is_empty() || completion.trim().is_empty() {
        return None;
    }
    Some((
        normalize_code_field(&prefix, language, "ctx"),
        normalize_code_field(&completion, language, "reply"),
    ))
}

fn parquet_readers_from_hub(
    dataset_id: &str,
    split: &str,
) -> Result<Vec<SerializedFileReader<File>>> {
    use hf_hub::{Repo, RepoType};
    let api = hf_hub::api::sync::Api::new().context("hf-hub API")?;
    let repo = Repo::with_revision(
        dataset_id.to_string(),
        RepoType::Dataset,
        PARQUET_REVISION.to_string(),
    );
    let api_repo = api.repo(repo);
    let info = api_repo
        .info()
        .map_err(|e| anyhow::anyhow!("hub info: {}", e))?;
    let mut siblings: Vec<String> = info
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".parquet"))
        .map(|s| s.rfilename)
        .collect();
    siblings.sort();
    let mut filtered: Vec<String> = siblings
        .iter()
        .filter(|name| name.contains(split))
        .cloned()
        .collect();
    if filtered.is_empty() {
        filtered = siblings;
    }
    let mut readers = Vec::new();
    for rfilename in filtered {
        let local_path = api_repo
            .get(&rfilename)
            .map_err(|e| anyhow::anyhow!("hub get: {}", e))?;
        readers.push(SerializedFileReader::new(File::open(local_path)?).context("parquet reader")?);
    }
    Ok(readers)
}

fn iter_dataset_rows(dataset_id: &str, split: &str) -> Result<Vec<Row>> {
    let readers = parquet_readers_from_hub(dataset_id, split)?;
    let mut rows = Vec::new();
    for reader in &readers {
        let iter = reader
            .get_row_iter(None)
            .map_err(|e| anyhow::anyhow!("parquet row iter: {}", e))?;
        for row_result in iter {
            rows.push(row_result.map_err(|e| anyhow::anyhow!("parquet row: {}", e))?);
        }
    }
    Ok(rows)
}

fn row_field<'a>(row: &'a Row, name: &str) -> Option<&'a Field> {
    for (field_name, field) in row.get_column_iter() {
        if field_name == name {
            return Some(field);
        }
    }
    None
}

fn row_string(row: &Row, name: &str) -> Option<String> {
    match row_field(row, name)? {
        Field::Str(value) => Some(value.trim().to_string()),
        _ => None,
    }
}

fn row_to_json_value(row: &Row) -> Value {
    let mut object = serde_json::Map::new();
    for (name, field) in row.get_column_iter() {
        object.insert(name.to_string(), field_to_json_value(field));
    }
    Value::Object(object)
}

fn field_to_json_value(field: &Field) -> Value {
    match field {
        Field::Null => Value::Null,
        Field::Bool(v) => Value::Bool(*v),
        Field::Byte(v) => Value::Number((*v as i64).into()),
        Field::Short(v) => Value::Number((*v as i64).into()),
        Field::Int(v) => Value::Number((*v as i64).into()),
        Field::Long(v) => Value::Number((*v).into()),
        Field::UByte(v) => Value::Number((*v as u64).into()),
        Field::UShort(v) => Value::Number((*v as u64).into()),
        Field::UInt(v) => Value::Number((*v as u64).into()),
        Field::ULong(v) => Value::Number((*v).into()),
        Field::Float16(v) => serde_json::Number::from_f64(f64::from(*v))
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Field::Float(v) => serde_json::Number::from_f64((*v).into())
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Field::Double(v) => serde_json::Number::from_f64(*v)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Field::Str(v) => Value::String(v.clone()),
        Field::Group(row) => row_to_json_value(row),
        Field::ListInternal(list) => {
            Value::Array(list.elements().iter().map(field_to_json_value).collect())
        }
        Field::Decimal(_)
        | Field::Bytes(_)
        | Field::Date(_)
        | Field::TimeMillis(_)
        | Field::TimeMicros(_)
        | Field::TimestampMillis(_)
        | Field::TimestampMicros(_)
        | Field::MapInternal(_) => Value::String(field.to_string()),
    }
}

fn iter_dataset_json_rows(dataset_id: &str, split: &str) -> Result<Vec<Value>> {
    Ok(iter_dataset_rows(dataset_id, split)?
        .into_iter()
        .map(|row| row_to_json_value(&row))
        .collect())
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn sanitize_pair_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn json_answers_texts(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(answers) = value.get("answers") else {
        return out;
    };
    if let Some(obj) = answers.as_object() {
        if let Some(items) = obj.get("text").and_then(Value::as_array) {
            for item in items {
                if let Some(text) = item.as_str() {
                    let text = sanitize_pair_text(text);
                    if !text.is_empty() {
                        out.push(text);
                    }
                }
            }
        }
        return out;
    }
    if let Some(items) = answers.as_array() {
        for item in items {
            if let Some(text) = item.as_str() {
                let text = sanitize_pair_text(text);
                if !text.is_empty() {
                    out.push(text);
                }
                continue;
            }
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                let text = sanitize_pair_text(text);
                if !text.is_empty() {
                    out.push(text);
                }
            }
        }
    }
    out
}

fn run_prepare_rust_function_tasks(args: &[String]) -> Result<()> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut github_top_code = false;
    let mut split = "train".to_string();
    let mut max_files: Option<usize> = None;
    let mut max_rows: Option<usize> = None;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(parse_path_value(args, i, "--input")?);
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--github-top-code" => {
                github_top_code = true;
                i += 1;
            }
            "--split" => {
                split = parse_flag_value(args, i, "--split")?;
                i += 2;
            }
            "--max-files" => {
                max_files = Some(parse_usize_value(args, i, "--max-files")?);
                i += 2;
            }
            "--max-rows" => {
                max_rows = Some(parse_usize_value(args, i, "--max-rows")?);
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    if !github_top_code && input.is_none() {
        bail!("either --input or --github-top-code is required");
    }
    let input_paths = input.clone().into_iter().collect::<Vec<_>>();
    let params = json!({
        "github_top_code": github_top_code,
        "split": split,
        "max_files": max_files,
        "max_rows": max_rows,
    });
    if !force {
        if let Some(manifest) =
            artifact_cache_hit("rust_function_tasks", &output, &input_paths, &params)?
        {
            println!(
                "Rust instruction-pair cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let samples = if github_top_code {
        collect_rust_functions_from_github_top_code(&split, max_files)?
    } else {
        collect_rust_functions_from_pairs(input.as_ref().unwrap())?
    };
    let written = write_rust_instruction_pairs(&samples, &output, max_rows)?;
    write_artifact_manifest(
        "rust_function_tasks",
        &output,
        &input_paths,
        params,
        written,
    )?;
    println!(
        "Wrote {written} Rust instruction pairs to {}",
        output.display()
    );
    Ok(())
}

fn strip_tags(text: &str) -> String {
    let unescaped = unescape_pair_field(text);
    let re = Regex::new(r"(?i)<lang:rust>\s*<(?:ctx|reply)>\s*").unwrap();
    re.replace(&unescaped, "").trim().to_string()
}

fn collect_rust_functions_from_pairs(input: &Path) -> Result<Vec<(String, String)>> {
    let mut samples = Vec::new();
    let reader = BufReader::new(File::open(input)?);
    for raw in reader.lines() {
        let line = raw?;
        let Some((left, right)) = line.split_once('\t') else {
            continue;
        };
        let combined = [strip_tags(left), strip_tags(right)]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        samples.extend(extract_rust_functions(&combined));
    }
    Ok(samples)
}

fn collect_rust_functions_from_github_top_code(
    split: &str,
    max_files: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let mut samples = Vec::new();
    let mut seen_files = 0usize;
    for row in iter_dataset_rows(GITHUB_TOP_CODE_DATASET_ID, split)? {
        let language = row_string(&row, "file_language").unwrap_or_default();
        if language != "Rust" {
            continue;
        }
        let Some(content) = row_string(&row, "content") else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }
        samples.extend(extract_rust_functions(&content));
        seen_files += 1;
        if max_files.is_some_and(|max| seen_files >= max) {
            break;
        }
    }
    Ok(samples)
}

fn extract_rust_functions(src: &str) -> Vec<(String, String)> {
    let func_start_re = Regex::new(
        r"(?m)^(?P<indent>\s*)(?P<sig>(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*[\s\S]*?)\{",
    )
    .unwrap();
    let mut results = Vec::new();
    for capture in func_start_re.captures_iter(src) {
        let Some(sig_match) = capture.name("sig") else {
            continue;
        };
        let open_idx = capture.get(0).unwrap().end() - 1;
        let Some(close_idx) = find_matching_brace(src, open_idx) else {
            continue;
        };
        let signature = sig_match
            .as_str()
            .lines()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        let body = src[sig_match.start()..=close_idx].trim().to_string();
        let line_count = body.lines().count();
        if !(2..=120).contains(&line_count) || signature.len() > 240 || body.len() > 8_000 {
            continue;
        }
        if !signature.contains("fn ") {
            continue;
        }
        results.push((signature, body));
    }
    results
}

fn find_matching_brace(src: &str, open_idx: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut in_char = false;
    let mut in_line_comment = false;
    let mut in_block_comment = 0i32;
    let mut escape = false;
    let mut i = open_idx;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        let next = bytes.get(i + 1).copied().map(char::from).unwrap_or('\0');
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
        } else if in_block_comment > 0 {
            if ch == '/' && next == '*' {
                in_block_comment += 1;
                i += 1;
            } else if ch == '*' && next == '/' {
                in_block_comment -= 1;
                i += 1;
            }
        } else if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if in_char {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '\'' {
                in_char = false;
            }
        } else {
            if ch == '/' && next == '/' {
                in_line_comment = true;
                i += 1;
            } else if ch == '/' && next == '*' {
                in_block_comment = 1;
                i += 1;
            } else if ch == '"' {
                in_string = true;
            } else if ch == '\'' {
                in_char = true;
            } else if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

fn build_prompt_variants(signature: &str) -> [String; 3] {
    let rules = "Rules:\n- Keep the exact function name and signature.\n- Return only compilable Rust code for that function.\n- Do not add explanation.\n";
    [
        format!(
            "Return only Rust code. Implement exactly this function:\n{signature}\n\n{rules}"
        ),
        format!("Write the Rust function below and return code only:\n{signature}\n\n{rules}"),
        format!("Complete this Rust function implementation. Output only the function body in its full function form:\n{signature}\n\n{rules}"),
    ]
}

fn write_rust_instruction_pairs(
    samples: &[(String, String)],
    output: &Path,
    max_rows: Option<usize>,
) -> Result<usize> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = output.with_extension("txt.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);
    let mut seen = HashSet::new();
    let mut written = 0usize;
    for (signature, body) in samples {
        if max_rows.is_some_and(|max| written >= max) {
            break;
        }
        for prompt in build_prompt_variants(signature) {
            if max_rows.is_some_and(|max| written >= max) {
                break;
            }
            let digest = format!("{:x}", md5::compute(format!("{prompt}\t{body}")));
            if !seen.insert(digest) {
                continue;
            }
            writeln!(
                out,
                "{}\t{}",
                escape_pair_field(&prompt),
                escape_pair_field(body)
            )?;
            written += 1;
        }
    }
    out.flush()?;
    fs::rename(tmp, output)?;
    Ok(written)
}

fn run_prepare_rust_repair_tasks(args: &[String]) -> Result<()> {
    let mut input = None;
    let mut output = None;
    let mut rustc_bin = "rustc".to_string();
    let mut seed = 0u64;
    let mut timeout_sec = 4.0f64;
    let mut variants_per_sample = 2usize;
    let mut max_rows = 0usize;
    let mut progress_every = 5000usize;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(parse_path_value(args, i, "--input")?);
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--rustc" => {
                rustc_bin = parse_flag_value(args, i, "--rustc")?;
                i += 2;
            }
            "--seed" => {
                seed = parse_usize_value(args, i, "--seed")? as u64;
                i += 2;
            }
            "--timeout-sec" => {
                timeout_sec = parse_f64_value(args, i, "--timeout-sec")?;
                i += 2;
            }
            "--variants-per-sample" => {
                variants_per_sample = parse_usize_value(args, i, "--variants-per-sample")?;
                i += 2;
            }
            "--max-rows" => {
                max_rows = parse_usize_value(args, i, "--max-rows")?;
                i += 2;
            }
            "--progress-every" => {
                progress_every = parse_usize_value(args, i, "--progress-every")?;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let input = input.context("--input is required")?;
    let output = output.context("--output is required")?;
    let rustc_version = rustc_version_string(&rustc_bin)?;
    let params = json!({
        "rustc_bin": rustc_bin,
        "rustc_version": rustc_version,
        "seed": seed,
        "timeout_sec": timeout_sec,
        "variants_per_sample": variants_per_sample,
        "max_rows": max_rows,
    });
    let inputs = vec![input.clone()];
    if !force {
        if let Some(manifest) = artifact_cache_hit("rust_repair_pairs", &output, &inputs, &params)?
        {
            println!(
                "Repair pair cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let pairs = load_escaped_pairs(&input)?;
    if pairs.is_empty() {
        bail!("no usable pairs found in {}", input.display());
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut seen = HashSet::new();
    let mut written = 0usize;
    let tmp = output.with_extension("txt.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);
    for (idx, (task_prompt, correct_code)) in pairs.iter().enumerate() {
        let mut variants = corruption_variants(correct_code);
        variants.shuffle(&mut rng);
        let mut kept = 0usize;
        for (name, broken_code) in variants {
            if max_rows > 0 && written >= max_rows {
                break;
            }
            let mut feedback = compile_feedback(&rustc_bin, &broken_code, timeout_sec)?;
            if feedback.is_empty() {
                if name == "wrong_fn_name" {
                    feedback =
                        "error: function name does not match the requested signature".to_string();
                } else {
                    continue;
                }
            }
            let prompt = build_repair_prompt(task_prompt, &broken_code, &feedback);
            let digest = format!("{:x}", md5::compute(format!("{prompt}\t{correct_code}")));
            if !seen.insert(digest) {
                continue;
            }
            writeln!(
                out,
                "{}\t{}",
                escape_pair_field(&prompt),
                escape_pair_field(correct_code)
            )?;
            written += 1;
            kept += 1;
            if kept >= variants_per_sample.max(1) {
                break;
            }
        }
        if progress_every > 0 && (idx + 1) % progress_every == 0 {
            println!(
                "Repair pairs progress: processed={} written={written}",
                idx + 1
            );
        }
        if max_rows > 0 && written >= max_rows {
            break;
        }
    }
    out.flush()?;
    fs::rename(tmp, &output)?;
    write_artifact_manifest("rust_repair_pairs", &output, &inputs, params, written)?;
    println!("Wrote {written} repair rows to {}", output.display());
    Ok(())
}

fn rustc_version_string(rustc_bin: &str) -> Result<String> {
    for args in [["--version", "--verbose"], ["--version", ""]] {
        let mut cmd = Command::new(rustc_bin);
        cmd.arg(args[0]);
        if !args[1].is_empty() {
            cmd.arg(args[1]);
        }
        let out = cmd.output()?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !stdout.is_empty() {
                return Ok(stdout);
            }
        }
    }
    bail!("failed to query rustc version from '{rustc_bin}'")
}

fn load_escaped_pairs(path: &Path) -> Result<Vec<(String, String)>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for raw in reader.lines() {
        let line = raw?;
        let Some((left, right)) = line.split_once('\t') else {
            continue;
        };
        rows.push((unescape_pair_field(left), unescape_pair_field(right)));
    }
    Ok(rows)
}

fn corruption_variants(code: &str) -> Vec<(String, String)> {
    let mut variants = vec![(
        "stray_role_prefix".to_string(),
        format!("assistant\n{code}"),
    )];
    if code.contains('{') {
        variants.push(("missing_open_brace".to_string(), code.replacen('{', "", 1)));
    }
    if let Some(idx) = code.rfind('}') {
        variants.push((
            "missing_close_brace".to_string(),
            format!("{}{}", &code[..idx], &code[idx + 1..]),
        ));
    }
    if code.contains("->") {
        variants.push(("broken_arrow".to_string(), code.replacen("->", "=>", 1)));
    }
    if code.contains(')') {
        variants.push(("broken_paren".to_string(), code.replacen(')', "]", 1)));
    }
    let func_name_re = Regex::new(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    if let Some(capture) = func_name_re.captures(code) {
        let name = capture.get(1).unwrap().as_str();
        variants.push((
            "wrong_fn_name".to_string(),
            code.replacen(&format!("fn {name}"), &format!("fn broken_{name}"), 1),
        ));
    }
    variants
}

fn run_command_with_timeout(cmd: &mut Command, timeout_sec: f64) -> Result<std::process::Output> {
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    if timeout_sec <= 0.0 {
        return Ok(cmd.output()?);
    }
    let timeout = Duration::from_secs_f64(timeout_sec);
    let mut child = cmd.spawn()?;
    let start = Instant::now();
    loop {
        if let Some(_status) = child.try_wait()? {
            return Ok(child.wait_with_output()?);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            return Ok(child.wait_with_output()?);
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn compile_feedback(rustc_bin: &str, code: &str, timeout_sec: f64) -> Result<String> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("tofy-rust-repair-{unique}"));
    fs::create_dir_all(&dir)?;
    let src = dir.join("bad.rs");
    fs::write(&src, format!("{code}\n"))?;
    let mut cmd = Command::new(rustc_bin);
    cmd.arg("--crate-type")
        .arg("lib")
        .arg("--edition")
        .arg("2021")
        .arg(&src);
    let output = run_command_with_timeout(&mut cmd, timeout_sec)?;
    let _ = fs::remove_dir_all(&dir);
    if output.status.success() {
        return Ok(String::new());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut lines = Vec::new();
    for line in stderr.lines() {
        if !line.trim().is_empty() {
            lines.push(line.trim().to_string());
        }
        if lines.len() >= 12 {
            break;
        }
    }
    Ok(lines.join("\n"))
}

fn build_repair_prompt(task_prompt: &str, broken_code: &str, feedback: &str) -> String {
    format!(
        "<action:repair_patch>\n<tool:read_error>\n<tool:repair_patch>\nReturn only corrected Rust code.\nFix the previous attempt using the compiler feedback.\n\n<ctx:original_request>\nOriginal request:\n{task_prompt}\n\n<ctx:previous_attempt>\nPrevious attempt:\n```rust\n{broken_code}\n```\n\n<ctx:compiler_feedback>\nCompiler feedback:\n{feedback}\n\n<ctx:constraints>\nRules:\n- Keep the exact requested function name and signature.\n- Return only compilable Rust code.\n- Do not add explanation.\n"
    )
}

fn run_prepare_world_mix(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut text_pairs = None;
    let mut code_pairs = Vec::new();
    let mut code_ratio = 0.35f64;
    let mut done_ratio = 0.18f64;
    let mut max_rows = 0usize;
    let mut seed = 0u64;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--text-pairs" => {
                text_pairs = Some(parse_path_value(args, i, "--text-pairs")?);
                i += 2;
            }
            "--code-pairs" => {
                code_pairs.push(parse_path_value(args, i, "--code-pairs")?);
                i += 2;
            }
            "--code-ratio" => {
                code_ratio = parse_f64_value(args, i, "--code-ratio")?;
                i += 2;
            }
            "--done-ratio" => {
                done_ratio = parse_f64_value(args, i, "--done-ratio")?;
                i += 2;
            }
            "--max-rows" => {
                max_rows = parse_usize_value(args, i, "--max-rows")?;
                i += 2;
            }
            "--seed" => {
                seed = parse_usize_value(args, i, "--seed")? as u64;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    let text_pairs = text_pairs.context("--text-pairs is required")?;
    if code_pairs.is_empty() {
        bail!("--code-pairs is required at least once");
    }
    let mut inputs = vec![text_pairs.clone()];
    inputs.extend(code_pairs.clone());
    let params = json!({
        "code_ratio": code_ratio,
        "done_ratio": done_ratio,
        "max_rows": max_rows,
        "seed": seed,
    });
    if !force {
        if let Some(manifest) = artifact_cache_hit("world_mix", &output, &inputs, &params)? {
            println!(
                "World mix cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut text_rows = load_raw_pairs(&text_pairs)?;
    let mut code_rows = Vec::new();
    for path in &code_pairs {
        code_rows.extend(load_raw_pairs(path)?);
    }
    if text_rows.is_empty() {
        bail!("no usable text rows found in {}", text_pairs.display());
    }
    if code_rows.is_empty() {
        bail!("no usable code rows found");
    }
    text_rows.shuffle(&mut rng);
    code_rows.shuffle(&mut rng);
    let max_rows = if max_rows > 0 {
        max_rows
    } else {
        text_rows.len() + code_rows.len()
    };
    let code_ratio = code_ratio.clamp(0.0, 0.8);
    let done_ratio = done_ratio.clamp(0.0, 0.4);
    let mut output_rows = Vec::new();
    let mut terminal_candidates = Vec::new();
    let mut chosen_code = 0usize;
    let mut chosen_text = 0usize;
    let mut chosen_done = 0usize;
    while output_rows.len() < max_rows && (!text_rows.is_empty() || !code_rows.is_empty()) {
        let want_code = rng.random::<f64>() < code_ratio;
        if want_code && !code_rows.is_empty() {
            let (left, right) = code_rows.pop().unwrap();
            let action = world_mix_code_action(&left, &right);
            output_rows.push(format!("{left}\t{right}\t{action}"));
            terminal_candidates.push((left, right, action.to_string()));
            chosen_code += 1;
        } else if !text_rows.is_empty() {
            let (left, right) = text_rows.pop().unwrap();
            output_rows.push(format!("{left}\t{right}\ttext_reply"));
            terminal_candidates.push((left, right, "text_reply".to_string()));
            chosen_text += 1;
        } else if !code_rows.is_empty() {
            let (left, right) = code_rows.pop().unwrap();
            let action = world_mix_code_action(&left, &right);
            output_rows.push(format!("{left}\t{right}\t{action}"));
            terminal_candidates.push((left, right, action.to_string()));
            chosen_code += 1;
        }
    }
    let target_done = if done_ratio > 0.0 && !output_rows.is_empty() {
        ((output_rows.len() as f64 * done_ratio) / (1.0 - done_ratio).max(1e-6)).round() as usize
    } else {
        0
    };
    terminal_candidates.shuffle(&mut rng);
    for (left, right, action) in terminal_candidates.into_iter().take(target_done) {
        let terminal_state = if action == "text_reply" {
            format!("{left}\\nAssistant: {right}")
        } else {
            format!("{left}\\n{right}")
        };
        output_rows.push(format!("{terminal_state}\t<done>\tdone"));
        chosen_done += 1;
    }
    let mut content = String::new();
    for row in &output_rows {
        content.push_str(row);
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest("world_mix", &output, &inputs, params, output_rows.len())?;
    let actual_code_rate = chosen_code as f64 / output_rows.len().max(1) as f64;
    let actual_done_rate = chosen_done as f64 / output_rows.len().max(1) as f64;
    println!(
        "Wrote {} rows to {} (text_rows={}, code_rows={}, done_rows={}, requested code_ratio={:.2}, requested done_ratio={:.2})",
        output_rows.len(),
        output.display(),
        chosen_text,
        chosen_code,
        chosen_done,
        code_ratio,
        done_ratio
    );
    println!("Actual code ratio: {:.2}", actual_code_rate);
    println!("Actual done ratio: {:.2}", actual_done_rate);
    Ok(())
}

fn world_mix_code_action(left: &str, right: &str) -> &'static str {
    let left_lower = left.to_ascii_lowercase();
    let right_lower = right.to_ascii_lowercase();
    if left_lower.contains("<action:fetch_docs>")
        || left_lower.contains("<tool:fetch_docs>")
        || right_lower.contains("<ctx:rust_docs>")
    {
        "fetch_docs"
    } else {
        "code"
    }
}

fn load_raw_pairs(path: &Path) -> Result<Vec<(String, String)>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for raw in reader.lines() {
        let line = raw?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((left, right)) = split_pair_line_first_two(trimmed) {
            rows.push((left.to_string(), right.to_string()));
        }
    }
    Ok(rows)
}

fn split_pair_line_first_two(line: &str) -> Option<(&str, &str)> {
    if line.contains('\t') {
        let mut parts = line.split('\t');
        return Some((parts.next()?, parts.next()?));
    }
    if line.contains("|||") {
        let mut parts = line.split("|||");
        return Some((parts.next()?, parts.next()?));
    }
    None
}

fn run_prepare_code_poc_mix(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut base_pairs = None;
    let mut instruction_pairs = None;
    let mut extra_pairs = Vec::new();
    let mut instruction_repeat = 3usize;
    let mut extra_repeat = 1usize;
    let mut max_rows = 0usize;
    let mut seed = 1337u64;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--base-pairs" => {
                base_pairs = Some(parse_path_value(args, i, "--base-pairs")?);
                i += 2;
            }
            "--instruction-pairs" => {
                instruction_pairs = Some(parse_path_value(args, i, "--instruction-pairs")?);
                i += 2;
            }
            "--extra-pairs" => {
                extra_pairs.push(parse_path_value(args, i, "--extra-pairs")?);
                i += 2;
            }
            "--instruction-repeat" => {
                instruction_repeat = parse_usize_value(args, i, "--instruction-repeat")?;
                i += 2;
            }
            "--extra-repeat" => {
                extra_repeat = parse_usize_value(args, i, "--extra-repeat")?;
                i += 2;
            }
            "--max-rows" => {
                max_rows = parse_usize_value(args, i, "--max-rows")?;
                i += 2;
            }
            "--seed" => {
                seed = parse_usize_value(args, i, "--seed")? as u64;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    let base_pairs = base_pairs.context("--base-pairs is required")?;
    let instruction_pairs = instruction_pairs.context("--instruction-pairs is required")?;
    let mut inputs = vec![base_pairs.clone(), instruction_pairs.clone()];
    inputs.extend(extra_pairs.clone());
    let params = json!({
        "instruction_repeat": instruction_repeat,
        "extra_repeat": extra_repeat,
        "max_rows": if max_rows > 0 { Some(max_rows) } else { None::<usize> },
        "seed": seed,
    });
    if !force {
        if let Some(manifest) = artifact_cache_hit("code_poc_mix", &output, &inputs, &params)? {
            println!(
                "Code POC mix cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let mut rows = load_non_empty_lines(&base_pairs)?;
    let instruction = load_non_empty_lines(&instruction_pairs)?;
    let mut extras = Vec::new();
    for path in &extra_pairs {
        extras.extend(load_non_empty_lines(path)?);
    }
    let mut mixed = Vec::new();
    for _ in 0..instruction_repeat.max(1) {
        mixed.extend(instruction.clone());
    }
    for _ in 0..extra_repeat.max(1) {
        mixed.extend(extras.clone());
    }
    mixed.append(&mut rows);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    mixed.shuffle(&mut rng);
    if max_rows > 0 && mixed.len() > max_rows {
        mixed.truncate(max_rows);
    }
    let mut content = String::new();
    for row in &mixed {
        content.push_str(row);
        content.push('\n');
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest("code_poc_mix", &output, &inputs, params, mixed.len())?;
    println!(
        "Wrote {} mixed code POC rows to {}",
        mixed.len(),
        output.display()
    );
    Ok(())
}

fn load_non_empty_lines(path: &Path) -> Result<Vec<String>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for raw in reader.lines() {
        let line = raw?;
        if !line.trim().is_empty() {
            rows.push(line);
        }
    }
    Ok(rows)
}

fn run_prepare_expert_pairs(args: &[String]) -> Result<()> {
    let mut dataset = "sciq".to_string();
    let mut output: Option<PathBuf> = None;
    let mut split = "train".to_string();
    let mut max_rows: Option<usize> = None;
    let mut include_support = true;
    let mut min_answer_words = 1usize;
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                dataset = parse_flag_value(args, i, "--dataset")?;
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--split" => {
                split = parse_flag_value(args, i, "--split")?;
                i += 2;
            }
            "--max-rows" | "--max_rows" => {
                max_rows = Some(parse_usize_value(args, i, "--max-rows")?);
                i += 2;
            }
            "--no-support" | "--no_support" => {
                include_support = false;
                i += 1;
            }
            "--min-answer-words" | "--min_answer_words" => {
                min_answer_words = parse_usize_value(args, i, "--min-answer-words")?;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.unwrap_or_else(|| match dataset.as_str() {
        "squad" => PathBuf::from("data/squad_pairs.txt"),
        _ => PathBuf::from("data/sciq_pairs.txt"),
    });
    let params = json!({
        "dataset": dataset,
        "split": split,
        "max_rows": max_rows,
        "include_support": include_support,
        "min_answer_words": min_answer_words,
    });
    let input_paths: Vec<PathBuf> = Vec::new();
    if !force && output.exists() {
        if let Some(manifest) = artifact_cache_hit("expert_pairs", &output, &input_paths, &params)?
        {
            println!(
                "Expert pair cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let dataset_id = match dataset.as_str() {
        "sciq" => SCIQ_DATASET_ID,
        "squad" => SQUAD_V2_DATASET_ID,
        other => bail!("--dataset must be sciq or squad, got {other}"),
    };
    let rows = iter_dataset_json_rows(dataset_id, &split)?;
    let mut content = String::new();
    let mut written = 0usize;
    let mut skipped = 0usize;
    for row in rows {
        if max_rows.is_some_and(|max| written >= max) {
            break;
        }
        match dataset.as_str() {
            "sciq" => {
                let question = value_string(&row, "question")
                    .map(|v| sanitize_pair_text(&v))
                    .unwrap_or_default();
                let answer = value_string(&row, "answer")
                    .map(|v| sanitize_pair_text(&v))
                    .unwrap_or_default();
                let support = value_string(&row, "support")
                    .map(|v| sanitize_pair_text(&v))
                    .unwrap_or_default();
                if question.is_empty()
                    || answer.is_empty()
                    || answer.split_whitespace().count() < min_answer_words
                {
                    skipped += 1;
                    continue;
                }
                let context = if include_support && !support.is_empty() {
                    format!("User: {question}\nContext: {support}")
                } else {
                    format!("User: {question}")
                };
                content.push_str(&context);
                content.push('\t');
                content.push_str(&answer);
                content.push('\n');
                written += 1;
            }
            "squad" => {
                let question = value_string(&row, "question")
                    .map(|v| sanitize_pair_text(&v))
                    .unwrap_or_default();
                let context_para = value_string(&row, "context")
                    .map(|v| sanitize_pair_text(&v))
                    .unwrap_or_default();
                let answer = json_answers_texts(&row)
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                if question.is_empty()
                    || context_para.is_empty()
                    || answer.is_empty()
                    || answer.split_whitespace().count() < min_answer_words
                {
                    skipped += 1;
                    continue;
                }
                content.push_str(&format!("User: {question}\nContext: {context_para}"));
                content.push('\t');
                content.push_str(&answer);
                content.push('\n');
                written += 1;
            }
            _ => unreachable!(),
        }
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest("expert_pairs", &output, &input_paths, params, written)?;
    println!(
        "Wrote {written} expert pairs to {} (skipped={skipped})",
        output.display()
    );
    Ok(())
}

fn run_prepare_casual_conversation(args: &[String]) -> Result<()> {
    let mut output = PathBuf::from("data/casual_pairs.txt");
    let mut min_tokens = 2usize;
    let mut lowercase = false;
    let mut max_pairs: Option<usize> = None;
    let mut split = "train".to_string();
    let mut force = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = parse_path_value(args, i, "--output")?;
                i += 2;
            }
            "--min-tokens" | "--min_tokens" => {
                min_tokens = parse_usize_value(args, i, "--min-tokens")?;
                i += 2;
            }
            "--lower" => {
                lowercase = true;
                i += 1;
            }
            "--max-pairs" | "--max_pairs" => {
                max_pairs = Some(parse_usize_value(args, i, "--max-pairs")?);
                i += 2;
            }
            "--split" => {
                split = parse_flag_value(args, i, "--split")?;
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let params = json!({
        "split": split,
        "min_tokens": min_tokens,
        "lowercase": lowercase,
        "max_pairs": max_pairs,
    });
    let input_paths: Vec<PathBuf> = Vec::new();
    if !force && output.exists() {
        if let Some(manifest) =
            artifact_cache_hit("casual_conversation_pairs", &output, &input_paths, &params)?
        {
            println!(
                "Casual conversation cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let rows = iter_dataset_json_rows(CASUAL_CONVERSATION_DATASET_ID, &split)?;
    let mut content = String::new();
    let mut written = 0usize;
    let mut skipped = 0usize;
    for row in rows {
        if max_pairs.is_some_and(|max| written >= max) {
            break;
        }
        let mut question = value_string(&row, "question")
            .map(|v| sanitize_pair_text(&v))
            .unwrap_or_default();
        let mut answer = value_string(&row, "answer")
            .map(|v| sanitize_pair_text(&v))
            .unwrap_or_default();
        if lowercase {
            question = question.to_lowercase();
            answer = answer.to_lowercase();
        }
        if question.is_empty()
            || answer.is_empty()
            || question.split_whitespace().count() < min_tokens
            || answer.split_whitespace().count() < min_tokens
        {
            skipped += 1;
            continue;
        }
        content.push_str(&question);
        content.push('\t');
        content.push_str(&answer);
        content.push('\n');
        written += 1;
    }
    write_text_atomic(&output, &content)?;
    write_artifact_manifest(
        "casual_conversation_pairs",
        &output,
        &input_paths,
        params,
        written,
    )?;
    println!(
        "Wrote {written} casual conversation pairs to {} (skipped={skipped})",
        output.display()
    );
    Ok(())
}

fn run_generate_code_eval_suite(args: &[String]) -> Result<()> {
    let mut output = PathBuf::from("eval/code_assistant_rust_hard.jsonl");
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = parse_path_value(args, i, "--output")?;
                i += 2;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    write_text_atomic(&output, CODE_EVAL_SUITE_JSONL)?;
    let rows = CODE_EVAL_SUITE_JSONL
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    println!("Wrote {rows} tasks to {}", output.display());
    Ok(())
}

fn run_check_dtype_discipline(_args: &[String]) -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let allow_files = HashSet::from([
        String::from("src/util.rs"),
        String::from("src/tasks/prepare.rs"),
    ]);
    let mut violations = Vec::new();
    collect_rs_files(&src, &mut |path| -> Result<()> {
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if allow_files.contains(&rel) {
            return Ok(());
        }
        let text = fs::read_to_string(path)?;
        for (idx, line) in text.lines().enumerate() {
            let stripped = line.trim_start();
            if stripped.starts_with("//") || stripped.starts_with("///") {
                continue;
            }
            for pattern in FORBIDDEN_DTYPE_PATTERNS {
                if line.contains(pattern) {
                    violations.push(format!("{rel}:{}:{}", idx + 1, stripped.trim()));
                }
            }
        }
        Ok(())
    })?;
    if !violations.is_empty() {
        println!("dtype discipline violations found:");
        for violation in &violations {
            println!("{violation}");
        }
        bail!("Use util::scalar_f32 / util::vec1_f32 / util::vec2_f32 instead.")
    }
    println!("dtype discipline check passed.");
    Ok(())
}

fn collect_rs_files(dir: &Path, cb: &mut dyn FnMut(&Path) -> Result<()>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, cb)?;
        } else if path.extension().and_then(OsStr::to_str) == Some("rs") {
            cb(&path)?;
        }
    }
    Ok(())
}

fn run_convert_jsonl_context_response_to_tsv(args: &[String]) -> Result<()> {
    let mut input_dir = None;
    let mut output = None;
    let mut min_tokens = 1usize;
    let mut max_tokens = 200usize;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input_dir" | "--input-dir" => {
                input_dir = Some(parse_path_value(args, i, "--input-dir")?);
                i += 2;
            }
            "--output" => {
                output = Some(parse_path_value(args, i, "--output")?);
                i += 2;
            }
            "--min_tokens" | "--min-tokens" => {
                min_tokens = parse_usize_value(args, i, "--min-tokens")?;
                i += 2;
            }
            "--max_tokens" | "--max-tokens" => {
                max_tokens = parse_usize_value(args, i, "--max-tokens")?;
                i += 2;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    let input_dir = input_dir.context("--input-dir is required")?;
    let output = output.context("--output is required")?;
    let mut json_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(OsStr::to_str) == Some("json"))
        .collect();
    json_files.sort();
    if json_files.is_empty() {
        bail!("no .json files found in {}", input_dir.display());
    }
    let mut content = String::new();
    let mut written = 0usize;
    let mut skipped = 0usize;
    for path in &json_files {
        let reader = BufReader::new(File::open(path)?);
        for raw in reader.lines() {
            let line = raw?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                skipped += 1;
                continue;
            };
            let ctx = value
                .get("context")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .replace('\t', " ");
            let rsp = value
                .get("response")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .replace('\t', " ");
            if ctx.is_empty()
                || rsp.is_empty()
                || ctx.split_whitespace().count() < min_tokens
                || rsp.split_whitespace().count() < min_tokens
                || ctx.split_whitespace().count() > max_tokens
                || rsp.split_whitespace().count() > max_tokens
            {
                skipped += 1;
                continue;
            }
            content.push_str(&ctx);
            content.push('\t');
            content.push_str(&rsp);
            content.push('\n');
            written += 1;
        }
    }
    write_text_atomic(&output, &content)?;
    println!(
        "wrote {written} pairs, skipped {skipped}, output={}",
        output.display()
    );
    Ok(())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    repo_root().join("local_models")
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn parse_probe_stage(value: &str) -> Result<ProbeStage> {
    match value {
        "all" => Ok(ProbeStage::All),
        "latent" => Ok(ProbeStage::Latent),
        "world" => Ok(ProbeStage::World),
        "decoder" => Ok(ProbeStage::Decoder),
        other => bail!("--stage must be one of all|latent|world|decoder, got {other}"),
    }
}

fn default_oom_probe_args() -> OomProbeArgs {
    OomProbeArgs {
        stage: ProbeStage::All,
        quick: false,
        probe_dir: None,
        keep_local_models: false,
        build: false,
        binary: repo_root().join("target/release/jepa_ai"),
        dtype: "bf16".to_string(),
        sample_interval_sec: 0.10,
        min_headroom_mb: 512,
        max_late_growth_mb: 512,
        data_rows: 24_000,
        run_group: format!("oom_probe_{}", now_unix_secs()),
        dim: 640,
        max_seq: 256,
        layers: 7,
        heads: 8,
        bridge_dim: 640,
        planner_slots: 64,
        vocab: 8000,
        latent_steps: 1000,
        latent_batch: 12,
        latent_accum: 2,
        world_steps: 8000,
        world_batch: 64,
        world_accum: 2,
        world_warmup_batch: 64,
        world_warmup_accum: 1,
        world_warmup_steps: None,
        world_lambda: 0.2,
        world_lr: 2e-4,
        world_action_loss_weight: 0.0,
        decoder_steps: 1000,
        decoder_batch: 6,
        decoder_accum: 4,
        decoder_max_seq: 128,
        decoder_max_vocab: 16_000,
        setup_latent_steps: 1,
        setup_world_steps: 2,
        latent_model: None,
        encoder_vocab: None,
        world_model: None,
    }
}

fn parse_sustained_oom_probe_args(args: &[String]) -> Result<OomProbeArgs> {
    let mut parsed = default_oom_probe_args();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--stage" => {
                parsed.stage = parse_probe_stage(&parse_flag_value(args, i, "--stage")?)?;
                i += 2;
            }
            "--quick" => {
                parsed.quick = true;
                i += 1;
            }
            "--probe-dir" | "--probe_dir" => {
                parsed.probe_dir = Some(parse_path_value(args, i, "--probe-dir")?);
                i += 2;
            }
            "--keep-local-models" | "--keep_local_models" => {
                parsed.keep_local_models = true;
                i += 1;
            }
            "--build" => {
                parsed.build = true;
                i += 1;
            }
            "--binary" => {
                parsed.binary = parse_path_value(args, i, "--binary")?;
                i += 2;
            }
            "--dtype" => {
                parsed.dtype = parse_flag_value(args, i, "--dtype")?;
                i += 2;
            }
            "--sample-interval" | "--sample_interval" => {
                parsed.sample_interval_sec = parse_f64_value(args, i, "--sample-interval")?;
                i += 2;
            }
            "--min-headroom-mb" | "--min_headroom_mb" => {
                parsed.min_headroom_mb = parse_i64_value(args, i, "--min-headroom-mb")?;
                i += 2;
            }
            "--max-late-growth-mb" | "--max_late_growth_mb" => {
                parsed.max_late_growth_mb = parse_i64_value(args, i, "--max-late-growth-mb")?;
                i += 2;
            }
            "--data-rows" | "--data_rows" => {
                parsed.data_rows = parse_usize_value(args, i, "--data-rows")?;
                i += 2;
            }
            "--run-group" | "--run_group" => {
                parsed.run_group = parse_flag_value(args, i, "--run-group")?;
                i += 2;
            }
            "--dim" => {
                parsed.dim = parse_usize_value(args, i, "--dim")?;
                i += 2;
            }
            "--max-seq" | "--max_seq" => {
                parsed.max_seq = parse_usize_value(args, i, "--max-seq")?;
                i += 2;
            }
            "--layers" => {
                parsed.layers = parse_usize_value(args, i, "--layers")?;
                i += 2;
            }
            "--heads" => {
                parsed.heads = parse_usize_value(args, i, "--heads")?;
                i += 2;
            }
            "--bridge-dim" | "--bridge_dim" => {
                parsed.bridge_dim = parse_usize_value(args, i, "--bridge-dim")?;
                i += 2;
            }
            "--planner-slots" | "--planner_slots" => {
                parsed.planner_slots = parse_usize_value(args, i, "--planner-slots")?;
                i += 2;
            }
            "--vocab" => {
                parsed.vocab = parse_usize_value(args, i, "--vocab")?;
                i += 2;
            }
            "--latent-steps" | "--latent_steps" => {
                parsed.latent_steps = parse_usize_value(args, i, "--latent-steps")?;
                i += 2;
            }
            "--latent-batch" | "--latent_batch" => {
                parsed.latent_batch = parse_usize_value(args, i, "--latent-batch")?;
                i += 2;
            }
            "--latent-accum" | "--latent_accum" => {
                parsed.latent_accum = parse_usize_value(args, i, "--latent-accum")?;
                i += 2;
            }
            "--world-steps" | "--world_steps" => {
                parsed.world_steps = parse_usize_value(args, i, "--world-steps")?;
                i += 2;
            }
            "--world-batch" | "--world_batch" => {
                parsed.world_batch = parse_usize_value(args, i, "--world-batch")?;
                i += 2;
            }
            "--world-accum" | "--world_accum" => {
                parsed.world_accum = parse_usize_value(args, i, "--world-accum")?;
                i += 2;
            }
            "--world-warmup-batch" | "--world_warmup_batch" => {
                parsed.world_warmup_batch = parse_usize_value(args, i, "--world-warmup-batch")?;
                i += 2;
            }
            "--world-warmup-accum" | "--world_warmup_accum" => {
                parsed.world_warmup_accum = parse_usize_value(args, i, "--world-warmup-accum")?;
                i += 2;
            }
            "--world-warmup-steps" | "--world_warmup_steps" => {
                parsed.world_warmup_steps =
                    Some(parse_usize_value(args, i, "--world-warmup-steps")?);
                i += 2;
            }
            "--world-lambda" | "--world_lambda" => {
                parsed.world_lambda = parse_f64_value(args, i, "--world-lambda")?;
                i += 2;
            }
            "--world-lr" | "--world_lr" => {
                parsed.world_lr = parse_f64_value(args, i, "--world-lr")?;
                i += 2;
            }
            "--world-action-loss-weight" | "--world_action_loss_weight" => {
                parsed.world_action_loss_weight =
                    parse_f64_value(args, i, "--world-action-loss-weight")?;
                i += 2;
            }
            "--decoder-steps" | "--decoder_steps" => {
                parsed.decoder_steps = parse_usize_value(args, i, "--decoder-steps")?;
                i += 2;
            }
            "--decoder-batch" | "--decoder_batch" => {
                parsed.decoder_batch = parse_usize_value(args, i, "--decoder-batch")?;
                i += 2;
            }
            "--decoder-accum" | "--decoder_accum" => {
                parsed.decoder_accum = parse_usize_value(args, i, "--decoder-accum")?;
                i += 2;
            }
            "--decoder-max-seq" | "--decoder_max_seq" => {
                parsed.decoder_max_seq = parse_usize_value(args, i, "--decoder-max-seq")?;
                i += 2;
            }
            "--decoder-max-vocab" | "--decoder_max_vocab" => {
                parsed.decoder_max_vocab = parse_usize_value(args, i, "--decoder-max-vocab")?;
                i += 2;
            }
            "--setup-latent-steps" | "--setup_latent_steps" => {
                parsed.setup_latent_steps = parse_usize_value(args, i, "--setup-latent-steps")?;
                i += 2;
            }
            "--setup-world-steps" | "--setup_world_steps" => {
                parsed.setup_world_steps = parse_usize_value(args, i, "--setup-world-steps")?;
                i += 2;
            }
            "--latent-model" | "--latent_model" => {
                parsed.latent_model = Some(parse_path_value(args, i, "--latent-model")?);
                i += 2;
            }
            "--encoder-vocab" | "--encoder_vocab" => {
                parsed.encoder_vocab = Some(parse_path_value(args, i, "--encoder-vocab")?);
                i += 2;
            }
            "--world-model" | "--world_model" => {
                parsed.world_model = Some(parse_path_value(args, i, "--world-model")?);
                i += 2;
            }
            value => bail!("unknown flag: {value}"),
        }
    }
    if !matches!(parsed.dtype.as_str(), "bf16" | "f16" | "f32") {
        bail!("--dtype must be one of bf16|f16|f32");
    }
    if parsed.sample_interval_sec <= 0.0 {
        bail!("--sample-interval must be > 0");
    }
    if parsed.quick {
        parsed.latent_steps = parsed.latent_steps.min(3);
        parsed.world_steps = parsed.world_steps.min(4);
        parsed.decoder_steps = parsed.decoder_steps.min(3);
        parsed.min_headroom_mb = parsed.min_headroom_mb.min(128);
        parsed.max_late_growth_mb = 0;
    }
    Ok(parsed)
}

fn query_vram_snapshot() -> Option<(i64, i64, i64)> {
    let out = Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return None;
    }
    let used = parts[0].parse::<f64>().ok()? as i64;
    let free = parts[1].parse::<f64>().ok()? as i64;
    let total = parts[2].parse::<f64>().ok()? as i64;
    Some((used, free, total))
}

fn copy_tree_recursive(src: &Path, dst: &Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            copy_tree_recursive(&src_path, &dst_path)?;
        }
    } else {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dst)?;
    }
    Ok(())
}

fn backup_local_models(probe_dir: &Path) -> Result<PathBuf> {
    let backup = probe_dir.join("local_models.backup");
    if backup.exists() {
        fs::remove_dir_all(&backup)?;
    }
    fs::create_dir_all(&backup)?;
    let local_models = model_dir();
    if local_models.exists() {
        copy_tree_recursive(&local_models, &backup)?;
    }
    Ok(backup)
}

fn restore_local_models(backup: &Path) -> Result<()> {
    let local_models = model_dir();
    if local_models.exists() {
        fs::remove_dir_all(&local_models)?;
    }
    fs::create_dir_all(&local_models)?;
    if backup.exists() {
        copy_tree_recursive(backup, &local_models)?;
    }
    Ok(())
}

fn write_probe_data(path: &Path, rows: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(path)?);
    for i in 0..rows {
        let prompt = format!(
            "user asks rust task {i} value_{i} condition_{} context_token_{i} helper_name_{i} return only rust code",
            i % 97
        );
        let answer = format!(
            "<action:code> fn function_{i}(value_{i}: i32) -> i32 {{ let shifted_{i} = value_{i} + {}; if shifted_{i} > {} {{ shifted_{i} }} else {{ shifted_{i} + 1 }} }} unique_code_token_{i}",
            i % 17,
            i % 31
        );
        writeln!(out, "{prompt}\t{answer}\tcode")?;
    }
    out.flush()?;
    Ok(())
}

fn probe_base_env(args: &OomProbeArgs, stage_name: &str) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    env.insert("TOFY_TRAIN_DTYPE".to_string(), args.dtype.clone());
    env.insert("TOFY_LATENT_CONTEXT_SEGMENTS".to_string(), "4".to_string());
    env.insert(
        "TOFY_LATENT_RECENT_FULL_SEGMENTS".to_string(),
        "1".to_string(),
    );
    env.insert("TOFY_LATENT_HISTORY_RATIO".to_string(), "0.35".to_string());
    env.insert("TOFY_LATENT_WARMUP_STEPS".to_string(), "0".to_string());
    env.insert("TOFY_WORLD_CONTEXT_SEGMENTS".to_string(), "2".to_string());
    env.insert(
        "TOFY_WORLD_RECENT_FULL_SEGMENTS".to_string(),
        "1".to_string(),
    );
    env.insert("TOFY_RECURSIVE_PLANNER_MEMORY".to_string(), "1".to_string());
    env.insert(
        "TOFY_WORLD_TRAIN_ROLLOUT_STEPS".to_string(),
        "2".to_string(),
    );
    env.insert("TOFY_WORLD_ROLLOUT_STEPS".to_string(), "2".to_string());
    env.insert("TOFY_RUN_GROUP".to_string(), args.run_group.clone());
    env.insert("TOFY_RUN_STAGE_NAME".to_string(), stage_name.to_string());
    env
}

fn probe_world_env(args: &OomProbeArgs, stage_name: &str) -> HashMap<String, String> {
    let mut env = probe_base_env(args, stage_name);
    env.insert(
        "TOFY_WORLD_WARMUP_BATCH".to_string(),
        args.world_warmup_batch.to_string(),
    );
    env.insert(
        "TOFY_WORLD_WARMUP_GRAD_ACCUM".to_string(),
        args.world_warmup_accum.to_string(),
    );
    match args.world_warmup_steps {
        Some(steps) => {
            env.insert("TOFY_WORLD_WARMUP_STEPS".to_string(), steps.to_string());
        }
        None => {
            env.remove("TOFY_WORLD_WARMUP_STEPS");
        }
    }
    env
}

fn latent_probe_cmd(args: &OomProbeArgs, data_path: &Path, steps: usize) -> Vec<String> {
    vec![
        args.binary.to_string_lossy().to_string(),
        "--latent".to_string(),
        data_path.to_string_lossy().to_string(),
        steps.to_string(),
        args.latent_batch.to_string(),
        args.dim.to_string(),
        args.max_seq.to_string(),
        args.layers.to_string(),
        args.heads.to_string(),
        args.vocab.to_string(),
        "--grad-accum".to_string(),
        args.latent_accum.to_string(),
    ]
}

fn world_probe_cmd(
    args: &OomProbeArgs,
    data_path: &Path,
    latent: &Path,
    vocab: &Path,
    steps: usize,
) -> Vec<String> {
    vec![
        args.binary.to_string_lossy().to_string(),
        "--train-world".to_string(),
        latent.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        data_path.to_string_lossy().to_string(),
        steps.to_string(),
        args.world_batch.to_string(),
        args.dim.to_string(),
        args.max_seq.to_string(),
        args.layers.to_string(),
        args.heads.to_string(),
        args.bridge_dim.to_string(),
        args.planner_slots.to_string(),
        "--lambda".to_string(),
        args.world_lambda.to_string(),
        "--lr".to_string(),
        args.world_lr.to_string(),
        "--grad-accum".to_string(),
        args.world_accum.to_string(),
        "--action-loss-weight".to_string(),
        args.world_action_loss_weight.to_string(),
    ]
}

fn decoder_probe_cmd(
    args: &OomProbeArgs,
    data_path: &Path,
    latent: &Path,
    vocab: &Path,
    world: &Path,
    steps: usize,
    output_path: &Path,
) -> Vec<String> {
    vec![
        args.binary.to_string_lossy().to_string(),
        "--train-decoder".to_string(),
        latent.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        world.to_string_lossy().to_string(),
        data_path.to_string_lossy().to_string(),
        steps.to_string(),
        args.decoder_batch.to_string(),
        args.decoder_max_seq.to_string(),
        args.dim.to_string(),
        args.layers.to_string(),
        args.heads.to_string(),
        args.bridge_dim.to_string(),
        args.planner_slots.to_string(),
        "--decoder-kind".to_string(),
        "code".to_string(),
        "--decoder-max-vocab".to_string(),
        args.decoder_max_vocab.to_string(),
        "--decoder-output".to_string(),
        output_path.to_string_lossy().to_string(),
        "--grad-accum".to_string(),
        args.decoder_accum.to_string(),
    ]
}

fn latest_artifact_with_prefix(prefix: &str) -> Result<PathBuf> {
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(model_dir())? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if !name.starts_with(prefix) || !name.ends_with(".safetensors") {
            continue;
        }
        if name.matches(".safetensors").count() != 1 {
            continue;
        }
        let modified = fs::metadata(&path)?
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if latest
            .as_ref()
            .map(|(best_time, _)| modified > *best_time)
            .unwrap_or(true)
        {
            latest = Some((modified, path));
        }
    }
    latest
        .map(|(_, path)| path)
        .with_context(|| format!("no artifact found for prefix {prefix}"))
}

fn run_checked_command(cmd: &[String], env: &HashMap<String, String>) -> Result<()> {
    let mut process = Command::new(&cmd[0]);
    process.current_dir(repo_root());
    for arg in &cmd[1..] {
        process.arg(arg);
    }
    for (key, value) in env {
        process.env(key, value);
    }
    let status = process.status()?;
    if !status.success() {
        bail!("command failed: {}", cmd.join(" "));
    }
    Ok(())
}

fn ensure_setup_latent(args: &OomProbeArgs, data_path: &Path) -> Result<(PathBuf, PathBuf)> {
    if let Some(latent) = &args.latent_model {
        let vocab = args
            .encoder_vocab
            .clone()
            .unwrap_or_else(|| model_dir().join("vocabs/vocab_encoder.txt"));
        return Ok((latent.clone(), vocab));
    }
    let cmd = latent_probe_cmd(args, data_path, args.setup_latent_steps);
    run_checked_command(&cmd, &probe_base_env(args, "setup_latent"))?;
    Ok((
        latest_artifact_with_prefix("model_latent_")?,
        model_dir().join("vocabs/vocab_encoder.txt"),
    ))
}

fn ensure_setup_world(
    args: &OomProbeArgs,
    data_path: &Path,
    latent: &Path,
    vocab: &Path,
) -> Result<PathBuf> {
    if let Some(world) = &args.world_model {
        return Ok(world.clone());
    }
    let cmd = world_probe_cmd(args, data_path, latent, vocab, args.setup_world_steps);
    run_checked_command(&cmd, &probe_world_env(args, "setup_world"))?;
    latest_artifact_with_prefix("model_world_")
}

fn tail_lines(text: &str, lines: usize) -> String {
    let collected: Vec<&str> = text.lines().collect();
    let start = collected.len().saturating_sub(lines);
    collected[start..].join("\n")
}

fn rounded_sec(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn run_measured_probe(
    name: &str,
    cmd: &[String],
    env: &HashMap<String, String>,
    probe_dir: &Path,
    args: &OomProbeArgs,
) -> Result<ProbeResult> {
    let log_path = probe_dir.join(format!("{name}.log"));
    let stdout = File::create(&log_path)?;
    let stderr = stdout.try_clone()?;
    let mut process = Command::new(&cmd[0]);
    process.current_dir(repo_root());
    for arg in &cmd[1..] {
        process.arg(arg);
    }
    for (key, value) in env {
        process.env(key, value);
    }
    process.stdout(Stdio::from(stdout));
    process.stderr(Stdio::from(stderr));
    let mut child = process.spawn()?;
    let start = Instant::now();
    let mut samples = Vec::new();
    let sample_interval = Duration::from_secs_f64(args.sample_interval_sec);
    let return_code = loop {
        if let Some(status) = child.try_wait()? {
            break status.code().unwrap_or(-1);
        }
        if let Some((used_mb, free_mb, total_mb)) = query_vram_snapshot() {
            samples.push(VramSample {
                used_mb,
                free_mb,
                total_mb,
                elapsed_sec: rounded_sec(start.elapsed().as_secs_f64()),
            });
        }
        thread::sleep(sample_interval);
    };
    let text = fs::read_to_string(&log_path).unwrap_or_default();
    let sample_count = samples.len();
    let peak_used_mb = samples.iter().map(|s| s.used_mb).max();
    let min_free_mb = samples.iter().map(|s| s.free_mb).min();
    let total_mb = samples.iter().map(|s| s.total_mb).max();
    let late_growth_mb = if samples.is_empty() {
        None
    } else {
        let half_idx = samples.len() / 2;
        Some(samples.last().unwrap().used_mb - samples[half_idx].used_mb)
    };
    let peak_fraction = peak_used_mb.zip(total_mb).and_then(|(used, total)| {
        if total > 0 {
            Some(used as f64 / total as f64)
        } else {
            None
        }
    });
    let first_used_mb = samples.first().map(|s| s.used_mb);
    let last_used_mb = samples.last().map(|s| s.used_mb);
    let lowered = text.to_ascii_lowercase();
    let oom = text.contains("CUDA_ERROR_OUT_OF_MEMORY") || lowered.contains("out of memory");
    let headroom_ok = min_free_mb
        .map(|free| free >= args.min_headroom_mb)
        .unwrap_or(false);
    let growth_ok = late_growth_mb
        .map(|growth| args.max_late_growth_mb <= 0 || growth <= args.max_late_growth_mb)
        .unwrap_or(true);
    let passed = return_code == 0 && !oom && headroom_ok && growth_ok;
    let samples_path = probe_dir.join(format!("{name}.vram_samples.jsonl"));
    let mut sample_writer = BufWriter::new(File::create(&samples_path)?);
    for sample in &samples {
        serde_json::to_writer(&mut sample_writer, sample)?;
        sample_writer.write_all(b"\n")?;
    }
    sample_writer.flush()?;
    let result = ProbeResult {
        name: name.to_string(),
        cmd: cmd.to_vec(),
        return_code,
        seconds: rounded_sec(start.elapsed().as_secs_f64()),
        log: log_path.to_string_lossy().to_string(),
        oom,
        headroom_ok,
        growth_ok,
        passed,
        min_headroom_mb: args.min_headroom_mb,
        max_late_growth_mb: args.max_late_growth_mb,
        sample_count,
        peak_used_mb,
        min_free_mb,
        total_mb,
        peak_fraction,
        late_growth_mb,
        first_used_mb,
        last_used_mb,
        samples_path: samples_path.to_string_lossy().to_string(),
        tail: tail_lines(&text, 30),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "name": result.name,
            "cmd": result.cmd,
            "return_code": result.return_code,
            "seconds": result.seconds,
            "log": result.log,
            "oom": result.oom,
            "headroom_ok": result.headroom_ok,
            "growth_ok": result.growth_ok,
            "passed": result.passed,
            "min_headroom_mb": result.min_headroom_mb,
            "max_late_growth_mb": result.max_late_growth_mb,
            "sample_count": result.sample_count,
            "peak_used_mb": result.peak_used_mb,
            "min_free_mb": result.min_free_mb,
            "total_mb": result.total_mb,
            "peak_fraction": result.peak_fraction,
            "late_growth_mb": result.late_growth_mb,
            "first_used_mb": result.first_used_mb,
            "last_used_mb": result.last_used_mb,
            "samples_path": result.samples_path,
        }))?
    );
    Ok(result)
}

fn write_probe_summary(probe_dir: &Path, results: &[ProbeResult]) -> Result<PathBuf> {
    let path = probe_dir.join("summary.json");
    write_json_atomic(&path, &results)?;
    Ok(path)
}

fn run_sustained_oom_probe(args: &[String]) -> Result<()> {
    let args = parse_sustained_oom_probe_args(args)?;
    if query_vram_snapshot().is_none() {
        bail!("nvidia-smi is required for sustained OOM probes");
    }
    if args.build {
        let status = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .current_dir(repo_root())
            .status()?;
        if !status.success() {
            bail!("cargo build --release failed");
        }
    }
    if !args.binary.exists() {
        bail!(
            "release binary not found: {} (run cargo build --release)",
            args.binary.display()
        );
    }
    let probe_dir = args.probe_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("tofy-oom-probe-{}", now_unix_secs()))
    });
    fs::create_dir_all(&probe_dir)?;
    let data_path = probe_dir.join("world_code_synth.txt");
    write_probe_data(&data_path, args.data_rows)?;
    let backup = backup_local_models(&probe_dir)?;
    println!("OOM probe dir: {}", probe_dir.display());
    println!(
        "Stage: {}",
        match args.stage {
            ProbeStage::All => "all",
            ProbeStage::Latent => "latent",
            ProbeStage::World => "world",
            ProbeStage::Decoder => "decoder",
        }
    );
    println!("Minimum required free VRAM: {} MB", args.min_headroom_mb);
    println!(
        "Maximum allowed late growth: {} MB",
        args.max_late_growth_mb
    );

    let run = || -> Result<()> {
        let mut results = Vec::new();
        let mut latent: Option<PathBuf> = None;
        let mut vocab: Option<PathBuf> = None;
        let mut world: Option<PathBuf> = None;

        if matches!(args.stage, ProbeStage::All | ProbeStage::Latent) {
            let cmd = latent_probe_cmd(&args, &data_path, args.latent_steps);
            let result = run_measured_probe(
                "latent",
                &cmd,
                &probe_base_env(&args, "latent"),
                &probe_dir,
                &args,
            )?;
            results.push(result.clone());
            if !result.passed {
                let summary = write_probe_summary(&probe_dir, &results)?;
                println!("Summary: {}", summary.display());
                bail!("OOM probe failed");
            }
            latent = Some(latest_artifact_with_prefix("model_latent_")?);
            vocab = Some(model_dir().join("vocabs/vocab_encoder.txt"));
        } else if matches!(args.stage, ProbeStage::World | ProbeStage::Decoder) {
            let (latent_path, vocab_path) = ensure_setup_latent(&args, &data_path)?;
            latent = Some(latent_path);
            vocab = Some(vocab_path);
        }

        if matches!(args.stage, ProbeStage::All | ProbeStage::World) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            let cmd = world_probe_cmd(&args, &data_path, latent_ref, vocab_ref, args.world_steps);
            let result = run_measured_probe(
                "world",
                &cmd,
                &probe_world_env(&args, "world"),
                &probe_dir,
                &args,
            )?;
            results.push(result.clone());
            if !result.passed {
                let summary = write_probe_summary(&probe_dir, &results)?;
                println!("Summary: {}", summary.display());
                bail!("OOM probe failed");
            }
            world = Some(latest_artifact_with_prefix("model_world_")?);
        } else if matches!(args.stage, ProbeStage::Decoder) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            world = Some(ensure_setup_world(
                &args, &data_path, latent_ref, vocab_ref,
            )?);
        }

        if matches!(args.stage, ProbeStage::All | ProbeStage::Decoder) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            let world_ref = world
                .or_else(|| latest_artifact_with_prefix("model_world_").ok())
                .context("world setup missing")?;
            let output_path = probe_dir.join("code_decoder_oom_probe.safetensors");
            let cmd = decoder_probe_cmd(
                &args,
                &data_path,
                latent_ref,
                vocab_ref,
                &world_ref,
                args.decoder_steps,
                &output_path,
            );
            let result = run_measured_probe(
                "decoder",
                &cmd,
                &probe_base_env(&args, "decoder"),
                &probe_dir,
                &args,
            )?;
            results.push(result.clone());
            if !result.passed {
                let summary = write_probe_summary(&probe_dir, &results)?;
                println!("Summary: {}", summary.display());
                bail!("OOM probe failed");
            }
        }

        let summary = write_probe_summary(&probe_dir, &results)?;
        let failed: Vec<&ProbeResult> = results.iter().filter(|result| !result.passed).collect();
        println!("Summary: {}", summary.display());
        if !failed.is_empty() {
            println!("OOM probe failed:");
            for result in failed {
                println!(
                    "- {}: rc={} oom={} min_free={}MB late_growth={}MB log={}",
                    result.name,
                    result.return_code,
                    result.oom,
                    result
                        .min_free_mb
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    result
                        .late_growth_mb
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    result.log
                );
            }
            bail!("OOM probe failed");
        }
        println!("OOM probe passed.");
        Ok(())
    };

    let result = run();
    if args.keep_local_models {
        println!("Keeping local_models changes from probe.");
    } else {
        restore_local_models(&backup)?;
        println!("Restored local_models after probe.");
    }
    result
}
