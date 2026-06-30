use anyhow::{bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use rand::{seq::SliceRandom, RngExt, SeedableRng};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::data::CODE_EOS_TOKEN;

const ARTIFACT_CACHE_VERSION: u32 = 1;
const PARQUET_REVISION: &str = "refs/convert/parquet";
const GITHUB_TOP_CODE_DATASET_ID: &str = "ronantakizawa/github-top-code";
const CASUAL_CONVERSATION_DATASET_ID: &str = "SohamGhadge/casual-conversation";
const SCIQ_DATASET_ID: &str = "sciq";
const SQUAD_V2_DATASET_ID: &str = "rajpurkar/squad_v2";
const DEFAULT_GITHUB_LANGUAGES: &[&str] = &["Go"];
const CODE_EVAL_SUITE_JSONL: &str = include_str!("../../eval/code_assistant_go_hard.jsonl");
const DEFAULT_PREPARE_CHUNK_LINES: usize = 16_384;
const DEFAULT_CODE_MIX_SHUFFLE_BUCKETS: usize = 128;
const FORBIDDEN_DTYPE_PATTERNS: &[&str] = &[
    ".to_scalar::<f32>()",
    ".to_vec1::<f32>()",
    ".to_vec2::<f32>()",
];

fn prepare_chunk_lines() -> usize {
    std::env::var("TOFY_PREPARE_CHUNK_LINES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_PREPARE_CHUNK_LINES)
}

fn code_mix_shuffle_buckets() -> usize {
    std::env::var("TOFY_CODE_MIX_SHUFFLE_BUCKETS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_CODE_MIX_SHUFFLE_BUCKETS)
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
    HighWorld,
    Decoder,
}

#[derive(Clone, Debug)]
struct OomProbeArgs {
    stage: ProbeStage,
    quick: bool,
    max_vram: bool,
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
    latent_warmup_batch: usize,
    latent_warmup_accum: usize,
    latent_warmup_steps: Option<usize>,
    world_steps: usize,
    world_batch: usize,
    world_accum: usize,
    world_warmup_batch: usize,
    world_warmup_accum: usize,
    world_warmup_steps: Option<usize>,
    world_lambda: f64,
    world_lr: f64,
    world_action_loss_weight: f64,
    high_world_steps: usize,
    high_world_batch: usize,
    high_world_accum: usize,
    high_world_warmup_batch: usize,
    high_world_warmup_accum: usize,
    high_world_warmup_steps: Option<usize>,
    decoder_steps: usize,
    decoder_batch: usize,
    decoder_accum: usize,
    decoder_warmup_batch: usize,
    decoder_warmup_accum: usize,
    decoder_warmup_steps: Option<usize>,
    decoder_max_seq: usize,
    decoder_max_vocab: usize,
    decoder_dim: usize,
    decoder_layers: usize,
    decoder_heads: usize,
    decoder_ff_dim: usize,
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

#[derive(Clone, Debug, Serialize)]
struct FullRunVramEstimate {
    stage: String,
    measured_peak_mb: Option<i64>,
    historical_multiplier: f64,
    estimated_full_peak_mb: Option<i64>,
    rationale: String,
}

#[derive(Clone, Debug, Serialize)]
struct ProbeSummary {
    results: Vec<ProbeResult>,
    full_run_estimates: Vec<FullRunVramEstimate>,
}

#[derive(Deserialize)]
struct OomProbeProfileFile {
    #[serde(rename = "8gb")]
    eight_gb: OomProbeProfile,
    #[serde(rename = "48gb")]
    forty_eight_gb: OomProbeProfile,
    #[serde(rename = "80gb")]
    eighty_gb: OomProbeProfile,
}

#[derive(Clone, Copy, Deserialize)]
struct OomProbeProfile {
    latent_steps: usize,
    world_steps: usize,
    high_world_steps: usize,
    code_decoder_steps: usize,
    dim: usize,
    latent_max_seq: usize,
    world_max_seq: usize,
    code_decoder_max_seq: usize,
    layers: usize,
    heads: usize,
    max_vocab: usize,
    code_decoder_max_vocab: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    decoder_dim: usize,
    decoder_layers: usize,
    decoder_heads: usize,
    decoder_ff_dim: usize,
    latent_batch: usize,
    latent_warmup_batch: usize,
    world_batch: usize,
    world_warmup_batch: usize,
    code_decoder_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    code_decoder_grad_accum: usize,
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
        "--prepare-github-top-code" | "prepare-github-top-code" => {
            run_prepare_github_top_code(&args[2..])?;
            Ok(true)
        }
        "--prepare-go-function-tasks" | "prepare-go-function-tasks" => {
            run_prepare_go_function_tasks(&args[2..])?;
            Ok(true)
        }
        "--prepare-go-algorithm-tasks" | "prepare-go-algorithm-tasks" => {
            run_prepare_go_algorithm_tasks(&args[2..])?;
            Ok(true)
        }
        "--prepare-go-semantics-tasks" | "prepare-go-semantics-tasks" => {
            run_prepare_go_semantics_tasks(&args[2..])?;
            Ok(true)
        }
        "--prepare-go-repair-tasks" | "prepare-go-repair-tasks" => {
            run_prepare_go_repair_tasks(&args[2..])?;
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
        "--generate-go-code-eval-suite" | "generate-go-code-eval-suite" => {
            run_generate_go_code_eval_suite(&args[2..])?;
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
        "--max-vram-probe" | "max-vram-probe" => {
            run_max_vram_probe(&args[2..])?;
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
    let tmp = atomic_tmp_path(path);
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
    let tmp = atomic_tmp_path(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&tmp)?);
    writer.write_all(content.as_bytes())?;
    writer.flush()?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn atomic_tmp_path(path: &Path) -> PathBuf {
    path.with_extension(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| format!("{ext}.tmp"))
            .unwrap_or_else(|| "tmp".to_string()),
    )
}

fn side_tmp_path(path: &Path, label: &str) -> PathBuf {
    let file_name = path.file_name().and_then(OsStr::to_str).unwrap_or("output");
    path.with_file_name(format!("{file_name}.{label}.tmp"))
}

fn source_fingerprint(path: &Path) -> Result<SourceFingerprint> {
    if is_remote_source_path(path) {
        return Ok(remote_source_fingerprint(path));
    }
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

fn is_remote_source_path(path: &Path) -> bool {
    path.to_string_lossy().starts_with("hub:")
}

fn remote_source_fingerprint(path: &Path) -> SourceFingerprint {
    let source = path.to_string_lossy().to_string();
    SourceFingerprint {
        path: source.clone(),
        len: 0,
        modified_unix_nanos: 0,
        content_hash: sha256_text(&source),
    }
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

fn sha256_text(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn prepared_cache_required() -> bool {
    std::env::var("TOFY_REQUIRE_PREPARED_CACHE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes"
        })
        .unwrap_or(false)
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
    {
        return Ok(None);
    }
    if prepared_cache_required() {
        if manifest.rows > 0 {
            println!(
                "{} prepared-cache handoff hit: {} (rows={})",
                kind,
                output_path.display(),
                manifest.rows
            );
            return Ok(Some(manifest));
        }
        return Ok(None);
    }
    if manifest.params != *params || manifest.inputs.len() != input_paths.len() {
        return Ok(None);
    }
    for (path, stored) in input_paths.iter().zip(manifest.inputs.iter()) {
        if is_remote_source_path(path) {
            if &remote_source_fingerprint(path) != stored {
                return Ok(None);
            }
            continue;
        }
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

pub(crate) fn unescape_pair_field(text: &str) -> String {
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

pub(crate) fn escape_pair_field(text: &str) -> String {
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
    escape_pair_field(unescape_pair_field(text).trim())
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
    let chunk_lines = prepare_chunk_lines();
    println!(
        "Encoder corpus parallel prepare: chunk_lines={} rayon_threads={}",
        chunk_lines,
        rayon::current_num_threads()
    );
    for path in &inputs {
        let mut lines = BufReader::new(File::open(path)?).lines();
        loop {
            let chunk = read_line_chunk(&mut lines, chunk_lines)?;
            if chunk.is_empty() {
                break;
            }
            let prepared = chunk
                .par_iter()
                .flat_map_iter(|line| {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        return Vec::new();
                    }
                    if let Some((left, right)) = split_pair_line(trimmed) {
                        return [left, right]
                            .into_iter()
                            .filter(|part| !part.trim().is_empty())
                            .map(|part| flatten_for_encoder(part.trim()))
                            .collect::<Vec<_>>();
                    }
                    vec![flatten_for_encoder(trimmed)]
                })
                .collect::<Vec<_>>();
            for line in prepared {
                if !line.is_empty() {
                    writeln!(out, "{line}")?;
                    written += 1;
                }
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
    let input_paths = vec![PathBuf::from(format!(
        "hub:{GITHUB_TOP_CODE_DATASET_ID}:{split}"
    ))];
    if !force && output.exists() {
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

fn run_prepare_go_function_tasks(args: &[String]) -> Result<()> {
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
            artifact_cache_hit("go_function_tasks", &output, &input_paths, &params)?
        {
            println!(
                "Go instruction-pair cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    let samples = if github_top_code {
        collect_go_functions_from_github_top_code(&split, max_files)?
    } else {
        collect_go_functions_from_pairs(input.as_ref().unwrap())?
    };
    let written = write_go_instruction_pairs(&samples, &output, max_rows)?;
    write_artifact_manifest("go_function_tasks", &output, &input_paths, params, written)?;
    println!(
        "Wrote {written} Go instruction pairs to {}",
        output.display()
    );
    Ok(())
}

fn collect_go_functions_from_pairs(input: &Path) -> Result<Vec<(String, String)>> {
    let mut lines = BufReader::new(File::open(input)?).lines();
    let chunk_lines = prepare_chunk_lines();
    let mut samples = Vec::new();
    println!(
        "Go function extraction parallel prepare: chunk_lines={} rayon_threads={}",
        chunk_lines,
        rayon::current_num_threads()
    );
    loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        let extracted = chunk
            .par_iter()
            .map(|line| {
                let Some((left, right)) = line.split_once('\t') else {
                    return Vec::new();
                };
                let combined = [strip_code_tags(left), strip_code_tags(right)]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                extract_go_functions(&combined)
            })
            .collect::<Vec<_>>();
        samples.extend(extracted.into_iter().flatten());
    }
    Ok(samples)
}

fn collect_go_functions_from_github_top_code(
    split: &str,
    max_files: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let mut samples = Vec::new();
    let mut seen_files = 0usize;
    for row in iter_dataset_rows(GITHUB_TOP_CODE_DATASET_ID, split)? {
        let language = row_string(&row, "file_language").unwrap_or_default();
        if language != "Go" {
            continue;
        }
        let Some(content) = row_string(&row, "content") else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }
        samples.extend(extract_go_functions(&content));
        seen_files += 1;
        if max_files.is_some_and(|max| seen_files >= max) {
            break;
        }
    }
    Ok(samples)
}

fn strip_code_tags(text: &str) -> String {
    let unescaped = unescape_pair_field(text);
    code_tag_re().replace(&unescaped, "").trim().to_string()
}

fn extract_go_functions(src: &str) -> Vec<(String, String)> {
    let mut results = Vec::new();
    for capture in go_func_start_re().captures_iter(src) {
        let Some(sig_match) = capture.name("sig") else {
            continue;
        };
        if sig_match.as_str().contains("func (") {
            continue;
        }
        let open_idx = capture.get(0).unwrap().end() - 1;
        let Some(close_idx) = find_matching_brace(src, open_idx) else {
            continue;
        };
        let signature = sig_match
            .as_str()
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        let body = src[sig_match.start()..=close_idx].trim().to_string();
        let line_count = body.lines().count();
        if !(2..=120).contains(&line_count) || signature.len() > 240 || body.len() > 8_000 {
            continue;
        }
        results.push((signature, body));
    }
    results
}

fn build_go_prompt_variants(signature: &str) -> [String; 3] {
    let rules = "Rules:\n- Keep the exact function name and signature.\n- Return compilable Go code for package main.\n- Imports are allowed.\n- Do not add explanation.\n";
    [
        format!("Return only Go code. Implement exactly this function:\n{signature}\n\n{rules}"),
        format!("Write the Go function below and return code only:\n{signature}\n\n{rules}"),
        format!("Complete this Go function implementation. Output only the complete function and required imports:\n{signature}\n\n{rules}"),
    ]
}

fn write_go_instruction_pairs(
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
    let prepared = samples
        .par_iter()
        .map(|(signature, body)| {
            build_go_prompt_variants(signature)
                .into_iter()
                .map(|prompt| {
                    let digest = format!("{:x}", md5::compute(format!("{prompt}\t{body}")));
                    (digest, prompt, body.clone())
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for rows in prepared {
        if max_rows.is_some_and(|max| written >= max) {
            break;
        }
        for (digest, prompt, body) in rows {
            if max_rows.is_some_and(|max| written >= max) {
                break;
            }
            if !seen.insert(digest) {
                continue;
            }
            writeln!(
                out,
                "{}\t{}",
                escape_pair_field(&prompt),
                escape_pair_field(&body)
            )?;
            written += 1;
        }
    }
    out.flush()?;
    fs::rename(tmp, output)?;
    Ok(written)
}

struct CuratedGoPair {
    id: &'static str,
    prompt: &'static str,
    target: &'static str,
}

fn run_prepare_go_algorithm_tasks(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut force = false;
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
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    write_curated_go_pairs(
        &output,
        "go_algorithm_tasks",
        "go_algorithm_tasks_v2",
        &curated_go_algorithm_pairs(),
        false,
        force,
    )
}

fn run_prepare_go_semantics_tasks(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut force = false;
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
            value => bail!("unknown flag: {value}"),
        }
    }
    let output = output.context("--output is required")?;
    write_curated_go_pairs(
        &output,
        "go_semantics_tasks",
        "go_semantics_tasks_v1",
        &curated_go_semantics_pairs(),
        true,
        force,
    )
}

fn write_curated_go_pairs(
    output: &Path,
    artifact_name: &str,
    version: &str,
    pairs: &[CuratedGoPair],
    semantic_targets: bool,
    force: bool,
) -> Result<()> {
    let inputs = Vec::new();
    let prompt_frames = [
        "Return only the requested Go code.",
        "Write a complete Go solution. Output code only.",
        "Implement this Go API exactly. Do not explain.",
        "Produce compilable Go for package main. Imports are allowed.",
        "Complete the function and any required helper types. Return code only.",
        "Use deterministic behavior for ties and errors. Output only Go.",
        "Solve the task with clear control flow. Code only.",
        "Implement the specification exactly as written.",
    ];
    let semantic_frames = [
        "Analyze the Go code and return only the requested answer.",
        "Track the execution state exactly. Do not explain.",
        "Return the deterministic result for this Go snippet.",
        "Infer the final values from the Go program.",
    ];
    let params = json!({
        "version": version,
        "pairs": pairs.len(),
        "semantic_targets": semantic_targets,
        "prompt_frames": if semantic_targets { semantic_frames.len() } else { prompt_frames.len() },
    });
    if !force {
        if let Some(manifest) = artifact_cache_hit(artifact_name, output, &inputs, &params)? {
            println!(
                "Curated Go pair cache hit: {} (rows={})",
                output.display(),
                manifest.rows
            );
            return Ok(());
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = output.with_extension("txt.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);
    let mut seen = HashSet::new();
    let frames: &[&str] = if semantic_targets {
        &semantic_frames
    } else {
        &prompt_frames
    };
    let mut written = 0usize;
    for pair in pairs {
        for frame in frames {
            let prompt = format!("{frame}\n\n{}", pair.prompt);
            let target = if semantic_targets {
                format!("<action:text_reply> {}", pair.target)
            } else {
                pair.target.to_string()
            };
            let digest = format!("{:x}", md5::compute(format!("{}\t{}", pair.id, prompt)));
            if !seen.insert(digest) {
                continue;
            }
            writeln!(
                out,
                "{}\t{}",
                escape_pair_field(&prompt),
                escape_pair_field(&target)
            )?;
            written += 1;
        }
    }
    out.flush()?;
    fs::rename(tmp, output)?;
    write_artifact_manifest(artifact_name, output, &inputs, params, written)?;
    println!("Wrote {written} curated Go rows to {}", output.display());
    Ok(())
}

fn curated_go_algorithm_pairs() -> Vec<CuratedGoPair> {
    vec![
        CuratedGoPair {
            id: "parse_csv_line",
            prompt: r#"Implement exactly this function:
func ParseCSVLine(line string) ([]string, error)

Rules:
- Parse one RFC4180-style CSV record.
- Commas split fields unless they are inside double quotes.
- A doubled quote inside a quoted field becomes one quote.
- Return an error for an unclosed quote or junk after a closing quote.
- Do not use encoding/csv."#,
            target: r#"import "errors"

func ParseCSVLine(line string) ([]string, error) {
	fields := []string{}
	field := make([]byte, 0, len(line))
	inQuotes := false
	quoted := false
	for i := 0; i < len(line); i++ {
		c := line[i]
		if inQuotes {
			if c == '"' {
				if i+1 < len(line) && line[i+1] == '"' {
					field = append(field, '"')
					i++
				} else {
					inQuotes = false
					quoted = true
				}
			} else {
				field = append(field, c)
			}
			continue
		}
		switch c {
		case ',':
			fields = append(fields, string(field))
			field = field[:0]
			quoted = false
		case '"':
			if len(field) != 0 || quoted {
				return nil, errors.New("unexpected quote")
			}
			inQuotes = true
		default:
			if quoted {
				return nil, errors.New("trailing data after quote")
			}
			field = append(field, c)
		}
	}
	if inQuotes {
		return nil, errors.New("unclosed quote")
	}
	fields = append(fields, string(field))
	return fields, nil
}"#,
        },
        CuratedGoPair {
            id: "stable_toposort",
            prompt: r#"Implement exactly this function:
func StableTopoSort(edges [][2]string) ([]string, bool)

Rules:
- Each edge is [from, to].
- Include every node mentioned in an edge.
- Return nodes in topological order.
- When multiple nodes are ready, pick the lexicographically smallest node.
- Return false when a cycle exists."#,
            target: r#"import "sort"

func StableTopoSort(edges [][2]string) ([]string, bool) {
	adj := map[string][]string{}
	indeg := map[string]int{}
	for _, edge := range edges {
		from, to := edge[0], edge[1]
		if _, ok := indeg[from]; !ok {
			indeg[from] = 0
		}
		if _, ok := indeg[to]; !ok {
			indeg[to] = 0
		}
		adj[from] = append(adj[from], to)
		indeg[to]++
	}
	ready := make([]string, 0)
	for node, degree := range indeg {
		if degree == 0 {
			ready = append(ready, node)
		}
	}
	out := make([]string, 0, len(indeg))
	for len(ready) > 0 {
		sort.Strings(ready)
		node := ready[0]
		ready = ready[1:]
		out = append(out, node)
		sort.Strings(adj[node])
		for _, next := range adj[node] {
			indeg[next]--
			if indeg[next] == 0 {
				ready = append(ready, next)
			}
		}
	}
	return out, len(out) == len(indeg)
}"#,
        },
        CuratedGoPair {
            id: "shortest_path",
            prompt: r#"Implement exactly this function:
func ShortestPath(grid []string) int

Rules:
- Grid cells are bytes.
- 'S' is the start, 'E' is the end, '#' is blocked.
- Move in four directions only.
- Return the shortest distance in steps, or -1 if unreachable or malformed."#,
            target: r#"func ShortestPath(grid []string) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return -1
	}
	h, w := len(grid), len(grid[0])
	sr, sc, er, ec := -1, -1, -1, -1
	for r := 0; r < h; r++ {
		if len(grid[r]) != w {
			return -1
		}
		for c := 0; c < w; c++ {
			if grid[r][c] == 'S' {
				sr, sc = r, c
			}
			if grid[r][c] == 'E' {
				er, ec = r, c
			}
		}
	}
	if sr < 0 || er < 0 {
		return -1
	}
	type point struct{ r, c, d int }
	q := []point{{sr, sc, 0}}
	seen := make([][]bool, h)
	for i := range seen {
		seen[i] = make([]bool, w)
	}
	seen[sr][sc] = true
	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for len(q) > 0 {
		cur := q[0]
		q = q[1:]
		if cur.r == er && cur.c == ec {
			return cur.d
		}
		for _, dir := range dirs {
			nr, nc := cur.r+dir[0], cur.c+dir[1]
			if nr < 0 || nc < 0 || nr >= h || nc >= w || seen[nr][nc] || grid[nr][nc] == '#' {
				continue
			}
			seen[nr][nc] = true
			q = append(q, point{nr, nc, cur.d + 1})
		}
	}
	return -1
}"#,
        },
        CuratedGoPair {
            id: "render_template",
            prompt: r#"Implement exactly this function:
func RenderTemplate(input string, values map[string]string) (string, error)

Rules:
- Replace ${name} with values["name"].
- Names may contain ASCII letters, digits, and underscore.
- Return an error for an unknown name, empty name, invalid name byte, or unclosed placeholder.
- Preserve all other bytes."#,
            target: r#"import "fmt"

func RenderTemplate(input string, values map[string]string) (string, error) {
	out := make([]byte, 0, len(input))
	for i := 0; i < len(input); i++ {
		if input[i] != '$' || i+1 >= len(input) || input[i+1] != '{' {
			out = append(out, input[i])
			continue
		}
		j := i + 2
		for j < len(input) && input[j] != '}' {
			b := input[j]
			ok := b == '_' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9'
			if !ok {
				return "", fmt.Errorf("invalid placeholder")
			}
			j++
		}
		if j >= len(input) {
			return "", fmt.Errorf("unclosed placeholder")
		}
		name := input[i+2 : j]
		if name == "" {
			return "", fmt.Errorf("empty placeholder")
		}
		value, ok := values[name]
		if !ok {
			return "", fmt.Errorf("unknown placeholder %s", name)
		}
		out = append(out, value...)
		i = j
	}
	return string(out), nil
}"#,
        },
        CuratedGoPair {
            id: "compact_sorted_numbers",
            prompt: r#"Implement exactly this function:
func CompactSortedNumbers(nums []int) string

Rules:
- Input is sorted ascending and may contain duplicates.
- Collapse consecutive runs into "start-end".
- Single values are rendered as just the number.
- Remove duplicates before compacting.
- Join ranges with commas."#,
            target: r#"import (
	"strconv"
	"strings"
)

func CompactSortedNumbers(nums []int) string {
	parts := []string{}
	for i := 0; i < len(nums); {
		start := nums[i]
		end := start
		i++
		for i < len(nums) && nums[i] == end {
			i++
		}
		for i < len(nums) && nums[i] == end+1 {
			end = nums[i]
			i++
			for i < len(nums) && nums[i] == end {
				i++
			}
		}
		if start == end {
			parts = append(parts, strconv.Itoa(start))
		} else {
			parts = append(parts, strconv.Itoa(start)+"-"+strconv.Itoa(end))
		}
	}
	return strings.Join(parts, ",")
}"#,
        },
        CuratedGoPair {
            id: "parse_header_block",
            prompt: r#"Implement exactly this function:
func ParseHeaderBlock(input string) (map[string][]string, error)

Rules:
- Input contains newline-separated HTTP-like headers.
- Each non-empty line must contain "Name: value".
- Trim spaces around names and values.
- Header names are ASCII case-insensitive and must be returned in canonical lowercase.
- Preserve repeated values in input order."#,
            target: r#"import (
	"fmt"
	"strings"
)

func ParseHeaderBlock(input string) (map[string][]string, error) {
	out := map[string][]string{}
	for _, line := range strings.Split(input, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("missing colon")
		}
		name := strings.ToLower(strings.TrimSpace(parts[0]))
		value := strings.TrimSpace(parts[1])
		if name == "" {
			return nil, fmt.Errorf("empty header name")
		}
		for i := 0; i < len(name); i++ {
			b := name[i]
			if !(b >= 'a' && b <= 'z' || b >= '0' && b <= '9' || b == '-') {
				return nil, fmt.Errorf("invalid header name")
			}
		}
		out[name] = append(out[name], value)
	}
	return out, nil
}"#,
        },
    ]
}

fn curated_go_semantics_pairs() -> Vec<CuratedGoPair> {
    vec![
        CuratedGoPair {
            id: "trace_loop_accumulator",
            prompt: r#"Given this Go function call, return JSON with the final local variables and return value.

func score(xs []int) int {
    total := 0
    last := 0
    for i, x := range xs {
        if x%2 == 0 {
            total += x * (i + 1)
            last = x
        }
    }
    return total + last
}

Call: score([]int{3, 4, 5, 2})"#,
            target: r#"{"return":18,"locals":{"total":16,"last":2,"i":3,"x":2}}"#,
        },
        CuratedGoPair {
            id: "trace_map_counts",
            prompt: r#"Given this Go function call, return JSON with the final map and return value.

func count(items []string) (map[string]int, int) {
    seen := map[string]int{}
    max := 0
    for _, item := range items {
        seen[item]++
        if seen[item] > max {
            max = seen[item]
        }
    }
    return seen, max
}

Call: count([]string{"go", "rs", "go", "go", "rs"})"#,
            target: r#"{"return":[{"go":3,"rs":2},3],"locals":{"seen":{"go":3,"rs":2},"max":3}}"#,
        },
        CuratedGoPair {
            id: "trace_branching_string",
            prompt: r#"Given this Go function call, return JSON with the exact return value.

func rewrite(s string) string {
    out := ""
    for i := 0; i < len(s); i++ {
        if s[i] == '-' {
            out += "_"
        } else if i%2 == 0 {
            out += string(s[i] - 32)
        } else {
            out += string(s[i])
        }
    }
    return out
}

Call: rewrite("ab-cd")"#,
            target: r#"{"return":"Ab_cD"}"#,
        },
        CuratedGoPair {
            id: "trace_nested_control",
            prompt: r#"Given this Go function call, return JSON with the final local variables and return value.

func firstWindow(xs []int, limit int) int {
    sum := 0
    left := 0
    for right, x := range xs {
        sum += x
        for sum > limit {
            sum -= xs[left]
            left++
        }
        if right-left+1 == 3 {
            return left
        }
    }
    return -1
}

Call: firstWindow([]int{4, 2, 3, 7, 1}, 10)"#,
            target: r#"{"return":0,"locals":{"sum":9,"left":0,"right":2,"x":3}}"#,
        },
    ]
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

fn run_prepare_go_repair_tasks(args: &[String]) -> Result<()> {
    let mut input = None;
    let mut output = None;
    let mut go_bin = "go".to_string();
    let mut seed = 0u64;
    let mut timeout_sec = 3.0f64;
    let mut variants_per_sample = 2usize;
    let mut max_rows = 0usize;
    let mut progress_every = 5000usize;
    let mut workers = None;
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
            "--go" => {
                go_bin = parse_flag_value(args, i, "--go")?;
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
            "--workers" => {
                workers = Some(parse_usize_value(args, i, "--workers")?.max(1));
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
    let go_version = go_version_string(&go_bin)?;
    let repair_workers = workers.unwrap_or_else(default_go_repair_workers);
    let params = json!({
        "go_bin": go_bin,
        "go_version": go_version,
        "seed": seed,
        "timeout_sec": timeout_sec,
        "variants_per_sample": variants_per_sample,
        "max_rows": max_rows,
        "workers": repair_workers,
        "cache_strategy": "shared-warmed-go-build-cache-v1",
    });
    let inputs = vec![input.clone()];
    if !force {
        if let Some(manifest) = artifact_cache_hit("go_repair_pairs", &output, &inputs, &params)? {
            println!(
                "Go repair-pair cache hit: {} (rows={})",
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
    let mut seen = HashSet::new();
    let mut written = 0usize;
    let tmp = output.with_extension("txt.tmp");
    let mut out = BufWriter::new(File::create(&tmp)?);
    let go_feedback = GoCompileFeedback::new(&go_bin, &go_version, timeout_sec)?;
    println!(
        "Go repair pairs parallel prepare: workers={} shared_cache={}",
        repair_workers,
        go_feedback.cache_dir.display()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(repair_workers)
        .build()
        .context("build Go repair worker pool")?;
    let mut processed = 0usize;
    let mut batch_start = 0usize;
    while batch_start < pairs.len() && (max_rows == 0 || written < max_rows) {
        let remaining_rows = max_rows.saturating_sub(written);
        let target_pairs = if max_rows == 0 {
            4096
        } else {
            ((remaining_rows + variants_per_sample).saturating_div(variants_per_sample.max(1)))
                .saturating_mul(2)
                .max(256)
        };
        let batch_len = target_pairs.min(4096).min(pairs.len() - batch_start);
        let batch_end = batch_start + batch_len;
        let mut batch_rows = pool.install(|| {
            pairs[batch_start..batch_end]
                .par_iter()
                .enumerate()
                .map(|(offset, (task_prompt, correct_code))| {
                    let idx = batch_start + offset;
                    prepare_go_repair_rows_for_pair(
                        idx,
                        task_prompt,
                        correct_code,
                        seed,
                        variants_per_sample,
                        &go_feedback,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?;
        batch_rows.sort_by_key(|rows| rows.pair_index);
        for pair_rows in batch_rows {
            processed += 1;
            for row in pair_rows.rows {
                if max_rows > 0 && written >= max_rows {
                    break;
                }
                if !seen.insert(row.digest) {
                    continue;
                }
                writeln!(
                    out,
                    "{}\t{}",
                    escape_pair_field(&row.prompt),
                    escape_pair_field(&row.correct_code)
                )?;
                written += 1;
            }
            if progress_every > 0 && processed.is_multiple_of(progress_every) {
                println!("Go repair pairs progress: processed={processed} written={written}");
            }
        }
        batch_start = batch_end;
    }
    out.flush()?;
    fs::rename(tmp, &output)?;
    write_artifact_manifest("go_repair_pairs", &output, &inputs, params, written)?;
    println!("Wrote {written} Go repair rows to {}", output.display());
    Ok(())
}

pub(crate) fn default_go_repair_workers() -> usize {
    thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or_else(|_| rayon::current_num_threads().max(1))
        .saturating_sub(2)
        .max(1)
}

struct GoRepairPairRows {
    pair_index: usize,
    rows: Vec<GoRepairRow>,
}

struct GoRepairRow {
    digest: String,
    prompt: String,
    correct_code: String,
}

fn prepare_go_repair_rows_for_pair(
    pair_index: usize,
    task_prompt: &str,
    correct_code: &str,
    seed: u64,
    variants_per_sample: usize,
    go_feedback: &GoCompileFeedback,
) -> Result<GoRepairPairRows> {
    let mut variants = go_corruption_variants(correct_code);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ ((pair_index as u64) << 32));
    variants.shuffle(&mut rng);
    let mut rows = Vec::new();
    for (_name, broken_code) in variants {
        let feedback = go_feedback.compile(&broken_code)?;
        if feedback.is_empty() {
            continue;
        }
        let prompt = build_language_repair_prompt("Go", "go", task_prompt, &broken_code, &feedback);
        let digest = format!("{:x}", md5::compute(format!("{prompt}\t{correct_code}")));
        rows.push(GoRepairRow {
            digest,
            prompt,
            correct_code: correct_code.to_string(),
        });
        if rows.len() >= variants_per_sample.max(1) {
            break;
        }
    }
    Ok(GoRepairPairRows { pair_index, rows })
}

pub(crate) fn go_version_string(go_bin: &str) -> Result<String> {
    let out = Command::new(go_bin).arg("version").output()?;
    if out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !stdout.is_empty() {
            return Ok(stdout);
        }
    }
    bail!("failed to query Go version from '{go_bin}'")
}

pub(crate) struct GoCompileFeedback {
    go_bin: String,
    timeout_sec: f64,
    cache_dir: PathBuf,
    mod_cache_dir: PathBuf,
    tmp_dir: PathBuf,
}

impl GoCompileFeedback {
    pub(crate) fn new(go_bin: &str, go_version: &str, timeout_sec: f64) -> Result<Self> {
        let cache_root = std::env::var_os("TOFY_GO_REPAIR_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "tofy-go-repair-cache-{:x}",
                    md5::compute(go_version)
                ))
            });
        let cache_dir = std::env::var_os("GOCACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_root.join("go-build"));
        let mod_cache_dir = std::env::var_os("GOMODCACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_root.join("go-mod"));
        let tmp_dir = std::env::var_os("GOTMPDIR")
            .or_else(|| std::env::var_os("TMPDIR"))
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_root.join("go-tmp"));
        fs::create_dir_all(&cache_dir)?;
        fs::create_dir_all(&mod_cache_dir)?;
        fs::create_dir_all(&tmp_dir)?;
        let runner = Self {
            go_bin: go_bin.to_string(),
            timeout_sec,
            cache_dir,
            mod_cache_dir,
            tmp_dir,
        };
        runner.warm_cache()?;
        Ok(runner)
    }

    fn warm_cache(&self) -> Result<()> {
        let _ = go_compile_feedback(
            &self.go_bin,
            "func tofyGoRepairCacheWarmup() {}",
            self.timeout_sec.max(120.0),
            &self.cache_dir,
            &self.mod_cache_dir,
            &self.tmp_dir,
        )?;
        Ok(())
    }

    pub(crate) fn compile(&self, code: &str) -> Result<String> {
        go_compile_feedback(
            &self.go_bin,
            code,
            self.timeout_sec,
            &self.cache_dir,
            &self.mod_cache_dir,
            &self.tmp_dir,
        )
    }
}

pub(crate) fn load_escaped_pairs(path: &Path) -> Result<Vec<(String, String)>> {
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

fn go_corruption_variants(code: &str) -> Vec<(String, String)> {
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
    if code.contains(":=") {
        variants.push((
            "broken_short_assign".to_string(),
            code.replacen(":=", "=", 1),
        ));
    }
    if code.contains("!=") {
        variants.push(("broken_comparison".to_string(), code.replacen("!=", "=", 1)));
    }
    let func_name_re = Regex::new(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    if let Some(capture) = func_name_re.captures(code) {
        let name = capture.get(1).unwrap().as_str();
        variants.push((
            "wrong_func_name".to_string(),
            code.replacen(&format!("func {name}"), &format!("func Broken{name}"), 1),
        ));
    }
    variants
}

fn run_command_with_timeout(cmd: &mut Command, timeout_sec: f64) -> Result<std::process::Output> {
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
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
            terminate_timed_out_child(&mut child);
            return Ok(child.wait_with_output()?);
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn terminate_timed_out_child(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pgid = format!("-{}", child.id());
        let _ = Command::new("kill").arg("-TERM").arg(&pgid).status();
        thread::sleep(Duration::from_millis(100));
        if child.try_wait().ok().flatten().is_none() {
            let _ = Command::new("kill").arg("-KILL").arg(&pgid).status();
        }
    }
    let _ = child.kill();
}

fn go_compile_feedback(
    go_bin: &str,
    code: &str,
    timeout_sec: f64,
    cache_dir: &Path,
    mod_cache_dir: &Path,
    tmp_dir: &Path,
) -> Result<String> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("tofy-go-repair-{unique}"));
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("go.mod"), "module tofy_repair\n\ngo 1.22\n")?;
    fs::write(
        dir.join("candidate.go"),
        format!("package main\n\n{}\n", strip_go_package_line(code)),
    )?;
    let mut cmd = Command::new(go_bin);
    cmd.arg("test")
        .arg("-c")
        .arg("-p")
        .arg("1")
        .arg(".")
        .current_dir(&dir)
        .env("GOCACHE", cache_dir)
        .env("GOMODCACHE", mod_cache_dir)
        .env("GOTMPDIR", tmp_dir)
        .env("TMPDIR", tmp_dir)
        .env("GOMAXPROCS", "1");
    let output = run_command_with_timeout(&mut cmd, timeout_sec)?;
    let _ = fs::remove_dir_all(&dir);
    if output.status.success() {
        return Ok(String::new());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = Vec::new();
    for line in stderr.lines().chain(stdout.lines()) {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
        if lines.len() >= 12 {
            break;
        }
    }
    Ok(lines.join("\n"))
}

fn strip_go_package_line(code: &str) -> String {
    let package_re = Regex::new(r"(?m)^\s*package\s+\w+\s*$").unwrap();
    package_re.replace_all(code, "").trim().to_string()
}

fn code_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<lang:[a-z0-9_+\-#]+>\s*<(?:ctx|reply)>\s*").unwrap())
}

fn go_func_start_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?m)^(?P<sig>\s*func\s+(?:\([^)]+\)\s*)?[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:\([^{};]*\)|[A-Za-z_][A-Za-z0-9_\.\*\[\]]*)?\s*)\{",
        )
        .unwrap()
    })
}

pub(crate) fn build_language_repair_prompt(
    language: &str,
    fence: &str,
    task_prompt: &str,
    broken_code: &str,
    feedback: &str,
) -> String {
    format!(
        "Return only corrected {language} code.\nFix the previous attempt using the compiler feedback.\n\nOriginal request:\n{task_prompt}\n\nPrevious attempt:\n```{fence}\n{broken_code}\n```\n\nCompiler feedback:\n{feedback}\n\nRules:\n- Keep the exact requested function name and signature.\n- Return only compilable {language} code.\n- Do not add explanation.\n"
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
    println!(
        "World mix streaming load: code_inputs={} rayon_threads={}",
        code_pairs.len(),
        rayon::current_num_threads()
    );
    let text_total = count_raw_pairs(&text_pairs)?;
    let code_total = count_raw_pairs_many(&code_pairs)?;
    if text_total == 0 {
        bail!("no usable text rows found in {}", text_pairs.display());
    }
    if code_total == 0 {
        bail!("no usable code rows found");
    }
    let total_rows = text_total + code_total;
    let max_rows = if max_rows > 0 { max_rows } else { total_rows };
    let target_rows = max_rows.min(total_rows);
    let code_ratio = code_ratio.clamp(0.0, 0.8);
    let done_ratio = done_ratio.clamp(0.0, 0.4);
    let target_done = if done_ratio > 0.0 && target_rows > 0 {
        ((target_rows as f64 * done_ratio) / (1.0 - done_ratio).max(1e-6)).round() as usize
    } else {
        0
    };
    let tmp = atomic_tmp_path(&output);
    let done_tmp = side_tmp_path(&output, "done");
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&tmp)?);
    let mut done_out = BufWriter::new(File::create(&done_tmp)?);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut text_stream = RawPairStream::open(&text_pairs)?;
    let mut code_stream = MultiRawPairStream::open(&code_pairs)?;
    let mut remaining_text = text_total;
    let mut remaining_code = code_total;
    let mut remaining_candidates = target_rows;
    let mut remaining_done = target_done;
    let mut chosen_code = 0usize;
    let mut chosen_text = 0usize;
    let mut chosen_done = 0usize;
    while chosen_code + chosen_text < target_rows && (remaining_text > 0 || remaining_code > 0) {
        let want_code = rng.random::<f64>() < code_ratio;
        let use_code = (want_code && remaining_code > 0) || remaining_text == 0;
        let (left, right, action) = if use_code {
            let Some((left, right)) = code_stream.next_pair()? else {
                remaining_code = 0;
                continue;
            };
            remaining_code -= 1;
            let action = world_mix_code_action(&left, &right);
            chosen_code += 1;
            (left, right, action)
        } else {
            let Some((left, right)) = text_stream.next_pair()? else {
                remaining_text = 0;
                continue;
            };
            remaining_text -= 1;
            chosen_text += 1;
            (left, right, "text_reply")
        };
        writeln!(
            out,
            "{}\t{}\t{action}",
            escape_pair_field(&left),
            escape_pair_field(&right)
        )?;
        if remaining_done > 0
            && remaining_candidates > 0
            && rng.random_range(0..remaining_candidates) < remaining_done
        {
            let terminal_state = if action == "text_reply" {
                format!("{left}\\nAssistant: {right}")
            } else {
                format!("{left}\\n{right}")
            };
            writeln!(
                done_out,
                "{}\t<done>\tdone",
                escape_pair_field(&terminal_state)
            )?;
            remaining_done -= 1;
            chosen_done += 1;
        }
        remaining_candidates = remaining_candidates.saturating_sub(1);
    }
    out.flush()?;
    done_out.flush()?;
    drop(done_out);
    let mut done_reader = BufReader::new(File::open(&done_tmp)?);
    std::io::copy(&mut done_reader, &mut out)?;
    out.flush()?;
    drop(out);
    fs::rename(&tmp, &output)?;
    let _ = fs::remove_file(&done_tmp);
    let output_len = chosen_text + chosen_code + chosen_done;
    write_artifact_manifest("world_mix", &output, &inputs, params, output_len)?;
    let actual_code_rate = chosen_code as f64 / output_len.max(1) as f64;
    let actual_done_rate = chosen_done as f64 / output_len.max(1) as f64;
    println!(
        "Wrote {} rows to {} (text_rows={}, code_rows={}, done_rows={}, requested code_ratio={:.2}, requested done_ratio={:.2})",
        output_len,
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
    if let Some(action) = explicit_world_mix_action(right) {
        return action;
    }
    let left_lower = left.to_ascii_lowercase();
    if left_lower.contains("<action:fetch_docs>") || left_lower.contains("<tool:fetch_docs>") {
        "fetch_docs"
    } else {
        "code"
    }
}

fn explicit_world_mix_action(text: &str) -> Option<&'static str> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("<action:")?;
    let label = rest.split_once('>')?.0.trim().to_ascii_lowercase();
    match label.as_str() {
        "text" | "text_reply" | "explain" | "summarize" => Some("text_reply"),
        "code" | "inspect_file" | "read_file" | "edit_file" | "apply_patch" | "run_tests"
        | "read_error" | "repair_patch" | "compiler_feedback" => Some("code"),
        "done" | "final" | "finalize" | "stop" => Some("done"),
        "fetch_docs" | "docs" | "retrieve_docs" | "doc_lookup" => Some("fetch_docs"),
        _ => None,
    }
}

fn split_pair_line_first_two(line: &str) -> Option<(String, String)> {
    let parts = if line.contains('\t') {
        line.split('\t').collect::<Vec<_>>()
    } else if line.contains("|||") {
        line.split("|||").collect::<Vec<_>>()
    } else {
        return None;
    };
    let left = parts.first()?.trim().to_string();
    let mut right = parts.get(1)?.trim().to_string();
    if let Some(action) = parts
        .get(2)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        if !right.trim_start().starts_with("<action:") {
            right = format!("<action:{action}> {right}");
        }
    }
    Some((left, right))
}

fn count_raw_pairs(path: &Path) -> Result<usize> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let chunk_lines = prepare_chunk_lines();
    let mut count = 0usize;
    loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        count += chunk
            .par_iter()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && (trimmed.contains('\t') || trimmed.contains("|||"))
            })
            .count();
    }
    Ok(count)
}

fn count_raw_pairs_many(paths: &[PathBuf]) -> Result<usize> {
    paths
        .par_iter()
        .map(|path| count_raw_pairs(path))
        .collect::<Result<Vec<_>>>()
        .map(|counts| counts.into_iter().sum())
}

struct RawPairStream {
    lines: std::io::Lines<BufReader<File>>,
    buffer: VecDeque<(String, String)>,
    chunk_lines: usize,
}

impl RawPairStream {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            lines: BufReader::new(File::open(path)?).lines(),
            buffer: VecDeque::new(),
            chunk_lines: prepare_chunk_lines(),
        })
    }

    fn next_pair(&mut self) -> Result<Option<(String, String)>> {
        loop {
            if let Some(pair) = self.buffer.pop_front() {
                return Ok(Some(pair));
            }
            let chunk = read_line_chunk(&mut self.lines, self.chunk_lines)?;
            if chunk.is_empty() {
                return Ok(None);
            }
            self.buffer = VecDeque::from(
                chunk
                    .par_iter()
                    .filter_map(|line| {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            return None;
                        }
                        split_pair_line_first_two(trimmed)
                    })
                    .collect::<Vec<_>>(),
            );
        }
    }
}

struct MultiRawPairStream {
    streams: Vec<RawPairStream>,
    index: usize,
}

impl MultiRawPairStream {
    fn open(paths: &[PathBuf]) -> Result<Self> {
        let streams = paths
            .iter()
            .map(|path| RawPairStream::open(path))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { streams, index: 0 })
    }

    fn next_pair(&mut self) -> Result<Option<(String, String)>> {
        while self.index < self.streams.len() {
            if let Some(pair) = self.streams[self.index].next_pair()? {
                return Ok(Some(pair));
            }
            self.index += 1;
        }
        Ok(None)
    }
}

fn run_prepare_code_poc_mix(args: &[String]) -> Result<()> {
    let mut output = None;
    let mut base_pairs = None;
    let mut instruction_pairs = None;
    let mut extra_pairs = Vec::new();
    let mut base_repeat = 1usize;
    let mut instruction_repeat = 3usize;
    let mut extra_repeat = 1usize;
    let mut fim_repeat = 0usize;
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
            "--base-repeat" => {
                base_repeat = parse_usize_value(args, i, "--base-repeat")?;
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
            "--fim-repeat" => {
                fim_repeat = parse_usize_value(args, i, "--fim-repeat")?;
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
        "base_repeat": base_repeat,
        "instruction_repeat": instruction_repeat,
        "extra_repeat": extra_repeat,
        "fim_repeat": fim_repeat,
        "code_targets_only": true,
        "target_stop_token": CODE_EOS_TOKEN,
        "max_rows": if max_rows > 0 { Some(max_rows) } else { None::<usize> },
        "seed": seed,
        "row_shuffle": "bucket_v1",
        "shuffle_buckets": code_mix_shuffle_buckets(),
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
    println!(
        "Code POC mix streaming write: extra_inputs={} rayon_threads={}",
        extra_pairs.len(),
        rayon::current_num_threads()
    );
    let mut units = Vec::new();
    for _ in 0..instruction_repeat {
        units.push(CodePocMixUnit::Instruction);
    }
    for _ in 0..extra_repeat {
        for path in &extra_pairs {
            units.push(CodePocMixUnit::Extra(path.clone()));
        }
    }
    for _ in 0..fim_repeat {
        units.push(CodePocMixUnit::Fim);
    }
    for _ in 0..base_repeat {
        units.push(CodePocMixUnit::Base);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    units.shuffle(&mut rng);
    let tmp = atomic_tmp_path(&output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = CodePocShuffledWriter::new(&tmp, seed, code_mix_shuffle_buckets())?;
    let mut remaining = if max_rows > 0 { Some(max_rows) } else { None };
    let mut written = 0usize;
    let mut fim_rows = 0usize;
    for unit in units {
        if !has_row_budget(&remaining) {
            break;
        }
        match unit {
            CodePocMixUnit::Base => {
                written += write_code_poc_rows_from_path(&mut out, &base_pairs, &mut remaining)?;
            }
            CodePocMixUnit::Instruction => {
                written +=
                    write_code_poc_rows_from_path(&mut out, &instruction_pairs, &mut remaining)?;
            }
            CodePocMixUnit::Extra(path) => {
                written += write_code_poc_rows_from_path(&mut out, &path, &mut remaining)?;
            }
            CodePocMixUnit::Fim => {
                let wrote = write_code_poc_fim_rows(
                    &mut out,
                    &instruction_pairs,
                    &extra_pairs,
                    &mut remaining,
                )?;
                written += wrote;
                fim_rows += wrote;
            }
        }
    }
    out.finish()?;
    fs::rename(&tmp, &output)?;
    write_artifact_manifest("code_poc_mix", &output, &inputs, params, written)?;
    println!(
        "Wrote {} mixed code POC rows to {} (streamed_fim_rows={})",
        written,
        output.display(),
        fim_rows
    );
    Ok(())
}

#[derive(Clone)]
enum CodePocMixUnit {
    Base,
    Instruction,
    Extra(PathBuf),
    Fim,
}

fn has_row_budget(remaining: &Option<usize>) -> bool {
    remaining.map(|rows| rows > 0).unwrap_or(true)
}

struct CodePocShuffledWriter {
    output: PathBuf,
    bucket_dir: PathBuf,
    buckets: Vec<BufWriter<File>>,
    rng: rand::rngs::StdRng,
}

impl CodePocShuffledWriter {
    fn new(output: &Path, seed: u64, bucket_count: usize) -> Result<Self> {
        let bucket_dir = side_tmp_path(output, "shuffle-buckets");
        if bucket_dir.exists() {
            fs::remove_dir_all(&bucket_dir)?;
        }
        fs::create_dir_all(&bucket_dir)?;
        let buckets = (0..bucket_count.max(1))
            .map(|idx| File::create(bucket_dir.join(format!("{idx:04}.rows"))).map(BufWriter::new))
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Self {
            output: output.to_path_buf(),
            bucket_dir,
            buckets,
            rng: rand::rngs::StdRng::seed_from_u64(seed ^ 0xA076_1D64_78BD_642F),
        })
    }

    fn write_row(&mut self, row: &str) -> Result<()> {
        let bucket = self.rng.random_range(0..self.buckets.len());
        writeln!(self.buckets[bucket], "{row}")?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        let bucket_count = self.buckets.len();
        for bucket in &mut self.buckets {
            bucket.flush()?;
        }
        drop(self.buckets);

        let mut order = (0..bucket_count).collect::<Vec<_>>();
        order.shuffle(&mut self.rng);
        let mut output = BufWriter::new(File::create(&self.output)?);
        for idx in order {
            let path = self.bucket_dir.join(format!("{idx:04}.rows"));
            let mut rows = BufReader::new(File::open(&path)?)
                .lines()
                .collect::<std::io::Result<Vec<_>>>()?;
            rows.shuffle(&mut self.rng);
            for row in rows {
                writeln!(output, "{row}")?;
            }
        }
        output.flush()?;
        fs::remove_dir_all(&self.bucket_dir)?;
        Ok(())
    }
}

fn write_limited_row(
    out: &mut CodePocShuffledWriter,
    row: &str,
    remaining: &mut Option<usize>,
) -> Result<bool> {
    if !has_row_budget(remaining) {
        return Ok(false);
    }
    out.write_row(row)?;
    if let Some(rows) = remaining.as_mut() {
        *rows -= 1;
    }
    Ok(true)
}

fn write_code_poc_rows_from_path(
    out: &mut CodePocShuffledWriter,
    path: &Path,
    remaining: &mut Option<usize>,
) -> Result<usize> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let chunk_lines = prepare_chunk_lines();
    let mut written = 0usize;
    loop {
        if !has_row_budget(remaining) {
            break;
        }
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        let rows = chunk
            .into_par_iter()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| code_poc_row_with_eos(&line))
            .collect::<Vec<_>>();
        for row in rows {
            if write_limited_row(out, &row, remaining)? {
                written += 1;
            } else {
                break;
            }
        }
    }
    Ok(written)
}

fn write_code_poc_fim_rows(
    out: &mut CodePocShuffledWriter,
    instruction_pairs: &Path,
    extra_pairs: &[PathBuf],
    remaining: &mut Option<usize>,
) -> Result<usize> {
    let mut written = write_code_poc_fim_rows_from_path(out, instruction_pairs, remaining)?;
    for path in extra_pairs {
        if !has_row_budget(remaining) {
            break;
        }
        written += write_code_poc_fim_rows_from_path(out, path, remaining)?;
    }
    Ok(written)
}

fn write_code_poc_fim_rows_from_path(
    out: &mut CodePocShuffledWriter,
    path: &Path,
    remaining: &mut Option<usize>,
) -> Result<usize> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let chunk_lines = prepare_chunk_lines();
    let mut written = 0usize;
    loop {
        if !has_row_budget(remaining) {
            break;
        }
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        let rows = chunk
            .into_par_iter()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| code_poc_row_with_eos(&line).and_then(|row| code_poc_fim_row(&row)))
            .collect::<Vec<_>>();
        for row in rows {
            if write_limited_row(out, &row, remaining)? {
                written += 1;
            } else {
                break;
            }
        }
    }
    Ok(written)
}

fn code_poc_row_with_eos(row: &str) -> Option<String> {
    let (left, right) = row.split_once('\t')?;
    let state = unescape_pair_field(left);
    let target = code_poc_code_target(&unescape_pair_field(right))?;
    let target = append_code_eos(&target);
    Some(format!(
        "{}\t{}",
        escape_pair_field(&state),
        escape_pair_field(&target)
    ))
}

fn code_poc_code_target(target: &str) -> Option<String> {
    if let Some((_, body)) = split_explicit_action_prefix(target) {
        return (explicit_world_mix_action(target) == Some("code"))
            .then(|| body.trim_start().to_string());
    }
    Some(target.to_string())
}

fn split_explicit_action_prefix(text: &str) -> Option<(&str, &str)> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("<action:")?;
    let (label, after_label) = rest.split_once('>')?;
    Some((label.trim(), after_label))
}

fn append_code_eos(text: &str) -> String {
    let trimmed = text.trim_end();
    if trimmed.ends_with(CODE_EOS_TOKEN) {
        trimmed.to_string()
    } else if trimmed.is_empty() {
        CODE_EOS_TOKEN.to_string()
    } else {
        format!("{trimmed}\n{CODE_EOS_TOKEN}")
    }
}

fn strip_code_eos(text: &str) -> String {
    text.find(CODE_EOS_TOKEN)
        .map(|idx| text[..idx].trim_end().to_string())
        .unwrap_or_else(|| text.trim_end().to_string())
}

fn code_poc_fim_row(row: &str) -> Option<String> {
    let (left, right) = row.split_once('\t')?;
    let state = unescape_pair_field(left);
    let code = strip_code_eos(&unescape_pair_field(right));
    let (prefix, middle, suffix) = split_code_for_fim(&code)?;
    let middle = middle.trim_matches('\n');
    if middle.chars().filter(|ch| !ch.is_whitespace()).count() < 12 {
        return None;
    }
    let prompt = format!(
        "{}\n<fim_prefix>\n{}\n<fim_suffix>\n{}\n<fim_middle>\n",
        state.trim_end(),
        prefix.trim_end(),
        suffix.trim_start()
    );
    Some(format!(
        "{}\t{}",
        escape_pair_field(&prompt),
        escape_pair_field(&append_code_eos(middle))
    ))
}

fn split_code_for_fim(code: &str) -> Option<(String, String, String)> {
    if let Some(capture) = go_func_start_re().captures(code) {
        let whole = capture.get(0)?;
        let open_idx = whole.end().checked_sub(1)?;
        let close_idx = find_matching_brace(code, open_idx)?;
        if close_idx > open_idx + 1 {
            return Some((
                code[..=open_idx].to_string(),
                code[open_idx + 1..close_idx].to_string(),
                code[close_idx..].to_string(),
            ));
        }
    }
    split_code_for_fim_by_lines(code)
}

fn split_code_for_fim_by_lines(code: &str) -> Option<(String, String, String)> {
    let lines = code.lines().collect::<Vec<_>>();
    if lines.len() < 6 {
        return None;
    }
    let first_end = (lines.len() / 3).max(1);
    let middle_end = (lines.len() * 2 / 3)
        .max(first_end + 1)
        .min(lines.len() - 1);
    let prefix = lines[..first_end].join("\n");
    let middle = lines[first_end..middle_end].join("\n");
    let suffix = lines[middle_end..].join("\n");
    if prefix.trim().is_empty() || middle.trim().is_empty() || suffix.trim().is_empty() {
        None
    } else {
        Some((prefix, middle, suffix))
    }
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
    let mut output = PathBuf::from("eval/code_assistant_go_hard.jsonl");
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
    println!("Wrote {rows} Go tasks to {}", output.display());
    Ok(())
}

fn run_generate_go_code_eval_suite(args: &[String]) -> Result<()> {
    let mut output = PathBuf::from("eval/code_assistant_go_hard.jsonl");
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
    println!("Wrote {rows} Go tasks to {}", output.display());
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
        .filter(|path| {
            matches!(
                path.extension().and_then(OsStr::to_str),
                Some("json") | Some("jsonl")
            )
        })
        .collect();
    json_files.sort();
    if json_files.is_empty() {
        bail!("no .json/.jsonl files found in {}", input_dir.display());
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
                .to_string();
            let rsp = value
                .get("response")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let ctx_tokens =
                crate::data::tokenize_with_mode(&ctx, crate::data::TokenizationMode::CodeAware)
                    .len();
            let rsp_tokens =
                crate::data::tokenize_with_mode(&rsp, crate::data::TokenizationMode::CodeAware)
                    .len();
            if ctx.is_empty()
                || rsp.is_empty()
                || ctx_tokens < min_tokens
                || rsp_tokens < min_tokens
                || ctx_tokens > max_tokens
                || rsp_tokens > max_tokens
            {
                skipped += 1;
                continue;
            }
            content.push_str(&escape_pair_field(&ctx));
            content.push('\t');
            content.push_str(&escape_pair_field(&rsp));
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
        "high-world" | "high_world" | "highworld" => Ok(ProbeStage::HighWorld),
        "decoder" => Ok(ProbeStage::Decoder),
        other => bail!("--stage must be one of all|latent|world|high-world|decoder, got {other}"),
    }
}

fn default_oom_probe_args() -> OomProbeArgs {
    OomProbeArgs {
        stage: ProbeStage::All,
        quick: false,
        max_vram: false,
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
        latent_warmup_batch: 12,
        latent_warmup_accum: 1,
        latent_warmup_steps: None,
        world_steps: 8000,
        world_batch: 64,
        world_accum: 2,
        world_warmup_batch: 64,
        world_warmup_accum: 1,
        world_warmup_steps: None,
        world_lambda: 0.2,
        world_lr: 2e-4,
        world_action_loss_weight: 1.0,
        high_world_steps: 1000,
        high_world_batch: 64,
        high_world_accum: 2,
        high_world_warmup_batch: 64,
        high_world_warmup_accum: 1,
        high_world_warmup_steps: None,
        decoder_steps: 1000,
        decoder_batch: 6,
        decoder_accum: 4,
        decoder_warmup_batch: 6,
        decoder_warmup_accum: 1,
        decoder_warmup_steps: None,
        decoder_max_seq: 160,
        decoder_max_vocab: 24_000,
        decoder_dim: 640,
        decoder_layers: 6,
        decoder_heads: 8,
        decoder_ff_dim: 3072,
        setup_latent_steps: 1,
        setup_world_steps: 2,
        latent_model: None,
        encoder_vocab: None,
        world_model: None,
    }
}

struct ProbeContextDefaults {
    latent_segments: usize,
    world_segments: usize,
    retrieval_slots: usize,
    exact_old_tokens: usize,
    segment_batch: usize,
}

fn probe_context_defaults(args: &OomProbeArgs) -> ProbeContextDefaults {
    if args.dim >= 1024 || args.max_seq >= 384 || args.planner_slots >= 128 {
        ProbeContextDefaults {
            latent_segments: 8,
            world_segments: 8,
            retrieval_slots: 16,
            exact_old_tokens: 32,
            segment_batch: 128,
        }
    } else if args.dim >= 768 || args.max_seq >= 320 || args.planner_slots >= 96 {
        ProbeContextDefaults {
            latent_segments: 6,
            world_segments: 6,
            retrieval_slots: 12,
            exact_old_tokens: 24,
            segment_batch: 64,
        }
    } else {
        ProbeContextDefaults {
            latent_segments: 4,
            world_segments: 4,
            retrieval_slots: 8,
            exact_old_tokens: 16,
            segment_batch: 16,
        }
    }
}

fn model_profiles_path() -> PathBuf {
    std::env::var("TOFY_MODEL_PROFILES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("config/model_profiles.json"))
}

fn load_oom_probe_profile(name: &str) -> Result<OomProbeProfile> {
    let path = model_profiles_path();
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read model profile config from {:?}", path))?;
    let profiles: OomProbeProfileFile = serde_json::from_str(&raw)
        .with_context(|| format!("parse model profile config from {:?}", path))?;
    match name {
        "8gb" => Ok(profiles.eight_gb),
        "48gb" => Ok(profiles.forty_eight_gb),
        "80gb" => Ok(profiles.eighty_gb),
        other => bail!("--profile must be one of 8gb|48gb|80gb, got {other}"),
    }
}

fn apply_oom_probe_profile(args: &mut OomProbeArgs, profile: OomProbeProfile) {
    args.dim = profile.dim;
    args.max_seq = profile.latent_max_seq.max(profile.world_max_seq);
    args.layers = profile.layers;
    args.heads = profile.heads;
    args.bridge_dim = profile.bridge_dim;
    args.planner_slots = profile.num_latent_tokens;
    args.vocab = profile.max_vocab;
    args.latent_steps = profile.latent_steps;
    args.latent_batch = profile.latent_batch;
    args.latent_accum = profile.latent_grad_accum;
    args.latent_warmup_batch = profile.latent_warmup_batch;
    args.latent_warmup_accum = 1;
    args.world_steps = profile.world_steps;
    args.world_batch = profile.world_batch;
    args.world_accum = profile.world_grad_accum;
    args.world_warmup_batch = profile.world_warmup_batch;
    args.world_warmup_accum = 1;
    args.high_world_steps = profile.high_world_steps;
    args.high_world_batch = profile.world_batch;
    args.high_world_accum = profile.world_grad_accum;
    args.high_world_warmup_batch = profile.world_warmup_batch.min(args.high_world_batch);
    args.high_world_warmup_accum = 1;
    args.decoder_steps = profile.code_decoder_steps;
    args.decoder_batch = profile.code_decoder_batch;
    args.decoder_accum = profile.code_decoder_grad_accum;
    args.decoder_warmup_batch = profile.world_warmup_batch.min(args.decoder_batch);
    args.decoder_warmup_accum = 1;
    args.decoder_max_seq = profile.code_decoder_max_seq;
    args.decoder_max_vocab = profile.code_decoder_max_vocab;
    args.decoder_dim = profile.decoder_dim;
    args.decoder_layers = profile.decoder_layers;
    args.decoder_heads = profile.decoder_heads;
    args.decoder_ff_dim = profile.decoder_ff_dim;
}

fn apply_max_vram_probe_defaults(args: &mut OomProbeArgs) {
    args.max_vram = true;
    args.latent_steps = args.latent_steps.min(200);
    args.world_steps = args.world_steps.min(200);
    args.high_world_steps = args.high_world_steps.min(200);
    args.decoder_steps = args.decoder_steps.min(200);
    args.setup_latent_steps = args.setup_latent_steps.clamp(4, 32);
    args.setup_world_steps = args.setup_world_steps.clamp(4, 32);
    args.latent_warmup_steps = Some(args.latent_warmup_steps.unwrap_or(10).min(10));
    args.world_warmup_steps = Some(args.world_warmup_steps.unwrap_or(10).min(10));
    args.high_world_warmup_steps = Some(args.high_world_warmup_steps.unwrap_or(10).min(10));
    args.decoder_warmup_steps = Some(args.decoder_warmup_steps.unwrap_or(10).min(10));
    args.max_late_growth_mb = 0;
    args.sample_interval_sec = args.sample_interval_sec.min(0.10);
}

fn parse_sustained_oom_probe_args(args: &[String]) -> Result<OomProbeArgs> {
    let mut parsed = default_oom_probe_args();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--profile" {
            let profile_name = parse_flag_value(args, i, "--profile")?;
            let profile = load_oom_probe_profile(&profile_name)?;
            apply_oom_probe_profile(&mut parsed, profile);
        }
    }
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 2;
            }
            "--stage" => {
                parsed.stage = parse_probe_stage(&parse_flag_value(args, i, "--stage")?)?;
                i += 2;
            }
            "--max-vram" | "--max_vram" => {
                parsed.max_vram = true;
                i += 1;
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
            "--latent-warmup-batch" | "--latent_warmup_batch" => {
                parsed.latent_warmup_batch = parse_usize_value(args, i, "--latent-warmup-batch")?;
                i += 2;
            }
            "--latent-warmup-accum" | "--latent_warmup_accum" => {
                parsed.latent_warmup_accum = parse_usize_value(args, i, "--latent-warmup-accum")?;
                i += 2;
            }
            "--latent-warmup-steps" | "--latent_warmup_steps" => {
                parsed.latent_warmup_steps =
                    Some(parse_usize_value(args, i, "--latent-warmup-steps")?);
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
            "--high-world-steps" | "--high_world_steps" => {
                parsed.high_world_steps = parse_usize_value(args, i, "--high-world-steps")?;
                i += 2;
            }
            "--high-world-batch" | "--high_world_batch" => {
                parsed.high_world_batch = parse_usize_value(args, i, "--high-world-batch")?;
                i += 2;
            }
            "--high-world-accum" | "--high_world_accum" => {
                parsed.high_world_accum = parse_usize_value(args, i, "--high-world-accum")?;
                i += 2;
            }
            "--high-world-warmup-batch" | "--high_world_warmup_batch" => {
                parsed.high_world_warmup_batch =
                    parse_usize_value(args, i, "--high-world-warmup-batch")?;
                i += 2;
            }
            "--high-world-warmup-accum" | "--high_world_warmup_accum" => {
                parsed.high_world_warmup_accum =
                    parse_usize_value(args, i, "--high-world-warmup-accum")?;
                i += 2;
            }
            "--high-world-warmup-steps" | "--high_world_warmup_steps" => {
                parsed.high_world_warmup_steps =
                    Some(parse_usize_value(args, i, "--high-world-warmup-steps")?);
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
            "--decoder-warmup-batch" | "--decoder_warmup_batch" => {
                parsed.decoder_warmup_batch = parse_usize_value(args, i, "--decoder-warmup-batch")?;
                i += 2;
            }
            "--decoder-warmup-accum" | "--decoder_warmup_accum" => {
                parsed.decoder_warmup_accum = parse_usize_value(args, i, "--decoder-warmup-accum")?;
                i += 2;
            }
            "--decoder-warmup-steps" | "--decoder_warmup_steps" => {
                parsed.decoder_warmup_steps =
                    Some(parse_usize_value(args, i, "--decoder-warmup-steps")?);
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
            "--decoder-dim" | "--decoder_dim" => {
                parsed.decoder_dim = parse_usize_value(args, i, "--decoder-dim")?;
                i += 2;
            }
            "--decoder-layers" | "--decoder_layers" => {
                parsed.decoder_layers = parse_usize_value(args, i, "--decoder-layers")?;
                i += 2;
            }
            "--decoder-heads" | "--decoder_heads" => {
                parsed.decoder_heads = parse_usize_value(args, i, "--decoder-heads")?;
                i += 2;
            }
            "--decoder-ff-dim" | "--decoder_ff_dim" => {
                parsed.decoder_ff_dim = parse_usize_value(args, i, "--decoder-ff-dim")?;
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
        parsed.high_world_steps = parsed.high_world_steps.min(4);
        parsed.decoder_steps = parsed.decoder_steps.min(3);
        parsed.min_headroom_mb = parsed.min_headroom_mb.min(128);
        parsed.max_late_growth_mb = 0;
    }
    if parsed.max_vram {
        apply_max_vram_probe_defaults(&mut parsed);
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
            "user asks go task {i} value_{i} condition_{} context_token_{i} helper_name_{i} return only go code",
            i % 97
        );
        let answer = format!(
            "<action:code> func function{i}(value{i} int) int {{ shifted{i} := value{i} + {}; if shifted{i} > {} {{ return shifted{i} }}; return shifted{i} + 1 }} unique_code_token_{i}",
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
    let context_defaults = probe_context_defaults(args);
    env.insert("TOFY_TRAIN_DTYPE".to_string(), args.dtype.clone());
    if let Some(path) = cached_encoder_vocab_for_probe(args) {
        env.insert(
            "TOFY_ENCODER_VOCAB".to_string(),
            path.to_string_lossy().to_string(),
        );
    }
    env.insert(
        "TOFY_LATENT_CONTEXT_SEGMENTS".to_string(),
        context_defaults.latent_segments.to_string(),
    );
    env.insert(
        "TOFY_LATENT_RECENT_FULL_SEGMENTS".to_string(),
        "1".to_string(),
    );
    env.insert("TOFY_LATENT_HISTORY_RATIO".to_string(), "0.35".to_string());
    env.insert(
        "TOFY_LATENT_WARMUP_BATCH".to_string(),
        args.latent_warmup_batch.to_string(),
    );
    env.insert(
        "TOFY_LATENT_WARMUP_GRAD_ACCUM".to_string(),
        args.latent_warmup_accum.to_string(),
    );
    match args.latent_warmup_steps {
        Some(steps) => {
            env.insert("TOFY_LATENT_WARMUP_STEPS".to_string(), steps.to_string());
        }
        None => {
            env.remove("TOFY_LATENT_WARMUP_STEPS");
        }
    }
    env.insert(
        "TOFY_WORLD_CONTEXT_SEGMENTS".to_string(),
        context_defaults.world_segments.to_string(),
    );
    env.insert(
        "TOFY_ENCODER_CONTEXT_SEGMENTS".to_string(),
        context_defaults.world_segments.to_string(),
    );
    env.insert(
        "TOFY_WORLD_RECENT_FULL_SEGMENTS".to_string(),
        "1".to_string(),
    );
    env.insert(
        "TOFY_RECURSIVE_CONTEXT_COMPRESSION".to_string(),
        "0".to_string(),
    );
    env.insert("TOFY_CONTEXT_HYBRID_MEMORY".to_string(), "1".to_string());
    env.insert(
        "TOFY_CONTEXT_HYBRID_EXACT_TAIL".to_string(),
        crate::tasks::world::default_context_hybrid_exact_tail(args.max_seq, 1).to_string(),
    );
    env.insert(
        "TOFY_CONTEXT_HYBRID_BLOCK_SIZE".to_string(),
        "32".to_string(),
    );
    env.insert(
        "TOFY_CONTEXT_RETRIEVAL_SLOTS".to_string(),
        context_defaults.retrieval_slots.to_string(),
    );
    env.insert(
        "TOFY_CONTEXT_EXACT_OLD_TOKENS".to_string(),
        context_defaults.exact_old_tokens.to_string(),
    );
    let decoder_local_window = args.decoder_max_seq.clamp(128, 256);
    env.insert(
        "TOFY_DECODER_LOCAL_WINDOW".to_string(),
        decoder_local_window.to_string(),
    );
    env.insert(
        "TOFY_DECODER_CSA_COMPRESS_RATE".to_string(),
        "8".to_string(),
    );
    env.insert(
        "TOFY_DECODER_HCA_COMPRESS_RATE".to_string(),
        "128".to_string(),
    );
    env.insert("TOFY_DECODER_ANCHOR_PERIOD".to_string(), "3".to_string());
    env.insert("TOFY_DECODER_CSA_TOPK".to_string(), "16".to_string());
    env.insert(
        "TOFY_DECODER_ATTENTION_QUERY_BLOCK".to_string(),
        decoder_local_window.clamp(64, 256).to_string(),
    );
    env.insert(
        "TOFY_WORLD_TRAIN_ROLLOUT_STEPS".to_string(),
        "2".to_string(),
    );
    env.insert("TOFY_WORLD_ROLLOUT_STEPS".to_string(), "2".to_string());
    env.insert(
        "TOFY_CONTEXT_SEGMENT_BATCH".to_string(),
        context_defaults.segment_batch.to_string(),
    );
    if args.max_vram {
        env.insert("TOFY_WORLD_LOG_EVERY".to_string(), "25".to_string());
        env.insert("TOFY_HIGH_WORLD_LOG_EVERY".to_string(), "25".to_string());
        env.insert("TOFY_CACHE_PREFETCH_BATCHES".to_string(), "8".to_string());
    }
    env.insert("TOFY_RUN_GROUP".to_string(), args.run_group.clone());
    env.insert("TOFY_RUN_STAGE_NAME".to_string(), stage_name.to_string());
    env
}

fn probe_high_world_env(args: &OomProbeArgs, stage_name: &str) -> HashMap<String, String> {
    let mut env = probe_base_env(args, stage_name);
    env.insert(
        "TOFY_HIGH_WORLD_WARMUP_BATCH".to_string(),
        args.high_world_warmup_batch.to_string(),
    );
    env.insert(
        "TOFY_HIGH_WORLD_WARMUP_GRAD_ACCUM".to_string(),
        args.high_world_warmup_accum.to_string(),
    );
    match args.high_world_warmup_steps {
        Some(steps) => {
            env.insert(
                "TOFY_HIGH_WORLD_WARMUP_STEPS".to_string(),
                steps.to_string(),
            );
        }
        None => {
            env.remove("TOFY_HIGH_WORLD_WARMUP_STEPS");
        }
    }
    env
}

fn probe_decoder_env(args: &OomProbeArgs, stage_name: &str) -> HashMap<String, String> {
    let mut env = probe_base_env(args, stage_name);
    env.insert(
        "TOFY_DECODER_WARMUP_BATCH".to_string(),
        args.decoder_warmup_batch.to_string(),
    );
    env.insert(
        "TOFY_DECODER_WARMUP_GRAD_ACCUM".to_string(),
        args.decoder_warmup_accum.to_string(),
    );
    match args.decoder_warmup_steps {
        Some(steps) => {
            env.insert("TOFY_DECODER_WARMUP_STEPS".to_string(), steps.to_string());
        }
        None => {
            env.remove("TOFY_DECODER_WARMUP_STEPS");
        }
    }
    env
}

fn cached_encoder_vocab_for_probe(args: &OomProbeArgs) -> Option<PathBuf> {
    if let Some(path) = &args.encoder_vocab {
        if path.exists() {
            return Some(path.clone());
        }
    }
    [
        model_dir().join(format!("vocabs/vocab_encoder_{}_default.txt", args.vocab)),
        model_dir().join("vocabs/vocab_encoder.txt"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn cached_decoder_vocab_for_probe(args: &OomProbeArgs) -> Option<PathBuf> {
    [model_dir().join(format!(
        "vocabs/vocab_code_{}_codeaware.txt",
        args.decoder_max_vocab
    ))]
    .into_iter()
    .find(|path| path.exists())
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
        "--freeze-encoder".to_string(),
    ]
}

fn default_world_encoder_path_for_probe(model_path: &Path) -> PathBuf {
    let raw = model_path.to_string_lossy();
    if let Some(prefix) = raw.strip_suffix(".safetensors") {
        PathBuf::from(format!("{prefix}.encoder.safetensors"))
    } else {
        PathBuf::from(format!("{raw}.encoder.safetensors"))
    }
}

fn high_world_probe_cmd(
    args: &OomProbeArgs,
    data_path: &Path,
    encoder: &Path,
    vocab: &Path,
    world: &Path,
    steps: usize,
    output_path: &Path,
) -> Vec<String> {
    vec![
        args.binary.to_string_lossy().to_string(),
        "--train-high-world".to_string(),
        encoder.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        world.to_string_lossy().to_string(),
        data_path.to_string_lossy().to_string(),
        steps.to_string(),
        args.high_world_batch.to_string(),
        args.dim.to_string(),
        args.max_seq.to_string(),
        args.layers.to_string(),
        args.heads.to_string(),
        args.bridge_dim.to_string(),
        args.planner_slots.to_string(),
        "--grad-accum".to_string(),
        args.high_world_accum.to_string(),
        "--output".to_string(),
        output_path.to_string_lossy().to_string(),
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
    let mut cmd = vec![
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
        "--decoder-dim".to_string(),
        args.decoder_dim.to_string(),
        "--decoder-layers".to_string(),
        args.decoder_layers.to_string(),
        "--decoder-heads".to_string(),
        args.decoder_heads.to_string(),
        "--decoder-ff-dim".to_string(),
        args.decoder_ff_dim.to_string(),
        "--decoder-output".to_string(),
        output_path.to_string_lossy().to_string(),
        "--grad-accum".to_string(),
        args.decoder_accum.to_string(),
    ];
    if let Some(path) = cached_decoder_vocab_for_probe(args) {
        cmd.push("--decoder-vocab".to_string());
        cmd.push(path.to_string_lossy().to_string());
    }
    cmd
}

fn is_base_model_artifact(name: &str, prefix: &str) -> bool {
    let Some(param_part) = name
        .strip_prefix(prefix)
        .and_then(|name| name.strip_suffix(".safetensors"))
    else {
        return false;
    };
    let Some((suffix, number)) = param_part
        .chars()
        .last()
        .map(|suffix| (suffix, &param_part[..param_part.len() - suffix.len_utf8()]))
    else {
        return false;
    };
    matches!(suffix, 'k' | 'M' | 'B')
        && !number.is_empty()
        && number.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
        && number.chars().any(|ch| ch.is_ascii_digit())
}

fn latest_base_model_artifact_with_prefix(prefix: &str) -> Result<PathBuf> {
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
        if !is_base_model_artifact(name, prefix) {
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
        .with_context(|| format!("no base model artifact found for prefix {prefix}"))
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
        let vocab = cached_encoder_vocab_for_probe(args)
            .unwrap_or_else(|| model_dir().join("vocabs/vocab_encoder.txt"));
        return Ok((latent.clone(), vocab));
    }
    let cmd = latent_probe_cmd(args, data_path, args.setup_latent_steps);
    run_checked_command(&cmd, &probe_base_env(args, "setup_latent"))?;
    let vocab = cached_encoder_vocab_for_probe(args)
        .unwrap_or_else(|| model_dir().join("vocabs/vocab_encoder.txt"));
    Ok((
        latest_base_model_artifact_with_prefix("model_latent_")?,
        vocab,
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
    latest_base_model_artifact_with_prefix("model_world_")
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
    write_json_atomic(
        &path,
        &ProbeSummary {
            results: results.to_vec(),
            full_run_estimates: full_run_vram_estimates(results),
        },
    )?;
    Ok(path)
}

fn historical_vram_multiplier(stage: &str) -> (f64, &'static str) {
    match stage {
        "latent" => (
            1.19,
            "latest/prior latent runs grew about 17-19% from early samples to the 10k/plateau sample",
        ),
        "world" => (
            3.65,
            "world has only 1000-step early historical samples; latest run grew from 2093MB at step 1000 to 7629MB plateau",
        ),
        "high_world" => (
            1.05,
            "latest high-world grew about 4.6% from step 100 to 10k and 2.8% from step 500 to plateau",
        ),
        "decoder" => (
            1.02,
            "decoder runs historically grew 0-1.7% from step 500 to 10k/plateau",
        ),
        _ => (1.0, "no historical multiplier available for this stage"),
    }
}

fn full_run_vram_estimates(results: &[ProbeResult]) -> Vec<FullRunVramEstimate> {
    results
        .iter()
        .map(|result| {
            let (multiplier, rationale) = historical_vram_multiplier(&result.name);
            let estimated_full_peak_mb = result
                .peak_used_mb
                .map(|peak| ((peak as f64) * multiplier).ceil() as i64);
            FullRunVramEstimate {
                stage: result.name.clone(),
                measured_peak_mb: result.peak_used_mb,
                historical_multiplier: multiplier,
                estimated_full_peak_mb,
                rationale: rationale.to_string(),
            }
        })
        .collect()
}

fn print_full_run_vram_estimates(results: &[ProbeResult]) -> Result<()> {
    let estimates = full_run_vram_estimates(results);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "full_run_vram_estimates": estimates,
        }))?
    );
    Ok(())
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
            ProbeStage::HighWorld => "high-world",
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
            latent = Some(latest_base_model_artifact_with_prefix("model_latent_")?);
            vocab = Some(
                cached_encoder_vocab_for_probe(&args)
                    .unwrap_or_else(|| model_dir().join("vocabs/vocab_encoder.txt")),
            );
        } else if matches!(
            args.stage,
            ProbeStage::World | ProbeStage::HighWorld | ProbeStage::Decoder
        ) {
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
            world = Some(latest_base_model_artifact_with_prefix("model_world_")?);
        } else if matches!(args.stage, ProbeStage::HighWorld | ProbeStage::Decoder) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            world = Some(ensure_setup_world(
                &args, &data_path, latent_ref, vocab_ref,
            )?);
        }

        if matches!(args.stage, ProbeStage::All | ProbeStage::HighWorld) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            let world_ref = world.as_ref().context("world setup missing")?;
            let encoder_ref = default_world_encoder_path_for_probe(world_ref);
            let encoder_ref = if encoder_ref.exists() {
                encoder_ref
            } else {
                latent_ref.clone()
            };
            let output_path = probe_dir.join("high_world_oom_probe.safetensors");
            let cmd = high_world_probe_cmd(
                &args,
                &data_path,
                &encoder_ref,
                vocab_ref,
                world_ref,
                args.high_world_steps,
                &output_path,
            );
            let result = run_measured_probe(
                "high_world",
                &cmd,
                &probe_high_world_env(&args, "high_world"),
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

        if matches!(args.stage, ProbeStage::All | ProbeStage::Decoder) {
            let latent_ref = latent.as_ref().context("latent setup missing")?;
            let vocab_ref = vocab.as_ref().context("vocab setup missing")?;
            let world_ref = world
                .or_else(|| latest_base_model_artifact_with_prefix("model_world_").ok())
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
                &probe_decoder_env(&args, "decoder"),
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

        print_full_run_vram_estimates(&results)?;
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

fn run_max_vram_probe(args: &[String]) -> Result<()> {
    let mut probe_args = Vec::with_capacity(args.len() + 1);
    probe_args.push("--max-vram".to_string());
    probe_args.extend(args.iter().cloned());
    run_sustained_oom_probe(&probe_args)
}

#[cfg(test)]
mod tests {
    use super::{
        artifact_cache_hit, code_poc_fim_row, code_poc_row_with_eos, escape_pair_field,
        flatten_for_encoder, is_base_model_artifact, remote_source_fingerprint,
        run_prepare_code_poc_mix, run_prepare_world_mix, split_pair_line_first_two,
        unescape_pair_field, world_mix_code_action, write_artifact_manifest,
    };
    use crate::data::CODE_EOS_TOKEN;
    use serde_json::json;
    use std::path::PathBuf;

    fn unique_prepare_test_dir(label: &str) -> PathBuf {
        let unique = format!(
            "tofy_prepare_{label}_{}_{}",
            std::process::id(),
            super::SystemTime::now()
                .duration_since(super::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn base_model_artifact_accepts_param_count_names() {
        assert!(is_base_model_artifact(
            "model_latent_94.12M.safetensors",
            "model_latent_"
        ));
        assert!(is_base_model_artifact(
            "model_world_977.0k.safetensors",
            "model_world_"
        ));
        assert!(is_base_model_artifact(
            "model_world_1.25B.safetensors",
            "model_world_"
        ));
    }

    #[test]
    fn base_model_artifact_rejects_sidecars_and_wrong_prefixes() {
        assert!(!is_base_model_artifact(
            "model_world_19.56M.encoder.safetensors",
            "model_world_"
        ));
        assert!(!is_base_model_artifact(
            "model_world_19.56M.high_world.safetensors",
            "model_world_"
        ));
        assert!(!is_base_model_artifact(
            "model_world_19.56M.train.safetensors",
            "model_world_"
        ));
        assert!(!is_base_model_artifact(
            "model_latent_94.12M.safetensors",
            "model_world_"
        ));
    }

    #[test]
    fn remote_hub_source_fingerprint_is_stable_without_filesystem_stat() {
        let path = PathBuf::from("hub:ronantakizawa/github-top-code:train");
        let first = remote_source_fingerprint(&path);
        let second = remote_source_fingerprint(&path);

        assert_eq!(first, second);
        assert_eq!(first.path, "hub:ronantakizawa/github-top-code:train");
        assert_eq!(first.len, 0);
        assert_eq!(first.modified_unix_nanos, 0);
        assert!(!first.content_hash.is_empty());
    }

    #[test]
    fn artifact_cache_hit_accepts_remote_hub_input() {
        let dir = unique_prepare_test_dir("manifest");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("github_top_code.txt");
        std::fs::write(&output, "package main\nfunc main() {}\n").unwrap();

        let inputs = vec![PathBuf::from("hub:ronantakizawa/github-top-code:train")];
        let params = json!({"split": "train"});
        write_artifact_manifest("github_top_code", &output, &inputs, params.clone(), 1).unwrap();

        let hit = artifact_cache_hit("github_top_code", &output, &inputs, &params).unwrap();
        assert!(hit.is_some());

        let changed_inputs = vec![PathBuf::from("hub:ronantakizawa/github-top-code:test")];
        let miss =
            artifact_cache_hit("github_top_code", &output, &changed_inputs, &params).unwrap();
        assert!(miss.is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn artifact_cache_hit_trusts_handoff_when_prepared_cache_required() {
        let dir = unique_prepare_test_dir("handoff");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("go_repair_pairs.txt");
        std::fs::write(&output, "prompt\tcode\n").unwrap();

        let input = dir.join("go_instruction_pairs.txt");
        std::fs::write(&input, "task\tanswer\n").unwrap();
        let inputs = vec![input];
        let params = json!({
            "go_bin": "go",
            "go_version": "go version go1.26.3 linux/amd64",
            "workers": 14
        });
        write_artifact_manifest("go_repair_pairs", &output, &inputs, params, 1).unwrap();

        let previous = std::env::var("TOFY_REQUIRE_PREPARED_CACHE").ok();
        std::env::set_var("TOFY_REQUIRE_PREPARED_CACHE", "1");
        let changed_inputs = vec![dir.join("other_input.txt")];
        let changed_params = json!({
            "go_bin": "go",
            "go_version": "go version go1.22.0 linux/amd64",
            "workers": 4
        });
        let hit = artifact_cache_hit("go_repair_pairs", &output, &changed_inputs, &changed_params)
            .unwrap();
        match previous {
            Some(value) => std::env::set_var("TOFY_REQUIRE_PREPARED_CACHE", value),
            None => std::env::remove_var("TOFY_REQUIRE_PREPARED_CACHE"),
        }

        assert!(hit.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn world_mix_streaming_writes_exact_done_rows() {
        let dir = unique_prepare_test_dir("world_mix");
        std::fs::create_dir_all(&dir).unwrap();
        let text = dir.join("text.txt");
        let code = dir.join("code.txt");
        let output = dir.join("world.txt");
        std::fs::write(
            &text,
            format!(
                "{}\t{}\n{}\t{}\n",
                escape_pair_field("hello"),
                escape_pair_field("world"),
                escape_pair_field("question"),
                escape_pair_field("answer")
            ),
        )
        .unwrap();
        std::fs::write(
            &code,
            format!(
                "{}\t{}\n{}\t{}\n",
                escape_pair_field("write add"),
                escape_pair_field("func Add() {}"),
                escape_pair_field("write sub"),
                escape_pair_field("func Sub() {}")
            ),
        )
        .unwrap();

        run_prepare_world_mix(&[
            "--output".to_string(),
            output.to_string_lossy().to_string(),
            "--text-pairs".to_string(),
            text.to_string_lossy().to_string(),
            "--code-pairs".to_string(),
            code.to_string_lossy().to_string(),
            "--code-ratio".to_string(),
            "0.5".to_string(),
            "--done-ratio".to_string(),
            "0.25".to_string(),
            "--max-rows".to_string(),
            "4".to_string(),
            "--seed".to_string(),
            "7".to_string(),
        ])
        .unwrap();

        let rows = std::fs::read_to_string(&output).unwrap();
        let lines = rows.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 5);
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.ends_with("\t<done>\tdone"))
                .count(),
            1
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn code_poc_mix_streaming_respects_max_rows() {
        let dir = unique_prepare_test_dir("code_poc_mix");
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.txt");
        let instruction = dir.join("instruction.txt");
        let extra = dir.join("extra.txt");
        let output = dir.join("code_mix.txt");
        let pair = |prompt: &str, code: &str| {
            format!(
                "{}\t{}\n",
                escape_pair_field(prompt),
                escape_pair_field(code)
            )
        };
        std::fs::write(
            &base,
            pair("base", "func Base() {\n    println(\"base\")\n}"),
        )
        .unwrap();
        std::fs::write(
            &instruction,
            pair(
                "instruction",
                "func Instruction() {\n    println(\"inst\")\n}",
            ),
        )
        .unwrap();
        std::fs::write(
            &extra,
            pair("extra", "func Extra() {\n    println(\"extra\")\n}"),
        )
        .unwrap();

        run_prepare_code_poc_mix(&[
            "--output".to_string(),
            output.to_string_lossy().to_string(),
            "--base-pairs".to_string(),
            base.to_string_lossy().to_string(),
            "--instruction-pairs".to_string(),
            instruction.to_string_lossy().to_string(),
            "--extra-pairs".to_string(),
            extra.to_string_lossy().to_string(),
            "--base-repeat".to_string(),
            "1".to_string(),
            "--instruction-repeat".to_string(),
            "2".to_string(),
            "--extra-repeat".to_string(),
            "1".to_string(),
            "--fim-repeat".to_string(),
            "0".to_string(),
            "--max-rows".to_string(),
            "3".to_string(),
            "--seed".to_string(),
            "3".to_string(),
        ])
        .unwrap();

        let rows = std::fs::read_to_string(&output).unwrap();
        let lines = rows.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        for line in lines {
            let (_, right) = line.split_once('\t').expect("row should be a pair");
            assert!(unescape_pair_field(right).ends_with(CODE_EOS_TOKEN));
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn world_mix_code_action_honors_explicit_target_action() {
        assert_eq!(
            world_mix_code_action("prompt", "<action:text_reply> 42"),
            "text_reply"
        );
        assert_eq!(
            world_mix_code_action("prompt", "<action:fetch_docs> fmt.Println"),
            "fetch_docs"
        );
        assert_eq!(world_mix_code_action("prompt", "func add() {}"), "code");
    }

    #[test]
    fn pair_field_escape_keeps_jsonl_converter_records_single_line() {
        let escaped = escape_pair_field("line1\nline2\twith tab");
        assert_eq!(escaped, "line1\\nline2    with tab");
    }

    #[test]
    fn code_poc_rows_append_eos_to_completion_only() {
        let row = format!(
            "{}\t{}",
            escape_pair_field("write add"),
            escape_pair_field("func Add(a int, b int) int {\n    return a + b\n}")
        );
        let transformed = code_poc_row_with_eos(&row).expect("row should transform");
        let (left, right) = transformed
            .split_once('\t')
            .expect("pair should remain a pair");

        assert_eq!(unescape_pair_field(left), "write add");
        assert!(
            unescape_pair_field(right).ends_with(CODE_EOS_TOKEN),
            "completion should end with code EOS"
        );
    }

    #[test]
    fn code_poc_rows_drop_explicit_non_code_targets() {
        let row = format!(
            "{}\t{}",
            escape_pair_field("track snippet"),
            escape_pair_field("<action:text_reply> 42")
        );

        assert!(code_poc_row_with_eos(&row).is_none());
    }

    #[test]
    fn code_poc_rows_strip_explicit_code_action() {
        let row = format!(
            "{}\t{}",
            escape_pair_field("write add"),
            escape_pair_field("<action:code> func Add(a int, b int) int { return a + b }")
        );
        let transformed = code_poc_row_with_eos(&row).expect("code row should transform");
        let (_, right) = transformed
            .split_once('\t')
            .expect("pair should remain a pair");
        let target = unescape_pair_field(right);

        assert!(target.starts_with("func Add"));
        assert!(!target.contains("<action:code>"));
        assert!(target.ends_with(CODE_EOS_TOKEN));
    }

    #[test]
    fn code_poc_fim_row_uses_go_function_body_as_middle() {
        let code = "func Add(a int, b int) int {\n    total := a + b\n    return total\n}";
        let row = format!(
            "{}\t{}",
            escape_pair_field("write Add"),
            escape_pair_field(code)
        );
        let fim = code_poc_fim_row(&row).expect("Go function should produce FIM row");
        let (left, right) = fim.split_once('\t').expect("FIM row should remain a pair");
        let prompt = unescape_pair_field(left);
        let target = unescape_pair_field(right);

        assert!(prompt.contains("<fim_prefix>"));
        assert!(prompt.contains("<fim_suffix>"));
        assert!(prompt.contains("<fim_middle>"));
        assert!(prompt.contains("func Add(a int, b int) int {"));
        assert!(prompt.contains("}"));
        assert!(target.contains("return total"));
        assert!(target.ends_with(CODE_EOS_TOKEN));
    }

    #[test]
    fn encoder_corpus_escape_preserves_code_layout() {
        assert_eq!(
            flatten_for_encoder("func main() {\\n    fmt.Println(\"hi\")\\n}"),
            "func main() {\\n    fmt.Println(\"hi\")\\n}"
        );
    }

    #[test]
    fn world_mix_pair_parser_preserves_explicit_action_label() {
        let (_left, right) =
            split_pair_line_first_two("prompt\tanswer\ttext_reply").expect("pair should parse");
        assert_eq!(right, "<action:text_reply> answer");
    }
}
