use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::UNIX_EPOCH;

use crate::data::{
    build_vocab_from_pair_file, build_vocab_from_raw_world_file_with_mode,
    encode_line_with_vocab_mode, encode_raw_world_line_with_vocab_mode, tokenizer_spec,
    tokenizer_spec_signature, TokenizationMode, TokenizerSpec, DEFAULT_MIN_TOKENS_PER_LINE,
};
use crate::model::vocab::vocab_signature;
use crate::model::{load_vocab_from_file, save_vocab_to_file, Vocab};

const CACHE_VERSION: u32 = 3;
const TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_TOKEN_CACHE_V2\n";
const DUAL_TOKEN_CACHE_MAGIC: &[u8] = b"TOFY_DUAL_TOKEN_CACHE_V2\n";
const NO_ACTION: u32 = u32::MAX;
const PROGRESS_EVERY_LINES: usize = 500_000;
const DEFAULT_TOKEN_CACHE_ENCODE_CHUNK_LINES: usize = 16_384;
const DEFAULT_TOKEN_CACHE_RAW_CHARS_PER_TOKEN: usize = 24;
const DEFAULT_TOKEN_CACHE_RAW_CHAR_CAP: usize = 64 * 1024;

struct DualWorldCacheRow {
    encoder_state_tokens: Vec<u32>,
    encoder_next_tokens: Vec<u32>,
    decoder_state_tokens: Vec<u32>,
    decoder_next_tokens: Vec<u32>,
    action_label: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SourceFingerprint {
    path: String,
    len: u64,
    modified_unix_secs: u64,
    content_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VocabManifest {
    version: u32,
    kind: String,
    source: SourceFingerprint,
    tokenizer: String,
    tokenizer_spec: TokenizerSpec,
    tokenizer_spec_signature: String,
    max_vocab: usize,
    vocab_path: String,
    vocab_signature: String,
    rows: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenManifest {
    version: u32,
    kind: String,
    source: SourceFingerprint,
    tokenizer: String,
    tokenizer_spec: TokenizerSpec,
    tokenizer_spec_signature: String,
    max_seq: usize,
    vocab_path: String,
    vocab_signature: String,
    token_cache_path: String,
    rows: usize,
}

#[derive(Debug)]
struct PrepareCacheConfig {
    encoder_data: PathBuf,
    world_data: PathBuf,
    code_data: PathBuf,
    encoder_vocab_path: PathBuf,
    code_vocab_path: PathBuf,
    cache_dir: PathBuf,
    encoder_max_vocab: usize,
    code_max_vocab: usize,
    encoder_max_seq: usize,
    world_max_seq: usize,
    code_max_seq: usize,
    force: bool,
}

struct VocabCacheSpec<'a> {
    kind: &'a str,
    data_path: &'a PathBuf,
    source: &'a SourceFingerprint,
    mode: TokenizationMode,
    max_vocab: usize,
    vocab_path: &'a Path,
    manifest_path: &'a Path,
    force: bool,
}

struct TokenCacheRawCaps {
    world_side: usize,
}

struct TokenCacheSpec<'a> {
    kind: &'a str,
    data_path: &'a PathBuf,
    source: &'a SourceFingerprint,
    mode: TokenizationMode,
    max_seq: usize,
    token_cache_path: &'a Path,
    manifest_path: &'a Path,
    force: bool,
}

pub fn try_run_prepare_pipeline_cache(args: &[String]) -> Result<bool> {
    if args.len() < 2
        || (args[1] != "--prepare-pipeline-cache" && args[1] != "prepare-pipeline-cache")
    {
        return Ok(false);
    }
    let config = PrepareCacheConfig::from_args(&args[2..])?;
    prepare_pipeline_cache(&config)?;
    Ok(true)
}

impl PrepareCacheConfig {
    fn from_args(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                "usage: --prepare-pipeline-cache <encoder_pairs> <world_pairs> <code_pairs> [encoder_vocab_out] [code_vocab_out] [cache_dir] [--encoder-max-vocab N] [--code-max-vocab N] [--encoder-max-seq N] [--world-max-seq N] [--code-max-seq N] [--force]"
            );
        }
        let mut encoder_max_vocab = 8_000usize;
        let mut code_max_vocab = 16_000usize;
        let mut encoder_max_seq = 256usize;
        let mut world_max_seq = 256usize;
        let mut code_max_seq = 192usize;
        let mut force = false;
        let mut positional = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--encoder-max-vocab" => {
                    encoder_max_vocab = parse_next_usize(args, i, "--encoder-max-vocab")?;
                    i += 2;
                }
                "--code-max-vocab" => {
                    code_max_vocab = parse_next_usize(args, i, "--code-max-vocab")?;
                    i += 2;
                }
                "--encoder-max-seq" => {
                    encoder_max_seq = parse_next_usize(args, i, "--encoder-max-seq")?;
                    i += 2;
                }
                "--world-max-seq" => {
                    world_max_seq = parse_next_usize(args, i, "--world-max-seq")?;
                    i += 2;
                }
                "--code-max-seq" => {
                    code_max_seq = parse_next_usize(args, i, "--code-max-seq")?;
                    i += 2;
                }
                "--force" => {
                    force = true;
                    i += 1;
                }
                value if value.starts_with("--") => bail!("unknown cache flag: {value}"),
                value => {
                    positional.push(value.to_string());
                    i += 1;
                }
            }
        }
        let encoder_data = PathBuf::from(&positional[0]);
        let world_data = PathBuf::from(&positional[1]);
        let code_data = PathBuf::from(&positional[2]);
        let encoder_vocab_path = positional
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("local_models/vocabs/vocab_encoder_8000_default.txt"));
        let code_vocab_path = positional
            .get(4)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("local_models/vocabs/vocab_code_16000_codeaware.txt"));
        let cache_dir = positional
            .get(5)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/cache"));
        Ok(Self {
            encoder_data,
            world_data,
            code_data,
            encoder_vocab_path,
            code_vocab_path,
            cache_dir,
            encoder_max_vocab,
            code_max_vocab,
            encoder_max_seq,
            world_max_seq,
            code_max_seq,
            force,
        })
    }
}

fn parse_next_usize(args: &[String], index: usize, flag: &str) -> Result<usize> {
    args.get(index + 1)
        .ok_or_else(|| anyhow::anyhow!("{flag} requires an integer"))?
        .parse()
        .with_context(|| format!("{flag} must be an integer"))
}

fn join_result<T>(handle: thread::ScopedJoinHandle<'_, Result<T>>, label: &str) -> Result<T> {
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("{label} panicked"))?
}

fn prepare_pipeline_cache(config: &PrepareCacheConfig) -> Result<()> {
    fs::create_dir_all(&config.cache_dir)?;
    if let Some(parent) = config.encoder_vocab_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = config.code_vocab_path.parent() {
        fs::create_dir_all(parent)?;
    }
    println!("Preparing pipeline vocab/token cache");
    println!("cache dir: {}", config.cache_dir.display());

    let (encoder_source, world_source, code_source) = thread::scope(|scope| -> Result<_> {
        let encoder = scope.spawn(|| source_fingerprint(&config.encoder_data));
        let world = scope.spawn(|| source_fingerprint(&config.world_data));
        let code = scope.spawn(|| source_fingerprint(&config.code_data));
        Ok((
            join_result(encoder, "encoder source fingerprint")?,
            join_result(world, "world source fingerprint")?,
            join_result(code, "code source fingerprint")?,
        ))
    })?;

    let encoder_vocab_manifest = config.cache_dir.join("encoder_vocab.manifest.json");
    let code_vocab_manifest = config.cache_dir.join("code_decoder_vocab.manifest.json");
    let (encoder_vocab, code_vocab) = thread::scope(|scope| -> Result<_> {
        let encoder = scope.spawn(|| {
            ensure_vocab_cache(VocabCacheSpec {
                kind: "encoder",
                data_path: &config.encoder_data,
                source: &encoder_source,
                mode: TokenizationMode::Default,
                max_vocab: config.encoder_max_vocab,
                vocab_path: &config.encoder_vocab_path,
                manifest_path: &encoder_vocab_manifest,
                force: config.force,
            })
        });
        let code = scope.spawn(|| {
            ensure_vocab_cache(VocabCacheSpec {
                kind: "code_decoder",
                data_path: &config.code_data,
                source: &code_source,
                mode: TokenizationMode::CodeAware,
                max_vocab: config.code_max_vocab,
                vocab_path: &config.code_vocab_path,
                manifest_path: &code_vocab_manifest,
                force: config.force,
            })
        });
        Ok((
            join_result(encoder, "encoder vocab cache")?,
            join_result(code, "code decoder vocab cache")?,
        ))
    })?;

    let encoder_tokens = config.cache_dir.join("encoder.tokens.bin");
    let encoder_tokens_manifest = config.cache_dir.join("encoder_tokens.manifest.json");
    let world_tokens = config.cache_dir.join("world.tokens.bin");
    let world_tokens_manifest = config.cache_dir.join("world_tokens.manifest.json");
    let code_tokens = config.cache_dir.join("code_decoder.tokens.bin");
    let code_tokens_manifest = config.cache_dir.join("code_decoder_tokens.manifest.json");
    let dual_tokens = config.cache_dir.join("code_decoder_dual.tokens.bin");
    let dual_tokens_manifest = config
        .cache_dir
        .join("code_decoder_dual_tokens.manifest.json");
    thread::scope(|scope| -> Result<()> {
        let encoder = scope.spawn(|| {
            ensure_sequence_token_cache(
                TokenCacheSpec {
                    kind: "encoder",
                    data_path: &config.encoder_data,
                    source: &encoder_source,
                    mode: TokenizationMode::Default,
                    max_seq: config.encoder_max_seq,
                    token_cache_path: &encoder_tokens,
                    manifest_path: &encoder_tokens_manifest,
                    force: config.force,
                },
                &encoder_vocab,
            )
        });
        let world = scope.spawn(|| {
            ensure_world_token_cache(
                TokenCacheSpec {
                    kind: "world",
                    data_path: &config.world_data,
                    source: &world_source,
                    mode: TokenizationMode::Default,
                    max_seq: config.world_max_seq,
                    token_cache_path: &world_tokens,
                    manifest_path: &world_tokens_manifest,
                    force: config.force,
                },
                &encoder_vocab,
            )
        });
        let code = scope.spawn(|| {
            ensure_world_token_cache(
                TokenCacheSpec {
                    kind: "code_decoder",
                    data_path: &config.code_data,
                    source: &code_source,
                    mode: TokenizationMode::CodeAware,
                    max_seq: config.code_max_seq,
                    token_cache_path: &code_tokens,
                    manifest_path: &code_tokens_manifest,
                    force: config.force,
                },
                &code_vocab,
            )
        });
        let dual = scope.spawn(|| {
            ensure_dual_world_token_cache(
                TokenCacheSpec {
                    kind: "code_decoder_dual",
                    data_path: &config.code_data,
                    source: &code_source,
                    mode: TokenizationMode::CodeAware,
                    max_seq: config.code_max_seq,
                    token_cache_path: &dual_tokens,
                    manifest_path: &dual_tokens_manifest,
                    force: config.force,
                },
                &encoder_vocab,
                &code_vocab,
            )
        });
        join_result(encoder, "encoder token cache")?;
        join_result(world, "world token cache")?;
        join_result(code, "code decoder token cache")?;
        join_result(dual, "dual code decoder token cache")?;
        Ok(())
    })?;
    println!("Pipeline cache ready.");
    Ok(())
}

fn ensure_vocab_cache(spec: VocabCacheSpec<'_>) -> Result<Vocab> {
    let tokenizer_spec = tokenizer_spec(spec.mode);
    let tokenizer_spec_sig = tokenizer_spec_signature(spec.mode);
    if !spec.force && spec.vocab_path.exists() && spec.manifest_path.exists() {
        if let Ok(manifest) = load_json::<VocabManifest>(spec.manifest_path) {
            if manifest.version == CACHE_VERSION
                && manifest.kind == spec.kind
                && source_matches(&manifest.source, spec.source)
                && manifest.tokenizer == spec.mode.as_str()
                && manifest.tokenizer_spec_signature == tokenizer_spec_sig
                && manifest.max_vocab == spec.max_vocab
                && manifest.vocab_path == path_string(spec.vocab_path)
            {
                let vocab = load_vocab_from_file(spec.vocab_path)?;
                if manifest.vocab_signature == vocab_signature(&vocab) {
                    println!(
                        "{} vocab cache hit: {} (rows={}, signature={})",
                        spec.kind,
                        spec.vocab_path.display(),
                        manifest.rows,
                        manifest.vocab_signature
                    );
                    return Ok(vocab);
                }
            }
        }
    }

    println!(
        "{} vocab cache miss: building {} vocab from {}",
        spec.kind,
        spec.mode.as_str(),
        spec.data_path.display()
    );
    let (vocab, stats, rows) = if spec.mode == TokenizationMode::Default {
        build_vocab_from_pair_file(
            spec.data_path,
            spec.max_vocab,
            Some(DEFAULT_MIN_TOKENS_PER_LINE),
        )?
    } else {
        build_vocab_from_raw_world_file_with_mode(spec.data_path, spec.max_vocab, spec.mode)?
    };
    save_vocab_to_file(&vocab, spec.vocab_path)?;
    let signature = vocab_signature(&vocab);
    write_json_atomic(
        spec.manifest_path,
        &VocabManifest {
            version: CACHE_VERSION,
            kind: spec.kind.to_string(),
            source: spec.source.clone(),
            tokenizer: spec.mode.as_str().to_string(),
            tokenizer_spec,
            tokenizer_spec_signature: tokenizer_spec_sig,
            max_vocab: spec.max_vocab,
            vocab_path: path_string(spec.vocab_path),
            vocab_signature: signature.clone(),
            rows,
        },
    )?;
    println!(
        "{} vocab saved: {} (rows={}, vocab={}, OOV={})",
        spec.kind,
        spec.vocab_path.display(),
        rows,
        stats.vocab_size,
        stats.oov_tokens
    );
    Ok(vocab)
}

fn ensure_sequence_token_cache(spec: TokenCacheSpec<'_>, vocab: &Vocab) -> Result<()> {
    let vocab_sig = vocab_signature(vocab);
    if token_cache_is_valid(&spec, &vocab_sig) {
        println!(
            "{} token cache hit: {}",
            spec.kind,
            spec.token_cache_path.display()
        );
        return Ok(());
    }

    println!(
        "{} token cache miss: encoding {} to {}",
        spec.kind,
        spec.data_path.display(),
        spec.token_cache_path.display()
    );
    let tmp_path = tmp_path_for(spec.token_cache_path);
    let mut writer = BufWriter::new(File::create(&tmp_path)?);
    writer.write_all(TOKEN_CACHE_MAGIC)?;
    let mut lines = BufReader::new(File::open(spec.data_path)?).lines();
    let chunk_lines = token_cache_encode_chunk_lines();
    let raw_cap = token_cache_raw_sequence_cap(spec.max_seq);
    println!(
        "{} token cache parallel encode: chunk_lines={} rayon_threads={} raw_char_cap={}",
        spec.kind,
        chunk_lines,
        rayon::current_num_threads(),
        raw_cap
    );
    let mut rows = 0usize;
    let mut raw_lines = 0usize;
    let mut next_progress = PROGRESS_EVERY_LINES;
    loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        raw_lines += chunk.len();
        let encoded = chunk
            .par_iter()
            .filter_map(|line| {
                let line = cap_str_chars(line, raw_cap);
                let mut ids = encode_line_with_vocab_mode(
                    line,
                    vocab,
                    spec.mode,
                    DEFAULT_MIN_TOKENS_PER_LINE,
                )?;
                truncate_ids(&mut ids, spec.max_seq);
                Some(ids)
            })
            .collect::<Vec<_>>();
        for ids in &encoded {
            write_token_record(&mut writer, ids, &[], NO_ACTION)?;
        }
        rows += encoded.len();
        while raw_lines >= next_progress {
            println!(
                "{} token cache progress: {raw_lines} raw lines, {rows} rows",
                spec.kind
            );
            next_progress += PROGRESS_EVERY_LINES;
        }
    }
    writer.flush()?;
    fs::rename(&tmp_path, spec.token_cache_path)?;
    write_json_atomic(
        spec.manifest_path,
        &TokenManifest {
            version: CACHE_VERSION,
            kind: spec.kind.to_string(),
            source: spec.source.clone(),
            tokenizer: spec.mode.as_str().to_string(),
            tokenizer_spec: tokenizer_spec(spec.mode),
            tokenizer_spec_signature: tokenizer_spec_signature(spec.mode),
            max_seq: spec.max_seq,
            vocab_path: String::new(),
            vocab_signature: vocab_sig,
            token_cache_path: path_string(spec.token_cache_path),
            rows,
        },
    )?;
    println!("{} token cache saved: {} rows", spec.kind, rows);
    Ok(())
}

fn ensure_world_token_cache(spec: TokenCacheSpec<'_>, vocab: &Vocab) -> Result<()> {
    let vocab_sig = vocab_signature(vocab);
    if token_cache_is_valid(&spec, &vocab_sig) {
        println!(
            "{} token cache hit: {}",
            spec.kind,
            spec.token_cache_path.display()
        );
        return Ok(());
    }

    println!(
        "{} token cache miss: encoding {} to {}",
        spec.kind,
        spec.data_path.display(),
        spec.token_cache_path.display()
    );
    let tmp_path = tmp_path_for(spec.token_cache_path);
    let mut writer = BufWriter::new(File::create(&tmp_path)?);
    writer.write_all(TOKEN_CACHE_MAGIC)?;
    let mut lines = BufReader::new(File::open(spec.data_path)?).lines();
    let chunk_lines = token_cache_encode_chunk_lines();
    let raw_caps = token_cache_raw_world_caps(spec.max_seq);
    println!(
        "{} token cache parallel encode: chunk_lines={} rayon_threads={} raw_side_char_cap={}",
        spec.kind,
        chunk_lines,
        rayon::current_num_threads(),
        raw_caps.world_side
    );
    let mut rows = 0usize;
    let mut raw_lines = 0usize;
    let mut next_progress = PROGRESS_EVERY_LINES;
    loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        raw_lines += chunk.len();
        let encoded = chunk
            .par_iter()
            .filter_map(|line| {
                let capped = cap_raw_world_line(line, raw_caps.world_side);
                let mut example = encode_raw_world_line_with_vocab_mode(&capped, vocab, spec.mode)?;
                truncate_ids(&mut example.state_tokens, spec.max_seq);
                truncate_ids(&mut example.next_tokens, spec.max_seq);
                Some(example)
            })
            .collect::<Vec<_>>();
        for example in &encoded {
            write_token_record(
                &mut writer,
                &example.state_tokens,
                &example.next_tokens,
                example.action_label,
            )?;
        }
        rows += encoded.len();
        while raw_lines >= next_progress {
            println!(
                "{} token cache progress: {raw_lines} raw lines, {rows} rows",
                spec.kind
            );
            next_progress += PROGRESS_EVERY_LINES;
        }
    }
    writer.flush()?;
    fs::rename(&tmp_path, spec.token_cache_path)?;
    write_json_atomic(
        spec.manifest_path,
        &TokenManifest {
            version: CACHE_VERSION,
            kind: spec.kind.to_string(),
            source: spec.source.clone(),
            tokenizer: spec.mode.as_str().to_string(),
            tokenizer_spec: tokenizer_spec(spec.mode),
            tokenizer_spec_signature: tokenizer_spec_signature(spec.mode),
            max_seq: spec.max_seq,
            vocab_path: String::new(),
            vocab_signature: vocab_sig,
            token_cache_path: path_string(spec.token_cache_path),
            rows,
        },
    )?;
    println!("{} token cache saved: {} rows", spec.kind, rows);
    Ok(())
}

fn ensure_dual_world_token_cache(
    spec: TokenCacheSpec<'_>,
    encoder_vocab: &Vocab,
    decoder_vocab: &Vocab,
) -> Result<()> {
    let vocab_sig = format!(
        "{}+{}",
        vocab_signature(encoder_vocab),
        vocab_signature(decoder_vocab)
    );
    if token_cache_is_valid(&spec, &vocab_sig) {
        println!(
            "{} token cache hit: {}",
            spec.kind,
            spec.token_cache_path.display()
        );
        return Ok(());
    }

    println!(
        "{} token cache miss: encoding {} to {}",
        spec.kind,
        spec.data_path.display(),
        spec.token_cache_path.display()
    );
    let tmp_path = tmp_path_for(spec.token_cache_path);
    let mut writer = BufWriter::new(File::create(&tmp_path)?);
    writer.write_all(DUAL_TOKEN_CACHE_MAGIC)?;
    let mut lines = BufReader::new(File::open(spec.data_path)?).lines();
    let chunk_lines = token_cache_encode_chunk_lines();
    let raw_caps = token_cache_raw_world_caps(spec.max_seq);
    println!(
        "{} token cache parallel encode: chunk_lines={} rayon_threads={} raw_side_char_cap={}",
        spec.kind,
        chunk_lines,
        rayon::current_num_threads(),
        raw_caps.world_side
    );
    let mut rows = 0usize;
    let mut raw_lines = 0usize;
    let mut next_progress = PROGRESS_EVERY_LINES;
    loop {
        let chunk = read_line_chunk(&mut lines, chunk_lines)?;
        if chunk.is_empty() {
            break;
        }
        raw_lines += chunk.len();
        let encoded = chunk
            .par_iter()
            .filter_map(|line| {
                let capped = cap_raw_world_line(line, raw_caps.world_side);
                let mut encoder_example = encode_raw_world_line_with_vocab_mode(
                    &capped,
                    encoder_vocab,
                    TokenizationMode::Default,
                )?;
                let mut decoder_example =
                    encode_raw_world_line_with_vocab_mode(&capped, decoder_vocab, spec.mode)?;
                truncate_ids(&mut encoder_example.state_tokens, spec.max_seq);
                truncate_ids(&mut encoder_example.next_tokens, spec.max_seq);
                truncate_ids(&mut decoder_example.state_tokens, spec.max_seq);
                truncate_ids(&mut decoder_example.next_tokens, spec.max_seq);
                Some(DualWorldCacheRow {
                    encoder_state_tokens: encoder_example.state_tokens,
                    encoder_next_tokens: encoder_example.next_tokens,
                    decoder_state_tokens: decoder_example.state_tokens,
                    decoder_next_tokens: decoder_example.next_tokens,
                    action_label: decoder_example.action_label,
                })
            })
            .collect::<Vec<_>>();
        for row in &encoded {
            write_dual_token_record(
                &mut writer,
                &row.encoder_state_tokens,
                &row.encoder_next_tokens,
                &row.decoder_state_tokens,
                &row.decoder_next_tokens,
                row.action_label,
            )?;
        }
        rows += encoded.len();
        while raw_lines >= next_progress {
            println!(
                "{} token cache progress: {raw_lines} raw lines, {rows} rows",
                spec.kind
            );
            next_progress += PROGRESS_EVERY_LINES;
        }
    }
    writer.flush()?;
    fs::rename(&tmp_path, spec.token_cache_path)?;
    write_json_atomic(
        spec.manifest_path,
        &TokenManifest {
            version: CACHE_VERSION,
            kind: spec.kind.to_string(),
            source: spec.source.clone(),
            tokenizer: spec.mode.as_str().to_string(),
            tokenizer_spec: tokenizer_spec(spec.mode),
            tokenizer_spec_signature: tokenizer_spec_signature(spec.mode),
            max_seq: spec.max_seq,
            vocab_path: String::new(),
            vocab_signature: vocab_sig,
            token_cache_path: path_string(spec.token_cache_path),
            rows,
        },
    )?;
    println!("{} token cache saved: {} rows", spec.kind, rows);
    Ok(())
}

fn token_cache_is_valid(spec: &TokenCacheSpec<'_>, vocab_signature: &str) -> bool {
    if spec.force || !spec.token_cache_path.exists() || !spec.manifest_path.exists() {
        return false;
    }
    let Ok(manifest) = load_json::<TokenManifest>(spec.manifest_path) else {
        return false;
    };
    manifest.version == CACHE_VERSION
        && manifest.kind == spec.kind
        && source_matches(&manifest.source, spec.source)
        && manifest.tokenizer == spec.mode.as_str()
        && manifest.tokenizer_spec_signature == tokenizer_spec_signature(spec.mode)
        && manifest.max_seq >= spec.max_seq
        && manifest.vocab_signature == vocab_signature
        && manifest.token_cache_path == path_string(spec.token_cache_path)
}

fn token_cache_encode_chunk_lines() -> usize {
    std::env::var("TOFY_TOKEN_CACHE_ENCODE_CHUNK_LINES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_TOKEN_CACHE_ENCODE_CHUNK_LINES)
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

fn write_token_record<W: Write>(
    writer: &mut W,
    left_ids: &[u32],
    right_ids: &[u32],
    action: u32,
) -> Result<()> {
    write_ids(writer, left_ids)?;
    write_ids(writer, right_ids)?;
    writer.write_all(&action.to_le_bytes())?;
    Ok(())
}

fn write_dual_token_record<W: Write>(
    writer: &mut W,
    encoder_left_ids: &[u32],
    encoder_right_ids: &[u32],
    decoder_left_ids: &[u32],
    decoder_right_ids: &[u32],
    action: u32,
) -> Result<()> {
    write_ids(writer, encoder_left_ids)?;
    write_ids(writer, encoder_right_ids)?;
    write_ids(writer, decoder_left_ids)?;
    write_ids(writer, decoder_right_ids)?;
    writer.write_all(&action.to_le_bytes())?;
    Ok(())
}

fn write_ids<W: Write>(writer: &mut W, ids: &[u32]) -> Result<()> {
    writer.write_all(&(ids.len() as u32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(ids));
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    writer.write_all(&bytes)?;
    Ok(())
}

fn truncate_ids(ids: &mut Vec<u32>, max_seq: usize) {
    if max_seq > 0 && ids.len() > max_seq {
        ids.truncate(max_seq);
    }
}

fn token_cache_raw_sequence_cap(max_seq: usize) -> usize {
    token_cache_raw_cap_env()
        .unwrap_or_else(|| {
            max_seq
                .saturating_mul(DEFAULT_TOKEN_CACHE_RAW_CHARS_PER_TOKEN)
                .max(DEFAULT_TOKEN_CACHE_RAW_CHAR_CAP)
        })
        .max(1)
}

fn token_cache_raw_world_caps(max_seq: usize) -> TokenCacheRawCaps {
    let side = token_cache_raw_sequence_cap(max_seq);
    TokenCacheRawCaps { world_side: side }
}

fn token_cache_raw_cap_env() -> Option<usize> {
    std::env::var("TOFY_TOKEN_CACHE_RAW_CHAR_CAP")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
}

fn cap_str_chars(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        return text;
    }
    let mut end = 0usize;
    for (idx, _) in text.char_indices() {
        if idx > max_chars {
            break;
        }
        end = idx;
    }
    if end == 0 {
        ""
    } else {
        &text[..end]
    }
}

fn cap_raw_world_line(line: &str, max_side_chars: usize) -> String {
    let Some((left, rest)) = line.split_once('\t') else {
        return cap_str_chars(line, max_side_chars.saturating_mul(2)).to_string();
    };
    let Some((right, action)) = rest.split_once('\t') else {
        return format!(
            "{}\t{}",
            cap_str_chars(left, max_side_chars),
            cap_str_chars(rest, max_side_chars)
        );
    };
    format!(
        "{}\t{}\t{}",
        cap_str_chars(left, max_side_chars),
        cap_str_chars(right, max_side_chars),
        action
    )
}

fn source_fingerprint(path: &Path) -> Result<SourceFingerprint> {
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let modified_unix_secs = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        for byte in &buf[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(SourceFingerprint {
        path: path_string(path),
        len: metadata.len(),
        modified_unix_secs,
        content_hash: format!("{hash:016x}"),
    })
}

fn source_matches(a: &SourceFingerprint, b: &SourceFingerprint) -> bool {
    a.path == b.path && a.len == b.len && a.content_hash == b.content_hash
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = tmp_path_for(path);
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&tmp_path, text)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("cache")
        .to_string();
    name.push_str(&format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}
