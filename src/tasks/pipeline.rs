use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{data, tasks, util};

const WORLD_TEXT_DATA: &str = "data/ultrachat_pairs.txt";
const WORLD_DATA: &str = "data/world_mix_pairs.txt";
const GO_CODE_DATA: &str = "data/go_code_pairs.txt";
const WIKI_DATA: &str = "data/cached_wikimedia_wikipedia_1.txt";
const ENCODER_DATA: &str = "data/encoder_mix.txt";
const EVAL_SUITE: &str = "eval/code_assistant_go_hard.jsonl";
const GO_TASK_DATA: &str = "data/go_instruction_pairs.txt";
const GO_REPAIR_DATA: &str = "data/go_repair_pairs.txt";
const GO_FEEDBACK_TRAIN_DATA: &str = "data/code_poc_go_mix.txt";
const CODE_TRAIN_DATA: &str = "data/code_poc_mix.txt";
const CACHE_DIR: &str = "data/cache";
const MODEL_PROFILES_PATH: &str = "config/model_profiles.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoryProfile {
    Eight,
    FortyEight,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ProfileDefaults {
    latent_steps: usize,
    world_steps: usize,
    high_world_steps: usize,
    code_decoder_steps: usize,
    go_feedback_steps: usize,
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
    go_feedback_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    code_decoder_grad_accum: usize,
    go_feedback_grad_accum: usize,
}

#[derive(Deserialize)]
struct ModelProfiles {
    #[serde(rename = "8gb")]
    eight_gb: ProfileDefaults,
    #[serde(rename = "48gb")]
    forty_eight_gb: ProfileDefaults,
}

#[derive(Debug)]
struct PipelineConfig {
    profile: MemoryProfile,
    resume: bool,
    resume_selector: Option<String>,
    with_code_eval: bool,
}

#[derive(Debug)]
struct PipelinePaths {
    run_id: String,
    run_root: PathBuf,
    latent_stage_dir: PathBuf,
    world_stage_dir: PathBuf,
    high_world_stage_dir: PathBuf,
    decoder_stage_dir: PathBuf,
    decoder_go_feedback_stage_dir: PathBuf,
    code_eval_stage_dir: PathBuf,
    latent_model: PathBuf,
    world_model: PathBuf,
    world_encoder_model: PathBuf,
    high_world_model: PathBuf,
    code_decoder_base_model: PathBuf,
    code_decoder_go_feedback_model: PathBuf,
    code_decoder_model: PathBuf,
    code_decoder_vocab: PathBuf,
    encoder_cache_vocab: PathBuf,
    code_decoder_cache_vocab: PathBuf,
}

#[derive(Serialize)]
struct PipelineMeta<'a> {
    pipeline_run_id: &'a str,
    pipeline_kind: &'a str,
    resume_enabled: bool,
    resume_selector: &'a str,
    run_root: String,
    latent_model: String,
    world_model: String,
    world_encoder_model: String,
    high_world_model: String,
    code_decoder_model: String,
    code_decoder_base_model: String,
    code_decoder_go_feedback_model: String,
    code_decoder_vocab: String,
    encoder_data: &'a str,
    world_data: &'a str,
    code_train_data: &'a str,
    eval_suite: &'a str,
    profile: &'a str,
    with_code_eval: bool,
    latent_steps: usize,
    world_steps: usize,
    high_world_steps: usize,
    code_decoder_steps: usize,
    go_feedback_steps: usize,
    latent_batch: usize,
    world_batch: usize,
    code_decoder_batch: usize,
    go_feedback_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    code_decoder_grad_accum: usize,
    go_feedback_grad_accum: usize,
    dim: usize,
    layers: usize,
    heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    decoder_dim: usize,
    decoder_layers: usize,
    decoder_heads: usize,
    decoder_ff_dim: usize,
}

pub fn try_run_pipeline(args: &[String]) -> Result<bool> {
    if args.len() < 2 || !matches!(args[1].as_str(), "train" | "--train") {
        return Ok(false);
    }
    let cfg = PipelineConfig::from_args(&args[2..])?;
    run_pipeline(cfg)?;
    Ok(true)
}

impl MemoryProfile {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "8gb" => Ok(Self::Eight),
            "48gb" => Ok(Self::FortyEight),
            other => bail!("unsupported train profile '{other}' (expected 8gb or 48gb)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Eight => "8gb",
            Self::FortyEight => "48gb",
        }
    }

    fn defaults(self) -> Result<ProfileDefaults> {
        let path = std::env::var("TOFY_MODEL_PROFILES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(MODEL_PROFILES_PATH));
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read model profile config from {:?}", path))?;
        let profiles: ModelProfiles = serde_json::from_str(&raw)
            .with_context(|| format!("parse model profile config from {:?}", path))?;
        Ok(match self {
            Self::Eight => profiles.eight_gb,
            Self::FortyEight => profiles.forty_eight_gb,
        })
    }
}

impl PipelineConfig {
    fn from_args(args: &[String]) -> Result<Self> {
        let profile_arg = args.first().ok_or_else(|| {
            anyhow::anyhow!("usage: train <8gb|48gb> [--resume [latest|run]] [--with-code-eval]")
        })?;
        let profile = MemoryProfile::parse(profile_arg)?;
        let mut resume = false;
        let mut resume_selector = None;
        let mut with_code_eval = false;
        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--resume" => {
                    resume = true;
                    if args
                        .get(i + 1)
                        .is_some_and(|value| !value.starts_with("--"))
                    {
                        resume_selector = args.get(i + 1).cloned();
                        i += 2;
                    } else {
                        resume_selector = Some("latest".to_string());
                        i += 1;
                    }
                }
                "--with-code-eval" => {
                    with_code_eval = true;
                    i += 1;
                }
                other => bail!(
                    "unsupported train argument '{other}' (accepted: --resume, --with-code-eval)"
                ),
            }
        }
        Ok(Self {
            profile,
            resume,
            resume_selector,
            with_code_eval,
        })
    }
}

fn run_pipeline(cfg: PipelineConfig) -> Result<()> {
    let defaults = cfg.profile.defaults()?;
    set_pipeline_env(&cfg, &defaults);
    maybe_export_cuda_compat();

    let paths = resolve_pipeline_paths(&cfg, &defaults)?;
    fs::create_dir_all(&paths.run_root)?;
    for dir in [
        &paths.latent_stage_dir,
        &paths.world_stage_dir,
        &paths.high_world_stage_dir,
        &paths.decoder_stage_dir,
        &paths.decoder_go_feedback_stage_dir,
        &paths.code_eval_stage_dir,
    ] {
        fs::create_dir_all(dir)?;
    }
    std::env::set_var("TOFY_RUN_GROUP", &paths.run_id);
    write_launch(&paths, &cfg)?;
    write_meta(&paths, &cfg, &defaults)?;

    println!(
        "Training profile: {} | run: {}",
        cfg.profile.as_str(),
        paths.run_root.display()
    );
    println!(
        "Model: dim={} layers={} heads={} slots={} high_world_steps={}",
        defaults.dim,
        defaults.layers,
        defaults.heads,
        defaults.num_latent_tokens,
        defaults.high_world_steps
    );

    prepare_data(&paths, &defaults, cfg.resume)?;
    configure_encoder_vocab_env(&paths, &cfg)?;
    train_encoder(&paths, &cfg, &defaults)?;
    train_world(&paths, &cfg, &defaults)?;
    train_high_world(&paths, &cfg, &defaults)?;
    train_code_decoder(&paths, &cfg, &defaults)?;
    train_go_feedback_decoder(&paths, &cfg, &defaults)?;
    let selected_decoder = if cfg.with_code_eval {
        select_decoder_checkpoint(&paths, &defaults)?
    } else {
        select_trained_decoder_checkpoint(&paths)
    };
    let mut paths = paths;
    paths.code_decoder_model = selected_decoder;
    write_meta(&paths, &cfg, &defaults)?;
    if cfg.with_code_eval {
        final_eval(&paths, &defaults)?;
    } else {
        println!("Skipping code eval suite; pass --with-code-eval to run model code tests.");
    }

    println!("Pipeline complete.");
    println!(
        "Serve with: cargo run --release -- --serve {} {} {} {} {} {} {} {} {} {}",
        paths.world_encoder_model.display(),
        matched_encoder_vocab(&paths).display(),
        paths.world_model.display(),
        pipeline_serve_bind(),
        defaults.dim,
        defaults.world_max_seq,
        defaults.layers,
        defaults.heads,
        defaults.bridge_dim,
        defaults.num_latent_tokens
    );
    Ok(())
}

fn set_pipeline_env(cfg: &PipelineConfig, defaults: &ProfileDefaults) {
    set_env_default("TOFY_TRAIN_DTYPE", "bf16");
    set_env_default("TOFY_SIGREG_SLICES", "1024");
    set_env_default("TOFY_SIGREG_POINTS", "17");
    set_env_default("TOFY_LATENT_CONTEXT_SEGMENTS", "4");
    set_env_default("TOFY_LATENT_RECENT_FULL_SEGMENTS", "1");
    set_env_default("TOFY_LATENT_HISTORY_RATIO", "0.35");
    set_env_default("TOFY_WORLD_CONTEXT_SEGMENTS", "2");
    set_env_default("TOFY_WORLD_RECENT_FULL_SEGMENTS", "1");
    set_env_default("TOFY_RECURSIVE_CONTEXT_COMPRESSION", "1");
    set_env_default("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", "2");
    set_env_default("TOFY_WORLD_ROLLOUT_STEPS", "2");
    set_env_default("TOFY_WORLD_INVERSE_LOSS_WEIGHT", "0.0");
    set_env_default("TOFY_DECODER_SYNTAX_LOSS_WEIGHT", "0.0");
    set_env_default("TOFY_DECODER_SIGNATURE_LOSS_WEIGHT", "0.0");
    set_env_default("TOFY_DECODER_STRUCTURE_LOSS_WEIGHT", "0.0");
    set_env_default("TOFY_DECODER_CONDITIONING_LOSS_WEIGHT", "0.0");
    set_env_default("TOFY_DECODER_CONDITIONING_MARGIN", "0.10");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_ROWS", "500000");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_BYTES", "67108864");
    set_env_default("TOFY_BPE_MAX_MERGES", "8192");
    set_env_default("TOFY_BPE_PROGRESS_EVERY_MERGES", "128");
    set_env_default("TOFY_CODE_VOCAB_SAMPLE_ROWS", "25000");
    set_env_default("TOFY_CODE_VOCAB_SAMPLE_BYTES", "16777216");
    set_env_default_owned(
        "TOFY_LATENT_WARMUP_BATCH",
        defaults.latent_warmup_batch.to_string(),
    );
    set_env_default("TOFY_LATENT_WARMUP_GRAD_ACCUM", "1");
    set_env_default_owned(
        "TOFY_WORLD_WARMUP_BATCH",
        defaults.world_warmup_batch.to_string(),
    );
    set_env_default("TOFY_WORLD_WARMUP_GRAD_ACCUM", "1");
    set_env_default("TOFY_WORLD_WARMUP_STEPS", "1200");
    set_env_default("TOFY_WORLD_LOG_EVERY", "1000");
    set_env_default("TOFY_ORCHESTRATOR_LOG_EVERY", "500");
    set_env_default("TOFY_DECODER_WARMUP_BATCH", "16");
    set_env_default("TOFY_DECODER_WARMUP_GRAD_ACCUM", "16");
    set_env_default("TOFY_DECODER_WARMUP_STEPS", "1000");
    set_env_default("TOFY_DECODER_LOG_EVERY", "500");
    set_env_default("TOFY_HWM_MACRO_MIN_LEN", "2");
    set_env_default("TOFY_HWM_MACRO_MAX_LEN", "4");
    set_env_default("TOFY_CACHE_DIR", CACHE_DIR);
    set_env_default("TOFY_CACHE_PREFETCH_BATCHES", "4");
    std::env::remove_var("TOFY_USE_TOKEN_CACHE");
    std::env::remove_var("TOFY_ENCODER_VOCAB");
    if cfg.profile == MemoryProfile::Eight {
        set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "16");
    }
    if cfg.resume {
        std::env::set_var("TOFY_RESUME", "1");
    } else {
        std::env::remove_var("TOFY_RESUME");
    }
}

fn resolve_pipeline_paths(
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<PipelinePaths> {
    let (run_id, run_root) = if cfg.resume {
        let selector = cfg.resume_selector.as_deref().unwrap_or("latest");
        let run_root = resolve_run_root(selector)?;
        let run_id = run_root
            .file_name()
            .and_then(|name| name.to_str())
            .context("resume run path has no final component")?
            .to_string();
        (run_id, run_root)
    } else {
        let run_id = format!("code_poc_{}", unix_timestamp()?);
        let run_root = runs_dir().join(&run_id);
        if run_root.exists() {
            bail!("run directory already exists: {}", run_root.display());
        }
        (run_id, run_root)
    };

    let latent_stage_dir = run_root.join("latent");
    let world_stage_dir = run_root.join("world");
    let high_world_stage_dir = run_root.join("high_world");
    let decoder_stage_dir = run_root.join("decoder_code");
    let decoder_go_feedback_stage_dir = run_root.join("decoder_code_go_feedback");
    let code_eval_stage_dir = run_root.join("code_eval");
    let latent_model = latent_stage_dir.join("model.safetensors");
    let world_model = world_stage_dir.join("model.safetensors");
    let world_encoder_model = world_stage_dir.join("model.encoder.safetensors");
    let high_world_model = high_world_stage_dir.join("model.safetensors");
    let code_decoder_base_model = decoder_stage_dir.join("model.safetensors");
    let code_decoder_go_feedback_model = decoder_go_feedback_stage_dir.join("model.safetensors");
    let code_decoder_model = code_decoder_base_model.clone();
    let code_decoder_vocab = PathBuf::from(format!(
        "{}.vocab.txt",
        trim_safetensors(&code_decoder_base_model)
    ));
    let encoder_cache_vocab =
        vocab_dir().join(format!("vocab_encoder_{}_default.txt", defaults.max_vocab));
    let code_decoder_cache_vocab = vocab_dir().join(format!(
        "vocab_code_{}_codeaware.txt",
        defaults.code_decoder_max_vocab
    ));

    Ok(PipelinePaths {
        run_id,
        run_root,
        latent_stage_dir,
        world_stage_dir,
        high_world_stage_dir,
        decoder_stage_dir,
        decoder_go_feedback_stage_dir,
        code_eval_stage_dir,
        latent_model,
        world_model,
        world_encoder_model,
        high_world_model,
        code_decoder_base_model,
        code_decoder_go_feedback_model,
        code_decoder_model,
        code_decoder_vocab,
        encoder_cache_vocab,
        code_decoder_cache_vocab,
    })
}

fn prepare_data(paths: &PipelinePaths, defaults: &ProfileDefaults, resume: bool) -> Result<()> {
    println!("== Stage 1/6: data prep + vocab/token cache ==");
    thread::scope(|scope| -> Result<()> {
        let code_handle = scope.spawn(prepare_code_source_data);
        let source_handle = scope.spawn(ensure_pipeline_source_data);
        let eval_handle =
            scope.spawn(|| run_prepare(["--generate-go-code-eval-suite", "--output", EVAL_SUITE]));

        join_result(code_handle, "github code data")?;
        join_result(source_handle, "pipeline source data")?;
        join_result(eval_handle, "eval suite")?;
        Ok(())
    })?;

    let encoder_inputs = vec![
        WORLD_TEXT_DATA.to_string(),
        WIKI_DATA.to_string(),
        GO_CODE_DATA.to_string(),
    ];

    thread::scope(|scope| -> Result<()> {
        let encoder_inputs_for_corpus = encoder_inputs.clone();
        let encoder_corpus_handle = scope.spawn(move || {
            let mut encoder_args = vec![
                "--prepare-encoder-corpus".to_string(),
                "--output".to_string(),
                ENCODER_DATA.to_string(),
            ];
            encoder_args.extend(encoder_inputs_for_corpus);
            run_prepare_vec(encoder_args)
        });

        prepare_go_task_data()?;

        let go_repair_handle = if command_available("go") {
            Some(scope.spawn(|| {
                run_prepare([
                    "--prepare-go-repair-tasks",
                    "--input",
                    GO_TASK_DATA,
                    "--output",
                    GO_REPAIR_DATA,
                    "--go",
                    "go",
                    "--variants-per-sample",
                    "2",
                    "--timeout-sec",
                    "3.0",
                    "--max-rows",
                    "4000",
                    "--progress-every",
                    "100",
                ])
            }))
        } else {
            println!("Go compiler-feedback repair pairs skipped: go not found.");
            None
        };

        if let Some(handle) = go_repair_handle {
            join_result(handle, "go repair tasks")?;
        }

        let (world_args, code_mix_args, go_feedback_mix_args) = build_stage1_mix_args();
        let world_handle = scope.spawn(move || run_prepare_vec(world_args));
        let code_mix_handle = scope.spawn(move || run_prepare_vec(code_mix_args));
        let go_feedback_mix_handle = scope.spawn(move || run_prepare_vec(go_feedback_mix_args));

        join_result(world_handle, "world mix")?;
        join_result(code_mix_handle, "code decoder mix")?;
        join_result(go_feedback_mix_handle, "go feedback decoder mix")?;
        join_result(encoder_corpus_handle, "encoder corpus")?;
        Ok(())
    })?;

    run_cache(vec![
        "--prepare-pipeline-cache".to_string(),
        ENCODER_DATA.to_string(),
        WORLD_DATA.to_string(),
        CODE_TRAIN_DATA.to_string(),
        paths.encoder_cache_vocab.to_string_lossy().to_string(),
        paths.code_decoder_cache_vocab.to_string_lossy().to_string(),
        cache_dir().to_string_lossy().to_string(),
        "--encoder-max-vocab".to_string(),
        defaults.max_vocab.to_string(),
        "--code-max-vocab".to_string(),
        defaults.code_decoder_max_vocab.to_string(),
        "--encoder-max-seq".to_string(),
        (defaults.latent_max_seq * 4).to_string(),
        "--world-max-seq".to_string(),
        defaults.world_max_seq.to_string(),
        "--code-max-seq".to_string(),
        defaults.code_decoder_max_seq.to_string(),
    ])?;
    if paths.code_decoder_cache_vocab.exists() && !paths.code_decoder_vocab.exists() {
        if let Some(parent) = paths.code_decoder_vocab.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&paths.code_decoder_cache_vocab, &paths.code_decoder_vocab)?;
    }
    if !resume {
        std::env::set_var("TOFY_ENCODER_VOCAB", &paths.encoder_cache_vocab);
    }
    Ok(())
}

fn prepare_code_source_data() -> Result<()> {
    if !nonempty_file(GO_CODE_DATA) {
        run_prepare([
            "--prepare-github-top-code",
            "--output",
            GO_CODE_DATA,
            "--languages",
            "Go",
            "--max-files",
            "120000",
        ])?;
    }
    ensure_nonempty_file(GO_CODE_DATA)?;
    Ok(())
}

fn prepare_go_task_data() -> Result<()> {
    run_prepare([
        "--prepare-go-function-tasks",
        "--input",
        GO_CODE_DATA,
        "--output",
        GO_TASK_DATA,
    ])?;
    if !nonempty_file(GO_TASK_DATA) {
        run_prepare([
            "--prepare-go-function-tasks",
            "--github-top-code",
            "--max-files",
            "120000",
            "--output",
            GO_TASK_DATA,
        ])?;
    }
    Ok(())
}

fn build_stage1_mix_args() -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut extra_code_mix_args = Vec::new();
    let mut world_args = vec![
        "--prepare-world-mix".to_string(),
        "--output".to_string(),
        WORLD_DATA.to_string(),
        "--text-pairs".to_string(),
        WORLD_TEXT_DATA.to_string(),
        "--code-pairs".to_string(),
        GO_CODE_DATA.to_string(),
        "--code-pairs".to_string(),
        GO_TASK_DATA.to_string(),
    ];
    if nonempty_file(GO_REPAIR_DATA) {
        world_args.extend(["--code-pairs".to_string(), GO_REPAIR_DATA.to_string()]);
        extra_code_mix_args.extend([
            "--extra-pairs".to_string(),
            GO_REPAIR_DATA.to_string(),
            "--extra-repeat".to_string(),
            "2".to_string(),
        ]);
    }
    world_args.extend([
        "--code-ratio".to_string(),
        "0.45".to_string(),
        "--done-ratio".to_string(),
        "0.18".to_string(),
        "--max-rows".to_string(),
        "0".to_string(),
    ]);

    let mut code_mix_args = vec![
        "--prepare-code-poc-mix".to_string(),
        "--output".to_string(),
        CODE_TRAIN_DATA.to_string(),
        "--base-pairs".to_string(),
        GO_CODE_DATA.to_string(),
        "--instruction-pairs".to_string(),
        GO_TASK_DATA.to_string(),
        "--instruction-repeat".to_string(),
        "6".to_string(),
    ];
    code_mix_args.extend(extra_code_mix_args);
    code_mix_args.extend(["--max-rows".to_string(), "0".to_string()]);
    let mut go_feedback_mix_args = vec![
        "--prepare-code-poc-mix".to_string(),
        "--output".to_string(),
        GO_FEEDBACK_TRAIN_DATA.to_string(),
        "--base-pairs".to_string(),
        GO_CODE_DATA.to_string(),
        "--instruction-pairs".to_string(),
        GO_TASK_DATA.to_string(),
        "--instruction-repeat".to_string(),
        "6".to_string(),
    ];
    if nonempty_file(GO_REPAIR_DATA) {
        go_feedback_mix_args.extend([
            "--extra-pairs".to_string(),
            GO_REPAIR_DATA.to_string(),
            "--extra-repeat".to_string(),
            "3".to_string(),
        ]);
    }
    go_feedback_mix_args.extend(["--max-rows".to_string(), "0".to_string()]);
    (world_args, code_mix_args, go_feedback_mix_args)
}

fn join_result<T>(handle: thread::ScopedJoinHandle<'_, Result<T>>, label: &str) -> Result<T> {
    handle.join().map_err(|_| anyhow!("{label} panicked"))?
}

fn configure_encoder_vocab_env(paths: &PipelinePaths, cfg: &PipelineConfig) -> Result<()> {
    if cfg.resume {
        let matched_vocab = matched_encoder_vocab(paths);
        if matched_vocab.exists() {
            std::env::set_var("TOFY_ENCODER_VOCAB", matched_vocab);
            return Ok(());
        }
        if paths.encoder_cache_vocab.exists() {
            std::env::set_var("TOFY_ENCODER_VOCAB", &paths.encoder_cache_vocab);
            return Ok(());
        }
        let default_vocab = vocab_dir().join("vocab_encoder.txt");
        if default_vocab.exists() {
            std::env::set_var("TOFY_ENCODER_VOCAB", default_vocab);
            return Ok(());
        }
        return Ok(());
    }

    if paths.encoder_cache_vocab.exists() {
        std::env::set_var("TOFY_ENCODER_VOCAB", &paths.encoder_cache_vocab);
        println!(
            "Training will use cached encoder vocab {}",
            paths.encoder_cache_vocab.display()
        );
    } else {
        bail!(
            "pipeline cache did not produce encoder vocab: {}",
            paths.encoder_cache_vocab.display()
        );
    }
    Ok(())
}

fn ensure_pipeline_source_data() -> Result<()> {
    if !nonempty_file(WORLD_TEXT_DATA) {
        run_ultrachat_prepare(["--prepare-ultrachat", WORLD_TEXT_DATA, "6", "2"])?;
    }
    ensure_nonempty_file(WORLD_TEXT_DATA)?;

    if !nonempty_file(WIKI_DATA) {
        if Path::new(WIKI_DATA).exists() {
            fs::remove_file(WIKI_DATA)
                .with_context(|| format!("remove empty Wikipedia cache {}", WIKI_DATA))?;
        }
        let cached = data::ensure_hub_wikipedia_cached_with_max_files(
            "wikimedia/wikipedia",
            &hub_cache_dir(),
            Some(1),
        )?;
        if cached != Path::new(WIKI_DATA) && !Path::new(WIKI_DATA).exists() {
            fs::copy(&cached, WIKI_DATA).with_context(|| {
                format!(
                    "copy cached Wikipedia data from {} to {}",
                    cached.display(),
                    WIKI_DATA
                )
            })?;
        }
    }
    ensure_nonempty_file(WIKI_DATA)?;
    Ok(())
}

fn train_encoder(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 2/6: encoder ==");
    if stage_complete(cfg, &paths.latent_model, "latent", defaults.latent_steps)? {
        println!(
            "Skipping encoder; resume state already reached {} steps.",
            defaults.latent_steps
        );
        return Ok(());
    }
    let args = vec![
        "jepa_ai".to_string(),
        "--latent".to_string(),
        ENCODER_DATA.to_string(),
        defaults.latent_steps.to_string(),
        defaults.latent_batch.to_string(),
        defaults.dim.to_string(),
        defaults.latent_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.max_vocab.to_string(),
        "--grad-accum".to_string(),
        defaults.latent_grad_accum.to_string(),
        "--output".to_string(),
        paths.latent_model.to_string_lossy().to_string(),
    ];
    with_stage("latent", || {
        tasks::latent::try_run_train(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.latent_model)?;
    Ok(())
}

fn train_world(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 3/6: action-conditioned state transition ==");
    if stage_complete(cfg, &paths.world_model, "world", defaults.world_steps)? {
        println!(
            "Skipping action-conditioned state transition; resume state already reached {} steps.",
            defaults.world_steps
        );
        return Ok(());
    }
    let vocab = matched_encoder_vocab(paths);
    let mut args = vec![
        "jepa_ai".to_string(),
        "--train-world".to_string(),
        paths.latent_model.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        WORLD_DATA.to_string(),
        defaults.world_steps.to_string(),
        defaults.world_batch.to_string(),
        defaults.dim.to_string(),
        defaults.world_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--lambda".to_string(),
        "0.2".to_string(),
        "--lr".to_string(),
        "2e-4".to_string(),
        "--grad-accum".to_string(),
        defaults.world_grad_accum.to_string(),
        "--output".to_string(),
        paths.world_model.to_string_lossy().to_string(),
        "--encoder-output".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        "--action-loss-weight".to_string(),
        "0.0".to_string(),
    ];
    args.push("--freeze-encoder".to_string());
    with_stage("world", || {
        tasks::world::try_run_train(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.world_model)?;
    if !paths.world_encoder_model.exists() {
        fs::copy(&paths.latent_model, &paths.world_encoder_model).with_context(|| {
            format!(
                "copy frozen encoder export from {:?} to {:?}",
                paths.latent_model, paths.world_encoder_model
            )
        })?;
    }
    ensure_file(&paths.world_encoder_model)?;
    Ok(())
}

fn train_high_world(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 3b/6: integrated high-level action-conditioned state transition ==");
    if stage_complete(
        cfg,
        &paths.high_world_model,
        "high_world",
        defaults.high_world_steps,
    )? {
        println!(
            "Skipping high-level action-conditioned state transition; resume state already reached {} steps.",
            defaults.high_world_steps
        );
        std::env::set_var("TOFY_HIGH_WORLD_MODEL", &paths.high_world_model);
        return Ok(());
    }
    let vocab = matched_encoder_vocab(paths);
    let args = vec![
        "jepa_ai".to_string(),
        "--train-high-world".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        paths.world_model.to_string_lossy().to_string(),
        WORLD_DATA.to_string(),
        defaults.high_world_steps.to_string(),
        defaults.world_batch.to_string(),
        defaults.dim.to_string(),
        defaults.world_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--macro-min-len".to_string(),
        "2".to_string(),
        "--macro-max-len".to_string(),
        "4".to_string(),
        "--lambda".to_string(),
        "0.2".to_string(),
        "--lr".to_string(),
        "2e-4".to_string(),
        "--grad-accum".to_string(),
        defaults.world_grad_accum.to_string(),
        "--output".to_string(),
        paths.high_world_model.to_string_lossy().to_string(),
    ];
    with_stage("high_world", || {
        tasks::world::try_run_train_high_world(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.high_world_model)?;
    std::env::set_var("TOFY_HIGH_WORLD_MODEL", &paths.high_world_model);
    Ok(())
}

fn train_code_decoder(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 5/6: code decoder ==");
    if stage_complete(
        cfg,
        &paths.code_decoder_base_model,
        "decoder_code",
        defaults.code_decoder_steps,
    )? {
        println!(
            "Skipping code decoder; resume state already reached {} steps.",
            defaults.code_decoder_steps
        );
        return Ok(());
    }
    let args = decoder_args(
        paths,
        defaults,
        CODE_TRAIN_DATA,
        defaults.code_decoder_steps,
        defaults.code_decoder_batch,
        defaults.code_decoder_grad_accum,
        &paths.code_decoder_base_model,
        None,
        "3e-4",
        "0.0",
    );
    with_stage("decoder_code", || {
        tasks::world::try_run_train_decoder(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.code_decoder_base_model)?;
    Ok(())
}

fn train_go_feedback_decoder(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 5b/6: Go execution-feedback decoder training ==");
    if stage_complete(
        cfg,
        &paths.code_decoder_go_feedback_model,
        "decoder_code_go_feedback",
        defaults.go_feedback_steps,
    )? {
        println!(
            "Skipping Go feedback decoder training; resume state already reached {} steps.",
            defaults.go_feedback_steps
        );
        return Ok(());
    }
    let args = decoder_args(
        paths,
        defaults,
        GO_FEEDBACK_TRAIN_DATA,
        defaults.go_feedback_steps,
        defaults.go_feedback_batch,
        defaults.go_feedback_grad_accum,
        &paths.code_decoder_go_feedback_model,
        Some(&paths.code_decoder_base_model),
        "1e-4",
        "0.0",
    );
    with_stage("decoder_code_go_feedback", || {
        tasks::world::try_run_train_decoder(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.code_decoder_go_feedback_model)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decoder_args(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
    data_path: &str,
    steps: usize,
    batch: usize,
    grad_accum: usize,
    output: &Path,
    init_decoder: Option<&Path>,
    lr: &str,
    conditioning_loss_weight: &str,
) -> Vec<String> {
    let mut args = vec![
        "jepa_ai".to_string(),
        "--train-decoder".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        matched_encoder_vocab(paths).to_string_lossy().to_string(),
        paths.world_model.to_string_lossy().to_string(),
        data_path.to_string(),
        steps.to_string(),
        batch.to_string(),
        defaults.code_decoder_max_seq.to_string(),
        defaults.dim.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--decoder-kind".to_string(),
        "code".to_string(),
        "--decoder-max-vocab".to_string(),
        defaults.code_decoder_max_vocab.to_string(),
        "--decoder-dim".to_string(),
        defaults.decoder_dim.to_string(),
        "--decoder-layers".to_string(),
        defaults.decoder_layers.to_string(),
        "--decoder-heads".to_string(),
        defaults.decoder_heads.to_string(),
        "--decoder-ff-dim".to_string(),
        defaults.decoder_ff_dim.to_string(),
        "--decoder-vocab".to_string(),
        paths.code_decoder_vocab.to_string_lossy().to_string(),
        "--decoder-output".to_string(),
        output.to_string_lossy().to_string(),
        "--grad-accum".to_string(),
        grad_accum.to_string(),
        "--lr".to_string(),
        lr.to_string(),
        "--conditioning-loss-weight".to_string(),
        conditioning_loss_weight.to_string(),
    ];
    if let Some(init_decoder) = init_decoder {
        args.extend([
            "--init-decoder".to_string(),
            init_decoder.to_string_lossy().to_string(),
        ]);
    }
    args
}

fn select_decoder_checkpoint(paths: &PipelinePaths, defaults: &ProfileDefaults) -> Result<PathBuf> {
    println!("== Stage 5c/6: verifier-guided checkpoint selection ==");
    run_code_eval_with_label(paths, defaults, &paths.code_decoder_base_model, "base")?;
    if paths.code_decoder_go_feedback_model.exists() {
        run_code_eval_with_label(
            paths,
            defaults,
            &paths.code_decoder_go_feedback_model,
            "go_feedback",
        )?;
    }
    let base = read_summary_metrics(&paths.code_eval_stage_dir.join("base_summary.txt"))?;
    let mut selected = (&paths.code_decoder_base_model, "base", base);
    let go_feedback_summary = paths.code_eval_stage_dir.join("go_feedback_summary.txt");
    if go_feedback_summary.exists() {
        let go_feedback = read_summary_metrics(&go_feedback_summary)?;
        if metrics_better(&go_feedback, &selected.2) {
            selected = (
                &paths.code_decoder_go_feedback_model,
                "go_feedback",
                go_feedback,
            );
        }
    }
    println!(
        "Selected {} decoder for final eval/promotion: {}",
        selected.1,
        selected.0.display()
    );
    Ok(selected.0.clone())
}

fn select_trained_decoder_checkpoint(paths: &PipelinePaths) -> PathBuf {
    if paths.code_decoder_go_feedback_model.exists() {
        println!(
            "Code eval skipped; selecting Go feedback decoder by default: {}",
            paths.code_decoder_go_feedback_model.display()
        );
        paths.code_decoder_go_feedback_model.clone()
    } else {
        println!(
            "Code eval skipped; selecting base decoder by default: {}",
            paths.code_decoder_base_model.display()
        );
        paths.code_decoder_base_model.clone()
    }
}

fn final_eval(paths: &PipelinePaths, defaults: &ProfileDefaults) -> Result<()> {
    println!("== Stage 6/6: code eval suite ==");
    run_code_eval(paths, defaults, &paths.code_decoder_model, "code_eval")
}

fn run_code_eval_with_label(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
    decoder_path: &Path,
    label: &str,
) -> Result<()> {
    let stage = format!("code_eval_{label}");
    println!("Evaluating {label} decoder: {}", decoder_path.display());
    run_code_eval(paths, defaults, decoder_path, &stage)?;
    let summary_path = paths.run_root.join(&stage).join("summary.txt");
    let summary_dest = paths
        .code_eval_stage_dir
        .join(format!("{label}_summary.txt"));
    ensure_file(&summary_path)?;
    fs::copy(&summary_path, &summary_dest)?;
    Ok(())
}

fn run_code_eval(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
    decoder_path: &Path,
    stage: &str,
) -> Result<()> {
    let mut args = vec![
        "jepa_ai".to_string(),
        "--eval-code-assistant".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        matched_encoder_vocab(paths).to_string_lossy().to_string(),
        paths.world_model.to_string_lossy().to_string(),
        EVAL_SUITE.to_string(),
        "384".to_string(),
        defaults.dim.to_string(),
        defaults.world_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--high-world-model".to_string(),
        paths.high_world_model.to_string_lossy().to_string(),
        "--code-decoder".to_string(),
        decoder_path.to_string_lossy().to_string(),
        "--code-decoder-vocab".to_string(),
        paths.code_decoder_vocab.to_string_lossy().to_string(),
        "--candidates".to_string(),
        "4".to_string(),
        "--repair-attempts".to_string(),
        "2".to_string(),
        "--go-timeout-sec".to_string(),
        "6".to_string(),
    ];
    with_stage(stage, || tasks::eval::try_run_code_eval(&args))?;
    args.clear();
    Ok(())
}

fn run_prepare<const N: usize>(args: [&str; N]) -> Result<()> {
    run_prepare_vec(args.into_iter().map(str::to_string).collect())
}

fn run_ultrachat_prepare<const N: usize>(args: [&str; N]) -> Result<()> {
    let mut full_args = vec!["jepa_ai".to_string()];
    full_args.extend(args.into_iter().map(str::to_string));
    if !tasks::latent::try_run_prepare_ultrachat(&full_args)? {
        bail!(
            "UltraChat prepare command was not handled: {}",
            full_args[1..].join(" ")
        );
    }
    Ok(())
}

fn run_prepare_vec(args: Vec<String>) -> Result<()> {
    let mut full_args = vec!["jepa_ai".to_string()];
    full_args.extend(args);
    if !tasks::prepare::try_run_prepare(&full_args)? {
        bail!(
            "prepare command was not handled: {}",
            full_args[1..].join(" ")
        );
    }
    Ok(())
}

fn run_cache(args: Vec<String>) -> Result<()> {
    let mut full_args = vec!["jepa_ai".to_string()];
    full_args.extend(args);
    if !tasks::cache::try_run_prepare_pipeline_cache(&full_args)? {
        bail!(
            "pipeline cache preparation was not handled: {}",
            full_args[1..].join(" ")
        );
    }
    Ok(())
}

fn with_stage<F>(stage: &str, f: F) -> Result<()>
where
    F: FnOnce() -> Result<bool>,
{
    std::env::set_var("TOFY_RUN_STAGE_NAME", stage);
    let result = f();
    std::env::remove_var("TOFY_RUN_STAGE_NAME");
    let handled = result?;
    if !handled {
        bail!("pipeline stage {stage} was not handled by its task entrypoint");
    }
    Ok(())
}

fn append_resume(mut args: Vec<String>, resume: bool) -> Vec<String> {
    if resume {
        args.push("--resume".to_string());
    }
    args
}

fn stage_complete(
    cfg: &PipelineConfig,
    model_path: &Path,
    stage: &str,
    target_steps: usize,
) -> Result<bool> {
    if !cfg.resume || !model_path.exists() {
        return Ok(false);
    }
    let state_path = util::checkpoint_sidecar_path(model_path, stage, "resume.json");
    let Some(state) = util::load_resume_state(&state_path, stage)? else {
        return Ok(false);
    };
    Ok(state.step >= target_steps)
}

fn resolve_run_root(selector: &str) -> Result<PathBuf> {
    if selector == "latest" || selector.is_empty() {
        return latest_run_root("code_poc_");
    }
    let direct = PathBuf::from(selector);
    if direct.is_dir() {
        return Ok(direct);
    }
    let under_runs = runs_dir().join(selector);
    if under_runs.is_dir() {
        return Ok(under_runs);
    }
    bail!("could not resolve resume run '{selector}'")
}

fn latest_run_root(prefix: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    let runs_dir = runs_dir();
    for entry in fs::read_dir(&runs_dir)
        .with_context(|| format!("read runs directory {}", runs_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(prefix) {
            continue;
        }
        let modified = entry
            .metadata()?
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((modified, entry.path()));
    }
    candidates.sort_by(|(a, _), (b, _)| b.cmp(a));
    candidates
        .into_iter()
        .map(|(_, path)| path)
        .next()
        .context("no code_poc_ runs found")
}

fn read_summary_metrics(path: &Path) -> Result<SummaryMetrics> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut metrics = SummaryMetrics::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.parse::<f32>().unwrap_or(0.0);
        match key {
            "suite_pass_rate" => metrics.suite = value,
            "test_pass_rate" => metrics.tests = value,
            "compile_rate" => metrics.compile = value,
            "constraint_pass_rate" => metrics.constraints = value,
            _ => {}
        }
    }
    Ok(metrics)
}

#[derive(Clone, Copy, Debug, Default)]
struct SummaryMetrics {
    suite: f32,
    tests: f32,
    compile: f32,
    constraints: f32,
}

fn metrics_better(candidate: &SummaryMetrics, current: &SummaryMetrics) -> bool {
    for (lhs, rhs) in [
        (candidate.suite, current.suite),
        (candidate.tests, current.tests),
        (candidate.compile, current.compile),
        (candidate.constraints, current.constraints),
    ] {
        match lhs.partial_cmp(&rhs).unwrap_or(Ordering::Equal) {
            Ordering::Greater => return true,
            Ordering::Less => return false,
            Ordering::Equal => {}
        }
    }
    false
}

fn write_launch(paths: &PipelinePaths, cfg: &PipelineConfig) -> Result<()> {
    let selector = cfg.resume_selector.as_deref().unwrap_or("");
    let eval_flag = if cfg.with_code_eval {
        " --with-code-eval"
    } else {
        ""
    };
    let command = format!(
        "train {}{}{}",
        cfg.profile.as_str(),
        if selector.is_empty() {
            String::new()
        } else {
            format!(" --resume {selector}")
        },
        eval_flag
    );
    let content = format!(
        "timestamp_unix={}\ncommand={}\n",
        unix_timestamp()?,
        command
    );
    write_text_atomic(&paths.run_root.join("launch.txt"), &content)?;
    Ok(())
}

fn write_meta(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    let selector = cfg.resume_selector.as_deref().unwrap_or("");
    let meta = PipelineMeta {
        pipeline_run_id: &paths.run_id,
        pipeline_kind: "code",
        resume_enabled: cfg.resume,
        resume_selector: selector,
        run_root: paths.run_root.to_string_lossy().to_string(),
        latent_model: paths.latent_model.to_string_lossy().to_string(),
        world_model: paths.world_model.to_string_lossy().to_string(),
        world_encoder_model: paths.world_encoder_model.to_string_lossy().to_string(),
        high_world_model: paths.high_world_model.to_string_lossy().to_string(),
        code_decoder_model: paths.code_decoder_model.to_string_lossy().to_string(),
        code_decoder_base_model: paths.code_decoder_base_model.to_string_lossy().to_string(),
        code_decoder_go_feedback_model: paths
            .code_decoder_go_feedback_model
            .to_string_lossy()
            .to_string(),
        code_decoder_vocab: paths.code_decoder_vocab.to_string_lossy().to_string(),
        encoder_data: ENCODER_DATA,
        world_data: WORLD_DATA,
        code_train_data: CODE_TRAIN_DATA,
        eval_suite: EVAL_SUITE,
        profile: cfg.profile.as_str(),
        with_code_eval: cfg.with_code_eval,
        latent_steps: defaults.latent_steps,
        world_steps: defaults.world_steps,
        high_world_steps: defaults.high_world_steps,
        code_decoder_steps: defaults.code_decoder_steps,
        go_feedback_steps: defaults.go_feedback_steps,
        latent_batch: defaults.latent_batch,
        world_batch: defaults.world_batch,
        code_decoder_batch: defaults.code_decoder_batch,
        go_feedback_batch: defaults.go_feedback_batch,
        latent_grad_accum: defaults.latent_grad_accum,
        world_grad_accum: defaults.world_grad_accum,
        code_decoder_grad_accum: defaults.code_decoder_grad_accum,
        go_feedback_grad_accum: defaults.go_feedback_grad_accum,
        dim: defaults.dim,
        layers: defaults.layers,
        heads: defaults.heads,
        bridge_dim: defaults.bridge_dim,
        num_latent_tokens: defaults.num_latent_tokens,
        decoder_dim: defaults.decoder_dim,
        decoder_layers: defaults.decoder_layers,
        decoder_heads: defaults.decoder_heads,
        decoder_ff_dim: defaults.decoder_ff_dim,
    };
    write_text_atomic(
        &paths.run_root.join("meta.json"),
        &format!("{}\n", serde_json::to_string_pretty(&meta)?),
    )?;
    Ok(())
}

fn write_text_atomic(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}.tmp"))
            .unwrap_or_else(|| "tmp".to_string()),
    );
    fs::write(&tmp_path, content)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn maybe_export_cuda_compat() {
    if std::env::var("CUDA_COMPUTE_CAP").is_ok() || !command_available("nvidia-smi") {
        return;
    }
    let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
    else {
        return;
    };
    let value = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .replace(['.', ' '], "");
    if !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()) {
        std::env::set_var("CUDA_COMPUTE_CAP", value);
    }
}

fn matched_encoder_vocab(paths: &PipelinePaths) -> PathBuf {
    PathBuf::from(format!(
        "{}.vocab.txt",
        trim_safetensors(&paths.latent_model)
    ))
}

fn trim_safetensors(path: &Path) -> String {
    let raw = path.to_string_lossy();
    raw.strip_suffix(".safetensors").unwrap_or(&raw).to_string()
}

fn ensure_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        bail!("required file not found: {}", path.display());
    }
    Ok(())
}

fn ensure_nonempty_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    ensure_file(path)?;
    if !nonempty_file(path) {
        bail!("required file is empty: {}", path.display());
    }
    Ok(())
}

fn nonempty_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
}

fn command_available(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
        || Command::new(program)
            .arg("version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
}

fn unix_timestamp() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_secs())
}

fn set_env_default(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

fn set_env_default_owned(key: &str, value: String) {
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

fn pipeline_serve_bind() -> String {
    std::env::var("TOFY_SERVE_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string())
}

fn runs_dir() -> PathBuf {
    std::env::var("TOFY_RUNS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("runs"))
}

fn cache_dir() -> PathBuf {
    std::env::var("TOFY_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(CACHE_DIR))
}

fn vocab_dir() -> PathBuf {
    std::env::var("TOFY_VOCAB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("local_models/vocabs"))
}

fn hub_cache_dir() -> PathBuf {
    std::env::var("TOFY_HUB_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("TOFY_DATA_DIR").map(|dir| PathBuf::from(dir).join("hub")))
        .unwrap_or_else(|_| PathBuf::from("data"))
}
