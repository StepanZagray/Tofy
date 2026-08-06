use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::{tasks, util};

const WORLD_TEXT_DATA: &str = "data/fictional/veclab_knowledge_train.txt";
const ENCODER_DATA: &str = "data/fictional/veclab_encoder_mix.txt";
const EVAL_SUITE: &str = "eval/veclab_eval.jsonl";
const VECLAB_TASKS: &str = "data/fictional/veclab_tasks_train.txt";
const BRIDGE_TRANSFER_DATA: &str = "data/fictional/veclab_bridge_transfer.txt";
const CACHE_DIR: &str = "data/cache";
const MODEL_PROFILES_PATH: &str = "config/model_profiles.json";
const PREPARED_CACHE_COMPRESS_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;
const TRAINING_BRIDGE_EVALS: &[(&str, &str, &str)] = &[
    ("rag_ceiling", "rag", "weights"),
    ("latent_channel", "bridge", "context"),
    ("knowledge_in_weights", "bridge", "weights"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoryProfile {
    Minimal,
    FortyEight,
    Eighty,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ProfileDefaults {
    latent_steps: usize,
    world_steps: usize,
    bridge_steps: usize,
    dim: usize,
    latent_max_seq: usize,
    world_max_seq: usize,
    bridge_max_seq: usize,
    layers: usize,
    heads: usize,
    max_vocab: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    latent_batch: usize,
    latent_warmup_batch: usize,
    world_batch: usize,
    world_warmup_batch: usize,
    bridge_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    bridge_grad_accum: usize,
}

#[derive(Deserialize)]
struct ModelProfiles {
    minimal: ProfileDefaults,
    #[serde(rename = "48gb")]
    forty_eight_gb: ProfileDefaults,
    #[serde(rename = "80gb")]
    eighty_gb: ProfileDefaults,
}

#[derive(Debug)]
struct PipelineConfig {
    profile: MemoryProfile,
    until: PipelineUntil,
    resume: bool,
    resume_selector: Option<String>,
    skip_trained_stages: Vec<String>,
    with_code_eval: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PipelineUntil {
    Full,
}

#[derive(Debug)]
struct PipelinePaths {
    run_id: String,
    run_root: PathBuf,
    latent_stage_dir: PathBuf,
    world_stage_dir: PathBuf,
    bridge_stage_dir: PathBuf,
    eval_stage_dir: PathBuf,
    latent_model: PathBuf,
    world_model: PathBuf,
    world_encoder_model: PathBuf,
    bridge_context_model: PathBuf,
    bridge_weights_model: PathBuf,
    encoder_cache_vocab: PathBuf,
}

#[derive(Debug)]
struct PreparedCacheUploadFile {
    local_path: PathBuf,
    remote_path: PathBuf,
    size: u64,
    sha256: Option<String>,
}

#[derive(Debug)]
struct RemoteRepoFile {
    size: u64,
    oid: Option<String>,
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
    bridge_context_model: String,
    bridge_weights_model: String,
    encoder_data: &'a str,
    world_data: &'a str,
    eval_suite: &'a str,
    profile: &'a str,
    pipeline_until: &'a str,
    with_code_eval: bool,
    latent_steps: usize,
    world_steps: usize,
    bridge_steps: usize,
    latent_batch: usize,
    world_batch: usize,
    bridge_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    bridge_grad_accum: usize,
    dim: usize,
    layers: usize,
    heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
}

pub fn try_run_pipeline(args: &[String]) -> Result<bool> {
    if args.len() < 2 || !matches!(args[1].as_str(), "train" | "--train") {
        return Ok(false);
    }
    let cfg = PipelineConfig::from_args(&args[2..])?;
    run_pipeline(cfg)?;
    Ok(true)
}

pub fn try_run_prepare_cache(args: &[String]) -> Result<bool> {
    let rest = if args.get(1).is_some_and(|arg| arg == "prepare")
        && args.get(2).is_some_and(|arg| arg == "cache")
    {
        &args[3..]
    } else if args
        .get(1)
        .is_some_and(|arg| arg == "prepare-cache" || arg == "--prepare-cache")
    {
        &args[2..]
    } else {
        return Ok(false);
    };

    let profile_arg = rest.first().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: prepare cache <minimal|48gb|80gb> [--force] [--auto-hf-upload --hf-dataset <org/dataset-name>]"
        )
    })?;
    let profile = MemoryProfile::parse(profile_arg)?;
    let mut force = false;
    let mut auto_hf_upload = false;
    let mut hf_upload_dataset = None;
    let mut idx = 1usize;
    while idx < rest.len() {
        match rest[idx].as_str() {
            "--force" => {
                force = true;
                idx += 1;
            }
            "--auto-hf-upload" | "-auto-hf-upload" => {
                if auto_hf_upload {
                    bail!("--auto-hf-upload may only be specified once");
                }
                auto_hf_upload = true;
                idx += 1;
            }
            "--hf-dataset" | "-hf-dataset" => {
                let value = rest.get(idx + 1).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--hf-dataset requires a Hugging Face dataset id (org/dataset-name)"
                    )
                })?;
                if value.starts_with('-') {
                    bail!("--hf-dataset requires a Hugging Face dataset id (org/dataset-name)");
                }
                if hf_upload_dataset.is_some() {
                    bail!("--hf-dataset may only be specified once");
                }
                hf_upload_dataset = Some(parse_hf_dataset_repo(value)?);
                idx += 2;
            }
            other => bail!(
                "unsupported prepare cache argument '{other}' (accepted: --force, --auto-hf-upload, --hf-dataset <org/dataset-name>)"
            ),
        }
    }

    if auto_hf_upload && hf_upload_dataset.is_none() {
        bail!("--auto-hf-upload requires --hf-dataset <org/dataset-name>");
    }
    if hf_upload_dataset.is_some() && !auto_hf_upload {
        bail!("--hf-dataset requires --auto-hf-upload");
    }

    run_prepare_cache(profile, force, hf_upload_dataset)?;
    Ok(true)
}

impl MemoryProfile {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "minimal" => Ok(Self::Minimal),
            "48gb" => Ok(Self::FortyEight),
            "80gb" => Ok(Self::Eighty),
            other => bail!("unsupported train profile '{other}' (expected minimal, 48gb, or 80gb)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::FortyEight => "48gb",
            Self::Eighty => "80gb",
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
            Self::Minimal => profiles.minimal,
            Self::FortyEight => profiles.forty_eight_gb,
            Self::Eighty => profiles.eighty_gb,
        })
    }
}

impl PipelineUntil {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "full" => Ok(Self::Full),
            other => bail!("unsupported --until value '{other}' (expected full)"),
        }
    }

    fn as_str(self) -> &'static str {
        "full"
    }
}

impl PipelineConfig {
    fn from_args(args: &[String]) -> Result<Self> {
        let profile_arg = args.first().ok_or_else(|| {
            anyhow::anyhow!(
                "usage: train <minimal|48gb|80gb> [--until full] [--resume [latest|run]] [--skip-trained STAGE[,STAGE...]] [--with-code-eval]"
            )
        })?;
        let profile = MemoryProfile::parse(profile_arg)?;
        let mut until = PipelineUntil::Full;
        let mut resume = false;
        let mut resume_selector = None;
        let mut skip_trained_stages = parse_stage_list_env("TOFY_SKIP_TRAINED_STAGES");
        let mut with_code_eval = true;
        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--until" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--until requires full"))?;
                    until = PipelineUntil::parse(value)?;
                    i += 2;
                }
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
                "--skip-trained" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--skip-trained requires STAGE[,STAGE...]"))?;
                    skip_trained_stages.extend(parse_stage_list(value));
                    i += 2;
                }
                "--with-code-eval" => {
                    with_code_eval = true;
                    i += 1;
                }
                other => bail!(
                    "unsupported train argument '{other}' (accepted: --until, --resume, --skip-trained, --with-code-eval)"
                ),
            }
        }
        Ok(Self {
            profile,
            until,
            resume,
            resume_selector,
            skip_trained_stages,
            with_code_eval,
        })
    }
}

fn run_prepare_cache(
    profile: MemoryProfile,
    force: bool,
    hf_upload_dataset: Option<String>,
) -> Result<()> {
    let defaults = profile.defaults()?;
    let cfg = PipelineConfig {
        profile,
        until: PipelineUntil::Full,
        resume: false,
        resume_selector: None,
        skip_trained_stages: Vec::new(),
        with_code_eval: true,
    };
    set_pipeline_env(&cfg, &defaults);
    println!(
        "Preparing local data/cache handoff for {} profile; no model training will run.",
        profile.as_str()
    );
    prepare_data(
        &prepare_cache_paths(&defaults),
        &defaults,
        false,
        force,
        false,
        false,
    )?;
    println!("Data/cache preparation complete.");
    if let Some(repo) = hf_upload_dataset {
        upload_prepare_cache_tree(profile, &repo)?;
    }
    Ok(())
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
        &paths.bridge_stage_dir,
        &paths.eval_stage_dir,
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
        "Model: dim={} layers={} heads={} slots={} world_steps={} bridge_steps={}",
        defaults.dim,
        defaults.layers,
        defaults.heads,
        defaults.num_latent_tokens,
        defaults.world_steps,
        defaults.bridge_steps
    );
    let context_defaults = context_defaults_for_profile(cfg.profile, &defaults);
    println!(
        "Context: latent={} tokens, world/runtime={} tokens (hybrid memory)",
        defaults.latent_max_seq * context_defaults.latent_segments,
        defaults.world_max_seq * context_defaults.world_segments,
    );

    let reuse_encoder_data = skip_trained_stage(&cfg, "latent") && paths.latent_model.exists();
    let reuse_world_data = skip_trained_stage(&cfg, "world") && paths.world_model.exists();
    prepare_data(
        &paths,
        &defaults,
        cfg.resume,
        false,
        reuse_encoder_data,
        reuse_world_data,
    )?;
    configure_encoder_vocab_env(&paths, &cfg)?;
    train_encoder(&paths, &cfg, &defaults)?;
    train_world(&paths, &cfg, &defaults)?;
    train_bridge(&paths, &cfg, &defaults)?;
    if cfg.with_code_eval {
        final_eval(&paths, &cfg, &defaults)?;
    }

    println!("Pipeline complete.");
    Ok(())
}

fn set_pipeline_env(cfg: &PipelineConfig, defaults: &ProfileDefaults) {
    let context_defaults = context_defaults_for_profile(cfg.profile, defaults);
    set_env_default("TOFY_TRAIN_DTYPE", "bf16");
    set_env_default("TOFY_OPTIMIZER", "muon");
    set_env_default("TOFY_ADAMW_BETA2", "0.95");
    set_env_default("TOFY_WEIGHT_DECAY", "0.1");
    set_env_default("TOFY_MUON_MOMENTUM", "0.95");
    set_env_default("TOFY_MUON_NS_STEPS", "5");
    set_env_default("TOFY_MUON_RMS_SCALE", "0.18");
    set_env_default("TOFY_SIGREG_SLICES", "1024");
    set_env_default("TOFY_SIGREG_POINTS", "17");
    set_env_default_owned("TOFY_ENCODER_DIM", defaults.dim.to_string());
    set_env_default_owned("TOFY_ENCODER_LAYERS", defaults.layers.to_string());
    set_env_default_owned("TOFY_ENCODER_HEADS", defaults.heads.to_string());
    set_env_default_owned("TOFY_BRIDGE_DIM", defaults.bridge_dim.to_string());
    set_env_default_owned(
        "TOFY_NUM_LATENT_TOKENS",
        defaults.num_latent_tokens.to_string(),
    );
    set_env_default_owned("TOFY_WORLD_MAX_SEQ", defaults.world_max_seq.to_string());
    set_env_default_owned("TOFY_BRIDGE_MAX_SEQ", defaults.bridge_max_seq.to_string());
    set_env_default("TOFY_DECODER_CONDITIONING_NEGATIVES", "hard");
    set_env_default("TOFY_BRIDGE_MIN_SEMANTIC_GAP", "0.02");
    set_env_default("TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS", "true");
    set_env_default("TOFY_BRIDGE_TRAIN_FUNCTION_MAX", "80");
    set_env_default("TOFY_BRIDGE_VALIDATION_FUNCTION_MAX", "100");
    set_env_default("TOFY_DECODER_CONDITIONING_UNLIKELIHOOD_WEIGHT", "0.25");
    set_env_default("TOFY_DECODER_CONDITIONING_SEPARATION_WEIGHT", "0.05");
    set_env_default("TOFY_DECODER_CONDITIONING_MIN_DISTANCE", "0.1");
    set_env_default("TOFY_BRIDGE_SEMANTIC_WARMUP", "400");
    set_env_default("TOFY_BRIDGE_SEMANTIC_PATIENCE", "1200");
    set_env_default("TOFY_BRIDGE_MIN_SEMANTIC_PROGRESS", "0.002");
    set_env_default("TOFY_WORLD_RECON_LOSS_WEIGHT", "0.25");
    set_env_default("TOFY_WORLD_ASSOC_LOSS_WEIGHT", "1.0");
    set_env_default_owned(
        "TOFY_BRIDGE_GRAD_ACCUM",
        defaults.bridge_grad_accum.to_string(),
    );
    set_env_default_owned(
        "TOFY_LATENT_CONTEXT_SEGMENTS",
        context_defaults.latent_segments.to_string(),
    );
    set_env_default("TOFY_LATENT_RECENT_FULL_SEGMENTS", "1");
    set_env_default("TOFY_LATENT_HISTORY_RATIO", "0.35");
    set_env_default_owned(
        "TOFY_WORLD_CONTEXT_SEGMENTS",
        context_defaults.world_segments.to_string(),
    );
    set_env_default_owned(
        "TOFY_ENCODER_CONTEXT_SEGMENTS",
        context_defaults.world_segments.to_string(),
    );
    set_env_default("TOFY_WORLD_RECENT_FULL_SEGMENTS", "1");
    set_env_default("TOFY_ENCODER_RECENT_FULL_SEGMENTS", "1");
    set_env_default("TOFY_CONTEXT_HYBRID_MEMORY", "1");
    set_env_default_owned(
        "TOFY_CONTEXT_HYBRID_EXACT_TAIL",
        context_defaults.hybrid_exact_tail.to_string(),
    );
    set_env_default_owned(
        "TOFY_CONTEXT_HYBRID_BLOCK_SIZE",
        context_defaults.hybrid_block_size.to_string(),
    );
    set_env_default_owned(
        "TOFY_CONTEXT_RETRIEVAL_SLOTS",
        context_defaults.hybrid_retrieval_slots.to_string(),
    );
    set_env_default_owned(
        "TOFY_CONTEXT_EXACT_OLD_TOKENS",
        context_defaults.hybrid_exact_old_tokens.to_string(),
    );
    set_env_default_owned(
        "JEPA_CANDLE_DECODER_CTX",
        defaults
            .world_max_seq
            .saturating_mul(4)
            .max(768)
            .to_string(),
    );
    set_env_default("TOFY_LABEL_SMOOTHING", "0.05");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_ROWS", "500000");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_BYTES", "67108864");
    set_env_default("TOFY_BPE_MAX_MERGES", "24000");
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
    set_env_default("TOFY_WORLD_WARMUP_STEPS", "5000");
    set_env_default("TOFY_WORLD_LOG_EVERY", "1000");
    set_env_default("TOFY_CACHE_DIR", CACHE_DIR);
    set_env_default(
        "TOFY_CACHE_PREFETCH_BATCHES",
        if cfg.profile == MemoryProfile::Eighty {
            "24"
        } else if cfg.profile == MemoryProfile::FortyEight {
            "16"
        } else {
            "8"
        },
    );
    set_env_default("TOFY_TOKEN_CACHE_READER_MB", "64");
    set_env_default_owned(
        "TOFY_CACHE_PREFETCH_CHUNK",
        defaults
            .world_batch
            .max(defaults.latent_batch)
            .saturating_mul(2)
            .max(1)
            .to_string(),
    );
    std::env::remove_var("TOFY_USE_TOKEN_CACHE");
    std::env::remove_var("TOFY_ENCODER_VOCAB");
    match cfg.profile {
        MemoryProfile::Minimal => set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "16"),
        MemoryProfile::FortyEight => set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "64"),
        MemoryProfile::Eighty => set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "128"),
    }
    if cfg.resume {
        std::env::set_var("TOFY_RESUME", "1");
    } else {
        std::env::remove_var("TOFY_RESUME");
    }
}

struct ContextDefaults {
    latent_segments: usize,
    world_segments: usize,
    hybrid_exact_tail: usize,
    hybrid_block_size: usize,
    hybrid_retrieval_slots: usize,
    hybrid_exact_old_tokens: usize,
}

fn context_defaults_for_profile(
    profile: MemoryProfile,
    defaults: &ProfileDefaults,
) -> ContextDefaults {
    let (latent_segments, world_segments, retrieval_slots, exact_old_tokens) = match profile {
        MemoryProfile::Minimal => (4, 4, 8, 16),
        MemoryProfile::FortyEight => (6, 6, 12, 24),
        MemoryProfile::Eighty => (8, 8, 16, 32),
    };
    ContextDefaults {
        latent_segments,
        world_segments,
        hybrid_exact_tail: tasks::world_context::default_context_hybrid_exact_tail(
            defaults.world_max_seq,
            1,
        ),
        hybrid_block_size: 32,
        hybrid_retrieval_slots: retrieval_slots,
        hybrid_exact_old_tokens: exact_old_tokens,
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
    let bridge_stage_dir = run_root.join("bridge");
    let eval_stage_dir = run_root.join("eval");
    let latent_model = latent_stage_dir.join("model.safetensors");
    let world_model = world_stage_dir.join("model.safetensors");
    let world_encoder_model = world_stage_dir.join("model.encoder.safetensors");
    let bridge_context_model = bridge_stage_dir.join("context.safetensors");
    let bridge_weights_model = bridge_stage_dir.join("weights.safetensors");
    let encoder_cache_vocab =
        vocab_dir().join(format!("vocab_encoder_{}_default.txt", defaults.max_vocab));

    Ok(PipelinePaths {
        run_id,
        run_root,
        latent_stage_dir,
        world_stage_dir,
        bridge_stage_dir,
        eval_stage_dir,
        latent_model,
        world_model,
        world_encoder_model,
        bridge_context_model,
        bridge_weights_model,
        encoder_cache_vocab,
    })
}

fn prepare_cache_paths(defaults: &ProfileDefaults) -> PipelinePaths {
    let run_root = runs_dir().join("prepare_cache");
    let encoder_cache_vocab =
        vocab_dir().join(format!("vocab_encoder_{}_default.txt", defaults.max_vocab));
    PipelinePaths {
        run_id: "prepare_cache".to_string(),
        latent_stage_dir: run_root.join("latent"),
        world_stage_dir: run_root.join("world"),
        bridge_stage_dir: run_root.join("bridge"),
        eval_stage_dir: run_root.join("eval"),
        latent_model: run_root.join("latent/model.safetensors"),
        world_model: run_root.join("world/model.safetensors"),
        world_encoder_model: run_root.join("world/model.encoder.safetensors"),
        bridge_context_model: run_root.join("bridge/context.safetensors"),
        bridge_weights_model: run_root.join("bridge/weights.safetensors"),
        encoder_cache_vocab,
        run_root,
    }
}

fn prepare_data(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
    resume: bool,
    force_cache: bool,
    _reuse_encoder_data: bool,
    _reuse_world_data: bool,
) -> Result<()> {
    println!("== Stage 1/5: data prep + vocab/token cache ==");
    tasks::veclab::prepare(
        Path::new("."),
        crate::tasks::prepare_veclab::DEFAULT_SEED,
        None,
    )?;
    ensure_nonempty_file(ENCODER_DATA)?;
    ensure_nonempty_file(WORLD_TEXT_DATA)?;
    ensure_nonempty_file(EVAL_SUITE)?;

    let mut cache_args = vec![
        "--prepare-pipeline-cache".to_string(),
        ENCODER_DATA.to_string(),
        WORLD_TEXT_DATA.to_string(),
        paths.encoder_cache_vocab.to_string_lossy().to_string(),
        cache_dir().to_string_lossy().to_string(),
        "--encoder-max-vocab".to_string(),
        defaults.max_vocab.to_string(),
        "--encoder-max-seq".to_string(),
        (defaults.latent_max_seq * 4).to_string(),
        "--world-max-seq".to_string(),
        defaults.world_max_seq.to_string(),
    ];
    if force_cache {
        cache_args.push("--force".to_string());
    }
    add_require_prepared_cache_arg(&mut cache_args);
    run_cache(cache_args)?;
    if !resume {
        std::env::set_var("TOFY_ENCODER_VOCAB", &paths.encoder_cache_vocab);
    }
    Ok(())
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

fn train_encoder(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 2/5: encoder ==");
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
    println!("== Stage 3/5: world knowledge ==");
    if stage_complete(cfg, &paths.world_model, "world", defaults.world_steps)? {
        println!(
            "Skipping world knowledge; resume state already reached {} steps.",
            defaults.world_steps
        );
        return Ok(());
    }
    let vocab = matched_encoder_vocab(paths);
    let args = vec![
        "jepa_ai".to_string(),
        "--train-world-knowledge".to_string(),
        paths.latent_model.to_string_lossy().to_string(),
        vocab.to_string_lossy().to_string(),
        WORLD_TEXT_DATA.to_string(),
        defaults.world_steps.to_string(),
        defaults.world_batch.to_string(),
        defaults.dim.to_string(),
        defaults.world_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--lambda".to_string(),
        "0.09".to_string(),
        "--lr".to_string(),
        "2e-4".to_string(),
        "--grad-accum".to_string(),
        defaults.world_grad_accum.to_string(),
        "--output".to_string(),
        paths.world_model.to_string_lossy().to_string(),
        "--encoder-output".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
    ];
    with_stage("world", || {
        tasks::knowledge::try_run_train(&append_resume(args, cfg.resume))
    })?;
    ensure_file(&paths.world_model)?;
    ensure_file(&paths.world_encoder_model)?;
    Ok(())
}

fn train_bridge(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 4/5: Qwen context and weights bridges ==");
    let qwen_dir = std::env::var("TOFY_QWEN_DIR")
        .context("TOFY_QWEN_DIR must point to the Qwen3-1.7B-Base model directory")?;
    let old_regime = std::env::var_os("TOFY_BRIDGE_REGIME");
    let old_unfreeze = std::env::var_os("TOFY_KNOWLEDGE_UNFREEZE_WORLD");
    let old_bridge_lr = std::env::var_os("TOFY_BRIDGE_LR");
    let old_bridge_grad_accum = std::env::var_os("TOFY_BRIDGE_GRAD_ACCUM");
    let transfer_rows = bridge_transfer_rows(
        &fs::read_to_string(WORLD_TEXT_DATA)?,
        &fs::read_to_string(VECLAB_TASKS)?,
    )?;
    write_text_atomic(Path::new(BRIDGE_TRANSFER_DATA), &transfer_rows)?;
    for (stage, regime, output) in [
        ("bridge_context", "context", &paths.bridge_context_model),
        ("bridge_weights", "weights", &paths.bridge_weights_model),
    ] {
        std::env::set_var("TOFY_BRIDGE_REGIME", regime);
        if old_unfreeze.is_none() {
            std::env::set_var(
                "TOFY_KNOWLEDGE_UNFREEZE_WORLD",
                if regime == "weights" { "true" } else { "false" },
            );
        }
        if old_bridge_lr.is_none() {
            if regime == "weights" {
                std::env::set_var("TOFY_BRIDGE_LR", "1e-4");
            } else {
                std::env::remove_var("TOFY_BRIDGE_LR");
            }
        }
        let (bridge_batch, bridge_grad_accum) = if regime == "weights" {
            (
                env_usize_or("TOFY_WEIGHTS_BRIDGE_BATCH", defaults.bridge_batch).max(1),
                env_usize_or("TOFY_WEIGHTS_BRIDGE_GRAD_ACCUM", defaults.bridge_grad_accum).max(1),
            )
        } else {
            (
                defaults.bridge_batch,
                env_usize_or("TOFY_BRIDGE_GRAD_ACCUM", defaults.bridge_grad_accum).max(1),
            )
        };
        std::env::set_var("TOFY_BRIDGE_GRAD_ACCUM", bridge_grad_accum.to_string());
        if stage_complete(cfg, output, stage, defaults.bridge_steps)? {
            println!(
                "Skipping {stage}; resume state reached {} steps.",
                defaults.bridge_steps
            );
            continue;
        }
        let args = vec![
            "jepa_ai".to_string(),
            "--train-bridge".to_string(),
            qwen_dir.clone(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            BRIDGE_TRANSFER_DATA.to_string(),
            defaults.bridge_steps.to_string(),
            bridge_batch.to_string(),
            output.to_string_lossy().to_string(),
        ];
        let bridge_result = with_stage(stage, || {
            tasks::bridge::try_run_train_bridge(&append_resume(args, cfg.resume))
        });
        if let Err(error) = bridge_result {
            // Context is a causal-channel control. If it has already proven
            // wrong-latent invariance, keep its diagnostic latest checkpoint
            // and continue to the distinct joint world/weights test.
            if regime == "context" && error.to_string().contains("semantic conditioning plateau") {
                println!(
                    "Context bridge stopped for semantic plateau; continuing to weights bridge: {error:#}"
                );
                continue;
            }
            return Err(error);
        }
        ensure_file(output)?;
        if regime == "weights"
            && std::env::var("TOFY_KNOWLEDGE_UNFREEZE_WORLD")
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        {
            ensure_file(output.with_extension("world.safetensors"))?;
        }
    }
    match old_regime {
        Some(value) => std::env::set_var("TOFY_BRIDGE_REGIME", value),
        None => std::env::remove_var("TOFY_BRIDGE_REGIME"),
    }
    match old_unfreeze {
        Some(value) => std::env::set_var("TOFY_KNOWLEDGE_UNFREEZE_WORLD", value),
        None => std::env::remove_var("TOFY_KNOWLEDGE_UNFREEZE_WORLD"),
    }
    match old_bridge_lr {
        Some(value) => std::env::set_var("TOFY_BRIDGE_LR", value),
        None => std::env::remove_var("TOFY_BRIDGE_LR"),
    }
    match old_bridge_grad_accum {
        Some(value) => std::env::set_var("TOFY_BRIDGE_GRAD_ACCUM", value),
        None => std::env::remove_var("TOFY_BRIDGE_GRAD_ACCUM"),
    }
    Ok(())
}

fn eval_ladder_includes(name: &str) -> bool {
    let ladder = std::env::var("TOFY_EVAL_LADDER").unwrap_or_else(|_| "full".into());
    match ladder.as_str() {
        "full" => true,
        "bridge" | "world" | "world_bridge" => {
            matches!(name, "latent_channel" | "knowledge_in_weights")
        }
        other => other.split(',').map(str::trim).any(|part| part == name),
    }
}

fn bridge_transfer_rows(world_rows: &str, task_rows: &str) -> Result<String> {
    let mut output = String::with_capacity(world_rows.len() + task_rows.len());
    for (index, line) in world_rows.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 || fields[..2].iter().any(|field| field.trim().is_empty()) {
            bail!(
                "bridge-transfer world row {} must be state<TAB>documentation<TAB>action",
                index + 1
            );
        }
        if !matches!(
            fields[2].trim().to_ascii_lowercase().as_str(),
            "3" | "fetch_docs"
        ) {
            bail!("bridge-transfer world row {} must use FetchDocs", index + 1);
        }
        output.push_str(fields[0]);
        output.push('\t');
        output.push_str(fields[1]);
        output.push('\n');
    }
    for (index, line) in task_rows.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 || fields.iter().any(|field| field.trim().is_empty()) {
            bail!(
                "bridge-transfer task row {} must be task<TAB>completion",
                index + 1
            );
        }
        output.push_str(line);
        output.push('\n');
    }
    Ok(output)
}

fn final_eval(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    let ladder = std::env::var("TOFY_EVAL_LADDER").unwrap_or_else(|_| "full".into());
    println!("== Stage 5/5: experimental ladder and controls (TOFY_EVAL_LADDER={ladder}) ==");
    let qwen_dir = std::env::var("TOFY_QWEN_DIR")?;
    let old_mode = std::env::var_os("TOFY_EVAL_MODE");
    let old_regime = std::env::var_os("TOFY_BRIDGE_REGIME");
    let old_static = std::env::var_os("TOFY_STATIC_SOFT_PREFIX");
    let old_lora = std::env::var_os("TOFY_QWEN_LORA_RANK");
    std::env::remove_var("TOFY_STATIC_SOFT_PREFIX");
    std::env::remove_var("TOFY_QWEN_LORA_RANK");
    for &(name, mode, regime) in TRAINING_BRIDGE_EVALS {
        let bridge_model = match regime {
            "context" => &paths.bridge_context_model,
            "weights" => &paths.bridge_weights_model,
            _ => unreachable!("training bridge eval has an invalid regime"),
        };
        if !eval_ladder_includes(name) {
            println!("Skipping eval_{name}; not in TOFY_EVAL_LADDER.");
            continue;
        }
        if !bridge_model.exists() {
            println!(
                "Skipping eval_{name}; no qualifying bridge checkpoint at {}.",
                bridge_model.display()
            );
            continue;
        }
        std::env::set_var("TOFY_EVAL_MODE", mode);
        std::env::set_var("TOFY_BRIDGE_REGIME", regime);
        let args = vec![
            "jepa_ai".to_string(),
            "--eval-bridge".to_string(),
            qwen_dir.clone(),
            bridge_model.to_string_lossy().to_string(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            EVAL_SUITE.to_string(),
            paths
                .eval_stage_dir
                .join(format!("{name}.json"))
                .to_string_lossy()
                .to_string(),
        ];
        with_stage(&format!("eval_{name}"), || {
            tasks::eval::try_run_code_eval(&args)
        })?;
    }

    if !eval_ladder_includes("static_prefix")
        && !eval_ladder_includes("lora_r16")
        && !eval_ladder_includes("lora_r512")
    {
        match old_mode {
            Some(value) => std::env::set_var("TOFY_EVAL_MODE", value),
            None => std::env::remove_var("TOFY_EVAL_MODE"),
        }
        match old_regime {
            Some(value) => std::env::set_var("TOFY_BRIDGE_REGIME", value),
            None => std::env::remove_var("TOFY_BRIDGE_REGIME"),
        }
        match old_static {
            Some(value) => std::env::set_var("TOFY_STATIC_SOFT_PREFIX", value),
            None => std::env::remove_var("TOFY_STATIC_SOFT_PREFIX"),
        }
        match old_lora {
            Some(value) => std::env::set_var("TOFY_QWEN_LORA_RANK", value),
            None => std::env::remove_var("TOFY_QWEN_LORA_RANK"),
        }
        return Ok(());
    }

    let lora_data = paths.eval_stage_dir.join("lora_train.txt");
    let lora_rows = bridge_transfer_rows(
        &fs::read_to_string(WORLD_TEXT_DATA)?,
        &fs::read_to_string(VECLAB_TASKS)?,
    )?;
    write_text_atomic(&lora_data, &lora_rows)?;
    for (name, static_prefix, lora_rank) in [
        ("static_prefix", true, None),
        ("lora_r16", false, Some(16)),
        ("lora_r512", false, Some(512)),
    ] {
        if !eval_ladder_includes(name) {
            println!("Skipping {name}; not in TOFY_EVAL_LADDER.");
            continue;
        }
        if static_prefix {
            std::env::set_var("TOFY_STATIC_SOFT_PREFIX", "true");
        } else {
            std::env::remove_var("TOFY_STATIC_SOFT_PREFIX");
        }
        match lora_rank {
            Some(rank) => std::env::set_var("TOFY_QWEN_LORA_RANK", rank.to_string()),
            None => std::env::remove_var("TOFY_QWEN_LORA_RANK"),
        }
        std::env::set_var("TOFY_BRIDGE_REGIME", "weights");
        let model = paths.eval_stage_dir.join(format!("{name}.safetensors"));
        if !stage_complete(cfg, &model, name, defaults.bridge_steps)? {
            let train_args = vec![
                "jepa_ai".to_string(),
                "--train-bridge".to_string(),
                qwen_dir.clone(),
                paths.world_encoder_model.to_string_lossy().to_string(),
                matched_encoder_vocab(paths).to_string_lossy().to_string(),
                paths.world_model.to_string_lossy().to_string(),
                if lora_rank.is_some() {
                    lora_data.to_string_lossy().to_string()
                } else {
                    VECLAB_TASKS.to_string()
                },
                defaults.bridge_steps.to_string(),
                defaults.bridge_batch.to_string(),
                model.to_string_lossy().to_string(),
            ];
            with_stage(name, || {
                tasks::bridge::try_run_train_bridge(&append_resume(train_args, cfg.resume))
            })?;
        }
        std::env::set_var(
            "TOFY_EVAL_MODE",
            if lora_rank.is_some() {
                "unconditioned"
            } else {
                "bridge"
            },
        );
        let eval_args = vec![
            "jepa_ai".to_string(),
            "--eval-bridge".to_string(),
            qwen_dir.clone(),
            model.to_string_lossy().to_string(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            EVAL_SUITE.to_string(),
            paths
                .eval_stage_dir
                .join(format!("{name}.json"))
                .to_string_lossy()
                .to_string(),
        ];
        with_stage(&format!("eval_{name}"), || {
            tasks::eval::try_run_code_eval(&eval_args)
        })?;
    }

    std::env::remove_var("TOFY_STATIC_SOFT_PREFIX");
    std::env::remove_var("TOFY_QWEN_LORA_RANK");
    std::env::set_var("TOFY_BRIDGE_REGIME", "weights");
    if !eval_ladder_includes("channel_probe") {
        match old_mode {
            Some(value) => std::env::set_var("TOFY_EVAL_MODE", value),
            None => std::env::remove_var("TOFY_EVAL_MODE"),
        }
        match old_regime {
            Some(value) => std::env::set_var("TOFY_BRIDGE_REGIME", value),
            None => std::env::remove_var("TOFY_BRIDGE_REGIME"),
        }
        match old_static {
            Some(value) => std::env::set_var("TOFY_STATIC_SOFT_PREFIX", value),
            None => std::env::remove_var("TOFY_STATIC_SOFT_PREFIX"),
        }
        match old_lora {
            Some(value) => std::env::set_var("TOFY_QWEN_LORA_RANK", value),
            None => std::env::remove_var("TOFY_QWEN_LORA_RANK"),
        }
        return Ok(());
    }
    let probe_output = paths.eval_stage_dir.join("channel_probe.safetensors");
    let probe_args = vec![
        "jepa_ai".to_string(),
        "--train-channel-probe".to_string(),
        qwen_dir,
        paths.bridge_weights_model.to_string_lossy().to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        matched_encoder_vocab(paths).to_string_lossy().to_string(),
        paths.world_model.to_string_lossy().to_string(),
        VECLAB_TASKS.to_string(),
        "data/fictional/veclab_tasks_heldout.txt".to_string(),
        probe_output.to_string_lossy().to_string(),
        "1000".to_string(),
    ];
    if !probe_output.exists() {
        with_stage("channel_probe", || tasks::probe::try_run(&probe_args))?;
    }
    match old_mode {
        Some(value) => std::env::set_var("TOFY_EVAL_MODE", value),
        None => std::env::remove_var("TOFY_EVAL_MODE"),
    }
    match old_regime {
        Some(value) => std::env::set_var("TOFY_BRIDGE_REGIME", value),
        None => std::env::remove_var("TOFY_BRIDGE_REGIME"),
    }
    match old_static {
        Some(value) => std::env::set_var("TOFY_STATIC_SOFT_PREFIX", value),
        None => std::env::remove_var("TOFY_STATIC_SOFT_PREFIX"),
    }
    match old_lora {
        Some(value) => std::env::set_var("TOFY_QWEN_LORA_RANK", value),
        None => std::env::remove_var("TOFY_QWEN_LORA_RANK"),
    }
    Ok(())
}

#[allow(dead_code)]
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

fn parse_hf_dataset_repo(value: &str) -> Result<String> {
    let repo = value.trim();
    if repo.is_empty() {
        bail!("Hugging Face dataset id must not be empty");
    }
    if repo.contains(char::is_whitespace) {
        bail!("Hugging Face dataset id must not contain whitespace: {repo:?}");
    }
    let Some((org, name)) = repo.split_once('/') else {
        bail!("Hugging Face dataset id must be org/dataset-name, got {repo:?}");
    };
    if org.is_empty() || name.is_empty() {
        bail!("Hugging Face dataset id must be org/dataset-name, got {repo:?}");
    }
    Ok(repo.to_string())
}

fn upload_prepare_cache_tree(profile: MemoryProfile, repo: &str) -> Result<()> {
    ensure_cache_upload_tools()?;
    let artifact_dir = runs_dir().join("prepare_cache");
    fs::create_dir_all(&artifact_dir)?;

    let git_sha = short_git_sha();
    let timestamp = unix_timestamp()?;
    let base_name = format!("tofy-cache-{}-{}-{}", profile.as_str(), git_sha, timestamp);
    let info_path = artifact_dir.join(format!("{base_name}.info.txt"));

    let upload_roots = prepare_cache_upload_roots()?;
    write_prepare_cache_info(&info_path, profile, repo, &base_name, &upload_roots)?;

    let staging_dir = artifact_dir.join(format!("{base_name}-upload"));
    if staging_dir.exists() {
        fs::remove_dir_all(&staging_dir).with_context(|| {
            format!("remove stale upload staging dir {}", staging_dir.display())
        })?;
    }
    fs::create_dir_all(&staging_dir)?;

    let upload_files = prepare_cache_upload_files(&upload_roots, &staging_dir)?;
    let remote_files = list_hf_dataset_files(repo)?;

    let mut skipped = 0usize;
    println!(
        "Uploading compressed prepared cache tree to Hugging Face dataset {repo}: {} files",
        upload_files.len()
    );
    for file in &upload_files {
        if remote_file_matches(file, &remote_files) {
            skipped += 1;
            println!(
                "Skipping unchanged Hugging Face file: {}",
                file.remote_path.display()
            );
            continue;
        }
        upload_hf_file(repo, &file.local_path, &file.remote_path)?;
    }
    upload_hf_file(
        repo,
        &info_path,
        &PathBuf::from(format!("runs/prepare_cache/{base_name}.info.txt")),
    )?;
    delete_stale_hf_cache_files(repo, &upload_files, &remote_files)?;
    fs::remove_dir_all(&staging_dir)
        .with_context(|| format!("remove upload staging dir {}", staging_dir.display()))?;
    println!(
        "Uploaded compressed prepared cache tree to Hugging Face dataset {repo}: {base_name} (skipped {skipped} unchanged files)"
    );
    Ok(())
}

fn ensure_cache_upload_tools() -> Result<()> {
    if !command_available("hf") {
        bail!(
            "--auto-hf-upload requires `hf` on PATH; install it and authenticate with `hf auth login` if needed"
        );
    }
    if !command_available("pzstd") && !command_available("zstd") {
        bail!("--auto-hf-upload requires `pzstd` or `zstd` on PATH for prepared cache compression");
    }
    Ok(())
}

fn prepare_cache_upload_roots() -> Result<Vec<PathBuf>> {
    let mut inputs = vec![PathBuf::from("data"), PathBuf::from("eval"), vocab_dir()];
    let configured_cache_dir = cache_dir();
    if configured_cache_dir.as_path() != Path::new(CACHE_DIR)
        && !inputs.contains(&configured_cache_dir)
    {
        inputs.push(configured_cache_dir);
    }

    for input in &inputs {
        if !input.exists() {
            bail!(
                "cannot upload prepared cache because {} does not exist",
                input.display()
            );
        }
    }
    Ok(inputs)
}

fn prepare_cache_upload_files(
    roots: &[PathBuf],
    staging_dir: &Path,
) -> Result<Vec<PreparedCacheUploadFile>> {
    let mut files = Vec::new();
    for root in roots {
        collect_prepare_cache_upload_files(root, staging_dir, &mut files)?;
    }
    files.sort_by(|left, right| {
        left.size
            .cmp(&right.size)
            .then_with(|| left.remote_path.cmp(&right.remote_path))
    });
    Ok(files)
}

fn collect_prepare_cache_upload_files(
    path: &Path,
    staging_dir: &Path,
    files: &mut Vec<PreparedCacheUploadFile>,
) -> Result<()> {
    if path.is_file() {
        if should_upload_prepare_cache_file(path) {
            let source_size = fs::metadata(path)
                .with_context(|| format!("stat prepared cache upload file {}", path.display()))?
                .len();
            let (local_path, remote_path) =
                if should_compress_prepare_cache_upload_file(source_size) {
                    let remote_path = PathBuf::from(format!("{}.zst", path.display()));
                    let local_path = staging_dir.join(&remote_path);
                    compress_prepare_cache_file(path, &local_path)?;
                    (local_path, remote_path)
                } else {
                    (path.to_path_buf(), path.to_path_buf())
                };
            let size = fs::metadata(&local_path)
                .with_context(|| {
                    format!("stat prepared cache upload file {}", local_path.display())
                })?
                .len();
            let sha256 = should_hash_prepare_cache_upload_file(&local_path, size)
                .then(|| sha256_file(&local_path))
                .transpose()?;
            files.push(PreparedCacheUploadFile {
                local_path,
                remote_path,
                size,
                sha256,
            });
        }
        return Ok(());
    }
    let mut entries = fs::read_dir(path)
        .with_context(|| format!("read prepared cache upload directory {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read prepared cache upload directory {}", path.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        collect_prepare_cache_upload_files(&entry.path(), staging_dir, files)?;
    }
    Ok(())
}

fn should_upload_prepare_cache_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    !(name.ends_with(".lock")
        || name.ends_with(".incomplete")
        || name.contains(".tmp.")
        || name.starts_with('.'))
}

fn should_hash_prepare_cache_upload_file(path: &Path, size: u64) -> bool {
    size >= 8 * 1024 * 1024 || path.extension().and_then(|ext| ext.to_str()) == Some("bin")
}

fn should_compress_prepare_cache_upload_file(size: u64) -> bool {
    size >= PREPARED_CACHE_COMPRESS_THRESHOLD_BYTES
}

fn compress_prepare_cache_file(input: &Path, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    println!(
        "Compressing prepared cache file {} -> {}",
        input.display(),
        output.display()
    );
    let mut command = if command_available("pzstd") {
        let mut command = Command::new("pzstd");
        command
            .arg("-p")
            .arg(default_parallel_threads().to_string())
            .arg("-1")
            .arg("-f")
            .arg(input)
            .arg("-o")
            .arg(output);
        command
    } else {
        let mut command = Command::new("zstd");
        command
            .arg("-T0")
            .arg("-1")
            .arg("-f")
            .arg(input)
            .arg("-o")
            .arg(output);
        command
    };
    run_external_command(&mut command, "compress prepared cache file")
}

fn default_parallel_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("open {} for sha256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {} for sha256", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn managed_prepare_cache_remote_path(path: &str) -> bool {
    path == "eval"
        || path.starts_with("eval/")
        || path == "data"
        || path.starts_with("data/")
        || path == "local_models/vocabs"
        || path.starts_with("local_models/vocabs/")
}

fn remote_file_matches(
    local: &PreparedCacheUploadFile,
    remote_files: &HashMap<String, RemoteRepoFile>,
) -> bool {
    let remote_path = local.remote_path.to_string_lossy();
    let Some(remote) = remote_files.get(remote_path.as_ref()) else {
        return false;
    };
    if remote.size != local.size {
        return false;
    }
    match (&local.sha256, &remote.oid) {
        (Some(local_hash), Some(remote_oid)) if remote_oid.len() == 64 => local_hash == remote_oid,
        (Some(_), Some(_)) => false,
        (Some(_), None) => false,
        (None, _) => true,
    }
}

fn list_hf_dataset_files(repo: &str) -> Result<HashMap<String, RemoteRepoFile>> {
    let endpoint =
        format!("https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&expand=true");
    let mut curl = Command::new("curl");
    curl.args(["-fsSL", &endpoint]);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            curl.args(["-H", &format!("Authorization: Bearer {token}")]);
        }
    }
    let output = curl
        .output()
        .with_context(|| format!("list Hugging Face dataset files for {repo}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "list Hugging Face dataset files for {repo} failed with status {}\nstderr:\n{}",
            output.status,
            stderr.trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parse Hugging Face dataset file tree for {repo}"))?;
    let entries = value
        .as_array()
        .ok_or_else(|| anyhow!("Hugging Face dataset tree response was not an array"))?;
    let mut files = HashMap::new();
    for entry in entries {
        if entry.get("type").and_then(Value::as_str) != Some("file") {
            continue;
        }
        let Some(path) = entry.get("path").and_then(Value::as_str) else {
            continue;
        };
        let size = entry.get("size").and_then(Value::as_u64).unwrap_or(0);
        let oid = entry.get("oid").and_then(Value::as_str).map(str::to_string);
        files.insert(path.to_string(), RemoteRepoFile { size, oid });
    }
    println!(
        "Scanned Hugging Face dataset {repo}: {} remote files",
        files.len()
    );
    Ok(files)
}

fn delete_stale_hf_cache_files(
    repo: &str,
    upload_files: &[PreparedCacheUploadFile],
    remote_files: &HashMap<String, RemoteRepoFile>,
) -> Result<()> {
    let local_paths = upload_files
        .iter()
        .map(|file| file.remote_path.to_string_lossy().to_string())
        .collect::<HashSet<_>>();
    let mut stale_paths = remote_files
        .keys()
        .filter(|path| managed_prepare_cache_remote_path(path) && !local_paths.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    stale_paths.sort();
    if stale_paths.is_empty() {
        return Ok(());
    }
    println!(
        "Deleting {} stale Hugging Face cache files from {repo}",
        stale_paths.len()
    );
    for path in stale_paths {
        delete_hf_file(repo, &path)?;
    }
    Ok(())
}

fn write_prepare_cache_info(
    path: &Path,
    profile: MemoryProfile,
    repo: &str,
    tree_name: &str,
    upload_inputs: &[PathBuf],
) -> Result<()> {
    let inputs = upload_inputs
        .iter()
        .map(|path| format!("- {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    let content = format!(
        "profile: {}\nrepo: {repo}\nupload_mode: compressed-tree\ntree: {tree_name}\ncompression: zstd level 1 for files >= {} bytes\ncreated_unix_secs: {}\ngit_sha: {}\ncommand: cargo run --release -- prepare cache {} --auto-hf-upload --hf-dataset {repo}\ncontents:\n{inputs}\n",
        profile.as_str(),
        PREPARED_CACHE_COMPRESS_THRESHOLD_BYTES,
        unix_timestamp()?,
        short_git_sha(),
        profile.as_str()
    );
    write_text_atomic(path, &content)
}

fn upload_hf_file(repo: &str, local_path: &Path, remote_path: &Path) -> Result<()> {
    println!(
        "Uploading {} to Hugging Face dataset {repo}: {}",
        local_path.display(),
        remote_path.display()
    );
    run_hf_upload_with_retries(repo, local_path, remote_path)?;
    Ok(())
}

fn delete_hf_file(repo: &str, remote_path: &str) -> Result<()> {
    println!("Deleting stale Hugging Face file from {repo}: {remote_path}");
    let attempts = env_usize_or("TOFY_HF_UPLOAD_RETRIES", 5).max(1);
    let retry_sleep_secs = env_usize_or("TOFY_HF_UPLOAD_RETRY_SLEEP_SECS", 30);
    let mut last_error = None;
    for attempt in 1..=attempts {
        let mut hf = Command::new("hf");
        hf.args([
            "repo-files",
            "delete",
            "--repo-type",
            "dataset",
            repo,
            remote_path,
        ]);
        match run_external_command(
            &mut hf,
            "delete stale prepared cache file from Hugging Face",
        ) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err);
                if attempt < attempts {
                    eprintln!(
                        "Hugging Face delete failed for {remote_path} (attempt {attempt}/{attempts}); retrying in {retry_sleep_secs}s"
                    );
                    std::thread::sleep(std::time::Duration::from_secs(retry_sleep_secs as u64));
                }
            }
        }
    }
    Err(last_error.expect("delete retry loop must record an error"))
}

fn run_hf_upload_with_retries(repo: &str, local_path: &Path, remote_path: &Path) -> Result<()> {
    let attempts = env_usize_or("TOFY_HF_UPLOAD_RETRIES", 5).max(1);
    let retry_sleep_secs = env_usize_or("TOFY_HF_UPLOAD_RETRY_SLEEP_SECS", 30);
    let mut last_error = None;
    for attempt in 1..=attempts {
        let mut hf = Command::new("hf");
        if std::env::var_os("HF_HUB_DISABLE_XET").is_none() {
            hf.env("HF_HUB_DISABLE_XET", "1");
        }
        hf.args(["upload", "--repo-type", "dataset", repo])
            .arg(local_path)
            .arg(remote_path);
        match run_external_command(&mut hf, "upload prepared cache file to Hugging Face") {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err);
                if attempt < attempts {
                    eprintln!(
                        "Hugging Face upload failed for {} (attempt {attempt}/{attempts}); retrying in {retry_sleep_secs}s",
                        local_path.display()
                    );
                    std::thread::sleep(std::time::Duration::from_secs(retry_sleep_secs as u64));
                }
            }
        }
    }
    Err(last_error.expect("upload retry loop must record an error"))
}

fn short_git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "nogit".to_string())
}

fn run_external_command(command: &mut Command, label: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to run {label}"))?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            println!("{stdout}");
        }
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!(
        "{label} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout.trim(),
        stderr.trim()
    )
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

fn env_usize_or(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn stage_complete(
    cfg: &PipelineConfig,
    model_path: &Path,
    stage: &str,
    target_steps: usize,
) -> Result<bool> {
    if !cfg.resume {
        return Ok(false);
    }
    if skip_trained_stage(cfg, stage) {
        ensure_file(model_path)?;
        println!(
            "Skipping {stage}; --skip-trained accepted existing model {}.",
            model_path.display()
        );
        return Ok(true);
    }
    if !model_path.exists() {
        return Ok(false);
    }
    let state_path = util::checkpoint_sidecar_path(model_path, stage, "resume.json");
    let Some(state) = util::load_resume_state(&state_path, stage)? else {
        return Ok(false);
    };
    Ok(state.step >= target_steps)
}

fn parse_stage_list_env(key: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .map(|value| parse_stage_list(&value))
        .unwrap_or_default()
}

fn parse_stage_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|stage| !stage.is_empty())
        .map(canonical_skip_stage)
        .collect()
}

fn canonical_skip_stage(stage: &str) -> String {
    match stage {
        "encoder" => "latent".to_string(),
        other => other.to_string(),
    }
}

fn skip_trained_stage(cfg: &PipelineConfig, stage: &str) -> bool {
    let stage = canonical_skip_stage(stage);
    cfg.skip_trained_stages
        .iter()
        .any(|skip| skip == &stage || (skip == "bridge" && stage.starts_with("bridge_")))
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

fn write_launch(paths: &PipelinePaths, cfg: &PipelineConfig) -> Result<()> {
    let selector = cfg.resume_selector.as_deref().unwrap_or("");
    let command = format!(
        "train {} --until {}{}",
        cfg.profile.as_str(),
        cfg.until.as_str(),
        if selector.is_empty() {
            String::new()
        } else {
            format!(" --resume {selector}")
        },
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
        pipeline_kind: "knowledge",
        resume_enabled: cfg.resume,
        resume_selector: selector,
        run_root: paths.run_root.to_string_lossy().to_string(),
        latent_model: paths.latent_model.to_string_lossy().to_string(),
        world_model: paths.world_model.to_string_lossy().to_string(),
        world_encoder_model: paths.world_encoder_model.to_string_lossy().to_string(),
        bridge_context_model: paths.bridge_context_model.to_string_lossy().to_string(),
        bridge_weights_model: paths.bridge_weights_model.to_string_lossy().to_string(),
        encoder_data: ENCODER_DATA,
        world_data: WORLD_TEXT_DATA,
        eval_suite: EVAL_SUITE,
        profile: cfg.profile.as_str(),
        pipeline_until: cfg.until.as_str(),
        with_code_eval: cfg.with_code_eval,
        latent_steps: defaults.latent_steps,
        world_steps: defaults.world_steps,
        bridge_steps: defaults.bridge_steps,
        latent_batch: defaults.latent_batch,
        world_batch: defaults.world_batch,
        bridge_batch: defaults.bridge_batch,
        latent_grad_accum: defaults.latent_grad_accum,
        world_grad_accum: defaults.world_grad_accum,
        bridge_grad_accum: defaults.bridge_grad_accum,
        dim: defaults.dim,
        layers: defaults.layers,
        heads: defaults.heads,
        bridge_dim: defaults.bridge_dim,
        num_latent_tokens: defaults.num_latent_tokens,
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

fn prepared_cache_required() -> bool {
    std::env::var("TOFY_REQUIRE_PREPARED_CACHE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes"
        })
        .unwrap_or(false)
}

fn add_require_prepared_cache_arg(args: &mut Vec<String>) {
    if prepared_cache_required() {
        args.push("--require-hit".to_string());
    }
}

fn vocab_dir() -> PathBuf {
    std::env::var("TOFY_VOCAB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("local_models/vocabs"))
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn current_profile_schema_parses_minimal() -> Result<()> {
        let profiles: ModelProfiles =
            serde_json::from_str(include_str!("../../config/model_profiles.json"))?;
        assert_eq!(profiles.minimal.bridge_dim, 640);
        assert_eq!(profiles.minimal.bridge_max_seq, 256);
        assert!(profiles.minimal.bridge_grad_accum > 0);
        Ok(())
    }

    #[test]
    fn training_never_schedules_the_base_decoder_floor() {
        assert!(TRAINING_BRIDGE_EVALS
            .iter()
            .all(|(name, mode, _)| *name != "floor" && *mode != "floor"));
    }

    #[test]
    fn bridge_transfer_rows_normalize_world_and_task_schemas() -> Result<()> {
        let rows = bridge_transfer_rows(
            "[fn:001] query\tdocumentation\tfetch_docs\n",
            "[fn:001] task\tpackage solution\\nfunc Solve() {}\n",
        )?;
        let parsed = rows
            .lines()
            .map(|line| line.split('\t').collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(parsed.len(), 2);
        assert!(parsed.iter().all(|fields| fields.len() == 2));
        assert_eq!(parsed[0], vec!["[fn:001] query", "documentation"]);
        Ok(())
    }
}
