use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::{data, tasks, util};

const WORLD_TEXT_DATA: &str = "data/ultrachat_pairs.txt";
const WORLD_DATA: &str = "data/world_mix_pairs.txt";
const GO_CODE_DATA: &str = "data/go_code_pairs.txt";
const WIKI_DATA: &str = "data/cached_wikimedia_wikipedia_1.txt";
const ENCODER_DATA: &str = "data/encoder_mix.txt";
const EVAL_SUITE: &str = "eval/code_assistant_go_hard.jsonl";
const GO_TASK_DATA: &str = "data/go_instruction_pairs.txt";
const GO_ALGORITHM_TASK_DATA: &str = "data/go_algorithm_pairs.txt";
const GO_SEMANTIC_TASK_DATA: &str = "data/go_semantic_pairs.txt";
const GO_REPAIR_DATA: &str = "data/go_repair_pairs.txt";
const GO_MODEL_FAILURE_REPAIR_DATA: &str = "data/go_model_failure_repair_pairs.txt";
const GO_MODEL_PREFERENCE_DATA: &str = "data/go_model_preference_pairs.jsonl";
const GO_PASS_SELF_TRAIN_DATA: &str = "data/go_pass_self_train_pairs.txt";
const GO_FEEDBACK_TRAIN_DATA: &str = "data/code_poc_go_mix.txt";
const CODE_TRAIN_DATA: &str = "data/code_poc_mix.txt";
const CACHE_DIR: &str = "data/cache";
const MODEL_PROFILES_PATH: &str = "config/model_profiles.json";
const PREPARED_CACHE_COMPRESS_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoryProfile {
    Eight,
    FortyEight,
    Eighty,
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
    high_world_batch: usize,
    code_decoder_batch: usize,
    go_feedback_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    high_world_grad_accum: usize,
    code_decoder_grad_accum: usize,
    go_feedback_grad_accum: usize,
}

#[derive(Deserialize)]
struct ModelProfiles {
    #[serde(rename = "8gb")]
    eight_gb: ProfileDefaults,
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
    build_conditioned_cache: bool,
    from_conditioned_cache: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PipelineUntil {
    Decoder,
    DecoderCache,
    Full,
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
    pipeline_until: &'a str,
    with_code_eval: bool,
    build_conditioned_cache: bool,
    from_conditioned_cache: bool,
    latent_steps: usize,
    world_steps: usize,
    high_world_steps: usize,
    code_decoder_steps: usize,
    go_feedback_steps: usize,
    latent_batch: usize,
    world_batch: usize,
    high_world_batch: usize,
    code_decoder_batch: usize,
    go_feedback_batch: usize,
    latent_grad_accum: usize,
    world_grad_accum: usize,
    high_world_grad_accum: usize,
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
            "usage: prepare cache <8gb|48gb|80gb> [--force] [--auto-hf-upload --hf-dataset <org/dataset-name>]"
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
            "8gb" => Ok(Self::Eight),
            "48gb" => Ok(Self::FortyEight),
            "80gb" => Ok(Self::Eighty),
            other => bail!("unsupported train profile '{other}' (expected 8gb, 48gb, or 80gb)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Eight => "8gb",
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
            Self::Eight => profiles.eight_gb,
            Self::FortyEight => profiles.forty_eight_gb,
            Self::Eighty => profiles.eighty_gb,
        })
    }
}

impl PipelineUntil {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "decoder" => Ok(Self::Decoder),
            "decoder-cache" => Ok(Self::DecoderCache),
            "full" => Ok(Self::Full),
            other => bail!(
                "unsupported --until value '{other}' (expected decoder, decoder-cache, or full)"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Decoder => "decoder",
            Self::DecoderCache => "decoder-cache",
            Self::Full => "full",
        }
    }
}

impl PipelineConfig {
    fn from_args(args: &[String]) -> Result<Self> {
        let profile_arg = args.first().ok_or_else(|| {
            anyhow::anyhow!(
                "usage: train <8gb|48gb|80gb> [--until decoder|decoder-cache|full] [--resume [latest|run]] [--skip-trained STAGE[,STAGE...]] [--with-code-eval] [--build-conditioned-cache] [--from-conditioned-cache]"
            )
        })?;
        let profile = MemoryProfile::parse(profile_arg)?;
        let mut until = PipelineUntil::Full;
        let mut resume = false;
        let mut resume_selector = None;
        let mut skip_trained_stages = parse_stage_list_env("TOFY_SKIP_TRAINED_STAGES");
        let mut with_code_eval = true;
        let mut build_conditioned_cache = false;
        let mut from_conditioned_cache = false;
        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--until" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--until requires decoder|decoder-cache|full"))?;
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
                "--build-conditioned-cache" => {
                    build_conditioned_cache = true;
                    i += 1;
                }
                "--from-conditioned-cache" => {
                    from_conditioned_cache = true;
                    i += 1;
                }
                other => bail!(
                    "unsupported train argument '{other}' (accepted: --until, --resume, --skip-trained, --with-code-eval, --build-conditioned-cache, --from-conditioned-cache)"
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
            build_conditioned_cache,
            from_conditioned_cache,
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
        build_conditioned_cache: false,
        from_conditioned_cache: false,
    };
    set_pipeline_env(&cfg, &defaults);
    if !command_available("go") {
        bail!(
            "prepare cache requires `go` on PATH to build go_repair_pairs and dependent mixes before Hugging Face upload"
        );
    }
    println!(
        "Preparing full local data/cache handoff for {} profile; no model training will run.",
        profile.as_str()
    );
    prepare_data(&prepare_cache_paths(&defaults), &defaults, false, force)?;
    prepare_go_feedback_decoder_token_cache(&prepare_cache_paths(&defaults), &defaults)?;
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
    let context_defaults = context_defaults_for_profile(cfg.profile, &defaults);
    println!(
        "Context: latent={} tokens, world/runtime={} tokens, decoder_prompt={} tokens (hybrid memory)",
        defaults.latent_max_seq * context_defaults.latent_segments,
        defaults.world_max_seq * context_defaults.world_segments,
        context_defaults.decoder_prompt_tokens
    );

    prepare_data(&paths, &defaults, cfg.resume, false)?;
    configure_encoder_vocab_env(&paths, &cfg)?;
    train_encoder(&paths, &cfg, &defaults)?;
    train_world(&paths, &cfg, &defaults)?;
    train_high_world(&paths, &cfg, &defaults)?;
    train_code_decoder(&paths, &cfg, &defaults)?;
    if cfg.until == PipelineUntil::Decoder {
        println!("Pipeline stopped after base decoder (--until decoder).");
        return Ok(());
    }
    let should_build_conditioned_cache =
        cfg.until == PipelineUntil::DecoderCache || cfg.build_conditioned_cache;
    if cfg.from_conditioned_cache && !should_build_conditioned_cache {
        println!(
            "Skipping Go feedback data rebuild because --from-conditioned-cache requires the existing cache/data pair."
        );
    } else {
        prepare_go_model_feedback_data(&paths, &cfg, &defaults)?;
    }
    if !cfg.from_conditioned_cache || should_build_conditioned_cache {
        prepare_go_feedback_decoder_token_cache(&paths, &defaults)?;
    }
    if cfg.until == PipelineUntil::DecoderCache || cfg.build_conditioned_cache {
        build_go_feedback_conditioned_cache(&paths, &cfg, &defaults)?;
        if cfg.until == PipelineUntil::DecoderCache {
            println!("Pipeline stopped after conditioned decoder cache (--until decoder-cache).");
            return Ok(());
        }
    }
    train_go_feedback_decoder(&paths, &cfg, &defaults)?;
    let selected_decoder = select_decoder_checkpoint(&paths, &defaults)?;
    let mut paths = paths;
    paths.code_decoder_model = selected_decoder;
    write_meta(&paths, &cfg, &defaults)?;
    final_eval(&paths, &defaults)?;

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
    set_env_default("TOFY_RECURSIVE_CONTEXT_COMPRESSION", "0");
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
        context_defaults.decoder_prompt_tokens.to_string(),
    );
    set_env_default_owned(
        "TOFY_DECODER_LOCAL_WINDOW",
        context_defaults.decoder_local_window.to_string(),
    );
    set_env_default("TOFY_DECODER_CSA_COMPRESS_RATE", "8");
    set_env_default("TOFY_DECODER_HCA_COMPRESS_RATE", "128");
    set_env_default("TOFY_DECODER_ANCHOR_PERIOD", "3");
    set_env_default("TOFY_DECODER_CSA_TOPK", "16");
    set_env_default("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", "2");
    set_env_default("TOFY_WORLD_ROLLOUT_STEPS", "2");
    set_env_default("TOFY_WORLD_INVERSE_LOSS_WEIGHT", "0.2");
    set_env_default("TOFY_WORLD_TRANS_COSINE_WEIGHT", "0.1");
    set_env_default("TOFY_WORLD_SIGREG_PRED_WEIGHT", "0.6");
    set_env_default("TOFY_ACTION_FOCAL_GAMMA", "2.0");
    set_env_default("TOFY_LABEL_SMOOTHING", "0.05");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_ROWS", "500000");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_BYTES", "67108864");
    set_env_default("TOFY_BPE_MAX_MERGES", "24000");
    set_env_default("TOFY_BPE_PROGRESS_EVERY_MERGES", "128");
    set_env_default("TOFY_CODE_VOCAB_SAMPLE_ROWS", "100000");
    set_env_default("TOFY_CODE_VOCAB_SAMPLE_BYTES", "67108864");
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
    set_env_default("TOFY_ORCHESTRATOR_LOG_EVERY", "500");
    set_env_default("TOFY_DECODER_NEGATIVE_FORWARDS", "1");
    set_env_default("TOFY_DECODER_ABLATION_METRICS", "1");
    set_env_default("TOFY_DECODER_PROMPT_DROPOUT", "0.12");
    set_env_default("TOFY_DECODER_SYNTAX_LOSS_WEIGHT", "0.05");
    set_env_default("TOFY_DECODER_SIGNATURE_LOSS_WEIGHT", "0.15");
    set_env_default("TOFY_DECODER_STRUCTURE_LOSS_WEIGHT", "0.05");
    set_env_default("TOFY_DECODER_CONDITIONING_MARGIN", "0.10");
    set_env_default("TOFY_DECODER_CONDITIONING_NEGATIVES", "zero,shuffle");
    set_env_default("TOFY_DECODER_CHECKPOINT_EVERY", "1000");
    set_env_default_owned(
        "TOFY_DECODER_ATTENTION_QUERY_BLOCK",
        context_defaults
            .decoder_local_window
            .clamp(64, 256)
            .to_string(),
    );
    set_env_default("TOFY_HWM_MACRO_MIN_LEN", "2");
    set_env_default("TOFY_HWM_MACRO_MAX_LEN", "4");
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
            .max(defaults.code_decoder_batch)
            .saturating_mul(2)
            .max(1)
            .to_string(),
    );
    std::env::remove_var("TOFY_USE_TOKEN_CACHE");
    std::env::remove_var("TOFY_ENCODER_VOCAB");
    match cfg.profile {
        MemoryProfile::Eight => set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "16"),
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
    decoder_prompt_tokens: usize,
    decoder_local_window: usize,
}

fn context_defaults_for_profile(
    profile: MemoryProfile,
    defaults: &ProfileDefaults,
) -> ContextDefaults {
    let (latent_segments, world_segments, retrieval_slots, exact_old_tokens) = match profile {
        MemoryProfile::Eight => (4, 4, 8, 16),
        MemoryProfile::FortyEight => (6, 6, 12, 24),
        MemoryProfile::Eighty => (8, 8, 16, 32),
    };
    ContextDefaults {
        latent_segments,
        world_segments,
        hybrid_exact_tail: tasks::world::default_context_hybrid_exact_tail(
            defaults.world_max_seq,
            1,
        ),
        hybrid_block_size: 32,
        hybrid_retrieval_slots: retrieval_slots,
        hybrid_exact_old_tokens: exact_old_tokens,
        decoder_prompt_tokens: defaults.code_decoder_max_seq.saturating_mul(4).max(768),
        decoder_local_window: defaults.code_decoder_max_seq.clamp(128, 256),
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

fn prepare_cache_paths(defaults: &ProfileDefaults) -> PipelinePaths {
    let run_root = runs_dir().join("prepare_cache");
    let encoder_cache_vocab =
        vocab_dir().join(format!("vocab_encoder_{}_default.txt", defaults.max_vocab));
    let code_decoder_cache_vocab = vocab_dir().join(format!(
        "vocab_code_{}_codeaware.txt",
        defaults.code_decoder_max_vocab
    ));
    PipelinePaths {
        run_id: "prepare_cache".to_string(),
        latent_stage_dir: run_root.join("latent"),
        world_stage_dir: run_root.join("world"),
        high_world_stage_dir: run_root.join("high_world"),
        decoder_stage_dir: run_root.join("decoder_code"),
        decoder_go_feedback_stage_dir: run_root.join("decoder_code_go_feedback"),
        code_eval_stage_dir: run_root.join("code_eval"),
        latent_model: run_root.join("latent/model.safetensors"),
        world_model: run_root.join("world/model.safetensors"),
        world_encoder_model: run_root.join("world/model.encoder.safetensors"),
        high_world_model: run_root.join("high_world/model.safetensors"),
        code_decoder_base_model: run_root.join("decoder_code/model.safetensors"),
        code_decoder_go_feedback_model: run_root.join("decoder_code_go_feedback/model.safetensors"),
        code_decoder_model: run_root.join("decoder_code/model.safetensors"),
        code_decoder_vocab: code_decoder_cache_vocab.clone(),
        encoder_cache_vocab,
        code_decoder_cache_vocab,
        run_root,
    }
}

fn prepare_data(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
    resume: bool,
    force_cache: bool,
) -> Result<()> {
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
                    "4",
                    "--timeout-sec",
                    "5.0",
                    "--max-rows",
                    "60000",
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

        join_result(encoder_corpus_handle, "encoder corpus")?;

        let (world_args, code_mix_args, go_feedback_mix_args) = build_stage1_mix_args();
        run_stage1_mix_jobs(vec![
            PrepareMixJob::new("world mix", world_args, 2_048, 4_096),
            PrepareMixJob::new("code decoder mix", code_mix_args, 1_024, 3_072),
            PrepareMixJob::new(
                "go feedback decoder mix",
                go_feedback_mix_args,
                1_024,
                3_072,
            ),
        ])?;
        Ok(())
    })?;

    let mut cache_args = vec![
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
    ];
    if force_cache {
        cache_args.push("--force".to_string());
    }
    add_require_prepared_cache_arg(&mut cache_args);
    run_cache(cache_args)?;
    if prepared_cache_required() {
        prepare_go_feedback_decoder_token_cache(paths, defaults)?;
    }
    if paths.code_decoder_cache_vocab.exists() && !paths.code_decoder_vocab.exists() {
        if let Some(parent) = paths.code_decoder_vocab.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&paths.code_decoder_cache_vocab, &paths.code_decoder_vocab)?;
    }
    if paths.code_decoder_vocab.exists() {
        tasks::world::ensure_code_decoder_vocab_manifest(
            &paths.code_decoder_vocab,
            defaults.code_decoder_max_vocab,
        )?;
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
    run_prepare([
        "--prepare-go-algorithm-tasks",
        "--output",
        GO_ALGORITHM_TASK_DATA,
    ])?;
    run_prepare([
        "--prepare-go-semantics-tasks",
        "--output",
        GO_SEMANTIC_TASK_DATA,
    ])?;
    Ok(())
}

fn build_stage1_mix_args() -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut extra_code_pairs = Vec::new();
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
    if nonempty_file(GO_ALGORITHM_TASK_DATA) {
        world_args.extend([
            "--code-pairs".to_string(),
            GO_ALGORITHM_TASK_DATA.to_string(),
        ]);
        extra_code_pairs.push(GO_ALGORITHM_TASK_DATA.to_string());
    }
    if nonempty_file(GO_SEMANTIC_TASK_DATA) {
        world_args.extend([
            "--code-pairs".to_string(),
            GO_SEMANTIC_TASK_DATA.to_string(),
        ]);
    }
    if nonempty_file(GO_REPAIR_DATA) {
        world_args.extend(["--code-pairs".to_string(), GO_REPAIR_DATA.to_string()]);
        extra_code_pairs.push(GO_REPAIR_DATA.to_string());
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
        "--base-repeat".to_string(),
        "1".to_string(),
        "--instruction-pairs".to_string(),
        GO_TASK_DATA.to_string(),
        "--instruction-repeat".to_string(),
        "18".to_string(),
        "--fim-repeat".to_string(),
        "4".to_string(),
    ];
    for path in &extra_code_pairs {
        code_mix_args.extend(["--extra-pairs".to_string(), path.clone()]);
    }
    if !extra_code_pairs.is_empty() {
        code_mix_args.extend(["--extra-repeat".to_string(), "24".to_string()]);
    }
    code_mix_args.extend(["--max-rows".to_string(), "0".to_string()]);
    let mut go_feedback_mix_args = vec![
        "--prepare-code-poc-mix".to_string(),
        "--output".to_string(),
        GO_FEEDBACK_TRAIN_DATA.to_string(),
        "--base-pairs".to_string(),
        GO_CODE_DATA.to_string(),
        "--base-repeat".to_string(),
        "1".to_string(),
        "--instruction-pairs".to_string(),
        GO_TASK_DATA.to_string(),
        "--instruction-repeat".to_string(),
        "20".to_string(),
        "--fim-repeat".to_string(),
        "6".to_string(),
    ];
    if nonempty_file(GO_ALGORITHM_TASK_DATA) {
        go_feedback_mix_args.extend([
            "--extra-pairs".to_string(),
            GO_ALGORITHM_TASK_DATA.to_string(),
        ]);
    }
    if nonempty_file(GO_REPAIR_DATA) {
        go_feedback_mix_args.extend(["--extra-pairs".to_string(), GO_REPAIR_DATA.to_string()]);
    }
    if nonempty_file(GO_MODEL_FAILURE_REPAIR_DATA) {
        go_feedback_mix_args.extend([
            "--extra-pairs".to_string(),
            GO_MODEL_FAILURE_REPAIR_DATA.to_string(),
        ]);
    }
    if nonempty_file(GO_PASS_SELF_TRAIN_DATA) {
        go_feedback_mix_args.extend([
            "--extra-pairs".to_string(),
            GO_PASS_SELF_TRAIN_DATA.to_string(),
        ]);
    }
    if nonempty_file(GO_ALGORITHM_TASK_DATA)
        || nonempty_file(GO_REPAIR_DATA)
        || nonempty_file(GO_MODEL_FAILURE_REPAIR_DATA)
        || nonempty_file(GO_PASS_SELF_TRAIN_DATA)
    {
        go_feedback_mix_args.extend(["--extra-repeat".to_string(), "32".to_string()]);
    }
    go_feedback_mix_args.extend(["--max-rows".to_string(), "0".to_string()]);
    (world_args, code_mix_args, go_feedback_mix_args)
}

struct PrepareMixJob {
    label: &'static str,
    args: Vec<String>,
    estimated_mb: u64,
}

struct PrepareMixJobResult {
    label: &'static str,
    estimated_mb: u64,
    result: Result<()>,
}

#[derive(Clone, Copy)]
struct PrepareMixMemory {
    mem_available_mb: u64,
    swap_free_mb: u64,
}

impl PrepareMixJob {
    fn new(label: &'static str, args: Vec<String>, base_mb: u64, max_mb: u64) -> Self {
        let input_mb = prepare_mix_input_mb(&args);
        let estimated_mb = base_mb.saturating_add(input_mb / 16).clamp(base_mb, max_mb);
        Self {
            label,
            args,
            estimated_mb,
        }
    }
}

fn run_stage1_mix_jobs(jobs: Vec<PrepareMixJob>) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let max_parallel = prepare_mix_max_parallel(jobs.len());
    let initial_budget = prepare_mix_memory_budget_mb();
    let memory_note = prepare_mix_memory_snapshot()
        .map(|snapshot| {
            format!(
                "available={}MB swap_free={}MB",
                snapshot.mem_available_mb, snapshot.swap_free_mb
            )
        })
        .unwrap_or_else(|| "available=unknown swap_free=unknown".to_string());
    println!(
        "Stage 1 mix scheduler: jobs={} max_parallel={} {} budget={}MB",
        jobs.len(),
        max_parallel,
        memory_note,
        display_memory_budget(initial_budget)
    );

    let mut pending = VecDeque::from(jobs);
    let (tx, rx) = std::sync::mpsc::channel::<PrepareMixJobResult>();
    let mut active_jobs = 0usize;
    let mut active_estimated_mb = 0u64;
    let mut first_error: Option<anyhow::Error> = None;

    while !pending.is_empty() || active_jobs > 0 {
        while first_error.is_none() && active_jobs < max_parallel && !pending.is_empty() {
            let budget_mb = prepare_mix_memory_budget_mb();
            let Some(index) =
                select_prepare_mix_job(&pending, active_estimated_mb, budget_mb, active_jobs == 0)
            else {
                break;
            };
            let job = pending
                .remove(index)
                .expect("pending job index should exist");
            let label = job.label;
            let estimated_mb = job.estimated_mb;
            let args = job.args;
            println!(
                "Stage 1 mix scheduler: starting {label} (estimate={}MB active_estimate={}MB budget={}MB)",
                estimated_mb,
                active_estimated_mb.saturating_add(estimated_mb),
                display_memory_budget(budget_mb)
            );
            active_jobs += 1;
            active_estimated_mb = active_estimated_mb.saturating_add(estimated_mb);
            let tx = tx.clone();
            thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_prepare_vec(args).with_context(|| label)
                }))
                .map_err(|_| anyhow!("{label} panicked"))
                .and_then(|result| result);
                let _ = tx.send(PrepareMixJobResult {
                    label,
                    estimated_mb,
                    result,
                });
            });
        }

        if active_jobs == 0 {
            if let Some(error) = first_error {
                return Err(error);
            }
            bail!("stage 1 mix scheduler could not schedule any pending job");
        }

        let finished = rx.recv().context("stage 1 mix scheduler channel closed")?;
        active_jobs -= 1;
        active_estimated_mb = active_estimated_mb.saturating_sub(finished.estimated_mb);
        match finished.result {
            Ok(()) => println!("Stage 1 mix scheduler: finished {}", finished.label),
            Err(error) => {
                println!("Stage 1 mix scheduler: {} failed", finished.label);
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }

    if let Some(error) = first_error {
        Err(error)
    } else {
        Ok(())
    }
}

fn select_prepare_mix_job(
    pending: &VecDeque<PrepareMixJob>,
    active_estimated_mb: u64,
    budget_mb: u64,
    force_one: bool,
) -> Option<usize> {
    pending
        .iter()
        .position(|job| active_estimated_mb.saturating_add(job.estimated_mb) <= budget_mb)
        .or_else(|| force_one.then_some(0))
}

fn prepare_mix_max_parallel(job_count: usize) -> usize {
    let hardware_threads = thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
        .max(1);
    let default_parallel = job_count.min(hardware_threads).max(1);
    std::env::var("TOFY_PREPARE_MIX_MAX_PARALLEL")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_parallel)
        .min(job_count)
        .max(1)
}

fn prepare_mix_input_mb(args: &[String]) -> u64 {
    let mut total = 0u64;
    let mut index = 0usize;
    while index + 1 < args.len() {
        let flag = args[index].as_str();
        if matches!(
            flag,
            "--text-pairs"
                | "--code-pairs"
                | "--base-pairs"
                | "--instruction-pairs"
                | "--extra-pairs"
        ) {
            total = total.saturating_add(path_len_mb(Path::new(&args[index + 1])));
            index += 2;
        } else {
            index += 1;
        }
    }
    total
}

fn path_len_mb(path: &Path) -> u64 {
    const MB: u64 = 1_048_576;
    path.metadata()
        .map(|metadata| metadata.len().saturating_add(MB - 1) / MB)
        .unwrap_or(0)
}

fn prepare_mix_memory_budget_mb() -> u64 {
    let Some(snapshot) = prepare_mix_memory_snapshot() else {
        return u64::MAX;
    };
    let headroom_mb = std::env::var("TOFY_PREPARE_MIX_HEADROOM_MB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(4_096);
    let swap_divisor = std::env::var("TOFY_PREPARE_MIX_SWAP_DIVISOR")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4);
    snapshot
        .mem_available_mb
        .saturating_add(snapshot.swap_free_mb / swap_divisor)
        .saturating_sub(headroom_mb)
}

fn prepare_mix_memory_snapshot() -> Option<PrepareMixMemory> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let mut mem_available_mb = None;
    let mut swap_free_mb = None;
    for line in text.lines() {
        if line.starts_with("MemAvailable:") {
            mem_available_mb = meminfo_line_mb(line);
        } else if line.starts_with("SwapFree:") {
            swap_free_mb = meminfo_line_mb(line);
        }
    }
    Some(PrepareMixMemory {
        mem_available_mb: mem_available_mb?,
        swap_free_mb: swap_free_mb.unwrap_or(0),
    })
}

fn meminfo_line_mb(line: &str) -> Option<u64> {
    line.split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kb| kb / 1024)
}

fn display_memory_budget(budget_mb: u64) -> String {
    if budget_mb == u64::MAX {
        "unknown".to_string()
    } else {
        budget_mb.to_string()
    }
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
        "1.0".to_string(),
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
        defaults.high_world_batch.to_string(),
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
        defaults.high_world_grad_accum.to_string(),
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
        "1e-4",
        "0.10",
    );
    with_base_decoder_pretrain_env(|| {
        with_stage("decoder_code", || {
            tasks::world::try_run_train_decoder(&append_resume(args, cfg.resume))
        })
    })?;
    ensure_file(&paths.code_decoder_base_model)?;
    Ok(())
}

fn prepare_go_model_feedback_data(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    let default_rows = match cfg.profile {
        MemoryProfile::Eight => 1024,
        MemoryProfile::FortyEight => 2048,
        MemoryProfile::Eighty => 4096,
    };
    let max_rows = env_usize_or("TOFY_GO_MODEL_FEEDBACK_ROWS", default_rows);
    if max_rows == 0 {
        println!("Skipping model-failure Go feedback mining; TOFY_GO_MODEL_FEEDBACK_ROWS=0.");
        return Ok(());
    }
    if !command_available("go") {
        println!("Skipping model-failure Go feedback mining; go not found.");
        return Ok(());
    }
    println!("== Stage 5a/6: model-failure Go feedback mining ==");
    let candidates = env_usize_or("TOFY_GO_MODEL_FEEDBACK_CANDIDATES", 1).max(1);
    let repair_rounds = env_usize_or("TOFY_GO_MODEL_FEEDBACK_REPAIR_ROUNDS", 2);
    let max_new_tokens = env_usize_or("TOFY_GO_MODEL_FEEDBACK_MAX_TOKENS", 256).max(1);
    let workers = env_usize_or("TOFY_GO_MODEL_FEEDBACK_WORKERS", 0);
    let pass_min_compile_rate = std::env::var("TOFY_GO_MODEL_FEEDBACK_PASS_MIN_COMPILE_RATE")
        .unwrap_or_else(|_| "0.10".to_string());
    let mut args = vec![
        "jepa_ai".to_string(),
        "--prepare-go-model-feedback-pairs".to_string(),
        paths.world_encoder_model.to_string_lossy().to_string(),
        matched_encoder_vocab(paths).to_string_lossy().to_string(),
        paths.world_model.to_string_lossy().to_string(),
        GO_TASK_DATA.to_string(),
        GO_MODEL_FAILURE_REPAIR_DATA.to_string(),
        max_new_tokens.to_string(),
        defaults.dim.to_string(),
        defaults.code_decoder_max_seq.to_string(),
        defaults.layers.to_string(),
        defaults.heads.to_string(),
        defaults.bridge_dim.to_string(),
        defaults.num_latent_tokens.to_string(),
        "--high-world-model".to_string(),
        paths.high_world_model.to_string_lossy().to_string(),
        "--code-decoder".to_string(),
        paths.code_decoder_base_model.to_string_lossy().to_string(),
        "--code-decoder-vocab".to_string(),
        paths.code_decoder_vocab.to_string_lossy().to_string(),
        "--preference-output".to_string(),
        GO_MODEL_PREFERENCE_DATA.to_string(),
        "--pass-output".to_string(),
        GO_PASS_SELF_TRAIN_DATA.to_string(),
        "--candidates".to_string(),
        candidates.to_string(),
        "--repair-rounds".to_string(),
        repair_rounds.to_string(),
        "--max-rows".to_string(),
        max_rows.to_string(),
        "--go-timeout-sec".to_string(),
        "6".to_string(),
        "--pass-min-compile-rate".to_string(),
        pass_min_compile_rate,
    ];
    if workers > 0 {
        args.extend(["--workers".to_string(), workers.to_string()]);
    }
    if !cfg.resume {
        args.push("--force".to_string());
    }
    with_stage("go_model_feedback", || {
        tasks::eval::try_run_prepare_go_model_feedback_pairs(&args)
    })?;

    let (_, _, go_feedback_mix_args) = build_stage1_mix_args();
    run_prepare_vec(go_feedback_mix_args)?;
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
    let mut args = decoder_args(
        paths,
        defaults,
        GO_FEEDBACK_TRAIN_DATA,
        defaults.go_feedback_steps,
        defaults.go_feedback_batch,
        defaults.go_feedback_grad_accum,
        &paths.code_decoder_go_feedback_model,
        Some(&paths.code_decoder_base_model),
        "5e-5",
        "0.20",
    );
    if cfg.from_conditioned_cache || cfg.build_conditioned_cache {
        args.push("--from-conditioned-cache".to_string());
    }
    with_go_feedback_cache_dir(|| {
        with_stage("decoder_code_go_feedback", || {
            tasks::world::try_run_train_decoder(&append_resume(args, cfg.resume))
        })
    })?;
    ensure_file(&paths.code_decoder_go_feedback_model)?;
    Ok(())
}

fn prepare_go_feedback_decoder_token_cache(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 5a-cache/6: Go feedback decoder token cache ==");
    let mut args = vec![
        "--prepare-pipeline-cache".to_string(),
        ENCODER_DATA.to_string(),
        WORLD_DATA.to_string(),
        GO_FEEDBACK_TRAIN_DATA.to_string(),
        paths.encoder_cache_vocab.to_string_lossy().to_string(),
        paths.code_decoder_cache_vocab.to_string_lossy().to_string(),
        go_feedback_cache_dir().to_string_lossy().to_string(),
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
    ];
    add_require_prepared_cache_arg(&mut args);
    run_cache(args)
}

fn build_go_feedback_conditioned_cache(
    paths: &PipelinePaths,
    _cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 5b-cache/6: Go feedback frozen conditioning cache ==");
    let mut args = decoder_args(
        paths,
        defaults,
        GO_FEEDBACK_TRAIN_DATA,
        0,
        defaults.go_feedback_batch,
        defaults.go_feedback_grad_accum,
        &paths.code_decoder_go_feedback_model,
        Some(&paths.code_decoder_base_model),
        "5e-5",
        "0.20",
    );
    args.push("--build-conditioned-cache".to_string());
    with_go_feedback_cache_dir(|| {
        with_stage("decoder_code_go_feedback_cache", || {
            tasks::world::try_run_train_decoder(&args)
        })
    })?;
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
        "--conditioning-negative-forwards".to_string(),
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
        defaults.code_decoder_max_seq.to_string(),
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
        "8".to_string(),
        "--repair-attempts".to_string(),
        "4".to_string(),
        "--pi-agent-env".to_string(),
        "--go-timeout-sec".to_string(),
        "10".to_string(),
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
        "encoder" => "latent",
        "decoder" => "decoder_code",
        "go-feedback" | "go_feedback" | "feedback_decoder" => "decoder_code_go_feedback",
        other => other,
    }
    .to_string()
}

fn skip_trained_stage(cfg: &PipelineConfig, stage: &str) -> bool {
    let stage = canonical_skip_stage(stage);
    cfg.skip_trained_stages.iter().any(|skip| skip == &stage)
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
    let command = format!(
        "train {} --until {}{}{}{}",
        cfg.profile.as_str(),
        cfg.until.as_str(),
        if selector.is_empty() {
            String::new()
        } else {
            format!(" --resume {selector}")
        },
        if cfg.build_conditioned_cache {
            " --build-conditioned-cache"
        } else {
            ""
        },
        if cfg.from_conditioned_cache {
            " --from-conditioned-cache"
        } else {
            ""
        }
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
        pipeline_until: cfg.until.as_str(),
        with_code_eval: cfg.with_code_eval,
        build_conditioned_cache: cfg.build_conditioned_cache,
        from_conditioned_cache: cfg.from_conditioned_cache,
        latent_steps: defaults.latent_steps,
        world_steps: defaults.world_steps,
        high_world_steps: defaults.high_world_steps,
        code_decoder_steps: defaults.code_decoder_steps,
        go_feedback_steps: defaults.go_feedback_steps,
        latent_batch: defaults.latent_batch,
        world_batch: defaults.world_batch,
        high_world_batch: defaults.high_world_batch,
        code_decoder_batch: defaults.code_decoder_batch,
        go_feedback_batch: defaults.go_feedback_batch,
        latent_grad_accum: defaults.latent_grad_accum,
        world_grad_accum: defaults.world_grad_accum,
        high_world_grad_accum: defaults.high_world_grad_accum,
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

fn go_feedback_cache_dir() -> PathBuf {
    cache_dir().join("go_feedback")
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

fn with_go_feedback_cache_dir<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let key = "TOFY_CACHE_DIR";
    let previous = std::env::var_os(key);
    std::env::set_var(key, go_feedback_cache_dir());
    let result = f();
    if let Some(previous) = previous {
        std::env::set_var(key, previous);
    } else {
        std::env::remove_var(key);
    }
    result
}

fn with_env_defaults<F>(defaults: &[(&str, &str)], f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let mut added = Vec::new();
    for (key, value) in defaults {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
            added.push(*key);
        }
    }
    let result = f();
    for key in added {
        std::env::remove_var(key);
    }
    result
}

fn with_base_decoder_pretrain_env<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    with_env_defaults(&[("TOFY_DECODER_CLIP_NORM", "0.30")], f)
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
