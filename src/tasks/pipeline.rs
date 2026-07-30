use anyhow::{anyhow, bail, Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
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
    ("latent_channel", "bridge", "context"),
    ("knowledge_in_weights", "bridge", "weights"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoryProfile {
    Minimal,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize)]
struct ScientificRates {
    tasks: usize,
    passed: usize,
    suite_pass_rate: f64,
}

#[derive(Debug, Deserialize)]
struct ScientificTaskResult {
    id: String,
    subset: String,
    condition: String,
    category: String,
}

#[derive(Debug, Deserialize)]
struct ScientificReport {
    schema_version: u32,
    arm: String,
    suite_sha256: String,
    selected_task_ids: Vec<String>,
    results: BTreeMap<String, BTreeMap<String, ScientificRates>>,
    #[serde(default)]
    task_results: Vec<ScientificTaskResult>,
}

#[derive(Debug)]
struct ValidatedCondition {
    tasks: usize,
    passed: usize,
    outcomes: BTreeMap<String, bool>,
}

impl ValidatedCondition {
    fn suite_pass_rate(&self) -> f64 {
        self.passed as f64 / self.tasks.max(1) as f64
    }
}

#[derive(Debug)]
struct ExpectedSuite {
    sha256: String,
    all_ids: BTreeSet<String>,
    heldout_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FileIdentity {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ImmutableRunIdentity {
    training_env: BTreeMap<String, String>,
    training_env_sha256: String,
    qwen_files: BTreeMap<String, FileIdentity>,
    qwen_sha256: String,
    prepared_inputs: BTreeMap<String, FileIdentity>,
    prepared_inputs_sha256: String,
}

#[derive(Serialize)]
struct PipelineMeta<'a> {
    schema_version: u32,
    git_commit: &'a str,
    source_sha256: &'a str,
    profile_defaults_sha256: &'a str,
    immutable_identity: &'a ImmutableRunIdentity,
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

#[derive(Debug)]
struct ResumeMismatch {
    key: String,
    recorded: Value,
    current: Value,
}

#[derive(Debug)]
struct ResumeValidation {
    mismatches: Vec<ResumeMismatch>,
    forced: bool,
}

impl ResumeValidation {
    fn identity_matches(&self) -> bool {
        self.mismatches.is_empty()
    }

    fn enforce(&self, meta_path: &Path) -> Result<()> {
        if self.identity_matches() || self.forced {
            return Ok(());
        }
        let keys = self
            .mismatches
            .iter()
            .map(|mismatch| mismatch.key.as_str())
            .collect::<Vec<_>>();
        bail!(
            "resume metadata does not match the current code/profile/training inputs; keys={keys:?}, metadata={}. Start a new run or set TOFY_ALLOW_RESUME_MISMATCH=1 only after auditing compatibility.",
            meta_path.display()
        )
    }
}

struct RunRootLock {
    _file: fs::File,
}

impl RunRootLock {
    fn acquire(run_root: &Path) -> Result<Self> {
        let path = run_root.join(".pipeline.lock");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open pipeline lock {}", path.display()))?;
        file.try_lock_exclusive().with_context(|| {
            format!(
                "another pipeline process already owns run {}; refusing concurrent mutation",
                run_root.display()
            )
        })?;
        file.set_len(0)?;
        writeln!(
            file,
            "pid={} acquired_unix={}",
            std::process::id(),
            unix_timestamp()?
        )?;
        file.sync_data()?;
        Ok(Self { _file: file })
    }
}

struct PreparedInputsLock {
    _file: fs::File,
}

impl PreparedInputsLock {
    fn acquire() -> Result<Self> {
        // The prepared datasets and evaluation harness are repository-global,
        // regardless of where a caller places its run outputs.
        let directory = PathBuf::from("data");
        fs::create_dir_all(&directory)?;
        let path = directory.join(".tofy-prepared-inputs.lock");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open prepared-input lock {}", path.display()))?;
        file.try_lock_exclusive().with_context(|| {
            format!(
                "another pipeline is preparing or using the shared input tree; lock={}",
                path.display()
            )
        })?;
        file.set_len(0)?;
        writeln!(
            file,
            "pid={} acquired_unix={}",
            std::process::id(),
            unix_timestamp()?
        )?;
        file.sync_data()?;
        Ok(Self { _file: file })
    }
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
            "usage: prepare cache minimal [--force] [--auto-hf-upload --hf-dataset <org/dataset-name>]"
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
            other => bail!("unsupported train profile '{other}' (expected minimal)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
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
        Ok(profiles.minimal)
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
                "usage: train minimal [--until full] [--resume [latest|run]] [--skip-trained STAGE[,STAGE...]] [--with-code-eval]"
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
    let _prepared_inputs_lock = PreparedInputsLock::acquire()?;
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
    // Held for the lifetime of this parent pipeline process. Stage subprocesses
    // invoke their task entrypoints directly and therefore do not reacquire it.
    let _run_lock = RunRootLock::acquire(&paths.run_root)?;
    // Preparation writes repository-global datasets, vocabularies, and caches.
    // Keep that tree stable from preparation through the last training/eval
    // child so a different run cannot invalidate the recorded input identity.
    let _prepared_inputs_lock = PreparedInputsLock::acquire()?;
    for dir in [
        &paths.latent_stage_dir,
        &paths.world_stage_dir,
        &paths.bridge_stage_dir,
        &paths.eval_stage_dir,
    ] {
        fs::create_dir_all(dir)?;
    }
    let git_commit = current_git_commit()?;
    let source_sha256 = source_fingerprint()?;
    let profile_defaults_sha256 = sha256_bytes(&serde_json::to_vec(&defaults)?);
    if !cfg.resume {
        write_launch(&paths, &cfg)?;
    }
    std::env::set_var("TOFY_RUN_GROUP", &paths.run_id);

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
    // Validate only after deterministic preparation/restoration has completed,
    // and while the shared-input lock prevents another run changing the tree.
    let immutable_identity = immutable_run_identity(&paths, &defaults)?;
    let historical_identity_matches = if cfg.resume {
        let validation = validate_resume_meta(
            &paths,
            &cfg,
            &defaults,
            &git_commit,
            &source_sha256,
            &profile_defaults_sha256,
            &immutable_identity,
        )?;
        write_resume_launch(
            &paths,
            &cfg,
            &git_commit,
            &source_sha256,
            &profile_defaults_sha256,
            &immutable_identity,
            &validation,
        )?;
        let identity_matches = validation.identity_matches();
        validation.enforce(&paths.run_root.join("meta.json"))?;
        identity_matches
    } else {
        write_meta(
            &paths,
            &cfg,
            &defaults,
            &git_commit,
            &source_sha256,
            &profile_defaults_sha256,
            &immutable_identity,
        )?;
        true
    };
    reconcile_adaptive_batch_file(&paths)?;
    preflight_rag_ceiling(&paths)?;
    train_encoder(&paths, &cfg, &defaults)?;
    train_world(&paths, &cfg, &defaults)?;
    train_bridge(&paths, &cfg, &defaults, historical_identity_matches)?;
    if cfg.with_code_eval {
        final_eval(&paths, &cfg, &defaults)?;
    }

    println!("Pipeline complete.");
    Ok(())
}

fn restore_env_var(name: &str, value: Option<std::ffi::OsString>) {
    match value {
        Some(value) => std::env::set_var(name, value),
        None => std::env::remove_var(name),
    }
}

struct EnvRestoreGuard(Vec<(String, Option<std::ffi::OsString>)>);

impl EnvRestoreGuard {
    fn capture(names: &[&str]) -> Self {
        Self(
            names
                .iter()
                .map(|name| ((*name).to_string(), std::env::var_os(name)))
                .collect(),
        )
    }
}

impl Drop for EnvRestoreGuard {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            restore_env_var(&name, value);
        }
    }
}

/// Prove that the frozen decoder can use explicit documentation before
/// spending accelerator-days learning a latent interface to that decoder.
fn preflight_rag_ceiling(paths: &PipelinePaths) -> Result<()> {
    let _environment_guard = EnvRestoreGuard::capture(&[
        "TOFY_EVAL_MODE",
        "TOFY_BRIDGE_REGIME",
        "TOFY_STATIC_SOFT_PREFIX",
        "TOFY_QWEN_LORA_RANK",
        "TOFY_EVAL_MIN_PASS_RATE",
        "TOFY_EVAL_MAX_TASKS",
        "TOFY_EVAL_TASK_OFFSET",
        "TOFY_EVAL_ARM",
    ]);
    let required = std::env::var("TOFY_REQUIRE_RAG_CEILING")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    if !required {
        println!("Skipping mandatory RAG ceiling because TOFY_REQUIRE_RAG_CEILING=0.");
        return Ok(());
    }
    println!("== Frozen-decoder preflight: direct-documentation RAG ceiling ==");
    let qwen_dir = std::env::var("TOFY_QWEN_DIR")
        .context("TOFY_QWEN_DIR is required for the RAG ceiling preflight")?;
    let old_mode = std::env::var_os("TOFY_EVAL_MODE");
    let old_regime = std::env::var_os("TOFY_BRIDGE_REGIME");
    let old_static = std::env::var_os("TOFY_STATIC_SOFT_PREFIX");
    let old_lora = std::env::var_os("TOFY_QWEN_LORA_RANK");
    let old_floor = std::env::var_os("TOFY_EVAL_MIN_PASS_RATE");
    let old_limit = std::env::var_os("TOFY_EVAL_MAX_TASKS");
    let old_offset = std::env::var_os("TOFY_EVAL_TASK_OFFSET");
    let old_arm = std::env::var_os("TOFY_EVAL_ARM");

    std::env::set_var("TOFY_EVAL_MODE", "rag");
    std::env::set_var("TOFY_BRIDGE_REGIME", "weights");
    std::env::remove_var("TOFY_STATIC_SOFT_PREFIX");
    std::env::remove_var("TOFY_QWEN_LORA_RANK");
    std::env::set_var("TOFY_EVAL_MAX_TASKS", "300");

    let mut result = Ok(());
    for (split, offset) in [("seen", "0"), ("heldout", "300")] {
        std::env::set_var("TOFY_EVAL_ARM", format!("rag_preflight_{split}"));
        let minimum = std::env::var("TOFY_RAG_MIN_PASS_RATE").unwrap_or_else(|_| {
            let (split_key, default) = if split == "seen" {
                ("TOFY_RAG_MIN_SEEN_PASS_RATE", "0.30")
            } else {
                ("TOFY_RAG_MIN_HELDOUT_PASS_RATE", "0.40")
            };
            std::env::var(split_key).unwrap_or_else(|_| default.into())
        });
        let minimum = parse_probability(&minimum)
            .with_context(|| format!("invalid RAG minimum for {split} split"))?;
        std::env::set_var("TOFY_EVAL_MIN_PASS_RATE", minimum.to_string());
        std::env::set_var("TOFY_EVAL_TASK_OFFSET", offset);
        let report_path = paths
            .eval_stage_dir
            .join(format!("rag_preflight_{split}.json"));
        remove_file_if_exists(&report_path)?;
        let args = vec![
            "jepa_ai".to_string(),
            "--eval-bridge".to_string(),
            qwen_dir.clone(),
            paths
                .eval_stage_dir
                .join("rag_preflight_untrained_bridge.safetensors")
                .to_string_lossy()
                .to_string(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            EVAL_SUITE.to_string(),
            report_path.to_string_lossy().to_string(),
        ];
        if let Err(error) = run_required_stage_command(&format!("rag_preflight_{split}"), &args) {
            result = Err(error);
            break;
        }
    }

    restore_env_var("TOFY_EVAL_MODE", old_mode);
    restore_env_var("TOFY_BRIDGE_REGIME", old_regime);
    restore_env_var("TOFY_STATIC_SOFT_PREFIX", old_static);
    restore_env_var("TOFY_QWEN_LORA_RANK", old_lora);
    restore_env_var("TOFY_EVAL_MIN_PASS_RATE", old_floor);
    restore_env_var("TOFY_EVAL_MAX_TASKS", old_limit);
    restore_env_var("TOFY_EVAL_TASK_OFFSET", old_offset);
    restore_env_var("TOFY_EVAL_ARM", old_arm);
    result?;
    println!("RAG ceiling passed on seen and held-out tasks; bridge training is permitted.");
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
    set_env_default("TOFY_SIGREG_SLICE_CHUNK", "128");
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
    set_env_default("TOFY_LATENT_EARLY_STOP_PATIENCE", "3000");
    set_env_default("TOFY_LATENT_EARLY_STOP_WARMUP", "2000");
    set_env_default("TOFY_WORLD_EARLY_STOP_PATIENCE", "3000");
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
    set_env_default("TOFY_LABEL_SMOOTHING", "0.05");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_ROWS", "500000");
    set_env_default("TOFY_ENCODER_VOCAB_SAMPLE_BYTES", "67108864");
    set_env_default("TOFY_BPE_MAX_MERGES", "24000");
    set_env_default_owned(
        "TOFY_LATENT_WARMUP_BATCH",
        defaults.latent_warmup_batch.to_string(),
    );
    set_env_default_owned(
        "TOFY_LATENT_WARMUP_GRAD_ACCUM",
        defaults.latent_grad_accum.to_string(),
    );
    set_env_default_owned(
        "TOFY_WORLD_WARMUP_BATCH",
        defaults.world_warmup_batch.to_string(),
    );
    set_env_default_owned(
        "TOFY_WORLD_WARMUP_GRAD_ACCUM",
        defaults.world_grad_accum.to_string(),
    );
    set_env_default("TOFY_WORLD_WARMUP_STEPS", "5000");
    set_env_default("TOFY_WORLD_LOG_EVERY", "1000");
    set_env_default("TOFY_CACHE_DIR", CACHE_DIR);
    set_env_default("TOFY_CACHE_PREFETCH_BATCHES", "8");
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
    set_env_default("TOFY_CONTEXT_SEGMENT_BATCH", "16");
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
    _profile: MemoryProfile,
    defaults: &ProfileDefaults,
) -> ContextDefaults {
    let (latent_segments, world_segments, retrieval_slots, exact_old_tokens) = (4, 4, 8, 16);
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
    if stage_complete(
        cfg,
        &paths.latent_model,
        "latent",
        defaults.latent_steps,
        &[],
    )? {
        println!(
            "Skipping encoder; resume state already reached {} steps.",
            defaults.latent_steps
        );
        return Ok(());
    }
    let _warmup_environment_guard =
        EnvRestoreGuard::capture(&["TOFY_LATENT_WARMUP_BATCH", "TOFY_LATENT_WARMUP_GRAD_ACCUM"]);
    let resume_state_path =
        util::checkpoint_sidecar_path(&paths.latent_model, "latent", "resume.json");
    let initial_warmup_batch =
        env_usize_or("TOFY_LATENT_WARMUP_BATCH", defaults.latent_warmup_batch).max(1);
    let initial_warmup_grad_accum =
        env_usize_or("TOFY_LATENT_WARMUP_GRAD_ACCUM", defaults.latent_grad_accum)
            .max(1)
            .min(defaults.latent_grad_accum);
    let outcome = run_training_stage_with_oom_recovery(
        paths,
        cfg,
        "latent",
        defaults.latent_batch,
        defaults.latent_grad_accum,
        &resume_state_path,
        None,
        |physical_batch, grad_accum| {
            vec![
                "jepa_ai".to_string(),
                "--latent".to_string(),
                ENCODER_DATA.to_string(),
                defaults.latent_steps.to_string(),
                physical_batch.to_string(),
                defaults.dim.to_string(),
                defaults.latent_max_seq.to_string(),
                defaults.layers.to_string(),
                defaults.heads.to_string(),
                defaults.max_vocab.to_string(),
                "--grad-accum".to_string(),
                grad_accum.to_string(),
                "--output".to_string(),
                paths.latent_model.to_string_lossy().to_string(),
            ]
        },
        |physical_batch, _grad_accum| {
            let reductions = batch_reductions(defaults.latent_batch, physical_batch)?;
            let (warmup_batch, warmup_grad_accum) = batch_pair_after_reductions(
                initial_warmup_batch,
                initial_warmup_grad_accum,
                reductions,
            )?;
            std::env::set_var("TOFY_LATENT_WARMUP_BATCH", warmup_batch.to_string());
            std::env::set_var(
                "TOFY_LATENT_WARMUP_GRAD_ACCUM",
                warmup_grad_accum.to_string(),
            );
            Ok(())
        },
    )?;
    require_stage_success("latent", outcome)?;
    ensure_nonempty_file(&paths.latent_model)?;
    Ok(())
}

fn train_world(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
) -> Result<()> {
    println!("== Stage 3/5: world knowledge ==");
    if stage_complete(
        cfg,
        &paths.world_model,
        "world",
        defaults.world_steps,
        &[paths.world_encoder_model.as_path()],
    )? {
        println!(
            "Skipping world knowledge; resume state already reached {} steps.",
            defaults.world_steps
        );
        return Ok(());
    }
    let _warmup_environment_guard =
        EnvRestoreGuard::capture(&["TOFY_WORLD_WARMUP_BATCH", "TOFY_WORLD_WARMUP_GRAD_ACCUM"]);
    let vocab = matched_encoder_vocab(paths);
    let resume_state_path =
        util::checkpoint_sidecar_path(&paths.world_model, "world", "resume.json");
    let initial_warmup_batch =
        env_usize_or("TOFY_WORLD_WARMUP_BATCH", defaults.world_warmup_batch).max(1);
    let initial_warmup_grad_accum =
        env_usize_or("TOFY_WORLD_WARMUP_GRAD_ACCUM", defaults.world_grad_accum)
            .max(1)
            .min(defaults.world_grad_accum);
    let outcome = run_training_stage_with_oom_recovery(
        paths,
        cfg,
        "world",
        defaults.world_batch,
        defaults.world_grad_accum,
        &resume_state_path,
        None,
        |physical_batch, grad_accum| {
            vec![
                "jepa_ai".to_string(),
                "--train-world-knowledge".to_string(),
                paths.latent_model.to_string_lossy().to_string(),
                vocab.to_string_lossy().to_string(),
                WORLD_TEXT_DATA.to_string(),
                defaults.world_steps.to_string(),
                physical_batch.to_string(),
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
                grad_accum.to_string(),
                "--output".to_string(),
                paths.world_model.to_string_lossy().to_string(),
                "--encoder-output".to_string(),
                paths.world_encoder_model.to_string_lossy().to_string(),
            ]
        },
        |physical_batch, _grad_accum| {
            let reductions = batch_reductions(defaults.world_batch, physical_batch)?;
            let (warmup_batch, warmup_grad_accum) = batch_pair_after_reductions(
                initial_warmup_batch,
                initial_warmup_grad_accum,
                reductions,
            )?;
            std::env::set_var("TOFY_WORLD_WARMUP_BATCH", warmup_batch.to_string());
            std::env::set_var(
                "TOFY_WORLD_WARMUP_GRAD_ACCUM",
                warmup_grad_accum.to_string(),
            );
            Ok(())
        },
    )?;
    require_stage_success("world", outcome)?;
    ensure_nonempty_file(&paths.world_model)?;
    ensure_nonempty_file(&paths.world_encoder_model)?;
    Ok(())
}

fn train_bridge(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
    historical_identity_matches: bool,
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
        let status_path = paths.bridge_stage_dir.join(format!("{stage}.outcome.json"));
        std::env::set_var("TOFY_BRIDGE_REGIME", regime);
        if old_unfreeze.is_none() {
            // The world model is the knowledge source, not disposable bridge
            // state. Decoder CE must not rewrite it without replaying the
            // original prediction + SIGReg objective.
            std::env::set_var("TOFY_KNOWLEDGE_UNFREEZE_WORLD", "false");
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
        if regime == "context"
            && cfg.resume
            && historical_identity_matches
            && !output.exists()
            && read_bridge_nonqualification(&status_path, stage, None)?.is_some()
        {
            println!(
                "Skipping {stage}; the prior complete attempt recorded a non-qualifying causal control."
            );
            continue;
        }
        let unfreeze_world = env_truthy("TOFY_KNOWLEDGE_UNFREEZE_WORLD");
        let world_sidecar = output.with_extension("world.safetensors");
        let additional_artifacts = if unfreeze_world {
            vec![world_sidecar.as_path()]
        } else {
            Vec::new()
        };
        if stage_complete(
            cfg,
            output,
            stage,
            defaults.bridge_steps,
            &additional_artifacts,
        )? {
            println!(
                "Skipping {stage}; resume state reached {} steps.",
                defaults.bridge_steps
            );
            continue;
        }
        let resume_state_path = util::checkpoint_sidecar_path(output, stage, "resume.json");
        let outcome = run_training_stage_with_oom_recovery(
            paths,
            cfg,
            stage,
            bridge_batch,
            bridge_grad_accum,
            &resume_state_path,
            Some(&status_path),
            |physical_batch, _grad_accum| {
                vec![
                    "jepa_ai".to_string(),
                    "--train-bridge".to_string(),
                    qwen_dir.clone(),
                    paths.world_encoder_model.to_string_lossy().to_string(),
                    matched_encoder_vocab(paths).to_string_lossy().to_string(),
                    paths.world_model.to_string_lossy().to_string(),
                    BRIDGE_TRANSFER_DATA.to_string(),
                    defaults.bridge_steps.to_string(),
                    physical_batch.to_string(),
                    output.to_string_lossy().to_string(),
                ]
            },
            |_physical_batch, grad_accum| {
                std::env::set_var("TOFY_BRIDGE_GRAD_ACCUM", grad_accum.to_string());
                Ok(())
            },
        )?;
        match outcome {
            StageCommandOutcome::Success => {}
            StageCommandOutcome::BridgeNonQualified(status) if regime == "context" => {
                println!(
                    "Context bridge did not qualify ({:?} at step {}); continuing to the weights bridge.",
                    status.reason, status.step
                );
                continue;
            }
            StageCommandOutcome::BridgeNonQualified(status) => bail!(
                "weights bridge did not qualify ({:?} at step {}); outcome={}",
                status.reason,
                status.step,
                status_path.display()
            ),
            StageCommandOutcome::CudaOutOfMemory(status) => bail!(
                "{stage} exhausted CUDA memory without adaptive recovery: {}",
                status.error
            ),
        }
        ensure_nonempty_file(output)?;
        if unfreeze_world {
            ensure_nonempty_file(&world_sidecar)?;
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
    let _environment_guard = EnvRestoreGuard::capture(&[
        "TOFY_EVAL_MODE",
        "TOFY_BRIDGE_REGIME",
        "TOFY_STATIC_SOFT_PREFIX",
        "TOFY_QWEN_LORA_RANK",
        "TOFY_EVAL_MIN_PASS_RATE",
        "TOFY_EVAL_MIN_CAUSAL_ADVANTAGE",
        "TOFY_EVAL_MAX_CAUSAL_P_VALUE",
        "TOFY_EVAL_MAX_TASKS",
        "TOFY_EVAL_TASK_OFFSET",
        "TOFY_EVAL_ARM",
        "TOFY_BRIDGE_GRAD_ACCUM",
    ]);
    let ladder = std::env::var("TOFY_EVAL_LADDER").unwrap_or_else(|_| "full".into());
    println!("== Stage 5/5: experimental ladder and controls (TOFY_EVAL_LADDER={ladder}) ==");
    let qwen_dir = std::env::var("TOFY_QWEN_DIR")?;
    let baseline_bridge_grad_accum =
        env_usize_or("TOFY_BRIDGE_GRAD_ACCUM", defaults.bridge_grad_accum).max(1);
    let old_mode = std::env::var_os("TOFY_EVAL_MODE");
    let old_regime = std::env::var_os("TOFY_BRIDGE_REGIME");
    let old_static = std::env::var_os("TOFY_STATIC_SOFT_PREFIX");
    let old_lora = std::env::var_os("TOFY_QWEN_LORA_RANK");
    let old_min_pass = std::env::var_os("TOFY_EVAL_MIN_PASS_RATE");
    let old_min_advantage = std::env::var_os("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE");
    let old_max_causal_p = std::env::var_os("TOFY_EVAL_MAX_CAUSAL_P_VALUE");
    let old_limit = std::env::var_os("TOFY_EVAL_MAX_TASKS");
    let old_offset = std::env::var_os("TOFY_EVAL_TASK_OFFSET");
    let old_arm = std::env::var_os("TOFY_EVAL_ARM");
    let mut evaluation_failures = Vec::new();
    std::env::remove_var("TOFY_STATIC_SOFT_PREFIX");
    std::env::remove_var("TOFY_QWEN_LORA_RANK");
    std::env::remove_var("TOFY_EVAL_MIN_PASS_RATE");
    std::env::remove_var("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE");
    std::env::remove_var("TOFY_EVAL_MAX_CAUSAL_P_VALUE");
    if ladder == "full" {
        std::env::remove_var("TOFY_EVAL_MAX_TASKS");
        std::env::remove_var("TOFY_EVAL_TASK_OFFSET");
    }
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
        let report_path = paths.eval_stage_dir.join(format!("{name}.json"));
        remove_file_if_exists(&report_path)?;
        if !bridge_model.exists() {
            println!(
                "Skipping eval_{name}; no qualifying bridge checkpoint at {}.",
                bridge_model.display()
            );
            continue;
        }
        std::env::set_var("TOFY_EVAL_MODE", mode);
        std::env::set_var("TOFY_BRIDGE_REGIME", regime);
        std::env::set_var("TOFY_EVAL_ARM", name);
        let args = vec![
            "jepa_ai".to_string(),
            "--eval-bridge".to_string(),
            qwen_dir.clone(),
            bridge_model.to_string_lossy().to_string(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            EVAL_SUITE.to_string(),
            report_path.to_string_lossy().to_string(),
        ];
        if let Err(error) = run_required_stage_command(&format!("eval_{name}"), &args) {
            evaluation_failures.push(format!("eval_{name} failed: {error:#}"));
        }
    }
    std::env::remove_var("TOFY_EVAL_MIN_PASS_RATE");
    std::env::remove_var("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE");

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
        restore_env_var("TOFY_EVAL_MIN_PASS_RATE", old_min_pass);
        restore_env_var("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE", old_min_advantage);
        restore_env_var("TOFY_EVAL_MAX_CAUSAL_P_VALUE", old_max_causal_p);
        restore_env_var("TOFY_EVAL_MAX_TASKS", old_limit);
        restore_env_var("TOFY_EVAL_TASK_OFFSET", old_offset);
        restore_env_var("TOFY_EVAL_ARM", old_arm);
        return finish_diagnostic_ladder(&ladder, evaluation_failures);
    }

    let lora_data = paths.eval_stage_dir.join("lora_train.txt");
    let lora_data_ready = match (|| -> Result<()> {
        let lora_rows = bridge_transfer_rows(
            &fs::read_to_string(WORLD_TEXT_DATA)?,
            &fs::read_to_string(VECLAB_TASKS)?,
        )?;
        write_text_atomic(&lora_data, &lora_rows)
    })() {
        Ok(()) => true,
        Err(error) => {
            evaluation_failures.push(format!("prepare LoRA training data failed: {error:#}"));
            false
        }
    };
    for (name, static_prefix, lora_rank) in [
        ("static_prefix", true, None),
        ("lora_r16", false, Some(16)),
        ("lora_r512", false, Some(512)),
    ] {
        if !eval_ladder_includes(name) {
            println!("Skipping {name}; not in TOFY_EVAL_LADDER.");
            continue;
        }
        let report_path = paths.eval_stage_dir.join(format!("{name}.json"));
        remove_file_if_exists(&report_path)?;
        if lora_rank.is_some() && !lora_data_ready {
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
        let stage_is_complete = match stage_complete(cfg, &model, name, defaults.bridge_steps, &[])
        {
            Ok(complete) => complete,
            Err(error) => {
                evaluation_failures.push(format!("{name} resume validation failed: {error:#}"));
                continue;
            }
        };
        if !stage_is_complete {
            let resume_state_path = util::checkpoint_sidecar_path(&model, name, "resume.json");
            let outcome = run_training_stage_with_oom_recovery(
                paths,
                cfg,
                name,
                defaults.bridge_batch,
                baseline_bridge_grad_accum,
                &resume_state_path,
                None,
                |physical_batch, _grad_accum| {
                    vec![
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
                        physical_batch.to_string(),
                        model.to_string_lossy().to_string(),
                    ]
                },
                |_physical_batch, grad_accum| {
                    std::env::set_var("TOFY_BRIDGE_GRAD_ACCUM", grad_accum.to_string());
                    Ok(())
                },
            );
            if let Err(error) = outcome.and_then(|outcome| require_stage_success(name, outcome)) {
                evaluation_failures.push(format!("{name} training failed: {error:#}"));
                continue;
            }
        }
        if !model.exists() {
            evaluation_failures.push(format!(
                "{name} training completed without required model {}",
                model.display()
            ));
            continue;
        }
        std::env::set_var(
            "TOFY_EVAL_MODE",
            if lora_rank.is_some() {
                "unconditioned"
            } else {
                "bridge"
            },
        );
        std::env::set_var("TOFY_EVAL_ARM", name);
        let eval_args = vec![
            "jepa_ai".to_string(),
            "--eval-bridge".to_string(),
            qwen_dir.clone(),
            model.to_string_lossy().to_string(),
            paths.world_encoder_model.to_string_lossy().to_string(),
            matched_encoder_vocab(paths).to_string_lossy().to_string(),
            paths.world_model.to_string_lossy().to_string(),
            EVAL_SUITE.to_string(),
            report_path.to_string_lossy().to_string(),
        ];
        if let Err(error) = run_required_stage_command(&format!("eval_{name}"), &eval_args) {
            evaluation_failures.push(format!("eval_{name} failed: {error:#}"));
        }
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
        restore_env_var("TOFY_EVAL_MIN_PASS_RATE", old_min_pass);
        restore_env_var("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE", old_min_advantage);
        restore_env_var("TOFY_EVAL_MAX_CAUSAL_P_VALUE", old_max_causal_p);
        restore_env_var("TOFY_EVAL_MAX_TASKS", old_limit);
        restore_env_var("TOFY_EVAL_TASK_OFFSET", old_offset);
        restore_env_var("TOFY_EVAL_ARM", old_arm);
        return finish_diagnostic_ladder(&ladder, evaluation_failures);
    }
    let probe_output = paths.eval_stage_dir.join("channel_probe.safetensors");
    let probe_report = probe_output.with_extension("json");
    if ladder == "full" {
        remove_file_if_exists(&probe_output)?;
        remove_file_if_exists(&probe_report)?;
    }
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
    if !paths.bridge_weights_model.exists() {
        evaluation_failures.push(
            "channel_probe skipped because the weights bridge has no qualifying checkpoint"
                .to_string(),
        );
    } else if !probe_output.exists() || !probe_report.exists() {
        if let Err(error) = run_required_stage_command("channel_probe", &probe_args) {
            evaluation_failures.push(format!("channel_probe failed: {error:#}"));
        }
        if !probe_output.exists() || !probe_report.exists() {
            evaluation_failures.push(format!(
                "channel_probe did not produce both {} and {}",
                probe_output.display(),
                probe_report.display()
            ));
        }
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
    restore_env_var("TOFY_EVAL_MIN_PASS_RATE", old_min_pass);
    restore_env_var("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE", old_min_advantage);
    restore_env_var("TOFY_EVAL_MAX_CAUSAL_P_VALUE", old_max_causal_p);
    restore_env_var("TOFY_EVAL_MAX_TASKS", old_limit);
    restore_env_var("TOFY_EVAL_TASK_OFFSET", old_offset);
    restore_env_var("TOFY_EVAL_ARM", old_arm);
    if ladder == "full" {
        enforce_full_experiment_success(paths, evaluation_failures)?;
    } else {
        finish_diagnostic_ladder(&ladder, evaluation_failures)?;
    }
    Ok(())
}

fn enforce_full_experiment_success(paths: &PipelinePaths, mut failures: Vec<String>) -> Result<()> {
    let required_reports = [
        (
            "rag",
            paths.eval_stage_dir.join("rag_preflight_heldout.json"),
        ),
        ("context", paths.eval_stage_dir.join("latent_channel.json")),
        (
            "weights",
            paths.eval_stage_dir.join("knowledge_in_weights.json"),
        ),
        (
            "static_prefix",
            paths.eval_stage_dir.join("static_prefix.json"),
        ),
        ("lora_r16", paths.eval_stage_dir.join("lora_r16.json")),
        ("lora_r512", paths.eval_stage_dir.join("lora_r512.json")),
    ];
    let mut reports = BTreeMap::new();
    for (name, path) in required_reports {
        match load_scientific_report(&path) {
            Ok(report) => {
                reports.insert(name, report);
            }
            Err(error) => failures.push(format!("{name} report unavailable: {error:#}")),
        }
    }
    let expected_suite = match expected_suite() {
        Ok(suite) => suite,
        Err(error) => {
            failures.push(format!("evaluation suite validation failed: {error:#}"));
            return write_and_fail_success_criteria(paths, failures, serde_json::json!({}));
        }
    };
    let report_specs: [(&str, &str, &[&str], bool); 6] = [
        ("rag", "rag_preflight_heldout", &["base"], true),
        (
            "context",
            "latent_channel",
            &["matched", "shuffled", "swapped", "zeroed"],
            false,
        ),
        (
            "weights",
            "knowledge_in_weights",
            &["matched", "shuffled", "swapped", "zeroed"],
            false,
        ),
        ("static_prefix", "static_prefix", &["matched"], false),
        ("lora_r16", "lora_r16", &["base"], false),
        ("lora_r512", "lora_r512", &["base"], false),
    ];
    let mut validated = BTreeMap::new();
    for (name, arm, conditions, heldout_selection_only) in report_specs {
        let Some(report) = reports.get(name) else {
            continue;
        };
        let expected_selection = if heldout_selection_only {
            &expected_suite.heldout_ids
        } else {
            &expected_suite.all_ids
        };
        match validate_scientific_report(
            report,
            name,
            arm,
            &expected_suite.sha256,
            expected_selection,
            &expected_suite.heldout_ids,
            conditions,
        ) {
            Ok(report) => {
                validated.insert(name, report);
            }
            Err(error) => failures.push(format!("{name} report invalid: {error:#}")),
        }
    }
    let channel_probe_metrics: Value =
        match fs::read(paths.eval_stage_dir.join("channel_probe.json"))
            .context("read channel_probe.json")
            .and_then(|bytes| {
                serde_json::from_slice::<Value>(&bytes).context("parse channel_probe.json")
            })
            .and_then(|report| validate_channel_probe_report(report, &paths.bridge_weights_model))
        {
            Ok(report) => report,
            Err(error) => {
                failures.push(format!("channel_probe report invalid: {error:#}"));
                Value::Null
            }
        };
    if validated.len() != report_specs.len() || !failures.is_empty() {
        let metrics = validated_metrics_json(&validated, &channel_probe_metrics);
        return write_and_fail_success_criteria(paths, failures, metrics);
    }

    let rag = &validated["rag"]["base"];
    let context = &validated["context"]["matched"];
    let weights = &validated["weights"]["matched"];
    let static_prefix = &validated["static_prefix"]["matched"];
    let lora_r16 = &validated["lora_r16"]["base"];

    let context_rag_fraction =
        threshold_or_failure("TOFY_CONTEXT_MIN_RAG_FRACTION", 0.5, &mut failures);
    if !recovers_rag_fraction(context, rag, context_rag_fraction) {
        failures.push(format!(
            "context held-out pass rate {:.4} did not recover {:.1}% of the measured RAG ceiling {:.4}",
            context.suite_pass_rate(),
            context_rag_fraction * 100.0,
            rag.suite_pass_rate()
        ));
    }
    let weights_min_pass_rate =
        threshold_or_failure("TOFY_WEIGHTS_MIN_HELDOUT_PASS_RATE", 0.05, &mut failures);
    if weights.suite_pass_rate() < weights_min_pass_rate {
        failures.push(format!(
            "weights held-out pass rate {:.4} did not decisively beat the closed zero-shot floor; required={weights_min_pass_rate:.4}",
            weights.suite_pass_rate(),
        ));
    }

    let max_causal_p = threshold_or_failure(
        "TOFY_EXPERIMENT_MAX_CAUSAL_P_VALUE",
        0.05 / 6.0,
        &mut failures,
    );
    enforce_causal_controls(
        "context",
        &validated["context"],
        threshold_or_failure("TOFY_CONTEXT_MIN_CAUSAL_ADVANTAGE", 0.02, &mut failures),
        max_causal_p,
        &mut failures,
    );
    enforce_causal_controls(
        "weights",
        &validated["weights"],
        threshold_or_failure("TOFY_WEIGHTS_MIN_CAUSAL_ADVANTAGE", 0.02, &mut failures),
        max_causal_p,
        &mut failures,
    );

    let (context_wins, static_wins, paired_tasks) = paired_condition_wins(context, static_prefix);
    let context_static_advantage =
        (context_wins as f64 - static_wins as f64) / paired_tasks.max(1) as f64;
    let context_static_p = tasks::eval::exact_one_sided_sign_test(context_wins, static_wins);
    let min_static_advantage = threshold_or_failure(
        "TOFY_CONTEXT_MIN_STATIC_PREFIX_ADVANTAGE",
        0.02,
        &mut failures,
    );
    let max_static_p = threshold_or_failure(
        "TOFY_CONTEXT_MAX_STATIC_PREFIX_P_VALUE",
        0.05,
        &mut failures,
    );
    if paired_tasks == 0
        || context_static_advantage < min_static_advantage
        || context_static_p > max_static_p
    {
        failures.push(format!(
            "context did not clearly beat static prefix: paired_tasks={paired_tasks}, context_only={context_wins}, static_only={static_wins}, advantage={context_static_advantage:.4}, one_sided_p={context_static_p:.6}"
        ));
    }

    if !rate_at_least(weights, lora_r16) {
        failures.push(format!(
            "weights held-out pass rate {:.4} did not reach the required LoRA-r16 comparator {:.4}",
            weights.suite_pass_rate(),
            lora_r16.suite_pass_rate()
        ));
    }

    let mut metrics = validated_metrics_json(&validated, &channel_probe_metrics);
    let metrics_object = metrics
        .as_object_mut()
        .expect("validated_metrics_json always returns an object");
    metrics_object.insert(
        "context_vs_static_prefix".to_string(),
        serde_json::json!({
            "paired_tasks": paired_tasks,
            "context_only": context_wins,
            "static_only": static_wins,
            "advantage": context_static_advantage,
            "one_sided_p_value": context_static_p,
        }),
    );
    metrics_object.insert(
        "thresholds".to_string(),
        serde_json::json!({
            "context_rag_fraction": context_rag_fraction,
            "weights_min_heldout_pass_rate": weights_min_pass_rate,
            "max_causal_p_value": max_causal_p,
            "context_min_static_prefix_advantage": min_static_advantage,
            "context_max_static_prefix_p_value": max_static_p,
        }),
    );
    let outcome_path = paths.eval_stage_dir.join("success_criteria.json");
    write_text_atomic(
        &outcome_path,
        &format!(
            "{}\n",
            serde_json::to_string_pretty(&serde_json::json!({
                "passed": failures.is_empty(),
                "failures": failures,
                "metrics": metrics,
            }))?
        ),
    )?;
    if !failures.is_empty() {
        bail!(
            "full experiment failed {} scientific success criterion/criteria; see {}:\n- {}",
            failures.len(),
            outcome_path.display(),
            failures.join("\n- ")
        );
    }
    println!(
        "Full experiment passed all scientific success criteria; summary={}.",
        outcome_path.display()
    );
    Ok(())
}

fn finish_diagnostic_ladder(ladder: &str, failures: Vec<String>) -> Result<()> {
    println!(
        "Diagnostic evaluation ladder '{ladder}' completed; full experiment success criteria were not evaluated."
    );
    if !failures.is_empty() {
        bail!(
            "diagnostic evaluation encountered {} failure(s):\n- {}",
            failures.len(),
            failures.join("\n- ")
        );
    }
    Ok(())
}

fn validate_channel_probe_report(report: Value, expected_bridge_model: &Path) -> Result<Value> {
    if report.get("schema_version").and_then(Value::as_u64) != Some(1)
        || report.get("arm").and_then(Value::as_str) != Some("channel_probe")
    {
        bail!("unsupported schema or incorrect arm");
    }
    let recorded_bridge = report
        .get("bridge_model")
        .and_then(Value::as_str)
        .context("missing bridge_model")?;
    if recorded_bridge != expected_bridge_model.to_string_lossy() {
        bail!(
            "bridge checkpoint mismatch: report={recorded_bridge:?} expected={:?}",
            expected_bridge_model.to_string_lossy()
        );
    }
    for field in [
        "steps",
        "batch",
        "seen_validation_tasks",
        "heldout_validation_tasks",
    ] {
        if report.get(field).and_then(Value::as_u64).unwrap_or(0) == 0 {
            bail!("{field} must be a positive integer");
        }
    }
    for field in ["seen_accuracy", "heldout_accuracy"] {
        let value = report
            .get(field)
            .and_then(Value::as_f64)
            .with_context(|| format!("{field} must be numeric"))?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            bail!("{field} must be finite and in [0,1], got {value}");
        }
    }
    Ok(report)
}

fn validated_metrics_json(
    validated: &BTreeMap<&str, BTreeMap<String, ValidatedCondition>>,
    channel_probe: &Value,
) -> Value {
    let mut metrics = serde_json::Map::new();
    for (arm, conditions) in validated {
        let mut arm_metrics = serde_json::Map::new();
        for (condition, result) in conditions {
            arm_metrics.insert(
                condition.clone(),
                serde_json::json!({
                    "passed": result.passed,
                    "tasks": result.tasks,
                    "rate": result.suite_pass_rate(),
                }),
            );
        }
        metrics.insert((*arm).to_string(), Value::Object(arm_metrics));
    }
    if !channel_probe.is_null() {
        metrics.insert("channel_probe".to_string(), channel_probe.clone());
    }
    Value::Object(metrics)
}

fn write_and_fail_success_criteria(
    paths: &PipelinePaths,
    failures: Vec<String>,
    metrics: Value,
) -> Result<()> {
    let outcome_path = paths.eval_stage_dir.join("success_criteria.json");
    write_text_atomic(
        &outcome_path,
        &format!(
            "{}\n",
            serde_json::to_string_pretty(&serde_json::json!({
                "passed": false,
                "failures": failures,
                "metrics": metrics,
            }))?
        ),
    )?;
    bail!(
        "full experiment could not evaluate all scientific success criteria; see {}:\n- {}",
        outcome_path.display(),
        failures.join("\n- ")
    )
}

fn load_scientific_report(path: &Path) -> Result<ScientificReport> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read evaluation report {}", path.display()))?,
    )
    .with_context(|| format!("parse evaluation report {}", path.display()))
}

fn recovers_rag_fraction(
    candidate: &ValidatedCondition,
    rag: &ValidatedCondition,
    required_fraction: f64,
) -> bool {
    candidate.tasks > 0
        && rag.tasks > 0
        && candidate.passed as f64 * rag.tasks as f64
            >= required_fraction * rag.passed as f64 * candidate.tasks as f64
}

fn rate_at_least(candidate: &ValidatedCondition, comparator: &ValidatedCondition) -> bool {
    candidate.tasks > 0
        && comparator.tasks > 0
        && candidate.passed as u128 * comparator.tasks as u128
            >= comparator.passed as u128 * candidate.tasks as u128
}

fn enforce_causal_controls(
    label: &str,
    conditions: &BTreeMap<String, ValidatedCondition>,
    minimum_advantage: f64,
    maximum_p_value: f64,
    failures: &mut Vec<String>,
) {
    let matched = &conditions["matched"];
    for control in ["shuffled", "swapped", "zeroed"] {
        let (matched_only, control_only, tasks) =
            paired_condition_wins(matched, &conditions[control]);
        let matched_advantage = (matched_only as f64 - control_only as f64) / tasks.max(1) as f64;
        let one_sided_p_value = tasks::eval::exact_one_sided_sign_test(matched_only, control_only);
        if tasks == 0
            || matched_advantage < minimum_advantage
            || one_sided_p_value > maximum_p_value
        {
            failures.push(format!(
                "{label}/{control} causal gate failed: tasks={}, matched_only={}, control_only={}, advantage={:.4}, one_sided_p={:.6}, required_advantage={minimum_advantage:.4}, required_max_p={maximum_p_value:.6}",
                tasks,
                matched_only,
                control_only,
                matched_advantage,
                one_sided_p_value,
            ));
        }
    }
}

fn paired_condition_wins(
    left: &ValidatedCondition,
    right: &ValidatedCondition,
) -> (usize, usize, usize) {
    let mut left_wins = 0;
    let mut right_wins = 0;
    let mut paired = 0;
    for (id, left_pass) in &left.outcomes {
        let Some(right_pass) = right.outcomes.get(id).copied() else {
            continue;
        };
        paired += 1;
        match (*left_pass, right_pass) {
            (true, false) => left_wins += 1,
            (false, true) => right_wins += 1,
            _ => {}
        }
    }
    (left_wins, right_wins, paired)
}

fn expected_suite() -> Result<ExpectedSuite> {
    let bytes = fs::read(EVAL_SUITE).with_context(|| format!("read {EVAL_SUITE}"))?;
    let mut all_ids = BTreeSet::new();
    let mut heldout_ids = BTreeSet::new();
    for (index, line) in String::from_utf8(bytes.clone())?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {EVAL_SUITE} row {}", index + 1))?;
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .with_context(|| format!("{EVAL_SUITE} row {} is missing id", index + 1))?
            .to_string();
        if !all_ids.insert(id.clone()) {
            bail!("{EVAL_SUITE} contains duplicate task id {id}");
        }
        if row.get("subset").and_then(Value::as_str) == Some("heldout") {
            heldout_ids.insert(id);
        }
    }
    if heldout_ids.len() != 300 {
        bail!(
            "{EVAL_SUITE} must contain exactly 300 unique held-out tasks, found {}",
            heldout_ids.len()
        );
    }
    Ok(ExpectedSuite {
        sha256: sha256_bytes(&bytes),
        all_ids,
        heldout_ids,
    })
}

fn validate_scientific_report(
    report: &ScientificReport,
    label: &str,
    expected_arm: &str,
    expected_suite_sha256: &str,
    expected_selection: &BTreeSet<String>,
    expected_heldout: &BTreeSet<String>,
    required_conditions: &[&str],
) -> Result<BTreeMap<String, ValidatedCondition>> {
    if report.schema_version != 2 {
        bail!(
            "schema_version={} is unsupported, expected 2",
            report.schema_version
        );
    }
    if report.arm != expected_arm {
        bail!(
            "arm {:?} does not match expected {expected_arm:?}",
            report.arm
        );
    }
    if report.suite_sha256 != expected_suite_sha256 {
        bail!("{label} suite hash does not match the current evaluation suite");
    }
    let selected = report
        .selected_task_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if selected.len() != report.selected_task_ids.len() {
        bail!("{label} selected_task_ids contains duplicates");
    }
    if &selected != expected_selection {
        bail!(
            "{label} selected task coverage is incomplete or mismatched: selected={} expected={}",
            selected.len(),
            expected_selection.len()
        );
    }

    let mut validated = BTreeMap::new();
    for condition in required_conditions {
        let mut outcomes = BTreeMap::new();
        for row in report
            .task_results
            .iter()
            .filter(|row| row.subset == "heldout" && row.condition == *condition)
        {
            if !matches!(
                row.category.as_str(),
                "compile_error" | "tests_failed" | "must_call_violation" | "timeout" | "pass"
            ) {
                bail!(
                    "{label}/{condition} task {} has unknown category {:?}",
                    row.id,
                    row.category
                );
            }
            if outcomes
                .insert(row.id.clone(), row.category == "pass")
                .is_some()
            {
                bail!("{label}/{condition} contains duplicate task id {}", row.id);
            }
        }
        let ids = outcomes.keys().cloned().collect::<BTreeSet<_>>();
        if &ids != expected_heldout {
            bail!(
                "{label}/{condition} held-out coverage is incomplete or mismatched: tasks={} expected={}",
                ids.len(),
                expected_heldout.len()
            );
        }
        let passed = outcomes.values().filter(|passed| **passed).count();
        let serialized = report
            .results
            .get("heldout")
            .and_then(|conditions| conditions.get(*condition))
            .with_context(|| format!("{label} report is missing heldout/{condition} rates"))?;
        let derived_rate = passed as f64 / outcomes.len() as f64;
        if serialized.tasks != outcomes.len()
            || serialized.passed != passed
            || !serialized.suite_pass_rate.is_finite()
            || (serialized.suite_pass_rate - derived_rate).abs() > 1e-12
        {
            bail!("{label}/{condition} serialized rates are inconsistent with task outcomes");
        }
        validated.insert(
            (*condition).to_string(),
            ValidatedCondition {
                tasks: outcomes.len(),
                passed,
                outcomes,
            },
        );
    }
    Ok(validated)
}

fn threshold_or_failure(name: &str, default: f64, failures: &mut Vec<String>) -> f64 {
    let Some(raw) = std::env::var_os(name) else {
        return default;
    };
    match parse_probability(&raw.to_string_lossy()) {
        Ok(value) => value,
        Err(_) => {
            failures.push(format!(
                "{name} must be a finite number in [0,1], got {:?}",
                raw
            ));
            default
        }
    }
}

fn parse_probability(raw: &str) -> Result<f64> {
    let value = raw
        .parse::<f64>()
        .with_context(|| format!("parse probability {raw:?}"))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("probability must be finite and in [0,1], got {raw:?}");
    }
    Ok(value)
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

#[derive(Debug)]
enum StageCommandOutcome {
    Success,
    BridgeNonQualified(tasks::bridge::BridgeStageStatus),
    CudaOutOfMemory(StageFailureStatus),
}

const ADAPTIVE_BATCH_SCHEMA_VERSION: u32 = 1;
const ADAPTIVE_TRAINING_STAGES: [&str; 7] = [
    "latent",
    "world",
    "bridge_context",
    "bridge_weights",
    "static_prefix",
    "lora_r16",
    "lora_r512",
];

#[derive(Debug, Deserialize, Serialize)]
struct AdaptiveBatchState {
    schema_version: u32,
    stages: BTreeMap<String, AdaptiveStageBatch>,
}

impl Default for AdaptiveBatchState {
    fn default() -> Self {
        Self {
            schema_version: ADAPTIVE_BATCH_SCHEMA_VERSION,
            stages: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct AdaptiveStageBatch {
    initial_physical_batch: usize,
    initial_grad_accum: usize,
    effective_batch: usize,
    current_physical_batch: usize,
    current_grad_accum: usize,
    attempts: Vec<AdaptiveBatchAttempt>,
}

#[derive(Debug, Deserialize, Serialize)]
struct AdaptiveBatchAttempt {
    sequence: usize,
    physical_batch: usize,
    grad_accum: usize,
    effective_batch: usize,
    started_unix: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    finished_unix: Option<u64>,
    outcome: AdaptiveBatchAttemptOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AdaptiveBatchAttemptOutcome {
    Running,
    Interrupted,
    Succeeded,
    NonQualified,
    CudaOutOfMemory,
    Failed,
}

fn adaptive_oom_recovery_enabled(_cfg: &PipelineConfig) -> bool {
    !std::env::var("TOFY_AUTO_BATCH_OOM_RECOVERY")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
}

fn validate_batch_pair(physical_batch: usize, grad_accum: usize) -> Result<usize> {
    if !physical_batch.is_power_of_two() || !grad_accum.is_power_of_two() {
        bail!(
            "adaptive CUDA OOM recovery requires power-of-two physical batch and gradient accumulation, got {physical_batch}/{grad_accum}"
        );
    }
    physical_batch
        .checked_mul(grad_accum)
        .context("adaptive effective batch overflow")
}

fn next_batch_pair(physical_batch: usize, grad_accum: usize) -> Result<Option<(usize, usize)>> {
    validate_batch_pair(physical_batch, grad_accum)?;
    if physical_batch == 1 {
        return Ok(None);
    }
    let next_grad_accum = grad_accum
        .checked_mul(2)
        .context("adaptive gradient accumulation overflow")?;
    Ok(Some((physical_batch / 2, next_grad_accum)))
}

fn batch_reductions(initial_batch: usize, current_batch: usize) -> Result<u32> {
    if !initial_batch.is_power_of_two()
        || !current_batch.is_power_of_two()
        || current_batch > initial_batch
    {
        bail!("invalid adaptive batch reduction {initial_batch} -> {current_batch}");
    }
    let ratio = initial_batch / current_batch;
    Ok(ratio.trailing_zeros())
}

fn batch_pair_after_reductions(
    initial_physical_batch: usize,
    initial_grad_accum: usize,
    reductions: u32,
) -> Result<(usize, usize)> {
    validate_batch_pair(initial_physical_batch, initial_grad_accum)?;
    let mut pair = (initial_physical_batch, initial_grad_accum);
    for _ in 0..reductions {
        let Some(next) = next_batch_pair(pair.0, pair.1)? else {
            break;
        };
        pair = next;
    }
    Ok(pair)
}

fn load_adaptive_batch_state(path: &Path) -> Result<AdaptiveBatchState> {
    if !path.exists() {
        return Ok(AdaptiveBatchState::default());
    }
    let state: AdaptiveBatchState = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read adaptive batch state {}", path.display()))?,
    )
    .with_context(|| format!("parse adaptive batch state {}", path.display()))?;
    if state.schema_version != ADAPTIVE_BATCH_SCHEMA_VERSION {
        bail!(
            "unsupported adaptive batch schema {} in {}",
            state.schema_version,
            path.display()
        );
    }
    for (stage, schedule) in &state.stages {
        if !ADAPTIVE_TRAINING_STAGES.contains(&stage.as_str()) {
            bail!(
                "unknown adaptive training stage {stage} in {}",
                path.display()
            );
        }
        validate_adaptive_stage(stage, schedule)?;
    }
    Ok(state)
}

fn validate_adaptive_stage(stage: &str, schedule: &AdaptiveStageBatch) -> Result<()> {
    let initial_effective =
        validate_batch_pair(schedule.initial_physical_batch, schedule.initial_grad_accum)
            .with_context(|| format!("validate initial batch schedule for {stage}"))?;
    if schedule.effective_batch != initial_effective {
        bail!(
            "adaptive batch state for {stage} changed effective batch: initial={initial_effective} recorded={}",
            schedule.effective_batch
        );
    }
    let mut expected_pair = (schedule.initial_physical_batch, schedule.initial_grad_accum);
    for (index, attempt) in schedule.attempts.iter().enumerate() {
        let attempt_effective = validate_batch_pair(attempt.physical_batch, attempt.grad_accum)
            .with_context(|| format!("validate adaptive attempt {} for {stage}", index + 1))?;
        if attempt.sequence != index + 1
            || attempt.effective_batch != schedule.effective_batch
            || attempt_effective != schedule.effective_batch
            || (attempt.physical_batch, attempt.grad_accum) != expected_pair
        {
            bail!("invalid adaptive attempt {} for {stage}", index + 1);
        }
        match attempt.outcome {
            AdaptiveBatchAttemptOutcome::Running => {
                if index + 1 != schedule.attempts.len()
                    || attempt.finished_unix.is_some()
                    || attempt.error.is_some()
                {
                    bail!("invalid running adaptive attempt {} for {stage}", index + 1);
                }
            }
            AdaptiveBatchAttemptOutcome::CudaOutOfMemory => {
                if attempt.finished_unix.is_none() || attempt.error.is_none() {
                    bail!("incomplete CUDA OOM attempt {} for {stage}", index + 1);
                }
                if let Some(next) = next_batch_pair(expected_pair.0, expected_pair.1)? {
                    expected_pair = next;
                }
            }
            AdaptiveBatchAttemptOutcome::Failed | AdaptiveBatchAttemptOutcome::Interrupted => {
                if attempt.finished_unix.is_none() || attempt.error.is_none() {
                    bail!("incomplete failed attempt {} for {stage}", index + 1);
                }
            }
            AdaptiveBatchAttemptOutcome::Succeeded | AdaptiveBatchAttemptOutcome::NonQualified => {
                if attempt.finished_unix.is_none() {
                    bail!("unfinished terminal attempt {} for {stage}", index + 1);
                }
            }
        }
        if attempt
            .finished_unix
            .is_some_and(|finished| finished < attempt.started_unix)
        {
            bail!(
                "adaptive attempt {} for {stage} ends before it starts",
                index + 1
            );
        }
    }
    if (schedule.current_physical_batch, schedule.current_grad_accum) != expected_pair {
        bail!(
            "adaptive batch state for {stage} has current pair {}/{}, expected {}/{}",
            schedule.current_physical_batch,
            schedule.current_grad_accum,
            expected_pair.0,
            expected_pair.1
        );
    }
    Ok(())
}

fn reconcile_interrupted_attempts(state: &mut AdaptiveBatchState) -> Result<bool> {
    let now = unix_timestamp()?;
    let mut changed = false;
    for schedule in state.stages.values_mut() {
        if let Some(attempt) = schedule.attempts.last_mut() {
            if attempt.outcome == AdaptiveBatchAttemptOutcome::Running {
                attempt.outcome = AdaptiveBatchAttemptOutcome::Interrupted;
                attempt.finished_unix = Some(now.max(attempt.started_unix));
                attempt.error = Some("pipeline parent stopped before attempt completion".into());
                changed = true;
            }
        }
    }
    Ok(changed)
}

fn reconcile_adaptive_batch_file(paths: &PipelinePaths) -> Result<()> {
    let path = paths.run_root.join("adaptive_batches.json");
    if !path.exists() {
        return Ok(());
    }
    let mut state = load_adaptive_batch_state(&path)?;
    if reconcile_interrupted_attempts(&mut state)? {
        save_adaptive_batch_state(&path, &state)?;
    }
    Ok(())
}

fn save_adaptive_batch_state(path: &Path, state: &AdaptiveBatchState) -> Result<()> {
    write_text_atomic(path, &serde_json::to_string_pretty(state)?)
}

fn finish_adaptive_attempt(
    path: &Path,
    state: &mut AdaptiveBatchState,
    stage: &str,
    outcome: AdaptiveBatchAttemptOutcome,
    error: Option<String>,
) -> Result<()> {
    let schedule = state
        .stages
        .get_mut(stage)
        .with_context(|| format!("missing adaptive batch state for {stage}"))?;
    let attempt = schedule
        .attempts
        .last_mut()
        .with_context(|| format!("missing adaptive batch attempt for {stage}"))?;
    if attempt.outcome != AdaptiveBatchAttemptOutcome::Running {
        bail!("latest adaptive batch attempt for {stage} is not running");
    }
    attempt.outcome = outcome;
    attempt.finished_unix = Some(unix_timestamp()?);
    attempt.error = error;
    save_adaptive_batch_state(path, state)
}

#[allow(clippy::too_many_arguments)]
fn run_training_stage_with_oom_recovery<BuildArgs, ConfigureEnv>(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    stage: &str,
    initial_physical_batch: usize,
    initial_grad_accum: usize,
    resume_state_path: &Path,
    bridge_status_path: Option<&Path>,
    build_args: BuildArgs,
    mut configure_env: ConfigureEnv,
) -> Result<StageCommandOutcome>
where
    BuildArgs: Fn(usize, usize) -> Vec<String>,
    ConfigureEnv: FnMut(usize, usize) -> Result<()>,
{
    if !adaptive_oom_recovery_enabled(cfg) {
        return run_stage_command(
            stage,
            &append_resume(
                build_args(initial_physical_batch, initial_grad_accum),
                cfg.resume,
            ),
            bridge_status_path,
        );
    }
    if !ADAPTIVE_TRAINING_STAGES.contains(&stage) {
        bail!("CUDA OOM recovery is not configured for training stage {stage}");
    }

    let state_path = paths.run_root.join("adaptive_batches.json");
    let initial_effective = validate_batch_pair(initial_physical_batch, initial_grad_accum)?;
    let mut state = load_adaptive_batch_state(&state_path)?;
    if reconcile_interrupted_attempts(&mut state)? {
        save_adaptive_batch_state(&state_path, &state)?;
    }
    match state.stages.get(stage) {
        Some(schedule)
            if schedule.initial_physical_batch != initial_physical_batch
                || schedule.initial_grad_accum != initial_grad_accum
                || schedule.effective_batch != initial_effective =>
        {
            bail!(
                "adaptive batch state for {stage} was initialized as {}/{}, but this launch requested {}/{}",
                schedule.initial_physical_batch,
                schedule.initial_grad_accum,
                initial_physical_batch,
                initial_grad_accum
            );
        }
        Some(schedule) => validate_adaptive_stage(stage, schedule)?,
        None => {
            state.stages.insert(
                stage.to_string(),
                AdaptiveStageBatch {
                    initial_physical_batch,
                    initial_grad_accum,
                    effective_batch: initial_effective,
                    current_physical_batch: initial_physical_batch,
                    current_grad_accum: initial_grad_accum,
                    attempts: Vec::new(),
                },
            );
            save_adaptive_batch_state(&state_path, &state)?;
        }
    }

    loop {
        let (physical_batch, grad_accum, sequence) = {
            let schedule = state
                .stages
                .get(stage)
                .with_context(|| format!("missing adaptive batch state for {stage}"))?;
            (
                schedule.current_physical_batch,
                schedule.current_grad_accum,
                schedule.attempts.len() + 1,
            )
        };
        configure_env(physical_batch, grad_accum)?;
        let resume = should_resume_stage(cfg.resume, resume_state_path);
        state
            .stages
            .get_mut(stage)
            .expect("adaptive stage was initialized")
            .attempts
            .push(AdaptiveBatchAttempt {
                sequence,
                physical_batch,
                grad_accum,
                effective_batch: initial_effective,
                started_unix: unix_timestamp()?,
                finished_unix: None,
                outcome: AdaptiveBatchAttemptOutcome::Running,
                error: None,
            });
        save_adaptive_batch_state(&state_path, &state)?;
        println!(
            "Launching {stage} with physical_batch={physical_batch} grad_accum={grad_accum} effective_batch={initial_effective} resume={resume}"
        );
        let result = run_stage_command(
            stage,
            &append_resume(build_args(physical_batch, grad_accum), resume),
            bridge_status_path,
        );
        match result {
            Ok(StageCommandOutcome::Success) => {
                finish_adaptive_attempt(
                    &state_path,
                    &mut state,
                    stage,
                    AdaptiveBatchAttemptOutcome::Succeeded,
                    None,
                )?;
                return Ok(StageCommandOutcome::Success);
            }
            Ok(StageCommandOutcome::BridgeNonQualified(status)) => {
                finish_adaptive_attempt(
                    &state_path,
                    &mut state,
                    stage,
                    AdaptiveBatchAttemptOutcome::NonQualified,
                    None,
                )?;
                return Ok(StageCommandOutcome::BridgeNonQualified(status));
            }
            Ok(StageCommandOutcome::CudaOutOfMemory(failure)) => {
                let next_pair = next_batch_pair(physical_batch, grad_accum)?;
                {
                    let schedule = state
                        .stages
                        .get_mut(stage)
                        .expect("adaptive stage was initialized");
                    let attempt = schedule
                        .attempts
                        .last_mut()
                        .expect("adaptive attempt was recorded");
                    if attempt.outcome != AdaptiveBatchAttemptOutcome::Running {
                        bail!("latest adaptive batch attempt for {stage} is not running");
                    }
                    attempt.outcome = AdaptiveBatchAttemptOutcome::CudaOutOfMemory;
                    attempt.finished_unix = Some(unix_timestamp()?);
                    attempt.error = Some(failure.error);
                    if let Some((next_batch, next_accum)) = next_pair {
                        schedule.current_physical_batch = next_batch;
                        schedule.current_grad_accum = next_accum;
                    }
                }
                save_adaptive_batch_state(&state_path, &state)?;
                let Some((next_batch, next_accum)) = next_pair else {
                    bail!(
                        "pipeline stage {stage} exhausted CUDA memory at minimum physical batch 1; effective batch remains {initial_effective}"
                    );
                };
                println!(
                    "CUDA OOM in {stage}; retrying with physical_batch={next_batch} grad_accum={next_accum} effective_batch={initial_effective}"
                );
            }
            Err(error) => {
                finish_adaptive_attempt(
                    &state_path,
                    &mut state,
                    stage,
                    AdaptiveBatchAttemptOutcome::Failed,
                    Some(format!("{error:#}")),
                )?;
                return Err(error);
            }
        }
    }
}

fn should_resume_stage(pipeline_resume: bool, resume_state_path: &Path) -> bool {
    pipeline_resume || resume_state_path.exists()
}

fn run_required_stage_command(stage: &str, args: &[String]) -> Result<()> {
    require_stage_success(stage, run_stage_command(stage, args, None)?)
}

fn require_stage_success(stage: &str, outcome: StageCommandOutcome) -> Result<()> {
    match outcome {
        StageCommandOutcome::Success => Ok(()),
        StageCommandOutcome::BridgeNonQualified(status) => bail!(
            "pipeline stage {stage} did not qualify ({:?} at step {})",
            status.reason,
            status.step
        ),
        StageCommandOutcome::CudaOutOfMemory(status) => bail!(
            "pipeline stage {stage} exhausted CUDA memory: {}",
            status.error
        ),
    }
}

const STAGE_FAILURE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum StageFailureReason {
    CudaOutOfMemory,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StageFailureStatus {
    schema_version: u32,
    stage: String,
    attempt_id: String,
    reason: StageFailureReason,
    error: String,
}

fn is_confirmed_cuda_oom(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}");
    let lowered = text.to_ascii_lowercase();
    lowered.contains("cuda_error_out_of_memory")
        || lowered.contains("cudaerrormemoryallocation")
        || lowered.contains("cublas_status_alloc_failed")
        || lowered.contains("cudnn_status_alloc_failed")
        || (lowered.contains("out of memory")
            && error.chain().any(|cause| {
                cause
                    .downcast_ref::<candle_core::Error>()
                    .is_some_and(candle_error_is_cuda)
            }))
}

fn candle_error_is_cuda(error: &candle_core::Error) -> bool {
    match error {
        candle_core::Error::Cuda(_) => true,
        candle_core::Error::Context { inner, .. }
        | candle_core::Error::WithPath { inner, .. }
        | candle_core::Error::WithBacktrace { inner, .. } => candle_error_is_cuda(inner),
        _ => false,
    }
}

pub fn record_stage_failure_from_env(error: &anyhow::Error) -> Result<bool> {
    if !is_confirmed_cuda_oom(error) {
        return Ok(false);
    }
    let Some(path) = std::env::var_os("TOFY_STAGE_FAILURE_PATH") else {
        return Ok(false);
    };
    let Some(stage) = std::env::var("TOFY_RUN_STAGE_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(false);
    };
    let Some(attempt_id) = std::env::var("TOFY_STAGE_ATTEMPT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(false);
    };
    let status = StageFailureStatus {
        schema_version: STAGE_FAILURE_SCHEMA_VERSION,
        stage,
        attempt_id,
        reason: StageFailureReason::CudaOutOfMemory,
        error: format!("{error:#}"),
    };
    write_text_atomic(
        Path::new(&path),
        &serde_json::to_string_pretty(&status).context("serialize stage failure report")?,
    )?;
    Ok(true)
}

fn run_stage_command(
    stage: &str,
    args: &[String],
    status_path: Option<&Path>,
) -> Result<StageCommandOutcome> {
    let executable = std::env::current_exe().context("resolve current pipeline executable")?;
    let attempt_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before UNIX_EPOCH")?
            .as_nanos()
    );
    let failure_path = std::env::temp_dir().join(format!(
        "tofy-stage-failure-{}-{attempt_id}.json",
        std::process::id()
    ));
    remove_file_if_exists(&failure_path)?;
    if let Some(path) = status_path {
        if path.exists() {
            fs::remove_file(path)
                .with_context(|| format!("remove stale stage status {}", path.display()))?;
        }
    }
    let mut command = Command::new(&executable);
    command
        .args(args.iter().skip(1))
        .env("TOFY_RUN_STAGE_NAME", stage)
        .env("TOFY_STAGE_ATTEMPT_ID", &attempt_id)
        .env("TOFY_STAGE_FAILURE_PATH", &failure_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;

        let expected_parent = std::process::id() as libc::pid_t;
        // SAFETY: this closure runs after fork and before exec, calls only
        // async-signal-safe libc operations, and does not access shared state.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != expected_parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
    }
    if let Some(path) = status_path {
        command.env("TOFY_STAGE_STATUS_PATH", path);
    } else {
        command.env_remove("TOFY_STAGE_STATUS_PATH");
    }
    let status = command
        .status()
        .with_context(|| format!("launch isolated pipeline stage {stage}"))?;
    let outcome = status_path
        .map(|path| read_bridge_nonqualification(path, stage, Some(&attempt_id)))
        .transpose()?
        .flatten();
    let failure = read_stage_failure(&failure_path, stage, &attempt_id)?;
    remove_file_if_exists(&failure_path)?;
    if status.success() {
        if outcome.is_some() || failure.is_some() {
            bail!("pipeline stage {stage} exited successfully but recorded a failure outcome");
        }
        return Ok(StageCommandOutcome::Success);
    }
    if let Some(outcome) = outcome {
        return Ok(StageCommandOutcome::BridgeNonQualified(outcome));
    }
    if let Some(failure) = failure {
        return Ok(StageCommandOutcome::CudaOutOfMemory(failure));
    }
    bail!(
        "isolated pipeline stage {stage} failed with status {status}; executable={}",
        executable.display()
    )
}

fn read_stage_failure(
    path: &Path,
    expected_stage: &str,
    expected_attempt: &str,
) -> Result<Option<StageFailureStatus>> {
    if !path.exists() {
        return Ok(None);
    }
    let status: StageFailureStatus = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read stage failure {}", path.display()))?,
    )
    .with_context(|| format!("parse stage failure {}", path.display()))?;
    if status.schema_version != STAGE_FAILURE_SCHEMA_VERSION
        || status.stage != expected_stage
        || status.attempt_id != expected_attempt
        || status.reason != StageFailureReason::CudaOutOfMemory
    {
        return Ok(None);
    }
    Ok(Some(status))
}

fn read_bridge_nonqualification(
    path: &Path,
    expected_stage: &str,
    expected_attempt: Option<&str>,
) -> Result<Option<tasks::bridge::BridgeStageStatus>> {
    if !path.exists() {
        return Ok(None);
    }
    let status: tasks::bridge::BridgeStageStatus = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read stage status {}", path.display()))?,
    )
    .with_context(|| format!("parse stage status {}", path.display()))?;
    if status.stage != expected_stage
        || status.outcome != "non_qualified"
        || expected_attempt.is_some_and(|attempt| status.attempt_id != attempt)
    {
        if expected_attempt.is_some() {
            return Ok(None);
        }
        bail!(
            "stage status {} does not describe a completed non-qualifying {expected_stage} attempt",
            path.display()
        );
    }
    Ok(Some(status))
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

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn stage_complete(
    cfg: &PipelineConfig,
    model_path: &Path,
    stage: &str,
    target_steps: usize,
    additional_artifacts: &[&Path],
) -> Result<bool> {
    if !cfg.resume {
        return Ok(false);
    }
    if skip_trained_stage(cfg, stage) {
        ensure_stage_artifacts(stage, model_path, additional_artifacts)?;
        println!(
            "Skipping {stage}; --skip-trained accepted existing model {}.",
            model_path.display()
        );
        return Ok(true);
    }
    let state_path = util::checkpoint_sidecar_path(model_path, stage, "resume.json");
    let Some(state) = util::load_resume_state(&state_path, stage)? else {
        return Ok(false);
    };
    let complete = state.step >= target_steps || state.terminal.is_some();
    if complete {
        if !state.saved_checkpoint {
            bail!(
                "stage {stage} is terminal but has no qualifying saved checkpoint: {}",
                state_path.display()
            );
        }
        ensure_stage_artifacts(stage, model_path, additional_artifacts)?;
    }
    Ok(complete)
}

fn ensure_stage_artifacts(
    stage: &str,
    model_path: &Path,
    additional_artifacts: &[&Path],
) -> Result<()> {
    ensure_nonempty_file(model_path)
        .with_context(|| format!("completed stage {stage} is missing its primary checkpoint"))?;
    for artifact in additional_artifacts {
        ensure_nonempty_file(artifact).with_context(|| {
            format!(
                "completed stage {stage} is missing required artifact {}",
                artifact.display()
            )
        })?;
    }
    Ok(())
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
    git_commit: &str,
    source_sha256: &str,
    profile_defaults_sha256: &str,
    immutable_identity: &ImmutableRunIdentity,
) -> Result<()> {
    let selector = cfg.resume_selector.as_deref().unwrap_or("");
    let meta = PipelineMeta {
        schema_version: 3,
        git_commit,
        source_sha256,
        profile_defaults_sha256,
        immutable_identity,
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

fn validate_resume_meta(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    defaults: &ProfileDefaults,
    git_commit: &str,
    source_sha256: &str,
    profile_defaults_sha256: &str,
    immutable_identity: &ImmutableRunIdentity,
) -> Result<ResumeValidation> {
    let meta_path = paths.run_root.join("meta.json");
    let actual: Value =
        serde_json::from_slice(&fs::read(&meta_path).with_context(|| {
            format!("resume requires original metadata {}", meta_path.display())
        })?)
        .with_context(|| format!("parse resume metadata {}", meta_path.display()))?;
    let selector = cfg.resume_selector.as_deref().unwrap_or("");
    let expected = serde_json::to_value(PipelineMeta {
        schema_version: 3,
        git_commit,
        source_sha256,
        profile_defaults_sha256,
        immutable_identity,
        pipeline_run_id: &paths.run_id,
        pipeline_kind: "knowledge",
        resume_enabled: false,
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
    })?;
    let identity_keys = [
        "schema_version",
        "git_commit",
        "source_sha256",
        "profile_defaults_sha256",
        "pipeline_kind",
        "profile",
        "pipeline_until",
        "latent_steps",
        "world_steps",
        "bridge_steps",
        "latent_batch",
        "world_batch",
        "bridge_batch",
        "latent_grad_accum",
        "world_grad_accum",
        "bridge_grad_accum",
        "dim",
        "layers",
        "heads",
        "bridge_dim",
        "num_latent_tokens",
    ];
    let mut mismatches = identity_keys
        .iter()
        .filter_map(|key| {
            let recorded = actual.get(*key).cloned().unwrap_or(Value::Null);
            let current = expected.get(*key).cloned().unwrap_or(Value::Null);
            (recorded != current).then(|| ResumeMismatch {
                key: (*key).to_string(),
                recorded,
                current,
            })
        })
        .collect::<Vec<_>>();
    for key in [
        "training_env_sha256",
        "qwen_sha256",
        "prepared_inputs_sha256",
    ] {
        let recorded = actual
            .get("immutable_identity")
            .and_then(|identity| identity.get(key))
            .cloned()
            .unwrap_or(Value::Null);
        let current = expected
            .get("immutable_identity")
            .and_then(|identity| identity.get(key))
            .cloned()
            .unwrap_or(Value::Null);
        if recorded != current {
            mismatches.push(ResumeMismatch {
                key: format!("immutable_identity.{key}"),
                recorded,
                current,
            });
        }
    }
    let forced = env_truthy("TOFY_ALLOW_RESUME_MISMATCH") && !mismatches.is_empty();
    if forced {
        let keys = mismatches
            .iter()
            .map(|mismatch| mismatch.key.as_str())
            .collect::<Vec<_>>();
        eprintln!(
            "WARNING: resuming despite metadata mismatches in {}; keys={keys:?}",
            meta_path.display()
        );
    }
    Ok(ResumeValidation { mismatches, forced })
}

fn write_resume_launch(
    paths: &PipelinePaths,
    cfg: &PipelineConfig,
    git_commit: &str,
    source_sha256: &str,
    profile_defaults_sha256: &str,
    immutable_identity: &ImmutableRunIdentity,
    validation: &ResumeValidation,
) -> Result<()> {
    let selector = cfg.resume_selector.as_deref().unwrap_or("latest");
    let record = serde_json::json!({
        "timestamp_unix": unix_timestamp()?,
        "command": format!(
            "train {} --until {} --resume {}",
            cfg.profile.as_str(),
            cfg.until.as_str(),
            selector
        ),
        "git_commit": git_commit,
        "current_identities": {
            "source_sha256": source_sha256,
            "profile_defaults_sha256": profile_defaults_sha256,
            "training_env_sha256": immutable_identity.training_env_sha256,
            "qwen_sha256": immutable_identity.qwen_sha256,
            "prepared_inputs_sha256": immutable_identity.prepared_inputs_sha256,
        },
        "identity_match": validation.identity_matches(),
        "forced_mismatch": validation.forced,
        "mismatches": validation.mismatches.iter().map(|mismatch| serde_json::json!({
            "key": mismatch.key,
            "recorded": mismatch.recorded,
            "current": mismatch.current,
        })).collect::<Vec<_>>(),
    });
    let journal = paths.run_root.join("resume_attempts.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&journal)
        .with_context(|| format!("open resume journal {}", journal.display()))?;
    writeln!(file, "{}", serde_json::to_string(&record)?)?;
    Ok(())
}

fn immutable_run_identity(
    paths: &PipelinePaths,
    defaults: &ProfileDefaults,
) -> Result<ImmutableRunIdentity> {
    let training_env = immutable_training_env(defaults);
    let training_env_sha256 = sha256_bytes(&serde_json::to_vec(&training_env)?);
    let qwen_files = qwen_file_identities()?;
    let qwen_sha256 = identity_map_sha256(&qwen_files);
    let prepared_inputs = prepared_input_identities(paths)?;
    let prepared_inputs_sha256 = identity_map_sha256(&prepared_inputs);
    Ok(ImmutableRunIdentity {
        training_env,
        training_env_sha256,
        qwen_files,
        qwen_sha256,
        prepared_inputs,
        prepared_inputs_sha256,
    })
}

fn immutable_training_env(defaults: &ProfileDefaults) -> BTreeMap<String, String> {
    // Deliberately excludes credentials and operational controls such as binary
    // paths, logging/checkpoint cadence, cache prefetch sizing, tmux/RunPod
    // settings, and final-evaluation-only gates.
    const KEYS: &[&str] = &[
        "TOFY_TRAIN_DTYPE",
        "TOFY_OPTIMIZER",
        "TOFY_ADAMW_BETA1",
        "TOFY_ADAMW_BETA2",
        "TOFY_ADAMW_EPS",
        "TOFY_WEIGHT_DECAY",
        "TOFY_MUON_MOMENTUM",
        "TOFY_MUON_NS_STEPS",
        "TOFY_MUON_RMS_SCALE",
        "TOFY_SIGREG_SLICES",
        "TOFY_SIGREG_POINTS",
        "TOFY_SIGREG_POSITION_CHUNK",
        "TOFY_SIGREG_SLICE_CHUNK",
        "TOFY_LABEL_SMOOTHING",
        "TOFY_LR_SCHEDULE",
        "TOFY_LR_WARMUP_STEPS",
        "TOFY_LR_MIN_RATIO",
        "TOFY_RUNTIME_DTYPE",
        "TOFY_ENCODER_DIM",
        "TOFY_ENCODER_LAYERS",
        "TOFY_ENCODER_HEADS",
        "TOFY_LATENT_WARMUP_BATCH",
        "TOFY_LATENT_WARMUP_GRAD_ACCUM",
        "TOFY_LATENT_WARMUP_STEPS",
        "TOFY_LATENT_CONTEXT_SEGMENTS",
        "TOFY_LATENT_RECENT_FULL_SEGMENTS",
        "TOFY_LATENT_HISTORY_RATIO",
        "TOFY_LATENT_CLIP_NORM",
        "TOFY_LATENT_VAL_BATCHES",
        "TOFY_LATENT_EARLY_STOP_PATIENCE",
        "TOFY_LATENT_EARLY_STOP_WARMUP",
        "TOFY_WORLD_WARMUP_BATCH",
        "TOFY_WORLD_WARMUP_GRAD_ACCUM",
        "TOFY_WORLD_WARMUP_STEPS",
        "TOFY_WORLD_MAX_SEQ",
        "TOFY_WORLD_ROLLOUT_STEPS",
        "TOFY_WORLD_TRAIN_ROLLOUT_STEPS",
        "TOFY_WORLD_CONTEXT_SEGMENTS",
        "TOFY_WORLD_RECENT_FULL_SEGMENTS",
        "TOFY_WORLD_CLIP_NORM",
        "TOFY_WORLD_VAL_BATCHES",
        "TOFY_WORLD_GRAD_SPIKE_RATIO",
        "TOFY_WORLD_GRAD_SPIKE_FLOOR",
        "TOFY_WORLD_EARLY_STOP_PATIENCE",
        "TOFY_WORLD_MIN_ASSOCIATION",
        "TOFY_WORLD_ASSOCIATION_PENALTY",
        "TOFY_ENCODER_CONTEXT_SEGMENTS",
        "TOFY_ENCODER_RECENT_FULL_SEGMENTS",
        "TOFY_CONTEXT_HYBRID_MEMORY",
        "TOFY_CONTEXT_HYBRID_EXACT_TAIL",
        "TOFY_CONTEXT_HYBRID_BLOCK_SIZE",
        "TOFY_CONTEXT_RETRIEVAL_SLOTS",
        "TOFY_CONTEXT_EXACT_OLD_TOKENS",
        "TOFY_CONTEXT_SEGMENT_BATCH",
        "TOFY_CONTEXT_COMPRESSOR_DEPTH",
        "TOFY_RECURSIVE_CONTEXT_COMPRESSION",
        "TOFY_BRIDGE_DIM",
        "TOFY_NUM_LATENT_TOKENS",
        "TOFY_BRIDGE_MAX_SEQ",
        "TOFY_ADAPTER_OUTPUT_SLOTS",
        "TOFY_ADAPTER_DEPTH",
        "TOFY_DECODER_ADAPTER_COMPRESS_RATE",
        "TOFY_DECODER_ATTENTION_QUERY_BLOCK",
        "TOFY_QWEN_CROSS_DIM",
        "TOFY_QWEN_CROSS_EVERY",
        "TOFY_BRIDGE_ALIGNMENT_STEPS",
        "TOFY_BRIDGE_ALIGNMENT_TEMPERATURE",
        "TOFY_BRIDGE_CLIP_NORM",
        "TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS",
        "TOFY_BRIDGE_TRAIN_FUNCTION_MAX",
        "TOFY_BRIDGE_VALIDATION_FUNCTION_MAX",
        "TOFY_BRIDGE_VAL_EVERY",
        "TOFY_BRIDGE_AR_VAL_EVERY",
        "TOFY_BRIDGE_AR_VAL_ROWS",
        "TOFY_BRIDGE_AR_MAX_NEW",
        "TOFY_BRIDGE_MIN_AR_PASS_RATE",
        "TOFY_BRIDGE_MIN_AR_ADVANTAGE",
        "TOFY_BRIDGE_MIN_SEMANTIC_GAP",
        "TOFY_BRIDGE_SEMANTIC_WARMUP",
        "TOFY_BRIDGE_SEMANTIC_PATIENCE",
        "TOFY_BRIDGE_MIN_SEMANTIC_PROGRESS",
        "TOFY_CONDITIONING_DROPOUT",
        "TOFY_DECODER_CONDITIONING_MARGIN",
        "TOFY_DECODER_CONDITIONING_MARGIN_WEIGHT",
        "TOFY_DECODER_CONDITIONING_NEGATIVES",
        "TOFY_DECODER_CONDITIONING_UNLIKELIHOOD_WEIGHT",
        "TOFY_DECODER_CONDITIONING_SEPARATION_WEIGHT",
        "TOFY_DECODER_CONDITIONING_MIN_DISTANCE",
        "TOFY_ENCODER_VOCAB_SAMPLE_ROWS",
        "TOFY_ENCODER_VOCAB_SAMPLE_BYTES",
        "TOFY_BPE_MAX_MERGES",
    ];
    let mut env = KEYS
        .iter()
        .map(|key| {
            (
                (*key).to_string(),
                std::env::var(key).unwrap_or_else(|_| "<unset>".to_string()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let bridge_lr = std::env::var("TOFY_BRIDGE_LR").unwrap_or_else(|_| "1e-4".to_string());
    env.insert("TOFY_BRIDGE_LR.context".into(), bridge_lr.clone());
    env.insert("TOFY_BRIDGE_LR.weights".into(), bridge_lr);
    env.insert(
        "TOFY_BRIDGE_GRAD_ACCUM.context".into(),
        std::env::var("TOFY_BRIDGE_GRAD_ACCUM")
            .unwrap_or_else(|_| defaults.bridge_grad_accum.to_string()),
    );
    env.insert(
        "TOFY_BRIDGE_GRAD_ACCUM.weights".into(),
        std::env::var("TOFY_WEIGHTS_BRIDGE_GRAD_ACCUM")
            .unwrap_or_else(|_| defaults.bridge_grad_accum.to_string()),
    );
    env.insert(
        "TOFY_BRIDGE_BATCH.weights".into(),
        std::env::var("TOFY_WEIGHTS_BRIDGE_BATCH")
            .unwrap_or_else(|_| defaults.bridge_batch.to_string()),
    );
    env.insert(
        "TOFY_KNOWLEDGE_UNFREEZE_WORLD".into(),
        env_truthy("TOFY_KNOWLEDGE_UNFREEZE_WORLD").to_string(),
    );
    env
}

fn prepared_input_identities(paths: &PipelinePaths) -> Result<BTreeMap<String, FileIdentity>> {
    let mut inputs = BTreeMap::new();
    for (label, path) in [
        ("encoder_data", PathBuf::from(ENCODER_DATA)),
        ("world_data", PathBuf::from(WORLD_TEXT_DATA)),
        (
            "world_validation",
            PathBuf::from("data/fictional/veclab_knowledge_val.txt"),
        ),
        ("veclab_tasks", PathBuf::from(VECLAB_TASKS)),
        (
            "veclab_heldout_tasks",
            PathBuf::from("data/fictional/veclab_tasks_heldout.txt"),
        ),
        (
            "veclab_docs",
            PathBuf::from("data/fictional/veclab_docs.txt"),
        ),
        ("eval_suite", PathBuf::from(EVAL_SUITE)),
    ] {
        if path.is_file() {
            inputs.insert(
                label.to_string(),
                file_identity(&path, &path.to_string_lossy())?,
            );
        }
    }
    let matched_vocab = matched_encoder_vocab(paths);
    let default_vocab = vocab_dir().join("vocab_encoder.txt");
    let selected_vocab = if matched_vocab.is_file() {
        Some(matched_vocab)
    } else if paths.encoder_cache_vocab.is_file() {
        Some(paths.encoder_cache_vocab.clone())
    } else if default_vocab.is_file() {
        Some(default_vocab)
    } else {
        None
    };
    if let Some(vocab) = selected_vocab {
        inputs.insert(
            "encoder_vocab".into(),
            file_identity(&vocab, &vocab.to_string_lossy())?,
        );
    }
    collect_identity_tree(Path::new("eval/veclab"), "eval_harness", &mut inputs)?;
    collect_identity_tree(
        Path::new("data/fictional/veclab"),
        "veclab_library",
        &mut inputs,
    )?;
    Ok(inputs)
}

fn collect_identity_tree(
    root: &Path,
    label_prefix: &str,
    identities: &mut BTreeMap<String, FileIdentity>,
) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)
        .with_context(|| format!("read identity directory {}", root.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_identity_tree(&path, label_prefix, identities)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let label = format!("{label_prefix}:{}", path.to_string_lossy());
        identities.insert(label, file_identity(&path, &relative)?);
    }
    Ok(())
}

fn qwen_file_identities() -> Result<BTreeMap<String, FileIdentity>> {
    let qwen_dir = PathBuf::from(
        std::env::var("TOFY_QWEN_DIR")
            .context("TOFY_QWEN_DIR is required to record immutable model identity")?,
    );
    if !qwen_dir.is_dir() {
        bail!(
            "TOFY_QWEN_DIR is not a model directory: {}",
            qwen_dir.display()
        );
    }
    let mut paths = Vec::new();
    collect_qwen_identity_files(&qwen_dir, &qwen_dir, &mut paths)?;
    if paths.is_empty() {
        bail!(
            "TOFY_QWEN_DIR contains no model/tokenizer identity files: {}",
            qwen_dir.display()
        );
    }
    paths.sort_by(|(left, _), (right, _)| left.cmp(right));
    paths
        .into_iter()
        .map(|(relative, path)| {
            let identity = file_identity(&path, &relative)?;
            Ok((relative, identity))
        })
        .collect()
}

fn collect_qwen_identity_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("read Qwen model directory {}", directory.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_qwen_identity_files(root, &path, files)?;
            continue;
        }
        if !path.is_file() || !is_qwen_identity_file(&path) {
            continue;
        }
        let relative = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        files.push((relative, path));
    }
    Ok(())
}

fn is_qwen_identity_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("safetensors" | "bin" | "json" | "model" | "txt" | "tiktoken")
    )
}

fn file_identity(path: &Path, recorded_path: &str) -> Result<FileIdentity> {
    let bytes = fs::metadata(path)
        .with_context(|| format!("stat identity file {}", path.display()))?
        .len();
    Ok(FileIdentity {
        path: recorded_path.to_string(),
        bytes,
        sha256: sha256_file(path)?,
    })
}

fn identity_map_sha256(files: &BTreeMap<String, FileIdentity>) -> String {
    let mut hasher = Sha256::new();
    for (label, identity) in files {
        hasher.update(label.as_bytes());
        hasher.update([0]);
        hasher.update(identity.bytes.to_le_bytes());
        hasher.update(identity.sha256.as_bytes());
        hasher.update([0xff]);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn current_git_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("resolve git commit for pipeline provenance")?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed; pipeline provenance cannot be recorded");
    }
    let commit = String::from_utf8(output.stdout)?.trim().to_string();
    if commit.is_empty() {
        bail!("git rev-parse HEAD returned an empty commit");
    }
    Ok(commit)
}

fn source_fingerprint() -> Result<String> {
    let mut files = vec![
        PathBuf::from("Cargo.toml"),
        PathBuf::from("Cargo.lock"),
        PathBuf::from(MODEL_PROFILES_PATH),
    ];
    collect_source_files(Path::new("src"), &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for path in files {
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(
            fs::read(&path).with_context(|| format!("hash pipeline source {}", path.display()))?,
        );
        hasher.update([0xff]);
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

fn collect_source_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("read source directory {}", directory.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_source_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("remove stale evaluation report {}", path.display()))
        }
    }
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

    fn validated_condition(tasks: usize, passed: usize) -> ValidatedCondition {
        let outcomes = (0..tasks)
            .map(|index| (format!("task-{index}"), index < passed))
            .collect();
        ValidatedCondition {
            tasks,
            passed,
            outcomes,
        }
    }

    #[test]
    fn current_profile_schema_parses_minimal() -> Result<()> {
        let text = include_str!("../../config/model_profiles.json");
        let profiles: ModelProfiles = serde_json::from_str(text)?;
        assert_eq!(profiles.minimal.bridge_dim, 640);
        assert_eq!(profiles.minimal.bridge_max_seq, 256);
        assert_eq!(
            (
                profiles.minimal.latent_batch,
                profiles.minimal.latent_grad_accum
            ),
            (16, 8)
        );
        assert_eq!(
            (
                profiles.minimal.world_batch,
                profiles.minimal.world_grad_accum
            ),
            (32, 8)
        );
        assert_eq!(
            (
                profiles.minimal.bridge_batch,
                profiles.minimal.bridge_grad_accum
            ),
            // Measured ceiling on an 80 GiB H100: conditional_generation keeps
            // several full-vocab logit/log-softmax buffers live per micro-step,
            // and 16/8 OOMs where 8/16 holds. The alignment prologue runs no
            // decoder forward, so it fits far larger batches and must not be
            // used to pick this pair.
            (8, 16)
        );
        let value: Value = serde_json::from_str(text)?;
        assert_eq!(
            value.as_object().map(|object| object.len()),
            Some(1),
            "model profile config must contain only minimal"
        );
        assert!(MemoryProfile::parse("48gb").is_err());
        assert!(MemoryProfile::parse("80gb").is_err());
        Ok(())
    }

    #[test]
    fn training_never_schedules_the_base_decoder_floor() {
        assert!(TRAINING_BRIDGE_EVALS
            .iter()
            .all(|(name, mode, _)| *name != "floor" && *mode != "floor" && *name != "rag_ceiling"));
    }

    #[test]
    fn channel_probe_report_is_bound_to_checkpoint_and_complete() -> Result<()> {
        let bridge = Path::new("runs/test/bridge/weights.safetensors");
        let report = serde_json::json!({
            "schema_version": 1,
            "arm": "channel_probe",
            "bridge_model": bridge.to_string_lossy(),
            "steps": 1000,
            "batch": 16,
            "seen_validation_tasks": 300,
            "heldout_validation_tasks": 300,
            "seen_accuracy": 0.75,
            "heldout_accuracy": 0.5,
        });
        validate_channel_probe_report(report, bridge)?;

        let stale = serde_json::json!({
            "schema_version": 1,
            "arm": "channel_probe",
            "bridge_model": "runs/old/bridge/weights.safetensors",
            "steps": 1000,
            "batch": 16,
            "seen_validation_tasks": 300,
            "heldout_validation_tasks": 300,
            "seen_accuracy": 0.75,
            "heldout_accuracy": 0.5,
        });
        assert!(validate_channel_probe_report(stale, bridge).is_err());
        Ok(())
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

    #[test]
    fn half_rag_gate_uses_observed_pass_counts() {
        let rag = validated_condition(300, 125);
        let passing = validated_condition(300, 63);
        let failing = validated_condition(300, 62);
        assert!(recovers_rag_fraction(&passing, &rag, 0.5));
        assert!(!recovers_rag_fraction(&failing, &rag, 0.5));
    }

    #[test]
    fn scientific_thresholds_reject_non_finite_or_out_of_range_values() {
        assert_eq!(parse_probability("0.5").unwrap(), 0.5);
        for invalid in ["NaN", "inf", "-0.01", "1.01", "not-a-number"] {
            assert!(parse_probability(invalid).is_err(), "{invalid} must fail");
        }
    }

    #[test]
    fn lora_parity_cross_multiplies_observed_rates() {
        let weights = validated_condition(300, 60);
        let equal_lora = validated_condition(100, 20);
        let stronger_lora = validated_condition(100, 21);
        assert!(rate_at_least(&weights, &equal_lora));
        assert!(!rate_at_least(&weights, &stronger_lora));
    }

    #[test]
    fn scientific_report_rejects_partial_heldout_coverage() {
        let expected = ["a".to_string(), "b".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let mut report = ScientificReport {
            schema_version: 2,
            arm: "latent_channel".into(),
            suite_sha256: "suite".into(),
            selected_task_ids: expected.iter().cloned().collect(),
            results: BTreeMap::from([(
                "heldout".into(),
                BTreeMap::from([(
                    "matched".into(),
                    ScientificRates {
                        tasks: 2,
                        passed: 1,
                        suite_pass_rate: 0.5,
                    },
                )]),
            )]),
            task_results: vec![
                ScientificTaskResult {
                    id: "a".into(),
                    subset: "heldout".into(),
                    condition: "matched".into(),
                    category: "pass".into(),
                },
                ScientificTaskResult {
                    id: "b".into(),
                    subset: "heldout".into(),
                    condition: "matched".into(),
                    category: "tests_failed".into(),
                },
            ],
        };
        assert!(validate_scientific_report(
            &report,
            "context",
            "latent_channel",
            "suite",
            &expected,
            &expected,
            &["matched"],
        )
        .is_ok());
        report.task_results.pop();
        assert!(validate_scientific_report(
            &report,
            "context",
            "latent_channel",
            "suite",
            &expected,
            &expected,
            &["matched"],
        )
        .is_err());
    }

    #[test]
    fn bridge_status_requires_matching_stage_and_attempt() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-bridge-status-{}-{}.json",
            std::process::id(),
            unix_timestamp()?
        ));
        for reason in [
            tasks::bridge::BridgeNonqualificationReason::SemanticPlateau,
            tasks::bridge::BridgeNonqualificationReason::BudgetExhausted,
        ] {
            let status = tasks::bridge::BridgeStageStatus {
                attempt_id: "attempt-a".into(),
                stage: "bridge_context".into(),
                outcome: "non_qualified".into(),
                reason,
                step: 20_000,
            };
            fs::write(&path, serde_json::to_vec(&status)?)?;
            assert!(
                read_bridge_nonqualification(&path, "bridge_context", Some("attempt-a"))?.is_some()
            );
        }
        assert!(
            read_bridge_nonqualification(&path, "bridge_context", Some("stale-attempt"))?.is_none()
        );
        let wrong_stage = read_bridge_nonqualification(&path, "bridge_weights", None);
        assert!(wrong_stage.is_err());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn adaptive_batch_sequence_preserves_effective_batch() -> Result<()> {
        let mut pair = (16, 4);
        let mut observed = vec![pair];
        while let Some(next) = next_batch_pair(pair.0, pair.1)? {
            pair = next;
            observed.push(pair);
        }
        assert_eq!(observed, vec![(16, 4), (8, 8), (4, 16), (2, 32), (1, 64)]);
        assert!(observed.iter().all(|(batch, accum)| batch * accum == 64));
        Ok(())
    }

    #[test]
    fn adaptive_recovery_covers_only_the_seven_training_arms() {
        assert_eq!(
            ADAPTIVE_TRAINING_STAGES,
            [
                "latent",
                "world",
                "bridge_context",
                "bridge_weights",
                "static_prefix",
                "lora_r16",
                "lora_r512",
            ]
        );
    }

    #[test]
    fn retry_resumes_only_when_requested_or_a_state_sidecar_exists() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-resume-selection-test-{}-{}.json",
            std::process::id(),
            unix_timestamp()?
        ));
        assert!(!should_resume_stage(false, &path));
        assert!(should_resume_stage(true, &path));
        fs::write(&path, b"{}")?;
        assert!(should_resume_stage(false, &path));
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn adaptive_batch_rejects_invalid_pairs_and_stops_at_one() {
        assert!(validate_batch_pair(3, 8).is_err());
        assert!(validate_batch_pair(8, 3).is_err());
        assert!(validate_batch_pair(0, 8).is_err());
        assert!(next_batch_pair(1, 64).unwrap().is_none());
        assert!(next_batch_pair(2, usize::MAX / 2 + 1).is_err());
    }

    #[test]
    fn warmup_pair_keeps_its_own_effective_batch_across_retries() -> Result<()> {
        assert_eq!(batch_pair_after_reductions(8, 2, 0)?, (8, 2));
        assert_eq!(batch_pair_after_reductions(8, 2, 1)?, (4, 4));
        assert_eq!(batch_pair_after_reductions(8, 2, 2)?, (2, 8));
        assert_eq!(batch_pair_after_reductions(8, 2, 4)?, (1, 16));
        assert_eq!(batch_reductions(16, 4)?, 2);
        assert!(batch_pair_after_reductions(6, 2, 1).is_err());
        Ok(())
    }

    #[test]
    fn cuda_oom_classifier_is_cuda_specific() {
        for message in [
            "CUDA_ERROR_OUT_OF_MEMORY during backward",
            "cudaErrorMemoryAllocation",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDNN_STATUS_ALLOC_FAILED",
        ] {
            assert!(
                is_confirmed_cuda_oom(&anyhow!(message)),
                "{message} must be classified"
            );
        }
        for message in [
            "out of memory",
            "process exited with status 137",
            "host out of memory while preparing CUDA metadata",
            "semantic conditioning plateau",
            "no bridge checkpoint passed the joint autoregressive/causal gate",
            "thread panicked",
        ] {
            assert!(
                !is_confirmed_cuda_oom(&anyhow!(message)),
                "{message} must not be classified"
            );
        }
        let candle_cuda_oom = anyhow!(candle_core::Error::Cuda(Box::new(std::io::Error::other(
            "out of memory"
        ))));
        assert!(is_confirmed_cuda_oom(&candle_cuda_oom));
    }

    #[test]
    fn stage_failure_report_is_bound_to_stage_and_attempt() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-stage-failure-test-{}-{}.json",
            std::process::id(),
            unix_timestamp()?
        ));
        let status = StageFailureStatus {
            schema_version: STAGE_FAILURE_SCHEMA_VERSION,
            stage: "world".into(),
            attempt_id: "attempt-a".into(),
            reason: StageFailureReason::CudaOutOfMemory,
            error: "CUDA_ERROR_OUT_OF_MEMORY".into(),
        };
        fs::write(&path, serde_json::to_vec(&status)?)?;
        assert!(read_stage_failure(&path, "world", "attempt-a")?.is_some());
        assert!(read_stage_failure(&path, "latent", "attempt-a")?.is_none());
        assert!(read_stage_failure(&path, "world", "attempt-b")?.is_none());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn adaptive_batch_state_round_trips_and_detects_effective_mismatch() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "tofy-adaptive-batch-test-{}-{}.json",
            std::process::id(),
            unix_timestamp()?
        ));
        let mut state = AdaptiveBatchState::default();
        state.stages.insert(
            "latent".into(),
            AdaptiveStageBatch {
                initial_physical_batch: 16,
                initial_grad_accum: 4,
                effective_batch: 64,
                current_physical_batch: 8,
                current_grad_accum: 8,
                attempts: vec![AdaptiveBatchAttempt {
                    sequence: 1,
                    physical_batch: 16,
                    grad_accum: 4,
                    effective_batch: 64,
                    started_unix: 1,
                    finished_unix: Some(2),
                    outcome: AdaptiveBatchAttemptOutcome::CudaOutOfMemory,
                    error: Some("CUDA_ERROR_OUT_OF_MEMORY".into()),
                }],
            },
        );
        save_adaptive_batch_state(&path, &state)?;
        let loaded = load_adaptive_batch_state(&path)?;
        assert_eq!(loaded.stages["latent"].current_physical_batch, 8);

        state.stages.get_mut("latent").unwrap().effective_batch = 128;
        save_adaptive_batch_state(&path, &state)?;
        assert!(load_adaptive_batch_state(&path).is_err());
        fs::write(&path, b"{not-json")?;
        assert!(load_adaptive_batch_state(&path).is_err());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn interrupted_adaptive_attempt_is_reconciled_before_retry() -> Result<()> {
        let mut state = AdaptiveBatchState::default();
        state.stages.insert(
            "world".into(),
            AdaptiveStageBatch {
                initial_physical_batch: 8,
                initial_grad_accum: 32,
                effective_batch: 256,
                current_physical_batch: 8,
                current_grad_accum: 32,
                attempts: vec![AdaptiveBatchAttempt {
                    sequence: 1,
                    physical_batch: 8,
                    grad_accum: 32,
                    effective_batch: 256,
                    started_unix: unix_timestamp()?,
                    finished_unix: None,
                    outcome: AdaptiveBatchAttemptOutcome::Running,
                    error: None,
                }],
            },
        );
        validate_adaptive_stage("world", &state.stages["world"])?;
        assert!(reconcile_interrupted_attempts(&mut state)?);
        let attempt = &state.stages["world"].attempts[0];
        assert_eq!(attempt.outcome, AdaptiveBatchAttemptOutcome::Interrupted);
        assert!(attempt.finished_unix.is_some());
        assert!(attempt.error.is_some());
        validate_adaptive_stage("world", &state.stages["world"])?;
        Ok(())
    }

    #[test]
    fn latent_early_stop_completes_a_resume_stage() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-latent-terminal-{}-{}",
            std::process::id(),
            unix_timestamp()?
        ));
        fs::create_dir_all(&root)?;
        let model = root.join("latent.safetensors");
        fs::write(&model, b"checkpoint")?;
        let state_path = util::checkpoint_sidecar_path(&model, "latent", "resume.json");
        util::save_resume_state(
            &state_path,
            &util::TrainingResumeState {
                stage: "latent".into(),
                step: 8_200,
                best_metric: 0.0113,
                best_aux_metric: 0.0113,
                saved_checkpoint: true,
                terminal: Some(util::TrainingTerminal::EarlyStopped),
            },
        )?;
        let cfg = PipelineConfig {
            profile: MemoryProfile::Minimal,
            until: PipelineUntil::Full,
            resume: true,
            resume_selector: Some(root.to_string_lossy().to_string()),
            skip_trained_stages: Vec::new(),
            with_code_eval: true,
        };
        assert!(stage_complete(&cfg, &model, "latent", 20_000, &[])?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn terminal_early_stop_completes_a_resume_stage() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-terminal-stage-{}-{}",
            std::process::id(),
            unix_timestamp()?
        ));
        fs::create_dir_all(&root)?;
        let model = root.join("world.safetensors");
        let encoder = root.join("world.encoder.safetensors");
        fs::write(&model, b"checkpoint")?;
        let state_path = util::checkpoint_sidecar_path(&model, "world", "resume.json");
        util::save_resume_state(
            &state_path,
            &util::TrainingResumeState {
                stage: "world".into(),
                step: 1_000,
                best_metric: 0.1,
                best_aux_metric: 0.1,
                saved_checkpoint: true,
                terminal: Some(util::TrainingTerminal::EarlyStopped),
            },
        )?;
        let cfg = PipelineConfig {
            profile: MemoryProfile::Minimal,
            until: PipelineUntil::Full,
            resume: true,
            resume_selector: Some(root.to_string_lossy().to_string()),
            skip_trained_stages: Vec::new(),
            with_code_eval: true,
        };
        assert!(stage_complete(&cfg, &model, "world", 20_000, &[&encoder]).is_err());
        fs::write(&encoder, b"encoder checkpoint")?;
        assert!(stage_complete(&cfg, &model, "world", 20_000, &[&encoder])?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn skip_trained_bridge_requires_unfrozen_world_sidecar() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-bridge-artifacts-{}-{}",
            std::process::id(),
            unix_timestamp()?
        ));
        fs::create_dir_all(&root)?;
        let adapter = root.join("weights.safetensors");
        let world = root.join("weights.world.safetensors");
        fs::write(&adapter, b"adapter checkpoint")?;
        let cfg = PipelineConfig {
            profile: MemoryProfile::Minimal,
            until: PipelineUntil::Full,
            resume: true,
            resume_selector: Some(root.to_string_lossy().to_string()),
            skip_trained_stages: vec!["bridge_weights".into()],
            with_code_eval: true,
        };
        assert!(stage_complete(&cfg, &adapter, "bridge_weights", 20_000, &[&world]).is_err());
        fs::write(&world, b"world checkpoint")?;
        assert!(stage_complete(
            &cfg,
            &adapter,
            "bridge_weights",
            20_000,
            &[&world]
        )?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn identity_digest_uses_content_not_local_path() {
        let left = BTreeMap::from([(
            "encoder_vocab".into(),
            FileIdentity {
                path: "cache/vocab.txt".into(),
                bytes: 4,
                sha256: "abcd".into(),
            },
        )]);
        let right = BTreeMap::from([(
            "encoder_vocab".into(),
            FileIdentity {
                path: "run/latent/model.vocab.txt".into(),
                bytes: 4,
                sha256: "abcd".into(),
            },
        )]);
        assert_eq!(identity_map_sha256(&left), identity_map_sha256(&right));
    }

    #[test]
    fn run_root_lock_rejects_concurrent_owner() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-run-lock-{}-{}",
            std::process::id(),
            unix_timestamp()?
        ));
        fs::create_dir_all(&root)?;
        {
            let _first = RunRootLock::acquire(&root)?;
            assert!(RunRootLock::acquire(&root).is_err());
        }
        {
            let _after_release = RunRootLock::acquire(&root)?;
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
