//! Relocatable, hash-bound evidence manifest for one P2 training run.

use crate::p2::cg_profile::ProfileState;
use crate::p2::experiment::{ResolvedExperiment, TrainingRecipe, WorldCoreFamily};
use crate::p2::train::{
    read_run_attempts, GradientPressureDiagnostics, RunAttempt, RunAttemptRepairState, TrainConfig,
    TrainReport, TrainStatus,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::{collections::BTreeSet, io::ErrorKind};

pub const EVIDENCE_MANIFEST_SCHEMA: &str = "tofy/p2/evidence/2";
pub const EVIDENCE_MANIFEST_FILE: &str = "evidence_manifest.json";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Explicit marker for provenance the trainer could not determine at launch.
/// Manifests record it instead of omitting the field, so a missing value is
/// visible as a claim rather than as an absent key.
pub const UNKNOWN_PROVENANCE: &str = "unknown";

fn unknown_string() -> String {
    UNKNOWN_PROVENANCE.into()
}

/// Runtime checkout context. This is diagnostic context only and is never
/// presented as the source identity of the already-built executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCheckoutProvenance {
    pub path: PathBuf,
    pub revision: String,
    #[serde(default)]
    pub dirty: Option<bool>,
}

impl Default for RuntimeCheckoutProvenance {
    fn default() -> Self {
        Self {
            path: PathBuf::from(UNKNOWN_PROVENANCE),
            revision: UNKNOWN_PROVENANCE.into(),
            dirty: None,
        }
    }
}

/// Identity of the running trainer, captured once per process. Source,
/// dependency, target, profile, and feature claims are embedded by `build.rs`;
/// runtime git inspection is recorded separately and cannot redefine them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchProvenance {
    pub source_revision: String,
    /// `embedded-build:git`, an explicit build-time environment field, or
    /// `unknown`. A runtime checkout is never a valid origin here.
    pub source_revision_origin: String,
    /// Includes tracked and untracked source changes.
    #[serde(default)]
    pub source_dirty: Option<bool>,
    /// Whether the embedded source revision was reachable from its configured
    /// upstream at build time. `None` means the fact was not knowable.
    #[serde(default)]
    pub source_pushed: Option<bool>,
    #[serde(default = "unknown_string")]
    pub build_command: String,
    #[serde(default)]
    pub cargo_features: Vec<String>,
    #[serde(default = "unknown_string")]
    pub cargo_profile: String,
    #[serde(default = "unknown_string")]
    pub cargo_target: String,
    pub binary_path: PathBuf,
    pub binary_sha256: String,
    pub candle_graph_revision: String,
    #[serde(default)]
    pub candle_graph_dirty: Option<bool>,
    #[serde(default)]
    pub candle_graph_pushed: Option<bool>,
    #[serde(default)]
    pub runtime_checkout: RuntimeCheckoutProvenance,
}

impl LaunchProvenance {
    /// Placeholder for callers that own no process identity (unit fixtures).
    pub fn unknown(binary: &Path) -> Self {
        Self {
            source_revision: UNKNOWN_PROVENANCE.into(),
            source_revision_origin: UNKNOWN_PROVENANCE.into(),
            source_dirty: None,
            source_pushed: None,
            build_command: UNKNOWN_PROVENANCE.into(),
            cargo_features: Vec::new(),
            cargo_profile: UNKNOWN_PROVENANCE.into(),
            cargo_target: UNKNOWN_PROVENANCE.into(),
            binary_path: binary.to_path_buf(),
            binary_sha256: UNKNOWN_PROVENANCE.into(),
            candle_graph_revision: UNKNOWN_PROVENANCE.into(),
            candle_graph_dirty: None,
            candle_graph_pushed: None,
            runtime_checkout: RuntimeCheckoutProvenance::default(),
        }
    }

    pub fn source_revision_known(&self) -> bool {
        self.source_revision != UNKNOWN_PROVENANCE
            && (self.source_revision_origin.starts_with("embedded-build:")
                || self.source_revision_origin.starts_with("build-env:"))
    }
}

/// Capture the process's launch provenance once; later calls return the same
/// snapshot so every attempt record and manifest in one process agree.
pub fn launch_provenance() -> &'static LaunchProvenance {
    static PROVENANCE: OnceLock<LaunchProvenance> = OnceLock::new();
    PROVENANCE.get_or_init(capture_launch_provenance)
}

fn capture_launch_provenance() -> LaunchProvenance {
    let binary_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from(UNKNOWN_PROVENANCE));
    let binary_sha256 = hash_file(&binary_path)
        .map(|(_, sha256)| sha256)
        .unwrap_or_else(|_| UNKNOWN_PROVENANCE.into());
    let runtime_checkout = std::env::current_dir()
        .ok()
        .map(|path| RuntimeCheckoutProvenance {
            revision: git_head(&path).unwrap_or_else(|| UNKNOWN_PROVENANCE.into()),
            dirty: git_dirty(&path),
            path,
        })
        .unwrap_or_default();
    LaunchProvenance {
        source_revision: env!("TOFY_EMBEDDED_SOURCE_REVISION").into(),
        source_revision_origin: env!("TOFY_EMBEDDED_SOURCE_REVISION_ORIGIN").into(),
        source_dirty: parse_embedded_bool(env!("TOFY_EMBEDDED_SOURCE_DIRTY")),
        source_pushed: parse_embedded_bool(env!("TOFY_EMBEDDED_SOURCE_PUSHED")),
        build_command: env!("TOFY_EMBEDDED_BUILD_COMMAND").into(),
        cargo_features: parse_embedded_features(env!("TOFY_EMBEDDED_CARGO_FEATURES")),
        cargo_profile: env!("TOFY_EMBEDDED_CARGO_PROFILE").into(),
        cargo_target: env!("TOFY_EMBEDDED_CARGO_TARGET").into(),
        binary_path,
        binary_sha256,
        candle_graph_revision: env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION").into(),
        candle_graph_dirty: parse_embedded_bool(env!("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY")),
        candle_graph_pushed: parse_embedded_bool(env!("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED")),
        runtime_checkout,
    }
}

fn parse_embedded_bool(value: &str) -> Option<bool> {
    match value {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn parse_embedded_features(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter(|feature| !feature.is_empty())
        .map(str::to_owned)
        .collect()
}

fn git_head(tree: &Path) -> Option<String> {
    if !tree.is_dir() {
        return None;
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(tree)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let revision = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (revision.len() >= 7 && revision.chars().all(|c| c.is_ascii_hexdigit())).then_some(revision)
}

fn git_dirty(tree: &Path) -> Option<bool> {
    let output = Command::new("git")
        .arg("-C")
        .arg(tree)
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| !output.stdout.iter().all(u8::is_ascii_whitespace))
}

#[derive(Debug, Serialize, Deserialize)]
struct EvidenceManifest {
    schema: String,
    /// What was trained, stated once at the top level so a reader never has
    /// to infer the world-core family from nested comparison context.
    identity: RunIdentity,
    terminal: TerminalState,
    comparison: ComparisonContext,
    provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    gradient_pressure: Option<GradientPressureBinding>,
    artifacts: Vec<ArtifactDigest>,
    bundle_sha256: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    gaps: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TerminalState {
    status: TrainStatus,
    global_step: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonContext {
    invariants: RunInvariants,
    treatment: ResolvedExperiment,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunInvariants {
    seed: u64,
    physical_batch: usize,
    grad_accum: usize,
    device: String,
    training_population_fingerprint: String,
    training_content_fingerprint: String,
    training_population_rows: u64,
}

/// Treatment identity of the run. Flags are read from the published
/// `config.json`; when that config cannot be parsed they are recorded as
/// `None` (serialized `null`) and a gap names the cause.
#[derive(Debug, Serialize, Deserialize)]
struct RunIdentity {
    recipe: TrainingRecipe,
    family: WorldCoreFamily,
    world_core_schema: String,
    world_core_v6: Option<bool>,
    data_contract_v6: Option<bool>,
    /// Recursion depth (inner = outer) of a v6 run; `None` for pre-v6 runs.
    v6_recursion_steps: Option<usize>,
    physical_batch: usize,
    grad_accum: usize,
    effective_batch: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Provenance {
    /// Identity of the process that published this manifest.
    #[serde(flatten)]
    launch: LaunchProvenance,
    /// Launches of this run root after the fresh start.
    resume_count: u64,
    /// Every launch of this run root, oldest first, with its own revision and binary.
    attempts: Vec<RunAttempt>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GradientPressureBinding {
    sample_count: usize,
    updates: Vec<u64>,
    samples_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ArtifactDigest {
    role: String,
    path: String,
    bytes: u64,
    sha256: String,
}

/// Publish the canonical evidence entry point after the run's report and checkpoint exist.
pub(crate) fn publish_training_evidence(run_dir: &Path, report: &TrainReport) -> Result<PathBuf> {
    let binary = std::env::current_exe().context("resolve current Tofy executable")?;
    publish_training_evidence_with_provenance(run_dir, report, &binary, launch_provenance())
}

fn publish_training_evidence_with_provenance(
    run_dir: &Path,
    report: &TrainReport,
    binary: &Path,
    launch: &LaunchProvenance,
) -> Result<PathBuf> {
    fs::create_dir_all(run_dir).with_context(|| format!("create {}", run_dir.display()))?;
    preserve_prior_evidence(run_dir)?;
    let canonical_run_dir = run_dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", run_dir.display()))?;
    let mut artifacts = Vec::new();
    let report_path = run_dir.join("train_report.json");
    let published_report = fs::read(&report_path)
        .with_context(|| format!("read published report {}", report_path.display()))?;
    let expected_report = serde_json::to_vec_pretty(report).context("serialize training report")?;
    if published_report != expected_report {
        bail!(
            "published train report does not match the completed in-memory report: {}",
            report_path.display()
        );
    }
    let journal_attempts = read_run_attempts(run_dir)?;
    if journal_attempts != report.run_attempts {
        bail!(
            "run attempt journal does not match the published training report: {}",
            run_dir.join("run_attempts.jsonl").display()
        );
    }

    push_artifact(
        &mut artifacts,
        &canonical_run_dir,
        "training_binary",
        binary,
        true,
    )?;
    for (role, path) in [
        ("exported_model", run_dir.join("model.safetensors")),
        ("train_config", run_dir.join("config.json")),
        ("train_report", report_path),
        (
            "checkpoint_model",
            report.latest_checkpoint.join("model.safetensors"),
        ),
        (
            "checkpoint_optimizer",
            report.latest_checkpoint.join("optimizer.safetensors"),
        ),
        (
            "checkpoint_trainer_state",
            report.latest_checkpoint.join("trainer_state.json"),
        ),
        ("checkpoint_latest", run_dir.join("checkpoints/latest.json")),
        ("run_attempt_journal", run_dir.join("run_attempts.jsonl")),
    ] {
        push_artifact(&mut artifacts, &canonical_run_dir, role, &path, false)?;
    }
    let loss_log = run_dir.join("loss_log.jsonl");
    if loss_log.is_file() {
        push_artifact(
            &mut artifacts,
            &canonical_run_dir,
            "active_loss_log",
            &loss_log,
            false,
        )?;
    }
    push_repair_sidecars(
        &mut artifacts,
        &canonical_run_dir,
        run_dir,
        &report.run_attempts,
    )?;
    push_prior_evidence(&mut artifacts, &canonical_run_dir, run_dir)?;
    if let Some(export) = report.export_checkpoint.as_deref() {
        push_artifact(
            &mut artifacts,
            &canonical_run_dir,
            "best_model_snapshot",
            export,
            false,
        )?;
    }
    if let ProfileState::Published(profile) = &report.profile {
        // Derived reports and Nsight files can be augmented after training. Bind
        // the immutable source trace so the root manifest remains valid.
        push_artifact(
            &mut artifacts,
            &canonical_run_dir,
            "profile_trace",
            &profile.trace,
            false,
        )?;
    }
    if let Some(foundation) = report.foundation_v2.as_ref() {
        for bundle in &foundation.profile_bundles {
            push_directory_artifacts(
                &mut artifacts,
                &canonical_run_dir,
                "foundation_v2_profile_bundle",
                bundle,
            )?;
        }
    }

    let pressure_samples = &report.gradient_pressure_samples;
    let gradient_pressure = (!pressure_samples.is_empty())
        .then(|| bind_gradient_pressure(pressure_samples))
        .transpose()?;
    let mut gaps = Vec::new();
    append_provenance_gaps(&mut gaps, "publishing binary", launch);
    if matches!(report.profile, ProfileState::Pending) {
        gaps.push("representative-update profile was not published".into());
    }
    let identity = run_identity(run_dir, report, &mut gaps);
    let resume_count = report.resume_count;
    if report.run_attempts.is_empty() {
        gaps.push("run attempt history unavailable (report predates attempt records)".into());
    } else {
        for attempt in &report.run_attempts {
            append_provenance_gaps(
                &mut gaps,
                &format!("attempt {} binary", attempt.attempt),
                &attempt.provenance,
            );
            match attempt.repair_state {
                RunAttemptRepairState::Pending => gaps.push(format!(
                    "attempt {} has no durable repair result (the process may have stopped mid-repair)",
                    attempt.attempt
                )),
                RunAttemptRepairState::Failed => gaps.push(format!(
                    "attempt {} loss-log repair failed: {}",
                    attempt.attempt,
                    attempt.repair_failure.as_deref().unwrap_or(UNKNOWN_PROVENANCE)
                )),
                RunAttemptRepairState::Completed => {}
            }
        }
        if let Some(repaired) = report
            .run_attempts
            .iter()
            .filter_map(|attempt| attempt.loss_log_repair.as_ref())
            .filter(|repair| repair.rows_removed > 0)
            .map(|repair| repair.rows_removed)
            .reduce(|a, b| a + b)
        {
            gaps.push(format!(
                "loss log was repaired on resume: {repaired} stale row(s) moved to attempt sidecars"
            ));
        }
    }
    let orphan_sidecars = artifacts
        .iter()
        .filter(|artifact| artifact.role == "orphan_loss_log_repair_sidecar")
        .count();
    if orphan_sidecars > 0 {
        gaps.push(format!(
            "{orphan_sidecars} orphan loss-log repair sidecar(s) indicate an interrupted or legacy repair"
        ));
    }
    gaps.sort();
    gaps.dedup();
    let bundle_sha256 = bundle_sha256(&artifacts, gradient_pressure.as_ref());
    let manifest = EvidenceManifest {
        schema: EVIDENCE_MANIFEST_SCHEMA.into(),
        identity,
        terminal: TerminalState {
            status: report.status,
            global_step: report.global_step,
        },
        comparison: ComparisonContext {
            invariants: RunInvariants {
                seed: report.seed,
                physical_batch: report.physical_batch,
                grad_accum: report.grad_accum,
                device: report.device.clone(),
                training_population_fingerprint: report.training_population_fingerprint.clone(),
                training_content_fingerprint: report.training_content_fingerprint.clone(),
                training_population_rows: report.training_population_rows,
            },
            treatment: report.experiment.clone(),
        },
        provenance: Provenance {
            launch: launch.clone(),
            resume_count,
            attempts: report.run_attempts.clone(),
        },
        gradient_pressure,
        artifacts,
        bundle_sha256,
        gaps,
    };
    let path = run_dir.join(EVIDENCE_MANIFEST_FILE);
    write_json_atomic(&path, &manifest)?;
    Ok(path)
}

fn append_provenance_gaps(gaps: &mut Vec<String>, label: &str, provenance: &LaunchProvenance) {
    if !provenance.source_revision_known() {
        gaps.push(format!(
            "{label}: build source revision is unknown or not build-bound"
        ));
    }
    match provenance.source_dirty {
        Some(true) => gaps.push(format!(
            "{label}: build source included tracked or untracked changes"
        )),
        None => gaps.push(format!("{label}: build source dirty state is unknown")),
        Some(false) => {}
    }
    match provenance.source_pushed {
        Some(false) => gaps.push(format!(
            "{label}: build source revision was not on its configured upstream"
        )),
        None => gaps.push(format!("{label}: build source pushed state is unknown")),
        Some(true) => {}
    }
    if provenance.build_command == UNKNOWN_PROVENANCE {
        gaps.push(format!("{label}: build command is unknown"));
    }
    if provenance.cargo_profile == UNKNOWN_PROVENANCE {
        gaps.push(format!("{label}: Cargo profile is unknown"));
    }
    if provenance.cargo_target == UNKNOWN_PROVENANCE {
        gaps.push(format!("{label}: Cargo target is unknown"));
    }
    if provenance.candle_graph_revision == UNKNOWN_PROVENANCE {
        gaps.push(format!("{label}: candle_graph build revision is unknown"));
    }
    match provenance.candle_graph_dirty {
        Some(true) => gaps.push(format!(
            "{label}: candle_graph build source included tracked or untracked changes"
        )),
        None => gaps.push(format!("{label}: candle_graph dirty state is unknown")),
        Some(false) => {}
    }
    match provenance.candle_graph_pushed {
        Some(false) => gaps.push(format!(
            "{label}: candle_graph revision was not on its configured upstream"
        )),
        None => gaps.push(format!("{label}: candle_graph pushed state is unknown")),
        Some(true) => {}
    }
}

/// Preserve an evidence/1 manifest before publishing evidence/2 at the
/// canonical entry point. Re-publication of v2 is still atomic and stable.
fn preserve_prior_evidence(run_dir: &Path) -> Result<()> {
    let current = run_dir.join(EVIDENCE_MANIFEST_FILE);
    if !current.is_file() {
        return Ok(());
    }
    let bytes = fs::read(&current).with_context(|| format!("read {}", current.display()))?;
    let schema = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|value| value.get("schema")?.as_str().map(str::to_owned));
    if schema.as_deref() == Some(EVIDENCE_MANIFEST_SCHEMA) {
        return Ok(());
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let archive = run_dir.join(format!("evidence_manifest.prior-{}.json", &digest[..16]));
    match OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&archive)
    {
        Ok(mut file) => {
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .with_context(|| format!("preserve {}", archive.display()))?;
            File::open(run_dir)?.sync_all()?;
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            if fs::read(&archive)? != bytes {
                bail!("prior evidence archive collision at {}", archive.display());
            }
        }
        Err(error) => {
            return Err(error).with_context(|| format!("create {}", archive.display()));
        }
    }
    Ok(())
}

fn push_prior_evidence(
    artifacts: &mut Vec<ArtifactDigest>,
    canonical_run_dir: &Path,
    run_dir: &Path,
) -> Result<()> {
    let mut paths = fs::read_dir(run_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("evidence_manifest.prior-")
                })
        })
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        push_artifact(
            artifacts,
            canonical_run_dir,
            "prior_evidence_manifest",
            &path,
            false,
        )?;
    }
    Ok(())
}

fn push_repair_sidecars(
    artifacts: &mut Vec<ArtifactDigest>,
    canonical_run_dir: &Path,
    run_dir: &Path,
    attempts: &[RunAttempt],
) -> Result<()> {
    let mut referenced = BTreeSet::new();
    for path in attempts
        .iter()
        .filter_map(|attempt| attempt.loss_log_repair.as_ref())
        .filter_map(|repair| repair.removed_rows_path.as_ref())
    {
        referenced.insert(path.canonicalize().with_context(|| {
            format!("canonicalize referenced repair sidecar {}", path.display())
        })?);
    }
    let mut discovered = fs::read_dir(run_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("loss_log.jsonl.attempt-")
                })
        })
        .collect::<Vec<_>>();
    discovered.sort();
    for path in discovered {
        let canonical = path.canonicalize()?;
        let role = if referenced.remove(&canonical) {
            "loss_log_repair_sidecar"
        } else {
            "orphan_loss_log_repair_sidecar"
        };
        push_artifact(artifacts, canonical_run_dir, role, &path, false)?;
    }
    if let Some(missing) = referenced.into_iter().next() {
        bail!(
            "referenced loss-log repair sidecar is missing from run directory: {}",
            missing.display()
        );
    }
    Ok(())
}

/// Treatment flags come from the published `config.json` so the manifest
/// states exactly what the binary was launched with. The config artifact is
/// already required (its absence fails closed above); an unparsable one is
/// recorded as unknown flags plus a gap rather than an empty object.
fn run_identity(run_dir: &Path, report: &TrainReport, gaps: &mut Vec<String>) -> RunIdentity {
    let experiment = &report.experiment;
    let config = fs::read(run_dir.join("config.json"))
        .context("read published train config")
        .and_then(|bytes| {
            serde_json::from_slice::<TrainConfig>(&bytes).context("parse published train config")
        });
    let (world_core_v6, data_contract_v6, v6_recursion_steps) = match config {
        Ok(config) => (
            Some(config.world_core_v6),
            Some(config.data_contract_v6),
            config.world_core_v6.then_some(config.inner_steps),
        ),
        Err(error) => {
            gaps.push(format!(
                "treatment flags unknown: published config.json is not a train config ({error:#})"
            ));
            (None, None, None)
        }
    };
    RunIdentity {
        recipe: experiment.recipe,
        family: experiment.family,
        world_core_schema: experiment.report_schema.clone(),
        world_core_v6,
        data_contract_v6,
        v6_recursion_steps,
        physical_batch: report.physical_batch,
        grad_accum: report.grad_accum,
        effective_batch: report.physical_batch.saturating_mul(report.grad_accum),
    }
}

fn bind_gradient_pressure(
    samples: &[GradientPressureDiagnostics],
) -> Result<GradientPressureBinding> {
    let encoded = serde_json::to_vec(samples).context("serialize gradient-pressure samples")?;
    Ok(GradientPressureBinding {
        sample_count: samples.len(),
        updates: samples.iter().map(|sample| sample.update).collect(),
        samples_sha256: format!("sha256:{:x}", Sha256::digest(encoded)),
    })
}

fn push_artifact(
    artifacts: &mut Vec<ArtifactDigest>,
    run_dir: &Path,
    role: &str,
    path: &Path,
    allow_outside_run: bool,
) -> Result<()> {
    if !path.is_file() {
        bail!("required evidence artifact is missing: {}", path.display());
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize evidence artifact {}", path.display()))?;
    let stored_path = match canonical.strip_prefix(run_dir) {
        Ok(relative) => relative,
        Err(_) if allow_outside_run => canonical.as_path(),
        Err(_) => bail!(
            "evidence artifact `{role}` is outside run directory {}: {}",
            run_dir.display(),
            canonical.display()
        ),
    };
    let (bytes, sha256) = hash_file(&canonical)?;
    artifacts.push(ArtifactDigest {
        role: role.into(),
        path: stored_path.to_string_lossy().into_owned(),
        bytes,
        sha256,
    });
    Ok(())
}

fn push_directory_artifacts(
    artifacts: &mut Vec<ArtifactDigest>,
    run_dir: &Path,
    role: &str,
    directory: &Path,
) -> Result<()> {
    if !directory.is_dir() {
        bail!(
            "required evidence bundle directory is missing: {}",
            directory.display()
        );
    }
    let mut pending = vec![directory.to_path_buf()];
    let mut files = Vec::new();
    while let Some(current) = pending.pop() {
        let mut entries = fs::read_dir(&current)
            .with_context(|| format!("read evidence bundle {}", current.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                files.push(entry.path());
            } else {
                bail!(
                    "evidence bundle contains unsupported artifact {}",
                    entry.path().display()
                );
            }
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("evidence bundle is empty: {}", directory.display());
    }
    for path in files {
        push_artifact(artifacts, run_dir, role, &path, false)?;
    }
    Ok(())
}

fn bundle_sha256(
    artifacts: &[ArtifactDigest],
    pressure: Option<&GradientPressureBinding>,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"tofy.p2.evidence.bundle.v2\0");
    for artifact in artifacts {
        for value in [
            artifact.role.as_bytes(),
            artifact.path.as_bytes(),
            artifact.sha256.as_bytes(),
        ] {
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value);
        }
        hash.update(artifact.bytes.to_le_bytes());
    }
    if let Some(pressure) = pressure {
        hash.update((pressure.sample_count as u64).to_le_bytes());
        hash.update((pressure.samples_sha256.len() as u64).to_le_bytes());
        hash.update(pressure.samples_sha256.as_bytes());
    } else {
        hash.update(0_u64.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let file = File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync evidence artifact {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes = bytes.saturating_add(read as u64);
        hash.update(&buffer[..read]);
    }
    Ok((bytes, format!("sha256:{:x}", hash.finalize())))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        serde_json::to_writer_pretty(&mut file, value)
            .with_context(|| format!("write {}", tmp.display()))?;
        file.write_all(b"\n")?;
        file.sync_all()
            .with_context(|| format!("sync {}", tmp.display()))?;
        fs::rename(&tmp, path)
            .with_context(|| format!("publish {} -> {}", tmp.display(), path.display()))?;
        if let Some(parent) = path.parent() {
            File::open(parent)
                .with_context(|| format!("open {} for sync", parent.display()))?
                .sync_all()
                .with_context(|| format!("sync {}", parent.display()))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::cg_profile::ProfileArtifacts;
    use crate::p2::data::EventLabelCensus;
    use crate::p2::train::{
        FoundationV2LossMeans, FoundationV2TrainingReport, PromotionMetric, RunAttemptKind,
        RunAttemptRepairState,
    };

    fn write(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn publishes_relocatable_hash_bound_run_evidence() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-evidence-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let checkpoint = root.join("checkpoints/step-000000000007");
        let profile_dir = root.join("profile/update-000000000002");
        let foundation_profile_dir = root.join("profile/update-000000000004");
        let binary = root.join("test-tofy");
        for (path, contents) in [
            (root.join("model.safetensors"), b"export".as_slice()),
            (root.join("config.json"), b"{}".as_slice()),
            (checkpoint.join("model.safetensors"), b"model".as_slice()),
            (
                checkpoint.join("optimizer.safetensors"),
                b"optimizer".as_slice(),
            ),
            (checkpoint.join("trainer_state.json"), b"state".as_slice()),
            (root.join("checkpoints/latest.json"), b"latest".as_slice()),
            (profile_dir.join("bundle.json"), b"bundle".as_slice()),
            (profile_dir.join("trace.jsonl"), b"trace".as_slice()),
            (profile_dir.join("evidence.json"), b"evidence".as_slice()),
            (profile_dir.join("report.md"), b"markdown".as_slice()),
            (profile_dir.join("viewer.html"), b"viewer".as_slice()),
            (
                profile_dir.join("nsight/status.txt"),
                b"available".as_slice(),
            ),
            (
                foundation_profile_dir.join("bundle.json"),
                b"foundation bundle".as_slice(),
            ),
            (
                foundation_profile_dir.join("trace.jsonl"),
                b"foundation trace".as_slice(),
            ),
            (
                foundation_profile_dir.join("evidence.json"),
                b"foundation evidence".as_slice(),
            ),
            (
                foundation_profile_dir.join("report.md"),
                b"foundation report".as_slice(),
            ),
            (
                foundation_profile_dir.join("viewer.html"),
                b"foundation viewer".as_slice(),
            ),
            (
                foundation_profile_dir.join("nsight/status.txt"),
                b"not requested".as_slice(),
            ),
            (binary.clone(), b"binary".as_slice()),
        ] {
            write(&path, contents);
        }
        let pressure = GradientPressureDiagnostics {
            update: 3,
            encoder_next_latent_l2: 1.0,
            encoder_sigreg_weighted_l2: 0.5,
            sigreg_to_next_ratio: Some(0.5),
            encoder_grounding_weighted_l2: None,
            grounding_to_next_ratio: None,
            grounding_head_weighted_l2: None,
            sigreg_next_cosine: None,
            grounding_next_cosine: None,
            grounding_sigreg_cosine: None,
            encoder_readout_weighted_l2: None,
            readout_to_next_ratio: None,
            model_next_latent_l2: None,
            displacement_health_weighted_l2: None,
            displacement_health_to_next_ratio: None,
        };
        let launch = LaunchProvenance {
            source_revision: "deadbeef".into(),
            source_revision_origin: "embedded-build:test".into(),
            source_dirty: Some(false),
            source_pushed: Some(true),
            build_command: "cargo test evidence".into(),
            cargo_features: vec!["jemalloc".into()],
            cargo_profile: "test".into(),
            cargo_target: "test-target".into(),
            binary_path: binary.clone(),
            binary_sha256: hash_file(&binary)?.1,
            candle_graph_revision: "cafef00d".into(),
            candle_graph_dirty: Some(false),
            candle_graph_pushed: Some(true),
            runtime_checkout: RuntimeCheckoutProvenance::default(),
        };
        let report = TrainReport {
            schema: "p2.train_report.v7".into(),
            world_core_schema: "legacy_p2_eval_compatible".into(),
            experiment: ResolvedExperiment::default(),
            seed: 7,
            physical_batch: 2,
            grad_accum: 4,
            lr: 1e-3,
            weight_decay: 0.01,
            parameter_count: 10,
            training_population_fingerprint: "fnv1a64:01".into(),
            training_content_fingerprint: "sha256:02".into(),
            training_population_rows: 8,
            device: "cuda:0".into(),
            lessons: vec![],
            status: TrainStatus::Completed,
            global_step: 7,
            latest_checkpoint: checkpoint.clone(),
            resumed_from: None,
            batch_schedule_migrations: vec![],
            checkpoint: root.join("model.safetensors"),
            export_checkpoint: None,
            config_path: root.join("config.json"),
            profile: ProfileState::Published(ProfileArtifacts {
                update: 2,
                directory: profile_dir.clone(),
                trace: profile_dir.join("trace.jsonl"),
                evidence_json: profile_dir.join("evidence.json"),
                evidence_markdown: profile_dir.join("report.md"),
                viewer_html: profile_dir.join("viewer.html"),
                nsight_directory: profile_dir.join("nsight"),
            }),
            gradient_pressure: Some(pressure.clone()),
            gradient_pressure_samples: vec![pressure],
            foundation_v2: Some(FoundationV2TrainingReport {
                total_steps: 7,
                mean_losses: FoundationV2LossMeans::default(),
                ep_weight: 0.01,
                ep_gradient_budget: vec![],
                gate_history: vec![],
                best_changed_exact: None,
                promotion_metric: PromotionMetric::ChangedExact,
                best_promotion_value: None,
                best_checkpoint: None,
                rollout_enabled: true,
                rollout_population: None,
                gradient_pressure: vec![],
                permanent_checkpoints: vec![],
                event_label_census: EventLabelCensus::default(),
                event_label_census_complete: true,
                mechanism_history: vec![],
                profile_bundles: vec![foundation_profile_dir.clone()],
                clip_strategy: "test".into(),
            }),
            research_claim: false,
            resume_count: 1,
            run_attempts: vec![
                RunAttempt {
                    attempt: 1,
                    kind: RunAttemptKind::Fresh,
                    started_unix_secs: 10,
                    pid: 1,
                    resumed_from: None,
                    resumed_step: None,
                    provenance: launch.clone(),
                    repair_state: RunAttemptRepairState::Completed,
                    loss_log_repair: None,
                    repair_failure: None,
                },
                RunAttempt {
                    attempt: 2,
                    kind: RunAttemptKind::Resume,
                    started_unix_secs: 20,
                    pid: 2,
                    resumed_from: Some(checkpoint.clone()),
                    resumed_step: Some(7),
                    provenance: launch.clone(),
                    repair_state: RunAttemptRepairState::Completed,
                    loss_log_repair: None,
                    repair_failure: None,
                },
            ],
        };
        let attempt_journal = report
            .run_attempts
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .join("\n")
            + "\n";
        write(&root.join("run_attempts.jsonl"), attempt_journal.as_bytes());
        write(
            &root.join("loss_log.jsonl"),
            b"{\"global_step\":7,\"total\":1.0}\n",
        );
        write(
            &root.join("loss_log.jsonl.attempt-99"),
            b"{\"global_step\":8,\"orphan\":true}\n",
        );
        write(
            &root.join("train_report.json"),
            &serde_json::to_vec_pretty(&report)?,
        );
        write(
            &root.join(EVIDENCE_MANIFEST_FILE),
            b"{\"schema\":\"tofy/p2/evidence/1\",\"legacy\":true}\n",
        );

        let path = publish_training_evidence_with_provenance(&root, &report, &binary, &launch)?;
        let first_publication = fs::read(&path)?;
        publish_training_evidence_with_provenance(&root, &report, &binary, &launch)?;
        assert_eq!(fs::read(&path)?, first_publication);
        let manifest: EvidenceManifest = serde_json::from_reader(File::open(&path)?)?;

        assert_eq!(manifest.schema, EVIDENCE_MANIFEST_SCHEMA);
        assert_eq!(manifest.provenance.launch, launch);
        assert_eq!(manifest.provenance.resume_count, 1);
        assert_eq!(manifest.provenance.attempts.len(), 2);
        assert_eq!(manifest.identity.family, WorldCoreFamily::Legacy);
        assert_eq!(manifest.identity.effective_batch, 8);
        // `{}` is not a train config: flags are recorded as unknown, not dropped.
        let raw: serde_json::Value = serde_json::from_slice(&first_publication)?;
        assert!(raw["identity"]["world_core_v6"].is_null());
        assert!(raw["provenance"]["source_revision"] == "deadbeef");
        assert!(manifest
            .gaps
            .iter()
            .any(|gap| gap.contains("treatment flags unknown")));
        assert_eq!(manifest.artifacts.len(), 19);
        assert!(manifest.artifacts.iter().any(|artifact| {
            artifact.role == "checkpoint_optimizer"
                && artifact.path == "checkpoints/step-000000000007/optimizer.safetensors"
                && artifact.sha256 == hash_file(&root.join(&artifact.path)).unwrap().1
        }));
        assert!(manifest
            .artifacts
            .iter()
            .all(|artifact| artifact.path != EVIDENCE_MANIFEST_FILE));
        assert_eq!(
            manifest
                .artifacts
                .iter()
                .filter(|artifact| artifact.role == "foundation_v2_profile_bundle")
                .count(),
            6
        );
        assert_eq!(manifest.gradient_pressure.unwrap().updates, vec![3]);
        assert!(manifest
            .artifacts
            .iter()
            .any(|artifact| artifact.role == "prior_evidence_manifest"));
        assert!(manifest
            .artifacts
            .iter()
            .any(|artifact| artifact.role == "orphan_loss_log_repair_sidecar"));
        assert!(manifest.bundle_sha256.starts_with("sha256:"));
        assert_eq!(manifest.gaps.len(), 2, "{:?}", manifest.gaps);
        assert!(manifest
            .gaps
            .iter()
            .any(|gap| gap.contains("orphan loss-log repair sidecar")));
        let mut mismatched_report = report.clone();
        mismatched_report.global_step += 1;
        assert!(publish_training_evidence_with_provenance(
            &root,
            &mismatched_report,
            &binary,
            &launch
        )
        .is_err());
        assert!(fs::read_dir(&root)?.all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    /// The process provenance is captured once and never empty: every field
    /// is either a real value or the explicit `unknown` marker.
    #[test]
    fn launch_provenance_is_explicit_and_stable() {
        let first = launch_provenance();
        let second = launch_provenance();
        assert_eq!(first, second);
        assert!(!first.source_revision.is_empty());
        assert!(!first.source_revision_origin.is_empty());
        assert!(!first.binary_sha256.is_empty());
        assert!(!first.candle_graph_revision.is_empty());
        if first.source_revision_known() {
            assert!(
                first.source_revision_origin.starts_with("embedded-build:")
                    || first.source_revision_origin.starts_with("build-env:")
            );
        } else {
            assert_eq!(first.source_revision_origin, UNKNOWN_PROVENANCE);
        }
        let json = serde_json::to_value(first).unwrap();
        assert!(json["source_revision"].is_string());
        assert!(json["binary_sha256"].is_string());
        assert!(json["source_dirty"].is_boolean() || json["source_dirty"].is_null());
        assert!(json["source_pushed"].is_boolean() || json["source_pushed"].is_null());
        assert!(json["build_command"].is_string());
        assert!(json["cargo_features"].is_array());
        assert!(json["cargo_profile"].is_string());
        assert!(json["cargo_target"].is_string());
        assert!(json["runtime_checkout"].is_object());
    }

    #[test]
    fn embedded_build_metadata_parsing_is_deterministic() {
        assert_eq!(parse_embedded_bool("true"), Some(true));
        assert_eq!(parse_embedded_bool("false"), Some(false));
        assert_eq!(parse_embedded_bool("unknown"), None);
        assert_eq!(parse_embedded_bool("TRUE"), None);
        assert_eq!(
            parse_embedded_features("cuda,cudnn,jemalloc"),
            ["cuda", "cudnn", "jemalloc"]
        );
        assert!(parse_embedded_features("").is_empty());
    }

    #[test]
    fn unknown_dirty_and_unpushed_builds_are_explicit_evidence_gaps() {
        let mut provenance = LaunchProvenance::unknown(Path::new("binary"));
        provenance.source_revision = "deadbeef".into();
        provenance.source_revision_origin = "embedded-build:test".into();
        provenance.source_dirty = Some(true);
        provenance.source_pushed = Some(false);
        let mut gaps = Vec::new();
        append_provenance_gaps(&mut gaps, "test", &provenance);
        assert!(gaps.iter().any(|gap| gap.contains("tracked or untracked")));
        assert!(gaps
            .iter()
            .any(|gap| gap.contains("not on its configured upstream")));
        assert!(gaps
            .iter()
            .any(|gap| gap.contains("build command is unknown")));

        provenance.source_dirty = None;
        provenance.source_pushed = None;
        gaps.clear();
        append_provenance_gaps(&mut gaps, "test", &provenance);
        assert!(gaps
            .iter()
            .any(|gap| gap.contains("dirty state is unknown")));
        assert!(gaps
            .iter()
            .any(|gap| gap.contains("pushed state is unknown")));
    }
}
