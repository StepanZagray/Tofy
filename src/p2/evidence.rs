//! Relocatable, hash-bound evidence manifest for one P2 training run.

use crate::p2::cg_profile::ProfileState;
use crate::p2::experiment::{ResolvedExperiment, TrainingRecipe, WorldCoreFamily};
use crate::p2::train::{
    GradientPressureDiagnostics, RunAttempt, TrainConfig, TrainReport, TrainStatus,
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

pub const EVIDENCE_MANIFEST_SCHEMA: &str = "tofy/p2/evidence/1";
pub const EVIDENCE_MANIFEST_FILE: &str = "evidence_manifest.json";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Explicit marker for provenance the trainer could not determine at launch.
/// Manifests record it instead of omitting the field, so a missing value is
/// visible as a claim rather than as an absent key.
pub const UNKNOWN_PROVENANCE: &str = "unknown";

/// Identity of the running trainer, captured once per process at launch:
/// the source revision (git HEAD of the tree the binary was built from, or of
/// the working directory as a fallback), whether that tree carried
/// uncommitted changes, the executable's SHA-256, and the sibling
/// `candle_graph` checkout the profile evidence depends on. Every value falls
/// back to [`UNKNOWN_PROVENANCE`] rather than being dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchProvenance {
    pub source_revision: String,
    /// Where `source_revision` came from: `git:<tree>`,
    /// `env:TOFY_SOURCE_REVISION`, or `unknown`.
    pub source_revision_origin: String,
    /// `None` when the tree state could not be inspected (env override or no git).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dirty: Option<bool>,
    pub binary_path: PathBuf,
    pub binary_sha256: String,
    pub candle_graph_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle_graph_dirty: Option<bool>,
}

impl LaunchProvenance {
    /// Placeholder for callers that own no process identity (unit fixtures).
    pub fn unknown(binary: &Path) -> Self {
        Self {
            source_revision: UNKNOWN_PROVENANCE.into(),
            source_revision_origin: UNKNOWN_PROVENANCE.into(),
            source_dirty: None,
            binary_path: binary.to_path_buf(),
            binary_sha256: UNKNOWN_PROVENANCE.into(),
            candle_graph_revision: UNKNOWN_PROVENANCE.into(),
            candle_graph_dirty: None,
        }
    }

    pub fn source_revision_known(&self) -> bool {
        self.source_revision != UNKNOWN_PROVENANCE
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
    // The tree this binary was compiled from is the most faithful source
    // identity; the launch directory is the fallback for relocated binaries.
    let build_tree = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = vec![build_tree.clone()];
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    let mut source = None;
    for tree in &candidates {
        if let Some(revision) = git_head(tree) {
            source = Some((revision, format!("git:{}", tree.display()), git_dirty(tree)));
            break;
        }
    }
    let (source_revision, source_revision_origin, source_dirty) = source
        .or_else(|| {
            std::env::var("TOFY_SOURCE_REVISION")
                .ok()
                .map(|revision| revision.trim().to_string())
                .filter(|revision| !revision.is_empty())
                .map(|revision| (revision, "env:TOFY_SOURCE_REVISION".to_string(), None))
        })
        .unwrap_or_else(|| (UNKNOWN_PROVENANCE.into(), UNKNOWN_PROVENANCE.into(), None));
    let candle_graph = build_tree.join("..").join("candle_graph");
    let (candle_graph_revision, candle_graph_dirty) = match git_head(&candle_graph) {
        Some(revision) => (revision, git_dirty(&candle_graph)),
        None => (UNKNOWN_PROVENANCE.into(), None),
    };
    LaunchProvenance {
        source_revision,
        source_revision_origin,
        source_dirty,
        binary_path,
        binary_sha256,
        candle_graph_revision,
        candle_graph_dirty,
    }
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
        .args(["status", "--porcelain", "--untracked-files=no"])
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
    ] {
        push_artifact(&mut artifacts, &canonical_run_dir, role, &path, false)?;
    }
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
    let bundle_sha256 = bundle_sha256(&artifacts, gradient_pressure.as_ref());
    let mut gaps = Vec::new();
    if !launch.source_revision_known() {
        gaps.push(
            "source revision unavailable (no git tree found); set TOFY_SOURCE_REVISION".into(),
        );
    }
    if launch.source_dirty == Some(true) {
        gaps.push("source tree had uncommitted changes at launch".into());
    }
    if launch.candle_graph_revision == UNKNOWN_PROVENANCE {
        gaps.push("sibling candle_graph revision unavailable".into());
    }
    if matches!(report.profile, ProfileState::Pending) {
        gaps.push("representative-update profile was not published".into());
    }
    let identity = run_identity(run_dir, report, &mut gaps);
    let resume_count = report.resume_count;
    if report.run_attempts.is_empty() {
        gaps.push("run attempt history unavailable (report predates attempt records)".into());
    } else if let Some(repaired) = report
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
    hash.update(b"tofy.p2.evidence.bundle.v1\0");
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
                    provenance: LaunchProvenance::unknown(&binary),
                    loss_log_repair: None,
                },
                RunAttempt {
                    attempt: 2,
                    kind: RunAttemptKind::Resume,
                    started_unix_secs: 20,
                    pid: 2,
                    resumed_from: Some(checkpoint.clone()),
                    resumed_step: Some(7),
                    provenance: LaunchProvenance::unknown(&binary),
                    loss_log_repair: None,
                },
            ],
        };
        write(
            &root.join("train_report.json"),
            &serde_json::to_vec_pretty(&report)?,
        );

        let launch = LaunchProvenance {
            source_revision: "deadbeef".into(),
            source_revision_origin: "git:test".into(),
            source_dirty: Some(false),
            binary_path: binary.clone(),
            binary_sha256: hash_file(&binary)?.1,
            candle_graph_revision: "cafef00d".into(),
            candle_graph_dirty: Some(false),
        };
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
        assert_eq!(manifest.artifacts.len(), 15);
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
        assert!(manifest.bundle_sha256.starts_with("sha256:"));
        assert_eq!(manifest.gaps.len(), 1, "{:?}", manifest.gaps);
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
                first.source_revision_origin.starts_with("git:")
                    || first.source_revision_origin.starts_with("env:")
            );
        } else {
            assert_eq!(first.source_revision_origin, UNKNOWN_PROVENANCE);
        }
        let json = serde_json::to_value(first).unwrap();
        assert!(json["source_revision"].is_string());
        assert!(json["binary_sha256"].is_string());
    }
}
