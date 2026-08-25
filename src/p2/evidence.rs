//! Relocatable, hash-bound evidence manifest for one P2 training run.

use crate::p2::cg_profile::ProfileState;
use crate::p2::experiment::ResolvedExperiment;
use crate::p2::train::{GradientPressureDiagnostics, TrainReport, TrainStatus};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVIDENCE_MANIFEST_SCHEMA: &str = "tofy/p2/evidence/1";
pub const EVIDENCE_MANIFEST_FILE: &str = "evidence_manifest.json";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
struct EvidenceManifest {
    schema: String,
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

#[derive(Debug, Serialize, Deserialize)]
struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    source_revision: Option<String>,
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
    let source_revision = std::env::var("TOFY_SOURCE_REVISION")
        .ok()
        .filter(|revision| !revision.trim().is_empty());
    publish_training_evidence_with_provenance(run_dir, report, &binary, source_revision)
}

fn publish_training_evidence_with_provenance(
    run_dir: &Path,
    report: &TrainReport,
    binary: &Path,
    source_revision: Option<String>,
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

    let pressure_samples = &report.gradient_pressure_samples;
    let gradient_pressure = (!pressure_samples.is_empty())
        .then(|| bind_gradient_pressure(pressure_samples))
        .transpose()?;
    let bundle_sha256 = bundle_sha256(&artifacts, gradient_pressure.as_ref());
    let mut gaps = Vec::new();
    if source_revision.is_none() {
        gaps.push("source revision unavailable; set TOFY_SOURCE_REVISION".into());
    }
    if matches!(report.profile, ProfileState::Pending) {
        gaps.push("representative-update profile was not published".into());
    }
    let manifest = EvidenceManifest {
        schema: EVIDENCE_MANIFEST_SCHEMA.into(),
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
        provenance: Provenance { source_revision },
        gradient_pressure,
        artifacts,
        bundle_sha256,
        gaps,
    };
    let path = run_dir.join(EVIDENCE_MANIFEST_FILE);
    write_json_atomic(&path, &manifest)?;
    Ok(path)
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
            (profile_dir.join("application.jsonl"), b"trace".as_slice()),
            (profile_dir.join("evidence.json"), b"evidence".as_slice()),
            (profile_dir.join("EVIDENCE.md"), b"markdown".as_slice()),
            (profile_dir.join("viewer.html"), b"viewer".as_slice()),
            (
                profile_dir.join("nsight/status.txt"),
                b"available".as_slice(),
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
            latest_checkpoint: checkpoint,
            resumed_from: None,
            batch_schedule_migrations: vec![],
            checkpoint: root.join("model.safetensors"),
            export_checkpoint: None,
            config_path: root.join("config.json"),
            profile: ProfileState::Published(ProfileArtifacts {
                update: 2,
                directory: profile_dir.clone(),
                trace: profile_dir.join("application.jsonl"),
                evidence_json: profile_dir.join("evidence.json"),
                evidence_markdown: profile_dir.join("EVIDENCE.md"),
                viewer_html: profile_dir.join("viewer.html"),
                nsight_directory: profile_dir.join("nsight"),
            }),
            gradient_pressure: Some(pressure.clone()),
            gradient_pressure_samples: vec![pressure],
            foundation_v2: None,
            research_claim: false,
        };
        write(
            &root.join("train_report.json"),
            &serde_json::to_vec_pretty(&report)?,
        );

        let path = publish_training_evidence_with_provenance(
            &root,
            &report,
            &binary,
            Some("deadbeef".into()),
        )?;
        let first_publication = fs::read(&path)?;
        publish_training_evidence_with_provenance(
            &root,
            &report,
            &binary,
            Some("deadbeef".into()),
        )?;
        assert_eq!(fs::read(&path)?, first_publication);
        let manifest: EvidenceManifest = serde_json::from_reader(File::open(&path)?)?;

        assert_eq!(manifest.schema, EVIDENCE_MANIFEST_SCHEMA);
        assert_eq!(manifest.artifacts.len(), 9);
        assert!(manifest.artifacts.iter().any(|artifact| {
            artifact.role == "checkpoint_optimizer"
                && artifact.path == "checkpoints/step-000000000007/optimizer.safetensors"
                && artifact.sha256 == hash_file(&root.join(&artifact.path)).unwrap().1
        }));
        assert!(manifest
            .artifacts
            .iter()
            .all(|artifact| artifact.path != EVIDENCE_MANIFEST_FILE));
        assert_eq!(manifest.gradient_pressure.unwrap().updates, vec![3]);
        assert!(manifest.bundle_sha256.starts_with("sha256:"));
        assert!(manifest.gaps.is_empty());
        let mut mismatched_report = report.clone();
        mismatched_report.global_step += 1;
        assert!(publish_training_evidence_with_provenance(
            &root,
            &mismatched_report,
            &binary,
            Some("deadbeef".into()),
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
}
