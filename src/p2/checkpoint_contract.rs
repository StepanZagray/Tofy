//! Fail-closed checkpoint requirements for the Phase A controller.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const EVIDENCE_MANIFEST_FILE: &str = "evidence_manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseABundleRequirements {
    pub selected_ema_sha256: String,
    pub config_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPhaseABundle {
    pub bundle: PathBuf,
    pub source_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointRejection {
    EvidenceManifestMissing,
    EvidenceManifestUnreadable,
    MissingSelectedEma,
    MissingConfig,
    MissingSourceRevision,
    ArtifactPathUnsafe { role: String },
    ArtifactMissing { role: String },
    ArtifactHashMismatch { role: String },
    SelectedEmaHashMismatch,
    ConfigHashMismatch,
    NotLoadable,
}

impl fmt::Display for CheckpointRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidenceManifestMissing => write!(f, "evidence manifest is missing"),
            Self::EvidenceManifestUnreadable => write!(f, "evidence manifest is unreadable"),
            Self::MissingSelectedEma => write!(f, "selected EMA artifact is missing"),
            Self::MissingConfig => write!(f, "training config artifact is missing"),
            Self::MissingSourceRevision => write!(f, "source revision is missing"),
            Self::ArtifactPathUnsafe { role } => write!(f, "artifact path for {role} is unsafe"),
            Self::ArtifactMissing { role } => write!(f, "artifact for {role} is missing"),
            Self::ArtifactHashMismatch { role } => {
                write!(f, "artifact hash for {role} does not match")
            }
            Self::SelectedEmaHashMismatch => write!(f, "selected EMA hash does not match contract"),
            Self::ConfigHashMismatch => write!(f, "config hash does not match contract"),
            Self::NotLoadable => write!(f, "bundle failed the loadability gate"),
        }
    }
}

impl std::error::Error for CheckpointRejection {}

pub fn verify_phase_a_bundle(
    bundle: &Path,
    requirements: &PhaseABundleRequirements,
    loadable: bool,
) -> Result<VerifiedPhaseABundle, CheckpointRejection> {
    verify_phase_a_bundle_with(bundle, requirements, || loadable)
}

/// Verifies every recorded byte before invoking the final, model-owning loadability seam.
pub fn verify_phase_a_bundle_with<F>(
    bundle: &Path,
    requirements: &PhaseABundleRequirements,
    loadable: F,
) -> Result<VerifiedPhaseABundle, CheckpointRejection>
where
    F: FnOnce() -> bool,
{
    let manifest_path = bundle.join(EVIDENCE_MANIFEST_FILE);
    if !manifest_path.is_file() {
        return Err(CheckpointRejection::EvidenceManifestMissing);
    }
    let manifest: Value = serde_json::from_slice(
        &fs::read(manifest_path).map_err(|_| CheckpointRejection::EvidenceManifestUnreadable)?,
    )
    .map_err(|_| CheckpointRejection::EvidenceManifestUnreadable)?;
    let selected =
        artifact(&manifest, "selected_ema")?.ok_or(CheckpointRejection::MissingSelectedEma)?;
    if selected.sha256 != requirements.selected_ema_sha256 {
        return Err(CheckpointRejection::SelectedEmaHashMismatch);
    }
    verify_artifact(bundle, &selected)?;
    let source_revision = manifest
        .pointer("/provenance/source_revision")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or(CheckpointRejection::MissingSourceRevision)?
        .into();
    let config = artifact(&manifest, "train_config")?.ok_or(CheckpointRejection::MissingConfig)?;
    if config.sha256 != requirements.config_sha256 {
        return Err(CheckpointRejection::ConfigHashMismatch);
    }
    verify_artifact(bundle, &config)?;
    if !loadable() {
        return Err(CheckpointRejection::NotLoadable);
    }
    Ok(VerifiedPhaseABundle {
        bundle: bundle.to_path_buf(),
        source_revision,
    })
}

#[derive(Debug)]
struct ManifestArtifact {
    role: String,
    path: String,
    sha256: String,
}

fn artifact(
    manifest: &Value,
    required_role: &str,
) -> Result<Option<ManifestArtifact>, CheckpointRejection> {
    let Some(artifacts) = manifest.get("artifacts").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(value) = artifacts
        .iter()
        .find(|artifact| artifact.get("role").and_then(Value::as_str) == Some(required_role))
    else {
        return Ok(None);
    };
    let role = value
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or(required_role)
        .to_owned();
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| CheckpointRejection::ArtifactPathUnsafe { role: role.clone() })?
        .to_owned();
    let sha256 = value
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| CheckpointRejection::ArtifactHashMismatch { role: role.clone() })?
        .to_owned();
    Ok(Some(ManifestArtifact { role, path, sha256 }))
}

fn verify_artifact(bundle: &Path, artifact: &ManifestArtifact) -> Result<(), CheckpointRejection> {
    let path = Path::new(&artifact.path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(CheckpointRejection::ArtifactPathUnsafe {
            role: artifact.role.clone(),
        });
    }
    let path = bundle.join(path);
    if !path.is_file() {
        return Err(CheckpointRejection::ArtifactMissing {
            role: artifact.role.clone(),
        });
    }
    if sha256_file(&path).map_err(|_| CheckpointRejection::ArtifactHashMismatch {
        role: artifact.role.clone(),
    })? != artifact.sha256
    {
        return Err(CheckpointRejection::ArtifactHashMismatch {
            role: artifact.role.clone(),
        });
    }
    Ok(())
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (PathBuf, PhaseABundleRequirements) {
        let root = std::env::temp_dir().join(format!(
            "tofy-phase-a-contract-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("ema.safetensors"), b"ema").unwrap();
        fs::write(root.join("config.json"), b"config").unwrap();
        let ema = sha256_file(&root.join("ema.safetensors")).unwrap();
        let config = sha256_file(&root.join("config.json")).unwrap();
        let manifest = serde_json::json!({
            "provenance": { "source_revision": "c0bfb532" },
            "artifacts": [
                { "role": "selected_ema", "path": "ema.safetensors", "sha256": ema },
                { "role": "train_config", "path": "config.json", "sha256": config }
            ]
        });
        fs::write(
            root.join(EVIDENCE_MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        (
            root,
            PhaseABundleRequirements {
                selected_ema_sha256: ema,
                config_sha256: config,
            },
        )
    }

    #[test]
    fn matching_fixture_passes() {
        let (root, requirements) = fixture();
        let verified = verify_phase_a_bundle(&root, &requirements, true).unwrap();
        assert_eq!(verified.source_revision, "c0bfb532");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn altered_artifact_fails_before_loadability() {
        let (root, requirements) = fixture();
        fs::write(root.join("ema.safetensors"), b"flipped").unwrap();
        let called = std::cell::Cell::new(false);
        let rejection = verify_phase_a_bundle_with(&root, &requirements, || {
            called.set(true);
            true
        })
        .unwrap_err();
        assert!(matches!(
            rejection,
            CheckpointRejection::ArtifactHashMismatch { .. }
        ));
        assert!(!called.get());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrong_contract_hash_fails_before_loadability() {
        let (root, mut requirements) = fixture();
        requirements.selected_ema_sha256 = "sha256:wrong".into();
        let called = std::cell::Cell::new(false);
        let rejection = verify_phase_a_bundle_with(&root, &requirements, || {
            called.set(true);
            true
        })
        .unwrap_err();
        assert_eq!(rejection, CheckpointRejection::SelectedEmaHashMismatch);
        assert!(!called.get());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "the imported handoff is intentionally rejected: selected EMA sha256:117f... and source revision are absent"]
    fn imported_foundation_v2_handoff_is_rejected_today() {
        let requirements = PhaseABundleRequirements {
            selected_ema_sha256: "sha256:117f...".into(),
            config_sha256: "sha256:unavailable".into(),
        };
        let bundle =
            Path::new("runs/p2/_pod_handoffs/6zp5oip7tvokfl-20260827-foundation-v2/foundation-v2/");
        assert!(verify_phase_a_bundle(bundle, &requirements, false).is_err());
    }
}
