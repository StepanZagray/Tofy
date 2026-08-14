//! One representative optimizer-update evidence bundle for candle-graph.

use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Tensor};
use candle_graph::instrument::candle::{self, CandleCapture};
use candle_graph::trace::schema::GradientState;
use candle_graph::{ExecutionStep, ProfileRun, SpanKind, TraceSession};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::perf::NvtxRange;

const ENTRYPOINT: &str = "tofy::p2::train::optimizer_update";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileArtifacts {
    pub update: u64,
    pub directory: PathBuf,
    pub trace: PathBuf,
    pub evidence_json: PathBuf,
    pub evidence_markdown: PathBuf,
    pub viewer_html: PathBuf,
    pub nsight_directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileState {
    Pending,
    Published(ProfileArtifacts),
}

pub struct RepresentativeUpdateCapture {
    session: Option<TraceSession>,
    staging_dir: Option<PathBuf>,
    final_dir: Option<PathBuf>,
    artifacts: Option<ProfileArtifacts>,
    correlation_id: String,
}

pub struct ProfileRange<'a> {
    _candle: Option<candle_graph::SpanGuard<'a>>,
    _nvtx: NvtxRange,
}

pub struct CaptureSpec<'a> {
    pub completed_updates: u64,
    pub selected_update: u64,
    pub state: &'a ProfileState,
    pub output_dir: &'a Path,
    pub device: &'a str,
    pub measured_region_device_synchronized: bool,
    pub lesson: &'a str,
    pub physical_batch: usize,
    pub grad_accum: usize,
    pub hidden_dim: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    pub precision: &'a str,
}

impl RepresentativeUpdateCapture {
    pub fn begin(spec: CaptureSpec<'_>) -> Result<Self> {
        let update = spec.completed_updates.saturating_add(1);
        if update != spec.selected_update || !matches!(spec.state, ProfileState::Pending) {
            return Ok(Self::inactive());
        }
        let profile_root = spec.output_dir.join("profile");
        let final_dir = profile_root.join(format!("update-{update:012}"));
        if final_dir.exists() {
            bail!(
                "profile bundle already exists while state is pending: {}",
                final_dir.display()
            );
        }
        fs::create_dir_all(&profile_root)
            .with_context(|| format!("create {}", profile_root.display()))?;
        let staging_dir = profile_root.join(format!(
            ".update-{update:012}.staging-{}",
            std::process::id()
        ));
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir).with_context(|| {
                format!("remove stale staging bundle {}", staging_dir.display())
            })?;
        }
        fs::create_dir_all(&staging_dir)
            .with_context(|| format!("create {}", staging_dir.display()))?;

        let correlation_id = format!("tofy.p2/update-{update:012}");
        let mut run = ProfileRun::training(ENTRYPOINT, update, spec.device)
            .correlation_id(correlation_id.clone())
            .tag("lesson", spec.lesson)
            .tag("physical_batch", spec.physical_batch.to_string())
            .tag("grad_accum", spec.grad_accum.to_string())
            .tag(
                "effective_batch",
                spec.physical_batch
                    .saturating_mul(spec.grad_accum)
                    .to_string(),
            )
            .tag("hidden_dim", spec.hidden_dim.to_string())
            .tag("inner_steps", spec.inner_steps.to_string())
            .tag("outer_steps", spec.outer_steps.to_string())
            .tag("precision", spec.precision);
        if spec.measured_region_device_synchronized {
            run = run.measured_region_device_synchronized();
        }
        if let Ok(revision) = std::env::var("TOFY_SOURCE_REVISION") {
            run = run.tag("source_revision", revision);
        }
        let trace = staging_dir.join("application.jsonl");
        let session = TraceSession::open(&trace, run)?;
        let artifacts = ProfileArtifacts {
            update,
            directory: final_dir.clone(),
            trace: final_dir.join("application.jsonl"),
            evidence_json: final_dir.join("evidence.json"),
            evidence_markdown: final_dir.join("EVIDENCE.md"),
            viewer_html: final_dir.join("viewer.html"),
            nsight_directory: final_dir.join("nsight"),
        };
        Ok(Self {
            session: Some(session),
            staging_dir: Some(staging_dir),
            final_dir: Some(final_dir),
            artifacts: Some(artifacts),
            correlation_id,
        })
    }

    fn inactive() -> Self {
        Self {
            session: None,
            staging_dir: None,
            final_dir: None,
            artifacts: None,
            correlation_id: String::new(),
        }
    }

    pub fn active(&self) -> bool {
        self.session.is_some()
    }

    pub fn measurement(&self) -> Option<ProfileRange<'_>> {
        let session = self.session.as_ref()?;
        Some(ProfileRange {
            _candle: Some(session.begin_measurement(self.correlation_id.clone())),
            _nvtx: NvtxRange::new(&self.correlation_id),
        })
    }

    pub fn phase(
        &self,
        name: &str,
        kind: SpanKind,
        step: Option<ExecutionStep>,
    ) -> Option<ProfileRange<'_>> {
        let session = self.session.as_ref()?;
        let label = format!("{}/{}", self.correlation_id, name);
        let candle = match step {
            Some(step) => session.begin_step_span(label.clone(), step, kind),
            None => session.begin_span(label.clone(), kind),
        };
        Some(ProfileRange {
            _candle: Some(candle),
            _nvtx: NvtxRange::new(&label),
        })
    }

    pub fn record_tensor(
        &self,
        range: &ProfileRange<'_>,
        name: &str,
        tensor: &Tensor,
        step: Option<ExecutionStep>,
    ) -> Result<()> {
        let (Some(session), Some(guard)) = (self.session.as_ref(), range._candle.as_ref()) else {
            return Ok(());
        };
        let mut capture = CandleCapture::from_tensor(tensor, step);
        capture.tensor_id = name.to_string();
        candle::record_tensor(session, guard.id(), &capture)
    }

    pub fn record_gradients(&self, varmap: &VarMap, grads: &GradStore) -> Result<()> {
        let Some(session) = self.session.as_ref() else {
            return Ok(());
        };
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<_> = data.keys().cloned().collect();
        names.sort();
        let mut entries = Vec::with_capacity(names.len());
        let mut norm_tensors = Vec::new();
        for name in names {
            let var = data
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?;
            let norm_index = if let Some(grad) = grads.get(var.as_tensor()) {
                let norm = grad
                    .to_dtype(DType::F32)?
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .reshape(1)?;
                norm_tensors.push(norm);
                Some(norm_tensors.len() - 1)
            } else {
                None
            };
            entries.push((name, norm_index));
        }
        let norms = if norm_tensors.is_empty() {
            Vec::new()
        } else {
            Tensor::cat(&norm_tensors, 0)?.to_vec1::<f32>()?
        };
        for (name, norm_index) in entries {
            let (state, norm) = match norm_index {
                None => (GradientState::Missing, None),
                Some(index) if !norms[index].is_finite() => (GradientState::NonFinite, None),
                Some(index) if norms[index] == 0.0 => (GradientState::Zero, None),
                Some(index) => (GradientState::Present, Some(norms[index] as f64)),
            };
            session.record_gradient("vb", name, state, norm)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Option<ProfileArtifacts>> {
        let Some(session) = self.session.take() else {
            return Ok(None);
        };
        let staging = self.staging_dir.take().expect("active capture staging dir");
        let final_dir = self.final_dir.take().expect("active capture final dir");
        let artifacts = self.artifacts.take().expect("active capture artifacts");
        let trace = session.finish()?;
        let evidence = candle_graph::build_evidence(&trace, None, None)?;
        fs::write(
            staging.join("evidence.json"),
            serde_json::to_vec_pretty(&evidence)?,
        )?;
        fs::write(staging.join("EVIDENCE.md"), evidence.markdown())?;
        fs::write(
            staging.join("viewer.html"),
            candle_graph::viewer::render_evidence_html(&evidence),
        )?;
        fs::create_dir_all(staging.join("nsight"))?;
        fs::rename(&staging, &final_dir).with_context(|| {
            format!(
                "publish profile bundle {} -> {}",
                staging.display(),
                final_dir.display()
            )
        })?;
        Ok(Some(artifacts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_update_is_one_based() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("tofy-profile-selection-{}", std::process::id()));
        let pending = ProfileState::Pending;
        let inactive = RepresentativeUpdateCapture::begin(CaptureSpec {
            completed_updates: 0,
            selected_update: 2,
            state: &pending,
            output_dir: &dir,
            device: "cpu",
            measured_region_device_synchronized: false,
            lesson: "dynamics",
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 16,
            inner_steps: 1,
            outer_steps: 1,
            precision: "f32",
        })?;
        assert!(!inactive.active());
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }
}
