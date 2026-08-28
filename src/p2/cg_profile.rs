//! One representative optimizer-update evidence bundle for candle-graph.

use anyhow::{Context, Result};
use candle_core::{backprop::GradStore, Device, Tensor};
use candle_graph::candle::{self, CandleCapture, GradientCapturePlan};
use candle_graph::{
    reconcile_published_bundle, CaptureBegin, CaptureContract, CaptureRun, CoverageLevel,
    ExecutionStep, GradientFamilyContract, MeasurementScope, ProfileRun, PublicationReceipt,
    SpanId, SpanKind,
};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::perf::NvtxRange;

pub(crate) const PROFILE_ENTRYPOINT: &str = "tofy::p2::train::optimizer_update";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientClipState {
    PreClip,
    PostClip,
}

impl GradientClipState {
    fn root(self) -> &'static str {
        match self {
            Self::PreClip => "vb/pre_clip",
            Self::PostClip => "vb/post_clip",
        }
    }
}

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
    capture: Option<CaptureRun>,
    reconciled: Option<PublicationReceipt>,
    gradient_plan: Option<GradientCapturePlan>,
    update: Option<u64>,
    correlation_id: String,
    failure_reason: RefCell<Option<String>>,
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
    pub varmap: &'a VarMap,
    pub gradient_clip_state: GradientClipState,
}

impl RepresentativeUpdateCapture {
    pub fn begin(spec: CaptureSpec<'_>) -> Result<Self> {
        let update = spec.completed_updates.saturating_add(1);
        if update != spec.selected_update || !matches!(spec.state, ProfileState::Pending) {
            return Ok(Self::inactive());
        }
        let correlation_id = format!("tofy.p2/update-{update:012}");
        let vars = spec
            .varmap
            .data()
            .lock()
            .expect("VarMap lock poisoned")
            .iter()
            .map(|(key, var)| (key.clone(), var.clone()))
            .collect::<Vec<_>>();
        let families = vars
            .iter()
            .map(|(key, _)| gradient_family(key))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|family| GradientFamilyContract::data_conditional(family, 1))
            .collect();
        let gradient_plan = GradientCapturePlan::from_named_vars(
            spec.gradient_clip_state.root(),
            vars,
            gradient_family,
            families,
        )?;
        let required_semantic_labels = required_semantic_labels(&correlation_id, spec.lesson);
        let (gpu_expected_semantic_labels, cpu_only_semantic_labels) =
            if spec.measured_region_device_synchronized {
                (required_semantic_labels.clone(), Vec::new())
            } else {
                (Vec::new(), required_semantic_labels.clone())
            };
        let contract = CaptureContract {
            // Synchronization and mechanism probes deliberately add work to the
            // selected update, so this is representative diagnostic work rather
            // than an uninstrumented production-equivalent timing claim.
            measurement_scope: MeasurementScope::ProfiledWork,
            operations: CoverageLevel::None,
            tensors: CoverageLevel::Partial,
            gradients: CoverageLevel::Complete,
            gradient_contract: Some(gradient_plan.contract().clone()),
            logical_memory: CoverageLevel::None,
            physical_memory: CoverageLevel::None,
            device_timing: CoverageLevel::None,
            required_semantic_labels,
            gpu_expected_semantic_labels,
            cpu_only_semantic_labels,
        };
        let mut run = planned_profile_run(update, spec.device)
            .capture_contract(contract)
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
        let destination = profile_destination(spec.output_dir, update);
        match CaptureRun::begin(destination, run)? {
            CaptureBegin::AlreadyPublished(receipt) => Ok(Self {
                capture: None,
                reconciled: Some(*receipt),
                gradient_plan: None,
                update: Some(update),
                correlation_id,
                failure_reason: RefCell::new(None),
            }),
            CaptureBegin::Active(capture) => Ok(Self {
                capture: Some(capture),
                reconciled: None,
                gradient_plan: Some(gradient_plan),
                update: Some(update),
                correlation_id,
                failure_reason: RefCell::new(None),
            }),
        }
    }

    fn inactive() -> Self {
        Self {
            capture: None,
            reconciled: None,
            gradient_plan: None,
            update: None,
            correlation_id: String::new(),
            failure_reason: RefCell::new(None),
        }
    }

    pub fn active(&self) -> bool {
        self.capture.is_some()
    }

    pub fn measurement(&self) -> Option<ProfileRange<'_>> {
        let session = self.capture.as_ref()?.session();
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
        let session = self.capture.as_ref()?.session();
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

    /// Run one profiled phase with CUDA synchronized at both ends.
    ///
    /// Candle operations enqueue asynchronously, so an ordinary host span can
    /// charge a module's device work to the next scalar readback. This interface
    /// is deliberately active only for the selected representative update: it
    /// gives the evidence packet device-complete module spans without slowing
    /// the rest of training.
    pub fn synchronized_phase<T>(
        &self,
        device: &Device,
        name: &str,
        kind: SpanKind,
        step: Option<ExecutionStep>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.synchronized_phase_with_range(device, name, kind, step, |_| f())
    }

    pub fn synchronized_phase_with_range<T>(
        &self,
        device: &Device,
        name: &str,
        kind: SpanKind,
        step: Option<ExecutionStep>,
        f: impl FnOnce(Option<&ProfileRange<'_>>) -> Result<T>,
    ) -> Result<T> {
        self.synchronized_phase_with(
            name,
            kind,
            step,
            || {
                if device.is_cuda() {
                    device.synchronize()?;
                }
                Ok(())
            },
            f,
        )
    }

    fn synchronized_phase_with<T>(
        &self,
        name: &str,
        kind: SpanKind,
        step: Option<ExecutionStep>,
        mut synchronize: impl FnMut() -> Result<()>,
        f: impl FnOnce(Option<&ProfileRange<'_>>) -> Result<T>,
    ) -> Result<T> {
        if !self.active() {
            return f(None);
        }
        synchronize()?;
        let range = self.phase(name, kind, step);
        let result = f(range.as_ref());
        let final_sync = synchronize();
        drop(range);
        if let Err(error) = &result {
            self.failure_reason.replace(Some(format!("{error:#}")));
        } else if let Err(error) = &final_sync {
            self.failure_reason.replace(Some(format!("{error:#}")));
        }
        match (result, final_sync) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    pub fn record_tensor(
        &self,
        range: &ProfileRange<'_>,
        name: &str,
        tensor: &Tensor,
        step: Option<ExecutionStep>,
    ) -> Result<()> {
        let (Some(capture_run), Some(guard)) = (self.capture.as_ref(), range._candle.as_ref())
        else {
            return Ok(());
        };
        let capture = CandleCapture::from_tensor(tensor, step).with_label(name);
        candle::record_tensor(capture_run.session(), guard.id(), &capture)
    }

    pub fn record_tensor_stats(
        &self,
        range: &ProfileRange<'_>,
        label: &str,
        tensor: &Tensor,
    ) -> Result<()> {
        let (Some(capture), Some(guard)) = (self.capture.as_ref(), range._candle.as_ref()) else {
            return Ok(());
        };
        capture
            .session()
            .record_tensor_stats(guard.id(), label, tensor)
    }

    pub fn record_scalar(&self, span_id: Option<SpanId>, label: &str, value: f64) -> Result<()> {
        let (Some(capture), Some(span_id)) = (self.capture.as_ref(), span_id) else {
            return Ok(());
        };
        capture.session().record_scalar(span_id, label, value)
    }

    pub fn record_gradients(&self, grads: &GradStore) -> Result<()> {
        let (Some(capture), Some(plan)) = (self.capture.as_ref(), self.gradient_plan.as_ref())
        else {
            return Ok(());
        };
        plan.record(capture.session(), grads)
    }

    pub fn finish(mut self) -> Result<Option<ProfileArtifacts>> {
        let receipt = if let Some(receipt) = self.reconciled.take() {
            receipt
        } else if let Some(capture) = self.capture.take() {
            capture.publish()?
        } else {
            return Ok(None);
        };
        Ok(Some(profile_artifacts(
            self.update.expect("published capture update"),
            &receipt,
        )))
    }

    pub fn finish_failed(mut self, reason: impl Into<String>) -> Result<Option<ProfileArtifacts>> {
        let receipt = if let Some(receipt) = self.reconciled.take() {
            receipt
        } else if let Some(capture) = self.capture.take() {
            capture.publish_failed(reason)?
        } else {
            return Ok(None);
        };
        Ok(Some(profile_artifacts(
            self.update.expect("published capture update"),
            &receipt,
        )))
    }
}

impl ProfileRange<'_> {
    pub fn span_id(&self) -> Option<SpanId> {
        self._candle.as_ref().map(|guard| guard.id())
    }
}

impl Drop for RepresentativeUpdateCapture {
    fn drop(&mut self) {
        let Some(capture) = self.capture.take() else {
            return;
        };
        // Every `?` that exits a selected training step before `finish` reaches
        // this guard, so caught step failures become verified failed bundles
        // instead of unterminated staging traces.
        let fallback = if std::thread::panicking() {
            "training step panicked before profile publication"
        } else {
            "training step exited before profile publication"
        };
        let reason = self
            .failure_reason
            .get_mut()
            .take()
            .unwrap_or_else(|| fallback.to_owned());
        if let Err(error) = capture.publish_failed(reason) {
            tracing::error!("failed to publish diagnostic candle-graph bundle: {error:#}");
        }
    }
}

pub fn reconcile_profile_bundle(
    output_dir: &Path,
    update: u64,
    device: &str,
) -> Result<Option<ProfileArtifacts>> {
    let destination = profile_destination(output_dir, update);
    let run = planned_profile_run(update, device);
    reconcile_published_bundle(&destination, &run)
        .with_context(|| format!("reconcile profile bundle {}", destination.display()))
        .map(|receipt| receipt.map(|receipt| profile_artifacts(update, &receipt)))
}

fn planned_profile_run(update: u64, device: &str) -> ProfileRun {
    ProfileRun::training(PROFILE_ENTRYPOINT, update, device)
        .correlation_id(format!("tofy.p2/update-{update:012}"))
}

fn profile_destination(output_dir: &Path, update: u64) -> PathBuf {
    output_dir
        .join("profile")
        .join(format!("update-{update:012}"))
}

fn profile_artifacts(update: u64, receipt: &PublicationReceipt) -> ProfileArtifacts {
    let directory = receipt.bundle_path.clone();
    ProfileArtifacts {
        update,
        trace: directory.join("trace.jsonl"),
        evidence_json: directory.join("evidence.json"),
        evidence_markdown: directory.join("report.md"),
        viewer_html: directory.join("viewer.html"),
        nsight_directory: directory.join("nsight"),
        directory,
    }
}

fn gradient_family(key: &str) -> String {
    let prefix = key.split('.').next().unwrap_or(key);
    match prefix {
        "exact_grounding_head" => "exact_decoder",
        "event_head" | "q_head" | "reliability_head" | "consumer_readout" => "observers",
        "action_decoder"
        | "coordinate_decoder"
        | "grounding_head"
        | "prefix_head"
        | "spatial_prefix_head" => "auxiliary_decoders",
        _ => "world",
    }
    .to_owned()
}

fn required_semantic_labels(correlation_id: &str, lesson: &str) -> Vec<String> {
    let phases: &[&str] = if lesson == "foundation_v2" {
        &[
            "forward_loss",
            "loss_tensors",
            "backward",
            "gradients",
            "gradient_clip",
            "optimizer",
        ]
    } else {
        &[
            "loss_readback",
            "gradient_clip",
            "gradients",
            "optimizer",
            "metrics",
        ]
    };
    std::iter::once(correlation_id.to_owned())
        .chain(
            phases
                .iter()
                .map(|phase| format!("{correlation_id}/{phase}")),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use std::fs;

    #[test]
    fn selected_update_is_one_based() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("tofy-profile-selection-{}", std::process::id()));
        let pending = ProfileState::Pending;
        let varmap = VarMap::new();
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
            varmap: &varmap,
            gradient_clip_state: GradientClipState::PostClip,
        })?;
        assert!(!inactive.active());
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn synchronized_phase_brackets_body_even_when_body_fails() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "tofy-profile-synchronized-phase-{}",
            std::process::id()
        ));
        let pending = ProfileState::Pending;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _linear = candle_nn::linear(1, 1, vb.pp("test"))?;
        let capture = RepresentativeUpdateCapture::begin(CaptureSpec {
            completed_updates: 0,
            selected_update: 1,
            state: &pending,
            output_dir: &dir,
            device: "cpu",
            measured_region_device_synchronized: false,
            lesson: "test",
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 16,
            inner_steps: 1,
            outer_steps: 1,
            precision: "f32",
            varmap: &varmap,
            gradient_clip_state: GradientClipState::PostClip,
        })?;
        let events = RefCell::new(Vec::new());
        let error = capture
            .synchronized_phase_with(
                "test_phase",
                SpanKind::Module,
                None,
                || {
                    events.borrow_mut().push("sync");
                    Ok(())
                },
                |_| -> Result<()> {
                    events.borrow_mut().push("body");
                    bail!("expected body failure")
                },
            )
            .expect_err("body failure must propagate");
        assert!(error.to_string().contains("expected body failure"));
        assert_eq!(*events.borrow(), ["sync", "body", "sync"]);
        drop(capture);
        let bundle = dir.join("profile/update-000000000001");
        candle_graph::verify_bundle(&bundle)?;
        let trace = candle_graph::parse_trace(bundle.join("trace.jsonl"))?;
        assert_eq!(
            trace.terminal.outcome,
            candle_graph::trace::RunOutcome::Failed
        );
        assert!(trace
            .terminal
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("expected body failure")));
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }
}
