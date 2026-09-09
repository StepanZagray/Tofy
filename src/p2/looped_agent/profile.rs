//! Selected looped-agent diagnostic captures. Callers synchronize the device before
//! opening the measurement and before dropping it (and any chosen phase boundaries).
//! These guards never synchronize; nested spans are host timings. Tensor/gradient
//! reductions add device work, so the contract is deliberately `ProfiledWork`.
//! CUDA metadata asserts the caller-owned measurement-boundary synchronization;
//! callers must satisfy this precondition even when their measured work fails.

use anyhow::{bail, ensure, Context, Result};
use candle_core::{backprop::GradStore, Device, Tensor};
use candle_graph::candle::{record_tensor_with_label, GradientCapturePlan};
use candle_graph::{
    CaptureBegin, CaptureContract, CaptureRun, CoverageLevel, ExecutionStep,
    GradientFamilyContract, MeasurementScope, ProfileRun, PublicationReceipt, SpanGuard, SpanKind,
    TraceSession,
};
use candle_nn::VarMap;
use std::cell::{Cell, RefCell};
use std::path::Path;

use crate::perf::NvtxRange;

pub struct LoopedCapture {
    capture: Option<CaptureRun>,
    gradients: Option<GradientCapturePlan>,
    label: String,
    measurements: Cell<usize>,
    measuring: Cell<bool>,
    observations: Cell<usize>,
    gradients_recorded: Cell<bool>,
    failure: RefCell<Option<String>>,
}

/// Drop phases before their measurement, then drop the measurement before `finish`.
pub struct LoopedRange<'a> {
    candle: SpanGuard<'a>,
    _nvtx: NvtxRange,
    _tracing: tracing::span::EnteredSpan,
    owner: &'a LoopedCapture,
    measurement: bool,
    step: Option<ExecutionStep>,
}

impl LoopedCapture {
    /// Binder-only work; no vision-core or ordinary vision-head families exist.
    // Keep batch/depth capture identity and input provenance explicit at the call.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_binding(
        destination: &Path,
        step: u64,
        device: &Device,
        vars: Option<&VarMap>,
        physical_batch: usize,
        effective_batch: usize,
        loops: usize,
        input_source: &str,
    ) -> Result<Self> {
        ensure!(
            matches!(input_source, "abstract_effects" | "precomputed_visual"),
            "unknown binder input source"
        );
        ensure!(
            (1..=super::binding::MAX_LOOPS).contains(&loops),
            "invalid binder depth"
        );
        let family = |name: &str| match name.split('.').next()? {
            "input_projection" | "readout_token" => Some("input"),
            "block_0" | "block_1" => Some("shared_core"),
            "policy_head" => Some("policy"),
            _ => None,
        };
        let gradients = vars
            .map(|vars| -> Result<GradientCapturePlan> {
                let mut named = vars
                    .data()
                    .lock()
                    .map_err(|_| anyhow::anyhow!("binder parameter lock poisoned"))?
                    .iter()
                    .map(|(n, v)| (n.clone(), v.clone()))
                    .collect::<Vec<_>>();
                named.sort_by(|a, b| a.0.cmp(&b.0));
                ensure!(
                    named.len() == 29
                        && named.iter().map(|(_, v)| v.elem_count()).sum::<usize>()
                            == super::binding::PARAMETERS
                        && named.iter().all(|(n, _)| family(n).is_some()),
                    "unexpected binder parameters"
                );
                GradientCapturePlan::from_named_vars(
                    "parameters/pre_clip",
                    named,
                    |n| family(n).expect("validated binder family").into(),
                    ["input", "shared_core", "policy"]
                        .into_iter()
                        .map(|name| GradientFamilyContract::active(name, 1))
                        .collect(),
                )
            })
            .transpose()?;
        let run = if vars.is_some() {
            ProfileRun::training(
                "tofy::p2::action_binding::optimizer_update",
                step,
                device_name(device),
            )
        } else {
            ProfileRun::inference(
                "tofy::p2::action_binding::evaluation",
                step,
                device_name(device),
            )
        }
        .correlation_id(format!(
            "tofy.looped/binding-{}-{step:012}",
            if vars.is_some() { "update" } else { "eval" }
        ))
        .tag("workload", "looped-action-binding")
        .tag("input_source", input_source)
        .tag("feature_seam", "action-effect-records")
        .tag("parameter_count", super::binding::PARAMETERS.to_string())
        .tag("shared_blocks", "2")
        .tag("vision_core_forwards", "0");
        Self::open(
            destination,
            run,
            gradients,
            physical_batch,
            effective_batch,
            loops,
        )
    }

    /// Frozen body plus the same spatial policy applied independently to all
    /// seven observed frames. No optimizer work or gradient contract is declared.
    pub fn begin_demonstration_grounding(
        destination: &Path,
        step: u64,
        device: &Device,
        batch: usize,
    ) -> Result<Self> {
        let run = ProfileRun::inference(
            "tofy::p2::demonstration_grounding::evaluation",
            step,
            device_name(device),
        )
        .correlation_id(format!(
            "tofy.looped/demonstration-grounding-eval-{step:012}"
        ))
        .tag("workload", "looped-demonstration-grounding")
        .tag("head_kind", "spatial")
        .tag("feature_seam", "post-final-rms-each-observed-frame")
        .tag("frame_count", "7")
        .tag(
            "frame_order",
            "before0,after0,before1,after1,before2,after2,current",
        )
        .tag("core_forward_batches", "1")
        .tag("spatial_head_forward_batches", "7")
        .tag("optimizer_updates", "0")
        .tag("ordinary_heads", "not_executed")
        .tag("head_recurrence", "none");
        Self::open(destination, run, None, batch, batch, 4)
    }

    /// Feature-only looped body plus trainable spatial policy. A training capture
    /// includes both complete VarMaps so absent ordinary-head gradients are explicit.
    /// `None` selects evaluation without a gradient contract.
    pub fn begin_grounded_policy(
        destination: &Path,
        step: u64,
        device: &Device,
        vars: Option<(&VarMap, &VarMap)>,
        physical_batch: usize,
        effective_batch: usize,
        loops: usize,
    ) -> Result<Self> {
        use super::grounded_policy::{named_parameters, parameter_family};
        let gradients = vars
            .map(|(core, policy)| -> Result<GradientCapturePlan> {
                let named = named_parameters(core, policy)?;
                let families = ["core", "spatial_queries", "spatial_policy"]
                    .into_iter()
                    .map(|family| GradientFamilyContract::active(family, 1))
                    .chain(
                        ["policy", "value", "reward", "dynamics"]
                            .into_iter()
                            .map(GradientFamilyContract::inactive),
                    )
                    .collect();
                GradientCapturePlan::from_named_vars(
                    "parameters/pre_clip",
                    named,
                    |name| {
                        parameter_family(name)
                            .expect("validated parameter family")
                            .into()
                    },
                    families,
                )
            })
            .transpose()?;
        let run = if vars.is_some() {
            ProfileRun::training(
                "tofy::p2::grounded_policy::optimizer_update",
                step,
                device_name(device),
            )
        } else {
            ProfileRun::inference(
                "tofy::p2::grounded_policy::evaluation",
                step,
                device_name(device),
            )
        }
        .correlation_id(format!(
            "tofy.looped/grounded-policy-{}-{step:012}",
            if vars.is_some() { "update" } else { "eval" }
        ))
        .tag("workload", "looped-grounded-policy")
        .tag("head_kind", "spatial")
        .tag("feature_seam", "post-final-rms-current")
        .tag("ordinary_heads", "not_executed")
        .tag("head_recurrence", "none");
        Self::open(
            destination,
            run,
            gradients,
            physical_batch,
            effective_batch,
            loops,
        )
    }

    /// Head-only work on cached tensors: never declares frozen-core gradients.
    pub fn begin_readout(
        destination: &Path,
        step: u64,
        device: &Device,
        vars: Option<&VarMap>,
        batch: usize,
        kind: &str,
    ) -> Result<Self> {
        ensure!(matches!(kind, "spatial" | "cls"), "unknown learned readout");
        let family = |name: &str| match name.split('.').next()? {
            "queries" if kind == "spatial" => Some("readout_queries"),
            "hidden" if kind == "cls" => Some("readout_hidden"),
            "output" => Some("readout_policy"),
            _ => None,
        };
        let gradients = vars
            .map(|vars| -> Result<GradientCapturePlan> {
                let mut named = vars
                    .data()
                    .lock()
                    .map_err(|_| anyhow::anyhow!("readout VarMap lock poisoned"))?
                    .iter()
                    .map(|(key, var)| (key.clone(), var.clone()))
                    .collect::<Vec<_>>();
                named.sort_by(|a, b| a.0.cmp(&b.0));
                ensure!(
                    !named.is_empty() && named.iter().all(|(name, _)| family(name).is_some()),
                    "readout capture contains unexpected or empty parameter population"
                );
                let families = [
                    if kind == "spatial" {
                        "readout_queries"
                    } else {
                        "readout_hidden"
                    },
                    "readout_policy",
                ]
                .into_iter()
                .map(|name| GradientFamilyContract::active(name, 1))
                .collect();
                GradientCapturePlan::from_named_vars(
                    "parameters/pre_clip",
                    named,
                    |name| family(name).expect("validated head family").into(),
                    families,
                )
            })
            .transpose()?;
        let run = if vars.is_some() {
            ProfileRun::training(
                "tofy::p2::learned_readout::optimizer_update",
                step,
                device_name(device),
            )
        } else {
            ProfileRun::inference(
                "tofy::p2::learned_readout::evaluation",
                step,
                device_name(device),
            )
        }
        .correlation_id(format!(
            "tofy.looped/readout-{}-{step:012}",
            if vars.is_some() { "update" } else { "eval" }
        ))
        .tag("workload", "cached-learned-readout")
        .tag("head_kind", kind)
        .tag("executed_core_forwards", "0")
        .tag("cached_extraction_loops", "4")
        .tag("head_recurrence", "none");
        Self::open(destination, run, gradients, batch, batch, 1)
    }

    pub fn begin(
        destination: &Path,
        update: u64,
        device: &Device,
        vars: &VarMap,
        physical_batch: usize,
        effective_batch: usize,
        loops: usize,
    ) -> Result<Self> {
        let named = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("looped VarMap lock poisoned"))?
            .iter()
            .map(|(key, var)| (key.clone(), var.clone()))
            .collect::<Vec<_>>();
        for (key, _) in &named {
            ensure!(
                gradient_family(key).is_some(),
                "unclassified looped parameter {key}"
            );
        }
        let plan = GradientCapturePlan::from_named_vars(
            "parameters/pre_clip",
            named,
            |key| {
                gradient_family(key)
                    .expect("validated parameter family")
                    .into()
            },
            ["core", "policy", "value", "reward", "dynamics"]
                .into_iter()
                .map(|family| GradientFamilyContract::active(family, 1))
                .collect(),
        )?;
        let run = ProfileRun::training(
            "tofy::p2::looped_agent::optimizer_update",
            update,
            device_name(device),
        )
        .correlation_id(format!("tofy.looped/update-{update:012}"));
        Self::open(
            destination,
            run,
            Some(plan),
            physical_batch,
            effective_batch,
            loops,
        )
    }

    /// Evaluation has no gradient contract. Both batch sizes describe the actual
    /// measured workload, including any smaller tail microbatch.
    pub fn begin_eval(
        destination: &Path,
        capture_step: u64,
        device: &Device,
        physical_batch: usize,
        effective_batch: usize,
        loops: usize,
    ) -> Result<Self> {
        let run = ProfileRun::inference(
            "tofy::p2::looped_agent::evaluation",
            capture_step,
            device_name(device),
        )
        .correlation_id(format!("tofy.looped/eval-{capture_step:012}"));
        Self::open(
            destination,
            run,
            None,
            physical_batch,
            effective_batch,
            loops,
        )
    }

    fn open(
        destination: &Path,
        run: ProfileRun,
        gradients: Option<GradientCapturePlan>,
        physical_batch: usize,
        effective_batch: usize,
        loops: usize,
    ) -> Result<Self> {
        ensure!(
            physical_batch > 0 && effective_batch >= physical_batch,
            "invalid looped capture batch sizes"
        );
        ensure!(loops > 0, "looped capture needs positive loops");
        let label = run.correlation_id.clone();
        let gpu = run.device == "cuda";
        let mut labels = vec!["tofy.looped/capture".to_owned()];
        if gradients.is_some() {
            for micro in 0..effective_batch.div_ceil(physical_batch) {
                labels.push(format!("{label}/micro-{micro}/forward"));
                labels.push(format!("{label}/micro-{micro}/backward"));
            }
            labels.push(format!("{label}/gradient-inspection-and-clip"));
            labels.push(format!("{label}/optimizer"));
        } else {
            labels.push(format!("{label}/forward"));
        }
        let contract = CaptureContract {
            measurement_scope: MeasurementScope::ProfiledWork,
            tensors: CoverageLevel::Partial,
            gradients: if gradients.is_some() {
                CoverageLevel::Complete
            } else {
                CoverageLevel::None
            },
            gradient_contract: gradients.as_ref().map(|plan| plan.contract().clone()),
            required_semantic_labels: labels.clone(),
            gpu_expected_semantic_labels: if gpu { labels.clone() } else { vec![] },
            cpu_only_semantic_labels: if gpu { vec![] } else { labels },
            // Operations, activations, logical/physical memory and device events
            // are unwired. Cargo features do not supply their instrumentation.
            ..CaptureContract::default()
        };
        let cached_readout = run
            .tags
            .get("workload")
            .is_some_and(|x| x == "cached-learned-readout");
        let mut run = run
            .capture_contract(contract)
            .tag("physical_batch", physical_batch.to_string())
            .tag("effective_batch", effective_batch.to_string())
            .tag(
                "microbatches",
                effective_batch.div_ceil(physical_batch).to_string(),
            )
            .tag(
                "tail_batch",
                ((effective_batch - 1) % physical_batch + 1).to_string(),
            )
            .tag(
                "loops",
                if cached_readout {
                    "not_applicable".into()
                } else {
                    loops.to_string()
                },
            )
            .tag("boundary_synchronization", "caller_owned");
        if gpu {
            run = run.measured_region_device_synchronized();
        }
        let capture = match CaptureRun::begin(destination, run)? {
            CaptureBegin::Active(capture) => capture,
            CaptureBegin::AlreadyPublished(_) => bail!(
                "looped capture destination already published: {}",
                destination.display()
            ),
        };
        Ok(Self {
            capture: Some(capture),
            gradients,
            label,
            measurements: Cell::new(0),
            measuring: Cell::new(false),
            observations: Cell::new(0),
            gradients_recorded: Cell::new(false),
            failure: RefCell::new(None),
        })
    }

    pub fn measurement(&self) -> LoopedRange<'_> {
        self.measurements.set(self.measurements.get() + 1);
        self.measuring.set(true);
        self.range("tofy.looped/capture", None, true)
    }

    pub fn phase(&self, name: &str, step: Option<ExecutionStep>) -> LoopedRange<'_> {
        self.range(&format!("{}/{name}", self.label), step, false)
    }

    fn session(&self) -> &TraceSession {
        self.capture
            .as_ref()
            .expect("active looped capture")
            .session()
    }

    fn range(
        &self,
        label: &str,
        step: Option<ExecutionStep>,
        measurement: bool,
    ) -> LoopedRange<'_> {
        let session = self.session();
        let candle = if measurement {
            session.begin_measurement(label)
        } else if let Some(step) = step {
            session.begin_step_span(label, step, SpanKind::Function)
        } else {
            session.begin_span(label, SpanKind::Function)
        };
        LoopedRange {
            candle,
            _nvtx: NvtxRange::new(label),
            _tracing: tracing::info_span!("looped_capture", semantic_label = label).entered(),
            owner: self,
            measurement,
            step,
        }
    }

    pub fn record_tensor_stats(
        &self,
        guard: &LoopedRange<'_>,
        label: &str,
        tensor: &Tensor,
    ) -> Result<()> {
        self.observe(guard, || {
            let session = self.session();
            record_tensor_with_label(session, guard.candle.id(), label, tensor, guard.step)?;
            session.record_tensor_stats(guard.candle.id(), label, tensor)
        })
    }

    pub fn record_scalar(&self, guard: &LoopedRange<'_>, label: &str, value: f64) -> Result<()> {
        self.observe(guard, || {
            self.session()
                .record_scalar(guard.candle.id(), label, value)
        })
    }

    /// Record the whole accumulated gradient store once, before clipping. The
    /// guard locates the probe's host/NVTX work; gradient events themselves have
    /// parameter identities, not span identities, in candle-graph's schema.
    pub fn record_gradients(&self, guard: &LoopedRange<'_>, grads: &GradStore) -> Result<()> {
        self.observe(guard, || {
            ensure!(
                !self.gradients_recorded.get(),
                "looped gradients already recorded"
            );
            let plan = self
                .gradients
                .as_ref()
                .context("evaluation has no gradient contract")?;
            plan.record(self.session(), grads)?;
            self.gradients_recorded.set(true);
            Ok(())
        })
    }

    fn observe(&self, guard: &LoopedRange<'_>, record: impl FnOnce() -> Result<()>) -> Result<()> {
        let result = (|| {
            ensure!(
                std::ptr::eq(self, guard.owner),
                "range belongs to another looped capture"
            );
            ensure!(
                self.measuring.get(),
                "looped observations require a live measurement"
            );
            record()
        })();
        match &result {
            Ok(()) => self.observations.set(self.observations.get() + 1),
            Err(error) => {
                self.failure
                    .borrow_mut()
                    .get_or_insert_with(|| format!("{error:#}"));
            }
        }
        result
    }

    pub fn finish(mut self) -> Result<PublicationReceipt> {
        ensure!(
            self.measurements.get() == 1 && !self.measuring.get(),
            "looped capture requires exactly one completed measurement"
        );
        ensure!(
            self.observations.get() > 0,
            "refusing empty successful looped capture"
        );
        ensure!(
            self.gradients.is_none() || self.gradients_recorded.get(),
            "looped capture is missing its gradient population"
        );
        if let Some(reason) = self.failure.borrow().as_ref() {
            bail!("looped capture failed: {reason}");
        }
        self.capture
            .take()
            .expect("active looped capture")
            .publish()
    }
}

impl Drop for LoopedRange<'_> {
    fn drop(&mut self) {
        if self.measurement {
            self.owner.measuring.set(false);
        }
    }
}

impl Drop for LoopedCapture {
    fn drop(&mut self) {
        if let Some(capture) = self.capture.take() {
            let reason = self.failure.get_mut().take().unwrap_or_else(|| {
                "looped invocation exited before successful capture publication".into()
            });
            if let Err(error) = capture.publish_failed(reason) {
                tracing::error!("failed to publish looped diagnostic bundle: {error:#}");
            }
        }
    }
}

fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

fn gradient_family(key: &str) -> Option<&'static str> {
    match key.split('.').next()? {
        "policy_head" => Some("policy"),
        "value_head" => Some("value"),
        "reward_head" => Some("reward"),
        name if name.starts_with("next_head_") => Some("dynamics"),
        "palette_embedding" | "patch_projection" | "metadata_projection" | "readout_token" => {
            Some("core")
        }
        name if name.starts_with("block_") => Some("core"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{LoopedAgent, LoopedConfig};
    use super::*;
    use candle_core::DType;
    use candle_graph::trace::{analyze_health, GradientState, RunOutcome};
    use candle_nn::VarBuilder;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!(
                    "tofy-looped-profile-{name}-{}-{}",
                    std::process::id(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )))
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn binding_contract_has_only_the_three_executed_families() -> Result<()> {
        use super::super::binding::ControlBinder;
        let dir = TestDir::new("binding");
        let device = Device::Cpu;
        let vars = VarMap::new();
        let _model = ControlBinder::new(VarBuilder::from_varmap(&vars, DType::F32, &device))?;
        let destination = dir.0.join("train");
        let cap = LoopedCapture::begin_binding(
            &destination,
            2,
            &device,
            Some(&vars),
            64,
            64,
            4,
            "abstract_effects",
        )?;
        {
            let _measured = cap.measurement();
            let loss = {
                let forward = cap.phase("micro-0/forward", Some(ExecutionStep::Forward));
                let loss = vars
                    .all_vars()
                    .iter()
                    .try_fold(Tensor::zeros((), DType::F32, &device)?, |sum, var| {
                        sum.add(&var.sum_all()?)
                    })?;
                cap.record_tensor_stats(&forward, "loss/policy", &loss)?;
                loss
            };
            let grads = {
                let _backward = cap.phase("micro-0/backward", Some(ExecutionStep::Backward));
                loss.backward()?
            };
            {
                let inspect = cap.phase("gradient-inspection-and-clip", None);
                cap.record_gradients(&inspect, &grads)?;
            }
            drop(cap.phase("optimizer", Some(ExecutionStep::Optimizer)));
        }
        cap.finish()?;
        candle_graph::verify_bundle(&destination)?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        let health = analyze_health(&trace);
        assert!(health.structurally_valid && health.capture_complete);
        assert_eq!(trace.gradients.len(), 29);
        let contract = trace
            .run
            .capture_contract
            .gradient_contract
            .as_ref()
            .unwrap();
        let families = contract
            .families
            .iter()
            .map(|f| f.family.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            families,
            std::collections::BTreeSet::from(["input", "shared_core", "policy"])
        );
        assert_eq!(trace.run.tags["parameter_count"], "100292");
        assert_eq!(trace.run.tags["vision_core_forwards"], "0");
        let destination = dir.0.join("eval");
        let cap = LoopedCapture::begin_binding(
            &destination,
            1,
            &device,
            None,
            4,
            4,
            4,
            "precomputed_visual",
        )?;
        {
            let _measured = cap.measurement();
            let forward = cap.phase("forward", Some(ExecutionStep::Forward));
            cap.record_tensor_stats(
                &forward,
                "policy/logits",
                &Tensor::zeros((4, 4), DType::F32, &device)?,
            )?;
        }
        cap.finish()?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        assert!(trace.run.capture_contract.gradient_contract.is_none());
        assert_eq!(trace.run.tags["input_source"], "precomputed_visual");
        Ok(())
    }

    #[test]
    fn demonstration_grounding_declares_seven_frozen_frames() -> Result<()> {
        let dir = TestDir::new("demonstration-grounding");
        let destination = dir.0.join("eval");
        let device = Device::Cpu;
        let capture = LoopedCapture::begin_demonstration_grounding(&destination, 1, &device, 2)?;
        {
            let _measured = capture.measurement();
            let forward = capture.phase("forward", Some(ExecutionStep::Forward));
            for frame in 0..7 {
                for (name, shape) in [
                    ("features", vec![2, 64, 128]),
                    ("attention", vec![2, 2, 64]),
                    ("pooled", vec![2, 256]),
                ] {
                    capture.record_tensor_stats(
                        &forward,
                        &format!("frames/{frame}/{name}"),
                        &Tensor::zeros(shape, DType::F32, &device)?,
                    )?;
                }
            }
            capture.record_tensor_stats(
                &forward,
                "policy/logits",
                &Tensor::zeros((2, 4), DType::F32, &device)?,
            )?;
        }
        capture.finish()?;
        candle_graph::verify_bundle(&destination)?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        let health = analyze_health(&trace);
        assert!(
            health.structurally_valid && health.capture_complete,
            "{health:?}"
        );
        assert!(trace.run.capture_contract.gradient_contract.is_none());
        assert_eq!(trace.run.capture_contract.gradients, CoverageLevel::None);
        assert!(trace.gradients.is_empty());
        assert_eq!(trace.tensors.len(), 22);
        assert_eq!(trace.run.tags["workload"], "looped-demonstration-grounding");
        assert_eq!(
            trace.run.tags["feature_seam"],
            "post-final-rms-each-observed-frame"
        );
        assert_eq!(trace.run.tags["frame_count"], "7");
        assert_eq!(trace.run.tags["loops"], "4");
        assert_eq!(trace.run.tags["physical_batch"], "2");
        assert_eq!(trace.run.tags["effective_batch"], "2");
        assert_eq!(trace.run.tags["ordinary_heads"], "not_executed");
        assert_eq!(trace.run.tags["optimizer_updates"], "0");
        assert_eq!(fs::read_dir(&dir.0)?.count(), 1);
        Ok(())
    }

    #[test]
    fn grounded_policy_contract_keeps_ordinary_heads_explicitly_inactive() -> Result<()> {
        use super::super::grounded_policy::{active_parameters, parameter_family, SpatialPolicy};
        use candle_graph::GradientFamilyExpectation;
        let dir = TestDir::new("grounded");
        let device = Device::Cpu;
        let core = VarMap::new();
        let policy = VarMap::new();
        let _model = LoopedAgent::new(
            LoopedConfig {
                hidden: 4,
                heads: 1,
                layers: 1,
                max_loops: 4,
            },
            VarBuilder::from_varmap(&core, DType::F32, &device),
        )?;
        let _policy = SpatialPolicy::new(VarBuilder::from_varmap(&policy, DType::F32, &device))?;
        let destination = dir.0.join("train");
        let capture = LoopedCapture::begin_grounded_policy(
            &destination,
            2,
            &device,
            Some((&core, &policy)),
            2,
            2,
            4,
        )?;
        {
            let _measured = capture.measurement();
            let loss = {
                let forward = capture.phase("micro-0/forward", Some(ExecutionStep::Forward));
                // Isolate manifest expectations from the separate real-network
                // gradient test: every active parameter has gradient exactly one.
                let loss = active_parameters(&core, &policy)?
                    .iter()
                    .try_fold(Tensor::zeros((), DType::F32, &device)?, |sum, (_, var)| {
                        sum.add(&var.sum_all()?)
                    })?;
                capture.record_tensor_stats(&forward, "loss/policy", &loss)?;
                loss
            };
            let grads = {
                let _backward = capture.phase("micro-0/backward", Some(ExecutionStep::Backward));
                loss.backward()?
            };
            {
                let inspect = capture.phase("gradient-inspection-and-clip", None);
                capture.record_gradients(&inspect, &grads)?;
            }
            drop(capture.phase("optimizer", Some(ExecutionStep::Optimizer)));
        }
        capture.finish()?;
        candle_graph::verify_bundle(&destination)?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        let health = analyze_health(&trace);
        assert!(
            health.structurally_valid && health.capture_complete,
            "{health:?}"
        );
        let contract = trace
            .run
            .capture_contract
            .gradient_contract
            .as_ref()
            .unwrap();
        assert_eq!(contract.families.len(), 7);
        for family in &contract.families {
            let active = matches!(
                family.family.as_str(),
                "core" | "spatial_queries" | "spatial_policy"
            );
            assert_eq!(
                family.expectation,
                if active {
                    GradientFamilyExpectation::Active
                } else {
                    GradientFamilyExpectation::Inactive
                }
            );
        }
        for (expected, actual) in contract.expected.iter().zip(&trace.gradients) {
            let active = matches!(
                parameter_family(&expected.key),
                Some("core" | "spatial_queries" | "spatial_policy")
            );
            assert_eq!(
                actual.state,
                if active {
                    GradientState::Present
                } else {
                    GradientState::Missing
                }
            );
        }
        assert_eq!(trace.run.tags["ordinary_heads"], "not_executed");
        assert_eq!(trace.run.tags["loops"], "4");
        let destination = dir.0.join("eval");
        let capture =
            LoopedCapture::begin_grounded_policy(&destination, 1, &device, None, 2, 2, 4)?;
        {
            let _measured = capture.measurement();
            let forward = capture.phase("forward", Some(ExecutionStep::Forward));
            capture.record_tensor_stats(
                &forward,
                "policy/logits",
                &Tensor::ones((2, 4), DType::F32, &device)?,
            )?;
        }
        capture.finish()?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        assert!(trace.run.capture_contract.gradient_contract.is_none());
        assert_eq!(trace.run.tags["workload"], "looped-grounded-policy");
        assert_eq!(fs::read_dir(&dir.0)?.count(), 2);
        Ok(())
    }

    #[test]
    fn cpu_bundle_covers_real_parameter_manifest_and_exact_tail_batch() -> Result<()> {
        let dir = TestDir::new("train");
        let destination = dir.0.join("update-2");
        let device = Device::Cpu;
        let vars = VarMap::new();
        let _model = LoopedAgent::new(
            LoopedConfig {
                hidden: 4,
                heads: 1,
                layers: 1,
                max_loops: 1,
            },
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        let capture = LoopedCapture::begin(&destination, 2, &device, &vars, 33, 64, 1)?;
        {
            let measured = capture.measurement();
            // A controlled loss touches every real parameter with gradient one;
            // this tests capture coverage independently of the training objective.
            let loss = {
                let forward = capture.phase("micro-0/forward", Some(ExecutionStep::Forward));
                let loss = vars
                    .all_vars()
                    .iter()
                    .try_fold(Tensor::zeros((), DType::F32, &device)?, |sum, var| {
                        sum.add(&var.sum_all()?)
                    })?;
                capture.record_tensor_stats(&forward, "loss/total", &loss)?;
                loss
            };
            {
                drop(capture.phase("micro-0/backward", Some(ExecutionStep::Backward)));
                drop(capture.phase("micro-1/forward", Some(ExecutionStep::Forward)));
                drop(capture.phase("micro-1/backward", Some(ExecutionStep::Backward)));
                let backward = capture.phase("gradient-inspection-and-clip", None);
                capture.record_gradients(&backward, &loss.backward()?)?;
            }
            drop(capture.phase("optimizer", Some(ExecutionStep::Optimizer)));
            capture.record_scalar(&measured, "batch/rows", 64.0)?;
        }
        let receipt = capture.finish()?;
        assert_eq!(
            candle_graph::verify_bundle(&destination)?,
            receipt.verification
        );
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        let health = analyze_health(&trace);
        assert!(
            health.structurally_valid && health.capture_complete,
            "{health:?}"
        );
        let contract = trace
            .run
            .capture_contract
            .gradient_contract
            .as_ref()
            .unwrap();
        contract.validate()?;
        let mut keys = vars
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            contract.expected.iter().map(|p| &p.key).collect::<Vec<_>>(),
            keys.iter().collect::<Vec<_>>()
        );
        assert_eq!(trace.gradients.len(), keys.len());
        assert!(trace
            .gradients
            .iter()
            .all(|g| g.state == GradientState::Present));
        assert_eq!(contract.families.len(), 5);
        assert_eq!(trace.run.tags["physical_batch"], "33");
        assert_eq!(trace.run.tags["effective_batch"], "64");
        assert_eq!(trace.run.tags["tail_batch"], "31");
        assert_eq!(
            trace.run.capture_contract.measurement_scope,
            MeasurementScope::ProfiledWork
        );
        assert_eq!(trace.tensors.len(), 1);
        assert_eq!(
            fs::read_dir(&dir.0)?.count(),
            1,
            "publication must remove staging files"
        );
        assert!(LoopedCapture::begin(&destination, 2, &device, &vars, 33, 64, 1).is_err());
        Ok(())
    }

    #[test]
    fn evaluation_publishes_without_gradients_and_abandoned_or_empty_work_fails() -> Result<()> {
        let dir = TestDir::new("eval");
        let destination = dir.0.join("complete");
        let capture = LoopedCapture::begin_eval(&destination, 1, &Device::Cpu, 1, 1, 1)?;
        {
            let _measured = capture.measurement();
            let forward = capture.phase("forward", Some(ExecutionStep::Forward));
            capture.record_tensor_stats(
                &forward,
                "policy/logits",
                &Tensor::ones((1, 4), DType::F32, &Device::Cpu)?,
            )?;
        }
        capture.finish()?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        assert!(analyze_health(&trace).structurally_valid);
        assert_eq!(trace.run.capture_contract.gradients, CoverageLevel::None);
        assert!(trace.run.capture_contract.gradient_contract.is_none());
        for case in ["abandoned", "empty"] {
            let destination = dir.0.join(case);
            let capture = LoopedCapture::begin_eval(&destination, 1, &Device::Cpu, 1, 1, 1)?;
            drop(capture.measurement());
            if case == "empty" {
                assert!(capture.finish().is_err());
            } else {
                drop(capture);
            }
            candle_graph::verify_bundle(&destination)?;
            let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
            assert_eq!(trace.terminal.outcome, RunOutcome::Failed);
        }
        assert_eq!(
            fs::read_dir(&dir.0)?.count(),
            3,
            "failed publication must remove staging files"
        );
        Ok(())
    }
}
