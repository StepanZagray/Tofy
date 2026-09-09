//! Frozen tensor composition. No labels, map IDs or public-pixel role parser
//! enter forward; export and validation readbacks do not feed the inference path.
use super::{Config, Control};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
};
use tofy::p2::looped_agent::{
    binding::EquivariantBinder,
    grounded_policy::SpatialPolicy,
    model::{LoopedAgent, LoopedConfig},
    native_binding,
    profile::LoopedCapture,
};

type Parameters = HashMap<String, Tensor>;
pub struct Model {
    core: LoopedAgent,
    selector: SpatialPolicy,
    binder: EquivariantBinder,
    parameters: [Parameters; 3],
}
pub struct Output {
    pub attention: Tensor,
    pub records: Tensor,
    pub logits: Tensor,
}
fn digest(bytes: impl Iterator<Item = u8>) -> String {
    format!("{:x}", Sha256::digest(bytes.collect::<Vec<_>>()))
}
pub fn check(t: &Tensor) -> Result<()> {
    ensure!(
        t.dtype() == DType::F32 && !t.track_op(),
        "frozen tensor dtype/autograd differs"
    );
    ensure!(
        t.abs()?
            .le(f32::MAX)?
            .to_dtype(DType::F32)?
            .min_all()?
            .to_scalar::<f32>()?
            == 1.,
        "nonfinite frozen tensor"
    );
    Ok(())
}
fn imported(bytes: &[u8], device: &Device) -> Result<Parameters> {
    ensure!(bytes.len() == 5136, "canonical head length differs");
    let mut offset = 0;
    let mut map = HashMap::new();
    for (name, shape) in [
        ("queries", vec![2, 128]),
        ("output.weight", vec![4, 256]),
        ("output.bias", vec![4]),
    ] {
        let count = shape.iter().product::<usize>();
        let values = bytes[offset..offset + count * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite imported head"
        );
        map.insert(name.into(), Tensor::from_vec(values, shape, device)?);
        offset += count * 4;
    }
    Ok(map)
}
impl Model {
    pub fn load(config: &Config, device: &Device) -> Result<Self> {
        config.verify_checkpoints()?;
        let manifest: Value = serde_json::from_slice(&fs::read(&config.import_manifest)?)?;
        let bytes = fs::read(&config.head_checkpoint)?;
        let head = imported(&bytes, device)?;
        ensure!(
            manifest["source_kind"] == "c10_role_ridge"
                && manifest["arm"] == "c10_true"
                && manifest["core_checkpoint_sha256"] == super::CORE
                && manifest["artifact"]["sha256"] == super::HEAD
                && manifest["recipe"]["privileged_role_supervision"] == true,
            "import provenance differs"
        );
        for section in manifest["tensors"]
            .as_array()
            .context("missing import tensor table")?
        {
            let name = section["name"].as_str().context("missing tensor name")?;
            let tensor = head.get(name).context("unknown import tensor")?;
            let raw = tensor.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                digest(raw.iter().flat_map(|x| x.to_le_bytes())) == section["sha256"],
                "import tensor bytes differ"
            );
        }
        Self::from_parameters(
            [
                candle_core::safetensors::load(&config.core_checkpoint, device)?,
                head,
                candle_core::safetensors::load(&config.binder_checkpoint, device)?,
            ],
            device,
        )
    }
    fn from_parameters(parameters: [Parameters; 3], device: &Device) -> Result<Self> {
        // Constructors request all exact names/shapes. These exact populations
        // additionally reject unconsumed foreign tensors, including unused heads.
        ensure!(
            parameters.iter().map(HashMap::len).eq([44, 3, 28]),
            "checkpoint tensor population differs"
        );
        for map in &parameters {
            for t in map.values() {
                check(t)?;
            }
        }
        let core = LoopedAgent::new(
            LoopedConfig {
                hidden: 128,
                heads: 4,
                layers: 2,
                max_loops: 8,
            },
            VarBuilder::from_tensors(parameters[0].clone(), DType::F32, device),
        )?;
        let selector = SpatialPolicy::new(VarBuilder::from_tensors(
            parameters[1].clone(),
            DType::F32,
            device,
        ))?;
        let binder = EquivariantBinder::new(VarBuilder::from_tensors(
            parameters[2].clone(),
            DType::F32,
            device,
        ))?;
        Ok(Self {
            core,
            selector,
            binder,
            parameters,
        })
    }
    pub fn identity(&self) -> Result<Value> {
        let mut result = serde_json::Map::new();
        for (kind, map) in ["core", "head", "binder"].into_iter().zip(&self.parameters) {
            let mut canonical = Sha256::new();
            canonical.update(b"looped-action-binding-parameters-v1\0");
            let mut tensors = serde_json::Map::new();
            let mut stored = 0;
            let mut executed = 0;
            for (name, t) in map.iter().collect::<BTreeMap<_, _>>() {
                check(t)?;
                let v = t.flatten_all()?.to_vec1::<f32>()?;
                canonical.update((name.len() as u64).to_le_bytes());
                canonical.update(name.as_bytes());
                canonical.update((t.rank() as u64).to_le_bytes());
                for &d in t.dims() {
                    canonical.update((d as u64).to_le_bytes());
                }
                canonical.update((v.len() as u64).to_le_bytes());
                for x in &v {
                    canonical.update(x.to_le_bytes());
                }
                let active = kind != "core"
                    || !(name.starts_with("next_head_")
                        || name.starts_with("policy_head.")
                        || name.starts_with("value_head.")
                        || name.starts_with("reward_head."));
                stored += v.len();
                if active {
                    executed += v.len();
                }
                tensors.insert(name.clone(),json!({"shape":t.dims(),"dtype":"F32LE","sha256":digest(v.iter().flat_map(|x|x.to_le_bytes())),"executed":active}));
            }
            result.insert(kind.into(),json!({"parameter_sha256":format!("{:x}",canonical.finalize()),"stored_parameters":stored,"executed_parameters":executed,"tensors":tensors}));
        }
        Ok(Value::Object(result))
    }
    /// Only actual public image tensors, public metadata and a declared control.
    pub fn forward(
        &self,
        patches: &Tensor,
        metadata: &Tensor,
        control: Control,
        device: &Device,
        capture: Option<&LoopedCapture>,
    ) -> Result<Output> {
        device.synchronize()?;
        let measurement = capture.map(LoopedCapture::measurement);
        let result = (|| {
            let phase =
                capture.map(|c| c.phase("vision-core", Some(candle_graph::ExecutionStep::Forward)));
            let features = self.core.forward_features(patches, metadata, 4)?;
            check(&features.cls)?;
            check(&features.current)?;
            if let (Some(c), Some(p)) = (capture, &phase) {
                c.record_tensor_stats(p, "core/cls", &features.cls)?;
            }
            device.synchronize()?;
            drop(phase);
            let phase = capture.map(|c| {
                c.phase(
                    "seven-selectors",
                    Some(candle_graph::ExecutionStep::Forward),
                )
            });
            let mut attention = Vec::new();
            for index in 0..7 {
                let frame = features.frame(index)?;
                check(&frame)?;
                if index == 6 {
                    // frame(6) promises the very same immutable Tensor handle;
                    // this establishes bit identity, including signed zero.
                    ensure!(
                        frame.id() == features.current.id(),
                        "current frame tensor identity differs"
                    );
                    let difference = frame
                        .sub(&features.current)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    ensure!(difference == 0.0, "current frame values differ");
                    if let (Some(c), Some(p)) = (capture, &phase) {
                        c.record_scalar(p, "parity/current_frame_input_bitwise_equal", 1.0)?;
                        c.record_scalar(
                            p,
                            "parity/current_frame_max_absolute_difference",
                            f64::from(difference),
                        )?;
                    }
                }
                let out = self.selector.forward(&frame)?;
                for t in [&out.attention, &out.pooled, &out.logits] {
                    check(t)?;
                }
                if let (Some(c), Some(p)) = (capture, &phase) {
                    for (name, t) in [
                        ("features", &frame),
                        ("attention", &out.attention),
                        ("pooled", &out.pooled),
                        ("discarded_logits", &out.logits),
                    ] {
                        c.record_tensor_stats(p, &format!("frames/{index}/{name}"), t)?;
                    }
                }
                attention.push(out.attention);
            }
            let attention = Tensor::stack(&attention, 1)?;
            device.synchronize()?;
            drop(phase);
            let phase = capture
                .map(|c| c.phase("native-adapter", Some(candle_graph::ExecutionStep::Forward)));
            let records = native_binding::records(&attention, metadata, control.native())?;
            check(&records)?;
            if let (Some(c), Some(p)) = (capture, &phase) {
                c.record_tensor_stats(p, "adapter/records", &records)?;
            }
            device.synchronize()?;
            drop(phase);
            let phase = capture.map(|c| {
                c.phase(
                    "equivariant-binder",
                    Some(candle_graph::ExecutionStep::Forward),
                )
            });
            let logits = self.binder.forward(&records, 4)?;
            check(&logits)?;
            if let (Some(c), Some(p)) = (capture, &phase) {
                c.record_tensor_stats(p, "policy/logits", &logits)?;
                c.record_scalar(p, "optimizer/updates", 0.)?;
            }
            device.synchronize()?;
            drop(phase);
            Ok(Output {
                attention,
                records,
                logits,
            })
        })();
        let sync = device.synchronize();
        drop(measurement);
        sync?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    fn fixture() -> Result<Model> {
        let device = &Device::Cpu;
        let core = VarMap::new();
        let head = VarMap::new();
        let binder = VarMap::new();
        LoopedAgent::new(
            LoopedConfig {
                hidden: 128,
                heads: 4,
                layers: 2,
                max_loops: 8,
            },
            VarBuilder::from_varmap(&core, DType::F32, device),
        )?;
        SpatialPolicy::new(VarBuilder::from_varmap(&head, DType::F32, device))?;
        EquivariantBinder::new(VarBuilder::from_varmap(&binder, DType::F32, device))?;
        let freeze = |v: &VarMap| -> Result<Parameters> {
            v.data()
                .lock()
                .unwrap()
                .iter()
                .map(|(n, t)| Ok((n.clone(), t.detach().copy()?)))
                .collect()
        };
        Model::from_parameters([freeze(&core)?, freeze(&head)?, freeze(&binder)?], device)
    }
    #[test]
    fn strict_frozen_population_and_canonical_import() -> Result<()> {
        let model = fixture()?;
        let before = model.identity()?;
        assert_eq!(before["binder"]["stored_parameters"], 1579265);
        assert_eq!(before["head"]["stored_parameters"], 1284);
        let mut bad = model.parameters.clone();
        bad[2].insert(
            "foreign".into(),
            Tensor::zeros((), DType::F32, &Device::Cpu)?,
        );
        assert!(Model::from_parameters(bad, &Device::Cpu).is_err());
        assert!(imported(&vec![0; 5135], &Device::Cpu).is_err());
        let mut raw = vec![0; 5136];
        raw[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(imported(&raw, &Device::Cpu).is_err());
        assert_eq!(before, model.identity()?);
        Ok(())
    }
    #[test]
    fn synthetic_online_composition_matches_retained_record_replay() -> Result<()> {
        struct Temp(std::path::PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let temporary = Temp(
            std::env::temp_dir().join(format!("tofy-native-composition-{}", std::process::id())),
        );
        fs::create_dir(&temporary.0)?;
        let device = &Device::Cpu;
        let model = fixture()?;
        let before = model.identity()?;
        let episode = tofy::p2::looped_agent::task::episode_with_permutation(71, 9, 0, 1, 1)?;
        let input = tofy::p2::looped_agent::task::sample(&episode)?.inputs;
        let patches = Tensor::from_vec(input.patches, (1, 448, 64), device)?;
        let metadata = Tensor::from_vec(input.metadata, (1, 448, 10), device)?;
        let destination = temporary.0.join("capture");
        let capture = LoopedCapture::begin_native_binding(&destination, 1, device, 1, "factual")?;
        let out = model.forward(
            &patches,
            &metadata,
            Control::Factual,
            device,
            Some(&capture),
        )?;
        capture.finish()?;
        candle_graph::verify_bundle(&destination)?;
        let trace = candle_graph::parse_trace(destination.join("trace.jsonl"))?;
        assert_eq!(trace.run.tags["spatial_head_forward_batches"], "7");
        assert_eq!(trace.run.tags["binder_forward_batches"], "1");
        assert_eq!(trace.run.tags["optimizer_updates"], "0");
        assert!(
            trace.gradients.is_empty() && trace.run.capture_contract.gradient_contract.is_none()
        );
        let mut expected = vec!["tofy.looped/capture".to_owned()];
        expected.extend(
            [
                "vision-core",
                "seven-selectors",
                "native-adapter",
                "equivariant-binder",
            ]
            .map(|phase| format!("{}/{phase}", trace.run.correlation_id)),
        );
        assert_eq!(
            trace.run.capture_contract.required_semantic_labels,
            expected
        );
        assert_eq!(
            trace.run.capture_contract.cpu_only_semantic_labels,
            expected
        );
        assert!(trace
            .run
            .capture_contract
            .gpu_expected_semantic_labels
            .is_empty());
        let executed = fs::read_to_string(destination.join("trace.jsonl"))?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|row| row["kind"] == "span_start")
            .filter_map(|row| row["name"].as_str().map(str::to_owned))
            .filter(|name| name.starts_with("tofy.looped/"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(executed, expected.into_iter().collect());
        let replay = model.binder.forward(&out.records, 4)?;
        assert_eq!(out.logits.to_vec2::<f32>()?, replay.to_vec2::<f32>()?);
        assert_eq!(out.attention.dims(), &[1, 7, 2, 64]);
        assert!(!out.logits.track_op());
        assert_eq!(before, model.identity()?);
        Ok(())
    }
}
