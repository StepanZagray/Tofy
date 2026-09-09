//! Binder numerics. Dataset authority, checkpoint hashes and lifecycle are caller-owned.
use anyhow::{ensure, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var, D};
use candle_graph::ExecutionStep;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    time::Instant,
};
use tofy::p2::{
    looped_agent::{
        binding::{ControlBinder, INPUT_WIDTH, PARAMETERS, RECORDS},
        profile::{LoopedCapture, LoopedRange},
    },
    optimizer::{accumulate_parameter_gradients, clip_gradients_gpu_with_stats},
};

pub const EFFECTIVE: usize = 512;
pub const DIGEST_SCHEMA: &str = "looped-action-binding-parameters-v1";

#[derive(Clone)]
pub struct Row {
    pub features: [[f32; INPUT_WIDTH]; RECORDS],
    pub correct_action: usize,
    pub raw: Value,
}
impl Row {
    pub fn parse(raw: Value) -> Result<Self> {
        ensure!(raw.is_object(), "dataset row must be an object");
        ensure!(
            [
                "logits",
                "stage",
                "loops",
                "cleared",
                "query_cleared",
                "model_input_sha256"
            ]
            .iter()
            .all(|key| raw.get(key).is_none()),
            "dataset uses reserved output fields"
        );
        let features: [[f32; INPUT_WIDTH]; RECORDS] =
            serde_json::from_value(raw["features"].clone())?;
        let label = raw["correct_action"]
            .as_u64()
            .context("integer action label required")?;
        ensure!(
            label < 4 && features.iter().flatten().all(|x| x.is_finite()),
            "invalid row features/label"
        );
        ensure!(
            raw["input_sha256"].as_str() == Some(&feature_hash(&features))
                && raw["label_sha256"].as_str()
                    == Some(&format!(
                        "{:x}",
                        Sha256::digest((label as u32).to_le_bytes())
                    )),
            "row feature/label hashes differ"
        );
        Ok(Self {
            features,
            correct_action: label as usize,
            raw,
        })
    }
}

fn float_hash(values: impl Iterator<Item = f32>) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub fn feature_hash(features: &[[f32; INPUT_WIDTH]; RECORDS]) -> String {
    float_hash(features.iter().flatten().copied())
}

#[derive(Clone, Copy, Default)]
pub struct Intervention {
    pub cleared: bool,
    pub query_cleared: bool,
}

pub fn tensors(rows: &[Row], intervention: Intervention, device: &Device) -> Result<Tensor> {
    ensure!(!rows.is_empty(), "empty binder batch");
    ensure!(
        !(intervention.cleared && intervention.query_cleared),
        "combined clearing is not registered"
    );
    let mut values = Vec::with_capacity(rows.len() * RECORDS * INPUT_WIDTH);
    for row in rows {
        ensure!(
            row.correct_action < 4 && row.features.iter().flatten().all(|x| x.is_finite()),
            "invalid binder batch"
        );
        for (i, record) in row.features.iter().enumerate() {
            for (j, &value) in record.iter().enumerate() {
                let erase = j < 2
                    && ((intervention.cleared && i < 3) || (intervention.query_cleared && i == 3));
                values.push(if erase { 0.0 } else { value });
            }
        }
    }
    Ok(Tensor::from_vec(
        values,
        (rows.len(), RECORDS, INPUT_WIDTH),
        device,
    )?)
}

fn named(vars: &VarMap) -> Result<Vec<(String, Var)>> {
    let mut names = vars
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("parameter lock poisoned"))?
        .iter()
        .map(|(n, v)| (n.clone(), v.clone()))
        .collect::<Vec<_>>();
    names.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(names)
}

// Exact algorithm from examples/looped_agent_probe.rs::initialize, including
// its first/last-axis Glorot bound for the three-dimensional readout token.
pub fn initialize(varmap: &VarMap, seed: u64) -> Result<()> {
    for (name, var) in named(varmap)? {
        let mut hash = Sha256::new();
        hash.update(seed.to_le_bytes());
        hash.update(name.as_bytes());
        let digest = hash.finalize();
        let mut rng = ChaCha8Rng::seed_from_u64(u64::from_le_bytes(digest[..8].try_into()?));
        let shape = var.dims();
        let values = if name.ends_with("bias") {
            vec![0.0; var.elem_count()]
        } else if shape.len() == 1 && name.ends_with("weight") {
            vec![1.0; var.elem_count()]
        } else {
            let bound = (6.0
                / (shape.last().copied().unwrap_or(1) + shape.first().copied().unwrap_or(1))
                    as f32)
                .sqrt();
            (0..var.elem_count())
                .map(|_| rng.random_range(-bound..bound))
                .collect()
        };
        var.set(&Tensor::from_vec(values, var.shape(), var.device())?)?;
    }
    Ok(())
}

fn bits(tensor: &Tensor) -> Result<Vec<u32>> {
    ensure!(tensor.dtype() == DType::F32, "parameters must be F32");
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite parameter");
    Ok(values.into_iter().map(f32::to_bits).collect())
}

pub type Snapshot = BTreeMap<String, (Vec<usize>, Vec<u32>)>;

/// Domain bytes + NUL, then sorted entries: u64-LE name length/name bytes,
/// u64-LE rank/dimensions, u64-LE element count, and u32-LE F32 bit patterns.
pub fn parameter_digest(snapshot: &Snapshot) -> String {
    let mut hash = Sha256::new();
    hash.update(DIGEST_SCHEMA.as_bytes());
    hash.update([0]);
    for (name, (shape, values)) in snapshot {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((shape.len() as u64).to_le_bytes());
        for &dim in shape {
            hash.update((dim as u64).to_le_bytes());
        }
        hash.update((values.len() as u64).to_le_bytes());
        for &value in values {
            hash.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub struct Model {
    pub binder: ControlBinder,
    pub vars: VarMap,
    pub device: Device,
    names: Vec<(String, Var)>,
}
#[derive(Serialize)]
pub struct Changes {
    pub all_parameters_unchanged: bool,
    pub active_parameter_names: Vec<String>,
    pub changed_body_names: Vec<String>,
    pub changed_head_names: Vec<String>,
}
impl Model {
    pub fn new(device: &Device) -> Result<Self> {
        let vars = VarMap::new();
        let binder = ControlBinder::new(VarBuilder::from_varmap(&vars, DType::F32, device))?;
        initialize(&vars, 0)?;
        let names = named(&vars)?;
        ensure!(
            names.iter().map(|(_, v)| v.elem_count()).sum::<usize>() == PARAMETERS
                && names.len() == 29,
            "binder parameter population differs"
        );
        Ok(Self {
            binder,
            vars,
            device: device.clone(),
            names,
        })
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.names
            .iter()
            .map(|(n, v)| Ok((n.clone(), (v.dims().to_vec(), bits(v)?))))
            .collect()
    }
    pub fn changes(&self, before: &Snapshot) -> Result<Changes> {
        let after = self.snapshot()?;
        ensure!(
            before.keys().eq(after.keys()),
            "snapshot population differs"
        );
        let mut changes = Changes {
            all_parameters_unchanged: true,
            active_parameter_names: after.keys().cloned().collect(),
            changed_body_names: vec![],
            changed_head_names: vec![],
        };
        for (name, value) in &after {
            ensure!(value.0 == before[name].0, "snapshot shape differs");
            if value != &before[name] {
                changes.all_parameters_unchanged = false;
                if name.starts_with("policy_head.") {
                    changes.changed_head_names.push(name.clone());
                } else {
                    changes.changed_body_names.push(name.clone());
                }
            }
        }
        Ok(changes)
    }
    pub fn load(&self, path: &Path) -> Result<()> {
        ensure!(
            path.is_file() && !path.is_symlink(),
            "regular checkpoint required"
        );
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        ensure!(
            tensors.len() == self.names.len(),
            "checkpoint population differs"
        );
        for (name, var) in &self.names {
            let tensor = tensors.get(name).context("checkpoint tensor missing")?;
            ensure!(
                tensor.dims() == var.dims(),
                "checkpoint shape differs: {name}"
            );
            bits(tensor)?;
        }
        for (name, var) in &self.names {
            var.set(&tensors[name])?;
            ensure!(
                bits(var)? == bits(&tensors[name])?,
                "checkpoint load changed bits"
            );
        }
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        ensure!(!path.exists(), "checkpoint output reused");
        let before = self.snapshot()?;
        self.vars.save(path)?;
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        ensure!(
            tensors.len() == before.len(),
            "saved checkpoint population differs"
        );
        for (name, (shape, expected)) in before {
            let tensor = tensors.get(&name).context("saved tensor missing")?;
            ensure!(
                tensor.dims() == shape && bits(tensor)? == expected,
                "saved checkpoint roundtrip differs"
            );
        }
        Ok(())
    }
    pub fn restore(&self, snapshot: &Snapshot) -> Result<()> {
        let current = self.snapshot()?;
        ensure!(
            current.keys().eq(snapshot.keys())
                && current.iter().all(|(n, (s, v))| s == &snapshot[n].0
                    && v.len() == snapshot[n].1.len()
                    && snapshot[n].1.iter().all(|&b| f32::from_bits(b).is_finite())),
            "invalid restore snapshot"
        );
        for (name, var) in &self.names {
            var.set(&Tensor::from_vec(
                snapshot[name]
                    .1
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect::<Vec<_>>(),
                var.shape(),
                &self.device,
            )?)?;
        }
        ensure!(
            self.changes(snapshot)?.all_parameters_unchanged,
            "restore changed parameter bits"
        );
        Ok(())
    }
    pub fn frozen(&self) -> Result<ControlBinder> {
        let tensors = self
            .names
            .iter()
            .map(|(n, v)| Ok((n.clone(), v.detach().copy()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        ControlBinder::new(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }
    pub fn optimizer(&self) -> Result<AdamW> {
        Ok(AdamW::new(
            self.names.iter().map(|(_, v)| v.clone()).collect(),
            ParamsAdamW {
                lr: 0.0003,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?)
    }
}

pub fn cross_entropy(logits: &Tensor, rows: &[Row]) -> Result<Tensor> {
    ensure!(
        logits.dims() == [rows.len(), 4] && !rows.is_empty(),
        "label/logit shape differs"
    );
    let labels = Tensor::from_vec(
        rows.iter()
            .map(|r| r.correct_action as u32)
            .collect::<Vec<_>>(),
        rows.len(),
        logits.device(),
    )?;
    Ok(candle_nn::ops::log_softmax(logits, D::Minus1)?
        .gather(&labels.unsqueeze(1)?, 1)?
        .mean_all()?
        .neg()?)
}
fn finite_logits(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
    let rows = logits.to_vec2::<f32>()?;
    ensure!(
        rows.iter().flatten().all(|v| v.is_finite()),
        "nonfinite logits"
    );
    Ok(rows)
}
fn record(
    cap: &LoopedCapture,
    guard: &LoopedRange<'_>,
    inputs: &Tensor,
    logits: &Tensor,
    loss: &Tensor,
) -> Result<()> {
    for (name, tensor) in [
        ("inputs/effects", inputs),
        ("policy/logits", logits),
        ("loss/policy", loss),
    ] {
        cap.record_tensor_stats(guard, name, tensor)?;
    }
    cap.record_scalar(guard, "batch/rows", inputs.dim(0)? as f64)
}
fn phase<T>(
    capture: Option<&LoopedCapture>,
    device: &Device,
    name: &str,
    step: Option<ExecutionStep>,
    work: impl FnOnce(Option<&LoopedRange<'_>>) -> Result<T>,
) -> Result<T> {
    let guard = capture.map(|c| c.phase(name, step));
    let result = work(guard.as_ref());
    let sync = device.synchronize();
    drop(guard);
    sync?;
    result
}

pub struct Evaluation {
    pub logits: Vec<Vec<f32>>,
    pub model_input_sha256: Vec<String>,
    pub mean_ce: f64,
    pub correct: usize,
}
pub fn evaluate(
    binder: &ControlBinder,
    rows: &[Row],
    loops: usize,
    intervention: Intervention,
    device: &Device,
    capture: Option<&LoopedCapture>,
) -> Result<Evaluation> {
    device.synchronize()?;
    let measured = capture.map(LoopedCapture::measurement);
    let result = phase(
        capture,
        device,
        "forward",
        Some(ExecutionStep::Forward),
        |guard| {
            let inputs = tensors(rows, intervention, device)?;
            let logits = binder.forward(&inputs, loops)?;
            let loss = cross_entropy(&logits, rows)?;
            ensure!(
                !inputs.track_op() && !logits.track_op() && !loss.track_op(),
                "frozen evaluation constructed autograd graph"
            );
            let mean_ce = f64::from(loss.to_scalar::<f32>()?);
            ensure!(mean_ce.is_finite(), "nonfinite frozen CE");
            let correct = logits
                .argmax(1)?
                .to_vec1::<u32>()?
                .iter()
                .zip(rows)
                .filter(|(a, r)| **a as usize == r.correct_action)
                .count();
            if let (Some(c), Some(g)) = (capture, guard) {
                record(c, g, &inputs, &logits, &loss)?;
            }
            Ok(Evaluation {
                logits: finite_logits(&logits)?,
                model_input_sha256: inputs
                    .to_vec3::<f32>()?
                    .iter()
                    .map(|row| float_hash(row.iter().flatten().copied()))
                    .collect(),
                mean_ce,
                correct,
            })
        },
    );
    let sync = device.synchronize();
    drop(measured);
    sync?;
    result
}

#[derive(Serialize)]
pub struct UpdateMetrics {
    pub rows: usize,
    pub physical_batch: usize,
    pub microbatches: usize,
    pub tail_batch: usize,
    pub mean_ce: f64,
    pub pre_update_correct: usize,
    pub pre_clip_norm: f64,
    pub clip_scale: f64,
    pub input_gradient_norm: f64,
    pub shared_core_gradient_norm: f64,
    pub body_gradient_norm: f64,
    pub head_gradient_norm: f64,
    pub elapsed_seconds: f64,
}

fn norm(grads: &GradStore, names: &[(String, Var)], select: impl Fn(&str) -> bool) -> Result<f64> {
    let mut sum: Option<Tensor> = None;
    for (name, var) in names.iter().filter(|(n, _)| select(n)) {
        let square = grads
            .get(var)
            .with_context(|| format!("missing gradient {name}"))?
            .sqr()?
            .sum_all()?;
        sum = Some(match sum {
            Some(x) => x.add(&square)?,
            None => square,
        });
    }
    let value = f64::from(
        sum.context("empty gradient family")?
            .sqrt()?
            .to_scalar::<f32>()?,
    );
    ensure!(value.is_finite(), "nonfinite gradient norm");
    Ok(value)
}

pub fn train_update(
    model: &Model,
    rows: &[Row],
    physical: usize,
    optimizer: &mut AdamW,
    capture: Option<&LoopedCapture>,
) -> Result<UpdateMetrics> {
    ensure!(
        rows.len() == EFFECTIVE && physical.is_power_of_two() && physical <= EFFECTIVE,
        "update requires {EFFECTIVE} rows and a power-of-two batch in 1..={EFFECTIVE}"
    );
    train_update_rows(model, rows, physical, optimizer, capture)
}

// The registered entry point fixes 512 rows. This shared implementation also
// permits small CPU fixtures to check row weighting, including uneven chunks.
fn train_update_rows(
    model: &Model,
    rows: &[Row],
    physical: usize,
    optimizer: &mut AdamW,
    capture: Option<&LoopedCapture>,
) -> Result<UpdateMetrics> {
    let effective = rows.len();
    ensure!(
        effective > 0 && (1..=effective).contains(&physical),
        "invalid update batch"
    );
    ensure!(
        rows.iter()
            .all(|r| r.correct_action < 4 && r.features.iter().flatten().all(|v| v.is_finite())),
        "invalid training row"
    );
    model.device.synchronize()?;
    let started = Instant::now();
    let measured = capture.map(LoopedCapture::measurement);
    let result = (|| {
        let mut gradients = None;
        let mut mean_ce = 0.0;
        let mut correct = 0;
        for (micro, rows) in rows.chunks(physical).enumerate() {
            let weight = rows.len() as f64 / effective as f64;
            let loss = phase(
                capture,
                &model.device,
                &format!("micro-{micro}/forward"),
                Some(ExecutionStep::Forward),
                |guard| {
                    let inputs = tensors(rows, Intervention::default(), &model.device)?;
                    let logits = model.binder.forward(&inputs, 4)?;
                    finite_logits(&logits)?;
                    let loss = cross_entropy(&logits, rows)?;
                    let ce = f64::from(loss.to_scalar::<f32>()?);
                    ensure!(ce.is_finite(), "nonfinite training CE");
                    mean_ce += ce * weight;
                    correct += logits
                        .argmax(1)?
                        .to_vec1::<u32>()?
                        .iter()
                        .zip(rows)
                        .filter(|(a, r)| **a as usize == r.correct_action)
                        .count();
                    if let (Some(c), Some(g)) = (capture, guard) {
                        record(c, g, &inputs, &logits, &loss)?;
                    }
                    Ok((loss * weight)?)
                },
            )?;
            phase(
                capture,
                &model.device,
                &format!("micro-{micro}/backward"),
                Some(ExecutionStep::Backward),
                |_| accumulate_parameter_gradients(&mut gradients, loss.backward()?, &model.vars),
            )?;
        }
        let mut gradients = gradients.context("missing accumulated gradients")?;
        let (clip, input, core, body, head) = phase(
            capture,
            &model.device,
            "gradient-inspection-and-clip",
            None,
            |guard| {
                let input = norm(&gradients, &model.names, |n| {
                    n.starts_with("input_projection.") || n == "readout_token"
                })?;
                let core = norm(&gradients, &model.names, |n| n.starts_with("block_"))?;
                let body = norm(&gradients, &model.names, |n| !n.starts_with("policy_head."))?;
                let head = norm(&gradients, &model.names, |n| n.starts_with("policy_head."))?;
                ensure!(body > 0.0 && head > 0.0, "zero body/head gradient");
                if let (Some(c), Some(g)) = (capture, guard) {
                    c.record_gradients(g, &gradients)?;
                }
                let clip = clip_gradients_gpu_with_stats(&mut gradients, &model.vars, 1.0)?;
                if let (Some(c), Some(g)) = (capture, guard) {
                    for (name, value) in [
                        ("gradient/input_norm", input),
                        ("gradient/shared_core_norm", core),
                        ("gradient/body_norm", body),
                        ("gradient/head_norm", head),
                        ("gradient/pre_clip_norm", clip.pre_clip_norm),
                        ("gradient/clip_scale", clip.scale),
                    ] {
                        c.record_scalar(g, name, value)?;
                    }
                }
                Ok((clip, input, core, body, head))
            },
        )?;
        phase(
            capture,
            &model.device,
            "optimizer",
            Some(ExecutionStep::Optimizer),
            |_| {
                optimizer.step(&gradients)?;
                Ok(())
            },
        )?;
        Ok(UpdateMetrics {
            rows: effective,
            physical_batch: physical,
            microbatches: effective.div_ceil(physical),
            tail_batch: (effective - 1) % physical + 1,
            mean_ce,
            pre_update_correct: correct,
            pre_clip_norm: clip.pre_clip_norm,
            clip_scale: clip.scale,
            input_gradient_norm: input,
            shared_core_gradient_norm: core,
            body_gradient_norm: body,
            head_gradient_norm: head,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        })
    })();
    let sync = model.device.synchronize();
    drop(measured);
    sync?;
    result
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    pub fn row(index: usize) -> Row {
        let features = [
            [0., -1., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 1., 0., 0.],
            [-1., 0., 0., 1., 0., 0., 0.],
            [
                if index.is_multiple_of(2) { 1. } else { -1. },
                0.,
                0.,
                0.,
                0.,
                0.,
                1.,
            ],
        ];
        let label = (index % 4) as u32;
        Row::parse(json!({"index":index,"features":features,"correct_action":label,"input_sha256":feature_hash(&features),"label_sha256":format!("{:x}",Sha256::digest(label.to_le_bytes())),"audit":{"marker":"synthetic-only"}})).unwrap()
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "tofy-binding-engine-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // Frozen reference algorithm from looped_agent_probe at61e13356, not the
    // implementation's initializer helper. Includes rank3 CLS bound semantics.
    fn legacy_initialize(varmap: &VarMap, seed: u64) -> Result<()> {
        let mut vars = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(n, v)| (n.clone(), v.clone()))
            .collect::<Vec<_>>();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, var) in vars {
            let mut hash = Sha256::new();
            hash.update(seed.to_le_bytes());
            hash.update(name.as_bytes());
            let digest = hash.finalize();
            let mut rng = ChaCha8Rng::seed_from_u64(u64::from_le_bytes(digest[..8].try_into()?));
            let shape = var.dims();
            let values = if name.ends_with("bias") {
                vec![0.0; var.elem_count()]
            } else if shape.len() == 1 && name.ends_with("weight") {
                vec![1.0; var.elem_count()]
            } else {
                let bound = (6.0
                    / (shape.last().copied().unwrap_or(1) + shape.first().copied().unwrap_or(1))
                        as f32)
                    .sqrt();
                (0..var.elem_count())
                    .map(|_| rng.random_range(-bound..bound))
                    .collect()
            };
            var.set(&Tensor::from_vec(values, var.shape(), var.device())?)?;
        }
        Ok(())
    }

    #[test]
    fn exact_initializer_parity_and_stable_bit_digest() -> Result<()> {
        let a = Model::new(&Device::Cpu)?;
        let b = Model::new(&Device::Cpu)?;
        for seed in [0, 17] {
            initialize(&a.vars, seed)?;
            legacy_initialize(&b.vars, seed)?;
            assert_eq!(a.snapshot()?, b.snapshot()?);
        }
        initialize(&a.vars, 0)?;
        initialize(&b.vars, 0)?;
        let original = a.snapshot()?;
        assert_eq!(
            parameter_digest(&original),
            parameter_digest(&b.snapshot()?)
        );
        let mut changed = original.clone();
        changed.get_mut("policy_head.bias").unwrap().1[0] = (-0.0f32).to_bits();
        assert_ne!(parameter_digest(&original), parameter_digest(&changed));
        assert_eq!(
            original.values().map(|(_, v)| v.len()).sum::<usize>(),
            PARAMETERS
        );
        Ok(())
    }

    #[test]
    fn strict_checkpoint_population_dtype_and_restore() -> Result<()> {
        let dir = Temp::new()?;
        let model = Model::new(&Device::Cpu)?;
        let before = model.snapshot()?;
        let path = dir.0.join("original.safetensors");
        model.save(&path)?;
        assert!(model.save(&path).is_err());
        for kind in 0..4 {
            let mut tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
            match kind {
                0 => {
                    tensors.remove("policy_head.bias");
                }
                1 => {
                    tensors.insert(
                        "foreign".into(),
                        Tensor::zeros(1, DType::F32, &Device::Cpu)?,
                    );
                }
                2 => {
                    tensors.insert(
                        "policy_head.bias".into(),
                        Tensor::zeros(4, DType::F64, &Device::Cpu)?,
                    );
                }
                _ => {
                    tensors.insert(
                        "policy_head.bias".into(),
                        Tensor::full(f32::NAN, 4, &Device::Cpu)?,
                    );
                }
            }
            let bad = dir.0.join(format!("bad{kind}.safetensors"));
            candle_core::safetensors::save(&tensors, &bad)?;
            assert!(model.load(&bad).is_err());
            assert_eq!(model.snapshot()?, before);
        }
        let mut changed = before.clone();
        changed
            .get_mut("policy_head.bias")
            .unwrap()
            .1
            .fill(0.25f32.to_bits());
        model.restore(&changed)?;
        assert_ne!(model.snapshot()?, before);
        model.load(&path)?;
        assert_eq!(model.snapshot()?, before);
        Ok(())
    }

    #[test]
    fn clamp_changes_only_requested_effects_and_hashes_actual_tensor() -> Result<()> {
        let model = Model::new(&Device::Cpu)?;
        let frozen = model.frozen()?;
        let rows = vec![row(0), row(1)];
        let before = model.snapshot()?;
        for intervention in [
            Intervention::default(),
            Intervention {
                cleared: true,
                query_cleared: false,
            },
            Intervention {
                cleared: false,
                query_cleared: true,
            },
        ] {
            let tensor = tensors(&rows, intervention, &Device::Cpu)?;
            let values = tensor.to_vec3::<f32>()?;
            for (r, value) in rows.iter().zip(&values) {
                for (i, record) in value.iter().enumerate() {
                    for (j, &v) in record.iter().enumerate() {
                        let erased = j < 2
                            && ((intervention.cleared && i < 3)
                                || (intervention.query_cleared && i == 3));
                        assert_eq!(v, if erased { 0.0 } else { r.features[i][j] });
                    }
                }
            }
            let evaluated = evaluate(&frozen, &rows, 4, intervention, &Device::Cpu, None)?;
            for (hash, values) in evaluated.model_input_sha256.iter().zip(values) {
                assert_eq!(*hash, float_hash(values.into_iter().flatten()));
            }
        }
        assert!(evaluate(
            &model.binder,
            &rows,
            4,
            Intervention::default(),
            &Device::Cpu,
            None
        )
        .is_err());
        assert!(tensors(
            &rows,
            Intervention {
                cleared: true,
                query_cleared: true
            },
            &Device::Cpu
        )
        .is_err());
        assert_eq!(model.snapshot()?, before);
        Ok(())
    }

    #[test]
    fn invalid_rows_and_identity_hashes_fail_closed() {
        for kind in 0..6 {
            let mut raw = row(0).raw;
            match kind {
                0 => raw["correct_action"] = json!(4),
                1 => raw["features"] = json!(vec![vec![0.0; 7]; 3]),
                2 => raw["features"][0][0] = json!(1e100),
                3 => raw["input_sha256"] = json!("0".repeat(64)),
                4 => raw["label_sha256"] = json!("0".repeat(64)),
                _ => raw["logits"] = json!(vec![0.0; 4]),
            }
            assert!(Row::parse(raw).is_err());
        }
    }

    fn accumulated(model: &Model, rows: &[Row], physical: usize) -> Result<GradStore> {
        let mut gradients = None;
        for chunk in rows.chunks(physical) {
            let logits = model
                .binder
                .forward(&tensors(chunk, Intervention::default(), &Device::Cpu)?, 4)?;
            let loss = (cross_entropy(&logits, chunk)? * (chunk.len() as f64 / rows.len() as f64))?;
            accumulate_parameter_gradients(&mut gradients, loss.backward()?, &model.vars)?;
        }
        Ok(gradients.unwrap())
    }

    #[test]
    fn bounded_accumulation_matches_gradients_and_one_update_behavior() -> Result<()> {
        let a = Model::new(&Device::Cpu)?;
        let b = Model::new(&Device::Cpu)?;
        let rows = (0..8).map(row).collect::<Vec<_>>();
        let ga = accumulated(&a, &rows, 8)?;
        let gb = accumulated(&b, &rows, 3)?;
        for ((name, va), (_, vb)) in a.names.iter().zip(&b.names) {
            let x = ga
                .get(va)
                .context("full gradient missing")?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let y = gb
                .get(vb)
                .context("split gradient missing")?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (x, y) in x.iter().zip(y) {
                assert!(
                    (x - y).abs() <= 2e-6 + 1e-4 * x.abs(),
                    "gradient differs {name}: {x} vs {y}"
                );
            }
        }
        let before = a.snapshot()?;
        let ma = train_update_rows(&a, &rows, 8, &mut a.optimizer()?, None)?;
        let mb = train_update_rows(&b, &rows, 3, &mut b.optimizer()?, None)?;
        assert_eq!((ma.rows, ma.microbatches, ma.tail_batch), (8, 1, 8));
        assert_eq!((mb.rows, mb.microbatches, mb.tail_batch), (8, 3, 2));
        assert!((ma.mean_ce - mb.mean_ce).abs() < 1e-5);
        assert!((ma.pre_clip_norm - mb.pre_clip_norm).abs() < 1e-4);
        // Near-zero softmax key-bias gradients can give different AdamW bit
        // updates without changing policy. Compare the executed one-step output.
        let x = evaluate(
            &a.frozen()?,
            &rows[..4],
            4,
            Intervention::default(),
            &Device::Cpu,
            None,
        )?;
        let y = evaluate(
            &b.frozen()?,
            &rows[..4],
            4,
            Intervention::default(),
            &Device::Cpu,
            None,
        )?;
        for (x, y) in x.logits.iter().flatten().zip(y.logits.iter().flatten()) {
            assert!(
                (x - y).abs() < 2e-4,
                "one-update behavior differs: {x} vs {y}"
            );
        }
        let changes = a.changes(&before)?;
        assert!(!changes.changed_body_names.is_empty() && !changes.changed_head_names.is_empty());
        a.restore(&before)?;
        assert!(a.changes(&before)?.all_parameters_unchanged);
        let mut optimizer = a.optimizer()?;
        assert!(train_update(&a, &rows, 8, &mut optimizer, None).is_err());
        let registered_rows = (0..EFFECTIVE).map(row).collect::<Vec<_>>();
        for physical in [0, 3, EFFECTIVE + 1, 2 * EFFECTIVE] {
            assert!(train_update(&a, &registered_rows, physical, &mut optimizer, None).is_err());
        }
        assert!(a.changes(&before)?.all_parameters_unchanged);
        Ok(())
    }
}
