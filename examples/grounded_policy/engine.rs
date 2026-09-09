//! Numerical engine only. The caller binds source/checkpoint/import hashes and
//! owns capture publication, run lifecycle, dataset identities and stop budgets.

use anyhow::{ensure, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var, D};
use candle_graph::ExecutionStep;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    time::Instant,
};
use tofy::p2::{
    looped_agent::{
        grounded_policy::{active_parameters, SpatialPolicy, SpatialPolicyOutput},
        model::{LoopedAgent, LoopedConfig, LoopedFeatures},
        profile::{LoopedCapture, LoopedRange},
        task::Inputs,
        ACTIONS,
    },
    optimizer::{accumulate_parameter_gradients, clip_gradients_gpu_with_stats},
};

pub use super::data::tensors;

pub const EFFECTIVE_BATCH: usize = 64;
pub const LOOPS: usize = 4;

pub struct Forward {
    pub features: LoopedFeatures,
    pub policy: SpatialPolicyOutput,
}

pub struct Model {
    pub core: LoopedAgent,
    pub policy: SpatialPolicy,
    pub core_vars: VarMap,
    pub head_vars: VarMap,
    pub device: Device,
    active: Vec<(String, Var)>,
    active_map: VarMap,
}

/// An independent, immutable parameter copy; no Var-backed forward graph.
pub struct FrozenModel {
    pub core: LoopedAgent,
    pub policy: SpatialPolicy,
    pub device: Device,
}

#[derive(Clone)]
pub struct Snapshot {
    /// F32 bit patterns preserve signed zero and exact restoration identity.
    pub tensors: BTreeMap<String, Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct ChangeAudit {
    pub active_parameter_names: Vec<String>,
    pub changed_body_names: Vec<String>,
    pub changed_head_names: Vec<String>,
    pub changed_unused_names: Vec<String>,
    pub unused_heads_unchanged: bool,
    pub all_parameters_unchanged: bool,
}

#[derive(Debug, Serialize)]
pub struct UpdateMetrics {
    pub rows: usize,
    pub physical_batch: usize,
    pub microbatches: usize,
    pub tail_batch: usize,
    pub mean_ce: f64,
    pub pre_update_correct: usize,
    pub pre_clip_norm: f64,
    pub clip_scale: f64,
    pub body_gradient_norm: f64,
    pub head_gradient_norm: f64,
    pub query_gradient_norms: [f64; 2],
    pub elapsed_seconds: f64,
}

fn config() -> LoopedConfig {
    LoopedConfig {
        hidden: 128,
        heads: 4,
        layers: 2,
        max_loops: 8,
    }
}

fn named(vars: &VarMap) -> Result<Vec<(String, Var)>> {
    let mut entries = vars
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap lock poisoned"))?
        .iter()
        .map(|(key, var)| (key.clone(), var.clone()))
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn bits(tensor: &Tensor) -> Result<Vec<u32>> {
    ensure!(tensor.dtype() == DType::F32, "parameter is not F32");
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite parameter");
    Ok(values.into_iter().map(f32::to_bits).collect())
}

/// Refuse unknown/missing tensors rather than accepting a partially loaded model.
fn load_exact(vars: &VarMap, path: &Path, device: &Device) -> Result<()> {
    ensure!(
        path.is_file() && !path.is_symlink(),
        "checkpoint must be a regular file"
    );
    let tensors = candle_core::safetensors::load(path, device)?;
    let named = named(vars)?;
    ensure!(
        tensors.len() == named.len(),
        "checkpoint tensor population mismatch"
    );
    // Complete validation precedes the first mutation.
    for (name, var) in &named {
        let tensor = tensors
            .get(name)
            .with_context(|| format!("missing checkpoint tensor {name}"))?;
        ensure!(
            tensor.dims() == var.dims() && tensor.dtype() == DType::F32,
            "checkpoint shape/type mismatch: {name}"
        );
        bits(tensor)?;
    }
    for (name, var) in named {
        var.set(&tensors[&name])?;
        ensure!(
            bits(&var)? == bits(&tensors[&name])?,
            "checkpoint load changed bytes: {name}"
        );
    }
    Ok(())
}

fn detached(vars: &VarMap) -> Result<HashMap<String, Tensor>> {
    named(vars)?
        .into_iter()
        .map(|(name, var)| Ok((name, var.detach().copy()?)))
        .collect()
}

impl Model {
    /// Temporary constructor values must be replaced by the hash-validated core
    /// and canonical imported head before any registered model invocation.
    pub fn new(device: &Device) -> Result<Self> {
        let core_vars = VarMap::new();
        let head_vars = VarMap::new();
        let core = LoopedAgent::new(
            config(),
            VarBuilder::from_varmap(&core_vars, DType::F32, device),
        )?;
        let policy = SpatialPolicy::new(VarBuilder::from_varmap(&head_vars, DType::F32, device))?;
        let active = active_parameters(&core_vars, &head_vars)?;
        let active_map = VarMap::new();
        active_map
            .data()
            .lock()
            .unwrap()
            .extend(active.iter().cloned());
        Ok(Self {
            core,
            policy,
            core_vars,
            head_vars,
            device: device.clone(),
            active,
            active_map,
        })
    }

    pub fn load_core(&self, path: &Path) -> Result<()> {
        load_exact(&self.core_vars, path, &self.device)
    }
    pub fn load_head(&self, path: &Path) -> Result<()> {
        load_exact(&self.head_vars, path, &self.device)
    }

    pub fn forward(&self, rows: &[Inputs]) -> Result<Forward> {
        forward(&self.core, &self.policy, rows, &self.device)
    }

    pub fn frozen(&self) -> Result<FrozenModel> {
        Ok(FrozenModel {
            core: LoopedAgent::new(
                config(),
                VarBuilder::from_tensors(detached(&self.core_vars)?, DType::F32, &self.device),
            )?,
            policy: SpatialPolicy::new(VarBuilder::from_tensors(
                detached(&self.head_vars)?,
                DType::F32,
                &self.device,
            ))?,
            device: self.device.clone(),
        })
    }

    pub fn optimizer(&self) -> Result<AdamW> {
        Ok(AdamW::new(
            self.active.iter().map(|(_, var)| var.clone()).collect(),
            ParamsAdamW {
                lr: 0.0003,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?)
    }

    pub fn active_parameter_names(&self) -> Vec<String> {
        self.active.iter().map(|(name, _)| name.clone()).collect()
    }

    fn all_named(&self) -> Result<Vec<(String, Var)>> {
        let mut all = Vec::new();
        for (prefix, vars) in [
            ("core", &self.core_vars),
            ("spatial_policy", &self.head_vars),
        ] {
            all.extend(
                named(vars)?
                    .into_iter()
                    .map(|(name, var)| (format!("{prefix}.{name}"), var)),
            );
        }
        Ok(all)
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(Snapshot {
            tensors: self
                .all_named()?
                .into_iter()
                .map(|(name, var)| Ok((name, bits(&var)?)))
                .collect::<Result<_>>()?,
        })
    }

    pub fn change_audit(&self, original: &Snapshot) -> Result<ChangeAudit> {
        let now = self.snapshot()?;
        ensure!(
            now.tensors.keys().eq(original.tensors.keys()),
            "snapshot parameter population mismatch"
        );
        let mut audit = ChangeAudit {
            active_parameter_names: self.active_parameter_names(),
            changed_body_names: vec![],
            changed_head_names: vec![],
            changed_unused_names: vec![],
            unused_heads_unchanged: true,
            all_parameters_unchanged: true,
        };
        for (name, value) in &now.tensors {
            ensure!(
                value.len() == original.tensors[name].len(),
                "snapshot parameter length mismatch"
            );
            if value == &original.tensors[name] {
                continue;
            }
            audit.all_parameters_unchanged = false;
            if name.starts_with("spatial_policy.") {
                audit.changed_head_names.push(name.clone());
            } else if audit.active_parameter_names.binary_search(name).is_ok() {
                audit.changed_body_names.push(name.clone());
            } else {
                audit.changed_unused_names.push(name.clone());
                audit.unused_heads_unchanged = false;
            }
        }
        Ok(audit)
    }

    /// The caller must discard any old optimizer and construct a fresh one after
    /// restoring a qualification snapshot; optimizer-state resume is unsupported.
    pub fn restore(&self, snapshot: &Snapshot) -> Result<()> {
        let named = self.all_named()?;
        ensure!(
            named.len() == snapshot.tensors.len(),
            "snapshot tensor count mismatch"
        );
        for (name, var) in &named {
            let values = snapshot
                .tensors
                .get(name)
                .with_context(|| format!("missing snapshot tensor {name}"))?;
            ensure!(
                values.len() == var.elem_count()
                    && values.iter().all(|&x| f32::from_bits(x).is_finite()),
                "invalid snapshot tensor {name}"
            );
        }
        for (name, var) in named {
            var.set(&Tensor::from_vec(
                snapshot.tensors[&name]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
                var.shape(),
                &self.device,
            )?)?;
        }
        ensure!(
            self.change_audit(snapshot)?.all_parameters_unchanged,
            "snapshot restoration changed bytes"
        );
        Ok(())
    }

    pub fn save_pair(&self, core_path: &Path, head_path: &Path) -> Result<()> {
        ensure!(
            core_path != head_path && !core_path.exists() && !head_path.exists(),
            "checkpoint destination reused"
        );
        self.core_vars.save(core_path)?;
        self.head_vars.save(head_path)?;
        for (vars, path) in [(&self.core_vars, core_path), (&self.head_vars, head_path)] {
            let loaded = candle_core::safetensors::load(path, &self.device)?;
            ensure!(
                loaded.len() == vars.all_vars().len(),
                "saved checkpoint tensor count"
            );
            for (name, var) in named(vars)? {
                let tensor = loaded.get(&name).context("missing saved tensor")?;
                ensure!(
                    tensor.dims() == var.dims() && bits(&var)? == bits(tensor)?,
                    "saved checkpoint roundtrip {name}"
                );
            }
        }
        Ok(())
    }
}

impl FrozenModel {
    pub fn forward(&self, rows: &[Inputs]) -> Result<Forward> {
        let output = forward(&self.core, &self.policy, rows, &self.device)?;
        ensure!(
            !output.features.current.track_op()
                && !output.features.cls.track_op()
                && !output.policy.logits.track_op()
                && !output.policy.attention.track_op()
                && !output.policy.pooled.track_op(),
            "frozen model constructed an autograd graph"
        );
        Ok(output)
    }
}

fn forward(
    core: &LoopedAgent,
    policy: &SpatialPolicy,
    rows: &[Inputs],
    device: &Device,
) -> Result<Forward> {
    let (patches, metadata) = tensors(rows, device)?;
    let features = core.forward_features(&patches, &metadata, LOOPS)?;
    let policy = policy.forward(&features.current)?;
    Ok(Forward { features, policy })
}

pub fn cross_entropy(logits: &Tensor, labels: &[usize]) -> Result<Tensor> {
    ensure!(
        !labels.is_empty()
            && logits.dims() == [labels.len(), ACTIONS]
            && labels.iter().all(|&a| a < ACTIONS),
        "policy label/logit mismatch"
    );
    let labels = Tensor::from_vec(
        labels.iter().map(|&a| a as u32).collect::<Vec<_>>(),
        labels.len(),
        logits.device(),
    )?;
    Ok(candle_nn::ops::log_softmax(logits, D::Minus1)?
        .gather(&labels.unsqueeze(1)?, 1)?
        .mean_all()?
        .neg()?)
}

pub fn record_forward(
    capture: &LoopedCapture,
    guard: &LoopedRange<'_>,
    output: &Forward,
    loss: &Tensor,
) -> Result<()> {
    for (label, tensor) in [
        ("features/current", &output.features.current),
        ("policy/attention", &output.policy.attention),
        ("policy/pooled", &output.policy.pooled),
        ("policy/logits", &output.policy.logits),
        ("loss/policy", loss),
    ] {
        capture.record_tensor_stats(guard, label, tensor)?;
    }
    Ok(())
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
    let synced = device.synchronize();
    drop(guard);
    let result = result?;
    synced?;
    Ok(result)
}

fn gradient_norm(grads: &GradStore, named: &[(String, Var)], prefix: &str) -> Result<f64> {
    let mut total: Option<Tensor> = None;
    for (name, var) in named.iter().filter(|(name, _)| name.starts_with(prefix)) {
        let gradient = grads
            .get(var)
            .with_context(|| format!("missing active gradient {name}"))?;
        let square = gradient.sqr()?.sum_all()?;
        total = Some(match total {
            Some(sum) => sum.add(&square)?,
            None => square,
        });
    }
    let norm = total
        .context("empty gradient family")?
        .sqrt()?
        .to_scalar::<f32>()? as f64;
    ensure!(norm.is_finite(), "nonfinite gradient family {prefix}");
    Ok(norm)
}

pub fn train_update(
    model: &Model,
    rows: &[Inputs],
    labels: &[usize],
    physical: usize,
    optimizer: &mut AdamW,
    capture: Option<&LoopedCapture>,
) -> Result<UpdateMetrics> {
    ensure!(
        rows.len() == EFFECTIVE_BATCH
            && labels.len() == EFFECTIVE_BATCH
            && (1..=EFFECTIVE_BATCH).contains(&physical),
        "training requires exactly64 rows and physical batch1..64"
    );
    ensure!(
        labels.iter().all(|&label| label < ACTIONS),
        "invalid training labels"
    );
    model.device.synchronize()?;
    let started = Instant::now();
    let measured = capture.map(LoopedCapture::measurement);
    let result = (|| -> Result<UpdateMetrics> {
        let mut gradients = None;
        let mut mean_ce = 0.0;
        let mut correct = 0;
        let all = model.all_named()?;
        for (micro, (inputs, targets)) in rows
            .chunks(physical)
            .zip(labels.chunks(physical))
            .enumerate()
        {
            let fraction = inputs.len() as f64 / EFFECTIVE_BATCH as f64;
            let loss = phase(
                capture,
                &model.device,
                &format!("micro-{micro}/forward"),
                Some(ExecutionStep::Forward),
                |guard| {
                    let output = model.forward(inputs)?;
                    let loss = cross_entropy(&output.policy.logits, targets)?;
                    let ce = loss.to_scalar::<f32>()? as f64;
                    ensure!(ce.is_finite(), "nonfinite policy CE");
                    mean_ce += ce * fraction;
                    correct += output
                        .policy
                        .logits
                        .argmax(1)?
                        .to_vec1::<u32>()?
                        .iter()
                        .zip(targets)
                        .filter(|(a, b)| **a as usize == **b)
                        .count();
                    if let (Some(c), Some(guard)) = (capture, guard) {
                        record_forward(c, guard, &output, &loss)?;
                    }
                    Ok((loss * fraction)?)
                },
            )?;
            phase(
                capture,
                &model.device,
                &format!("micro-{micro}/backward"),
                Some(ExecutionStep::Backward),
                |_| {
                    let grads = loss.backward()?;
                    for (name, var) in &all {
                        if model
                            .active
                            .binary_search_by(|(key, _)| key.cmp(name))
                            .is_err()
                        {
                            ensure!(
                                grads.get(var).is_none(),
                                "unused head received a gradient: {name}"
                            );
                        }
                    }
                    accumulate_parameter_gradients(&mut gradients, grads, &model.active_map)
                },
            )?;
        }
        let mut gradients = gradients.context("no accumulated gradients")?;
        let (clip, body, head, queries) = phase(
            capture,
            &model.device,
            "gradient-inspection-and-clip",
            None,
            |guard| {
                let body = gradient_norm(&gradients, &model.active, "core.")?;
                let head = gradient_norm(&gradients, &model.active, "spatial_policy.")?;
                let query_var = model
                    .active
                    .iter()
                    .find(|(name, _)| name == "spatial_policy.queries")
                    .context("missing queries")?;
                let query = gradients
                    .get(&query_var.1)
                    .context("missing query gradient")?;
                let mut queries = [0.0; 2];
                for (i, norm) in queries.iter_mut().enumerate() {
                    *norm = query.get(i)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
                    ensure!(norm.is_finite(), "nonfinite query gradient");
                }
                if let (Some(c), Some(guard)) = (capture, guard) {
                    c.record_gradients(guard, &gradients)?;
                }
                let clip = clip_gradients_gpu_with_stats(&mut gradients, &model.active_map, 1.0)?;
                if let (Some(c), Some(guard)) = (capture, guard) {
                    for (name, value) in [
                        ("gradient/body_norm", body),
                        ("gradient/head_norm", head),
                        ("gradient/query0_norm", queries[0]),
                        ("gradient/query1_norm", queries[1]),
                        ("gradient/pre_clip_norm", clip.pre_clip_norm),
                        ("gradient/clip_scale", clip.scale),
                    ] {
                        c.record_scalar(guard, name, value)?;
                    }
                }
                Ok((clip, body, head, queries))
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
            rows: EFFECTIVE_BATCH,
            physical_batch: physical,
            microbatches: EFFECTIVE_BATCH.div_ceil(physical),
            tail_batch: (EFFECTIVE_BATCH - 1) % physical + 1,
            mean_ce,
            pre_update_correct: correct,
            pre_clip_norm: clip.pre_clip_norm,
            clip_scale: clip.scale,
            body_gradient_norm: body,
            head_gradient_norm: head,
            query_gradient_norms: queries,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        })
    })();
    let synced = model.device.synchronize();
    drop(measured);
    let result = result?;
    synced?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    use tofy::p2::looped_agent::{META_DIM, PATCH_PIXELS, TOKENS};

    struct TestDir(PathBuf);
    impl TestDir {
        fn new() -> Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "tofy-grounded-engine-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn synthetic_inputs() -> Inputs {
        Inputs {
            patches: (0..TOKENS * PATCH_PIXELS).map(|i| (i % 5) as u32).collect(),
            metadata: (0..TOKENS * META_DIM)
                .map(|i| (i % 7) as f32 / 7.0)
                .collect(),
        }
    }

    #[test]
    fn frozen_copy_is_untracked_independent_and_exact_before_mutation() -> Result<()> {
        let model = Model::new(&Device::Cpu)?;
        let frozen = model.frozen()?;
        let rows = [synthetic_inputs()];
        let train = model.forward(&rows)?;
        let frozen_before = frozen.forward(&rows)?;
        assert!(train.features.current.track_op() && train.policy.logits.track_op());
        for (a, b) in [
            (&train.features.current, &frozen_before.features.current),
            (&train.policy.logits, &frozen_before.policy.logits),
            (&train.policy.attention, &frozen_before.policy.attention),
            (&train.policy.pooled, &frozen_before.policy.pooled),
        ] {
            assert_eq!(bits(a)?, bits(b)?);
        }
        let bias = model.head_vars.data().lock().unwrap()["output.bias"].clone();
        bias.set(&bias.affine(1.0, 2.0)?)?;
        assert_ne!(
            bits(&model.forward(&rows)?.policy.logits)?,
            bits(&frozen_before.policy.logits)?
        );
        assert_eq!(
            bits(&frozen.forward(&rows)?.policy.logits)?,
            bits(&frozen_before.policy.logits)?
        );
        Ok(())
    }

    #[test]
    fn registered_optimizer_excludes_unused_heads_and_restores_exact_bits() -> Result<()> {
        let model = Model::new(&Device::Cpu)?;
        let bias = model.head_vars.data().lock().unwrap()["output.bias"].clone();
        bias.set(&Tensor::new(&[-0.0f32, 0.0, 0.5, -0.5], &Device::Cpu)?)?;
        let initial = model.snapshot()?;
        let mut grads = GradStore::default();
        for (_, var) in &model.active {
            grads.insert(var, Tensor::ones_like(var)?);
        }
        let mut optimizer = model.optimizer()?;
        optimizer.step(&grads)?;
        let audit = model.change_audit(&initial)?;
        assert!(!audit.changed_body_names.is_empty() && !audit.changed_head_names.is_empty());
        assert!(audit.unused_heads_unchanged && audit.changed_unused_names.is_empty());
        assert!(audit.active_parameter_names.windows(2).all(|p| p[0] < p[1]));
        model.restore(&initial)?;
        assert!(model.change_audit(&initial)?.all_parameters_unchanged);
        assert_eq!(bits(&bias)?[0], (-0.0f32).to_bits());
        let mut corrupt = initial.clone();
        corrupt
            .tensors
            .get_mut("spatial_policy.output.bias")
            .unwrap()[0] = f32::NAN.to_bits();
        assert!(model.restore(&corrupt).is_err());
        assert!(model.change_audit(&initial)?.all_parameters_unchanged);
        Ok(())
    }

    #[test]
    fn checkpoint_roundtrip_is_complete_and_malformed_load_does_not_mutate() -> Result<()> {
        let dir = TestDir::new()?;
        let model = Model::new(&Device::Cpu)?;
        let original = model.snapshot()?;
        let (core, head) = (
            dir.0.join("core.safetensors"),
            dir.0.join("head.safetensors"),
        );
        model.save_pair(&core, &head)?;
        assert!(model.save_pair(&core, &head).is_err());
        let other = Model::new(&Device::Cpu)?;
        other.load_core(&core)?;
        other.load_head(&head)?;
        assert!(other.change_audit(&original)?.all_parameters_unchanged);
        let malformed = VarMap::new();
        let vb = VarBuilder::from_varmap(&malformed, DType::F32, &Device::Cpu);
        vb.get((2, 128), "queries")?;
        vb.get((4, 256), "output.weight")?;
        vb.get(4, "foreign.bias")?;
        let path = dir.0.join("malformed.safetensors");
        malformed.save(&path)?;
        assert!(other.load_head(&path).is_err());
        assert!(other.change_audit(&original)?.all_parameters_unchanged);
        Ok(())
    }

    #[test]
    fn uneven_33_31_row_weighted_gradients_match_single_batch() -> Result<()> {
        let vars = VarMap::new();
        let weight =
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).get((2, 4), "weight")?;
        let x = Tensor::from_vec(
            (0..128).map(|i| (i % 11) as f32 - 5.0).collect::<Vec<_>>(),
            (64, 2),
            &Device::Cpu,
        )?;
        let labels = (0..64).map(|i| (i / 5) % 4).collect::<Vec<_>>();
        let loss = cross_entropy(&x.matmul(&weight)?, &labels)?;
        let expected = loss.backward()?;
        let mut actual = None;
        let mut weighted_loss = 0.0;
        for (offset, length) in [(0, 33), (33, 31)] {
            let loss = cross_entropy(
                &x.narrow(0, offset, length)?.matmul(&weight)?,
                &labels[offset..offset + length],
            )?;
            let fraction = length as f64 / 64.0;
            weighted_loss += loss.to_scalar::<f32>()? as f64 * fraction;
            accumulate_parameter_gradients(&mut actual, (loss * fraction)?.backward()?, &vars)?;
        }
        assert!((loss.to_scalar::<f32>()? as f64 - weighted_loss).abs() < 1e-6);
        let actual = actual.unwrap();
        let a = actual.get(&weight).unwrap();
        let b = expected.get(&weight).unwrap();
        assert!(a.sub(b)?.abs()?.max_all()?.to_scalar::<f32>()? < 2e-6);
        let mut actual = actual;
        let clip = clip_gradients_gpu_with_stats(&mut actual, &vars, 0.01)?;
        assert!(clip.pre_clip_norm > 0.01 && clip.scale < 1.0);
        assert!(
            (actual
                .get(&weight)
                .unwrap()
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()?
                - 0.01)
                .abs()
                < 1e-7
        );
        Ok(())
    }

    #[test]
    fn malformed_update_and_inputs_fail_before_any_change() -> Result<()> {
        let model = Model::new(&Device::Cpu)?;
        let initial = model.snapshot()?;
        let mut optimizer = model.optimizer()?;
        assert!(train_update(&model, &[], &[], 1, &mut optimizer, None).is_err());
        let row = synthetic_inputs();
        assert!(train_update(
            &model,
            &vec![row.clone(); 64],
            &[0; 64],
            0,
            &mut optimizer,
            None
        )
        .is_err());
        let mut invalid = row;
        invalid.metadata[0] = f32::NAN;
        assert!(tensors(&[invalid], &Device::Cpu).is_err());
        assert!(model.change_audit(&initial)?.all_parameters_unchanged);
        Ok(())
    }
}
