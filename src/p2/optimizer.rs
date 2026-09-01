//! Resumable hybrid Muon + AdamW optimizer for P2 training.

use crate::p2::muon::{matrix_view, muon_update, uses_muon};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use candle_nn::VarMap;
use std::path::Path;

pub const DEFAULT_EMA_DECAY: f64 = 0.999;

/// Device-resident exponential moving average of all floating model weights.
///
/// The EMA owns independent storage. Evaluation can temporarily install it in
/// the live [`VarMap`] and restore the training weights afterward.
pub struct ModelEma {
    decay: f64,
    weights: VarMap,
    eval_backup: Option<Vec<(String, Tensor)>>,
}

impl ModelEma {
    pub fn new(model_vars: &VarMap, decay: f64) -> Result<Self> {
        if !decay.is_finite() || !(0.0..1.0).contains(&decay) {
            bail!("EMA decay must be finite and in [0,1), got {decay}");
        }
        let weights = VarMap::new();
        for (name, var) in named_float_vars(model_vars) {
            let value = var.as_tensor().copy()?.detach();
            weights
                .data()
                .lock()
                .unwrap()
                .insert(name, Var::from_tensor(&value)?);
        }
        Ok(Self {
            decay,
            weights,
            eval_backup: None,
        })
    }

    pub fn with_default_decay(model_vars: &VarMap) -> Result<Self> {
        Self::new(model_vars, DEFAULT_EMA_DECAY)
    }

    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// EMA storage for checkpoint save/load alongside the training weights.
    pub fn weights(&self) -> &VarMap {
        &self.weights
    }

    /// Apply `ema = decay*ema + (1-decay)*model` after one optimizer step.
    pub fn update(&mut self, model_vars: &VarMap) -> Result<()> {
        if self.eval_backup.is_some() {
            bail!("cannot update EMA while EMA weights are installed for evaluation");
        }
        let model = named_float_vars(model_vars);
        let ema = named_float_vars(&self.weights);
        ensure_matching_vars(&model, &ema, "update EMA")?;
        for ((_, model), (_, ema)) in model.iter().zip(&ema) {
            let next = ema
                .as_tensor()
                .affine(self.decay, 0.0)?
                .add(&model.as_tensor().affine(1.0 - self.decay, 0.0)?)?
                .detach();
            ema.set(&next)?;
        }
        Ok(())
    }

    /// Replace live model weights with EMA weights until
    /// [`Self::restore_after_eval`] is called.
    pub fn swap_in_for_eval(&mut self, model_vars: &VarMap) -> Result<()> {
        if self.eval_backup.is_some() {
            bail!("EMA weights are already installed for evaluation");
        }
        let model = named_float_vars(model_vars);
        let ema = named_float_vars(&self.weights);
        ensure_matching_vars(&model, &ema, "install EMA")?;
        let backup = model
            .iter()
            .map(|(name, var)| Ok((name.clone(), var.as_tensor().copy()?.detach())))
            .collect::<Result<Vec<_>>>()?;
        self.eval_backup = Some(backup);
        for ((name, model), (_, ema)) in model.iter().zip(&ema) {
            if let Err(install) = model.set(&ema.as_tensor().detach()) {
                let install =
                    anyhow::Error::from(install).context(format!("install EMA parameter {name}"));
                return match self.restore_after_eval(model_vars) {
                    Ok(()) => Err(install),
                    Err(rollback) => Err(install.context(format!(
                        "partial EMA install also failed to roll back training weights: {rollback:#}"
                    ))),
                };
            }
        }
        Ok(())
    }

    /// Restore the training weights saved by [`Self::swap_in_for_eval`].
    pub fn restore_after_eval(&mut self, model_vars: &VarMap) -> Result<()> {
        let backup = self
            .eval_backup
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EMA weights are not installed"))?;
        let model = named_float_vars(model_vars);
        let backup_names = backup
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        let model_names = model
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        if model_names != backup_names {
            bail!("restore EMA model parameter names changed");
        }
        for ((_, model), (_, value)) in model.iter().zip(backup) {
            model.set(value)?;
        }
        self.eval_backup = None;
        Ok(())
    }

    /// Run an evaluation closure with EMA weights, restoring training weights
    /// even when evaluation itself returns an error.
    pub fn with_eval_weights<T>(
        &mut self,
        model_vars: &VarMap,
        evaluate: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut guard = EvalWeightsGuard {
            ema: self,
            model_vars,
        };
        guard.install()?;
        let evaluated = evaluate();
        let restored = guard.restore();
        match (evaluated, restored) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error.context("restore training weights after EMA eval")),
            (Err(eval), Err(restore)) => Err(eval.context(format!(
                "EMA evaluation also failed to restore training weights: {restore:#}"
            ))),
        }
    }
}

struct EvalWeightsGuard<'a> {
    ema: &'a mut ModelEma,
    model_vars: &'a VarMap,
}

impl EvalWeightsGuard<'_> {
    fn install(&mut self) -> Result<()> {
        self.ema.swap_in_for_eval(self.model_vars)
    }

    fn restore(&mut self) -> Result<()> {
        self.ema.restore_after_eval(self.model_vars)
    }
}

impl Drop for EvalWeightsGuard<'_> {
    fn drop(&mut self) {
        if self.ema.eval_backup.is_some() {
            let _ = self.ema.restore_after_eval(self.model_vars);
        }
    }
}

fn named_float_vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let data = varmap.data().lock().unwrap();
    let mut vars = data
        .iter()
        .filter(|(_, var)| var.dtype().is_float())
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars
}

fn ensure_matching_vars(
    left: &[(String, Var)],
    right: &[(String, Var)],
    operation: &str,
) -> Result<()> {
    if left.len() != right.len() {
        bail!(
            "{operation}: parameter count mismatch {} vs {}",
            left.len(),
            right.len()
        );
    }
    for ((left_name, left), (right_name, right)) in left.iter().zip(right) {
        if left_name != right_name || left.shape() != right.shape() || left.dtype() != right.dtype()
        {
            bail!(
                "{operation}: parameter mismatch {left_name} {:?} {:?} vs {right_name} {:?} {:?}",
                left.shape(),
                left.dtype(),
                right.shape(),
                right.dtype()
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptKind {
    Adam,
    Muon,
}

struct ParamOpt {
    name: String,
    var: Var,
    kind: OptKind,
    adam_step: Option<u32>,
}

/// Candle AdamW + DeepSeek-V4 Muon; steps directly on `VarMap` vars (no duplicate weights).
pub struct CheckpointHybridOptimizer {
    params: Vec<ParamOpt>,
    moments: VarMap,
    step_t: usize,
    adam: ParamsAdamW,
    muon_momentum: f64,
    muon_rms_scale: f64,
}

impl CheckpointHybridOptimizer {
    pub fn new(
        varmap: &VarMap,
        adam: ParamsAdamW,
        muon_momentum: f64,
        muon_rms_scale: f64,
    ) -> Result<Self> {
        if !muon_momentum.is_finite() || !(0.0..1.0).contains(&muon_momentum) {
            bail!("muon_momentum must be finite and in [0,1), got {muon_momentum}");
        }
        if !muon_rms_scale.is_finite() || muon_rms_scale <= 0.0 {
            bail!("muon_rms_scale must be finite and > 0, got {muon_rms_scale}");
        }
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<_> = data.keys().cloned().collect();
        names.sort();
        let moments = VarMap::new();
        let mut params = Vec::with_capacity(names.len());
        let mut muon_count = 0usize;
        let mut adam_count = 0usize;
        for name in names {
            let var = data
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing parameter {name}"))?
                .clone();
            if !var.dtype().is_float() {
                continue;
            }
            let kind = if uses_muon(&name, var.shape().dims()) {
                muon_count += 1;
                let m = matrix_view(var.as_tensor())?;
                let momentum = Var::from_tensor(&Tensor::zeros(m.shape(), m.dtype(), m.device())?)?;
                moments
                    .data()
                    .lock()
                    .unwrap()
                    .insert(format!("muon.momentum.{name}"), momentum);
                OptKind::Muon
            } else {
                adam_count += 1;
                let first =
                    Var::from_tensor(&Tensor::zeros(var.shape(), var.dtype(), var.device())?)?;
                let second =
                    Var::from_tensor(&Tensor::zeros(var.shape(), var.dtype(), var.device())?)?;
                {
                    let mut md = moments.data().lock().unwrap();
                    md.insert(format!("first_moment.{name}"), first);
                    md.insert(format!("second_moment.{name}"), second);
                    md.insert(
                        format!("adam.step.{name}"),
                        Var::from_tensor(&Tensor::new(0u32, var.device())?)?,
                    );
                }
                OptKind::Adam
            };
            let adam_step = (kind == OptKind::Adam).then_some(0);
            params.push(ParamOpt {
                name,
                var,
                kind,
                adam_step,
            });
        }
        tracing::info!(
            "optimizer: hybrid Muon+AdamW (muon_vars={muon_count}, adamw_vars={adam_count}, momentum={muon_momentum:.4}, rms_scale={muon_rms_scale:.4})"
        );
        Ok(Self {
            params,
            moments,
            step_t: 0,
            adam,
            muon_momentum,
            muon_rms_scale,
        })
    }

    pub fn parameter_names(&self) -> Vec<String> {
        self.params.iter().map(|p| p.name.clone()).collect()
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.adam.lr;
        let wd = self.adam.weight_decay;
        let beta1 = self.adam.beta1;
        let beta2 = self.adam.beta2;
        let eps = self.adam.eps;
        let moment_data = self.moments.data().lock().unwrap();

        for p in &mut self.params {
            let Some(g) = grads.get(p.var.as_tensor()) else {
                continue;
            };
            match p.kind {
                OptKind::Adam => {
                    let first = moment_data
                        .get(&format!("first_moment.{}", p.name))
                        .expect("adam first moment");
                    let second = moment_data
                        .get(&format!("second_moment.{}", p.name))
                        .expect("adam second moment");
                    let next_step = p
                        .adam_step
                        .expect("adam parameter host step")
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("adam step overflow for {}", p.name))?;
                    let scale_m = 1.0 / (1.0 - beta1.powf(next_step as f64));
                    let scale_v = 1.0 / (1.0 - beta2.powf(next_step as f64));
                    let next_m = first
                        .as_tensor()
                        .affine(beta1, 0.0)?
                        .add(&g.affine(1.0 - beta1, 0.0)?)?;
                    let next_v = second
                        .as_tensor()
                        .affine(beta2, 0.0)?
                        .add(&g.sqr()?.affine(1.0 - beta2, 0.0)?)?;
                    let m_hat = next_m.affine(scale_m, 0.0)?;
                    let v_hat = next_v.affine(scale_v, 0.0)?;
                    let next_theta = p
                        .var
                        .as_tensor()
                        .affine(1.0 - lr * wd, 0.0)?
                        .sub(&m_hat.div(&(v_hat.sqrt()? + eps)?)?.affine(lr, 0.0)?)?;
                    p.var.set(&next_theta)?;
                    first.set(&next_m)?;
                    second.set(&next_v)?;
                    p.adam_step = Some(next_step);
                }
                OptKind::Muon => {
                    let momentum = moment_data
                        .get(&format!("muon.momentum.{}", p.name))
                        .expect("muon momentum");
                    let (new_m, delta) = muon_update(
                        g,
                        momentum.as_tensor(),
                        self.muon_momentum,
                        lr,
                        self.muon_rms_scale,
                    )?;
                    let orig = p.var.as_tensor();
                    let shape = orig.dims();
                    let view = matrix_view(orig)?;
                    let (rows, cols) = view.dims2()?;
                    let delta = delta.reshape((rows, cols))?;
                    let updated = view.affine(1.0 - lr * wd, 0.0)?.add(&delta)?;
                    let next = if orig.rank() >= 3 {
                        updated.reshape(shape)?
                    } else {
                        updated
                    };
                    p.var.set(&next)?;
                    momentum.set(&new_m)?;
                }
            }
        }
        Ok(())
    }

    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// Update the shared AdamW/Muon learning rate without rebuilding either
    /// optimizer state. Foundation-v2 calls this once per WSD update.
    pub fn set_learning_rate(&mut self, lr: f64) -> Result<()> {
        if !(lr.is_finite() && lr > 0.0) {
            bail!("optimizer learning rate must be finite and > 0, got {lr}");
        }
        self.adam.lr = lr;
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        // Adam clocks are hot-loop host state. Materialize them only at a
        // checkpoint boundary so training does not allocate and upload one
        // scalar tensor per Adam parameter on every update.
        let moment_data = self.moments.data().lock().unwrap();
        for p in &self.params {
            if let Some(step) = p.adam_step {
                let checkpoint_step = moment_data
                    .get(&format!("adam.step.{}", p.name))
                    .expect("adam parameter step");
                checkpoint_step.set(&Tensor::new(step, checkpoint_step.device())?)?;
            }
        }
        drop(moment_data);
        self.moments
            .save(path)
            .with_context(|| format!("save optimizer {}", path.display()))
    }

    pub fn load(&mut self, path: &Path, step_t: usize) -> Result<()> {
        let device = self
            .params
            .first()
            .map(|p| p.var.device().clone())
            .ok_or_else(|| anyhow::anyhow!("optimizer has no params"))?;
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        let expected: Vec<(String, Var)> = {
            let md = self.moments.data().lock().unwrap();
            let mut vars: Vec<_> = md
                .iter()
                .map(|(name, var)| (name.clone(), var.clone()))
                .collect();
            vars.sort_by(|a, b| a.0.cmp(&b.0));
            vars
        };
        let expected_names: Vec<_> = expected.iter().map(|(name, _)| name.clone()).collect();
        let mut checkpoint_names: Vec<_> =
            mmap.tensors().into_iter().map(|(name, _)| name).collect();
        checkpoint_names.sort();
        if checkpoint_names != expected_names {
            let missing: Vec<_> = expected_names
                .iter()
                .filter(|name| !checkpoint_names.contains(name))
                .cloned()
                .collect();
            let extra: Vec<_> = checkpoint_names
                .iter()
                .filter(|name| !expected_names.contains(name))
                .cloned()
                .collect();
            bail!(
                "optimizer checkpoint tensor names mismatch: missing={missing:?} extra={extra:?}"
            );
        }

        let mut loaded = Vec::with_capacity(expected.len());
        for (name, var) in expected {
            let tensor = mmap
                .load(&name, &device)
                .with_context(|| format!("load optimizer tensor {name}"))?;
            if tensor.dims() != var.shape().dims() {
                bail!(
                    "optimizer checkpoint shape mismatch for {name}: checkpoint={:?} optimizer={:?}",
                    tensor.dims(),
                    var.shape().dims()
                );
            }
            if tensor.dtype() != var.dtype() {
                bail!(
                    "optimizer checkpoint dtype mismatch for {name}: checkpoint={:?} optimizer={:?}",
                    tensor.dtype(),
                    var.dtype()
                );
            }
            loaded.push((var, tensor));
        }
        for (var, tensor) in loaded {
            var.set(&tensor)?;
        }
        let moment_data = self.moments.data().lock().unwrap();
        for p in &mut self.params {
            if p.kind == OptKind::Adam {
                p.adam_step = Some(
                    moment_data[&format!("adam.step.{}", p.name)]
                        .as_tensor()
                        .to_scalar::<u32>()?,
                );
            }
        }
        self.step_t = step_t;
        Ok(())
    }
}

/// Add microbatch gradients into an accumulator (scaled backward per micro, then merge).
pub fn accumulate_grad_store(acc: &mut GradStore, micro: GradStore) -> Result<()> {
    acc.extend(micro).map_err(Into::into)
}

/// Retain and accumulate only trainable-parameter gradients from one microbatch.
///
/// Backward also returns gradients for intermediate tensors. Keeping those stores
/// across microbatches retains the corresponding graphs and needlessly increases
/// accelerator memory, so move only `VarMap` entries into the durable accumulator.
pub fn accumulate_parameter_gradients(
    acc: &mut Option<GradStore>,
    mut micro: GradStore,
    varmap: &VarMap,
) -> Result<()> {
    let mut parameter_grads = GradStore::default();
    for var in varmap.all_vars() {
        let tensor = var.as_tensor();
        if let Some(gradient) = micro.remove(tensor) {
            parameter_grads.insert(tensor, gradient);
        }
    }
    match acc {
        Some(accumulated) => accumulate_grad_store(accumulated, parameter_grads),
        None => {
            *acc = Some(parameter_grads);
            Ok(())
        }
    }
}

/// GPU-side global L2 gradient clip (no host download of full grad vectors).
pub fn clip_gradients_gpu(grads: &mut GradStore, varmap: &VarMap, max_norm: f64) -> Result<()> {
    clip_gradients_gpu_with_stats(grads, varmap, max_norm).map(|_| ())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientClipStats {
    pub pre_clip_norm: f64,
    pub scale: f64,
}

/// Clip gradients and expose the global scale that mediates treatment effects.
/// Fails closed on a non-finite norm; callers that can skip an update should
/// use [`try_clip_gradients_gpu_with_stats`].
pub fn clip_gradients_gpu_with_stats(
    grads: &mut GradStore,
    varmap: &VarMap,
    max_norm: f64,
) -> Result<GradientClipStats> {
    try_clip_gradients_gpu_with_stats(grads, varmap, max_norm)?
        .ok_or_else(|| anyhow::anyhow!("gradient norm is not finite: NaN"))
}

/// Like [`clip_gradients_gpu_with_stats`] but reports a non-finite global
/// norm as `Ok(None)` so the caller can skip the update (standard loss-scaler
/// behavior for data/state-specific NaN gradients) instead of dying.
pub fn try_clip_gradients_gpu_with_stats(
    grads: &mut GradStore,
    varmap: &VarMap,
    max_norm: f64,
) -> Result<Option<GradientClipStats>> {
    let mut sum_sq: Option<Tensor> = None;
    for var in varmap.all_vars() {
        let t = var.as_tensor();
        if let Some(g) = grads.get(t) {
            let sq = g.to_dtype(DType::F32)?.sqr()?.sum_all()?;
            sum_sq = Some(match sum_sq {
                None => sq,
                Some(acc) => acc.add(&sq)?,
            });
        }
    }
    let Some(sum_sq) = sum_sq else {
        return Ok(Some(GradientClipStats {
            pre_clip_norm: 0.0,
            scale: 1.0,
        }));
    };
    let norm = sum_sq.sqrt()?.to_scalar::<f32>()? as f64;
    if !norm.is_finite() {
        return Ok(None);
    }
    if norm <= max_norm {
        return Ok(Some(GradientClipStats {
            pre_clip_norm: norm,
            scale: 1.0,
        }));
    }
    let scale = max_norm / norm;
    for var in varmap.all_vars() {
        let t = var.as_tensor();
        if let Some(g) = grads.get(t) {
            grads.insert(t, g.affine(scale, 0.0)?);
        }
    }
    Ok(Some(GradientClipStats {
        pre_clip_norm: norm,
        scale,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::muon::MUON_RMS_SCALE;
    use candle_core::Device;
    use candle_nn::{linear, Module, VarBuilder, VarMap};

    #[test]
    fn hybrid_optimizer_steps_without_error() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lin = linear(4, 2, vb.pp("lin"))?;
        let mut opt = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: 1e-3,
                ..ParamsAdamW::default()
            },
            0.95,
            MUON_RMS_SCALE,
        )?;
        let x = Tensor::randn(0f32, 1.0, (2, 4), &device)?;
        let loss = lin.forward(&x)?.sqr()?.mean_all()?;
        opt.step(&loss.backward()?)?;
        assert_eq!(opt.step_t(), 1);
        Ok(())
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
        Ok(())
    }

    #[test]
    fn ema_eval_error_restores_training_weights() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let initial = Tensor::new(&[1.0f32, -2.0], &device)?;
        let weight = Var::from_tensor(&initial)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("probe.weight".into(), weight.clone());
        let mut ema = ModelEma::new(&varmap, 0.9)?;
        let training = Tensor::new(&[3.0f32, 4.0], &device)?;
        weight.set(&training)?;

        let error = ema
            .with_eval_weights(&varmap, || -> Result<()> {
                assert_tensor_close(weight.as_tensor(), &initial, 0.0)?;
                bail!("expected evaluation failure")
            })
            .expect_err("evaluation failure must propagate");

        assert!(error.to_string().contains("expected evaluation failure"));
        assert_tensor_close(weight.as_tensor(), &training, 0.0)?;
        assert!(ema.eval_backup.is_none());
        Ok(())
    }

    #[test]
    fn ema_eval_panic_restores_training_weights() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let initial = Tensor::new(&[1.0f32, -2.0], &device)?;
        let weight = Var::from_tensor(&initial)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("probe.weight".into(), weight.clone());
        let mut ema = ModelEma::new(&varmap, 0.9)?;
        let training = Tensor::new(&[3.0f32, 4.0], &device)?;
        weight.set(&training)?;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ema.with_eval_weights(&varmap, || -> Result<()> {
                assert_tensor_close(weight.as_tensor(), &initial, 0.0)?;
                panic!("expected evaluation panic")
            });
        }));

        assert!(panic.is_err());
        assert_tensor_close(weight.as_tensor(), &training, 0.0)?;
        assert!(ema.eval_backup.is_none());
        Ok(())
    }

    #[test]
    fn hybrid_optimizer_rejects_invalid_muon_hyperparameters() {
        let varmap = VarMap::new();
        for (momentum, rms_scale, expected) in [
            (f64::NAN, MUON_RMS_SCALE, "muon_momentum"),
            (-0.1, MUON_RMS_SCALE, "muon_momentum"),
            (1.0, MUON_RMS_SCALE, "muon_momentum"),
            (0.95, f64::NAN, "muon_rms_scale"),
            (0.95, 0.0, "muon_rms_scale"),
            (0.95, -1.0, "muon_rms_scale"),
        ] {
            let error = match CheckpointHybridOptimizer::new(
                &varmap,
                ParamsAdamW::default(),
                momentum,
                rms_scale,
            ) {
                Ok(_) => panic!("invalid {expected} was accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[test]
    fn muon_decay_applies_before_unattenuated_update() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let initial = Tensor::from_vec(
            (0..64)
                .map(|index| index as f32 / 8.0 - 4.0)
                .collect::<Vec<_>>(),
            (8, 8),
            &device,
        )?;
        let weight = Var::from_tensor(&initial)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("block.weight".into(), weight.clone());
        let lr = 0.2;
        let weight_decay = 0.5;
        let momentum = 0.9;
        let mut optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr,
                weight_decay,
                ..ParamsAdamW::default()
            },
            momentum,
            MUON_RMS_SCALE,
        )?;
        let gradient = Tensor::from_vec(
            (0..64)
                .map(|index| (index % 11) as f32 - 5.0)
                .collect::<Vec<_>>(),
            (8, 8),
            &device,
        )?;
        let zero = Tensor::zeros((8, 8), DType::F32, &device)?;
        let (_, delta) = muon_update(&gradient, &zero, momentum, lr, MUON_RMS_SCALE)?;
        let expected = initial.affine(1.0 - lr * weight_decay, 0.0)?.add(&delta)?;
        let attenuated_update = initial.add(&delta)?.affine(1.0 - lr * weight_decay, 0.0)?;

        let mut gradients = GradStore::default();
        gradients.insert(weight.as_tensor(), gradient);
        optimizer.step(&gradients)?;

        assert_tensor_close(weight.as_tensor(), &expected, 1e-6)?;
        let actual = weight.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let old = attenuated_update.flatten_all()?.to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(old)
            .any(|(actual, old)| (actual - old).abs() > 1e-5));
        Ok(())
    }

    #[test]
    fn adam_late_first_gradient_uses_parameter_local_bias_step() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let always = Var::from_tensor(&Tensor::new(&[1f32, -1.0], &device)?)?;
        let late = Var::from_tensor(&Tensor::new(&[1f32, -1.0], &device)?)?;
        {
            let mut data = varmap.data().lock().unwrap();
            data.insert("always.bias".into(), always.clone());
            data.insert("late.bias".into(), late.clone());
        }
        let mut optimizer = CheckpointHybridOptimizer::new(
            &varmap,
            ParamsAdamW {
                lr: 0.1,
                beta1: 0.5,
                beta2: 0.25,
                eps: 0.0,
                weight_decay: 0.0,
            },
            0.9,
            MUON_RMS_SCALE,
        )?;

        for _ in 0..2 {
            let mut gradients = GradStore::default();
            gradients.insert(always.as_tensor(), Tensor::new(&[2f32, -4.0], &device)?);
            optimizer.step(&gradients)?;
        }
        let mut gradients = GradStore::default();
        gradients.insert(late.as_tensor(), Tensor::new(&[2f32, -4.0], &device)?);
        optimizer.step(&gradients)?;

        assert_tensor_close(
            late.as_tensor(),
            &Tensor::new(&[0.9f32, -0.9], &device)?,
            1e-6,
        )?;
        assert_eq!(
            optimizer
                .params
                .iter()
                .find(|parameter| parameter.name == "always.bias")
                .and_then(|parameter| parameter.adam_step),
            Some(2)
        );
        assert_eq!(
            optimizer
                .params
                .iter()
                .find(|parameter| parameter.name == "late.bias")
                .and_then(|parameter| parameter.adam_step),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn exact_optimizer_load_rejects_missing_individual_moment() -> Result<()> {
        let device = Device::Cpu;
        let model_vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&model_vars, DType::F32, &device);
        let _lin = linear(4, 2, vb.pp("lin"))?;
        let mut optimizer = CheckpointHybridOptimizer::new(
            &model_vars,
            ParamsAdamW::default(),
            0.95,
            MUON_RMS_SCALE,
        )?;
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-optimizer-missing-moment-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        let complete = root.join("complete.safetensors");
        let missing = root.join("missing.safetensors");
        optimizer.save(&complete)?;

        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&complete)? };
        let incomplete = VarMap::new();
        let omitted = "first_moment.lin.bias";
        for (name, _) in mmap.tensors() {
            if name != omitted {
                incomplete
                    .data()
                    .lock()
                    .unwrap()
                    .insert(name.clone(), Var::from_tensor(&mmap.load(&name, &device)?)?);
            }
        }
        incomplete.save(&missing)?;
        let err = optimizer
            .load(&missing, 1)
            .expect_err("exact optimizer load must reject a missing moment");
        assert!(err.to_string().contains(omitted), "{err:#}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn exact_optimizer_load_rejects_extra_shape_and_dtype_mismatches() -> Result<()> {
        let device = Device::Cpu;
        let model_vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&model_vars, DType::F32, &device);
        let _lin = linear(4, 2, vb.pp("lin"))?;
        let mut optimizer = CheckpointHybridOptimizer::new(
            &model_vars,
            ParamsAdamW::default(),
            0.95,
            MUON_RMS_SCALE,
        )?;
        let root = std::env::temp_dir().join(format!(
            "tofy-p2-optimizer-exact-mismatch-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        let complete = root.join("complete.safetensors");
        optimizer.save(&complete)?;
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(&complete)? };
        let changed_name = mmap
            .tensors()
            .first()
            .map(|(name, _)| name.clone())
            .expect("optimizer has moments");

        let extra = VarMap::new();
        for (name, _) in mmap.tensors() {
            extra
                .data()
                .lock()
                .unwrap()
                .insert(name.clone(), Var::from_tensor(&mmap.load(&name, &device)?)?);
        }
        extra.data().lock().unwrap().insert(
            "unexpected".into(),
            Var::from_tensor(&Tensor::zeros((1,), DType::F32, &device)?)?,
        );
        let path = root.join("extra.safetensors");
        extra.save(&path)?;
        let err = optimizer
            .load(&path, 1)
            .expect_err("extra moment must reject");
        assert!(err.to_string().contains("extra"), "{err:#}");

        let wrong_shape = VarMap::new();
        for (name, _) in mmap.tensors() {
            let tensor = if name == changed_name {
                Tensor::zeros(
                    (mmap.load(&name, &device)?.elem_count() + 1,),
                    DType::F32,
                    &device,
                )?
            } else {
                mmap.load(&name, &device)?
            };
            wrong_shape
                .data()
                .lock()
                .unwrap()
                .insert(name, Var::from_tensor(&tensor)?);
        }
        let path = root.join("shape.safetensors");
        wrong_shape.save(&path)?;
        let err = optimizer
            .load(&path, 1)
            .expect_err("shape mismatch must reject");
        assert!(err.to_string().contains("shape mismatch"), "{err:#}");

        let wrong_dtype = VarMap::new();
        for (name, _) in mmap.tensors() {
            let tensor = if name == changed_name {
                mmap.load(&name, &device)?.to_dtype(DType::F64)?
            } else {
                mmap.load(&name, &device)?
            };
            wrong_dtype
                .data()
                .lock()
                .unwrap()
                .insert(name, Var::from_tensor(&tensor)?);
        }
        let path = root.join("dtype.safetensors");
        wrong_dtype.save(&path)?;
        let err = optimizer
            .load(&path, 1)
            .expect_err("dtype mismatch must reject");
        assert!(err.to_string().contains("dtype mismatch"), "{err:#}");

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn gradient_clip_rejects_non_finite_norm() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let variable = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &device)?)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("probe.weight".to_string(), variable.clone());
        let mut grads = GradStore::default();
        grads.insert(
            variable.as_tensor(),
            Tensor::from_vec(vec![f32::NAN, 0.0], (2,), &device)?,
        );

        let error = clip_gradients_gpu(&mut grads, &varmap, 1.0).unwrap_err();
        assert!(error.to_string().contains("gradient norm is not finite"));
        Ok(())
    }
    /// The checked variant reports a non-finite norm as a skippable outcome
    /// instead of an error, and leaves finite gradients clipped as before.
    #[test]
    fn try_clip_reports_non_finite_norm_as_skippable() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let variable = Var::from_tensor(&Tensor::zeros((2,), DType::F32, &device)?)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("probe.weight".to_string(), variable.clone());
        let mut grads = GradStore::default();
        grads.insert(
            variable.as_tensor(),
            Tensor::from_vec(vec![f32::NAN, 0.0], (2,), &device)?,
        );
        assert!(try_clip_gradients_gpu_with_stats(&mut grads, &varmap, 1.0)?.is_none());

        let mut finite = GradStore::default();
        finite.insert(
            variable.as_tensor(),
            Tensor::from_vec(vec![3.0f32, 4.0], (2,), &device)?,
        );
        let stats = try_clip_gradients_gpu_with_stats(&mut finite, &varmap, 1.0)?
            .expect("finite gradients clip normally");
        assert!((stats.pre_clip_norm - 5.0).abs() < 1e-6);
        assert!((stats.scale - 0.2).abs() < 1e-6);
        Ok(())
    }
}
