//! Resumable hybrid Muon + AdamW optimizer for P2 training.

use crate::p2::muon::{matrix_view, muon_update, uses_muon};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use candle_nn::VarMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptKind {
    Adam,
    Muon,
}

struct ParamOpt {
    name: String,
    var: Var,
    kind: OptKind,
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
                }
                OptKind::Adam
            };
            params.push(ParamOpt { name, var, kind });
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
        let scale_m = 1.0 / (1.0 - beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.step_t as i32));
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
                        wd,
                        self.muon_rms_scale,
                    )?;
                    let orig = p.var.as_tensor();
                    let shape = orig.dims();
                    let view = matrix_view(orig)?;
                    let (rows, cols) = view.dims2()?;
                    let delta = delta.reshape((rows, cols))?;
                    let updated = view.add(&delta)?;
                    let next = if orig.rank() >= 3 {
                        updated.reshape(shape)?
                    } else {
                        updated
                    };
                    let next = next.affine(1.0 - lr * wd, 0.0)?;
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

    pub fn save(&self, path: &Path) -> Result<()> {
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
        let mut md = self.moments.data().lock().unwrap();
        for (name, var) in md.iter_mut() {
            if let Ok(t) = mmap.load(name, &device) {
                var.set(&t)?;
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

/// GPU-side global L2 gradient clip (no host download of full grad vectors).
pub fn clip_gradients_gpu(grads: &mut GradStore, varmap: &VarMap, max_norm: f64) -> Result<()> {
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
        return Ok(());
    };
    let norm = sum_sq.sqrt()?.to_scalar::<f32>()? as f64;
    if !norm.is_finite() {
        bail!("gradient norm is not finite: {norm}");
    }
    if norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / norm;
    for var in varmap.all_vars() {
        let t = var.as_tensor();
        if let Some(g) = grads.get(t) {
            grads.insert(t, g.affine(scale, 0.0)?);
        }
    }
    Ok(())
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
}
