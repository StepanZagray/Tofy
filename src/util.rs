//! Shared helpers.

use anyhow::{Context, Result};
use candle_core::{
    backprop::GradStore, shape::ShapeWithOneHole, DType, Device, Tensor, Var, WithDType,
};
use candle_nn::{Optimizer, ParamsAdamW, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Format parameter count: <1M → k, <1B → M, ≥1B → B.
pub fn format_params(n: usize) -> String {
    const K: usize = 1_000;
    const M: usize = 1_000_000;
    const B: usize = 1_000_000_000;
    if n < M {
        format!("{:.1}k", n as f64 / K as f64)
    } else if n < B {
        format!("{:.2}M", n as f64 / M as f64)
    } else {
        format!("{:.2}B", n as f64 / B as f64)
    }
}

pub fn resolve_train_dtype(device: &Device, requested: DType) -> DType {
    match device {
        Device::Cpu => DType::F32,
        _ => requested,
    }
}

pub fn parse_dtype_name(value: &str) -> Option<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f16" | "float16" | "fp16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        "f32" | "float32" | "fp32" => Some(DType::F32),
        _ => None,
    }
}

pub fn requested_runtime_dtype() -> Option<DType> {
    std::env::var("TOFY_RUNTIME_DTYPE")
        .ok()
        .and_then(|value| parse_dtype_name(&value))
        .or_else(|| {
            std::env::var("TOFY_TRAIN_DTYPE")
                .ok()
                .and_then(|value| parse_dtype_name(&value))
        })
}

pub fn resolve_runtime_dtype(device: &Device) -> DType {
    let requested = requested_runtime_dtype().unwrap_or(DType::BF16);
    resolve_train_dtype(device, requested)
}

pub fn checkpoint_float_dtype(path: &Path) -> Result<Option<DType>> {
    use candle_core::safetensors::MmapedSafetensors;

    let mapped = unsafe { MmapedSafetensors::new(path) }
        .with_context(|| format!("read safetensors metadata from {:?}", path))?;
    for (_, view) in mapped.tensors() {
        let dtype = DType::try_from(view.dtype())
            .with_context(|| format!("read tensor dtype from {:?}", path))?;
        if dtype.is_float() {
            return Ok(Some(dtype));
        }
    }
    Ok(None)
}

pub fn scalar_f32(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
        .map_err(Into::into)
}

pub fn vec1_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Into::into)
}

pub fn vec2_f32(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    tensor
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()
        .map_err(Into::into)
}

pub fn from_vec_like<T: WithDType, S: ShapeWithOneHole>(
    data: Vec<T>,
    shape: S,
    reference: &Tensor,
) -> Result<Tensor> {
    Tensor::from_vec(data, shape, reference.device())?
        .to_dtype(reference.dtype())
        .map_err(Into::into)
}

pub fn ensure_same_dtype(lhs: &Tensor, rhs: &Tensor, context: &str) -> Result<()> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    if lhs_dtype != rhs_dtype {
        anyhow::bail!(
            "dtype mismatch in {}: lhs={:?} rhs={:?}",
            context,
            lhs_dtype,
            rhs_dtype
        );
    }
    Ok(())
}

pub fn cast_varmap_dtype(varmap: &mut VarMap, dtype: DType) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock varmap for dtype cast"))?;
    for var in data.values() {
        if var.as_tensor().dtype() != dtype {
            let casted = var.as_tensor().to_dtype(dtype)?;
            var.set(&casted)?;
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct NamedVar {
    pub name: String,
    pub var: Var,
}

pub fn named_train_vars(varmap: &VarMap) -> Result<Vec<NamedVar>> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock varmap for named train vars"))?;
    let mut vars = data
        .iter()
        .filter(|(_, var)| var.dtype().is_float())
        .map(|(name, var)| NamedVar {
            name: name.clone(),
            var: var.clone(),
        })
        .collect::<Vec<_>>();
    vars.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(vars)
}

pub fn frozen_tensors_from_varmap(varmap: &VarMap) -> Result<HashMap<String, Tensor>> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to lock varmap for frozen tensors"))?;
    Ok(data
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect())
}

#[derive(Debug)]
struct AdamWVarState {
    name: String,
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

/// AdamW with explicit save/load support for optimizer moments and step count.
///
/// Candle's built-in AdamW keeps moment buffers private, so long-running training could only
/// restart from weights. This optimizer mirrors Candle's update rule but stores moments as F32 for
/// better mixed-precision stability and resumability.
#[derive(Debug)]
pub struct ResumableAdamW {
    vars: Vec<AdamWVarState>,
    step_t: usize,
    params: ParamsAdamW,
}

impl ResumableAdamW {
    pub fn new_lr_named(vars: Vec<NamedVar>, learning_rate: f64) -> Result<Self> {
        let params = optimizer_params_from_env(learning_rate);
        let vars = vars
            .into_iter()
            .filter(|entry| entry.var.dtype().is_float())
            .map(|entry| {
                let shape = entry.var.shape().clone();
                let device = entry.var.device().clone();
                Ok(AdamWVarState {
                    name: entry.name,
                    var: entry.var,
                    first_moment: Var::zeros(shape.clone(), DType::F32, &device)?,
                    second_moment: Var::zeros(shape, DType::F32, &device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    pub fn step_t(&self) -> usize {
        self.step_t
    }

    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let device = self
            .vars
            .first()
            .map(|state| state.var.device().clone())
            .unwrap_or(Device::Cpu);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "__step".to_string(),
            Tensor::from_vec(vec![self.step_t as i64], (1,), &device)?,
        );
        tensors.insert(
            "__lr".to_string(),
            Tensor::from_vec(vec![self.params.lr], (1,), &device)?,
        );
        for state in &self.vars {
            tensors.insert(
                format!("{}.first_moment", state.name),
                state.first_moment.as_tensor().clone(),
            );
            tensors.insert(
                format!("{}.second_moment", state.name),
                state.second_moment.as_tensor().clone(),
            );
        }
        let tmp_path = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
        candle_core::safetensors::save(&tensors, &tmp_path)?;
        fs::rename(tmp_path, path)?;
        Ok(())
    }

    pub fn load_state<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let device = self
            .vars
            .first()
            .map(|state| state.var.device().clone())
            .unwrap_or(Device::Cpu);
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("failed to load optimizer state from {:?}", path))?;
        if let Some(step) = tensors.get("__step") {
            let step_values = step.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
            self.step_t = step_values.first().copied().unwrap_or(0).max(0) as usize;
        }
        for state in &mut self.vars {
            let m_key = format!("{}.first_moment", state.name);
            let v_key = format!("{}.second_moment", state.name);
            let m = tensors.get(&m_key).ok_or_else(|| {
                anyhow::anyhow!("optimizer state {:?} missing tensor {}", path, m_key)
            })?;
            let v = tensors.get(&v_key).ok_or_else(|| {
                anyhow::anyhow!("optimizer state {:?} missing tensor {}", path, v_key)
            })?;
            state.first_moment.set(&m.to_dtype(DType::F32)?)?;
            state.second_moment.set(&v.to_dtype(DType::F32)?)?;
        }
        Ok(())
    }
}

impl Optimizer for ResumableAdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> candle_core::Result<Self> {
        let vars = vars
            .into_iter()
            .enumerate()
            .map(|(idx, var)| NamedVar {
                name: format!("var_{idx}"),
                var,
            })
            .collect::<Vec<_>>();
        let mut optimizer =
            Self::new_lr_named(vars, params.lr).map_err(candle_core::Error::wrap)?;
        optimizer.params = params;
        Ok(optimizer)
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for state in &self.vars {
            if let Some(g) = grads.get(&state.var) {
                let theta_f32 = state.var.as_tensor().to_dtype(DType::F32)?;
                let grad_f32 = g.to_dtype(DType::F32)?;
                let m = state.first_moment.as_tensor();
                let v = state.second_moment.as_tensor();
                let next_m = ((m * beta1)? + (&grad_f32 * (1.0 - beta1))?)?;
                let next_v = ((v * beta2)? + (grad_f32.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let decayed_theta = (theta_f32 * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (decayed_theta - (adjusted_grad * lr)?)?;
                state.first_moment.set(&next_m)?;
                state.second_moment.set(&next_v)?;
                state.var.set(&next_theta.to_dtype(state.var.dtype())?)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct MuonVarState {
    name: String,
    var: Var,
    momentum: Var,
}

#[derive(Debug)]
enum HybridMuonVarState {
    Muon(MuonVarState),
    AdamW(AdamWVarState),
}

/// Hybrid Muon optimizer for LLM-style training.
///
/// Muon is applied only to 2D hidden-weight matrices. AdamW remains the
/// fallback for embeddings, prediction heads, normalization vectors, and
/// biases, where matrix orthogonalization is not the right geometry.
#[derive(Debug)]
pub struct ResumableHybridMuon {
    vars: Vec<HybridMuonVarState>,
    step_t: usize,
    params: ParamsAdamW,
    muon_momentum: f64,
    muon_ns_steps: usize,
    muon_rms_scale: f64,
}

impl ResumableHybridMuon {
    pub fn new_lr_named(vars: Vec<NamedVar>, learning_rate: f64) -> Result<Self> {
        let params = optimizer_params_from_env(learning_rate);
        let muon_momentum = env_f64("TOFY_MUON_MOMENTUM", 0.95).clamp(0.0, 0.9999);
        let muon_ns_steps = env_usize("TOFY_MUON_NS_STEPS", 5).clamp(1, 20);
        let muon_rms_scale = env_f64("TOFY_MUON_RMS_SCALE", 0.18).max(1e-6);
        let mut muon_count = 0usize;
        let mut adamw_count = 0usize;
        let vars = vars
            .into_iter()
            .filter(|entry| entry.var.dtype().is_float())
            .map(|entry| {
                let shape = entry.var.shape().clone();
                let device = entry.var.device().clone();
                if should_use_muon(&entry.name, &entry.var) {
                    muon_count += 1;
                    Ok(HybridMuonVarState::Muon(MuonVarState {
                        name: entry.name,
                        var: entry.var,
                        momentum: Var::zeros(shape, DType::F32, &device)?,
                    }))
                } else {
                    adamw_count += 1;
                    Ok(HybridMuonVarState::AdamW(AdamWVarState {
                        name: entry.name,
                        var: entry.var,
                        first_moment: Var::zeros(shape.clone(), DType::F32, &device)?,
                        second_moment: Var::zeros(shape, DType::F32, &device)?,
                    }))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        println!(
            "Optimizer: hybrid Muon+AdamW (muon_vars={muon_count}, adamw_vars={adamw_count}, momentum={muon_momentum:.4}, ns_steps={muon_ns_steps}, rms_scale={muon_rms_scale:.4})"
        );
        Ok(Self {
            vars,
            step_t: 0,
            params,
            muon_momentum,
            muon_ns_steps,
            muon_rms_scale,
        })
    }

    pub fn step_t(&self) -> usize {
        self.step_t
    }

    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let device = self
            .vars
            .first()
            .map(|state| match state {
                HybridMuonVarState::Muon(state) => state.var.device().clone(),
                HybridMuonVarState::AdamW(state) => state.var.device().clone(),
            })
            .unwrap_or(Device::Cpu);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "__step".to_string(),
            Tensor::from_vec(vec![self.step_t as i64], (1,), &device)?,
        );
        tensors.insert(
            "__lr".to_string(),
            Tensor::from_vec(vec![self.params.lr], (1,), &device)?,
        );
        tensors.insert(
            "__optimizer_kind".to_string(),
            Tensor::from_vec(vec![1i64], (1,), &device)?,
        );
        for state in &self.vars {
            match state {
                HybridMuonVarState::Muon(state) => {
                    tensors.insert(
                        format!("{}.muon_momentum", state.name),
                        state.momentum.as_tensor().clone(),
                    );
                }
                HybridMuonVarState::AdamW(state) => {
                    tensors.insert(
                        format!("{}.first_moment", state.name),
                        state.first_moment.as_tensor().clone(),
                    );
                    tensors.insert(
                        format!("{}.second_moment", state.name),
                        state.second_moment.as_tensor().clone(),
                    );
                }
            }
        }
        let tmp_path = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
        candle_core::safetensors::save(&tensors, &tmp_path)?;
        fs::rename(tmp_path, path)?;
        Ok(())
    }

    pub fn load_state<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let device = self
            .vars
            .first()
            .map(|state| match state {
                HybridMuonVarState::Muon(state) => state.var.device().clone(),
                HybridMuonVarState::AdamW(state) => state.var.device().clone(),
            })
            .unwrap_or(Device::Cpu);
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("failed to load optimizer state from {:?}", path))?;
        if let Some(step) = tensors.get("__step") {
            let step_values = step.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
            self.step_t = step_values.first().copied().unwrap_or(0).max(0) as usize;
        }
        let mut loaded = 0usize;
        let mut warm_started_from_adamw = 0usize;
        for state in &mut self.vars {
            match state {
                HybridMuonVarState::Muon(state) => {
                    let key = format!("{}.muon_momentum", state.name);
                    if let Some(momentum) = tensors.get(&key) {
                        state.momentum.set(&momentum.to_dtype(DType::F32)?)?;
                        loaded += 1;
                        continue;
                    }
                    let adamw_key = format!("{}.first_moment", state.name);
                    if let Some(momentum) = tensors.get(&adamw_key) {
                        state.momentum.set(&momentum.to_dtype(DType::F32)?)?;
                        warm_started_from_adamw += 1;
                    }
                }
                HybridMuonVarState::AdamW(state) => {
                    let m_key = format!("{}.first_moment", state.name);
                    let v_key = format!("{}.second_moment", state.name);
                    if let Some(m) = tensors.get(&m_key) {
                        state.first_moment.set(&m.to_dtype(DType::F32)?)?;
                    }
                    if let Some(v) = tensors.get(&v_key) {
                        state.second_moment.set(&v.to_dtype(DType::F32)?)?;
                    }
                    loaded += 1;
                }
            }
        }
        if warm_started_from_adamw > 0 {
            println!(
                "Hybrid Muon optimizer warm-started {warm_started_from_adamw} Muon momenta from AdamW first moments"
            );
        }
        println!(
            "Hybrid Muon optimizer state loaded from {} (matched_vars={loaded})",
            path.display()
        );
        Ok(())
    }
}

impl Optimizer for ResumableHybridMuon {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> candle_core::Result<Self> {
        let vars = vars
            .into_iter()
            .enumerate()
            .map(|(idx, var)| NamedVar {
                name: format!("var_{idx}"),
                var,
            })
            .collect::<Vec<_>>();
        let mut optimizer =
            Self::new_lr_named(vars, params.lr).map_err(candle_core::Error::wrap)?;
        optimizer.params = params;
        Ok(optimizer)
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let weight_decay = self.params.weight_decay;
        let lr_decay = lr * weight_decay;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for state in &self.vars {
            match state {
                HybridMuonVarState::Muon(state) => {
                    let Some(g) = grads.get(&state.var) else {
                        continue;
                    };
                    let theta_f32 = state.var.as_tensor().to_dtype(DType::F32)?;
                    let grad_f32 = g.to_dtype(DType::F32)?;
                    let next_m = ((state.momentum.as_tensor() * self.muon_momentum)? + &grad_f32)?;
                    let nesterov_update = ((&next_m * self.muon_momentum)? + &grad_f32)?;
                    let orthogonal_update =
                        muon_orthogonal_update(&nesterov_update, self.muon_ns_steps)?;
                    let update = rescale_update_rms(&orthogonal_update, self.muon_rms_scale)?;
                    let decayed_theta = (theta_f32 * (1f64 - lr_decay))?;
                    let next_theta = (decayed_theta - (update * lr)?)?;
                    state.momentum.set(&next_m)?;
                    state.var.set(&next_theta.to_dtype(state.var.dtype())?)?;
                }
                HybridMuonVarState::AdamW(state) => {
                    let Some(g) = grads.get(&state.var) else {
                        continue;
                    };
                    let theta_f32 = state.var.as_tensor().to_dtype(DType::F32)?;
                    let grad_f32 = g.to_dtype(DType::F32)?;
                    let m = state.first_moment.as_tensor();
                    let v = state.second_moment.as_tensor();
                    let next_m = ((m * beta1)? + (&grad_f32 * (1.0 - beta1))?)?;
                    let next_v = ((v * beta2)? + (grad_f32.sqr()? * (1.0 - beta2))?)?;
                    let m_hat = (&next_m * scale_m)?;
                    let v_hat = (&next_v * scale_v)?;
                    let decayed_theta = (theta_f32 * (1f64 - lr_decay))?;
                    let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                    let next_theta = (decayed_theta - (adjusted_grad * lr)?)?;
                    state.first_moment.set(&next_m)?;
                    state.second_moment.set(&next_v)?;
                    state.var.set(&next_theta.to_dtype(state.var.dtype())?)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum TrainOptimizer {
    AdamW(ResumableAdamW),
    HybridMuon(ResumableHybridMuon),
}

impl TrainOptimizer {
    pub fn new_lr_named(vars: Vec<NamedVar>, learning_rate: f64) -> Result<Self> {
        match optimizer_kind_from_env()? {
            OptimizerKind::AdamW => Ok(Self::AdamW(ResumableAdamW::new_lr_named(
                vars,
                learning_rate,
            )?)),
            OptimizerKind::HybridMuon => Ok(Self::HybridMuon(ResumableHybridMuon::new_lr_named(
                vars,
                learning_rate,
            )?)),
        }
    }

    pub fn step_t(&self) -> usize {
        match self {
            Self::AdamW(opt) => opt.step_t(),
            Self::HybridMuon(opt) => opt.step_t(),
        }
    }

    pub fn load_state<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        match self {
            Self::AdamW(opt) => opt.load_state(path),
            Self::HybridMuon(opt) => opt.load_state(path),
        }
    }

    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        match self {
            Self::AdamW(opt) => opt.save_state(path),
            Self::HybridMuon(opt) => opt.save_state(path),
        }
    }
}

impl Optimizer for TrainOptimizer {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> candle_core::Result<Self> {
        let vars = vars
            .into_iter()
            .enumerate()
            .map(|(idx, var)| NamedVar {
                name: format!("var_{idx}"),
                var,
            })
            .collect::<Vec<_>>();
        let mut optimizer =
            Self::new_lr_named(vars, params.lr).map_err(candle_core::Error::wrap)?;
        match &mut optimizer {
            Self::AdamW(opt) => opt.params = params,
            Self::HybridMuon(opt) => opt.params = params,
        }
        Ok(optimizer)
    }

    fn learning_rate(&self) -> f64 {
        match self {
            Self::AdamW(opt) => opt.learning_rate(),
            Self::HybridMuon(opt) => opt.learning_rate(),
        }
    }

    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::AdamW(opt) => opt.set_learning_rate(lr),
            Self::HybridMuon(opt) => opt.set_learning_rate(lr),
        }
    }

    fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        match self {
            Self::AdamW(opt) => opt.step(grads),
            Self::HybridMuon(opt) => opt.step(grads),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OptimizerKind {
    AdamW,
    HybridMuon,
}

fn optimizer_kind_from_env() -> Result<OptimizerKind> {
    let value = std::env::var("TOFY_OPTIMIZER").unwrap_or_else(|_| "adamw".to_string());
    match value.trim().to_ascii_lowercase().as_str() {
        "adamw" | "adam" => Ok(OptimizerKind::AdamW),
        "muon" | "hybrid_muon" | "muon_adamw" | "hybrid-muon" => Ok(OptimizerKind::HybridMuon),
        other => anyhow::bail!("unsupported TOFY_OPTIMIZER={other:?}; expected adamw or muon"),
    }
}

fn optimizer_params_from_env(learning_rate: f64) -> ParamsAdamW {
    let mut params = ParamsAdamW {
        lr: learning_rate,
        ..ParamsAdamW::default()
    };
    params.beta1 = env_f64("TOFY_ADAMW_BETA1", params.beta1);
    params.beta2 = env_f64("TOFY_ADAMW_BETA2", params.beta2);
    params.eps = env_f64("TOFY_ADAMW_EPS", params.eps);
    params.weight_decay = env_f64("TOFY_WEIGHT_DECAY", params.weight_decay);
    params
}

fn should_use_muon(name: &str, var: &Var) -> bool {
    let dims = var.shape().dims();
    if dims.len() != 2 || dims[0] < 2 || dims[1] < 2 {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    ![
        "embed",
        "embedding",
        "token",
        "lm_head",
        "head",
        "norm",
        "ln",
        "bias",
        "pos",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn muon_orthogonal_update(update: &Tensor, steps: usize) -> candle_core::Result<Tensor> {
    let (rows, cols) = update.dims2()?;
    let transposed = rows > cols;
    let mut x = if transposed {
        update.transpose(0, 1)?.contiguous()?
    } else {
        update.clone()
    };
    let x_norm = x.sqr()?.sum_all()?.sqrt()?;
    x = x.broadcast_div(&(x_norm + 1e-7)?)?;
    // Newton-Schulz coefficients used by current Muon recipes.
    let a = 3.4445;
    let b = -4.7750;
    let c = 2.0315;
    for _ in 0..steps {
        let xxt = x.matmul(&x.transpose(0, 1)?.contiguous()?)?;
        let xxt2 = xxt.matmul(&xxt)?;
        let poly = ((&xxt * b)? + (xxt2 * c)?)?;
        let correction = poly.matmul(&x)?;
        x = ((&x * a)? + correction)?;
    }
    if transposed {
        x.transpose(0, 1)?.contiguous()
    } else {
        Ok(x)
    }
}

fn rescale_update_rms(update: &Tensor, target_rms: f64) -> candle_core::Result<Tensor> {
    let rms = update.sqr()?.mean_all()?.sqrt()?;
    let normalized = update.broadcast_div(&(rms + 1e-7)?)?;
    normalized * target_rms
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingResumeState {
    pub stage: String,
    pub step: usize,
    pub best_metric: f32,
    pub best_aux_metric: f32,
    pub saved_checkpoint: bool,
}

impl TrainingResumeState {
    pub fn new(stage: &str) -> Self {
        Self {
            stage: stage.to_string(),
            step: 0,
            best_metric: f32::MAX,
            best_aux_metric: f32::MAX,
            saved_checkpoint: false,
        }
    }
}

pub fn checkpoint_sidecar_path(model_path: &Path, stage: &str, suffix: &str) -> PathBuf {
    PathBuf::from(format!(
        "{}.{}.{}",
        model_path.to_string_lossy(),
        stage,
        suffix
    ))
}

pub fn resume_stage_name(default_stage: &str) -> String {
    std::env::var("TOFY_RUN_STAGE_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_stage.to_string())
}

pub fn save_varmap_atomic(varmap: &VarMap, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
    varmap.save(&tmp_path)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

pub fn load_resume_state(path: &Path, expected_stage: &str) -> Result<Option<TrainingResumeState>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)?;
    let state: TrainingResumeState = serde_json::from_str(&text)?;
    if state.stage != expected_stage {
        anyhow::bail!(
            "resume state {:?} is for stage {:?}, expected {:?}",
            path,
            state.stage,
            expected_stage
        );
    }
    Ok(Some(state))
}

pub fn save_resume_state(path: &Path, state: &TrainingResumeState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
    fs::write(&tmp_path, serde_json::to_string_pretty(state)?)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

pub fn create_run_dir(stage: &str) -> Result<String> {
    if let Some(group) = std::env::var("TOFY_RUN_GROUP")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let stage_name = std::env::var("TOFY_RUN_STAGE_NAME")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| stage.to_string());
        let root = PathBuf::from("runs").join(group);
        let path = root.join(stage_name);
        fs::create_dir_all(&path)?;
        return Ok(path.to_string_lossy().to_string());
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("runs/{stage}/{stamp}");
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn accumulate_scaled_gradients(
    accumulated: &mut Option<GradStore>,
    train_vars: &[Var],
    loss: &Tensor,
    grad_accum_steps: usize,
) -> Result<()> {
    let scale = 1.0 / grad_accum_steps.max(1) as f64;
    let scaled_loss = if (scale - 1.0).abs() < f64::EPSILON {
        loss.clone()
    } else {
        loss.affine(scale, 0.0)?
    };
    let mut grads = scaled_loss.backward()?;
    if let Some(existing) = accumulated.as_mut() {
        for var in train_vars {
            if let Some(grad) = grads.remove(var) {
                if let Some(prev) = existing.remove(var) {
                    existing.insert(var, prev.broadcast_add(&grad)?);
                } else {
                    existing.insert(var, grad);
                }
            }
        }
    } else {
        *accumulated = Some(grads);
    }
    Ok(())
}

pub fn optimizer_step_from_accumulated<O: Optimizer>(
    optimizer: &mut O,
    accumulated: &mut Option<GradStore>,
) -> Result<()> {
    if let Some(grads) = accumulated.take() {
        optimizer.step(&grads)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct VramSnapshot {
    pub used_mb: f32,
    pub free_mb: f32,
    pub total_mb: f32,
    pub peak_used_mb: f32,
}

#[derive(Default)]
pub struct VramTracker {
    peak_used_mb: f32,
    last_snapshot: Option<VramSnapshot>,
}

impl VramTracker {
    pub fn sample(&mut self) -> Option<VramSnapshot> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        let first_line = stdout.lines().next()?.trim();
        let mut parts = first_line
            .split(',')
            .map(|part| part.trim().parse::<f32>().ok());
        let used_mb = parts.next().flatten()?;
        let free_mb = parts.next().flatten()?;
        let total_mb = parts.next().flatten()?;
        self.peak_used_mb = self.peak_used_mb.max(used_mb);
        let snapshot = VramSnapshot {
            used_mb,
            free_mb,
            total_mb,
            peak_used_mb: self.peak_used_mb,
        };
        self.last_snapshot = Some(snapshot.clone());
        Some(snapshot)
    }

    pub fn write_summary(&self, run_dir: &str, stage: &str) -> Result<()> {
        let path = Path::new(run_dir).join("memory_summary.txt");
        let content = if let Some(snapshot) = &self.last_snapshot {
            format!(
                "stage={stage}\nlast_used_mb={:.2}\nlast_free_mb={:.2}\ntotal_mb={:.2}\npeak_used_mb={:.2}\n",
                snapshot.used_mb,
                snapshot.free_mb,
                snapshot.total_mb,
                self.peak_used_mb
            )
        } else {
            format!("stage={stage}\npeak_used_mb={:.2}\n", self.peak_used_mb)
        };
        fs::write(path, content)?;
        Ok(())
    }
}
