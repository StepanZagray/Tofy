//! Shared helpers.

use anyhow::Result;
use candle_core::{
    backprop::GradStore, shape::ShapeWithOneHole, DType, Device, Tensor, Var, WithDType,
};
use candle_nn::{Optimizer, VarMap};
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

pub fn resolve_runtime_dtype(device: &Device) -> DType {
    let requested = std::env::var("TOFY_RUNTIME_DTYPE")
        .ok()
        .and_then(|value| parse_dtype_name(&value))
        .or_else(|| {
            std::env::var("TOFY_TRAIN_DTYPE")
                .ok()
                .and_then(|value| parse_dtype_name(&value))
        })
        .unwrap_or(DType::F32);
    resolve_train_dtype(device, requested)
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
