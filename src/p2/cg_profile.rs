//! Step-0 execution trace for candle-graph (`profile.jsonl`, trace/4).

use anyhow::Result;
use candle_core::{backprop::GradStore, DType, Tensor};
use candle_nn::VarMap;
use candle_graph::trace::schema::GradientState;
use candle_graph::{ExecutionPhase, SpanKind, TraceSession};
use std::path::{Path, PathBuf};

const ENTRYPOINT: &str = "p2::train::leworld_loss";

/// Optional candle-graph trace for the first training update only.
pub struct StepProfileCapture {
    inner: Option<TraceSession>,
}

impl StepProfileCapture {
    /// Open `output_dir/profile.jsonl` on global step 0 when no profile was emitted yet.
    pub fn begin(step: u64, profile_emitted: bool, output_dir: &Path) -> Result<Self> {
        if step != 0 || profile_emitted {
            return Ok(Self { inner: None });
        }
        let path = output_dir.join("profile.jsonl");
        let session = TraceSession::open(path, ENTRYPOINT, ExecutionPhase::Train)?;
        Ok(Self {
            inner: Some(session),
        })
    }

    pub fn active(&self) -> bool {
        self.inner.is_some()
    }

    pub fn span<'a>(
        &'a self,
        name: impl Into<String>,
        kind: SpanKind,
    ) -> Option<candle_graph::SpanGuard<'a>> {
        Some(self.inner.as_ref()?.begin_span(name, kind))
    }

    pub fn record_gradients(&self, varmap: &VarMap, grads: &GradStore) -> Result<()> {
        let Some(session) = self.inner.as_ref() else {
            return Ok(());
        };
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<_> = data.keys().cloned().collect();
        names.sort();
    for name in names {
        let var = data
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?;
        let (state, norm) = gradient_fact(grads, var.as_tensor())?;
        session.record_gradient("vb", name, state, norm)?;
    }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Option<PathBuf>> {
        let Some(session) = self.inner.take() else {
            return Ok(None);
        };
        let path = session.finish()?;
        Ok(Some(path))
    }
}

fn gradient_fact(grads: &GradStore, var: &Tensor) -> Result<(GradientState, Option<f64>)> {
    if let Some(grad) = grads.get(var) {
        let norm = grad
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()? as f64;
        if !norm.is_finite() {
            return Ok((GradientState::NonFinite, None));
        }
        if norm == 0.0 {
            return Ok((GradientState::Zero, None));
        }
        return Ok((GradientState::Present, Some(norm)));
    }
    Ok((GradientState::Missing, None))
}
