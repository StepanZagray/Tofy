//! Trainable two-query policy over the looped body's current-patch features.
//! Parameter names match the frozen spatial readout, but inputs may retain gradients.

use super::{ACTIONS, PATCH_COUNT};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Tensor, Var, D};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

pub const WIDTH: usize = 128;
pub const PARAMETERS: usize = 1284;

pub struct SpatialPolicy {
    queries: Tensor,
    output: Linear,
}

pub struct SpatialPolicyOutput {
    pub logits: Tensor,
    pub attention: Tensor,
    pub pooled: Tensor,
}

impl SpatialPolicy {
    /// Use a separate root VarMap: queries, output.weight and output.bias.
    /// Initialization/import is caller-owned, as for the existing frozen head.
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            queries: vb.get((2, WIDTH), "queries")?,
            output: linear(2 * WIDTH, ACTIONS, vb.pp("output"))?,
        })
    }

    /// Same operations and ordering as C11's spatial head; no label/role input.
    pub fn forward(&self, features: &Tensor) -> Result<SpatialPolicyOutput> {
        let (batch, tokens, width) = features.dims3()?;
        ensure!(
            batch > 0 && tokens == PATCH_COUNT && width == WIDTH,
            "spatial policy requires nonempty [B, {PATCH_COUNT}, {WIDTH}] features"
        );
        ensure!(
            features.dtype() == DType::F32,
            "spatial policy requires F32 features"
        );
        let scores = self
            .queries
            .unsqueeze(0)?
            .broadcast_as((batch, 2, WIDTH))?
            .contiguous()?
            .matmul(&features.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (WIDTH as f64).sqrt(), 0.0)?;
        let attention = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let pooled = attention.matmul(features)?.reshape((batch, 2 * WIDTH))?;
        Ok(SpatialPolicyOutput {
            logits: self.output.forward(&pooled)?,
            attention,
            pooled,
        })
    }
}

/// Sorted optimizer identities `core.<original-name>` and
/// `spatial_policy.<original-name>`. They do not rename the two VarMaps or their
/// checkpoint tensors. Ordinary policy/value/reward/successor heads are excluded.
pub fn active_parameters(core: &VarMap, policy: &VarMap) -> Result<Vec<(String, Var)>> {
    Ok(named_parameters(core, policy)?
        .into_iter()
        .filter(|(name, _)| {
            matches!(
                parameter_family(name),
                Some("core" | "spatial_queries" | "spatial_policy")
            )
        })
        .collect())
}

pub(crate) fn named_parameters(core: &VarMap, policy: &VarMap) -> Result<Vec<(String, Var)>> {
    let mut named = Vec::new();
    for (prefix, vars) in [("core", core), ("spatial_policy", policy)] {
        let vars = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("{prefix} VarMap lock poisoned"))?;
        ensure!(!vars.is_empty(), "empty {prefix} parameter population");
        for (name, var) in vars.iter() {
            let key = format!("{prefix}.{name}");
            parameter_family(&key)
                .with_context(|| format!("unclassified grounded-policy parameter {key}"))?;
            named.push((key, var.clone()));
        }
        if prefix == "spatial_policy" {
            ensure!(
                vars.len() == 3,
                "spatial policy must have exactly three named tensors"
            );
            for (name, shape) in [
                ("queries", vec![2, WIDTH]),
                ("output.weight", vec![ACTIONS, 2 * WIDTH]),
                ("output.bias", vec![ACTIONS]),
            ] {
                ensure!(
                    vars.get(name)
                        .is_some_and(|v| v.dims() == shape && v.dtype() == DType::F32),
                    "invalid spatial parameter {name}"
                );
            }
        }
    }
    named.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(named)
}

pub(crate) fn parameter_family(key: &str) -> Option<&'static str> {
    if let Some(name) = key.strip_prefix("core.") {
        return match name.split('.').next()? {
            "policy_head" => Some("policy"),
            "value_head" => Some("value"),
            "reward_head" => Some("reward"),
            name if name.starts_with("next_head_") => Some("dynamics"),
            "palette_embedding" | "patch_projection" | "metadata_projection" | "readout_token" => {
                Some("core")
            }
            name if name.starts_with("block_") => Some("core"),
            _ => None,
        };
    }
    match key {
        "spatial_policy.queries" => Some("spatial_queries"),
        "spatial_policy.output.weight" | "spatial_policy.output.bias" => Some("spatial_policy"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        model::{LoopedAgent, LoopedConfig},
        META_DIM, PATCH_PIXELS, TOKENS,
    };
    use super::*;
    use candle_core::Device;
    use candle_nn::{Optimizer, SGD};

    fn values(t: &Tensor) -> Result<Vec<f32>> {
        Ok(t.flatten_all()?.to_vec1::<f32>()?)
    }

    fn initialize(vars: &VarMap) -> Result<()> {
        for (name, var) in vars.data().lock().unwrap().iter() {
            let phase = name.bytes().map(usize::from).sum::<usize>();
            let data = (0..var.elem_count())
                .map(|i| ((i * 37 + phase) % 101) as f32 / 500.0 - 0.1)
                .collect::<Vec<_>>();
            var.set(&Tensor::from_vec(data, var.shape(), var.device())?)?;
        }
        Ok(())
    }

    #[test]
    fn batched_spatial_policy_matches_scalar_formula_and_accepts_tracked_features() -> Result<()> {
        let vars = VarMap::new();
        let policy = SpatialPolicy::new(VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu))?;
        initialize(&vars)?;
        assert_eq!(
            vars.all_vars()
                .iter()
                .map(|v| v.elem_count())
                .sum::<usize>(),
            PARAMETERS
        );
        let data = (0..3 * PATCH_COUNT * WIDTH)
            .map(|i| ((i * 19) % 127) as f32 / 63.0 - 1.0)
            .collect::<Vec<_>>();
        let feature = Var::from_tensor(&Tensor::from_vec(
            data.clone(),
            (3, PATCH_COUNT, WIDTH),
            &Device::Cpu,
        )?)?;
        let output = policy.forward(feature.as_tensor())?;
        let weights = vars.data().lock().unwrap();
        let q = values(&weights["queries"])?;
        let w = values(&weights["output.weight"])?;
        let bias = values(&weights["output.bias"])?;
        let attention = values(&output.attention)?;
        let pooled = values(&output.pooled)?;
        let logits = values(&output.logits)?;
        for b in 0..3 {
            let mut expected_pool = vec![0.0f64; 2 * WIDTH];
            for role in 0..2 {
                let score = (0..PATCH_COUNT)
                    .map(|t| {
                        (0..WIDTH)
                            .map(|d| {
                                q[role * WIDTH + d] as f64
                                    * data[(b * PATCH_COUNT + t) * WIDTH + d] as f64
                            })
                            .sum::<f64>()
                            / (WIDTH as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator = score.iter().map(|s| (s - max).exp()).sum::<f64>();
                for (t, score) in score.iter().enumerate() {
                    let mass = (score - max).exp() / denominator;
                    assert!(
                        (mass - attention[(b * 2 + role) * PATCH_COUNT + t] as f64).abs() < 1e-6
                    );
                    for d in 0..WIDTH {
                        expected_pool[role * WIDTH + d] +=
                            mass * data[(b * PATCH_COUNT + t) * WIDTH + d] as f64;
                    }
                }
            }
            for d in 0..2 * WIDTH {
                assert!((expected_pool[d] - pooled[b * 2 * WIDTH + d] as f64).abs() < 1e-6);
            }
            for a in 0..ACTIONS {
                let expected = bias[a] as f64
                    + (0..2 * WIDTH)
                        .map(|d| expected_pool[d] * w[a * 2 * WIDTH + d] as f64)
                        .sum::<f64>();
                assert!((expected - logits[b * ACTIONS + a] as f64).abs() < 1e-6);
            }
        }
        assert!(output
            .logits
            .sqr()?
            .sum_all()?
            .backward()?
            .get(feature.as_tensor())
            .is_some());
        assert!(policy
            .forward(&Tensor::zeros((1, 63, WIDTH), DType::F32, &Device::Cpu)?)
            .is_err());
        assert!(policy
            .forward(&Tensor::zeros(
                (1, PATCH_COUNT, WIDTH),
                DType::F64,
                &Device::Cpu
            )?)
            .is_err());
        Ok(())
    }

    #[test]
    fn policy_loss_updates_body_and_queries_but_not_ordinary_heads_then_restores() -> Result<()> {
        let device = Device::Cpu;
        let core = VarMap::new();
        let policy_vars = VarMap::new();
        let model = LoopedAgent::new(
            LoopedConfig {
                hidden: WIDTH,
                heads: 4,
                layers: 1,
                max_loops: 4,
            },
            VarBuilder::from_varmap(&core, DType::F32, &device),
        )?;
        let policy =
            SpatialPolicy::new(VarBuilder::from_varmap(&policy_vars, DType::F32, &device))?;
        initialize(&core)?;
        initialize(&policy_vars)?;
        let named = named_parameters(&core, &policy_vars)?;
        let before = named
            .iter()
            .map(|(_, v)| values(v))
            .collect::<Result<Vec<_>>>()?;
        let active = active_parameters(&core, &policy_vars)?;
        assert!(active.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let patches = Tensor::from_vec(
            (0..2 * TOKENS * PATCH_PIXELS)
                .map(|i| ((i / PATCH_PIXELS + i % 5) % 16) as u32)
                .collect::<Vec<_>>(),
            (2, TOKENS, PATCH_PIXELS),
            &device,
        )?;
        let metadata = Tensor::from_vec(
            (0..2 * TOKENS * META_DIM)
                .map(|i| (i % 17) as f32 / 17.0)
                .collect::<Vec<_>>(),
            (2, TOKENS, META_DIM),
            &device,
        )?;
        let features = model.forward_features(&patches, &metadata, 4)?;
        assert!(features.current.track_op());
        let output = policy.forward(&features.current)?;
        let labels = Tensor::new(&[0u32, 3], &device)?;
        let loss = candle_nn::ops::log_softmax(&output.logits, D::Minus1)?
            .gather(&labels.unsqueeze(1)?, 1)?
            .mean_all()?
            .neg()?;
        let grads = loss.backward()?;
        for (name, var) in &named {
            if active.iter().any(|(key, _)| key == name) {
                let gradient = values(
                    grads
                        .get(var)
                        .with_context(|| format!("missing gradient {name}"))?,
                )?;
                assert!(
                    gradient.iter().all(|v| v.is_finite()),
                    "nonfinite gradient {name}"
                );
                assert!(gradient.iter().any(|&v| v != 0.0), "zero gradient {name}");
            } else {
                assert!(grads.get(var).is_none(), "inactive gradient {name}");
            }
        }
        let mut optimizer = SGD::new(active.iter().map(|(_, v)| v.clone()).collect(), 0.01)?;
        optimizer.step(&grads)?;
        for prefix in ["core.", "spatial_policy."] {
            assert!(
                named
                    .iter()
                    .zip(&before)
                    .any(|((name, var), old)| name.starts_with(prefix)
                        && values(var).unwrap() != *old),
                "unchanged {prefix}"
            );
        }
        for ((name, var), old) in named.iter().zip(&before) {
            if !active.iter().any(|(key, _)| key == name) {
                assert_eq!(&values(var)?, old, "inactive update {name}");
            }
            var.set(&Tensor::from_vec(old.clone(), var.shape(), &device)?)?;
            assert_eq!(&values(var)?, old, "restore {name}");
        }
        Ok(())
    }
}
