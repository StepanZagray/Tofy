use anyhow::{ensure, Result};
use candle_core::{DType, Tensor, Var, D};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const WIDTH: usize = 128;
pub const TOKENS: usize = 64;
pub const ACTIONS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Spatial,
    Cls,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Spatial => "spatial",
            Self::Cls => "cls",
        }
    }
    pub fn parameters(self) -> usize {
        match self {
            Self::Spatial => 1284,
            Self::Cls => 1334,
        }
    }
    pub fn shape(self, rows: usize) -> Vec<usize> {
        match self {
            Self::Spatial => vec![rows, TOKENS, WIDTH],
            Self::Cls => vec![rows, WIDTH],
        }
    }
}

pub enum Head {
    Spatial { queries: Tensor, output: Linear },
    Cls { hidden: Linear, output: Linear },
}

pub struct Output {
    pub logits: Tensor,
    pub attention: Option<Tensor>,
    pub pooled: Tensor,
}

impl Head {
    pub fn new(kind: Kind, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(match kind {
            Kind::Spatial => Self::Spatial {
                queries: vb.get((2, WIDTH), "queries")?,
                output: linear(2 * WIDTH, ACTIONS, vb.pp("output"))?,
            },
            Kind::Cls => Self::Cls {
                hidden: linear(WIDTH, 10, vb.pp("hidden"))?,
                output: linear(10, ACTIONS, vb.pp("output"))?,
            },
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Output> {
        ensure!(
            features.dtype() == DType::F32 && !features.track_op(),
            "readout features must be frozen F32 tensors"
        );
        let rows = features.dim(0)?;
        let (attention, pooled, output) = match self {
            Self::Spatial { queries, output } => {
                ensure!(
                    features.dims() == [rows, TOKENS, WIDTH],
                    "spatial feature shape mismatch"
                );
                let scores = queries
                    .unsqueeze(0)?
                    .broadcast_as((rows, 2, WIDTH))?
                    .contiguous()?
                    .matmul(&features.transpose(1, 2)?.contiguous()?)?
                    .affine(1.0 / (WIDTH as f64).sqrt(), 0.0)?;
                let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
                let pooled = weights.matmul(features)?.reshape((rows, 2 * WIDTH))?;
                (Some(weights), pooled, output)
            }
            Self::Cls { hidden, output } => {
                ensure!(
                    features.dims() == [rows, WIDTH],
                    "CLS feature shape mismatch"
                );
                (None, hidden.forward(features)?.silu()?, output)
            }
        };
        Ok(Output {
            logits: output.forward(&pooled)?,
            attention,
            pooled,
        })
    }

    pub fn query_separation(&self) -> Result<Option<f32>> {
        match self {
            Self::Spatial { queries, .. } => Ok(Some(
                queries
                    .get(0)?
                    .sub(&queries.get(1)?)?
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_scalar()?,
            )),
            Self::Cls { .. } => Ok(None),
        }
    }
}

pub fn named(vars: &VarMap) -> Vec<(String, Var)> {
    let mut named = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

/// Exact existing per-name initializer; query entries are independent draws.
pub fn initialize(varmap: &VarMap, seed: u64) -> Result<()> {
    for (name, var) in named(varmap) {
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

pub fn cross_entropy(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    ensure!(
        logits.dims() == [labels.dim(0)?, ACTIONS] && labels.dtype() == DType::U32,
        "CE label/logit shape mismatch"
    );
    Ok(candle_nn::ops::log_softmax(logits, D::Minus1)?
        .gather(&labels.unsqueeze(1)?, 1)?
        .mean_all()?
        .neg()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn artificial(kind: Kind) -> Result<Tensor> {
        let n = kind.shape(8).iter().product();
        Ok(Tensor::from_vec(
            (0..n)
                .map(|i| ((i * 31 % 211) as f32 - 105.0) / 53.0)
                .collect::<Vec<_>>(),
            kind.shape(8),
            &Device::Cpu,
        )?)
    }
    #[test]
    fn heads_have_exact_counts_repeatable_independent_queries_and_real_gradients() -> Result<()> {
        for kind in [Kind::Spatial, Kind::Cls] {
            let vars = VarMap::new();
            let head = Head::new(
                kind,
                VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
            )?;
            initialize(&vars, 0)?;
            assert_eq!(
                named(&vars)
                    .iter()
                    .map(|(_, v)| v.elem_count())
                    .sum::<usize>(),
                kind.parameters()
            );
            let before = named(&vars)
                .iter()
                .map(|(_, v)| v.flatten_all()?.to_vec1::<f32>())
                .collect::<candle_core::Result<Vec<_>>>()?;
            initialize(&vars, 0)?;
            assert_eq!(
                before,
                named(&vars)
                    .iter()
                    .map(|(_, v)| v.flatten_all()?.to_vec1::<f32>())
                    .collect::<candle_core::Result<Vec<_>>>()?
            );
            if kind == Kind::Spatial {
                assert!(head.query_separation()?.unwrap() > 0.1);
            }
            let features = artificial(kind)?;
            let original = features.flatten_all()?.to_vec1::<f32>()?;
            let output = head.forward(&features)?;
            let labels = Tensor::new(&[0u32, 1, 2, 3, 0, 1, 2, 3], &Device::Cpu)?;
            let grads = cross_entropy(&output.logits, &labels)?.backward()?;
            assert!(grads.get(&features).is_none());
            for (name, var) in named(&vars) {
                let values = grads
                    .get(&var)
                    .expect("every head parameter has a gradient")
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|x| x.is_finite()));
                assert!(values.iter().any(|&x| x != 0.0), "zero gradient {name}");
                if name == "queries" {
                    for query in values.chunks_exact(WIDTH) {
                        assert!(query.iter().any(|&x| x != 0.0));
                    }
                }
            }
            assert_eq!(features.flatten_all()?.to_vec1::<f32>()?, original);
            assert!(head
                .forward(Var::from_tensor(&features)?.as_tensor())
                .is_err());
        }
        Ok(())
    }

    #[test]
    fn spatial_pooling_uses_all_tokens_and_is_invariant_to_their_joint_permutation() -> Result<()> {
        let vars = VarMap::new();
        let head = Head::new(
            Kind::Spatial,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        initialize(&vars, 0)?;
        let features = artificial(Kind::Spatial)?;
        let indices = Tensor::new(
            &(0..TOKENS as u32).rev().collect::<Vec<_>>()[..],
            &Device::Cpu,
        )?;
        let a = head.forward(&features)?;
        let b = head.forward(&features.index_select(&indices, 1)?)?;
        assert!(
            a.logits
                .sub(&b.logits)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                < 1e-5
        );
        let weights = a.attention.unwrap();
        assert_eq!(weights.dims(), [8, 2, TOKENS]);
        assert!(weights.min_all()?.to_scalar::<f32>()? > 0.0);
        assert!(
            weights
                .sum(D::Minus1)?
                .affine(1.0, -1.0)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                < 1e-6
        );
        Ok(())
    }
}
