//! Learned action binding from public action-effect records, without a solver.

use super::{
    model::{rms_norm, TransformerBlock},
    ACTIONS,
};
use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub const RECORDS: usize = 4;
pub const INPUT_WIDTH: usize = 7;
pub const WIDTH: usize = 64;
pub const HEADS: usize = 4;
pub const LAYERS: usize = 2;
pub const MAX_LOOPS: usize = 8;
pub const PARAMETERS: usize = 100_292;

/// Two transformer blocks reused at every loop, followed by a four-action head.
/// No positional encoding distinguishes the support records: reordering them
/// preserves the CLS output in exact arithmetic (up to floating-point reduction
/// differences in execution). Initialization and record validation are caller-owned.
pub struct ControlBinder {
    input_projection: Linear,
    readout_token: Tensor,
    blocks: Vec<TransformerBlock>,
    policy_head: Linear,
}

impl ControlBinder {
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let blocks = (0..LAYERS)
            .map(|layer| TransformerBlock::new(WIDTH, HEADS, vb.pp(format!("block_{layer}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_projection: linear(INPUT_WIDTH, WIDTH, vb.pp("input_projection"))?,
            readout_token: vb.get((1, 1, WIDTH), "readout_token")?,
            blocks,
            policy_head: linear(WIDTH, ACTIONS, vb.pp("policy_head"))?,
        })
    }

    /// Maps F32 [batch, 4, 7] records to [batch, 4] action logits.
    /// Three support rows carry [dx, dy, action-one-hot(4), 0]; the query carries
    /// [desired_dx, desired_dy, 0, 0, 0, 0, 1]. These fields are learned inputs;
    /// forward does not parse actions, match effects, or receive target labels.
    /// Inputs remain differentiable for a future learned visual adapter.
    pub fn forward(&self, records: &Tensor, loops: usize) -> Result<Tensor> {
        ensure!(
            (1..=MAX_LOOPS).contains(&loops),
            "binder loops must be in 1..={MAX_LOOPS}"
        );
        let (batch, count, width) = records.dims3()?;
        ensure!(
            batch > 0 && count == RECORDS && width == INPUT_WIDTH && records.dtype() == DType::F32,
            "binder requires nonempty F32 [B, {RECORDS}, {INPUT_WIDTH}] records"
        );
        let projected = self.input_projection.forward(records)?;
        let readout = self.readout_token.broadcast_as((batch, 1, WIDTH))?;
        let recalled_input = Tensor::cat(&[&readout, &projected], 1)?;
        let mut state = Tensor::zeros_like(&recalled_input)?;
        for _ in 0..loops {
            state = state.add(&recalled_input)?;
            for block in &self.blocks {
                state = block.forward(&state)?;
            }
        }
        let state = rms_norm(&state)?;
        let readout = state.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        self.policy_head.forward(&readout).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    use candle_nn::VarMap;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    fn fixture() -> Result<(ControlBinder, VarMap)> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let model = ControlBinder::new(VarBuilder::from_varmap(&vars, DType::F32, &device))?;
        // Deterministic synthetic parameters; no dataset or optimizer is used.
        let named = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for (name, var) in named {
            let values = (0..var.elem_count())
                .map(|_| {
                    if name.ends_with(".bias") {
                        0.0
                    } else {
                        rng.random_range(-0.05f32..0.05)
                    }
                })
                .collect::<Vec<_>>();
            var.set(&Tensor::from_vec(values, var.shape(), &device)?)?;
        }
        Ok((model, vars))
    }

    fn records(batch: usize) -> Result<Tensor> {
        let one: [f32; RECORDS * INPUT_WIDTH] = [
            0., -1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., -1., 0., 0., 1., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 1.,
        ];
        let mut values = Vec::with_capacity(batch * RECORDS * INPUT_WIDTH);
        for b in 0..batch {
            let mut row = one;
            if b % 2 == 1 {
                row[3 * INPUT_WIDTH] = -1.;
            }
            values.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(
            values,
            (batch, RECORDS, INPUT_WIDTH),
            &Device::Cpu,
        )?)
    }

    fn values(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn shapes_bounds_and_parameter_sharing_are_fixed() -> Result<()> {
        let (model, vars) = fixture()?;
        let names = || {
            vars.data()
                .lock()
                .unwrap()
                .iter()
                .map(|(name, var)| (name.clone(), var.elem_count()))
                .collect::<BTreeMap<_, _>>()
        };
        let before = names();
        assert_eq!(before.len(), 29);
        assert_eq!(before.values().sum::<usize>(), PARAMETERS);
        assert!(before.contains_key("block_0.attention.query.weight"));
        assert!(before.contains_key("block_1.attention.query.weight"));
        assert!(!before.keys().any(|name| name.contains("loop_")));
        for batch in [1, 3] {
            for loops in [1, 2, 4, 8] {
                let logits = model.forward(&records(batch)?, loops)?;
                assert_eq!(logits.dims(), &[batch, ACTIONS]);
                assert!(values(&logits)?.iter().all(|v| v.is_finite()));
                assert_eq!(names(), before);
            }
        }
        for loops in [0, 9, usize::MAX] {
            assert!(model.forward(&records(1)?, loops).is_err());
        }
        for shape in [vec![1, 4], vec![0, 4, 7], vec![1, 3, 7], vec![1, 4, 6]] {
            assert!(model
                .forward(&Tensor::zeros(shape, DType::F32, &Device::Cpu)?, 1)
                .is_err());
        }
        assert!(model
            .forward(&records(1)?.to_dtype(DType::F64)?, 1)
            .is_err());
        Ok(())
    }

    #[test]
    fn input_and_shared_blocks_receive_finite_nonzero_gradients() -> Result<()> {
        let (model, vars) = fixture()?;
        let input = Var::from_tensor(&records(2)?)?;
        let logits = model.forward(&input, 4)?;
        let gradients = logits.sqr()?.mean_all()?.backward()?;
        let input_grad = gradients
            .get(&input)
            .expect("input gradient missing")
            .to_vec3::<f32>()?;
        for row in input_grad.iter().flatten() {
            assert!(row.iter().all(|v| v.is_finite()));
            assert!(row.iter().any(|&v| v != 0.0), "record gradient is zero");
        }
        let named = vars.data().lock().unwrap();
        for name in [
            "input_projection.weight",
            "readout_token",
            "policy_head.weight",
            "block_0.attention.query.weight",
            "block_0.attention.value.weight",
            "block_0.mlp_in.weight",
            "block_1.attention.query.weight",
            "block_1.mlp_out.weight",
        ] {
            let gradient = values(
                gradients
                    .get(&named[name])
                    .unwrap_or_else(|| panic!("missing gradient for {name}")),
            )?;
            assert!(gradient.iter().all(|v| v.is_finite()), "nonfinite {name}");
            assert!(gradient.iter().any(|&v| v != 0.0), "zero {name}");
        }
        Ok(())
    }

    #[test]
    fn loop_depth_changes_outputs_and_shared_block_gradients() -> Result<()> {
        let (model, vars) = fixture()?;
        let input = records(2)?;
        let shallow = model.forward(&input, 1)?;
        let deep = model.forward(&input, 4)?;
        let change = shallow.sub(&deep)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            change.is_finite() && change > 1e-6,
            "loop depth was ignored"
        );
        let a = shallow.sqr()?.mean_all()?.backward()?;
        let b = deep.sqr()?.mean_all()?.backward()?;
        let named = vars.data().lock().unwrap();
        let shared = &named["block_0.attention.query.weight"];
        assert_ne!(
            values(a.get(shared).expect("shallow gradient"))?,
            values(b.get(shared).expect("deep gradient"))?
        );
        Ok(())
    }

    #[test]
    fn all_support_record_permutations_preserve_logits() -> Result<()> {
        let (model, _) = fixture()?;
        let input = records(2)?;
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for loops in [1, 4, 8] {
            let expected = values(&model.forward(&input, loops)?)?;
            for permutation in permutations {
                let parts = permutation
                    .into_iter()
                    .chain([3])
                    .map(|index| input.narrow(1, index, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let shuffled = Tensor::cat(&parts, 1)?;
                let actual = values(&model.forward(&shuffled, loops)?)?;
                for (a, e) in actual.iter().zip(&expected) {
                    assert!(
                        (a - e).abs() <= 1e-5 + 1e-5 * e.abs(),
                        "support order changed logits at depth {loops}: {a} vs {e}"
                    );
                }
            }
        }
        Ok(())
    }
}
