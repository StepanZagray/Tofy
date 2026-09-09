//! Learned action binding from public action-effect records, without a solver.

use super::{
    model::{rms_norm, TransformerBlock},
    ACTIONS,
};
use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

pub const RECORDS: usize = 4;
pub const INPUT_WIDTH: usize = 7;
pub const WIDTH: usize = 256;
pub const HEADS: usize = 4;
pub const LAYERS: usize = 2;
pub const MAX_LOOPS: usize = 8;
pub const PARAMETERS: usize = 1_580_804;
pub const EQUIVARIANT_PARAMETERS: usize = 1_579_265;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Legacy,
    Equivariant,
}

impl ModelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Equivariant => "equivariant",
        }
    }
    pub fn parameter_count(self) -> usize {
        match self {
            Self::Legacy => PARAMETERS,
            Self::Equivariant => EQUIVARIANT_PARAMETERS,
        }
    }
    pub fn tensor_count(self) -> usize {
        match self {
            Self::Legacy => 29,
            Self::Equivariant => 28,
        }
    }
}

pub enum Binder {
    Legacy(ControlBinder),
    Equivariant(EquivariantBinder),
}

impl Binder {
    pub fn new(kind: ModelKind, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(match kind {
            ModelKind::Legacy => Self::Legacy(ControlBinder::new(vb)?),
            ModelKind::Equivariant => Self::Equivariant(EquivariantBinder::new(vb)?),
        })
    }
    pub fn forward(&self, records: &Tensor, loops: usize) -> Result<Tensor> {
        match self {
            Self::Legacy(model) => model.forward(records, loops),
            Self::Equivariant(model) => model.forward(records, loops),
        }
    }
}

/// Four action tokens share every learned operation. The missing action keeps
/// zero observed effect and an explicit zero observation mask; no effect is inferred.
pub struct EquivariantBinder {
    input_projection: Linear,
    blocks: Vec<TransformerBlock>,
    policy_head: Linear,
}

impl EquivariantBinder {
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let blocks = (0..LAYERS)
            .map(|layer| TransformerBlock::new(WIDTH, HEADS, vb.pp(format!("block_{layer}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_projection: linear(5, WIDTH, vb.pp("input_projection"))?,
            blocks,
            policy_head: linear(WIDTH, 1, vb.pp("policy_head"))?,
        })
    }

    fn action_tokens(records: &Tensor) -> Result<Tensor> {
        let (batch, count, width) = records.dims3()?;
        ensure!(
            batch > 0 && count == RECORDS && width == INPUT_WIDTH && records.dtype() == DType::F32,
            "binder requires nonempty F32 [B, {RECORDS}, {INPUT_WIDTH}] records"
        );
        let support = records.narrow(1, 0, 3)?;
        let actions = support
            .narrow(2, 2, ACTIONS)?
            .transpose(1, 2)?
            .contiguous()?;
        let effects = actions.matmul(&support.narrow(2, 0, 2)?.contiguous()?)?;
        let observed = actions.sum_keepdim(2)?;
        let desired = records
            .narrow(1, 3, 1)?
            .narrow(2, 0, 2)?
            .broadcast_as((batch, ACTIONS, 2))?;
        Ok(Tensor::cat(&[effects, observed, desired], 2)?.contiguous()?)
    }

    pub fn forward(&self, records: &Tensor, loops: usize) -> Result<Tensor> {
        ensure!(
            (1..=MAX_LOOPS).contains(&loops),
            "binder loops must be in 1..={MAX_LOOPS}"
        );
        let recalled = self
            .input_projection
            .forward(&Self::action_tokens(records)?)?;
        let mut state = Tensor::zeros_like(&recalled)?;
        for _ in 0..loops {
            state = state.add(&recalled)?;
            for block in &self.blocks {
                state = block.forward(&state)?;
            }
        }
        Ok(self.policy_head.forward(&rms_norm(&state)?)?.squeeze(2)?)
    }
}

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
        initialize_fixture(&vars)?;
        Ok((model, vars))
    }

    fn initialize_fixture(vars: &VarMap) -> Result<()> {
        let device = Device::Cpu;
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
        Ok(())
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
    fn equivariant_tokens_do_not_complete_the_missing_effect() -> Result<()> {
        let tokens = EquivariantBinder::action_tokens(&records(1)?)?.to_vec3::<f32>()?;
        assert_eq!(
            tokens[0],
            vec![
                vec![0., -1., 1., 1., 0.],
                vec![-1., 0., 1., 1., 0.],
                vec![0., 1., 1., 1., 0.],
                vec![0., 0., 0., 1., 0.],
            ]
        );
        Ok(())
    }

    #[test]
    fn equivariant_model_relabels_all_actions_and_ignores_support_order() -> Result<()> {
        let vars = VarMap::new();
        let model =
            EquivariantBinder::new(VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu))?;
        initialize_fixture(&vars)?;
        assert_eq!(
            vars.all_vars()
                .iter()
                .map(|v| v.elem_count())
                .sum::<usize>(),
            EQUIVARIANT_PARAMETERS
        );
        assert_eq!(EQUIVARIANT_PARAMETERS, 24 * WIDTH * WIDTH + 25 * WIDTH + 1);
        assert_eq!(vars.all_vars().len(), 28);
        assert!(!vars.data().lock().unwrap().contains_key("readout_token"));
        let input = records(1)?;
        let raw = input.to_vec3::<f32>()?;
        let permutations = (0..4)
            .flat_map(|a| {
                (0..4).flat_map(move |b| {
                    (0..4).flat_map(move |c| {
                        (0..4).filter_map(move |d| {
                            let p = [a, b, c, d];
                            p.iter()
                                .enumerate()
                                .all(|(i, x)| !p[..i].contains(x))
                                .then_some(p)
                        })
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(permutations.len(), 24);
        for loops in [1, 4, 8] {
            let expected = values(&model.forward(&input, loops)?)?;
            for p in &permutations {
                let mut renamed = raw.clone();
                for support in 0..3 {
                    for action in 0..4 {
                        renamed[0][support][2 + action] = raw[0][support][2 + p[action]];
                    }
                }
                let renamed = Tensor::from_vec(
                    renamed.into_iter().flatten().flatten().collect::<Vec<_>>(),
                    (1, 4, 7),
                    &Device::Cpu,
                )?;
                let actual = values(&model.forward(&renamed, loops)?)?;
                for action in 0..4 {
                    let target = expected[p[action]];
                    assert!((actual[action] - target).abs() <= 1e-5 + 1e-5 * target.abs());
                }
            }
            for p in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let parts = p
                    .into_iter()
                    .chain([3])
                    .map(|i| input.narrow(1, i, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                assert_eq!(
                    values(&model.forward(&Tensor::cat(&parts, 1)?, loops)?)?,
                    expected
                );
            }
        }
        for loops in [0, 9] {
            assert!(model.forward(&input, loops).is_err());
        }
        assert!(model
            .forward(&Tensor::zeros((1, 3, 7), DType::F32, &Device::Cpu)?, 4)
            .is_err());
        Ok(())
    }

    #[test]
    fn equivariant_records_and_shared_weights_receive_gradients() -> Result<()> {
        let vars = VarMap::new();
        let model =
            EquivariantBinder::new(VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu))?;
        initialize_fixture(&vars)?;
        let input = Var::from_tensor(&records(2)?)?;
        let logits = model.forward(&input, 4)?;
        assert_eq!(logits.dims(), [2, 4]);
        let grads = logits.sqr()?.mean_all()?.backward()?;
        for record in grads
            .get(&input)
            .unwrap()
            .to_vec3::<f32>()?
            .iter()
            .flatten()
        {
            assert!(record.iter().all(|v| v.is_finite()) && record.iter().any(|&v| v != 0.));
        }
        for (name, var) in vars.data().lock().unwrap().iter() {
            let gradient = values(grads.get(var).unwrap_or_else(|| panic!("missing {name}")))?;
            assert!(gradient.iter().all(|v| v.is_finite()), "nonfinite {name}");
            if name.ends_with("weight") {
                assert!(gradient.iter().any(|&v| v != 0.), "zero {name}");
            }
        }
        Ok(())
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
        assert_eq!(WIDTH / HEADS, 64);
        assert_eq!(PARAMETERS, 24 * WIDTH * WIDTH + 31 * WIDTH + 4);
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
