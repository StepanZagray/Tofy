use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::model::{zero_action_film_projections, ModelConfig, WorldModel, FRAME_SIDE};
use tofy::p2::train::reinit_varmap_deterministic;

#[test]
fn zero_initialized_film_preserves_the_pre_film_patch8_forward() -> Result<()> {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        patch_size: 8,
        hidden_dim: 8,
        action_dim: 4,
        goal_dim: 6,
        inner_steps: 2,
        outer_steps: 2,
        ..ModelConfig::default()
    };
    let vars = VarMap::new();
    let model = WorldModel::new(cfg, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    reinit_varmap_deterministic(&vars, 0x5eed)?;
    zero_action_film_projections(&vars)?;
    let frames = Tensor::from_vec(
        (0..FRAME_SIDE * FRAME_SIDE)
            .map(|index| (index % 16) as u8)
            .collect::<Vec<_>>(),
        (1, 1, FRAME_SIDE, FRAME_SIDE),
        &device,
    )?;
    let actions = Tensor::from_vec(vec![6u32], (1,), &device)?;
    let coords = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device)?;
    let goals = Tensor::zeros((1, 6), DType::F32, &device)?;
    let actual = model
        .forward(&frames, &actions, &coords, &goals)?
        .y
        .flatten_all()?
        .narrow(0, 0, 16)?
        .to_vec1::<f32>()?;
    let expected = [
        0.4634512,
        0.05370098,
        0.010538837,
        -0.72074324,
        0.65817064,
        -0.3627804,
        0.42257944,
        -0.80939364,
        0.08850571,
        -0.08353961,
        -0.47270256,
        -0.40279397,
        0.23295873,
        -0.3964673,
        0.15358227,
        -0.52538896,
    ];
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "pre-FiLM output changed at {index}: actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn film_projections_start_at_exact_identity() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let _model = WorldModel::new(
        ModelConfig::default(),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    let data = vars.data().lock().unwrap();
    let film = data
        .iter()
        .filter(|(name, _)| name.starts_with("action_film_"))
        .collect::<Vec<_>>();
    assert_eq!(film.len(), 4, "gamma/beta linears each have weight+bias");
    for (name, value) in film {
        assert!(
            value
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| *value == 0.0),
            "{name} must be zero initialized"
        );
    }
    Ok(())
}
