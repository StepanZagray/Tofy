use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;
use tofy::p2::muon::{muon_shape_rescale, uses_muon, MUON_RMS_SCALE};
use tofy::p2::optimizer::{ModelEma, DEFAULT_EMA_DECAY};

#[test]
fn tiny_or_degenerate_matrices_route_to_adam() {
    assert!(uses_muon("action_proj.weight", &[128, 8]));
    assert!(!uses_muon("coord_proj.weight", &[128, 2]));
    assert!(!uses_muon("spatial_action_proj.weight", &[128, 4, 1, 1]));
    assert!(!uses_muon("narrow.weight", &[7, 128]));
    assert!(uses_muon("square.weight", &[8, 8]));
}

#[test]
fn muon_rescale_matches_point_two_times_sqrt_max_fan() -> Result<()> {
    assert_eq!(MUON_RMS_SCALE, 0.2);
    let device = Device::Cpu;
    let update = Tensor::ones((4, 9), DType::F32, &device)?;
    let scaled = muon_shape_rescale(&update, MUON_RMS_SCALE)?;
    let expected = 0.2f32 * 3.0;
    assert!(scaled
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|value| (*value - expected).abs() < 1e-6));
    Ok(())
}

#[test]
fn ema_update_and_eval_swap_follow_the_registered_decay() -> Result<()> {
    let device = Device::Cpu;
    let model_vars = VarMap::new();
    let weight = Var::from_tensor(&Tensor::new(&[1f32, 2.0], &device)?)?;
    model_vars
        .data()
        .lock()
        .unwrap()
        .insert("probe.weight".into(), weight.clone());
    let mut ema = ModelEma::new(&model_vars, 0.9)?;
    weight.set(&Tensor::new(&[3f32, 6.0], &device)?)?;
    ema.update(&model_vars)?;

    ema.swap_in_for_eval(&model_vars)?;
    let installed = weight.as_tensor().to_vec1::<f32>()?;
    assert!((installed[0] - 1.2).abs() < 1e-6);
    assert!((installed[1] - 2.4).abs() < 1e-6);
    ema.restore_after_eval(&model_vars)?;
    assert_eq!(weight.as_tensor().to_vec1::<f32>()?, vec![3.0, 6.0]);
    assert_eq!(DEFAULT_EMA_DECAY, 0.999);
    Ok(())
}
