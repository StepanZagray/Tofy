
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::data::{compose_mixed_stream_batch, foundation_v2_stream_schedule, MixedStreamConfig, V5DataSplit};
use tofy::p2::model::{init_copy_bypass_gate, restore_copy_gate_bias_prior, zero_action_film_projections, WorldModel};
use tofy::p2::train::{foundation_v2_training_loss, reinit_varmap_deterministic, FoundationV2ObjectiveConfig, TrainConfig};

#[test]
fn bundle_treatments_first_update_gradient_is_finite_and_bounded() -> Result<()> {
    // The deterministic post-fix fixture is ~704; retain modest headroom for
    // backend reduction-order variation while catching conditioning regressions.
    const MAX_FIRST_UPDATE_GRADIENT_L2: f64 = 800.0;
    let device = Device::Cpu;
    let mut cfg = TrainConfig::default();
    cfg.apply_foundation_v2_recipe();
    cfg.seed = 9993;
    cfg.physical_batch = 32;
    cfg.copy_bypass_gate = true;
    cfg.grid_scaled_action_impulse = true;
    cfg.copy_gate_bias_prior = Some(0.02);
    cfg.decode_composition = tofy::p2::grounding::DecodeComposition::JointCopyMixture;
    cfg.allow_multi_treatment_arm = true;
    let varmap = VarMap::new();
    let model = WorldModel::new(cfg.model_config(), VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
    reinit_varmap_deterministic(&varmap, 9993)?;
    zero_action_film_projections(&varmap)?;
    init_copy_bypass_gate(&varmap)?;
    restore_copy_gate_bias_prior(&varmap, cfg.copy_gate_bias_prior)?;
    let mixed = compose_mixed_stream_batch(
        &MixedStreamConfig { batch_size: 32, seed: 9993, schedule: foundation_v2_stream_schedule, ..MixedStreamConfig::default() },
        0.0, 0, V5DataSplit::Train,
    )?;
    let losses = foundation_v2_training_loss(&model, &mixed, &device, FoundationV2ObjectiveConfig {
        ep_weight: 0.01, sigreg_projections: 8, sigreg_knots: 5, sigreg_seed: 1,
        rollout_enabled: true, split_ce_weighting: Default::default(), split_ce_changed_budget: None,
        capture_mechanism_seams: false,
    })?;
    let grads = losses.total.backward()?;
    let mut nonfinite = Vec::new();
    let mut gradient_squared_l2 = 0.0;
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if let Some(g) = grads.get(var.as_tensor()) {
            let s = g.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            if !s.is_finite() { nonfinite.push(format!("{name}={s}")); }
            gradient_squared_l2 +=
                f64::from(g.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    assert!(nonfinite.is_empty(), "non-finite grads: {nonfinite:?}");
    let gradient_l2 = gradient_squared_l2.sqrt();
    assert!(
        gradient_l2 <= MAX_FIRST_UPDATE_GRADIENT_L2,
        "copy-bypass init gradient L2 {gradient_l2} exceeds conditioning bound"
    );
    Ok(())
}
