use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::data::{
    compose_mixed_stream_batch, MixedStreamConfig, V5DataSplit, GOAL_FEATURES_DIM,
};
use tofy::p2::experiment::ConsumerReadoutTopology;
use tofy::p2::model::{zero_action_film_projections, ModelConfig, WorldModel};
use tofy::p2::train::{
    foundation_v2_training_loss, reinit_varmap_deterministic, FoundationV2ObjectiveConfig,
};

#[test]
fn mixed_batch_runs_the_complete_foundation_v2_objective_on_cpu() -> Result<()> {
    let device = Device::Cpu;
    let mixed = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size: 50,
            seed: 17,
            symmetry_augmentation: false,
            ..MixedStreamConfig::default()
        },
        0.0,
        0,
        V5DataSplit::Train,
    )?;
    assert!(!mixed.factual_group_ranges().is_empty());

    let vars = VarMap::new();
    let model = WorldModel::new(
        ModelConfig {
            patch_size: 4,
            hidden_dim: 8,
            action_dim: 8,
            goal_dim: GOAL_FEATURES_DIM,
            inner_steps: 1,
            outer_steps: 1,
            residual_y_update: true,
            warm_start_y: true,
            spatial_action_field: true,
            world_core_v4: true,
            world_core_v5: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..ModelConfig::default()
        },
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 23)?;
    zero_action_film_projections(&vars)?;

    let losses = foundation_v2_training_loss(
        &model,
        &mixed,
        &device,
        FoundationV2ObjectiveConfig::default(),
    )?;
    for (name, loss) in [
        ("total", &losses.total),
        ("pred_ce", &losses.pred_ce),
        ("gate", &losses.gate),
        ("latent", &losses.latent),
        ("enc_ce", &losses.enc_ce),
        ("separation", &losses.separation),
        ("pull", &losses.pull),
        ("inverse_action", &losses.inverse_action),
        ("ep", &losses.ep),
        ("event", &losses.event),
        ("q", &losses.q),
        ("reliability", &losses.reliability),
    ] {
        let value = loss.to_scalar::<f32>()?;
        assert!(value.is_finite(), "{name} must be finite, got {value}");
        assert!(value > 0.0, "{name} must be active, got {value}");
    }
    assert!(losses.factual_groups > 0);
    assert!(losses.equivalent_pairs > 0);
    assert!(losses.distinct_pairs > 0);
    assert!(losses.inverse_action_rows > 0);
    assert_eq!(losses.rollout_fragments, 0);
    assert_eq!(losses.rollout.to_scalar::<f32>()?, 0.0);
    Ok(())
}
