use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::experiment::ConsumerReadoutTopology;
use tofy::p2::grounding::patch_tokens_to_pixels;
use tofy::p2::model::{
    ModelConfig, PtrmConfig, RecursionDepth, RecursionOpts, WorldModel, FRAME_SIDE,
};
use tofy::p2::train::reinit_varmap_deterministic;

fn exact_config(patch_size: usize) -> ModelConfig {
    ModelConfig {
        patch_size,
        hidden_dim: 8,
        action_dim: 8,
        goal_dim: 6,
        inner_steps: 1,
        outer_steps: 1,
        spatial_action_field: true,
        world_core_v4: true,
        consumer_readout: ConsumerReadoutTopology::SpatialQuery,
        ..ModelConfig::default()
    }
}

fn frames(device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..FRAME_SIDE * FRAME_SIDE)
            .map(|index| (index % 16) as u8)
            .collect::<Vec<_>>(),
        (1, 1, FRAME_SIDE, FRAME_SIDE),
        device,
    )
    .map_err(Into::into)
}

fn v5_config(patch_size: usize) -> ModelConfig {
    ModelConfig {
        world_core_v5: true,
        ..exact_config(patch_size)
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok(a.to_dtype(DType::F32)?
        .sub(&b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .fold(0.0, f32::max))
}

fn spatial_with_values(device: &Device, values: &[(usize, usize, f32)]) -> Result<Tensor> {
    let mut data = vec![0.0; 8 * 16 * 16];
    for &(y, x, value) in values {
        for channel in 0..8 {
            data[(channel * 16 + y) * 16 + x] = value;
        }
    }
    Tensor::from_vec(data, (1, 8, 16, 16), device).map_err(Into::into)
}

#[test]
fn patch4_is_default_and_patch8_remains_configurable() -> Result<()> {
    let device = Device::Cpu;
    for (patch_size, grid) in [(4, 16), (8, 8)] {
        let vars = VarMap::new();
        let model = WorldModel::new(
            exact_config(patch_size),
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        let latent = model.encode_state(&frames(&device)?)?;
        assert_eq!(latent.dims4()?, (1, 8, grid, grid));
        assert_eq!(
            model.exact_gameplay_logits_trainable(&latent)?.dims4()?,
            (1, FRAME_SIDE - 1, FRAME_SIDE, 16)
        );
        assert_eq!(
            model.exact_copy_gate(&latent)?.dims3()?,
            (1, FRAME_SIDE - 1, FRAME_SIDE)
        );
    }
    assert_eq!(ModelConfig::default().patch_size, 4);
    Ok(())
}

#[test]
fn patch_token_offsets_rearrange_to_their_exact_pixels() -> Result<()> {
    // Layout is [batch, patch-y, patch-x, dy, dx, channel].
    let device = Device::Cpu;
    let patch_tokens = Tensor::from_vec(
        (0..16).map(|value| value as u32).collect::<Vec<_>>(),
        (1, 2, 2, 2, 2, 1),
        &device,
    )?;
    let pixels = patch_tokens_to_pixels(&patch_tokens)?
        .squeeze(0)?
        .squeeze(2)?
        .to_vec2::<u32>()?;
    assert_eq!(
        pixels,
        vec![
            vec![0, 1, 4, 5],
            vec![2, 3, 6, 7],
            vec![8, 9, 12, 13],
            vec![10, 11, 14, 15],
        ]
    );
    Ok(())
}

fn has_nonzero_grad_with_prefix(
    grads: &candle_core::backprop::GradStore,
    vars: &VarMap,
    prefix: &str,
) -> Result<bool> {
    for (name, var) in vars.data().lock().unwrap().iter() {
        if name.starts_with(prefix) {
            if let Some(grad) = grads.get(var.as_tensor()) {
                if grad
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .any(|value| value.abs() > 0.0)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

#[test]
fn predicted_decode_has_an_explicit_trainable_and_detached_path() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new(
        exact_config(4),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    let current = model.encode_state(&frames(&device)?)?;
    let actions = Tensor::from_vec(vec![6u32], (1,), &device)?;
    let coords = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device)?;
    let predicted = model
        .training_latents_from_encoded_state(
            &current,
            &actions,
            &coords,
            RecursionDepth {
                inner_steps: 1,
                outer_steps: 1,
            },
            0.0,
            None,
            RecursionOpts::training(true),
        )?
        .y;
    let trainable_loss = model
        .exact_gameplay_logits_trainable(&predicted)?
        .sqr()?
        .mean_all()?;
    let trainable_grads = trainable_loss.backward()?;
    assert!(has_nonzero_grad_with_prefix(
        &trainable_grads,
        &vars,
        "encoder."
    )?);

    let detached_loss = model
        .exact_gameplay_logits_detached(&predicted)?
        .sqr()?
        .mean_all()?;
    let detached_grads = detached_loss.backward()?;
    assert!(!has_nonzero_grad_with_prefix(
        &detached_grads,
        &vars,
        "encoder."
    )?);
    Ok(())
}

#[test]
fn composed_decode_selects_prediction_or_current_pixel_from_the_gate() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new(
        exact_config(4),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    let latent = Tensor::zeros((1, 8, 16, 16), DType::F32, &device)?;
    let current = Tensor::full(3u32, (1, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
    {
        let data = vars.data().lock().unwrap();
        data["exact_grounding_head.decoder.weight"].set(&Tensor::zeros(
            (4 * 4 * 16, 8),
            DType::F32,
            &device,
        )?)?;
        let mut decoder_bias = vec![0f32; 4 * 4 * 16];
        for subpixel in 0..4 * 4 {
            decoder_bias[subpixel * 16 + 7] = 10.0;
        }
        data["exact_grounding_head.decoder.bias"].set(&Tensor::from_vec(
            decoder_bias,
            (4 * 4 * 16,),
            &device,
        )?)?;
        data["exact_grounding_head.copy_gate.weight"].set(&Tensor::zeros(
            (4 * 4, 8, 1, 1),
            DType::F32,
            &device,
        )?)?;
        data["exact_grounding_head.copy_gate.bias"].set(&Tensor::full(
            100f32,
            (4 * 4,),
            &device,
        )?)?;
    }
    let changed = model.composed_gameplay_decode(&latent, &current)?;
    assert!(changed
        .flatten_all()?
        .to_vec1::<u32>()?
        .iter()
        .all(|pixel| *pixel == 7));
    let next = Tensor::full(7u32, (1, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
    assert_eq!(
        model
            .raw_decoder_transition_correctness(&latent, &current, &next)?
            .to_vec2::<f32>()?[0][0],
        1.0
    );
    assert_eq!(
        model
            .composed_transition_correctness(&latent, &current, &next)?
            .to_vec2::<f32>()?[0][0],
        1.0
    );
    vars.data().lock().unwrap()["exact_grounding_head.copy_gate.bias"].set(&Tensor::full(
        -100f32,
        (4 * 4,),
        &device,
    )?)?;
    let copied = model.composed_gameplay_decode(&latent, &current)?;
    assert!(copied
        .flatten_all()?
        .to_vec1::<u32>()?
        .iter()
        .all(|pixel| *pixel == 3));
    assert_eq!(
        model
            .raw_decoder_transition_correctness(&latent, &current, &next)?
            .to_vec2::<f32>()?[0][0],
        1.0
    );
    assert_eq!(
        model
            .composed_transition_correctness(&latent, &current, &next)?
            .to_vec2::<f32>()?[0][0],
        0.0
    );
    Ok(())
}

#[test]
fn v5_prefix_reuses_the_action_faithful_one_step_recurrence() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new(
        v5_config(4),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 37)?;
    vars.data().lock().unwrap()["spatial_action_proj.weight"].set(&Tensor::ones(
        (8, 4, 1, 1),
        DType::F32,
        &device,
    )?)?;
    let frame = frames(&device)?;
    let state = Tensor::cat(
        &[&model.encode_state(&frame)?, &model.encode_state(&frame)?],
        0,
    )?;
    let actions = Tensor::from_vec(vec![3u32, 6], (2,), &device)?;
    let coords = Tensor::from_vec(vec![0.1f32, 0.2, 0.8, 0.9], (2, 2), &device)?;
    let depth = RecursionDepth {
        inner_steps: 1,
        outer_steps: 1,
    };

    let prefix = model.prefix_predict(&state, &actions, &coords)?;
    let direct = model.predict_latent_with_depth(&state, &actions, &coords, depth)?;
    assert_eq!(
        prefix.flatten_all()?.to_vec1::<f32>()?,
        direct.flatten_all()?.to_vec1::<f32>()?
    );

    let simple_coords = Tensor::from_vec(vec![0.9f32, 0.8, 0.8, 0.9], (2, 2), &device)?;
    let simple_a = Tensor::from_vec(vec![3u32, 3], (2,), &device)?;
    assert_eq!(
        model
            .prefix_predict(&state, &simple_a, &coords)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        model
            .prefix_predict(&state, &simple_a, &simple_coords)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    let action6 = Tensor::from_vec(vec![6u32], (1,), &device)?;
    assert!(
        max_abs_diff(
            &model.prefix_predict(&state.narrow(0, 1, 1)?, &action6, &coords.narrow(0, 1, 1)?,)?,
            &model.prefix_predict(
                &state.narrow(0, 1, 1)?,
                &action6,
                &Tensor::from_vec(vec![0.1f32, 0.2], (1, 2), &device)?,
            )?,
        )? > 0.0
    );

    let rows = Tensor::cat(
        &[
            &model.prefix_predict(
                &state.narrow(0, 0, 1)?,
                &actions.narrow(0, 0, 1)?,
                &coords.narrow(0, 0, 1)?,
            )?,
            &model.prefix_predict(
                &state.narrow(0, 1, 1)?,
                &actions.narrow(0, 1, 1)?,
                &coords.narrow(0, 1, 1)?,
            )?,
        ],
        0,
    )?;
    assert_eq!(
        prefix.flatten_all()?.to_vec1::<f32>()?,
        rows.flatten_all()?.to_vec1::<f32>()?
    );

    vars.data().lock().unwrap()["prefix_head.weight"].set(&Tensor::full(
        1000f32,
        (8, 16),
        &device,
    )?)?;
    assert_eq!(
        prefix.flatten_all()?.to_vec1::<f32>()?,
        model
            .prefix_predict(&state, &actions, &coords)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn v5_ptrm_from_latent_matches_deterministic_and_is_action_faithful() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new(
        v5_config(4),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 91)?;
    let frame = frames(&device)?;
    let state = Tensor::cat(
        &[&model.encode_state(&frame)?, &model.encode_state(&frame)?],
        0,
    )?;
    let actions = Tensor::from_vec(vec![1u32, 6], (2,), &device)?;
    let coords = Tensor::from_vec(vec![0.0f32, 0.0, 0.75, 0.25], (2, 2), &device)?;
    let goals = Tensor::zeros((2, 6), DType::F32, &device)?;
    let depth = RecursionDepth {
        inner_steps: 1,
        outer_steps: 1,
    };
    let ptrm_cfg = PtrmConfig {
        k: 1,
        sigma: 0.0,
        seed: Some(11),
    };
    let deterministic =
        model.forward_from_latent_with_depth(&state, &actions, &coords, &goals, depth)?;
    let ptrm =
        model.forward_ptrm_from_latent(&state, &actions, &coords, &goals, depth, ptrm_cfg)?;
    let trajectory = &ptrm.trajectories[0];
    assert_eq!(
        deterministic.y.flatten_all()?.to_vec1::<f32>()?,
        trajectory.y.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        deterministic.event_logits.flatten_all()?.to_vec1::<f32>()?,
        trajectory.event_logits.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        deterministic.q_logit.flatten_all()?.to_vec1::<f32>()?,
        trajectory.q_logit.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        deterministic
            .reliability_logit
            .flatten_all()?
            .to_vec1::<f32>()?,
        trajectory
            .reliability_logit
            .flatten_all()?
            .to_vec1::<f32>()?
    );

    for row in 0..2 {
        let per_row = model.forward_ptrm_from_latent(
            &state.narrow(0, row, 1)?,
            &actions.narrow(0, row, 1)?,
            &coords.narrow(0, row, 1)?,
            &goals.narrow(0, row, 1)?,
            depth,
            ptrm_cfg,
        )?;
        assert_eq!(
            trajectory
                .y
                .narrow(0, row, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            per_row.trajectories[0].y.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            trajectory
                .event_logits
                .narrow(0, row, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            per_row.trajectories[0]
                .event_logits
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert_eq!(
            trajectory
                .q_logit
                .narrow(0, row, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            per_row.trajectories[0]
                .q_logit
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert_eq!(
            trajectory
                .reliability_logit
                .narrow(0, row, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            per_row.trajectories[0]
                .reliability_logit
                .flatten_all()?
                .to_vec1::<f32>()?
        );
    }

    let err = model
        .forward_ptrm_prepared(
            &Tensor::zeros((2, 8, 16, 16), DType::F32, &device)?,
            &Tensor::zeros((2, 8), DType::F32, &device)?,
            None,
            depth,
            ptrm_cfg,
        )
        .unwrap_err();
    assert!(err.to_string().contains("action-aware preparation"));

    vars.data().lock().unwrap()["action_film_beta.bias"].set(&Tensor::full(
        0.25f32,
        (8,),
        &device,
    )?)?;
    let changed = model.forward_from_latent_with_depth(&state, &actions, &coords, &goals, depth)?;
    let changed_ptrm =
        model.forward_ptrm_from_latent(&state, &actions, &coords, &goals, depth, ptrm_cfg)?;
    assert!(max_abs_diff(&deterministic.y, &changed.y)? > 0.0);
    assert_eq!(
        changed.y.flatten_all()?.to_vec1::<f32>()?,
        changed_ptrm.trajectories[0]
            .y
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn positional_value_readout_preserves_native_patch4_positions() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new_with_positional_value_readout(
        v5_config(4),
        true,
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 7)?;
    assert!(vars
        .data()
        .lock()
        .unwrap()
        .contains_key("consumer_readout.position_value_embedding.weight"));

    let translated = [
        (
            spatial_with_values(&device, &[(0, 0, 1.0)])?,
            spatial_with_values(&device, &[(3, 5, 1.0)])?,
        ),
        (
            spatial_with_values(&device, &[(8, 1, 1.0)])?,
            spatial_with_values(&device, &[(12, 14, 1.0)])?,
        ),
    ];
    for (left, right) in translated {
        assert!(
            max_abs_diff(
                &model.canonical_representation(&left)?,
                &model.canonical_representation(&right)?,
            )? > 0.0
        );
    }
    let first = spatial_with_values(&device, &[(0, 0, 1.0), (0, 1, -1.0)])?;
    let swapped = spatial_with_values(&device, &[(0, 0, -1.0), (0, 1, 1.0)])?;
    assert!(
        max_abs_diff(
            &model.canonical_representation(&first)?,
            &model.canonical_representation(&swapped)?,
        )? > 0.0
    );

    let legacy_vars = VarMap::new();
    let legacy = WorldModel::new(
        v5_config(4),
        VarBuilder::from_varmap(&legacy_vars, DType::F32, &device),
    )?;
    let explicit_legacy_vars = VarMap::new();
    let explicit_legacy = WorldModel::new_with_positional_value_readout(
        v5_config(4),
        false,
        VarBuilder::from_varmap(&explicit_legacy_vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&legacy_vars, 7)?;
    reinit_varmap_deterministic(&explicit_legacy_vars, 7)?;
    assert!(!legacy_vars
        .data()
        .lock()
        .unwrap()
        .contains_key("consumer_readout.position_value_embedding.weight"));
    assert_eq!(
        legacy
            .canonical_representation(&first)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        explicit_legacy
            .canonical_representation(&first)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn phase_a_capability_probe_reports_specific_gaps() -> Result<()> {
    let device = Device::Cpu;
    let full = WorldModel::new(
        v5_config(4),
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
    )?;
    let capabilities = full.phase_a_inference_capabilities();
    for check in [
        capabilities.patch4_grid,
        capabilities.spatial_prefix_faithful,
        capabilities.action_faithful_ptrm,
        capabilities.composed_decode_available,
        capabilities.null_action_row_present,
    ] {
        assert!(check.passed, "{:?}", check.reason);
    }

    let patch8 = WorldModel::new(
        v5_config(8),
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
    )?;
    let patch8 = patch8.phase_a_inference_capabilities().patch4_grid;
    assert!(!patch8.passed);
    assert_eq!(
        patch8.reason.as_deref(),
        Some("requires the canonical patch-4 latent grid")
    );

    let neutral_ptrm = WorldModel::new(
        exact_config(4),
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
    )?;
    let neutral_ptrm = neutral_ptrm
        .phase_a_inference_capabilities()
        .action_faithful_ptrm;
    assert!(!neutral_ptrm.passed);
    assert_eq!(
        neutral_ptrm.reason.as_deref(),
        Some("requires world-core-v5 action-aware PTRM preparation")
    );

    let pooled_prefix = WorldModel::new(
        ModelConfig {
            hidden_dim: 8,
            action_dim: 8,
            goal_dim: 6,
            inner_steps: 1,
            outer_steps: 1,
            ..ModelConfig::default()
        },
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
    )?;
    let pooled_prefix = pooled_prefix
        .phase_a_inference_capabilities()
        .spatial_prefix_faithful;
    assert!(!pooled_prefix.passed);
    assert_eq!(
        pooled_prefix.reason.as_deref(),
        Some("requires the world-core-v4 recurrence prefix path")
    );
    Ok(())
}
