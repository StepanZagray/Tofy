use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::experiment::ConsumerReadoutTopology;
use tofy::p2::grounding::patch_tokens_to_pixels;
use tofy::p2::model::{
    ModelConfig, PtrmConfig, RecursionDepth, RecursionOpts, WorldModel, FRAME_SIDE,
};
use tofy::p2::model::{restore_copy_gate_bias_prior, zero_copy_bypass_gate};
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

fn treatment_config() -> ModelConfig {
    ModelConfig {
        residual_y_update: true,
        warm_start_y: true,
        ..v5_config(4)
    }
}

fn action6(device: &Device, x: f32, y: f32) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        Tensor::from_vec(vec![6u32], (1,), device)?,
        Tensor::from_vec(vec![x, y], (1, 2), device)?,
        Tensor::zeros((1, 6), DType::F32, device)?,
    ))
}

fn set_alpha(vars: &VarMap, value: f32) -> Result<()> {
    let data = vars.data().lock().unwrap();
    let var = data
        .get("y_copy_bypass_alpha")
        .expect("copy-bypass gate parameter exists");
    var.set(&Tensor::full(value, var.shape().dims(), var.device())?)?;
    Ok(())
}

#[test]
fn copy_bypass_alpha_accessor_tracks_flag_and_value() -> Result<()> {
    let device = Device::Cpu;
    let disabled_vars = VarMap::new();
    let disabled = WorldModel::new(
        treatment_config(),
        VarBuilder::from_varmap(&disabled_vars, DType::F32, &device),
    )?;
    assert_eq!(disabled.copy_bypass_alpha()?, None);

    let enabled_vars = VarMap::new();
    let enabled = WorldModel::new(
        ModelConfig {
            copy_bypass_gate: true,
            ..treatment_config()
        },
        VarBuilder::from_varmap(&enabled_vars, DType::F32, &device),
    )?;
    set_alpha(&enabled_vars, 0.375)?;
    assert_eq!(enabled.copy_bypass_alpha()?, Some(0.375));
    Ok(())
}

#[test]
fn copy_bypass_zero_gate_is_exact_latent_copy_for_any_finite_state() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let cfg = ModelConfig {
        copy_bypass_gate: true,
        ..treatment_config()
    };
    let model = WorldModel::new(cfg, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    reinit_varmap_deterministic(&vars, 7)?;
    zero_copy_bypass_gate(&vars)?;
    // Deliberately non-unit-RMS: the zero-gate fixpoint must hold for any
    // finite latent, not only encoder-manifold states.
    let state = spatial_with_values(&device, &[(3, 5, 3.7), (10, 2, -8.0)])?;
    let (actions, coords, goals) = action6(&device, 0.5, 0.5)?;
    let out = model.forward_from_latent(&state, &actions, &coords, &goals)?;
    assert!(max_abs_diff(&out.y, &state)? < 1e-7);
    Ok(())
}

#[test]
fn copy_bypass_alpha_one_reproduces_the_legacy_update() -> Result<()> {
    let device = Device::Cpu;
    let legacy_vars = VarMap::new();
    let legacy = WorldModel::new(
        treatment_config(),
        VarBuilder::from_varmap(&legacy_vars, DType::F32, &device),
    )?;
    let gated_vars = VarMap::new();
    let gated = WorldModel::new(
        ModelConfig {
            copy_bypass_gate: true,
            ..treatment_config()
        },
        VarBuilder::from_varmap(&gated_vars, DType::F32, &device),
    )?;
    // Identical name-seeded weights in both models; only the gate differs.
    reinit_varmap_deterministic(&legacy_vars, 11)?;
    reinit_varmap_deterministic(&gated_vars, 11)?;
    set_alpha(&gated_vars, 1.0)?;
    let state = spatial_with_values(&device, &[(1, 1, 0.9), (7, 12, -0.4)])?;
    let (actions, coords, goals) = action6(&device, 0.25, 0.75)?;
    let legacy_out = legacy.forward_from_latent(&state, &actions, &coords, &goals)?;
    let gated_out = gated.forward_from_latent(&state, &actions, &coords, &goals)?;
    assert!(max_abs_diff(&legacy_out.y, &gated_out.y)? < 1e-6);
    Ok(())
}

#[test]
fn copy_bypass_gate_receives_gradient_at_zero() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = WorldModel::new(
        ModelConfig {
            copy_bypass_gate: true,
            ..treatment_config()
        },
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 13)?;
    zero_copy_bypass_gate(&vars)?;
    let state = spatial_with_values(&device, &[(2, 2, 1.0)])?;
    let target = spatial_with_values(&device, &[(2, 3, 1.0)])?;
    let (actions, coords, goals) = action6(&device, 0.1, 0.1)?;
    let out = model.forward_from_latent(&state, &actions, &coords, &goals)?;
    let loss = out.y.sub(&target)?.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    let alpha_grad = {
        let data = vars.data().lock().unwrap();
        let alpha = data.get("y_copy_bypass_alpha").expect("gate exists");
        grads
            .get(alpha.as_tensor())
            .expect("gate must receive a gradient")
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?
    };
    assert!(alpha_grad.is_finite() && alpha_grad > 0.0);
    Ok(())
}

#[test]
fn reinit_then_zero_helper_restores_the_zero_gate() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let _model = WorldModel::new(
        ModelConfig {
            copy_bypass_gate: true,
            ..treatment_config()
        },
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&vars, 17)?;
    zero_copy_bypass_gate(&vars)?;
    let data = vars.data().lock().unwrap();
    let alpha = data.get("y_copy_bypass_alpha").expect("gate exists");
    assert_eq!(alpha.as_tensor().abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
    Ok(())
}

#[test]
fn copy_gate_bias_prior_starts_as_calibrated_copy() -> Result<()> {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        copy_gate_bias_prior: Some(0.02),
        ..v5_config(4)
    };
    let vars = VarMap::new();
    let model = WorldModel::new(cfg, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    // Pipeline-faithful: the generic reinitializer zeroes every bias, so the
    // prior must survive via the same restore step training uses.
    reinit_varmap_deterministic(&vars, 29)?;
    restore_copy_gate_bias_prior(&vars, Some(0.02))?;
    // A zero latent reaches exactly the bias through the 1x1 gate conv.
    let zero_latent = Tensor::zeros((1, 8, 16, 16), DType::F32, &device)?;
    let gate = model.exact_copy_gate(&zero_latent)?;
    let max_gate = gate.max_all()?.to_scalar::<f32>()?;
    let min_gate = gate.min_all()?.to_scalar::<f32>()?;
    assert!((max_gate - 0.02).abs() < 1e-4 && (min_gate - 0.02).abs() < 1e-4);
    // Composition at init is a pure copy: zero false edits anywhere.
    let current = frames(&device)?;
    let composed = model.composed_gameplay_decode(&zero_latent, &current)?;
    let gameplay = current
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    assert!(max_abs_diff(&composed, &gameplay)? == 0.0);
    Ok(())
}

#[test]
fn grid_scaled_impulse_sharpens_adjacent_action6_coordinates() -> Result<()> {
    let device = Device::Cpu;
    let build = |grid_scaled: bool, seed: u64| -> Result<(WorldModel, VarMap)> {
        let vars = VarMap::new();
        let model = WorldModel::new(
            ModelConfig {
                grid_scaled_action_impulse: grid_scaled,
                ..treatment_config()
            },
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&vars, seed)?;
        Ok((model, vars))
    };
    let (legacy, _lv) = build(false, 19)?;
    let (scaled, _sv) = build(true, 19)?;
    let state = spatial_with_values(&device, &[(8, 8, 1.0)])?;
    let goals = Tensor::zeros((1, 6), DType::F32, &device)?;
    let one_cell = 1.0 / 15.0;
    let sensitivity = |model: &WorldModel| -> Result<f32> {
        let (actions, coords_a, _) = action6(&device, 0.5, 0.5)?;
        let coords_b = Tensor::from_vec(vec![0.5 + one_cell, 0.5], (1, 2), &device)?;
        let out_a = model.forward_from_latent(&state, &actions, &coords_a, &goals)?;
        let out_b = model.forward_from_latent(&state, &actions, &coords_b, &goals)?;
        max_abs_diff(&out_a.y, &out_b.y)
    };
    let legacy_sensitivity = sensitivity(&legacy)?;
    let scaled_sensitivity = sensitivity(&scaled)?;
    assert!(
        scaled_sensitivity > legacy_sensitivity,
        "grid-scaled impulse must increase one-cell coordinate sensitivity: \
         legacy {legacy_sensitivity}, scaled {scaled_sensitivity}"
    );
    Ok(())
}

#[test]
fn joint_copy_mixture_matches_two_candidate_rule_and_never_edits_below_half() -> Result<()> {
    let device = Device::Cpu;
    let model = WorldModel::new(
        ModelConfig {
            decode_composition: tofy::p2::grounding::DecodeComposition::JointCopyMixture,
            ..v5_config(4)
        },
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
    )?;
    let latent = spatial_with_values(&device, &[(0, 0, 2.0), (9, 9, -1.5), (15, 15, 0.7)])?;
    let current = frames(&device)?;
    let composed = model.composed_gameplay_decode(&latent, &current)?;
    let logits = model.exact_gameplay_logits_detached(&latent)?;
    let gate = model.exact_copy_gate(&latent)?;
    let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
    let gameplay = current
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let composed_v = composed.flatten_all()?.to_vec1::<u32>()?;
    let current_v = gameplay.flatten_all()?.to_vec1::<u32>()?;
    let gate_v = gate.flatten_all()?.to_vec1::<f32>()?;
    let probs_v = probs.flatten_all()?.to_vec1::<f32>()?;
    for (pixel, (&out, (&cur, &g))) in composed_v
        .iter()
        .zip(current_v.iter().zip(gate_v.iter()))
        .enumerate()
    {
        let row = &probs_v[pixel * 16..(pixel + 1) * 16];
        let (argmax, p_max) = row
            .iter()
            .enumerate()
            .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
        let p_cur = row[cur as usize];
        let expected = if g * p_max > (1.0 - g) + g * p_cur {
            argmax as u32
        } else {
            cur
        };
        assert_eq!(out, expected, "pixel {pixel}: mixture MAP mismatch");
        if g < 0.5 {
            assert_eq!(out, cur, "pixel {pixel}: sub-0.5 gate must copy");
        }
    }
    Ok(())
}

#[test]
fn positional_value_readout_is_reachable_from_config() -> Result<()> {
    let device = Device::Cpu;
    let build = |flag: bool| -> Result<Vec<String>> {
        let vars = VarMap::new();
        let _model = WorldModel::new(
            ModelConfig {
                positional_value_readout: flag,
                ..v5_config(4)
            },
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        let data = vars.data().lock().unwrap();
        Ok(data.keys().cloned().collect())
    };
    let with_flag = build(true)?;
    let without_flag = build(false)?;
    assert!(with_flag
        .iter()
        .any(|name| name.contains("position_value_embedding")));
    assert!(!without_flag
        .iter()
        .any(|name| name.contains("position_value_embedding")));
    Ok(())
}

#[test]
fn treatment_config_validation_fails_closed() -> Result<()> {
    let no_warm_start = ModelConfig {
        copy_bypass_gate: true,
        warm_start_y: false,
        ..treatment_config()
    };
    assert!(no_warm_start.validate().is_err());
    let bad_prior = ModelConfig {
        copy_gate_bias_prior: Some(1.5),
        ..v5_config(4)
    };
    assert!(bad_prior.validate().is_err());
    Ok(())
}
