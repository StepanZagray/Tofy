use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tofy::p2::experiment::ConsumerReadoutTopology;
use tofy::p2::grounding::patch_tokens_to_pixels;
use tofy::p2::model::{ModelConfig, RecursionDepth, RecursionOpts, WorldModel, FRAME_SIDE};

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
    Ok(())
}
