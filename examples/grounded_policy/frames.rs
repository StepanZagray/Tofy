//! Frozen inspection of the existing spatial policy on each observed frame.
//! Public cells and metadata are exported for independent input reconstruction;
//! only the original Inputs and learned features enter the model computation.

use super::{data::Cohort, engine, Mode};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;
use tofy::p2::looped_agent::{
    grounded_policy::{SpatialPolicyOutput, WIDTH},
    profile::LoopedCapture,
    task::Inputs,
    ACTIONS, META_DIM, OBSERVED_FRAMES, PALETTE, PATCH_COUNT, PATCH_PIXELS, TOKENS,
};

pub(super) fn validate_mode(
    mode: Mode,
    cohort: Option<Cohort>,
    cleared: bool,
    physical_batch: usize,
    updates: usize,
) -> Result<()> {
    if mode.frame_evaluation() {
        ensure!(
            cohort == Some(Cohort::Seen) && !cleared,
            "frame evaluation requires factual seen rows"
        );
        ensure!(
            physical_batch == 34 && updates == 0,
            "frame evaluation requires physical batch 34 and zero updates"
        );
    }
    Ok(())
}

/// Input order: before0, after0, before1, after1, before2, after2, current.
fn public_cells(input: &Inputs) -> Result<Vec<Vec<u32>>> {
    ensure!(
        input.patches.len() == TOKENS * PATCH_PIXELS && input.metadata.len() == TOKENS * META_DIM,
        "frame input lengths differ"
    );
    ensure!(
        input.metadata.iter().all(|v| v.is_finite()),
        "nonfinite public metadata"
    );
    input
        .patches
        .chunks_exact(PATCH_COUNT * PATCH_PIXELS)
        .map(|frame| {
            frame
                .chunks_exact(PATCH_PIXELS)
                .map(|patch| {
                    let color = patch[0];
                    ensure!(
                        color < PALETTE as u32 && patch.iter().all(|&v| v == color),
                        "public frame patch must be uniform and in the palette"
                    );
                    Ok(color)
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Serialize)]
pub(super) struct FrameExport {
    cells: Vec<u32>,
    attention: Vec<Vec<f32>>,
    pooled: Vec<f32>,
}

fn frozen_tensor(tensor: &Tensor, shape: &[usize]) -> Result<()> {
    ensure!(
        tensor.dims() == shape && tensor.dtype() == DType::F32 && !tensor.track_op(),
        "frame tensor has unexpected shape, dtype, or autograd graph"
    );
    Ok(())
}

fn export_policy(
    policy: &SpatialPolicyOutput,
    cells: &[Vec<Vec<u32>>],
    index: usize,
    exports: &mut [Vec<FrameExport>],
) -> Result<()> {
    let batch = cells.len();
    frozen_tensor(&policy.logits, &[batch, ACTIONS])?;
    frozen_tensor(&policy.attention, &[batch, 2, PATCH_COUNT])?;
    frozen_tensor(&policy.pooled, &[batch, 2 * WIDTH])?;
    let logits = policy.logits.to_vec2::<f32>()?;
    let attention = policy.attention.to_vec3::<f32>()?;
    let pooled = policy.pooled.to_vec2::<f32>()?;
    ensure!(
        logits
            .iter()
            .flatten()
            .chain(attention.iter().flatten().flatten())
            .chain(pooled.iter().flatten())
            .all(|v| v.is_finite()),
        "nonfinite frame readout"
    );
    ensure!(
        attention.iter().flatten().all(|weights| {
            weights.iter().all(|&v| (0.0..=1.0).contains(&v))
                && (weights.iter().map(|&v| f64::from(v)).sum::<f64>() - 1.0).abs() <= 1e-5
        }),
        "invalid frame attention probabilities"
    );
    for (row, (attention, pooled)) in attention.into_iter().zip(pooled).enumerate() {
        exports[row].push(FrameExport {
            cells: cells[row][index].clone(),
            attention,
            pooled,
        });
    }
    Ok(())
}

pub(super) fn measured_forward(
    model: &engine::FrozenModel,
    rows: &[Inputs],
    device: &Device,
    capture: Option<&LoopedCapture>,
) -> Result<(engine::Forward, Vec<Vec<FrameExport>>)> {
    ensure!(!rows.is_empty(), "frame batch must be nonempty");
    let cells = rows.iter().map(public_cells).collect::<Result<Vec<_>>>()?;
    let mut exports = (0..rows.len())
        .map(|_| Vec::with_capacity(OBSERVED_FRAMES))
        .collect::<Vec<_>>();
    device.synchronize()?;
    let measured = capture.map(LoopedCapture::measurement);
    let phase = capture.map(|c| c.phase("forward", Some(candle_graph::ExecutionStep::Forward)));
    let result = (|| {
        // The ordinary current-policy forward executes exactly once. Support
        // frames reuse the same frozen head parameters, with no label routing.
        let out = model.forward(rows)?;
        for index in 0..OBSERVED_FRAMES {
            let features = out.features.frame(index)?;
            frozen_tensor(&features, &[rows.len(), PATCH_COUNT, WIDTH])?;
            let support = (index < OBSERVED_FRAMES - 1)
                .then(|| model.policy.forward(&features))
                .transpose()?;
            let policy = support.as_ref().unwrap_or(&out.policy);
            export_policy(policy, &cells, index, &mut exports)?;
            if let (Some(c), Some(p)) = (capture, &phase) {
                c.record_tensor_stats(p, &format!("frames/{index}/features"), &features)?;
                c.record_tensor_stats(p, &format!("frames/{index}/attention"), &policy.attention)?;
                c.record_tensor_stats(p, &format!("frames/{index}/pooled"), &policy.pooled)?;
            }
        }
        if let (Some(c), Some(p)) = (capture, &phase) {
            c.record_tensor_stats(p, "policy/logits", &out.policy.logits)?;
            c.record_scalar(p, "batch/rows", rows.len() as f64)?;
            c.record_scalar(p, "frames/count", OBSERVED_FRAMES as f64)?;
            c.record_scalar(p, "optimizer/updates", 0.0)?;
        }
        Ok((out, exports))
    })();
    // Close both phase and measurement only after synchronization, including
    // failures during any support-head invocation or export validation.
    let synchronized = device.synchronize();
    drop(phase);
    drop(measured);
    synchronized?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    fn input() -> Inputs {
        Inputs {
            patches: (0..TOKENS)
                .flat_map(|token| {
                    let color = (3 * (token / PATCH_COUNT) + token % PATCH_COUNT) % PALETTE;
                    std::iter::repeat_n(color as u32, PATCH_PIXELS)
                })
                .collect(),
            metadata: (0..TOKENS * META_DIM)
                .map(|i| (i % 11) as f32 / 11.0)
                .collect(),
        }
    }

    #[test]
    fn public_frame_order_reconstructs_exact_input_and_hash() -> Result<()> {
        let input = input();
        let cells = public_cells(&input)?;
        assert_eq!(cells.len(), 7);
        for (frame, values) in cells.iter().enumerate() {
            assert_eq!(values.len(), 64);
            for (patch, &color) in values.iter().enumerate() {
                assert_eq!(color, ((frame * 3 + patch) % PALETTE) as u32);
            }
        }
        let rebuilt = Inputs {
            patches: cells
                .iter()
                .flatten()
                .flat_map(|&color| std::iter::repeat_n(color, PATCH_PIXELS))
                .collect(),
            metadata: input.metadata.clone(),
        };
        assert_eq!(rebuilt.patches, input.patches);
        assert_eq!(
            super::super::data::input_hash(&rebuilt),
            super::super::data::input_hash(&input)
        );
        Ok(())
    }

    #[test]
    fn malformed_public_inputs_fail_closed() {
        for corruption in 0..6 {
            let mut input = input();
            match corruption {
                0 => {
                    input.patches.pop();
                }
                1 => input.patches.push(0),
                2 => {
                    input.metadata.pop();
                }
                3 => input.metadata[0] = f32::NAN,
                4 => input.patches[1] = 1,
                5 => input.patches[..PATCH_PIXELS].fill(PALETTE as u32),
                _ => unreachable!(),
            }
            assert!(public_cells(&input).is_err(), "corruption {corruption}");
        }
    }

    #[test]
    fn frame_modes_are_scoped_and_preserve_legacy_mode_checks() -> Result<()> {
        for mode in [Mode::EvalFramesInitial, Mode::EvalFramesFinal] {
            validate_mode(mode, Some(Cohort::Seen), false, 34, 0)?;
            assert!(validate_mode(mode, None, false, 34, 0).is_err());
            for cohort in [Cohort::Training, Cohort::Familiar, Cohort::Heldout] {
                assert!(validate_mode(mode, Some(cohort), false, 34, 0).is_err());
            }
            assert!(validate_mode(mode, Some(Cohort::Seen), true, 34, 0).is_err());
            assert!(validate_mode(mode, Some(Cohort::Seen), false, 33, 0).is_err());
            assert!(validate_mode(mode, Some(Cohort::Seen), false, 34, 1).is_err());
        }
        for mode in [
            Mode::Audit,
            Mode::Qualify,
            Mode::BatchSmoke,
            Mode::Train,
            Mode::EvalInitial,
            Mode::EvalFinal,
        ] {
            validate_mode(mode, None, true, 1, 1150)?;
            assert!(!mode.frame_evaluation());
        }
        assert!(Mode::EvalFramesFinal.final_checkpoint());
        assert!(!Mode::EvalFramesInitial.final_checkpoint());
        assert_eq!(
            serde_json::to_string(&Mode::EvalFramesInitial)?,
            "\"eval_frames_initial\""
        );
        assert_eq!(
            serde_json::to_string(&Mode::EvalFramesFinal)?,
            "\"eval_frames_final\""
        );
        Ok(())
    }

    #[test]
    fn frozen_frame_exports_reuse_current_without_changing_legacy_values() -> Result<()> {
        let device = Device::Cpu;
        // Constructor parameters and synthetic inputs only: no checkpoint,
        // canonical import, registered population, or optimizer invocation.
        let model = engine::Model::new(&device)?;
        let before = model.snapshot()?;
        let frozen = model.frozen()?;
        let rows = vec![input(), input()];
        let legacy = super::super::measured_forward(&frozen, &rows, &device, None)?;
        let (out, frames) = measured_forward(&frozen, &rows, &device, None)?;
        assert_eq!(
            out.policy.logits.to_vec2::<f32>()?,
            legacy.policy.logits.to_vec2::<f32>()?
        );
        assert_eq!(
            out.policy.attention.to_vec3::<f32>()?,
            legacy.policy.attention.to_vec3::<f32>()?
        );
        assert_eq!(
            out.policy.pooled.to_vec2::<f32>()?,
            legacy.policy.pooled.to_vec2::<f32>()?
        );
        let current_attention = out.policy.attention.to_vec3::<f32>()?;
        let current_pooled = out.policy.pooled.to_vec2::<f32>()?;
        for index in 0..7 {
            let expected = frozen.policy.forward(&out.features.frame(index)?)?;
            let attention = expected.attention.to_vec3::<f32>()?;
            let pooled = expected.pooled.to_vec2::<f32>()?;
            for row in 0..rows.len() {
                assert_eq!(frames[row][index].attention, attention[row]);
                assert_eq!(frames[row][index].pooled, pooled[row]);
            }
        }
        assert_eq!(frames.len(), 2);
        for (row, frames) in frames.iter().enumerate() {
            assert_eq!(frames.len(), 7);
            assert_eq!(frames[6].attention, current_attention[row]);
            assert_eq!(frames[6].pooled, current_pooled[row]);
            assert!(frames[..6]
                .iter()
                .any(|frame| frame.pooled != current_pooled[row]));
        }
        assert!(model.change_audit(&before)?.all_parameters_unchanged);
        Ok(())
    }

    #[test]
    fn tracked_nonfinite_or_wrong_shape_exports_fail() -> Result<()> {
        let device = Device::Cpu;
        let tracked = Var::zeros((1, 4), DType::F32, &device)?;
        assert!(frozen_tensor(&tracked, &[1, 4]).is_err());
        assert!(frozen_tensor(&Tensor::zeros((1, 5), DType::F32, &device)?, &[1, 4]).is_err());
        let cells = vec![public_cells(&input())?];
        for bad_tensor in 0..5 {
            let mut policy = SpatialPolicyOutput {
                logits: Tensor::zeros((1, 4), DType::F32, &device)?,
                attention: Tensor::full(1.0f32 / 64.0, (1, 2, 64), &device)?,
                pooled: Tensor::zeros((1, 256), DType::F32, &device)?,
            };
            if bad_tensor < 3 {
                let bad = match bad_tensor {
                    0 => &mut policy.logits,
                    1 => &mut policy.attention,
                    _ => &mut policy.pooled,
                };
                *bad = Tensor::full(f32::NAN, bad.shape(), &device)?;
            } else {
                policy.attention = Tensor::full(
                    if bad_tensor == 3 { -1.0f32 } else { 0.0f32 },
                    (1, 2, 64),
                    &device,
                )?;
            }
            assert!(export_policy(&policy, &cells, 0, &mut [vec![]]).is_err());
        }
        Ok(())
    }
}
