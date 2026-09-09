//! Preparation-only claim: normalized learned role attention [B, 7, 2, 64]
//! and validated public metadata produce the coordinate-expectation formula
//! for three observed support displacements and one desired query displacement.
//! The tensor path preserves differentiation to attention and never receives
//! pixels, labels, hidden roles, a rule mapping, or an analytic action solver.
//!
//! Frames are before0, after0, before1, after1, before2, after2, current.
//! Fixed learned slots are agent=0 and goal=1; their semantic accuracy is a
//! caller-owned empirical premise, not something this adapter discovers.
//! Coordinates are public patch x/y in cell units. No rounding, argmax, or
//! per-example slot rearrangement occurs. Scalar validation readbacks do not
//! form a host-vector inference bridge. This is not a learning/generalization
//! claim and does not validate a privileged-fit selector on a new population.

use super::{
    ACTIONS, FRAME_SIDE, META_DIM, OBSERVED_FRAMES, PATCH_COUNT, PATCH_SIDE, SUPPORT_STEPS, TOKENS,
};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Control {
    #[default]
    Factual,
    UniformAttention,
    EffectsZero,
    QueryZero,
}

fn all(mask: Tensor) -> Result<bool> {
    Ok(mask.to_dtype(DType::F32)?.min_all()?.to_scalar::<f32>()? == 1.0)
}

// Constant public encoding, with the action columns left zero. No input or
// learned values are copied to the host to construct this validation template.
fn public_template(device: &Device) -> Result<Tensor> {
    let side = FRAME_SIDE / PATCH_SIDE;
    let mut values = vec![0.0f32; TOKENS * META_DIM];
    for frame in 0..OBSERVED_FRAMES {
        for patch in 0..PATCH_COUNT {
            let row = &mut values[(frame * PATCH_COUNT + patch) * META_DIM..][..META_DIM];
            row[if frame < 2 * SUPPORT_STEPS {
                frame % 2
            } else {
                2
            }] = 1.0;
            row[7] = (patch % side) as f32 / (side - 1) as f32;
            row[8] = (patch / side) as f32 / (side - 1) as f32;
            row[9] = if frame < 2 * SUPPORT_STEPS {
                (frame / 2) as f32 / SUPPORT_STEPS as f32
            } else {
                1.0
            };
        }
    }
    Ok(Tensor::from_vec(
        values,
        (1, OBSERVED_FRAMES, PATCH_COUNT, META_DIM),
        device,
    )?)
}

fn observed_actions(metadata: &Tensor, batch: usize) -> Result<Vec<Tensor>> {
    ensure!(
        all(metadata.abs()?.le(f32::MAX)?)?,
        "nonfinite public metadata"
    );
    let actions = (0..SUPPORT_STEPS)
        .map(|step| {
            metadata
                .narrow(1, 2 * step, 1)?
                .narrow(2, 0, 1)?
                .narrow(3, 3, ACTIONS)?
                .squeeze(2)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let selected = Tensor::cat(&actions, 1)?;
    ensure!(
        all(selected.eq(0.0f32)?.add(&selected.eq(1.0f32)?)?.eq(1u8)?)?
            && all(selected.sum_keepdim(2)?.eq(1.0f32)?)?
            && all(selected.sum_keepdim(1)?.le(1.0f32)?)?,
        "support actions must be three distinct one-hot public IDs"
    );
    let mut frames = Vec::with_capacity(OBSERVED_FRAMES);
    for action in &actions {
        frames.extend([action.clone(), action.clone()]);
    }
    frames.push(Tensor::zeros(
        (batch, 1, ACTIONS),
        DType::F32,
        metadata.device(),
    )?);
    let zeros = Tensor::zeros((batch, OBSERVED_FRAMES, 3), DType::F32, metadata.device())?;
    let fields = Tensor::cat(&[&zeros, &Tensor::cat(&frames, 1)?, &zeros], 2)?
        .unsqueeze(2)?
        .broadcast_as((batch, OBSERVED_FRAMES, PATCH_COUNT, META_DIM))?;
    let expected = fields.broadcast_add(&public_template(metadata.device())?)?;
    ensure!(
        all(metadata.eq(&expected)?)?,
        "public frame order, action IDs, coordinates or chronology differ"
    );
    Ok(actions)
}

/// Convert F32 attention [B, 7, 2, 64] and public metadata [B, 448, 10]
/// into F32 [B, 4, 7] records on the same device. Each attention row must be
/// finite, nonnegative and sum to one within 1e-5; it is never renormalized.
/// Metadata must match the exact public grid/frame encoding, including three
/// distinct support actions repeated on every patch of both observed frames.
///
/// The first three records are [after_agent - before_agent, action_one_hot, 0];
/// the last is [current_goal - current_agent, 0, 0, 0, 0, 1]. Soft expectations
/// need not be cardinal/integer. Controls replace only their named quantities;
/// all inputs are validated even for the uniform-attention control.
pub fn records(attention: &Tensor, metadata: &Tensor, control: Control) -> Result<Tensor> {
    let (batch, frames, roles, patches) = attention.dims4()?;
    ensure!(
        batch > 0
            && (frames, roles, patches) == (OBSERVED_FRAMES, 2, PATCH_COUNT)
            && metadata.dims() == [batch, TOKENS, META_DIM]
            && attention.dtype() == DType::F32
            && metadata.dtype() == DType::F32
            && attention.device().same_device(metadata.device()),
        "adapter requires same-device F32 [B, 7, 2, 64] attention and [B, 448, 10] metadata"
    );
    ensure!(
        all(attention.ge(0.0f32)?)?
            && all(attention.le(1.0f32)?)?
            && all(attention
                .sum_keepdim(3)?
                .affine(1.0, -1.0)?
                .abs()?
                .le(1e-5f32)?)?,
        "invalid normalized role attention"
    );
    let metadata = metadata.reshape((batch, OBSERVED_FRAMES, PATCH_COUNT, META_DIM))?;
    let actions = observed_actions(&metadata, batch)?;
    let weights = if control == Control::UniformAttention {
        Tensor::full(
            1.0f32 / PATCH_COUNT as f32,
            attention.shape(),
            attention.device(),
        )?
    } else {
        attention.clone()
    };
    let coordinates = metadata
        .narrow(3, 7, 2)?
        .affine((FRAME_SIDE / PATCH_SIDE - 1) as f64, 0.0)?
        .contiguous()?
        .reshape((batch * OBSERVED_FRAMES, PATCH_COUNT, 2))?;
    let positions = weights
        .contiguous()?
        .reshape((batch * OBSERVED_FRAMES, 2, PATCH_COUNT))?
        .matmul(&coordinates)?
        .reshape((batch, OBSERVED_FRAMES, 2, 2))?;
    let agent = |frame| -> candle_core::Result<Tensor> {
        positions.narrow(1, frame, 1)?.narrow(2, 0, 1)?.squeeze(2)
    };
    let zeros = Tensor::zeros((batch, 1, 1), DType::F32, attention.device())?;
    let mut output = Vec::with_capacity(SUPPORT_STEPS + 1);
    for (step, action) in actions.iter().enumerate() {
        let effect = agent(2 * step + 1)?.sub(&agent(2 * step)?)?;
        let effect = if control == Control::EffectsZero {
            effect.zeros_like()?
        } else {
            effect
        };
        output.push(Tensor::cat(&[&effect, action, &zeros], 2)?);
    }
    let current = positions.narrow(1, OBSERVED_FRAMES - 1, 1)?;
    let desired = current
        .narrow(2, 1, 1)?
        .squeeze(2)?
        .sub(&agent(OBSERVED_FRAMES - 1)?)?;
    let desired = if control == Control::QueryZero {
        desired.zeros_like()?
    } else {
        desired
    };
    output.push(Tensor::cat(
        &[
            desired,
            Tensor::zeros((batch, 1, ACTIONS), DType::F32, attention.device())?,
            zeros.ones_like()?,
        ],
        2,
    )?);
    Ok(Tensor::cat(&output, 1)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::looped_agent::task::{self, Transition};
    use candle_core::Var;

    fn metadata(batch: usize) -> Result<Tensor> {
        let mut values = Vec::new();
        for b in 0..batch {
            let support = (0..SUPPORT_STEPS)
                .map(|step| Transition {
                    before: vec![0; FRAME_SIDE * FRAME_SIDE],
                    after: vec![0; FRAME_SIDE * FRAME_SIDE],
                    action: [[2, 0, 3], [1, 3, 0]][b % 2][step],
                })
                .collect::<Vec<_>>();
            values.extend(task::inputs(&support, &support[0].before)?.metadata);
        }
        Ok(Tensor::from_vec(
            values,
            (batch, TOKENS, META_DIM),
            &Device::Cpu,
        )?)
    }
    fn one_hot(batch: usize) -> Result<Tensor> {
        let mut values = vec![0f32; batch * OBSERVED_FRAMES * 2 * PATCH_COUNT];
        for b in 0..batch {
            let agents = [[18, 19, 19, 27, 27, 26, 10], [36, 28, 28, 29, 29, 37, 45]][b % 2];
            for (frame, agent) in agents.into_iter().enumerate() {
                let base = (b * OBSERVED_FRAMES + frame) * 2 * PATCH_COUNT;
                values[base + agent] = 1.0;
                values[base + PATCH_COUNT + if frame == 6 { [11, 37][b % 2] } else { 63 }] = 1.0;
            }
        }
        Ok(Tensor::from_vec(
            values,
            (batch, OBSERVED_FRAMES, 2, PATCH_COUNT),
            &Device::Cpu,
        )?)
    }
    fn values(t: &Tensor) -> Result<Vec<f32>> {
        Ok(t.flatten_all()?.to_vec1::<f32>()?)
    }
    fn uniform(batch: usize) -> Result<Tensor> {
        Ok(Tensor::full(
            1.0f32 / 64.0,
            (batch, 7, 2, 64),
            &Device::Cpu,
        )?)
    }

    #[test]
    fn one_hot_coordinates_and_public_actions_match_both_batches() -> Result<()> {
        let output = records(&one_hot(2)?, &metadata(2)?, Control::Factual)?;
        assert_eq!(output.dims(), &[2, 4, 7]);
        let expected = [
            1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0.,
            1., 0., 0., 0., 0., 0., 1., 0., -1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
            0., 1., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1.,
        ];
        for (actual, expected) in values(&output)?.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(!output.track_op());
        Ok(())
    }

    #[test]
    fn soft_expectations_match_independent_coordinate_sums() -> Result<()> {
        let attention = one_hot(2)?.affine(0.75, 0.25 / PATCH_COUNT as f64)?;
        let result = values(&records(&attention, &metadata(2)?, Control::Factual)?)?;
        let weights = values(&attention)?;
        for b in 0..2 {
            let position = |frame: usize, role: usize, axis: usize| {
                (0..PATCH_COUNT)
                    .map(|patch| {
                        let coord = if axis == 0 { patch % 8 } else { patch / 8 };
                        f64::from(weights[((b * 7 + frame) * 2 + role) * 64 + patch]) * coord as f64
                    })
                    .sum::<f64>()
            };
            for step in 0..4 {
                for axis in 0..2 {
                    let expected = if step < 3 {
                        position(2 * step + 1, 0, axis) - position(2 * step, 0, axis)
                    } else {
                        position(6, 1, axis) - position(6, 0, axis)
                    };
                    assert!((f64::from(result[(b * 4 + step) * 7 + axis]) - expected).abs() < 2e-6);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn controls_change_only_the_declared_displacements() -> Result<()> {
        let attention = one_hot(2)?;
        let metadata = metadata(2)?;
        let before = values(&metadata)?;
        let factual = values(&records(&attention, &metadata, Control::Factual)?)?;
        for control in [
            Control::UniformAttention,
            Control::EffectsZero,
            Control::QueryZero,
        ] {
            let output = values(&records(&attention, &metadata, control)?)?;
            for (row, (actual, factual)) in output
                .chunks_exact(7)
                .zip(factual.chunks_exact(7))
                .enumerate()
            {
                assert_eq!(&actual[2..], &factual[2..]);
                let zero = control == Control::UniformAttention
                    || (control == Control::EffectsZero && row % 4 < 3)
                    || (control == Control::QueryZero && row % 4 == 3);
                assert_eq!(&actual[..2], if zero { &[0., 0.] } else { &factual[..2] });
            }
        }
        assert_eq!(values(&metadata)?, before);
        Ok(())
    }

    #[test]
    fn metadata_corruption_fails_closed() -> Result<()> {
        let valid = values(&metadata(1)?)?;
        for case in 0..10 {
            let mut bad = valid.clone();
            match case {
                0 => bad[7] = 0.1,
                1 => bad[3 + 2] = 0.5,
                2 => {
                    // A distinctness violation, otherwise consistent encoding.
                    for frame in [2, 3] {
                        for patch in 0..64 {
                            let offset = (frame * 64 + patch) * 10 + 3;
                            bad[offset..offset + 4].copy_from_slice(&[0., 0., 1., 0.]);
                        }
                    }
                }
                3 => bad[64 * 10 + 3 + 2] = 0.0,
                4 => bad[10 + 3 + 2] = 0.0,
                5 => bad[0] = 0.0,
                6 => bad[9] = 1.0,
                7 => bad[6 * 64 * 10 + 3] = 1.0,
                8 => bad[8] = f32::NAN,
                _ => bad[8] = f32::INFINITY,
            }
            let bad = Tensor::from_vec(bad, (1, TOKENS, META_DIM), &Device::Cpu)?;
            assert!(
                records(&one_hot(1)?, &bad, Control::Factual).is_err(),
                "case {case}"
            );
        }
        Ok(())
    }

    #[test]
    fn attention_and_shape_corruption_fail_even_when_uniformized() -> Result<()> {
        let metadata = metadata(1)?;
        for value in [-0.1, 1.1, f32::NAN, f32::INFINITY, 0.0] {
            let bad = Tensor::full(value, (1, 7, 2, 64), &Device::Cpu)?;
            assert!(records(&bad, &metadata, Control::UniformAttention).is_err());
        }
        for shape in [(0, 7, 2, 64), (1, 6, 2, 64), (1, 7, 1, 64), (1, 7, 2, 63)] {
            assert!(records(
                &Tensor::zeros(shape, DType::F32, &Device::Cpu)?,
                &metadata,
                Control::Factual
            )
            .is_err());
        }
        assert!(records(
            &uniform(1)?.to_dtype(DType::F64)?,
            &metadata,
            Control::Factual
        )
        .is_err());
        assert!(records(
            &uniform(1)?,
            &metadata.to_dtype(DType::F64)?,
            Control::Factual
        )
        .is_err());
        assert!(records(&uniform(2)?, &metadata, Control::Factual).is_err());
        Ok(())
    }

    #[test]
    fn exact_coordinate_gradients_reach_only_unclamped_attention() -> Result<()> {
        let attention = Var::from_tensor(&uniform(2)?)?;
        let metadata = metadata(2)?;
        for control in [Control::Factual, Control::EffectsZero, Control::QueryZero] {
            let output = records(attention.as_tensor(), &metadata, control)?;
            let gradients = output.sum_all()?.backward()?;
            let gradient = values(gradients.get(attention.as_tensor()).unwrap())?;
            for b in 0..2 {
                for frame in 0..7 {
                    for role in 0..2 {
                        for patch in 0..64 {
                            let sign = if frame < 6 {
                                if role == 1 || control == Control::EffectsZero {
                                    0.0
                                } else if frame % 2 == 0 {
                                    -1.0
                                } else {
                                    1.0
                                }
                            } else if control == Control::QueryZero {
                                0.0
                            } else if role == 0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let expected = sign * ((patch % 8 + patch / 8) as f32);
                            let actual = gradient[((b * 7 + frame) * 2 + role) * 64 + patch];
                            assert!(actual.is_finite() && (actual - expected).abs() < 1e-6);
                        }
                    }
                }
            }
        }
        assert!(!records(attention.as_tensor(), &metadata, Control::UniformAttention)?.track_op());
        Ok(())
    }
}
