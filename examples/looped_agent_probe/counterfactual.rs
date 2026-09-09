//! Frozen successor readouts; targets come only from the synthetic simulator.
use anyhow::{ensure, Result};
use candle_core::{Device, D};
use serde::Serialize;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tofy::p2::looped_agent::model::LoopedAgent;
use tofy::p2::looped_agent::task::{self, Sample, Split};
use tofy::p2::looped_agent::{ACTIONS, PALETTE, PATCH_COUNT, PATCH_PIXELS};

use super::{deadline, file_hash, input_hash, inspected_predict, predict, query_hash, PIXELS};

const SCHEMA: &str = "looped-successor-counterfactual-v1";
const EPISODE_TAG: u64 = 0x43464c4f4f50;
const PROBABILITY_TOLERANCE: f64 = 1e-5;
const PROBABILITIES_PER_INPUT: usize = ACTIONS * PIXELS * PALETTE;
const STATES_PER_INPUT: usize = (1 + 2 * ACTIONS) * PIXELS;

#[derive(Serialize)]
struct InputIdentity {
    schema: &'static str,
    input_index: usize,
    layout_index: usize,
    episode_id: u64,
    data_seed: u64,
    permutation_id: usize,
    split: Split,
    min_distance: usize,
    max_distance: usize,
    oracle_distance: usize,
    observed_support_action_ids: Vec<usize>,
    inferred_controls: [usize; ACTIONS],
    input_sha256: String,
    query_sha256: String,
}

#[derive(PartialEq)]
struct LayoutReference {
    current: Vec<u32>,
    oracle_distance: usize,
    observed_support_action_ids: Vec<usize>,
}

fn panel_input(
    data_seed: u64,
    layout_index: usize,
    permutation_id: usize,
    reference: &mut Option<LayoutReference>,
) -> Result<(InputIdentity, Sample)> {
    let episode_id = EPISODE_TAG + layout_index as u64;
    let (min_distance, max_distance) = if layout_index.is_multiple_of(2) {
        (1, 1)
    } else {
        (2, 10)
    };
    let episode = task::episode_with_permutation(
        data_seed,
        episode_id,
        permutation_id,
        min_distance,
        max_distance,
    )?;
    let sample = task::sample(&episode)?;
    let (_, oracle_distance) = task::oracle(&episode.support, &episode.maze.render())?;
    ensure!(
        (min_distance..=max_distance).contains(&oracle_distance),
        "query distance outside requested population"
    );
    let observed_support_action_ids: Vec<_> = episode.support.iter().map(|s| s.action).collect();
    let layout = LayoutReference {
        current: sample.current.clone(),
        oracle_distance,
        observed_support_action_ids: observed_support_action_ids.clone(),
    };
    if let Some(expected) = reference {
        ensure!(
            expected == &layout,
            "query pixels, oracle distance or support action order vary across control permutations"
        );
    } else {
        *reference = Some(layout);
    }
    let inferred_controls = task::inferred_controls(&episode.support)?;
    let mut sorted = inferred_controls;
    sorted.sort_unstable();
    ensure!(
        sorted == [0, 1, 2, 3],
        "public inferred controls are not bijective"
    );
    ensure!(
        inferred_controls == task::permutations()[permutation_id],
        "public controls disagree with generating permutation"
    );
    let identity = InputIdentity {
        schema: SCHEMA,
        input_index: layout_index * 24 + permutation_id,
        layout_index,
        episode_id,
        data_seed,
        permutation_id,
        split: if task::permutation_ids(Split::HeldOut).contains(&permutation_id) {
            Split::HeldOut
        } else {
            Split::Train
        },
        min_distance,
        max_distance,
        oracle_distance,
        observed_support_action_ids,
        inferred_controls,
        input_sha256: input_hash(&sample.inputs),
        query_sha256: query_hash(&sample.current),
    };
    Ok((identity, sample))
}

/// Hash the exact evaluation panel before constructing a device or model.
pub(super) fn audit(args: &super::Args, started: Instant) -> Result<Value> {
    ensure!(args.eval_episodes > 0, "counterfactual audit needs layouts");
    let path = args.output_dir.join("successor-input-audit.jsonl");
    let mut rows = BufWriter::new(File::create_new(&path)?);
    let mut input_rows = 0;
    for layout_index in 0..args.eval_episodes {
        let mut reference = None;
        for permutation_id in 0..24 {
            deadline(args, started)?;
            let (identity, _) =
                panel_input(args.data_seed, layout_index, permutation_id, &mut reference)?;
            serde_json::to_writer(&mut rows, &identity)?;
            rows.write_all(b"\n")?;
            input_rows += 1;
        }
    }
    rows.flush()?;
    drop(rows);
    Ok(json!({
        "schema": SCHEMA,
        "task_schema": task::SCHEMA,
        "status": "complete_pending_analysis",
        "evidence_class": "data_audit",
        "optimizer_updates": 0,
        "model_forwards": 0,
        "data_seed": args.data_seed,
        "episode_id_base": EPISODE_TAG,
        "layouts": args.eval_episodes,
        "input_rows": input_rows,
        "action_rows": input_rows * ACTIONS,
        "query_distance_and_support_action_order_identical_across_rules": true,
        "public_inferred_controls_bijective_and_match_generator": true,
        "artifacts": [{ "file": "successor-input-audit.jsonl", "bytes": path.metadata()?.len(), "sha256": file_hash(&path)? }],
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }))
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ByteRange {
    offset_bytes: usize,
    length_bytes: usize,
}

#[derive(Debug, Serialize)]
struct Offsets {
    probabilities: ByteRange,
    states: ByteRange,
    current: ByteRange,
    targets: ByteRange,
    predicted: ByteRange,
}

fn offsets(input_index: usize) -> Offsets {
    let state_start = input_index * STATES_PER_INPUT;
    Offsets {
        probabilities: ByteRange {
            offset_bytes: input_index * PROBABILITIES_PER_INPUT * 4,
            length_bytes: PROBABILITIES_PER_INPUT * 4,
        },
        states: ByteRange {
            offset_bytes: state_start,
            length_bytes: STATES_PER_INPUT,
        },
        current: ByteRange {
            offset_bytes: state_start,
            length_bytes: PIXELS,
        },
        targets: ByteRange {
            offset_bytes: state_start + PIXELS,
            length_bytes: ACTIONS * PIXELS,
        },
        predicted: ByteRange {
            offset_bytes: state_start + (1 + ACTIONS) * PIXELS,
            length_bytes: ACTIONS * PIXELS,
        },
    }
}

fn action_range(range: ByteRange, action: usize) -> ByteRange {
    let length_bytes = range.length_bytes / ACTIONS;
    ByteRange {
        offset_bytes: range.offset_bytes + action * length_bytes,
        length_bytes,
    }
}

fn write_binary(
    probabilities_file: &mut impl Write,
    states_file: &mut impl Write,
    probabilities: &[f32],
    current: &[u32],
    targets: &[u32],
    predicted: &[u32],
) -> Result<()> {
    ensure!(
        probabilities.len() == PROBABILITIES_PER_INPUT
            && current.len() == PIXELS
            && targets.len() == ACTIONS * PIXELS
            && predicted.len() == ACTIONS * PIXELS,
        "invalid successor binary dimensions"
    );
    let states = current
        .iter()
        .chain(targets)
        .chain(predicted)
        .map(|&pixel| {
            ensure!(
                (pixel as usize) < PALETTE,
                "invalid successor palette index"
            );
            Ok(pixel as u8)
        })
        .collect::<Result<Vec<_>>>()?;
    let probability_bytes: Vec<_> = probabilities.iter().flat_map(|p| p.to_le_bytes()).collect();
    probabilities_file.write_all(&probability_bytes)?;
    states_file.write_all(&states)?;
    Ok(())
}

fn validate_probabilities(values: &[f32], classes: usize) -> Result<f64> {
    ensure!(
        !values.is_empty() && values.len().is_multiple_of(classes),
        "invalid probability dimensions"
    );
    let mut maximum_error = 0.0f64;
    for pixel in values.chunks_exact(classes) {
        ensure!(
            pixel
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "non-finite or out-of-range probability"
        );
        let error = (pixel.iter().map(|&p| f64::from(p)).sum::<f64>() - 1.0).abs();
        maximum_error = maximum_error.max(error);
        ensure!(
            error <= PROBABILITY_TOLERANCE,
            "probabilities do not sum to one: error {error}"
        );
    }
    Ok(maximum_error)
}

#[derive(Debug, Default, Serialize)]
struct Region {
    correct: usize,
    total: usize,
}

#[derive(Debug, Serialize)]
struct Metrics {
    class: &'static str,
    exact: bool,
    copy_exact: bool,
    changed: Region,
    unchanged: Region,
    vacated: Region,
    destination: Region,
    patch_inconsistency_count: usize,
}

fn metrics(current: &[u32], target: &[u32], predicted: &[u32], reward: f32) -> Result<Metrics> {
    ensure!(
        current.len() == PIXELS && target.len() == PIXELS && predicted.len() == PIXELS,
        "invalid successor metric dimensions"
    );
    ensure!(reward == 0.0 || reward == 1.0, "nonbinary target reward");
    let copy_exact = target == current;
    ensure!(
        !(reward == 1.0 && copy_exact),
        "terminal target cannot be a copy"
    );
    let mut result = Metrics {
        class: if reward == 1.0 {
            "terminal"
        } else if copy_exact {
            "blocked"
        } else {
            "nonterminal"
        },
        exact: target == predicted,
        copy_exact,
        changed: Region::default(),
        unchanged: Region::default(),
        vacated: Region::default(),
        destination: Region::default(),
        patch_inconsistency_count: predicted
            .chunks_exact(PATCH_PIXELS)
            .filter(|patch| patch.iter().any(|p| *p != patch[0]))
            .count(),
    };
    for ((&before, &after), &prediction) in current.iter().zip(target).zip(predicted) {
        let correct = usize::from(after == prediction);
        if before == after {
            result.unchanged.total += 1;
            result.unchanged.correct += correct;
            continue;
        }
        result.changed.total += 1;
        result.changed.correct += correct;
        let region = if before == 2 && after == 0 {
            &mut result.vacated
        } else {
            ensure!(
                [0, 3].contains(&before) && [2, 4].contains(&after),
                "unexpected simulator changed-region transition {before}->{after}"
            );
            &mut result.destination
        };
        region.total += 1;
        region.correct += correct;
    }
    ensure!(
        result.changed.total == result.vacated.total + result.destination.total,
        "incomplete changed-region decomposition"
    );
    Ok(result)
}

pub(super) fn evaluate(
    args: &super::Args,
    model: &LoopedAgent,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    ensure!(
        args.eval_episodes > 0,
        "counterfactual evaluation needs layouts"
    );
    let probability_path = args.output_dir.join("successor-probabilities.f32");
    let states_path = args.output_dir.join("successor-states.u8");
    let rows_path = args.output_dir.join("successor-rows.jsonl");
    let mut probability_file = BufWriter::new(File::create_new(&probability_path)?);
    let mut states_file = BufWriter::new(File::create_new(&states_path)?);
    let mut rows_file = BufWriter::new(File::create_new(&rows_path)?);
    let held_out = task::permutation_ids(Split::HeldOut);
    let mut input_index = 0;
    let mut class_counts = [0usize; 3];
    let mut demonstrated_counts = [0usize; 2];
    let mut changed_pixels = 0;
    let mut maximum_probability_error = 0.0f64;
    for layout_index in 0..args.eval_episodes {
        let mut reference = None;
        for permutation_id in 0..24 {
            deadline(args, started)?;
            let (identity, sample) =
                panel_input(args.data_seed, layout_index, permutation_id, &mut reference)?;
            ensure!(
                identity.input_index == input_index,
                "panel input order mismatch"
            );
            let out = if input_index == 0 && args.profile_eval {
                inspected_predict(args, model, sample.inputs.clone(), device)?
            } else {
                predict(
                    model,
                    std::slice::from_ref(&sample.inputs),
                    args.loops,
                    device,
                )?
            };
            ensure!(
                out.next_logits.dims() == [1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE]
                    && out.policy_logits.dims() == [1, ACTIONS]
                    && out.reward_logits.dims() == [1, ACTIONS]
                    && out.value.dims() == [1, 1],
                "unexpected frozen model readout shape"
            );
            let probabilities = candle_nn::ops::softmax(&out.next_logits, D::Minus1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            maximum_probability_error =
                maximum_probability_error.max(validate_probabilities(&probabilities, PALETTE)?);
            let policy_logits = out.policy_logits.flatten_all()?.to_vec1::<f32>()?;
            let reward_logits = out.reward_logits.flatten_all()?.to_vec1::<f32>()?;
            let value_logits = out.value.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                policy_logits
                    .iter()
                    .chain(&reward_logits)
                    .chain(&value_logits)
                    .all(|x| x.is_finite()),
                "non-finite policy, reward or value logits"
            );
            let policy_probabilities = candle_nn::ops::softmax(&out.policy_logits, D::Minus1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            validate_probabilities(&policy_probabilities, ACTIONS)?;
            let value = candle_nn::ops::sigmoid(&out.value)?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            let reward_probabilities = candle_nn::ops::sigmoid(&out.reward_logits)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            ensure!(
                reward_probabilities
                    .iter()
                    .chain(std::iter::once(&value))
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
                "non-finite or out-of-range sigmoid value or reward"
            );
            // Resolve probability ties at the first palette index, deterministically.
            let predicted: Vec<u32> = probabilities
                .chunks_exact(PALETTE)
                .map(|pixel| {
                    (1..PALETTE).fold(0, |best, i| if pixel[i] > pixel[best] { i } else { best })
                        as u32
                })
                .collect();
            let byte_offsets = offsets(input_index);
            write_binary(
                &mut probability_file,
                &mut states_file,
                &probabilities,
                &sample.current,
                &sample.next,
                &predicted,
            )?;
            let mut action_rows = Vec::new();
            for action in 0..ACTIONS {
                let target = &sample.next[action * PIXELS..(action + 1) * PIXELS];
                let prediction = &predicted[action * PIXELS..(action + 1) * PIXELS];
                let action_metrics =
                    metrics(&sample.current, target, prediction, sample.rewards[action])?;
                let demonstrated = identity.observed_support_action_ids.contains(&action);
                demonstrated_counts[usize::from(demonstrated)] += 1;
                class_counts[match action_metrics.class {
                    "blocked" => 0,
                    "terminal" => 1,
                    _ => 2,
                }] += 1;
                changed_pixels += action_metrics.changed.total;
                let target_sha256 = query_hash(target);
                let mut row = serde_json::to_value(action_metrics)?;
                row.as_object_mut()
                    .expect("serialized metric object")
                    .extend(
                        json!({
                            "action": action,
                            "direction": identity.inferred_controls[action],
                            "demonstrated": demonstrated,
                            "target_reward": sample.rewards[action],
                            "target_sha256": target_sha256,
                            "predicted_sha256": query_hash(prediction),
                            "probabilities": action_range(byte_offsets.probabilities, action),
                            "target_state": action_range(byte_offsets.targets, action),
                            "predicted_state": action_range(byte_offsets.predicted, action),
                        })
                        .as_object()
                        .expect("action metadata object")
                        .clone(),
                    );
                action_rows.push(row);
            }
            let mut row = serde_json::to_value(identity)?;
            row.as_object_mut().expect("input identity object").extend(
                json!({
                    "policy_logits": policy_logits,
                    "policy_probabilities": policy_probabilities,
                    "value": value,
                    "reward_probabilities": reward_probabilities,
                    "target_policy": sample.policy,
                    "target_value": sample.value,
                    "byte_offsets": byte_offsets,
                    "actions": action_rows,
                })
                .as_object()
                .expect("readout object")
                .clone(),
            );
            serde_json::to_writer(&mut rows_file, &row)?;
            rows_file.write_all(b"\n")?;
            input_index += 1;
        }
    }
    probability_file.flush()?;
    states_file.flush()?;
    rows_file.flush()?;
    drop((probability_file, states_file, rows_file));
    ensure!(
        probability_path.metadata()?.len() == (input_index * PROBABILITIES_PER_INPUT * 4) as u64
            && states_path.metadata()?.len() == (input_index * STATES_PER_INPUT) as u64,
        "successor output byte length mismatch"
    );
    let mut artifacts = Vec::new();
    for path in [&probability_path, &states_path, &rows_path] {
        artifacts.push(json!({
            "file": path.file_name().expect("artifact filename").to_string_lossy(),
            "bytes": path.metadata()?.len(),
            "sha256": file_hash(path)?,
        }));
    }
    Ok(json!({
        "schema": SCHEMA,
        "task_schema": task::SCHEMA,
        "status": "complete_pending_analysis",
        "optimizer_updates": 0,
        "data_seed": args.data_seed,
        "episode_id_base": EPISODE_TAG,
        "layout_count": args.eval_episodes,
        "layout_distance_distribution": "even indices: distance 1; odd indices: distance 2 through 10 inclusive",
        "permutation_ids": (0..24).collect::<Vec<_>>(),
        "train_permutation_ids": task::permutation_ids(Split::Train),
        "held_out_permutation_ids": held_out,
        "loops": args.loops,
        "physical_batch": 1,
        "effective_batch": 1,
        "input_rows": input_index,
        "action_rows": input_index * ACTIONS,
        "action_classes": { "blocked": class_counts[0], "terminal": class_counts[1], "nonterminal": class_counts[2] },
        "demonstrated_action_rows": demonstrated_counts[1],
        "undemonstrated_action_rows": demonstrated_counts[0],
        "changed_pixels": changed_pixels,
        "causal_positive_definition": "simulator successor changes; changing a rule or direction alone is insufficient",
        "binary_layout": {
            "input_order": "layout_index ascending, then permutation_id 0 through 23",
            "probability_dtype": "little-endian IEEE-754 float32",
            "probability_shape_per_input": [ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE],
            "state_dtype": "uint8",
            "state_sections_per_input": ["current", "targets_by_action", "predicted_by_action"],
            "state_section_lengths": [PIXELS, ACTIONS * PIXELS, ACTIONS * PIXELS],
            "pixel_order": "patch-major, then within-patch row-major; never frame-raster order",
            "state_hash_encoding": "SHA-256 of patch-order palette indices encoded as little-endian u32 (query_hash convention)",
            "argmax_ties": "first palette index",
            "offset_unit": "bytes from start of named file"
        },
        "integrity": {
            "finite_probabilities_policy_value_rewards": true,
            "query_and_oracle_distance_identical_across_rules": true,
            "support_action_ids_and_order_identical_across_rules": true,
            "public_inferred_controls_bijective_and_match_generator": true,
            "probability_normalization_tolerance": PROBABILITY_TOLERANCE,
            "maximum_successor_probability_normalization_error": maximum_probability_error,
            "output_byte_lengths_verified": true
        },
        "profiled_forward_input_index": args.profile_eval.then_some(0),
        "artifacts": artifacts,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_layout_roundtrips_multiple_input_and_action_offsets() -> Result<()> {
        let mut probabilities_file = Vec::new();
        let mut states_file = Vec::new();
        for input_index in 0..2 {
            let probabilities: Vec<_> = (0..PROBABILITIES_PER_INPUT)
                .map(|i| (i + input_index * PROBABILITIES_PER_INPUT) as f32 / 13.0)
                .collect();
            let current: Vec<_> = (0..PIXELS)
                .map(|i| ((i + input_index) % PALETTE) as u32)
                .collect();
            let targets: Vec<_> = (0..ACTIONS * PIXELS)
                .map(|i| ((i + i / PIXELS + 2) % PALETTE) as u32)
                .collect();
            let predicted: Vec<_> = (0..ACTIONS * PIXELS)
                .map(|i| ((i * 3 + i / PIXELS + 8) % PALETTE) as u32)
                .collect();
            let layout = offsets(input_index);
            assert_eq!(probabilities_file.len(), layout.probabilities.offset_bytes);
            assert_eq!(states_file.len(), layout.states.offset_bytes);
            write_binary(
                &mut probabilities_file,
                &mut states_file,
                &probabilities,
                &current,
                &targets,
                &predicted,
            )?;
            let current_range = layout.current.offset_bytes
                ..layout.current.offset_bytes + layout.current.length_bytes;
            assert_eq!(
                &states_file[current_range],
                current.iter().map(|&p| p as u8).collect::<Vec<_>>()
            );
            for action in 0..ACTIONS {
                let range = action_range(layout.probabilities, action);
                let decoded: Vec<_> = probabilities_file
                    [range.offset_bytes..range.offset_bytes + range.length_bytes]
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    decoded,
                    probabilities[action * PIXELS * PALETTE..(action + 1) * PIXELS * PALETTE]
                );
                for (section, source) in
                    [(layout.targets, &targets), (layout.predicted, &predicted)]
                {
                    let range = action_range(section, action);
                    assert_eq!(
                        &states_file[range.offset_bytes..range.offset_bytes + range.length_bytes],
                        source[action * PIXELS..(action + 1) * PIXELS]
                            .iter()
                            .map(|&p| p as u8)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        assert_eq!(probabilities_file.len(), 2 * PROBABILITIES_PER_INPUT * 4);
        assert_eq!(states_file.len(), 2 * STATES_PER_INPUT);
        Ok(())
    }

    #[test]
    fn categories_and_changed_regions_distinguish_copy_and_correct_successors() -> Result<()> {
        let mut current = vec![0; PIXELS];
        current[..PATCH_PIXELS].fill(2);
        let blocked = metrics(&current, &current, &current, 0.0)?;
        assert_eq!(blocked.class, "blocked");
        assert!(blocked.exact && blocked.copy_exact);
        assert_eq!(blocked.changed.total, 0);
        assert_eq!(blocked.unchanged.correct, PIXELS);
        for terminal in [false, true] {
            let mut before = current.clone();
            if terminal {
                before[PATCH_PIXELS..2 * PATCH_PIXELS].fill(3);
            }
            let mut target = before.clone();
            target[..PATCH_PIXELS].fill(0);
            target[PATCH_PIXELS..2 * PATCH_PIXELS].fill(if terminal { 4 } else { 2 });
            let copied = metrics(&before, &target, &before, f32::from(terminal))?;
            assert_eq!(
                copied.class,
                if terminal { "terminal" } else { "nonterminal" }
            );
            assert!(!copied.exact && !copied.copy_exact);
            assert_eq!(
                (copied.changed.correct, copied.changed.total),
                (0, 2 * PATCH_PIXELS)
            );
            assert_eq!(copied.vacated.total, PATCH_PIXELS);
            assert_eq!(copied.destination.total, PATCH_PIXELS);
            let perfect = metrics(&before, &target, &target, f32::from(terminal))?;
            assert!(perfect.exact);
            assert_eq!(perfect.changed.correct, 2 * PATCH_PIXELS);
            assert_eq!(perfect.patch_inconsistency_count, 0);
            let mut imperfect = target.clone();
            imperfect[0] = 1;
            let imperfect = metrics(&before, &target, &imperfect, f32::from(terminal))?;
            assert_eq!(imperfect.patch_inconsistency_count, 1);
            assert_eq!(imperfect.vacated.correct, PATCH_PIXELS - 1);
        }
        Ok(())
    }

    #[test]
    fn registered_population_preserves_queries_and_identifies_all_rules() -> Result<()> {
        let mut classes = [0usize; 3];
        for layout_index in 0..32 {
            let mut reference = None;
            let mut by_direction: Vec<Option<Vec<u32>>> = vec![None; ACTIONS];
            for permutation_id in 0..24 {
                let (identity, sample) =
                    panel_input(20260910, layout_index, permutation_id, &mut reference)?;
                let controls = identity.inferred_controls;
                assert_eq!(controls, task::permutations()[permutation_id]);
                let distance = identity.oracle_distance;
                assert_eq!(identity.input_index, layout_index * 24 + permutation_id);
                assert_eq!(identity.input_sha256, input_hash(&sample.inputs));
                assert_eq!(identity.query_sha256, query_hash(&sample.current));
                assert_eq!(
                    (0..ACTIONS)
                        .filter(|a| identity.observed_support_action_ids.contains(a))
                        .count(),
                    3
                );
                for (action, direction) in controls.into_iter().enumerate() {
                    let target = &sample.next[action * PIXELS..(action + 1) * PIXELS];
                    if let Some(expected) = &by_direction[direction] {
                        assert_eq!(target, expected);
                    } else {
                        by_direction[direction] = Some(target.to_vec());
                    }
                    let metric = metrics(&sample.current, target, target, sample.rewards[action])?;
                    classes[match metric.class {
                        "blocked" => 0,
                        "terminal" => 1,
                        _ => 2,
                    }] += 1;
                    assert!(metric.exact);
                    if metric.class == "terminal" {
                        assert_eq!(distance, 1);
                    }
                    assert_eq!(
                        metric.changed.total,
                        if metric.copy_exact {
                            0
                        } else {
                            2 * PATCH_PIXELS
                        }
                    );
                }
            }
        }
        assert!(classes.iter().all(|&count| count > 0));
        assert_eq!(classes.iter().sum::<usize>(), 32 * 24 * ACTIONS);
        Ok(())
    }

    #[test]
    fn panel_identity_is_repeatable_and_rejects_changed_support_order() -> Result<()> {
        let mut reference = None;
        let (identity, _) = panel_input(20260910, 0, 0, &mut reference)?;
        let (repeated, _) = panel_input(20260910, 0, 0, &mut None)?;
        assert_eq!(
            serde_json::to_value(identity)?,
            serde_json::to_value(repeated)?
        );
        reference
            .as_mut()
            .unwrap()
            .observed_support_action_ids
            .swap(0, 1);
        assert!(panel_input(20260910, 0, 1, &mut reference).is_err());
        Ok(())
    }

    #[test]
    fn probability_validation_rejects_nonfinite_and_unnormalized_rows() -> Result<()> {
        assert_eq!(validate_probabilities(&[0.25; ACTIONS], ACTIONS)?, 0.0);
        for invalid in [
            [f32::NAN, 0.0, 0.0, 1.0],
            [f32::INFINITY, 0.0, 0.0, 1.0],
            [0.3; ACTIONS],
            [-0.1, 0.1, 0.5, 0.5],
        ] {
            assert!(validate_probabilities(&invalid, ACTIONS).is_err());
        }
        Ok(())
    }
}
