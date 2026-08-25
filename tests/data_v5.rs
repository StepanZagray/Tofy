use anyhow::Result;
use std::collections::BTreeSet;
use tofy::domain::Split;
use tofy::p2::data::{
    apply_episode_operator, compose_mixed_stream_batch, foundation_v2_stream_schedule,
    generate_factual_branch_group, BranchGroup, FactualActionBranch, FactualBatch,
    MixedStreamConfig, MixedStreamKind, OperatorFamily, V5DataSplit, FACTUAL_BRANCHES_PER_GROUP,
    FRAME_SIDE, GOAL_FEATURES_DIM, V5_PLAYFIELD_HEIGHT,
};

fn approx(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
}

fn test_config(batch_size: usize) -> MixedStreamConfig {
    MixedStreamConfig {
        batch_size,
        seed: 0xDADA_0005,
        ..MixedStreamConfig::default()
    }
}

#[test]
fn mixed_stream_schedule_anneals_to_the_specified_endpoints() {
    let start = foundation_v2_stream_schedule(0.0);
    approx(start.random_one_step, 0.35);
    approx(start.factual_branches, 0.20);
    approx(start.exploration, 0.20);
    approx(start.sequential_fragments, 0.15);
    approx(start.hazard_one_step, 0.10);
    approx(start.total(), 1.0);

    let middle = foundation_v2_stream_schedule(0.5);
    approx(middle.random_one_step, 0.30);
    approx(middle.factual_branches, 0.25);
    approx(middle.exploration, 0.20);
    approx(middle.sequential_fragments, 0.15);
    approx(middle.hazard_one_step, 0.075);

    let end = foundation_v2_stream_schedule(1.0);
    approx(end.random_one_step, 0.25);
    approx(end.factual_branches, 0.30);
    approx(end.exploration, 0.20);
    approx(end.sequential_fragments, 0.15);
    approx(end.hazard_one_step, 0.05);
    // ADR 0003's written endpoint weights sum to 95%; fixed-size composition
    // normalizes these exact raw weights rather than changing an endpoint.
    approx(end.total(), 0.95);
}

#[test]
fn factual_groups_stay_intact_and_effect_labels_ignore_status_row() -> Result<()> {
    let batch = compose_mixed_stream_batch(&test_config(100), 0.0, 3, V5DataSplit::Train)?;
    assert!(!batch.factual_group_ranges().is_empty());
    for range in batch.factual_group_ranges() {
        assert_eq!(range.len(), FACTUAL_BRANCHES_PER_GROUP);
        let ids = batch.samples()[range.clone()]
            .iter()
            .map(|sample| sample.provenance.branch_group_id.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 1);
        assert!(ids.iter().next().unwrap().is_some());
        assert!(batch.samples()[range.clone()]
            .iter()
            .all(|sample| sample.provenance.stream == MixedStreamKind::FactualBranches));
    }
    assert_eq!(
        batch.stream_counts()[&MixedStreamKind::FactualBranches],
        batch.factual_group_ranges().len() * FACTUAL_BRANCHES_PER_GROUP
    );
    let mixed_factual_rows = batch
        .factual_group_ranges()
        .iter()
        .flat_map(|range| {
            batch.samples()[range.clone()]
                .iter()
                .map(|sample| sample.transition.clone())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        batch.factual().expect("factual sidecar").rows(),
        mixed_factual_rows
    );

    let generated = generate_factual_branch_group(41, 8, Split::Train)?;
    let mut transitions = generated.into_transitions().collect::<Vec<_>>();
    transitions.sort_by_key(|sample| (sample.action.id, sample.action.x, sample.action.y));
    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
    let first_board = transitions[0].next.pixels[..status_start].to_vec();
    transitions[1].next.pixels[..status_start].copy_from_slice(&first_board);
    transitions[1].next.pixels[status_start..].fill(15);
    let group = BranchGroup::try_new(
        transitions
            .into_iter()
            .map(FactualActionBranch::try_from_transition)
            .collect::<Result<Vec<_>>>()?,
    )?;
    assert!(group.effect_equivalence_matrix()[0][1]);
    let factual = FactualBatch::from_groups(vec![group])?;
    let label = factual
        .pairwise_board_effect_labels()
        .into_iter()
        .find(|label| label.left_row == 0 && label.right_row == 1)
        .expect("first branch pair");
    assert!(
        label.equivalent,
        "status-only differences must be equivalent"
    );
    Ok(())
}

#[test]
fn augmented_operator_pairs_remain_exact_under_color_and_d4_conjugation() -> Result<()> {
    let batch = compose_mixed_stream_batch(&test_config(120), 0.35, 17, V5DataSplit::Train)?;
    let mut checked = 0usize;
    let mut saw_non_identity_color_permutation = false;
    for sample in batch.samples() {
        let permutation = &sample.provenance.augmentation.color_permutation;
        assert_eq!(permutation[0], 0);
        assert_eq!(
            permutation.iter().copied().collect::<BTreeSet<_>>().len(),
            16
        );
        saw_non_identity_color_permutation |= permutation
            .iter()
            .copied()
            .enumerate()
            .any(|(index, color)| usize::from(color) != index);
        if !matches!(sample.transition.action.id, 5 | 6) {
            continue;
        }
        let simulated = apply_episode_operator(
            &sample.transition.current,
            &sample.transition.action,
            sample.provenance.content_rect,
            sample.provenance.operator,
        )?;
        for (index, &inside) in sample.content_mask.values.iter().enumerate() {
            if inside == 1 {
                assert_eq!(
                    simulated.pixels[index], sample.transition.next.pixels[index],
                    "operator {:?}, D4 {:?}, pixel {index}",
                    sample.provenance.operator.family, sample.provenance.augmentation.d4
                );
            }
        }
        checked += 1;
    }
    assert!(
        checked >= 4,
        "batch should exercise ACTION5/ACTION6 branches"
    );
    assert!(saw_non_identity_color_permutation);
    Ok(())
}

#[test]
fn content_masks_match_exact_provenance_for_all_geometry_splits() -> Result<()> {
    for (split, expected_size) in [
        (V5DataSplit::UnseenSeed7x7, 7),
        (V5DataSplit::Composition8x8, 8),
        (V5DataSplit::Translated7x7, 7),
        (V5DataSplit::Size16x16, 16),
    ] {
        let batch = compose_mixed_stream_batch(&test_config(20), 0.0, 9, split)?;
        for sample in batch.samples() {
            let rect = sample.provenance.content_rect;
            assert_eq!((rect.width, rect.height), (expected_size, expected_size));
            assert_eq!(
                sample
                    .content_mask
                    .values
                    .iter()
                    .map(|&value| usize::from(value))
                    .sum::<usize>(),
                usize::from(expected_size) * usize::from(expected_size)
            );
            for y in 0..FRAME_SIDE as u8 {
                for x in 0..FRAME_SIDE as u8 {
                    let expected = u8::from(rect.contains(x, y));
                    assert_eq!(
                        sample.content_mask.values[usize::from(y) * FRAME_SIDE + usize::from(x)],
                        expected
                    );
                }
            }
            assert!(
                sample.content_mask.values[V5_PLAYFIELD_HEIGHT * FRAME_SIDE..]
                    .iter()
                    .all(|&value| value == 0)
            );
            if split == V5DataSplit::Translated7x7 {
                assert_ne!((rect.x, rect.y), (0, 0));
            }
        }
    }
    Ok(())
}

#[test]
fn fixed_seed_is_deterministic_and_operator_holdout_is_family_complete() -> Result<()> {
    let config = test_config(50);
    let first = compose_mixed_stream_batch(&config, 0.73, 88, V5DataSplit::Train)?;
    let second = compose_mixed_stream_batch(&config, 0.73, 88, V5DataSplit::Train)?;
    assert_eq!(first, second);
    let null = first
        .samples()
        .iter()
        .find(|sample| sample.transition.action.id == 0)
        .expect("random stream should train the NULL no-op action");
    assert_eq!(null.transition.noop, Some(true));
    assert_eq!(null.transition.current, null.transition.next);
    assert!(first
        .samples()
        .iter()
        .any(|sample| sample.provenance.goal_dropped));
    assert!(first.samples().iter().all(|sample| {
        !sample.provenance.goal_dropped
            || sample.transition.goal_features.values == [0.0; GOAL_FEATURES_DIM]
    }));

    let held_out = compose_mixed_stream_batch(
        &test_config(20),
        1.0,
        5,
        V5DataSplit::HeldOutOperator(OperatorFamily::SwapRegion),
    )?;
    assert!(held_out
        .samples()
        .iter()
        .all(|sample| sample.provenance.operator.family == OperatorFamily::SwapRegion));
    Ok(())
}
