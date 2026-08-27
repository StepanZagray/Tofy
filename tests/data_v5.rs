use anyhow::Result;
use rayon::ThreadPoolBuilder;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use tofy::domain::Split;
use tofy::p2::data::{
    apply_episode_operator, census_branch_coverage, census_event_labels,
    compose_mixed_stream_batch, foundation_v2_stream_schedule, generate_coordinate_one_step,
    generate_factual_branch_group, generate_hazard_one_step, palette, BranchGroup,
    FactualActionBranch, FactualBatch, MixedStreamConfig, MixedStreamKind, OperatorFamily,
    V5DataSplit, FACTUAL_BRANCHES_PER_GROUP, FRAME_SIDE, GOAL_FEATURES_DIM, V5_PLAYFIELD_HEIGHT,
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

fn batch_hash(batch: &tofy::p2::data::MixedStreamBatch) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(batch)?)))
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
fn realized_stream_proportions_are_explicit_and_keep_factual_groups_intact() -> Result<()> {
    for (batch_size, expected) in [
        (
            512,
            [
                [182, 100, 102, 77, 51],
                [159, 130, 105, 79, 39],
                [136, 160, 108, 81, 27],
            ],
        ),
        (
            2_048,
            [
                [717, 410, 409, 307, 205],
                [635, 520, 420, 315, 158],
                [546, 640, 431, 323, 108],
            ],
        ),
    ] {
        let config = test_config(batch_size);
        for (progress, expected_counts) in [0.0, 0.5, 1.0].into_iter().zip(expected) {
            let realized = config.realized_proportions(progress)?;
            let counts = [
                realized.counts[&MixedStreamKind::RandomOneStep],
                realized.counts[&MixedStreamKind::FactualBranches],
                realized.counts[&MixedStreamKind::Exploration],
                realized.counts[&MixedStreamKind::SequentialFragments],
                realized.counts[&MixedStreamKind::HazardOneStep],
            ];
            assert_eq!(
                counts, expected_counts,
                "batch {batch_size}, progress {progress}"
            );
            assert_eq!(counts.iter().sum::<usize>(), batch_size);
            for (actual, expected) in [
                realized.fractions.random_one_step,
                realized.fractions.factual_branches,
                realized.fractions.exploration,
                realized.fractions.sequential_fragments,
                realized.fractions.hazard_one_step,
            ]
            .into_iter()
            .zip(expected_counts)
            {
                approx(actual, expected as f32 / batch_size as f32);
            }
            assert_eq!(
                counts[1] % FACTUAL_BRANCHES_PER_GROUP,
                0,
                "factual rows remain complete groups"
            );
        }
    }

    let invalid = MixedStreamConfig {
        batch_size: FACTUAL_BRANCHES_PER_GROUP,
        ..test_config(512)
    };
    assert!(invalid.validate().is_err());
    Ok(())
}

#[test]
fn hazard_rows_are_positive_avoid_hazard_failures() -> Result<()> {
    let rows = generate_hazard_one_step(0xBADA_5505, 17, Split::Train, 128)?;
    assert_eq!(rows.len(), 128);
    for row in &rows {
        assert_eq!(row.goal_failed, Some(true));
        assert!(row.goal_features.values.iter().any(|&value| value != 0.0));
        assert_ne!(
            &row.current.pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE],
            &row.next.pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE],
            "hazard entry must change the semantic board"
        );
    }
    let census = census_event_labels(&rows);
    assert_eq!(census.rows, 128);
    assert_eq!(census.labeled[2], 128);
    assert_eq!(census.positive[2], 128);
    Ok(())
}

#[test]
fn coordinate_rows_stay_inside_the_playfield_and_keep_one_agent() -> Result<()> {
    let rows = generate_coordinate_one_step(0xC006_DA7A, 29, Split::Train, 128)?;
    assert_eq!(rows.len(), 128);
    for row in rows {
        assert_eq!(row.action.id, 6);
        assert!(
            usize::from(row.action.x.expect("ACTION6 x"))
                < usize::from(row.provenance.content_width)
        );
        assert!(
            usize::from(row.action.y.expect("ACTION6 y"))
                < usize::from(row.provenance.content_height)
        );
        for frame in [&row.current, &row.next] {
            assert_eq!(
                frame.pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE]
                    .iter()
                    .filter(|&&pixel| pixel == palette::AGENT)
                    .count(),
                1
            );
        }
    }
    Ok(())
}

#[test]
fn goal_dropout_census_reports_eligible_and_changed_rows() -> Result<()> {
    let batch = compose_mixed_stream_batch(&test_config(200), 0.35, 17, V5DataSplit::Train)?;
    let census = batch.goal_dropout_census();
    // Current deterministic golden for the fixed seed/batch/progress above.
    assert_eq!(census.total, 200);
    assert_eq!(census.eligible, 47);
    assert_eq!(census.changed, 13);
    assert_eq!(census.final_zero_goal, 166);
    assert!(census.changed <= census.eligible);
    assert!(census.eligible <= census.total);
    assert!(census.final_zero_goal >= census.changed);
    Ok(())
}

#[test]
fn branch_coverage_census_rejects_missing_duplicate_and_collapsed_controls() -> Result<()> {
    let rows = (0..16)
        .map(|episode_id| {
            generate_factual_branch_group(0xC0DE_C0DE, episode_id, Split::Train)
                .map(|group| group.into_transitions().collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let census = census_branch_coverage(&rows);
    // Current deterministic golden for this fixed 16-group population.
    let strata = census
        .strata
        .iter()
        .map(|stratum| {
            (
                stratum.family.as_str(),
                stratum.action_id,
                stratum.eligible_rows,
                stratum.changed_outcomes,
                stratum.distinct_effect_classes,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        strata,
        vec![
            ("factual_branch_v5", 1, 8, 0, 8),
            ("factual_branch_v5", 2, 8, 1, 8),
            ("factual_branch_v5", 3, 8, 8, 8),
            ("factual_branch_v5", 4, 8, 8, 8),
            ("factual_branch_v5", 5, 8, 8, 8),
            ("factual_branch_v5", 6, 32, 29, 30),
            ("factual_branch_v5", 7, 8, 8, 8),
            ("factual_coordinate_branch", 1, 8, 0, 8),
            ("factual_coordinate_branch", 2, 8, 1, 8),
            ("factual_coordinate_branch", 3, 8, 8, 8),
            ("factual_coordinate_branch", 4, 8, 8, 8),
            ("factual_coordinate_branch", 5, 8, 8, 8),
            ("factual_coordinate_branch", 6, 32, 27, 29),
            ("factual_coordinate_branch", 7, 8, 8, 8),
        ]
    );
    assert!(census.missing_action_keys.is_empty());
    assert!(census.duplicate_action_keys.is_empty());
    census.validate()?;

    let mut missing = rows.clone();
    let missing_index = missing
        .iter()
        .position(|row| row.action.id == 1)
        .expect("ACTION1 exists");
    missing.remove(missing_index);
    let err = census_branch_coverage(&missing).validate().unwrap_err();
    assert!(err.to_string().contains("missing action keys"));

    let mut duplicate = rows.clone();
    let first_action6 = duplicate
        .iter()
        .position(|row| row.action.id == 6)
        .expect("first ACTION6 exists");
    let second_action6 = duplicate
        .iter()
        .enumerate()
        .skip(first_action6 + 1)
        .find(|(_, row)| row.action.id == 6)
        .map(|(index, _)| index)
        .expect("second ACTION6 exists");
    duplicate[second_action6].action = duplicate[first_action6].action.clone();
    let err = census_branch_coverage(&duplicate).validate().unwrap_err();
    assert!(err.to_string().contains("duplicate action keys"));

    let mut collapsed = rows.clone();
    let outcome = collapsed[0].next.clone();
    for row in &mut collapsed {
        row.next = outcome.clone();
    }
    let err = census_branch_coverage(&collapsed).validate().unwrap_err();
    assert!(err.to_string().contains("no distinct effect classes"));
    Ok(())
}

#[test]
fn factual_groups_stay_intact_and_effect_labels_ignore_status_row() -> Result<()> {
    let batch = compose_mixed_stream_batch(&test_config(50), 0.0, 3, V5DataSplit::Train)?;
    let event_census = batch.event_label_census();
    assert_eq!(event_census.labeled[3], 0);
    assert_eq!(event_census.positive[3], 0);
    assert!(batch.samples().iter().all(|sample| {
        sample.transition.exhausted.is_none()
            && (!sample.provenance.goal_dropped
                || (sample.transition.goal_satisfied.is_none()
                    && sample.transition.goal_failed.is_none()))
    }));
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
    let batch = compose_mixed_stream_batch(&test_config(170), 0.35, 17, V5DataSplit::Train)?;
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
        let batch = compose_mixed_stream_batch(&test_config(50), 0.0, 9, split)?;
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
    let config = test_config(120);
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
        &test_config(100),
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

#[test]
fn mixed_stream_bytes_are_identical_across_rayon_thread_counts() -> Result<()> {
    let config = test_config(170);
    let serial = ThreadPoolBuilder::new().num_threads(1).build()?;
    let parallel = ThreadPoolBuilder::new().num_threads(4).build()?;
    for (progress, batch_index, split) in [
        (0.35, 17, V5DataSplit::Train),
        (0.73, 88, V5DataSplit::Train),
        (0.0, 9, V5DataSplit::Translated7x7),
    ] {
        let one =
            serial.install(|| compose_mixed_stream_batch(&config, progress, batch_index, split))?;
        let many = parallel
            .install(|| compose_mixed_stream_batch(&config, progress, batch_index, split))?;
        assert_eq!(batch_hash(&one)?, batch_hash(&many)?);
        assert_eq!(one, many);
        if (progress, batch_index, split) == (0.35, 17, V5DataSplit::Train) {
            // Re-pinned after hazard rows gained their real avoid-hazard
            // labels and the tolerance-compliant 170-row allocation.
            assert_eq!(
                batch_hash(&one)?,
                "036fa2fdd57f8a10fc65b17a2111fa9d9badd441a56fa8cbfb8c1afed790e4d0"
            );
        }
    }
    Ok(())
}
