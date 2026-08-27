use anyhow::Result;
use std::collections::BTreeSet;
use tofy::p2::data::{
    apply_episode_operator, census_rule_identifiability, generate_meta_episode,
    generate_meta_episode_shuffled_control, ContentRect, MetaEpisode, MetaEpisodeConfig,
    OperatorFamilySplit, V5DataSplit, FRAME_SIDE, V5_PLAYFIELD_HEIGHT,
};

const SEED: u64 = 0x4D45_5441_0001;

fn families() -> OperatorFamilySplit {
    OperatorFamilySplit::default()
}

fn board(pixels: &[u8]) -> &[u8] {
    &pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE]
}

fn true_episode(meta_episode_id: u64) -> Result<MetaEpisode> {
    generate_meta_episode(
        SEED,
        meta_episode_id,
        V5DataSplit::Train,
        &families(),
        &MetaEpisodeConfig::default(),
    )
}

#[test]
fn meta_episode_is_a_pure_function_of_its_inputs() -> Result<()> {
    let first = true_episode(11)?;
    let second = true_episode(11)?;
    assert_eq!(first, second);
    assert_eq!(serde_json::to_vec(&first)?, serde_json::to_vec(&second)?);
    let round_trip: MetaEpisode = serde_json::from_slice(&serde_json::to_vec(&first)?)?;
    assert_eq!(round_trip, first);

    let control = |id| {
        generate_meta_episode_shuffled_control(
            SEED,
            id,
            V5DataSplit::Train,
            &families(),
            &MetaEpisodeConfig::default(),
        )
    };
    assert_eq!(
        serde_json::to_vec(&control(11)?)?,
        serde_json::to_vec(&control(11)?)?
    );
    Ok(())
}

#[test]
fn true_meta_episode_keeps_one_stable_rule_across_changed_layouts() -> Result<()> {
    let episode = true_episode(3)?;
    episode.validate()?;
    let config = MetaEpisodeConfig::default();
    assert_eq!(episode.levels.len(), config.levels);
    assert!(!episode.shuffled_control);
    assert!(episode
        .levels
        .iter()
        .all(|level| level.operator == episode.operator));

    // Explicit boundaries index the flattened stream; no sample is mutated.
    let rows_per_level = config.steps_per_level + config.operator_decisions_per_level;
    assert_eq!(episode.level_boundaries, vec![0, 7, 14, 21]);
    assert_eq!(
        episode.flattened_transitions().count(),
        config.levels * rows_per_level
    );
    assert!(episode
        .levels
        .iter()
        .all(|level| level.episode_id & (1 << 63) != 0));
    assert_eq!(
        episode
            .levels
            .iter()
            .map(|level| level.episode_id)
            .collect::<BTreeSet<_>>()
            .len(),
        config.levels
    );

    // Layouts change across levels: each level starts from a distinct board.
    let starting_boards = episode
        .levels
        .iter()
        .map(|level| board(&level.transitions[0].current.pixels).to_vec())
        .collect::<BTreeSet<_>>();
    assert_eq!(starting_boards.len(), config.levels);

    // Every operator decision replays exactly under the shared rule.
    for level in &episode.levels {
        for transition in &level.transitions {
            if !matches!(transition.action.id, 5 | 6) {
                continue;
            }
            let rect = ContentRect {
                x: 0,
                y: 0,
                width: config.content_size,
                height: config.content_size,
            };
            let replayed = apply_episode_operator(
                &transition.current,
                &transition.action,
                rect,
                episode.operator,
            )?;
            assert_eq!(
                board(&replayed.pixels),
                board(&transition.next.pixels),
                "level {} action {} must replay under the shared rule",
                level.level_index,
                transition.action.id
            );
        }
    }
    Ok(())
}

#[test]
fn shuffled_control_shares_layouts_but_breaks_the_rule() -> Result<()> {
    let config = MetaEpisodeConfig::default();
    let truth = true_episode(5)?;
    let control =
        generate_meta_episode_shuffled_control(SEED, 5, V5DataSplit::Train, &families(), &config)?;
    control.validate()?;
    assert!(control.shuffled_control);
    assert_eq!(control.operator, truth.operator);
    // Level 0 is byte-identical: the control diverges only after the boundary.
    assert_eq!(control.levels[0], truth.levels[0]);
    for (true_level, control_level) in truth.levels.iter().zip(&control.levels).skip(1) {
        // Same layout and movement walk (operator-independent RNG lane) ...
        assert_eq!(
            &true_level.transitions[..config.steps_per_level],
            &control_level.transitions[..config.steps_per_level]
        );
        // ... while the later rule is an independent draw from the same train
        // marginal (a repeat is valid and carries no negative information).
        assert!(families().train.contains(&control_level.operator.family));
    }
    Ok(())
}

#[test]
fn census_counts_later_alternative_outcome_sensitivity_and_rule_free_zero() -> Result<()> {
    let families = families();
    let population = (0..8).map(true_episode).collect::<Result<Vec<_>>>()?;
    let census = census_rule_identifiability(&population, &families)?;
    println!("rule-identifiability census: {census:?}");
    assert_eq!(census.episodes, 8);
    assert_eq!(census.levels, 24);
    assert!(census.earlier_rule_identified_episodes > 0);
    // Default config: 3 decision points in each of 2 later levels, 8 episodes.
    assert_eq!(census.overall.later_operator_points, 48);
    assert!(
        census.overall.alternative_outcome_sensitive_fraction > 0.0,
        "identified earlier rules must distinguish some later fixed-action outcomes"
    );
    assert!(census
        .per_family
        .iter()
        .any(|bucket| bucket.counts.alternative_outcome_sensitive > 0));
    assert!(census
        .per_level
        .iter()
        .all(|bucket| bucket.level_index >= 1));
    let round_trip: tofy::p2::data::RuleIdentifiabilityCensus =
        serde_json::from_str(&serde_json::to_string(&census)?)?;
    assert_eq!(round_trip, census);

    // Rule-independent construction: later levels contain movement only, so no
    // decision depends on the hidden rule and the census reads exactly zero.
    let rule_free_config = MetaEpisodeConfig {
        operator_decisions_per_level: 0,
        ..MetaEpisodeConfig::default()
    };
    let rule_free = (0..4)
        .map(|id| generate_meta_episode(SEED, id, V5DataSplit::Train, &families, &rule_free_config))
        .collect::<Result<Vec<_>>>()?;
    let zero_census = census_rule_identifiability(&rule_free, &families)?;
    assert_eq!(zero_census.overall.later_operator_points, 0);
    assert_eq!(
        zero_census.overall.alternative_outcome_sensitive_fraction,
        0.0
    );
    assert!(zero_census.per_family.is_empty());
    assert!(zero_census.per_level.is_empty());

    let shuffled = generate_meta_episode_shuffled_control(
        SEED,
        99,
        V5DataSplit::Train,
        &families,
        &MetaEpisodeConfig::default(),
    )?;
    assert!(census_rule_identifiability(&[shuffled], &families).is_err());
    Ok(())
}

#[test]
fn shuffled_control_reports_realized_changed_and_outcome_changing_populations() -> Result<()> {
    let families = families();
    let config = MetaEpisodeConfig::default();
    let mut aggregate = tofy::p2::data::ShuffledRuleRealizationCensus::default();
    for id in 0..32 {
        let episode = generate_meta_episode_shuffled_control(
            SEED,
            id,
            V5DataSplit::Train,
            &families,
            &config,
        )?;
        let census = episode
            .shuffled_rule_realization_census()?
            .expect("shuffled episode has a realization census");
        aggregate.later_levels += census.later_levels;
        aggregate.repeated_level0_operator_levels += census.repeated_level0_operator_levels;
        aggregate.total_rows += census.total_rows;
        aggregate.eligible_operator_rows += census.eligible_operator_rows;
        aggregate.genuinely_changed_operator_tuples += census.genuinely_changed_operator_tuples;
        aggregate.outcome_changing_tuples += census.outcome_changing_tuples;
    }
    assert_eq!(aggregate.later_levels, 64);
    assert_eq!(aggregate.eligible_operator_rows, 32 * 2 * 3);
    assert!(aggregate.repeated_level0_operator_levels > 0);
    assert!(aggregate.genuinely_changed_operator_tuples > 0);
    assert!(aggregate.outcome_changing_tuples > 0);
    assert!(aggregate.outcome_changing_tuples <= aggregate.genuinely_changed_operator_tuples);
    Ok(())
}
