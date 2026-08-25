use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tofy::domain::Split;
use tofy::p2::data::{
    palette, ArcAction, ArcFrame, GoalFeatures, TransitionProvenance, TransitionSample, FRAME_SIDE,
};
use tofy::p2::eval::{board_changed_transition_count, evaluate_gate_support};
use tofy::p2::model::WorldModel;
use tofy::p2::semantic_eval::{
    action_controllability_probe, ambiguity_ceiling, shuffled_action_control_samples,
};
use tofy::p2::train::{reinit_varmap_deterministic, TrainConfig};

fn sample(
    source: &str,
    trajectory: &str,
    transition_index: u64,
    current_marker: u8,
    next_marker: u8,
    action: u8,
    noop: Option<bool>,
) -> Result<TransitionSample> {
    let mut current = vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE];
    let mut next = current.clone();
    current[0] = current_marker;
    next[0] = next_marker;
    Ok(TransitionSample {
        current: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current)?,
        next: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next)?,
        action: ArcAction::new(action, None, None)?,
        goal_features: GoalFeatures::zeros(),
        noop,
        goal_satisfied: None,
        goal_failed: None,
        exhausted: None,
        split: Split::HeldOutComposition,
        family: source.into(),
        seed: 17,
        episode_id: 3,
        transition_index,
        provenance: TransitionProvenance {
            content_width: 7,
            content_height: 7,
            source_kind: source.into(),
            trajectory_id: trajectory.into(),
        },
        oracle_latent: None,
    })
}

#[test]
fn noop_stratification_counts_only_board_changed_transitions() -> Result<()> {
    let mut status_only = sample("a", "a/0", 0, 1, 1, 1, Some(true))?;
    status_only.next.pixels[(FRAME_SIDE - 1) * FRAME_SIDE] = 7;
    let board_changed = sample("a", "a/1", 0, 1, 2, 2, Some(false))?;
    let unknown = sample("a", "a/2", 0, 1, 2, 3, None)?;

    assert_eq!(
        board_changed_transition_count(&[status_only, board_changed, unknown]),
        1
    );
    Ok(())
}

#[test]
fn shuffled_control_never_crosses_source_kind() -> Result<()> {
    let samples = vec![
        sample("alpha", "alpha/0", 0, 0, 1, 1, Some(false))?,
        sample("beta", "beta/0", 0, 0, 1, 3, Some(false))?,
        sample("alpha", "alpha/1", 0, 0, 1, 2, Some(false))?,
        sample("beta", "beta/1", 0, 0, 1, 4, Some(false))?,
    ];

    let shuffled = shuffled_action_control_samples(&samples);
    assert_eq!(shuffled[0].action.id, 2);
    assert_eq!(shuffled[2].action.id, 1);
    assert_eq!(shuffled[1].action.id, 4);
    assert_eq!(shuffled[3].action.id, 3);
    for (before, after) in samples.iter().zip(&shuffled) {
        assert_eq!(before.provenance.source_kind, after.provenance.source_kind);
    }
    Ok(())
}

#[test]
fn action_controllability_distinguishes_blind_and_sensitive_predictors() -> Result<()> {
    let actions = vec![vec![
        ArcAction::new(1, None, None)?,
        ArcAction::new(2, None, None)?,
        ArcAction::new(3, None, None)?,
    ]];
    let blind = action_controllability_probe(&actions, 0.1, |_state, _action| Ok(vec![1.0, -1.0]))?;
    assert_eq!(blind.mean_pairwise_latent_distance, Some(0.0));
    assert_eq!(blind.fraction_states_above_threshold, Some(0.0));

    let sensitive = action_controllability_probe(&actions, 0.1, |_state, action| {
        Ok(vec![f32::from(action.id), 0.0])
    })?;
    assert!(sensitive.mean_pairwise_latent_distance.unwrap() > 0.0);
    assert_eq!(sensitive.fraction_states_above_threshold, Some(1.0));
    Ok(())
}

#[test]
fn ambiguity_ceiling_reports_known_history_collision_rates() -> Result<()> {
    let rows = vec![
        sample("family", "trajectory/one", 0, 1, 3, 2, Some(false))?,
        sample("family", "trajectory/one", 1, 3, 4, 1, Some(false))?,
        sample("family", "trajectory/two", 0, 2, 3, 2, Some(false))?,
        sample("family", "trajectory/two", 1, 3, 5, 1, Some(false))?,
    ];

    let ceiling = ambiguity_ceiling(&rows);
    assert_eq!(ceiling.history_1.groups, 3);
    assert_eq!(ceiling.history_1.ambiguous_groups, 1);
    assert_eq!(ceiling.history_1.ambiguous_group_fraction, Some(1.0 / 3.0));
    assert_eq!(ceiling.history_2.groups, 2);
    assert_eq!(ceiling.history_2.ambiguous_groups, 0);
    assert_eq!(ceiling.history_2.ambiguous_group_fraction, Some(0.0));
    Ok(())
}

#[test]
fn gate_support_is_deterministic_across_calls() -> Result<()> {
    let device = Device::Cpu;
    let mut config = TrainConfig::default();
    config.apply_full_v4_recipe();
    config.hidden_dim = 16;
    config.action_dim = 4;
    config.inner_steps = 1;
    config.outer_steps = 1;
    let varmap = VarMap::new();
    let model = WorldModel::new(
        config.model_config(),
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )?;
    reinit_varmap_deterministic(&varmap, config.seed)?;
    let rows = vec![
        sample("alpha", "alpha/0", 0, 1, 2, 1, Some(false))?,
        sample("alpha", "alpha/1", 0, 2, 3, 2, Some(false))?,
    ];

    let first = evaluate_gate_support(&model, &rows, &device)?;
    let second = evaluate_gate_support(&model, &rows, &device)?;
    assert_eq!(first, second);
    Ok(())
}
