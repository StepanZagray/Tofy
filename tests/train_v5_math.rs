use candle_core::{Device, Tensor};
use tofy::p2::eval::GateSupportMetrics;
use tofy::p2::train::{
    foundation_v2_ep_weight_update, foundation_v2_gate_evaluation,
    foundation_v2_gate_history_aborts, foundation_v2_loss_weights_from_masks,
    foundation_v2_promotion_improved, foundation_v2_promotion_value,
    foundation_v2_wsd_learning_rate, split_ce_with_weighting, split_weighted_ce, PromotionMetric,
    SplitCeWeighting, TrainConfig,
};

#[test]
fn foundation_v2_accepts_the_required_default_batch() {
    let mut config = TrainConfig {
        physical_batch: 2_048,
        steps_per_lesson: 24_576,
        ..TrainConfig::default()
    };
    config.apply_foundation_v2_recipe();
    config.validate().unwrap();
}

#[test]
fn foundation_v2_rejects_the_reserved_gate_seed() {
    let mut config = TrainConfig {
        seed: 0xF0A2_DA7A_0000_0005,
        physical_batch: 2_048,
        steps_per_lesson: 24_576,
        ..TrainConfig::default()
    };
    config.apply_foundation_v2_recipe();
    assert!(config.validate().is_err());
}

#[test]
fn changed_pixel_weights_use_content_only_and_clamp_the_ratio() {
    let current = vec![0; 100];
    let mut target = current.clone();
    let mut content = vec![1; 100];
    target[..20].fill(1);
    content[90..].fill(0);
    let weights = foundation_v2_loss_weights_from_masks(&current, &target, &content).unwrap();
    assert_eq!(weights.content_pixels, 90);
    assert_eq!(weights.changed_pixels, 20);
    assert_eq!(weights.unchanged_pixels, 70);
    assert!((weights.changed_weight - 3.5).abs() < 1e-12);
    assert_eq!(weights.unchanged_weight, 1.0);

    let mut rare_target = vec![0; 100];
    rare_target[0] = 1;
    let rare = foundation_v2_loss_weights_from_masks(&[0; 100], &rare_target, &[1; 100]).unwrap();
    assert_eq!(rare.changed_weight, 64.0);

    let all_changed = foundation_v2_loss_weights_from_masks(&[0; 8], &[1; 8], &[1; 8]).unwrap();
    assert_eq!(all_changed.changed_weight, 1.0);
    let none_changed = foundation_v2_loss_weights_from_masks(&[0; 8], &[0; 8], &[1; 8]).unwrap();
    assert_eq!(none_changed.changed_weight, 64.0);
}

#[test]
fn ep_gradient_budget_update_targets_thirty_percent_and_clamps() {
    let adjusted = foundation_v2_ep_weight_update(0.01, 10.0, 1.0);
    assert!((adjusted - 0.03).abs() < 1e-12);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1e9, 1.0), 1e-4);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1e-9, 1.0), 0.1);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 0.0, 1.0), 0.1);
}

#[test]
fn wsd_schedule_matches_every_boundary() {
    let total = 24_576;
    let decay_steps = ((total as f64) * 0.15).ceil() as usize;
    let decay_start = total - decay_steps;
    assert_eq!(foundation_v2_wsd_learning_rate(0, total), 0.0);
    assert!((foundation_v2_wsd_learning_rate(250, total) - 5e-4).abs() < 1e-12);
    assert!((foundation_v2_wsd_learning_rate(500, total) - 1e-3).abs() < 1e-12);
    assert!((foundation_v2_wsd_learning_rate(decay_start, total) - 1e-3).abs() < 1e-12);
    assert!(foundation_v2_wsd_learning_rate(decay_start + 1, total) < 1e-3);
    assert!((foundation_v2_wsd_learning_rate(total, total) - 1e-4).abs() < 1e-12);
}

#[test]
fn short_wsd_schedules_reach_the_final_learning_rate() {
    for total in [1, 2, 16, 499, 500] {
        assert_eq!(foundation_v2_wsd_learning_rate(total, total), 1e-4);
        assert!(foundation_v2_wsd_learning_rate(0, total) <= 1e-3);
    }
}

fn metrics(shuffled_ratio: f64) -> GateSupportMetrics {
    GateSupportMetrics {
        samples: 512,
        population_fingerprint: "sha256:test".into(),
        content_mask_fingerprint: Some("sha256:masks".into()),
        evidence_class: "selection_only".into(),
        changed_transitions: 128,
        changed_pixels: 256,
        foreground_pixels: 512,
        improvement_fraction: Some(-25.0),
        shuffled_action_changed_pixel_ratio: Some(shuffled_ratio),
        shuffled_action_rows: 512,
        shuffled_action_eligible_rows: 512,
        shuffled_action_changed_tuples: 512,
        shuffled_action_outcome_changing_tuples: None,
        foreground_reconstruction_accuracy: Some(0.9),
        one_step_changed_exact: Some(0.5),
        one_step_full_exact: Some(0.2),
        one_step_raw_full_exact: Some(0.1),
        false_edit_rate: Some(0.01),
        padding_false_edit_rate: Some(0.0),
        raw_false_edit_rate: Some(0.02),
        raw_padding_false_edit_rate: Some(0.01),
        population_contract: "fixed test population".into(),
    }
}

#[test]
fn gates_pass_fail_and_abort_only_on_consecutive_failure() {
    let pass = foundation_v2_gate_evaluation(4_096, metrics(0.9), Some(0.6));
    assert!(pass.gates.iter().all(|gate| gate.passed));
    // Latent-MSE improvement is diagnostic-only: deeply negative yet passing.
    assert_eq!(pass.gates[0].name, "positive_improvement");
    assert_eq!(pass.gates[0].measured, Some(-25.0));

    let first_fail = foundation_v2_gate_evaluation(5_120, metrics(0.97), Some(0.6));
    assert!(first_fail.gates[0].passed);
    assert!(!first_fail.gates[1].passed);
    assert!(!foundation_v2_gate_history_aborts(&[first_fail.clone()]));

    let recovery = foundation_v2_gate_evaluation(6_144, metrics(0.9), Some(0.6));
    assert!(!foundation_v2_gate_history_aborts(&[
        first_fail.clone(),
        recovery.clone()
    ]));

    let second_fail = foundation_v2_gate_evaluation(7_168, metrics(0.98), Some(0.6));
    assert!(!foundation_v2_gate_history_aborts(&[
        first_fail.clone(),
        recovery,
        second_fail.clone()
    ]));
    let consecutive = foundation_v2_gate_evaluation(8_192, metrics(0.99), Some(0.6));
    assert!(foundation_v2_gate_history_aborts(&[
        first_fail,
        second_fail,
        consecutive
    ]));
}

/// Six pixels on a 2x3 grid: changed {0.5, 2.5}, unchanged {1.5, 3.5, 0.25},
/// and one non-content (PAD) pixel. p = 2/5, so (1-p)/p = 1.5 (below the cap).
fn split_ce_fixture(device: &Device) -> (Tensor, Tensor, Tensor) {
    let per_pixel =
        Tensor::from_vec(vec![0.5f32, 1.5, 2.5, 3.5, 0.25, 0.75], (2, 3), device).unwrap();
    let changed = Tensor::from_vec(vec![1f32, 0.0, 1.0, 0.0, 0.0, 0.0], (2, 3), device).unwrap();
    let unchanged = Tensor::from_vec(vec![0f32, 1.0, 0.0, 1.0, 1.0, 0.0], (2, 3), device).unwrap();
    (per_pixel, changed, unchanged)
}

fn fixture_changed_weights() -> tofy::p2::train::ChangedPixelWeights {
    foundation_v2_loss_weights_from_masks(
        &[0, 0, 0, 0, 0, 0],
        &[1, 0, 1, 0, 0, 9],
        &[1, 1, 1, 1, 1, 0],
    )
    .unwrap()
}

#[test]
fn current_double_mode_is_bit_identical_to_split_weighted_ce() {
    let device = Device::Cpu;
    let (per_pixel, changed, unchanged) = split_ce_fixture(&device);
    let weights = fixture_changed_weights();
    assert!((weights.changed_weight - 1.5).abs() < 1e-12);
    let legacy = split_weighted_ce(
        &per_pixel,
        &changed,
        &unchanged,
        2,
        3,
        weights.changed_weight,
    )
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    let moded = split_ce_with_weighting(
        &per_pixel,
        &changed,
        &unchanged,
        2,
        3,
        weights.changed_weight,
        SplitCeWeighting::CurrentDouble,
        None,
    )
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert_eq!(legacy.to_bits(), moded.to_bits());
    // 1.5 * mean{0.5, 2.5} + mean{1.5, 3.5, 0.25} = 1.5 * 1.5 + 1.75.
    assert!((moded - 4.0).abs() < 1e-6);
}

#[test]
fn equal_means_mode_assigns_equal_shares_of_legacy_coefficient_mass() {
    let device = Device::Cpu;
    let (per_pixel, changed, unchanged) = split_ce_fixture(&device);
    let value = split_ce_with_weighting(
        &per_pixel,
        &changed,
        &unchanged,
        2,
        3,
        fixture_changed_weights().changed_weight,
        SplitCeWeighting::EqualMeans,
        None,
    )
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    // Equal geometry is normalized to the legacy coefficient mass 1.5+1=2.5:
    // 1.25 * (mean{0.5,2.5} + mean{1.5,3.5,0.25}) = 4.0625.
    assert!((value - 4.0625).abs() < 1e-6);
}

#[test]
fn pooled_mode_matches_hand_computed_pool_and_single_ratio() {
    let device = Device::Cpu;
    let (per_pixel, changed, unchanged) = split_ce_fixture(&device);
    let weights = fixture_changed_weights();
    let pooled = |values: &Tensor| {
        split_ce_with_weighting(
            values,
            &changed,
            &unchanged,
            2,
            3,
            weights.changed_weight,
            SplitCeWeighting::PooledPerPixel,
            None,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
    };
    // The pooled construction has equal changed/unchanged aggregate shares in
    // this fixture, then is normalized to the legacy coefficient mass 2.5.
    assert!((pooled(&per_pixel) - 4.0625).abs() < 1e-6);
    // Per-pixel coefficients via indicators: each of the 2 changed pixels gets
    // w/5, one unchanged pixel gets 1/5; ratio is (1-p)/p once, not squared.
    let changed_total = pooled(&changed);
    let one_unchanged =
        Tensor::from_vec(vec![0f32, 1.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &device).unwrap();
    let per_changed_coefficient = changed_total / 2.0;
    let per_unchanged_coefficient = pooled(&one_unchanged);
    let single_ratio = (1.0 - weights.changed_fraction) / weights.changed_fraction;
    assert!(
        (f64::from(per_changed_coefficient / per_unchanged_coefficient) - single_ratio).abs()
            < 1e-6
    );
}

#[test]
fn changed_budget_hits_the_aggregate_share_in_both_constructions() {
    let device = Device::Cpu;
    let (_, changed, unchanged) = split_ce_fixture(&device);
    let ones = Tensor::ones((2, 3), candle_core::DType::F32, &device).unwrap();
    let budget = 0.75;
    let run = |values: &Tensor, mode: SplitCeWeighting| {
        split_ce_with_weighting(values, &changed, &unchanged, 2, 3, 1.5, mode, Some(budget))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    // Every construction preserves the legacy coefficient mass 1.5+1=2.5;
    // the budget redistributes 75% of it to the changed stratum.
    let stratum_total = run(&ones, SplitCeWeighting::CurrentDouble);
    let stratum_changed = run(&changed, SplitCeWeighting::CurrentDouble);
    assert!((stratum_total - 2.5).abs() < 1e-6);
    assert!((f64::from(stratum_changed / stratum_total) - budget).abs() < 1e-6);
    assert!((run(&ones, SplitCeWeighting::EqualMeans) - stratum_total).abs() < 1e-6);
    // Pooled mode has the same coefficient mass and requested share.
    let pooled_total = run(&ones, SplitCeWeighting::PooledPerPixel);
    let pooled_changed = run(&changed, SplitCeWeighting::PooledPerPixel);
    assert!((pooled_total - 2.5).abs() < 1e-6);
    assert!((f64::from(pooled_changed / pooled_total) - budget).abs() < 1e-6);
}

#[test]
fn split_ce_empty_strata_fall_back_to_the_legacy_observed_mass() {
    let device = Device::Cpu;
    let values = Tensor::from_vec(vec![2.0f32, 4.0], 2, &device).unwrap();
    let all = Tensor::ones(2, candle_core::DType::F32, &device).unwrap();
    let none = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
    for mode in [
        SplitCeWeighting::CurrentDouble,
        SplitCeWeighting::EqualMeans,
        SplitCeWeighting::PooledPerPixel,
    ] {
        let positive_only =
            split_ce_with_weighting(&values, &all, &none, 2, 0, 1.5, mode, Some(0.75))
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
        let negative_only =
            split_ce_with_weighting(&values, &none, &all, 0, 2, 1.5, mode, Some(0.75))
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
        assert!((positive_only - 4.5).abs() < 1e-6);
        assert!((negative_only - 3.0).abs() < 1e-6);
    }
}

#[test]
fn configs_without_split_ce_fields_default_to_current_double() {
    let mut value = serde_json::to_value(TrainConfig::default()).unwrap();
    let map = value.as_object_mut().unwrap();
    assert!(map.remove("split_ce_weighting").is_some());
    assert!(map.remove("split_ce_changed_budget").is_some());
    let config: TrainConfig = serde_json::from_value(value).unwrap();
    assert_eq!(config.split_ce_weighting, SplitCeWeighting::CurrentDouble);
    assert_eq!(config.split_ce_changed_budget, None);
}

#[test]
fn split_ce_budget_validation_is_strict_and_recipe_keeps_the_knob_caller_owned() {
    let mut config = TrainConfig {
        physical_batch: 2_048,
        steps_per_lesson: 24_576,
        ..TrainConfig::default()
    };
    config.apply_foundation_v2_recipe();
    config.split_ce_weighting = SplitCeWeighting::PooledPerPixel;
    config.split_ce_changed_budget = Some(0.3);
    config.validate().unwrap();
    for invalid in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        config.split_ce_changed_budget = Some(invalid);
        assert!(config.validate().is_err());
    }
}

#[test]
fn absolute_gates_warm_up_and_foreground_enforces_at_8192() {
    // Before 4096: shuffled-action gate passes by warmup fiat even when awful.
    let early = foundation_v2_gate_evaluation(1_024, metrics(0.99), None);
    assert!(early.gates[1].passed);
    // 4096..8192: foreground below the floor still passes (warmup).
    let mut mid_metrics = metrics(0.9);
    mid_metrics.foreground_reconstruction_accuracy = Some(0.55);
    let mid = foundation_v2_gate_evaluation(5_120, mid_metrics, Some(0.6));
    assert!(mid.gates[2].passed);
    // From 8192 the collapse floor is enforced: the observed asymptote
    // (~0.67) passes, genuine regression below 0.60 fails.
    let mut ok_metrics = metrics(0.9);
    ok_metrics.foreground_reconstruction_accuracy = Some(0.67);
    let ok = foundation_v2_gate_evaluation(8_192, ok_metrics, Some(0.6));
    assert!(ok.gates[2].passed);
    let mut late_metrics = metrics(0.9);
    late_metrics.foreground_reconstruction_accuracy = Some(0.55);
    let late = foundation_v2_gate_evaluation(8_192, late_metrics, Some(0.6));
    assert!(!late.gates[2].passed);
}

#[test]
fn promotion_default_matches_the_historical_changed_exact_rule() {
    // Synthetic gate-history sequence: bests 0.4 -> 0.6 on changed-exact.
    let history = vec![
        foundation_v2_gate_evaluation(
            4_096,
            {
                let mut first = metrics(0.9);
                first.one_step_changed_exact = Some(0.4);
                first.one_step_full_exact = Some(0.35);
                first
            },
            None,
        ),
        foundation_v2_gate_evaluation(
            5_120,
            {
                let mut second = metrics(0.9);
                second.one_step_changed_exact = Some(0.6);
                second.one_step_full_exact = Some(0.1);
                second
            },
            Some(0.4),
        ),
    ];
    let best_changed = Some(0.6);
    for (changed, expected) in [(0.59, false), (0.6, false), (0.61, true)] {
        let mut candidate = metrics(0.9);
        candidate.one_step_changed_exact = Some(changed);
        // The default rule ignores full-exact entirely, even when it soars.
        candidate.one_step_full_exact = Some(0.99);
        assert_eq!(
            foundation_v2_promotion_improved(
                PromotionMetric::default(),
                best_changed,
                &history,
                &candidate,
            ),
            expected,
        );
        // Byte-identical to the historical inline expression.
        assert_eq!(
            expected,
            candidate
                .one_step_changed_exact
                .is_some_and(|current| best_changed.is_none_or(|best| current > best)),
        );
    }
    // Metric absent: never promotes, exactly like the historical rule.
    let mut absent = metrics(0.9);
    absent.one_step_changed_exact = None;
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::ChangedExact,
        best_changed,
        &history,
        &absent,
    ));
}

#[test]
fn promotion_metric_full_exact_switches_selection_to_the_full_frame_best() {
    let history = vec![
        foundation_v2_gate_evaluation(
            4_096,
            {
                let mut first = metrics(0.9);
                first.one_step_changed_exact = Some(0.4);
                first.one_step_full_exact = Some(0.35);
                first
            },
            None,
        ),
        foundation_v2_gate_evaluation(
            5_120,
            {
                let mut second = metrics(0.9);
                second.one_step_changed_exact = Some(0.6);
                second.one_step_full_exact = Some(0.1);
                second
            },
            Some(0.4),
        ),
    ];
    // Worse on changed-exact, better on full-exact than the history best 0.35.
    let mut candidate = metrics(0.9);
    candidate.one_step_changed_exact = Some(0.5);
    candidate.one_step_full_exact = Some(0.36);
    assert_eq!(
        foundation_v2_promotion_value(PromotionMetric::FullExact, &candidate),
        Some(0.36),
    );
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::ChangedExact,
        Some(0.6),
        &history,
        &candidate,
    ));
    assert!(foundation_v2_promotion_improved(
        PromotionMetric::FullExact,
        Some(0.6),
        &history,
        &candidate,
    ));
    // Below the history's full-exact best: not promoted under FullExact.
    candidate.one_step_full_exact = Some(0.35);
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::FullExact,
        Some(0.6),
        &history,
        &candidate,
    ));
    // Empty history promotes the first full-exact measurement.
    assert!(foundation_v2_promotion_improved(
        PromotionMetric::FullExact,
        Some(0.6),
        &[],
        &candidate,
    ));
}

#[test]
fn configs_without_promotion_metric_default_to_changed_exact() {
    let mut value = serde_json::to_value(TrainConfig::default()).unwrap();
    let map = value.as_object_mut().unwrap();
    assert!(map.remove("promotion_metric").is_some());
    let config: TrainConfig = serde_json::from_value(value).unwrap();
    assert_eq!(config.promotion_metric, PromotionMetric::ChangedExact);
}
