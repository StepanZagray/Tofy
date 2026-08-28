use candle_core::{Device, Tensor};
use tofy::p2::eval::GateSupportMetrics;
use tofy::p2::train::{
    foundation_v2_candidate_improves, foundation_v2_ep_weight_update,
    foundation_v2_evaluation_improves, foundation_v2_gate_evaluation,
    foundation_v2_gate_history_aborts, foundation_v2_loss_weights_from_masks,
    foundation_v2_named_gate_passed, foundation_v2_promotion_improved,
    foundation_v2_promotion_value, foundation_v2_selected_best_step,
    foundation_v2_wsd_learning_rate, separation_hinge_term, split_ce_with_weighting,
    split_weighted_ce, FoundationV2GateEvaluation, PromotionMetric, SplitCeWeighting, TrainConfig,
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
    // A budget-required weight below the floor disables EP instead of holding
    // a floor that would violate the `<= 0.3x` bound (ADR 0003 §3.6).
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1e9, 1.0), 0.0);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1.0, 0.0), 0.0);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1.0, 1e-6), 0.0);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 1e-9, 1.0), 0.1);
    assert_eq!(foundation_v2_ep_weight_update(0.01, 0.0, 1.0), 0.1);
}

#[test]
fn ep_weight_always_satisfies_the_gradient_budget() {
    for ep in [0.0, 1e-9, 1e-3, 1.0, 1e3, 1e9] {
        for pred in [0.0, 1e-9, 1e-3, 1.0, 1e3, 1e9] {
            let weight = foundation_v2_ep_weight_update(0.01, ep, pred);
            assert!(weight.is_finite() && weight >= 0.0);
            if ep > 0.0 {
                assert!(
                    weight * ep <= 0.3 * pred + 1e-12,
                    "budget violated: w={weight} ep={ep} pred={pred}"
                );
            }
        }
    }
}

#[test]
fn separation_hinge_uses_reachable_l2_distance_with_finite_gradients() {
    use candle_core::{Device, Tensor, Var};
    let device = Device::Cpu;
    // Opposite 128-dimensional unit vectors: L2 distance 2 >= margin 0.3.
    let mut unit = vec![0f32; 128];
    unit[0] = 1.0;
    let left = Tensor::from_vec(unit.clone(), (1, 128), &device).unwrap();
    let right = left.affine(-1.0, 0.0).unwrap();
    let hinge = separation_hinge_term(&left, &right, 0.3).unwrap();
    assert!(hinge.to_scalar::<f32>().unwrap().abs() < 1e-5);
    // Equal vectors: full margin, and the backward pass stays finite (a bare
    // sqrt(0) would produce a non-finite gradient exactly at collapse).
    let var = Var::from_tensor(&left).unwrap();
    let equal_hinge = separation_hinge_term(&var, &left, 0.3).unwrap();
    let value = equal_hinge.to_scalar::<f32>().unwrap();
    assert!((value - 0.3).abs() < 1e-5);
    let grads = equal_hinge.backward().unwrap();
    let grad = grads.get(&var).expect("gradient for left displacement");
    for value in grad.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
        assert!(value.is_finite());
    }
    // The old RMS distance capped at 2/sqrt(128) < 0.3; the L2 form reaches
    // the boundary: distance exactly 0.3 yields a zero hinge.
    let mut offset = vec![0f32; 128];
    offset[0] = 1.0 - 0.3;
    let boundary = Tensor::from_vec(offset, (1, 128), &device).unwrap();
    let boundary_hinge = separation_hinge_term(&left, &boundary, 0.3).unwrap();
    assert!(boundary_hinge.to_scalar::<f32>().unwrap().abs() < 1e-5);
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
        one_step_composed_changed_exact: Some(0.45),
        one_step_all_rows_exact: Some(0.6),
        false_edit_rate: Some(0.01),
        padding_false_edit_rate: Some(0.0),
        raw_false_edit_rate: Some(0.02),
        raw_padding_false_edit_rate: Some(0.01),
        population_contract: "fixed test population".into(),
    }
}

fn gate(evaluation: &FoundationV2GateEvaluation, name: &str) -> bool {
    evaluation
        .gates
        .iter()
        .find(|gate| gate.name == name)
        .unwrap_or_else(|| panic!("missing gate {name}"))
        .passed
}

#[test]
fn gates_pass_fail_and_abort_only_on_consecutive_failure() {
    let pass = foundation_v2_gate_evaluation(4_096, metrics(0.9), Some(0.6), Some(0.45));
    assert!(pass.gates.iter().all(|gate| gate.passed));
    // Latent-MSE improvement is diagnostic-only, never a passing gate.
    assert_eq!(pass.diagnostics[0].name, "positive_improvement");
    assert_eq!(pass.diagnostics[0].measured, Some(-25.0));

    let first_fail = foundation_v2_gate_evaluation(5_120, metrics(0.97), Some(0.6), Some(0.45));
    assert!(!gate(&first_fail, "shuffled_action_ratio"));
    assert!(!foundation_v2_gate_history_aborts(&[first_fail.clone()]));

    let recovery = foundation_v2_gate_evaluation(6_144, metrics(0.9), Some(0.6), Some(0.45));
    assert!(!foundation_v2_gate_history_aborts(&[
        first_fail.clone(),
        recovery.clone()
    ]));

    let second_fail = foundation_v2_gate_evaluation(7_168, metrics(0.98), Some(0.6), Some(0.45));
    assert!(!foundation_v2_gate_history_aborts(&[
        first_fail.clone(),
        recovery,
        second_fail.clone()
    ]));
    let consecutive = foundation_v2_gate_evaluation(8_192, metrics(0.99), Some(0.6), Some(0.45));
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
    let early = foundation_v2_gate_evaluation(1_024, metrics(0.99), None, None);
    assert!(gate(&early, "shuffled_action_ratio"));
    // 4096..8192: foreground below the floor still passes (warmup).
    let mut mid_metrics = metrics(0.9);
    mid_metrics.foreground_reconstruction_accuracy = Some(0.55);
    let mid = foundation_v2_gate_evaluation(5_120, mid_metrics, Some(0.6), Some(0.45));
    assert!(gate(&mid, "foreground_reconstruction"));
    // From 8192 the collapse floor is enforced: the observed asymptote
    // (~0.67) passes, genuine regression below 0.60 fails.
    let mut ok_metrics = metrics(0.9);
    ok_metrics.foreground_reconstruction_accuracy = Some(0.67);
    let ok = foundation_v2_gate_evaluation(8_192, ok_metrics, Some(0.6), Some(0.45));
    assert!(gate(&ok, "foreground_reconstruction"));
    let mut late_metrics = metrics(0.9);
    late_metrics.foreground_reconstruction_accuracy = Some(0.55);
    let late = foundation_v2_gate_evaluation(8_192, late_metrics, Some(0.6), Some(0.45));
    assert!(!gate(&late, "foreground_reconstruction"));
}

#[test]
fn composed_changed_exact_gate_arms_at_4096_and_tracks_its_own_best() {
    let mut copy_only = metrics(0.9);
    copy_only.one_step_composed_changed_exact = Some(0.0);
    let warmup = foundation_v2_gate_evaluation(3_072, copy_only.clone(), Some(0.6), None);
    assert!(gate(&warmup, "composed_changed_exact_collapse"));

    let armed = foundation_v2_gate_evaluation(4_096, copy_only, Some(0.6), None);
    assert!(!gate(&armed, "composed_changed_exact_collapse"));

    let mut collapsed = metrics(0.9);
    collapsed.one_step_composed_changed_exact = Some(0.39);
    let collapsed = foundation_v2_gate_evaluation(5_120, collapsed, Some(0.6), Some(0.5));
    assert!(!gate(&collapsed, "composed_changed_exact_collapse"));

    let mut retained = metrics(0.9);
    retained.one_step_composed_changed_exact = Some(0.4);
    let retained = foundation_v2_gate_evaluation(5_120, retained, Some(0.6), Some(0.5));
    assert!(gate(&retained, "composed_changed_exact_collapse"));
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
            Some(0.45),
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
            Some(0.45),
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
fn composed_exact_guarded_rejects_false_edit_regressions() {
    let incumbent = foundation_v2_gate_evaluation(
        4_096,
        {
            let mut best = metrics(0.9);
            best.one_step_all_rows_exact = Some(0.5);
            best.false_edit_rate = Some(0.01);
            best.padding_false_edit_rate = Some(0.0);
            best
        },
        None,
        None,
    );
    let history = vec![incumbent];

    // Equal composed changed-exact uses all-row exactness as its tiebreak.
    let mut clean = metrics(0.9);
    clean.one_step_all_rows_exact = Some(0.55);
    clean.false_edit_rate = Some(0.01);
    clean.padding_false_edit_rate = Some(0.0);
    assert!(foundation_v2_promotion_improved(
        PromotionMetric::ComposedExactGuarded,
        None,
        &history,
        &clean,
    ));

    // The same exactness gain with a regressed content false-edit rate loses.
    let mut hallucinating = clean.clone();
    hallucinating.false_edit_rate = Some(0.05);
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::ComposedExactGuarded,
        None,
        &history,
        &hallucinating,
    ));

    // A missing candidate rate against a measured incumbent fails closed.
    let mut unmeasured = clean.clone();
    unmeasured.false_edit_rate = None;
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::ComposedExactGuarded,
        None,
        &history,
        &unmeasured,
    ));

    // The raw-decoder checkpoint with a closed copy gate cannot outrank the
    // composed-correct one: selection reads composed metrics only.
    let mut raw_only = metrics(0.9);
    raw_only.one_step_changed_exact = Some(0.9);
    raw_only.one_step_all_rows_exact = Some(0.4);
    assert!(!foundation_v2_promotion_improved(
        PromotionMetric::ComposedExactGuarded,
        None,
        &history,
        &raw_only,
    ));
}

#[test]
fn copy_only_composed_checkpoint_cannot_freeze_the_first_useful_edit() {
    let mut copy_only = metrics(0.9);
    copy_only.one_step_composed_changed_exact = Some(0.0);
    copy_only.one_step_all_rows_exact = Some(0.30);
    copy_only.false_edit_rate = Some(0.00045);
    copy_only.padding_false_edit_rate = Some(0.0);
    let incumbent = copy_only.clone();
    let history = vec![foundation_v2_gate_evaluation(
        4_096,
        copy_only,
        Some(0.5),
        None,
    )];

    let mut useful = metrics(0.9);
    useful.one_step_composed_changed_exact = Some(0.01);
    useful.one_step_all_rows_exact = Some(0.20);
    useful.false_edit_rate = Some(0.001);
    useful.padding_false_edit_rate = Some(0.0005);
    assert_eq!(
        foundation_v2_promotion_value(PromotionMetric::ComposedExactGuarded, &useful),
        Some(0.01),
    );
    assert!(foundation_v2_candidate_improves(
        PromotionMetric::ComposedExactGuarded,
        Some(&incumbent),
        &useful,
    ));
    assert!(foundation_v2_promotion_improved(
        PromotionMetric::ComposedExactGuarded,
        None,
        &history,
        &useful,
    ));
}

#[test]
fn promotion_is_blocked_when_an_armed_gate_fails() {
    let incumbent = foundation_v2_gate_evaluation(4_096, metrics(0.9), Some(0.5), Some(0.45));
    let history = vec![incumbent];
    let mut candidate_metrics = metrics(0.99);
    candidate_metrics.one_step_changed_exact = Some(0.9);
    candidate_metrics.one_step_composed_changed_exact = Some(0.9);
    let candidate = foundation_v2_gate_evaluation(5_120, candidate_metrics, Some(0.5), Some(0.45));
    assert!(!gate(&candidate, "shuffled_action_ratio"));
    assert!(!foundation_v2_evaluation_improves(
        PromotionMetric::ChangedExact,
        Some(0.5),
        &history,
        &candidate,
    ));
}

#[test]
fn named_gate_lookup_is_independent_of_vector_order() {
    let mut evaluation = foundation_v2_gate_evaluation(5_120, metrics(0.99), Some(0.6), Some(0.45));
    evaluation.gates.reverse();
    assert!(foundation_v2_named_gate_passed(
        &evaluation,
        "one_step_collapse"
    ));
    assert!(!foundation_v2_named_gate_passed(
        &evaluation,
        "shuffled_action_ratio"
    ));
    assert!(!foundation_v2_named_gate_passed(&evaluation, "missing"));
}

#[test]
fn old_gate_history_with_positive_improvement_gate_deserializes() {
    let evaluation = foundation_v2_gate_evaluation(4_096, metrics(0.9), Some(0.5), Some(0.45));
    let mut old = serde_json::to_value(evaluation).unwrap();
    let object = old.as_object_mut().unwrap();
    object.remove("diagnostics");
    let gates = object
        .get_mut("gates")
        .and_then(serde_json::Value::as_array_mut)
        .unwrap();
    gates.retain(|gate| {
        gate.get("name").and_then(serde_json::Value::as_str)
            != Some("composed_changed_exact_collapse")
    });
    gates.insert(
        0,
        serde_json::json!({
            "name": "positive_improvement",
            "passed": true,
            "measured": -25.0,
            "threshold": "diagnostic-only (latent-MSE; superseded by pixel-space gates)"
        }),
    );

    let restored: FoundationV2GateEvaluation = serde_json::from_value(old).unwrap();
    assert!(restored.diagnostics.is_empty());
    assert!(foundation_v2_named_gate_passed(
        &restored,
        "positive_improvement"
    ));
    assert_eq!(
        foundation_v2_selected_best_step(PromotionMetric::ChangedExact, &[restored]),
        Some(4_096),
    );
}

#[test]
fn selected_best_step_replays_the_promotion_scan() {
    let mut first = metrics(0.9);
    first.one_step_changed_exact = Some(0.4);
    let mut second = metrics(0.9);
    second.one_step_changed_exact = Some(0.6);
    let mut third = metrics(0.9);
    third.one_step_changed_exact = Some(0.6); // tie: not a strict improvement
    let history = vec![
        foundation_v2_gate_evaluation(1_024, first, None, None),
        foundation_v2_gate_evaluation(2_048, second, Some(0.4), Some(0.45)),
        foundation_v2_gate_evaluation(3_072, third, Some(0.6), Some(0.45)),
    ];
    assert_eq!(
        foundation_v2_selected_best_step(PromotionMetric::ChangedExact, &history),
        Some(2_048),
    );
    assert_eq!(
        foundation_v2_selected_best_step(PromotionMetric::ChangedExact, &[]),
        None,
    );
}

#[test]
fn configs_without_promotion_metric_default_to_changed_exact() {
    let mut value = serde_json::to_value(TrainConfig::default()).unwrap();
    let map = value.as_object_mut().unwrap();
    assert!(map.remove("promotion_metric").is_some());
    let config: TrainConfig = serde_json::from_value(value).unwrap();
    assert_eq!(config.promotion_metric, PromotionMetric::ChangedExact);
}
