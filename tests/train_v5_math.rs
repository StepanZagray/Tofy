use tofy::p2::eval::GateSupportMetrics;
use tofy::p2::train::{
    foundation_v2_ep_weight_update, foundation_v2_gate_evaluation,
    foundation_v2_gate_history_aborts, foundation_v2_loss_weights_from_masks,
    foundation_v2_wsd_learning_rate, TrainConfig,
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

fn metrics(shuffled_ratio: f64) -> GateSupportMetrics {
    GateSupportMetrics {
        samples: 512,
        changed_transitions: 128,
        changed_pixels: 256,
        foreground_pixels: 512,
        improvement_fraction: Some(-25.0),
        shuffled_action_changed_pixel_ratio: Some(shuffled_ratio),
        foreground_reconstruction_accuracy: Some(0.9),
        one_step_changed_exact: Some(0.5),
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

#[test]
fn absolute_gates_warm_up_and_foreground_enforces_at_8192() {
    // Before 4096: shuffled-action gate passes by warmup fiat even when awful.
    let early = foundation_v2_gate_evaluation(1_024, metrics(0.99), None);
    assert!(early.gates[1].passed);
    // 4096..8192: foreground below threshold still passes (warmup).
    let mut mid_metrics = metrics(0.9);
    mid_metrics.foreground_reconstruction_accuracy = Some(0.63);
    let mid = foundation_v2_gate_evaluation(5_120, mid_metrics, Some(0.6));
    assert!(mid.gates[2].passed);
    // From 8192 the foreground threshold is enforced.
    let mut late_metrics = metrics(0.9);
    late_metrics.foreground_reconstruction_accuracy = Some(0.63);
    let late = foundation_v2_gate_evaluation(8_192, late_metrics, Some(0.6));
    assert!(!late.gates[2].passed);
}
