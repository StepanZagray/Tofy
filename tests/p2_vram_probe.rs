//! CUDA VRAM high-water probe for Phase B gates. Ignored unless `TOFY_VRAM_PROBE=1`.

use tofy::p2::train::{train, TrainConfig};

#[test]
#[ignore = "requires CUDA and TOFY_VRAM_PROBE=1"]
fn cuda_vram_high_water_slope() {
    if std::env::var("TOFY_VRAM_PROBE").ok().as_deref() != Some("1") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("tofy-vram-probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = TrainConfig {
        seed: 7,
        lessons: vec!["dynamics".into()],
        steps_per_lesson: 64,
        physical_batch: 256,
        grad_accum: 2,
        outer_steps: 8,
        inner_steps: 2,
        randomize_depth: false,
        device: "cuda".into(),
        output_dir: dir.clone(),
        checkpoint_every_steps: 0,
        max_steps_this_run: Some(64),
        shuffled_episodes: false,
        ..Default::default()
    };
    let report = train(&cfg).expect("vram probe train");
    assert_eq!(report.grad_accum, 2);
    assert_eq!(report.physical_batch, 256);
}
