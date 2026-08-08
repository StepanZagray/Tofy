//! Official ARC-AGI-3 RHAE scoring per https://docs.arcprize.org/methodology .
//!
//! Relative Human Action Efficiency compares per-level action counts to human
//! baselines, aggregates per game with level-weighted averages, then averages
//! across games. This module recomputes RHAE from scorecard JSON returned by the
//! official API; it does not estimate human baselines from recordings alone.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Maximum per-level score multiplier when AI beats the human baseline.
pub const LEVEL_SCORE_CAP: f64 = 1.15;

/// Per-level score: `(human_baseline / ai_actions)²`, capped at [`LEVEL_SCORE_CAP`].
pub fn level_score(human_baseline: u32, ai_actions: u32) -> f64 {
    if human_baseline == 0 || ai_actions == 0 {
        return 0.0;
    }
    let raw = (f64::from(human_baseline) / f64::from(ai_actions)).powi(2);
    raw.min(LEVEL_SCORE_CAP)
}

/// Weighted game score using 1-indexed level numbers as weights.
///
/// `level_scores` contains one entry per **completed** level (in order). Levels
/// not completed contribute zero to the numerator; the denominator always sums
/// `1..=total_levels`.
pub fn game_score(level_scores: &[f64], total_levels: usize) -> f64 {
    if total_levels == 0 {
        return 0.0;
    }
    let denom: f64 = (1..=total_levels).map(|i| i as f64).sum();
    let numer: f64 = level_scores
        .iter()
        .enumerate()
        .map(|(i, score)| (i as f64 + 1.0) * score)
        .sum();
    let weighted = numer / denom;
    let completed_weight: f64 = (1..=level_scores.len()).map(|i| i as f64).sum();
    let cap = completed_weight / denom;
    weighted.min(cap)
}

/// Total RHAE as the unweighted mean of per-game scores, scaled to 0–100%.
pub fn total_rhae_percent(game_scores: &[f64]) -> Option<f64> {
    if game_scores.is_empty() {
        return None;
    }
    let mean = game_scores.iter().sum::<f64>() / game_scores.len() as f64;
    Some(mean * 100.0)
}

#[derive(Debug, Clone, Deserialize)]
struct ScorecardFile {
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    environments: Vec<EnvironmentEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct EnvironmentEntry {
    #[serde(default)]
    level_count: Option<i64>,
    #[serde(default)]
    runs: Vec<RunEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct RunEntry {
    #[serde(default)]
    levels_completed: Option<i64>,
    #[serde(default)]
    number_of_levels: i64,
    #[serde(default)]
    level_scores: Vec<f64>,
    #[serde(default)]
    level_actions: Vec<i64>,
    #[serde(default)]
    level_baseline_actions: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScorecardBenchmark {
    pub api_score: Option<f64>,
    pub recomputed_rhae_percent: Option<f64>,
    pub n_environments: usize,
    pub n_runs: usize,
    pub game_scores: Vec<f64>,
}

/// Parse official scorecard JSON and recompute RHAE from per-level baselines.
pub fn benchmark_from_scorecard_json(path: &Path) -> Result<ScorecardBenchmark> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read scorecard {}", path.display()))?;
    benchmark_from_scorecard_str(&text)
}

pub fn benchmark_from_scorecard_str(json: &str) -> Result<ScorecardBenchmark> {
    let card: ScorecardFile = serde_json::from_str(json).context("parse scorecard JSON")?;
    let mut game_scores = Vec::new();
    let mut n_runs = 0usize;

    for env in &card.environments {
        let total_levels = env
            .level_count
            .and_then(|n| (n > 0).then_some(n as usize))
            .or_else(|| {
                env.runs
                    .first()
                    .map(|run| run.number_of_levels.max(0) as usize)
            })
            .unwrap_or(0);

        for run in &env.runs {
            n_runs += 1;
            let per_level = if run.level_actions.len() == run.level_baseline_actions.len()
                && !run.level_actions.is_empty()
            {
                run.level_actions
                    .iter()
                    .zip(run.level_baseline_actions.iter())
                    .map(|(&actions, &baseline)| level_score(baseline as u32, actions as u32))
                    .collect::<Vec<_>>()
            } else if !run.level_scores.is_empty() {
                run.level_scores
                    .iter()
                    .map(|&s| s / 100.0)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let completed = run
                .levels_completed
                .map(|count| count.max(0) as usize)
                .unwrap_or(per_level.len());
            let truncated: Vec<f64> = per_level.into_iter().take(completed).collect();
            let total = if total_levels > 0 {
                total_levels
            } else {
                run.number_of_levels.max(truncated.len() as i64) as usize
            };
            if total > 0 {
                game_scores.push(game_score(&truncated, total));
            }
        }
    }

    let recomputed = total_rhae_percent(&game_scores);
    let api_score = card.score;

    Ok(ScorecardBenchmark {
        api_score,
        recomputed_rhae_percent: recomputed,
        n_environments: card.environments.len(),
        n_runs,
        game_scores,
    })
}

/// Prefer recomputed RHAE when baselines are present; fall back to API aggregate score.
pub fn official_rhae_from_benchmark(bench: &ScorecardBenchmark) -> Option<f64> {
    bench.recomputed_rhae_percent.or(bench.api_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methodology_level_score_examples() {
        assert!((level_score(10, 10) - 1.0).abs() < 1e-9);
        assert!((level_score(10, 20) - 0.25).abs() < 1e-9);
        assert!((level_score(10, 100) - 0.01).abs() < 1e-9);
        assert!((level_score(10, 5) - LEVEL_SCORE_CAP).abs() < 1e-9);
    }

    #[test]
    fn incomplete_game_caps_max_score() {
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let g = game_score(&scores, 5);
        assert!((g - 10.0 / 15.0).abs() < 1e-9);
    }

    #[test]
    fn capped_level_scores_cannot_exceed_completion_fraction() {
        let scores = vec![LEVEL_SCORE_CAP; 4];
        let g = game_score(&scores, 5);
        assert!(
            g <= 10.0 / 15.0 + 1e-9,
            "four completed levels of 1.15 must not exceed 4/5 weighted fraction, got {g}"
        );
        assert!((g - 10.0 / 15.0).abs() < 1e-9);
    }

    #[test]
    fn total_rhae_is_mean_of_game_scores() {
        let pct = total_rhae_percent(&[0.667, 1.0]).unwrap();
        assert!((pct - 83.35).abs() < 0.01);
    }

    #[test]
    fn scorecard_recompute_from_baselines() {
        let json = r#"{
            "score": 67,
            "environments": [{
                "id": "demo",
                "level_count": 5,
                "runs": [{
                    "levels_completed": 4,
                    "number_of_levels": 5,
                    "level_actions": [10, 10, 10, 10],
                    "level_baseline_actions": [10, 10, 10, 10]
                }]
            }]
        }"#;
        let bench = benchmark_from_scorecard_str(json).unwrap();
        let rhae = bench.recomputed_rhae_percent.unwrap();
        assert!((rhae - (10.0 / 15.0 * 100.0)).abs() < 0.01);
    }

    #[test]
    fn live_scorecard_accepts_fractional_numeric_fields() {
        let json = r#"{
            "score": 0.0,
            "environments": [{
                "level_count": 2,
                "runs": [{
                    "levels_completed": 0,
                    "level_scores": [0.0, 0.0],
                    "level_actions": [1, 0],
                    "level_baseline_actions": [10, 20]
                }]
            }]
        }"#;
        let bench = benchmark_from_scorecard_str(json).unwrap();
        assert_eq!(bench.api_score, Some(0.0));
        assert_eq!(bench.recomputed_rhae_percent, Some(0.0));
    }
}
