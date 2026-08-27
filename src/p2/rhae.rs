//! Official ARC-AGI-3 RHAE scoring per https://docs.arcprize.org/methodology .
//!
//! Relative Human Action Efficiency compares per-level action counts to human
//! baselines, aggregates per game with level-weighted averages, then averages
//! across games. This module recomputes RHAE from scorecard JSON returned by the
//! official API; it does not estimate human baselines from recordings alone.

use anyhow::{bail, Context, Result};
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
            .filter(|&n| n > 0)
            .map(|n| usize::try_from(n).context("scorecard level_count overflows usize"))
            .transpose()?
            .or_else(|| {
                env.runs
                    .first()
                    .and_then(|run| usize::try_from(run.number_of_levels).ok())
            })
            .unwrap_or(0);

        for run in &env.runs {
            n_runs += 1;
            let completed = match run.levels_completed {
                Some(count) if count < 0 => bail!("scorecard levels_completed is negative"),
                Some(count) => {
                    usize::try_from(count).context("scorecard levels_completed overflows usize")?
                }
                None => run
                    .level_actions
                    .len()
                    .max(run.level_baseline_actions.len())
                    .max(run.level_scores.len()),
            };
            let per_level = if run.level_actions.len() >= completed
                && run.level_baseline_actions.len() >= completed
                && run.level_actions.len() == run.level_baseline_actions.len()
                && run
                    .level_actions
                    .iter()
                    .take(completed)
                    .all(|&actions| actions > 0 && u32::try_from(actions).is_ok())
                && run
                    .level_baseline_actions
                    .iter()
                    .take(completed)
                    .all(|&baseline| baseline > 0 && u32::try_from(baseline).is_ok())
            {
                run.level_actions
                    .iter()
                    .zip(run.level_baseline_actions.iter())
                    .take(completed)
                    .map(|(&actions, &baseline)| {
                        Ok(level_score(
                            u32::try_from(baseline).context("baseline action overflows u32")?,
                            u32::try_from(actions).context("level action overflows u32")?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else if run.level_scores.len() >= completed
                && run
                    .level_scores
                    .iter()
                    .take(completed)
                    .all(|score| score.is_finite() && (0.0..=115.0).contains(score))
            {
                run.level_scores
                    .iter()
                    .take(completed)
                    .map(|&score| score / 100.0)
                    .collect::<Vec<_>>()
            } else if completed == 0 {
                Vec::new()
            } else {
                bail!(
                    "scorecard has no valid per-level actions/baselines or level_scores for {completed} completed levels"
                );
            };
            let truncated: Vec<f64> = per_level.into_iter().take(completed).collect();
            let total = if total_levels > 0 {
                total_levels
            } else {
                usize::try_from(run.number_of_levels)
                    .unwrap_or(0)
                    .max(truncated.len())
            };
            if completed > total {
                bail!("scorecard completed levels {completed} exceeds total levels {total}");
            }
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

    #[test]
    fn unavailable_baseline_falls_back_to_valid_level_scores() {
        let json = r#"{
            "environments": [{
                "level_count": 1,
                "runs": [{
                    "levels_completed": 1,
                    "level_actions": [1],
                    "level_baseline_actions": [-1],
                    "level_scores": [0.0]
                }]
            }]
        }"#;
        let bench = benchmark_from_scorecard_str(json).unwrap();
        assert_eq!(bench.recomputed_rhae_percent, Some(0.0));
    }

    #[test]
    fn invalid_action_arrays_need_valid_score_fallback() {
        let negative_actions = r#"{"environments":[{"level_count":1,"runs":[{"levels_completed":1,"level_actions":[-1],"level_baseline_actions":[1]}]}]}"#;
        assert!(benchmark_from_scorecard_str(negative_actions).is_err());

        let mismatched_lengths = r#"{"environments":[{"level_count":2,"runs":[{"levels_completed":2,"level_actions":[1],"level_baseline_actions":[1,1]}]}]}"#;
        assert!(benchmark_from_scorecard_str(mismatched_lengths).is_err());
    }

    #[test]
    fn completed_levels_cannot_exceed_total() {
        let json = r#"{
            "environments": [{
                "level_count": 1,
                "runs": [{
                    "levels_completed": 2,
                    "level_actions": [1, 1],
                    "level_baseline_actions": [1, 1]
                }]
            }]
        }"#;
        assert!(benchmark_from_scorecard_str(json).is_err());
    }
}
