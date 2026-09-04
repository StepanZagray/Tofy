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
    /// Required by the official scorecard schema; forms the RHAE denominator
    /// (`1..=level_count`) for every run of the environment. Missing or
    /// nonpositive values are rejected rather than defaulted from a run.
    #[serde(default)]
    level_count: Option<i64>,
    #[serde(default)]
    runs: Vec<RunEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct RunEntry {
    #[serde(default)]
    levels_completed: Option<i64>,
    /// Optional per-run copy of the environment level count. When present it
    /// must agree with `level_count`; it never supplies the denominator.
    #[serde(default)]
    number_of_levels: Option<i64>,
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

/// Resolve the RHAE denominator for one environment.
///
/// The official scorecard schema requires `level_count` on every environment,
/// and the level-weighted game score divides by `1 + 2 + ... + level_count`.
/// A missing, zero, or negative count therefore cannot form a denominator and
/// is rejected instead of being defaulted from the first run (which would
/// silently reuse one run's count for every run of the environment).
fn environment_level_count(env_index: usize, level_count: Option<i64>) -> Result<usize> {
    match level_count {
        None => bail!("scorecard environment {env_index} is missing level_count"),
        Some(count) if count <= 0 => {
            bail!("scorecard environment {env_index} level_count {count} must be positive")
        }
        Some(count) => usize::try_from(count).with_context(|| {
            format!("scorecard environment {env_index} level_count overflows usize")
        }),
    }
}

/// A run's optional `number_of_levels` describes the same environment and
/// must agree with the environment `level_count` when present.
fn validate_run_level_count(
    env_index: usize,
    run_index: usize,
    number_of_levels: Option<i64>,
    total_levels: usize,
) -> Result<()> {
    match number_of_levels {
        None => Ok(()),
        Some(count) if count <= 0 => bail!(
            "scorecard environment {env_index} run {run_index} number_of_levels {count} must be positive"
        ),
        Some(count) if usize::try_from(count).ok() == Some(total_levels) => Ok(()),
        Some(count) => bail!(
            "scorecard environment {env_index} run {run_index} number_of_levels {count} does not match environment level_count {total_levels}"
        ),
    }
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

    for (env_index, env) in card.environments.iter().enumerate() {
        let total_levels = environment_level_count(env_index, env.level_count)?;

        for (run_index, run) in env.runs.iter().enumerate() {
            n_runs += 1;
            validate_run_level_count(env_index, run_index, run.number_of_levels, total_levels)?;
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
            if completed > total_levels {
                bail!("scorecard completed levels {completed} exceeds total levels {total_levels}");
            }
            game_scores.push(game_score(&truncated, total_levels));
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

    /// Wave 22 finding 20 witness: two runs of one environment with omitted
    /// `level_count` used to inherit the first run's `number_of_levels` as the
    /// denominator, so `[1/1, 1/2]` scored 100% in one order and 33.33% in the
    /// other. Missing `level_count` is schema-invalid and must be rejected in
    /// both orders instead of scoring anything.
    #[test]
    fn missing_level_count_is_rejected_in_both_run_orders() {
        let run_one = r#"{"levels_completed":1,"number_of_levels":1,"level_actions":[10],"level_baseline_actions":[10]}"#;
        let run_two = r#"{"levels_completed":1,"number_of_levels":2,"level_actions":[10],"level_baseline_actions":[10]}"#;
        for (first, second) in [(run_one, run_two), (run_two, run_one)] {
            let json = format!(r#"{{"environments":[{{"runs":[{first},{second}]}}]}}"#);
            let err = benchmark_from_scorecard_str(&json).unwrap_err();
            assert!(
                err.to_string().contains("missing level_count"),
                "unexpected error: {err:#}"
            );
        }
    }

    #[test]
    fn nonpositive_level_count_is_rejected() {
        for count in ["0", "-1"] {
            let json = format!(
                r#"{{"environments":[{{"level_count":{count},"runs":[{{"levels_completed":0}}]}}]}}"#
            );
            let err = benchmark_from_scorecard_str(&json).unwrap_err();
            assert!(
                err.to_string().contains("must be positive"),
                "level_count {count}: unexpected error: {err:#}"
            );
        }
        // Environments without any runs still need a schema-valid count.
        let no_runs = r#"{"environments":[{"level_count":0,"runs":[]}]}"#;
        assert!(benchmark_from_scorecard_str(no_runs).is_err());
        let missing_no_runs = r#"{"environments":[{"runs":[]}]}"#;
        assert!(benchmark_from_scorecard_str(missing_no_runs).is_err());
    }

    #[test]
    fn run_level_count_must_match_environment_level_count() {
        let mismatched = r#"{"environments":[{"level_count":2,"runs":[
            {"levels_completed":1,"number_of_levels":2,"level_actions":[10],"level_baseline_actions":[10]},
            {"levels_completed":1,"number_of_levels":3,"level_actions":[10],"level_baseline_actions":[10]}
        ]}]}"#;
        let err = benchmark_from_scorecard_str(mismatched).unwrap_err();
        assert!(
            err.to_string()
                .contains("run 1 number_of_levels 3 does not match"),
            "unexpected error: {err:#}"
        );
        for count in ["0", "-2"] {
            let json = format!(
                r#"{{"environments":[{{"level_count":2,"runs":[{{"levels_completed":0,"number_of_levels":{count}}}]}}]}}"#
            );
            let err = benchmark_from_scorecard_str(&json).unwrap_err();
            assert!(
                err.to_string().contains("number_of_levels")
                    && err.to_string().contains("must be positive"),
                "number_of_levels {count}: unexpected error: {err:#}"
            );
        }
    }

    #[test]
    fn well_formed_multi_run_environment_uses_environment_denominator_for_every_run() {
        // Two runs of a two-level game: one completes both levels at baseline,
        // the other completes only level 1. Expected game scores are
        // [1.0, 1/3]; the order of the runs must not change the result.
        let run_full = r#"{"levels_completed":2,"number_of_levels":2,"level_actions":[10,10],"level_baseline_actions":[10,10]}"#;
        let run_partial =
            r#"{"levels_completed":1,"level_actions":[10],"level_baseline_actions":[10]}"#;
        let expected_mean = (1.0 + 1.0 / 3.0) / 2.0 * 100.0;
        for (first, second) in [(run_full, run_partial), (run_partial, run_full)] {
            let json = format!(
                r#"{{"score":66.67,"environments":[{{"level_count":2,"runs":[{first},{second}]}}]}}"#
            );
            let bench = benchmark_from_scorecard_str(&json).unwrap();
            assert_eq!(bench.n_environments, 1);
            assert_eq!(bench.n_runs, 2);
            let mut sorted = bench.game_scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((sorted[0] - 1.0 / 3.0).abs() < 1e-9, "{sorted:?}");
            assert!((sorted[1] - 1.0).abs() < 1e-9, "{sorted:?}");
            let rhae = bench.recomputed_rhae_percent.unwrap();
            assert!((rhae - expected_mean).abs() < 1e-9, "rhae={rhae}");
        }
    }

    #[test]
    fn unplayed_environment_with_valid_level_count_contributes_no_game_score() {
        let json = r#"{"environments":[{"level_count":3,"runs":[]}]}"#;
        let bench = benchmark_from_scorecard_str(json).unwrap();
        assert_eq!(bench.n_environments, 1);
        assert_eq!(bench.n_runs, 0);
        assert!(bench.game_scores.is_empty());
        assert_eq!(bench.recomputed_rhae_percent, None);
    }
}
