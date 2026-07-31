//! Compile-and-test evaluation for the fictional veclab experiment.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tasks::bridge::{
    pass_rate_rag_fraction, rag_ceiling_for_function, BridgeDecodeConfig, BridgeRuntime,
};
use crate::tasks::prepare_veclab::MODULE_PATH;
use crate::tasks::veclab::SEEN_FUNCTION_MAX;
use crate::tasks::veclab::{load_docs_map, VeclabTaskRow};
use crate::tasks::world_support::different_group_conditioning_latent;

#[derive(Clone, Deserialize)]
pub(crate) struct EvalTask {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) fn_ids: Vec<usize>,
    pub(crate) subset: String,
    pub(crate) task: String,
    #[serde(default)]
    pub(crate) must_call: Vec<String>,
    pub(crate) harness_dir: String,
    #[serde(default = "default_max_new")]
    pub(crate) max_new_tokens: usize,
}

fn default_max_new() -> usize {
    512
}

#[derive(Default, Serialize)]
struct Metrics {
    tasks: usize,
    compiled: usize,
    passed: usize,
    categories: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct Rates {
    tasks: usize,
    compile_rate: f64,
    passed: usize,
    suite_pass_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pass_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rag_ceiling: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rag_fraction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pass_at_k_rag_fraction: Option<f64>,
    failure_categories: BTreeMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureCategory {
    CompileError,
    TestsFailed,
    MustCallViolation,
    Timeout,
    Pass,
}

impl FailureCategory {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CompileError => "compile_error",
            Self::TestsFailed => "tests_failed",
            Self::MustCallViolation => "must_call_violation",
            Self::Timeout => "timeout",
            Self::Pass => "pass",
        }
    }

    pub(crate) fn is_pass(self) -> bool {
        matches!(self, Self::Pass)
    }
}

#[derive(Serialize)]
struct TaskResult {
    id: String,
    subset: String,
    condition: String,
    category: FailureCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_code: Option<String>,
}

#[derive(Default, Serialize)]
struct PairedCausalMetrics {
    tasks: usize,
    both_pass: usize,
    matched_only: usize,
    control_only: usize,
    neither_pass: usize,
    matched_advantage: f64,
    one_sided_p_value: f64,
}

#[derive(Serialize)]
struct EvalReport {
    schema_version: u32,
    arm: String,
    suite_path: String,
    suite_sha256: String,
    task_offset: usize,
    task_limit: Option<usize>,
    selected_task_ids: Vec<String>,
    regime: String,
    bridge_model: String,
    results: BTreeMap<String, BTreeMap<String, Rates>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    paired_causal_controls: BTreeMap<String, BTreeMap<String, PairedCausalMetrics>>,
    task_results: Vec<TaskResult>,
}

fn paired_causal_controls(
    task_results: &[TaskResult],
) -> BTreeMap<String, BTreeMap<String, PairedCausalMetrics>> {
    let mut outcomes: BTreeMap<(String, String), BTreeMap<String, bool>> = BTreeMap::new();
    for result in task_results {
        outcomes
            .entry((result.subset.clone(), result.id.clone()))
            .or_default()
            .insert(
                result.condition.clone(),
                matches!(result.category, FailureCategory::Pass),
            );
    }
    let mut controls: BTreeMap<String, BTreeMap<String, PairedCausalMetrics>> = BTreeMap::new();
    for ((subset, _), conditions) in outcomes {
        let Some(&matched) = conditions.get("matched") else {
            continue;
        };
        for control in ["shuffled", "swapped", "zeroed"] {
            let Some(&control_pass) = conditions.get(control) else {
                continue;
            };
            let metrics = controls
                .entry(subset.clone())
                .or_default()
                .entry(control.to_string())
                .or_default();
            metrics.tasks += 1;
            match (matched, control_pass) {
                (true, true) => metrics.both_pass += 1,
                (true, false) => metrics.matched_only += 1,
                (false, true) => metrics.control_only += 1,
                (false, false) => metrics.neither_pass += 1,
            }
        }
    }
    for subset in controls.values_mut() {
        for metrics in subset.values_mut() {
            metrics.matched_advantage = (metrics.matched_only as f64 - metrics.control_only as f64)
                / metrics.tasks.max(1) as f64;
            metrics.one_sided_p_value =
                exact_one_sided_sign_test(metrics.matched_only, metrics.control_only);
        }
    }
    controls
}

pub(crate) fn exact_one_sided_sign_test(matched_only: usize, control_only: usize) -> f64 {
    let discordant = matched_only + control_only;
    if discordant == 0 {
        return 1.0;
    }
    let mut probability = 2f64.powi(-(discordant as i32));
    let mut upper_tail = 0.0;
    for successes in 0..=discordant {
        if successes >= matched_only {
            upper_tail += probability;
        }
        if successes < discordant {
            probability *= (discordant - successes) as f64 / (successes + 1) as f64;
        }
    }
    upper_tail.min(1.0)
}

fn optional_probability_env(name: &str) -> Result<Option<f64>> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = raw
        .to_string_lossy()
        .parse::<f64>()
        .with_context(|| format!("{name} must be a probability in [0,1]"))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite probability in [0,1], got {raw:?}");
    }
    Ok(Some(value))
}

pub(crate) fn load_suite(path: &Path) -> Result<Vec<EvalTask>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn strip_fence(text: &str) -> String {
    let text = text.trim();
    if !text.starts_with("```") {
        return text
            .find("package solution")
            .map(|start| text[start..].trim().to_string())
            .unwrap_or_else(|| text.to_string());
    }
    let body = text.split_once('\n').map(|(_, body)| body).unwrap_or("");
    body.rsplit_once("```")
        .map(|(body, _)| body)
        .unwrap_or(body)
        .trim()
        .to_string()
}

enum CommandOutcome {
    Success,
    Failed,
    Timeout,
}

fn run_go_with_timeout(dir: &Path, args: &[&str], timeout: Duration) -> Result<CommandOutcome> {
    let mut child = Command::new("go").args(args).current_dir(dir).spawn()?;
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(if status.success() {
                CommandOutcome::Success
            } else {
                CommandOutcome::Failed
            });
        }
        if started.elapsed() >= timeout {
            child.kill()?;
            let _ = child.wait();
            return Ok(CommandOutcome::Timeout);
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn executable_api_call(code: &str, required: &str) -> bool {
    fn contains_exact_selector_call(
        node: tree_sitter::Node<'_>,
        source: &[u8],
        qualifier: &str,
        field: &str,
    ) -> bool {
        if node.kind() == "call_expression" {
            let exact_selector = node
                .child_by_field_name("function")
                .filter(|function| function.kind() == "selector_expression")
                .and_then(|function| {
                    Some((
                        function.child_by_field_name("operand")?,
                        function.child_by_field_name("field")?,
                    ))
                })
                .is_some_and(|(operand, selected)| {
                    operand.kind() == "identifier"
                        && operand.utf8_text(source).ok() == Some(qualifier)
                        && selected.utf8_text(source).ok() == Some(field)
                });
            if exact_selector {
                return true;
            }
        }
        let mut cursor = node.walk();
        let found = node
            .named_children(&mut cursor)
            .any(|child| contains_exact_selector_call(child, source, qualifier, field));
        found
    }

    let (qualifier, field) = required.rsplit_once('.').unwrap_or(("veclab", required));
    let mut parser = tree_sitter::Parser::new();
    if parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .is_err()
    {
        return false;
    }
    parser.parse(code, None).is_some_and(|tree| {
        contains_exact_selector_call(tree.root_node(), code.as_bytes(), qualifier, field)
    })
}

pub(crate) fn compile_and_test(
    task: &EvalTask,
    code: &str,
    condition: &str,
) -> Result<FailureCategory> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = std::env::temp_dir().join(format!("tofy-veclab-{}-{condition}-{stamp}", task.id));
    fs::create_dir_all(&dir)?;
    let harness = PathBuf::from(&task.harness_dir);
    let harness = fs::canonicalize(&harness)
        .with_context(|| format!("resolve harness dir {}", harness.display()))?;
    let veclab = fs::canonicalize("data/fictional/veclab")?;
    fs::write(
        dir.join("go.mod"),
        format!(
            "module solution\n\ngo 1.22\n\nrequire {MODULE_PATH} v0.0.0\nreplace {MODULE_PATH} => {}\n",
            veclab.display()
        ),
    )?;
    fs::write(dir.join("solution.go"), strip_fence(code))?;
    fs::copy(harness.join("main_test.go"), dir.join("main_test.go"))?;
    let timeout = Duration::from_secs(
        std::env::var("TOFY_EVAL_TASK_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(30),
    );
    let category = if !task
        .must_call
        .iter()
        .all(|name| executable_api_call(code, name))
    {
        FailureCategory::MustCallViolation
    } else {
        match run_go_with_timeout(&dir, &["build", "."], timeout)? {
            CommandOutcome::Timeout => FailureCategory::Timeout,
            CommandOutcome::Failed => FailureCategory::CompileError,
            CommandOutcome::Success => match run_go_with_timeout(&dir, &["test", "."], timeout)? {
                CommandOutcome::Timeout => FailureCategory::Timeout,
                CommandOutcome::Failed => FailureCategory::TestsFailed,
                CommandOutcome::Success => FailureCategory::Pass,
            },
        }
    };
    let _ = fs::remove_dir_all(dir);
    Ok(category)
}

fn rate(
    metrics: &Metrics,
    pass_at_k: Option<usize>,
    subset: &str,
) -> Rates {
    let denom = metrics.tasks.max(1) as f64;
    let suite_pass_rate = metrics.passed as f64 / denom;
    let pass_at_k_rate = pass_at_k.map(|passed| passed as f64 / denom);
    let rag_ceiling = if metrics.tasks > 0 {
        Some(
            if subset == "heldout" {
                rag_ceiling_for_function(SEEN_FUNCTION_MAX + 1)
            } else {
                rag_ceiling_for_function(1)
            } as f64,
        )
    } else {
        None
    };
    Rates {
        tasks: metrics.tasks,
        compile_rate: metrics.compiled as f64 / denom,
        passed: metrics.passed,
        suite_pass_rate,
        pass_at_k: pass_at_k_rate,
        rag_ceiling,
        rag_fraction: rag_ceiling.map(|ceiling| {
            pass_rate_rag_fraction(suite_pass_rate as f32, ceiling as f32) as f64
        }),
        pass_at_k_rag_fraction: pass_at_k_rate.zip(rag_ceiling).map(|(passed, ceiling)| {
            pass_rate_rag_fraction(passed as f32, ceiling as f32) as f64
        }),
        failure_categories: metrics.categories.clone(),
    }
}

pub fn try_run_code_eval(args: &[String]) -> Result<bool> {
    if !matches!(
        args.get(1).map(String::as_str),
        Some("--eval-code-assistant" | "eval-code-assistant" | "--eval-bridge" | "eval-bridge")
    ) {
        return Ok(false);
    }
    if args.len() < 8 {
        bail!("usage: --eval-bridge <qwen_dir> <bridge.safetensors> <encoder.safetensors> <vocab.txt> <world.safetensors> <suite.jsonl> [report.json]");
    }
    let runtime = BridgeRuntime::load(
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        Path::new(&args[5]),
        Path::new(&args[6]),
    )?;
    let suite_path = Path::new(&args[7]);
    let suite_bytes = fs::read(suite_path)
        .with_context(|| format!("read evaluation suite {}", suite_path.display()))?;
    let suite_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(&suite_bytes);
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let mut tasks = load_suite(suite_path)?;
    let task_offset = std::env::var("TOFY_EVAL_TASK_OFFSET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if task_offset > 0 {
        tasks = tasks.into_iter().skip(task_offset).collect();
    }
    let task_limit = std::env::var("TOFY_EVAL_MAX_TASKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    if let Some(limit) = task_limit {
        tasks.truncate(limit.max(1));
    }
    if tasks.is_empty() {
        bail!("evaluation suite is empty");
    }
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt"))?;
    let rows = tasks
        .iter()
        .map(|task| {
            let function_id = task.fn_ids.first().copied().unwrap_or(0);
            VeclabTaskRow {
                task: task.task.clone(),
                completion: String::new(),
                function_id,
                docs: function_id
                    .checked_sub(1)
                    .and_then(|_| docs.get(&function_id).cloned())
                    .unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    let eval_mode = std::env::var("TOFY_EVAL_MODE").unwrap_or_else(|_| "bridge".into());
    if !matches!(
        eval_mode.as_str(),
        "bridge" | "floor" | "rag" | "unconditioned"
    ) {
        bail!("TOFY_EVAL_MODE must be bridge, floor, rag, or unconditioned");
    }
    let all_cond = if eval_mode == "bridge" {
        let cond_parts = rows
            .chunks(8)
            .map(|chunk| runtime.conditioning(chunk).map(|tensor| tensor.detach()))
            .collect::<Result<Vec<_>>>()?;
        let cond_refs = cond_parts.iter().collect::<Vec<_>>();
        candle_core::Tensor::cat(&cond_refs, 0)?
    } else {
        runtime.zero_conditioning(rows.len())?
    };
    let function_ids = rows.iter().map(|row| row.function_id).collect::<Vec<_>>();
    let shuffled = different_group_conditioning_latent(&all_cond, &function_ids, 7)?;
    let swapped = different_group_conditioning_latent(&all_cond, &function_ids, 1)?;
    let decode = BridgeDecodeConfig::from_env();
    let mut totals: BTreeMap<(String, String), Metrics> = BTreeMap::new();
    let mut pass_at_k_totals: BTreeMap<(String, String), usize> = BTreeMap::new();
    let failure_code_limit = std::env::var("TOFY_EVAL_FAILURE_CODE_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20usize);
    let mut stored_failure_codes = 0usize;
    let mut task_results = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        let matched = all_cond.narrow(0, index, 1)?;
        let conditions = if eval_mode != "bridge" {
            vec![("base", matched.zeros_like()?)]
        } else {
            vec![
                ("matched", matched.clone()),
                ("zeroed", matched.zeros_like()?),
                ("shuffled", shuffled.narrow(0, index, 1)?),
                ("swapped", swapped.narrow(0, index, 1)?),
            ]
        };
        for (condition, cond) in conditions {
            let prompt = if eval_mode == "rag" {
                let doc = rows[index].docs.clone();
                format!("Relevant veclab documentation:\n{doc}\n\n{}", task.task)
            } else {
                task.task.clone()
            };
            let cond = if eval_mode == "bridge" {
                cond
            } else {
                cond.zeros_like()?
            };
            let samples = if decode.uses_sampling() {
                runtime.generate_samples(&prompt, &cond, task.max_new_tokens.min(512), decode)?
            } else {
                vec![runtime.generate(&prompt, &cond, task.max_new_tokens.min(512))?]
            };
            let mut sample_passed = false;
            for (sample_index, code) in samples.iter().take(decode.pass_at_k).enumerate() {
                let category = compile_and_test(task, code, condition)
                    .with_context(|| format!("evaluate {} under {condition}", task.id))?;
                sample_passed |= category.is_pass();
                if sample_index == 0 {
                    let metric = totals
                        .entry((task.subset.clone(), condition.into()))
                        .or_default();
                    metric.tasks += 1;
                    metric.compiled += usize::from(matches!(
                        category,
                        FailureCategory::TestsFailed | FailureCategory::Pass
                    ));
                    metric.passed += usize::from(matches!(category, FailureCategory::Pass));
                    *metric
                        .categories
                        .entry(category.as_str().to_string())
                        .or_default() += 1;
                    let retain_code = !matches!(category, FailureCategory::Pass)
                        && stored_failure_codes < failure_code_limit;
                    if retain_code {
                        stored_failure_codes += 1;
                    }
                    task_results.push(TaskResult {
                        id: task.id.clone(),
                        subset: task.subset.clone(),
                        condition: condition.to_string(),
                        category,
                        generated_code: retain_code.then(|| code.clone()),
                    });
                }
            }
            if sample_passed {
                *pass_at_k_totals
                    .entry((task.subset.clone(), condition.into()))
                    .or_default() += 1;
            }
        }
    }
    let mut results: BTreeMap<String, BTreeMap<String, Rates>> = BTreeMap::new();
    for ((subset, condition), metrics) in totals {
        let pass_at_k = pass_at_k_totals
            .get(&(subset.clone(), condition.clone()))
            .copied();
        results
            .entry(subset.clone())
            .or_default()
            .insert(condition, rate(&metrics, pass_at_k, &subset));
    }
    let paired_causal_controls = paired_causal_controls(&task_results);
    let report = EvalReport {
        schema_version: 2,
        arm: std::env::var("TOFY_EVAL_ARM").unwrap_or_else(|_| eval_mode.clone()),
        suite_path: suite_path.to_string_lossy().to_string(),
        suite_sha256,
        task_offset,
        task_limit,
        selected_task_ids: tasks.iter().map(|task| task.id.clone()).collect(),
        regime: if eval_mode == "bridge" {
            std::env::var("TOFY_BRIDGE_REGIME").unwrap_or_else(|_| "weights".into())
        } else {
            eval_mode.clone()
        },
        bridge_model: args[3].clone(),
        results,
        paired_causal_controls,
        task_results,
    };
    let default_report = PathBuf::from(format!(
        "runs/bridge_eval/{}/report.json",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
    ));
    let report_path = args.get(8).map(PathBuf::from).unwrap_or(default_report);
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    for (subset, conditions) in &report.results {
        for (condition, rates) in conditions {
            if let (Some(pass_at_k), Some(rag_fraction)) = (rates.pass_at_k, rates.rag_fraction) {
                println!(
                    "eval {subset}/{condition}: pass_rate={:.4} pass@k={pass_at_k:.4} rag_ceiling={:.4} rag_fraction={rag_fraction:.4} pass@k_rag_fraction={:.4}",
                    rates.suite_pass_rate,
                    rates.rag_ceiling.unwrap_or(0.0),
                    rates.pass_at_k_rag_fraction.unwrap_or(0.0),
                );
            }
        }
    }
    println!("Evaluation report: {}", report_path.display());
    if let Some(minimum) = optional_probability_env("TOFY_EVAL_MIN_PASS_RATE")? {
        if eval_mode == "bridge" {
            let rates = report
                .results
                .get("heldout")
                .and_then(|conditions| conditions.get("matched"))
                .context("gated bridge evaluation is missing heldout/matched results")?;
            if rates.suite_pass_rate < minimum {
                bail!(
                    "held-out knowledge-transfer gate failed: matched_pass_rate={:.4} required={minimum:.4}; report={}",
                    rates.suite_pass_rate,
                    report_path.display()
                );
            }
        } else {
            for (subset, conditions) in &report.results {
                for (condition, rates) in conditions {
                    if rates.suite_pass_rate < minimum {
                        bail!(
                            "evaluation ceiling failed for subset={subset} condition={condition}: pass_rate={:.4} required={minimum:.4}; report={}",
                            rates.suite_pass_rate,
                            report_path.display()
                        );
                    }
                }
            }
        }
    }
    if eval_mode == "bridge" {
        if let Some(minimum) = optional_probability_env("TOFY_EVAL_MIN_CAUSAL_ADVANTAGE")? {
            let controls = report
                .paired_causal_controls
                .get("heldout")
                .context("gated bridge evaluation is missing heldout causal controls")?;
            for control in ["shuffled", "swapped", "zeroed"] {
                let metrics = controls.get(control).with_context(|| {
                    format!("gated bridge evaluation is missing heldout/{control}")
                })?;
                if metrics.matched_advantage < minimum {
                    bail!(
                        "held-out causal gate failed for {control}: matched_advantage={:.4} required={minimum:.4}; report={}",
                        metrics.matched_advantage,
                        report_path.display()
                    );
                }
            }
        }
        if let Some(maximum) = optional_probability_env("TOFY_EVAL_MAX_CAUSAL_P_VALUE")? {
            let controls = report
                .paired_causal_controls
                .get("heldout")
                .context("statistically gated bridge evaluation is missing heldout controls")?;
            for control in ["shuffled", "swapped", "zeroed"] {
                let metrics = controls.get(control).with_context(|| {
                    format!("statistically gated bridge evaluation is missing heldout/{control}")
                })?;
                if metrics.one_sided_p_value > maximum {
                    bail!(
                        "held-out paired causal significance gate failed for {control}: matched_only={} control_only={} one_sided_p={:.6} required_max={maximum:.6}; report={}",
                        metrics.matched_only,
                        metrics.control_only,
                        metrics.one_sided_p_value,
                        report_path.display()
                    );
                }
            }
        }
    }
    let mut results_file = OpenOptions::new().append(true).open("docs/RESULTS.md")?;
    writeln!(
        results_file,
        "\n- Bridge eval `{}`: regime `{}`; report `{}`.",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        report.regime,
        report_path.display()
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_task_deserializes_new_schema() {
        let row = r#"{"id":"veclab-137-2","fn_ids":[137],"subset":"heldout","task":"Write Solve","must_call":["Vorbel"],"harness_dir":"eval/veclab/137-2/"}"#;
        let task: EvalTask = serde_json::from_str(row).unwrap();
        assert_eq!(task.fn_ids, vec![137]);
        assert_eq!(task.must_call, vec!["Vorbel"]);
    }

    #[test]
    fn failure_categories_serialize_as_report_values() {
        assert_eq!(
            serde_json::to_string(&FailureCategory::MustCallViolation).unwrap(),
            "\"must_call_violation\""
        );
        assert_eq!(
            serde_json::to_string(&FailureCategory::Timeout).unwrap(),
            "\"timeout\""
        );
    }

    #[test]
    fn required_api_must_be_an_executable_call() {
        assert!(executable_api_call(
            "package solution\nfunc Solve() { _ = veclab.Vorbel(1) }",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\n// veclab.Vorbel(1)\nfunc Solve() {}",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\nfunc Solve() { _ = \"veclab.Vorbel(1)\" }",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\nfunc Solve() { _ = notveclab.Vorbel(1) }",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\nfunc Solve() { _ = veclab.VorbelExtra(1) }",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\nfunc Solve() { _ = holder.veclab.Vorbel(1) }",
            "Vorbel"
        ));
        assert!(!executable_api_call(
            "package solution\nfunc Solve() { _ = éveclab.Vorbel(1) }",
            "Vorbel"
        ));
    }

    #[test]
    fn paired_controls_report_matched_advantage() {
        let row = |id: &str, condition: &str, category| TaskResult {
            id: id.into(),
            subset: "heldout".into(),
            condition: condition.into(),
            category,
            generated_code: None,
        };
        let metrics = paired_causal_controls(&[
            row("a", "matched", FailureCategory::Pass),
            row("a", "shuffled", FailureCategory::MustCallViolation),
            row("b", "matched", FailureCategory::MustCallViolation),
            row("b", "shuffled", FailureCategory::Pass),
        ]);
        let shuffled = &metrics["heldout"]["shuffled"];
        assert_eq!(shuffled.tasks, 2);
        assert_eq!(shuffled.matched_only, 1);
        assert_eq!(shuffled.control_only, 1);
        assert_eq!(shuffled.matched_advantage, 0.0);
        assert_eq!(shuffled.one_sided_p_value, 0.75);
    }

    #[test]
    fn exact_sign_test_supports_bonferroni_causal_gate() {
        assert!((exact_one_sided_sign_test(6, 0) - 0.015625).abs() < 1e-12);
        assert!((exact_one_sided_sign_test(5, 0) - 0.03125).abs() < 1e-12);
        assert!(exact_one_sided_sign_test(7, 0) < 0.05 / 6.0);
        assert!(exact_one_sided_sign_test(6, 0) > 0.05 / 6.0);
        assert!(exact_one_sided_sign_test(6, 1) > 0.016_666_667);
        assert_eq!(exact_one_sided_sign_test(0, 0), 1.0);
    }
}
