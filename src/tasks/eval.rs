//! Compile-and-test evaluation for the fictional veclab experiment.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tasks::bridge::BridgeRuntime;
use crate::tasks::prepare_veclab::MODULE_PATH;
use crate::tasks::veclab::{load_docs_map, VeclabTaskRow};
use crate::tasks::world_support::different_group_conditioning_latent;

#[derive(Clone, Deserialize)]
struct EvalTask {
    id: String,
    #[serde(default)]
    fn_ids: Vec<usize>,
    subset: String,
    task: String,
    #[serde(default)]
    must_call: Vec<String>,
    harness_dir: String,
    #[serde(default = "default_max_new")]
    max_new_tokens: usize,
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
    suite_pass_rate: f64,
    failure_categories: BTreeMap<String, usize>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FailureCategory {
    CompileError,
    TestsFailed,
    MustCallViolation,
    Timeout,
    Pass,
}

impl FailureCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::CompileError => "compile_error",
            Self::TestsFailed => "tests_failed",
            Self::MustCallViolation => "must_call_violation",
            Self::Timeout => "timeout",
            Self::Pass => "pass",
        }
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
}

#[derive(Serialize)]
struct EvalReport {
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
        }
    }
    controls
}

fn load_suite(path: &Path) -> Result<Vec<EvalTask>> {
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

fn compile_and_test(task: &EvalTask, code: &str, condition: &str) -> Result<FailureCategory> {
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
    let category = if !task.must_call.iter().all(|name| code.contains(name)) {
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

fn rate(metrics: &Metrics) -> Rates {
    let denom = metrics.tasks.max(1) as f64;
    Rates {
        tasks: metrics.tasks,
        compile_rate: metrics.compiled as f64 / denom,
        suite_pass_rate: metrics.passed as f64 / denom,
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
    let mut tasks = load_suite(Path::new(&args[7]))?;
    if let Some(offset) = std::env::var("TOFY_EVAL_TASK_OFFSET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        tasks = tasks.into_iter().skip(offset).collect();
    }
    if let Some(limit) = std::env::var("TOFY_EVAL_MAX_TASKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
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
    let cond_parts = rows
        .chunks(8)
        .map(|chunk| runtime.conditioning(chunk).map(|tensor| tensor.detach()))
        .collect::<Result<Vec<_>>>()?;
    let cond_refs = cond_parts.iter().collect::<Vec<_>>();
    let all_cond = candle_core::Tensor::cat(&cond_refs, 0)?;
    let function_ids = rows.iter().map(|row| row.function_id).collect::<Vec<_>>();
    let shuffled = different_group_conditioning_latent(&all_cond, &function_ids, 7)?;
    let swapped = different_group_conditioning_latent(&all_cond, &function_ids, 1)?;
    let eval_mode = std::env::var("TOFY_EVAL_MODE").unwrap_or_else(|_| "bridge".into());
    if !matches!(
        eval_mode.as_str(),
        "bridge" | "floor" | "rag" | "unconditioned"
    ) {
        bail!("TOFY_EVAL_MODE must be bridge, floor, rag, or unconditioned");
    }
    let mut totals: BTreeMap<(String, String), Metrics> = BTreeMap::new();
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
            let code = runtime.generate(&prompt, &cond, task.max_new_tokens.min(512))?;
            let category = compile_and_test(task, &code, condition)
                .with_context(|| format!("evaluate {} under {condition}", task.id))?;
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
                generated_code: retain_code.then_some(code),
            });
        }
    }
    let mut results: BTreeMap<String, BTreeMap<String, Rates>> = BTreeMap::new();
    for ((subset, condition), metrics) in totals {
        results
            .entry(subset)
            .or_default()
            .insert(condition, rate(&metrics));
    }
    let paired_causal_controls = paired_causal_controls(&task_results);
    let report = EvalReport {
        regime: if eval_mode == "bridge" {
            std::env::var("TOFY_BRIDGE_REGIME").unwrap_or_else(|_| "weights".into())
        } else {
            eval_mode
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
    let mut results_file = OpenOptions::new().append(true).open("docs/RESULTS.md")?;
    writeln!(
        results_file,
        "\n- Bridge eval `{}`: regime `{}`; report `{}`.",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        report.regime,
        report_path.display()
    )?;
    println!("Evaluation report: {}", report_path.display());
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
    }
}
