use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::tasks::orchestrator::Action;
use crate::tasks::world::AgentEngine;
use crate::util;

const DEFAULT_MAX_NEW_TOKENS: usize = 384;
const DEFAULT_RUST_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Deserialize)]
struct CodeEvalTask {
    id: String,
    prompt: String,
    harness_template: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_expected_action")]
    expected_action: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    must_contain: Vec<String>,
    #[serde(default)]
    must_not_contain: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CodeEvalTaskResult {
    id: String,
    predicted_action: String,
    expected_action: String,
    route_ok: bool,
    constraints_ok: bool,
    compile_ok: bool,
    tests_ok: bool,
    pass: bool,
    duration_ms: u128,
    response_preview: String,
    code_preview: String,
    detail: String,
    tags: Vec<String>,
}

#[derive(Default)]
struct CodeEvalSummary {
    task_count: usize,
    route_ok: usize,
    constraints_ok: usize,
    compile_ok: usize,
    tests_ok: usize,
    pass_ok: usize,
}

#[derive(Clone)]
struct EvalConfig {
    encoder_model_path: PathBuf,
    encoder_vocab_path: PathBuf,
    world_model_path: PathBuf,
    suite_path: PathBuf,
    max_new_tokens: usize,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    num_latent_tokens: usize,
    ablate_conditioning: bool,
    code_decoder_path: Option<PathBuf>,
    code_decoder_vocab_path: Option<PathBuf>,
    rustc_bin: String,
    rust_timeout_secs: u64,
}

fn default_language() -> String {
    "rust".to_string()
}

fn default_expected_action() -> String {
    "code".to_string()
}

fn default_max_new_tokens() -> usize {
    DEFAULT_MAX_NEW_TOKENS
}

pub fn try_run_code_eval(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--eval-code-assistant" && args[1] != "eval-code-assistant") {
        return Ok(false);
    }
    let cfg = EvalConfig::from_args_after(&args[2..])?;
    run_code_eval(cfg)?;
    Ok(true)
}

impl EvalConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --eval-code-assistant <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <suite.jsonl> [max_new_tokens] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--code-decoder <path>] [--code-decoder-vocab <path>] [--ablate-conditioning] [--rustc <bin>] [--rust-timeout-sec <int>]"
            );
        }
        let mut filtered = Vec::new();
        let mut ablate_conditioning = false;
        let mut code_decoder_path = None;
        let mut code_decoder_vocab_path = None;
        let mut rustc_bin = "rustc".to_string();
        let mut rust_timeout_secs = DEFAULT_RUST_TIMEOUT_SECS;
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--ablate-conditioning" => {
                    ablate_conditioning = true;
                    i += 1;
                }
                "--code-decoder" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--code-decoder requires path"))?;
                    code_decoder_path = Some(PathBuf::from(value));
                    i += 2;
                }
                "--code-decoder-vocab" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--code-decoder-vocab requires path"))?;
                    code_decoder_vocab_path = Some(PathBuf::from(value));
                    i += 2;
                }
                "--rustc" => {
                    rustc_bin = args
                        .get(i + 1)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--rustc requires binary path"))?;
                    i += 2;
                }
                "--rust-timeout-sec" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--rust-timeout-sec requires integer"))?;
                    rust_timeout_secs = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--rust-timeout-sec must be integer"))?;
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            suite_path: PathBuf::from(&filtered[3]),
            max_new_tokens: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(384),
            dim: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(64),
            ablate_conditioning,
            code_decoder_path,
            code_decoder_vocab_path,
            rustc_bin,
            rust_timeout_secs: rust_timeout_secs.max(1),
        })
    }
}

fn run_code_eval(cfg: EvalConfig) -> Result<()> {
    if let Some(path) = cfg.code_decoder_path.as_ref() {
        std::env::set_var("JEPA_USE_CANDLE_DECODER", "1");
        std::env::set_var("JEPA_CANDLE_DECODER", path);
        if let Some(vocab_path) = cfg.code_decoder_vocab_path.as_ref() {
            std::env::set_var("JEPA_CANDLE_DECODER_VOCAB", vocab_path);
        }
    }

    let tasks = load_suite(&cfg.suite_path)?;
    if tasks.is_empty() {
        bail!("suite {:?} contains no tasks", cfg.suite_path);
    }

    let engine = AgentEngine::load(
        &cfg.encoder_model_path,
        &cfg.encoder_vocab_path,
        &cfg.world_model_path,
        cfg.dim,
        cfg.max_seq,
        cfg.num_layers,
        cfg.num_heads,
        cfg.bridge_dim,
        cfg.num_latent_tokens,
    )?;

    let run_dir = util::create_run_dir("code_eval")?;
    let run_path = PathBuf::from(&run_dir);
    let scratch_dir = run_path.join("scratch");
    fs::create_dir_all(&scratch_dir)?;
    let mut results_file = File::create(run_path.join("results.jsonl"))?;

    println!("Code-first eval suite");
    println!("suite: {:?}", cfg.suite_path);
    println!("tasks: {}", tasks.len());
    println!("run dir: {}", run_dir);

    let mut summary = CodeEvalSummary {
        task_count: tasks.len(),
        ..CodeEvalSummary::default()
    };

    for task in tasks {
        let started = Instant::now();
        let predicted_action = engine.predict_action(&task.prompt)?;
        let response = engine.generate(
            &task.prompt,
            task.max_new_tokens.min(cfg.max_new_tokens),
            cfg.ablate_conditioning,
        )?;
        let code = extract_code_candidate(&response);
        let route_ok = predicted_action == parse_expected_action(&task.expected_action)?;
        let (constraints_ok, constraint_detail) = check_constraints(&code, &task);
        let (compile_ok, tests_ok, exec_detail) = if task.language.eq_ignore_ascii_case("rust") {
            match run_rust_harness(
                &scratch_dir,
                &cfg.rustc_bin,
                cfg.rust_timeout_secs,
                &task,
                &code,
            ) {
                Ok(result) => result,
                Err(err) => (false, false, format!("harness_error: {err}")),
            }
        } else {
            (
                false,
                false,
                format!("unsupported language {}", task.language),
            )
        };
        let pass = route_ok && constraints_ok && compile_ok && tests_ok;
        summary.route_ok += usize::from(route_ok);
        summary.constraints_ok += usize::from(constraints_ok);
        summary.compile_ok += usize::from(compile_ok);
        summary.tests_ok += usize::from(tests_ok);
        summary.pass_ok += usize::from(pass);

        let detail = if !constraint_detail.is_empty() {
            constraint_detail
        } else {
            exec_detail
        };
        let result = CodeEvalTaskResult {
            id: task.id,
            predicted_action: action_name(predicted_action).to_string(),
            expected_action: task.expected_action,
            route_ok,
            constraints_ok,
            compile_ok,
            tests_ok,
            pass,
            duration_ms: started.elapsed().as_millis(),
            response_preview: preview_text(&response, 240),
            code_preview: preview_text(&code, 240),
            detail: preview_text(&detail, 600),
            tags: task.tags,
        };
        writeln!(results_file, "{}", serde_json::to_string(&result)?)?;
        println!(
            "{} route={} constraints={} compile={} tests={} pass={} {}",
            result.id,
            result.route_ok,
            result.constraints_ok,
            result.compile_ok,
            result.tests_ok,
            result.pass,
            if result.detail.is_empty() {
                String::new()
            } else {
                format!("detail={}", result.detail)
            }
        );
    }

    let summary_text = format!(
        "suite_pass_rate={:.4}\nroute_code_acc={:.4}\nconstraint_pass_rate={:.4}\ncompile_rate={:.4}\ntest_pass_rate={:.4}\ntasks={}\n",
        summary.pass_ok as f32 / summary.task_count.max(1) as f32,
        summary.route_ok as f32 / summary.task_count.max(1) as f32,
        summary.constraints_ok as f32 / summary.task_count.max(1) as f32,
        summary.compile_ok as f32 / summary.task_count.max(1) as f32,
        summary.tests_ok as f32 / summary.task_count.max(1) as f32,
        summary.task_count,
    );
    fs::write(run_path.join("summary.txt"), &summary_text)?;
    println!("\n{}", summary_text);
    Ok(())
}

fn load_suite(path: &Path) -> Result<Vec<CodeEvalTask>> {
    let file = File::open(path).with_context(|| format!("open eval suite {:?}", path))?;
    let reader = BufReader::new(file);
    let mut tasks = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let task = serde_json::from_str::<CodeEvalTask>(trimmed)
            .with_context(|| format!("parse eval suite line {} in {:?}", line_no + 1, path))?;
        tasks.push(task);
    }
    Ok(tasks)
}

fn parse_expected_action(value: &str) -> Result<Action> {
    match value.trim().to_ascii_lowercase().as_str() {
        "code" => Ok(Action::Code),
        "text" | "text_reply" => Ok(Action::TextReply),
        "done" => Ok(Action::Done),
        other => bail!("unsupported expected action {:?}", other),
    }
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::TextReply => "text_reply",
        Action::Code => "code",
        Action::Done => "done",
    }
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let mut out = text.replace('\n', "\\n");
    if out.len() > max_chars {
        out.truncate(max_chars);
        out.push_str("...");
    }
    out
}

fn extract_code_candidate(response: &str) -> String {
    let cleaned =
        crate::model::decoders::decoder_candle_runtime::clean_candle_decoder_output(response);
    let mut best_rust = None;
    let mut best_any = None;
    let parts: Vec<&str> = cleaned.split("```").collect();
    for fenced in parts.iter().skip(1).step_by(2) {
        let mut lines = fenced.lines();
        let first = lines.next().unwrap_or("").trim();
        let looks_like_lang = !first.is_empty()
            && first.len() <= 16
            && first
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '+' | '-'));
        let body = if looks_like_lang {
            lines.collect::<Vec<_>>().join("\n")
        } else {
            fenced.trim().to_string()
        };
        if body.trim().is_empty() {
            continue;
        }
        if looks_like_lang && first.eq_ignore_ascii_case("rust") && best_rust.is_none() {
            best_rust = Some(body.trim().to_string());
        }
        if best_any.is_none() {
            best_any = Some(body.trim().to_string());
        }
    }
    best_rust
        .or(best_any)
        .unwrap_or_else(|| cleaned.trim().to_string())
}

fn check_constraints(code: &str, task: &CodeEvalTask) -> (bool, String) {
    let mut problems = Vec::new();
    for needle in &task.must_contain {
        if !code.contains(needle) {
            problems.push(format!("missing {:?}", needle));
        }
    }
    for needle in &task.must_not_contain {
        if code.contains(needle) {
            problems.push(format!("forbidden {:?}", needle));
        }
    }
    if problems.is_empty() {
        (true, String::new())
    } else {
        (false, problems.join("; "))
    }
}

fn sanitize_id(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "task".to_string()
    } else {
        out
    }
}

fn run_rust_harness(
    scratch_dir: &Path,
    rustc_bin: &str,
    timeout_secs: u64,
    task: &CodeEvalTask,
    code: &str,
) -> Result<(bool, bool, String)> {
    let stem = sanitize_id(&task.id);
    let source_path = scratch_dir.join(format!("{stem}.rs"));
    let bin_path = scratch_dir.join(format!("{stem}.bin"));
    let source = task.harness_template.replace("{{code}}", code);
    fs::write(&source_path, source)?;

    let compile_output = run_command_with_timeout(
        Command::new(rustc_bin)
            .arg("--edition=2021")
            .arg("--test")
            .arg(&source_path)
            .arg("-o")
            .arg(&bin_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
        Duration::from_secs(timeout_secs),
    )?;
    if !compile_output.status.success() {
        return Ok((false, false, summarize_output("compile", &compile_output)));
    }

    let test_output = run_command_with_timeout(
        Command::new(&bin_path)
            .arg("--quiet")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
        Duration::from_secs(timeout_secs),
    )?;
    let tests_ok = test_output.status.success();
    Ok((true, tests_ok, summarize_output("tests", &test_output)))
}

fn summarize_output(stage: &str, output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!(
        "{} status={} stdout={} stderr={}",
        stage,
        output.status,
        preview_text(&stdout, 240),
        preview_text(&stderr, 240)
    );
    combined.trim().to_string()
}

fn run_command_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<Output> {
    let mut child = cmd.spawn().context("spawn command")?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().context("poll child process")? {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(mut pipe) = child.stdout.take() {
                let _ = std::io::Read::read_to_end(&mut pipe, &mut stdout);
            }
            if let Some(mut pipe) = child.stderr.take() {
                let _ = std::io::Read::read_to_end(&mut pipe, &mut stderr);
            }
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            return Err(anyhow::anyhow!(
                "command timed out after {:.1}s",
                timeout.as_secs_f32()
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}
