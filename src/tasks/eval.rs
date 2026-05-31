use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::model::decoders::{CandleCrossAttnDecoder, DecoderKind, LocalDecoderRuntime};
use crate::tasks::orchestrator::Action;
use crate::tasks::world::AgentEngine;
use crate::util;

const DEFAULT_MAX_NEW_TOKENS: usize = 384;
const DEFAULT_RUST_TIMEOUT_SECS: u64 = 10;
const DEFAULT_GO_TIMEOUT_SECS: u64 = 6;

#[derive(Debug, Clone, Deserialize)]
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
    rlm_used: bool,
    docs_used: bool,
    candidate_count: usize,
    repair_attempts_used: usize,
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
    rlm_used: usize,
    docs_used: usize,
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
    high_world_model_path: Option<PathBuf>,
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
    go_bin: String,
    rust_timeout_secs: u64,
    go_timeout_secs: u64,
    candidates: usize,
    repair_attempts: usize,
    conditioning_pareto: bool,
    condition_budgets: Vec<usize>,
    cross_schedules: Vec<String>,
}

#[derive(Clone)]
struct CandidateEval {
    response: String,
    code: String,
    route_ok: bool,
    constraints_ok: bool,
    compile_ok: bool,
    tests_ok: bool,
    quality_score: i32,
    detail: String,
    repair_attempts_used: usize,
}

fn default_language() -> String {
    "go".to_string()
}

fn default_expected_action() -> String {
    "code".to_string()
}

fn default_max_new_tokens() -> usize {
    DEFAULT_MAX_NEW_TOKENS
}

fn parse_usize_csv(value: &str) -> Result<Vec<usize>> {
    let parsed = value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| anyhow::anyhow!("invalid integer in csv: {s}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if parsed.is_empty() {
        bail!("csv list must not be empty");
    }
    Ok(parsed)
}

pub fn try_run_code_eval(args: &[String]) -> Result<bool> {
    if args.len() < 6 || (args[1] != "--eval-code-assistant" && args[1] != "eval-code-assistant") {
        return Ok(false);
    }
    let cfg = EvalConfig::from_args_after(&args[2..])?;
    run_code_eval(cfg)?;
    Ok(true)
}

pub fn try_run_decoder_only_eval(args: &[String]) -> Result<bool> {
    if args.len() < 4 || (args[1] != "--eval-decoder-only" && args[1] != "eval-decoder-only") {
        return Ok(false);
    }
    let cfg = DecoderOnlyEvalConfig::from_args_after(&args[2..])?;
    run_decoder_only_eval(cfg)?;
    Ok(true)
}

fn set_eval_env_default(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

impl EvalConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --eval-code-assistant <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <suite.jsonl> [max_new_tokens] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_context_slots] [--high-world-model <override>] [--code-decoder <path>] [--code-decoder-vocab <path>] [--ablate-conditioning] [--rustc <bin>] [--rust-timeout-sec <int>] [--candidates <int>] [--repair-attempts <int>] [--conditioning-pareto] [--condition-budgets <csv>] [--cross-schedules <csv>]"
            );
        }
        let mut filtered = Vec::new();
        let mut ablate_conditioning = false;
        let mut code_decoder_path = None;
        let mut code_decoder_vocab_path = None;
        let mut high_world_model_path = None;
        let mut rustc_bin = "rustc".to_string();
        let mut go_bin = "go".to_string();
        let mut rust_timeout_secs = DEFAULT_RUST_TIMEOUT_SECS;
        let mut go_timeout_secs = DEFAULT_GO_TIMEOUT_SECS;
        let mut candidates = 1usize;
        let mut repair_attempts = 2usize;
        let mut conditioning_pareto = false;
        let mut condition_budgets = vec![0, 4, 8, 16, 32, 64];
        let mut cross_schedules = vec![
            "last-only".to_string(),
            "every-3rd".to_string(),
            "every-2nd".to_string(),
            "all".to_string(),
        ];
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--conditioning-pareto" => {
                    conditioning_pareto = true;
                    i += 1;
                }
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
                "--high-world-model" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--high-world-model requires path"))?;
                    high_world_model_path = Some(PathBuf::from(value));
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
                "--go" => {
                    go_bin = args
                        .get(i + 1)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--go requires binary path"))?;
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
                "--go-timeout-sec" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--go-timeout-sec requires integer"))?;
                    go_timeout_secs = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--go-timeout-sec must be integer"))?;
                    i += 2;
                }
                "--candidates" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--candidates requires integer"))?;
                    candidates = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--candidates must be integer"))?;
                    i += 2;
                }
                "--repair-attempts" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--repair-attempts requires integer"))?;
                    repair_attempts = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--repair-attempts must be integer"))?;
                    i += 2;
                }
                "--condition-budgets" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--condition-budgets requires csv"))?;
                    condition_budgets = parse_usize_csv(value)?;
                    i += 2;
                }
                "--cross-schedules" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--cross-schedules requires csv"))?;
                    cross_schedules = value
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if cross_schedules.is_empty() {
                        bail!("--cross-schedules produced no schedules");
                    }
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
            high_world_model_path,
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
            go_bin,
            rust_timeout_secs: rust_timeout_secs.max(1),
            go_timeout_secs: go_timeout_secs.max(1),
            candidates: candidates.max(1),
            repair_attempts,
            conditioning_pareto,
            condition_budgets,
            cross_schedules,
        })
    }
}

#[derive(Clone)]
struct DecoderOnlyEvalConfig {
    decoder_path: PathBuf,
    decoder_vocab_path: PathBuf,
    suite_path: PathBuf,
    max_new_tokens: usize,
    planner_dim: usize,
    num_context_slots: usize,
    rustc_bin: String,
    go_bin: String,
    rust_timeout_secs: u64,
    go_timeout_secs: u64,
    candidates: usize,
}

impl DecoderOnlyEvalConfig {
    fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                "usage: --eval-decoder-only <decoder.safetensors> <decoder_vocab.txt> <suite.jsonl> [max_new_tokens] [planner_dim] [num_context_slots] [--rustc <bin>] [--rust-timeout-sec <int>] [--candidates <int>]"
            );
        }
        let mut filtered = Vec::new();
        let mut rustc_bin = "rustc".to_string();
        let mut go_bin = "go".to_string();
        let mut rust_timeout_secs = DEFAULT_RUST_TIMEOUT_SECS;
        let mut go_timeout_secs = DEFAULT_GO_TIMEOUT_SECS;
        let mut candidates = 1usize;
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--rustc" => {
                    rustc_bin = args
                        .get(i + 1)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--rustc requires binary path"))?;
                    i += 2;
                }
                "--go" => {
                    go_bin = args
                        .get(i + 1)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--go requires binary path"))?;
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
                "--go-timeout-sec" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--go-timeout-sec requires integer"))?;
                    go_timeout_secs = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--go-timeout-sec must be integer"))?;
                    i += 2;
                }
                "--candidates" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--candidates requires integer"))?;
                    candidates = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--candidates must be integer"))?;
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        Ok(Self {
            decoder_path: PathBuf::from(&filtered[0]),
            decoder_vocab_path: PathBuf::from(&filtered[1]),
            suite_path: PathBuf::from(&filtered[2]),
            max_new_tokens: filtered.get(3).and_then(|v| v.parse().ok()).unwrap_or(384),
            planner_dim: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(640),
            num_context_slots: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(64),
            rustc_bin,
            go_bin,
            rust_timeout_secs: rust_timeout_secs.max(1),
            go_timeout_secs: go_timeout_secs.max(1),
            candidates: candidates.max(1),
        })
    }
}

fn run_code_eval(cfg: EvalConfig) -> Result<()> {
    set_eval_env_default("TOFY_DECODER_RLM", "0");
    set_eval_env_default("TOFY_LATENT_REASONING", "0");
    let default_eval_temp = if cfg.candidates > 1 { "0.35" } else { "0" };
    set_eval_env_default("JEPA_DECODER_TEMP", default_eval_temp);

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
        cfg.high_world_model_path.as_ref(),
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

    println!("Code-first eval suite");
    println!("suite: {:?}", cfg.suite_path);
    println!("tasks: {}", tasks.len());
    println!("run dir: {}", run_dir);
    match engine.high_world_model_path() {
        Some(path) => println!("high_world: integrated planner {}", path.display()),
        None => println!("high_world: unavailable; train the integrated high-world stage"),
    }
    println!(
        "search: candidates={} repair_attempts={} temp={}",
        cfg.candidates,
        cfg.repair_attempts,
        std::env::var("JEPA_DECODER_TEMP").unwrap_or_else(|_| default_eval_temp.to_string())
    );

    let pareto_points = if cfg.conditioning_pareto {
        cfg.condition_budgets
            .iter()
            .flat_map(|&budget| {
                cfg.cross_schedules
                    .iter()
                    .map(move |schedule| (budget, schedule.clone()))
            })
            .collect::<Vec<_>>()
    } else {
        vec![(
            std::env::var("TOFY_DECODER_CONDITION_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(cfg.num_latent_tokens),
            std::env::var("TOFY_DECODER_CROSS_ATTN_SCHEDULE").unwrap_or_else(|_| "all".to_string()),
        )]
    };

    let mut pareto_rows = Vec::new();
    for (budget, schedule) in pareto_points {
        std::env::set_var("TOFY_DECODER_CONDITION_BUDGET", budget.to_string());
        std::env::set_var("TOFY_DECODER_CROSS_ATTN_SCHEDULE", &schedule);
        let label = format!("budget_{budget}_schedule_{}", schedule.replace('-', "_"));
        let results_name = if cfg.conditioning_pareto {
            format!("results_{label}.jsonl")
        } else {
            "results.jsonl".to_string()
        };
        let mut results_file = File::create(run_path.join(results_name))?;
        println!("\npareto_point: condition_budget={budget} cross_schedule={schedule}");
        let summary =
            evaluate_code_suite_once(&engine, &cfg, &scratch_dir, &tasks, &mut results_file)?;
        let summary_text = code_eval_summary_text(&summary);
        fs::write(run_path.join(format!("summary_{label}.txt")), &summary_text)?;
        pareto_rows.push(format!(
            "{budget},{schedule},{:.4},{:.4},{:.4},{:.4}",
            summary.pass_ok as f32 / summary.task_count.max(1) as f32,
            summary.constraints_ok as f32 / summary.task_count.max(1) as f32,
            summary.compile_ok as f32 / summary.task_count.max(1) as f32,
            summary.tests_ok as f32 / summary.task_count.max(1) as f32,
        ));
        if !cfg.conditioning_pareto {
            fs::write(run_path.join("summary.txt"), &summary_text)?;
            println!("\n{}", summary_text);
        }
    }
    if cfg.conditioning_pareto {
        let mut csv =
            "condition_budget,cross_schedule,suite_pass_rate,constraint_pass_rate,compile_rate,test_pass_rate\n"
                .to_string();
        csv.push_str(&pareto_rows.join("\n"));
        csv.push('\n');
        fs::write(run_path.join("conditioning_pareto.csv"), &csv)?;
        println!("\n{csv}");
    }
    Ok(())
}

fn evaluate_code_suite_once(
    engine: &AgentEngine,
    cfg: &EvalConfig,
    scratch_dir: &Path,
    tasks: &[CodeEvalTask],
    results_file: &mut File,
) -> Result<CodeEvalSummary> {
    let mut summary = CodeEvalSummary {
        task_count: tasks.len(),
        ..CodeEvalSummary::default()
    };
    for task in tasks {
        let started = Instant::now();
        let predicted_action = engine.predict_action(&task.prompt)?;
        let rlm_used = engine.uses_recursive_code_generation(&task.prompt, predicted_action);
        let docs_used = engine.uses_fetch_docs(&task.prompt, predicted_action);
        let expected_action = parse_expected_action(&task.expected_action)?;
        let route_ok = predicted_action == expected_action
            || (expected_action == Action::Code && predicted_action == Action::FetchDocs);
        let best = evaluate_best_candidate(engine, cfg, scratch_dir, task, route_ok)?;
        let pass = best.route_ok && best.constraints_ok && best.compile_ok && best.tests_ok;
        summary.route_ok += usize::from(route_ok);
        summary.rlm_used += usize::from(rlm_used);
        summary.docs_used += usize::from(docs_used);
        summary.constraints_ok += usize::from(best.constraints_ok);
        summary.compile_ok += usize::from(best.compile_ok);
        summary.tests_ok += usize::from(best.tests_ok);
        summary.pass_ok += usize::from(pass);
        let result = CodeEvalTaskResult {
            id: task.id.clone(),
            predicted_action: action_name(predicted_action).to_string(),
            expected_action: task.expected_action.clone(),
            rlm_used,
            docs_used,
            candidate_count: cfg.candidates,
            repair_attempts_used: best.repair_attempts_used,
            route_ok: best.route_ok,
            constraints_ok: best.constraints_ok,
            compile_ok: best.compile_ok,
            tests_ok: best.tests_ok,
            pass,
            duration_ms: started.elapsed().as_millis(),
            response_preview: preview_text(&best.response, 240),
            code_preview: preview_text(&best.code, 240),
            detail: preview_text(&best.detail, 600),
            tags: task.tags.clone(),
        };
        writeln!(results_file, "{}", serde_json::to_string(&result)?)?;
        println!(
            "{} route={} rlm={} docs={} constraints={} compile={} tests={} pass={} {}",
            result.id,
            result.route_ok,
            result.rlm_used,
            result.docs_used,
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
    Ok(summary)
}

fn code_eval_summary_text(summary: &CodeEvalSummary) -> String {
    format!(
        "suite_pass_rate={:.4}\nroute_code_acc={:.4}\nrlm_used_rate={:.4}\ndocs_used_rate={:.4}\nconstraint_pass_rate={:.4}\ncompile_rate={:.4}\ntest_pass_rate={:.4}\ntasks={}\n",
        summary.pass_ok as f32 / summary.task_count.max(1) as f32,
        summary.route_ok as f32 / summary.task_count.max(1) as f32,
        summary.rlm_used as f32 / summary.task_count.max(1) as f32,
        summary.docs_used as f32 / summary.task_count.max(1) as f32,
        summary.constraints_ok as f32 / summary.task_count.max(1) as f32,
        summary.compile_ok as f32 / summary.task_count.max(1) as f32,
        summary.tests_ok as f32 / summary.task_count.max(1) as f32,
        summary.task_count,
    )
}

fn run_decoder_only_eval(cfg: DecoderOnlyEvalConfig) -> Result<()> {
    let default_eval_temp = if cfg.candidates > 1 { "0.35" } else { "0" };
    set_eval_env_default("JEPA_DECODER_TEMP", default_eval_temp);
    let tasks = load_suite(&cfg.suite_path)?;
    if tasks.is_empty() {
        bail!("suite {:?} contains no tasks", cfg.suite_path);
    }

    let decoder = CandleCrossAttnDecoder::new(
        cfg.decoder_path.clone(),
        cfg.decoder_vocab_path.clone(),
        cfg.planner_dim,
        cfg.planner_dim,
        cfg.num_context_slots,
        std::env::var("JEPA_DECODER_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0),
        DecoderKind::CodeSpecialist,
    )?;
    let conditioning = vec![0.0f32; cfg.planner_dim * cfg.num_context_slots];
    let run_dir = util::create_run_dir("decoder_only_eval")?;
    let run_path = PathBuf::from(&run_dir);
    let scratch_dir = run_path.join("scratch");
    fs::create_dir_all(&scratch_dir)?;
    let mut results_file = File::create(run_path.join("results.jsonl"))?;

    println!("Decoder-only code eval suite");
    println!("suite: {:?}", cfg.suite_path);
    println!("tasks: {}", tasks.len());
    println!("run dir: {}", run_dir);
    println!("decoder: {}", cfg.decoder_path.display());
    println!(
        "conditioning: zero context slots={} dim={}",
        cfg.num_context_slots, cfg.planner_dim
    );
    println!(
        "search: candidates={} temp={}",
        cfg.candidates,
        std::env::var("JEPA_DECODER_TEMP").unwrap_or_else(|_| default_eval_temp.to_string())
    );

    let mut summary = CodeEvalSummary {
        task_count: tasks.len(),
        ..CodeEvalSummary::default()
    };
    for task in tasks {
        let started = Instant::now();
        let mut best = None;
        let max_new_tokens = task.max_new_tokens.min(cfg.max_new_tokens);
        for _ in 0..cfg.candidates {
            let prompt = build_code_eval_prompt(&task);
            let response = decoder.generate(&prompt, "code", &conditioning, max_new_tokens)?;
            let candidate = evaluate_candidate_response(
                &scratch_dir,
                &cfg.rustc_bin,
                &cfg.go_bin,
                cfg.rust_timeout_secs,
                cfg.go_timeout_secs,
                &task,
                true,
                response,
                0,
            )?;
            best = Some(select_better_candidate(best, candidate));
        }
        let best = best.context("decoder-only eval produced no candidates")?;
        let pass = best.constraints_ok && best.compile_ok && best.tests_ok;
        summary.route_ok += 1;
        summary.rlm_used += 1;
        summary.constraints_ok += usize::from(best.constraints_ok);
        summary.compile_ok += usize::from(best.compile_ok);
        summary.tests_ok += usize::from(best.tests_ok);
        summary.pass_ok += usize::from(pass);
        let result = CodeEvalTaskResult {
            id: task.id,
            predicted_action: "code".to_string(),
            expected_action: task.expected_action,
            rlm_used: true,
            docs_used: false,
            candidate_count: cfg.candidates,
            repair_attempts_used: 0,
            route_ok: true,
            constraints_ok: best.constraints_ok,
            compile_ok: best.compile_ok,
            tests_ok: best.tests_ok,
            pass,
            duration_ms: started.elapsed().as_millis(),
            response_preview: preview_text(&best.response, 240),
            code_preview: preview_text(&best.code, 240),
            detail: preview_text(&best.detail, 600),
            tags: task.tags,
        };
        writeln!(results_file, "{}", serde_json::to_string(&result)?)?;
        println!(
            "{} constraints={} compile={} tests={} pass={} {}",
            result.id,
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
        "suite_pass_rate={:.4}\nroute_code_acc={:.4}\nrlm_used_rate={:.4}\ndocs_used_rate={:.4}\nconstraint_pass_rate={:.4}\ncompile_rate={:.4}\ntest_pass_rate={:.4}\ntasks={}\n",
        summary.pass_ok as f32 / summary.task_count.max(1) as f32,
        summary.route_ok as f32 / summary.task_count.max(1) as f32,
        summary.rlm_used as f32 / summary.task_count.max(1) as f32,
        summary.docs_used as f32 / summary.task_count.max(1) as f32,
        summary.constraints_ok as f32 / summary.task_count.max(1) as f32,
        summary.compile_ok as f32 / summary.task_count.max(1) as f32,
        summary.tests_ok as f32 / summary.task_count.max(1) as f32,
        summary.task_count,
    );
    fs::write(run_path.join("summary.txt"), &summary_text)?;
    println!("\n{}", summary_text);
    Ok(())
}

fn evaluate_best_candidate(
    engine: &AgentEngine,
    cfg: &EvalConfig,
    scratch_dir: &Path,
    task: &CodeEvalTask,
    route_ok: bool,
) -> Result<CandidateEval> {
    let mut best = None;
    let max_new_tokens = task.max_new_tokens.min(cfg.max_new_tokens);
    let repair_route_ok = matches!(
        parse_expected_action(&task.expected_action),
        Ok(Action::Code)
    );
    for _ in 0..cfg.candidates.max(1) {
        let prompt = build_code_eval_prompt(task);
        let response = engine.generate(&prompt, max_new_tokens, cfg.ablate_conditioning)?;
        let mut candidate = evaluate_candidate_response(
            scratch_dir,
            &cfg.rustc_bin,
            &cfg.go_bin,
            cfg.rust_timeout_secs,
            cfg.go_timeout_secs,
            task,
            route_ok,
            response,
            0,
        )?;
        best = Some(select_better_candidate(best, candidate.clone()));
        for repair_idx in 0..cfg.repair_attempts {
            if candidate.constraints_ok && candidate.compile_ok && candidate.tests_ok {
                break;
            }
            let repair_prompt = build_repair_prompt(task, &candidate.code, &candidate.detail);
            let repaired_response = engine.generate_for_action(
                &repair_prompt,
                Action::Code,
                max_new_tokens,
                cfg.ablate_conditioning,
            )?;
            candidate = evaluate_candidate_response(
                scratch_dir,
                &cfg.rustc_bin,
                &cfg.go_bin,
                cfg.rust_timeout_secs,
                cfg.go_timeout_secs,
                task,
                route_ok || repair_route_ok,
                repaired_response,
                repair_idx + 1,
            )?;
            best = Some(select_better_candidate(best, candidate.clone()));
        }
    }
    best.context("eval produced no candidates")
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate_response(
    scratch_dir: &Path,
    rustc_bin: &str,
    go_bin: &str,
    timeout_secs: u64,
    go_timeout_secs: u64,
    task: &CodeEvalTask,
    route_ok: bool,
    response: String,
    repair_attempts_used: usize,
) -> Result<CandidateEval> {
    let code = extract_code_candidate_for_language(&response, &task.language);
    let (constraints_ok, constraint_detail) = check_constraints(&code, task);
    let quality_score = candidate_quality_score(&code, task);
    let language = task.language.to_ascii_lowercase();
    let (compile_ok, tests_ok, exec_detail) = match language.as_str() {
        "rust" => match run_rust_harness(scratch_dir, rustc_bin, timeout_secs, task, &code) {
            Ok(result) => result,
            Err(err) => (false, false, format!("harness_error: {err}")),
        },
        "go" | "golang" => {
            match run_go_harness(scratch_dir, go_bin, go_timeout_secs, task, &code) {
                Ok(result) => result,
                Err(err) => (false, false, format!("harness_error: {err}")),
            }
        }
        _ => (
            false,
            false,
            format!("unsupported language {}", task.language),
        ),
    };
    let detail = if !constraint_detail.is_empty() {
        constraint_detail
    } else {
        exec_detail
    };
    Ok(CandidateEval {
        response,
        code,
        route_ok,
        constraints_ok,
        compile_ok,
        tests_ok,
        quality_score,
        detail,
        repair_attempts_used,
    })
}

fn select_better_candidate(
    current: Option<CandidateEval>,
    challenger: CandidateEval,
) -> CandidateEval {
    match current {
        None => challenger,
        Some(existing) => {
            if candidate_rank(&challenger) > candidate_rank(&existing) {
                challenger
            } else {
                existing
            }
        }
    }
}

fn candidate_rank(candidate: &CandidateEval) -> i32 {
    128 * i32::from(
        candidate.route_ok
            && candidate.constraints_ok
            && candidate.compile_ok
            && candidate.tests_ok,
    ) + 64 * i32::from(candidate.tests_ok)
        + 32 * i32::from(candidate.compile_ok)
        + 16 * i32::from(candidate.constraints_ok)
        + 8 * i32::from(candidate.route_ok)
        + candidate.quality_score.clamp(-8, 8)
        - candidate.repair_attempts_used as i32
}

fn build_repair_prompt(task: &CodeEvalTask, previous_code: &str, failure_detail: &str) -> String {
    let language = task.language.trim();
    let language_name =
        if language.eq_ignore_ascii_case("go") || language.eq_ignore_ascii_case("golang") {
            "Go"
        } else {
            "Rust"
        };
    let fence = if language_name == "Go" { "go" } else { "rust" };
    let constraints = constraints_prompt_block(task);
    format!(
        "Return only corrected {language_name} code.\nFix the previous attempt using the compiler feedback.\n\nOriginal request:\n{}\n{constraints}\nPrevious attempt:\n```{fence}\n{}\n```\n\nCompiler feedback:\n{}\n\nRules:\n- Keep the exact requested function name and signature.\n- Return only compilable {language_name} code.\n- Do not add markdown fences or explanation.\n",
        task.prompt,
        previous_code,
        failure_detail
    )
}

fn build_code_eval_prompt(task: &CodeEvalTask) -> String {
    let language = task.language.trim();
    let language_name =
        if language.eq_ignore_ascii_case("go") || language.eq_ignore_ascii_case("golang") {
            "Go"
        } else {
            "Rust"
        };
    format!(
        "Return only {language_name} code. Do not use markdown fences or explanation.\n{constraints}Task:\n{}",
        task.prompt,
        constraints = constraints_prompt_block(task)
    )
}

fn constraints_prompt_block(task: &CodeEvalTask) -> String {
    let mut out = String::new();
    if !task.must_contain.is_empty() {
        out.push_str("Required exact substrings:\n");
        for value in &task.must_contain {
            out.push_str("- ");
            out.push_str(value);
            out.push('\n');
        }
    }
    if !task.must_not_contain.is_empty() {
        out.push_str("Forbidden substrings:\n");
        for value in &task.must_not_contain {
            out.push_str("- ");
            out.push_str(value);
            out.push('\n');
        }
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
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
        "fetch_docs" | "docs" => Ok(Action::FetchDocs),
        other => bail!("unsupported expected action {:?}", other),
    }
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::TextReply => "text_reply",
        Action::Code => "code",
        Action::Done => "done",
        Action::FetchDocs => "fetch_docs",
    }
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let mut out = text.replace('\n', "\\n");
    if out.chars().count() > max_chars {
        out = out.chars().take(max_chars).collect();
        out.push_str("...");
    }
    out
}

fn extract_code_candidate_for_language(response: &str, language: &str) -> String {
    let cleaned =
        crate::model::decoders::decoder_candle_runtime::clean_candle_decoder_output(response);
    let wanted = language.to_ascii_lowercase();
    let wanted_aliases = match wanted.as_str() {
        "go" | "golang" => &["go", "golang"][..],
        "rust" => &["rust", "rs"][..],
        _ => &[wanted.as_str()][..],
    };
    let mut best_lang = None;
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
        if looks_like_lang
            && wanted_aliases
                .iter()
                .any(|alias| first.eq_ignore_ascii_case(alias))
            && best_lang.is_none()
        {
            best_lang = Some(body.trim().to_string());
        }
        if best_any.is_none() {
            best_any = Some(body.trim().to_string());
        }
    }
    best_lang
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

fn candidate_quality_score(code: &str, task: &CodeEvalTask) -> i32 {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return -8;
    }
    let mut score = 0;
    score += task
        .must_contain
        .iter()
        .filter(|needle| trimmed.contains(needle.as_str()))
        .count() as i32;
    score -= task
        .must_not_contain
        .iter()
        .filter(|needle| trimmed.contains(needle.as_str()))
        .count() as i32
        * 2;
    if delimiters_look_balanced(trimmed) {
        score += 2;
    } else {
        score -= 2;
    }
    if trimmed.contains("```") || trimmed.contains("Return only") {
        score -= 2;
    }
    let language = task.language.to_ascii_lowercase();
    if language == "go" || language == "golang" {
        if trimmed.contains("func ") {
            score += 1;
        }
        if trimmed.contains("package ") {
            score -= 1;
        }
    }
    score
}

fn delimiters_look_balanced(code: &str) -> bool {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut quote = '\0';
    let mut escaped = false;
    for ch in code.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' | '\'' | '`' => {
                in_string = true;
                quote = ch;
            }
            '(' | '[' | '{' => stack.push(ch),
            ')' if stack.pop() != Some('(') => return false,
            ']' if stack.pop() != Some('[') => return false,
            '}' if stack.pop() != Some('{') => return false,
            ')' | ']' | '}' => {}
            _ => {}
        }
    }
    !in_string && stack.is_empty()
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

fn run_go_harness(
    scratch_dir: &Path,
    go_bin: &str,
    timeout_secs: u64,
    task: &CodeEvalTask,
    code: &str,
) -> Result<(bool, bool, String)> {
    let stem = sanitize_id(&task.id);
    let task_dir = scratch_dir.join(format!("{stem}_go"));
    if task_dir.exists() {
        fs::remove_dir_all(&task_dir)?;
    }
    fs::create_dir_all(&task_dir)?;
    fs::write(task_dir.join("go.mod"), "module tofy_eval\n\ngo 1.22\n")?;
    let source = task
        .harness_template
        .replace("{{code}}", &sanitize_go_submission(code));
    fs::write(task_dir.join("solution_test.go"), source)?;

    let compile_output = run_command_with_timeout(
        Command::new(go_bin)
            .arg("test")
            .arg("-c")
            .arg("-o")
            .arg("solution.test")
            .arg(".")
            .current_dir(&task_dir)
            .env("GOCACHE", scratch_dir.join("go-build-cache"))
            .env("GOMODCACHE", scratch_dir.join("go-mod-cache"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
        Duration::from_secs(timeout_secs),
    )?;
    if !compile_output.status.success() {
        return Ok((false, false, summarize_output("compile", &compile_output)));
    }

    let test_output = run_command_with_timeout(
        Command::new(task_dir.join("solution.test"))
            .arg("-test.v")
            .current_dir(&task_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
        Duration::from_secs(timeout_secs),
    )?;
    let tests_ok = test_output.status.success();
    Ok((true, tests_ok, summarize_output("tests", &test_output)))
}

fn sanitize_go_submission(code: &str) -> String {
    let mut lines = code.lines().peekable();
    while lines
        .peek()
        .is_some_and(|line| line.trim().is_empty() || line.trim_start().starts_with("//"))
    {
        lines.next();
    }
    let mut out = lines.collect::<Vec<_>>().join("\n");
    let package_re = regex::Regex::new(r"(?m)^\s*package\s+\w+\s*$").unwrap();
    out = package_re.replace_all(&out, "").to_string();
    out.trim().to_string()
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

#[cfg(test)]
mod tests {
    use super::{
        build_code_eval_prompt, build_repair_prompt, candidate_quality_score,
        delimiters_look_balanced, CodeEvalTask,
    };

    fn go_task() -> CodeEvalTask {
        CodeEvalTask {
            id: "add".to_string(),
            prompt:
                "Return only Go code. Implement exactly this function:\nfunc Add(a int, b int) int"
                    .to_string(),
            harness_template: "{{code}}".to_string(),
            language: "go".to_string(),
            expected_action: "code".to_string(),
            max_new_tokens: 64,
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn repair_prompt_matches_training_feedback_schema() {
        let prompt =
            build_repair_prompt(&go_task(), "func Add() int { return 0 }", "compile failed");
        assert!(prompt.contains("Compiler feedback:\ncompile failed"));
        assert!(prompt.contains("Original request:\nReturn only Go code."));
        assert!(prompt.contains("Return only compilable Go code."));
        assert!(!prompt.contains("<tool:"));
        assert!(!prompt.contains("<ctx:"));
        assert!(!prompt.contains("<ctx:failure_feedback>"));
    }

    #[test]
    fn eval_prompt_reinforces_code_only_constraints() {
        let mut task = go_task();
        task.must_contain = vec!["func Add".to_string()];
        task.must_not_contain = vec!["panic(".to_string()];

        let prompt = build_code_eval_prompt(&task);

        assert!(prompt.contains("Return only Go code."));
        assert!(prompt.contains("Do not use markdown fences"));
        assert!(prompt.contains("Required exact substrings:\n- func Add"));
        assert!(prompt.contains("Forbidden substrings:\n- panic("));
    }

    #[test]
    fn candidate_quality_rewards_required_balanced_code() {
        let mut task = go_task();
        task.must_contain = vec!["func Add".to_string()];
        task.must_not_contain = vec!["panic(".to_string()];

        let good = candidate_quality_score("func Add(a int, b int) int { return a + b }", &task);
        let bad = candidate_quality_score("func Add(a int, b int) int { panic(\"x\")", &task);

        assert!(good > bad);
        assert!(delimiters_look_balanced("func Add() int { return 1 }"));
        assert!(!delimiters_look_balanced("func Add() int { return 1 "));
    }
}
