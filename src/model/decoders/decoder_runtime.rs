use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;

pub trait LocalDecoderRuntime {
    #[allow(dead_code)]
    fn is_available(&self) -> bool;
    fn generate(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String>;

    /// Stream generated text in chunks (e.g. for SSE). Default: run generate() and call on_chunk once with full result.
    fn generate_stream(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let full = self.generate(prompt, action, conditioning, max_new_tokens)?;
        if !full.is_empty() {
            on_chunk(&full);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RlmDecoderConfig {
    max_units: usize,
    max_depth: usize,
    max_ops: usize,
    chunk_chars: usize,
    min_chars: usize,
    leaf_tokens: usize,
    program_tokens: usize,
    root_prefix_chars: usize,
    memory_chars: usize,
    model_program: bool,
}

impl RlmDecoderConfig {
    fn from_env(max_new_tokens: usize) -> Self {
        let max_units = env_usize("TOFY_DECODER_RLM_MAX_UNITS", 8).max(1);
        Self {
            max_units,
            max_depth: env_usize("TOFY_DECODER_RLM_MAX_DEPTH", 3).max(1),
            max_ops: env_usize(
                "TOFY_DECODER_RLM_MAX_OPS",
                max_units.saturating_mul(3).saturating_add(8),
            )
            .max(4),
            chunk_chars: env_usize("TOFY_DECODER_RLM_CHUNK_CHARS", 2400).max(512),
            min_chars: env_usize("TOFY_DECODER_RLM_MIN_CHARS", 3600).max(256),
            leaf_tokens: env_usize("TOFY_DECODER_RLM_LEAF_TOKENS", 256)
                .max(32)
                .min(max_new_tokens.max(1)),
            program_tokens: env_usize("TOFY_DECODER_RLM_PROGRAM_TOKENS", 192).max(32),
            root_prefix_chars: env_usize("TOFY_DECODER_RLM_ROOT_PREFIX_CHARS", 1200).max(128),
            memory_chars: env_usize("TOFY_DECODER_RLM_MEMORY_CHARS", 2800).max(256),
            model_program: env_bool("TOFY_DECODER_RLM_MODEL_PROGRAM", false),
        }
    }
}

#[derive(Debug, Clone)]
struct RlmWorkUnit {
    index: usize,
    start_char: usize,
    end_char: usize,
    text: String,
}

#[derive(Debug, Clone)]
enum RlmProgramOp {
    Unit {
        index: usize,
        var: String,
    },
    Peek {
        start: usize,
        len: usize,
        var: String,
    },
    SubRlm {
        input_var: String,
        output_var: String,
    },
    Append {
        var: String,
    },
    Final,
}

#[derive(Debug, Clone)]
struct RlmDecoderEnvironment {
    prompt: String,
    action: String,
    units: Vec<RlmWorkUnit>,
    vars: HashMap<String, String>,
    outputs: Vec<String>,
    trace: Vec<String>,
}

impl RlmDecoderEnvironment {
    fn new(prompt: &str, action: &str, cfg: &RlmDecoderConfig) -> Self {
        Self {
            prompt: prompt.to_string(),
            action: action.to_string(),
            units: semantic_work_units(prompt, cfg.chunk_chars, cfg.max_units),
            vars: HashMap::new(),
            outputs: Vec::new(),
            trace: Vec::new(),
        }
    }

    fn metadata(&self, cfg: &RlmDecoderConfig) -> String {
        let mut out = format!(
            "External prompt P: chars={} lines={} action={} work_units={}\n",
            self.prompt.chars().count(),
            self.prompt.lines().count(),
            self.action,
            self.units.len()
        );
        for unit in &self.units {
            out.push_str(&format!(
                "unit[{}] chars={}..{} len={}\n",
                unit.index,
                unit.start_char,
                unit.end_char,
                unit.text.chars().count()
            ));
        }
        out.push_str("prefix:\n");
        out.push_str(&excerpt_chars(&self.prompt, cfg.root_prefix_chars));
        out
    }

    fn store(&mut self, var: &str, value: String) {
        self.vars.insert(var.to_string(), value);
    }

    fn load(&self, var: &str) -> Option<String> {
        self.vars.get(var).cloned()
    }

    fn memory(&self, cfg: &RlmDecoderConfig) -> String {
        let joined = self
            .outputs
            .iter()
            .rev()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n\n");
        excerpt_chars(&joined, cfg.memory_chars)
    }
}

/// Recursive decoder scaffold. The wrapper keeps the full prompt as external state,
/// dispatches bounded work units through the wrapped decoder, and lets the wrapped
/// implementation keep using its native hybrid KV/cache strategy inside each call.
pub struct RlmDecoderRuntime {
    inner: Box<dyn LocalDecoderRuntime>,
}

impl RlmDecoderRuntime {
    pub fn new(inner: Box<dyn LocalDecoderRuntime>) -> Self {
        Self { inner }
    }

    pub fn enabled() -> bool {
        env_bool("TOFY_DECODER_RLM", true)
    }

    pub fn should_wrap_action(action: &str) -> bool {
        if !Self::enabled() {
            return false;
        }
        let action = action.trim().to_ascii_lowercase();
        std::env::var("TOFY_DECODER_RLM_ACTIONS")
            .unwrap_or_else(|_| "code,text,text_reply".to_string())
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .any(|allowed| allowed == action)
    }

    fn should_recurse(prompt: &str, action: &str, cfg: &RlmDecoderConfig) -> bool {
        if !Self::should_wrap_action(action) {
            return false;
        }
        action.eq_ignore_ascii_case("code") || prompt.chars().count() >= cfg.min_chars
    }

    fn generate_rlm(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        let cfg = RlmDecoderConfig::from_env(max_new_tokens);
        if !Self::should_recurse(prompt, action, &cfg) {
            return self
                .inner
                .generate(prompt, action, conditioning, max_new_tokens);
        }
        let mut env = RlmDecoderEnvironment::new(prompt, action, &cfg);
        let mut program = if cfg.model_program {
            self.generate_program(&env, conditioning, &cfg)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        if program.is_empty() {
            program = default_rlm_program(env.units.len());
            env.trace
                .push("root: deterministic recursive program".to_string());
        } else {
            env.trace.push("root: model recursive program".to_string());
        }
        let response = self.execute_program(&mut env, &program, 0, conditioning, &cfg)?;
        if std::env::var("JEPA_DEBUG").is_ok() {
            let _ = writeln!(
                std::io::stderr(),
                "[tofy] decoder rlm: {}",
                env.trace.join(" | ")
            );
        }
        Ok(response)
    }

    fn generate_program(
        &self,
        env: &RlmDecoderEnvironment,
        conditioning: &[f32],
        cfg: &RlmDecoderConfig,
    ) -> Result<Vec<RlmProgramOp>> {
        let prompt = build_program_prompt(env, cfg);
        let raw = self
            .inner
            .generate(&prompt, &env.action, conditioning, cfg.program_tokens)?;
        Ok(parse_rlm_program(&raw))
    }

    fn execute_program(
        &self,
        env: &mut RlmDecoderEnvironment,
        program: &[RlmProgramOp],
        depth: usize,
        conditioning: &[f32],
        cfg: &RlmDecoderConfig,
    ) -> Result<String> {
        for op in program.iter().take(cfg.max_ops) {
            match op {
                RlmProgramOp::Unit { index, var } => {
                    if let Some(unit) = env.units.get(*index) {
                        env.store(var, unit.text.clone());
                        env.trace
                            .push(format!("depth={depth} UNIT {index} AS {var}"));
                    }
                }
                RlmProgramOp::Peek { start, len, var } => {
                    let value = env.prompt.chars().skip(*start).take(*len).collect();
                    env.store(var, value);
                    env.trace
                        .push(format!("depth={depth} PEEK {start} {len} AS {var}"));
                }
                RlmProgramOp::SubRlm {
                    input_var,
                    output_var,
                } => {
                    let Some(input) = env.load(input_var) else {
                        continue;
                    };
                    let output = self.invoke_sub_rlm(env, &input, depth + 1, conditioning, cfg)?;
                    env.store(output_var, output);
                    env.trace
                        .push(format!("depth={depth} SUB_RLM {input_var} AS {output_var}"));
                }
                RlmProgramOp::Append { var } => {
                    if let Some(value) = env.load(var).filter(|value| !value.trim().is_empty()) {
                        env.outputs.push(value.trim().to_string());
                    }
                    env.trace.push(format!("depth={depth} APPEND {var}"));
                }
                RlmProgramOp::Final => {
                    env.trace.push(format!("depth={depth} FINAL"));
                    break;
                }
            }
        }
        Ok(env.outputs.join("\n\n"))
    }

    fn invoke_sub_rlm(
        &self,
        env: &RlmDecoderEnvironment,
        input: &str,
        depth: usize,
        conditioning: &[f32],
        cfg: &RlmDecoderConfig,
    ) -> Result<String> {
        let nested = semantic_work_units(input, cfg.chunk_chars, cfg.max_units);
        if depth < cfg.max_depth && nested.len() > 1 {
            let mut child = RlmDecoderEnvironment {
                prompt: input.to_string(),
                action: env.action.clone(),
                units: nested,
                vars: HashMap::new(),
                outputs: Vec::new(),
                trace: Vec::new(),
            };
            let program = default_rlm_program(child.units.len());
            return self.execute_program(&mut child, &program, depth, conditioning, cfg);
        }
        let leaf_prompt = build_leaf_prompt(env, input, depth, cfg);
        self.inner
            .generate(&leaf_prompt, &env.action, conditioning, cfg.leaf_tokens)
    }
}

impl LocalDecoderRuntime for RlmDecoderRuntime {
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn generate(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        self.generate_rlm(prompt, action, conditioning, max_new_tokens)
    }
}

/// Placeholder backend used when no local decoder runtime is available.
pub struct StubLocalDecoder;

impl StubLocalDecoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StubLocalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalDecoderRuntime for StubLocalDecoder {
    fn is_available(&self) -> bool {
        false
    }

    fn generate(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        Ok(format!(
            "[decoder_unavailable action={action} cond_dim={} max_new_tokens={max_new_tokens}] {prompt}",
            conditioning.len()
        ))
    }
}

/// Local decoder backend using llama.cpp CLI (`llama-cli`) and GGUF models.
pub struct LlamaCppDecoder {
    bin: String,
    model_path: PathBuf,
    ctx_size: usize,
    gpu_layers: i32,
    temperature: f32,
    repeat_penalty: f32,
}

impl LlamaCppDecoder {
    pub fn try_new() -> Result<Self> {
        // Prefer llama-completion for non-interactive use (llama-cli often rejects --no-conversation).
        let bin =
            std::env::var("JEPA_DECODER_BIN").unwrap_or_else(|_| "llama-completion".to_string());
        let model_path = if let Ok(p) = std::env::var("JEPA_DECODER_MODEL") {
            PathBuf::from(p)
        } else {
            discover_gguf_model(Path::new("models"))?
        };
        let ctx_size = std::env::var("JEPA_DECODER_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096usize);
        let gpu_layers = std::env::var("JEPA_DECODER_NGL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(99i32);
        let temperature = std::env::var("JEPA_DECODER_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.7f32);
        let repeat_penalty = std::env::var("JEPA_DECODER_REPEAT_PENALTY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.12f32);

        if !model_path.exists() {
            return Err(anyhow!(
                "decoder model not found at {:?}; set JEPA_DECODER_MODEL or place .gguf under ./models",
                model_path
            ));
        }
        Ok(Self {
            bin,
            model_path,
            ctx_size,
            gpu_layers,
            temperature,
            repeat_penalty,
        })
    }
}

impl LocalDecoderRuntime for LlamaCppDecoder {
    fn is_available(&self) -> bool {
        true
    }

    fn generate(
        &self,
        prompt: &str,
        action: &str,
        _conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        let full_prompt = format!(
            "System: You are Tofy, a JEPA-style dialog-transition agent. Action={action}. Reply directly to the user.\nUser: {prompt}\nAssistant:"
        );
        let output = Command::new(&self.bin)
            .arg("-m")
            .arg(&self.model_path)
            .arg("-p")
            .arg(&full_prompt)
            .arg("-n")
            .arg(max_new_tokens.to_string())
            .arg("-c")
            .arg(self.ctx_size.to_string())
            .arg("-ngl")
            .arg(self.gpu_layers.to_string())
            .arg("--temp")
            .arg(self.temperature.to_string())
            .arg("--repeat-penalty")
            .arg(self.repeat_penalty.to_string())
            .arg("--simple-io")
            .arg("-r")
            .arg("\nUser:")
            .arg("-r")
            .arg("\n>")
            .output()
            .with_context(|| {
                format!(
                    "failed to run '{}' with model {:?}; install llama.cpp or set JEPA_DECODER_BIN",
                    self.bin, self.model_path
                )
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "decoder process failed (status={}): {}",
                output.status,
                stderr.trim()
            ));
        }
        let raw = String::from_utf8_lossy(&output.stdout);
        if std::env::var("JEPA_DEBUG").is_ok() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            print_speed_lines(&raw);
            print_speed_lines(&stderr);
        }
        let text = clean_generated_text(&raw);
        if text.is_empty() {
            return Err(anyhow!("decoder returned empty output (action={action})"));
        }
        Ok(text)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        action: &str,
        _conditioning: &[f32],
        max_new_tokens: usize,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let full_prompt = format!(
            "System: You are Tofy, a JEPA-style dialog-transition agent. Action={action}. Reply directly to the user.\nUser: {prompt}\nAssistant:"
        );
        let debug = std::env::var("JEPA_DEBUG").is_ok();
        let mut child = Command::new(&self.bin)
            .arg("-m")
            .arg(&self.model_path)
            .arg("-p")
            .arg(&full_prompt)
            .arg("-n")
            .arg(max_new_tokens.to_string())
            .arg("-c")
            .arg(self.ctx_size.to_string())
            .arg("-ngl")
            .arg(self.gpu_layers.to_string())
            .arg("--temp")
            .arg(self.temperature.to_string())
            .arg("--repeat-penalty")
            .arg(self.repeat_penalty.to_string())
            .arg("--simple-io")
            .arg("-r")
            .arg("\nUser:")
            .arg("-r")
            .arg("\n>")
            .stdout(Stdio::piped())
            .stderr(if debug { Stdio::piped() } else { Stdio::null() })
            .spawn()
            .with_context(|| {
                format!(
                    "failed to run '{}' with model {:?}; install llama.cpp or set JEPA_DECODER_BIN",
                    self.bin, self.model_path
                )
            })?;
        let stderr_rx = match child.stderr.take() {
            Some(mut stderr) => {
                let (tx, rx) = mpsc::sync_channel(0);
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let _ = std::io::Read::read_to_end(&mut stderr, &mut buf);
                    let _ = tx.send(buf);
                });
                Some(rx)
            }
            None => None,
        };
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("decoder stdout not captured"))?;
        let mut reader = std::io::BufReader::new(stdout);
        let mut buf = [0u8; 64];
        let mut buffer = String::new();
        let mut debug_buf = if debug { Some(String::new()) } else { None };
        // Skip banner and prompt echo: only send content after "Assistant:".
        loop {
            let n = reader.read(&mut buf).context("read decoder stdout")?;
            if n == 0 {
                break;
            }
            let s = String::from_utf8_lossy(&buf[..n]);
            buffer.push_str(&s);
            if let Some(ref mut db) = debug_buf {
                db.push_str(&s);
            }
            if let Some(pos) = buffer.find("Assistant:") {
                let start = pos + "Assistant:".len();
                let tail = buffer[start..].trim_start();
                if !tail.is_empty() {
                    on_chunk(tail);
                }
                buffer.clear();
                break;
            }
        }
        // Stream the rest. Exit as soon as we see the stop sequence (decoder may not close stdout).
        loop {
            let n = reader.read(&mut buf).context("read decoder stdout")?;
            if n == 0 {
                break;
            }
            let s = String::from_utf8_lossy(&buf[..n]);
            buffer.push_str(&s);
            if let Some(ref mut db) = debug_buf {
                db.push_str(&s);
            }
            if !s.is_empty() {
                on_chunk(&s);
            }
            // Decoder uses -r "\nUser:" and -r "\n>"; once we see these, generation is done.
            if buffer.contains("\nUser:") || buffer.contains("\n>") {
                break;
            }
        }
        // Reap the child in a background thread so we return immediately and the stream can end.
        // Otherwise child.wait() can block (decoder cleanup) and the client never sees finish.
        let mut child = child;
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        // Do not block stream completion on decoder stderr EOF (can lag after content ends).
        let stderr_str = stderr_rx
            .and_then(|rx| rx.try_recv().ok())
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        let printed_any = stderr_str.as_deref().is_some_and(print_speed_lines)
            || debug_buf.as_ref().is_some_and(|s| print_speed_lines(s));
        if debug && !printed_any {
            let mut stderr = std::io::stderr();
            let dump = stderr_str
                .as_deref()
                .unwrap_or("")
                .lines()
                .take(30)
                .collect::<Vec<_>>()
                .join("\n");
            if !dump.is_empty() {
                let _ = writeln!(
                    stderr,
                    "[tofy] decoder stderr (no t/s line found):\n{}",
                    dump
                );
            } else if let Some(db) = &debug_buf {
                let line_count = db.lines().count();
                let _ = writeln!(
                    stderr,
                    "[tofy] decoder stdout: {} lines (response not printed)",
                    line_count
                );
            }
            let _ = stderr.flush();
        }
        Ok(())
    }
}

fn print_speed_lines(s: &str) -> bool {
    let mut stderr = std::io::stderr();
    let mut any = false;
    for line in s.lines() {
        let t = line.trim();
        let low = t.to_lowercase();
        if low.contains("t/s") || low.contains("tokens/s") {
            let _ = writeln!(stderr, "[tofy] {}", t);
            any = true;
        }
    }
    if any {
        let _ = stderr.flush();
    }
    any
}

fn discover_gguf_model(models_dir: &Path) -> Result<PathBuf> {
    if !models_dir.exists() {
        return Err(anyhow!(
            "models directory {:?} does not exist; place a .gguf there or set JEPA_DECODER_MODEL",
            models_dir
        ));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(models_dir).context("read models directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            candidates.push(path);
        }
    }
    if candidates.is_empty() {
        for entry in std::fs::read_dir(models_dir).context("read models directory")? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                for sub in std::fs::read_dir(&path).with_context(|| format!("read {:?}", path))? {
                    let sub = sub?;
                    let p = sub.path();
                    if p.is_file()
                        && p.extension()
                            .and_then(|e| e.to_str())
                            .map(|e| e.eq_ignore_ascii_case("gguf"))
                            .unwrap_or(false)
                    {
                        candidates.push(p);
                    }
                }
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .gguf model found under {:?}", models_dir))
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn valid_rlm_var(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn parse_as_var(tokens: &[&str], as_pos: usize) -> Option<String> {
    if tokens.get(as_pos)?.eq_ignore_ascii_case("AS") {
        let var = tokens.get(as_pos + 1)?.trim();
        if valid_rlm_var(var) {
            return Some((*var).to_string());
        }
    }
    None
}

fn parse_rlm_program(raw: &str) -> Vec<RlmProgramOp> {
    let cleaned = raw
        .split("```")
        .nth(1)
        .map(str::trim)
        .unwrap_or_else(|| raw.trim());
    let mut ops = Vec::new();
    for line in cleaned.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].to_ascii_uppercase().as_str() {
            "UNIT" if tokens.len() >= 4 => {
                if let (Ok(index), Some(var)) = (tokens[1].parse(), parse_as_var(&tokens, 2)) {
                    ops.push(RlmProgramOp::Unit { index, var });
                }
            }
            "PEEK" if tokens.len() >= 5 => {
                if let (Ok(start), Ok(len), Some(var)) = (
                    tokens[1].parse(),
                    tokens[2].parse(),
                    parse_as_var(&tokens, 3),
                ) {
                    ops.push(RlmProgramOp::Peek { start, len, var });
                }
            }
            "SUB_RLM" if tokens.len() >= 4 => {
                let input_var = tokens[1];
                if let Some(output_var) = parse_as_var(&tokens, 2) {
                    if valid_rlm_var(input_var) {
                        ops.push(RlmProgramOp::SubRlm {
                            input_var: input_var.to_string(),
                            output_var,
                        });
                    }
                }
            }
            "APPEND" if tokens.len() >= 2 && valid_rlm_var(tokens[1]) => {
                ops.push(RlmProgramOp::Append {
                    var: tokens[1].to_string(),
                });
            }
            "FINAL" => ops.push(RlmProgramOp::Final),
            _ => {}
        }
    }
    ops
}

fn default_rlm_program(unit_count: usize) -> Vec<RlmProgramOp> {
    let mut ops = Vec::new();
    for index in 0..unit_count.max(1) {
        let unit_var = format!("unit_{index}");
        let out_var = format!("out_{index}");
        ops.push(RlmProgramOp::Unit {
            index,
            var: unit_var.clone(),
        });
        ops.push(RlmProgramOp::SubRlm {
            input_var: unit_var,
            output_var: out_var.clone(),
        });
        ops.push(RlmProgramOp::Append { var: out_var });
    }
    ops.push(RlmProgramOp::Final);
    ops
}

fn build_program_prompt(env: &RlmDecoderEnvironment, cfg: &RlmDecoderConfig) -> String {
    let mut out = String::new();
    out.push_str("Write a recursive decoder program. The full prompt is external string P, not copied into your context.\n");
    out.push_str("Allowed commands only:\n");
    out.push_str("UNIT <index> AS <var>\n");
    out.push_str("PEEK <start_char> <len_chars> AS <var>\n");
    out.push_str("SUB_RLM <var> AS <out_var>\n");
    out.push_str("APPEND <var>\n");
    out.push_str("FINAL\n\n");
    out.push_str("Use UNIT for semantic work units, SUB_RLM to process each unit, APPEND useful outputs, then FINAL. Return only commands.\n\n");
    out.push_str(&env.metadata(cfg));
    out
}

fn build_leaf_prompt(
    env: &RlmDecoderEnvironment,
    input: &str,
    depth: usize,
    cfg: &RlmDecoderConfig,
) -> String {
    let mut out = String::new();
    out.push_str("<ctx:recursive_decoder>\n");
    out.push_str("The full user prompt is stored externally as P. Process this bounded work unit in the context of the root request.\n");
    out.push_str(&format!("action={} recursion_depth={depth}\n", env.action));
    out.push_str("</ctx:recursive_decoder>\n\n");
    let memory = env.memory(cfg);
    if !memory.trim().is_empty() {
        out.push_str("<ctx:recursive_memory>\n");
        out.push_str(&memory);
        out.push_str("\n</ctx:recursive_memory>\n\n");
    }
    out.push_str("<ctx:root_metadata>\n");
    out.push_str(&env.metadata(cfg));
    out.push_str("\n</ctx:root_metadata>\n\n");
    out.push_str("<ctx:work_unit>\n");
    out.push_str(input);
    out.push_str("\n</ctx:work_unit>\n\n");
    out.push_str("Return the best partial result for this work unit. Do not mention the recursive scaffold unless the user asked about it.");
    out
}

fn semantic_work_units(prompt: &str, chunk_chars: usize, max_units: usize) -> Vec<RlmWorkUnit> {
    let max_units = max_units.max(1);
    let total_chars = prompt.chars().count();
    if total_chars == 0 {
        return vec![RlmWorkUnit {
            index: 0,
            start_char: 0,
            end_char: 0,
            text: String::new(),
        }];
    }
    let mut spans = paragraph_spans(prompt);
    if spans.len() <= 1 {
        spans = line_spans(prompt);
    }

    let mut units = Vec::new();
    let mut current = String::new();
    let mut start = 0usize;
    let mut end = 0usize;
    for (span_start, span_end, span_text) in spans {
        if current.is_empty() {
            start = span_start;
        }
        let would_len = current.chars().count() + span_text.chars().count() + 2;
        if !current.is_empty() && would_len > chunk_chars && units.len() + 1 < max_units {
            units.push(RlmWorkUnit {
                index: units.len(),
                start_char: start,
                end_char: end,
                text: current.trim().to_string(),
            });
            current.clear();
            start = span_start;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&span_text);
        end = span_end;
    }
    if !current.trim().is_empty() {
        units.push(RlmWorkUnit {
            index: units.len(),
            start_char: start,
            end_char: end,
            text: current.trim().to_string(),
        });
    }
    if units.is_empty() {
        units.push(RlmWorkUnit {
            index: 0,
            start_char: 0,
            end_char: total_chars,
            text: prompt.to_string(),
        });
    }
    units
}

fn paragraph_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut char_pos = 0usize;
    let mut buf = String::new();
    for line in text.lines() {
        let line_len = line.chars().count();
        if line.trim().is_empty() {
            if !buf.trim().is_empty() {
                out.push((start, char_pos, buf.trim().to_string()));
                buf.clear();
            }
            char_pos += line_len + 1;
            start = char_pos;
            continue;
        }
        if buf.is_empty() {
            start = char_pos;
        } else {
            buf.push('\n');
        }
        buf.push_str(line);
        char_pos += line_len + 1;
    }
    if !buf.trim().is_empty() {
        out.push((start, char_pos, buf.trim().to_string()));
    }
    out
}

fn line_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut char_pos = 0usize;
    for line in text.lines() {
        let len = line.chars().count();
        if !line.trim().is_empty() {
            out.push((char_pos, char_pos + len, line.to_string()));
        }
        char_pos += len + 1;
    }
    out
}

fn excerpt_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("\n...");
    out
}

fn clean_generated_text(raw: &str) -> String {
    let mut s = strip_ansi(raw)
        .replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .replace("</s>", "")
        .replace('\r', "\n");

    // Keep only content after the last "Assistant:" (drops banner, prompt echo, etc.).
    if let Some((_, tail)) = s.rsplit_once("Assistant:") {
        s = tail.to_string();
    }

    let mut lines = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.chars().all(|c| c == '>') {
            continue;
        }
        // Skip decoder UI lines (stats, exit message).
        if t.starts_with("Prompt:") || t.starts_with("Generation:") || t.starts_with("Exiting") {
            continue;
        }
        let cleaned = t.trim_start_matches('>').trim_start();
        if cleaned.is_empty() || cleaned.chars().all(|c| c == '>') {
            continue;
        }
        if !cleaned.is_empty() {
            lines.push(cleaned.to_string());
        }
    }
    lines.join("\n").trim().to_string()
}

fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if matches!(chars.peek(), Some('[')) {
                chars.next();
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}
