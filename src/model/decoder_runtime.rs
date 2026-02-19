use anyhow::{anyhow, Context, Result};
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

/// Placeholder backend used when no local decoder runtime is available.
pub struct StubLocalDecoder;

impl StubLocalDecoder {
    pub fn new() -> Self {
        Self
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
        let bin = std::env::var("JEPA_DECODER_BIN").unwrap_or_else(|_| "llama-completion".to_string());
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
        conditioning: &[f32],
        max_new_tokens: usize,
    ) -> Result<String> {
        // Conditioning is only injected as text in the prompt; the decoder is not conditioned by the vector in its forward pass.
        let cond_summary = summarize_conditioning(conditioning);
        let full_prompt = format!(
            "System: You are Tofy, a JEPA-style world-model agent (encoder, transition, bridge + decoder). Action={action}. Use the latent condition below to guide your reply (tone, depth, focus)—it encodes desired response style. Do not mention, repeat, or refer to the numbers themselves.\nLatentCondition: {cond_summary}\nUser: {prompt}\nAssistant:"
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
            return Err(anyhow!(
                "decoder returned empty output (action={action}, cond_summary={cond_summary})"
            ));
        }
        Ok(text)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        action: &str,
        conditioning: &[f32],
        max_new_tokens: usize,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let cond_summary = summarize_conditioning(conditioning);
        let full_prompt = format!(
            "System: You are Tofy, a JEPA-style world-model agent (encoder, transition, bridge + decoder). Action={action}. Use the latent condition below to guide your reply (tone, depth, focus)—it encodes desired response style. Do not mention, repeat, or refer to the numbers themselves.\nLatentCondition: {cond_summary}\nUser: {prompt}\nAssistant:"
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
        let mut debug_buf = if debug {
            Some(String::new())
        } else {
            None
        };
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
        let printed_any = stderr_str
            .as_deref()
            .map_or(false, print_speed_lines)
            || debug_buf.as_ref().map_or(false, |s| print_speed_lines(s));
        if debug && !printed_any {
            let mut stderr = std::io::stderr();
            let dump = stderr_str.as_deref().unwrap_or("").lines().take(30).collect::<Vec<_>>().join("\n");
            if !dump.is_empty() {
                let _ = writeln!(stderr, "[tofy] decoder stderr (no t/s line found):\n{}", dump);
            } else if let Some(db) = &debug_buf {
                let line_count = db.lines().count();
                let _ = writeln!(stderr, "[tofy] decoder stdout: {} lines (response not printed)", line_count);
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
                        && p
                            .extension()
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

fn summarize_conditioning(conditioning: &[f32]) -> String {
    if conditioning.is_empty() {
        return "empty".to_string();
    }
    // Prefer chunked summary (8 aspects) so the model gets a short, structured signal to guide style.
    let fmt = std::env::var("JEPA_DECODER_COND_FORMAT").unwrap_or_else(|_| "chunks".to_string());
    let chunked = format_chunked_conditioning(conditioning);
    match fmt.as_str() {
        "stats" => format_stats_only(conditioning),
        "both" => format!("{} {}", chunked, format_stats_only(conditioning)),
        _ => chunked,
    }
}

/// Eight chunk means (aspects) over the condition vector — short, stable signal for the LM.
fn format_chunked_conditioning(conditioning: &[f32]) -> String {
    const N_CHUNKS: usize = 8;
    let chunk_size = conditioning.len().max(1) / N_CHUNKS;
    let mut chunks = Vec::with_capacity(N_CHUNKS);
    for i in 0..N_CHUNKS {
        let start = i * chunk_size;
        let end = if i == N_CHUNKS - 1 {
            conditioning.len()
        } else {
            (i + 1) * chunk_size
        };
        if start >= conditioning.len() {
            chunks.push(0.0);
            continue;
        }
        let slice = &conditioning[start..end.min(conditioning.len())];
        let mean: f64 = slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
        chunks.push(mean);
    }
    let parts: Vec<String> = chunks.iter().map(|c| format!("{:.3}", c)).collect();
    format!("chunks=[{}]", parts.join(","))
}

fn format_stats_only(conditioning: &[f32]) -> String {
    let n = conditioning.len() as f64;
    let mean = conditioning.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = conditioning
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1.0);
    let std = var.sqrt();
    let l2 = conditioning
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    let head: Vec<String> = conditioning
        .iter()
        .take(8)
        .map(|v| format!("{:.3}", v))
        .collect();
    format!(
        "mean={:.4},std={:.4},l2={:.4},head=[{}]",
        mean,
        std,
        l2,
        head.join(",")
    )
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
        let stripped = strip_latent_condition_leak(cleaned);
        if !stripped.is_empty() {
            lines.push(stripped);
        }
    }
    lines.join("\n").trim().to_string()
}

/// Remove phrases where the model echoes the latent condition stats (mean=..., std=..., l2=...).
fn strip_latent_condition_leak(s: &str) -> String {
    let mut out = s.trim().to_string();
    // Remove parenthetical (mean=..., std=..., l2=...) — match first ( then find matching ).
    while let Some(start) = out.find("(mean=") {
        if let Some(end) = out[start..].find(')') {
            let end = start + end + 1;
            let before = out[..start].trim_end().trim_end_matches(',');
            let after = out[end..].trim_start().trim_start_matches(',');
            out = format!("{} {}", before, after).trim().to_string();
            continue;
        }
        break;
    }
    // Remove echoed chunks=[...] (new conditioning format).
    while let Some(start) = out.find("chunks=[") {
        if let Some(end) = out[start..].find(']') {
            let end = start + end + 1;
            let before = out[..start].trim_end().trim_end_matches(',');
            let after = out[end..].trim_start().trim_start_matches(',');
            out = format!("{} {}", before, after).trim().to_string();
            continue;
        }
        break;
    }
    // Strip leading "Based on your latent condition(s) ..., " (with or without stats already removed).
    for prefix in [
        "Based on your latent conditions , ",
        "Based on your latent conditions, ",
        "Based on your latent condition , ",
        "Based on your latent condition, ",
    ] {
        if let Some(rest) = out.strip_prefix(prefix) {
            out = rest.trim().to_string();
            break;
        }
    }
    out.trim().to_string()
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
