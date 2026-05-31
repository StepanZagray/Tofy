use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct BashTool {
    name: String,
    description: String,
    script: String,
    cwd: PathBuf,
}

#[derive(Clone, Debug)]
pub struct BashToolRegistry {
    tools: Vec<BashTool>,
    by_name: HashMap<String, usize>,
    workspace: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ToolExecutionResult {
    tool_name: String,
    ok: bool,
    exit_code: Option<i32>,
    timed_out: bool,
    stdout: String,
    stderr: String,
    error: Option<String>,
}

pub fn agentic_decoder_requested() -> bool {
    env_bool("TOFY_AGENTIC_DECODER", false)
        || std::env::var_os("TOFY_TOOL_FILE").is_some()
        || std::env::var_os("TOFY_TOOL_DIR").is_some()
}

impl BashToolRegistry {
    pub fn load_from_env() -> Result<Option<Self>> {
        let explicit_path = std::env::var_os("TOFY_TOOL_FILE").map(PathBuf::from);
        let explicit_dir = std::env::var_os("TOFY_TOOL_DIR").map(PathBuf::from);
        let paths = explicit_path
            .into_iter()
            .chain(explicit_dir)
            .collect::<Vec<_>>();
        let paths = if paths.is_empty() {
            default_tool_paths()
        } else {
            paths
        };
        if paths.is_empty() {
            if agentic_decoder_requested() {
                bail!(
                    "tool-calling decoder requested, but no tool file was found; set TOFY_TOOL_FILE or TOFY_TOOL_DIR"
                );
            }
            return Ok(None);
        };
        let mut tools = Vec::new();
        for path in paths {
            if !path.exists() {
                bail!("tool path {:?} does not exist", path);
            }
            load_tools_from_path(&path, &mut tools)?;
        }
        if tools.is_empty() {
            bail!(
                "tool markdown did not define any executable bash tools; use Pi-style skill frontmatter with a bash fence or !`command`"
            );
        }
        let mut by_name = HashMap::new();
        for (idx, tool) in tools.iter().enumerate() {
            if by_name.insert(tool.name.clone(), idx).is_some() {
                bail!("duplicate tool name {:?}", tool.name);
            }
        }
        Ok(Some(Self {
            tools,
            by_name,
            workspace: std::env::current_dir().context("resolve current workspace")?,
        }))
    }

    pub fn build_prompt(&self, transcript: &str, step: usize, max_steps: usize) -> String {
        let mut out = String::new();
        out.push_str("<ctx:tool_calling_decoder>\n");
        out.push_str("You are Tofy's tool-calling decoder. Use bash tools when you need current repository state, file contents, command output, or checks before answering.\n");
        out.push_str("When a tool is needed, return exactly one tool call and no final answer:\n");
        out.push_str(
            "<tool_call>{\"tool\":\"tool_name\",\"args\":{\"key\":\"value\"}}</tool_call>\n",
        );
        out.push_str("Tool args are exposed to bash as TOFY_ARG_<UPPER_KEY> and are substituted into {{key}} or <key> placeholders with shell quoting.\n");
        out.push_str("After a <tool_result> appears, use that result as context and continue. If the answer is ready, return the final answer directly with no tool_call tag.\n");
        out.push_str(&format!("tool_step={step} max_tool_steps={max_steps}\n"));
        out.push_str("</ctx:tool_calling_decoder>\n\n");
        out.push_str("<ctx:available_bash_tools>\n");
        out.push_str(&self.prompt_catalog());
        out.push_str("</ctx:available_bash_tools>\n\n");
        out.push_str(transcript.trim());
        out
    }

    pub fn build_final_prompt(&self, transcript: &str, max_steps: usize) -> String {
        let mut out = self.build_prompt(transcript, max_steps, max_steps);
        out.push_str("\n\n<ctx:tool_budget_exhausted>\n");
        out.push_str("No more tool calls are available. Return the best final answer using the gathered tool results.\n");
        out.push_str("</ctx:tool_budget_exhausted>");
        out
    }

    pub fn execute(&self, call: &ToolCall) -> ToolExecutionResult {
        let Some(tool) = self
            .by_name
            .get(&normalize_tool_name(&call.name))
            .and_then(|idx| self.tools.get(*idx))
        else {
            return ToolExecutionResult::error(
                &call.name,
                format!(
                    "unknown tool {:?}; available tools: {}",
                    call.name,
                    self.tool_names()
                ),
            );
        };
        let timeout = Duration::from_millis(env_usize("TOFY_TOOL_TIMEOUT_MS", 10_000) as u64);
        let script = render_script(&tool.script, &call.args);
        let mut command = Command::new("bash");
        command
            .arg("-lc")
            .arg(&script)
            .current_dir(&tool.cwd)
            .env("TOFY_WORKSPACE", &self.workspace)
            .env("TOFY_SKILL_DIR", &tool.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in &call.args {
            if let Some(env_name) = arg_env_name(key) {
                command.env(env_name, json_arg_to_env(value));
            }
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                return ToolExecutionResult::error(
                    &tool.name,
                    format!("failed to spawn bash tool: {err}"),
                )
            }
        };
        let start = Instant::now();
        let mut timed_out = false;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if start.elapsed() >= timeout => {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(err) => {
                    return ToolExecutionResult::error(
                        &tool.name,
                        format!("failed to poll bash tool: {err}"),
                    )
                }
            }
        }
        match child.wait_with_output() {
            Ok(output) => ToolExecutionResult {
                tool_name: tool.name.clone(),
                ok: output.status.success() && !timed_out,
                exit_code: output.status.code(),
                timed_out,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                error: None,
            },
            Err(err) => ToolExecutionResult::error(
                &tool.name,
                format!("failed to collect bash tool output: {err}"),
            ),
        }
    }

    fn prompt_catalog(&self) -> String {
        let mut out = String::new();
        for tool in &self.tools {
            out.push_str("- ");
            out.push_str(&tool.name);
            if !tool.description.trim().is_empty() {
                out.push_str(": ");
                out.push_str(&single_line(&tool.description));
            }
            out.push('\n');
            out.push_str("  bash: ");
            out.push_str(&single_line(&excerpt_chars(&tool.script, 240)));
            out.push('\n');
        }
        out
    }

    fn tool_names(&self) -> String {
        self.tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn load_tools_from_path(path: &PathBuf, tools: &mut Vec<BashTool>) -> Result<()> {
    if path.is_dir() {
        for file in markdown_files_in_dir(path)? {
            load_tools_from_path(&file, tools)?;
        }
        return Ok(());
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read tool definitions from {:?}", path))?;
    let cwd = path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    tools.extend(parse_markdown_tools_from_file(
        &raw,
        path.file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("tool"),
        cwd,
    ));
    Ok(())
}

fn markdown_files_in_dir(path: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("read tool dir {:?}", path))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let skill = path.join("SKILL.md");
            if skill.exists() {
                files.push(skill);
            }
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

impl ToolExecutionResult {
    pub fn to_prompt_block(&self, max_chars: usize) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "<tool_result tool=\"{}\" ok=\"{}\"",
            self.tool_name, self.ok
        ));
        if let Some(code) = self.exit_code {
            out.push_str(&format!(" exit_code=\"{code}\""));
        }
        if self.timed_out {
            out.push_str(" timed_out=\"true\"");
        }
        out.push_str(">\n");
        if let Some(error) = &self.error {
            out.push_str("<error>\n");
            out.push_str(error.trim());
            out.push_str("\n</error>\n");
        }
        if !self.stdout.trim().is_empty() {
            out.push_str("<stdout>\n");
            out.push_str(&excerpt_chars(self.stdout.trim(), max_chars));
            out.push_str("\n</stdout>\n");
        }
        if !self.stderr.trim().is_empty() {
            out.push_str("<stderr>\n");
            out.push_str(&excerpt_chars(self.stderr.trim(), max_chars / 2));
            out.push_str("\n</stderr>\n");
        }
        out.push_str("</tool_result>");
        out
    }

    fn error(tool_name: &str, message: String) -> Self {
        Self {
            tool_name: normalize_tool_name(tool_name),
            ok: false,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: String::new(),
            error: Some(message),
        }
    }
}

pub fn parse_tool_call(raw: &str) -> Option<ToolCall> {
    tagged_tool_call(raw)
        .or_else(|| fenced_json_tool_call(raw))
        .or_else(|| first_json_tool_call(raw))
}

pub fn clean_agentic_final(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    for (open, close) in [
        ("<final>", "</final>"),
        ("<answer>", "</answer>"),
        ("<final_answer>", "</final_answer>"),
    ] {
        if let Some(inner) = text
            .split_once(open)
            .and_then(|(_, tail)| tail.split_once(close).map(|(inner, _)| inner))
        {
            text = inner.trim().to_string();
            break;
        }
    }
    text
}

fn parse_markdown_tools_from_file(raw: &str, default_name: &str, cwd: PathBuf) -> Vec<BashTool> {
    let (frontmatter, body) = split_frontmatter(raw);
    let frontmatter_tools =
        parse_frontmatter_tool(frontmatter.as_deref(), body, default_name, cwd.clone());
    let heading_tools = parse_heading_tools(body, cwd);
    if frontmatter_tools.is_empty() {
        heading_tools
    } else {
        frontmatter_tools
    }
}

fn parse_frontmatter_tool(
    frontmatter: Option<&str>,
    body: &str,
    default_name: &str,
    cwd: PathBuf,
) -> Vec<BashTool> {
    if frontmatter.is_none() {
        return Vec::new();
    }
    let meta = frontmatter
        .map(parse_simple_frontmatter)
        .unwrap_or_default();
    let name = meta
        .get("name")
        .map(String::as_str)
        .or_else(|| meta.get("command").map(String::as_str))
        .unwrap_or(default_name);
    let Some(name) = parse_tool_name(name) else {
        return Vec::new();
    };
    let description = meta
        .get("description")
        .cloned()
        .unwrap_or_else(|| first_non_empty_markdown_line(body).unwrap_or_default());
    let scripts = extract_frontmatter_tool_scripts(body);
    if scripts.is_empty() {
        return Vec::new();
    }
    let single = scripts.len() == 1;
    scripts
        .into_iter()
        .filter_map(|(section, script)| {
            if script.trim().is_empty() {
                return None;
            }
            let tool_name = if single {
                name.clone()
            } else {
                normalize_tool_name(&format!("{name}_{section}"))
            };
            Some(BashTool {
                name: tool_name,
                description: if matches!(section.as_str(), "execute" | "usage" | "run" | "command")
                {
                    description.clone()
                } else {
                    format!("{description} Section: {section}.")
                },
                script,
                cwd: cwd.clone(),
            })
        })
        .collect()
}

fn parse_heading_tools(raw: &str, cwd: PathBuf) -> Vec<BashTool> {
    let mut tools = Vec::new();
    let mut current_name: Option<String> = None;
    let mut description = Vec::new();
    let mut in_bash_fence = false;
    let mut in_other_fence = false;
    let mut script = String::new();

    for line in raw.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            let lang = trimmed
                .trim_start_matches("```")
                .trim()
                .to_ascii_lowercase();
            if in_bash_fence {
                if let Some(name) = current_name.take() {
                    tools.push(BashTool {
                        name,
                        description: description.join("\n").trim().to_string(),
                        script: script.trim().to_string(),
                        cwd: cwd.clone(),
                    });
                }
                description.clear();
                script.clear();
                in_bash_fence = false;
                continue;
            }
            if in_other_fence {
                in_other_fence = false;
                continue;
            }
            if matches!(lang.as_str(), "bash" | "sh" | "shell") && current_name.is_some() {
                in_bash_fence = true;
            } else {
                in_other_fence = true;
            }
            continue;
        }
        if in_bash_fence {
            script.push_str(line);
            script.push('\n');
            continue;
        }
        if in_other_fence {
            continue;
        }
        if let Some(heading) = markdown_heading_text(line) {
            current_name = parse_tool_name(heading);
            description.clear();
            script.clear();
            continue;
        }
        if current_name.is_some() {
            description.push(line.trim().to_string());
        }
    }
    tools
        .into_iter()
        .filter(|tool| !tool.script.trim().is_empty())
        .collect()
}

fn split_frontmatter(raw: &str) -> (Option<String>, &str) {
    let Some(rest) = raw.strip_prefix("---\n") else {
        return (None, raw);
    };
    let Some((frontmatter, body)) = rest.split_once("\n---") else {
        return (None, raw);
    };
    let body = body.strip_prefix('\n').unwrap_or(body);
    (Some(frontmatter.to_string()), body)
}

fn parse_simple_frontmatter(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(' ') {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            let value = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            out.insert(key.trim().to_ascii_lowercase(), value);
        }
    }
    out
}

fn extract_frontmatter_tool_scripts(body: &str) -> Vec<(String, String)> {
    let mut sections = bash_sections(body);
    sections.retain(|(heading, _)| !is_setup_section(heading));
    let preferred = sections
        .iter()
        .filter(|(heading, _)| matches!(heading.as_str(), "execute" | "usage" | "run" | "command"))
        .cloned()
        .collect::<Vec<_>>();
    if !preferred.is_empty() {
        return preferred;
    }
    if !sections.is_empty() {
        return sections;
    }
    let bang = extract_bang_commands(body).join("\n");
    if bang.trim().is_empty() {
        Vec::new()
    } else {
        vec![("usage".to_string(), bang)]
    }
}

fn bash_sections(body: &str) -> Vec<(String, String)> {
    let mut current_heading = "usage".to_string();
    let mut in_bash = false;
    let mut script = String::new();
    let mut out = Vec::new();
    for line in body.lines() {
        if let Some(heading) = markdown_heading_text(line) {
            current_heading = normalize_tool_name(heading);
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            let lang = trimmed
                .trim_start_matches("```")
                .trim()
                .to_ascii_lowercase();
            if in_bash {
                out.push((current_heading.clone(), script.trim().to_string()));
                script.clear();
                in_bash = false;
                continue;
            }
            in_bash = matches!(lang.as_str(), "bash" | "sh" | "shell");
            continue;
        }
        if in_bash {
            script.push_str(line);
            script.push('\n');
        }
    }
    out
}

fn is_setup_section(heading: &str) -> bool {
    matches!(
        heading,
        "setup" | "install" | "installation" | "prerequisites" | "requirements"
    )
}

fn extract_bang_commands(body: &str) -> Vec<String> {
    let mut commands = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("!`") {
            if let Some((command, _)) = rest.split_once('`') {
                commands.push(command.trim().to_string());
            }
        }
    }
    commands
}

fn first_non_empty_markdown_line(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with('#')
                && !line.starts_with("```")
                && !line.starts_with("!`")
        })
        .map(ToOwned::to_owned)
        .next()
}

fn markdown_heading_text(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    if (2..=4).contains(&hashes) && trimmed.chars().nth(hashes) == Some(' ') {
        Some(trimmed[hashes..].trim())
    } else {
        None
    }
}

fn parse_tool_name(heading: &str) -> Option<String> {
    let heading = heading.trim().trim_matches('`');
    let name = heading
        .split(|ch: char| ch.is_whitespace() || ch == '(' || ch == ':')
        .find(|part| !part.is_empty())?;
    let normalized = normalize_tool_name(name);
    (!normalized.is_empty()).then_some(normalized)
}

fn tagged_tool_call(raw: &str) -> Option<ToolCall> {
    let start = raw.find("<tool_call")?;
    let tag_end = raw[start..].find('>')? + start + 1;
    let end = raw[tag_end..].find("</tool_call>")? + tag_end;
    parse_tool_call_json(raw[tag_end..end].trim()).ok()
}

fn fenced_json_tool_call(raw: &str) -> Option<ToolCall> {
    let mut in_json = false;
    let mut buf = String::new();
    for line in raw.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            let lang = trimmed
                .trim_start_matches("```")
                .trim()
                .to_ascii_lowercase();
            if in_json {
                if let Ok(call) = parse_tool_call_json(buf.trim()) {
                    return Some(call);
                }
                buf.clear();
                in_json = false;
            } else if lang == "json" {
                in_json = true;
            }
            continue;
        }
        if in_json {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    None
}

fn first_json_tool_call(raw: &str) -> Option<ToolCall> {
    let start = raw.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in raw[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return parse_tool_call_json(&raw[start..start + offset + 1]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_tool_call_json(raw: &str) -> Result<ToolCall> {
    let value: Value = serde_json::from_str(raw).with_context(|| "parse tool_call JSON")?;
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("tool_call JSON must be an object"))?;
    let name = obj
        .get("tool")
        .or_else(|| obj.get("name"))
        .or_else(|| obj.get("tool_name"))
        .and_then(Value::as_str)
        .map(normalize_tool_name)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow!("tool_call JSON is missing string field tool/name"))?;
    let args = match obj.get("args").or_else(|| obj.get("arguments")) {
        Some(Value::Object(args)) => args.clone(),
        Some(Value::String(raw_args)) => serde_json::from_str::<Value>(raw_args)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_else(|| {
                let mut args = Map::new();
                args.insert("input".to_string(), Value::String(raw_args.clone()));
                args
            }),
        Some(value) => {
            let mut args = Map::new();
            args.insert("input".to_string(), value.clone());
            args
        }
        None => Map::new(),
    };
    Ok(ToolCall { name, args })
}

fn default_tool_paths() -> Vec<PathBuf> {
    let mut candidates = [
        "TOOLS.md",
        ".tofy/tools.md",
        ".tofy/tools",
        ".pi/skills",
        ".pi/prompts",
        ".agents/skills",
        "docs/TOOLS.md",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect::<Vec<_>>();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".pi/agent/skills"));
        candidates.push(home.join(".agents/skills"));
    }
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

fn normalize_tool_name(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else if ch == '-' {
                '_'
            } else {
                '\0'
            }
        })
        .filter(|ch| *ch != '\0')
        .collect()
}

fn arg_env_name(key: &str) -> Option<String> {
    let normalized = normalize_tool_name(key);
    if normalized.is_empty() {
        return None;
    }
    Some(format!("TOFY_ARG_{}", normalized.to_ascii_uppercase()))
}

fn json_arg_to_env(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn render_script(script: &str, args: &Map<String, Value>) -> String {
    let mut rendered = script.to_string();
    for (key, value) in args {
        let normalized = normalize_tool_name(key);
        if normalized.is_empty() {
            continue;
        }
        let quoted = shell_quote(&json_arg_to_env(value));
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), &quoted);
        rendered = rendered.replace(&format!("{{{{{normalized}}}}}"), &quoted);
        rendered = rendered.replace(&format!("<{key}>"), &quoted);
        rendered = rendered.replace(&format!("<{normalized}>"), &quoted);
    }
    rendered
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn excerpt_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("\n...<truncated>");
    out
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

#[cfg(test)]
mod tests {
    use super::{parse_markdown_tools_from_file, parse_tool_call};
    use serde_json::Value;
    use std::path::PathBuf;

    #[test]
    fn parses_markdown_bash_tools() {
        let raw = r#"
## read_file
Read a file from the workspace.

```bash
sed -n '1,120p' "$TOFY_ARG_PATH"
```
"#;
        let tools = parse_markdown_tools_from_file(raw, "tools", PathBuf::from("."));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert!(tools[0].script.contains("TOFY_ARG_PATH"));
    }

    #[test]
    fn parses_pi_skill_style_tool() {
        let raw = r#"---
name: repo-status
description: Show repository status.
allowed-tools: bash
---

## Execute

```bash
git status --short
```
"#;
        let tools = parse_markdown_tools_from_file(raw, "repo-status", PathBuf::from("."));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "repo_status");
        assert!(tools[0].script.contains("git status"));
    }

    #[test]
    fn parses_pi_bang_command_style_tool() {
        let raw = r#"---
description: Review current changes.
---

Recent changes:

!`git diff HEAD`
"#;
        let tools = parse_markdown_tools_from_file(raw, "review-changes", PathBuf::from("."));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "review_changes");
        assert_eq!(tools[0].script, "git diff HEAD");
    }

    #[test]
    fn parses_tagged_tool_call() {
        let call = parse_tool_call(
            r#"<tool_call>{"tool":"read-file","args":{"path":"src/lib.rs"}}</tool_call>"#,
        )
        .unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(
            call.args.get("path"),
            Some(&Value::String("src/lib.rs".to_string()))
        );
    }
}
