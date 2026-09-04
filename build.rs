use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const UNKNOWN: &str = "unknown";

fn command_output(directory: &Path, arguments: &[&str]) -> Option<(ExitStatus, Vec<u8>)> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .output()
        .ok()?;
    Some((output.status, output.stdout))
}

fn git_revision(directory: &Path) -> Option<String> {
    let (status, stdout) = command_output(directory, &["rev-parse", "--verify", "HEAD"])?;
    if !status.success() {
        return None;
    }
    let revision = String::from_utf8(stdout).ok()?.trim().to_owned();
    valid_revision(&revision).then_some(revision)
}

fn valid_revision(revision: &str) -> bool {
    revision.len() >= 7
        && revision
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

fn git_dirty(directory: &Path) -> Option<bool> {
    let (status, stdout) = command_output(
        directory,
        &["status", "--porcelain", "--untracked-files=normal"],
    )?;
    status
        .success()
        .then(|| !stdout.iter().all(u8::is_ascii_whitespace))
}

fn git_pushed(directory: &Path) -> Option<bool> {
    let (upstream_status, upstream) =
        command_output(directory, &["rev-parse", "--verify", "@{upstream}"])?;
    if !upstream_status.success() || upstream.iter().all(u8::is_ascii_whitespace) {
        return None;
    }
    let (status, _) = command_output(
        directory,
        &["merge-base", "--is-ancestor", "HEAD", "@{upstream}"],
    )?;
    match status.code() {
        Some(0) => Some(true),
        Some(1) => Some(false),
        _ => None,
    }
}

fn explicit_or_git(
    directory: &Path,
    revision_env: &str,
    dirty_env: &str,
    pushed_env: &str,
) -> (String, String, String, String) {
    if let Some(revision) = env::var(revision_env)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        assert!(
            valid_revision(&revision),
            "{revision_env} must be a hexadecimal git object id"
        );
        return (
            revision,
            format!("build-env:{revision_env}"),
            env::var(dirty_env).unwrap_or_else(|_| UNKNOWN.to_owned()),
            env::var(pushed_env).unwrap_or_else(|_| UNKNOWN.to_owned()),
        );
    }
    match git_revision(directory) {
        Some(revision) => (
            revision,
            "embedded-build:git".to_owned(),
            git_dirty(directory)
                .map(|dirty| dirty.to_string())
                .unwrap_or_else(|| UNKNOWN.to_owned()),
            git_pushed(directory)
                .map(|pushed| pushed.to_string())
                .unwrap_or_else(|| UNKNOWN.to_owned()),
        ),
        None => (
            UNKNOWN.to_owned(),
            UNKNOWN.to_owned(),
            UNKNOWN.to_owned(),
            UNKNOWN.to_owned(),
        ),
    }
}

fn emit(name: &str, value: &str) {
    println!("cargo:rustc-env={name}={value}");
}

fn main() {
    for variable in [
        "TOFY_BUILD_SOURCE_REVISION",
        "TOFY_BUILD_SOURCE_DIRTY",
        "TOFY_BUILD_SOURCE_PUSHED",
        "TOFY_BUILD_CANDLE_GRAPH_REVISION",
        "TOFY_BUILD_CANDLE_GRAPH_DIRTY",
        "TOFY_BUILD_CANDLE_GRAPH_PUSHED",
        "TOFY_BUILD_COMMAND",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for path in [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "src",
        "benches",
        "examples",
        "tests",
        "vendor",
        "../candle_graph/src",
        "../candle_graph/Cargo.toml",
        "../candle_graph/Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let candle_graph_dir = manifest_dir.join("..").join("candle_graph");
    let source = explicit_or_git(
        &manifest_dir,
        "TOFY_BUILD_SOURCE_REVISION",
        "TOFY_BUILD_SOURCE_DIRTY",
        "TOFY_BUILD_SOURCE_PUSHED",
    );
    let candle_graph = explicit_or_git(
        &candle_graph_dir,
        "TOFY_BUILD_CANDLE_GRAPH_REVISION",
        "TOFY_BUILD_CANDLE_GRAPH_DIRTY",
        "TOFY_BUILD_CANDLE_GRAPH_PUSHED",
    );

    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort();

    emit("TOFY_EMBEDDED_SOURCE_REVISION", &source.0);
    emit("TOFY_EMBEDDED_SOURCE_REVISION_ORIGIN", &source.1);
    emit("TOFY_EMBEDDED_SOURCE_DIRTY", &source.2);
    emit("TOFY_EMBEDDED_SOURCE_PUSHED", &source.3);
    emit("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION", &candle_graph.0);
    emit("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY", &candle_graph.2);
    emit("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED", &candle_graph.3);
    emit("TOFY_EMBEDDED_CARGO_FEATURES", &features.join(","));
    emit(
        "TOFY_EMBEDDED_CARGO_PROFILE",
        &env::var("PROFILE").unwrap_or_else(|_| UNKNOWN.to_owned()),
    );
    emit(
        "TOFY_EMBEDDED_CARGO_TARGET",
        &env::var("TARGET").unwrap_or_else(|_| UNKNOWN.to_owned()),
    );
    emit(
        "TOFY_EMBEDDED_BUILD_COMMAND",
        &env::var("TOFY_BUILD_COMMAND").unwrap_or_else(|_| UNKNOWN.to_owned()),
    );
}
