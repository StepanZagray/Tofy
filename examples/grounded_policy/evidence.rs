//! Source and artifact bindings for the registered standalone experiment.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
    process::Command,
};

pub fn file_hash(path: &Path) -> Result<String> {
    ensure!(
        path.is_file() && !path.is_symlink(),
        "expected regular file: {}",
        path.display()
    );
    let mut source = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = [0u8; 65536];
    loop {
        let n = source.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn git(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    ensure!(output.status.success(), "git provenance command failed");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}
pub fn provenance() -> Result<Value> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dependency = source.join("../candle_graph");
    for (root, revision, dirty, pushed) in [
        (
            source,
            env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            env!("TOFY_EMBEDDED_SOURCE_DIRTY"),
            env!("TOFY_EMBEDDED_SOURCE_PUSHED"),
        ),
        (
            dependency.as_path(),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED"),
        ),
    ] {
        ensure!(
            dirty == "false" && pushed == "true",
            "build must use clean pushed source"
        );
        ensure!(
            git(root, &["rev-parse", "HEAD"])? == revision
                && git(root, &["status", "--porcelain", "--untracked-files=all"])?.is_empty(),
            "build/runtime source mismatch"
        );
        git(
            root,
            &["merge-base", "--is-ancestor", "HEAD", "@{upstream}"],
        )?;
    }
    ensure!(
        env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION") == "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a",
        "dependency differs"
    );
    ensure!(
        env!("TOFY_EMBEDDED_CARGO_FEATURES")
            .split(',')
            .any(|x| x == "cudnn")
            && env!("TOFY_EMBEDDED_CARGO_FEATURES")
                .split(',')
                .any(|x| x == "profiling"),
        "cudnn and profiling required"
    );
    ensure!(
        env!("TOFY_EMBEDDED_BUILD_COMMAND") != "unknown",
        "missing build command"
    );
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output()?;
    ensure!(gpu.status.success(), "GPU identity query failed");
    Ok(
        json!({"gpu":String::from_utf8(gpu.stdout)?.trim(),"source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),"candle_graph_revision":env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),"binary_sha256":file_hash(&std::env::current_exe()?)?,"features":env!("TOFY_EMBEDDED_CARGO_FEATURES"),"build_command":env!("TOFY_EMBEDDED_BUILD_COMMAND")}),
    )
}
fn tree(root: &Path, path: &Path, files: &mut Map<String, Value>) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            tree(root, &path, files)?;
        } else {
            ensure!(kind.is_file(), "unexpected evidence artifact type");
            files.insert(
                path.strip_prefix(root)?.to_string_lossy().into_owned(),
                json!(file_hash(&path)?),
            );
        }
    }
    Ok(())
}
pub fn bind_profiles(root: &Path) -> Result<()> {
    let profiles = root.with_extension("profiles");
    let mut files = Map::new();
    if profiles.exists() {
        tree(&profiles, &profiles, &mut files)?;
    }
    let host = std::env::var_os("TOFY_PERF_TRACE").context("host trace unset")?;
    let host = Path::new(&host);
    write_json(
        &root.join("profiles.json"),
        &json!({"root":profiles,"files":files,"host_trace":{"path":host,"sha256":file_hash(host)?},"nsight":"external capture; bind separate immutable bundle after exit"}),
    )
}
pub fn seal(root: &Path) -> Result<()> {
    let mut files = Map::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "unexpected nested run artifact"
        );
        let name = entry.file_name().to_string_lossy().into_owned();
        ensure!(name != "manifest.json", "never reseal an existing manifest");
        files.insert(name, json!(file_hash(&entry.path())?));
    }
    let path = root.join("manifest.json");
    write_json(
        &path,
        &json!({"schema":"looped-grounded-policy-artifacts-v1","files":files}),
    )?;
    let hash = file_hash(&path)?;
    verify_root(root, &hash)?;
    fs::write(root.with_extension("manifest.sha256"), format!("{hash}\n"))?;
    Ok(())
}
pub fn verify_root(root: &Path, expected: &str) -> Result<()> {
    ensure!(
        root.is_absolute() && root.is_dir() && !root.is_symlink(),
        "invalid evidence root"
    );
    let path = root.join("manifest.json");
    ensure!(file_hash(&path)? == expected, "audit manifest hash differs");
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        value["schema"] == "looped-grounded-policy-artifacts-v1",
        "unexpected audit schema"
    );
    let files = value["files"].as_object().context("missing files")?;
    ensure!(
        fs::read_dir(root)?.count() == files.len() + 1,
        "evidence artifact population differs"
    );
    for (name, hash) in files {
        ensure!(
            Path::new(name)
                .file_name()
                .is_some_and(|n| n == name.as_str())
                && name != "manifest.json",
            "invalid artifact name"
        );
        ensure!(
            file_hash(&root.join(name))? == hash.as_str().context("invalid hash")?,
            "evidence file differs: {name}"
        );
    }
    Ok(())
}
