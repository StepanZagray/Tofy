//! Exclusive CUDA session lock per P2 output directory.
//!
//! Prevents `p2-train` and `p2-eval` from sharing a GPU simultaneously — a common
//! cause of immediate OOM on resume when a stuck eval holds ~500+ MiB.

use anyhow::{bail, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const LOCK_NAME: &str = "gpu.lock";
const TRAIN_PID_NAME: &str = "train.pid";
const DEFAULT_WAIT: Duration = Duration::from_secs(3600);
const POLL: Duration = Duration::from_millis(500);

fn lock_path(output_dir: &Path) -> PathBuf {
    output_dir.join(LOCK_NAME)
}

fn train_pid_path(output_dir: &Path) -> PathBuf {
    output_dir.join(TRAIN_PID_NAME)
}

fn read_pid_file(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    text.trim().parse().ok()
}

fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn process_cmdline(pid: u32) -> Option<String> {
    let path = format!("/proc/{pid}/cmdline");
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    Some(
        buf.split(|&b| b == 0)
            .map(|s| String::from_utf8_lossy(s))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn is_tofy_gpu_process(pid: u32) -> bool {
    process_cmdline(pid)
        .map(|cmd| {
            cmd.contains("tofy")
                && (cmd.contains("p2-train")
                    || cmd.contains("p2-eval")
                    || cmd.contains("p2-arc3-live-eval"))
        })
        .unwrap_or(false)
}

fn stale_lock_pid(path: &Path) -> Option<u32> {
    let pid = read_pid_file(path)?;
    if process_alive(pid) && is_tofy_gpu_process(pid) {
        Some(pid)
    } else {
        let _ = fs::remove_file(path);
        None
    }
}

/// Block until no other Tofy train/eval process holds the output-dir GPU lock.
pub fn wait_for_gpu_idle(output_dir: &Path) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_WAIT;
    loop {
        if let Some(pid) = read_pid_file(&train_pid_path(output_dir)) {
            if pid != std::process::id() && process_alive(pid) && is_tofy_gpu_process(pid) {
                if Instant::now() >= deadline {
                    bail!(
                        "timed out waiting for train.pid={pid} to exit before GPU work in {}",
                        output_dir.display()
                    );
                }
                std::thread::sleep(POLL);
                continue;
            }
        }
        if let Some(pid) = stale_lock_pid(&lock_path(output_dir)) {
            if pid != std::process::id() {
                if Instant::now() >= deadline {
                    bail!(
                        "timed out waiting for gpu.lock holder pid={pid} in {}",
                        output_dir.display()
                    );
                }
                std::thread::sleep(POLL);
                continue;
            }
        }
        return Ok(());
    }
}

/// RAII marker file so watchers can find the live `p2-train` PID (not the shell).
pub struct TrainPidGuard {
    path: PathBuf,
}

impl TrainPidGuard {
    pub fn install(output_dir: &Path) -> Result<Self> {
        let path = train_pid_path(output_dir);
        fs::write(&path, format!("{}\n", std::process::id()))
            .with_context(|| format!("write {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TrainPidGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// RAII exclusive GPU session for one output directory.
pub struct GpuSessionGuard {
    path: PathBuf,
}

impl GpuSessionGuard {
    pub fn acquire(output_dir: &Path) -> Result<Self> {
        fs::create_dir_all(output_dir)
            .with_context(|| format!("create {}", output_dir.display()))?;
        wait_for_gpu_idle(output_dir)?;
        let path = lock_path(output_dir);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("acquire gpu lock {}", path.display()))?;
        writeln!(file, "{}", std::process::id())
            .with_context(|| format!("write gpu lock {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for GpuSessionGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_lock_acquire_and_release() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("tofy-gpu-lock-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        {
            let _guard = GpuSessionGuard::acquire(&dir)?;
            assert!(lock_path(&dir).is_file());
        }
        assert!(!lock_path(&dir).exists());
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
