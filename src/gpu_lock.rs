//! Exclusive CUDA session locks for P2 GPU work.
//!
//! A user-scoped global lock prevents runs rooted in different directories from
//! sharing the GPU, while the legacy per-output lock keeps local tooling and
//! watchers compatible.

use anyhow::{bail, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const LOCK_NAME: &str = "gpu.lock";
const TRAIN_PID_NAME: &str = "train.pid";
const DEFAULT_WAIT: Duration = Duration::from_secs(3600);
const POLL: Duration = Duration::from_millis(500);

fn lock_path(output_dir: &Path) -> PathBuf {
    output_dir.join(LOCK_NAME)
}

fn global_lock_path() -> PathBuf {
    #[cfg(unix)]
    {
        let user_id = unsafe { libc::geteuid() };
        PathBuf::from("/tmp").join(format!("tofy-p2-gpu-{user_id}.lock"))
    }
    #[cfg(not(unix))]
    {
        std::env::temp_dir().join("tofy-p2-gpu.lock")
    }
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
                    || cmd.contains("p2-arc3-live-eval")
                    || cmd.contains("p2-arc3-bridge")
                    || cmd.contains("p2-context-wiring"))
        })
        .unwrap_or(false)
}

fn stale_lock_pid(path: &Path) -> Option<u32> {
    let pid = read_pid_file(path)?;
    if process_alive(pid) && (pid == std::process::id() || is_tofy_gpu_process(pid)) {
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
        let mut blocked = false;
        for path in [global_lock_path(), lock_path(output_dir)] {
            if let Some(pid) = stale_lock_pid(&path) {
                if pid != std::process::id() {
                    if Instant::now() >= deadline {
                        bail!(
                            "timed out waiting for GPU lock holder pid={pid} at {}",
                            path.display()
                        );
                    }
                    std::thread::sleep(POLL);
                    blocked = true;
                    break;
                }
            }
        }
        if blocked {
            continue;
        }
        return Ok(());
    }
}

/// RAII marker file so watchers can find the live `p2-train` PID (not the shell).
pub struct TrainPidGuard {
    path: PathBuf,
    pid: u32,
}

impl TrainPidGuard {
    pub fn install(output_dir: &Path) -> Result<Self> {
        let path = train_pid_path(output_dir);
        let pid = std::process::id();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{pid}").with_context(|| format!("write {}", path.display()))?;
                    file.sync_all()
                        .with_context(|| format!("sync {}", path.display()))?;
                    return Ok(Self { path, pid });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                    if let Some(existing) = read_pid_file(&path).filter(|pid| process_alive(*pid)) {
                        bail!(
                            "refusing to overwrite {} naming live process {existing}",
                            path.display()
                        );
                    }
                    match fs::remove_file(&path) {
                        Ok(()) => {}
                        Err(error) if error.kind() == ErrorKind::NotFound => {}
                        Err(error) => {
                            return Err(error)
                                .with_context(|| format!("remove stale {}", path.display()));
                        }
                    }
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("create {}", path.display()));
                }
            }
        }
    }
}

impl Drop for TrainPidGuard {
    fn drop(&mut self) {
        if read_pid_file(&self.path) == Some(self.pid) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// RAII exclusive GPU session across every Tofy output root for this user.
pub struct GpuSessionGuard {
    paths: Vec<PathBuf>,
    pid: u32,
}

impl GpuSessionGuard {
    pub fn acquire(output_dir: &Path) -> Result<Self> {
        fs::create_dir_all(output_dir)
            .with_context(|| format!("create {}", output_dir.display()))?;
        let pid = std::process::id();
        let mut guard = Self {
            paths: Vec::with_capacity(2),
            pid,
        };
        for path in [global_lock_path(), lock_path(output_dir)] {
            loop {
                wait_for_gpu_idle(output_dir)?;
                match OpenOptions::new().write(true).create_new(true).open(&path) {
                    Ok(mut file) => {
                        let written = writeln!(file, "{pid}")
                            .with_context(|| format!("write gpu lock {}", path.display()))
                            .and_then(|()| {
                                file.sync_all()
                                    .with_context(|| format!("sync gpu lock {}", path.display()))
                            });
                        if let Err(error) = written {
                            let _ = fs::remove_file(&path);
                            return Err(error);
                        }
                        guard.paths.push(path);
                        break;
                    }
                    Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("acquire gpu lock {}", path.display()));
                    }
                }
            }
        }
        Ok(guard)
    }
}

impl Drop for GpuSessionGuard {
    fn drop(&mut self) {
        for path in self.paths.iter().rev() {
            if read_pid_file(path) == Some(self.pid) {
                let _ = fs::remove_file(path);
            }
        }
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
            assert!(global_lock_path().is_file());
        }
        assert!(!lock_path(&dir).exists());
        assert!(!global_lock_path().exists());
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn train_pid_refuses_live_owner_and_drop_preserves_replacement() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("tofy-train-pid-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        let path = train_pid_path(&dir);
        fs::write(&path, format!("{}\n", std::process::id()))?;
        assert!(TrainPidGuard::install(&dir).is_err());

        fs::remove_file(&path)?;
        let guard = TrainPidGuard::install(&dir)?;
        fs::write(&path, "1\n")?;
        drop(guard);
        assert_eq!(fs::read_to_string(&path)?, "1\n");
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
