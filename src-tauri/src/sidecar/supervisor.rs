//! Sidecar process supervisor.
//!
//! Owns a single `tokio::process::Child` running the Python sidecar. Spawns
//! lazily on first request. If the sidecar dies (stdout closes, or a stdin
//! write fails), the running handle is marked dead, every pending request is
//! failed immediately, and the next `call` respawns a fresh process and
//! retries once. Model state does NOT survive a respawn — the backends
//! detect the "no ASR model loaded" error and re-issue `load_model`.
//!
//! IPC framing: newline-delimited JSON over stdin (host → sidecar) and
//! stdout (sidecar → host). Sidecar logs go to stderr, which we forward to
//! `tracing::warn` so they show up in dev.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::error::{AppError, Result};
use crate::sidecar::protocol::{RpcError, RpcRequest, RpcResponse};

const SIDECAR_BINARY_NAME: &str = if cfg!(windows) {
    "talk_to_type_sidecar.exe"
} else {
    "talk_to_type_sidecar"
};

const PYTHON_EXE: &str = if cfg!(windows) { "python.exe" } else { "python" };

/// Error code used for transport-level failures we synthesise on the host
/// side (sidecar exited, pipe broken). Distinct from any code Python sends.
const TRANSPORT_ERROR_CODE: i32 = -32000;

/// How to invoke the sidecar. Either a single self-contained executable
/// (PyInstaller-built bundle) or `<python> <main.py>` (dev mode).
enum SidecarCommand {
    Binary(PathBuf),
    Python { interpreter: PathBuf, script: PathBuf },
}

impl SidecarCommand {
    fn display(&self) -> String {
        match self {
            SidecarCommand::Binary(p) => p.display().to_string(),
            SidecarCommand::Python { interpreter, script } => {
                format!("{} {}", interpreter.display(), script.display())
            }
        }
    }

    fn build_command(&self) -> Command {
        match self {
            SidecarCommand::Binary(p) => Command::new(p),
            SidecarCommand::Python { interpreter, script } => {
                let mut cmd = Command::new(interpreter);
                cmd.arg(script);
                // Python buffers stdout when not attached to a TTY. Force
                // unbuffered so framed JSON-RPC actually flushes per write.
                cmd.env("PYTHONUNBUFFERED", "1");
                cmd
            }
        }
    }
}

type ResponseSender = oneshot::Sender<std::result::Result<serde_json::Value, RpcError>>;

pub struct SidecarSupervisor {
    inner: RwLock<Option<Arc<RunningSidecar>>>,
    next_id: AtomicU64,
}

struct RunningSidecar {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<u64, ResponseSender>>>,
    /// Set by the stdout-reader task when the pipe closes (sidecar exited
    /// or crashed). `ensure_started` treats a dead handle like `None`.
    dead: Arc<AtomicBool>,
    /// Kept so `shutdown_blocking` can kill the process explicitly on a
    /// hard exit (where Drop / kill_on_drop never runs).
    child: std::sync::Mutex<Child>,
}

impl SidecarSupervisor {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
            next_id: AtomicU64::new(1),
        }
    }
}

impl Default for SidecarSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl SidecarSupervisor {
    /// Send an RPC request and await the response. Spawns the sidecar if it
    /// isn't already running, and retries ONCE on a transport failure (dead
    /// pipe / sidecar exit) after respawning a fresh process.
    pub async fn call<P, R>(&self, method: &'static str, params: P) -> Result<R>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let mut last_err: Option<AppError> = None;
        for attempt in 0..2 {
            let running = self.ensure_started().await?;
            match self.call_on(&running, method, &params).await {
                Ok(value) => return Ok(serde_json::from_value(value)?),
                Err(CallError::Transport(msg)) => {
                    warn!(attempt, error = %msg, "sidecar transport failure; discarding instance");
                    self.discard(&running).await;
                    last_err = Some(AppError::Sidecar(msg));
                    // loop → respawn and retry once
                }
                Err(CallError::Rpc(e)) => {
                    return Err(AppError::Sidecar(format!("{}: {}", e.code, e.message)))
                }
                Err(CallError::Timeout) => {
                    // Don't kill the process — inference may legitimately be
                    // slow; the pending entry was already cleaned up.
                    return Err(AppError::Sidecar("rpc timed out".to_string()));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| AppError::Sidecar("sidecar unavailable".into())))
    }

    async fn call_on<P: Serialize>(
        &self,
        running: &Arc<RunningSidecar>,
        method: &'static str,
        params: &P,
    ) -> std::result::Result<serde_json::Value, CallError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();

        // Register the pending future before writing — avoids a race where
        // the response lands before we've inserted the sender.
        running.pending.lock().await.insert(id, tx);

        let req = RpcRequest { id, method, params };
        let bytes = match serde_json::to_vec(&req) {
            Ok(b) => b,
            Err(e) => {
                running.pending.lock().await.remove(&id);
                return Err(CallError::Transport(format!("serialize request: {e}")));
            }
        };

        {
            let mut stdin = running.stdin.lock().await;
            let write = async {
                stdin.write_all(&bytes).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await
            };
            if let Err(e) = write.await {
                running.pending.lock().await.remove(&id);
                return Err(CallError::Transport(format!("stdin write: {e}")));
            }
        }

        // 5 minutes is a reasonable upper bound for cold-loading a 3 GB
        // Whisper-large model on CPU.
        match tokio::time::timeout(Duration::from_secs(300), rx).await {
            Err(_) => {
                running.pending.lock().await.remove(&id);
                Err(CallError::Timeout)
            }
            // Sender dropped without a response — reader task died between
            // our insert and its drain, or the map was torn down.
            Ok(Err(_)) => Err(CallError::Transport("sidecar exited".into())),
            Ok(Ok(Err(e))) if e.code == TRANSPORT_ERROR_CODE => {
                Err(CallError::Transport(e.message))
            }
            Ok(Ok(Err(e))) => Err(CallError::Rpc(e)),
            Ok(Ok(Ok(value))) => Ok(value),
        }
    }

    /// Drop `running` from `inner` (if it's still the current instance) so
    /// the next call respawns. Kills the old process for good measure.
    async fn discard(&self, running: &Arc<RunningSidecar>) {
        let mut guard = self.inner.write().await;
        if let Some(current) = guard.as_ref() {
            if Arc::ptr_eq(current, running) {
                *guard = None;
            }
        }
        running.fail_all_pending("sidecar exited").await;
        if let Ok(mut child) = running.child.lock() {
            let _ = child.start_kill();
        }
    }

    /// Synchronously kill the sidecar process, if running. Safe to call
    /// from a `RunEvent::Exit` handler — Tauri's runtime is still alive
    /// at that point, but `process::exit` will skip Drop, so we have to
    /// take and kill the `Child` explicitly here. `kill_on_drop(true)`
    /// covers the normal-Drop path; this covers the hard-exit path.
    pub fn shutdown_blocking(&self) {
        let mut guard = tauri::async_runtime::block_on(self.inner.write());
        if let Some(running) = guard.take() {
            if let Ok(mut child) = running.child.lock() {
                let _ = child.start_kill();
            }
        }
    }

    async fn ensure_started(&self) -> Result<Arc<RunningSidecar>> {
        {
            let guard = self.inner.read().await;
            if let Some(r) = guard.as_ref() {
                if !r.dead.load(Ordering::Acquire) {
                    return Ok(r.clone());
                }
            }
        }
        let mut guard = self.inner.write().await;
        if let Some(r) = guard.as_ref() {
            if !r.dead.load(Ordering::Acquire) {
                return Ok(r.clone());
            }
            // Dead instance still parked here — fail its stragglers before
            // replacing it.
            r.fail_all_pending("sidecar exited").await;
        }
        let running = Arc::new(self.spawn_sidecar()?);
        *guard = Some(running.clone());
        Ok(running)
    }

    fn spawn_sidecar(&self) -> Result<RunningSidecar> {
        let cmd = self.locate_sidecar()?;
        let label = cmd.display();
        info!(sidecar = %label, "spawning sidecar");

        let mut command = cmd.build_command();
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Suppress the flash of black console window that Windows
        // would otherwise allocate for a console-subsystem child
        // (e.g. a PyInstaller-built `talk_to_type_sidecar.exe`)
        // when its parent (us) is itself running in the GUI
        // subsystem. Without this every dictation kicks up a
        // console window for the lifetime of the sidecar.
        // Mirrors the flag we already pass for the cursor-agent
        // subprocess in commands.rs.
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command
            .spawn()
            .map_err(|e| AppError::Sidecar(format!("spawn {}: {}", label, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AppError::Sidecar("no stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::Sidecar("no stdout".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AppError::Sidecar("no stderr".to_string()))?;

        let pending: Arc<Mutex<HashMap<u64, ResponseSender>>> = Arc::new(Mutex::new(HashMap::new()));
        let dead = Arc::new(AtomicBool::new(false));

        // stdout reader: parse one JSON object per line and dispatch to the
        // matching pending sender. When the pipe closes the sidecar is gone:
        // mark the instance dead and fail every in-flight request so callers
        // get an immediate error instead of a 5-minute timeout.
        let pending_reader = pending.clone();
        let dead_reader = dead.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                debug!(line = %line, "sidecar stdout");
                let parsed: std::result::Result<RpcResponse<serde_json::Value>, _> =
                    serde_json::from_str(&line);
                match parsed {
                    Ok(resp) => {
                        let sender = pending_reader.lock().await.remove(&resp.id);
                        if let Some(tx) = sender {
                            let payload = if let Some(err) = resp.error {
                                Err(err)
                            } else {
                                Ok(resp.result.unwrap_or(serde_json::Value::Null))
                            };
                            let _ = tx.send(payload);
                        } else {
                            warn!(id = resp.id, "no pending sender for response");
                        }
                    }
                    Err(e) => warn!(error = %e, line = %line, "malformed sidecar response"),
                }
            }
            warn!("sidecar stdout closed; marking instance dead");
            dead_reader.store(true, Ordering::Release);
            let mut map = pending_reader.lock().await;
            for (_, tx) in map.drain() {
                let _ = tx.send(Err(RpcError {
                    code: TRANSPORT_ERROR_CODE,
                    message: "sidecar exited".into(),
                }));
            }
        });

        // stderr forwarder.
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if !line.trim().is_empty() {
                    warn!(target: "sidecar", "{line}");
                }
            }
        });

        Ok(RunningSidecar {
            stdin: Mutex::new(stdin),
            pending,
            dead,
            child: std::sync::Mutex::new(child),
        })
    }

    /// Locate the sidecar. Search order:
    ///   1. $TALK_TO_TYPE_SIDECAR (override — path to a binary OR a .py script)
    ///   2. <project>/sidecar/dist/talk_to_type_sidecar/<exe> (PyInstaller --onedir output, baked in at compile time)
    ///   3. <project>/sidecar/.venv/.../python + main.py (project venv)
    ///   4. system `python` from PATH + <project>/sidecar/main.py (last-resort)
    ///
    /// We deliberately do NOT bundle the sidecar into the installer — the
    /// PyInstaller payload is multi-GB and pushes NSIS past its 2 GB
    /// addressing limit. The host points at the on-disk dist folder via
    /// `env!("CARGO_MANIFEST_DIR")`, which is resolved at compile time.
    fn locate_sidecar(&self) -> Result<SidecarCommand> {
        if let Ok(env) = std::env::var("TALK_TO_TYPE_SIDECAR") {
            let p = PathBuf::from(&env);
            // Allow pointing at a script directly — the user might want to run
            // `main.py` from a custom venv they've already activated.
            if p.extension().and_then(|s| s.to_str()) == Some("py") {
                if let Some(py) = which_python() {
                    return Ok(SidecarCommand::Python { interpreter: py, script: p });
                }
            }
            return Ok(SidecarCommand::Binary(p));
        }

        // env!() resolves at compile time — the path baked in here is the
        // build host's project tree, which is exactly what we want for a
        // local-only dictation app.
        let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");

        let dev_bin = project
            .join("sidecar")
            .join("dist")
            .join("talk_to_type_sidecar")
            .join(SIDECAR_BINARY_NAME);
        if dev_bin.exists() {
            return Ok(SidecarCommand::Binary(dev_bin));
        }

        let main_py = project.join("sidecar").join("main.py");
        if main_py.exists() {
            let venv_python = if cfg!(windows) {
                project.join("sidecar").join(".venv").join("Scripts").join("python.exe")
            } else {
                project.join("sidecar").join(".venv").join("bin").join("python")
            };
            if venv_python.exists() {
                return Ok(SidecarCommand::Python {
                    interpreter: venv_python,
                    script: main_py.clone(),
                });
            }

            if let Some(py) = which_python() {
                return Ok(SidecarCommand::Python { interpreter: py, script: main_py });
            }
        }

        Err(AppError::Sidecar(format!(
            "sidecar not found. Tried: TALK_TO_TYPE_SIDECAR env var, \
             {} (baked-in dev binary), sidecar/.venv python, and system 'python' on PATH. \
             Either run sidecar/build.ps1, install Python and run `pip install -e sidecar`, \
             or set TALK_TO_TYPE_SIDECAR to a binary or main.py path.",
            dev_bin.display()
        )))
    }
}

impl RunningSidecar {
    async fn fail_all_pending(&self, msg: &str) {
        let mut map = self.pending.lock().await;
        for (_, tx) in map.drain() {
            let _ = tx.send(Err(RpcError {
                code: TRANSPORT_ERROR_CODE,
                message: msg.to_string(),
            }));
        }
    }
}

enum CallError {
    /// The process or its pipes are gone — safe to respawn and retry.
    Transport(String),
    /// The sidecar answered with an application-level error — do NOT retry.
    Rpc(RpcError),
    /// No answer within the deadline — surface without killing the process.
    Timeout,
}

/// Find a Python interpreter on PATH. Prefers `python3` on Unix-like systems
/// where `python` may be Python 2 or missing entirely.
fn which_python() -> Option<PathBuf> {
    let candidates: &[&str] = if cfg!(windows) {
        &["python.exe", "py.exe", "python3.exe"]
    } else {
        &["python3", "python"]
    };
    for name in candidates {
        if let Some(path) = which_on_path(name) {
            return Some(path);
        }
    }
    let _ = PYTHON_EXE; // keep the constant alive even if unused on a platform
    None
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
