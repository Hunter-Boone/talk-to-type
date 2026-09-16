//! Model download manager.
//!
//! Streams a single LFS file from the HF resolve endpoint to disk, emitting
//! progress events to the JS frontend (`download://progress`). Supports
//! cancellation via a per-download AtomicBool flag.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

use crate::error::{AppError, Result};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub job_id: String,
    pub repo_id: String,
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub finished: bool,
    pub error: Option<String>,
}

pub struct DownloadManager {
    cancels: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DownloadManager {
    pub fn new() -> Self {
        Self {
            cancels: Mutex::new(HashMap::new()),
        }
    }

    pub fn cancel(&self, job_id: &str) {
        if let Some(flag) = self.cancels.lock().get(job_id) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    pub async fn download(
        self: Arc<Self>,
        app: AppHandle,
        http: reqwest::Client,
        token: Option<String>,
        job_id: String,
        repo_id: String,
        file: String,
        url: String,
        dest: PathBuf,
    ) -> Result<PathBuf> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancels.lock().insert(job_id.clone(), cancel.clone());

        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        info!(%url, dest = ?dest, "starting download");
        let mut req = http.get(&url);
        if let Some(t) = token.as_deref().filter(|t| !t.trim().is_empty()) {
            req = req.bearer_auth(t);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(%url, error = %e, "download request failed");
                emit_error(&app, &job_id, &repo_id, &file, format!("request failed: {e}"));
                self.cancels.lock().remove(&job_id);
                return Err(AppError::from(e));
            }
        };
        let status = resp.status();
        if !status.is_success() {
            // Try to capture a hint from the body — HF returns a small JSON
            // error description for 401/403/404 that's helpful to surface.
            let body = resp.text().await.unwrap_or_default();
            let snippet = body.chars().take(200).collect::<String>();
            let hint = match status.as_u16() {
                401 => " (authentication required — set a HuggingFace token in Settings → Models for gated/private repos)",
                403 => " (access denied — this repo may be gated; accept its license on huggingface.co and set a token in Settings → Models)",
                404 => " (not found — the file may not exist on the main branch)",
                _ => "",
            };
            let msg = format!(
                "{} returned {}{}{}",
                url,
                status,
                hint,
                if snippet.is_empty() { String::new() } else { format!(": {snippet}") }
            );
            warn!(%url, %status, body = %snippet, "download HTTP error");
            emit_error(&app, &job_id, &repo_id, &file, msg.clone());
            self.cancels.lock().remove(&job_id);
            return Err(AppError::Http(msg));
        }
        let total = resp.content_length();
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();
        let mut file_w = tokio::fs::File::create(&dest).await?;

        while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::Relaxed) {
                drop(file_w);
                let _ = tokio::fs::remove_file(&dest).await;
                emit(
                    &app,
                    DownloadProgress {
                        job_id: job_id.clone(),
                        repo_id: repo_id.clone(),
                        file: file.clone(),
                        downloaded,
                        total,
                        finished: true,
                        error: Some("cancelled".into()),
                    },
                );
                self.cancels.lock().remove(&job_id);
                return Err(AppError::Other("cancelled".into()));
            }
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    warn!(%url, error = %e, "stream chunk failed");
                    emit_error(&app, &job_id, &repo_id, &file, format!("stream error: {e}"));
                    self.cancels.lock().remove(&job_id);
                    return Err(AppError::Http(format!("chunk: {e}")));
                }
            };
            downloaded += bytes.len() as u64;
            if let Err(e) = file_w.write_all(&bytes).await {
                warn!("write failed: {e}");
                let _ = tokio::fs::remove_file(&dest).await;
                emit_error(&app, &job_id, &repo_id, &file, format!("write failed: {e}"));
                self.cancels.lock().remove(&job_id);
                return Err(AppError::Io(e));
            }
            // Throttle progress emission to roughly 10/sec — an emit per
            // chunk is overkill and causes IPC pressure on Windows.
            if downloaded % (256 * 1024) < bytes.len() as u64 {
                emit(
                    &app,
                    DownloadProgress {
                        job_id: job_id.clone(),
                        repo_id: repo_id.clone(),
                        file: file.clone(),
                        downloaded,
                        total,
                        finished: false,
                        error: None,
                    },
                );
            }
        }
        file_w.flush().await?;
        info!(%url, bytes = downloaded, "download complete");
        emit(
            &app,
            DownloadProgress {
                job_id: job_id.clone(),
                repo_id: repo_id.clone(),
                file: file.clone(),
                downloaded,
                total,
                finished: true,
                error: None,
            },
        );
        self.cancels.lock().remove(&job_id);
        Ok(dest)
    }
}

fn emit(app: &AppHandle, p: DownloadProgress) {
    if let Err(e) = app.emit("download://progress", &p) {
        warn!("emit progress: {e}");
    }
}

fn emit_error(app: &AppHandle, job_id: &str, repo_id: &str, file: &str, msg: String) {
    emit(
        app,
        DownloadProgress {
            job_id: job_id.to_string(),
            repo_id: repo_id.to_string(),
            file: file.to_string(),
            downloaded: 0,
            total: None,
            finished: true,
            error: Some(msg),
        },
    );
}
