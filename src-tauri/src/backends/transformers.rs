//! HuggingFace Transformers backend via the same Python sidecar.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;
use tracing::{info, warn};

use crate::error::{AppError, Result};
use crate::sidecar::{
    protocol::{Engine, LoadModelParams, LoadModelResult, TranscribeParams, TranscribeResult},
    SidecarSupervisor,
};

use super::{
    faster_whisper::write_temp_wav, BackendKind, BackendOptions, TranscriptionBackend,
    TranscriptionResult, WordTimestamp,
};

#[derive(Clone, PartialEq)]
struct LoadedState {
    /// HF repo id or local path handed to `from_pretrained()`.
    model: String,
    /// Load-time options (device). Part of the identity so changing the
    /// device in settings actually triggers a reload.
    options: serde_json::Value,
}

pub struct TransformersBackend {
    sidecar: Arc<SidecarSupervisor>,
    loaded: Mutex<Option<LoadedState>>,
}

impl TransformersBackend {
    pub fn new(sidecar: Arc<SidecarSupervisor>) -> Self {
        Self {
            sidecar,
            loaded: Mutex::new(None),
        }
    }

    async fn push_load(&self, state: &LoadedState) -> Result<()> {
        let params = LoadModelParams {
            engine: Engine::Transformers,
            model: state.model.clone(),
            options: state.options.clone(),
        };
        let _: LoadModelResult = self.sidecar.call("load_model", params).await?;
        Ok(())
    }
}

#[async_trait]
impl TranscriptionBackend for TransformersBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Transformers
    }

    async fn load_model(&self, model_path: &Path, opts: &BackendOptions) -> Result<()> {
        // For transformers the "path" might really be a repo id (e.g.
        // "openai/whisper-base"). The local-models registry will store such
        // entries as a sentinel file containing the repo id.
        let model_str = if let Some(repo) = read_repo_id_marker(model_path)? {
            repo
        } else {
            model_path
                .to_str()
                .ok_or_else(|| AppError::Backend("non-UTF-8 model path".into()))?
                .to_string()
        };

        let state = LoadedState {
            model: model_str,
            options: json!({
                "device": opts.device.clone().unwrap_or_else(|| "cpu".to_string()),
            }),
        };
        if self.loaded.lock().as_ref() == Some(&state) {
            return Ok(());
        }

        self.push_load(&state).await?;
        info!(model = %state.model, "transformers model loaded");
        *self.loaded.lock() = Some(state);
        Ok(())
    }

    async fn is_ready(&self) -> bool {
        self.loaded.lock().is_some()
    }

    async fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
        opts: &BackendOptions,
    ) -> Result<TranscriptionResult> {
        if !self.is_ready().await {
            return Err(AppError::BackendNotReady);
        }
        let path = write_temp_wav(samples, sample_rate)?;
        let options = json!({
            "language": opts.language,
            "word_timestamps": opts.word_timestamps,
        });
        let params = TranscribeParams {
            samples_path: path.to_string_lossy().into_owned(),
            options: options.clone(),
        };

        let first: Result<TranscribeResult> = self.sidecar.call("transcribe", params).await;
        // Self-heal: a respawned sidecar (post-crash) has no model in
        // memory. Replay the load and retry once instead of failing the
        // dictation.
        let result = match first {
            Err(AppError::Sidecar(msg)) if msg.contains("no ASR model loaded") => {
                warn!("sidecar lost its model (respawned?); reloading and retrying once");
                let state = self.loaded.lock().clone();
                match state {
                    Some(state) => {
                        let reload = self.push_load(&state).await;
                        match reload {
                            Ok(()) => {
                                self.sidecar
                                    .call("transcribe", TranscribeParams {
                                        samples_path: path.to_string_lossy().into_owned(),
                                        options,
                                    })
                                    .await
                            }
                            Err(e) => Err(e),
                        }
                    }
                    None => Err(AppError::BackendNotReady),
                }
            }
            other => other,
        };
        // Always unlink the temp WAV, success or not.
        let _ = std::fs::remove_file(&path);
        let result = result?;

        let words = result
            .words
            .into_iter()
            .map(|w| WordTimestamp {
                text: w.text,
                start: w.start,
                end: w.end,
            })
            .collect();

        Ok(TranscriptionResult {
            text: result.text,
            language: result.language,
            duration_ms: result.duration_ms,
            words,
        })
    }

    async fn unload(&self) -> Result<()> {
        if self.loaded.lock().is_none() {
            return Ok(());
        }
        let _: serde_json::Value = self.sidecar.call("unload", json!({})).await?;
        *self.loaded.lock() = None;
        Ok(())
    }
}

/// If the "model path" is a local file ending in `.repo` containing a HF
/// repo id, return that id. This is how the registry tracks Transformers
/// models that aren't downloaded but pinned by id.
fn read_repo_id_marker(path: &Path) -> Result<Option<String>> {
    if path.extension().and_then(|s| s.to_str()) != Some("repo") {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
}
