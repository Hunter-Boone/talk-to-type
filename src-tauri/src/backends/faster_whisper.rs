//! faster-whisper backend (CTranslate2 via Python sidecar).

use std::path::{Path, PathBuf};
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

use super::{BackendKind, BackendOptions, TranscriptionBackend, TranscriptionResult, WordTimestamp};

#[derive(Clone, PartialEq)]
struct LoadedState {
    /// Model directory pushed to the sidecar.
    path: PathBuf,
    /// Load-time options (compute_type / device). Part of the identity so
    /// changing them in settings actually triggers a reload.
    options: serde_json::Value,
}

pub struct FasterWhisperBackend {
    sidecar: Arc<SidecarSupervisor>,
    loaded: Mutex<Option<LoadedState>>,
}

impl FasterWhisperBackend {
    pub fn new(sidecar: Arc<SidecarSupervisor>) -> Self {
        Self {
            sidecar,
            loaded: Mutex::new(None),
        }
    }

    async fn push_load(&self, state: &LoadedState) -> Result<()> {
        let params = LoadModelParams {
            engine: Engine::FasterWhisper,
            model: state
                .path
                .to_str()
                .ok_or_else(|| AppError::Backend("non-UTF-8 model path".into()))?
                .to_string(),
            options: state.options.clone(),
        };
        let _: LoadModelResult = self.sidecar.call("load_model", params).await?;
        Ok(())
    }
}

#[async_trait]
impl TranscriptionBackend for FasterWhisperBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::FasterWhisper
    }

    async fn load_model(&self, model_path: &Path, opts: &BackendOptions) -> Result<()> {
        // Registry entries from older builds point at `<dir>/model.bin`;
        // CTranslate2 loads the directory, so normalise to it.
        let mut model_path = model_path.to_path_buf();
        if model_path.file_name().and_then(|n| n.to_str()) == Some("model.bin") {
            if let Some(parent) = model_path.parent() {
                model_path = parent.to_path_buf();
            }
        }

        let state = LoadedState {
            path: model_path,
            options: json!({
                "compute_type": opts.compute_type.clone().unwrap_or_else(|| "int8".to_string()),
                "device": opts.device.clone().unwrap_or_else(|| "auto".to_string()),
            }),
        };
        if self.loaded.lock().as_ref() == Some(&state) {
            return Ok(());
        }

        self.push_load(&state).await?;
        info!(path = ?state.path, "faster-whisper model loaded");
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
            "initial_prompt": opts.initial_prompt,
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

/// Write a mono PCM_S16LE WAV to a temp file at the given sample
/// rate and return the path. The sidecar reads this back with
/// librosa/soundfile, which handle resampling to 16 kHz internally
/// with high quality — we ship the device's native rate to avoid
/// putting our own resampler in the audio path. Files are tiny
/// (even 60 s of speech is < 2 MB at 48 kHz) so a temp file is easier
/// than streaming the buffer over stdio.
pub(crate) fn write_temp_wav(samples: &[f32], sample_rate: u32) -> Result<PathBuf> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("ttt-{}.wav", uuid::Uuid::new_v4()));
    crate::audio::write_wav(&path, samples, sample_rate).map_err(AppError::Backend)?;
    Ok(path)
}
