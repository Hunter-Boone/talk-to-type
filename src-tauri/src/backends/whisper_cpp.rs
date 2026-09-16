//! In-process whisper.cpp backend via `whisper-rs`.
//!
//! Whisper context creation is *not* cheap (model load + memory map), so we
//! keep one alive for the lifetime of the active model and rebuild only when
//! the user picks a different file.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::Mutex;
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::TARGET_SAMPLE_RATE;
use crate::error::{AppError, Result};

use super::{BackendKind, BackendOptions, TranscriptionBackend, TranscriptionResult, WordTimestamp};

struct LoadedModel {
    path: PathBuf,
    ctx: Arc<WhisperContext>,
}

pub struct WhisperCppBackend {
    state: Mutex<Option<LoadedModel>>,
}

impl WhisperCppBackend {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }
}

impl Default for WhisperCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TranscriptionBackend for WhisperCppBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperCpp
    }

    async fn load_model(&self, model_path: &Path, _opts: &BackendOptions) -> Result<()> {
        let model_path_owned = model_path.to_path_buf();

        // Skip reload if already loaded.
        if let Some(loaded) = &*self.state.lock() {
            if loaded.path == model_path_owned {
                return Ok(());
            }
        }

        // WhisperContext::new is blocking and can take seconds — run on the
        // blocking thread pool so we don't stall the runtime.
        let path_for_thread = model_path_owned.clone();
        let ctx = tokio::task::spawn_blocking(move || {
            let params = WhisperContextParameters::default();
            WhisperContext::new_with_params(
                path_for_thread.to_str().ok_or_else(|| {
                    AppError::Backend("model path is not valid UTF-8".to_string())
                })?,
                params,
            )
            .map_err(|e| AppError::Backend(format!("whisper init: {e}")))
        })
        .await
        .map_err(|e| AppError::Backend(format!("join error: {e}")))??;

        info!(model = ?model_path_owned, "whisper.cpp model loaded");
        *self.state.lock() = Some(LoadedModel {
            path: model_path_owned,
            ctx: Arc::new(ctx),
        });
        Ok(())
    }

    async fn is_ready(&self) -> bool {
        self.state.lock().is_some()
    }

    async fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
        opts: &BackendOptions,
    ) -> Result<TranscriptionResult> {
        let ctx = {
            let guard = self.state.lock();
            guard
                .as_ref()
                .ok_or(AppError::BackendNotReady)?
                .ctx
                .clone()
        };

        // whisper.cpp wants 16 kHz mono. If cpal handed us anything
        // else we resample inline; the sidecar-backed engines side-
        // step this by handing the WAV to librosa/ffmpeg directly.
        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            samples.to_vec()
        } else {
            crate::audio::resample_mono(samples, sample_rate, TARGET_SAMPLE_RATE)
                .map_err(AppError::Backend)?
        };
        let lang = opts.language.clone();
        let initial_prompt = opts.initial_prompt.clone();
        let want_words = opts.word_timestamps;
        let threads = opts
            .threads
            .unwrap_or_else(|| (num_cpus::get() as u32 / 2).max(1));

        // Inference is CPU/GPU-bound and blocking — push it to a worker.
        tokio::task::spawn_blocking(move || -> Result<TranscriptionResult> {
            let started = Instant::now();
            let mut state = ctx
                .create_state()
                .map_err(|e| AppError::Backend(format!("whisper state: {e}")))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(threads as i32);
            params.set_translate(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_special(false);
            params.set_print_timestamps(false);
            params.set_suppress_blank(true);
            // Token timestamps power our word-level splicing for the
            // streaming pipeline. Cheap-ish to enable (small extra cost
            // in the decoder); only flipped when callers ask for it.
            if want_words {
                params.set_token_timestamps(true);
            }
            if let Some(lang) = lang.as_deref() {
                params.set_language(Some(lang));
            }
            if let Some(prompt) = initial_prompt.as_deref() {
                params.set_initial_prompt(prompt);
            }

            debug!(
                samples = samples.len(),
                duration_s = samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
                threads,
                want_words,
                "running whisper.cpp inference",
            );

            state
                .full(params, &samples)
                .map_err(|e| AppError::Backend(format!("whisper full: {e}")))?;

            let n_segments = state
                .full_n_segments()
                .map_err(|e| AppError::Backend(format!("whisper segments: {e}")))?;
            let mut text = String::new();
            for i in 0..n_segments {
                let seg = state
                    .full_get_segment_text(i)
                    .map_err(|e| AppError::Backend(format!("whisper segment text: {e}")))?;
                text.push_str(&seg);
            }
            let text = text.trim().to_string();

            let words = if want_words {
                extract_word_timestamps(&state, n_segments)?
            } else {
                Vec::new()
            };

            let duration_ms = started.elapsed().as_millis() as u64;
            Ok(TranscriptionResult {
                text,
                language: lang,
                duration_ms,
                words,
            })
        })
        .await
        .map_err(|e| AppError::Backend(format!("join error: {e}")))?
    }

    async fn unload(&self) -> Result<()> {
        *self.state.lock() = None;
        Ok(())
    }
}

/// Walk every (segment, token) pair and group BPE tokens into words by
/// "leading space starts a new word" — Whisper's tokenizer convention.
///
/// Each emitted `WordTimestamp` carries the token group's combined text
/// (including the leading space, so the host can splice without having
/// to re-introduce whitespace), the start time of the first token in
/// the group, and the end time of the last. Times are converted from
/// whisper.cpp's centisecond units (10 ms) to seconds.
///
/// Special tokens (`[_BEG_]`, `<|...|>`, etc.) and tokens with invalid
/// timing (`t0 < 0`) are skipped.
fn extract_word_timestamps(
    state: &whisper_rs::WhisperState,
    n_segments: i32,
) -> Result<Vec<WordTimestamp>> {
    let mut out: Vec<WordTimestamp> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_start: Option<f32> = None;
    let mut cur_end: f32 = 0.0;

    let flush = |out: &mut Vec<WordTimestamp>,
                 cur_text: &mut String,
                 cur_start: &mut Option<f32>,
                 cur_end: f32| {
        if let Some(s) = *cur_start {
            if !cur_text.trim().is_empty() {
                out.push(WordTimestamp {
                    text: std::mem::take(cur_text),
                    start: s,
                    end: cur_end.max(s),
                });
            } else {
                cur_text.clear();
            }
        } else {
            cur_text.clear();
        }
        *cur_start = None;
    };

    for seg in 0..n_segments {
        let n_tokens = state
            .full_n_tokens(seg)
            .map_err(|e| AppError::Backend(format!("whisper n_tokens: {e}")))?;
        for tok in 0..n_tokens {
            let text = match state.full_get_token_text(seg, tok) {
                Ok(t) => t,
                // Some tokens fail strict UTF-8 decoding (Whisper
                // emits raw bytes for partial multi-byte sequences
                // mid-word). Fall back to the lossy path so the
                // word doesn't get dropped on the floor.
                Err(_) => state
                    .full_get_token_text_lossy(seg, tok)
                    .map_err(|e| AppError::Backend(format!("whisper token text: {e}")))?,
            };
            // Skip whisper's internal special tokens. They wrap the
            // real transcription with markers like `[_BEG_]` and
            // language hints like `<|en|>` that have no acoustic
            // realisation.
            if text.starts_with('[') || text.starts_with("<|") {
                continue;
            }

            let data = state
                .full_get_token_data(seg, tok)
                .map_err(|e| AppError::Backend(format!("whisper token data: {e}")))?;
            if data.t0 < 0 {
                continue;
            }
            let t0 = data.t0 as f32 / 100.0;
            let t1 = (data.t1.max(data.t0) as f32) / 100.0;

            if text.starts_with(' ') {
                flush(&mut out, &mut cur_text, &mut cur_start, cur_end);
                cur_text.push_str(&text);
                cur_start = Some(t0);
                cur_end = t1;
            } else {
                if cur_start.is_none() {
                    cur_start = Some(t0);
                }
                cur_text.push_str(&text);
                cur_end = t1;
            }
        }
    }
    flush(&mut out, &mut cur_text, &mut cur_start, cur_end);

    Ok(out)
}

// Tiny shim so we don't pull in num_cpus for a one-liner. Standard library's
// `available_parallelism` is good enough.
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
    }
}
