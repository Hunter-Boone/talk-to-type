//! Speech-to-text backend abstraction.
//!
//! Three concrete implementations:
//!   1. WhisperCppBackend     — in-process via `whisper-rs`
//!   2. FasterWhisperBackend  — out-of-process Python sidecar (CTranslate2)
//!   3. TransformersBackend   — out-of-process Python sidecar (HF transformers)
//!
//! Backends 2 and 3 share the same sidecar process; only the `engine` field
//! in the `load_model` JSON-RPC request differs.

use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

mod faster_whisper;
mod transformers;
#[cfg(feature = "whisper-cpp")]
mod whisper_cpp;
#[cfg(not(feature = "whisper-cpp"))]
#[path = "whisper_cpp_stub.rs"]
mod whisper_cpp;

pub use faster_whisper::FasterWhisperBackend;
pub use transformers::TransformersBackend;
pub use whisper_cpp::WhisperCppBackend;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    WhisperCpp,
    FasterWhisper,
    Transformers,
    /// Pseudo-backend that piggybacks on the model registry + HF
    /// download flow but isn't a `TranscriptionBackend`. Holds
    /// causal-LMs used by the post-processing cleanup pass.
    CleanupTransformers,
}

impl BackendKind {
    pub fn label(self) -> &'static str {
        match self {
            BackendKind::WhisperCpp => "whisper.cpp",
            BackendKind::FasterWhisper => "faster-whisper",
            BackendKind::Transformers => "Transformers",
            BackendKind::CleanupTransformers => "Transformers (cleanup)",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            BackendKind::WhisperCpp => "In-process GGUF/GGML via whisper-rs. Best for CPU and Apple Silicon.",
            BackendKind::FasterWhisper => "CTranslate2 quantized Whisper via Python sidecar. Fastest on x86 CPU.",
            BackendKind::Transformers => "Hugging Face Transformers via Python sidecar. Any ASR model.",
            BackendKind::CleanupTransformers => "HuggingFace causal-LM run in the sidecar to clean up dictated text.",
        }
    }

    /// True for kinds that implement `TranscriptionBackend`. The cleanup
    /// pseudo-backend lives in the same registry / HF-search infrastructure
    /// as the ASR backends but is never instantiated as one.
    pub fn is_asr(self) -> bool {
        matches!(
            self,
            BackendKind::WhisperCpp
                | BackendKind::FasterWhisper
                | BackendKind::Transformers
        )
    }
}

/// User-tunable knobs that vary per backend. Stored in settings, applied at
/// model-load time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BackendOptions {
    /// whisper.cpp: number of threads. Default = num_cpus / 2.
    pub threads: Option<u32>,

    /// faster-whisper compute_type: "int8", "int8_float16", "float16", "float32".
    pub compute_type: Option<String>,

    /// transformers: torch device ("cpu", "cuda", "mps").
    pub device: Option<String>,

    /// Optional language hint (ISO 639-1). `None` = auto-detect.
    pub language: Option<String>,

    /// Optional initial prompt to bias the decoder.
    pub initial_prompt: Option<String>,

    /// Request word-level timestamps in the result.
    ///
    /// Used by the streaming pipeline so that overlapping windows can
    /// be spliced at exact word boundaries (instead of the host having
    /// to guess where one window's transcription "should" pick up
    /// from another's). Adds a small per-call cost — Whisper has to
    /// emit per-token timing — so it's opt-in rather than always on.
    ///
    /// Backend behaviour when `true`:
    ///   - whisper.cpp: enables `set_token_timestamps(true)` and emits
    ///     a `WordTimestamp` per token group.
    ///   - faster-whisper: passes `word_timestamps=True` to
    ///     `model.transcribe(...)`; per-word timing is exact.
    ///   - transformers: passes `return_timestamps="word"` to the
    ///     decode path; per-word timing is exact for Whisper, less
    ///     reliable for non-Whisper ASR repos.
    ///
    /// When `false`, the backend MAY return an empty `words` vector
    /// even if the model emits something internally — the host treats
    /// the field as best-effort.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub word_timestamps: bool,
}

/// A single word with its acoustic time range, in seconds, relative to
/// the start of the audio buffer the backend was given.
///
/// `text` is the word as the model emitted it, including any leading
/// space. `start` <= `end` always; backends should clamp to a positive
/// non-zero range even for very short tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub duration_ms: u64,
    /// Per-word timing if the backend was asked for it (via
    /// `BackendOptions::word_timestamps`) AND the underlying model
    /// returned timing info. Empty otherwise — callers must handle
    /// the empty case (e.g. by falling back to whole-text append).
    #[serde(default)]
    pub words: Vec<WordTimestamp>,
}

/// Object-safe transcription backend. All three implementations live behind
/// a `Box<dyn TranscriptionBackend>` in `AppState::backend`.
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    fn kind(&self) -> BackendKind;

    /// Load a model from disk. Should be cheap to call multiple times — if
    /// the same path is already loaded, it should be a no-op.
    async fn load_model(&self, model_path: &Path, opts: &BackendOptions) -> Result<()>;

    /// Whether `load_model` has been called successfully.
    async fn is_ready(&self) -> bool;

    /// Run inference on mono f32 samples at the given `sample_rate`.
    ///
    /// Backends are responsible for any resampling they need: whisper.cpp
    /// expects 16 kHz so it resamples in-process, while the sidecar-based
    /// backends pass the file straight to librosa/ffmpeg which both have
    /// production-grade resamplers built in. Doing it this way keeps the
    /// pristine native-rate audio out of any homebrew resampler pipeline.
    ///
    /// Implementations must be safe to call concurrently from multiple
    /// tasks (a Mutex inside is fine).
    async fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
        opts: &BackendOptions,
    ) -> Result<TranscriptionResult>;

    /// Free model memory. Optional — default does nothing.
    async fn unload(&self) -> Result<()> {
        Ok(())
    }
}
