//! Stub whisper.cpp backend used when the `whisper-cpp` feature is disabled.
//!
//! The crate would otherwise fail to build on hosts that lack libclang
//! (which `whisper-rs-sys` needs to generate FFI bindings). The stub keeps
//! the trait surface intact so the rest of the host compiles; selecting
//! this backend at runtime simply returns a clear error.

use std::path::Path;

use async_trait::async_trait;

use crate::error::{AppError, Result};

use super::{BackendKind, BackendOptions, TranscriptionBackend, TranscriptionResult};

pub struct WhisperCppBackend;

impl WhisperCppBackend {
    pub fn new() -> Self {
        Self
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

    async fn load_model(&self, _path: &Path, _opts: &BackendOptions) -> Result<()> {
        Err(AppError::Backend(
            "whisper.cpp support was not compiled in (rebuild with --features whisper-cpp; \
             requires libclang on the build host)"
                .into(),
        ))
    }

    async fn is_ready(&self) -> bool {
        false
    }

    async fn transcribe(
        &self,
        _samples: &[f32],
        _sample_rate: u32,
        _opts: &BackendOptions,
    ) -> Result<TranscriptionResult> {
        Err(AppError::Backend(
            "whisper.cpp support was not compiled in".into(),
        ))
    }
}
