//! Integration test for the TranscriptionBackend trait via a mock
//! implementation. Verifies that the trait is object-safe, that the
//! lifecycle (load → transcribe → unload) holds, and that `is_ready`
//! tracks state correctly.
//!
//! We don't test the real whisper.cpp / sidecar implementations here —
//! those need real models and out-of-process resources. Their integration
//! is exercised by manual smoke tests in the README.

use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;

use talk_to_type_lib::backends::{
    BackendKind, BackendOptions, TranscriptionBackend, TranscriptionResult,
};
use talk_to_type_lib::error::{AppError, Result};

#[derive(Default)]
struct MockBackend {
    state: Mutex<MockState>,
}

#[derive(Default)]
struct MockState {
    loaded: bool,
    last_audio_len: usize,
}

#[async_trait]
impl TranscriptionBackend for MockBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperCpp
    }
    async fn load_model(&self, _path: &Path, _opts: &BackendOptions) -> Result<()> {
        self.state.lock().unwrap().loaded = true;
        Ok(())
    }
    async fn is_ready(&self) -> bool {
        self.state.lock().unwrap().loaded
    }
    async fn transcribe(
        &self,
        samples: &[f32],
        _sample_rate: u32,
        _opts: &BackendOptions,
    ) -> Result<TranscriptionResult> {
        let mut s = self.state.lock().unwrap();
        if !s.loaded {
            return Err(AppError::BackendNotReady);
        }
        s.last_audio_len = samples.len();
        Ok(TranscriptionResult {
            text: format!("mocked {} samples", samples.len()),
            language: Some("en".into()),
            duration_ms: 7,
            words: Vec::new(),
        })
    }
    async fn unload(&self) -> Result<()> {
        self.state.lock().unwrap().loaded = false;
        Ok(())
    }
}

#[tokio::test]
async fn lifecycle_load_transcribe_unload() {
    let b: Box<dyn TranscriptionBackend> = Box::new(MockBackend::default());
    assert!(!b.is_ready().await);

    let opts = BackendOptions::default();
    b.load_model(Path::new("dummy"), &opts).await.unwrap();
    assert!(b.is_ready().await);

    let r = b
        .transcribe(&[0.0_f32; 16_000], 16_000, &opts)
        .await
        .unwrap();
    assert_eq!(r.text, "mocked 16000 samples");
    assert_eq!(r.language.as_deref(), Some("en"));

    b.unload().await.unwrap();
    assert!(!b.is_ready().await);
}

#[tokio::test]
async fn transcribe_before_load_errors() {
    let b: Box<dyn TranscriptionBackend> = Box::new(MockBackend::default());
    let err = b
        .transcribe(&[0.0_f32; 100], 16_000, &BackendOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, AppError::BackendNotReady));
}

#[tokio::test]
async fn backend_kind_is_stable() {
    let b: Box<dyn TranscriptionBackend> = Box::new(MockBackend::default());
    assert_eq!(b.kind(), BackendKind::WhisperCpp);
}
