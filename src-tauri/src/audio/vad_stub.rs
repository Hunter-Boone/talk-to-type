//! Stub VAD used when the `vad` feature is disabled.
//!
//! Returns the input unchanged (no trimming). Selecting "VAD enabled"
//! in settings still works at the type level, but the OS-level Silero
//! ONNX runtime is not available, so we cannot trim silence.

use crate::audio::TARGET_SAMPLE_RATE;

pub struct SileroVad;

impl SileroVad {
    pub fn new() -> std::result::Result<Self, String> {
        Ok(Self)
    }

    /// Stub: report the whole clip as speech (in seconds), so callers
    /// fall through to passing the entire buffer to the backend.
    pub fn trim_bounds_sec(&mut self, samples_16k: &[f32]) -> Option<(f32, f32)> {
        if samples_16k.is_empty() {
            None
        } else {
            Some((0.0, samples_16k.len() as f32 / TARGET_SAMPLE_RATE as f32))
        }
    }
}
