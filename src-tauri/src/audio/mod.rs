//! Audio capture pipeline.
//!
//! cpal runs the input callback on a high-priority OS thread. We MUST NOT
//! block in there — anything heavier than a copy into a bounded ring buffer
//! is a glitch waiting to happen. The pipeline therefore looks like:
//!
//!   cpal callback (sync, realtime)
//!     → resample to 16 kHz mono f32
//!     → push into a bounded crossbeam channel
//!   consumer (Tokio task)
//!     → optional Silero VAD trim
//!     → hand the f32 buffer to the active TranscriptionBackend
//!
//! The controller in this module owns the cpal Stream (which is `!Send` on
//! some platforms) inside an async-aware mutex. Recording start/stop
//! transitions just toggle a flag inside the callback so we never tear the
//! stream down on every keystroke.

mod capture;
mod ring;
#[cfg(feature = "vad")]
pub mod vad;
#[cfg(not(feature = "vad"))]
#[path = "vad_stub.rs"]
pub mod vad;

pub use capture::{resample_mono, AudioController, AudioFrame, InputDeviceInfo};
pub use ring::RingBuffer;

use std::path::Path;

/// Write 16-bit PCM mono WAV at the given sample rate. Used both for the
/// temp file we hand the sidecar and for the debug-last-utterance copy on
/// disk so users can play back what the model actually heard.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| format!("wav create: {e}"))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(v)
            .map_err(|e| format!("wav write: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("wav finalize: {e}"))?;
    Ok(())
}

/// Sample rate Whisper and most ASR models operate on. Backends that
/// need 16 kHz (whisper.cpp, Silero VAD) resample to this rate themselves;
/// the sidecar-backed engines hand the WAV to librosa/ffmpeg and let
/// those production-grade resamplers do the conversion.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;
