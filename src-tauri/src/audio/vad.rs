//! Silero VAD wrapper.
//!
//! `voice_activity_detector` ships the Silero ONNX model embedded in the
//! crate, so no extra download is needed. We only use it to *trim* leading
//! and trailing silence — the backend itself decides what counts as speech.

use voice_activity_detector::VoiceActivityDetector;

use crate::audio::TARGET_SAMPLE_RATE;

const FRAME_SIZE: usize = 512; // 32 ms at 16 kHz — the size Silero expects.
const SPEECH_THRESHOLD: f32 = 0.5;

pub struct SileroVad {
    detector: VoiceActivityDetector,
}

impl SileroVad {
    pub fn new() -> Result<Self, String> {
        let detector = VoiceActivityDetector::builder()
            .sample_rate(TARGET_SAMPLE_RATE as i64)
            .chunk_size(FRAME_SIZE)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self { detector })
    }

    /// Find the leading + trailing silence boundaries in 16 kHz mono
    /// audio. Returns `(start_sec, end_sec)` of the speech region in
    /// the input, or `None` if the entire clip scored below threshold
    /// (i.e., nothing was said).
    ///
    /// We pad with ~120 ms either side so we don't clip plosives.
    /// Returning seconds rather than sample indices lets callers apply
    /// the trim to a buffer at any sample rate, which we use to keep
    /// the model-bound audio at its native rate while still letting
    /// Silero do silence detection on a 16 kHz scratch copy.
    pub fn trim_bounds_sec(&mut self, samples_16k: &[f32]) -> Option<(f32, f32)> {
        if samples_16k.len() < FRAME_SIZE {
            return None;
        }

        let mut speech_frames = Vec::with_capacity(samples_16k.len() / FRAME_SIZE);
        for chunk in samples_16k.chunks(FRAME_SIZE) {
            if chunk.len() < FRAME_SIZE {
                break;
            }
            let mut buf = [0.0f32; FRAME_SIZE];
            buf.copy_from_slice(chunk);
            let prob = self.detector.predict(buf);
            speech_frames.push(prob >= SPEECH_THRESHOLD);
        }

        let first = speech_frames.iter().position(|&v| v)?;
        let last = speech_frames.iter().rposition(|&v| v)?;

        // 120 ms of padding on each side, clamped to the buffer edges.
        let pad_frames = (120 / 32).max(1);
        let start_frame = first.saturating_sub(pad_frames);
        let end_frame = (last + pad_frames + 1).min(speech_frames.len());
        let start_sec = (start_frame * FRAME_SIZE) as f32 / TARGET_SAMPLE_RATE as f32;
        let end_sec = (end_frame * FRAME_SIZE) as f32 / TARGET_SAMPLE_RATE as f32;
        if end_sec <= start_sec {
            return None;
        }
        Some((start_sec, end_sec))
    }
}
