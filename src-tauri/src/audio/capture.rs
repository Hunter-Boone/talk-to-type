//! cpal capture controller.
//!
//! On most platforms cpal's `Stream` is `!Send`, so we keep it inside a
//! dedicated thread that owns it for life. The thread sits in a parking-lot
//! `Condvar` waiting for start/stop commands from the controller.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use parking_lot::{Condvar, Mutex};
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use serde::Serialize;
use tracing::{debug, error, info};

use crate::audio::{ring::RingBuffer, TARGET_SAMPLE_RATE};

/// One chunk of audio coming back from a recording session.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub peak: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InputDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Public handle to the audio thread. All methods are cheap.
pub struct AudioController {
    cmd: Arc<(Mutex<AudioCmd>, Condvar)>,
    is_recording: Arc<AtomicBool>,
    is_streaming: Arc<AtomicBool>,
    peak_q15: Arc<AtomicU32>,
    ring: Arc<RingBuffer>,
    /// Sample rate the active stream is delivering (set by the audio
    /// thread when a stream starts, read by `stop()` when draining).
    /// Stays at TARGET_SAMPLE_RATE when no stream is active.
    capture_rate: Arc<AtomicU32>,
    /// "The level meter wants a monitor stream up." Set by
    /// `start_monitor`, cleared by `stop_monitor`. Recording entry
    /// points (`start` / `stop`) deliberately do NOT touch it: when a
    /// hotkey-driven recording finishes we re-arm the monitor stream
    /// with the device stored below so the Audio settings page's VU
    /// meter keeps animating without the user having to flip the input
    /// device dropdown to kick it back to life.
    monitor_desired: AtomicBool,
    monitor_device: Mutex<Option<String>>,
    /// Why the most recent stream failed to come up, if it did. Set by the
    /// audio thread on build/play failure, cleared when a new start command
    /// is issued. Lets `press_to_talk` tell the user their mic is broken
    /// instead of silently recording nothing.
    last_error: Arc<Mutex<Option<String>>>,
    _thread: thread::JoinHandle<()>,
}

#[derive(Debug, Clone)]
enum AudioCmd {
    Idle,
    /// Start capturing for transcription. Samples flow into the ring buffer.
    Start { device_name: Option<String> },
    /// Start capturing for level metering only. The cpal stream runs and
    /// updates the peak atomic, but samples are NOT written to the ring —
    /// so a press_to_talk that follows can build clean audio without any
    /// stale prefix.
    StartMonitor { device_name: Option<String> },
    Stop,
    Shutdown,
}

impl AudioController {
    pub fn new() -> Self {
        // Ring is sized for 2 minutes at 48 kHz mono f32 (~22 MB). We hold
        // the raw input rate here now (resampling happens at stop time),
        // so we size for the worst-case common rate.
        let ring = Arc::new(RingBuffer::new(48_000 * 120));
        let cmd = Arc::new((Mutex::new(AudioCmd::Idle), Condvar::new()));
        let is_recording = Arc::new(AtomicBool::new(false));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let peak_q15 = Arc::new(AtomicU32::new(0));
        let capture_rate = Arc::new(AtomicU32::new(TARGET_SAMPLE_RATE));
        let last_error = Arc::new(Mutex::new(None));

        let cmd_thread = cmd.clone();
        let ring_thread = ring.clone();
        let is_recording_thread = is_recording.clone();
        let is_streaming_thread = is_streaming.clone();
        let peak_thread = peak_q15.clone();
        let rate_thread = capture_rate.clone();
        let error_thread = last_error.clone();

        let handle = thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                audio_thread_main(
                    cmd_thread,
                    ring_thread,
                    is_recording_thread,
                    is_streaming_thread,
                    peak_thread,
                    rate_thread,
                    error_thread,
                )
            })
            .expect("spawn audio thread");

        Self {
            cmd,
            is_recording,
            is_streaming,
            peak_q15,
            ring,
            capture_rate,
            monitor_desired: AtomicBool::new(false),
            monitor_device: Mutex::new(None),
            last_error,
            _thread: handle,
        }
    }

    pub fn list_input_devices() -> Vec<InputDeviceInfo> {
        let host = cpal::default_host();
        let default_name = host
            .default_input_device()
            .and_then(|d| d.name().ok())
            .unwrap_or_default();
        host.input_devices()
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|d| {
                let name = d.name().ok()?;
                let is_default = name == default_name;
                Some(InputDeviceInfo { name, is_default })
            })
            .collect()
    }

    pub fn start(&self, device_name: Option<String>) {
        self.ring.clear();
        self.peak_q15.store(0, Ordering::Relaxed);
        *self.last_error.lock() = None;
        let (lock, cvar) = &*self.cmd;
        *lock.lock() = AudioCmd::Start { device_name };
        cvar.notify_all();
    }

    /// Why the most recent stream failed to start, if it did. Cleared on
    /// every new start command.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }

    /// Start a level-metering stream. Samples are discarded; only `peak()`
    /// updates. Used by the Audio settings page to confirm the mic is alive.
    ///
    /// Records the monitor as "desired" so a recording started afterwards
    /// will revert to monitor mode on stop, instead of leaving the level
    /// meter dead until the user changes the input device.
    pub fn start_monitor(&self, device_name: Option<String>) {
        *self.monitor_device.lock() = device_name.clone();
        self.monitor_desired.store(true, Ordering::Release);

        if self.is_recording.load(Ordering::Acquire) {
            // A recording is already in flight (the user pressed the
            // hotkey before navigating to the Audio page). Don't yank
            // its stream out from under it — the recording's cpal
            // callback already updates the peak meter, so the VU bar
            // will animate. We'll re-arm the monitor on its release.
            return;
        }

        self.ring.clear();
        self.peak_q15.store(0, Ordering::Relaxed);
        *self.last_error.lock() = None;
        let (lock, cvar) = &*self.cmd;
        *lock.lock() = AudioCmd::StartMonitor { device_name };
        cvar.notify_all();
    }

    /// Stop a monitor stream. If a recording is in progress, we leave the
    /// stream alone (it's the recording's stream now) and only clear the
    /// "monitor desired" flag so the recording's stop won't try to
    /// resurrect it.
    pub fn stop_monitor(&self) {
        self.monitor_desired.store(false, Ordering::Release);
        *self.monitor_device.lock() = None;

        if self.is_recording.load(Ordering::Acquire) {
            return;
        }

        let (lock, cvar) = &*self.cmd;
        *lock.lock() = AudioCmd::Stop;
        cvar.notify_all();
    }

    /// Whether ANY cpal stream is currently up — recording or monitoring.
    pub fn is_streaming(&self) -> bool {
        self.is_streaming.load(Ordering::Relaxed)
    }

    /// Stop recording and return whatever has been captured. Samples are
    /// returned at the **input device's** native rate. Resampling to 16
    /// kHz happens later in `release_to_talk`, off the realtime thread.
    ///
    /// If a level-meter monitor was active before recording started, we
    /// re-arm it after draining so the Audio settings page's VU bar
    /// keeps responding without the user having to flip the input
    /// device dropdown to wake it back up.
    pub fn stop(&self) -> AudioFrame {
        let (lock, cvar) = &*self.cmd;
        *lock.lock() = AudioCmd::Stop;
        cvar.notify_all();

        // Spin briefly until the thread acks. ~1 ms is enough in practice.
        for _ in 0..100 {
            if !self.is_recording.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let samples = self.ring.drain();
        let peak = f32::from_bits(self.peak_q15.load(Ordering::Relaxed));
        let sample_rate = self.capture_rate.load(Ordering::Relaxed);

        if self.monitor_desired.load(Ordering::Acquire) {
            let device = self.monitor_device.lock().clone();
            self.peak_q15.store(0, Ordering::Relaxed);
            let (lock, cvar) = &*self.cmd;
            *lock.lock() = AudioCmd::StartMonitor { device_name: device };
            cvar.notify_all();
        }

        AudioFrame {
            samples,
            sample_rate,
            peak,
        }
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::Relaxed)
    }

    pub fn current_peak(&self) -> f32 {
        f32::from_bits(self.peak_q15.load(Ordering::Relaxed))
    }

    /// Snapshot the audio buffered so far without draining it. Returns the
    /// samples and the current capture rate. Used by the pre-transcription
    /// path while the user is still holding the hotkey.
    pub fn snapshot(&self) -> AudioFrame {
        AudioFrame {
            samples: self.ring.snapshot(),
            sample_rate: self.capture_rate.load(Ordering::Relaxed),
            peak: f32::from_bits(self.peak_q15.load(Ordering::Relaxed)),
        }
    }
}

impl Drop for AudioController {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.cmd;
        *lock.lock() = AudioCmd::Shutdown;
        cvar.notify_all();
    }
}

// -----------------------------------------------------------------------------
// Audio thread implementation
// -----------------------------------------------------------------------------

fn audio_thread_main(
    cmd: Arc<(Mutex<AudioCmd>, Condvar)>,
    ring: Arc<RingBuffer>,
    is_recording: Arc<AtomicBool>,
    is_streaming: Arc<AtomicBool>,
    peak_q15: Arc<AtomicU32>,
    capture_rate: Arc<AtomicU32>,
    last_error: Arc<Mutex<Option<String>>>,
) {
    let host = cpal::default_host();
    let mut active_stream: Option<cpal::Stream> = None;

    loop {
        let next = {
            let (lock, cvar) = &*cmd;
            let mut guard = lock.lock();
            // Park while idle. A new command flips us out of this wait.
            cvar.wait_while(&mut guard, |c| matches!(c, AudioCmd::Idle));
            std::mem::replace(&mut *guard, AudioCmd::Idle)
        };

        match next {
            AudioCmd::Idle => continue,
            AudioCmd::Shutdown => {
                drop(active_stream.take());
                break;
            }
            AudioCmd::Start { device_name } => {
                drop(active_stream.take());
                // The old stream is gone; only a successful build below may
                // flip this back on. Without the reset, a failed Start after
                // an active monitor stream would leave is_streaming stuck
                // true and mask the failure.
                is_streaming.store(false, Ordering::Release);
                // Set is_recording BEFORE the stream starts so the cpal
                // callback (which checks it) writes from the very first
                // sample.
                is_recording.store(true, Ordering::Release);
                match build_stream(
                    &host,
                    device_name.as_deref(),
                    ring.clone(),
                    is_recording.clone(),
                    peak_q15.clone(),
                    capture_rate.clone(),
                ) {
                    Ok(stream) => {
                        if let Err(e) = stream.play() {
                            error!("failed to start input stream: {e}");
                            // Publish the error BEFORE clearing is_recording:
                            // the press_to_talk watchdog reads error first,
                            // and this ordering guarantees it can't miss it.
                            *last_error.lock() = Some(e.to_string());
                            is_recording.store(false, Ordering::Release);
                        } else {
                            is_streaming.store(true, Ordering::Release);
                            active_stream = Some(stream);
                        }
                    }
                    Err(e) => {
                        error!("failed to build input stream: {e}");
                        *last_error.lock() = Some(e);
                        is_recording.store(false, Ordering::Release);
                    }
                }
            }
            AudioCmd::StartMonitor { device_name } => {
                drop(active_stream.take());
                is_streaming.store(false, Ordering::Release);
                // Monitor mode: stream runs but ring stays empty.
                is_recording.store(false, Ordering::Release);
                match build_stream(
                    &host,
                    device_name.as_deref(),
                    ring.clone(),
                    is_recording.clone(),
                    peak_q15.clone(),
                    capture_rate.clone(),
                ) {
                    Ok(stream) => {
                        if let Err(e) = stream.play() {
                            error!("failed to start monitor stream: {e}");
                            *last_error.lock() = Some(e.to_string());
                        } else {
                            is_streaming.store(true, Ordering::Release);
                            active_stream = Some(stream);
                        }
                    }
                    Err(e) => {
                        error!("failed to build monitor stream: {e}");
                        *last_error.lock() = Some(e);
                    }
                }
            }
            AudioCmd::Stop => {
                drop(active_stream.take());
                is_recording.store(false, Ordering::Release);
                is_streaming.store(false, Ordering::Release);
            }
        }
    }
}

fn build_stream(
    host: &cpal::Host,
    device_name: Option<&str>,
    ring: Arc<RingBuffer>,
    is_recording: Arc<AtomicBool>,
    peak_q15: Arc<AtomicU32>,
    capture_rate: Arc<AtomicU32>,
) -> Result<cpal::Stream, String> {
    let device = match device_name {
        Some(name) => {
            let named = host
                .input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().map(|n| n == name).unwrap_or(false));
            match named {
                Some(d) => d,
                // Devices come and go across reboots and USB replugs. A
                // saved name that no longer exists shouldn't kill dictation
                // entirely — fall back to whatever the OS calls default.
                None => {
                    tracing::warn!(
                        device = %name,
                        "saved input device not found; falling back to default"
                    );
                    host.default_input_device().ok_or_else(|| {
                        format!("input device '{name}' not found and no default input device")
                    })?
                }
            }
        }
        None => host
            .default_input_device()
            .ok_or_else(|| "no default input device".to_string())?,
    };

    let config = device
        .default_input_config()
        .map_err(|e| e.to_string())?;

    info!(
        device = %device.name().unwrap_or_default(),
        sample_rate = config.sample_rate().0,
        channels = config.channels(),
        format = ?config.sample_format(),
        "starting capture",
    );

    let in_rate = config.sample_rate().0;
    let in_channels = config.channels() as usize;

    // Publish the active capture rate so `stop()` knows what rate to label
    // the buffered samples with.
    capture_rate.store(in_rate, Ordering::Release);

    let err_fn = |e| error!("cpal stream error: {e}");

    let make_callback = |format: SampleFormat| -> Result<cpal::Stream, String> {
        match format {
            SampleFormat::F32 => device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[f32], _| {
                        process_samples(
                            data,
                            in_channels,
                            &ring,
                            &is_recording,
                            &peak_q15,
                        );
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string()),
            SampleFormat::I16 => device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[i16], _| {
                        let f: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
                        process_samples(&f, in_channels, &ring, &is_recording, &peak_q15);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string()),
            SampleFormat::U16 => device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[u16], _| {
                        let f: Vec<f32> = data.iter().map(|s| s.to_sample::<f32>()).collect();
                        process_samples(&f, in_channels, &ring, &is_recording, &peak_q15);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string()),
            other => Err(format!("unsupported sample format: {other:?}")),
        }
    };

    make_callback(config.sample_format())
}

/// Realtime callback: down-mix to mono, update the peak meter, and push
/// raw samples (at the input device's native rate) into the ring.
///
/// Resampling is deliberately NOT done here — it happens once at stop
/// time on a non-realtime thread. This keeps the callback dead simple
/// and immune to any subtle resampler bugs.
fn process_samples(
    data: &[f32],
    in_channels: usize,
    ring: &RingBuffer,
    is_recording: &AtomicBool,
    peak_q15: &AtomicU32,
) {
    if data.is_empty() {
        return;
    }

    // Down-mix to mono.
    //
    // Mono passes through. Stereo averages both channels (the standard
    // stereo→mono downmix). For >2 channels (array mics, conferencing
    // devices) averaging spreads any single hot mic capsule's signal
    // across silent ones; we take channel 0 instead.
    let mono: Vec<f32> = if in_channels <= 1 {
        data.to_vec()
    } else if in_channels == 2 {
        data.chunks_exact(2)
            .map(|frame| (frame[0] + frame[1]) * 0.5)
            .collect()
    } else {
        data.chunks_exact(in_channels)
            .map(|frame| frame[0])
            .collect()
    };

    // Peak meter — always update, regardless of mode. The Audio settings
    // page reads this for the level meter even outside recording. Decay
    // the previous peak slightly each callback so the meter falls back
    // to zero between syllables instead of latching at the all-time max.
    let frame_peak = mono
        .iter()
        .map(|s| s.abs())
        .fold(0.0_f32, f32::max);
    let prev = f32::from_bits(peak_q15.load(Ordering::Relaxed));
    let decayed = prev * 0.85;
    let next_peak = decayed.max(frame_peak);
    peak_q15.store(next_peak.to_bits(), Ordering::Relaxed);

    // Monitor mode: peak updated above; don't push samples into the ring.
    if !is_recording.load(Ordering::Relaxed) {
        return;
    }

    ring.push(&mono);
    debug!(samples = mono.len(), "captured");
}

/// Resample mono f32 audio from `in_rate` to `out_rate`.
///
/// We use rubato's `SincFixedIn` (windowed-sinc, Blackman2) instead of
/// the FFT family. Two earlier attempts with `FftFixedIn` produced
/// audibly bad output even with a 100 ms FFT window: the brick-wall
/// frequency-domain filter rings hard at speech transients, and the
/// resulting buffer fed Whisper a distorted-sounding spectrogram that
/// it mostly hallucinated through. Sinc resampling sidesteps that
/// entirely — it's a time-domain FIR with a smooth window, no block
/// boundaries, and is what rubato itself documents as the
/// audio-quality choice.
///
/// Implementation follows rubato's `process_f64.rs` example: feed
/// fixed-size chunks until the input is exhausted, finish with one
/// `process_partial_into_buffer` for the leftover, and skip
/// `output_delay()` frames at the start when copying the result so
/// the speech onset isn't time-shifted into the resampler's group
/// delay.
pub fn resample_mono(
    samples: &[f32],
    in_rate: u32,
    out_rate: u32,
) -> Result<Vec<f32>, String> {
    if in_rate == out_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    // Sinc parameters lifted from rubato's recommended high-quality
    // example. `sinc_len = 256` gives a stopband well below -90 dB,
    // far better than anything we need for 16 kHz Whisper input, and
    // costs only a few ms of CPU on a 3 s utterance.
    let sinc_len = 256;
    let oversampling_factor = 256;
    let window = WindowFunction::Blackman2;
    let f_cutoff = calculate_cutoff(sinc_len, window);
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor,
        window,
    };

    let ratio = out_rate as f64 / in_rate as f64;
    // ~50 ms input chunks. SincFixedIn doesn't have the
    // multiple-of-gcd constraint that FftFixedIn does, so we just
    // pick something convenient. 2048 is larger than 2 * sinc_len so
    // the resampler is happy.
    let chunk_size = ((in_rate as usize) / 20).max(2 * sinc_len);
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, chunk_size, 1)
        .map_err(|e| format!("resampler init: {e}"))?;

    let output_delay = resampler.output_delay();
    let total_input = samples.len();
    let expected_output = ((total_input as f64) * ratio).round() as usize;

    // Pre-allocate one output buffer that's reused across calls. A
    // single full chunk plus headroom is enough for any of rubato's
    // sync resamplers.
    let mut out_buffer = vec![vec![0.0f32; resampler.output_frames_max()]];
    let mut out: Vec<f32> = Vec::with_capacity(expected_output + output_delay + chunk_size);

    let mut idx = 0;
    let mut input_frames_next = resampler.input_frames_next();
    while total_input - idx >= input_frames_next {
        let chunk = &samples[idx..idx + input_frames_next];
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&[chunk], &mut out_buffer, None)
            .map_err(|e| format!("resample: {e}"))?;
        out.extend_from_slice(&out_buffer[0][..nbr_out]);
        idx += nbr_in;
        input_frames_next = resampler.input_frames_next();
    }

    // Last partial chunk (if any). `process_partial_into_buffer`
    // handles the zero-padding internally and emits whatever output
    // it can — we don't follow it up with a zero-input flush because
    // in practice that produces an audible click as the filter rings
    // on the transition between real audio and silence.
    if idx < total_input {
        let tail = &samples[idx..];
        let (_nbr_in, nbr_out) = resampler
            .process_partial_into_buffer(Some(&[tail]), &mut out_buffer, None)
            .map_err(|e| format!("resample tail: {e}"))?;
        out.extend_from_slice(&out_buffer[0][..nbr_out]);
    }

    // Skip the resampler's group delay at the head and clamp to the
    // ratio-derived expected length so we don't ship a few hundred
    // trailing zero samples (they confuse VAD and don't help Whisper).
    let drop_lead = output_delay.min(out.len());
    out.drain(..drop_lead);
    out.truncate(expected_output);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 kHz sine wave at 48 kHz should resample to 16 kHz cleanly.
    /// We verify length is ~1/3 of input and that the output has the
    /// same energy ballpark as the input (i.e. resampling didn't zero
    /// it out or wildly amplify it).
    #[test]
    fn resample_48k_to_16k_preserves_signal() {
        let in_rate = 48_000u32;
        let out_rate = 16_000u32;
        let dur_s = 1.0_f32;
        let n = (in_rate as f32 * dur_s) as usize;
        let freq = 1_000.0_f32;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / in_rate as f32).sin() * 0.5)
            .collect();

        let out = resample_mono(&input, in_rate, out_rate).expect("resample");

        let expected = (n as f64 * out_rate as f64 / in_rate as f64) as usize;
        let drift = (out.len() as i64 - expected as i64).abs();
        assert!(
            drift < 200,
            "output length {} drifted from expected {}",
            out.len(),
            expected
        );

        let in_rms = rms(&input);
        let out_rms = rms(&out);
        assert!(
            (in_rms - out_rms).abs() < 0.1,
            "RMS changed too much: in={in_rms}, out={out_rms}"
        );
    }

    #[test]
    fn resample_passthrough_when_rates_match() {
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = resample_mono(&input, 16_000, 16_000).unwrap();
        assert_eq!(input, out);
    }

    /// Regression test for the "compressed-sounding utterance" bug.
    ///
    /// A 12 kHz sine at 48 kHz is above the 8 kHz Nyquist of 16 kHz
    /// output, so a correct anti-alias filter must remove almost all
    /// of it. The previous resampler (10 ms chunks, sub_chunks = 2,
    /// fft_size = 80) had a 5 ms filter that leaked the entire tone
    /// straight through as an alias at 4 kHz, which is why speech
    /// sounded distorted and confused Whisper. We require the output
    /// RMS to drop by at least 30 dB relative to the input.
    #[test]
    fn resample_attenuates_above_nyquist() {
        let in_rate = 48_000u32;
        let out_rate = 16_000u32;
        let n = in_rate as usize; // 1 second
        let freq = 12_000.0_f32; // above 8 kHz output Nyquist
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / in_rate as f32).sin() * 0.5)
            .collect();

        let out = resample_mono(&input, in_rate, out_rate).expect("resample");
        let in_rms = rms(&input);
        let out_rms = rms(&out);

        // 30 dB ≈ 31.6× attenuation. A passing implementation should
        // do far better; this is a loose floor that still excludes the
        // unfiltered-aliasing case (which has out_rms ≈ in_rms).
        let ratio = out_rms / in_rms;
        assert!(
            ratio < 0.0316,
            "out-of-band tone leaked through resampler: in_rms={in_rms}, out_rms={out_rms}, ratio={ratio}"
        );
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        ((s / samples.len() as f64).sqrt()) as f32
    }
}
