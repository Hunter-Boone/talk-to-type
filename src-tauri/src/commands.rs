//! `#[tauri::command]` handlers exposed to the React frontend, plus the
//! shared press/release-to-talk logic the global hotkey driver also uses.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tracing::{info, warn};
use uuid::Uuid;

use crate::audio::{AudioController, InputDeviceInfo, TARGET_SAMPLE_RATE};
use crate::backends::{
    BackendKind, FasterWhisperBackend, TranscriptionBackend, TransformersBackend,
    WhisperCppBackend,
};
use crate::db::{HistoryEntry, NewHistoryEntry};
use crate::error::{AppError, Result};
use crate::hotkey::{self, ArcsBundle};
use crate::models::{
    pipeline_tag_for_backend, select_for_transformers, DownloadManager, HubClient, HubModel,
    LocalModel,
};
use crate::sidecar::protocol::{
    CleanupParams, CleanupResult, Engine, LoadModelParams, LoadModelResult,
};
use crate::overlay::{self, OverlayPayload, OverlayState};
use crate::paste;
use crate::settings::{PostAction, PostProcessing, PostProcessingProvider, Settings};
use crate::state::{AppState, PreTranscribeState};

// =============================================================================
// Lifecycle
// =============================================================================

#[derive(Serialize)]
pub struct AppInfo {
    pub version: &'static str,
    pub platform: &'static str,
}

#[tauri::command]
pub fn get_app_info() -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        platform: std::env::consts::OS,
    }
}

#[tauri::command]
pub fn quit(app: AppHandle) {
    app.exit(0);
}

// =============================================================================
// Backend management
// =============================================================================

#[derive(Serialize)]
pub struct BackendDescriptor {
    pub kind: BackendKind,
    pub label: &'static str,
    pub description: &'static str,
}

/// List the ASR backends the user can pick as `active_backend`. The
/// `CleanupTransformers` pseudo-backend lives in the same registry but
/// isn't a transcription backend, so it's intentionally excluded here.
#[tauri::command]
pub fn list_backends() -> Vec<BackendDescriptor> {
    vec![
        BackendKind::WhisperCpp,
        BackendKind::FasterWhisper,
        BackendKind::Transformers,
    ]
    .into_iter()
    .map(|k| BackendDescriptor {
        kind: k,
        label: k.label(),
        description: k.description(),
    })
    .collect()
}

#[tauri::command]
pub async fn switch_backend(state: State<'_, AppState>, kind: BackendKind) -> Result<()> {
    info!(?kind, "switching backend");
    let bundle = state.clone_arcs();
    instantiate_backend(&bundle, kind).await?;
    {
        let mut s = state.settings.write().await;
        s.active_backend = kind;
        s.save()?;
    }
    if let Err(e) = ensure_active_model_loaded(&bundle).await {
        warn!(error = %e, "failed to auto-load model after backend switch");
    }
    Ok(())
}

#[tauri::command]
pub fn active_backend(state: State<'_, AppState>) -> Option<BackendKind> {
    *state.backend_kind.lock()
}

/// Mark a downloaded model as active and load it into the live backend.
/// The frontend calls this from the Backend route's model dropdown.
#[tauri::command]
pub async fn select_model(state: State<'_, AppState>, model_id: String) -> Result<()> {
    info!(%model_id, "selecting model");
    {
        let mut s = state.settings.write().await;
        s.active_model_id = Some(model_id);
        s.save()?;
    }
    ensure_active_model_loaded(&state.clone_arcs()).await
}

/// Idempotent: instantiates the backend matching `settings.active_backend` if
/// it isn't already the live one, and loads `settings.active_model_id` into
/// it. Called from app startup, backend-switch, and model-pick paths.
pub(crate) async fn ensure_backend_ready(state: &AppState) -> Result<()> {
    ensure_backend_ready_bundle(&state.clone_arcs()).await
}

/// Bundle-flavoured `ensure_backend_ready` so the hotkey-driven recording
/// paths (which have no `tauri::State`) can self-heal a not-yet-loaded
/// backend instead of failing the dictation.
pub(crate) async fn ensure_backend_ready_bundle(bundle: &ArcsBundle) -> Result<()> {
    let kind = bundle.settings.read().await.active_backend;
    if bundle.backend_kind.lock().map(|k| k != kind).unwrap_or(true) {
        instantiate_backend(bundle, kind).await?;
    }
    ensure_active_model_loaded(bundle).await
}

async fn instantiate_backend(bundle: &ArcsBundle, kind: BackendKind) -> Result<()> {
    let new_backend: Box<dyn TranscriptionBackend> = match kind {
        BackendKind::WhisperCpp => Box::new(WhisperCppBackend::new()),
        BackendKind::FasterWhisper => Box::new(FasterWhisperBackend::new(bundle.sidecar.clone())),
        BackendKind::Transformers => Box::new(TransformersBackend::new(bundle.sidecar.clone())),
        BackendKind::CleanupTransformers => {
            // CleanupTransformers shares the registry/HF-search infrastructure
            // but isn't a transcription backend. The frontend filters it out
            // of the backend picker; reject explicitly here as a defence in
            // depth in case a stale `active_backend` ends up persisted.
            return Err(AppError::Backend(
                "cleanup-transformers is not an ASR backend".into(),
            ));
        }
    };
    *bundle.backend.write().await = Some(new_backend);
    *bundle.backend_kind.lock() = Some(kind);
    Ok(())
}

async fn ensure_active_model_loaded(bundle: &ArcsBundle) -> Result<()> {
    let (model_id, opts) = {
        let s = bundle.settings.read().await;
        (s.active_model_id.clone(), s.backend_options.clone())
    };
    let Some(id) = model_id else { return Ok(()) };
    let Some(local) = bundle.models.get(&id) else {
        warn!(model_id = %id, "active model id not found in registry; clearing");
        let mut s = bundle.settings.write().await;
        s.active_model_id = None;
        s.save()?;
        return Ok(());
    };
    let backend = bundle.backend.read().await;
    let backend = backend.as_ref().ok_or(AppError::BackendNotReady)?;
    backend.load_model(&local.path, &opts).await
}

// =============================================================================
// Audio
// =============================================================================

#[tauri::command]
pub fn list_input_devices() -> Vec<InputDeviceInfo> {
    AudioController::list_input_devices()
}

/// Start a level-metering stream on the given device. Used by the Audio
/// settings page so the user can confirm the mic is alive.
#[tauri::command]
pub async fn start_audio_monitor(
    state: State<'_, AppState>,
    device: Option<String>,
) -> Result<()> {
    let device = match device {
        Some(d) if !d.is_empty() => Some(d),
        _ => state.settings.read().await.input_device.clone(),
    };
    state.audio.start_monitor(device);
    Ok(())
}

#[tauri::command]
pub fn stop_audio_monitor(state: State<'_, AppState>) {
    state.audio.stop_monitor();
}

/// Latest peak amplitude in [0.0, 1.0]. The audio thread updates this
/// continuously while a stream is up — recording or monitoring.
#[tauri::command]
pub fn current_audio_level(state: State<'_, AppState>) -> f32 {
    state.audio.current_peak()
}

// =============================================================================
// Hotkey
// =============================================================================

/// Re-register both global hotkeys from current settings. Called from app
/// startup (so the hotkeys work even if the webview never loads) and by
/// both hotkey-update commands so dictate + repaste stay in sync.
pub(crate) async fn sync_hotkeys(app: &AppHandle, state: &AppState) -> Result<()> {
    let (dictate_str, repaste_str) = {
        let s = state.settings.read().await;
        (s.hotkey.clone(), s.repaste_hotkey.clone())
    };
    let dictate = hotkey::parse_shortcut(&dictate_str)?;
    let repaste = match repaste_str.as_deref() {
        Some(s) if !s.trim().is_empty() => Some(hotkey::parse_shortcut(s)?),
        _ => None,
    };
    hotkey::register(app, dictate, repaste)
}

#[tauri::command]
pub async fn register_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    combo: Option<String>,
) -> Result<()> {
    if let Some(s) = combo {
        // Validate before persisting so we never save a garbage combo.
        let _ = hotkey::parse_shortcut(&s)?;
        let mut settings = state.settings.write().await;
        settings.hotkey = s;
        settings.save()?;
    }
    sync_hotkeys(&app, &state).await
}

/// Enable/disable the OS launch-at-login registration to match the
/// setting. Errors are logged, not fatal — a failed registry write
/// shouldn't take the settings save down with it.
pub(crate) fn sync_autostart(app: &AppHandle, want: bool) {
    use tauri_plugin_autostart::ManagerExt;
    let autolaunch = app.autolaunch();
    let currently = autolaunch.is_enabled().unwrap_or(false);
    if want == currently {
        return;
    }
    let result = if want {
        autolaunch.enable()
    } else {
        autolaunch.disable()
    };
    match result {
        Ok(()) => info!(enabled = want, "launch-at-login registration updated"),
        Err(e) => warn!(error = %e, enabled = want, "failed to update launch-at-login"),
    }
}

#[tauri::command]
pub async fn register_repaste_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    combo: Option<String>,
) -> Result<()> {
    let new_value = match combo {
        Some(s) if !s.trim().is_empty() => {
            // Validate before persisting.
            let _ = hotkey::parse_shortcut(s.trim())?;
            Some(s.trim().to_string())
        }
        _ => None,
    };
    {
        let mut settings = state.settings.write().await;
        settings.repaste_hotkey = new_value;
        settings.save()?;
    }
    sync_hotkeys(&app, &state).await
}

#[tauri::command]
pub fn unregister_hotkey(app: AppHandle) -> Result<()> {
    hotkey::unregister_all(&app)
}

#[tauri::command]
pub async fn current_hotkey(state: State<'_, AppState>) -> Result<String> {
    Ok(state.settings.read().await.hotkey.clone())
}

/// Paste the most recent transcription into the focused app. Used by the
/// repaste hotkey and the manual `repaste_last` command.
pub(crate) async fn repaste_last_impl(bundle: &ArcsBundle, app: &AppHandle) -> Result<()> {
    let entries = bundle.history.list(1)?;
    let Some(latest) = entries.into_iter().next() else {
        info!("repaste: no transcription in history");
        return Ok(());
    };
    paste::copy_and_paste(app, &latest.text)?;
    Ok(())
}

#[tauri::command]
pub async fn repaste_last(state: State<'_, AppState>, app: AppHandle) -> Result<()> {
    let bundle = state.clone_arcs();
    repaste_last_impl(&bundle, &app).await
}

// =============================================================================
// Recording (manual + hotkey-driven entry points share `press_to_talk` /
// `release_to_talk` so behaviour is identical regardless of source).
// =============================================================================

#[tauri::command]
pub async fn start_recording(state: State<'_, AppState>, app: AppHandle) -> Result<()> {
    let bundle = state.clone_arcs();
    press_to_talk(&bundle, &app).await
}

#[tauri::command]
pub async fn stop_recording(state: State<'_, AppState>, app: AppHandle) -> Result<()> {
    let bundle = state.clone_arcs();
    release_to_talk(&bundle, &app).await
}

pub(crate) async fn press_to_talk(bundle: &ArcsBundle, app: &AppHandle) -> Result<()> {
    if bundle.audio.is_recording() {
        return Ok(());
    }
    let (device, pretranscribe_enabled) = {
        let s = bundle.settings.read().await;
        (s.input_device.clone(), s.pretranscribe)
    };
    bundle.audio.start(device);
    overlay::show(app);
    overlay::emit(
        app,
        OverlayPayload {
            state: OverlayState::Listening,
            message: None,
            peak: Some(0.0),
        },
    );

    // Warm the backend while the user is speaking. Right after app start
    // (or a sidecar crash) the model may not be loaded yet — kicking the
    // load off here means release_to_talk usually finds it ready instead
    // of paying the whole load inside the user's wait. Errors are ignored;
    // release_to_talk retries and surfaces them properly.
    {
        let warm = bundle.clone();
        tokio::spawn(async move {
            if let Err(e) = ensure_backend_ready_bundle(&warm).await {
                warn!(error = %e, "background backend warm-up failed");
            }
        });
    }

    // Watchdog: if the input stream never comes up (device unplugged,
    // exclusive-mode lock, missing driver after a reboot), tell the user.
    // Without this the overlay says "Listening" while zero samples arrive
    // and the release path silently discards the empty utterance.
    {
        let app_w = app.clone();
        let audio_w = bundle.audio.clone();
        tokio::spawn(async move {
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if audio_w.is_streaming() {
                    return; // stream is up — all good
                }
                if let Some(err) = audio_w.last_error() {
                    warn!(error = %err, "input stream failed to start");
                    overlay::emit(
                        &app_w,
                        OverlayPayload {
                            state: OverlayState::Error,
                            message: Some(format!("microphone error: {err}")),
                            peak: None,
                        },
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
                    if !audio_w.is_recording() {
                        overlay::hide(&app_w);
                    }
                    return;
                }
                if !audio_w.is_recording() {
                    return; // released (or stopped) before the stream settled
                }
            }
        });
    }

    // Spawn a low-frequency ticker that pushes peak values to the overlay
    // so the waveform/VU-meter animates smoothly. Stops automatically when
    // recording ends.
    let app_for_ticker = app.clone();
    let audio_for_ticker = bundle.audio.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(50));
        while audio_for_ticker.is_recording() {
            interval.tick().await;
            overlay::emit(
                &app_for_ticker,
                OverlayPayload {
                    state: OverlayState::Listening,
                    message: None,
                    peak: Some(audio_for_ticker.current_peak()),
                },
            );
        }
    });

    // Reset committed state and (optionally) start the streaming
    // pretranscribe task. We always clear so a leftover state from a
    // prior failed session can't pollute this one.
    *bundle.pretranscribe_state.lock() = PreTranscribeState::default();
    if let Some(prev) = bundle.pretranscribe_task.lock().take() {
        prev.abort();
    }
    if pretranscribe_enabled {
        let handle = spawn_pretranscribe_task(bundle);
        *bundle.pretranscribe_task.lock() = Some(handle);
    }

    Ok(())
}

/// Streaming pre-transcription, "sliding-window with overlap" edition.
///
/// While the user holds the hotkey, this loop fires each time enough
/// new audio has accumulated since `committed`. Each iteration
/// transcribes a WINDOW that includes:
///
///   1. **Lookback**: up to `LOOKBACK_SECS` of audio from BEFORE
///      `committed` (so the model has acoustic context for the words
///      that border the committed boundary). Without this, every new
///      chunk starts cold and Whisper makes the same kind of phoneme
///      errors that would happen on any short clip — `agent wrappers`
///      gets heard as `Asian rappers` when the model has no left
///      context.
///   2. **New audio**: from `committed` to a freshly-chosen cut point.
///      We prefer a real silence cut, but force a cut at
///      `MAX_NEW_SECS` if the user is fluent and never pauses.
///
/// The backend is asked for **word-level timestamps** so we can splice
/// precisely: words whose start time falls before the committed
/// boundary are dropped (their audio was already committed by an
/// earlier window's transcription), and the rest is appended.
///
/// Because the lookback overlaps with previously-committed audio, each
/// region of audio gets transcribed at least twice: once at the END of
/// some window (where it has only left context) and once at the START
/// of the next window (where it has both left and right context). The
/// second pass is the higher-quality one, and timestamp-based splicing
/// uses it.
///
/// Backends that don't support word timestamps for the active model
/// (currently: the Transformers sidecar with non-Whisper checkpoints)
/// fall back to a no-lookback chunked mode after the first window —
/// graceful degradation, same accuracy as the previous implementation.
fn spawn_pretranscribe_task(bundle: &ArcsBundle) -> tokio::task::JoinHandle<()> {
    let audio = bundle.audio.clone();
    let backend = bundle.backend.clone();
    let settings = bundle.settings.clone();
    let state = bundle.pretranscribe_state.clone();

    tokio::spawn(async move {
        // Give the user a moment to actually be talking before we burn
        // CPU on the first window.
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;

        let mut committed: usize = 0;
        let mut committed_text = String::new();
        // Whether the active backend has been observed to return word
        // timestamps. `None` = unknown (try with lookback);
        // `Some(true)` = lookback is safe; `Some(false)` = no lookback,
        // chunked-style append.
        let mut have_words: Option<bool> = None;

        loop {
            if !audio.is_recording() {
                break;
            }
            let frame = audio.snapshot();
            let snap_len = frame.samples.len();
            let rate = frame.sample_rate;

            // Need ≥ MIN_NEW_SECS of unprocessed audio before firing.
            // 6 seconds is past the threshold where Whisper's per-clip
            // word-error-rate stabilises near its long-form value.
            // Smaller windows mean more chunked-style hallucinations.
            let new_audio = snap_len.saturating_sub(committed);
            let min_new = rate as usize * MIN_NEW_SECS;
            if new_audio < min_new {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                continue;
            }

            // Pick a cut point: prefer silence inside [committed +
            // HEAD_PAD .. snap_len - TAIL_PAD]. If we're past
            // MAX_NEW_SECS of new audio, force a cut even without a
            // clean silence — the next window's lookback will re-
            // transcribe whatever's around the forced cut, so a
            // mid-word boundary here gets repaired downstream.
            let head_pad = (rate as usize / 1000) * 200;
            let tail_pad = (rate as usize / 1000) * 300;
            let search_start = committed + head_pad;
            let search_end = snap_len.saturating_sub(tail_pad);
            if search_end <= search_start {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                continue;
            }
            let region = &frame.samples[search_start..search_end];
            let max_new = rate as usize * MAX_NEW_SECS;
            let cut = match find_silence_cut(region, rate) {
                Some(c) => search_start + c,
                None if new_audio >= max_new => search_end,
                None => {
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    continue;
                }
            };
            if cut <= committed {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                continue;
            }

            // Lookback decision. Disabled for backends that don't
            // emit word timestamps (would otherwise duplicate audio
            // we've already committed text for).
            let lookback_enabled = !matches!(have_words, Some(false));
            let lookback_samples = if lookback_enabled {
                (rate as usize * LOOKBACK_SECS).min(committed)
            } else {
                0
            };
            let window_start = committed - lookback_samples;
            let raw_window = &frame.samples[window_start..cut];

            // Trim leading silence — kills "you" / "Thanks for
            // watching" hallucinations Whisper produces on
            // silence-padded inputs.
            let speech_range = trim_silence_padding(raw_window, rate);
            let (trim_lo, trim_hi) = match speech_range {
                Some(r) => r,
                None => {
                    info!(
                        window_samples = raw_window.len(),
                        "pretranscribe window all silence; skipping"
                    );
                    committed = cut;
                    *state.lock() = PreTranscribeState {
                        committed_samples: committed,
                        text: committed_text.clone(),
                        sample_rate: rate,
                    };
                    continue;
                }
            };
            let window_audio: Vec<f32> = raw_window[trim_lo..trim_hi].to_vec();
            let effective_window_start = window_start + trim_lo;
            // Time (within the trimmed window) corresponding to the
            // committed boundary. Any word with `start < committed_secs`
            // covers audio we already have text for and gets dropped.
            let committed_secs = if effective_window_start >= committed {
                0.0
            } else {
                (committed - effective_window_start) as f32 / rate as f32
            };

            // Backend options + biasing prompt + ask for words.
            let mut chunk_opts = settings.read().await.backend_options.clone();
            chunk_opts.word_timestamps = true;
            if chunk_opts.initial_prompt.is_none() {
                let prompt = tail_words(&committed_text, 32);
                if !prompt.is_empty() {
                    chunk_opts.initial_prompt = Some(prompt);
                }
            }

            let started = std::time::Instant::now();
            let transcribe_result = {
                let backend_guard = backend.read().await;
                let Some(b) = backend_guard.as_ref() else {
                    drop(backend_guard);
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    continue;
                };
                b.transcribe(&window_audio, rate, &chunk_opts).await
            };

            match transcribe_result {
                Ok(r) => {
                    let new_text: String = if !r.words.is_empty() {
                        have_words = Some(true);
                        splice_new_words(&r.words, committed_secs)
                    } else if lookback_enabled && committed > 0 {
                        // Backend gave no timestamps AND we used
                        // lookback. The full text covers audio we've
                        // already committed; falling back to whole-
                        // text append would duplicate. Mark the
                        // backend as no-words and drop this result —
                        // the next iteration retries without
                        // lookback.
                        have_words = Some(false);
                        warn!(
                            "backend returned no word timestamps with lookback active; \
                             dropping window and disabling lookback"
                        );
                        // Don't advance committed — re-transcribe
                        // this region next iteration without lookback.
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        continue;
                    } else {
                        have_words = Some(false);
                        r.text.trim().to_string()
                    };

                    if !new_text.is_empty() {
                        append_chunk_text(&mut committed_text, &new_text);
                    }
                    committed = cut;
                    info!(
                        committed_samples = committed,
                        window_samples = window_audio.len(),
                        used_words = !r.words.is_empty(),
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        text_len = new_text.len(),
                        "pretranscribe window committed",
                    );
                    *state.lock() = PreTranscribeState {
                        committed_samples: committed,
                        text: committed_text.clone(),
                        sample_rate: rate,
                    };
                }
                Err(e) => {
                    warn!(error = %e, "pretranscribe window failed; will retry");
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
        }
    })
}

/// Minimum new audio (seconds) we need before firing a window. Past
/// this point Whisper has enough context for stable per-clip accuracy.
const MIN_NEW_SECS: usize = 6;
/// Hard ceiling on new audio per window. Forces a cut even without a
/// silence break so fluent speakers still get streaming progress.
/// Mid-word forced cuts are repaired by the next window's lookback +
/// timestamp splice.
const MAX_NEW_SECS: usize = 12;
/// Lookback duration (seconds): how far before `committed` we re-feed
/// to the model so it has acoustic left-context for the new portion.
const LOOKBACK_SECS: usize = 3;

/// Build the new-text portion of a window's transcription by keeping
/// only words whose start time is at or after the committed boundary.
///
/// The model can place a word a hair before the boundary even when the
/// word's audio is unambiguously "new" (timestamps drift ~100 ms in
/// either direction); we apply a small tolerance so we don't drop
/// legitimate words.
///
/// `boundary_secs` is the offset (in seconds, from window start) of
/// the committed-vs-new audio split inside this particular window.
pub(crate) fn splice_new_words(
    words: &[crate::backends::WordTimestamp],
    boundary_secs: f32,
) -> String {
    /// 60 ms tolerance — wider than the typical Whisper timestamp
    /// jitter (~30 ms) and narrow enough that we don't accidentally
    /// pull in a word from the lookback region. Duplicates that DO
    /// slip through are caught downstream by `boundary_overlap`.
    const TOLERANCE_SECS: f32 = 0.06;
    let cutoff = (boundary_secs - TOLERANCE_SECS).max(0.0);

    let mut out = String::new();
    for w in words {
        if w.start >= cutoff {
            out.push_str(&w.text);
        }
    }
    out.trim().to_string()
}

/// 10 ms hop in samples.
fn hop_size(rate: u32) -> usize {
    (rate as usize / 100).max(1)
}

/// Sustained-silence threshold for chunking decisions. Below this peak
/// amplitude is "quiet"; above is "speech". Tuned conservatively so
/// typical room noise (mic-floor 0.001–0.008) and breath sounds
/// (~0.01) don't register as speech, but quietly spoken words
/// (~0.04+) do.
const SILENCE_PEAK_THRESHOLD: f32 = 0.02;

/// Per-hop peak amplitudes over `samples`. Hops are 10 ms.
fn hop_peaks(samples: &[f32], rate: u32) -> Vec<f32> {
    let hop = hop_size(rate);
    let n_hops = samples.len() / hop;
    (0..n_hops)
        .map(|h| {
            let start = h * hop;
            samples[start..start + hop]
                .iter()
                .map(|s| s.abs())
                .fold(0.0_f32, f32::max)
        })
        .collect()
}

/// Find a chunk-cut point inside `samples` (offset relative to the slice).
///
/// Returns the sample index *within* `samples` of the midpoint of the
/// LATEST sustained silence run, or `None` if no run of ≥ 120 ms quiet
/// audio exists. We deliberately do NOT force-cut on fluent speech with
/// no pauses — slicing mid-word makes Whisper hallucinate complete
/// words on each side, and falling through to the release-time tail
/// pass is strictly better than that.
///
/// 120 ms minimum silence is enough to skip inter-phoneme gaps within
/// words (~10–30 ms) and short inter-word gaps inside fluent phrases
/// (~50–100 ms), and only fires on inter-phrase boundaries — which is
/// where chunk cuts belong.
fn find_silence_cut(samples: &[f32], rate: u32) -> Option<usize> {
    let hop = hop_size(rate);
    let min_quiet_hops = 12; // 120 ms

    let peaks = hop_peaks(samples, rate);
    let n_hops = peaks.len();
    if n_hops < min_quiet_hops {
        return None;
    }

    // Walk backwards looking for the latest sustained quiet run.
    let mut h = n_hops;
    while h > 0 {
        h -= 1;
        if peaks[h] >= SILENCE_PEAK_THRESHOLD {
            continue;
        }
        let mut start = h;
        while start > 0 && peaks[start - 1] < SILENCE_PEAK_THRESHOLD {
            start -= 1;
        }
        let end = h + 1;
        if end - start >= min_quiet_hops {
            return Some(((start + end) / 2) * hop);
        }
        if start == 0 {
            break;
        }
        // Skip past this too-short run and keep searching backwards.
        h = start;
    }

    None
}

/// Trim leading and trailing silence from `samples`, returning the
/// `(start, end)` byte range of the speech region — or `None` if the
/// whole clip is below the silence threshold.
///
/// Used on every pretranscribe chunk before we hand it to the backend.
/// Two reasons:
///
/// 1. **Whisper hallucinates on silence.** Feed it leading silence and
///    you reliably get "you", "Thanks for watching", "Bye", etc. —
///    these are training-data artifacts the model emits when it sees
///    nothing useful in the spectrogram. Trimming the silence removes
///    the trigger.
/// 2. **Each chunk is a self-contained utterance to Whisper.** Even
///    speech-padding silence on the edges nudges the model toward
///    starting/ending its sentence with filler. Tighter boundaries =
///    cleaner output.
///
/// We pad each end with ~80 ms back into the silent region so we don't
/// clip plosives or breath onsets.
fn trim_silence_padding(samples: &[f32], rate: u32) -> Option<(usize, usize)> {
    let hop = hop_size(rate);
    let peaks = hop_peaks(samples, rate);
    if peaks.is_empty() {
        return None;
    }
    let first_speech = peaks
        .iter()
        .position(|&p| p >= SILENCE_PEAK_THRESHOLD)?;
    let last_speech = peaks
        .iter()
        .rposition(|&p| p >= SILENCE_PEAK_THRESHOLD)?;

    let pad_hops = 8; // 80 ms pad on each side.
    let start_hop = first_speech.saturating_sub(pad_hops);
    let end_hop = (last_speech + 1 + pad_hops).min(peaks.len());

    let start = start_hop * hop;
    let end = (end_hop * hop).min(samples.len());
    if end <= start {
        return None;
    }
    Some((start, end))
}

/// Last `n` whitespace-separated words of `s`, joined with spaces.
/// Used as `initial_prompt` for chunk + tail decoding to preserve
/// stylistic / capitalization continuity across chunk boundaries.
fn tail_words(s: &str, n: usize) -> String {
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() <= n {
        s.trim().to_string()
    } else {
        words[words.len() - n..].join(" ")
    }
}

/// Append a streaming chunk's text onto `committed`, smoothing the join.
///
/// Whisper transcribes each chunk in isolation and treats it as its own
/// complete utterance, which produces two boundary artifacts:
///
/// 1. A trailing period (or `?` / `!` / `;` / `,`) on every chunk, even
///    when the audio cut mid-sentence — so naive concatenation gives
///    `"…for it. to do."` where the speaker actually meant
///    `"…for it to do."`.
/// 2. The same word duplicated at chunk boundaries when the cut lands
///    inside a brief silence in the middle of a word: the audio
///    "happen…" fades into silence in chunk N and the silence into
///    "…ening" opens chunk N+1, and Whisper hallucinates a complete
///    word at each side — producing `"happen. happening"`.
///
/// We correct both at the join:
///
/// - If the new chunk starts with a lowercase letter, the trailing
///   sentence-final punctuation on `committed` is almost certainly
///   spurious — strip it. (We don't strip when the next word starts
///   capital because that case is genuinely ambiguous: it could be a
///   real sentence break or a chunk-start over-capitalization, and
///   wrongly stripping a real one is more disruptive than leaving an
///   awkward join.)
/// - If the last alphanumeric word of `committed` equals the first
///   alphanumeric word of `new_chunk` (case-insensitively, ignoring
///   surrounding punctuation), drop the duplicate from the new chunk.
pub(crate) fn append_chunk_text(committed: &mut String, new_chunk: &str) {
    let new_chunk = new_chunk.trim();
    if new_chunk.is_empty() {
        return;
    }
    if committed.is_empty() {
        committed.push_str(new_chunk);
        return;
    }

    // (1) Boundary word dedup. Three patterns we recognize:
    //       - exact:        "go." + "go now"          → drop "go" from new
    //       - new is longer:"happen." + "happening"   → drop "happen" from committed
    //       - new is shorter:"happening." + "happen"  → drop "happen" from new
    //     Stem-prefix matches require ≥ 3 alphanumeric chars on the
    //     shorter side so "do" / "doing" doesn't over-correct.
    //     Done BEFORE the punctuation strip so that if the entire new
    //     chunk turns out to be a duplicate, the committed prefix's
    //     trailing period (which IS its real terminator at that point)
    //     stays put.
    let to_append_owned: String = match boundary_overlap(committed, new_chunk) {
        BoundaryOverlap::None => new_chunk.to_string(),
        BoundaryOverlap::DropFirstWordOfNew => new_chunk
            .trim_start()
            .splitn(2, char::is_whitespace)
            .nth(1)
            .map(str::trim_start)
            .unwrap_or("")
            .to_string(),
        BoundaryOverlap::DropLastWordOfCommitted => {
            let trimmed_len = committed.trim_end().len();
            committed.truncate(trimmed_len);
            let cut = committed.rfind(char::is_whitespace).unwrap_or(0);
            committed.truncate(cut);
            new_chunk.to_string()
        }
    };

    // Strip leading "..." / "…" / whitespace off the new chunk.
    // Whisper emits these when a chunk's audio starts mid-thought, and
    // they never read right concatenated to anything.
    let to_append = to_append_owned
        .trim_start_matches(|c: char| matches!(c, '.' | '\u{2026}' | ' ' | '\t'))
        .to_string();
    if to_append.is_empty() {
        return;
    }

    // (2) Decide whether this chunk boundary is a sentence break or a
    //     mid-sentence join. Whisper biases STRONGLY toward emitting a
    //     period at chunk ends and capitalizing the start of the next
    //     chunk — we have to undo that bias when there's evidence the
    //     thought is continuing, and only respect a break when there
    //     isn't.
    //
    //     Continuation signals (any of):
    //       (a) the appended chunk starts with a lowercase letter,
    //       (b) committed already ends in a "trailing" punctuation
    //           (comma or ellipsis) — Whisper's "thought trailing
    //           off" cue,
    //       (c) committed's last word is a function word (article,
    //           preposition, conjunction, modal/aux, common adverb)
    //           that almost never ends a real sentence in English.
    //
    //     When ANY of those fire we (i) strip terminal sentence
    //     punctuation off committed and (ii) lowercase the first
    //     letter of the appended chunk (preserving "I" / "I'm" /
    //     "I've" etc. and acronym-shaped tokens like "AI" / "LLM").
    let starts_lower = to_append.chars().next().is_some_and(char::is_lowercase);
    let trailing_off = committed_ends_with_trailing_punct(committed);
    let last_committed_word = last_alpha_word(committed);
    let last_is_function = is_function_word(&last_committed_word);
    let is_continuation = starts_lower || trailing_off || last_is_function;

    if is_continuation {
        let trimmed_end = committed
            .trim_end()
            .trim_end_matches(|c: char| {
                matches!(c, '.' | '!' | '?' | ';' | ',' | '\u{2026}')
            })
            .len();
        committed.truncate(trimmed_end);
    }

    let final_to_append: String = if is_continuation && !starts_lower {
        lowercase_first_word_unless_proper(&to_append)
    } else {
        to_append
    };

    if final_to_append.is_empty() {
        return;
    }

    if !committed.is_empty() && !committed.ends_with(char::is_whitespace) {
        committed.push(' ');
    }
    committed.push_str(&final_to_append);
}

/// True when `committed` ends in punctuation indicating the speaker was
/// trailing off mid-thought (comma, ellipsis). These are unambiguous
/// continuation signals from Whisper.
fn committed_ends_with_trailing_punct(committed: &str) -> bool {
    let trimmed = committed.trim_end();
    trimmed.ends_with(',')
        || trimmed.ends_with("...")
        || trimmed.ends_with('\u{2026}')
}

/// The final whitespace-delimited word of `committed`, lowercased and
/// stripped of non-alphanumeric chars. Empty string if committed has
/// no words.
fn last_alpha_word(committed: &str) -> String {
    committed
        .split_whitespace()
        .last()
        .map(alpha_lower)
        .unwrap_or_default()
}

/// True when `word` is a closed-class English function word that almost
/// never ends a real sentence — articles, conjunctions, prepositions,
/// modal/auxiliary verbs, and a few discourse markers / common adverbs
/// that Whisper loves to terminate chunks on.
///
/// Used to detect chunk-boundary periods that should be stripped: if
/// the previous committed text ends with one of these, the period
/// after it is almost certainly a Whisper artifact, not a real
/// sentence terminator.
fn is_function_word(word: &str) -> bool {
    matches!(
        word,
        // Articles / determiners
        "a" | "an" | "the" | "this" | "that" | "these" | "those"
        | "my" | "your" | "his" | "her" | "its" | "our" | "their"
        // Coordinating + subordinating conjunctions
        | "and" | "or" | "but" | "so" | "yet" | "nor"
        | "if" | "because" | "since" | "though" | "although"
        | "unless" | "until" | "while" | "when" | "as" | "than"
        // Prepositions
        | "to" | "of" | "with" | "from" | "at" | "by" | "in" | "on"
        | "up" | "down" | "out" | "off" | "over" | "under" | "into"
        | "onto" | "about" | "around" | "through" | "against"
        | "between" | "among" | "before" | "after" | "for"
        // Modal / auxiliary verbs (rarely sentence-final)
        | "is" | "was" | "are" | "were" | "be" | "been" | "being"
        | "am" | "have" | "has" | "had" | "do" | "does" | "did"
        | "will" | "would" | "could" | "should" | "might" | "must"
        | "may" | "can" | "shall"
        // Adverbs Whisper especially loves to terminate chunks on
        | "just" | "really" | "very" | "quite" | "still" | "also"
        | "too" | "even" | "only" | "actually" | "basically"
        | "literally"
        // Discourse markers
        | "like" | "well"
    )
}

/// Lowercase the first letter of `s`, except when the first word looks
/// like a proper noun, an acronym, or the pronoun "I" — those should
/// stay capitalized even mid-sentence.
fn lowercase_first_word_unless_proper(s: &str) -> String {
    let trimmed = s.trim_start();
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return s.to_string();
    };
    if !first.is_uppercase() {
        return s.to_string();
    }

    // Find the first whitespace-delimited token to inspect it.
    let first_word = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches(|c: char| !c.is_alphanumeric());

    // Pronoun "I" + its contractions.
    if matches!(
        first_word,
        "I" | "I'm" | "I've" | "I'd" | "I'll" | "I'M" | "I'VE" | "I'D" | "I'LL"
    ) {
        return s.to_string();
    }

    // ALL-CAPS short tokens — likely acronyms (AI, LLM, API, USA …).
    // 2–5 chars, all uppercase letters → leave alone.
    let alpha_only: String = first_word
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect();
    if (2..=5).contains(&alpha_only.len())
        && alpha_only.chars().all(|c| c.is_uppercase())
    {
        return s.to_string();
    }

    // Otherwise: lowercase the very first character and keep the rest.
    // We deliberately don't try to detect proper nouns beyond acronyms
    // and "I" — false negatives ("john said" instead of "John said")
    // are easier for the user to skim past than false positives
    // ("And then... And then..." sentence stutter).
    let mut out = String::with_capacity(s.len());
    let leading_ws_len = s.len() - trimmed.len();
    out.push_str(&s[..leading_ws_len]);
    for lower in first.to_lowercase() {
        out.push(lower);
    }
    out.push_str(chars.as_str());
    out
}

enum BoundaryOverlap {
    None,
    /// Drop the first whitespace-delimited token from the new chunk.
    DropFirstWordOfNew,
    /// Drop the last whitespace-delimited token from the committed prefix.
    DropLastWordOfCommitted,
}

fn boundary_overlap(committed: &str, new_chunk: &str) -> BoundaryOverlap {
    let Some(last) = committed.split_whitespace().last() else {
        return BoundaryOverlap::None;
    };
    let Some(first) = new_chunk.split_whitespace().next() else {
        return BoundaryOverlap::None;
    };
    let last_alpha = alpha_lower(last);
    let first_alpha = alpha_lower(first);
    if last_alpha.is_empty() || first_alpha.is_empty() {
        return BoundaryOverlap::None;
    }
    if last_alpha == first_alpha {
        return BoundaryOverlap::DropFirstWordOfNew;
    }
    const MIN_STEM: usize = 3;
    if last_alpha.len() >= MIN_STEM && first_alpha.starts_with(last_alpha.as_str()) {
        return BoundaryOverlap::DropLastWordOfCommitted;
    }
    if first_alpha.len() >= MIN_STEM && last_alpha.starts_with(first_alpha.as_str()) {
        return BoundaryOverlap::DropFirstWordOfNew;
    }
    BoundaryOverlap::None
}

fn alpha_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

pub(crate) async fn release_to_talk(bundle: &ArcsBundle, app: &AppHandle) -> Result<()> {
    if !bundle.audio.is_recording() {
        return Ok(());
    }
    let frame = bundle.audio.stop();
    let raw_samples = frame.samples;
    let raw_rate = frame.sample_rate;

    // Diagnostic snapshot of the captured audio at its native rate.
    // Peak < ~0.005 means the mic effectively heard silence (or is muted /
    // wrong device); a high peak with low RMS suggests a click rather
    // than speech. RMS is the most useful figure — a normal speaking
    // voice sits around 0.05 - 0.2 in the 16-bit-normalized domain.
    let rms = if raw_samples.is_empty() {
        0.0
    } else {
        let sum_sq: f64 = raw_samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum_sq / raw_samples.len() as f64).sqrt() as f32
    };
    let secs = raw_samples.len() as f32 / raw_rate as f32;
    info!(
        samples = raw_samples.len(),
        rate = raw_rate,
        seconds = secs,
        peak = frame.peak,
        rms,
        "captured utterance"
    );

    // Save the raw input audio at the device's native rate. This is now
    // the *only* WAV the pipeline writes — the model receives the same
    // bytes — so if it plays back fine but the model still hallucinates,
    // the bug is on the model side, not in our audio path.
    if let Ok(dir) = app.path().app_data_dir() {
        let raw_path = dir.join("debug-last-raw.wav");
        match crate::audio::write_wav(&raw_path, &raw_samples, raw_rate) {
            Ok(_) => info!(path = %raw_path.display(), rate = raw_rate, "saved raw audio"),
            Err(e) => warn!(error = %e, "failed to save raw audio"),
        }
    }

    // Discard accidental taps. We compare against ~300 ms of audio at
    // the device's native rate.
    let min_native_samples = (raw_rate as usize / 1000) * 300;
    if raw_samples.len() < min_native_samples {
        info!(samples = raw_samples.len(), "discarded short utterance");
        // Cancel any in-flight pretranscribe chunk; its result is irrelevant
        // because we're discarding the utterance.
        if let Some(h) = bundle.pretranscribe_task.lock().take() {
            h.abort();
        }
        *bundle.pretranscribe_state.lock() = PreTranscribeState::default();
        overlay::hide(app);
        return Ok(());
    }

    overlay::emit(
        app,
        OverlayPayload {
            state: OverlayState::Transcribing,
            message: None,
            peak: None,
        },
    );

    let (vad_enabled, opts, post_action) = {
        let s = bundle.settings.read().await;
        (s.vad_enabled, s.backend_options.clone(), s.post_action)
    };

    // Wait for the streaming pretranscribe task to drain. After
    // `audio.stop()` flipped `is_recording` to false, the task will exit
    // at the top of its next iteration — but if it's currently inside a
    // chunk's `transcribe(...)` await, we wait for that chunk to finish
    // (and update committed state) before we pull the prefix.
    //
    // The wait is bounded by the time it takes one chunk to transcribe.
    // For sidecar backends this is also where we pay the cost the user
    // called out: the sidecar serializes inference, so the tail's call
    // can only start after the in-flight chunk completes. The win is
    // that the tail itself transcribes a small slice of audio.
    //
    // NOTE: we take the JoinHandle out of the parking_lot mutex into a
    // local before awaiting it. parking_lot's MutexGuard isn't `Send`
    // and the surrounding tokio task requires a `Send` future.
    let pretranscribe_handle = bundle.pretranscribe_task.lock().take();
    if let Some(handle) = pretranscribe_handle {
        if tokio::time::timeout(std::time::Duration::from_secs(30), handle)
            .await
            .is_err()
        {
            warn!("pretranscribe task did not exit within 30s; abandoning");
        }
    }

    // Take the committed prefix. Reset the shared state so a
    // subsequent recording starts clean even if any code path below
    // bails early.
    let pre = std::mem::take(&mut *bundle.pretranscribe_state.lock());
    // Same-rate guard: the pretranscribe task records the rate in
    // effect when each chunk was captured. If the user changed input
    // device mid-recording (rare but possible) the rates won't match
    // and we can't safely splice the prefix into the final buffer.
    let (committed, committed_text) = if pre.sample_rate == raw_rate
        && pre.committed_samples <= raw_samples.len()
    {
        (pre.committed_samples, pre.text)
    } else {
        if pre.committed_samples > 0 {
            warn!(
                pre_rate = pre.sample_rate,
                raw_rate,
                pre_samples = pre.committed_samples,
                raw_samples = raw_samples.len(),
                "discarding pretranscribe prefix (rate / length mismatch)"
            );
        }
        (0, String::new())
    };

    // Build the audio that goes into the tail/final-pass inference.
    //
    // Two cases:
    //
    //  1. **No committed prefix** (streaming was off, no chunks fired,
    //     or pretranscribe-disabled): the "tail" is the whole
    //     utterance. Run it through VAD (if enabled) plus energy
    //     trim, exactly like the pre-streaming code path used to —
    //     this is the single-pass baseline.
    //
    //  2. **Committed prefix exists**: treat the release pass as one
    //     more window in the streaming pipeline. Pull `LOOKBACK_SECS`
    //     of audio from BEFORE the committed boundary so the model
    //     has acoustic context for the words that border the splice
    //     point, request word timestamps, and keep only the words
    //     whose start time lands at or after the boundary. This is
    //     the same logic `spawn_pretranscribe_task` uses; doing it
    //     once more on release means the LAST few seconds of speech
    //     (which are the most likely to be remembered by the user)
    //     get the full-context-on-both-sides treatment too.
    let lookback_secs_for_tail = if committed > 0 { LOOKBACK_SECS } else { 0 };
    let lookback_samples = (raw_rate as usize * lookback_secs_for_tail).min(committed);
    let window_start = committed - lookback_samples;
    let raw_window = &raw_samples[window_start..];

    let after_vad: Vec<f32> = if raw_window.is_empty() {
        Vec::new()
    } else if vad_enabled && committed == 0 {
        // VAD path: only used when there's no committed prefix
        // (we'd lose the lookback↔buffer offset mapping otherwise,
        // and VAD's main job — detecting "is there ANY speech" —
        // is irrelevant once a chunk has already committed text).
        match run_vad_trim(raw_window, raw_rate) {
            Ok(Some(trimmed)) => trimmed,
            Ok(None) => {
                info!("VAD found no speech in tail");
                Vec::new()
            }
            Err(e) => {
                warn!(error = %e, "VAD failed on tail; using untrimmed");
                raw_window.to_vec()
            }
        }
    } else {
        raw_window.to_vec()
    };

    // Energy-based trim with offset-tracking so we can map word
    // timestamps back to absolute sample positions in the raw buffer.
    // (When VAD ran above, `effective_window_start` tracks the
    // post-VAD start — which we approximate by skipping over leading
    // silence in `after_vad` here too. Good enough since the
    // committed-prefix path skips VAD anyway.)
    let (window_audio, effective_window_start) = if after_vad.is_empty() {
        (Vec::new(), window_start)
    } else {
        match trim_silence_padding(&after_vad, raw_rate) {
            Some((s, e)) => (after_vad[s..e].to_vec(), window_start + s),
            None => (Vec::new(), window_start),
        }
    };

    // Debug aid: save the (post-trim) audio that's about to be handed
    // to the backend. Same sample rate as the raw file so a/b
    // comparison is clean.
    if let Ok(dir) = app.path().app_data_dir() {
        if !window_audio.is_empty() {
            let debug_path = dir.join("debug-last-utterance.wav");
            match crate::audio::write_wav(&debug_path, &window_audio, raw_rate) {
                Ok(_) => info!(path = %debug_path.display(), rate = raw_rate, "saved debug audio"),
                Err(e) => warn!(error = %e, "failed to save debug audio"),
            }
        }
    }

    // committed_secs: where inside the trimmed window the committed
    // boundary falls. 0 when the lookback was entirely silence (or no
    // lookback was used).
    let committed_secs = if effective_window_start >= committed {
        0.0
    } else {
        (committed - effective_window_start) as f32 / raw_rate as f32
    };

    // Three cases for the final text:
    //   1. Window has audio → transcribe it (with word timestamps if
    //      we used lookback so we can splice precisely).
    //   2. Window empty but committed_text non-empty → use committed
    //      as-is (entire utterance was covered by streaming chunks
    //      and the trailing audio is silence).
    //   3. Both empty → silent utterance, abort.
    // Self-heal before the tail pass. Right after app start (or after a
    // sidecar crash) the backend may not be ready yet even though audio
    // captured fine. Waiting for the load here turns "first dictation
    // after boot errors out" into "first dictation after boot takes a few
    // extra seconds" — the audio is already in hand, nothing is lost.
    if !window_audio.is_empty() {
        let ready = {
            let backend = bundle.backend.read().await;
            match backend.as_ref() {
                Some(b) => b.is_ready().await,
                None => false,
            }
        };
        if !ready {
            info!("backend not ready at release; bringing it up now");
            overlay::emit(
                app,
                OverlayPayload {
                    state: OverlayState::Transcribing,
                    message: Some("Loading model…".into()),
                    peak: None,
                },
            );
            if let Err(e) = ensure_backend_ready_bundle(bundle).await {
                warn!(error = %e, "backend bring-up at release failed");
                overlay::emit(
                    app,
                    OverlayPayload {
                        state: OverlayState::Error,
                        message: Some(e.to_string()),
                        peak: None,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(1800)).await;
                overlay::hide(app);
                return Err(e);
            }
        }
    }

    let (raw_text, language, duration_ms, audio_seconds) = if !window_audio.is_empty() {
        let mut tail_opts = opts.clone();
        // Always request word timestamps when there's a committed
        // prefix to splice against. Cheap on Whisper-family backends
        // and a no-op (returns []) on backends that don't support it.
        tail_opts.word_timestamps = committed > 0;
        if tail_opts.initial_prompt.is_none() && !committed_text.is_empty() {
            tail_opts.initial_prompt = Some(tail_words(&committed_text, 32));
        }

        let started_tail = std::time::Instant::now();
        let result = {
            let backend = bundle.backend.read().await;
            match backend.as_ref() {
                Some(b) => b.transcribe(&window_audio, raw_rate, &tail_opts).await,
                // Backend vanished between the readiness check and here —
                // fall into the error path below so the overlay is cleaned
                // up rather than left stuck on "Transcribing".
                None => Err(AppError::BackendNotReady),
            }
        };
        let mut result = match result {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "tail transcription failed");
                overlay::emit(
                    app,
                    OverlayPayload {
                        state: OverlayState::Error,
                        message: Some(e.to_string()),
                        peak: None,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                overlay::hide(app);
                return Err(e);
            }
        };

        // If we used lookback but the backend returned no word
        // timestamps, the result text covers the lookback region
        // too — concatenating it to committed_text would duplicate
        // a few seconds of words. Re-run inference on JUST the
        // post-committed audio (the old chunked-style behaviour).
        if committed > 0 && result.words.is_empty() {
            warn!(
                "release window: backend has no word timestamps; \
                 retrying on post-committed audio without lookback"
            );
            let post_committed = &raw_samples[committed..];
            let no_lookback_audio: Vec<f32> = match trim_silence_padding(post_committed, raw_rate)
            {
                Some((s, e)) => post_committed[s..e].to_vec(),
                None => Vec::new(),
            };
            if !no_lookback_audio.is_empty() {
                let mut alt_opts = opts.clone();
                alt_opts.initial_prompt = Some(tail_words(&committed_text, 32));
                let alt_result = {
                    let backend = bundle.backend.read().await;
                    match backend.as_ref() {
                        Some(b) => {
                            b.transcribe(&no_lookback_audio, raw_rate, &alt_opts).await
                        }
                        None => Err(AppError::BackendNotReady),
                    }
                };
                if let Ok(alt) = alt_result {
                    result = alt;
                }
            }
        }

        let tail_text = if !result.words.is_empty() {
            splice_new_words(&result.words, committed_secs)
        } else {
            result.text.trim().to_string()
        };

        info!(
            committed_samples = committed,
            window_samples = window_audio.len(),
            used_words = !result.words.is_empty(),
            committed_secs,
            tail_elapsed_ms = started_tail.elapsed().as_millis() as u64,
            "tail transcription done"
        );

        let mut combined = committed_text;
        append_chunk_text(&mut combined, &tail_text);
        let secs = raw_samples.len() as f32 / raw_rate as f32;
        (combined, result.language, result.duration_ms, secs)
    } else if !committed_text.is_empty() {
        info!(
            committed_samples = committed,
            "release covered entirely by committed prefix; no tail pass needed"
        );
        let secs = raw_samples.len() as f32 / raw_rate as f32;
        (committed_text, None, 0u64, secs)
    } else {
        info!("nothing to transcribe (no committed prefix and tail had no speech)");
        overlay::hide(app);
        return Ok(());
    };

    if raw_text.trim().is_empty() {
        info!("transcription empty; skipping paste");
        overlay::hide(app);
        return Ok(());
    }

    // Optional post-processing pass through Ollama.
    let final_text = run_post_processing(bundle, &raw_text)
        .await
        .unwrap_or_else(|e| {
            warn!(error = %e, "post-processing failed; using raw text");
            raw_text.clone()
        });

    // History — done before paste so a paste failure still leaves a record.
    let backend_kind = {
        let backend = bundle.backend.read().await;
        backend.as_ref().map(|b| b.kind())
    };
    let model_id = bundle.settings.read().await.active_model_id.clone();
    let _ = bundle.history.insert(&NewHistoryEntry {
        text: final_text.clone(),
        backend: backend_kind.map(|k| k.label().to_string()).unwrap_or_default(),
        model: model_id,
        language,
        duration_ms: duration_ms as i64,
        audio_seconds,
    });
    let max_history = bundle.settings.read().await.max_history;
    let _ = bundle.history.trim(max_history);
    // Notify any open History view so it can refresh without a tab switch.
    let _ = app.emit("history://updated", ());

    // Paste / clipboard. A failure here must still clean up the overlay —
    // the transcription is already saved to history, so the user can
    // recover it with the repaste hotkey.
    let paste_result = match post_action {
        PostAction::Paste => paste::copy_and_paste(app, &final_text),
        PostAction::ClipboardOnly => paste::copy_to_clipboard(app, &final_text),
    };
    if let Err(e) = paste_result {
        warn!(error = %e, "paste failed; text is in history");
        overlay::emit(
            app,
            OverlayPayload {
                state: OverlayState::Error,
                message: Some(format!("paste failed: {e}")),
                peak: None,
            },
        );
        tokio::time::sleep(std::time::Duration::from_millis(1800)).await;
        overlay::hide(app);
        return Err(e);
    }

    overlay::emit(
        app,
        OverlayPayload {
            state: OverlayState::Pasted,
            message: None,
            peak: None,
        },
    );
    tokio::time::sleep(std::time::Duration::from_millis(450)).await;
    overlay::hide(app);
    Ok(())
}

/// Run Silero VAD on a 16 kHz scratch copy of `samples` and return
/// the speech region from the original native-rate buffer. The 16 kHz
/// copy is only used to find the boundaries — the returned slice is
/// pulled from `samples` directly so no resampler artifacts ever
/// reach the model.
fn run_vad_trim(samples: &[f32], rate: u32) -> std::result::Result<Option<Vec<f32>>, String> {
    let resampled_for_vad = if rate == TARGET_SAMPLE_RATE {
        samples.to_vec()
    } else {
        crate::audio::resample_mono(samples, rate, TARGET_SAMPLE_RATE)?
    };

    let mut vad = crate::audio::vad::SileroVad::new()?;
    let Some((start_sec, end_sec)) = vad.trim_bounds_sec(&resampled_for_vad) else {
        return Ok(None);
    };

    let start = (start_sec * rate as f32) as usize;
    let end = ((end_sec * rate as f32) as usize).min(samples.len());
    if end <= start {
        return Ok(None);
    }
    Ok(Some(samples[start..end].to_vec()))
}

async fn run_post_processing(bundle: &ArcsBundle, raw: &str) -> Result<String> {
    let cfg = bundle.settings.read().await.post_processing.clone();
    if !cfg.enabled || raw.trim().is_empty() {
        return Ok(raw.to_string());
    }
    match cfg.provider {
        PostProcessingProvider::Ollama => run_post_processing_ollama(&cfg, raw).await,
        PostProcessingProvider::CursorCli => run_post_processing_cursor_cli(&cfg, raw).await,
        PostProcessingProvider::Transformers => {
            run_post_processing_transformers(bundle, &cfg, raw).await
        }
    }
}

/// Resolve the configured cleanup model + push it into the sidecar's
/// cleanup slot. Idempotent — the sidecar no-ops a second `load_model`
/// for the same `(engine, model_path)` pair, so calling this from both
/// the lazy cleanup-pass path AND the explicit Preload button is fine.
async fn ensure_cleanup_model_loaded(
    sidecar: &Arc<crate::sidecar::SidecarSupervisor>,
    models: &Arc<crate::models::ModelRegistry>,
    cfg: &PostProcessing,
) -> Result<()> {
    let model_id = cfg
        .transformers_model_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            AppError::Other(
                "no Transformers cleanup model selected. Pick one in Settings → Backend → Post-processing."
                    .into(),
            )
        })?;

    let local = models.get(model_id).ok_or_else(|| {
        AppError::Model(format!(
            "cleanup model `{model_id}` not found in registry. Re-download it from the Models tab."
        ))
    })?;

    let model_path = local
        .path
        .to_str()
        .ok_or_else(|| AppError::Other("non-UTF-8 cleanup model path".into()))?
        .to_string();

    let load_params = LoadModelParams {
        engine: Engine::Cleanup,
        model: model_path,
        options: serde_json::json!({
            "device": cfg.transformers_device,
        }),
    };
    let _: LoadModelResult = sidecar.call("load_model", load_params).await?;
    Ok(())
}

/// Run the cleanup pass through a HuggingFace causal-LM in the Python
/// sidecar's cleanup slot. The sidecar keeps the cleanup model resident
/// alongside the active ASR model — first call after a settings change
/// pays the load cost, every call after is just `generate()`.
async fn run_post_processing_transformers(
    bundle: &ArcsBundle,
    cfg: &PostProcessing,
    raw: &str,
) -> Result<String> {
    ensure_cleanup_model_loaded(&bundle.sidecar, &bundle.models, cfg).await?;

    let cleanup_params = CleanupParams {
        system_prompt: cfg.system_prompt.clone(),
        raw_text: raw.to_string(),
        options: serde_json::json!({
            "max_new_tokens": cfg.transformers_max_tokens,
        }),
    };
    let result: CleanupResult = bundle.sidecar.call("cleanup", cleanup_params).await?;
    Ok(strip_post_processed_artifacts(&result.text))
}

/// Eagerly load the configured cleanup model so the first dictation
/// after enabling cleanup doesn't pay the multi-second model-load
/// cost. The Backend route exposes a "Preload" button that fires this.
#[tauri::command]
pub async fn preload_cleanup_model(state: State<'_, AppState>) -> Result<()> {
    let cfg = state.settings.read().await.post_processing.clone();
    if !matches!(cfg.provider, PostProcessingProvider::Transformers) {
        return Err(AppError::Other(
            "post-processing provider is not Transformers — nothing to preload.".into(),
        ));
    }
    ensure_cleanup_model_loaded(&state.sidecar, &state.models, &cfg).await
}

async fn run_post_processing_ollama(cfg: &PostProcessing, raw: &str) -> Result<String> {
    // Ollama /api/generate returns a stream by default. Disable it.
    let body = serde_json::json!({
        "model": cfg.model,
        "prompt": raw,
        "system": cfg.system_prompt,
        "stream": false,
    });
    let url = format!("{}/api/generate", cfg.ollama_url.trim_end_matches('/'));
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Http(format!("ollama: {e}")))?;
    if !resp.status().is_success() {
        return Err(AppError::Http(format!(
            "ollama returned {}",
            resp.status()
        )));
    }
    #[derive(Deserialize)]
    struct OllamaResp {
        response: String,
    }
    let parsed: OllamaResp = resp.json().await?;
    Ok(parsed.response.trim().to_string())
}

/// Shell out to `cursor-agent -p` to run the cleanup pass through whichever
/// Cursor model the user picked (Auto / Composer / etc).
///
/// Cursor CLI doesn't have a separate `--system` flag, so we inline the
/// system prompt and the dictated text into a single positional `-p`
/// argument with explicit delimiters and a trailing instruction telling the
/// model to emit ONLY the cleaned text — no preamble, no fences. We also
/// pass `--mode ask` so the agent can't make file edits or run shell
/// commands, and `--output-format text` so we get a clean final string on
/// stdout instead of the JSON event stream.
///
/// Auth is whatever Cursor CLI is already signed in as on this machine
/// (i.e. `cursor-agent login` was run once, or `CURSOR_API_KEY` is set in
/// the host's environment). We don't manage credentials here.
async fn run_post_processing_cursor_cli(cfg: &PostProcessing, raw: &str) -> Result<String> {
    use tokio::process::Command;

    let configured = cfg
        .cursor_cli_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("cursor-agent");

    let launch = resolve_cursor_cli_launch(configured).ok_or_else(|| {
        AppError::Other(format!(
            "could not locate `{configured}`. Tried PATH (with PATHEXT) and the standard install location at `%LOCALAPPDATA%\\cursor-agent\\`. \
             Install with `irm 'https://cursor.com/install?win32=true' | iex` (Windows) or set an explicit path in Settings → Backend → Post-processing."
        ))
    })?;

    let model = if cfg.cursor_model.trim().is_empty() {
        "auto"
    } else {
        cfg.cursor_model.trim()
    };

    // The agent CLI expects the entire prompt as one string. Wrap the
    // dictated text in a clearly-marked block so even an instruction-tuned
    // model that's tempted to "respond conversationally" knows exactly
    // which span to clean up and pass back.
    let combined = format!(
        "{system}\n\nThe text to clean is between the markers below. Return ONLY the cleaned version of that text — no explanation, no preamble, no Markdown, no code fences.\n\n<<<DICTATED_TEXT\n{raw}\nDICTATED_TEXT>>>",
        system = cfg.system_prompt.trim(),
        raw = raw,
    );

    // Pin the agent to a dedicated empty workspace under the OS temp dir.
    // Reasons:
    //   1. Without --workspace the agent inherits our CWD (`src-tauri`),
    //      which trips its workspace-trust prompt and exits 1 in headless
    //      mode.
    //   2. We don't want the agent crawling the user's source tree — this
    //      is a pure text-cleanup task. An empty dir gives it nothing
    //      relevant to read even if the model decides to look.
    // Using a stable path means we don't churn through fresh temp dirs on
    // every invocation, and the trust marker (if any) persists.
    let workspace = std::env::temp_dir().join("talk-to-type-cursor-cli");
    let _ = std::fs::create_dir_all(&workspace);

    // Spawn the resolved program with any prepended args (e.g. node.exe +
    // index.js when we found the bundled-Node layout), then our flags.
    //
    // We deliberately route through node.exe rather than the `.cmd` shim
    // when possible: Rust's batch-file safety check (CVE-2024-24576) refuses
    // to pass arguments containing newlines through a `.cmd` invocation,
    // and our prompt is multi-line.
    let mut cmd = Command::new(&launch.program);
    for arg in &launch.leading_args {
        cmd.arg(arg);
    }
    cmd.arg("--print")
        .arg("--output-format")
        .arg("text")
        .arg("--mode")
        .arg("ask")
        .arg("--model")
        .arg(model)
        // Skip the interactive workspace-trust gate; --print sessions
        // can't answer the prompt and would otherwise exit 1.
        .arg("--trust")
        .arg("--workspace")
        .arg(&workspace)
        .arg(&combined);

    // Run from inside the sandboxed workspace too, so any default
    // "current directory" semantics inside the agent line up with the
    // workspace flag.
    cmd.current_dir(&workspace);

    // Mirror the env vars the bundled PowerShell launcher sets, so the
    // node entry point sees the same world it would inside a normal shell
    // invocation. `CURSOR_INVOKED_AS` is used by the CLI for its help
    // strings; `NODE_COMPILE_CACHE` is a startup-time cache that shaves
    // ~100 ms per launch on cold runs.
    cmd.env("CURSOR_INVOKED_AS", "cursor-agent");
    if std::env::var_os("NODE_COMPILE_CACHE").is_none() {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            let mut cache = std::path::PathBuf::from(local_app_data);
            cache.push("cursor-compile-cache");
            cmd.env("NODE_COMPILE_CACHE", cache);
        }
    }

    // Don't let the spawned process inherit our stdin — it would block on
    // the parent's terminal otherwise.
    cmd.stdin(std::process::Stdio::null());

    // Hide the console window on Windows so spawning the CLI doesn't flash
    // a black box every time we paste.
    // tokio::process::Command exposes `creation_flags` as an inherent method
    // on Windows targets, so we don't need to pull in `CommandExt` here.
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output().await.map_err(|e| {
        AppError::Other(format!(
            "failed to spawn cursor-agent at `{}`: {e}.",
            launch.program.display()
        ))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(AppError::Other(format!(
            "cursor-agent exited with {}: {}{}",
            output.status,
            stderr.trim(),
            if stderr.trim().is_empty() && !stdout.trim().is_empty() {
                format!(" (stdout: {})", stdout.trim())
            } else {
                String::new()
            },
        )));
    }

    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    Ok(strip_post_processed_artifacts(&text))
}

/// How to actually invoke `cursor-agent` on this machine.
///
/// On Windows the official installer ships a `cursor-agent.cmd` shim that
/// shells out to PowerShell, which then calls a bundled `node.exe` against
/// `index.js`. Going through the `.cmd` would force every argument through
/// Rust's batch-file safety escaping (CVE-2024-24576), which refuses
/// arguments containing newlines — fatal for a multi-line system prompt.
/// So when we can find the bundled Node + `index.js`, we point straight at
/// those and put `index.js` in `leading_args` so it shows up before the
/// agent's own flags.
struct CursorCliLaunch {
    program: std::path::PathBuf,
    leading_args: Vec<std::ffi::OsString>,
}

/// Locate `cursor-agent` and decide how to launch it. See
/// [`CursorCliLaunch`] for the rationale behind the two-tier shape.
///
/// Resolution order:
///   1. Walk PATH ourselves and probe each entry against PATHEXT — fixes
///      the common Windows bug where `Command::new("cursor-agent")` can't
///      see `cursor-agent.cmd` because `CreateProcessW` ignores PATHEXT.
///   2. Fall back to the installer's well-known directories. The Tauri
///      host inherits PATH from whatever launched it (Explorer, the IDE,
///      a service manager, ...), which often lacks the freshly-added
///      installer entry until next login.
///   3. Once we have a path, if it's a `.cmd`/`.bat`/`.ps1` shim AND we
///      can find `node.exe` + `index.js` either next to it or under
///      `versions/<latest>/`, swap in that pair. Otherwise launch the
///      shim directly (Rust 1.77.2+ handles the safe-arg subset for us).
fn resolve_cursor_cli_launch(configured: &str) -> Option<CursorCliLaunch> {
    let resolved = resolve_cursor_cli_binary(configured)?;

    #[cfg(windows)]
    {
        let ext = resolved
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        if matches!(ext.as_deref(), Some("cmd") | Some("bat") | Some("ps1")) {
            if let Some(parent) = resolved.parent() {
                if let Some(spec) = find_bundled_node_launch(parent) {
                    return Some(spec);
                }
            }
        }
    }

    Some(CursorCliLaunch {
        program: resolved,
        leading_args: Vec::new(),
    })
}

/// Look for `node.exe` + `index.js` underneath the cursor-agent install
/// dir. Returns a launch spec that runs them directly so we bypass the
/// shell shims.
///
/// Mirrors the discovery logic in `cursor-agent.ps1`:
///   - First checks for a flat layout (`<dir>\node.exe`, `<dir>\index.js`).
///   - Then falls back to versioned subdirs (`<dir>\versions\<latest>\...`).
///     Cursor's version dirs follow `YYYY.MM.DD-<commit>`, which sorts
///     correctly lexicographically, so we sort `read_dir` descending and
///     pick the first hit.
#[cfg(windows)]
fn find_bundled_node_launch(install_dir: &std::path::Path) -> Option<CursorCliLaunch> {
    let direct_node = install_dir.join("node.exe");
    let direct_index = install_dir.join("index.js");
    if direct_node.is_file() && direct_index.is_file() {
        return Some(CursorCliLaunch {
            program: direct_node,
            leading_args: vec![direct_index.into_os_string()],
        });
    }

    let versions = install_dir.join("versions");
    if versions.is_dir() {
        let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(&versions)
            .ok()?
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .collect();
        // YYYY.MM.DD-<commit> sorts cleanly with a lexicographic descending sort.
        entries.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
        for entry in entries {
            let v = entry.path();
            let n = v.join("node.exe");
            let i = v.join("index.js");
            if n.is_file() && i.is_file() {
                return Some(CursorCliLaunch {
                    program: n,
                    leading_args: vec![i.into_os_string()],
                });
            }
        }
    }

    None
}

/// PATH+PATHEXT walk plus known-install-location fallbacks. Returns just
/// the resolved file path; the launch-spec layer above decides what to do
/// with it.
fn resolve_cursor_cli_binary(configured: &str) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};

    fn probe_with_pathext(candidate: &Path) -> Option<PathBuf> {
        if candidate.is_file() {
            return Some(candidate.to_path_buf());
        }
        // Only retry with PATHEXT if the original didn't already carry an
        // extension. Otherwise `cursor-agent.cmd` would happily promote
        // itself to `cursor-agent.exe` if that ALSO existed somewhere.
        if candidate.extension().is_some() {
            return None;
        }
        #[cfg(windows)]
        {
            let pathext = std::env::var("PATHEXT")
                .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
            for ext in pathext.split(';') {
                let trimmed = ext.trim_start_matches('.').trim();
                if trimmed.is_empty() {
                    continue;
                }
                let with_ext = candidate.with_extension(trimmed.to_ascii_lowercase());
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
        }
        None
    }

    let configured_path = Path::new(configured);
    let looks_like_path = configured.contains('\\')
        || configured.contains('/')
        || configured_path.is_absolute();

    if looks_like_path {
        return probe_with_pathext(configured_path);
    }

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if let Some(found) = probe_with_pathext(&dir.join(configured)) {
                return Some(found);
            }
        }
    }

    #[cfg(windows)]
    {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            let lad = PathBuf::from(local_app_data);
            for sub in [
                lad.join("cursor-agent"),
                lad.join("Programs").join("cursor-agent"),
            ] {
                if let Some(found) = probe_with_pathext(&sub.join(configured)) {
                    return Some(found);
                }
            }
        }
    }

    #[cfg(unix)]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            for sub in [home.join(".local").join("bin"), home.join("bin")] {
                if let Some(found) = probe_with_pathext(&sub.join(configured)) {
                    return Some(found);
                }
            }
        }
        for sub in ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin"] {
            if let Some(found) = probe_with_pathext(&Path::new(sub).join(configured)) {
                return Some(found);
            }
        }
    }

    None
}

/// Trim whitespace and unwrap a single ```fenced``` block if the model
/// included one despite the prompt asking it not to. We only strip a
/// single, outermost fence — anything more aggressive risks eating real
/// content (the user might dictate the word "backtick").
fn strip_post_processed_artifacts(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") && trimmed.ends_with("```") && trimmed.len() >= 6 {
        let inner = &trimmed[3..trimmed.len() - 3];
        // Drop the language tag on the opening fence (e.g. ```text\n…).
        let inner = inner.split_once('\n').map(|(_, rest)| rest).unwrap_or(inner);
        return inner.trim().to_string();
    }
    trimmed.to_string()
}

// =============================================================================
// Models
// =============================================================================

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubSearchResult {
    pub repo_id: String,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub last_modified: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
}

impl From<HubModel> for HubSearchResult {
    fn from(m: HubModel) -> Self {
        Self {
            repo_id: m.repo_id,
            author: m.author,
            downloads: m.downloads,
            likes: m.likes,
            last_modified: m.last_modified,
            pipeline_tag: m.pipeline_tag,
            tags: m.tags,
        }
    }
}

#[tauri::command]
pub async fn search_hub_models(
    state: State<'_, AppState>,
    backend: BackendKind,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<HubSearchResult>> {
    let token = state.settings.read().await.hf_token.clone();
    let client = HubClient::new(token);
    let pipeline = pipeline_tag_for_backend(backend);
    let raw = client.search(&query, limit.unwrap_or(40), pipeline).await?;
    let mut filtered = Vec::new();
    for m in raw {
        // Cheap pass first (id/tags only). For ambiguous cases we'd need
        // /tree/main, but we skip that here to keep search snappy.
        if crate::models::filter_for_backend(backend, &m, None) {
            filtered.push(m.into());
        }
    }
    Ok(filtered)
}

#[tauri::command]
pub fn list_local_models(state: State<'_, AppState>, backend: Option<BackendKind>) -> Vec<LocalModel> {
    match backend {
        Some(b) => state.models.list_for(b),
        None => state.models.list(),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadRequest {
    pub backend: BackendKind,
    pub repo_id: String,
    /// For Transformers we accept `null` — we'll write the marker file and
    /// let the sidecar pull weights via the HF cache on first load.
    pub file: Option<String>,
    pub display_name: String,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    req: DownloadRequest,
) -> Result<LocalModel> {
    let mgr = state.downloads.clone();
    let token = state.settings.read().await.hf_token.clone();
    let client = HubClient::new(token);

    // Multi-file backends download a per-repo directory:
    //   - Transformers (ASR + cleanup): the from_pretrained() file set
    //     (config, tokenizer, preprocessor/processor, weights).
    //   - faster-whisper: the CTranslate2 set — model.bin alone is NOT
    //     loadable; WhisperModel() also needs config.json, tokenizer.json
    //     and vocabulary files.
    if req.file.is_none() {
        let select: fn(&[crate::models::HubModelFile]) -> Vec<&crate::models::HubModelFile> =
            match req.backend {
                BackendKind::Transformers | BackendKind::CleanupTransformers => {
                    select_for_transformers
                }
                BackendKind::FasterWhisper => crate::models::select_for_faster_whisper,
                BackendKind::WhisperCpp => {
                    return Err(AppError::Model(
                        "file is required for whisper.cpp downloads".into(),
                    ))
                }
            };
        return download_repo_files(app, state, req, client, mgr, select).await;
    }

    let file = req
        .file
        .clone()
        .ok_or_else(|| AppError::Model("file is required for this backend".into()))?;
    let url = client.resolve_url(&req.repo_id, &file);

    let dir = state.models.dir_for(req.backend);
    let safe_repo = req.repo_id.replace('/', "__");
    let dest = dir.join(&safe_repo).join(&file);

    let job_id = Uuid::new_v4().to_string();
    let dest = mgr
        .clone()
        .download(
            app,
            client.download_http().clone(),
            client.token().map(|t| t.to_string()),
            job_id,
            req.repo_id.clone(),
            file.clone(),
            url,
            dest,
        )
        .await?;

    let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    let local = LocalModel {
        id: format!("{:?}:{}:{}", req.backend, req.repo_id, file),
        backend: req.backend,
        name: req.display_name,
        path: dest,
        repo_id: Some(req.repo_id),
        size,
        languages: req.languages,
        description: req.description,
        installed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    state.models.upsert(local.clone())?;
    Ok(local)
}

async fn download_repo_files(
    app: AppHandle,
    state: State<'_, AppState>,
    req: DownloadRequest,
    client: HubClient,
    mgr: Arc<DownloadManager>,
    select: fn(&[crate::models::HubModelFile]) -> Vec<&crate::models::HubModelFile>,
) -> Result<LocalModel> {
    let files = client.list_files(&req.repo_id).await?;
    let picked = select(&files);
    if picked.is_empty() {
        return Err(AppError::Model(format!(
            "no downloadable files found in {}",
            req.repo_id
        )));
    }

    let safe_repo = req.repo_id.replace('/', "__");
    let backend_dir = state.models.dir_for(req.backend);
    let dir = backend_dir.join(&safe_repo);
    std::fs::create_dir_all(&dir)?;

    // Clean up the legacy `.repo` marker file (if any) from older builds
    // that didn't actually download model files. The new registry entry
    // points at `dir`, so the marker becomes orphaned otherwise.
    let legacy_marker = backend_dir.join(format!("{safe_repo}.repo"));
    if legacy_marker.exists() {
        let _ = std::fs::remove_file(&legacy_marker);
    }

    info!(
        repo = %req.repo_id,
        files = picked.len(),
        backend = ?req.backend,
        "starting repo download"
    );

    let mut total_bytes: u64 = 0;
    for f in &picked {
        let url = client.resolve_url(&req.repo_id, &f.path);
        let dest = dir.join(&f.path);
        let job_id = Uuid::new_v4().to_string();
        let written = mgr
            .clone()
            .download(
                app.clone(),
                client.download_http().clone(),
                client.token().map(|t| t.to_string()),
                job_id,
                req.repo_id.clone(),
                f.path.clone(),
                url,
                dest,
            )
            .await?;
        total_bytes += std::fs::metadata(&written).map(|m| m.len()).unwrap_or(0);
    }

    let local = LocalModel {
        id: format!("{:?}:{}", req.backend, req.repo_id),
        backend: req.backend,
        name: req.display_name,
        path: dir,
        repo_id: Some(req.repo_id),
        size: total_bytes,
        languages: req.languages,
        description: req.description,
        installed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    state.models.upsert(local.clone())?;
    Ok(local)
}

#[tauri::command]
pub fn cancel_download(state: State<'_, AppState>, job_id: String) -> Result<()> {
    state.downloads.cancel(&job_id);
    Ok(())
}

#[tauri::command]
pub fn delete_model(state: State<'_, AppState>, id: String) -> Result<()> {
    state.models.remove(&id)?;
    Ok(())
}

// =============================================================================
// History
// =============================================================================

#[tauri::command]
pub fn list_history(state: State<'_, AppState>, limit: Option<u32>) -> Result<Vec<HistoryEntry>> {
    state.history.list(limit.unwrap_or(100))
}

#[tauri::command]
pub fn clear_history(state: State<'_, AppState>) -> Result<()> {
    state.history.clear()
}

#[tauri::command]
pub fn delete_history_entry(state: State<'_, AppState>, id: i64) -> Result<()> {
    state.history.delete(id)
}

// =============================================================================
// Settings
// =============================================================================

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<Settings> {
    Ok(state.settings.read().await.clone())
}

#[tauri::command]
pub async fn set_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<()> {
    let launch_at_login = settings.launch_at_login;
    {
        let mut s = state.settings.write().await;
        let path = s.path.clone();
        *s = settings;
        s.path = path;
        s.save()?;
    }
    // Keep the OS launch-at-login registration in lockstep with the saved
    // setting — this is what actually makes the toggle DO something.
    sync_autostart(&app, launch_at_login);
    Ok(())
}

// =============================================================================
// Permissions (mostly a macOS concern)
// =============================================================================

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionStatus {
    pub microphone: PermissionState,
    pub accessibility: PermissionState,
    pub input_monitoring: PermissionState,
}

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionState {
    Granted,
    Denied,
    NotDetermined,
    NotApplicable,
}

#[tauri::command]
pub fn check_permissions() -> PermissionStatus {
    // We don't poll macOS TCC here — the OS will show the prompt the first
    // time we touch each subsystem. The frontend uses these stub values to
    // pace the onboarding screens; "not-determined" just means "we'll
    // know after the user tries the feature once".
    #[cfg(target_os = "macos")]
    {
        return PermissionStatus {
            microphone: PermissionState::NotDetermined,
            accessibility: PermissionState::NotDetermined,
            input_monitoring: PermissionState::NotDetermined,
        };
    }
    #[allow(unreachable_code)]
    PermissionStatus {
        microphone: PermissionState::NotApplicable,
        accessibility: PermissionState::NotApplicable,
        input_monitoring: PermissionState::NotApplicable,
    }
}

#[tauri::command]
pub fn open_system_setting(_app: AppHandle, pane: String) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let url = match pane.as_str() {
            "microphone" => "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
            "accessibility" => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            }
            "input-monitoring" => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
            }
            _ => return Err(AppError::Other(format!("unknown pane: {pane}"))),
        };
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map_err(AppError::Io)?;
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        let url = match pane.as_str() {
            "microphone" => "ms-settings:privacy-microphone",
            _ => "ms-settings:privacy",
        };
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(AppError::Io)?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    Err(AppError::Other("not supported on this platform".into()))
}

#[cfg(test)]
mod tests {
    use super::append_chunk_text;

    fn join(parts: &[&str]) -> String {
        let mut s = String::new();
        for p in parts {
            append_chunk_text(&mut s, p);
        }
        s
    }

    #[test]
    fn first_chunk_kept_verbatim() {
        assert_eq!(join(&["Hello world."]), "Hello world.");
    }

    #[test]
    fn strips_chunk_end_period_before_lowercase_word() {
        // Whisper terminates each chunk with a period. When the next
        // chunk starts lowercase that period is spurious.
        assert_eq!(
            join(&["this takes for it.", "to do."]),
            "this takes for it to do."
        );
    }

    #[test]
    fn keeps_period_before_capital_word() {
        // Capital word after period is genuinely ambiguous (could be a
        // real sentence break) — leave it alone.
        assert_eq!(
            join(&["I'm done.", "Let's continue."]),
            "I'm done. Let's continue."
        );
    }

    #[test]
    fn strips_semicolon_or_comma_before_lowercase() {
        assert_eq!(
            join(&["all the transcription;", "ahead of time"]),
            "all the transcription ahead of time"
        );
        assert_eq!(
            join(&["I'm hoping that,", "all of it works"]),
            "I'm hoping that all of it works"
        );
    }

    #[test]
    fn dedupes_exact_boundary_word() {
        // "go go now" → "go now" (Whisper transcribed "go" on both sides
        // of the silence).
        assert_eq!(join(&["I want to go.", "go now"]), "I want to go now");
    }

    #[test]
    fn dedupes_stem_boundary_when_new_is_longer() {
        // "happen" / "happening" — keep the longer form on the new side.
        assert_eq!(
            join(&["multiple chunks happen.", "happening all at once."]),
            "multiple chunks happening all at once."
        );
    }

    #[test]
    fn dedupes_stem_boundary_when_committed_is_longer() {
        // Symmetric case: "happening" / "happen" — keep the longer
        // form on the committed side.
        assert_eq!(
            join(&["it's happening.", "happen now"]),
            "it's happening now"
        );
    }

    #[test]
    fn dedupe_does_not_fire_on_short_overlap() {
        // "do" (2 chars) and "doing" share a prefix but the shorter
        // side is below MIN_STEM (3), so we leave them alone — keeps
        // legitimate "I do" + "doing this" type sequences intact.
        assert_eq!(
            join(&["I do.", "doing this"]),
            // "doing" starts lowercase so the period strips, but the
            // word stays.
            "I do doing this"
        );
        // Unrelated words → no dedup either way. Capital next word
        // preserves the period.
        assert_eq!(
            join(&["The cat sat.", "On the mat."]),
            "The cat sat. On the mat."
        );
    }

    #[test]
    fn empty_chunk_after_dedup_keeps_committed_period() {
        // The new chunk is JUST the duplicated word — drop the dup,
        // and don't touch committed's trailing period (it's the real
        // terminator at that point because we added nothing new).
        assert_eq!(join(&["I want to go.", "go"]), "I want to go.");
    }

    #[test]
    fn proper_noun_inside_chunk_preserved() {
        // We only smooth the BOUNDARY between chunks. Proper nouns
        // emitted by Whisper inside a chunk pass through unchanged.
        // "Tauri" stays capitalized; the period after it strips
        // because the next chunk continues with lowercase "and", which
        // is the right call for "I use Tauri and Rust" being one
        // dictated phrase.
        assert_eq!(
            join(&["I use Tauri.", "and Rust together."]),
            "I use Tauri and Rust together."
        );
    }

    #[test]
    fn function_word_terminal_period_strips_and_lowercases_next() {
        // Real example from the user: "I think you really just." +
        // "Need a good system prompt" was being joined as
        // "I think you really just. Need a good system prompt"
        // because "Need" is capitalized — but "just" almost never
        // ends a real sentence, so the period is a Whisper artifact.
        assert_eq!(
            join(&["I think you really just.", "Need a good system prompt"]),
            "I think you really just need a good system prompt"
        );
    }

    #[test]
    fn function_word_because_strips_capital_continuation() {
        // "it's because They send so much" — "because" links clauses,
        // it never ends a sentence, so "They" is a chunk-start
        // capitalization artifact.
        assert_eq!(
            join(&["it's because.", "They send so much"]),
            "it's because they send so much"
        );
    }

    #[test]
    fn discourse_marker_like_before_capital_continuation() {
        // "Like, You use your usage" — "Like" used as a discourse
        // marker followed by a comma, then capitalized continuation.
        // The comma already triggers `trailing_off`; the continuation
        // signal then lowercases "You".
        assert_eq!(
            join(&["Like,", "You use your usage"]),
            "Like you use your usage"
        );
    }

    #[test]
    fn ellipsis_at_end_treated_as_continuation() {
        // Whisper emits "..." at chunks that end mid-thought.
        // "system prompts to..." + "Define basically" → join as
        // "system prompts to define basically", lowercasing the
        // capital that Whisper put on the new chunk's first word.
        assert_eq!(
            join(&["system prompts to...", "Define basically"]),
            "system prompts to define basically"
        );
    }

    #[test]
    fn unicode_ellipsis_at_end_treated_as_continuation() {
        // Same but with the single-codepoint ellipsis '\u{2026}'.
        assert_eq!(
            join(&["system prompts to\u{2026}", "Define basically"]),
            "system prompts to define basically"
        );
    }

    #[test]
    fn leading_ellipsis_on_new_chunk_stripped() {
        // The new chunk's audio started mid-thought and Whisper
        // prefixed it with "..." or "…" — drop those so the join
        // reads cleanly.
        assert_eq!(
            join(&["I want", "...to do this."]),
            "I want to do this."
        );
        assert_eq!(
            join(&["I want", "\u{2026}to do this."]),
            "I want to do this."
        );
    }

    #[test]
    fn pronoun_i_stays_capitalized_when_lowercasing_continuation() {
        // The continuation lowercase rule must NOT touch the pronoun
        // "I" — "to" is a function word so we'd otherwise lowercase
        // "I'm" to "i'm", which is wrong.
        assert_eq!(
            join(&["I want to.", "I'm going home"]),
            "I want to I'm going home"
        );
    }

    #[test]
    fn acronym_stays_capitalized_when_lowercasing_continuation() {
        // ALL-CAPS short tokens (AI / LLM / API) are almost certainly
        // acronyms — keep their casing even at chunk boundaries.
        assert_eq!(
            join(&["this works with.", "AI in check"]),
            "this works with AI in check"
        );
    }

    #[test]
    fn non_function_word_keeps_capital_break() {
        // Sentence-final word that's NOT a function word and a
        // capitalized next chunk — leave the break alone, this is
        // the genuinely-ambiguous case where wrongly merging is
        // worse than wrongly splitting.
        assert_eq!(
            join(&["their harness.", "So it keeps"]),
            "their harness. So it keeps"
        );
    }

    use super::{find_silence_cut, trim_silence_padding};

    fn loud_block(n: usize) -> Vec<f32> {
        // 0.3 amplitude — well above the 0.02 silence threshold.
        vec![0.3; n]
    }

    fn silent_block(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn trim_silence_returns_none_for_pure_silence() {
        let buf = silent_block(16_000); // 1 s of silence at 16 kHz
        assert_eq!(trim_silence_padding(&buf, 16_000), None);
    }

    #[test]
    fn trim_silence_finds_speech_in_middle() {
        // 200 ms silence, 600 ms speech, 200 ms silence.
        let mut buf = silent_block(3_200);
        buf.extend(loud_block(9_600));
        buf.extend(silent_block(3_200));

        let (s, e) = trim_silence_padding(&buf, 16_000).expect("speech detected");

        // Speech region is samples [3200, 12800). With 80 ms (1280 sample)
        // pad on each side and 10 ms hop quantisation, start/end should
        // be within a hop-width of 3200-1280 / 12800+1280.
        let hop = 160; // 10 ms at 16 kHz
        assert!((s as i64 - (3_200 - 1_280)) .abs() <= hop as i64, "start={s}");
        assert!((e as i64 - (12_800 + 1_280)).abs() <= hop as i64, "end={e}");
    }

    #[test]
    fn find_silence_cut_returns_none_without_real_silence() {
        // 4 s of continuous loud audio — fluent speech with no pauses.
        // We must NOT cut here; releasing must fall through to the
        // tail-only baseline pass.
        let buf = loud_block(16_000 * 4);
        assert_eq!(find_silence_cut(&buf, 16_000), None);
    }

    #[test]
    fn find_silence_cut_finds_pause_between_phrases() {
        // 1 s loud, 200 ms silence (≥ 120 ms minimum), 1 s loud.
        let mut buf = loud_block(16_000);
        buf.extend(silent_block(3_200));
        buf.extend(loud_block(16_000));

        let cut = find_silence_cut(&buf, 16_000).expect("cut found");
        // Cut should land inside the silence run [16000, 19200).
        assert!(cut >= 16_000 && cut < 19_200, "cut={cut}");
    }

    use super::splice_new_words;
    use crate::backends::WordTimestamp;

    fn w(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn splice_keeps_only_words_after_committed_boundary() {
        // Window had 3 s of lookback (already-committed audio) +
        // some new audio. The first three words land in the
        // lookback region; the rest are new.
        let words = vec![
            w(" the", 0.10, 0.30),
            w(" last", 0.30, 0.55),
            w(" word.", 0.55, 0.90),
            w(" That's", 3.05, 3.35),
            w(" new.", 3.35, 3.70),
        ];
        let spliced = splice_new_words(&words, 3.0);
        assert_eq!(spliced, "That's new.");
    }

    #[test]
    fn splice_includes_word_within_tolerance_before_boundary() {
        // Whisper timestamps drift up to ~30 ms; we tolerate 60 ms.
        // A word starting at 2.95 s when the boundary is 3.00 s is
        // unambiguously "new" audio, just timestamp jitter — keep it.
        let words = vec![w(" new", 2.95, 3.20), w(" word", 3.21, 3.40)];
        let spliced = splice_new_words(&words, 3.0);
        assert_eq!(spliced, "new word");
    }

    #[test]
    fn splice_drops_word_well_before_boundary() {
        // 200 ms before boundary is outside tolerance — drop.
        let words = vec![w(" old", 2.80, 3.00), w(" new", 3.10, 3.30)];
        let spliced = splice_new_words(&words, 3.0);
        assert_eq!(spliced, "new");
    }

    #[test]
    fn splice_with_zero_boundary_keeps_everything() {
        // First window has no lookback (committed = 0). All words
        // are new.
        let words = vec![w(" hello", 0.10, 0.40), w(" world", 0.40, 0.90)];
        let spliced = splice_new_words(&words, 0.0);
        assert_eq!(spliced, "hello world");
    }

    #[test]
    fn splice_with_empty_words_returns_empty() {
        assert_eq!(splice_new_words(&[], 0.0), "");
        assert_eq!(splice_new_words(&[], 3.0), "");
    }

    #[test]
    fn splice_drops_all_when_no_words_after_boundary() {
        // The committed boundary is past the end of all words —
        // backend transcribed nothing in the new region (e.g. all
        // silence after committed). Returns empty.
        let words = vec![w(" old", 0.50, 0.90), w(" stuff", 0.90, 1.40)];
        let spliced = splice_new_words(&words, 3.0);
        assert_eq!(spliced, "");
    }

    #[test]
    fn splice_handles_words_without_leading_space() {
        // whisper.cpp / token-grouped backends may emit the very
        // first word without a leading space. The output should
        // still trim cleanly.
        let words = vec![w("Hello", 0.10, 0.40), w(" world", 0.40, 0.90)];
        let spliced = splice_new_words(&words, 0.0);
        assert_eq!(spliced, "Hello world");
    }

    #[test]
    fn find_silence_cut_ignores_too_short_silence() {
        // 1 s loud, 50 ms silence (under 120 ms threshold), 1 s loud.
        // That's a normal inter-word gap, not a phrase break — we
        // shouldn't cut there.
        let mut buf = loud_block(16_000);
        buf.extend(silent_block(800)); // 50 ms
        buf.extend(loud_block(16_000));

        assert_eq!(find_silence_cut(&buf, 16_000), None);
    }
}
