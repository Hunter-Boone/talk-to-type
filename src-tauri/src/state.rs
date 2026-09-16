use std::sync::Arc;

use parking_lot::Mutex;
use tauri::{AppHandle, Manager};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::audio::{AudioController, TARGET_SAMPLE_RATE};
use crate::backends::{BackendKind, TranscriptionBackend};
use crate::db::HistoryDb;
use crate::error::Result;
use crate::models::{DownloadManager, ModelRegistry};
use crate::settings::Settings;
use crate::sidecar::SidecarSupervisor;

/// Streaming pre-transcription state.
///
/// While the user is holding the hotkey, a background task slices off
/// "committed" prefixes of the captured audio at silence boundaries
/// (or forced cuts on long, fluent speech) and transcribes them
/// incrementally. `committed_samples` is the offset (in input-rate
/// samples) up through which `text` is final — `release_to_talk` only
/// needs to transcribe `samples[committed_samples..]` to produce the
/// full utterance, which is usually <2 s of audio instead of the
/// entire 5–15 s buffer.
#[derive(Debug, Clone)]
pub struct PreTranscribeState {
    pub committed_samples: usize,
    pub text: String,
    pub sample_rate: u32,
}

impl Default for PreTranscribeState {
    fn default() -> Self {
        Self {
            committed_samples: 0,
            text: String::new(),
            sample_rate: TARGET_SAMPLE_RATE,
        }
    }
}

/// Shared application state managed by Tauri (`app.manage(...)`).
///
/// All long-lived runtime objects live here so commands can pull them out
/// with `state: tauri::State<AppState>`.
pub struct AppState {
    pub app: AppHandle,
    pub settings: Arc<RwLock<Settings>>,
    pub audio: Arc<AudioController>,
    pub backend: Arc<RwLock<Option<Box<dyn TranscriptionBackend>>>>,
    pub backend_kind: Arc<Mutex<Option<BackendKind>>>,
    pub sidecar: Arc<SidecarSupervisor>,
    pub models: Arc<ModelRegistry>,
    pub history: Arc<HistoryDb>,
    /// Long-lived download manager so `cancel_download` can reach the
    /// cancellation flag of an in-flight job.
    pub downloads: Arc<DownloadManager>,
    /// Streaming pre-transcription state — committed prefix + offset.
    /// The pretranscribe task pushes commits into here; `release_to_talk`
    /// reads it once the task has exited and uses it to skip the bulk
    /// of the final inference pass.
    pub pretranscribe_state: Arc<Mutex<PreTranscribeState>>,
    /// Handle to the live pretranscribe task (one per recording session).
    /// `release_to_talk` takes this and awaits it so any in-flight chunk
    /// settles into `pretranscribe_state` before we look at it.
    pub pretranscribe_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl AppState {
    pub fn new(app: AppHandle) -> Result<Self> {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|e| crate::error::AppError::Other(format!("app_data_dir: {e}")))?;
        std::fs::create_dir_all(&app_data_dir)?;

        let settings = Settings::load(&app_data_dir)?;
        let history = HistoryDb::open(&app_data_dir.join("history.db"))?;
        let models = ModelRegistry::new(app_data_dir.join("models"))?;
        let audio = AudioController::new();
        let sidecar = SidecarSupervisor::new();

        Ok(Self {
            app,
            settings: Arc::new(RwLock::new(settings)),
            audio: Arc::new(audio),
            backend: Arc::new(RwLock::new(None)),
            backend_kind: Arc::new(Mutex::new(None)),
            sidecar: Arc::new(sidecar),
            models: Arc::new(models),
            history: Arc::new(history),
            downloads: Arc::new(DownloadManager::new()),
            pretranscribe_state: Arc::new(Mutex::new(PreTranscribeState::default())),
            pretranscribe_task: Arc::new(Mutex::new(None)),
        })
    }
}
