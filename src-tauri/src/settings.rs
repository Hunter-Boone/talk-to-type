//! User settings — single JSON blob under <app_data>/settings.json.
//!
//! We use a hand-written serde struct instead of tauri-plugin-store so the
//! Rust side can read settings directly without a JS round-trip. The
//! frontend gets the same struct via the `get_settings` command.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backends::{BackendKind, BackendOptions};
use crate::error::{AppError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub launch_at_login: bool,
    pub show_in_tray: bool,
    pub hotkey: String,
    /// Optional second global shortcut that re-pastes the most recent
    /// transcription. Empty/None means disabled.
    pub repaste_hotkey: Option<String>,
    pub post_action: PostAction,

    pub input_device: Option<String>,
    pub vad_enabled: bool,
    /// Stream transcription while the user is still holding the hotkey.
    ///
    /// Uses a sliding-window pipeline: every few seconds the host fires
    /// inference on the latest audio with a few seconds of LOOKBACK
    /// from before the previously-committed boundary, asks the backend
    /// for word-level timestamps, and splices results at exact word
    /// boundaries. Each region of audio gets transcribed at least
    /// twice — once at the END of one window (with only left context)
    /// and once at the START of the next (with full context on both
    /// sides) — and the higher-context pass wins via timestamp-based
    /// merging.
    ///
    /// Net effect: release-to-paste latency drops dramatically on
    /// multi-second utterances (only the trailing window still needs
    /// inference) AND each word benefits from full-context Whisper
    /// inference, rather than the small-window quality hit chunked
    /// streaming used to suffer.
    ///
    /// Backends that can't emit word timestamps for the active model
    /// (currently: Transformers with non-Whisper checkpoints) fall
    /// back to no-lookback chunked mode automatically.
    ///
    /// Default true; toggle off if you'd rather pay full single-pass
    /// inference cost on every release for absolute parity with
    /// non-streaming transcription.
    pub pretranscribe: bool,

    pub active_backend: BackendKind,
    pub active_model_id: Option<String>,
    pub backend_options: BackendOptions,

    pub post_processing: PostProcessing,

    pub max_history: u32,

    /// HuggingFace user access token. Optional — required for downloading
    /// gated or private repos (`huggingface.co` will return 401/403 without).
    #[serde(default)]
    pub hf_token: Option<String>,

    #[serde(skip)]
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PostAction {
    #[default]
    Paste,
    ClipboardOnly,
}

/// Where the cleanup pass runs.
///
/// `CursorCli` shells out to the `cursor-agent` binary (the local Cursor CLI)
/// in non-interactive print mode and uses the user's signed-in Cursor account.
/// `Ollama` hits a local Ollama HTTP server. `Transformers` runs a HuggingFace
/// causal-LM in the same Python sidecar that hosts the ASR engines. All three
/// produce the same shape of result — a single cleaned string — so the rest
/// of the pipeline doesn't care.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PostProcessingProvider {
    /// Local LLM via Ollama HTTP API.
    Ollama,
    /// Cursor's CLI (`cursor-agent -p`) — Composer / Auto / etc.
    #[default]
    CursorCli,
    /// HuggingFace causal-LM run in the Python sidecar's cleanup slot.
    Transformers,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PostProcessing {
    pub enabled: bool,
    /// Which engine performs the cleanup pass.
    pub provider: PostProcessingProvider,

    // --- Ollama-specific ---------------------------------------------------
    pub ollama_url: String,
    /// Ollama model tag, e.g. "llama3.2:3b".
    pub model: String,

    // --- Cursor CLI specific ----------------------------------------------
    /// Cursor model name passed via `--model` (e.g. "auto", "composer-1").
    pub cursor_model: String,
    /// Optional override for the `cursor-agent` binary location. Empty/None
    /// means "look on PATH". Useful on Windows where the installer puts it
    /// under `%LOCALAPPDATA%\Programs\cursor-agent` and PATH may not be set
    /// for the spawned process.
    pub cursor_cli_path: Option<String>,

    // --- Transformers-specific --------------------------------------------
    /// Local-model registry id (a `cleanup-transformers` row) of the
    /// downloaded LLM to use. None until the user picks one in the UI.
    pub transformers_model_id: Option<String>,
    /// torch device hint passed to `model.to(...)` ("cpu", "cuda", "mps").
    /// Falls back to CPU on the sidecar side if the requested device
    /// isn't available, with a log line explaining why.
    pub transformers_device: String,
    /// Maximum new tokens generated per cleanup pass. Caps runaway
    /// hallucinations and bounds latency on slow CPUs.
    pub transformers_max_tokens: u32,

    // --- Shared -----------------------------------------------------------
    pub system_prompt: String,
}

impl Default for PostProcessing {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: PostProcessingProvider::default(),
            ollama_url: "http://localhost:11434".into(),
            model: "llama3.2:3b".into(),
            cursor_model: "auto".into(),
            cursor_cli_path: None,
            transformers_model_id: None,
            transformers_device: "cpu".into(),
            transformers_max_tokens: 512,
            system_prompt: default_post_processing_system_prompt(),
        }
    }
}

fn default_post_processing_system_prompt() -> String {
    "You clean up dictated text that was transcribed in streaming chunks, so it may contain artifacts: spurious mid-sentence periods at chunk boundaries, unnecessary capitalization at chunk starts, ellipses ('...') where a thought continues, and false sentence breaks. Fix punctuation and capitalization to read naturally. Remove obvious filler words ('um', 'uh'). DO NOT change the actual words spoken — preserve all content verbatim except for filler. DO NOT add new content. DO NOT correct what you think might be misheard words; output exactly the words given. Output only the cleaned text, no preamble.".into()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            launch_at_login: false,
            show_in_tray: true,
            hotkey: default_hotkey_string(),
            repaste_hotkey: Some(default_repaste_hotkey_string()),
            post_action: PostAction::Paste,

            input_device: None,
            vad_enabled: true,
            pretranscribe: true,

            active_backend: BackendKind::WhisperCpp,
            active_model_id: None,
            backend_options: BackendOptions::default(),

            post_processing: PostProcessing::default(),

            max_history: 100,

            hf_token: None,
            path: PathBuf::new(),
        }
    }
}

fn default_hotkey_string() -> String {
    if cfg!(target_os = "macos") {
        "Cmd+Shift+Space".into()
    } else {
        "Ctrl+Shift+Space".into()
    }
}

fn default_repaste_hotkey_string() -> String {
    if cfg!(target_os = "macos") {
        "Cmd+Alt+V".into()
    } else {
        "Ctrl+Alt+V".into()
    }
}

impl Settings {
    pub fn load(app_data_dir: &Path) -> Result<Self> {
        let path = app_data_dir.join("settings.json");
        if !path.exists() {
            let mut s = Settings::default();
            s.path = path.clone();
            s.save()?;
            return Ok(s);
        }
        let text = fs::read_to_string(&path)?;
        match serde_json::from_str::<Settings>(&text) {
            Ok(mut s) => {
                s.path = path;
                Ok(s)
            }
            // A corrupt settings file (crash mid-write, disk hiccup, manual
            // edit gone wrong) must not brick the whole app at startup.
            // Preserve the broken file for inspection and continue with
            // defaults.
            Err(e) => {
                let backup = app_data_dir.join("settings.json.corrupt");
                let _ = fs::copy(&path, &backup);
                tracing::error!(
                    error = %e,
                    backup = %backup.display(),
                    "settings.json unreadable; backed it up and reset to defaults"
                );
                let mut s = Settings::default();
                s.path = path;
                s.save()?;
                Ok(s)
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        if self.path.as_os_str().is_empty() {
            return Err(AppError::Settings("settings path not set".into()));
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        // Write-then-rename so a crash mid-write can never leave a
        // half-written settings.json behind.
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}
