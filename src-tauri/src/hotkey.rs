//! Push-to-talk global hotkey wiring.
//!
//! `tauri-plugin-global-shortcut` v2 exposes both `Pressed` and `Released`
//! events on supported platforms (Windows + macOS). We register a single
//! shortcut and dispatch press/release to start/stop the recording task.
//!
//! KNOWN LIMITATIONS:
//! - Some single-key shortcuts (bare `Right Alt`, `Fn`) aren't expressible
//!   in the cross-platform API. We require at least one modifier to keep
//!   the experience consistent.
//! - Some Windows IME stacks suppress `Pressed` events for shortcuts that
//!   collide with system shortcuts (e.g., Ctrl+Space if an IME is active).
//!   We surface a warning to the user in that case.

use std::sync::Arc;

use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{
    GlobalShortcutExt, Shortcut, ShortcutEvent, ShortcutState,
};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::audio::AudioController;
use crate::backends::TranscriptionBackend;
use crate::db::HistoryDb;
use crate::error::{AppError, Result};
use crate::models::ModelRegistry;
use crate::settings::Settings;
use crate::sidecar::SidecarSupervisor;
use crate::state::{AppState, PreTranscribeState};

pub fn parse_shortcut(s: &str) -> Result<Shortcut> {
    s.parse::<Shortcut>()
        .map_err(|e| AppError::Hotkey(format!("invalid hotkey '{s}': {e}")))
}

/// Register both the dictate (press-and-hold) shortcut and an optional
/// repaste (press-once) shortcut. Always clears existing registrations
/// first so calling this is idempotent.
pub fn register(
    app: &AppHandle,
    dictate: Shortcut,
    repaste: Option<Shortcut>,
) -> Result<()> {
    let gsc = app.global_shortcut();

    // Always clear before registering so re-registration works cleanly.
    let _ = gsc.unregister_all();

    let app_for_dictate = app.clone();
    gsc.on_shortcut(dictate.clone(), move |_app, _sc, event: ShortcutEvent| {
        let state = match app_for_dictate.try_state::<AppState>() {
            Some(s) => s,
            None => {
                warn!("AppState not yet managed; dropping hotkey event");
                return;
            }
        };

        match event.state() {
            ShortcutState::Pressed => {
                debug!("hotkey pressed → start");
                spawn_press(state.inner().clone_arcs(), app_for_dictate.clone());
            }
            ShortcutState::Released => {
                debug!("hotkey released → stop");
                spawn_release(state.inner().clone_arcs(), app_for_dictate.clone());
            }
        }
    })
    .map_err(|e| AppError::Hotkey(format!("on_shortcut dictate: {e}")))?;
    info!("dictate hotkey registered: {dictate:?}");

    if let Some(rp) = repaste {
        let app_for_repaste = app.clone();
        gsc.on_shortcut(rp.clone(), move |_app, _sc, event: ShortcutEvent| {
            // Only fire on Released — by then the user has lifted their
            // modifier keys, so the synthesized Ctrl/Cmd+V from `enigo`
            // doesn't collide with the hotkey's own modifiers.
            if !matches!(event.state(), ShortcutState::Released) {
                return;
            }
            let state = match app_for_repaste.try_state::<AppState>() {
                Some(s) => s,
                None => {
                    warn!("AppState not yet managed; dropping repaste event");
                    return;
                }
            };
            debug!("repaste hotkey fired");
            spawn_repaste(state.inner().clone_arcs(), app_for_repaste.clone());
        })
        .map_err(|e| AppError::Hotkey(format!("on_shortcut repaste: {e}")))?;
        info!("repaste hotkey registered: {rp:?}");
    }

    Ok(())
}

pub fn unregister_all(app: &AppHandle) -> Result<()> {
    app.global_shortcut()
        .unregister_all()
        .map_err(|e| AppError::Hotkey(e.to_string()))
}

/// Spawn the press handler. Kept here (not in commands.rs) so the hotkey
/// driver and the manual "start_recording" command share the same logic.
///
/// IMPORTANT: this runs from the OS message-pump thread (Win32 wndproc on
/// Windows, the main thread on macOS) — no Tokio reactor is attached, so
/// `tokio::spawn` panics. Use `tauri::async_runtime::spawn`, which posts
/// onto Tauri's global runtime and is safe to call from anywhere.
fn spawn_press(state: ArcsBundle, app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::commands::press_to_talk(&state, &app).await {
            warn!(error = %e, "press_to_talk failed");
        }
    });
}

fn spawn_release(state: ArcsBundle, app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::commands::release_to_talk(&state, &app).await {
            warn!(error = %e, "release_to_talk failed");
        }
    });
}

fn spawn_repaste(state: ArcsBundle, app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::commands::repaste_last_impl(&state, &app).await {
            warn!(error = %e, "repaste_last failed");
        }
    });
}

/// A cheap bundle of `Arc`-cloned references — saves typing them out at
/// every spawn site and keeps the borrow pattern obvious.
#[derive(Clone)]
pub struct ArcsBundle {
    pub audio: Arc<AudioController>,
    pub backend: Arc<RwLock<Option<Box<dyn TranscriptionBackend>>>>,
    /// Which backend kind is currently instantiated in `backend`. Needed by
    /// the self-healing `ensure_backend_ready` path that runs from the
    /// hotkey driver (no `tauri::State` available there).
    pub backend_kind: Arc<parking_lot::Mutex<Option<crate::backends::BackendKind>>>,
    pub settings: Arc<RwLock<Settings>>,
    pub history: Arc<HistoryDb>,
    /// Needed by the post-processing path so the Transformers cleanup
    /// provider can issue load_model + cleanup RPCs without going back
    /// through `tauri::State<AppState>`.
    pub sidecar: Arc<SidecarSupervisor>,
    /// Needed by the Transformers cleanup provider to resolve the
    /// configured model id to its on-disk path.
    pub models: Arc<ModelRegistry>,
    pub pretranscribe_state: Arc<parking_lot::Mutex<PreTranscribeState>>,
    pub pretranscribe_task: Arc<parking_lot::Mutex<Option<JoinHandle<()>>>>,
}

impl AppState {
    pub fn clone_arcs(&self) -> ArcsBundle {
        ArcsBundle {
            audio: self.audio.clone(),
            backend: self.backend.clone(),
            backend_kind: self.backend_kind.clone(),
            settings: self.settings.clone(),
            history: self.history.clone(),
            sidecar: self.sidecar.clone(),
            models: self.models.clone(),
            pretranscribe_state: self.pretranscribe_state.clone(),
            pretranscribe_task: self.pretranscribe_task.clone(),
        }
    }
}
