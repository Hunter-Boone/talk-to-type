//! Always-on-top dictation overlay.
//!
//! The overlay is a small undecorated webview window declared in
//! `tauri.conf.json` (label: "overlay"). We position it once near the
//! current cursor, mark it click-through, and emit `overlay://state`
//! events to the JS side so the waveform/VU-meter component animates.

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tracing::warn;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OverlayState {
    Hidden,
    Listening,
    Transcribing,
    Pasted,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverlayPayload {
    pub state: OverlayState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak: Option<f32>,
}

pub fn show(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("overlay") {
        // Click-through so the overlay never steals input from the focused app.
        let _ = w.set_ignore_cursor_events(true);
        let _ = w.set_always_on_top(true);
        if let Err(e) = w.show() {
            warn!("overlay show: {e}");
        }
    }
}

pub fn hide(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("overlay") {
        if let Err(e) = w.hide() {
            warn!("overlay hide: {e}");
        }
    }
}

pub fn emit(app: &AppHandle, payload: OverlayPayload) {
    if let Err(e) = app.emit("overlay://state", payload) {
        warn!("emit overlay state: {e}");
    }
}
