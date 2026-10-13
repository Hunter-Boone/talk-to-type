//! Always-on-top dictation overlay.
//!
//! The overlay is a small undecorated webview window declared in
//! `tauri.conf.json` (label: "overlay"). On every show we move it to the
//! bottom-centre of whichever monitor the mouse is on, mark it
//! click-through, and emit `overlay://state` events to the JS side so the
//! waveform/VU-meter component animates.

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, WebviewWindow};
use tracing::warn;

/// Gap between the overlay and the bottom of the monitor's work area
/// (i.e. the top of the taskbar), in logical pixels.
const BOTTOM_MARGIN: f64 = 48.0;

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
        place_on_cursor_monitor(app, &w);
        // Click-through so the overlay never steals input from the focused app.
        let _ = w.set_ignore_cursor_events(true);
        let _ = w.set_always_on_top(true);
        if let Err(e) = w.show() {
            warn!("overlay show: {e}");
        }
    }
}

/// Bottom-centre of the monitor under the mouse, just above the taskbar.
/// The window's config position is a fixed (0, 0) — the top-left corner of
/// the primary display — which on a multi-monitor desk is usually not the
/// screen the user is looking at, and even there it sits on top of the
/// focused app's title bar.
fn place_on_cursor_monitor(app: &AppHandle, w: &WebviewWindow) {
    let monitor = app
        .cursor_position()
        .ok()
        .and_then(|c| app.monitor_from_point(c.x, c.y).ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else {
        warn!("overlay: no monitor found; leaving position unchanged");
        return;
    };
    let (Ok(size), Ok(current_scale)) = (w.outer_size(), w.scale_factor()) else {
        return;
    };
    // outer_size is in the *current* monitor's pixels; rescale to the
    // target monitor so centring is right across mixed-DPI displays.
    let ratio = monitor.scale_factor() / current_scale;
    let width = size.width as f64 * ratio;
    let height = size.height as f64 * ratio;
    let area = monitor.work_area();
    let x = area.position.x as f64 + (area.size.width as f64 - width) / 2.0;
    let y = area.position.y as f64 + area.size.height as f64
        - height
        - BOTTOM_MARGIN * monitor.scale_factor();
    if let Err(e) = w.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32)) {
        warn!("overlay set_position: {e}");
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
