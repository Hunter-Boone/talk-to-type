//! Auto-paste: write text to the system clipboard, then synthesize the
//! platform's paste keystroke.
//!
//! `enigo` simulates real keyboard events so the paste happens in whatever
//! window currently has focus — which is exactly what we want, since the
//! user pressed our hotkey while their target app was focused, and our
//! overlay window is non-focusable.
//!
//! On macOS the user must grant Accessibility permission. On Windows no
//! special permission is required.

use std::time::Duration;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;
use tracing::{debug, info};

use crate::error::{AppError, Result};

pub fn copy_to_clipboard(app: &AppHandle, text: &str) -> Result<()> {
    app.clipboard()
        .write_text(text.to_string())
        .map_err(|e| AppError::Paste(format!("clipboard write: {e}")))?;
    Ok(())
}

/// Synthesise Cmd+V (mac) / Ctrl+V (Win). Caller is responsible for having
/// already written the desired text to the clipboard.
pub fn synthesize_paste() -> Result<()> {
    // Small delay so the foreground app sees focus return to it before the
    // keystroke arrives. ~30 ms is reliable across both platforms.
    std::thread::sleep(Duration::from_millis(30));

    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| AppError::Paste(format!("enigo init: {e}")))?;

    #[cfg(target_os = "macos")]
    let mod_key = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let mod_key = Key::Control;

    enigo
        .key(mod_key, Direction::Press)
        .map_err(|e| AppError::Paste(format!("press mod: {e}")))?;
    enigo
        .key(Key::Unicode('v'), Direction::Click)
        .map_err(|e| AppError::Paste(format!("v: {e}")))?;
    enigo
        .key(mod_key, Direction::Release)
        .map_err(|e| AppError::Paste(format!("release mod: {e}")))?;

    debug!("paste keystroke sent");
    Ok(())
}

pub fn copy_and_paste(app: &AppHandle, text: &str) -> Result<()> {
    copy_to_clipboard(app, text)?;
    synthesize_paste()?;
    info!(chars = text.chars().count(), "auto-pasted");
    Ok(())
}
