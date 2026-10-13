//! System tray icon — installed once at app startup so closing or
//! minimising the main window can hide it instead of exiting.
//!
//! Click semantics:
//!   - Left click on the icon       → toggle main window visibility
//!   - Tray menu "Show talk-to-type" → show + focus + unminimise
//!   - Tray menu "Quit"             → real exit
//!
//! Window-event hookup (in `lib.rs`):
//!   - `WindowEvent::CloseRequested` → `event.api().prevent_close()` + hide
//!   - `WindowEvent::Resized(0, 0)`  → hide (Windows fires this on minimise)

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager,
};
use tracing::warn;

const TRAY_ID: &str = "main";

/// Build the tray icon and install it on the running `AppHandle`.
///
/// Idempotent-by-construction: the function should only be called once
/// from `Builder::setup`. A second call would error out of
/// `TrayIconBuilder::build` because Tauri only allows a single icon
/// per `id`.
pub fn install(app: &AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "tray-show", "Show talk-to-type", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "tray-quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    let icon = app
        .default_window_icon()
        .ok_or_else(|| tauri::Error::Anyhow(anyhow::anyhow!(
            "no default window icon — tauri.conf.json must declare bundle.icon"
        )))?
        .clone();

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .tooltip("talk-to-type")
        .menu(&menu)
        // `false` so a left click reaches our `on_tray_icon_event`
        // handler instead of being eaten by the menu popup. The
        // menu still opens on right click.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "tray-show" => show_main(app),
            "tray-quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Left click anywhere on the icon → bring the window
            // back. Touch events on Windows also surface as a
            // left click here.
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

/// Bring the main window back from a hidden / minimised state and
/// give it focus. Used by the tray menu's Show item, the
/// left-click-on-icon handler, and a second launch of the app.
pub(crate) fn show_main(app: &AppHandle) {
    let Some(window) = app.get_webview_window("main") else {
        warn!("tray: main window not found");
        return;
    };
    if let Err(e) = window.unminimize() {
        warn!(error = %e, "tray: unminimize failed");
    }
    if let Err(e) = window.show() {
        warn!(error = %e, "tray: show failed");
    }
    if let Err(e) = window.set_focus() {
        warn!(error = %e, "tray: set_focus failed");
    }
}
