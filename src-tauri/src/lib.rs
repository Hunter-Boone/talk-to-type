//! talk-to-type — Tauri host crate.
//!
//! Module map:
//!   audio/     — cpal capture, ring buffer, optional Silero VAD
//!   backends/  — TranscriptionBackend trait + three implementations
//!   sidecar/   — Python sidecar supervisor + JSON-RPC framing
//!   models/    — Hugging Face Hub client + download manager + local registry
//!   hotkey     — push-to-talk global hotkey wiring
//!   paste      — clipboard write + simulated paste keystroke
//!   overlay    — show/hide/position the always-on-top status overlay
//!   db         — history storage (rusqlite)
//!   settings   — typed wrapper around tauri-plugin-store
//!   commands   — #[tauri::command] surface exposed to the React frontend
//!   error      — AppError + Result alias used everywhere

pub mod audio;
pub mod backends;
pub mod commands;
pub mod db;
pub mod error;
pub mod hotkey;
pub mod models;
pub mod overlay;
pub mod paste;
pub mod settings;
pub mod sidecar;
pub mod state;
pub mod tray;

use tauri::{Manager, RunEvent, WindowEvent};
use tracing::info;

/// Entry point invoked from `main.rs`. Builds the Tauri app, wires plugins,
/// initialises shared state, registers commands, and runs the event loop.
pub fn run() {
    init_tracing();
    info!("starting talk-to-type v{}", env!("CARGO_PKG_VERSION"));

    tauri::Builder::default()
        // Must be the first plugin. A second copy (Start menu while the
        // login copy sits in the tray) would otherwise fight over the
        // global hotkey: whichever registered second fails, and once the
        // first exits nobody owns it and dictation silently stops.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            info!("second launch; focusing the running instance");
            tray::show_main(app);
        }))
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            // Autostarted instances open straight to the tray — nobody wants
            // the settings window in their face at every login.
            Some(vec!["--hidden"]),
        ))
        .setup(|app| {
            // App-wide async runtime + shared state. Anything that needs
            // a long-lived task (audio thread, sidecar supervisor) goes here.
            let handle = app.handle().clone();
            let app_state = state::AppState::new(handle.clone())?;
            let (show_in_tray, launch_at_login) = {
                let s = tauri::async_runtime::block_on(app_state.settings.read());
                (s.show_in_tray, s.launch_at_login)
            };
            app.manage(app_state);

            // Launched by the OS at login (or with --hidden): start in the
            // tray instead of popping the settings window.
            if std::env::args().any(|a| a == "--hidden") {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }

            // Tray icon — when enabled, close/minimise hide to tray and the
            // menu's Quit item is what actually ends the process. When the
            // user turned the tray off, closing the window quits instead
            // (see on_window_event below) so the app can't become
            // unreachable.
            if show_in_tray {
                if let Err(e) = tray::install(&handle) {
                    tracing::warn!(error = %e, "tray icon setup failed");
                }
            }

            // Everything below runs off the main thread so a slow model
            // load can't block window creation.
            let handle_for_init = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state: tauri::State<state::AppState> = handle_for_init.state();

                // Register global hotkeys from saved settings NOW — not
                // when the frontend happens to mount. Dictation must work
                // even if the window stays hidden forever.
                if let Err(e) = commands::sync_hotkeys(&handle_for_init, &state).await {
                    tracing::warn!(error = %e, "startup hotkey registration failed");
                    // Without its hotkey the app is inert, and the log is
                    // the last place anyone looks — tell the user directly.
                    use tauri_plugin_notification::NotificationExt;
                    let _ = handle_for_init
                        .notification()
                        .builder()
                        .title("talk-to-type: dictation hotkey not registered")
                        .body(format!(
                            "{e}\nAnother app may own it. Choose a different hotkey in Settings → General."
                        ))
                        .show();
                }

                // Make the OS launch-at-login registration match the saved
                // setting (repairs installs from before this was wired up).
                commands::sync_autostart(&handle_for_init, launch_at_login);

                // Bring up the saved backend + model.
                if let Err(e) = commands::ensure_backend_ready(&state).await {
                    tracing::warn!(error = %e, "initial backend bring-up failed");
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Only manage the main window — the overlay is decoration-
            // less and click-through, never receives close events
            // worth intercepting.
            if window.label() != "main" {
                return;
            }
            // Hide-to-tray only makes sense when a tray icon exists to
            // come back through. With the tray disabled in settings, the
            // close button quits and minimise stays a normal minimise.
            let hide_to_tray = window
                .app_handle()
                .try_state::<state::AppState>()
                .map(|s| {
                    s.settings
                        .try_read()
                        .map(|g| g.show_in_tray)
                        .unwrap_or(true)
                })
                .unwrap_or(true);
            match event {
                // Close button (×) → hide to tray instead of exit.
                // The tray's Quit menu item is the only path that
                // actually ends the process.
                WindowEvent::CloseRequested { api, .. } => {
                    if hide_to_tray {
                        api.prevent_close();
                        if let Err(e) = window.hide() {
                            tracing::warn!(error = %e, "hide-on-close failed");
                        }
                    } else {
                        // No tray to come back through — closing means quit.
                        // (The hidden overlay window would otherwise keep
                        // the process alive with no way to reach it.)
                        window.app_handle().exit(0);
                    }
                }
                // On Windows, minimising fires Resized(0, 0). Treat
                // that as a hide-to-tray request so the app vanishes
                // from the taskbar entirely. macOS/Linux don't fire
                // a 0×0 resize on minimise, so this is a no-op there.
                WindowEvent::Resized(size)
                    if size.width == 0 && size.height == 0 && hide_to_tray =>
                {
                    if let Err(e) = window.hide() {
                        tracing::warn!(error = %e, "hide-on-minimise failed");
                    }
                }
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            // Lifecycle
            commands::get_app_info,
            commands::quit,
            // Backend
            commands::list_backends,
            commands::switch_backend,
            commands::active_backend,
            commands::select_model,
            // Models
            commands::search_hub_models,
            commands::list_local_models,
            commands::download_model,
            commands::cancel_download,
            commands::delete_model,
            // Audio
            commands::list_input_devices,
            commands::start_audio_monitor,
            commands::stop_audio_monitor,
            commands::current_audio_level,
            // Hotkey
            commands::register_hotkey,
            commands::register_repaste_hotkey,
            commands::unregister_hotkey,
            commands::current_hotkey,
            // Recording (also driven by hotkey, but exposed for testing)
            commands::start_recording,
            commands::stop_recording,
            commands::repaste_last,
            // History
            commands::list_history,
            commands::clear_history,
            commands::delete_history_entry,
            // Settings
            commands::get_settings,
            commands::set_settings,
            // Post-processing
            commands::preload_cleanup_model,
            // Permissions (mac)
            commands::check_permissions,
            commands::open_system_setting,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // `app.exit(0)` ultimately calls `std::process::exit`, which
            // skips Drop. The Python sidecar's Child has kill_on_drop
            // set, but that only fires on a clean drop — on a hard exit
            // we have to terminate it ourselves or it gets orphaned.
            if let RunEvent::Exit = event {
                if let Some(state) = app_handle.try_state::<state::AppState>() {
                    state.sidecar.shutdown_blocking();
                }
            }
        });
}

fn init_tracing() {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,talk_to_type_lib=debug"));

    let stderr_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_level(true);

    // Bundled Windows GUI builds have no console — without a file sink, every
    // tracing line is dropped and the app is undebuggable post-install.
    if let Some(dir) = log_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let file_appender = tracing_appender::rolling::daily(&dir, "talk-to-type.log");
        let file_layer = fmt::layer()
            .with_writer(file_appender)
            .with_target(false)
            .with_thread_ids(false)
            .with_level(true)
            .with_ansi(false);
        tracing_subscriber::registry()
            .with(filter)
            .with(stderr_layer)
            .with(file_layer)
            .init();
        tracing::info!(log_dir = %dir.display(), "file logger initialised");
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(stderr_layer)
            .init();
        tracing::warn!("could not determine log directory; logging to stderr only");
    }
}

fn log_dir() -> Option<std::path::PathBuf> {
    const IDENT: &str = "software.underscore.talktotype";
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir().map(|h| h.join("Library/Logs").join(IDENT))
    }
    #[cfg(not(target_os = "macos"))]
    {
        dirs::data_local_dir().map(|d| d.join(IDENT).join("logs"))
    }
}
