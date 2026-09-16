// Thin typed wrappers around `tauri.invoke`. One per Rust command. If the
// Rust side renames a command, only this file changes.

import { invoke } from "@tauri-apps/api/core";
import type {
  AppInfo,
  BackendDescriptor,
  BackendKind,
  HistoryEntry,
  HubSearchResult,
  InputDeviceInfo,
  LocalModel,
  Settings,
} from "./types";

export const api = {
  // Lifecycle
  appInfo: () => invoke<AppInfo>("get_app_info"),
  quit: () => invoke<void>("quit"),

  // Backends
  listBackends: () => invoke<BackendDescriptor[]>("list_backends"),
  switchBackend: (kind: BackendKind) =>
    invoke<void>("switch_backend", { kind }),
  activeBackend: () => invoke<BackendKind | null>("active_backend"),
  selectModel: (modelId: string) =>
    invoke<void>("select_model", { modelId }),

  // Models
  searchHubModels: (backend: BackendKind, query: string, limit = 40) =>
    invoke<HubSearchResult[]>("search_hub_models", { backend, query, limit }),
  listLocalModels: (backend?: BackendKind) =>
    invoke<LocalModel[]>("list_local_models", { backend: backend ?? null }),
  downloadModel: (req: {
    backend: BackendKind;
    repoId: string;
    file: string | null;
    displayName: string;
    languages?: string[];
    description?: string | null;
  }) => invoke<LocalModel>("download_model", { req }),
  cancelDownload: (jobId: string) =>
    invoke<void>("cancel_download", { jobId }),
  deleteModel: (id: string) => invoke<void>("delete_model", { id }),

  // Audio
  listInputDevices: () => invoke<InputDeviceInfo[]>("list_input_devices"),
  startAudioMonitor: (device?: string | null) =>
    invoke<void>("start_audio_monitor", { device: device ?? null }),
  stopAudioMonitor: () => invoke<void>("stop_audio_monitor"),
  currentAudioLevel: () => invoke<number>("current_audio_level"),

  // Hotkey
  registerHotkey: (combo?: string) =>
    invoke<void>("register_hotkey", { combo: combo ?? null }),
  /// Pass an empty string or null to disable the repaste hotkey.
  registerRepasteHotkey: (combo: string | null) =>
    invoke<void>("register_repaste_hotkey", { combo }),
  unregisterHotkey: () => invoke<void>("unregister_hotkey"),
  currentHotkey: () => invoke<string>("current_hotkey"),

  // Recording (manual triggers — usually driven by hotkey)
  startRecording: () => invoke<void>("start_recording"),
  stopRecording: () => invoke<void>("stop_recording"),
  repasteLast: () => invoke<void>("repaste_last"),

  // History
  listHistory: (limit = 100) => invoke<HistoryEntry[]>("list_history", { limit }),
  clearHistory: () => invoke<void>("clear_history"),
  deleteHistoryEntry: (id: number) =>
    invoke<void>("delete_history_entry", { id }),

  // Settings
  getSettings: () => invoke<Settings>("get_settings"),
  setSettings: (settings: Settings) =>
    invoke<void>("set_settings", { settings }),

  // Post-processing
  preloadCleanupModel: () => invoke<void>("preload_cleanup_model"),

  // Permissions
  checkPermissions: () =>
    invoke<{
      microphone: string;
      accessibility: string;
      inputMonitoring: string;
    }>("check_permissions"),
  openSystemSetting: (pane: "microphone" | "accessibility" | "input-monitoring") =>
    invoke<void>("open_system_setting", { pane }),
};
