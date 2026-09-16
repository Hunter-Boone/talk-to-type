// Mirrored from src-tauri/src/* serde definitions. Keep in sync if you
// rename a field on the Rust side — TypeScript can't catch a drift here.

export type BackendKind =
  | "whisper-cpp"
  | "faster-whisper"
  | "transformers"
  // Pseudo-backend that piggybacks on the model registry + HF download
  // flow but isn't a transcription backend. Used by the cleanup LLM
  // post-processing provider. Excluded from the ASR backend picker.
  | "cleanup-transformers";

export interface BackendDescriptor {
  kind: BackendKind;
  label: string;
  description: string;
}

export interface BackendOptions {
  threads?: number;
  computeType?: string;
  device?: string;
  language?: string;
  initialPrompt?: string;
}

export type PostAction = "paste" | "clipboard-only";

export type PostProcessingProvider = "ollama" | "cursor-cli" | "transformers";

export interface PostProcessing {
  enabled: boolean;
  /// Engine that runs the cleanup pass. Defaults to Cursor CLI on fresh
  /// installs; Ollama is kept for users who already have a local model
  /// server running; Transformers runs an HF causal-LM in the same
  /// Python sidecar that hosts the ASR engines.
  provider: PostProcessingProvider;

  /** Ollama HTTP base URL (e.g. `http://localhost:11434`). */
  ollamaUrl: string;
  /** Ollama model tag (e.g. `llama3.2:3b`). */
  model: string;

  /** Cursor model name passed via `--model` (`auto`, `composer-1`, ...). */
  cursorModel: string;
  /** Optional override for the `cursor-agent` binary location. */
  cursorCliPath: string | null;

  /** Local-model registry id of the cleanup LLM to load. */
  transformersModelId: string | null;
  /** torch device passed to the sidecar (`cpu`, `cuda`, `mps`). */
  transformersDevice: string;
  /** Cap on tokens generated per cleanup pass. */
  transformersMaxTokens: number;

  systemPrompt: string;
}

export interface Settings {
  launchAtLogin: boolean;
  showInTray: boolean;
  hotkey: string;
  /// Optional second global shortcut that re-pastes the most recent
  /// transcription. `null` or empty string = disabled.
  repasteHotkey: string | null;
  postAction: PostAction;
  inputDevice: string | null;
  vadEnabled: boolean;
  /// Experimental: while you hold the hotkey, periodically transcribe the
  /// audio captured so far. On release, if the most recent chunk's audio
  /// length is within ~250 ms of the final length, reuse that result for
  /// an instant paste instead of running another inference pass.
  pretranscribe: boolean;
  activeBackend: BackendKind;
  activeModelId: string | null;
  backendOptions: BackendOptions;
  postProcessing: PostProcessing;
  maxHistory: number;
  /// HuggingFace user access token (for gated/private repos).
  hfToken: string | null;
}

export interface InputDeviceInfo {
  name: string;
  isDefault: boolean;
}

export interface HubSearchResult {
  repoId: string;
  author: string | null;
  downloads: number | null;
  likes: number | null;
  lastModified: string | null;
  pipelineTag: string | null;
  tags: string[];
}

export interface LocalModel {
  id: string;
  backend: BackendKind;
  name: string;
  path: string;
  repoId: string | null;
  size: number;
  languages: string[];
  description: string | null;
  installedAt: string | null;
}

export interface HistoryEntry {
  id: number;
  createdAt: string;
  text: string;
  backend: string;
  model: string | null;
  language: string | null;
  durationMs: number;
  audioSeconds: number;
}

export interface DownloadProgress {
  jobId: string;
  repoId: string;
  file: string;
  downloaded: number;
  total: number | null;
  finished: boolean;
  error: string | null;
}

export type OverlayState =
  | "hidden"
  | "listening"
  | "transcribing"
  | "pasted"
  | "error";

export interface OverlayPayload {
  state: OverlayState;
  message?: string;
  peak?: number;
}

export interface AppInfo {
  version: string;
  platform: string;
}
