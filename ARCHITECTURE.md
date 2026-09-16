# Architecture

Three views: process layout, audio pipeline, sidecar IPC. If you're new
to the codebase, read in that order.

## Process layout

```
                       ┌──────────────────────────────┐
                       │   talk-to-type (Tauri host)  │
                       │                              │
  ┌────────────────┐   │  ┌────────────────────────┐  │
  │ React UI       │◀──┼──│ commands.rs            │  │
  │  (settings,    │   │  │  #[tauri::command]     │  │
  │   models,      │   │  └────────────────────────┘  │
  │   history,     │   │  ┌────────────────────────┐  │
  │   overlay)     │   │  │ AppState               │  │
  └────────────────┘   │  │  audio · backend ·     │  │
                       │  │  sidecar · models ·    │  │
  ┌────────────────┐   │  │  history · settings    │  │
  │ Global hotkey  │──▶│  └────────────────────────┘  │
  │ (OS event tap) │   │             │                │
  └────────────────┘   │             ▼                │
                       │  ┌────────────────────────┐  │
                       │  │ TranscriptionBackend   │  │
                       │  │   trait                │  │
                       │  │  ├─ WhisperCpp (in-proc)│ │
                       │  │  ├─ FasterWhisper       │ │
                       │  │  └─ Transformers        │ │
                       │  └────────────────────────┘  │
                       └──────┬───────────────────────┘
                              │ stdin/stdout JSON-RPC
                              ▼
                       ┌────────────────────────┐
                       │ talk_to_type_sidecar   │
                       │  (Python, PyInstaller) │
                       │  faster-whisper        │
                       │  transformers          │
                       └────────────────────────┘
```

Three runtime processes:

1. **Tauri host** — Rust binary embedding a webview. Owns audio capture,
   the global hotkey, the active `TranscriptionBackend`, and the sidecar.
2. **WebView** — renders the settings/overlay UI; talks to the host
   exclusively through `tauri.invoke` and event subscriptions.
3. **Sidecar** — optional Python subprocess that runs faster-whisper or
   the HF Transformers ASR pipeline. Lives only when one of those two
   backends is active.

## Audio pipeline

The realtime loop must never block, never allocate beyond a small ring
buffer, and never panic.

```
   cpal input callback (sync, realtime priority)
        │  16-bit / 32-bit mixed → f32 mono
        │  resample to 16 kHz via rubato (Sinc, len 128)
        ▼
   parking_lot::Mutex<Vec<f32>>  ←—— shared ring buffer (capacity ~120 s)
        │
        │  on hotkey release: drain + clear
        ▼
   Optional: Silero VAD (voice_activity_detector crate)
        │  trims silence at start/end, pads ~120 ms either side
        ▼
   TranscriptionBackend::transcribe(&[f32], &BackendOptions)
        │
        ├─ WhisperCpp:   spawn_blocking → whisper_rs::create_state.full()
        ├─ FasterWhisper: write WAV → JSON-RPC → sidecar reads + decodes
        └─ Transformers:  same WAV → JSON-RPC → HF pipeline
        ▼
   TranscriptionResult { text, language, duration_ms }
        │
        │  Optional: Ollama post-processing (HTTP localhost:11434)
        ▼
   clipboard::write_text + enigo Cmd/Ctrl+V
```

Important properties:

- The cpal callback only ever does: down-mix → resample → push into the
  ring. It never holds locks across the resampler call (the resampler is
  itself protected by a small mutex used only inside the callback).
- Resampling is a one-time amortized cost: the `SincFixedIn` is built on
  the first callback invocation per recording session and reused.
- `MIN_DURATION_SAMPLES` = 300 ms. Anything shorter is treated as a
  hotkey misfire and silently discarded.
- VAD failures are non-fatal — we fall back to the raw audio and log a
  warning.

## Backend trait

```rust
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    async fn load_model(&self, path: &Path, opts: &BackendOptions) -> Result<()>;
    async fn is_ready(&self) -> bool;
    async fn transcribe(&self, samples: &[f32], opts: &BackendOptions) -> Result<TranscriptionResult>;
    async fn unload(&self) -> Result<()>;
}
```

Object-safe by design. Concrete implementations:

- **`WhisperCppBackend`** — wraps a `whisper_rs::WhisperContext`. `load_model`
  is idempotent for the same path. Inference runs on `tokio::task::spawn_blocking`
  to avoid stalling the runtime.
- **`FasterWhisperBackend`** / **`TransformersBackend`** — both sit on the
  shared `SidecarSupervisor`. They differ only in the `engine` field of
  the `load_model` JSON-RPC request and in their accepted "model path"
  format (Transformers accepts a `*.repo` marker file containing a
  Hugging Face repo id; faster-whisper expects a directory of CT2 weights).

Adding a fourth backend: implement the trait, add a variant to
`BackendKind`, wire it into `commands::switch_backend`, expose it in the
React Backend route. No other code touches backend internals.

## Sidecar IPC protocol

Newline-delimited JSON over stdin (host → sidecar) and stdout (sidecar →
host). Every request is independent — there's no streaming, no callbacks,
no server-initiated messages. The sidecar processes one request at a time
in `main()`'s `for line in sys.stdin` loop; concurrency is the host's
problem.

Why files instead of inline samples? PCM samples balloon JSON: 30 s of
16-bit/16 kHz audio is ~960 KB even after numpy's int16 packing, and HF
pipelines want a numpy array anyway. We write a 16-bit WAV to
`std::env::temp_dir()`, send the path, and unlink after the response.
Round-trip overhead is dominated by inference time; the file shuffle is
in the noise.

### Methods

| method        | params                                | result                                     |
| ------------- | ------------------------------------- | ------------------------------------------ |
| `ping`        | `{}`                                  | `{ "pong": true }`                         |
| `load_model`  | `{ engine, model, options }`          | `{ "ok": true }`                           |
| `transcribe`  | `{ samples_path, options }`           | `{ text, language?, duration_ms }`         |
| `unload`      | `{}`                                  | `{ "ok": true }`                           |

`engine` is `"faster_whisper"` or `"transformers"`. `options` is
backend-specific:

- faster-whisper load: `compute_type` (`int8` | `int8_float16` | `float16` | `float32`),
  `device` (`auto` | `cpu` | `cuda`).
- transformers load: `device` (`cpu` | `cuda` | `mps`).
- transcribe: `language` (ISO 639-1 string or null), `initial_prompt` (string
  or null, faster-whisper only).

### Error encoding

```json
{ "id": 42, "error": { "code": 1, "message": "<exception str>" } }
```

The host treats any non-zero code as failure; we don't currently
distinguish error classes (the host never recovers from a sidecar error
beyond surfacing it to the user and continuing).

### Lifecycle

The host spawns the sidecar lazily on first call to any `SidecarSupervisor::call`.
A reader task on stdout dispatches responses to per-id `oneshot::Sender`s.
A second task forwards stderr to `tracing::warn` so Python tracebacks land in
the host's log. On host exit, `kill_on_drop(true)` ensures the sidecar
goes with us.

Crash handling: when stdout closes (sidecar exited or crashed) the
running handle is marked dead and every pending request is failed
immediately with a transport error. The next `call` — or a retry of the
one that hit the dead pipe — respawns a fresh process. A respawned
sidecar has lost its in-memory models, so the sidecar-backed backends
watch for the "no ASR model loaded" error, replay `load_model` from
their cached state, and retry the transcription once. Net effect: a
mid-session sidecar crash costs one model reload, not an app restart.

## Frontend ↔ host boundary

- Commands are typed wrappers around `tauri.invoke` in `src/lib/api.ts`.
- Events flow host → frontend over `app.emit()` / `listen()`. Two events:
  - `overlay://state` — payload `OverlayPayload`. Drives the floating
    overlay window's animation.
  - `download://progress` — payload `DownloadProgress`. Drives the
    Models route's progress bars.
- The overlay is a separate webview window declared in `tauri.conf.json`
  (label `overlay`, transparent + always-on-top). It loads the same
  React bundle but at `#/overlay`, where `App.tsx` short-circuits the
  main shell and renders only the overlay component. Click-through is
  applied via `core:window:allow-set-ignore-cursor-events` permission +
  `set_ignore_cursor_events(true)` at show time.

## Storage layout

```
<app_data>/
├── settings.json            JSON-serialized Settings struct
├── history.db               SQLite via rusqlite (single-table schema)
└── models/
    ├── registry.json        Manifest of installed models
    ├── whisper-cpp/<repo>/<file>
    ├── faster-whisper/<repo>/<file>
    └── transformers/<repo>.repo   (HF id marker; weights live in HF cache)
```

Where `<app_data>` is:

- macOS: `~/Library/Application Support/software.underscore.talktotype/`
- Windows: `%APPDATA%\software.underscore.talktotype\`

History is trimmed to `Settings.maxHistory` rows after every insert
(default 100, configurable).
