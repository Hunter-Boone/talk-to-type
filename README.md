# talk-to-type

Privacy-first local voice dictation for Windows and macOS. Hold a global
hotkey, speak, release — your speech is transcribed locally and pasted
into the focused app. No audio ever leaves your device.

Three pluggable backends:

| Backend           | Process       | Strengths                                                           |
| ----------------- | ------------- | ------------------------------------------------------------------- |
| **whisper.cpp**   | in-process    | Smallest install, Metal/CUDA/Vulkan acceleration, no Python needed. |
| **faster-whisper**| Python sidecar| INT8 CTranslate2 quantization. Fastest CPU path on x86.             |
| **Transformers** | Python sidecar| Any Hugging Face ASR pipeline (Whisper, Distil-Whisper, Parakeet…). |

---

## Prerequisites

- **Rust** 1.77+ (`rustup install stable`)
- **Node.js** 20+ — npm works fine; if you prefer pnpm, use **pnpm 9+**
  (pnpm 7 has an `ERR_INVALID_THIS` regression with Node 22).
- **Python** 3.10–3.12 (only if you want to use the sidecar backends)
- Platform toolchain:
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Visual Studio **2022** Build Tools (NOT 2019) with the
    "Desktop development with C++" workload, the Windows 11 SDK, and the
    "C++ CMake tools" component. The `ort-sys` (ONNX Runtime) and
    `whisper-rs-sys` crates link against MSVC 14.30+ symbols; VS 2019's
    14.28 toolset will fail with errors like
    `unresolved external symbol __std_find_trivial_8`.
  - **Windows + whisper.cpp**: also install **LLVM** (provides `libclang`
    needed by bindgen). Use `winget install LLVM.LLVM` and ensure
    `LIBCLANG_PATH` points at `C:\Program Files\LLVM\bin`. Alternatively,
    build without whisper.cpp via
    `cargo build --no-default-features` and rely on the sidecar backends.

## Run in dev

```bash
npm install
npm run tauri:dev
```

The first build downloads ~2 GB of crates and links libwhisper. Subsequent
builds are <30 s.

To use the faster-whisper or Transformers backends in dev, build the
sidecar once:

```bash
cd sidecar
./build.sh
cd ..
TALK_TO_TYPE_SIDECAR="$(pwd)/sidecar/dist/talk_to_type_sidecar" pnpm tauri dev
```

Or, for a faster iteration loop, run the sidecar via Python directly:

```bash
TALK_TO_TYPE_SIDECAR="$(pwd)/sidecar/.venv/bin/python" pnpm tauri dev
# (and edit src-tauri/src/sidecar/supervisor.rs to pass main.py as an arg)
```

## Release builds

```bash
# macOS — produces dmg + app bundle for the host architecture.
npm run tauri:build

# Cross-arch on Apple Silicon (build x86_64 + aarch64 dmgs):
npm run tauri:build --target aarch64-apple-darwin
npm run tauri:build --target x86_64-apple-darwin

# Windows — produces MSI + NSIS .exe.
npm run tauri:build
```

GPU-accelerated builds:

```bash
# whisper.cpp on Apple Silicon
npm run tauri:build -- --features metal

# whisper.cpp on Windows + NVIDIA
npm run tauri:build -- --features cuda
```

## Code-signing and notarization

The project does **not** sign release artefacts by default. Plug in your
own credentials:

### macOS

1. Set `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD` (an
   app-specific password from appleid.apple.com), and `APPLE_TEAM_ID` in
   your environment.
2. `npm run tauri:build` will sign and notarize.
3. `src-tauri/entitlements.plist` enables the `allow-jit` and
   `disable-library-validation` entitlements that PyTorch requires; do not
   remove them or the bundled sidecar will fail to start.

### Windows

Configure `tauri.conf.json → bundle.windows.signCommand` to call your code
signing tool against the produced `.exe`/`.msi`. We don't bake a default
in because every shop has different signing infrastructure (sectigo,
azure trusted-signing, smart cards, etc.).

A stub GitHub Actions workflow lives at `.github/workflows/release.yml`
(create it from the template comments — it's intentionally not committed
to keep the repo clean of half-finished CI).

## Permissions

### macOS — first-run consent dialogs

Three OS permissions are requested on first use:

1. **Microphone** — opens automatically when audio capture starts.
2. **Accessibility** — required for `enigo` to simulate Cmd+V. The OS
   prompts the first time you trigger a paste; click "Open System Settings"
   and toggle talk-to-type on.
3. **Input Monitoring** — required for the global hotkey to fire while
   talk-to-type is in the background. Same flow as Accessibility.

The app's onboarding screen surfaces buttons that deep-link to each pane
in System Settings.

### Windows

Microphone permission is requested via the Privacy & Security settings
the first time we open the input stream. No installer elevation is needed
unless you're shipping a per-machine NSIS install (we ship the per-user
default).

## Troubleshooting

**App doesn't start after a reboot** — enable Settings → General →
"Launch at login". This registers the app with the OS (HKCU Run key on
Windows, LaunchAgent on macOS) to start with a `--hidden` flag, so it
opens straight to the tray. The registration is re-synced from settings
on every app start, so toggling it once is enough.

**"sidecar binary not found"** — build the sidecar
(`cd sidecar && ./build.sh`) or set `TALK_TO_TYPE_SIDECAR` to the path of
your dev `python` and pass `main.py` to it.

**Hotkey doesn't fire on macOS** — open System Settings → Privacy &
Security → Input Monitoring, ensure talk-to-type is listed and enabled.
Some apps capture keystrokes before the OS dispatches them (game
launchers, screen recorders); try a different combo.

**`Cmd+Shift+Space` collides with another app** — change it in Settings →
General → Hotkey. The format is `Mod1+Mod2+Key` (e.g. `Alt+F12`,
`Ctrl+Alt+M`).

**Transcription is gibberish** — the wrong language is being auto-detected.
Set a language hint in Settings → Backend → Language hint
(`en`, `de`, `fr`, …).

**Whisper.cpp fails to load on Windows** — make sure you built with the
right feature flag for your hardware (`cuda` for NVIDIA, `vulkan` for AMD/
Intel iGPU, none for CPU-only).

**Bundle is huge (>500 MB)** — that's the Python sidecar carrying torch.
Use the whisper.cpp backend; the build skips the sidecar bundle if
`bundle.resources` is empty.

## License

MIT. See `LICENSE`.
