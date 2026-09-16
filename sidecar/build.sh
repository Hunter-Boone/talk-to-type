#!/usr/bin/env bash
# Build the sidecar into a standalone executable using PyInstaller.
#
# Output:
#   dist/talk_to_type_sidecar          (macOS / Linux)
#   dist/talk_to_type_sidecar.exe      (Windows, when run under Git Bash / WSL)
#
# We chose PyInstaller over a uv-managed venv because:
#   - the resulting binary is a single drop-in artifact the Tauri bundler
#     can include via the `bundle.resources` map,
#   - users don't need a working Python on their machine,
#   - launch is faster (no fork+exec of /usr/bin/python every time),
#   - fewer "is the right venv selected?" failure modes in the field.
#
# Trade-offs:
#   - Bundle is 400-800 MB depending on torch flavour.
#   - Code-signing on macOS requires the entitlements in
#     src-tauri/entitlements.plist (allow-jit + disable-library-validation).
#
# Usage:
#   cd sidecar
#   ./build.sh                       # CPU torch, default
#   PYTORCH_VARIANT=cuda ./build.sh  # CUDA torch (cu124 by default)
#   PYTORCH_VARIANT=cuda TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ./build.sh

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    echo "Creating venv at sidecar/.venv ..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv
    else
        python -m venv .venv
    fi
fi

# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f .venv/Scripts/activate ]]; then
    source .venv/Scripts/activate
else
    echo "Could not find venv activation script" >&2
    exit 1
fi

# Default PyPI gives CPU-only torch on Linux/Windows. Force the CUDA wheel
# before the editable install so pip's resolver doesn't pull CPU torch as a
# transitive of transformers/accelerate. --force-reinstall covers the case
# where a CPU wheel is already in the venv from a prior run.
PYTORCH_VARIANT="${PYTORCH_VARIANT:-cpu}"
if [[ "$PYTORCH_VARIANT" == "cuda" ]]; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
    echo "Installing CUDA torch from $TORCH_INDEX_URL ..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --force-reinstall torch --index-url "$TORCH_INDEX_URL"
    else
        pip install --force-reinstall torch --index-url "$TORCH_INDEX_URL"
    fi
fi

echo "Installing dependencies ..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[build]"
else
    pip install -e ".[build]"
fi

echo "Building with PyInstaller ..."
rm -rf build dist
# Note: `--console` is correct on macOS/Linux. The Windows build (build.ps1)
# uses `--windowed` instead so spawning from the GUI-subsystem Tauri host
# doesn't pop up a console window. We can't share that flag across platforms
# because PyInstaller's `--windowed` on macOS produces a `.app` bundle and
# changes the output layout.
pyinstaller \
    --noconfirm \
    --onedir \
    --name talk_to_type_sidecar \
    --console \
    --hidden-import faster_whisper \
    --hidden-import ctranslate2 \
    --hidden-import transformers \
    --collect-data transformers \
    --collect-data tokenizers \
    main.py

echo
echo "Built: dist/$(ls dist | head -n1)"
