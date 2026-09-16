# Build the sidecar into a standalone executable using PyInstaller (Windows / PowerShell).
# Mirror of build.sh — see that file for rationale and trade-offs.
#
# Output:
#   dist\talk_to_type_sidecar\talk_to_type_sidecar.exe (+ supporting files)
#
# Usage (from PowerShell):
#   cd sidecar
#   .\build.ps1                              # CPU torch, default
#   $env:PYTORCH_VARIANT='cuda'; .\build.ps1 # CUDA torch (cu124 by default)
#   $env:PYTORCH_VARIANT='cuda'; $env:TORCH_INDEX_URL='https://download.pytorch.org/whl/cu126'; .\build.ps1
#
# If you hit "running scripts is disabled" when activating the venv, run:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$hasUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

if (-not (Test-Path .venv)) {
    Write-Host "Creating venv at sidecar\.venv ..."
    if ($hasUv) {
        uv venv .venv
    } else {
        python -m venv .venv
    }
}

$activate = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $activate)) {
    throw "Could not find venv activation script at $activate"
}
& $activate

# Default PyPI gives CPU-only torch on Windows. Force the CUDA wheel before
# the editable install so pip's resolver doesn't pull CPU torch as a transitive
# of transformers/accelerate. --force-reinstall covers the case where a CPU
# wheel is already in the venv from a prior run.
$variant = if ($env:PYTORCH_VARIANT) { $env:PYTORCH_VARIANT } else { 'cpu' }
if ($variant -eq 'cuda') {
    $torchIndex = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/cu124' }
    Write-Host "Installing CUDA torch from $torchIndex ..."
    if ($hasUv) {
        uv pip install --force-reinstall torch --index-url $torchIndex
    } else {
        pip install --force-reinstall torch --index-url $torchIndex
    }
}

Write-Host "Installing dependencies ..."
if ($hasUv) {
    uv pip install -e ".[build]"
} else {
    pip install -e ".[build]"
}

Write-Host "Building with PyInstaller ..."
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist)  { Remove-Item dist  -Recurse -Force }

# `--windowed` builds a GUI-subsystem exe so spawning it from the Tauri
# host doesn't pop up a console window. JSON-RPC over stdin/stdout still
# works because the parent explicitly pipes those handles via
# `Stdio::piped()`. Use `--console` instead if you need to launch the
# binary standalone for debugging.
pyinstaller `
    --noconfirm `
    --onedir `
    --name talk_to_type_sidecar `
    --windowed `
    --hidden-import faster_whisper `
    --hidden-import ctranslate2 `
    --hidden-import transformers `
    --collect-data transformers `
    --collect-data tokenizers `
    main.py

Write-Host ""
$built = Get-ChildItem dist | Select-Object -First 1
Write-Host "Built: dist\$($built.Name)"
