# talk-to-type sidecar

Python subprocess that the Rust host shells out to for the
**faster-whisper** and **Hugging Face Transformers** backends. Speaks
newline-delimited JSON-RPC over stdin/stdout. See
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) for protocol details.

## Local dev

```bash
cd sidecar
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

# Smoke test:
echo '{"id":1,"method":"ping","params":{}}' | python main.py
```

Tell the host to use this checkout instead of the bundled binary:

```bash
TALK_TO_TYPE_SIDECAR=$(pwd)/.venv/bin/python pnpm tauri dev
# (and pass `main.py` to that python interpreter — see Bash wrapper below)
```

## Building the bundled binary

```bash
./build.sh                   # CPU torch, default
PYTORCH_VARIANT=cuda ./build.sh   # use the CUDA torch already in your venv
```

The output is `dist/talk_to_type_sidecar[.exe]`. The Tauri bundler picks
this up via `tauri.conf.json → bundle.resources`.

## GPU notes

- **Windows + NVIDIA**: install `torch` with CUDA before running `build.sh`:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **macOS Apple Silicon**: torch's MPS backend works, but most ASR models are
  CPU-bound below the medium tier — leave `device="cpu"` unless you've
  benchmarked.
- **Intel macOS**: faster-whisper with `compute_type="int8"` is still the
  best path. CTranslate2 has no GPU support on Intel macs.
