"""talk-to-type Python sidecar.

Spawned by the Rust host as a subprocess. Speaks newline-delimited JSON-RPC
over stdin/stdout. Diagnostics go to stderr (the host forwards them to its
own tracing).

Wire format:
    request:  {"id": <u64>, "method": "<name>", "params": <obj>}
    response: {"id": <u64>, "result": <obj>}            on success
              {"id": <u64>, "error":  {"code": <i32>, "message": <str>}}

Methods:
    ping          → {"pong": true}
    load_model    {engine, model, options}              → {"ok": true}
    transcribe    {samples_path, options}               →
        {"text", "language", "duration_ms",
         "words": [{"text", "start", "end"}, ...]}
    cleanup       {system_prompt, raw_text, options}    →
        {"text", "duration_ms"}
    unload        {engine?}                             → {"ok": true}

If `options.word_timestamps` is true, the response's `words` array
holds per-word timing (in seconds, relative to the start of
`samples_path`). Engines that don't support timestamps for the
chosen model architecture return `"words": []` and the host falls
back to a non-timestamped concat path.

Engines, lazy-imported on first use, and the slot they occupy:
    - faster_whisper      → ASR slot         (CTranslate2)
    - transformers        → ASR slot         (HF speech-seq2seq)
    - cleanup             → cleanup slot     (HF causal-LM, chat-templated)

Two slots can be loaded simultaneously: one ASR model + one cleanup
LLM. `unload` with no params drops both; with `{"engine": "asr"}` or
`{"engine": "cleanup"}` it drops just that slot.

We deliberately import nothing heavy at startup. PyTorch and CTranslate2
each take ~1 s to import — the host doesn't pay that cost until it
actually picks a sidecar-backed backend.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Force unbuffered IO. PyInstaller-bundled apps default to buffered, which
# would cause requests to sit in the kernel pipe forever.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def log(msg: str) -> None:
    """Write a diagnostic to stderr; host forwards to its tracing layer."""
    print(msg, file=sys.stderr, flush=True)


# =============================================================================
# Engines
# =============================================================================


@dataclass
class LoadedFasterWhisper:
    model: Any  # faster_whisper.WhisperModel
    options: dict


@dataclass
class LoadedTransformers:
    model: Any        # transformers.PreTrainedModel
    processor: Any    # transformers.ProcessorMixin
    device: Any       # torch.device
    dtype: Any        # torch.dtype
    options: dict


@dataclass
class LoadedCleanup:
    model: Any        # transformers.PreTrainedModel (causal LM)
    tokenizer: Any    # transformers.PreTrainedTokenizerBase
    device: Any       # torch.device
    dtype: Any        # torch.dtype
    options: dict


# Two independent slots: one ASR model and one cleanup LLM can live in
# memory at the same time. The host explicitly chose to allow this so
# users running Transformers ASR + Transformers cleanup don't pay
# load/unload latency on every utterance.
_state: dict[str, Any] = {
    "asr_engine": None,            # "faster_whisper" | "transformers" | None
    "asr_loaded": None,            # LoadedFasterWhisper | LoadedTransformers | None
    "asr_model_key": None,         # str — for idempotency on reload
    "cleanup_loaded": None,        # LoadedCleanup | None
    "cleanup_model_key": None,     # str
}


def _load_faster_whisper(model: str, options: dict) -> LoadedFasterWhisper:
    # Lazy import — fastest startup if user only ever uses transformers.
    from faster_whisper import WhisperModel  # type: ignore[import]

    compute_type = options.get("compute_type", "int8")
    device = options.get("device", "auto")

    log(f"loading faster-whisper from {model} (device={device}, compute_type={compute_type})")
    wm = WhisperModel(model, device=device, compute_type=compute_type)
    return LoadedFasterWhisper(model=wm, options=options)


def _resolve_torch_device(requested: str) -> tuple[Any, Any]:
    """Translate a string device request into (torch.device, dtype),
    falling back to CPU with a loud log when the requested accelerator
    isn't actually available. Shared by both ASR-transformers and the
    cleanup LLM loader so the diagnostics are identical."""
    import torch  # type: ignore[import]

    if requested == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            log(
                "device=cuda requested but torch.cuda.is_available() is False — "
                "your installed torch wheel may be CPU-only (look for '+cpu' in "
                "torch.__version__) or you have no NVIDIA GPU/driver. Falling "
                "back to CPU. To use the GPU, reinstall torch from "
                "https://download.pytorch.org/whl/cu124 (or your CUDA's index)."
            )
            device = torch.device("cpu")
    elif requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            log("device=mps requested but not available; falling back to CPU.")
            device = torch.device("cpu")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype


def _load_transformers(model: str, options: dict) -> LoadedTransformers:
    """Load a HF Transformers ASR model directly (without pipeline()).

    Going through `pipeline()` is convenient but it doesn't forward kwargs
    to the processor — which breaks multilingual ASR repos like Cohere
    Transcribe that take a `language=` arg on the processor to set up
    decoder conditioning. Calling the processor and `model.generate()`
    directly gives us full control.

    We deliberately load with `trust_remote_code=False`. The Cohere
    Transcribe repo ships a `modeling_cohere_asr.py` (custom_code) that
    is *known broken* on transformers ≥ 5.3 — it produces multilingual
    token salad. Native support landed in transformers 5.4+, so when we
    don't pass `trust_remote_code=True` the AutoModel resolves to the
    bundled `CohereAsrForConditionalGeneration` which works correctly.
    See https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/28
    for the upstream confirmation.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore[import]

    device, dtype = _resolve_torch_device(options.get("device", "cpu"))

    log(f"loading transformers ASR model from {model} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=False)
    m = AutoModelForSpeechSeq2Seq.from_pretrained(
        model,
        trust_remote_code=False,
        dtype=dtype,
    )
    m.to(device)
    m.eval()
    return LoadedTransformers(
        model=m, processor=processor, device=device, dtype=dtype, options=options
    )


def _load_audio(path: str, target_sr: int = 16_000) -> tuple[Any, int]:
    """Load a WAV at any sample rate, downmix to mono, resample to
    `target_sr`, return (float32 ndarray in [-1, 1], target_sr).

    librosa uses scipy's polyphase resampler under the hood — same
    algorithm class as ffmpeg/sox, with proper anti-alias filtering.
    Doing the resample here instead of in the Rust host means the
    model receives the cpal-native audio routed through one
    well-tested conversion, rather than two homebrew passes.
    """
    import librosa  # type: ignore[import]

    samples, sr = librosa.load(path, sr=target_sr, mono=True, dtype="float32")
    return samples, sr


def _transcribe_faster_whisper(
    loaded: LoadedFasterWhisper, samples_path: str, options: dict
) -> dict:
    # Hand the WAV to faster-whisper directly; it shells out to
    # ffmpeg internally for any rate conversion we need. This is
    # better than us pre-resampling in librosa because faster-whisper
    # also strips silence padding the same way it does at evaluation
    # time, which helps decoder accuracy on short clips.
    language = options.get("language") or None
    initial_prompt = options.get("initial_prompt") or None
    want_words = bool(options.get("word_timestamps") or False)

    started = time.time()
    segments, info = loaded.model.transcribe(
        samples_path,
        language=language,
        initial_prompt=initial_prompt,
        beam_size=5,
        vad_filter=False,  # Host already trimmed with Silero.
        word_timestamps=want_words,
    )

    text_parts: list[str] = []
    words: list[dict] = []
    # `segments` is a generator — we have to drain it once. Build the
    # text and (optionally) the per-word timing in the same pass.
    for seg in segments:
        text_parts.append(seg.text)
        if want_words and getattr(seg, "words", None):
            for w in seg.words:
                # faster-whisper emits whitespace as a leading space
                # on each word ("hello", " world"). The Rust side
                # relies on that leading-space convention to detect
                # word boundaries — pass it through untouched.
                start = float(getattr(w, "start", 0.0) or 0.0)
                end = float(getattr(w, "end", start) or start)
                if end < start:
                    end = start
                words.append({"text": w.word, "start": start, "end": end})

    text = "".join(text_parts).strip()
    duration_ms = int((time.time() - started) * 1000)
    return {
        "text": text,
        "language": getattr(info, "language", None),
        "duration_ms": duration_ms,
        "words": words,
    }


def _transcribe_transformers(
    loaded: LoadedTransformers, samples_path: str, options: dict
) -> dict:
    import torch  # type: ignore[import]

    samples, sr = _load_audio(samples_path, target_sr=16_000)
    started = time.time()

    # Multilingual ASR models hallucinate without a language prior.
    # All the modern Transformers ASR processors we care about
    # (Whisper, Cohere Transcribe, Seamless, Parakeet) accept
    # `language=` on the processor itself; the processor then sets
    # whatever decoder hint the underlying model needs (Whisper:
    # forced_decoder_ids; Cohere: a language embedding bound into
    # the audio features). The processor-level hint is what gets
    # picked up automatically by `model.generate()` — explicitly
    # passing `language=` or `task=` to `generate()` is a Whisper-
    # only convention that Cohere silently accepts as ignored
    # kwargs and then produces token salad. Don't do it.
    language = options.get("language") or "en"

    proc_kwargs: dict = {"sampling_rate": sr, "return_tensors": "pt"}
    try:
        inputs = loaded.processor(samples, language=language, **proc_kwargs)
    except TypeError:
        inputs = loaded.processor(samples, **proc_kwargs)

    # Pull `audio_chunk_index` out before `.to(device)` — it's the
    # tensor Cohere's decoder uses to stitch long-form chunks back
    # together, and it's a non-tensor or CPU-only field on some
    # transformers versions.
    audio_chunk_index = None
    if hasattr(inputs, "get"):
        audio_chunk_index = inputs.get("audio_chunk_index")

    # `BatchFeature.to(device, dtype=...)` only casts floating-point
    # tensors, leaving integer tensors (decoder ids, attention
    # masks, chunk indices) intact.
    inputs = inputs.to(loaded.device, dtype=loaded.dtype)

    gen_kwargs: dict = {"max_new_tokens": 256}
    with torch.inference_mode():
        outputs = loaded.model.generate(**inputs, **gen_kwargs)

    text = _decode_transformers_output(
        loaded.processor, outputs, audio_chunk_index, language
    )
    duration_ms = int((time.time() - started) * 1000)
    # Word-level timestamps are model-architecture-specific via the raw
    # `generate()` path (Whisper supports `return_timestamps=True`, but
    # Cohere/Seamless/Parakeet decode through different routes that
    # we'd need bespoke logic for). Return [] here and let the host
    # fall back to its non-timestamped append path. Users who want the
    # full streaming-quality pipeline should select faster-whisper or
    # whisper.cpp, both of which expose timestamps natively.
    return {
        "text": text,
        "language": language,
        "duration_ms": duration_ms,
        "words": [],
    }


def _decode_transformers_output(
    processor: Any, outputs: Any, audio_chunk_index: Any, language: str
) -> str:
    """Decode generation output, handling both Cohere-style
    `processor.decode(..., audio_chunk_index=, language=)` and
    plain Whisper-style `processor.batch_decode(...)`. Cohere's
    decode reassembles per-chunk transcripts using the chunk
    index; without it long-form audio comes out jumbled.
    """
    # Try the Cohere-friendly path first. It accepts a batch-shaped
    # tensor and returns a list[str], so the call works for any
    # processor whose `decode` takes **kwargs.
    try:
        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        if isinstance(decoded, list):
            return (decoded[0] if decoded else "").strip()
        return str(decoded).strip()
    except TypeError:
        pass

    # Whisper / Wav2Vec2 / generic — `decode` is per-sample, so
    # use batch_decode for safety.
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    return (decoded[0] if decoded else "").strip()


def _load_cleanup(model: str, options: dict) -> LoadedCleanup:
    """Load a HuggingFace causal-LM for the cleanup pass.

    The cleanup pipeline mirrors the ASR-transformers loader but uses
    `AutoModelForCausalLM` + `AutoTokenizer`. We rely on the
    tokenizer's chat template to format the system+user messages,
    which is what every modern instruct model (Llama-3.x-Instruct,
    Qwen2.5-Instruct, SmolLM2-Instruct, Phi-3.5-Instruct, etc.) ships.
    Models without a chat template fall back to a plain "system /
    user / assistant" string concat at generate time.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    device, dtype = _resolve_torch_device(options.get("device", "cpu"))

    log(f"loading cleanup LLM from {model} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    m = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=False,
        dtype=dtype,
    )
    m.to(device)
    m.eval()
    return LoadedCleanup(model=m, tokenizer=tokenizer, device=device, dtype=dtype, options=options)


def _run_cleanup(loaded: LoadedCleanup, params: dict) -> dict:
    """Run a single cleanup pass: format the (system, user) pair via
    the tokenizer's chat template (or a plain fallback if absent),
    generate greedily, and return only the newly-generated text.

    `do_sample=False` keeps the cleanup deterministic — we do NOT
    want a temperature-driven LLM rewriting words it thinks were
    misheard. The system prompt forbids that, but a non-zero
    temperature gives it permission to drift anyway.
    """
    import torch  # type: ignore[import]

    system_prompt = params.get("system_prompt", "")
    raw_text = params.get("raw_text", "")
    options = params.get("options", {}) or {}
    max_new_tokens = int(options.get("max_new_tokens", 512))

    started = time.time()

    chat_template = getattr(loaded.tokenizer, "chat_template", None)
    if chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ]
        # `enable_thinking=False` suppresses the `<think>...</think>`
        # preamble that Qwen3 (and other thinking-mode models) emit by
        # default. We don't want chain-of-thought for a text cleanup
        # task — it just burns tokens and risks the model "rethinking"
        # the user's words. The kwarg is passed straight through to
        # the Jinja chat template as a variable; templates that don't
        # reference it ignore it silently, so it's safe to set on
        # every model.
        template_kwargs = {"enable_thinking": False}
        try:
            encoded = loaded.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        except Exception as exc:
            # Some models reject a system role (e.g. Gemma). Fold the
            # system prompt into the user message and retry.
            log(f"chat template rejected system role ({exc}); folding into user message")
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{raw_text}"},
            ]
            encoded = loaded.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
    else:
        # Fallback for models without a chat template — plain text
        # concat with explicit role markers. Less reliable than a
        # proper chat template but covers the long tail.
        prompt = (
            f"{system_prompt}\n\n"
            f"### Input\n{raw_text}\n\n"
            f"### Output\n"
        )
        encoded = loaded.tokenizer(prompt, return_tensors="pt")

    # Normalise the encoder output. Across transformers versions:
    #   - `apply_chat_template(..., return_tensors="pt")` returns either a
    #     bare LongTensor (older releases) or a BatchEncoding/dict
    #     (transformers 5.x). The latter goes through BatchEncoding's
    #     `__getattr__` and would silently swallow `.shape` accesses as
    #     KeyError → AttributeError, hence we cannot just call
    #     `.shape` on the result.
    #   - `tokenizer(prompt, return_tensors="pt")` always returns a
    #     BatchEncoding.
    # Funnel everything into a plain dict of {input_ids, attention_mask?},
    # then push every tensor to device with `.to`.
    if isinstance(encoded, torch.Tensor):
        inputs = {"input_ids": encoded}
    elif hasattr(encoded, "data"):
        inputs = dict(encoded.data)
    else:
        inputs = dict(encoded)

    inputs = {
        k: (v.to(loaded.device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[1]

    pad_id = loaded.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = loaded.tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    generated = outputs[0, input_len:]
    text = loaded.tokenizer.decode(generated, skip_special_tokens=True).strip()
    duration_ms = int((time.time() - started) * 1000)
    return {"text": text, "duration_ms": duration_ms}


# =============================================================================
# RPC dispatch
# =============================================================================


def _gc_torch() -> None:
    """Drop torch caches after unloading — important on CUDA where
    `del model` alone leaves the cached blocks pinned to the device."""
    import gc

    gc.collect()
    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def handle(method: str, params: dict) -> Any:
    if method == "ping":
        return {"pong": True}

    if method == "load_model":
        engine = params.get("engine")
        model = params.get("model")
        options = params.get("options", {}) or {}
        # Options are part of the identity: switching device (cpu→cuda) or
        # compute_type must trigger a real reload, not hit the idempotency
        # short-circuit.
        key = f"{engine}:{model}:{json.dumps(options, sort_keys=True)}"

        if engine == "cleanup":
            if _state["cleanup_model_key"] == key:
                return {"ok": True}  # idempotent
            _state["cleanup_loaded"] = _load_cleanup(model, options)
            _state["cleanup_model_key"] = key
            return {"ok": True}

        if engine in ("faster_whisper", "transformers"):
            if _state["asr_engine"] == engine and _state["asr_model_key"] == key:
                return {"ok": True}  # idempotent
            if engine == "faster_whisper":
                loaded = _load_faster_whisper(model, options)
            else:
                loaded = _load_transformers(model, options)
            _state["asr_engine"] = engine
            _state["asr_loaded"] = loaded
            _state["asr_model_key"] = key
            return {"ok": True}

        raise ValueError(f"unknown engine: {engine}")

    if method == "transcribe":
        loaded = _state["asr_loaded"]
        if loaded is None:
            raise RuntimeError("no ASR model loaded")
        samples_path = params["samples_path"]
        options = params.get("options", {}) or {}
        if isinstance(loaded, LoadedFasterWhisper):
            return _transcribe_faster_whisper(loaded, samples_path, options)
        if isinstance(loaded, LoadedTransformers):
            return _transcribe_transformers(loaded, samples_path, options)
        raise RuntimeError(f"unknown ASR loaded type: {type(loaded).__name__}")

    if method == "cleanup":
        loaded = _state["cleanup_loaded"]
        if loaded is None:
            raise RuntimeError("no cleanup model loaded")
        return _run_cleanup(loaded, params)

    if method == "unload":
        # No `engine` field → drop both slots. Otherwise drop just one.
        target = params.get("engine")
        if target in (None, "asr"):
            _state["asr_engine"] = None
            _state["asr_loaded"] = None
            _state["asr_model_key"] = None
        if target in (None, "cleanup"):
            _state["cleanup_loaded"] = None
            _state["cleanup_model_key"] = None
        _gc_torch()
        return {"ok": True}

    raise ValueError(f"unknown method: {method}")


def main() -> None:
    log("sidecar started")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"bad json: {e}: {line!r}")
            continue

        rid = req.get("id", 0)
        method = req.get("method", "")
        params = req.get("params") or {}

        try:
            result = handle(method, params)
            sys.stdout.write(json.dumps({"id": rid, "result": result}) + "\n")
        except Exception as exc:  # noqa: BLE001 — RPC boundary
            tb = traceback.format_exc()
            log(f"RPC {method} failed: {exc}\n{tb}")
            err = {
                "code": 1,
                "message": str(exc),
            }
            sys.stdout.write(json.dumps({"id": rid, "error": err}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
