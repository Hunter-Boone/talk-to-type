use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct RpcRequest<P: Serialize> {
    pub id: u64,
    pub method: &'static str,
    pub params: P,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RpcResponse<R> {
    pub id: u64,
    #[serde(default)]
    pub result: Option<R>,
    #[serde(default)]
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

// --- Concrete request/response payloads ------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    FasterWhisper,
    Transformers,
    /// HuggingFace causal-LM loaded into the sidecar's cleanup slot.
    /// Independent of the ASR slot — both can be resident at once.
    Cleanup,
}

#[derive(Debug, Serialize)]
pub struct LoadModelParams {
    pub engine: Engine,
    /// For faster-whisper: a directory containing the CT2 model.
    /// For transformers: a HF model id OR a local path.
    pub model: String,
    pub options: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelResult {
    #[serde(default)]
    pub ok: bool,
}

#[derive(Debug, Serialize)]
pub struct TranscribeParams {
    /// Path to a temporary WAV file written by the host. We don't push the
    /// raw f32 buffer over stdio — large utterances would balloon the JSON
    /// frame; a 16-bit WAV at 16 kHz is ~32 KB/s and roundtrips cleanly.
    pub samples_path: String,
    pub options: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct TranscribeResult {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration_ms: u64,
    /// Per-word timing if the request set `options.word_timestamps =
    /// true` AND the engine produced any. Empty / missing otherwise.
    #[serde(default)]
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Deserialize)]
pub struct WordTiming {
    /// The word as emitted by the model. For Whisper-family models
    /// this includes the leading space when the word is mid-utterance
    /// (e.g. " hello"); the host normalises whitespace at concat time.
    pub text: String,
    /// Word start time, in seconds, relative to the start of the
    /// audio buffer the sidecar was given.
    pub start: f32,
    /// Word end time, in seconds, relative to the start of the audio
    /// buffer the sidecar was given. Always >= `start`.
    pub end: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnloadTarget {
    Asr,
    Cleanup,
}

#[derive(Debug, Serialize, Default)]
pub struct UnloadParams {
    /// `None` → drop both slots. `Some(...)` → drop just that one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<UnloadTarget>,
}

#[derive(Debug, Deserialize)]
pub struct UnloadResult {
    #[serde(default)]
    pub ok: bool,
}

#[derive(Debug, Serialize)]
pub struct CleanupParams {
    pub system_prompt: String,
    pub raw_text: String,
    pub options: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct CleanupResult {
    pub text: String,
    #[serde(default)]
    pub duration_ms: u64,
}
