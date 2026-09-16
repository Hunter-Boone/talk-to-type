//! Hugging Face Hub API client.
//!
//! Two endpoints we hit:
//!   1. /api/models?filter=automatic-speech-recognition  → list ASR models
//!   2. /api/models/<repo>/tree/main                     → file listing
//!      (we need 2 because the /models endpoint doesn't return file names).
//!
//! No auth — we only read public metadata.

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::backends::BackendKind;
use crate::error::{AppError, Result};

const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HubModel {
    #[serde(rename = "id")]
    pub repo_id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub likes: Option<u64>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HubModelFile {
    pub path: String,
    /// "file" or "directory". Defaults to "file" when missing.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<HubLfs>,
}

impl HubModelFile {
    pub fn is_file(&self) -> bool {
        self.kind.as_deref().map(|k| k == "file").unwrap_or(true)
    }

    /// Best-effort byte size, preferring the LFS pointer's size when present
    /// (HF reports the pointer file's small size in `size` for LFS objects).
    pub fn effective_size(&self) -> Option<u64> {
        self.lfs.as_ref().and_then(|l| l.size).or(self.size)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HubLfs {
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub oid: Option<String>,
}

pub struct HubClient {
    /// Short-timeout client for metadata calls (search, tree).
    http: reqwest::Client,
    /// Long-lived client for streaming file bodies. No overall timeout, only
    /// a connect timeout and a per-read timeout — so multi-GB LFS downloads
    /// don't get killed mid-stream while genuinely-stalled sockets still
    /// fail fast.
    download_http: reqwest::Client,
    token: Option<String>,
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new(None)
    }
}

impl HubClient {
    /// Build a client. Pass `Some(token)` to attach an `Authorization: Bearer`
    /// header to every request — required for gated/private repos.
    pub fn new(token: Option<String>) -> Self {
        let ua = concat!("talk-to-type/", env!("CARGO_PKG_VERSION"));
        let http = reqwest::Client::builder()
            .user_agent(ua)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("build metadata reqwest client");
        let download_http = reqwest::Client::builder()
            .user_agent(ua)
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("build download reqwest client");
        Self {
            http,
            download_http,
            token: token.filter(|t| !t.trim().is_empty()),
        }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let mut rb = self.http.get(url);
        if let Some(token) = &self.token {
            rb = rb.bearer_auth(token);
        }
        rb
    }

    /// Search a HF pipeline tag. `query` is matched against the repo id.
    /// Returns at most `limit` results, sorted by downloads desc.
    ///
    /// `pipeline` should be a HF pipeline tag like
    /// `automatic-speech-recognition` or `text-generation`.
    pub async fn search(
        &self,
        query: &str,
        limit: u32,
        pipeline: &str,
    ) -> Result<Vec<HubModel>> {
        let url = format!(
            "{HF_BASE}/api/models?filter={}&search={}&sort=downloads&direction=-1&limit={}",
            urlencoding(pipeline),
            urlencoding(query),
            limit,
        );
        debug!(url, "GET");
        let resp = self.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(AppError::Http(format!(
                "HF search returned {}",
                resp.status()
            )));
        }
        let models: Vec<HubModel> = resp.json().await?;
        Ok(models)
    }

    /// List files in a repo's main revision.
    pub async fn list_files(&self, repo_id: &str) -> Result<Vec<HubModelFile>> {
        let url = format!("{HF_BASE}/api/models/{repo_id}/tree/main");
        debug!(url, "GET");
        let resp = self.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(AppError::Http(format!(
                "HF tree returned {} for {}",
                resp.status(),
                repo_id
            )));
        }
        let files: Vec<HubModelFile> = resp.json().await?;
        Ok(files)
    }

    /// Build the LFS-resolved URL for a given file.
    pub fn resolve_url(&self, repo_id: &str, path: &str) -> String {
        format!("{HF_BASE}/{repo_id}/resolve/main/{path}")
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Long-running client to use for streaming model file bodies.
    pub fn download_http(&self) -> &reqwest::Client {
        &self.download_http
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}

/// HF pipeline tag to query when searching for models compatible with this
/// backend. Used as the `filter=` query-string parameter on `/api/models`.
pub fn pipeline_tag_for_backend(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::CleanupTransformers => "text-generation",
        // The three ASR backends all live on automatic-speech-recognition;
        // the per-backend file filter narrows further (gguf vs ct2 vs hf).
        _ => "automatic-speech-recognition",
    }
}

/// Decide whether a model is compatible with a given backend, given the
/// repo's tags + (optionally) its file list.
///
/// - whisper.cpp: must contain a .gguf or ggml-*.bin file
/// - faster-whisper: must contain a model.bin (CT2 format) at root or in subdir
/// - transformers: any ASR model is acceptable
/// - cleanup-transformers: any text-generation model is acceptable
pub fn filter_for_backend(
    backend: BackendKind,
    model: &HubModel,
    files: Option<&[HubModelFile]>,
) -> bool {
    match backend {
        BackendKind::CleanupTransformers => {
            model.pipeline_tag.as_deref() == Some("text-generation")
                || model.tags.iter().any(|t| t == "text-generation")
        }
        BackendKind::Transformers => {
            model.pipeline_tag.as_deref() == Some("automatic-speech-recognition")
                || model.tags.iter().any(|t| t == "automatic-speech-recognition")
        }
        BackendKind::WhisperCpp => match files {
            Some(files) => files.iter().any(|f| {
                let p = f.path.to_ascii_lowercase();
                p.ends_with(".gguf") || (p.starts_with("ggml-") && p.ends_with(".bin"))
            }),
            // Heuristic on metadata alone: known repos that ship gguf/ggml.
            None => {
                let id = model.repo_id.to_ascii_lowercase();
                id.contains("ggml") || id.contains("gguf") || id.contains("whisper.cpp")
            }
        },
        BackendKind::FasterWhisper => match files {
            Some(files) => {
                files
                    .iter()
                    .any(|f| f.path.to_ascii_lowercase().ends_with("/model.bin"))
                    || files
                        .iter()
                        .any(|f| f.path.eq_ignore_ascii_case("model.bin"))
            }
            None => {
                let id = model.repo_id.to_ascii_lowercase();
                id.contains("ct2") || id.contains("faster-whisper") || id.contains("/systran/")
            }
        },
    }
}

/// Select the subset of repo files to download for a HuggingFace Transformers
/// model load. We want the config/tokenizer/preprocessor JSONs plus a single
/// weights format (safetensors preferred over pytorch). Anything specific to
/// other runtimes (ONNX, TFLite, Flax, TF, CoreML, OpenVINO) is skipped.
pub fn select_for_transformers(files: &[HubModelFile]) -> Vec<&HubModelFile> {
    let has_safetensors = files
        .iter()
        .any(|f| f.is_file() && is_safetensors_weight(&f.path));

    files
        .iter()
        .filter(|f| f.is_file() && include_for_transformers(&f.path, has_safetensors))
        .collect()
}

/// Select the file set a CTranslate2 faster-whisper load needs.
/// `WhisperModel(<dir>)` reads model.bin plus config.json, tokenizer.json /
/// vocabulary.* and (when present) preprocessor_config.json — model.bin
/// alone is not loadable.
pub fn select_for_faster_whisper(files: &[HubModelFile]) -> Vec<&HubModelFile> {
    files
        .iter()
        .filter(|f| {
            if !f.is_file() {
                return false;
            }
            // CT2 repos are flat; only take root-level files.
            if f.path.contains('/') {
                return false;
            }
            matches!(
                f.path.to_ascii_lowercase().as_str(),
                "model.bin"
                    | "config.json"
                    | "tokenizer.json"
                    | "vocabulary.txt"
                    | "vocabulary.json"
                    | "preprocessor_config.json"
            )
        })
        .collect()
}

fn is_safetensors_weight(path: &str) -> bool {
    let name = base_name(path);
    name == "model.safetensors"
        || (name.starts_with("model-") && name.ends_with(".safetensors"))
        || name == "model.safetensors.index.json"
}

fn is_pytorch_weight(path: &str) -> bool {
    let name = base_name(path);
    name == "pytorch_model.bin"
        || (name.starts_with("pytorch_model-") && name.ends_with(".bin"))
        || name == "pytorch_model.bin.index.json"
}

fn include_for_transformers(path: &str, has_safetensors: bool) -> bool {
    let lower = path.to_ascii_lowercase();

    // Skip framework-specific subtrees and runtime variants.
    for skip_dir in [
        "onnx/", "tflite/", "coreml/", "openvino/", "ggml/", "gguf/",
    ] {
        if lower.contains(skip_dir) {
            return false;
        }
    }

    let name = base_name(&lower);

    // Skip docs and metadata that aren't needed for inference.
    if matches!(
        name,
        ".gitattributes" | "readme.md" | "license" | "license.md" | "license.txt"
    ) {
        return false;
    }
    if name.ends_with(".md") {
        return false;
    }

    // Skip non-PyTorch/safetensors weights.
    if name.starts_with("flax_model") || name.starts_with("tf_model") {
        return false;
    }
    if name.ends_with(".h5")
        || name.ends_with(".tflite")
        || name.ends_with(".onnx")
        || name.ends_with(".msgpack")
        || name == "rust_model.ot"
        || name.ends_with(".onnx_data")
    {
        return false;
    }

    // Skip half-precision duplicates and other variants.
    if name.contains(".fp16.") || name.contains(".bf16.") {
        return false;
    }

    // If a safetensors weight set is available, drop the redundant pytorch
    // variant — they're typically multi-GB each.
    if has_safetensors && is_pytorch_weight(&lower) {
        return false;
    }

    true
}

fn base_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn urlencoding(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(id: &str, tags: &[&str], pipeline: Option<&str>) -> HubModel {
        HubModel {
            repo_id: id.to_string(),
            author: None,
            downloads: None,
            likes: None,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            last_modified: None,
            pipeline_tag: pipeline.map(|s| s.to_string()),
        }
    }

    fn file(path: &str) -> HubModelFile {
        HubModelFile {
            path: path.to_string(),
            kind: Some("file".into()),
            size: None,
            lfs: None,
        }
    }

    #[test]
    fn transformers_accepts_asr_pipeline() {
        let m = model_with("openai/whisper-base", &[], Some("automatic-speech-recognition"));
        assert!(filter_for_backend(BackendKind::Transformers, &m, None));
    }

    #[test]
    fn transformers_rejects_non_asr() {
        let m = model_with("bert-base-uncased", &["fill-mask"], Some("fill-mask"));
        assert!(!filter_for_backend(BackendKind::Transformers, &m, None));
    }

    #[test]
    fn whisper_cpp_accepts_gguf_file() {
        let m = model_with("ggerganov/whisper.cpp", &[], None);
        let files = vec![file("ggml-base.en.bin"), file("README.md")];
        assert!(filter_for_backend(BackendKind::WhisperCpp, &m, Some(&files)));
    }

    #[test]
    fn whisper_cpp_rejects_pytorch_only() {
        let m = model_with("openai/whisper-base", &[], None);
        let files = vec![file("pytorch_model.bin"), file("config.json")];
        assert!(!filter_for_backend(BackendKind::WhisperCpp, &m, Some(&files)));
    }

    #[test]
    fn faster_whisper_accepts_ct2_model_bin() {
        let m = model_with("Systran/faster-whisper-medium", &[], None);
        let files = vec![file("model.bin"), file("config.json"), file("tokenizer.json")];
        assert!(filter_for_backend(BackendKind::FasterWhisper, &m, Some(&files)));
    }

    #[test]
    fn faster_whisper_id_heuristic_no_files() {
        let m = model_with("Systran/faster-whisper-tiny", &[], None);
        assert!(filter_for_backend(BackendKind::FasterWhisper, &m, None));
    }

    #[test]
    fn transformers_selector_prefers_safetensors() {
        let files = vec![
            file("config.json"),
            file("tokenizer.json"),
            file("preprocessor_config.json"),
            file("model.safetensors"),
            file("pytorch_model.bin"),
            file("README.md"),
            file("flax_model.msgpack"),
            file("tf_model.h5"),
            file("onnx/model.onnx"),
            file(".gitattributes"),
        ];
        let picked: Vec<&str> = select_for_transformers(&files)
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert!(picked.contains(&"config.json"));
        assert!(picked.contains(&"tokenizer.json"));
        assert!(picked.contains(&"preprocessor_config.json"));
        assert!(picked.contains(&"model.safetensors"));
        assert!(!picked.contains(&"pytorch_model.bin"));
        assert!(!picked.contains(&"README.md"));
        assert!(!picked.contains(&"flax_model.msgpack"));
        assert!(!picked.contains(&"tf_model.h5"));
        assert!(!picked.contains(&"onnx/model.onnx"));
        assert!(!picked.contains(&".gitattributes"));
    }

    #[test]
    fn transformers_selector_keeps_pytorch_when_no_safetensors() {
        let files = vec![
            file("config.json"),
            file("tokenizer.json"),
            file("pytorch_model.bin"),
        ];
        let picked: Vec<&str> = select_for_transformers(&files)
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert!(picked.contains(&"pytorch_model.bin"));
    }

    #[test]
    fn transformers_selector_handles_sharded_safetensors() {
        let files = vec![
            file("config.json"),
            file("model.safetensors.index.json"),
            file("model-00001-of-00002.safetensors"),
            file("model-00002-of-00002.safetensors"),
            file("pytorch_model.bin.index.json"),
            file("pytorch_model-00001-of-00002.bin"),
            file("pytorch_model-00002-of-00002.bin"),
        ];
        let picked: Vec<&str> = select_for_transformers(&files)
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert!(picked.contains(&"model.safetensors.index.json"));
        assert!(picked.contains(&"model-00001-of-00002.safetensors"));
        assert!(picked.contains(&"model-00002-of-00002.safetensors"));
        assert!(!picked.contains(&"pytorch_model.bin.index.json"));
        assert!(!picked.contains(&"pytorch_model-00001-of-00002.bin"));
    }

    #[test]
    fn transformers_selector_skips_directories() {
        let mut dir = file("subfolder");
        dir.kind = Some("directory".into());
        let files = vec![file("config.json"), dir];
        let picked: Vec<&str> = select_for_transformers(&files)
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert_eq!(picked, vec!["config.json"]);
    }
}
