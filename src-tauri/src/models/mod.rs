//! Local model registry + Hugging Face Hub client + download manager.
//!
//! Models live under <app_data>/models/<backend>/<model_id>:
//!   - whisper.cpp and faster-whisper:  a single weight file under a
//!     per-repo directory.
//!   - Transformers:                    a per-repo directory containing the
//!     subset of repo files needed by `from_pretrained()` (config, tokenizer,
//!     preprocessor, safetensors weights).
//!
//! Older builds wrote a `<repo>.repo` marker file for Transformers and
//! deferred the actual download to the Python sidecar's HF cache. Those
//! entries still load (see `read_repo_id_marker` in the transformers backend)
//! but new downloads are eager and self-contained.

mod download;
mod hub;
mod registry;

pub use download::{DownloadManager, DownloadProgress};
pub use hub::{
    filter_for_backend, pipeline_tag_for_backend, select_for_faster_whisper,
    select_for_transformers, HubClient, HubModel, HubModelFile,
};
pub use registry::{LocalModel, ModelRegistry};
