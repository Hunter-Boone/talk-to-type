//! Local model registry. Persists a JSON manifest under <models>/registry.json.
//!
//! For whisper.cpp:        each entry points at a single .bin/.gguf file.
//! For faster-whisper:     each entry points at a directory containing model.bin.
//! For transformers:       each entry points at a *.repo marker file (the HF id).

use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::backends::BackendKind;
use crate::error::{AppError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalModel {
    pub id: String,
    pub backend: BackendKind,
    /// Display name (e.g., "Whisper Base (English)").
    pub name: String,
    /// Local file or directory.
    pub path: PathBuf,
    /// Source repo on HF, if applicable.
    pub repo_id: Option<String>,
    /// Bytes on disk.
    pub size: u64,
    pub languages: Vec<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub installed_at: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct Manifest {
    #[serde(default)]
    models: Vec<LocalModel>,
}

pub struct ModelRegistry {
    root: PathBuf,
    manifest_path: PathBuf,
    inner: Mutex<Manifest>,
}

impl ModelRegistry {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        for kind in [
            "whisper-cpp",
            "faster-whisper",
            "transformers",
            "cleanup-transformers",
        ] {
            fs::create_dir_all(root.join(kind))?;
        }
        let manifest_path = root.join("registry.json");
        let manifest = if manifest_path.exists() {
            let text = fs::read_to_string(&manifest_path)?;
            serde_json::from_str::<Manifest>(&text).unwrap_or_default()
        } else {
            Manifest::default()
        };
        Ok(Self {
            root,
            manifest_path,
            inner: Mutex::new(manifest),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir_for(&self, backend: BackendKind) -> PathBuf {
        match backend {
            BackendKind::WhisperCpp => self.root.join("whisper-cpp"),
            BackendKind::FasterWhisper => self.root.join("faster-whisper"),
            BackendKind::Transformers => self.root.join("transformers"),
            BackendKind::CleanupTransformers => self.root.join("cleanup-transformers"),
        }
    }

    pub fn list(&self) -> Vec<LocalModel> {
        self.inner.lock().models.clone()
    }

    pub fn list_for(&self, backend: BackendKind) -> Vec<LocalModel> {
        self.inner
            .lock()
            .models
            .iter()
            .filter(|m| m.backend == backend)
            .cloned()
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<LocalModel> {
        self.inner
            .lock()
            .models
            .iter()
            .find(|m| m.id == id)
            .cloned()
    }

    pub fn upsert(&self, model: LocalModel) -> Result<()> {
        {
            let mut m = self.inner.lock();
            m.models.retain(|x| x.id != model.id);
            m.models.push(model);
        }
        self.persist()
    }

    pub fn remove(&self, id: &str) -> Result<Option<LocalModel>> {
        let removed = {
            let mut m = self.inner.lock();
            let idx = m.models.iter().position(|x| x.id == id);
            idx.map(|i| m.models.remove(i))
        };
        if let Some(model) = &removed {
            // Remove the on-disk artefact too.
            let _ = if model.path.is_dir() {
                fs::remove_dir_all(&model.path)
            } else {
                fs::remove_file(&model.path)
            };
        }
        self.persist()?;
        Ok(removed)
    }

    fn persist(&self) -> Result<()> {
        let m = self.inner.lock().clone();
        let bytes = serde_json::to_vec_pretty(&m)?;
        fs::write(&self.manifest_path, bytes)
            .map_err(|e| AppError::Model(format!("manifest write: {e}")))?;
        Ok(())
    }
}
