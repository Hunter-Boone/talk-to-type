use serde::Serialize;
use thiserror::Error;

/// All fallible operations in the host bubble up through this error.
/// `Serialize` makes it usable as a `tauri::command` return type.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("audio device error: {0}")]
    Audio(String),

    #[error("hotkey error: {0}")]
    Hotkey(String),

    #[error("backend not initialised")]
    BackendNotReady,

    #[error("backend error: {0}")]
    Backend(String),

    #[error("sidecar error: {0}")]
    Sidecar(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("paste error: {0}")]
    Paste(String),

    #[error("settings error: {0}")]
    Settings(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError::Http(e.to_string())
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(e: rusqlite::Error) -> Self {
        AppError::Database(e.to_string())
    }
}

// Tauri requires command errors to be Serialize. Render as a string.
// `std::result::Result` is required here because our `Result` alias takes
// only one generic parameter.
impl Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
