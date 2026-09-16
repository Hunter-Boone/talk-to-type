//! History DB (rusqlite, bundled SQLite).

use std::path::Path;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryEntry {
    pub id: i64,
    pub created_at: DateTime<Utc>,
    pub text: String,
    pub backend: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub duration_ms: i64,
    pub audio_seconds: f32,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    backend       TEXT    NOT NULL,
    model         TEXT,
    language      TEXT,
    duration_ms   INTEGER NOT NULL,
    audio_seconds REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at DESC);
"#;

pub struct HistoryDb {
    conn: Mutex<Connection>,
}

impl HistoryDb {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn insert(&self, entry: &NewHistoryEntry) -> Result<i64> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO history (created_at, text, backend, model, language, duration_ms, audio_seconds)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                now,
                entry.text,
                entry.backend,
                entry.model,
                entry.language,
                entry.duration_ms,
                entry.audio_seconds,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list(&self, limit: u32) -> Result<Vec<HistoryEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, text, backend, model, language, duration_ms, audio_seconds
             FROM history ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit], |row| {
            let created_at: String = row.get(1)?;
            Ok(HistoryEntry {
                id: row.get(0)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                text: row.get(2)?,
                backend: row.get(3)?,
                model: row.get(4)?,
                language: row.get(5)?,
                duration_ms: row.get(6)?,
                audio_seconds: row.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn delete(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM history WHERE id = ?1", [id])?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM history", [])?;
        Ok(())
    }

    /// Trim history to at most `n` rows (keeps the newest).
    pub fn trim(&self, n: u32) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM history WHERE id NOT IN (
                SELECT id FROM history ORDER BY created_at DESC LIMIT ?1
            )",
            [n],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NewHistoryEntry {
    pub text: String,
    pub backend: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub duration_ms: i64,
    pub audio_seconds: f32,
}
