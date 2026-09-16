//! Python sidecar process supervisor + JSON-RPC client.
//!
//! Wire format (newline-delimited JSON):
//!
//!   request:  { "id": <u64>, "method": "<name>", "params": <obj> }
//!   response: { "id": <u64>, "result": <obj> }                   on success
//!             { "id": <u64>, "error":  { "code": <i32>, "message": <str> } }
//!
//! Methods (see sidecar/main.py for the canonical list):
//!   - ping                                          → {"pong": true}
//!   - load_model { engine, model, options }         → {"ok": true}
//!   - transcribe { samples_path|samples, options }  → {"text": "...", ...}
//!   - unload                                        → {"ok": true}
//!
//! The supervisor lazily spawns the sidecar on first use and respawns it on
//! crash with capped exponential backoff.

pub mod protocol;
mod supervisor;

pub use protocol::{RpcError, RpcRequest, RpcResponse};
pub use supervisor::SidecarSupervisor;
