import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Download, Trash2, Loader2, Search, AlertCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { Card, Row } from "../components/Card";
import { Button, Select, TextInput } from "../components/Field";
import { useSettings } from "../hooks/useSettings";
import { api } from "../lib/api";
import type {
  BackendKind,
  DownloadProgress,
  HubSearchResult,
  LocalModel,
} from "../lib/types";

const backendOptions: { value: BackendKind; label: string }[] = [
  { value: "whisper-cpp", label: "whisper.cpp" },
  { value: "faster-whisper", label: "faster-whisper" },
  { value: "transformers", label: "Transformers" },
  { value: "cleanup-transformers", label: "Transformers (cleanup)" },
];

function bytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function ModelsRoute() {
  const { settings, patch } = useSettings();
  const [backend, setBackend] = useState<BackendKind>("whisper-cpp");
  const [query, setQuery] = useState("whisper");
  const [results, setResults] = useState<HubSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [tokenDraft, setTokenDraft] = useState("");

  useEffect(() => {
    if (settings) setTokenDraft(settings.hfToken ?? "");
  }, [settings?.hfToken]);

  // Sync default backend with settings on first mount.
  useEffect(() => {
    if (settings) setBackend(settings.activeBackend);
  }, [settings?.activeBackend]);

  useEffect(() => {
    refreshLocal();
  }, [backend]);

  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    listen<DownloadProgress>("download://progress", (e) => {
      const p = e.payload;
      setProgress((prev) => ({ ...prev, [p.repoId + "/" + p.file]: p }));
      if (p.error) {
        setDownloadError(`${p.repoId}/${p.file}: ${p.error}`);
      } else if (p.finished) {
        refreshLocal();
      }
    }).then((u) => (unlisten = u));
    return () => unlisten?.();
  }, []);

  const refreshLocal = async () => {
    try {
      setLocal(await api.listLocalModels(backend));
    } catch (e) {
      console.error("listLocalModels", e);
    }
  };

  const search = async () => {
    setSearching(true);
    setSearchError(null);
    try {
      setResults(await api.searchHubModels(backend, query));
    } catch (e) {
      setSearchError(String(e));
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  const download = async (repoId: string) => {
    setDownloadError(null);
    setDownloading(repoId);
    try {
      if (backend === "whisper-cpp") {
        // Single-file backend. Heuristic file pick; users with more
        // specific needs can edit registry.json directly.
        await api.downloadModel({
          backend,
          repoId,
          file: guessWhisperCppFile(repoId),
          displayName: repoId,
        });
      } else {
        // transformers / cleanup-transformers / faster-whisper all need a
        // multi-file directory (config + tokenizer + weights); the Rust
        // side picks the right subset per backend.
        await api.downloadModel({
          backend,
          repoId,
          file: null,
          displayName: repoId,
        });
      }
      refreshLocal();
    } catch (e) {
      setDownloadError(`${repoId}: ${String(e)}`);
    } finally {
      setDownloading(null);
    }
  };

  const saveToken = () => {
    const trimmed = tokenDraft.trim();
    patch({ hfToken: trimmed === "" ? null : trimmed });
  };

  const remove = async (id: string) => {
    await api.deleteModel(id);
    refreshLocal();
  };

  return (
    <div className="max-w-3xl">
      <h1 className="mb-4 text-lg font-semibold">Models</h1>

      <Card title="Browse Hugging Face Hub">
        <div className="flex items-end gap-2">
          <div>
            <div className="mb-1 text-xs text-ink-muted">Backend</div>
            <Select
              value={backend}
              onValue={(v) => setBackend(v as BackendKind)}
            >
              {backendOptions.map((b) => (
                <option key={b.value} value={b.value}>
                  {b.label}
                </option>
              ))}
            </Select>
          </div>
          <div className="flex-1">
            <div className="mb-1 text-xs text-ink-muted">Search</div>
            <TextInput
              value={query}
              onValue={setQuery}
              placeholder="whisper, parakeet, distil…"
              className="w-full"
              onKeyDown={(e) => {
                if (e.key === "Enter") void search();
              }}
            />
          </div>
          <Button variant="primary" onClick={search} disabled={searching}>
            {searching ? (
              <Loader2 className="animate-spin" size={14} />
            ) : (
              <Search size={14} />
            )}
          </Button>
        </div>

        {searchError && (
          <div className="mt-2 text-xs text-red-500">{searchError}</div>
        )}

        <div className="mt-3 max-h-72 overflow-y-auto rounded-md border border-surface-strong/40">
          {results.length === 0 && !searching && (
            <div className="p-3 text-xs text-ink-muted">
              {searchError ? "" : "No results yet — try a search."}
            </div>
          )}
          {results.map((r) => (
            <div
              key={r.repoId}
              className="flex items-center justify-between border-b border-surface-strong/40 p-2 last:border-b-0"
            >
              <div className="min-w-0">
                <div className="truncate text-sm">{r.repoId}</div>
                <div className="text-xs text-ink-muted">
                  {r.downloads ? `${r.downloads.toLocaleString()} downloads` : ""}
                  {r.likes ? ` · ${r.likes} likes` : ""}
                </div>
              </div>
              <Button
                variant="secondary"
                onClick={() => download(r.repoId)}
                title="Download"
                disabled={downloading === r.repoId}
              >
                {downloading === r.repoId ? (
                  <Loader2 className="animate-spin" size={14} />
                ) : (
                  <Download size={14} />
                )}
              </Button>
            </div>
          ))}
        </div>

        {downloadError && (
          <div className="mt-3 flex items-start gap-2 rounded-md border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-300">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <div className="break-words">{downloadError}</div>
          </div>
        )}
      </Card>

      <Card
        title="HuggingFace token"
        description="Optional. Required for downloading gated or private repos. Create one at huggingface.co → Settings → Access Tokens (read access is enough)."
      >
        <Row label="Token">
          <TextInput
            value={tokenDraft}
            onValue={setTokenDraft}
            placeholder="hf_..."
            type="password"
            className="flex-1"
          />
          <Button variant="primary" onClick={saveToken}>
            Save
          </Button>
        </Row>
      </Card>

      <Card title="Downloaded">
        {local.length === 0 ? (
          <div className="text-xs text-ink-muted">
            Nothing yet. Pick a model above.
          </div>
        ) : (
          local.map((m) => (
            <div
              key={m.id}
              className="flex items-center justify-between rounded-md bg-surface-subtle/50 p-2"
            >
              <div className="min-w-0">
                <div className="truncate text-sm">{m.name}</div>
                <div className="text-xs text-ink-muted">
                  {bytes(m.size)} · {m.path}
                </div>
              </div>
              <Button variant="danger" onClick={() => remove(m.id)}>
                <Trash2 size={14} />
              </Button>
            </div>
          ))
        )}
      </Card>

      {Object.values(progress).filter((p) => !p.finished).length > 0 && (
        <Card title="In progress">
          {Object.values(progress)
            .filter((p) => !p.finished)
            .map((p) => {
              const pct =
                p.total && p.total > 0
                  ? Math.round((p.downloaded / p.total) * 100)
                  : null;
              return (
                <div key={p.jobId} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="truncate">
                      {p.repoId} / {p.file}
                    </span>
                    <span className="text-ink-muted">
                      {pct !== null ? `${pct}%` : bytes(p.downloaded)}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded bg-surface-strong">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{ width: `${pct ?? 0}%` }}
                    />
                  </div>
                </div>
              );
            })}
        </Card>
      )}
    </div>
  );
}

// --- Heuristic file picks --------------------------------------------------
// In a future iteration we'd surface a real file picker (calls
// /api/models/<repo>/tree/main and lets the user choose). For v0 we map a
// few well-known repos to their canonical artifact and bail otherwise.

function guessWhisperCppFile(repoId: string): string {
  const lower = repoId.toLowerCase();
  if (lower.includes("ggerganov")) return "ggml-base.en.bin";
  if (lower.includes("distil-whisper")) return "ggml-distil-large-v3.bin";
  return "model.gguf";
}
