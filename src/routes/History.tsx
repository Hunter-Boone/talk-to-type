import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Copy, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Card } from "../components/Card";
import { Button, TextInput } from "../components/Field";
import { api } from "../lib/api";
import type { HistoryEntry } from "../lib/types";

export function HistoryRoute() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [filter, setFilter] = useState("");

  const refresh = () => {
    api
      .listHistory(100)
      .then(setEntries)
      .catch((e) => console.error("listHistory", e));
  };

  // Initial fetch + live refresh whenever a new transcription is saved
  // by the Rust side. The Rust `release_to_talk` flow emits
  // `history://updated` after each insert.
  useEffect(() => {
    refresh();
    let unlisten: UnlistenFn | undefined;
    listen("history://updated", () => refresh()).then((u) => (unlisten = u));
    return () => unlisten?.();
  }, []);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter((e) => e.text.toLowerCase().includes(q));
  }, [entries, filter]);

  const copy = (text: string) => {
    void navigator.clipboard.writeText(text);
  };

  return (
    <div className="max-w-3xl">
      <h1 className="mb-4 text-lg font-semibold">History</h1>

      <Card title="Recent transcriptions">
        <div className="flex items-center gap-2">
          <TextInput
            placeholder="Search…"
            value={filter}
            onValue={setFilter}
            className="flex-1"
          />
          <Button
            variant="danger"
            onClick={async () => {
              if (confirm("Clear all history? This can't be undone.")) {
                await api.clearHistory();
                refresh();
              }
            }}
          >
            Clear all
          </Button>
        </div>

        <div className="mt-3 space-y-2">
          {filtered.length === 0 && (
            <div className="text-xs text-ink-muted">
              {entries.length === 0
                ? "No transcriptions yet."
                : "No matches."}
            </div>
          )}
          {filtered.map((e) => (
            <div
              key={e.id}
              className="rounded-md border border-surface-strong/40 bg-surface-subtle/30 p-3"
            >
              <div className="mb-1 flex items-center justify-between text-xs text-ink-muted">
                <span>
                  {new Date(e.createdAt).toLocaleString()} · {e.backend}
                  {e.language ? ` · ${e.language}` : ""}
                </span>
                <span>
                  {e.audioSeconds.toFixed(1)} s audio · {e.durationMs} ms inf.
                </span>
              </div>
              <div className="whitespace-pre-wrap text-sm">{e.text}</div>
              <div className="mt-2 flex justify-end gap-1">
                <Button onClick={() => copy(e.text)} title="Copy">
                  <Copy size={14} />
                </Button>
                <Button
                  variant="danger"
                  onClick={async () => {
                    await api.deleteHistoryEntry(e.id);
                    refresh();
                  }}
                  title="Delete"
                >
                  <Trash2 size={14} />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
