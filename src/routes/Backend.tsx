import { Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Card, Row } from "../components/Card";
import { Button, NumberInput, Select, TextArea, TextInput, Toggle } from "../components/Field";
import { useSettings } from "../hooks/useSettings";
import { api } from "../lib/api";
import type { BackendDescriptor, BackendKind, LocalModel } from "../lib/types";

export function BackendRoute() {
  const { settings, patch } = useSettings();
  const [backends, setBackends] = useState<BackendDescriptor[]>([]);
  const [localModels, setLocalModels] = useState<LocalModel[]>([]);
  const [cleanupModels, setCleanupModels] = useState<LocalModel[]>([]);
  const [preloading, setPreloading] = useState(false);
  const [preloadStatus, setPreloadStatus] = useState<
    | { kind: "idle" }
    | { kind: "ok" }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  useEffect(() => {
    api.listBackends().then(setBackends);
  }, []);

  useEffect(() => {
    if (!settings) return;
    api
      .listLocalModels(settings.activeBackend)
      .then(setLocalModels)
      .catch(() => setLocalModels([]));
  }, [settings?.activeBackend]);

  // Refresh the cleanup-transformers model list whenever the user opens
  // this route, so newly-downloaded LLMs show up without a full reload.
  useEffect(() => {
    api
      .listLocalModels("cleanup-transformers")
      .then(setCleanupModels)
      .catch(() => setCleanupModels([]));
  }, [settings?.postProcessing.provider]);

  // Any change to the configured cleanup model or device invalidates the
  // "Loaded" indicator — the next preload click should pay the load cost
  // again rather than misleadingly stay green.
  useEffect(() => {
    setPreloadStatus({ kind: "idle" });
  }, [
    settings?.postProcessing.transformersModelId,
    settings?.postProcessing.transformersDevice,
  ]);

  const preloadCleanup = async () => {
    setPreloading(true);
    setPreloadStatus({ kind: "idle" });
    try {
      await api.preloadCleanupModel();
      setPreloadStatus({ kind: "ok" });
    } catch (e) {
      setPreloadStatus({ kind: "error", message: String(e) });
    } finally {
      setPreloading(false);
    }
  };

  const optionsForBackend = useMemo(() => {
    if (!settings) return null;
    switch (settings.activeBackend) {
      case "whisper-cpp":
        return (
          <>
            <Row
              label="CPU threads"
              description="Default is half your logical cores. More isn't always faster."
            >
              <NumberInput
                value={settings.backendOptions.threads ?? 0}
                min={0}
                max={64}
                onValue={(threads) =>
                  patch({
                    backendOptions: {
                      ...settings.backendOptions,
                      threads: threads || undefined,
                    },
                  })
                }
              />
            </Row>
          </>
        );
      case "faster-whisper":
        return (
          <>
            <Row
              label="Compute type"
              description="int8 for CPU, float16 / int8_float16 for GPU."
            >
              <Select
                value={settings.backendOptions.computeType ?? "int8"}
                onValue={(v) =>
                  patch({
                    backendOptions: {
                      ...settings.backendOptions,
                      computeType: v,
                    },
                  })
                }
              >
                {["int8", "int8_float16", "float16", "float32"].map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
            </Row>
            <Row label="Device">
              <Select
                value={settings.backendOptions.device ?? "auto"}
                onValue={(v) =>
                  patch({
                    backendOptions: { ...settings.backendOptions, device: v },
                  })
                }
              >
                {["auto", "cpu", "cuda"].map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </Select>
            </Row>
          </>
        );
      case "transformers":
        return (
          <Row label="Torch device">
            <Select
              value={settings.backendOptions.device ?? "cpu"}
              onValue={(v) =>
                patch({
                  backendOptions: { ...settings.backendOptions, device: v },
                })
              }
            >
              {["cpu", "cuda", "mps"].map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </Select>
          </Row>
        );
    }
  }, [settings, patch]);

  if (!settings) return <div className="text-ink-muted">Loading…</div>;

  const switchTo = async (kind: BackendKind) => {
    await api.switchBackend(kind);
    patch({ activeBackend: kind });
  };

  return (
    <div className="max-w-2xl">
      <h1 className="mb-4 text-lg font-semibold">Backend</h1>

      <Card title="Engine">
        <div className="grid grid-cols-1 gap-2">
          {backends.map((b) => (
            <label
              key={b.kind}
              className={[
                "flex cursor-pointer items-start gap-3 rounded-lg border border-surface-strong/40 p-3",
                settings.activeBackend === b.kind
                  ? "bg-accent/10 border-accent/60"
                  : "bg-surface-subtle/30 hover:bg-surface-subtle/60",
              ].join(" ")}
            >
              <input
                type="radio"
                checked={settings.activeBackend === b.kind}
                onChange={() => switchTo(b.kind)}
                className="mt-1"
              />
              <div>
                <div className="text-sm font-medium">{b.label}</div>
                <div className="text-xs text-ink-muted">{b.description}</div>
              </div>
            </label>
          ))}
        </div>
      </Card>

      <Card
        title="Active model"
        description="Models you've downloaded show up here. Use the Models tab to add more."
      >
        <Row label="Model">
          <Select
            value={settings.activeModelId ?? ""}
            onValue={async (v) => {
              if (!v) {
                patch({ activeModelId: null });
                return;
              }
              try {
                await api.selectModel(v);
                patch({ activeModelId: v });
              } catch (e) {
                console.error("selectModel", e);
                alert(`Failed to load model: ${e}`);
              }
            }}
          >
            <option value="">— Select —</option>
            {localModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </Select>
        </Row>
        <Row label="Language hint">
          <TextInput
            value={settings.backendOptions.language ?? ""}
            onValue={(v) =>
              patch({
                backendOptions: {
                  ...settings.backendOptions,
                  language: v || undefined,
                },
              })
            }
            placeholder="auto-detect"
            className="w-32"
          />
        </Row>
      </Card>

      <Card title="Tuning">{optionsForBackend}</Card>

      <Card
        title="Post-processing"
        description="Optional pass through an LLM to clean up punctuation and remove filler words."
      >
        <Row label="Enabled">
          <Toggle
            checked={settings.postProcessing.enabled}
            onChange={(v) =>
              patch({
                postProcessing: { ...settings.postProcessing, enabled: v },
              })
            }
          />
        </Row>
        <Row
          label="Provider"
          description="Cursor CLI runs through your signed-in Cursor account; Ollama hits a local model server; Transformers runs an HF causal-LM inside the same Python sidecar that does ASR."
        >
          <Select
            value={settings.postProcessing.provider}
            onValue={(v) =>
              patch({
                postProcessing: {
                  ...settings.postProcessing,
                  provider: v as "cursor-cli" | "ollama" | "transformers",
                },
              })
            }
          >
            <option value="cursor-cli">Cursor CLI</option>
            <option value="ollama">Ollama</option>
            <option value="transformers">Transformers (local)</option>
          </Select>
        </Row>

        {settings.postProcessing.provider === "cursor-cli" && (
          <>
            <Row
              label="Model"
              description="Passed to `cursor-agent --model`. `auto` lets Cursor pick."
            >
              <Select
                value={settings.postProcessing.cursorModel || "auto"}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      cursorModel: v,
                    },
                  })
                }
              >
                {["auto", "composer-1", "gpt-5.2", "claude-4.5-sonnet"].map(
                  (m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  )
                )}
              </Select>
            </Row>
            <Row
              label="Binary path"
              description="Leave blank to use `cursor-agent` from PATH. Override if it isn't found."
            >
              <TextInput
                value={settings.postProcessing.cursorCliPath ?? ""}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      cursorCliPath: v || null,
                    },
                  })
                }
                placeholder="cursor-agent"
                className="w-80"
              />
            </Row>
          </>
        )}

        {settings.postProcessing.provider === "ollama" && (
          <>
            <Row label="Ollama URL">
              <TextInput
                value={settings.postProcessing.ollamaUrl}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      ollamaUrl: v,
                    },
                  })
                }
                className="w-64"
              />
            </Row>
            <Row label="Model">
              <TextInput
                value={settings.postProcessing.model}
                onValue={(v) =>
                  patch({
                    postProcessing: { ...settings.postProcessing, model: v },
                  })
                }
                placeholder="llama3.2:3b"
              />
            </Row>
          </>
        )}

        {settings.postProcessing.provider === "transformers" && (
          <>
            <Row
              label="Model"
              description="Loaded into the sidecar's cleanup slot. Download more from the Models tab (Backend = Transformers (cleanup))."
            >
              <Select
                value={settings.postProcessing.transformersModelId ?? ""}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      transformersModelId: v || null,
                    },
                  })
                }
              >
                <option value="">— Select —</option>
                {cleanupModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </Select>
            </Row>
            {cleanupModels.length === 0 && (
              <div className="text-xs text-ink-muted">
                No cleanup models downloaded yet. Open the Models tab and pick
                an instruct-tuned causal LM (e.g. Llama-3.2-1B-Instruct,
                Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct).
              </div>
            )}
            <Row label="Device">
              <Select
                value={settings.postProcessing.transformersDevice || "cpu"}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      transformersDevice: v,
                    },
                  })
                }
              >
                {["cpu", "cuda", "mps"].map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </Select>
            </Row>
            <Row
              label="Max new tokens"
              description="Cap on tokens generated per cleanup pass. Lower = faster, fewer hallucinations; higher = handles longer dictations."
            >
              <NumberInput
                value={settings.postProcessing.transformersMaxTokens}
                min={32}
                max={4096}
                onValue={(v) =>
                  patch({
                    postProcessing: {
                      ...settings.postProcessing,
                      transformersMaxTokens: v || 512,
                    },
                  })
                }
              />
            </Row>
            <Row
              label="Preload"
              description="Load the model into RAM/VRAM now so the first dictation doesn't pay the multi-second load cost."
            >
              <div className="flex items-center gap-2">
                <Button
                  variant="primary"
                  onClick={preloadCleanup}
                  disabled={
                    preloading ||
                    !settings.postProcessing.transformersModelId
                  }
                >
                  {preloading ? (
                    <Loader2 className="animate-spin" size={14} />
                  ) : (
                    "Preload now"
                  )}
                </Button>
                {preloadStatus.kind === "ok" && (
                  <span className="text-xs text-green-500">Loaded</span>
                )}
                {preloadStatus.kind === "error" && (
                  <span className="break-words text-xs text-red-400">
                    {preloadStatus.message}
                  </span>
                )}
              </div>
            </Row>
            <div className="text-xs text-ink-muted">
              Heads up: this LLM is loaded alongside your active ASR model.
              Expect 1–4 GB extra RAM/VRAM depending on the model size.
            </div>
          </>
        )}

        <div>
          <div className="mb-1 text-sm">System prompt</div>
          <TextArea
            rows={4}
            value={settings.postProcessing.systemPrompt}
            onValue={(v) =>
              patch({
                postProcessing: {
                  ...settings.postProcessing,
                  systemPrompt: v,
                },
              })
            }
          />
        </div>
      </Card>
    </div>
  );
}
