import { useEffect, useState } from "react";

import { Card, Row } from "../components/Card";
import { Button, Select, TextInput, Toggle } from "../components/Field";
import { useSettings } from "../hooks/useSettings";
import { api } from "../lib/api";

export function GeneralRoute() {
  const { settings, patch } = useSettings();
  const [draftHotkey, setDraftHotkey] = useState("");
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);
  const [draftRepaste, setDraftRepaste] = useState("");
  const [repasteError, setRepasteError] = useState<string | null>(null);
  const [repasteSaved, setRepasteSaved] = useState(false);

  useEffect(() => {
    if (settings) setDraftHotkey(settings.hotkey);
  }, [settings?.hotkey]);

  useEffect(() => {
    if (settings) setDraftRepaste(settings.repasteHotkey ?? "");
  }, [settings?.repasteHotkey]);

  if (!settings) return <div className="text-ink-muted">Loading…</div>;

  const saveHotkey = async () => {
    setHotkeyError(null);
    try {
      await api.registerHotkey(draftHotkey);
    } catch (e) {
      setHotkeyError(String(e));
    }
  };

  const saveRepaste = async () => {
    setRepasteError(null);
    setRepasteSaved(false);
    try {
      const trimmed = draftRepaste.trim();
      await api.registerRepasteHotkey(trimmed === "" ? null : trimmed);
      patch({ repasteHotkey: trimmed === "" ? null : trimmed });
      setRepasteSaved(true);
    } catch (e) {
      setRepasteError(String(e));
    }
  };

  return (
    <div className="max-w-2xl">
      <h1 className="mb-4 text-lg font-semibold">General</h1>

      <Card title="Behavior">
        <Row
          label="Launch at login"
          description="Open talk-to-type when you sign in to your computer."
        >
          <Toggle
            checked={settings.launchAtLogin}
            onChange={(v) => patch({ launchAtLogin: v })}
          />
        </Row>
        <Row
          label="Show in tray / menu bar"
          description="With the tray off, closing the window quits the app. Takes effect after a restart."
        >
          <Toggle
            checked={settings.showInTray}
            onChange={(v) => patch({ showInTray: v })}
          />
        </Row>
        <Row
          label="After transcription"
          description="Auto-paste into the focused app, or only put it on the clipboard."
        >
          <Select
            value={settings.postAction}
            onValue={(v) =>
              patch({ postAction: v as "paste" | "clipboard-only" })
            }
          >
            <option value="paste">Auto-paste</option>
            <option value="clipboard-only">Clipboard only</option>
          </Select>
        </Row>
        <Row
          label="Stream transcription while speaking"
          description="Transcribe with overlapping windows in the background while you keep talking. Each window includes a few seconds of lookback from before the last committed boundary so the model has full acoustic context, and word-level timestamps are used to splice the windows together at exact word boundaries. Result: release-to-paste latency drops dramatically AND each word benefits from full-context inference. Uses extra CPU/GPU during recording — turn off if your machine feels overloaded, or for absolute parity with single-pass transcription."
        >
          <Toggle
            checked={settings.pretranscribe}
            onChange={(v) => patch({ pretranscribe: v })}
          />
        </Row>
      </Card>

      <Card
        title="Hotkey"
        description="Hold this key combo to dictate. Use modifier+letter syntax — e.g. Ctrl+Shift+Space."
      >
        <Row label="Combo">
          <TextInput
            value={draftHotkey}
            onValue={setDraftHotkey}
            placeholder="Ctrl+Shift+Space"
          />
          <Button variant="primary" onClick={saveHotkey}>
            Apply
          </Button>
        </Row>
        {hotkeyError && (
          <div className="text-xs text-red-500">{hotkeyError}</div>
        )}
      </Card>

      <Card
        title="Repaste last transcription"
        description="Press this combo to paste the most recent transcription into the focused app — useful when auto-paste missed the target window. Leave empty to disable."
      >
        <Row label="Combo">
          <TextInput
            value={draftRepaste}
            onValue={(v) => {
              setDraftRepaste(v);
              setRepasteSaved(false);
            }}
            placeholder="Ctrl+Alt+V"
          />
          <Button variant="primary" onClick={saveRepaste}>
            Apply
          </Button>
        </Row>
        {repasteError && (
          <div className="text-xs text-red-500">{repasteError}</div>
        )}
        {repasteSaved && !repasteError && (
          <div className="text-xs text-ink-muted">Saved.</div>
        )}
      </Card>
    </div>
  );
}
