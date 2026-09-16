import { useEffect, useRef, useState } from "react";

import { Card, Row } from "../components/Card";
import { Select, Toggle } from "../components/Field";
import { useSettings } from "../hooks/useSettings";
import { api } from "../lib/api";
import type { InputDeviceInfo } from "../lib/types";

export function AudioRoute() {
  const { settings, patch } = useSettings();
  const [devices, setDevices] = useState<InputDeviceInfo[]>([]);

  useEffect(() => {
    api.listInputDevices().then(setDevices).catch(() => setDevices([]));
  }, []);

  if (!settings) return <div className="text-ink-muted">Loading…</div>;

  return (
    <div className="max-w-2xl">
      <h1 className="mb-4 text-lg font-semibold">Audio</h1>

      <Card title="Input device">
        <Row label="Microphone">
          <Select
            value={settings.inputDevice ?? ""}
            onValue={(v) => patch({ inputDevice: v || null })}
          >
            <option value="">System default</option>
            {devices.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
                {d.isDefault ? " (default)" : ""}
              </option>
            ))}
          </Select>
        </Row>
        <LevelMeter device={settings.inputDevice ?? null} />
      </Card>

      <Card
        title="Voice activity detection"
        description="Trim leading and trailing silence with Silero VAD before sending audio to the backend. Recommended."
      >
        <Row label="Enabled">
          <Toggle
            checked={settings.vadEnabled}
            onChange={(v) => patch({ vadEnabled: v })}
          />
        </Row>
      </Card>
    </div>
  );
}

/// Live mic VU meter. Mounting this component starts a monitor stream on
/// the audio thread; unmounting (or device change) tears it down. We poll
/// the peak ~30Hz and decay it visually so a short utterance fades out
/// smoothly instead of latching at max.
function LevelMeter({ device }: { device: string | null }) {
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const rafRef = useRef<number | null>(null);
  const smoothedRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .startAudioMonitor(device)
      .then(() => {
        if (cancelled) {
          api.stopAudioMonitor().catch(() => {});
          return;
        }
        setActive(true);
      })
      .catch((e: unknown) => {
        setError(String(e));
        setActive(false);
      });

    return () => {
      cancelled = true;
      setActive(false);
      api.stopAudioMonitor().catch(() => {});
    };
  }, [device]);

  useEffect(() => {
    if (!active) return;
    let stopped = false;

    const tick = async () => {
      if (stopped) return;
      try {
        const peak = await api.currentAudioLevel();
        // Light visual smoothing so the bar doesn't strobe.
        const target = Math.min(1, Math.max(0, peak));
        const prev = smoothedRef.current;
        const next = target > prev ? target : prev * 0.9 + target * 0.1;
        smoothedRef.current = next;
        setLevel(next);
      } catch {
        // Polling failures are non-fatal; just retry on the next frame.
      }
      rafRef.current = window.setTimeout(tick, 33) as unknown as number;
    };
    tick();

    return () => {
      stopped = true;
      if (rafRef.current != null) {
        clearTimeout(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [active]);

  const pct = Math.round(level * 100);
  // Speech sits roughly in the 5%–40% range. Anything above ~70% is loud /
  // possibly clipping. Color the bar accordingly.
  const color =
    level > 0.7
      ? "bg-rose-500"
      : level > 0.05
      ? "bg-emerald-500"
      : "bg-emerald-500/40";

  return (
    <div className="pt-1">
      <div className="mb-1 flex items-center justify-between text-xs text-ink-muted">
        <span>Input level</span>
        <span className="tabular-nums">
          {error ? "error" : active ? `${pct}%` : "starting…"}
        </span>
      </div>
      <div
        className="relative h-2 w-full overflow-hidden rounded-full bg-surface-strong/40"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <div
          className={`h-full ${color} transition-[width] duration-75 ease-out`}
          style={{ width: `${pct}%` }}
        />
        {/* Reference tick at ~5% so the user can see the speech threshold. */}
        <div className="absolute inset-y-0 left-[5%] w-px bg-ink-muted/40" />
      </div>
      <p className="mt-1.5 text-xs text-ink-muted">
        {error
          ? `Could not start monitor: ${error}`
          : "Speak normally — you should see green movement above the tick mark. If it stays flat, the wrong device is selected or the mic is muted."}
      </p>
    </div>
  );
}
