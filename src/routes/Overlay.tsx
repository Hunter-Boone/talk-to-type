import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { useEffect, useRef, useState } from "react";

import type { OverlayPayload, OverlayState } from "../lib/types";

// Lightweight VU meter — animates 14 bars whose heights are smoothed from
// the peak value the Rust side pushes 20×/sec.
function Bars({
  peak,
  active,
}: {
  peak: number;
  active: boolean;
}) {
  const heights = useRef<number[]>(Array(14).fill(0.05));
  // Read the latest peak through a ref so the interval below survives
  // prop changes. With `peak` in the effect deps, every ~50ms peak event
  // recreated the interval and cancelled the pending tick — the timer
  // never fired while the user was speaking, freezing the meter exactly
  // when it should move.
  const peakRef = useRef(peak);
  peakRef.current = peak;
  const [, force] = useState(0);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      const p = peakRef.current;
      // Perceptual (dB) scale: -50 dB → 0, -6 dB → full. A linear scale
      // pinned quiet sources (virtual cables like Voicemeeter, distant
      // mics) at the minimum bar height even though ASR hears them fine.
      const target =
        p > 0
          ? Math.min(1, Math.max(0, (20 * Math.log10(p) + 50) / 44))
          : 0;
      heights.current = heights.current.map((h) => {
        // Random per-bar noise so the meter doesn't look uniform.
        const noise = 0.4 + Math.random() * 0.6;
        const goal = target * noise;
        return h * 0.5 + goal * 0.5;
      });
      force((n) => n + 1);
    }, 50);
    return () => clearInterval(id);
  }, [active]);

  return (
    <div className="flex h-7 items-end gap-[3px]">
      {heights.current.map((h, i) => (
        <span
          key={i}
          className="w-[3px] rounded-sm bg-white/85"
          style={{ height: `${Math.max(8, h * 100)}%` }}
        />
      ))}
    </div>
  );
}

const labelFor = (state: OverlayState): string => {
  switch (state) {
    case "listening":
      return "Listening…";
    case "transcribing":
      return "Transcribing…";
    case "pasted":
      return "Pasted ✓";
    case "error":
      return "Error";
    default:
      return "";
  }
};

export function OverlayWindow() {
  const [payload, setPayload] = useState<OverlayPayload>({
    state: "hidden",
    peak: 0,
  });

  useEffect(() => {
    document.body.classList.add("overlay-window");
    let unlisten: UnlistenFn | undefined;
    listen<OverlayPayload>("overlay://state", (e) => {
      setPayload(e.payload);
    }).then((u) => (unlisten = u));
    return () => {
      document.body.classList.remove("overlay-window");
      unlisten?.();
    };
  }, []);

  return (
    <div className="flex h-screen w-screen items-center justify-center pointer-events-none">
      <div
        className={[
          "flex items-center gap-3 rounded-full px-4 py-2 shadow-lg backdrop-blur-md transition",
          payload.state === "error"
            ? "bg-red-600/90 text-white"
            : "bg-black/75 text-white",
        ].join(" ")}
      >
        {payload.state === "listening" ? (
          <Bars peak={payload.peak ?? 0} active />
        ) : (
          <span className="inline-block h-2 w-2 rounded-full bg-white/80 animate-pulse" />
        )}
        <span className="text-xs font-medium">
          {labelFor(payload.state)}
          {payload.message ? ` — ${payload.message}` : ""}
        </span>
      </div>
    </div>
  );
}
