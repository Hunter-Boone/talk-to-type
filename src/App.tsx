import { useEffect } from "react";
import { Routes, Route, NavLink, Navigate, useLocation } from "react-router-dom";
import {
  Mic,
  Cpu,
  Database,
  History as HistoryIcon,
  Settings as SettingsIcon,
} from "lucide-react";

import { api } from "./lib/api";
import { OverlayWindow } from "./routes/Overlay";
import { GeneralRoute } from "./routes/General";
import { AudioRoute } from "./routes/Audio";
import { BackendRoute } from "./routes/Backend";
import { ModelsRoute } from "./routes/Models";
import { HistoryRoute } from "./routes/History";

function MainShell() {
  // Register the hotkey once when the main window mounts. Idempotent on Rust
  // side — re-registering is cheap.
  useEffect(() => {
    api.registerHotkey().catch((e) => console.error("hotkey register", e));
  }, []);

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
      isActive
        ? "bg-surface-strong text-ink"
        : "text-ink-muted hover:text-ink hover:bg-surface-subtle",
    ].join(" ");

  return (
    <div className="flex h-full">
      <aside className="w-56 shrink-0 border-r border-surface-strong/40 bg-surface-subtle/40 p-3">
        <div className="px-2 py-3 text-xs uppercase tracking-wider text-ink-muted">
          talk-to-type
        </div>
        <nav className="flex flex-col gap-0.5">
          <NavLink to="/general" className={linkClass}>
            <SettingsIcon size={16} /> General
          </NavLink>
          <NavLink to="/audio" className={linkClass}>
            <Mic size={16} /> Audio
          </NavLink>
          <NavLink to="/backend" className={linkClass}>
            <Cpu size={16} /> Backend
          </NavLink>
          <NavLink to="/models" className={linkClass}>
            <Database size={16} /> Models
          </NavLink>
          <NavLink to="/history" className={linkClass}>
            <HistoryIcon size={16} /> History
          </NavLink>
        </nav>
      </aside>
      <main className="flex-1 overflow-y-auto p-6">
        <Routes>
          <Route path="/" element={<Navigate to="/general" replace />} />
          <Route path="/general" element={<GeneralRoute />} />
          <Route path="/audio" element={<AudioRoute />} />
          <Route path="/backend" element={<BackendRoute />} />
          <Route path="/models" element={<ModelsRoute />} />
          <Route path="/history" element={<HistoryRoute />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  const location = useLocation();
  // Overlay window mounts the same React bundle, but at #/overlay. We
  // branch the entire UI here so the overlay never paints the main shell.
  if (location.pathname.startsWith("/overlay")) {
    return <OverlayWindow />;
  }
  return <MainShell />;
}
