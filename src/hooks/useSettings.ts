import { useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { Settings } from "../lib/types";

export function useSettings() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSettings()
      .then(setSettings)
      .catch((e) => setError(String(e)));
  }, []);

  const update = useCallback(async (next: Settings) => {
    setSaving(true);
    setError(null);
    try {
      await api.setSettings(next);
      setSettings(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }, []);

  const patch = useCallback(
    (partial: Partial<Settings>) => {
      if (!settings) return;
      void update({ ...settings, ...partial });
    },
    [settings, update]
  );

  return { settings, update, patch, saving, error };
}
