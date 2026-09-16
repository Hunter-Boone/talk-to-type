import { type ReactNode } from "react";

export function Card({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-surface-strong/40 bg-surface-subtle/30 p-5 mb-4">
      <header className="mb-3">
        <h2 className="text-sm font-medium">{title}</h2>
        {description && (
          <p className="mt-0.5 text-xs text-ink-muted">{description}</p>
        )}
      </header>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

export function Row({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-4">
      <div>
        <div className="text-sm">{label}</div>
        {description && (
          <div className="text-xs text-ink-muted">{description}</div>
        )}
      </div>
      <div className="flex items-center gap-2">{children}</div>
    </div>
  );
}
