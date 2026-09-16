import {
  type ChangeEvent,
  type InputHTMLAttributes,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

const baseInput =
  "rounded-md border border-surface-strong/60 bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-accent/50";

export function TextInput(
  props: InputHTMLAttributes<HTMLInputElement> & {
    onValue?: (v: string) => void;
  }
) {
  const { onValue, onChange, ...rest } = props;
  return (
    <input
      {...rest}
      className={[baseInput, props.className].filter(Boolean).join(" ")}
      onChange={(e: ChangeEvent<HTMLInputElement>) => {
        onValue?.(e.target.value);
        onChange?.(e);
      }}
    />
  );
}

export function NumberInput(
  props: InputHTMLAttributes<HTMLInputElement> & {
    onValue?: (v: number) => void;
  }
) {
  const { onValue, onChange, ...rest } = props;
  return (
    <input
      {...rest}
      type="number"
      className={[baseInput, "w-24", props.className].filter(Boolean).join(" ")}
      onChange={(e: ChangeEvent<HTMLInputElement>) => {
        const n = Number(e.target.value);
        if (!Number.isNaN(n)) onValue?.(n);
        onChange?.(e);
      }}
    />
  );
}

export function Select(
  props: SelectHTMLAttributes<HTMLSelectElement> & {
    onValue?: (v: string) => void;
  }
) {
  const { onValue, onChange, children, ...rest } = props;
  return (
    <select
      {...rest}
      className={[baseInput, props.className].filter(Boolean).join(" ")}
      onChange={(e: ChangeEvent<HTMLSelectElement>) => {
        onValue?.(e.target.value);
        onChange?.(e);
      }}
    >
      {children}
    </select>
  );
}

export function TextArea(
  props: TextareaHTMLAttributes<HTMLTextAreaElement> & {
    onValue?: (v: string) => void;
  }
) {
  const { onValue, onChange, ...rest } = props;
  return (
    <textarea
      {...rest}
      className={[baseInput, "w-full font-mono text-xs", props.className]
        .filter(Boolean)
        .join(" ")}
      onChange={(e) => {
        onValue?.(e.target.value);
        onChange?.(e);
      }}
    />
  );
}

export function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={[
        "relative inline-flex h-5 w-9 items-center rounded-full transition",
        checked ? "bg-accent" : "bg-surface-strong",
      ].join(" ")}
      aria-pressed={checked}
    >
      <span
        className={[
          "inline-block h-4 w-4 transform rounded-full bg-white transition",
          checked ? "translate-x-4" : "translate-x-0.5",
        ].join(" ")}
      />
    </button>
  );
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
};

export function Button(props: ButtonProps) {
  const { variant = "secondary", className, ...rest } = props;
  const variants: Record<string, string> = {
    primary: "bg-accent text-white hover:bg-accent/90",
    secondary:
      "bg-surface-strong text-ink hover:bg-surface-strong/80",
    danger: "bg-red-500/90 text-white hover:bg-red-500",
  };
  return (
    <button
      {...rest}
      className={[
        "rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-50",
        variants[variant],
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    />
  );
}
