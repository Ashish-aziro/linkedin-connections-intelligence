import type { ReactNode } from "react";
import { formatPercent, scoreColor } from "../lib/format";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-slate-200 bg-card shadow-card ${className}`}>
      {children}
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
}) {
  const styles = {
    primary: "bg-accent text-white hover:bg-indigo-600 disabled:bg-slate-300",
    ghost: "bg-white text-ink border border-slate-200 hover:bg-slate-50",
    danger: "bg-white text-red-600 border border-red-200 hover:bg-red-50",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed ${styles}`}
    >
      {children}
    </button>
  );
}

export function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
      <div
        className="h-full rounded-full bg-accent transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  );
}

export function StatCard({ label, value, tone = "default" }: { label: string; value: ReactNode; tone?: "default" | "good" | "warn" | "bad" }) {
  const toneCls = {
    default: "text-ink",
    good: "text-emerald-600",
    warn: "text-amber-600",
    bad: "text-red-600",
  }[tone];
  return (
    <Card className="px-5 py-4">
      <div className="text-xs font-medium uppercase tracking-wide text-ink-faint">{label}</div>
      <div className={`mt-1 font-mono text-2xl font-semibold ${toneCls}`}>{value}</div>
    </Card>
  );
}

export function ScoreMeter({
  label,
  value,
  max = 100,
  emphasis = false,
}: {
  label: string;
  value: number;
  max?: number;
  emphasis?: boolean;
}) {
  const pct = formatPercent(value, max);
  const c = scoreColor(pct);
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="font-medium uppercase tracking-wide text-ink-faint">{label}</span>
        <span className={`font-mono ${emphasis ? "text-lg font-bold" : "text-sm"} ${c.text}`}>
          {pct}%
        </span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full ${c.bar}`}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  );
}

export function Badge({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "accent" | "fact" | "inferred" | "warn";
}) {
  const cls = {
    default: "bg-slate-100 text-ink-soft",
    accent: "bg-accent-soft text-accent",
    fact: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
    inferred: "bg-violet-50 text-violet-700 ring-1 ring-violet-200",
    warn: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
  }[tone];
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>
      {children}
    </span>
  );
}
