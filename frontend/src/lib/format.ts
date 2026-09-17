/**
 * Round a raw score to a whole-number percentage of its max (default 100).
 * Returns 0 for a non-positive max instead of NaN/Infinity.
 */
export function formatPercent(value: number, max = 100): number {
  if (!max || max <= 0) return 0;
  return Math.round((value / max) * 100);
}

export function scoreColor(score: number): { bar: string; text: string } {
  if (score >= 85) return { bar: "bg-emerald-500", text: "text-emerald-700" };
  if (score >= 70) return { bar: "bg-amber-500", text: "text-amber-700" };
  if (score >= 50) return { bar: "bg-slate-400", text: "text-slate-600" };
  return { bar: "bg-slate-300", text: "text-slate-500" };
}

export function stateLabel(state: string): string {
  return state.replace(/_/g, " ").toLowerCase();
}

export function initials(name: string | null | undefined): string {
  if (!name) return "?";
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}
