/** Number formatting for the Benchmarking tab (the metrics' reports use three decimals). */

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return value.toFixed(digits);
}

export function signed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined) return "–";
  return value.toLocaleString();
}

export function usd(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "–";
  return `$${value.toFixed(digits)}`;
}

/** "[0.626, 0.663]" or "" when there is no interval. */
export function interval(
  low: number | null | undefined,
  high: number | null | undefined,
  digits = 3,
  format: (v: number, d: number) => string = (v, d) => v.toFixed(d),
): string {
  if (low === null || low === undefined || high === null || high === undefined) return "";
  return `[${format(low, digits)}, ${format(high, digits)}]`;
}

/** Underscores to spaces: "treatment_history" -> "treatment history". */
export function words(value: string): string {
  return value.replace(/_/g, " ");
}

export function capitalize(value: string): string {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/** A timestamp in seconds, or an ISO string, as "Nov 21, 9:30 AM". */
export function stamp(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "";
  if (seconds < 90) return `${String(Math.round(seconds))} s`;
  if (seconds < 5400) return `${String(Math.round(seconds / 60))} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

/** The judges' names the adjudication reveal uses (evaluation/dual_judge/judging.py JUDGE_NAMES). */
const JUDGE_NAMES: Record<string, string> = {
  vertex: "Claude Opus 5.5 (Vertex)",
  databricks: "Claude Opus 5.5 (Databricks)",
  azure: "GPT-6.1 Sol (Azure)",
};

export function judgeName(judge: string): string {
  return JUDGE_NAMES[judge] ?? judge;
}

export const COMPONENT_LABELS: Record<string, string> = {
  extraction: "Cohort extraction",
  summary: "Patient summary",
  decision: "Decision questions",
};

export function componentLabel(component: string): string {
  return COMPONENT_LABELS[component] ?? capitalize(component);
}
