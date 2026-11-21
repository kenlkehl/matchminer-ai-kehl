/**
 * Inline SVG charts for the Benchmarking tab: interval (forest) marks for table cells, reliability plots and thin
 * segmented bars. One hue for single-series marks (the brand blue), hairline grey axes, 2px lines, 8px markers with
 * a 2px surface ring; every value they draw is also printed in the table beside them.
 */
import { cn } from "cn";
import { type ReactNode, useId, useState } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import { num } from "./format";

export const INK = {
  mark: "var(--brand-blue)",
  surface: "#ffffff",
  grid: "#e5e7eb", // gray-200
  axis: "#d1d5db", // gray-300
  muted: "#6b7280", // gray-500
};

/** Job-state colours: the status palette (fixed meanings), in the bar's order. Validated as adjacent pairs. */
export const STATE_COLORS: Record<string, string> = {
  done: "#0ca30c",
  running: "#00629b",
  blocked: "#ec835a",
  failed: "#d03b3b",
  pending: "#d1d5db",
};

export type Domain = [number, number];

/** Round tick values covering ``domain`` (about ``count`` of them). */
export function niceTicks([lo, hi]: Domain, count = 4): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * power >= raw) ?? 10) * power;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    out.push(Math.abs(v) < 1e-12 ? 0 : Number(v.toPrecision(12)));
  }
  return out;
}

/** A domain that holds every value (and ``include``), padded a little and snapped to ticks. */
export function domainOf(values: (number | null | undefined)[], include: number[] = []): Domain {
  const finite = [...values, ...include].filter(
    (v): v is number => v !== null && v !== undefined && Number.isFinite(v),
  );
  if (!finite.length) return [0, 1];
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  const ticks = niceTicks([lo, hi], 4);
  const step = ticks.length > 1 ? (ticks[1] ?? 0) - (ticks[0] ?? 0) : (hi - lo) / 4;
  return [Math.floor(lo / step + 1e-9) * step, Math.ceil(hi / step - 1e-9) * step];
}

function scale([lo, hi]: Domain, [a, b]: [number, number]) {
  return (v: number) => a + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1)) * (b - a);
}

/** A tick label with as many decimals as the step needs (0.25 -> 2, 2.5 -> 1). */
function tickText(v: number, step: number): string {
  let digits = 0;
  while (digits < 4 && Math.abs(Math.round(step * 10 ** digits) - step * 10 ** digits) > 1e-6) {
    digits += 1;
  }
  return v.toFixed(digits).replace(/^-/, "−");
}

const PAD = 12;

/** About one tick per 40px. */
function tickCount(width: number): number {
  return Math.max(2, Math.min(4, Math.floor((width - 2 * PAD) / 40)));
}

/**
 * One estimate and its 95% interval on a shared horizontal scale (a forest-plot row): a 2px interval line and an
 * 8px point; ``reference`` draws a hairline (0 for differences, 0.5 for AUROC). The tooltip repeats the numbers.
 */
export function IntervalMark({
  value,
  low,
  high,
  domain,
  reference,
  width = 140,
  label,
  digits = 3,
}: {
  value: number | null | undefined;
  low?: number | null;
  high?: number | null;
  domain: Domain;
  reference?: number;
  width?: number;
  label: string;
  digits?: number;
}) {
  const height = 20;
  const x = scale(domain, [PAD, width - PAD]);
  const ticks = niceTicks(domain, tickCount(width));
  const ok = value !== null && value !== undefined && Number.isFinite(value);
  const hasInterval =
    ok && low !== null && low !== undefined && high !== null && high !== undefined;
  const svg = (
    <svg width={width} height={height} role="img" aria-label={label} className="block">
      {ticks.map((t) => (
        <line key={t} x1={x(t)} x2={x(t)} y1={2} y2={height - 2} stroke={INK.grid} />
      ))}
      {reference !== undefined ? (
        <line
          x1={x(reference)}
          x2={x(reference)}
          y1={0}
          y2={height}
          stroke={INK.muted}
          strokeWidth={1}
        />
      ) : null}
      {hasInterval ? (
        <line
          x1={x(low)}
          x2={x(high)}
          y1={height / 2}
          y2={height / 2}
          stroke={INK.mark}
          strokeWidth={2}
          strokeLinecap="round"
        />
      ) : null}
      {ok ? (
        <circle
          cx={x(value)}
          cy={height / 2}
          r={4}
          fill={INK.mark}
          stroke={INK.surface}
          strokeWidth={2}
          paintOrder="stroke"
        />
      ) : null}
    </svg>
  );
  if (!ok) return svg;
  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" tabIndex={0} />}>{svg}</TooltipTrigger>
      <TooltipContent>
        <span className="font-semibold tabular-nums">{num(value, digits)}</span>
        {hasInterval ? (
          <span className="tabular-nums opacity-80">
            95% CI {num(low, digits)} to {num(high, digits)}
          </span>
        ) : null}
        <span className="opacity-80">{label}</span>
      </TooltipContent>
    </Tooltip>
  );
}

/** The tick labels under a column of IntervalMarks (same width and domain). */
export function IntervalAxis({
  domain,
  width = 140,
  format,
}: {
  domain: Domain;
  width?: number;
  format?: (v: number) => string;
}) {
  const ticks = niceTicks(domain, tickCount(width));
  const step = ticks.length > 1 ? (ticks[1] ?? 0) - (ticks[0] ?? 0) : 1;
  const x = scale(domain, [PAD, width - PAD]);
  return (
    <svg width={width} height={16} aria-hidden className="block">
      <line x1={PAD} x2={width - PAD} y1={1} y2={1} stroke={INK.axis} />
      {ticks.map((t) => (
        <text
          key={t}
          x={x(t)}
          y={13}
          textAnchor="middle"
          fontSize={10}
          fill={INK.muted}
          className="tabular-nums"
        >
          {format ? format(t) : tickText(t, step)}
        </text>
      ))}
    </svg>
  );
}

// --------------------------------------------------------------------------- reliability

export interface ReliabilityPoint {
  n: number;
  p_min: number;
  p_max: number;
  mean_predicted: number;
  observed: number;
}

/**
 * Observed yes rate against mean P(yes) per equal-mass bin, with the diagonal of perfect calibration. Hover or
 * focus a point for its bin.
 */
export function ReliabilityPlot({
  bins,
  title,
  size = 232,
}: {
  bins: ReliabilityPoint[];
  title: string;
  size?: number;
}) {
  const [active, setActive] = useState<number | null>(null);
  const id = useId();
  const m = { left: 34, right: 10, top: 8, bottom: 30 };
  const w = size;
  const h = size;
  const x = scale([0, 1], [m.left, w - m.right]);
  const y = scale([0, 1], [h - m.bottom, m.top]);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const path = bins
    .map((b, i) => `${i ? "L" : "M"}${x(b.mean_predicted).toFixed(1)},${y(b.observed).toFixed(1)}`)
    .join("");
  const shown = active === null ? null : bins[active];
  return (
    <div className="relative inline-block">
      <svg width={w} height={h} role="img" aria-labelledby={`${id}-t`} className="block">
        <title id={`${id}-t`}>
          {`${title}: observed yes rate against mean P(yes) in ${String(bins.length)} bins`}
        </title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke={INK.grid} />
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke={INK.grid} />
            <text x={x(t)} y={h - m.bottom + 13} textAnchor="middle" fontSize={10} fill={INK.muted}>
              {t === 0 || t === 1 ? String(t) : t.toFixed(2).replace(/0$/, "")}
            </text>
            <text x={m.left - 5} y={y(t) + 3} textAnchor="end" fontSize={10} fill={INK.muted}>
              {t === 0 || t === 1 ? String(t) : t.toFixed(2).replace(/0$/, "")}
            </text>
          </g>
        ))}
        <line x1={x(0)} x2={x(1)} y1={y(0)} y2={y(0)} stroke={INK.axis} />
        <line x1={x(0)} x2={x(0)} y1={y(0)} y2={y(1)} stroke={INK.axis} />
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={INK.muted} strokeWidth={1} />
        <text x={(x(0) + x(1)) / 2} y={h - 3} textAnchor="middle" fontSize={10} fill={INK.muted}>
          Mean P(yes)
        </text>
        <text
          transform={`translate(9 ${String((y(0) + y(1)) / 2)}) rotate(-90)`}
          textAnchor="middle"
          fontSize={10}
          fill={INK.muted}
        >
          Observed yes
        </text>
        <path
          d={path}
          fill="none"
          stroke={INK.mark}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {bins.map((b, i) => (
          <g
            key={i}
            tabIndex={0}
            role="img"
            aria-label={`Bin ${String(i + 1)}: mean P(yes) ${num(b.mean_predicted)}, observed ${num(b.observed)}, n ${String(b.n)}`}
            onPointerEnter={() => {
              setActive(i);
            }}
            onPointerLeave={() => {
              setActive(null);
            }}
            onFocus={() => {
              setActive(i);
            }}
            onBlur={() => {
              setActive(null);
            }}
            className="cursor-default outline-none"
          >
            <circle cx={x(b.mean_predicted)} cy={y(b.observed)} r={12} fill="transparent" />
            <circle
              cx={x(b.mean_predicted)}
              cy={y(b.observed)}
              r={active === i ? 5 : 4}
              fill={INK.mark}
              stroke={INK.surface}
              strokeWidth={2}
              paintOrder="stroke"
            />
          </g>
        ))}
      </svg>
      {shown ? (
        <div
          className="pointer-events-none absolute z-10 rounded-md bg-foreground px-2.5 py-1.5 text-xs whitespace-nowrap text-background shadow-md"
          style={{
            left: Math.min(x(shown.mean_predicted) + 10, w - 150),
            top: Math.max(y(shown.observed) - 64, 0),
          }}
        >
          <div>
            <span className="font-semibold tabular-nums">{num(shown.observed)}</span>{" "}
            <span className="opacity-80">observed yes</span>
          </div>
          <div>
            <span className="font-semibold tabular-nums">{num(shown.mean_predicted)}</span>{" "}
            <span className="opacity-80">mean P(yes)</span>
          </div>
          <div className="tabular-nums opacity-80">
            P(yes) {num(shown.p_min)}–{num(shown.p_max)} · n {shown.n.toLocaleString()}
          </div>
        </div>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------- segmented bars

export interface Part {
  key: string;
  label: string;
  value: number;
  color: string;
}

/** A thin (8px) bar split into parts with 2px gaps; the legend beside it carries the names and figures. */
export function ThinSegmentedBar({
  parts,
  label,
  className,
}: {
  parts: Part[];
  label: string;
  className?: string;
}) {
  const total = parts.reduce((s, p) => s + Math.max(0, p.value), 0);
  const shown = parts.filter((p) => p.value > 0);
  return (
    <div
      role="img"
      aria-label={`${label}: ${parts.map((p) => `${p.label} ${String(p.value)}`).join(", ")}`}
      className={cn("flex h-2 w-full gap-0.5 overflow-hidden rounded-full", className)}
    >
      {total > 0
        ? shown.map((p) => (
            <Tooltip key={p.key}>
              <TooltipTrigger
                render={
                  <span
                    className="h-full first:rounded-l-full last:rounded-r-full hover:opacity-80"
                    style={{ flexGrow: p.value, flexBasis: 0, background: p.color }}
                  />
                }
              />
              <TooltipContent>
                <span className="font-semibold tabular-nums">{p.value.toLocaleString()}</span>
                <span className="opacity-80">
                  {p.label} ({Math.round((p.value / total) * 100)}%)
                </span>
              </TooltipContent>
            </Tooltip>
          ))
        : null}
    </div>
  );
}

/** A legend row: a swatch, the name and a figure. */
export function LegendItem({
  color,
  children,
  shape = "rect",
}: {
  color: string;
  children: ReactNode;
  shape?: "rect" | "line";
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-600">
      <span
        aria-hidden
        className={shape === "line" ? "h-0.5 w-3 rounded-full" : "size-2.5 rounded-[3px]"}
        style={{ background: color }}
      />
      {children}
    </span>
  );
}

/** A meter: the share done of a whole, on a lighter track of the same hue. */
export function Meter({ value, total, label }: { value: number; total: number; label: string }) {
  const share = total > 0 ? Math.min(1, value / total) : 0;
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={value}
      className="h-2 w-full overflow-hidden rounded-full bg-brand-blue/15"
    >
      <div
        className="h-full rounded-full bg-brand-blue"
        style={{ width: `${String(share * 100)}%` }}
      />
    </div>
  );
}
