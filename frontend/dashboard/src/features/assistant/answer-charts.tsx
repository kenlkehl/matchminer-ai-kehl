/**
 * The Assistant's answer charts: pie, donut, bar and line charts drawn in plain SVG from an answer's checked chart
 * data (assistant/actions.py: chart_view), in the Benchmarking charts' style: hairline grey grid and axes, bars at
 * most 24px thick with 4px rounded data ends and 2px surface gaps, 2px lines with ringed 8px markers.
 *
 * - **Values:** every value is printed on or beside its mark (a pie's in its legend, with its share), a tooltip
 *   repeats it on hover, and each chart has a text alternative (the SVG's title and description), a table view and a
 *   CSV download. Controls are left out of print.
 * - **Colours:** one series takes the brand blue; several series, or a pie's parts, take the slots below in order
 *   (never cycled: the server caps a pie at 6 parts and a chart at 4 series). The order was chosen with the dataviz
 *   skill's validator on the page's white surface: every adjacent pair, and a pie's closing pair, clears CVD ΔE 16.3
 *   and normal-vision ΔE 19.6. Yellow, magenta and light blue are below 3:1 on white, so their values are always
 *   printed. Parts that are absence or remainder ("Not documented", "Unknown", "Other") are grey, so they never read
 *   as a category of their own.
 */
import { DownloadIcon, TableIcon } from "lucide-react";
import { type ReactNode, useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { domainOf, INK, LegendItem, niceTicks } from "@/features/benchmarking/charts";
import { saveBlob } from "@/lib/api";

import type { Chart, Step } from "./types";

/** The categorical slots in their validated order (brand blue, the brand orange's in-band step, then the rest). */
export const SLOTS = ["#00629b", "#eda100", "#e87ba4", "#008300", "#41b6e6", "#eb6834"];
/** De-emphasis grey for absence or remainder (gray-300; ΔE ≥ 15 from every slot). */
const QUIET = "#d1d5db";
const QUIET_NAME =
  /^(not documented|undocumented|unknown|other|others|not checkable|not given|not reported|not available|missing|no data|n\/a)\b/i;
/** Fills a label inside takes near-black ink on (by APCA); white on the others. */
const DARK_INK = new Set(["#eda100", "#41b6e6", QUIET]);
const INK_TEXT = "#1c2024";
const SURFACE = "#ffffff";
const MAX_BAR = 24;

const isPie = (chart: Chart) => chart.kind === "pie" || chart.kind === "donut";
/** Whether values carry a share of the denominator (a pie always; bars when they are parts of one population). */
const hasShares = (chart: Chart) =>
  chart.unit !== "%" && (isPie(chart) || (chart.kind === "bar" && chart.total > 0));

function colorsFor(names: string[], oneHue: boolean): string[] {
  let slot = 0;
  return names.map((name) => {
    if (QUIET_NAME.test(name.trim())) return QUIET;
    const color = SLOTS[oneHue ? 0 : slot % SLOTS.length] ?? SLOTS[0] ?? INK.muted;
    if (!oneHue) slot += 1;
    return color;
  });
}

const NUMBER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

function fmt(value: number, unit: string): string {
  const text = NUMBER.format(value).replace(/^-/, "−");
  return unit === "%" ? `${text}%` : text;
}

function share(value: number, total: number): string {
  if (!(total > 0)) return "";
  const p = (value / total) * 100;
  if (p > 0 && p < 1) return "<1%";
  if (p > 99 && p < 100) return ">99%";
  return `${String(Math.round(p))}%`;
}

function valueText(value: number, chart: Chart, total = chart.total): string {
  const base = fmt(value, chart.unit);
  return hasShares(chart) && total > 0 ? `${base} (${share(value, total)})` : base;
}

function pieTotal(chart: Chart): number {
  const sum = (chart.series[0]?.values ?? []).reduce((a, b) => a + b, 0);
  return chart.total > 0 ? chart.total : sum;
}

// --------------------------------------------------------------------------- text measurement

let measurer: CanvasRenderingContext2D | null | undefined;
let family = "system-ui, sans-serif";

/** The rendered width of SVG text in the page's font (an estimate where canvas is unavailable). */
function textWidth(text: string, size = 11, weight = 400): number {
  if (measurer === undefined) {
    measurer = document.createElement("canvas").getContext("2d");
    family = getComputedStyle(document.body).fontFamily || family;
  }
  if (!measurer) return text.length * size * 0.58;
  measurer.font = `${String(weight)} ${String(size)}px ${family}`;
  return measurer.measureText(text).width;
}

/** The text cut with an ellipsis to fit ``width``. */
function fit(text: string, width: number, size = 11): string {
  if (textWidth(text, size) <= width) return text;
  let cut = text.length;
  while (cut > 1 && textWidth(`${text.slice(0, cut).trimEnd()}…`, size) > width) cut -= 1;
  return `${text.slice(0, cut).trimEnd()}…`;
}

/** Words wrapped into at most ``lines`` lines of ``width`` (the last cut with an ellipsis). */
function wrap(text: string, width: number, lines = 3, size = 11): string[] {
  const out: string[] = [];
  let line = "";
  const words = text.split(/\s+/).filter(Boolean);
  for (let i = 0; i < words.length; i += 1) {
    const word = words[i] ?? "";
    const next = line ? `${line} ${word}` : word;
    if (!line || textWidth(next, size) <= width) {
      line = next;
      continue;
    }
    if (out.length === lines - 1) {
      line = `${line} ${words.slice(i).join(" ")}`;
      break;
    }
    out.push(line);
    line = word;
  }
  if (line) out.push(line);
  const last = out.length - 1;
  if (last >= 0) out[last] = fit(out[last] ?? "", width, size);
  return out;
}

function useWidth(fallback = 640) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(260, Math.floor(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => {
      observer.disconnect();
    };
  }, []);
  return [ref, width] as const;
}

function linear([lo, hi]: [number, number], [a, b]: [number, number]) {
  return (v: number) => a + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1)) * (b - a);
}

/** A value axis's domain: always holding 0 (bars grow from it), [0, 1] when every value is 0. */
function valueDomain(values: number[], withZero = true): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length || finite.every((v) => v === 0)) return [0, 1];
  return domainOf(finite, withZero ? [0] : []);
}

function tickLabel(v: number, unit: string): string {
  const abs = Math.abs(v);
  const text =
    abs >= 10000 ? `${NUMBER.format(v / 1000)}K` : NUMBER.format(Number(v.toPrecision(12)));
  return (unit === "%" ? `${text}%` : text).replace(/^-/, "−");
}

// --------------------------------------------------------------------------- tooltip

interface Tip {
  x: number;
  y: number;
  heading?: string;
  rows: { color: string; value: string; label: string }[];
}

/** The hover readout: values first, names after, each keyed by a short stroke of its colour. */
function TipBox({ tip, width }: { tip: Tip | null; width: number }) {
  if (!tip) return null;
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute z-10 rounded-md bg-foreground px-2.5 py-1.5 text-xs whitespace-nowrap text-background shadow-md print:hidden"
      style={{
        left: Math.max(0, Math.min(tip.x + 12, width - 220)),
        top: Math.max(0, tip.y - 8),
        transform: "translateY(-100%)",
      }}
    >
      {tip.heading ? <div className="mb-0.5 opacity-80">{tip.heading}</div> : null}
      {tip.rows.map((row, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: row.color }} />
          <span className="font-semibold tabular-nums">{row.value}</span>
          {row.label ? <span className="opacity-80">{row.label}</span> : null}
        </div>
      ))}
    </div>
  );
}

/** The chart in words: the SVG's description for screen readers. */
function describe(chart: Chart): string {
  const kind =
    chart.kind === "bar"
      ? `${chart.stacked ? "Stacked bar" : "Bar"} chart`
      : `${chart.kind.charAt(0).toUpperCase()}${chart.kind.slice(1)} chart`;
  const total = isPie(chart) ? pieTotal(chart) : chart.total;
  const parts = chart.labels.map((label, i) => {
    const values = chart.series.map((s) => {
      const v = s.values[i] ?? 0;
      return `${s.name ? `${s.name} ` : ""}${valueText(v, chart, total)}`;
    });
    return `${label}: ${values.join(", ")}`;
  });
  return `${kind}${chart.unit ? ` of ${chart.unit}` : ""}. ${parts.join("; ")}.${
    total > 0 && hasShares(chart) ? ` Total ${fmt(total, chart.unit)}.` : ""
  }`;
}

function SvgFrame({
  width,
  height,
  chart,
  children,
  onLeave,
}: {
  width: number;
  height: number;
  chart: Chart;
  children: ReactNode;
  onLeave?: () => void;
}) {
  const id = useId();
  return (
    <svg
      viewBox={`0 0 ${String(width)} ${String(height)}`}
      width={width}
      height={height}
      role="img"
      aria-labelledby={`${id}-t`}
      aria-describedby={`${id}-d`}
      className="block h-auto max-w-full overflow-visible"
      style={{ width: "100%", maxWidth: width }}
      onPointerLeave={onLeave}
    >
      <title id={`${id}-t`}>{chart.title}</title>
      <desc id={`${id}-d`}>{describe(chart)}</desc>
      {children}
    </svg>
  );
}

// --------------------------------------------------------------------------- pie and donut

function arcPath(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number): string {
  const p = (r: number, a: number) =>
    `${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`;
  if (a1 - a0 >= 2 * Math.PI - 1e-6) {
    const mid = a0 + Math.PI;
    const ring = (r: number) =>
      `M${p(r, a0)}A${String(r)},${String(r)} 0 1 1 ${p(r, mid)}A${String(r)},${String(r)} 0 1 1 ${p(r, a0)}Z`;
    return r0 > 0 ? ring(r1) + ring(r0) : ring(r1);
  }
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const outer = `A${String(r1)},${String(r1)} 0 ${String(large)} 1 ${p(r1, a1)}`;
  if (r0 <= 0) return `M${String(cx)},${String(cy)}L${p(r1, a0)}${outer}Z`;
  return `M${p(r1, a0)}${outer}L${p(r0, a1)}A${String(r0)},${String(r0)} 0 ${String(large)} 0 ${p(r0, a0)}Z`;
}

function PieChart({ chart }: { chart: Chart }) {
  const [active, setActive] = useState<number | null>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const values = chart.series[0]?.values ?? [];
  const colors = colorsFor(chart.labels, false);
  const total = pieTotal(chart);
  const size = 208;
  const c = size / 2;
  const r = c - 4;
  const inner = chart.kind === "donut" ? r * 0.6 : 0;
  let angle = -Math.PI / 2;
  const slices = values.map((v, i) => {
    const sweep = total > 0 ? (Math.max(0, v) / total) * 2 * Math.PI : 0;
    const slice = { i, v, a0: angle, a1: angle + sweep };
    angle += sweep;
    return slice;
  });
  const labelRadius = inner > 0 ? (r + inner) / 2 : r * 0.62;
  const show = (i: number, x: number, y: number) => {
    setActive(i);
    setTip({
      x,
      y,
      rows: [
        {
          color: colors[i] ?? QUIET,
          value: valueText(values[i] ?? 0, chart, total),
          label: chart.labels[i] ?? "",
        },
      ],
    });
  };
  const clear = () => {
    setActive(null);
    setTip(null);
  };
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
      <div className="relative max-w-full shrink-0" style={{ width: size }}>
        <SvgFrame width={size} height={size} chart={chart} onLeave={clear}>
          {slices.map((s) => {
            if (s.a1 - s.a0 <= 0) return null;
            const color = colors[s.i] ?? QUIET;
            const mid = (s.a0 + s.a1) / 2;
            const full = s.a1 - s.a0 >= 2 * Math.PI - 1e-6;
            const label = share(s.v, total);
            const lx = full && inner === 0 ? c : c + labelRadius * Math.cos(mid);
            const ly = full && inner === 0 ? c : c + labelRadius * Math.sin(mid);
            const room = full ? 2 * labelRadius : (s.a1 - s.a0) * labelRadius;
            const fits = room >= textWidth(label, 12, 600) + 8 && (inner === 0 || r - inner >= 18);
            return (
              <g
                key={s.i}
                onPointerMove={(e) => {
                  const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
                  const scale = box ? size / box.width : 1;
                  show(
                    s.i,
                    box ? (e.clientX - box.left) * scale : lx,
                    box ? (e.clientY - box.top) * scale : ly,
                  );
                }}
              >
                <path
                  d={arcPath(c, c, inner, r, s.a0, s.a1)}
                  fill={color}
                  fillRule="evenodd"
                  stroke={SURFACE}
                  strokeWidth={slices.length > 1 && !full ? 2 : 0}
                  strokeLinejoin="round"
                  opacity={active === null || active === s.i ? 1 : 0.55}
                />
                {fits && !(full && inner > 0) ? (
                  <text
                    x={lx}
                    y={ly}
                    dy="0.35em"
                    textAnchor="middle"
                    fontSize={12}
                    fontWeight={600}
                    fill={DARK_INK.has(color) ? INK_TEXT : SURFACE}
                    pointerEvents="none"
                  >
                    {label}
                  </text>
                ) : null}
              </g>
            );
          })}
          {inner > 0 ? (
            <g pointerEvents="none">
              <text
                x={c}
                y={c}
                dy={chart.unit ? "-0.1em" : "0.35em"}
                textAnchor="middle"
                fontSize={22}
                fontWeight={600}
                fill={INK_TEXT}
              >
                {fmt(total, chart.unit)}
              </text>
              {chart.unit && chart.unit !== "%" ? (
                <text x={c} y={c + 16} textAnchor="middle" fontSize={11} fill={INK.muted}>
                  {fit(chart.unit, inner * 1.6)}
                </text>
              ) : null}
            </g>
          ) : null}
        </SvgFrame>
        <TipBox tip={tip} width={size + 220} />
      </div>
      {/* The parts' names and figures, in slice order (clockwise from the top). */}
      <ul className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto] items-baseline gap-x-2 gap-y-1 text-sm">
        {chart.labels.map((label, i) => (
          <li
            key={label}
            className="contents"
            onPointerEnter={() => {
              setActive(i);
            }}
            onPointerLeave={clear}
          >
            <span
              aria-hidden
              className="size-2.5 self-center rounded-[3px]"
              style={{ background: colors[i] ?? QUIET }}
            />
            <span className={active === i ? "text-gray-950" : "text-gray-700"}>{label}</span>
            <span className="text-right font-semibold text-gray-900 tabular-nums">
              {fmt(values[i] ?? 0, chart.unit)}
            </span>
            <span className="text-right text-xs text-gray-500 tabular-nums">
              {chart.unit === "%" ? "" : share(values[i] ?? 0, total)}
            </span>
          </li>
        ))}
        {chart.unit !== "%" ? (
          <li className="contents text-gray-500">
            <span />
            <span className="border-t border-gray-200 pt-1 text-xs">
              Total{chart.unit ? ` ${chart.unit}` : ""}
            </span>
            <span className="border-t border-gray-200 pt-1 text-right text-xs font-semibold tabular-nums">
              {fmt(total, chart.unit)}
            </span>
            <span className="border-t border-gray-200 pt-1" />
          </li>
        ) : null}
      </ul>
    </div>
  );
}

// --------------------------------------------------------------------------- bars

/** A bar from ``base`` to ``end`` along its value axis, its data end rounded (4px) and its base square. */
function barPath(
  horizontal: boolean,
  across: number,
  thick: number,
  base: number,
  end: number,
  round: boolean,
): string {
  const length = Math.abs(end - base);
  const r = round ? Math.min(4, thick / 2, length) : 0;
  const sign = end >= base ? 1 : -1;
  const f = (v: number) => v.toFixed(2);
  if (!horizontal) {
    // Columns: the value axis runs up the page (end < base for a positive value).
    const x0 = across;
    const x1 = across + thick;
    const tip = end;
    const inward = tip - sign * r; // toward the base
    return `M${f(x0)},${f(base)}V${f(inward)}Q${f(x0)},${f(tip)} ${f(x0 + r)},${f(tip)}H${f(x1 - r)}Q${f(x1)},${f(tip)} ${f(x1)},${f(inward)}V${f(base)}Z`;
  }
  const y0 = across;
  const y1 = across + thick;
  const inward = end - sign * r;
  return `M${f(base)},${f(y0)}H${f(inward)}Q${f(end)},${f(y0)} ${f(end)},${f(y0 + r)}V${f(y1 - r)}Q${f(end)},${f(y1)} ${f(inward)},${f(y1)}H${f(base)}Z`;
}

interface BarMark {
  series: number;
  category: number;
  across: number;
  thick: number;
  base: number;
  end: number;
  round: boolean;
  color: string;
  value: number;
  /** A label inside a stacked segment, when it fits. */
  inside: string | null;
}

/** The marks of a bar chart along one value scale: grouped side by side, or stacked with 2px gaps. */
function barMarks(
  chart: Chart,
  colors: string[],
  categoryColors: string[] | null,
  bandStart: (i: number) => number,
  band: number,
  thick: number,
  value: (v: number) => number,
  horizontal: boolean,
): BarMark[] {
  const k = chart.series.length;
  const marks: BarMark[] = [];
  chart.labels.forEach((_, i) => {
    if (chart.stacked) {
      const across = bandStart(i) + (band - thick) / 2;
      const parts = chart.series.map((s, j) => ({ j, v: Math.max(0, s.values[i] ?? 0) }));
      const last = parts.filter((p) => p.v > 0).at(-1)?.j;
      let sum = 0;
      for (const p of parts) {
        if (p.v <= 0) continue;
        const from = value(sum);
        const to = value(sum + p.v);
        sum += p.v;
        // The 2px surface gap: each segment gives up 1px at each end it shares with a neighbour.
        const dir = to >= from ? 1 : -1;
        const gapStart = sum - p.v > 0 ? 1 : 0;
        const gapEnd = p.j !== last ? 1 : 0;
        const base = from + dir * gapStart;
        const end = to - dir * gapEnd;
        if ((end - base) * dir <= 0) continue;
        const color = colors[p.j] ?? QUIET;
        const text = fmt(p.v, chart.unit);
        const length = Math.abs(end - base);
        const fitsInside = horizontal
          ? length >= textWidth(text, 10, 600) + 8 && thick >= 12
          : length >= 16 && thick >= textWidth(text, 10, 600) + 4;
        marks.push({
          series: p.j,
          category: i,
          across,
          thick,
          base,
          end,
          round: p.j === last,
          color,
          value: p.v,
          inside: fitsInside ? text : null,
        });
      }
      return;
    }
    const group = k * thick + (k - 1) * 2;
    chart.series.forEach((s, j) => {
      const v = s.values[i] ?? 0;
      marks.push({
        series: j,
        category: i,
        across: bandStart(i) + (band - group) / 2 + j * (thick + 2),
        thick,
        base: value(0),
        end: value(v),
        round: true,
        color: categoryColors ? (categoryColors[i] ?? QUIET) : (colors[j] ?? QUIET),
        value: v,
        inside: null,
      });
    });
  });
  return marks;
}

function BarChart({ chart, width }: { chart: Chart; width: number }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const k = chart.series.length;
  const n = chart.labels.length;
  const grouped = k > 1 && !chart.stacked;
  // One series: one hue for every bar (grey for an absence or remainder category); several: a hue per series.
  const colors = colorsFor(
    chart.series.map((s) => s.name),
    false,
  );
  const categoryColors = k === 1 ? colorsFor(chart.labels, true) : null;
  const totals = chart.labels.map((_, i) =>
    chart.series.reduce((a, s) => a + Math.max(0, s.values[i] ?? 0), 0),
  );
  const values = chart.stacked ? totals : chart.series.flatMap((s) => s.values);
  const domain = valueDomain(values);
  const ticks = niceTicks(domain, 4);
  const tipRows = (i: number) =>
    chart.series.map((s, j) => ({
      color: categoryColors ? (categoryColors[i] ?? QUIET) : (colors[j] ?? QUIET),
      value: valueText(s.values[i] ?? 0, chart),
      label: s.name || chart.unit,
    }));
  const endLabel = (i: number, j: number) =>
    chart.stacked
      ? valueText(totals[i] ?? 0, chart)
      : valueText(chart.series[j]?.values[i] ?? 0, chart);
  const tickWidth = Math.max(...ticks.map((t) => textWidth(tickLabel(t, chart.unit), 10)), 10);

  // Columns, unless they would be thinner than 8px or a word of a label would not fit its column.
  const columnBand = (width - tickWidth - 16) / n;
  const columnThick = Math.min(
    MAX_BAR,
    grouped ? (columnBand * 0.8 - (k - 1) * 2) / k : columnBand * 0.6,
  );
  const longestWord = Math.max(
    ...chart.labels.flatMap((l) => l.split(/\s+/).map((w) => textWidth(w, 11))),
  );
  const horizontal =
    chart.orientation === "horizontal" || columnThick < 8 || longestWord > columnBand - 4;

  const show = (i: number, x: number, y: number) => {
    setActive(i);
    const label = chart.labels[i] ?? "";
    setTip({
      x,
      y,
      heading: chart.stacked ? `${label} · total ${valueText(totals[i] ?? 0, chart)}` : label,
      rows: tipRows(i),
    });
  };
  const clear = () => {
    setActive(null);
    setTip(null);
  };
  const legend =
    k > 1 ? (
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {chart.series.map((s, j) => (
          <LegendItem key={s.name} color={colors[j] ?? QUIET}>
            {s.name}
          </LegendItem>
        ))}
      </div>
    ) : null;

  if (!horizontal) {
    const lines = chart.labels.map((l) => wrap(l, columnBand - 4, 3));
    const labelBand = Math.max(...lines.map((l) => l.length)) * 13 + 8;
    const m = { top: 20, right: 8, bottom: labelBand, left: tickWidth + 10 };
    const plotH = 200;
    const height = m.top + plotH + m.bottom;
    const band = (width - m.left - m.right) / n;
    const thick = Math.min(MAX_BAR, grouped ? (band * 0.8 - (k - 1) * 2) / k : band * 0.6);
    const y = linear(domain, [m.top + plotH, m.top]);
    const bandStart = (i: number) => m.left + i * band;
    const marks = barMarks(chart, colors, categoryColors, bandStart, band, thick, y, false);
    const slot = grouped ? band / k : band;
    return (
      <div className="flex flex-col gap-2">
        {legend}
        <div className="relative">
          <SvgFrame width={width} height={height} chart={chart} onLeave={clear}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} stroke={INK.grid} />
                <text
                  x={m.left - 6}
                  y={y(t)}
                  dy="0.32em"
                  textAnchor="end"
                  fontSize={10}
                  fill={INK.muted}
                  className="tabular-nums"
                >
                  {tickLabel(t, chart.unit)}
                </text>
              </g>
            ))}
            {chart.unit && chart.unit !== "%" ? (
              <text x={m.left - 6} y={m.top - 8} textAnchor="end" fontSize={10} fill={INK.muted}>
                {fit(chart.unit, m.left + 40, 10)}
              </text>
            ) : null}
            {chart.labels.map((_, i) => (
              <rect
                key={`hit-${String(i)}`}
                x={bandStart(i)}
                y={m.top}
                width={band}
                height={plotH}
                fill={active === i ? INK.grid : "transparent"}
                fillOpacity={0.35}
                onPointerMove={() => {
                  show(i, bandStart(i) + band / 2, m.top + 10);
                }}
              />
            ))}
            {marks.map((b, idx) => (
              <g key={idx} pointerEvents="none">
                <path
                  d={barPath(false, b.across, b.thick, b.base, b.end, b.round)}
                  fill={b.color}
                />
                {b.inside ? (
                  <text
                    x={b.across + b.thick / 2}
                    y={(b.base + b.end) / 2}
                    dy="0.35em"
                    textAnchor="middle"
                    fontSize={10}
                    fontWeight={600}
                    fill={DARK_INK.has(b.color) ? INK_TEXT : SURFACE}
                  >
                    {b.inside}
                  </text>
                ) : null}
              </g>
            ))}
            {/* Values at the bars' tips (a stack's total), when they fit their slot. */}
            {chart.labels.flatMap((_, i) =>
              (chart.stacked ? [0] : chart.series.map((__, j) => j)).map((j) => {
                const v = chart.stacked ? (totals[i] ?? 0) : (chart.series[j]?.values[i] ?? 0);
                let text = endLabel(i, j);
                if (textWidth(text, 11) > slot - 2) text = fmt(v, chart.unit);
                if (textWidth(text, 11) > slot - 2) return null;
                const across = chart.stacked
                  ? bandStart(i) + band / 2
                  : bandStart(i) +
                    (band - (k * thick + (k - 1) * 2)) / 2 +
                    j * (thick + 2) +
                    thick / 2;
                const tipY = y(v);
                return (
                  <text
                    key={`v-${String(i)}-${String(j)}`}
                    x={across}
                    y={v < 0 ? tipY + 13 : tipY - 5}
                    textAnchor="middle"
                    fontSize={11}
                    fill={INK_TEXT}
                    pointerEvents="none"
                    className="tabular-nums"
                  >
                    {text}
                  </text>
                );
              }),
            )}
            <line
              x1={m.left}
              x2={width - m.right}
              y1={y(0)}
              y2={y(0)}
              stroke={INK.axis}
              pointerEvents="none"
            />
            {lines.map((l, i) => (
              <text
                key={`l-${String(i)}`}
                x={bandStart(i) + band / 2}
                y={m.top + plotH + 14}
                textAnchor="middle"
                fontSize={11}
                fill="#4b5563"
              >
                <title>{chart.labels[i]}</title>
                {l.map((part, p) => (
                  <tspan key={p} x={bandStart(i) + band / 2} dy={p ? 13 : 0}>
                    {part}
                  </tspan>
                ))}
              </text>
            ))}
          </SvgFrame>
          <TipBox tip={tip} width={width} />
        </div>
      </div>
    );
  }

  // Horizontal bars: category names on the left, values at the bars' tips.
  const thick = grouped ? 12 : 18;
  const rowH = (grouped ? k * thick + (k - 1) * 2 : thick) + 12;
  const labelW = Math.min(
    Math.max(...chart.labels.map((l) => textWidth(l, 11))) + 10,
    Math.max(90, width * 0.34),
    260,
  );
  const valueW =
    Math.max(
      ...chart.labels.flatMap((_, i) =>
        (chart.stacked ? [0] : chart.series.map((__, j) => j)).map((j) =>
          textWidth(endLabel(i, j), 11),
        ),
      ),
    ) + 10;
  const m = { top: 4, right: valueW, bottom: chart.unit ? 34 : 22, left: labelW };
  const height = m.top + rowH * n + m.bottom;
  const x = linear(domain, [m.left, width - m.right]);
  const bandStart = (i: number) => m.top + i * rowH;
  const marks = barMarks(chart, colors, categoryColors, bandStart, rowH, thick, x, true);
  const plotBottom = m.top + rowH * n;
  return (
    <div className="flex flex-col gap-2">
      {legend}
      <div className="relative">
        <SvgFrame width={width} height={height} chart={chart} onLeave={clear}>
          {niceTicks(
            domain,
            Math.max(2, Math.min(5, Math.floor((width - m.left - m.right) / 70))),
          ).map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.top} y2={plotBottom} stroke={INK.grid} />
              <text
                x={x(t)}
                y={plotBottom + 13}
                textAnchor="middle"
                fontSize={10}
                fill={INK.muted}
                className="tabular-nums"
              >
                {tickLabel(t, chart.unit)}
              </text>
            </g>
          ))}
          {chart.unit && chart.unit !== "%" ? (
            <text
              x={(m.left + width - m.right) / 2}
              y={plotBottom + 28}
              textAnchor="middle"
              fontSize={10}
              fill={INK.muted}
            >
              {chart.unit}
            </text>
          ) : null}
          {chart.labels.map((label, i) => (
            <g key={`row-${String(i)}`}>
              <rect
                x={0}
                y={bandStart(i)}
                width={width}
                height={rowH}
                fill={active === i ? INK.grid : "transparent"}
                fillOpacity={0.35}
                onPointerMove={() => {
                  show(i, Math.min(x(totals[i] ?? 0), width - 230), bandStart(i) + 4);
                }}
              />
              <text
                x={m.left - 8}
                y={bandStart(i) + rowH / 2}
                dy="0.35em"
                textAnchor="end"
                fontSize={11}
                fill="#4b5563"
                pointerEvents="none"
              >
                <title>{label}</title>
                {fit(label, m.left - 12)}
              </text>
            </g>
          ))}
          {marks.map((b, idx) => (
            <g key={idx} pointerEvents="none">
              <path d={barPath(true, b.across, b.thick, b.base, b.end, b.round)} fill={b.color} />
              {b.inside ? (
                <text
                  x={(b.base + b.end) / 2}
                  y={b.across + b.thick / 2}
                  dy="0.35em"
                  textAnchor="middle"
                  fontSize={10}
                  fontWeight={600}
                  fill={DARK_INK.has(b.color) ? INK_TEXT : SURFACE}
                >
                  {b.inside}
                </text>
              ) : null}
            </g>
          ))}
          {chart.labels.flatMap((_, i) =>
            (chart.stacked ? [0] : chart.series.map((__, j) => j)).map((j) => {
              const v = chart.stacked ? (totals[i] ?? 0) : (chart.series[j]?.values[i] ?? 0);
              const across = chart.stacked
                ? bandStart(i) + rowH / 2
                : bandStart(i) +
                  (rowH - (k * thick + (k - 1) * 2)) / 2 +
                  j * (thick + 2) +
                  thick / 2;
              return (
                <text
                  key={`v-${String(i)}-${String(j)}`}
                  x={v < 0 ? x(v) - 5 : x(v) + 5}
                  y={across}
                  dy="0.35em"
                  textAnchor={v < 0 ? "end" : "start"}
                  fontSize={11}
                  fill={INK_TEXT}
                  pointerEvents="none"
                  className="tabular-nums"
                >
                  {endLabel(i, j)}
                </text>
              );
            }),
          )}
          <line
            x1={x(0)}
            x2={x(0)}
            y1={m.top}
            y2={plotBottom}
            stroke={INK.axis}
            pointerEvents="none"
          />
        </SvgFrame>
        <TipBox tip={tip} width={width} />
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- line

function LineChart({ chart, width }: { chart: Chart; width: number }) {
  const [active, setActive] = useState<number | null>(null);
  const k = chart.series.length;
  const n = chart.labels.length;
  const colors = colorsFor(
    chart.series.map((s) => s.name),
    false,
  );
  const values = chart.series.flatMap((s) => s.values);
  const domain = valueDomain(values, Math.min(...values) >= 0);
  const ticks = niceTicks(domain, 4);
  const tickWidth = Math.max(...ticks.map((t) => textWidth(tickLabel(t, chart.unit), 10)), 10);
  // Direct labels: every point of a single short series; else each series' last value (and name), unless they collide.
  const everyPoint = k === 1 && n <= 12;
  const endTexts = chart.series.map(
    (s) => `${k > 1 ? `${s.name} ` : ""}${fmt(s.values[n - 1] ?? 0, chart.unit)}`,
  );
  const plotH = 200;
  const m = {
    top: 16,
    right: everyPoint ? 16 : Math.max(...endTexts.map((t) => textWidth(t, 11))) + 14,
    bottom: chart.unit ? 40 : 28,
    left: tickWidth + 10,
  };
  const height = m.top + plotH + m.bottom;
  const plotW = width - m.left - m.right;
  const x = (i: number) => m.left + (n === 1 ? plotW / 2 : (i * plotW) / (n - 1));
  const y = linear(domain, [m.top + plotH, m.top]);
  const spacing = n > 1 ? plotW / (n - 1) : plotW;
  const labelEvery = Math.max(
    1,
    Math.ceil((Math.max(...chart.labels.map((l) => textWidth(l, 11))) + 10) / spacing),
  );
  const ends = chart.series
    .map((s, j) => ({ j, y: y(s.values[n - 1] ?? 0) }))
    .sort((a, b) => a.y - b.y);
  const endsFit = ends.every((e, i) => i === 0 || e.y - (ends[i - 1]?.y ?? 0) >= 13);
  const tip: Tip | null =
    active === null
      ? null
      : {
          x: x(active),
          y: m.top + 8,
          heading: chart.labels[active],
          rows: chart.series.map((s, j) => ({
            color: colors[j] ?? QUIET,
            value: fmt(s.values[active] ?? 0, chart.unit),
            label: s.name || chart.unit,
          })),
        };
  return (
    <div className="flex flex-col gap-2">
      {k > 1 ? (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {chart.series.map((s, j) => (
            <LegendItem key={s.name} color={colors[j] ?? QUIET} shape="line">
              {s.name}
            </LegendItem>
          ))}
        </div>
      ) : null}
      <div className="relative">
        <SvgFrame
          width={width}
          height={height}
          chart={chart}
          onLeave={() => {
            setActive(null);
          }}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={m.left} x2={m.left + plotW} y1={y(t)} y2={y(t)} stroke={INK.grid} />
              <text
                x={m.left - 6}
                y={y(t)}
                dy="0.32em"
                textAnchor="end"
                fontSize={10}
                fill={INK.muted}
                className="tabular-nums"
              >
                {tickLabel(t, chart.unit)}
              </text>
            </g>
          ))}
          <line
            x1={m.left}
            x2={m.left + plotW}
            y1={y(domain[0])}
            y2={y(domain[0])}
            stroke={INK.axis}
          />
          {chart.labels.map((label, i) =>
            i % labelEvery === 0 ? (
              <text
                key={`x-${String(i)}`}
                x={x(i)}
                y={m.top + plotH + 15}
                textAnchor="middle"
                fontSize={11}
                fill="#4b5563"
              >
                <title>{label}</title>
                {fit(label, Math.max(spacing * labelEvery - 6, 30))}
              </text>
            ) : null,
          )}
          {chart.unit && chart.unit !== "%" ? (
            <text x={m.left - 6} y={m.top - 6} textAnchor="end" fontSize={10} fill={INK.muted}>
              {fit(chart.unit, m.left + 40, 10)}
            </text>
          ) : null}
          {active !== null ? (
            <line
              x1={x(active)}
              x2={x(active)}
              y1={m.top}
              y2={m.top + plotH}
              stroke={INK.axis}
              pointerEvents="none"
            />
          ) : null}
          {chart.series.map((s, j) => (
            <g key={s.name || j} pointerEvents="none">
              <path
                d={s.values
                  .map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
                  .join("")}
                fill="none"
                stroke={colors[j] ?? QUIET}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {s.values.map((v, i) => (
                <circle
                  key={i}
                  cx={x(i)}
                  cy={y(v)}
                  r={active === i ? 5 : 4}
                  fill={colors[j] ?? QUIET}
                  stroke={SURFACE}
                  strokeWidth={2}
                  paintOrder="stroke"
                />
              ))}
              {everyPoint
                ? s.values.map((v, i) => (
                    <text
                      key={`t-${String(i)}`}
                      x={x(i)}
                      y={y(v) - 9}
                      textAnchor="middle"
                      fontSize={11}
                      fill={INK_TEXT}
                      className="tabular-nums"
                    >
                      {fmt(v, chart.unit)}
                    </text>
                  ))
                : endsFit
                  ? [
                      <text
                        key="end"
                        x={x(n - 1) + 8}
                        y={y(s.values[n - 1] ?? 0)}
                        dy="0.35em"
                        fontSize={11}
                        fill={INK_TEXT}
                        className="tabular-nums"
                      >
                        {endTexts[j]}
                      </text>,
                    ]
                  : null}
            </g>
          ))}
          {/* The crosshair's hit area: the pointer finds the nearest x. */}
          <rect
            x={m.left - spacing / 2}
            y={m.top}
            width={plotW + spacing}
            height={plotH}
            fill="transparent"
            onPointerMove={(e) => {
              const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
              if (!box) return;
              const px = ((e.clientX - box.left) * width) / box.width;
              const i = n === 1 ? 0 : Math.round(((px - m.left) / plotW) * (n - 1));
              setActive(Math.max(0, Math.min(n - 1, i)));
            }}
          />
        </SvgFrame>
        <TipBox tip={tip} width={width} />
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- table, CSV and the figure

function tableRows(chart: Chart): { header: string[]; rows: string[][] } {
  const pie = isPie(chart);
  const total = pie ? pieTotal(chart) : chart.total;
  const shares = hasShares(chart) && total > 0;
  const name = (s: { name: string }) => s.name || chart.unit || "Value";
  const header = [
    chart.kind === "line" ? "Label" : "Category",
    ...chart.series.flatMap((s) =>
      shares ? [name(s), `${name(s)} (% of ${fmt(total, "")})`] : [name(s)],
    ),
  ];
  const rows = chart.labels.map((label, i) => [
    label,
    ...chart.series.flatMap((s) => {
      const v = s.values[i] ?? 0;
      return shares ? [String(v), share(v, total)] : [String(v)];
    }),
  ]);
  return { header, rows };
}

/** A CSV cell: quoted when needed; text that a spreadsheet would read as a formula is prefixed with '. */
function csvCell(value: string, text: boolean): string {
  const safe = text && /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

function downloadCsv(chart: Chart) {
  const { header, rows } = tableRows(chart);
  const lines = [
    header.map((h) => csvCell(h, true)).join(","),
    ...rows.map((r) => r.map((c, i) => csvCell(c, i === 0)).join(",")),
  ];
  const slug =
    chart.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 60) || "chart";
  saveBlob(
    new Blob([`${lines.join("\r\n")}\r\n`], { type: "text/csv;charset=utf-8" }),
    `${slug}.csv`,
  );
}

function DataTable({ chart }: { chart: Chart }) {
  const { header, rows } = tableRows(chart);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <caption className="sr-only">{chart.title}</caption>
        <thead>
          <tr className="border-b border-gray-200 text-gray-600">
            {header.map((h, i) => (
              <th
                key={i}
                scope="col"
                className={`py-1 pr-3 font-medium ${i ? "text-right" : "text-left"}`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-b border-gray-100 last:border-0">
              {row.map((cell, i) =>
                i ? (
                  <td key={i} className="py-1 pr-3 text-right tabular-nums">
                    {cell}
                  </td>
                ) : (
                  <th key={i} scope="row" className="py-1 pr-3 text-left font-normal text-gray-700">
                    {cell}
                  </th>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sourceText(chart: Chart, steps: Map<number, Step>): string {
  if (!chart.steps.length) return "Numbers from the data overview.";
  const parts = chart.steps.map((n) => {
    const step = steps.get(n);
    return step ? `step ${String(n)} (${step.label})` : `step ${String(n)}`;
  });
  return `Numbers from ${parts.join(", ")}.`;
}

function ChartFigure({ chart, steps }: { chart: Chart; steps: Map<number, Step> }) {
  const [table, setTable] = useState(false);
  const [ref, width] = useWidth();
  return (
    <figure className="flex break-inside-avoid flex-col gap-3 rounded-md border border-gray-200 p-3">
      <figcaption className="flex flex-col gap-0.5">
        <span className="text-sm font-semibold text-gray-900">{chart.title}</span>
        {chart.caption ? <span className="text-xs text-gray-600">{chart.caption}</span> : null}
      </figcaption>
      <div ref={ref} className="min-w-0">
        {isPie(chart) ? (
          <PieChart chart={chart} />
        ) : chart.kind === "line" ? (
          <LineChart chart={chart} width={width} />
        ) : (
          <BarChart chart={chart} width={width} />
        )}
      </div>
      {table ? <DataTable chart={chart} /> : null}
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-auto text-xs text-gray-500">{sourceText(chart, steps)}</span>
        <Button
          size="xs"
          variant="outline"
          aria-pressed={table}
          className="print:hidden"
          onClick={() => {
            setTable(!table);
          }}
        >
          <TableIcon />
          {table ? "Hide table" : "Table"}
        </Button>
        <Button
          size="xs"
          variant="outline"
          className="print:hidden"
          onClick={() => {
            downloadCsv(chart);
          }}
        >
          <DownloadIcon />
          CSV
        </Button>
      </div>
    </figure>
  );
}

/** An answer's charts, below its Markdown. ``steps``: the conversation's tool steps by number (the sources). */
export function AnswerCharts({ charts, steps }: { charts: Chart[]; steps: Map<number, Step> }) {
  return (
    <div className="flex flex-col gap-3">
      {charts.map((chart, i) => (
        <ChartFigure key={i} chart={chart} steps={steps} />
      ))}
    </div>
  );
}
