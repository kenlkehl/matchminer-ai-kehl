import { useLayoutEffect, useRef, useState } from "react";

import type { ManagedUnit } from "@/lib/management";
import { slotText } from "@/lib/management";

const MAX_NODE_W = 236;
const MIN_NODE_W = 104;
const NODE_H = 46;
const ROW = NODE_H + 12;
const HEAD = 22;
const BLOCK_GAP = 18;
/** Below this node width the slot counts are shown as "available · consented · enrolled" numbers only. */
const COMPACT_W = 200;

interface Placed {
  unit: ManagedUnit;
  x: number;
  y: number;
}

const STROKE: Record<ManagedUnit["kind"], string> = {
  merged: "#1d4ed8",
  original: "#6b7280",
  subcohort: "#0f766e",
};

function short(text: string, n: number) {
  return text.length > n ? `${text.slice(0, Math.max(1, n - 1))}…` : text;
}

/** The element's width, followed as it is resized (the Trial management sidebar can be). */
function useWidth() {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      setWidth(element.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, []);
  return { ref, width };
}

/**
 * Where a trial's cohorts come from: the extracted (original) cohorts merged into each dashboard cohort, and the
 * subcohorts users defined under a merged or an original cohort. One block per merged cohort; originals on the
 * left, the merged cohort (and the subcohorts of its originals) in the middle, its own subcohorts on the right.
 * Nodes fit the space it has (a name is shortened, in full on hover). A unit with slots entered is shaded; each
 * unit's slots are its own.
 */
export function LineageDiagram({
  units,
  selected,
  onSelect,
}: {
  units: ManagedUnit[];
  selected: string | null;
  onSelect: (unitId: string) => void;
}) {
  const { ref, width: available } = useWidth();
  const byId = new Map(units.map((u) => [u.id, u]));
  const merged = units.filter((u) => u.kind === "merged");
  const hasOriginals = merged.some((m) => m.originals.length > 0);
  const columns = hasOriginals ? 3 : 2;
  const gap = available && available < 560 ? 22 : 44;
  const nodeW = Math.round(
    Math.min(
      MAX_NODE_W,
      Math.max(MIN_NODE_W, ((available || 800) - (columns - 1) * gap) / columns),
    ),
  );
  const compact = nodeW < COMPACT_W;
  const nameChars = Math.floor((nodeW - 16) / 6.4);
  const x = (column: number) => (hasOriginals ? column : column - 1) * (nodeW + gap);
  const nodes: Placed[] = [];
  const edges: { from: Placed; to: Placed; kind: "merge" | "derive" }[] = [];
  let top = HEAD;
  for (const m of merged) {
    const originals = m.originals.map((id) => byId.get(id)).filter((u) => u !== undefined);
    const fromOriginals = originals.flatMap((o) =>
      o.children.map((id) => byId.get(id)).filter((u) => u !== undefined),
    );
    const own = m.children.map((id) => byId.get(id)).filter((u) => u !== undefined);
    const rows = Math.max(originals.length, 1 + fromOriginals.length, own.length, 1);
    const place = (unit: ManagedUnit, column: number, row: number) => {
      const node = { unit, x: x(column), y: top + row * ROW };
      nodes.push(node);
      return node;
    };
    const center = place(m, 1, 0);
    const placedOriginals = new Map(originals.map((o, i) => [o.id, place(o, 0, i)]));
    for (const o of originals)
      edges.push({ from: placedOriginals.get(o.id)!, to: center, kind: "merge" });
    fromOriginals.forEach((s, i) => {
      const node = place(s, 1, i + 1);
      const parent = placedOriginals.get(s.parent ?? "");
      if (parent) edges.push({ from: parent, to: node, kind: "derive" });
    });
    own.forEach((s, i) => {
      edges.push({ from: center, to: place(s, 2, i), kind: "derive" });
    });
    top += rows * ROW + BLOCK_GAP;
  }
  const width = columns * nodeW + (columns - 1) * gap;
  const height = top - BLOCK_GAP + 4;
  const headings = hasOriginals
    ? [compact ? "Originals" : "Original cohorts (extracted)", "Merged cohort", "Subcohorts"]
    : ["Cohort", "Subcohorts"];
  const counts = (unit: ManagedUnit) => {
    const s = unit.slots;
    if (!s) return "no slots";
    if (!compact) return slotText(s);
    const n = (v: number | null) => (v === null ? "–" : String(v));
    return `${n(s.available)} · ${n(s.consented)} · ${n(s.enrolled)}`;
  };

  return (
    <figure ref={ref} className="flex flex-col gap-1">
      <svg
        viewBox={`0 0 ${String(width)} ${String(height)}`}
        className="h-auto w-full"
        style={{ maxWidth: width }}
        role="img"
        aria-label="Where this trial's cohorts come from: original cohorts, merged cohorts and subcohorts"
      >
        {headings.map((label, i) => (
          <text
            key={label}
            x={i * (nodeW + gap)}
            y={12}
            className="fill-gray-500 text-[11px] font-semibold"
          >
            {label}
          </text>
        ))}
        {edges.map((e) => {
          const x1 = e.from.x + nodeW;
          const y1 = e.from.y + NODE_H / 2;
          const x2 = e.to.x;
          const y2 = e.to.y + NODE_H / 2;
          const mid = (x1 + x2) / 2;
          return (
            <path
              key={`${e.from.unit.id}-${e.to.unit.id}`}
              d={`M${String(x1)},${String(y1)} C${String(mid)},${String(y1)} ${String(mid)},${String(y2)} ${String(x2)},${String(y2)}`}
              fill="none"
              stroke={e.kind === "merge" ? "#9ca3af" : "#0f766e"}
              strokeWidth={1.5}
              strokeDasharray={e.kind === "derive" ? "5 3" : undefined}
              markerEnd="url(#lineage-arrow)"
            />
          );
        })}
        <defs>
          <marker
            id="lineage-arrow"
            viewBox="0 0 8 8"
            refX={7}
            refY={4}
            markerWidth={6}
            markerHeight={6}
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 z" fill="#6b7280" />
          </marker>
        </defs>
        {nodes.map(({ unit, x: nx, y }) => {
          const slots = slotText(unit.slots);
          const notes = [
            unit.edited ? "criteria edited" : null,
            unit.stale ? "its base changed since the edit" : null,
            unit.unclassified ? `${String(unit.unclassified)} lines unclassified` : null,
          ].filter(Boolean);
          const marks = [unit.edited ? "✎" : "", unit.stale ? "⚠" : ""].join("");
          return (
            <g
              key={unit.id}
              transform={`translate(${String(nx)},${String(y)})`}
              className="cursor-pointer"
              onClick={() => {
                onSelect(unit.id);
              }}
            >
              <title>
                {`${unit.name}${slots ? ` — slots: ${slots}` : " — no slots entered"}${notes.length ? ` — ${notes.join("; ")}` : ""}`}
              </title>
              <rect
                width={nodeW}
                height={NODE_H}
                rx={6}
                fill={unit.slots ? "#ecfdf5" : "#ffffff"}
                stroke={STROKE[unit.kind]}
                strokeWidth={selected === unit.id ? 2.5 : 1.2}
              />
              <text x={8} y={17} className="fill-gray-900 text-[11.5px] font-semibold">
                {short(unit.name, marks ? nameChars - 3 : nameChars)}
              </text>
              <text
                x={8}
                y={34}
                className={
                  unit.slots ? "fill-emerald-800 text-[10.5px]" : "fill-gray-400 text-[10.5px]"
                }
              >
                {short(counts(unit), Math.floor((nodeW - 16) / 5.6))}
              </text>
              {marks ? (
                <text x={nodeW - 8} y={17} textAnchor="end" className="fill-amber-700 text-[12px]">
                  {marks}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <figcaption className="text-xs text-gray-500">
        Grey lines: extracted cohorts merged into one cohort with the same criteria. Dashed lines:
        subcohorts defined from a cohort. Shaded: slots entered
        {compact ? ", shown as available · consented · enrolled" : ""} (each cohort's slots are its
        own and are never added up).
      </figcaption>
    </figure>
  );
}
