/**
 * The adjudication page's left pane: the item's source text (a ctgov listing, a patient's notes, or a decision
 * task) with search, a jump list of note dates, and the judges' evidence quotes highlighted on request. Text is
 * rendered as React nodes split around the marks, never as HTML.
 */
import { cn } from "cn";
import { memo, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { Input } from "@/components/ui/input";

import type { ItemSource } from "./types";

export interface Highlight {
  quote: string;
  date: string | null;
  /** Changes on every click, so clicking the same quote again scrolls to it again. */
  nonce: number;
}

interface Block {
  date: string | null;
  text: string;
}

type MarkKind = "quote" | "search" | "current";

interface Mark {
  start: number;
  end: number;
  kind: MarkKind;
}

/** Scroll ``box`` (only) so that ``el`` is in view: unlike scrollIntoView, the page itself stays put. */
export function scrollWithin(
  box: HTMLElement,
  el: HTMLElement,
  block: "start" | "center" | "nearest",
  behavior: ScrollBehavior = "auto",
) {
  const outer = box.getBoundingClientRect();
  const inner = el.getBoundingClientRect();
  const offset = inner.top - outer.top + box.scrollTop;
  let top: number;
  if (block === "start") top = offset;
  else if (block === "center") top = offset - (box.clientHeight - inner.height) / 2;
  else if (inner.top < outer.top) top = offset - 8;
  else if (inner.bottom > outer.bottom) top = offset - box.clientHeight + inner.height + 8;
  else return;
  box.scrollTo({ top: Math.max(0, top), behavior });
}

const NOTE_HEADER = /=== Clinical Note dated (.+?) ===\n?/;
const MAX_MATCHES = 5000;

export function splitSource(source: ItemSource): Block[] {
  if (source.kind !== "notes") return [{ date: null, text: source.text }];
  const parts = source.text.split(NOTE_HEADER);
  const blocks: Block[] = [];
  if (parts[0]?.trim()) blocks.push({ date: null, text: parts[0] });
  for (let i = 1; i < parts.length; i += 2) {
    blocks.push({ date: parts[i] ?? "", text: parts[i + 1] ?? "" });
  }
  return blocks;
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** A case-insensitive pattern for ``text`` in which any run of whitespace matches any other. */
function pattern(text: string, flags: string): RegExp | null {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length ? new RegExp(words.map(escape).join("\\s+"), flags) : null;
}

/** The quote's pieces ('...' joins pieces in order), found in one block: the quote's dated note first. */
function findQuote(blocks: Block[], highlight: Highlight): Map<number, Mark[]> | null {
  const pieces = highlight.quote
    .split(/\.\.\.|…/)
    .map((p) => p.trim())
    .filter(Boolean);
  const first = pieces[0] ? pattern(pieces[0], "i") : null;
  if (!first) return null;
  const order = blocks
    .map((b, i) => ({ b, i }))
    .sort((x, y) => Number(y.b.date === highlight.date) - Number(x.b.date === highlight.date));
  for (const { b, i } of order) {
    const m = first.exec(b.text);
    if (!m) continue;
    const marks: Mark[] = [{ start: m.index, end: m.index + m[0].length, kind: "quote" }];
    let from = m.index + m[0].length;
    for (const piece of pieces.slice(1)) {
      const re = pattern(piece, "gi");
      if (!re) continue;
      re.lastIndex = from;
      const next = re.exec(b.text);
      if (!next) break;
      marks.push({ start: next.index, end: next.index + next[0].length, kind: "quote" });
      from = next.index + next[0].length;
    }
    return new Map([[i, marks]]);
  }
  return null;
}

interface Match {
  block: number;
  start: number;
  end: number;
}

function findAll(blocks: Block[], term: string): Match[] {
  const out: Match[] = [];
  blocks.forEach((b, block) => {
    const re = pattern(term, "gi");
    if (!re) return;
    for (let m = re.exec(b.text); m && out.length < MAX_MATCHES; m = re.exec(b.text)) {
      if (!m[0].length) {
        re.lastIndex += 1;
        continue;
      }
      out.push({ block, start: m.index, end: m.index + m[0].length });
    }
  });
  return out;
}

const MARK_CLASS: Record<MarkKind, string> = {
  quote: "rounded-sm bg-amber-200 text-gray-950 ring-2 ring-amber-400",
  search: "rounded-sm bg-yellow-100 text-gray-950",
  current: "rounded-sm bg-orange-300 text-gray-950",
};

const SourceBlock = memo(function SourceBlock({
  block,
  index,
  marks,
}: {
  block: Block;
  index: number;
  marks: Mark[] | undefined;
}) {
  let body: ReactNode = block.text;
  if (marks?.length) {
    const sorted = [...marks].sort((a, b) => a.start - b.start);
    const nodes: ReactNode[] = [];
    let at = 0;
    for (const [n, m] of sorted.entries()) {
      if (m.start < at) continue; // overlaps an earlier mark (quote marks are listed first)
      if (m.start > at) nodes.push(block.text.slice(at, m.start));
      nodes.push(
        <mark
          key={n}
          className={MARK_CLASS[m.kind]}
          data-target={m.kind === "quote" || m.kind === "current" ? m.kind : undefined}
        >
          {block.text.slice(m.start, m.end)}
        </mark>,
      );
      at = m.end;
    }
    nodes.push(block.text.slice(at));
    body = nodes;
  }
  return (
    <div data-block={index}>
      {block.date !== null ? (
        <h4 className="sticky top-0 z-[1] -mx-4 border-b border-gray-100 bg-white/95 px-4 py-1 text-xs font-semibold text-brand-blue">
          {block.date}
        </h4>
      ) : null}
      <pre className="m-0 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap text-gray-800">
        {body}
      </pre>
    </div>
  );
});

export function SourcePane({
  source,
  highlight,
}: {
  source: ItemSource;
  highlight: Highlight | null;
}) {
  const { kind, text } = source;
  const blocks = useMemo(() => splitSource({ kind, text }), [kind, text]);
  const [draft, setDraft] = useState("");
  const [term, setTerm] = useState("");
  const [current, setCurrent] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const matches = useMemo(() => (term ? findAll(blocks, term) : []), [blocks, term]);
  const quote = useMemo(
    () => (highlight ? findQuote(blocks, highlight) : null),
    [blocks, highlight],
  );
  const dated = blocks.flatMap((b, i) => (b.date !== null ? [{ date: b.date, i }] : []));

  // A new item: a fresh search.
  useEffect(() => {
    setDraft("");
    setTerm("");
    setCurrent(0);
    box.current?.scrollTo({ top: 0 });
  }, [kind, text]);

  const marks = useMemo(() => {
    const out = new Map<number, Mark[]>();
    for (const [i, list] of quote ?? []) out.set(i, [...list]);
    matches.forEach((m, n) => {
      const list = out.get(m.block) ?? [];
      list.push({ start: m.start, end: m.end, kind: n === current ? "current" : "search" });
      out.set(m.block, list);
    });
    return out;
  }, [quote, matches, current]);

  // Scroll to a clicked quote, or to the current search match.
  useEffect(() => {
    const el = box.current?.querySelector<HTMLElement>('[data-target="quote"]');
    if (box.current && el) scrollWithin(box.current, el, "center", "smooth");
  }, [quote, highlight?.nonce]);
  useEffect(() => {
    if (!matches.length) return;
    const el = box.current?.querySelector<HTMLElement>('[data-target="current"]');
    if (box.current && el) scrollWithin(box.current, el, "center");
  }, [matches, current]);

  const go = (step: number) => {
    if (!matches.length) return;
    setCurrent((c) => (c + step + matches.length) % matches.length);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b bg-gray-50 px-3 py-2">
        <Input
          value={draft}
          placeholder="Search the source (Enter)"
          aria-label="Search the source"
          className="h-8 min-w-40 flex-1 bg-white"
          onChange={(e) => {
            setDraft(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              if (draft.trim() !== term) {
                setTerm(draft.trim());
                setCurrent(0);
              } else go(e.shiftKey ? -1 : 1);
            }
          }}
        />
        {term ? (
          <span className="text-xs whitespace-nowrap text-gray-600 tabular-nums">
            {matches.length
              ? `${String(current + 1)} of ${matches.length >= MAX_MATCHES ? `${String(MAX_MATCHES)}+` : String(matches.length)}`
              : "no matches"}
          </span>
        ) : null}
        {dated.length ? (
          <select
            aria-label="Jump to a note"
            className="h-8 max-w-48 rounded-md border border-input bg-white px-2 text-sm"
            value=""
            onChange={(e) => {
              const el = box.current?.querySelector<HTMLElement>(
                `[data-block="${e.target.value}"]`,
              );
              if (box.current && el) scrollWithin(box.current, el, "start");
            }}
          >
            <option value="" disabled>
              {dated.length} notes: jump to…
            </option>
            {dated.map(({ date, i }) => (
              <option key={i} value={i}>
                {date}
              </option>
            ))}
          </select>
        ) : null}
        {highlight && !quote ? (
          <span className="text-xs text-red-700">The quote was not found in the source.</span>
        ) : null}
      </div>
      <div
        ref={box}
        className={cn("scrollbar-visible min-h-0 flex-1 overflow-auto px-4 pb-4", "bg-white")}
      >
        {blocks.map((b, i) => (
          <SourceBlock key={i} block={b} index={i} marks={marks.get(i)} />
        ))}
      </div>
    </div>
  );
}
