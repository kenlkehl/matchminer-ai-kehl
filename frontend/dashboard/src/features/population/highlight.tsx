import type { ReactNode } from "react";

/** A character span of a note's text (offsets index the note's own text). */
export interface Span {
  start: number;
  end: number;
}

/** Whether `span` lies inside a text of this length. */
function validSpan(span: Span | null | undefined, length: number): span is Span {
  return (
    !!span &&
    Number.isInteger(span.start) &&
    Number.isInteger(span.end) &&
    span.start >= 0 &&
    span.start < span.end &&
    span.end <= length
  );
}

/** Where a quote appears in a text, exactly or ignoring differences in whitespace; null if it does not. */
export function locateQuote(text: string, quote: string): Span | null {
  const needle = quote.trim();
  if (!needle) return null;
  const exact = text.indexOf(needle);
  if (exact >= 0) return { start: exact, end: exact + needle.length };
  // Collapse whitespace runs in the text, keeping each kept character's original offset.
  const positions: number[] = [];
  let collapsed = "";
  let space = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charAt(i);
    if (/\s/.test(ch)) {
      if (!space && collapsed) {
        collapsed += " ";
        positions.push(i);
      }
      space = true;
    } else {
      collapsed += ch;
      positions.push(i);
      space = false;
    }
  }
  const target = needle.replace(/\s+/g, " ");
  const found = collapsed.indexOf(target);
  if (found < 0) return null;
  const first = positions[found];
  const last = positions[found + target.length - 1];
  return first === undefined || last === undefined ? null : { start: first, end: last + 1 };
}

/** A note's text with one span marked, as React nodes (never HTML). */
export function HighlightedText({ text, span }: { text: string; span: Span | null }): ReactNode {
  if (!validSpan(span, text.length)) return text;
  return (
    <>
      {text.slice(0, span.start)}
      <mark className="rounded-sm bg-brand-orange/40 text-inherit">
        {text.slice(span.start, span.end)}
      </mark>
      {text.slice(span.end)}
    </>
  );
}
