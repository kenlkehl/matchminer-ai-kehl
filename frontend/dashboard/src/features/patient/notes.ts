/** The patient's notes as the guideline APIs take them, and where a quote sits in a note. */
import type { Note } from "@/lib/api";

import type { GuidelineNote, NotesInput, SavedCompression } from "./types";

/** The server's limit on a request's note text (routes/guidelines.py, NotesRequest). */
export const MAX_NOTE_CHARACTERS = 2_000_000;

/** Notes joined into one text with date headers, as the server's upload reader joins a CSV's notes. */
export function joinedNotes(notes: GuidelineNote[]): string {
  return notes
    .map((n) => `[Note date: ${n.note_date || "unavailable"}]\n${n.note_text}`)
    .join("\n\n");
}

/** A corpus patient's notes (`/api/patients/{id}/notes`) in the guideline APIs' shape. */
export function corpusNotes(rows: Note[]): NotesInput {
  const notes: GuidelineNote[] = rows
    .filter((r) => r.text.trim())
    .map((r) => ({
      note_text: r.text,
      note_date: r.date || null,
      ...(r.type?.trim() ? { note_type: r.type } : {}),
    }));
  return { text: joinedNotes(notes), notes: notes.length ? notes : null };
}

const normalize = (rows: GuidelineNote[] | null | undefined) =>
  rows?.length
    ? rows.map((row) => ({
        note_text: row.note_text,
        note_date: row.note_date ?? null,
        note_type: row.note_type ?? null,
      }))
    : null;

/** Whether the saved compression was made from exactly these notes (text, rows, dates and types). */
export function savedCompressionMatches(
  saved: SavedCompression | null | undefined,
  text: string,
  notes: GuidelineNote[] | null,
): saved is SavedCompression {
  if (!saved || saved.notes_input.text !== text) return false;
  return JSON.stringify(normalize(saved.notes_input.notes)) === JSON.stringify(normalize(notes));
}

/** The notes as rows: the separate notes, else the whole text as one undated note. */
export function noteRows(text: string, notes: GuidelineNote[] | null): GuidelineNote[] {
  if (notes?.length) return notes;
  return text.trim() ? [{ note_text: text, note_date: null }] : [];
}

/** Note `number` (1-based, as reviews cite them), or null when it is not in the current notes. */
export function noteByNumber(
  text: string,
  notes: GuidelineNote[] | null,
  number: number,
): GuidelineNote | null {
  return noteRows(text, notes)[number - 1] ?? null;
}

/** A note's text split around one span: [before, marked, after], or null when the span is invalid. */
export function splitSpan(
  text: string,
  start: number | null | undefined,
  end: number | null | undefined,
): [string, string, string] | null {
  if (
    typeof start !== "number" ||
    typeof end !== "number" ||
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    start >= end ||
    end > text.length
  ) {
    return null;
  }
  return [text.slice(0, start), text.slice(start, end), text.slice(end)];
}

/** Where a quote first appears verbatim in a note (a full-note review cites quotes without offsets). */
export function findQuote(text: string, quote: string): { start: number; end: number } | null {
  const trimmed = quote.trim();
  if (!trimmed) return null;
  const start = text.indexOf(trimmed);
  return start < 0 ? null : { start, end: start + trimmed.length };
}

/** A note date for display: the date part of an ISO date-time. */
export function noteDate(value: string | null | undefined): string {
  if (!value) return "Date unavailable";
  return value.replace(/T00:00:00(\.0+)?$/, "").replace("T", " ");
}
