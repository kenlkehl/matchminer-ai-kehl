import type { ReactNode } from "react";

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { Helper } from "./controls";
import type { Span } from "./highlight";
import {
  APPLICABILITY_LABELS,
  EXTRACTION_REASON_LABELS,
  EXTRACTION_STATUS_LABELS,
  STATE_LABELS,
  workupStatusLabel,
} from "./labels";
import type { Assessment, Evidence, QueryCriterion, QueryEntry } from "./types";

/** Where a quote points: a note of the dialog's list, its span, and which text (original or compressed). */
export interface NoteTarget {
  index: number;
  span: Span | null;
  compressed: boolean;
}

const LINK = "text-left text-xs font-medium text-brand-blue hover:underline";

export const noteDate = (value: string | null | undefined) =>
  value
    ? value.replace(/T00:00:00(\.0+)?(Z|[+-]00:00)?$/, "").replace("T", " ")
    : "date unavailable";

function Block({
  summary,
  open,
  children,
}: {
  summary: ReactNode;
  open?: boolean;
  children: ReactNode;
}) {
  return (
    <details className="rounded-md border border-gray-200 bg-white" open={open}>
      <summary className="cursor-pointer px-3 py-2 text-sm select-none">{summary}</summary>
      <div className="flex flex-col gap-2 border-t border-gray-200 p-3">{children}</div>
    </details>
  );
}

function Json({ value }: { value: unknown }) {
  return (
    <pre className="max-h-80 overflow-auto rounded-md bg-gray-50 p-2 font-mono text-xs break-words whitespace-pre-wrap">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

// ---------------------------------------------------------------- full-note review

/** A full-note finding: status, bottom line, limitations and the reviewed note excerpts. */
export function FullNoteAssessment({
  item,
  open,
  compressed,
  locate,
  onShow,
}: {
  item: Assessment;
  open: boolean;
  compressed: boolean;
  /** The note and span an excerpt points to, or null when its note is not in the list. */
  locate: (evidence: Evidence) => NoteTarget | null;
  onShow: (target: NoteTarget) => void;
}) {
  const evidence = item.evidence ?? [];
  return (
    <Block
      open={open}
      summary={
        <>
          <span className="font-medium">{item.name || item.recommendation.name}</span> ·{" "}
          {workupStatusLabel(item.status, "full_notes", compressed)}
        </>
      }
    >
      <p className="text-sm">
        <strong>{APPLICABILITY_LABELS[item.applicability]}</strong> · {item.bottom_line}
      </p>
      {item.recommendation.conditions ? <Helper>{item.recommendation.conditions}</Helper> : null}
      {(item.limitations ?? []).map((l, i) => (
        <Helper key={i}>{l}</Helper>
      ))}
      <Block
        summary={`${compressed ? "Reviewed compressed-note excerpts" : "Reviewed patient-note excerpts"} · ${String(evidence.length)}`}
      >
        <Helper>
          Automatically retained review context, including potentially irrelevant passages.
        </Helper>
        {evidence.map((v, i) => {
          const target = locate(v);
          return (
            <blockquote key={i} className="flex flex-col gap-1 border-l-2 border-gray-300 pl-3">
              <span className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
                Note {v.note_number ?? "?"} · {noteDate(v.note_date)}
                {v.note_type ? ` · ${v.note_type}` : ""}
                {target ? (
                  <button
                    type="button"
                    className={LINK}
                    onClick={() => {
                      onShow(target);
                    }}
                  >
                    Show in note
                  </button>
                ) : null}
              </span>
              <pre className="max-h-60 overflow-auto font-mono text-xs break-words whitespace-pre-wrap">
                {v.quote}
              </pre>
            </blockquote>
          );
        })}
      </Block>
      <Block summary="Guideline recommendation & source provenance">
        <Json value={item.recommendation} />
      </Block>
    </Block>
  );
}

// ---------------------------------------------------------------- extraction checks

function CriteriaTable({ criteria }: { criteria: QueryCriterion[] }) {
  return (
    <Table variant="list" className="text-xs">
      <TableCaption className="mt-1 text-left text-xs">
        Query criteria (code-evaluated)
      </TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Criterion</TableHead>
          <TableHead>Item type</TableHead>
          <TableHead>Holds</TableHead>
          <TableHead>Certain / possible / needed</TableHead>
          <TableHead>Anchor</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {criteria.map((c, i) => (
          <TableRow key={i}>
            <TableCell className="whitespace-normal">{c.label}</TableCell>
            <TableCell>
              {c.item_type.replace(/_/g, " ")}
              {c.status !== "any" ? ` · ${c.status}` : ""}
            </TableCell>
            <TableCell>{STATE_LABELS[c.state] ?? c.state}</TableCell>
            <TableCell className="tabular-nums">
              {c.definite_count} / {c.possible_count} / {c.min_count}
            </TableCell>
            <TableCell>
              {c.anchor
                ? `${c.anchor.kind.replace(/_/g, " ")}${c.anchor.date ? ` ${c.anchor.date}` : ""}${c.anchor.fallback ? " (reference date)" : ""}`
                : "any time"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

/** The SQL agent's accepted query for one workup item, with how the agent reached it. */
function SqlQueryDetail({ entry }: { entry: QueryEntry }) {
  const submissions = entry.submit_count ?? 0;
  const how =
    entry.expressible === false
      ? `Not expressible with extracted items: ${entry.not_expressible_reason ?? ""}`
      : `${entry.status === "compiled" ? "Accepted" : "Not accepted"}${entry.accepted_by === "budget" ? " at the end of the action budget" : entry.accepted_by === "agent" ? " by the agent" : ""} after ${String(entry.action_count ?? 0)} actions (${String(submissions)} submission${submissions === 1 ? "" : "s"})${entry.cached ? " · cached" : ""}${entry.model ? ` · ${entry.model}` : ""}`;
  const spread = entry.distribution?.status_reason;
  return (
    <>
      <Helper>{how}</Helper>
      {entry.recommends_against ? (
        <Helper>
          The guideline advises against this step: patients who had it are reported as done though
          not recommended.
        </Helper>
      ) : null}
      {entry.error ? <Helper>{entry.error}</Helper> : null}
      {spread ? (
        <Helper>
          On the exploration population ({entry.distribution?.patients ?? 0} patients):{" "}
          {Object.entries(spread)
            .map(([k, n]) => `${k.replace("/", " / ").replace(/_/g, " ")} ${String(n)}`)
            .join(" · ")}
        </Helper>
      ) : null}
      {entry.sql ? (
        <pre className="max-h-80 overflow-auto rounded-md bg-gray-50 p-2 font-mono text-xs whitespace-pre-wrap">
          {entry.sql}
        </pre>
      ) : null}
    </>
  );
}

const STATE_STYLE: Record<string, string> = {
  yes: "border-emerald-700 text-emerald-800",
  no: "border-red-700 text-red-800",
  planned: "border-amber-700 text-amber-800",
};

function ExtractionQuote({
  evidence: v,
  target,
  noteNumber,
  onShow,
}: {
  evidence: Evidence;
  target: NoteTarget | null;
  noteNumber: number | null;
  onShow: (target: NoteTarget) => void;
}) {
  const where = `${noteNumber ? `Note ${String(noteNumber)}` : "Note"} · ${noteDate(v.note_date)}`;
  const state = v.state ?? "";
  return (
    <blockquote className="flex flex-col gap-1 border-l-2 border-gray-300 pl-3 text-sm">
      <p className="break-words whitespace-pre-wrap">
        <span
          className={`mr-1.5 inline-block rounded border px-1 text-[0.7rem] ${STATE_STYLE[state] ?? "border-gray-500 text-gray-700"}`}
        >
          {STATE_LABELS[state] ?? state}
        </span>
        {v.quote}
      </p>
      <cite className="text-xs text-gray-600 not-italic">
        {[v.criterion, v.item, v.date_text].filter(Boolean).join(" · ")}
        <br />
        {target ? (
          <button
            type="button"
            className={LINK}
            onClick={() => {
              onShow(target);
            }}
          >
            {where} · show in note
          </button>
        ) : (
          <span>{where} · note not loaded</span>
        )}
      </cite>
    </blockquote>
  );
}

/** An extraction check's finding: status and reason, the query's criteria and the quoted items. */
export function ExtractionAssessment({
  item,
  open,
  method,
  query,
  locate,
  onShow,
}: {
  item: Assessment;
  open: boolean;
  method: string | undefined;
  /** The report's compiled query or SQL entry for this item. */
  query: QueryEntry | null;
  locate: (evidence: Evidence) => { target: NoteTarget; number: number } | null;
  onShow: (target: NoteTarget) => void;
}) {
  const evidence = item.evidence ?? [];
  const sql =
    method === "extraction_sql"
      ? (query ?? (item.sql ? { sql: item.sql, status: "compiled" } : null))
      : null;
  const compiled = method !== "extraction_sql" ? query?.query : undefined;
  return (
    <Block
      open={open}
      summary={
        <>
          <span className="font-medium">{item.name || item.recommendation.name}</span> ·{" "}
          <strong>{EXTRACTION_STATUS_LABELS[item.status] ?? item.status}</strong>
        </>
      }
    >
      <p className="text-sm">
        <strong>Applicability: {APPLICABILITY_LABELS[item.applicability]}</strong>
        {item.reason ? ` · ${EXTRACTION_REASON_LABELS[item.reason] ?? item.reason}` : ""}
      </p>
      <p className="text-sm break-words whitespace-pre-wrap">{item.bottom_line}</p>
      {item.query_summary ? <Helper>Query: {item.query_summary}</Helper> : null}
      {(item.limitations ?? []).map((l, i) => (
        <Helper key={i}>Limitation: {l}</Helper>
      ))}
      {item.criteria?.length ? <CriteriaTable criteria={item.criteria} /> : null}
      {evidence.length ? (
        <>
          <h4 className="text-sm font-semibold">
            Extracted-item evidence · {evidence.length} quote{evidence.length === 1 ? "" : "s"}
          </h4>
          {evidence.map((v, i) => {
            const found = locate(v);
            return (
              <ExtractionQuote
                key={i}
                evidence={v}
                target={found?.target ?? null}
                noteNumber={found?.number ?? null}
                onShow={onShow}
              />
            );
          })}
        </>
      ) : (
        <Helper>No extracted item quotes for this result.</Helper>
      )}
      <Block
        summary={`Guideline recommendation${sql ? " & SQL query" : compiled !== undefined && compiled !== null ? " & compiled query" : ""}`}
      >
        <Json value={item.recommendation} />
        {sql ? <SqlQueryDetail entry={sql} /> : compiled != null ? <Json value={compiled} /> : null}
      </Block>
    </Block>
  );
}
