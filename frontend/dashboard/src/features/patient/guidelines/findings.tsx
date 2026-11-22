import { cn } from "cn";
import { DownloadIcon } from "lucide-react";
import { type ReactNode, useState } from "react";

import { Button } from "@/components/ui/button";
import { saveBlob } from "@/lib/api";

import { findQuote, noteByNumber, noteDate } from "../notes";
import { getPatient } from "../store";
import type {
  GuidelineSpace,
  QueryCriterion,
  SqlEntry,
  WorkupAssessment,
  WorkupEvidence,
  WorkupReview,
} from "../types";
import { NoteDialog, type NoteView } from "./note-dialog";
import {
  applicabilityLabels,
  extractionReasonLabels,
  extractionStatusLabels,
  isAgentic,
  isExtractionReview,
  missingEvidenceLabel,
  patientEvidenceCountLabel,
  patientEvidenceLabel,
  reviewAid,
  reviewMethodLabel,
  stateLabels,
  workupBackupDetail,
  workupReviewDetail,
  workupStatusLabel,
} from "./report";

const label = (labels: Record<string, string>, key: string | undefined) =>
  key ? (labels[key] ?? key) : "";
const Helper = ({ children }: { children: ReactNode }) =>
  children ? <p className="text-xs text-gray-600">{children}</p> : null;

/** A completed workup review of one space: its findings, each with the quotes it rests on. */
export function Findings({ review, space }: { review: WorkupReview; space: GuidelineSpace }) {
  const [note, setNote] = useState<NoteView | null>(null);
  const extraction = isExtractionReview(review);
  const sql = review.metadata.method === "extraction_sql";
  const download = () => {
    const report = {
      space_trial_id: space.space_trial_id,
      space_name: space.name,
      source: space.source,
      ...review,
    };
    saveBlob(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
      "workup-documentation-review.json",
    );
  };
  /** Show a quote in its note: by its offsets (extraction reviews), else where it first appears. */
  const show = (evidence: WorkupEvidence) => {
    const number = evidence.note_number;
    if (!number) return;
    const s = getPatient();
    const original = noteByNumber(s.notesText, s.notes, number);
    if (!original) return;
    const text = original.note_text;
    const title = `Note ${String(number)} · ${noteDate(evidence.note_date ?? original.note_date)}${
      evidence.note_type || original.note_type
        ? ` · ${evidence.note_type ?? original.note_type ?? ""}`
        : ""
    }`;
    const span =
      extraction && typeof evidence.start === "number" && typeof evidence.end === "number"
        ? { start: evidence.start, end: evidence.end }
        : findQuote(text, evidence.quote);
    setNote({
      title,
      text,
      start: span?.start ?? null,
      end: span?.end ?? null,
      unmarked: "The quote was not found verbatim in this note; read the whole note.",
    });
  };
  const canShow = (evidence: WorkupEvidence) => {
    const s = getPatient();
    return !!evidence.note_number && !!noteByNumber(s.notesText, s.notes, evidence.note_number);
  };
  return (
    <div className="flex flex-col gap-3 border-t border-gray-200 pt-3">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-sm font-semibold">
          {extraction
            ? sql
              ? "SQL-agent findings"
              : "Extraction-query findings"
            : "Documentation findings"}
        </h4>
        <Button size="sm" variant="outline" className="ml-auto" onClick={download}>
          <DownloadIcon />
          Download JSON
        </Button>
      </div>
      <p className="text-sm font-semibold">{reviewMethodLabel(review)}</p>
      {extraction ? (
        <p className="rounded-md bg-gray-100 px-3 py-2 text-xs text-gray-700">
          {reviewAid(review.metadata.method)}
        </p>
      ) : null}
      <Helper>{review.notice}</Helper>
      {extraction ? <ExtractionStats review={review} /> : <ReviewStats review={review} />}
      {review.assessments.map((item, i) =>
        extraction ? (
          <ExtractionAssessment
            key={i}
            item={item}
            sql={
              sql ? review.metadata.queries?.queries?.[item.recommendation_index ?? i] : undefined
            }
            query={
              sql
                ? undefined
                : review.metadata.queries?.queries?.[item.recommendation_index ?? i]?.query
            }
            onShow={show}
            canShow={canShow}
          />
        ) : (
          <NoteReviewAssessment
            key={i}
            item={item}
            review={review}
            onShow={show}
            canShow={canShow}
          />
        ),
      )}
      {!extraction ? (
        <Helper>
          {review.metadata.note_count ?? 0}{" "}
          {isAgentic(review) ? "notes available to search" : "raw notes reviewed"} ·{" "}
          {review.metadata.requests ?? 0} LLM requests
        </Helper>
      ) : null}
      {note ? (
        <NoteDialog
          note={note}
          onClose={() => {
            setNote(null);
          }}
        />
      ) : null}
    </div>
  );
}

function ReviewStats({ review }: { review: WorkupReview }) {
  return <Helper>{workupBackupDetail(review.metadata)}</Helper>;
}

function ExtractionStats({ review }: { review: WorkupReview }) {
  const meta = review.metadata;
  const x = meta.extraction ?? {};
  const q = meta.queries ?? {};
  const compiled =
    meta.method === "extraction_sql"
      ? `SQL queries ${String(q.compiled ?? 0)}/${String(q.distinct ?? 0)} accepted, ${String(q.cached ?? 0)} cached, ${String(q.requests ?? 0)} agent requests over ${q.exploration?.name || "the store"} (${(q.exploration?.patients ?? 0).toLocaleString()} patients)`
      : `queries ${String(q.compiled ?? 0)}/${String(q.distinct ?? 0)} compiled, ${String(q.cached ?? 0)} cached, ${String(q.requests ?? 0)} compile requests`;
  const levels = (x.levels ?? []).join(" + ");
  return (
    <Helper>
      {meta.note_count ?? 0} notes ·{" "}
      {x.reused ? "session extraction reused" : `${String(x.requests ?? 0)} extraction requests`} (
      {levels} level{x.failed ? `, ${String(x.failed)} notes failed` : ""}
      {x.partial ? `, ${String(x.partial)} partial` : ""}) · {compiled}
      {meta.model ? ` · ${meta.model}` : ""}
    </Helper>
  );
}

function StatusBadge({ text, good }: { text: string; good: boolean }) {
  return (
    <span
      className={cn(
        "shrink-0 rounded-md px-2 py-1 text-xs font-semibold",
        good ? "bg-emerald-100 text-emerald-900" : "bg-slate-100 text-slate-800",
      )}
    >
      {text}
    </span>
  );
}

function NoteLink({
  evidence,
  onShow,
  canShow,
}: {
  evidence: WorkupEvidence;
  onShow: (e: WorkupEvidence) => void;
  canShow: (e: WorkupEvidence) => boolean;
}) {
  const where = `${evidence.note_number ? `Note ${String(evidence.note_number)}` : "Note"} · ${noteDate(evidence.note_date)}${evidence.note_type ? ` · ${evidence.note_type}` : ""}`;
  return canShow(evidence) ? (
    <button
      type="button"
      className="link text-left"
      onClick={() => {
        onShow(evidence);
      }}
    >
      {where} · show in note
    </button>
  ) : (
    <span>{where} · note not loaded</span>
  );
}

/** A full-note or agentic review's finding for one workup item. */
function NoteReviewAssessment({
  item,
  review,
  onShow,
  canShow,
}: {
  item: WorkupAssessment;
  review: WorkupReview;
  onShow: (e: WorkupEvidence) => void;
  canShow: (e: WorkupEvidence) => boolean;
}) {
  const evidence = item.evidence ?? [];
  return (
    <article className="flex flex-col gap-1.5 rounded-md border border-gray-200 bg-white p-3">
      <div className="flex items-start justify-between gap-3">
        <h5 className="font-medium text-gray-900">{item.name ?? item.recommendation?.name}</h5>
        <StatusBadge text={workupStatusLabel(item, review)} good={item.status === "completed"} />
      </div>
      <p className="text-xs font-semibold text-gray-700">
        Applicability: {label(applicabilityLabels, item.applicability)}
      </p>
      <p className="text-sm whitespace-pre-wrap text-gray-800">{item.bottom_line}</p>
      <Helper>{workupReviewDetail(item)}</Helper>
      <Helper>{workupBackupDetail(item.metadata)}</Helper>
      {item.limitations?.length ? (
        <Helper>Limitations: {item.limitations.join("; ")}</Helper>
      ) : null}
      {evidence.length ? (
        <details className="text-sm">
          <summary className="cursor-pointer text-brand-blue hover:underline">
            {patientEvidenceLabel(item, review)} · {patientEvidenceCountLabel(item, review)}
          </summary>
          <div className="mt-2 flex flex-col gap-2">
            {evidence.map((e, i) => (
              <blockquote
                key={i}
                className="border-l-2 border-gray-300 pl-3 whitespace-pre-wrap text-gray-800"
              >
                {e.quote}
                <cite className="mt-1 block text-xs text-gray-500 not-italic">
                  <NoteLink evidence={e} onShow={onShow} canShow={canShow} />
                </cite>
              </blockquote>
            ))}
          </div>
        </details>
      ) : (
        <Helper>{missingEvidenceLabel(item, review)}</Helper>
      )}
    </article>
  );
}

const STATE_STYLES: Record<string, string> = {
  yes: "text-emerald-800",
  no: "text-red-800",
  planned: "text-amber-800",
  unknown: "text-gray-600",
};

function CriteriaTable({ criteria }: { criteria: QueryCriterion[] | null | undefined }) {
  if (!criteria?.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <caption className="mb-1 text-left text-xs font-medium text-gray-600">
          Query criteria (code-evaluated)
        </caption>
        <thead className="bg-gray-50 text-left text-gray-600">
          <tr>
            {["Criterion", "Item type", "Holds", "Certain / possible / needed", "Anchor"].map(
              (h) => (
                <th key={h} scope="col" className="px-2 py-1 font-medium">
                  {h}
                </th>
              ),
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {criteria.map((c, i) => (
            <tr key={i}>
              <th scope="row" className="px-2 py-1 text-left font-medium">
                {c.label}
              </th>
              <td className="px-2 py-1">
                {c.item_type.replace(/_/g, " ")}
                {c.status !== "any" ? ` · ${c.status}` : ""}
              </td>
              <td className="px-2 py-1">{label(stateLabels, c.state)}</td>
              <td className="px-2 py-1">
                {c.definite_count} / {c.possible_count} / {c.min_count}
              </td>
              <td className="px-2 py-1">
                {c.anchor
                  ? `${c.anchor.kind.replace(/_/g, " ")}${c.anchor.date ? ` ${c.anchor.date}` : ""}${c.anchor.fallback ? " (reference date)" : ""}`
                  : "any time"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The SQL agent's accepted query for one workup item, and how the agent reached it. */
function SqlDetail({ entry }: { entry: SqlEntry }) {
  const how =
    entry.expressible === false
      ? `Not expressible with extracted items: ${entry.not_expressible_reason ?? ""}`
      : `${entry.status === "compiled" ? "Accepted" : "Not accepted"}${
          entry.accepted_by === "budget"
            ? " at the end of the action budget"
            : entry.accepted_by === "agent"
              ? " by the agent"
              : ""
        } after ${String(entry.action_count ?? 0)} actions (${String(entry.submit_count ?? 0)} submission${
          entry.submit_count === 1 ? "" : "s"
        })${entry.cached ? " · cached" : ""}${entry.model ? ` · ${entry.model}` : ""}`;
  const spread = entry.distribution?.status_reason;
  return (
    <div className="flex flex-col gap-1">
      <Helper>{how}</Helper>
      {entry.recommends_against ? (
        <Helper>
          The guideline advises against this step: patients who had it are reported as done though
          not recommended.
        </Helper>
      ) : null}
      <Helper>{entry.error}</Helper>
      {spread ? (
        <Helper>
          On the exploration population ({entry.distribution?.patients ?? 0} patients):{" "}
          {Object.entries(spread)
            .map(([k, n]) => `${k.replace("/", " / ").replace(/_/g, " ")} ${String(n)}`)
            .join(" · ")}
        </Helper>
      ) : null}
      {entry.sql ? (
        <pre className="overflow-x-auto rounded-md bg-gray-900 p-2 text-xs text-gray-100">
          {entry.sql}
        </pre>
      ) : null}
    </div>
  );
}

/** An extraction review's result for one workup item. */
function ExtractionAssessment({
  item,
  sql,
  query,
  onShow,
  canShow,
}: {
  item: WorkupAssessment;
  sql: SqlEntry | undefined;
  query: unknown;
  onShow: (e: WorkupEvidence) => void;
  canShow: (e: WorkupEvidence) => boolean;
}) {
  const evidence = item.evidence ?? [];
  const sqlEntry = sql ?? (item.sql ? { sql: item.sql, status: "compiled" } : null);
  return (
    <details className="rounded-md border border-gray-200 bg-white p-3 text-sm">
      <summary className="cursor-pointer">
        {item.name ?? item.recommendation?.name} ·{" "}
        <strong>{label(extractionStatusLabels, item.status)}</strong>
      </summary>
      <div className="mt-2 flex flex-col gap-1.5">
        <p>
          <strong>Applicability: {label(applicabilityLabels, item.applicability)}</strong> ·{" "}
          {label(extractionReasonLabels, item.reason)}
        </p>
        <p className="whitespace-pre-wrap text-gray-800">{item.bottom_line}</p>
        {item.query_summary ? <Helper>Query: {item.query_summary}</Helper> : null}
        {(item.limitations ?? []).map((l, i) => (
          <Helper key={i}>Limitation: {l}</Helper>
        ))}
        <CriteriaTable criteria={item.criteria} />
        {evidence.length ? (
          <>
            <h6 className="text-xs font-semibold text-gray-700">
              Extracted-item evidence · {evidence.length} quote{evidence.length === 1 ? "" : "s"}
            </h6>
            {evidence.map((e, i) => (
              <blockquote
                key={i}
                className="border-l-2 border-gray-300 pl-3 whitespace-pre-wrap text-gray-800"
              >
                {e.state ? (
                  <span
                    className={cn(
                      "mr-1 inline-block rounded-sm border border-current px-1 text-[0.7rem]",
                      STATE_STYLES[e.state] ?? "text-gray-600",
                    )}
                  >
                    {label(stateLabels, e.state)}
                  </span>
                ) : null}
                {e.quote}
                <cite className="mt-1 block text-xs text-gray-500 not-italic">
                  {[e.criterion, e.item, e.date_text].filter(Boolean).join(" · ")}
                  <br />
                  <NoteLink evidence={e} onShow={onShow} canShow={canShow} />
                </cite>
              </blockquote>
            ))}
          </>
        ) : (
          <Helper>No extracted item quotes for this result.</Helper>
        )}
        <details>
          <summary className="cursor-pointer text-xs text-brand-blue hover:underline">
            Guideline recommendation{sqlEntry ? " & SQL query" : query ? " & compiled query" : ""}
          </summary>
          <div className="mt-1 flex flex-col gap-1">
            <pre className="overflow-x-auto rounded-md bg-gray-50 p-2 text-xs">
              {JSON.stringify(item.recommendation, null, 2)}
            </pre>
            {sqlEntry ? (
              <SqlDetail entry={sqlEntry} />
            ) : query ? (
              <pre className="overflow-x-auto rounded-md bg-gray-50 p-2 text-xs">
                {JSON.stringify(query, null, 2)}
              </pre>
            ) : null}
          </div>
        </details>
      </div>
    </details>
  );
}
