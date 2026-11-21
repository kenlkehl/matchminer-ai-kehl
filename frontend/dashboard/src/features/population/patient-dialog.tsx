import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";

import { ExtractionAssessment, FullNoteAssessment, type NoteTarget, noteDate } from "./assessments";
import { Helper } from "./controls";
import { HighlightedText, locateQuote } from "./highlight";
import { isExtractionMethod, METHOD_NAMES, reviewAid } from "./labels";
import type { Evidence, PatientDrilldown, PopulationNote, PopulationReport } from "./types";
import { errorMessage } from "./use-population";

/**
 * One active patient: the cancer history summary, every workup finding with its quoted evidence (a quote
 * opens its source note with the span marked) and the patient's source notes.
 */
export function PatientDialog({
  version,
  patientId,
  item,
  report,
  onClose,
  onStale,
}: {
  version: string;
  patientId: string;
  /** The workup item the dialog was opened from (its finding opens). */
  item: number | null;
  report: PopulationReport | null;
  onClose: () => void;
  onStale: () => void;
}) {
  const detail = useQuery({
    queryKey: ["population", "patient", version, patientId],
    queryFn: async () => {
      try {
        return await api.get<PatientDrilldown>(
          `/api/population/patient?version=${encodeURIComponent(version)}&patient_id=${encodeURIComponent(patientId)}`,
        );
      } catch (err) {
        onStale();
        throw err;
      }
    },
    retry: false,
    staleTime: Infinity,
  });
  const data = detail.data;
  const [openNotes, setOpenNotes] = useState<ReadonlySet<number>>(new Set());
  const [marked, setMarked] = useState<NoteTarget | null>(null);

  const show = useCallback((target: NoteTarget) => {
    setOpenNotes((current) => new Set(current).add(target.index));
    setMarked({ ...target });
  }, []);
  // Each "show in note" (a new target object) scrolls to its note, then centres the marked span.
  useEffect(() => {
    if (!marked) return;
    const note = document.getElementById(`pop-note-${String(marked.index)}`);
    note?.scrollIntoView({ block: "start" });
    note?.querySelector("mark")?.scrollIntoView({ block: "center" });
  }, [marked]);

  const review = data?.review ?? null;
  const method = review?.metadata?.method;
  const extraction = isExtractionMethod(method);
  const compressed = review?.metadata?.note_source === "compressed";
  const notes = data?.notes ?? [];
  const byNoteId = useMemo(
    () => new Map(notes.map((n, i) => [n.note_id ?? "", i] as const)),
    [notes],
  );

  /** A full-note excerpt: its note by number, the quote located in the text the review read. */
  const locateExcerpt = (v: Evidence): NoteTarget | null => {
    const index = (v.note_number ?? 0) - 1;
    const note = notes[index];
    if (!note) return null;
    const useCompressed = compressed && !!note.compressed_note_text;
    const text = useCompressed ? (note.compressed_note_text ?? "") : note.note_text;
    return { index, span: locateQuote(text, v.quote), compressed: useCompressed };
  };
  /** An extraction quote: its note by note id; the offsets index the original note text. */
  const locateExtraction = (v: Evidence) => {
    const index = v.note_id ? byNoteId.get(v.note_id) : undefined;
    if (index === undefined) return null;
    const span =
      typeof v.start === "number" && typeof v.end === "number"
        ? { start: v.start, end: v.end }
        : null;
    return { target: { index, span, compressed: false }, number: index + 1 };
  };

  const coverage = review?.metadata?.coverage ?? {};
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="h-[90svh] sm:max-w-5xl" bodyClassName="gap-4 auto-rows-min">
        <DialogHeader>
          <DialogTitle>Patient {patientId}</DialogTitle>
          <DialogDescription>
            {data?.space?.name ? `Active cohort: ${data.space.name}` : "Active cohort patient"}
          </DialogDescription>
        </DialogHeader>
        {detail.isPending ? (
          <Spinner />
        ) : detail.isError ? (
          <p className="text-sm text-red-700">{errorMessage(detail.error)}</p>
        ) : data ? (
          <>
            <section className="flex flex-col gap-1">
              <h3 className="text-sm font-semibold">Cancer history summary</h3>
              <p className="rounded-md bg-gray-50 p-3 font-mono text-xs break-words whitespace-pre-wrap">
                {data.patient.cancer_history_summary}
              </p>
            </section>
            <section className="flex flex-col gap-2">
              <h3 className="text-sm font-semibold">Workup findings</h3>
              {!review ? (
                <Helper>No workup review available for this patient yet.</Helper>
              ) : extraction ? (
                <>
                  <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900">
                    {reviewAid(method)}
                  </p>
                  <Helper>{review.notice}</Helper>
                  <Helper>
                    {method === "extraction_sql"
                      ? METHOD_NAMES.extraction_sql
                      : METHOD_NAMES.extraction_queries}{" "}
                    · {coverage.documents ?? 0} document records
                    {coverage.patient_records
                      ? ` · ${String(coverage.patient_records)} patient-level record`
                      : ""}
                    {coverage.notes_without_records
                      ? ` · ${String(coverage.notes_without_records)} notes without records`
                      : ""}
                    {coverage.failed_documents
                      ? ` · ${String(coverage.failed_documents)} failed`
                      : ""}{" "}
                    · {coverage.items ?? 0} extracted items
                  </Helper>
                  {review.assessments.map((a) => (
                    <ExtractionAssessment
                      key={a.recommendation_index}
                      item={a}
                      open={a.recommendation_index === item}
                      method={method}
                      query={report?.queries?.[a.recommendation_index] ?? null}
                      locate={locateExtraction}
                      onShow={show}
                    />
                  ))}
                </>
              ) : (
                <>
                  <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900">
                    {review.notice}
                  </p>
                  <Helper>
                    {review.metadata?.note_source ?? "original"} notes · Full-note review ·{" "}
                    {review.metadata?.model ?? "See individual findings for model provenance"}
                  </Helper>
                  {review.assessments.map((a) => (
                    <FullNoteAssessment
                      key={a.recommendation_index}
                      item={a}
                      open={a.recommendation_index === item}
                      compressed={compressed}
                      locate={locateExcerpt}
                      onShow={show}
                    />
                  ))}
                </>
              )}
            </section>
            <section className="flex flex-col gap-2">
              <h3 className="text-sm font-semibold">Source notes · {notes.length}</h3>
              {notes.map((n, i) => (
                <SourceNote
                  key={i}
                  note={n}
                  index={i}
                  showId={extraction}
                  open={openNotes.has(i)}
                  onToggle={(open) => {
                    setOpenNotes((current) => {
                      const next = new Set(current);
                      if (open) next.add(i);
                      else next.delete(i);
                      return next;
                    });
                  }}
                  marked={marked?.index === i ? marked : null}
                />
              ))}
            </section>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}

function SourceNote({
  note,
  index,
  showId,
  open,
  onToggle,
  marked,
}: {
  note: PopulationNote;
  index: number;
  showId: boolean;
  open: boolean;
  onToggle: (open: boolean) => void;
  marked: NoteTarget | null;
}) {
  return (
    <details
      id={`pop-note-${String(index)}`}
      className="rounded-md border border-gray-200 bg-white open:ring-1 open:ring-brand-light-blue/40"
      open={open}
      onToggle={(e) => {
        if (e.currentTarget.open !== open) onToggle(e.currentTarget.open);
      }}
    >
      <summary className="cursor-pointer px-3 py-2 text-sm select-none">
        Note {index + 1} · {noteDate(note.note_date)}
        {note.note_type ? ` · ${note.note_type}` : ""}
        {showId && note.note_id ? (
          <span className="ml-1 font-mono text-xs text-gray-500">· {note.note_id}</span>
        ) : null}
      </summary>
      <div className="flex flex-col gap-2 border-t border-gray-200 p-3">
        <h4 className="text-xs font-semibold text-gray-700">Original note</h4>
        <pre className="max-h-[60svh] overflow-auto font-mono text-xs break-words whitespace-pre-wrap">
          <HighlightedText
            text={note.note_text}
            span={marked && !marked.compressed ? marked.span : null}
          />
        </pre>
        {note.compressed_note_text ? (
          <>
            <h4 className="text-xs font-semibold text-gray-700">
              Saved compressed note · losslessness unverified
            </h4>
            <pre className="max-h-[60svh] overflow-auto font-mono text-xs break-words whitespace-pre-wrap">
              <HighlightedText
                text={note.compressed_note_text}
                span={marked?.compressed ? marked.span : null}
              />
            </pre>
          </>
        ) : null}
      </div>
    </details>
  );
}
