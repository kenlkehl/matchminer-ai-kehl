import { useQuery, useQueryClient } from "@tanstack/react-query";
import { DatabaseIcon, FileTextIcon, QuoteIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, type Job, waitForJob } from "@/lib/api";
import {
  describeSource,
  type Levels,
  type RecordEvidence,
  type RecordItem,
  type RecordsResponse,
  type RecordsView as Records,
} from "@/lib/records";

import { EndpointNotice } from "../fields";
import { NoteDialog, type NoteView } from "../guidelines/note-dialog";
import { noteByNumber, noteDate } from "../notes";
import { getPatient, recordsBody, setPatient, usePatient } from "../store";

/** The fields shown for each item type, after its label and date. */
const FIELDS: Record<string, string[]> = {
  diagnosis: ["primary_site", "histology", "stage_group", "extent_at_diagnosis", "grade"],
  biomarker: ["biomarker", "result", "alteration", "value", "method", "origin"],
  treatment: [
    "treatment_type",
    "regimen",
    "agents",
    "setting",
    "line_of_therapy",
    "treatment_status",
    "best_response",
  ],
  procedure: ["procedure_type", "specimen_site", "specimen_result", "result_text"],
  imaging_study: ["modality", "body_regions", "contrast", "status"],
  lab_test: ["lab_test", "result_text", "result_flag"],
  performance_status: ["scale", "score"],
  disease_assessment: ["disease_status", "disease_sites"],
  event: ["event_type", "description"],
  counseling: ["topic", "details"],
};

/** Whether a field has a value to show (a line of therapy of 0 means not stated). */
const shown = (name: string, value: unknown) =>
  value !== null &&
  value !== undefined &&
  value !== "" &&
  value !== "not_stated" &&
  !(Array.isArray(value) && value.length === 0) &&
  !(name === "line_of_therapy" && value === 0);

function fieldText(name: string, value: unknown): string {
  const text = Array.isArray(value) ? value.join(", ") : String(value);
  return `${name.replace(/_/g, " ")}: ${text.replace(/_/g, " ")}`;
}

/**
 * The Patient tab's Structured data: the clinical facts extracted from this patient's notes, patient
 * level (one record built from all notes) or note by note, each with the quotes it rests on. Stored
 * records when the mode's extraction store holds the corpus patient; else the session's extraction of
 * the notes, made on request.
 */
export function RecordsView() {
  const s = usePatient();
  const [levels, setLevels] = useState<Levels>("document_and_patient");
  const [level, setLevel] = useState<"patient" | "document">("patient");
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<NoteView | null>(null);
  const queryClient = useQueryClient();
  const body = { ...recordsBody(s), levels };
  const hasInput = Boolean(body.patient_id || body.text || body.notes?.length);
  const records = useQuery({
    queryKey: ["patient-records", body],
    queryFn: () => api.post<RecordsResponse>("/api/patient/records", body),
    enabled: hasInput,
  });

  const extract = async () => {
    setError(null);
    setPatient({ recordsJob: true });
    try {
      const { job } = await api.post<{ job: Job }>("/api/patient/records/extract", body);
      await waitForJob(job.id, (j) => {
        setProgress(j.message);
      });
      await queryClient.invalidateQueries({ queryKey: ["patient-records"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPatient({ recordsJob: false });
      setProgress("");
    }
  };

  const show = (evidence: RecordEvidence) => {
    const p = getPatient();
    const found = evidence.note_number
      ? noteByNumber(p.notesText, p.notes, evidence.note_number)
      : null;
    if (!found || !evidence.note_number) return;
    setNote({
      title: `Note ${String(evidence.note_number)} · ${noteDate(evidence.note_date ?? found.note_date)}${
        found.note_type ? ` · ${found.note_type}` : ""
      }`,
      text: found.note_text,
      start: evidence.start,
      end: evidence.end,
      unmarked: "The quote's position was not found in this note; read the whole note.",
    });
  };
  const canShow = (evidence: RecordEvidence) => {
    const p = getPatient();
    return !!evidence.note_number && !!noteByNumber(p.notesText, p.notes, evidence.note_number);
  };

  if (!hasInput) {
    return (
      <p className="text-sm text-gray-600">
        Choose a patient or add clinical notes above to see the clinical facts extracted from the
        notes.
      </p>
    );
  }
  if (records.isPending) return <Spinner className="size-5" />;
  if (records.isError) return <p className="text-sm text-red-700">{records.error.message}</p>;
  const data = records.data;
  if (!data.source || !data.view) {
    const estimate = data.estimate;
    return (
      <div className="flex flex-col gap-3 rounded-md border border-gray-200 p-4">
        <p className="text-sm text-gray-800">
          {body.patient_id && data.store?.available
            ? "The extraction store has no records of this patient."
            : "There are no structured records of these notes yet."}{" "}
          Extracting reads each note (and, with patient level, all notes together) with the model of
          the guideline concordance checks.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={levels === "document_and_patient"}
              onChange={(e) => {
                setLevels(e.target.checked ? "document_and_patient" : "document");
              }}
            />
            Also build a patient-level record
          </label>
          {estimate ? (
            <span className="text-sm text-gray-600">
              {estimate.notes} notes · {estimate.requests} requests to {estimate.model} · about{" "}
              {Math.round(estimate.approximate_note_tokens / 1000)}k note tokens
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" disabled={s.recordsJob || !estimate} onClick={() => void extract()}>
            {s.recordsJob ? <Spinner /> : <DatabaseIcon />}
            Extract structured data
          </Button>
          {s.recordsJob ? (
            <span className="text-sm text-gray-500">{progress || "Extracting…"}</span>
          ) : null}
          {error ? <span className="text-sm text-red-700">{error}</span> : null}
        </div>
        <EndpointNotice tasks={["workup"]} />
      </div>
    );
  }
  const view = data.view;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-gray-700">
          <strong>{describeSource(data.source)}</strong> · {view.counts.patient_items} patient-level
          items · {view.counts.document_items} items in {view.counts.notes} notes
          {view.counts.failed_notes ? ` · ${String(view.counts.failed_notes)} notes failed` : ""}
          {view.counts.partial_notes ? ` · ${String(view.counts.partial_notes)} partial` : ""}
        </p>
        <Tabs
          value={level}
          onValueChange={(value: "patient" | "document") => {
            setLevel(value);
          }}
        >
          <TabsList>
            <TabsTrigger value="patient" disabled={!view.patient.available}>
              Patient level
            </TabsTrigger>
            <TabsTrigger value="document">By note</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <p className="text-xs text-gray-500">
        Extracted by an LLM from the notes for review: check every item against its quoted note. An
        item exists only when a note states it; a missing item does not mean it is absent.
      </p>
      {level === "patient" && view.patient.available ? (
        <PatientLevel view={view} show={show} canShow={canShow} />
      ) : (
        <ByNote view={view} show={show} canShow={canShow} />
      )}
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

type Show = (evidence: RecordEvidence) => void;
type CanShow = (evidence: RecordEvidence) => boolean;

function PatientLevel({ view, show, canShow }: { view: Records; show: Show; canShow: CanShow }) {
  if (!view.patient.types.length) {
    return <p className="text-sm text-gray-600">The patient-level record has no items.</p>;
  }
  return (
    <div className="flex flex-col gap-4">
      {view.patient.types.map((group) => (
        <section key={group.type}>
          <h4 className="mb-1 text-sm font-bold">
            {group.label} <span className="font-normal text-gray-500">({group.items.length})</span>
          </h4>
          <ul className="flex flex-col divide-y divide-gray-100 rounded-md border border-gray-200">
            {group.items.map((item, i) => (
              <ItemRow key={item.item_id ?? i} item={item} show={show} canShow={canShow} />
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

function ByNote({ view, show, canShow }: { view: Records; show: Show; canShow: CanShow }) {
  return (
    <div className="flex flex-col gap-2">
      {view.documents.map((n) => (
        <details key={n.note_id ?? n.note_number} className="rounded-md border border-gray-200">
          <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm">
            <FileTextIcon className="size-4 text-gray-400" />
            <span className="font-medium">
              {n.note_number ? `Note ${String(n.note_number)}` : "Note"} · {noteDate(n.note_date)}
            </span>
            <span className="text-gray-500">
              {[n.note_type, n.document_type?.replace(/_/g, " ")].filter(Boolean).join(" · ")}
            </span>
            <span className="ml-auto text-xs text-gray-500">
              {n.items.length} {n.items.length === 1 ? "item" : "items"}
              {n.status && n.status !== "complete" ? ` · ${n.status}` : ""}
            </span>
          </summary>
          {n.items.length ? (
            <ul className="flex flex-col divide-y divide-gray-100 border-t border-gray-100">
              {n.items.map((item, i) => (
                <ItemRow key={i} item={item} show={show} canShow={canShow} />
              ))}
            </ul>
          ) : (
            <p className="border-t border-gray-100 px-3 py-2 text-sm text-gray-500">
              No items extracted from this note.
            </p>
          )}
        </details>
      ))}
    </div>
  );
}

function ItemRow({ item, show, canShow }: { item: RecordItem; show: Show; canShow: CanShow }) {
  const fields = (FIELDS[item.type] ?? [])
    .filter((name) => shown(name, item.fields[name]))
    .map((name) => fieldText(name, item.fields[name]));
  return (
    <li className="flex flex-col gap-1 px-3 py-2 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="font-medium text-gray-900">{item.label}</span>
        <span className="text-xs text-gray-500">{item.date_text}</span>
      </div>
      {fields.length ? <p className="text-xs text-gray-600">{fields.join(" · ")}</p> : null}
      {item.evidence.map((e, i) => (
        <blockquote
          key={i}
          className="flex items-start gap-2 border-l-2 border-gray-200 pl-2 text-xs text-gray-700"
        >
          <QuoteIcon className="mt-0.5 size-3 shrink-0 text-gray-400" />
          <span className="flex-1">{e.quote}</span>
          {canShow(e) ? (
            <button
              type="button"
              className="shrink-0 link"
              onClick={() => {
                show(e);
              }}
            >
              Note {e.note_number} · show in note
            </button>
          ) : (
            <span className="shrink-0 text-gray-400">{noteDate(e.note_date)}</span>
          )}
        </blockquote>
      ))}
    </li>
  );
}
