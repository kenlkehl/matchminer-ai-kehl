import { useMutation, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { ChevronUpIcon, RotateCcwIcon, TableIcon, UploadIcon } from "lucide-react";
import { useEffect, useId, useState } from "react";

import { SearchInput } from "@/components/search-input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api, type PatientDetail } from "@/lib/api";
import { patientLine } from "@/lib/format";
import { modelsOptions } from "@/lib/models";
import { metaOptions, notesOptions, patientsOptions } from "@/lib/queries";

import {
  clearSummary,
  clearWorkspace,
  corpusNotesFailed,
  editNotes,
  editSummary,
  fillCorpusNotes,
  loadSyntheticNotes,
  pickCorpusPatient,
  summarize,
  uploadRecord,
} from "./actions";
import { EndpointNotice, Field, NativeSelect, StatusLine } from "./fields";
import { JobProgress } from "./job-progress";
import { MAX_NOTE_CHARACTERS, noteRows } from "./notes";
import { NotesDialog } from "./notes-dialog";
import {
  CANCER_SUMMARY_SCAFFOLD,
  hasSummary,
  type InputTab,
  isBusy,
  type PatientState,
  setPatient,
  usePatient,
} from "./store";

/** "synthetic patient" in synthetic mode, "patient" in real mode. */
function usePatientNoun(): string {
  const mode = useQuery(metaOptions()).data?.mode;
  return mode === "real" ? "patient" : "synthetic patient";
}

function words(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/** Who the shared patient is, in a few words. */
function patientTitle(s: PatientState, noun: string): string {
  if (s.patientId && s.fromCorpus)
    return `${noun[0]?.toUpperCase() ?? ""}${noun.slice(1)} · ${s.patientId}`;
  if (s.patientId) return `${s.patientId} · edited summary`;
  switch (s.notesOrigin?.kind) {
    case "synthetic":
      return `Synthetic example · ${s.notesOrigin.label}`;
    case "upload":
      return `Uploaded record · ${s.notesOrigin.name}`;
    case "pasted":
      return "Pasted notes";
    case "restored":
      return "Restored from this browser session";
    default:
      return hasSummary(s.summary) ? "Entered summary" : "No patient chosen";
  }
}

/**
 * The patient both views use: a patient chosen from the server's corpus, an entered summary and
 * other relevant history, or clinical notes (pasted, uploaded or a synthetic example) and the summary
 * made from them. It folds to one line after a search.
 */
export function PatientInput() {
  const s = usePatient();
  const noun = usePatientNoun();
  const busy = isBusy(s);
  const id = useId();
  const noteCount = noteRows(s.notesText, s.notes).length;
  const inputJob = s.job?.kind === "summarize" ? s.job : null;
  return (
    <section aria-labelledby={`${id}-title`} className="rounded-lg border border-gray-200 bg-white">
      <CorpusNotesLoader />
      <div
        className={cn(
          "flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5",
          s.inputOpen && "border-b border-gray-200",
        )}
      >
        <h2 id={`${id}-title`} className="text-base font-semibold">
          Patient
        </h2>
        <span className="min-w-0 truncate text-sm text-gray-700">{patientTitle(s, noun)}</span>
        <Badge variant={hasSummary(s.summary) ? "secondary" : "outline"}>
          {hasSummary(s.summary) ? "Summary ready" : "No summary"}
        </Badge>
        <Badge variant={noteCount ? "secondary" : "outline"}>
          {s.corpusNotesPending ? (
            <>
              <Spinner className="size-3" /> Notes loading
            </>
          ) : noteCount ? (
            `${String(noteCount)} ${noteCount === 1 ? "note" : "notes"}`
          ) : (
            "No notes"
          )}
        </Badge>
        <div className="ml-auto flex gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            title="Forget the patient, the guideline results and the trial matches"
            onClick={() => {
              void clearWorkspace();
            }}
          >
            <RotateCcwIcon />
            Clear workspace
          </Button>
          <Button
            size="sm"
            variant="outline"
            aria-expanded={s.inputOpen}
            aria-controls={`${id}-body`}
            onClick={() => {
              setPatient({ inputOpen: !s.inputOpen });
            }}
          >
            {s.inputOpen ? "Collapse" : "Change patient"}
            <ChevronUpIcon className={cn("transition-transform", !s.inputOpen && "rotate-180")} />
          </Button>
        </div>
      </div>
      <div id={`${id}-body`} hidden={!s.inputOpen}>
        <Tabs
          value={s.input}
          onValueChange={(value: InputTab) => {
            setPatient({ input: value });
          }}
        >
          <TabsList variant="line" className="w-full justify-start border-b px-4">
            <TabsTrigger value="corpus" className="flex-none" disabled={busy}>
              {noun === "patient" ? "Choose a patient" : "Synthetic patients"}
            </TabsTrigger>
            <TabsTrigger value="summary" className="flex-none" disabled={busy}>
              Patient summary
            </TabsTrigger>
            <TabsTrigger value="notes" className="flex-none" disabled={busy}>
              Clinical notes
            </TabsTrigger>
          </TabsList>
          <div className="flex flex-col gap-3 p-4">
            <TabsContent value="corpus">
              <CorpusPane noun={noun} />
            </TabsContent>
            <TabsContent value="summary">
              <SummaryPane />
            </TabsContent>
            <TabsContent value="notes">
              <NotesPane noun={noun} />
            </TabsContent>
            <StatusLine where="input" />
            {inputJob ? <JobProgress job={inputJob} /> : null}
          </div>
        </Tabs>
      </div>
      {!s.inputOpen && inputJob ? (
        <div className="px-4 pb-3">
          <JobProgress job={inputJob} />
        </div>
      ) : null}
    </section>
  );
}

/** Loads a chosen corpus patient's notes (the server indexes them once, which takes a while). */
function CorpusNotesLoader() {
  const pending = usePatient().corpusNotesPending;
  const notes = useQuery({ ...notesOptions(pending ?? ""), enabled: pending !== null });
  const data = notes.data;
  useEffect(() => {
    if (!pending) return;
    if (notes.isError) corpusNotesFailed(pending, notes.error.message);
    if (!data || data.state === "building") return;
    if (data.state === "ready") fillCorpusNotes(pending, data.notes);
    else if (data.state === "error" || data.state === "none") {
      corpusNotesFailed(pending, data.message ?? "The patient's notes could not be read.");
    } else corpusNotesFailed(pending, "This server has no notes for its patients.");
  }, [pending, data, notes.isError, notes.error]);
  return null;
}

// ---------------------------------------------------------------- the corpus

function CorpusPane({ noun }: { noun: string }) {
  const s = usePatient();
  return (
    <div className="flex flex-col gap-3">
      <PatientPicker noun={noun} disabled={isBusy(s)} />
      <p className="text-sm text-gray-600">
        {s.patientId ? (
          <>
            Selected: <strong className="text-gray-800">{s.patientId}</strong>
            {s.fromCorpus ? null : " · its summary has been edited or replaced"}.{" "}
          </>
        ) : null}
        A {noun}'s cancer summary and other history fill Patient summary, and its notes load into
        Clinical notes for the guideline workup.
      </p>
    </div>
  );
}

/** Search the corpus patients by their summaries' text, filter by cancer type, and pick one. */
function PatientPicker({ noun, disabled }: { noun: string; disabled: boolean }) {
  const patientId = usePatient().patientId;
  const meta = useQuery(metaOptions()).data;
  const [q, setQ] = useState("");
  const [cancerType, setCancerType] = useState("");
  const [offset, setOffset] = useState(0);
  const list = useQuery(patientsOptions(q, cancerType, "", offset));
  const pick = useMutation({
    mutationFn: (id: string) => api.get<PatientDetail>(`/api/patients/${encodeURIComponent(id)}`),
    onSuccess: pickCorpusPatient,
  });
  const data = list.data;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-64 flex-1 flex-col gap-1">
          <span className="text-sm font-medium">Search the {noun}s</span>
          <SearchInput
            value={q}
            placeholder="Words or regular expressions in the summary, e.g. EGFR osimertinib"
            onChange={(value) => {
              setQ(value);
              setOffset(0);
            }}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium">Cancer type</span>
          <NativeSelect
            value={cancerType}
            onChange={(e) => {
              setCancerType(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">Any</option>
            {meta?.cancer_types
              .filter((t) => t.group !== "Basket")
              .map((t) => (
                <option key={t.key} value={t.key}>
                  {t.label}
                </option>
              ))}
          </NativeSelect>
        </label>
      </div>
      <div className="overflow-hidden rounded-md border border-gray-200">
        {data ? (
          <>
            <ul className="max-h-72 divide-y divide-gray-100 overflow-y-auto">
              {data.rows.map((p) => (
                <li key={p.id}>
                  <button
                    type="button"
                    disabled={disabled}
                    aria-pressed={patientId === p.id}
                    className={cn(
                      "flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-brand-blue/5 disabled:cursor-not-allowed disabled:opacity-60",
                      patientId === p.id && "bg-brand-blue/10",
                    )}
                    onClick={() => {
                      pick.mutate(p.id);
                    }}
                  >
                    <span className="flex-1 text-gray-800">{patientLine(p)}</span>
                    <span className="shrink-0 font-mono text-xs text-gray-400">{p.id}</span>
                    {pick.isPending && pick.variables === p.id ? (
                      <Spinner className="size-3.5" />
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
            <div className="flex items-center gap-2 border-t border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
              {data.total.toLocaleString()} of {data.of.toLocaleString()} patients
              <div className="ml-auto flex gap-1">
                <Button
                  size="xs"
                  variant="outline"
                  disabled={offset === 0}
                  onClick={() => {
                    setOffset(Math.max(0, offset - 25));
                  }}
                >
                  Previous
                </Button>
                <Button
                  size="xs"
                  variant="outline"
                  disabled={offset + 25 >= data.total}
                  onClick={() => {
                    setOffset(offset + 25);
                  }}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        ) : list.isError ? (
          <p className="p-3 text-sm text-red-700">{list.error.message}</p>
        ) : (
          <div className="flex justify-center p-6">
            <Spinner />
          </div>
        )}
      </div>
      {pick.isError ? <p className="text-sm text-red-700">{pick.error.message}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- the summary

function SummaryPane() {
  const s = usePatient();
  const id = useId();
  const busy = isBusy(s);
  const count = hasSummary(s.summary) ? words(s.summary) : 0;
  return (
    <div className="flex flex-col gap-2">
      <div className="grid items-start gap-4 md:grid-cols-[3fr_1fr]">
        <Field
          label="Cancer history summary"
          htmlFor={`${id}-summary`}
          help="A high-quality cancer summary includes age, sex, cancer type/primary site, disease histology, current extent of disease, relevant biomarkers, and treatment history."
        >
          <Textarea
            id={`${id}-summary`}
            value={s.summary}
            disabled={busy}
            spellCheck={false}
            maxLength={100_000}
            onChange={(e) => {
              editSummary({ summary: e.target.value });
            }}
            className="field-sizing-fixed scrollbar-visible h-[calc(10lh+10px)] font-mono text-sm"
          />
        </Field>
        <Field
          label="Other relevant history"
          htmlFor={`${id}-other`}
          help="Common exclusions: conditions that often preclude enrollment."
        >
          <Textarea
            id={`${id}-other`}
            value={s.otherHistory}
            disabled={busy}
            spellCheck={false}
            placeholder="E.g. poor performance status, pneumonitis, active brain metastases."
            onChange={(e) => {
              editSummary({ otherHistory: e.target.value });
            }}
            className="field-sizing-fixed scrollbar-visible h-[calc(10lh+10px)] font-mono text-sm placeholder:font-sans placeholder:italic"
          />
        </Field>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-600">
        <span>
          {s.summaryNote ??
            (s.fromCorpus ? "From the chosen patient · editable" : "Your summary stays editable")}
        </span>
        <span>
          {count.toLocaleString()} {count === 1 ? "word" : "words"}
        </span>
        <Button
          size="xs"
          variant="outline"
          className="ml-auto"
          disabled={busy || (s.summary === CANCER_SUMMARY_SCAFFOLD && !s.otherHistory)}
          onClick={clearSummary}
        >
          Clear summary
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the notes

const CSV_HELP =
  "CSV: note_text, optional note_date, note_type and patient_id. One patient per file. PDF uploads need readable text (local OCR reads scanned pages).";

function NotesPane({ noun }: { noun: string }) {
  const s = usePatient();
  const id = useId();
  const config = useQuery(modelsOptions()).data;
  const sources = config?.synthetic_note_sources ?? [];
  const maxMb = config?.max_upload_mb ?? 20;
  const busy = isBusy(s);
  const hasNotes = !!s.notesText.trim();
  const tooLong = s.notesText.length > MAX_NOTE_CHARACTERS;
  const [table, setTable] = useState(false);
  const syntheticCase = s.syntheticCase || sources[0]?.id || "";
  return (
    <div className="flex flex-col gap-4">
      <div className="grid items-start gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Field
          label={
            s.notesOrigin?.kind === "corpus"
              ? `Notes of ${noun} ${s.notesOrigin.patientId}`
              : "Paste clinical notes"
          }
          htmlFor={`${id}-notes`}
          help={
            tooLong
              ? `These notes exceed the ${MAX_NOTE_CHARACTERS.toLocaleString()} character limit for summaries and workup reviews; split the record.`
              : (s.notesNotice ?? CSV_HELP)
          }
        >
          {s.corpusNotesPending ? (
            <p className="flex h-[calc(12lh+10px)] items-center justify-center gap-2 rounded-md border border-dashed border-gray-300 text-sm text-gray-500">
              <Spinner /> Loading the patient's notes (the server indexes them once)…
            </p>
          ) : (
            <Textarea
              id={`${id}-notes`}
              value={s.notesText}
              disabled={busy}
              spellCheck={false}
              maxLength={MAX_NOTE_CHARACTERS}
              placeholder="Paste the longitudinal record here. Include clinical dates when available."
              onChange={(e) => {
                editNotes(e.target.value);
              }}
              className="field-sizing-fixed scrollbar-visible h-[calc(12lh+10px)] font-mono text-xs"
            />
          )}
        </Field>
        <div className="flex flex-col gap-4">
          {sources.length ? (
            <Field label="Synthetic example" htmlFor={`${id}-case`}>
              <div className="flex gap-2">
                <NativeSelect
                  id={`${id}-case`}
                  className="min-w-0 flex-1"
                  value={syntheticCase}
                  disabled={busy}
                  onChange={(e) => {
                    setPatient({ syntheticCase: e.target.value });
                  }}
                >
                  {sources.map((source) => (
                    <option key={source.id} value={source.id}>
                      {source.label}
                    </option>
                  ))}
                </NativeSelect>
                <Button
                  variant="outline"
                  disabled={busy || !syntheticCase}
                  onClick={() => {
                    void loadSyntheticNotes(syntheticCase);
                  }}
                >
                  Load
                </Button>
              </div>
            </Field>
          ) : null}
          <label
            className={cn(
              "flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed border-gray-300 px-4 py-5 text-center text-sm hover:border-brand-blue hover:bg-brand-blue/5 has-focus-visible:ring-3 has-focus-visible:ring-ring/50",
              busy && "pointer-events-none opacity-50",
            )}
          >
            {s.loading ? <Spinner /> : <UploadIcon className="size-5 text-gray-500" />}
            <strong>Upload patient records</strong>
            <span className="text-xs text-gray-600">
              PDF, CSV, TXT or Markdown · up to {maxMb} MB
            </span>
            <input
              type="file"
              accept=".pdf,.csv,.txt,.md"
              className="sr-only"
              disabled={busy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) void uploadRecord(file, maxMb);
              }}
            />
          </label>
        </div>
      </div>
      <div className="flex flex-col gap-2 border-t border-gray-100 pt-3">
        <div className="flex flex-wrap items-end gap-2">
          <Button
            variant="outline"
            disabled={!hasNotes}
            onClick={() => {
              setTable(true);
            }}
          >
            <TableIcon />
            View notes table
          </Button>
          <Button className="ml-auto" disabled={busy || !hasNotes || tooLong} onClick={summarize}>
            Create patient summary →
          </Button>
        </div>
        <p className="text-xs text-gray-600">The summary fills Patient summary for both views.</p>
        <EndpointNotice tasks={["summarize"]} />
      </div>
      {table ? (
        <NotesDialog
          onClose={() => {
            setTable(false);
          }}
        />
      ) : null}
    </div>
  );
}
