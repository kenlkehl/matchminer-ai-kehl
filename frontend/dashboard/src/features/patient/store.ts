import { useSyncExternalStore } from "react";

import type { JobStatus } from "@/lib/api";

import type {
  CompressionStats,
  ExtractionLevels,
  GuidelineJobKind,
  GuidelineNote,
  NoteSource,
  RetrieveResult,
  SavedCompression,
  WorkupMethod,
  WorkupReview,
} from "./types";

// ---------------------------------------------------------------- the URL

export type PatientView = "trials" | "guidelines";

/** The Patient tab's URL state: its sub-tab, and the Trials view's open dialogs. */
export interface PatientSearch {
  /** The sub-tab; Trials when absent. */
  view?: PatientView;
  /** The trial whose Drugs / mechanism dialog is open. */
  drugs?: string;
  /** The trial whose Good option check dialog is open. */
  good?: string;
  /** The catalog evidence dialog open over it: `drug:<id>` or `class:<id>`. */
  evidence?: string;
}

export function validatePatientSearch(search: Record<string, unknown>): PatientSearch {
  const text = (v: unknown) => (typeof v === "string" && v ? v : undefined);
  return {
    view: search.view === "guidelines" ? "guidelines" : undefined,
    drugs: text(search.drugs),
    good: text(search.good),
    evidence: text(search.evidence),
  };
}

// ---------------------------------------------------------------- the shared patient

export const CANCER_SUMMARY_SCAFFOLD = `Age:
Sex:
Cancer type:
Histology:
Current extent:
Biomarkers:
Treatment history:
`;

/** Whether a cancer summary has been written (not empty, nor the scaffold's headings alone). */
export const hasSummary = (summary: string) =>
  !!summary.trim() && summary.trim() !== CANCER_SUMMARY_SCAFFOLD.trim();

export type InputTab = "corpus" | "summary" | "notes";

/** Where the current notes came from. */
export type NotesOrigin =
  | { kind: "corpus"; patientId: string }
  | { kind: "synthetic"; label: string }
  | { kind: "upload"; name: string }
  | { kind: "pasted" }
  | { kind: "restored" };

/** The session's running guideline job (summary, compression, guideline search or workup review). */
export interface ActiveJob {
  id: string | null;
  kind: GuidelineJobKind;
  title: string;
  status: JobStatus;
  message: string;
  done: number;
  total: number;
  /** When it started, in milliseconds since the epoch. */
  started: number;
  stopping: boolean;
  /** A workup review's space (its index in the guideline search). */
  space?: number;
}

/** A message for one area of the page: the patient input, or the Guidelines view. */
export interface StatusMessage {
  where: "input" | "guidelines";
  kind: "info" | "error";
  text: string;
}

/**
 * The Patient tab's state, shared by its Trials and Guidelines views and kept in memory while other
 * tabs show: the patient (a corpus patient, an entered summary, or clinical notes and the summary made
 * from them), the guideline search and its workup reviews, and the session's guideline job. A reload
 * starts over, then restores the session's latest guideline job from the server.
 */
export interface PatientState {
  input: InputTab;
  /** Whether the patient card shows its inputs (it folds to one line after a search). */
  inputOpen: boolean;
  /** The corpus patient chosen. */
  patientId: string | null;
  /** Whether the summary and other history are still the chosen corpus patient's own. */
  fromCorpus: boolean;
  /** The cancer history summary. */
  summary: string;
  /** Other relevant history: the evidence about common exclusions. */
  otherHistory: string;
  /** Where the summary came from, when it was generated or restored. */
  summaryNote: string | null;
  /** The compression a generated summary was made from. */
  summaryCompression: CompressionStats | null;
  notesText: string;
  /** The separate notes, when the source has them (CSV, corpus patient); else the text is one note. */
  notes: GuidelineNote[] | null;
  notesOrigin: NotesOrigin | null;
  notesNotice: string | null;
  /** Counts changes of the notes. */
  notesVersion: number;
  /** A corpus patient whose notes are being loaded. */
  corpusNotesPending: string | null;
  /** An upload or a synthetic example is being read. */
  loading: boolean;
  syntheticCase: string;
  noteSource: NoteSource;
  noteCompression: SavedCompression | null;

  catalog: string;
  topN: number;
  candidateK: number;
  retrieval: RetrieveResult | null;
  retrievalJobId: string | null;
  /** Completed workup reviews by space index (of `retrieval`). */
  reviews: Partial<Record<number, WorkupReview>>;
  workupMethod: WorkupMethod;
  extractionLevels: ExtractionLevels;
  /** The space cards that are open. */
  openSpaces: number[];

  job: ActiveJob | null;
  status: StatusMessage | null;
  /** Whether the session's saved workspace has been read since the page loaded. */
  restored: boolean;
}

export function initialPatientState(): PatientState {
  return {
    input: "corpus",
    inputOpen: true,
    patientId: null,
    fromCorpus: false,
    summary: CANCER_SUMMARY_SCAFFOLD,
    otherHistory: "",
    summaryNote: null,
    summaryCompression: null,
    notesText: "",
    notes: null,
    notesOrigin: null,
    notesNotice: null,
    notesVersion: 0,
    corpusNotesPending: null,
    loading: false,
    syntheticCase: "",
    noteSource: "original",
    noteCompression: null,
    catalog: "",
    topN: 1,
    candidateK: 5,
    retrieval: null,
    retrievalJobId: null,
    reviews: {},
    workupMethod: "full_notes",
    extractionLevels: "document",
    openSpaces: [],
    job: null,
    status: null,
    restored: false,
  };
}

let state: PatientState = initialPatientState();
const listeners = new Set<() => void>();

export function getPatient(): PatientState {
  return state;
}

export function setPatient(
  patch: Partial<PatientState> | ((s: PatientState) => Partial<PatientState>),
): void {
  state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
  for (const listener of listeners) listener();
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export function usePatient(): PatientState {
  return useSyncExternalStore(subscribe, () => state);
}

/** One value of the state (re-rendering only when it changes): a field, or a derived primitive. */
export function usePatientValue<T>(select: (s: PatientState) => T): T {
  return useSyncExternalStore(subscribe, () => select(state));
}

/** Whether a guideline job runs, or an upload or example is being read. */
export const isBusy = (s: PatientState) => s.job !== null || s.loading;

export const usesExtraction = (method: WorkupMethod) =>
  method === "extractions" || method === "extraction_sql";
