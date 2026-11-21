/** Structured extraction records and criteria screening on them (routes/records.py). */

export interface RecordEvidence {
  note_id: string | null;
  note_date: string | null;
  quote: string;
  start: number | null;
  end: number | null;
  /** The note's 1-based position in the patient's notes as the app shows them; null when not among them. */
  note_number: number | null;
}

export interface RecordItem {
  item_id: string | null;
  type: string;
  label: string;
  date: string | null;
  date_text: string;
  status: string;
  fields: Record<string, unknown>;
  evidence: RecordEvidence[];
  /** The item as extracted (its record's JSON). */
  raw: unknown;
}

export interface RecordNote {
  note_id: string | null;
  note_number: number | null;
  note_date: string | null;
  note_type: string | null;
  document_type: string | null;
  status: string | null;
  items: RecordItem[];
  /** The note's whole document record, as extracted. */
  record: unknown;
}

/** One assessment of a cancer's status in a note (disease_assessment items, in date order). */
export interface CourseRow {
  date: string | null;
  note_id: string | null;
  note_number: number | null;
  note_type: string | null;
  document_type: string | null;
  status: string | null;
  sites: string[];
  site_groups: string[];
  evidence: RecordEvidence[];
  raw: unknown;
}

export interface CourseCancer {
  cancer_ids: string[];
  label: string;
  rows: CourseRow[];
}

export interface RecordsView {
  patient: {
    available: boolean;
    status: string | null;
    types: { type: string; label: string; items: RecordItem[] }[];
    /** The patient-level record, as extracted. */
    record: unknown;
  };
  documents: RecordNote[];
  disease_course: CourseCancer[];
  counts: {
    notes: number;
    patient_items: number;
    document_items: number;
    failed_notes: number;
    partial_notes: number;
  };
}

export interface RecordsSource {
  kind: "store" | "session";
  name?: string;
  created_utc?: string;
  model?: string | null;
  schema_version?: string | null;
  levels?: string[];
  requests?: number;
}

export interface StoreStatus {
  available: boolean;
  name?: string;
  created_utc?: string;
  patients?: number;
  model?: string | null;
  error?: string;
}

export interface ExtractionEstimate {
  notes: number;
  document_requests: number;
  patient_requests: number;
  requests: number;
  approximate_note_tokens: number;
  levels: string[];
  model: string;
}

export interface RecordsResponse {
  source: RecordsSource | null;
  view: RecordsView | null;
  estimate: ExtractionEstimate | null;
  store?: StoreStatus;
}

export type Levels = "document" | "document_and_patient";
export type Engine = "queries" | "sql";

/** Which records a request means: a corpus patient's stored records, else the workspace notes' extraction. */
export interface RecordsBody {
  patient_id?: string | null;
  text?: string | null;
  notes?: { note_text: string; note_date: string | null; note_type?: string | null }[] | null;
  levels?: Levels;
}

export type StructuredStatus =
  | "clearly_fails"
  | "meets"
  | "not_documented"
  | "uncertain"
  | "conflicting"
  | "not_applicable"
  | "not_checkable"
  | "no_records"
  | "query_unavailable";

export interface StructuredEvidence extends RecordEvidence {
  side: "fails" | "meets" | "not_applicable";
  item?: string;
  date_text?: string;
  criterion?: string;
}

export interface StructuredLine {
  number: number;
  text: string;
  boilerplate: boolean;
  status: StructuredStatus;
  clearly_fails: boolean;
  reason: string;
  applicability: string;
  criterion_type: "inclusion" | "exclusion" | null;
  bottom_line: string;
  evidence: StructuredEvidence[];
  limitations: string[];
  query_summary: string | null;
}

export interface CompileSummary {
  engine: Engine;
  model: string | null;
  compiled_by?: string[];
  requests: number | null;
  cached: number | null;
  compiled: number | null;
  distinct: number | null;
  failed?: number | null;
  unchecked_lines?: number | null;
}

export interface LineCounts extends Record<StructuredStatus, number> {
  lines: number;
  checked: number;
  clearly_failed_lines: number[];
}

export interface CriteriaRecordsResult {
  cohort_id: string;
  engine: Engine;
  source: RecordsSource;
  lines: StructuredLine[];
  counts: LineCounts;
  compile: CompileSummary;
  notice: string;
}

export interface ScreenResult {
  cohort_id: string;
  engine: Engine;
  store: RecordsSource;
  patients: Record<
    string,
    { counts: LineCounts; failed: { number: number; text: string; bottom_line: string }[] }
  >;
  without_records: string[];
  aggregate: {
    number: number;
    text: string;
    criterion_type: string | null;
    clearly_fail: number;
    checked: number;
    patient_count: number;
    statuses: Record<StructuredStatus, number>;
  }[];
  compile: CompileSummary;
  notice: string;
}

export const ENGINE_LABELS: Record<Engine, string> = {
  queries: "Compiled queries",
  sql: "SQL agent",
};

export const STATUS_LABELS: Record<StructuredStatus, string> = {
  clearly_fails: "Clearly fails",
  meets: "Met",
  not_documented: "Not documented",
  uncertain: "Uncertain",
  conflicting: "Conflicting",
  not_applicable: "Not applicable",
  not_checkable: "Not checkable from records",
  no_records: "No records",
  query_unavailable: "No query",
};

/** The source of records in a sentence. */
export function describeSource(source: RecordsSource): string {
  if (source.kind === "store") {
    return `Stored records (${source.name ?? "extraction store"}${source.model ? `, ${source.model}` : ""}${
      source.created_utc ? `, built ${source.created_utc.slice(0, 10)}` : ""
    })`;
  }
  return `Extracted in this session${source.model ? ` with ${source.model}` : ""}`;
}
