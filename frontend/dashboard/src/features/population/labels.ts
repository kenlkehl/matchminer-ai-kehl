/** The Population tab's wording for statuses, methods and file facts (the guideline fetcher's labels). */
import type {
  Applicability,
  CacheInfo,
  DatabaseInfo,
  ExtractionStoreInfo,
  ReviewMethod,
} from "./types";

export const isExtractionMethod = (method: string | null | undefined) =>
  method === "extraction_queries" || method === "extraction_sql";

export const METHOD_OPTIONS: { value: ReviewMethod; label: string }[] = [
  { value: "full_notes", label: "Full-note review · the LLM reads each patient's notes" },
  {
    value: "extraction_queries",
    label: "Extraction queries · code checks precomputed extraction records",
  },
  {
    value: "extraction_sql",
    label: "SQL agent · an LLM writes one SQL query per workup item over the extraction records",
  },
];

export const METHOD_NAMES: Record<ReviewMethod, string> = {
  full_notes: "Full-note review",
  extraction_queries: "Extraction queries",
  extraction_sql: "SQL agent over extraction records",
};

/** Full-note review statuses, in the report's order. */
const STATUS_LABELS: Record<string, string> = {
  completed: "Completed",
  partially_completed: "Partially completed",
  planned: "Planned / pending",
  not_done: "Explicitly not done",
  not_documented: "Not documented",
  unclear: "Unclear / conflicting",
  error: "Review failed",
};

/** The population view's full-note statuses (not documented: in the notes it reviewed). */
const POPULATION_STATUS_LABELS: Record<string, string> = {
  ...STATUS_LABELS,
  not_documented: "Not documented in reviewed notes",
};

export const EXTRACTION_STATUS_LABELS: Record<string, string> = {
  met: "Met · documented in extraction",
  not_met: "Not documented in extraction",
  unknown: "Unknown",
  not_applicable: "Not applicable",
  done_not_recommended: "Done though not recommended",
};

export const EXTRACTION_REASON_LABELS: Record<string, string> = {
  documented: "Documented by extracted items",
  not_documented: "No extracted item matched",
  planned_only: "Only planned or ordered",
  uncertain_dates: "Dates too imprecise to tell",
  fields_not_stated: "Needed details not stated",
  anchor_unavailable: "Reference event undated or missing",
  incomplete_extraction: "Some notes lack extraction records",
  no_extraction: "No extraction records",
  not_expressible: "Not checkable from extracted items",
  query_unavailable: "No valid query compiled",
  documented_not_applicable: "Documented facts make it not apply",
  documented_not_recommended: "Documented although the guideline advises against it",
};

export const APPLICABILITY_LABELS: Record<Applicability, string> = {
  applies: "Applies",
  not_applicable: "Not applicable",
  uncertain: "Uncertain",
};

export const STATE_LABELS: Record<string, string> = {
  yes: "yes",
  unknown: "unknown",
  no: "no",
  planned: "planned",
};

/** The status columns and filter options of a method's report. */
export function methodLabels(method: string | null | undefined): Record<string, string> {
  return isExtractionMethod(method) ? EXTRACTION_STATUS_LABELS : POPULATION_STATUS_LABELS;
}

/** One patient's status for an item, as the patient view words it. */
export function workupStatusLabel(status: string, method: string | undefined, compressed: boolean) {
  if (isExtractionMethod(method)) return EXTRACTION_STATUS_LABELS[status] ?? status;
  if (compressed && status === "not_documented") return "Not documented in compressed notes";
  return STATUS_LABELS[status] ?? status;
}

const REVIEW_AID =
  "Review aid, not a determination: an LLM compiled each workup item into a query and extracted the items from the notes; code ran the query. Check each cited quote against its note. Not documented does not mean not done.";
const SQL_REVIEW_AID =
  "Review aid, not a determination: an LLM agent wrote each workup item's SQL query after exploring extracted items, an LLM extracted the items from the notes, and code ran each query once over the cohort. Check each cited quote against its note. Not documented does not mean not done.";

export const reviewAid = (method: string | undefined) =>
  method === "extraction_sql" ? SQL_REVIEW_AID : REVIEW_AID;

// ---------------------------------------------------------------- numbers and file facts

export const count = (n: number | null | undefined) => (n ?? 0).toLocaleString();

export function fileSize(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${String(Math.ceil(n / 1e3))} kB`;
}

export const fileDate = (seconds: number) => new Date(seconds * 1000).toLocaleDateString();

export const score = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(3);

export function cacheLine(c: CacheInfo): string {
  const sample = c.sample_limit ? ` (trial sample of ${c.sample_limit.toLocaleString()})` : "";
  return `${c.patients.toLocaleString()} patients · ${c.notes.toLocaleString()} notes · built ${new Date(c.created_utc).toLocaleString()} from ${c.database}${sample} · ${c.embedding_model ?? "TrialSpace"}`;
}

export function databaseLine(d: DatabaseInfo): string {
  return `${d.name} · ${fileSize(d.size)} · modified ${fileDate(d.mtime_ns / 1e9)} · patient tables found`;
}

export function storeLine(x: ExtractionStoreInfo): string {
  const docs = x.counts.documents;
  const patients = x.counts.patient_records;
  return [
    `${x.counts.patients.toLocaleString()} patients`,
    `${x.levels.join(" + ")} records`,
    docs
      ? `${(docs.complete + docs.partial).toLocaleString()} notes extracted, ${docs.failed.toLocaleString()} failed`
      : null,
    patients ? `${String(patients.complete + patients.partial)} patient records` : null,
    `built ${new Date(x.created_utc).toLocaleString()} by ${x.model ?? "an unrecorded model"}${x.source.name ? ` from ${x.source.name}` : ""}`,
  ]
    .filter(Boolean)
    .join(" · ");
}
