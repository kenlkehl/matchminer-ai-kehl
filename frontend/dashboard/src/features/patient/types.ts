/**
 * The Patient tab's guideline APIs (the dashboard's routes/guidelines.py, with the results of
 * guidelines/services.py and guidelines/extraction_workup.py). Fields keep the server's snake_case.
 */
import type { Job } from "@/lib/api";
import type { LLMMatchAnswer, Scorer } from "@/lib/llm-decisions";

// ---------------------------------------------------------------- notes

/** One clinical note, as the guideline APIs take and return them. */
export interface GuidelineNote {
  note_text: string;
  /** ISO date (or date-time); null when the record gives none. */
  note_date: string | null;
  note_type?: string | null;
}

/** A patient's notes: the full text, and its separate notes when the source has them (CSV, corpus). */
export interface NotesInput {
  text: string;
  notes: GuidelineNote[] | null;
}

/** An uploaded record or a synthetic example, read into notes (`/api/patient/upload`, `.../synthetic-notes`). */
export interface ImportedNotes extends NotesInput {
  notice: string;
  source?: string;
  label?: string;
}

// ---------------------------------------------------------------- the guideline library and retrieval

export interface Catalog {
  id: string;
  disease: string;
  title: string;
  /** What the views show as its source: the title, or "Derived from PDQ®: <title>" for a PDQ-derived catalog. */
  label: string;
  version: string;
  spaces: number;
  updated: string;
}

export interface CatalogList {
  catalogs: Catalog[];
  spaces: number;
}

/** A guideline source passage that supports a catalog item. */
export interface SourceExcerpt {
  quote: string;
  printed_label?: string | null;
  page_id?: string | null;
  pdf_page?: number | null;
  line_ids?: (string | number)[] | null;
}

export interface CitationReview {
  issues?: string[] | null;
}

/** A workup item or treatment option of a guideline space. */
export interface Consideration {
  name: string;
  category: string;
  conditions?: string | null;
  evidence?: SourceExcerpt[] | null;
  citation_review?: CitationReview | null;
}

export interface GuidelineSpace {
  rank: number;
  name: string;
  /** `label` is what the views show: the title, or "Derived from PDQ®: <title>" for a PDQ-derived catalog. */
  source: { title: string; version: string; label?: string };
  space_trial_id: string;
  /** The decision model's ranking (absent when the LLM ranked the search: `llm_match`). */
  match_quality_score?: number;
  match_quality_pass?: boolean;
  /** The LLM's match answer, when the LLM ranked the search. */
  llm_match?: LLMMatchAnswer | null;
  decision_model_rank?: number;
  similarity_score: number;
  retrieval_rank: number;
  diagnostic_workup: Consideration[];
  treatment_options: Consideration[];
  uncertainties: string[];
  cohort_key_criteria?: string | null;
  clinical_space_summary?: string | null;
  evidence?: SourceExcerpt[] | null;
  citation_review?: CitationReview | null;
}

export interface SearchScope {
  catalogs: string[];
  top_n: number;
  candidate_k: number;
  scorer?: Scorer;
}

/** A finished guideline search (`/api/guidelines/retrieve`'s job result). */
export interface RetrieveResult {
  spaces: GuidelineSpace[];
  summary: string;
  /** The search's Markdown report (the package's write_guideline_considerations_report). */
  report: string;
  provenance: {
    catalog?: { catalog_spaces?: number } & Record<string, unknown>;
  } & Record<string, unknown>;
  search?: SearchScope;
  /** How the LLM ranked the candidates (an LLM-ranked search only). */
  ranking?: { scorer: "llm"; model: string; candidates: number; top_n: number; ranking: string };
}

// ---------------------------------------------------------------- workup reviews

export type WorkupMethod = "full_notes" | "agentic" | "extractions" | "extraction_sql";
export type ExtractionLevels = "document" | "document_and_patient";

/** A quote from the patient's notes that a finding rests on. */
export interface WorkupEvidence {
  quote: string;
  note_number?: number | null;
  note_date?: string | null;
  note_type?: string | null;
  /** Extraction reviews: whether the extracted item holds (yes, no, unknown, planned). */
  state?: string;
  criterion?: string;
  item?: string;
  date_text?: string;
  /** Extraction reviews: the quote's character span in its note's text. */
  start?: number | null;
  end?: number | null;
}

export interface FollowupReview {
  status?: string;
  max_review_passes?: number;
  full_record?: { complete?: boolean } | null;
}

export interface BackupEvent {
  reason: string;
  backup_model: string;
  primary_model: string;
  failure_threshold: number;
}

export interface BackupMetadata {
  backup_used?: boolean;
  backup_events?: BackupEvent[] | null;
  backup_items?: number;
  backup_resolution_failed?: boolean;
}

export interface QueryCriterion {
  label: string;
  item_type: string;
  status: string;
  state: string;
  definite_count: number;
  possible_count: number;
  min_count: number;
  anchor?: { kind: string; date?: string | null; fallback?: boolean } | null;
}

/** The SQL agent's query for one workup item. */
export interface SqlEntry {
  status?: string;
  expressible?: boolean | null;
  not_expressible_reason?: string | null;
  sql?: string | null;
  recommends_against?: boolean | null;
  accepted_by?: string | null;
  error?: string | null;
  action_count?: number | null;
  submit_count?: number | null;
  distribution?: { patients?: number; status_reason?: Record<string, number> } | null;
  cached?: boolean | null;
  model?: string | null;
}

export interface WorkupAssessment {
  name?: string;
  status: string;
  applicability: string;
  bottom_line: string;
  limitations?: string[] | null;
  evidence?: WorkupEvidence[] | null;
  metadata?:
    | ({ followup_review?: FollowupReview | null; evidence_selection?: string } & BackupMetadata)
    | null;
  // Extraction reviews:
  recommendation?: ({ name?: string } & Record<string, unknown>) | null;
  reason?: string;
  query_summary?: string | null;
  criteria?: QueryCriterion[] | null;
  sql?: string | null;
  recommendation_index?: number;
}

export interface ReviewMetadata extends BackupMetadata {
  method?: string;
  scope?: string;
  note_count?: number;
  requests?: number;
  model?: string | null;
  evidence_selection?: string;
  followup_review?: FollowupReview | null;
  extraction?: {
    reused?: boolean;
    requests?: number;
    levels?: string[];
    failed?: number;
    partial?: number;
  } | null;
  queries?: {
    compiled?: number;
    distinct?: number;
    cached?: number;
    requests?: number;
    exploration?: { name?: string; patients?: number } | null;
    queries?: ({ query?: unknown } & SqlEntry)[] | null;
  } | null;
}

export interface WorkupReview {
  assessments: WorkupAssessment[];
  metadata: ReviewMetadata;
  notice: string;
}

/** What an extraction review would send (`/api/guidelines/workup/extraction-cost`; no model calls). */
export interface ExtractionCost {
  notes: number;
  characters: number;
  document_requests: number;
  patient_requests: number;
  compile_requests: number;
  cached_queries: number;
  workup_items: number;
  requests: number;
  output_reserve_per_request: number;
  typical_reasoning_tokens: string;
  method: "extractions" | "extraction_sql";
  reused_extraction: boolean;
  /** Where reused records come from: the mode's extraction store, or this session's extraction. */
  records_source?: "store" | "session" | null;
  model: string;
  sql_agent?: {
    store: string;
    patients: number;
    database_built: boolean;
    items_to_explore: number;
    max_actions_per_item: number;
    model?: string | null;
  };
}

// ---------------------------------------------------------------- jobs and the saved workspace

export type GuidelineJobKind = "summarize" | "retrieve" | "workup";

export interface SummaryResult {
  summary: string;
  exclusion_evidence: string;
  notes_input?: NotesInput;
}

/** A workup job's result, and (without `review`) what it restores while it runs. */
export interface WorkupResult {
  review?: WorkupReview;
  space_index: number;
  method?: WorkupMethod;
  extraction_levels?: ExtractionLevels;
  retrieval_job_id: string;
  retrieval: RetrieveResult;
  notes_input?: NotesInput;
}

/** `/api/patient/session`: the session's latest guideline job. */
export interface SessionView {
  job: Job | null;
}
