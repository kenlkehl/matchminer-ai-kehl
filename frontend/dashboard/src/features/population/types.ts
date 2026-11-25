/**
 * The Population tab's server shapes (routes/population.py, guidelines/population.py and the package's
 * matchminer_ai.population reports). Fields keep the server's snake_case.
 */
import type { Job } from "@/lib/api";

export type ReviewMethod = "full_notes" | "extraction_queries" | "extraction_sql";
export type Applicability = "applies" | "not_applicable" | "uncertain";

/** One diagnostic-workup item of a guideline space. */
export interface Recommendation {
  name: string;
  conditions?: string | null;
  [key: string]: unknown;
}

/** A guideline trial space (a row of `/api/population/spaces`, or the active cohort's space). */
export interface GuidelineSpace {
  space_trial_id: string;
  name?: string | null;
  cohort_key_criteria?: string | null;
  clinical_space_summary?: string | null;
  disease?: string | null;
  title?: string | null;
  diagnostic_workup?: Recommendation[];
}

export interface CatalogInfo {
  id: string;
  disease: string;
  title: string;
  /** What the views show: the title, or "Derived from PDQ®: <title>" for a PDQ-derived catalog. */
  label: string;
  version: string;
  spaces: number;
  updated: string;
}

/** A retrieved patient's scores (the active cohort's `matches`): the decision model's, or (an LLM-selected
 * cohort) the LLM's 0-6 match score and reasonableness answer, with the key of its full answer. */
export interface MatchRow {
  patient_id: string;
  match_quality_reasonableness?: number | null;
  match_quality_score?: number | null;
  llm_match_score?: number | null;
  llm_reasonable?: boolean | null;
  llm_answer_key?: string | null;
  similarity_score: number | null;
  retrieval_rank: number;
}

export interface Retrieval {
  candidate_k?: number;
  scored_count?: number;
  population_count?: number;
  selected_count?: number;
  min_reasonableness?: number | null;
  min_match_score?: number | null;
  /** An LLM-selected cohort: who selected and ranked it. */
  scorer?: { kind: "llm"; model: string; selection: string; ranking: string };
}

/** One workup item's counts across the active cohort. */
export interface AggregateRow {
  recommendation_index: number;
  recommendation: Recommendation;
  patient_count: number;
  statuses: Record<string, number>;
  applicability: Record<Applicability, number>;
  applicable_assessed: number;
  completed_applicable: number;
  documented_completion_fraction: number | null;
  recommends_against?: boolean;
}

/** A compiled extraction query (extraction_queries) or the SQL agent's entry (extraction_sql). */
export interface QueryEntry {
  status?: string | null;
  query?: unknown;
  sql?: string | null;
  error?: string | null;
  expressible?: boolean | null;
  not_expressible_reason?: string | null;
  accepted_by?: string | null;
  action_count?: number | null;
  submit_count?: number | null;
  cached?: boolean | null;
  model?: string | null;
  recommends_against?: boolean | null;
  distribution?: { patients?: number; status_reason?: Record<string, number> } | null;
}

export interface PopulationReport {
  method: ReviewMethod;
  max_concurrent_patients?: number;
  patient_count: number;
  aggregate: AggregateRow[];
  notice: string;
  connection?: { provider: string; model: string; endpoint: string; reasoning: boolean };
  extraction?: {
    documents_used?: number;
    documents_ignored?: number;
    notes_without_records?: number;
  };
  compile?: {
    compiled?: number;
    distinct?: number;
    cached?: number;
    requests?: number;
    model?: string | null;
    exploration?: { patients?: number };
  };
  extraction_store?: { name?: string; levels?: string[] };
  queries?: (QueryEntry | null)[];
}

export interface ActiveCohort {
  patient_count: number;
  note_count: number;
  space: GuidelineSpace | null;
  matches: MatchRow[];
  report: PopulationReport | null;
  retrieval: Retrieval;
}

/** A browser session's population (`PopulationSessions.view`); `version` is the optimistic token. */
export interface PopulationView {
  version: string;
  source_count: number;
  source_note_count: number;
  notes_on_demand: boolean;
  /** Whether the patients are already embedded with TrialSpace (made when the population loads). */
  embedded?: boolean;
  provenance: { cache?: { name?: string }; database?: { name?: string }; [key: string]: unknown };
  active: ActiveCohort | null;
}

export interface PopulationDefaults {
  size: number;
  seed: number;
  extraction_records: string;
  /** Synthetic mode: the split samples come from unless training patients are included (null: every patient). */
  split?: string | null;
  /** Real mode only. */
  database?: string;
  cache?: string;
}

/** `GET /api/population`. */
export interface PopulationState {
  population: PopulationView | null;
  job: Job<PopulationView | null> | null;
  defaults: PopulationDefaults;
}

// ---------------------------------------------------------------- real-mode sources

export interface CacheInfo {
  path: string;
  name: string;
  created_utc: string;
  patients: number;
  notes: number;
  database: string;
  sample_limit: number | null;
  embedding_model: string | null;
}

export interface DatabaseInfo {
  path: string;
  name: string;
  size: number;
  mtime_ns: number;
}

export interface FolderListing {
  path: string;
  parent: string | null;
  roots: string[];
  directories: string[];
  files: { name: string; path: string; size: number; modified: number }[];
  caches: { name: string; path: string }[];
  truncated: boolean;
}

export interface ExtractionStoreInfo {
  path: string;
  name: string;
  created_utc: string;
  levels: string[];
  counts: {
    patients: number;
    documents?: { complete: number; partial: number; failed: number } | null;
    patient_records?: { complete: number; partial: number } | null;
  };
  source: { name?: string };
  schema_version: string;
  model: string | null;
}

// ---------------------------------------------------------------- drilldowns

/** A quoted passage: full-note reviews give its note number; extraction records its note id and span. */
export interface Evidence {
  quote: string;
  note_number?: number | null;
  note_date?: string | null;
  note_type?: string | null;
  note_id?: string | null;
  start?: number | null;
  end?: number | null;
  state?: string | null;
  criterion?: string | null;
  item?: string | null;
  date_text?: string | null;
}

export interface QueryCriterion {
  label: string;
  item_type: string;
  status: string;
  state: string;
  min_count: number;
  definite_count: number;
  possible_count: number;
  anchor: { kind: string; date?: string | null; fallback?: boolean } | null;
}

export interface Assessment {
  recommendation_index: number;
  recommendation: Recommendation;
  name?: string;
  status: string;
  applicability: Applicability;
  bottom_line: string;
  limitations?: string[];
  evidence?: Evidence[];
  reason?: string | null;
  criteria?: QueryCriterion[];
  query_summary?: string | null;
  sql?: string | null;
}

export interface PatientReview {
  assessments: Assessment[];
  notice: string;
  metadata?: {
    method?: string;
    model?: string;
    coverage?: {
      documents?: number;
      patient_records?: number;
      notes_without_records?: number;
      failed_documents?: number;
      items?: number;
    };
  };
}

export interface PopulationNote {
  note_text: string;
  note_date: string | null;
  note_type: string | null;
  note_id?: string | null;
}

/** `GET /api/population/patient`. */
export interface PatientDrilldown {
  patient: { patient_id: string; cancer_history_summary: string };
  notes: PopulationNote[];
  review: PatientReview | null;
  space: GuidelineSpace | null;
}

/** A row of `GET /api/population/items/{n}` (the assessment without its evidence). */
export interface ItemPatient {
  patient_id: string;
  status: string;
  applicability: Applicability;
  bottom_line: string;
  reason?: string | null;
}
