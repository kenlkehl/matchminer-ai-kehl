/** The dashboard server's JSON API (matchminer_ai/apps/dashboard/server.py). Fields keep the server's snake_case. */

export class ApiError extends Error {
  status: number;
  body: Record<string, unknown>;
  constructor(status: number, message: string, body: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // A FormData body (a file upload) sets its own multipart content type.
  const json = !(init?.body instanceof FormData);
  const response = await fetch(path, {
    ...init,
    headers: { ...(json ? { "Content-Type": "application/json" } : {}), ...(init?.headers ?? {}) },
  });
  if (response.status === 401 && path.startsWith("/api/")) {
    // Signed out (real mode always signs in): back to the sign-in page.
    window.location.assign("/login");
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { error: text };
  }
  if (!response.ok) {
    const b = (body ?? {}) as Record<string, unknown>;
    const message =
      (typeof b.error === "string" && b.error) ||
      (typeof b.detail === "string" && b.detail) ||
      `${String(response.status)} ${response.statusText}`;
    throw new ApiError(response.status, message, b);
  }
  return body as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  delete: <T = null>(path: string) => request<T>(path, { method: "DELETE" }),
  /** A multipart upload: each entry a file (or a text field). */
  upload: <T>(path: string, fields: Record<string, Blob | string>) => {
    const form = new FormData();
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    return request<T>(path, { method: "POST", body: form });
  },
};

/** Download a server response as a file (the server names it in Content-Disposition). */
export async function download(path: string, fallbackName: string): Promise<void> {
  const response = await fetch(path);
  if (!response.ok) {
    const text = await response.text();
    let message = `${String(response.status)} ${response.statusText}`;
    try {
      const body = JSON.parse(text) as { error?: string };
      if (body.error) message = body.error;
    } catch {
      // not JSON
    }
    throw new ApiError(response.status, message);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const name = /filename="([^"]+)"/.exec(disposition)?.[1] ?? fallbackName;
  saveBlob(await response.blob(), name);
}

/** Save text or a blob made in the browser (an exported report). */
export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------- dashboard

export interface Contact {
  name: string;
  email: string;
  role: string;
}

export type Version = "protocol" | "ctgov";
export const VERSION_LABELS: Record<Version, string> = { protocol: "Protocol", ctgov: "CT.gov" };

/** A trial's drug in the GoodOption catalog (LLM-assisted roles, NCIt classes): `trial_drug_index` with its classes. */
export interface DrugBrief {
  drug_id: string;
  name: string;
  /** The name the CT.gov listing gives, when it differs. */
  listed_name: string | null;
  /** investigational, uncertain, control or background (supportive drugs are left out). */
  role: string;
  /** Its most specific classes (none an NCIt ancestor of another). */
  mechanism_types: string[];
  /** Every class, generic ones included (the mechanism filter). */
  all_classes: string[];
  /** Each class name's catalog ID (the evidence dialogs); absent from a store older than the dialogs. */
  class_ids?: Record<string, string>;
  /** An antibody-drug conjugate's payload, when its classes or facts name one. */
  payload: string | null;
  /** The classes that name the payload, when they do (else `payload` came from the catalog's facts). */
  payload_classes?: PayloadClass[];
}

export interface PayloadClass {
  /** "Exatecan Derivative", from "Exatecan Derivative-containing Antibody-drug Conjugate". */
  label: string;
  class_name: string;
  /** The more generic payload classes above it. */
  generic: { label: string; class_name: string }[];
}

export interface Drug extends DrugBrief {
  arms: string[];
  /** The catalog's first cited mechanism-and-targets fact. */
  mechanism: string | null;
}

export interface TrialSummary {
  id: string;
  nct_id: string;
  protocol_no: string;
  short_title: string;
  phase: string;
  status: string;
  is_open: boolean;
  center: string;
  center_id: string;
  /** The center's name without the site's home-institution prefix. */
  center_short_name: string;
  site_pi: string;
  overall_pi: string;
  contact: Contact;
  sites: string[];
  primary_version: Version | null;
  state: string;
  /** In role order; absent from a server older than the drugs. */
  drugs?: DrugBrief[];
}

export interface CohortRow {
  cohort_id: string;
  trial: TrialSummary;
  name: string;
  number: number;
  version: Version;
  merged_from: number;
  group: string | null;
  group_reason: string | null;
  cancer_types: string[];
  key_inclusion: string[];
  key_exclusion: string[];
  common_count: number;
  line_count: number;
  completed_at: string | null;
  stale: boolean;
}

export interface CriteriaItem {
  number: number;
  heading: string;
  text: string;
  criterion_type: "inclusion" | "exclusion" | null;
  boilerplate: boolean;
}

export interface CohortDetail extends CohortRow {
  key: string;
  boilerplate: string;
  items: CriteriaItem[];
}

export interface CenterInfo {
  id: string;
  name: string;
  short_name: string;
  trials: number;
  cohorts: number;
  pending: number;
}

export interface CancerType {
  key: string;
  label: string;
  group: string;
}

/** What the dashboard shows of the site (the server's site config). */
export interface SiteBranding {
  institution: string;
  confidentiality: string;
  pi_label: string;
  /** The AI tab's checkbox that leaves out the higher_threshold_groups. */
  higher_threshold_label: string;
  /** Disease centers grouped by institution: a center joins the first group one of whose prefixes it starts with. */
  institutions: { label: string; prefixes: string[] }[];
  /** Links from a trial's protocol number: URL templates with {protocol_no}. */
  trial_links: { label: string; url: string }[];
}

/** The app's mode: real (the OA trials and real patients) or synthetic (the synthetic run's). */
export type Mode = "real" | "synthetic";
export type AppTabName =
  "trials" | "population" | "patient" | "assistant" | "benchmarking" | "status";

export interface Meta {
  mode: Mode;
  /** The tabs this mode shows (Benchmarking only when there are runs). */
  tabs: AppTabName[];
  /** Whether this server asks for the access token (always in real mode). */
  sign_in: boolean;
  built_at: number;
  counts: {
    trials: number;
    cohorts: number;
    dashboard_cohorts: number;
    trials_by_state: Record<string, number>;
  };
  higher_threshold_groups: string[];
  centers: CenterInfo[];
  cancer_types: CancerType[];
  auto_score_n: number;
  default_k: number;
  max_k: number;
  round: number | null;
  vectors: { state: string; vectors: number; error: string | null };
  patients: number;
  /** Absent from an older server. */
  site?: SiteBranding;
  /** The GoodOption catalog's state; null: a store built before the drugs; absent: an older server. */
  good_options?: { state: "ready" | "missing" | "error"; message: string | null } | null;
}

export interface OverviewGroup {
  id: string;
  name: string;
  trial_count: number;
  rows: CohortRow[];
}

export interface PendingTrial extends TrialSummary {
  versions: Record<Version, { state: string; excluded_reason: string | null }>;
}

export interface CenterView {
  id: string;
  name: string;
  short_name: string;
  built_at: number;
  trial_count: number;
  groups: OverviewGroup[];
  pending: PendingTrial[];
  cancer_type_labels: Record<string, string>;
}

export interface TrialVersion {
  state: string;
  completed_at: string | null;
  screen: string | null;
  extracted_cohorts: number | null;
  excluded_reason: string | null;
  cohorts: CohortDetail[];
}

export interface TrialDetail extends TrialSummary {
  drugs?: Drug[];
  long_title: string;
  official_title: string;
  brief_summary: string;
  detailed_description: string;
  versions: Record<Version, TrialVersion>;
}

// ---------------------------------------------------------------- GoodOption catalog evidence

/** A drug's or class's evidence dialog: `drug:<id>` or `class:<id>`. */
export type CatalogKind = "drug" | "class";

export interface CatalogFact {
  text: string;
  /** [name, value] in reading order (scope, evidence level, tumor type, histology, regimen ...). */
  qualifiers: [string, string][];
  /** Evidence IDs; one missing from `evidence` is "source not in catalog". */
  support_ids: string[];
}

export interface CatalogFacet {
  facet: string;
  label: string;
  facts: CatalogFact[];
}

export interface CatalogPassage {
  evidence_id: string;
  source: string;
  source_type: string;
  title: string;
  passage: string;
  /** http(s) only; empty otherwise. */
  url: string;
  locator: string;
  published_at: string;
  license: string;
}

export interface CatalogEntry {
  kind: CatalogKind;
  id: string;
  name: string;
  ncit_code: string | null;
  label: string;
  summary: string | null;
  evidence_count: number | null;
  facets: CatalogFacet[];
  evidence: Record<string, CatalogPassage>;
  missing_citations: string[];
  counts: { facts: number; citations: number; cited_sources: number; resolved_sources: number };
  catalog: { path: string; build_id: string | null; created_at: string | null };
  /** A drug's classes; `specific`: among its mechanism types here (null when no trial here lists it). */
  classes?: {
    id: string;
    name: string;
    ncit_code: string;
    assigned_by: string;
    specific: boolean | null;
  }[];
  /** A class's drugs in the catalog. */
  drugs?: { id: string; name: string }[];
  /** The store's trials with a drug in the class. */
  trials?: {
    id: string;
    protocol_no: string;
    short_title: string;
    center: string;
    state: string;
    drugs: { name: string; role: string }[];
  }[];
}

// ---------------------------------------------------------------- LLM checks (dashboard/llm.py, llm_checks.py)

export interface LLMConnection {
  /** "openai" (an OpenAI-compatible URL) or a Vertex profile key. */
  backend: string;
  endpoint: string;
  model: string;
  tokenizer: string;
  configured: boolean;
  label: string;
  vertex: boolean;
  project: string | null;
}

export interface Availability {
  available: boolean;
  reason: string | null;
}

export interface RubricCriterion {
  criterion: string;
  question: string;
  text: string;
}

export interface LLMStatus {
  connection: LLMConnection;
  notice: string;
  defaults: { backend: string; endpoint: string; model: string };
  vertex_profiles: Record<
    string,
    { endpoint: string; model: string; tokenizer: string; label: string; project: string }
  >;
  good_option: Availability;
  criteria: Availability;
  package: { version: string | null; path: string };
  rubric: { intro: string; criteria: RubricCriterion[]; score_meaning: string };
}

export interface GoodOptionAssessment {
  drug_name: string;
  drug_id: string | null;
  role: string | null;
  mechanism_types: string[];
  class_ids: Record<string, string>;
  targeted_biomarkers: string[];
  points: number;
  criteria: { criterion: string; point: number; rationale: string }[];
}

export interface GoodOptionResult {
  kind: "good_option";
  patient_id: string;
  trial_id: string;
  /** ok, parse_failed, no_scoreable_drug, missing_catalog_trial, trial_ctgov_blocked, ... */
  status: string;
  status_text: string;
  score: number | null;
  points: number | null;
  max_points: number | null;
  drug_count: number | null;
  patient_disease_type: string;
  assessments: GoodOptionAssessment[];
  uncertainties: string[];
  reasoning: string;
  finish_reason: string;
  parse_error: string;
  /** Requests sent for the pair; null from a package that does not report it. */
  parse_attempts: number | null;
  reasoning_off_fallback_used: boolean | null;
  max_parse_attempts: number | null;
  reasoning_off_fallback: boolean | null;
  evidence_packing: Record<string, unknown>;
  context: { context_tokens?: number | null; context_source?: string; context_warning?: string };
  model_metadata: Record<string, string | number | boolean>;
  connection: LLMConnection;
  seconds: number;
  checked_at: number;
  cached?: boolean;
}

export interface LLMCriterionLine {
  number: number;
  type: "inclusion" | "exclusion" | null;
  boilerplate: boolean;
  text: string;
  /** yes = does not clearly fail the line; no = clearly fails it; null when no attempt validated. */
  answer: "yes" | "no" | null;
  fails: boolean | null;
  status: string | null;
  reason: string;
  finish_reason: string;
  attempts: number;
  parse_status: string;
  reasoning: string;
}

export interface CriteriaLLMResult {
  kind: "criteria_llm";
  lines: LLMCriterionLine[];
  counts: {
    lines: number;
    clearly_fail: number;
    unanswered: number;
    requests: number | null;
    retries: number | null;
    parse_failures: number | null;
  };
  structured_outputs: boolean | null;
  connection: LLMConnection;
  seconds: number;
  checked_at: number;
  cached?: boolean;
}

/** A check's cached answer, or the job computing it. */
export interface CheckStart<T> {
  result?: T;
  job?: Job<T>;
}

/** Start an LLM check (or get its cached answer) and wait for it, reporting the job's progress. */
export async function runCheck<T>(
  path: string,
  body: unknown,
  onProgress?: (job: Job<T>) => void,
): Promise<T> {
  const started = await api.post<CheckStart<T>>(path, body);
  if (started.result !== undefined) return started.result;
  if (!started.job) throw new ApiError(500, "the server started no job");
  return waitForJob<T>(started.job.id, onProgress, 1500);
}

// ---------------------------------------------------------------- patients and matching

export interface PatientSummary {
  id: string;
  custom: boolean;
  label?: string;
  age?: string;
  sex?: string;
  cancer_type?: string;
  histology?: string;
  cls?: string | null;
  cancer_types?: string[];
  /** The mode's extraction store holds this patient (patient lists only). */
  has_records?: boolean;
}

export interface PatientDetail extends PatientSummary {
  key: string;
  boilerplate: string;
  cancer_type_labels: string[];
}

export interface PatientList {
  total: number;
  of: number;
  offset: number;
  rows: PatientSummary[];
  search_ms: number;
  /** Patients the mode's extraction store holds. */
  with_records?: number;
}

export interface Note {
  date: string | null;
  type: string | null;
  text: string;
}

export interface MatchScore {
  score: number;
  pass: boolean;
  reasonableness: number;
  cancer_type: number;
  burden: number;
  treatment_history: number;
  biomarker: number;
}

export type BoilerplateScore = { p: number; pass: boolean } | { skipped: string };

export interface PairScores {
  match: MatchScore | null;
  boilerplate: BoilerplateScore | null;
}

export interface PatientHit {
  rank: number;
  patient: PatientSummary;
  cosine: number;
  scores: PairScores;
}

export interface PatientsForCohortResult {
  direction: "patients_for_cohort";
  query: CohortDetail;
  k: number;
  searched: number;
  of: number;
  round: number;
  search_ms: number;
  results: PatientHit[];
}

export interface CohortHit {
  rank: number;
  cohort: CohortRow;
  cosine: number;
  scores: PairScores;
}

export interface TrialsForPatientResult {
  direction: "trials_for_patient";
  patient: PatientSummary & { key: string; boilerplate: string };
  k: number;
  searched: number;
  of: number;
  round: number;
  search_ms: number;
  results: CohortHit[];
}

export type JobStatus = "queued" | "running" | "cancelling" | "done" | "error" | "cancelled";

/** A background job (the server's core/jobs.py): polled with /api/jobs/{id}, stopped with .../stop. */
export interface Job<T = unknown> {
  id: string;
  kind: string;
  label: string;
  status: JobStatus;
  done: number;
  total: number;
  message: string;
  error: string | null;
  device: string | null;
  created: number;
  updated: number;
  started: number | null;
  finished: number | null;
  seconds: number | null;
  /** What a reloaded page needs to show the job's inputs again (guideline jobs). */
  restore?: unknown;
  result?: T;
}

/** A started job, as the routes that start one return it. */
export interface JobStart {
  job: Job;
}

export const isActive = (job: Pick<Job, "status"> | null | undefined) =>
  !!job && (job.status === "queued" || job.status === "running" || job.status === "cancelling");

/** Stop a job of this browser session (a queued job never starts; a running one stops at its next step). */
export const stopJob = (jobId: string) => api.post<Job>(`/api/jobs/${jobId}/stop`, {});

export interface ScoreResult {
  scores: ({ patient_id: string; cohort_id: string } & PairScores)[];
}

export interface CriterionLine {
  number: number;
  type: "inclusion" | "exclusion" | null;
  boilerplate: boolean;
  text: string;
  p: number;
  fails: boolean;
}

export interface CriteriaResult {
  lines: CriterionLine[];
  unchecked_lines: number;
  cached?: boolean;
}

/** Poll a job until it ends, reporting progress. */
export async function waitForJob<T>(
  jobId: string,
  onProgress?: (job: Job<T>) => void,
  intervalMs = 800,
): Promise<T> {
  for (;;) {
    const job = await api.get<Job<T>>(`/api/jobs/${jobId}`);
    onProgress?.(job);
    if (job.status === "done") return job.result as T;
    if (job.status === "error") throw new ApiError(500, job.error ?? "job failed");
    if (job.status === "cancelled") throw new ApiError(499, job.message || "Stopped.");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
