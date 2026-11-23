/**
 * The Benchmarking tab's API (apps/dashboard/benchmarking.py, routes/benchmarking.py). Fields keep the server's
 * snake_case; metrics follow evaluation/retro_fidelity/metrics.py and evaluation/dual_judge/metrics.py.
 */
import type { Job } from "@/lib/api";

export type RunKind = "fidelity" | "judge";

export interface RunRow {
  id: string;
  kind: RunKind;
  name: string;
  folder: string;
  updated: number | null;
  has_metrics?: boolean;
  /** Fidelity: job counts by state. */
  jobs?: Record<string, number>;
  /** Judge: items per component. */
  components?: Record<string, number | null>;
  /** The two judges compared (the judges with answers, before a comparison). */
  judges?: string[];
  /** Judges whose earlier answers the run keeps, outside the comparison. */
  other_judges?: string[];
  compared?: boolean;
  adjudications?: number;
}

export interface RunList {
  runs: RunRow[];
  roots: number;
}

// --------------------------------------------------------------------------- distillation fidelity

/** run.py's job states. */
export type JobState = "pending" | "running" | "done" | "failed" | "blocked";

export interface FidelityJob {
  state: string;
  tries: number;
  exit: number | null;
  started: number | null;
  finished: number | null;
}

export interface StageReceipt {
  content: Record<string, unknown> | null;
  finished_at: string | null;
  commit: string | null;
}

/** A bootstrap mean with its percentile interval (retrieval). */
export interface MeanInterval {
  mean: number | null;
  low: number | null;
  high: number | null;
  n: number;
}

export interface RetrievalRow {
  queries: number;
  queries_with_fewer_than_top: number;
  queries_without_labels: number;
  unlabeled_pairs: number;
  precision_at_top: MeanInterval;
  mean_score: MeanInterval;
}

/** A point metric with its cluster-bootstrap interval (decision model). */
export interface ValueInterval {
  value: number | null;
  low: number | null;
  high: number | null;
}

export interface ReliabilityBin {
  n: number;
  p_min: number;
  p_max: number;
  mean_predicted: number;
  observed: number;
}

export interface FamilyMetrics {
  n: number;
  clusters?: number;
  prevalence_yes?: number | null;
  prevalence_no?: number | null;
  mean_predicted?: number | null;
  observed_rate?: number | null;
  bootstrap_draws?: number;
  degenerate_draws?: number;
  ece_bins?: string;
  reliability?: ReliabilityBin[];
  auroc?: ValueInterval;
  auprc_yes?: ValueInterval;
  auprc_no?: ValueInterval;
  brier?: ValueInterval;
  ece?: ValueInterval;
  calibration_intercept?: ValueInterval;
  calibration_slope?: ValueInterval;
}

/** A ranker's P@20 over every query with cluster-bootstrap intervals (full-denominator runs). */
export interface PipelineRow extends RetrievalRow {
  clusters: number;
}

export interface FidelityMetrics {
  retrieval?: {
    top: number;
    table: Record<string, RetrievalRow>;
    paired_differences: Record<string, MeanInterval>;
  };
  /** Full-denominator runs: ranker (c) over every query; clusters are persons or trials per workflow. */
  full_pipeline?: {
    top: number;
    ranker: string;
    clusters: Record<string, string>;
    table: Record<string, PipelineRow>;
  };
  /** "pooled", "patient_centric", "trial_centric": question family -> metrics. */
  decision?: Record<string, Record<string, FamilyMetrics>>;
  counts?: Record<string, number>;
}

export interface FidelityRun {
  id: string;
  name: string;
  jobs: Record<string, FidelityJob>;
  updated: string | null;
  released: boolean | null;
  design_notes: unknown[];
  stages: Record<string, StageReceipt>;
  k_binding: Record<string, Record<string, number>> | null;
  metrics: FidelityMetrics | null;
  /** The last exported PDF in the run's reports/ (benchmarking.py export). */
  report: { built: number; bytes: number } | null;
  /** The latest export jobs of this run. */
  exports: Job<ExportResult>[];
}

/** An export job's result: the PDF's size, whether it has Table 1, and why not. */
export interface ExportResult {
  bytes: number;
  demographics: boolean;
  note: string | null;
}

// --------------------------------------------------------------------------- LLM as judge

export type UnitStatus = "agreed" | "disputed" | "single" | "awaiting" | "missing";

export interface JudgeComponentBuild {
  items: number | null;
  requests: number | null;
  request_chars: number | null;
  max_request_chars: number | null;
  built_at: string | null;
  seconds: number | null;
}

export interface CompareComponent {
  items: number;
  judge_status: Record<string, Record<string, number>>;
  units: Partial<Record<UnitStatus, number>>;
  queued: Partial<Record<"disputed" | "single" | "audit", number>>;
  queued_items: number;
}

export interface CompareSummary {
  components: Record<string, CompareComponent>;
  judges?: string[];
  audit_fraction?: number;
  seed?: number;
  units?: number;
  queued?: number;
}

export interface JudgeCost {
  ran: boolean;
  superseded_cost_usd: number;
  final?: Record<string, number>;
  attempts?: number;
  cost_usd?: number;
  cost_per_item_usd?: number;
  median_seconds?: number | null;
  retried?: number;
  input_tokens?: number;
  output_tokens?: number;
}

export interface JudgeAccuracy {
  accuracy: number | null;
  n: number;
  accuracy_adjudicated: number | null;
  n_adjudicated: number;
}

export interface KindAgreement {
  units: number;
  agreed: number;
  disputed: number;
  single: number;
  awaiting: number;
  missing: number;
  queued: number;
  adjudicated: number;
  pending: number;
  agreement: number | null;
  kappa: number | null;
  judges: Record<string, JudgeAccuracy>;
  distribution: Record<string, Record<string, number>>;
  audit: { decided: number; both_wrong: number };
}

export type Interval95 = [number, number] | null;

/** A weighted rate (extraction and summary results, the teacher's accuracy). */
export interface Rate {
  n: number;
  items: number;
  rate: number | null;
  ci95: Interval95;
}

export interface Scored {
  value: number | null;
  ci95: Interval95;
}

export interface DecisionResult {
  n: number;
  reference_yes: number;
  teacher_accuracy: Rate;
  model_auroc: Scored;
  model_auprc_yes: Scored;
  model_auprc_no: Scored;
  model_brier: Scored;
}

export interface JudgeComponentMetrics {
  items: number;
  judges: Record<string, JudgeCost>;
  agreement: Record<string, KindAgreement>;
  /** Extraction and summary: measure -> Rate (or, "major omissions by section", section -> Rate);
   * decision: question family -> DecisionResult. */
  results: Record<string, unknown>;
}

export interface JudgeMetrics {
  units: number;
  with_reference: number;
  adjudications: number;
  /** Absent from metrics written by older code: take the judges from each component's `judges` keys. */
  judges?: string[];
  components: Record<string, JudgeComponentMetrics>;
}

export interface RecomputeResult {
  [step: string]: { returncode: number; log: string[] };
}

export interface JudgeRun {
  id: string;
  name: string;
  version: string | null;
  components: Record<string, JudgeComponentBuild>;
  /** judge -> component -> answer rows in its journal. */
  responses: Record<string, Record<string, number>>;
  compare: CompareSummary | null;
  metrics: JudgeMetrics | null;
  adjudications: number;
  recompute: Job<RecomputeResult>[];
}

// --------------------------------------------------------------------------- adjudication

export interface AdjudicationItemRow {
  item_id: string;
  title: string;
  stratum: string | null;
  units: number;
  queued: number;
  decided: number;
}

export interface AdjudicationState {
  components: Record<
    string,
    { items: AdjudicationItemRow[]; queued: number; decided: number } | undefined
  >;
  compared_at: number | null;
}

export interface ItemSource {
  kind: "listing" | "notes" | "task";
  text: string;
}

export interface CohortLine {
  number: number;
  text: string;
  criterion_type: string;
  boilerplate: boolean;
  heading?: string | null;
}

export interface ExtractionCohort {
  number: number;
  name: string;
  merged_from_names?: string[];
  items: CohortLine[];
}

export interface Statement {
  n: number;
  section: string;
  text: string;
}

export type ItemView =
  | {
      kind: "extraction";
      title: string;
      source: ItemSource;
      screen: string;
      cohort_names: string[];
      cohorts: ExtractionCohort[];
    }
  | { kind: "summary"; title: string; source: ItemSource; statements: Statement[] }
  | {
      kind: "decision";
      title: string;
      source: ItemSource;
      task: string;
      criterion_number: number | null;
      criterion_type: string | null;
    };

export interface JudgePanel {
  slot: number;
  value: string | null;
  compared: string | null;
  detail: Record<string, unknown> | null;
}

export interface Question {
  text: string;
  answers: Record<string, string>;
  status_field: string | null;
  statuses: Record<string, string>;
  note: string;
}

export interface Reference {
  teacher: {
    answer: string | null;
    status?: string | null;
    reason?: string | null;
    raw?: string | null;
  };
  model: { p_yes: number | null; answer: string | null; threshold: number };
}

export interface UnitDecision {
  value: string;
  note: string;
  at: string;
  stale: boolean;
}

export interface UnitView {
  unit_id: string;
  key: string;
  kind: string;
  label: string;
  choices: string[];
  target: string;
  status: UnitStatus;
  queued: boolean;
  queue_reason: string | null;
  judges: JudgePanel[];
  decision: UnitDecision | null;
  question: Question | null;
  reference?: Reference;
  reveal?: { judges: Record<string, string> };
}

export interface AdjudicationItem {
  component: string;
  item_id: string;
  view: ItemView;
  units: UnitView[];
}

export type Choice = "judge1" | "judge2" | "other" | "skip";

export interface DecideBody {
  unit_id: string;
  choice: Choice;
  value: string | null;
  note: string;
  show_reference: boolean;
}
