/**
 * The LLM decision checks (the server's llm_decisions.py): the LLM alternative to the decision model wherever the
 * dashboard scores a pair, and the browser session's choice between them ("Decision model" or "LLM").
 *
 * The choice is one per browser session (sessionStorage), shown as the same toggle in every view that scores
 * pairs; changing it in one view changes it in all. The LLM's answers stay on the server, in the session's memory.
 */
import { useSyncExternalStore } from "react";

import { api, ApiError, type Job, type LLMConnection, waitForJob } from "./api";

export type Scorer = "decision_model" | "llm";

export const SCORER_LABELS: Record<Scorer, string> = {
  decision_model: "Decision model",
  llm: "LLM",
};

const KEY = "mmai.scorer";
const listeners = new Set<() => void>();

function read(): Scorer {
  try {
    return sessionStorage.getItem(KEY) === "llm" ? "llm" : "decision_model";
  } catch {
    return "decision_model";
  }
}

let current: Scorer = read();

export function setScorer(next: Scorer): void {
  current = next;
  try {
    sessionStorage.setItem(KEY, next);
  } catch {
    // storage unavailable: the choice lasts until the page reloads
  }
  for (const listener of listeners) listener();
}

/** The session's scorer now (outside React). */
export const currentScorer = (): Scorer => current;

/** The session's scorer; every view re-renders when it changes. */
export function useScorer(): Scorer {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => current,
  );
}

// ---------------------------------------------------------------- answers

export interface LabelAnswer {
  answer: "yes" | "no";
  evidence_status: string;
  reason: string;
}

interface AnswerBase {
  scorer: "llm";
  key: string;
  scale: string;
  connection: Pick<LLMConnection, "backend" | "model" | "label">;
  /** What reasoning text this backend returns (or that it returns none). */
  reasoning_note: string;
  contract: string;
  seconds: number;
  checked_at: number;
  parse_status: string;
  /** The model's raw answer text. */
  output: string;
  /** The backend's reasoning (thinking) text; empty when it returned none. */
  reasoning: string;
}

/** The five match questions, combined into the package's 0-6 match score (null: no valid answer). */
export interface LLMMatchAnswer extends AnswerBase {
  family: "match";
  score: number | null;
  max: number;
  reasonable: boolean | null;
  answers: Record<string, LabelAnswer> | null;
  labels: string[];
}

/** "Is this patient clearly excluded from this cohort?" (null: no valid answer). */
export interface LLMBoilerplateAnswer extends AnswerBase {
  family: "boilerplate";
  pass: boolean | null;
  excluded: boolean | null;
  answer: "Yes" | "No" | null;
}

export type LLMBoilerplate = LLMBoilerplateAnswer | { skipped: string };

export interface LLMPairResult {
  patient_id: string;
  cohort_id: string;
  match?: LLMMatchAnswer | null;
  boilerplate?: LLMBoilerplate | null;
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
  output: string;
  reasoning: string;
}

export interface LLMCriteriaResult extends Omit<
  AnswerBase,
  "parse_status" | "output" | "reasoning"
> {
  family: "criteria";
  kind: "criteria_llm";
  patient_id: string;
  cohort_id: string;
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
  cached?: boolean;
}

/** What a run would send: requests (cached answers are not asked again), the model, approximate input tokens. */
export interface LLMEstimate {
  pairs: number;
  requests: number;
  by_family: Record<string, number>;
  cached: number;
  approx_input_tokens: number;
  model: string;
  backend: string;
  large: boolean;
  large_run: number;
}

export interface LLMScale {
  name: string;
  range: (number | string)[];
  meaning: string;
  ranking?: string;
}

/** The session's LLM decision checks (part of /api/llm). */
export interface LLMDecisionsStatus {
  available: boolean;
  reason: string | null;
  connection: LLMConnection;
  notice: string;
  reasoning: string;
  scales: Record<"match" | "boilerplate" | "criteria", LLMScale>;
  large_run: number;
  max_pairs: number;
  max_candidates: number;
  in_flight: number;
  cached: number;
}

export type Direction = "patients_for_cohort" | "trials_for_patient";

export interface PairsBody {
  direction: Direction;
  query_id: string;
  target_ids: string[];
  families?: ("match" | "boilerplate")[];
  rerun?: boolean;
  confirm?: boolean;
}

/** The session's answers and an estimate for the rest (no model calls). */
export const lookupPairs = (body: PairsBody) =>
  api.post<{ results: LLMPairResult[]; estimate: LLMEstimate }>("/api/llm-decisions/lookup", body);

/** Ask the LLM about the pairs (a session job), reporting its progress; the pairs' answers. */
export async function scorePairs(
  body: PairsBody,
  onProgress?: (job: Job) => void,
): Promise<LLMPairResult[]> {
  const started = await api.post<{ results?: LLMPairResult[]; job?: Job }>(
    "/api/llm-decisions/score",
    body,
  );
  if (started.results) return started.results;
  if (!started.job) throw new ApiError(500, "the server started no job");
  const result = await waitForJob<{ results: LLMPairResult[] }>(started.job.id, onProgress, 1000);
  return result.results;
}

/** One of the session's answers in full. */
export const fetchAnswer = (key: string) =>
  api.get<LLMMatchAnswer | LLMBoilerplateAnswer>(
    `/api/llm-decisions/answers/${encodeURIComponent(key)}`,
  );

/** "6 requests to Claude Haiku 5.5 (Databricks gateway), about 9,000 input tokens". */
export function estimateText(e: Pick<LLMEstimate, "requests" | "model" | "approx_input_tokens">) {
  return (
    `${e.requests.toLocaleString()} ${e.requests === 1 ? "request" : "requests"} to ${e.model}` +
    (e.approx_input_tokens
      ? `, about ${roundTokens(e.approx_input_tokens).toLocaleString()} input tokens`
      : "")
  );
}

function roundTokens(n: number) {
  const step = n > 100_000 ? 10_000 : n > 10_000 ? 1_000 : 100;
  return Math.max(step, Math.round(n / step) * step);
}
