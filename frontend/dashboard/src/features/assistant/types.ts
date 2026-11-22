/** The Assistant tab's server view (matchminer_ai/apps/dashboard/routes/assistant.py). */
import type { Job } from "@/lib/api";

export interface Step {
  /** The step's number, through the whole conversation (what a chart cites). */
  number?: number;
  tool: string;
  label: string;
  /** What the model said the step is for. */
  purpose: string;
  /** What was done and what came back ("Searched trials: EGFR … → 23 cohorts in 14 trials"). */
  summary: string;
  ok: boolean | null;
  seconds: number | null;
  /** The LLM answers a scoring step got when the session's scores came from the LLM. */
  llm_answers?: {
    patient_id: string;
    cohort_id: string;
    family: "match" | "boilerplate";
    key: string;
  }[];
}

export interface UserEntry {
  id: number;
  role: "user";
  text: string;
  at: number;
}

export interface AssistantEntry {
  id: number;
  role: "assistant";
  status: "running" | "done" | "stopped" | "error";
  model: string;
  steps: Step[];
  progress: string;
  /** The turn's outcome: an answer or a clarifying question. */
  kind: "answer" | "question" | null;
  text: string;
  options: string[];
  why: string;
  interpretation: string;
  /** The answer's charts (absent in entries made before charts existed). */
  charts?: Chart[];
  error: string | null;
  at: number;
  seconds: number | null;
}

/** A chart of an answer, as the server checked it (assistant/actions.py, chart_view). */
export interface Chart {
  kind: "pie" | "donut" | "bar" | "line";
  title: string;
  caption: string;
  /** The categories (slices, bars), or the x-axis values in order (line). */
  labels: string[];
  /** One value per label each; exactly one series for pie and donut. */
  series: { name: string; values: number[] }[];
  /** What the values count ("patients", "trials", "%"), or "". */
  unit: string;
  /** The denominator percentages are out of (a pie's sum); 0 for none. */
  total: number;
  /** Bars only: "vertical" (columns) or "horizontal"; "" for other kinds. */
  orientation: "" | "vertical" | "horizontal";
  /** Bars with 2+ series: stacked rather than grouped. */
  stacked: boolean;
  /** The step numbers of the tool results the numbers came from ([]: the data overview). */
  steps: number[];
}

export type Entry = UserEntry | AssistantEntry;

/** An ID a tool returned, and where the page opens it. */
export interface LinkTarget {
  kind: "trial" | "cohort" | "patient";
  center_id?: string | null;
  trial_id?: string;
  /** Whether the trial is open (a closed one opens with closed trials shown). */
  open?: boolean;
}

export interface Usage {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
}

export interface Conversation {
  id: string;
  created: number;
  updated: number;
  entries: Entry[];
  links: Record<string, LinkTarget>;
  usage: Usage;
  job_id: string | null;
  user_messages: number;
}

export interface AssistantView {
  conversation: Conversation | null;
  running: Job | null;
  model: { task: "assistant"; label: string; llm_backend: string; model: string };
  mode: "real" | "synthetic";
  tools: string[];
  unavailable: Record<string, string>;
  limits: { max_tool_calls: number; max_messages: number; max_message_chars: number };
}
