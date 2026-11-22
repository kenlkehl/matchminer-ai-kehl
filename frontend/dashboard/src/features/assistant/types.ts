/** The Assistant tab's server view (matchminer_ai/apps/dashboard/routes/assistant.py). */
import type { Job } from "@/lib/api";

export interface Step {
  tool: string;
  label: string;
  /** What the model said the step is for. */
  purpose: string;
  /** What was done and what came back ("Searched trials: EGFR … → 23 cohorts in 14 trials"). */
  summary: string;
  ok: boolean | null;
  seconds: number | null;
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
  error: string | null;
  at: number;
  seconds: number | null;
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
