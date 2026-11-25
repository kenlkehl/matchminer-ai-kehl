/** The model connection of each LLM task, per browser session (the server's /api/models). */
import { queryOptions } from "@tanstack/react-query";

import { api } from "./api";

export type Backend =
  | "local"
  | "gemini4_vertex"
  | "gemini_vertex"
  | "gemma_vertex"
  | "claude_databricks"
  | "openrouter";
export type Task =
  | "summarize"
  | "workup"
  | "workup_sql"
  | "workup_full_backup"
  | "workup_agent_backup"
  | "good_option"
  | "llm_decisions"
  | "assistant"
  | "criteria_labels";

export interface TaskConnection {
  llm_backend: Backend;
  endpoint: string;
  model: string;
  tokenizer: string;
  reasoning: boolean;
  thinking_supported: boolean;
  sampling: Record<string, unknown>;
  backup_enabled: boolean;
  failure_threshold: number;
}

export interface ModelConfig extends TaskConnection {
  tasks: Record<Task, TaskConnection & { label: string }>;
  vertex_profiles: Record<
    string,
    { endpoint: string; model: string; tokenizer: string; label: string; project: string }
  >;
  /** Claude through the Databricks gateway: every task may use it; `models` are the site's endpoint names. */
  claude_profiles: Record<
    string,
    {
      endpoint: string;
      model: string;
      models: Record<string, string>;
      label: string;
      databricks_profile: string;
    }
  >;
  local_defaults: { endpoint: string; model: string; tokenizer: string };
  /** The local model server (Data and models): its state and, once ready, its connection. `only`: the app runs
   * with --local-server-only, so no other connection is accepted. */
  local_server: {
    only: boolean;
    state: string | null;
    endpoint: string | null;
    model: string | null;
    tokenizer: string | null;
    allow_openrouter?: boolean;
    /** False when no local server is offered here (--no-local-server). */
    enabled?: boolean;
  };
  /** OpenRouter with the session's own key (synthetic mode only). The key is never sent back: only whether one
   * is set and its last four characters. */
  openrouter: OpenRouterView;
  device: string;
  ttl_minutes: number;
  max_upload_mb: number;
  synthetic_note_sources: { id: string; label: string }[];
  notice: string;
}

export type OpenRouterView =
  | { offered: false }
  | {
      offered: true;
      endpoint: string;
      notice: string;
      key_set: boolean;
      key_hint: string | null;
      default_model: string;
      models: { id: string; label: string; reasoning: boolean; claude: boolean }[];
    };

export interface ModelCheck {
  available: boolean;
  endpoint: string;
  models: string[];
  model: string | null;
  error: string | null;
}

export const BACKUP_TASKS: Task[] = ["workup_full_backup", "workup_agent_backup"];

export const modelsOptions = () =>
  queryOptions({
    queryKey: ["models"],
    queryFn: () => api.get<ModelConfig>("/api/models"),
    staleTime: 30_000,
  });

/** What a task's connection is, in a few words. */
export function connectionLabel(config: ModelConfig, c: TaskConnection): string {
  if (!c.endpoint) return "No connection (start the local model server above)";
  const vertex = config.vertex_profiles[c.llm_backend];
  if (vertex) return `${vertex.label} (Google Vertex)`;
  if (c.llm_backend === "claude_databricks") return `${claudeName(c.model)} (Databricks gateway)`;
  if (c.llm_backend === "openrouter") return `${c.model} (OpenRouter)`;
  return `${c.model} at ${c.endpoint}`;
}

/** "Claude Haiku 5.5" from a gateway endpoint name ending in claude-haiku-5-5 (else the name). */
export function claudeName(endpoint: string): string {
  const match = /claude-([a-z]+)-(\d+)-(\d+)$/.exec(endpoint);
  if (!match?.[1]) return endpoint;
  const model = match[1].charAt(0).toUpperCase() + match[1].slice(1);
  return `Claude ${model} ${match[2] ?? ""}.${match[3] ?? ""}`;
}
