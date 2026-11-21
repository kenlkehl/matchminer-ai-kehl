/** The model connection of each LLM task, per browser session (the server's /api/models). */
import { queryOptions } from "@tanstack/react-query";

import { api } from "./api";

export type Backend =
  "local" | "gemini4_vertex" | "gemini_vertex" | "gemma_vertex" | "claude_databricks";
export type Task =
  | "summarize"
  | "compress"
  | "workup"
  | "workup_sql"
  | "workup_full_backup"
  | "workup_agent_backup"
  | "good_option"
  | "criteria_llm";

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
  device: string;
  ttl_minutes: number;
  max_upload_mb: number;
  synthetic_note_sources: { id: string; label: string }[];
  notice: string;
}

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
  const vertex = config.vertex_profiles[c.llm_backend];
  if (vertex) return vertex.label;
  if (c.llm_backend === "claude_databricks") return `${claudeName(c.model)} (Databricks gateway)`;
  return `${c.model} at ${c.endpoint}`;
}

/** "Claude Haiku 5.5" from a gateway endpoint name ending in claude-haiku-5-5 (else the name). */
export function claudeName(endpoint: string): string {
  const match = /claude-([a-z]+)-(\d+)-(\d+)$/.exec(endpoint);
  if (!match?.[1]) return endpoint;
  const model = match[1].charAt(0).toUpperCase() + match[1].slice(1);
  return `Claude ${model} ${match[2] ?? ""}.${match[3] ?? ""}`;
}
