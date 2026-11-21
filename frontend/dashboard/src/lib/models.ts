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
  claude_profiles: Record<
    string,
    { endpoint: string; model: string; label: string; databricks_profile: string; task: Task }
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

export const SQL_TASK: Task = "workup_sql";
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
  if (c.llm_backend === "claude_databricks")
    return `Claude (${c.model.split(".").at(-1) ?? c.model})`;
  return `${c.model} at ${c.endpoint}`;
}
