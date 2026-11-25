/** The local model server of Data and models (the server's /api/local-server*, in every dashboard). */
import { queryOptions } from "@tanstack/react-query";

import { api } from "./api";

export type Engine = "vllm" | "llama_cpp";
export type ServerState =
  "downloading" | "starting" | "ready" | "stopping" | "stopped" | "exited" | "failed";
export const ACTIVE_STATES: ServerState[] = ["downloading", "starting", "ready", "stopping"];

/** llm.local_server.ServerSpec: null leaves a setting at the engine's own default. */
export interface ServerSpec {
  engine: Engine;
  model: string;
  revision: string | null;
  served_name: string;
  tokenizer: string;
  gpus: number[];
  extra_args: string[];
  env: Record<string, string>;
  speculative_config: Record<string, unknown> | null;
  [field: string]: unknown;
}

export interface GpuRow {
  index: number;
  name: string;
  uuid: string;
  compute_capability: string | null;
  memory_total: number;
  memory_used: number | null;
  memory_free: number | null;
  utilization: number | null;
  processes: number;
  source: string;
  dashboard: number;
  local_server: number;
  other: number | null;
  decision_model: boolean;
  server_gpu: boolean;
}

export interface GpuList {
  gpus: GpuRow[];
  checked_at: number;
  note: string | null;
}

export interface EngineInfo {
  label: string;
  available: boolean;
  version: string | null;
  reason: string | null;
}

export interface ServerInfo {
  spec: ServerSpec;
  engine: Engine;
  state: ServerState;
  phase: string;
  error: string | null;
  exit_code: number | null;
  progress: { done: number; total: number } | null;
  endpoint: string;
  model: string;
  tokenizer: string;
  command: string | null;
  started_at: number;
  ready_at: number | null;
  elapsed_seconds: number;
  log: string[];
}

export interface LocalServerStatus {
  /** --local-server-only: the local server is the only model connection the app accepts. */
  only: boolean;
  /** With --local-server-only, OpenRouter with a user's own API key is also accepted. */
  allow_openrouter: boolean;
  /** False on a machine that cannot host an LLM (--no-local-server): no server can be started here. */
  enabled: boolean;
  /** Why no local server is offered (when not `enabled`). */
  disabled_reason: string | null;
  server: ServerInfo | null;
  default_on_ready: boolean;
  allow_downloads: boolean;
  hub_cache: string;
  default_model: string;
  known_models: { model: string; label: string }[];
}

export interface Recipe {
  known: boolean;
  label: string;
  reasoning: boolean;
  proposal: { engine: Engine; reason: string };
  engine: Engine;
  spec: ServerSpec;
  variant: string | null;
  variants: { name: string; memory_gb: number; note: string | null }[];
  note: string | null;
  source: string | null;
  gguf_choices: { name: string; size: number; parts: number }[];
  warnings: string[];
  error: string | null;
  command: string;
}

export interface Preview {
  valid: boolean;
  error: string | null;
  warnings: string[];
  command: string | null;
  download_bytes?: number | null;
}

export const isActive = (s: ServerInfo | null | undefined) =>
  Boolean(s && ACTIVE_STATES.includes(s.state));

export const localServerOptions = () =>
  queryOptions({
    queryKey: ["local-server"],
    queryFn: () => api.get<LocalServerStatus>("/api/local-server"),
    // Every 2 s while a server starts or runs (its phase and log), else every 10 s.
    refetchInterval: (query) => (isActive(query.state.data?.server) ? 2_000 : 10_000),
  });

export const gpusOptions = () =>
  queryOptions({
    queryKey: ["local-server", "gpus"],
    queryFn: () => api.get<GpuList>("/api/local-server/gpus"),
    refetchInterval: 5_000,
  });

export const enginesOptions = () =>
  queryOptions({
    queryKey: ["local-server", "engines"],
    queryFn: () => api.get<Record<Engine, EngineInfo>>("/api/local-server/engines"),
    staleTime: 60_000,
  });

export function recipe(model: string, engine: Engine | null, gpus: number[] | null) {
  const query = new URLSearchParams({ model });
  if (engine) query.set("engine", engine);
  if (gpus) query.set("gpus", gpus.join(","));
  return api.get<Recipe>(`/api/local-server/recipe?${query.toString()}`);
}

/** "12.3 GB" (GiB, as the GPUs report memory). */
export function gb(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "?";
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/** The engine parameters the form shows, per engine (the rest are in Advanced). */
export interface Field {
  name: string;
  label: string;
  kind: "int" | "float" | "text" | "bool" | "choice";
  choices?: string[];
  help?: string;
  /** The value an empty input means (default null: the engine's own default). */
  empty?: string;
}

const CACHE_TYPES = ["f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];

export const COMMON_FIELDS: Field[] = [
  {
    name: "served_name",
    label: "Served name",
    kind: "text",
    help: "the API's model name",
    empty: "",
  },
  {
    name: "tokenizer",
    label: "Tokenizer",
    kind: "text",
    help: "for the app's prompt budgets",
    empty: "",
  },
  { name: "revision", label: "Revision", kind: "text" },
];

export const ENGINE_FIELDS: Record<Engine, Field[]> = {
  vllm: [
    { name: "tensor_parallel_size", label: "Tensor parallel", kind: "int", help: "default: GPUs" },
    { name: "pipeline_parallel_size", label: "Pipeline parallel", kind: "int" },
    { name: "max_model_len", label: "Max model length", kind: "int" },
    { name: "gpu_memory_utilization", label: "GPU memory utilization", kind: "float" },
    { name: "max_num_seqs", label: "Max sequences", kind: "int" },
    { name: "dtype", label: "dtype", kind: "text" },
    { name: "quantization", label: "Quantization", kind: "text" },
    { name: "kv_cache_dtype", label: "KV cache dtype", kind: "text" },
    {
      name: "reasoning_parser",
      label: "Reasoning parser",
      kind: "text",
      help: "auto, none or a name",
      empty: "auto",
    },
    {
      name: "reasoning_parser_plugin",
      label: "Parser plugin",
      kind: "text",
      help: "a file in the repo",
    },
    { name: "tool_call_parser", label: "Tool-call parser", kind: "text", empty: "auto" },
    {
      name: "enable_prefix_caching",
      label: "Prefix caching",
      kind: "choice",
      choices: ["on", "off"],
    },
    { name: "trust_remote_code", label: "Trust remote code", kind: "bool" },
    { name: "enforce_eager", label: "Enforce eager", kind: "bool" },
  ],
  llama_cpp: [
    { name: "ctx_size", label: "Context size", kind: "int", help: "empty: fitted to free memory" },
    { name: "parallel", label: "Parallel slots", kind: "int" },
    { name: "n_gpu_layers", label: "GPU layers", kind: "int" },
    {
      name: "flash_attn",
      label: "Flash attention",
      kind: "choice",
      choices: ["auto", "on", "off"],
      empty: "auto",
    },
    { name: "cache_type_k", label: "K cache type", kind: "choice", choices: CACHE_TYPES },
    { name: "cache_type_v", label: "V cache type", kind: "choice", choices: CACHE_TYPES },
    { name: "batch_size", label: "Batch size", kind: "int" },
    { name: "ubatch_size", label: "Micro-batch size", kind: "int" },
    {
      name: "split_mode",
      label: "Split mode",
      kind: "choice",
      choices: ["none", "layer", "row", "tensor"],
    },
    { name: "reasoning_format", label: "Reasoning format", kind: "text", empty: "deepseek" },
    { name: "draft_model", label: "Draft model (repo)", kind: "text" },
    { name: "draft_gguf", label: "Draft GGUF", kind: "text" },
    { name: "draft_type", label: "Speculative type", kind: "text" },
    { name: "draft_tokens", label: "Draft tokens", kind: "int" },
    { name: "kv_unified", label: "Unified KV cache", kind: "bool" },
  ],
};
