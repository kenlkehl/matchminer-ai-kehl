import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";
import {
  BACKUP_TASKS,
  type Backend,
  claudeName,
  connectionLabel,
  type ModelCheck,
  type ModelConfig,
  modelsOptions,
  type Task,
} from "@/lib/models";
import { llmOptions } from "@/lib/queries";

const SELECT = "h-9 rounded-md border border-input bg-background px-2 text-sm";
/** One connector for the Vertex profiles (each is a model); the form's own value, not a backend. */
const VERTEX = "vertex";

/**
 * Each LLM task's connection for this browser session: an OpenAI-compatible server (its models read
 * from /v1/models), a Vertex profile, or Claude through the Databricks gateway (the default). A job
 * keeps the connection it started with. The server's command line sets the default.
 */
export function ModelsPanel() {
  const config = useQuery(modelsOptions());
  const llm = useQuery(llmOptions()).data;
  const [task, setTask] = useState<Task>("summarize");
  if (!config.data) return config.isError ? <p>{config.error.message}</p> : <Spinner />;
  const c = config.data;
  return (
    <div className="flex flex-col gap-4 text-sm">
      <p className="rounded-md bg-amber-50 px-3 py-2 text-amber-900">{c.notice}</p>
      <table className="w-full text-left">
        <thead className="text-xs text-gray-500">
          <tr>
            <th className="py-1 pr-3 font-medium">Task</th>
            <th className="py-1 pr-3 font-medium">Connection</th>
            <th className="py-1 font-medium" />
          </tr>
        </thead>
        <tbody>
          {(Object.keys(c.tasks) as Task[]).map((name) => (
            <tr key={name} className={name === task ? "bg-brand-light-blue/30" : undefined}>
              <td className="py-1 pr-3">{c.tasks[name].label}</td>
              <td className="py-1 pr-3 text-gray-700">
                {connectionLabel(c, c.tasks[name])}
                {BACKUP_TASKS.includes(name) && !c.tasks[name].backup_enabled ? " (off)" : ""}
              </td>
              <td className="py-1 text-right">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setTask(name);
                  }}
                >
                  Change
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <TaskForm key={task} config={c} task={task} />
      {llm ? (
        <dl className="flex flex-col gap-0.5 text-xs text-gray-600">
          <div>
            <dt className="inline font-medium">Good option check: </dt>
            <dd className="inline">
              {llm.good_option.available ? "available" : llm.good_option.reason}
            </dd>
          </div>
          <div>
            <dt className="inline font-medium">LLM decision checks: </dt>
            <dd className="inline">
              {llm.decisions.available
                ? `available (${llm.decisions.connection.label}; reasoning: ${llm.decisions.reasoning})`
                : llm.decisions.reason}
            </dd>
          </div>
          <div>
            <dt className="inline font-medium">Inference package: </dt>
            <dd className="inline">
              {llm.package.version ?? "?"} ({llm.package.path})
            </dd>
          </div>
        </dl>
      ) : null}
    </div>
  );
}

function TaskForm({ config, task }: { config: ModelConfig; task: Task }) {
  const client = useQueryClient();
  const current = config.tasks[task];
  const [backend, setBackend] = useState<Backend>(current.llm_backend);
  const local = current.llm_backend === "local" ? current : config.local_defaults;
  const [endpoint, setEndpoint] = useState(local.endpoint);
  const [model, setModel] = useState(
    current.llm_backend === "local" || current.llm_backend === "claude_databricks"
      ? current.model
      : config.local_defaults.model,
  );
  const [tokenizer, setTokenizer] = useState(
    current.llm_backend === "local" ? current.tokenizer : "",
  );
  const [reasoning, setReasoning] = useState(current.reasoning);
  const [backup, setBackup] = useState(current.backup_enabled);
  const [threshold, setThreshold] = useState(current.failure_threshold);
  const [checked, setChecked] = useState<ModelCheck | null>(null);
  useEffect(() => {
    setChecked(null);
  }, [backend, endpoint]);
  const vertexKeys = Object.keys(config.vertex_profiles);
  const modelLabel = useId();
  const vertex = config.vertex_profiles[backend];
  const claude = backend === "claude_databricks" ? config.claude_profiles[backend] : undefined;
  useEffect(() => {
    // Claude offers the site's endpoint name first.
    if (claude && !model.includes("claude")) setModel(claude.model);
  }, [claude, model]);
  const isBackup = BACKUP_TASKS.includes(task);
  const body = (applyToAll: boolean) => ({
    llm_backend: backend,
    endpoint: vertex?.endpoint ?? claude?.endpoint ?? endpoint,
    model: vertex?.model ?? model,
    tokenizer: vertex ? vertex.tokenizer : tokenizer,
    task,
    apply_to_all: applyToAll,
    reasoning,
    backup_enabled: isBackup ? backup : false,
    failure_threshold: threshold,
  });
  const check = useMutation({
    mutationFn: () =>
      api.post<ModelCheck>("/api/models/check", {
        task,
        llm_backend: backend,
        endpoint: vertex || claude ? null : endpoint,
        model: vertex ? null : model,
      }),
    onSuccess: (found) => {
      setChecked(found);
      // An OpenAI-compatible server: take the model it serves (the first, if several and this one is not).
      if (
        backend === "local" &&
        found.model &&
        (found.models.length === 1 || !found.models.includes(model))
      )
        setModel(found.model);
    },
  });
  const save = useMutation({
    mutationFn: (applyToAll: boolean) => api.post<ModelConfig>("/api/models", body(applyToAll)),
    onSuccess: (next) => {
      client.setQueryData(["models"], next);
      void client.invalidateQueries({ queryKey: ["llm"] });
    },
  });
  return (
    <section className="flex flex-col gap-3 rounded-md border border-gray-200 p-3">
      <h3 className="font-semibold">{current.label}</h3>
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">Connection</span>
          <select
            className={SELECT}
            value={vertex ? VERTEX : backend}
            onChange={(e) => {
              const value = e.target.value;
              if (value === VERTEX) {
                const first = vertexKeys[0];
                if (!vertex && first) setBackend(first as Backend);
              } else setBackend(value as Backend);
            }}
          >
            <option value="local">OpenAI-compatible server (vLLM)</option>
            {vertexKeys.length ? (
              <option value={VERTEX}>Google Vertex / Agent Platform</option>
            ) : null}
            {Object.entries(config.claude_profiles).map(([key, p]) => (
              <option key={key} value={key}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        {vertex ? (
          <>
            <label className="flex min-w-72 flex-col gap-1">
              <span className="text-xs text-gray-500">Model</span>
              <select
                className={SELECT}
                value={backend}
                onChange={(e) => {
                  setBackend(e.target.value as Backend);
                }}
              >
                {Object.entries(config.vertex_profiles).map(([key, p]) => (
                  <option key={key} value={key}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="pb-2 text-xs text-gray-500">
              {vertex.model} in project {vertex.project}, with this server's Google Application
              Default Credentials
            </p>
          </>
        ) : claude ? (
          <label className="flex min-w-96 flex-1 flex-col gap-1">
            <span className="text-xs text-gray-500">
              Model (gateway endpoint; profile {claude.databricks_profile})
            </span>
            <select
              className={SELECT}
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
              }}
            >
              {Object.values(claude.models).includes(model) ? null : (
                <option value={model}>{claudeName(model)}</option>
              )}
              {Object.values(claude.models).map((name) => (
                <option key={name} value={name}>
                  {claudeName(name)}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <>
            <label className="flex min-w-72 flex-1 flex-col gap-1">
              <span className="text-xs text-gray-500">Endpoint (/v1 base URL)</span>
              <Input
                value={endpoint}
                onChange={(e) => {
                  setEndpoint(e.target.value);
                }}
              />
            </label>
            <div className="flex min-w-72 flex-1 flex-col gap-1">
              <span className="flex items-center justify-between text-xs text-gray-500">
                <span id={modelLabel}>Model</span>
                <button
                  type="button"
                  className="text-brand-blue hover:underline disabled:opacity-50"
                  disabled={check.isPending || !endpoint.trim()}
                  onClick={() => {
                    check.mutate();
                  }}
                >
                  {check.isPending ? "Detecting…" : "Detect served model"}
                </button>
              </span>
              {checked?.models.length ? (
                <select
                  aria-labelledby={modelLabel}
                  className={SELECT}
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                  }}
                >
                  {checked.models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <Input
                  aria-labelledby={modelLabel}
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                  }}
                />
              )}
            </div>
            <label className="flex min-w-48 flex-col gap-1">
              <span className="text-xs text-gray-500">Tokenizer (default: the model)</span>
              <Input
                value={tokenizer}
                onChange={(e) => {
                  setTokenizer(e.target.value);
                }}
              />
            </label>
          </>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-4">
        {!claude ? (
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={reasoning}
              onChange={(e) => {
                setReasoning(e.target.checked);
              }}
            />
            Reasoning (thinking), where the model has a switch
          </label>
        ) : null}
        {isBackup ? (
          <>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={backup}
                onChange={(e) => {
                  setBackup(e.target.checked);
                }}
              />
              Use as a backup
            </label>
            <label className="flex items-center gap-2">
              after
              <Input
                type="number"
                min={1}
                max={10}
                className="w-16"
                value={threshold}
                onChange={(e) => {
                  setThreshold(Number(e.target.value) || 1);
                }}
              />
              failures
            </label>
          </>
        ) : null}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          disabled={check.isPending}
          onClick={() => {
            check.mutate();
          }}
        >
          {check.isPending ? <Spinner /> : null}
          Check connection
        </Button>
        <Button
          disabled={save.isPending}
          onClick={() => {
            save.mutate(false);
          }}
        >
          Save for this task
        </Button>
        <Button
          variant="outline"
          disabled={save.isPending}
          onClick={() => {
            save.mutate(true);
          }}
        >
          Use for every task
        </Button>
      </div>
      {checked ? (
        <p className={checked.available ? "text-green-700" : "text-red-700"}>
          {checked.available
            ? `Reachable: ${checked.model ?? ""} at ${checked.endpoint}`
            : (checked.error ?? "Not reachable.")}
        </p>
      ) : null}
      {check.isError ? <p className="text-red-700">{check.error.message}</p> : null}
      {save.isError ? <p className="text-red-700">{save.error.message}</p> : null}
      {save.isSuccess ? <p className="text-green-700">Saved for this browser session.</p> : null}
    </section>
  );
}
