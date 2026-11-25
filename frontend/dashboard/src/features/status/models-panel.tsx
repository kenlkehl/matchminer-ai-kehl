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
  type OpenRouterView,
  type Task,
} from "@/lib/models";
import { llmOptions } from "@/lib/queries";

const SELECT = "h-9 rounded-md border border-input bg-background px-2 text-sm";
/** One connector for the Vertex profiles (each is a model); the form's own value, not a backend. */
const VERTEX = "vertex";

/**
 * Each LLM task's connection for this browser session: an OpenAI-compatible server (its models read
 * from /v1/models), a Vertex profile, Claude through the Databricks gateway (the default) or, in
 * synthetic mode, OpenRouter with the user's own key. A job keeps the connection it started with. The
 * server's command line sets the default.
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

/** One task's connection form (Data and models; the Assistant tab's model choice uses it too). */
export function TaskForm({ config, task }: { config: ModelConfig; task: Task }) {
  const client = useQueryClient();
  const current = config.tasks[task];
  // The local model server (Data and models), once it is ready; with --local-server-only the only connection
  // (and OpenRouter, when the app allows it).
  const localServer = config.local_server;
  const localReady = localServer.state === "ready" && Boolean(localServer.endpoint);
  const only = localServer.only;
  const openRouter = config.openrouter.offered ? config.openrouter : null;
  const [backend, setBackend] = useState<Backend>(
    only && current.llm_backend !== "openrouter" ? "local" : current.llm_backend,
  );
  const local = current.llm_backend === "local" ? current : config.local_defaults;
  const [endpoint, setEndpoint] = useState(local.endpoint);
  const [model, setModel] = useState(
    current.llm_backend === "local" ||
      current.llm_backend === "claude_databricks" ||
      current.llm_backend === "openrouter"
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
    // Claude offers the site's endpoint name first (the chat assistant's: Opus); an OpenRouter ID has a "/".
    if (claude && (!model.includes("claude") || model.includes("/")))
      setModel((task === "assistant" ? claude.models.opus : undefined) ?? claude.model);
  }, [claude, model, task]);
  const routed = backend === "openrouter" ? openRouter : null;
  // Claude through OpenRouter takes an effort, not a thinking switch.
  const routedClaude = Boolean(
    routed?.models.find((m) => m.id === model)?.claude ?? model.startsWith("anthropic/"),
  );
  const isBackup = BACKUP_TASKS.includes(task);
  const useLocal = only && !routed;
  const chooseBackend = (value: Backend) => {
    if (value === "openrouter" && openRouter && backend !== "openrouter") {
      // OpenRouter offers its default model first, with its reasoning setting.
      const first = openRouter.models.find((m) => m.id === openRouter.default_model);
      setModel(openRouter.default_model);
      setReasoning(first?.reasoning ?? true);
    }
    setBackend(value);
  };
  const body = (applyToAll: boolean) => ({
    llm_backend: useLocal ? "local" : backend,
    endpoint: useLocal
      ? localServer.endpoint
      : routed
        ? routed.endpoint
        : (vertex?.endpoint ?? claude?.endpoint ?? endpoint),
    model: useLocal ? localServer.model : (vertex?.model ?? model),
    tokenizer: useLocal
      ? localServer.tokenizer
      : vertex
        ? vertex.tokenizer
        : routed
          ? ""
          : tokenizer,
    task,
    apply_to_all: applyToAll,
    reasoning,
    backup_enabled: isBackup ? backup : false,
    failure_threshold: threshold,
  });
  const check = useMutation({
    mutationFn: () =>
      api.post<ModelCheck>(
        "/api/models/check",
        useLocal
          ? { task, llm_backend: "local", endpoint: localServer.endpoint, model: localServer.model }
          : {
              task,
              llm_backend: backend,
              endpoint: vertex || claude || routed ? null : endpoint,
              model: vertex ? null : model,
            },
      ),
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
      {only && !openRouter ? (
        <p className="text-gray-700">
          {localReady
            ? `Local model server: ${localServer.model ?? ""} at ${localServer.endpoint ?? ""}`
            : "No connection: this dashboard uses only its local model server (--local-server-only). Start it above."}
        </p>
      ) : (
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
                } else chooseBackend(value as Backend);
              }}
            >
              <option value="local">
                {only
                  ? "This app's local model server"
                  : "OpenAI-compatible server (vLLM, llama.cpp)"}
              </option>
              {vertexKeys.length ? (
                <option value={VERTEX}>Google Vertex / Agent Platform</option>
              ) : null}
              {Object.entries(config.claude_profiles).map(([key, p]) => (
                <option key={key} value={key}>
                  {p.label}
                </option>
              ))}
              {openRouter ? (
                <option value="openrouter">OpenRouter (your own API key)</option>
              ) : null}
            </select>
          </label>
          {useLocal ? (
            <p className="pb-2 text-gray-700">
              {localReady
                ? `Local model server: ${localServer.model ?? ""} at ${localServer.endpoint ?? ""}`
                : "The local model server is not ready: start it above, or use OpenRouter with your own API key."}
            </p>
          ) : routed ? (
            <OpenRouterModel view={routed} model={model} setModel={setModel} />
          ) : vertex ? (
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
                <span className="flex items-center justify-between text-xs text-gray-500">
                  <span>Endpoint (/v1 base URL)</span>
                  {localReady ? (
                    <button
                      type="button"
                      className="text-brand-blue hover:underline"
                      onClick={() => {
                        setEndpoint(localServer.endpoint ?? "");
                        setModel(localServer.model ?? "");
                        setTokenizer(localServer.tokenizer ?? "");
                      }}
                    >
                      Use the local server
                    </button>
                  ) : null}
                </span>
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
      )}
      {routed ? <OpenRouterKey view={routed} /> : null}
      <div className="flex flex-wrap items-center gap-4">
        {!claude && !(routed && routedClaude) ? (
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
          disabled={check.isPending || (useLocal && !localReady)}
          onClick={() => {
            check.mutate();
          }}
        >
          {check.isPending ? <Spinner /> : null}
          Check connection
        </Button>
        <Button
          disabled={save.isPending || (useLocal && !localReady)}
          onClick={() => {
            save.mutate(false);
          }}
        >
          Save for this task
        </Button>
        <Button
          variant="outline"
          disabled={save.isPending || (useLocal && !localReady)}
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

type OfferedOpenRouter = Extract<OpenRouterView, { offered: true }>;

/** An OpenRouter model ID: any, with the app's suggestions (presets/openrouter.yaml). */
function OpenRouterModel({
  view,
  model,
  setModel,
}: {
  view: OfferedOpenRouter;
  model: string;
  setModel: (value: string) => void;
}) {
  const list = useId();
  const known = view.models.find((m) => m.id === model);
  return (
    <label className="flex min-w-96 flex-1 flex-col gap-1">
      <span className="text-xs text-gray-500">
        Model (an OpenRouter model ID{known ? `: ${known.label}` : ""})
      </span>
      <Input
        list={list}
        value={model}
        onChange={(e) => {
          setModel(e.target.value.trim());
        }}
      />
      <datalist id={list}>
        {view.models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </datalist>
    </label>
  );
}

/**
 * The session's OpenRouter key. The server keeps it in memory and never sends it back: the form shows
 * only that a key is set and its last four characters.
 */
function OpenRouterKey({ view }: { view: OfferedOpenRouter }) {
  const client = useQueryClient();
  const [key, setKey] = useState("");
  const done = (next: ModelConfig) => {
    setKey("");
    client.setQueryData(["models"], next);
    void client.invalidateQueries({ queryKey: ["llm"] });
  };
  const save = useMutation({
    mutationFn: () => api.post<ModelConfig>("/api/models/openrouter-key", { key }),
    onSuccess: done,
    onError: () => {
      setKey("");
    },
  });
  const remove = useMutation({
    mutationFn: () => api.delete<ModelConfig>("/api/models/openrouter-key"),
    onSuccess: done,
  });
  return (
    <div className="flex flex-col gap-2 rounded-md bg-amber-50 px-3 py-2 text-amber-900">
      <p>{view.notice}</p>
      {view.key_set ? (
        <div className="flex flex-wrap items-center gap-2">
          <span>API key set ({view.key_hint ?? ""}) for this browser session.</span>
          <Button
            size="sm"
            variant="outline"
            disabled={remove.isPending}
            onClick={() => {
              remove.mutate();
            }}
          >
            Remove key
          </Button>
        </div>
      ) : (
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            save.mutate();
          }}
        >
          <label className="flex min-w-72 flex-1 flex-col gap-1">
            <span className="text-xs">OpenRouter API key</span>
            <Input
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder="sk-or-..."
              value={key}
              onChange={(e) => {
                setKey(e.target.value);
              }}
            />
          </label>
          <Button type="submit" disabled={!key.trim() || save.isPending}>
            Set key
          </Button>
        </form>
      )}
      {save.isError ? <p className="text-red-700">{save.error.message}</p> : null}
      {remove.isError ? <p className="text-red-700">{remove.error.message}</p> : null}
    </div>
  );
}
