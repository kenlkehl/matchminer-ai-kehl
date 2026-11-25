import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";
import {
  COMMON_FIELDS,
  ENGINE_FIELDS,
  type Engine,
  type EngineInfo,
  enginesOptions,
  type Field,
  gb,
  type GpuRow,
  gpusOptions,
  isActive,
  type LocalServerStatus,
  localServerOptions,
  type Preview,
  recipe as fetchRecipe,
  type Recipe,
  type ServerInfo,
  type ServerSpec,
} from "@/lib/local-server";
import type { ModelConfig } from "@/lib/models";

const SELECT = "h-9 rounded-md border border-input bg-background px-2 text-sm";

/**
 * The local model server (offered when the app runs with --local-server): this machine's GPUs, a vLLM
 * or llama.cpp server started from a model's recipe, its progress and log, and using it for this
 * session's LLM tasks. Anyone using the app may start and stop it.
 */
export function LocalServerPanel() {
  const status = useQuery(localServerOptions());
  const gpus = useQuery(gpusOptions());
  const engines = useQuery(enginesOptions());
  if (status.isError) return <p className="text-red-700">{status.error.message}</p>;
  if (!status.data || !engines.data) return <Spinner />;
  const installed = Object.values(engines.data).some((e) => e.available);
  return (
    <div className="flex flex-col gap-4 text-sm">
      {status.data.only ? (
        <p className="rounded-md bg-sky-50 px-3 py-2 text-sky-900">
          This dashboard uses only this local model server (--local-server-only): every LLM task
          uses it once it is ready, and no other model connection is accepted.
        </p>
      ) : null}
      {!installed ? (
        <p className="rounded-md bg-amber-50 px-3 py-2 text-amber-900">
          Neither vLLM nor llama.cpp (llama-server) is installed for this app, so no local server
          can start here. {engines.data.vllm.reason} {engines.data.llama_cpp.reason}
        </p>
      ) : null}
      <GpuTable rows={gpus.data?.gpus ?? []} note={gpus.data?.note ?? null} />
      <ServerForm status={status.data} engines={engines.data} gpus={gpus.data?.gpus ?? []} />
    </div>
  );
}

function GpuTable({ rows, note }: { rows: GpuRow[]; note: string | null }) {
  if (!rows.length)
    return (
      <p className="text-gray-600">
        No NVIDIA GPU was found on this machine; llama.cpp can still run a small model on the CPU.
      </p>
    );
  return (
    <div className="flex flex-col gap-1">
      <table className="w-full text-left">
        <thead className="text-xs text-gray-500">
          <tr>
            <th className="py-1 pr-3 font-medium">GPU</th>
            <th className="py-1 pr-3 font-medium">Compute</th>
            <th className="py-1 pr-3 font-medium">Memory</th>
            <th className="py-1 font-medium">Free</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => (
            <tr key={g.uuid || g.index}>
              <td className="py-1 pr-3">
                {g.index}: {g.name}
                {g.decision_model ? (
                  <span className="ml-2 text-xs text-gray-500">decision model</span>
                ) : null}
                {g.server_gpu ? (
                  <span className="ml-2 text-xs text-emerald-700">local server</span>
                ) : null}
              </td>
              <td className="py-1 pr-3">SM {g.compute_capability ?? "?"}</td>
              <td className="py-1 pr-3">
                <MemoryBar gpu={g} />
              </td>
              <td className="py-1">
                {gb(g.memory_free)} of {gb(g.memory_total)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span>
          <span className="mr-1 inline-block size-2 rounded-sm bg-sky-600" />
          this dashboard
        </span>
        <span>
          <span className="mr-1 inline-block size-2 rounded-sm bg-emerald-600" />
          local server
        </span>
        <span>
          <span className="mr-1 inline-block size-2 rounded-sm bg-gray-400" />
          other processes
        </span>
        {note ? <span>{note}</span> : null}
      </p>
    </div>
  );
}

function MemoryBar({ gpu }: { gpu: GpuRow }) {
  const total = gpu.memory_total || 1;
  const width = (bytes: number | null) => `${String(Math.max(0, ((bytes ?? 0) / total) * 100))}%`;
  return (
    <div
      className="flex h-3 w-56 overflow-hidden rounded-sm bg-gray-100"
      title={`this dashboard ${gb(gpu.dashboard)}, local server ${gb(gpu.local_server)}, other ${gb(gpu.other)}`}
    >
      <div className="bg-sky-600" style={{ width: width(gpu.dashboard) }} />
      <div className="bg-emerald-600" style={{ width: width(gpu.local_server) }} />
      <div className="bg-gray-400" style={{ width: width(gpu.other) }} />
    </div>
  );
}

function elapsed(seconds: number): string {
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  return m ? `${String(m)} min ${String(s % 60)} s` : `${String(s)} s`;
}

function phaseText(server: ServerInfo): string {
  if (
    server.state === "downloading" &&
    server.progress &&
    server.progress.done < server.progress.total
  )
    return `Downloading ${gb(server.progress.done)} of ${gb(server.progress.total)}`;
  if (server.state === "failed") return `Failed: ${server.error ?? ""}`;
  return server.phase;
}

function ServerForm({
  status,
  engines,
  gpus,
}: {
  status: LocalServerStatus;
  engines: Record<Engine, EngineInfo>;
  gpus: GpuRow[];
}) {
  const client = useQueryClient();
  const server = status.server;
  const active = isActive(server);
  const [model, setModel] = useState(server?.spec.model ?? status.default_model);
  const [spec, setSpec] = useState<ServerSpec | null>(server?.spec ?? null);
  const [info, setInfo] = useState<Recipe | null>(null);
  const [reasoning, setReasoning] = useState(false);
  const [loads, setLoads] = useState(0); // remounts Advanced's text boxes with each recipe
  const load = useMutation({
    mutationFn: (args: {
      model: string;
      engine: Engine | null;
      gpus: number[] | null;
      keep?: boolean;
    }) => fetchRecipe(args.model, args.engine, args.gpus),
    onSuccess: (found, args) => {
      setInfo(found);
      setReasoning(found.reasoning);
      if (!args.keep) {
        setSpec(found.spec);
        setModel(found.spec.model);
        setLoads((n) => n + 1);
      }
    },
  });
  const started = useRef(false);
  useEffect(() => {
    // The running server's settings, else the default model's recipe on the GPU with the most free memory.
    if (started.current) return;
    started.current = true;
    if (server)
      load.mutate({
        model: server.spec.model,
        engine: server.spec.engine,
        gpus: server.spec.gpus,
        keep: true,
      });
    else load.mutate({ model: status.default_model, engine: null, gpus: null });
  }, [load, server, status.default_model]);
  const [debounced, setDebounced] = useState(spec);
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebounced(spec);
    }, 400);
    return () => {
      clearTimeout(timer);
    };
  }, [spec]);
  const preview = useQuery({
    queryKey: ["local-server", "preview", JSON.stringify(debounced)],
    queryFn: () => api.post<Preview>("/api/local-server/preview", { spec: debounced }),
    enabled: debounced !== null,
    staleTime: 30_000,
  });
  const refresh = () => client.invalidateQueries({ queryKey: ["local-server"] });
  const start = useMutation({
    mutationFn: async (restart: boolean) => {
      if (restart) await api.post("/api/local-server/stop", {});
      return api.post<LocalServerStatus>("/api/local-server/start", { spec });
    },
    onSettled: refresh,
  });
  const stop = useMutation({
    mutationFn: () => api.post<LocalServerStatus>("/api/local-server/stop", {}),
    onSettled: refresh,
  });
  const use = useMutation({
    mutationFn: () =>
      api.post<ModelConfig>("/api/local-server/use", { apply_to_all: true, reasoning }),
    onSuccess: (next) => {
      client.setQueryData(["models"], next);
      void client.invalidateQueries({ queryKey: ["llm"] });
    },
  });
  if (!spec)
    return load.isError ? <p className="text-red-700">{load.error.message}</p> : <Spinner />;
  const set = (name: string, value: unknown) => {
    setSpec({ ...spec, [name]: value });
  };
  const toggleGpu = (index: number, on: boolean) => {
    const chosen = on ? [...spec.gpus, index] : spec.gpus.filter((g) => g !== index);
    set(
      "gpus",
      [...new Set(chosen)].sort((a, b) => a - b),
    );
  };
  const engineInfo = engines[spec.engine];
  const proposal = info?.proposal;
  const fields = [...ENGINE_FIELDS[spec.engine], ...COMMON_FIELDS];
  return (
    <section className="flex flex-col gap-3 rounded-md border border-gray-200 p-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">Engine</span>
          <select
            className={SELECT}
            value={spec.engine}
            disabled={active}
            onChange={(e) => {
              load.mutate({ model, engine: e.target.value as Engine, gpus: spec.gpus });
            }}
          >
            {(Object.keys(engines) as Engine[]).map((key) => (
              <option key={key} value={key} disabled={!engines[key].available}>
                {engines[key].label}
                {engines[key].available
                  ? engines[key].version
                    ? ` ${engines[key].version ?? ""}`
                    : ""
                  : ` (${engines[key].reason ?? "not installed"})`}
              </option>
            ))}
          </select>
        </label>
        <label className="flex min-w-96 flex-1 flex-col gap-1">
          <span className="text-xs text-gray-500">Model (Hugging Face repository)</span>
          <Input
            list="local-server-models"
            value={model}
            disabled={active}
            onChange={(e) => {
              setModel(e.target.value);
            }}
          />
          <datalist id="local-server-models">
            {status.known_models.map((m) => (
              <option key={m.model} value={m.model}>
                {m.label}
              </option>
            ))}
          </datalist>
        </label>
        <Button
          variant="outline"
          disabled={active || load.isPending || !model.trim()}
          onClick={() => {
            load.mutate({ model: model.trim(), engine: null, gpus: spec.gpus });
          }}
        >
          {load.isPending ? <Spinner /> : null}
          Load recommended settings
        </Button>
        {spec.engine === "llama_cpp" && info?.gguf_choices.length ? (
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500">GGUF file</span>
            <select
              className={SELECT}
              value={String(spec.gguf ?? "")}
              disabled={active}
              onChange={(e) => {
                set("gguf", e.target.value);
              }}
            >
              {info.gguf_choices.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({gb(c.size)})
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      {info ? (
        <div className="flex flex-col gap-0.5 text-xs text-gray-600">
          <p>
            {info.label}
            {info.variant ? ` · ${info.variant}` : ""}
            {info.note ? ` · ${info.note}` : ""}
          </p>
          {proposal ? (
            <p>
              {proposal.engine === spec.engine
                ? proposal.reason
                : `Proposed: ${engines[proposal.engine].label}. ${proposal.reason}`}
            </p>
          ) : null}
          {info.error ? <p className="text-red-700">{info.error}</p> : null}
        </div>
      ) : null}
      {!engineInfo.available ? <p className="text-red-700">{engineInfo.reason}</p> : null}
      <fieldset className="flex flex-wrap gap-3" disabled={active}>
        <legend className="mb-1 text-xs text-gray-500">GPUs</legend>
        {gpus.map((g) => (
          <label key={g.index} className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={spec.gpus.includes(g.index)}
              onChange={(e) => {
                toggleGpu(g.index, e.target.checked);
              }}
            />
            GPU {g.index} ({gb(g.memory_free)} free)
          </label>
        ))}
        {!gpus.length ? <span className="text-gray-500">none (CPU only)</span> : null}
      </fieldset>
      <fieldset
        className="grid grid-cols-[repeat(auto-fill,minmax(13rem,1fr))] gap-2"
        disabled={active}
      >
        <legend className="mb-1 flex items-center gap-3 text-xs text-gray-500">
          Engine parameters
          <button
            type="button"
            className="text-brand-blue hover:underline disabled:opacity-50"
            disabled={active || load.isPending}
            onClick={() => {
              load.mutate({ model, engine: spec.engine, gpus: spec.gpus });
            }}
          >
            Reset to recommended
          </button>
        </legend>
        {fields.map((f) => (
          <FieldInput
            key={f.name}
            field={f}
            value={spec[f.name]}
            onChange={(v) => {
              set(f.name, v);
            }}
          />
        ))}
      </fieldset>
      <details className="text-sm">
        <summary className="cursor-pointer text-xs text-gray-500">Advanced</summary>
        <Advanced key={loads} spec={spec} disabled={active} onChange={setSpec} />
      </details>
      <div className="flex flex-col gap-1">
        <code className="rounded-md bg-gray-50 px-2 py-1.5 text-xs break-all text-gray-700">
          {preview.data?.command ?? info?.command ?? ""}
        </code>
        {preview.data?.error ? <p className="text-red-700">{preview.data.error}</p> : null}
        {(preview.data?.warnings ?? []).map((w) => (
          <p key={w} className="text-amber-800">
            {w}
          </p>
        ))}
        {preview.data?.download_bytes ? (
          <p className="text-xs text-gray-500">
            Files: {gb(preview.data.download_bytes)} in the Hugging Face cache ({status.hub_cache})
            {status.allow_downloads
              ? ""
              : "; this app does not download, so they must be cached already"}
          </p>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          disabled={active || start.isPending || !engineInfo.available}
          onClick={() => {
            start.mutate(false);
          }}
        >
          Start
        </Button>
        <Button
          variant="outline"
          disabled={!active || stop.isPending || server?.state === "stopping"}
          onClick={() => {
            stop.mutate();
          }}
        >
          {stop.isPending ? <Spinner /> : null}
          Stop
        </Button>
        <Button
          variant="outline"
          disabled={!active || start.isPending}
          onClick={() => {
            start.mutate(true);
          }}
        >
          Restart
        </Button>
        {server ? (
          <span
            className={server.state === "failed" || server.state === "exited" ? "text-red-700" : ""}
          >
            {server.state === "ready" ? "Ready" : phaseText(server)}
            {server.state === "exited" && server.exit_code !== null
              ? ""
              : ` · ${elapsed(server.elapsed_seconds)}`}
            {server.state === "ready" ? ` · ${server.model} at ${server.endpoint}` : ""}
          </span>
        ) : null}
      </div>
      {start.isError ? <p className="text-red-700">{start.error.message}</p> : null}
      {stop.isError ? <p className="text-red-700">{stop.error.message}</p> : null}
      {server?.state === "ready" ? (
        <div className="flex flex-wrap items-center gap-3 rounded-md bg-emerald-50 px-3 py-2">
          <Button
            disabled={use.isPending}
            onClick={() => {
              use.mutate();
            }}
          >
            Use for every task
          </Button>
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
          {use.isSuccess ? (
            <span className="text-green-700">
              This browser session's tasks now use the local server.
            </span>
          ) : null}
          {use.isError ? <span className="text-red-700">{use.error.message}</span> : null}
        </div>
      ) : null}
      {status.default_on_ready && !status.only ? (
        <p className="text-xs text-gray-500">
          Once ready, the server is also the default connection of every task in sessions that chose
          none, until it stops.
        </p>
      ) : null}
      {server?.log.length ? (
        <details className="text-sm">
          <summary className="cursor-pointer text-xs text-gray-500">
            Log (last {server.log.length} lines)
          </summary>
          <pre className="mt-1 max-h-80 overflow-auto rounded-md bg-gray-50 p-2 text-xs whitespace-pre-wrap">
            {server.log.join("\n")}
          </pre>
        </details>
      ) : null}
    </section>
  );
}

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const label = (
    <span className="text-xs text-gray-500">
      {field.label}
      {field.help ? <span className="text-gray-400"> ({field.help})</span> : null}
    </span>
  );
  if (field.kind === "bool")
    return (
      <label className="flex items-center gap-2 pt-4">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => {
            onChange(e.target.checked);
          }}
        />
        {field.label}
      </label>
    );
  if (field.kind === "choice") {
    // Prefix caching is a boolean on the server: on/off here.
    const shown = typeof value === "boolean" ? (value ? "on" : "off") : String(value ?? "");
    return (
      <label className="flex flex-col gap-1">
        {label}
        <select
          className={SELECT}
          value={shown}
          onChange={(e) => {
            const v = e.target.value;
            if (field.name === "enable_prefix_caching") onChange(v === "" ? null : v === "on");
            else onChange(v === "" ? (field.empty ?? null) : v);
          }}
        >
          {field.empty ? null : <option value="">default</option>}
          {field.choices?.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const numeric = field.kind === "int" || field.kind === "float";
  return (
    <label className="flex flex-col gap-1">
      {label}
      <Input
        type={numeric ? "number" : "text"}
        step={field.kind === "float" ? "0.01" : undefined}
        value={value === null || value === undefined ? "" : String(value as string | number)}
        placeholder="default"
        onChange={(e) => {
          const v = e.target.value;
          if (numeric) onChange(v === "" ? null : Number(v));
          else onChange(v === "" ? (field.empty ?? null) : v);
        }}
      />
    </label>
  );
}

/** Extra command-line arguments (one per line), environment variables and vLLM's speculative config. */
function Advanced({
  spec,
  disabled,
  onChange,
}: {
  spec: ServerSpec;
  disabled: boolean;
  onChange: (spec: ServerSpec) => void;
}) {
  const [speculative, setSpeculative] = useState(
    spec.speculative_config ? JSON.stringify(spec.speculative_config) : "",
  );
  const [error, setError] = useState<string | null>(null);
  const area = "min-h-20 w-full rounded-md border border-input bg-background p-2 font-mono text-xs";
  return (
    <div className="mt-2 flex flex-col gap-2">
      <label className="flex flex-col gap-1">
        <span className="text-xs text-gray-500">
          Extra arguments, one per line (no --host or --port)
        </span>
        <textarea
          className={area}
          disabled={disabled}
          value={spec.extra_args.join("\n")}
          onChange={(e) => {
            onChange({ ...spec, extra_args: e.target.value.split("\n").filter((a) => a.trim()) });
          }}
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-xs text-gray-500">
          Environment, NAME=value per line (VLLM_, GGML_, NCCL_, PYTORCH_, TORCH_, FLASHINFER_,
          CUDA_)
        </span>
        <textarea
          className={area}
          disabled={disabled}
          value={Object.entries(spec.env)
            .map(([k, v]) => `${k}=${v}`)
            .join("\n")}
          onChange={(e) => {
            const env: Record<string, string> = {};
            for (const line of e.target.value.split("\n")) {
              const at = line.indexOf("=");
              if (at > 0) env[line.slice(0, at).trim()] = line.slice(at + 1);
            }
            onChange({ ...spec, env });
          }}
        />
      </label>
      {spec.engine === "vllm" ? (
        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">
            Speculative decoding config (JSON; its model is downloaded first)
          </span>
          <textarea
            className={area}
            disabled={disabled}
            value={speculative}
            onChange={(e) => {
              setSpeculative(e.target.value);
              try {
                const parsed = e.target.value.trim()
                  ? (JSON.parse(e.target.value) as Record<string, unknown>)
                  : null;
                setError(null);
                onChange({ ...spec, speculative_config: parsed });
              } catch {
                setError("Not valid JSON yet.");
              }
            }}
          />
          {error ? <span className="text-xs text-red-700">{error}</span> : null}
        </label>
      ) : null}
    </div>
  );
}
