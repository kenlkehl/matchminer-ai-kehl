import { queryOptions, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { ChevronRightIcon, SendIcon, SquareIcon, SquarePenIcon } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { Markdown } from "@/components/markdown";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { api, type PatientDetail } from "@/lib/api";
import {
  fetchAnswer,
  type LLMBoilerplateAnswer,
  type LLMMatchAnswer,
  useScorer,
} from "@/lib/llm-decisions";
import { modelsOptions } from "@/lib/models";
import { LLMAnswerDialog } from "@/features/matching/llm-score-view";
import { ScorerToggle } from "@/features/matching/scorer-toggle";
import { pickCorpusPatient } from "@/features/patient/actions";
import { TaskForm } from "@/features/status/models-panel";

import { AnswerCharts } from "./answer-charts";
import type { AssistantEntry, AssistantView, LinkTarget, Step, Usage } from "./types";

const assistantOptions = () =>
  queryOptions({
    queryKey: ["assistant"],
    queryFn: () => api.get<AssistantView>("/api/assistant"),
    // Each step shows as it runs.
    refetchInterval: (query) => (query.state.data?.running ? 1000 : false),
    staleTime: 0,
  });

const EXAMPLES = [
  "What trials are open for EGFR-mutant lung cancer?",
  "Find patients with metastatic colorectal cancer",
];

/**
 * The Assistant tab: a chat over this dashboard's trials, patients, structured records and guideline
 * spaces. The assistant calls the app's read-only tools, asks when a clinical concept is ambiguous, and
 * answers with links to the Clinical trial management and Patient-centric views.
 */
export function AssistantPage() {
  const client = useQueryClient();
  const view = useQuery(assistantOptions());
  const [draft, setDraft] = useState("");
  const [modelOpen, setModelOpen] = useState(false);
  const data = view.data;
  const running = !!data?.running;
  // The session's "Scores from" choice, sent with each message: the turn's reranks and pair scores use it.
  const scorer = useScorer();
  const send = useMutation({
    mutationFn: (text: string) =>
      api.post<AssistantView & { job: unknown }>("/api/assistant/messages", { text, scorer }),
    onSuccess: (next) => {
      setDraft("");
      client.setQueryData(["assistant"], next);
    },
  });
  const stop = useMutation({
    mutationFn: () => api.post("/api/assistant/stop", {}),
    onSettled: () => void client.invalidateQueries({ queryKey: ["assistant"] }),
  });
  const reset = useMutation({
    mutationFn: () => api.delete("/api/assistant"),
    onSettled: () => void client.invalidateQueries({ queryKey: ["assistant"] }),
  });
  const entries = data?.conversation?.entries ?? [];
  const last = entries[entries.length - 1];
  const pending = last?.role === "assistant" && last.kind === "question" && !running ? last : null;
  const bottom = useRef<HTMLDivElement>(null);
  const stepCount = entries.reduce((n, e) => n + (e.role === "assistant" ? e.steps.length : 0), 0);
  // Every tool step by its conversation-wide number: the sources a chart cites.
  const steps = useMemo(
    () =>
      new Map(
        entries.flatMap((e) =>
          e.role === "assistant"
            ? e.steps.flatMap((s) => (s.number ? [[s.number, s] as const] : []))
            : [],
        ),
      ),
    [entries],
  );
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [entries.length, stepCount, last?.role === "assistant" ? last.status : null]);

  const submit = (text: string) => {
    const value = text.trim();
    if (value && !running && !send.isPending) send.mutate(value);
  };

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Assistant">
        <p>
          Ask about this dashboard's trials, patients, structured records and guideline spaces. The
          assistant searches them with the dashboard's own tools, asks when a clinical concept could
          be read more than one way, and links the trials, cohorts and patients it finds.
        </p>
      </PageHeader>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span className="text-gray-600">
          Model: <strong className="text-gray-900">{data?.model.label ?? "…"}</strong>
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setModelOpen(true);
          }}
        >
          Change model
        </Button>
        <ScorerToggle />
        {data?.conversation ? <UsageLine usage={data.conversation.usage} /> : null}
        <Button
          size="sm"
          variant="outline"
          className="ml-auto"
          disabled={reset.isPending || !data?.conversation}
          onClick={() => {
            reset.mutate();
          }}
        >
          <SquarePenIcon />
          New conversation
        </Button>
      </div>
      {view.isError ? <p className="text-sm text-red-700">{view.error.message}</p> : null}
      {data && Object.keys(data.unavailable).length ? (
        <p className="text-xs text-gray-500">
          Not available in this mode:{" "}
          {Object.entries(data.unavailable)
            .map(([tool, why]) => `${TOOL_NAMES[tool] ?? tool} (${why})`)
            .join("; ")}
          .
        </p>
      ) : null}

      <section aria-label="Conversation" className="flex flex-col gap-4">
        {entries.length === 0 && data ? (
          <div className="flex flex-col gap-2 rounded-md border border-dashed border-gray-300 p-4 text-sm">
            <p className="text-gray-600">For example:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <Button
                  key={example}
                  size="sm"
                  variant="outline"
                  disabled={running || send.isPending}
                  onClick={() => {
                    submit(example);
                  }}
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>
        ) : null}
        {entries.map((entry) =>
          entry.role === "user" ? (
            <div key={entry.id} className="flex justify-end">
              <p className="max-w-3xl rounded-lg bg-brand-light-blue/40 px-3 py-2 text-sm whitespace-pre-wrap">
                {entry.text}
              </p>
            </div>
          ) : (
            <Turn
              key={entry.id}
              entry={entry}
              steps={steps}
              links={data?.conversation?.links ?? {}}
              current={entry === pending}
              onOption={submit}
            />
          ),
        )}
      </section>

      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          submit(draft);
        }}
      >
        <Textarea
          value={draft}
          maxLength={data?.limits.max_message_chars ?? 4000}
          placeholder={
            pending
              ? "Choose an option above, or type your own answer"
              : "Ask about trials or patients"
          }
          className="min-h-20"
          onChange={(e) => {
            setDraft(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(draft);
            }
          }}
        />
        <div className="flex items-center gap-2">
          {running ? (
            <Button
              type="button"
              variant="outline"
              disabled={stop.isPending}
              onClick={() => {
                stop.mutate();
              }}
            >
              <SquareIcon />
              Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!draft.trim() || send.isPending}>
              {send.isPending ? <Spinner /> : <SendIcon />}
              Send
            </Button>
          )}
          <span className="text-xs text-gray-500">Enter sends; Shift+Enter adds a line.</span>
          {send.isError ? <span className="text-sm text-red-700">{send.error.message}</span> : null}
        </div>
      </form>
      {/* Kept in view as the transcript grows, with the message box above it. */}
      <div ref={bottom} />
      <ModelDialog
        open={modelOpen}
        onClose={() => {
          setModelOpen(false);
          void client.invalidateQueries({ queryKey: ["assistant"] });
        }}
      />
    </div>
  );
}

const TOOL_NAMES: Record<string, string> = {
  search_trials: "trial search",
  match_trials: "TrialSpace trial matching",
  trial_details: "trial details",
  search_patients: "patient search",
  match_patients: "TrialSpace patient matching",
  patient_details: "patient details",
  screen_records: "structured-record screening",
  score_pairs: "decision-model scoring",
  guideline_spaces: "guideline spaces",
};

function UsageLine({ usage }: { usage: Usage }) {
  const n = (value: number) => value.toLocaleString();
  return (
    <span className="text-xs text-gray-500 tabular-nums">
      {n(usage.requests)} requests · {n(usage.input_tokens)} input tokens
      {usage.cache_read_tokens ? ` (${n(usage.cache_read_tokens)} from cache)` : ""} ·{" "}
      {n(usage.output_tokens)} output tokens
    </span>
  );
}

/** One assistant turn: its tool steps, then its answer, question, or why it ended. */
function Turn({
  entry,
  steps,
  links,
  current,
  onOption,
}: {
  entry: AssistantEntry;
  steps: Map<number, Step>;
  links: Record<string, LinkTarget>;
  current: boolean;
  onOption: (text: string) => void;
}) {
  const renderText = useIdLinks(links);
  const working = (
    <p role="status" aria-live="polite" className="flex items-center gap-2 text-sm text-gray-600">
      <Spinner className="text-brand-blue" />
      {entry.progress || "Working"}
    </p>
  );
  return (
    <article className="flex max-w-4xl flex-col gap-2 rounded-lg border border-gray-200 p-3">
      {entry.steps.length ? (
        <ol className="flex flex-col gap-1">
          {entry.steps.map((step, n) => (
            <StepLine key={n} step={step} />
          ))}
        </ol>
      ) : null}
      {entry.status === "running" && entry.kind !== "answer" ? working : null}
      {entry.kind === "answer" ? (
        <>
          <Markdown text={entry.text} renderText={renderText} />
          {entry.charts?.length ? <AnswerCharts charts={entry.charts} steps={steps} /> : null}
          {/* The answer shows while its charts are drawn. */}
          {entry.status === "running" ? working : null}
          {entry.chart_error ? <p className="text-sm text-amber-800">{entry.chart_error}</p> : null}
          {entry.interpretation ? (
            <p className="text-sm text-gray-600 italic">{entry.interpretation}</p>
          ) : null}
        </>
      ) : null}
      {entry.kind === "question" ? (
        <div className="flex flex-col gap-2">
          <Markdown text={entry.text} renderText={renderText} />
          {entry.why ? <p className="text-xs text-gray-500">{entry.why}</p> : null}
          <div className="flex flex-wrap gap-2">
            {entry.options.map((option) => (
              <Button
                key={option}
                size="sm"
                variant={current ? "default" : "outline"}
                disabled={!current}
                onClick={() => {
                  onOption(option);
                }}
              >
                {option}
              </Button>
            ))}
          </div>
        </div>
      ) : null}
      {entry.status === "stopped" ? <p className="text-sm text-gray-500">Stopped.</p> : null}
      {entry.status === "error" && entry.error ? (
        <p className="text-sm text-red-700">{entry.error}</p>
      ) : null}
      <p className="text-xs text-gray-400">
        {entry.model}
        {entry.seconds !== null ? ` · ${String(entry.seconds)} s` : ""}
      </p>
    </article>
  );
}

function StepLine({ step }: { step: Step }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="text-sm">
      <button
        type="button"
        aria-expanded={open}
        className="flex w-full items-start gap-1 text-left text-gray-700 hover:text-gray-950"
        onClick={() => {
          setOpen(!open);
        }}
      >
        <ChevronRightIcon
          className={`mt-0.5 size-4 shrink-0 text-gray-400 transition-transform ${open ? "rotate-90" : ""}`}
        />
        {step.ok === null ? <Spinner className="mt-0.5 size-3.5 text-brand-blue" /> : null}
        {step.number ? (
          <span className="shrink-0 text-gray-400 tabular-nums">{step.number}.</span>
        ) : null}
        <span className={step.ok === false ? "text-amber-800" : undefined}>
          {step.summary || `${step.label}…`}
        </span>
      </button>
      {open ? (
        <div className="ml-5 flex flex-col gap-1 text-xs text-gray-500">
          <p>
            {step.purpose}
            {step.seconds !== null ? ` (${String(step.seconds)} s)` : ""}
          </p>
          {step.llm_answers?.length ? <StepAnswers answers={step.llm_answers} /> : null}
        </div>
      ) : null}
    </li>
  );
}

/** A scoring step's LLM answers: each opens its reasoning and raw output (the views' answer dialog). */
function StepAnswers({ answers }: { answers: NonNullable<Step["llm_answers"]> }) {
  const [shown, setShown] = useState<LLMMatchAnswer | LLMBoilerplateAnswer | null>(null);
  const open = useMutation({
    mutationFn: (key: string) => fetchAnswer(key),
    onSuccess: setShown,
  });
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span>LLM answers:</span>
      {answers.map((a) => (
        <button
          key={a.key}
          type="button"
          className="link"
          disabled={open.isPending}
          onClick={() => {
            open.mutate(a.key);
          }}
        >
          {a.patient_id} × {a.cohort_id} ({a.family === "match" ? "match" : "common exclusions"})
        </button>
      ))}
      {open.isError ? <span className="text-red-700">{open.error.message}</span> : null}
      {shown ? (
        <LLMAnswerDialog
          answer={shown}
          onClose={() => {
            setShown(null);
          }}
        />
      ) : null}
    </div>
  );
}

/** Plain text with every ID a tool returned made a link to its view. */
function useIdLinks(links: Record<string, LinkTarget>) {
  const pattern = useMemo(() => {
    const ids = Object.keys(links).sort((a, b) => b.length - a.length);
    if (!ids.length) return null;
    const escaped = ids.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    return new RegExp(`(?<![\\w-])(${escaped.join("|")})(?![\\w-])`, "g");
  }, [links]);
  return (text: string, key: string): ReactNode => {
    if (!pattern) return text;
    const out: ReactNode[] = [];
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      const id = match[1] ?? "";
      const target = links[id];
      if (match.index > last) out.push(text.slice(last, match.index));
      out.push(
        target ? <IdLink key={`${key}-${String(match.index)}`} id={id} target={target} /> : id,
      );
      last = match.index + match[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  };
}

function IdLink({ id, target }: { id: string; target: LinkTarget }) {
  const navigate = useNavigate();
  const [opening, setOpening] = useState(false);
  const style = "font-medium text-brand-blue underline-offset-2 hover:underline";
  if (target.kind === "patient") {
    return (
      <button
        type="button"
        className={style}
        disabled={opening}
        title="Open in Patient-centric search"
        onClick={() => {
          setOpening(true);
          void api
            .get<PatientDetail>(`/api/patients/${encodeURIComponent(id)}`)
            .then((patient) => {
              pickCorpusPatient(patient);
              return navigate({ to: "/patient" });
            })
            .finally(() => {
              setOpening(false);
            });
        }}
      >
        {id}
      </button>
    );
  }
  const search =
    target.kind === "trial"
      ? { center: target.center_id ?? undefined, details: id }
      : { center: target.center_id ?? undefined, cohort: id };
  return (
    <Link
      to="/"
      search={{ ...search, ...(target.open === false ? { closed: true } : {}) }}
      className={style}
      title="Open in Clinical trial management"
    >
      {id}
    </Link>
  );
}

function ModelDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const config = useQuery({ ...modelsOptions(), enabled: open });
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <DialogContent className="sm:max-w-3xl" bodyClassName="gap-4 auto-rows-min">
        <DialogHeader>
          <DialogTitle>Chat assistant model</DialogTitle>
          <DialogDescription>
            The Chat assistant task's connection for this browser session, as on Data and models. A
            running turn keeps the connection it started with.
          </DialogDescription>
        </DialogHeader>
        {config.data ? (
          <>
            <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900">
              {config.data.notice}
            </p>
            <TaskForm
              key={config.data.tasks.assistant.model}
              config={config.data}
              task="assistant"
            />
          </>
        ) : config.isError ? (
          <p className="text-sm text-red-700">{config.error.message}</p>
        ) : (
          <Spinner />
        )}
      </DialogContent>
    </Dialog>
  );
}
