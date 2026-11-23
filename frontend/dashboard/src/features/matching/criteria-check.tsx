import { useMutation, useQuery } from "@tanstack/react-query";
import { BotIcon, DatabaseIcon, ListChecksIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { CriteriaList, disagree } from "@/components/criteria-list";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  api,
  type CriteriaItem,
  type CriteriaResult,
  type Job,
  runCheck,
  waitForJob,
} from "@/lib/api";
import {
  estimateText,
  type LLMCriteriaResult,
  type LLMEstimate,
  useScorer,
} from "@/lib/llm-decisions";
import { llmOptions } from "@/lib/queries";
import {
  type CriteriaRecordsResult,
  describeSource,
  ENGINE_LABELS,
  type Engine,
  type RecordsBody,
} from "@/lib/records";

import { ScorerToggle } from "./scorer-toggle";

/**
 * The per-criterion check of one patient against every line of one cohort: yes means the patient does not
 * clearly fail the line (an inclusion line clearly not met, or an exclusion line clearly met). The session's
 * scorer (the toggle) says who answers: the decision model, or the LLM (the per-criterion teacher prompt, one
 * request per line, its estimate shown first; a large run waits for confirmation). When both have answered, the
 * other's answers stay beside, labeled, with their disagreements. "Check with structured data" runs each line's
 * query (compiled once per line) on the patient's extraction records: the stored records of a corpus patient,
 * else (`records`) the Patient tab's extraction of its notes.
 */
export function CriteriaCheck({
  patientId,
  cohortId,
  items,
  autoStart = false,
  records,
}: {
  patientId: string;
  cohortId: string;
  items: CriteriaItem[];
  autoStart?: boolean;
  /** Which records to screen (default: the corpus patient's stored records). */
  records?: () => RecordsBody;
}) {
  const scorer = useScorer();
  const [progress, setProgress] = useState("");
  const check = useMutation({
    mutationFn: async () => {
      const { job } = await api.post<{ job: Job }>("/api/criteria", {
        patient_id: patientId,
        cohort_id: cohortId,
      });
      return waitForJob<CriteriaResult>(job.id, (j) => {
        setProgress(j.message);
      });
    },
  });
  const result = check.data;
  const fails = result?.lines.filter((l) => l.fails) ?? [];
  const status = useQuery(llmOptions()).data?.decisions;
  const llmReason = !status
    ? "Checking the LLM connection…"
    : status.available
      ? null
      : status.reason;
  // The session's answer and the run's estimate (no model calls), for the session's connection.
  const lookup = useQuery({
    queryKey: [
      "llm-criteria",
      patientId,
      cohortId,
      status?.connection.backend,
      status?.connection.model,
    ],
    queryFn: () =>
      api.post<{ result: LLMCriteriaResult | null; estimate: LLMEstimate }>(
        "/api/llm-decisions/criteria",
        { patient_id: patientId, cohort_id: cohortId, lookup: true },
      ),
    enabled: scorer === "llm" && !!status,
    staleTime: Infinity,
  });
  const [llmProgress, setLlmProgress] = useState("");
  const [confirming, setConfirming] = useState(false);
  const llm = useMutation({
    mutationFn: ({ rerun, confirm }: { rerun: boolean; confirm: boolean }) =>
      runCheck<LLMCriteriaResult>(
        "/api/llm-decisions/criteria",
        { patient_id: patientId, cohort_id: cohortId, rerun, confirm },
        (j) => {
          setLlmProgress(
            `${j.message || j.status}${j.seconds ? ` · ${String(Math.round(j.seconds))} s` : ""}`,
          );
        },
      ),
  });
  const answers = llm.data ?? lookup.data?.result ?? undefined;
  const estimate = lookup.data?.estimate;
  const askLLM = (rerun: boolean) => {
    if (estimate?.large && !confirming) {
      setConfirming(true);
      return;
    }
    setConfirming(false);
    llm.mutate({ rerun, confirm: Boolean(estimate?.large) });
  };
  // Opened from a list: the chosen scorer answers at once (the LLM only when its run is small and not cached).
  const started = useRef<string | null>(null);
  const { mutate: runDecision } = check;
  const { mutate: runLLM } = llm;
  useEffect(() => {
    if (!autoStart || started.current === scorer) return;
    if (scorer === "decision_model") {
      started.current = scorer;
      runDecision();
    } else if (lookup.data) {
      started.current = scorer;
      if (!lookup.data.result && !lookup.data.estimate.large && llmReason === null)
        runLLM({ rerun: false, confirm: false });
    }
  }, [autoStart, scorer, lookup.data, llmReason, runDecision, runLLM]);
  const [engine, setEngine] = useState<Engine>("queries");
  const [recordsProgress, setRecordsProgress] = useState("");
  const structured = useMutation({
    mutationFn: async (which: Engine) => {
      const body = {
        ...(records ? records() : { patient_id: patientId }),
        cohort_id: cohortId,
        engine: which,
      };
      const { job } = await api.post<{ job: Job }>("/api/criteria-records", body);
      return waitForJob<CriteriaRecordsResult>(job.id, (j) => {
        setRecordsProgress(j.message);
      });
    },
  });
  const screened = structured.data;
  const decisionSummary = result ? (
    <span className="text-sm text-gray-700">
      {scorer === "llm" ? <strong>Decision model: </strong> : null}
      {fails.length === 0 ? (
        <>No criterion is clearly failed ({result.lines.length} checked).</>
      ) : (
        <>
          <strong className="text-red-700">
            {fails.length} {fails.length === 1 ? "criterion" : "criteria"} clearly failed
          </strong>{" "}
          of {result.lines.length} checked.
        </>
      )}
    </span>
  ) : null;
  return (
    <div className="flex flex-col gap-3">
      <ScorerToggle />
      {scorer === "decision_model" ? (
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="sm"
            variant={result ? "outline" : "default"}
            disabled={check.isPending}
            onClick={() => {
              check.mutate();
            }}
          >
            {check.isPending ? <Spinner /> : <ListChecksIcon />}
            {result ? "Check again" : "Check each criterion"}
          </Button>
          {check.isPending ? (
            <span className="text-sm text-gray-500">{progress || "Checking each criterion…"}</span>
          ) : null}
          {check.isError ? (
            <span className="text-sm text-red-700">{check.error.message}</span>
          ) : null}
          {decisionSummary}
          {answers ? <LLMSummary answers={answers} decisions={result} /> : null}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="sm"
            variant={answers ? "outline" : "default"}
            disabled={llm.isPending || llmReason !== null || !estimate}
            title={llmReason ?? undefined}
            onClick={() => {
              askLLM(Boolean(answers));
            }}
          >
            {llm.isPending ? <Spinner /> : <BotIcon />}
            {answers ? "Ask the LLM again" : "Check each criterion with the LLM"}
          </Button>
          {estimate && !answers && !llm.isPending ? (
            <span className="text-xs text-gray-500">{estimateText(estimate)}</span>
          ) : null}
          {confirming && estimate ? (
            <span className="flex flex-wrap items-center gap-2 rounded-md bg-amber-50 px-2 py-1 text-sm text-amber-950">
              This sends {estimateText(estimate)}.
              <Button
                size="xs"
                onClick={() => {
                  askLLM(Boolean(answers));
                }}
              >
                Check them
              </Button>
              <Button
                size="xs"
                variant="outline"
                onClick={() => {
                  setConfirming(false);
                }}
              >
                Cancel
              </Button>
            </span>
          ) : null}
          {llm.isPending ? (
            <span className="text-sm text-gray-500">{llmProgress || "Asking the LLM…"}</span>
          ) : null}
          {llm.isError ? <span className="text-sm text-red-700">{llm.error.message}</span> : null}
          {lookup.isError ? (
            <span className="text-sm text-red-700">{lookup.error.message}</span>
          ) : null}
          {answers ? <LLMSummary answers={answers} decisions={result} /> : null}
          {decisionSummary}
          {status ? <span className="w-full text-xs text-gray-500">{status.notice}</span> : null}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <select
          aria-label="Structured data engine"
          className="h-8 rounded-md border border-gray-300 bg-white px-2 text-sm"
          value={engine}
          disabled={structured.isPending}
          onChange={(e) => {
            setEngine(e.target.value as Engine);
          }}
        >
          {(Object.keys(ENGINE_LABELS) as Engine[]).map((key) => (
            <option key={key} value={key}>
              {ENGINE_LABELS[key]}
            </option>
          ))}
        </select>
        <Button
          size="sm"
          variant="outline"
          disabled={structured.isPending}
          onClick={() => {
            structured.mutate(engine);
          }}
        >
          {structured.isPending ? <Spinner /> : <DatabaseIcon />}
          {screened ? "Check structured data again" : "Check with structured data"}
        </Button>
        {structured.isPending ? (
          <span className="text-sm text-gray-500">
            {recordsProgress || "Compiling and running the criteria queries…"}
          </span>
        ) : null}
        {structured.isError ? (
          <span className="text-sm text-red-700">{structured.error.message}</span>
        ) : null}
        {screened ? <RecordsSummary result={screened} decisions={result} /> : null}
      </div>
      <CriteriaList
        items={items}
        checks={result?.lines ?? null}
        llm={answers?.lines ?? null}
        records={screened?.lines ?? null}
      />
    </div>
  );
}

/** The LLM's per-criterion answers in one line: failures, agreement with the decision model, model and time. */
function LLMSummary({
  answers,
  decisions,
}: {
  answers: LLMCriteriaResult;
  decisions: CriteriaResult | undefined;
}) {
  const byNumber = new Map((decisions?.lines ?? []).map((l) => [l.number, l]));
  const compared = answers.lines.filter((l) => l.fails !== null && byNumber.has(l.number));
  const differ = answers.lines.filter((l) => disagree(byNumber.get(l.number), l));
  const c = answers.counts;
  return (
    <span className="text-sm text-gray-700">
      <strong>LLM:</strong>{" "}
      {c.clearly_fail ? (
        <strong className="text-red-700">
          {c.clearly_fail} {c.clearly_fail === 1 ? "criterion" : "criteria"} clearly failed
        </strong>
      ) : (
        "no criterion clearly failed"
      )}{" "}
      of {c.lines}
      {c.unanswered ? ` (${String(c.unanswered)} without a valid answer)` : ""}
      {decisions ? (
        <>
          {" · "}agrees with the decision model on {compared.length - differ.length} of{" "}
          {compared.length}
          {differ.length ? (
            <strong className="text-amber-700">
              {" "}
              ({differ.length} {differ.length === 1 ? "disagreement" : "disagreements"})
            </strong>
          ) : null}
        </>
      ) : null}
      <span className="text-xs text-gray-500">
        {" · "}
        {answers.connection.label} · {answers.seconds} s{answers.cached ? " · cached" : ""}
        {c.retries ? ` · ${String(c.retries)} retries` : ""}
      </span>
    </span>
  );
}

/** The structured-record screen in one line: failures, what could be checked, agreement, source. */
function RecordsSummary({
  result,
  decisions,
}: {
  result: CriteriaRecordsResult;
  decisions: CriteriaResult | undefined;
}) {
  const c = result.counts;
  const notCheckable = c.not_checkable + c.query_unavailable;
  const byNumber = new Map((decisions?.lines ?? []).map((l) => [l.number, l]));
  const compared = result.lines.filter(
    (l) =>
      byNumber.has(l.number) &&
      !["not_checkable", "no_records", "query_unavailable"].includes(l.status),
  );
  const differ = compared.filter((l) => byNumber.get(l.number)?.fails !== l.clearly_fails);
  return (
    <span className="text-sm text-gray-700">
      <strong>Records:</strong>{" "}
      {c.clearly_failed_lines.length ? (
        <strong className="text-red-700">
          {c.clearly_failed_lines.length} of {c.checked} checkable{" "}
          {c.checked === 1 ? "criterion" : "criteria"} clearly fail
        </strong>
      ) : (
        <>no criterion clearly fails ({c.checked} checkable)</>
      )}
      {notCheckable ? ` · ${String(notCheckable)} not checkable from records` : ""}
      {decisions && compared.length ? (
        <>
          {" · "}agrees with the decision model on {compared.length - differ.length} of{" "}
          {compared.length}
        </>
      ) : null}
      <span className="text-xs text-gray-500">
        {" · "}
        {ENGINE_LABELS[result.engine]} · {describeSource(result.source)}
        {result.compile.requests
          ? ` · ${String(result.compile.requests)} compile requests`
          : " · cached queries"}
      </span>
    </span>
  );
}
