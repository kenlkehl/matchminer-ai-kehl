import { cn } from "cn";
import { BotIcon, CircleAlertIcon, CircleCheckIcon, CircleHelpIcon } from "lucide-react";
import { useState } from "react";

import { JsonDialog } from "@/components/json-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  estimateText,
  type LLMBoilerplate,
  type LLMBoilerplateAnswer,
  type LLMMatchAnswer,
  type LLMPairResult,
} from "@/lib/llm-decisions";

import type { LLMScoring } from "./use-llm-scoring";

/** The five match questions in the teacher prompt's order, as the dialog labels them. */
const LABELS: Record<string, string> = {
  reasonableness: "Reasonable candidate (no clear failure of the key criteria)",
  cancer_type: "Cancer type: required by the cohort and met",
  burden: "Disease extent: required by the cohort and met",
  treatment_history: "Treatment history: required by the cohort and met",
  biomarker: "Biomarker: required by the cohort and met (2 points)",
};
const REQUIREMENTS: [string, string][] = [
  ["cancer_type", "Cancer type"],
  ["burden", "Disease extent"],
  ["treatment_history", "Treatment history"],
  ["biomarker", "Biomarker"],
];
const TONES = {
  good: "bg-green-100 text-green-800",
  poor: "bg-gray-100 text-gray-600",
  none: "bg-amber-100 text-amber-900",
};

export type LLMAnswer = LLMMatchAnswer | LLMBoilerplateAnswer;

/** The LLM's match answer: Possible or Unlikely from its reasonableness answer, and its 0-6 match score. A
 * click opens the answers, the reasoning and the raw output (without `onOpen`, a plain badge). */
export function LLMMatchBadge({
  answer,
  scoring,
  onOpen,
}: {
  answer: LLMMatchAnswer | null | undefined;
  scoring: boolean;
  onOpen?: (answer: LLMAnswer) => void;
}) {
  if (!answer) {
    return scoring ? (
      <span className="flex items-center gap-1.5 text-xs text-gray-500">
        <Spinner className="size-3.5" /> Asking the LLM…
      </span>
    ) : (
      <span className="text-xs text-gray-400">Not scored by the LLM</span>
    );
  }
  const tone = answer.score === null ? "none" : answer.reasonable ? "good" : "poor";
  const text =
    answer.score === null
      ? "LLM: no valid answer"
      : answer.reasonable
        ? "Possible match (LLM)"
        : "Unlikely match (LLM)";
  const badge = (
    <Badge className={cn("h-6 gap-1.5 px-2.5 text-xs", onOpen && "cursor-pointer", TONES[tone])}>
      <BotIcon />
      {text}
      {answer.score !== null ? (
        <span className="font-semibold tabular-nums">
          {answer.score}/{answer.max}
        </span>
      ) : null}
    </Badge>
  );
  if (!onOpen) return badge;
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            className="inline-flex"
            onClick={() => {
              onOpen(answer);
            }}
          />
        }
      >
        {badge}
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        {answer.connection.label}: the 0–6 match score from its five yes/no answers (not a
        probability). Click for its answers, reasoning and raw output.
      </TooltipContent>
    </Tooltip>
  );
}

/** The requirements the LLM found the cohort to have and the patient to meet. */
export function LLMRequirements({ answer }: { answer: LLMMatchAnswer | null | undefined }) {
  const met = REQUIREMENTS.filter(([key]) => answer?.answers?.[key]?.answer === "yes");
  if (!answer?.reasonable || !met.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="text-gray-500">Matches based on (LLM):</span>
      {met.map(([key, label]) => (
        <span
          key={key}
          title={answer.answers?.[key]?.reason}
          className="inline-flex items-center gap-1 rounded bg-green-50 px-1.5 py-0.5 text-green-800"
        >
          <CircleCheckIcon className="size-3" />
          {label}
        </span>
      ))}
    </div>
  );
}

/** The LLM's common-exclusion answer, with a link to its reasoning. */
export function LLMExclusionLine({
  answer,
  scoring,
  onOpen,
}: {
  answer: LLMBoilerplate | null | undefined;
  scoring: boolean;
  onOpen: (answer: LLMAnswer) => void;
}) {
  if (!answer) {
    return scoring ? (
      <p className="flex items-center gap-1.5 text-sm text-gray-500">
        <Spinner className="size-3.5" /> Asking the LLM about common exclusions…
      </p>
    ) : null;
  }
  if ("skipped" in answer) return null;
  const link = (
    <button
      type="button"
      className="shrink-0 link text-xs"
      onClick={() => {
        onOpen(answer);
      }}
    >
      reasoning
    </button>
  );
  if (answer.excluded === null) {
    return (
      <p className="flex items-center gap-1.5 text-sm text-gray-600">
        <CircleHelpIcon className="size-4 shrink-0" />
        LLM: no valid answer on common exclusions. {link}
      </p>
    );
  }
  return answer.excluded ? (
    <p className="flex items-center gap-1.5 text-sm text-amber-700">
      <CircleAlertIcon className="size-4 shrink-0" />
      LLM: the patient may be clearly excluded by a common exclusion criterion. {link}
    </p>
  ) : (
    <p className="flex items-center gap-1.5 text-sm text-green-700">
      <CircleCheckIcon className="size-4 shrink-0" />
      LLM: not clearly excluded by the common exclusion criteria. {link}
    </p>
  );
}

export function llmLikelyExcluded(result: LLMPairResult | undefined): boolean {
  const b = result?.boilerplate;
  return Boolean(b && !("skipped" in b) && b.excluded === true);
}

/** Rows the LLM scored first, by its 0-6 match score then similarity; the rest by similarity. */
export function byLLMScore<T extends { id: string; cosine: number }>(
  rows: T[],
  results: Record<string, LLMPairResult>,
): T[] {
  const score = (id: string) => results[id]?.match?.score ?? null;
  return [...rows].sort((a, b) => {
    const sa = score(a.id);
    const sb = score(b.id);
    if (sa !== null && sb !== null && sa !== sb) return sb - sa;
    if (sa !== null && sb === null) return -1;
    if (sb !== null && sa === null) return 1;
    return b.cosine - a.cosine;
  });
}

/** A run's state for a list: its progress with the estimate, a large run waiting for confirmation, errors. */
export function LLMRunStatus({ scoring, noun }: { scoring: LLMScoring; noun: string }) {
  if (scoring.error)
    return <span className="text-sm text-red-700">LLM scoring failed: {scoring.error}</span>;
  if (scoring.pending) {
    const e = scoring.pending.estimate;
    return (
      <span className="flex flex-wrap items-center gap-2 rounded-md bg-amber-50 px-2 py-1 text-sm text-amber-950">
        Scoring {scoring.pending.ids.length} {noun} sends {estimateText(e)}.
        <Button size="xs" onClick={scoring.confirm}>
          Score them
        </Button>
        <Button size="xs" variant="outline" onClick={scoring.dismiss}>
          Not now
        </Button>
      </span>
    );
  }
  if (!scoring.progress) return null;
  return (
    <span className="flex items-center gap-1.5 text-sm text-gray-500">
      <Spinner className="size-3.5" />
      Asking the LLM about {scoring.running.length} {noun}
      {scoring.estimate ? ` (${estimateText(scoring.estimate)})` : ""}
      {scoring.progress.total
        ? ` · ${String(Math.round((100 * scoring.progress.done) / scoring.progress.total))}%`
        : ""}
    </span>
  );
}

/** How a list ranks under the LLM, in one line. */
export function LLMRankingNote() {
  return (
    <p className="text-xs text-gray-500">
      Ranked by the LLM's 0–6 match score (its five yes/no answers, not a probability), then
      TrialSpace similarity; results the LLM has not scored follow by similarity. Decision-model
      scores are not shown or used while the LLM is selected.
    </p>
  );
}

/**
 * One LLM answer in full: the parsed answer, the backend's reasoning (or a plain statement that it returned
 * none), the raw output, and the whole entry as JSON (the app's JSON dialog). "Ask again" re-asks the pair.
 */
export function LLMAnswerDialog({
  answer,
  onClose,
  onRerun,
}: {
  answer: LLMAnswer;
  onClose: () => void;
  onRerun?: () => void;
}) {
  const [json, setJson] = useState(false);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4 auto-rows-min">
        <DialogHeader>
          <DialogTitle>
            {answer.family === "match" ? "LLM match answers" : "LLM common-exclusion check"}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            {answer.connection.label} · {answer.scale} ·{" "}
            {new Date(answer.checked_at * 1000).toLocaleString()} · parse: {answer.parse_status}
          </DialogDescription>
        </DialogHeader>
        {answer.family === "match" ? (
          <MatchAnswers answer={answer} />
        ) : (
          <ExclusionAnswer answer={answer} />
        )}
        <section className="flex flex-col gap-1">
          <h3 className="text-xs font-medium tracking-wide text-gray-500 uppercase">Reasoning</h3>
          {answer.reasoning.trim() ? (
            <p className="max-h-64 overflow-y-auto rounded-md border border-gray-200 bg-gray-50 p-2 text-xs whitespace-pre-wrap text-gray-700">
              {answer.reasoning}
            </p>
          ) : (
            <p className="text-sm text-gray-600">
              This backend returned no reasoning text for this answer. {answer.reasoning_note}
            </p>
          )}
        </section>
        <section className="flex flex-col gap-1">
          <h3 className="text-xs font-medium tracking-wide text-gray-500 uppercase">Raw output</h3>
          <pre className="max-h-64 overflow-auto rounded-md border border-gray-200 bg-gray-50 p-2 font-mono text-xs whitespace-pre-wrap text-gray-800">
            {answer.output || "(empty)"}
          </pre>
        </section>
        <DialogFooter className="pt-1">
          {onRerun ? (
            <Button
              variant="outline"
              onClick={() => {
                onRerun();
                onClose();
              }}
            >
              Ask again
            </Button>
          ) : null}
          <Button
            variant="outline"
            onClick={() => {
              setJson(true);
            }}
          >
            Raw JSON
          </Button>
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
        {json ? (
          <JsonDialog
            view={{
              title: "LLM answer (JSON)",
              name: `llm-${answer.family}-answer.json`,
              value: answer,
              description:
                "The dashboard's record of the answer: the parsed answer, the raw output and the reasoning.",
            }}
            onClose={() => {
              setJson(false);
            }}
          />
        ) : null}
      </DialogContent>
    </Dialog>
  );
}

function MatchAnswers({ answer }: { answer: LLMMatchAnswer }) {
  return (
    <section className="flex flex-col gap-2">
      <p className="text-sm text-gray-700">
        {answer.score === null ? (
          <>The answer did not pass the package's validation, so it has no score.</>
        ) : (
          <>
            Match score <strong>{answer.score}</strong> of {answer.max}:{" "}
            {answer.reasonable
              ? "a reasonable candidate, plus a point for each requirement the cohort has and the patient meets (biomarker 2)."
              : "not a reasonable candidate (the score is 0 whatever the other answers)."}
          </>
        )}
      </p>
      {answer.answers ? (
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-gray-500">
            <tr>
              <th className="py-1 pr-3 font-medium">Question</th>
              <th className="py-1 pr-3 font-medium">Answer</th>
              <th className="py-1 font-medium">Reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {(answer.labels.length ? answer.labels : Object.keys(answer.answers)).map((label) => {
              const a = answer.answers?.[label];
              if (!a) return null;
              return (
                <tr key={label} className="align-top">
                  <td className="py-1.5 pr-3 text-gray-700">{LABELS[label] ?? label}</td>
                  <td className="py-1.5 pr-3 whitespace-nowrap">
                    <span className={a.answer === "yes" ? "text-green-700" : "text-gray-700"}>
                      {a.answer}
                    </span>{" "}
                    <span className="text-xs text-gray-500">
                      ({a.evidence_status.replace(/_/g, " ")})
                    </span>
                  </td>
                  <td className="py-1.5 text-gray-700">{a.reason}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}

function ExclusionAnswer({ answer }: { answer: LLMBoilerplateAnswer }) {
  return (
    <p className="text-sm text-gray-700">
      “Is this patient clearly excluded from this cohort?” (common criteria):{" "}
      <strong>{answer.answer ?? "no valid answer"}</strong>. The raw output below holds the LLM's
      criterion-by-criterion review.
    </p>
  );
}

/** A pair's answers in a dialog: the decision model's or the LLM's (with a button to ask the LLM). */
export function LLMPairAnswers({
  result,
  scoring,
  onScore,
  onOpen,
}: {
  result: LLMPairResult | undefined;
  scoring: boolean;
  onScore?: () => void;
  onOpen: (answer: LLMAnswer) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <LLMMatchBadge answer={result?.match} scoring={scoring} onOpen={onOpen} />
        {!result?.match && !scoring && onScore ? (
          <Button size="xs" variant="outline" onClick={onScore}>
            <BotIcon />
            Score this pair with the LLM
          </Button>
        ) : null}
      </div>
      <LLMRequirements answer={result?.match} />
      <LLMExclusionLine answer={result?.boilerplate} scoring={scoring} onOpen={onOpen} />
    </div>
  );
}
