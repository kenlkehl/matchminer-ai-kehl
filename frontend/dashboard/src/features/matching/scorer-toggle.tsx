import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "cn";

import { SCORER_LABELS, type Scorer, setScorer, useScorer } from "@/lib/llm-decisions";
import { llmOptions } from "@/lib/queries";

const SCORERS: Scorer[] = ["decision_model", "llm"];

/**
 * "Scores from: Decision model | LLM", the same control in every view that scores pairs. The choice is the
 * browser session's (changing it here changes it in every view); the LLM is the session's connection for the
 * LLM decision checks task (Data and Models).
 */
export function ScorerToggle({ className }: { className?: string }) {
  const scorer = useScorer();
  const status = useQuery(llmOptions()).data?.decisions;
  const reason = !status ? "Checking the LLM connection…" : status.available ? null : status.reason;
  return (
    <div className={cn("inline-flex flex-wrap items-center gap-2 text-xs", className)}>
      <span className="text-gray-500" id="scorer-label">
        Scores from
      </span>
      <div
        role="radiogroup"
        aria-labelledby="scorer-label"
        className="inline-flex rounded-md border border-gray-300 bg-white p-0.5"
      >
        {SCORERS.map((value) => {
          const disabled = value === "llm" && scorer !== "llm" && reason !== null;
          return (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={scorer === value}
              disabled={disabled}
              title={disabled ? (reason ?? undefined) : undefined}
              className={cn(
                "rounded px-2 py-0.5 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                scorer === value ? "bg-brand-blue text-white" : "text-gray-600 hover:bg-gray-100",
              )}
              onClick={() => {
                setScorer(value);
              }}
            >
              {SCORER_LABELS[value]}
            </button>
          );
        })}
      </div>
      {scorer === "llm" && status ? (
        <span className="text-gray-500">
          {status.connection.label} ·{" "}
          <Link to="/status" className="link">
            change in Data and Models
          </Link>
        </span>
      ) : null}
      {scorer === "llm" && reason && status ? <span className="text-red-700">{reason}</span> : null}
    </div>
  );
}
