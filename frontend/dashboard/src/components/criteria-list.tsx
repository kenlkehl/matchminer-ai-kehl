import { cn } from "cn";
import { CircleCheckIcon, CircleHelpIcon, CircleXIcon } from "lucide-react";
import { useState } from "react";

import { JsonDialog } from "@/components/json-dialog";
import type { CriteriaItem, CriterionLine } from "@/lib/api";
import { percent } from "@/lib/format";
import type { LLMCriterionLine } from "@/lib/llm-decisions";
import { STATUS_LABELS, type StructuredLine } from "@/lib/records";

/** Whether the decision model and the LLM both answered a line, differently. */
export function disagree(check: CriterionLine | undefined, llm: LLMCriterionLine | undefined) {
  return Boolean(check && llm && llm.fails !== null && check.fails !== llm.fails);
}

/**
 * A cohort's criteria lines under Inclusion and Exclusion, common ("boilerplate") lines marked, and,
 * when a per-criterion check ran, each line's answer: a clear failure, or no clear failure. With
 * `llm`, the LLM's answer and reason follow the decision model's (with its reasoning and raw output on
 * request), and a line they answer differently is highlighted. With `records`, the structured-record screen's status of each line follows, with the
 * extracted items it rests on.
 */
export function CriteriaList({
  items,
  checks,
  llm,
  records,
  showCommon = true,
}: {
  items: CriteriaItem[];
  checks?: CriterionLine[] | null;
  llm?: LLMCriterionLine[] | null;
  records?: StructuredLine[] | null;
  showCommon?: boolean;
}) {
  const byNumber = new Map((checks ?? []).map((c) => [c.number, c]));
  const llmByNumber = new Map((llm ?? []).map((c) => [c.number, c]));
  const recordsByNumber = new Map((records ?? []).map((c) => [c.number, c]));
  const sections = (["inclusion", "exclusion"] as const).map((kind) => ({
    kind,
    items: items.filter(
      (i) =>
        (i.criterion_type ??
          (i.heading.toLowerCase().startsWith("excl") ? "exclusion" : "inclusion")) === kind,
    ),
  }));
  return (
    <div className="flex flex-col gap-4">
      {sections.map((section) =>
        section.items.length ? (
          <section key={section.kind}>
            <h4 className="mb-1 text-sm font-bold capitalize">{section.kind} criteria</h4>
            <ol className="flex flex-col divide-y divide-gray-100">
              {section.items
                .filter((i) => showCommon || !i.boilerplate)
                .map((item) => {
                  const check = byNumber.get(item.number);
                  const answer = llmByNumber.get(item.number);
                  const record = recordsByNumber.get(item.number);
                  const differ = disagree(check, answer);
                  return (
                    <li
                      key={item.number}
                      className={cn(
                        "flex items-start gap-2 py-1.5 text-sm",
                        differ && "-mx-2 rounded bg-amber-50 px-2 ring-1 ring-amber-200",
                      )}
                    >
                      {checks ? <CheckMark check={check} /> : null}
                      {llm ? <LLMMark line={answer} /> : null}
                      {records ? <RecordMark line={record} /> : null}
                      <div className="flex flex-1 flex-col gap-0.5">
                        <span
                          className={cn(
                            "whitespace-pre-wrap",
                            item.boilerplate ? "text-gray-500" : "text-gray-800",
                            (check?.fails ?? answer?.fails ?? record?.clearly_fails) &&
                              "font-medium text-red-800",
                          )}
                        >
                          {item.text}
                        </span>
                        {answer ? <LLMReason line={answer} differ={differ} /> : null}
                        {record ? <RecordReason line={record} decision={check} /> : null}
                      </div>
                      {item.boilerplate ? (
                        <span className="shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-500">
                          Common
                        </span>
                      ) : null}
                    </li>
                  );
                })}
            </ol>
          </section>
        ) : null,
      )}
    </div>
  );
}

function LLMMark({ line }: { line: LLMCriterionLine | undefined }) {
  if (!line) return <span className="mt-0.5 size-4 shrink-0" />;
  const Icon = line.fails === null ? CircleHelpIcon : line.fails ? CircleXIcon : CircleCheckIcon;
  const tone =
    line.fails === null ? "text-gray-400" : line.fails ? "text-red-600" : "text-green-600";
  return (
    <span
      title={
        line.fails === null
          ? `LLM: no valid answer (${line.parse_status})`
          : `LLM: ${line.fails ? "clearly fails" : "no clear failure"} (${line.status ?? ""})`
      }
      className={cn("mt-0.5 shrink-0 rounded-full ring-1 ring-offset-1", tone, "ring-current/30")}
    >
      <Icon className="size-4" />
    </span>
  );
}

function LLMReason({ line, differ }: { line: LLMCriterionLine; differ: boolean }) {
  const [raw, setRaw] = useState(false);
  return (
    <div className="text-xs text-gray-600">
      <span className="font-medium">
        LLM{line.status ? ` (${line.status.replace(/_/g, " ")})` : ""}:
      </span>{" "}
      {line.answer === null ? `no valid answer: ${line.parse_status}` : line.reason}
      {differ ? (
        <span className="ml-1 rounded bg-amber-200 px-1 py-0.5 text-[11px] font-medium text-amber-900">
          differs from the decision model
        </span>
      ) : null}
      <button
        type="button"
        className="ml-1.5 text-gray-400 hover:text-gray-600 hover:underline"
        onClick={() => {
          setRaw(true);
        }}
      >
        raw output
      </button>
      {line.reasoning ? (
        <details className="mt-0.5">
          <summary className="cursor-pointer text-gray-400 hover:text-gray-600">Reasoning</summary>
          <p className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap text-gray-500">
            {line.reasoning}
          </p>
        </details>
      ) : (
        <span className="ml-1.5 text-gray-400">· no reasoning text returned</span>
      )}
      {raw ? (
        <JsonDialog
          view={{
            title: `LLM answer, line ${String(line.number)}`,
            name: `llm-criterion-${String(line.number)}.json`,
            value: line,
            description: "The LLM's parsed answer, its raw output (output) and its reasoning.",
          }}
          onClose={() => {
            setRaw(false);
          }}
        />
      ) : null}
    </div>
  );
}

function CheckMark({ check }: { check: CriterionLine | undefined }) {
  if (!check) return <span className="mt-0.5 size-4 shrink-0" />;
  return check.fails ? (
    <span
      title={`Clearly fails (P(no clear failure) ${percent(check.p)})`}
      className="mt-0.5 shrink-0 text-red-600"
    >
      <CircleXIcon className="size-4" />
    </span>
  ) : (
    <span
      title={`No clear failure (${percent(check.p)})`}
      className="mt-0.5 shrink-0 text-green-600"
    >
      <CircleCheckIcon className="size-4" />
    </span>
  );
}

const RECORD_TONES: Record<string, string> = {
  clearly_fails: "text-red-600",
  meets: "text-green-600",
  conflicting: "text-amber-600",
  uncertain: "text-amber-500",
};

/** The structured-record screen's status of a line: a square mark (the decision model's is round). */
function RecordMark({ line }: { line: StructuredLine | undefined }) {
  if (!line) return <span className="mt-0.5 size-4 shrink-0" />;
  const Icon = line.clearly_fails
    ? CircleXIcon
    : line.status === "meets"
      ? CircleCheckIcon
      : CircleHelpIcon;
  return (
    <span
      title={`Records: ${STATUS_LABELS[line.status]}`}
      className={cn(
        "mt-0.5 shrink-0 rounded-sm bg-gray-50 ring-1 ring-gray-300",
        RECORD_TONES[line.status] ?? "text-gray-400",
      )}
    >
      <Icon className="size-4" />
    </span>
  );
}

function RecordReason({
  line,
  decision,
}: {
  line: StructuredLine;
  decision: CriterionLine | undefined;
}) {
  const differ = Boolean(
    decision &&
    line.clearly_fails !== decision.fails &&
    line.status !== "not_checkable" &&
    line.status !== "no_records" &&
    line.status !== "query_unavailable",
  );
  if (line.status === "not_checkable" || line.status === "no_records") {
    return <div className="text-xs text-gray-400">Records: {line.bottom_line}</div>;
  }
  return (
    <div className="text-xs text-gray-600">
      <span className="font-medium">Records ({STATUS_LABELS[line.status].toLowerCase()}):</span>{" "}
      {line.bottom_line}
      {differ ? (
        <span className="ml-1 rounded bg-amber-200 px-1 py-0.5 text-[11px] font-medium text-amber-900">
          differs from the decision model
        </span>
      ) : null}
      {line.evidence.length ? (
        <details className="mt-0.5">
          <summary className="cursor-pointer text-gray-400 hover:text-gray-600">
            {line.evidence.length} {line.evidence.length === 1 ? "quote" : "quotes"}
          </summary>
          <ul className="mt-1 flex flex-col gap-1">
            {line.evidence.map((e, i) => (
              <li key={i} className="border-l-2 border-gray-200 pl-2">
                “{e.quote}”{" "}
                <span className="text-gray-400">
                  {e.note_number ? `Note ${String(e.note_number)} · ` : ""}
                  {e.note_date?.slice(0, 10) ?? "undated"}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {line.limitations.length ? (
        <span className="block text-gray-400">{line.limitations.join(" ")}</span>
      ) : null}
    </div>
  );
}
