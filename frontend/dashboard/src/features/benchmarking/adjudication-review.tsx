/**
 * The adjudication page's right pane: the item under review (the extraction's cohorts and tagged lines, the
 * summary's numbered statements, or a decision task's verdict table) and one card per unit with the two blinded
 * judges' answers, the choices and the reveal after deciding.
 */
import { cn } from "cn";
import type { ReactNode, Ref } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

import type { Highlight } from "./adjudication-source";
import { stamp, words } from "./format";
import type { AdjudicationItem, JudgePanel, Question, Reference, UnitView } from "./types";

/** The journal's "2026-11-21T09:48:26-0500" as a local date and time. */
function decidedAt(at: string): string {
  return stamp(at.replace(/([+-]\d\d)(\d\d)$/, "$1:$2")) || at;
}

type OnQuote = (h: Omit<Highlight, "nonce">) => void;

// --------------------------------------------------------------------------- small pieces

function Answer({
  value,
  question,
}: {
  value: string | null | undefined;
  question?: Question | null;
}) {
  if (value === null || value === undefined) return <span className="text-gray-500">—</span>;
  return (
    <span>
      <span
        className={cn(
          "font-semibold",
          value === "yes" ? "text-green-700" : value === "no" ? "text-red-700" : "text-gray-900",
        )}
      >
        {words(value)}
      </span>
      {question?.answers[value] ? (
        <span className="text-gray-600"> — {question.answers[value]}</span>
      ) : null}
    </span>
  );
}

function Verdict({ value, extra }: { value: string | null | undefined; extra?: string }) {
  if (value === null || value === undefined) {
    return <span className="text-gray-500 italic">no answer</span>;
  }
  return (
    <span className="whitespace-nowrap">
      <span
        className={cn(
          "font-semibold",
          value === "yes" ? "text-green-700" : value === "no" ? "text-red-700" : "text-gray-900",
        )}
      >
        {words(value)}
      </span>
      {extra ? <span className="text-gray-600 tabular-nums">{extra}</span> : null}
    </span>
  );
}

function Hidden({ short = false }: { short?: boolean }) {
  return (
    <span className="text-gray-500 italic" title="Hidden until you decide">
      {short ? "hidden" : "hidden until you decide"}
    </span>
  );
}

/** Judge 1, Judge 2, the teacher, the decision model and the user's answer to one decision question. */
function verdicts(u: UnitView, short = false): [string, ReactNode | null][] {
  const ref = u.reference;
  const p = ref?.model.p_yes;
  return [
    ["Judge 1", u.judges[0] ? <Verdict value={u.judges[0].compared} /> : null],
    ["Judge 2", u.judges[1] ? <Verdict value={u.judges[1].compared} /> : null],
    ["Teacher (Nemotron)", ref ? <Verdict value={ref.teacher.answer} /> : null],
    [
      "Decision model",
      ref ? (
        <Verdict
          value={ref.model.answer}
          extra={p !== null && p !== undefined ? ` (P=${p.toFixed(2)})` : undefined}
        />
      ) : null,
    ],
    [
      "You",
      u.decision ? (
        <Verdict value={u.decision.value} />
      ) : (
        <span className="text-gray-500 italic">{short ? "–" : "not decided"}</span>
      ),
    ],
  ];
}

const BADGE: Record<string, string> = {
  disputed: "bg-red-50 text-red-700",
  single: "bg-amber-50 text-amber-800",
  audit: "bg-blue-50 text-brand-blue",
  agreed: "bg-green-50 text-green-700",
};

function Quote({
  text,
  date,
  grounded,
  onQuote,
}: {
  text: string;
  date?: string | null;
  grounded?: boolean;
  onQuote: OnQuote;
}) {
  return (
    <button
      type="button"
      title="Show in the source"
      onClick={(e) => {
        e.stopPropagation();
        onQuote({ quote: text, date: date ?? null });
      }}
      className={cn(
        "my-1 block w-full border-l-[3px] px-2 py-0.5 text-left text-[13px] whitespace-pre-wrap",
        grounded === false
          ? "border-red-500 bg-red-50 text-red-900"
          : "border-brand-blue bg-blue-50/60 hover:bg-blue-50",
      )}
    >
      {date ? <span className="text-gray-600">{date}: </span> : null}“{text}”
      {grounded === false ? (
        <span className="ml-1 text-red-700">(not found in the source)</span>
      ) : null}
    </button>
  );
}

function str(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(str).join("; ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value as string | number | boolean);
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((v): v is Record<string, unknown> => !!v && typeof v === "object")
    : [];
}

function Evidence({ list, onQuote }: { list: unknown; onQuote: OnQuote }) {
  return (
    <>
      {records(list).map((e, i) => (
        <Quote
          key={i}
          text={str(e.quote)}
          date={typeof e.date === "string" ? e.date : null}
          grounded={typeof e.grounded === "boolean" ? e.grounded : undefined}
          onQuote={onQuote}
        />
      ))}
    </>
  );
}

const SKIP = new Set(["line", "n", "cohort"]);

/** An extraction or summary judge's detail: its fields, its evidence quotes, its missing criteria or omissions. */
function Detail({ detail, onQuote }: { detail: Record<string, unknown> | null; onQuote: OnQuote }) {
  if (!detail) {
    return (
      <p className="text-gray-500 italic">no answer (refused, truncated, invalid or not run)</p>
    );
  }
  const rows: ReactNode[] = [];
  for (const [k, v] of Object.entries(detail)) {
    if (SKIP.has(k) || v === "" || v === null || (Array.isArray(v) && !v.length)) continue;
    if (k === "evidence") {
      rows.push(<Evidence key={k} list={v} onQuote={onQuote} />);
    } else if (k === "missing" || k === "omissions") {
      records(v).forEach((m, i) => {
        rows.push(
          <div key={`${k}${String(i)}`} className="mt-1">
            <span className="text-gray-500">
              {str(m.severity)}
              {m.section ? ` · ${words(str(m.section))}` : ""}:{" "}
            </span>
            {m.description ? str(m.description) : null}
            {typeof m.listing_text === "string" ? (
              <Quote
                text={m.listing_text}
                grounded={typeof m.grounded === "boolean" ? m.grounded : undefined}
                onQuote={onQuote}
              />
            ) : null}
            <Evidence list={m.evidence} onQuote={onQuote} />
          </div>,
        );
      });
    } else {
      rows.push(
        <div key={k}>
          <span className="text-gray-500">{words(k)}: </span>
          <span className="whitespace-pre-wrap">{str(v)}</span>
        </div>,
      );
    }
  }
  return <>{rows}</>;
}

function StatusLine({
  field,
  value,
  question,
}: {
  field: string | null;
  value: unknown;
  question: Question;
}) {
  const text = str(value);
  return (
    <div>
      <span className="text-gray-500">{words(field ?? "status")}: </span>
      <span className="font-semibold">{words(text)}</span>
      {question.statuses[text] ? (
        <span className="text-gray-600"> — {question.statuses[text]}</span>
      ) : null}
    </div>
  );
}

function DecisionDetail({
  detail,
  question,
}: {
  detail: Record<string, unknown> | null;
  question: Question;
}) {
  if (!detail) {
    return (
      <p className="text-gray-500 italic">no answer (refused, truncated, invalid or not run)</p>
    );
  }
  const field = question.status_field;
  return (
    <>
      {field && detail[field] !== undefined ? (
        <StatusLine field={field} value={detail[field]} question={question} />
      ) : null}
      {detail.reason ? (
        <div>
          <span className="text-gray-500">reason: </span>
          <span className="whitespace-pre-wrap">{str(detail.reason)}</span>
        </div>
      ) : null}
    </>
  );
}

function ReferenceBox({
  reference,
  question,
}: {
  reference: Reference;
  question: Question | null;
}) {
  const q = question ?? { text: "", answers: {}, status_field: null, statuses: {}, note: "" };
  const t = reference.teacher;
  const m = reference.model;
  return (
    <div className="mt-2 divide-y rounded-md border text-[13px]">
      <div className="px-3 py-1.5">
        <span className="font-semibold">Teacher (Nemotron 3.5 Super, the training label): </span>
        <Answer value={t.answer} question={q} />
        {t.raw ? (
          <span className="text-gray-600">
            {" "}
            (it answered "{t.raw}" to "Is this patient clearly excluded?")
          </span>
        ) : null}
        {t.status ? <StatusLine field={q.status_field} value={t.status} question={q} /> : null}
        {t.reason ? (
          <div>
            <span className="text-gray-500">reason: </span>
            <span className="whitespace-pre-wrap">{t.reason}</span>
          </div>
        ) : null}
      </div>
      <div className="px-3 py-1.5">
        <span className="font-semibold">Decision model (fine-tuned d1-2B): </span>
        {m.p_yes === null ? (
          "no score"
        ) : (
          <>
            <span className="tabular-nums">P(yes) = {m.p_yes.toFixed(3)} → </span>
            <Answer value={m.answer} question={q} />
            <span className="text-gray-600"> (yes when P(yes) ≥ {m.threshold})</span>
          </>
        )}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- the item

export function ItemHeader({ item }: { item: AdjudicationItem }) {
  const v = item.view;
  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-white p-3 text-sm">
      <p className="font-semibold break-words">{v.title}</p>
      {v.kind === "extraction" ? (
        <>
          <p className="text-gray-600">
            Screen: <span className="font-medium text-gray-900">{v.screen}</span>; cohort names:{" "}
            {v.cohort_names.length ? v.cohort_names.join(" | ") : "none"}
          </p>
          {v.cohorts.map((c) => (
            <div key={c.number}>
              <p className="font-medium">
                Cohort {c.number}: {c.name}
                {(c.merged_from_names ?? []).length > 1 ? (
                  <span className="font-normal text-gray-600">
                    {" "}
                    (merged from {(c.merged_from_names ?? []).join(" | ")})
                  </span>
                ) : null}
              </p>
              <div className="mt-0.5 flex flex-col gap-0.5 text-[13px]">
                {c.items.map((line, i) => (
                  <div key={line.number}>
                    {line.heading && line.heading !== c.items[i - 1]?.heading ? (
                      <p className="mt-1 text-xs text-gray-500">{line.heading}</p>
                    ) : null}
                    <p>
                      <span className="text-gray-500 tabular-nums">[{line.number}]</span>{" "}
                      <span className="text-gray-500">
                        ({line.criterion_type}; {line.boilerplate ? "boilerplate" : "key"})
                      </span>{" "}
                      {line.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </>
      ) : v.kind === "summary" ? (
        <div className="flex flex-col gap-0.5 text-[13px]">
          {v.statements.map((s, i) => (
            <div key={s.n}>
              {s.section !== v.statements[i - 1]?.section ? (
                <p className="mt-1 text-xs text-gray-500">{s.section}</p>
              ) : null}
              <p>
                <span className="text-gray-500 tabular-nums">[{s.n}]</span> {s.text}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <>
          <p className="text-gray-600">
            The source pane shows the task exactly as the teacher and both judges received it.
          </p>
          {item.units.length ? <VerdictTable units={item.units} /> : null}
          <p className="text-xs text-gray-500">
            Yes and no are in the decision model's polarity; each card below says what they mean.
            The teacher's and the model's answers stay hidden until you decide a question, unless
            "Show teacher and model before deciding" is on.
          </p>
        </>
      )}
    </div>
  );
}

function VerdictTable({ units }: { units: UnitView[] }) {
  const first = units[0];
  if (!first) return null;
  return (
    <div className="overflow-x-auto">
      <table className="text-[13px]">
        <thead>
          <tr>
            <th className="border bg-gray-50 px-2 py-1 text-left font-semibold text-gray-500">
              Question
            </th>
            {verdicts(first).map(([who]) => (
              <th
                key={who}
                className="border bg-gray-50 px-2 py-1 text-left font-semibold whitespace-nowrap text-gray-500"
              >
                {who}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {units.map((u) => (
            <tr key={u.unit_id}>
              <td className="border px-2 py-1">{u.label}</td>
              {verdicts(u, true).map(([who, cell]) => (
                <td key={who} className="border px-2 py-1 whitespace-nowrap">
                  {cell ?? <Hidden short />}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --------------------------------------------------------------------------- one unit

export function UnitCard({
  unit,
  focused,
  busy,
  note,
  onNote,
  noteRef,
  cardRef,
  onFocus,
  onDecide,
  onQuote,
}: {
  unit: UnitView;
  focused: boolean;
  busy: boolean;
  note: string;
  onNote: (value: string) => void;
  noteRef: Ref<HTMLInputElement>;
  cardRef: Ref<HTMLDivElement>;
  onFocus: () => void;
  onDecide: (value: string | null, choice: "judge1" | "judge2" | "other" | "skip") => void;
  onQuote: OnQuote;
}) {
  const u = unit;
  const q = u.question;
  const reason = u.queue_reason ?? u.status;
  return (
    <div
      ref={cardRef}
      data-unit={u.unit_id}
      onClick={onFocus}
      className={cn(
        "flex flex-col gap-2 rounded-lg border bg-white p-3 text-sm",
        focused && "ring-2 ring-brand-blue",
        u.decision && !focused && "opacity-80",
      )}
    >
      <div className="font-semibold break-words whitespace-pre-wrap">
        {u.target}
        <Badge
          className={cn(
            "ml-2 align-middle font-medium",
            BADGE[reason] ?? "bg-gray-100 text-gray-700",
          )}
        >
          {words(reason)}
        </Badge>
        <Badge variant="outline" className="ml-1.5 align-middle font-medium">
          {u.label}
        </Badge>
      </div>
      {q ? (
        <>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
            {verdicts(u).map(([who, cell]) => (
              <span key={who}>
                <span className="text-gray-500">{who}: </span>
                {cell ?? <Hidden />}
              </span>
            ))}
          </div>
          <div className="rounded-md bg-gray-50 px-3 py-1.5 text-[13px]">
            <p>Question: {q.text}</p>
            <p className="text-gray-700">
              yes = {q.answers.yes}; no = {q.answers.no}
            </p>
            {q.note ? <p className="text-gray-500">{q.note}</p> : null}
          </div>
        </>
      ) : null}
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {u.judges.map((j) => (
          <JudgeBox key={j.slot} judge={j} question={q} onQuote={onQuote} />
        ))}
      </div>
      {u.reference ? <ReferenceBox reference={u.reference} question={q} /> : null}
      <div className="flex flex-wrap items-center gap-1.5">
        {u.choices.map((c, n) => (
          <Button
            key={c}
            size="sm"
            variant={u.decision?.value === c ? "default" : "outline"}
            disabled={busy}
            onClick={(e) => {
              e.stopPropagation();
              onDecide(c, "other");
            }}
          >
            {String.fromCharCode(97 + n)}) {words(c)}
          </Button>
        ))}
        <Input
          ref={noteRef}
          value={note}
          placeholder="note (optional)"
          aria-label="Note"
          className="h-8 min-w-40 flex-1"
          onChange={(e) => {
            onNote(e.target.value);
          }}
          onFocus={onFocus}
        />
        <Button
          size="sm"
          variant="ghost"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            onDecide(null, "skip");
          }}
        >
          s) skip
        </Button>
      </div>
      {u.decision ? (
        <div className="rounded-md bg-gray-50 px-3 py-1.5 text-[13px]">
          You decided: <span className="font-semibold">{words(u.decision.value)}</span> (
          {decidedAt(u.decision.at)}). Judge 1 = {u.reveal?.judges["1"] ?? "?"}; Judge 2 ={" "}
          {u.reveal?.judges["2"] ?? "?"}.
          {u.decision.stale ? (
            <p className="text-amber-800">
              The judges' answers changed after this decision; check it again.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function JudgeBox({
  judge,
  question,
  onQuote,
}: {
  judge: JudgePanel;
  question: Question | null;
  onQuote: OnQuote;
}) {
  return (
    <div className="min-w-0 rounded-md border px-3 py-2 text-[13px]">
      <div>
        <span className="text-gray-500">Judge {judge.slot}: </span>
        {question ? (
          <Answer value={judge.compared} question={question} />
        ) : (
          <>
            <span className="font-semibold">
              {judge.compared === null ? "—" : words(judge.compared)}
            </span>
            {judge.value !== null && judge.value !== judge.compared ? (
              <span className="text-gray-500"> ({words(judge.value)})</span>
            ) : null}
          </>
        )}
      </div>
      {question ? (
        <DecisionDetail detail={judge.detail} question={question} />
      ) : (
        <Detail detail={judge.detail} onQuote={onQuote} />
      )}
    </div>
  );
}
