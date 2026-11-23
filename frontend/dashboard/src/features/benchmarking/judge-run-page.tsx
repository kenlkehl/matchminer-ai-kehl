import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getRouteApi, Link } from "@tanstack/react-router";
import { RefreshCwIcon, ScaleIcon } from "lucide-react";
import { Fragment, type ReactNode, useEffect, useRef, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, isActive, type Job, waitForJob } from "@/lib/api";

import { type Domain, IntervalAxis, IntervalMark, Meter } from "./charts";
import { BackToRuns, Section } from "./common";
import {
  componentLabel,
  count,
  duration,
  interval,
  judgeName,
  num,
  stamp,
  usd,
  words,
} from "./format";
import { adjudicationStateOptions, judgeOptions } from "./queries";
import type {
  AdjudicationState,
  CompareComponent,
  DecisionResult,
  JudgeComponentMetrics,
  JudgeRun,
  KindAgreement,
  Rate,
  RecomputeResult,
  Scored,
} from "./types";

const route = getRouteApi("/_app/benchmarking/judge/$runId/");

const COMPONENTS = ["extraction", "summary", "decision"];

/** The rubric modules' unit kinds (rubric_*.py KINDS labels). */
export const KIND_LABELS: Record<string, string> = {
  screen: "Cancer-treatment screen",
  cohort_set: "Cohort set (names and merges)",
  fidelity: "Criterion copied faithfully",
  boilerplate_tag: "Key/boilerplate tag",
  type_tag: "Inclusion/exclusion tag",
  missing: "Missing criteria",
  overall: "Overall",
  statement: "Statement",
  omission: "Omissions in a section",
  reasonableness: "Reasonableness",
  cancer_type: "Cancer type",
  burden: "Burden",
  treatment_history: "Treatment history",
  biomarker: "Biomarker",
  boilerplate: "Boilerplate (potentially eligible)",
  criterion: "Criterion (does not clearly fail)",
};

export function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? words(kind);
}

/** A dual-judge run: the judges' cost, their agreement, the adjudication queue and the results. */
export function JudgeRunPage() {
  const { runId } = route.useParams();
  const query = useQuery(judgeOptions(runId));
  const run = query.data;
  const state = useQuery(adjudicationStateOptions(runId, !!run?.compare));
  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-3">
        <BackToRuns />
        <PageHeader title={`LLM as judge: ${run?.name ?? ""}`}>
          <p>
            Two LLM judges rate the pipeline's outputs blind to each other. The reference for each
            unit is the judges' agreed answer or your adjudication; these are LLM judges' ratings,
            not a clinician's review. Rates are weighted by each item's sampling weight; 95%
            intervals resample items.
          </p>
          {run?.version ? <p>Build {run.version}.</p> : null}
        </PageHeader>
      </div>
      {query.isPending ? (
        <Spinner className="size-6" />
      ) : query.isError ? (
        <p className="text-sm text-red-700">{query.error.message}</p>
      ) : run ? (
        <>
          <Overview run={run} runId={runId} state={state.data ?? null} />
          <Components run={run} state={state.data ?? null} />
        </>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------- overview and recompute

function StatTile({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="flex min-w-36 flex-col gap-0.5 rounded-lg border px-4 py-3">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-2xl font-semibold">{value}</span>
      {note ? <span className="text-xs text-gray-500">{note}</span> : null}
    </div>
  );
}

function Overview({
  run,
  runId,
  state,
}: {
  run: JudgeRun;
  runId: string;
  state: AdjudicationState | null;
}) {
  const metrics = run.metrics;
  const queued = Object.values(state?.components ?? {}).reduce((s, c) => s + (c?.queued ?? 0), 0);
  const decided = Object.values(state?.components ?? {}).reduce((s, c) => s + (c?.decided ?? 0), 0);
  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {run.compare ? (
          <Button
            nativeButton={false}
            render={<Link to="/benchmarking/judge/$runId/adjudicate" params={{ runId }} />}
          >
            <ScaleIcon />
            Adjudicate
          </Button>
        ) : null}
        <Recompute run={run} runId={runId} />
      </div>
      <div className="flex flex-wrap gap-3">
        <StatTile
          label="Units"
          value={count(metrics?.units ?? run.compare?.units)}
          note={
            run.compare
              ? `${count(run.compare.queued)} queued for adjudication`
              : "not compared yet"
          }
        />
        <StatTile
          label="With a reference"
          value={count(metrics?.with_reference)}
          note="agreed or adjudicated"
        />
        <StatTile
          label="Adjudicated"
          value={state ? `${count(decided)} of ${count(queued)}` : count(run.adjudications)}
          note={state ? "queued units decided" : undefined}
        />
      </div>
      {metrics && metrics.adjudications !== run.adjudications ? (
        <p className="text-sm text-amber-800">
          The metrics count {count(metrics.adjudications)} adjudications and the run now has{" "}
          {count(run.adjudications)}: recompute them to include the latest decisions.
        </p>
      ) : null}
    </section>
  );
}

function Recompute({ run, runId }: { run: JudgeRun; runId: string }) {
  const client = useQueryClient();
  const [job, setJob] = useState<Job<RecomputeResult> | null>(null);
  const polling = useRef<string | null>(null);
  const latest = job ?? run.recompute.at(-1) ?? null;

  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["benchmarks"] });
  };
  const follow = async (id: string) => {
    if (polling.current === id) return;
    polling.current = id;
    try {
      await waitForJob<RecomputeResult>(id, setJob, 1000);
    } finally {
      polling.current = null;
      refresh();
    }
  };
  const start = useMutation({
    mutationFn: async () => {
      const started = await api.post<{ job: Job<RecomputeResult> }>(
        `/api/benchmarks/judge/${encodeURIComponent(runId)}/recompute`,
        { steps: ["compare", "metrics"] },
      );
      setJob(started.job);
      await follow(started.job.id);
    },
  });
  // A recompute started earlier (or from another page) is still running: follow it.
  const serverActive = run.recompute.findLast((j) => isActive(j));
  useEffect(() => {
    if (serverActive && !job) void follow(serverActive.id).catch(() => undefined);
  }, [serverActive?.id]);

  const active = isActive(latest) || start.isPending;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button
        variant="outline"
        disabled={active}
        onClick={() => {
          start.mutate();
        }}
      >
        {active ? <Spinner /> : <RefreshCwIcon />}
        Recompute comparison and metrics
      </Button>
      <span className="text-sm text-gray-600">
        {latest ? <JobLine job={latest} /> : null}
        {start.isError && !latest ? (
          <span className="text-red-700">{start.error.message}</span>
        ) : null}
      </span>
    </div>
  );
}

function JobLine({ job }: { job: Job<RecomputeResult> }) {
  if (isActive(job)) {
    return (
      <>
        {job.message || job.status}
        {job.total
          ? ` (step ${String(Math.min(job.done + 1, job.total))} of ${String(job.total)})`
          : ""}
      </>
    );
  }
  if (job.status === "error") return <span className="text-red-700">{job.error ?? "failed"}</span>;
  if (job.status === "cancelled") return <>Stopped.</>;
  const log = Object.entries(job.result ?? {}).flatMap(([step, r]) =>
    r.log.slice(-1).map((line) => `${step}: ${line}`),
  );
  return (
    <span title={log.join("\n")}>
      Recomputed {stamp(job.finished)}
      {job.seconds ? ` in ${duration(job.seconds)}` : ""}.
    </span>
  );
}

// --------------------------------------------------------------------------- one component

function judgesOf(run: JudgeRun, metrics: JudgeComponentMetrics | undefined): string[] {
  return (
    run.metrics?.judges ??
    run.compare?.judges ??
    (metrics ? Object.keys(metrics.judges) : Object.keys(run.responses))
  );
}

/** One component at a time, chosen with tabs (all three are long). */
function Components({ run, state }: { run: JudgeRun; state: AdjudicationState | null }) {
  const present = COMPONENTS.filter(
    (c) => run.components[c] || run.metrics?.components[c] || run.compare?.components[c],
  );
  const [chosen, setChosen] = useState<string | null>(null);
  const component = chosen && present.includes(chosen) ? chosen : present[0];
  if (!component) return <p className="text-sm text-gray-600">This run has no components yet.</p>;
  return (
    <div className="flex flex-col gap-4">
      {present.length > 1 ? (
        <Tabs
          value={component}
          onValueChange={(value) => {
            setChosen(String(value));
          }}
        >
          <TabsList>
            {present.map((c) => {
              const queue = state?.components[c];
              return (
                <TabsTrigger key={c} value={c} className="px-3">
                  {componentLabel(c)}
                  {queue ? (
                    <span className="text-xs text-gray-500 tabular-nums">
                      {queue.decided}/{queue.queued}
                    </span>
                  ) : null}
                </TabsTrigger>
              );
            })}
          </TabsList>
        </Tabs>
      ) : null}
      <ComponentSection component={component} run={run} state={state} />
    </div>
  );
}

function ComponentSection({
  component,
  run,
  state,
}: {
  component: string;
  run: JudgeRun;
  state: AdjudicationState | null;
}) {
  const metrics = run.metrics?.components[component];
  const compare = run.compare?.components[component];
  const build = run.components[component];
  const judges = judgesOf(run, metrics);
  const queue = state?.components[component];
  const items = metrics?.items ?? compare?.items ?? build?.items;
  return (
    <Section title={`${componentLabel(component)} (${count(items)} items)`}>
      {build ? (
        <p className="text-sm text-gray-600">
          Built {stamp(build.built_at)}: {count(build.requests)} requests,{" "}
          {count(build.request_chars)} characters (largest {count(build.max_request_chars)}).
        </p>
      ) : null}
      <JudgeTable
        component={component}
        run={run}
        judges={judges}
        metrics={metrics}
        compare={compare}
      />
      {compare ? <QueueSummary compare={compare} queue={queue} component={component} /> : null}
      {metrics ? (
        <>
          <AgreementTable agreement={metrics.agreement} />
          <JudgeAccuracyTable agreement={metrics.agreement} judges={judges} />
          {component === "decision" ? (
            <DecisionResults results={metrics.results} />
          ) : (
            <RateResults results={metrics.results} />
          )}
        </>
      ) : (
        <p className="text-sm text-gray-600">Metrics appear here once they have been computed.</p>
      )}
    </Section>
  );
}

function Subheading({ children }: { children: ReactNode }) {
  return <h3 className="mt-2 text-sm font-semibold text-gray-700">{children}</h3>;
}

function statusText(statuses: Record<string, number> | undefined): string {
  return Object.entries(statuses ?? {})
    .map(([s, n]) => `${words(s)} ${n.toLocaleString()}`)
    .join(", ");
}

function JudgeTable({
  component,
  run,
  judges,
  metrics,
  compare,
}: {
  component: string;
  run: JudgeRun;
  judges: string[];
  metrics: JudgeComponentMetrics | undefined;
  compare: CompareComponent | undefined;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Subheading>Judges</Subheading>
      <Table variant="list" className="[&_th]:whitespace-normal">
        <TableHeader>
          <TableRow>
            <TableHead>Judge</TableHead>
            <TableHead className="text-right">Answer rows</TableHead>
            <TableHead>Final statuses</TableHead>
            <TableHead className="text-right">Retried</TableHead>
            <TableHead className="text-right">Cost</TableHead>
            <TableHead className="text-right">Cost per item</TableHead>
            <TableHead className="text-right">Median seconds per request</TableHead>
            <TableHead className="text-right">Input tokens</TableHead>
            <TableHead className="text-right">Output tokens</TableHead>
            <TableHead className="text-right">Superseded spend</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {judges.map((judge) => {
            const j = metrics?.judges[judge];
            const answers = run.responses[judge]?.[component];
            return (
              <TableRow key={judge}>
                <TableCell>
                  <p className="font-medium whitespace-nowrap">{judgeName(judge)}</p>
                  <p className="font-mono text-xs text-gray-500">{judge}</p>
                </TableCell>
                <TableCell className="text-right tabular-nums">{count(answers)}</TableCell>
                {j && !j.ran ? (
                  <TableCell colSpan={7} className="text-gray-500">
                    not run
                  </TableCell>
                ) : (
                  <>
                    <TableCell>
                      {statusText(j?.final ?? compare?.judge_status[judge]) || "–"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{count(j?.retried)}</TableCell>
                    <TableCell className="text-right tabular-nums">{usd(j?.cost_usd)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {usd(j?.cost_per_item_usd, 3)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {num(j?.median_seconds, 1)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {count(j?.input_tokens)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {count(j?.output_tokens)}
                    </TableCell>
                  </>
                )}
                <TableCell className="text-right tabular-nums">
                  {usd(j?.superseded_cost_usd)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function QueueSummary({
  compare,
  queue,
  component,
}: {
  compare: CompareComponent;
  queue: { queued: number; decided: number } | undefined;
  component: string;
}) {
  const units = compare.units;
  const reasons = compare.queued;
  const queued = queue?.queued ?? Object.values(reasons).reduce((a, b) => a + (b ?? 0), 0);
  return (
    <div className="flex flex-col gap-2">
      <Subheading>Adjudication queue</Subheading>
      <div className="flex max-w-xl flex-col gap-1.5">
        {queue ? (
          <>
            <Meter
              value={queue.decided}
              total={queue.queued}
              label={`${componentLabel(component)}: queued units decided`}
            />
            <p className="text-sm text-gray-700">
              <span className="font-semibold tabular-nums">{count(queue.decided)}</span> of{" "}
              <span className="tabular-nums">{count(queued)}</span> queued units decided
              {compare.queued_items ? ` (in ${count(compare.queued_items)} items)` : ""}
            </p>
          </>
        ) : null}
        <p className="text-xs text-gray-600">
          Units: {count(units.agreed)} agreed, {count(units.disputed)} disputed,{" "}
          {count(units.single)} answered by one judge, {count(units.awaiting)} awaiting a judge,{" "}
          {count(units.missing)} unanswered. Queued: {count(reasons.disputed)} disputed,{" "}
          {count(reasons.single)} single, {count(reasons.audit)} audit samples of agreed units.
        </p>
      </div>
    </div>
  );
}

function AgreementTable({ agreement }: { agreement: Record<string, KindAgreement> }) {
  const kinds = Object.entries(agreement);
  if (!kinds.length) return null;
  return (
    <div className="flex flex-col gap-1">
      <Subheading>Agreement</Subheading>
      <Table variant="list">
        <TableHeader>
          <TableRow>
            <TableHead>Unit</TableHead>
            <TableHead className="text-right">Units</TableHead>
            <TableHead className="text-right">Agreed</TableHead>
            <TableHead className="text-right">Disputed</TableHead>
            <TableHead className="text-right">Single</TableHead>
            <TableHead className="text-right">Awaiting</TableHead>
            <TableHead className="text-right">Agreement</TableHead>
            <TableHead className="text-right">Cohen's kappa</TableHead>
            <TableHead className="text-right">Queued</TableHead>
            <TableHead className="text-right">Adjudicated</TableHead>
            <TableHead className="text-right">Pending</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {kinds.map(([kind, a]) => (
            <TableRow key={kind}>
              <TableCell className="font-medium whitespace-nowrap">{kindLabel(kind)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.units)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.agreed)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.disputed)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.single)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.awaiting)}</TableCell>
              <TableCell className="text-right tabular-nums">{num(a.agreement)}</TableCell>
              <TableCell className="text-right tabular-nums">{num(a.kappa)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.queued)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.adjudicated)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(a.pending)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function JudgeAccuracyTable({
  agreement,
  judges,
}: {
  agreement: Record<string, KindAgreement>;
  judges: string[];
}) {
  const kinds = Object.entries(agreement);
  if (!kinds.length || !judges.length) return null;
  const group = "border-l border-gray-200";
  return (
    <div className="flex flex-col gap-1">
      <Subheading>Each judge against the reference</Subheading>
      <p className="text-xs text-gray-600">
        Accuracy over every unit with a reference (agreed or adjudicated), and over adjudicated
        units only; each judge's answers before any adjudication. Audit: agreed units drawn for
        adjudication where both judges were wrong.
      </p>
      <Table variant="list">
        <TableHeader>
          <TableRow className="border-b-0">
            <TableHead rowSpan={2} className="align-bottom">
              Unit
            </TableHead>
            {judges.map((j) => (
              <TableHead
                key={j}
                colSpan={3}
                className={`${group} pb-0 text-center text-gray-700 normal-case`}
              >
                {judgeName(j)}
              </TableHead>
            ))}
            <TableHead rowSpan={2} className={`${group} text-right align-bottom`}>
              Audit: both wrong
            </TableHead>
          </TableRow>
          <TableRow>
            {judges.map((j) => (
              <Fragment key={j}>
                <TableHead className={`${group} text-right`}>Accuracy</TableHead>
                <TableHead className="text-right">Adjudicated</TableHead>
                <TableHead>Answers</TableHead>
              </Fragment>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {kinds.map(([kind, a]) => (
            <TableRow key={kind}>
              <TableCell className="font-medium whitespace-nowrap">{kindLabel(kind)}</TableCell>
              {judges.map((j) => (
                <Fragment key={j}>
                  <TableCell className={`${group} text-right tabular-nums`}>
                    <AccuracyValue value={a.judges[j]?.accuracy} n={a.judges[j]?.n} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    <AccuracyValue
                      value={a.judges[j]?.accuracy_adjudicated}
                      n={a.judges[j]?.n_adjudicated}
                    />
                  </TableCell>
                  <TableCell className="text-xs text-gray-600">
                    {Object.entries(a.distribution[j] ?? {})
                      .map(([v, n]) => `${words(v)} ${String(n)}`)
                      .join(", ") || "–"}
                  </TableCell>
                </Fragment>
              ))}
              <TableCell className={`${group} text-right tabular-nums`}>
                {count(a.audit.both_wrong)} of {count(a.audit.decided)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function AccuracyValue({ value, n }: { value: number | null | undefined; n: number | undefined }) {
  return (
    <>
      <span className="font-medium text-gray-900">{num(value)}</span>{" "}
      <span className="text-xs text-gray-500">n={count(n)}</span>
    </>
  );
}

// --------------------------------------------------------------------------- results

function isRate(value: unknown): value is Rate {
  return !!value && typeof value === "object" && "rate" in value && "n" in value;
}

function isDecisionResult(value: unknown): value is DecisionResult {
  return !!value && typeof value === "object" && "teacher_accuracy" in value;
}

const RATE_DOMAIN: Domain = [0, 1];
const MARK = 150;

function RateEstimate({ r }: { r: Rate | null | undefined }) {
  return (
    <span className="whitespace-nowrap tabular-nums">
      <span className="font-medium text-gray-900">{num(r?.rate)}</span>{" "}
      <span className="text-xs text-gray-500">{interval(r?.ci95?.[0], r?.ci95?.[1])}</span>
    </span>
  );
}

function RateTable({
  rows,
  first,
  units = true,
}: {
  rows: [string, Rate | null][];
  first: string;
  units?: boolean;
}) {
  return (
    <Table variant="list">
      <TableHeader>
        <TableRow>
          <TableHead>{first}</TableHead>
          {units ? <TableHead className="text-right">Units</TableHead> : null}
          <TableHead className="text-right">Items</TableHead>
          <TableHead>Rate [95% CI]</TableHead>
          <TableHead />
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(([name, r]) => (
          <TableRow key={name}>
            <TableCell className="min-w-72">{name}</TableCell>
            {units ? (
              <TableCell className="text-right tabular-nums">{count(r?.n ?? 0)}</TableCell>
            ) : null}
            <TableCell className="text-right tabular-nums">{count(r?.items ?? 0)}</TableCell>
            <TableCell>
              <RateEstimate r={r} />
            </TableCell>
            <TableCell>
              <IntervalMark
                value={r?.rate}
                low={r?.ci95?.[0]}
                high={r?.ci95?.[1]}
                domain={RATE_DOMAIN}
                width={MARK}
                label={name}
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
      <TableFooter className="border-t-0 bg-transparent font-normal">
        <TableRow>
          <TableCell colSpan={units ? 4 : 3} />
          <TableCell className="pt-0">
            <IntervalAxis domain={RATE_DOMAIN} width={MARK} />
          </TableCell>
        </TableRow>
      </TableFooter>
    </Table>
  );
}

function RateResults({ results }: { results: Record<string, unknown> }) {
  const rows: [string, Rate | null][] = [];
  let sections: [string, Rate | null][] = [];
  for (const [name, value] of Object.entries(results)) {
    if (isRate(value) || value === null) rows.push([name, value]);
    else if (value && typeof value === "object") {
      sections = Object.entries(value as Record<string, unknown>).map(([s, r]) => [
        sectionLabel(s),
        isRate(r) ? r : null,
      ]);
    }
  }
  return (
    <div className="flex flex-col gap-1">
      <Subheading>Results against the reference</Subheading>
      <RateTable rows={rows.map(([n, r]) => [capitalize(n), r])} first="Measure" />
      {sections.length ? (
        <>
          <Subheading>Major omissions by section</Subheading>
          <RateTable
            rows={sections.map(([n, r]) => [capitalize(n), r])}
            first="Section"
            units={false}
          />
        </>
      ) : null}
    </div>
  );
}

const SECTION_LABELS: Record<string, string> = {
  cns_metastases: "CNS metastases",
  lvef_qtc: "LVEF and QTc",
};

/** A summary section (rubric_summary.py SECTIONS) for display. */
function sectionLabel(section: string): string {
  return SECTION_LABELS[section] ?? capitalize(words(section));
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function ScoredCell({ s }: { s: Scored | undefined }) {
  return (
    <TableCell className="whitespace-nowrap tabular-nums">
      <div className="font-medium text-gray-900">{num(s?.value)}</div>
      <div className="text-xs text-gray-500">{interval(s?.ci95?.[0], s?.ci95?.[1])}</div>
    </TableCell>
  );
}

function DecisionResults({ results }: { results: Record<string, unknown> }) {
  const rows = Object.entries(results).filter((e): e is [string, DecisionResult] =>
    isDecisionResult(e[1]),
  );
  return (
    <div className="flex flex-col gap-1">
      <Subheading>Results against the reference</Subheading>
      <p className="text-xs text-gray-600">
        The teacher's accuracy against the reference, and the decision model's P(yes) scored against
        it.
      </p>
      <Table variant="list">
        <TableHeader>
          <TableRow>
            <TableHead>Question</TableHead>
            <TableHead className="text-right">n</TableHead>
            <TableHead className="text-right">Reference yes</TableHead>
            <TableHead>Teacher accuracy</TableHead>
            <TableHead />
            <TableHead>Model AUROC</TableHead>
            <TableHead>AUPRC (yes)</TableHead>
            <TableHead>AUPRC (no)</TableHead>
            <TableHead>Brier</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([family, r]) => (
            <TableRow key={family}>
              <TableCell className="font-medium">{kindLabel(family)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(r.n)}</TableCell>
              <TableCell className="text-right tabular-nums">{count(r.reference_yes)}</TableCell>
              <TableCell className="whitespace-nowrap tabular-nums">
                <div className="font-medium text-gray-900">{num(r.teacher_accuracy.rate)}</div>
                <div className="text-xs text-gray-500">
                  {interval(r.teacher_accuracy.ci95?.[0], r.teacher_accuracy.ci95?.[1])}
                </div>
              </TableCell>
              <TableCell>
                <IntervalMark
                  value={r.teacher_accuracy.rate}
                  low={r.teacher_accuracy.ci95?.[0]}
                  high={r.teacher_accuracy.ci95?.[1]}
                  domain={RATE_DOMAIN}
                  width={MARK}
                  label={`Teacher accuracy, ${kindLabel(family)}`}
                />
              </TableCell>
              <ScoredCell s={r.model_auroc} />
              <ScoredCell s={r.model_auprc_yes} />
              <ScoredCell s={r.model_auprc_no} />
              <ScoredCell s={r.model_brier} />
            </TableRow>
          ))}
        </TableBody>
        <TableFooter className="border-t-0 bg-transparent font-normal">
          <TableRow>
            <TableCell colSpan={4} />
            <TableCell className="pt-0">
              <IntervalAxis domain={RATE_DOMAIN} width={MARK} />
            </TableCell>
            <TableCell colSpan={4} />
          </TableRow>
        </TableFooter>
      </Table>
    </div>
  );
}
