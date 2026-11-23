import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getRouteApi } from "@tanstack/react-router";
import { DownloadIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

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
import { api, download, isActive, type Job, waitForJob } from "@/lib/api";

import {
  type Domain,
  domainOf,
  IntervalAxis,
  IntervalMark,
  LegendItem,
  ReliabilityPlot,
  STATE_COLORS,
  ThinSegmentedBar,
} from "./charts";
import { BackToRuns, Section, StateBar } from "./common";
import { count, duration, interval, num, signed, stamp, words } from "./format";
import { fidelityOptions } from "./queries";
import type {
  ExportResult,
  FamilyMetrics,
  FidelityJob,
  FidelityMetrics,
  FidelityRun,
  MeanInterval,
  PipelineRow,
  StageReceipt,
  ValueInterval,
} from "./types";

const route = getRouteApi("/_app/benchmarking/fidelity/$runId");

const RANKERS: [string, string][] = [
  ["embeddinggemma2", "(a) EmbeddingGemma 2 (base)"],
  ["trialspace_round2", "(b) TrialSpace round 2"],
  ["trialspace_round2_decision", "(c) TrialSpace round 2 top K + decision reasonableness"],
];
const SHORT: Record<string, string> = {
  embeddinggemma2: "(a)",
  trialspace_round2: "(b)",
  trialspace_round2_decision: "(c)",
};
const WORKFLOWS: [string, string][] = [
  ["patient_centric", "Trials for a patient (patient-centric)"],
  ["trial_centric", "Patients for a trial (trial-centric)"],
];
const DIFFERENCES: [string, string][] = [
  ["trialspace_round2", "embeddinggemma2"],
  ["trialspace_round2_decision", "trialspace_round2"],
  ["trialspace_round2_decision", "embeddinggemma2"],
];
const FAMILY_LABELS: Record<string, string> = {
  reasonableness: "Reasonableness",
  cancer_type: "Cancer type",
  burden: "Burden",
  treatment_history: "Treatment history",
  biomarker: "Biomarker",
  boilerplate: "Boilerplate",
  "criterion (all lines)": "Criteria: all",
  "criterion (inclusion lines)": "Criteria: inclusion",
  "criterion (exclusion lines)": "Criteria: exclusion",
};
const BANDS = ["1-20", "21-50", "51-80", "81-100"];
/** Rank bands of the retriever's list: an ordinal blue ramp, nearest ranks darkest (validated --ordinal). */
const BAND_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"];

/** A distillation-fidelity run: its stages, then retrieval and the decision model against the teacher. */
export function FidelityRunPage() {
  const { runId } = route.useParams();
  const query = useQuery(fidelityOptions(runId));
  const run = query.data;
  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-3">
        <BackToRuns />
        <PageHeader title={`Distillation fidelity: ${run?.name ?? ""}`}>
          <p>
            How closely the distilled models (TrialSpace retrieval, its rerank by the decision
            model, and the decision model's questions) reproduce the teacher model's labels on real
            patients. The reference is the teacher's labels, not a clinician's review.
          </p>
          {run ? (
            <p>
              {run.updated ? `Status updated ${stamp(run.updated)}.` : null}
              {run.released ? " Teacher labeling has finished." : null}
            </p>
          ) : null}
        </PageHeader>
        {run?.metrics ? <ExportResults run={run} /> : null}
      </div>
      {query.isPending ? (
        <Spinner className="size-6" />
      ) : query.isError ? (
        <p className="text-sm text-red-700">{query.error.message}</p>
      ) : run ? (
        <>
          <Stages run={run} />
          {run.metrics ? (
            <Results metrics={run.metrics} kBinding={run.k_binding} />
          ) : (
            <p className="text-sm text-gray-600">
              The results appear here when the metrics stage has run.
            </p>
          )}
          <Notes run={run} />
        </>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------- export

const enc = encodeURIComponent;

/**
 * Export results: the server draws the page's results as a PDF (in real mode with Table 1 and the metrics by
 * demographic group, small cells suppressed), then the browser downloads it.
 */
function ExportResults({ run }: { run: FidelityRun }) {
  const client = useQueryClient();
  const [job, setJob] = useState<Job<ExportResult> | null>(null);
  const polling = useRef<string | null>(null);
  const latest = job ?? run.exports.at(-1) ?? null;
  const save = () =>
    download(`/api/benchmarks/fidelity/${enc(run.id)}/report.pdf`, "fidelity-report.pdf");

  const follow = async (id: string) => {
    if (polling.current === id) return;
    polling.current = id;
    try {
      await waitForJob<ExportResult>(id, setJob, 1000);
    } finally {
      polling.current = null;
      void client.invalidateQueries({ queryKey: ["benchmarks", "fidelity", run.id] });
    }
  };
  const start = useMutation({
    mutationFn: async () => {
      const started = await api.post<{ job: Job<ExportResult> }>(
        `/api/benchmarks/fidelity/${enc(run.id)}/report`,
        {},
      );
      setJob(started.job);
      await follow(started.job.id);
      await save();
    },
  });
  const again = useMutation({ mutationFn: save });
  // An export started earlier (or in another tab) is still running: follow it.
  const serverActive = run.exports.findLast((j) => isActive(j));
  useEffect(() => {
    if (serverActive && !job) void follow(serverActive.id).catch(() => undefined);
  }, [serverActive?.id]);

  const active = isActive(latest) || start.isPending;
  const error = start.error ?? again.error;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button
        disabled={active}
        onClick={() => {
          start.mutate();
        }}
      >
        {active ? <Spinner /> : <DownloadIcon />}
        Export results
      </Button>
      <span className="text-sm text-gray-600">
        {latest && isActive(latest) ? (
          <>
            {latest.message || "waiting for the worker"}
            {latest.total
              ? ` (step ${String(Math.min(latest.done + 1, latest.total))} of ${String(latest.total)})`
              : ""}
          </>
        ) : error ? (
          <span className="text-red-700">{error.message}</span>
        ) : run.report ? (
          <>
            PDF built {stamp(run.report.built)}.{" "}
            <button
              type="button"
              className="link"
              disabled={again.isPending}
              onClick={() => {
                again.mutate();
              }}
            >
              Download it again
            </button>
            {latest?.status === "done" && latest.result?.note ? (
              <span className="block text-xs text-gray-500">{latest.result.note}</span>
            ) : null}
          </>
        ) : (
          "A PDF of the results below, for sharing."
        )}
      </span>
    </div>
  );
}

// --------------------------------------------------------------------------- stages

/** The stage receipt a job writes (benchmarking.py FIDELITY_STAGES); the embedding jobs have none here. */
function receiptKey(job: string): string {
  if (job === "retrieve") return "retrieval";
  const m = /^(decide|label)_(.+)$/.exec(job);
  if (m?.[1] && m[2]) return `${m[1] === "label" ? "labels" : "decide"}/${m[2]}`;
  return job;
}

function stageLabel(job: string): string {
  const fixed: Record<string, string> = {
    extraction: "Cohort extraction",
    patients: "Patients",
    cohort_table: "Cohort table",
    corpus: "Corpus and query samples",
    retrieve: "Retrieval",
    lists_ab: "Lists (a) and (b)",
    lists_c: "Lists (c): decision rerank",
    metrics: "Metrics",
  };
  if (fixed[job]) return fixed[job];
  let m = /^embed_(.+)_(patients|cohorts)$/.exec(job);
  if (m?.[1] && m[2]) {
    const embedder = m[1] === "embeddinggemma2" ? "EmbeddingGemma 2" : words(m[1]);
    return `Embed ${m[2]}: ${embedder.replace("trialspace round2", "TrialSpace round 2")}`;
  }
  m = /^decide_(.+)$/.exec(job);
  if (m?.[1]) return `Decision model: ${words(m[1])}`;
  m = /^label_(.+)_(ab|c)$/.exec(job);
  if (m?.[1] && m[2])
    return `Teacher labels: ${words(m[1])}, lists ${m[2] === "ab" ? "(a)(b)" : "(c)"}`;
  return words(job);
}

/** A receipt's counts, nested objects flattened one level ("strata.x: 3"). */
function flatten(content: Record<string, unknown> | null): [string, string][] {
  const out: [string, string][] = [];
  const show = (v: unknown): string | null =>
    typeof v === "number"
      ? v.toLocaleString()
      : typeof v === "string" || typeof v === "boolean"
        ? String(v)
        : null;
  for (const [key, value] of Object.entries(content ?? {})) {
    const text = show(value);
    if (text !== null) out.push([words(key), text]);
    else if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        const t = show(v);
        if (t !== null) out.push([`${words(key)} · ${words(k)}`, t]);
      }
    }
  }
  return out;
}

function took(job: FidelityJob, now: number): string {
  if (!job.started) return "";
  if (job.state === "running") return `${duration(now - job.started)} so far`;
  return job.finished ? duration(job.finished - job.started) : "";
}

function Stages({ run }: { run: FidelityRun }) {
  const jobs = Object.entries(run.jobs);
  const counts: Record<string, number> = {};
  for (const [, job] of jobs) counts[job.state] = (counts[job.state] ?? 0) + 1;
  const allDone = jobs.length > 0 && jobs.every(([, j]) => j.state === "done");
  const [open, setOpen] = useState<boolean | null>(null);
  const shown = open ?? !allDone;
  const now = Date.now() / 1000;
  return (
    <Section
      title="Stages"
      action={
        jobs.length ? (
          <Button
            variant="outline"
            size="xs"
            onClick={() => {
              setOpen(!shown);
            }}
          >
            {shown ? "Hide stages" : "Show stages"}
          </Button>
        ) : null
      }
    >
      {jobs.length ? (
        <div className="max-w-2xl">
          <StateBar counts={counts} />
        </div>
      ) : (
        <p className="text-sm text-gray-600">The orchestrator has not written its status yet.</p>
      )}
      {shown && jobs.length ? (
        <Table variant="list">
          <TableHeader>
            <TableRow>
              <TableHead>Stage</TableHead>
              <TableHead>State</TableHead>
              <TableHead>Tries</TableHead>
              <TableHead>Started</TableHead>
              <TableHead>Took</TableHead>
              <TableHead>Receipt</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map(([name, job]) => (
              <TableRow key={name}>
                <TableCell>
                  <p className="font-medium">{stageLabel(name)}</p>
                  <p className="font-mono text-xs text-gray-500">{name}</p>
                </TableCell>
                <TableCell>
                  <StateLabel state={job.state} exit={job.exit} />
                </TableCell>
                <TableCell className="tabular-nums">{job.tries || ""}</TableCell>
                <TableCell className="whitespace-nowrap text-gray-600">
                  {stamp(job.started)}
                </TableCell>
                <TableCell className="whitespace-nowrap text-gray-600">{took(job, now)}</TableCell>
                <TableCell>
                  <Receipt receipt={run.stages[receiptKey(name)]} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : null}
    </Section>
  );
}

function StateLabel({ state, exit }: { state: string; exit: number | null }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        aria-hidden
        className="size-2.5 rounded-full"
        style={{ background: STATE_COLORS[state] ?? "#d1d5db" }}
      />
      {state}
      {state === "failed" && exit !== null ? (
        <span className="text-xs text-gray-500">exit {exit}</span>
      ) : null}
    </span>
  );
}

function Receipt({ receipt }: { receipt: StageReceipt | undefined }) {
  const [all, setAll] = useState(false);
  if (!receipt) return null;
  const rows = flatten(receipt.content);
  const shown = all ? rows : rows.slice(0, 4);
  return (
    <div className="flex flex-col gap-0.5 text-xs text-gray-600">
      <div className="flex flex-wrap gap-x-3">
        {shown.map(([k, v]) => (
          <span key={k} className="whitespace-nowrap">
            {k} <span className="font-medium text-gray-900 tabular-nums">{v}</span>
          </span>
        ))}
        {rows.length > 4 ? (
          <button
            type="button"
            className="link"
            onClick={() => {
              setAll(!all);
            }}
          >
            {all ? "fewer" : `+${String(rows.length - 4)} more`}
          </button>
        ) : null}
      </div>
      <span className="text-gray-500">
        {receipt.finished_at ? `finished ${stamp(receipt.finished_at)}` : ""}
        {receipt.commit ? ` · commit ${receipt.commit.slice(0, 7)}` : ""}
      </span>
    </div>
  );
}

// --------------------------------------------------------------------------- results

function Results({
  metrics,
  kBinding,
}: {
  metrics: FidelityMetrics;
  kBinding: FidelityRun["k_binding"];
}) {
  return (
    <>
      {metrics.retrieval ? (
        <Retrieval retrieval={metrics.retrieval} full={metrics.full_pipeline} kBinding={kBinding} />
      ) : null}
      {metrics.decision ? <Decision decision={metrics.decision} /> : null}
    </>
  );
}

function Retrieval({
  retrieval,
  full,
  kBinding,
}: {
  retrieval: NonNullable<FidelityMetrics["retrieval"]>;
  full: FidelityMetrics["full_pipeline"];
  kBinding: FidelityRun["k_binding"];
}) {
  const top = retrieval.top;
  const rows = Object.values(retrieval.table);
  const scoreDomain = domainOf(
    rows.flatMap((r) => [r.mean_score.low, r.mean_score.high]),
    [0],
  );
  const diffs = Object.entries(retrieval.paired_differences);
  const diffDomain = (metric: string) =>
    domainOf(
      diffs.filter(([k]) => k.includes(`/${metric}/`)).flatMap(([, d]) => [d.low, d.high]),
      [0],
    );
  const domains = { precision: diffDomain("precision"), mean_score: diffDomain("mean_score") };
  return (
    <Section
      title={`Retrieval against the teacher: P@${String(top)}`}
      description={
        <>
          P@{top}: the share of each query's top {top} that the teacher judged a reasonable match,
          over the list's labeled pairs (unlabeled pairs are counted beside it). Mean score: the
          teacher's 0–6 match score of the list. 95% intervals bootstrap the queries; paired
          differences resample the same queries.
          {full
            ? " These ranker comparisons cover the seeded comparison queries; the full pipeline's table covers every query."
            : ""}
        </>
      }
    >
      <div className="flex flex-col gap-8">
        {WORKFLOWS.map(([workflow, title]) => (
          <div key={workflow} className="flex flex-col gap-4">
            <h3 className="text-base font-semibold">{title}</h3>
            <RetrievalTable retrieval={retrieval} workflow={workflow} scoreDomain={scoreDomain} />
            <Differences retrieval={retrieval} workflow={workflow} domains={domains} />
            {full?.table[`${workflow}/${full.ranker}`] ? (
              <FullPipeline
                row={full.table[`${workflow}/${full.ranker}`]}
                cluster={full.clusters[workflow] ?? "cluster"}
                top={top}
                scoreDomain={scoreDomain}
              />
            ) : null}
            {kBinding?.[workflow] ? <KBinding values={kBinding[workflow]} top={top} /> : null}
          </div>
        ))}
      </div>
    </Section>
  );
}

function Estimate({
  value,
  low,
  high,
  digits = 3,
  sign = false,
}: {
  value: number | null | undefined;
  low?: number | null;
  high?: number | null;
  digits?: number;
  sign?: boolean;
}) {
  const fmt = sign ? signed : num;
  return (
    <span className="whitespace-nowrap tabular-nums">
      <span className="font-medium text-gray-900">{fmt(value, digits)}</span>{" "}
      <span className="text-xs text-gray-500">
        {interval(low, high, digits, sign ? (v, d) => signed(v, d) : undefined)}
      </span>
    </span>
  );
}

const MARK_WIDTH = 150;

function RetrievalTable({
  retrieval,
  workflow,
  scoreDomain,
}: {
  retrieval: NonNullable<FidelityMetrics["retrieval"]>;
  workflow: string;
  scoreDomain: Domain;
}) {
  const present = RANKERS.filter(([r]) => retrieval.table[`${workflow}/${r}`]);
  if (!present.length) return <p className="text-sm text-gray-500">No queries.</p>;
  const pDomain: Domain = [0, 1];
  return (
    <Table variant="list">
      <TableHeader>
        <TableRow>
          <TableHead>Ranker</TableHead>
          <TableHead className="text-right">Queries</TableHead>
          <TableHead>P@{retrieval.top} [95% CI]</TableHead>
          <TableHead />
          <TableHead>Mean 0–6 score [95% CI]</TableHead>
          <TableHead />
          <TableHead className="text-right">Lists &lt; {retrieval.top}</TableHead>
          <TableHead className="text-right">Unlabeled pairs</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {present.map(([ranker, label]) => {
          const row = retrieval.table[`${workflow}/${ranker}`];
          if (!row) return null;
          const p = row.precision_at_top;
          const s = row.mean_score;
          return (
            <TableRow key={ranker}>
              <TableCell className="min-w-56">{label}</TableCell>
              <TableCell className="text-right tabular-nums">{count(row.queries)}</TableCell>
              <TableCell>
                <Estimate value={p.mean} low={p.low} high={p.high} />
              </TableCell>
              <TableCell>
                <IntervalMark
                  value={p.mean}
                  low={p.low}
                  high={p.high}
                  domain={pDomain}
                  width={MARK_WIDTH}
                  label={`P@${String(retrieval.top)}, ${label}`}
                />
              </TableCell>
              <TableCell>
                <Estimate value={s.mean} low={s.low} high={s.high} digits={2} />
              </TableCell>
              <TableCell>
                <IntervalMark
                  value={s.mean}
                  low={s.low}
                  high={s.high}
                  domain={scoreDomain}
                  width={MARK_WIDTH}
                  digits={2}
                  label={`Mean score, ${label}`}
                />
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {count(row.queries_with_fewer_than_top)}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {count(row.unlabeled_pairs)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
      <TableFooter className="border-t-0 bg-transparent font-normal">
        <TableRow>
          <TableCell colSpan={3} />
          <TableCell className="pt-0">
            <IntervalAxis domain={pDomain} width={MARK_WIDTH} />
          </TableCell>
          <TableCell />
          <TableCell className="pt-0">
            <IntervalAxis domain={scoreDomain} width={MARK_WIDTH} />
          </TableCell>
          <TableCell colSpan={2} />
        </TableRow>
      </TableFooter>
    </Table>
  );
}

function Differences({
  retrieval,
  workflow,
  domains,
}: {
  retrieval: NonNullable<FidelityMetrics["retrieval"]>;
  workflow: string;
  domains: { precision: Domain; mean_score: Domain };
}) {
  const get = (metric: string, first: string, second: string): MeanInterval | undefined =>
    retrieval.paired_differences[`${workflow}/${metric}/${first}-minus-${second}`];
  const rows = DIFFERENCES.filter(
    ([a, b]) => get("precision", a, b)?.mean != null || get("mean_score", a, b)?.mean != null,
  );
  if (!rows.length) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-sm font-semibold text-gray-700">Paired differences</h4>
      <Table variant="list">
        <TableHeader>
          <TableRow>
            <TableHead>Difference</TableHead>
            <TableHead>P@{retrieval.top} [95% CI]</TableHead>
            <TableHead />
            <TableHead>Mean score [95% CI]</TableHead>
            <TableHead />
            <TableHead className="text-right">Queries</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([a, b]) => {
            const p = get("precision", a, b);
            const s = get("mean_score", a, b);
            const label = `${SHORT[a] ?? a} − ${SHORT[b] ?? b}`;
            return (
              <TableRow key={`${a}-${b}`}>
                <TableCell className="min-w-56">{label}</TableCell>
                <TableCell>
                  <Estimate value={p?.mean} low={p?.low} high={p?.high} sign />
                </TableCell>
                <TableCell>
                  <IntervalMark
                    value={p?.mean}
                    low={p?.low}
                    high={p?.high}
                    domain={domains.precision}
                    reference={0}
                    width={MARK_WIDTH}
                    label={`P@${String(retrieval.top)} difference ${label}`}
                  />
                </TableCell>
                <TableCell>
                  <Estimate value={s?.mean} low={s?.low} high={s?.high} digits={2} sign />
                </TableCell>
                <TableCell>
                  <IntervalMark
                    value={s?.mean}
                    low={s?.low}
                    high={s?.high}
                    domain={domains.mean_score}
                    reference={0}
                    width={MARK_WIDTH}
                    digits={2}
                    label={`Mean score difference ${label}`}
                  />
                </TableCell>
                <TableCell className="text-right tabular-nums">{count(p?.n ?? s?.n)}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
        <TableFooter className="border-t-0 bg-transparent font-normal">
          <TableRow>
            <TableCell colSpan={2} />
            <TableCell className="pt-0">
              <IntervalAxis domain={domains.precision} width={MARK_WIDTH} />
            </TableCell>
            <TableCell />
            <TableCell className="pt-0">
              <IntervalAxis domain={domains.mean_score} width={MARK_WIDTH} />
            </TableCell>
            <TableCell />
          </TableRow>
        </TableFooter>
      </Table>
    </div>
  );
}

/** Ranker (c) over every query of a full-denominator run, with intervals from bootstrapping clusters. */
function FullPipeline({
  row,
  cluster,
  top,
  scoreDomain,
}: {
  row: PipelineRow | undefined;
  cluster: string;
  top: number;
  scoreDomain: Domain;
}) {
  if (!row || row.precision_at_top.mean === null) return null;
  const p = row.precision_at_top;
  const s = row.mean_score;
  const pDomain: Domain = [0, 1];
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-sm font-semibold text-gray-700">
        Full pipeline over every query: P@{top} of (c)
      </h4>
      <p className="text-xs text-gray-600">
        Every query of this workflow; 95% intervals bootstrap {cluster}s (all of a {cluster}'s
        queries together).
      </p>
      <Table variant="list">
        <TableHeader>
          <TableRow>
            <TableHead className="text-right">Queries</TableHead>
            <TableHead className="text-right">{capitalizeFirst(cluster)}s</TableHead>
            <TableHead>P@{top} [95% CI]</TableHead>
            <TableHead />
            <TableHead>Mean 0–6 score [95% CI]</TableHead>
            <TableHead />
            <TableHead className="text-right">Lists &lt; {top}</TableHead>
            <TableHead className="text-right">Unlabeled pairs</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className="text-right tabular-nums">{count(row.queries)}</TableCell>
            <TableCell className="text-right tabular-nums">{count(row.clusters)}</TableCell>
            <TableCell>
              <Estimate value={p.mean} low={p.low} high={p.high} />
            </TableCell>
            <TableCell>
              <IntervalMark
                value={p.mean}
                low={p.low}
                high={p.high}
                domain={pDomain}
                width={MARK_WIDTH}
                label={`P@${String(top)}, full pipeline`}
              />
            </TableCell>
            <TableCell>
              <Estimate value={s.mean} low={s.low} high={s.high} digits={2} />
            </TableCell>
            <TableCell>
              <IntervalMark
                value={s.mean}
                low={s.low}
                high={s.high}
                domain={scoreDomain}
                width={MARK_WIDTH}
                digits={2}
                label="Mean score, full pipeline"
              />
            </TableCell>
            <TableCell className="text-right tabular-nums">
              {count(row.queries_with_fewer_than_top)}
            </TableCell>
            <TableCell className="text-right tabular-nums">{count(row.unlabeled_pairs)}</TableCell>
          </TableRow>
        </TableBody>
        <TableFooter className="border-t-0 bg-transparent font-normal">
          <TableRow>
            <TableCell colSpan={3} />
            <TableCell className="pt-0">
              <IntervalAxis domain={pDomain} width={MARK_WIDTH} />
            </TableCell>
            <TableCell />
            <TableCell className="pt-0">
              <IntervalAxis domain={scoreDomain} width={MARK_WIDTH} />
            </TableCell>
            <TableCell colSpan={2} />
          </TableRow>
        </TableFooter>
      </Table>
    </div>
  );
}

/** Where (c)'s picks sit in the retriever's top K (lists.py k_binding). */
function KBinding({ values, top }: { values: Record<string, number>; top: number }) {
  const bands = Object.keys(values).filter((k) => /^\d+-\d+$/.test(k));
  const ordered = [
    ...BANDS.filter((b) => bands.includes(b)),
    ...bands.filter((b) => !BANDS.includes(b)),
  ];
  const beyond = values.queries_with_a_pick_beyond_80;
  return (
    <div className="flex max-w-2xl flex-col gap-1.5">
      <h4 className="text-sm font-semibold text-gray-700">
        Where (c)'s top {top} sit in TrialSpace's ranking
      </h4>
      <ThinSegmentedBar
        label={`Share of (c)'s top ${String(top)} by TrialSpace rank`}
        parts={ordered.map((band, i) => ({
          key: band,
          label: `ranks ${band}`,
          value: values[band] ?? 0,
          color: BAND_COLORS[Math.min(i, BAND_COLORS.length - 1)] ?? "#86b6ef",
        }))}
      />
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {ordered.map((band, i) => (
          <LegendItem key={band} color={BAND_COLORS[Math.min(i, BAND_COLORS.length - 1)] ?? ""}>
            ranks {band.replace("-", "–")}{" "}
            <span className="font-medium text-gray-900 tabular-nums">
              {num((values[band] ?? 0) * 100, 1)}%
            </span>
          </LegendItem>
        ))}
        {beyond !== undefined ? (
          <span className="text-xs text-gray-600">
            {count(beyond)} queries with a pick beyond rank 80
          </span>
        ) : null}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- decision model

const SCOPES: [string, string][] = [
  ["pooled", "Both workflows pooled"],
  ["patient_centric", "Trials for a patient"],
  ["trial_centric", "Patients for a trial"],
];

function Decision({ decision }: { decision: NonNullable<FidelityMetrics["decision"]> }) {
  const scopes = SCOPES.filter(([s]) => decision[s]);
  const [scope, setScope] = useState(scopes[0]?.[0] ?? "pooled");
  const tables = decision[scope] ?? {};
  const families = Object.entries(tables);
  const draws = families.find(([, m]) => m.bootstrap_draws)?.[1].bootstrap_draws;
  const aurocDomain = domainOf(
    families.flatMap(([, m]) => [m.auroc?.low, m.auroc?.high]),
    [1],
  );
  return (
    <Section
      title="Decision model against the teacher"
      description={
        <>
          On the merged top {20} pairs of the three rankers: one row per unique pair (boilerplate:
          per unique combination of the two boilerplate texts; criteria: a stratified sample). Yes
          means potentially eligible or not clearly failing; for the cancer type, burden, treatment
          history and biomarker questions, yes means the cohort specifically requires it and the
          patient is documented to meet it. AUPRC (no) scores 1 − P(yes) against the no class; its
          chance level is the no prevalence. 95% intervals
          {draws ? ` from ${draws.toLocaleString()} draws` : ""} resample{" "}
          {scope === "pooled" ? "patients" : "queries"}.
        </>
      }
    >
      <Tabs
        value={scope}
        onValueChange={(value) => {
          setScope(String(value));
        }}
      >
        <TabsList>
          {scopes.map(([s, label]) => (
            <TabsTrigger key={s} value={s} className="px-3">
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <Table variant="list" className="[&_td]:px-2 [&_th]:px-2">
        <TableHeader>
          <TableRow>
            <TableHead>Question</TableHead>
            <TableHead className="text-right whitespace-normal">n (clusters)</TableHead>
            <TableHead>AUROC</TableHead>
            <TableHead />
            <TableHead>AUPRC (yes)</TableHead>
            <TableHead>AUPRC (no)</TableHead>
            <TableHead>Brier</TableHead>
            <TableHead>ECE</TableHead>
            <TableHead className="whitespace-normal">Cal. intercept</TableHead>
            <TableHead>Cal. slope</TableHead>
            <TableHead className="w-24 text-right whitespace-normal">
              Mean P(yes) / observed
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {families.map(([family, m]) => (
            <TableRow key={family}>
              <TableCell className="font-medium whitespace-nowrap">
                {FAMILY_LABELS[family] ?? family}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                <div>{count(m.n)}</div>
                {m.n ? <div className="text-xs text-gray-500">({count(m.clusters)})</div> : null}
              </TableCell>
              {m.n ? (
                <>
                  <MetricCell m={m.auroc} />
                  <TableCell>
                    <IntervalMark
                      value={m.auroc?.value}
                      low={m.auroc?.low}
                      high={m.auroc?.high}
                      domain={aurocDomain}
                      width={96}
                      label={`AUROC, ${FAMILY_LABELS[family] ?? family}`}
                    />
                  </TableCell>
                  <MetricCell m={m.auprc_yes} base={m.prevalence_yes} />
                  <MetricCell m={m.auprc_no} base={m.prevalence_no} />
                  <MetricCell m={m.brier} />
                  <MetricCell m={m.ece} />
                  <MetricCell m={m.calibration_intercept} sign />
                  <MetricCell m={m.calibration_slope} />
                  <TableCell className="text-right tabular-nums">
                    <div>{num(m.mean_predicted)}</div>
                    <div className="text-xs text-gray-500">{num(m.observed_rate)}</div>
                  </TableCell>
                </>
              ) : (
                <TableCell colSpan={9} className="text-gray-500">
                  no rows
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
        <TableFooter className="border-t-0 bg-transparent font-normal">
          <TableRow>
            <TableCell colSpan={3} />
            <TableCell className="pt-0">
              <IntervalAxis domain={aurocDomain} width={96} />
            </TableCell>
            <TableCell colSpan={7} />
          </TableRow>
        </TableFooter>
      </Table>
      <Reliability tables={tables} />
    </Section>
  );
}

function MetricCell({
  m,
  base,
  sign = false,
}: {
  m: ValueInterval | undefined;
  base?: number | null;
  sign?: boolean;
}) {
  const fmt = sign ? signed : num;
  return (
    <TableCell className="whitespace-nowrap tabular-nums">
      <div className="font-medium text-gray-900">{fmt(m?.value)}</div>
      <div className="text-xs text-gray-500">
        {interval(m?.low, m?.high, 3, sign ? (v, d) => signed(v, d) : undefined)}
      </div>
      {base !== undefined && base !== null ? (
        <div className="text-xs text-gray-500">base {num(base)}</div>
      ) : null}
    </TableCell>
  );
}

function Reliability({ tables }: { tables: Record<string, FamilyMetrics> }) {
  const [asTables, setAsTables] = useState(false);
  const families = Object.entries(tables).filter(([, m]) => m.n && m.reliability?.length);
  if (!families.length) return null;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <h3 className="text-base font-semibold">Reliability</h3>
        <Button
          variant="outline"
          size="xs"
          onClick={() => {
            setAsTables(!asTables);
          }}
        >
          {asTables ? "Show plots" : "Show as tables"}
        </Button>
      </div>
      <p className="text-sm text-gray-600">
        Observed yes rate against the mean P(yes) in equal-mass bins of P(yes); the diagonal is
        perfect calibration.
      </p>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
        {families.map(([family, m]) => {
          const title = FAMILY_LABELS[family] ?? family;
          return (
            <figure key={family} className="flex flex-col gap-1 rounded-lg border p-3">
              <figcaption>
                <p className="text-sm font-semibold">{title}</p>
                <p className="text-xs text-gray-500 tabular-nums">
                  ECE {num(m.ece?.value)} · n {count(m.n)}
                </p>
              </figcaption>
              {asTables ? (
                <BinTable bins={m.reliability ?? []} />
              ) : (
                <ReliabilityPlot bins={m.reliability ?? []} title={title} />
              )}
            </figure>
          );
        })}
      </div>
    </div>
  );
}

function BinTable({ bins }: { bins: NonNullable<FamilyMetrics["reliability"]> }) {
  return (
    <table className="w-full text-xs tabular-nums">
      <thead className="text-gray-500">
        <tr>
          <th className="py-1 text-left font-medium">Bin</th>
          <th className="py-1 text-right font-medium">n</th>
          <th className="py-1 text-right font-medium">P(yes) range</th>
          <th className="py-1 text-right font-medium">Mean</th>
          <th className="py-1 text-right font-medium">Observed</th>
        </tr>
      </thead>
      <tbody>
        {bins.map((b, i) => (
          <tr key={i} className="even:bg-gray-50">
            <td className="py-0.5">{i + 1}</td>
            <td className="py-0.5 text-right">{b.n.toLocaleString()}</td>
            <td className="py-0.5 text-right whitespace-nowrap">
              {num(b.p_min)}–{num(b.p_max)}
            </td>
            <td className="py-0.5 text-right">{num(b.mean_predicted)}</td>
            <td className="py-0.5 text-right">{num(b.observed)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// --------------------------------------------------------------------------- counts and design

function Notes({ run }: { run: FidelityRun }) {
  const counts = Object.entries(run.metrics?.counts ?? {});
  const notes = run.design_notes.map((n) => (typeof n === "string" ? n : JSON.stringify(n)));
  if (!counts.length && !notes.length) return null;
  return (
    <Section title="Counts and design">
      {counts.length ? (
        <dl className="grid max-w-3xl grid-cols-[1fr_auto] gap-x-6 gap-y-1 text-sm">
          {counts.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-gray-600">{capitalizeFirst(words(k))}</dt>
              <dd className="text-right tabular-nums">{count(v)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {notes.length ? (
        <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
          {notes.map((n, i) => (
            <li key={i} className="whitespace-pre-wrap">
              {n}
            </li>
          ))}
        </ul>
      ) : null}
    </Section>
  );
}

function capitalizeFirst(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
