import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { metaOptions } from "@/lib/queries";

import { StateBar } from "./common";
import { componentLabel, judgeName, stamp } from "./format";
import { runsOptions } from "./queries";
import type { RunRow } from "./types";

/** Every evaluation run under the mode's benchmark roots, newest first. */
export function BenchmarksPage() {
  const runs = useQuery(runsOptions());
  const real = useQuery(metaOptions()).data?.mode === "real";
  const data = runs.data;
  const fidelity = data?.runs.filter((r) => r.kind === "fidelity") ?? [];
  const judge = data?.runs.filter((r) => r.kind === "judge") ?? [];
  return (
    <div className="flex flex-col gap-8">
      <PageHeader title="Benchmarking">
        <p>
          Evaluation runs of the matching pipeline. Runs are started from the command line and
          appear here as they write their results.
        </p>
      </PageHeader>
      {runs.isPending ? (
        <Spinner className="size-6" />
      ) : runs.isError ? (
        <p className="text-sm text-red-700">{runs.error.message}</p>
      ) : data && !data.runs.length ? (
        <p className="text-sm text-gray-600">
          {data.roots
            ? "No evaluation runs were found under this mode's benchmark folders."
            : "This mode has no benchmark folders (the site config's dashboard benchmark_roots)."}
        </p>
      ) : (
        <>
          <section className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold">Distillation fidelity</h2>
            <p className="text-sm text-gray-600">
              How closely TrialSpace retrieval and the decision model reproduce the teacher model's
              labels on {real ? "real" : "synthetic"} patients, in both matching directions.
            </p>
            {fidelity.length ? (
              <FidelityRuns runs={fidelity} />
            ) : (
              <p className="text-sm text-gray-500">No fidelity runs.</p>
            )}
          </section>
          <section className="flex flex-col gap-2">
            <h2 className="text-lg font-semibold">LLM as judge</h2>
            <p className="text-sm text-gray-600">
              Two LLM judges rate the pipeline's cohort extraction, patient summaries and decision
              answers; their disagreements are adjudicated here.
            </p>
            {judge.length ? (
              <JudgeRuns runs={judge} />
            ) : (
              <p className="text-sm text-gray-500">No judge runs.</p>
            )}
          </section>
        </>
      )}
    </div>
  );
}

function FidelityRuns({ runs }: { runs: RunRow[] }) {
  return (
    <Table variant="list">
      <TableHeader>
        <TableRow>
          <TableHead>Run</TableHead>
          <TableHead className="w-[40%]">Stages</TableHead>
          <TableHead>Metrics</TableHead>
          <TableHead>Updated</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => {
          const jobs = run.jobs ?? {};
          const total = Object.values(jobs).reduce((a, b) => a + b, 0);
          return (
            <TableRow key={run.id}>
              <TableCell>
                <Link
                  to="/benchmarking/fidelity/$runId"
                  params={{ runId: run.id }}
                  className="link font-medium"
                >
                  {run.name}
                </Link>
                <p className="text-xs text-gray-500">{run.folder}</p>
              </TableCell>
              <TableCell>
                {total ? (
                  <StateBar counts={jobs} />
                ) : (
                  <span className="text-gray-500">not started</span>
                )}
              </TableCell>
              <TableCell>
                {run.has_metrics ? (
                  <Badge variant="secondary">ready</Badge>
                ) : (
                  <span className="text-gray-500">not yet</span>
                )}
              </TableCell>
              <TableCell className="whitespace-nowrap text-gray-600">
                {stamp(run.updated)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

function JudgeRuns({ runs }: { runs: RunRow[] }) {
  return (
    <Table variant="list">
      <TableHeader>
        <TableRow>
          <TableHead>Run</TableHead>
          <TableHead>Items</TableHead>
          <TableHead>Judges</TableHead>
          <TableHead>Progress</TableHead>
          <TableHead>Updated</TableHead>
          <TableHead />
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.id}>
            <TableCell>
              <Link
                to="/benchmarking/judge/$runId"
                params={{ runId: run.id }}
                className="link font-medium"
              >
                {run.name}
              </Link>
              <p className="text-xs text-gray-500">{run.folder}</p>
            </TableCell>
            <TableCell>
              {Object.entries(run.components ?? {}).map(([c, n]) => (
                <p key={c} className="whitespace-nowrap">
                  {componentLabel(c)}: {n ?? "–"}
                </p>
              ))}
            </TableCell>
            <TableCell>
              {(run.judges ?? []).map((j) => (
                <p key={j} className="whitespace-nowrap">
                  {judgeName(j)}
                </p>
              ))}
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                <Badge variant={run.compared ? "secondary" : "outline"}>
                  {run.compared ? "compared" : "not compared"}
                </Badge>
                <Badge variant={run.has_metrics ? "secondary" : "outline"}>
                  {run.has_metrics ? "metrics" : "no metrics"}
                </Badge>
                <Badge variant="outline">{run.adjudications ?? 0} adjudicated</Badge>
              </div>
            </TableCell>
            <TableCell className="whitespace-nowrap text-gray-600">{stamp(run.updated)}</TableCell>
            <TableCell>
              {run.compared ? (
                <Link
                  to="/benchmarking/judge/$runId/adjudicate"
                  params={{ runId: run.id }}
                  className="link text-sm whitespace-nowrap"
                >
                  Adjudicate
                </Link>
              ) : null}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
