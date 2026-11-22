import { DownloadIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { download } from "@/lib/api";

import { Helper } from "./controls";
import { APPLICABILITY_LABELS, count, isExtractionMethod, methodLabels, score } from "./labels";
import type { AggregateRow, MatchRow, PopulationReport, PopulationView } from "./types";
import type { PopulationController } from "./use-population";

const LINK = "text-left font-medium text-brand-blue hover:underline";

function Panel({ children }: { children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3 rounded-xl bg-card p-4 shadow-xs ring-1 ring-foreground/10">
      {children}
    </section>
  );
}

/** The four headline counts. */
export function Metrics({ population: p }: { population: PopulationView | null }) {
  const a = p?.active;
  const tiles: [string, number][] = [
    ["Source population", p?.source_count ?? 0],
    ["Active patients", a?.patient_count ?? 0],
    ["Patients reviewed", a?.report?.patient_count ?? 0],
    ["Workup items", a?.space?.diagnostic_workup?.length ?? 0],
  ];
  return (
    <dl className="grid flex-1 grid-cols-2 gap-3 xl:grid-cols-4">
      {tiles.map(([label, value]) => (
        <div
          key={label}
          className="flex flex-col gap-1 rounded-xl bg-card px-4 py-3 shadow-xs ring-1 ring-foreground/10"
        >
          <dt className="text-xs font-medium text-gray-600">{label}</dt>
          <dd className="text-2xl leading-none font-semibold text-brand-blue">
            {value.toLocaleString()}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function activeStatus(p: PopulationView | null): string {
  const a = p?.active;
  if (!p || !a) {
    return "Load a population, then select a finalized trial space to find matching patients.";
  }
  const r = a.retrieval;
  const threshold =
    r.min_match_score !== undefined
      ? `Score threshold: ${r.min_match_score === null ? "none" : String(r.min_match_score)}`
      : `Reasonableness threshold: ${r.min_reasonableness == null ? "none" : String(r.min_reasonableness)}`;
  return `${count(a.patient_count)} active patients / ${count(r.scored_count ?? a.matches.length)} match-quality scored / ${count(r.population_count ?? p.source_count)} in source population. ${threshold}. New selection settings apply when you fetch again.`;
}

/** The active cohort: its guideline space, its matched patients and the cohort export. */
export function ActiveCohort({
  pop,
  onPatient,
}: {
  pop: PopulationController;
  onPatient: (patientId: string) => void;
}) {
  const p = pop.population;
  const a = p?.active;
  return (
    <Panel>
      <div className="flex items-start justify-between gap-3">
        <h2 className="min-w-0 flex-1 text-lg font-semibold">
          {a?.space?.name || "Choose a guideline population"}
        </h2>
        <Button
          variant="outline"
          size="sm"
          className="shrink-0"
          disabled={pop.busy || !a?.patient_count}
          onClick={() => {
            if (!p) return;
            // The browser streams a large archive straight to its download folder.
            const link = document.createElement("a");
            link.href = `/api/population/export?version=${encodeURIComponent(p.version)}`;
            link.download = "guideline-cohort.zip";
            document.body.append(link);
            link.click();
            link.remove();
            pop.setNotice(
              "Preparing the cohort archive on the server. The browser download begins when it is ready.",
            );
          }}
        >
          <DownloadIcon />
          Export cohort
        </Button>
      </div>
      {a?.space ? (
        <p className="max-h-48 overflow-y-auto text-sm break-words whitespace-pre-wrap text-gray-800">
          {a.space.cohort_key_criteria || a.space.clinical_space_summary}
        </p>
      ) : null}
      <Helper>{activeStatus(p)}</Helper>
      {a?.matches.length ? (
        <details className="rounded-md border border-gray-200">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-gray-700 select-none">
            Matched patients & scores · {a.matches.length}
          </summary>
          <MatchesTable matches={a.matches} onPatient={onPatient} />
        </details>
      ) : null}
      <Helper>
        Export cohort saves the active patients, source notes, summaries, guideline sources and any
        completed results in one ZIP. Downloads contain the patient record.
      </Helper>
    </Panel>
  );
}

function MatchesTable({
  matches,
  onPatient,
}: {
  matches: MatchRow[];
  onPatient: (patientId: string) => void;
}) {
  return (
    <div className="max-h-[28rem] overflow-y-auto border-t border-gray-200">
      <Table variant="list">
        <TableHeader>
          <TableRow>
            <TableHead>Patient</TableHead>
            <TableHead className="text-right">Reasonableness</TableHead>
            <TableHead className="text-right">Match quality</TableHead>
            <TableHead className="text-right">TrialSpace similarity</TableHead>
            <TableHead className="text-right">Retrieval rank</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {matches.map((m) => (
            <TableRow key={m.patient_id}>
              <TableCell>
                <button
                  type="button"
                  className={LINK}
                  onClick={() => {
                    onPatient(m.patient_id);
                  }}
                >
                  {m.patient_id}
                </button>
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {score(m.match_quality_reasonableness)}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {score(m.match_quality_score)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{score(m.similarity_score)}</TableCell>
              <TableCell className="text-right tabular-nums">{m.retrieval_rank}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

/** How the report was made, in one line. */
function reportStatus(r: PopulationReport | null | undefined): string {
  if (!r) return "Check every active patient against every workup guideline.";
  if (isExtractionMethod(r.method)) {
    const x = r.extraction ?? {};
    const c = r.compile ?? {};
    const store = r.extraction_store;
    const records = `records ${store?.name ?? ""} (${(store?.levels ?? []).join(" + ")} level) · ${count(x.documents_used)} document records used${x.documents_ignored ? `, ${count(x.documents_ignored)} for other notes ignored` : ""}${x.notes_without_records ? `, ${count(x.notes_without_records)} notes without records` : ""}`;
    const model = c.model || r.connection?.model || "";
    if (r.method === "extraction_sql") {
      return `${count(r.patient_count)} patients checked with the SQL agent · ${records} · SQL queries ${count(c.compiled)}/${count(c.distinct)} accepted (${count(c.cached)} cached, ${count(c.requests)} agent requests over ${count(c.exploration?.patients)} store patients) by ${model}; each ran once over the cohort. Review aids, not determinations.`;
    }
    return `${count(r.patient_count)} patients checked with extraction queries · ${records} · queries ${count(c.compiled)}/${count(c.distinct)} compiled (${count(c.cached)} cached, ${count(c.requests)} requests) by ${model}. Review aids, not determinations.`;
  }
  return `${count(r.patient_count)} patients reviewed · full notes · ${String(r.max_concurrent_patients ?? 1)} at once${r.connection?.model ? ` · ${r.connection.model}` : ""}.`;
}

/** Workup documentation by guideline item, with the results downloads. */
export function Distribution({
  pop,
  onItem,
}: {
  pop: PopulationController;
  onItem: (number: number) => void;
}) {
  const p = pop.population;
  const r = p?.active?.report;
  const fetchResults = (format: "json" | "csv") => {
    if (!p) return;
    void pop.request(() =>
      download(
        `/api/population/results?version=${encodeURIComponent(p.version)}&format=${format}`,
        `population-results.${format}`,
      ),
    );
  };
  return (
    <Panel>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[0.7rem] font-semibold tracking-wider text-brand-blue uppercase">
            Workup documentation
          </p>
          <h2 className="text-lg font-semibold">Distribution by guideline</h2>
        </div>
        <div className="flex gap-2">
          {(["json", "csv"] as const).map((format) => (
            <Button
              key={format}
              variant="outline"
              size="sm"
              disabled={pop.busy || !r}
              onClick={() => {
                fetchResults(format);
              }}
            >
              <DownloadIcon />
              Results {format.toUpperCase()}
            </Button>
          ))}
        </div>
      </div>
      <Helper>{reportStatus(r)}</Helper>
      {r?.aggregate.length ? <DistributionTable report={r} onItem={onItem} /> : null}
      <Helper>
        Select a workup guideline to inspect individual patients.{" "}
        {r && isExtractionMethod(r.method)
          ? "Met / applicable assessed counts met or not documented among patients the item applies to; unknown and not-applicable results stay out."
          : "Completed / applicable assessed excludes uncertain applicability, not applicable findings and review errors."}{" "}
        Counts always include every active patient. Missing documentation means not located in the
        reviewed notes or records, not that care was omitted; applicability, timing and completeness
        need human review, and this does not establish overall guideline concordance.
      </Helper>
    </Panel>
  );
}

function DistributionTable({
  report,
  onItem,
}: {
  report: PopulationReport;
  onItem: (number: number) => void;
}) {
  const names = methodLabels(report.method);
  const statuses = Object.keys(names);
  const extraction = isExtractionMethod(report.method);
  return (
    <Table variant="list" className="text-xs">
      <TableCaption className="mt-2 text-left text-xs">
        {extraction
          ? `${report.method === "extraction_sql" ? "SQL-agent" : "Extraction-query"} results across the active cohort (met / met or not documented, among patients it applies to)`
          : "Documentation statuses across the active cohort"}
      </TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead rowSpan={2} className={`${FIRST} bg-gray-50 align-bottom ${GROUP}`}>
            Workup guideline
          </TableHead>
          <TableHead rowSpan={2} className={`min-w-40 align-bottom ${GROUP}`}>
            {extraction ? "Met / applicable assessed" : "Completed / applicable assessed"}
          </TableHead>
          <TableHead colSpan={statuses.length} className={`border-l border-gray-200 ${GROUP}`}>
            {extraction ? "Result" : "Documentation status"}
          </TableHead>
          <TableHead colSpan={3} className={`border-l border-gray-200 ${GROUP}`}>
            Applicability
          </TableHead>
        </TableRow>
        <TableRow>
          {statuses.map((s, i) => (
            <TableHead key={s} className={`${HEAD} ${i === 0 ? "border-l border-gray-200" : ""}`}>
              {names[s]}
            </TableHead>
          ))}
          {(["applies", "not_applicable", "uncertain"] as const).map((s, i) => (
            <TableHead key={s} className={`${HEAD} ${i === 0 ? "border-l border-gray-200" : ""}`}>
              {APPLICABILITY_LABELS[s]}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {report.aggregate.map((row) => (
          <TableRow key={row.recommendation_index}>
            <TableHead
              scope="row"
              className={`${FIRST} bg-inherit font-normal whitespace-normal normal-case`}
            >
              <button
                type="button"
                className={`${LINK} text-xs`}
                onClick={() => {
                  onItem(row.recommendation_index);
                }}
              >
                {row.recommendation.name}
              </button>
            </TableHead>
            <TableCell className="px-2">
              <Completion row={row} />
            </TableCell>
            {statuses.map((s, i) => (
              <Count key={s} value={row.statuses[s] ?? 0} first={i === 0} />
            ))}
            {(["applies", "not_applicable", "uncertain"] as const).map((s, i) => (
              <Count key={s} value={row.applicability[s]} first={i === 0} />
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

/** The guideline column stays in view while the counts scroll sideways. */
const FIRST = "sticky left-0 z-10 min-w-44 max-w-60 px-2";
const GROUP = "px-2 text-[0.7rem] whitespace-normal normal-case text-gray-600";
const HEAD =
  "min-w-[4.75rem] px-2 align-bottom text-right text-[0.7rem] leading-tight whitespace-normal normal-case";

function Count({ value, first }: { value: number; first: boolean }) {
  return (
    <TableCell
      className={`px-2 text-right tabular-nums ${value ? "text-gray-900" : "text-gray-400"} ${first ? "border-l border-gray-200" : ""}`}
    >
      {value.toLocaleString()}
    </TableCell>
  );
}

/** Documented completion among applicable assessed patients: a meter and its figures. */
function Completion({ row }: { row: AggregateRow }) {
  if (row.recommends_against) {
    return (
      <span className="text-gray-700">
        Advised against · {row.statuses.done_not_recommended ?? 0} done anyway
      </span>
    );
  }
  const total = row.applicable_assessed;
  if (!total) return <span className="text-gray-500">— · no applicable assessments</span>;
  const done = row.completed_applicable;
  const share = done / total;
  const label = `${String(done)} of ${String(total)} applicable assessed patients documented (${String(Math.round(share * 100))}%)`;
  return (
    <div className="flex items-center gap-2" title={label}>
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
        aria-label={label}
        className="h-2 w-16 shrink-0 rounded-full bg-brand-blue/15"
      >
        <div
          className="h-full rounded-full bg-brand-blue"
          style={{ width: `${String(share * 100)}%` }}
        />
      </div>
      <span className="whitespace-nowrap text-gray-900 tabular-nums">
        {done} / {total}
      </span>
    </div>
  );
}
