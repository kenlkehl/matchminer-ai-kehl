import { useQuery } from "@tanstack/react-query";
import { BookOpenIcon, DownloadIcon, InfoIcon, SearchIcon } from "lucide-react";
import { useId, useState } from "react";

import { Accordion } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { saveBlob } from "@/lib/api";

import { findGuidelineSpaces, say } from "../actions";
import { Field, NativeSelect, StatusLine } from "../fields";
import { JobProgress } from "../job-progress";
import { hasSummary, isBusy, setPatient, usePatient } from "../store";
import type { RetrieveResult } from "../types";
import { catalogsOptions, LibraryDialog } from "./library-dialog";
import { buildExportReport } from "./report";
import { SpaceCard } from "./space-card";

const TOP_N = [1, 3, 5, 10, 20];

/**
 * The Guidelines view: find the guideline populations (spaces) that fit the patient's summary, then
 * review each space's workup and treatment considerations and check the documented workup in the
 * patient's notes.
 */
export function GuidelinesView() {
  const s = usePatient();
  const retrieving = s.job?.kind === "retrieve" ? s.job : null;
  return (
    <div className="flex flex-col gap-4">
      <SearchScope />
      <StatusLine where="guidelines" />
      {retrieving ? <JobProgress job={retrieving} /> : null}
      {s.retrieval ? <Results result={s.retrieval} /> : retrieving ? null : <EmptyState />}
    </div>
  );
}

function SearchScope() {
  const s = usePatient();
  const id = useId();
  const library = useQuery(catalogsOptions());
  const [libraryOpen, setLibraryOpen] = useState(false);
  const busy = isBusy(s);
  const catalogs = library.data?.catalogs ?? [];
  const ready = hasSummary(s.summary) && catalogs.length > 0 && !busy;
  return (
    <section
      aria-label="Search scope"
      className="flex flex-col gap-3 rounded-lg border border-gray-200 bg-white p-4"
    >
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Guideline collection" htmlFor={`${id}-catalog`} className="min-w-64 flex-1">
          <div className="flex gap-2">
            <NativeSelect
              id={`${id}-catalog`}
              className="min-w-0 flex-1"
              value={s.catalog}
              disabled={busy || !library.data}
              onChange={(e) => {
                setPatient({ catalog: e.target.value });
              }}
            >
              {library.data ? (
                <>
                  <option value="">All completed guidelines ({catalogs.length})</option>
                  {catalogs.map((row) => (
                    <option key={row.id} value={row.id}>
                      {row.title} · {row.spaces} spaces
                    </option>
                  ))}
                </>
              ) : (
                <option value="">
                  {library.isError ? "Catalogs unavailable" : "Loading completed catalogs…"}
                </option>
              )}
            </NativeSelect>
            <Button
              variant="outline"
              onClick={() => {
                setLibraryOpen(true);
              }}
            >
              <BookOpenIcon />
              Library
              {library.data ? <Badge variant="secondary">{catalogs.length}</Badge> : null}
            </Button>
          </div>
        </Field>
        <Field label="Spaces to return" htmlFor={`${id}-top`}>
          <NativeSelect
            id={`${id}-top`}
            value={s.topN}
            disabled={busy}
            onChange={(e) => {
              const topN = Number(e.target.value);
              setPatient((p) => ({ topN, candidateK: Math.max(p.candidateK, topN) }));
            }}
          >
            {TOP_N.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </NativeSelect>
        </Field>
        <Field label="Candidate spaces to rerank" htmlFor={`${id}-k`}>
          <Input
            id={`${id}-k`}
            type="number"
            min={1}
            max={100}
            className="h-9 w-24"
            value={Number.isFinite(s.candidateK) ? s.candidateK : ""}
            disabled={busy}
            onChange={(e) => {
              setPatient({ candidateK: e.target.valueAsNumber });
            }}
          />
        </Field>
        <Button
          className="ml-auto"
          disabled={!ready}
          onClick={() => {
            findGuidelineSpaces();
          }}
        >
          <SearchIcon />
          Find guideline spaces →
        </Button>
      </div>
      <p className="text-xs text-gray-500">
        {hasSummary(s.summary)
          ? "Matched to the population and ranked by match quality: TrialSpace retrieves candidate spaces and the decision model's match questions rank their clinical fit. The candidate count must cover the spaces returned. Search uses local TrialSpace and decision models."
          : "Add a patient summary above (choose a patient, enter one, or create one from clinical notes) to find guideline spaces."}
      </p>
      {library.isError ? <p className="text-sm text-red-700">{library.error.message}</p> : null}
      {libraryOpen ? (
        <LibraryDialog
          disabled={busy}
          onSelect={(catalog) => {
            setPatient({ catalog });
            setLibraryOpen(false);
            say("guidelines", "info", "Guideline selected for the next search.");
          }}
          onClose={() => {
            setLibraryOpen(false);
          }}
        />
      ) : null}
    </section>
  );
}

function Results({ result }: { result: RetrieveResult }) {
  const s = usePatient();
  const rows = result.spaces;
  const stale = result.summary.trim() !== s.summary.trim();
  const searched = result.provenance.catalog?.catalog_spaces;
  return (
    <section aria-labelledby="guideline-results" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="guideline-results" className="text-xl font-bold">
          Considerations for review
        </h2>
        <Badge variant="secondary">
          {rows.length} ranked {rows.length === 1 ? "space" : "spaces"}
        </Badge>
        <span className="text-sm text-gray-600">
          {searched !== undefined ? searched.toLocaleString() : "Selected"} catalog spaces searched
        </span>
        <Button
          size="sm"
          variant="outline"
          className="ml-auto"
          disabled={!rows.length}
          onClick={() => {
            saveBlob(
              new Blob([buildExportReport(result, s.reviews)], {
                type: "text/markdown;charset=utf-8",
              }),
              "guideline-considerations.md",
            );
          }}
        >
          <DownloadIcon />
          Export report
        </Button>
      </div>
      {stale ? (
        <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950">
          The patient summary has changed since this search. Find guideline spaces again to search
          with the current summary.
        </p>
      ) : null}
      <p className="flex items-start gap-2 text-sm text-gray-600">
        <InfoIcon className="mt-0.5 size-4 shrink-0" />
        Ranking reflects population fit. Individual options may require additional tests,
        biomarkers, or prior treatments. Review each condition and its source.
      </p>
      {rows.length ? (
        <Accordion
          multiple
          className="gap-3"
          value={s.openSpaces}
          onValueChange={(value) => {
            setPatient({ openSpaces: value.filter((v): v is number => typeof v === "number") });
          }}
        >
          {rows.map((space, i) => (
            <SpaceCard key={`${space.space_trial_id}-${String(i)}`} space={space} index={i} />
          ))}
        </Accordion>
      ) : (
        <p className="py-6 text-center text-sm text-gray-500">
          No spaces were returned. Try a broader guideline scope.
        </p>
      )}
      <details className="rounded-md border border-gray-200 bg-white p-3 text-sm">
        <summary className="cursor-pointer font-medium">Retrieval &amp; model provenance</summary>
        <pre className="mt-2 max-h-96 overflow-auto text-xs whitespace-pre-wrap text-gray-700">
          {JSON.stringify(result.provenance, null, 2)}
        </pre>
      </details>
    </section>
  );
}

function EmptyState() {
  const steps: [string, string][] = [
    ["Describe", "The patient's summary and notes"],
    ["Match", "Guideline spaces"],
    ["Review", "Source-backed options and documented workup"],
  ];
  return (
    <section className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-gray-300 px-6 py-10 text-center">
      <h2 className="text-base font-semibold">No guideline results yet</h2>
      <p className="max-w-xl text-sm text-gray-600">
        Find the guideline populations that fit the patient, then explore their diagnostic and
        treatment considerations and check the documented workup in the notes.
      </p>
      <ol className="flex flex-wrap justify-center gap-4">
        {steps.map(([title, what], i) => (
          <li key={title} className="flex items-center gap-2 text-left text-sm">
            <span className="flex size-6 items-center justify-center rounded-full bg-brand-blue/10 text-xs font-semibold text-brand-blue">
              {i + 1}
            </span>
            <span className="flex flex-col">
              <strong>{title}</strong>
              <small className="text-gray-500">{what}</small>
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
